module DerivedFunctors

using LinearAlgebra
using SparseArrays

using ..CoreModules: QQ
import ..CoreModules: _append_scaled_triplets!
using ..ExactQQ: rankQQ, nullspaceQQ, colspaceQQ, solve_fullcolumnQQ,
                 SparseRow, SparseRREF, SparseRREFAugmented, normalize_sparse_row!,
                 _sparse_rref_push_homogeneous!, _sparse_rref_push_augmented!,
                 _nullspace_from_pivots
import ..FiniteFringe
using ..FiniteFringe: FinitePoset, FringeModule, cover_edges, Upset, Downset

# We use qualified calls like IndicatorResolutions.upset_resolution(...) below.
# In Julia, "using ..IndicatorResolutions: name1, name2, ..." does NOT bind the module name
# "IndicatorResolutions" itself, so we must also import the module.
import ..IndicatorResolutions
using ..IndicatorResolutions: PModule, PMorphism, pmodule_from_fringe, map_leq,
                              projective_cover, kernel_with_inclusion,
                              _injective_hull, _cokernel_module, ShortExactSequence

using ..ChainComplexes
using ..ChainComplexes: CochainComplex, cohomology_data, cohomology_coordinates, 
       induced_map_on_cohomology, induced_map_on_homology
using ..FlangeZn: Flange
using ..ZnEncoding
using ..PLPolyhedra
using ..HomExt

# ----------------------------
# Basic utilities: morphism composition (local, explicit, reliable)
# ----------------------------

function compose(g::PMorphism{QQ}, f::PMorphism{QQ})
    @assert f.cod === g.dom
    Q = f.dom.Q
    comps = Vector{Matrix{QQ}}(undef, Q.n)
    for i in 1:Q.n
        comps[i] = g.comps[i] * f.comps[i]
    end
    return PMorphism{QQ}(f.dom, g.cod, comps)
end

# ----------------------------
# Hom space with explicit basis
# ----------------------------

struct HomSpace{K}
    dom::PModule{K}
    cod::PModule{K}
    basis::Vector{PMorphism{K}}
    basis_matrix::Matrix{K}  # columns are vectorizations of basis morphisms
    offsets::Vector{Int}     # per vertex block offsets in the vectorization
end

function _hom_offsets(M::PModule{QQ}, N::PModule{QQ})
    Q = M.Q
    @assert N.Q === Q
    offs = zeros(Int, Q.n + 1)
    for i in 1:Q.n
        offs[i+1] = offs[i] + N.dims[i] * M.dims[i]
    end
    return offs
end

function _morphism_to_vector(f::PMorphism{QQ}, offs::Vector{Int})
    Q = f.dom.Q
    v = zeros(QQ, offs[end], 1)
    for i in 1:Q.n
        di = f.cod.dims[i]
        ei = f.dom.dims[i]
        if di == 0 || ei == 0
            continue
        end
        block = vec(f.comps[i]) # column-major
        s = offs[i] + 1
        t = offs[i+1]
        v[s:t, 1] = block
    end
    return v
end

function _vector_to_morphism(dom::PModule{QQ}, cod::PModule{QQ}, offs::Vector{Int}, x::Vector{QQ})
    Q = dom.Q
    comps = Vector{Matrix{QQ}}(undef, Q.n)
    for i in 1:Q.n
        di = cod.dims[i]
        ei = dom.dims[i]
        if di == 0 || ei == 0
            comps[i] = zeros(QQ, di, ei)
            continue
        end
        s = offs[i] + 1
        t = offs[i+1]
        comps[i] = reshape(x[s:t], di, ei)
    end
    return PMorphism{QQ}(dom, cod, comps)
end

"""
    Hom(M::PModule{QQ}, N::PModule{QQ}) -> HomSpace{QQ}

Compute Hom_Q(M,N) together with an explicit basis of module morphisms.

Performance notes
- This function used to assemble a dense constraint matrix A=zeros(QQ,neqs,nvars)
  and then call nullspaceQQ(A).  That is prohibitively expensive when nvars is
  large, even though each constraint row is very sparse.
- We now stream each naturality equation into an exact sparse RREF reducer
  (dictionary-of-rows) without ever materializing A.

Mathematical content
- Unknowns are the entries of the vertex maps F_u : M_u -> N_u for all u.
- Constraints are the naturality equations along cover edges (u <. v):
    N_{uv} * F_u - F_v * M_{uv} = 0.
"""
function Hom(M::PModule{QQ}, N::PModule{QQ})
    Q = M.Q
    @assert N.Q === Q

    offs = _hom_offsets(M, N)
    nvars = offs[end]

    # Degenerate case: all vertex dimensions are zero.
    if nvars == 0
        return HomSpace{QQ}(M, N, PMorphism{QQ}[], zeros(QQ, 0, 0), offs)
    end

    dM = M.dims
    dN = N.dims

    # RREF basis of the row space of the constraint system, streamed row-by-row.
    R = SparseRREF{QQ}(nvars)
    row = SparseRow{QQ}()
    fullrank = false

    storeM = M.edge_maps
    storeN = N.edge_maps

    # Performance: iterate cover edges using the store-aligned adjacency
    # (succs + maps_to_succ) to avoid keyed lookups like store[(u,v)] in hot loops.
    succs = storeM.succs
    mapsM = storeM.maps_to_succ
    mapsN = storeN.maps_to_succ

    @inbounds for u in 1:Q.n
        su = succs[u]
        Mu = mapsM[u]
        Nu = mapsN[u]

        du = dM[u]
        dNu = dN[u]
        (du == 0 || dNu == 0) && continue

        for j in eachindex(su)
            v = su[j]

            dv = dM[v]
            dNv = dN[v]
            (dv == 0 || dNv == 0) && continue

            Nuv = Nu[j]
            Muv = Mu[j]

            for ii in 1:dNv, jj in 1:du
                empty!(row.idx)
                empty!(row.val)

                # Nuv * F_u
                for k in 1:dNu
                    c = Nuv[ii, k]
                    if !iszero(c)
                        col = offs[u] + k + (jj - 1) * dNu
                        push!(row.idx, col)
                        push!(row.val, c)
                    end
                end

                # -F_v * Muv
                for l in 1:dv
                    c = Muv[l, jj]
                    if !iszero(c)
                        col = offs[v] + ii + (l - 1) * dNv
                        push!(row.idx, col)
                        push!(row.val, -c)
                    end
                end

                normalize_sparse_row!(row.idx, row.val)
                _sparse_rref_push_homogeneous!(R, row)

                if length(R.pivot_cols) == nvars
                    fullrank = true
                    break
                end
            end

            fullrank && break
        end
        fullrank && break
    end

    basis_matrix = fullrank ? zeros(QQ, nvars, 0) : _nullspace_from_pivots(R, nvars)

    basis = PMorphism{QQ}[]
    for j in 1:size(basis_matrix, 2)
        push!(basis, _vector_to_morphism(M, N, offs, basis_matrix[:, j]))
    end

    return HomSpace{QQ}(M, N, basis, basis_matrix, offs)
end


dimension(H::HomSpace) = length(H.basis)
basis(H::HomSpace) = H.basis
dim(H::HomSpace) = length(H.basis)

# ----------------------------
# Projective resolution (explicit summands + coefficient matrices)
# ----------------------------

struct ProjectiveResolution{K}
    M::PModule{K}
    Pmods::Vector{PModule{K}}                    # P_0 .. P_L
    gens::Vector{Vector{Int}}                    # base vertex per summand (same order as summands)
    d_mor::Vector{PMorphism{K}}                  # d_a : P_a -> P_{a-1}, a=1..L
    d_mat::Vector{SparseMatrixCSC{K, Int}}       # coefficient matrices (rows cod summands, cols dom summands)
    aug::PMorphism{K}                            # P_0 -> M
end

function _flatten_gens_at(gens_at)
    out = Int[]
    for u in 1:length(gens_at)
        for tup in gens_at[u]
            push!(out, tup[1])
        end
    end
    return out
end

function _active_upset_indices(P::FinitePoset, base_vertices::Vector{Int})
    active = [Int[] for _ in 1:P.n]
    @inbounds for i in 1:length(base_vertices)
        p = base_vertices[i]
        for u in 1:P.n
            if P.leq[p, u]
                push!(active[u], i)
            end
        end
    end
    return active
end



"""
    _active_downset_indices(P::FinitePoset, base_vertices::Vector{Int}) -> Vector{Vector{Int}}

For a direct sum of principal downsets

    oplus_i k[Dn(base_vertices[i])],

return `active[u]` = the list of summand indices that are active at vertex `u`.

Convention:
- A principal downset Dn(v) contains u iff u <= v.
- Summand i is active at u iff P.leq[u, base_vertices[i]].

The returned lists are in increasing summand index order. This matches the fiber
basis ordering used in `_injective_hull` and makes coefficient extraction stable.
"""
function _active_downset_indices(P::FinitePoset, base_vertices::Vector{Int})
    active = [Int[] for _ in 1:P.n]
    for i in 1:length(base_vertices)
        v = base_vertices[i]
        for u in 1:P.n
            if P.leq[u, v]
                push!(active[u], i)
            end
        end
    end
    return active
end


"""
    _coeff_matrix_downsets(P, dom_bases, cod_bases, f) -> SparseMatrixCSC{QQ,Int}

Extract the scalar coefficient matrix of a morphism between direct sums of
principal downsets.

Inputs:
- `dom_bases`: base vertices of the domain summands (each summand is k[Dn(v)]).
- `cod_bases`: base vertices of the codomain summands (each summand is k[Dn(w)]).
- `f`: a `PMorphism{QQ}` whose domain/codomain are those direct sums.

Output:
- A sparse matrix C of size (length(cod_bases) x length(dom_bases)) such that
  C[row, col] is the scalar multiplying the unique (up to scalar) map

      k[Dn(dom_bases[col])] -> k[Dn(cod_bases[row])].

Implementation detail:
- For downsets, the distinguished generator of k[Dn(w)] lives at vertex w.
  To read the scalar for a map into k[Dn(w)], evaluate at u = w (the codomain
  base vertex), where the codomain generator is visible.
"""
function _coeff_matrix_downsets(P::FinitePoset,
                                dom_bases::Vector{Int},
                                cod_bases::Vector{Int},
                                f::PMorphism{QQ})
    n_dom = length(dom_bases)
    n_cod = length(cod_bases)

    active_dom = _active_downset_indices(P, dom_bases)
    active_cod = _active_downset_indices(P, cod_bases)

    C = spzeros(QQ, n_cod, n_dom)

    for row in 1:n_cod
        u = cod_bases[row]   # read at the codomain base vertex
        act_dom_u = active_dom[u]
        act_cod_u = active_cod[u]

        # Locate this codomain summand in the fiber basis at u.
        pos_row = findfirst(x -> x == row, act_cod_u)
        if pos_row === nothing
            error("_coeff_matrix_downsets: could not locate cod summand in fiber basis")
        end

        Fu = f.comps[u]
        # Read the row restricted to active domain summands at u.
        for pos_col in 1:length(act_dom_u)
            col = act_dom_u[pos_col]
            val = Fu[pos_row, pos_col]
            if !iszero(val)
                C[row, col] = val
            end
        end
    end

    return C
end

"""
    _coeff_matrix_upsets(P, dom_bases, cod_bases, f) -> SparseMatrixCSC{QQ,Int}

Extract the coefficient matrix of a morphism `f` between direct sums of principal
upsets, where those direct sums are indexed by lists of base vertices.

Interpretation:

- `dom_bases[i]` is the base vertex of the i-th domain summand (a principal upset).
- `cod_bases[j]` is the base vertex of the j-th codomain summand.

The coefficient C[j,i] is read at the *domain base vertex* u = dom_bases[i].
At that vertex, the domain summand i is guaranteed active, and any codomain
summand j that can receive a nonzero map is also active there.

This is the "upset-side" analog of `_coeff_matrix_downsets`, but it is
column-driven (read at domain base vertices) rather than row-driven.

Performance notes:
- Uses precomputed active-index lists at each vertex.
- Uses `searchsortedfirst` instead of a closure-based `findfirst` to avoid
  allocations and to exploit sorted active lists.
- Assembles as sparse triplets, then calls `sparse(...)` once.
"""
function _coeff_matrix_upsets(P::FinitePoset,
                              dom_bases::Vector{Int},
                              cod_bases::Vector{Int},
                              f::PMorphism{QQ})
    n_dom = length(dom_bases)
    n_cod = length(cod_bases)

    # Active summand indices at each vertex u:
    # active_dom[u] = [i : dom_bases[i] <= u]
    # active_cod[u] = [j : cod_bases[j] <= u]
    #
    # Ordering is by summand index (increasing), which matches how the indicator
    # projectives are assembled in IndicatorResolutions.
    active_dom = _active_upset_indices(P, dom_bases)
    active_cod = _active_upset_indices(P, cod_bases)

    I = Int[]
    J = Int[]
    V = QQ[]

    # A mild sizehint: in many resolutions, differentials are relatively sparse.
    sizehint!(I, n_dom)
    sizehint!(J, n_dom)
    sizehint!(V, n_dom)

    @inbounds for col in 1:n_dom
        u = dom_bases[col]

        act_dom_u = active_dom[u]
        act_cod_u = active_cod[u]

        # Column position of the domain summand "col" inside the fiber basis at u.
        # act_dom_u is sorted (increasing indices) by construction.
        pos_col = searchsortedfirst(act_dom_u, col)
        if pos_col > length(act_dom_u) || act_dom_u[pos_col] != col
            error("_coeff_matrix_upsets: internal basis mismatch at u=$u for dom summand col=$col")
        end

        Fu = f.comps[u]  # (#active cod at u) x (#active dom at u)

        # Read the image of the domain generator at u as a column of Fu.
        for pos_row in 1:length(act_cod_u)
            row = act_cod_u[pos_row]
            val = Fu[pos_row, pos_col]
            if !iszero(val)
                push!(I, row)
                push!(J, col)
                push!(V, val)
            end
        end
    end

    return sparse(I, J, V, n_cod, n_dom)
end


# Extract coefficient matrix for a morphism between sums of principal upsets.
# Domain and codomain are direct sums of principal upsets indexed by base vertex lists.
function _coeff_matrix_upsets(dom_bases::Vector{Upset}, cod_bases::Vector{Upset})
    n_dom = length(dom_bases)
    n_cod = length(cod_bases)

    I = Int[]
    J = Int[]
    V = QQ[]

    @inbounds for i in 1:n_dom
        Ui = dom_bases[i]
        for j in 1:n_cod
            Uj = cod_bases[j]
            # Nonzero iff cod upset is contained in dom upset.
            if FiniteFringe.is_subset(Uj, Ui)
                push!(I, j)
                push!(J, i)
                push!(V, one(QQ))
            end
        end
    end

    return sparse(I, J, V, n_cod, n_dom)
end


function projective_resolution(M::PModule{QQ}; maxlen::Int=3)
    # Step 0
    P0, pi0, gens0 = projective_cover(M)
    bases0 = _flatten_gens_at(gens0)

    Pmods = PModule{QQ}[P0]
    gens = Vector{Int}[bases0]
    d_mor = PMorphism{QQ}[]
    d_mat = SparseMatrixCSC{QQ, Int}[]

    # kernel K1 -> P0
    K, iota = kernel_with_inclusion(pi0)

    prevP = P0
    prevBases = bases0
    prevK = K
    prevIota = iota

    for step in 1:maxlen
        # stop if kernel is zero
        if sum(prevK.dims) == 0
            break
        end

        Pn, pin, gensn = projective_cover(prevK)
        basesn = _flatten_gens_at(gensn)

        # differential d_step = prevIota circ pin : Pn -> prevP
        d = compose(prevIota, pin)
        push!(Pmods, Pn)
        push!(gens, basesn)
        push!(d_mor, d)
        push!(d_mat, _coeff_matrix_upsets(M.Q, basesn, prevBases, d))

        # next kernel
        Kn, iotan = kernel_with_inclusion(pin)

        prevP = Pn
        prevBases = basesn
        prevK = Kn
        prevIota = iotan
    end

    res = ProjectiveResolution{QQ}(M, Pmods, gens, d_mor, d_mat, pi0)
    _pad_projective_resolution!(res, maxlen)   # pads with zero P-modules/differentials
    return res
end

# Convenience overload for fringe modules
function projective_resolution(M::FringeModule{QQ}; maxlen::Int=3)
    return projective_resolution(pmodule_from_fringe(M); maxlen=maxlen)
end

# =============================================================================
# Betti and Bass numbers (multiplicities of indecomposable summands)
# =============================================================================

# Trim trailing all-zero rows (but keep at least 1 row if the table is nonempty).
function _trim_trailing_zero_rows(B::AbstractMatrix{<:Integer})
    r, c = size(B)
    if r == 0 || c == 0
        return B
    end
    last = r
    while last > 1
        allzero = true
        for j in 1:c
            if B[last, j] != 0
                allzero = false
                break
            end
        end
        if allzero
            last -= 1
        else
            break
        end
    end
    return B[1:last, :]
end

function _pad_or_truncate_rows(B::AbstractMatrix{T}, nrows::Int) where {T<:Integer}
    r, c = size(B)
    if nrows == r
        return Matrix{T}(B)
    elseif nrows < r
        return Matrix{T}(B[1:nrows, :])
    else
        out = zeros(T, nrows, c)
        out[1:r, :] .= B
        return out
    end
end

"""
    betti(res::ProjectiveResolution{QQ}) -> Dict{Tuple{Int,Int},Int}

Return the Betti numbers of a projective resolution.

Interpretation:
- `res.Pmods[a+1]` is the projective in homological degree `a`.
- Each term splits as a direct sum of indecomposable projectives k[Up(v)].
- `res.gens[a+1]` stores the base vertex `v` for each summand, with repetition.

Output convention:
- The dictionary key `(a, v)` means "homological degree a, vertex v".
- The value is the multiplicity of k[Up(v)] in P_a.
- Only positive multiplicities appear as keys.

This is the poset-module analogue of multigraded Betti numbers in commutative algebra.
It is *not* a polynomial-ring Betti table unless you have explicitly modeled a
polynomial-ring module as a poset module and computed its resolution in that category.
"""
function betti(res::ProjectiveResolution{QQ})
    out = Dict{Tuple{Int,Int},Int}()
    L = length(res.Pmods) - 1
    for a in 0:L
        for v in res.gens[a+1]
            key = (a, v)
            out[key] = get(out, key, 0) + 1
        end
    end
    return out
end


"""
    betti_table(res::ProjectiveResolution{QQ}) -> Matrix{Int}

Return a dense Betti table B.

- Rows are homological degrees a = 0,1,2,...
- Columns are vertices v = 1,...,Q.n
- Entry B[a+1, v] is the multiplicity of k[Up(v)] in P_a.

This is purely a formatting/convenience layer over `betti(res)`.
"""
function betti_table(res::ProjectiveResolution{QQ}; pad_to::Union{Nothing,Int}=nothing)
    Q = res.M.Q
    L = length(res.Pmods) - 1
    B = zeros(Int, L + 1, Q.n)
    for a in 0:L
        for v in res.gens[a + 1]
            B[a + 1, v] += 1
        end
    end

    if pad_to === nothing
        return _trim_trailing_zero_rows(B)
    else
        pad_to >= 0 || error("betti_table: pad_to must be >= 0")
        return _pad_or_truncate_rows(B, pad_to + 1)
    end
end


"""
    betti(M::PModule{QQ}; maxlen::Int=3) -> Dict{Tuple{Int,Int},Int}

Convenience wrapper:
- build `projective_resolution(M; maxlen=maxlen)`,
- return its Betti numbers.

If you need full control over the chosen resolution object, call
`projective_resolution` yourself and then call `betti(res)`.
"""
function betti(M::PModule{QQ}; maxlen::Int=3)
    return betti(projective_resolution(M; maxlen=maxlen))
end

# ----------------------------
# Minimality diagnostics for projective resolutions
# ----------------------------

"""
    _vertex_counts(bases::Vector{Int}, nverts::Int) -> Vector{Int}

Return the multiplicity vector of vertices in `bases`.

If `bases` is the list of base vertices for a direct sum of principal upsets,
then the output c satisfies:

    c[v] = number of copies of k[Up(v)].

We use this for minimality certification, since multiplicities are the canonical
data of a minimal resolution (up to isomorphism).
"""
function _vertex_counts(bases::Vector{Int}, nverts::Int)
    c = zeros(Int, nverts)
    for v in bases
        c[v] += 1
    end
    return c
end


"""
    ProjectiveMinimalityReport

Returned by `minimality_report(res::ProjectiveResolution{QQ})`.

Fields:
- `minimal`:
    True iff all requested checks passed.

- `cover_ok`:
    True iff the augmentation P0 -> M is a projective cover, checked by comparing
    vertex multiplicities with a freshly computed cover of M.

- `cover_expected`, `cover_actual`:
    Multiplicity vectors of principal upsets in a projective cover of M and in
    the resolution's P0.

- `diagonal_violations`:
    A list of tuples (a, v, row, col, val) witnessing non-minimality in higher
    degrees. Interpretation:
      - a is the homological degree of the differential d_a : P_a -> P_{a-1},
      - v is the vertex,
      - (row, col) is an entry in the coefficient matrix of d_a,
      - val is the nonzero scalar coefficient,
      - and the entry corresponds to a map k[Up(v)] -> k[Up(v)].
    Any such nonzero scalar is an isomorphism on that indecomposable summand,
    hence it splits off a contractible subcomplex. Minimal projective resolutions
    forbid this.
"""
struct ProjectiveMinimalityReport
    minimal::Bool
    cover_ok::Bool
    cover_expected::Vector{Int}
    cover_actual::Vector{Int}
    diagonal_violations::Vector{Tuple{Int,Int,Int,Int,QQ}}
end


"""
    minimality_report(res::ProjectiveResolution{QQ}; check_cover=true) -> ProjectiveMinimalityReport

Certify minimality of a projective resolution in the standard finite-dimensional
algebra sense (incidence algebra / poset representation sense).

Checks performed:

1. (Optional) `check_cover`:
   Verify that P0 -> M is a projective cover by comparing multiplicities of
   principal upsets against a freshly computed `projective_cover(M)`.

2. Verify the "no units on diagonal" condition:
   For every differential d_a : P_a -> P_{a-1}, there is no nonzero coefficient
   from a k[Up(v)] summand in P_a to a k[Up(v)] summand in P_{a-1}.
"""
function minimality_report(res::ProjectiveResolution{QQ}; check_cover::Bool=true)
    Q = res.M.Q
    n = Q.n

    cover_actual = _vertex_counts(res.gens[1], n)
    cover_expected = copy(cover_actual)
    cover_ok = true

    if check_cover
        _, _, gens_at = projective_cover(res.M)
        cover_expected = _vertex_counts(_flatten_gens_at(gens_at), n)
        cover_ok = (cover_expected == cover_actual)
    end

    # Detect forbidden k[Up(v)] -> k[Up(v)] coefficients in the differentials.
    violations = Tuple{Int,Int,Int,Int,QQ}[]
    for a in 1:length(res.d_mat)
        D = res.d_mat[a]
        dom = res.gens[a+1]   # bases in P_a
        cod = res.gens[a]     # bases in P_{a-1}

        for col in 1:size(D, 2)
            for ptr in D.colptr[col]:(D.colptr[col+1] - 1)
                row = D.rowval[ptr]
                val = D.nzval[ptr]
                if !iszero(val) && (cod[row] == dom[col])
                    v = dom[col]
                    push!(violations, (a, v, row, col, val))
                end
            end
        end
    end

    minimal = cover_ok && isempty(violations)
    return ProjectiveMinimalityReport(minimal, cover_ok, cover_expected, cover_actual, violations)
end


"""
    is_minimal(res::ProjectiveResolution{QQ}; check_cover=true) -> Bool

Return `true` iff `minimality_report(res; check_cover=check_cover).minimal` is true.
"""
function is_minimal(res::ProjectiveResolution{QQ}; check_cover::Bool=true)
    return minimality_report(res; check_cover=check_cover).minimal
end


"""
    assert_minimal(res::ProjectiveResolution{QQ}; check_cover=true) -> true

Throw a descriptive error if the resolution fails minimality checks.
Return `true` otherwise.

This is intended for:
- test suites,
- defensively checking assumptions before extracting "minimal Betti invariants".
"""
function assert_minimal(res::ProjectiveResolution{QQ}; check_cover::Bool=true)
    R = minimality_report(res; check_cover=check_cover)
    if R.minimal
        return true
    end

    if !R.cover_ok
        error("Projective resolution is not minimal: P0 -> M is not a projective cover. " *
              "Expected cover multiplicities = $(R.cover_expected), got = $(R.cover_actual).")
    end

    if !isempty(R.diagonal_violations)
        (a, v, row, col, val) = R.diagonal_violations[1]
        error("Projective resolution is not minimal: differential d_$a has a nonzero coefficient " *
              "val = $val from k[Up($v)] in P_$a (column $col) to k[Up($v)] in P_$(a-1) (row $row).")
    end

    error("Projective resolution failed minimality checks for an unknown reason.")
end


"""
    minimal_projective_resolution(M::PModule{QQ}; maxlen=3, check=true) -> ProjectiveResolution{QQ}

Build a projective resolution via successive projective covers (the standard minimal
construction in the finite-dimensional algebra setting).

If `check=true`, call `assert_minimal` before returning.
"""
function minimal_projective_resolution(M::PModule{QQ}; maxlen::Int=3, check::Bool=true)
    res = projective_resolution(M; maxlen=maxlen)
    if check
        assert_minimal(res; check_cover=true)
    end
    return res
end


"""
    minimal_betti(M::PModule{QQ}; maxlen=3, check=true) -> Dict{Tuple{Int,Int},Int}

Compute Betti multiplicities from a certified minimal projective resolution.

This is a user-facing "invariant" function:
- it constructs the minimal resolution,
- optionally certifies minimality,
- and returns the Betti table.

If you already have a resolution in hand, use `betti(res)` instead.
"""
function minimal_betti(M::PModule{QQ}; maxlen::Int=3, check::Bool=true)
    res = minimal_projective_resolution(M; maxlen=maxlen, check=check)
    return betti(res)
end


# ----------------------------
# Ext via projective resolution: explicit cochains, cycles, boundaries, basis
# ----------------------------

struct ExtSpaceProjective{K}
    # Canonical fields (no Base.getproperty / propertynames shims).
    M::PModule{K}                              # first argument module
    N::PModule{K}                              # second argument module
    res::ProjectiveResolution{K}               # fixed projective resolution of M
    complex::ChainComplexes.CochainComplex{K}  # cochain complex Hom(P_*, N)
    offsets::Vector{Vector{Int}}               # per degree, block offsets
    cohom::Vector{ChainComplexes.CohomologyData{K}}  # per degree
    tmin::Int
    tmax::Int
end

# NOTE:
# The degree range is also available as E.complex.tmin / E.complex.tmax.
# We cache tmin/tmax as plain Int fields to keep dot-access ergonomic without
# relying on Base.getproperty overrides.

"""
    _block_offsets_for_gens(M, gens) -> Vector{Int}

Internal helper for Ext/Tor constructions.

In the projective-resolution model of Ext, each term P_a is represented as a
direct sum of indecomposable projectives k[Up(v)], recorded by the list

    res.gens[a+1] = [v1, v2, ..., vk].

For any target module M, there is a canonical identification

    Hom(k[Up(v)], M) ~= M_v

(a morphism out of a principal upset is determined by its value on the
generator at v). Consequently,

    Hom(P_a, M) ~= direct_sum_{i=1..k} M_{v_i}.

We store cochains in a single coordinate vector by concatenating these fibers.
This function returns the cumulative offsets that locate each block M_{v_i}.

Return value
------------
The returned vector offs has length k+1 with offs[1] = 0 and

    offs[i+1] = offs[i] + dim(M_{v_i}).

So the block corresponding to v_i occupies indices

    (offs[i] + 1):(offs[i+1])

in a coordinate vector.
"""
function _block_offsets_for_gens(M::PModule{K}, gens::Vector{Int}) where {K}
    offs = zeros(Int, length(gens) + 1)
    for i in 1:length(gens)
        v = gens[i]
        offs[i+1] = offs[i] + M.dims[v]
    end
    return offs
end

function _build_hom_differential(res::ProjectiveResolution{QQ}, N::PModule{QQ}, a::Int,
                                 offs_cod::Vector{Int}, offs_dom::Vector{Int})
    # a is the chain degree of the projective differential d_a: P_a -> P_{a-1}
    # On cochains: d^{a-1} : Hom(P_{a-1}, N) -> Hom(P_a, N)
    dom_gens = res.gens[a+1]      # summands of P_a
    cod_gens = res.gens[a]        # summands of P_{a-1}
    delta = res.d_mat[a]          # rows cod, cols dom

    out_dim = offs_dom[end]
    in_dim = offs_cod[end]

    Itrip = Int[]
    Jtrip = Int[]
    Vtrip = QQ[]

    Ii, Jj, Vv = findnz(delta)

    for k in 1:length(Vv)
        j = Ii[k]   # cod summand index
        i = Jj[k]   # dom summand index
        c = Vv[k]
        iszero(c) && continue

        vj = cod_gens[j]
        ui = dom_gens[i]

        A = map_leq(N, vj, ui)  # N_vj -> N_ui

        # Insert into block (rows for ui) x (cols for vj)
        _append_scaled_triplets!(Itrip, Jtrip, Vtrip, A,
                                offs_dom[i], offs_cod[j]; scale=c)
    end

    return sparse(Itrip, Jtrip, Vtrip, out_dim, in_dim)
end

function Ext(M::PModule{QQ}, N::PModule{QQ}; maxdeg::Int=3)
    res = projective_resolution(M; maxlen=maxdeg)
    _pad_projective_resolution!(res, maxdeg)
    return Ext(res, N)
end


function Ext(res::ProjectiveResolution{QQ}, N::PModule{QQ})
    # Canonical record of the first argument module.
    # (The resolution already stores this as res.M, but we duplicate the field
    # on ExtSpaceProjective so downstream code can use E.M without any
    # Base.getproperty magic.)
    M = res.M

    L = length(res.Pmods) - 1
    # build cochain dims and offsets
    dimsC = Int[]
    offs = Vector{Vector{Int}}()
    for a in 0:L
        oa = _block_offsets_for_gens(N, res.gens[a+1])
        push!(offs, oa)
        push!(dimsC, oa[end])
    end

    # differentials d^a : C^a -> C^{a+1} for a=0..L-1
    dC = Vector{SparseMatrixCSC{QQ, Int}}()
    for a in 1:L
        push!(dC, _build_hom_differential(res, N, a, offs[a], offs[a+1]))
    end

    C = ChainComplexes.CochainComplex{QQ}(0, L, dimsC, dC, [Any[] for _ in 0:L])
    cohom = [ChainComplexes.cohomology_data(C, t) for t in 0:L]
    return ExtSpaceProjective{QQ}(M, N, res, C, offs, cohom, C.tmin, C.tmax)
end

# Convenience: fringe modules
function Ext(M::FringeModule{QQ}, N::FringeModule{QQ}; maxdeg::Int=3)
    Mp = pmodule_from_fringe(M)
    Np = pmodule_from_fringe(N)
    return Ext(Mp, Np; maxdeg=maxdeg)
end

# Internal helper: the traditional projective-resolution model of Ext.
# This is the behavior that `Ext(M,N)` historically had in this codebase.
function _Ext_projective(M::PModule{QQ}, N::PModule{QQ}; maxdeg::Int=3)
    res = projective_resolution(M; maxlen=maxdeg)
    _pad_projective_resolution!(res, maxdeg)
    return Ext(res, N)
end

"""
    Ext(M, N; maxdeg=3, model=:projective, canon=:projective)

Compute Ext^t(M, N) for 0 <= t <= maxdeg.

The keyword `model` controls which computational realization is returned:

- `model = :projective` (default) returns an `ExtSpaceProjective`, computed from a
  projective resolution of `M`. This matches the historical behavior of this codebase.

- `model = :injective` returns an `ExtSpaceInjective`, computed from an injective
  resolution of `N`.

- `model = :unified` (aliases: `:bifunctor`, `:auto`) returns a model-independent
  `ExtSpace` object that stores BOTH models together with explicit comparison
  isomorphisms and coherent basis transport. The user-facing coordinate basis is
  chosen by `canon` (either `:projective` or `:injective`).

Notes
-----
The returned object determines the basis in which `basis`, `coordinates`,
`representative`, and induced-map helpers (e.g. `ext_map_first/second`) are expressed.

The unified `ExtSpace` lets you ask for representatives in either model while
keeping a fixed, canonical coordinate basis for "the same Ext class".
"""
function Ext(
    M::PModule{QQ},
    N::PModule{QQ};
    maxdeg::Int=3,
    model::Symbol=:projective,
    canon::Symbol=:projective
)
    if model === :projective
        return _Ext_projective(M, N; maxdeg=maxdeg)
    elseif model === :injective
        return ExtInjective(M, N; maxdeg=maxdeg)
    elseif model === :unified || model === :bifunctor || model === :auto
        return ExtSpace(M, N; maxdeg=maxdeg, canon=canon)
    else
        error("Ext: unknown model=$(model). Supported: :projective, :injective, :unified.")
    end
end

function Ext(
    M::FringeModule{QQ},
    N::FringeModule{QQ};
    maxdeg::Int=3,
    model::Symbol=:projective,
    canon::Symbol=:projective
)
    return Ext(M.M, N.M; maxdeg=maxdeg, model=model, canon=canon)
end

function dim(E::ExtSpaceProjective, t::Int)
    if t < E.tmin || t > E.tmax
        return 0
    end
    return E.cohom[t+1].dimH
end

function cycles(E::ExtSpaceProjective, t::Int)
    return E.cohom[t+1].K
end

function boundaries(E::ExtSpaceProjective, t::Int)
    return E.cohom[t+1].B
end

function representative(E::ExtSpaceProjective, t::Int, i::Int)
    Hrep = E.cohom[t+1].Hrep
    return Hrep[:, i]
end

"""
    representative(E::ExtSpaceProjective, t::Int, coords::AbstractVector{QQ}) -> Vector{QQ}

Return an explicit cocycle representative in the cochain space C^t of the Ext class
whose coordinates (in the basis chosen internally by `E`) are given by `coords`.

Mathematically:
- `E` stores a basis of H^t(C) by choosing cocycle representatives (columns of `Hrep`).
- This function returns the linear combination of those cocycles.

This is useful when you want:
- explicit chain-level representatives of arbitrary Ext elements,
- Yoneda products on representatives,
- custom linear combinations without manually forming them.
"""
function representative(E::ExtSpaceProjective{QQ}, t::Int, coords::AbstractVector{QQ})
    if t < 0 || t > E.tmax
        error("representative: degree t must satisfy 0 <= t <= tmax.")
    end
    Hrep = E.cohom[t+1].Hrep
    if length(coords) != size(Hrep, 2)
        error("representative: coordinate vector has length $(length(coords)), expected $(size(Hrep,2)).")
    end
    v = Hrep * reshape(coords, :, 1)
    return vec(v)
end


function basis(E::ExtSpaceProjective, t::Int)
    Hrep = E.cohom[t+1].Hrep
    out = Vector{Vector{QQ}}(undef, size(Hrep, 2))
    for i in 1:size(Hrep, 2)
        out[i] = Vector{QQ}(Hrep[:, i])
    end
    return out
end

# Split a cochain vector into a list of fiber-vectors, one per projective summand generator.
function split_cochain(E::ExtSpaceProjective, t::Int, v::AbstractVector{QQ})
    offs = E.offsets[t+1]
    gens = E.res.gens[t+1]
    parts = Vector{Vector{QQ}}(undef, length(gens))
    for i in 1:length(gens)
        parts[i] = Vector{QQ}(v[(offs[i]+1):offs[i+1]])
    end
    return gens, parts
end

# Reduce a cocycle in C^t to Ext-coordinates in the chosen basis.
function coordinates(E::ExtSpaceProjective, t::Int, cocycle::AbstractVector{QQ})
    data = E.cohom[t+1]
    return ChainComplexes.cohomology_coordinates(data, cocycle)[:, 1]
end

# ----------------------------
# Functoriality in second argument: g : N -> N2 induces Ext(M,N) -> Ext(M,N2)
# ----------------------------

function _blockdiag_on_hom_cochains(g::PMorphism{QQ}, gens::Vector{Int}, offs_src::Vector{Int}, offs_tgt::Vector{Int})
    F = zeros(QQ, offs_tgt[end], offs_src[end])
    for i in 1:length(gens)
        u = gens[i]
        rows = (offs_tgt[i] + 1):(offs_tgt[i+1])
        cols = (offs_src[i] + 1):(offs_src[i+1])
        F[rows, cols] = g.comps[u]
    end
    return F
end

function _blockdiag_on_hom_cochains_sparse(
    g::PMorphism{QQ},
    gens::Vector{Int},
    offs_src::Vector{Int},
    offs_tgt::Vector{Int}
)
    out_dim = offs_tgt[end]
    in_dim  = offs_src[end]

    I = Int[]
    J = Int[]
    V = QQ[]

    @inbounds for i in 1:length(gens)
        u = gens[i]
        _append_scaled_triplets!(I, J, V, g.comps[u], offs_tgt[i], offs_src[i])
    end

    return sparse(I, J, V, out_dim, in_dim)
end


function ext_map_second(E1::ExtSpaceProjective{QQ}, E2::ExtSpaceProjective{QQ}, g::PMorphism{QQ}; t::Int)
    # Both E1 and E2 must be built from the same projective resolution (same M).
    @assert length(E1.res.gens) == length(E2.res.gens)
    gens_t = E1.res.gens[t+1]
    F = _blockdiag_on_hom_cochains(g, gens_t, E1.offsets[t+1], E2.offsets[t+1])
    return ChainComplexes.induced_map_on_cohomology(E1.cohom[t+1], E2.cohom[t+1], F)
end

# -------------------------------------------------------------------------------------
# Functoriality: Ext via projective resolutions
# -------------------------------------------------------------------------------------

# Internal: given a map of projectives P_dom -> P_cod expressed in the indicator-summand
# bases by the coefficient matrix `coeff`, build the induced map on Hom cochains
#
#   Hom(P_cod, N) -> Hom(P_dom, N)
#
# by precomposition. This is the same pattern as _build_hom_differential, but with an
# arbitrary coefficient matrix instead of a differential from a resolution.
function _precompose_on_hom_cochains_from_projective_coeff(
    N::PModule{QQ},
    dom_gens::Vector{Int},
    cod_gens::Vector{Int},
    dom_offsets::Vector{Int},
    cod_offsets::Vector{Int},
    coeff::AbstractMatrix{QQ}
)
    out_dim = dom_offsets[end]
    in_dim  = cod_offsets[end]

    I = Int[]
    J = Int[]
    V = QQ[]

    # Fast path: sparse coeff -> iterate only nonzero blocks.
    if issparse(coeff)
        Icoeff, Jcoeff, Vcoeff = findnz(coeff)  # rows, cols, vals
        @inbounds for k in eachindex(Vcoeff)
            j = Icoeff[k]   # cod index
            i = Jcoeff[k]   # dom index
            c = Vcoeff[k]
            iszero(c) && continue

            ui = dom_gens[i]
            vj = cod_gens[j]

            A = map_leq(N, vj, ui)
            _append_scaled_triplets!(I, J, V, A, dom_offsets[i], cod_offsets[j]; scale=c)
        end
    else
        # Dense coeff -> scan directly (avoid sparse(coeff) + findnz intermediates).
        @inbounds for i in 1:length(dom_gens)
            ui = dom_gens[i]
            for j in 1:length(cod_gens)
                c = coeff[j, i]
                iszero(c) && continue

                vj = cod_gens[j]
                A  = map_leq(N, vj, ui)
                _append_scaled_triplets!(I, J, V, A, dom_offsets[i], cod_offsets[j]; scale=c)
            end
        end
    end

    return sparse(I, J, V, out_dim, in_dim)
end



# Internal: lift a module morphism f : M -> Mp to a chain map between the chosen
# projective resolutions resM -> resMp. The output is a vector F where
#
#   F[k+1] is the coefficient matrix of the chain map component P_k(M) -> P_k(Mp)
#
# written in the indicator-summand bases of resM and resMp.
#
# This uses the same deterministic lifting routine used for Yoneda products (q = 0).
function _lift_pmodule_map_to_projective_resolution_chainmap_coeff(
    resM::ProjectiveResolution{QQ},
    resMp::ProjectiveResolution{QQ},
    f::PMorphism{QQ};
    upto::Int
)
    @assert f.dom === resM.M
    @assert f.cod === resMp.M
    @assert upto >= 0
    @assert upto <= (length(resM.Pmods) - 1)
    @assert upto <= (length(resMp.Pmods) - 1)

    # View f as a degree-0 cocycle in Hom(P_0(M), Mp) by composing with the
    # augmentation P_0(M) -> M.
    alpha_mor = compose(f, resM.aug)

    # We only need the splitting/offset conventions for degree 0; Ext(resM, Mp)
    # provides those consistently with the resolution resM that we are using.
    EMMp = Ext(resM, resMp.M)
    alpha_vec = _cochain_vector_from_morphism(EMMp, 0, alpha_mor)

    return _lift_cocycle_to_chainmap_coeff(resM, resMp, EMMp, 0, alpha_vec; upto=upto)
end

"""
    lift_chainmap(res_dom, res_cod, f; maxlen, check=true)

Lift a P-module morphism `f : res_dom.M -> res_cod.M` to a chain map between the
chosen projective resolutions `res_dom` and `res_cod`.

This returns a vector `H` of length `maxlen+1` where `H[k+1]` is the coefficient
matrix of the lifted map on degree `k`:

    H[k+1] : res_dom.Pmods[k+1] -> res_cod.Pmods[k+1]

in the indicator-summand bases used by the projective resolutions.

Notes
-----
- This is deliberately "low-level": it returns coefficient matrices (sparse).
  `ChangeOfPosets.pushforward_left_complex` converts these into `PMorphism`s
  using `_pmorphism_from_upset_coeff`.
- If you want the lift only up to the resolution length, you should pass
  `maxlen <= min(length(res_dom.Pmods), length(res_cod.Pmods)) - 1`.
"""
function lift_chainmap(res_dom::ProjectiveResolution{QQ},
                       res_cod::ProjectiveResolution{QQ},
                       f::PMorphism{QQ};
                       maxlen::Int,
                       check::Bool=true)

    if check
        @assert f.dom === res_dom.M "lift_chainmap: f.dom must equal res_dom.M"
        @assert f.cod === res_cod.M "lift_chainmap: f.cod must equal res_cod.M"
    end

    # Internal routine expects `upto` = highest homological degree to lift.
    return _lift_pmodule_map_to_projective_resolution_chainmap_coeff(
        res_dom, res_cod, f; upto=maxlen
    )
end


"""
    ext_map_first(EMN::ExtSpaceProjective{QQ}, EMPN::ExtSpaceProjective{QQ}, f::PMorphism{QQ}; t::Int)

Contravariant map in the first argument for the projective-resolution Ext model.

Given a morphism `f : M -> Mp` and a fixed module `N`, this returns the induced map

    Ext^t(Mp, N) -> Ext^t(M, N)

with respect to the bases stored in `EMPN` (source) and `EMN` (target).

This method:
1. Lifts `f` to a chain map between the stored projective resolutions
   `P(M) -> P(Mp)` (deterministically, using the same lifting routine as the Yoneda product).
2. Applies `Hom(-,N)` to obtain a cochain map
   `Hom(P(Mp),N) -> Hom(P(M),N)`.
3. Passes to cohomology in degree `t`.

See also: `ext_map_second`.
"""
function ext_map_first(
    EMN::ExtSpaceProjective{QQ},
    EMPN::ExtSpaceProjective{QQ},
    f::PMorphism{QQ};
    t::Int
)
    @assert EMN.N === EMPN.N
    @assert EMN.res.M === f.dom
    @assert EMPN.res.M === f.cod
    @assert t >= 0
    @assert t <= EMN.tmax
    @assert t <= EMPN.tmax

    # Lift f to a chain map between the chosen projective resolutions up to degree t.
    coeffs = _lift_pmodule_map_to_projective_resolution_chainmap_coeff(
        EMN.res, EMPN.res, f; upto=t
    )
    coeff_t = coeffs[t+1]  # P_t(M) -> P_t(Mp)

    # The induced cochain map at degree t is precomposition:
    # Hom(P_t(Mp), N) -> Hom(P_t(M), N).
    dom_gens = EMN.res.gens[t+1]    # generators of P_t(M)
    cod_gens = EMPN.res.gens[t+1]   # generators of P_t(Mp)

    F = _precompose_on_hom_cochains_from_projective_coeff(
        EMN.N,
        dom_gens,
        cod_gens,
        coeff_t,
        EMN.offsets[t+1],
        EMPN.offsets[t+1]
    )

    return induced_map_on_cohomology(EMPN.cohom[t+1], EMN.cohom[t+1], F)
end

# NOTE:
# The Ext functoriality helper ext_map_second for the injective model is defined
# later in this file, after ExtSpaceInjective is defined.

# ----------------------------
# Connecting homomorphism for 0 -> A --i--> B --p--> C -> 0 in the second argument
# Uses the projective model, so Hom(P_a,-) is exact.
# ----------------------------

function connecting_hom(EA::ExtSpaceProjective{QQ}, EB::ExtSpaceProjective{QQ}, EC::ExtSpaceProjective{QQ},
                        i::PMorphism{QQ}, p::PMorphism{QQ}; t::Int)
    
    # Sanity checks (avoid BoundsError when the resolution is too short).
    if EA.res !== EB.res || EA.res !== EC.res
        error("connecting_hom: EA, EB, EC must share the same ProjectiveResolution.")
    end
    if t < 0
        error("connecting_hom: t must be >= 0.")
    end
    # To compute delta^t we need resolution data through degree t+1.
    if t + 1 > EA.tmax || t + 1 > EB.tmax || t > EC.tmax
        error("connecting_hom: need EA and EB computed through degree t+1=$(t+1) and EC through degree t=$t. Recompute with larger maxdeg/maxlen.")
    end
    
    
    # delta : Ext^t(M,C) -> Ext^{t+1}(M,A)

    # cochain degree t maps:
    gens_t = EA.res.gens[t+1]
    It = _blockdiag_on_hom_cochains(i, gens_t, EA.offsets[t+1], EB.offsets[t+1])   # CA^t -> CB^t
    Pt = _blockdiag_on_hom_cochains(p, gens_t, EB.offsets[t+1], EC.offsets[t+1])   # CB^t -> CC^t

    gens_tp1 = EA.res.gens[t+2]
    Itp1 = _blockdiag_on_hom_cochains(i, gens_tp1, EA.offsets[t+2], EB.offsets[t+2])
    Ptp1 = _blockdiag_on_hom_cochains(p, gens_tp1, EB.offsets[t+2], EC.offsets[t+2])

    dBt = EB.complex.d[t+1]      # CB^t -> CB^{t+1}
    # Basis of Ext^t(M,C) as cocycles (columns)
    HrepC = EC.cohom[t+1].Hrep
    out = zeros(QQ, EA.cohom[t+2].dimH, EC.cohom[t+1].dimH)

    for j in 1:size(HrepC, 2)
        z = HrepC[:, j]  # cocycle in CC^t

        # lift to y in CB^t: Pt * y = z
        y = ChainComplexes.solve_particularQQ(Pt, Matrix{QQ}(reshape(z, :, 1)))[:, 1]

        # dy in CB^{t+1}
        dy = dBt * y

        # dy is in ker(Ptp1) = im(Itp1), solve Itp1 * x = dy
        x = ChainComplexes.solve_particularQQ(Itp1, Matrix{QQ}(reshape(dy, :, 1)))[:, 1]

        # reduce x to Ext^{t+1}(M,A) coordinates
        coords = ChainComplexes.cohomology_coordinates(EA.cohom[t+2], x)
        out[:, j] = coords[:, 1]
    end

    return out
end

struct ExtLongExactSequenceSecond{K}
    tmin::Int
    tmax::Int
    EA::ExtSpaceProjective{K}
    EB::ExtSpaceProjective{K}
    EC::ExtSpaceProjective{K}
    iH::Vector{Matrix{K}}      # Ext^t(M,A) -> Ext^t(M,B), index t+1
    pH::Vector{Matrix{K}}      # Ext^t(M,B) -> Ext^t(M,C), index t+1
    deltaH::Vector{Matrix{K}}  # Ext^t(M,C) -> Ext^{t+1}(M,A), index t+1
end

"""
    ExtLongExactSequenceSecond(M, A, B, C, i, p; maxdeg)

Package the long exact sequence in the second argument of Ext coming from
a short exact sequence 0 -> A --i--> B --p--> C -> 0:

... -> Ext^t(M,A) -> Ext^t(M,B) -> Ext^t(M,C) -> Ext^{t+1}(M,A) -> ...

The object stores maps for t = 0..maxdeg, including the connecting maps delta^t.
Internally, a single projective resolution of M is built and shared across EA, EB, EC.
"""
function ExtLongExactSequenceSecond(M::PModule{QQ},
                                   A::PModule{QQ},
                                   B::PModule{QQ},
                                   C::PModule{QQ},
                                   i::PMorphism{QQ},
                                   p::PMorphism{QQ};
                                   maxdeg::Int)
    # Need Ext up to degree maxdeg+1 to define delta^maxdeg.
    res = projective_resolution(M; maxlen=maxdeg+1)
    _pad_projective_resolution!(res, maxdeg+1)

    EA = Ext(res, A)
    EB = Ext(res, B)
    EC = Ext(res, C)

    iH = Matrix{QQ}[]
    pH = Matrix{QQ}[]
    deltaH = Matrix{QQ}[]
    for t in 0:maxdeg
        push!(iH, ext_map_second(EA, EB, i; t=t))
        push!(pH, ext_map_second(EB, EC, p; t=t))
        push!(deltaH, connecting_hom(EA, EB, EC, i, p; t=t))
    end

    return ExtLongExactSequenceSecond{QQ}(0, maxdeg, EA, EB, EC, iH, pH, deltaH)
end


# -----------------------------------------------------------------------------
# Convenience wrappers accepting a ShortExactSequence
# -----------------------------------------------------------------------------

"""
    ExtLongExactSequenceSecond(M, ses; maxdeg)

Convenience wrapper: build the Ext long exact sequence in the second argument from
a checked short exact sequence object.
"""
function ExtLongExactSequenceSecond(M::PModule{QQ}, ses::ShortExactSequence{QQ}; maxdeg::Int)
    return ExtLongExactSequenceSecond(M, ses.A, ses.B, ses.C, ses.i, ses.p; maxdeg=maxdeg)
end

"""
    connecting_hom(EA, EB, EC, ses; t)

Convenience wrapper for the Ext connecting morphism when the short exact sequence is
passed as a `ShortExactSequence` object.
"""
function connecting_hom(EA::ExtSpaceProjective{QQ},
                        EB::ExtSpaceProjective{QQ},
                        EC::ExtSpaceProjective{QQ},
                        ses::ShortExactSequence{QQ}; t::Int)
    return connecting_hom(EA, EB, EC, ses.i, ses.p; t=t)
end



# =============================================================================
# Yoneda product on Ext (projective-resolution model)
# =============================================================================

# -----------------------------------------------------------------------------
# Poset/module compatibility checks (do not assume FinitePoset has a `covers` field)
# -----------------------------------------------------------------------------

function _same_poset(Q1::FinitePoset, Q2::FinitePoset)::Bool
    # The leq-matrix is the canonical source of truth.
    if Q1.n != Q2.n
        return false
    end
    if Q1.leq != Q2.leq
        return false
    end

    # Some layers of the codebase historically stored cover edges as a cached field
    # `Q.covers`.  The current finite-poset layer computes cover edges from `leq`
    # via `cover_edges(Q)`.  We avoid relying on a non-existent or stale cache.
    C1 = cover_edges(Q1)
    C2 = cover_edges(Q2)
    return convert(BitMatrix, C1) == convert(BitMatrix, C2)
end


function _assert_same_pmodule_structure(A::PModule{QQ}, B::PModule{QQ}, ctx::String)
    if !_same_poset(A.Q, B.Q)
        error("$ctx: modules live on different posets.")
    end
    if A.dims != B.dims
        error("$ctx: modules have different fiber dimensions.")
    end

    # Compare cover-edge structure maps.
    # Fast path when the poset object is shared: compare store-aligned arrays in lockstep
    # (no tuple keys, no searching, no allocating default zeros).
    if A.Q === B.Q && (A.edge_maps.succs === B.edge_maps.succs)
        storeA = A.edge_maps
        storeB = B.edge_maps
        succs = storeA.succs
        mapsA = storeA.maps_to_succ
        mapsB = storeB.maps_to_succ

        @inbounds for u in 1:A.Q.n
            su = succs[u]
            Au = mapsA[u]
            Bu = mapsB[u]
            for j in eachindex(su)
                v = su[j]
                if Au[j] != Bu[j]
                    error("$ctx: modules have different structure maps on cover edge ($u,$v).")
                end
            end
        end
    else
        # Safe path for structurally-equal but not pointer-equal posets.
        for (u, v) in cover_edges(A.Q)
            if A.edge_maps[u, v] != B.edge_maps[u, v]
                error("$ctx: modules have different structure maps on cover edge ($u,$v).")
            end
        end
    end
    return nothing
end



# -----------------------------------------------------------------------------
# Internal: lift a cocycle in Hom(P_q(L), M) to a degree-q chain map P(L) -> P(M)
# (enough components to compose with a degree-p cocycle from Ext^p(M,N)).
# -----------------------------------------------------------------------------

# ---------------------------------------------------------------------
# Convenience overload: allow a single cocycle to be passed as a vector.
#
# Internally, the lifting routine is implemented for matrices because it
# can lift multiple cocycles simultaneously (each column is one RHS).
# Many user-facing helpers (and `representative`) naturally produce a
# single cocycle as a vector, so we interpret that as a 1-column matrix.
# ---------------------------------------------------------------------
function _lift_cocycle_to_chainmap_coeff(resL::ProjectiveResolution{QQ},
                                        resM::ProjectiveResolution{QQ},
                                        E_LM::ExtSpaceProjective{QQ},
                                        q::Int,
                                        alpha_cocycle::AbstractVector{QQ};
                                        upto::Int)
    # Treat the cochain vector as a single column.
    alpha_mat = reshape(alpha_cocycle, :, 1)
    return _lift_cocycle_to_chainmap_coeff(resL, resM, E_LM, q, alpha_mat; upto=upto)
end

"""
    _lift_cocycle_to_chainmap_coeff(resL, resM, E_LM, q, alpha_cocycle; upto)

Given:
- `resL`: a projective resolution of L,
- `resM`: a projective resolution of M,
- a cocycle `alpha_cocycle` in C^q = Hom(P_q(L), M),

construct (non-canonically, but deterministically) coefficient matrices describing
a degree-q chain map
    F : P(L) -> P(M),
i.e. maps
    F_k : P_{q+k}(L) -> P_k(M)
for k = 0,1,...,upto

Return value:
- a vector `F` where `F[k+1]` is the coefficient matrix of F_k.

This is the standard "comparison map" construction used to implement the Yoneda
product via projective resolutions.
"""
function _lift_cocycle_to_chainmap_coeff(resL::ProjectiveResolution{QQ},
                                         resM::ProjectiveResolution{QQ},
                                         E_LM::ExtSpaceProjective{QQ},
                                         q::Int,
                                         alpha_cocycle::AbstractMatrix{QQ};
                                         upto::Int = 0)

    # Return a chain map F lifting the cocycle alpha:
    #
    #   alpha : P_q(L) -> M
    #
    # The lift is a chain map between projective resolutions:
    #
    #   F_k : P_{q+k}(L) -> P_k(M)
    #
    # represented by coefficient matrices in the chosen direct-sum bases.
    #
    # Performance note:
    # Historically this routine built dense coefficient matrices via zeros(QQ, ...)
    # and then filled only the entries allowed by the poset constraints. In most
    # poset-resolution situations, these chain maps are sparse, so dense zeros are
    # a large avoidable cost. We now build SparseMatrixCSC directly from (row,col,val)
    # triples. Downstream code uses AbstractMatrix{QQ}, so sparsity is compatible.

    if upto < 0
        error("_lift_cocycle_to_chainmap_coeff: keyword `upto` must be >= 0.")
    end

    # Precompute which summands in each P_k(M) are eligible to receive maps from a base u.
    active_M = Vector{Dict{Int, Vector{Int}}}(undef, upto + 1)
    for k in 0:upto
        active_M[k+1] = Dict{Int, Vector{Int}}()
        cod_bases_k = resM.gens[k+1]  # indices in the direct sum decomposition of P_k(M)
        for u in 1:resM.M.Q.n
            allowed = Int[]
            for j in 1:length(cod_bases_k)
                v = cod_bases_k[j]
                if resM.M.Q.leq[v, u]
                    push!(allowed, j)
                end
            end
            active_M[k+1][u] = allowed
        end
    end

    # Coefficients of F_k, stored as SparseMatrixCSC{QQ}.
    # Convention: F[k+1] is the coefficient matrix for F_k.
    F = Vector{SparseMatrixCSC{QQ, Int}}(undef, upto + 1)

    # Step k = 0: solve for F_0 column-by-column via augmented constraints from alpha.
    dom_gens_q  = resL.gens[q+1]      # summands in P_q(L)
    cod_bases_0 = resM.gens[1]        # summands in P_0(M)

    m0 = length(cod_bases_0)
    n0 = length(dom_gens_q)

    I0 = Int[]   # row indices
    J0 = Int[]   # col indices
    V0 = QQ[]    # values

    # alpha_cocycle is stored in the cochain basis of E_LM at degree q:
    # it is a concatenation of blocks in M(u), one block for each generator u in P_q(L).
    # Split it into one RHS vector per generator of P_q(L).
    if size(alpha_cocycle, 2) != 1
        throw(ArgumentError("_lift_cocycle_to_chainmap_coeff currently expects a single cocycle column (got $(size(alpha_cocycle,2)))."))
    end
    _, alpha_parts = split_cochain(E_LM, q, Vector{QQ}(alpha_cocycle[:, 1]))

    for i in 1:n0
        u = dom_gens_q[i]  # base vertex of the i-th summand in P_q(L)

        rhs = alpha_parts[i]
        if isempty(rhs)
            continue
        end

        act = active_M[1][u]
        if isempty(act)
            continue
        end

        # Lift along the augmentation pi_u : P_0(M)(u) -> M(u).
        # resM.aug.comps[u] already uses the fiber basis corresponding to `act` (same order).
        pi_u = Matrix(resM.aug.comps[u])
        x = ChainComplexes.solve_particularQQ(pi_u, reshape(rhs, :, 1))  # length(act) x 1

        for (pos, j) in enumerate(act)
            c = x[pos, 1]
            if !iszero(c)
                push!(I0, j)
                push!(J0, i)
                push!(V0, c)
            end
        end
    end

    F[1] = sparse(I0, J0, V0, m0, n0)

    # Steps k >= 1: solve the chain-map equations
    #   d_k^M * F_k = F_{k-1} * d_{q+k}^L
    for k in 1:upto
        DkM  = Matrix(resM.d_mat[k])          # P_k(M) -> P_{k-1}(M)
        DqkL = Matrix(resL.d_mat[q + k])      # P_{q+k}(L) -> P_{q+k-1}(L)

        RHS = F[k] * DqkL                     # matrix in Hom(P_{q+k}(L), P_{k-1}(M))

        cod_bases_k  = resM.gens[k+1]         # summands in P_k(M) (these are the columns of DkM)
        dom_bases_qk = resL.gens[q+k+1]       # summands in P_{q+k}(L)

        mk = length(cod_bases_k)
        nk = length(dom_bases_qk)

        Ik = Int[]
        Jk = Int[]
        Vk = QQ[]

        for col in 1:nk
            u = dom_bases_qk[col]
            allowed = active_M[k+1][u]        # eligible summands in P_k(M)

            # If no columns are allowed, RHS[:,col] must be zero.
            if isempty(allowed)
                if any(!iszero, RHS[:, col])
                    error("_lift_cocycle_to_chainmap_coeff: inconsistent constraints at (k=$k, dom_summand=$col).")
                end
                continue
            end

            A = DkM[:, allowed]               # restrict to allowed columns
            b = RHS[:, col]
            x = ChainComplexes.solve_particularQQ(A, reshape(b, :, 1))  # length(allowed) x 1

            for (pos, j) in enumerate(allowed)
                c = x[pos, 1]
                if !iszero(c)
                    push!(Ik, j)
                    push!(Jk, col)
                    push!(Vk, c)
                end
            end
        end

        F[k+1] = sparse(Ik, Jk, Vk, mk, nk)
    end

    return F
end



# -----------------------------------------------------------------------------
# Internal: compose a chain-map component into a cocycle for Hom(P_{p+q}(L), N).
# -----------------------------------------------------------------------------

function _compose_into_module_cocycle(resL::ProjectiveResolution{QQ},
                                      resM::ProjectiveResolution{QQ},
                                      N::PModule{QQ},
                                      p::Int,
                                      q::Int,
                                      Fp::AbstractMatrix{QQ},
                                      beta_cocycle::AbstractVector{QQ},
                                      E_MN::ExtSpaceProjective{QQ},
                                      E_LN::ExtSpaceProjective{QQ})

    deg = p + q
    dom_bases = resL.gens[deg+1]   # summands in P_{p+q}(L)
    mid_bases = resM.gens[p+1]     # summands in P_p(M)

    # Split beta into its pieces on the summands of P_p(M).
    _, beta_parts = split_cochain(E_MN, p, beta_cocycle)

    offs = E_LN.offsets[deg+1]
    out = zeros(QQ, offs[end])

    # If Fp is sparse, iterate only over nonzeros in each column.
    if issparse(Fp)
        Fps = sparse(Fp)  # ensure SparseMatrixCSC so colptr/rowval/nzval exist

        for i in 1:length(dom_bases)
            u = dom_bases[i]
            block = zeros(QQ, N.dims[u])

            # Nonzeros in column i live in ptr range [colptr[i], colptr[i+1)-1].
            ptr_lo = Fps.colptr[i]
            ptr_hi = Fps.colptr[i+1] - 1

            for ptr in ptr_lo:ptr_hi
                j = Fps.rowval[ptr]
                c = Fps.nzval[ptr]
                # c is guaranteed nonzero in SparseMatrixCSC, but keep it defensive.
                if iszero(c)
                    continue
                end

                v = mid_bases[j]
                if !resL.M.Q.leq[v, u]
                    continue
                end

                A = map_leq(N, v, u)           # N_v -> N_u
                block .+= c .* (A * beta_parts[j])
            end

            out[(offs[i]+1):offs[i+1]] = block
        end

        return out
    end

    # Dense fallback: original behavior.
    for i in 1:length(dom_bases)
        u = dom_bases[i]
        block = zeros(QQ, N.dims[u])

        for j in 1:length(mid_bases)
            c = Fp[j, i]
            if iszero(c)
                continue
            end
            v = mid_bases[j]
            if !resL.M.Q.leq[v, u]
                continue
            end

            A = map_leq(N, v, u)
            block .+= c .* (A * beta_parts[j])
        end

        out[(offs[i]+1):offs[i+1]] = block
    end

    return out
end


# -----------------------------------------------------------------------------
# Public API: Yoneda product
# -----------------------------------------------------------------------------

"""
    yoneda_product(E_MN, p, beta_coords, E_LM, q, alpha_coords; ELN=nothing, return_cocycle=false)

Compute the Yoneda product

    Ext^p(M, N) x Ext^q(L, M) -> Ext^{p+q}(L, N).

Inputs:
- `E_MN` is an `ExtSpaceProjective` for (M,N).
- `E_LM` is an `ExtSpaceProjective` for (L,M).
- `beta_coords` are coordinates of a class in Ext^p(M,N) in the basis used by `E_MN`.
- `alpha_coords` are coordinates of a class in Ext^q(L,M) in the basis used by `E_LM`.

Output:
- `(E_LN, coords)` where `coords` are coordinates of the product class in Ext^{p+q}(L,N)
  in the basis used by `E_LN`.
- If `return_cocycle=true`, returns `(E_LN, coords, cocycle)` where `cocycle` is an explicit
  representative in the cochain space Hom(P_{p+q}(L), N).

Notes for mathematicians:
- This implements the classical Yoneda product by constructing a comparison map
  between projective resolutions and composing at the chain level.
- The result is well-defined in cohomology; chain-level representatives depend on
  deterministic but non-canonical lift choices (as always).

Technical requirements:
- `E_MN` must have `tmax >= p`.
- `E_LM` must have `tmax >= p+q` (because we need P_{p+q}(L)).
- The "middle" module M used by `E_MN.res` and the second argument of `E_LM` must agree
  as poset-modules (same fibers and structure maps).
"""
function yoneda_product(E_MN::ExtSpaceProjective{QQ},
                        p::Int,
                        beta_coords::AbstractVector{QQ},
                        E_LM::ExtSpaceProjective{QQ},
                        q::Int,
                        alpha_coords::AbstractVector{QQ};
                        ELN::Union{Nothing,ExtSpaceProjective{QQ}}=nothing,
                        return_cocycle::Bool=false)

    if p < 0 || q < 0
        error("yoneda_product: degrees p and q must be >= 0.")
    end
    if p > E_MN.tmax
        error("yoneda_product: E_MN.tmax is too small for p = $p.")
    end
    if (p + q) > E_LM.tmax
        error("yoneda_product: E_LM.tmax is too small for p+q = $(p+q).")
    end

    resM = E_MN.res
    resL = E_LM.res
    N = E_MN.N

    # Compatibility: the middle module in Ext^q(L,M) must match the resolved module M.
    _assert_same_pmodule_structure(E_LM.N, resM.M, "yoneda_product (middle module check)")

    # Build (or validate) the target Ext space Ext(L,N).
    if ELN === nothing
        ELN_use = Ext(resL, N)
    else
        ELN_use = ELN
        # Very conservative checks: same resolved L and same N.
        _assert_same_pmodule_structure(ELN_use.N, N, "yoneda_product (target N check)")
        _assert_same_pmodule_structure(ELN_use.res.M, resL.M, "yoneda_product (target L check)")
        if ELN_use.tmax < (p + q)
            error("yoneda_product: provided ELN has tmax < p+q.")
        end
    end

    # Convert coordinates to explicit cocycles.
    beta_cocycle  = representative(E_MN, p, beta_coords)
    alpha_cocycle = reshape(representative(E_LM, q, alpha_coords), :, 1)

    # Lift alpha to a degree-q chain map into the projective resolution of M, up to component p.
    F = _lift_cocycle_to_chainmap_coeff(resL, resM, E_LM, q, alpha_cocycle; upto=p)
    Fp = F[p+1]  # P_{p+q}(L) -> P_p(M)

    # Compose at chain level to get a cocycle in Hom(P_{p+q}(L), N).
    cocycle = _compose_into_module_cocycle(resL, resM, N, p, q, Fp, beta_cocycle, E_MN, ELN_use)

    coords = coordinates(ELN_use, p+q, cocycle)

    if return_cocycle
        return (ELN_use, coords, cocycle)
    else
        return (ELN_use, coords)
    end
end

# =============================================================================
# Ext algebra: Ext^*(M,M) with cached Yoneda multiplication
# =============================================================================

"""
    ExtAlgebra(M::PModule{QQ}; maxdeg::Int=3) -> ExtAlgebra{QQ}
    ExtAlgebra(M::FringeModule{QQ}; maxdeg::Int=3) -> ExtAlgebra{QQ}

Construct the truncated graded Ext algebra Ext^*(M,M) up to degree `maxdeg`,
with multiplication given by the Yoneda product.

This wrapper is intentionally "mathematician-facing":

- It chooses (once) a projective resolution and Ext bases via `Ext(M,M; maxdeg=...)`.
- It exposes homogeneous elements as `ExtElement` objects.
- It supports `*` for Ext multiplication and caches the structure constants.

Caching model (key point):
For each bidegree (p,q) with p+q <= tmax, we cache a matrix

    MU[p,q] : Ext^p(M,M) x Ext^q(M,M) -> Ext^{p+q}(M,M)

in coordinate bases as a linear map on the Kronecker product coordinates.

Column convention:
If dim_p = dim Ext^p and dim_q = dim Ext^q, we index basis pairs (i,j) by

    col = (i-1)*dim_q + j

and we use Julia's `kron(x, y)` to build the vector of coefficients x_i*y_j
in the same ordering.  This makes multiplication a single matrix-vector product:

    coords(x * y) = MU[p,q] * kron(coords(x), coords(y)).

Truncation:
The product is only defined when deg(x) + deg(y) <= A.tmax.
"""
mutable struct ExtAlgebra{K}
    E::ExtSpaceProjective{K}
    mult_cache::Dict{Tuple{Int,Int}, Matrix{K}}
    unit_coords::Union{Nothing, Vector{K}}
    # Cached degree range (no Base.getproperty shims).
    tmin::Int
    tmax::Int
end

"""
    ExtElement(A::ExtAlgebra{QQ}, deg::Int, coords::Vector{QQ})

A homogeneous element of Ext^deg(M,M), expressed in the basis chosen by `A.E`.

This is deliberately lightweight: it is just (algebra handle, degree, coordinate vector).
Use:
- `element(A, deg, coords)` to construct,
- `basis(A, deg)` or `A[deg, i]` for basis elements,
- multiplication via `*`.
"""
struct ExtElement{K}
    A::ExtAlgebra{K}
    deg::Int
    coords::Vector{K}
end


"""
    ExtDoubleComplex(M, N; maxlen=nothing) -> ChainComplexes.DoubleComplex{QQ}

Build the bounded double complex C^{a,b} = Hom(F_a, E^b) where:
- F is an upset resolution of M,
- E is a downset resolution of N.

Tot(ExtDoubleComplex(M,N)) computes Ext^*(M,N).

If maxlen is provided, both resolutions are truncated at that length.
If maxlen is nothing, each resolution is computed until it terminates.
"""
function ExtDoubleComplex(M::PModule{QQ}, N::PModule{QQ}; maxlen::Union{Nothing,Int}=nothing)
    F, dF = IndicatorResolutions.upset_resolution(M; maxlen=maxlen)
    E, dE = IndicatorResolutions.downset_resolution(N; maxlen=maxlen)
    return ExtDoubleComplex(F, dF, E, dE)
end

"""
    ExtDoubleComplex(F, dF, E, dE) -> ChainComplexes.DoubleComplex{QQ}

Low-level constructor from precomputed resolutions.
"""
function ExtDoubleComplex(F, dF, E, dE)
    dims, dv, dh = HomExt.build_hom_bicomplex_data(F, dF, E, dE)
    A = length(F) - 1
    B = length(E) - 1
    return ChainComplexes.DoubleComplex{QQ}(0, A, 0, B, dims, dv, dh)
end

"""
    ExtSpectralSequence(M, N; first=:vertical, maxlen=nothing) -> ChainComplexes.SpectralSequence{QQ}

Compute the spectral sequence associated to the Ext bicomplex Hom(F_a, E^b).

- first=:vertical uses vertical cohomology first (E1^{a,b} = H^b of columns).
- first=:horizontal uses horizontal cohomology first.

The returned object includes E1, d1, E2, Einf (graded pieces), and dim H^*(Tot).
"""
function ExtSpectralSequence(M::PModule{QQ}, N::PModule{QQ};
                             first::Symbol=:vertical,
                             maxlen::Union{Nothing,Int}=nothing)
    DC = ExtDoubleComplex(M, N; maxlen=maxlen)
    return ChainComplexes.spectral_sequence(DC; first=first)
end

# ----------------------------------------------------------------------
# Tor bicomplex / spectral sequence helpers
# ----------------------------------------------------------------------

"""
    TorDoubleComplex(Rop, L; maxlen=nothing, maxlenR=nothing, maxlenL=nothing)

Build a double complex computing Tor_*(Rop,L) from:

- a projective resolution of Rop as a P^op-module (right module),
- a projective resolution of L as a P-module (left module).

Conventions and indexing:

The underlying (homological) bicomplex is
    C_{a,b} = P_a(Rop) otimes Q_b(L),  a,b >= 0,

with horizontal differential induced by d_P and vertical differential induced by d_Q.

We return it as a `ChainComplexes.DoubleComplex` in *cochain* bidegrees (A,B) = (-a,-b),
so that the total cohomology satisfies
    H^t(Tot(C)) = Tor_{-t}(Rop, L).

Sign convention:

`ChainComplexes.total_complex` forms the total differential as dv + dh, so we must build dv and dh
so that dv*dh + dh*dv = 0. We use the standard Koszul sign on the vertical differential:
    dv = (-1)^a * (id otimes d_Q).

Keyword arguments:
- maxlen: if set, both resolutions are built/padded to this length (override with maxlenR/maxlenL).
- maxlenR: length for the Rop resolution.
- maxlenL: length for the L resolution.
"""
function TorDoubleComplex(Rop::PModule{QQ}, L::PModule{QQ}; maxlen=nothing, maxlenR=nothing, maxlenL=nothing)
    # Choose resolution lengths.
    lenR = (maxlenR === nothing) ? (maxlen === nothing ? 3 : maxlen) : maxlenR
    lenL = (maxlenL === nothing) ? (maxlen === nothing ? 3 : maxlen) : maxlenL
    @assert lenR >= 0 && lenL >= 0

    # Build and pad both resolutions so the double complex is a full rectangle.
    resR = projective_resolution(Rop; maxlen=lenR)
    resL = projective_resolution(L; maxlen=lenL)
    _pad_projective_resolution!(resR, lenR)
    _pad_projective_resolution!(resL, lenL)

    # Cochain indices are (A,B)=(-a,-b).
    amin, amax = -lenR, 0
    bmin, bmax = -lenL, 0
    na = amax - amin + 1   # = lenR+1
    nb = bmax - bmin + 1   # = lenL+1

    # Precompute dimensions of each bidegree term.
    dims = zeros(Int64, na, nb)
    for ia in 1:na
        A = amin + (ia - 1)
        a = -A
        gens_a = resR.gens[a + 1]  # vertices in P labeling
        for ib in 1:nb
            B = bmin + (ib - 1)
            b = -B
            Qb = resL.Pmods[b + 1]
            d = Int64(0)
            for u in gens_a
                d += Qb.dims[u]
            end
            dims[ia, ib] = d
        end
    end

    # Offsets for a tensor term at fixed (a,b):
    # Term is direct_sum_{u in gens_a} Qb_u.
    function _tensor_offsets(gens_a::Vector{Int}, Qb::PModule{QQ})
        offs = zeros(Int64, length(gens_a) + 1)
        for i in 1:length(gens_a)
            u = gens_a[i]
            offs[i + 1] = offs[i] + Qb.dims[u]
        end
        return offs
    end

    # Vertical differentials: dv_{A,B} : C^{A,B} -> C^{A,B+1}
    # Corresponds to (-1)^a * (id otimes d_Q_b) in the original chain bicomplex.
    dv = Array{SparseMatrixCSC{QQ, Int64}, 2}(undef, na, nb - 1)
    for ia in 1:na
        A = amin + (ia - 1)
        a = -A
        gens_a = resR.gens[a + 1]
        sgn = isodd(a) ? QQ(-1) : QQ(1)
        for ib in 1:(nb - 1)
            B = bmin + (ib - 1)
            b = -B
            # d_Q_b : Q_b -> Q_{b-1}, stored at index b (since b>=1 here)
            dQ = resL.d_mor[b]
            Qb = resL.Pmods[b + 1]
            Qbm1 = resL.Pmods[b]  # b-1

            offs_dom = _tensor_offsets(gens_a, Qb)
            offs_cod = _tensor_offsets(gens_a, Qbm1)

            Itrip = Int[]; Jtrip = Int[]; Vtrip = QQ[]
            for (i,u) in enumerate(gens_a)
                _append_scaled_triplets!(Itrip, Jtrip, Vtrip, dQ.comps[u],
                                        offs_cod[i], offs_dom[i]; scale=sgn)
            end
            dv[ia, ib] = sparse(Itrip, Jtrip, Vtrip, offs_cod[end], offs_dom[end])
        end
    end

    # Horizontal differentials: dh_{A,B} : C^{A,B} -> C^{A+1,B}
    # Corresponds to (d_P_a otimes id) in the original chain bicomplex.
    dh = Array{SparseMatrixCSC{QQ, Int64}, 2}(undef, na - 1, nb)
    for ia in 1:(na - 1)
        A = amin + (ia - 1)
        a = -A
        gens_dom = resR.gens[a + 1]
        gens_cod = resR.gens[a]  # a-1
        dP = resR.d_mat[a]       # P_a -> P_{a-1}, rows=cod, cols=dom
        I, J, V = findnz(dP)
        for ib in 1:nb
            B = bmin + (ib - 1)
            b = -B
            Qb = resL.Pmods[b + 1]

            offs_dom = _tensor_offsets(gens_dom, Qb)
            offs_cod = _tensor_offsets(gens_cod, Qb)

            Itrip = Int[]; Jtrip = Int[]; Vtrip = QQ[]
            for (i,u) in enumerate(gens_a)
                _append_scaled_triplets!(Itrip, Jtrip, Vtrip, dQ.comps[u],
                                        offs_cod[i], offs_dom[i]; scale=sgn)
            end
            dv[ia, ib] = sparse(Itrip, Jtrip, Vtrip, offs_cod[end], offs_dom[end])
        end
    end

    return ChainComplexes.DoubleComplex(amin, amax, bmin, bmax, dims, dv, dh)
end

"""
    TorSpectralSequence(Rop, L; maxlen=nothing, maxlenR=nothing, maxlenL=nothing, first=:vertical)

Return a spectral sequence associated to `TorDoubleComplex(Rop,L)`.

This is a small wrapper around `ChainComplexes.SpectralSequence` that reindexes bidegrees
to a homological convention (a,b) >= 0.

Internals:
- Underlying SS is computed on cochain bidegrees (A,B) = (-a,-b).
- Public indexing uses (a,b) >= 0.

The wrapper extends the `ChainComplexes` SS API so you can call:
- `E_r(ss, r)` / `page(ss,r)` and index the returned page as `pg[a,b]`.
- `term(ss,r,(a,b))`, `differential(ss,r,(a,b))`, `dr_target`, `dr_source`.
- `edge_inclusion(ss,(a,b))`, `edge_projection(ss,(a,b))`.
- `convergence_report(ss)`.

Note: the abutment is total cohomology of the cochain total complex; interpret it as Tor via
    H^t(Tot) = Tor_{-t}.
"""
struct TorSpectralSequence{K}
    ss::ChainComplexes.SpectralSequence{K}
end

struct TorSpectralPage{K}
    pg::ChainComplexes.SpectralPage{K}
end

# Reindexing: term at (a,b) corresponds to underlying (-a,-b).
Base.getindex(P::TorSpectralPage, a::Int, b::Int) = P.pg[-a, -b]


# ---------------------------------------------------------------------------
# Term-level access for Tor spectral sequences (SubquotientData pages)
# ---------------------------------------------------------------------------

"""
    TorSpectralTermsPage

A thin wrapper around `ChainComplexes.SpectralTermsPage` that reindexes the
bidegrees so that `P[(a,b)]` corresponds to the underlying cohomological
bidegree `(-a,-b)`.

This mirrors `TorSpectralPage` (dimensions) but for actual `SubquotientData`
terms (object-level access).
"""
struct TorSpectralTermsPage <: AbstractMatrix{ChainComplexes.SubquotientData{QQ}}
    pg::ChainComplexes.SpectralTermsPage
end

Base.size(P::TorSpectralTermsPage) = size(P.pg)
Base.getindex(P::TorSpectralTermsPage, i::Int, j::Int) = P.pg[i, j]
Base.getindex(P::TorSpectralTermsPage, ab::Tuple{Int,Int}) = P.pg[(-ab[1], -ab[2])]

function ChainComplexes.E_r_terms(TSS::TorSpectralSequence, r::Union{Int,Symbol})
    return TorSpectralTermsPage(ChainComplexes.E_r_terms(TSS.ss, r))
end

ChainComplexes.E2_terms(TSS::TorSpectralSequence) = ChainComplexes.E_r_terms(TSS, 2)

# Convenience: allow `page_terms(TSS, r)` in Tor indexing convention.
ChainComplexes.page_terms(TSS::TorSpectralSequence, r::Union{Int,Symbol}) = ChainComplexes.E_r_terms(TSS, r)

function ChainComplexes.page_terms_dict(TSS::TorSpectralSequence, r::Union{Int,Symbol}; nonzero_only::Bool=true)
    ss = TSS.ss
    pd = ChainComplexes._ss_page_data(ss, r)
    out = Dict{Tuple{Int,Int}, ChainComplexes.SubquotientData{QQ}}()

    if nonzero_only
        for (A,B) in ss.support
            a, b = -A, -B
            sq = pd.spaces[A - ss.DC.amin + 1, B - ss.DC.bmin + 1]
            if sq.dimH != 0
                out[(a,b)] = sq
            end
        end
    else
        for A in ss.DC.amin:ss.DC.amax
            for B in ss.DC.bmin:ss.DC.bmax
                out[(-A, -B)] = pd.spaces[A - ss.DC.amin + 1, B - ss.DC.bmin + 1]
            end
        end
    end
    return out
end

function ChainComplexes.page_dims_dict(TSS::TorSpectralSequence, r::Union{Int,Symbol}; nonzero_only::Bool=true)
    ss = TSS.ss
    pd = ChainComplexes._ss_page_data(ss, r)
    out = Dict{Tuple{Int,Int}, Int}()

    if nonzero_only
        for (A,B) in ss.support
            a, b = -A, -B
            d = pd.dims[A - ss.DC.amin + 1, B - ss.DC.bmin + 1]
            if d != 0
                out[(a,b)] = d
            end
        end
    else
        for A in ss.DC.amin:ss.DC.amax
            for B in ss.DC.bmin:ss.DC.bmax
                out[(-A, -B)] = pd.dims[A - ss.DC.amin + 1, B - ss.DC.bmin + 1]
            end
        end
    end
    return out
end



# Extend the ChainComplexes spectral sequence API for the Tor wrapper.
function ChainComplexes.E_r(TSS::TorSpectralSequence{K}, r::Int) where {K}
    return TorSpectralPage{K}(ChainComplexes.E_r(TSS.ss, r))
end

ChainComplexes.page(TSS::TorSpectralSequence{K}, r::Int) where {K} = ChainComplexes.E_r(TSS, r)

function ChainComplexes.term(TSS::TorSpectralSequence, r::Int, ab::Tuple{Int, Int})
    a, b = ab
    return ChainComplexes.term(TSS.ss, r, (-a, -b))
end

function ChainComplexes.differential(TSS::TorSpectralSequence, r::Int, ab::Tuple{Int, Int})
    a, b = ab
    return ChainComplexes.differential(TSS.ss, r, (-a, -b))
end

function ChainComplexes.dr_target(TSS::TorSpectralSequence, r::Int, ab::Tuple{Int, Int})
    a, b = ab
    A, B = ChainComplexes.dr_target(TSS.ss, r, (-a, -b))
    return (-A, -B)
end

function ChainComplexes.dr_source(TSS::TorSpectralSequence, r::Int, ab::Tuple{Int, Int})
    a, b = ab
    A, B = ChainComplexes.dr_source(TSS.ss, r, (-a, -b))
    return (-A, -B)
end

ChainComplexes.convergence_report(TSS::TorSpectralSequence) = ChainComplexes.convergence_report(TSS.ss)

function ChainComplexes.edge_inclusion(TSS::TorSpectralSequence, ab::Tuple{Int, Int})
    a, b = ab
    return ChainComplexes.edge_inclusion(TSS.ss, (-a, -b))
end

function ChainComplexes.edge_projection(TSS::TorSpectralSequence, ab::Tuple{Int, Int})
    a, b = ab
    return ChainComplexes.edge_projection(TSS.ss, (-a, -b))
end

function TorSpectralSequence(Rop::PModule{QQ}, L::PModule{QQ}; maxlen=nothing, maxlenR=nothing, maxlenL=nothing, first=:vertical)
    DC = TorDoubleComplex(Rop, L; maxlen=maxlen, maxlenR=maxlenR, maxlenL=maxlenL)
    ss = ChainComplexes.spectral_sequence(DC; first=first)
    return TorSpectralSequence{QQ}(ss)
end


# ----------------------------
# Construction
# ----------------------------

function ExtAlgebra(M::PModule{QQ}; maxdeg::Int=3)
    E = Ext(M, M; maxdeg=maxdeg)
    return ExtAlgebra{QQ}(E, Dict{Tuple{Int,Int}, Matrix{QQ}}(), nothing, E.tmin, E.tmax)
end

function ExtAlgebra(M::FringeModule{QQ}; maxdeg::Int=3)
    return ExtAlgebra(pmodule_from_fringe(M); maxdeg=maxdeg)
end


## NOTE:
## ExtAlgebra stores `tmin`/`tmax` as explicit fields.
## This avoids Base.getproperty shims while keeping dot-access ergonomic.


# ----------------------------
# Basic queries and constructors for elements
# ----------------------------

"Dimension of Ext^deg(M,M) in the basis chosen by the underlying Ext space."
dim(A::ExtAlgebra{QQ}, deg::Int) = dim(A.E, deg)

"""
    element(A::ExtAlgebra{QQ}, deg::Int, coords::AbstractVector{QQ}) -> ExtElement{QQ}

Construct a homogeneous Ext element in degree `deg` with the given coordinate vector.
"""
function element(A::ExtAlgebra{QQ}, deg::Int, coords::AbstractVector{QQ})
    if deg < 0 || deg > A.tmax
        error("element: degree must satisfy 0 <= deg <= tmax.")
    end
    d = dim(A, deg)
    if length(coords) != d
        error("element: expected coordinate vector of length $d in degree $deg, got length $(length(coords)).")
    end
    return ExtElement{QQ}(A, deg, Vector{QQ}(coords))
end


"""
    basis(A::ExtAlgebra{QQ}, deg::Int) -> Vector{ExtElement{QQ}}

Return the standard coordinate basis of Ext^deg(M,M) as ExtElement objects.
"""
function basis(A::ExtAlgebra{QQ}, deg::Int)
    d = dim(A, deg)
    out = Vector{ExtElement{QQ}}(undef, d)
    for i in 1:d
        c = zeros(QQ, d)
        c[i] = one(QQ)
        out[i] = ExtElement{QQ}(A, deg, c)
    end
    return out
end


"""
    A[deg, i] -> ExtElement

Indexing convenience: the i-th basis element in degree `deg`.
"""
function Base.getindex(A::ExtAlgebra{QQ}, deg::Int, i::Int)
    d = dim(A, deg)
    if i < 1 || i > d
        error("ExtAlgebra getindex: basis index i=$i out of range 1:$d in degree $deg.")
    end
    c = zeros(QQ, d)
    c[i] = one(QQ)
    return ExtElement{QQ}(A, deg, c)
end


"Return the coordinate vector of an ExtElement."
coordinates(x::ExtElement{QQ}) = x.coords

"""
    representative(x::ExtElement{QQ}) -> Vector{QQ}

Return a cocycle representative (in the internal cochain model) for the Ext class.
This is useful for debugging and for users who want explicit representatives.
"""
representative(x::ExtElement{QQ}) = representative(x.A.E, x.deg, x.coords)


# ----------------------------
# Linear structure on ExtElement
# ----------------------------

function _assert_same_algebra(x::ExtElement{QQ}, y::ExtElement{QQ}, ctx::String)
    if x.A !== y.A
        error("$ctx: elements live in different ExtAlgebra objects.")
    end
    if x.deg != y.deg
        error("$ctx: degrees differ (deg(x)=$(x.deg), deg(y)=$(y.deg)).")
    end
    return nothing
end

Base.:+(x::ExtElement{QQ}, y::ExtElement{QQ}) = (_assert_same_algebra(x, y, "ExtElement +");
                                                ExtElement{QQ}(x.A, x.deg, x.coords + y.coords))

Base.:-(x::ExtElement{QQ}, y::ExtElement{QQ}) = (_assert_same_algebra(x, y, "ExtElement -");
                                                ExtElement{QQ}(x.A, x.deg, x.coords - y.coords))

Base.:-(x::ExtElement{QQ}) = ExtElement{QQ}(x.A, x.deg, -x.coords)

Base.:*(c::QQ, x::ExtElement{QQ}) = ExtElement{QQ}(x.A, x.deg, c .* x.coords)
Base.:*(x::ExtElement{QQ}, c::QQ) = c * x

Base.:*(c::Integer, x::ExtElement{QQ}) = QQ(c) * x
Base.:*(x::ExtElement{QQ}, c::Integer) = QQ(c) * x

Base.iszero(x::ExtElement{QQ}) = all(x.coords .== 0)


# ----------------------------
# Unit element in Ext^0(M,M)
# ----------------------------

# Internal: encode a morphism P_t -> N as a cochain vector in C^t = Hom(P_t, N)
# using the generator ordering stored in the projective resolution.
function _cochain_vector_from_morphism(E::ExtSpaceProjective{QQ}, t::Int, f::PMorphism{QQ})
    if t < 0 || t > E.tmax
        error("_cochain_vector_from_morphism: degree out of range.")
    end
    if f.dom !== E.res.Pmods[t+1] || f.cod !== E.N
        error("_cochain_vector_from_morphism: expected a morphism P_t -> N for the given Ext space.")
    end

    bases = E.res.gens[t+1]
    offs  = E.offsets[t+1]
    out = zeros(QQ, offs[end])

    # Which summands of P_t are active at each vertex?
    active = _active_upset_indices(E.res.M.Q, bases)

    # For each summand i with base vertex u = bases[i],
    # locate the column position of i inside the fiber (P_t)_u,
    # then read off the image of that generator under f.
    for i in 1:length(bases)
        u = bases[i]
        du = E.N.dims[u]
        if du == 0
            continue
        end

        act_u = active[u]
        pos = searchsortedfirst(act_u, i)
        if pos > length(act_u) || act_u[pos] != i
            error("_cochain_vector_from_morphism: internal mismatch locating summand $i at vertex $u.")
        end

        out[(offs[i]+1):offs[i+1]] = f.comps[u][:, pos]
    end

    return out
end


"""
    unit(A::ExtAlgebra{QQ}) -> ExtElement{QQ}

Return the multiplicative identity in Ext^0(M,M).

Mathematically, Ext^0(M,M) = Hom(M,M), and the unit is id_M.
In the projective-resolution model
    ... -> P_1 -> P_0 -> M -> 0,
the inclusion Hom(M,M) -> Hom(P_0,M) sends id_M to the augmentation map P_0 -> M.
That augmentation is a cocycle in C^0 and represents the unit class in H^0.
"""
function unit(A::ExtAlgebra{QQ})
    if A.unit_coords === nothing
        if dim(A, 0) == 0
            # Zero module edge case: Ext^0(0,0) is 0 as a vector space.
            A.unit_coords = zeros(QQ, 0)
        else
            cocycle = _cochain_vector_from_morphism(A.E, 0, A.E.res.aug)
            A.unit_coords = coordinates(A.E, 0, cocycle)
        end
    end
    return ExtElement{QQ}(A, 0, copy(A.unit_coords))
end

Base.one(A::ExtAlgebra{QQ}) = unit(A)


# ----------------------------
# Cached multiplication: ExtElement * ExtElement
# ----------------------------

# Ensure the multiplication matrix MU[p,q] is present in the cache.
function _ensure_mult_cache!(A::ExtAlgebra{QQ}, p::Int, q::Int)
    key = (p, q)
    if haskey(A.mult_cache, key)
        return A.mult_cache[key]
    end

    if p < 0 || q < 0
        error("_ensure_mult_cache!: degrees must be nonnegative.")
    end
    if p + q > A.tmax
        error("_ensure_mult_cache!: requested product degree p+q=$(p+q) exceeds truncation tmax=$(A.tmax).")
    end

    dp = dim(A, p)
    dq = dim(A, q)
    dr = dim(A, p + q)

    MU = zeros(QQ, dr, dp * dq)

    # Cache even the trivial cases so repeated calls are O(1).
    if dp == 0 || dq == 0 || dr == 0
        A.mult_cache[key] = MU
        return MU
    end

    # Precompute all products of basis elements e_i in Ext^p and e_j in Ext^q.
    # Each product is computed by the trusted "mathematical core" `yoneda_product`,
    # then stored as a column of MU in the kron(x,y) ordering.
    ei = zeros(QQ, dp)
    ej = zeros(QQ, dq)

    for i in 1:dp
        fill!(ei, zero(QQ))
        ei[i] = one(QQ)
        for j in 1:dq
            fill!(ej, zero(QQ))
            ej[j] = one(QQ)

            # Multiply e_i (degree p) by e_j (degree q) in Ext(M,M).
            _, coords = yoneda_product(A.E, p, ei, A.E, q, ej; ELN=A.E)

            MU[:, (i - 1) * dq + j] = coords
        end
    end

    A.mult_cache[key] = MU
    return MU
end


"""
    multiply(A::ExtAlgebra{QQ}, p::Int, x::AbstractVector{QQ}, q::Int, y::AbstractVector{QQ}) -> Vector{QQ}

Multiply two homogeneous elements given by coordinate vectors x in Ext^p and y in Ext^q.
Returns the coordinate vector in Ext^{p+q}.
"""
function multiply(A::ExtAlgebra{QQ}, p::Int, x::AbstractVector{QQ}, q::Int, y::AbstractVector{QQ})
    dp = dim(A, p)
    dq = dim(A, q)
    if length(x) != dp
        error("multiply: left coordinate vector has length $(length(x)) but dim Ext^$p = $dp.")
    end
    if length(y) != dq
        error("multiply: right coordinate vector has length $(length(y)) but dim Ext^$q = $dq.")
    end

    MU = _ensure_mult_cache!(A, p, q)

    # kron(x,y) uses exactly the ordering we used for MU columns.
    v = kron(Vector{QQ}(x), Vector{QQ}(y))
    out = MU * v
    return Vector{QQ}(out)
end


"""
    precompute!(A::ExtAlgebra{QQ}) -> ExtAlgebra{QQ}

Eagerly compute and cache all multiplication matrices MU[p,q] with p+q <= A.tmax.
This is optional. Most users will rely on lazy caching via `*`.
"""
function precompute!(A::ExtAlgebra{QQ})
    for p in 0:A.tmax
        for q in 0:(A.tmax - p)
            _ensure_mult_cache!(A, p, q)
        end
    end
    return A
end


# The user-facing multiplication on homogeneous Ext elements.
function Base.:*(x::ExtElement{QQ}, y::ExtElement{QQ})
    if x.A !== y.A
        error("ExtElement *: elements live in different ExtAlgebra objects.")
    end
    A = x.A
    p = x.deg
    q = y.deg
    if p + q > A.tmax
        error("ExtElement *: degree p+q=$(p+q) exceeds truncation tmax=$(A.tmax).")
    end
    coords = multiply(A, p, x.coords, q, y.coords)
    return ExtElement{QQ}(A, p + q, coords)
end


# =============================================================================
# Injective resolutions
# =============================================================================

"""
Injective resolution of a module N:
    0 -> N -> E^0 -> E^1 -> ...

The field `gens[b+1]` stores the base vertices of the indecomposable injective
summands k[Dn(v)] appearing in E^b, with repetition.

This makes it possible to extract Bass-type multiplicity data (injective summands
by vertex and cohomological degree) in a canonical, user-facing way.
"""
struct InjectiveResolution{K}
    N::PModule{K}
    Emods::Vector{PModule{K}}         # E^0, E^1, ...
    gens::Vector{Vector{Int}}         # base vertices per injective summand in each E^b
    d_mor::Vector{PMorphism{K}}       # E^b -> E^{b+1}
    iota0::PMorphism{K}               # N -> E^0
end


"""
    injective_resolution(N::PModule{QQ}; maxlen::Int=3) -> InjectiveResolution{QQ}

Build an injective resolution
    0 -> N -> E^0 -> E^1 -> ... -> E^maxlen

Implementation notes:
- `injective_hull` is computed degreewise via `_injective_hull`.
- The differentials are obtained by extending the map N^b -> E^b to E^b -> E^{b+1}.

The resulting resolution is intended to be suitable for Ext computations and
Bass-number extraction (multiplicity of injective indecomposables).
"""
function injective_resolution(N::PModule{QQ}; maxlen::Int=3)
    E0, iota0, gens0 = _injective_hull(N)
    Emods = [E0]
    gens  = [_flatten_gens_at(gens0)]
    d_mor = PMorphism{QQ}[]

    C0, pi0 = _cokernel_module(iota0)
    prevC  = C0
    prevPi = pi0

    for step in 1:maxlen
        En, iotan, gensn = _injective_hull(prevC)
        push!(Emods, En)
        push!(gens, _flatten_gens_at(gensn))

        dn = compose(iotan, prevPi)   # E^{step-1} -> E^{step}
        push!(d_mor, dn)

        Cn, pin = _cokernel_module(iotan)
        prevC  = Cn
        prevPi = pin
    end

    return InjectiveResolution{QQ}(N, Emods, gens, d_mor, iota0)
end

# Convenience overload for fringe modules.
# This mirrors the existing projective_resolution(::FringeModule) overload and
# matters because encoding layers (Zn and PL/Rn) naturally produce FringeModule data.
function injective_resolution(N::FringeModule{QQ}; maxlen::Int=3)
    return injective_resolution(pmodule_from_fringe(N); maxlen=maxlen)
end



"""
    bass(res::InjectiveResolution{QQ}) -> Dict{Tuple{Int,Int},Int}

Bass numbers for an injective resolution.

Interpretation:
- `res.Emods[b+1]` is the injective in cohomological degree b (i.e. E^b).
- Each E^b splits as a direct sum of indecomposable injectives k[Dn(v)].
- `res.gens[b+1]` stores the base vertex v for each summand, with repetition.

Output convention:
- Key `(b, v)` means "cohomological degree b, vertex v".
- Value is multiplicity of k[Dn(v)] in E^b.
"""
function bass(res::InjectiveResolution{QQ})
    out = Dict{Tuple{Int,Int},Int}()
    L = length(res.Emods) - 1
    for b in 0:L
        for v in res.gens[b+1]
            key = (b, v)
            out[key] = get(out, key, 0) + 1
        end
    end
    return out
end


"""
    bass_table(res::InjectiveResolution{QQ}) -> Matrix{Int}

Dense Bass table, analogous to `betti_table`:

- Rows are cohomological degrees b = 0,1,2,...
- Columns are vertices v = 1,...,Q.n
- Entry B[b+1, v] is the multiplicity of k[Dn(v)] in E^b.
"""
function bass_table(res::InjectiveResolution{QQ}; pad_to::Union{Nothing,Int}=nothing)
    Q = res.N.Q
    L = length(res.Emods) - 1
    B = zeros(Int, L + 1, Q.n)
    for b in 0:L
        for v in res.gens[b + 1]
            B[b + 1, v] += 1
        end
    end

    if pad_to === nothing
        return _trim_trailing_zero_rows(B)
    else
        pad_to >= 0 || error("bass_table: pad_to must be >= 0")
        return _pad_or_truncate_rows(B, pad_to + 1)
    end
end


"""
    bass(N::PModule{QQ}; maxlen::Int=3) -> Dict{Tuple{Int,Int},Int}

Convenience wrapper:
- build `injective_resolution(N; maxlen=maxlen)`,
- return its Bass numbers.
"""
function bass(N::PModule{QQ}; maxlen::Int=3)
    return bass(injective_resolution(N; maxlen=maxlen))
end


# ----------------------------
# Minimality diagnostics for injective resolutions
# ----------------------------

"""
    InjectiveMinimalityReport

Returned by `minimality_report(res::InjectiveResolution{QQ})`.

Fields mirror `ProjectiveMinimalityReport`, but for injective resolutions:

- `hull_ok` compares multiplicities in E^0 against a freshly computed injective hull.
- `diagonal_violations` records nonzero coefficients k[Dn(v)] -> k[Dn(v)] in the
  differentials E^{b-1} -> E^b. Such coefficients split off contractible summands
  and are forbidden in a minimal injective resolution.
"""
struct InjectiveMinimalityReport
    minimal::Bool
    hull_ok::Bool
    hull_expected::Vector{Int}
    hull_actual::Vector{Int}
    diagonal_violations::Vector{Tuple{Int,Int,Int,Int,QQ}}
end


"""
    minimality_report(res::InjectiveResolution{QQ}; check_hull=true) -> InjectiveMinimalityReport

Certify minimality of an injective resolution in the standard finite-dimensional
algebra sense.

Checks performed:

1. (Optional) `check_hull`:
   Verify that N -> E^0 is an injective hull by comparing multiplicities of principal
   downsets against a freshly computed `_injective_hull(N)`.

2. Verify the "no units on diagonal" condition:
   For every differential d^b : E^{b-1} -> E^b, there is no nonzero coefficient
   k[Dn(v)] -> k[Dn(v)].
"""
function minimality_report(res::InjectiveResolution{QQ}; check_hull::Bool=true)
    Q = res.N.Q
    n = Q.n

    hull_actual = _vertex_counts(res.gens[1], n)
    hull_expected = copy(hull_actual)
    hull_ok = true

    if check_hull
        _, _, gens_at = _injective_hull(res.N)
        hull_expected = _vertex_counts(_flatten_gens_at(gens_at), n)
        hull_ok = (hull_expected == hull_actual)
    end

    violations = Tuple{Int,Int,Int,Int,QQ}[]
    for b in 1:length(res.d_mor)
        dom = res.gens[b]     # bases in E^{b-1}
        cod = res.gens[b+1]   # bases in E^b

        D = _coeff_matrix_downsets(Q, dom, cod, res.d_mor[b])

        for col in 1:size(D, 2)
            for ptr in D.colptr[col]:(D.colptr[col+1] - 1)
                row = D.rowval[ptr]
                val = D.nzval[ptr]
                if !iszero(val) && (cod[row] == dom[col])
                    v = dom[col]
                    push!(violations, (b, v, row, col, val))
                end
            end
        end
    end

    minimal = hull_ok && isempty(violations)
    return InjectiveMinimalityReport(minimal, hull_ok, hull_expected, hull_actual, violations)
end


"""
    is_minimal(res::InjectiveResolution{QQ}; check_hull=true) -> Bool
"""
function is_minimal(res::InjectiveResolution{QQ}; check_hull::Bool=true)
    return minimality_report(res; check_hull=check_hull).minimal
end


"""
    assert_minimal(res::InjectiveResolution{QQ}; check_hull=true) -> true

Throw a descriptive error if the injective resolution fails minimality checks.
"""
function assert_minimal(res::InjectiveResolution{QQ}; check_hull::Bool=true)
    R = minimality_report(res; check_hull=check_hull)
    if R.minimal
        return true
    end

    if !R.hull_ok
        error("Injective resolution is not minimal: N -> E0 is not an injective hull. " *
              "Expected hull multiplicities = $(R.hull_expected), got = $(R.hull_actual).")
    end

    if !isempty(R.diagonal_violations)
        (b, v, row, col, val) = R.diagonal_violations[1]
        error("Injective resolution is not minimal: differential d^$b has a nonzero coefficient " *
              "val = $val from k[Dn($v)] in E_$(b-1) (column $col) to k[Dn($v)] in E_$b (row $row).")
    end

    error("Injective resolution failed minimality checks for an unknown reason.")
end


"""
    minimal_injective_resolution(N::PModule{QQ}; maxlen=3, check=true) -> InjectiveResolution{QQ}

Build an injective resolution via successive injective hulls (the standard minimal
construction in the finite-dimensional algebra setting).

If `check=true`, call `assert_minimal` before returning.
"""
function minimal_injective_resolution(N::PModule{QQ}; maxlen::Int=3, check::Bool=true)
    res = injective_resolution(N; maxlen=maxlen)
    if check
        assert_minimal(res; check_hull=true)
    end
    return res
end


"""
    minimal_bass(N::PModule{QQ}; maxlen=3, check=true) -> Dict{Tuple{Int,Int},Int}

Compute Bass multiplicities from a certified minimal injective resolution.
"""
function minimal_bass(N::PModule{QQ}; maxlen::Int=3, check::Bool=true)
    res = minimal_injective_resolution(N; maxlen=maxlen, check=check)
    return bass(res)
end


# ---------------------------------------------------------------------------
# Injective chain-map lifting
# ---------------------------------------------------------------------------

# Injective resolutions constructed by `injective_resolution` are explicit direct sums
# of principal downset modules. Each summand is determined by a base vertex b, and the
# summand contributes a 1-dimensional fiber at u iff u <= b.
#
# A morphism between such sums is determined by coefficients between summands. A map
# Dn(v) -> Dn(w) can be nonzero only when w <= v. This restriction is essential for the
# resulting fiberwise matrices to commute with structure maps (edges).
#
# The helpers below solve Phi o f = g for Phi between downset sums, then package Phi as a
# genuine PMorphism. The public API is `lift_injective_chainmap`.

# Build an index map for unknown coefficients, restricting to valid downset maps:
# coefficient (r,c) corresponds to Dn(dom_bases[c]) -> Dn(cod_bases[r]) and is allowed
# only if cod_bases[r] <= dom_bases[c].
function _downset_hom_varidx(Q, dom_bases::Vector{Int}, cod_bases::Vector{Int})
    n_dom = length(dom_bases)
    n_cod = length(cod_bases)
    var_idx = zeros(Int, n_cod, n_dom)
    idx = 0
    for r in 1:n_cod
        br = cod_bases[r]
        for c in 1:n_dom
            bc = dom_bases[c]
            if Q.leq[br, bc]   # br <= bc
                idx += 1
                var_idx[r, c] = idx
            end
        end
    end
    return var_idx, idx
end

# Turn a global coefficient matrix into an actual PMorphism by restricting to active
# summands at each vertex (in the standard injective-hull basis order).
function _pmorphism_from_downset_coeff(E::PModule{QQ}, Ep::PModule{QQ},
                                       act_dom::Vector{Vector{Int}},
                                       act_cod::Vector{Vector{Int}},
                                       C::Matrix{QQ})
    Q = E.Q
    @assert _same_poset(Q, Ep.Q)
    comps = Vector{Matrix{QQ}}(undef, Q.n)
    for u in 1:Q.n
        rows = act_cod[u]
        cols = act_dom[u]
        comps[u] = C[rows, cols]
    end
    return PMorphism{QQ}(E, Ep, comps)
end

# Solve for C (coefficients) such that Phi(C) o f = g, where Phi(C) is a morphism between
# direct sums of principal downsets described by dom_bases/cod_bases.
#
# This is used internally by lift_injective_chainmap, which threads the downset-basis
# metadata explicitly (dom_bases, cod_bases, act_dom, act_cod) to avoid recomputation and
# to keep the solver honest.
"""
    _solve_downset_postcompose_coeff(f, g, dom_bases, cod_bases, act_dom, act_cod; check=true) -> Matrix{QQ}

Solve for a coefficient matrix `C` such that, fiberwise for every vertex `u`,
`C[act_cod[u], act_dom[u]] * f(u) = g(u)`.

This is implemented as a streaming sparse augmented elimination:
each scalar equation is assembled as a sparse row in the unknown coefficients and pushed into
`ExactQQ.SparseRREFAugmented`. A deterministic particular solution is returned (free variables set to 0).

If the system is inconsistent, this throws an error (as that indicates a bug or incompatible truncation).
"""
function _solve_downset_postcompose_coeff(
    f::PMorphism{QQ},
    g::PMorphism{QQ},
    dom_bases::Vector{Int},
    cod_bases::Vector{Int},
    act_dom::Vector{Vector{Int}},
    act_cod::Vector{Vector{Int}};
    check::Bool = true,
)
    Q = f.dom.Q

    if check
        @assert g.dom === f.dom
        @assert f.cod.Q === Q
        @assert g.cod.Q === Q
        @assert length(act_dom) == Q.n
        @assert length(act_cod) == Q.n
        for u in 1:Q.n
            @assert f.cod.dims[u] == length(act_dom[u])
            @assert g.cod.dims[u] == length(act_cod[u])
            @assert size(f.comps[u], 1) == f.cod.dims[u]
            @assert size(g.comps[u], 1) == g.cod.dims[u]
            @assert size(f.comps[u], 2) == f.dom.dims[u]
            @assert size(g.comps[u], 2) == g.dom.dims[u]
            @assert size(f.comps[u], 2) == size(g.comps[u], 2)
        end
    end

    # Unknowns are the downset-hom coefficients C[r,c] (subject to cod_bases[r] <= dom_bases[c]).
    var_idx, nunk = _downset_hom_varidx(Q, dom_bases, cod_bases)

    R = SparseRREFAugmented{QQ}(nunk, 1)
    row = SparseRow{QQ}()
    rhs = Vector{QQ}(undef, 1)

    # Stream constraints:
    # For each u, equation is: C_u * F_u = G_u with C_u = C[rows_u, cols_u].
    for u in 1:Q.n
        rows_u = act_cod[u]
        cols_u = act_dom[u]
        Fu = f.comps[u]
        Gu = g.comps[u]
        dX = size(Fu, 2)

        # Row assembly uses cols_u in increasing summand-index order, and var_idx[r, c] is increasing
        # in c for fixed r. Thus row.idx is already sorted (no normalization needed).
        for rpos in 1:length(rows_u)
            r = rows_u[rpos]
            for j in 1:dX
                empty!(row.idx)
                empty!(row.val)

                @inbounds for cpos in 1:length(cols_u)
                    c = cols_u[cpos]
                    vidx = var_idx[r, c]
                    vidx == 0 && continue
                    a = Fu[cpos, j]
                    iszero(a) && continue
                    push!(row.idx, vidx)
                    push!(row.val, a)
                end

                rhs[1] = Gu[rpos, j]
                status = _sparse_rref_push_augmented!(R, row, rhs)
                if status === :inconsistent
                    error("Inconsistent system in injective lift. This indicates a bug or incompatible resolution truncation.")
                end
            end
        end
    end

    # Read back a deterministic particular solution: pivot vars = pivot RHS, free vars = 0.
    x = zeros(QQ, nunk)
    @inbounds for pos in 1:length(R.rref.pivot_cols)
        pcol = R.rref.pivot_cols[pos]
        x[pcol] = R.pivot_rhs[pos][1]
    end

    # Assemble the full coefficient matrix C (zeros where no downset-hom variable exists).
    n_dom = length(dom_bases)
    n_cod = length(cod_bases)
    C = zeros(QQ, n_cod, n_dom)
    for r in 1:n_cod, c in 1:n_dom
        vidx = var_idx[r, c]
        vidx == 0 && continue
        C[r, c] = x[vidx]
    end
    return C
end



"""
    lift_injective_chainmap(g, res_dom, res_cod; upto=nothing, check=true)

Lift a module morphism `g : N -> Np` to a cochain map between chosen injective resolutions.

If

    res_dom : 0 -> N  -> E0 -> E1 -> E2 -> ...
    res_cod : 0 -> Np -> F0 -> F1 -> F2 -> ...

then the result is a vector `phis` with

    phis[k+1] : E_k -> F_k    (k = 0..upto)

satisfying:

    phis[1] o res_dom.iota0 = res_cod.iota0 o g
    phis[k+1] o res_dom.d_mor[k] = res_cod.d_mor[k] o phis[k]   for k = 1..upto

The lift exists because the `F_k` are injective. The choice is deterministic: linear
systems are solved via `solve_particularQQ` with free variables set to 0.

Speed tip: reuse `res_dom` and `res_cod` when lifting many maps between the same modules.
"""
function lift_injective_chainmap(g::PMorphism{QQ},
                                 res_dom::InjectiveResolution{QQ},
                                 res_cod::InjectiveResolution{QQ};
                                 upto::Union{Nothing, Int} = nothing,
                                 check::Bool = true)
    if check
        @assert g.dom === res_dom.N
        @assert g.cod === res_cod.N
    end
    Q = g.dom.Q

    Ldom = length(res_dom.d_mor)
    Lcod = length(res_cod.d_mor)
    L = (upto === nothing) ? min(Ldom, Lcod) : upto

    if check
        @assert 0 <= L
        @assert L <= Ldom
        @assert L <= Lcod
    end

    phis = Vector{PMorphism{QQ}}(undef, L + 1)

    # Degree 0: solve phi0 o iota0_dom = iota0_cod o g.
    dom_bases0 = res_dom.gens[1]
    cod_bases0 = res_cod.gens[1]
    act_dom0 = _active_downset_indices(Q, dom_bases0)
    act_cod0 = _active_downset_indices(Q, cod_bases0)

    rhs0_comps = [res_cod.iota0.comps[u] * g.comps[u] for u in 1:Q.n]
    rhs0 = PMorphism{QQ}(g.dom, res_cod.Emods[1], rhs0_comps)

    C0 = _solve_downset_postcompose_coeff(res_dom.iota0, rhs0,
                                         dom_bases0, cod_bases0,
                                         act_dom0, act_cod0;
                                         check = check)
    phis[1] = _pmorphism_from_downset_coeff(res_dom.Emods[1], res_cod.Emods[1],
                                           act_dom0, act_cod0, C0)

    # Higher degrees: solve phi^k o d_dom^{k-1} = d_cod^{k-1} o phi^{k-1}.
    for k in 1:L
        dom_bases = res_dom.gens[k+1]
        cod_bases = res_cod.gens[k+1]
        act_dom = _active_downset_indices(Q, dom_bases)
        act_cod = _active_downset_indices(Q, cod_bases)

        rhs_comps = [res_cod.d_mor[k].comps[u] * phis[k].comps[u] for u in 1:Q.n]
        rhs = PMorphism{QQ}(res_dom.Emods[k], res_cod.Emods[k+1], rhs_comps)

        Ck = _solve_downset_postcompose_coeff(res_dom.d_mor[k], rhs,
                                             dom_bases, cod_bases,
                                             act_dom, act_cod;
                                             check = check)
        phis[k+1] = _pmorphism_from_downset_coeff(res_dom.Emods[k+1], res_cod.Emods[k+1],
                                                 act_dom, act_cod, Ck)
    end

    return phis
end

"""
    lift_injective_chainmap(g; maxlen=3, check=true)

Convenience wrapper:
- builds injective resolutions of `g.dom` and `g.cod` up to `maxlen`
- lifts `g` up to degree `maxlen`

Returns a named tuple `(res_dom, res_cod, phis)`.
"""
function lift_injective_chainmap(g::PMorphism{QQ}; maxlen::Int = 3, check::Bool = true)
    res_dom = injective_resolution(g.dom; maxlen=maxlen)
    res_cod = injective_resolution(g.cod; maxlen=maxlen)
    phis = lift_injective_chainmap(g, res_dom, res_cod; upto=maxlen, check=check)
    return (res_dom=res_dom, res_cod=res_cod, phis=phis)
end



struct ExtSpaceInjective{K}
    M::PModule{K}
    res::InjectiveResolution{K}
    homs::Vector{HomSpace{K}}                     # Hom(M, E^b)
    complex::ChainComplexes.CochainComplex{K}
    cohom::Vector{ChainComplexes.CohomologyData{K}}
    # Cached degree range (no Base.getproperty shims).
    tmin::Int
    tmax::Int
end

# NOTE:
# The degree range is also available as E.complex.tmin / E.complex.tmax.
# We cache tmin/tmax as explicit fields to keep dot-access ergonomic without
# relying on Base.getproperty overrides.

"""
    representative(E::ExtSpaceInjective, t::Int, coords::AbstractVector{QQ}) -> Vector{QQ}

Same as the projective-model method, but for an Ext space computed via an injective
resolution of the second argument.

Returns a cocycle in the cochain space Hom(M, E^t) (assembled over all degrees).
"""
function representative(E::ExtSpaceInjective{QQ}, t::Int, coords::AbstractVector{QQ})
    if t < 0 || t > E.tmax
        error("representative: degree t must satisfy 0 <= t <= tmax.")
    end
    Hrep = E.cohom[t+1].Hrep
    if length(coords) != size(Hrep, 2)
        error("representative: coordinate vector has length $(length(coords)), expected $(size(Hrep,2)).")
    end
    v = Hrep * reshape(coords, :, 1)
    return vec(v)
end

# -----------------------------------------------------------------------------
# Basic queries for ExtSpaceInjective (parity with ExtSpaceProjective)
# -----------------------------------------------------------------------------

function dim(E::ExtSpaceInjective, t::Int)
    if t < E.tmin || t > E.tmax
        return 0
    end
    return E.cohom[t + 1].dimH
end

function cycles(E::ExtSpaceInjective, t::Int)
    return E.cohom[t + 1].K
end

function boundaries(E::ExtSpaceInjective, t::Int)
    return E.cohom[t + 1].B
end

function representative(E::ExtSpaceInjective, t::Int, i::Int)
    Hrep = E.cohom[t + 1].Hrep
    return Hrep[:, i]
end

function basis(E::ExtSpaceInjective, t::Int)
    Hrep = E.cohom[t + 1].Hrep
    out = Vector{Vector{QQ}}(undef, size(Hrep, 2))
    for i in 1:size(Hrep, 2)
        out[i] = Vector{QQ}(Hrep[:, i])
    end
    return out
end

# Reduce a cocycle in C^t = Hom(M, E^t) to Ext-coordinates in the chosen basis.
function coordinates(E::ExtSpaceInjective, t::Int, cocycle::AbstractVector{QQ})
    data = E.cohom[t + 1]
    return ChainComplexes.cohomology_coordinates(data, cocycle)[:, 1]
end


function ExtInjective(M::PModule{QQ}, N::PModule{QQ}; maxdeg::Int=3)
    resN = injective_resolution(N; maxlen=maxdeg)
    return ExtInjective(M, resN)
end

function ExtInjective(M::PModule{QQ}, resN::InjectiveResolution{QQ})
    L = length(resN.Emods) - 1
    homs = HomSpace{QQ}[]
    dims = Int[]
    for b in 0:L
        Hb = Hom(M, resN.Emods[b+1])
        push!(homs, Hb)
        push!(dims, size(Hb.basis_matrix, 2))
    end

    # differential on Hom(M, E^b) is postcomposition with d^b : E^b -> E^{b+1}
    dC = Vector{SparseMatrixCSC{QQ, Int}}()
    for b in 0:(L-1)
        Hb = homs[b+1]
        Hb1 = homs[b+2]
        db = resN.d_mor[b+1]

        # build matrix dim(Hb1) x dim(Hb)
        # Assemble the differential as a SparseMatrixCSC directly (column-by-column).
        colptr = Vector{Int}(undef, dimHb + 1)
        colptr[1] = 1
        rowval = Int[]
        nzval  = QQ[]

        for j in 1:dimHb
            fj = Hb.basis_elements[j]
            fimg = compose(db, fj)
            vimg = _morphism_to_vector(fimg, verts, v_offsets)

            coeffs = solve_fullcolumnQQ(Hb1.basis_matrix, vimg)
            if !isempty(coeffs)
                # coeffs is (dimHb1 x 1)
                @inbounds for i in 1:dimHb1
                    cij = coeffs[i]   # linear indexing = (i,1)
                    if !iszero(cij)
                        push!(rowval, i)
                        push!(nzval, cij)
                    end
                end
            end

            colptr[j + 1] = length(rowval) + 1
        end

        push!(dC, SparseMatrixCSC(dimHb1, dimHb, colptr, rowval, nzval))
    end

    C = ChainComplexes.CochainComplex{QQ}(0, L, dims, dC, [Any[] for _ in 0:L])
    cohom = [ChainComplexes.cohomology_data(C, t) for t in 0:L]
    return ExtSpaceInjective{QQ}(M, resN, homs, C, cohom, C.tmin, C.tmax)
end

# Contravariant map in first argument: f : M -> Mp induces Ext^t(Mp,N) -> Ext^t(M,N)
function ext_map_first(EMN::ExtSpaceInjective{QQ}, EMPN::ExtSpaceInjective{QQ}, f::PMorphism{QQ}; t::Int)
    # map on cochains at degree t: Hom(Mp, E^t) -> Hom(M, E^t), g |-> g circ f
    Hsrc = EMPN.homs[t+1]
    Htgt = EMN.homs[t+1]

    F = zeros(QQ, size(Htgt.basis_matrix, 2), size(Hsrc.basis_matrix, 2))
    for j in 1:size(Hsrc.basis_matrix, 2)
        gj = Hsrc.basis[j]
        img = compose(gj, f)
        vimg = _morphism_to_vector(img, Htgt.offsets)
        coeffs = solve_fullcolumnQQ(Htgt.basis_matrix, vimg)
        F[:, j] = coeffs[:, 1]
    end

    return ChainComplexes.induced_map_on_cohomology(EMPN.cohom[t+1], EMN.cohom[t+1], F)
end

"""
    ext_map_second(EMN, EMNp, g; t, check=true)

Induced map Ext^t(M,N) -> Ext^t(M,Np) computed in the injective model.

Implementation:
1. Lift `g : N -> Np` to a cochain map between the chosen injective resolutions using
   `lift_injective_chainmap` (only up to degree t).
2. Postcompose on Hom(M,-) in degree t and take the induced map on cohomology.

This avoids the older basis-transport through the projective model and is usually faster.
"""
function ext_map_second(EMN::ExtSpaceInjective{QQ},
                        EMNp::ExtSpaceInjective{QQ},
                        g::PMorphism{QQ};
                        t::Int,
                        check::Bool = true)
    @assert g.dom === EMN.N
    @assert g.cod === EMNp.N
    @assert 0 <= t <= EMN.maxdeg
    @assert 0 <= t <= EMNp.maxdeg

    phis = lift_injective_chainmap(g, EMN.res, EMNp.res; upto=t, check=check)

    # postcompose by phi^t : E^t -> E'^t
    F = _postcompose_matrix(EMNp.homs[t+1], EMN.homs[t+1], phis[t+1])

    return induced_map_on_cohomology(EMN.cohom[t+1], EMNp.cohom[t+1], F)
end


# -----------------------------------------------------------------------------
# Internal helper: precomposition matrices on Hom spaces
# -----------------------------------------------------------------------------

"""
    _precompose_matrix(Hdom, Hcod, f) -> Matrix{QQ}

Given a morphism f: A -> B and Hom spaces

- Hcod = Hom(B, E)
- Hdom = Hom(A, E)

return the matrix of the linear map
    f^* : Hom(B, E) -> Hom(A, E),   g |-> g circ f
in the bases stored inside the HomSpace objects.

This is the cochain-level map used by Ext functoriality in the first argument
when Ext is computed via an injective resolution.
"""
function _precompose_matrix(Hdom::HomSpace{QQ}, Hcod::HomSpace{QQ}, f::PMorphism{QQ})
    F = zeros(QQ, dim(Hdom), dim(Hcod))
    for j in 1:dim(Hcod)
        gj = Hcod.basis[j]                 # (cod of f) -> E
        comp = compose(gj, f)              # (dom of f) -> E

        vimg = _morphism_to_vector(comp, Hdom.offsets)
        coeffs = solve_fullcolumnQQ(Hdom.basis_matrix, vimg)

        F[:, j] = coeffs[:, 1]
    end
    return F
end

"""
    _postcompose_matrix(Hdom, Hcod, g)

Return the matrix representing postcomposition with `g` on Hom spaces.

If
  * `Hcod == Hom(M, E)` and
  * `Hdom == Hom(M, Eprime)` and
  * `g : E -> Eprime`,
then this returns the matrix of the linear map

    g_* : Hom(M, E) -> Hom(M, Eprime),   phi |-> g circ phi.

The basis used is the one stored in `Hcod` and `Hdom`.
"""
function _postcompose_matrix(Hdom::HomSpace{QQ}, Hcod::HomSpace{QQ}, g::PMorphism{QQ})
    # Sanity checks: all Hom spaces must share the same domain module.
    @assert Hdom.dom === Hcod.dom
    @assert Hcod.cod === g.dom
    @assert Hdom.cod === g.cod

    F = zeros(QQ, dim(Hdom), dim(Hcod))

    # Column j = coordinates of g circ basis_j expressed in basis(Hdom).
    for j in 1:dim(Hcod)
        fj = Hcod.basis[j]
        gj = compose(g, fj)  # gj : Hdom.dom -> Hdom.cod
        vimg = _morphism_to_vector(gj, Hdom.offsets)
        coeffs = solve_fullcolumnQQ(Hdom.basis_matrix, vimg)
        F[:, j] = coeffs[:, 1]
    end

    return F
end



"""
    connecting_hom_first(EA, EB, EC, i, p; t) -> Matrix{QQ}

Connecting homomorphism for a short exact sequence in the first (contravariant) argument:

    0 -> A --i--> B --p--> C -> 0.

Fix an injective resolution of N and compute Ext via the cochain complexes Hom(-, E^*).
The associated long exact sequence contains the connecting map

    delta^t : Ext^t(A, N) -> Ext^{t+1}(C, N).

This function returns the matrix of delta^t in the chosen Ext bases.

Requirements:
- EA, EB, EC must be `ExtSpaceInjective` objects built from the *same*
  `InjectiveResolution` of N.
- i: A -> B and p: B -> C should define a short exact sequence in the first argument.
"""
function connecting_hom_first(EA::ExtSpaceInjective{QQ},
                              EB::ExtSpaceInjective{QQ},
                              EC::ExtSpaceInjective{QQ},
                              i::PMorphism{QQ},
                              p::PMorphism{QQ}; t::Int)

    if EA.res !== EB.res || EA.res !== EC.res
        error("connecting_hom_first: EA, EB, EC must share the same InjectiveResolution.")
    end
    if t < 0 || t >= EA.tmax
        error("connecting_hom_first: t must satisfy 0 <= t <= tmax-1.")
    end

    # Cochain-level maps at the relevant degrees:
    #   i^* : Hom(B, E^t) -> Hom(A, E^t)
    #   p^* : Hom(C, E^{t+1}) -> Hom(B, E^{t+1})
    It   = _precompose_matrix(EA.homs[t+1], EB.homs[t+1], i)
    Ptp1 = _precompose_matrix(EB.homs[t+2], EC.homs[t+2], p)

    # Coboundary on Hom(B, E^*):
    dBt  = EB.complex.d[t+1]   # degree t: Hom(B,E^t) -> Hom(B,E^{t+1})

    # Domain and codomain cohomology data:
    HdA  = EA.cohom[t+1]       # Ext^t(A,N)
    HdC  = EC.cohom[t+2]       # Ext^{t+1}(C,N)

    delta = zeros(QQ, size(HdC.Hrep, 2), size(HdA.Hrep, 2))

    # For each basis class [z] in Ext^t(A,N), pick a cocycle rep z in Hom(A,E^t),
    # lift it to y in Hom(B,E^t), take dy, then lift dy back through p^* to x in Hom(C,E^{t+1}).
    for j in 1:size(HdA.Hrep, 2)
        z = HdA.Hrep[:, j]

        y = ChainComplexes.solve_particularQQ(It, reshape(z, :, 1))
        dy = dBt * y

        x = ChainComplexes.solve_particularQQ(Ptp1, dy)

        delta[:, j] = cohomology_coordinates(HdC, vec(x))
    end

    return delta
end

# =============================================================================
# Packaged long exact sequence in the FIRST argument (symmetric to Second-argument LES)
# =============================================================================

"""
    ExtLongExactSequenceFirst(A, B, C, N, i, p; maxdeg=3)

Package the long exact sequence in Ext induced by a short exact sequence

    0 -> A --i--> B --p--> C -> 0

in the FIRST argument (contravariant) with a fixed second argument N.

This is the symmetric companion of `ExtLongExactSequenceSecond`, which packages
the LES in the second argument.

Implementation notes
--------------------
We use the injective model (a fixed injective resolution of N) because:

- contravariant functoriality in the first argument is already implemented in
  the injective model via `ext_map_first(::ExtSpaceInjective, ...)`, and
- the connecting morphism is already implemented as `connecting_hom_first`.

All maps (including connecting morphisms) are returned as explicit matrices in
the chosen bases of the injective Ext spaces.
"""
struct ExtLongExactSequenceFirst{K}
    tmin::Int
    tmax::Int
    EA::ExtSpaceInjective{K}  # Ext^*(A, N)
    EB::ExtSpaceInjective{K}  # Ext^*(B, N)
    EC::ExtSpaceInjective{K}  # Ext^*(C, N)

    # Maps on Ext in degree t are stored at index t+1.
    # pH[t+1] : Ext^t(C,N) -> Ext^t(B,N)
    # iH[t+1] : Ext^t(B,N) -> Ext^t(A,N)
    # deltaH[t+1] : Ext^t(A,N) -> Ext^(t+1)(C,N)
    pH::Vector{Matrix{K}}
    iH::Vector{Matrix{K}}
    deltaH::Vector{Matrix{K}}
end

function ExtLongExactSequenceFirst(
    A::PModule{QQ},
    B::PModule{QQ},
    C::PModule{QQ},
    N::PModule{QQ},
    i::PMorphism{QQ},
    p::PMorphism{QQ};
    maxdeg::Int=3
)
    @assert i.dom === A
    @assert i.cod === B
    @assert p.dom === B
    @assert p.cod === C

    # We need Ext^(t+1)(C,N) for delta in degree t, so compute one extra step.
    resN = injective_resolution(N; maxlen=maxdeg + 1)

    EA = ExtInjective(A, resN)
    EB = ExtInjective(B, resN)
    EC = ExtInjective(C, resN)

    tmin = 0
    tmax = maxdeg

    pH = Vector{Matrix{QQ}}(undef, tmax - tmin + 1)
    iH = Vector{Matrix{QQ}}(undef, tmax - tmin + 1)
    deltaH = Vector{Matrix{QQ}}(undef, tmax - tmin + 1)

    for t in tmin:tmax
        pH[t+1] = ext_map_first(EB, EC, p; t=t)
        iH[t+1] = ext_map_first(EA, EB, i; t=t)
        deltaH[t+1] = connecting_hom_first(EA, EB, EC, i, p; t=t)
    end

    return ExtLongExactSequenceFirst{QQ}(tmin, tmax, EA, EB, EC, pH, iH, deltaH)
end


# -----------------------------------------------------------------------------
# Convenience wrappers accepting a ShortExactSequence
# -----------------------------------------------------------------------------

"""
    ExtLongExactSequenceFirst(ses, N; maxdeg)

Convenience wrapper: build the Ext long exact sequence in the first argument from
a checked short exact sequence object.
"""
function ExtLongExactSequenceFirst(ses::ShortExactSequence{QQ}, N::PModule{QQ}; maxdeg::Int)
    return ExtLongExactSequenceFirst(ses.A, ses.B, ses.C, N, ses.i, ses.p; maxdeg=maxdeg)
end

"""
    connecting_hom_first(EA, EB, EC, ses; t)

Convenience wrapper for the Ext connecting morphism in the first argument when the
short exact sequence is passed as a `ShortExactSequence` object.
"""
function connecting_hom_first(EA::ExtSpaceInjective{QQ},
                              EB::ExtSpaceInjective{QQ},
                              EC::ExtSpaceInjective{QQ},
                              ses::ShortExactSequence{QQ}; t::Int)
    return connecting_hom_first(EA, EB, EC, ses.i, ses.p; t=t)
end



# =============================================================================
# Model-independent Ext layer: explicit comparison isomorphisms + coherent basis transport
# =============================================================================

# We build an explicit comparison between:
#  - the projective model: Ext computed from a projective resolution of M, i.e. Hom(P_*, N)
#  - the injective model:  Ext computed from an injective resolution of N, i.e. Hom(M, E^*)
#
# The bridge is the standard total complex Tot(Hom(P_*, E^*)), with explicit cochain maps:
#   Hom(P_*, N)  -> Tot(Hom(P_*, E^*))  <- Hom(M, E^*)
#
# We then solve for basis transport on cohomology degree-by-degree.

# Compute the same block offsets used by `ChainComplexes.total_complex`:
# For each total degree t, the blocks are ordered by increasing a, where b = t-a.
#
# NOTE:
# We qualify DoubleComplex as ChainComplexes.DoubleComplex because ChainComplexes does not
# export it, and DerivedFunctors should not rely on export lists for internal types.
function _total_offsets(DC::ChainComplexes.DoubleComplex{QQ})
    tmin = DC.amin + DC.bmin
    tmax = DC.amax + DC.bmax
    offsets = [Dict{Tuple{Int,Int},Int}() for _ in tmin:tmax]

    for t in tmin:tmax
        off = 0
        for a in DC.amin:DC.amax
            b = t - a
            if DC.bmin <= b <= DC.bmax
                offsets[t - tmin + 1][(a,b)] = off
                # dims is indexed by (a-amin+1, b-bmin+1)
                off += DC.dims[a - DC.amin + 1, b - DC.bmin + 1]
            end
        end
    end
    return offsets, tmin, tmax
end


# Given a morphism f : P -> Target where P = oplus_i Up(bases[i]) is a projective module,
# extract the "cochain coordinate vector" in the basis used for Hom(P, Target).
#
# This mirrors `_cochain_vector_from_morphism` but is specialized to projectives, and
# it uses precomputed `active` upset indices for speed/stability.
function _cochain_vector_from_projective_morphism(
    Q::FinitePoset,
    bases::Vector{Int},
    active::Vector{Vector{Int}},
    offs::Vector{Int},
    Target::PModule{QQ},
    f::PMorphism{QQ}
)
    out = zeros(QQ, offs[end])
    for i in 1:length(bases)
        u = bases[i]
        du = Target.dims[u]
        du == 0 && continue

        act_u = active[u]
        # The column corresponding to generator i at vertex u sits at position `pos`
        # in the ordering of summands active at u.
        pos = searchsortedfirst(act_u, i)
        if pos > length(act_u) || act_u[pos] != i
            error("_cochain_vector_from_projective_morphism: generator index not active at its base")
        end

        out[(offs[i] + 1):offs[i+1]] = f.comps[u][:, pos]
    end
    return out
end

"""
    comparison_isomorphisms(Eproj, Einj; maxdeg=min(Eproj.tmax, Einj.tmax), check=true)

Return explicit basis-transport matrices between the projective and injective models.

The result is a pair `(P2I, I2P)` where, for each degree t:

- `P2I[t+1]` maps coordinates in Ext^t(M,N) from the projective basis to the injective basis.
- `I2P[t+1]` maps coordinates in Ext^t(M,N) from the injective basis to the projective basis.

These matrices are computed by an explicit total-complex comparison
Tot(Hom(P_*, E^*)) and are exact over QQ.
"""
function comparison_isomorphisms(
    Eproj::ExtSpaceProjective{QQ},
    Einj::ExtSpaceInjective{QQ};
    maxdeg::Int=min(Eproj.tmax, Einj.tmax),
    check::Bool=true
)
    return _comparison_projective_injective(Eproj, Einj; maxdeg=maxdeg, check=check)
end

# Internal engine: compute the comparison using Tot(Hom(P_*,E^*)).
function _comparison_projective_injective(
    Eproj::ExtSpaceProjective{QQ},
    Einj::ExtSpaceInjective{QQ};
    maxdeg::Int=min(Eproj.tmax, Einj.tmax),
    check::Bool=true
)
    @assert Eproj.M === Einj.M
    @assert Eproj.N === Einj.res.N

    maxdeg = min(maxdeg, Eproj.tmax, Einj.tmax)
    resP = Eproj.res
    resE = Einj.res
    Q = Eproj.M.Q

    amax = maxdeg
    bmax = maxdeg

    # Precompute offsets for each block Hom(P_a, E^b) in the coordinate basis.
    offs_blocks = Array{Vector{Int}}(undef, amax + 1, bmax + 1)
    dims_blocks = zeros(Int, amax + 1, bmax + 1)

    for a in 0:amax
        gens_a = resP.gens[a+1]
        for b in 0:bmax
            Eb = resE.Emods[b+1]
            off = _block_offsets_for_gens(Eb, gens_a)
            offs_blocks[a+1, b+1] = off
            dims_blocks[a+1, b+1] = off[end]
        end
    end

    # Build the double complex differentials:
    #  dv: postcomposition with d_E : E^b -> E^(b+1)
    #  dh: signed precomposition with d_P : P_(a+1) -> P_a
    dv = Array{SparseMatrixCSC{QQ,Int}}(undef, amax + 1, bmax + 1)
    dh = Array{SparseMatrixCSC{QQ,Int}}(undef, amax + 1, bmax + 1)

    for a in 0:amax
        gens_a = resP.gens[a+1]
        for b in 0:bmax
            dom_dim = dims_blocks[a+1, b+1]

            # vertical differential
            if b < bmax
                g = resE.d_mor[b+1]  # E^b -> E^(b+1)
                dv[a+1,b+1] = _blockdiag_on_hom_cochains_sparse(db, gens_a, offs_src, offs_tgt)
            else
                dv[a+1, b+1] = spzeros(QQ, 0, dom_dim)
            end

            # horizontal differential (signed by (-1)^b)
            if a < amax
                Eb = resE.Emods[b+1]
                H = _build_hom_differential(resP, Eb, a+1, offs_blocks[a+1, b+1], offs_blocks[a+2, b+1])
                if isodd(b)
                    H = -H
                end
                dh[a+1, b+1] = H
            else
                dh[a+1, b+1] = spzeros(QQ, 0, dom_dim)
            end
        end
    end

    # Build the bicomplex and total complex used to compare projective and injective Ext models.
    # Qualify names to avoid relying on exports from ChainComplexes.
    DC = ChainComplexes.DoubleComplex{QQ}(0, amax, 0, bmax, dims_blocks, dv, dh)
    Tot = ChainComplexes.total_complex(DC)
    tot_offsets, tot_tmin, tot_tmax = _total_offsets(DC)

    # Cohomology data for the total complex up to maxdeg.
    cohomT = [cohomology_data(Tot, t) for t in 0:maxdeg]

    # For building the injective-side comparison map, we need coordinates in Hom(P0, E^b).
    basesP0 = resP.gens[1]
    activeP0 = _active_upset_indices(Q, basesP0)

    P2I = Vector{Matrix{QQ}}(undef, maxdeg + 1)
    I2P = Vector{Matrix{QQ}}(undef, maxdeg + 1)

    for t in 0:maxdeg
        # ---- cochain map: Hom(P_t, N) -> Tot in degree t via N -> E^0
        tot_dim = Tot.dims[t - Tot.tmin + 1]
        dom_dim_proj = Eproj.complex.dims[t - Eproj.complex.tmin + 1]

        Fproj = zeros(QQ, tot_dim, dom_dim_proj)
        off_t0 = tot_offsets[t - tot_tmin + 1][(t, 0)]
        block_dim = dims_blocks[t+1, 1]  # (a=t, b=0)
        # Postcompose with iota0: N -> E^0, block-diagonal over generators of P_t.
        block_map = _blockdiag_on_hom_cochains(resE.iota0, resP.gens[t+1], Eproj.offsets[t+1], offs_blocks[t+1, 1])
        Fproj[(off_t0 + 1):(off_t0 + block_dim), :] = block_map

        # ---- cochain map: Hom(M, E^t) -> Tot in degree t via precompose with aug: P0 -> M
        dom_dim_inj = Einj.complex.dims[t - Einj.complex.tmin + 1]
        Finj = zeros(QQ, tot_dim, dom_dim_inj)
        off_0t = tot_offsets[t - tot_tmin + 1][(0, t)]
        Eb = resE.Emods[t+1]
        offs0t = offs_blocks[1, t+1]     # (a=0, b=t)
        block_dim_0t = dims_blocks[1, t+1]

        # Build columns by composing basis Hom(M,E^t) with augmentation P0 -> M.
        for j in 1:dom_dim_inj
            psi = Einj.homs[t+1].basis[j]        # M -> E^t
            comp = compose(psi, resP.aug)         # P0 -> E^t
            Finj[(off_0t + 1):(off_0t + block_dim_0t), j] =
                _cochain_vector_from_projective_morphism(Q, basesP0, activeP0, offs0t, Eb, comp)
        end

        # induced maps on cohomology
        mp = induced_map_on_cohomology(Eproj.cohom[t+1], cohomT[t+1], Fproj)
        mi = induced_map_on_cohomology(Einj.cohom[t+1],  cohomT[t+1], Finj)

        # Solve for explicit basis transport:
        #   mi * (proj->inj) = mp
        # so proj->inj = solve_fullcolumnQQ(mi, mp)
        P2I[t+1] = solve_fullcolumnQQ(mi, mp)
        I2P[t+1] = solve_fullcolumnQQ(mp, mi)

        if check
            dp = dim(Eproj, t)
            di = dim(Einj, t)
            if dp != di
                error("comparison: dim mismatch at t=$(t): projective=$(dp), injective=$(di)")
            end
            I_d = Matrix{QQ}(I, dp, dp)
            if I2P[t+1] * P2I[t+1] != I_d
                error("comparison: I2P*P2I != identity at t=$(t)")
            end
            if P2I[t+1] * I2P[t+1] != I_d
                error("comparison: P2I*I2P != identity at t=$(t)")
            end
        end
    end

    return P2I, I2P
end

"""
    ExtSpace(M, N; maxdeg=3, canon=:projective, check=true)

A single, model-independent "Ext object" that contains BOTH:

- the projective model ExtSpaceProjective(M,N), and
- the injective model  ExtSpaceInjective(M,N),

together with explicit comparison isomorphisms (basis-transport matrices) between them.

The user-facing coordinate basis is fixed by `canon`:
- canon = :projective or :injective

Key guarantees
--------------
- You can request representatives in either model without changing your Ext coordinates.
- You can transport coordinates and induced maps coherently between model realizations.

This directly addresses the "Ext as a bifunctor" ergonomic layer.
"""
struct ExtSpace{K}
    M::PModule{K}
    N::PModule{K}
    Eproj::ExtSpaceProjective{K}
    Einj::ExtSpaceInjective{K}
    P2I::Vector{Matrix{K}}
    I2P::Vector{Matrix{K}}
    canon::Symbol
    # Cached degree range (no Base.getproperty shims).
    tmin::Int
    tmax::Int
end

## NOTE:
## ExtSpace stores `tmin`/`tmax` as explicit fields.
## This avoids Base.getproperty shims while keeping dot-access ergonomic.

function Base.show(io::IO, E::ExtSpace{QQ})
    print(io, "ExtSpace(unified; canon=$(E.canon), tmax=$(E.tmax))")
end

function ExtSpace(
    M::PModule{QQ},
    N::PModule{QQ};
    maxdeg::Int=3,
    canon::Symbol=:projective,
    check::Bool=true
)
    if !(canon === :projective || canon === :injective)
        error("ExtSpace: canon must be :projective or :injective, got $(canon)")
    end

    Eproj = _Ext_projective(M, N; maxdeg=maxdeg)
    Einj  = ExtInjective(M, N; maxdeg=maxdeg)

    P2I, I2P = _comparison_projective_injective(Eproj, Einj; maxdeg=maxdeg, check=check)

    return ExtSpace{QQ}(M, N, Eproj, Einj, P2I, I2P, canon, 0, maxdeg)
end

# Accessors for the two realizations
projective_model(E::ExtSpace{QQ}) = E.Eproj
injective_model(E::ExtSpace{QQ})  = E.Einj

"""
    comparison_isomorphism(E, t; from=:projective, to=:injective)

Return the explicit basis-transport matrix in degree t.

Allowed model symbols: :projective, :injective, :canonical.
"""
function comparison_isomorphism(
    E::ExtSpace{QQ},
    t::Int;
    from::Symbol=:projective,
    to::Symbol=:injective
)
    @assert 0 <= t <= E.tmax

    if from === :canonical
        from = E.canon
    end
    if to === :canonical
        to = E.canon
    end

    if from === to
        d = dim(E, t)
        return Matrix{QQ}(I, d, d)
    end

    if from === :projective && to === :injective
        return E.P2I[t+1]
    elseif from === :injective && to === :projective
        return E.I2P[t+1]
    else
        error("comparison_isomorphism: from=$(from), to=$(to) not supported")
    end
end

# Dimensions agree across models; use the projective one as reference.
dim(E::ExtSpace{QQ}, t::Int) = dim(E.Eproj, t)

"""
    representative(E, t, coords; model=:canonical)

Return a cocycle representative of the Ext class given by `coords` (in the CANONICAL basis).

The keyword `model` chooses the computational realization in which the cocycle is returned:
- model = :canonical (default) returns a cocycle in the canonical model chosen by E.canon
- model = :projective returns a cocycle in Hom(P_t, N)
- model = :injective  returns a cocycle in Hom(M, E^t)

The coordinates are always interpreted in the canonical basis.
"""
function representative(
    E::ExtSpace{QQ},
    t::Int,
    coords::AbstractVector{QQ};
    model::Symbol=:canonical
)
    @assert 0 <= t <= E.tmax
    if model === :canonical
        model = E.canon
    end

    if model === :projective
        if E.canon === :projective
            return representative(E.Eproj, t, coords)
        else
            # coords are canonical injective; convert to projective
            coordsP = E.I2P[t+1] * coords
            return representative(E.Eproj, t, coordsP)
        end
    elseif model === :injective
        if E.canon === :injective
            return representative(E.Einj, t, coords)
        else
            # coords are canonical projective; convert to injective
            coordsI = E.P2I[t+1] * coords
            return representative(E.Einj, t, coordsI)
        end
    else
        error("representative(::ExtSpace): model must be :projective, :injective, or :canonical")
    end
end

"""
    coordinates(E, t, cocycle; model=:canonical)

Compute the coordinates (in the CANONICAL basis) of a cocycle representative.

The keyword `model` specifies which cochain complex the cocycle lives in:
- model = :canonical means the cocycle is in E.canon's complex
- model = :projective means cocycle is in Hom(P_t, N)
- model = :injective means cocycle is in Hom(M, E^t)
"""
function coordinates(
    E::ExtSpace{QQ},
    t::Int,
    cocycle::AbstractVector{QQ};
    model::Symbol=:canonical
)
    @assert 0 <= t <= E.tmax
    if model === :canonical
        model = E.canon
    end

    if model === :projective
        coordsP = coordinates(E.Eproj, t, cocycle)
        if E.canon === :projective
            return coordsP
        else
            return E.P2I[t+1] * coordsP
        end
    elseif model === :injective
        coordsI = coordinates(E.Einj, t, cocycle)
        if E.canon === :injective
            return coordsI
        else
            return E.I2P[t+1] * coordsI
        end
    else
        error("coordinates(::ExtSpace): model must be :projective, :injective, or :canonical")
    end
end

"""
    basis(E, t; model=:canonical)

Return a basis of Ext^t(M,N) as cocycle representatives in the requested model,
ordered by the canonical coordinate basis.

This is the "coherent basis transport" API: the basis vectors correspond to the
same Ext classes regardless of which model you ask for.
"""
function basis(E::ExtSpace{QQ}, t::Int; model::Symbol=:canonical)
    d = dim(E, t)
    out = Vector{Vector{QQ}}(undef, d)
    for i in 1:d
        e = zeros(QQ, d)
        e[i] = one(QQ)
        out[i] = representative(E, t, e; model=model)
    end
    return out
end

"""
    ext_map_first(EMN, EMPN, f; t, backend=:projective)

Induced map on Ext in degree t, contravariant in the first argument:
given f: Mp -> M, returns f^*: Ext^t(M,N) -> Ext^t(Mp,N).

The matrix is always returned in the CANONICAL bases of EMN and EMPN.

`backend` chooses which realization is used for computation. `:projective` is always
available and is the default.
"""
function ext_map_first(
    EMN::ExtSpace{QQ},
    EMPN::ExtSpace{QQ},
    f::PMorphism{QQ};
    t::Int,
    backend::Symbol=:projective
)
    @assert EMN.N === EMPN.N
    @assert EMN.canon === EMPN.canon

    if backend === :projective || backend === :auto
        Aproj = ext_map_first(EMN.Eproj, EMPN.Eproj, f; t=t)
        if EMN.canon === :projective
            return Aproj
        else
            return EMN.P2I[t+1] * Aproj * EMPN.I2P[t+1]
        end
    elseif backend === :injective
        # Only safe if they literally share the same injective resolution object.
        if EMN.Einj.res !== EMPN.Einj.res
            error("ext_map_first(::ExtSpace, backend=:injective) requires a shared injective resolution.")
        end
        Ainj = ext_map_first(EMN.Einj, EMPN.Einj, f; t=t)
        if EMN.canon === :injective
            return Ainj
        else
            return EMN.I2P[t+1] * Ainj * EMPN.P2I[t+1]
        end
    else
        error("ext_map_first(::ExtSpace): unknown backend=$(backend)")
    end
end

"""
    ext_map_second(EMN, EMNp, g; t, backend=:projective)

Induced map on Ext in degree t, covariant in the second argument:
given g: N -> Np, returns g_*: Ext^t(M,N) -> Ext^t(M,Np).

The matrix is always returned in the CANONICAL bases of EMN and EMNp.

`backend=:projective` is fastest and always available.
`backend=:injective` uses the symmetric injective functoriality layer.
"""
function ext_map_second(
    EMN::ExtSpace{QQ},
    EMNp::ExtSpace{QQ},
    g::PMorphism{QQ};
    t::Int,
    backend::Symbol=:projective
)
    @assert EMN.M === EMNp.M
    @assert EMN.canon === EMNp.canon

    if backend === :projective || backend === :auto
        Aproj = ext_map_second(EMN.Eproj, EMNp.Eproj, g; t=t)
        if EMN.canon === :projective
            return Aproj
        else
            return EMNp.P2I[t+1] * Aproj * EMN.I2P[t+1]
        end
    elseif backend === :injective
        Ainj = ext_map_second(EMN.Einj, EMNp.Einj, g; t=t)
        if EMN.canon === :injective
            return Ainj
        else
            return EMNp.I2P[t+1] * Ainj * EMN.P2I[t+1]
        end
    else
        error("ext_map_second(::ExtSpace): unknown backend=$(backend)")
    end
end


# ----------------------------
# Tor for right module (as left module over opposite poset) vs left module
# ----------------------------

struct TorSpace{K}
    resRop::ProjectiveResolution{K}    # projective resolution computed on P^op
    L::PModule{K}                      # left module on P
    bd::Vector{SparseMatrixCSC{K, Int}}  # boundaries bd_s : C_s -> C_{s-1}, s=1..S
    dims::Vector{Int}                    # dim C_s for s=0..S
    offsets::Vector{Vector{Int}}         # offsets per degree
    homol::Vector{ChainComplexes.HomologyData{K}}  # homology data per degree
end

# NOTE: The maximum computed Tor degree is (length(T.dims) - 1).

function _op_poset(P::FinitePoset)
    leq = transpose(P.leq)
    return FinitePoset(leq; check=false)
end

# ----------------------------------------------------------------------
# Alternative Tor model: resolve the second argument.
# ----------------------------------------------------------------------

"""
    TorSpaceSecond

A Tor computation obtained by resolving the *second* argument L (a P-module) and
tensoring that projective resolution with the fixed right module Rop (a P^op-module).

Chain groups:
    C_s = oplus_{u in gens_s(resL)} Rop_u

The boundary matrices are built from the resolution differentials and the structure
maps of Rop.

This model is especially convenient for:
- functoriality and long exact sequences in the *first* argument (Rop),
- cap/actions of Ext(L,L) on Tor(Rop,L), where chain-level maps live on the L-resolution.
"""
struct TorSpaceSecond{K}
    resL::ProjectiveResolution{K}
    Rop::PModule{K}
    bd::Vector{SparseMatrixCSC{K, Int64}}
    dims::Vector{Int64}
    offsets::Vector{Vector{Int64}}
    homol::Vector{ChainComplexes.HomologyData{K}}
end

# NOTE: The maximum computed Tor degree is (length(T.dims) - 1).

# Small helper used for defensive checks when the user supplies a precomputed resolution.
# (We avoid requiring object identity `===` and instead check structural equality.)
function _same_pmodule(M::PModule{QQ}, N::PModule{QQ})
    return (M.Q.leq == N.Q.leq) && (M.dims == N.dims) && (M.edge_maps == N.edge_maps)
end

# Internal implementation: resolve the first argument (existing Tor behavior).
function _Tor_resolve_first(Rop::PModule{QQ}, L::PModule{QQ};
                            maxdeg::Int=3,
                            res::Union{Nothing, ProjectiveResolution{QQ}}=nothing)
    Pop = Rop.Q
    P = _op_poset(Pop)
    @assert L.Q.leq == P.leq

    # Projective resolution of Rop as a Pop module.
    if res === nothing
        res = projective_resolution(Rop; maxlen=maxdeg)
    end
    S = length(res.Pmods) - 1

    # Chain group dims and block offsets.
    dims = Int64[]
    offs = Vector{Vector{Int64}}()
    for s in 0:S
        gens_s = res.gens[s + 1]
        os = zeros(Int64, length(gens_s) + 1)
        for i in 1:length(gens_s)
            u = gens_s[i]
            os[i + 1] = os[i] + L.dims[u]
        end
        push!(offs, os)
        push!(dims, os[end])
    end

    # Boundary matrices C_s -> C_{s-1}.
    bd = SparseMatrixCSC{QQ, Int64}[]
    for s in 1:S
        dom_bases = res.gens[s + 1]
        cod_bases = res.gens[s]
        delta = res.d_mat[s]   # rows=cod, cols=dom

        B = spzeros(QQ, dims[s], dims[s + 1])
        I, J, V = findnz(delta)
        for k in 1:length(V)
            j = I[k]
            i = J[k]
            c = V[k]
            u = dom_bases[i]
            v = cod_bases[j]

            # Nonzero delta entry implies v <=op u in Pop, i.e. u <= v in P,
            # so L has a structure map L_u -> L_v.
            Muv = map_leq(L, u, v)

            rows = (offs[s][j] + 1):offs[s][j + 1]
            cols = (offs[s + 1][i] + 1):offs[s + 1][i + 1]
            B[rows, cols] = c * Muv
        end
        push!(bd, B)
    end

    # Homology data per degree.
    homol = Vector{ChainComplexes.HomologyData{QQ}}(undef, S + 1)
    for s in 0:S
        bd_curr = (s == 0) ? zeros(QQ, 0, dims[s + 1]) : bd[s]
        bd_next = (s == S) ? zeros(QQ, dims[s + 1], 0) : bd[s + 1]
        homol[s + 1] = ChainComplexes.homology_data(bd_next, bd_curr, s)
    end

    return TorSpace{QQ}(res, L, bd, dims, offs, homol)
end

# Internal implementation: resolve the second argument.
function _Tor_resolve_second(Rop::PModule{QQ}, L::PModule{QQ};
                             maxdeg::Int=3,
                             res::Union{Nothing, ProjectiveResolution{QQ}}=nothing)
    Pop = Rop.Q
    P = _op_poset(Pop)
    @assert L.Q.leq == P.leq

    # Projective resolution of L as a P module.
    resL = (res === nothing) ? projective_resolution(L; maxlen=maxdeg) : res
    S = length(resL.Pmods) - 1

    # Chain group dims and block offsets: C_s = oplus Rop_u.
    dims = Int64[]
    offs = Vector{Vector{Int64}}()
    for s in 0:S
        gens_s = resL.gens[s + 1]
        os = zeros(Int64, length(gens_s) + 1)
        for i in 1:length(gens_s)
            u = gens_s[i]
            os[i + 1] = os[i] + Rop.dims[u]
        end
        push!(offs, os)
        push!(dims, os[end])
    end

    # Boundary matrices C_s -> C_{s-1}.
    bd = SparseMatrixCSC{QQ, Int64}[]
    for s in 1:S
        dom_bases = resL.gens[s + 1]
        cod_bases = resL.gens[s]
        delta = resL.d_mat[s]   # rows=cod, cols=dom

        B = spzeros(QQ, dims[s], dims[s + 1])
        I, J, V = findnz(delta)
        for k in 1:length(V)
            j = I[k]
            i = J[k]
            c = V[k]
            u = dom_bases[i]
            v = cod_bases[j]

            # Nonzero delta entry implies v <= u in P, hence u <=op v in Pop,
            # so Rop has a structure map Rop_u -> Rop_v.
            Muv = map_leq(Rop, u, v)

            rows = (offs[s][j] + 1):offs[s][j + 1]
            cols = (offs[s + 1][i] + 1):offs[s + 1][i + 1]
            B[rows, cols] = c * Muv
        end
        push!(bd, B)
    end

    # Homology data per degree.
    homol = Vector{ChainComplexes.HomologyData{QQ}}(undef, S + 1)
    for s in 0:S
        bd_curr = (s == 0) ? zeros(QQ, 0, dims[s + 1]) : bd[s]
        bd_next = (s == S) ? zeros(QQ, dims[s + 1], 0) : bd[s + 1]
        homol[s + 1] = ChainComplexes.homology_data(bd_next, bd_curr, s)
    end

    return TorSpaceSecond{QQ}(resL, Rop, bd, dims, offs, homol)
end

"""
    Tor(Rop, L; maxdeg=3, model=:first, res=nothing)

Compute Tor_i(Rop,L) for 0 <= i <= maxdeg, where:
- `Rop` is a P^op-module (i.e. right module over P),
- `L`   is a P-module.

Two equivalent chain models are available:

- `model=:first`  (default) resolves `Rop` and tensors with `L`. Returns a `TorSpace`.
- `model=:second` resolves `L` and tensors with `Rop`. Returns a `TorSpaceSecond`.

You may optionally supply a projective resolution via `res`:
- if model=:first, `res` must resolve `Rop` (as a P^op-module),
- if model=:second, `res` must resolve `L` (as a P-module).

If `res` is supplied, `maxdeg` is ignored and the computed maximum degree is determined
by the length of `res`.
"""
function Tor(Rop::PModule{QQ}, L::PModule{QQ}; maxdeg::Int=3, model::Symbol=:first, res=nothing)
    if res !== nothing && !(res isa ProjectiveResolution{QQ})
        error("Tor: keyword res must be nothing or a ProjectiveResolution{QQ}.")
    end

    if model === :first
        if res !== nothing
            @assert _same_pmodule(res.M, Rop)
        end
        return _Tor_resolve_first(Rop, L; maxdeg=maxdeg, res=res)
    elseif model === :second
        if res !== nothing
            @assert _same_pmodule(res.M, L)
        end
        return _Tor_resolve_second(Rop, L; maxdeg=maxdeg, res=res)
    else
        error("Tor: unknown model=$(model). Expected :first or :second.")
    end
end


dim(T::TorSpace, s::Int) = T.homol[s+1].dimH
cycles(T::TorSpace, s::Int) = T.homol[s+1].Z
boundaries(T::TorSpace, s::Int) = T.homol[s+1].B

function basis(T::TorSpace, s::Int)
    Hrep = T.homol[s+1].Hrep
    out = Vector{Vector{QQ}}(undef, size(Hrep, 2))
    for i in 1:size(Hrep, 2)
        out[i] = Vector{QQ}(Hrep[:, i])
    end
    return out
end

# Basic accessors for the resolve-second Tor object.
dim(T::TorSpaceSecond, s::Int) = T.homol[s + 1].dimH
cycles(T::TorSpaceSecond, s::Int) = T.homol[s + 1].Zrep
boundaries(T::TorSpaceSecond, s::Int) = T.homol[s + 1].Brep
basis(T::TorSpaceSecond, s::Int) = T.homol[s + 1].Hrep

# ----------------------------------------------------------------------
# Functoriality helpers (shared by TorSpace and TorSpaceSecond)
# ----------------------------------------------------------------------

# This helper takes a coefficient matrix describing a map between direct sums of upsets
# (principal projectives) and tensors it with an arbitrary PModule `M` via structure maps.
#
# The coefficient matrix `coeff` has:
#   - columns indexed by dom_bases
#   - rows indexed by cod_bases
#
# For each nonzero entry (row j, col i) with coefficient c, we place the block:
#   c * map_leq(M, dom_bases[i], cod_bases[j])
#
# This works for:
# - TorSpace (coeff lives in P^op, M is a P-module),
# - TorSpaceSecond (coeff lives in P,   M is a P^op-module),
# because in both cases the nonzero entries correspond to comparable pairs in the relevant poset.
function _tensor_map_on_tor_chains_from_projective_coeff(
    M::PModule{QQ},
    dom_bases::Vector{Int},
    cod_bases::Vector{Int},
    dom_offsets::Vector{Int},
    cod_offsets::Vector{Int},
    coeff::AbstractMatrix{QQ}
)
    out_dim = cod_offsets[end]
    in_dim  = dom_offsets[end]

    I = Int[]
    J = Int[]
    V = QQ[]

    if issparse(coeff)
        Icoeff, Jcoeff, Vcoeff = findnz(coeff)  # rows, cols, vals
        @inbounds for k in eachindex(Vcoeff)
            j = Icoeff[k]   # cod index
            i = Jcoeff[k]   # dom index
            c = Vcoeff[k]
            iszero(c) && continue

            u = dom_bases[i]
            v = cod_bases[j]

            Muv = map_leq(M, u, v)
            _append_scaled_triplets!(I, J, V, Muv, cod_offsets[j], dom_offsets[i]; scale=c)
        end
    else
        # Dense coeff: scan directly (no sparse(coeff) + findnz).
        @inbounds for i in 1:length(dom_bases)
            u = dom_bases[i]
            for j in 1:length(cod_bases)
                c = coeff[j, i]
                iszero(c) && continue

                v = cod_bases[j]
                Muv = map_leq(M, u, v)
                _append_scaled_triplets!(I, J, V, Muv, cod_offsets[j], dom_offsets[i]; scale=c)
            end
        end
    end

    return sparse(I, J, V, out_dim, in_dim)
end

# Block-diagonal chain map induced by a module morphism on each summand of a direct sum.
# Used in tor_map_first for TorSpaceSecond and in connecting morphisms for Tor LES.
function _tor_blockdiag_map_on_chains(
    f::PMorphism{QQ},
    gens::Vector{Int},
    dom_offsets::Vector{Int},
    cod_offsets::Vector{Int}
)
    out_dim = cod_offsets[end]
    in_dim  = dom_offsets[end]

    I = Int[]
    J = Int[]
    V = QQ[]

    @inbounds for i in 1:length(gens)
        u = gens[i]
        _append_scaled_triplets!(I, J, V, f.comps[u], cod_offsets[i], dom_offsets[i])
    end

    return sparse(I, J, V, out_dim, in_dim)
end


# ----------------------------------------------------------------------
# Functoriality: resolve-second model
# ----------------------------------------------------------------------

"""
    tor_map_first(T1, T2, f; s)

For `TorSpaceSecond` objects, Tor is *strictly functorial* in the first argument by a block-diagonal map.

Here `f : Rop -> Rop'` is a P^op-module map (right module map).
The output is the induced linear map:
    Tor_s(Rop, L) -> Tor_s(Rop', L)
in the chosen homology bases.
"""
function tor_map_first(T1::TorSpaceSecond{QQ}, T2::TorSpaceSecond{QQ}, f::PMorphism{QQ}; s::Int)
    @assert T1.resL.M.Q.leq == T2.resL.M.Q.leq
    @assert T1.resL.gens == T2.resL.gens
    @assert f.dom.Q.leq == T1.Rop.Q.leq
    @assert f.cod.Q.leq == T2.Rop.Q.leq

    gens_s = T1.resL.gens[s + 1]
    F = _tor_blockdiag_map_on_chains(f, gens_s, T1.offsets[s + 1], T2.offsets[s + 1])
    return induced_map_on_homology(T1.homol[s + 1], T2.homol[s + 1], F)
end

"""
    tor_map_second(T1, T2, g; s)

For `TorSpaceSecond` objects, functoriality in the second argument requires lifting the module map
to a chain map between projective resolutions, and then tensoring with the fixed right module.

Here `g : L -> L'` is a P-module map.
The output is the induced linear map:
    Tor_s(Rop, L) -> Tor_s(Rop, L')
in the chosen homology bases.
"""
function tor_map_second(T1::TorSpaceSecond{QQ}, T2::TorSpaceSecond{QQ}, g::PMorphism{QQ}; s::Int)
    @assert T1.Rop.Q.leq == T2.Rop.Q.leq
    @assert g.dom.Q.leq == T1.resL.M.Q.leq
    @assert g.cod.Q.leq == T2.resL.M.Q.leq

    # Lift g to a chain map between the chosen projective resolutions.
    coeffs = _lift_pmodule_map_to_projective_resolution_chainmap_coeff(T1.resL, T2.resL, g; upto=s)
    coeff = coeffs[s + 1]  # rows=cod gens, cols=dom gens

    dom_bases = T1.resL.gens[s + 1]
    cod_bases = T2.resL.gens[s + 1]
    F = _tensor_map_on_tor_chains_from_projective_coeff(
        T1.Rop, dom_bases, cod_bases, coeff, T1.offsets[s + 1], T2.offsets[s + 1]
    )
    return induced_map_on_homology(T1.homol[s + 1], T2.homol[s + 1], F)
end

# ============================================================
# Functoriality for TorSpace (resolve FIRST variable)
# ============================================================

"""
    tor_map_second(T1::TorSpace{QQ}, T2::TorSpace{QQ}, g::PMorphism{QQ}; s::Int) -> Matrix{QQ}

Induced map on Tor_s in the second variable, when Tor was computed by resolving
the first variable (i.e. using `model=:first`).

Mathematically: if g : L -> L' is a morphism of P-modules, then there is an induced map
Tor_s(Rop, L) -> Tor_s(Rop, L').

Implementation notes:
- Requires that T1 and T2 were computed using the SAME projective resolution of Rop,
  so that the chain-level direct sum decomposition matches degreewise.
- The chain-level map is block-diagonal over the resolution summands: each summand
  is a copy of L_u, and we apply g_u on that block.
"""
function tor_map_second(T1::TorSpace{QQ}, T2::TorSpace{QQ}, g::PMorphism{QQ}; s::Int)
    @assert T1.resRop.gens == T2.resRop.gens
    gens_s = T1.resRop.gens[s + 1]
    F = _tor_blockdiag_map_on_chains(g, gens_s, T1.offsets[s + 1], T2.offsets[s + 1])
    return induced_map_on_homology(T1.homol[s+1], T2.homol[s+1], F)
end

"""
    tor_map_first(T1::TorSpace{QQ}, T2::TorSpace{QQ}, f::PMorphism{QQ}; s::Int) -> Matrix{QQ}

Induced map on Tor_s in the first variable, when Tor was computed by resolving
the first variable (i.e. using `model=:first`).

Mathematically: if f : Rop -> Rop' is a morphism of P^op-modules, then there is an induced map
Tor_s(Rop, L) -> Tor_s(Rop', L).

Implementation strategy:
- Lift f to a chain map between projective resolutions (coefficient matrices)
- Tensor that chain map with L (using the same helper as the Tor boundary construction)
- Pass to induced map on homology.
"""
function tor_map_first(T1::TorSpace{QQ}, T2::TorSpace{QQ}, f::PMorphism{QQ}; s::Int)
    # Lift f to a chain map between the two resolutions (must be compatible).
    coeffs = _lift_pmodule_map_to_projective_resolution_chainmap_coeff(T1.resRop, T2.resRop, f; upto=s)
    coeff = coeffs[s+1]

    dom_bases = T1.resRop.gens[s + 1]
    cod_bases = T2.resRop.gens[s + 1]

    F = _tensor_map_on_tor_chains_from_projective_coeff(
        T1.L, dom_bases, cod_bases, coeff, T1.offsets[s + 1], T2.offsets[s + 1]
    )

    return induced_map_on_homology(T1.homol[s+1], T2.homol[s+1], F)
end

# ============================
# Packaged Tor long exact sequences (LES) for short exact sequences
# ============================

"""
    TorLongExactSequenceSecond(Rop, i, p; maxdeg=3)

Package the Tor long exact sequence induced by a short exact sequence in the *second* variable:

    0 -> A --i--> B --p--> C -> 0

with fixed first argument `Rop` (a P^op-module).

The object contains:
- Tor spaces `TA=Tor(Rop,A)`, `TB=Tor(Rop,B)`, `TC=Tor(Rop,C)` computed using ONE fixed resolution
  of `Rop` (so that chain-level decompositions match and functorial maps are defined).
- induced maps `iH[s+1] : Tor_s(Rop,A) -> Tor_s(Rop,B)` for s = 0..maxdeg,
- induced maps `pH[s+1] : Tor_s(Rop,B) -> Tor_s(Rop,C)` for s = 0..maxdeg,
- connecting maps `delta[s] : Tor_s(Rop,C) -> Tor_{s-1}(Rop,A)` for s = 1..maxdeg.

Indexing convention:
- `iH[s+1]` and `pH[s+1]` are indexed by homological degree `s=0..maxdeg`.
- The connecting maps are stored WITHOUT the dummy degree-0 map:
    * `delta[1]` is the map in degree 1, i.e. Tor_1(C) -> Tor_0(A).
    * `length(delta) == maxdeg`.

API notes:
- This is a plain struct with one canonical set of field names.
- In particular, we do NOT use Base.getproperty / propertynames shims.
"""
struct TorLongExactSequenceSecond{K}
    Rop::PModule{K}
    A::PModule{K}
    B::PModule{K}
    C::PModule{K}
    i::PMorphism{K}
    p::PMorphism{K}
    TorA::TorSpace{K}
    TorB::TorSpace{K}
    TorC::TorSpace{K}
    iH::Vector{Matrix{K}}
    pH::Vector{Matrix{K}}
    delta::Vector{Matrix{K}}  # stored as delta[s] for s=1..maxdeg
    maxdeg::Int
end

function TorLongExactSequenceSecond(Rop::PModule{QQ}, i::PMorphism{QQ}, p::PMorphism{QQ}; maxdeg::Int=3)
    A = i.dom
    B = i.cod
    C = p.cod
    @assert p.dom.Q.leq == B.Q.leq

    # IMPORTANT:
    # Tor(Rop, -; maxdeg=m) internally builds a projective resolution with maxlen=m.
    # If we build maxlen=m+1 here and then pass it via `res=...`, Tor will compute one
    # extra degree (because `res` overrides `maxdeg`), and tests expecting equality with
    # Tor(Rop, -; maxdeg=m) will fail.
    #
    # So: use maxlen=maxdeg to match Tor's degree convention exactly.
    resR = projective_resolution(Rop; maxlen=maxdeg)
    _pad_projective_resolution!(resR, maxdeg)

    TorA = Tor(Rop, A; model=:first, res=resR)
    TorB = Tor(Rop, B; model=:first, res=resR)
    TorC = Tor(Rop, C; model=:first, res=resR)

    iH = [tor_map_second(TorA, TorB, i; s=s) for s in 0:maxdeg]
    pH = [tor_map_second(TorB, TorC, p; s=s) for s in 0:maxdeg]

    # Store only the genuine connecting morphisms delta_s for s=1..maxdeg.
    delta = Vector{Matrix{QQ}}(undef, maxdeg)
    for s in 1:maxdeg
        delta[s] = _connecting_tor(TorA, TorB, TorC, i, p; s=s)
    end

    return TorLongExactSequenceSecond{QQ}(Rop, A, B, C, i, p, TorA, TorB, TorC, iH, pH, delta, maxdeg)
end


"""
    TorLongExactSequenceFirst(L, i, p; maxdeg=3)

Package the Tor long exact sequence induced by a short exact sequence in the *first* variable:

    0 -> A --i--> B --p--> C -> 0

with fixed second argument `L` (a P-module). Here A,B,C are P^op-modules (right modules).

Implementation note:
We compute Tor using `model=:second` so that one fixed resolution of `L` is shared across A,B,C.

Conventions:
- `iH[s+1]`, `pH[s+1]` act on Tor_s for s=0..maxdeg,
- `delta[s]` is the connecting map Tor_s(C,L) -> Tor_{s-1}(A,L) for s=1..maxdeg,
  so `length(delta) == maxdeg`.
"""
struct TorLongExactSequenceFirst{K}
    L::PModule{K}
    A::PModule{K}
    B::PModule{K}
    C::PModule{K}
    i::PMorphism{K}
    p::PMorphism{K}
    TorA::TorSpaceSecond{K}
    TorB::TorSpaceSecond{K}
    TorC::TorSpaceSecond{K}
    iH::Vector{Matrix{K}}
    pH::Vector{Matrix{K}}
    delta::Vector{Matrix{K}}  # stored as delta[s] for s=1..maxdeg
    maxdeg::Int
end

function TorLongExactSequenceFirst(L::PModule{QQ}, i::PMorphism{QQ}, p::PMorphism{QQ}; maxdeg::Int=3)
    A = i.dom
    B = i.cod
    C = p.cod
    @assert p.dom.Q.leq == B.Q.leq

    # Match Tor(A, L; maxdeg=m, model=:second) degree convention: resolve L with maxlen=m.
    resL = projective_resolution(L; maxlen=maxdeg)
    _pad_projective_resolution!(resL, maxdeg)

    TorA = Tor(A, L; model=:second, res=resL)
    TorB = Tor(B, L; model=:second, res=resL)
    TorC = Tor(C, L; model=:second, res=resL)

    iH = [tor_map_first(TorA, TorB, i; s=s) for s in 0:maxdeg]
    pH = [tor_map_first(TorB, TorC, p; s=s) for s in 0:maxdeg]

    delta = Vector{Matrix{QQ}}(undef, maxdeg)
    for s in 1:maxdeg
        delta[s] = _connecting_tor(TorA, TorB, TorC, i, p; s=s)
    end

    return TorLongExactSequenceFirst{QQ}(L, A, B, C, i, p, TorA, TorB, TorC, iH, pH, delta, maxdeg)
end


# -----------------------------------------------------------------------------
# Convenience wrappers accepting a ShortExactSequence
# -----------------------------------------------------------------------------

"""
    TorLongExactSequenceSecond(Rop, ses; maxdeg=3)

Convenience wrapper: build the Tor long exact sequence in the second argument from
a checked short exact sequence object.
"""
function TorLongExactSequenceSecond(Rop::PModule{QQ}, ses::ShortExactSequence{QQ}; maxdeg::Int=3)
    return TorLongExactSequenceSecond(Rop, ses.i, ses.p; maxdeg=maxdeg)
end

"""
    TorLongExactSequenceFirst(L, ses; maxdeg=3)

Convenience wrapper: build the Tor long exact sequence in the first argument from
a checked short exact sequence object.
"""
function TorLongExactSequenceFirst(L::PModule{QQ}, ses::ShortExactSequence{QQ}; maxdeg::Int=3)
    return TorLongExactSequenceFirst(L, ses.i, ses.p; maxdeg=maxdeg)
end



# ----------------------------------------------------------------------
# Ext action on Tor (cap/action flavor)
# ----------------------------------------------------------------------

"""
    ext_action_on_tor(A, T, x; s)

Given:
- `A::ExtAlgebra` for a module L (so A computes Ext^*(L,L)),
- `T::TorSpaceSecond` computing Tor_*(Rop, L) using the *same* projective resolution of L,
- an `ExtElement` x in degree m,

return the induced degree-lowering action matrix:
    x cap - : Tor_s(Rop, L) -> Tor_{s-m}(Rop, L)

in the homology bases of `T`.

Notes:
- This is implemented by lifting a cocycle representative of x to a chain map of the resolution
  (via `_lift_cocycle_to_chainmap_coeff`) and then tensoring that chain map with Rop.
- For s < m, the target degree is negative, so the action is the zero map.
"""
function ext_action_on_tor(A::ExtAlgebra{QQ}, T::TorSpaceSecond{QQ}, x::ExtElement{QQ}; s::Int)
    m = x.deg
    if s < m
        return zeros(QQ, 0, dim(T, s))
    end

    # Basic compatibility checks: same resolved module and same chosen resolution.
    @assert A.E.M.Q.leq == T.resL.M.Q.leq
    @assert A.E.res.gens == T.resL.gens

    # Choose a cocycle representative alpha in cochain degree m.
    alpha = reshape(representative(A.E, m, x.coords), :, 1)

    # Lift to a chain map P_{m+k} -> P_k, for k up to s-m.
    coeffs = _lift_cocycle_to_chainmap_coeff(A.E.res, A.E.res, A.E, m, alpha; upto=(s - m))

    # We need the coefficient matrix at k = s-m, which maps P_s -> P_{s-m}.
    coeff = coeffs[(s - m) + 1]

    dom_bases = T.resL.gens[s + 1]
    cod_bases = T.resL.gens[(s - m) + 1]

    F = _tensor_map_on_tor_chains_from_projective_coeff(
        T.Rop, dom_bases, cod_bases, coeff, T.offsets[s + 1], T.offsets[(s - m) + 1]
    )

    # Apply F to chosen Tor_s basis reps and express in Tor_{s-m} coordinates.
    reps = T.homol[s + 1].Hrep
    images = F * reps
    return ChainComplexes.homology_coordinates(T.homol[(s - m) + 1], images)
end

# Convenience: compute the action matrices for all degrees up to maxdeg.
function ext_action_on_tor(A::ExtAlgebra{QQ}, T::TorSpaceSecond{QQ}, x::ExtElement{QQ}; maxdeg::Int=(length(T.dims) - 1))
    mats = Vector{Matrix{QQ}}(undef, maxdeg + 1)
    for s in 0:maxdeg
        mats[s + 1] = ext_action_on_tor(A, T, x; s=s)
    end
    return mats
end


# ============================================================
# TorAlgebra with lazy mu_chain generation (generator + cache)
# ============================================================

"""
    TorAlgebra(T; mu_chain=Dict(), mu_chain_gen=nothing, unit_coords=nothing)

A thin wrapper that equips a computed Tor space `T` with a bilinear graded multiplication.

Mathematical input:
- `T` is a Tor computation object (either TorSpace or TorSpaceSecond).
- A chain-level product is given by matrices

    mu_chain[(p,q)] : C_p tensor C_q -> C_{p+q}

in the chosen chain bases.

Practical API:
- You may supply all maps explicitly via `mu_chain`.
- Or, supply a lazy generator `mu_chain_gen(p,q)` that returns the required sparse matrix.
  The result is cached in `A.mu_chain` automatically on first use.

This design is exactly what the screenshot describes: once the infrastructure exists,
adding a specific canonical multiplication is "just supplying mu_chain[(p,q)] maps
(or a generator that builds them)".
"""
mutable struct TorAlgebra{K}
    T::Any
    mu_chain::Dict{Tuple{Int,Int}, SparseMatrixCSC{K, Int}}
    mu_chain_gen::Union{Nothing, Function}
    mu_H_cache::Dict{Tuple{Int,Int}, Matrix{K}}
    unit_coords::Union{Nothing, Vector{K}}
end

TorAlgebra(T; mu_chain=Dict{Tuple{Int, Int}, SparseMatrixCSC{QQ, Int64}}(), unit_coords=nothing) =
    TorAlgebra{QQ}(T, mu_chain, Dict{Tuple{Int, Int}, Matrix{QQ}}(), unit_coords)

"""
    TorAlgebra(T::Any; mu_chain=Dict(), mu_chain_gen=nothing, unit_coords=nothing)

Constructor with optional lazy generator.
"""
function TorAlgebra(T::Any;
    mu_chain::Dict{Tuple{Int,Int}, SparseMatrixCSC{QQ,Int}}=Dict{Tuple{Int,Int}, SparseMatrixCSC{QQ,Int}}(),
    mu_chain_gen::Union{Nothing,Function}=nothing,
    unit_coords::Union{Nothing,Vector{QQ}}=nothing)

    return TorAlgebra{QQ}(T, mu_chain, mu_chain_gen, Dict{Tuple{Int,Int}, Matrix{QQ}}(), unit_coords)
end

# Internal: obtain a chain-level multiplication map, generating it if needed.
function _get_mu_chain(A::TorAlgebra{QQ}, p::Int, q::Int)
    key = (p,q)
    if haskey(A.mu_chain, key)
        return A.mu_chain[key]
    end
    if A.mu_chain_gen === nothing
        error("TorAlgebra: no mu_chain[(p,q)] provided and no mu_chain_gen set for (p,q)=($p,$q).")
    end
    M = A.mu_chain_gen(p,q)
    isa(M, SparseMatrixCSC{QQ,Int}) || error("mu_chain_gen must return SparseMatrixCSC{QQ,Int}")
    A.mu_chain[key] = M
    return M
end

"""
    TorElement(A, deg, coords)

A Tor class in degree `deg`, expressed in the basis used by `A.T`.
"""
struct TorElement{K}
    A::TorAlgebra{K}
    deg::Int
    coords::Vector{K}
end

element(A::TorAlgebra{QQ}, deg::Int, coords::AbstractVector{QQ}) =
    TorElement{QQ}(A, deg, collect(coords))

"""
    set_chain_product!(A, p, q, mu)

Set the chain-level product map for degrees (p,q).
This overrides any lazily generated value.
"""
function set_chain_product!(A::TorAlgebra{QQ}, p::Int, q::Int, mu::SparseMatrixCSC{QQ,Int})
    A.mu_chain[(p,q)] = mu
    # If we already cached induced homology matrices, clear them.
    empty!(A.mu_H_cache)
    return A
end

"""
    set_chain_product_generator!(A, gen)

Attach a lazy generator `gen(p,q)` to supply mu_chain maps on demand.
Clears caches.
"""
function set_chain_product_generator!(A::TorAlgebra{QQ}, gen::Function)
    A.mu_chain_gen = gen
    empty!(A.mu_chain)
    empty!(A.mu_H_cache)
    return A
end

"""
    multiplication_matrix(A, p, q)

Return the induced multiplication matrix on Tor homology:

    Tor_p x Tor_q -> Tor_{p+q}

The returned matrix has size:
    dim(Tor_{p+q}) x (dim(Tor_p)*dim(Tor_q)),

and its column ordering matches `kron(x.coords, y.coords)`.
"""
function multiplication_matrix(A::TorAlgebra{QQ}, p::Int, q::Int)
    key = (p,q)
    if haskey(A.mu_H_cache, key)
        return A.mu_H_cache[key]
    end

    # Get chain-level multiplication map (possibly generated lazily)
    mu = _get_mu_chain(A, p, q)

    # Existing logic (unchanged): push reps through mu, then project to homology.
    Tp = A.T.homol[p+1]
    Tq = A.T.homol[q+1]
    Tr = A.T.homol[p+q+1]

    Hp = size(Tp.Hrep, 2)
    Hq = size(Tq.Hrep, 2)
    Hr = size(Tr.Hrep, 2)

    out = zeros(QQ, Hr, Hp * Hq)
    col = 0
    for i in 1:Hp
        for j in 1:Hq
            col += 1
            xp = Tp.Hrep[:, i:i]
            xq = Tq.Hrep[:, j:j]
            x = kron(xq, xp)
            y = mu * x
            out[:, col:col] .= ChainComplexes.homology_coordinates(Tr, y)
        end
    end

    A.mu_H_cache[key] = out
    return out
end

"""
    multiply(A, x, y)

Multiply Tor elements using the registered chain-level product.
"""
function multiply(A::TorAlgebra{QQ}, x::TorElement{QQ}, y::TorElement{QQ})
    @assert x.A === A && y.A === A
    p, q = x.deg, y.deg
    M = multiplication_matrix(A, p, q)
    out_coords = M * kron(x.coords, y.coords)
    return TorElement{QQ}(A, p + q, out_coords)
end

"""
    trivial_tor_product_generator(T)

Return a mu_chain_gen(p,q) implementing the canonical "degree-0 only" product:
- if p==0 and q==0, multiply by identity on C_0 (using the chain basis)
- otherwise, return the zero map.

This is the maximal canonical choice available without extra structure on the poset/algebra.
It is always a valid chain-level multiplication (and hence induces a graded algebra structure),
and serves as a safe default. More sophisticated products (bar/shuffle/Koszul) can be plugged
in by supplying a different generator via `set_chain_product_generator!`.
"""
function trivial_tor_product_generator(T)
    # We need chain group dimensions. Both TorSpace and TorSpaceSecond store dims as T.dims.
    function gen(p::Int, q::Int)
        if p != 0 || q != 0
            return spzeros(QQ, T.dims[p+1+q], T.dims[p+1] * T.dims[q+1])
        end
        # degree 0: C_0 tensor C_0 -> C_0
        # Use basis-dependent diagonal multiplication: e_i tensor e_j -> delta_{ij} e_i
        n = T.dims[1]
        M = spzeros(QQ, n, n*n)
        for i in 1:n
            # column index for (i,i) in kron basis: (j-1)*n + i
            col = (i-1)*n + i
            M[i, col] = 1//1
        end
        return M
    end
    return gen
end



# ----------------------------
# Zn wrappers: compute on a box by first building a finite poset module
# ----------------------------

function pmodule_on_box(FG; a::Vector{Int}, b::Vector{Int})
    return ZnEncoding.pmodule_on_box(FG; a=a, b=b)
end

"""
    ExtZn(FG1, FG2; method=:regions, a=nothing, b=nothing, max_regions=200_000, maxdeg=3)

Compute `Ext^s(FG1, FG2)` for Z^n modules given by flange presentations.

Keyword arguments
* `method`:
    - `:regions` (default) builds a common finite encoding poset from the union
      of the generators in `FG1` and `FG2` and computes Ext on that poset.
    - `:box` restricts both modules to the finite grid poset on `[a,b]`.
* `a, b`: required when `method == :box`.
* `max_regions`: cap on number of regions for `method == :regions`.
* `maxdeg`: compute Ext^s for 0 <= s <= maxdeg.
"""
function ExtZn(FG1, FG2;
               method::Symbol = :regions,
               a::Union{Nothing,Vector{Int}} = nothing,
               b::Union{Nothing,Vector{Int}} = nothing,
               max_regions::Int = 200_000,
               maxdeg::Int = 3)
    if method == :box
        a === nothing && error("ExtZn(method=:box): missing keyword a")
        b === nothing && error("ExtZn(method=:box): missing keyword b")
        M = pmodule_on_box(FG1; a=a, b=b)
        N = pmodule_on_box(FG2; a=a, b=b)
        return Ext(M, N; maxdeg=maxdeg)
    elseif method == :regions
        P, Hs, pi = ZnEncoding.encode_from_flanges(FG1, FG2; max_regions=max_regions)
        M = pmodule_from_fringe(Hs[1])
        N = pmodule_from_fringe(Hs[2])
        return Ext(M, N; maxdeg=maxdeg)
    else
        error("ExtZn: unknown method=$(method); expected :regions or :box")
    end
end

# =============================================================================
# Z^n convenience layer: compute on a finite encoding poset P
#
# These helpers implement the same architectural pattern as the R^n wrappers:
#   flange (Z^n presentation) -> (P, M, pi) via region encoding,
#   then do homological algebra in the finite-poset category on P.
#
# All output is therefore "on P" data. Interpreting it back on Z^n requires
# using the classifier pi : Z^n -> P.
# =============================================================================

"""
    encode_pmodule_from_flange(FG::Flange{QQ}; max_regions=200_000)

Encode a single Z^n flange presentation to a finite encoding poset `P` using the
region-based encoder in `ZnEncoding`, then convert the resulting fringe module to
a finite-poset module `M::PModule{QQ}`.

Returns a NamedTuple with fields:
- `P`  : the finite encoding poset
- `M`  : a `PModule{QQ}` on `P`
- `pi` : the classifier map `Z^n -> P` (`ZnEncodingMap`)

You may either access fields (`enc.P`, `enc.M`, `enc.pi`) or destructure:

    P, M, pi = encode_pmodule_from_flange(FG)

Keyword arguments
- `max_regions::Int = 200_000`:
  Hard cap on the number of regions/signatures allowed during encoding.
"""
function encode_pmodule_from_flange(FG::Flange{QQ};
                                    max_regions::Int = 200_000)
    P, H, pi = ZnEncoding.encode_from_flange(FG; max_regions = max_regions)
    M = pmodule_from_fringe(H)
    return (P = P, M = M, pi = pi)
end


"""
    encode_pmodules_from_flanges(FG1, FG2, ...; max_regions=200_000)

Common-encode several Z^n flange presentations to a *single* finite encoding poset `P`,
convert each to a finite-poset module, and return them as `Ms`.

Returns a NamedTuple with fields:
- `P`  : the common finite encoding poset
- `Ms` : `Vector{PModule{QQ}}`, one per input flange (all over the same `P`)
- `pi` : classifier map `Z^n -> P` (`ZnEncodingMap`)

You may either access fields (`enc.P`, `enc.Ms`, `enc.pi`) or destructure:

    P, Ms, pi = encode_pmodules_from_flanges(FG1, FG2)

Keyword arguments
- `max_regions::Int = 200_000`:
  Hard cap on the number of regions/signatures allowed during encoding.
"""
function encode_pmodules_from_flanges(FGs::Vararg{Flange{QQ}};
                                      max_regions::Int = 200_000)
    P, Hs, pi = ZnEncoding.encode_from_flanges(FGs...; max_regions = max_regions)
    Ms = [pmodule_from_fringe(H) for H in Hs]
    return (P = P, Ms = Ms, pi = pi)
end




"""
    projective_resolution_Zn(FG::Flange{QQ};
                            max_regions=200_000,
                            maxlen=3,
                            return_encoding=false)

Compute a projective resolution of the Z^n module presented by FG by:
    FG -> (P, M, pi) via region encoding, then projective_resolution(M) on P.

If return_encoding=true, return a named tuple:
    (res=res, P=P, pi=pi)

This is a computation on the encoding poset P. Minimality and Betti-style data
are minimality/Betti on P, not a Z^n-native commutative-algebra theory.
"""
function projective_resolution_Zn(FG::Flange{QQ};
                                 max_regions::Int=200_000,
                                 maxlen::Int=3,
                                 return_encoding::Bool=false)

    P, M, pi = encode_pmodule_from_flange(FG; max_regions=max_regions)
    res = projective_resolution(M; maxlen=maxlen)
    return return_encoding ? (res=res, P=P, pi=pi) : res
end


"""
    injective_resolution_Zn(FG::Flange{QQ};
                           max_regions=200_000,
                           maxlen=3,
                           return_encoding=false)

Compute an injective resolution of the Z^n module presented by FG by:
    FG -> (P, M, pi) via region encoding, then injective_resolution(M) on P.

If return_encoding=true, return a named tuple:
    (res=res, P=P, pi=pi)

This is a computation on the encoding poset P.
"""
function injective_resolution_Zn(FG::Flange{QQ};
                                max_regions::Int=200_000,
                                maxlen::Int=3,
                                return_encoding::Bool=false)

    P, M, pi = encode_pmodule_from_flange(FG; max_regions=max_regions)
    res = injective_resolution(M; maxlen=maxlen)
    return return_encoding ? (res=res, P=P, pi=pi) : res
end


"""
    minimal_projective_resolution_Zn(FG::Flange{QQ};
                                    max_regions=200_000,
                                    maxlen=3,
                                    check=true,
                                    return_encoding=false)

Compute a certified minimal projective resolution on the finite encoding poset P.

If return_encoding=true, return (res=res, P=P, pi=pi).
"""
function minimal_projective_resolution_Zn(FG::Flange{QQ};
                                         max_regions::Int=200_000,
                                         maxlen::Int=3,
                                         check::Bool=true,
                                         return_encoding::Bool=false)

    P, M, pi = encode_pmodule_from_flange(FG; max_regions=max_regions)
    res = minimal_projective_resolution(M; maxlen=maxlen, check=check)
    return return_encoding ? (res=res, P=P, pi=pi) : res
end


"""
    minimal_injective_resolution_Zn(FG::Flange{QQ};
                                   max_regions=200_000,
                                   maxlen=3,
                                   check=true,
                                   return_encoding=false)

Compute a certified minimal injective resolution on the finite encoding poset P.

If return_encoding=true, return (res=res, P=P, pi=pi).
"""
function minimal_injective_resolution_Zn(FG::Flange{QQ};
                                        max_regions::Int=200_000,
                                        maxlen::Int=3,
                                        check::Bool=true,
                                        return_encoding::Bool=false)

    P, M, pi = encode_pmodule_from_flange(FG; max_regions=max_regions)
    res = minimal_injective_resolution(M; maxlen=maxlen, check=check)
    return return_encoding ? (res=res, P=P, pi=pi) : res
end


"""
    minimal_betti_Zn(FG::Flange{QQ};
                     max_regions=200_000,
                     maxlen=3,
                     check=true)

Compute Betti multiplicities from a certified minimal projective resolution on
the encoding poset P.
"""
function minimal_betti_Zn(FG::Flange{QQ};
                          max_regions::Int=200_000,
                          maxlen::Int=3,
                          check::Bool=true)

    P, M, pi = encode_pmodule_from_flange(FG; max_regions=max_regions)
    return minimal_betti(M; maxlen=maxlen, check=check)
end


"""
    minimal_bass_Zn(FG::Flange{QQ};
                    max_regions=200_000,
                    maxlen=3,
                    check=true)

Compute Bass multiplicities from a certified minimal injective resolution on
the encoding poset P.
"""
function minimal_bass_Zn(FG::Flange{QQ};
                         max_regions::Int=200_000,
                         maxlen::Int=3,
                         check::Bool=true)

    P, M, pi = encode_pmodule_from_flange(FG; max_regions=max_regions)
    return minimal_bass(M; maxlen=maxlen, check=check)
end



"""
    ExtRn(F1::PLPolyhedra.PLFringe, F2::PLPolyhedra.PLFringe;
          max_regions=10_000, strict_eps=PLPolyhedra.STRICT_EPS_QQ, maxdeg=3)

Compute Ext^s(F1, F2) for modules over R^n given by PL fringe presentations.

This follows the Miller-style pattern: do homological algebra on a finite encoding
poset, not on the infinite poset R^n directly.

Algorithm:
1. Build a single common encoding poset P that simultaneously encodes the union of
   all birth/death shapes appearing in F1 and F2.
2. Push both presentations down to fringe modules on P.
3. Convert to P-modules and compute Ext on the finite poset.

Keyword arguments:
- `max_regions`: cap on number of regions/signatures enumerated by the polyhedral backend.
- `strict_eps`: rational slack used when forcing "outside" constraints.
- `maxdeg`: compute Ext^s for 0 <= s <= maxdeg.
"""
function ExtRn(F1::PLPolyhedra.PLFringe, F2::PLPolyhedra.PLFringe;
               max_regions::Int = 10_000,
               strict_eps::QQ = PLPolyhedra.STRICT_EPS_QQ,
               maxdeg::Int = 3)

    P, Hs, pi = PLPolyhedra.encode_from_PL_fringes(F1, F2;
                                                  max_regions=max_regions,
                                                  strict_eps=strict_eps)
    M = pmodule_from_fringe(Hs[1])
    N = pmodule_from_fringe(Hs[2])
    return Ext(M, N; maxdeg=maxdeg)
end

"""
    encode_pmodule_from_PL_fringe(F::PLPolyhedra.PLFringe;
                                  max_regions=10_000,
                                  strict_eps=PLPolyhedra.STRICT_EPS_QQ)

Encode a single PL fringe presentation over R^n to a finite encoding poset P,
convert to a finite-poset module.

Returns a NamedTuple with fields:
- P  : the finite encoding poset
- M  : a PModule{QQ} on P
- pi : classifier R^n -> P (PLEncodingMap)

You may destructure:

    P, M, pi = encode_pmodule_from_PL_fringe(F)
"""
function encode_pmodule_from_PL_fringe(F::PLPolyhedra.PLFringe;
                                       max_regions::Int=10_000,
                                       strict_eps::QQ=PLPolyhedra.STRICT_EPS_QQ)
    P, H, pi = PLPolyhedra.encode_from_PL_fringe(F;
                                                max_regions=max_regions,
                                                strict_eps=strict_eps)
    M = pmodule_from_fringe(H)
    return (P=P, M=M, pi=pi)
end


"""
    encode_pmodules_from_PL_fringes(F1, F2, ...;
                                    max_regions=10_000,
                                    strict_eps=PLPolyhedra.STRICT_EPS_QQ)

Common-encode several PL fringe presentations over R^n to a single finite encoding poset P.

Returns a NamedTuple with fields:
- P  : the common finite encoding poset
- Ms : Vector{PModule{QQ}}, one per input PLFringe, all over the same P
- pi : classifier R^n -> P (PLEncodingMap)

You may destructure:

    P, Ms, pi = encode_pmodules_from_PL_fringes(F1, F2)
"""
function encode_pmodules_from_PL_fringes(Fs::Vararg{PLPolyhedra.PLFringe};
                                        max_regions::Int=10_000,
                                        strict_eps::QQ=PLPolyhedra.STRICT_EPS_QQ)

    P, Hs, pi = PLPolyhedra.encode_from_PL_fringes(Fs...;
                                                  max_regions=max_regions,
                                                  strict_eps=strict_eps)

    Ms = Vector{PModule{QQ}}(undef, length(Hs))
    for i in 1:length(Hs)
        Ms[i] = pmodule_from_fringe(Hs[i])
    end

    return (P=P, Ms=Ms, pi=pi)
end

"""
        ExtDoubleComplex(FG1, FG2; method=:regions, a=nothing, b=nothing,
                     maxlen=nothing, max_regions=5000)

Z^n wrapper: encode two flange modules onto a common finite poset, then build the Ext bicomplex.
If method=:box, you must provide a and b defining the bounding box.
If method=:regions, a finite encoding is built via region sampling.
"""
function ExtDoubleComplex(FG1::Flange{QQ}, FG2::Flange{QQ};
                          method::Symbol=:regions,
                          a=nothing,
                          b=nothing,
                          maxlen::Union{Nothing,Int}=nothing,
                          max_regions::Int=5000)
    if method == :box
        if a === nothing || b === nothing
            error("ExtDoubleComplex: method=:box requires keywords a and b.")
        end
        M1 = pmodule_on_box(FG1, a, b)
        M2 = pmodule_on_box(FG2, a, b)
        return ExtDoubleComplex(M1, M2; maxlen=maxlen)
    elseif method == :regions
        # NOTE: encode_pmodules_from_flanges expects varargs, not a Vector argument.
        # It also only supports `max_regions` as a keyword.
        _, Ms, _ = encode_pmodules_from_flanges(FG1, FG2; max_regions=max_regions)
        return ExtDoubleComplex(Ms[1], Ms[2]; maxlen=maxlen)
    else
        error("ExtDoubleComplex: unknown method (use :box or :regions).")
    end
end

function ExtSpectralSequence(FG1::Flange{QQ}, FG2::Flange{QQ};
                             first::Symbol=:vertical,
                             method::Symbol=:regions,
                             a=nothing,
                             b=nothing,
                             maxlen::Union{Nothing,Int}=nothing,
                             max_regions::Int=5000)
    DC = ExtDoubleComplex(FG1, FG2;
                          method=method,
                          a=a, b=b,
                          maxlen=maxlen,
                          max_regions=max_regions)
    return ChainComplexes.spectral_sequence(DC; first=first)
end

"""
    ExtDoubleComplex(PL1, PL2; maxlen=nothing,
                     max_regions=10_000,
                     strict_eps=PLPolyhedra.STRICT_EPS_QQ)

R^n wrapper: encode two PL fringes onto a common finite poset, then build the Ext bicomplex.
"""
function ExtDoubleComplex(PL1::PLFringe, PL2::PLFringe;
                          maxlen::Union{Nothing,Int}=nothing,
                          max_regions::Int=10_000,
                          strict_eps::QQ=PLPolyhedra.STRICT_EPS_QQ)
    _, Ms, _ = encode_pmodules_from_PL_fringes(PL1, PL2;
                                              max_regions=max_regions,
                                              strict_eps=strict_eps)
    return ExtDoubleComplex(Ms[1], Ms[2]; maxlen=maxlen)
end

function ExtSpectralSequence(PL1::PLFringe, PL2::PLFringe;
                             first::Symbol=:vertical,
                             maxlen::Union{Nothing,Int}=nothing,
                             max_regions::Int=10_000,
                             strict_eps::QQ=PLPolyhedra.STRICT_EPS_QQ)
    DC = ExtDoubleComplex(PL1, PL2;
                          maxlen=maxlen,
                          max_regions=max_regions,
                          strict_eps=strict_eps)
    return ChainComplexes.spectral_sequence(DC; first=first)
end


"""
    projective_resolution_Rn(F::PLPolyhedra.PLFringe;
                            max_regions=10_000,
                            strict_eps=PLPolyhedra.STRICT_EPS_QQ,
                            maxlen=3,
                            return_encoding=false)

Compute a projective resolution of the module over R^n presented by F by:
    F -> (P, M, pi) via encoding, then projective_resolution(M) on the finite poset P.

If return_encoding=true, return a named tuple:
    (res=res, P=P, pi=pi)

This is a computation on the encoding poset P. Minimality and Betti-style data
are minimality/Betti on P, not an R^n-native commutative-algebra theory.
"""
function projective_resolution_Rn(F::PLPolyhedra.PLFringe;
                                 max_regions::Int=10_000,
                                 strict_eps::QQ=PLPolyhedra.STRICT_EPS_QQ,
                                 maxlen::Int=3,
                                 return_encoding::Bool=false)

    P, M, pi = encode_pmodule_from_PL_fringe(F;
                                            max_regions=max_regions,
                                            strict_eps=strict_eps)
    res = projective_resolution(M; maxlen=maxlen)
    return return_encoding ? (res=res, P=P, pi=pi) : res
end


"""
    injective_resolution_Rn(F::PLPolyhedra.PLFringe;
                           max_regions=10_000,
                           strict_eps=PLPolyhedra.STRICT_EPS_QQ,
                           maxlen=3,
                           return_encoding=false)

Compute an injective resolution of the module over R^n presented by F by:
    F -> (P, M, pi) via encoding, then injective_resolution(M) on the finite poset P.

If return_encoding=true, return a named tuple:
    (res=res, P=P, pi=pi)

This is a computation on the encoding poset P.
"""
function injective_resolution_Rn(F::PLPolyhedra.PLFringe;
                                max_regions::Int=10_000,
                                strict_eps::QQ=PLPolyhedra.STRICT_EPS_QQ,
                                maxlen::Int=3,
                                return_encoding::Bool=false)

    P, M, pi = encode_pmodule_from_PL_fringe(F;
                                            max_regions=max_regions,
                                            strict_eps=strict_eps)
    res = injective_resolution(M; maxlen=maxlen)
    return return_encoding ? (res=res, P=P, pi=pi) : res
end


"""
    minimal_projective_resolution_Rn(F::PLPolyhedra.PLFringe;
                                    max_regions=10_000,
                                    strict_eps=PLPolyhedra.STRICT_EPS_QQ,
                                    maxlen=3,
                                    check=true,
                                    return_encoding=false)

Compute a certified minimal projective resolution on the finite encoding poset P.

If return_encoding=true, return (res=res, P=P, pi=pi).
"""
function minimal_projective_resolution_Rn(F::PLPolyhedra.PLFringe;
                                         max_regions::Int=10_000,
                                         strict_eps::QQ=PLPolyhedra.STRICT_EPS_QQ,
                                         maxlen::Int=3,
                                         check::Bool=true,
                                         return_encoding::Bool=false)

    P, M, pi = encode_pmodule_from_PL_fringe(F;
                                            max_regions=max_regions,
                                            strict_eps=strict_eps)
    res = minimal_projective_resolution(M; maxlen=maxlen, check=check)
    return return_encoding ? (res=res, P=P, pi=pi) : res
end


"""
    minimal_injective_resolution_Rn(F::PLPolyhedra.PLFringe;
                                   max_regions=10_000,
                                   strict_eps=PLPolyhedra.STRICT_EPS_QQ,
                                   maxlen=3,
                                   check=true,
                                   return_encoding=false)

Compute a certified minimal injective resolution on the finite encoding poset P.

If return_encoding=true, return (res=res, P=P, pi=pi).
"""
function minimal_injective_resolution_Rn(F::PLPolyhedra.PLFringe;
                                        max_regions::Int=10_000,
                                        strict_eps::QQ=PLPolyhedra.STRICT_EPS_QQ,
                                        maxlen::Int=3,
                                        check::Bool=true,
                                        return_encoding::Bool=false)

    P, M, pi = encode_pmodule_from_PL_fringe(F;
                                            max_regions=max_regions,
                                            strict_eps=strict_eps)
    res = minimal_injective_resolution(M; maxlen=maxlen, check=check)
    return return_encoding ? (res=res, P=P, pi=pi) : res
end


"""
    minimal_betti_Rn(F::PLPolyhedra.PLFringe;
                    max_regions=10_000,
                    strict_eps=PLPolyhedra.STRICT_EPS_QQ,
                    maxlen=3,
                    check=true)

Compute Betti multiplicities from a certified minimal projective resolution on
the encoding poset P.
"""
function minimal_betti_Rn(F::PLPolyhedra.PLFringe;
                          max_regions::Int=10_000,
                          strict_eps::QQ=PLPolyhedra.STRICT_EPS_QQ,
                          maxlen::Int=3,
                          check::Bool=true)

    P, M, pi = encode_pmodule_from_PL_fringe(F;
                                            max_regions=max_regions,
                                            strict_eps=strict_eps)
    return minimal_betti(M; maxlen=maxlen, check=check)
end


"""
    minimal_bass_Rn(F::PLPolyhedra.PLFringe;
                   max_regions=10_000,
                   strict_eps=PLPolyhedra.STRICT_EPS_QQ,
                   maxlen=3,
                   check=true)

Compute Bass multiplicities from a certified minimal injective resolution on
the encoding poset P.
"""
function minimal_bass_Rn(F::PLPolyhedra.PLFringe;
                         max_regions::Int=10_000,
                         strict_eps::QQ=PLPolyhedra.STRICT_EPS_QQ,
                         maxlen::Int=3,
                         check::Bool=true)

    P, M, pi = encode_pmodule_from_PL_fringe(F;
                                            max_regions=max_regions,
                                            strict_eps=strict_eps)
    return minimal_bass(M; maxlen=maxlen, check=check)
end


# Pad a projective resolution with zeros so Ext(M,N; maxdeg=d) always has tmax=d.

function _zero_pmodule(Q::FinitePoset)
    edge = Dict{Tuple{Int,Int},Matrix{QQ}}()
    for (u,v) in cover_edges(Q)
        edge[(u,v)] = zeros(QQ, 0, 0)
    end
    return PModule{QQ}(Q, zeros(Int, Q.n), edge)
end

function _pad_projective_resolution!(res::ProjectiveResolution{QQ}, maxdeg::Int)
    L = length(res.Pmods) - 1
    if L >= maxdeg
        return
    end

    Q = res.M.Q

    for a in (L+1):maxdeg
        # Add P_a = 0 and an empty generator list.
        push!(res.Pmods, _zero_pmodule(Q))
        push!(res.gens, Int[])

        # Add the zero differential d_a : P_a -> P_{a-1} as a PMorphism.
        dom = res.Pmods[a+1]   # P_a
        cod = res.Pmods[a]     # P_{a-1}
        comps = [zeros(QQ, cod.dims[v], dom.dims[v]) for v in 1:Q.n]
        push!(res.d_mor, PMorphism{QQ}(dom, cod, comps))

        # Also pad the matrix-on-generators representation.
        cod_summands = length(res.gens[a])     # summands of P_{a-1}
        dom_summands = length(res.gens[a+1])   # summands of P_a (0)
        push!(res.d_mat, spzeros(QQ, cod_summands, dom_summands))
    end

    @assert length(res.d_mor) == length(res.Pmods) - 1
    @assert length(res.d_mat) == length(res.Pmods) - 1
    return
end



end
