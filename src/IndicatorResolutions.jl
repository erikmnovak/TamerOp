module IndicatorResolutions

# Indicator resolutions: build (F,dF) and (E,dE) data (upset/downset) and provide
# utilities for converting between presentations and PModule objects.
#
# Responsibilities:
#   - turn a finite fringe module into indicator presentations (upset/downset)
#   - build longer indicator resolutions (F,dF) and (E,dE)
#   - verification routines and small caching helpers
#
# Design notes:
#   - This module does not compute Ext itself. It supplies resolutions and minimal
#     presentation helpers consumed by `DerivedFunctors.HomExtEngine`.
#   - Derived-functor drivers that interpret resolutions (e.g. first-page diagnostics
#     and Ext dimension computations) live in `DerivedFunctors.ExtTorSpaces`.

using SparseArrays, LinearAlgebra
using ..FiniteFringe
using ..IndicatorTypes: UpsetPresentation, DownsetCopresentation
using ..ExactQQ: QQ, rrefQQ, rankQQ, rankQQ_restricted, nullspaceQQ, solve_fullcolumnQQ, colspaceQQ

using ..Modules: CoverCache, cover_cache, clear_cover_cache!, _cover_cache,
                 CoverEdgeMapStore, _find_sorted_index,
                 PModule, PMorphism, dim_at,
                 zero_pmodule, zero_morphism, map_leq, id_morphism

import ..AbelianCategories
using ..AbelianCategories: kernel_with_inclusion, _cokernel_module

import ..FiniteFringe: FinitePoset, Upset, Downset, principal_upset, principal_downset, cover_edges

# =============================================================================
# Public exports
export pmodule_from_fringe, projective_cover, injective_hull,
       upset_resolution, downset_resolution,
       indicator_resolutions,
       verify_upset_resolution, verify_downset_resolution

# ----------------------- from FringeModule to PModule ----------------------------

"""
    pmodule_from_fringe(H::FiniteFringe.FringeModule{K})
Return an internal `PModule{QQ}` whose fibers and structure maps are induced by the
fringe presentation `phi : oplus k[U_i] to oplus k[D_j]` (Defs. 3.16-3.17).
Implementation: M_q = im(phi_q) inside E_q; along a cover u<v the map is the restriction
E_u to E_v followed by projection to M_v.
"""
function pmodule_from_fringe(H::FiniteFringe.FringeModule{K}) where {K}
    Q = H.P
    n = Q.n

    # Basis for each fiber M_q as columns of a QQ matrix B[q] spanning im(phi_q).
    B = Vector{Matrix{QQ}}(undef, n)
    dims = zeros(Int, n)
    for q in 1:n
        cols = findall(U -> U.mask[q], H.U)
        rows = findall(D -> D.mask[q], H.D)
        if isempty(cols) || isempty(rows)
            B[q] = zeros(QQ, length(rows), 0)
            dims[q] = 0
            continue
        end
        phi_q = Matrix{QQ}(Matrix(H.phi[rows, cols]))
        B[q] = colspaceQQ(phi_q)
        dims[q] = size(B[q], 2)
    end

    # Death projection E_u \to E_v on a cover u<v: keep row indices j that remain active at v.
    function death_projection(u::Int, v::Int)
        rows_u = findall(D -> D.mask[u], H.D)
        rows_v = findall(D -> D.mask[v], H.D)
        pos_v = Dict{Int,Int}(rows_v[i] => i for i in 1:length(rows_v))
        P = zeros(QQ, length(rows_v), length(rows_u))
        for (jpos, jidx) in enumerate(rows_u)
            if haskey(pos_v, jidx)
                P[pos_v[jidx], jpos] = 1//1
            end
        end
        P
    end

    # Structure map on a cover u<v: M_u --incl--> E_u --proj--> E_v --coords--> M_v
    edge_maps = Dict{Tuple{Int,Int}, Matrix{QQ}}()
    C = cover_edges(Q)

    @inbounds for (u, v) in C
        du = dims[u]
        dv = dims[v]
        if du == 0 || dv == 0
            edge_maps[(u, v)] = zeros(QQ, dv, du)
        else
            Puv = death_projection(u, v)      # E_u -> E_v
            Im  = Puv * B[u]                  # in E_v coordinates
            X   = solve_fullcolumnQQ(B[v], Im)
            edge_maps[(u, v)] = X             # M_u -> M_v
        end
    end

    PModule{QQ}(Q, dims, edge_maps)
end

# -------------------------- projective cover (Def. 6.4.1) --------------------------

"Incoming image at v from immediate predecessors; basis matrix with QQ columns."
function _incoming_image_basis(M::PModule{QQ}, v::Int; cache::Union{Nothing,CoverCache}=nothing)
    dv = M.dims[v]
    pv = M.edge_maps.preds[v]
    maps = M.edge_maps.maps_from_pred[v]

    if dv == 0 || isempty(pv)
        return zeros(QQ, dv, 0)
    end

    tot = 0
    @inbounds for u in pv
        tot += M.dims[u]
    end
    if tot == 0
        return zeros(QQ, dv, 0)
    end

    # dense fast path
    if eltype(maps) <: Matrix{QQ}
        A = Matrix{QQ}(undef, dv, tot)
        col = 1
        @inbounds for i in eachindex(pv)
            u = pv[i]
            du = M.dims[u]
            if du > 0
                A[:, col:col+du-1] .= maps[i]
                col += du
            end
        end
        return colspaceQQ(A)
    end

    # fallback: sparse/abstract matrix types
    return colspaceQQ(hcat(maps...))
end



"""
    projective_cover(M::PModule{QQ})
Return (F0, pi0, gens_at) where F0 is a direct sum of principal upsets covering M,
pi0 : F0 \to M is the natural surjection, and `gens_at[v]` lists the generators activated
at vertex v (each item is a pair (p, local_index_in_Mp)).
"""
function projective_cover(M::PModule{QQ}; cache::Union{Nothing,CoverCache}=nothing)
    Q = M.Q; n = Q.n
    cc = cache === nothing ? cover_cache(Q) : cache

    # number of generators at each vertex = dim(M_v) - rank(incoming_image)
    gens_at = Vector{Vector{Tuple{Int,Int}}}(undef, n)
    gen_of_p = fill(0, n)
    for v in 1:n
        Img = _incoming_image_basis(M, v; cache=cc)
        beta = M.dims[v] - size(Img, 2)
        chosen = Int[]
        if beta > 0 && M.dims[v] > 0
            S = Img
            Id = Matrix{QQ}(I, M.dims[v], M.dims[v])
            rS = size(S, 2)
            for j in 1:M.dims[v]
                T = hcat(S, Id[:, j])
                if rankQQ(T) > rS
                    push!(chosen, j); S = colspaceQQ(T); rS += 1
                    length(chosen) == beta && break
                end
            end
        end
        gens_at[v] = [(v, j) for j in chosen]
        gen_of_p[v] = length(chosen)
    end

        # F0 as a direct sum of principal upsets.
    #
    # IMPORTANT CORRECTNESS NOTE
    # On a general finite poset, the cover-edge maps of a direct sum of principal
    # upsets are NOT given by a "rectangular identity" unless the chosen basis at
    # every vertex extends the basis at each predecessor as a prefix.
    #
    # We therefore build the cover-edge maps by matching generator labels (p,j)
    # across vertices. This agrees with the representable functor structure and is
    # independent of the arbitrary ordering of vertices.
    F0_dims = [sum(gen_of_p[p] for p in 1:n if Q.leq[p,i]) for i in 1:n]

    # Active generator lists (and positions) at each vertex i.
    active_at = Vector{Vector{Tuple{Int,Int}}}(undef, n)
    pos_at    = Vector{Dict{Tuple{Int,Int},Int}}(undef, n)
    for i in 1:n
        lst = Tuple{Int,Int}[]
        for p in 1:n
            if gen_of_p[p] > 0 && Q.leq[p,i]
                append!(lst, gens_at[p])
            end
        end
        active_at[i] = lst
        d = Dict{Tuple{Int,Int},Int}()
        for (k, g) in enumerate(lst)
            d[g] = k
        end
        pos_at[i] = d
    end

    F0_edges = Dict{Tuple{Int,Int}, Matrix{QQ}}()
    @inbounds for u in 1:n
        su = cc.succs[u]
        for v in su
            Muv = zeros(QQ, F0_dims[v], F0_dims[u])
            # Inclusion: every generator active at u is also active at v.
            for (j, g) in enumerate(active_at[u])
                i = pos_at[v][g]
                Muv[i, j] = 1//1
            end
            F0_edges[(u, v)] = Muv
        end
    end
    F0 = PModule{QQ}(Q, F0_dims, F0_edges)


    # pi0 : F0 -> M
    #
    # Preallocate each component and fill blockwise.
    # Old code used repeated hcat (allocates many temporaries).
    gen_vertices = [p for p in 1:n if gen_of_p[p] > 0]

    # Cache the chosen basis indices in each M_p once.
    J_at = Vector{Vector{Int}}(undef, n)
    for p in 1:n
        J_at[p] = Int[]
    end
    for p in gen_vertices
        J_at[p] = [pair[2] for pair in gens_at[p]]
    end

    comps = Vector{Matrix{QQ}}(undef, n)
    for i in 1:n
        Mi = M.dims[i]
        Fi = F0_dims[i]
        cols = zeros(QQ, Mi, Fi)
        col = 1
        for p in gen_vertices
            k = gen_of_p[p]
            if !Q.leq[p, i]
                continue
            end
            A = map_leq(M, p, i; cache=cc)  # M_p -> M_i
            Jp = J_at[p]
            @inbounds for t in 1:k
                j = Jp[t]
                copyto!(view(cols, :, col), view(A, :, j))
                col += 1
            end
        end
        comps[i] = cols
    end
    pi0 = PMorphism{QQ}(F0, M, comps)
    return F0, pi0, gens_at
end


# Basis of the "socle" at vertex u: kernel of the stacked outgoing map
# M_u to oplus_{u<v} M_v along cover edges u < v.
# Columns of the returned matrix span soc(M)_u subseteq M_u.
function _socle_basis(M::PModule{QQ}, u::Int; cache::Union{Nothing,CoverCache}=nothing)
    cc = (cache === nothing ? cover_cache(M.Q) : cache)
    su = cc.succs[u]
    du = M.dims[u]

    if isempty(su) || du == 0
        return Matrix{QQ}(I, du, du)
    end

    # Build the stacked outgoing map A : M_u -> (direct sum over cover successors).
    # A has size (sum_v dim(M_v)) x dim(M_u).
    tot = 0
    @inbounds for j in eachindex(su)
        tot += M.dims[su[j]]
    end
    if tot == 0
        # No nonzero target fibers; outgoing map is zero, so socle is all of M_u.
        return Matrix{QQ}(I, du, du)
    end

    A = Matrix{QQ}(undef, tot, du)
    row = 1
    maps_u = M.edge_maps.maps_to_succ[u]

    @inbounds for j in eachindex(su)
        v = su[j]
        dv = M.dims[v]
        if dv > 0
            A[row:row+dv-1, :] .= maps_u[j]
            row += dv
        end
    end

    return nullspaceQQ(A)  # columns span the socle at u
end


# A canonical left-inverse for a full-column-rank matrix S: L*S = I.
# Implemented as L = (S^T S)^{-1} S^T using exact QQ solves.
function _left_inverse_full_column(S::AbstractMatrix{QQ})
    s = size(S,2)
    if s == 0
        return zeros(QQ, 0, size(S,1))
    end
    G = transpose(S) * S                       # s*s Gram matrix, invertible over QQ
    return solve_fullcolumnQQ(G, transpose(S)) # returns (S^T S)^{-1} S^T with size s * m
end

# Build the injective (downset) hull:  iota : M into E  where
# E is a direct sum of principal downsets with multiplicities = socle dimensions.
# Also return the generator labels as (u, j) with u the vertex and j the column.
function _injective_hull(M::PModule{QQ}; cache::Union{Nothing,CoverCache}=nothing)
    Q = M.Q; n = Q.n
    cc = cache === nothing ? cover_cache(Q) : cache

    # socle bases at each vertex and their multiplicities
    Soc = Vector{Matrix{QQ}}(undef, n)
    mult = zeros(Int, n)
    for u in 1:n
        Soc[u]  = _socle_basis(M, u; cache=cc)
        mult[u] = size(Soc[u], 2)
    end

        # generator labels for the chosen downset summands
    gens_at = Vector{Vector{Tuple{Int,Int}}}(undef, n)
    for u in 1:n
        gens_at[u] = [(u, j) for j in 1:mult[u]]
    end

    # fiber dimensions of E
    Edims = [sum(mult[u] for u in 1:n if Q.leq[i,u]) for i in 1:n]

    # Active generator lists (and positions) at each vertex i.
    # This ordering matches the row-stacking order used in the inclusion iota below.
    active_at = Vector{Vector{Tuple{Int,Int}}}(undef, n)
    pos_at    = Vector{Dict{Tuple{Int,Int},Int}}(undef, n)
    for i in 1:n
        lst = Tuple{Int,Int}[]
        for u in 1:n
            if mult[u] > 0 && Q.leq[i,u]
                append!(lst, gens_at[u])
            end
        end
        active_at[i] = lst
        d = Dict{Tuple{Int,Int},Int}()
        for (k, g) in enumerate(lst)
            d[g] = k
        end
        pos_at[i] = d
    end

    # E structure maps along cover edges u<v are coordinate projections:
    # keep exactly those generators that are still active at v.
    Eedges = Dict{Tuple{Int,Int}, Matrix{QQ}}()
    @inbounds for u in 1:n
        su = cc.succs[u]
        for v in su
            Muv = zeros(QQ, Edims[v], Edims[u])
            for (i, g) in enumerate(active_at[v])
                j = pos_at[u][g]
                Muv[i, j] = 1//1
            end
            Eedges[(u, v)] = Muv
        end
    end
    E = PModule{QQ}(Q, Edims, Eedges)

    # iota : M -> E
    Linv = [ _left_inverse_full_column(Soc[u]) for u in 1:n ]

    # Only vertices with nontrivial socle contribute rows.
    mult_vertices = [u for u in 1:n if mult[u] > 0]

    comps = Vector{Matrix{QQ}}(undef, n)
    for i in 1:n
        rows = zeros(QQ, Edims[i], M.dims[i])
        r = 1
        for u in mult_vertices
            if Q.leq[i, u]
                Mi_to_Mu = map_leq(M, i, u; cache=cc)
                m = mult[u]
                @views mul!(rows[r:r+m-1, :], Linv[u], Mi_to_Mu)
                r += m
            end
        end
        @assert r == Edims[i] + 1
        comps[i] = rows
    end
    iota = PMorphism{QQ}(M, E, comps)

    return E, iota, gens_at
end

"""
    upset_presentation_one_step(Hfringe::FringeModule)
Compute the one-step upset presentation (Def. 6.4.1):
    F1 --d1--> F0 --pi0-->> M,
and return the lightweight wrapper `UpsetPresentation{QQ}(P, U0, U1, delta, H)`.
"""
function upset_presentation_one_step(H::FiniteFringe.FringeModule)
    M = pmodule_from_fringe(H)          # internal PModule over QQ
    cc = _cover_cache(M.Q)
    # First step: projective cover of M
    F0, pi0, gens_at_F0 = projective_cover(M; cache=cc)
    # Kernel K1 with inclusion i1 : K1 \into F0
    K1, i1 = kernel_with_inclusion(pi0; cache=cc)
    # Second projective cover (of K1)
    F1, pi1, gens_at_F1 = projective_cover(K1; cache=cc)
    # Differential d1 = i1 \circ pi1 : F1 \to F0
    comps = [i1.comps[i] * pi1.comps[i] for i in 1:length(M.dims)]
    d1 = PMorphism{QQ}(F1, F0, comps)

    # Build indicator wrapper:
    # U0: list of principal upsets, one per generator in gens_at_F0[p]
    P = M.Q
    U0 = Upset[]
    for p in 1:P.n
        for _ in gens_at_F0[p]
            push!(U0, principal_upset(P, p))
        end
    end
    U1 = Upset[]
    for p in 1:P.n
        for _ in gens_at_F1[p]
            push!(U1, principal_upset(P, p))
        end
    end

    # scalar delta block: one entry for each pair (theta in U1, lambda in U0) if ptheta <= plambda.
    # Extract the scalar from d1 at the minimal vertex i = plambda.
    m1 = length(U1); m0 = length(U0)
    delta = spzeros(QQ, m1, m0)
    # local index maps: at vertex i, Fk_i basis is "all generators at p with p <= i" in (p increasing) order.
    # Build offsets to find local coordinates.
    function local_index_list(gens_at)
        # return vector of vectors L[i] listing global generator indices active at vertex i
        L = Vector{Vector{Tuple{Int,Int}}}(undef, P.n)
        for i in 1:P.n
            L[i] = Tuple{Int,Int}[]
            for p in 1:P.n
                if P.leq[p,i]
                    append!(L[i], gens_at[p])
                end
            end
        end
        L
    end
    L0 = local_index_list(gens_at_F0)
    L1 = local_index_list(gens_at_F1)

    # Build a map from "global generator number in U0/U1" to its vertex p and position j in M_p
    globalU0 = Tuple{Int,Int}[]  # (p,j)
    for p in 1:P.n; append!(globalU0, gens_at_F0[p]); end
    globalU1 = Tuple{Int,Int}[]
    for p in 1:P.n; append!(globalU1, gens_at_F1[p]); end

    # helper: find local column index of global generator g=(p,j) at vertex i
    function local_col_of(L, g::Tuple{Int,Int}, i::Int)
        for (c, gg) in enumerate(L[i])
            if gg == g; return c; end
        end
        return 0
    end

    for (lambda, (plambda, jlambda)) in enumerate(globalU0)
        for (theta, (ptheta, jtheta)) in enumerate(globalU1)
            if P.leq[plambda, ptheta]   # containment for principal upsets: Up(ptheta) subseteq Up(plambda)
                i = ptheta              # read at the minimal vertex where the domain generator exists
                col = local_col_of(L1, (ptheta, jtheta), i)
                row = local_col_of(L0, (plambda, jlambda), i)
                if col > 0 && row > 0
                    val = d1.comps[i][row, col]
                    if val != 0
                        delta[theta, lambda] = val
                    end
                end
            end

        end
    end

    UpsetPresentation{QQ}(P, U0, U1, delta, H)
end

# ------------------------ downset copresentation (Def. 6.4.2) ------------------------

# The dual story: compute an injective hull E0 of M and the next step E1 with rho0 : E0 \to E1.
# For brevity we implement the duals by applying the above steps to M^op using
# down-closures/up-closures symmetry. Here we implement directly degreewise.

"Outgoing coimage at u to immediate successors; basis for the span of maps M_u to oplus_{u<v} M_v."
function _outgoing_span_basis(M::PModule{QQ}, u::Int; cache::Union{Nothing,CoverCache}=nothing)
    cc = (cache === nothing ? cover_cache(M.Q) : cache)
    su = cc.succs[u]
    du = M.dims[u]

    if isempty(su) || du == 0
        return zeros(QQ, 0, du)
    end

    # Stack outgoing maps B : M_u -> (direct sum over cover successors).
    tot = 0
    @inbounds for j in eachindex(su)
        tot += M.dims[su[j]]
    end
    if tot == 0
        return zeros(QQ, 0, du)
    end

    B = Matrix{QQ}(undef, tot, du)
    row = 1
    maps_u = M.edge_maps.maps_to_succ[u]

    @inbounds for j in eachindex(su)
        v = su[j]
        dv = M.dims[v]
        if dv > 0
            B[row:row+dv-1, :] .= maps_u[j]
            row += dv
        end
    end

    # Return a basis for the row space of B as an r x du matrix with full row rank.
    return transpose(colspaceQQ(transpose(B)))
end


"Essential socle dimension at u = dim(M_u) - rank(outgoing span) (dual of generators)."
function _socle_count(M::PModule{QQ}, u::Int)
    S = _outgoing_span_basis(M, u)
    M.dims[u] - rankQQ(S)
end

"""
    downset_copresentation_one_step(Hfringe::FringeModule)

Compute the one-step downset **copresentation** (Def. 6.4(2)):
    M = ker(rho : E^0 to E^1),
with E^0 and E^1 expressed as direct sums of principal downsets and rho assembled
from the actual vertexwise maps, not just from the partial order.  Steps:

1. Build the injective (downset) hull iota0 : M into E^0.
2. Form C = coker(iota0) as a P-module together with q : E^0 to C.
3. Build the injective (downset) hull j : C into E^1 and set rho0 = j circ q : E^0 -> E^1.
4. Read scalar entries of rho at minimal vertices, as for delta on the upset side.
"""
function downset_copresentation_one_step(H::FiniteFringe.FringeModule)
    # Convert fringe to internal PModule over QQ
    M = pmodule_from_fringe(H)
    Q = M.Q; n = Q.n
    cc = _cover_cache(Q)

    # (1) Injective hull of M: E0 with inclusion iota0
    E0, iota0, gens_at_E0 = _injective_hull(M; cache=cc)

    # (2) Degreewise cokernel C and the quotient q : E0 \to C
    C, q = _cokernel_module(iota0; cache=cc)

    # (3) Injective hull of C: E1 with inclusion j : C \into E1
    E1, j, gens_at_E1 = _injective_hull(C; cache=cc)

    # Compose to get rho0 : E0 \to E1 (at each vertex i: rho0[i] = j[i] * q[i])
    comps_rho0 = [ j.comps[i] * q.comps[i] for i in 1:n ]
    rho0 = PMorphism{QQ}(E0, E1, comps_rho0)

    # (4) Assemble the indicator wrapper: labels D0, D1 and the scalar block rho.
    # D0/D1 each contain one principal downset for every generator (u,j) chosen above.
    D0 = Downset[]
    for u in 1:n, _ in gens_at_E0[u]
        push!(D0, principal_downset(Q, u))
    end
    D1 = Downset[]
    for u in 1:n, _ in gens_at_E1[u]
        push!(D1, principal_downset(Q, u))
    end

    # Local index lists: at vertex i, the active generators are those born at u with i <= u.
    function _local_index_list_D(gens_at)
        L = Vector{Vector{Tuple{Int,Int}}}(undef, n)
        for i in 1:n
            lst = Tuple{Int,Int}[]
            for u in 1:n
                if Q.leq[i,u]
                    append!(lst, gens_at[u])
                end
            end
            L[i] = lst
        end
        L
    end
    L0 = _local_index_list_D(gens_at_E0)
    L1 = _local_index_list_D(gens_at_E1)

    # Global enumerations (lambda in D0, theta in D1) with their birth vertices u_lambda, u_theta.
    globalD0 = Tuple{Int,Int}[]; for u in 1:n; append!(globalD0, gens_at_E0[u]); end
    globalD1 = Tuple{Int,Int}[]; for u in 1:n; append!(globalD1, gens_at_E1[u]); end

    # Helper to find the local column index of a global generator at vertex i
    function _local_col_of(L, g::Tuple{Int,Int}, i::Int)
        for (c, gg) in enumerate(L[i])
            if gg == g; return c; end
        end
        return 0
    end

    # Assemble the scalar monomial matrix rho by reading the minimal vertex i = u_theta.
    m1 = length(globalD1); m0 = length(globalD0)
    rho = spzeros(QQ, m1, m0)

    for (lambda, (ulambda, jlambda)) in enumerate(globalD0)
        for (theta, (utheta, jtheta)) in enumerate(globalD1)
            if Q.leq[utheta, ulambda] # D(utheta) subseteq D(ulambda)
                i   = utheta
                col = _local_col_of(L0, (ulambda, jlambda), i)
                row = _local_col_of(L1, (utheta, jtheta), i)
                if col > 0 && row > 0
                    val = rho0.comps[i][row, col]
                    if val != 0
                        rho[theta, lambda] = val
                    end
                end
            end
        end
    end

    return DownsetCopresentation{QQ}(Q, D0, D1, rho, H)
end



"""
    prune_zero_relations(F::UpsetPresentation{QQ}) -> UpsetPresentation{QQ}

Remove rows of `delta` that are identically zero (redundant relations) and drop the
corresponding entries of `U1`. The cokernel is unchanged.
"""
function prune_zero_relations(F::UpsetPresentation{QQ})
    m1, m0 = size(F.delta)
    keep = trues(m1)
    # mark zero rows
    rows, _, _ = findnz(F.delta)
    seen = falses(m1); @inbounds for r in rows; seen[r] = true; end
    @inbounds for r in 1:m1
        if !seen[r]; keep[r] = false; end
    end
    new_U1 = [F.U1[i] for i in 1:m1 if keep[i]]
    new_delta = F.delta[keep, :]
    UpsetPresentation{QQ}(F.P, F.U0, new_U1, new_delta, F.H)
end

"""
    cancel_isolated_unit_pairs(F::UpsetPresentation{QQ}) -> UpsetPresentation{QQ}

Iteratively cancels isolated nonzero entries `delta[theta,lambda]` for which:
  * the theta-th row has exactly that one nonzero,
  * the lambda-th column has exactly that one nonzero, and
  * U1[theta] == U0[lambda] as Upsets (principal upsets match).

Each cancellation removes one generator in U0 and one relation in U1 without
changing the cokernel.
"""
function cancel_isolated_unit_pairs(F::UpsetPresentation{QQ})
    P, U0, U1, Delta = F.P, F.U0, F.U1, F.delta
    while true
        m1, m0 = size(Delta)
        rows, cols, _ = findnz(Delta)
        # count nonzeros per row/col
        rcount = zeros(Int, m1)
        ccount = zeros(Int, m0)
        @inbounds for k in eachindex(rows)
            rcount[rows[k]] += 1; ccount[cols[k]] += 1
        end
        # search an isolated pair with matching principal upsets
        found = false
        theta = 0; lambda = 0
        @inbounds for k in eachindex(rows)
            r = rows[k]; c = cols[k]
            if rcount[r] == 1 && ccount[c] == 1
                # require identical principal upsets
                if U1[r].P === U0[c].P && U1[r].mask == U0[c].mask
                    theta, lambda = r, c; found = true; break
                end
            end
        end
        if !found; break; end
        # remove row theta and column lambda
        keep_rows = trues(m1); keep_rows[theta] = false
        keep_cols = trues(m0); keep_cols[lambda] = false
        U1 = [U1[i] for i in 1:m1 if keep_rows[i]]
        U0 = [U0[j] for j in 1:m0 if keep_cols[j]]
        Delta = Delta[keep_rows, keep_cols]
    end
    UpsetPresentation{QQ}(P, U0, U1, Delta, F.H)
end

"""
    minimal_upset_presentation_one_step(H::FiniteFringe.FringeModule)
        -> UpsetPresentation{QQ}

Build a one-step upset presentation and apply safe minimality passes:
1) drop zero relations; 2) cancel isolated isomorphism pairs.
"""
function minimal_upset_presentation_one_step(H::FiniteFringe.FringeModule)
    F = upset_presentation_one_step(H)     # existing builder
    F = prune_zero_relations(F)
    F = cancel_isolated_unit_pairs(F)
    return F
end


"""
    prune_unused_targets(E::DownsetCopresentation{QQ}) -> DownsetCopresentation{QQ}

Drop rows of `rho` that are identically zero (unused target summands in E^1). The kernel is unchanged.
"""
function prune_unused_targets(E::DownsetCopresentation{QQ})
    m1, m0 = size(E.rho)
    keep = trues(m1)
    rows, _, _ = findnz(E.rho)
    seen = falses(m1); @inbounds for r in rows; seen[r] = true; end
    @inbounds for r in 1:m1
        if !seen[r]; keep[r] = false; end
    end
    new_D1 = [E.D1[i] for i in 1:m1 if keep[i]]
    new_rho = E.rho[keep, :]
    DownsetCopresentation{QQ}(E.P, E.D0, new_D1, new_rho, E.H)
end

"""
    cancel_isolated_unit_pairs(E::DownsetCopresentation{QQ}) -> DownsetCopresentation{QQ}

Iteratively cancels isolated nonzero entries `rho[theta,lambda]` with matching principal downsets
(D1[theta] == D0[lambda]) and unique in their row/column.
"""
function cancel_isolated_unit_pairs(E::DownsetCopresentation{QQ})
    P, D0, D1, R = E.P, E.D0, E.D1, E.rho
    while true
        m1, m0 = size(R)
        rows, cols, _ = findnz(R)
        rcount = zeros(Int, m1)
        ccount = zeros(Int, m0)
        @inbounds for k in eachindex(rows)
            rcount[rows[k]] += 1; ccount[cols[k]] += 1
        end
        found = false; theta = 0; lambda = 0
        @inbounds for k in eachindex(rows)
            r = rows[k]; c = cols[k]
            if rcount[r] == 1 && ccount[c] == 1
                if D1[r].P === D0[c].P && D1[r].mask == D0[c].mask
                    theta, lambda = r, c; found = true; break
                end
            end
        end
        if !found; break; end
        keep_rows = trues(m1); keep_rows[theta] = false
        keep_cols = trues(m0); keep_cols[lambda] = false
        D1 = [D1[i] for i in 1:m1 if keep_rows[i]]
        D0 = [D0[j] for j in 1:m0 if keep_cols[j]]
        R = R[keep_rows, keep_cols]
    end
    DownsetCopresentation{QQ}(P, D0, D1, R, E.H)
end

"""
    minimal_downset_copresentation_one_step(H::FiniteFringe.FringeModule)
        -> DownsetCopresentation{QQ}

Build a one-step downset copresentation and apply safe minimality passes:
1) drop zero target rows; 2) cancel isolated isomorphism pairs.
"""
function minimal_downset_copresentation_one_step(H::FiniteFringe.FringeModule)
    E = downset_copresentation_one_step(H)
    E = prune_unused_targets(E)
    E = cancel_isolated_unit_pairs(E)
    return E
end



# =============================================================================
# Longer indicator resolutions and high-level Ext driver
# =============================================================================
# We expose:
#   * upset_resolution(H; maxlen)     -> (F, dF)
#   * downset_resolution(H; maxlen)   -> (E, dE)
#   * indicator_resolutions(HM, HN; maxlen) -> (F, dF, E, dE)
#
# The outputs (F, dF, E, dE) are exactly the shapes expected by
# HomExt.build_hom_tot_complex / HomExt.ext_dims_via_resolutions:
#   - F is a Vector{UpsetPresentation{QQ}} with F[a+1].U0 = U_a
#   - dF[a] is the sparse delta_a : U_a <- U_{a+1}  (shape |U_{a+1}| x |U_a|)
#   - E is a Vector{DownsetCopresentation{QQ}} with E[b+1].D0 = D_b
#   - dE[b] is the sparse rho_b : D_b -> D_{b+1}    (shape |D_{b+1}| x |D_b|)
#
# Construction mirrors section 6.1 and the one-step routines already present.
# =============================================================================

# ------------------------------ small helpers --------------------------------

# Build the list of principal upsets from per-vertex generator labels returned by
# projective_cover: gens_at[v] is a vector of pairs (p, j).  Each pair contributes
# one principal upset at vertex p.
function _principal_upsets_from_gens(P::FinitePoset,
                                     gens_at::Vector{Vector{Tuple{Int,Int}}})
    U = Upset[]
    for p in 1:P.n
        for _ in gens_at[p]
            push!(U, principal_upset(P, p))
        end
    end
    U
end

# Build the list of principal downsets from per-vertex labels returned by _injective_hull
function _principal_downsets_from_gens(P::FinitePoset,
                                       gens_at::Vector{Vector{Tuple{Int,Int}}})
    D = Downset[]
    for u in 1:P.n
        for _ in gens_at[u]
            push!(D, principal_downset(P, u))
        end
    end
    D
end

# For upset side: at vertex i, which global generators are active (born at p <= i)?
# Returns L[i] = vector of global generator labels (p,j) visible at i.
function _local_index_list_up(P::FinitePoset,
                              gens_at::Vector{Vector{Tuple{Int,Int}}})
    L = Vector{Vector{Tuple{Int,Int}}}(undef, P.n)
    for i in 1:P.n
        lst = Tuple{Int,Int}[]
        for p in 1:P.n
            if P.leq[p,i]
                append!(lst, gens_at[p])
            end
        end
        L[i] = lst
    end
    L
end

# For downset side: at vertex i, which global generators are active (born at u with i <= u)?
# Returns L[i] = vector of global generator labels (u,j) visible at i.
function _local_index_list_down(P::FinitePoset,
                                gens_at::Vector{Vector{Tuple{Int,Int}}})
    L = Vector{Vector{Tuple{Int,Int}}}(undef, P.n)
    for i in 1:P.n
        lst = Tuple{Int,Int}[]
        for u in 1:P.n
            if P.leq[i,u]
                append!(lst, gens_at[u])
            end
        end
        L[i] = lst
    end
    L
end

# Find the local column index (1-based) of a global generator g=(p,j) or (u,j) in L[i].
# Returns 0 if not present.
function _local_col_of(L::Vector{Vector{Tuple{Int,Int}}}, g::Tuple{Int,Int}, i::Int)
    for (c, gg) in enumerate(L[i])
        if gg == g; return c; end
    end
    return 0
end

# Dense -> sparse helper over QQ (already have a general helper in FiniteFringe, but here
# we build directly from triplets to avoid materializing full dense blocks).
function _empty_sparse_QQ(nr::Int, nc::Int)
    return spzeros(QQ, nr, nc)
end

# ------------------------------ upset resolution ------------------------------

"""
    upset_resolution(H::FiniteFringe.FringeModule{QQ}; maxlen=nothing)

Compute an upset (projective) indicator resolution of the fringe module `H`.
`maxlen` is a cutoff on the number of differentials computed (it does not pad output).
"""
function upset_resolution(H::FiniteFringe.FringeModule{QQ}; maxlen::Union{Int,Nothing}=nothing)
    return upset_resolution(pmodule_from_fringe(H); maxlen=maxlen)
end


"""
upset_resolution(M::PModule{QQ}; maxlen::Union{Int,Nothing}=nothing)
        -> (F::Vector{UpsetPresentation{QQ}}, dF::Vector{SparseMatrixCSC{QQ,Int}})

Overload of `upset_resolution` for an already-constructed finite-poset module `M`.

This is the same construction as `upset_resolution(::FiniteFringe.FringeModule{QQ})`,
but it skips the `pmodule_from_fringe` conversion step. This matters when callers
already have a `PModule{QQ}` (for example, after encoding a Z^n or R^n module to a
finite poset, or after explicitly calling `pmodule_from_fringe`).
"""
function upset_resolution(M::PModule{QQ}; maxlen::Union{Int,Nothing}=nothing)
    P = M.Q

    # First projective cover: F0 --pi0--> M, with labels gens_at_F0
    F0, pi0, gens_at_F0 = projective_cover(M)

    U_by_a = Vector{Vector{Upset}}()
    push!(U_by_a, _principal_upsets_from_gens(P, gens_at_F0))

    dF = Vector{SparseMatrixCSC{QQ,Int}}()

    # Iteration state
    curr_dom  = F0
    curr_pi   = pi0
    curr_gens = gens_at_F0
    steps = 0

    while true
        if maxlen !== nothing && steps >= maxlen
            break
        end

        # Next kernel and projective cover
        K, iota = kernel_with_inclusion(curr_pi)   # K = ker(F_prev -> ...)
        if sum(K.dims) == 0
            break
        end
        Fnext, pinext, gens_at_next = projective_cover(K)

        # d = iota o pinext : Fnext -> curr_dom
        comps = Vector{Matrix{QQ}}(undef, P.n)
        for i in 1:P.n
            comps[i] = iota.comps[i] * pinext.comps[i]
        end
        d = PMorphism{QQ}(Fnext, curr_dom, comps)

        # Labels and local index lists
        U_next = _principal_upsets_from_gens(P, gens_at_next)
        Lprev  = _local_index_list_up(P, curr_gens)
        Lnext  = _local_index_list_up(P, gens_at_next)

        # Global enumerations of generators (record their birth vertices)
        global_prev = Tuple{Int,Int}[]
        for p in 1:P.n
            append!(global_prev, curr_gens[p])
        end
        global_next = Tuple{Int,Int}[]
        for p in 1:P.n
            append!(global_next, gens_at_next[p])
        end

        # Assemble sparse delta: rows index next, cols index prev
        delta = _empty_sparse_QQ(length(global_next), length(global_prev))
        for (lambda, (plambda, jlambda)) in enumerate(global_prev)
            for (theta, (ptheta, jtheta)) in enumerate(global_next)
                # Containment for principal upsets: Up(ptheta) subseteq Up(plambda)
                if P.leq[plambda, ptheta]
                    # Read at the minimal vertex where the domain generator exists
                    i = ptheta
                    col = _local_col_of(Lnext, (ptheta, jtheta), i)
                    row = _local_col_of(Lprev, (plambda, jlambda), i)
                    if col > 0 && row > 0
                        val = d.comps[i][row, col]
                        if val != 0
                            delta[theta, lambda] = val
                        end
                    end
                end
            end
        end

        push!(dF, delta)
        push!(U_by_a, U_next)

        # Advance
        curr_dom  = Fnext
        curr_pi   = pinext
        curr_gens = gens_at_next
        steps += 1
    end

    # Package as UpsetPresentation list
    F = Vector{UpsetPresentation{QQ}}(undef, length(U_by_a))
    for a in 1:length(U_by_a)
        U0 = U_by_a[a]
        if a < length(U_by_a)
            U1 = U_by_a[a+1]
            delta = dF[a]
        else
            U1 = Upset[]
            delta = spzeros(QQ, 0, length(U0))
        end
        F[a] = UpsetPresentation{QQ}(P, U0, U1, delta, nothing)
    end

    return F, dF
end


# ---------------------------- downset resolution ------------------------------

"""
    downset_resolution(H::FiniteFringe.FringeModule{QQ}; maxlen=nothing)

Compute a downset (injective) indicator resolution of the fringe module `H`.
`maxlen` is a cutoff on the number of differentials computed (it does not pad output).
"""
function downset_resolution(H::FiniteFringe.FringeModule{QQ}; maxlen::Union{Int,Nothing}=nothing)
    return downset_resolution(pmodule_from_fringe(H); maxlen=maxlen)
end


"""
downset_resolution(M::PModule{QQ}; maxlen::Union{Int,Nothing}=nothing)
        -> (E::Vector{DownsetCopresentation{QQ}}, dE::Vector{SparseMatrixCSC{QQ,Int}})

Overload of `downset_resolution` for an already-constructed finite-poset module `M`.

This is the same construction as `downset_resolution(::FiniteFringe.FringeModule{QQ})`,
but it skips the `pmodule_from_fringe` conversion step.
"""
function downset_resolution(M::PModule{QQ}; maxlen::Union{Int,Nothing}=nothing)
    P = M.Q

    # First injective hull: iota0 : M -> E0
    E0, iota0, gens_at_E0 = _injective_hull(M)
    D_by_b = Vector{Vector{Downset}}()
    push!(D_by_b, _principal_downsets_from_gens(P, gens_at_E0))

    dE = Vector{SparseMatrixCSC{QQ,Int}}()

    # First cokernel
    C, q = _cokernel_module(iota0)

    # Iteration state
    prev_E    = E0
    prev_gens = gens_at_E0
    prev_q    = q
    steps = 0

    while true
        if sum(C.dims) == 0
            break
        end
        if maxlen !== nothing && steps >= maxlen
            break
        end

        # Injective hull of the current cokernel: j : C -> E1
        E1, j, gens_at_E1 = _injective_hull(C)

        # rho_b = j o prev_q : prev_E -> E1
        comps = Vector{Matrix{QQ}}(undef, P.n)
        for i in 1:P.n
            comps[i] = j.comps[i] * prev_q.comps[i]
        end
        rho = PMorphism{QQ}(prev_E, E1, comps)

        # Labels and local index lists
        D_next = _principal_downsets_from_gens(P, gens_at_E1)
        L0     = _local_index_list_down(P, prev_gens)
        L1     = _local_index_list_down(P, gens_at_E1)

        globalD0 = Tuple{Int,Int}[]
        for u in 1:P.n
            append!(globalD0, prev_gens[u])
        end
        globalD1 = Tuple{Int,Int}[]
        for u in 1:P.n
            append!(globalD1, gens_at_E1[u])
        end

        # Assemble sparse rho: rows index D1 (next), cols index D0 (prev)
        Rh = _empty_sparse_QQ(length(globalD1), length(globalD0))
        for (lambda, (ulambda, jlambda)) in enumerate(globalD0)      # col in D0
            for (theta,  (utheta,  jtheta))  in enumerate(globalD1)  # row in D1
                # Containment for principal downsets: D(utheta) subseteq D(ulambda)
                if P.leq[utheta, ulambda]
                    i   = utheta
                    col = _local_col_of(L0, (ulambda, jlambda), i)
                    row = _local_col_of(L1, (utheta,  jtheta),  i)
                    if col > 0 && row > 0
                        val = rho.comps[i][row, col]
                        if val != 0
                            Rh[theta, lambda] = val
                        end
                    end
                end
            end
        end

        push!(dE, Rh)
        push!(D_by_b, D_next)

        # Next cokernel and advance
        C, q_next = _cokernel_module(j)
        prev_E    = E1
        prev_gens = gens_at_E1
        prev_q    = q_next
        steps += 1
    end

    # Package as DownsetCopresentation list
    E = Vector{DownsetCopresentation{QQ}}(undef, length(D_by_b))
    for b in 1:length(D_by_b)
        D0 = D_by_b[b]
        if b < length(D_by_b)
            D1 = D_by_b[b+1]
            rho_b = dE[b]
        else
            D1 = Downset[]
            rho_b = spzeros(QQ, 0, length(D0))
        end
        E[b] = DownsetCopresentation{QQ}(P, D0, D1, rho_b, nothing)
    end

    return E, dE
end


# --------------------- aggregator + high-level Ext driver ---------------------

"""
    indicator_resolutions(HM, HN; maxlen=nothing)
        -> (F, dF, E, dE)

Convenience wrapper: build an upset resolution for the source module (fringe HM)
and a downset resolution for the target module (fringe HN).  The `maxlen` keyword
cuts off each side after that many steps (useful for quick tests).
"""
function indicator_resolutions(HM::FiniteFringe.FringeModule{QQ},
                               HN::FiniteFringe.FringeModule{QQ};
                               maxlen::Union{Int,Nothing}=nothing)
    F, dF = upset_resolution(HM; maxlen=maxlen)
    E, dE = downset_resolution(HN; maxlen=maxlen)
    return F, dF, E, dE
end

# --------------------- resolution verification (structural checks) ---------------------

"""
    verify_upset_resolution(F, dF; vertices=:all,
                            check_d2=true,
                            check_exactness=true,
                            check_connected=true)

Structural checks for an upset (projective) indicator resolution:
  * d^2 = 0 (as coefficient matrices),
  * exactness at intermediate stages (vertexwise),
  * each nonzero entry corresponds to a connected homomorphism in the principal-upset case,
    i.e. support inclusion U_{a+1} subseteq U_a.

Throws an error if a check fails; returns true otherwise.
"""
function verify_upset_resolution(F::Vector{UpsetPresentation{QQ}},
                                dF::Vector{SparseMatrixCSC{QQ,Int}};
                                vertices = :all,
                                check_d2::Bool = true,
                                check_exactness::Bool = true,
                                check_connected::Bool = true)

    @assert length(dF) == length(F) - 1 "verify_upset_resolution: expected length(dF) == length(F)-1"
    P = F[1].P
    for f in F
        @assert f.P === P "verify_upset_resolution: all presentations must use the same poset"
    end

    U_by_a = [f.U0 for f in F]  # U_0,...,U_A
    vs = (vertices === :all) ? (1:P.n) : vertices


    # Connectedness / valid monomial support (principal-upset case):
    # a nonzero entry for k[Udom] -> k[Ucod] is only allowed when Udom subseteq Ucod.
    if check_connected
        for a in 1:length(dF)
            delta = dF[a]                 # rows: U_{a+1}, cols: U_a
            Udom  = U_by_a[a+1]
            Ucod  = U_by_a[a]
            for col in 1:size(delta,2)
                for ptr in delta.colptr[col]:(delta.colptr[col+1]-1)
                    row = delta.rowval[ptr]
                    val = delta.nzval[ptr]
                    if !iszero(val) && !FiniteFringe.is_subset(Udom[row], Ucod[col])
                        error("Upset resolution: nonzero delta at (row=$row,col=$col) but U_{a+1}[row] not subset of U_a[col] (a=$(a-1))")
                    end
                end
            end
        end
    end

    # d^2 = 0
    if check_d2 && length(dF) >= 2
        for a in 1:(length(dF)-1)
            C = dF[a+1] * dF[a]
            dropzeros!(C)
            if nnz(C) != 0
                error("Upset resolution: dF[$(a+1)]*dF[$a] != 0 (nnz=$(nnz(C)))")
            end
        end
    end

    # Vertexwise exactness at intermediate stages:
    # For each q, check rank(d_{a}) + rank(d_{a+1}) = dim(F_a(q)).
    if check_exactness && length(dF) >= 2
        for q in vs
            for k in 2:(length(U_by_a)-1)  # check exactness at U_k (k=1..A-1)
                active_k   = findall(u -> u.mask[q], U_by_a[k])
                dim_mid = length(active_k)
                if dim_mid == 0
                    continue
                end
                active_km1 = findall(u -> u.mask[q], U_by_a[k-1])
                active_kp1 = findall(u -> u.mask[q], U_by_a[k+1])

                delta_prev = dF[k-1]  # U_k -> U_{k-1}
                delta_next = dF[k]    # U_{k+1} -> U_k

                r_prev = isempty(active_km1) ? 0 : rankQQ_restricted(delta_prev, active_k, active_km1)
                r_next = isempty(active_kp1) ? 0 : rankQQ_restricted(delta_next, active_kp1, active_k)

                if r_prev + r_next != dim_mid
                    error("Upset resolution: vertex q=$q fails exactness at degree k=$(k-1): rank(prev)=$r_prev, rank(next)=$r_next, dim=$dim_mid")
                end
            end
        end
    end

    return true
end


"""
    verify_downset_resolution(E, dE; vertices=:all,
                              check_d2=true,
                              check_exactness=true,
                              check_connected=true)

Structural checks for a downset (injective) indicator resolution:
  * d^2 = 0,
  * exactness at intermediate stages (vertexwise),
  * valid monomial support (principal-downset case): nonzero entry implies D_b subseteq D_{b+1}.

Throws an error if a check fails; returns true otherwise.
"""
function verify_downset_resolution(E::Vector{DownsetCopresentation{QQ}},
                                  dE::Vector{SparseMatrixCSC{QQ,Int}};
                                  vertices = :all,
                                  check_d2::Bool = true,
                                  check_exactness::Bool = true,
                                  check_connected::Bool = true)

    @assert length(dE) == length(E) - 1 "verify_downset_resolution: expected length(dE) == length(E)-1"
    P = E[1].P
    for e in E
        @assert e.P === P "verify_downset_resolution: all copresentations must use the same poset"
    end

    D_by_b = [e.D0 for e in E]  # D_0,...,D_B
    vs = (vertices === :all) ? (1:P.n) : vertices


    # Valid monomial support (principal-downset case):
    # rho has rows in D_{b+1} and cols in D_b, so nonzero implies D_b subseteq D_{b+1}.
    if check_connected
        for b in 1:length(dE)
            rho = dE[b]                 # rows: D_{b+1}, cols: D_b
            Ddom = D_by_b[b]
            Dcod = D_by_b[b+1]
            for col in 1:size(rho,2)
                for ptr in rho.colptr[col]:(rho.colptr[col+1]-1)
                    row = rho.rowval[ptr]
                    val = rho.nzval[ptr]
                    if !iszero(val) && !FiniteFringe.is_subset(Dcod[row], Ddom[col])
                        error("Downset resolution: nonzero rho at (row=$row,col=$col) but D_{b+1}[row] not subset of D_b[col] (b=$(b-1))")
                    end
                end
            end
        end
    end

    # d^2 = 0
    if check_d2 && length(dE) >= 2
        for b in 1:(length(dE)-1)
            C = dE[b+1] * dE[b]
            dropzeros!(C)
            if nnz(C) != 0
                error("Downset resolution: dE[$(b+1)]*dE[$b] != 0 (nnz=$(nnz(C)))")
            end
        end
    end

    # Vertexwise exactness at intermediate stages:
    if check_exactness && length(dE) >= 2
        for q in vs
            for k in 2:(length(D_by_b)-1)  # check exactness at D_k (k=1..B-1)
                active_k   = findall(d -> d.mask[q], D_by_b[k])
                dim_mid = length(active_k)
                if dim_mid == 0
                    continue
                end
                active_km1 = findall(d -> d.mask[q], D_by_b[k-1])
                active_kp1 = findall(d -> d.mask[q], D_by_b[k+1])

                rho_prev = dE[k-1]  # D_{k-1} -> D_k
                rho_next = dE[k]    # D_k -> D_{k+1}

                r_prev = isempty(active_km1) ? 0 : rankQQ_restricted(rho_prev, active_k, active_km1)
                r_next = isempty(active_kp1) ? 0 : rankQQ_restricted(rho_next, active_kp1, active_k)

                if r_prev + r_next != dim_mid
                    error("Downset resolution: vertex q=$q fails exactness at degree k=$(k-1): rank(prev)=$r_prev, rank(next)=$r_next, dim=$dim_mid")
                end
            end
        end
    end

    return true
end

end # module
