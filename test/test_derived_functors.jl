using Test

using LinearAlgebra

# This file assumes the helper constructors defined in runtests.jl:
# - chain_poset(n)
# - diamond_poset()
# - one_by_one_fringe(P, U, D)
# - boolean_lattice_B3_poset()


# -----------------------------------------------------------------------------
# Small helper: direct sum of poset-modules and the split short exact sequence
# -----------------------------------------------------------------------------

function direct_sum_with_split_sequence(A::MD.PModule{QQ}, C::MD.PModule{QQ})
    Q = A.Q
    @test FF.poset_equal(Q, C.Q)

    dimsB = [A.dims[v] + C.dims[v] for v in 1:Q.n]

    # Block-diagonal structure maps on cover edges.
    edge_mapsB = Dict{Tuple{Int,Int}, Matrix{QQ}}()
    for (i, j) in FF.cover_edges(Q)
        # We are iterating over cover edges of Q. The module edge-map store
        # guarantees cover-edge maps exist (missing entries are filled with zeros
        # at construction), so we can index directly and avoid Base.get().
        Aij = A.edge_maps[i, j]
        Cij = C.edge_maps[i, j]

        top    = hcat(Aij, zeros(QQ, size(Aij,1), size(Cij,2)))
        bottom = hcat(zeros(QQ, size(Cij,1), size(Aij,2)), Cij)
        edge_mapsB[(i,j)] = vcat(top, bottom)
    end

    B = MD.PModule{QQ}(Q, dimsB, edge_mapsB)

    # Inclusion i: A -> A oplus C (first summand)
    comps_i = Vector{Matrix{QQ}}(undef, Q.n)
    for v in 1:Q.n
        comps_i[v] = vcat(Matrix{QQ}(I, A.dims[v], A.dims[v]),
                          zeros(QQ, C.dims[v], A.dims[v]))
    end
    i = MD.PMorphism{QQ}(A, B, comps_i)

    # Projection p: A oplus C -> C (second summand)
    comps_p = Vector{Matrix{QQ}}(undef, Q.n)
    for v in 1:Q.n
        comps_p[v] = hcat(zeros(QQ, C.dims[v], A.dims[v]),
                          Matrix{QQ}(I, C.dims[v], C.dims[v]))
    end
    p = MD.PMorphism{QQ}(B, C, comps_p)

    return B, i, p
end

@testset "Minimality diagnostics for projective/injective resolutions" begin
    # Use a small poset where minimal resolutions are nontrivial.
    P = diamond_poset()

    # Simple at vertex 1.
    S1 = IR.pmodule_from_fringe(
        one_by_one_fringe(P, FF.principal_upset(P, 1), FF.principal_downset(P, 1))
    )

    # Projective resolution should be minimal (and certified minimal by the checker).
    resP = DF.projective_resolution(S1, PM.ResolutionOptions(maxlen=4))
    repP = PM.minimality_report(resP)
    @test repP.cover_ok
    @test repP.minimal
    @test isempty(repP.diagonal_violations)
    @test PM.is_minimal(resP)
    PM.assert_minimal(resP)

    # Passing ResolutionOptions(minimal=true, check=true) should succeed and return a minimal resolution.
    resPmin = DF.projective_resolution(S1, PM.ResolutionOptions(maxlen=4, minimal=true, check=true))
    @test PM.is_minimal(resPmin)

    # Corrupt the resolution by inserting a diagonal coefficient in d^1 (degree 1 -> 0),
    # which should violate minimality (a generator mapping to itself at the same vertex).
    if length(resP.gens) >= 2 && !isempty(resP.gens[1]) && !isempty(resP.d_mat)
        gens_bad = [copy(g) for g in resP.gens]
        d_mat_bad = copy(resP.d_mat)

        # Ensure we can reference a "same vertex" pair across cod/domain by duplicating
        # an existing vertex label in degree 1.
        push!(gens_bad[2], gens_bad[1][1])

        D = d_mat_bad[1]
        extra = spzeros(QQ, size(D, 1), 1)
        extra[1, 1] = QQ(1)   # explicit diagonal coefficient
        d_mat_bad[1] = hcat(D, extra)

        res_bad = PM.ProjectiveResolution(resP.M, resP.Pmods, gens_bad, resP.d_mor, d_mat_bad, resP.aug)
        rep_bad = PM.minimality_report(res_bad; check_cover=true)
        @test !rep_bad.minimal
        @test !isempty(rep_bad.diagonal_violations)
        @test !PM.is_minimal(res_bad)
        @test_throws ErrorException PM.assert_minimal(res_bad)
    end

    # Injective resolution: also expected to be minimal for these constructions.
    resI = DF.injective_resolution(S1, PM.ResolutionOptions(maxlen=4))
    repI = PM.minimality_report(resI)
    @test repI.hull_ok
    @test repI.minimal
    @test isempty(repI.diagonal_violations)
    @test PM.is_minimal(resI)
    PM.assert_minimal(resI)

    resImin = DF.injective_resolution(S1, PM.ResolutionOptions(maxlen=4, minimal=true, check=true))
    @test PM.is_minimal(resImin)
end

@testset "Betti extraction from projective resolutions (diamond)" begin
    P = diamond_poset()

    # Simple modules S1..S4 as fringe modules, then as poset-modules.
    Sm = Vector{MD.PModule{QQ}}(undef, P.n)
    for v in 1:P.n
        Hv = one_by_one_fringe(P, FF.principal_upset(P, v), FF.principal_downset(P, v))
        Sm[v] = IR.pmodule_from_fringe(Hv)
    end
    S1, S2, S3, S4 = Sm

    # Minimal projective resolution of S1 on the diamond should have:
    # P0 = P1, P1 = P2 oplus P3, P2 = P4.
    res = DF.projective_resolution(S1, PM.ResolutionOptions(maxlen=2))
    b = DF.betti(res)

    @test length(b) == 4
    @test b[(0, 1)] == 1
    @test b[(1, 2)] == 1
    @test b[(1, 3)] == 1
    @test b[(2, 4)] == 1

    Btbl = DF.betti_table(res)
    @test Btbl[1,1] == 1
    @test Btbl[2,2] == 1
    @test Btbl[2,3] == 1
    @test Btbl[3,4] == 1
end


@testset "Yoneda product (diamond: Ext^1 x Ext^1 -> Ext^2)" begin
    P = diamond_poset()

    Sm = Vector{MD.PModule{QQ}}(undef, P.n)
    for v in 1:P.n
        Hv = one_by_one_fringe(P, FF.principal_upset(P, v), FF.principal_downset(P, v))
        Sm[v] = IR.pmodule_from_fringe(Hv)
    end
    S1, S2, S3, S4 = Sm

    # Compute the target Ext space once so coordinates are comparable.
    E14 = DF.Ext(S1, S4, PM.DerivedFunctorOptions(maxdeg=2))
    @test PM.dim(E14, 2) == 1

    # Via the chain 1 -> 2 -> 4
    E24 = DF.Ext(S2, S4, PM.DerivedFunctorOptions(maxdeg=1))
    E12 = DF.Ext(S1, S2, PM.DerivedFunctorOptions(maxdeg=2))  # needs tmax >= 2 because p+q = 2
    @test PM.dim(E24, 1) == 1
    @test PM.dim(E12, 1) == 1

    beta = [QQ(1)]
    alpha = [QQ(1)]
    _, coords_2 = PM.DerivedFunctors.yoneda_product(E24, 1, beta, E12, 1, alpha; ELN=E14)
    @test coords_2[1] != 0

    # Via the chain 1 -> 3 -> 4
    E34 = DF.Ext(S3, S4, PM.DerivedFunctorOptions(maxdeg=1))
    E13 = DF.Ext(S1, S3, PM.DerivedFunctorOptions(maxdeg=2))
    _, coords_3 = PM.DerivedFunctors.yoneda_product(E34, 1, [QQ(1)], E13, 1, [QQ(1)]; ELN=E14)
    @test coords_3[1] != 0

    # In a 1-dimensional target, the two products must be proportional.
    # With our deterministic lifts/basis choices, they should agree up to sign.
    @test coords_2[1] == coords_3[1] || coords_2[1] == -coords_3[1]
end


@testset "Yoneda associativity sanity check (B3, degree 3 is nonzero)" begin
    P = boolean_lattice_B3_poset()

    Sm = Vector{MD.PModule{QQ}}(undef, P.n)
    for v in 1:P.n
        Hv = one_by_one_fringe(P, FF.principal_upset(P, v), FF.principal_downset(P, v))
        Sm[v] = IR.pmodule_from_fringe(Hv)
    end

    # Element numbering is the bitmask order described in boolean_lattice_B3_poset().
    S0   = Sm[1]  # {}
    S1   = Sm[2]  # {1}
    S12  = Sm[5]  # {1,2}
    S123 = Sm[8]  # {1,2,3}

    # Target space: Ext^3(S0, S123) should be 1-dimensional for B3.
    E03 = DF.Ext(S0, S123, PM.DerivedFunctorOptions(maxdeg=3))
    @test PM.dim(E03, 3) == 1

    # Degree-1 generators on the cover chain:
    #   {} -> {1} -> {1,2} -> {1,2,3}
    E23 = DF.Ext(S12, S123, PM.DerivedFunctorOptions(maxdeg=3))
    E12 = DF.Ext(S1,  S12, PM.DerivedFunctorOptions(maxdeg=3))
    E01 = DF.Ext(S0,  S1,   PM.DerivedFunctorOptions(maxdeg=3))

    @test PM.dim(E23, 1) == 1
    @test PM.dim(E12, 1) == 1
    @test PM.dim(E01, 1) == 1

    # Intermediate targets in degree 2 (also 1-dimensional for this choice).
    E13 = DF.Ext(S1,  S123, PM.DerivedFunctorOptions(maxdeg=3))
    E02 = DF.Ext(S0,  S12, PM.DerivedFunctorOptions(maxdeg=3))
    @test PM.dim(E13, 2) == 1
    @test PM.dim(E02, 2) == 1

    # Left bracketing: (e23 * e12) * e01
    _, x = PM.DerivedFunctors.yoneda_product(E23, 1, [QQ(1)], E12, 1, [QQ(1)]; ELN=E13)  # x in Ext^2(S1,S123)
    _, left = PM.DerivedFunctors.yoneda_product(E13, 2, x, E01, 1, [QQ(1)]; ELN=E03)

    # Right bracketing: e23 * (e12 * e01)
    _, y = PM.DerivedFunctors.yoneda_product(E12, 1, [QQ(1)], E01, 1, [QQ(1)]; ELN=E02)  # y in Ext^2(S0,S12)
    _, right = PM.DerivedFunctors.yoneda_product(E23, 1, [QQ(1)], E02, 2, y; ELN=E03)

    # Nontriviality + associativity up to sign in a 1-dimensional target.
    @test left[1] != 0
    @test right[1] != 0
    @test left[1] == right[1] || left[1] == -right[1]
end


@testset "Connecting homomorphisms: split exact sequences give zero maps" begin
    Q = chain_poset(4)

    Sm = Vector{MD.PModule{QQ}}(undef, Q.n)
    for v in 1:Q.n
        Hv = one_by_one_fringe(Q, FF.principal_upset(Q, v), FF.principal_downset(Q, v))
        Sm[v] = IR.pmodule_from_fringe(Hv)
    end
    S1, S2, S3, S4 = Sm

    # -------------------------------------------------------------------------
    # Second argument LES: 0 -> A -> A oplus C -> C -> 0
    # delta : Ext^t(M,C) -> Ext^{t+1}(M,A)
    # Choose M=S1, C=S3 (Ext^2), A=S4 (Ext^3), t=2.
    # -------------------------------------------------------------------------
    A = S4
    C = S3
    B, i, p = direct_sum_with_split_sequence(A, C)

    # We test delta^2 : Ext^2(M,C) -> Ext^3(M,A), so we need the resolution through degree t+1 = 3.
    t = 2
    resM = DF.projective_resolution(S1, PM.ResolutionOptions(maxlen=t+1))   # i.e. maxlen=3
    EMA = DF.Ext(resM, A)
    EMB = DF.Ext(resM, B)
    EMC = DF.Ext(resM, C)

    delta2 = PM.connecting_hom(EMA, EMB, EMC, i, p; t=2)
    @test all(delta2 .== 0)

    # -------------------------------------------------------------------------
    # First argument LES: 0 -> A -> A oplus C -> C -> 0
    # delta : Ext^t(A,N) -> Ext^{t+1}(C,N)
    # Choose N=S4, A=S3 (Ext^1), C=S2 (Ext^2), t=1.
    # -------------------------------------------------------------------------
    A1 = S3
    C1 = S2
    B1, i1, p1 = direct_sum_with_split_sequence(A1, C1)

    resN = DF.injective_resolution(S4, PM.ResolutionOptions(maxlen=2))
    EA = PM.ExtInjective(A1, resN)
    EB = PM.ExtInjective(B1, resN)
    EC = PM.ExtInjective(C1, resN)

    delta1 = PM.connecting_hom_first(EA, EB, EC, i1, p1; t=1)
    @test all(delta1 .== 0)
end

@testset "Connecting homomorphisms: nonsplit extension gives nonzero delta" begin
    # On the 2-element chain 1<2, Ext^1(S1,S2) has dimension 1.
    # The interval module k[1,2] (dims [1,1] and identity map along 1<2)
    # is a nonsplit extension 0 -> S2 -> k[1,2] -> S1 -> 0.
    P = chain_poset(2)

    S1 = IR.pmodule_from_fringe(one_by_one_fringe(P, FF.principal_upset(P, 1), FF.principal_downset(P, 1)))
    S2 = IR.pmodule_from_fringe(one_by_one_fringe(P, FF.principal_upset(P, 2), FF.principal_downset(P, 2)))
    I12 = IR.pmodule_from_fringe(one_by_one_fringe(P, FF.principal_upset(P, 1), FF.principal_downset(P, 2)))

    # i: S2 -> I12 is inclusion on stalk 2.
    comps_i = [zeros(QQ, I12.dims[v], S2.dims[v]) for v in 1:P.n]
    comps_i[2] = Matrix{QQ}(I, 1, 1)
    i = MD.PMorphism{QQ}(S2, I12, comps_i)

    # p: I12 -> S1 is projection on stalk 1.
    comps_p = [zeros(QQ, S1.dims[v], I12.dims[v]) for v in 1:P.n]
    comps_p[1] = Matrix{QQ}(I, 1, 1)
    p = MD.PMorphism{QQ}(I12, S1, comps_p)

    # Fix M = S1. Then delta^0: Hom(S1,S1) -> Ext^1(S1,S2) sends id to the extension class,
    # so it must be nonzero (hence rank 1, since both sides are 1-dimensional).
    resM = DF.projective_resolution(S1, PM.ResolutionOptions(maxlen=2))
    EMA = DF.Ext(resM, S2)
    EMB = DF.Ext(resM, I12)
    EMC = DF.Ext(resM, S1)
    delta0 = PM.connecting_hom(EMA, EMB, EMC, i, p; t=0)
    @test EX.rankQQ(delta0) == 1

    # The packaged long exact sequence should expose the same delta^0.
    les = PM.ExtLongExactSequenceSecond(S1, S2, I12, S1, i, p, PM.DerivedFunctorOptions(maxdeg=0))
    @test EX.rankQQ(les.delta[1]) == 1
    @test Matrix(les.delta[1]) == Matrix(delta0)
end

@testset "ExtAlgebra: cached multiplication agrees with yoneda_product" begin
    P = diamond_poset()

    # Build simples S1..S4 as 1x1 fringe modules, then as poset-modules.
    Sm = Vector{MD.PModule{QQ}}(undef, P.n)
    for v in 1:P.n
        Hv = one_by_one_fringe(P, FF.principal_upset(P, v), FF.principal_downset(P, v))
        Sm[v] = IR.pmodule_from_fringe(Hv)
    end
    S1, S2, S3, S4 = Sm

    # We take a direct sum with enough structure to produce nontrivial Ext in degree 1
    # and allow products in degree 2 on the diamond.
    #
    # M = S1 oplus S2 oplus S4 is small but already contains:
    # - degree-1 extension classes among summands, and
    # - degree-2 composites (in the full Ext(M,M)).
    M12, _, _ = direct_sum_with_split_sequence(S1, S2)
    M, _, _ = direct_sum_with_split_sequence(M12, S4)

    A = PM.ExtAlgebra(M, PM.DerivedFunctorOptions(maxdeg=2))
    E = A.E

    # Sanity: dimensions agree with the underlying Ext space.
    for t in 0:E.tmax
        @test PM.dim(A, t) == PM.dim(E, t)
    end

    # The unit should act as both-sided identity on every homogeneous degree <= tmax.
    oneA = one(A)
    for t in 0:E.tmax
        dt = PM.dim(A, t)
        if dt == 0
            continue
        end

        # Deterministic "generic" element: (1,2,3,...,dt).
        x = PM.element(A, t, [QQ(i) for i in 1:dt])

        @test (oneA * x).coords == x.coords
        @test (x * oneA).coords == x.coords
    end

    # Cache behavior: after one multiplication in (p,q), the multiplication matrix should exist,
    # and repeated multiplication should not grow the cache.
    if E.tmax >= 2 && PM.dim(A, 1) > 0
        d1 = PM.dim(A, 1)
        x = PM.element(A, 1, [QQ(i) for i in 1:d1])
        y = PM.element(A, 1, [QQ(d1 - i + 1) for i in 1:d1])

        prod1 = x * y
        @test haskey(A.mult_cache, (1, 1))
        nkeys = length(A.mult_cache)

        prod2 = x * y
        @test length(A.mult_cache) == nkeys

        # Cached multiplication must match a direct call to the mathematical core (Yoneda product)
        # in the same Ext space and bases.
        _, coords_direct = PM.DerivedFunctors.yoneda_product(A.E, 1, x.coords, A.E, 1, y.coords; ELN=A.E)
        @test prod1.coords == coords_direct
        @test prod2.coords == coords_direct
    end

    # Associativity in the cached algebra (within truncation).
    #
    # On the diamond, Ext^3 is expected to vanish for many modules; we therefore test an
    # associativity instance that stays inside degrees <= 2 by including a degree-0 factor.
    if E.tmax >= 2 && PM.dim(A, 1) > 0
        d1 = PM.dim(A, 1)

        # First basis direction e_1
        a = PM.element(A, 1, [one(QQ); zeros(QQ, d1 - 1)])

        # Last basis direction e_{d1}
        b = PM.element(A, 1, [zeros(QQ, d1 - 1); one(QQ)])
        
        c = oneA

        @test ((a * b) * c).coords == (a * (b * c)).coords
        @test ((c * a) * b).coords == (c * (a * b)).coords
    end
end

@testset "Sparse assembly replacements (dense->sparse removed)" begin
    
    # 0) _append_scaled_triplets! matches the old findnz(sparse(F)) pattern (up to matrix equality)
    let
        F = QQ[1 0; 2 3]
        I1 = Int[]; J1 = Int[]; V1 = QQ[]
        PM.CoreModules._append_scaled_triplets!(I1, J1, V1, F, 10, 20; scale=QQ(2))

        S1 = sparse(I1, J1, V1, 12, 22)

        Ii, Ji, Vi = findnz(sparse(F))
        S2 = sparse(Ii .+ 10, Ji .+ 20, Vi .* QQ(2), 12, 22)

        @test S1 == S2
    end

    # Build a small poset and module with nontrivial (but diagonal) structure maps.
    P = chain_poset(3)
    dims = fill(2, 3)

    A12 = QQ[1 0; 0 2]
    A23 = QQ[3 0; 0 5]

    edge_maps = Dict{Tuple{Int,Int}, Matrix{QQ}}()
    edge_maps[(1,2)] = A12
    edge_maps[(2,3)] = A23

    M = MD.PModule{QQ}(P, dims, edge_maps)

    # 1) _coeff_matrix_upsets: new sparse assembly equals old dense->sparse reference
    let
        U1 = PM.FiniteFringe.principal_upset(P, 1)
        U2 = PM.FiniteFringe.principal_upset(P, 2)
        U3 = PM.FiniteFringe.principal_upset(P, 3)
        domU = [U1, U2, U3]
        codU = [U1, U2, U3]

        Cnew = DF._coeff_matrix_upsets(domU, codU)

        Cold_dense = zeros(QQ, length(codU), length(domU))
        for i in 1:length(domU), j in 1:length(codU)
            c = one(QQ)
            for v in codU[j]
                if !(v in domU[i])
                    c = zero(QQ)
                    break
                end
            end
            Cold_dense[j,i] = c
        end
        Cold = sparse(Cold_dense)

        @test Cnew == Cold
        @test Cnew isa SparseMatrixCSC{QQ,Int}
    end

    # 1b) _coeff_matrix_upsets(P, dom_bases, cod_bases, f) agrees with the top-vertex matrix
    #
    # On a chain poset, vertex n is a top element. At that vertex, all principal upsets
    # are active, so the morphism component matrix is the full coefficient matrix.
    let
        P = chain_poset(3)
        DF = PM.DerivedFunctors

        # A simple module with zero structure maps. This forces projective_cover to
        # pick generators at multiple vertices in a predictable way, producing a
        # nontrivial next-stage differential P1 -> P0.
        dims = fill(1, 3)
        edge_maps = Dict{Tuple{Int,Int}, Matrix{QQ}}()
        edge_maps[(1,2)] = zeros(QQ, 1, 1)
        edge_maps[(2,3)] = zeros(QQ, 1, 1)
        M = MD.PModule{QQ}(P, dims, edge_maps)

        P0, pi0, gens0 = IR.projective_cover(M)
        K, iota = PM.kernel_with_inclusion(pi0)
        P1, pi1, gens1 = IR.projective_cover(K)

        # Differential d : P1 -> P0
        d = DF.compose(iota, pi1)

        dom_bases = DF._flatten_gens_at(gens1)
        cod_bases = DF._flatten_gens_at(gens0)

        C = DF._coeff_matrix_upsets(P, dom_bases, cod_bases, d)

        # On the chain poset(3), vertex 3 is top, so all summands are active there.
        @test C == sparse(d.comps[3])
        @test C isa SparseMatrixCSC{QQ,Int}
    end


    # 2) _precompose_on_hom_cochains_from_projective_coeff: dense & sparse coeff match dense reference
    let
        dom_gens = [3,2,1]
        cod_gens = [3,2,1]

        dom_offsets = zeros(Int, length(dom_gens) + 1)
        cod_offsets = zeros(Int, length(cod_gens) + 1)
        for i in 1:length(dom_gens)
            dom_offsets[i+1] = dom_offsets[i] + M.dims[dom_gens[i]]
            cod_offsets[i+1] = cod_offsets[i] + M.dims[cod_gens[i]]
        end

        coeff_dense = QQ[
            1 0 0;
            2 3 0;
            0 4 5
        ]
        coeff_sparse = sparse(coeff_dense)

        # Old dense reference
        Fref_dense = zeros(QQ, dom_offsets[end], cod_offsets[end])
        for i in 1:length(dom_gens), j in 1:length(cod_gens)
            c = coeff_dense[j,i]
            iszero(c) && continue
            ui = dom_gens[i]
            vj = cod_gens[j]
            A = MD.map_leq(M, vj, ui)
            rows = (dom_offsets[i] + 1):dom_offsets[i+1]
            cols = (cod_offsets[j] + 1):cod_offsets[j+1]
            Fref_dense[rows, cols] .+= c .* A
        end
        Fref = sparse(Fref_dense)

        F1 = DF._precompose_on_hom_cochains_from_projective_coeff(M, dom_gens, cod_gens, dom_offsets, cod_offsets, coeff_dense)
        F2 = DF._precompose_on_hom_cochains_from_projective_coeff(M, dom_gens, cod_gens, dom_offsets, cod_offsets, coeff_sparse)

        @test F1 == sparse(Fref_dense)
        @test F2 == sparse(Fref_dense)
        @test F1 isa SparseMatrixCSC{QQ,Int}
        @test F2 isa SparseMatrixCSC{QQ,Int}
    end

    # 3) _tensor_map_on_tor_chains_from_projective_coeff: dense & sparse coeff match dense reference
    let
        dom_bases = [1,2]
        cod_bases = [2,3]

        dom_offsets = zeros(Int, length(dom_bases) + 1)
        cod_offsets = zeros(Int, length(cod_bases) + 1)
        for i in 1:length(dom_bases)
            dom_offsets[i+1] = dom_offsets[i] + M.dims[dom_bases[i]]
        end
        for j in 1:length(cod_bases)
            cod_offsets[j+1] = cod_offsets[j] + M.dims[cod_bases[j]]
        end

        coeff_dense = QQ[
            0 1;
            2 0
        ]
        coeff_sparse = sparse(coeff_dense)

        Bref_dense = zeros(QQ, cod_offsets[end], dom_offsets[end])
        for i in 1:length(dom_bases), j in 1:length(cod_bases)
            c = coeff_dense[j,i]
            iszero(c) && continue
            u = dom_bases[i]
            v = cod_bases[j]
            A = MD.map_leq(M, u, v)
            rows = (cod_offsets[j] + 1):cod_offsets[j+1]
            cols = (dom_offsets[i] + 1):dom_offsets[i+1]
            Bref_dense[rows, cols] = c .* A
        end
        Bref = sparse(Bref_dense)

        B1 = DF._tensor_map_on_tor_chains_from_projective_coeff(M, dom_bases, cod_bases, dom_offsets, cod_offsets, coeff_dense)
        B2 = DF._tensor_map_on_tor_chains_from_projective_coeff(M, dom_bases, cod_bases, dom_offsets, cod_offsets, coeff_sparse)

        @test B1 == Bref
        @test B2 == Bref
    end

    # 4) _tor_blockdiag_map_on_chains matches dense reference
    let
        f = IR.id_morphism(M)
        gens = [1,2,3]

        dom_offsets = zeros(Int, length(gens) + 1)
        cod_offsets = zeros(Int, length(gens) + 1)
        for i in 1:length(gens)
            dom_offsets[i+1] = dom_offsets[i] + f.dom.dims[gens[i]]
            cod_offsets[i+1] = cod_offsets[i] + f.cod.dims[gens[i]]
        end

        T = DF._tor_blockdiag_map_on_chains(f, gens, dom_offsets, cod_offsets)

        Tref_dense = zeros(QQ, cod_offsets[end], dom_offsets[end])
        for i in 1:length(gens)
            u = gens[i]
            rows = (cod_offsets[i] + 1):cod_offsets[i+1]
            cols = (dom_offsets[i] + 1):dom_offsets[i+1]
            Tref_dense[rows, cols] = f.comps[u]
        end
        @test T == sparse(Tref_dense)
    end
end

@testset "DerivedFunctors: downset postcompose coefficient solver" begin
    # Minimal regression: 1-vertex poset, so all downsets are trivial and the fiberwise equation is
    # just C * F = G at the unique vertex.

    Q = FF.FinitePoset(trues(1, 1); check = false)

    # X(u) has dimension 3, E(u) has 3 downset summands, Ep(u) has 2 downset summands.
    X  = MD.PModule{QQ}(Q, [3], Dict{Tuple{Int, Int}, Matrix{QQ}}())
    E  = MD.PModule{QQ}(Q, [3], Dict{Tuple{Int, Int}, Matrix{QQ}}())
    Ep = MD.PModule{QQ}(Q, [2], Dict{Tuple{Int, Int}, Matrix{QQ}}())

    F1 = Matrix{QQ}(I, 3, 3)
    G1 = QQ[1 2 3;
            4 5 6]

    f = MD.PMorphism{QQ}(X, E,  [F1])
    g = MD.PMorphism{QQ}(X, Ep, [G1])

    dom_bases = [1, 1, 1]
    cod_bases = [1, 1]
    act_dom = [collect(1:3)]
    act_cod = [collect(1:2)]

    C = DF._solve_downset_postcompose_coeff(f, g, dom_bases, cod_bases, act_dom, act_cod)
    @test C == G1
    @test C * F1 == G1

    # Inconsistent system: F = 0, G != 0 should throw.
    F0 = zeros(QQ, 3, 3)
    f0 = MD.PMorphism{QQ}(X, E, [F0])
    @test_throws ErrorException DF._solve_downset_postcompose_coeff(f0, g, dom_bases, cod_bases, act_dom, act_cod)
end
