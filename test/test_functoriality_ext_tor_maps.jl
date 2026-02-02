using Test
using LinearAlgebra

# Included from test/runtests.jl; uses shared aliases (PM, DF, MD, QQ, ...).


# Build an endomorphism of M that is the identity everywhere except at vertex u,
# where it is replaced by the matrix A (assumes A has size M.dims[u] x M.dims[u]).
function endo_at_vertex(M::MD.PModule{QQ}, u::Int, A::Matrix{QQ})
    comps = Matrix{QQ}[]
    for v in 1:M.Q.n
        dv = M.dims[v]
        push!(comps, Matrix{QQ}(I, dv, dv))
    end
    comps[u] = A
    return MD.PMorphism{QQ}(M, M, comps)
end

# Compose morphisms fiberwise: (g o f)_u = g_u * f_u.
function compose_morphism(g::MD.PMorphism{QQ}, f::MD.PMorphism{QQ})
    @assert f.cod === g.dom
    n = f.dom.Q.n
    comps = [g.comps[u] * f.comps[u] for u in 1:n]
    return MD.PMorphism{QQ}(f.dom, g.cod, comps)
end

# Scalar endomorphism s*id on each fiber.
function scalar_endo(M::MD.PModule{QQ}, s::QQ)
    comps = Vector{Matrix{QQ}}(undef, M.Q.n)
    for u in 1:M.Q.n
        d = M.dims[u]
        comps[u] = d == 0 ? zeros(QQ, 0, 0) : s .* Matrix{QQ}(I, d, d)
    end
    return MD.PMorphism{QQ}(M, M, comps)
end

# Helper: build a chain-poset module with a single cover edge map.
# We intentionally keep this tiny; it is enough to test connecting morphisms by hand.
function _chain_module(P, dims::Vector{Int}, edge_map::Matrix{QQ})
    edges = FF.cover_edges(P)
    D = Dict{Tuple{Int, Int}, Matrix{QQ}}()
    for (u, v) in edges
        D[(u, v)] = zeros(QQ, dims[v], dims[u])
    end
    @assert length(edges) == 1
    D[first(edges)] = edge_map
    return MD.PModule{QQ}(P, dims, D)
end


@testset "Ext functoriality (projective model) in both arguments" begin
    P = chain_poset(2)

    # Simple at 1 and simple at 2 on P.
    S1 = IR.pmodule_from_fringe(one_by_one_fringe(P, FF.principal_upset(P, 1), FF.principal_downset(P, 1)))
    S2 = IR.pmodule_from_fringe(one_by_one_fringe(P, FF.principal_upset(P, 2), FF.principal_downset(P, 2)))

    # Build M = S1 oplus S1 so End(M) is noncommutative (Mat_2).
    M = MD.direct_sum(S1, S1)

    # Build N = S2 oplus S2 so End(N) is noncommutative (Mat_2).
    N = MD.direct_sum(S2, S2)

    EMN = DF.Ext(M, N, PM.DerivedFunctorOptions(maxdeg=2))

    # Nonvacuous check: Ext^1 should be 4 = 2*2 times Ext^1(S1,S2) (which is 1 on this poset).
    @test PM.dim(EMN, 1) == 4

    # Two noncommuting endomorphisms of M at vertex 1 (dims there are 2).
    A = [QQ(1) QQ(1); QQ(0) QQ(1)]
    B = [QQ(1) QQ(0); QQ(1) QQ(1)]
    fA = endo_at_vertex(M, 1, A)
    fB = endo_at_vertex(M, 1, B)

    # Contravariant functoriality in the first argument:
    # Ext(fB o fA) = Ext(fA) o Ext(fB).
    F_A = PM.ext_map_first(EMN, EMN, fA; t=1)
    F_B = PM.ext_map_first(EMN, EMN, fB; t=1)
    F_BA = PM.ext_map_first(EMN, EMN, compose_morphism(fB, fA); t=1)
    @test F_BA == F_A * F_B

    # Two noncommuting endomorphisms of N at vertex 2 (dims there are 2).
    C = [QQ(2) QQ(1); QQ(0) QQ(1)]
    D = [QQ(1) QQ(0); QQ(1) QQ(2)]
    gC = endo_at_vertex(N, 2, C)
    gD = endo_at_vertex(N, 2, D)

    # Covariant functoriality in the second argument:
    # Ext(gD o gC) = Ext(gD) o Ext(gC).
    G_C = PM.ext_map_second(EMN, EMN, gC; t=1)
    G_D = PM.ext_map_second(EMN, EMN, gD; t=1)
    G_DC = PM.ext_map_second(EMN, EMN, compose_morphism(gD, gC); t=1)
    @test G_DC == G_D * G_C
end


@testset "Tor functoriality in both arguments" begin
    P = chain_poset(2)
    Pop = FF.FinitePoset(transpose(FF.leq_matrix(P)))

    # L = simple at 1 on P (as in existing Tor-by-hand test).
    L = IR.pmodule_from_fringe(one_by_one_fringe(P, FF.principal_upset(P, 1), FF.principal_downset(P, 1)))

    # Rop = simple at 2 on P^op (as in existing Tor-by-hand test).
    Rop = IR.pmodule_from_fringe(one_by_one_fringe(Pop, FF.principal_upset(Pop, 2), FF.principal_downset(Pop, 2)))

    # Make direct sums to get noncommuting endomorphisms.
    R2 = MD.direct_sum(Rop, Rop)
    L2 = MD.direct_sum(L, L)

    # Tor should be additive in each argument, so Tor_1 doubles here.
    T_R2_L = DF.Tor(R2, L, PM.DerivedFunctorOptions(maxdeg=3))
    T_R2_L2 = DF.Tor(R2, L2, PM.DerivedFunctorOptions(maxdeg=3))

    @test PM.dim(T_R2_L, 1) == 2
    @test PM.dim(T_R2_L2, 1) == 4

    # Noncommuting endomorphisms of R2 at vertex 2 (dims there are 2).
    A = [QQ(1) QQ(1); QQ(0) QQ(1)]
    B = [QQ(1) QQ(0); QQ(1) QQ(1)]
    fA = endo_at_vertex(R2, 2, A)
    fB = endo_at_vertex(R2, 2, B)

    # Covariant functoriality in the first argument:
    # Tor(fB o fA) = Tor(fB) o Tor(fA).
    F_A = PM.tor_map_first(T_R2_L, T_R2_L, fA; s=1)
    F_B = PM.tor_map_first(T_R2_L, T_R2_L, fB; s=1)
    F_BA = PM.tor_map_first(T_R2_L, T_R2_L, compose_morphism(fB, fA); s=1)
    @test F_BA == F_B * F_A

    # Noncommuting endomorphisms of L2 at vertex 1 (dims there are 2).
    C = [QQ(2) QQ(1); QQ(0) QQ(1)]
    D = [QQ(1) QQ(0); QQ(1) QQ(2)]
    gC = endo_at_vertex(L2, 1, C)
    gD = endo_at_vertex(L2, 1, D)

    # Covariant functoriality in the second argument:
    # Tor(gD o gC) = Tor(gD) o Tor(gC).
    G_C = PM.tor_map_second(T_R2_L2, T_R2_L2, gC; s=1)
    G_D = PM.tor_map_second(T_R2_L2, T_R2_L2, gD; s=1)
    G_DC = PM.tor_map_second(T_R2_L2, T_R2_L2, compose_morphism(gD, gC); s=1)
    @test G_DC == G_D * G_C
end

# Helper: build a tiny chain poset and some simple modules.
# The existing tests already use chain_poset and one_by_one_fringe etc.
# We reuse that style for consistency.

@testset "TorLongExactSequenceSecond + TorAlgebra generator" begin
    P = chain_poset(3)
    # Choose a genuine short exact sequence 0 -> A -> B -> C -> 0:
    # A = [2,2], B = [1,2], C = [1,1] as interval modules on the chain.
    A = IR.pmodule_from_fringe(one_by_one_fringe(P,
            FF.principal_upset(P, 2), FF.principal_downset(P, 2)))
    B = IR.pmodule_from_fringe(one_by_one_fringe(P,
            FF.principal_upset(P, 1), FF.principal_downset(P, 2)))
    C = IR.pmodule_from_fringe(one_by_one_fringe(P,
            FF.principal_upset(P, 1), FF.principal_downset(P, 1)))

    # Explicit inclusion i: A -> B and projection p: B -> C (components per vertex).
    # dims(A) = (0,1,0), dims(B) = (1,1,0), dims(C) = (1,0,0).
    i = MD.PMorphism{QQ}(A, B, [zeros(QQ, 1, 0), QQ[1;;],         zeros(QQ, 0, 0)])
    p = MD.PMorphism{QQ}(B, C, [QQ[1;;],         zeros(QQ, 0, 1), zeros(QQ, 0, 0)])

    # Opposite poset
    Pop = FF.FinitePoset(transpose(FF.leq_matrix(P)))

    # Right module on P^op (simple at vertex 2).
    Rop = IR.pmodule_from_fringe(one_by_one_fringe(Pop,
            FF.principal_upset(Pop, 2), FF.principal_downset(Pop, 2)))

    LES = PM.TorLongExactSequenceSecond(Rop, i, p, PM.DerivedFunctorOptions(maxdeg=2))

    TorRA = DF.Tor(Rop, A, PM.DerivedFunctorOptions(maxdeg=2))
    TorRB = DF.Tor(Rop, B, PM.DerivedFunctorOptions(maxdeg=2))
    TorRC = DF.Tor(Rop, C, PM.DerivedFunctorOptions(maxdeg=2))
    @test LES.maxdeg == 2
    @test length(LES.iH) == 3
    @test length(LES.pH) == 3
    @test length(LES.delta) == 3

    @test [PM.dim(LES.TorA, s) for s in 0:LES.maxdeg] == [PM.dim(TorRA, s) for s in 0:LES.maxdeg]
    @test [PM.dim(LES.TorB, s) for s in 0:LES.maxdeg] == [PM.dim(TorRB, s) for s in 0:LES.maxdeg]
    @test [PM.dim(LES.TorC, s) for s in 0:LES.maxdeg] == [PM.dim(TorRC, s) for s in 0:LES.maxdeg]

    # Tor algebra (exercise multiplication)
    T = DF.Tor(Rop, B, PM.DerivedFunctorOptions(model=:second, maxdeg=2))

    Aalg = PM.TorAlgebra(T; mu_chain_gen=PM.DerivedFunctors.trivial_tor_product_generator(T))

    M00  = PM.multiplication_matrix(Aalg, 0, 0)
    M00b = PM.multiplication_matrix(Aalg, 0, 0)
    @test M00 == M00b

    if PM.dim(T, 1) > 0
        M01 = PM.multiplication_matrix(Aalg, 0, 1)
        @test all(M01 .== 0)
    end
end

@testset "hyperTor_map_first/second: induced maps on Tor_n" begin
    P = chain_poset(2)
    Pop = FF.FinitePoset(transpose(FF.leq_matrix(P)))

    # L = simple at vertex 1 on P.
    L = IR.pmodule_from_fringe(one_by_one_fringe(P,
            FF.principal_upset(P, 1), FF.principal_downset(P, 1)))

    # Rop = simple at vertex 2 on P^op.
    Rop = IR.pmodule_from_fringe(one_by_one_fringe(Pop,
            FF.principal_upset(Pop, 2), FF.principal_downset(Pop, 2)))

    # Complex concentrated in degree 0.
    C = PM.ModuleCochainComplex([L], MD.PMorphism{QQ}[]; tmin=0, check=true)

    HT = PM.hyperTor(Rop, C; maxlen=2)
    T  = DF.Tor(Rop, L, PM.DerivedFunctorOptions(maxdeg=2))

    # Tor_1 is known nonzero in this classical example.
    @test PM.dim(HT, 1) == PM.dim(T, 1)
    d1 = PM.dim(HT, 1)
    @test d1 > 0

    f2 = scalar_endo(Rop, QQ(2))
    f3 = scalar_endo(Rop, QQ(3))
    g2 = scalar_endo(L,   QQ(2))
    g3 = scalar_endo(L,   QQ(3))

    gC2 = PM.ModuleCochainMap(C, C, [g2]; check=true)
    gC3 = PM.ModuleCochainMap(C, C, [g3]; check=true)

    # --- Compare to Tor maps for degree-0 complexes ---
    F2h = PM.hyperTor_map_first(f2, HT, HT; n=1)
    F2t = PM.tor_map_first(f2, T, T; n=1)
    @test F2h == F2t
    @test F2h == QQ(2) .* Matrix{QQ}(I, d1, d1)

    G2h = PM.hyperTor_map_second(gC2, HT, HT; n=1)
    G2t = PM.tor_map_second(g2, T, T; n=1)
    @test G2h == G2t
    @test G2h == QQ(2) .* Matrix{QQ}(I, d1, d1)

    # --- Identity behavior ---
    Fid = PM.hyperTor_map_first(IR.id_morphism(Rop), HT, HT; n=1)
    @test Fid == Matrix{QQ}(I, d1, d1)

    Gid = PM.hyperTor_map_second(PM.idmap(C), HT, HT; n=1)
    @test Gid == Matrix{QQ}(I, d1, d1)

    # --- Functoriality in first variable ---
    f32 = compose_morphism(f3, f2)   # f3 o f2 = 6*id
    F32 = PM.hyperTor_map_first(f32, HT, HT; n=1)
    F3  = PM.hyperTor_map_first(f3,  HT, HT; n=1)
    @test F32 == F3 * F2h

    # --- Functoriality in second variable ---
    g32 = compose_morphism(g3, g2)
    gC32 = PM.ModuleCochainMap(C, C, [g32]; check=true)
    G32 = PM.hyperTor_map_second(gC32, HT, HT; n=1)
    G3  = PM.hyperTor_map_second(gC3,  HT, HT; n=1)
    @test G32 == G3 * G2h

    # --- Bifunctorial commutativity (natural in both vars) ---
    @test (G3 * F2h) == (F2h * G3)
end


@testset "Tor by hand on chain of length 2" begin
    P = chain_poset(2)
    Pop = FF.FinitePoset(transpose(FF.leq_matrix(P)))

    # L = simple at 1 on P
    Lfr = one_by_one_fringe(P, FF.principal_upset(P, 1), FF.principal_downset(P, 1))
    L = IR.pmodule_from_fringe(Lfr)

    # Rop = simple at 2 on Pop (= P^op)
    Rfr = one_by_one_fringe(Pop, FF.principal_upset(Pop, 2), FF.principal_downset(Pop, 2))
    Rop = IR.pmodule_from_fringe(Rfr)

    T = DF.Tor(Rop, L, PM.DerivedFunctorOptions(maxdeg=3))

    @test PM.dim(T, 0) == 0
    @test PM.dim(T, 1) == 1
    @test PM.dim(T, 2) == 0
    @test PM.dim(T, 3) == 0
end

@testset "Tor extra structure: LES, actions, bicomplex" begin
    # Poset: chain 1 < 2
    P = chain_poset(2)
    Pop = FF.FinitePoset(transpose(FF.leq_matrix(P)))  # opposite

    # Left modules on P
    S1 = _chain_module(P, [1, 0], zeros(QQ, 0, 1))
    S2 = _chain_module(P, [0, 1], zeros(QQ, 1, 0))
    P1 = _chain_module(P, [1, 1], QQ[1;;])  # projective at 1

    # Right modules (as P^op-modules) on Pop
    S1op = _chain_module(Pop, [1, 0], zeros(QQ, 1, 0))
    S2op = _chain_module(Pop, [0, 1], zeros(QQ, 0, 1))
    P2op = _chain_module(Pop, [1, 1], QQ[1;;])  # projective at 2 (in Pop)

    # Short exact sequence in the second variable: 0 -> S2 -> P1 -> S1 -> 0
    i = MD.PMorphism{QQ}(S2, P1, [zeros(QQ, 1, 0), QQ[1;;]])
    p = MD.PMorphism{QQ}(P1, S1, [QQ[1;;], zeros(QQ, 0, 1)])

    les2 = PM.TorLongExactSequenceSecond(S2op, i, p, PM.DerivedFunctorOptions(maxdeg=1))

    # Connecting map delta: Tor_1(S2op, S1) -> Tor_0(S2op, S2)
    # In this toy example it is nonzero (this is the standard non-split SES).
    @test size(les2.delta[1], 1) == 0
    @test size(les2.delta[2]) == (PM.dim(les2.TorA, 0), PM.dim(les2.TorC, 1))
    @test les2.delta[2][1, 1] != 0

    # Short exact sequence in the first variable: 0 -> S1op -> P2op -> S2op -> 0
    i1 = MD.PMorphism{QQ}(S1op, P2op, [QQ[1;;], zeros(QQ, 1, 0)])
    p1 = MD.PMorphism{QQ}(P2op, S2op, [zeros(QQ, 0, 1), QQ[1;;]])

    les1 = PM.TorLongExactSequenceFirst(S1, i1, p1, PM.DerivedFunctorOptions(maxdeg=1))

    @test size(les1.delta[2]) == (PM.dim(les1.TorA, 0), PM.dim(les1.TorC, 1))
    @test les1.delta[2][1, 1] != 0

    # Ext action on Tor via the resolve-second model:
    # The Ext^0 unit should act as identity on Tor_1.
    EA = PM.ExtAlgebra(S1, PM.DerivedFunctorOptions(maxdeg=2))
    Tsec = DF.Tor(S2op, S1, PM.DerivedFunctorOptions(model=:second); res=EA.E.res)
    u = PM.unit(EA)
    act = PM.ext_action_on_tor(EA, Tsec, u; s=1)

    @test size(act) == (PM.dim(Tsec, 1), PM.dim(Tsec, 1))
    @test act[1, 1] == QQ(1)

    # Tor double complex total cohomology agrees with Tor groups (degree reindexing).
    # Using small lengths is enough for this example.
    DC = PM.TorDoubleComplex(S2op, S1; maxlen=1)
    Tot = CC.total_complex(DC)

    # Tor_n corresponds to H^{-n} of the total cochain complex.
    Tfirst = DF.Tor(S2op, S1, PM.DerivedFunctorOptions(maxdeg=2))
    @test CC.cohomology_data(Tot, 0).dimH == PM.dim(Tfirst, 0)
    @test CC.cohomology_data(Tot, -1).dimH == PM.dim(Tfirst, 1)
    @test CC.cohomology_data(Tot, -2).dimH == PM.dim(Tfirst, 2)

    # TorAlgebra infrastructure smoke test: a trivial degree-0 product.
    # Choose a projective right module so Tor_0 is 1-dim and higher Tor vanishes.
    T0 = DF.Tor(P2op, S2, PM.DerivedFunctorOptions(maxdeg=0))
    @test PM.dim(T0, 0) == 1

    Alg = PM.TorAlgebra(T0)
    PM.set_chain_product!(Alg, 0, 0, sparse(QQ[1;;]))
    M00 = PM.multiplication_matrix(Alg, 0, 0)
    @test size(M00) == (1, 1)
    @test M00[1, 1] == QQ(1)

    x = PM.element(Alg, 0, [QQ(1)])
    y = PM.multiply(Alg, x, x)
    @test y.deg == 0
    @test y.coords[1] == QQ(1)
end
