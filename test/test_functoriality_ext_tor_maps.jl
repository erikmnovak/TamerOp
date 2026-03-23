using Test
using LinearAlgebra

if !@isdefined(CM)
    const CM = TamerOp.CoreModules
end
if !@isdefined(FF)
    const FF = TamerOp.FiniteFringe
end
if !@isdefined(IR)
    const IR = TamerOp.IndicatorResolutions
end
if !@isdefined(MD)
    const MD = TamerOp.Modules
end
if !@isdefined(DF)
    const DF = TamerOp.DerivedFunctors
end
if !@isdefined(TO)
    const TO = TamerOp
end

# Included from test/runtests.jl; can also run standalone with the aliases above.

if !@isdefined(FIELD_QQ)
    const FIELD_QQ = CM.QQField()
    const FIELD_F2 = CM.F2()
    const FIELD_F3 = CM.F3()
    const FIELD_F5 = CM.Fp(5)
    const FIELD_R64 = CM.RealField(Float64; rtol=1e-10, atol=1e-12)
end
if !@isdefined(FIELDS_FULL)
    const FIELDS_FULL = (FIELD_QQ, FIELD_F2, FIELD_F3, FIELD_F5, FIELD_R64)
end
if !@isdefined(with_fields)
    with_fields(fields, f::Function) = foreach(f, fields)
    with_fields(f::Function, fields) = foreach(f, fields)
end
if !@isdefined(chain_poset)
    function chain_poset(n::Integer; check::Bool=false)
        nn = Int(n)
        rel = falses(nn, nn)
        @inbounds for i in 1:nn, j in i:nn
            rel[i, j] = true
        end
        return FF.FinitePoset(rel; check=check)
    end
end
if !@isdefined(one_by_one_fringe)
    one_by_one_fringe(P, U, D; scalar, field) = FF.one_by_one_fringe(P, U, D, scalar; field=field)
end


# Build an endomorphism of M that is the identity everywhere except at vertex u,
# where it is replaced by the matrix A (assumes A has size M.dims[u] x M.dims[u]).
function endo_at_vertex(M::MD.PModule{K}, u::Int, A::AbstractMatrix{K}) where {K}
    comps = Vector{Matrix{K}}(undef, M.Q.n)
    for v in 1:M.Q.n
        dv = M.dims[v]
        comps[v] = CM.eye(M.field, dv)
    end
    comps[u] = Matrix{K}(A)
    return MD.PMorphism(M, M, comps)
end

# Compose morphisms fiberwise: (g o f)_u = g_u * f_u.
function compose_morphism(g::MD.PMorphism{K}, f::MD.PMorphism{K}) where {K}
    @assert f.cod === g.dom
    n = f.dom.Q.n
    comps = [g.comps[u] * f.comps[u] for u in 1:n]
    return MD.PMorphism(f.dom, g.cod, comps)
end

# Scalar endomorphism s*id on each fiber.
function scalar_endo(M::MD.PModule{K}, s::K) where {K}
    comps = Vector{Matrix{K}}(undef, M.Q.n)
    for u in 1:M.Q.n
        d = M.dims[u]
        comps[u] = d == 0 ? CM.zeros(M.field, 0, 0) : s .* CM.eye(M.field, d)
    end
    return MD.PMorphism(M, M, comps)
end

# Helper: build a chain-poset module with a single cover edge map.
# We intentionally keep this tiny; it is enough to test connecting morphisms by hand.
function _chain_module(P, dims::Vector{Int}, edge_map::AbstractMatrix{K}, field::CM.AbstractCoeffField) where {K}
    CM.coeff_type(field) == K || error("_chain_module: coeff_type(field) != eltype(edge_map)")
    edges = FF.cover_edges(P)
    D = Dict{Tuple{Int, Int}, Matrix{K}}()
    for (u, v) in edges
        D[(u, v)] = CM.zeros(field, dims[v], dims[u])
    end
    @assert length(edges) == 1
    D[first(edges)] = edge_map
    return MD.PModule{K}(P, dims, D; field=field)
end


with_fields(FIELDS_FULL) do field
if !(field isa CM.RealField)
K = CM.coeff_type(field)
c(x) = CM.coerce(field, x)

@testset "Ext functoriality (projective model) in both arguments" begin
    P = chain_poset(2)
    # Simple at 1 and simple at 2 on P.
    S1 = IR.pmodule_from_fringe(one_by_one_fringe(P, FF.principal_upset(P, 1), FF.principal_downset(P, 1); scalar=one(K), field=field))
    S2 = IR.pmodule_from_fringe(one_by_one_fringe(P, FF.principal_upset(P, 2), FF.principal_downset(P, 2); scalar=one(K), field=field))

    # Build M = S1 oplus S1 so End(M) is noncommutative (Mat_2).
    M = MD.direct_sum(S1, S1)

    # Build N = S2 oplus S2 so End(N) is noncommutative (Mat_2).
    N = MD.direct_sum(S2, S2)

    EMN = DF.Ext(M, N, TO.DerivedFunctorOptions(maxdeg=2))

    # Nonvacuous check: Ext^1 should be 4 = 2*2 times Ext^1(S1,S2) (which is 1 on this poset).
    @test TO.dim(EMN, 1) == 4

    # Two noncommuting endomorphisms of M at vertex 1 (dims there are 2).
    A = [c(1) c(1); c(0) c(1)]
    B = [c(1) c(0); c(1) c(1)]
    fA = endo_at_vertex(M, 1, A)
    fB = endo_at_vertex(M, 1, B)

    # Contravariant functoriality in the first argument:
    # Ext(fB o fA) = Ext(fA) o Ext(fB).
    F_A = TO.ext_map_first(EMN, EMN, fA; t=1)
    F_B = TO.ext_map_first(EMN, EMN, fB; t=1)
    F_BA = TO.ext_map_first(EMN, EMN, compose_morphism(fB, fA); t=1)
    @test F_BA == F_A * F_B

    # Two noncommuting endomorphisms of N at vertex 2 (dims there are 2).
    C = [c(2) c(1); c(0) c(1)]
    D = [c(1) c(0); c(1) c(2)]
    gC = endo_at_vertex(N, 2, C)
    gD = endo_at_vertex(N, 2, D)

    # Covariant functoriality in the second argument:
    # Ext(gD o gC) = Ext(gD) o Ext(gC).
    G_C = TO.ext_map_second(EMN, EMN, gC; t=1)
    G_D = TO.ext_map_second(EMN, EMN, gD; t=1)
    G_DC = TO.ext_map_second(EMN, EMN, compose_morphism(gD, gC); t=1)
    @test G_DC == G_D * G_C
end


@testset "Tor functoriality in both arguments" begin
    P = chain_poset(2)
    Pop = FF.FinitePoset(transpose(FF.leq_matrix(P)))

    # L = simple at 1 on P (as in existing Tor-by-hand test).
    L = IR.pmodule_from_fringe(one_by_one_fringe(P, FF.principal_upset(P, 1), FF.principal_downset(P, 1); scalar=one(K), field=field))

    # Rop = simple at 2 on P^op (as in existing Tor-by-hand test).
    Rop = IR.pmodule_from_fringe(one_by_one_fringe(Pop, FF.principal_upset(Pop, 2), FF.principal_downset(Pop, 2); scalar=one(K), field=field))

    # Make direct sums to get noncommuting endomorphisms.
    R2 = MD.direct_sum(Rop, Rop)
    L2 = MD.direct_sum(L, L)

    # Tor should be additive in each argument, so Tor_1 doubles here.
    T_R2_L = DF.Tor(R2, L, TO.DerivedFunctorOptions(maxdeg=3))
    T_R2_L2 = DF.Tor(R2, L2, TO.DerivedFunctorOptions(maxdeg=3))

    @test TO.dim(T_R2_L, 1) == 2
    @test TO.dim(T_R2_L2, 1) == 4

    # Noncommuting endomorphisms of R2 at vertex 2 (dims there are 2).
    A = [c(1) c(1); c(0) c(1)]
    B = [c(1) c(0); c(1) c(1)]
    fA = endo_at_vertex(R2, 2, A)
    fB = endo_at_vertex(R2, 2, B)

    # Covariant functoriality in the first argument:
    # Tor(fB o fA) = Tor(fB) o Tor(fA).
    F_A = TO.tor_map_first(T_R2_L, T_R2_L, fA; s=1)
    F_B = TO.tor_map_first(T_R2_L, T_R2_L, fB; s=1)
    F_BA = TO.tor_map_first(T_R2_L, T_R2_L, compose_morphism(fB, fA); s=1)
    @test F_BA == F_B * F_A

    # Noncommuting endomorphisms of L2 at vertex 1 (dims there are 2).
    C = [c(2) c(1); c(0) c(1)]
    D = [c(1) c(0); c(1) c(2)]
    gC = endo_at_vertex(L2, 1, C)
    gD = endo_at_vertex(L2, 1, D)

    # Covariant functoriality in the second argument:
    # Tor(gD o gC) = Tor(gD) o Tor(gC).
    G_C = TO.tor_map_second(T_R2_L2, T_R2_L2, gC; s=1)
    G_D = TO.tor_map_second(T_R2_L2, T_R2_L2, gD; s=1)
    G_DC = TO.tor_map_second(T_R2_L2, T_R2_L2, compose_morphism(gD, gC); s=1)
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
            FF.principal_upset(P, 2), FF.principal_downset(P, 2); scalar=one(K), field=field))
    B = IR.pmodule_from_fringe(one_by_one_fringe(P,
            FF.principal_upset(P, 1), FF.principal_downset(P, 2); scalar=one(K), field=field))
    C = IR.pmodule_from_fringe(one_by_one_fringe(P,
            FF.principal_upset(P, 1), FF.principal_downset(P, 1); scalar=one(K), field=field))

    # Explicit inclusion i: A -> B and projection p: B -> C (components per vertex).
    # dims(A) = (0,1,0), dims(B) = (1,1,0), dims(C) = (1,0,0).
    i = MD.PMorphism(A, B, [CM.zeros(field, 1, 0), CM.ones(field, 1, 1),         CM.zeros(field, 0, 0)])
    p = MD.PMorphism(B, C, [CM.ones(field, 1, 1),         CM.zeros(field, 0, 1), CM.zeros(field, 0, 0)])

    # Opposite poset
    Pop = FF.FinitePoset(transpose(FF.leq_matrix(P)))

    # Right module on P^op (simple at vertex 2).
    Rop = IR.pmodule_from_fringe(one_by_one_fringe(Pop,
            FF.principal_upset(Pop, 2), FF.principal_downset(Pop, 2); scalar=one(K), field=field))

    LES = DF.TorLongExactSequenceSecond(Rop, i, p, TO.DerivedFunctorOptions(maxdeg=2))

    TorRA = DF.Tor(Rop, A, TO.DerivedFunctorOptions(maxdeg=2))
    TorRB = DF.Tor(Rop, B, TO.DerivedFunctorOptions(maxdeg=2))
    TorRC = DF.Tor(Rop, C, TO.DerivedFunctorOptions(maxdeg=2))
    @test LES.maxdeg == 2
    @test length(LES.iH) == 3
    @test length(LES.pH) == 3
    @test length(LES.delta) == 3

    @test [TO.dim(LES.TorA, s) for s in 0:LES.maxdeg] == [TO.dim(TorRA, s) for s in 0:LES.maxdeg]
    @test [TO.dim(LES.TorB, s) for s in 0:LES.maxdeg] == [TO.dim(TorRB, s) for s in 0:LES.maxdeg]
    @test [TO.dim(LES.TorC, s) for s in 0:LES.maxdeg] == [TO.dim(TorRC, s) for s in 0:LES.maxdeg]

    # Tor algebra (exercise multiplication)
    T = DF.Tor(Rop, B, TO.DerivedFunctorOptions(model=:second, maxdeg=2))

    Aalg = DF.TorAlgebra(T; mu_chain_gen=DF.trivial_tor_product_generator(T))

    M00  = DF.multiplication_matrix(Aalg, 0, 0)
    M00b = DF.multiplication_matrix(Aalg, 0, 0)
    @test M00 == M00b

    if TO.dim(T, 1) > 0
        M01 = DF.multiplication_matrix(Aalg, 0, 1)
        @test all(M01 .== 0)
    end
end

@testset "hyperTor_map_first/second: induced maps on Tor_n" begin
    P = chain_poset(2)
    Pop = FF.FinitePoset(transpose(FF.leq_matrix(P)))

    # L = simple at vertex 1 on P.
    L = IR.pmodule_from_fringe(one_by_one_fringe(P,
            FF.principal_upset(P, 1), FF.principal_downset(P, 1); scalar=one(K), field=field))

    # Rop = simple at vertex 2 on P^op.
    Rop = IR.pmodule_from_fringe(one_by_one_fringe(Pop,
            FF.principal_upset(Pop, 2), FF.principal_downset(Pop, 2); scalar=one(K), field=field))

    # Complex concentrated in degree 0.
    C = TO.ModuleCochainComplex([L], MD.PMorphism{K}[]; tmin=0, check=true)

    HT = TO.hyperTor(Rop, C; maxlen=2)
    T  = DF.Tor(Rop, L, TO.DerivedFunctorOptions(maxdeg=2))

    # Tor_1 is known nonzero in this classical example.
    @test TO.dim(HT, 1) == TO.dim(T, 1)
    d1 = TO.dim(HT, 1)
    @test d1 > 0

    f2 = scalar_endo(Rop, c(2))
    f3 = scalar_endo(Rop, c(3))
    g2 = scalar_endo(L,   c(2))
    g3 = scalar_endo(L,   c(3))

    gC2 = TO.ModuleCochainMap(C, C, [g2]; check=true)
    gC3 = TO.ModuleCochainMap(C, C, [g3]; check=true)

    # --- Compare to Tor maps for degree-0 complexes ---
    F2h = TO.hyperTor_map_first(f2, HT, HT; n=1)
    F2t = TO.tor_map_first(f2, T, T; n=1)
    @test F2h == F2t
    @test F2h == c(2) .* CM.eye(field, d1)
    F2full = TO.ModuleComplexes.induced_map_on_cohomology(
        TO.derived_tensor_map_first(f2, HT.T, HT.T; check=true),
        HT.cohom,
        HT.cohom,
        -1,
    )
    @test F2h == F2full
    hcache = DF.HomSystemCache()
    F2full_cached = TO.ModuleComplexes.induced_map_on_cohomology(
        TO.derived_tensor_map_first(f2, HT.T, HT.T; check=true, cache=hcache),
        HT.cohom,
        HT.cohom,
        -1,
    )
    @test F2full_cached == F2full
    F2map_cached1 = TO.derived_tensor_map_first(f2, HT.T, HT.T; check=true, cache=hcache)
    F2map_cached2 = TO.derived_tensor_map_first(f2, HT.T, HT.T; check=true, cache=hcache)
    @test F2map_cached1 === F2map_cached2
    @test TO.ChainComplexes.is_cochain_map(F2map_cached1)
    @test TO.hyperTor_map_first(f2, HT, HT; n=1, check=true, cache=hcache) == F2h
    @test TO.tor_map_first(T, T, f2; n=1, cache=hcache) == F2t

    G2h = TO.hyperTor_map_second(gC2, HT, HT; n=1)
    G2t = TO.tor_map_second(g2, T, T; n=1)
    @test G2h == G2t
    @test G2h == c(2) .* CM.eye(field, d1)
    G2full = TO.ModuleComplexes.induced_map_on_cohomology(
        TO.derived_tensor_map_second(gC2, HT.T, HT.T; check=true),
        HT.cohom,
        HT.cohom,
        -1,
    )
    @test G2full == G2h
    G2full_cached = TO.ModuleComplexes.induced_map_on_cohomology(
        TO.derived_tensor_map_second(gC2, HT.T, HT.T; check=true, cache=hcache),
        HT.cohom,
        HT.cohom,
        -1,
    )
    @test G2full_cached == G2full
    G2map_cached1 = TO.derived_tensor_map_second(gC2, HT.T, HT.T; check=true, cache=hcache)
    G2map_cached2 = TO.derived_tensor_map_second(gC2, HT.T, HT.T; check=true, cache=hcache)
    @test G2map_cached1 === G2map_cached2
    @test TO.tor_map_second(g2, T, T; n=1, cache=hcache) == G2t

    plan_cached = TO.ModuleComplexes._tensor_map_first_plan(HT.T, HT.T, hcache)
    idx = findfirst(d -> !isempty(d.adeg), plan_cached.degrees)
    @test idx !== nothing
    dplan = plan_cached.degrees[idx]
    upto_cached = maximum(dplan.adeg; init=0)
    coeffs_cached = TO.ModuleComplexes._lift_projective_chainmap_coeff_cached(
        f2, HT.T.resR, HT.T.resR; upto=upto_cached, cache=hcache
    )
    block1 = TO._tensor_map_on_tor_chains_from_projective_coeff(
        dplan.terms[1],
        dplan.dom_gens[1],
        dplan.cod_gens[1],
        dplan.dom_offsets[1],
        dplan.cod_offsets[1],
        coeffs_cached[dplan.adeg[1] + 1];
        cache=hcache,
    )
    block2 = TO._tensor_map_on_tor_chains_from_projective_coeff(
        dplan.terms[1],
        dplan.dom_gens[1],
        dplan.cod_gens[1],
        dplan.dom_offsets[1],
        dplan.cod_offsets[1],
        coeffs_cached[dplan.adeg[1] + 1];
        cache=hcache,
    )
    @test block1 === block2

    coeffs_scalar = TO.ModuleComplexes._lift_projective_chainmap_coeff_uncached(
        f2, HT.T.resR, HT.T.resR; upto=upto_cached
    )
    for a in 0:upto_cached
        n = length(HT.T.resR.gens[a + 1])
        idxs = collect(1:n)
        expect = sparse(idxs, idxs, fill(c(2), n), n, n)
        @test coeffs_scalar[a + 1] == expect
    end

    # --- Identity behavior ---
    Fid = TO.hyperTor_map_first(IR.id_morphism(Rop), HT, HT; n=1)
    @test Fid == CM.eye(field, d1)

    Gid = TO.hyperTor_map_second(TO.ModuleComplexes.idmap(C), HT, HT; n=1)
    @test Gid == CM.eye(field, d1)

    # --- Functoriality in first variable ---
    f32 = compose_morphism(f3, f2)   # f3 o f2 = 6*id
    F32 = TO.hyperTor_map_first(f32, HT, HT; n=1)
    F3  = TO.hyperTor_map_first(f3,  HT, HT; n=1)
    @test F32 == F3 * F2h

    # --- Functoriality in second variable ---
    g32 = compose_morphism(g3, g2)
    gC32 = TO.ModuleCochainMap(C, C, [g32]; check=true)
    G32 = TO.hyperTor_map_second(gC32, HT, HT; n=1)
    G3  = TO.hyperTor_map_second(gC3,  HT, HT; n=1)
    @test G32 == G3 * G2h

    # --- Bifunctorial commutativity (natural in both vars) ---
    @test (G3 * F2h) == (F2h * G3)
end


@testset "Tor by hand on chain of length 2" begin
    P = chain_poset(2)
    Pop = FF.FinitePoset(transpose(FF.leq_matrix(P)))

    # L = simple at 1 on P
    Lfr = one_by_one_fringe(P, FF.principal_upset(P, 1), FF.principal_downset(P, 1); scalar=one(K), field=field)
    L = IR.pmodule_from_fringe(Lfr)

    # Rop = simple at 2 on Pop (= P^op)
    Rfr = one_by_one_fringe(Pop, FF.principal_upset(Pop, 2), FF.principal_downset(Pop, 2); scalar=one(K), field=field)
    Rop = IR.pmodule_from_fringe(Rfr)

    T = DF.Tor(Rop, L, TO.DerivedFunctorOptions(maxdeg=3))

    @test TO.dim(T, 0) == 0
    @test TO.dim(T, 1) == 1
    @test TO.dim(T, 2) == 0
    @test TO.dim(T, 3) == 0
end

@testset "Tor extra structure: LES, actions, bicomplex" begin
    # Poset: chain 1 < 2
    P = chain_poset(2)
    Pop = FF.FinitePoset(transpose(FF.leq_matrix(P)))  # opposite

    # Left modules on P
    S1 = _chain_module(P, [1, 0], CM.zeros(field, 0, 1), field)
    S2 = _chain_module(P, [0, 1], CM.zeros(field, 1, 0), field)
    P1 = _chain_module(P, [1, 1], CM.ones(field, 1, 1), field)  # projective at 1

    # Right modules (as P^op-modules) on Pop
    S1op = _chain_module(Pop, [1, 0], CM.zeros(field, 1, 0), field)
    S2op = _chain_module(Pop, [0, 1], CM.zeros(field, 0, 1), field)
    P2op = _chain_module(Pop, [1, 1], CM.ones(field, 1, 1), field)  # projective at 2 (in Pop)

    # Short exact sequence in the second variable: 0 -> S2 -> P1 -> S1 -> 0
    i = MD.PMorphism(S2, P1, [CM.zeros(field, 1, 0), CM.ones(field, 1, 1)])
    p = MD.PMorphism(P1, S1, [CM.ones(field, 1, 1), CM.zeros(field, 0, 1)])

    les2 = DF.TorLongExactSequenceSecond(S2op, i, p, TO.DerivedFunctorOptions(maxdeg=1))

    # Connecting map delta: Tor_1(S2op, S1) -> Tor_0(S2op, S2)
    # In this toy example it is nonzero (this is the standard non-split SES).
    @test size(les2.delta[1], 1) == 0
    @test size(les2.delta[2]) == (TO.dim(les2.TorA, 0), TO.dim(les2.TorC, 1))
    @test les2.delta[2][1, 1] != 0

    # Short exact sequence in the first variable: 0 -> S1op -> P2op -> S2op -> 0
    i1 = MD.PMorphism(S1op, P2op, [CM.ones(field, 1, 1), CM.zeros(field, 1, 0)])
    p1 = MD.PMorphism(P2op, S2op, [CM.zeros(field, 0, 1), CM.ones(field, 1, 1)])

    les1 = TO.TorLongExactSequenceFirst(S1, i1, p1, TO.DerivedFunctorOptions(maxdeg=1))

    @test size(les1.delta[2]) == (TO.dim(les1.TorA, 0), TO.dim(les1.TorC, 1))
    @test les1.delta[2][1, 1] != 0

    # Ext action on Tor via the resolve-second model:
    # The Ext^0 unit should act as identity on Tor_1.
    EA = TO.ExtAlgebra(S1, TO.DerivedFunctorOptions(maxdeg=2))
    Tsec = DF.Tor(S2op, S1, TO.DerivedFunctorOptions(model=:second); res=EA.E.res)
    u = DF.unit(EA)
    direct_old = DF.Algebras._EXT_ACTION_USE_DIRECT_STREAM[]
    coeff_old = DF.Functoriality._FUNCTORIALITY_DIRECT_COEFF_MIN_NNZ[]
    action_old = DF.Algebras._EXT_ACTION_DIRECT_STREAM_MIN_NNZ[]
    work_old = DF.Algebras._EXT_ACTION_DIRECT_STREAM_MIN_EST_WORK[]
    product_old = DF.Algebras._EXT_ACTION_DIRECT_STREAM_USE_PRODUCT_WORK[]
    try
        DF.Algebras._EXT_ACTION_USE_DIRECT_STREAM[] = false
        DF.Functoriality._FUNCTORIALITY_DIRECT_COEFF_MIN_NNZ[] = typemax(Int)
        DF.Algebras._EXT_ACTION_DIRECT_STREAM_MIN_NNZ[] = typemax(Int)
        DF.Algebras._EXT_ACTION_DIRECT_STREAM_MIN_EST_WORK[] = typemax(Int)
        DF.Algebras._EXT_ACTION_DIRECT_STREAM_USE_PRODUCT_WORK[] = false
        act_off = DF.ext_action_on_tor(EA, Tsec, u; s=1)

        DF.Algebras._EXT_ACTION_USE_DIRECT_STREAM[] = true
        DF.Functoriality._FUNCTORIALITY_DIRECT_COEFF_MIN_NNZ[] = 0
        DF.Algebras._EXT_ACTION_DIRECT_STREAM_MIN_NNZ[] = 0
        DF.Algebras._EXT_ACTION_DIRECT_STREAM_MIN_EST_WORK[] = 0
        DF.Algebras._EXT_ACTION_DIRECT_STREAM_USE_PRODUCT_WORK[] = true
        act_on = DF.ext_action_on_tor(EA, Tsec, u; s=1)

        @test act_on == act_off
        @test size(act_on) == (TO.dim(Tsec, 1), TO.dim(Tsec, 1))
        @test act_on[1, 1] == c(1)
    finally
        DF.Algebras._EXT_ACTION_USE_DIRECT_STREAM[] = direct_old
        DF.Functoriality._FUNCTORIALITY_DIRECT_COEFF_MIN_NNZ[] = coeff_old
        DF.Algebras._EXT_ACTION_DIRECT_STREAM_MIN_NNZ[] = action_old
        DF.Algebras._EXT_ACTION_DIRECT_STREAM_MIN_EST_WORK[] = work_old
        DF.Algebras._EXT_ACTION_DIRECT_STREAM_USE_PRODUCT_WORK[] = product_old
    end

    # Tor double complex total cohomology agrees with Tor groups (degree reindexing).
    # Using small lengths is enough for this example.
    DC = TO.TorDoubleComplex(S2op, S1; maxlen=1)
    Tot = CC.total_complex(DC)

    # Tor_n corresponds to H^{-n} of the total cochain complex.
    Tfirst = DF.Tor(S2op, S1, TO.DerivedFunctorOptions(maxdeg=2))
    @test CC.cohomology_data(Tot, 0).dimH == TO.dim(Tfirst, 0)
    @test CC.cohomology_data(Tot, -1).dimH == TO.dim(Tfirst, 1)
    @test CC.cohomology_data(Tot, -2).dimH == TO.dim(Tfirst, 2)

    # TorAlgebra infrastructure smoke test: a trivial degree-0 product.
    # Choose a projective right module so Tor_0 is 1-dim and higher Tor vanishes.
    T0 = DF.Tor(P2op, S2, TO.DerivedFunctorOptions(maxdeg=0))
    @test TO.dim(T0, 0) == 1

    Alg = DF.TorAlgebra(T0)
    DF.set_chain_product!(Alg, 0, 0, sparse(CM.ones(field, 1, 1)))
    M00 = DF.multiplication_matrix(Alg, 0, 0)
    @test size(M00) == (1, 1)
    @test M00[1, 1] == c(1)

    x = DF.element(Alg, 0, [c(1)])
    y = DF.multiply(Alg, x, x)
    @test y.deg == 0
    @test y.coords[1] == c(1)
end
end
end
