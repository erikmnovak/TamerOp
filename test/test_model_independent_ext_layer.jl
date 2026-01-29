using Test
using LinearAlgebra
using SparseArrays

# Included from test/runtests.jl; uses shared aliases (PM, FF, IR, QQ, ...).
#
# The fallback helpers below are only for standalone runs; the main test suite
# does not use them.


# ---------------------------------------------------------------------------
# Fallback helpers (only used if this file is run standalone).
# When included from runtests.jl, these are usually already defined globally.
# ---------------------------------------------------------------------------

if !isdefined(@__MODULE__, :chain_poset)
    """
        chain_poset(n) -> FF.FinitePoset

    Construct the chain (total order) on {1,2,...,n}.
    """
    function chain_poset(n::Int)
        leq = falses(n, n)
        for i in 1:n
            for j in i:n
                leq[i, j] = true
            end
        end
        return FF.FinitePoset(leq)
    end
end

if !isdefined(@__MODULE__, :one_by_one_fringe)
    """
        one_by_one_fringe(P, U, D) -> FF.FringeModule{QQ}

    A 1-generator / 1-cogenerator fringe module with phi = [1].
    Fiber at q is 1-dimensional iff q lies in U intersect D.
    """
    function one_by_one_fringe(P::FF.FinitePoset, U::FF.Upset, D::FF.Downset)
        phi = spzeros(QQ, 1, 1)
        phi[1, 1] = QQ(1)
        return FF.FringeModule{QQ}(P, [U], [D], phi)
    end
end

# ---------------------------------------------------------------------------
# Convenience constructors for *this test file* (do not assume public API).
# ---------------------------------------------------------------------------

"""
    simple_pmodule(P, v) -> MD.PModule{QQ}

The simple module supported only at vertex v (dimension 1 at v, 0 elsewhere).
"""
function simple_pmodule(P::FF.FinitePoset, v::Int)
    Hv = one_by_one_fringe(P, FF.principal_upset(P, v), FF.principal_downset(P, v))
    return IR.pmodule_from_fringe(Hv)
end

"""
    interval_pmodule(P, a, b) -> MD.PModule{QQ}

The interval module supported on {q : a <= q <= b}, built via a 1x1 fringe.
Requires a <= b in the poset.
"""
function interval_pmodule(P::FF.FinitePoset, a::Int, b::Int)
    @assert P.leq[a, b]
    Hab = one_by_one_fringe(P, FF.principal_upset(P, a), FF.principal_downset(P, b))
    return IR.pmodule_from_fringe(Hab)
end

"""
    direct_sum(mods) -> MD.PModule{QQ}

Direct sum of a list of modules on the same finite poset.
Cover-edge maps are block diagonal.
"""
function direct_sum(mods::Vector{MD.PModule{QQ}})
    @assert !isempty(mods)
    Q = mods[1].Q
    n = Q.n
    for M in mods
        @assert M.Q === Q
    end

    dims = [sum(M.dims[i] for M in mods) for i in 1:n]

    edge_maps = Dict{Tuple{Int,Int}, Matrix{QQ}}()
    for (u, v) in FF.cover_edges(Q)
        out_sizes = [M.dims[v] for M in mods]
        in_sizes  = [M.dims[u] for M in mods]

        total_out = sum(out_sizes)
        total_in  = sum(in_sizes)

        B = zeros(QQ, total_out, total_in)

        roff = 0
        coff = 0
        for (k, M) in enumerate(mods)
            r = out_sizes[k]
            c = in_sizes[k]
            # We iterate over cover edges, so the store guarantees a map exists.
            Auv = M.edge_maps[u, v]

            if r > 0 && c > 0
                B[(roff + 1):(roff + r), (coff + 1):(coff + c)] .= Auv
            end

            roff += r
            coff += c
        end

        edge_maps[(u, v)] = B
    end

    return MD.PModule{QQ}(Q, dims, edge_maps)
end

"""
    endo_at_vertex(M, u, A) -> MD.PMorphism{QQ}

Endomorphism of M which is identity at every vertex except u,
where it is replaced by A (size dims[u] x dims[u]).
"""
function endo_at_vertex(M::MD.PModule{QQ}, u::Int, A::Matrix{QQ})
    comps = Matrix{QQ}[]
    for v in 1:M.Q.n
        dv = M.dims[v]
        push!(comps, Matrix{QQ}(I, dv, dv))
    end
    comps[u] = A
    return MD.PMorphism{QQ}(M, M, comps)
end

"""
    compose_morphism(g, f) -> MD.PMorphism{QQ}

Fiberwise composition: (g o f)_v = g_v * f_v.
"""
function compose_morphism(g::MD.PMorphism{QQ}, f::MD.PMorphism{QQ})
    @assert f.cod === g.dom
    n = f.dom.Q.n
    comps = [g.comps[v] * f.comps[v] for v in 1:n]
    return MD.PMorphism{QQ}(f.dom, g.cod, comps)
end

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@testset "Model-independent Ext layer: comparison + coherent transport" begin
    Q = chain_poset(2)

    # Simples at 1 and 2.
    S1 = simple_pmodule(Q, 1)
    S2 = simple_pmodule(Q, 2)

    # M = S1 oplus S1, N = S2 oplus S2.
    M = direct_sum([S1, S1])
    N = direct_sum([S2, S2])

    # Unified Ext object, canonical basis chosen from the projective model.
    E = PM.Ext(M, N, PM.DerivedFunctorOptions(maxdeg = 1, model = :unified, canon = :projective))

    @test PM.dim(E, 1) == 4

    # Comparison maps should be inverse isomorphisms.
    P2I = PM.comparison_isomorphism(E, 1; from = :projective, to = :injective)
    I2P = PM.comparison_isomorphism(E, 1; from = :injective, to = :projective)

    I4 = Matrix{QQ}(I, 4, 4)
    @test P2I * I2P == I4
    @test I2P * P2I == I4

    # Coherent representative transport: coords (canonical) -> inj cocycle -> coords back.
    e1 = vcat([QQ(1)], zeros(QQ, 3))
    rep_inj = PM.representative(E, 1, e1; model = :injective)
    coords_back = PM.coordinates(E, 1, rep_inj; model = :injective)
    @test coords_back == e1
end

@testset "Injective model functoriality in second argument" begin
    Q = chain_poset(2)

    S1 = simple_pmodule(Q, 1)
    S2 = simple_pmodule(Q, 2)

    M = direct_sum([S1, S1])
    N = direct_sum([S2, S2])

    Einj = PM.ExtInjective(M, N, PM.DerivedFunctorOptions(maxdeg = 2))

    # Noncommuting endomorphisms of N at vertex 2 (dims there are 2).
    C = [QQ(1) QQ(1); QQ(0) QQ(1)]
    D = [QQ(1) QQ(0); QQ(1) QQ(1)]

    gC = endo_at_vertex(N, 2, C)
    gD = endo_at_vertex(N, 2, D)
    gDC = compose_morphism(gD, gC)

    GC  = PM.ext_map_second(Einj, Einj, gC;  t = 1)
    GD  = PM.ext_map_second(Einj, Einj, gD;  t = 1)
    GDC = PM.ext_map_second(Einj, Einj, gDC; t = 1)

    # Functoriality: Ext(M, gD o gC) = Ext(M, gD) o Ext(M, gC).
    @test GDC == GD * GC

    # Identity induces identity.
    idN = IR.id_morphism(N)
    Gid = PM.ext_map_second(Einj, Einj, idN; t = 1)

    d = PM.dim(Einj, 1)
    @test Gid == Matrix{QQ}(I, d, d)
end

@testset "Ext long exact sequence in first argument (degree 0 connecting map)" begin
    Q = chain_poset(2)

    S1 = simple_pmodule(Q, 1)
    S2 = simple_pmodule(Q, 2)
    I12 = interval_pmodule(Q, 1, 2)

    # Short exact sequence 0 -> S2 -> I12 -> S1 -> 0 on the chain 1 < 2.
    #
    # Build i: S2 -> I12 and p: I12 -> S1 with correctly-sized component matrices.
    i_comps = [zeros(QQ, I12.dims[v], S2.dims[v]) for v in 1:Q.n]
    i_comps[2] = ones(QQ, 1, 1)
    i = MD.PMorphism{QQ}(S2, I12, i_comps)

    p_comps = [zeros(QQ, S1.dims[v], I12.dims[v]) for v in 1:Q.n]
    p_comps[1] = ones(QQ, 1, 1)
    p = MD.PMorphism{QQ}(I12, S1, p_comps)

    df0 = PM.DerivedFunctorOptions(maxdeg=0)
    les = PM.ExtLongExactSequenceFirst(S2, I12, S1, S2, i, p, df0)

    # Connecting map delta: Ext^0(S2,S2) -> Ext^1(S1,S2) should be nonzero (rank 1).
    @test PM.rankQQ(les.delta[1]) == 1
    @test size(les.delta[1]) == (PM.dim(les.EC, 1), PM.dim(les.EA, 0))
end

@testset "Injective chain-map lifting (public API)" begin
    Q = chain_poset(2)
    S1 = simple_pmodule(Q, 1)
    S2 = simple_pmodule(Q, 2)
    N = PM.direct_sum([S1, S2])

    # Endomorphism: scale vertex 1 by 2 and vertex 2 by 3.
    g = MD.PMorphism{QQ}(N, N, [fill(QQ(2), 1, 1), fill(QQ(3), 1, 1)])

    res = PM.injective_resolution(N, PM.ResolutionOptions(maxlen=2))
    phis = PM.lift_injective_chainmap(g, res, res; upto=2)

    @test length(phis) == 3

    # Degree 0: phi0 o iota0 = iota0 o g
    for u in 1:Q.n
        @test phis[1].comps[u] * res.iota0.comps[u] == res.iota0.comps[u] * g.comps[u]
    end

    # Chain map condition: phi^k o d^{k-1} = d^{k-1} o phi^{k-1}
    for k in 1:2
        for u in 1:Q.n
            @test phis[k+1].comps[u] * res.d_mor[k].comps[u] ==
                  res.d_mor[k].comps[u] * phis[k].comps[u]
        end
    end

    # Sanity: phi0 commutes with the unique edge (1,2) on E0.
    e = res.Emods[1].edges[(1,2)]
    @test phis[1].comps[2] * e == e * phis[1].comps[1]

    # Convenience wrapper builds resolutions too.
    lifted = PM.lift_injective_chainmap(g; maxlen=2)
    @test length(lifted.phis) == 3
    @test lifted.res_dom.N === N
    @test lifted.res_cod.N === N
end