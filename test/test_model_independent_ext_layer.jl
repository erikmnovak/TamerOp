using Test
using LinearAlgebra
using SparseArrays

const FL = TamerOp.FieldLinAlg

# Included from test/runtests.jl; uses shared aliases (TO, FF, IR, ...).
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
        one_by_one_fringe(P, U, D; field=QQField()) -> FF.FringeModule

    A 1-generator / 1-cogenerator fringe module with phi = [1].
    Fiber at q is 1-dimensional iff q lies in U intersect D.
    """
    function one_by_one_fringe(P::FF.FinitePoset, U::FF.Upset, D::FF.Downset;
                               field::CM.AbstractCoeffField=CM.QQField())
        K = CM.coeff_type(field)
        phi = spzeros(K, 1, 1)
        phi[1, 1] = CM.coerce(field, 1)
        return FF.FringeModule{K}(P, [U], [D], phi; field=field)
    end
end

# ---------------------------------------------------------------------------
# Convenience constructors for *this test file* (do not assume public API).
# ---------------------------------------------------------------------------

"""
    simple_pmodule(P, v; field=QQField()) -> MD.PModule

The simple module supported only at vertex v (dimension 1 at v, 0 elsewhere).
"""
function simple_pmodule(P::FF.FinitePoset, v::Int; field::CM.AbstractCoeffField=CM.QQField())
    Hv = one_by_one_fringe(P, FF.principal_upset(P, v), FF.principal_downset(P, v); field=field)
    return IR.pmodule_from_fringe(Hv)
end

"""
    interval_pmodule(P, a, b; field=QQField()) -> MD.PModule

The interval module supported on {q : a <= q <= b}, built via a 1x1 fringe.
Requires a <= b in the poset.
"""
function interval_pmodule(P::FF.FinitePoset, a::Int, b::Int; field::CM.AbstractCoeffField=CM.QQField())
    @assert FF.leq(P, a, b)
    Hab = one_by_one_fringe(P, FF.principal_upset(P, a), FF.principal_downset(P, b); field=field)
    return IR.pmodule_from_fringe(Hab)
end

"""
    direct_sum(mods) -> MD.PModule

Direct sum of a list of modules on the same finite poset.
Cover-edge maps are block diagonal.
"""
function direct_sum(mods::AbstractVector{<:MD.PModule{K}}) where {K}
    @assert !isempty(mods)
    Q = mods[1].Q
    field = mods[1].field
    n = Q.n
    for M in mods
        @assert M.Q === Q
        M.field == field || error("direct_sum: field mismatch")
    end

    dims = [sum(M.dims[i] for M in mods) for i in 1:n]

    edge_maps = Dict{Tuple{Int,Int}, Matrix{K}}()
    for (u, v) in FF.cover_edges(Q)
        out_sizes = [M.dims[v] for M in mods]
        in_sizes  = [M.dims[u] for M in mods]

        total_out = sum(out_sizes)
        total_in  = sum(in_sizes)

        B = CM.zeros(field, total_out, total_in)

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

    return MD.PModule{K}(Q, dims, edge_maps; field=field)
end

"""
    endo_at_vertex(M, u, A) -> MD.PMorphism

Endomorphism of M which is identity at every vertex except u,
where it is replaced by A (size dims[u] x dims[u]).
"""
function endo_at_vertex(M::MD.PModule{K}, u::Int, A::AbstractMatrix{K}) where {K}
    comps = Vector{Matrix{K}}(undef, M.Q.n)
    for v in 1:M.Q.n
        dv = M.dims[v]
        comps[v] = CM.eye(M.field, dv)
    end
    comps[u] = Matrix{K}(A)
    return MD.PMorphism(M, M, comps)
end

"""
    compose_morphism(g, f) -> MD.PMorphism

Fiberwise composition: (g o f)_v = g_v * f_v.
"""
function compose_morphism(g::MD.PMorphism{K}, f::MD.PMorphism{K}) where {K}
    @assert f.cod === g.dom
    n = f.dom.Q.n
    comps = [g.comps[v] * f.comps[v] for v in 1:n]
    return MD.PMorphism(f.dom, g.cod, comps)
end

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@testset "Model-independent Ext layer: comparison + coherent transport" begin
    Q = chain_poset(2)
    field = CM.QQField()
    K = CM.coeff_type(field)
    c(x) = CM.coerce(field, x)

    # Simples at 1 and 2.
    S1 = simple_pmodule(Q, 1; field=field)
    S2 = simple_pmodule(Q, 2; field=field)

    # M = S1 oplus S1, N = S2 oplus S2.
    M = direct_sum([S1, S1])
    N = direct_sum([S2, S2])

    # Unified Ext object, canonical basis chosen from the projective model.
    E = DF.Ext(M, N, TO.DerivedFunctorOptions(maxdeg = 1, model = :unified, canon = :projective))
    cmp = getfield(E, :comparison)

    @test DF.dim(E, 1) == 4
    @test all(isnothing, cmp.P2I)
    @test all(isnothing, cmp.I2P)
    @test !cmp.complete

    # Comparison maps should be inverse isomorphisms.
    P2I = DF.comparison_isomorphism(E, 1; from = :projective, to = :injective)
    I2P = DF.comparison_isomorphism(E, 1; from = :injective, to = :projective)
    @test cmp.P2I[2] !== nothing
    @test cmp.I2P[2] !== nothing
    @test cmp.P2I[1] === nothing
    @test cmp.I2P[1] === nothing
    @test !cmp.complete

    famP2I, famI2P = DF.comparison_isomorphisms(E)
    @test length(famP2I) == 2
    @test famP2I[2] == P2I
    @test famI2P[2] == I2P
    _ = collect(famP2I)
    @test cmp.P2I[1] !== nothing
    @test cmp.I2P[1] !== nothing
    @test cmp.complete

    I4 = CM.eye(field, 4)
    @test P2I * I2P == I4
    @test I2P * P2I == I4

    # Coherent representative transport: coords (canonical) -> inj cocycle -> coords back.
    e1 = vcat([c(1)], zeros(K, 3))
    rep_inj = DF.representative(E, 1, e1; model = :injective)
    coords_back = DF.coordinates(E, 1, rep_inj; model = :injective)
    @test coords_back == e1
end

@testset "Injective model functoriality in second argument" begin
    Q = chain_poset(2)
    field = CM.QQField()
    K = CM.coeff_type(field)
    c(x) = CM.coerce(field, x)

    S1 = simple_pmodule(Q, 1; field=field)
    S2 = simple_pmodule(Q, 2; field=field)

    M = direct_sum([S1, S1])
    N = direct_sum([S2, S2])

    Einj = DF.ExtInjective(M, N, TO.DerivedFunctorOptions(maxdeg = 2))

    # Noncommuting endomorphisms of N at vertex 2 (dims there are 2).
    C = [c(1) c(1); c(0) c(1)]
    D = [c(1) c(0); c(1) c(1)]

    gC = endo_at_vertex(N, 2, C)
    gD = endo_at_vertex(N, 2, D)
    gDC = compose_morphism(gD, gC)

    GC  = DF.ext_map_second(Einj, Einj, gC;  t = 1)
    GD  = DF.ext_map_second(Einj, Einj, gD;  t = 1)
    GDC = DF.ext_map_second(Einj, Einj, gDC; t = 1)

    # Functoriality: Ext(M, gD o gC) = Ext(M, gD) o Ext(M, gC).
    @test GDC == GD * GC

    # Identity induces identity.
    idN = IR.id_morphism(N)
    Gid = DF.ext_map_second(Einj, Einj, idN; t = 1)

    d = DF.dim(Einj, 1)
    @test Gid == CM.eye(field, d)
end

@testset "Ext long exact sequence in first argument (degree 0 connecting map)" begin
    Q = chain_poset(2)
    field = CM.QQField()
    K = CM.coeff_type(field)
    c(x) = CM.coerce(field, x)

    S1 = simple_pmodule(Q, 1; field=field)
    S2 = simple_pmodule(Q, 2; field=field)
    I12 = interval_pmodule(Q, 1, 2; field=field)

    # Short exact sequence 0 -> S2 -> I12 -> S1 -> 0 on the chain 1 < 2.
    #
    # Build i: S2 -> I12 and p: I12 -> S1 with correctly-sized component matrices.
    i_comps = [CM.zeros(field, I12.dims[v], S2.dims[v]) for v in 1:Q.n]
    i_comps[2] = CM.ones(field, 1, 1)
    i = MD.PMorphism(S2, I12, i_comps)

    p_comps = [CM.zeros(field, S1.dims[v], I12.dims[v]) for v in 1:Q.n]
    p_comps[1] = CM.ones(field, 1, 1)
    p = MD.PMorphism(I12, S1, p_comps)

    df0 = TO.DerivedFunctorOptions(maxdeg=0)
    les = DF.ExtLongExactSequenceFirst(S2, I12, S1, S2, i, p, df0)

    # Connecting map delta: Ext^0(S2,S2) -> Ext^1(S1,S2) should be nonzero (rank 1).
    @test FL.rank(field, les.delta[1]) == 1
    @test size(les.delta[1]) == (DF.dim(les.EC, 1), DF.dim(les.EA, 0))
end

@testset "Injective chain-map lifting (public API)" begin
    Q = chain_poset(2)
    field = CM.QQField()
    K = CM.coeff_type(field)
    c(x) = CM.coerce(field, x)
    S1 = simple_pmodule(Q, 1; field=field)
    S2 = simple_pmodule(Q, 2; field=field)
    N = MD.direct_sum(S1, S2)

    # Endomorphism: scale vertex 1 by 2 and vertex 2 by 3.
    g = MD.PMorphism(N, N, [fill(c(2), 1, 1), fill(c(3), 1, 1)])

    res = DF.injective_resolution(N, TO.ResolutionOptions(maxlen=2))
    phis = DF.lift_injective_chainmap(g, res, res; upto=2)

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
    e = res.Emods[1].edge_maps[1, 2]
    @test phis[1].comps[2] * e == e * phis[1].comps[1]

    # Convenience wrapper builds resolutions too.
    lifted = DF.lift_injective_chainmap(g; maxlen=2)
    @test length(lifted.phis) == 3
    @test lifted.res_dom.N === N
    @test lifted.res_cod.N === N
end
