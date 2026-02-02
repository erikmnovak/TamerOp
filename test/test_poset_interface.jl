using Test

# Included from test/runtests.jl; uses shared aliases (PM, FF, ...).

@testset "Structured posets: leq and cover_edges" begin
    coords = (collect(1:2), collect(10:10:30))
    P = FF.GridPoset(coords)

    # Index mapping: first axis varies fastest.
    idx(i, j) = i + (j - 1) * length(coords[1])

    @test FF.leq(P, idx(1, 1), idx(2, 3))
    @test !FF.leq(P, idx(2, 3), idx(1, 1))
    @test FF.leq(P, idx(2, 2), idx(2, 3))

    C = FF.cover_edges(P)
    @test length(C) == 7
    @test C[idx(1, 1), idx(2, 1)]
    @test C[idx(2, 1), idx(2, 2)]
    @test !C[idx(1, 1), idx(2, 2)]

    P1 = chain_poset(2)
    P2 = chain_poset(2)
    Pprod = FF.ProductPoset(P1, P2)
    idxp(i, j) = i + (j - 1) * P1.n

    @test FF.leq(Pprod, idxp(1, 1), idxp(2, 2))
    @test !FF.leq(Pprod, idxp(2, 2), idxp(1, 1))

    Cprod = FF.cover_edges(Pprod)
    @test length(Cprod) == 4
    @test Cprod[idxp(1, 1), idxp(2, 1)]
    @test Cprod[idxp(1, 1), idxp(1, 2)]
    @test !Cprod[idxp(1, 1), idxp(2, 2)]
end

@testset "GridPoset rejects duplicate or unsorted coords" begin
    @test_throws ErrorException FF.GridPoset((Float64[0.0, 0.0, 1.0], Float64[0.0, 1.0]))
    @test_throws ErrorException FF.GridPoset((Float64[0.0, 2.0, 1.0], Float64[0.0, 1.0]))
end

@testset "GridPoset size does not scale like n^2" begin
    coords = (collect(1:200), collect(1:200))
    P = FF.GridPoset(coords)

    # A dense leq matrix would be enormous; GridPoset should be tiny.
    nverts = length(coords[1]) * length(coords[2])
    dense_bytes = div(nverts * nverts, 8)  # BitMatrix uses 1 bit per entry.

    @test Base.summarysize(P) < div(dense_bytes, 100)

    # leq should be allocation-free in a tight loop.
    alloc = @allocated begin
        for _ in 1:10_000
            FF.leq(P, 1, 1)
        end
    end
    @test alloc < 1_000_000
end

@testset "monotone_upper_closure works on structured posets" begin
    # 2x2 grid, indices (1,1)=1, (2,1)=2, (1,2)=3, (2,2)=4
    coords = (collect(1:2), collect(1:2))
    P = FF.GridPoset(coords)
    s = [1.0, 3.0, 2.0, 4.0]
    t = PM.Invariants._monotone_upper_closure(P, s)
    @test t == [1.0, 3.0, 2.0, 4.0]

    # Non-monotone input should be corrected to the minimal isotone majorant.
    s2 = [1.0, 0.0, 2.0, 3.0]
    t2 = PM.Invariants._monotone_upper_closure(P, s2)
    @test t2 == [1.0, 1.0, 2.0, 3.0]
end

@testset "encode_pmodules_to_common_poset works for structured posets" begin
    P1 = FF.GridPoset((collect(1:2), collect(1:2)))
    P2 = FF.ProductPoset(chain_poset(2), chain_poset(3))

    dims1 = ones(Int, FF.nvertices(P1))
    dims2 = ones(Int, FF.nvertices(P2))

    edge_maps1 = Dict{Tuple{Int,Int}, Matrix{QQ}}()
    for (u, v) in FF.cover_edges(P1)
        edge_maps1[(u, v)] = Matrix{QQ}(I, 1, 1)
    end
    edge_maps2 = Dict{Tuple{Int,Int}, Matrix{QQ}}()
    for (u, v) in FF.cover_edges(P2)
        edge_maps2[(u, v)] = Matrix{QQ}(I, 1, 1)
    end

    M1 = MD.PModule{QQ}(P1, dims1, edge_maps1)
    M2 = MD.PModule{QQ}(P2, dims2, edge_maps2)

    out = PM.ChangeOfPosets.encode_pmodules_to_common_poset(M1, M2; method=:product, check_poset=false)
    @test FF.nvertices(out.P) == FF.nvertices(P1) * FF.nvertices(P2)
    @test out.Ms[1].Q === out.P
    @test out.Ms[2].Q === out.P
    @test length(out.Ms[1].dims) == FF.nvertices(out.P)
    @test length(out.Ms[2].dims) == FF.nvertices(out.P)

    n1 = FF.nvertices(P1)
    @test out.pi1.Q === out.P
    @test out.pi2.Q === out.P
    @test out.pi1.P === P1
    @test out.pi2.P === P2
    @test out.pi1.pi_of_q[1] == 1
    @test out.pi2.pi_of_q[1] == 1
    @test out.pi1.pi_of_q[2] == 2
    @test out.pi2.pi_of_q[2] == 1
    @test out.pi1.pi_of_q[n1 + 1] == 1
    @test out.pi2.pi_of_q[n1 + 1] == 2
end

@testset "SignaturePoset leq and cover_edges" begin
    sig_y = BitVector[
        BitVector([false, false]),
        BitVector([true,  false]),
        BitVector([true,  true]),
    ]
    sig_z = BitVector[
        BitVector([false]),
        BitVector([false]),
        BitVector([false]),
    ]
    P = PM.ZnEncoding.SignaturePoset(sig_y, sig_z)
    @test FF.leq(P, 1, 2)
    @test FF.leq(P, 2, 3)
    @test !FF.leq(P, 3, 1)

    C = FF.cover_edges(P)
    @test length(C) == 2
    @test C[1, 2]
    @test C[2, 3]
    @test !C[1, 3]
end

@testset "encode_from_flanges supports poset_kind=:signature" begin
    tau = FZ.face(1, [false])
    flat = FZ.IndFlat(tau, [0])
    inj = FZ.IndInj(tau, [0])
    FG = FZ.Flange(1, [flat], [inj], reshape([QQ(1)], 1, 1))
    opts = CM.EncodingOptions(backend = :zn, max_regions = 16)

    P, Hs, pi = PM.ZnEncoding.encode_from_flanges([FG], opts; poset_kind = :signature)
    @test P isa PM.ZnEncoding.SignaturePoset
    @test Hs[1].P === P
    @test length(pi.sig_y) == FF.nvertices(P)
end

@testset "uptight encoding supports poset_kind=:regions" begin
    Q = chain_poset(3)
    U1 = FF.principal_upset(Q, 1)
    U2 = FF.principal_upset(Q, 2)
    D1 = FF.principal_downset(Q, 2)
    M = FF.FringeModule{QQ}(Q, [U1, U2], [D1], reshape([QQ(1), QQ(0)], 1, 2))

    upt = PM.Encoding.build_uptight_encoding_from_fringe(M; poset_kind = :regions)
    P = upt.pi.P
    @test P isa FF.RegionsPoset

    upt_dense = PM.Encoding.build_uptight_encoding_from_fringe(M; poset_kind = :dense)
    @test FF.nvertices(P) == FF.nvertices(upt_dense.pi.P)
    @test FF.leq_matrix(P) == FF.leq_matrix(upt_dense.pi.P)
end

@testset "Structured poset caches reuse by identity" begin
    P = FF.GridPoset((collect(1:3), collect(1:3)))
    C1 = FF.cover_edges(P)
    C2 = FF.cover_edges(P)
    @test C1 === C2

    P2 = FF.GridPoset((collect(1:2), collect(1:2)))
    Q1 = FF.ProductPoset(P, P2)
    Q2 = FF.ProductPoset(P, P2)
    prod1 = PM.ChangeOfPosets.product_poset(Q1, Q2; use_cache=true)
    prod2 = PM.ChangeOfPosets.product_poset(Q1, Q2; use_cache=true)
    @test prod1.P === prod2.P
end
