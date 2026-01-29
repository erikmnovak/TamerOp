using Test
using SparseArrays
using LinearAlgebra
using Random

# Included from test/runtests.jl.
# Uses shared aliases defined there (PM, FF, EN, IR, DF, MD, EX, QQ, ...).


@testset "Finite encoding from fringe (Defs 4.12-4.18)" begin
    P = chain_poset(3)
    U2 = FF.principal_upset(P, 2)
    D2 = FF.principal_downset(P, 2)

    M = one_by_one_fringe(P, U2, D2)

    enc = EN.build_uptight_encoding_from_fringe(M)
    pi = enc.pi

    # pi should be order-preserving: i <= j in Q => pi(i) <= pi(j) in P_Y
    for i in 1:pi.Q.n, j in 1:pi.Q.n
        if pi.Q.leq[i,j]
            @test pi.P.leq[pi.pi_of_q[i], pi.pi_of_q[j]]
        end
    end

    # Each generator upset U_i and each death downset D_j should be unions of fibers.
    # Hence preimage(image(U)) = U and preimage(image(D)) = D.
    for U in M.U
        Uhat = EN.image_upset(pi, U)
        Uback = EN.preimage_upset(pi, Uhat)
        @test Uback.mask == U.mask
    end
    for D in M.D
        Dhat = EN.image_downset(pi, D)
        Dback = EN.preimage_downset(pi, Dhat)
        @test Dback.mask == D.mask
    end

    # Build the induced module on the encoded poset by pushing U,D forward.
    Hhat = EN.pushforward_fringe_along_encoding(M, pi)


    # Pull back and recover the original U,D exactly.
    M2 = EN.pullback_fringe_along_encoding(Hhat, pi)
    @test M2.U[1].mask == M.U[1].mask
    @test M2.D[1].mask == M.D[1].mask
    for q in 1:P.n
        @test FF.fiber_dimension(M2, q) == FF.fiber_dimension(M, q)
    end
end

@testset "Uptight poset uses transitive closure (Example 4.16)" begin
    # Miller Example 4.16 exhibits non-transitivity of the raw "exists a<=b" relation
    # between uptight regions; Definition 4.17 then defines the uptight poset as its
    # transitive closure. We build a finite truncation of the N^2 example to test that
    # Encoding._uptight_poset really closes transitively.

    # Q = {0..3} x {0..2} with product order.
    function grid_poset(ax::Int, by::Int)
        coords = Tuple{Int,Int}[]
        for a in 0:ax, b in 0:by
            push!(coords, (a, b))
        end
        nQ = length(coords)
        leq = falses(nQ, nQ)
        for i in 1:nQ, j in 1:nQ
            ai, bi = coords[i]
            aj, bj = coords[j]
            leq[i, j] = (ai <= aj) && (bi <= bj)
        end
        Q = FF.FinitePoset(leq)
        idx = Dict{Tuple{Int,Int}, Int}()
        for (i, c) in enumerate(coords)
            idx[c] = i
        end
        return Q, coords, idx
    end

    Q, coords, idx = grid_poset(3, 2)

    # Upsets corresponding to the monomial ideals in Example 4.16:
    #   U1 = <x^2, y>    => (a>=2) or (b>=1)
    #   U2 = <x^3, y>    => (a>=3) or (b>=1)
    #   U3 = <x*y>       => (a>=1) and (b>=1)
    #   U4 = <x^2*y>     => (a>=2) and (b>=1)
    U1 = FF.upset_closure(Q, BitVector([(a >= 2) || (b >= 1) for (a, b) in coords]))
    U2 = FF.upset_closure(Q, BitVector([(a >= 3) || (b >= 1) for (a, b) in coords]))
    U3 = FF.upset_closure(Q, BitVector([(a >= 1) && (b >= 1) for (a, b) in coords]))
    U4 = FF.upset_closure(Q, BitVector([(a >= 2) && (b >= 1) for (a, b) in coords]))

    # No deaths: we only need the upsets to form Y for uptight signatures.
    phi0 = spzeros(QQ, 0, 4)
    M = FF.FringeModule{QQ}(Q, [U1, U2, U3, U4], FF.Downset[], phi0)

    enc = EN.build_uptight_encoding_from_fringe(M)
    pi = enc.pi
    P = pi.P

    # Identify the regions by representative lattice points (degrees):
    #   A: x^2  = (2,0) has signature {U1}
    #   B: x^3  = (3,0) and y = (0,1) share signature {U1,U2}
    #   C: x*y  = (1,1) has signature {U1,U2,U3}
    q_x2 = idx[(2, 0)]
    q_x3 = idx[(3, 0)]
    q_y  = idx[(0, 1)]
    q_xy = idx[(1, 1)]

    A = pi.pi_of_q[q_x2]
    B = pi.pi_of_q[q_x3]
    B2 = pi.pi_of_q[q_y]
    C = pi.pi_of_q[q_xy]

    @test B == B2
    @test length(Set([A, B, C])) == 3

    # In the transitive closure P, we must have A <= B <= C, hence A <= C.
    @test P.leq[A, B]
    @test P.leq[B, C]
    @test P.leq[A, C]

    # But in the underlying "exists a<=c" relation on regions, there is no witness for A <= C:
    # the only point in A is (2,0), and every point in C has x=1, so (2,0) is not <= any c in C.
    has_witness = false
    for a in 1:Q.n, c in 1:Q.n
        if pi.pi_of_q[a] == A && pi.pi_of_q[c] == C && Q.leq[a, c]
            has_witness = true
            break
        end
    end
    @test !has_witness
end


@testset "Workflow coarsen (uptight compression)" begin
    P = chain_poset(4)

    # A tiny 1 times 1 fringe module on the chain.
    U = FF.principal_upset(P, 2)
    D = FF.principal_downset(P, 3)
    Phi = sparse([1], [1], [QQ(1)], 1, 1)
    H = FF.FringeModule{QQ}(P, [U], [D], Phi)

    # A minimal ambient encoding map: point (q,) is sent to region q.
    struct DummyEncodingMap <: CM.AbstractPLikeEncodingMap
        n::Int
    end
    CM.locate(pi::DummyEncodingMap, x::NTuple{1,Int}) =
        (1 <= x[1] <= pi.n ? x[1] : 0)
    CM.locate(pi::DummyEncodingMap, x::AbstractVector{<:Integer}) =
        (length(x) >= 1 && 1 <= x[1] <= pi.n ? Int(x[1]) : 0)
    CM.dimension(::DummyEncodingMap) = 1
    CM.axes_from_encoding(pi::DummyEncodingMap) = (collect(1:pi.n),)
    CM.representatives(pi::DummyEncodingMap) = [(i,) for i in 1:pi.n]

    pi0 = DummyEncodingMap(P.n)

    # Construct an EncodingResult manually.
    enc = CM.EncodingResult(P, IR.pmodule_from_fringe(H), pi0;
        H = H,
        backend = :dummy,
        meta = Dict{Symbol,Any}(),
    )

    enc2 = PM.coarsen(enc)

    upt = EN.build_uptight_encoding_from_fringe(H)
    pi = upt.pi

    # New region poset size is the (uptight) compression of the old one.
    @test enc2.P.n == pi.P.n

    # Classifier should be postcomposition: (q,) <-> pi(q).
    for q in 1:P.n
        @test CM.locate(enc2.pi, (q,)) == pi.pi_of_q[q]
    end

    # Representatives are forwarded to the coarse regions (and cached).
    reps2 = CM.representatives(enc2.pi)
    @test length(reps2) == enc2.P.n
    for p in 1:enc2.P.n
        @test CM.locate(enc2.pi, reps2[p]) == p
    end

    # Fringe module was pushed forward along the coarsening map.
    H2_expected = EN.pushforward_fringe_along_encoding(H, pi)
    @test enc2.H.U == H2_expected.U
    @test enc2.H.D == H2_expected.D
    @test enc2.H.phi == H2_expected.phi
end


@testset "JSON serialization round-trips" begin
    # Flange round-trip
    n = 1
    tau0 = FZ.Face(n, [false])
    F1 = FZ.IndFlat(tau0, [1]; id=:F1)
    E1 = FZ.IndInj(tau0, [3]; id=:E1)
    Phi = reshape(QQ[QQ(1)], 1, 1)
    FG = FZ.Flange{QQ}(n, [F1], [E1], Phi)

    mktempdir() do dir
        path = joinpath(dir, "flange.json")
        SER.save_flange_json(path, FG)
        FG2 = SER.load_flange_json(path)

        @test FG2.n == FG.n
        @test length(FG2.flats) == length(FG.flats)
        @test length(FG2.injectives) == length(FG.injectives)
        @test FG2.phi == FG.phi
        @test FG2.flats[1].b == FG.flats[1].b
        @test FG2.injectives[1].b == FG.injectives[1].b
        @test FG2.flats[1].tau.coords == FG.flats[1].tau.coords
        @test FG2.injectives[1].tau.coords == FG.injectives[1].tau.coords
    end

    # Finite encoding fringe round-trip
    P = chain_poset(3)
    M = one_by_one_fringe(P, FF.principal_upset(P, 2), FF.principal_downset(P, 2))

    mktempdir() do dir
        path = joinpath(dir, "encoding.json")
        SER.save_encoding_json(path, M)
        M_loaded = SER.load_encoding_json(path)

        @test M_loaded.P.n == M.P.n
        @test M_loaded.P.leq == M.P.leq
        @test M_loaded.U[1].mask == M.U[1].mask
        @test M_loaded.D[1].mask == M.D[1].mask
        @test Matrix(M_loaded.phi) == Matrix(M.phi)
        for q in 1:M.P.n
            @test FF.fiber_dimension(M_loaded, q) == FF.fiber_dimension(M, q)
        end
    end


    # M2/Singular bridge parser (pure JSON input)
    json = """
    {
      "n": 1,
      "field": "QQ",
      "flats":      [ {"b":[1], "tau":[false], "id":"F1"} ],
      "injectives": [ {"b":[3], "tau":[false], "id":"E1"} ],
      "phi": [ ["1/1"] ]
    }
    """
    FG3 = SER.parse_flange_json(json)
    @test FG3.n == 1
    @test length(FG3.flats) == 1
    @test length(FG3.injectives) == 1
    @test FG3.phi[1,1] == QQ(1)
end

@testset "M2SingularBridge.parse_flange_json edge cases" begin
    # tau given as index list (1-based) rather than Bool vector; phi omitted -> canonical_matrix
    json1 = """
    {
      "n": 1,
      "flats": [ { "id": "F", "b": [0], "tau": [1] } ],
      "injectives": [ { "id": "E", "b": [0], "tau": [1] } ]
    }
    """
    H1 = SER.parse_flange_json(json1)
    @test H1.n == 1
    @test length(H1.flats) == 1
    @test length(H1.injectives) == 1
    @test Matrix(H1.phi) == reshape(QQ[1], 1, 1)

    # Explicit phi entries can be rationals in string form.
    json2 = """
    {
      "n": 1,
      "flats": [ { "id": "F", "b": [0], "tau": [false] } ],
      "injectives": [ { "id": "E", "b": [0], "tau": [false] } ],
      "phi": [ [ "-2/3" ] ]
    }
    """
    H2 = SER.parse_flange_json(json2)
    @test Matrix(H2.phi)[1, 1] == (-QQ(2) / QQ(3))

    # Non-intersecting flat/injective pairs must force Phi entries to 0 (monomial condition).
    json3 = """
    {
      "n": 1,
      "flats": [ { "id": "F", "b": [5], "tau": [false] } ],
      "injectives": [ { "id": "E", "b": [3], "tau": [false] } ],
      "phi": [ [ 1 ] ]
    }
    """
    H3 = SER.parse_flange_json(json3)
    @test Matrix(H3.phi) == reshape(QQ[0], 1, 1)
    @test FZ.dim_at(H3, [0]) == 0
end

@testset "Serialization.load_encoding_json strict schema" begin
    json = """
    {
      "kind": "FiniteEncodingFringe",
      "poset": {
        "n": 3,
        "leq": [
          [true,  true,  true],
          [false, true,  true],
          [false, false, true]
        ]
      },
      "U": [[false, true,  true]],
      "D": [[true,  true,  false]],
      "phi": [["-2/3"]]
    }
    """
    path, io = mktemp()
    write(io, json)
    close(io)

    M = SER.load_encoding_json(path)
    @test M.P.n == 3
    @test M.P.leq == BitMatrix([1 1 1; 0 1 1; 0 0 1])
    @test length(M.U) == 1
    @test length(M.D) == 1
    @test M.U[1].mask == BitVector([false, true, true])
    @test M.D[1].mask == BitVector([true, true, false])
    @test Matrix(M.phi) == reshape(QQ[QQ(-2,3)], 1, 1)
end

@testset "Serialization.load_encoding_json rejects legacy schema" begin
    json_old = """
    {
      "P": {"n": 3, "leq": [[1,1,1],[0,1,1],[0,0,1]]},
      "U": [[0,1,1]],
      "D": [[1,1,0]],
      "phi": [["-2/3"]]
    }
    """
    path, io = mktemp()
    write(io, json_old)
    close(io)

    @test_throws ErrorException SER.load_encoding_json(path)
end


# Helper: build an antichain poset with n elements
function antichain_poset(n::Int)
    leq = falses(n, n)
    for i in 1:n
        leq[i,i] = true
    end
    return FF.FinitePoset(leq)
end

# Helper: "V" poset with 1<2 and 1<3
function v_poset()
    leq = falses(3,3)
    for i in 1:3
        leq[i,i] = true
    end
    leq[1,2] = true
    leq[1,3] = true
    return FF.FinitePoset(leq)
end

# Poset of nonempty proper faces of a triangle (order complex = S^1):
# {1},{2},{3},{1,2},{1,3},{2,3} ordered by inclusion.
function triangle_boundary_poset()
    n = 6
    leq = falses(n, n)
    for i in 1:n
        leq[i,i] = true
    end
    leq[1,4] = true
    leq[1,5] = true
    leq[2,4] = true
    leq[2,6] = true
    leq[3,5] = true
    leq[3,6] = true
    return FF.FinitePoset(n, leq)
end

@testset "Change-of-posets: pullback / Kan extensions / derived" begin

    # -------------------------------------------------------------------------
    # Pullback along collapse chain2 -> point
    # -------------------------------------------------------------------------
    Q = chain_poset(2)
    P = chain_poset(1)
    pi = EN.EncodingMap(Q, P, [1,1])

    N = MD.PModule{QQ}(P, [3], Dict{Tuple{Int,Int},SparseMatrixCSC{QQ,Int}}())
    pbN = PM.pullback(pi, N)

    @test pbN.Q.n == 2
    @test pbN.dims == [3,3]
    @test Matrix(pbN.edge_maps[1, 2]) == Matrix{QQ}(I, 3, 3)

    # -------------------------------------------------------------------------
    # Left/right Kan extension collapse chain2 -> point (terminal/initial fast path)
    # -------------------------------------------------------------------------
    A = spzeros(QQ, 2, 1)
    A[1,1] = QQ(1)  # 1 -> first coordinate

    M = MD.PModule{QQ}(Q, [1,2], Dict((1,2)=>A))

    Lan = PM.pushforward_left(pi, M)
    Ran = PM.pushforward_right(pi, M)

    @test Lan.Q.n == 1
    @test Lan.dims == [2]   # terminal object is vertex 2
    @test Ran.dims == [1]   # initial object is vertex 1

    # -------------------------------------------------------------------------
    # Functoriality on morphisms (collapse)
    # -------------------------------------------------------------------------
    # Choose a module endomorphism commuting with A:
    # f1 = [3], f2 = diag(3,5) works because A hits only coord1.
    f1 = QQ[3]
    f2 = [QQ(3) QQ(0);
          QQ(0) QQ(5)]
    f = MD.PMorphism(M, M, [reshape(f1,1,1), f2])

    Lan_f = PM.pushforward_left(pi, f)
    Ran_f = PM.pushforward_right(pi, f)

    @test Matrix(Lan_f.comps[1]) == f2
    @test Matrix(Ran_f.comps[1]) == reshape(f1,1,1)

    # -------------------------------------------------------------------------
    # Identity encoding map should give identity functors (fast paths guarantee exact equality)
    # -------------------------------------------------------------------------
    pid = EN.EncodingMap(Q, Q, [1,2])
    pb_id = PM.pullback(pid, M)
    Lan_id = PM.pushforward_left(pid, M)
    Ran_id = PM.pushforward_right(pid, M)

    @test pb_id.dims == M.dims
    @test Matrix(pb_id.edge_maps[1, 2]) == Matrix(M.edge_maps[1, 2])

    @test Lan_id.dims == M.dims
    @test Matrix(Lan_id.edge_maps[1, 2]) == Matrix(M.edge_maps[1, 2])

    @test Ran_id.dims == M.dims
    @test Matrix(Ran_id.edge_maps[1, 2]) == Matrix(M.edge_maps[1, 2])

    # -------------------------------------------------------------------------
    # Nontrivial colimit: V-poset collapsed to point with identity maps gives dim 1
    # -------------------------------------------------------------------------
    Qv = v_poset()
    Pv = chain_poset(1)
    piv = EN.EncodingMap(Qv, Pv, [1,1,1])

    # dims all 1, maps 1->2 and 1->3 are identity
    Ev = Dict{Tuple{Int,Int},SparseMatrixCSC{QQ,Int}}()
    for (u,v) in FF.cover_edges(Qv).edges
        mat = spzeros(QQ, 1, 1)
        mat[1,1] = QQ(1)
        Ev[(u,v)] = mat
    end
    Mv = MD.PModule{QQ}(Qv, [1,1,1], Ev)

    Lan_v = PM.pushforward_left(piv, Mv)
    @test Lan_v.dims == [1]   # pushout of k <- k -> k is k

    # -------------------------------------------------------------------------
    # General-case limit/colimit: antichain of 2 collapsed to point -> direct sum (dim 5)
    # -------------------------------------------------------------------------
    Qa = antichain_poset(2)
    Pa = chain_poset(1)
    pia = EN.EncodingMap(Qa, Pa, [1,1])

    Ma = MD.PModule{QQ}(Qa, [2,3], Dict{Tuple{Int,Int},SparseMatrixCSC{QQ,Int}}())
    Lan_a = PM.pushforward_left(pia, Ma)
    Ran_a = PM.pushforward_right(pia, Ma)

    @test Lan_a.dims == [5]
    @test Ran_a.dims == [5]

    # -------------------------------------------------------------------------
    # Derived functors vanish for collapse with terminal/initial objects
    # -------------------------------------------------------------------------
    Lmods = PM.Lpushforward_left(pi, M, CM.DerivedFunctorOptions(maxdeg=2))
    Rmods = PM.Rpushforward_right(pi, M, CM.DerivedFunctorOptions(maxdeg=2))

    @test Lmods[1].dims == Lan.dims
    @test Lmods[2].dims == [0]
    @test Lmods[3].dims == [0]

    @test Rmods[1].dims == Ran.dims
    @test Rmods[2].dims == [0]
    @test Rmods[3].dims == [0]

end

@testset "Derived pushforward maps (morphism action)" begin
    Qtri = triangle_boundary_poset()
    Ppt = chain_poset(1)
    # Collapse Qtri -> *.
    # EncodingMap(Q, P, pi_of_q) stores the domain poset first.
    pi = EN.EncodingMap(Qtri, Ppt, fill(1, Qtri.n))

    # Constant 1D module on Qtri.
    dims = ones(Int, Qtri.n)
    edge_maps = Dict{Tuple{Int,Int}, SparseMatrixCSC{QQ,Int}}()
    for (a,b) in FF.cover_edges(Qtri)
        edge_maps[(a,b)] = sparse(fill(QQ(1), 1, 1))
    end
    M = MD.PModule{QQ}(Qtri, dims, edge_maps)

    # Scalar endomorphisms of M.
    function scalar_endomorphism(M::MD.PModule{QQ}, a::QQ)
        comps = [fill(a, 1, 1) for _ in 1:M.Q.n]
        return MD.PMorphism{QQ}(M, M, comps)
    end

    f2 = scalar_endomorphism(M, QQ(2))
    f3 = scalar_endomorphism(M, QQ(3))

    # Compose fiberwise.
    function compose_morphism(g::MD.PMorphism{QQ}, f::MD.PMorphism{QQ})
        @assert f.cod === g.dom
        n = f.dom.Q.n
        comps = [g.comps[u] * f.comps[u] for u in 1:n]
        return MD.PMorphism{QQ}(f.dom, g.cod, comps)
    end

    f6 = compose_morphism(f3, f2)

    # Higher derived functors are nontrivial (S^1):
    Lmods = PM.Lpushforward_left(pi, M, CM.DerivedFunctorOptions(maxdeg=1))
    Rmods = PM.Rpushforward_right(pi, M, CM.DerivedFunctorOptions(maxdeg=1))

    @test FZ.dim_at(Lmods[1], 1) == 1   # L_0
    @test FZ.dim_at(Lmods[2], 1) == 1   # L_1
    @test FZ.dim_at(Rmods[1], 1) == 1   # R^0
    @test FZ.dim_at(Rmods[2], 1) == 1   # R^1

    # Induced derived maps: should act by scalar multiplication in all degrees.
    Lf2 = PM.Lpushforward_left(pi, f2, CM.DerivedFunctorOptions(maxdeg=1))
    Lf3 = PM.Lpushforward_left(pi, f3, CM.DerivedFunctorOptions(maxdeg=1))
    Lf6 = PM.Lpushforward_left(pi, f6, CM.DerivedFunctorOptions(maxdeg=1))

    @test Lf2[1].comps[1] == fill(QQ(2), 1, 1)
    @test Lf2[2].comps[1] == fill(QQ(2), 1, 1)

    # Functoriality in each degree: L(f3 o f2) = L(f3) o L(f2)
    @test Lf6[1].comps[1] == Lf3[1].comps[1] * Lf2[1].comps[1]
    @test Lf6[2].comps[1] == Lf3[2].comps[1] * Lf2[2].comps[1]

    # Right-derived maps.
    Rf2 = PM.Rpushforward_right(pi, f2, CM.DerivedFunctorOptions(maxdeg=1))
    Rf3 = PM.Rpushforward_right(pi, f3, CM.DerivedFunctorOptions(maxdeg=1))
    Rf6 = PM.Rpushforward_right(pi, f6, CM.DerivedFunctorOptions(maxdeg=1))

    @test Rf2[1].comps[1] == fill(QQ(2), 1, 1)
    @test Rf2[2].comps[1] == fill(QQ(2), 1, 1)

    # Functoriality: R(f3 o f2) = R(f3) o R(f2)
    @test Rf6[1].comps[1] == Rf3[1].comps[1] * Rf2[1].comps[1]
    @test Rf6[2].comps[1] == Rf3[2].comps[1] * Rf2[2].comps[1]
end

@testset "Sparse naturality systems (no dense QQ matrices)" begin
    @testset "ExactQQ.nullspaceQQ works on SparseMatrixCSC{QQ}" begin
        # A is 3x4 with a small nullspace.  Verify:
        # - returned basis vectors are in the nullspace
        # - dimension matches dense nullspace dimension
        A = sparse([1, 1, 2, 3],
                   [1, 3, 2, 4],
                   QQ[1, 2, -1, 3],
                   3, 4)

        Ns = EX.nullspaceQQ(A)
        Nd = EX.nullspaceQQ(Matrix(A))

        @test size(Ns, 1) == 4
        @test size(Ns, 2) == size(Nd, 2)
        @test A * Ns == zeros(QQ, size(A, 1), size(Ns, 2))

        # columns should be independent
        @test EX.rankQQ(Ns) == size(Ns, 2)
    end

    @testset "ChainComplexes.solve_particularQQ works on SparseMatrixCSC{QQ}" begin
        # A is 3x3, consistent system with 2 RHS columns.
        A = sparse([1, 2, 2, 3],
                   [1, 1, 3, 2],
                   QQ[1, 1, 1, 1],
                   3, 3)

        B = QQ[1 0;
               0 1;
               1 1]

        Xs = CC.solve_particularQQ(A, B)
        @test Xs !== nothing
        @test A * Xs == B

        # Dense version should also solve.
        Xd = CC.solve_particularQQ(Matrix(A), B)
        @test Xd !== nothing
        @test A * Xd == B

        # Inconsistent system: 0*x = 1.
        A0 = sparse(Int[], Int[], QQ[], 1, 1)
        B0 = reshape(QQ[1], 1, 1)
        @test CC.solve_particularQQ(A0, B0) === nothing
    end

    @testset "DerivedFunctors.Hom produces correct basis (sanity cases)" begin
        Q = chain_poset(2)

        # Case 1: identity edge maps -> Hom is 1-dimensional.
        edge_id = Dict((1, 2) => Matrix{QQ}(I, 1, 1))
        M = MD.PModule{QQ}(Q, [1, 1], edge_id)
        N = MD.PModule{QQ}(Q, [1, 1], edge_id)

        H = DF.Hom(M, N)
        @test length(H.basis) == 1
        for f in H.basis
            for (u, v) in FF.cover_edges(Q)
                @test N.edge_maps[u, v] * f.comps[u] == f.comps[v] * M.edge_maps[u, v]
            end
        end

        # Case 2: zero edge maps -> no coupling between vertices -> dimension 2.
        edge_zero = Dict((1, 2) => zeros(QQ, 1, 1))
        M0 = MD.PModule{QQ}(Q, [1, 1], edge_zero)
        N0 = MD.PModule{QQ}(Q, [1, 1], edge_zero)

        H0 = DF.Hom(M0, N0)
        @test length(H0.basis) == 2
        for f in H0.basis
            for (u, v) in FF.cover_edges(Q)
                @test N0.edge_maps[u, v] * f.comps[u] == f.comps[v] * M0.edge_maps[u, v]
            end
        end
    end
end

@testset "ExactQQ restricted rank (sparse submatrix)" begin
    rng = MersenneTwister(123456)

    m, n = 30, 40
    nnz_target = 180

    I = rand(rng, 1:m, nnz_target)
    J = rand(rng, 1:n, nnz_target)
    V = [QQ(rand(rng, -3:3)) for _ in 1:nnz_target]
    A = sparse(I, J, V, m, n)
    dropzeros!(A)

    # Random restricted slices: rankQQ_restricted must match explicit slicing.
    for _ in 1:25
        rows = sort!(unique(rand(rng, 1:m, rand(rng, 1:15))))
        cols = sort!(unique(rand(rng, 1:n, rand(rng, 1:18))))
        r1 = EX.rankQQ_restricted(A, rows, cols)
        r2 = EX.rankQQ(A[rows, cols])
        @test r1 == r2
    end

    # Edge cases
    @test EX.rankQQ_restricted(A, Int[], collect(1:n)) == 0
    @test EX.rankQQ_restricted(A, collect(1:m), Int[]) == 0
    @test EX.rankQQ_restricted(A, 1:m, 1:n) == EX.rankQQ(A)
end

@testset "Common refinement encoding for different posets" begin
    # A small helper for a constant 1-dimensional module on a finite poset:
    # dims[v] = 1 for all v, and every cover-edge map is the 1x1 identity.
    function constant_1_module(P::FF.FinitePoset)
        dims = fill(1, P.n)
        edge_maps = Dict{Tuple{Int,Int}, Matrix{QQ}}()
        C = FF.cover_edges(P)
        sizehint!(edge_maps, length(C))
        for (u, v) in C
            edge_maps[(u, v)] = Matrix{QQ}(I, 1, 1)
        end
        return MD.PModule{QQ}(P, dims, edge_maps)
    end

    P1 = chain_poset(2)
    P2 = chain_poset(3)

    M1 = constant_1_module(P1)
    M2 = constant_1_module(P2)

    # The new API: build a common refinement automatically.
    P, Ms, pi1, pi2 = PM.encode_pmodules_to_common_poset(M1, M2)

    @test P.n == P1.n * P2.n
    @test Ms[1].Q === P
    @test Ms[2].Q === P
    @test pi1.Q === P && pi1.P === P1
    @test pi2.Q === P && pi2.P === P2

    # Sanity: product cover-edge count should be |E1|*n2 + n1*|E2|
    C1 = FF.cover_edges(P1)
    C2 = FF.cover_edges(P2)
    C  = FF.cover_edges(P)  # should be fast because product_poset pre-caches it
    @test length(C) == length(C1) * P2.n + P1.n * length(C2)

    # Hom between constant rank-1 modules should be 1-dimensional.
    H = PM.Hom(Ms[1], Ms[2])
    @test PM.dim(H) == 1

    # Compare against the generic restriction-based pullback (correctness check).
    # This also ensures restriction(pi, M) works concretely (not just defined).
    M1r = PM.restriction(pi1, M1)
    M2r = PM.restriction(pi2, M2)

    @test M1r.dims == Ms[1].dims
    @test M2r.dims == Ms[2].dims

    for e in FF.cover_edges(P)
        u, v = e
        @test Ms[1].edge_maps[u, v] == M1r.edge_maps[u, v]
        @test Ms[2].edge_maps[u, v] == M2r.edge_maps[u, v]
    end

    # A nontrivial example where commutativity forces Hom = 0.
    # S1: simple at vertex 1 on P1=chain(2).
    # S2: simple at vertex 2 on P2=chain(3).
    # On the product, vertical identity maps in S1 and zero maps in S2
    # force any would-be morphism to vanish.
    S1 = IR.pmodule_from_fringe(one_by_one_fringe(P1, FF.principal_upset(P1, 1), FF.principal_downset(P1, 1)))
    S2 = IR.pmodule_from_fringe(one_by_one_fringe(P2, FF.principal_upset(P2, 2), FF.principal_downset(P2, 2)))

    _, Ms2, _, _ = PM.encode_pmodules_to_common_poset(S1, S2)
    H2 = PM.Hom(Ms2[1], Ms2[2])
    @test PM.dim(H2) == 0
end

@testset "Identical leq matrices avoid product blowup" begin
    function constant_1_module(P::FF.FinitePoset)
        dims = fill(1, P.n)
        edge_maps = Dict{Tuple{Int,Int}, Matrix{QQ}}()
        C = FF.cover_edges(P)
        sizehint!(edge_maps, length(C))
        for (u, v) in C
            edge_maps[(u, v)] = Matrix{QQ}(I, 1, 1)
        end
        return MD.PModule{QQ}(P, dims, edge_maps)
    end

    P  = chain_poset(3)
    Pc = FF.FinitePoset(BitMatrix(P.leq); check = false)  # structurally identical, different object

    M  = constant_1_module(P)
    Mc = constant_1_module(Pc)

    Pout, Ms, pi1, pi2 = PM.encode_pmodules_to_common_poset(M, Mc)

    # Should NOT return a 9-vertex product. It should rebase Mc onto P.
    @test Pout === P
    @test Ms[1].Q === P
    @test Ms[2].Q === P

    H = PM.Hom(Ms[1], Ms[2])
    @test PM.dim(H) == 1
end

@testset "product_poset caching" begin
    P1 = chain_poset(2)
    P2 = chain_poset(3)

    prod1 = PM.product_poset(P1, P2)
    prod2 = PM.product_poset(P1, P2)

    # With default settings, this should hit the cache.
    @test prod1.P === prod2.P
end

@testset "Sparse triplet assembly helper" begin
    # Dense block, offset placement
    A = QQ[0 2;
           3 0]
    I = Int[]; J = Int[]; V = QQ[]
    CM._append_scaled_triplets!(I, J, V, A, 0, 0)
    S = sparse(I, J, V, 2, 2)
    @test S == sparse(A)

    # Scaled block into larger matrix
    I = Int[]; J = Int[]; V = QQ[]
    CM._append_scaled_triplets!(I, J, V, A, 1, 2; scale = -QQ(2))
    S = sparse(I, J, V, 3, 4)
    R = spzeros(QQ, 3, 4)
    R[2:3, 3:4] = -QQ(2) * A
    @test S == R

    # Indexed (noncontiguous) placement
    rows = [2, 4]
    cols = [1, 3]
    B = QQ[1 0;
           0 2]
    I = Int[]; J = Int[]; V = QQ[]
    CM._append_scaled_triplets!(I, J, V, B, rows, cols)
    S = sparse(I, J, V, 5, 5)
    R = spzeros(QQ, 5, 5)
    R[rows, cols] = B
    @test S == R
end

@testset "Hom differential assembly matches dense reference" begin
    # poset 1 <= 2
    leq = Bool[true true;
               false true]
    P = FF.FinitePoset(leq; check=true)

    dims = [1, 1]
    edge_maps = Dict((1,2) => QQ[1;;])  # 1x1 matrix
    N = MD.PModule(P, dims, edge_maps)

    # dummy resolution data for _build_hom_differential
    gens = Vector{Vector{Int}}(undef, 2)
    gens[1] = [1]      # degree 0 gens
    gens[2] = [1, 2]   # degree 1 gens

    # Coefficient matrix of d_1 : P_1 -> P_0 (rows = cod summands, cols = dom summands)
    # delta is stored with rows = codomain summands (P_0) and cols = domain summands (P_1).
    # Here P_0 has 1 summand (base vertex 1) and P_1 has 2 summands (base vertices 1,2).
    delta = sparse([1,1], [1,2], [QQ(1), QQ(1)], 1, 2)

    # Minimal resolution object: only `gens` and `d_mat` are used by _build_hom_differential,
    # but the struct requires the full field list.
    Pmods = [N, N]  # placeholders (not used in this test)

    # dummy morphisms (not used by _build_hom_differential; only needed to build the struct)
    dummy_pm = MD.id_morphism(N)

    # Canonical ProjectiveResolution API: (M, Pmods, gens, d_mor, d_mat, aug)
    res = DF.ProjectiveResolution(N, Pmods, gens, [dummy_pm], [delta], dummy_pm)

    offs_cod = [0, N.dims[1]]                       # [0,1]
    offs_dom = [0, N.dims[1], N.dims[1]+N.dims[2]]  # [0,1,2]

    Dnew = DF._build_hom_differential(res, N, 1, offs_cod, offs_dom)
    @test issparse(Dnew)

    # dense reference:
    Dref = zeros(QQ, 2, 1)
    Dref[1,1] = 1   # map_leq(N,1,1)
    Dref[2,1] = 1   # map_leq(N,1,2)
    @test Matrix(Dnew) == Dref
end

@testset "CoverEdgeMapStore equality and rebasing" begin
    P = chain_poset(3)
    dims = [1,1,1]
    edge_maps = Dict{Tuple{Int,Int}, Matrix{QQ}}()
    for (u,v) in IR.cover_edges(P)
        edge_maps[(u,v)] = QQ[1;;]
    end

    M1 = MD.PModule{QQ}(P, dims, edge_maps)
    M2 = MD.PModule{QQ}(P, dims, edge_maps)

    # Structural equality of stores (not pointer equality)
    @test M1.edge_maps == M2.edge_maps

    # Rebase onto an "equivalent" poset object (same leq matrix, new instance)
    P2 = FF.FinitePoset(copy(P.leq))
    M3 = MD.PModule{QQ}(P2, M1.dims, M1.edge_maps)  # should not error
    @test M3.dims == M1.dims
    @test M3.edge_maps == M1.edge_maps
end

@testset "Store-aligned cover-edge iteration" begin
    # Small non-chain poset with branching.
    Q = diamond_poset()

    # Deterministic module on Q.
    dims = [2, 1, 1, 2]

    # Cover maps: (1,2), (1,3), (2,4), (3,4).
    edge = Dict{Tuple{Int,Int}, Matrix{QQ}}()
    edge[(1,2)] = QQ[1 0]          # 1x2
    edge[(1,3)] = QQ[0 1]          # 1x2
    # NOTE: use 2x1 matrices (not length-2 vectors) for cover-edge maps.
    edge[(2,4)] = QQ[1; 0;;]       # 2x1
    edge[(3,4)] = QQ[0; 1;;]       # 2x1

    M = MD.PModule{QQ}(Q, dims, edge)

    store = M.edge_maps
    succs = store.succs
    preds = store.preds

    # (1) Store-aligned traversal agrees with keyed lookup.
    @testset "maps_to_succ agrees with getindex" begin
        for u in 1:Q.n
            su = succs[u]
            Mu = store.maps_to_succ[u]
            for j in eachindex(su)
                v = su[j]
                A1 = Mu[j]
                A2 = M.edge_maps[u, v]
                A3 = M.edge_maps[u, v]
                @test A1 == A2
                @test A1 == A3
            end
        end
    end

    # (2) maps_to_succ and maps_from_pred are consistent.
    @testset "maps_from_pred consistency" begin
        for u in 1:Q.n
            su = succs[u]
            Mu = store.maps_to_succ[u]
            for j in eachindex(su)
                v = su[j]
                ip = MD._find_sorted_index(preds[v], u)
                @test store.maps_from_pred[v][ip] == Mu[j]
            end
        end
    end

    # (3) Kernel/image routines still commute on edges after the store-direct rewrite.
    @testset "kernel/image commuting diagrams" begin
        # Simple chain so we can reason about results.
        Qc = chain_poset(3)

        dimsC = [1, 1, 1]
        edgeC = Dict{Tuple{Int,Int}, Matrix{QQ}}()
        edgeC[(1,2)] = QQ[1;;]
        edgeC[(2,3)] = QQ[1;;]
        Mc = MD.PModule{QQ}(Qc, dimsC, edgeC)

        # Zero map Mc -> Mc.
        fcomps = [zeros(QQ, 1, 1) for _ in 1:3]
        f = MD.PMorphism{QQ}(Mc, Mc, fcomps)

        K, iotaK = PM.kernel_with_inclusion(f)
        Im, iotaIm = PM.image_with_inclusion(f)

        # Kernel of zero map should be the whole module.
        @test K.dims == Mc.dims

        # Image of zero map should be the zero module.
        @test all(Im.dims .== 0)

        # Commutativity on cover edges:
        # Mc(u->v) * iotaK[u] == iotaK[v] * K(u->v)
        for (u, v) in FF.cover_edges(Qc)
            @test Mc.edge_maps[u, v] * iotaK.comps[u] == iotaK.comps[v] * K.edge_maps[u, v]
            @test Mc.edge_maps[u, v] * iotaIm.comps[u] == iotaIm.comps[v] * Im.edge_maps[u, v]
        end
    end
end