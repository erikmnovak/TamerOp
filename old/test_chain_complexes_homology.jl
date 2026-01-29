using Test
using SparseArrays
using LinearAlgebra
import Base.Threads

const PM = PosetModules
const FF = PM.FiniteFringe
const IR = PM.IndicatorResolutions
const DF = PM.DerivedFunctors
const QQ = PM.QQ

# helper: simple 1 times 1 fringe module on interval [a,b]
function interval_module(P::FF.FinitePoset, a::Int, b::Int)
    if !FF.leq(P, a, b)
        error("interval_module expects a <= b in the given poset; got a=$a, b=$b")
    end
    U = FF.principal_upset(P, a)
    D = FF.principal_downset(P, b)
    return FF.one_by_one_fringe(P, U, D, QQ(1))
end

# A tiny helper: one-vertex poset module = just a vector space.
function one_vertex_module(dim::Int)
    P = chain_poset(1)
    return IR.PModule{QQ}(P, [dim], Dict{Tuple{Int,Int}, Matrix{QQ}}())
end

function scalar_morphism(M, a::Int)
    A = fill(QQ(a), M.dims[1], M.dims[1])
    return IR.PMorphism{QQ}(M, M, [A])
end

@testset "Homological algebra edge cases on finite posets" begin
    # One-point poset: incidence algebra is just the base field.
    P1 = chain_poset(1)
    S = IR.pmodule_from_fringe(one_by_one_fringe(P1, FF.principal_upset(P1, 1), FF.principal_downset(P1, 1)))

    E = PM.Ext(S, S; maxdeg=3)
    @test PM.dim(E, 0) == 1
    @test all(PM.dim(E, t) == 0 for t in 1:3)

    T = PM.Tor(S, S; maxdeg=3)
    @test PM.dim(T, 0) == 1
    @test all(PM.dim(T, t) == 0 for t in 1:3)

    # Zero module: Ext and Tor should vanish in all degrees.
    Z = IR.PModule{QQ}(P1, [0], Dict{Tuple{Int,Int}, Matrix{QQ}}())

    EZS = PM.Ext(Z, S; maxdeg=2)
    ESZ = PM.Ext(S, Z; maxdeg=2)
    EZZ = PM.Ext(Z, Z; maxdeg=2)
    @test all(PM.dim(EZS, t) == 0 for t in 0:2)
    @test all(PM.dim(ESZ, t) == 0 for t in 0:2)
    @test all(PM.dim(EZZ, t) == 0 for t in 0:2)

    TZS = PM.Tor(Z, S; maxdeg=2)
    TSZ = PM.Tor(S, Z; maxdeg=2)
    TZZ = PM.Tor(Z, Z; maxdeg=2)
    @test all(PM.dim(TZS, t) == 0 for t in 0:2)
    @test all(PM.dim(TSZ, t) == 0 for t in 0:2)
    @test all(PM.dim(TZZ, t) == 0 for t in 0:2)

    # Disconnected poset: Ext between components should vanish.
    P = disjoint_two_chains_poset(2, 2)  # vertices {1,2} and {3,4} are separate components
    S1 = IR.pmodule_from_fringe(one_by_one_fringe(P, FF.principal_upset(P, 1), FF.principal_downset(P, 1)))
    S3 = IR.pmodule_from_fringe(one_by_one_fringe(P, FF.principal_upset(P, 3), FF.principal_downset(P, 3)))
    E13 = PM.Ext(S1, S3; maxdeg=2)
    @test all(PM.dim(E13, t) == 0 for t in 0:2)

    # Cross-check: Ext computed via projectives vs via injectives agree on dimensions.
    Pd = diamond_poset()
    A = IR.pmodule_from_fringe(one_by_one_fringe(Pd, FF.principal_upset(Pd, 1), FF.principal_downset(Pd, 1)))
    B = IR.pmodule_from_fringe(one_by_one_fringe(Pd, FF.principal_upset(Pd, 4), FF.principal_downset(Pd, 4)))

    Eproj = PM.Ext(A, B; maxdeg=2)
    resBinj = PM.injective_resolution(B; maxlen=2)
    Einj = PM.ExtInjective(A, resBinj)

    @test [PM.dim(Eproj, t) for t in 0:2] == [PM.dim(Einj, t) for t in 0:2]
end

@testset "ChainComplexes homology_data and homology_coordinates by hand" begin
    # ----------------
    # Circle: C1=QQ, C0=QQ, d1=0
    # ----------------
    d1 = zeros(QQ, 1, 1)      # C1 -> C0
    d0 = zeros(QQ, 0, 1)      # C0 -> 0
    d2 = zeros(QQ, 1, 0)      # 0  -> C1

    H1 = CC.homology_data(d2, d1, 1)
    @test H1.dimH == 1

    H0 = CC.homology_data(d1, d0, 0)
    @test H0.dimH == 1

    # coordinate sanity on the chosen basis representative
    c = CC.homology_coordinates(H1, H1.Hrep[:, 1])
    @test c == reshape(QQ[1], 1, 1)

    # ----------------
    # Interval: C1=QQ, C0=QQ^2, d1 = [ -1; 1 ]
    # ----------------
    d1_int = reshape(QQ[-1, 1], 2, 1)
    d0_int = zeros(QQ, 0, 2)
    d2_int = zeros(QQ, 1, 0)

    H1_int = CC.homology_data(d2_int, d1_int, 1)
    @test H1_int.dimH == 0

    H0_int = CC.homology_data(d1_int, d0_int, 0)
    @test H0_int.dimH == 1

    # any nonzero vector in C1 is not a cycle since d1_int is injective
    @test_throws ErrorException CC.homology_coordinates(H1_int, reshape(QQ[1], 1, 1))

    # boundary class should map to 0 in H0
    b = H0_int.B[:, 1]
    c0 = CC.homology_coordinates(H0_int, b)
    @test c0 == zeros(QQ, H0_int.dimH, 1)

    # ----------------
    # Filled triangle: C2=QQ, C1=QQ^3, C0=QQ^3
    # edge basis: e01,e02,e12; vertex basis: v0,v1,v2; face basis: f012
    # d2(f) = e12 - e02 + e01 => column [1,-1,1]
    # d1 columns are boundary of edges:
    #   e01 -> v1 - v0  => [-1, 1, 0]
    #   e02 -> v2 - v0  => [-1, 0, 1]
    #   e12 -> v2 - v1  => [ 0,-1, 1]
    # ----------------
    d2_tri = reshape(QQ[1, -1, 1], 3, 1)
    d1_tri = QQ[
        -1  -1   0;
         1   0  -1;
         0   1   1
    ]
    d0_tri = zeros(QQ, 0, 3)

    H2_tri = CC.homology_data(zeros(QQ, 1, 0), d2_tri, 2)
    @test H2_tri.dimH == 0

    H1_tri = CC.homology_data(d2_tri, d1_tri, 1)
    @test H1_tri.dimH == 0

    H0_tri = CC.homology_data(d1_tri, d0_tri, 0)
    @test H0_tri.dimH == 1
end

@testset "ExactQQ linear algebra" begin
    # Rank test
    A = QQ[QQ(1) QQ(2);
           QQ(2) QQ(4)]
    @test EX.rankQQ(A) == 1

    # Nullspace test: A * v = 0
    N = EX.nullspaceQQ(A)
    @test size(N, 1) == 2
    @test size(N, 2) == 1
    v = N[:, 1]
    @test A * v == zeros(QQ, 2)

    # Solve full column rank system B*x = y
    B = QQ[QQ(1) QQ(0);
           QQ(0) QQ(1);
           QQ(1) QQ(1)]
    x_true = QQ[QQ(1), QQ(2)]
    y = B * x_true
    x = EX.solve_fullcolumnQQ(B, y)
    @test B * x == y

    # Multiple right-hand sides
    Y = hcat(y, QQ(2) .* y)
    X = EX.solve_fullcolumnQQ(B, Y)
    @test B * X == Y


       @testset "rrefQQ / colspaceQQ / solve_fullcolumnQQ edge cases" begin
              # A has rank 2: second row is 2*first, third row breaks full dependence.
              A = QQ[1 2 3;
                     2 4 6;
                     1 1 1]
              R, piv = EX.rrefQQ(A)

              @test piv == [1, 2]
              @test R == QQ[1 0 -1;
                            0 1  2;
                            0 0  0]

              # colspaceQQ should return exactly the pivot columns of A.
              C = EX.colspaceQQ(A)
              @test size(C) == (3, 2)
              @test C == A[:, piv]

              # solve_fullcolumnQQ should reject matrices that are not full column rank.
              B = QQ[1 2;
                     2 4;
                     3 6]
              b = QQ[1, 2, 3]
              @test_throws ErrorException EX.solve_fullcolumnQQ(B, b)
       end
end

@testset "Sparse colspaceQQ agrees with dense" begin
    A = sparse(QQ[1 0 2;
                  0 1 3;
                  0 0 0])
    Cd = EX.colspaceQQ(Matrix{QQ}(A))
    Cs = EX.colspaceQQ(A)

    @test EX.rankQQ(Cd) == EX.rankQQ(Cs)
    @test EX.rankQQ(Cs) == EX.rankQQ(A)

    # columns of Cs should lie in col(A): solve_fullcolumn on basis test
    # Check that each column of Cs is in colspace by solving using dense basis
    B = EX.colspaceQQ(A)
    for j in 1:size(Cs,2)
        x = EX.solve_fullcolumnQQ(B, Cs[:,j])
        @test B*x == Cs[:,j]
    end
end

@testset "FullColumnFactor and cache correctness" begin
    B = QQ[1 0;
           0 1;
           1 1]  # full column rank 3x2
    y = B * QQ[2, 3]

    EX.clear_fullcolumn_cache!()
    x1 = EX.solve_fullcolumnQQ(B, y; cache=true)
    @test B*x1 == y

    # cache should now contain B
    @test haskey(EX._FULLCOLUMN_FACTOR_CACHE, B)

    # repeated solve should still work
    x2 = EX.solve_fullcolumnQQ(B, y; cache=true)
    @test x2 == x1

    # RHS not in column space should throw
    bady = QQ[1,0,0]
    @test_throws ErrorException EX.solve_fullcolumnQQ(B, bady; cache=true)
end

@testset "rankQQ_dim matches exact rank on small matrices" begin
    A = QQ[1 2 3;
           2 4 6;
           1 0 1]
    @test EX.rankQQ(A) == 2
    @test EX.rankQQ_dim(A; backend=:auto) == 2
    @test EX.rankQQ_dim(A; backend=:modular) == 2
end


@testset "ChainComplexes: shift and extend_range" begin
    # Cochain complex C: degrees 0..1, zero differential, dims 1 in each degree.
    d0 = spzeros(QQ, 1, 1)
    C = CC.CochainComplex{QQ}(0, 1, [1, 1], [d0])

    @test CC.cohomology_data(C, 0).dimH == 1
    @test CC.cohomology_data(C, 1).dimH == 1

    # Shift by +2: (C[2])^t = C^{t+2}
    Cs = CC.shift(C, 2)
    @test Cs.tmin == -2
    @test Cs.tmax == -1
    @test CC.cohomology_data(Cs, -2).dimH == 1
    @test CC.cohomology_data(Cs, -1).dimH == 1

    # Extend range should not change cohomology in the original degrees.
    Ce = CC.extend_range(C, -3, 4)
    @test CC.cohomology_data(Ce, 0).dimH == 1
    @test CC.cohomology_data(Ce, 1).dimH == 1
end

@testset "ChainComplexes: induced_map_on_cohomology for zero-differential complexes" begin
    # Complexes concentrated in degree 0.
    C = CC.CochainComplex{QQ}(0, 0, [2], SparseMatrixCSC{QQ,Int}[])
    D = CC.CochainComplex{QQ}(0, 0, [3], SparseMatrixCSC{QQ,Int}[])

    HC = CC.cohomology_data(C, 0)
    HD = CC.cohomology_data(D, 0)

    # A concrete linear map f: C^0 -> D^0
    f = sparse([1,2], [1,2], [QQ(1), QQ(1)], 3, 2)
    fH = CC.induced_map_on_cohomology(HC, HD, f)

    # Since differentials are zero, cohomology equals the underlying vector space.
    @test Matrix(fH) == Matrix(f)

    z = spzeros(QQ, 3, 2)
    zH = CC.induced_map_on_cohomology(HC, HD, z)
    
    @test count(!iszero, zH) == 0
end

@testset "ModuleCochainComplex accepts FringeModule inputs (auto-convert via pmodule_from_fringe)" begin
    # Use a chain poset so the interval-style up/down sets intersect in a predictable way.
    P = chain_poset(5)
    n = P.n
    a, b = 2, 4

    # These are the same style as the helper interval_module in this file.
    U = BitVector([i <= b for i in 1:n])
    D = BitVector([i >= a for i in 1:n])

    H = PM.one_by_one_fringe(P, U, D, QQ(1))
    M = IR.pmodule_from_fringe(H)

    C_from_fringe = PM.ModuleCochainComplex([H], PM.PMorphism{QQ}[]; tmin = 0)
    C_from_pmodule = PM.ModuleCochainComplex([M], PM.PMorphism{QQ}[]; tmin = 0)

    @test C_from_fringe.tmin == 0
    @test C_from_fringe.tmax == 0
    @test length(C_from_fringe.terms) == 1
    @test C_from_fringe.terms[1] isa IR.PModule{QQ}

    # The internal converted term should match explicit conversion.
    @test C_from_fringe.terms[1].dims == C_from_pmodule.terms[1].dims
    @test C_from_fringe.terms[1].edge_maps == C_from_pmodule.terms[1].edge_maps
end

@testset "Module complexes / hyperExt / hyperTor" begin

    # --------------------------
    # 1) hyperExt agrees with Ext for degree-0 complex
    # --------------------------

    # Build the 3-element chain poset: 1 <= 2 <= 3.
    P = FF.FinitePoset([i <= j for i in 1:3, j in 1:3])
    M = interval_module(P,1,2)
    N = interval_module(P,2,3)

    C0 = PM.ModuleCochainComplex([M], PM.PMorphism{QQ}[]; tmin=0)
    maxdeg = 2
    E = DF.Ext(M,N; maxdeg=maxdeg, model=:injective)
    H = PM.hyperExt(C0,N; maxlen=maxdeg)

    for t in 0:maxdeg
        @test DF.dim(E,t) == PM.dim(H,t)
    end

    if Threads.nthreads() > 1
        H_serial = PM.hyperExt(C0, N; maxlen = maxdeg, threads = false)
        H_thread = PM.hyperExt(C0, N; maxlen = maxdeg, threads = true)
        @test dim(H_serial, 0) == dim(H_thread, 0)
        @test dim(H_serial, 1) == dim(H_thread, 1)
        @test dim(H_serial, 2) == dim(H_thread, 2)
    end

    # --------------------------
    # 2) mapping cone of id is acyclic in module cohomology
    # --------------------------
    idM = IR.id_morphism(M)
    f = PM.ModuleCochainMap(C0,C0,[idM]; tmin=0, tmax=0)
    Cone = PM.mapping_cone(f)

    for t in Cone.tmin:Cone.tmax
        Ht = PM.cohomology_module(Cone,t)
        @test all(d == 0 for d in Ht.dims)
    end

    # --------------------------
    # 3) RHom functoriality: (g circ f)^* = f^* circ g^*
    # --------------------------
    # scaling endomorphisms
    function scale(M::IR.PModule{QQ}, c::Int)
        comps = Matrix{QQ}[]
        for u in 1:M.Q.n
            d = M.dims[u]
            push!(comps, QQ(c)*Matrix{QQ}(I,d,d))
        end
        return IR.PMorphism{QQ}(M,M,comps)
    end

    fM = scale(M,2)
    gM = scale(M,3)
    gfM = scale(M,6)

    C1 = PM.ModuleCochainComplex([M], PM.PMorphism{QQ}[]; tmin=0)
    C2 = PM.ModuleCochainComplex([M], PM.PMorphism{QQ}[]; tmin=0)

    fmap = PM.ModuleCochainMap(C0,C1,[fM]; tmin=0, tmax=0)
    gmap = PM.ModuleCochainMap(C1,C2,[gM]; tmin=0, tmax=0)
    gfmap = PM.ModuleCochainMap(C0,C2,[gfM]; tmin=0, tmax=0)

    resN = DF.injective_resolution(N; maxlen=maxdeg)
    R0 = PM.RHomComplex(C0,N; maxlen=maxdeg, resN=resN)
    R1 = PM.RHomComplex(C1,N; maxlen=maxdeg, resN=resN)
    R2 = PM.RHomComplex(C2,N; maxlen=maxdeg, resN=resN)

    # Smoke test: RHomComplex must carry the Hom-space blocks used to
    # build the double complex, and they should be properly typed.
    @test R0.homs isa Array{DF.HomSpace{QQ},2}
    @test size(R0.homs, 2) == maxdeg + 1

    # induced maps on Tot are just precomposition matrices degreewise, so compare by multiplication
    F = DF._precompose_matrix(DF.Hom(M,resN.Emods[1]), DF.Hom(M,resN.Emods[1]), fM) # sanity call

    # We test cone-level functoriality indirectly: induced cohomology maps compose.
    hf = PM.induced_map_on_cohomology_modules(fmap,0)
    hg = PM.induced_map_on_cohomology_modules(gmap,0)
    hgf = PM.induced_map_on_cohomology_modules(gfmap,0)

    for u in 1:M.Q.n
        @test hgf.comps[u] == hg.comps[u]*hf.comps[u]
    end

    # --------------------------
    # 4) hyperTor agrees with Tor for degree-0 complex
    # --------------------------
    Pop = FF.FinitePoset(transpose(P.leq))

    # Pop is the opposite poset of P.
    # The interval [1,3] in P corresponds to the interval [3,1] in Pop.
    Rop = interval_module(Pop, 3, 1)

    Tplain = DF.Tor(Rop,M; maxdeg=maxdeg)
    HT = PM.hyperTor(Rop,C0; maxlen=maxdeg)

    for s in 0:maxdeg
        @test DF.dim(Tplain,s) == PM.dim(HT,s)
    end

    if Threads.nthreads() > 1
        T_serial = PM.hyperTor(Rop, C0; maxlen = maxdeg, threads = false)
        T_thread = PM.hyperTor(Rop, C0; maxlen = maxdeg, threads = true)
        @test dim(T_serial, 0) == dim(T_thread, 0)
        @test dim(T_serial, 1) == dim(T_thread, 1)
        @test dim(T_serial, 2) == dim(T_thread, 2)
    end

    # --------------------------
    # 5) RHom spectral sequence degenerates at E2 if horizontal differential=0
    # --------------------------
    # build a 2-term complex with zero differential
    Ctwo = PM.ModuleCochainComplex([M,M],[IR.zero_morphism(M,M)]; tmin=0)

    R = PM.RHomComplex(Ctwo,N; maxlen=maxdeg, resN=resN)
    ss = PM.spectral_sequence(R.DC; first=:vertical)
    E2 = PM.page(ss,2)

    # E2(A,B) should equal Ext^B(C^{-A},N) since d_h=0
    for A in (-1):0
        for B in 0:maxdeg
            Mp = (A==0) ? M : M
            Eab = DF.Ext(Mp,N; maxdeg=maxdeg, model=:injective)
            @test E2[(A,B)] == DF.dim(Eab,B)
        end
    end

end

@testset "ModuleCochainComplex check d^2=0" begin
    M = one_vertex_module(1)
    id = PM.id_morphism(M)
    z  = PM.zero_morphism(M, M)

    terms = [M, M, M]

    # Invalid: d1*d0 = id != 0
    @test_throws ErrorException PM.ModuleCochainComplex(terms, [id, id]; tmin=0, check=true)

    # Valid
    C = PM.ModuleCochainComplex(terms, [id, z]; tmin=0, check=true)
    @test C.tmin == 0
    @test C.tmax == 2

    # check=false should allow
    Cbad = PM.ModuleCochainComplex(terms, [id, id]; tmin=0, check=false)
    @test Cbad.tmax == 2
end

@testset "ModuleCochainComplex keyword tmin (positional endpoints removed)" begin
    P = chain_poset(1)
    M = IR.PModule{QQ}(P, [1], Dict{Tuple{Int,Int}, Matrix{QQ}}())
    id = scalar_morphism(M, 1)

    terms = [M, M]
    diffs = [id]

    C_kw = PM.ModuleCochainComplex(terms, diffs; tmin=-1, check=true)

    @test C_kw.tmin == -1
    @test C_kw.tmax == 0
    @test length(C_kw.terms) == 2
    @test length(C_kw.diffs) == 1

    # The old positional-endpoint signature is intentionally removed.
    @test_throws MethodError PM.ModuleCochainComplex(terms, diffs, -1, 0; check=true)

    # Sanity: diff-length mismatch should fail fast (constructor invariant).
    @test_throws AssertionError PM.ModuleCochainComplex(terms, PM.PMorphism{QQ}[]; tmin=-1, check=false)
end

@testset "ModuleCochainMap chain map validation" begin
    M = one_vertex_module(1)
    id = PM.id_morphism(M)
    z  = PM.zero_morphism(M, M)

    C = PM.ModuleCochainComplex([M, M], [id]; tmin=0, check=true)
    D = PM.ModuleCochainComplex([M, M], [id]; tmin=0, check=true)

    # Valid map: identity
    f = PM.ModuleCochainMap(C, D, [id, id]; tmin=0, tmax=1, check=true)
    @test f.tmin == 0

    # Invalid: breaks commutativity
    @test_throws ErrorException PM.ModuleCochainMap(C, D, [id, z]; tmin=0, tmax=1, check=true)

    # Boundary check: providing only degree 1 component implies degree 0 is zero -> fails
    @test_throws ErrorException PM.ModuleCochainMap(C, D, [id]; tmin=1, tmax=1, check=true)
end

@testset "ModuleCochainHomotopy exists and validates" begin
    M = one_vertex_module(1)
    id = PM.id_morphism(M)
    zM = PM.zero_morphism(M, M)
    Z  = PM.zero_pmodule(PM.FinitePosets.chain_poset(1), QQ)

    C = PM.ModuleCochainComplex([M, M], [id]; tmin=0, check=true)
    D = C

    f = PM.ModuleCochainMap(C, D, [id, id]; tmin=0, tmax=1, check=true)
    g = PM.ModuleCochainMap(C, D, [zM, zM]; tmin=0, tmax=1, check=true)

    h0 = PM.zero_morphism(M, Z)  # C^0 -> D^-1 = 0
    h1 = id                      # C^1 -> D^0

    H = PM.ModuleCochainHomotopy(f, g, [h0, h1]; tmin=0, tmax=1, check=true)
    @test PM.is_cochain_homotopy(H)

    # Wrong homotopy
    @test_throws ErrorException PM.ModuleCochainHomotopy(f, g, [h0, zM]; tmin=0, tmax=1, check=true)
end

@testset "mapping_cone(identity) is acyclic and id is quasi-iso" begin
    M = one_vertex_module(1)
    C = PM.ModuleCochainComplex([M], PM.PMorphism{QQ}[]; tmin=0, check=true)
    id = PM.ModuleCochainMap(C, C, [PM.id_morphism(M)]; tmin=0, tmax=0, check=true)

    cone = PM.mapping_cone(id)
    Hm1 = PM.cohomology_module(cone, -1)
    H0  = PM.cohomology_module(cone, 0)

    @test all(d == 0 for d in Hm1.dims)
    @test all(d == 0 for d in H0.dims)

    @test PM.is_quasi_isomorphism(id)
end

@testset "rhom_map_first strict functoriality under composition" begin
    M = one_vertex_module(2)
    N = one_vertex_module(2)

    C = PM.ModuleCochainComplex([M], PM.PMorphism{QQ}[]; tmin=0, check=true)

    A = scalar_morphism(M, 2)
    B = scalar_morphism(M, 3)
    BA = scalar_morphism(M, 6)

    f = PM.ModuleCochainMap(C, C, [A]; tmin=0, tmax=0, check=true)
    g = PM.ModuleCochainMap(C, C, [B]; tmin=0, tmax=0, check=true)
    gf = PM.ModuleCochainMap(C, C, [BA]; tmin=0, tmax=0, check=true)

    resN = PM.injective_resolution(N; maxlen=1)

    Fmap = PM.rhom_map_first(f, N; maxlen=1, resN=resN, check=true)
    Gmap = PM.rhom_map_first(g, N; maxlen=1, resN=resN, check=true)
    GFmap = PM.rhom_map_first(gf, N; maxlen=1, resN=resN, check=true)

    # Contravariance: (gcircf)^* = f^* circ g^*
    t0 = Fmap.tmin
    @test GFmap.maps[1] == (Fmap.maps[1] * Gmap.maps[1])
end

@testset "rhom_map_second and hyperExt_map_second functoriality" begin
    P = FF.chain_poset(2)

    # simples at vertices 1 and 2
    S1 = IR.pmodule_from_fringe(IR.one_by_one_fringe(P, FF.principal_upset(P, 1), FF.principal_downset(P, 1)))
    S2 = IR.pmodule_from_fringe(IR.one_by_one_fringe(P, FF.principal_upset(P, 2), FF.principal_downset(P, 2)))

    # C concentrated in degree 0
    C = PM.ModuleCochainComplex([S1], PM.PMorphism{QQ}[]; tmin=0, check=true)

    # N = S2 oplus S2 (so endomorphisms can be noncommuting 2x2 matrices)
    N, i1, i2, p1, p2 = PM.direct_sum(S2, S2)

    H = PM.hyperExt(C, N; maxlen=3)
    @test PM.dim(H, 1) == 2  # Ext^1(S1, S2^2) should be 2

    # helper: endomorphism defined only at vertex u
    function endo_at_vertex(M::IR.PModule{QQ}, u::Int, A::Matrix{QQ})
        comps = Vector{Matrix{QQ}}(undef, M.Q.n)
        for v in 1:M.Q.n
            dv = M.dims[v]
            comps[v] = Matrix{QQ}(I, dv, dv)
        end
        comps[u] = A
        return IR.PMorphism{QQ}(M, M, comps)
    end

    idN = IR.id_morphism(N)
    Mid = PM.hyperExt_map_second(idN, H, H; t=1)
    @test Mid == Matrix{QQ}(I, 2, 2)

    twoN = scalar_morphism(N, N, 2)
    Mtwo = PM.hyperExt_map_second(twoN, H, H; t=1)
    @test Mtwo == 2 * Matrix{QQ}(I, 2, 2)

    gC = endo_at_vertex(N, 2, QQ[0 1; 0 0])
    gD = endo_at_vertex(N, 2, QQ[0 0; 1 0])

    MC = PM.hyperExt_map_second(gC, H, H; t=1)
    MD = PM.hyperExt_map_second(gD, H, H; t=1)

    gDC = compose_morphism(gD, gC)  # gD circ gC
    MDC = PM.hyperExt_map_second(gDC, H, H; t=1)

    # Covariant functoriality: F(gD circ gC) == F(gD) * F(gC)
    @test MDC == MD * MC
end

@testset "hyperExt_map_first contravariant functoriality" begin
    P = FF.chain_poset(2)

    S1 = IR.pmodule_from_fringe(IR.one_by_one_fringe(P, FF.principal_upset(P, 1), FF.principal_downset(P, 1)))
    S2 = IR.pmodule_from_fringe(IR.one_by_one_fringe(P, FF.principal_upset(P, 2), FF.principal_downset(P, 2)))

    M, _, _, _, _ = PM.direct_sum(S1, S1)
    N, _, _, _, _ = PM.direct_sum(S2, S2)

    C = PM.ModuleCochainComplex([M], PM.PMorphism{QQ}[]; tmin=0, check=true)
    H = PM.hyperExt(C, N; maxlen=3)
    @test PM.dim(H, 1) == 4

    function endo_at_vertex(M::PModule{QQ}, u::Int, A::Matrix{QQ})
        comps = Vector{Matrix{QQ}}(undef, M.Q.n)
        for v in 1:M.Q.n
            dv = M.dims[v]
            comps[v] = Matrix{QQ}(I, dv, dv)
        end
        comps[u] = A
        return IR.PMorphism{QQ}(M, M, comps)
    end

    fA = endo_at_vertex(M, 1, QQ[0 1; 0 0])
    fB = endo_at_vertex(M, 1, QQ[0 0; 1 0])

    FA = PM.ModuleCochainMap(C, C, [fA])
    FB = PM.ModuleCochainMap(C, C, [fB])
    fBA = compose_morphism(fB, fA)
    FBA = PM.ModuleCochainMap(C, C, [fBA])

    MA = PM.hyperExt_map_first(FA, H, H; t=1)
    MB = PM.hyperExt_map_first(FB, H, H; t=1)
    MBA = PM.hyperExt_map_first(FBA, H, H; t=1)

    # Contravariant functoriality: F(fB circ fA) == F(fA) circ F(fB)
    @test MBA == MA * MB
end

# This test file focuses on the "abelian category" public API:
# kernels/cokernels/images/coimages/quotients, pushout/pullback,
# short exact sequences, and a basic snake lemma sanity check.

# Helper: a P-module on a 1-vertex poset (so no edge maps are needed).
function _vspace_module(P::FF.FinitePoset, d::Int)
    edge_maps = Dict{Tuple{Int,Int}, Matrix{QQ}}()
    return IR.PModule{QQ}(P, [d], edge_maps)
end

@testset "Abelian-category API" begin
    P = chain_poset(1)

    # -------------------------------------------------------------------------
    # Kernel / image / cokernel / coimage
    # -------------------------------------------------------------------------
    A = _vspace_module(P, 2)
    B = _vspace_module(P, 3)

    f_mat = Matrix{QQ}([
        QQ(1) QQ(0);
        QQ(0) QQ(0);
        QQ(0) QQ(0)
    ])
    f = IR.PMorphism{QQ}(A, B, [f_mat])

    K, iK = PM.kernel_with_inclusion(f)
    @test K.dims == [1]
    @test f.comps[1] * iK.comps[1] == zeros(QQ, 3, 1)
    @test PM.kernel(f).dims == K.dims

    Im, iIm = PM.image_with_inclusion(f)
    @test Im.dims == [1]
    @test PM.image(f).dims == [1]

    # Factorization test: since iIm is an inclusion, f should factor through it.
    X = EX.solve_fullcolumnQQ(iIm.comps[1], f.comps[1])
    @test iIm.comps[1] * X == f.comps[1]

    Cok, q = PM.cokernel_with_projection(f)
    @test Cok.dims == [2]
    @test q.comps[1] * f.comps[1] == zeros(QQ, 2, 2)
    @test PM.cokernel(f).dims == [2]

    Coim, pco = PM.coimage_with_projection(f)
    @test Coim.dims == [1]
    @test pco.comps[1] * iK.comps[1] == zeros(QQ, 1, 1)
    @test PM.coimage(f).dims == [1]

    # Quotient by the image submodule should match the cokernel.
    Simg = PM.image_submodule(f)
    Qmod = PM.quotient(Simg)
    @test Qmod.dims == Cok.dims

    # quotient(M, N) convenience: same result when ambient matches
    Qmod2 = PM.quotient(B, Simg)
    @test Qmod2.dims == Qmod.dims
    @test_throws ErrorException PM.quotient(_vspace_module(P, 999), Simg)

    # -------------------------------------------------------------------------
    # Pushout / pullback (1-vertex sanity checks)
    # -------------------------------------------------------------------------
    A1 = _vspace_module(P, 1)
    B1 = _vspace_module(P, 1)
    C1 = _vspace_module(P, 1)

    f_id = IR.PMorphism{QQ}(A1, B1, [Matrix{QQ}([QQ(1)])])
    g_id = IR.PMorphism{QQ}(A1, C1, [Matrix{QQ}([QQ(1)])])

    Pout, inB, inC, qpo, phi = PM.pushout(f_id, g_id)
    @test Pout.dims == [1]
    @test inB.comps[1] * f_id.comps[1] == inC.comps[1] * g_id.comps[1]

    # Pullback of identities should be the diagonal (dim 1).
    D1 = _vspace_module(P, 1)
    f_toD = IR.PMorphism{QQ}(B1, D1, [Matrix{QQ}([QQ(1)])])
    g_toD = IR.PMorphism{QQ}(C1, D1, [Matrix{QQ}([QQ(1)])])

    Pin, prB, prC, iota, psi = PM.pullback(f_toD, g_toD)
    @test Pin.dims == [1]
    @test f_toD.comps[1] * prB.comps[1] == g_toD.comps[1] * prC.comps[1]

    # -------------------------------------------------------------------------
    # Short exact sequences
    # -------------------------------------------------------------------------
    A2 = _vspace_module(P, 1)
    B2 = _vspace_module(P, 2)
    C2 = _vspace_module(P, 1)

    i_mat = Matrix{QQ}([QQ(1); QQ(0)])
    p_mat = Matrix{QQ}([QQ(0) QQ(1)])

    i = IR.PMorphism{QQ}(A2, B2, [i_mat])
    p = IR.PMorphism{QQ}(B2, C2, [p_mat])

    ses = PM.ShortExactSequence(i, p)
    @test PM.is_exact(ses)

    # Alias constructor
    ses_alias = PM.short_exact_sequence(i, p)
    @test PM.is_exact(ses_alias)

    # A non-exact variant: switch the projection.
    p_bad = IR.PMorphism{QQ}(B2, C2, [Matrix{QQ}([QQ(1) QQ(0)])])
    ses_bad = PM.ShortExactSequence(i, p_bad; check=false)
    @test !PM.is_exact(ses_bad)

    # -------------------------------------------------------------------------
    # Snake lemma (rank sanity check for the connecting morphism)
    # -------------------------------------------------------------------------
    # Top SES: 0 -> Q -> Q^2 -> Q -> 0
    At = _vspace_module(P, 1)
    Bt = _vspace_module(P, 2)
    Ct = _vspace_module(P, 1)
    it = IR.PMorphism{QQ}(At, Bt, [Matrix{QQ}([QQ(1); QQ(0)])])
    pt = IR.PMorphism{QQ}(Bt, Ct, [Matrix{QQ}([QQ(0) QQ(1)])])
    top = PM.ShortExactSequence(it, pt)

    # Bottom SES: 0 -> Q^2 -> Q^3 -> Q -> 0
    Ab = _vspace_module(P, 2)
    Bb = _vspace_module(P, 3)
    Cb = _vspace_module(P, 1)
    ib = IR.PMorphism{QQ}(Ab, Bb, [Matrix{QQ}([
        QQ(1) QQ(0);
        QQ(0) QQ(1);
        QQ(0) QQ(0)
    ])])
    pb = IR.PMorphism{QQ}(Bb, Cb, [Matrix{QQ}([QQ(0) QQ(0) QQ(1)])])
    bottom = PM.ShortExactSequence(ib, pb)

    # Vertical maps: alpha injective, beta injective, gamma = 0.
    alpha = IR.PMorphism{QQ}(At, Ab, [Matrix{QQ}([QQ(1); QQ(0)])])
    beta = IR.PMorphism{QQ}(Bt, Bb, [Matrix{QQ}([
        QQ(1) QQ(0);
        QQ(0) QQ(1);
        QQ(0) QQ(0)
    ])])
    gamma = IR.PMorphism{QQ}(Ct, Cb, [Matrix{QQ}([QQ(0)])])

    sn = PM.snake_lemma(top, bottom, alpha, beta, gamma)
    delta = sn.delta

    @test sn.kerC[1].dims == [1]   # ker(gamma) = Ct
    @test sn.cokA[1].dims == [1]   # coker(alpha) has dim 1
    @test EX.rankQQ(delta.comps[1]) == 1
    @test !PM.is_zero_morphism(delta)

    @testset "Products/coproducts/equalizers/coequalizers + diagram interface" begin
        # Use a 1-vertex poset to keep the matrices small and the universal
        # property checks completely explicit.
        P1 = chain_poset(1)
        Z = PM.zero_pmodule(P1, QQ)

        A = _vspace_module(P1, 2)
        B = _vspace_module(P1, 3)

        # --- biproduct sanity: p_i o i_j = delta_ij, and i1 p1 + i2 p2 = id
        S, iA, iB, pA, pB = PM.biproduct(A, B)

        @test pA.comps[1] * iA.comps[1] == Matrix{QQ}(I, 2, 2)
        @test pB.comps[1] * iB.comps[1] == Matrix{QQ}(I, 3, 3)
        @test pA.comps[1] * iB.comps[1] == zeros(QQ, 2, 3)
        @test pB.comps[1] * iA.comps[1] == zeros(QQ, 3, 2)

        @test iA.comps[1] * pA.comps[1] + iB.comps[1] * pB.comps[1] == Matrix{QQ}(I, 5, 5)

        # --- product/coproduct wrappers exist and return the expected maps
        Pprod, prA, prB = PM.product(A, B)
        Ccop, inA, inB = PM.coproduct(A, B)
        @test PM.is_morphism(prA)
        @test PM.is_morphism(prB)
        @test PM.is_morphism(inA)
        @test PM.is_morphism(inB)

        # Explicit universal property check for product:
        # given maps f:X->A and g:X->B, we can build (f,g): X -> A x B
        # and verify prA*(f,g)=f and prB*(f,g)=g.
        X = _vspace_module(P1, 2)
        f = PM.id_morphism(X)  # X -> A (both 2-dim, so treat as "identity")
        # A real map X->B: 3x2
        g = PM.PMorphism{QQ}(X, B, [QQ[1 0; 0 1; 0 0]])

        # (f,g) : X -> Pprod has block form [f; g]
        fg = PM.PMorphism{QQ}(X, Pprod, [vcat(f.comps[1], g.comps[1])])

        @test prA.comps[1] * fg.comps[1] == f.comps[1]
        @test prB.comps[1] * fg.comps[1] == g.comps[1]

        # --- equalizer/coequalizer checks
        # Build a nonzero map h : A -> B
        h = PM.PMorphism{QQ}(A, B, [QQ[1 0; 0 1; 0 0]])
        z = PM.zero_morphism(A, B)

        E, e = PM.equalizer(h, z)
        @test PM.is_morphism(e)
        # h o e == 0 o e
        he = compose_morphism(h, e)
        ze = compose_morphism(z, e)
        @test he.comps[1] == ze.comps[1]

        Q, q = PM.coequalizer(h, z)
        @test PM.is_morphism(q)
        # q o h == q o 0
        qh = compose_morphism(q, h)
        qz = compose_morphism(q, z)
        @test qh.comps[1] == qz.comps[1]

        # --- diagram object interface: limit/colimit dispatch
        Ddisc = PM.DiscretePairDiagram(A, B)
        _, dprA, dprB = PM.limit(Ddisc)
        @test PM.is_morphism(dprA)
        @test PM.is_morphism(dprB)

        Dpar = PM.ParallelPairDiagram(h, z)
        _, de = PM.limit(Dpar)
        @test PM.is_morphism(de)

        Dspan = PM.SpanDiagram(h, z)  # A -> B, A -> B (same codomain is fine for pushout)
        PO, p1, p2 = PM.colimit(Dspan)
        @test PM.is_morphism(p1)
        @test PM.is_morphism(p2)

        Dcosp = PM.CospanDiagram(h, z) # A -> B and A -> B; pullback exists
        PB, r1, r2 = PM.limit(Dcosp)
        @test PM.is_morphism(r1)
        @test PM.is_morphism(r2)
    end


end

# Local helper: check string is ASCII-only.
_is_ascii(s::AbstractString) = all(c -> Int(c) <= 0x7f, s)

@testset "Pretty printing: Submodule / ShortExactSequence / SnakeLemmaResult" begin
    P = chain_poset(1)

    # 1-vertex PModules (no edge maps needed).
    edge_maps = Dict{Tuple{Int,Int}, Matrix{QQ}}()

    A = IR.PModule{QQ}(P, [1], edge_maps)
    B = IR.PModule{QQ}(P, [2], edge_maps)
    C = IR.PModule{QQ}(P, [1], edge_maps)

    # i : A -> B (inclusion)
    i = IR.PMorphism{QQ}(A, B, [Matrix{QQ}([QQ(1); QQ(0)])])

    # p : B -> C (projection)
    p = IR.PMorphism{QQ}(B, C, [Matrix{QQ}([QQ(0) QQ(1)])])

    # --- PModule show ---
    sA0 = sprint(show, A)
    @test occursin("PModule", sA0)
    @test occursin("nverts=1", sA0)
    @test occursin("dims=", sA0)
    @test _is_ascii(sA0)

    sA1 = sprint(show, MIME("text/plain"), A)
    @test occursin("PModule", sA1)
    @test occursin("scalars = QQ", sA1)
    @test occursin("nverts = 1", sA1)
    @test occursin("dims =", sA1)
    @test _is_ascii(sA1)

    # --- PMorphism show ---
    si0 = sprint(show, i)
    @test occursin("PMorphism", si0)
    @test occursin("nverts=1", si0)
    @test occursin("dom_sum=1", si0)
    @test occursin("cod_sum=2", si0)
    @test _is_ascii(si0)

    si1 = sprint(show, MIME("text/plain"), i)
    @test occursin("PMorphism", si1)
    @test occursin("endomorphism = false", si1)
    @test occursin("dom dims", si1)
    @test occursin("cod dims", si1)
    @test _is_ascii(si1)

    # Truncation sanity check under IOContext(:limit=>true).
    Pbig = chain_poset(20)
    dims_big = collect(1:20)
    Mbig = IR.PModule{QQ}(Pbig, dims_big, Dict{Tuple{Int,Int}, Matrix{QQ}}())

    sbig = sprint(show, Mbig; context=:limit=>true)
    @test occursin("...", sbig)
    @test _is_ascii(sbig)

    # --- Submodule show ---
    S = PM.submodule(i; check_mono=true)

    s1 = sprint(show, S)
    @test occursin("Submodule", s1)
    @test _is_ascii(s1)

    s2 = sprint(show, MIME("text/plain"), S)
    @test occursin("Submodule", s2)
    @test occursin("sub dims", s2)
    @test occursin("ambient dims", s2)
    @test _is_ascii(s2)

    # --- ShortExactSequence show (must not force exactness check) ---
    ses = PM.ShortExactSequence(i, p; check=false)
    @test ses.checked == false

    s3 = sprint(show, ses)
    @test occursin("ShortExactSequence", s3)
    @test _is_ascii(s3)
    @test ses.checked == false  # show must not mutate cached status

    s4 = sprint(show, MIME("text/plain"), ses)
    @test occursin("0 -> A -(i)-> B -(p)-> C -> 0", s4)
    @test occursin("exact = unknown", s4)
    @test _is_ascii(s4)
    @test ses.checked == false

    # --- SnakeLemmaResult show ---
    # Build a small snake lemma instance (1-vertex, so linear algebra is tiny).
    top = PM.ShortExactSequence(i, p; check=true)

    Ab = IR.PModule{QQ}(P, [2], edge_maps)
    Bb = IR.PModule{QQ}(P, [3], edge_maps)
    Cb = IR.PModule{QQ}(P, [1], edge_maps)

    ib = IR.PMorphism{QQ}(Ab, Bb, [Matrix{QQ}([
        QQ(1) QQ(0);
        QQ(0) QQ(1);
        QQ(0) QQ(0)
    ])])
    pb = IR.PMorphism{QQ}(Bb, Cb, [Matrix{QQ}([QQ(0) QQ(0) QQ(1)])])
    bottom = PM.ShortExactSequence(ib, pb; check=true)

    alpha = IR.PMorphism{QQ}(A, Ab, [Matrix{QQ}([QQ(1); QQ(0)])])
    beta  = IR.PMorphism{QQ}(B, Bb, [Matrix{QQ}([
        QQ(1) QQ(0);
        QQ(0) QQ(1);
        QQ(0) QQ(0)
    ])])
    gamma = IR.PMorphism{QQ}(C, Cb, [Matrix{QQ}([QQ(0)])])

    sn = PM.snake_lemma(top, bottom, alpha, beta, gamma; check=true)

    s5 = sprint(show, sn)
    @test occursin("SnakeLemmaResult", s5)
    @test occursin("delta:", s5)
    @test _is_ascii(s5)

    s6 = sprint(show, MIME("text/plain"), sn)
    @test occursin("kerA -> kerB -> kerC --delta--> cokerA -> cokerB -> cokerC", s6)
    @test occursin("maps: k1, k2, delta, c1, c2", s6)
    @test _is_ascii(s6)
end

@testset "Derived-category primitives: cone triangle and LES" begin
    # C and D concentrated in degree 0, map f = 0.
    C = CC.CochainComplex{QQ}(0, 0, [1], SparseMatrixCSC{QQ,Int}[]; labels=[["c0"]])
    D = CC.CochainComplex{QQ}(0, 0, [1], SparseMatrixCSC{QQ,Int}[]; labels=[["d0"]])

    f0 = spzeros(QQ, 1, 1)
    f = CC.CochainMap(C, D, [f0]; check=true)

    tri = CC.mapping_cone_triangle(f)
    les = CC.long_exact_sequence(tri)

    @test les.tmin == -1
    @test les.tmax == 0

    # At t = -1: H^{-1}(D)=0 -> H^{-1}(Cone)=QQ -> H^0(C)=QQ
    idx = (-1) - les.tmin + 1
    @test les.HD[idx].dimH == 0
    @test les.Hcone[idx].dimH == 1
    @test les.HC[idx+1].dimH == 1

    # Connecting map delta^{-1} : H^{-1}(Cone) -> H^0(C) should be an isomorphism in this case.
    delta = les.deltaH[idx]
    @test size(delta) == (1, 1)
    @test delta[1, 1] == one(QQ)
end

@testset "Derived-category primitives: cone of identity is acyclic" begin
    # C: k -> k with zero differential. Cone(id_C) should be contractible, hence acyclic.
    C = CC.CochainComplex{QQ}(0, 1, [1, 1], [spzeros(QQ, 1, 1)])
    id1 = sparse([1], [1], [QQ(1)], 1, 1)
    f = CC.CochainMap(C, C, [id1, id1])
    tri = CC.mapping_cone_triangle(f)
    @test all(==(0), CC.homology_dims(tri.cone))
end

@testset "Spectral sequence: toy double complex" begin
    # Double complex with all blocks 1-dim:
    # dv at (0,0) is identity, other dv zero.
    # dh at (0,0) is identity, other dh zero.
    dims = [1 1;
            1 1]

    dv = Array{SparseMatrixCSC{QQ,Int},2}(undef, 2, 2)
    dh = Array{SparseMatrixCSC{QQ,Int},2}(undef, 2, 2)

    dv[1,1] = sparse([1],[1],[one(QQ)], 1, 1)
    dv[2,1] = spzeros(QQ, 1, 1)
    dv[1,2] = spzeros(QQ, 0, 1)
    dv[2,2] = spzeros(QQ, 0, 1)

    dh[1,1] = sparse([1],[1],[one(QQ)], 1, 1)
    dh[1,2] = spzeros(QQ, 1, 1)
    dh[2,1] = spzeros(QQ, 0, 1)
    dh[2,2] = spzeros(QQ, 0, 1)

    DC = CC.DoubleComplex{QQ}(0, 1, 0, 1, dims, dv, dh)
    ss = CC.spectral_sequence(DC; first=:vertical)

    @test ss.E1_dims == [0 0;
                         1 1]
    @test ss.E2_dims == [0 0;
                         1 1]
    @test ss.Einf_dims == [0 0;
                           1 1]
    @test ss.Htot_dims == [0, 1, 1]

    # Einf diagonal sums must match total cohomology dims.
    tmin = DC.amin + DC.bmin
    tmax = DC.amax + DC.bmax
    for t in tmin:tmax
        s = 0
        for a in DC.amin:DC.amax
            b = t - a
            if DC.bmin <= b <= DC.bmax
                s += ss.Einf_dims[a - DC.amin + 1, b - DC.bmin + 1]
            end
        end
        @test s == ss.Htot_dims[t - tmin + 1]
    end
end

@testset "Spectral sequence: horizontal identity forces E2 = 0 and Htot = 0" begin
    # Double complex concentrated in b=0 with an isomorphism horizontally:
    # (0,0)=k --id--> (1,0)=k. Total complex has zero cohomology.
    dims = reshape([1, 1], 2, 1)  # a=0,1; b=0
    dv = Array{SparseMatrixCSC{QQ,Int},2}(undef, 2, 1)
    dh = Array{SparseMatrixCSC{QQ,Int},2}(undef, 2, 1)

    dv[1,1] = spzeros(QQ, 0, 1)
    dv[2,1] = spzeros(QQ, 0, 1)

    dh[1,1] = sparse([1], [1], [QQ(1)], 1, 1)  # identity (0,0)->(1,0)
    dh[2,1] = spzeros(QQ, 0, 1)                # boundary

    DC = CC.DoubleComplex{QQ}(0, 1, 0, 0, dims, dv, dh)
    ss = CC.spectral_sequence(DC; first=:vertical)

    @test CC.page(ss, 1)[(0,0)] == 1
    @test CC.page(ss, 1)[(1,0)] == 1

    d1 = CC.differential(ss, 1, (0,0))
    @test size(d1) == (1, 1)
    @test d1[1,1] == QQ(1)

    @test CC.page(ss, 2)[(0,0)] == 0
    @test CC.page(ss, 2)[(1,0)] == 0
    @test ss.Htot_dims == [0, 0]
end

@testset "Ext spectral sequence wrapper: consistency with total complex dims" begin
    # Use helper constructors from runtests.jl (as other tests already do).
    P = chain_poset(3)

    # Build simple modules at vertices 1 and 3 via 1x1 fringe presentations.
    # Note: one_by_one_fringe expects (P, U, D), not just a vertex index.
    M = IR.pmodule_from_fringe(one_by_one_fringe(
        P,
        FF.principal_upset(P, 1),
        FF.principal_downset(P, 1),
    ))

    N = IR.pmodule_from_fringe(one_by_one_fringe(
        P,
        FF.principal_upset(P, 3),
        FF.principal_downset(P, 3),
    ))

    ss = PM.ExtSpectralSequence(M, N; maxlen=2)

    # Compare to ext_dims_via_resolutions on the same truncation.
    F, dF = PM.IndicatorResolutions.upset_resolution(M; maxlen=2)
    E, dE = PM.IndicatorResolutions.downset_resolution(N; maxlen=2)
    ext_dims = PM.HomExt.ext_dims_via_resolutions(F, dF, E, dE)

    tmin = ss.DC.amin + ss.DC.bmin
    tmax = ss.DC.amax + ss.DC.bmax
    for t in tmin:tmax
        @test ss.Htot_dims[t - tmin + 1] == get(ext_dims, t, 0)
    end

    # Einf diagonal sums must match Htot dims.
    for t in tmin:tmax
        s = 0
        for a in ss.DC.amin:ss.DC.amax
            b = t - a
            if ss.DC.bmin <= b <= ss.DC.bmax
                s += ss.Einf_dims[a - ss.DC.amin + 1, b - ss.DC.bmin + 1]
            end
        end
        @test s == ss.Htot_dims[t - tmin + 1]
    end
end


@testset "Spectral sequence v2: higher differential d2 example" begin
    # Construct a double complex with a known nontrivial d2:
    #
    # Columns a=0,1,2; rows b=0,1
    #
    # C^{0,1} = Q  --dh-->  C^{1,1} = Q
    #                     ^dv
    #                     |
    #             C^{1,0} = Q  --dh-->  C^{2,0} = Q
    #
    # dv(1,0)->(1,1) is identity, dh(0,1)->(1,1) is identity, dh(1,0)->(2,0) is identity.
    #
    # This forces d1=0 but d2: E2^{0,1} -> E2^{2,0} is an isomorphism.
    dims = [0 1;
            1 1;
            1 0]  # rows are a=0,1,2; cols are b=0,1

    dv = Array{SparseMatrixCSC{QQ,Int},2}(undef, 3, 2)
    dh = Array{SparseMatrixCSC{QQ,Int},2}(undef, 3, 2)

    # dv blocks: dv[a,b] maps (a,b)->(a,b+1)
    dv[1,1] = spzeros(QQ, 1, 0)                      # (0,0)->(0,1)
    dv[2,1] = sparse([1],[1],[one(QQ)], 1, 1)        # (1,0)->(1,1) identity
    dv[3,1] = spzeros(QQ, 0, 1)                      # (2,0)->(2,1) but (2,1)=0

    dv[1,2] = spzeros(QQ, 0, 1)                      # (0,1)->(0,2)=0
    dv[2,2] = spzeros(QQ, 0, 1)                      # (1,1)->(1,2)=0
    dv[3,2] = spzeros(QQ, 0, 0)                      # (2,1)->(2,2)=0

    # dh blocks: dh[a,b] maps (a,b)->(a+1,b)
    dh[1,1] = spzeros(QQ, 1, 0)                      # (0,0)->(1,0)
    dh[1,2] = sparse([1],[1],[one(QQ)], 1, 1)        # (0,1)->(1,1) identity

    dh[2,1] = sparse([1],[1],[one(QQ)], 1, 1)        # (1,0)->(2,0) identity
    dh[2,2] = spzeros(QQ, 0, 1)                      # (1,1)->(2,1)=0

    dh[3,1] = spzeros(QQ, 0, 1)                      # boundary
    dh[3,2] = spzeros(QQ, 0, 0)

    DC = CC.DoubleComplex{QQ}(0, 2, 0, 1, dims, dv, dh)
    ss = CC.spectral_sequence(DC; first=:vertical)

    # E2 has 1-dim at (0,1) and (2,0).
    @test CC.page(ss, 2)[(0,1)] == 1
    @test CC.page(ss, 2)[(2,0)] == 1

    # d2: E2^{0,1} -> E2^{2,0} is nonzero (in fact iso).
    d2 = CC.differential(ss, 2, (0,1))
    @test size(d2) == (1, 1)
    @test d2[1,1] == one(QQ)

    # After r=3, the page is stable and equals E_infty.
    @test CC.page(ss, 3)[(0,1)] == 0
    @test CC.page(ss, 3)[(2,0)] == 0
    @test ss.Einf_dims == zeros(Int, size(ss.Einf_dims))

    # Collapse detection should report collapse at r=3.
    @test CC.collapse_page(ss) == 3
end

@testset "Spectral sequence v2: filtration + edge maps on simple example" begin
    # Reuse existing example from earlier test: horizontal identity in b=0.
    dims = reshape([1, 1], 2, 1)
    dv = Array{SparseMatrixCSC{QQ,Int},2}(undef, 2, 1)
    dh = Array{SparseMatrixCSC{QQ,Int},2}(undef, 2, 1)

    dv[1,1] = spzeros(QQ, 0, 1)
    dv[2,1] = spzeros(QQ, 0, 1)

    dh[1,1] = sparse([1], [1], [QQ(1)], 1, 1)
    dh[2,1] = spzeros(QQ, 0, 1)

    DC = CC.DoubleComplex{QQ}(0, 1, 0, 0, dims, dv, dh)
    ss = CC.spectral_sequence(DC; first=:vertical)

    # total cohomology is zero, so filtration dims are all zero.
    @test ss.Htot_dims == [0, 0]
    @test CC.filtration_dims(ss, 0) == Dict(0 => 0, 1 => 0, 2 => 0) ||
          CC.filtration_dims(ss, 0) == Dict(0 => 0, 1 => 0)  # tolerate range differences

    # convergence report should produce a nonempty string
    rep = CC.convergence_report(ss)
    @test occursin("SpectralSequence", rep)
end

@testset "Spectral sequence v2: API helpers + E_infty edge projections" begin
    # Reuse the nontrivial d2 example from above to test bidegree bookkeeping.
    dims = [0 1;
            1 1;
            1 0]  # rows are a=0,1,2; cols are b=0,1

    dv = Array{SparseMatrixCSC{QQ,Int},2}(undef, 3, 2)
    dh = Array{SparseMatrixCSC{QQ,Int},2}(undef, 3, 2)

    dv[1,1] = spzeros(QQ, 1, 0)
    dv[2,1] = sparse([1],[1],[one(QQ)], 1, 1)
    dv[3,1] = spzeros(QQ, 0, 1)

    dv[1,2] = spzeros(QQ, 0, 1)
    dv[2,2] = spzeros(QQ, 0, 1)
    dv[3,2] = spzeros(QQ, 0, 0)

    dh[1,1] = spzeros(QQ, 1, 0)
    dh[1,2] = sparse([1],[1],[one(QQ)], 1, 1)

    dh[2,1] = sparse([1],[1],[one(QQ)], 1, 1)
    dh[2,2] = spzeros(QQ, 0, 1)

    dh[3,1] = spzeros(QQ, 0, 1)
    dh[3,2] = spzeros(QQ, 0, 0)

    DC = CC.DoubleComplex{QQ}(0, 2, 0, 1, dims, dv, dh)
    ss = CC.spectral_sequence(DC; first=:vertical)

    @test CC.dr_target(ss, 2, (0,1)) == (2,0)
    @test CC.dr_source(ss, 2, (2,0)) == (0,1)
    @test CC.dr_target(ss, 3, (0,1)) === nothing

    # ss[r] and ss[r,(a,b)] indexing
    @test ss[2][(0,1)] == 1
    @test ss[2,(0,1)] == 1

    # page_terms provides explicit subquotient models on intermediate pages.
    spaces2 = CC.page_terms(ss, 2)
    @test spaces2[1,2].dimH == 1  # (a,b)=(0,1)
    @test spaces2[3,1].dimH == 1  # (a,b)=(2,0)

    # E_infty terms are 0-dimensional in this example.
    @test CC.page_terms(ss, :inf)[1,2].dimH == 0

    # A zero-differential example to test explicit E_infty edge splittings.
    dims2 = [1 1;
             1 1]  # rows are a=0,1; cols are b=0,1

    dv2 = Array{SparseMatrixCSC{QQ,Int},2}(undef, 2, 2)
    dh2 = Array{SparseMatrixCSC{QQ,Int},2}(undef, 2, 2)

    dv2[1,1] = spzeros(QQ, 1, 1)
    dv2[2,1] = spzeros(QQ, 1, 1)
    dv2[1,2] = spzeros(QQ, 0, 1)
    dv2[2,2] = spzeros(QQ, 0, 1)

    dh2[1,1] = spzeros(QQ, 1, 1)
    dh2[1,2] = spzeros(QQ, 1, 1)
    dh2[2,1] = spzeros(QQ, 0, 1)
    dh2[2,2] = spzeros(QQ, 0, 1)

    DC2 = CC.DoubleComplex{QQ}(0, 1, 0, 1, dims2, dv2, dh2)
    ss2 = CC.spectral_sequence(DC2; first=:vertical)

    @test ss2.Htot_dims == [1,2,1]

    # Multiplicative structure helpers (with a toy "unit" multiplication on Tot).
    # This multiplication treats the unique basis element in Tot^0 as a unit and
    # sets all other products to zero.
    mul = function (t1::Int, x::Vector{QQ}, t2::Int, y::Vector{QQ})
        tout = t1 + t2
        outdim = ss2.Tot.dims[tout - ss2.Tot.tmin + 1]
        if t1 == 0
            return x[1] * y
        elseif t2 == 0
            return y[1] * x
        else
            return zeros(QQ, outdim)
        end
    end

    Punit = CC.product_matrix(ss2, 1, (0,0), (0,1), mul)
    @test size(Punit) == (1, 1)
    @test Punit[1,1] == one(QQ)

    Pzero = CC.product_matrix(ss2, 1, (0,1), (1,0), mul)
    @test size(Pzero) == (1, 1)
    @test Pzero[1,1] == zero(QQ)

    cunit = CC.product_coords(ss2, 1, (0,0), [one(QQ)], (0,1), [one(QQ)], mul)
    @test size(cunit) == (1, 1)
    @test cunit[1,1] == one(QQ)

    # On total degree t=1, there are two graded pieces: (0,1) and (1,0).
    inc01 = CC.edge_inclusion(ss2, (0,1))
    inc10 = CC.edge_inclusion(ss2, (1,0))
    proj01 = CC.edge_projection(ss2, (0,1))
    proj10 = CC.edge_projection(ss2, (1,0))

    @test size(inc01) == (2, 1)
    @test size(inc10) == (2, 1)
    @test size(proj01) == (1, 2)
    @test size(proj10) == (1, 2)

    M01 = proj01 * inc01
    M10 = proj10 * inc10
    @test size(M01) == (1, 1)
    @test size(M10) == (1, 1)
    @test M01[1,1] == one(QQ)
    @test M10[1,1] == one(QQ)

    # The chosen splitting gives a direct-sum decomposition of H^1.
    Psum = inc01 * proj01 + inc10 * proj10
    I2 = [one(QQ) zero(QQ);
          zero(QQ) one(QQ)]
    @test Psum == I2

    spl = CC.split_total_cohomology(ss2, 1)
    @test size(spl.B) == (2, 2)
    @test size(spl.Binv) == (2, 2)
    @test spl.B * spl.Binv == I2

    # filtration_subquotient agrees with the E_infty terms.
    @test CC.filtration_subquotient(ss2, 0, 1).dimH == 1
    @test CC.filtration_subquotient(ss2, 1, 1).dimH == 1

    # Pretty-printing smoke tests (ASCII).
    ptxt = repr("text/plain", CC.page(ss2, 1))
    @test occursin("E_1", ptxt)
    stxt = repr("text/plain", ss2)
    @test occursin("SpectralSequence", stxt)

    @testset "Spectral sequence workflow helpers (E2 objects, filtrations, extensions)" begin
        # E2 as objects (SubquotientData) via an indexable page wrapper
        P2 = PM.E2_terms(ss)
        @test P2[(0,0)].dimH == PM.term(ss, 2, (0,0)).dimH

        # Dict helpers keyed by bidegree
        d2 = PM.page_terms_dict(ss, 2)
        @test haskey(d2, (0,0))
        @test d2[(0,0)].dimH == PM.term(ss, 2, (0,0)).dimH

        d2dims = PM.page_dims_dict(ss, 2)
        @test haskey(d2dims, (0,0))
        @test d2dims[(0,0)] == PM.page(ss, 2)[(0,0)]

        # Diagonal criterion should hold at E_infty for this convergent toy example
        @test PM.diagonal_criterion(ss; r=:inf)

        # Filtration packaging: graded piece dims sum to total cohomology dim
        Htot = PM.total_cohomology_dims(ss)
        for t in keys(Htot)
            fd = PM.filtration_data(ss, t)
            gsum = 0
            for p in fd.pmin:(fd.pmax - 1)
                gsum += fd.graded[p].dimH
            end
            @test gsum == Htot[t]
        end

        # Collapse info returns explicit filtrations
        cd = PM.collapse_data(ss)
        @test cd.collapse_r == PM.collapse_page(ss)
        @test cd.diagonal_ok

        # Extension problem helper returns an explicit splitting of H^t
        for t in keys(Htot)
            ep = PM.extension_problem(ss, t)
            d = Htot[t]
            @test size(ep.B, 1) == d
            @test size(ep.B, 2) == d
            @test ep.Binv * ep.B == Matrix{QQ}(I, d, d)
            @test ep.B * ep.Binv == Matrix{QQ}(I, d, d)
        end
    end
end