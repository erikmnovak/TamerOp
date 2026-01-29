using Test
using Random
using SparseArrays

const FZ = PM.FlangeZn
const ZE = PM.ZnEncoding
const IR = PM.IndicatorResolutions
const QQ = PM.QQ

# ---------------------------------------------------------------------------
# Local helpers for the updated FlangeZn indicator API.
#
# The library intentionally removed the old Flat/Injective and flat/inj shims
# (and also disallows ambiguous reordered IndFlat/IndInj constructors) because
# both inputs are vectors and mixups were a frequent source of subtle bugs.
# ---------------------------------------------------------------------------
mk_face(n, coords) = FZ.face(n, coords)
mk_flat(b, coords; id=:F) = FZ.IndFlat(mk_face(length(b), coords), b; id=id)
mk_inj(b, coords; id=:E) = FZ.IndInj(mk_face(length(b), coords), b; id=id)


@testset "Zn wrappers: common encoding matches explicit encoding route" begin
    # Two 1D flanges representing interval modules [0,5] and [2,7].
    tau = FZ.face(1, [])  # no free coordinates for n=1 (i.e. an interval-type flange)

    F1 = FZ.IndFlat(tau, [0]; id=:F1)
    E1 = FZ.IndInj(tau, [5]; id=:E1)
    FG1 = FZ.Flange{QQ}(1, [F1], [E1], reshape([QQ(1)], 1, 1))

    F2 = FZ.IndFlat(tau, [2]; id=:F2)
    E2 = FZ.IndInj(tau, [7]; id=:E2)
    FG2 = FZ.Flange{QQ}(1, [F2], [E2], reshape([QQ(1)], 1, 1))

    enc = PM.EncodingOptions(backend=:zn, max_regions=50_000)
    df = PM.DerivedFunctorOptions(maxdeg=2)
    res = PM.ResolutionOptions(maxlen=3)

    enc_out = PM.encode_pmodules_from_flanges(FG1, FG2, enc)
    P = enc_out.P
    Ms = enc_out.Ms

    @test length(Ms) == 2
    @test Ms[1].Q === P
    @test Ms[2].Q === P

    E_explicit = PM.Ext(Ms[1], Ms[2], df)
    E_wrap = PM.ExtZn(FG1, FG2, enc, df)

    @test [PM.dim(E_explicit, t) for t in 0:2] == [PM.dim(E_wrap, t) for t in 0:2]

    # Resolution wrappers: compare against "encode + resolution" directly.
    enc1 = PM.encode_pmodule_from_flange(FG1, enc)
    res_wrap = PM.projective_resolution_Zn(FG1, enc, res; return_encoding=true)
    @test res_wrap.P.leq == enc1.P.leq
    @test PM.betti_table(res_wrap.res) == PM.betti_table(PM.projective_resolution(enc1.M, res))

    res_min = PM.ResolutionOptions(maxlen=3, minimal=true, check=true)

    bt_wrap = PM.betti(PM.projective_resolution_Zn(FG1, enc, res_min))
    bt_explicit = PM.betti(PM.projective_resolution(enc1.M, res_min))
    @test bt_wrap == bt_explicit
end

@testset "Wrappers for Z^n: injective resolutions and minimal Bass data" begin
    n = 1
    flats = [mk_flat([0],[false]), mk_flat([2],[false])]
    inj   = [mk_inj([4],[false])]
    Phi = [QQ(1); QQ(1)] |> x -> reshape(x, 1, 2)
    FG = FZ.Flange{QQ}(n, flats, inj, Phi)

    enc = PM.EncodingOptions(backend=:zn, max_regions=50_000)
    res = PM.ResolutionOptions(maxlen=3, check=true)

    P, M, pi = PM.encode_pmodule_from_flange(FG, enc)

    # Compare injective resolutions (wrapper vs explicit encode-then-resolve)
    resI = PM.injective_resolution(M, res)
    resI_Z = PM.injective_resolution_Zn(FG, enc, res)
    @test PM.bass_table(resI_Z) == PM.bass_table(resI)

    # Minimal injective resolutions and minimal Bass invariants
    res_min = PM.ResolutionOptions(maxlen=3, minimal=true, check=true)

    resMinI = PM.injective_resolution(M, res_min)
    resMinI_Z = PM.injective_resolution_Zn(FG, enc, res_min)
    @test PM.bass_table(resMinI_Z) == PM.bass_table(resMinI)

    @test PM.bass(resMinI_Z) == PM.bass(resMinI)
end

@testset "FlangeZn: IndFlat/IndInj constructors (new API)" begin
    tau = FZ.face(2, [2])

    # Canonical constructor order: (tau, b; id=...)
    F = FZ.IndFlat(tau, [1, 2])
    @test F isa FZ.IndFlat
    @test F.id == :F
    @test F.b == [1, 2]
    @test F.tau == tau

    E = FZ.IndInj(tau, [3, 4]; id=:E)
    @test E isa FZ.IndInj
    @test E.id == :E
    @test E.b == [3, 4]
    @test E.tau == tau

    # The library intentionally does NOT support reordered positional arguments
    # (both inputs are vectors, so such constructors were a source of subtle bugs).
    @test_throws MethodError FZ.IndFlat([1, 2], tau, :F2)
    @test_throws MethodError FZ.IndInj([3, 4], tau, :E2)
end

@testset "FlangeZn dim_at + minimize invariance" begin
    # n = 1 interval [b,c] via flat (>= b) and injective (<= c)
    n = 1
    tau0 = FZ.Face(n, [false])
    b = 1
    c = 3
    F1 = FZ.IndFlat(tau0, [b]; id=:F1)
    E1 = FZ.IndInj(tau0, [c]; id=:E1)
    Phi = reshape(QQ[QQ(1)], 1, 1)
    FG = FZ.Flange{QQ}(n, [F1], [E1], Phi)

    # dim is 1 on [b,c], 0 otherwise
    for g in (b-2):(c+2)
        d = FZ.dim_at(FG, [g]; rankfun=EX.rankQQ)
        expected = (b <= g <= c) ? 1 : 0
        @test d == expected
    end

    # intersects should detect empty intersection when b > c
    F_bad = FZ.IndFlat(tau0, [5]; id=:Fbad)
    E_bad = FZ.IndInj(tau0, [2]; id=:Ebad)
    @test FZ.intersects(F_bad, E_bad) == false

    # Minimize should merge proportional duplicate columns without changing dim_at
    F2 = FZ.IndFlat(tau0, [b]; id=:F2)
    Phi2 = reshape(QQ[QQ(1), QQ(2)], 1, 2)    # second column is 2x the first
    FG2 = FZ.Flange{QQ}(n, [F1, F2], [E1], Phi2)
    FG2m = FZ.minimize(FG2)

    for g in (b-1):(c+1)
        d1 = FZ.dim_at(FG2, [g];  rankfun=EX.rankQQ)
        d2 = FZ.dim_at(FG2m, [g]; rankfun=EX.rankQQ)
        @test d1 == d2
    end

    @testset "canonical_matrix / degree_matrix / bounding_box" begin
        # 1D: flat is x >= 1, inj is x <= 3.
        flats = [FZ.IndFlat(mk_face(length([1]), [false]), [1]; id=:F)]
        injectives = [FZ.IndInj(mk_face(length([3]), [false]), [3]; id=:E)]
        Phi = FZ.canonical_matrix(flats, injectives)
        @test Phi == reshape(QQ[1], 1, 1)

        # Non-intersecting pair: x >= 5 and x <= 3.
        flats_bad = [FZ.IndFlat(mk_face(length([5]), [false]), [5]; id=:Fbad)]
        injectives_bad = [FZ.IndInj(mk_face(length([3]), [false]), [3]; id=:Ebad)]
        Phibad = FZ.canonical_matrix(flats_bad, injectives_bad)
        @test Phibad == reshape(QQ[0], 1, 1)

        # degree_matrix should pick out exactly the active row/col at a given degree.
        Phi2 = reshape(QQ[2], 1, 1)
        H = FZ.Flange{QQ}(1, flats, injectives, Phi2)
        Phi_sub, rows, cols = FZ.degree_matrix(H, [2])
        @test rows == [1]
        @test cols == [1]
        @test Phi_sub == reshape(QQ[2], 1, 1)

        # Outside the intersection, there should be no active flats or injectives.
        Phi_sub2, rows2, cols2 = FZ.degree_matrix(H, [10])
        @test rows2 == Int[]
        @test cols2 == Int[]
        @test size(Phi_sub2) == (0, 0)

        # bounding_box in 1D with margin 1:
        #   flats force a >= (b_flat - margin) = 0
        #   injectives force b <= (b_inj + margin) = 4
        a_box, b_box = FZ.bounding_box(H; margin=1)
        @test a_box == [0]
        @test b_box == [4]
    end

    @testset "minimize: do not merge different labels, but merge proportional duplicates" begin
        # Two proportional columns with different underlying flats must not be merged.
        F1 = FZ.IndFlat(mk_face(length([0]), [false]), [0]; id=:F1)
        F2 = FZ.IndFlat(mk_face(length([1]), [false]), [1]; id=:F2)  # different threshold => different upset
        E1 = FZ.IndInj(mk_face(length([2]), [false]), [2]; id=:E1)

        # Columns are proportional but flats differ.
        Phi = QQ[1 2]
        H = FZ.Flange{QQ}(1, [F1, F2], [E1], Phi)
        Hmin = FZ.minimize(H)
        @test length(Hmin.flats) == 2

        # Two proportional rows with identical injectives should be merged.
        Fin = [FZ.IndFlat(mk_face(length([0]), [false]), [0]; id=:F)]
        Einj1 = FZ.IndInj(mk_face(length([0]), [false]), [0]; id=:E)
        Einj2 = FZ.IndInj(mk_face(length([0]), [false]), [0]; id=:Edup) # same underlying downset as Einj1
        Phi_rows = reshape(QQ[1, 2], 2, 1)        # 2x1, proportional rows
        H2 = FZ.Flange{QQ}(1, Fin, [Einj1, Einj2], Phi_rows)
        H2min = FZ.minimize(H2)
        @test length(H2min.injectives) == 1

        # Rank at degree 0 should be unchanged by minimization.
        @test FZ.dim_at(H2, [0]) == FZ.dim_at(H2min, [0])
    end

    @testset "ZnEncoding: region encoding without enumerating a box" begin
        # FG, b, c are in scope here because this testset is nested.
        enc = PM.EncodingOptions(backend=:zn, max_regions=1000)
        P, Henc, pi = PM.encode_from_flange(FG, enc)

        # The encoding poset should be a 3-chain:
        #   left of b  <  between  <  right of c.
        @test P.n == 3
        @test Set(FF.cover_edges(P)) == Set([(1,2),(2,3)])

        # In 1D, critical coordinates come from:
        #   flat threshold b
        #   complement(injective) threshold c+1
        @test pi.coords[1] == [b, c+1]

        # Spot-check the encoded fiber dimensions via locate.
        @test FF.fiber_dimension(Henc, PM.locate(pi, [b-5])) == 0
        @test FF.fiber_dimension(Henc, PM.locate(pi, [b])) == 1
        @test FF.fiber_dimension(Henc, PM.locate(pi, [c])) == 1
        @test FF.fiber_dimension(Henc, PM.locate(pi, [c+1])) == 0
    end

    @testset "ZnEncoding: direct flange -> fringe (Remark 6.14 bridge)" begin
        # Build the encoding only, then push FG down to a fringe presentation.
        enc = PM.EncodingOptions(backend=:zn, max_regions=1000)
        P, pi = PM.encode_poset_from_flanges(FG, enc)
        H = PM.fringe_from_flange(P, pi, FG)   # strict=true by default

        # Compare with the convenience wrapper.
        P2, H2, pi2 = PM.encode_from_flange(FG, enc)
        @test P.n == P2.n
        @test Set(FF.cover_edges(P)) == Set(FF.cover_edges(P2))

        # Fiber dimensions should match on all sampled degrees.
        for g in (b-5):(c+5)
            t  = PM.locate(pi,  [g])
            t2 = PM.locate(pi2, [g])
            @test t == t2
            if t != 0
                @test FF.fiber_dimension(H,  t)  == FF.fiber_dimension(H2, t2)
            end
        end

        # strictness: labels not present in the encoding must be rejected.
        F_extra = FZ.IndFlat(mk_face(length([b + 1]), [false]), [b + 1]; id=:Fextra)
        E = FZ.IndInj(mk_face(length([c]), [false]), [c]; id=:E)
        FG_extra = FZ.Flange{QQ}(1, [FZ.IndFlat(mk_face(length([b]), [false]), [b]; id=:F), F_extra], [E], QQ[1 1])

        @test_throws ErrorException PM.fringe_from_flange(P, pi, FG_extra)
    end
end

@testset "CrossValidateFlangePL smoke test" begin
    n = 1
    tau0 = FZ.Face(n, [false])
    F1 = FZ.IndFlat(tau0, [1]; id=:F1)
    E1 = FZ.IndInj(tau0, [3]; id=:E1)
    Phi = reshape(QQ[QQ(1)], 1, 1)
    FG = FZ.Flange{QQ}(n, [F1], [E1], Phi)

    ok, report = FZ.cross_validate(FG; margin=1, rankfun=EX.rankQQ)
    @test ok == true
    @test haskey(report, "mismatches")
    @test isempty(report["mismatches"])
end

@testset "ZnEncoding 2D: free coordinate compression and correctness" begin
    # A 2D flange that depends only on coordinate 1; coordinate 2 is free everywhere.
    flats = [mk_flat([0, 0], [false, true])]
    inj   = [mk_inj([1, 0], [false, true])]
    Phi   = reshape(QQ[1], 1, 1)
    FG = FZ.Flange{QQ}(2, flats, inj, Phi)

    enc = PM.EncodingOptions(backend=:zn, max_regions=100)
    P, M, pi = PM.encode_pmodule_from_flange(FG, enc)

    # Expected: only 3 regions along coordinate 1 (below, inside, above), and 1 slab along coord 2.
    @test P.n == 3

    # pi should ignore coordinate 2
    for g1 in -3:4
        u = PM.locate(pi, [g1, -10])
        v = PM.locate(pi, [g1,  10])
        @test u == v
    end

    # Monotonicity: g <= h implies pi(g) <= pi(h)
    for g1 in -3:3, h1 in g1:4
        ug = PM.locate(pi, [g1, 0])
        uh = PM.locate(pi, [h1, 0])
        @test P.leq[ug, uh]
    end

    # Dimension consistency on a representative grid of lattice points
    for g1 in -3:4, g2 in (-2, 0, 7)
        g = [g1, g2]
        @test FZ.dim_at(FG, g) == M.dims[PM.locate(pi, g)]
    end
end

@testset "ZnEncoding 2D: common encoding for multiple flanges" begin
    # Two slabs along coordinate 1, coordinate 2 free.
    FG1 = FZ.Flange{QQ}(2,
        [mk_flat([0,0],[false,true])],
        [mk_inj([1,0],[false,true])],
        reshape(QQ[1], 1, 1)
    )
    FG2 = FZ.Flange{QQ}(2,
        [mk_flat([1,0],[false,true])],
        [mk_inj([2,0],[false,true])],
        reshape(QQ[1], 1, 1)
    )

    enc = PM.EncodingOptions(backend=:zn, max_regions=200)
    P, Ms, pi = PM.encode_pmodules_from_flanges(FG1, FG2, enc)
    M1, M2 = Ms

    # Critical coordinates along g1 are {0,1,2,3} giving <= 5 slabs => P.n <= 5.
    @test P.n <= 5

    for g1 in -1:4, g2 in (-5, 0, 5)
        g = [g1, g2]
        u = PM.locate(pi, g)
        @test u != 0
        @test FZ.dim_at(FG1, g) == M1.dims[u]
        @test FZ.dim_at(FG2, g) == M2.dims[u]
    end
end

@testset "ZnEncoding 2D: strict fringe_from_flange rejects missing generators" begin
    FG1 = FZ.Flange{QQ}(2,
        [mk_flat([0,0],[false,true])],
        [mk_inj([1,0],[false,true])],
        reshape(QQ[1], 1, 1)
    )
    FG2 = FZ.Flange{QQ}(2,
        [mk_flat([10,0],[false,true])],  # new flat label not present in FG1 encoding
        [mk_inj([11,0],[false,true])],
        reshape(QQ[1], 1, 1)
    )

    enc = PM.EncodingOptions(backend=:zn, max_regions=200)
    P, pi = ZE.encode_poset_from_flanges(FG1, enc)

    @test_throws ErrorException ZE.fringe_from_flange(P, pi, FG2; strict=true)

    # Non-strict mode should still push forward, dropping unmatched generators safely.
    H2 = ZE.fringe_from_flange(P, pi, FG2; strict=false)
    M2 = IR.pmodule_from_fringe(H2)

    # Sanity: the pushed module is defined on P
    @test length(M2.dims) == P.n
end

@testset "ZnEncodingMap region_weights: exact methods agree" begin
    # Same 2D setup used in test_znencoding_2d_regions.jl:
    # - one flat at x1 >= 0 (x2 free)
    # - one injective at x1 <= 1 (x2 free)
    flats = [mk_flat([0, 0], [false, true])]
    injs  = [mk_inj([1, 0], [false, true])]
    Phi   = reshape(QQ[1], 1, 1)
    FG    = FZ.Flange{QQ}(2, flats, injs, Phi)

    enc = PM.EncodingOptions(backend=:zn, max_regions=100)
    P, Henc, pi = PM.encode_pmodule_from_flange(FG, enc)

    a = [-2, -3]
    b = [ 3,  4]
    len2 = b[2] - a[2] + 1  # length in free coordinate

    # Determine region indices by locating representative points.
    rid_left  = PM.locate(pi, [-1, 0])  # x1 < 0
    rid_mid   = PM.locate(pi, [ 0, 0])  # 0 <= x1 <= 1
    rid_right = PM.locate(pi, [ 2, 0])  # x1 > 1

    expected = zeros(Int, length(pi.sig_y))
    expected[rid_left]  = 2 * len2  # x1 = -2,-1
    expected[rid_mid]   = 2 * len2  # x1 = 0,1
    expected[rid_right] = 2 * len2  # x1 = 2,3

    w_cells   = PM.region_weights(pi; box=(a, b), method=:cells)
    w_points  = PM.region_weights(pi; box=(a, b), method=:points)
    w_auto    = PM.region_weights(pi; box=(a, b), method=:auto)

    @test w_cells == expected
    @test w_points == expected
    @test w_auto == expected
end

@testset "ZnEncodingMap region_weights: Monte Carlo sampling is close (reproducible)" begin
    flats = [mk_flat([0, 0], [false, true])]
    injs  = [mk_inj([1, 0], [false, true])]
    Phi   = reshape(QQ[1], 1, 1)
    FG    = FZ.Flange{QQ}(2, flats, injs, Phi)

    enc = PM.EncodingOptions(backend=:zn, max_regions=100)
    P, Henc, pi = PM.encode_pmodule_from_flange(FG, enc)

    a = [-50, -500]
    b = [ 49,  499]
    len2 = b[2] - a[2] + 1  # 1000

    rid_left  = PM.locate(pi, [-1, 0])
    rid_mid   = PM.locate(pi, [ 0, 0])
    rid_right = PM.locate(pi, [ 2, 0])

    expected = zeros(Int, length(pi.sig_y))
    expected[rid_left]  = 50 * len2
    expected[rid_mid]   =  2 * len2
    expected[rid_right] = 48 * len2

    # Monte Carlo estimate
    rng = MersenneTwister(0)
    info = PM.region_weights(pi; box=(a, b), method=:sample, nsamples=20_000, rng=rng, return_info=true)

    @test info.method == :sample
    @test eltype(info.weights) == Float64
    @test length(info.weights) == length(expected)
    @test info.stderr !== nothing
    @test length(info.stderr) == length(expected)

    # Total points in the box:
    total_points = (b[1] - a[1] + 1) * (b[2] - a[2] + 1)
    @test isapprox(sum(info.weights), total_points; rtol=0.03)

    # Each bin should be reasonably close. Tolerances chosen for test stability.
    for j in eachindex(expected)
        @test isapprox(info.weights[j], expected[j]; rtol=0.06, atol=5_000.0)
    end
end

@testset "ZnEncodingMap region_weights: auto can be forced to sampling" begin
    flats = [mk_flat([0, 0], [false, true])]
    injs  = [mk_inj([1, 0], [false, true])]
    Phi   = reshape(QQ[1], 1, 1)
    FG    = FZ.Flange{QQ}(2, flats, injs, Phi)

    enc = PM.EncodingOptions(backend=:zn, max_regions=100)
    P, Henc, pi = PM.encode_pmodule_from_flange(FG, enc)

    a = [-50, -500]
    b = [ 49,  499]

    rng = MersenneTwister(1)
    info = PM.region_weights(pi; box=(a, b), method=:auto, max_cells=0, max_points=0,
                             nsamples=1_000, rng=rng, return_info=true)

    @test info.method == :sample
    @test eltype(info.weights) == Float64
end

@testset "ZnEncodingMap region_weights: :auto count_type promotes to BigInt on overflow" begin
    flats = [mk_flat([0, 0], [false, true])]
    injs  = [mk_inj([1, 0], [false, true])]
    Phi   = reshape(QQ[1], 1, 1)
    FG    = FZ.Flange{QQ}(2, flats, injs, Phi)

    enc = PM.EncodingOptions(backend=:zn, max_regions=100)
    P, Henc, pi = PM.encode_pmodule_from_flange(FG, enc)

    # Choose an interval length > typemax(Int) while endpoints still fit in Int64.
    a = [-9_000_000_000_000_000_000, 0]
    b = [ 9_000_000_000_000_000_000, 0]

    rid_left  = PM.locate(pi, [-1, 0])
    rid_mid   = PM.locate(pi, [ 0, 0])
    rid_right = PM.locate(pi, [ 2, 0])

    expected = zeros(BigInt, length(pi.sig_y))
    expected[rid_left]  = BigInt(-1) - BigInt(a[1]) + 1                 # a1..-1
    expected[rid_mid]   = BigInt(2)                                     # 0..1
    expected[rid_right] = BigInt(b[1]) - BigInt(2) + 1                  # 2..b1

    w = PM.region_weights(pi; box=(a, b), method=:cells, count_type=:auto)
    @test eltype(w) == BigInt
    @test w == expected
    @test sum(w) == (BigInt(b[1]) - BigInt(a[1]) + 1) * (BigInt(b[2]) - BigInt(a[2]) + 1)
end

@testset "region_poset(pi) reconstructs the encoder region poset" begin
    # -------------------------------------------------------------------------
    # 1) ZnEncoding: pi has (sig_y, sig_z) but no pi.P; region_poset should match P
    # -------------------------------------------------------------------------
    let
        n = 1
        b = [0]
        c = [5]
        I = FZ.Face(n, Int[])
        flats = [FZ.IndFlat(I, b; id=:F)]
        injectives = [FZ.IndInj(I, c; id=:E)]
        Phi = spzeros(QQ, 1, 1); Phi[1, 1] = 1
        FG = FZ.Flange{QQ}(n, flats, injectives, Phi)

        enc = PM.EncodingOptions(backend=:zn, max_regions=1000)
        Penc, Henc, pi = PM.encode_from_flange(FG, enc)

        Q = PM.Invariants.region_poset(pi)
        @test Q.n == Penc.n
        @test Q.leq == Penc.leq

        # Cached repeat call should return the exact same poset object.
        Q2 = PM.Invariants.region_poset(pi)
        @test Q2 === Q

        # Projected arrangement should work without requiring pi.P.
        arr = PM.projected_arrangement(pi; dirs=[[1.0]])
        @test arr.Q.leq == Penc.leq

        # And should accept a provided Q for maximum speed.
        arr2 = PM.projected_arrangement(pi; dirs=[[1.0]], Q=Penc)
        @test arr2.Q.leq == Penc.leq
    end

    # -------------------------------------------------------------------------
    # 2) PLPolyhedra: PLEncodingMap has (sig_y, sig_z) but no pi.P
    # -------------------------------------------------------------------------
    let
        n = 1

        # PLPolyhedra uses H-polytopes { x : A*x <= b }.
        # Upset: x >= 0  <=>  (-x <= 0)
        Uhp = PLP.make_hpoly([-1.0], 0.0)

        # Downset: x <= 2 <=>  (x <= 2)
        Dhp = PLP.make_hpoly([1.0], 2.0)

        U = PLP.PLUpset(PLP.PolyUnion(n, [Uhp]))
        D = PLP.PLDownset(PLP.PolyUnion(n, [Dhp]))

        # PLFringe requires an explicit Phi of size (#Downs) x (#Ups).
        F1 = PLP.PLFringe([U], PLP.PLDownset[], zeros(QQ, 0, 1))
        F2 = PLP.PLFringe(PLP.PLUpset[], [D], zeros(QQ, 1, 0))

        enc = PM.EncodingOptions(backend=:pl, max_regions=10_000)
        Ppl, Hpl, pipl = PLP.encode_from_PL_fringes(F1, F2, enc)

        Qpl = PM.Invariants.region_poset(pipl)
        @test Qpl.n == Ppl.n
        @test Qpl.leq == Ppl.leq

        Qpl2 = PM.Invariants.region_poset(pipl)
        @test Qpl2 === Qpl

        arr = PM.projected_arrangement(pipl; dirs=[[1.0]])
        @test arr.Q.leq == Ppl.leq

        arr2 = PM.projected_arrangement(pipl; dirs=[[1.0]], Q=Ppl)
        @test arr2.Q.leq == Ppl.leq
    end


    # -------------------------------------------------------------------------
    # 3) PLBackend: PLEncodingMapBoxes has (sig_y, sig_z) but no pi.P
    # -------------------------------------------------------------------------
    if isdefined(PM, :PLBackend)
        let
            PB = PM.PLBackend
            n = 1
            U = PB.BoxUpset([0.0], [Inf])
            D = PB.BoxDownset([-Inf], [2.0])

            Ups = [U]
            Downs = [D]

            enc = PM.EncodingOptions(backend=:pl_backend, max_regions=10_000)
            Pbx, Hbx, pibx = PB.encode_fringe_boxes(Ups, Downs, enc)

            Qbx = PM.Invariants.region_poset(pibx)
            @test Qbx.n == Pbx.n
            @test Qbx.leq == Pbx.leq

            Qbx2 = PM.Invariants.region_poset(pibx)
            @test Qbx2 === Qbx

            arr = PM.projected_arrangement(pibx; dirs=[[1.0]])
            @test arr.Q.leq == Pbx.leq

            arr2 = PM.projected_arrangement(pibx; dirs=[[1.0]], Q=Pbx)
            @test arr2.Q.leq == Pbx.leq
        end
    end
end
