using Test

# Included from test/runtests.jl; uses shared aliases (PM, PLP, PLB, FF, QQ, ...).


@testset "Rn wrappers: common encoding matches explicit encoding route" begin
    # This test checks that the one-line R^n wrappers:
    #   - build the same finite encoding poset P as an explicit "encode then compute" route, and
    #   - give the same answers as running the finite-poset homological algebra directly.
    #
    # IMPORTANT:
    # The PL -> finite encoding step requires Polyhedra/CDDLib for region enumeration.
    # If Polyhedra is not available, we only test that the wrapper fails loudly.

    # Build two simple box modules in R^2 using the current PLPolyhedra API.
    # PLUpset/PLDownset wrap PolyUnion(n, parts::Vector{HPoly}).
    # make_hpoly(A, b) builds an HPoly for inequalities A*x <= b.

    # F1: support approximately [0,1]^2
    U1_hp = PLP.make_hpoly(QQ[-1 0; 0 -1], QQ[0, 0])    # x >= 0, y >= 0
    D1_hp = PLP.make_hpoly(QQ[ 1 0; 0  1], QQ[1, 1])    # x <= 1, y <= 1
    U1 = PLP.PLUpset(PLP.PolyUnion(2, [U1_hp]))
    D1 = PLP.PLDownset(PLP.PolyUnion(2, [D1_hp]))
    F1 = PLP.PLFringe([U1], [D1], reshape(QQ[1], 1, 1))

    # F2: support approximately [1,2]^2
    U2_hp = PLP.make_hpoly(QQ[-1 0; 0 -1], QQ[-1, -1])  # x >= 1, y >= 1
    D2_hp = PLP.make_hpoly(QQ[ 1 0; 0  1], QQ[2, 2])    # x <= 2, y <= 2
    U2 = PLP.PLUpset(PLP.PolyUnion(2, [U2_hp]))
    D2 = PLP.PLDownset(PLP.PolyUnion(2, [D2_hp]))
    F2 = PLP.PLFringe([U2], [D2], reshape(QQ[1], 1, 1))

    enc_pl = PM.EncodingOptions(backend=:pl, max_regions=50_000, strict_eps=PLP.STRICT_EPS_QQ)
    df2 = PM.DerivedFunctorOptions(maxdeg=2)
    res3 = PM.ResolutionOptions(maxlen=3)

    # If Polyhedra is missing, encoding is unavailable by design.
    if !PLP.HAVE_POLY
        @test_throws ErrorException DF.encode_pmodules_from_PL_fringes(F1, F2, enc_pl)
        return
    end

    # Common-encode both PL presentations to the same finite poset P and modules Ms on P.
    enc = DF.encode_pmodules_from_PL_fringes(F1, F2, enc_pl)
    P = enc.P
    Ms = enc.Ms

    # Ext computed on P should match ExtRn wrapper (which internally does the same steps).
    E_explicit = DF.Ext(Ms[1], Ms[2], df2)
    E_wrap = DF.ExtRn(F1, F2, enc_pl, df2)
    @test [PM.dim(E_explicit, t) for t in 0:2] == [PM.dim(E_wrap, t) for t in 0:2]

    # Resolution wrapper should use the same encoded poset as explicit encoding.
    enc1 = DF.encode_pmodule_from_PL_fringe(F1, enc_pl)
    res_wrap = DF.projective_resolution_Rn(F1, enc_pl, res3; return_encoding=true)
    @test FF.poset_equal(res_wrap.P, enc1.P)
    @test DF.betti_table(res_wrap.res) == DF.betti_table(DF.projective_resolution(enc1.M, res3))

    # Minimal Betti data: obtain it by requesting a checked-minimal resolution.
    res_min = PM.ResolutionOptions(maxlen=3, minimal=true, check=true)

    bt_wrap = DF.betti(DF.projective_resolution_Rn(F1, enc_pl, res_min))
    bt_explicit = DF.betti(DF.projective_resolution(enc1.M, res_min))
    @test bt_wrap == bt_explicit
end




@testset "PL common encoding for multiple fringes" begin

    enc_pl_10k = PM.EncodingOptions(backend=:pl, max_regions=10_000)

    # We can build HPolys even without Polyhedra, but we can only encode if Polyhedra is available.
    if !PLP.HAVE_POLY
        U1 = PLP.PLUpset(PLP.PolyUnion(1, [PLP.make_hpoly([-1.0], [0.0])]))   # x >= 0
        D1 = PLP.PLDownset(PLP.PolyUnion(1, [PLP.make_hpoly([ 1.0], [2.0])])) # x <= 2
        F1 = PLP.PLFringe([U1], [D1], reshape(QQ[1], 1, 1))

        @test_throws ErrorException PLP.encode_from_PL_fringes(F1, F1, enc_pl_10k)
    else
        # Two 1D modules: support [0,2] and support [1,3].
        U1 = PLP.PLUpset(PLP.PolyUnion(1, [PLP.make_hpoly([-1.0], [ 0.0])]))  # x >= 0
        D1 = PLP.PLDownset(PLP.PolyUnion(1, [PLP.make_hpoly([ 1.0], [ 2.0])]))# x <= 2

        U2 = PLP.PLUpset(PLP.PolyUnion(1, [PLP.make_hpoly([-1.0], [-1.0])]))  # x >= 1
        D2 = PLP.PLDownset(PLP.PolyUnion(1, [PLP.make_hpoly([ 1.0], [ 3.0])]))# x <= 3

        F1 = PLP.PLFringe([U1], [D1], reshape(QQ[1], 1, 1))
        F2 = PLP.PLFringe([U2], [D2], reshape(QQ[1], 1, 1))

        P, Hs, pi = PLP.encode_from_PL_fringes(F1, F2, enc_pl_10k)

        @test length(Hs) == 2
        H1, H2 = Hs[1], Hs[2]

        # Helper: insist locate returns a valid region.
        function fd(H, x)
            q = PLP.locate(pi, x)
            @test q != 0
            return FF.fiber_dimension(H, q)
        end

        @test fd(H1, [-1.0]) == 0
        @test fd(H2, [-1.0]) == 0

        @test fd(H1, [0.5]) == 1
        @test fd(H2, [0.5]) == 0

        @test fd(H1, [1.5]) == 1
        @test fd(H2, [1.5]) == 1

        @test fd(H1, [2.5]) == 0
        @test fd(H2, [2.5]) == 1

        @test fd(H1, [4.0]) == 0
        @test fd(H2, [4.0]) == 0
    end
end


@testset "PLPolyhedra optional backend" begin
    if PLP.HAVE_POLY
        # Unit square: 0 <= x <= 1, 0 <= y <= 1
        A = QQ[QQ(1)  QQ(0);
               QQ(0)  QQ(1);
               QQ(-1) QQ(0);
               QQ(0)  QQ(-1)]
        b = QQ[QQ(1), QQ(1), QQ(0), QQ(0)]
        h = PLP.make_hpoly(A, b)

        @test PLP._in_hpoly(h, [0, 0]) == true
        @test PLP._in_hpoly(h, [1, 1]) == true
        @test PLP._in_hpoly(h, [2, 0]) == false
        @test PLP._in_hpoly(h, [-1, 0]) == false
    else
        # Even without Polyhedra/CDDLib, make_hpoly should still build an HPoly
        # that supports exact membership tests via its stored A*x <= b data.
        A = reshape(QQ[QQ(1)], 1, 1)
        b = QQ[QQ(1)]
        h = PLP.make_hpoly(A, b)
        @test h.poly === nothing
        @test PLP._in_hpoly(h, [0]) == true
        @test PLP._in_hpoly(h, [1]) == true
        @test PLP._in_hpoly(h, [2]) == false
    end
end

@testset "PLBackend axis encoding (axis-aligned boxes)" begin
    enc_axis = PM.EncodingOptions(backend=:pl_backend)
    enc_axis_small = PM.EncodingOptions(backend=:pl_backend, max_regions=2)

    # One-dimensional example:
    # Birth upset:  x >= 0
    # Death downset: x <= 2
    # phi = [1], so the represented module should be supported exactly in the middle region.
    Ups = [PLB.BoxUpset([0.0])]
    Downs = [PLB.BoxDownset([2.0])]
    Phi = reshape(QQ[1], 1, 1)

    # Error guard: if max_regions is too small, the backend should refuse.
    @test_throws ErrorException PLB.encode_fringe_boxes(Ups, Downs, Phi, enc_axis_small)

    P, H, pi = PLB.encode_fringe_boxes(Ups, Downs, Phi, enc_axis)

    @test P.n == 3
    @test pi.coords[1] == [0.0, 2.0]

    # The encoded poset should be a 3-element chain.
    @test Set(FF.cover_edges(P)) == Set([(1, 2), (2, 3)])

    # Locate regions by sampling points in each cell and check fiber dimensions.
    @test FF.fiber_dimension(H, PLB.locate(pi, [-1.0])) == 0   # left of 0: outside upset
    @test FF.fiber_dimension(H, PLB.locate(pi, [ 1.0])) == 1   # between: in upset and downset
    @test FF.fiber_dimension(H, PLB.locate(pi, [ 3.0])) == 0   # right of 2: outside downset

        # Boundary points: should agree with inequality convention (>= ell, <= u).
    @test FF.fiber_dimension(H, PLB.locate(pi, [0.0])) == 1
    @test FF.fiber_dimension(H, PLB.locate(pi, [2.0])) == 1

    @testset "locate caches (O(1) point location)" begin
        # Dense cell map should be present for encoded maps.
        @test !isempty(pi.cell_to_region)
        @test length(pi.cell_to_region) == prod(map(c -> length(c) + 1, pi.coords))

        # Signature Dict should be present and cover every region.
        @test length(pi.sig_to_region) == length(pi.sig_y)

        # For each 1D cell, locate(rep) should equal the cached value.
        ci = pi.coords[1]
        reps = [ci[1] - 1.0, (ci[1] + ci[2]) / 2, ci[end] + 1.0]
        for (k, x) in enumerate(reps)
            @test PLB.locate(pi, [x]) == pi.cell_to_region[k]
        end

        # Warm-up compile, then ensure tuple input is allocation-free.
        # Use the generic exported `locate` via the constant module `PM` so the
        # call is fully inferred (otherwise a module-valued `PLB` can force
        # dynamic dispatch and show spurious allocations).
        PM.locate(pi, (1.0,))
        @test (@allocated PM.locate(pi, (1.0,))) == 0

        # Canonical reconstruction: encode_fringe_boxes is the supported constructor.
        _, _, pi2 = PLB.encode_fringe_boxes(Ups, Downs, Phi, enc_axis)
        @test PLB.locate(pi2, [1.0]) == PLB.locate(pi, [1.0])
        @test PLB.locate(pi2, [0.0]) == PLB.locate(pi, [0.0])
        @test PLB.locate(pi2, [2.0]) == PLB.locate(pi, [2.0])

        # lock in the removal of the old 7-arg constructor.
        @test_throws MethodError PLB.PLEncodingMapBoxes(pi.n, pi.coords, pi.sig_y, pi.sig_z, pi.reps, pi.Ups, pi.Downs)
    end

    @testset "locate caches in 2D" begin
        Ups2 = [PLB.BoxUpset([0.0, 0.0])]
        Downs2 = [PLB.BoxDownset([2.0, 2.0])]
        Phi2 = reshape(QQ[1], 1, 1)

        P2, H2, pi2 = PLB.encode_fringe_boxes(Ups2, Downs2, Phi2, enc_axis)

        @test !isempty(pi2.cell_to_region)
        @test length(pi2.cell_to_region) == (length(pi2.coords[1]) + 1) * (length(pi2.coords[2]) + 1)

        r_in  = PLB.locate(pi2, [ 1.0,  1.0])  # in upset and in downset
        r_q   = PLB.locate(pi2, [ 3.0,  1.0])  # in upset, outside downset
        r_d   = PLB.locate(pi2, [-1.0,  1.0])  # in downset, outside upset
        r_out = PLB.locate(pi2, [-1.0,  3.0])  # outside both

        @test length(Set([r_in, r_q, r_d, r_out])) == 4

        # Boundary checks: x=2 and x=0 are inclusive as expected.
        @test PLB.locate(pi2, [2.0, 1.0]) == r_in   # on u boundary
        @test PLB.locate(pi2, [0.0, 1.0]) == r_in   # on ell boundary

        # Tuple input: allocation-free after warm-up.
        # Same note as in 1D: use the exported `locate` via `PM` to avoid
        # dynamic module-property dispatch.
        PM.locate(pi2, (1.0, 1.0))
        @test (@allocated PM.locate(pi2, (1.0, 1.0))) == 0
    end

end
