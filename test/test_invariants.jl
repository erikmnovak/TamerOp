using Test
import Base.Threads

@testset "PLikeEncodingMap dispatch hook" begin
    @test PM.ZnEncoding.ZnEncodingMap <: PM.CoreModules.PLikeEncodingMap
    @test PM.PLPolyhedra.PLEncodingMap    <: PM.CoreModules.PLikeEncodingMap

    # PLBackend is optional at load time; only test if present.
    if isdefined(PosetModules, :PLBackend)
        @test PosetModules.PLBackend.PLEncodingMapBoxes <: PM.CoreModules.PLikeEncodingMap
    end
end

@testset "Finite-encoding invariants: rank and restricted Hilbert" begin
    P = chain_poset(3)

    # Interval module supported on {2,3} for the chain 1 < 2 < 3.
    H23 = one_by_one_fringe(P, FF.principal_upset(P, 2), FF.principal_downset(P, 3), QQ(1))
    M23 = IR.pmodule_from_fringe(H23)

    @test PM.rank_map(M23, 2, 3) == 1
    @test PM.rank_map(M23, 1, 1) == 0
    @test PM.rank_map(M23, 1, 2) == 0

    rinv = PM.rank_invariant(M23)
    @test length(rinv) == 3
    @test rinv[(2, 2)] == 1
    @test rinv[(2, 3)] == 1
    @test rinv[(3, 3)] == 1
    @test !haskey(rinv, (1, 1))

    rinv_all = PM.rank_invariant(M23; store_zeros=true)
    @test length(rinv_all) == 6
    @test rinv_all[(1, 1)] == 0

    # Noncomparable pair should error.
    Pd = diamond_poset()
    Hd2 = one_by_one_fringe(Pd, FF.principal_upset(Pd, 2), FF.principal_downset(Pd, 2))
    Md2 = IR.pmodule_from_fringe(Hd2)
    @test_throws ErrorException PM.rank_map(Md2, 2, 3)

    if Threads.nthreads() > 1
        rinv_serial = PM.rank_invariant(M23; threads = false)
        rinv_thread = PM.rank_invariant(M23; threads = true)
        @test rinv_thread == rinv_serial

        rinv_all_serial = PM.rank_invariant(M23; store_zeros = true, threads = false)
        rinv_all_thread = PM.rank_invariant(M23; store_zeros = true, threads = true)
        @test rinv_all_thread == rinv_all_serial
    end
    
end


@testset "Hilbert distances" begin
    P = chain_poset(3)

    H23 = one_by_one_fringe(P, FF.principal_upset(P, 2), FF.principal_downset(P, 3))
    M23 = IR.pmodule_from_fringe(H23)

    H3 = one_by_one_fringe(P, FF.principal_upset(P, 3), FF.principal_downset(P, 3))
    M3 = IR.pmodule_from_fringe(H3)

    @test PM.restricted_hilbert(M23) == [0, 1, 1]
    @test PM.restricted_hilbert(M3) == [0, 0, 1]

    @test PM.hilbert_distance(M23, M3; norm=:L1) == 1
    @test PM.hilbert_distance(M23, M3; norm=:Linf) == 1
    @test isapprox(PM.hilbert_distance(M23, M3; norm=:L2), 1.0)

    w = [2, 3, 5]
    @test PM.hilbert_distance(M23, M3; norm=:L1, weights=w) == 3
    @test PM.hilbert_distance(M23, M3; norm=:Linf, weights=w) == 3
end

@testset "Euler characteristic surface and Euler signed-measure" begin
    P = chain_poset(3)

    # Interval module supported on {2,3} for chain 1<2<3.
    H23 = one_by_one_fringe(P, FF.principal_upset(P, 2), FF.principal_downset(P, 3), QQ(1))
    M23 = IR.pmodule_from_fringe(H23)

    # Module supported on {3}.
    H3  = one_by_one_fringe(P, FF.principal_upset(P, 3), FF.principal_downset(P, 3), QQ(1))
    M3  = IR.pmodule_from_fringe(H3)

    # Toy encoding: locate returns the coordinate itself as the poset element.
    struct ToyPi <: PM.CoreModules.PLikeEncodingMap end
    PM.dimension(::ToyPi) = 1
    function PM.locate(::ToyPi, x::AbstractVector)
        length(x) == 1 || error("ToyPi expects a 1D point")
        return round(Int, float(x[1]))
    end

    pi = ToyPi()
    axes = ([1,2,3],)
    opts_axes = PM.InvariantOptions(axes=axes, axes_policy=:as_given)

    # Euler surface of a module should equal its restricted Hilbert values on this encoding.
    surf23 = PM.euler_surface(M23, pi, opts_axes)
    @test surf23 == reshape([0,1,1], 3)

    if Threads.nthreads() > 1
        opts_serial = PM.InvariantOptions(axes=axes, axes_policy=:as_given, threads=false)
        opts_thread = PM.InvariantOptions(axes=axes, axes_policy=:as_given, threads=true)
        surf_serial = PM.euler_surface(M23, pi, opts_serial)
        surf_thread = PM.euler_surface(M23, pi, opts_thread)
        @test surf_thread == surf_serial
    end

    # Mobius inversion in 1D is discrete derivative: weights [0,1,0].
    pm23   = PM.euler_signed_measure(M23, pi, opts_axes)
    @test length(pm23) == 1
    @test pm23.inds[1] == (2,)
    @test pm23.wts[1]  == 1

    # Reconstruction from point measure recovers surface.
    rec23 = PM.surface_from_point_signed_measure(pm23)
    @test rec23 == surf23

    # Test raw point_signed_measure on a known surface.
    surf = reshape([0,1,1], 3)
    pm = PM.point_signed_measure(surf, axes; drop_zeros=true)
    @test length(pm) == 1
    @test pm.inds[1] == (2,)
    @test pm.wts[1] == 1
    @test PM.surface_from_point_signed_measure(pm) == surf

    # Euler distance between M23 and M3 on this grid.
    surf3 = PM.euler_surface(M3, pi, opts_axes)
    @test surf3 == reshape([0,0,1], 3)
    @test PM.euler_distance(surf23, surf3; ord=1) == 1
    @test PM.euler_distance(M23, M3, pi, opts_axes; ord=Inf) == 1.0

    # Euler for a 2-term cochain complex: chi = dim(C^0) - dim(C^1) (tmin=0)
    C = PM.ModuleCochainComplex([M23, M3], [PM.zero_morphism(M23, M3)]; tmin=0)
    surfC = PM.euler_surface(C, pi, opts_axes)
    # reuse the opts already defined earlier in this testset
    @test surfC == reshape([0,1,0], 3)

    pmC = PM.euler_signed_measure(C, pi, opts_axes)
    # Mobius derivative of [0,1,0] is [0,1,-1] (drop_zeros keeps 2 points)
    @test length(pmC) == 2
    @test pmC.inds[1] == (2,)
    @test pmC.wts[1]  == 1
    @test pmC.inds[2] == (3,)
    @test pmC.wts[2]  == -1
    @test PM.surface_from_point_signed_measure(pmC) == surfC

    # mma_decomposition integration: generic Euler-only front-end.
    outM = PM.mma_decomposition(M23, pi, opts_axes; method=:euler)
    @test outM.euler_surface == surf23
    @test PM.surface_from_point_signed_measure(outM.euler_signed_measure) == surf23

    outC = PM.mma_decomposition(C, pi, opts_axes; method=:euler)
    @test outC.euler_surface == surfC
    @test PM.surface_from_point_signed_measure(outC.euler_signed_measure) == surfC

    # The generic signature only supports method=:euler.
    @test_throws ArgumentError PM.mma_decomposition(M23, pi, opts_axes; method=:all)
end


@testset "Slice restrictions and 1D barcodes" begin
    P = chain_poset(3)

    H23 = one_by_one_fringe(P, FF.principal_upset(P, 2), FF.principal_downset(P, 3))
    M23 = IR.pmodule_from_fringe(H23)

    H3 = one_by_one_fringe(P, FF.principal_upset(P, 3), FF.principal_downset(P, 3))
    M3 = IR.pmodule_from_fringe(H3)

    chain = [1, 2, 3]

    b23 = PM.slice_barcode(M23, chain)
    @test b23 == Dict((2, 4) => 1)

    b3 = PM.slice_barcode(M3, chain)
    @test b3 == Dict((3, 4) => 1)

    b0 = Dict{Tuple{Int,Int},Int}()

    @test PM.bottleneck_distance(b23, b23) == 0.0
    @test PM.bottleneck_distance(b23, b3) == 1.0
    @test isapprox(PM.bottleneck_distance(b3, b0), 0.5)

    # Approx matching distance over a single slice is just the bottleneck distance.
    @test PM.matching_distance_approx(M23, M3, [chain]) == 1.0

    # Approx matching distance over a single slice is just the bottleneck distance.
    @test PM.matching_distance_approx(M23, M3, [chain]) == 1.0

    # Restriction to a chain should produce a 1D module with the expected dims.
    Mc = PM.restrict_to_chain(M23, chain)
    @test Mc.dims == [0, 1, 1]
    @test PM.rank_map(Mc, 2, 3) == 1

end


@testset "slice_chain wrapper for ZnEncodingMap" begin
    n = 1
    b = [0]
    c = [5]
    I = FZ.Face(n, Int[])

    flats = [FZ.IndFlat(I, b; id=:F)]
    injectives = [FZ.IndInj(I, c; id=:E)]

    Phi = spzeros(QQ, 1, 1)
    Phi[1, 1] = 1

    FG = FZ.Flange{QQ}(n, flats, injectives, Phi)

    enc = PM.EncodingOptions(backend=:zn, max_regions=1000)
    Penc, Henc, pi = PM.encode_from_flange(FG, enc)

    @test Penc.n == 3

    # `ZnEncodingMap`s are defined on the integer lattice Z^n, but generic
    # helpers may represent an integer lattice point as a float (e.g. `2.0`).
    # Such points should be accepted if they are integer-valued.
    @test PM.locate(pi, [-2.0]) == PM.locate(pi, [-2])

    opts = PM.InvariantOptions()
    chain, tvals = PM.slice_chain(pi, [-2], [1], opts; kmin=0, kmax=9)
    @test chain == [1, 2, 3]
    @test tvals == [0, 2, 8]

    Menc = IR.pmodule_from_fringe(Henc)
    bM = PM.slice_barcode(Menc, chain)
    @test bM == Dict((2, 3) => 1)
end



@testset "Wrappers using locate(pi, x)" begin
    # A tiny toy encoding map: locate(pi, x) = x.
    struct ToyPi end
    function PM.locate(::ToyPi, x::AbstractVector)
        length(x) == 1 || error("ToyPi expects a 1D point")
        return round(Int, float(x[1]))
    end

    P = chain_poset(3)
    H23 = one_by_one_fringe(P, FF.principal_upset(P, 2), FF.principal_downset(P, 3))
    M23 = IR.pmodule_from_fringe(H23)

    pi = ToyPi()
    @test PM.rank_map(M23, pi, [2], [3]) == 1
    @test PM.restricted_hilbert(M23, pi, [2]) == 1
end


@testset "Betti/Bass tables from indicator resolutions" begin
    # Use the diamond poset because it supports nontrivial length-2 resolutions.
    P = diamond_poset()
    S1 = one_by_one_fringe(P, FF.principal_upset(P, 1), FF.principal_downset(P, 1))
    M1 = IR.pmodule_from_fringe(S1)

    # Upset indicator resolution
    F, _dF = IR.upset_resolution(M1; maxlen=2)
    resP = PM.projective_resolution(M1, PM.ResolutionOptions(maxlen=2))

    @test PM.betti(F) == PM.betti(resP)
    @test PM.betti_table(F) == PM.betti_table(resP)

    # Downset indicator resolution
    E, _dE = IR.downset_resolution(M1; maxlen=2)
    resI = PM.injective_resolution(M1, PM.ResolutionOptions(maxlen=2))

    @test PM.bass(E) == PM.bass(resI)
    @test PM.bass_table(E) == PM.bass_table(resI)
end


@testset "region_weights for ZnEncodingMap (1D sanity check)" begin
    # Reproduce the 1D Z-encoding test and additionally check region size weights.
    n = 1
    b = [0]
    c = [5]
    I = FZ.Face(n, Int[])

    flats = [FZ.IndFlat(I, b; id=:F)]
    injectives = [FZ.IndInj(I, c; id=:E)]

    Phi = spzeros(QQ, 1, 1)
    Phi[1, 1] = 1

    FG = FZ.Flange{QQ}(n, flats, injectives, Phi)

    enc = PM.EncodingOptions(backend=:zn, max_regions=1000)
    Penc, _Henc, pi = PM.encode_from_flange(FG, enc)

    @test Penc.n == 3

    # Box [b-2, c+2] = [-2, 7].
    w = PM.region_weights(pi; box=([b[1] - 2], [c[1] + 2]))
    @test w == [2, c[1] - b[1] + 1, 2]

        # mma_decomposition integration: PModule + ZnEncodingMap signature.
    # This should work out-of-the-box for method=:euler because Euler surfaces can be
    # evaluated on the encoding grid (axes_policy=:encoding).
    Menc = IR.pmodule_from_fringe(_Henc)
    opts_enc = PM.InvariantOptions(axes_policy=:encoding)
    outE = PM.mma_decomposition(Menc, pi, opts_enc; method=:euler)

    surfE = PM.euler_surface(Menc, pi, opts_enc)
    @test outE.euler_surface == surfE
    @test PM.surface_from_point_signed_measure(outE.euler_signed_measure) == surfE

    # method=:all combines rectangles + slices + Euler. For slices we must provide
    # directions and offsets; we keep this tiny so the test stays cheap.
    outAll = PM.mma_decomposition(
        Menc,
        pi;
        method=:all,
        slice_kwargs=(
            directions=[[1.0]],
            offsets=[[0.0]],
            tmin=-2.0,
            tmax=7.0,
            nsteps=11,
            strict=false,
            drop_unknown=true,
            dedup=true,
        ),
    )

    @test hasproperty(outAll, :rectangles)
    @test hasproperty(outAll, :slices)
    @test hasproperty(outAll, :euler_surface)
    @test hasproperty(outAll, :euler_signed_measure)
    @test PM.surface_from_point_signed_measure(outAll.euler_signed_measure) == outAll.euler_surface
    @test hasproperty(outAll.slices, :barcodes)
    @test hasproperty(outAll.slices, :weights)

end


@testset "Extra slice + bottleneck tests (optional edge cases)" begin

    @testset "Non-chain input rejection" begin
        P = diamond_poset()
        H = one_by_one_fringe(P, FF.principal_upset(P, 2), FF.principal_downset(P, 2))
        M = IR.pmodule_from_fringe(H)

        # 2 and 3 are incomparable in the diamond poset.
        chain_bad = [2, 3]
        @test_throws ErrorException PM.slice_barcode(M, chain_bad)
        @test_throws ErrorException PM.restrict_to_chain(M, chain_bad)
    end

    @testset "Geometric slicing wrapper for a toy locate" begin
        # Toy encoding map for R^2: regions are determined by (floor(x1), floor(x2)).
        # Along the diagonal line x(t) = (t,t), this yields a strictly increasing chain.
        struct ToyPi2D <: PM.CoreModules.PLikeEncodingMap end
        PM.dimension(::ToyPi2D) = 2
        function PM.locate(::ToyPi2D, x::AbstractVector)
            length(x) == 2 || error("ToyPi2D expects a 2D point")
            # Add a tiny epsilon to avoid roundoff landing just below an integer.
            eps = 1e-9
            i = floor(Int, float(x[1]) + eps)
            j = floor(Int, float(x[2]) + eps)
            return 1 + i + 10 * j
        end

        pi = ToyPi2D()
        chain, tvals = PM.slice_chain(pi, [0.0, 0.0], [1.0, 1.0], PM.InvariantOptions(strict=true);
                                      tmin=0.0, tmax=3.0, nsteps=301,
                                      drop_unknown=true, dedup=true)

        @test chain == [1, 12, 23, 34]
        @test length(chain) == length(tvals)
        @test all(diff(chain) .> 0)               # monotone and de-duplicated
        @test all(diff(tvals) .> 0)               # parameter values strictly increasing
    end

    @testset "slice_chain clips to box window" begin
        PLB = PM.PLBackend

        Ups = [PLB.BoxUpset([0.0, 0.0], [1.0, 1.0])]
        Downs = [PLB.BoxDownset([2.0, 2.0], [3.0, 3.0])]
        _P, _H, pi = PLB.encode_fringe_boxes(Ups, Downs)

        x0 = [0.5, 0.5]
        dir = [1.0, 0.0]

        # Without clipping: the grid quickly leaves the encoded region set, so strict=true should throw.
        @test_throws ErrorException PM.slice_chain(pi, x0, dir, PM.InvariantOptions(strict=true); 
                                                   tmin=0.0, tmax=100.0, nsteps=3)

        opts_strict = PM.InvariantOptions(strict=true, box = box)
        # Clip to an interior window. Along x(t) = (0.5+t, 0.5), staying in x in [0.2,0.8]
        # means t in [-0.3, 0.3], and with tmin=0 we expect max t <= 0.3.
        box = ([0.2, 0.2], [0.8, 0.8])
        chain, tvals = PM.slice_chain(pi, x0, dir, opts_strict;
                                    tmin=0.0, tmax=100.0, nsteps=11,
                                    drop_unknown=true, dedup=true)
        @test !isempty(chain)
        @test maximum(tvals) <= 0.3 + 1e-12

        # With box2, use intersection. Shrink x-high from 0.8 to 0.6 => tmax from 0.3 to 0.1.
        box2 = ([0.2, 0.2], [0.6, 0.8])
        chain2, tvals2 = PM.slice_chain(pi, x0, dir, opts_strict;
                                        tmin=0.0, tmax=100.0, nsteps=11,
                                        drop_unknown=true, dedup=true, box2=box2)
        @test !isempty(chain2)
        @test maximum(tvals2) <= 0.1 + 1e-12

        # Explicit ts: should filter to those integers in [0, 0.1], i.e. just 0.
        chain3, tvals3 = PM.slice_chain(pi, x0, dir, opts_strict;
                                        ts=0:10,
                                        drop_unknown=true, dedup=true, box2=box2)
        @test tvals3 == [0]
        @test length(chain3) == 1
        
        # Semantic default:
        # If a window is present and the caller omits tmin/tmax, slice_chain should
        # sample the full line-window intersection interval (not [0,1] intersected).
        chain4, tvals4 = PM.slice_chain(pi, x0, dir, opts_strict;
                                        nsteps=11,
                                        drop_unknown=true, dedup=false)
        @test length(tvals4) == 11
        @test isapprox(first(tvals4), -0.3; atol=1e-12)
        @test isapprox(last(tvals4), 0.3; atol=1e-12)

        # With two windows, the effective window is their intersection.
        chain5, tvals5 = PM.slice_chain(pi, x0, dir, opts_strict;
                                        nsteps=11,
                                        drop_unknown=true, dedup=false, box2=box2)
        @test length(tvals5) == 11
        @test isapprox(first(tvals5), -0.3; atol=1e-12)
        @test isapprox(last(tvals5), 0.1; atol=1e-12)

    end


    @testset "Bottleneck triangle inequality sanity check" begin
        # A has one interval; B and C add a short interval that changes length.
        A = Dict((0.0, 2.0) => 1)
        B = Dict((0.0, 2.0) => 1, (3.0, 3.2) => 1)
        C = Dict((0.0, 2.0) => 1, (3.0, 3.4) => 1)

        dAB = PM.bottleneck_distance(A, B)
        dBC = PM.bottleneck_distance(B, C)
        dAC = PM.bottleneck_distance(A, C)

        @test isapprox(dAB, 0.1; atol=1e-12)
        @test isapprox(dBC, 0.2; atol=1e-12)
        @test isapprox(dAC, 0.2; atol=1e-12)

        # Triangle inequality (allow a tiny floating tolerance).
        @test dAC <= dAB + dBC + 1e-12
    end
end


@testset "sample_directions_2d helper" begin
    dirs = PM.sample_directions_2d(max_den=3)
    @test !isempty(dirs)

    # All directions are 2D, strictly positive, and L1-normalized.
    @test all(length(d) == 2 for d in dirs)
    @test all(d[1] > 0 && d[2] > 0 for d in dirs)
    @test all(isapprox(d[1] + d[2], 1.0; atol=1e-12) for d in dirs)

    # The diagonal direction (1,1) must be present.
    @test any(d -> isapprox(d[1], 0.5; atol=1e-12) && isapprox(d[2], 0.5; atol=1e-12), dirs)

    # Integer directions are useful for Z^2 encodings.
    dirsZ = PM.sample_directions_2d(max_den=3; normalize=:none)
    @test any(d -> d == [1, 1], dirsZ)
end

@testset "Multiparameter persistence landscapes" begin
    P = chain_poset(3)

    # Module supported on 2 -> 3 (so barcode is (1,3) in the slice values below).
    H23 = one_by_one_fringe(P, FF.principal_upset(P, 2), FF.principal_downset(P, 3))
    M23 = IR.pmodule_from_fringe(H23)

    # Module supported only at 3 (barcode (2,3) in the slice values below).
    H3 = one_by_one_fringe(P, FF.principal_upset(P, 3), FF.principal_downset(P, 3))
    M3 = IR.pmodule_from_fringe(H3)

    # A simple ground-truth barcode and its 1D persistence landscape.
    bar = Dict((1.0, 3.0) => 1)
    tgrid = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    pl = PM.persistence_landscape(bar; kmax=3, tgrid=tgrid)

    # lambda_1 is a tent with peak 1 at t=2.
    @test isapprox(pl.values[1, 1], 0.0; atol=1e-12)
    @test isapprox(pl.values[1, 3], 0.0; atol=1e-12)  # t = 1.0 (birth)
    @test isapprox(pl.values[1, 4], 0.5; atol=1e-12)  # t = 1.5
    @test isapprox(pl.values[1, 5], 1.0; atol=1e-12)  # t = 2.0 (midpoint)
    @test isapprox(pl.values[1, 6], 0.5; atol=1e-12)  # t = 2.5
    @test isapprox(pl.values[1, 7], 0.0; atol=1e-12)  # t = 3.0 (death)

    # Higher layers vanish for a single interval barcode.
    @test maximum(abs.(pl.values[2:end, :])) == 0.0

    # Build a multiparameter landscape from an explicit chain slice.
    slice = (chain=[1, 2, 3], values=[0.0, 1.0, 2.0], weight=1.0)
    L23 = PM.mp_landscape(M23, [slice]; kmax=2, tgrid=tgrid)
    L3  = PM.mp_landscape(M3,  [slice]; kmax=2, tgrid=tgrid)

    @test size(L23.values) == (1, 1, 2, length(tgrid))
    @test isapprox(sum(L23.weights), 1.0; atol=1e-12)

    # The first layer for M23 should match the tent coming from (1,3).
    @test isapprox(L23.values[1, 1, 1, 5], 1.0; atol=1e-12)
    @test isapprox(L23.values[1, 1, 1, 4], 0.5; atol=1e-12)
    @test isapprox(L23.values[1, 1, 1, 6], 0.5; atol=1e-12)
    @test isapprox(L23.values[1, 1, 2, 5], 0.0; atol=1e-12)

    # Distance and kernel sanity checks.
    d_same = PM.mp_landscape_distance(L23, L23; p=2)
    @test isapprox(d_same, 0.0; atol=1e-12)

    d_23_3 = PM.mp_landscape_distance(L23, L3; p=2)
    d_3_23 = PM.mp_landscape_distance(L3, L23; p=2)
    @test d_23_3 >= 0.0
    @test isapprox(d_23_3, d_3_23; atol=1e-12)
    @test d_23_3 > 0.0

    k_same = PM.mp_landscape_kernel(L23, L23; kind=:gaussian, sigma=1.0)
    k_diff = PM.mp_landscape_kernel(L23, L3; kind=:gaussian, sigma=1.0)
    @test isapprox(k_same, 1.0; atol=1e-12)
    @test k_diff < 1.0

    # Convenience wrappers: construct landscapes internally.
    d_wrap = PM.mp_landscape_distance(M23, M3, [slice]; p=2, kmax=2, tgrid=tgrid)
    @test isapprox(d_wrap, d_23_3; atol=1e-12)

    k_wrap = PM.mp_landscape_kernel(M23, M3, [slice]; kind=:gaussian, sigma=1.0, p=2, kmax=2, tgrid=tgrid)
    @test isapprox(k_wrap, k_diff; atol=1e-12)

    # Geometric slicing wrapper sanity check with a toy locate() on R^1.
    struct ToyPi1D end
    function PM.locate(::ToyPi1D, x::AbstractVector)
        length(x) == 1 || error("ToyPi1D expects a 1D point")
        t = float(x[1])
        if t < 1.0
            return 1
        elseif t < 2.0
            return 2
        elseif t < 3.0
            return 3
        else
            return 0
        end
    end

    pi = ToyPi1D()
    dirs = [[1.0]]
    offs = [[0.0]]

    L23_geo = PM.mp_landscape(M23, pi;
                              directions=dirs,
                              offsets=offs,
                              kmax=2,
                              tgrid=tgrid,
                              tmin=0.0,
                              tmax=3.0,
                              nsteps=301,
                              strict=false,
                              drop_unknown=true,
                              dedup=true)

    @test isapprox(L23_geo.values[1, 1, 1, 5], 1.0; atol=1e-12)

    # -------------------------------------------------------------------------
    # show(::MPLandscape) should give a compact, informative summary.
    # -------------------------------------------------------------------------

    s = sprint(show, L23)
    @test occursin("MPLandscape(", s)
    @test occursin("ndirs=1", s)
    @test occursin("noffsets=1", s)
    @test occursin("kmax=2", s)
    @test occursin("nt=$(length(tgrid))", s)
    @test occursin("weights_normalized=true", s)

    # If we intentionally skip normalization and use a non-unit weight,
    # the summary should report "weights_normalized=false".
    slice2 = (chain=[1, 2, 3], values=[0.0, 1.0, 2.0], weight=2.0)
    L_un = PM.mp_landscape(M23, [slice2]; kmax=2, tgrid=tgrid, normalize_weights=false)
    s_un = sprint(show, L_un)
    @test occursin("weights_normalized=false", s_un)

    # The text/plain show should include the same core metadata in a readable form.
    s_plain = sprint(show, MIME("text/plain"), L23)
    @test occursin("MPLandscape", s_plain)
    @test occursin("ndirs = 1", s_plain)
    @test occursin("noffsets = 1", s_plain)
    @test occursin("kmax = 2", s_plain)
    @test occursin("weights_normalized = true", s_plain)

end

@testset "Slice vectorizations and sliced kernels" begin
    # ---------------------------------------------------------------------
    # Barcode-level vectorizations.
    # ---------------------------------------------------------------------
    bar = Dict((1.0, 3.0) => 1)
    tgrid = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    pl = PM.persistence_landscape(bar; kmax=1, tgrid=tgrid)
    sil = PM.persistence_silhouette(bar; tgrid=tgrid, weighting=:persistence, p=1, normalize=true)

    # For a single interval, silhouette (with normalize=true) equals the tent itself,
    # i.e. the first landscape layer.
    @test maximum(abs.(sil .- pl.values[1, :])) < 1e-12

    img = PM.persistence_image(bar;
                               xgrid=[0.0, 1.0, 2.0, 3.0],
                               ygrid=[0.0, 1.0, 2.0],
                               sigma=0.2,
                               weighting=:none,
                               normalize=:none)
    @test size(img.values) == (3, 4)
    ind = argmax(img.values)
    iy, ix = Tuple(CartesianIndices(img.values)[ind])
    @test ix == 2   # birth = 1.0 (xgrid[2])
    @test iy == 3   # pers  = 2.0 (ygrid[3])

    imgN = PM.persistence_image(bar;
                                xgrid=[0.0, 1.0, 2.0, 3.0],
                                ygrid=[0.0, 1.0, 2.0],
                                sigma=0.2,
                                weighting=:none,
                                normalize=:max)
    @test isapprox(maximum(imgN.values), 1.0; atol=1e-12)

    bar2 = Dict((0.0, 2.0) => 1, (0.0, 1.0) => 1)
    H = PM.barcode_entropy(bar2; normalize=false)
    expected = -(2 / 3 * log(2 / 3) + 1 / 3 * log(1 / 3))
    @test isapprox(H, expected; atol=1e-12)

    Hn = PM.barcode_entropy(bar2; normalize=true)
    @test isapprox(Hn, H / log(2); atol=1e-12)

    summ = PM.barcode_summary(bar2; normalize_entropy=false)
    @test summ.n_intervals == 2
    @test isapprox(summ.total_persistence, 3.0; atol=1e-12)
    @test isapprox(summ.max_persistence, 2.0; atol=1e-12)
    @test isapprox(summ.mean_persistence, 1.5; atol=1e-12)

    # ---------------------------------------------------------------------
    # Module-level slice_features and slice_kernel.
    # Build a small direct sum module with two intervals along a chain slice.
    # ---------------------------------------------------------------------
    P = chain_poset(3)
    U1 = FF.principal_upset(P, 2)
    D1 = FF.principal_downset(P, 3)
    U2 = FF.principal_upset(P, 3)
    D2 = FF.principal_downset(P, 3)

    Phi = spzeros(QQ, 2, 2)
    Phi[1, 1] = 1
    Phi[2, 2] = 1

    Hsum = FF.FringeModule{QQ}(P, [U1, U2], [D1, D2], Phi)
    Msum = IR.pmodule_from_fringe(Hsum)

    slice = (chain=[1, 2, 3], values=[0.0, 1.0, 2.0], weight=1.0)
    bc_sum = PM.slice_barcode(Msum, slice.chain; values=slice.values)

    @test bc_sum == Dict((1.0, 3.0) => 1, (2.0, 3.0) => 1)

    # slice_features uses normalized persistent entropy by default (entropy_normalize=true),
    # so it should match Hn computed above.
    ent_norm = PM.slice_features(Msum, [slice]; featurizer=:entropy, aggregate=:mean)
    @test isapprox(ent_norm, Hn; atol=1e-12)

    # Also test the raw (unnormalized) entropy path explicitly.
    ent_raw = PM.slice_features(Msum, [slice];
                                featurizer=:entropy,
                                aggregate=:mean,
                                entropy_normalize=false)
    @test isapprox(ent_raw, expected; atol=1e-12)

    # Also check landscape features shape and a known value for M23.
    H23 = FF.one_by_one_fringe(P, U1, D1, QQ(1))
    M23 = IR.pmodule_from_fringe(H23)

    f_land = PM.slice_features(M23, [slice];
                               featurizer=:landscape,
                               kmax=2,
                               tgrid=tgrid,
                               aggregate=:mean)
    @test length(f_land) == 2 * length(tgrid)
    # For barcode (1,3) and tgrid above, layer 1 at t=2.0 is 1.0.
    @test isapprox(f_land[5], 1.0; atol=1e-12)

    # Geometric version: compare to explicit slice using a tiny toy encoding map.
    struct ToyPi1D end
    function PM.locate(::ToyPi1D, x::AbstractVector)
        t = x[1]
        if 0.0 <= t < 1.0
            return 1
        elseif 1.0 <= t < 2.0
            return 2
        elseif 2.0 <= t < 3.0
            return 3
        else
            return 0
        end
    end

    pi = ToyPi1D()
    f_geo = PM.slice_features(M23, pi;
                              directions=[[1.0]],
                              offsets=[[0.0]],
                              tmin=0.0,
                              tmax=3.0,
                              nsteps=301,
                              strict=false,
                              drop_unknown=true,
                              dedup=true,
                              featurizer=:landscape,
                              kmax=2,
                              tgrid=tgrid,
                              aggregate=:mean)
    @test isapprox(f_geo[5], f_land[5]; atol=1e-12)

    # Sliced kernels: identical inputs should give kernel value 1 for gaussian kinds.
    k_same = PM.slice_kernel(M23, M23, [slice]; kind=:bottleneck_gaussian, sigma=1.0)
    @test isapprox(k_same, 1.0; atol=1e-12)

    # Symmetry and strict inequality for different modules.
    H3 = FF.one_by_one_fringe(P, U2, D2, QQ(1))
    M3 = IR.pmodule_from_fringe(H3)
    k_diff1 = PM.slice_kernel(M23, M3, [slice]; kind=:bottleneck_gaussian, sigma=1.0)
    k_diff2 = PM.slice_kernel(M3, M23, [slice]; kind=:bottleneck_gaussian, sigma=1.0)
    @test isapprox(k_diff1, k_diff2; atol=1e-12)
    @test k_diff1 < 1.0

    # Landscape kernel (gaussian) on identical inputs is also 1.
    k_land = PM.slice_kernel(M23, M23, [slice];
                             kind=:landscape_gaussian,
                             sigma=1.0,
                             tgrid=tgrid,
                             kmax=2)
    @test isapprox(k_land, 1.0; atol=1e-12)
end

@testset "Signed rectangles / Mobius inversion" begin
    axes = ([0, 1, 2], [0, 1])

    rects = PM.Rect{2}[
        PM.Rect{2}((0, 0), (2, 1)),
        PM.Rect{2}((1, 0), (2, 0)),
        PM.Rect{2}((0, 1), (1, 1)),
    ]
    weights = [2, -1, 3]
    sb_true = PM.RectSignedBarcode{2,Int}(axes, rects, weights)

    function r_idx(p, q)
        x = (axes[1][p[1]], axes[2][p[2]])
        y = (axes[1][q[1]], axes[2][q[2]])
        return PM.rank_from_signed_barcode(sb_true, x, y)
    end

    sb_est = PM.rectangle_signed_barcode(r_idx, axes)

    d_true = Dict(zip(sb_true.rects, sb_true.weights))
    d_est = Dict(zip(sb_est.rects, sb_est.weights))
    @test d_est == d_true

    dims = (length(axes[1]), length(axes[2]))
    for pCI in CartesianIndices(dims)
        p = Tuple(pCI)
        for qCI in CartesianIndices((p[1]:dims[1], p[2]:dims[2]))
            q = Tuple(qCI)
            x = (axes[1][p[1]], axes[2][p[2]])
            y = (axes[1][q[1]], axes[2][q[2]])
            @test PM.rank_from_signed_barcode(sb_est, x, y) == r_idx(p, q)
        end
    end

    sb_trunc = PM.truncate_signed_barcode(sb_est; max_terms=1)
    @test length(sb_trunc) == 1
    @test abs(sb_trunc.weights[1]) == maximum(abs.(sb_est.weights))

    k_lin = PM.rectangle_signed_barcode_kernel(sb_est, sb_est; kind=:linear)
    @test isapprox(k_lin, sum(float(w)^2 for w in sb_est.weights); atol=1e-12)

    img = PM.rectangle_signed_barcode_image(sb_est; sigma=1.0, mode=:center)
    @test size(img) == (length(axes[1]), length(axes[2]))
end

@testset "Wasserstein on barcodes" begin
    bar1 = Dict((0, 2) => 1)
    bar2 = Dict((0, 3) => 1)

    d12 = PM.wasserstein_distance(bar1, bar2; p=1, q=Inf)
    @test isapprox(d12, 1.0; atol=1e-12)

    dempty = PM.wasserstein_distance(Dict{Tuple{Int,Int},Int}(), bar1; p=2, q=Inf)
    @test isapprox(dempty, 1.0; atol=1e-12)

    k = PM.wasserstein_kernel(bar1, bar2; p=1, q=Inf, sigma=1.0, kind=:gaussian)
    @test isapprox(k, exp(-0.5); atol=1e-12)
end

@testset "Rectangle signed barcode speed options" begin
    # -------------------------------------------------------------------------
    # 1) max_span restriction should equal filtering the full result by span.
    # -------------------------------------------------------------------------
    axes = ([0, 1, 2], [0, 1, 2])

    rects_true = PM.Invariants.Rect{2}[
        PM.Invariants.Rect{2}((0, 0), (2, 2)),  # span 2 in both directions
        PM.Invariants.Rect{2}((0, 0), (1, 1)),  # span 1
        PM.Invariants.Rect{2}((1, 0), (2, 1)),  # span 1
    ]
    weights_true = [1, -1, 2]
    sb_true = PM.Invariants.RectSignedBarcode{2,Int}(axes, rects_true, weights_true)

    r_idx(pI, qI) = PM.Invariants.rank_from_signed_barcode(
        sb_true,
        ntuple(k -> axes[k][pI[k]], 2),
        ntuple(k -> axes[k][qI[k]], 2),
    )

    sb_full = PM.rectangle_signed_barcode(r_idx, axes)
    sb_span = PM.rectangle_signed_barcode(r_idx, axes; max_span=(1, 1))

    function sb_to_dict(sb)
        d = Dict{Tuple{NTuple{2,Int},NTuple{2,Int}},Int}()
        for (r, w) in zip(sb.rects, sb.weights)
            d[(r.lo, r.hi)] = w
        end
        return d
    end

    d_full = sb_to_dict(sb_full)
    d_span = sb_to_dict(sb_span)

    d_full_small = Dict{Tuple{NTuple{2,Int},NTuple{2,Int}},Int}()
    for (k, w) in d_full
        lo, hi = k
        if (hi[1] - lo[1] <= 1) && (hi[2] - lo[2] <= 1)
            d_full_small[k] = w
        end
    end

    @test d_span == d_full_small

    # -------------------------------------------------------------------------
    # 1b) Bulk vs local algorithm parity + rank_idx call counts.
    # -------------------------------------------------------------------------
    sb_local = PM.rectangle_signed_barcode(r_idx, axes; method=:local)
    sb_bulk = PM.rectangle_signed_barcode(r_idx, axes; method=:bulk)

    @test Dict(zip(sb_local.rects, sb_local.weights)) ==
          Dict(zip(sb_bulk.rects, sb_bulk.weights))

    # Bulk calls rank_idx exactly once per comparable pair (p <= q).
    dims = map(length, axes)
    n_pairs = prod(div(d * (d + 1), 2) for d in dims)

    c_bulk = Ref(0)
    r_idx_count_bulk(pI, qI) = (c_bulk[] += 1; r_idx(pI, qI))
    _ = PM.rectangle_signed_barcode(r_idx_count_bulk, axes; method=:bulk)
    @test c_bulk[] == n_pairs

    # Local method should make strictly more rank_idx calls on the same problem.
    c_local = Ref(0)
    r_idx_count_local(pI, qI) = (c_local[] += 1; r_idx(pI, qI))
    _ = PM.rectangle_signed_barcode(r_idx_count_local, axes; method=:local)
    @test c_local[] > n_pairs

    # Fast inverse transform: barcode -> dense rank array matches rank_idx on p <= q.
    R = PM.rectangle_signed_barcode_rank(sb_bulk)
    for p1 in 1:dims[1], p2 in 1:dims[2]
        for q1 in p1:dims[1], q2 in p2:dims[2]
            @test R[p1, p2, q1, q2] == r_idx((p1, p2), (q1, q2))
        end
    end


    # -------------------------------------------------------------------------
    # 2) Axis coarsening helper keeps endpoints and respects max_len.
    # -------------------------------------------------------------------------
    ax = collect(0:9)
    axc = PM.Invariants.coarsen_axis(ax; max_len=5)
    @test length(axc) <= 5
    @test first(axc) == 0
    @test last(axc) == 9

    # -------------------------------------------------------------------------
    # 3) Wrapper-level memoization and axes_policy smoke tests (N = 1).
    # -------------------------------------------------------------------------
    R = PM.QQ
    n = 1
    flats = PM.IndFlat[
        PM.IndFlat(PM.face(n, []), [0]),
        PM.IndFlat(PM.face(n, []), [2]),
    ]
    injectives = PM.IndInj[
        PM.IndInj(PM.face(n, []), [1]),
        PM.IndInj(PM.face(n, []), [3]),
    ]
    # Phi must be (#injectives x #flats): rows index injectives, cols index flats.
    # Here we just take the 2x2 identity (a convenient "direct sum" style choice).
    Phi = zeros(R, length(injectives), length(flats))
    Phi[1, 1] = one(R)
    Phi[2, 2] = one(R)
    F = PM.Flange{R}(n, flats, injectives, Phi)

    enc = PM.EncodingOptions(backend=:zn, max_regions=100)
    (Penc, Henc, pi) = PM.encode_from_flange(F, enc)
    Menc = IR.pmodule_from_fringe(Henc)

    axes_user = (collect(-2:6),)

    # Regression test: ZnEncodingMap.coords is axis-wise critical coordinates, so
    # axes_from_encoding(pi) must return an N-tuple where N == pi.n.
    enc_axes = PM.axes_from_encoding(pi)
    @test length(enc_axes) == n
    @test enc_axes[1] == [-1, 0, 2, 4]

    cache = PM.Invariants.RankQueryCache(pi)
    @test typeof(cache).parameters[1] == n

    cache = PM.Invariants.RankQueryCache(pi)

    sb_enc_1 = PM.rectangle_signed_barcode(Menc, pi;
        axes=axes_user,
        axes_policy=:encoding,
        rq_cache=cache)

    sb_enc_2 = PM.rectangle_signed_barcode(Menc, pi;
        axes=axes_user,
        axes_policy=:encoding,
        rq_cache=cache)

    # Cache reuse must not change the output.
    @test sb_enc_1.rects == sb_enc_2.rects
    @test sb_enc_1.weights == sb_enc_2.weights
    @test sb_enc_1.axes == sb_enc_2.axes

    # encoding restriction should keep endpoints and only include encoding-axis points in between.
    enc_axes = PM.axes_from_encoding(pi)[1]
    @test first(sb_enc_1.axes[1]) == first(axes_user[1])
    @test last(sb_enc_1.axes[1]) == last(axes_user[1])
    @test all(v == first(axes_user[1]) || v == last(axes_user[1]) || (v in enc_axes) for v in sb_enc_1.axes[1])

    # coarsen policy must reduce axis length to max_axis_len.
    sb_coarse = PM.rectangle_signed_barcode(Menc, pi;
        axes=axes_user,
        axes_policy=:coarsen,
        max_axis_len=4)
    @test length(sb_coarse.axes[1]) <= 4

    # Bulk and local algorithms must agree (modulo ordering and zero pruning).
    sb_enc_bulk = PM.rectangle_signed_barcode(Menc, pi; axes=axes_user, axes_policy=:encoding,
                                              rq_cache=cache, method=:bulk)
    sb_enc_local = PM.rectangle_signed_barcode(Menc, pi; axes=axes_user, axes_policy=:encoding,
                                               rq_cache=cache, method=:local)
    @test Dict(zip(sb_enc_bulk.rects, sb_enc_bulk.weights)) ==
          Dict(zip(sb_enc_local.rects, sb_enc_local.weights))

end

@testset "Defaults that unblock usability (matching distance and sliced Wasserstein)" begin
    # NOTE: `const` is only allowed at global scope in Julia (module / top-level).
    # A @testset block is local scope, so we use plain local bindings here.
    PM  = Main.PosetModules
    PLB = PM.PLBackend
    PLP = PM.PLPolyhedra
    IR  = PM.IndicatorResolutions
    Inv = PM.Invariants

    # -------------------------
    # PLBackend: defaults from pi.reps
    # -------------------------
    Ups = [PLB.BoxUpset([0.0, 0.0], [1.0, 1.0])]
    Downs = [PLB.BoxDownset([2.0, 2.0], [3.0, 3.0])]

    P, H, pi = PLB.encode_fringe_boxes(Ups, Downs)
    M = IR.pmodule_from_fringe(H)

    # Out-of-the-box: no directions, no offsets.
    dmatch = Inv.matching_distance_approx(M, M, pi)
    @test isapprox(dmatch, 0.0; atol=1e-12)

    dsw = Inv.sliced_wasserstein_distance(M, M, pi)
    @test isapprox(dsw, 0.0; atol=1e-12)

    # Bounding box extraction: derived from pi.reps (finite points).
    opts = PM.InvariantOptions()
    lo, hi = Inv.encoding_box(pi, opts; margin=0.0)
    reps = pi.reps
    d = length(reps[1])
    lo_expected = [minimum(float(r[i]) for r in reps) for i in 1:d]
    hi_expected = [maximum(float(r[i]) for r in reps) for i in 1:d]
    @test lo == lo_expected
    @test hi == hi_expected

    # default_offsets: correct count and dimension in float case
    opts_box = PM.InvariantOptions(box=(lo, hi))
    offs = Inv.default_offsets(pi, opts_box; n_offsets=7, margin=0.0)
    @test length(offs) == 7
    @test all(length(x) == d for x in offs)

    # default_directions: correct count and dimension
    dirs = Inv.default_directions(pi; n_dirs=5, max_den=4, include_axes=false, normalize=:L1)
    @test length(dirs) == 5
    @test all(length(v) == d for v in dirs)

    # -------------------------
    # PLPolyhedra: defaults from pi.witnesses
    # -------------------------
    if PLP.HAVE_POLY
        # Unit box [0,2]^2 (as an upset/downset pair) so encoding produces a nontrivial pi.
        Upl = [PLP.PLUpset(PLP.PolyUnion(2, [PLP.make_hpoly([-1.0 0.0; 0.0 -1.0], [0.0, 0.0])]))]
        Downl = [PLP.PLDownset(PLP.PolyUnion(2, [PLP.make_hpoly([1.0 0.0; 0.0 1.0], [2.0, 2.0])]))]
        Phi = reshape(QQ[1], 1, 1)
        Fpl = PLP.PLFringe(Upl, Downl, Phi)

        P2, Hs2, pi2 = PLP.encode_from_PL_fringes(Fpl)
        H2 = Hs2[1]
        M2 = IR.pmodule_from_fringe(H2)

        dmatch2 = Inv.matching_distance_approx(M2, M2, pi2)
        @test isapprox(dmatch2, 0.0; atol=1e-12)

        dsw2 = Inv.sliced_wasserstein_distance(M2, M2, pi2)
        @test isapprox(dsw2, 0.0; atol=1e-12)

        lo2, hi2 = Inv.encoding_box(pi2; margin=0.0)
        wit = pi2.witnesses
        d2 = length(wit[1])
        lo2_expected = [minimum(float(w[i]) for w in wit) for i in 1:d2]
        hi2_expected = [maximum(float(w[i]) for w in wit) for i in 1:d2]
        @test lo2 == lo2_expected
        @test hi2 == hi2_expected
    end

    @test matching_distance_approx(M, M, pi) == 0.0
    @test sliced_wasserstein_distance(M, M, pi) == 0.0

    # New: Lp-generalization family members should also work out-of-the-box.
    @test sliced_bottleneck_distance(M, M, pi) == 0.0
    @test matching_wasserstein_distance_approx(M, M, pi) == 0.0

    if PLP.HAVE_POLY
        @test matching_distance_approx(M2, M2, pi2) == 0.0
        @test sliced_wasserstein_distance(M2, M2, pi2) == 0.0

        # New: same for the PLPolyhedra encoding map path.
        @test sliced_bottleneck_distance(M2, M2, pi2) == 0.0
        @test matching_wasserstein_distance_approx(M2, M2, pi2) == 0.0
    end

    @testset "Exact 2D matching distance: deterministic and correct on a toy example" begin
        # Toy coords-based encoding map:
        # Box [0,3]x[0,3] partitioned into three vertical stripes:
        #   region 1: x1 in [0,1), region 2: x1 in [1,2), region 3: x1 in [2,3]
        #
        # For modules M23 vs M3 on the chain 1<=2<=3, the exact Lesnick matching distance
        # over this box is 1.0 (achieved for sufficiently steep directions where weight=dir1
        # and bottleneck scales as 1/dir1).
        struct ToyBoxes2D
            coords::Vector{Vector{Float64}}
        end

        # locate for ToyBoxes2D: depends only on x[1], returns 1/2/3 inside [0,3], else 0.
        function PM.locate(pi::ToyBoxes2D, x::Vector{Float64})
            x1 = x[1]
            x2 = x[2]
            if (x1 < 0.0) || (x1 > 3.0) || (x2 < 0.0) || (x2 > 3.0)
                return 0
            elseif x1 < 1.0
                return 1
            elseif x1 < 2.0
                return 2
            else
                return 3
            end
        end

        pi = ToyBoxes2D([ [0.0, 1.0, 2.0, 3.0], [0.0, 3.0] ])
        box = ([0.0, 0.0], [3.0, 3.0])

        P = chain_poset(3)
        M23 = PM.interval_module(P, [2,3])
        M3  = PM.interval_module(P, [3])

        # Slice-chain exactness sanity check at one noncritical line:
        # dir = (1,1) normalized L1 -> (0.5,0.5).
        # offset chosen so y-x = 0.5 (i.e. dot([-0.5,0.5], x) = 0.25).
        chain, vals = PM.slice_chain_exact_2d(pi, [1.0, 1.0], 0.25; box=box, normalize_dirs=:L1)
        @test chain == [1,2,3]
        @test isapprox(vals[1], 0.5; atol=1e-12)
        @test isapprox(vals[2], 2.5; atol=1e-12)
        @test isapprox(vals[3], 4.5; atol=1e-12)
        @test isapprox(vals[4], 5.5; atol=1e-12)

        # Exact distance should be 1.0 on this toy configuration.
        d1 = PM.matching_distance_exact_2d(M23, M3, pi; box=box, weight=:lesnick_l1, normalize_dirs=:L1)
        d2 = PM.matching_distance_exact_2d(M23, M3, pi; box=box, weight=:lesnick_l1, normalize_dirs=:L1)
        @test d1 == d2  # determinism
        @test isapprox(d1, 1.0; atol=1e-10)

        # Agreement with the slice-based evaluator using the exact slice list.
        slices = PM.matching_distance_exact_slices_2d(pi; box=box, weight=:lesnick_l1, normalize_dirs=:L1)
        d3 = PM.matching_distance_approx(M23, M3, slices)
        @test isapprox(d3, d1; atol=1e-12)

        # Toy polyhedral encoding: same three vertical stripes, expressed as HPolys.
        # This exercises the polyhedral exact slice extraction path.
        function hpoly_box(xlo, xhi, ylo, yhi)
            A = Matrix{PM.QQ}(undef, 4, 2)
            b = Vector{PM.QQ}(undef, 4)
            # x >= xlo  <=>  -x <= -xlo
            A[1,1] = -1; A[1,2] =  0; b[1] = -PM.QQ(xlo)
            # x <= xhi
            A[2,1] =  1; A[2,2] =  0; b[2] =  PM.QQ(xhi)
            # y >= ylo  <=>  -y <= -ylo
            A[3,1] =  0; A[3,2] = -1; b[3] = -PM.QQ(ylo)
            # y <= yhi
            A[4,1] =  0; A[4,2] =  1; b[4] =  PM.QQ(yhi)
    # Canonical HPoly constructor: always pass strictness data explicitly.
    strict_mask = falses(size(A, 1))
    return PM.PLPolyhedra.HPoly(2, A, b, nothing, strict_mask, PM.PLPolyhedra.STRICT_EPS_QQ)
        end

        hp1 = hpoly_box(0.0, 1.0, 0.0, 3.0)
        hp2 = hpoly_box(1.0, 2.0, 0.0, 3.0)
        hp3 = hpoly_box(2.0, 3.0, 0.0, 3.0)

        sigy = [BitVector([0]), BitVector([0]), BitVector([0])]
        sigz = [BitVector([0]), BitVector([0]), BitVector([0])]
        witnesses = [[0.5, 1.5], [1.5, 1.5], [2.5, 1.5]]

        pi_poly = PM.PLPolyhedra.PLEncodingMap(2, sigy, sigz, [hp1, hp2, hp3], witnesses)

        d_poly = PM.matching_distance_exact_2d(M23, M3, pi_poly; box=box, weight=:lesnick_l1, normalize_dirs=:L1)
        @test isapprox(d_poly, 1.0; atol=1e-10)

        @testset "Fibered barcode cache 2D: augmented arrangement" begin
            # Full precompute cache (cells + barcodes)
            cache_full = Inv.fibered_barcode_cache_2d(M23, pi;
                box=box, normalize_dirs=:L1, precompute=:full)

            bar = Inv.fibered_barcode(cache_full, [1.0, 1.0], 0.25; values=:t)
            @test bar == Dict((2.5, 5.5) => 1)

            bar_idx = Inv.fibered_barcode_index(cache_full, [1.0, 1.0], 0.25)
            @test bar_idx == Dict((2, 4) => 1)

            # basepoint query matches offset query
            x0 = [-0.25, 0.25]
            @test Inv.fibered_barcode(cache_full, [1.0, 1.0], x0; values=:t) == bar

            # scaling direction does not change the slice (after normalization)
            @test Inv.fibered_barcode(cache_full, [2.0, 2.0], 0.25; values=:t) == bar

            st = Inv.fibered_barcode_cache_stats(cache_full)
            @test st.total_cells == st.n_cells_computed
            @test st.n_chains > 0
            @test st.n_index_barcodes_computed == st.n_chains

            # Lazy cache: ensure first query computes exactly one cell
            cache_lazy = Inv.fibered_barcode_cache_2d(M23, pi;
                box=box, normalize_dirs=:L1, precompute=:none)

            st0 = Inv.fibered_barcode_cache_stats(cache_lazy)
            @test st0.n_cells_computed == 0

            bar_lazy = Inv.fibered_barcode(cache_lazy, [1.0, 1.0], 0.25; values=:t)
            @test bar_lazy == bar

            st1 = Inv.fibered_barcode_cache_stats(cache_lazy)
            @test st1.n_cells_computed == 1

            _ = Inv.fibered_barcode(cache_lazy, [1.0, 1.0], 0.25; values=:t)
            st2 = Inv.fibered_barcode_cache_stats(cache_lazy)
            @test st2.n_cells_computed == 1

            # Polyhedral backend: index barcodes must agree exactly
            cache_poly = Inv.fibered_barcode_cache_2d(M23, pi_poly;
                box=box, normalize_dirs=:L1, precompute=:full)

            bar_poly_idx = Inv.fibered_barcode_index(cache_poly, [1.0, 1.0], 0.25)
            @test bar_poly_idx == Dict((2, 4) => 1)

            bar_poly = Inv.fibered_barcode(cache_poly, [1.0, 1.0], 0.25; values=:t)
            kv = collect(bar_poly)
            @test length(kv) == 1
            ((b, dth), mult) = kv[1]
            @test mult == 1
            @test isapprox(b, 2.5; atol=1e-8)
            @test isapprox(dth, 5.5; atol=1e-8)

            # Convenience wrapper: batched barcode queries on the cache.
            #
            # We support scalar (normal-offset) queries and basepoint queries.
            dirs = [[1.0, 1.0], [1.0, 2.0]]
            offs = Any[0.25, x0]
            sb = Inv.slice_barcodes(cache_full; directions=dirs, offsets=offs,
                                    values=:t, direction_weight=:none,
                                    normalize_weights=true)
            @test size(sb.barcodes) == (2, 2)
            @test size(sb.weights) == (2, 2)
            @test isapprox(sum(sb.weights), 1.0; atol=1e-12)
            @test all(isapprox.(sb.weights, 0.25; atol=1e-12))

            if Threads.nthreads() > 1
                S_serial = PM.slice_barcodes(cache_full, dirs, offs; threads = false)
                S_thread = PM.slice_barcodes(cache_full, dirs, offs; threads = true)
                @test S_thread.barcodes == S_serial.barcodes
                @test S_thread.weights == S_serial.weights
            end

            # Each entry agrees with the single-query API.
            @test sb.barcodes[1, 1] == Inv.fibered_barcode(cache_full, dirs[1], 0.25; values=:t)
            @test sb.barcodes[1, 2] == Inv.fibered_barcode(cache_full, dirs[1], x0; values=:t)

            # Cache-aware exact matching distance: reuse a shared arrangement.
            arr_shared = Inv.fibered_arrangement_2d(pi; box=box, normalize_dirs=:L1, precompute=:cells)
            cache23 = Inv.fibered_barcode_cache_2d(M23, arr_shared; precompute=:full)
            cache3  = Inv.fibered_barcode_cache_2d(M3,  arr_shared; precompute=:full)

            d_cache = Inv.matching_distance_exact_2d(cache23, cache3; weight=:lesnick_l1)
            @test isapprox(d_cache, d1; atol=1e-12)

            if Threads.nthreads() > 1
                d_serial = PM.matching_distance_exact_2d(cacheM, cacheN; threads = false)
                d_thread = PM.matching_distance_exact_2d(cacheM, cacheN; threads = true)
                @test d_thread == d_serial
            end

            # Arrangement-exact sliced kernel: compare to the slice-list backend.
            slices2 = PM.matching_distance_exact_slices_2d(pi; box=box, weight=:lesnick_l1, normalize_dirs=:L1)
            k_slices = PM.slice_kernel(M23, M3, slices2; kind=:bottleneck_gaussian, sigma=1.0)
            k_cache  = Inv.slice_kernel(cache23, cache3;
                                        kind=:bottleneck_gaussian, sigma=1.0,
                                        direction_weight=:lesnick_l1, cell_weight=:uniform)
            @test isapprox(k_cache, k_slices; atol=1e-12)

            if Threads.nthreads() > 1
                k_serial = PM.slice_kernel(cacheM, cacheN; kernel = :wasserstein, p = 1.0, threads = false)
                k_thread = PM.slice_kernel(cacheM, cacheN; kernel = :wasserstein, p = 1.0, threads = true)
                @test isapprox(k_thread, k_serial; rtol = 1e-12, atol = 1e-12)
            end
        end

        @testset "Multiparameter persistence images (Carriere MPPI)" begin
            # Reuse cache_full built above in this testset.

            imgA = Inv.mpp_image(cache_full; resolution=16, sigma=0.1, N=6, delta=:auto, q=1.0)
            imgB = Inv.mpp_image(cache_full; resolution=16, sigma=0.1, N=6, delta=:auto, q=1.0)

            @test size(imgA.img) == (16, 16)
            @test all(isfinite, imgA.img)
            @test all(x -> x >= 0.0, imgA.img)

            # Deterministic for same inputs
            @test imgA.img == imgB.img

            # Self-distance is zero
            @test isapprox(Inv.mpp_image_distance(imgA, imgA), 0.0; atol=1e-12)

            # Symmetry
            dAB = Inv.mpp_image_distance(imgA, imgB)
            dBA = Inv.mpp_image_distance(imgB, imgA)
            @test isapprox(dAB, dBA; atol=1e-12)

            # Kernel is 1 on identical inputs
            @test isapprox(Inv.mpp_image_kernel(imgA, imgA), 1.0; atol=1e-12)
        end


    end

end


@testset "Lp generalizations of slice-based distances" begin
    Inv = PM.Invariants
    PB  = PM.PLBackend
    IR  = PM.IndicatorResolutions

    # Build a small 2D PLBackend encoding and a fringe module on its poset.
    Ups = [PB.BoxUpset([0.0, 0.0], [1.0, 1.0])]
    Downs = [PB.BoxDownset([2.0, 2.0], [3.0, 3.0])]

    P, H, pi = PB.encode_fringe_boxes(Ups, Downs)
    M = IR.pmodule_from_fringe(H)
    Z = PM.zero_pmodule(M.Q, QQ)

    # Deterministic directions/offsets so the test is stable.
    dirs = Inv.default_directions(pi; n_dirs=3, max_den=3, include_axes=true, normalize=:L1)
    offs = Inv.default_offsets(pi; n_offsets=3)

    # -----------------------------------------------------------------------
    # Manual weighted p-mean check for sliced Wasserstein.
    # -----------------------------------------------------------------------
    outM = Inv.slice_barcodes(M, pi;
        directions=dirs,
        offsets=offs,
        direction_weight=:none,
        offset_weights=nothing,
        normalize_weights=true
    )
    bcsM = outM.barcodes
    W    = outM.weights

    outZ = Inv.slice_barcodes(Z, pi;
        directions=dirs,
        offsets=offs,
        direction_weight=:none,
        offset_weights=nothing,
        normalize_weights=true
    )
    bcsZ = outZ.barcodes

    agg_p = 2.0
    s = 0.0
    for i in axes(bcsM, 1), j in axes(bcsM, 2)
        d = Inv.wasserstein_distance(bcsM[i, j], bcsZ[i, j]; p=2, q=1)
        s += W[i, j] * d^agg_p
    end
    manual_sw = s^(1 / agg_p)

    got_sw = Inv.sliced_wasserstein_distance(
        M, Z, pi;
        directions=dirs,
        offsets=offs,
        weight=:none,
        offset_weights=:none,
        normalize_weights=true,
        p=2,
        q=1,
        agg=:pmean,
        agg_p=agg_p,
        strict=true
    )
    @test isapprox(got_sw, manual_sw; atol=1e-12, rtol=1e-12)

    # -----------------------------------------------------------------------
    # Manual weighted p-mean check for sliced bottleneck.
    # -----------------------------------------------------------------------
    s2 = 0.0
    for i in axes(bcsM, 1), j in axes(bcsM, 2)
        d = Inv.bottleneck_distance(bcsM[i, j], bcsZ[i, j])
        s2 += W[i, j] * d^agg_p
    end
    manual_sb = s2^(1 / agg_p)

    got_sb = Inv.sliced_bottleneck_distance(
        M, Z, pi;
        directions=dirs,
        offsets=offs,
        weight=:none,
        offset_weights=:none,
        normalize_weights=true,
        agg=:pmean,
        agg_p=agg_p,
        strict=true
    )
    @test isapprox(got_sb, manual_sb; atol=1e-12, rtol=1e-12)

    # -----------------------------------------------------------------------
    # Manual matching-style max check for matching_wasserstein_distance_approx.
    # -----------------------------------------------------------------------
    outM2 = Inv.slice_barcodes(M, pi;
        directions=dirs,
        offsets=offs,
        direction_weight=:lesnick_linf,
        offset_weights=nothing,
        normalize_weights=false
    )
    bcsM2 = outM2.barcodes
    W2    = outM2.weights

    outZ2 = Inv.slice_barcodes(Z, pi;
        directions=dirs,
        offsets=offs,
        direction_weight=:lesnick_linf,
        offset_weights=nothing,
        normalize_weights=false
    )
    bcsZ2 = outZ2.barcodes

    manual_mw = 0.0
    for i in axes(bcsM2, 1), j in axes(bcsM2, 2)
        d = Inv.wasserstein_distance(bcsM2[i, j], bcsZ2[i, j]; p=2, q=1)
        val = W2[i, j] * d
        manual_mw = max(manual_mw, val)
    end

    got_mw = Inv.matching_wasserstein_distance_approx(
        M, Z, pi;
        directions=dirs,
        offsets=offs,
        weight=:lesnick_linf,
        p=2,
        q=1,
        strict=true
    )
    @test isapprox(got_mw, manual_mw; atol=1e-12, rtol=1e-12)
end

# Included from test/runtests.jl; uses shared aliases (PM, QQ, PLB, ...).
#
# PLBackend is now always loaded by src/PosetModules.jl, so the historical
# force-load hack is no longer needed here.

# ASCII-only predicate (hard requirement for this project).
_is_ascii(s::AbstractString) = all(c -> Int(c) <= 0x7f, s)

@testset "PrettyPrinter (ASCII) for NamedTuple summaries" begin
    @testset "basic formatting, sorting, truncation" begin
        nt = (
            a = 1,
            b = 2.0,
            v = collect(1:10),
            d = Dict(2 => 3.0, 1 => 4.0),
            nested = (x = 1, y = [1.0, 2.0, 3.0, 4.0]),
        )

        s = PM.pretty(nt; name="nt", max_list=3, max_items=20)

        @test occursin("nt:", s)
        @test occursin("a = 1", s)
        @test occursin("b = 2.0", s)

        # Vector truncation uses ASCII "..." and reports length.
        @test occursin("v = [1, 2, 3, ...] (len=10)", s)
        # Hard requirement: pretty-print output must be ASCII-only.
        # In particular, no unicode ellipsis (U+2026).
        @test !occursin("\u2026", s)

        # Dict keys are sorted numerically (1 before 2).
        @test occursin("1 => 4.0", s)
        @test occursin("2 => 3.0", s)
        i1 = findfirst("1 => 4.0", s)
        i2 = findfirst("2 => 3.0", s)
        @test i1 !== nothing && i2 !== nothing && first(i1) < first(i2)

        @test _is_ascii(s)

        # Wrapper prints the same representation via MIME"text/plain".
        sw = repr("text/plain", PM.PrettyPrinter(nt; name="nt", max_list=3, max_items=20))
        @test sw == s
    end

    @testset "max_depth limits recursion" begin
        deep = (a = (b = (c = (d = 1,))),)
        s = PM.pretty(deep; name="deep", max_depth=1)
        @test occursin("deep:", s)
        @test occursin("a = ...", s)
        @test _is_ascii(s)
    end

    @testset "integration: module_geometry_summary pretty prints" begin
        # 1D example: one upset x >= 0 and one downset x <= 2.
        Ups = [PLB.BoxUpset([0.0])]
        Downs = [PLB.BoxDownset([2.0])]
        Phi = reshape(QQ[1], 1, 1)

        P, H, pi = PLB.encode_fringe_boxes(Ups, Downs, Phi)

        box = ([-2.0], [7.0])
        summ = PM.module_geometry_summary(H, pi; box=box, nbins=4)

        s = PM.pretty(summ; name="module_geometry_summary", max_items=50, max_list=4)

        @test occursin("module_geometry_summary:", s)
        @test occursin("size_summary", s)
        @test occursin("interface_measure", s)
        @test occursin("volume_samples_by_dim", s)
        @test occursin("graph_stats", s)

        @test _is_ascii(s)
        # Hard requirement: pretty-print output must be ASCII-only.
        # In particular, no unicode ellipsis (U+2026).
        @test !occursin("\u2026", s)
        @test length(s) > 0
    end
end

using Test


@testset "Derived-invariant size measures: strata + Betti/Bass support" begin
    # Build the same simple 1D PLBackend encoding:
    # thresholds at 0 and 2 produce 3 regions with reps near -1, 1, 3.
    Ups = [PLB.BoxUpset([0.0])]
    Downs = [PLB.BoxDownset([2.0])]
    Phi = reshape(QQ[1], 1, 1)
    P, Hhat, pi = PLB.encode_fringe_boxes(Ups, Downs, Phi)

    box = ([-1.0], [3.0])
    w = PM.region_weights(pi; box=box)

    # Identify regions robustly using locate (avoid relying on ordering).
    rL = PM.locate(pi, [-0.5])
    rM = PM.locate(pi, [1.0])
    rR = PM.locate(pi, [2.5])

    @test isapprox(w[rL], 1.0; atol=1e-9)
    @test isapprox(w[rM], 2.0; atol=1e-9)
    @test isapprox(w[rR], 1.0; atol=1e-9)

    # Value strata (e.g. where some invariant is constant)
    vals = zeros(Int, P.n)
    vals[rM] = 1
    vals[rR] = 1

    m0 = PM.measure_by_value(vals, 0, pi; box=box)
    m1 = PM.measure_by_value(vals, 1, pi; box=box)
    @test isapprox(m0, 1.0; atol=1e-9)
    @test isapprox(m1, 3.0; atol=1e-9)

    mask = falses(P.n)
    mask[rM] = true
    mask[rR] = true
    sm = PM.support_measure(mask, pi; box=box)
    @test isapprox(sm, 3.0; atol=1e-9)

    # Betti support by degree (toy numeric example)
    B = zeros(Int, 2, P.n)
    B[1, rL] = 1
    B[1, rR] = 2
    B[2, rM] = 1

    bs = PM.betti_support_measures(B, pi; box=box)
    @test isapprox(bs.support_total, 1.0 + 2.0 + 2.0; atol=1e-9)
    @test length(bs.support_by_degree) == 2

    # Same in dictionary form (multigraded)
    Bd = Dict{Tuple{Int,Int}, Int}(
        (0, rL) => 1,
        (0, rR) => 2,
        (1, rM) => 1
    )
    bs2 = PM.betti_support_measures(Bd, pi; box=box)
    @test isapprox(bs2.support_total, bs.support_total; atol=1e-9)
end

@testset "MPPI: cutoffs, pruning, serialization, and mma wrapper" begin
    IR = PM.IndicatorResolutions
    Inv = PM.Invariants

    # A tiny 2D PL backend encoding to keep the test fast.
    Ups = [PLB.BoxUpset([0.0, 0.0], [1.0, 1.0])]
    Downs = [PLB.BoxDownset([2.0, 2.0], [3.0, 3.0])]
    P, H, pi = PLB.encode_fringe_boxes(Ups, Downs)
    M = IR.pmodule_from_fringe(H)

    img_exact = Inv.mpp_image(M, pi; resolution=8, sigma=0.3, N=4)
    @test img_exact isa Inv.MPPImage
    @test length(img_exact.xgrid) == 8
    @test length(img_exact.ygrid) == 8
    @test size(img_exact.img) == (8, 8)
    @test isapprox(img_exact.sigma, 0.3; atol=0.0, rtol=0.0)

    # Re-evaluation from the stored decomposition should match.
    img2 = Inv.mpp_image(img_exact.decomp; xgrid=img_exact.xgrid, ygrid=img_exact.ygrid, sigma=img_exact.sigma)
    @test img2.img == img_exact.img

    # segment_prune toggles an exact optimization.
    img_np = Inv.mpp_image(img_exact.decomp;
                           xgrid=img_exact.xgrid,
                           ygrid=img_exact.ygrid,
                           sigma=img_exact.sigma,
                           segment_prune=false)
    @test img_np.img == img_exact.img

    # Large cutoff should be exact (no pruning).
    img_big = Inv.mpp_image(img_exact.decomp;
                            xgrid=img_exact.xgrid,
                            ygrid=img_exact.ygrid,
                            sigma=img_exact.sigma,
                            cutoff_radius=1e6)
    @test img_big.img == img_exact.img

    # cutoff_tol is an optional approximation; it can only decrease the image (nonnegative kernel).
    img_cut = Inv.mpp_image(img_exact.decomp;
                            xgrid=img_exact.xgrid,
                            ygrid=img_exact.ygrid,
                            sigma=img_exact.sigma,
                            cutoff_tol=1e-6)
    @test all(img_cut.img .<= img_exact.img .+ 1e-12)

    # Cutoff arguments are mutually exclusive.
    @test_throws ArgumentError Inv.mpp_image(img_exact.decomp;
                                            xgrid=img_exact.xgrid,
                                            ygrid=img_exact.ygrid,
                                            sigma=img_exact.sigma,
                                            cutoff_radius=1.0,
                                            cutoff_tol=1e-3)
    @test_throws ArgumentError Inv.mpp_image(img_exact.decomp;
                                            xgrid=img_exact.xgrid,
                                            ygrid=img_exact.ygrid,
                                            sigma=img_exact.sigma,
                                            cutoff_tol=1.0)

    # Inner product should require identical grids (not just the same lengths).
    img_shift = Inv.mpp_image(img_exact.decomp;
                              xgrid=img_exact.xgrid .+ 0.1,
                              ygrid=img_exact.ygrid,
                              sigma=img_exact.sigma)
    @test_throws ErrorException Inv.mpp_image_inner_product(img_exact, img_shift)

    # JSON round-trip: decomposition.
    mktemp() do path, io
        close(io)
        Inv.save_mpp_decomposition_json(path, img_exact.decomp)
        decomp_rt = Inv.load_mpp_decomposition_json(path)
        img_rt = Inv.mpp_image(decomp_rt; xgrid=img_exact.xgrid, ygrid=img_exact.ygrid, sigma=img_exact.sigma)
        @test isapprox(img_rt.img, img_exact.img; atol=1e-12, rtol=0.0)
    end

    # JSON round-trip: full image.
    mktemp() do path, io
        close(io)
        Inv.save_mpp_image_json(path, img_exact)
        img_rt = Inv.load_mpp_image_json(path)
        @test isapprox(img_rt.xgrid, img_exact.xgrid; atol=1e-12, rtol=0.0)
        @test isapprox(img_rt.ygrid, img_exact.ygrid; atol=1e-12, rtol=0.0)
        @test isapprox(img_rt.sigma, img_exact.sigma; atol=0.0, rtol=0.0)
        @test isapprox(img_rt.img, img_exact.img; atol=1e-12, rtol=0.0)
    end

    # mma_decomposition wrapper.
    out = Inv.mma_decomposition(M, pi; method=:mpp_image, mpp_kwargs=(resolution=6, N=3, sigma=0.25))
    @test out isa Inv.MPPImage
    @test length(out.xgrid) == 6
    @test length(out.ygrid) == 6
end

@testset "Projected distances and differentiable feature maps" begin
    Inv = PM.Invariants
    IR = PM.IndicatorResolutions

    # Simple chain module tests
    P = chain_poset(3)
    M23 = IR.pmodule_from_fringe(one_by_one_fringe(P, [2,3]))
    M3  = IR.pmodule_from_fringe(one_by_one_fringe(P, [3]))

    arr = Inv.projected_arrangement(P, [0.0, 1.0, 2.0])
    c23 = Inv.projected_barcode_cache(M23, arr)
    c3  = Inv.projected_barcode_cache(M3, arr)

    b23 = Inv.projected_barcodes(c23)[1]
    b3  = Inv.projected_barcodes(c3)[1]

    @test b23 == Dict((1.0,3.0)=>1)
    @test b3  == Dict((2.0,3.0)=>1)

    d = Inv.projected_distance(c23, c3; dist=:bottleneck, agg=:mean)
    @test isapprox(d, 1.0; atol=1e-12)

    ksame = Inv.projected_kernel(c23, c23; kind=:wasserstein_gaussian, sigma=1.0, agg=:mean)
    @test isapprox(ksame, 1.0; atol=1e-12)

    # Differentiable persistence image agrees with fast version
    bar = Dict((0.0,1.0)=>1, (0.5,1.5)=>1)
    xg = 0.0:0.25:2.0
    yg = 0.0:0.25:2.0

    pi_fast = Inv.persistence_image(bar; xgrid=xg, ygrid=yg, sigma=0.3,
                                   coords=:birth_death, weighting=:none, normalize=:none)

    pi_diff = Inv.persistence_image(bar; xgrid=xg, ygrid=yg, sigma=0.3,
                                   coords=:birth_death, weighting=:none, normalize=:none,
                                   differentiable=true)

    @test isapprox(pi_fast.values, pi_diff.values)

    fv = Inv.feature_vector(bar; kind=:persistence_image, xgrid=xg, ygrid=yg, sigma=0.3,
                            coords=:birth_death, weighting=:none, normalize=:none,
                            differentiable=true)

    @test length(fv) == length(xg)*length(yg)
    @test isapprox(reshape(fv, length(yg), length(xg)), pi_diff.values)
end

@testset "fibered slice family + typed caches" begin
    box = [-1.0, 1.0, -1.0, 1.0]
    arr_shared = Inv.fibered_arrangement_2d(pi; box=box, normalize_dirs=:L1, precompute=:cells)

    fam = Inv.fibered_slice_family_2d(arr_shared; direction_weight=:lesnick_l1, store_values=true)
    @test fam.arrangement === arr_shared
    @test Inv.nslices(fam) > 0

    cache23 = Inv.fibered_barcode_cache_2d(M23, arr_shared; precompute=:full)
    cache3  = Inv.fibered_barcode_cache_2d(M3,  arr_shared; precompute=:full)

    # typed cache: no Any
    @test eltype(cache23.index_barcodes) <: Union{Nothing,Inv.IndexBarcode}

    # exact distance matches slice-list backend
    d_slices = PM.matching_distance_exact_2d(M23, M3, pi; box=box, weight=:lesnick_l1, normalize_dirs=:L1)
    d_cache  = PM.matching_distance_exact_2d(cache23, cache3; weight=:lesnick_l1, family=fam, threads=false)
    @test isapprox(d_cache, d_slices; atol=1e-12)

    # thread determinism
    d_thr = PM.matching_distance_exact_2d(cache23, cache3; weight=:lesnick_l1, family=fam, threads=true)
    @test isapprox(d_thr, d_cache; atol=1e-12)

    # kernel matches slice-list backend (uniform cell weighting)
    slices2 = PM.matching_distance_exact_slices_2d(pi; box=box, weight=:lesnick_l1, normalize_dirs=:L1)
    k_slices = PM.slice_kernel(M23, M3, slices2; kind=:bottleneck_gaussian, sigma=1.0, normalize_weights=true)
    k_cache  = PM.slice_kernel(cache23, cache3; kind=:bottleneck_gaussian, sigma=1.0,
                               direction_weight=:lesnick_l1, cell_weight=:uniform,
                               family=fam, normalize_weights=true, threads=false)
    @test isapprox(k_cache, k_slices; atol=1e-12)
end