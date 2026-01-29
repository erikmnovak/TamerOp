using Test
using Random

# Included from test/runtests.jl; uses shared aliases (PM, FF, PLP, PLB, QQ, ...).

@testset "Region geometry for PLBackend (axis-aligned boxes)" begin

    # 1D example: one upset x >= 0 and one downset x <= 2.
    Ups = [PLB.BoxUpset([0.0])]
    Downs = [PLB.BoxDownset([2.0])]
    Phi = reshape(QQ[1], 1, 1)  # 1x1 matrix

    P, H, pi = PLB.encode_fringe_boxes(Ups, Downs, Phi)

    # Volume weights in the box [-2, 7] (length 9).
    box = ([-2.0], [7.0])
    w = PM.region_weights(pi; box=box)

    @test length(w) == P.n
    @test isapprox(sum(w), 9.0; atol=1e-12)

    t_left  = PLB.locate(pi, [-1.0])  # x < 0
    t_mid   = PLB.locate(pi, [ 1.0])  # 0 < x < 2
    t_right = PLB.locate(pi, [ 3.0])  # x > 2

    @test isapprox(w[t_left],  2.0; atol=1e-12)
    @test isapprox(w[t_mid],   2.0; atol=1e-12)
    @test isapprox(w[t_right], 5.0; atol=1e-12)

    # Bounding box and diameter for the middle region inside the ambient box.
    bb_mid = PM.region_bbox(pi, t_mid; box=box)
    @test bb_mid !== nothing
    lo, hi = bb_mid
    @test lo == [0.0]
    @test hi == [2.0]

    @test isapprox(PM.region_diameter(pi, t_mid; box=box, metric=:L2),   2.0; atol=1e-12)
    @test isapprox(PM.region_diameter(pi, t_mid; box=box, metric=:Linf), 2.0; atol=1e-12)
    @test isapprox(PM.region_diameter(pi, t_mid; box=box, metric=:L1),   2.0; atol=1e-12)

    # 2D example where a region is a UNION of multiple grid cells:
    # one downset D = {x <= 0, y <= 0}. Then z = !contains(D, x) is true outside that quadrant.
    Ups2 = PLB.BoxUpset[]
    Downs2 = [PLB.BoxDownset([0.0, 0.0])]
    Phi2 = zeros(QQ, 1, 0)  # r=1, m=0

    P2, H2, pi2 = PLB.encode_fringe_boxes(Ups2, Downs2, Phi2)

    box2 = ([-1.0, -1.0], [1.0, 1.0])
    w2 = PM.region_weights(pi2; box=box2)
    @test length(w2) == P2.n
    @test isapprox(sum(w2), 4.0; atol=1e-12)

    t_in  = PLB.locate(pi2, [-0.5, -0.5])  # inside D
    t_out = PLB.locate(pi2, [ 0.5,  0.5])  # outside D (union of 3 cells)

    @test isapprox(w2[t_in],  1.0; atol=1e-12)
    @test isapprox(w2[t_out], 3.0; atol=1e-12)

    bb_in = PM.region_bbox(pi2, t_in; box=box2)
    @test bb_in !== nothing
    lo_in, hi_in = bb_in
    @test lo_in == [-1.0, -1.0]
    @test hi_in == [0.0, 0.0]
    @test isapprox(PM.region_diameter(pi2, t_in; box=box2, metric=:Linf), 1.0; atol=1e-12)
    @test isapprox(PM.region_diameter(pi2, t_in; box=box2, metric=:L2),   sqrt(2.0); atol=1e-12)

    bb_out = PM.region_bbox(pi2, t_out; box=box2)
    @test bb_out !== nothing
    lo_out, hi_out = bb_out
    @test lo_out == [-1.0, -1.0]
    @test hi_out == [1.0, 1.0]
    @test isapprox(PM.region_diameter(pi2, t_out; box=box2, metric=:Linf), 2.0; atol=1e-12)
    @test isapprox(PM.region_diameter(pi2, t_out; box=box2, metric=:L2),   2.0 * sqrt(2.0); atol=1e-12)
end

@testset "Region geometry for PLEncodingMap (Monte Carlo weights + bbox/diameter)" begin
    # Build a toy 1D polyhedral encoding with two regions:
    #   R1 = [0, 1], R2 = [1, 2]
    A = reshape([1, -1], 2, 1)  # constraints are A*x <= b
    b1 = [1, 0]                 # x <= 1, -x <= 0  => x >= 0
    b2 = [2, -1]                # x <= 2, -x <= -1 => x >= 1

    hp1 = PLP.make_hpoly(A, b1)
    hp2 = PLP.make_hpoly(A, b2)

    sigy = [BitVector([false]), BitVector([true])]
    sigz = [BitVector([false]), BitVector([false])]
    pi = PLP.PLEncodingMap(1, sigy, sigz, [hp1, hp2], [Float64[0.5], Float64[1.5]])

    @test PLP.locate(pi, [0.25]) != 0
    @test PLP.locate(pi, [1.75]) != 0

    box = ([0.0], [2.0])

    # Default: uniform weights if no box.
    wunif = PM.region_weights(pi)
    @test length(wunif) == 2
    @test wunif == [1.0, 1.0]

    # Monte Carlo weights inside [0,2] should be close to [1,1].
    rng = MersenneTwister(12345)
    # Default is :exact (requires Polyhedra). For this test we want Monte Carlo.
    w = PM.region_weights(pi; box=box, method=:mc, nsamples=20_000, rng=rng, strict=true)

    t1 = PLP.locate(pi, [0.5])
    t2 = PLP.locate(pi, [1.5])

    @test isapprox(w[t1], 1.0; atol=0.05)
    @test isapprox(w[t2], 1.0; atol=0.05)
    @test isapprox(sum(w), 2.0; atol=0.05)

    # Exact bbox by vertex enumeration in 1D.
    bb1 = PM.region_bbox(pi, t1; box=box)
    @test bb1 !== nothing
    lo1, hi1 = bb1
    @test lo1 == [0.0]
    @test hi1 == [1.0]

    # Diameter by bbox and by vertices should match in 1D.
    @test isapprox(PM.region_diameter(pi, t1; box=box, metric=:L2, method=:bbox),     1.0; atol=1e-12)
    @test isapprox(PM.region_diameter(pi, t1; box=box, metric=:L2, method=:vertices), 1.0; atol=1e-12)
end

@testset "PLPolyhedra heavy geometry: perimeter/surface/adjacency/PCA" begin
    # 2D: square [0,1]^2 perimeter should be 4
    A = QQ[ 1 0;
            0 1;
           -1 0;
            0 -1 ]
    b = QQ[1, 1, 0, 0]
    hp = PLP.make_hpoly(A, b)

    pi = PLP.PLEncodingMap(2, [Bool[]], [Bool[]], [hp], [Float64[0.5, 0.5]])

    box = (Float64[0.0, 0.0], Float64[1.0, 1.0])
    perim = PM.region_perimeter(pi, 1; box=box)
    @test isapprox(perim, 4.0; atol=1e-9)

    # 3D: cube [0,1]^3 surface area is 6
    A3 = QQ[ 1 0 0;
             0 1 0;
             0 0 1;
            -1 0 0;
             0 -1 0;
             0 0 -1 ]
    b3 = QQ[1, 1, 1, 0, 0, 0]
    hp3 = PLP.make_hpoly(A3, b3)
    pi3 = PLP.PLEncodingMap(3, [Bool[]], [Bool[]], [hp3], [Float64[0.5, 0.5, 0.5]])
    box3 = (Float64[0,0,0], Float64[1,1,1])
    sa = PM.region_surface_area(pi3, 1; box=box3)
    @test isapprox(sa, 6.0; atol=1e-9)

    # Adjacency: two rectangles sharing a vertical edge of length 1
    # R1 = [0,1]x[0,1], R2 = [1,2]x[0,1]
    A1 = QQ[ 1 0; 0 1; -1 0; 0 -1 ]
    b1 = QQ[1, 1, 0, 0]
    A2 = QQ[ 1 0; 0 1; -1 0; 0 -1 ]
    b2 = QQ[2, 1, -1, 0]
    hp1 = PLP.make_hpoly(A1, b1)
    hp2 = PLP.make_hpoly(A2, b2)
    pi2 = PLP.PLEncodingMap(2, [Bool[], Bool[]], [Bool[], Bool[]],
                                    [hp1, hp2],
                                    [Float64[0.5,0.5], Float64[1.5,0.5]])
    box2 = (Float64[0,0], Float64[2,1])
    adj = PM.region_adjacency(pi2; box=box2)
    @test haskey(adj, (1,2))
    @test isapprox(adj[(1,2)], 1.0; atol=1e-8)

    # --- PCA / principal directions diagnostics ---
    rng = MersenneTwister(1)
    pca = PM.region_principal_directions(pi, 1; box=box, nsamples=5000, rng=rng, strict=true)

    @test :n_accepted in propertynames(pca)
    @test :n_proposed in propertynames(pca)
    @test pca.naccepted == pca.n_accepted
    @test pca.nproposed == pca.n_proposed

    @test pca.n_accepted == 5000
    @test pca.n_proposed == 5000

    @test isapprox(pca.mean[1], 0.5; atol=0.02)
    @test isapprox(pca.mean[2], 0.5; atol=0.02)
    @test isapprox(pca.evals[1], 1/12; atol=0.02)
    @test isapprox(pca.evals[2], 1/12; atol=0.02)

    rng2 = MersenneTwister(2)
    pca_info = PM.region_principal_directions(pi, 1;
        box=box, nsamples=4000, nbatches=4, rng=rng2, strict=true, return_info=true)

    @test pca_info.nbatches == 4
    @test length(pca_info.batch_evals) == 4
    @test length(pca_info.batch_n_accepted) == 4
    @test sum(pca_info.batch_n_accepted) == 4000

    @test all(pca_info.mean_stderr .>= 0.0)
    @test all(pca_info.evals_stderr .>= 0.0)



    # -------------------------------------------------------------------------
    # New features:
    #   (1) exact volume mode for region_weights via Polyhedra.volume
    #   (2) caching of polyhedra-in-box objects
    #   (3) per-facet boundary-measure breakdown diagnostics
    # -------------------------------------------------------------------------

    # (1) exact volume mode: unit square in unit box has area 1.
    w_exact = PM.region_weights(pi; box=box, method=:exact)
    @test length(w_exact) == 1
    @test isapprox(w_exact[1], 1.0; atol=1e-12, rtol=0.0)

    # (2) cache: results should match (and box may be omitted if cache is provided)
    cache1 = PLP.poly_in_box_cache(pi; box=box, closure=true)

    perim_cache = PM.region_perimeter(pi, 1; cache=cache1)
    @test isapprox(perim_cache, perim; atol=1e-12, rtol=0.0)

    # Define the non-cached centroid for comparison.
    c = PM.region_centroid(pi, 1; box=box)

    c_cache = PM.region_centroid(pi, 1; cache=cache1)
    @test isapprox(c_cache[1], c[1]; atol=1e-10)
    @test isapprox(c_cache[2], c[2]; atol=1e-10)

    # (3) boundary-measure breakdown: sum of facet measures should match perimeter.
    bd = PM.region_boundary_measure_breakdown(pi, 1; cache=cache1)
    @test !isempty(bd)
    @test all(e.measure > 0 for e in bd)
    @test isapprox(sum(e.measure for e in bd), perim; atol=1e-8, rtol=0.0)

    # Two adjacent rectangles in the same window; internal facet should be detected.
    A = [ 1 0;
         -1 0;
          0 1;
          0 -1]
    hp_left  = PLP.make_hpoly(A, [1,  0, 1, 0])   # 0 <= x <= 1, 0 <= y <= 1
    hp_right = PLP.make_hpoly(A, [2, -1, 1, 0])   # 1 <= x <= 2, 0 <= y <= 1
    pi2 = PLP.PLEncodingMap(2, BitVector[], BitVector[], [hp_left, hp_right],
                            [[0.5, 0.5], [1.5, 0.5]])
    box2 = ([0.0, 0.0], [2.0, 1.0])
    cache2 = PLP.poly_in_box_cache(pi2; box=box2, closure=true)

    w2 = PM.region_weights(pi2; cache=cache2, method=:exact)
    @test length(w2) == 2
    @test isapprox(w2[1], 1.0; atol=1e-12, rtol=0.0)
    @test isapprox(w2[2], 1.0; atol=1e-12, rtol=0.0)

    bd1 = PM.region_boundary_measure_breakdown(pi2, 1; cache=cache2)
    p1 = PM.region_perimeter(pi2, 1; cache=cache2)
    @test isapprox(sum(e.measure for e in bd1), p1; atol=1e-8, rtol=0.0)
    @test any(e.kind == :internal for e in bd1)
    # The only internal neighbor of region 1 in this setup is region 2; its shared edge has length 1.
    mint = sum(e.measure for e in bd1 if e.neighbor == 2)
    @test isapprox(mint, 1.0; atol=1e-8, rtol=0.0)

end

@testset "Extended region geometry descriptors (axis backend)" begin
        # Build a simple 2D encoding map: split the plane into quadrants by x=0 and y=0.
        # Each quadrant is one region. Intersect with box [-1,1]^2 to get unit squares.
        Ups = PLB.BoxUpset[
            PLB.BoxUpset([0.0, -Inf]),   # x >= 0
            PLB.BoxUpset([-Inf, 0.0]),   # y >= 0
        ]
        Downs = PLB.BoxDownset[]

        coords = Vector{Vector{Float64}}([[0.0], [0.0]])

        sig_y = BitVector[
            BitVector([0, 0]),  # lower-left
            BitVector([0, 1]),  # upper-left
            BitVector([1, 0]),  # lower-right
            BitVector([1, 1]),  # upper-right
        ]
        sig_z = BitVector[BitVector(), BitVector(), BitVector(), BitVector()]

        reps = [
            [-0.5, -0.5],
            [-0.5,  0.5],
            [ 0.5, -0.5],
            [ 0.5,  0.5],
        ]

        pi = PLB.PLEncodingMapBoxes(coords, sig_y, sig_z, reps, Ups, Downs)

        box = ([-1.0, -1.0], [1.0, 1.0])

        w = PM.region_weights(pi; box=box)
        @test length(w) == 4
        @test all(abs.(w .- 1.0) .< 1e-12)

        r = 1  # lower-left: [-1,0] x [-1,0]

        cb = PM.region_chebyshev_ball(pi, r; box=box)
        @test isapprox(cb.radius, 0.5; atol=1e-12)
        @test all(isapprox.(cb.center, [-0.5, -0.5]; atol=1e-12))

        @test isapprox(PM.region_inradius(pi, r; box=box), 0.5; atol=1e-12)

        cr = PM.region_circumradius(pi, r; box=box, center=:chebyshev, metric=:L2, method=:cells)
        @test isapprox(cr, sqrt(0.5); atol=1e-12)

        # Planar isoperimetric quotient of a unit square: 4*pi*A/P^2 = pi/4.
        iso_planar = PM.region_isoperimetric_ratio(pi, r; box=box, kind=:planar)
        @test isapprox(iso_planar, pi/4; atol=1e-10)

        # Boundary-to-volume for unit square: perimeter/area = 4.
        b2v = PM.region_boundary_to_volume_ratio(pi, r; box=box)
        @test isapprox(b2v, 4.0; atol=1e-10)

        # Mean width:
        # - cauchy formula (convex planar) gives 4/pi exactly
        mw_cauchy = PM.region_mean_width(pi, r; box=box, method=:cauchy)
        @test isapprox(mw_cauchy, 4.0/pi; atol=1e-10)

        # - cell-based estimate should be close (direction sampling only)
        mw_cells = PM.region_mean_width(pi, r; box=box, method=:cells, ndirs=2000, rng=MersenneTwister(1))
        @test isapprox(mw_cells, 4.0/pi; atol=0.03)

        mf = PM.region_minkowski_functionals(pi, r; box=box, mean_width_method=:cauchy)
        @test isapprox(mf.volume, 1.0; atol=1e-12)
        @test isapprox(mf.boundary_measure, 4.0; atol=1e-12)
        @test isapprox(mf.mean_width, 4.0/pi; atol=1e-10)
end

@testset "Module-level geometry distributions and adjacency graph stats" begin
        # Deterministic 1D encoding map with three regions:
        # thresholds at 0 and 2; intersect with box [-2,7] gives lengths [2,2,5].
        Ups = PLB.BoxUpset[
            PLB.BoxUpset([0.0]),
            PLB.BoxUpset([2.0]),
        ]
        Downs = PLB.BoxDownset[]

        coords = Vector{Vector{Float64}}([[0.0, 2.0]])

        sig_y = BitVector[
            BitVector([0, 0]),  # (-Inf,0)
            BitVector([1, 0]),  # (0,2)
            BitVector([1, 1]),  # (2,Inf)
        ]
        sig_z = BitVector[BitVector(), BitVector(), BitVector()]
        reps = [[-1.0], [1.0], [3.0]]

        pi = PLB.PLEncodingMapBoxes(coords, sig_y, sig_z, reps, Ups, Downs)

        box = ([-2.0], [7.0])
        weights = PM.region_weights(pi; box=box)
        @test isapprox.(weights, [2.0, 2.0, 5.0]; atol=1e-12) |> all

        adj = PM.region_adjacency(pi; box=box)
        @test length(adj) == 2
        @test isapprox(sum(values(adj)), 2.0; atol=1e-12)

        # "Module" dimensions per region (using the new restricted_hilbert overload).
        dims = [0, 1, 0]

        vols = PM.region_volume_samples_by_dim(dims, pi; weights=weights)
        @test sort(vols[0]) == sort([2.0, 5.0])
        @test vols[1] == [2.0]

        b2v = PM.region_boundary_to_volume_samples_by_dim(dims, pi; box=box, weights=weights)
        @test sort(b2v[0]) == sort([1.0, 0.4])
        @test b2v[1] == [1.0]

        opts = PM.InvariantOptions()
        gs = PM.region_adjacency_graph_stats(dims, pi, opts; adjacency=adj)
        @test gs.nregions == 3
        @test gs.nedges == 2
        @test isapprox(gs.total_edge_weight, 2.0; atol=1e-12)
        @test sort(gs.degrees) == [1, 1, 2]
        @test isapprox(gs.modularity_by_dim, -0.5; atol=1e-12)
        @test isapprox(gs.modularity_by_component, 0.0; atol=1e-12)
end

@testset "Covariance-based anisotropy and eccentricity (axis backend exact moments)" begin
    # Single-region encoding map (no thresholds, no constraints).
    coords = Vector{Vector{Float64}}([Float64[], Float64[]])
    sig_y = BitVector[BitVector()]
    sig_z = BitVector[BitVector()]
    reps = [[0.0, 0.0]]
    pi = PLB.PLEncodingMapBoxes(coords, sig_y, sig_z, reps, PLB.BoxUpset[], PLB.BoxDownset[])

    # Intersect with a rectangle [-2,2] x [-0.5,0.5].
    box = ([-2.0, -0.5], [2.0, 0.5])

    pd = PM.region_principal_directions(pi, 1; box=box)
    # Variances: (width^2)/12.
    @test isapprox(pd.evals[1], 16.0/12.0; atol=1e-12)
    @test isapprox(pd.evals[2], 1.0/12.0; atol=1e-12)

    ani = PM.region_covariance_anisotropy(pi, 1; box=box)
    @test isapprox(ani, 16.0; atol=1e-12)

    ecc = PM.region_covariance_eccentricity(pi, 1; box=box)
    @test isapprox(ecc, sqrt(15.0/16.0); atol=1e-12)

    scores = PM.region_anisotropy_scores(pi, 1; box=box)
    @test isapprox(scores.ratio, 16.0; atol=1e-12)
    @test isapprox(scores.eccentricity, sqrt(15.0/16.0); atol=1e-12)
end

@testset "window_box and box=:auto" begin
    Ups = [PLB.BoxUpset([0.0])]
    Downs = [PLB.BoxDownset([2.0])]
    Phi = reshape(QQ[1], 1, 1)
    P, Hhat, pi = PLB.encode_fringe_boxes(Ups, Downs, Phi)

    wb = PM.window_box(pi)
    @test isapprox(wb[1][1], -1.2; atol=1e-12)
    @test isapprox(wb[2][1],  3.2; atol=1e-12)

    # Choose dims supported only on the middle region
    dims = zeros(Int, P.n)
    dims[PM.locate(pi, [1.0])] = 1

    opts_auto = PM.InvariantOptions(box=:auto)
    opts_wb   = PM.InvariantOptions(box=wb)

    m_auto = PM.integrated_hilbert_mass(dims, pi, opts_auto)
    m_exp  = PM.integrated_hilbert_mass(dims, pi, opts_wb)
    @test isapprox(m_auto, m_exp; atol=1e-12)
end


@testset "ZnEncodingMap adjacency and asymptotics" begin
    # 2D flange depending only on coordinate 1, free in coordinate 2
    FG = PM.Flange{PM.QQ}(2,
        [PM.flat([0,0], [false,true])],
        [PM.inj([1,0], [false,true])],
        [PM.interval(0,0), PM.interval(1,1), PM.interval(2,2)]
    )
    enc = PM.EncodingOptions(backend=:zn, max_regions=100)
    P, M, pi = PM.encode_pmodule_from_flange(FG, enc)

    a = [-10,-10]
    b = [ 10, 10]
    adj = PM.region_adjacency(pi; box=(a,b))
    @test !isempty(adj)
    @test all(k[1] < k[2] for k in keys(adj))  # canonical ordering

    # asymptotic growth: total measure in Z^2 should scale like R^2, interface like R^(2-1)=R
    dims = ones(Int, P.n)
    opts = PM.InvariantOptions(box=:auto, strict=true)
    A = PM.module_geometry_asymptotics(dims, pi, opts;
        scales=[1,2,4,8],
        include_interface=true
    )

    @test isfinite(A.exponent_total_measure)
    @test abs(A.exponent_total_measure - 2.0) < 0.25

    @test isfinite(A.exponent_interface_measure)
    @test abs(A.exponent_interface_measure - 1.0) < 0.35
end

@testset "PL asymptotics exponent check" begin
    # Quadrant split example from existing tests (2D PLBackend)
    Ups = [
        PM.BoxUpset{PM.QQ}([0,-10], [10,10]),
        PM.BoxUpset{PM.QQ}([-10,0], [10,10])
    ]
    Downs = PM.BoxUpset{PM.QQ}[]
    Phi = [PM.interval(0,0), PM.interval(1,1), PM.interval(2,2)]
    pi = PM.encode_fringe_boxes(Ups, Downs, Phi)

    dims = ones(Int, 4)  # 4 quadrants
    opts = PM.InvariantOptions(box=:auto, strict=true)
    A = PM.module_geometry_asymptotics(dims, pi, opts;
        scales=[1,2,4,8],
        include_interface=true
    )

    @test isfinite(A.exponent_total_measure)
    @test abs(A.exponent_total_measure - 2.0) < 0.25

    @test isfinite(A.exponent_integrated_hilbert_mass)
    @test abs(A.exponent_integrated_hilbert_mass - 2.0) < 0.25
end

@testset "Global module size summaries and region adjacency" begin

    @testset "1D box backend: integrated mass, histograms, adjacency" begin
        # 1D example: one upset x >= 0 and one downset x <= 2.
        # Inside any window, the parameter line splits into three regions:
        #   (-inf,0), (0,2), (2,inf)
        Ups = [PLB.BoxUpset([0.0])]
        Downs = [PLB.BoxDownset([2.0])]
        Phi = ones(QQ, 1, 1)

        P, H, pi = PLB.encode_fringe_boxes(Ups, Downs, Phi)
        box = ([-2.0], [7.0])
        opts = PM.InvariantOptions(box=box)

        # Identify the three regions by sampling points.
        t_left  = PM.locate(pi, [-1.0])
        t_mid   = PM.locate(pi, [ 1.0])
        t_right = PM.locate(pi, [ 3.0])

        # Region weights (lengths) inside the window:
        # left: [-2,0] has length 2, mid: [0,2] has length 2, right: [2,7] has length 5.
        w = PM.region_weights(pi; box=box)
        @test w[t_left]  == 2.0
        @test w[t_mid]   == 2.0
        @test w[t_right] == 5.0

        # region_bbox, widths, centroid
        bb_mid = PM.region_bbox(pi, t_mid; box=box)
        @test bb_mid[1][1] == 0.0
        @test bb_mid[2][1] == 2.0
        @test PM.region_widths(pi, t_mid; box=box) == [2.0]
        @test PM.region_centroid(pi, t_mid; box=box) == [1.0]
        @test PM.region_aspect_ratio(pi, t_mid; box=box) == 1.0

        # Integrated Hilbert mass:
        # Here dim=1 only on the middle region of length 2, so the integral is 2.
        mass = PM.integrated_hilbert_mass(H, pi, opts)
        @test mass == 2.0

        # Histogram of measure by module dimension.
        hist = PM.measure_by_dimension(H, pi, opts)
        @test hist[0] == 7.0
        @test hist[1] == 2.0

        # Support measure (dim > 0)
        supp = PM.support_measure(H, pi, opts)
        @test supp == 2.0

        # Basic weighted stats for dim
        stats = PM.dim_stats(H, pi, opts)
        @test stats.total_measure == 9.0
        @test stats.integrated_mass == 2.0
        @test abs(stats.mean - (2.0 / 9.0)) < 1e-12
        @test abs(stats.var - (14.0 / 81.0)) < 1e-12

        # L^p norms of dim
        @test PM.dim_norm(H, pi, opts; p=1) == 2.0
        @test abs(PM.dim_norm(H, pi, opts; p=2) - sqrt(2.0)) < 1e-12
        @test PM.dim_norm(H, pi, opts; p=Inf) == 1.0

        # Entropy of region weights: w = [2,2,5] up to permutation.
        p1 = 2.0 / 9.0
        p2 = 2.0 / 9.0
        p3 = 5.0 / 9.0
        expected_entropy = -(p1 * log(p1) + p2 * log(p2) + p3 * log(p3))
        @test abs(PM.region_weight_entropy(pi, opts) - expected_entropy) < 1e-12

        # Aspect ratio stats in 1D: all regions have aspect ratio 1.
        ar = PM.aspect_ratio_stats(pi, opts)
        @test ar.total_measure == 9.0
        @test ar.mean == 1.0
        @test ar.min == 1.0
        @test ar.max == 1.0

        # module_size_summary should be consistent.
        summary = PM.module_size_summary(H, pi, opts)
        @test summary.total_measure == 9.0
        @test summary.integrated_hilbert_mass == 2.0
        @test summary.support_measure == 2.0
        @test summary.measure_by_dimension[0] == 7.0
        @test summary.measure_by_dimension[1] == 2.0

        # Adjacency graph: there are exactly two interior boundaries (at 0 and 2),
        # so we expect two edges each with 0-dim measure 1.
        edges = PM.region_adjacency(pi; box=box)
        @test length(edges) == 2
        @test edges[(min(t_left, t_mid), max(t_left, t_mid))] == 1.0
        @test edges[(min(t_mid, t_right), max(t_mid, t_right))] == 1.0

        total_iface = PM.interface_measure(pi, opts)
        @test total_iface == 2.0

        by_pair = PM.interface_measure_by_dim_pair(H, pi, opts)
        @test by_pair[(0, 1)] == 2.0

        changed = PM.interface_measure_dim_changes(H, pi, opts)
        @test changed == 2.0
    end

    @testset "2D box backend: adjacency lengths on a quadrant split" begin
        # Create two upsets that induce thresholds at x=0 and y=0 within the
        # window [-1,1]^2.
        # This forces the encoding to split the window into four quadrants.
        Ups = [
            PLB.BoxUpset([0.0,  -10.0]),  # x >= 0 matters, y >= -10 always true in box
            PLB.BoxUpset([-10.0, 0.0])    # y >= 0 matters, x >= -10 always true in box
        ]
        Downs = PLB.BoxDownset[]  # none
        Phi = zeros(QQ, 0, length(Ups))

        P, H, pi = PLB.encode_fringe_boxes(Ups, Downs, Phi)
        box = ([-1.0, -1.0], [1.0, 1.0])

        # Regions (quadrants)
        t00 = PM.locate(pi, [-0.5, -0.5])
        t10 = PM.locate(pi, [ 0.5, -0.5])
        t01 = PM.locate(pi, [-0.5,  0.5])
        t11 = PM.locate(pi, [ 0.5,  0.5])

        # Each quadrant has bbox width [1,1], centroid at its midpoint.
        @test PM.region_widths(pi, t00; box=box) == [1.0, 1.0]
        @test PM.region_centroid(pi, t00; box=box) == [-0.5, -0.5]
        @test PM.region_aspect_ratio(pi, t00; box=box) == 1.0

        # Adjacency edges: four unit-length interfaces inside the box:
        # (left-bottom)-(right-bottom), (left-bottom)-(left-top),
        # (right-bottom)-(right-top), (left-top)-(right-top).
        edges = PM.region_adjacency(pi; box=box)
        @test length(edges) == 4

        expected_pairs = [(t00, t10), (t00, t01), (t10, t11), (t01, t11)]
        for (u, v) in expected_pairs
            key = (min(u, v), max(u, v))
            @test abs(edges[key] - 1.0) < 1e-12
        end
    end

    @testset "Module size summary - polyhedral backend smoke test" begin
        # Allow passing a plain vector of region dimensions as "the module".
        dims = [3]
        pi = PLB.PLEncodingMapBoxes(
            Vector{Vector{Float64}}(),    # coords (no thresholds)
            BitVector[BitVector()],       # sig_y
            BitVector[BitVector()],       # sig_z
            [Float64[]],                  # reps
            PLB.BoxUpset[],
            PLB.BoxDownset[],
        )

        box = ([-2.0], [7.0])
        opts = PM.InvariantOptions(box=box)
        summary = PM.module_size_summary(dims, pi, opts; weights=[1.0])
        @test summary.integrated_hilbert_mass == 3.0
        @test summary.total_measure == 1.0
    end
end

@testset "uncertainty + support geometry + ehrhart" begin
    # --- PLBackend 1D encoding as in region geometry tests ---
    grid = PM.PLEndpoints([[-2.0, 0.0, 2.0, 7.0]])
    cells = [
        PM.BoxCell([-2.0], [0.0]),
        PM.BoxCell([0.0], [2.0]),
        PM.BoxCell([2.0], [7.0]),
    ]
    pi = PM.PLEncodingMapBoxes(grid, cells, 3)
    box = ([-2.0], [7.0])
    opts = PM.InvariantOptions(box=box)

    # 1) region_weights with return_info on exact backend
    info = PM.region_weights(pi; box=box, return_info=true)
    @test info.method == :exact
    @test length(info.weights) == 3
    @test isapprox(sum(info.weights), 9.0)
    @test all(info.stderr .== 0.0)
    @test all(c[1] == c[2] for c in info.ci)

    # 2) support geometry
    H1 = [0, 1, 0]
    sm = PM.support_measure_stats(H1, pi, opts; min_dim=1)
    @test isapprox(sm.estimate, 2.0)
    @test sm.stderr == 0.0
    @test sm.ci[1] == sm.ci[2] == 2.0

    H2 = [1, 1, 0]
    comps = PM.support_components(H2, pi, opts; min_dim=1)
    @test length(comps) == 1
    @test comps[1] == [1, 2]

    diams, overall = PM.support_graph_diameter(H2, pi, opts; min_dim=1)
    @test overall == 1
    @test diams[1] == 1

    H3 = [1, 0, 1]
    comps3 = PM.support_components(H3, pi, opts; min_dim=1)
    @test length(comps3) == 2
    @test comps3[1] in ([1], [3])
    @test comps3[2] in ([1], [3])

    bb = PM.support_bbox(H3, pi, opts; min_dim=1)
    @test isapprox(bb[1][1], -2.0)
    @test isapprox(bb[2][1], 7.0)
    @test isapprox(PM.support_geometric_diameter(H3, pi, opts; min_dim=1, metric=:Linf), 9.0)

    # --- PLPolyhedra MC region weights sanity check (2D halfspaces) ---
    A1 = [0//1  -1//1]
    b1 = [0//1]
    A2 = [0//1   1//1]
    b2 = [0//1]
    hp1 = PLP.make_hpoly(A1, b1)
    hp2 = PLP.make_hpoly(A2, b2)
    reps = [[0.0,  1.0], [0.0, -1.0]]
    pi2 = PLP.PLEncodingMap([hp1, hp2], reps, 2)
    box2 = ([-2.0, -2.0], [2.0, 2.0])
    rng = MersenneTwister(1)

    mcinfo = PM.region_weights(pi2; box=box2, method=:mc, nsamples=20000, rng=rng, return_info=true)
    @test mcinfo.method == :mc
    @test length(mcinfo.weights) == 2
    @test all(mcinfo.stderr .>= 0.0)
    @test all(ci[1] <= w <= ci[2] for (w,ci) in zip(mcinfo.weights, mcinfo.ci))
    @test isapprox(sum(mcinfo.weights), mcinfo.total_volume)

    # 3) anisotropy stability (ratio should be > 1 for half-box)
    rng2 = MersenneTwister(2)
    an = PM.region_anisotropy_scores(pi2, 1; box=box2, nsamples=10000, rng=rng2, return_info=true, nbatches=5)
    @test an.ratio > 1.0
    @test !isnan(an.ratio_stderr)

    # --- Ehrhart-like fit on ZnEncodingMap (period=1) ---
    FZ = PM.FlangeZn
    flats = [FZ.IndFlat(FZ.face(2, [true, true]), [0, 0]; id=:F)]  # tau = all free => no cuts, but sets n=2
    injs  = FZ.IndInj[]
    Phi   = zeros(QQ, 0, 1)
    FG    = PM.Flange{QQ}(2, flats, injs, Phi)

    enc = PM.EncodingOptions(backend=:zn, max_regions=10)
    P, M, piZ = PM.encode_pmodule_from_flange(FG, enc)

    base_box = ([-2, -2], [2, 2])
    scales = [1, 2, 3, 4]

    dims = ones(Int, P.n)

    opts = PM.InvariantOptions(box=base_box)
    asym = PM.module_geometry_asymptotics(dims, piZ, opts;
        scales=scales,
        include_interface=false,
        include_ehrhart=true,
        ehrhart_period=1
    )

    @test haskey(asym, :ehrhart_total_measure)
    fit = asym.ehrhart_total_measure
    @test fit.period == 1
    @test fit.fits[1] !== nothing

    c = fit.fits[1].coeffs
    @test isapprox(c[1], 1.0; atol=1e-8)
    @test isapprox(c[2], 8.0; atol=1e-8)
    @test isapprox(c[3], 16.0; atol=1e-8)

end