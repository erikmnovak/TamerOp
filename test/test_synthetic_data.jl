@testset "SyntheticData raw generators and families" begin
    pc = SD.noisy_annulus(n_annulus=24, n_noise=6, r_inner=0.5, r_outer=1.5, dim=2)
    @test pc isa DT.PointCloud
    @test DT.npoints(pc) == 30
    @test DT.ambient_dim(pc) == 2

    orbit = SD.coupled_orbit(npoints=12, r=1.7, x0=[0.1, 0.2])
    @test orbit isa DT.PointCloud
    @test DT.npoints(orbit) == 12
    @test DT.ambient_dim(orbit) == 2

    blobs = SD.gaussian_clusters(counts=[5, 7], centers=[[0.0, 0.0], [1.0, 1.0]], std=[0.05, 0.15])
    @test blobs isa DT.PointCloud
    @test DT.npoints(blobs) == 12

    img = SD.checkerboard_image(size=(6, 8), blocks=(2, 4), low=-1.0, high=2.0)
    @test img isa DT.ImageNd
    @test size(getfield(img, :data)) == (6, 8)
    @test Set(vec(getfield(img, :data))) == Set([-1.0, 2.0])

    graph = SD.planar_grid_graph(nrows=2, ncols=3, diagonals=false)
    @test graph isa DT.EmbeddedPlanarGraph2D
    @test DT.nvertices(graph) == 6
    @test DT.nedges(graph) == 7

    fam = SD.sweep_family(SD.noisy_annulus;
        sweep=(r_outer=[1.5, 2.0], n_noise=[0, 5]),
        n_annulus=12,
        r_inner=0.5,
        dim=2,
    )
    @test fam isa SD.SyntheticFamily
    @test length(fam) == 4
    @test SD.synthetic_generator(fam) == :noisy_annulus
    @test length(SD.synthetic_parameters(fam)) == 4
    @test length(SD.synthetic_labels(fam)) == 4
    @test SD.check_synthetic_family(fam).valid
    @test describe(fam).kind == :synthetic_family
    @test occursin("SyntheticFamily", sprint(show, fam))

    explicit = SD.synthetic_family(SD.checkerboard_image,
        [(; size=(4, 4), blocks=(2, 2), low=0.0, high=1.0),
         (; size=(6, 6), blocks=(3, 3), low=0.0, high=1.0)];
        generator=:checkerboard_suite,
    )
    @test explicit isa SD.SyntheticFamily
    @test SD.synthetic_generator(explicit) == :checkerboard_suite
    @test all(x -> x isa DT.ImageNd, SD.synthetic_items(explicit))

    @test TOA.SyntheticFamily === SD.SyntheticFamily
    @test TOA.noisy_annulus === SD.noisy_annulus
    @test TOA.sweep_family === SD.sweep_family
end

@testset "SyntheticData algebraic generators" begin
    M = SD.chain_bar_fringe(bars=[(2, 4), (3, 5)], n=6)
    @test M isa FF.FringeModule
    @test FF.ngenerators(M) == 2
    @test FF.nrelations(M) == 2
    @test FF.fiber_dimension(M, 1) == 0
    @test FF.fiber_dimension(M, 3) == 2
    @test FF.fiber_dimension(M, 5) == 1

    Mc = SD.coupled_chain_fringe()
    @test Mc isa FF.FringeModule
    @test size(FF.fringe_coefficients(Mc)) == (3, 3)
    @test !iszero(FF.fringe_coefficients(Mc)[1, 2])
    @test FF.fiber_dimension(Mc, 1) == 1
    @test FF.fiber_dimension(Mc, 3) == 3
    @test FF.fiber_dimension(Mc, 6) == 1

    Md = SD.diamond_fringe()
    @test Md isa FF.FringeModule
    @test FF.nvertices(TO.ambient_poset(Md)) == 4
    @test size(FF.fringe_coefficients(Md)) == (1, 2)
    @test FF.fiber_dimension(Md, 1) == 0
    @test FF.fiber_dimension(Md, 2) == 1
    @test FF.fiber_dimension(Md, 3) == 1
    @test FF.fiber_dimension(Md, 4) == 1

    B = SD.box_bar_fringe(bars=[([0.0, 0.0], [1.0, 1.0]), ([0.5, 0.5], [2.0, 2.0])])
    @test B isa SD.SyntheticBoxFringe
    @test DT.ambient_dim(B) == 2
    @test TO.birth_upsets(B) isa Vector{PLB.BoxUpset}
    @test TO.death_downsets(B) isa Vector{PLB.BoxDownset}
    @test size(TO.coefficient_matrix(B)) == (2, 2)
    @test SD.check_synthetic_box_fringe(B).valid
    @test describe(B).kind == :synthetic_box_fringe
    @test occursin("SyntheticBoxFringe", sprint(show, B))

    P, H, pi = PLB.encode_fringe_boxes(TO.birth_upsets(B), TO.death_downsets(B), TO.coefficient_matrix(B), OPT.EncodingOptions())
    @test H isa FF.FringeModule
    @test FF.nvertices(P) >= 1
    @test pi isa PLB.PLEncodingMapBoxes

    Bs = SD.staircase_box_fringe()
    @test Bs isa SD.SyntheticBoxFringe
    @test !iszero(TO.coefficient_matrix(Bs)[1, 2])
    @test PLB.contains(TO.birth_upsets(Bs)[2], [1.0, 1.0])
    @test PLB.contains(TO.death_downsets(Bs)[1], [1.0, 1.0])
    @test SD.check_synthetic_box_fringe(Bs).valid

    Fpl = SD.pl_box_fringe(bars=[([0.0, 0.0], [1.0, 1.0]), ([0.25, 0.5], [1.5, 2.0])])
    @test Fpl isa PLP.PLFringe
    @test PLP.check_pl_fringe(Fpl).valid

    Fpc = SD.coupled_pl_fringe()
    @test Fpc isa PLP.PLFringe
    @test !iszero(getfield(Fpc, :Phi)[1, 2])
    @test PLP.check_pl_fringe(Fpc).valid
    Ppl, Hpl, ppl = PLP.encode_from_PL_fringe(Fpc, OPT.EncodingOptions())
    @test Hpl isa FF.FringeModule
    @test FF.nvertices(Ppl) >= 1
    @test ppl isa PLP.PLEncodingMap

    FG = SD.orthant_bar_flange(bars=[([0, 0], [1, 1]), ([1, 0], [2, 2])])
    @test FG isa FZ.Flange
    @test FZ.check_flange(FG).valid
    @test FZ.dim_at(FG, (0, 0)) == 1
    @test FZ.dim_at(FG, (1, 1)) == 2

    Fmix = SD.mixed_face_flange()
    @test Fmix isa FZ.Flange
    @test !iszero(FZ.coefficient_matrix(Fmix)[1, 2])
    @test FZ.check_flange(Fmix).valid
    @test FZ.dim_at(Fmix, (0, 0)) == 2
    @test FZ.dim_at(Fmix, (2, 2)) == 3
    @test FZ.dim_at(Fmix, (3, 3)) == 1

    @test TOA.SyntheticBoxFringe === SD.SyntheticBoxFringe
    @test TOA.chain_bar_fringe === SD.chain_bar_fringe
    @test TOA.coupled_chain_fringe === SD.coupled_chain_fringe
    @test TOA.diamond_fringe === SD.diamond_fringe
    @test TOA.box_bar_fringe === SD.box_bar_fringe
    @test TOA.staircase_box_fringe === SD.staircase_box_fringe
    @test TOA.pl_box_fringe === SD.pl_box_fringe
    @test TOA.coupled_pl_fringe === SD.coupled_pl_fringe
    @test TOA.orthant_bar_flange === SD.orthant_bar_flange
    @test TOA.mixed_face_flange === SD.mixed_face_flange
end
