using SparseArrays

@testset "Visualization engine v1" begin
    TOA = TamerOp.Advanced
    VIZ = TamerOp.Visualization
    EC = TamerOp.EncodingCore
    ENC = TamerOp.Encoding
    RES = TamerOp.Results
    CO = TamerOp.ChangeOfPosets
    F2D = TamerOp.Fibered2D
    SMO = TamerOp.SignedMeasures
    MPI = TamerOp.MultiparameterImages
    FZ = TamerOp.FlangeZn
    CM = TamerOp.CoreModules
    FF = TamerOp.FiniteFringe
    PLB = TamerOp.PLBackend
    OPT = TamerOp.Options
    DT = TamerOp.DataTypes
    DI = TamerOp.DataIngestion
    SI = TamerOp.SliceInvariants

    Pgrid = FF.ProductOfChainsPoset((2, 2))
    grid_pi = EC.GridEncodingMap(Pgrid, ([0.0, 1.0], [0.0, 1.0]))
    compiled_grid = EC.compile_encoding(Pgrid, grid_pi)
    enc_result = RES.EncodingResult(Pgrid, nothing, compiled_grid)
    Pbox, _, box_pi = PLB.encode_fringe_boxes(
        [PLB.BoxUpset([0.0, -10.0]), PLB.BoxUpset([1.0, -10.0])],
        PLB.BoxDownset[],
        TOA.EncodingOptions(),
    )
    compiled_box = EC.compile_encoding(Pbox, box_pi)
    r_left = EC.locate(box_pi, [0.5, 0.0])
    r_right = EC.locate(box_pi, [2.0, 0.0])
    Hbox = TOA.one_by_one_fringe(Pbox,
                                 FF.principal_upset(Pbox, r_left),
                                 FF.principal_downset(Pbox, r_right),
                                 1)
    Mbox = TOA.pmodule_from_fringe(Hbox)
    enc_box_result = RES.EncodingResult(Pbox, Mbox, compiled_box)
    cdr_box = RES.CohomologyDimsResult(Pbox, [0, 1, 2], compiled_box; degree=1)
    rank_inv = TamerOp.invariant(enc_box_result; which=:rank_invariant)
    hilbert_inv = TamerOp.invariant(enc_box_result; which=:restricted_hilbert)
    rank_raw = TamerOp.rank_invariant(Mbox)
    opts_box = OPT.InvariantOptions(box=([-1.0, -1.0], [2.0, 1.0]), strict=true)
    arr_box = F2D.fibered_arrangement_2d(box_pi, opts_box; normalize_dirs=:L1, include_axes=true, precompute=:cells, threads=false)
    Hbox_alt = TOA.one_by_one_fringe(Pbox,
                                     FF.principal_upset(Pbox, r_right),
                                     FF.principal_downset(Pbox, r_right),
                                     1)
    Nbox = TOA.pmodule_from_fringe(Hbox_alt)
    cache_box = F2D.fibered_barcode_cache_2d(Mbox, arr_box; precompute=:none, threads=false)
    cache_box_alt = F2D.fibered_barcode_cache_2d(Nbox, arr_box; precompute=:none, threads=false)
    fam_box = F2D.fibered_slice_family_2d(arr_box)

    line = MPI.MPPLineSpec([1.0, 1.0], 0.0, [0.0, 0.0], 0.5)
    line2 = MPI.MPPLineSpec([1.0, 0.5], 0.25, [0.0, 0.25], 0.4)
    line3 = MPI.MPPLineSpec([0.5, 1.0], -0.1, [0.1, 0.0], 0.4)
    decomp = MPI.MPPDecomposition(
        [line, line2, line3],
        [
            [([0.0, 0.0], [1.0, 1.0], 0.5)],
            [([0.0, 0.5], [1.0, 0.8], 0.4), ([0.2, 0.0], [1.0, 0.6], 0.7)],
            [([0.0, 0.2], [0.8, 1.0], 0.6)],
        ],
        [0.25, 0.75, 0.5],
        ([0.0, 0.0], [1.0, 1.0]),
    )
    img = MPI.MPPImage([0.0, 1.0], [0.0, 1.0], [1.0 2.0; 3.0 4.0], 0.25, decomp)
    L = MPI.MPLandscape(2,
                        [0.0, 0.5, 1.0],
                        reshape(Float64[0.1, 0.2, 0.3, 0.0, 0.1, 0.0,
                                        0.2, 0.1, 0.0, 0.3, 0.2, 0.1,
                                        0.0, 0.1, 0.2, 0.1, 0.0, 0.0,
                                        0.3, 0.1, 0.1, 0.2, 0.2, 0.2], 2, 2, 2, 3),
                        fill(0.25, 2, 2),
                        [[1.0, 0.0], [0.0, 1.0]],
                        [-0.5, 0.5])

    sb = SMO.RectSignedBarcode((collect(1:3), collect(1:3)),
                               [SMO.Rect{2}((1, 1), (2, 2)), SMO.Rect{2}((2, 1), (2, 2))],
                               [2, -1])
    pm = SMO.PointSignedMeasure((collect(1:3), collect(1:3)), [(1, 1), (3, 2)], [1, -2])
    smd = SMO.SignedMeasureDecomposition(rectangles=sb, euler_signed_measure=pm, mpp_image=img)

    slice_single = SI.SliceBarcodesResult([Dict((0.0, 1.0) => 1)], [1.0], [[1.0, 1.0]], [0.0])
    slice_bank = SI.SliceBarcodesResult(reshape([Dict((0.0, 1.0) => 1), Dict((0.5, 1.5) => 2)], 1, 2),
                                        reshape([0.5, 0.5], 1, 2), Any[], Any[])

    parr = F2D.projected_arrangement(Pgrid, [0.0, 1.0, 2.0, 3.0])
    parr_grid = F2D.projected_arrangement(compiled_grid; dirs=[(1.0, 0.0), (0.0, 1.0)], include_axes=true, threads=false)
    pdres = F2D.ProjectedDistancesResult([0.2, 0.4], [1, 2], [(1.0, 0.0), (0.0, 1.0)], :bottleneck)
    pbres = F2D.ProjectedBarcodesResult([Dict((0.0, 1.0) => 1), Dict((0.5, 1.5) => 1)], [1, 2], [(1.0, 0.0), (0.0, 1.0)])

    opts = OPT.InvariantOptions(box=([0.0, 0.0], [2.0, 2.0]))
    arr = F2D.fibered_arrangement_2d(grid_pi, opts; include_axes=true, precompute=:cells, threads=false)
    fam = F2D.fibered_slice_family_2d(arr)
    Hgrid = TOA.one_by_one_fringe(Pgrid,
                                  FF.principal_upset(Pgrid, 2),
                                  FF.principal_downset(Pgrid, 4),
                                  1)
    Mgrid = TOA.pmodule_from_fringe(Hgrid)
    cache_grid = F2D.fibered_barcode_cache_2d(Mgrid, arr; precompute=:none, threads=false)
    slice_res = F2D.fibered_slice(cache_grid, (1.0, 1.0), 0.0)

    face = FZ.Face(2, [false, false])
    qq = CM.QQField()
    FG = FZ.Flange(2,
                   [FZ.IndFlat(face, (0, 0); id=:U1)],
                   [FZ.IndInj(face, (1, 1); id=:D1)],
                   reshape([CM.coerce(qq, 1)], 1, 1);
                   field=qq)

    Qmap = FF.ProductOfChainsPoset((2, 2))
    Pmap = FF.ProductOfChainsPoset((2, 2))
    emap = ENC.EncodingMap(Qmap, Pmap, [1, 2, 3, 4])
    trans = RES.ModuleTranslationResult(:pushforward_left, nothing, emap)

    Pcommon = FF.ProductPoset(Pmap, Pmap)
    n1 = FF.nvertices(Pmap)
    pi_left = ENC.EncodingMap(Pcommon, Pmap, [((q - 1) % n1) + 1 for q in 1:FF.nvertices(Pcommon)])
    pi_right = ENC.EncodingMap(Pcommon, Pmap, [div(q - 1, n1) + 1 for q in 1:FF.nvertices(Pcommon)])
    cref = CO.CommonRefinementTranslationResult(Pcommon, (nothing, nothing), pi_left, pi_right)

    pc2 = DT.PointCloud([0.0 0.0; 1.0 0.5; 2.0 1.0; 3.0 1.5])
    pc3 = DT.PointCloud([0.0 0.0 0.0; 1.0 0.5 0.25; 2.0 1.0 0.5; 3.0 1.5 0.75])
    g3 = DT.GraphData(4, [(1, 2), (2, 3), (3, 4)];
                      coords=[0.0 0.0 0.0; 1.0 0.5 0.25; 2.0 1.0 0.5; 3.0 1.5 0.75],
                      weights=[0.2, 0.5, 0.9])
    epg = DT.EmbeddedPlanarGraph2D([[0.0, 0.0], [1.0, 0.75], [2.0, 0.0]],
                                   [(1, 2), (2, 3)];
                                   polylines=[[(0.0, 0.0), (1.0, 0.75)], [(1.0, 0.75), (2.0, 0.0)]],
                                   bbox=(0.0, 0.0, 2.0, 1.0))
    img2 = DT.ImageNd(reshape(collect(1.0:16.0), 4, 4))
    img3 = DT.ImageNd(rand(4, 4, 3))
    Bgc = SparseArrays.sparse(Int[1, 2], Int[1, 2], Int[1, 1], 2, 2)
    gc = DT.GradedComplex([[1, 2], [3, 4]], [Bgc],
                          [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (2.0, 1.0)])
    st = DT.SimplexTreeMulti([1, 2, 4, 7], [1, 1, 2, 1, 2, 3], [0, 1, 2],
                             [1, 2, 3, 4], [1, 2, 3, 4],
                             [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)])
    est = DI.IngestionEstimate((n_cells_est=12,
                                cell_counts_by_dim=[3, 4, 5],
                                axis_sizes=[5, 6],
                                poset_size=30,
                                nnz_est=18,
                                dense_bytes_est=256,
                                warnings=String[]))
    plan = DI.IngestionPlan(pc2,
                            OPT.FiltrationSpec(kind=:rips),
                            OPT.FiltrationSpec(kind=:rips),
                            OPT.ConstructionOptions(output_stage=:graded_complex,
                                                    budget=(20, 12, 512)),
                            OPT.PipelineOptions(),
                            :graded_complex,
                            CM.QQField(),
                            nothing,
                            est,
                            :point_cloud_sparse,
                            :multi,
                            :first,
                            true)
    plan_no_pre = DI.IngestionPlan(pc2,
                                   OPT.FiltrationSpec(kind=:rips),
                                   OPT.FiltrationSpec(kind=:rips),
                                   OPT.ConstructionOptions(output_stage=:graded_complex,
                                                           budget=(nothing, nothing, nothing)),
                                   OPT.PipelineOptions(),
                                   :graded_complex,
                                   CM.QQField(),
                                   nothing,
                                   nothing,
                                   :point_cloud_sparse,
                                   :multi,
                                   :first,
                                   true)
    gcres = DI.GradedComplexBuildResult(gc, [[0.0, 1.0], [0.0, 1.0]], (:increasing, :increasing))
    codres = DI.point_codensity(pc2, DI.RipsCodensityFiltration(max_dim=1, knn=2, dtm_mass=0.5, nn_backend=:bruteforce))

    @test TamerOp.available_visuals(nothing) == ()
    @test TamerOp.available_visuals(grid_pi) == (:regions, :region_labels, :query_overlay)
    @test TamerOp.available_visuals(compiled_grid) == (:regions, :region_labels, :query_overlay)
    @test TamerOp.available_visuals(enc_result) == (:regions, :region_labels, :query_overlay)
    @test TamerOp.available_visuals(emap) == (:regions, :region_labels, :pushforward_overlay)
    @test TamerOp.available_visuals(cref) == (:common_refinement,)
    @test TamerOp.available_visuals(trans) == (:pushforward_overlay,)
    @test TamerOp.available_visuals(img) == (:mpp_image,)
    @test TamerOp.available_visuals(L) == (:mp_landscape, :landscape_slices)
    @test TamerOp.available_visuals(sb) == (:rectangles, :density_image)
    @test TamerOp.available_visuals(pm) == (:signed_atoms,)
    @test TamerOp.available_visuals(slice_single) == (:barcode, :barcode_bank, :slice_family)
    @test TamerOp.available_visuals(arr) == (:fibered_arrangement, :fibered_query, :fibered_cell_highlight, :fibered_tie_break, :fibered_offset_intervals, :fibered_projected_comparison)
    @test TamerOp.available_visuals(fam) == (:fibered_family, :fibered_chain_cells, :fibered_family_contributions, :fibered_distance_diagnostic)
    @test TamerOp.available_visuals(slice_res) == (:fibered_slice, :fibered_slice_overlay, :barcode)
    @test TamerOp.available_visuals(cache_grid) == (:fibered_arrangement, :fibered_query, :fibered_cell_highlight, :fibered_tie_break, :fibered_offset_intervals, :fibered_projected_comparison, :fibered_family, :fibered_chain_cells, :fibered_family_contributions, :fibered_distance_diagnostic, :fibered_query_barcode)
    @test TamerOp.available_visuals(parr) == (:projected_arrangement,)
    @test TamerOp.available_visuals(pdres) == (:projected_distances,)
    @test TamerOp.available_visuals(FG) == (:regions, :constant_subdivision)
    @test TamerOp.available_visuals(pc2) == (:points_2d, :point_density, :knn_graph, :radius_graph)
    @test TamerOp.available_visuals(pc3) == (:points_3d, :points_2d, :point_density, :knn_graph, :radius_graph)
    @test TamerOp.available_visuals(codres) == (:points_2d, :point_density, :knn_graph, :radius_graph, :codensity_radius_snapshots)
    @test TamerOp.available_visuals(g3) == (:graph_3d, :graph, :weighted_graph)
    @test TamerOp.available_visuals(epg) == (:embedded_planar_graph,)
    @test TamerOp.available_visuals(img2) == (:image,)
    @test TamerOp.available_visuals(img3) == (:image, :channels, :slice_viewer)
    @test TamerOp.available_visuals(est) == (:simplex_counts,)
    @test TamerOp.available_visuals(plan) == ()
    @test TamerOp.available_visuals(gcres) == ()
    @test TamerOp.available_visuals(gc) == ()
    @test TamerOp.available_visuals(st) == (:simplex_counts,)
    @test TamerOp.available_visuals(rank_raw) == (:rank_heatmap, :rank_rectangles)
    @test TamerOp.available_visuals(rank_inv) == (:rank_heatmap, :rank_rectangles, :rank_query_overlay)
    @test TamerOp.available_visuals(cdr_box) == (:cohomology_support_plane, :cohomology_support)
    @test TamerOp.available_visuals(hilbert_inv) == (:hilbert_heatmap, :hilbert_bars, :restricted_hilbert_curve)

    report_bad_kind = TOA.check_visual_request(grid_pi; kind=:fibered_arrangement, throw=false)
    @test !report_bad_kind.valid
    @test_throws ArgumentError TOA.check_visual_request(grid_pi; kind=:fibered_arrangement, throw=true)

    report_missing_query = TOA.check_visual_request(grid_pi; kind=:query_overlay, throw=false)
    @test !report_missing_query.valid
    @test_throws ArgumentError TOA.check_visual_request(grid_pi; kind=:query_overlay, throw=true)

    report_missing_landscape = TOA.check_visual_request(L; kind=:landscape_slices, throw=false)
    @test !report_missing_landscape.valid
    @test_throws ArgumentError TOA.check_visual_request(L; kind=:landscape_slices, throw=true)

    report_missing_tie = TOA.check_visual_request(arr; kind=:fibered_tie_break, throw=false)
    @test !report_missing_tie.valid
    @test_throws ArgumentError TOA.check_visual_request(arr; kind=:fibered_tie_break, throw=true)

    report_tie_ok = TOA.check_visual_request(arr_box; kind=:fibered_tie_break, dir=[1.0, 1.0], offset=0.0, throw=false)
    @test report_tie_ok.valid

    report_missing_slice_overlay = TOA.check_visual_request(slice_res; kind=:fibered_slice_overlay, throw=false)
    @test !report_missing_slice_overlay.valid
    @test_throws ArgumentError TOA.check_visual_request(slice_res; kind=:fibered_slice_overlay, throw=true)

    report_offset_ok = TOA.check_visual_request(arr_box; kind=:fibered_offset_intervals, dir=[1.0, 1.0], throw=false)
    @test report_offset_ok.valid

    report_cod_ok = TOA.check_visual_request(codres; kind=:codensity_radius_snapshots,
                                             radii=[0.1, 0.2], codensity_levels=:quantiles, throw=false)
    @test report_cod_ok.valid
    report_cod_bad = TOA.check_visual_request(codres; kind=:codensity_radius_snapshots,
                                              radii=[-0.1], codensity_levels=:bad, throw=false)
    @test !report_cod_bad.valid
    @test_throws ArgumentError TOA.check_visual_request(codres; kind=:codensity_radius_snapshots,
                                                        radii=[-0.1], codensity_levels=:bad, throw=true)

    report_missing_query_barcode = TOA.check_visual_request(cache_grid; kind=:fibered_query_barcode, throw=false)
    @test !report_missing_query_barcode.valid
    @test_throws ArgumentError TOA.check_visual_request(cache_grid; kind=:fibered_query_barcode, throw=true)

    report_family_missing = TOA.check_visual_request(fam_box; kind=:fibered_family_contributions, throw=false)
    @test !report_family_missing.valid
    @test_throws ArgumentError TOA.check_visual_request(fam_box; kind=:fibered_family_contributions, throw=true)

    report_family_ok = TOA.check_visual_request(fam_box; kind=:fibered_family_contributions,
                                                caches=(cache_box, cache_box_alt), throw=false)
    @test report_family_ok.valid

    report_projected_ok = TOA.check_visual_request(arr; kind=:fibered_projected_comparison,
                                                   projected=parr_grid, throw=false)
    @test report_projected_ok.valid

    report_removed_preflight = TOA.check_visual_request(plan; kind=:preflight_diagnostics, throw=false)
    @test !report_removed_preflight.valid
    @test_throws ArgumentError TOA.check_visual_request(plan; kind=:preflight_diagnostics, throw=true)

    report_removed_dashboard = TOA.check_visual_request(gcres; kind=:complex_dashboard, throw=false)
    @test !report_removed_dashboard.valid
    @test_throws ArgumentError TOA.check_visual_request(gcres; kind=:complex_dashboard, throw=true)

    report_bad_decomp_layout = TOA.check_visual_request(decomp; kind=:mpp_decomposition, layout=:bad, throw=false)
    @test !report_bad_decomp_layout.valid
    @test_throws ArgumentError TOA.check_visual_request(decomp; kind=:mpp_decomposition, layout=:bad, throw=true)

    report_missing_rank_pairs = TOA.check_visual_request(rank_inv; kind=:rank_query_overlay, throw=false)
    @test !report_missing_rank_pairs.valid
    @test_throws ArgumentError TOA.check_visual_request(rank_inv; kind=:rank_query_overlay, throw=true)

    spec_grid = TOA.visual_spec(grid_pi; kind=:query_overlay, points=[(0.25, 0.25), (0.75, 0.75)])
    @test TOA.visual_kind(spec_grid) == :query_overlay
    @test spec_grid.layers[1] isa TOA.RectLayer
    @test length(spec_grid.layers[1].rects) == 4
    @test spec_grid.layers[2] isa TOA.PointLayer
    @test length(spec_grid.layers[2].points) == 2
    @test TOA.check_visual_spec(spec_grid).valid

    spec_compiled = TOA.visual_spec(compiled_grid; kind=:regions)
    @test TOA.visual_kind(spec_compiled) == :regions

    spec_result = TOA.visual_spec(enc_result; kind=:region_labels)
    @test TOA.visual_kind(spec_result) == :region_labels

    spec_box = TOA.visual_spec(box_pi; kind=:region_labels)
    @test TOA.visual_kind(spec_box) == :region_labels
    @test all(isfinite, spec_box.axes.xlimits)
    @test all(isfinite, spec_box.axes.ylimits)
    @test TOA.visual_metadata(spec_box).figure_size == (860, 620)
    @test TOA.visual_metadata(spec_box).legend_position == :right
    spec_box_regions = TOA.visual_spec(box_pi; kind=:regions)
    @test spec_box_regions.legend.visible
    rect_layers = [layer for layer in spec_box.layers if layer isa TOA.RectLayer]
    @test length(rect_layers) == 3
    @test sum(length(layer.rects) for layer in rect_layers) == 6
    @test length(unique(layer.fill_color for layer in rect_layers)) == 3
    segment_layers = [layer for layer in spec_box.layers if layer isa TOA.SegmentLayer]
    @test length(segment_layers) == 3
    rep_layer = only(layer for layer in spec_box.layers if layer isa TOA.PointLayer)
    @test rep_layer.points == [(-1.0, -11.0), (0.5, -9.0), (2.0, -9.0)]
    @test rep_layer.markersize == 5.5
    label_layer = only(layer for layer in spec_box.layers if layer isa TOA.TextLayer)
    @test label_layer.labels == ["1", "2", "3"]
    @test all(isapprox(p[1], q[1]; atol=1e-12) && isapprox(p[2], q[2]; atol=1e-12)
              for (p, q) in zip(label_layer.positions, [(-0.93, -10.96), (0.57, -8.96), (2.07, -8.96)]))

    spec_rank_heat = TOA.visual_spec(rank_raw; kind=:rank_heatmap)
    @test spec_rank_heat.layers[1] isa TOA.HeatmapLayer
    @test size(spec_rank_heat.layers[1].values, 1) == FF.nvertices(Pbox)
    @test all(begin
                  if FF.leq(Pbox, a, b)
                      isapprox(spec_rank_heat.layers[1].values[b, a], float(TOA.value_at(rank_raw, a, b)); atol=1e-12)
                  else
                      isnan(spec_rank_heat.layers[1].values[b, a])
                  end
              end for a in 1:FF.nvertices(Pbox), b in 1:FF.nvertices(Pbox))
    @test TOA.visual_metadata(spec_rank_heat).figure_size == (720, 620)

    spec_rank_rect = TOA.visual_spec(rank_inv; kind=:rank_rectangles)
    @test any(layer -> layer isa TOA.RectLayer, spec_rank_rect.layers)
    @test spec_rank_rect.legend.visible
    @test TOA.visual_metadata(spec_rank_rect).legend_position == :right

    spec_rank_query = TOA.visual_spec(rank_inv; kind=:rank_query_overlay,
                                      pairs=[((0.5, -9.0), (2.0, -9.0))])
    @test TOA.visual_kind(spec_rank_query) == :rank_query_overlay
    @test any(layer -> layer isa TOA.SegmentLayer, spec_rank_query.layers)
    @test count(layer -> layer isa TOA.PointLayer, spec_rank_query.layers) >= 2
    rank_query_labels = [layer for layer in spec_rank_query.layers if layer isa TOA.TextLayer]
    @test any(lbl -> any(startswith(x, "r1 = ") for x in lbl.labels), rank_query_labels)

    spec_cdr_support_plane = TOA.visual_spec(cdr_box; kind=:cohomology_support_plane)
    @test TOA.visual_kind(spec_cdr_support_plane) == :cohomology_support_plane
    @test TOA.visual_metadata(spec_cdr_support_plane).degree == 1
    @test TOA.visual_metadata(spec_cdr_support_plane).support_count == 2
    @test TOA.visual_metadata(spec_cdr_support_plane).max_dim == 2
    @test spec_cdr_support_plane.legend.visible
    @test spec_cdr_support_plane.axes.aspect == :auto
    @test count(layer -> layer isa TOA.RectLayer, spec_cdr_support_plane.layers) == 3
    @test count(layer -> layer isa TOA.TextLayer, spec_cdr_support_plane.layers) == 0

    spec_cdr_support = TOA.visual_spec(cdr_box; kind=:cohomology_support)
    @test TOA.visual_kind(spec_cdr_support) == :cohomology_support
    @test TOA.visual_metadata(spec_cdr_support).degree == 1
    @test TOA.visual_metadata(spec_cdr_support).support_count == 2
    @test TOA.visual_metadata(spec_cdr_support).max_dim == 2
    @test spec_cdr_support.legend.visible
    @test count(layer -> layer isa TOA.RectLayer, spec_cdr_support.layers) == 3
    @test count(layer -> layer isa TOA.SegmentLayer, spec_cdr_support.layers) == 3
    cdr_labels = [layer for layer in spec_cdr_support.layers if layer isa TOA.TextLayer]
    @test length(cdr_labels) == 1
    @test only(cdr_labels).labels == ["1", "2"]

    spec_hilbert_heat = TOA.visual_spec(hilbert_inv; kind=:hilbert_heatmap)
    @test TOA.visual_kind(spec_hilbert_heat) == :hilbert_heatmap
    @test any(layer -> layer isa TOA.RectLayer, spec_hilbert_heat.layers)
    @test spec_hilbert_heat.legend.visible

    spec_hilbert_bars = TOA.visual_spec(hilbert_inv; kind=:hilbert_bars)
    @test spec_hilbert_bars.layers[1] isa TOA.RectLayer
    @test TOA.visual_metadata(spec_hilbert_bars).figure_size == (760, 460)

    spec_hilbert_curve = TOA.visual_spec(hilbert_inv; kind=:restricted_hilbert_curve)
    @test spec_hilbert_curve.layers[1] isa TOA.PolylineLayer
    @test spec_hilbert_curve.layers[2] isa TOA.PointLayer

    spec_map = TOA.visual_spec(emap; kind=:pushforward_overlay)
    @test spec_map.layers[1] isa TOA.SegmentLayer
    @test length(spec_map.layers[1].segments) == FF.nvertices(Qmap)

    spec_cref = TOA.visual_spec(cref; kind=:common_refinement)
    @test spec_cref.layers[1] isa TOA.PointLayer
    @test length(spec_cref.layers[1].points) == FF.nvertices(Pcommon)

    spec_bar = TOA.visual_spec(slice_single; kind=:barcode)
    @test spec_bar.layers[1] isa TOA.BarcodeLayer
    @test length(spec_bar.layers[1].intervals) == 1

    spec_bank = TOA.visual_spec(slice_bank; kind=:barcode_bank)
    @test spec_bank.layers[1] isa TOA.HeatmapLayer
    @test size(spec_bank.layers[1].values) == (1, 2)

    spec_farr = TOA.visual_spec(arr; kind=:fibered_arrangement)
    @test spec_farr.layers[1] isa TOA.PolylineLayer
    @test spec_farr.layers[2] isa TOA.PointLayer
    @test TOA.visual_metadata(spec_farr).figure_size == (780, 620)

    spec_fquery = TOA.visual_spec(arr; kind=:fibered_query, dir=[1.0, 1.0], offset=0.0)
    @test TOA.visual_kind(spec_fquery) == :fibered_query
    @test spec_fquery.axes.xlimits[1] < 0.0
    @test spec_fquery.axes.xlimits[2] > 2.0
    @test spec_fquery.axes.ylimits[1] < 0.0
    @test spec_fquery.axes.ylimits[2] > 2.0
    @test TOA.visual_metadata(spec_fquery).figure_size == (780, 620)

    spec_fcell = TOA.visual_spec(arr_box; kind=:fibered_cell_highlight, dir=[1.0, 1.0], offset=0.0)
    @test TOA.visual_kind(spec_fcell) == :fibered_cell_highlight
    @test count(layer -> layer isa TOA.PolylineLayer, spec_fcell.layers) >= 4
    @test TOA.visual_metadata(spec_fcell).figure_size == (860, 620)

    spec_ftie = TOA.visual_spec(arr_box; kind=:fibered_tie_break, dir=[1.0, 1.0], offset=0.0)
    @test TOA.visual_kind(spec_ftie) == :fibered_tie_break
    @test TOA.visual_metadata(spec_ftie).tie_break_relevant
    @test TOA.visual_metadata(spec_ftie).cell_up != TOA.visual_metadata(spec_ftie).cell_down
    @test count(layer -> layer isa TOA.PolylineLayer, spec_ftie.layers) >= 5

    spec_ffam = TOA.visual_spec(fam; kind=:fibered_family)
    @test spec_ffam.layers[2] isa TOA.PolylineLayer

    spec_fchain = TOA.visual_spec(fam_box; kind=:fibered_chain_cells)
    @test TOA.visual_kind(spec_fchain) == :fibered_chain_cells
    @test any(layer -> layer isa TOA.RectLayer, spec_fchain.layers)

    spec_fcontrib = TOA.visual_spec(fam_box; kind=:fibered_family_contributions, caches=(cache_box, cache_box_alt))
    @test TOA.visual_kind(spec_fcontrib) == :fibered_family_contributions
    @test length(TOA.visual_panels(spec_fcontrib)) == 2
    @test TOA.visual_metadata(spec_fcontrib).matching_distance >= 0.0

    spec_fdist = TOA.visual_spec(fam_box; kind=:fibered_distance_diagnostic, caches=(cache_box, cache_box_alt))
    @test TOA.visual_kind(spec_fdist) == :fibered_distance_diagnostic
    @test length(TOA.visual_panels(spec_fdist)) == 3
    @test TOA.visual_metadata(spec_fdist).argmax_index !== nothing

    spec_foffset = TOA.visual_spec(arr_box; kind=:fibered_offset_intervals, dir=[1.0, 1.0])
    @test TOA.visual_kind(spec_foffset) == :fibered_offset_intervals
    @test any(layer -> layer isa TOA.RectLayer, spec_foffset.layers)

    spec_slice = TOA.visual_spec(slice_res; kind=:fibered_slice)
    @test TOA.visual_kind(spec_slice) == :fibered_slice
    @test any(layer -> layer isa TOA.RectLayer, spec_slice.layers)
    @test any(layer -> layer isa TOA.SegmentLayer, spec_slice.layers)
    @test TOA.visual_metadata(spec_slice).chain_length == length(F2D.slice_chain(slice_res))
    @test TOA.visual_metadata(spec_slice).figure_size == (860, 520)
    spec_slice_overlay = TOA.visual_spec(slice_res; kind=:fibered_slice_overlay, arrangement=arr, dir=[1.0, 1.0], offset=0.0)
    @test TOA.visual_kind(spec_slice_overlay) == :fibered_slice_overlay
    @test any(layer -> layer isa TOA.SegmentLayer, spec_slice_overlay.layers)
    spec_slice_bar = TOA.visual_spec(slice_res; kind=:barcode)
    @test spec_slice_bar.layers[1] isa TOA.BarcodeLayer

    spec_query_bar = TOA.visual_spec(cache_grid; kind=:fibered_query_barcode, dir=[1.0, 1.0], offset=0.0)
    @test TOA.visual_kind(spec_query_bar) == :fibered_query_barcode
    @test length(TOA.visual_panels(spec_query_bar)) == 2

    spec_compare = TOA.visual_spec(arr; kind=:fibered_projected_comparison, projected=parr_grid)
    @test TOA.visual_kind(spec_compare) == :fibered_projected_comparison
    @test length(TOA.visual_panels(spec_compare)) == 2

    spec_parr = TOA.visual_spec(parr; kind=:projected_arrangement)
    @test spec_parr.layers[1] isa TOA.PointLayer

    spec_pd = TOA.visual_spec(pdres; kind=:projected_distances)
    @test spec_pd.layers[1] isa TOA.PolylineLayer
    @test length(spec_pd.layers[1].paths[1]) == 2

    spec_pb = TOA.visual_spec(pbres; kind=:barcode_bank)
    @test spec_pb.layers[1] isa TOA.HeatmapLayer
    @test size(spec_pb.layers[1].values) == (1, 2)

    spec_rect = TOA.visual_spec(sb; kind=:rectangles)
    @test any(layer -> layer isa TOA.RectLayer, spec_rect.layers)

    spec_atoms = TOA.visual_spec(pm; kind=:signed_atoms)
    @test spec_atoms.layers[1] isa TOA.PointLayer

    spec_decomp = TOA.visual_spec(smd; kind=:density_image)
    @test spec_decomp.layers[1] isa TOA.HeatmapLayer

    spec_line = TOA.visual_spec(line; kind=:mpp_line_spec)
    @test spec_line.layers[2] isa TOA.PolylineLayer

    spec_mpp_decomp = TOA.visual_spec(decomp; kind=:mpp_decomposition)
    @test TOA.visual_metadata(spec_mpp_decomp).layout == :overlay
    @test TOA.visual_metadata(spec_mpp_decomp).figure_size == (920, 620)
    @test TOA.visual_metadata(spec_mpp_decomp).legend_position == :right
    @test spec_mpp_decomp.layers[1] isa TOA.PolylineLayer
    decomp_layers = [layer for layer in spec_mpp_decomp.layers if layer isa TOA.PolylineLayer]
    @test length(decomp_layers) == 1 + MPI.nsummands(decomp)
    @test length(unique(layer.color for layer in decomp_layers[2:end])) == MPI.nsummands(decomp)
    @test decomp_layers[3].linewidth > decomp_layers[2].linewidth
    @test decomp_layers[3].alpha > decomp_layers[2].alpha

    spec_mpp_decomp_panels = TOA.visual_spec(decomp; kind=:mpp_decomposition, layout=:summands)
    @test TOA.visual_metadata(spec_mpp_decomp_panels).layout == :summands
    @test TOA.visual_metadata(spec_mpp_decomp_panels).figure_size == (1120, 560)
    @test isempty(TOA.visual_layers(spec_mpp_decomp_panels))
    @test length(TOA.visual_panels(spec_mpp_decomp_panels)) == MPI.nsummands(decomp)
    @test all(length(panel.layers) == 2 for panel in TOA.visual_panels(spec_mpp_decomp_panels))

    spec_img = TOA.visual_spec(img; kind=:mpp_image)
    @test spec_img.layers[1] isa TOA.HeatmapLayer
    @test size(spec_img.layers[1].values) == size(MPI.image_values(img))

    spec_land = TOA.visual_spec(L; kind=:mp_landscape)
    @test all(layer -> layer isa TOA.PolylineLayer, spec_land.layers)
    @test length(spec_land.layers) == MPI.landscape_layers(L)
    @test TOA.visual_metadata(spec_land).render_mode == :curves
    @test TOA.visual_metadata(spec_land).figure_size == (860, 520)

    spec_land_slice = TOA.visual_spec(L; kind=:landscape_slices, idir=1, ioff=1)
    @test spec_land_slice.layers[1] isa TOA.PolylineLayer
    @test length(spec_land_slice.layers[1].paths[1]) == length(MPI.landscape_grid(L))

    spec_flange_regions = TOA.visual_spec(FG; kind=:regions, box=([0.0, 0.0], [3.0, 3.0]))
    @test any(layer -> layer isa TOA.RectLayer, spec_flange_regions.layers)

    spec_flange_subdiv = TOA.visual_spec(FG; kind=:constant_subdivision, box=([0.0, 0.0], [3.0, 3.0]))
    @test spec_flange_subdiv.layers[1] isa TOA.HeatmapLayer
    @test size(spec_flange_subdiv.layers[1].values) == (3, 3)

    spec_pc2 = TOA.visual_spec(pc2; kind=:point_density, labels=["p1", "p2", "p3", "p4"])
    @test TOA.visual_kind(spec_pc2) == :point_density
    @test spec_pc2.layers[1] isa TOA.HeatmapLayer
    @test any(layer -> layer isa TOA.PointLayer, spec_pc2.layers)
    @test any(layer -> layer isa TOA.TextLayer, spec_pc2.layers)
    @test !spec_pc2.legend.visible
    @test spec_pc2.axes.xlabel == "x1"
    @test spec_pc2.axes.ylabel == "x2"
    @test TOA.visual_metadata(spec_pc2).figure_size == (760, 520)

    spec_cod_points = TOA.visual_spec(codres; kind=:points_2d)
    @test TOA.visual_kind(spec_cod_points) == :points_2d
    @test spec_cod_points.layers[1] isa TOA.PointLayer
    @test spec_cod_points.metadata.object == :point_codensity_result
    @test spec_cod_points.metadata.dtm_mass == 0.5
    @test spec_cod_points.metadata.neighbor_count == 3
    @test spec_cod_points.metadata.value_range == extrema(DI.codensity_values(codres))
    @test TOA.check_visual_spec(spec_cod_points).valid

    spec_cod_snap = TOA.visual_spec(codres; kind=:codensity_radius_snapshots,
                                    radii=[0.1, 0.2], codensity_levels=[1.0, 1.4])
    @test TOA.visual_kind(spec_cod_snap) == :codensity_radius_snapshots
    @test length(spec_cod_snap.panels) == 4
    @test spec_cod_snap.metadata.nlevels == 2
    @test spec_cod_snap.metadata.nradii == 2
    @test spec_cod_snap.metadata.panel_columns == 2
    @test spec_cod_snap.metadata.dtm_mass == 0.5
    @test spec_cod_snap.metadata.neighbor_count == 3
    @test all(occursin("c <=", panel.title) for panel in spec_cod_snap.panels)
    @test all(occursin("retained points", panel.subtitle) for panel in spec_cod_snap.panels)
    @test TOA.check_visual_spec(spec_cod_snap).valid

    spec_pc3 = TOA.visual_spec(pc3; kind=:points_3d)
    @test TOA.visual_kind(spec_pc3) == :points_3d
    @test spec_pc3.layers[1] isa TOA.Point3Layer
    @test TOA.check_visual_spec(spec_pc3).valid
    @test spec_pc3.axes.zlimits !== nothing
    @test !spec_pc3.legend.visible
    @test TOA.visual_metadata(spec_pc3).projected_dims == (1, 2, 3)

    spec_knn = TOA.visual_spec(pc2; kind=:knn_graph, k=2)
    @test any(layer -> layer isa TOA.SegmentLayer, spec_knn.layers)
    @test spec_knn.legend.visible
    @test TOA.visual_metadata(spec_knn).figure_size == (780, 560)
    @test TOA.visual_metadata(spec_knn).legend_position == :right
    @test isapprox((spec_knn.layers[1]::TOA.SegmentLayer).linewidth, 1.3)

    spec_graph = TOA.visual_spec(g3; kind=:weighted_graph)
    @test TOA.visual_kind(spec_graph) == :weighted_graph
    @test any(layer -> layer isa TOA.PointLayer, spec_graph.layers)
    @test count(layer -> layer isa TOA.SegmentLayer, spec_graph.layers) >= 1
    @test spec_graph.legend.visible

    spec_graph3 = TOA.visual_spec(g3; kind=:graph_3d)
    @test any(layer -> layer isa TOA.Point3Layer, spec_graph3.layers)
    @test any(layer -> layer isa TOA.Segment3Layer, spec_graph3.layers)

    spec_epg = TOA.visual_spec(epg; kind=:embedded_planar_graph)
    @test TOA.visual_kind(spec_epg) == :embedded_planar_graph
    @test any(layer -> layer isa TOA.PolylineLayer, spec_epg.layers)

    spec_img2 = TOA.visual_spec(img2; kind=:image)
    @test spec_img2.layers[1] isa TOA.HeatmapLayer
    @test spec_img2.layers[1].show_colorbar

    spec_img3 = TOA.visual_spec(img3; kind=:channels)
    @test length(TOA.visual_panels(spec_img3)) == 3
    @test all(panel -> panel.layers[1] isa TOA.HeatmapLayer, TOA.visual_panels(spec_img3))

    @test_throws ArgumentError TOA.visual_spec(plan; kind=:plan_dashboard)
    @test_throws ArgumentError TOA.visual_spec(est; kind=:preflight_diagnostics)
    @test_throws ArgumentError TOA.visual_spec(gc; kind=:complex_dashboard)
    @test_throws ArgumentError TOA.visual_spec(gcres; kind=:cell_histogram)
    @test_throws ArgumentError TOA.visual_spec(gcres; kind=:grade_ranges)
    @test_throws ArgumentError TOA.visual_spec(gc; kind=:cell_histogram)
    @test_throws ArgumentError TOA.visual_spec(gc; kind=:grade_ranges)
    @test_throws ArgumentError TOA.visual_spec(hilbert_inv; kind=:rank_heatmap)

    spec_st = TOA.visual_spec(st; kind=:simplex_counts)
    @test spec_st.layers[1] isa TOA.RectLayer

    desc = describe(spec_img)
    @test desc.kind == :visualization_spec
    @test TOA.visual_summary(spec_img) == desc
    @test TOA.visual_metadata(spec_img).image_shape == size(MPI.image_values(img))
    @test occursin("VisualizationSpec", repr(MIME("text/plain"), spec_img))
    @test occursin("<div", repr(MIME("text/html"), spec_img))

    bad_spec = TOA.VisualizationSpec(:bad;
                                     layers=TOA.AbstractVisualizationLayer[
                                         TOA.TextLayer(["a"], [(0.0, 0.0), (1.0, 1.0)], :black, 10.0),
                                     ])
    bad_report = TOA.check_visual_spec(bad_spec; throw=false)
    @test !bad_report.valid
    @test_throws ArgumentError TOA.check_visual_spec(bad_spec; throw=true)

    have_cairo = try
        import CairoMakie
        TamerOp.Visualization._try_load_visual_backend!(:cairomakie)
    catch
        false
    end
    if have_cairo
        fig_box = TamerOp.visualize(box_pi; kind=:regions, backend=:cairomakie)
        @test fig_box !== nothing
        fig_rank = TamerOp.visualize(rank_inv; kind=:rank_heatmap, backend=:cairomakie)
        @test fig_rank !== nothing
        fig_slice = TamerOp.visualize(slice_res; kind=:fibered_slice, backend=:cairomakie)
        @test fig_slice !== nothing
        fig_slice_overlay = TamerOp.visualize(slice_res; kind=:fibered_slice_overlay,
                                                   arrangement=arr, dir=[1.0, 1.0], offset=0.0,
                                                   backend=:cairomakie)
        @test fig_slice_overlay !== nothing
        fig_tie = TamerOp.visualize(arr_box; kind=:fibered_tie_break, dir=[1.0, 1.0], offset=0.0, backend=:cairomakie)
        @test fig_tie !== nothing
        fig_query_bar = TamerOp.visualize(cache_grid; kind=:fibered_query_barcode,
                                               dir=[1.0, 1.0], offset=0.0, backend=:cairomakie)
        @test fig_query_bar !== nothing
        fig_family_diag = TamerOp.visualize(fam_box; kind=:fibered_distance_diagnostic,
                                                 caches=(cache_box, cache_box_alt), backend=:cairomakie)
        @test fig_family_diag !== nothing
        fig_compare = TamerOp.visualize(arr; kind=:fibered_projected_comparison,
                                             projected=parr_grid, backend=:cairomakie)
        @test fig_compare !== nothing
        fig = TamerOp.visualize(img; backend=:cairomakie)
        @test fig !== nothing
        fig_panels = TamerOp.visualize(decomp; kind=:mpp_decomposition, layout=:summands, backend=:cairomakie)
        @test fig_panels !== nothing
        png_path = tempname() * ".png"
        TamerOp.save_visual(png_path, img; backend=:cairomakie)
        @test isfile(png_path)
        @test filesize(png_path) > 0
        png_export = TamerOp.save_visual(mktempdir(), "mpp_image_static", img; prefer=:static)
        @test png_export isa TamerOp.VisualExportResult
        @test TamerOp.export_format(png_export) == :png
        @test TamerOp.export_backend(png_export) == :cairomakie
        @test isfile(TamerOp.export_path(png_export))
    end

    have_wgl = try
        import WGLMakie
        TamerOp.Visualization._try_load_visual_backend!(:wglmakie)
    catch
        false
    end

    html_path = tempname() * ".html"
    html_saved = TamerOp.save_visual(html_path, img)
    @test html_saved == html_path
    @test isfile(html_path)
    @test filesize(html_path) > 0
    html_text = read(html_path, String)
    if have_wgl
        @test !occursin("VisualizationSpec", html_text)
    else
        @test occursin("VisualizationSpec", html_text)
    end

    export_dir = mktempdir()
    export_res = TamerOp.save_visual(export_dir, "mpp_image_export", img; format=:html)
    @test export_res isa TamerOp.VisualExportResult
    @test TamerOp.export_stem(export_res) == "mpp_image_export"
    @test TamerOp.export_kind(export_res) == :mpp_image
    @test TamerOp.export_format(export_res) == :html
    @test TamerOp.export_backend(export_res) in (:wglmakie, :spec_html)
    @test isfile(TamerOp.export_path(export_res))
    @test describe(export_res).kind == :visual_export_result
    @test occursin("VisualExportResult", repr(MIME("text/plain"), export_res))

    batch_dir = mktempdir()
    exports = TamerOp.save_visuals(batch_dir,
                                        [
                                            (; stem="mpp_image", obj=img, kind=:mpp_image),
                                            (; stem="mpp_decomposition", obj=decomp, kind=:mpp_decomposition, layout=:summands),
                                        ];
                                        format=:html)
    @test length(exports) == 2
    @test all(res -> res isa TamerOp.VisualExportResult, exports)
    @test [TamerOp.export_stem(res) for res in exports] == ["mpp_image", "mpp_decomposition"]
    @test all(res -> TamerOp.export_format(res) == :html, exports)
    @test all(res -> isfile(TamerOp.export_path(res)), exports)

    @test_throws ArgumentError TamerOp.save_visual(export_dir, "mpp_image_export.html", img)
    @test_throws ArgumentError TamerOp.save_visuals(export_dir, [(; obj=img)]; format=:html)

    save_doc = string(@doc TamerOp.save_visual)
    batch_doc = string(@doc TamerOp.save_visuals)
    @test occursin("save_visual(outdir, stem, obj", save_doc)
    @test occursin("save_visuals(outdir, requests", batch_doc)

    if have_wgl
        fig = TOA.render(spec_img; backend=:wglmakie)
        @test fig !== nothing
    end
end
