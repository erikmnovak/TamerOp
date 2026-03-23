using Test

# Included from test/runtests.jl; uses shared aliases (TO, PLP, PLB, FF, QQ, ...).

with_fields(FIELDS_FULL) do field
    K = CM.coeff_type(field)
    @inline c(x) = CM.coerce(field, x)
    if field isa CM.QQField

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
    U1_hp = PLP.make_hpoly(K[c(-1) c(0); c(0) c(-1)], K[c(0), c(0)])    # x >= 0, y >= 0
    D1_hp = PLP.make_hpoly(K[c(1) c(0); c(0) c(1)], K[c(1), c(1)])      # x <= 1, y <= 1
    U1 = PLP.PLUpset(PLP.PolyUnion(2, [U1_hp]))
    D1 = PLP.PLDownset(PLP.PolyUnion(2, [D1_hp]))
    F1 = PLP.PLFringe([U1], [D1], reshape(K[c(1)], 1, 1))

    # F2: support approximately [1,2]^2
    U2_hp = PLP.make_hpoly(K[c(-1) c(0); c(0) c(-1)], K[c(-1), c(-1)])  # x >= 1, y >= 1
    D2_hp = PLP.make_hpoly(K[c(1) c(0); c(0) c(1)], K[c(2), c(2)])      # x <= 2, y <= 2
    U2 = PLP.PLUpset(PLP.PolyUnion(2, [U2_hp]))
    D2 = PLP.PLDownset(PLP.PolyUnion(2, [D2_hp]))
    F2 = PLP.PLFringe([U2], [D2], reshape(K[c(1)], 1, 1))

    enc_pl = TO.EncodingOptions(backend=:pl, max_regions=50_000, strict_eps=PLP.STRICT_EPS_QQ)
    df2 = TO.DerivedFunctorOptions(maxdeg=2)
    res3 = TO.ResolutionOptions(maxlen=3)

    # If Polyhedra is missing, encoding is unavailable by design.
    if !PLP.HAVE_POLY
        @test_throws ErrorException TO.encode((F1, F2); enc=enc_pl)
        return
    end

    # Common-encode both PL presentations to the same finite poset P and modules Ms on P.
    enc = TO.encode((F1, F2); enc=enc_pl)
    P = enc[1].P
    Ms = [enc[1].M, enc[2].M]

    # Workflow-level auto-cache: encode() should attach an EncodingCache even
    # without an explicit SessionCache, and geometry calls should reuse it.
    enc_one = TO.encode(F1, TO.EncodingOptions(backend=:pl))
    @test enc_one.pi isa EC.CompiledEncoding
    @test enc_one.pi.meta isa CM.EncodingCache
    unit_box = ([0.0, 0.0], [1.0, 1.0])
    adj_1 = TO.RegionGeometry.region_adjacency(enc_one.pi; box=unit_box, strict=true, mode=:fast)
    adj_2 = TO.RegionGeometry.region_adjacency(enc_one.pi; box=unit_box, strict=true, mode=:fast)
    @test adj_2 == adj_1
    r_unit = EC.locate(enc_one.pi, [0.5, 0.5]; mode=:verified)
    @test r_unit != 0
    @test TO.RegionGeometry.region_bbox(enc_one.pi, r_unit; box=unit_box) == (Float64[0.0, 0.0], Float64[1.0, 1.0])
    @test isapprox(TO.RegionGeometry.region_diameter(enc_one.pi, r_unit; box=unit_box, method=:bbox, metric=:L2), sqrt(2.0); atol=1e-10)

    # Ext computed on P should match ExtRn wrapper (which internally does the same steps).
    E_explicit = DF.Ext(Ms[1], Ms[2], df2)
    E_wrap = DF.ExtRn(F1, F2, enc_pl, df2)
    @test [TO.dim(E_explicit, t) for t in 0:2] == [TO.dim(E_wrap, t) for t in 0:2]

    # Resolution wrapper should use the same encoded poset as explicit encoding.
    enc1 = TO.encode(F1, enc_pl)
    res_wrap = DF.projective_resolution_Rn(F1, enc_pl, res3; return_encoding=true)
    @test FF.poset_equal(res_wrap.P, enc1.P)
    @test DF.betti_table(res_wrap.res) == DF.betti_table(DF.projective_resolution(enc1.M, res3))

    # Minimal Betti data: obtain it by requesting a checked-minimal resolution.
    res_min = TO.ResolutionOptions(maxlen=3, minimal=true, check=true)

    bt_wrap = DF.betti(DF.projective_resolution_Rn(F1, enc_pl, res_min))
    bt_explicit = DF.betti(DF.projective_resolution(enc1.M, res_min))
    @test bt_wrap == bt_explicit
end




@testset "PLPolyhedra UX surface" begin
    hp_birth = PLP.make_hpoly(K[c(-1) c(0); c(0) c(-1)], K[c(0), c(0)])
    hp_death = PLP.make_hpoly(K[c(1) c(0); c(0) c(1)], K[c(1), c(1)])
    U = PLP.PLUpset(PLP.PolyUnion(2, [hp_birth]))
    D = PLP.PLDownset(PLP.PolyUnion(2, [hp_death]))
    F = PLP.PLFringe([U], [D], reshape(K[c(1)], 1, 1))

    @test TO.describe(hp_birth).kind == :hpoly
    @test TO.describe(U.U).kind == :poly_union
    @test TO.describe(U).kind == :pl_upset
    @test TO.describe(D).kind == :pl_downset
    @test TO.describe(F).kind == :pl_fringe

    @test TO.ambient_dim(hp_birth) == 2
    @test TO.ambient_dim(U.U) == 2
    @test TO.ambient_dim(U) == 2
    @test TO.ambient_dim(D) == 2
    @test TO.ambient_dim(F) == 2
    @test PLP.polyhedra(hp_birth)[1] === hp_birth
    @test PLP.polyhedra(U.U)[1] === hp_birth
    @test PLP.polyhedra(U)[1] === hp_birth
    @test PLP.polyhedra(D)[1] === hp_death
    @test PLP.npolyhedra(hp_birth) == 1
    @test PLP.npolyhedra(U.U) == 1
    @test PLP.npolyhedra(U) == 1
    @test PLP.npolyhedra(D) == 1
    @test FF.birth_upsets(F)[1] === U
    @test FF.death_downsets(F)[1] === D
    @test FZ.coefficient_matrix(F) == reshape(QQ[c(1)], 1, 1)
    @test PLP.nupsets(F) == 1
    @test PLP.ndownsets(F) == 1
    @test TO.field(F) isa CM.QQField
    @test PLP.pl_fringe_summary(F).matrix_size == (1, 1)
    @test PLP.pl_fringe_summary(F).field isa CM.QQField

    @test occursin("HPoly", sprint(show, hp_birth))
    @test occursin("PolyUnion", sprint(show, U.U))
    @test occursin("PLUpset", sprint(show, U))
    @test occursin("PLDownset", sprint(show, D))
    @test occursin("PLFringe", sprint(show, MIME("text/plain"), F))

    rep_h = PLP.check_hpoly(hp_birth)
    rep_union = PLP.check_poly_union(U.U)
    rep_up = PLP.check_pl_upset(U)
    rep_down = PLP.check_pl_downset(D)
    rep_fringe = PLP.check_pl_fringe(F)
    @test rep_h.valid
    @test rep_union.valid
    @test rep_up.valid
    @test rep_down.valid
    @test rep_fringe.valid
    @test PLP.plpolyhedra_validation_summary(rep_fringe).report.kind == :pl_fringe
    @test occursin("PLPolyhedraValidationSummary",
                   sprint(show, MIME("text/plain"), PLP.plpolyhedra_validation_summary(rep_fringe)))

    bad_hp = PLP.HPoly(2, reshape(QQ[c(1)], 1, 1), QQ[c(1)], nothing, falses(1), PLP.STRICT_EPS_QQ)
    bad_union = PLP.PolyUnion(2, [bad_hp])
    bad_fringe = PLP.PLFringe(2, [U], [D], zeros(QQ, 2, 2))
    @test !PLP.check_hpoly(bad_hp).valid
    @test !PLP.check_poly_union(bad_union).valid
    @test !PLP.check_pl_fringe(bad_fringe).valid
    @test_throws ArgumentError PLP.check_hpoly(bad_hp; throw=true)
    @test_throws ArgumentError PLP.check_poly_union(bad_union; throw=true)
    @test_throws ArgumentError PLP.check_pl_fringe(bad_fringe; throw=true)

    @test TOA.HPoly === PLP.HPoly
    @test TOA.PolyUnion === PLP.PolyUnion
    @test TOA.PLUpset === PLP.PLUpset
    @test TOA.PLDownset === PLP.PLDownset
    @test TOA.PLEncodingMap === PLP.PLEncodingMap
    @test TOA.PolyInBoxCache === PLP.PolyInBoxCache
    @test TOA.check_hpoly(hp_birth).valid
    @test TOA.pl_fringe_summary(F).matrix_size == (1, 1)
    @test TOA.nupsets(F) == 1
    @test TOA.ndownsets(F) == 1

    if !PLP.HAVE_POLY
        @test true
    else
        opts = TO.EncodingOptions(backend=:pl, max_regions=128, strict_eps=PLP.STRICT_EPS_QQ)
        P, H, pi = PLP.encode_from_PL_fringe(F, opts)
        enc = EC.compile_encoding(P, pi)

        @test TO.describe(pi).kind == :pl_encoding_map
        @test PLP.nregions(pi) == length(pi.regions)
        @test length(PLP.region_witnesses(pi)) == PLP.nregions(pi)
        @test PLP.region_witness(pi, 1) == PLP.region_witnesses(pi)[1]
        sig = PLP.region_signature(pi, 1)
        @test sig isa NamedTuple
        @test Set(keys(sig)) == Set((:y, :z))
        @test PLP.has_spatial_index(pi) isa Bool

        sum_pi = PLP.pl_encoding_summary(pi)
        sum_enc = PLP.pl_encoding_summary(enc)
        @test sum_pi.kind == :pl_encoding_map
        @test sum_pi.nregions == PLP.nregions(pi)
        @test sum_enc.compiled
        @test sum_enc.poset_size == FF.nvertices(P)

        inside = [0.5, 0.5]
        outside = [2.0, 2.0]
        qrep = PLP.check_pl_point(pi, inside)
        @test qrep.valid
        @test !PLP.check_pl_point(pi, [0.5]).valid
        @test_throws ArgumentError PLP.check_pl_point(pi, [0.5]; throw=true)

        X = Float64[0.25 0.75; 0.25 0.75]
        @test PLP.check_pl_points(pi, X).valid
        @test !PLP.check_pl_points(pi, reshape([0.25, 0.75], 1, 2)).valid
        @test_throws ArgumentError PLP.check_pl_points(pi, reshape([0.25, 0.75], 1, 2); throw=true)

        box = (Float64[0.0, 0.0], Float64[1.0, 1.0])
        @test PLP.check_pl_box(pi, box).valid
        @test !PLP.check_pl_box(pi, (Float64[1.0, 0.0], Float64[0.0, 1.0])).valid
        @test_throws ArgumentError PLP.check_pl_box(pi, (Float64[1.0, 0.0], Float64[0.0, 1.0]); throw=true)

        qsum_in = PLP.pl_query_summary(pi, inside)
        qsum_out = PLP.pl_query_summary(enc, outside)
        rid_out = PLP.locate(enc, outside)
        @test qsum_in.region != 0
        @test qsum_in.outside == false
        @test qsum_out.region == rid_out
        @test qsum_out.outside == (rid_out == 0)

        @test PLP.check_pl_encoding_map(pi).valid
        @test PLP.check_pl_encoding_map(enc).valid
        @test !PLP.check_pl_encoding_map(:not_a_pl_encoding).valid
        @test_throws ArgumentError PLP.check_pl_encoding_map(:not_a_pl_encoding; throw=true)
        @test !PLP.check_pl_region(pi, 0).valid
        @test_throws ArgumentError PLP.check_pl_region(pi, 0; throw=true)

        cache = PLP.compile_geometry_cache(pi; box=box, closure=true, level=:light,
                                           precompute_exact=false, precompute_facets=false,
                                           precompute_centroids=false)
        @test TO.describe(cache).kind == :poly_in_box_cache
        @test TO.ambient_dim(cache) == 2
        @test PLP.cache_box(cache)[1] == QQ[c(0), c(0)]
        @test PLP.cache_box(cache)[2] == QQ[c(1), c(1)]
        @test PLP.cache_level(cache) == :light
        @test PLP.cached_region_count(cache) >= 0
        @test PLP.check_poly_in_box_cache(cache).valid
        @test PLP.poly_cache_summary(cache).cache_level == :light
        @test occursin("PolyInBoxCache", sprint(show, MIME("text/plain"), cache))
        @test TOA.check_pl_box(cache, box).valid
        @test TOA.pl_encoding_summary(pi).nregions == PLP.nregions(pi)
        @test TOA.poly_cache_summary(cache).cache_level == :light
        rid = PLP.locate(pi, inside; mode=:verified)
        @test rid != 0
        reg_report = PLP.check_pl_region(pi, rid; box=box)
        reg_report_cache = PLP.check_pl_region(cache, rid)
        reg_summary = PLP.pl_region_summary(pi, rid; box=box, cache=cache)
        @test reg_report.valid
        @test reg_report.region_in_range
        @test reg_report.finite_box_required
        @test reg_report.signature_support_sizes !== nothing
        @test reg_report.bbox isa Tuple
        @test reg_report_cache.valid
        @test reg_report_cache.finite_box_required == false
        @test reg_summary.region == rid
        @test reg_summary.signature_support_sizes == reg_report.signature_support_sizes
        @test reg_summary.bbox isa Tuple
        @test TOA.check_pl_region(cache, rid).valid
        @test TOA.pl_region_summary(pi, rid; box=box, cache=cache).region == rid

        @test TO.RegionGeometry.region_bbox(pi, rid; box=box, cache=cache) isa Tuple
    end
end


@testset "PL common encoding for multiple fringes" begin

    enc_pl_10k = TO.EncodingOptions(backend=:pl, max_regions=10_000)

    # We can build HPolys even without Polyhedra, but we can only encode if Polyhedra is available.
    if !PLP.HAVE_POLY
        U1 = PLP.PLUpset(PLP.PolyUnion(1, [PLP.make_hpoly([-1.0], [0.0])]))   # x >= 0
        D1 = PLP.PLDownset(PLP.PolyUnion(1, [PLP.make_hpoly([ 1.0], [2.0])])) # x <= 2
        F1 = PLP.PLFringe([U1], [D1], reshape(K[c(1)], 1, 1))

        @test_throws ErrorException PLP.encode_from_PL_fringes(F1, F1, enc_pl_10k)
    else
        # Two 1D modules: support [0,2] and support [1,3].
        U1 = PLP.PLUpset(PLP.PolyUnion(1, [PLP.make_hpoly([-1.0], [ 0.0])]))  # x >= 0
        D1 = PLP.PLDownset(PLP.PolyUnion(1, [PLP.make_hpoly([ 1.0], [ 2.0])]))# x <= 2

        U2 = PLP.PLUpset(PLP.PolyUnion(1, [PLP.make_hpoly([-1.0], [-1.0])]))  # x >= 1
        D2 = PLP.PLDownset(PLP.PolyUnion(1, [PLP.make_hpoly([ 1.0], [ 3.0])]))# x <= 3

        F1 = PLP.PLFringe([U1], [D1], reshape(K[c(1)], 1, 1))
        F2 = PLP.PLFringe([U2], [D2], reshape(K[c(1)], 1, 1))

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
        A = K[c(1)  c(0);
              c(0)  c(1);
              c(-1) c(0);
              c(0)  c(-1)]
        b = K[c(1), c(1), c(0), c(0)]
        h = PLP.make_hpoly(A, b)

        @test PLP._in_hpoly(h, [0, 0]) == true
        @test PLP._in_hpoly(h, [1, 1]) == true
        @test PLP._in_hpoly(h, [2, 0]) == false
        @test PLP._in_hpoly(h, [-1, 0]) == false
    else
        # Even without Polyhedra/CDDLib, make_hpoly should still build an HPoly
        # that supports exact membership tests via its stored A*x <= b data.
        A = reshape(K[c(1)], 1, 1)
        b = K[c(1)]
        h = PLP.make_hpoly(A, b)
        @test h.poly === nothing
        @test PLP._in_hpoly(h, [0]) == true
        @test PLP._in_hpoly(h, [1]) == true
        @test PLP._in_hpoly(h, [2]) == false
    end
end

@testset "PLPolyhedra float kernels and relaxed membership cache plumbing" begin
    # Two adjacent unit squares [0,1]x[0,1] and [1,2]x[0,1].
    A = QQ[1 0; 0 1; -1 0; 0 -1]
    hp1 = PLP.make_hpoly(A, QQ[1, 1, 0, 0])
    hp2 = PLP.make_hpoly(A, QQ[2, 1, -1, 0])
    pi = PLP.PLEncodingMap(
        2,
        [BitVector([true]), BitVector([false])],
        [BitVector([false]), BitVector([false])],
        [hp1, hp2],
        [(0.5, 0.5), (1.5, 0.5)],
    )

    Af_strict, bf_strict = PLP._membership_mats(pi, nothing, false)
    Af_relaxed, bf_relaxed = PLP._membership_mats(pi, nothing, true)
    @test Af_strict === pi.Af
    @test bf_strict === pi.bf_strict
    @test Af_relaxed === pi.Af
    @test bf_relaxed === pi.bf_relaxed
    @test length(bf_relaxed) == 2
    @test bf_relaxed[1] == Float64.(PLP._relaxed_b(pi.regions[1]))
    @test bf_relaxed[2] == Float64.(PLP._relaxed_b(pi.regions[2]))

    Af = Float64[1 0; 0 1; -1 0; 0 -1]
    bf = Float64[1.0, 1.0, 0.0, 0.0]
    xin = [0.25, 0.75]
    xout = [1.25, 0.75]
    Xin = reshape(copy(xin), 2, 1)
    Xout = reshape(copy(xout), 2, 1)

    st_in_dense = PLP._hpoly_float_state(Af, bf, xin; tol=1e-12, boundary_tol=1e-12)
    st_in_generic = PLP._hpoly_float_state(@view(Af[:, :]), @view(bf[:]), @view(xin[:]); tol=1e-12, boundary_tol=1e-12)
    st_out_dense = PLP._hpoly_float_state(Af, bf, xout; tol=1e-12, boundary_tol=1e-12)
    st_out_generic = PLP._hpoly_float_state(@view(Af[:, :]), @view(bf[:]), @view(xout[:]); tol=1e-12, boundary_tol=1e-12)
    @test st_in_dense == st_in_generic
    @test st_out_dense == st_out_generic
    @test st_in_dense == Int8(0) || st_in_dense == Int8(1)
    @test st_out_dense == Int8(-1) || st_out_dense == Int8(0)

    st_col_in_dense = PLP._hpoly_float_state_col(Af, bf, Xin, 1; tol=1e-12, boundary_tol=1e-12)
    st_col_in_generic = PLP._hpoly_float_state_col(@view(Af[:, :]), @view(bf[:]), @view(Xin[:, :]), 1; tol=1e-12, boundary_tol=1e-12)
    st_col_out_dense = PLP._hpoly_float_state_col(Af, bf, Xout, 1; tol=1e-12, boundary_tol=1e-12)
    st_col_out_generic = PLP._hpoly_float_state_col(@view(Af[:, :]), @view(bf[:]), @view(Xout[:, :]), 1; tol=1e-12, boundary_tol=1e-12)
    @test st_col_in_dense == st_col_in_generic
    @test st_col_out_dense == st_col_out_generic

    @test PLP._in_hpoly_float(Af, bf, xin; tol=1e-12) == true
    @test PLP._in_hpoly_float(@view(Af[:, :]), @view(bf[:]), @view(xin[:]); tol=1e-12) == true
    @test PLP._in_hpoly_float(Af, bf, xout; tol=1e-12) == false
    @test PLP._in_hpoly_float(@view(Af[:, :]), @view(bf[:]), @view(xout[:]); tol=1e-12) == false

    seg = [Float64[0.0, 0.0], Float64[2.0, 0.0]]
    @test isapprox(PLP._facet_measure(seg, Float64[0.0, 1.0]), 2.0; atol=1e-12)

    square3 = [Float64[0.0, 0.0, 0.0],
               Float64[1.0, 0.0, 0.0],
               Float64[1.0, 1.0, 0.0],
               Float64[0.0, 1.0, 0.0]]
    @test isapprox(PLP._facet_measure(square3, Float64[0.0, 0.0, 1.0]), 1.0; atol=1e-12)

    if PLP.HAVE_POLY
        box = ([-0.25, -0.25], [2.25, 1.25])
        cache = PLP.compile_geometry_cache(pi; box=box, closure=true)
        rid = EC.locate(pi, [0.5, 0.5]; mode=:verified)
        facets = PLP._region_facets(cache, rid; tol=1e-10)
        @test !isempty(facets)
        @test length(facets) == 4
        scratch1 = PLP._facet_classify_scratch!(cache, 2 * length(facets))
        xid = objectid(scratch1.X)
        inbox_id = objectid(scratch1.in_box)
        loc_id = objectid(scratch1.loc)
        kinds, neigh = PLP._classify_cached_facets(
            pi, cache, facets, rid, cache.box_f[1], cache.box_f[2], 1e-8, :fast;
            strict=false, tol=1e-10,
        )
        @test length(kinds) == length(facets)
        @test length(neigh) == length(facets)
        @test any(k -> k == UInt8(1) || k == UInt8(2), kinds)
        scratch2 = PLP._facet_classify_scratch!(cache, 2 * length(facets))
        @test objectid(scratch2.X) == xid
        @test objectid(scratch2.in_box) == inbox_id
        @test objectid(scratch2.loc) == loc_id

        Xq = [0.25 0.75 1.25 1.75;
              0.50 0.50 0.50 0.50]
        loc_full = fill(0, size(Xq, 2))
        loc_pref = fill(-1, size(Xq, 2))
        PLP.locate_many!(loc_full, cache, Xq; threaded=false, mode=:fast)
        PLP._locate_many_prefix!(loc_pref, cache, Xq, size(Xq, 2); threaded=false, mode=:fast)
        @test loc_pref == loc_full

        adj_cache = TO.RegionGeometry.region_adjacency(pi; cache=cache, strict=false, mode=:fast)
        adj_box = TO.RegionGeometry.region_adjacency(pi; box=box, strict=false, mode=:fast)
        @test adj_cache == adj_box
        @test length(adj_cache) == 1
        @test haskey(adj_cache, (1, 2))
        @test isapprox(adj_cache[(1, 2)], 1.0; atol=1e-8)

        hf1 = PLP._hrep_float_in_box(cache, rid)
        hf2 = PLP._hrep_float_in_box(cache, rid)
        @test hf1 === hf2
        @test size(hf1.A, 2) == pi.n
        @test length(hf1.b) == size(hf1.A, 1)

        vre = PLP._vrep_in_box(cache, rid)
        hre = PLP._hrep_in_box(cache, rid)
        pts = collect(PLP.Polyhedra.points(vre))
        hs = collect(PLP.Polyhedra.halfspaces(hre))
        ptsf = PLP._points_float_in_box(cache, rid)
        @test !isempty(hs)
        i = 1
        idx_plain = PLP._incident_vertex_indices(vre, pts, hs[i]; tol=1e-12)
        idx_cached = PLP._incident_vertex_indices(
            vre, pts, hs[i];
            tol=1e-12,
            a_float=@view(hf1.A[i, :]),
            b_float=hf1.b[i],
            pts_float=ptsf,
        )
        @test idx_plain == idx_cached

        # Spatial prefilter (bbox-grid) should be enabled on larger 2D maps and
        # preserve locate_many correctness against exact membership scans.
        nxg, nyg = 16, 16
        ngrid = nxg * nyg
        regs_g = Vector{PLP.HPoly}(undef, ngrid)
        reps_g = Vector{Tuple{Float64,Float64}}(undef, ngrid)
        sigy_g = [BitVector() for _ in 1:ngrid]
        sigz_g = [BitVector() for _ in 1:ngrid]
        k = 1
        for gy in 0:nyg-1, gx in 0:nxg-1
            xlo = float(gx)
            xhi = float(gx + 1)
            ylo = float(gy)
            yhi = float(gy + 1)
            regs_g[k] = PLP.make_hpoly(A, [xhi, yhi, -xlo, -ylo])
            reps_g[k] = ((xlo + xhi) / 2.0, (ylo + yhi) / 2.0)
            k += 1
        end
        pi_g = PLP.PLEncodingMap(2, sigy_g, sigz_g, regs_g, reps_g)
        @test pi_g.prefilter.spatial.enabled
        cands_mid = PLP._spatial_prefilter_candidates(pi_g.prefilter, [8.25, 8.75])
        @test cands_mid !== nothing
        @test !isempty(cands_mid)
        @test length(cands_mid) < ngrid
        @test PLP._should_use_grouped_locate(pi_g, nothing, 5_000)
        if Threads.nthreads() >= 10
            @test !PLP._should_use_grouped_locate(pi_g, nothing, 50_000)
        else
            @test PLP._should_use_grouped_locate(pi_g, nothing, 50_000)
        end
        @test !PLP._should_use_grouped_locate(pi_g, nothing, 120_000)
        counts_bal = [0; fill(8, 32)]
        counts_skew = [0; vcat([256], fill(0, 31))]
        @test PLP._grouped_live_bucket_ok(counts_bal, 32, sum(counts_bal))
        @test !PLP._grouped_live_bucket_ok(counts_skew, 32, sum(counts_skew))

        Xg = Matrix{Float64}(undef, 2, 256)
        rng = Random.MersenneTwister(0x91)
        @inbounds for j in 1:size(Xg, 2)
            # Stay strictly inside cells to avoid boundary ambiguity.
            Xg[1, j] = rand(rng) * nxg - 1e-3
            Xg[2, j] = rand(rng) * nyg - 1e-3
        end
        loc_fast = fill(0, size(Xg, 2))
        PLP.locate_many!(loc_fast, pi_g, Xg; threaded=false, mode=:fast)

        loc_exact = fill(0, size(Xg, 2))
        @inbounds for j in 1:size(Xg, 2)
            xq = @view Xg[:, j]
            for t in 1:length(pi_g.regions)
                if PLP._in_hpoly(pi_g.regions[t], xq)
                    loc_exact[j] = t
                    break
                end
            end
        end
        @test loc_fast == loc_exact

        # Bucket-grouped locate_many path: parity with grouped toggle off/on.
        Xg_big = Matrix{Float64}(undef, 2, 5_000)
        rng_big = Random.MersenneTwister(0x9B1)
        @inbounds for j in 1:size(Xg_big, 2)
            Xg_big[1, j] = rand(rng_big) * nxg - 1e-3
            Xg_big[2, j] = rand(rng_big) * nyg - 1e-3
        end
        loc_group_off = fill(0, size(Xg_big, 2))
        loc_group_on = fill(0, size(Xg_big, 2))
        old_group_flag = PLP._LOCATE_BUCKET_GROUPING[]
        try
            PLP._LOCATE_BUCKET_GROUPING[] = false
            PLP.locate_many!(loc_group_off, pi_g, Xg_big; threaded=true, mode=:fast)
            PLP._LOCATE_BUCKET_GROUPING[] = true
            PLP.locate_many!(loc_group_on, pi_g, Xg_big; threaded=true, mode=:fast)
        finally
            PLP._LOCATE_BUCKET_GROUPING[] = old_group_flag
        end
        @test loc_group_on == loc_group_off

        cache_g = PLP.compile_geometry_cache(pi_g; box=(Float64[0.0, 0.0], Float64[float(nxg), float(nyg)]), closure=true)
        loc_cache_group_off = fill(0, size(Xg_big, 2))
        loc_cache_group_on = fill(0, size(Xg_big, 2))
        old_group_flag = PLP._LOCATE_BUCKET_GROUPING[]
        try
            PLP._LOCATE_BUCKET_GROUPING[] = false
            PLP.locate_many!(loc_cache_group_off, cache_g, Xg_big; threaded=true, mode=:fast)
            PLP._LOCATE_BUCKET_GROUPING[] = true
            PLP.locate_many!(loc_cache_group_on, cache_g, Xg_big; threaded=true, mode=:fast)
        finally
            PLP._LOCATE_BUCKET_GROUPING[] = old_group_flag
        end
        @test loc_cache_group_on == loc_cache_group_off
        @test pi_g.prefilter.spatial.buckets isa PLP._PackedBuckets
        @test cache_g.bucket_regions isa PLP._PackedBuckets

        old_exact_cache = PLP._LOCATE_COL_QQ_CACHE[]
        loc_exact_cache_off = fill(0, size(Xg_big, 2))
        loc_exact_cache_on = fill(0, size(Xg_big, 2))
        try
            PLP._LOCATE_COL_QQ_CACHE[] = false
            PLP.locate_many!(loc_exact_cache_off, pi_g, Xg_big; threaded=true, mode=:fast)
            PLP._LOCATE_COL_QQ_CACHE[] = true
            PLP.locate_many!(loc_exact_cache_on, pi_g, Xg_big; threaded=true, mode=:fast)
        finally
            PLP._LOCATE_COL_QQ_CACHE[] = old_exact_cache
        end
        @test loc_exact_cache_on == loc_exact_cache_off

        old_row_dot_cache = PLP._LOCATE_ROW_DOT_CACHE[]
        loc_row_dot_off = fill(0, size(Xg_big, 2))
        loc_row_dot_on = fill(0, size(Xg_big, 2))
        try
            PLP._LOCATE_COL_QQ_CACHE[] = true
            PLP._LOCATE_ROW_DOT_CACHE[] = false
            PLP.locate_many!(loc_row_dot_off, pi_g, Xg_big; threaded=true, mode=:fast)
            PLP._LOCATE_ROW_DOT_CACHE[] = true
            PLP.locate_many!(loc_row_dot_on, pi_g, Xg_big; threaded=true, mode=:fast)
        finally
            PLP._LOCATE_COL_QQ_CACHE[] = old_exact_cache
            PLP._LOCATE_ROW_DOT_CACHE[] = old_row_dot_cache
        end
        @test loc_row_dot_on == loc_row_dot_off

        # Direct Float64 cache dispatch should stay on the PLPolyhedra path.
        loc_cache_float = fill(0, size(Xg_big, 2))
        PLP.locate_many!(loc_cache_float, cache_g, Xg_big; threaded=false, mode=:fast)
        @test loc_cache_float == loc_cache_group_on

        # High-dimensional multiprojection prefilter: ensure enabled and parity
        # with exact membership scans for 3D query batches.
        nx3, ny3, nz3 = 5, 5, 3
        nreg3 = nx3 * ny3 * nz3
        A3 = QQ[1 0 0; 0 1 0; 0 0 1; -1 0 0; 0 -1 0; 0 0 -1]
        regs3 = Vector{PLP.HPoly}(undef, nreg3)
        reps3 = Vector{NTuple{3,Float64}}(undef, nreg3)
        sigy3 = [BitVector() for _ in 1:nreg3]
        sigz3 = [BitVector() for _ in 1:nreg3]
        k3 = 1
        for gz in 0:nz3-1, gy in 0:ny3-1, gx in 0:nx3-1
            xlo = float(gx)
            xhi = float(gx + 1)
            ylo = float(gy)
            yhi = float(gy + 1)
            zlo = float(gz)
            zhi = float(gz + 1)
            regs3[k3] = PLP.make_hpoly(A3, QQ[xhi, yhi, zhi, -xlo, -ylo, -zlo])
            reps3[k3] = ((xlo + xhi) / 2.0, (ylo + yhi) / 2.0, (zlo + zhi) / 2.0)
            k3 += 1
        end
        pi3 = PLP.PLEncodingMap(3, sigy3, sigz3, regs3, reps3)
        mp3 = pi3.prefilter.multiproj
        @test mp3.enabled
        @test mp3.ndims >= 2
        @test all(d -> d >= 1 && d <= 3, mp3.dims[1:mp3.ndims])
        rngs3 = PLP._multiproj_ranges(mp3, [2.5, 2.5, 1.5])
        @test rngs3 !== nothing
        @test !PLP._should_use_multiproj_prefilter(pi3.prefilter, length(pi3.regions), 240, true)
        @test !PLP._should_use_grouped_locate(pi3, nothing, 10_000)
        if Threads.nthreads() > 1
            @test !PLP._should_thread_locate_many(pi3, nothing, 12_000; grouped=false)
            @test !PLP._should_thread_locate_many(pi3, nothing, 25_000; grouped=false)
            @test !PLP._should_thread_locate_many(pi3, nothing, 150_000; grouped=false)
        end

        X3 = Matrix{Float64}(undef, 3, 240)
        rng3 = Random.MersenneTwister(0xBEE3)
        @inbounds for j in 1:size(X3, 2)
            rid3 = rand(rng3, 1:nreg3)
            ctr = reps3[rid3]
            # Stay strictly interior to avoid boundary ambiguity.
            X3[1, j] = ctr[1] + (rand(rng3) - 0.5) * 0.6
            X3[2, j] = ctr[2] + (rand(rng3) - 0.5) * 0.6
            X3[3, j] = ctr[3] + (rand(rng3) - 0.5) * 0.6
        end
        loc3_fast = fill(0, size(X3, 2))
        PLP.locate_many!(loc3_fast, pi3, X3; threaded=false, mode=:fast)
        loc3_exact = fill(0, size(X3, 2))
        @inbounds for j in 1:size(X3, 2)
            xq = @view X3[:, j]
            for t in 1:length(pi3.regions)
                if PLP._in_hpoly(pi3.regions[t], xq)
                    loc3_exact[j] = t
                    break
                end
            end
        end
        @test loc3_fast == loc3_exact

        # Heuristic contract: in a 3D family where x1 is effectively collapsed,
        # multi-projection prefilter should activate for large threaded batches.
        nx3b, ny3b = 18, 18
        nreg3b = nx3b * ny3b
        regs3b = Vector{PLP.HPoly}(undef, nreg3b)
        reps3b = Vector{NTuple{3,Float64}}(undef, nreg3b)
        sigy3b = [BitVector() for _ in 1:nreg3b]
        sigz3b = [BitVector() for _ in 1:nreg3b]
        kb = 1
        for gz in 0:ny3b-1, gy in 0:nx3b-1
            xlo = 0.0
            xhi = 1.0
            ylo = float(gy)
            yhi = float(gy + 1)
            zlo = float(gz)
            zhi = float(gz + 1)
            regs3b[kb] = PLP.make_hpoly(A3, QQ[xhi, yhi, zhi, -xlo, -ylo, -zlo])
            reps3b[kb] = ((xlo + xhi) / 2.0, (ylo + yhi) / 2.0, (zlo + zhi) / 2.0)
            kb += 1
        end
        pi3b = PLP.PLEncodingMap(3, sigy3b, sigz3b, regs3b, reps3b)
        @test pi3b.prefilter.multiproj.enabled
        @test PLP._should_use_multiproj_prefilter(pi3b.prefilter, length(pi3b.regions), 20_000, true)
        @test !PLP._should_use_multiproj_prefilter(pi3b.prefilter, length(pi3b.regions), 20_000, false)

        # Facet probe batching: parity with batch toggle off/on (cache and non-cache).
        function _canonical_bd(v)
            canon_entry(e) = (
                kind = e.kind,
                neighbor = e.neighbor === nothing ? 0 : e.neighbor,
                measure = round(e.measure; digits=10),
                point = Tuple(round.(e.point; digits=10)),
                normal = Tuple(round.(e.normal; digits=10)),
            )
            return sort!(map(canon_entry, v); by = x -> (x.kind, x.neighbor, x.point, x.normal, x.measure))
        end

        @test !PLP._should_batch_facet_probes(:cached, 4, 2)
        @test PLP._should_batch_facet_probes(:boundary, 4, 2)
        @test !PLP._should_batch_facet_probes(:adjacency, 4, 2)

        old_batch_flag = PLP._FACET_PROBE_BATCH[]
        try
            PLP._FACET_PROBE_BATCH[] = false
            bd_cache_off = TO.RegionGeometry.region_boundary_measure_breakdown(pi, rid; cache=cache, strict=false, mode=:fast)
            adj_cache_off = TO.RegionGeometry.region_adjacency(pi; cache=cache, strict=false, mode=:fast)
            bd_box_off = TO.RegionGeometry.region_boundary_measure_breakdown(pi, rid; box=box, strict=false, mode=:fast)
            adj_box_off = TO.RegionGeometry.region_adjacency(pi; box=box, strict=false, mode=:fast)

            PLP._FACET_PROBE_BATCH[] = true
            bd_cache_on = TO.RegionGeometry.region_boundary_measure_breakdown(pi, rid; cache=cache, strict=false, mode=:fast)
            adj_cache_on = TO.RegionGeometry.region_adjacency(pi; cache=cache, strict=false, mode=:fast)
            bd_box_on = TO.RegionGeometry.region_boundary_measure_breakdown(pi, rid; box=box, strict=false, mode=:fast)
            adj_box_on = TO.RegionGeometry.region_adjacency(pi; box=box, strict=false, mode=:fast)

            @test _canonical_bd(bd_cache_on) == _canonical_bd(bd_cache_off)
            @test _canonical_bd(bd_box_on) == _canonical_bd(bd_box_off)
            @test adj_cache_on == adj_cache_off
            @test adj_box_on == adj_box_off
        finally
            PLP._FACET_PROBE_BATCH[] = old_batch_flag
        end
    end
end

@testset "PLPolyhedra hand-solvable 2D oracle fixtures" begin
    if !PLP.HAVE_POLY
        @test true
    else
        Arect = QQ[1 0; 0 1; -1 0; 0 -1]
        rect_hpoly(xl, xu, yl, yu) = PLP.make_hpoly(Arect, QQ[xu, yu, -xl, -yl])

        function sigbits(i::Int, k::Int)
            bv = BitVector(undef, k)
            x = i - 1
            @inbounds for b in 1:k
                bv[b] = ((x >>> (b - 1)) & 0x1) == 0x1
            end
            return bv
        end

        function run_fixture(rects, witnesses, box, expected; outside=Tuple{Float64,Float64}[])
            nreg = length(rects)
            @test length(witnesses) == nreg

            k = max(1, ceil(Int, log2(nreg + 1)))
            sigy = [sigbits(i, k) for i in 1:nreg]
            sigz = [falses(k) for _ in 1:nreg]
            hps = [rect_hpoly(r[1], r[2], r[3], r[4]) for r in rects]
            reps = [(w[1], w[2]) for w in witnesses]
            pi = PLP.PLEncodingMap(2, sigy, sigz, hps, reps)

            rid = Vector{Int}(undef, nreg)
            for i in 1:nreg
                rid[i] = EC.locate(pi, [witnesses[i][1], witnesses[i][2]]; mode=:verified)
                @test rid[i] != 0
            end
            @test length(unique(rid)) == nreg

            for p in outside
                @test EC.locate(pi, [p[1], p[2]]; mode=:verified) == 0
            end

            w = TO.RegionGeometry.region_weights(pi; box=box, method=:exact)
            for i in 1:nreg
                @test isapprox(w[rid[i]], expected[:weights][i]; atol=1e-10)
            end
            @test isapprox(sum(w), sum(expected[:weights]); atol=1e-10)

            for i in 1:nreg
                bb = TO.RegionGeometry.region_bbox(pi, rid[i]; box=box)
                @test bb !== nothing
                lo, hi = bb
                elo, ehi = expected[:bbox][i]
                @test all(isapprox.(lo, elo; atol=1e-10))
                @test all(isapprox.(hi, ehi; atol=1e-10))
            end

            for i in 1:nreg
                d = TO.RegionGeometry.region_diameter(pi, rid[i]; box=box, metric=:L2, method=:bbox)
                @test isapprox(d, expected[:diameters][i]; atol=1e-10)
            end

            for i in 1:nreg
                bm = TO.RegionGeometry.region_boundary_measure(pi, rid[i]; box=box, strict=true, mode=:verified)
                @test isapprox(bm, expected[:boundary][i]; atol=1e-8)
            end

            adj = TO.RegionGeometry.region_adjacency(pi; box=box, strict=true, mode=:verified)
            exp_adj = Dict{Tuple{Int,Int},Float64}()
            for ((i, j), m) in expected[:adj]
                ri = rid[i]
                rj = rid[j]
                key = ri < rj ? (ri, rj) : (rj, ri)
                exp_adj[key] = float(m)
            end
            @test Set(keys(adj)) == Set(keys(exp_adj))
            for (kpair, mexp) in exp_adj
                @test isapprox(adj[kpair], mexp; atol=1e-8)
            end
        end

        fixtures = [
            (
                rects=[(0.0, 1.0, 0.0, 1.0)],
                witnesses=[(0.5, 0.5)],
                box=(Float64[0.0, 0.0], Float64[1.0, 1.0]),
                expected=(
                    weights=[1.0],
                    bbox=[(Float64[0.0, 0.0], Float64[1.0, 1.0])],
                    diameters=[sqrt(2.0)],
                    boundary=[4.0],
                    adj=Dict{Tuple{Int,Int},Float64}(),
                ),
                outside=[(-0.1, 0.5), (1.1, 0.5)],
            ),
            (
                rects=[(0.0, 1.0, 0.0, 1.0), (1.0, 2.0, 0.0, 1.0)],
                witnesses=[(0.5, 0.5), (1.5, 0.5)],
                box=(Float64[0.0, 0.0], Float64[2.0, 1.0]),
                expected=(
                    weights=[1.0, 1.0],
                    bbox=[(Float64[0.0, 0.0], Float64[1.0, 1.0]),
                          (Float64[1.0, 0.0], Float64[2.0, 1.0])],
                    diameters=[sqrt(2.0), sqrt(2.0)],
                    boundary=[4.0, 4.0],
                    adj=Dict((1, 2) => 1.0),
                ),
                outside=[(-0.1, 0.5), (2.1, 0.5)],
            ),
            (
                rects=[(-1.0, 0.0, -1.0, 0.0), (0.0, 1.0, -1.0, 0.0), (-1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0)],
                witnesses=[(-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5)],
                box=(Float64[-1.0, -1.0], Float64[1.0, 1.0]),
                expected=(
                    weights=[1.0, 1.0, 1.0, 1.0],
                    bbox=[(Float64[-1.0, -1.0], Float64[0.0, 0.0]),
                          (Float64[0.0, -1.0], Float64[1.0, 0.0]),
                          (Float64[-1.0, 0.0], Float64[0.0, 1.0]),
                          (Float64[0.0, 0.0], Float64[1.0, 1.0])],
                    diameters=[sqrt(2.0), sqrt(2.0), sqrt(2.0), sqrt(2.0)],
                    boundary=[4.0, 4.0, 4.0, 4.0],
                    adj=Dict((1, 2) => 1.0, (1, 3) => 1.0, (2, 4) => 1.0, (3, 4) => 1.0),
                ),
                outside=[(-1.1, 0.0), (1.1, 0.0)],
            ),
            (
                rects=[(0.0, 1.0, 0.0, 2.0), (1.0, 3.0, 0.0, 2.0)],
                witnesses=[(0.5, 1.0), (2.0, 1.0)],
                box=(Float64[0.0, 0.0], Float64[3.0, 2.0]),
                expected=(
                    weights=[2.0, 4.0],
                    bbox=[(Float64[0.0, 0.0], Float64[1.0, 2.0]),
                          (Float64[1.0, 0.0], Float64[3.0, 2.0])],
                    diameters=[sqrt(5.0), sqrt(8.0)],
                    boundary=[6.0, 8.0],
                    adj=Dict((1, 2) => 2.0),
                ),
                outside=[(-0.1, 1.0), (3.1, 1.0)],
            ),
            (
                rects=[(0.0, 2.0, 0.0, 1.0), (0.0, 2.0, 1.0, 2.0), (0.0, 2.0, 2.0, 3.0)],
                witnesses=[(1.0, 0.5), (1.0, 1.5), (1.0, 2.5)],
                box=(Float64[0.0, 0.0], Float64[2.0, 3.0]),
                expected=(
                    weights=[2.0, 2.0, 2.0],
                    bbox=[(Float64[0.0, 0.0], Float64[2.0, 1.0]),
                          (Float64[0.0, 1.0], Float64[2.0, 2.0]),
                          (Float64[0.0, 2.0], Float64[2.0, 3.0])],
                    diameters=[sqrt(5.0), sqrt(5.0), sqrt(5.0)],
                    boundary=[6.0, 6.0, 6.0],
                    adj=Dict((1, 2) => 2.0, (2, 3) => 2.0),
                ),
                outside=[(-0.1, 0.5), (2.1, 0.5)],
            ),
            (
                rects=[(0.0, 1.0, 0.0, 1.0), (1.0, 3.0, 0.0, 1.0), (0.0, 1.0, 1.0, 2.0), (1.0, 3.0, 1.0, 2.0)],
                witnesses=[(0.5, 0.5), (2.0, 0.5), (0.5, 1.5), (2.0, 1.5)],
                box=(Float64[0.0, 0.0], Float64[3.0, 2.0]),
                expected=(
                    weights=[1.0, 2.0, 1.0, 2.0],
                    bbox=[(Float64[0.0, 0.0], Float64[1.0, 1.0]),
                          (Float64[1.0, 0.0], Float64[3.0, 1.0]),
                          (Float64[0.0, 1.0], Float64[1.0, 2.0]),
                          (Float64[1.0, 1.0], Float64[3.0, 2.0])],
                    diameters=[sqrt(2.0), sqrt(5.0), sqrt(2.0), sqrt(5.0)],
                    boundary=[4.0, 6.0, 4.0, 6.0],
                    adj=Dict((1, 2) => 1.0, (1, 3) => 1.0, (2, 4) => 2.0, (3, 4) => 1.0),
                ),
                outside=[(-0.1, 0.5), (3.1, 0.5)],
            ),
        ]

        for fx in fixtures
            run_fixture(fx.rects, fx.witnesses, fx.box, fx.expected; outside=fx.outside)
        end
    end
end

@testset "PLPolyhedra non-convex union oracle fixtures (2D, QQ)" begin
    if !PLP.HAVE_POLY
        @test true
    else
        Arect = QQ[1 0; 0 1; -1 0; 0 -1]
        rect_hpoly(xl, xu, yl, yu) = PLP.make_hpoly(Arect, QQ[xu, yu, -xl, -yl])

        # Non-convex geometry: two disjoint boxes with a gap in between.
        pi = PLP.PLEncodingMap(
            2,
            [BitVector((true, false)), BitVector((false, true))],
            [falses(2), falses(2)],
            [rect_hpoly(0.0, 1.0, 0.0, 1.0), rect_hpoly(2.0, 3.0, 0.0, 1.0)],
            [(0.5, 0.5), (2.5, 0.5)],
        )
        box = (Float64[0.0, 0.0], Float64[3.0, 1.0])

        t_left = EC.locate(pi, [0.5, 0.5]; mode=:verified)
        t_gap = EC.locate(pi, [1.5, 0.5]; mode=:verified)
        t_right = EC.locate(pi, [2.5, 0.5]; mode=:verified)
        @test t_left != 0 && t_right != 0
        @test t_gap == 0
        @test t_left != t_right
        @test EC.locate(pi, [1.5, 1.5]; mode=:verified) == 0

        w = TO.RegionGeometry.region_weights(pi; box=box, method=:exact)
        @test isapprox(w[t_left], 1.0; atol=1e-10)
        @test isapprox(w[t_right], 1.0; atol=1e-10)
        @test isapprox(sum(w), 2.0; atol=1e-10)

        bb_left = TO.RegionGeometry.region_bbox(pi, t_left; box=box)
        bb_right = TO.RegionGeometry.region_bbox(pi, t_right; box=box)
        @test bb_left == (Float64[0.0, 0.0], Float64[1.0, 1.0])
        @test bb_right == (Float64[2.0, 0.0], Float64[3.0, 1.0])

        @test isapprox(TO.RegionGeometry.region_diameter(pi, t_left; box=box, metric=:L2, method=:bbox), sqrt(2.0); atol=1e-10)
        @test isapprox(TO.RegionGeometry.region_diameter(pi, t_right; box=box, metric=:L2, method=:bbox), sqrt(2.0); atol=1e-10)

        @test isapprox(TO.RegionGeometry.region_boundary_measure(pi, t_left; box=box, strict=false, mode=:verified), 4.0; atol=1e-8)
        @test isapprox(TO.RegionGeometry.region_boundary_measure(pi, t_right; box=box, strict=false, mode=:verified), 4.0; atol=1e-8)

        adj = TO.RegionGeometry.region_adjacency(pi; box=box, strict=false, mode=:verified)
        @test isempty(adj)
    end
end

@testset "PLPolyhedra exact 3D oracle fixtures (QQ)" begin
    if !PLP.HAVE_POLY
        @test true
    else
        Acube = QQ[1 0 0; 0 1 0; 0 0 1; -1 0 0; 0 -1 0; 0 0 -1]
        box_hpoly(xl, xu, yl, yu, zl, zu) = PLP.make_hpoly(Acube, QQ[xu, yu, zu, -xl, -yl, -zl])

        function mk_pi3(hps, reps)
            n = length(hps)
            k = max(1, ceil(Int, log2(n + 1)))
            sigy = [BitVector(((i - 1) >>> (b - 1)) & 0x1 == 0x1 for b in 1:k) for i in 1:n]
            sigz = [falses(k) for _ in 1:n]
            return PLP.PLEncodingMap(3, sigy, sigz, hps, reps)
        end

        # Fixture A: single unit cube.
        hp = box_hpoly(0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        pi = mk_pi3([hp], [(0.5, 0.5, 0.5)])
        box = (Float64[0.0, 0.0, 0.0], Float64[1.0, 1.0, 1.0])
        r = EC.locate(pi, [0.5, 0.5, 0.5]; mode=:verified)
        @test r != 0
        @test isapprox(TO.RegionGeometry.region_weights(pi; box=box, method=:exact)[r], 1.0; atol=1e-10)
        @test TO.RegionGeometry.region_bbox(pi, r; box=box) == (Float64[0.0, 0.0, 0.0], Float64[1.0, 1.0, 1.0])
        @test isapprox(TO.RegionGeometry.region_diameter(pi, r; box=box, metric=:L2, method=:bbox), sqrt(3.0); atol=1e-10)
        @test isapprox(TO.RegionGeometry.region_boundary_measure(pi, r; box=box, mode=:verified), 6.0; atol=1e-8)
        @test isempty(TO.RegionGeometry.region_adjacency(pi; box=box, strict=true, mode=:verified))

        # Fixture B: two adjacent unit cubes sharing one face.
        hp1 = box_hpoly(0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        hp2 = box_hpoly(1.0, 2.0, 0.0, 1.0, 0.0, 1.0)
        pi2 = mk_pi3([hp1, hp2], [(0.5, 0.5, 0.5), (1.5, 0.5, 0.5)])
        box2 = (Float64[0.0, 0.0, 0.0], Float64[2.0, 1.0, 1.0])
        r1 = EC.locate(pi2, [0.5, 0.5, 0.5]; mode=:verified)
        r2 = EC.locate(pi2, [1.5, 0.5, 0.5]; mode=:verified)
        @test r1 != 0 && r2 != 0 && r1 != r2
        w2 = TO.RegionGeometry.region_weights(pi2; box=box2, method=:exact)
        @test isapprox(w2[r1], 1.0; atol=1e-10)
        @test isapprox(w2[r2], 1.0; atol=1e-10)
        @test isapprox(TO.RegionGeometry.region_boundary_measure(pi2, r1; box=box2, mode=:verified), 6.0; atol=1e-8)
        @test isapprox(TO.RegionGeometry.region_boundary_measure(pi2, r2; box=box2, mode=:verified), 6.0; atol=1e-8)
        adj2 = TO.RegionGeometry.region_adjacency(pi2; box=box2, strict=true, mode=:verified)
        key = r1 < r2 ? (r1, r2) : (r2, r1)
        @test haskey(adj2, key)
        @test isapprox(adj2[key], 1.0; atol=1e-8)

        # Fixture C: rectangular prism.
        hp3 = box_hpoly(0.0, 2.0, 0.0, 1.0, 0.0, 3.0)
        pi3 = mk_pi3([hp3], [(1.0, 0.5, 1.5)])
        box3 = (Float64[0.0, 0.0, 0.0], Float64[2.0, 1.0, 3.0])
        r3 = EC.locate(pi3, [1.0, 0.5, 1.5]; mode=:verified)
        @test r3 != 0
        @test isapprox(TO.RegionGeometry.region_weights(pi3; box=box3, method=:exact)[r3], 6.0; atol=1e-10)
        @test isapprox(TO.RegionGeometry.region_diameter(pi3, r3; box=box3, metric=:L2, method=:bbox), sqrt(14.0); atol=1e-10)
        @test isapprox(TO.RegionGeometry.region_boundary_measure(pi3, r3; box=box3, mode=:verified), 22.0; atol=1e-8)
    end
end

@testset "PLBackend UX surface" begin
    enc_axis = TO.EncodingOptions(backend=:pl_backend)
    Ups = [PLB.BoxUpset([0.0])]
    Downs = [PLB.BoxDownset([2.0])]
    Phi = reshape(K[c(1)], 1, 1)

    @test TO.describe(Ups[1]).kind == :box_upset
    @test TO.describe(Downs[1]).kind == :box_downset
    @test TO.ambient_dim(Ups[1]) == 1
    @test TO.ambient_dim(Downs[1]) == 1
    @test PLB.lower_bounds(Ups[1]) == [0.0]
    @test PLB.upper_bounds(Downs[1]) == [2.0]
    @test occursin("BoxUpset", sprint(show, MIME("text/plain"), Ups[1]))
    @test occursin("BoxDownset", sprint(show, MIME("text/plain"), Downs[1]))

    good_up = PLB.check_box_upset(Ups[1])
    good_down = PLB.check_box_downset(Downs[1])
    bad_up = PLB.check_box_upset(PLB.BoxUpset([0.0, NaN]))
    bad_down = PLB.check_box_downset(PLB.BoxDownset([2.0, NaN]))
    @test good_up.valid
    @test good_down.valid
    @test !bad_up.valid
    @test !bad_down.valid
    @test_throws ArgumentError PLB.check_box_upset(PLB.BoxUpset([0.0, NaN]); throw=true)
    @test_throws ArgumentError PLB.check_box_downset(PLB.BoxDownset([2.0, NaN]); throw=true)

    P, H, pi = PLB.encode_fringe_boxes(Ups, Downs, Phi, enc_axis)
    enc = EC.compile_encoding(P, pi)
    rid = PLB.locate(pi, [1.0])
    box = (Float64[0.0], Float64[2.0])

    @test TO.describe(pi).kind == :pl_backend_encoding_map
    @test TO.ambient_dim(pi) == 1
    @test FF.birth_upsets(pi)[1] === Ups[1]
    @test FF.death_downsets(pi)[1] === Downs[1]
    @test PLB.nregions(pi) == length(pi.reps)
    @test PLB.generator_counts(pi) == (; upsets=1, downsets=1)
    @test PLB.critical_coordinate_counts(pi) == (2,)
    @test PLB.axes_uniformity(pi) == (true,)
    @test PLB.critical_coordinates(pi) == pi.coords
    @test PLB.region_representatives(pi) == pi.reps
    @test PLB.region_representative(pi, rid) == pi.reps[rid]
    sig = PLB.region_signature(pi, rid)
    @test sig isa NamedTuple
    @test Set(keys(sig)) == Set((:y, :z))
    @test PLB.cell_shape(pi) == pi.cell_shape
    @test PLB.has_direct_lookup(pi)

    sum_pi = PLB.box_encoding_summary(pi)
    sum_enc = PLB.box_encoding_summary(enc)
    @test sum_pi.kind == :pl_backend_encoding_map
    @test sum_pi.generator_counts == (; upsets=1, downsets=1)
    @test sum_pi.nregions == PLB.nregions(pi)
    @test sum_pi.cell_shape == pi.cell_shape
    @test sum_pi.direct_lookup_enabled
    @test sum_pi.all_axes_uniform
    @test sum_enc.compiled
    @test sum_enc.poset_size == FF.nvertices(P)
    @test occursin("PLEncodingMapBoxes", sprint(show, MIME("text/plain"), pi))

    good_map = PLB.check_box_encoding_map(pi)
    bad_map = PLB.check_box_encoding_map(:not_a_box_encoding)
    @test good_map.valid
    @test !bad_map.valid
    @test PLB.plbackend_validation_summary(good_map).report.kind == :box_encoding_map
    @test occursin("PLBackendValidationSummary",
                   sprint(show, MIME("text/plain"), PLB.plbackend_validation_summary(good_map)))
    @test_throws ArgumentError PLB.check_box_encoding_map(:not_a_box_encoding; throw=true)

    qpoint = PLB.check_box_point(pi, [1.0])
    qmatrix = PLB.check_box_points(pi, reshape([0.0, 1.0, 2.0], 1, :))
    qbox = PLB.check_box_query_box(pi, box)
    qregion = PLB.check_box_region(pi, rid; box=box)
    @test qpoint.valid
    @test qmatrix.valid
    @test qbox.valid
    @test qregion.valid
    @test qregion.finite_box_required
    @test qregion.signature_support_sizes !== nothing
    @test qregion.bbox isa Tuple
    @test !PLB.check_box_point(pi, [1.0, 2.0]).valid
    @test !PLB.check_box_points(pi, reshape([0.0, 1.0], 2, 1)).valid
    @test !PLB.check_box_query_box(pi, (Float64[2.0], Float64[0.0])).valid
    @test !PLB.check_box_region(pi, 0).valid
    @test_throws ArgumentError PLB.check_box_point(pi, [1.0, 2.0]; throw=true)
    @test_throws ArgumentError PLB.check_box_points(pi, reshape([0.0, 1.0], 2, 1); throw=true)
    @test_throws ArgumentError PLB.check_box_query_box(pi, (Float64[2.0], Float64[0.0]); throw=true)
    @test_throws ArgumentError PLB.check_box_region(pi, 0; throw=true)

    qsum = PLB.box_query_summary(pi, [1.0])
    rsum = PLB.box_region_summary(pi, rid; box=box)
    @test qsum.region == rid
    @test qsum.representative == PLB.region_representative(pi, rid)
    @test qsum.outside == false
    @test rsum.region == rid
    @test rsum.signature_support_sizes == qregion.signature_support_sizes
    @test rsum.bbox isa Tuple

    @test TOA.PLEncodingMapBoxes === PLB.PLEncodingMapBoxes
    @test TOA.lower_bounds(Ups[1]) == [0.0]
    @test TOA.upper_bounds(Downs[1]) == [2.0]
    @test TOA.generator_counts(pi) == (; upsets=1, downsets=1)
    @test TOA.critical_coordinate_counts(pi) == (2,)
    @test TOA.axes_uniformity(pi) == (true,)
    @test TOA.check_box_encoding_map(pi).valid
    @test TOA.check_box_point(pi, [1.0]).valid
    @test TOA.check_box_region(pi, rid; box=box).valid
    @test TOA.box_encoding_summary(enc).compiled
    @test TOA.box_query_summary(pi, [1.0]).region == rid
    @test TOA.box_region_summary(pi, rid; box=box).region == rid
end

@testset "PLBackend axis encoding (axis-aligned boxes)" begin
    enc_axis = TO.EncodingOptions(backend=:pl_backend)
    enc_axis_small = TO.EncodingOptions(backend=:pl_backend, max_regions=2)

    # One-dimensional example:
    # Birth upset:  x >= 0
    # Death downset: x <= 2
    # phi = [1], so the represented module should be supported exactly in the middle region.
    Ups = [PLB.BoxUpset([0.0])]
    Downs = [PLB.BoxDownset([2.0])]
    Phi = reshape(K[c(1)], 1, 1)

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
        # Use the generic exported `locate` via the constant module `TO` so the
        # call is fully inferred (otherwise a module-valued `PLB` can force
        # dynamic dispatch and show spurious allocations).
        EC.locate(pi, (1.0,))
        @test (@allocated EC.locate(pi, (1.0,))) == 0

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
        Phi2 = reshape(K[c(1)], 1, 1)

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
        # Same note as in 1D: use the exported `locate` via `TO` to avoid
        # dynamic module-property dispatch.
        EC.locate(pi2, (1.0, 1.0))
        @test (@allocated EC.locate(pi2, (1.0, 1.0))) == 0
    end

    @testset "incremental signature traversal consistency" begin
        Ups3 = [PLB.BoxUpset([0.0, 0.0]), PLB.BoxUpset([2.0, -1.0])]
        Downs3 = [PLB.BoxDownset([3.0, 1.0]), PLB.BoxDownset([1.0, 2.0])]
        Phi3 = zeros(K, length(Downs3), length(Ups3))

        _, _, pi3 = PLB.encode_fringe_boxes(Ups3, Downs3, Phi3, enc_axis)
        MY = cld(length(Ups3), 64)
        MZ = cld(length(Downs3), 64)

        lin = 1
        for I in CartesianIndices(Tuple(pi3.cell_shape))
            idx0 = ntuple(j -> I[j] - 1, pi3.n)
            x = PLB._cell_rep_axis(pi3.coords, idx0)
            y, z = PLB._signature(x, Ups3, Downs3)
            key = PLB._sigkey_from_bitvectors(y, z, Val(MY), Val(MZ))
            rid = get(pi3.sig_to_region, key, 0)
            @test rid != 0
            @test pi3.cell_to_region[lin] == rid
            lin += 1
        end
    end

end

@testset "PLBackend near-boundary parity (fast vs verified)" begin
    Ups = [PLB.BoxUpset([0.0, 0.0])]
    Downs = [PLB.BoxDownset([2.0, 2.0])]
    enc = TO.encode(Ups, Downs; backend=:pl_backend, field=field, output=:result, cache=:auto)

    epss = (1e-4, 1e-7, 1e-10, 1e-12)
    probes = Vector{Tuple{Float64,Float64}}()
    for e in epss
        push!(probes, (0.0 + e, 1.0))
        push!(probes, (0.0 - e, 1.0))
        push!(probes, (2.0 + e, 1.0))
        push!(probes, (2.0 - e, 1.0))
        push!(probes, (1.0, 0.0 + e))
        push!(probes, (1.0, 0.0 - e))
        push!(probes, (1.0, 2.0 + e))
        push!(probes, (1.0, 2.0 - e))
    end
    # Exact boundary probes included too.
    append!(probes, ((0.0, 1.0), (2.0, 1.0), (1.0, 0.0), (1.0, 2.0), (0.0, 0.0), (2.0, 2.0)))

    for x in probes
        rf = EC.locate(enc.pi, x; mode=:fast)
        rv = EC.locate(enc.pi, x; mode=:verified)
        @test rf == rv
        @test rf != 0
    end
end

@testset "PL runtime budget guards (deterministic, QQ)" begin
    if field isa CM.QQField
        @inline function _median_elapsed(f::Function; reps::Int=5)
            ts = Vector{Float64}(undef, reps)
            for i in 1:reps
                ts[i] = @elapsed f()
            end
            return sort(ts)[cld(reps, 2)]
        end
        @inline _ns_per_item(t::Float64, n::Int) = (t * 1.0e9) / max(1, n)
        strict_ci = get(ENV, "TAMER_STRICT_PERF_CI", "1") == "1"

        # PLBackend hot locate loop: fixed workload + conservative budget.
        Ups = [PLB.BoxUpset([0.0])]
        Downs = [PLB.BoxDownset([2.0])]
        enc_axis = TO.encode(Ups, Downs; backend=:pl_backend, output=:result, cache=:auto)
        xs = range(-2.0, 7.0; length=20_000)
        EC.locate(enc_axis.pi, (0.5,); mode=:fast) # warmup
        EC.locate(enc_axis.pi, (0.5,); mode=:verified) # warmup
        axis_fast = _median_elapsed() do
            s = 0
            @inbounds for x in xs
                s += EC.locate(enc_axis.pi, (x,); mode=:fast)
            end
            @test s > 0
        end
        axis_verified = _median_elapsed() do
            s = 0
            @inbounds for x in xs
                s += EC.locate(enc_axis.pi, (x,); mode=:verified)
            end
            @test s > 0
        end
        # Platform-normalized envelope (ns/query): strict in CI, looser local fallback.
        axis_fast_ns = _ns_per_item(axis_fast, length(xs))
        axis_verified_ns = _ns_per_item(axis_verified, length(xs))
        if strict_ci
            @test axis_fast_ns <= 1.15 * axis_verified_ns + 10.0
        else
            @test axis_fast_ns <= 1.3 * axis_verified_ns + 20.0
        end

        if PLP.HAVE_POLY
            # PLPolyhedra cached-vs-uncached locate_many! guard on fixed points.
            A1 = QQ[1 0; 0 1; -1 0; 0 -1]
            b1 = QQ[1, 1, 0, 0]
            A2 = QQ[1 0; 0 1; -1 0; 0 -1]
            b2 = QQ[2, 1, -1, 0]
            hp1 = PLP.make_hpoly(A1, b1)
            hp2 = PLP.make_hpoly(A2, b2)
            pi = PLP.PLEncodingMap(2,
                                   [BitVector(), BitVector()],
                                   [BitVector(), BitVector()],
                                   [hp1, hp2],
                                   [(0.5, 0.5), (1.5, 0.5)])
            box = (Float64[0, 0], Float64[2, 1])
            cache = PLP.compile_geometry_cache(pi; box=box, closure=true)

            npts = 12_000
            X = Matrix{Float64}(undef, 2, npts)
            @inbounds for j in 1:npts
                # Deterministic pseudo-grid samples.
                X[1, j] = (j % 2000) / 1000
                X[2, j] = ((j * 7) % 1000) / 1000
            end
            dest_uncached = zeros(Int, npts)
            dest_cached = zeros(Int, npts)

            # Warmup
            PLP.locate_many!(dest_uncached, pi, X; threaded=false, mode=:fast)
            PLP.locate_many!(dest_cached, cache, X; threaded=false, mode=:fast)
            t_uncached = _median_elapsed() do
                PLP.locate_many!(dest_uncached, pi, X; threaded=false, mode=:fast)
            end
            t_cached = _median_elapsed() do
                PLP.locate_many!(dest_cached, cache, X; threaded=false, mode=:fast)
            end

            @test dest_cached == dest_uncached
            many_uncached_ns = _ns_per_item(t_uncached, npts)
            many_cached_ns = _ns_per_item(t_cached, npts)
            # Platform-normalized envelope: cache should be faster on per-query cost.
            if strict_ci
                @test many_cached_ns <= 1.10 * many_uncached_ns + 20.0
            else
                @test many_cached_ns <= 1.2 * many_uncached_ns + 30.0
            end
        end
    end
end

@testset "PLPolyhedra adversarial degenerate geometry fixtures (QQ)" begin
    if !PLP.HAVE_POLY || !(field isa CM.QQField)
        @test true
    else
        # Fixture A: near-coplanar thin slab in 3D.
        eps = 1.0e-6
        A3 = QQ[1 0 0; 0 1 0; 0 0 1; -1 0 0; 0 -1 0; 0 0 -1]
        b3 = QQ[1, 1, eps, 0, 0, 0]
        hp3 = PLP.make_hpoly(A3, b3)
        pi3 = PLP.PLEncodingMap(3, [BitVector([true])], [BitVector([false])], [hp3], [(0.5, 0.5, eps / 2)])
        box3 = (Float64[0.0, 0.0, 0.0], Float64[1.0, 1.0, eps])
        r3 = EC.locate(pi3, [0.5, 0.5, eps / 2]; mode=:verified)
        @test r3 != 0
        w3 = TO.RegionGeometry.region_weights(pi3; box=box3, method=:exact)
        @test isapprox(w3[r3], eps; atol=1e-11)
        @test TO.RegionGeometry.region_bbox(pi3, r3; box=box3) == (Float64[0.0, 0.0, 0.0], Float64[1.0, 1.0, eps])

        for d in (1.0e-7, 1.0e-10, 1.0e-12)
            @test EC.locate(pi3, [0.5, 0.5, eps - d]; mode=:fast) == EC.locate(pi3, [0.5, 0.5, eps - d]; mode=:verified)
            @test EC.locate(pi3, [0.5, 0.5, eps + d]; mode=:fast) == EC.locate(pi3, [0.5, 0.5, eps + d]; mode=:verified)
        end

        # Fixture B: high-dimensional thin orthotope in 4D.
        t = 1.0e-4
        A4 = QQ[
            1 0 0 0;
            0 1 0 0;
            0 0 1 0;
            0 0 0 1;
            -1 0 0 0;
            0 -1 0 0;
            0 0 -1 0;
            0 0 0 -1
        ]
        b4 = QQ[1, 1, t, t, 0, 0, 0, 0]
        hp4 = PLP.make_hpoly(A4, b4)
        pi4 = PLP.PLEncodingMap(4, [BitVector([true])], [BitVector([false])], [hp4], [(0.5, 0.5, t / 2, t / 2)])
        box4 = (Float64[0.0, 0.0, 0.0, 0.0], Float64[1.0, 1.0, t, t])
        r4 = EC.locate(pi4, [0.5, 0.5, t / 2, t / 2]; mode=:verified)
        @test r4 != 0
        @test TO.RegionGeometry.region_bbox(pi4, r4; box=box4) == (Float64[0.0, 0.0, 0.0, 0.0], Float64[1.0, 1.0, t, t])
        @test isapprox(
            TO.RegionGeometry.region_diameter(pi4, r4; box=box4, metric=:L2, method=:bbox),
            sqrt(2.0 + 2.0 * t * t);
            atol=1e-10,
        )

        probes4 = (
            [0.5, 0.5, 0.0, t / 2],
            [0.5, 0.5, t, t / 2],
            [0.5, 0.5, t + 1.0e-8, t / 2],
            [0.5, 0.5, t / 2, t + 1.0e-8],
        )
        for x in probes4
            @test EC.locate(pi4, x; mode=:fast) == EC.locate(pi4, x; mode=:verified)
        end
    end
end

    end # field isa CM.QQField

    if !(field isa CM.QQField)
@testset "PL non-QQ field parity ($(field))" begin
    # Axis-aligned backend with non-QQ target field: geometry should stay the same,
    # while encoded module coefficients are coerced to the requested field.
    Ups = [PLB.BoxUpset([0.0])]
    Downs = [PLB.BoxDownset([2.0])]
    enc = TO.encode(Ups, Downs; backend=:pl_backend, field=field, output=:result, cache=:auto)

    @test enc.M.field == field
    @test enc.H.field == field

    box = ([-2.0], [7.0])
    w = TO.RegionGeometry.region_weights(enc.pi; box=box)
    t_left = EC.locate(enc.pi, [-1.0])
    t_mid = EC.locate(enc.pi, [1.0])
    t_right = EC.locate(enc.pi, [3.0])

    @test FF.fiber_dimension(enc.H, t_left) == 0
    @test FF.fiber_dimension(enc.H, t_mid) == 1
    @test FF.fiber_dimension(enc.H, t_right) == 0

    @test isapprox(w[t_left], 2.0; atol=1e-12)
    @test isapprox(w[t_mid], 2.0; atol=1e-12)
    @test isapprox(w[t_right], 5.0; atol=1e-12)
    @test isapprox(sum(w), 9.0; atol=1e-12)
end

@testset "PLBackend mode parity for non-QQ fields ($(field))" begin
    Ups = [PLB.BoxUpset([0.0, 0.0])]
    Downs = [PLB.BoxDownset([2.0, 2.0])]
    enc = TO.encode(Ups, Downs; backend=:pl_backend, field=field, output=:result, cache=:auto)
    box = ([-1.0, -1.0], [3.0, 3.0])

    adj_fast = TO.RegionGeometry.region_adjacency(enc.pi; box=box, strict=true, mode=:fast)
    adj_verified = TO.RegionGeometry.region_adjacency(enc.pi; box=box, strict=true, mode=:verified)
    @test adj_fast == adj_verified

    # Boundary and interior points should classify identically in both modes.
    @test EC.locate(enc.pi, [0.0, 1.0]; mode=:fast) == EC.locate(enc.pi, [0.0, 1.0]; mode=:verified)
    @test EC.locate(enc.pi, [2.0, 1.0]; mode=:fast) == EC.locate(enc.pi, [2.0, 1.0]; mode=:verified)
    @test EC.locate(enc.pi, [1.0, 1.0]; mode=:fast) == EC.locate(enc.pi, [1.0, 1.0]; mode=:verified)
end

@testset "PLPolyhedra non-QQ coercion and geometry parity ($(field))" begin
    if !PLP.HAVE_POLY
        @test true
    else
        A_up = -Matrix{QQ}(I, 1, 1)
        b_up = QQ[0]
        A_down = Matrix{QQ}(I, 1, 1)
        b_down = QQ[2]
        hp_up = PLP.make_hpoly(A_up, b_up)
        hp_down = PLP.make_hpoly(A_down, b_down)
        F = PLP.PLFringe(
            [PLP.PLUpset(PLP.PolyUnion(1, [hp_up]))],
            [PLP.PLDownset(PLP.PolyUnion(1, [hp_down]))],
            reshape(QQ[1], 1, 1),
        )
        enc_opts = TO.EncodingOptions(backend=:pl, field=field)
        enc = TO.encode(F, enc_opts; output=:result, cache=:auto)

        @test enc.M.field == field
        @test enc.H.field == field

        box = ([-2.0], [7.0])
        w = TO.RegionGeometry.region_weights(enc.pi; box=box, method=:exact)
        t_left = EC.locate(enc.pi, [-1.0]; mode=:verified)
        t_mid = EC.locate(enc.pi, [1.0]; mode=:verified)
        t_right = EC.locate(enc.pi, [3.0]; mode=:verified)

        @test FF.fiber_dimension(enc.H, t_left) == 0
        @test FF.fiber_dimension(enc.H, t_mid) == 1
        @test FF.fiber_dimension(enc.H, t_right) == 0

        @test isapprox(w[t_left], 2.0; atol=1e-9)
        @test isapprox(w[t_mid], 2.0; atol=1e-9)
        @test isapprox(w[t_right], 5.0; atol=1e-9)
        @test isapprox(sum(w), 9.0; atol=1e-8)
    end
end

@testset "PLPolyhedra near-boundary parity (fast vs verified, $(field))" begin
    if !PLP.HAVE_POLY
        @test true
    else
        A = reshape(QQ[1, -1], 2, 1)
        hp1 = PLP.make_hpoly(A, QQ[1, 0])   # [0,1]
        hp2 = PLP.make_hpoly(A, QQ[2, -1])  # [1,2]
        pi = PLP.PLEncodingMap(1,
                               [BitVector([false]), BitVector([true])],
                               [BitVector([false]), BitVector([false])],
                               [hp1, hp2],
                               [(0.5,), (1.5,)])

        epss = (1e-4, 1e-7, 1e-10, 1e-12)
        probes = Float64[0.0, 1.0, 2.0]
        for e in epss
            append!(probes, (0.0 - e, 0.0 + e, 1.0 - e, 1.0 + e, 2.0 - e, 2.0 + e))
        end
        for x in probes
            rf = PLP.locate(pi, [x]; mode=:fast)
            rv = PLP.locate(pi, [x]; mode=:verified)
            @test rf == rv
        end
    end
end

@testset "PL non-QQ deep geometry hand-oracles ($(field))" begin
    if field isa CM.QQField || !PLP.HAVE_POLY
        @test true
    else
        # 2D unit square exact oracle.
        A2 = QQ[1 0; 0 1; -1 0; 0 -1]
        b2 = QQ[1, 1, 0, 0]
        hp2 = PLP.make_hpoly(A2, b2)
        pi2 = PLP.PLEncodingMap(2, [BitVector([true])], [BitVector([false])], [hp2], [(0.5, 0.5)])
        box2 = (Float64[0.0, 0.0], Float64[1.0, 1.0])
        r2 = EC.locate(pi2, [0.5, 0.5]; mode=:verified)
        @test r2 != 0
        w2 = TO.RegionGeometry.region_weights(pi2; box=box2, method=:exact)
        @test isapprox(w2[r2], 1.0; atol=1e-10)
        @test isapprox(sum(w2), 1.0; atol=1e-10)
        @test TO.RegionGeometry.region_bbox(pi2, r2; box=box2) == (Float64[0.0, 0.0], Float64[1.0, 1.0])
        @test isapprox(TO.RegionGeometry.region_diameter(pi2, r2; box=box2, metric=:L2, method=:bbox), sqrt(2.0); atol=1e-10)
        @test isapprox(TO.RegionGeometry.region_boundary_measure(pi2, r2; box=box2, mode=:verified), 4.0; atol=1e-8)
        @test isempty(TO.RegionGeometry.region_adjacency(pi2; box=box2, strict=true, mode=:verified))

        # 3D unit cube exact oracle.
        A3 = QQ[1 0 0; 0 1 0; 0 0 1; -1 0 0; 0 -1 0; 0 0 -1]
        b3 = QQ[1, 1, 1, 0, 0, 0]
        hp3 = PLP.make_hpoly(A3, b3)
        pi3 = PLP.PLEncodingMap(3, [BitVector([true])], [BitVector([false])], [hp3], [(0.5, 0.5, 0.5)])
        box3 = (Float64[0.0, 0.0, 0.0], Float64[1.0, 1.0, 1.0])
        r3 = EC.locate(pi3, [0.5, 0.5, 0.5]; mode=:verified)
        @test r3 != 0
        w3 = TO.RegionGeometry.region_weights(pi3; box=box3, method=:exact)
        @test isapprox(w3[r3], 1.0; atol=1e-10)
        @test isapprox(TO.RegionGeometry.region_boundary_measure(pi3, r3; box=box3, mode=:verified), 6.0; atol=1e-8)
        @test isapprox(TO.RegionGeometry.region_diameter(pi3, r3; box=box3, metric=:L2, method=:bbox), sqrt(3.0); atol=1e-10)
        @test isempty(TO.RegionGeometry.region_adjacency(pi3; box=box3, strict=true, mode=:verified))
    end
end
    end
end # with_fields
