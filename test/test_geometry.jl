using Test
using Random

# Included from test/runtests.jl; uses shared aliases (TO, FF, PLP, PLB, QQ, ...).

const _REGION_HOOK_STUB = Val(:region_hook_stub)
TO.RegionGeometry.region_weights(::Val{:region_hook_stub}; kwargs...) = error("slow weights path should not run")
TO.RegionGeometry.region_bbox(::Val{:region_hook_stub}, ::Integer; kwargs...) = error("slow bbox path should not run")
TO.RegionGeometry._region_volume_fast(::Val{:region_hook_stub}, ::Integer; box, closure::Bool=true, cache=nothing) = 3.5
TO.RegionGeometry._region_centroid_fast(::Val{:region_hook_stub}, ::Integer; box, method::Symbol=:bbox, closure::Bool=true, cache=nothing) = [0.25, 0.75]
TO.RegionGeometry._region_circumradius_fast(::Val{:region_hook_stub}, ::Integer;
    box, center=:bbox, metric::Symbol=:L2, method::Symbol=:bbox,
    strict::Bool=true, closure::Bool=true, cache=nothing) = 4.25
TO.RegionGeometry._region_minkowski_functionals_fast(::Val{:region_hook_stub}, ::Integer;
    box, volume=nothing, boundary=nothing, mean_width_method::Symbol=:auto,
    mean_width_ndirs::Integer=256, mean_width_rng=Random.default_rng(),
    mean_width_directions=nothing, strict::Bool=true, closure::Bool=true,
    cache=nothing) = (volume=3.5, boundary_measure=7.0, mean_width=1.25)

with_fields(FIELDS_FULL) do field
K = CM.coeff_type(field)
@inline c(x) = CM.coerce(field, x)
@inline _batch_count(be) = be isa AbstractMatrix ? size(be, 2) : length(be)

if field isa CM.QQField
@testset "Region geometry for PLBackend (axis-aligned boxes)" begin

    # 1D example: one upset x >= 0 and one downset x <= 2.
    Ups = [PLB.BoxUpset([0.0])]
    Downs = [PLB.BoxDownset([2.0])]
    Phi = reshape(K[c(1)], 1, 1)  # 1x1 matrix

    P, H, pi = PLB.encode_fringe_boxes(Ups, Downs, Phi)

    # Volume weights in the box [-2, 7] (length 9).
    box = ([-2.0], [7.0])
    w = TO.RegionGeometry.region_weights(pi; box=box)

    @test length(w) == P.n
    @test isapprox(sum(w), 9.0; atol=1e-12)

    t_left  = PLB.locate(pi, [-1.0])  # x < 0
    t_mid   = PLB.locate(pi, [ 1.0])  # 0 < x < 2
    t_right = PLB.locate(pi, [ 3.0])  # x > 2

    @test isapprox(w[t_left],  2.0; atol=1e-12)
    @test isapprox(w[t_mid],   2.0; atol=1e-12)
    @test isapprox(w[t_right], 5.0; atol=1e-12)

    # Bounding box and diameter for the middle region inside the ambient box.
    bb_mid = TO.RegionGeometry.region_bbox(pi, t_mid; box=box)
    @test bb_mid !== nothing
    lo, hi = bb_mid
    @test lo == [0.0]
    @test hi == [2.0]

    @test isapprox(TO.RegionGeometry.region_diameter(pi, t_mid; box=box, metric=:L2),   2.0; atol=1e-12)
    @test isapprox(TO.RegionGeometry.region_diameter(pi, t_mid; box=box, metric=:Linf), 2.0; atol=1e-12)
    @test isapprox(TO.RegionGeometry.region_diameter(pi, t_mid; box=box, metric=:L1),   2.0; atol=1e-12)

    # 2D example where a region is a UNION of multiple grid cells:
    # one downset D = {x <= 0, y <= 0}. Then z = !contains(D, x) is true outside that quadrant.
    Ups2 = PLB.BoxUpset[]
    Downs2 = [PLB.BoxDownset([0.0, 0.0])]
    Phi2 = zeros(K, 1, 0)  # r=1, m=0

    P2, H2, pi2 = PLB.encode_fringe_boxes(Ups2, Downs2, Phi2)

    box2 = ([-1.0, -1.0], [1.0, 1.0])
    w2 = TO.RegionGeometry.region_weights(pi2; box=box2)
    @test length(w2) == P2.n
    @test isapprox(sum(w2), 4.0; atol=1e-12)

    t_in  = PLB.locate(pi2, [-0.5, -0.5])  # inside D
    t_out = PLB.locate(pi2, [ 0.5,  0.5])  # outside D (union of 3 cells)

    @test isapprox(w2[t_in],  1.0; atol=1e-12)
    @test isapprox(w2[t_out], 3.0; atol=1e-12)

    bb_in = TO.RegionGeometry.region_bbox(pi2, t_in; box=box2)
    @test bb_in !== nothing
    lo_in, hi_in = bb_in
    @test lo_in == [-1.0, -1.0]
    @test hi_in == [0.0, 0.0]
    @test isapprox(TO.RegionGeometry.region_diameter(pi2, t_in; box=box2, metric=:Linf), 1.0; atol=1e-12)
    @test isapprox(TO.RegionGeometry.region_diameter(pi2, t_in; box=box2, metric=:L2),   sqrt(2.0); atol=1e-12)

    bb_out = TO.RegionGeometry.region_bbox(pi2, t_out; box=box2)
    @test bb_out !== nothing
    lo_out, hi_out = bb_out
    @test lo_out == [-1.0, -1.0]
    @test hi_out == [1.0, 1.0]
    @test isapprox(TO.RegionGeometry.region_diameter(pi2, t_out; box=box2, metric=:Linf), 2.0; atol=1e-12)
    @test isapprox(TO.RegionGeometry.region_diameter(pi2, t_out; box=box2, metric=:L2),   2.0 * sqrt(2.0); atol=1e-12)
end

@testset "PL mode contract is strict" begin
    Ups = [PLB.BoxUpset([0.0])]
    Downs = [PLB.BoxDownset([1.0])]
    Phi = zeros(K, 1, 1)
    _, _, pi = PLB.encode_fringe_boxes(Ups, Downs, Phi)

    @test PLB.locate(pi, [0.25]; mode=:fast) != 0
    @test PLB.locate(pi, [0.25]; mode=:verified) != 0
    @test_throws ArgumentError PLB.locate(pi, [0.25]; mode=:hybrid_fast)
    @test_throws ArgumentError PLB.locate(pi, [0.25]; mode=:hybrid_verified)
    @test_throws ArgumentError OPT.InvariantOptions(pl_mode=:hybrid_fast)
    @test_throws ArgumentError OPT.InvariantOptions(pl_mode=:exact)
end
end

if field isa CM.QQField
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
    pi = PLP.PLEncodingMap(1, sigy, sigz, [hp1, hp2], [(0.5,), (1.5,)])

    @test PLP.locate(pi, [0.25]) != 0
    @test PLP.locate(pi, [1.75]) != 0

    box = ([0.0], [2.0])

    # Default: uniform weights if no box.
    wunif = TO.RegionGeometry.region_weights(pi)
    @test length(wunif) == 2
    @test wunif == [1.0, 1.0]

    # Monte Carlo weights inside [0,2] should be close to [1,1].
    rng = MersenneTwister(12345)
    # Default is :exact (requires Polyhedra). For this test we want Monte Carlo.
    w = TO.RegionGeometry.region_weights(pi; box=box, method=:mc, nsamples=20_000, rng=rng, strict=true)

    t1 = PLP.locate(pi, [0.5])
    t2 = PLP.locate(pi, [1.5])

    @test isapprox(w[t1], 1.0; atol=0.05)
    @test isapprox(w[t2], 1.0; atol=0.05)
    @test isapprox(sum(w), 2.0; atol=0.05)

    # Exact bbox by vertex enumeration in 1D.
    bb1 = TO.RegionGeometry.region_bbox(pi, t1; box=box)
    @test bb1 !== nothing
    lo1, hi1 = bb1
    @test lo1 == [0.0]
    @test hi1 == [1.0]

    # Diameter by bbox and by vertices should match in 1D.
    @test isapprox(TO.RegionGeometry.region_diameter(pi, t1; box=box, metric=:L2, method=:bbox),     1.0; atol=1e-12)
    @test isapprox(TO.RegionGeometry.region_diameter(pi, t1; box=box, metric=:L2, method=:vertices), 1.0; atol=1e-12)

    # Batch locate API parity (serial/threaded and cache/non-cache).
    Xq = [0.25 0.50 0.75 1.25 1.50 1.75;
          0.00 0.00 0.00 0.00 0.00 0.00]
    expected = [PLP.locate(pi, [Xq[1, j]]) for j in 1:size(Xq, 2)]
    expected_verified = [PLP.locate(pi, [Xq[1, j]]; mode=:verified) for j in 1:size(Xq, 2)]
    @test expected_verified == expected
    dest_serial = fill(0, size(Xq, 2))
    dest_threaded = fill(0, size(Xq, 2))
    PLP.locate_many!(dest_serial, pi, Xq[1:1, :]; threaded=false)
    PLP.locate_many!(dest_threaded, pi, Xq[1:1, :]; threaded=true)
    @test dest_serial == expected
    @test dest_threaded == expected

    dest_verified = fill(0, size(Xq, 2))
    PLP.locate_many!(dest_verified, pi, Xq[1:1, :]; threaded=false, mode=:verified)
    @test dest_verified == expected

    if PLP.HAVE_POLY
        cache = PLP.compile_geometry_cache(pi; box=box)
        dest_cache = fill(0, size(Xq, 2))
        PLP.locate_many!(dest_cache, cache, Xq[1:1, :]; threaded=false)
        @test dest_cache == expected
        dest_cache_verified = fill(0, size(Xq, 2))
        PLP.locate_many!(dest_cache_verified, cache, Xq[1:1, :]; threaded=false, mode=:verified)
        @test dest_cache_verified == expected
    end
end
end

if field isa CM.QQField
@testset "PL normalized performance envelopes (QQ)" begin
    @inline function _median_elapsed(f::Function; reps::Int=5)
        ts = Vector{Float64}(undef, reps)
        for i in 1:reps
            ts[i] = @elapsed f()
        end
        return sort(ts)[cld(reps, 2)]
    end
    @inline _ns_per_item(t::Float64, n::Int) = (t * 1.0e9) / max(1, n)
    strict_ci = get(ENV, "TAMER_STRICT_PERF_CI", "1") == "1"

    # PLBackend locate: compare fast vs verified with normalized ns/query envelopes.
    Ups = [PLB.BoxUpset([0.0])]
    Downs = [PLB.BoxDownset([2.0])]
    Phi = reshape(QQ[1], 1, 1)
    _, _, pi_axis = PLB.encode_fringe_boxes(Ups, Downs, Phi)

    xs = range(-2.0, 7.0; length=12_000)
    EC.locate(pi_axis, (0.5,); mode=:fast)      # warmup
    EC.locate(pi_axis, (0.5,); mode=:verified)  # warmup

    t_fast = _median_elapsed() do
        s = 0
        @inbounds for x in xs
            s += EC.locate(pi_axis, (x,); mode=:fast)
        end
        @test s > 0
    end
    t_verified = _median_elapsed() do
        s = 0
        @inbounds for x in xs
            s += EC.locate(pi_axis, (x,); mode=:verified)
        end
        @test s > 0
    end
    fast_ns = _ns_per_item(t_fast, length(xs))
    verified_ns = _ns_per_item(t_verified, length(xs))
    if strict_ci
        @test fast_ns <= 1.15 * verified_ns + 10.0
    else
        @test fast_ns <= 1.3 * verified_ns + 20.0
    end

    if PLP.HAVE_POLY
        # PLPolyhedra locate_many!: cached should improve per-query cost.
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

        npts = 8_000
        X = Matrix{Float64}(undef, 2, npts)
        @inbounds for j in 1:npts
            X[1, j] = (j % 2000) / 1000
            X[2, j] = ((j * 7) % 1000) / 1000
        end
        dest_uncached = zeros(Int, npts)
        dest_cached = zeros(Int, npts)

        PLP.locate_many!(dest_uncached, pi, X; threaded=false, mode=:fast)       # warmup
        PLP.locate_many!(dest_cached, cache, X; threaded=false, mode=:fast)      # warmup

        t_uncached = _median_elapsed() do
            PLP.locate_many!(dest_uncached, pi, X; threaded=false, mode=:fast)
        end
        t_cached = _median_elapsed() do
            PLP.locate_many!(dest_cached, cache, X; threaded=false, mode=:fast)
        end

        @test dest_uncached == dest_cached
        uncached_ns = _ns_per_item(t_uncached, npts)
        cached_ns = _ns_per_item(t_cached, npts)
        if strict_ci
            @test cached_ns <= 1.45 * uncached_ns + 25.0
        else
            @test cached_ns <= 1.75 * uncached_ns + 40.0
        end
    end
end
end

if field isa CM.QQField
@testset "PLPolyhedra heavy geometry: perimeter/surface/adjacency/PCA" begin
    if !PLP.HAVE_POLY
        @test true
    else
        # 2D: square [0,1]^2 perimeter should be 4
        A = K[ 1 0;
                0 1;
               -1 0;
                0 -1 ]
        b = K[1, 1, 0, 0]
        hp = PLP.make_hpoly(A, b)

        sigy = [BitVector()]
        sigz = [BitVector()]
        pi = PLP.PLEncodingMap(2, sigy, sigz, [hp], [(0.5, 0.5)])

        box = (Float64[0.0, 0.0], Float64[1.0, 1.0])
        perim = TO.RegionGeometry.region_perimeter(pi, 1; box=box)
        @test isapprox(perim, 4.0; atol=1e-9)

        # 3D: cube [0,1]^3 surface area is 6
        A3 = K[ 1 0 0;
                 0 1 0;
                 0 0 1;
                -1 0 0;
                 0 -1 0;
                 0 0 -1 ]
        b3 = K[1, 1, 1, 0, 0, 0]
        hp3 = PLP.make_hpoly(A3, b3)
        pi3 = PLP.PLEncodingMap(3, [BitVector()], [BitVector()], [hp3], [(0.5, 0.5, 0.5)])
        box3 = (Float64[0,0,0], Float64[1,1,1])
        sa = TO.RegionGeometry.region_surface_area(pi3, 1; box=box3)
        @test isapprox(sa, 6.0; atol=1e-9)

        # Adjacency: two rectangles sharing a vertical edge of length 1
        # R1 = [0,1]x[0,1], R2 = [1,2]x[0,1]
        A1 = K[ 1 0; 0 1; -1 0; 0 -1 ]
        b1 = K[1, 1, 0, 0]
        A2 = K[ 1 0; 0 1; -1 0; 0 -1 ]
        b2 = K[2, 1, -1, 0]
        hp1 = PLP.make_hpoly(A1, b1)
        hp2 = PLP.make_hpoly(A2, b2)
        sigy2 = [BitVector(), BitVector()]
        sigz2 = [BitVector(), BitVector()]
        pi2 = PLP.PLEncodingMap(2, sigy2, sigz2,
                                        [hp1, hp2],
                                        [(0.5,0.5), (1.5,0.5)])
        box2 = (Float64[0,0], Float64[2,1])
        adj = TO.RegionGeometry.region_adjacency(pi2; box=box2)
        @test haskey(adj, (1,2))
        @test isapprox(adj[(1,2)], 1.0; atol=1e-8)

        # --- PCA / principal directions diagnostics ---
        rng = MersenneTwister(1)
        pca = TO.RegionGeometry.region_principal_directions(pi, 1; box=box, nsamples=5000, rng=rng, strict=true)

        @test :n_accepted in propertynames(pca)
        @test :n_proposed in propertynames(pca)

        @test pca.n_accepted == 5000
        @test pca.n_proposed == 5000

        @test isapprox(pca.mean[1], 0.5; atol=0.02)
        @test isapprox(pca.mean[2], 0.5; atol=0.02)
        @test isapprox(pca.evals[1], 1/12; atol=0.02)
        @test isapprox(pca.evals[2], 1/12; atol=0.02)

        rng2 = MersenneTwister(2)
        pca_info = TO.RegionGeometry.region_principal_directions(pi, 1;
            box=box, nsamples=4000, nbatches=4, rng=rng2, strict=true, return_info=true)

        @test pca_info.nbatches == 4
        @test _batch_count(pca_info.batch_evals) == 4
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
        w_exact = TO.RegionGeometry.region_weights(pi; box=box, method=:exact)
        @test length(w_exact) == 1
        @test isapprox(w_exact[1], 1.0; atol=1e-12, rtol=0.0)

        # (2) cache: results should match (and box may be omitted if cache is provided)
        cache1 = PLP.poly_in_box_cache(pi; box=box, closure=true)

        perim_cache = TO.RegionGeometry.region_perimeter(pi, 1; cache=cache1)
        @test isapprox(perim_cache, perim; atol=1e-12, rtol=0.0)

        # Define the non-cached centroid for comparison.
        centroid = TO.RegionGeometry.region_centroid(pi, 1; box=box)

        c_cache = TO.RegionGeometry.region_centroid(pi, 1; cache=cache1)
        @test isapprox(c_cache[1], centroid[1]; atol=1e-10)
        @test isapprox(c_cache[2], centroid[2]; atol=1e-10)

        # (3) boundary-measure breakdown: sum of facet measures should match perimeter.
        bd = TO.RegionGeometry.region_boundary_measure_breakdown(pi, 1; cache=cache1)
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
        sigy3 = [BitVector(), BitVector()]
        sigz3 = [BitVector(), BitVector()]
        pi2 = PLP.PLEncodingMap(2, sigy3, sigz3, [hp_left, hp_right],
                                [(0.5, 0.5), (1.5, 0.5)])
        box2 = ([0.0, 0.0], [2.0, 1.0])
        cache2 = PLP.poly_in_box_cache(pi2; box=box2, closure=true)
        @test cache2.level == :light
        @test !cache2.activity_scanned
        @test isempty(cache2.active_regions)
        @test all(==(Int8(0)), cache2.activity_state)
        @test all(cache2.points_f[r] === nothing for r in cache2.active_regions)

        # Single-region geometry calls on tiny maps should use the no-auto-cache route.
        cache_skip = PLP._auto_geometry_cache(pi2, nothing, box2, true, :fast;
                                              level=:geometry, intent=:single_region_exact)
        @test cache_skip === nothing
        CM._clear_encoding_cache!(pi2.cache)
        bb_direct = TO.RegionGeometry.region_bbox(pi2, 1; box=box2)
        @test bb_direct == (Float64[0.0, 0.0], Float64[1.0, 1.0])
        @test isempty(pi2.cache.geometry)
        @test cache2.activity_state[2] == Int8(0)

        w2 = TO.RegionGeometry.region_weights(pi2; cache=cache2, method=:exact)
        @test length(w2) == 2
        @test isapprox(w2[1], 1.0; atol=1e-12, rtol=0.0)
        @test isapprox(w2[2], 1.0; atol=1e-12, rtol=0.0)
        @test length(cache2.active_regions) == 2
        @test count(cache2.exact_weight_ready) == 2
        w2_b = TO.RegionGeometry.region_weights(pi2; cache=cache2, method=:exact)
        @test w2_b == w2
        @test count(cache2.exact_weight_ready) == 2

        bd1 = TO.RegionGeometry.region_boundary_measure_breakdown(pi2, 1; cache=cache2)
        bd2 = TO.RegionGeometry.region_boundary_measure_breakdown(pi2, 2; cache=cache2)
        @test all(cache2.points_f[r] !== nothing for r in cache2.active_regions)
        p1 = TO.RegionGeometry.region_perimeter(pi2, 1; cache=cache2)
        @test isapprox(sum(e.measure for e in bd1), p1; atol=1e-8, rtol=0.0)
        @test isapprox(sum(e.measure for e in bd2), TO.RegionGeometry.region_perimeter(pi2, 2; cache=cache2); atol=1e-8, rtol=0.0)
        @test any(e.kind == :internal for e in bd1)
        # The only internal neighbor of region 1 in this setup is region 2; its shared edge has length 1.
        mint = sum(e.measure for e in bd1 if e.neighbor == 2)
        @test isapprox(mint, 1.0; atol=1e-8, rtol=0.0)

        # Final-product cache reuse (same key => same cached object instance).
        bd1_b = TO.RegionGeometry.region_boundary_measure_breakdown(pi2, 1; cache=cache2, strict=true, mode=:fast)
        @test bd1_b === bd1
        n_bd_before_adj = length(cache2.boundary_breakdown)
        adj_cache_a = TO.RegionGeometry.region_adjacency(pi2; cache=cache2, strict=true, mode=:fast)
        n_bd_after_adj = length(cache2.boundary_breakdown)
        adj_cache_b = TO.RegionGeometry.region_adjacency(pi2; cache=cache2, strict=true, mode=:fast)
        @test adj_cache_b === adj_cache_a
        @test n_bd_after_adj >= n_bd_before_adj
        @test length(cache2.boundary_breakdown) == n_bd_after_adj
        n_bm_before = length(cache2.boundary_measure)
        @test TO.RegionGeometry.region_perimeter(pi2, 1; cache=cache2) == p1
        @test length(cache2.boundary_measure) == n_bm_before

        # Auto-cache parity (no explicit cache argument).
        bd1_auto = TO.RegionGeometry.region_boundary_measure_breakdown(pi2, 1; box=box2, strict=true, mode=:fast)
        @test bd1_auto == bd1
        adj_auto = TO.RegionGeometry.region_adjacency(pi2; box=box2, strict=true, mode=:fast)
        @test adj_auto == adj_cache_a
        # Auto exact geometry cache on bare PLEncodingMap should be retained by
        # (pi, box, closure) and reused across :fast/:verified modes.
        CM._clear_encoding_cache!(pi2.cache)
        key_auto = PLP._canonical_box_key(pi2, box2, true)
        @test !haskey(pi2.cache.geometry, key_auto)
        _ = TO.RegionGeometry.region_adjacency(pi2; box=box2, strict=true, mode=:fast)
        @test haskey(pi2.cache.geometry, key_auto)
        auto_cache = pi2.cache.geometry[key_auto].value
        @test auto_cache isa PLP.PolyInBoxCache
        @test auto_cache.level == :full
        @test auto_cache.activity_scanned
        @test auto_cache.bucket_enabled
        @test count(auto_cache.exact_weight_ready) == length(auto_cache.active_regions)
        _ = TO.RegionGeometry.region_adjacency(pi2; box=box2, strict=true, mode=:verified)
        @test length(pi2.cache.geometry) == 1
        @test pi2.cache.geometry[key_auto].value === auto_cache
        @test TO.RegionGeometry.region_weights(pi2; box=box2, method=:exact, mode=:fast) == w2
        @test count(auto_cache.exact_weight_ready) == length(auto_cache.active_regions)

        # Compile-cache alias should behave like poly_in_box_cache and also accelerate
        # Monte Carlo membership probes by reusing precompiled float inequalities.
        cache_comp = PLP.compile_geometry_cache(pi2; box=box2, closure=true)
        @test cache_comp isa PLP.PolyInBoxCache
        @test cache_comp.level == :full
        @test length(cache_comp.Af) == 2
        @test length(cache_comp.bf_strict) == 2
        @test length(cache_comp.bf_relaxed) == 2
        @test count(cache_comp.exact_weight_ready) == length(cache_comp.active_regions)
        @test count(cache_comp.exact_centroid_ready) == length(cache_comp.active_regions)
        @test all(cache_comp.facets[r] !== nothing for r in cache_comp.active_regions)
        @test all(cache_comp.points_f[r] !== nothing for r in cache_comp.active_regions)

        cache_geom = PLP.compile_geometry_cache(pi2; box=box2, closure=true,
                                                precompute_exact=false, level=:geometry)
        @test cache_geom.level == :geometry
        @test cache_geom.activity_scanned
        @test cache_geom.bucket_enabled
        @test count(cache_geom.exact_weight_ready) == 0
        @test all(cache_geom.facets[r] === nothing for r in cache_geom.active_regions)

        # Near-boundary classify path should agree in fast/verified modes.
        for delta in (-1e-12, -1e-14, 0.0, 1e-14, 1e-12)
            q = [1.0 + delta, 0.5]
            @test EC.locate(cache_comp, q; mode=:fast) == EC.locate(cache_comp, q; mode=:verified)
        end

        rng_mc_a = MersenneTwister(77)
        rng_mc_b = MersenneTwister(77)
        w_mc_uncached = TO.RegionGeometry.region_weights(pi2; box=box2, method=:mc, nsamples=20_000, rng=rng_mc_a, strict=false)
        w_mc_cached = TO.RegionGeometry.region_weights(pi2; cache=cache_comp, method=:mc, nsamples=20_000, rng=rng_mc_b, strict=false)
        @test w_mc_cached == w_mc_uncached

        # Active-region pruning oracle for exact paths: only one region intersects the box.
        A1d = reshape(QQ[1, -1], 2, 1)
        hp_a = PLP.make_hpoly(A1d, QQ[1, 0])      # [0,1]
        hp_b = PLP.make_hpoly(A1d, QQ[11, -10])   # [10,11]
        hp_c = PLP.make_hpoly(A1d, QQ[21, -20])   # [20,21]
        pi_sparse = PLP.PLEncodingMap(
            1,
            [BitVector([false, false]), BitVector([true, false]), BitVector([false, true])],
            [falses(2), falses(2), falses(2)],
            [hp_a, hp_b, hp_c],
            [(0.5,), (10.5,), (20.5,)],
        )
        box_sparse = (Float64[0.0], Float64[1.0])
        cache_sparse = PLP.compile_geometry_cache(pi_sparse; box=box_sparse, closure=true)
        @test cache_sparse.active_regions == [1]
        @test count(cache_sparse.exact_weight_ready) == 1
        w_sparse = TO.RegionGeometry.region_weights(pi_sparse; cache=cache_sparse, method=:exact)
        @test isapprox(w_sparse[1], 1.0; atol=1e-12)
        @test isapprox(w_sparse[2], 0.0; atol=1e-12)
        @test isapprox(w_sparse[3], 0.0; atol=1e-12)
        @test count(cache_sparse.exact_weight_ready) == 1
        @test TO.RegionGeometry.region_weights(pi_sparse; cache=cache_sparse, method=:exact) == w_sparse
        @test count(cache_sparse.exact_weight_ready) == 1
        @test TO.RegionGeometry.region_bbox(pi_sparse, 2; cache=cache_sparse) === nothing
        @test TO.RegionGeometry.region_bbox(pi_sparse, 3; cache=cache_sparse) === nothing
        @test TO.RegionGeometry.region_bbox(pi_sparse, 1; cache=cache_sparse) == (Float64[0.0], Float64[1.0])
    end
end
end

if field isa CM.QQField
@testset "Extended region geometry descriptors (axis backend)" begin
        # Build a simple 2D encoding map: split the plane into quadrants by x=0 and y=0.
        # Each quadrant is one region. Intersect with box [-1,1]^2 to get unit squares.
        Ups = PLB.BoxUpset[
            PLB.BoxUpset([0.0, -10.0]),   # x >= 0
            PLB.BoxUpset([-10.0, 0.0]),   # y >= 0
        ]
        Downs = PLB.BoxDownset[]
        Phi = zeros(K, 0, length(Ups))

        _, _, pi = PLB.encode_fringe_boxes(Ups, Downs, Phi)

        box = ([-1.0, -1.0], [1.0, 1.0])

        w = TO.RegionGeometry.region_weights(pi; box=box)
        @test length(w) == 4
        @test all(abs.(w .- 1.0) .< 1e-12)

        r = 1  # lower-left: [-1,0] x [-1,0]

        cb = TO.RegionGeometry.region_chebyshev_ball(pi, r; box=box)
        @test isapprox(cb.radius, 0.5; atol=1e-12)
        @test all(isapprox.(cb.center, [-0.5, -0.5]; atol=1e-12))

        @test isapprox(TO.RegionGeometry.region_inradius(pi, r; box=box), 0.5; atol=1e-12)

        cr = TO.RegionGeometry.region_circumradius(pi, r; box=box, center=:chebyshev, metric=:L2, method=:cells)
        @test isapprox(cr, sqrt(0.5); atol=1e-12)

        # Planar isoperimetric quotient of a unit square: 4*pi*A/P^2 = pi/4.
        iso_planar = TO.RegionGeometry.region_isoperimetric_ratio(pi, r; box=box, kind=:planar)
        @test isapprox(iso_planar, Base.MathConstants.pi / 4; atol=1e-10)

        # Boundary-to-volume for unit square: perimeter/area = 4.
        b2v = TO.RegionGeometry.region_boundary_to_volume_ratio(pi, r; box=box)
        @test isapprox(b2v, 4.0; atol=1e-10)

        # Mean width:
        # - cauchy formula (convex planar) gives 4/pi exactly
        mw_cauchy = TO.RegionGeometry.region_mean_width(pi, r; box=box, method=:cauchy)
        @test isapprox(mw_cauchy, 4.0 / Base.MathConstants.pi; atol=1e-10)

        # - cell-based estimate should be close (direction sampling only)
        mw_cells = TO.RegionGeometry.region_mean_width(pi, r; box=box, method=:cells, ndirs=2000, rng=MersenneTwister(1))
        @test isapprox(mw_cells, 4.0 / Base.MathConstants.pi; atol=0.03)

        mf = TO.RegionGeometry.region_minkowski_functionals(pi, r; box=box, mean_width_method=:cauchy)
        @test isapprox(mf.volume, 1.0; atol=1e-12)
        @test isapprox(mf.boundary_measure, 4.0; atol=1e-12)
        @test isapprox(mf.mean_width, 4.0 / Base.MathConstants.pi; atol=1e-10)
end
end

if field isa CM.QQField
@testset "RegionGeometry direct volume hook parity" begin
    Ups = [PLB.BoxUpset([0.0, 0.0])]
    Downs = [PLB.BoxDownset([2.0, 1.0])]
    Phi = reshape(K[c(1)], 1, 1)
    _, _, pi = PLB.encode_fringe_boxes(Ups, Downs, Phi)
    box = ([0.0, 0.0], [2.0, 1.0])
    r = EC.locate(pi, [0.5, 0.5])
    @test r != 0

    w = TO.RegionGeometry.region_weights(pi; box=box)
    @test TO.RegionGeometry.region_volume(pi, r; box=box) == w[r]

    direct_prev = TO.RegionGeometry._REGION_DIRECT_VOLUME[]
    fast_prev = TO.RegionGeometry._REGION_FAST_WRAPPERS[]
    try
        TO.RegionGeometry._REGION_DIRECT_VOLUME[] = false
        TO.RegionGeometry._REGION_FAST_WRAPPERS[] = false
        ratio_slow = TO.RegionGeometry.region_boundary_to_volume_ratio(pi, r; box=box)
        mf_slow = TO.RegionGeometry.region_minkowski_functionals(pi, r;
            box=box, mean_width_method=:cauchy)
        TO.RegionGeometry._REGION_DIRECT_VOLUME[] = true
        TO.RegionGeometry._REGION_FAST_WRAPPERS[] = true
        ratio_fast = TO.RegionGeometry.region_boundary_to_volume_ratio(pi, r; box=box)
        mf_fast = TO.RegionGeometry.region_minkowski_functionals(pi, r;
            box=box, mean_width_method=:cauchy)
        @test ratio_fast == ratio_slow
        @test mf_fast == mf_slow
    finally
        TO.RegionGeometry._REGION_DIRECT_VOLUME[] = direct_prev
        TO.RegionGeometry._REGION_FAST_WRAPPERS[] = fast_prev
    end

    if PLP.HAVE_POLY
        A = K[ 1 0;
               0 1;
              -1 0;
               0 -1 ]
        b = K[1, 1, 0, 0]
        hp = PLP.make_hpoly(A, b)
        pi_poly = PLP.PLEncodingMap(2, [BitVector()], [BitVector()], [hp], [(0.5, 0.5)])
        cache = PLP.poly_in_box_cache(pi_poly; box=box, closure=true)
        w_poly = TO.RegionGeometry.region_weights(pi_poly; box=box, cache=cache)
        @test TO.RegionGeometry.region_volume(pi_poly, 1; box=box, cache=cache) == w_poly[1]
    end
end
end

if field isa CM.QQField
@testset "RegionGeometry generic batched locate parity" begin
    P = FF.ProductOfChainsPoset((2, 2))
    pi = EC.GridEncodingMap(P, ([0.0, 1.0], [0.0, 2.0]))
    box = ([0.0, 0.0], [2.0, 3.0])
    r = EC.locate(pi, [0.5, 1.0])
    @test r == 1

    X = [0.25 0.75 1.25 0.25 0.75 1.25;
         0.50 1.50 2.50 2.50 0.50 1.50]
    dest = fill(0, size(X, 2))
    EC.locate_many!(dest, pi, X)
    @test dest == [EC.locate(pi, X[:, j]) for j in 1:size(X, 2)]

    batched_prev = TO.RegionGeometry._REGION_BATCHED_LOCATE[]
    thresh_prev = TO.RegionGeometry._REGION_BATCHED_LOCATE_MIN_PROPOSALS[]
    TO.RegionGeometry._REGION_BATCHED_LOCATE_MIN_PROPOSALS[] = 1
    try
        rng_scalar = MersenneTwister(11)
        rng_batch = MersenneTwister(11)
        TO.RegionGeometry._REGION_BATCHED_LOCATE[] = false
        pd_scalar = TO.RegionGeometry.region_principal_directions(pi, r;
            box=box, nsamples=256, max_proposals=1024, rng=rng_scalar, strict=true,
            return_info=true, nbatches=4)
        TO.RegionGeometry._REGION_BATCHED_LOCATE[] = true
        pd_batch = TO.RegionGeometry.region_principal_directions(pi, r;
            box=box, nsamples=256, max_proposals=1024, rng=rng_batch, strict=true,
            return_info=true, nbatches=4)
        @test pd_batch.n_accepted == pd_scalar.n_accepted
        @test pd_batch.n_proposed >= pd_scalar.n_proposed
        @test pd_batch.n_proposed <= 1024
        @test pd_batch.mean == pd_scalar.mean
        @test pd_batch.cov == pd_scalar.cov
        @test pd_batch.evals == pd_scalar.evals
        @test isapprox.(abs.(pd_batch.evecs), abs.(pd_scalar.evecs); atol=1e-12, rtol=0.0) |> all
        @test pd_batch.batch_evals == pd_scalar.batch_evals
        @test pd_batch.batch_n_accepted == pd_scalar.batch_n_accepted

        dirs = TO.RegionGeometry._random_unit_directions(2, 48; rng=MersenneTwister(17))
        rng_scalar2 = MersenneTwister(19)
        rng_batch2 = MersenneTwister(19)
        TO.RegionGeometry._REGION_BATCHED_LOCATE[] = false
        mw_scalar = TO.RegionGeometry.region_mean_width(pi, r;
            box=box, method=:mc, ndirs=size(dirs, 2), directions=dirs,
            nsamples=256, max_proposals=1024, rng=rng_scalar2, strict=true)
        TO.RegionGeometry._REGION_BATCHED_LOCATE[] = true
        mw_batch = TO.RegionGeometry.region_mean_width(pi, r;
            box=box, method=:mc, ndirs=size(dirs, 2), directions=dirs,
            nsamples=256, max_proposals=1024, rng=rng_batch2, strict=true)
        @test mw_batch == mw_scalar
    finally
        TO.RegionGeometry._REGION_BATCHED_LOCATE[] = batched_prev
        TO.RegionGeometry._REGION_BATCHED_LOCATE_MIN_PROPOSALS[] = thresh_prev
    end
end
end

if field isa CM.QQField
@testset "RegionGeometry sampled summary cache and workspace parity" begin
    P = FF.ProductOfChainsPoset((2, 2))
    pi = EC.GridEncodingMap(P, ([0.0, 1.0], [0.0, 2.0]))
    pi_cached = EC.compile_encoding(P, pi; meta=CM.EncodingCache())
    box = ([0.0, 0.0], [2.0, 3.0])
    r = EC.locate(pi, [0.5, 1.0])
    dirs = TO.RegionGeometry._random_unit_directions(2, 48; rng=MersenneTwister(17))

    prev_batched = TO.RegionGeometry._REGION_BATCHED_LOCATE[]
    prev_thresh = TO.RegionGeometry._REGION_BATCHED_LOCATE_MIN_PROPOSALS[]
    prev_work = TO.RegionGeometry._REGION_WORKSPACE_REUSE[]
    prev_cache = TO.RegionGeometry._REGION_SAMPLED_SUMMARY_CACHE[]
    prev_style = EC._ENCODINGCORE_CACHED_LOCATE_STYLE[]
    try
        TO.RegionGeometry._REGION_BATCHED_LOCATE[] = true
        TO.RegionGeometry._REGION_BATCHED_LOCATE_MIN_PROPOSALS[] = 1

        TO.RegionGeometry._REGION_WORKSPACE_REUSE[] = false
        TO.RegionGeometry._REGION_SAMPLED_SUMMARY_CACHE[] = false
        EC._ENCODINGCORE_CACHED_LOCATE_STYLE[] = false
        TO.RegionGeometry._clear_region_geometry_runtime_caches!()
        EC._clear_locate_style_cache!()
        pd_slow = TO.RegionGeometry.region_principal_directions(pi, r;
            box=box, nsamples=256, max_proposals=1024, rng=MersenneTwister(31),
            strict=true, return_info=true, nbatches=4)
        mw_slow = TO.RegionGeometry.region_mean_width(pi, r;
            box=box, method=:mc, ndirs=size(dirs, 2), directions=dirs,
            nsamples=256, max_proposals=1024, rng=MersenneTwister(37), strict=true)

        TO.RegionGeometry._REGION_WORKSPACE_REUSE[] = true
        TO.RegionGeometry._REGION_SAMPLED_SUMMARY_CACHE[] = true
        EC._ENCODINGCORE_CACHED_LOCATE_STYLE[] = true
        TO.RegionGeometry._clear_region_geometry_runtime_caches!()
        EC._clear_locate_style_cache!()
        pd_fast = TO.RegionGeometry.region_principal_directions(pi, r;
            box=box, nsamples=256, max_proposals=1024, rng=MersenneTwister(31),
            strict=true, return_info=true, nbatches=4)
        mw_fast = TO.RegionGeometry.region_mean_width(pi, r;
            box=box, method=:mc, ndirs=size(dirs, 2), directions=dirs,
            nsamples=256, max_proposals=1024, rng=MersenneTwister(37), strict=true)

        @test isapprox(pd_fast.mean, pd_slow.mean; atol=1e-12, rtol=0.0)
        @test isapprox(pd_fast.cov, pd_slow.cov; atol=1e-12, rtol=0.0)
        @test isapprox(pd_fast.evals, pd_slow.evals; atol=1e-12, rtol=0.0)
        @test isapprox.(abs.(pd_fast.evecs), abs.(pd_slow.evecs); atol=1e-12, rtol=0.0) |> all
        @test all(isapprox.(pd_fast.batch_evals, pd_slow.batch_evals; atol=1e-12, rtol=0.0))
        @test pd_fast.batch_n_accepted == pd_slow.batch_n_accepted
        @test mw_fast == mw_slow

        TO.RegionGeometry._clear_region_geometry_runtime_caches!()
        empty!(pi_cached.meta.geometry)
        tid = min(Base.Threads.threadid(), length(TO.RegionGeometry._REGION_WORKSPACES))
        _ = TO.RegionGeometry.region_mean_width(pi_cached, r;
            box=box, method=:mc, ndirs=size(dirs, 2), directions=dirs,
            nsamples=128, max_proposals=512, rng=MersenneTwister(41), strict=true)
        nwork1 = length(TO.RegionGeometry._REGION_WORKSPACES[tid])
        ngeom1 = length(pi_cached.meta.geometry)
        _ = TO.RegionGeometry.region_mean_width(pi_cached, r;
            box=box, method=:mc, ndirs=size(dirs, 2), directions=dirs,
            nsamples=128, max_proposals=512, rng=MersenneTwister(41), strict=true)
        nwork2 = length(TO.RegionGeometry._REGION_WORKSPACES[tid])
        ngeom2 = length(pi_cached.meta.geometry)
        @test nwork1 == nwork2
        @test nwork1 >= 1
        @test ngeom2 == ngeom1
        @test ngeom2 >= 1
        nstyles1 = length(EC._LOCATE_STYLE_CACHE)
        _ = TO.RegionGeometry.region_mean_width(pi_cached, r;
            box=box, method=:mc, ndirs=size(dirs, 2), directions=dirs,
            nsamples=128, max_proposals=512, rng=MersenneTwister(41), strict=true)
        @test length(EC._LOCATE_STYLE_CACHE) == nstyles1

        TO.RegionGeometry._clear_region_geometry_runtime_caches!()
        empty!(pi_cached.meta.geometry)
        ani = TO.RegionGeometry.region_covariance_anisotropy(pi_cached, r;
            box=box, nsamples=256, max_proposals=1024, rng=MersenneTwister(43), strict=true)
        @test isfinite(ani)
        @test length(TO.RegionGeometry._REGION_PRINCIPAL_SUMMARY_CACHE) == 0
        @test length(pi_cached.meta.geometry) == 1
        ecc = TO.RegionGeometry.region_covariance_eccentricity(pi_cached, r;
            box=box, nsamples=256, max_proposals=1024, rng=MersenneTwister(43), strict=true)
        @test isfinite(ecc)
        @test length(TO.RegionGeometry._REGION_PRINCIPAL_SUMMARY_CACHE) == 0
        @test length(pi_cached.meta.geometry) == 1
        _ = TO.RegionGeometry.region_anisotropy_scores(pi_cached, r;
            box=box, nsamples=256, max_proposals=1024, rng=MersenneTwister(43), strict=true)
        @test length(TO.RegionGeometry._REGION_PRINCIPAL_SUMMARY_CACHE) == 0
        @test length(pi_cached.meta.geometry) == 1

        TO.RegionGeometry._clear_region_geometry_runtime_caches!()
        empty!(pi_cached.meta.geometry)
        mw1 = TO.RegionGeometry.region_mean_width(pi_cached, r;
            box=box, method=:mc, ndirs=size(dirs, 2),
            nsamples=256, max_proposals=1024, rng=MersenneTwister(47), strict=true)
        ncache1 = length(pi_cached.meta.geometry)
        @test length(TO.RegionGeometry._REGION_MEAN_WIDTH_CACHE) == 0
        mw2 = TO.RegionGeometry.region_mean_width(pi_cached, r;
            box=box, method=:mc, ndirs=size(dirs, 2),
            nsamples=256, max_proposals=1024, rng=MersenneTwister(47), strict=true)
        @test mw2 == mw1
        @test length(TO.RegionGeometry._REGION_MEAN_WIDTH_CACHE) == 0
        @test length(pi_cached.meta.geometry) == ncache1
    finally
        TO.RegionGeometry._REGION_BATCHED_LOCATE[] = prev_batched
        TO.RegionGeometry._REGION_BATCHED_LOCATE_MIN_PROPOSALS[] = prev_thresh
        TO.RegionGeometry._REGION_WORKSPACE_REUSE[] = prev_work
        TO.RegionGeometry._REGION_SAMPLED_SUMMARY_CACHE[] = prev_cache
        EC._ENCODINGCORE_CACHED_LOCATE_STYLE[] = prev_style
        TO.RegionGeometry._clear_region_geometry_runtime_caches!()
        EC._clear_locate_style_cache!()
    end
end
end

if field isa CM.QQField
@testset "RegionGeometry fast wrapper parity" begin
    if !PLP.HAVE_POLY
        @test true
    else
        A = K[ 1 0;
               0 1;
              -1 0;
               0 -1 ]
        b = K[1, 1, 0, 0]
        hp = PLP.make_hpoly(A, b)
        pi = PLP.PLEncodingMap(2, [BitVector()], [BitVector()], [hp], [(0.5, 0.5)])
        box = (Float64[0.0, 0.0], Float64[1.0, 1.0])
        cache = PLP.poly_in_box_cache(pi; box=box, closure=true)

        fast_prev = TO.RegionGeometry._REGION_FAST_WRAPPERS[]
        try
            TO.RegionGeometry._REGION_FAST_WRAPPERS[] = false
            @test !cache.exact_centroid_ready[1]
            perim_slow = TO.RegionGeometry.region_perimeter(pi, 1; cache=cache)
            ratio_slow = TO.RegionGeometry.region_boundary_to_volume_ratio(pi, 1; box=box, cache=cache)
            mf_slow = TO.RegionGeometry.region_minkowski_functionals(pi, 1;
                box=box, cache=cache, mean_width_method=:cauchy)
            @test !cache.exact_centroid_ready[1]

            TO.RegionGeometry._REGION_FAST_WRAPPERS[] = true
            perim_fast = TO.RegionGeometry.region_perimeter(pi, 1; cache=cache)
            ratio_fast = TO.RegionGeometry.region_boundary_to_volume_ratio(pi, 1; box=box, cache=cache)
            mf_fast = TO.RegionGeometry.region_minkowski_functionals(pi, 1;
                box=box, cache=cache, mean_width_method=:cauchy)

            @test perim_fast == perim_slow
            @test ratio_fast == ratio_slow
            @test mf_fast == mf_slow
            @test !cache.exact_centroid_ready[1]
        finally
            TO.RegionGeometry._REGION_FAST_WRAPPERS[] = fast_prev
        end
    end
end
end

if field isa CM.QQField
@testset "RegionGeometry fast hook contracts" begin
    box = ([0.0, 0.0], [1.0, 1.0])
    stub = _REGION_HOOK_STUB
    prev_fast = TO.RegionGeometry._REGION_FAST_WRAPPERS[]
    prev_direct = TO.RegionGeometry._REGION_DIRECT_VOLUME[]
    try
        TO.RegionGeometry._REGION_FAST_WRAPPERS[] = true
        TO.RegionGeometry._REGION_DIRECT_VOLUME[] = true
        @test TO.RegionGeometry.region_volume(stub, 1; box=box) == 3.5
        @test TO.RegionGeometry.region_centroid(stub, 1; box=box) == [0.25, 0.75]
        @test TO.RegionGeometry.region_circumradius(stub, 1; box=box) == 4.25
        @test TO.RegionGeometry.region_minkowski_functionals(stub, 1; box=box) ==
              (volume=3.5, boundary_measure=7.0, mean_width=1.25)

        TO.RegionGeometry._REGION_FAST_WRAPPERS[] = false
        TO.RegionGeometry._REGION_DIRECT_VOLUME[] = false
        @test_throws ErrorException TO.RegionGeometry.region_volume(stub, 1; box=box)
        @test_throws ErrorException TO.RegionGeometry.region_centroid(stub, 1; box=box)
        @test_throws ErrorException TO.RegionGeometry.region_circumradius(stub, 1; box=box)
        @test_throws ErrorException TO.RegionGeometry.region_minkowski_functionals(stub, 1; box=box)
    finally
        TO.RegionGeometry._REGION_FAST_WRAPPERS[] = prev_fast
        TO.RegionGeometry._REGION_DIRECT_VOLUME[] = prev_direct
    end
end
end

if field isa CM.QQField
@testset "RegionGeometry UX helpers and friendly errors" begin
    unsupported = Val(:region_geometry_unsupported_backend)
    report_bad = TO.RegionGeometry.check_region_geometry(unsupported)
    @test report_bad.kind == :region_geometry
    @test report_bad.valid == false
    @test !isempty(report_bad.issues)
    txt_bad = sprint(show, MIME"text/plain"(), TO.RegionGeometry.region_geometry_validation_summary(report_bad))
    @test occursin("RegionGeometryValidationSummary", txt_bad)
    @test occursin("supported hooks", txt_bad)

    err_weights = try
        TO.RegionGeometry.region_weights(unsupported)
        nothing
    catch err
        sprint(showerror, err)
    end
    @test err_weights !== nothing
    @test occursin("check_region_geometry(pi)", err_weights)

    err_boundary = try
        TO.RegionGeometry.region_boundary_measure(unsupported, 1; box=([0.0], [1.0]))
        nothing
    catch err
        sprint(showerror, err)
    end
    @test err_boundary !== nothing
    @test occursin("region_boundary_measure", err_boundary)
    @test occursin("check_region_geometry(pi)", err_boundary)

    Ups = [PLB.BoxUpset([0.0])]
    Downs = [PLB.BoxDownset([2.0])]
    Phi = reshape(K[c(1)], 1, 1)
    _, _, pi = PLB.encode_fringe_boxes(Ups, Downs, Phi)
    box = ([-1.0], [3.0])
    rid = PLB.locate(pi, [1.0])

    report = TO.RegionGeometry.check_region_geometry(pi; box=box)
    @test report.valid
    @test report.hooks.weights
    @test report.hooks.bbox
    @test report.queries.region_volume
    @test report.queries.region_circumradius
    @test :region_volume in TO.RegionGeometry.supported_region_queries(pi)
    @test :weights in TO.RegionGeometry.supported_region_hooks(pi)
    @test TO.RegionGeometry.supports_region_query(pi, :region_bbox)
    @test TO.RegionGeometry.supports_region_query(pi, :region_mean_width)
    @test TO.RegionGeometry.supports_region_query(pi, :region_chebyshev_ball)
    @test :region_bbox in TO.RegionGeometry.recommended_region_queries(pi)

    query_ok = TO.RegionGeometry.check_region_query(pi, rid; box=box, query=:region_bbox)
    @test query_ok.valid
    query_bad = TO.RegionGeometry.check_region_query(pi, 99; query=:region_bbox)
    @test !query_bad.valid
    @test !isempty(query_bad.issues)

    rg = TO.RegionGeometry.region_geometry_summary(pi, rid; box=box)
    @test rg isa TO.RegionGeometry.RegionGeometrySummary
    @test TO.RegionGeometry.region_volume(rg) !== nothing
    @test TO.RegionGeometry.region_bbox(rg) !== nothing
    @test rg.report.widths == [2.0]
    @test rg.report.centroid == [1.0]
    @test isapprox(rg.report.aspect_ratio, 1.0; atol=1e-12)
    @test TO.RegionGeometry.finite_box_used(rg)
    @test :volume in TO.RegionGeometry.available_quantities(rg)
    @test !(:boundary_breakdown in TO.RegionGeometry.available_quantities(rg))
    @test rg.report.boundary_breakdown === nothing
    @test TO.RegionGeometry.quantity_sources(rg).bbox == :bbox_based
    @test TO.describe(rg).kind == :region_geometry_summary
    txt_rg = sprint(show, MIME"text/plain"(), rg)
    @test occursin("RegionGeometrySummary", txt_rg)

    Ups2 = [PLB.BoxUpset([0.0, 0.0])]
    Downs2 = [PLB.BoxDownset([1.0, 1.0])]
    Phi2 = reshape(K[c(1)], 1, 1)
    _, _, pi2 = PLB.encode_fringe_boxes(Ups2, Downs2, Phi2)
    box2 = ([0.0, 0.0], [1.0, 1.0])
    rid2 = PLB.locate(pi2, [0.5, 0.5])
    ps = TO.RegionGeometry.region_principal_directions_summary(pi2, rid2; box=box2, nsamples=800, rng=MersenneTwister(7), nbatches=4)
    @test length(TO.RegionGeometry.principal_values(ps)) == 2
    @test size(TO.RegionGeometry.principal_vectors(ps)) == (2, 2)
    @test size(TO.RegionGeometry.covariance_matrix(ps)) == (2, 2)
    anis = TO.RegionGeometry.region_anisotropy_scores(ps)
    @test haskey(anis, :ratio)
    txt_ps = sprint(show, MIME"text/plain"(), ps)
    @test occursin("RegionPrincipalDirectionsSummary", txt_ps)
end
end

if field isa CM.QQField
@testset "Module-level geometry distributions and adjacency graph stats" begin
        # Deterministic 1D encoding map with three regions:
        # thresholds at 0 and 2; intersect with box [-2,7] gives lengths [2,2,5].
        Ups = PLB.BoxUpset[
            PLB.BoxUpset([0.0]),
            PLB.BoxUpset([2.0]),
        ]
        Downs = PLB.BoxDownset[]

        Phi = zeros(K, 0, length(Ups))
        _, _, pi = PLB.encode_fringe_boxes(Ups, Downs, Phi)

        box = ([-2.0], [7.0])
        weights = TO.RegionGeometry.region_weights(pi; box=box)
        @test isapprox.(weights, [2.0, 2.0, 5.0]; atol=1e-12) |> all

        adj = TO.RegionGeometry.region_adjacency(pi; box=box)
        @test length(adj) == 2
        @test isapprox(sum(values(adj)), 2.0; atol=1e-12)

        # "Module" dimensions per region (using the new restricted_hilbert overload).
        dims = [0, 1, 0]

        opts = TO.InvariantOptions(box=box)
        vols = Inv.region_volume_samples_by_dim(dims, pi, opts; weights=weights)
        @test sort(vols[0]) == sort([2.0, 5.0])
        @test vols[1] == [2.0]

        b2v = Inv.region_boundary_to_volume_samples_by_dim(dims, pi, opts; weights=weights)
        @test sort(b2v[0]) == sort([1.0, 0.4])
        @test b2v[1] == [1.0]

        gs = Inv.region_adjacency_graph_stats(dims, pi, opts; adjacency=adj)
        @test gs.nregions == 3
        @test gs.nedges == 2
        @test isapprox(sum(values(adj)), 2.0; atol=1e-12)
        @test sort(gs.degrees.degrees) == [1, 1, 2]
        @test isapprox(gs.modularity, -0.5; atol=1e-12)
        @test gs.ncomponents == 1
        @test gs.component_sizes == [3]
end
end

if field isa CM.QQField
@testset "Covariance-based anisotropy and eccentricity (axis backend exact moments)" begin
    # Single-region behavior inside the box (bounds chosen to contain the box).
    Ups = [PLB.BoxUpset([-10.0, -10.0])]
    Downs = [PLB.BoxDownset([10.0, 10.0])]
    Phi = reshape(K[c(1)], 1, 1)
    _, _, pi = PLB.encode_fringe_boxes(Ups, Downs, Phi)

    # Intersect with a rectangle [-2,2] x [-0.5,0.5].
    box = ([-2.0, -0.5], [2.0, 0.5])
    r = EC.locate(pi, [0.0, 0.0])
    @test r != 0

    pd = TO.RegionGeometry.region_principal_directions(pi, r; box=box)
    # Variances: (width^2)/12.
    @test isapprox(pd.evals[1], 16.0/12.0; atol=1e-12)
    @test isapprox(pd.evals[2], 1.0/12.0; atol=1e-12)

    ani = TO.RegionGeometry.region_covariance_anisotropy(pi, r; box=box)
    @test isapprox(ani, 16.0; atol=1e-12)

    ecc = TO.RegionGeometry.region_covariance_eccentricity(pi, r; box=box)
    @test isapprox(ecc, sqrt(15.0/16.0); atol=1e-12)

    scores = TO.RegionGeometry.region_anisotropy_scores(pi, r; box=box)
    @test isapprox(scores.ratio, 16.0; atol=1e-12)
    @test isapprox(scores.eccentricity, sqrt(15.0/16.0); atol=1e-12)
end
end

if field isa CM.QQField
@testset "window_box and box=:auto" begin
    Ups = [PLB.BoxUpset([0.0])]
    Downs = [PLB.BoxDownset([2.0])]
    Phi = reshape(K[c(1)], 1, 1)
    P, Hhat, pi = PLB.encode_fringe_boxes(Ups, Downs, Phi)

    wb = Inv.window_box(pi)
    @test isapprox(wb[1][1], -1.2; atol=1e-12)
    @test isapprox(wb[2][1],  3.2; atol=1e-12)

    # Choose dims supported only on the middle region
    dims = zeros(Int, P.n)
    dims[EC.locate(pi, [1.0])] = 1

    opts_auto = TO.InvariantOptions(box=:auto)
    opts_wb   = TO.InvariantOptions(box=wb)

    m_auto = Inv.integrated_hilbert_mass(dims, pi, opts_auto)
    m_exp  = Inv.integrated_hilbert_mass(dims, pi, opts_wb)
    @test isapprox(m_auto, m_exp; atol=1e-12)
end
end


@testset "ZnEncodingMap adjacency and asymptotics" begin
    # 2D flange depending only on coordinate 1, free in coordinate 2
    FZ = TO.FlangeZn
    tau = FZ.face(2, [false, true])
    flats = [FZ.IndFlat(tau, [0, 0])]
    injs  = [FZ.IndInj(tau, [1, 0])]
    Phi   = reshape(K[c(1)], 1, 1)
    FG = TO.Flange{K}(2, flats, injs, Phi; field=field)
    enc = TO.EncodingOptions(backend=:zn, max_regions=100)
    enc_fg = TO.encode(FG, enc)
    P, M, pi = enc_fg.P, enc_fg.M, enc_fg.pi

    a = [-10,-10]
    b = [ 10, 10]
    adj = TO.RegionGeometry.region_adjacency(pi; box=(a,b))
    @test !isempty(adj)
    @test all(k[1] < k[2] for k in keys(adj))  # canonical ordering

    # asymptotic growth: total measure in Z^2 should scale like R^2, interface like R^(2-1)=R
    dims = ones(Int, P.n)
    opts = TO.InvariantOptions(box=:auto, strict=true)
    A = Inv.module_geometry_asymptotics(dims, pi, opts;
        scales=[1,2,4,8],
        include_interface=true
    )

    @test isfinite(A.exponent_total_measure)
    @test abs(A.exponent_total_measure - 2.0) < 0.25

    @test isfinite(A.exponent_interface_measure)
    @test abs(A.exponent_interface_measure - 1.0) < 0.35
end

if field isa CM.QQField
@testset "PL asymptotics exponent check" begin
    # Quadrant split example from existing tests (2D PLBackend)
    Ups = [
        PLB.BoxUpset([0.0, -10.0]),
        PLB.BoxUpset([-10.0, 0.0]),
    ]
    Downs = PLB.BoxDownset[]
    Phi = zeros(K, 0, length(Ups))
    _, _, pi = PLB.encode_fringe_boxes(Ups, Downs, Phi)

    dims = ones(Int, 4)  # 4 quadrants
    opts = TO.InvariantOptions(box=:auto, strict=true)
    A = Inv.module_geometry_asymptotics(dims, pi, opts;
        scales=[1,2,4,8],
        include_interface=true
    )

    @test isfinite(A.exponent_total_measure)
    @test abs(A.exponent_total_measure - 2.0) < 0.25

    @test isfinite(A.exponent_integrated_hilbert_mass)
    @test abs(A.exponent_integrated_hilbert_mass - 2.0) < 0.25
end
end

if field isa CM.QQField
@testset "Global module size summaries and region adjacency" begin

    @testset "1D box backend: integrated mass, histograms, adjacency" begin
        # 1D example: one upset x >= 0 and one downset x <= 2.
        # Inside any window, the parameter line splits into three regions:
        #   (-inf,0), (0,2), (2,inf)
        Ups = [PLB.BoxUpset([0.0])]
        Downs = [PLB.BoxDownset([2.0])]
        Phi = ones(K, 1, 1)

        P, H, pi = PLB.encode_fringe_boxes(Ups, Downs, Phi)
        box = ([-2.0], [7.0])
        opts = TO.InvariantOptions(box=box)

        # Identify the three regions by sampling points.
        t_left  = EC.locate(pi, [-1.0])
        t_mid   = EC.locate(pi, [ 1.0])
        t_right = EC.locate(pi, [ 3.0])

        # Region weights (lengths) inside the window:
        # left: [-2,0] has length 2, mid: [0,2] has length 2, right: [2,7] has length 5.
        w = TO.RegionGeometry.region_weights(pi; box=box)
        @test w[t_left]  == 2.0
        @test w[t_mid]   == 2.0
        @test w[t_right] == 5.0

        # region_bbox, widths, centroid
        bb_mid = TO.RegionGeometry.region_bbox(pi, t_mid; box=box)
        @test bb_mid[1][1] == 0.0
        @test bb_mid[2][1] == 2.0
        @test TO.RegionGeometry.region_widths(pi, t_mid; box=box) == [2.0]
        @test TO.RegionGeometry.region_centroid(pi, t_mid; box=box) == [1.0]
        @test TO.RegionGeometry.region_aspect_ratio(pi, t_mid; box=box) == 1.0

        # Integrated Hilbert mass:
        # Here dim=1 only on the middle region of length 2, so the integral is 2.
        mass = Inv.integrated_hilbert_mass(H, pi, opts)
        @test mass == 2.0

        # Histogram of measure by module dimension.
        hist = Inv.measure_by_dimension(H, pi, opts)
        @test hist[0] == 7.0
        @test hist[1] == 2.0

        # Support measure (dim > 0)
        supp = Inv.support_measure(H, pi, opts)
        @test supp == 2.0

        # Basic weighted stats for dim
        stats = Inv.dim_stats(H, pi, opts)
        @test stats.total_measure == 9.0
        @test stats.integrated_mass == 2.0
        @test abs(stats.mean - (2.0 / 9.0)) < 1e-12
        @test abs(stats.var - (14.0 / 81.0)) < 1e-12

        # L^p norms of dim
        @test Inv.dim_norm(H, pi, opts; p=1) == 2.0
        @test abs(Inv.dim_norm(H, pi, opts; p=2) - sqrt(2.0)) < 1e-12
        @test Inv.dim_norm(H, pi, opts; p=Inf) == 1.0

        # Entropy of region weights: w = [2,2,5] up to permutation.
        p1 = 2.0 / 9.0
        p2 = 2.0 / 9.0
        p3 = 5.0 / 9.0
        expected_entropy = -(p1 * log(p1) + p2 * log(p2) + p3 * log(p3))
        @test abs(Inv.region_weight_entropy(pi, opts) - expected_entropy) < 1e-12

        # Aspect ratio stats in 1D: all regions have aspect ratio 1.
        ar = Inv.aspect_ratio_stats(pi, opts)
        @test ar.total_measure == 9.0
        @test ar.mean == 1.0
        @test ar.min == 1.0
        @test ar.max == 1.0

        # module_size_summary should be consistent.
        summary = Inv.module_size_summary(H, pi, opts)
        @test summary.total_measure == 9.0
        @test summary.integrated_hilbert_mass == 2.0
        @test summary.support_measure == 2.0
        @test summary.measure_by_dimension[0] == 7.0
        @test summary.measure_by_dimension[1] == 2.0

        # Adjacency graph: there are exactly two interior boundaries (at 0 and 2),
        # so we expect two edges each with 0-dim measure 1.
        edges = TO.RegionGeometry.region_adjacency(pi; box=box)
        @test length(edges) == 2
        @test edges[(min(t_left, t_mid), max(t_left, t_mid))] == 1.0
        @test edges[(min(t_mid, t_right), max(t_mid, t_right))] == 1.0

        total_iface = Inv.interface_measure(pi, opts)
        @test total_iface == 2.0

        by_pair = Inv.interface_measure_by_dim_pair(H, pi, opts)
        @test by_pair[(0, 1)] == 2.0

        changed = Inv.interface_measure_dim_changes(H, pi, opts)
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
        Phi = zeros(K, 0, length(Ups))

        P, H, pi = PLB.encode_fringe_boxes(Ups, Downs, Phi)
        box = ([-1.0, -1.0], [1.0, 1.0])

        # Regions (quadrants)
        t00 = EC.locate(pi, [-0.5, -0.5])
        t10 = EC.locate(pi, [ 0.5, -0.5])
        t01 = EC.locate(pi, [-0.5,  0.5])
        t11 = EC.locate(pi, [ 0.5,  0.5])

        # Each quadrant has bbox width [1,1], centroid at its midpoint.
        @test TO.RegionGeometry.region_widths(pi, t00; box=box) == [1.0, 1.0]
        @test TO.RegionGeometry.region_centroid(pi, t00; box=box) == [-0.5, -0.5]
        @test TO.RegionGeometry.region_aspect_ratio(pi, t00; box=box) == 1.0

        # Adjacency edges: four unit-length interfaces inside the box:
        # (left-bottom)-(right-bottom), (left-bottom)-(left-top),
        # (right-bottom)-(right-top), (left-top)-(right-top).
        edges = TO.RegionGeometry.region_adjacency(pi; box=box)
        @test length(edges) == 4

        expected_pairs = [(t00, t10), (t00, t01), (t10, t11), (t01, t11)]
        for (u, v) in expected_pairs
            key = (min(u, v), max(u, v))
            @test abs(edges[key] - 1.0) < 1e-12
        end
    end

    @testset "Module size summary - polyhedral backend smoke test" begin
        # Allow passing a plain vector of region dimensions as "the module".
        Ups = [PLB.BoxUpset([0.0])]
        Downs = [PLB.BoxDownset([2.0])]
        Phi = reshape(K[c(1)], 1, 1)
        P, H, pi = PLB.encode_fringe_boxes(Ups, Downs, Phi)

        dims = [3, 0, 0]
        weights = [1.0, 0.0, 0.0]
        opts = TO.InvariantOptions()
        summary = Inv.module_size_summary(dims, pi, opts; weights=weights)
        @test summary.integrated_hilbert_mass == 3.0
        @test summary.total_measure == 1.0
    end
end
end

if field isa CM.QQField
@testset "uncertainty + support geometry + ehrhart" begin
    # --- PLBackend 1D encoding as in region geometry tests ---
    Ups = [PLB.BoxUpset([0.0])]
    Downs = [PLB.BoxDownset([2.0])]
    Phi = reshape(K[c(1)], 1, 1)
    _, _, pi = PLB.encode_fringe_boxes(Ups, Downs, Phi)
    box = ([-2.0], [7.0])
    opts = TO.InvariantOptions(box=box)

    # 1) region_weights with return_info on exact backend
    info = TO.RegionGeometry.region_weights(pi; box=box, return_info=true)
    @test info.method == :exact
    @test length(info.weights) == 3
    @test isapprox(sum(info.weights), 9.0)
    @test all(info.stderr .== 0.0)
    @test all(c[1] == c[2] for c in info.ci)

    # 2) support geometry
    H1 = [0, 1, 0]
    sm = Inv.support_measure_stats(H1, pi, opts; min_dim=1)
    @test isapprox(sm.estimate, 2.0)
    @test sm.stderr == 0.0
    @test sm.ci[1] == sm.ci[2] == 2.0

    H2 = [1, 1, 0]
    comps = Inv.support_components(H2, pi, opts; min_dim=1)
    @test comps isa Inv.SupportComponentsSummary
    @test length(comps) == 1
    @test comps[1] == [1, 2]
    @test Inv.ncomponents(comps) == 1
    @test Inv.component_sizes(comps) == [2]
    @test Inv.components(comps) == [[1, 2]]
    @test Inv.largest_component(comps) == [1, 2]

    diams, overall = Inv.support_graph_diameter(H2, pi, opts; min_dim=1)
    @test overall == 1
    @test diams[1] == 1

    H3 = [1, 0, 1]
    comps3 = Inv.support_components(H3, pi, opts; min_dim=1)
    @test comps3 isa Inv.SupportComponentsSummary
    @test length(comps3) == 2
    @test comps3[1] in ([1], [3])
    @test comps3[2] in ([1], [3])

    bb = Inv.support_bbox(H3, pi, opts; min_dim=1)
    @test isapprox(bb[1][1], -2.0)
    @test isapprox(bb[2][1], 7.0)
    @test isapprox(Inv.support_geometric_diameter(H3, pi, opts; min_dim=1, metric=:Linf), 9.0)

    # --- PLPolyhedra MC region weights sanity check (2D halfspaces) ---
    A1 = [0//1  -1//1]
    b1 = [0//1]
    A2 = [0//1   1//1]
    b2 = [0//1]
    hp1 = PLP.make_hpoly(A1, b1)
    hp2 = PLP.make_hpoly(A2, b2)
    reps = [(0.0,  1.0), (0.0, -1.0)]
    sigy = [BitVector([false]), BitVector([true])]
    sigz = [BitVector([false]), BitVector([false])]
    pi2 = PLP.PLEncodingMap(2, sigy, sigz, [hp1, hp2], reps)
    box2 = ([-2.0, -2.0], [2.0, 2.0])
    rng = MersenneTwister(1)

    mcinfo = TO.RegionGeometry.region_weights(pi2; box=box2, method=:mc, nsamples=20000, rng=rng, return_info=true)
    @test mcinfo.method == :mc
    @test length(mcinfo.weights) == 2
    @test all(mcinfo.stderr .>= 0.0)
    @test all(ci[1] <= w <= ci[2] for (w,ci) in zip(mcinfo.weights, mcinfo.ci))
    @test isapprox(sum(mcinfo.weights), mcinfo.total_volume)

    # 3) anisotropy stability (ratio should be > 1 for half-box)
    rng2 = MersenneTwister(2)
    an = TO.RegionGeometry.region_anisotropy_scores(pi2, 1; box=box2, nsamples=10000, rng=rng2, return_info=true, nbatches=5)
    @test an.ratio > 1.0
    @test !isnan(an.ratio_stderr)

    # --- Ehrhart-like fit on ZnEncodingMap (period=1) ---
    FZ = TO.FlangeZn
    flats = [FZ.IndFlat(FZ.face(2, [true, true]), [0, 0]; id=:F)]  # tau = all free => no cuts, but sets n=2
    injs  = FZ.IndInj{2}[]
    Phi   = zeros(K, 0, 1)
    FG    = TO.Flange{K}(2, flats, injs, Phi; field=field)

    enc = TO.EncodingOptions(backend=:zn, max_regions=10)
    enc_fg = TO.encode(FG, enc)
    P, M, piZ = enc_fg.P, enc_fg.M, enc_fg.pi

    base_box = ([-2, -2], [2, 2])
    scales = [1, 2, 3, 4]

    dims = ones(Int, P.n)

    opts = TO.InvariantOptions(box=base_box)
    asym = Inv.module_geometry_asymptotics(dims, piZ, opts;
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
end
end # with_fields
