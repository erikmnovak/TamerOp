#!/usr/bin/env julia

using Random

try
    using TamerOp
catch
    include(joinpath(@__DIR__, "..", "src", "TamerOp.jl"))
    using .TamerOp
end

const TO = TamerOp.Advanced
const PLP = TO.PLPolyhedra
const RG = TO.RegionGeometry

function _parse_arg(args, key::String, default::Int)
    for a in args
        startswith(a, key * "=") || continue
        return parse(Int, split(a, "=", limit=2)[2])
    end
    return default
end

function _parse_out(args, default::String)
    for a in args
        startswith(a, "--out=") || continue
        return String(split(a, "=", limit=2)[2])
    end
    return default
end

function _bench(name::String, reps::Int, f::Function)
    GC.gc()
    f()  # warmup
    GC.gc()
    times_ms = Vector{Float64}(undef, reps)
    bytes = Vector{Int}(undef, reps)
    for i in 1:reps
        m = @timed f()
        times_ms[i] = m.time * 1e3
        bytes[i] = m.bytes
    end
    sort!(times_ms)
    sort!(bytes)
    return (probe=name, median_ms=times_ms[cld(reps, 2)], median_bytes=bytes[cld(reps, 2)])
end

function _with_bool_toggle(f::Function, ref::Base.RefValue{Bool}, value::Bool)
    old = ref[]
    ref[] = value
    try
        return f()
    finally
        ref[] = old
    end
end

function _packed_to_vectors(buckets)
    out = Vector{Vector{Int}}(undef, length(buckets.ptr) - 1)
    @inbounds for i in eachindex(out)
        lo = buckets.ptr[i]
        hi = buckets.ptr[i + 1] - 1
        out[i] = hi < lo ? Int[] : collect(@view buckets.idx[lo:hi])
    end
    return out
end

function _bench_prefetched_bucket_scan_packed!(dest::Vector{Int}, pi, X)
    @inbounds for col in 1:size(X, 2)
        bid = PLP._spatial_bucket_id_col(pi.prefilter, X, col)
        if bid == 0
            dest[col] = PLP._locate_hybrid_col(
                pi, X, col;
                cache=nothing,
                verify_safe=false,
                use_multiproj=false,
                tol=PLP.LOCATE_FLOAT_TOL,
                boundary_tol=PLP.LOCATE_BOUNDARY_TOL,
            )
        else
            cands = pi.prefilter.spatial.buckets[bid]
            dest[col] = isempty(cands) ? 0 : PLP._locate_hybrid_col_prefetched(
                pi, X, col, cands;
                verify_safe=false,
                use_multiproj=false,
                tol=PLP.LOCATE_FLOAT_TOL,
                boundary_tol=PLP.LOCATE_BOUNDARY_TOL,
            )
        end
    end
    return dest
end

function _bench_prefetched_bucket_scan_legacy!(dest::Vector{Int}, pi, X, legacy_buckets)
    @inbounds for col in 1:size(X, 2)
        bid = PLP._spatial_bucket_id_col(pi.prefilter, X, col)
        if bid == 0
            dest[col] = PLP._locate_hybrid_col(
                pi, X, col;
                cache=nothing,
                verify_safe=false,
                use_multiproj=false,
                tol=PLP.LOCATE_FLOAT_TOL,
                boundary_tol=PLP.LOCATE_BOUNDARY_TOL,
            )
        else
            cands = legacy_buckets[bid]
            dest[col] = isempty(cands) ? 0 : PLP._locate_hybrid_col_prefetched(
                pi, X, col, cands;
                verify_safe=false,
                use_multiproj=false,
                tol=PLP.LOCATE_FLOAT_TOL,
                boundary_tol=PLP.LOCATE_BOUNDARY_TOL,
            )
        end
    end
    return dest
end

function _make_poly_fixture(nregions::Int)
    A = [1.0 0.0; 0.0 1.0; -1.0 0.0; 0.0 -1.0]
    regs = Vector{PLP.HPoly}(undef, nregions)
    wits = Vector{Tuple{Float64,Float64}}(undef, nregions)
    sigy = [BitVector() for _ in 1:nregions]
    sigz = [BitVector() for _ in 1:nregions]
    @inbounds for k in 1:nregions
        xlo = float(k - 1)
        xhi = float(k)
        b = [xhi, 1.0, -xlo, 0.0]
        regs[k] = PLP.make_hpoly(A, b)
        wits[k] = ((xlo + xhi) / 2.0, 0.5)
    end
    pi = PLP.PLEncodingMap(2, sigy, sigz, regs, wits)
    box = ([-1.0, -0.5], [float(nregions + 1), 1.5])
    return pi, box
end

function _make_grid_fixture(side::Int)
    side >= 2 || error("_make_grid_fixture: side must be >= 2")
    nregions = side * side
    A = [1.0 0.0; 0.0 1.0; -1.0 0.0; 0.0 -1.0]
    regs = Vector{PLP.HPoly}(undef, nregions)
    wits = Vector{Tuple{Float64,Float64}}(undef, nregions)
    sigy = [BitVector() for _ in 1:nregions]
    sigz = [BitVector() for _ in 1:nregions]
    k = 1
    @inbounds for gy in 0:side-1, gx in 0:side-1
        xlo = float(gx)
        xhi = float(gx + 1)
        ylo = float(gy)
        yhi = float(gy + 1)
        regs[k] = PLP.make_hpoly(A, [xhi, yhi, -xlo, -ylo])
        wits[k] = ((xlo + xhi) / 2.0, (ylo + yhi) / 2.0)
        k += 1
    end
    return PLP.PLEncodingMap(2, sigy, sigz, regs, wits)
end

function _make_grid3_fixture(nx::Int, ny::Int, nz::Int)
    (nx >= 2 && ny >= 2 && nz >= 2) || error("_make_grid3_fixture: expected nx,ny,nz >= 2")
    nregions = nx * ny * nz
    A = [1.0 0.0 0.0;
         0.0 1.0 0.0;
         0.0 0.0 1.0;
        -1.0 0.0 0.0;
         0.0 -1.0 0.0;
         0.0 0.0 -1.0]
    regs = Vector{PLP.HPoly}(undef, nregions)
    wits = Vector{NTuple{3,Float64}}(undef, nregions)
    sigy = [BitVector() for _ in 1:nregions]
    sigz = [BitVector() for _ in 1:nregions]
    k = 1
    @inbounds for gz in 0:nz-1, gy in 0:ny-1, gx in 0:nx-1
        xlo = float(gx); xhi = float(gx + 1)
        ylo = float(gy); yhi = float(gy + 1)
        zlo = float(gz); zhi = float(gz + 1)
        regs[k] = PLP.make_hpoly(A, [xhi, yhi, zhi, -xlo, -ylo, -zlo])
        wits[k] = ((xlo + xhi) / 2.0, (ylo + yhi) / 2.0, (zlo + zhi) / 2.0)
        k += 1
    end
    return PLP.PLEncodingMap(3, sigy, sigz, regs, wits)
end

function run_microbench(; nregions::Int=64, nqueries::Int=12_000,
                          reps_locate::Int=9, reps_breakdown::Int=7,
                          reps_adj::Int=5, reps_membership::Int=13,
                          reps_facets::Int=5,
                          grid_side::Int=32, nqueries_grid::Int=50_000,
                          reps_grid::Int=5,
                          grid3_nx::Int=8, grid3_ny::Int=8, grid3_nz::Int=4,
                          nqueries_grid3::Int=40_000, reps_grid3::Int=5)
    PLP.HAVE_POLY || error("pl_polyhedra_cache_microbench: Polyhedra backend unavailable")

    pi, box = _make_poly_fixture(nregions)
    cache = PLP.compile_geometry_cache(pi; box=box, closure=true)
    rid = cache.active_regions[cld(length(cache.active_regions), 2)]

    rng = MersenneTwister(0xBEEF)
    X = Matrix{Float64}(undef, 2, nqueries)
    @inbounds for j in 1:size(X, 2)
        X[1, j] = rand(rng) * float(nregions + 2) - 1.0
        X[2, j] = rand(rng) * 2.0 - 0.5
    end
    loc_dest = Vector{Int}(undef, nqueries)

    rows = NamedTuple[]
    push!(rows, _bench("plpoly_cache_build_light", reps_locate, () -> begin
        PLP.poly_in_box_cache(pi; box=box, closure=true, level=:light)
    end))
    push!(rows, _bench("plpoly_compile_geometry_level_geometry", reps_breakdown, () -> begin
        PLP.compile_geometry_cache(pi; box=box, closure=true, precompute_exact=false, level=:geometry)
    end))
    push!(rows, _bench("plpoly_compile_geometry_level_full", reps_adj, () -> begin
        PLP.compile_geometry_cache(pi; box=box, closure=true, precompute_exact=true, level=:full)
    end))
    push!(rows, _bench("plpoly_locate_many_cached", reps_locate, () -> begin
        PLP.locate_many!(loc_dest, cache, X; threaded=true, mode=:fast)
    end))
    push!(rows, _bench("plpoly_boundary_breakdown_cachemiss", reps_breakdown, () -> begin
        empty!(cache.boundary_breakdown)
        empty!(cache.boundary_measure)
        RG.region_boundary_measure_breakdown(pi, rid; cache=cache, strict=false, mode=:fast)
    end))
    push!(rows, _bench("plpoly_region_adjacency_cachemiss", reps_adj, () -> begin
        empty!(cache.boundary_breakdown)
        empty!(cache.boundary_measure)
        empty!(cache.adjacency)
        RG.region_adjacency(pi; cache=cache, strict=false, mode=:fast)
    end))
    push!(rows, _bench("plpoly_membership_relaxed_uncached", reps_membership, () -> begin
        PLP._membership_mats(pi, nothing, true)
    end))
    push!(rows, _bench("plpoly_region_facets_rebuild", reps_facets, () -> begin
        fill!(cache.facets, nothing)
        for r in cache.active_regions
            PLP._region_facets(cache, r; tol=1e-12)
        end
    end))

    pi_grid = _make_grid_fixture(grid_side)
    rng_grid = MersenneTwister(0xACED)
    Xg = Matrix{Float64}(undef, 2, nqueries_grid)
    @inbounds for j in 1:size(Xg, 2)
        Xg[1, j] = rand(rng_grid) * float(grid_side) - 1e-3
        Xg[2, j] = rand(rng_grid) * float(grid_side) - 1e-3
    end
    dest_g = Vector{Int}(undef, nqueries_grid)
    legacy_grid_buckets = _packed_to_vectors(pi_grid.prefilter.spatial.buckets)
    push!(rows, _bench("plpoly_locate_many_grid_nocache_serial", reps_grid, () -> begin
        PLP.locate_many!(dest_g, pi_grid, Xg; threaded=false, mode=:fast)
    end))
    push!(rows, _bench("plpoly_locate_many_grid_nocache_threaded", reps_grid, () -> begin
        PLP.locate_many!(dest_g, pi_grid, Xg; threaded=true, mode=:fast)
    end))
    push!(rows, _bench("plpoly_locate_many_grid_group_off", reps_grid, () -> begin
        _with_bool_toggle(PLP._LOCATE_BUCKET_GROUPING, false) do
            PLP.locate_many!(dest_g, pi_grid, Xg; threaded=true, mode=:fast)
        end
    end))
    push!(rows, _bench("plpoly_locate_many_grid_group_on", reps_grid, () -> begin
        _with_bool_toggle(PLP._LOCATE_BUCKET_GROUPING, true) do
            PLP.locate_many!(dest_g, pi_grid, Xg; threaded=true, mode=:fast)
        end
    end))
    push!(rows, _bench("plpoly_locate_many_grid_group_auto", reps_grid, () -> begin
        PLP.locate_many!(dest_g, pi_grid, Xg; threaded=true, mode=:fast)
    end))
    push!(rows, _bench("plpoly_prefetched_bucket_scan_legacy", reps_grid, () -> begin
        _bench_prefetched_bucket_scan_legacy!(dest_g, pi_grid, Xg, legacy_grid_buckets)
    end))
    push!(rows, _bench("plpoly_prefetched_bucket_scan_packed", reps_grid, () -> begin
        _bench_prefetched_bucket_scan_packed!(dest_g, pi_grid, Xg)
    end))
    push!(rows, _bench("plpoly_prefetched_bucket_scan_exactcache_off", reps_grid, () -> begin
        _with_bool_toggle(() -> _bench_prefetched_bucket_scan_packed!(dest_g, pi_grid, Xg),
                          PLP._LOCATE_COL_QQ_CACHE, false)
    end))
    push!(rows, _bench("plpoly_prefetched_bucket_scan_exactcache_on", reps_grid, () -> begin
        _with_bool_toggle(() -> _bench_prefetched_bucket_scan_packed!(dest_g, pi_grid, Xg),
                          PLP._LOCATE_COL_QQ_CACHE, true)
    end))
    push!(rows, _bench("plpoly_prefetched_bucket_scan_rowdotcache_off", reps_grid, () -> begin
        _with_bool_toggle(() -> _with_bool_toggle(() -> _bench_prefetched_bucket_scan_packed!(dest_g, pi_grid, Xg),
                                                  PLP._LOCATE_COL_QQ_CACHE, true),
                          PLP._LOCATE_ROW_DOT_CACHE, false)
    end))
    push!(rows, _bench("plpoly_prefetched_bucket_scan_rowdotcache_on", reps_grid, () -> begin
        _with_bool_toggle(() -> _with_bool_toggle(() -> _bench_prefetched_bucket_scan_packed!(dest_g, pi_grid, Xg),
                                                  PLP._LOCATE_COL_QQ_CACHE, true),
                          PLP._LOCATE_ROW_DOT_CACHE, true)
    end))
    push!(rows, _bench("plpoly_locate_many_grid_group_on_exactcache_off", reps_grid, () -> begin
        _with_bool_toggle(() -> _with_bool_toggle(() -> PLP.locate_many!(dest_g, pi_grid, Xg; threaded=true, mode=:fast),
                                                  PLP._LOCATE_BUCKET_GROUPING, true),
                          PLP._LOCATE_COL_QQ_CACHE, false)
    end))
    push!(rows, _bench("plpoly_locate_many_grid_group_on_exactcache_on", reps_grid, () -> begin
        _with_bool_toggle(() -> _with_bool_toggle(() -> PLP.locate_many!(dest_g, pi_grid, Xg; threaded=true, mode=:fast),
                                                  PLP._LOCATE_BUCKET_GROUPING, true),
                          PLP._LOCATE_COL_QQ_CACHE, true)
    end))
    push!(rows, _bench("plpoly_locate_many_grid_group_on_rowdotcache_off", reps_grid, () -> begin
        _with_bool_toggle(() -> _with_bool_toggle(() -> _with_bool_toggle(() -> PLP.locate_many!(dest_g, pi_grid, Xg; threaded=true, mode=:fast),
                                                                          PLP._LOCATE_BUCKET_GROUPING, true),
                                                  PLP._LOCATE_COL_QQ_CACHE, true),
                          PLP._LOCATE_ROW_DOT_CACHE, false)
    end))
    push!(rows, _bench("plpoly_locate_many_grid_group_on_rowdotcache_on", reps_grid, () -> begin
        _with_bool_toggle(() -> _with_bool_toggle(() -> _with_bool_toggle(() -> PLP.locate_many!(dest_g, pi_grid, Xg; threaded=true, mode=:fast),
                                                                          PLP._LOCATE_BUCKET_GROUPING, true),
                                                  PLP._LOCATE_COL_QQ_CACHE, true),
                          PLP._LOCATE_ROW_DOT_CACHE, true)
    end))

    push!(rows, _bench("plpoly_boundary_breakdown_batch_off", reps_breakdown, () -> begin
        _with_bool_toggle(PLP._FACET_PROBE_BATCH, false) do
            empty!(cache.boundary_breakdown)
            empty!(cache.boundary_measure)
            RG.region_boundary_measure_breakdown(pi, rid; cache=cache, strict=false, mode=:fast)
        end
    end))
    push!(rows, _bench("plpoly_boundary_breakdown_batch_on", reps_breakdown, () -> begin
        _with_bool_toggle(PLP._FACET_PROBE_BATCH, true) do
            empty!(cache.boundary_breakdown)
            empty!(cache.boundary_measure)
            RG.region_boundary_measure_breakdown(pi, rid; cache=cache, strict=false, mode=:fast)
        end
    end))
    push!(rows, _bench("plpoly_region_adjacency_batch_off", reps_adj, () -> begin
        _with_bool_toggle(PLP._FACET_PROBE_BATCH, false) do
            empty!(cache.boundary_breakdown)
            empty!(cache.boundary_measure)
            empty!(cache.adjacency)
            RG.region_adjacency(pi; cache=cache, strict=false, mode=:fast)
        end
    end))
    push!(rows, _bench("plpoly_region_adjacency_batch_on", reps_adj, () -> begin
        _with_bool_toggle(PLP._FACET_PROBE_BATCH, true) do
            empty!(cache.boundary_breakdown)
            empty!(cache.boundary_measure)
            empty!(cache.adjacency)
            RG.region_adjacency(pi; cache=cache, strict=false, mode=:fast)
        end
    end))
    push!(rows, _bench("plpoly_boundary_breakdown_nocache_batch_off", reps_breakdown, () -> begin
        _with_bool_toggle(PLP._FACET_PROBE_BATCH, false) do
            RG.region_boundary_measure_breakdown(pi, rid; box=box, strict=false, mode=:fast)
        end
    end))
    push!(rows, _bench("plpoly_boundary_breakdown_nocache_batch_on", reps_breakdown, () -> begin
        _with_bool_toggle(PLP._FACET_PROBE_BATCH, true) do
            RG.region_boundary_measure_breakdown(pi, rid; box=box, strict=false, mode=:fast)
        end
    end))
    push!(rows, _bench("plpoly_region_adjacency_nocache_batch_off", reps_adj, () -> begin
        _with_bool_toggle(PLP._FACET_PROBE_BATCH, false) do
            RG.region_adjacency(pi; box=box, strict=false, mode=:fast)
        end
    end))
    push!(rows, _bench("plpoly_region_adjacency_nocache_batch_on", reps_adj, () -> begin
        _with_bool_toggle(PLP._FACET_PROBE_BATCH, true) do
            RG.region_adjacency(pi; box=box, strict=false, mode=:fast)
        end
    end))

    pi_small, box_small = _make_poly_fixture(32)
    rid_small = 16
    cache_small = PLP.poly_in_box_cache(pi_small; box=box_small, closure=true, level=:light)
    push!(rows, _bench("plpoly_single_bbox_small_auto", reps_breakdown, () -> begin
        RG.region_bbox(pi_small, rid_small; box=box_small, closure=true)
    end))
    push!(rows, _bench("plpoly_single_boundary_small_auto", reps_breakdown, () -> begin
        RG.region_boundary_measure(pi_small, rid_small; box=box_small, strict=false, mode=:fast)
    end))
    push!(rows, _bench("plpoly_single_bbox_small_cached", reps_breakdown, () -> begin
        RG.region_bbox(pi_small, rid_small; cache=cache_small, closure=true)
    end))

    pi_grid3 = _make_grid3_fixture(grid3_nx, grid3_ny, grid3_nz)
    rng_grid3 = MersenneTwister(0xACEE)
    X3 = Matrix{Float64}(undef, 3, nqueries_grid3)
    @inbounds for j in 1:size(X3, 2)
        X3[1, j] = rand(rng_grid3) * float(grid3_nx) - 1e-3
        X3[2, j] = rand(rng_grid3) * float(grid3_ny) - 1e-3
        X3[3, j] = rand(rng_grid3) * float(grid3_nz) - 1e-3
    end
    dest_3 = Vector{Int}(undef, nqueries_grid3)
    push!(rows, _bench("plpoly_locate_many_grid3_nocache_serial", reps_grid3, () -> begin
        PLP.locate_many!(dest_3, pi_grid3, X3; threaded=false, mode=:fast)
    end))
    push!(rows, _bench("plpoly_locate_many_grid3_nocache_threaded", reps_grid3, () -> begin
        PLP.locate_many!(dest_3, pi_grid3, X3; threaded=true, mode=:fast)
    end))

    Xg_skew = Matrix{Float64}(undef, 2, nqueries_grid)
    @inbounds for j in 1:size(Xg_skew, 2)
        Xg_skew[1, j] = 0.25
        Xg_skew[2, j] = 0.25
    end
    dest_skew = Vector{Int}(undef, nqueries_grid)
    push!(rows, _bench("plpoly_locate_many_grid_group_off_skewed", reps_grid, () -> begin
        _with_bool_toggle(PLP._LOCATE_BUCKET_GROUPING, false) do
            PLP.locate_many!(dest_skew, pi_grid, Xg_skew; threaded=true, mode=:fast)
        end
    end))
    push!(rows, _bench("plpoly_locate_many_grid_group_on_skewed", reps_grid, () -> begin
        _with_bool_toggle(PLP._LOCATE_BUCKET_GROUPING, true) do
            PLP.locate_many!(dest_skew, pi_grid, Xg_skew; threaded=true, mode=:fast)
        end
    end))
    push!(rows, _bench("plpoly_locate_many_grid_group_auto_skewed", reps_grid, () -> begin
        PLP.locate_many!(dest_skew, pi_grid, Xg_skew; threaded=true, mode=:fast)
    end))
    return rows
end

function _write_rows(path::AbstractString, rows)
    open(path, "w") do io
        println(io, "probe,median_ms,median_bytes")
        for r in rows
            println(io, string(r.probe, ",", r.median_ms, ",", r.median_bytes))
        end
    end
end

function main(args)
    if !PLP.HAVE_POLY
        println("Polyhedra backend unavailable; skipping PLPolyhedra microbench.")
        return
    end
    nregions = _parse_arg(args, "--nregions", 64)
    nqueries = _parse_arg(args, "--nqueries", 12_000)
    reps_locate = _parse_arg(args, "--reps_locate", 9)
    reps_breakdown = _parse_arg(args, "--reps_breakdown", 7)
    reps_adj = _parse_arg(args, "--reps_adj", 5)
    reps_membership = _parse_arg(args, "--reps_membership", 13)
    reps_facets = _parse_arg(args, "--reps_facets", 5)
    grid_side = _parse_arg(args, "--grid_side", 32)
    nqueries_grid = _parse_arg(args, "--nqueries_grid", 50_000)
    reps_grid = _parse_arg(args, "--reps_grid", 5)
    grid3_nx = _parse_arg(args, "--grid3_nx", 8)
    grid3_ny = _parse_arg(args, "--grid3_ny", 8)
    grid3_nz = _parse_arg(args, "--grid3_nz", 4)
    nqueries_grid3 = _parse_arg(args, "--nqueries_grid3", 40_000)
    reps_grid3 = _parse_arg(args, "--reps_grid3", 5)
    out = _parse_out(args, joinpath(@__DIR__, "_tmp_plpoly_cache_microbench.csv"))

    rows = run_microbench(; nregions=nregions, nqueries=nqueries,
                          reps_locate=reps_locate, reps_breakdown=reps_breakdown,
                          reps_adj=reps_adj, reps_membership=reps_membership,
                          reps_facets=reps_facets,
                          grid_side=grid_side, nqueries_grid=nqueries_grid,
                          reps_grid=reps_grid,
                          grid3_nx=grid3_nx, grid3_ny=grid3_ny, grid3_nz=grid3_nz,
                          nqueries_grid3=nqueries_grid3, reps_grid3=reps_grid3)
    _write_rows(out, rows)
    println("Wrote ", out)
    for r in rows
        println(r)
    end
end

main(ARGS)
