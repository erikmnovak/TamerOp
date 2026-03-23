using Random
using Printf

try
    using TamerOp
catch
    include(joinpath(@__DIR__, "..", "src", "TamerOp.jl"))
    using .TamerOp
end

const CM = TamerOp.CoreModules
const RG = TamerOp.RegionGeometry
const EC = TamerOp.EncodingCore
const FF = TamerOp.FiniteFringe
const PLP = TamerOp.PLPolyhedra
const PLB = TamerOp.PLBackend

function _parse_flag(args::Vector{String}, key::String, default)
    prefix = key * "="
    for arg in args
        startswith(arg, prefix) || continue
        value = split(arg, "=", limit=2)[2]
        return default isa Int ? parse(Int, value) : value
    end
    return default
end

function _median_stats(f::Function; reps::Int)
    times = Vector{Float64}(undef, reps)
    allocs = Vector{Int}(undef, reps)
    gctimes = Vector{Float64}(undef, reps)
    for i in 1:reps
        stats = @timed f()
        times[i] = stats.time
        allocs[i] = stats.bytes
        gctimes[i] = stats.gctime
    end
    p = sortperm(times)
    mid = p[cld(reps, 2)]
    return (time=times[mid], bytes=allocs[mid], gctime=gctimes[mid])
end

function _with_region_toggles(
    f::Function;
    batched::Bool,
    fast_wrappers::Bool,
    direct_volume::Bool,
    workspace_reuse::Bool,
    sampled_summary_cache::Bool,
    cached_locate_style::Bool,
)
    prev_batched = RG._REGION_BATCHED_LOCATE[]
    prev_fast = RG._REGION_FAST_WRAPPERS[]
    prev_thresh = RG._REGION_BATCHED_LOCATE_MIN_PROPOSALS[]
    prev_direct = RG._REGION_DIRECT_VOLUME[]
    prev_work = RG._REGION_WORKSPACE_REUSE[]
    prev_summary = RG._REGION_SAMPLED_SUMMARY_CACHE[]
    prev_style = EC._ENCODINGCORE_CACHED_LOCATE_STYLE[]
    RG._REGION_BATCHED_LOCATE[] = batched
    RG._REGION_FAST_WRAPPERS[] = fast_wrappers
    RG._REGION_BATCHED_LOCATE_MIN_PROPOSALS[] = 1
    RG._REGION_DIRECT_VOLUME[] = direct_volume
    RG._REGION_WORKSPACE_REUSE[] = workspace_reuse
    RG._REGION_SAMPLED_SUMMARY_CACHE[] = sampled_summary_cache
    EC._ENCODINGCORE_CACHED_LOCATE_STYLE[] = cached_locate_style
    RG._clear_region_geometry_runtime_caches!()
    EC._clear_locate_style_cache!()
    try
        return f()
    finally
        RG._REGION_BATCHED_LOCATE[] = prev_batched
        RG._REGION_FAST_WRAPPERS[] = prev_fast
        RG._REGION_BATCHED_LOCATE_MIN_PROPOSALS[] = prev_thresh
        RG._REGION_DIRECT_VOLUME[] = prev_direct
        RG._REGION_WORKSPACE_REUSE[] = prev_work
        RG._REGION_SAMPLED_SUMMARY_CACHE[] = prev_summary
        EC._ENCODINGCORE_CACHED_LOCATE_STYLE[] = prev_style
        RG._clear_region_geometry_runtime_caches!()
        EC._clear_locate_style_cache!()
    end
end

function _grid_fixture()
    P = FF.ProductOfChainsPoset((2, 2))
    pi = EC.GridEncodingMap(P, ([0.0, 1.0], [0.0, 2.0]))
    enc = EC.compile_encoding(P, pi; meta=CM.EncodingCache())
    box = ([0.0, 0.0], [2.0, 3.0])
    r = EC.locate(pi, [0.5, 1.0])
    dirs = RG._random_unit_directions(2, 96; rng=MersenneTwister(17))
    return (; pi, enc, box, r, dirs)
end

function _poly_fixture()
    PLP.HAVE_POLY || return nothing
    A = CM.QQ[1 0; 0 1; -1 0; 0 -1]
    b = CM.QQ[1, 1, 0, 0]
    hp = PLP.make_hpoly(A, b)
    pi = PLP.PLEncodingMap(2, [BitVector()], [BitVector()], [hp], [(0.5, 0.5)])
    box = (Float64[0.0, 0.0], Float64[1.0, 1.0])
    cache = PLP.poly_in_box_cache(pi; box=box, closure=true)
    return (; pi, box, cache)
end

function _boxes_fixture()
    Ups = [PLB.BoxUpset([0.0, 0.0])]
    Downs = [PLB.BoxDownset([2.0, 1.0])]
    Phi = reshape(CM.QQ[1], 1, 1)
    _, _, pi = PLB.encode_fringe_boxes(Ups, Downs, Phi)
    box = ([0.0, 0.0], [2.0, 1.0])
    r = EC.locate(pi, [0.5, 0.5])
    return (; pi, box, r)
end

const _BEFORE_TOGGLES = (
    batched=false,
    fast_wrappers=false,
    direct_volume=false,
    workspace_reuse=false,
    sampled_summary_cache=false,
    cached_locate_style=false,
)

const _AFTER_TOGGLES = (
    batched=true,
    fast_wrappers=true,
    direct_volume=true,
    workspace_reuse=true,
    sampled_summary_cache=true,
    cached_locate_style=true,
)

function main(args)
    reps = _parse_flag(args, "--reps", 7)
    out = _parse_flag(args, "--out", joinpath(@__DIR__, "_tmp_region_geometry_microbench.csv"))

    println("Timing policy: warm, same-process A/B toggle")
    println("reps=$reps")

    grid = _grid_fixture()
    poly = _poly_fixture()
    boxes = _boxes_fixture()

    cases = [
        ("principal_directions_grid", () -> begin
            RG.region_principal_directions(grid.pi, grid.r;
                box=grid.box, nsamples=2048, max_proposals=8192,
                rng=MersenneTwister(11), strict=true)
        end),
        ("principal_directions_info_grid", () -> begin
            RG.region_principal_directions(grid.pi, grid.r;
                box=grid.box, nsamples=2048, max_proposals=8192,
                rng=MersenneTwister(13), strict=true, return_info=true, nbatches=8)
        end),
        ("mean_width_grid", () -> begin
            RG.region_mean_width(grid.pi, grid.r;
                box=grid.box, method=:mc, ndirs=size(grid.dirs, 2), directions=grid.dirs,
                nsamples=2048, max_proposals=8192, rng=MersenneTwister(19), strict=true)
        end),
        ("anisotropy_descriptor_pair_grid", () -> begin
            RG.region_covariance_anisotropy(grid.enc, grid.r;
                box=grid.box, nsamples=2048, max_proposals=8192,
                rng=MersenneTwister(23), strict=true)
            RG.region_covariance_eccentricity(grid.enc, grid.r;
                box=grid.box, nsamples=2048, max_proposals=8192,
                rng=MersenneTwister(23), strict=true)
        end),
        ("mean_width_repeat_grid", () -> begin
            RG.region_mean_width(grid.enc, grid.r;
                box=grid.box, method=:mc, ndirs=size(grid.dirs, 2),
                nsamples=2048, max_proposals=8192, rng=MersenneTwister(29), strict=true)
            RG.region_mean_width(grid.enc, grid.r;
                box=grid.box, method=:mc, ndirs=size(grid.dirs, 2),
                nsamples=2048, max_proposals=8192, rng=MersenneTwister(29), strict=true)
        end),
        ("centroid_boxes", () -> begin
            RG.region_centroid(boxes.pi, boxes.r; box=boxes.box)
        end),
        ("boundary_to_volume_boxes", () -> begin
            RG.region_boundary_to_volume_ratio(boxes.pi, boxes.r; box=boxes.box)
        end),
        ("minkowski_boxes", () -> begin
            RG.region_minkowski_functionals(boxes.pi, boxes.r;
                box=boxes.box, mean_width_method=:cauchy)
        end),
    ]
    if poly !== nothing
        push!(cases, ("volume_poly_cache", () -> begin
            RG.region_volume(poly.pi, 1; box=poly.box, cache=poly.cache)
        end))
        push!(cases, ("boundary_to_volume_poly_cache", () -> begin
            RG.region_boundary_to_volume_ratio(poly.pi, 1; box=poly.box, cache=poly.cache)
        end))
        push!(cases, ("minkowski_poly_cache", () -> begin
            RG.region_minkowski_functionals(poly.pi, 1;
                box=poly.box, cache=poly.cache, mean_width_method=:cauchy)
        end))
    end

    rows = String[]
    push!(rows, "case,variant,time_ms,alloc_kib,gc_ms")

    for (label, f) in cases
        _with_region_toggles(f; _BEFORE_TOGGLES...)
        _with_region_toggles(f; _AFTER_TOGGLES...)

        before = _with_region_toggles(; _BEFORE_TOGGLES...) do
            _median_stats(f; reps=reps)
        end
        after = _with_region_toggles(; _AFTER_TOGGLES...) do
            _median_stats(f; reps=reps)
        end

        push!(rows, @sprintf("%s,before,%.6f,%.3f,%.6f", label, 1.0e3 * before.time, before.bytes / 1024, 1.0e3 * before.gctime))
        push!(rows, @sprintf("%s,after,%.6f,%.3f,%.6f", label, 1.0e3 * after.time, after.bytes / 1024, 1.0e3 * after.gctime))

        ratio = before.time / max(after.time, eps())
        println(@sprintf("%-28s  %.3f ms -> %.3f ms  (%.2fx),  %.1f KiB -> %.1f KiB",
            label, 1.0e3 * before.time, 1.0e3 * after.time, ratio,
            before.bytes / 1024, after.bytes / 1024))
    end

    open(out, "w") do io
        write(io, join(rows, "\n"))
        write(io, "\n")
    end
    println("wrote ", out)
end

main(ARGS)
