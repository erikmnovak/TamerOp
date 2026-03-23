#!/usr/bin/env julia
# fibered2d_microbench.jl
#
# Purpose
# - Benchmark the current hot surfaces in `Fibered2D.jl`.
# - Keep the benchmark sectional so arrangement, cache, exact-distance, and
#   projected-barcode paths can be profiled independently.
#
# Benchmark-to-kernel mapping
# - `arrangement` -> arrangement build, slice-family build/reuse, cell/chain queries
# - `cache`       -> module cache build, first/warm fibered barcode queries, cached slice grids
# - `distance`    -> exact 2D matching distance direct/cached, cached slice kernel
# - `projected`   -> projected arrangements, projected barcode caches, projected distances/kernels
#
# Timing policy
# - Warm-process microbenchmarking (`@timed` median over reps)
# - Probe names encode cache state explicitly when relevant
#
# Usage
#   julia --project=. benchmark/fibered2d_microbench.jl
#   julia --project=. benchmark/fibered2d_microbench.jl --section=distance --reps=3
#   julia --project=. benchmark/fibered2d_microbench.jl --profile=larger --field=f3

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)

using Random
using Statistics

function _load_posetmodules_for_fibered_bench()
    try
        @eval using TamerOp
        return :package
    catch
    end

    @eval module TamerOp
        include(joinpath(@__DIR__, "..", "src", "CoreModules.jl"))
        include(joinpath(@__DIR__, "..", "src", "Stats.jl"))
        include(joinpath(@__DIR__, "..", "src", "Options.jl"))
        include(joinpath(@__DIR__, "..", "src", "DataTypes.jl"))
        include(joinpath(@__DIR__, "..", "src", "EncodingCore.jl"))
        include(joinpath(@__DIR__, "..", "src", "Results.jl"))
        include(joinpath(@__DIR__, "..", "src", "RegionGeometry.jl"))
        include(joinpath(@__DIR__, "..", "src", "FieldLinAlg.jl"))
        include(joinpath(@__DIR__, "..", "src", "FiniteFringe.jl"))
        include(joinpath(@__DIR__, "..", "src", "IndicatorTypes.jl"))
        include(joinpath(@__DIR__, "..", "src", "Encoding.jl"))
        include(joinpath(@__DIR__, "..", "src", "Modules.jl"))
        include(joinpath(@__DIR__, "..", "src", "AbelianCategories.jl"))
        include(joinpath(@__DIR__, "..", "src", "IndicatorResolutions.jl"))
        include(joinpath(@__DIR__, "..", "src", "FlangeZn.jl"))
        include(joinpath(@__DIR__, "..", "src", "ZnEncoding.jl"))
        include(joinpath(@__DIR__, "..", "src", "PLPolyhedra.jl"))
        include(joinpath(@__DIR__, "..", "src", "PLBackend.jl"))
        include(joinpath(@__DIR__, "..", "src", "ChainComplexes.jl"))
        include(joinpath(@__DIR__, "..", "src", "DerivedFunctors.jl"))
        include(joinpath(@__DIR__, "..", "src", "ModuleComplexes.jl"))
        include(joinpath(@__DIR__, "..", "src", "ChangeOfPosets.jl"))
        module Serialization
            export save_mpp_decomposition_json, load_mpp_decomposition_json,
                   save_mpp_image_json, load_mpp_image_json
            save_mpp_decomposition_json(args...; kwargs...) = error("Serialization stub unavailable in fibered benchmark bootstrap")
            load_mpp_decomposition_json(args...; kwargs...) = error("Serialization stub unavailable in fibered benchmark bootstrap")
            save_mpp_image_json(args...; kwargs...) = error("Serialization stub unavailable in fibered benchmark bootstrap")
            load_mpp_image_json(args...; kwargs...) = error("Serialization stub unavailable in fibered benchmark bootstrap")
        end
        include(joinpath(@__DIR__, "..", "src", "InvariantCore.jl"))
        include(joinpath(@__DIR__, "..", "src", "SliceInvariants.jl"))
        include(joinpath(@__DIR__, "..", "src", "Fibered2D.jl"))

        DataTypes.nvertices(P::FiniteFringe.AbstractPoset) = FiniteFringe.nvertices(P)

        module Advanced
            const CoreModules = Main.TamerOp.CoreModules
            const Options = Main.TamerOp.Options
            const EncodingCore = Main.TamerOp.EncodingCore
            const FiniteFringe = Main.TamerOp.FiniteFringe
            const Modules = Main.TamerOp.Modules
            const IndicatorResolutions = Main.TamerOp.IndicatorResolutions
            const PLBackend = Main.TamerOp.PLBackend
            const InvariantCore = Main.TamerOp.InvariantCore
            const SliceInvariants = Main.TamerOp.SliceInvariants
            const Fibered2D = Main.TamerOp.Fibered2D
        end
    end
    @eval using .TamerOp
    return :source_fibered_only
end

const _PM_LOAD_MODE = _load_posetmodules_for_fibered_bench()
const TO = isdefined(TamerOp, :Advanced) ? TamerOp.Advanced : TamerOp
const CM = TO.CoreModules
const OPT = TO.Options
const EC = TO.EncodingCore
const FF = TO.FiniteFringe
const MD = TO.Modules
const IR = TO.IndicatorResolutions
const PLB = TO.PLBackend
const SI = TO.SliceInvariants
const F2D = TO.Fibered2D

function _parse_int_arg(args, key::String, default::Int)
    for a in args
        s = startswith(a, "--") ? a[3:end] : a
        startswith(s, key * "=") || continue
        return max(1, parse(Int, split(s, "=", limit=2)[2]))
    end
    return default
end

function _parse_string_arg(args, key::String, default::String)
    for a in args
        s = startswith(a, "--") ? a[3:end] : a
        startswith(s, key * "=") || continue
        return lowercase(strip(split(s, "=", limit=2)[2]))
    end
    return default
end

function _parse_bool_arg(args, key::String, default::Bool)
    for a in args
        s = startswith(a, "--") ? a[3:end] : a
        startswith(s, key * "=") || continue
        v = lowercase(strip(split(s, "=", limit=2)[2]))
        v in ("true", "1", "yes", "on") && return true
        v in ("false", "0", "no", "off") && return false
        error("invalid boolean for $key: $v")
    end
    return default
end

function _parse_path_arg(args, key::String, default::String)
    for a in args
        s = startswith(a, "--") ? a[3:end] : a
        startswith(s, key * "=") || continue
        return String(strip(split(s, "=", limit=2)[2]))
    end
    return default
end

_section_enabled(section::String, group::String) = section == "all" || section == group

function _field_from_name(name::String)
    name == "qq" && return CM.QQField()
    name == "f2" && return CM.F2()
    name == "f3" && return CM.F3()
    name == "f5" && return CM.Fp(5)
    error("unknown field '$name' (supported: qq, f2, f3, f5)")
end

function _profile_defaults(profile::String)
    if profile == "default"
        return (
            xcuts = 2,
            ycuts = 2,
            diag_copies = 4,
            side_copies = 2,
            tip_copies = 1,
            nproj = 3,
        )
    elseif profile == "larger"
        return (
            xcuts = 4,
            ycuts = 4,
            diag_copies = 8,
            side_copies = 4,
            tip_copies = 2,
            nproj = 5,
        )
    else
        error("unknown profile '$profile' (supported: default, larger)")
    end
end

function _bench(name::AbstractString, f::Function; reps::Int=7, setup::Union{Nothing,Function}=nothing)
    GC.gc()
    setup === nothing || setup()
    f()
    GC.gc()
    times_ms = Vector{Float64}(undef, reps)
    alloc_kib = Vector{Float64}(undef, reps)
    for i in 1:reps
        setup === nothing || setup()
        m = @timed f()
        times_ms[i] = 1000.0 * m.time
        alloc_kib[i] = m.bytes / 1024.0
    end
    sort!(times_ms)
    sort!(alloc_kib)
    mid = cld(reps, 2)
    row = (probe=String(name), median_ms=times_ms[mid], median_kib=alloc_kib[mid])
    println(rpad(row.probe, 46),
            " median_time=", round(row.median_ms, digits=3), " ms",
            " median_alloc=", round(row.median_kib, digits=1), " KiB")
    return row
end

_bench(f::Function, name::AbstractString; reps::Int=7, setup::Union{Nothing,Function}=nothing) =
    _bench(name, f; reps=reps, setup=setup)

function _write_csv(path::AbstractString, rows)
    open(path, "w") do io
        println(io, "probe,median_ms,median_kib")
        for r in rows
            println(io, string(r.probe, ",", r.median_ms, ",", r.median_kib))
        end
    end
end

function _repeat_direct_sum(ms::Vector{<:MD.PModule})
    isempty(ms) && error("_repeat_direct_sum: empty module list")
    cur = ms[1]
    for m in ms[2:end]
        cur = MD.direct_sum(cur, m)
    end
    return cur
end

function _build_grid_encoding(xcuts::Int, ycuts::Int)
    Ups = PLB.BoxUpset[]
    for t in 0:(xcuts - 1)
        push!(Ups, PLB.BoxUpset([float(t), -10.0]))
    end
    for t in 0:(ycuts - 1)
        push!(Ups, PLB.BoxUpset([-10.0, float(t)]))
    end
    Downs = PLB.BoxDownset[]
    P, _Henc, pi = PLB.encode_fringe_boxes(Ups, Downs, OPT.EncodingOptions())
    return (P=P, pi=pi)
end

function _projected_arrangement_from_reps(P, reps, nproj::Int)
    dirs_pool = (
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, 1.0],
        [1.0, 2.0],
    )
    nproj <= length(dirs_pool) || error("nproj=$nproj exceeds built-in projection pool")
    coords = [Float64[float(r[1]), float(r[2])] for r in reps]
    vals_pool = (
        [c[1] for c in coords],
        [c[2] for c in coords],
        [c[1] + c[2] for c in coords],
        [2c[1] + c[2] for c in coords],
        [c[1] + 2c[2] for c in coords],
    )
    projs = Vector{F2D.ProjectedArrangement1D{typeof(P)}}(undef, nproj)
    for i in 1:nproj
        tmp = F2D.projected_arrangement(P, vals_pool[i]; dir=dirs_pool[i])
        projs[i] = tmp.projections[1]
    end
    return F2D.ProjectedArrangement(P, projs)
end

function _build_fixture(field::CM.AbstractCoeffField, profile::String; threads::Bool, section::String="all")
    cfg = _profile_defaults(profile)
    cf(x) = CM.coerce(field, x)

    enc = _build_grid_encoding(cfg.xcuts, cfg.ycuts)
    P = enc.P
    pi = enc.pi

    xhi = cfg.xcuts - 0.5
    yhi = cfg.ycuts - 0.5
    r_center = EC.locate(pi, [0.5, 0.5])
    r_xhi = EC.locate(pi, [xhi, 0.5])
    r_yhi = EC.locate(pi, [0.5, yhi])
    r_hi = EC.locate(pi, [xhi, yhi])

    base_diag = IR.pmodule_from_fringe(
        FF.one_by_one_fringe(P, FF.principal_upset(P, r_center), FF.principal_downset(P, r_hi), cf(1); field=field)
    )
    base_x = IR.pmodule_from_fringe(
        FF.one_by_one_fringe(P, FF.principal_upset(P, r_xhi), FF.principal_downset(P, r_hi), cf(1); field=field)
    )
    base_y = IR.pmodule_from_fringe(
        FF.one_by_one_fringe(P, FF.principal_upset(P, r_yhi), FF.principal_downset(P, r_hi), cf(1); field=field)
    )
    base_tip = IR.pmodule_from_fringe(
        FF.one_by_one_fringe(P, FF.principal_upset(P, r_hi), FF.principal_downset(P, r_hi), cf(1); field=field)
    )

    M = _repeat_direct_sum(vcat(
        fill(base_diag, cfg.diag_copies),
        fill(base_x, cfg.side_copies),
        fill(base_tip, cfg.tip_copies),
    ))
    N = _repeat_direct_sum(vcat(
        fill(base_diag, cfg.diag_copies),
        fill(base_y, cfg.side_copies),
        fill(base_tip, cfg.tip_copies),
    ))

    opts = OPT.InvariantOptions(
        box = ([-1.0, -1.0], [cfg.xcuts + 1.0, cfg.ycuts + 1.0]),
        threads = threads,
    )

    need_arr = section in ("arrangement", "cache", "distance", "all")
    need_distance = section in ("distance", "all")
    need_projected = section in ("projected", "all")

    arr = need_arr ? F2D.fibered_arrangement_2d(pi, opts; normalize_dirs=:L1, precompute=:cells, threads=threads) : nothing
    fam = section in ("arrangement", "distance", "all") ?
        F2D.fibered_slice_family_2d(arr; direction_weight=:lesnick_l1, store_values=true) : nothing
    cacheM = section in ("cache", "distance", "all") ?
        F2D.fibered_barcode_cache_2d(M, arr; precompute=:full, threads=threads) : nothing
    cacheN = need_distance ?
        F2D.fibered_barcode_cache_2d(N, arr; precompute=:full, threads=threads) : nothing

    reps = need_projected ? EC.representatives(pi) : nothing
    proj_arr = need_projected ? _projected_arrangement_from_reps(P, reps, cfg.nproj) : nothing
    pcacheM = need_projected ? F2D.projected_barcode_cache(M, proj_arr; precompute=true) : nothing
    pcacheN = need_projected ? F2D.projected_barcode_cache(N, proj_arr; precompute=true) : nothing

    return (
        field = field,
        cfg = cfg,
        P = P,
        pi = pi,
        opts = opts,
        M = M,
        N = N,
        arr = arr,
        fam = fam,
        cacheM = cacheM,
        cacheN = cacheN,
        reps = reps,
        proj_arr = proj_arr,
        pcacheM = pcacheM,
        pcacheN = pcacheN,
        query_dir = [1.0, 1.0],
        alt_dir = [1.0, 2.0],
        query_offset = 0.0,
        alt_offset = 0.5,
    )
end

function _fibered_barcode_cache_distance_baseline(
    M,
    arr::F2D.FiberedArrangement2D;
    direction_weight::Symbol=:lesnick_l1,
    threads::Bool=(Threads.nthreads() > 1),
)
    cache = F2D.fibered_barcode_cache_2d(M, arr; precompute=:family, threads=threads)
    fam = F2D.fibered_slice_family_2d(arr; direction_weight=direction_weight, store_values=true)
    F2D._precompute_family_barcodes!(cache, fam; threads=threads)
    F2D._precompute_distance_payload!(cache, fam; threads=threads)
    return cache
end

function main(args=ARGS)
    section = _parse_string_arg(args, "section", "all")
    field_name = _parse_string_arg(args, "field", "qq")
    profile = _parse_string_arg(args, "profile", "default")
    reps = _parse_int_arg(args, "reps", 5)
    threads = _parse_bool_arg(args, "threads", false)
    out = _parse_path_arg(args, "out", joinpath(@__DIR__, "_tmp_fibered2d_microbench.csv"))

    field = _field_from_name(field_name)
    fx = _build_fixture(field, profile; threads=threads, section=section)
    rows = NamedTuple[]

    if _section_enabled(section, "arrangement")
        push!(rows, _bench("f2d arrangement build", reps=reps) do
            F2D.fibered_arrangement_2d(fx.pi, fx.opts; normalize_dirs=:L1, precompute=:cells, threads=threads)
        end)
        push!(rows, _bench("f2d slice_family cold", reps=reps) do
            local arr = F2D.fibered_arrangement_2d(fx.pi, fx.opts; normalize_dirs=:L1, precompute=:cells, threads=threads)
            F2D.fibered_slice_family_2d(arr; direction_weight=:lesnick_l1, store_values=true)
        end)
        push!(rows, _bench("f2d slice_family warm", reps=reps) do
            F2D.fibered_slice_family_2d(fx.arr; direction_weight=:lesnick_l1, store_values=true)
        end)
        push!(rows, _bench("f2d fibered_cell_id", reps=reps) do
            F2D.fibered_cell_id(fx.arr, fx.query_dir, fx.query_offset)
        end)
        push!(rows, _bench("f2d fibered_chain", reps=reps) do
            F2D.fibered_chain(fx.arr, fx.query_dir, fx.query_offset; copy=false)
        end)
    end

    if _section_enabled(section, "cache")
        push!(rows, _bench("f2d barcode_cache build family", reps=reps) do
            local arr = F2D.fibered_arrangement_2d(fx.pi, fx.opts; normalize_dirs=:L1, precompute=:cells, threads=threads)
            F2D.fibered_barcode_cache_2d(fx.M, arr; precompute=:family, threads=threads)
        end)
        push!(rows, _bench("f2d barcode_cache build barcodes", reps=reps) do
            local arr = F2D.fibered_arrangement_2d(fx.pi, fx.opts; normalize_dirs=:L1, precompute=:cells, threads=threads)
            F2D.fibered_barcode_cache_2d(fx.M, arr; precompute=:barcodes, threads=threads)
        end)
        push!(rows, _bench("f2d barcode_cache build distance", reps=reps) do
            local arr = F2D.fibered_arrangement_2d(fx.pi, fx.opts; normalize_dirs=:L1, precompute=:cells, threads=threads)
            F2D.fibered_barcode_cache_2d(fx.M, arr; precompute=:distance, threads=threads)
        end)
        push!(rows, _bench("f2d barcode_cache build distance baseline duplicate", reps=reps) do
            local arr = F2D.fibered_arrangement_2d(fx.pi, fx.opts; normalize_dirs=:L1, precompute=:cells, threads=threads)
            _fibered_barcode_cache_distance_baseline(fx.M, arr; threads=threads)
        end)
        push!(rows, _bench("f2d barcode_cache build full", reps=reps) do
            local arr = F2D.fibered_arrangement_2d(fx.pi, fx.opts; normalize_dirs=:L1, precompute=:cells, threads=threads)
            F2D.fibered_barcode_cache_2d(fx.M, arr; precompute=:full, threads=threads)
        end)
        push!(rows, _bench("f2d fibered_barcode first", reps=reps) do
            local cache = F2D.fibered_barcode_cache_2d(fx.M, fx.arr; precompute=:none, threads=threads)
            F2D.fibered_barcode(cache, fx.query_dir, fx.query_offset; values=:t)
        end)
        push!(rows, _bench("f2d fibered_barcode first family", reps=reps) do
            local cache = F2D.fibered_barcode_cache_2d(fx.M, fx.arr; precompute=:family, threads=threads)
            F2D.fibered_barcode(cache, fx.query_dir, fx.query_offset; values=:t)
        end)
        push!(rows, _bench("f2d fibered_barcode first barcodes", reps=reps) do
            local cache = F2D.fibered_barcode_cache_2d(fx.M, fx.arr; precompute=:barcodes, threads=threads)
            F2D.fibered_barcode(cache, fx.query_dir, fx.query_offset; values=:t)
        end)
        push!(rows, _bench("f2d fibered_barcode warm", reps=reps) do
            F2D.fibered_barcode(fx.cacheM, fx.query_dir, fx.query_offset; values=:t)
        end)
        push!(rows, _bench("f2d slice_barcodes cached", reps=reps) do
            F2D.slice_barcodes(fx.cacheM;
                dirs=[fx.query_dir, fx.alt_dir],
                offsets=[fx.query_offset, fx.alt_offset],
                values=:t,
                packed=true,
                threads=threads,
            )
        end)
    end

    if _section_enabled(section, "distance")
        push!(rows, _bench("f2d exact_distance direct", reps=reps) do
            F2D.matching_distance_exact_2d(fx.M, fx.N, fx.pi, fx.opts;
                weight=:lesnick_l1,
                normalize_dirs=:L1,
            )
        end)
        push!(rows, _bench("f2d exact_distance cached", reps=reps) do
            F2D.matching_distance_exact_2d(fx.cacheM, fx.cacheN;
                weight=:lesnick_l1,
                family=fx.fam,
                threads=threads,
            )
        end)
        push!(rows, _bench("f2d exact_distance cached second", reps=reps,
                           setup=() -> F2D.matching_distance_exact_2d(fx.cacheM, fx.cacheN;
                               weight=:lesnick_l1, family=fx.fam, threads=threads)) do
            F2D.matching_distance_exact_2d(fx.cacheM, fx.cacheN;
                weight=:lesnick_l1,
                family=fx.fam,
                threads=threads,
            )
        end)
        push!(rows, _bench("f2d slice_kernel cached", reps=reps) do
            F2D.slice_kernel(fx.cacheM, fx.cacheN;
                kind=:bottleneck_gaussian,
                sigma=1.0,
                direction_weight=:lesnick_l1,
                family=fx.fam,
                threads=threads,
            )
        end)
        push!(rows, _bench("f2d slice_kernel cached second", reps=reps,
                           setup=() -> F2D.slice_kernel(fx.cacheM, fx.cacheN;
                               kind=:bottleneck_gaussian, sigma=1.0,
                               direction_weight=:lesnick_l1, family=fx.fam, threads=threads)) do
            F2D.slice_kernel(fx.cacheM, fx.cacheN;
                kind=:bottleneck_gaussian,
                sigma=1.0,
                direction_weight=:lesnick_l1,
                family=fx.fam,
                threads=threads,
            )
        end)
    end

    if _section_enabled(section, "projected")
        push!(rows, _bench("f2d projected_arrangement multi", reps=reps) do
            _projected_arrangement_from_reps(fx.P, fx.reps, fx.cfg.nproj)
        end)
        push!(rows, _bench("f2d projected_cache build", reps=reps) do
            local arr = _projected_arrangement_from_reps(fx.P, fx.reps, fx.cfg.nproj)
            F2D.projected_barcode_cache(fx.M, arr; precompute=true)
        end)
        push!(rows, _bench("f2d projected_barcodes warm", reps=reps) do
            F2D.projected_barcodes(fx.pcacheM; threads=threads)
        end)
        push!(rows, _bench("f2d projected_distance", reps=reps) do
            F2D.projected_distance(fx.pcacheM, fx.pcacheN;
                dist=:bottleneck,
                agg=:mean,
                threads=threads,
            )
        end)
        push!(rows, _bench("f2d projected_kernel", reps=reps) do
            F2D.projected_kernel(fx.pcacheM, fx.pcacheN;
                kind=:wasserstein_gaussian,
                sigma=1.0,
                agg=:mean,
                threads=threads,
            )
        end)
    end

    _write_csv(out, rows)
    println("\nFibered2D benchmark complete.")
    println("load_mode = ", _PM_LOAD_MODE)
    println("rows = ", length(rows))
    println("output = ", out)
    return nothing
end

main()
