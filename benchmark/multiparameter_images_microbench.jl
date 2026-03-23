#!/usr/bin/env julia
# multiparameter_images_microbench.jl
#
# Purpose
# - Benchmark the current hot surfaces in `MultiparameterImages.jl`.
# - Keep the benchmark sectional so decomposition, image evaluation, and
#   multiparameter landscape paths can be profiled independently.
#
# Benchmark-to-kernel mapping
# - `decomp`    -> MPPI line-family construction and vineyard-style decomposition
#   plus isolated successive bottleneck matching over slice-barcode pools
# - `image`     -> MPPI image evaluation from decomposition/cache and image ops
# - `landscape` -> multiparameter landscape build/reuse and landscape ops
#
# Timing policy
# - Warm-process microbenchmarking (`@timed` median over reps)
# - Probe names encode cache state explicitly when relevant
#
# Usage
#   julia --project=. benchmark/multiparameter_images_microbench.jl
#   julia --project=. benchmark/multiparameter_images_microbench.jl --section=image --reps=3
#   julia --project=. benchmark/multiparameter_images_microbench.jl --profile=larger --field=f3

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)

using Random
using Statistics

function _load_posetmodules_for_mpi_bench()
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
            save_mpp_decomposition_json(args...; kwargs...) = error("Serialization stub unavailable in MPI benchmark bootstrap")
            load_mpp_decomposition_json(args...; kwargs...) = error("Serialization stub unavailable in MPI benchmark bootstrap")
            save_mpp_image_json(args...; kwargs...) = error("Serialization stub unavailable in MPI benchmark bootstrap")
            load_mpp_image_json(args...; kwargs...) = error("Serialization stub unavailable in MPI benchmark bootstrap")
        end
        include(joinpath(@__DIR__, "..", "src", "InvariantCore.jl"))
        include(joinpath(@__DIR__, "..", "src", "SliceInvariants.jl"))
        include(joinpath(@__DIR__, "..", "src", "Fibered2D.jl"))
        include(joinpath(@__DIR__, "..", "src", "MultiparameterImages.jl"))

        DataTypes.nvertices(P::FiniteFringe.AbstractPoset) = FiniteFringe.nvertices(P)

        module Advanced
            const CoreModules = Main.TamerOp.CoreModules
            const Options = Main.TamerOp.Options
            const EncodingCore = Main.TamerOp.EncodingCore
            const FiniteFringe = Main.TamerOp.FiniteFringe
            const Modules = Main.TamerOp.Modules
            const IndicatorResolutions = Main.TamerOp.IndicatorResolutions
            const PLBackend = Main.TamerOp.PLBackend
            const SliceInvariants = Main.TamerOp.SliceInvariants
            const Fibered2D = Main.TamerOp.Fibered2D
            const MultiparameterImages = Main.TamerOp.MultiparameterImages
        end
    end
    @eval using .TamerOp
    return :source_mpi_only
end

const _PM_LOAD_MODE = _load_posetmodules_for_mpi_bench()
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
const MPI = TO.MultiparameterImages

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
            n_dirs = 5,
            n_offsets = 5,
            max_den = 4,
            mpp_N = 6,
            mpp_resolution = 16,
            tgrid_n = 25,
            kmax = 2,
        )
    elseif profile == "larger"
        return (
            xcuts = 4,
            ycuts = 4,
            diag_copies = 8,
            side_copies = 4,
            tip_copies = 2,
            n_dirs = 9,
            n_offsets = 9,
            max_den = 6,
            mpp_N = 8,
            mpp_resolution = 24,
            tgrid_n = 41,
            kmax = 3,
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
    println(rpad(row.probe, 48),
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

    need_fibered = section in ("decomp", "image", "all")
    need_landscape = section in ("landscape", "all")

    arr = need_fibered ? F2D.fibered_arrangement_2d(pi, opts; include_axes=true, normalize_dirs=:L1, precompute=:cells, threads=threads) : nothing
    cacheM = need_fibered ? F2D.fibered_barcode_cache_2d(M, arr; precompute=:full, threads=threads) : nothing
    cacheN = need_fibered ? F2D.fibered_barcode_cache_2d(N, arr; precompute=:full, threads=threads) : nothing

    matchA = nothing
    matchB = nothing
    match_pool = nothing
    match_startA = 0
    match_countA = 0
    match_startB = 0
    match_countB = 0
    if need_fibered
        lines = MPI._line_families_carriere(arr; N=cfg.mpp_N, delta=:auto, tie_break=:center)
        bc_dicts = Vector{Dict{Tuple{Float64,Float64},Int}}(undef, length(lines))
        counts = zeros(Int, length(lines))
        total_pts = 0
        for i in eachindex(lines)
            spec = lines[i]
            bc = F2D.fibered_barcode(cacheM, spec.dir, spec.off; values=:t, tie_break=:center)
            bc_dicts[i] = bc
            cnt = MPI._barcode_point_count(bc)
            counts[i] = cnt
            total_pts += cnt
        end
        offsets = zeros(Int, length(lines))
        next_offset = 0
        for i in eachindex(lines)
            offsets[i] = next_offset
            next_offset += counts[i]
        end
        pool = Vector{Tuple{Float64,Float64}}(undef, total_pts)
        cursor = 1
        for i in eachindex(lines)
            cursor = MPI._fill_barcode_points!(pool, cursor, bc_dicts[i])
        end
        best_i = 0
        best_score = -1
        for i in 1:(length(lines) - 1)
            score = counts[i] * counts[i + 1]
            if score > best_score
                best_score = score
                best_i = i
            end
        end
        if best_i != 0 && counts[best_i] > 0 && counts[best_i + 1] > 0
            match_pool = pool
            match_startA = offsets[best_i] + 1
            match_countA = counts[best_i]
            match_startB = offsets[best_i + 1] + 1
            match_countB = counts[best_i + 1]
            matchA = collect(@view(pool[match_startA:(match_startA + match_countA - 1)]))
            matchB = collect(@view(pool[match_startB:(match_startB + match_countB - 1)]))
        end
    end

    decompM = section in ("image", "all") ? MPI.mpp_decomposition(cacheM; N=cfg.mpp_N, delta=:auto, q=1.0) : nothing
    decompN = section in ("image", "all") ? MPI.mpp_decomposition(cacheN; N=cfg.mpp_N, delta=:auto, q=1.0) : nothing
    imgM = section in ("image", "all") ? MPI.mpp_image(decompM; resolution=cfg.mpp_resolution, sigma=0.1, threads=threads) : nothing
    imgN = section in ("image", "all") ? MPI.mpp_image(decompN; resolution=cfg.mpp_resolution, sigma=0.1, threads=threads) : nothing

    dirs = need_landscape ? [collect(Float64, d) for d in SI.default_directions(pi; n_dirs=cfg.n_dirs, max_den=cfg.max_den, include_axes=true, normalize=:none)] : nothing
    offs = need_landscape ? SI.default_offsets(pi, opts; n_offsets=cfg.n_offsets) : nothing
    slice_cache = need_landscape ? SI.SlicePlanCache() : nothing
    plan = need_landscape ? SI.compile_slices(
        pi,
        opts;
        directions=dirs,
        offsets=offs,
        normalize_dirs=:none,
        direction_weight=:uniform,
        normalize_weights=true,
        threads=threads,
        cache=slice_cache,
    ) : nothing
    tgrid = need_landscape ? collect(range(-0.5, cfg.xcuts + cfg.ycuts + 1.5, length=cfg.tgrid_n)) : nothing
    Lm = section in ("landscape", "all") ? MPI.mp_landscape(M, plan; kmax=cfg.kmax, tgrid=tgrid, threads=threads) : nothing
    Ln = section in ("landscape", "all") ? MPI.mp_landscape(N, plan; kmax=cfg.kmax, tgrid=tgrid, threads=threads) : nothing

    return (
        field = field,
        cfg = cfg,
        P = P,
        pi = pi,
        opts = opts,
        M = M,
        N = N,
        arr = arr,
        cacheM = cacheM,
        cacheN = cacheN,
        matchA = matchA,
        matchB = matchB,
        match_pool = match_pool,
        match_startA = match_startA,
        match_countA = match_countA,
        match_startB = match_startB,
        match_countB = match_countB,
        decompM = decompM,
        decompN = decompN,
        imgM = imgM,
        imgN = imgN,
        dirs = dirs,
        offs = offs,
        slice_cache = slice_cache,
        plan = plan,
        tgrid = tgrid,
        Lm = Lm,
        Ln = Ln,
    )
end

function main(args=ARGS)
    section = _parse_string_arg(args, "section", "all")
    field_name = _parse_string_arg(args, "field", "qq")
    profile = _parse_string_arg(args, "profile", "default")
    reps = _parse_int_arg(args, "reps", 5)
    threads = _parse_bool_arg(args, "threads", false)
    out = _parse_path_arg(args, "out", joinpath(@__DIR__, "_tmp_multiparameter_images_microbench.csv"))

    field = _field_from_name(field_name)
    fx = _build_fixture(field, profile; threads=threads, section=section)
    rows = NamedTuple[]

    if _section_enabled(section, "decomp")
        push!(rows, _bench("mpi mpp_decomposition cache", reps=reps) do
            MPI.mpp_decomposition(fx.cacheM; N=fx.cfg.mpp_N, delta=:auto, q=1.0)
        end)
        push!(rows, _bench("mpi mpp_decomposition direct", reps=reps) do
            MPI.mpp_decomposition(fx.M, fx.pi, fx.opts)
        end)
        if fx.matchA !== nothing
            push!(rows, _bench("mpi bottleneck_matching generic", reps=reps) do
                MPI._bottleneck_matching_points(fx.matchA, fx.matchB)
            end)
            push!(rows, _bench("mpi bottleneck_matching flat", reps=reps) do
                MPI._bottleneck_matching_points_flat(
                    fx.match_pool,
                    fx.match_startA,
                    fx.match_countA,
                    fx.match_startB,
                    fx.match_countB,
                )
            end)
        end
    end

    if _section_enabled(section, "image")
        push!(rows, _bench("mpi mpp_image from decomp", reps=reps) do
            MPI.mpp_image(fx.decompM; resolution=fx.cfg.mpp_resolution, sigma=0.1, threads=threads)
        end)
        push!(rows, _bench("mpi mpp_image from cache", reps=reps) do
            MPI.mpp_image(fx.cacheM; resolution=fx.cfg.mpp_resolution, sigma=0.1, N=fx.cfg.mpp_N, delta=:auto, q=1.0, threads=threads)
        end)
        push!(rows, _bench("mpi mpp_image direct", reps=reps) do
            MPI.mpp_image(fx.M, fx.pi, fx.opts; resolution=fx.cfg.mpp_resolution, sigma=0.1, N=fx.cfg.mpp_N, delta=:auto, q=1.0, threads=threads)
        end)
        push!(rows, _bench("mpi mpp_image_distance", reps=reps) do
            MPI.mpp_image_distance(fx.imgM, fx.imgN)
        end)
        push!(rows, _bench("mpi mpp_image_kernel", reps=reps) do
            MPI.mpp_image_kernel(fx.imgM, fx.imgN; sigma=1.0)
        end)
    end

    if _section_enabled(section, "landscape")
        push!(rows, _bench("mpi mp_landscape plan", reps=reps) do
            MPI.mp_landscape(fx.M, fx.plan; kmax=fx.cfg.kmax, tgrid=fx.tgrid, threads=threads)
        end)
        push!(rows, _bench("mpi mp_landscape via pi", reps=reps) do
            MPI.mp_landscape(fx.M, fx.pi, fx.opts;
                directions=fx.dirs,
                offsets=fx.offs,
                kmax=fx.cfg.kmax,
                tgrid=fx.tgrid,
                cache=fx.slice_cache,
            )
        end)
        push!(rows, _bench("mpi mp_landscape_distance", reps=reps) do
            MPI.mp_landscape_distance(fx.Lm, fx.Ln)
        end)
        push!(rows, _bench("mpi mp_landscape_inner_product", reps=reps) do
            MPI.mp_landscape_inner_product(fx.Lm, fx.Ln)
        end)
        push!(rows, _bench("mpi mp_landscape_kernel", reps=reps) do
            MPI.mp_landscape_kernel(fx.Lm, fx.Ln; kind=:gaussian, sigma=1.0)
        end)
    end

    _write_csv(out, rows)
    println("\nMultiparameterImages benchmark complete.")
    println("load_mode = ", _PM_LOAD_MODE)
    println("rows = ", length(rows))
    println("output = ", out)
    return nothing
end

main()
