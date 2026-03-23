#!/usr/bin/env julia
# invariants_family_microbench.jl
#
# Purpose
# - Benchmark the current hot surfaces across the `Invariants` owner and its
#   sibling invariant families extracted from it.
# - Keep the benchmark sectional so each invariant family can be profiled in
#   isolation.
#
# Benchmark-to-owner mapping
# - `core`    -> `InvariantCore` + `Invariants` rank/Hilbert/statistics paths
# - `signed`  -> `SignedMeasures` Euler and rectangle-measure paths
# - `slice`   -> `SliceInvariants` slice-plan, barcode, and approximate distance paths
# - `fibered` -> `Fibered2D` arrangement/cache/exact matching paths
# - `images`  -> `MultiparameterImages` decomposition/image/landscape paths
#
# Timing policy
# - Warm-process microbenchmarking (`@timed` median over reps)
# - Probe names encode cold vs warm behavior explicitly when cache state matters
#
# Usage
#   julia --project=. benchmark/invariants_family_microbench.jl
#   julia --project=. benchmark/invariants_family_microbench.jl --section=slice --field=f3
#   julia --project=. benchmark/invariants_family_microbench.jl --profile=larger --reps=5

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)

using Random
using Statistics

try
    using TamerOp
catch
    include(joinpath(@__DIR__, "..", "src", "TamerOp.jl"))
    using .TamerOp
end

const CM = TamerOp.CoreModules
const OPT = TamerOp.Options
const FF = TamerOp.FiniteFringe
const MD = TamerOp.Modules
const IR = TamerOp.IndicatorResolutions
const EC = TamerOp.EncodingCore
const PLB = TamerOp.PLBackend
const IC = TamerOp.InvariantCore
const INV = TamerOp.Invariants
const SM = TamerOp.SignedMeasures
const SI = TamerOp.SliceInvariants
const F2D = TamerOp.Fibered2D
const MPI = TamerOp.MultiparameterImages

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

function _section_enabled(section::String, group::String)
    section == "all" && return true
    return section == group
end

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
            m23_copies = 3,
            m3_copies = 2,
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
            m23_copies = 6,
            m3_copies = 4,
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

function _build_fixture(field::CM.AbstractCoeffField, profile::String; threads::Bool, section::String="all")
    cfg = _profile_defaults(profile)
    cf(x) = CM.coerce(field, x)

    # Three vertical stripes; region labels form a chain of length 3.
    Ups = [PLB.BoxUpset([0.0, -10.0]), PLB.BoxUpset([1.0, -10.0])]
    Downs = PLB.BoxDownset[]
    P, _Henc, pi = PLB.encode_fringe_boxes(Ups, Downs, OPT.EncodingOptions())
    need_signed = section in ("signed", "all")
    cpi = need_signed ? EC.compile_encoding(pi, P; axes=EC.axes_from_encoding(pi), reps=EC.representatives(pi)) : nothing

    r2 = EC.locate(pi, [0.5, 0.0])
    r3 = EC.locate(pi, [2.0, 0.0])

    base23 = IR.pmodule_from_fringe(
        FF.one_by_one_fringe(P, FF.principal_upset(P, r2), FF.principal_downset(P, r3), cf(1); field=field)
    )
    base3 = IR.pmodule_from_fringe(
        FF.one_by_one_fringe(P, FF.principal_upset(P, r3), FF.principal_downset(P, r3), cf(1); field=field)
    )

    M = _repeat_direct_sum(vcat(fill(base23, cfg.m23_copies), fill(base3, cfg.m3_copies)))
    N = _repeat_direct_sum(vcat(fill(base3, cfg.m23_copies), fill(base23, cfg.m3_copies)))

    opts = OPT.InvariantOptions(box=([-1.0, -1.0], [2.0, 1.0]), threads=threads)
    dirs = [collect(Float64, d) for d in SI.default_directions(pi; n_dirs=cfg.n_dirs, max_den=cfg.max_den, include_axes=true, normalize=:L1)]
    offs = SI.default_offsets(pi, opts; n_offsets=cfg.n_offsets)

    need_slice = section in ("slice", "images", "all")
    need_fibered = section in ("fibered", "images", "all")

    slice_cache = need_slice ? SI.SlicePlanCache() : nothing
    plan_warm = need_slice ? SI.compile_slice_plan(pi, opts;
        directions=dirs,
        offsets=offs,
        normalize_dirs=:L1,
        cache=slice_cache,
        threads=threads,
    ) : nothing

    arr = need_fibered ? F2D.fibered_arrangement_2d(pi, opts; normalize_dirs=:L1, precompute=:cells, threads=threads) : nothing
    fam = need_fibered ? F2D.fibered_slice_family_2d(arr; direction_weight=:lesnick_l1, store_values=true) : nothing
    cacheM = need_fibered ? F2D.fibered_barcode_cache_2d(base23, arr; precompute=:full, threads=threads) : nothing
    cacheN = need_fibered ? F2D.fibered_barcode_cache_2d(base3,  arr; precompute=:full, threads=threads) : nothing

    tgrid = collect(range(-0.5, 5.5, length=cfg.tgrid_n))

    return (
        field = field,
        cfg = cfg,
        P = P,
        pi = pi,
        cpi = cpi,
        opts = opts,
        dirs = dirs,
        offs = offs,
        base23 = base23,
        base3 = base3,
        M = M,
        N = N,
        slice_cache = slice_cache,
        plan_warm = plan_warm,
        arr = arr,
        fam = fam,
        cacheM = cacheM,
        cacheN = cacheN,
        tgrid = tgrid,
        query_lo = [0.5, 0.0],
        query_hi = [2.0, 0.0],
    )
end

function main(args=ARGS)
    section = _parse_string_arg(args, "section", "all")
    field_name = _parse_string_arg(args, "field", "qq")
    profile = _parse_string_arg(args, "profile", "default")
    reps = _parse_int_arg(args, "reps", 5)
    threads = _parse_bool_arg(args, "threads", false)
    out = _parse_path_arg(args, "out", joinpath(@__DIR__, "_tmp_invariants_family_microbench.csv"))

    field = _field_from_name(field_name)
    fx = _build_fixture(field, profile; threads=threads, section=section)
    rows = NamedTuple[]

    if _section_enabled(section, "core")
        push!(rows, _bench("inv core rank_invariant", reps=reps) do
            INV.rank_invariant(fx.M, fx.opts; store_zeros=false)
        end)
        push!(rows, _bench("inv core rank_query encoded", reps=reps) do
            INV.rank_query(fx.M, fx.pi, fx.query_lo, fx.query_hi; opts=fx.opts)
        end)
        push!(rows, _bench("inv core restricted_hilbert global", reps=reps) do
            INV.restricted_hilbert(fx.M)
        end)
        push!(rows, _bench("inv core restricted_hilbert encoded", reps=reps) do
            INV.restricted_hilbert(fx.M, fx.pi, fx.query_lo, fx.opts)
        end)
        push!(rows, _bench("inv core hilbert_distance", reps=reps) do
            INV.hilbert_distance(fx.M, fx.N; norm=:L1)
        end)
        push!(rows, _bench("inv core integrated_hilbert_mass", reps=reps) do
            INV.integrated_hilbert_mass(fx.M, fx.pi, fx.opts)
        end)
        push!(rows, _bench("inv core module_size_summary", reps=reps) do
            INV.module_size_summary(fx.M, fx.pi, fx.opts)
        end)
        push!(rows, _bench("inv core support_measure_stats", reps=reps) do
            INV.support_measure_stats(fx.M, fx.pi, fx.opts)
        end)
    end

    if _section_enabled(section, "signed")
        push!(rows, _bench("inv signed euler_surface", reps=reps) do
            SM.euler_surface(fx.M, fx.pi, fx.opts)
        end)
        push!(rows, _bench("inv signed euler_signed_measure", reps=reps) do
            SM.euler_signed_measure(fx.M, fx.pi, fx.opts)
        end)
        push!(rows, _bench("inv signed rectangle_signed_barcode", reps=reps) do
            SM.rectangle_signed_barcode(fx.M, fx.cpi, fx.opts)
        end)
    end

    if _section_enabled(section, "slice")
        push!(rows, _bench("inv slice compile_plan cold", reps=reps) do
            local cache = SI.SlicePlanCache()
            SI.compile_slice_plan(fx.pi, fx.opts;
                directions=fx.dirs,
                offsets=fx.offs,
                normalize_dirs=:L1,
                cache=cache,
                threads=threads,
            )
        end)
        push!(rows, _bench("inv slice compile_plan warm", reps=reps) do
            SI.compile_slice_plan(fx.pi, fx.opts;
                directions=fx.dirs,
                offsets=fx.offs,
                normalize_dirs=:L1,
                cache=fx.slice_cache,
                threads=threads,
            )
        end)
        push!(rows, _bench("inv slice slice_barcodes via pi", reps=reps) do
            SI.slice_barcodes(fx.M, fx.pi, fx.opts;
                directions=fx.dirs,
                offsets=fx.offs,
                packed=true,
                threads=threads,
            )
        end)
        push!(rows, _bench("inv slice slice_barcodes via plan", reps=reps) do
            SI.slice_barcodes(fx.M, fx.plan_warm; packed=true, threads=threads)
        end)
        push!(rows, _bench("inv slice matching_distance_approx", reps=reps) do
            SI.matching_distance_approx(fx.M, fx.N, fx.pi, fx.opts;
                directions=fx.dirs,
                offsets=fx.offs,
                normalize_dirs=:L1,
                threads=threads,
            )
        end)
    end

    if _section_enabled(section, "fibered")
        push!(rows, _bench("inv fibered arrangement build", reps=reps) do
            F2D.fibered_arrangement_2d(fx.pi, fx.opts; normalize_dirs=:L1, precompute=:cells, threads=threads)
        end)
        push!(rows, _bench("inv fibered barcode_cache build", reps=reps) do
            local arr = F2D.fibered_arrangement_2d(fx.pi, fx.opts; normalize_dirs=:L1, precompute=:cells, threads=threads)
            F2D.fibered_barcode_cache_2d(fx.base23, arr; precompute=:full, threads=threads)
        end)
        push!(rows, _bench("inv fibered exact_distance direct", reps=reps) do
            F2D.matching_distance_exact_2d(fx.base23, fx.base3, fx.pi, fx.opts;
                weight=:lesnick_l1,
                normalize_dirs=:L1,
                threads=threads,
            )
        end)
        push!(rows, _bench("inv fibered exact_distance cached", reps=reps) do
            F2D.matching_distance_exact_2d(fx.cacheM, fx.cacheN;
                weight=:lesnick_l1,
                family=fx.fam,
                threads=threads,
            )
        end)
        push!(rows, _bench("inv fibered slice_barcodes cached", reps=reps) do
            F2D.slice_barcodes(fx.cacheM; dirs=[[1.0, 1.0]], offsets=[0.0], values=:t, threads=threads)
        end)
    end

    if _section_enabled(section, "images")
        push!(rows, _bench("inv images mpp_decomposition", reps=reps) do
            MPI.mpp_decomposition(fx.cacheM; N=fx.cfg.mpp_N, delta=:auto, q=1.0)
        end)
        push!(rows, _bench("inv images mpp_image", reps=reps) do
            MPI.mpp_image(fx.cacheM;
                resolution=fx.cfg.mpp_resolution,
                sigma=0.1,
                N=fx.cfg.mpp_N,
                delta=:auto,
                q=1.0,
            )
        end)
        push!(rows, _bench("inv images mp_landscape", reps=reps) do
            MPI.mp_landscape(fx.M, fx.pi, fx.opts;
                directions=fx.dirs,
                offsets=fx.offs,
                kmax=fx.cfg.kmax,
                tgrid=fx.tgrid,
                threads=threads,
                cache=fx.slice_cache,
            )
        end)
    end

    _write_csv(out, rows)
    println("wrote ", out)
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
