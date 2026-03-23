#!/usr/bin/env julia
# slice_invariants_microbench.jl
#
# Purpose
# - Benchmark the `SliceInvariants` owner module directly.
# - Keep the benchmark sectional so planning, barcode extraction, distance
#   computation, and feature/kernels can be profiled independently.
#
# Benchmark-to-kernel mapping
# - `chain`    -> slice-chain construction, chain restriction, and 1D barcode-on-chain paths
# - `plan`     -> direction/offset defaults, slice collection, and slice-plan compilation/cache reuse
# - `barcode`  -> multi-slice barcode extraction through `pi` and precompiled plans
# - `distance` -> approximate slice-distance kernels/distances
# - `feature`  -> per-barcode feature/vectorization plus aggregated sliced feature/kernel surfaces
#
# Timing policy
# - Warm-process microbenchmarking (`@timed` median over reps)
# - Probe names encode cold vs warm/cache-aware behavior explicitly where relevant
#
# Usage
#   julia --project=. benchmark/slice_invariants_microbench.jl
#   julia --project=. benchmark/slice_invariants_microbench.jl --section=plan --field=f3
#   julia --project=. benchmark/slice_invariants_microbench.jl --profile=larger --reps=5

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)

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
const IR = TamerOp.IndicatorResolutions
const EC = TamerOp.EncodingCore
const PLB = TamerOp.PLBackend
const MD = TamerOp.Modules
const SI = TamerOp.SliceInvariants

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
            m23_copies = 3,
            m3_copies = 2,
            n_dirs = 5,
            n_offsets = 5,
            max_den = 4,
            tgrid_n = 31,
            img_n = 20,
            kmax = 2,
        )
    elseif profile == "larger"
        return (
            m23_copies = 6,
            m3_copies = 4,
            n_dirs = 9,
            n_offsets = 9,
            max_den = 6,
            tgrid_n = 61,
            img_n = 32,
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
    println(rpad(row.probe, 52),
            " median_time=", round(row.median_ms, digits=3), " ms",
            " median_alloc=", round(row.median_kib, digits=1), " KiB")
    return row
end

_bench(f::Function, name::AbstractString; reps::Int=7, setup::Union{Nothing,Function}=nothing) =
    _bench(name, f; reps=reps, setup=setup)

function _with_slice_chain_batch(f::Function, enabled::Bool, min_samples::Int)
    old_enabled = SI._SLICE_CHAIN_USE_BATCHED_LOCATE[]
    old_min = SI._SLICE_CHAIN_BATCHED_LOCATE_MIN_SAMPLES[]
    SI._SLICE_CHAIN_USE_BATCHED_LOCATE[] = enabled
    SI._SLICE_CHAIN_BATCHED_LOCATE_MIN_SAMPLES[] = min_samples
    try
        return f()
    finally
        SI._SLICE_CHAIN_USE_BATCHED_LOCATE[] = old_enabled
        SI._SLICE_CHAIN_BATCHED_LOCATE_MIN_SAMPLES[] = old_min
    end
end

function _with_landscape_feature_cache(f::Function, enabled::Bool)
    old = SI._SLICE_USE_LANDSCAPE_FEATURE_CACHE[]
    SI._SLICE_USE_LANDSCAPE_FEATURE_CACHE[] = enabled
    try
        return f()
    finally
        SI._SLICE_USE_LANDSCAPE_FEATURE_CACHE[] = old
    end
end

function _with_packed_distance_fastpath(f::Function, enabled::Bool)
    old = SI._SLICE_USE_PACKED_DISTANCE_FASTPATH[]
    SI._SLICE_USE_PACKED_DISTANCE_FASTPATH[] = enabled
    try
        return f()
    finally
        SI._SLICE_USE_PACKED_DISTANCE_FASTPATH[] = old
    end
end

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

function _slice_chain_baseline(pi, x0::AbstractVector, dir::AbstractVector, opts::OPT.InvariantOptions;
    ts=nothing,
    tmin=nothing,
    tmax=nothing,
    nsteps::Int=1001,
    box2=nothing,
    drop_unknown::Bool=true,
    dedup::Bool=true,
    check_chain::Bool=false,
)
    strict0 = opts.strict === nothing ? true : opts.strict
    clip = (opts.box !== nothing) || (box2 !== nothing)
    atol = 1e-12

    tkeep = ts
    if clip
        b1 = opts.box === nothing ? nothing : SI._resolve_box(pi, opts.box)
        b2 = box2 === nothing ? nothing : SI._resolve_box(pi, box2)
        bx = SI._intersect_axis_aligned_boxes(b1, b2)
        if ts === nothing
            tlo, thi = SI._line_param_range_in_box_nd(x0, dir, bx; atol=atol)
            tmin_eff = tmin === nothing ? tlo : max(float(tmin), tlo)
            tmax_eff = tmax === nothing ? thi : min(float(tmax), thi)
            if !(tmin_eff <= tmax_eff)
                return Int[], Float64[]
            end
            if nsteps <= 1
                tkeep = Float64[float(tmin_eff)]
            else
                tkeep = collect(range(float(tmin_eff), float(tmax_eff); length=nsteps))
            end
        else
            tlo, thi = SI._line_param_range_in_box_nd(x0, dir, bx; atol=atol)
            tkeep = Float64[]
            for traw in ts
                t = float(traw)
                ((tlo - atol) <= t <= (thi + atol)) && push!(tkeep, t)
            end
        end
    else
        if ts === nothing
            (tmin === nothing || tmax === nothing) &&
                error("slice_chain: must provide tmin/tmax unless clipping is active (opts.box or box2)")
            if nsteps <= 1
                tkeep = Float64[float(tmin)]
            else
                tkeep = collect(range(float(tmin), float(tmax); length=nsteps))
            end
        else
            tkeep = Float64[float(t) for t in ts]
        end
    end

    chain = Int[]
    tvals = Float64[]
    last_rid = typemin(Int)
    for t in tkeep
        x = x0 .+ t .* dir
        rid = SI.locate(pi, x)
        if rid == 0
            if strict0
                error("slice_chain: locate(pi, x) returned 0 (unknown region). Set opts.strict=false to allow unknown samples.")
            end
            drop_unknown && continue
        end
        if dedup && rid == last_rid
            continue
        end
        push!(chain, rid)
        push!(tvals, t)
        last_rid = rid
    end
    if check_chain && !isempty(tvals)
        SI._check_chain_monotone(pi, Float64[float(x) for x in x0], Float64[float(x) for x in dir], chain, tvals; strict=strict0)
    end
    return chain, tvals
end

function _build_fixture(field::CM.AbstractCoeffField, profile::String; threads::Bool)
    cfg = _profile_defaults(profile)
    cf(x) = CM.coerce(field, x)

    Ups = [PLB.BoxUpset([0.0, -10.0]), PLB.BoxUpset([1.0, -10.0])]
    Downs = PLB.BoxDownset[]
    P, _Henc, pi = PLB.encode_fringe_boxes(Ups, Downs, OPT.EncodingOptions())

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
    x0 = [0.0, 0.0]
    dir = [1.0, 1.0]
    chain, tvals = SI.slice_chain(pi, x0, dir, opts)
    bar_single = SI.slice_barcode(M, chain)

    dirs = [collect(Float64, d) for d in SI.default_directions(pi; n_dirs=cfg.n_dirs, max_den=cfg.max_den, include_axes=true, normalize=:L1)]
    offs = SI.default_offsets(pi, opts; n_offsets=cfg.n_offsets)
    plan_cache = SI.SlicePlanCache()
    plan = SI.compile_slice_plan(pi, opts;
        directions=dirs,
        offsets=offs,
        normalize_dirs=:L1,
        cache=plan_cache,
        threads=threads,
    )
    slices = SI.collect_slices(plan; values=:t)
    data_barcodes = SI.slice_barcodes(M, plan; packed=true, threads=threads)
    tgrid = collect(range(-0.5, 5.5, length=cfg.tgrid_n))
    xgrid = collect(range(0.0, 3.0, length=cfg.img_n))
    ygrid = collect(range(0.0, 3.0, length=cfg.img_n))

    return (
        field = field,
        cfg = cfg,
        P = P,
        pi = pi,
        opts = opts,
        M = M,
        N = N,
        x0 = x0,
        dir = dir,
        chain = chain,
        tvals = tvals,
        bar_single = bar_single,
        dirs = dirs,
        offs = offs,
        slices = slices,
        plan_cache = plan_cache,
        plan = plan,
        data_barcodes = data_barcodes,
        tgrid = tgrid,
        xgrid = xgrid,
        ygrid = ygrid,
    )
end

function main(args=ARGS)
    section = _parse_string_arg(args, "section", "all")
    field_name = _parse_string_arg(args, "field", "qq")
    profile = _parse_string_arg(args, "profile", "default")
    reps = _parse_int_arg(args, "reps", 5)
    threads = _parse_bool_arg(args, "threads", false)
    out = _parse_path_arg(args, "out", joinpath(@__DIR__, "_tmp_slice_invariants_microbench.csv"))

    field = _field_from_name(field_name)
    fx = _build_fixture(field, profile; threads=threads)
    rows = NamedTuple[]

    if _section_enabled(section, "chain")
        push!(rows, _bench("si chain slice_chain baseline", reps=reps) do
            _slice_chain_baseline(fx.pi, fx.x0, fx.dir, fx.opts)
        end)
        push!(rows, _bench("si chain slice_chain", reps=reps) do
            SI.slice_chain(fx.pi, fx.x0, fx.dir, fx.opts)
        end)
        push!(rows, _bench("si chain slice_chain no_batch_locate", reps=reps) do
            _with_slice_chain_batch(false, typemax(Int)) do
                SI.slice_chain(fx.pi, fx.x0, fx.dir, fx.opts)
            end
        end)
        push!(rows, _bench("si chain restrict_to_chain", reps=reps) do
            SI.restrict_to_chain(fx.M, fx.chain)
        end)
        push!(rows, _bench("si chain slice_barcode direct", reps=reps) do
            SI.slice_barcode(fx.M, fx.chain)
        end)
        push!(rows, _bench("si chain slice_barcode geometric baseline", reps=reps) do
            chain, tvals = _slice_chain_baseline(fx.pi, fx.x0, fx.dir, fx.opts)
            SI.slice_barcode(fx.M, chain; values=tvals, check_chain=false)
        end)
        push!(rows, _bench("si chain slice_barcode geometric", reps=reps) do
            SI.slice_barcode(fx.M, fx.pi, fx.x0, fx.dir, fx.opts)
        end)
    end

    if _section_enabled(section, "plan")
        push!(rows, _bench("si plan default_directions", reps=reps) do
            SI.default_directions(fx.pi; n_dirs=fx.cfg.n_dirs, max_den=fx.cfg.max_den, include_axes=true, normalize=:L1)
        end)
        push!(rows, _bench("si plan default_offsets", reps=reps) do
            SI.default_offsets(fx.pi, fx.opts; n_offsets=fx.cfg.n_offsets)
        end)
        push!(rows, _bench("si plan collect_slices", reps=reps) do
            SI.collect_slices(fx.plan; values=:t)
        end)
        push!(rows, _bench("si plan compile_slice_plan cold", reps=reps) do
            local cache = SI.SlicePlanCache()
            SI.compile_slice_plan(fx.pi, fx.opts;
                directions=fx.dirs,
                offsets=fx.offs,
                normalize_dirs=:L1,
                cache=cache,
                threads=threads,
            )
        end)
        push!(rows, _bench("si plan compile_slice_plan warm", reps=reps) do
            SI.compile_slice_plan(fx.pi, fx.opts;
                directions=fx.dirs,
                offsets=fx.offs,
                normalize_dirs=:L1,
                cache=fx.plan_cache,
                threads=threads,
            )
        end)
        push!(rows, _bench("si plan compile_slice_plan cold no_batch_locate", reps=reps) do
            _with_slice_chain_batch(false, typemax(Int)) do
                local cache = SI.SlicePlanCache()
                SI.compile_slice_plan(fx.pi, fx.opts;
                    directions=fx.dirs,
                    offsets=fx.offs,
                    normalize_dirs=:L1,
                    cache=cache,
                    threads=threads,
                )
            end
        end)
    end

    if _section_enabled(section, "barcode")
        push!(rows, _bench("si barcode slice_barcodes explicit", reps=reps) do
            SI.slice_barcodes(fx.M, fx.slices; packed=true, threads=threads)
        end)
        push!(rows, _bench("si barcode slice_barcodes via pi", reps=reps) do
            SI.slice_barcodes(fx.M, fx.pi;
                opts=fx.opts,
                directions=fx.dirs,
                offsets=fx.offs,
                packed=true,
                threads=threads,
            )
        end)
        push!(rows, _bench("si barcode slice_barcodes via pi no_batch_locate", reps=reps) do
            _with_slice_chain_batch(false, typemax(Int)) do
                SI.slice_barcodes(fx.M, fx.pi;
                    opts=fx.opts,
                    directions=fx.dirs,
                    offsets=fx.offs,
                    packed=true,
                    threads=threads,
                )
            end
        end)
        push!(rows, _bench("si barcode slice_barcodes via plan uncached", reps=reps,
                           setup=SI.clear_slice_module_cache!) do
            SI.slice_barcodes(fx.M, fx.plan; packed=true, threads=threads)
        end)
        push!(rows, _bench("si barcode slice_barcodes via plan", reps=reps) do
            SI.slice_barcodes(fx.M, fx.plan; packed=true, threads=threads)
        end)
    end

    if _section_enabled(section, "distance")
        push!(rows, _bench("si distance matching_distance_approx", reps=reps) do
            SI.matching_distance_approx(fx.M, fx.N, fx.pi, fx.opts;
                directions=fx.dirs,
                offsets=fx.offs,
                normalize_dirs=:L1,
                threads=threads,
            )
        end)
        push!(rows, _bench("si distance matching_wasserstein_approx", reps=reps) do
            SI.matching_wasserstein_distance_approx(fx.M, fx.N, fx.pi, fx.opts;
                directions=fx.dirs,
                offsets=fx.offs,
                normalize_dirs=:L1,
                threads=threads,
            )
        end)
        push!(rows, _bench("si distance matching_distance_approx no_packed_fastpath", reps=reps) do
            _with_packed_distance_fastpath(false) do
                SI.matching_distance_approx(fx.M, fx.N, fx.pi, fx.opts;
                    directions=fx.dirs,
                    offsets=fx.offs,
                    normalize_dirs=:L1,
                    threads=threads,
                )
            end
        end)
        push!(rows, _bench("si distance matching_wasserstein_approx no_packed_fastpath", reps=reps) do
            _with_packed_distance_fastpath(false) do
                SI.matching_wasserstein_distance_approx(fx.M, fx.N, fx.pi, fx.opts;
                    directions=fx.dirs,
                    offsets=fx.offs,
                    normalize_dirs=:L1,
                    threads=threads,
                )
            end
        end)
        push!(rows, _bench("si distance matching_distance_approx no_batch_locate", reps=reps) do
            _with_slice_chain_batch(false, typemax(Int)) do
                SI.matching_distance_approx(fx.M, fx.N, fx.pi, fx.opts;
                    directions=fx.dirs,
                    offsets=fx.offs,
                    normalize_dirs=:L1,
                    threads=threads,
                )
            end
        end)
        push!(rows, _bench("si distance matching_wasserstein_approx no_batch_locate", reps=reps) do
            _with_slice_chain_batch(false, typemax(Int)) do
                SI.matching_wasserstein_distance_approx(fx.M, fx.N, fx.pi, fx.opts;
                    directions=fx.dirs,
                    offsets=fx.offs,
                    normalize_dirs=:L1,
                    threads=threads,
                )
            end
        end)
        push!(rows, _bench("si distance sliced_bottleneck_distance uncached", reps=reps,
                           setup=SI.clear_slice_module_cache!) do
            SI.sliced_bottleneck_distance(fx.M, fx.N, fx.pi, fx.opts;
                directions=fx.dirs,
                offsets=fx.offs,
                normalize_dirs=:L1,
                threads=threads,
                cache=fx.plan_cache,
            )
        end)
        push!(rows, _bench("si distance sliced_bottleneck_distance", reps=reps) do
            SI.sliced_bottleneck_distance(fx.M, fx.N, fx.pi, fx.opts;
                directions=fx.dirs,
                offsets=fx.offs,
                normalize_dirs=:L1,
                threads=threads,
                cache=fx.plan_cache,
            )
        end)
        push!(rows, _bench("si distance sliced_wasserstein_distance", reps=reps) do
            SI.sliced_wasserstein_distance(fx.M, fx.N, fx.pi, fx.opts;
                directions=fx.dirs,
                offsets=fx.offs,
                normalize_dirs=:L1,
                threads=threads,
                cache=fx.plan_cache,
            )
        end)
    end

    if _section_enabled(section, "feature")
        push!(rows, _bench("si feature persistence_landscape", reps=reps) do
            SI.persistence_landscape(fx.bar_single; kmax=fx.cfg.kmax, tgrid=fx.tgrid)
        end)
        push!(rows, _bench("si feature persistence_image", reps=reps) do
            SI.persistence_image(fx.bar_single;
                xgrid=fx.xgrid,
                ygrid=fx.ygrid,
                sigma=0.15,
                coords=:birth_persistence,
                weighting=:persistence,
            )
        end)
        push!(rows, _bench("si feature barcode_summary", reps=reps) do
            SI.barcode_summary(fx.bar_single)
        end)
        push!(rows, _bench("si feature slice_features summary uncached", reps=reps,
                           setup=SI.clear_slice_module_cache!) do
            SI.slice_features(fx.M, fx.plan;
                featurizer=:summary,
                aggregate=:mean,
                threads=threads,
            )
        end)
        push!(rows, _bench("si feature slice_features summary", reps=reps) do
            SI.slice_features(fx.M, fx.plan;
                featurizer=:summary,
                aggregate=:mean,
                threads=threads,
            )
        end)
        push!(rows, _bench("si feature slice_features landscape", reps=reps) do
            SI.slice_features(fx.M, fx.plan;
                featurizer=:landscape,
                aggregate=:mean,
                kmax=fx.cfg.kmax,
                tgrid=fx.tgrid,
                threads=threads,
            )
        end)
        push!(rows, _bench("si feature slice_features landscape uncached", reps=reps,
                           setup=SI.clear_slice_module_cache!) do
            SI.slice_features(fx.M, fx.plan;
                featurizer=:landscape,
                aggregate=:mean,
                kmax=fx.cfg.kmax,
                tgrid=fx.tgrid,
                threads=threads,
            )
        end)
        push!(rows, _bench("si feature slice_features landscape no_landscape_cache", reps=reps,
                           setup=SI.clear_slice_module_cache!) do
            _with_landscape_feature_cache(false) do
                SI.slice_features(fx.M, fx.plan;
                    featurizer=:landscape,
                    aggregate=:mean,
                    kmax=fx.cfg.kmax,
                    tgrid=fx.tgrid,
                    threads=threads,
                )
            end
        end)
        push!(rows, _bench("si feature slice_kernel", reps=reps) do
            SI.slice_kernel(fx.M, fx.N, fx.plan;
                kind=:landscape_linear,
                tgrid=fx.tgrid,
                kmax=fx.cfg.kmax,
                threads=threads,
            )
        end)
        push!(rows, _bench("si feature slice_kernel uncached", reps=reps,
                           setup=SI.clear_slice_module_cache!) do
            SI.slice_kernel(fx.M, fx.N, fx.plan;
                kind=:landscape_linear,
                tgrid=fx.tgrid,
                kmax=fx.cfg.kmax,
                threads=threads,
            )
        end)
        push!(rows, _bench("si feature slice_kernel no_landscape_cache", reps=reps,
                           setup=SI.clear_slice_module_cache!) do
            _with_landscape_feature_cache(false) do
                SI.slice_kernel(fx.M, fx.N, fx.plan;
                    kind=:landscape_linear,
                    tgrid=fx.tgrid,
                    kmax=fx.cfg.kmax,
                    threads=threads,
                )
            end
        end)
    end

    isempty(rows) && error("no benchmark rows selected; section='$section' produced nothing")
    _write_csv(out, rows)
    println("wrote ", out)
end

main()
