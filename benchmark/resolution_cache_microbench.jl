#!/usr/bin/env julia

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
const DF = TamerOp.DerivedFunctors

function _parse_int_arg(args, key::String, default::Int)
    for a in args
        startswith(a, key * "=") || continue
        return max(1, parse(Int, split(a, "=", limit=2)[2]))
    end
    return default
end

function _parse_path_arg(args, key::String, default::String)
    for a in args
        startswith(a, key * "=") || continue
        return String(strip(split(a, "=", limit=2)[2]))
    end
    return default
end

function _bench(name::AbstractString, f::Function; reps::Int=11, setup::Union{Nothing,Function}=nothing)
    GC.gc()
    f()  # warmup
    GC.gc()
    times_ms = Vector{Float64}(undef, reps)
    bytes = Vector{Int}(undef, reps)
    for i in 1:reps
        setup === nothing || setup()
        m = @timed f()
        times_ms[i] = 1000.0 * m.time
        bytes[i] = m.bytes
    end
    sort!(times_ms)
    sort!(bytes)
    mid = cld(reps, 2)
    return (probe=String(name), median_ms=times_ms[mid], median_kib=bytes[mid] / 1024.0)
end

function _write_csv(path::AbstractString, rows)
    open(path, "w") do io
        println(io, "probe,median_ms,median_kib")
        for r in rows
            println(io, string(r.probe, ",", r.median_ms, ",", r.median_kib))
        end
    end
end

function _with_ref!(f::Function, ref, val)
    old = ref[]
    ref[] = val
    try
        return f()
    finally
        ref[] = old
    end
end

function _chain_poset(n::Int)
    rel = falses(n, n)
    @inbounds for i in 1:n, j in i:n
        rel[i, j] = true
    end
    return FF.FinitePoset(rel; check=false)
end

function _fixture()
    field = CM.QQField()
    K = CM.coeff_type(field)
    P = _chain_poset(3)
    FF.build_cache!(P; cover=true, updown=true)
    U = FF.principal_upset(P, 2)
    D = FF.principal_downset(P, 2)
    H = FF.one_by_one_fringe(P, U, D, one(K); field=field)
    M = IR.pmodule_from_fringe(H)
    opts = OPT.ResolutionOptions(maxlen=2)
    Rproj = DF.projective_resolution(M, opts; threads=false)
    Rinj = DF.injective_resolution(M, opts; threads=false)
    return (pmodule=M, maxlen=opts.maxlen, projective=Rproj, injective=Rinj)
end

function _bench_family(rows, label::AbstractString, enabled_ref, key, payload, store!, get!, payload_type;
                       reps::Int, iters::Int)
    cache = CM.ResolutionCache()

    push!(rows, _with_ref!(enabled_ref, false) do
        _bench("$(label) store_fresh baseline", () -> begin
            for _ in 1:iters
                local_rc = CM.ResolutionCache()
                store!(local_rc, key, payload)
            end
            nothing
        end; reps=reps)
    end)
    push!(rows, _with_ref!(enabled_ref, true) do
        _bench("$(label) store_fresh candidate", () -> begin
            for _ in 1:iters
                local_rc = CM.ResolutionCache()
                store!(local_rc, key, payload)
            end
            nothing
        end; reps=reps)
    end)

    push!(rows, _with_ref!(enabled_ref, false) do
        _bench("$(label) store_after_clear baseline", () -> begin
            for _ in 1:iters
                CM._clear_resolution_cache!(cache)
                store!(cache, key, payload)
            end
            nothing
        end; reps=reps)
    end)
    push!(rows, _with_ref!(enabled_ref, true) do
        _bench("$(label) store_after_clear candidate", () -> begin
            for _ in 1:iters
                CM._clear_resolution_cache!(cache)
                store!(cache, key, payload)
            end
            nothing
        end; reps=reps)
    end)

    push!(rows, _with_ref!(enabled_ref, false) do
        local_rc = CM.ResolutionCache()
        store!(local_rc, key, payload)
        get!(local_rc, key, payload_type)
        get!(local_rc, key, payload_type)
        _bench("$(label) lookup_hit baseline", () -> begin
            for _ in 1:iters
                v = get!(local_rc, key, payload_type)
                v === nothing && error("unexpected cache miss in baseline lookup")
            end
            nothing
        end; reps=reps)
    end)
    push!(rows, _with_ref!(enabled_ref, true) do
        local_rc = CM.ResolutionCache()
        store!(local_rc, key, payload)
        get!(local_rc, key, payload_type)
        get!(local_rc, key, payload_type)
        _bench("$(label) lookup_hit candidate", () -> begin
            for _ in 1:iters
                v = get!(local_rc, key, payload_type)
                v === nothing && error("unexpected cache miss in candidate lookup")
            end
            nothing
        end; reps=reps)
    end)

    return rows
end

function _print_comparison(rows)
    index = Dict{String,NamedTuple}()
    for row in rows
        index[row.probe] = row
    end
    println("Resolution cache micro-benchmark")
    for label in ("projective", "injective")
        for probe in ("store_fresh", "store_after_clear", "lookup_hit")
            base = index["$label $probe baseline"]
            cand = index["$label $probe candidate"]
            ratio = base.median_ms / max(cand.median_ms, eps())
            alloc_text = cand.median_kib == 0.0 ?
                (base.median_kib == 0.0 ? "alloc_parity" : "alloc_drop_to_zero") :
                string(round(base.median_kib / cand.median_kib, digits=2), "x")
            println(rpad("$label $probe", 32),
                    " baseline=", round(base.median_ms, digits=4), " ms",
                    " candidate=", round(cand.median_ms, digits=4), " ms",
                    " speedup=", round(ratio, digits=2), "x",
                    " alloc=", alloc_text)
        end
    end
end

function main(; reps::Int=11, iters::Int=10_000,
              out::String=joinpath(@__DIR__, "_tmp_resolution_cache_microbench.csv"))
    fx = _fixture()
    rows = NamedTuple[]

    proj_key = CM._resolution_key2(fx.pmodule, fx.maxlen)
    inj_key = CM._resolution_key2((fx.pmodule, :injective), fx.maxlen)

    _bench_family(
        rows,
        "projective",
        DF.Resolutions.PROJECTIVE_PRIMARY_CACHE_ENABLED,
        proj_key,
        fx.projective,
        DF.Resolutions._cache_projective_store!,
        DF.Resolutions._cache_projective_get,
        typeof(fx.projective);
        reps=reps,
        iters=iters,
    )

    _bench_family(
        rows,
        "injective",
        DF.Resolutions.INJECTIVE_PRIMARY_CACHE_ENABLED,
        inj_key,
        fx.injective,
        DF.Resolutions._cache_injective_store!,
        DF.Resolutions._cache_injective_get,
        typeof(fx.injective);
        reps=reps,
        iters=iters,
    )

    _print_comparison(rows)
    _write_csv(out, rows)
    println("Wrote ", out)
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    reps = _parse_int_arg(ARGS, "--reps", 11)
    iters = _parse_int_arg(ARGS, "--iters", 10_000)
    out = _parse_path_arg(ARGS, "--out", joinpath(@__DIR__, "_tmp_resolution_cache_microbench.csv"))
    main(; reps=reps, iters=iters, out=out)
end
