#!/usr/bin/env julia
# invariant_core_microbench.jl
#
# Purpose
# - Benchmark the shared internal plumbing in `InvariantCore`.
# - Keep routing between sections explicit so changes to option helpers,
#   rank-query kernels, and memo/cache helpers can be measured independently.
#
# Benchmark-to-owner mapping
# - `options` -> `src/invariant_core/options_helpers.jl`
#   - `SliceSpec ctor`, `orthant_directions`, `_normalize_dir`, projected
#     keyword helpers
# - `rank`    -> `src/invariant_core/rank_api.jl`
#   - direct `rank_map`, fringe-wrapper `rank_map`, encoded-point `rank_map`
# - `cache`   -> `src/invariant_core/rank_cache.jl`
#   - `_map_leq_cached` warm/cold behavior
#   - `RankQueryCache` locate warm/cold behavior
#   - `RankQueryCache` rank warm/cold behavior
#
# Timing policy
# - Warm-process microbenchmarking (`@timed`, median over reps)
# - Probe names make cold vs warm cache state explicit where applicable

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
const IC = TamerOp.InvariantCore
const FF = TamerOp.FiniteFringe
const IR = TamerOp.IndicatorResolutions
const MD = TamerOp.Modules
const EC = TamerOp.EncodingCore
const PLB = TamerOp.PLBackend
const FZ = TamerOp.FlangeZn
const ZE = TamerOp.ZnEncoding

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
        return (m23_copies = 3, m3_copies = 2)
    elseif profile == "larger"
        return (m23_copies = 6, m3_copies = 4)
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
    println(rpad(row.probe, 44),
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

function _build_fixture(field::CM.AbstractCoeffField, profile::String)
    cfg = _profile_defaults(profile)
    cf(x) = CM.coerce(field, x)

    # Small encoded poset/module fixture for rank-map benchmarks.
    Ups = [PLB.BoxUpset([0.0, -10.0]), PLB.BoxUpset([1.0, -10.0])]
    Downs = PLB.BoxDownset[]
    P, _Henc, pi = PLB.encode_fringe_boxes(Ups, Downs, OPT.EncodingOptions())

    r2 = EC.locate(pi, [0.5, 0.0])
    r3 = EC.locate(pi, [2.0, 0.0])

    H23 = FF.one_by_one_fringe(P, FF.principal_upset(P, r2), FF.principal_downset(P, r3), cf(1); field=field)
    H3  = FF.one_by_one_fringe(P, FF.principal_upset(P, r3), FF.principal_downset(P, r3), cf(1); field=field)
    base23 = IR.pmodule_from_fringe(H23)
    base3 = IR.pmodule_from_fringe(H3)
    M = _repeat_direct_sum(vcat(fill(base23, cfg.m23_copies), fill(base3, cfg.m3_copies)))
    H = FF.one_by_one_fringe(P, FF.principal_upset(P, r2), FF.principal_downset(P, r3), cf(1); field=field)
    cc = MD._get_cover_cache(M.Q)
    nQ = FF.nvertices(M.Q)
    array_memo = IC._new_array_memo(CM.coeff_type(field), nQ)
    dict_memo = Dict{Tuple{Int,Int},Matrix{CM.coeff_type(field)}}()
    opts = OPT.InvariantOptions(box=([-1.0, -1.0], [2.0, 1.0]), strict=true, threads=false)

    # Tiny ZnEncoding fixture for RankQueryCache.
    tau = FZ.face(2, Int[])
    fg = FZ.Flange{CM.coeff_type(field)}(
        2,
        [FZ.IndFlat(tau, (0, 0)); FZ.IndFlat(tau, (1, 0))],
        [FZ.IndInj(tau, (0, 1)); FZ.IndInj(tau, (1, 1))],
        [cf(1) cf(0); cf(0) cf(1)];
        field=field,
    )
    _Pz, zpi = ZE.encode_poset_from_flanges((fg,), OPT.EncodingOptions())
    rq_cache = IC.RankQueryCache(zpi)

    return (
        field = field,
        cfg = cfg,
        P = P,
        pi = pi,
        M = M,
        H = H,
        cc = cc,
        opts = opts,
        array_memo = array_memo,
        dict_memo = dict_memo,
        q_lo = [0.5, 0.0],
        q_hi = [2.0, 0.0],
        zpi = zpi,
        rq_cache = rq_cache,
        lattice_pt = (0, 0),
        lattice_pt2 = (1, 1),
    )
end

function main(args=ARGS)
    section = _parse_string_arg(args, "section", "all")
    field_name = _parse_string_arg(args, "field", "qq")
    profile = _parse_string_arg(args, "profile", "default")
    reps = _parse_int_arg(args, "reps", 5)
    out = _parse_path_arg(args, "out", joinpath(@__DIR__, "_tmp_invariant_core_microbench.csv"))

    field = _field_from_name(field_name)
    fx = _build_fixture(field, profile)
    rows = NamedTuple[]

    if _section_enabled(section, "options")
        push!(rows, _bench("ic options SliceSpec ctor", reps=reps) do
            IC.SliceSpec([1, 2, 3]; values=[0.0, 0.5, 1.0], weight=1.0)
        end)
        push!(rows, _bench("ic options orthant_directions", reps=reps) do
            IC.orthant_directions(2, [[1.0, 2.0], [2.0, 1.0], [1.0, 1.0]])
        end)
        push!(rows, _bench("ic options normalize_dir tuple", reps=reps) do
            IC._normalize_dir((2.0, 1.0), :L1)
        end)
        push!(rows, _bench("ic options normalize_dir vector", reps=reps) do
            IC._normalize_dir([2.0, 1.0, 3.0], :Linf)
        end)
        push!(rows, _bench("ic options selection kwargs", reps=reps) do
            IC._selection_kwargs_from_opts(fx.opts)
        end)
    end

    if _section_enabled(section, "rank")
        push!(rows, _bench("ic rank rank_map direct", reps=reps) do
            IC.rank_map(fx.M, 1, FF.nvertices(fx.M.Q))
        end)
        push!(rows, _bench("ic rank rank_map fringe", reps=reps) do
            IC.rank_map(fx.H, 1, FF.nvertices(fx.H.P))
        end)
        push!(rows, _bench("ic rank rank_map encoded", reps=reps) do
            IC.rank_map(fx.M, fx.pi, fx.q_lo, fx.q_hi, fx.opts)
        end)
        push!(rows, _bench("ic rank rank_map memo array", reps=reps,
                           setup=() -> fill!(fx.array_memo, nothing)) do
            IC.rank_map(fx.M, 1, FF.nvertices(fx.M.Q); cache=fx.cc, memo=fx.array_memo)
        end)
        push!(rows, _bench("ic rank rank_map memo dict", reps=reps,
                           setup=() -> empty!(fx.dict_memo)) do
            IC.rank_map(fx.M, 1, FF.nvertices(fx.M.Q); cache=fx.cc, memo=fx.dict_memo)
        end)
    end

    if _section_enabled(section, "cache")
        push!(rows, _bench("ic cache map_leq array cold", reps=reps,
                           setup=() -> fill!(fx.array_memo, nothing)) do
            IC._map_leq_cached(fx.M, 1, FF.nvertices(fx.M.Q), fx.cc, fx.array_memo)
        end)
        push!(rows, _bench("ic cache map_leq array warm", reps=reps) do
            IC._map_leq_cached(fx.M, 1, FF.nvertices(fx.M.Q), fx.cc, fx.array_memo)
        end)
        push!(rows, _bench("ic cache rankquery locate cold", reps=reps,
                           setup=() -> empty!(fx.rq_cache.loc_cache)) do
            IC._rank_query_locate!(fx.rq_cache, fx.lattice_pt)
        end)
        push!(rows, _bench("ic cache rankquery locate warm", reps=reps) do
            IC._rank_query_locate!(fx.rq_cache, fx.lattice_pt)
        end)
        push!(rows, _bench("ic cache rankquery rank cold", reps=reps,
                           setup=() -> begin
                               empty!(fx.rq_cache.rank_cache)
                               if fx.rq_cache.use_linear_rank_cache
                                   fill!(fx.rq_cache.rank_cache_filled, false)
                               end
                           end) do
            IC._rank_cache_get!(fx.rq_cache, 1, 1) do
                7
            end
        end)
        push!(rows, _bench("ic cache rankquery rank warm", reps=reps) do
            IC._rank_cache_get!(fx.rq_cache, 1, 1) do
                7
            end
        end)
    end

    _write_csv(out, rows)
    println("wrote ", out)
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
