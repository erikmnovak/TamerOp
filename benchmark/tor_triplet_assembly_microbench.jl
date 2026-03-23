#!/usr/bin/env julia

using Random
using SparseArrays

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
    prefix = key * "="
    for arg in args
        startswith(arg, prefix) || continue
        return parse(Int, split(arg, "=", limit=2)[2])
    end
    return default
end

function _parse_float_arg(args, key::String, default::Float64)
    prefix = key * "="
    for arg in args
        startswith(arg, prefix) || continue
        return parse(Float64, split(arg, "=", limit=2)[2])
    end
    return default
end

function _parse_string_arg(args, key::String, default::String)
    prefix = key * "="
    for arg in args
        startswith(arg, prefix) || continue
        return lowercase(strip(split(arg, "=", limit=2)[2]))
    end
    return default
end

function _median_stats(f::Function; reps::Int)
    GC.gc()
    f()
    GC.gc()
    times = Vector{Float64}(undef, reps)
    bytes = Vector{Int}(undef, reps)
    for i in 1:reps
        stats = @timed f()
        times[i] = 1000.0 * stats.time
        bytes[i] = stats.bytes
    end
    sort!(times)
    sort!(bytes)
    mid = cld(reps, 2)
    return (ms=times[mid], kib=bytes[mid] / 1024.0)
end

function _grid_finite_poset(nx::Int, ny::Int)
    n = nx * ny
    rel = falses(n, n)
    @inline idx(ix, iy) = (iy - 1) * nx + ix
    @inbounds for y1 in 1:ny, x1 in 1:nx
        i = idx(x1, y1)
        for y2 in y1:ny, x2 in x1:nx
            j = idx(x2, y2)
            rel[i, j] = true
        end
    end
    return FF.FinitePoset(rel; check=false)
end

function _rand_coeff(rng::AbstractRNG, field::CM.AbstractCoeffField)
    v = rand(rng, -3:3)
    v == 0 && (v = 1)
    return CM.coerce(field, v)
end

function _random_fringe(Q::FF.AbstractPoset, field::CM.AbstractCoeffField;
                        nups::Int=24, ndowns::Int=24, density::Float64=0.35, seed::Int=0xB1B1)
    rng = MersenneTwister(seed)
    K = CM.coeff_type(field)
    n = FF.nvertices(Q)

    U = Vector{FF.Upset}(undef, nups)
    D = Vector{FF.Downset}(undef, ndowns)
    up_verts = Vector{Int}(undef, nups)
    down_verts = Vector{Int}(undef, ndowns)
    @inbounds for i in 1:nups
        up_verts[i] = rand(rng, 1:n)
        U[i] = FF.principal_upset(Q, up_verts[i])
    end
    @inbounds for j in 1:ndowns
        down_verts[j] = rand(rng, 1:n)
        D[j] = FF.principal_downset(Q, down_verts[j])
    end

    phi = zeros(K, ndowns, nups)
    @inbounds for j in 1:ndowns, i in 1:nups
        FF.leq(Q, up_verts[i], down_verts[j]) || continue
        rand(rng) < density || continue
        phi[j, i] = _rand_coeff(rng, field)
    end
    return FF.FringeModule{K}(Q, U, D, phi; field=field)
end

function _fixture(field::CM.AbstractCoeffField; nx::Int, ny::Int, maxlen::Int, density::Float64)
    P = _grid_finite_poset(nx, ny)
    Pop = FF.FinitePoset(transpose(FF.leq_matrix(P)); check=false)
    HM = _random_fringe(P, field; nups=max(16, div(nx * ny, 2)), ndowns=max(16, div(nx * ny, 2)),
                        density=density, seed=0xA12C + nx + 17 * ny + 131 * maxlen)
    HL = _random_fringe(P, field; nups=max(16, div(nx * ny, 2)), ndowns=max(16, div(nx * ny, 2)),
                        density=density, seed=0xA12D + nx + 19 * ny + 137 * maxlen)
    HRop = _random_fringe(Pop, field; nups=max(16, div(nx * ny, 2)), ndowns=max(16, div(nx * ny, 2)),
                          density=density, seed=0xA12E + nx + 23 * ny + 139 * maxlen)
    M = IR.pmodule_from_fringe(HM)
    L = IR.pmodule_from_fringe(HL)
    Rop = IR.pmodule_from_fringe(HRop)
    opts = OPT.ResolutionOptions(maxlen=maxlen)
    resRop = DF.projective_resolution(Rop, opts; threads=false)
    resL = DF.projective_resolution(L, opts; threads=false)
    df_first = OPT.DerivedFunctorOptions(maxdeg=maxlen, model=:first)
    df_second = OPT.DerivedFunctorOptions(maxdeg=maxlen, model=:second)
    return (; Rop, L, resRop, resL, df_first, df_second, maxlen)
end

_digest_tor(T) = sum(DF.dim(T, s) for s in DF.degree_range(T))
_digest_dc(DC) = sum(DC.dims) + sum(nnz, DC.dv) + sum(nnz, DC.dh)

function _bench_ab(name::String, f_before::Function, f_after::Function; reps::Int)
    before = _median_stats(f_before; reps=reps)
    after = _median_stats(f_after; reps=reps)
    speedup = before.ms / after.ms
    alloc_ratio = before.kib / max(after.kib, 1e-9)
    println(rpad(name, 42),
            " before=", round(before.ms, digits=3), " ms / ", round(before.kib, digits=1), " KiB",
            "  after=", round(after.ms, digits=3), " ms / ", round(after.kib, digits=1), " KiB",
            "  speedup=", round(speedup, digits=2), "x",
            "  alloc_ratio=", round(alloc_ratio, digits=2), "x")
    return (probe=name, before_ms=before.ms, after_ms=after.ms, before_kib=before.kib, after_kib=after.kib)
end

function _write_csv(path::AbstractString, rows)
    open(path, "w") do io
        println(io, "probe,before_ms,after_ms,before_kib,after_kib")
        for r in rows
            println(io, string(r.probe, ",", r.before_ms, ",", r.after_ms, ",", r.before_kib, ",", r.after_kib))
        end
    end
end

function _run_case!(rows, label::String, fx; reps::Int)
    old_trip = DF.ExtTorSpaces._TOR_USE_TRIPLET_BOUNDARY_ASSEMBLY[]
    old_direct = DF.ExtTorSpaces._TOR_USE_DIRECT_MAP_TRIPLET_ASSEMBLY[]
    try
        DF.ExtTorSpaces._TOR_USE_TRIPLET_BOUNDARY_ASSEMBLY[] = true

        push!(rows, _bench_ab(label * ".Tor first with_res",
            () -> begin
                DF.ExtTorSpaces._TOR_USE_DIRECT_MAP_TRIPLET_ASSEMBLY[] = false
                _digest_tor(DF.Tor(fx.Rop, fx.L, fx.df_first; res=fx.resRop))
            end,
            () -> begin
                DF.ExtTorSpaces._TOR_USE_DIRECT_MAP_TRIPLET_ASSEMBLY[] = true
                _digest_tor(DF.Tor(fx.Rop, fx.L, fx.df_first; res=fx.resRop))
            end;
            reps=reps))

        push!(rows, _bench_ab(label * ".Tor second with_res",
            () -> begin
                DF.ExtTorSpaces._TOR_USE_DIRECT_MAP_TRIPLET_ASSEMBLY[] = false
                _digest_tor(DF.Tor(fx.Rop, fx.L, fx.df_second; res=fx.resL))
            end,
            () -> begin
                DF.ExtTorSpaces._TOR_USE_DIRECT_MAP_TRIPLET_ASSEMBLY[] = true
                _digest_tor(DF.Tor(fx.Rop, fx.L, fx.df_second; res=fx.resL))
            end;
            reps=reps))

        push!(rows, _bench_ab(label * ".TorDoubleComplex",
            () -> begin
                DF.ExtTorSpaces._TOR_USE_DIRECT_MAP_TRIPLET_ASSEMBLY[] = false
                _digest_dc(DF.TorDoubleComplex(fx.Rop, fx.L; maxlen=fx.maxlen, threads=false))
            end,
            () -> begin
                DF.ExtTorSpaces._TOR_USE_DIRECT_MAP_TRIPLET_ASSEMBLY[] = true
                _digest_dc(DF.TorDoubleComplex(fx.Rop, fx.L; maxlen=fx.maxlen, threads=false))
            end;
            reps=reps))
    finally
        DF.ExtTorSpaces._TOR_USE_TRIPLET_BOUNDARY_ASSEMBLY[] = old_trip
        DF.ExtTorSpaces._TOR_USE_DIRECT_MAP_TRIPLET_ASSEMBLY[] = old_direct
    end
    return rows
end

function main(args=ARGS)
    reps = _parse_int_arg(args, "--reps", 3)
    density = _parse_float_arg(args, "--density", 0.35)
    section = _parse_string_arg(args, "--section", "all")
    out = _parse_string_arg(args, "--out", joinpath(@__DIR__, "_tmp_tor_triplet_assembly_microbench.csv"))
    field = CM.QQField()
    rows = NamedTuple[]

    println("Tor triplet assembly microbenchmark")
    println("timing_policy=warm_process_median")
    println("section=", section, " reps=", reps, " density=", density, " threads=", Base.Threads.nthreads())

    if section in ("all", "small")
        fx_small = _fixture(field; nx=3, ny=3, maxlen=2, density=density)
        _run_case!(rows, "small_3x3x2", fx_small; reps=reps)
    end
    if section in ("all", "medium")
        fx_medium = _fixture(field; nx=4, ny=4, maxlen=3, density=density)
        _run_case!(rows, "medium_4x4x3", fx_medium; reps=reps)
    end

    _write_csv(out, rows)
    println("wrote ", out)
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
