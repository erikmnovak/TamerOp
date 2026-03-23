#!/usr/bin/env julia

using Random
using SparseArrays

if Base.find_package("TamerOp") === nothing
    include(joinpath(@__DIR__, "..", "src", "TamerOp.jl"))
    using .TamerOp
else
    using TamerOp
end

const CM = TamerOp.CoreModules
const FF = TamerOp.FiniteFringe
const IR = TamerOp.IndicatorResolutions
const DF = TamerOp.DerivedFunctors
const OPT = TamerOp.Options

function _parse_int_arg(args, key::String, default::Int)
    prefix = key * "="
    for a in args
        startswith(a, prefix) || continue
        return parse(Int, split(a, "=", limit=2)[2])
    end
    return default
end

function _parse_string_arg(args, key::String, default::String)
    prefix = key * "="
    for a in args
        startswith(a, prefix) || continue
        return String(split(a, "=", limit=2)[2])
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

function _bench_ab(name::String, before::Function, after::Function; reps::Int)
    b = _median_stats(before; reps=reps)
    a = _median_stats(after; reps=reps)
    speedup = a.ms == 0.0 ? Inf : b.ms / a.ms
    alloc_ratio = b.kib == 0.0 ? Inf : a.kib / b.kib
    println(rpad(name, 34),
            " before=", round(b.ms, digits=3), " ms / ", round(b.kib, digits=1), " KiB",
            "  after=", round(a.ms, digits=3), " ms / ", round(a.kib, digits=1), " KiB",
            "  speedup=", round(speedup, digits=2), "x",
            "  alloc_ratio=", round(alloc_ratio, digits=2), "x")
    return (probe=name, before_ms=b.ms, before_kib=b.kib,
            after_ms=a.ms, after_kib=a.kib, speedup=speedup, alloc_ratio=alloc_ratio)
end

function _write_csv(path::AbstractString, rows)
    open(path, "w") do io
        println(io, "probe,before_ms,before_kib,after_ms,after_kib,speedup,alloc_ratio")
        for r in rows
            println(io, string(r.probe, ",", r.before_ms, ",", r.before_kib, ",",
                               r.after_ms, ",", r.after_kib, ",", r.speedup, ",", r.alloc_ratio))
        end
    end
end

function _grid_finite_poset(nx::Int, ny::Int)
    n = nx * ny
    rel = falses(n, n)
    @inline idx(ix, iy) = (iy - 1) * nx + ix
    @inbounds for y1 in 1:ny, x1 in 1:nx
        i = idx(x1, y1)
        for y2 in y1:ny, x2 in x1:nx
            rel[i, idx(x2, y2)] = true
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
                        nups::Int, ndowns::Int, density::Float64, seed::Int)
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

function _select_action_fixture(EA, Tsec, maxlen::Int)
    s = min(maxlen, max(0, length(Tsec.dims) - 1))
    chosen = (element=DF.unit(EA), ext_deg=0, coeff_nnz=0)
    for deg in reverse(0:min(maxlen, s))
        DF.dim(EA, deg) == 0 && continue
        elem = deg == 0 ? DF.unit(EA) : DF.basis(EA, deg)[1]
        alpha = reshape(DF.representative(EA.E, deg, elem.coords), :, 1)
        coeffs = DF.Functoriality._lift_cocycle_to_chainmap_coeff(EA.E.res, EA.E.res, EA.E, deg, alpha; upto=(s - deg))
        coeff = coeffs[(s - deg) + 1]
        chosen = (element=elem, ext_deg=deg, coeff_nnz=DF.Functoriality._coeff_nnz(coeff))
        chosen.coeff_nnz > 0 && break
    end
    return (element=chosen.element, ext_deg=chosen.ext_deg, action_deg=s, coeff_nnz=chosen.coeff_nnz)
end

function _fixture(; nx::Int, ny::Int, maxlen::Int, density::Float64, seed_base::Int)
    field = CM.QQField()
    P = _grid_finite_poset(nx, ny)
    Pop = FF.FinitePoset(transpose(FF.leq_matrix(P)); check=false)
    FF.build_cache!(P; cover=true, updown=true)
    FF.build_cache!(Pop; cover=true, updown=true)
    n = FF.nvertices(P)
    nups = max(12, min(24, 2n))
    ndowns = nups
    Rop = IR.pmodule_from_fringe(_random_fringe(Pop, field; nups=nups, ndowns=ndowns, density=density, seed=seed_base + 1))
    L = IR.pmodule_from_fringe(_random_fringe(P, field; nups=nups, ndowns=ndowns, density=density, seed=seed_base + 2))
    EA = DF.ExtAlgebra(L, OPT.DerivedFunctorOptions(maxdeg=maxlen))
    Tsec = DF.Tor(Rop, L, OPT.DerivedFunctorOptions(maxdeg=maxlen, model=:second); res=EA.E.res)
    action = _select_action_fixture(EA, Tsec, maxlen)
    coeff = DF.Functoriality._lift_cocycle_to_chainmap_coeff(
        EA.E.res, EA.E.res, EA.E, action.ext_deg,
        reshape(DF.representative(EA.E, action.ext_deg, action.element.coords), :, 1);
        upto=(action.action_deg - action.ext_deg)
    )[(action.action_deg - action.ext_deg) + 1]
    dom_bases = Tsec.resL.gens[action.action_deg + 1]
    cod_bases = Tsec.resL.gens[(action.action_deg - action.ext_deg) + 1]
    plan = DF.Functoriality._tensor_coeff_plan(Tsec.Rop, dom_bases, cod_bases, coeff)
    est_work = DF.Algebras._ext_action_direct_stream_work(
        plan,
        Tsec.offsets[(action.action_deg - action.ext_deg) + 1],
        Tsec.offsets[action.action_deg + 1],
        size(Tsec.homol[action.action_deg + 1].Hrep, 2),
    )
    return (label="nx=$(nx),ny=$(ny),maxlen=$(maxlen),density=$(density)",
            EA=EA, Tsec=Tsec, action=action, est_work=est_work)
end

function _digest_action(A)
    nnzcount = 0
    @inbounds for x in A
        iszero(x) || (nnzcount += 1)
    end
    return size(A, 1) + size(A, 2) + nnzcount
end

function main(; reps::Int=3,
              out::String=joinpath(@__DIR__, "_tmp_ext_action_on_tor_microbench.csv"))
    small = _fixture(nx=3, ny=3, maxlen=1, density=0.35, seed_base=Int(0xE100))
    moderate = _fixture(nx=4, ny=4, maxlen=2, density=0.45, seed_base=Int(0xE200))
    rows = NamedTuple[]

    old_direct = DF.Algebras._EXT_ACTION_USE_DIRECT_STREAM[]
    old_coeff = DF.Functoriality._FUNCTORIALITY_DIRECT_COEFF_MIN_NNZ[]
    old_action = DF.Algebras._EXT_ACTION_DIRECT_STREAM_MIN_NNZ[]
    old_work = DF.Algebras._EXT_ACTION_DIRECT_STREAM_MIN_EST_WORK[]
    old_product = DF.Algebras._EXT_ACTION_DIRECT_STREAM_USE_PRODUCT_WORK[]

    println("ext_action_on_tor microbenchmark")
    println("timing_policy=warm_process_median reps=", reps, " threads=", Threads.nthreads())
    println("small   ", small.label, " ext_deg=", small.action.ext_deg,
            " action_deg=", small.action.action_deg, " coeff_nnz=", small.action.coeff_nnz,
            " est_work=", small.est_work)
    println("moderate ", moderate.label, " ext_deg=", moderate.action.ext_deg,
            " action_deg=", moderate.action.action_deg, " coeff_nnz=", moderate.action.coeff_nnz,
            " est_work=", moderate.est_work)

    try
        for (name, fx) in (("small", small), ("moderate", moderate))
            push!(rows, _bench_ab("ext_action." * name * ".off_vs_on",
                () -> begin
                    DF.Algebras._EXT_ACTION_USE_DIRECT_STREAM[] = false
                    DF.Functoriality._FUNCTORIALITY_DIRECT_COEFF_MIN_NNZ[] = typemax(Int)
                    DF.Algebras._EXT_ACTION_DIRECT_STREAM_MIN_NNZ[] = typemax(Int)
                    DF.Algebras._EXT_ACTION_DIRECT_STREAM_MIN_EST_WORK[] = typemax(Int)
                    _digest_action(DF.ext_action_on_tor(fx.EA, fx.Tsec, fx.action.element; s=fx.action.action_deg))
                end,
                () -> begin
                    DF.Algebras._EXT_ACTION_USE_DIRECT_STREAM[] = true
                    DF.Functoriality._FUNCTORIALITY_DIRECT_COEFF_MIN_NNZ[] = 0
                    DF.Algebras._EXT_ACTION_DIRECT_STREAM_MIN_NNZ[] = 0
                    DF.Algebras._EXT_ACTION_DIRECT_STREAM_MIN_EST_WORK[] = 0
                    _digest_action(DF.ext_action_on_tor(fx.EA, fx.Tsec, fx.action.element; s=fx.action.action_deg))
                end; reps=reps))

            push!(rows, _bench_ab("ext_action." * name * ".nnz_only_vs_workgate",
                () -> begin
                    DF.Algebras._EXT_ACTION_USE_DIRECT_STREAM[] = true
                    DF.Functoriality._FUNCTORIALITY_DIRECT_COEFF_MIN_NNZ[] = 0
                    DF.Algebras._EXT_ACTION_DIRECT_STREAM_MIN_NNZ[] = 24
                    DF.Algebras._EXT_ACTION_DIRECT_STREAM_MIN_EST_WORK[] = 0
                    DF.Algebras._EXT_ACTION_DIRECT_STREAM_USE_PRODUCT_WORK[] = false
                    _digest_action(DF.ext_action_on_tor(fx.EA, fx.Tsec, fx.action.element; s=fx.action.action_deg))
                end,
                () -> begin
                    DF.Algebras._EXT_ACTION_USE_DIRECT_STREAM[] = true
                    DF.Functoriality._FUNCTORIALITY_DIRECT_COEFF_MIN_NNZ[] = 0
                    DF.Algebras._EXT_ACTION_DIRECT_STREAM_MIN_NNZ[] = 24
                    DF.Algebras._EXT_ACTION_DIRECT_STREAM_MIN_EST_WORK[] = old_work
                    DF.Algebras._EXT_ACTION_DIRECT_STREAM_USE_PRODUCT_WORK[] = true
                    _digest_action(DF.ext_action_on_tor(fx.EA, fx.Tsec, fx.action.element; s=fx.action.action_deg))
                end; reps=reps))
        end
        _write_csv(out, rows)
        println("wrote ", out)
        return rows
    finally
        DF.Algebras._EXT_ACTION_USE_DIRECT_STREAM[] = old_direct
        DF.Functoriality._FUNCTORIALITY_DIRECT_COEFF_MIN_NNZ[] = old_coeff
        DF.Algebras._EXT_ACTION_DIRECT_STREAM_MIN_NNZ[] = old_action
        DF.Algebras._EXT_ACTION_DIRECT_STREAM_MIN_EST_WORK[] = old_work
        DF.Algebras._EXT_ACTION_DIRECT_STREAM_USE_PRODUCT_WORK[] = old_product
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(reps=_parse_int_arg(ARGS, "--reps", 3),
         out=_parse_string_arg(ARGS, "--out",
                               joinpath(@__DIR__, "_tmp_ext_action_on_tor_microbench.csv")))
end
