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

function _parse_float_arg(args, key::String, default::Float64)
    prefix = key * "="
    for a in args
        startswith(a, prefix) || continue
        return parse(Float64, split(a, "=", limit=2)[2])
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

function _bench_ab(name::String, before::Function, after::Function; reps::Int, setup::Function=()->nothing)
    setup()
    b = _median_stats(before; reps=reps)
    setup()
    a = _median_stats(after; reps=reps)
    speedup = a.ms == 0.0 ? Inf : b.ms / a.ms
    alloc_ratio = b.kib == 0.0 ? Inf : a.kib / b.kib
    println(rpad(name, 36),
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

function _digest_sparse_family(A::AbstractArray)
    acc = 0
    @inbounds for M in A
        acc += size(M, 1) + size(M, 2) + nnz(M)
    end
    return acc
end

function _digest_hom_total(dims, dts)
    return sum(dims) + _digest_sparse_family(dts)
end

function _digest_hom_bicomplex(dims, dv, dh)
    return sum(dims) + _digest_sparse_family(dv) + _digest_sparse_family(dh)
end

function _digest_doublecomplex(DC)
    return sum(DC.dims) + _digest_sparse_family(DC.dv) + _digest_sparse_family(DC.dh)
end

function _digest_action(A)
    return size(A, 1) + size(A, 2) + mapreduce(x -> iszero(x) ? 0 : 1, +, A; init=0)
end

function _select_action_fixture(EA, Tsec, maxlen::Int)
    s = min(maxlen, max(0, length(Tsec.dims) - 1))
    best = (elem=DF.unit(EA), deg=0, coeff_nnz=0)
    for deg in reverse(0:min(maxlen, s))
        DF.dim(EA, deg) == 0 && continue
        elem = deg == 0 ? DF.unit(EA) : DF.basis(EA, deg)[1]
        alpha = reshape(DF.representative(EA.E, deg, elem.coords), :, 1)
        coeffs = DF.Functoriality._lift_cocycle_to_chainmap_coeff(EA.E.res, EA.E.res, EA.E, deg, alpha; upto=(s - deg))
        coeff = coeffs[(s - deg) + 1]
        best = (elem=elem, deg=deg, coeff_nnz=DF.Functoriality._coeff_nnz(coeff))
        best.coeff_nnz > 0 && break
    end
    return (element=best.elem, action_deg=s, ext_deg=best.deg, coeff_nnz=best.coeff_nnz)
end

function _fixture(; nx::Int=4, ny::Int=4, maxlen::Int=2, density::Float64=0.35)
    field = CM.QQField()
    P = _grid_finite_poset(nx, ny)
    Pop = FF.FinitePoset(transpose(FF.leq_matrix(P)); check=false)
    FF.build_cache!(P; cover=true, updown=true)
    FF.build_cache!(Pop; cover=true, updown=true)

    n = FF.nvertices(P)
    nups = max(12, min(24, 2n))
    ndowns = nups
    M = IR.pmodule_from_fringe(_random_fringe(P, field; nups=nups, ndowns=ndowns, density=density, seed=Int(0xD2A1)))
    N = IR.pmodule_from_fringe(_random_fringe(P, field; nups=nups, ndowns=ndowns, density=density, seed=Int(0xD2A2)))
    Rop = IR.pmodule_from_fringe(_random_fringe(Pop, field; nups=nups, ndowns=ndowns, density=density, seed=Int(0xD2A3)))
    L = IR.pmodule_from_fringe(_random_fringe(P, field; nups=nups, ndowns=ndowns, density=density, seed=Int(0xD2A4)))

    F, dF = IR.upset_resolution(M; maxlen=maxlen)
    E, dE = IR.downset_resolution(N; maxlen=maxlen)
    EA = DF.ExtAlgebra(L, OPT.DerivedFunctorOptions(maxdeg=maxlen))
    Tsec = DF.Tor(Rop, L, OPT.DerivedFunctorOptions(maxdeg=maxlen, model=:second); res=EA.E.res)
    action = _select_action_fixture(EA, Tsec, maxlen)

    return (field=field, M=M, N=N, Rop=Rop, L=L, F=F, dF=dF, E=E, dE=dE,
            EA=EA, Tsec=Tsec, action_deg=action.action_deg, action_ext_deg=action.ext_deg,
            action_coeff_nnz=action.coeff_nnz, action_el=action.element, maxlen=maxlen)
end

function main(; reps::Int=5,
              nx::Int=4,
              ny::Int=4,
              maxlen::Int=2,
              density::Float64=0.35,
              out::String=joinpath(@__DIR__, "_tmp_derived_functors_bicomplex_action_microbench.csv"))
    fx = _fixture(nx=nx, ny=ny, maxlen=maxlen, density=density)
    moderate_fx = _fixture(nx=max(nx + 1, 4), ny=max(ny + 1, 4),
                           maxlen=max(maxlen + 1, 2), density=max(density, 0.45))
    rows = NamedTuple[]

    old_ext_direct = DF.Algebras._EXT_ACTION_USE_DIRECT_STREAM[]
    old_coeff_nnz = DF.Functoriality._FUNCTORIALITY_DIRECT_COEFF_MIN_NNZ[]
    old_action_nnz = DF.Algebras._EXT_ACTION_DIRECT_STREAM_MIN_NNZ[]
    old_action_work = DF.Algebras._EXT_ACTION_DIRECT_STREAM_MIN_EST_WORK[]
    old_hom_trip = DF.HomExtEngine._HOM_ASSEMBLY_USE_TRIPLETS[]
    old_hom_offs = DF.HomExtEngine._HOM_ASSEMBLY_CACHE_OFFSETS[]
    old_tor_offs = DF.SpectralSequences._TOR_DOUBLE_COMPLEX_CACHE_TENSOR_OFFSETS[]
    old_tor_cache_fast = DF.SpectralSequences._TOR_DOUBLE_COMPLEX_CACHE_FASTPATH[]

    println("DerivedFunctors bicomplex/action microbenchmark")
    println("timing_policy=warm_process_median reps=", reps,
            " nx=", nx, " ny=", ny, " maxlen=", maxlen, " density=", density,
            " threads=", Threads.nthreads())
    println("ext_action.small ext_deg=", fx.action_ext_deg,
            " action_deg=", fx.action_deg, " coeff_nnz=", fx.action_coeff_nnz)
    println("ext_action.moderate ext_deg=", moderate_fx.action_ext_deg,
            " action_deg=", moderate_fx.action_deg, " coeff_nnz=", moderate_fx.action_coeff_nnz)

    try
        push!(rows, _bench_ab("ext_action_on_tor.small",
            () -> begin
                DF.Algebras._EXT_ACTION_USE_DIRECT_STREAM[] = false
                DF.Functoriality._FUNCTORIALITY_DIRECT_COEFF_MIN_NNZ[] = typemax(Int)
                DF.Algebras._EXT_ACTION_DIRECT_STREAM_MIN_NNZ[] = typemax(Int)
                DF.Algebras._EXT_ACTION_DIRECT_STREAM_MIN_EST_WORK[] = typemax(Int)
                _digest_action(DF.ext_action_on_tor(fx.EA, fx.Tsec, fx.action_el; s=fx.action_deg))
            end,
            () -> begin
                DF.Algebras._EXT_ACTION_USE_DIRECT_STREAM[] = true
                DF.Functoriality._FUNCTORIALITY_DIRECT_COEFF_MIN_NNZ[] = 0
                DF.Algebras._EXT_ACTION_DIRECT_STREAM_MIN_NNZ[] = 0
                DF.Algebras._EXT_ACTION_DIRECT_STREAM_MIN_EST_WORK[] = 0
                _digest_action(DF.ext_action_on_tor(fx.EA, fx.Tsec, fx.action_el; s=fx.action_deg))
            end; reps=reps))

        push!(rows, _bench_ab("ext_action_on_tor.moderate",
            () -> begin
                DF.Algebras._EXT_ACTION_USE_DIRECT_STREAM[] = false
                DF.Functoriality._FUNCTORIALITY_DIRECT_COEFF_MIN_NNZ[] = typemax(Int)
                DF.Algebras._EXT_ACTION_DIRECT_STREAM_MIN_NNZ[] = typemax(Int)
                DF.Algebras._EXT_ACTION_DIRECT_STREAM_MIN_EST_WORK[] = typemax(Int)
                _digest_action(DF.ext_action_on_tor(moderate_fx.EA, moderate_fx.Tsec,
                                                    moderate_fx.action_el; s=moderate_fx.action_deg))
            end,
            () -> begin
                DF.Algebras._EXT_ACTION_USE_DIRECT_STREAM[] = true
                DF.Functoriality._FUNCTORIALITY_DIRECT_COEFF_MIN_NNZ[] = 0
                DF.Algebras._EXT_ACTION_DIRECT_STREAM_MIN_NNZ[] = 0
                DF.Algebras._EXT_ACTION_DIRECT_STREAM_MIN_EST_WORK[] = 0
                _digest_action(DF.ext_action_on_tor(moderate_fx.EA, moderate_fx.Tsec,
                                                    moderate_fx.action_el; s=moderate_fx.action_deg))
            end; reps=reps))

        push!(rows, _bench_ab("build_hom_tot_complex",
            () -> begin
                DF.HomExtEngine._HOM_ASSEMBLY_USE_TRIPLETS[] = false
                DF.HomExtEngine._HOM_ASSEMBLY_CACHE_OFFSETS[] = false
                _digest_hom_total(DF.HomExtEngine.build_hom_tot_complex(fx.F, fx.dF, fx.E, fx.dE; threads=false)...)
            end,
            () -> begin
                DF.HomExtEngine._HOM_ASSEMBLY_USE_TRIPLETS[] = true
                DF.HomExtEngine._HOM_ASSEMBLY_CACHE_OFFSETS[] = true
                _digest_hom_total(DF.HomExtEngine.build_hom_tot_complex(fx.F, fx.dF, fx.E, fx.dE; threads=false)...)
            end; reps=reps))

        push!(rows, _bench_ab("build_hom_bicomplex_data",
            () -> begin
                DF.HomExtEngine._HOM_ASSEMBLY_USE_TRIPLETS[] = false
                DF.HomExtEngine._HOM_ASSEMBLY_CACHE_OFFSETS[] = false
                _digest_hom_bicomplex(DF.HomExtEngine.build_hom_bicomplex_data(fx.F, fx.dF, fx.E, fx.dE; threads=false)...)
            end,
            () -> begin
                DF.HomExtEngine._HOM_ASSEMBLY_USE_TRIPLETS[] = true
                DF.HomExtEngine._HOM_ASSEMBLY_CACHE_OFFSETS[] = true
                _digest_hom_bicomplex(DF.HomExtEngine.build_hom_bicomplex_data(fx.F, fx.dF, fx.E, fx.dE; threads=false)...)
            end; reps=reps))

        push!(rows, _bench_ab("build_hom_bicomplex_data.cache_hit",
            () -> begin
                rc = CM.ResolutionCache()
                _digest_hom_bicomplex(DF.HomExtEngine.build_hom_bicomplex_data(fx.F, fx.dF, fx.E, fx.dE;
                                                                                threads=false, cache=rc)...)
            end,
            () -> begin
                rc = CM.ResolutionCache()
                DF.HomExtEngine.build_hom_bicomplex_data(fx.F, fx.dF, fx.E, fx.dE; threads=false, cache=rc)
                _digest_hom_bicomplex(DF.HomExtEngine.build_hom_bicomplex_data(fx.F, fx.dF, fx.E, fx.dE;
                                                                                threads=false, cache=rc)...)
            end; reps=reps))

        push!(rows, _bench_ab("ExtDoubleComplex",
            () -> begin
                DF.HomExtEngine._HOM_ASSEMBLY_USE_TRIPLETS[] = false
                DF.HomExtEngine._HOM_ASSEMBLY_CACHE_OFFSETS[] = false
                _digest_doublecomplex(DF.ExtDoubleComplex(fx.M, fx.N; maxlen=fx.maxlen, threads=false))
            end,
            () -> begin
                DF.HomExtEngine._HOM_ASSEMBLY_USE_TRIPLETS[] = true
                DF.HomExtEngine._HOM_ASSEMBLY_CACHE_OFFSETS[] = true
                _digest_doublecomplex(DF.ExtDoubleComplex(fx.M, fx.N; maxlen=fx.maxlen, threads=false))
            end; reps=reps))

        push!(rows, _bench_ab("ExtDoubleComplex.cache_hit",
            () -> begin
                rc = CM.ResolutionCache()
                _digest_doublecomplex(DF.ExtDoubleComplex(fx.M, fx.N; maxlen=fx.maxlen, threads=false, cache=rc))
            end,
            () -> begin
                rc = CM.ResolutionCache()
                DF.ExtDoubleComplex(fx.M, fx.N; maxlen=fx.maxlen, threads=false, cache=rc)
                _digest_doublecomplex(DF.ExtDoubleComplex(fx.M, fx.N; maxlen=fx.maxlen, threads=false, cache=rc))
            end; reps=reps))

        push!(rows, _bench_ab("TorDoubleComplex",
            () -> begin
                DF.SpectralSequences._TOR_DOUBLE_COMPLEX_CACHE_TENSOR_OFFSETS[] = false
                _digest_doublecomplex(DF.TorDoubleComplex(fx.Rop, fx.L; maxlen=fx.maxlen, threads=false))
            end,
            () -> begin
                DF.SpectralSequences._TOR_DOUBLE_COMPLEX_CACHE_TENSOR_OFFSETS[] = true
                _digest_doublecomplex(DF.TorDoubleComplex(fx.Rop, fx.L; maxlen=fx.maxlen, threads=false))
            end; reps=reps))

        tordc_hit_cache = Ref{Any}(nothing)
        push!(rows, _bench_ab("TorDoubleComplex.cache_hit_call_only",
            () -> begin
                DF.SpectralSequences._TOR_DOUBLE_COMPLEX_CACHE_FASTPATH[] = false
                DF.TorDoubleComplex(fx.Rop, fx.L; maxlen=fx.maxlen, threads=false, cache=tordc_hit_cache[])
            end,
            () -> begin
                DF.SpectralSequences._TOR_DOUBLE_COMPLEX_CACHE_FASTPATH[] = true
                DF.TorDoubleComplex(fx.Rop, fx.L; maxlen=fx.maxlen, threads=false, cache=tordc_hit_cache[])
            end; reps=reps, setup=() -> begin
                rc = CM.ResolutionCache()
                DF.TorDoubleComplex(fx.Rop, fx.L; maxlen=fx.maxlen, threads=false, cache=rc)
                tordc_hit_cache[] = rc
                nothing
            end))

        push!(rows, _bench_ab("TorDoubleComplex.cache_hit",
            () -> begin
                DF.SpectralSequences._TOR_DOUBLE_COMPLEX_CACHE_FASTPATH[] = false
                _digest_doublecomplex(DF.TorDoubleComplex(fx.Rop, fx.L; maxlen=fx.maxlen, threads=false, cache=tordc_hit_cache[]))
            end,
            () -> begin
                DF.SpectralSequences._TOR_DOUBLE_COMPLEX_CACHE_FASTPATH[] = true
                _digest_doublecomplex(DF.TorDoubleComplex(fx.Rop, fx.L; maxlen=fx.maxlen, threads=false, cache=tordc_hit_cache[]))
            end; reps=reps, setup=() -> begin
                rc = CM.ResolutionCache()
                DF.TorDoubleComplex(fx.Rop, fx.L; maxlen=fx.maxlen, threads=false, cache=rc)
                tordc_hit_cache[] = rc
                nothing
            end))

        _write_csv(out, rows)
        println("wrote ", out)
        return rows
    finally
        DF.Algebras._EXT_ACTION_USE_DIRECT_STREAM[] = old_ext_direct
        DF.Functoriality._FUNCTORIALITY_DIRECT_COEFF_MIN_NNZ[] = old_coeff_nnz
        DF.Algebras._EXT_ACTION_DIRECT_STREAM_MIN_NNZ[] = old_action_nnz
        DF.Algebras._EXT_ACTION_DIRECT_STREAM_MIN_EST_WORK[] = old_action_work
        DF.HomExtEngine._HOM_ASSEMBLY_USE_TRIPLETS[] = old_hom_trip
        DF.HomExtEngine._HOM_ASSEMBLY_CACHE_OFFSETS[] = old_hom_offs
        DF.SpectralSequences._TOR_DOUBLE_COMPLEX_CACHE_TENSOR_OFFSETS[] = old_tor_offs
        DF.SpectralSequences._TOR_DOUBLE_COMPLEX_CACHE_FASTPATH[] = old_tor_cache_fast
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(reps=_parse_int_arg(ARGS, "--reps", 5),
         nx=_parse_int_arg(ARGS, "--nx", 4),
         ny=_parse_int_arg(ARGS, "--ny", 4),
         maxlen=_parse_int_arg(ARGS, "--maxlen", 2),
         density=_parse_float_arg(ARGS, "--density", 0.35),
         out=_parse_string_arg(ARGS, "--out",
                               joinpath(@__DIR__, "_tmp_derived_functors_bicomplex_action_microbench.csv")))
end
