#!/usr/bin/env julia

include(joinpath(@__DIR__, "derived_functors_bicomplex_action_microbench.jl"))

function main(; reps::Int=5,
              nx::Int=3,
              ny::Int=3,
              maxlen::Int=1,
              density::Float64=0.35,
              out::String=joinpath(@__DIR__, "_tmp_tor_doublecomplex_cache_microbench.csv"))
    fx = _fixture(nx=nx, ny=ny, maxlen=maxlen, density=density)
    rows = NamedTuple[]
    cache_ref = Ref{Any}(nothing)
    old_fast = DF.SpectralSequences._TOR_DOUBLE_COMPLEX_CACHE_FASTPATH[]

    println("TorDoubleComplex cache microbenchmark")
    println("timing_policy=warm_process_median reps=", reps,
            " nx=", nx, " ny=", ny, " maxlen=", maxlen, " density=", density,
            " threads=", Threads.nthreads())

    try
        push!(rows, _bench_ab("TorDoubleComplex.cache_hit_call_only",
            () -> begin
                DF.SpectralSequences._TOR_DOUBLE_COMPLEX_CACHE_FASTPATH[] = false
                DF.TorDoubleComplex(fx.Rop, fx.L; maxlen=fx.maxlen, threads=false, cache=cache_ref[])
            end,
            () -> begin
                DF.SpectralSequences._TOR_DOUBLE_COMPLEX_CACHE_FASTPATH[] = true
                DF.TorDoubleComplex(fx.Rop, fx.L; maxlen=fx.maxlen, threads=false, cache=cache_ref[])
            end; reps=reps, setup=() -> begin
                rc = CM.ResolutionCache()
                DF.TorDoubleComplex(fx.Rop, fx.L; maxlen=fx.maxlen, threads=false, cache=rc)
                cache_ref[] = rc
                nothing
            end))

        push!(rows, _bench_ab("TorDoubleComplex.cache_hit_digest",
            () -> begin
                DF.SpectralSequences._TOR_DOUBLE_COMPLEX_CACHE_FASTPATH[] = false
                _digest_doublecomplex(DF.TorDoubleComplex(fx.Rop, fx.L; maxlen=fx.maxlen, threads=false, cache=cache_ref[]))
            end,
            () -> begin
                DF.SpectralSequences._TOR_DOUBLE_COMPLEX_CACHE_FASTPATH[] = true
                _digest_doublecomplex(DF.TorDoubleComplex(fx.Rop, fx.L; maxlen=fx.maxlen, threads=false, cache=cache_ref[]))
            end; reps=reps, setup=() -> begin
                rc = CM.ResolutionCache()
                DF.TorDoubleComplex(fx.Rop, fx.L; maxlen=fx.maxlen, threads=false, cache=rc)
                cache_ref[] = rc
                nothing
            end))

        _write_csv(out, rows)
        println("wrote ", out)
        return rows
    finally
        DF.SpectralSequences._TOR_DOUBLE_COMPLEX_CACHE_FASTPATH[] = old_fast
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(reps=_parse_int_arg(ARGS, "--reps", 5),
         nx=_parse_int_arg(ARGS, "--nx", 3),
         ny=_parse_int_arg(ARGS, "--ny", 3),
         maxlen=_parse_int_arg(ARGS, "--maxlen", 1),
         density=_parse_float_arg(ARGS, "--density", 0.35),
         out=_parse_string_arg(ARGS, "--out",
                               joinpath(@__DIR__, "_tmp_tor_doublecomplex_cache_microbench.csv")))
end
