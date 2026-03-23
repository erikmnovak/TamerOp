#!/usr/bin/env julia
#
# batch_featurization_microbench.jl
#
# Purpose
# - Micro-benchmark dataset-level batch featurization throughput in `Workflow`.
# - Compare execution backends and cache policies on a deterministic fixture:
#   - backend: :serial vs :threads (and :folds when extension is present)
#   - cache: nothing vs :auto (session cache reuse)
#   - unsupported handling: :error-equivalent (all supported), :skip, :missing
#
# Scope
# - This is a focused throughput benchmark for `featurize`/`batch_transform`.
# - It is not a full-library benchmark.
#
# Usage
#   julia --project=. benchmark/batch_featurization_microbench.jl
#   julia --project=. benchmark/batch_featurization_microbench.jl --reps=12 --nsamples=128

using LinearAlgebra

try
    using TamerOp
catch
    include(joinpath(@__DIR__, "..", "src", "TamerOp.jl"))
    using .TamerOp
end

const TO = TamerOp.Advanced
const CM = TO.CoreModules
const FF = TO.FiniteFringe
const IR = TO.IndicatorResolutions
const PLB = TO.PLBackend

function _fixture(nsamples::Int)
    nsamples > 0 || error("nsamples must be > 0")
    field = CM.QQField()
    opts_enc = TO.EncodingOptions(field=field)
    Ups = [PLB.BoxUpset([0.0, -10.0]), PLB.BoxUpset([1.0, -10.0])]
    Downs = PLB.BoxDownset[]
    P, _, pi = PLB.encode_fringe_boxes(Ups, Downs, opts_enc)

    r2 = TO.locate(pi, [0.5, 0.0])
    r3 = TO.locate(pi, [2.0, 0.0])
    H23 = FF.one_by_one_fringe(P, FF.principal_upset(P, r2), FF.principal_downset(P, r3); field=field)
    H3 = FF.one_by_one_fringe(P, FF.principal_upset(P, r3), FF.principal_downset(P, r3); field=field)
    M23 = IR.pmodule_from_fringe(H23)
    M3 = IR.pmodule_from_fringe(H3)
    enc23 = TO.EncodingResult(P, M23, pi)
    enc3 = TO.EncodingResult(P, M3, pi)

    base = Any[enc23, enc3]
    samples = Any[base[mod1(i, 2)] for i in 1:nsamples]
    mixed = Any[s for s in samples]
    mixed[cld(nsamples, 2)] = M23

    opts_inv = TO.InvariantOptions(box=([-1.0, -1.0], [2.0, 1.0]), strict=false)
    lspec = TO.LandscapeSpec(
        directions=[[1.0, 1.0]],
        offsets=[[0.0, 0.0]],
        tgrid=collect(0.0:0.5:3.0),
        kmax=2,
        strict=false,
    )

    return (samples=samples, mixed=mixed, opts=opts_inv, spec=lspec)
end

function _bench(name::AbstractString, nsamples::Int, f::Function; reps::Int=10)
    GC.gc()
    f() # warmup
    GC.gc()

    times_ms = Vector{Float64}(undef, reps)
    bytes = Vector{Int}(undef, reps)
    for i in 1:reps
        m = @timed f()
        times_ms[i] = 1000.0 * m.time
        bytes[i] = m.bytes
    end
    sort!(times_ms)
    sort!(bytes)
    med_ms = times_ms[cld(reps, 2)]
    med_kib = bytes[cld(reps, 2)] / 1024.0
    throughput = nsamples / (med_ms / 1000.0)
    println(
        rpad(name, 44),
        " median_time=", round(med_ms, digits=3), " ms",
        " median_alloc=", round(med_kib, digits=1), " KiB",
        " throughput=", round(throughput, digits=1), " samples/s",
    )
    return (ms=med_ms, kib=med_kib, throughput=throughput)
end

function _parse_arg(args, key::String, default::Int)
    for a in args
        startswith(a, key * "=") || continue
        return max(1, parse(Int, split(a, "=", limit=2)[2]))
    end
    return default
end

function main(; reps::Int=10, nsamples::Int=128)
    fx = _fixture(nsamples)
    samples = fx.samples
    mixed = fx.mixed
    spec = fx.spec
    opts = fx.opts

    bserial = TO.BatchOptions(threaded=false, backend=:serial, deterministic=true)
    bthreads = TO.BatchOptions(threaded=true, backend=:threads, deterministic=true, chunk_size=0)
    bthreads_small = TO.BatchOptions(threaded=true, backend=:threads, deterministic=true, chunk_size=1)
    bfolds = TO.BatchOptions(threaded=true, backend=:folds, deterministic=true)

    println("Batch featurization micro-benchmark")
    println("reps=$(reps), nsamples=$(nsamples), nthreads=$(Threads.nthreads())\n")

    _bench("serial cache=nothing", nsamples,
           () -> TO.featurize(samples, spec; opts=opts, batch=bserial, cache=nothing);
           reps=reps)
    _bench("serial cache=:auto", nsamples,
           () -> TO.featurize(samples, spec; opts=opts, batch=bserial, cache=:auto);
           reps=reps)
    _bench("threads cache=:auto chunk=auto", nsamples,
           () -> TO.featurize(samples, spec; opts=opts, batch=bthreads, cache=:auto);
           reps=reps)
    _bench("threads cache=:auto chunk=1", nsamples,
           () -> TO.featurize(samples, spec; opts=opts, batch=bthreads_small, cache=:auto);
           reps=reps)

    if Base.find_package("Folds") !== nothing
        _bench("folds cache=:auto", nsamples,
               () -> TO.featurize(samples, spec; opts=opts, batch=bfolds, cache=:auto);
               reps=reps)
    else
        println(lpad("folds cache=:auto", 44), " skipped (Folds not available)")
    end

    _bench("serial mixed on_unsupported=:skip", nsamples,
           () -> TO.featurize(mixed, spec; opts=opts, batch=bserial, cache=:auto, on_unsupported=:skip);
           reps=reps)
    _bench("serial mixed on_unsupported=:missing", nsamples,
           () -> TO.featurize(mixed, spec; opts=opts, batch=bserial, cache=:auto, on_unsupported=:missing);
           reps=reps)
end

if abspath(PROGRAM_FILE) == @__FILE__
    reps = _parse_arg(ARGS, "--reps", 10)
    nsamples = _parse_arg(ARGS, "--nsamples", 128)
    main(; reps=reps, nsamples=nsamples)
end
