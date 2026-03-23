#!/usr/bin/env julia
#
# featurizer_cache_microbench.jl
#
# Purpose
# - Micro-benchmark cache-aware featurization paths in `Workflow`.
# - Quantify time/allocation gains from:
#   1) `transform(spec, sample)` vs `transform(spec, build_cache(sample, spec))`
#   2) `featurize(samples, spec; cache=nothing)` vs `cache=:auto`
#
# Scope
# - Uses a small deterministic in-repo fixture:
#   - two modules (`enc23`, `enc3`) on one encoding map
#   - one `CompositeSpec` (landscape + persistence image)
#   - one `CompositeSpec` of two `MPLandscapeSpec`s sharing a slice plan
#   - one `CompositeSpec` of two `MPPImageSpec`s sharing a decomposition
#   - one `ProjectedDistancesSpec` against a reference bank
# - Reports mean wall-time and mean allocation from repeated runs.
# - Not a full-library benchmark; this isolates cache protocol effectiveness.
#
# Usage
#   julia --project=. benchmark/featurizer_cache_microbench.jl
#   julia --project=. benchmark/featurizer_cache_microbench.jl --reps=15

using LinearAlgebra

try
    using TamerOp
catch
    include(joinpath(@__DIR__, "..", "src", "TamerOp.jl"))
    using .TamerOp
end

const TO = TamerOp.Advanced
const CM = TO.CoreModules
const OPT = TO.Options
const DT = TO.DataTypes
const EC = TO.EncodingCore
const RES = TO.Results
const FF = TO.FiniteFringe
const IR = TO.IndicatorResolutions
const PLB = TO.PLBackend

function _fixture()
    field = CM.QQField()
    opts_enc = OPT.EncodingOptions(field=field)
    Ups = [PLB.BoxUpset([0.0, -10.0]), PLB.BoxUpset([1.0, -10.0])]
    Downs = PLB.BoxDownset[]
    P, _, pi = PLB.encode_fringe_boxes(Ups, Downs, opts_enc)

    r2 = EC.locate(pi, [0.5, 0.0])
    r3 = EC.locate(pi, [2.0, 0.0])

    H23 = FF.one_by_one_fringe(P, FF.principal_upset(P, r2), FF.principal_downset(P, r3), CM.QQ(1); field=field)
    H3 = FF.one_by_one_fringe(P, FF.principal_upset(P, r3), FF.principal_downset(P, r3), CM.QQ(1); field=field)
    M23 = IR.pmodule_from_fringe(H23)
    M3 = IR.pmodule_from_fringe(H3)

    enc23 = RES.EncodingResult(P, M23, pi)
    enc3 = RES.EncodingResult(P, M3, pi)
    opts_inv = OPT.InvariantOptions(box=([-1.0, -1.0], [2.0, 1.0]), strict=false)

    lspec = TO.LandscapeSpec(
        directions=[[1.0, 1.0]],
        offsets=[[0.0, 0.0]],
        tgrid=collect(0.0:0.5:3.0),
        kmax=2,
        strict=false,
    )
    pispec = TO.PersistenceImageSpec(
        directions=[[1.0, 1.0]],
        offsets=[[0.0, 0.0]],
        xgrid=collect(0.0:0.5:2.0),
        ygrid=collect(0.0:0.5:2.0),
        sigma=0.35,
        strict=false,
    )
    cspec = TO.CompositeSpec((lspec, pispec), namespacing=true)

    mlspec = TO.MPLandscapeSpec(
        directions=[[1.0, 1.0]],
        offsets=[[0.0, 0.0]],
        tgrid=collect(0.0:0.5:3.0),
        kmax=2,
        strict=false,
        direction_weight=:uniform,
    )
    mlspec2 = TO.MPLandscapeSpec(
        directions=[[1.0, 1.0]],
        offsets=[[0.0, 0.0]],
        tgrid=collect(0.0:0.5:3.0),
        kmax=1,
        strict=false,
        direction_weight=:uniform,
    )
    mlpair = TO.CompositeSpec((mlspec, mlspec2), namespacing=true)

    mppspec = TO.MPPImageSpec(
        resolution=6,
        sigma=0.15,
        N=8,
        q=1.0,
        segment_prune=true,
    )
    mppspec2 = TO.MPPImageSpec(
        resolution=5,
        sigma=0.2,
        N=8,
        q=1.0,
        segment_prune=false,
    )
    mpppair = TO.CompositeSpec((mppspec, mppspec2), namespacing=true)

    dspec = TO.ProjectedDistancesSpec(
        [enc3];
        reference_names=[:M3],
        n_dirs=4,
        normalize=:L1,
        precompute=true,
    )

    return (
        enc23=enc23,
        enc3=enc3,
        opts_inv=opts_inv,
        lspec=lspec,
        pispec=pispec,
        cspec=cspec,
        mlspec=mlspec,
        mlpair=mlpair,
        mppspec=mppspec,
        mpppair=mpppair,
        dspec=dspec,
    )
end

function _bench(name::AbstractString, f::Function; reps::Int=15)
    GC.gc()
    f() # warmup
    GC.gc()
    total_t = 0.0
    total_bytes = 0
    for _ in 1:reps
        m = @timed f()
        total_t += m.time
        total_bytes += m.bytes
    end
    mean_ms = 1_000.0 * total_t / reps
    mean_kib = total_bytes / reps / 1024.0
    println(rpad(name, 46), "  mean_time=", round(mean_ms, digits=3), " ms  mean_alloc=", round(mean_kib, digits=2), " KiB")
    return (mean_ms=mean_ms, mean_kib=mean_kib)
end

function main(; reps::Int=15)
    fx = _fixture()
    bserial = TO.BatchOptions(threaded=false, backend=:serial, deterministic=true)

    raw_comp = () -> TO.transform(fx.cspec, fx.enc23; opts=fx.opts_inv, threaded=false)
    comp_cache = TO.build_cache(fx.enc23, fx.cspec; opts=fx.opts_inv, threaded=false)
    cached_comp = () -> TO.transform(fx.cspec, comp_cache; opts=fx.opts_inv, threaded=false)

    raw_mlpair = () -> TO.transform(fx.mlpair, fx.enc23; opts=fx.opts_inv, threaded=false)
    mlpair_cache = TO.build_cache(fx.enc23, fx.mlpair; opts=fx.opts_inv, threaded=false)
    cached_mlpair = () -> TO.transform(fx.mlpair, mlpair_cache; opts=fx.opts_inv, threaded=false)

    raw_mpppair = () -> TO.transform(fx.mpppair, fx.enc23; opts=fx.opts_inv, threaded=false)
    mpppair_cache = TO.build_cache(fx.enc23, fx.mpppair; opts=fx.opts_inv, threaded=false, level=:fibered)
    cached_mpppair = () -> TO.transform(fx.mpppair, mpppair_cache; opts=fx.opts_inv, threaded=false)

    raw_proj = () -> TO.transform(fx.dspec, fx.enc23; opts=fx.opts_inv, threaded=false)
    proj_cache = TO.build_cache(fx.enc23, fx.dspec; opts=fx.opts_inv, threaded=false)
    cached_proj = () -> TO.transform(fx.dspec, proj_cache; opts=fx.opts_inv, threaded=false)

    samples = Any[fx.enc23, fx.enc3, fx.enc23, fx.enc3]
    raw_feat = () -> TO.featurize(samples, fx.cspec; opts=fx.opts_inv, batch=bserial, cache=nothing)
    cached_feat = () -> TO.featurize(samples, fx.cspec; opts=fx.opts_inv, batch=bserial, cache=:auto)

    println("Featurizer cache micro-benchmark (reps=", reps, ")")
    println("Layout: mean time/alloc over repeated runs with warmup.\n")

    r1 = _bench("transform CompositeSpec (raw)", raw_comp; reps=reps)
    c1 = _bench("transform CompositeSpec (cache)", cached_comp; reps=reps)
    rml = _bench("transform MPL composite (raw)", raw_mlpair; reps=reps)
    cml = _bench("transform MPL composite (cache)", cached_mlpair; reps=reps)
    rmpp = _bench("transform MPP composite (raw)", raw_mpppair; reps=reps)
    cmpp = _bench("transform MPP composite (cache)", cached_mpppair; reps=reps)
    r2 = _bench("transform ProjectedDistancesSpec (raw)", raw_proj; reps=reps)
    c2 = _bench("transform ProjectedDistancesSpec (cache)", cached_proj; reps=reps)
    r3 = _bench("featurize batch (cache=nothing)", raw_feat; reps=max(5, div(reps, 2)))
    c3 = _bench("featurize batch (cache=:auto)", cached_feat; reps=max(5, div(reps, 2)))

    println("\nRelative speedups (raw/cache):")
    println("CompositeSpec transform:        ", round(r1.mean_ms / c1.mean_ms, digits=3), "x")
    println("MPL composite transform:        ", round(rml.mean_ms / cml.mean_ms, digits=3), "x")
    println("MPP composite transform:        ", round(rmpp.mean_ms / cmpp.mean_ms, digits=3), "x")
    println("ProjectedDistances transform:   ", round(r2.mean_ms / c2.mean_ms, digits=3), "x")
    println("Batch featurize:                ", round(r3.mean_ms / c3.mean_ms, digits=3), "x")
end

function _parse_reps(args)::Int
    for a in args
        startswith(a, "--reps=") || continue
        return max(1, parse(Int, split(a, "=", limit=2)[2]))
    end
    return 15
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(; reps=_parse_reps(ARGS))
end
