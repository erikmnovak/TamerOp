#!/usr/bin/env julia
#
# pl_backend_microbench.jl
#
# Purpose
# - Micro-benchmark core performance kernels of the PL encoding stack:
#   1) Axis-aligned backend (`PLBackend` / `PLEncodingMapBoxes`)
#   2) Polyhedral backend (`PLPolyhedra` / `PLEncodingMap`)
#
# What this measures
# - Encoding cost:
#   - `PLBackend.encode_fringe_boxes`
#   - `PLPolyhedra.encode_from_PL_fringe` (when Polyhedra/CDD are available)
# - Query kernels on prebuilt classifiers:
#   - `locate` throughput (vector and tuple inputs)
#   - scalar locate warm vs cold query locality
#   - exact-scan vs hybrid locate mode comparisons
#   - `region_weights` and `region_adjacency`
#   - Polyhedral cache impact (`poly_in_box_cache` / `compile_geometry_cache`) on:
#       - exact geometry queries
#       - Monte Carlo membership-heavy queries
#       - candidate-index on/off comparisons
#
# Scope
# - Deterministic synthetic fixtures (stable across runs).
# - Median wall-time and allocations across repeated runs.
# - Focused kernel benchmark (not a full end-to-end application benchmark).
#
# Usage
#   julia --project=. benchmark/pl_backend_microbench.jl
#   julia --project=. benchmark/pl_backend_microbench.jl --reps=8 --nqueries=40000 --splits=12
#   julia --project=. benchmark/pl_backend_microbench.jl --poly_regions=24 --poly_enc=1

using Random

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
const PLB = TO.PLBackend
const PLP = TO.PLPolyhedra
const RG = TamerOp.RegionGeometry

function _parse_arg(args, key::String, default::Int)
    for a in args
        startswith(a, key * "=") || continue
        return max(1, parse(Int, split(a, "=", limit=2)[2]))
    end
    return default
end

function _parse_bool(args, key::String, default::Bool)
    for a in args
        startswith(a, key * "=") || continue
        v = lowercase(split(a, "=", limit=2)[2])
        if v in ("1", "true", "yes", "y", "on")
            return true
        elseif v in ("0", "false", "no", "n", "off")
            return false
        else
            error("Invalid boolean for $key: $v")
        end
    end
    return default
end

function _bench(name::AbstractString, f::Function; reps::Int=10)
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
    println(rpad(name, 46), " median_time=", round(med_ms, digits=3),
            " ms  median_alloc=", round(med_kib, digits=1), " KiB")
    return (ms=med_ms, kib=med_kib)
end

function _axis_fixture(; splits::Int=10)
    field = CM.QQField()
    K = CM.coeff_type(field)
    xs = collect(-splits:splits)
    ys = collect(-splits:splits)

    # Axis-aligned halfspace families that induce a nontrivial 2D grid.
    Ups = PLB.BoxUpset[]
    Downs = PLB.BoxDownset[]
    for x in xs
        push!(Ups, PLB.BoxUpset([float(x), -100.0]))
        push!(Downs, PLB.BoxDownset([float(x + 1), 100.0]))
    end
    for y in ys
        push!(Ups, PLB.BoxUpset([-100.0, float(y)]))
        push!(Downs, PLB.BoxDownset([100.0, float(y + 1)]))
    end

    m = length(Ups)
    r = length(Downs)
    Phi = Matrix{K}(undef, r, m)
    @inbounds for i in 1:r, j in 1:m
        Phi[i, j] = CM.coerce(field, ((i + 3 * j) % 5) - 2)
    end

    opts = TO.EncodingOptions(backend=:pl_backend, max_regions=2_000_000, poset_kind=:signature, field=field)
    P, H, pi = PLB.encode_fringe_boxes(Ups, Downs, Phi, opts)
    return (P=P, H=H, pi=pi, Ups=Ups, Downs=Downs, Phi=Phi, opts=opts)
end

function _axis_queries(nq::Int; seed::Int=Int(0xB005))
    rng = Random.MersenneTwister(seed)
    vq = Vector{Vector{Float64}}(undef, nq)
    tq = Vector{NTuple{2,Float64}}(undef, nq)
    @inbounds for i in 1:nq
        x = rand(rng) * 2.0 * 15.0 - 15.0
        y = rand(rng) * 2.0 * 15.0 - 15.0
        vq[i] = [x, y]
        tq[i] = (x, y)
    end
    return vq, tq
end

@inline function _run_locate_vec(pi, qv)
    s = 0
    @inbounds for x in qv
        s += EC.locate(pi, x)
    end
    return s
end

@inline function _run_locate_tup(pi, qt)
    s = 0
    @inbounds for x in qt
        s += EC.locate(pi, x)
    end
    return s
end

@inline function _run_locate_vec_mode(pi, qv; mode::Symbol=:fast)
    s = 0
    @inbounds for x in qv
        s += EC.locate(pi, x; mode=mode)
    end
    return s
end

@inline function _run_locate_repeat(pi, x::AbstractVector, n::Int; mode::Symbol=:fast)
    s = 0
    @inbounds for _ in 1:n
        s += EC.locate(pi, x; mode=mode)
    end
    return s
end

@inline function _locate_exact_scan(pi, x::AbstractVector)
    for (idx, hp) in enumerate(pi.regions)
        if PLP._in_hpoly(hp, x)
            return idx
        end
    end
    return 0
end

@inline function _run_locate_exact_scan(pi, qv)
    s = 0
    @inbounds for x in qv
        s += _locate_exact_scan(pi, x)
    end
    return s
end

function _poly_region_hpoly(xlo::Float64, xhi::Float64, ylo::Float64, yhi::Float64)
    # x <= xhi, y <= yhi, x >= xlo, y >= ylo
    A = [1.0 0.0;
         0.0 1.0;
        -1.0 0.0;
         0.0 -1.0]
    b = [xhi, yhi, -xlo, -ylo]
    return PLP.make_hpoly(A, b)
end

function _poly_fixture(; nregions::Int=16)
    nregions >= 2 || error("poly fixture expects at least 2 regions")
    regs = Vector{PLP.HPoly}(undef, nregions)
    wits = Vector{Tuple{Float64,Float64}}(undef, nregions)
    sigy = Vector{BitVector}(undef, nregions)
    sigz = Vector{BitVector}(undef, nregions)

    # Stripe decomposition: [k-1, k] x [0,1]
    @inbounds for k in 1:nregions
        xlo = float(k - 1)
        xhi = float(k)
        regs[k] = _poly_region_hpoly(xlo, xhi, 0.0, 1.0)
        wits[k] = ((xlo + xhi) / 2.0, 0.5)
        sigy[k] = BitVector()
        sigz[k] = BitVector()
    end

    pi = PLP.PLEncodingMap(2, sigy, sigz, regs, wits)
    box = ([-1.0, -0.5], [float(nregions + 1), 1.5])
    return (pi=pi, box=box)
end

function _poly_queries(nq::Int, nregions::Int; seed::Int=Int(0xF011))
    rng = Random.MersenneTwister(seed)
    q = Vector{Vector{Float64}}(undef, nq)
    @inbounds for i in 1:nq
        x = rand(rng) * float(nregions + 2) - 1.0
        y = rand(rng) * 2.0 - 0.5
        q[i] = [x, y]
    end
    return q
end

function _clear_poly_auto_cache!(pi)
    if hasproperty(pi, :cache)
        c = getproperty(pi, :cache)
        c isa CM.EncodingCache && CM._clear_encoding_cache!(c)
    end
    return nothing
end

function _pl_fringe_fixture(; nups::Int=4, ndowns::Int=4)
    K = CM.QQ
    Ups = Vector{PLP.PLUpset}(undef, nups)
    Downs = Vector{PLP.PLDownset}(undef, ndowns)

    @inbounds for i in 1:nups
        a = float(i - 1) / 2.0
        hp = PLP.make_hpoly([-1.0 0.0; 0.0 -1.0], [-a, 0.0]) # x>=a, y>=0
        Ups[i] = PLP.PLUpset(PLP.PolyUnion(2, [hp]))
    end
    @inbounds for j in 1:ndowns
        b = float(j + 1) / 2.0
        hp = PLP.make_hpoly([1.0 0.0; 0.0 1.0], [b, 1.0])    # x<=b, y<=1
        Downs[j] = PLP.PLDownset(PLP.PolyUnion(2, [hp]))
    end

    Phi = Matrix{K}(undef, ndowns, nups)
    @inbounds for j in 1:ndowns, i in 1:nups
        Phi[j, i] = ((i + 2 * j) % 3 == 0) ? K(0) : K(1)
    end
    F = PLP.PLFringe(Ups, Downs, Phi)
    opts = TO.EncodingOptions(backend=:pl, max_regions=100_000, poset_kind=:signature)
    return (F=F, opts=opts)
end

function main(; reps::Int=8, nqueries::Int=30_000, splits::Int=10, poly_regions::Int=16, poly_enc::Bool=true)
    println("PL backend micro-benchmark")
    println("reps=$(reps), nqueries=$(nqueries), splits=$(splits), poly_regions=$(poly_regions)")
    println("HAVE_POLY=$(PLP.HAVE_POLY)\n")

    println("== PLBackend (axis-aligned) ==")
    axis = _axis_fixture(; splits=splits)
    qv, qt = _axis_queries(nqueries)
    box = ([-12.0, -12.0], [12.0, 12.0])

    _bench("axis encode_fringe_boxes", () -> PLB.encode_fringe_boxes(axis.Ups, axis.Downs, axis.Phi, axis.opts); reps=reps)
    _bench("axis locate vectors", () -> _run_locate_vec(axis.pi, qv); reps=reps)
    _bench("axis locate tuples", () -> _run_locate_tup(axis.pi, qt); reps=reps)
    _bench("axis region_weights(box)", () -> sum(RG.region_weights(axis.pi; box=box)); reps=reps)
    _bench("axis region_adjacency(box)", () -> length(RG.region_adjacency(axis.pi; box=box)); reps=reps)
    println()

    println("== PLPolyhedra (general polyhedral) ==")
    if !PLP.HAVE_POLY
        println("polyhedral benchmarks skipped (Polyhedra/CDD not available)")
        return nothing
    end

    poly = _poly_fixture(; nregions=poly_regions)
    pqueries = _poly_queries(nqueries, poly_regions)
    # Use a strict interior point (not on region boundaries) for warm-path timing.
    warm_query = [0.5 * float(poly_regions) + 0.25, 0.5]

    _bench("poly locate vectors", () -> begin
        s = 0
        @inbounds for x in pqueries
            s += EC.locate(poly.pi, x)
        end
        s
    end; reps=reps)
    _bench("poly scalar locate warm (repeat)", () -> _run_locate_repeat(poly.pi, warm_query, nqueries); reps=reps)
    _bench("poly scalar locate cold (random)", () -> _run_locate_vec_mode(poly.pi, pqueries; mode=:fast); reps=reps)
    _bench("poly scalar locate exact_scan", () -> _run_locate_exact_scan(poly.pi, pqueries); reps=reps)
    _bench("poly scalar locate verified", () -> _run_locate_vec_mode(poly.pi, pqueries; mode=:verified); reps=reps)
    Xq = Matrix{Float64}(undef, 2, length(pqueries))
    @inbounds for i in 1:length(pqueries)
        Xq[1, i] = pqueries[i][1]
        Xq[2, i] = pqueries[i][2]
    end
    _bench("poly locate_many! serial", () -> begin
        dest = Vector{Int}(undef, size(Xq, 2))
        PLP.locate_many!(dest, poly.pi, Xq; threaded=false)
        sum(dest)
    end; reps=reps)
    _bench("poly locate_many! threaded", () -> begin
        dest = Vector{Int}(undef, size(Xq, 2))
        PLP.locate_many!(dest, poly.pi, Xq; threaded=true)
        sum(dest)
    end; reps=reps)

    _bench("poly region_weights exact", () -> sum(RG.region_weights(poly.pi; box=poly.box, method=:exact)); reps=reps)
    _bench("poly region_adjacency", () -> length(RG.region_adjacency(poly.pi; box=poly.box, strict=false)); reps=reps)
    _bench("poly exact chain auto cold first", () -> begin
        _clear_poly_auto_cache!(poly.pi)
        s = sum(RG.region_weights(poly.pi; box=poly.box, method=:exact, mode=:fast))
        a = length(RG.region_adjacency(poly.pi; box=poly.box, strict=false, mode=:fast))
        p = RG.region_perimeter(poly.pi, 1; box=poly.box, strict=false, mode=:fast)
        s + a + p
    end; reps=reps)
    _bench("poly exact chain auto warm", () -> begin
        s = sum(RG.region_weights(poly.pi; box=poly.box, method=:exact, mode=:fast))
        a = length(RG.region_adjacency(poly.pi; box=poly.box, strict=false, mode=:fast))
        p = RG.region_perimeter(poly.pi, 1; box=poly.box, strict=false, mode=:fast)
        s + a + p
    end; reps=reps)
    _bench("poly exact chain auto verified", () -> begin
        s = sum(RG.region_weights(poly.pi; box=poly.box, method=:exact, mode=:verified))
        a = length(RG.region_adjacency(poly.pi; box=poly.box, strict=false, mode=:verified))
        p = RG.region_perimeter(poly.pi, 1; box=poly.box, strict=false, mode=:verified)
        s + a + p
    end; reps=reps)

    cache = PLP.compile_geometry_cache(poly.pi; box=poly.box, closure=true)
    _bench("poly compile_geometry_cache", () -> PLP.compile_geometry_cache(poly.pi; box=poly.box, closure=true); reps=reps)

    cache.bucket_enabled = true
    _bench("poly locate_many! cache (bucket on)", () -> begin
        dest = Vector{Int}(undef, size(Xq, 2))
        PLP.locate_many!(dest, cache, Xq; threaded=true)
        sum(dest)
    end; reps=reps)
    _bench("poly scalar locate cache (bucket on)", () -> begin
        s = 0
        @inbounds for x in pqueries
            s += EC.locate(cache, x)
        end
        s
    end; reps=reps)

    cache.bucket_enabled = false
    _bench("poly locate_many! cache (bucket off)", () -> begin
        dest = Vector{Int}(undef, size(Xq, 2))
        PLP.locate_many!(dest, cache, Xq; threaded=true)
        sum(dest)
    end; reps=reps)
    _bench("poly scalar locate cache (bucket off)", () -> begin
        s = 0
        @inbounds for x in pqueries
            s += EC.locate(cache, x)
        end
        s
    end; reps=reps)
    cache.bucket_enabled = true

    _bench("poly region_weights exact (cache)", () -> sum(RG.region_weights(poly.pi; cache=cache, method=:exact)); reps=reps)
    _bench("poly region_adjacency (cache)", () -> length(RG.region_adjacency(poly.pi; cache=cache, strict=false)); reps=reps)
    _bench("poly region_boundary_breakdown (cache)", () -> begin
        bd = RG.region_boundary_measure_breakdown(poly.pi, 1; cache=cache, strict=false, mode=:fast)
        length(bd)
    end; reps=reps)
    _bench("poly region_perimeter (cache)", () -> RG.region_perimeter(poly.pi, 1; cache=cache, strict=false, mode=:fast); reps=reps)
    _bench("poly region_bbox (cache)", () -> begin
        bb = RG.region_bbox(poly.pi, 1; cache=cache)
        bb === nothing ? 0.0 : sum(first(bb)) + sum(last(bb))
    end; reps=reps)
    _bench("poly region_weights exact (cache cold)", () -> begin
        ctmp = PLP.compile_geometry_cache(poly.pi; box=poly.box, closure=true)
        sum(RG.region_weights(poly.pi; cache=ctmp, method=:exact))
    end; reps=reps)
    _bench("poly region_adjacency (cache cold)", () -> begin
        ctmp = PLP.compile_geometry_cache(poly.pi; box=poly.box, closure=true)
        length(RG.region_adjacency(poly.pi; cache=ctmp, strict=false, mode=:fast))
    end; reps=reps)
    _bench("poly region_weights mc", () -> sum(RG.region_weights(poly.pi; box=poly.box, method=:mc, nsamples=8_000, strict=false)); reps=reps)
    _bench("poly region_weights mc (cache)", () -> sum(RG.region_weights(poly.pi; cache=cache, method=:mc, nsamples=8_000, strict=false)); reps=reps)

    if poly_enc
        ffx = _pl_fringe_fixture()
        _bench("poly encode_from_PL_fringe", () -> PLP.encode_from_PL_fringe(ffx.F, ffx.opts); reps=reps)
    end

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    reps = _parse_arg(ARGS, "--reps", 8)
    nqueries = _parse_arg(ARGS, "--nqueries", 30_000)
    splits = _parse_arg(ARGS, "--splits", 10)
    poly_regions = _parse_arg(ARGS, "--poly_regions", 16)
    poly_enc = _parse_bool(ARGS, "--poly_enc", false)
    main(; reps=reps, nqueries=nqueries, splits=splits, poly_regions=poly_regions, poly_enc=poly_enc)
end
