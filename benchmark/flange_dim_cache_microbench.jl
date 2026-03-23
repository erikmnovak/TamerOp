#!/usr/bin/env julia

using Random

if isdefined(Main, :TamerOp)
    const TamerOp = getfield(Main, :TamerOp)
else
    try
        using TamerOp
    catch
        include(joinpath(@__DIR__, "..", "src", "TamerOp.jl"))
        using .TamerOp
    end
end

const TO = TamerOp.Advanced
const CM = TO.CoreModules
const FZ = TO.FlangeZn

function _parse_arg(args, key::String, default::Int)
    for a in args
        startswith(a, key * "=") || continue
        return max(1, parse(Int, split(a, "=", limit=2)[2]))
    end
    return default
end

function _bench(name::AbstractString, f::Function; reps::Int=6)
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

function _random_face(n::Int, rng::AbstractRNG)
    return FZ.face(n, [rand(rng) < 0.4 for _ in 1:n])
end

function _fixture(; n::Int=2, nflats::Int=24, ninj::Int=24, seed::UInt=0x464c414e47455a4e)
    rng = Random.MersenneTwister(seed)
    field = CM.QQField()
    K = CM.coeff_type(field)
    flats = Vector{FZ.IndFlat{n}}(undef, nflats)
    injectives = Vector{FZ.IndInj{n}}(undef, ninj)
    @inbounds for i in 1:nflats
        b = [rand(rng, -8:12) for _ in 1:n]
        flats[i] = FZ.IndFlat(_random_face(n, rng), b; id=Symbol(:F, i))
    end
    @inbounds for j in 1:ninj
        b = [rand(rng, -6:14) for _ in 1:n]
        injectives[j] = FZ.IndInj(_random_face(n, rng), b; id=Symbol(:E, j))
    end
    phi = Matrix{K}(undef, ninj, nflats)
    @inbounds for i in 1:ninj, j in 1:nflats
        phi[i, j] = CM.coerce(field, rand(rng, -2:2))
    end
    FG = FZ.Flange{K}(n, flats, injectives, phi; field=field)
    return FG
end

function _query_points(n::Int, nqueries::Int; seed::UInt=0x51455259464c414e)
    rng = Random.MersenneTwister(seed)
    # Re-sample from a smaller pool to create strong active-set reuse.
    pool = Vector{NTuple{n,Int}}(undef, max(1, div(nqueries, 8)))
    @inbounds for i in eachindex(pool)
        pool[i] = ntuple(_ -> rand(rng, -10:14), n)
    end
    out = Vector{NTuple{n,Int}}(undef, nqueries)
    @inbounds for i in 1:nqueries
        out[i] = pool[rand(rng, eachindex(pool))]
    end
    return out
end

function _box_points(n::Int, side::Int)
    side >= 1 || error("_box_points: side must be >= 1")
    lo = ntuple(_ -> -div(side, 2), n)
    hi = ntuple(i -> lo[i] + side - 1, n)
    axes = ntuple(i -> lo[i]:hi[i], n)
    out = NTuple{n,Int}[]
    sizehint!(out, prod(length(r) for r in axes))
    for tup in Iterators.product(axes...)
        push!(out, ntuple(i -> tup[i], n))
    end
    return out
end

function main(; reps::Int=6, nqueries::Int=25_000, nflats::Int=24, ninj::Int=24)
    FG = _fixture(; nflats=nflats, ninj=ninj)
    points = _query_points(FG.n, nqueries)
    side = max(2, round(Int, nqueries^(1 / FG.n)))
    box_points = _box_points(FG.n, side)
    box_dup = vcat(box_points, box_points, box_points)
    out = Vector{Int}(undef, length(points))
    out_box = Vector{Int}(undef, length(box_points))
    out_box_dup = Vector{Int}(undef, length(box_dup))

    # Parity checks
    ref = [FZ.dim_at(FG, p) for p in points]
    ref_box = [FZ.dim_at(FG, p) for p in box_points]
    ref_box_dup = [FZ.dim_at(FG, p) for p in box_dup]
    @assert FZ.dim_at_many(FG, points; cache=FZ.FlangeDimCache(FG), sort_points=false) == ref
    @assert FZ.dim_at_many(FG, points; cache=FZ.FlangeDimCache(FG), sort_points=true) == ref
    @assert FZ.dim_at_many(FG, box_points; sweep=:box, dedup=true, sort_points=false) == ref_box
    @assert FZ.dim_at_many(FG, box_dup; sweep=:box, dedup=true, sort_points=false) == ref_box_dup

    println("Flange dim cache micro-benchmark")
    println("reps=$(reps), nqueries=$(nqueries), nflats=$(nflats), ninj=$(ninj), box_side=$(side), box_points=$(length(box_points))\n")

    b_scalar = _bench("dim_at scalar (no cache)", () -> begin
        s = 0
        @inbounds for p in points
            s += FZ.dim_at(FG, p)
        end
        s
    end; reps=reps)

    cache_warm = FZ.FlangeDimCache(FG)
    FZ.dim_at_many!(out, FG, points; cache=cache_warm, sort_points=true)
    b_scalar_cache = _bench("dim_at scalar (shared cache)", () -> begin
        s = 0
        @inbounds for p in points
            s += FZ.dim_at(FG, p; cache=cache_warm)
        end
        s
    end; reps=reps)
    println("  speedup no-cache/cache: ", round(b_scalar.ms / max(1e-9, b_scalar_cache.ms), digits=3), "x")
    println("  alloc ratio no-cache/cache: ", round(b_scalar.kib / max(1e-9, b_scalar_cache.kib), digits=3), "x\n")

    b_many_unsorted = _bench("dim_at_many! (unsorted)", () -> begin
        cache = FZ.FlangeDimCache(FG)
        FZ.dim_at_many!(out, FG, points; cache=cache, sort_points=false)
        sum(out)
    end; reps=reps)

    b_many_sorted = _bench("dim_at_many! (sorted)", () -> begin
        cache = FZ.FlangeDimCache(FG)
        FZ.dim_at_many!(out, FG, points; cache=cache, sort_points=true)
        sum(out)
    end; reps=reps)
    println("  speedup unsorted/sorted: ", round(b_many_unsorted.ms / max(1e-9, b_many_sorted.ms), digits=3), "x")
    println("  alloc ratio unsorted/sorted: ", round(b_many_unsorted.kib / max(1e-9, b_many_sorted.kib), digits=3), "x\n")

    b_many_warm = _bench("dim_at_many! (sorted + warm cache)", () -> begin
        FZ.dim_at_many!(out, FG, points; cache=cache_warm, sort_points=true)
        sum(out)
    end; reps=reps)
    println("  speedup scalar-no-cache/warm-many: ",
            round(b_scalar.ms / max(1e-9, b_many_warm.ms), digits=3), "x")
    println("  alloc ratio scalar-no-cache/warm-many: ",
            round(b_scalar.kib / max(1e-9, b_many_warm.kib), digits=3), "x\n")

    b_dedup_off = _bench("dim_at_many! duplicate points (dedup off)", () -> begin
        FZ.dim_at_many!(out_box_dup, FG, box_dup; dedup=false, sort_points=false, sweep=:none)
        sum(out_box_dup)
    end; reps=reps)

    b_dedup_on = _bench("dim_at_many! duplicate points (dedup on)", () -> begin
        FZ.dim_at_many!(out_box_dup, FG, box_dup; dedup=true, sort_points=false, sweep=:none)
        sum(out_box_dup)
    end; reps=reps)
    println("  speedup dedup-off/dedup-on: ", round(b_dedup_off.ms / max(1e-9, b_dedup_on.ms), digits=3), "x")
    println("  alloc ratio dedup-off/dedup-on: ", round(b_dedup_off.kib / max(1e-9, b_dedup_on.kib), digits=3), "x\n")

    b_sweep_off = _bench("dim_at_many! full box (sweep off)", () -> begin
        FZ.dim_at_many!(out_box, FG, box_points; dedup=true, sort_points=false, sweep=:none)
        sum(out_box)
    end; reps=reps)

    b_sweep_on = _bench("dim_at_many! full box (sweep box)", () -> begin
        FZ.dim_at_many!(out_box, FG, box_points; dedup=true, sort_points=false, sweep=:box)
        sum(out_box)
    end; reps=reps)
    println("  speedup sweep-off/sweep-box: ", round(b_sweep_off.ms / max(1e-9, b_sweep_on.ms), digits=3), "x")
    println("  alloc ratio sweep-off/sweep-box: ", round(b_sweep_off.kib / max(1e-9, b_sweep_on.kib), digits=3), "x\n")

    if Threads.nthreads() > 1
        b_thread_off = _bench("dim_at_many! (threaded off)", () -> begin
            FZ.dim_at_many!(out_box_dup, FG, box_dup;
                            dedup=true, sort_points=true, sweep=:none, threaded=false)
            sum(out_box_dup)
        end; reps=reps)
        b_thread_on = _bench("dim_at_many! (threaded on)", () -> begin
            FZ.dim_at_many!(out_box_dup, FG, box_dup;
                            dedup=true, sort_points=true, sweep=:none, threaded=true)
            sum(out_box_dup)
        end; reps=reps)
        println("  speedup threaded-off/threaded-on: ",
                round(b_thread_off.ms / max(1e-9, b_thread_on.ms), digits=3), "x")
        println("  alloc ratio threaded-off/threaded-on: ",
                round(b_thread_off.kib / max(1e-9, b_thread_on.kib), digits=3), "x")
    else
        println("threaded benchmark skipped: set JULIA_NUM_THREADS>1 to measure threaded dim_at_many!")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    reps = _parse_arg(ARGS, "--reps", 6)
    nqueries = _parse_arg(ARGS, "--nqueries", 25_000)
    nflats = _parse_arg(ARGS, "--nflats", 24)
    ninj = _parse_arg(ARGS, "--ninj", 24)
    main(; reps=reps, nqueries=nqueries, nflats=nflats, ninj=ninj)
end
