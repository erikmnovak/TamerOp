#!/usr/bin/env julia
# benchmark/flange_hotpaths_microbench.jl
#
# Integrated FlangeZn hot-path benchmark.
#
# Scope:
# - current live hot paths (`_dim_at_cached!`, `dim_at_many!`)
# - local A/B probes for the old decode-on-miss path vs the new words path
# - local A/B probes for slab queries without vs with the new sweep/event path
#
# This script expects the full owner-module dependency stack used by
# `src/FieldLinAlg.jl`.

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)

using Random

module _FlangeHotpathsBench
include(joinpath(@__DIR__, "..", "src", "CoreModules.jl"))
include(joinpath(@__DIR__, "..", "src", "FieldLinAlg.jl"))
module IndicatorResolutions
abstract type PModule end
end
include(joinpath(@__DIR__, "..", "src", "FlangeZn.jl"))
end

const CM = _FlangeHotpathsBench.CoreModules
const FL = _FlangeHotpathsBench.FieldLinAlg
const FZ = _FlangeHotpathsBench.FlangeZn

function _parse_arg(args, key::String, default::Int)
    for a in args
        startswith(a, key * "=") || continue
        return max(1, parse(Int, split(a, "=", limit=2)[2]))
    end
    return default
end

function _bench(name::AbstractString, f::Function; reps::Int=6)
    GC.gc()
    f()
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
    println(rpad(name, 38), " median_time=", round(med_ms, digits=3),
            " ms  median_alloc=", round(med_kib, digits=1), " KiB")
    return (ms=med_ms, kib=med_kib)
end

function _random_face(n::Int, rng::AbstractRNG)
    return FZ.face(n, [rand(rng) < 0.4 for _ in 1:n])
end

function _fixture(; n::Int=2, nflats::Int=32, ninj::Int=32, seed::UInt=0x464c414e47455a4e)
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
    return FZ.Flange{K}(n, flats, injectives, phi; field=field)
end

function _query_points(n::Int, nqueries::Int; seed::UInt=0x51455259464c414e, pool_size::Int=max(1, div(nqueries, 8)))
    rng = Random.MersenneTwister(seed)
    pool = Vector{NTuple{n,Int}}(undef, max(1, pool_size))
    @inbounds for i in eachindex(pool)
        pool[i] = ntuple(_ -> rand(rng, -10:14), n)
    end
    out = Vector{NTuple{n,Int}}(undef, nqueries)
    @inbounds for i in 1:nqueries
        out[i] = pool[rand(rng, eachindex(pool))]
    end
    return out
end

function _grid_points_2d(nx::Int, ny::Int; x0::Int=0, y0::Int=0, ystep::Int=1)
    pts = Vector{NTuple{2,Int}}(undef, nx * ny)
    k = 1
    @inbounds for j in 0:(ny - 1), i in 0:(nx - 1)
        pts[k] = (x0 + i, y0 + j * ystep)
        k += 1
    end
    return pts
end

function _grid_points_2d_axis2_dense(nx::Int, ny::Int; x0::Int=0, y0::Int=0, xstep::Int=2)
    pts = Vector{NTuple{2,Int}}(undef, nx * ny)
    k = 1
    @inbounds for i in 0:(nx - 1), j in 0:(ny - 1)
        pts[k] = (x0 + i * xstep, y0 + j)
        k += 1
    end
    return pts
end

function _dim_at_cached_olddecode!(cache::FZ.FlangeDimCache, FG::FZ.Flange, g)
    k = cache.kernel
    nr = FZ._fill_active_injectives_words_kernel!(cache.row_words, k, g)
    nc = FZ._fill_active_flats_words_kernel!(cache.col_words, k, g)
    if nr == 0 || nc == 0
        return 0
    end

    key = FZ._active_words_hash(cache.row_words, cache.col_words)
    cached = FZ._lookup_rank(cache, key, cache.row_words, cache.col_words)
    cached >= 0 && return cached

    cache.misses += 1
    FZ._decode_active_words!(cache.rows, cache.row_words, cache.kernel.ninj)
    FZ._decode_active_words!(cache.cols, cache.col_words, cache.kernel.nflat)
    rk = FL.rank_restricted(FG.field, FG.phi, cache.rows, cache.cols)
    return FZ._store_rank!(cache, key, cache.row_words, cache.col_words, rk)
end

function _rank_restricted_words_old(field, A, row_words, col_words, nr, nc; nrows::Int, ncols::Int)
    B = FL._materialize_restricted_dense_from_words(A, row_words, col_words, nr, nc, nrows, ncols)
    return FL.rank(field, B; backend=:auto)
end

function main(; reps::Int=6, nqueries::Int=8_000, nflats::Int=32, ninj::Int=32, pool_size::Int=64)
    FG = _fixture(; nflats=nflats, ninj=ninj)
    points = _query_points(FG.n, nqueries; pool_size=pool_size)
    out = Vector{Int}(undef, length(points))
    miss_points = _grid_points_2d(max(32, min(128, cld(nqueries, 8))), 8; x0=-16, y0=-6, ystep=3)
    slab_points = _grid_points_2d(max(256, min(1024, cld(nqueries, 4))), 4; x0=-24, y0=-6, ystep=2)
    slab_points_axis2 = _grid_points_2d_axis2_dense(4, max(256, min(1024, cld(nqueries, 4))); x0=-6, y0=-24, xstep=2)
    miss_probe = FZ.FlangeDimCache(FG)
    miss_states = Tuple{Vector{UInt64},Vector{UInt64},Int,Int}[]
    seen = Set{UInt64}()
    @inbounds for p in miss_points
        nr = FZ._fill_active_injectives_words_kernel!(miss_probe.row_words, miss_probe.kernel, p)
        nc = FZ._fill_active_flats_words_kernel!(miss_probe.col_words, miss_probe.kernel, p)
        (nr == 0 || nc == 0) && continue
        key = FZ._active_words_hash(miss_probe.row_words, miss_probe.col_words)
        key in seen && continue
        push!(seen, key)
        push!(miss_states, (copy(miss_probe.row_words), copy(miss_probe.col_words), nr, nc))
    end

    cache = FZ.FlangeDimCache(FG)
    ref = [FZ.dim_at(FG, p) for p in points]
    @assert [FZ.dim_at(FG, p; cache=cache) for p in points] == ref
    for p in points
        FZ.dim_at(FG, p; cache=cache)
    end
    FZ.dim_at_many!(out, FG, points; cache=cache, sort_points=true)
    @assert out == ref
    slab_ref = FZ.dim_at_many(FG, slab_points; sweep=:none, dedup=true, sort_points=true)
    @assert FZ.dim_at_many(FG, slab_points; sweep=:auto, dedup=true, sort_points=true) == slab_ref
    slab_ref_axis2 = FZ.dim_at_many(FG, slab_points_axis2; sweep=:none, dedup=true, sort_points=true)
    @assert FZ.dim_at_many(FG, slab_points_axis2; sweep=:auto, dedup=true, sort_points=true) == slab_ref_axis2

    println("Flange hotpaths micro-benchmark")
    println("reps=$(reps), nqueries=$(nqueries), pool_size=$(pool_size), nflats=$(nflats), ninj=$(ninj)\n")

    b_fill = _bench("_fill_active_*_kernel!", () -> begin
        rows = Int[]
        cols = Int[]
        sizehint!(rows, cache.kernel.ninj)
        sizehint!(cols, cache.kernel.nflat)
        row_words = Base.zeros(UInt64, length(cache.row_words))
        col_words = Base.zeros(UInt64, length(cache.col_words))
        s = 0
        @inbounds for p in points
            FZ._fill_active_injectives_kernel!(rows, row_words, cache.kernel, p)
            FZ._fill_active_flats_kernel!(cols, col_words, cache.kernel, p)
            s += length(rows) + length(cols)
        end
        s
    end; reps=reps)

    b_fill_words = _bench("_fill_active_*_words_kernel!", () -> begin
        row_words = Base.zeros(UInt64, length(cache.row_words))
        col_words = Base.zeros(UInt64, length(cache.col_words))
        s = 0
        @inbounds for p in points
            s += FZ._fill_active_injectives_words_kernel!(row_words, cache.kernel, p)
            s += FZ._fill_active_flats_words_kernel!(col_words, cache.kernel, p)
        end
        s
    end; reps=reps)

    b_cached = _bench("_dim_at_cached! warm", () -> begin
        s = 0
        @inbounds for p in points
            s += FZ._dim_at_cached!(cache, FG, p)
        end
        s
    end; reps=reps)

    b_miss_old = _bench("_dim_at_cached! miss olddecode", () -> begin
        c = FZ.FlangeDimCache(FG)
        s = 0
        @inbounds for p in miss_points
            s += _dim_at_cached_olddecode!(c, FG, p)
        end
        s
    end; reps=reps)

    b_miss_new = _bench("_dim_at_cached! miss words", () -> begin
        c = FZ.FlangeDimCache(FG)
        s = 0
        @inbounds for p in miss_points
            s += FZ._dim_at_cached!(c, FG, p)
        end
        s
    end; reps=reps)

    b_rank_words_old = _bench("rank_words old materialize", () -> begin
        s = 0
        @inbounds for (row_words, col_words, nr, nc) in miss_states
            s += _rank_restricted_words_old(FG.field, FG.phi, row_words, col_words, nr, nc;
                                            nrows=cache.kernel.ninj, ncols=cache.kernel.nflat)
        end
        s
    end; reps=reps)

    b_rank_words_new = _bench("rank_words current", () -> begin
        s = 0
        @inbounds for (row_words, col_words, nr, nc) in miss_states
            s += FL.rank_restricted_words(FG.field, FG.phi, row_words, col_words, nr, nc;
                                          nrows=cache.kernel.ninj, ncols=cache.kernel.nflat)
        end
        s
    end; reps=reps)

    small_states = [(row_words, col_words, nr, nc) for (row_words, col_words, nr, nc) in miss_states
                    if length(row_words) <= 2 && length(col_words) <= 2 && nr * nc <= 256]
    b_rank_words_small_old = _bench("rank_words old smallmask", () -> begin
        s = 0
        @inbounds for (row_words, col_words, nr, nc) in small_states
            s += _rank_restricted_words_old(FG.field, FG.phi, row_words, col_words, nr, nc;
                                            nrows=cache.kernel.ninj, ncols=cache.kernel.nflat)
        end
        s
    end; reps=reps)

    b_rank_words_small_new = _bench("rank_words current smallmask", () -> begin
        s = 0
        @inbounds for (row_words, col_words, nr, nc) in small_states
            s += FL.rank_restricted_words(FG.field, FG.phi, row_words, col_words, nr, nc;
                                          nrows=cache.kernel.ninj, ncols=cache.kernel.nflat)
        end
        s
    end; reps=reps)

    b_rank_words_exact = _bench("rank_words exact", () -> begin
        s = 0
        @inbounds for (row_words, col_words, nr, nc) in miss_states
            s += FL.rank_restricted_words(FG.field, FG.phi, row_words, col_words, nr, nc;
                                          nrows=cache.kernel.ninj, ncols=cache.kernel.nflat,
                                          backend=:exact)
        end
        s
    end; reps=reps)

    b_rank_words_modular = _bench("rank_words modular", () -> begin
        s = 0
        @inbounds for (row_words, col_words, nr, nc) in miss_states
            s += FL.rank_restricted_words(FG.field, FG.phi, row_words, col_words, nr, nc;
                                          nrows=cache.kernel.ninj, ncols=cache.kernel.nflat,
                                          backend=:modular)
        end
        s
    end; reps=reps)

    b_rank_words_nemo = _bench("rank_words nemo", () -> begin
        s = 0
        @inbounds for (row_words, col_words, nr, nc) in miss_states
            s += FL.rank_restricted_words(FG.field, FG.phi, row_words, col_words, nr, nc;
                                          nrows=cache.kernel.ninj, ncols=cache.kernel.nflat,
                                          backend=:nemo)
        end
        s
    end; reps=reps)

    b_many = _bench("dim_at_many! sorted warm", () -> begin
        FZ.dim_at_many!(out, FG, points; cache=cache, sort_points=true)
        sum(out)
    end; reps=reps)

    slab_out = Vector{Int}(undef, length(slab_points))
    b_slab_none = _bench("dim_at_many! slab none", () -> begin
        c = FZ.FlangeDimCache(FG)
        FZ.dim_at_many!(slab_out, FG, slab_points;
                        cache=c, dedup=true, sort_points=true, sweep=:none)
        sum(slab_out)
    end; reps=reps)

    b_slab_auto = _bench("dim_at_many! slab auto", () -> begin
        c = FZ.FlangeDimCache(FG)
        FZ.dim_at_many!(slab_out, FG, slab_points;
                        cache=c, dedup=true, sort_points=true, sweep=:auto)
        sum(slab_out)
    end; reps=reps)

    slab_out_axis2 = Vector{Int}(undef, length(slab_points_axis2))
    b_slab_axis2_none = _bench("dim_at_many! slab axis2 none", () -> begin
        c = FZ.FlangeDimCache(FG)
        FZ.dim_at_many!(slab_out_axis2, FG, slab_points_axis2;
                        cache=c, dedup=true, sort_points=true, sweep=:none)
        sum(slab_out_axis2)
    end; reps=reps)

    b_slab_axis2_auto = _bench("dim_at_many! slab axis2 auto", () -> begin
        c = FZ.FlangeDimCache(FG)
        FZ.dim_at_many!(slab_out_axis2, FG, slab_points_axis2;
                        cache=c, dedup=true, sort_points=true, sweep=:auto)
        sum(slab_out_axis2)
    end; reps=reps)

    println("  fill/pure-words speedup: ", round(b_fill.ms / max(1e-9, b_fill_words.ms), digits=3), "x")
    println("  fill/cache speedup: ", round(b_fill.ms / max(1e-9, b_cached.ms), digits=3), "x")
    println("  cache/many speedup: ", round(b_cached.ms / max(1e-9, b_many.ms), digits=3), "x")
    println("  miss words speedup: ", round(b_miss_old.ms / max(1e-9, b_miss_new.ms), digits=3), "x")
    println("  rank words speedup: ", round(b_rank_words_old.ms / max(1e-9, b_rank_words_new.ms), digits=3), "x")
    println("  rank words smallmask speedup: ", round(b_rank_words_small_old.ms / max(1e-9, b_rank_words_small_new.ms), digits=3), "x")
    println("  rank auto/exact speedup: ", round(b_rank_words_exact.ms / max(1e-9, b_rank_words_new.ms), digits=3), "x")
    println("  rank auto/modular speedup: ", round(b_rank_words_modular.ms / max(1e-9, b_rank_words_new.ms), digits=3), "x")
    println("  rank auto/nemo speedup: ", round(b_rank_words_nemo.ms / max(1e-9, b_rank_words_new.ms), digits=3), "x")
    println("  slab auto speedup: ", round(b_slab_none.ms / max(1e-9, b_slab_auto.ms), digits=3), "x")
    println("  slab axis2 auto speedup: ", round(b_slab_axis2_none.ms / max(1e-9, b_slab_axis2_auto.ms), digits=3), "x")
end

if abspath(PROGRAM_FILE) == @__FILE__
    reps = _parse_arg(ARGS, "--reps", 6)
    nqueries = _parse_arg(ARGS, "--nqueries", 8_000)
    nflats = _parse_arg(ARGS, "--nflats", 32)
    ninj = _parse_arg(ARGS, "--ninj", 32)
    pool_size = _parse_arg(ARGS, "--pool", 64)
    main(; reps=reps, nqueries=nqueries, nflats=nflats, ninj=ninj, pool_size=pool_size)
end
