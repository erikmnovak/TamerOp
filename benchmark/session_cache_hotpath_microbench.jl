#!/usr/bin/env julia
#
# session_cache_hotpath_microbench.jl
#
# Purpose
# - Measure Core SessionCache hot-path lookups/updates with:
#   (old baseline) global session lock path
#   (new path) dedicated Zn locks + single-thread read fastpath
# - Measure Zn tuple locate path:
#   (old baseline) tuple -> collect(vector) fallback
#   (new path) direct tuple locate method
#
# Usage
#   julia --project=. benchmark/session_cache_hotpath_microbench.jl
#   julia --project=. benchmark/session_cache_hotpath_microbench.jl --reps=12 --iters=400000 --queries=200000

using Random
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
const FZ = TO.FlangeZn
const ZE = TO.ZnEncoding
const _BASELINE_SESSION_LOCK = Base.ReentrantLock()

mutable struct _BaselineZnCaches
    plan::Dict{Tuple{UInt64,UInt64},Any}
    encoding::Dict{Tuple{UInt64,Symbol,Int},Any}
    fringe::Dict{Tuple{UInt64,Symbol,UInt64,UInt},Any}
    modules::Dict{Tuple{UInt64,Symbol,UInt64,UInt},Any}
end

_BaselineZnCaches() = _BaselineZnCaches(
    Dict{Tuple{UInt64,UInt64},Any}(),
    Dict{Tuple{UInt64,Symbol,Int},Any}(),
    Dict{Tuple{UInt64,Symbol,UInt64,UInt},Any}(),
    Dict{Tuple{UInt64,Symbol,UInt64,UInt},Any}(),
)

function _parse_int(args, key::String, default::Int)
    for a in args
        startswith(a, key * "=") || continue
        return max(1, parse(Int, split(a, "=", limit=2)[2]))
    end
    return default
end

function _bench(name::AbstractString, f::Function; reps::Int)
    GC.gc()
    f() # warmup
    GC.gc()
    times_ms = Vector{Float64}(undef, reps)
    alloc_b = Vector{Int}(undef, reps)
    for i in 1:reps
        m = @timed f()
        times_ms[i] = 1000.0 * m.time
        alloc_b[i] = m.bytes
    end
    sort!(times_ms)
    sort!(alloc_b)
    med_t = times_ms[cld(reps, 2)]
    med_a = alloc_b[cld(reps, 2)] / 1024.0
    println(rpad(name, 44), " median_time=", round(med_t, digits=3), " ms",
            "  median_alloc=", round(med_a, digits=2), " KiB")
    return (ms=med_t, kib=med_a)
end

@inline function _baseline_session_get_plan(state::_BaselineZnCaches,
                                            encoding_fp::UInt64,
                                            flange_fp::UInt64)
    key = (encoding_fp, flange_fp)
    Base.lock(_BASELINE_SESSION_LOCK)
    try
        return get(state.plan, key, nothing)
    finally
        Base.unlock(_BASELINE_SESSION_LOCK)
    end
end

@inline function _baseline_session_set_plan!(state::_BaselineZnCaches,
                                             encoding_fp::UInt64,
                                             flange_fp::UInt64,
                                             payload)
    key = (encoding_fp, flange_fp)
    Base.lock(_BASELINE_SESSION_LOCK)
    try
        state.plan[key] = payload
    finally
        Base.unlock(_BASELINE_SESSION_LOCK)
    end
    return payload
end

@inline function _baseline_session_get_encoding(state::_BaselineZnCaches,
                                                encoding_fp::UInt64,
                                                poset_kind::Symbol,
                                                max_regions::Int)
    key = (encoding_fp, poset_kind, max_regions)
    Base.lock(_BASELINE_SESSION_LOCK)
    try
        return get(state.encoding, key, nothing)
    finally
        Base.unlock(_BASELINE_SESSION_LOCK)
    end
end

@inline function _baseline_session_set_encoding!(state::_BaselineZnCaches,
                                                 encoding_fp::UInt64,
                                                 poset_kind::Symbol,
                                                 max_regions::Int,
                                                 payload)
    key = (encoding_fp, poset_kind, max_regions)
    Base.lock(_BASELINE_SESSION_LOCK)
    try
        state.encoding[key] = payload
    finally
        Base.unlock(_BASELINE_SESSION_LOCK)
    end
    return payload
end

@inline function _baseline_session_get_fringe(state::_BaselineZnCaches,
                                              encoding_fp::UInt64,
                                              poset_kind::Symbol,
                                              flange_fp::UInt64,
                                              field_key::UInt)
    key = (encoding_fp, poset_kind, flange_fp, field_key)
    Base.lock(_BASELINE_SESSION_LOCK)
    try
        return get(state.fringe, key, nothing)
    finally
        Base.unlock(_BASELINE_SESSION_LOCK)
    end
end

@inline function _baseline_session_set_fringe!(state::_BaselineZnCaches,
                                               encoding_fp::UInt64,
                                               poset_kind::Symbol,
                                               flange_fp::UInt64,
                                               field_key::UInt,
                                               payload)
    key = (encoding_fp, poset_kind, flange_fp, field_key)
    Base.lock(_BASELINE_SESSION_LOCK)
    try
        state.fringe[key] = payload
    finally
        Base.unlock(_BASELINE_SESSION_LOCK)
    end
    return payload
end

@inline function _baseline_session_get_module(state::_BaselineZnCaches,
                                              encoding_fp::UInt64,
                                              poset_kind::Symbol,
                                              flange_fp::UInt64,
                                              field_key::UInt)
    key = (encoding_fp, poset_kind, flange_fp, field_key)
    Base.lock(_BASELINE_SESSION_LOCK)
    try
        return get(state.modules, key, nothing)
    finally
        Base.unlock(_BASELINE_SESSION_LOCK)
    end
end

@inline function _baseline_session_set_module!(state::_BaselineZnCaches,
                                               encoding_fp::UInt64,
                                               poset_kind::Symbol,
                                               flange_fp::UInt64,
                                               field_key::UInt,
                                               payload)
    key = (encoding_fp, poset_kind, flange_fp, field_key)
    Base.lock(_BASELINE_SESSION_LOCK)
    try
        state.modules[key] = payload
    finally
        Base.unlock(_BASELINE_SESSION_LOCK)
    end
    return payload
end

@inline function _old_tuple_collect_locate(pi, g::NTuple{N,Float64}) where {N}
    return EC.locate(pi, collect(g))
end

mutable struct _BaselineSessionBuckets
    lock::Base.ReentrantLock
    encoding::Dict{UInt,CM.EncodingCache}
    modules::Dict{Tuple{UInt,UInt},CM.ModuleCache}
end

_BaselineSessionBuckets() =
    _BaselineSessionBuckets(Base.ReentrantLock(), Dict{UInt,CM.EncodingCache}(), Dict{Tuple{UInt,UInt},CM.ModuleCache}())

@inline function _baseline_encoding_cache!(state::_BaselineSessionBuckets, key::UInt)
    Base.lock(state.lock)
    try
        return get!(state.encoding, key) do
            CM.EncodingCache()
        end
    finally
        Base.unlock(state.lock)
    end
end

@inline function _baseline_module_cache!(state::_BaselineSessionBuckets, key::Tuple{UInt,UInt})
    Base.lock(state.lock)
    try
        return get!(state.modules, key) do
            CM.ModuleCache(key[1], key[2])
        end
    finally
        Base.unlock(state.lock)
    end
end

@inline _field_cache_key_old(field) = UInt(hash((typeof(field), field)))

mutable struct _BaselineProductCache
    dense::IdDict{Any,IdDict{Any,Any}}
end

_BaselineProductCache() = _BaselineProductCache(IdDict{Any,IdDict{Any,Any}}())

@inline function _baseline_product_dense_get!(cache::_BaselineProductCache, k1, k2)
    inner = get!(cache.dense, k1) do
        IdDict{Any,Any}()
    end
    return get(inner, k2, nothing)
end

@inline function _baseline_product_dense_set!(cache::_BaselineProductCache, k1, k2, payload)
    inner = get!(cache.dense, k1) do
        IdDict{Any,Any}()
    end
    inner[k2] = payload
    return payload
end

function _compile_encoding_old(P, pi)
    axes_val = nothing
    reps_val = nothing
    if hasmethod(EC.axes_from_encoding, (typeof(pi),))
        axes_val = EC.axes_from_encoding(pi)
    end
    if hasmethod(EC.representatives, (typeof(pi),))
        reps_val = EC.representatives(pi)
    end
    return EC.CompiledEncoding(P, pi, axes_val, reps_val, NamedTuple())
end

function _fixture()
    field = CM.QQField()
    K = CM.coeff_type(field)
    tau = FZ.face(2, [false, false])
    flats = [
        FZ.IndFlat(tau, [0, 0]; id=:F1),
        FZ.IndFlat(tau, [1, 0]; id=:F2),
        FZ.IndFlat(tau, [0, 1]; id=:F3),
    ]
    injectives = [
        FZ.IndInj(tau, [1, 1]; id=:E1),
        FZ.IndInj(tau, [2, 0]; id=:E2),
        FZ.IndInj(tau, [0, 2]; id=:E3),
    ]
    phi = Matrix{K}(I, length(injectives), length(flats))
    FG = FZ.Flange{K}(2, flats, injectives, phi; field=field)
    opts = TO.EncodingOptions(backend=:zn, max_regions=4096, field=field)
    P, pi = ZE.encode_poset_from_flanges((FG,), opts; poset_kind=:signature)
    return (FG=FG, P=P, pi=pi, opts=opts)
end

function main(args=ARGS)
    reps = _parse_int(args, "--reps", 10)
    iters = _parse_int(args, "--iters", 300_000)
    nqueries = _parse_int(args, "--queries", 150_000)
    rng = Random.MersenneTwister(0xCACE5E55)

    fx = _fixture()
    FG = fx.FG
    pi = fx.pi
    opts = fx.opts
    field = opts.field
    sc = CM.SessionCache()
    zn_baseline = _BaselineZnCaches()
    enc0 = TO.encode(FG, opts; cache=sc)

    fkey = ZE._flange_fingerprint(FG)
    ekey = pi.encoding_fingerprint
    poset_kind = opts.poset_kind
    max_regions = something(opts.max_regions, 0)
    field_key = CM._field_cache_key(field)
    payload = (
        flat_idxs = [1, 2, 3],
        inj_idxs = [1, 2, 3],
        zero_pairs = Tuple{Int,Int}[],
    )
    _baseline_session_set_plan!(zn_baseline, ekey, fkey, payload)
    CM._session_set_zn_pushforward_plan!(sc, ekey, fkey, payload)
    _baseline_session_set_encoding!(zn_baseline, ekey, poset_kind, max_regions, (P=fx.P, pi=pi))
    CM._session_set_zn_encoding_artifact!(sc, ekey, poset_kind, max_regions, (P=fx.P, pi=pi))
    _baseline_session_set_fringe!(zn_baseline, ekey, poset_kind, fkey, field_key, enc0.H)
    CM._session_set_zn_pushforward_fringe!(sc, ekey, poset_kind, fkey, field_key, enc0.H)
    _baseline_session_set_module!(zn_baseline, ekey, poset_kind, fkey, field_key, (H=enc0.H, M=enc0.M))
    CM._session_set_zn_pushforward_module!(sc, ekey, poset_kind, fkey, field_key, (H=enc0.H, M=enc0.M))

    queries = Vector{NTuple{2,Float64}}(undef, nqueries)
    @inbounds for i in 1:nqueries
        queries[i] = (rand(rng) * 5.0 - 2.0, rand(rng) * 5.0 - 2.0)
    end
    # Warm tuple paths before allocation measurement.
    EC.locate(pi, queries[1])
    _old_tuple_collect_locate(pi, queries[1])

    println("SessionCache hot-path benchmark")
    println("reps=$(reps), iters=$(iters), queries=$(nqueries), nthreads=$(Threads.nthreads())")
    println()

    bget_old = _bench("plan get (baseline global lock)", () -> begin
        s = 0
        @inbounds for _ in 1:iters
            x = _baseline_session_get_plan(zn_baseline, ekey, fkey)
            s += (x === nothing ? 0 : 1)
        end
        s
    end; reps=reps)
    bget_new = _bench("plan get (new session path)", () -> begin
        s = 0
        @inbounds for _ in 1:iters
            x = CM._session_get_zn_pushforward_plan(sc, ekey, fkey)
            s += (x === nothing ? 0 : 1)
        end
        s
    end; reps=reps)
    println("  speedup get baseline/new: ", round(bget_old.ms / bget_new.ms, digits=3), "x")
    println()

    bset_old = _bench("plan set (baseline global lock)", () -> begin
        for _ in 1:iters
            _baseline_session_set_plan!(zn_baseline, ekey, fkey, payload)
        end
        nothing
    end; reps=reps)
    bset_new = _bench("plan set (new session path)", () -> begin
        for _ in 1:iters
            CM._session_set_zn_pushforward_plan!(sc, ekey, fkey, payload)
        end
        nothing
    end; reps=reps)
    println("  speedup set baseline/new: ", round(bset_old.ms / bset_new.ms, digits=3), "x")
    println()

    benc_get_old = _bench("zn encoding get (baseline global lock)", () -> begin
        s = 0
        @inbounds for _ in 1:iters
            x = _baseline_session_get_encoding(zn_baseline, ekey, poset_kind, max_regions)
            s += (x === nothing ? 0 : 1)
        end
        s
    end; reps=reps)
    benc_get_new = _bench("zn encoding get (sharded)", () -> begin
        s = 0
        @inbounds for _ in 1:iters
            x = CM._session_get_zn_encoding_artifact(sc, ekey, poset_kind, max_regions)
            s += (x === nothing ? 0 : 1)
        end
        s
    end; reps=reps)
    println("  speedup zn-encoding get baseline/new: ", round(benc_get_old.ms / benc_get_new.ms, digits=3), "x")

    bfr_get_old = _bench("zn fringe get (baseline global lock)", () -> begin
        s = 0
        @inbounds for _ in 1:iters
            x = _baseline_session_get_fringe(zn_baseline, ekey, poset_kind, fkey, field_key)
            s += (x === nothing ? 0 : 1)
        end
        s
    end; reps=reps)
    bfr_get_new = _bench("zn fringe get (sharded)", () -> begin
        s = 0
        @inbounds for _ in 1:iters
            x = CM._session_get_zn_pushforward_fringe(sc, ekey, poset_kind, fkey, field_key)
            s += (x === nothing ? 0 : 1)
        end
        s
    end; reps=reps)
    println("  speedup zn-fringe get baseline/new: ", round(bfr_get_old.ms / bfr_get_new.ms, digits=3), "x")

    bmod_get_old = _bench("zn module get (baseline global lock)", () -> begin
        s = 0
        @inbounds for _ in 1:iters
            x = _baseline_session_get_module(zn_baseline, ekey, poset_kind, fkey, field_key)
            s += (x === nothing ? 0 : 1)
        end
        s
    end; reps=reps)
    bmod_get_new = _bench("zn module get (sharded)", () -> begin
        s = 0
        @inbounds for _ in 1:iters
            x = CM._session_get_zn_pushforward_module(sc, ekey, poset_kind, fkey, field_key)
            s += (x === nothing ? 0 : 1)
        end
        s
    end; reps=reps)
    println("  speedup zn-module get baseline/new: ", round(bmod_get_old.ms / bmod_get_new.ms, digits=3), "x")
    println()

    bloc_old = _bench("locate tuple (baseline collect)", () -> begin
        s = 0
        @inbounds for q in queries
            s += _old_tuple_collect_locate(pi, q)
        end
        s
    end; reps=reps)
    bloc_new = _bench("locate tuple (direct tuple path)", () -> begin
        s = 0
        @inbounds for q in queries
            s += EC.locate(pi, q)
        end
        s
    end; reps=reps)
    println("  speedup locate baseline/new: ", round(bloc_old.ms / bloc_new.ms, digits=3), "x")

    alloc_old = @allocated _old_tuple_collect_locate(pi, queries[1])
    alloc_new = @allocated EC.locate(pi, queries[1])
    println("  allocation bytes (single query): old=", alloc_old, ", new=", alloc_new)
    println()

    # encoding/module cache sharding benchmark
    keys_enc = Vector{UInt}(undef, iters)
    keys_mod = Vector{Tuple{UInt,UInt}}(undef, iters)
    @inbounds for i in 1:iters
        k = UInt(rand(rng, 1:1024))
        keys_enc[i] = k
        keys_mod[i] = (k, UInt(rand(rng, 1:8)))
    end
    baseline_state = _BaselineSessionBuckets()
    benc_old = _bench("encoding cache get! (baseline lock)", () -> begin
        x = 0
        @inbounds for k in keys_enc
            c = _baseline_encoding_cache!(baseline_state, k)
            x += length(c.geometry)
        end
        x
    end; reps=reps)
    benc_new = _bench("encoding cache get! (sharded)", () -> begin
        x = 0
        @inbounds for k in keys_enc
            c = CM._encoding_cache!(sc, k)
            x += length(c.geometry)
        end
        x
    end; reps=reps)
    println("  speedup encoding baseline/new: ", round(benc_old.ms / benc_new.ms, digits=3), "x")

    bmod_old = _bench("module cache get! (baseline lock)", () -> begin
        x = 0
        @inbounds for k in keys_mod
            c = _baseline_module_cache!(baseline_state, k)
            x += length(c.payload)
        end
        x
    end; reps=reps)
    bmod_new = _bench("module cache get! (sharded)", () -> begin
        x = 0
        @inbounds for k in keys_mod
            c = CM._module_cache!(sc, k)
            x += length(c.payload)
        end
        x
    end; reps=reps)
    println("  speedup module baseline/new: ", round(bmod_old.ms / bmod_new.ms, digits=3), "x")
    println()

    # product-poset cache keying benchmark (old nested IdDict vs tuple-id key map)
    prod_baseline = _BaselineProductCache()
    sc_prod = CM.SessionCache()
    dense_keys_1 = [falses(4, 4) for _ in 1:64]
    dense_keys_2 = [falses(4, 4) for _ in 1:64]
    @inbounds for i in eachindex(dense_keys_1)
        dense_keys_1[i][1, 1] = true
        dense_keys_2[i][1, 1] = true
        k1 = dense_keys_1[i]
        k2 = dense_keys_2[i]
        _baseline_product_dense_set!(prod_baseline, k1, k2, (k1, k2))
        key = CM._SessionProductKey(k1, k2)
        sc_prod.product_dense[key] = CM.ProductPosetCacheEntry{Any,Any,Any,Any,Any}(k1, k2, nothing, nothing, nothing)
    end
    prod_idxs = [rand(rng, eachindex(dense_keys_1)) for _ in 1:iters]

    bprod_old = _bench("product cache get (baseline nested id)", () -> begin
        s = 0
        @inbounds for idx in prod_idxs
            k1 = dense_keys_1[idx]
            k2 = dense_keys_2[idx]
            x = _baseline_product_dense_get!(prod_baseline, k1, k2)
            s += (x === nothing ? 0 : 1)
        end
        s
    end; reps=reps)
    bprod_new = _bench("product cache get (tuple-id key)", () -> begin
        s = 0
        @inbounds for idx in prod_idxs
            k1 = dense_keys_1[idx]
            k2 = dense_keys_2[idx]
            key = CM._SessionProductKey(k1, k2)
            entry = get(sc_prod.product_dense, key, nothing)
            s += (entry === nothing ? 0 : 1)
        end
        s
    end; reps=reps)
    println("  speedup product-cache baseline/new: ", round(bprod_old.ms / bprod_new.ms, digits=3), "x")
    println()

    # compile_encoding and field-key hashing hot paths
    bcomp_old = _bench("compile_encoding (old hasmethod)", () -> begin
        s = 0
        @inbounds for _ in 1:iters
            enc = _compile_encoding_old(fx.P, pi)
            s += (enc.axes === nothing ? 0 : 1) + (enc.reps === nothing ? 0 : 1)
        end
        s
    end; reps=reps)
    bcomp_new = _bench("compile_encoding (new trait hook)", () -> begin
        s = 0
        @inbounds for _ in 1:iters
            enc = EC.compile_encoding(fx.P, pi)
            s += (enc.axes === nothing ? 0 : 1) + (enc.reps === nothing ? 0 : 1)
        end
        s
    end; reps=reps)
    println("  speedup compile old/new: ", round(bcomp_old.ms / bcomp_new.ms, digits=3), "x")

    key_iters = max(2_000_000, 32 * iters)
    prime_pool = CM.PrimeField[
        CM.F2(),
        CM.F3(),
        CM.Fp(5),
        CM.Fp(7),
        CM.Fp(11),
        CM.Fp(13),
        CM.Fp(17),
        CM.Fp(19),
    ]
    nprime_pool = length(prime_pool)
    bkey_old = _bench("field key hash (old tuple hash)", () -> begin
        s = UInt(0)
        @inbounds for i in 1:key_iters
            f = prime_pool[mod1(i, nprime_pool)]
            s = xor(s, _field_cache_key_old(f))
        end
        Int(s & UInt(0x7fffffff))
    end; reps=reps)
    bkey_new = _bench("field key hash (specialized)", () -> begin
        s = UInt(0)
        @inbounds for i in 1:key_iters
            f = prime_pool[mod1(i, nprime_pool)]
            s = xor(s, CM._field_cache_key(f))
        end
        Int(s & UInt(0x7fffffff))
    end; reps=reps)
    println("  key_iters=", key_iters)
    println("  speedup key old/new: ", round(bkey_old.ms / bkey_new.ms, digits=3), "x")
end

main()
