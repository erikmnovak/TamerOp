#!/usr/bin/env julia
#
# modules_microbench.jl
#
# Purpose
# - Benchmark core `Modules.jl` kernels that dominate P-module workflows:
#   1) `map_leq` cover-chain composition
#   2) `direct_sum` assembly
#   3) `PModule` construction path (dict vs aligned store)
#
# What this compares
# - For each kernel, run:
#   - a baseline implementation (pre-optimization algorithmic shape),
#   - the current optimized implementation.
# - Includes a dedicated short-chain query block (<=2 cover edges) to track
#   tiny-query fast paths in `map_leq`.
#
# Scope
# - Deterministic synthetic fixtures on product-of-chains posets.
# - Median time + allocation over repeated runs.
# - Functional parity checks before timing.
#
# Usage
#   julia --project=. benchmark/modules_microbench.jl
#   julia --project=. benchmark/modules_microbench.jl --reps=7 --nx=12 --ny=12 --queries=3000
#   julia --project=. benchmark/modules_microbench.jl --fields=qq,f3,real
#   julia --project=. benchmark/modules_microbench.jl --section=longchain --fields=qq
#

using Random
using SparseArrays
using LinearAlgebra

function _load_posetmodules_for_modules_bench()
    try
        @eval using TamerOp
        return :package
    catch
    end

    src_entry = joinpath(@__DIR__, "..", "src", "TamerOp.jl")
    try
        include(src_entry)
        @eval using .TamerOp
        return :source_full
    catch
    end

    # Last-resort fallback for module-only benchmarking when unrelated files fail to load.
    @eval module TamerOp
        include(joinpath(@__DIR__, "..", "src", "CoreModules.jl"))
        include(joinpath(@__DIR__, "..", "src", "Options.jl"))
        include(joinpath(@__DIR__, "..", "src", "EncodingCore.jl"))
        include(joinpath(@__DIR__, "..", "src", "FieldLinAlg.jl"))
        include(joinpath(@__DIR__, "..", "src", "FiniteFringe.jl"))
        include(joinpath(@__DIR__, "..", "src", "Modules.jl"))
        module Advanced
            const CoreModules = Main.TamerOp.CoreModules
            const Options = Main.TamerOp.Options
            const EncodingCore = Main.TamerOp.EncodingCore
            const FiniteFringe = Main.TamerOp.FiniteFringe
            const Modules = Main.TamerOp.Modules
        end
    end
    @eval using .TamerOp
    return :source_modules_only
end

const _PM_LOAD_MODE = _load_posetmodules_for_modules_bench()

const TO = TamerOp.Advanced
const MD = TO.Modules
const FF = TO.FiniteFringe
const CM = TO.CoreModules
const FL = TamerOp.FieldLinAlg

function _parse_int_arg(args, key::String, default::Int)
    for a in args
        startswith(a, key * "=") || continue
        return max(1, parse(Int, split(a, "=", limit=2)[2]))
    end
    return default
end

function _parse_float_arg(args, key::String, default::Float64)
    for a in args
        startswith(a, key * "=") || continue
        return max(0.0, parse(Float64, split(a, "=", limit=2)[2]))
    end
    return default
end

function _parse_string_list_arg(args, key::String, default::Vector{String})
    for a in args
        startswith(a, key * "=") || continue
        raw = split(a, "=", limit=2)[2]
        vals = String[]
        for s in split(raw, ",")
            t = lowercase(strip(s))
            isempty(t) || push!(vals, t)
        end
        return isempty(vals) ? default : vals
    end
    return default
end

function _parse_sections(args)
    return Set(_parse_string_list_arg(args, "--section", ["all"]))
end

@inline function _want_section(sections::Set{String}, names::Vararg{String})
    ("all" in sections) && return true
    @inbounds for name in names
        name in sections && return true
    end
    return false
end

function _bench(name::AbstractString, f::Function; reps::Int=7)
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
    println(rpad(name, 44), " median_time=", round(med_ms, digits=3),
            " ms  median_alloc=", round(med_kib, digits=1), " KiB")
    return (ms=med_ms, kib=med_kib)
end

function _field_from_name(name::String)
    name == "qq" && return CM.QQField()
    name == "f2" && return CM.F2()
    name == "f3" && return CM.F3()
    name == "f5" && return CM.Fp(5)
    name == "real" && return CM.RealField(Float64; rtol=1e-10, atol=1e-12)
    error("unknown field name '$name' (supported: qq,f2,f3,f5,real)")
end

function _rand_coeff(rng::AbstractRNG, field::CM.AbstractCoeffField)
    if field isa CM.RealField
        return CM.coerce(field, 2rand(rng) - 1)
    end
    v = rand(rng, -3:3)
    v == 0 && (v = 1)
    return CM.coerce(field, v)
end

function _random_pmodule(Q::FF.AbstractPoset, field::CM.AbstractCoeffField;
                         dmin::Int=2, dmax::Int=6, density::Float64=0.4, seed::Int=0xA0A0)
    rng = MersenneTwister(seed)
    K = CM.coeff_type(field)
    n = FF.nvertices(Q)
    dims = rand(rng, dmin:dmax, n)

    edge = Dict{Tuple{Int,Int}, Matrix{K}}()
    for (u, v) in FF.cover_edges(Q)
        A = zeros(K, dims[v], dims[u])
        @inbounds for i in 1:dims[v], j in 1:dims[u]
            rand(rng) < density || continue
            A[i, j] = _rand_coeff(rng, field)
        end
        edge[(u, v)] = A
    end

    return MD.PModule{K}(Q, dims, edge; field=field)
end

function _unit_pmodule(Q::FF.AbstractPoset, field::CM.AbstractCoeffField)
    K = CM.coeff_type(field)
    n = FF.nvertices(Q)
    dims = ones(Int, n)
    edge = Dict{Tuple{Int,Int}, Matrix{K}}()
    oneK = CM.coerce(field, 1)
    for (u, v) in FF.cover_edges(Q)
        edge[(u, v)] = reshape(K[oneK], 1, 1)
    end
    return MD.PModule{K}(Q, dims, edge; field=field)
end

function _edge_dict(M::MD.PModule{K}) where {K}
    edge = Dict{Tuple{Int,Int}, Matrix{K}}()
    sizehint!(edge, length(M.edge_maps))
    for ((u, v), A) in M.edge_maps
        edge[(u, v)] = Matrix{K}(A)
    end
    return edge
end

function _random_comparable_pairs(Q::FF.AbstractPoset, nqueries::Int; seed::Int=0xB0B0)
    rng = MersenneTwister(seed)
    n = FF.nvertices(Q)
    pairs = Tuple{Int,Int}[]
    sizehint!(pairs, nqueries)
    while length(pairs) < nqueries
        u = rand(rng, 1:n)
        v = rand(rng, 1:n)
        FF.leq(Q, u, v) || continue
        push!(pairs, (u, v))
    end
    return pairs
end

@inline function _pred_lists(cc::FF.CoverCache, n::Int)
    return [collect(FF._preds(cc, v)) for v in 1:n]
end

@inline function _succ_lists(cc::FF.CoverCache, n::Int)
    return [collect(FF._succs(cc, u)) for u in 1:n]
end

function _random_short_chain_pairs(Q::FF.AbstractPoset, succs::Vector{Vector{Int}}, nqueries::Int; seed::Int=0xB1B1)
    rng = MersenneTwister(seed)
    n = FF.nvertices(Q)
    pairs = Tuple{Int,Int}[]
    sizehint!(pairs, nqueries)
    while length(pairs) < nqueries
        u = rand(rng, 1:n)
        su = succs[u]
        isempty(su) && continue
        if rand(rng, Bool)
            v = su[rand(rng, eachindex(su))]
            push!(pairs, (u, v))
        else
            mid = su[rand(rng, eachindex(su))]
            smid = succs[mid]
            isempty(smid) && continue
            v = smid[rand(rng, eachindex(smid))]
            push!(pairs, (u, v))
        end
    end
    return pairs
end

function _chosen_chain_hops_old(Q::FF.AbstractPoset, preds::Vector{Vector{Int}}, u::Int, v::Int)
    u == v && return 0
    hops = 0
    d = v
    while d != u
        idx = findfirst(x -> x != u && FF.leq(Q, u, x), preds[d])
        d = idx === nothing ? u : preds[d][idx]
        hops += 1
    end
    return hops
end

function _random_long_chain_pairs(Q::FF.AbstractPoset, preds::Vector{Vector{Int}}, nqueries::Int; seed::Int=0xB2B2)
    rng = MersenneTwister(seed)
    n = FF.nvertices(Q)
    pairs = Tuple{Int,Int}[]
    sizehint!(pairs, nqueries)
    while length(pairs) < nqueries
        u = rand(rng, 1:n)
        v = rand(rng, 1:n)
        u == v && continue
        FF.leq(Q, u, v) || continue
        _chosen_chain_hops_old(Q, preds, u, v) >= 3 || continue
        push!(pairs, (u, v))
    end
    return pairs
end

function _shared_target_long_pairs(Q::FF.AbstractPoset, preds::Vector{Vector{Int}}, nqueries::Int; seed::Int=0xB3B3)
    rng = MersenneTwister(seed)
    n = FF.nvertices(Q)
    targets = Int[]
    for v in 1:n
        for u in 1:n
            u == v && continue
            FF.leq(Q, u, v) || continue
            _chosen_chain_hops_old(Q, preds, u, v) >= 3 || continue
            push!(targets, v)
            break
        end
    end
    isempty(targets) && return Tuple{Int,Int}[]

    pairs = Tuple{Int,Int}[]
    sizehint!(pairs, nqueries)
    while length(pairs) < nqueries
        v = targets[rand(rng, eachindex(targets))]
        u = rand(rng, 1:n)
        u == v && continue
        FF.leq(Q, u, v) || continue
        _chosen_chain_hops_old(Q, preds, u, v) >= 3 || continue
        push!(pairs, (u, v))
    end
    return pairs
end

function _low_overlap_long_pairs(Q::FF.AbstractPoset, preds::Vector{Vector{Int}}, nqueries::Int; seed::Int=0xB4B4)
    rng = MersenneTwister(seed)
    n = FF.nvertices(Q)
    pairs = Tuple{Int,Int}[]
    seen = Set{Tuple{Int,Int}}()
    sizehint!(pairs, nqueries)
    while length(pairs) < nqueries
        u = rand(rng, 1:n)
        v = rand(rng, 1:n)
        u == v && continue
        FF.leq(Q, u, v) || continue
        _chosen_chain_hops_old(Q, preds, u, v) >= 3 || continue
        pair = (u, v)
        pair in seen && continue
        push!(seen, pair)
        push!(pairs, pair)
    end
    return pairs
end

function _map_leq_old(field::CM.AbstractCoeffField,
                      Q::FF.AbstractPoset,
                      dims::Vector{Int},
                      edge::Dict{Tuple{Int,Int}, <:AbstractMatrix},
                      preds::Vector{Vector{Int}},
                      u::Int, v::Int)
    if u == v
        return CM.eye(field, dims[v])
    end
    uv = (u, v)
    if haskey(edge, uv)
        return edge[uv]
    end
    idx = findfirst(x -> x != u && FF.leq(Q, u, x), preds[v])
    w = (idx === nothing) ? u : preds[v][idx]
    return FL._matmul(edge[(w, v)], _map_leq_old(field, Q, dims, edge, preds, u, w))
end

function _map_leq_batch_old(M::MD.PModule, edge::Dict, preds, pairs)
    acc = 0
    for (u, v) in pairs
        A = _map_leq_old(M.field, M.Q, M.dims, edge, preds, u, v)
        acc += size(A, 1) + size(A, 2)
    end
    return acc
end

function _map_leq_batch_new(M::MD.PModule, cc::FF.CoverCache, pairs)
    acc = 0
    for (u, v) in pairs
        A = MD.map_leq(M, u, v; cache=cc)
        acc += size(A, 1) + size(A, 2)
    end
    return acc
end

function _map_leq_batch_many(M::MD.PModule, cc::FF.CoverCache, pairs)
    mats = MD.map_leq_many(M, pairs; cache=cc)
    acc = 0
    for A in mats
        acc += size(A, 1) + size(A, 2)
    end
    return acc
end

function _map_leq_batch_many_prepared(M::MD.PModule, cc::FF.CoverCache, batch)
    mats = MD.map_leq_many(M, batch; cache=cc)
    acc = 0
    for A in mats
        acc += size(A, 1) + size(A, 2)
    end
    return acc
end

function _map_leq_batch_many_prepared_prealloc!(out::Vector, M::MD.PModule, cc::FF.CoverCache, batch)
    MD.map_leq_many!(out, M, batch; cache=cc)
    acc = 0
    for A in out
        acc += size(A, 1) + size(A, 2)
    end
    return acc
end

function _map_leq_single_old_repeat(M::MD.PModule, edge::Dict, preds, pair::Tuple{Int,Int}, nrep::Int)
    u, v = pair
    acc = 0
    @inbounds for _ in 1:nrep
        A = _map_leq_old(M.field, M.Q, M.dims, edge, preds, u, v)
        acc += size(A, 1) + size(A, 2)
    end
    return acc
end

function _map_leq_single_new_repeat(M::MD.PModule, cc::FF.CoverCache, pair::Tuple{Int,Int}, nrep::Int)
    u, v = pair
    acc = 0
    @inbounds for _ in 1:nrep
        A = MD.map_leq(M, u, v; cache=cc)
        acc += size(A, 1) + size(A, 2)
    end
    return acc
end

function _map_leq_single_new_cold(M::MD.PModule, cc::FF.CoverCache, pair::Tuple{Int,Int})
    FF._clear_chain_parent_cache!(cc)
    MD._clear_map_leq_memo!(M)
    return _map_leq_single_new_repeat(M, cc, pair, 1)
end

function _map_leq_accumulate(mats)
    acc = 0
    for A in mats
        acc += size(A, 1) + size(A, 2)
    end
    return acc
end

function _poset_scan_oldstyle(M::MD.PModule, pairs)
    # Emulates the pre-parametric Q::AbstractPoset field access path.
    Q::FF.AbstractPoset = M.Q
    acc = 0
    @inbounds for (u, v) in pairs
        acc += FF.leq(Q, u, v) ? 1 : 0
    end
    return acc
end

function _poset_scan_newstyle(M::MD.PModule, pairs)
    Q = M.Q
    acc = 0
    @inbounds for (u, v) in pairs
        acc += FF.leq(Q, u, v) ? 1 : 0
    end
    return acc
end

function _map_leq_identity_old(M::MD.PModule, pairs)
    acc = 0
    @inbounds for (_, v) in pairs
        A = CM.eye(M.field, M.dims[v])
        acc += size(A, 1)
    end
    return acc
end

function _map_leq_identity_new(M::MD.PModule, cc::FF.CoverCache, pairs)
    acc = 0
    @inbounds for (u, v) in pairs
        A = MD.map_leq(M, u, v; cache=cc)
        acc += size(A, 1)
    end
    return acc
end

function _with_direct_sum_policy(f::Function;
                                 min_entries::Int,
                                 qq::Float64,
                                 prime::Float64,
                                 real::Float64,
                                 qq_avg::Float64,
                                 prime_avg::Float64,
                                 real_avg::Float64)
    old_min = MD.DIRECT_SUM_SPARSE_MIN_TOTAL_ENTRIES[]
    old_qq_avg = MD.DIRECT_SUM_SPARSE_MIN_AVG_EDGE_ENTRIES_QQ[]
    old_prime_avg = MD.DIRECT_SUM_SPARSE_MIN_AVG_EDGE_ENTRIES_PRIME[]
    old_real_avg = MD.DIRECT_SUM_SPARSE_MIN_AVG_EDGE_ENTRIES_REAL[]
    old_qq = MD.DIRECT_SUM_SPARSE_MAX_DENSITY[]
    old_prime = MD.DIRECT_SUM_SPARSE_MAX_DENSITY_PRIME[]
    old_real = MD.DIRECT_SUM_SPARSE_MAX_DENSITY_REAL[]
    try
        MD.DIRECT_SUM_SPARSE_MIN_TOTAL_ENTRIES[] = min_entries
        MD.DIRECT_SUM_SPARSE_MIN_AVG_EDGE_ENTRIES_QQ[] = qq_avg
        MD.DIRECT_SUM_SPARSE_MIN_AVG_EDGE_ENTRIES_PRIME[] = prime_avg
        MD.DIRECT_SUM_SPARSE_MIN_AVG_EDGE_ENTRIES_REAL[] = real_avg
        MD.DIRECT_SUM_SPARSE_MAX_DENSITY[] = qq
        MD.DIRECT_SUM_SPARSE_MAX_DENSITY_PRIME[] = prime
        MD.DIRECT_SUM_SPARSE_MAX_DENSITY_REAL[] = real
        return f()
    finally
        MD.DIRECT_SUM_SPARSE_MIN_TOTAL_ENTRIES[] = old_min
        MD.DIRECT_SUM_SPARSE_MIN_AVG_EDGE_ENTRIES_QQ[] = old_qq_avg
        MD.DIRECT_SUM_SPARSE_MIN_AVG_EDGE_ENTRIES_PRIME[] = old_prime_avg
        MD.DIRECT_SUM_SPARSE_MIN_AVG_EDGE_ENTRIES_REAL[] = old_real_avg
        MD.DIRECT_SUM_SPARSE_MAX_DENSITY[] = old_qq
        MD.DIRECT_SUM_SPARSE_MAX_DENSITY_PRIME[] = old_prime
        MD.DIRECT_SUM_SPARSE_MAX_DENSITY_REAL[] = old_real
    end
end

function _with_map_many_plan_policy(f::Function; min_len::Int)
    old_min = MD.MAP_LEQ_MANY_PLAN_MIN_LEN[]
    try
        MD.MAP_LEQ_MANY_PLAN_MIN_LEN[] = min_len
        return f()
    finally
        MD.MAP_LEQ_MANY_PLAN_MIN_LEN[] = old_min
    end
end

function _with_scalar_long_executor_policy(f::Function)
    old_plan_min = MD.MAP_LEQ_MANY_PLAN_MIN_LEN[]
    old_oneoff_min = MD.MAP_LEQ_MANY_ONEOFF_LONG_MIN_LEN[]
    old_oneoff_long = MD.MAP_LEQ_MANY_ONEOFF_LONG_MIN_LONG[]
    old_oneoff_overlap = MD.MAP_LEQ_MANY_ONEOFF_MIN_OVERLAP_QQ[]
    old_oneoff_target = MD.MAP_LEQ_MANY_ONEOFF_MIN_TARGET_REPEAT_QQ[]
    old_scalar_overlap = MD.MAP_LEQ_MANY_LONG_SCALAR_MAX_OVERLAP_QQ[]
    old_scalar_target = MD.MAP_LEQ_MANY_LONG_SCALAR_MAX_TARGET_REPEAT_QQ[]
    old_scalar_hops = MD.MAP_LEQ_MANY_LONG_MIN_AVG_HOPS_QQ[]
    try
        MD.MAP_LEQ_MANY_PLAN_MIN_LEN[] = typemax(Int)
        MD.MAP_LEQ_MANY_ONEOFF_LONG_MIN_LEN[] = 1
        MD.MAP_LEQ_MANY_ONEOFF_LONG_MIN_LONG[] = 1
        MD.MAP_LEQ_MANY_ONEOFF_MIN_OVERLAP_QQ[] = 1.0
        MD.MAP_LEQ_MANY_ONEOFF_MIN_TARGET_REPEAT_QQ[] = 1.0
        MD.MAP_LEQ_MANY_LONG_SCALAR_MAX_OVERLAP_QQ[] = 1.0
        MD.MAP_LEQ_MANY_LONG_SCALAR_MAX_TARGET_REPEAT_QQ[] = 1.0
        MD.MAP_LEQ_MANY_LONG_MIN_AVG_HOPS_QQ[] = 0.0
        return f()
    finally
        MD.MAP_LEQ_MANY_PLAN_MIN_LEN[] = old_plan_min
        MD.MAP_LEQ_MANY_ONEOFF_LONG_MIN_LEN[] = old_oneoff_min
        MD.MAP_LEQ_MANY_ONEOFF_LONG_MIN_LONG[] = old_oneoff_long
        MD.MAP_LEQ_MANY_ONEOFF_MIN_OVERLAP_QQ[] = old_oneoff_overlap
        MD.MAP_LEQ_MANY_ONEOFF_MIN_TARGET_REPEAT_QQ[] = old_oneoff_target
        MD.MAP_LEQ_MANY_LONG_SCALAR_MAX_OVERLAP_QQ[] = old_scalar_overlap
        MD.MAP_LEQ_MANY_LONG_SCALAR_MAX_TARGET_REPEAT_QQ[] = old_scalar_target
        MD.MAP_LEQ_MANY_LONG_MIN_AVG_HOPS_QQ[] = old_scalar_hops
    end
end

function _direct_sum_sparse_preferred_old_emulated(A::MD.PModule, B::MD.PModule)
    total_entries = 0
    total_nnz = 0
    for ((u, v), Au) in A.edge_maps
        Bu = B.edge_maps[u, v]
        total_entries += (A.dims[v] + B.dims[v]) * (A.dims[u] + B.dims[u])
        total_nnz += nnz(Au) + nnz(Bu)
    end
    total_entries == 0 && return false
    total_entries < MD.DIRECT_SUM_SPARSE_MIN_TOTAL_ENTRIES[] && return false
    return total_nnz <= MD._direct_sum_sparse_density_cap(A.field) * total_entries
end

function _with_dense_memo_policy(f::Function; min_entries::Int, max_entries::Int)
    old_min = MD.MAP_LEQ_DENSE_MEMO_MIN_ENTRIES[]
    old_max = MD.MAP_LEQ_DENSE_MEMO_MAX_ENTRIES_PER_THREAD[]
    try
        MD.MAP_LEQ_DENSE_MEMO_MIN_ENTRIES[] = min_entries
        MD.MAP_LEQ_DENSE_MEMO_MAX_ENTRIES_PER_THREAD[] = max_entries
        return f()
    finally
        MD.MAP_LEQ_DENSE_MEMO_MIN_ENTRIES[] = old_min
        MD.MAP_LEQ_DENSE_MEMO_MAX_ENTRIES_PER_THREAD[] = old_max
    end
end

function _direct_sum_old(A::MD.PModule{K}, B::MD.PModule{K},
                         edgeA::Dict{Tuple{Int,Int}, Matrix{K}},
                         edgeB::Dict{Tuple{Int,Int}, Matrix{K}}) where {K}
    Q = A.Q
    n = FF.nvertices(Q)
    dims = [A.dims[i] + B.dims[i] for i in 1:n]
    edge = Dict{Tuple{Int,Int}, Matrix{K}}()
    sizehint!(edge, length(edgeA))
    for (u, v) in FF.cover_edges(Q)
        Au = edgeA[(u, v)]
        Bu = edgeB[(u, v)]
        aU, bU = A.dims[u], B.dims[u]
        aV, bV = A.dims[v], B.dims[v]
        Muv = zeros(K, aV + bV, aU + bU)
        if aV != 0 && aU != 0
            copyto!(view(Muv, 1:aV, 1:aU), Au)
        end
        if bV != 0 && bU != 0
            copyto!(view(Muv, aV+1:aV+bV, aU+1:aU+bU), Bu)
        end
        edge[(u, v)] = Muv
    end
    return MD.PModule{K}(Q, dims, edge; field=A.field)
end

@inline function _mat_equal(field::CM.AbstractCoeffField, A::AbstractMatrix, B::AbstractMatrix)
    size(A) == size(B) || return false
    if field isa CM.RealField
        return norm(Matrix{Float64}(A) - Matrix{Float64}(B)) <= 1e-8
    end
    return A == B
end

function _check_direct_sum_parity(field::CM.AbstractCoeffField,
                                  Sold::MD.PModule,
                                  Snew::MD.PModule)
    Sold.dims == Snew.dims || return false
    for ((u, v), A) in Sold.edge_maps
        _mat_equal(field, A, Snew.edge_maps[u, v]) || return false
    end
    return true
end

function main(args=ARGS)
    reps = _parse_int_arg(args, "--reps", 7)
    nx = _parse_int_arg(args, "--nx", 12)
    ny = _parse_int_arg(args, "--ny", 12)
    nqueries = _parse_int_arg(args, "--queries", 3000)
    density = _parse_float_arg(args, "--density", 0.40)
    field_names = _parse_string_list_arg(args, "--fields", ["qq", "f3", "real"])
    sections = _parse_sections(args)

    println("Modules microbench")
    println("reps=$(reps), grid=($(nx),$(ny)), queries=$(nqueries), density=$(density), fields=$(field_names), sections=$(sort!(collect(sections)))\n")
    println("load_mode=", _PM_LOAD_MODE)

    Q = FF.ProductOfChainsPoset((nx, ny))
    cc = FF._get_cover_cache(Q)
    preds = _pred_lists(cc, FF.nvertices(Q))
    succs = _succ_lists(cc, FF.nvertices(Q))
    pairs = _random_comparable_pairs(Q, nqueries; seed=Int(0xB0B0))
    short_pairs = _random_short_chain_pairs(Q, succs, nqueries; seed=Int(0xB1B1))
    long_pairs = _random_long_chain_pairs(Q, preds, nqueries; seed=Int(0xB2B2))
    overlap_pairs = _shared_target_long_pairs(Q, preds, nqueries; seed=Int(0xB3B3))
    scalar_long_pairs = _low_overlap_long_pairs(Q, preds, min(nqueries, 64); seed=Int(0xB4B4))
    long_pair = (1, FF.nvertices(Q))
    println("chain_parent_cache=", cc.chain_parent_dense === nothing ? "dict" : "dense")

    run_map = _want_section(sections, "map")
    run_longchain = _want_section(sections, "longchain")
    run_plan = _want_section(sections, "plan")
    run_dispatch = _want_section(sections, "dispatch")
    run_memo = _want_section(sections, "memo")
    run_identity = _want_section(sections, "identity")
    run_directsum = _want_section(sections, "directsum")
    run_ctor = _want_section(sections, "ctor")
    need_map_core = run_map || run_longchain || run_plan || run_dispatch || run_memo || run_identity

    for fname in field_names
        field = _field_from_name(fname)
        K = CM.coeff_type(field)
        println("== field=$(fname) (K=$(K)) ==")

        A = _random_pmodule(Q, field; density=density, seed=Int(0xA111))
        edgeA = _edge_dict(A)
        B = nothing
        edgeB = nothing

        if need_map_core
            batch_old = _map_leq_batch_old(A, edgeA, preds, pairs)
            batch_new = _map_leq_batch_new(A, cc, pairs)
            batch_many = _map_leq_batch_many(A, cc, pairs)
            batch_old == batch_new || error("map_leq parity failed for field=$(fname)")
            batch_new == batch_many || error("map_leq_many parity failed for field=$(fname)")
        end

        if run_directsum || run_ctor
            B = _random_pmodule(Q, field; density=density, seed=Int(0xA222))
            edgeB = _edge_dict(B)
        end

        if run_directsum
            Sold = _direct_sum_old(A, B, edgeA, edgeB)
            Snew = MD.direct_sum(A, B)
            _check_direct_sum_parity(field, Sold, Snew) || error("direct_sum parity failed for field=$(fname)")
        end

        if run_map
            println("  -- map_leq composition")
            b_old_map = _bench("map_leq old (dict+no parent memo)", () -> _map_leq_batch_old(A, edgeA, preds, pairs); reps=reps)

            b_new_map_cold = _bench("map_leq new cold (empty memo)", () -> begin
                FF._clear_chain_parent_cache!(cc)
                MD._clear_map_leq_memo!(A)
                _map_leq_batch_new(A, cc, pairs)
            end; reps=reps)

            _map_leq_batch_new(A, cc, pairs)
            b_new_map_warm = _bench("map_leq new warm (memoized)", () -> _map_leq_batch_new(A, cc, pairs); reps=reps)
            b_new_map_many = _bench("map_leq batch API (memoized)", () -> _map_leq_batch_many(A, cc, pairs); reps=reps)
            println("  speedup(new/old) cold: ", round(b_old_map.ms / b_new_map_cold.ms, digits=2), "x")
            println("  speedup(new/old) warm: ", round(b_old_map.ms / b_new_map_warm.ms, digits=2), "x")
            println("  speedup(batch/old) warm: ", round(b_old_map.ms / b_new_map_many.ms, digits=2), "x")
            short_old = _map_leq_batch_old(A, edgeA, preds, short_pairs)
            short_new = _map_leq_batch_new(A, cc, short_pairs)
            short_many = _map_leq_batch_many(A, cc, short_pairs)
            short_old == short_new || error("short-chain map_leq parity failed for field=$(fname)")
            short_new == short_many || error("short-chain map_leq_many parity failed for field=$(fname)")
            b_old_short = _bench("map_leq old short-chain (<=2 hops)", () -> _map_leq_batch_old(A, edgeA, preds, short_pairs); reps=reps)
            b_new_short = _bench("map_leq new short-chain fastpath", () -> begin
                FF._clear_chain_parent_cache!(cc)
                MD._clear_map_leq_memo!(A)
                _map_leq_batch_new(A, cc, short_pairs)
            end; reps=reps)
            b_many_short = _bench("map_leq batch short-chain fastpath", () -> begin
                FF._clear_chain_parent_cache!(cc)
                MD._clear_map_leq_memo!(A)
                _map_leq_batch_many(A, cc, short_pairs)
            end; reps=reps)
            println("  speedup(new/old) short-chain: ", round(b_old_short.ms / b_new_short.ms, digits=2), "x")
            println("  speedup(batch/old) short-chain: ", round(b_old_short.ms / b_many_short.ms, digits=2), "x")
        end

        if run_longchain
            long_old = _map_leq_batch_old(A, edgeA, preds, long_pairs)
            long_new = _map_leq_batch_new(A, cc, long_pairs)
            long_old == long_new || error("long-chain map_leq parity failed for field=$(fname)")
            MD._clear_map_leq_many_plan_cache!(A)
            println("  route(raw long-chain batch): ", MD._map_leq_many_raw_route_kind(A, long_pairs, cc))
            b_old_long = _bench("map_leq old long-chain (>=3 hops)", () -> _map_leq_batch_old(A, edgeA, preds, long_pairs); reps=reps)
            b_new_long = _bench("map_leq new long-chain cold", () -> begin
                FF._clear_chain_parent_cache!(cc)
                MD._clear_map_leq_memo!(A)
                _map_leq_batch_new(A, cc, long_pairs)
            end; reps=reps)
            b_many_long = _bench("map_leq batch long-chain cold", () -> begin
                FF._clear_chain_parent_cache!(cc)
                MD._clear_map_leq_memo!(A)
                MD._clear_map_leq_many_plan_cache!(A)
                _map_leq_batch_many(A, cc, long_pairs)
            end; reps=reps)
            println("  speedup(new/old) long-chain: ", round(b_old_long.ms / b_new_long.ms, digits=2), "x")
            println("  speedup(batch/old) long-chain: ", round(b_old_long.ms / b_many_long.ms, digits=2), "x")

            overlap_old = _map_leq_batch_old(A, edgeA, preds, overlap_pairs)
            overlap_new = _map_leq_batch_new(A, cc, overlap_pairs)
            MD._clear_map_leq_many_plan_cache!(A)
            println("  route(raw overlap batch): ", MD._map_leq_many_raw_route_kind(A, overlap_pairs, cc))
            overlap_many = _map_leq_batch_many(A, cc, overlap_pairs)
            overlap_old == overlap_new || error("overlap long-chain map_leq parity failed for field=$(fname)")
            overlap_new == overlap_many || error("overlap long-chain map_leq_many parity failed for field=$(fname)")
            b_old_overlap = _bench("map_leq old long-chain overlap", () -> _map_leq_batch_old(A, edgeA, preds, overlap_pairs); reps=reps)
            b_new_overlap = _bench("map_leq new long-chain overlap", () -> begin
                FF._clear_chain_parent_cache!(cc)
                MD._clear_map_leq_memo!(A)
                _map_leq_batch_new(A, cc, overlap_pairs)
            end; reps=reps)
            b_many_overlap = _bench("map_leq batch long-chain overlap", () -> begin
                FF._clear_chain_parent_cache!(cc)
                MD._clear_map_leq_memo!(A)
                MD._clear_map_leq_many_plan_cache!(A)
                _map_leq_batch_many(A, cc, overlap_pairs)
            end; reps=reps)
            println("  speedup(new/old) overlap: ", round(b_old_overlap.ms / b_new_overlap.ms, digits=2), "x")
            println("  speedup(batch/old) overlap: ", round(b_old_overlap.ms / b_many_overlap.ms, digits=2), "x")
        end

        if run_longchain
            short_pair = short_pairs[1]
            b_old_single_short = _bench("map_leq old single-short repeated", () -> _map_leq_single_old_repeat(A, edgeA, preds, short_pair, nqueries); reps=reps)
            b_new_single_short = _bench("map_leq new single-short repeated", () -> begin
                FF._clear_chain_parent_cache!(cc)
                MD._clear_map_leq_memo!(A)
                _map_leq_single_new_repeat(A, cc, short_pair, nqueries)
            end; reps=reps)
            println("  speedup(single-short new/old): ", round(b_old_single_short.ms / b_new_single_short.ms, digits=2), "x")

            b_old_single_long = _bench("map_leq old single-long cold", () -> _map_leq_single_old_repeat(A, edgeA, preds, long_pair, 1); reps=reps)
            b_new_single_long = _bench("map_leq new single-long cold", () -> _map_leq_single_new_cold(A, cc, long_pair); reps=reps)
            println("  speedup(single-long cold new/old): ", round(b_old_single_long.ms / b_new_single_long.ms, digits=2), "x")

            _with_scalar_long_executor_policy() do
                scalar_old = _map_leq_batch_old(A, edgeA, preds, scalar_long_pairs)
                scalar_new = _map_leq_batch_many(A, cc, scalar_long_pairs)
                scalar_old == scalar_new || error("scalar-long executor parity failed for field=$(fname)")
                MD._clear_map_leq_many_plan_cache!(A)
                println("  route(raw scalar-long batch): ", MD._map_leq_many_raw_route_kind(A, scalar_long_pairs, cc))
                arena = MD._map_leq_many_plan_arena(A)
                scalar_stats = MD._fill_map_leq_many_plan_arena!(arena, A, scalar_long_pairs, Q, cc)
                @assert MD._prefer_map_leq_many_scalar_long(scalar_stats, A.field)
                b_scalar_fresh = _bench("scalar-long executor fresh outputs", () -> begin
                    FF._clear_chain_parent_cache!(cc)
                    MD._clear_map_leq_memo!(A)
                    out = Vector{Matrix{K}}(undef, length(scalar_long_pairs))
                    MD._map_leq_many_scalar_long_batch!(out, A, arena, scalar_stats)
                    _map_leq_accumulate(out)
                end; reps=reps)
                scalar_out = Vector{Matrix{K}}(undef, length(scalar_long_pairs))
                MD._map_leq_many_scalar_long_batch!(scalar_out, A, arena, scalar_stats)
                b_scalar_reuse = _bench("scalar-long executor reused outputs", () -> begin
                    FF._clear_chain_parent_cache!(cc)
                    MD._clear_map_leq_memo!(A)
                    MD._map_leq_many_scalar_long_batch!(scalar_out, A, arena, scalar_stats)
                    _map_leq_accumulate(scalar_out)
                end; reps=reps)
                println("  speedup(reused/fresh) scalar executor: ", round(b_scalar_fresh.ms / b_scalar_reuse.ms, digits=2), "x")
                println("  alloc reduction(reused/fresh): ", round(b_scalar_fresh.kib / b_scalar_reuse.kib, digits=2), "x")
            end
        end

        # Additional overhead-focused map benchmark with 1x1 maps where matrix
        # arithmetic is trivial and lookup/chain logic dominates.
        if run_map
            U = _unit_pmodule(Q, field)
            edgeU = _edge_dict(U)
            unit_old = _map_leq_batch_old(U, edgeU, preds, pairs)
            unit_new = _map_leq_batch_new(U, cc, pairs)
            unit_many = _map_leq_batch_many(U, cc, pairs)
            unit_old == unit_new || error("map_leq unit parity failed for field=$(fname)")
            unit_new == unit_many || error("map_leq unit batch parity failed for field=$(fname)")
            b_old_map_unit = _bench("map_leq old overhead (1x1)", () -> _map_leq_batch_old(U, edgeU, preds, pairs); reps=reps)
            b_new_map_unit = _bench("map_leq new overhead (1x1)", () -> begin
                FF._clear_chain_parent_cache!(cc)
                MD._clear_map_leq_memo!(U)
                _map_leq_batch_new(U, cc, pairs)
            end; reps=reps)
            b_new_map_unit_many = _bench("map_leq batch overhead (1x1)", () -> begin
                FF._clear_chain_parent_cache!(cc)
                MD._clear_map_leq_memo!(U)
                _map_leq_batch_many(U, cc, pairs)
            end; reps=reps)
            println("  speedup(new/old) overhead: ", round(b_old_map_unit.ms / b_new_map_unit.ms, digits=2), "x")
            println("  speedup(batch/old) overhead: ", round(b_old_map_unit.ms / b_new_map_unit_many.ms, digits=2), "x")
        end

        if run_plan
            println("  -- map_leq_many compiled plan")
            prepared_pairs = MD.prepare_map_leq_batch(pairs)
            @assert _map_leq_batch_many_prepared(A, cc, prepared_pairs) == _map_leq_batch_many(A, cc, pairs)
            b_many_plan_off = _bench("map_leq_many (plan disabled)", () -> _with_map_many_plan_policy(; min_len=typemax(Int)) do
                FF._clear_chain_parent_cache!(cc)
                MD._clear_map_leq_memo!(A)
                MD._clear_map_leq_many_plan_cache!(A)
                _map_leq_batch_many(A, cc, pairs)
            end; reps=reps)
            b_many_plan_on_cold = _bench("map_leq_many (plan cold build)", () -> _with_map_many_plan_policy(; min_len=1) do
                FF._clear_chain_parent_cache!(cc)
                MD._clear_map_leq_memo!(A)
                MD._clear_map_leq_many_plan_cache!(A)
                _map_leq_batch_many(A, cc, pairs)
            end; reps=reps)
            _with_map_many_plan_policy(; min_len=1) do
                _map_leq_batch_many(A, cc, pairs)
            end
            b_many_plan_on_warm = _bench("map_leq_many (plan warm reuse)", () -> _with_map_many_plan_policy(; min_len=1) do
                _map_leq_batch_many(A, cc, pairs)
            end; reps=reps)
            b_many_prepared_cold = _bench("map_leq_many (prepared cold build)", () -> _with_map_many_plan_policy(; min_len=1) do
                FF._clear_chain_parent_cache!(cc)
                MD._clear_map_leq_memo!(A)
                MD._clear_map_leq_many_plan_cache!(A)
                _map_leq_batch_many_prepared(A, cc, prepared_pairs)
            end; reps=reps)
            _with_map_many_plan_policy(; min_len=1) do
                _map_leq_batch_many_prepared(A, cc, prepared_pairs)
            end
            b_many_prepared_warm = _bench("map_leq_many (prepared warm reuse)", () -> _with_map_many_plan_policy(; min_len=1) do
                _map_leq_batch_many_prepared(A, cc, prepared_pairs)
            end; reps=reps)
            prepared_out = Vector{Matrix{K}}(undef, length(prepared_pairs.pairs))
            b_many_prepared_prealloc = _bench("map_leq_many (prepared warm + prealloc)", () -> _with_map_many_plan_policy(; min_len=1) do
                _map_leq_batch_many_prepared_prealloc!(prepared_out, A, cc, prepared_pairs)
            end; reps=reps)
            println("  speedup(plan cold/disabled): ", round(b_many_plan_off.ms / b_many_plan_on_cold.ms, digits=2), "x")
            println("  speedup(plan warm/disabled): ", round(b_many_plan_off.ms / b_many_plan_on_warm.ms, digits=2), "x")
            println("  speedup(prepared warm/vector warm): ", round(b_many_plan_on_warm.ms / b_many_prepared_warm.ms, digits=2), "x")
            println("  speedup(prepared prealloc/vector warm): ", round(b_many_plan_on_warm.ms / b_many_prepared_prealloc.ms, digits=2), "x")
        end

        if run_dispatch
            println("  -- poset field dispatch (PModule.Q)")
            b_poset_old = _bench("Q access oldstyle (abstract field)", () -> _poset_scan_oldstyle(A, pairs); reps=reps)
            b_poset_new = _bench("Q access current (parametric field)", () -> _poset_scan_newstyle(A, pairs); reps=reps)
            println("  speedup(parametric/abstract): ", round(b_poset_old.ms / b_poset_new.ms, digits=2), "x")
        end

        # Dense-index map-compose memo table vs Dict memo.
        if run_memo
            println("  -- map_leq compose memo backend")
            A_dictmemo = _with_dense_memo_policy(; min_entries=typemax(Int), max_entries=0) do
                MD.PModule{K}(Q, A.dims, edgeA; field=field)
            end
            A_densememo = _with_dense_memo_policy(; min_entries=1, max_entries=10_000_000) do
                MD.PModule{K}(Q, A.dims, edgeA; field=field)
            end

            FF._clear_chain_parent_cache!(cc)
            MD._clear_map_leq_memo!(A_dictmemo)
            _map_leq_batch_new(A_dictmemo, cc, pairs)
            FF._clear_chain_parent_cache!(cc)
            MD._clear_map_leq_memo!(A_densememo)
            _map_leq_batch_new(A_densememo, cc, pairs)

            b_memo_dict = _bench("map_leq warm memo Dict", () -> _map_leq_batch_new(A_dictmemo, cc, pairs); reps=reps)
            b_memo_dense = _bench("map_leq warm memo dense", () -> _map_leq_batch_new(A_densememo, cc, pairs); reps=reps)
            println("  speedup(dense/dict): ", round(b_memo_dict.ms / b_memo_dense.ms, digits=2), "x")
        end

        # Item 2: identity-map cache (u == v path).
        if run_identity
            id_pairs = Tuple{Int,Int}[(v, v) for (_, v) in pairs]
            _map_leq_identity_old(A, id_pairs) == _map_leq_identity_new(A, cc, id_pairs) ||
                error("identity path parity failed for field=$(fname)")
            println("  -- map_leq identity path")
            b_id_old = _bench("map_leq identity old (eye alloc)", () -> _map_leq_identity_old(A, id_pairs); reps=reps)
            b_id_new = _bench("map_leq identity new (cached)", () -> _map_leq_identity_new(A, cc, id_pairs); reps=reps)
            println("  speedup(identity cache): ", round(b_id_old.ms / b_id_new.ms, digits=2), "x")
        end

        if run_directsum
            println("  -- direct_sum")
            b_old_sum = _bench("direct_sum old (dict keyed)", () -> _direct_sum_old(A, B, edgeA, edgeB); reps=reps)
            b_new_sum = _bench("direct_sum new (store aligned)", () -> MD.direct_sum(A, B); reps=reps)
            println("  speedup(new/old): ", round(b_old_sum.ms / b_new_sum.ms, digits=2), "x")

            sparse_edgeA = Dict{Tuple{Int,Int}, SparseMatrixCSC{K,Int}}()
            sparse_edgeB = Dict{Tuple{Int,Int}, SparseMatrixCSC{K,Int}}()
            sizehint!(sparse_edgeA, length(edgeA))
            sizehint!(sparse_edgeB, length(edgeB))
            for ((u, v), E) in edgeA
                sparse_edgeA[(u, v)] = sparse(E)
            end
            for ((u, v), E) in edgeB
                sparse_edgeB[(u, v)] = sparse(E)
            end
            As = MD.PModule{K}(Q, A.dims, sparse_edgeA; field=field)
            Bs = MD.PModule{K}(Q, B.dims, sparse_edgeB; field=field)
            println("  -- direct_sum (sparse maps)")
            b_old_sum_sparse = _bench("direct_sum old sparse (dict keyed)", () -> _direct_sum_old(As, Bs, _edge_dict(As), _edge_dict(Bs)); reps=reps)
            b_new_sum_sparse = _bench("direct_sum new sparse-aware", () -> MD.direct_sum(As, Bs); reps=reps)
            println("  speedup(new/old) sparse: ", round(b_old_sum_sparse.ms / b_new_sum_sparse.ms, digits=2), "x")

            println("  -- direct_sum route policy")
            b_policy_auto = _bench("direct_sum sparse auto route", () -> MD.direct_sum(As, Bs); reps=reps)
            b_policy_force_sparse = _bench("direct_sum forced sparse route", () -> _with_direct_sum_policy(; min_entries=1, qq=1.0, prime=1.0, real=1.0, qq_avg=0.0, prime_avg=0.0, real_avg=0.0) do
                MD.direct_sum(As, Bs)
            end; reps=reps)
            b_policy_force_dense = _bench("direct_sum forced dense route", () -> _with_direct_sum_policy(; min_entries=typemax(Int), qq=0.0, prime=0.0, real=0.0, qq_avg=typemax(Float64), prime_avg=typemax(Float64), real_avg=typemax(Float64)) do
                MD.direct_sum(As, Bs)
            end; reps=reps)
            println("  auto vs forced sparse: ", round(b_policy_force_sparse.ms / b_policy_auto.ms, digits=2), "x")
            println("  auto vs forced dense: ", round(b_policy_force_dense.ms / b_policy_auto.ms, digits=2), "x")

            b_route_old = _bench("direct_sum route decision old", () -> begin
                x = false
                @inbounds for _ in 1:2000
                    x = xor(x, _direct_sum_sparse_preferred_old_emulated(As, Bs))
                end
                x
            end; reps=reps)
            b_route_new = _bench("direct_sum route decision new", () -> begin
                x = false
                @inbounds for _ in 1:2000
                    x = xor(x, MD._direct_sum_sparse_preferred(As, Bs))
                end
                x
            end; reps=reps)
            println("  speedup(route new/old): ", round(b_route_old.ms / b_route_new.ms, digits=2), "x")
        end

        if run_ctor
            println("  -- pmodule construction")
            b_ctor_dict = _bench("PModule ctor from dict", () -> MD.PModule{K}(Q, A.dims, edgeA; field=field); reps=reps)
            b_ctor_store = _bench("PModule ctor from aligned store", () -> MD.PModule{K}(Q, A.dims, A.edge_maps; field=field); reps=reps)
            println("  speedup(store/dict): ", round(b_ctor_dict.ms / b_ctor_store.ms, digits=2), "x")
        end

        println()
    end
end

main()
