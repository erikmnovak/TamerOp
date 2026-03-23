#!/usr/bin/env julia
#
# indicator_resolutions_microbench.jl
#
# Purpose
# - Benchmark hot kernels in `IndicatorResolutions.jl` on deterministic synthetic fixtures.
# - Report median time and allocation for core entrypoints and cache behavior.
#
# Usage
#   julia --project=. benchmark/indicator_resolutions_microbench.jl
#   julia --project=. benchmark/indicator_resolutions_microbench.jl --reps=7 --nx=10 --ny=10 --maxlen=3 --field=f3
#

using Random
using SparseArrays

try
    using TamerOp
catch
    include(joinpath(@__DIR__, "..", "src", "TamerOp.jl"))
    using .TamerOp
end

const CM = TamerOp.CoreModules
const OPT = TamerOp.Options
const DT = TamerOp.DataTypes
const EC = TamerOp.EncodingCore
const RES = TamerOp.Results
const FF = TamerOp.FiniteFringe
const MD = TamerOp.Modules
const AB = TamerOp.AbelianCategories
const IR = TamerOp.IndicatorResolutions
const DF = TamerOp.DerivedFunctors

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

function _parse_string_arg(args, key::String, default::String)
    for a in args
        startswith(a, key * "=") || continue
        return lowercase(strip(split(a, "=", limit=2)[2]))
    end
    return default
end

function _parse_path_arg(args, key::String, default::String)
    for a in args
        startswith(a, key * "=") || continue
        return String(strip(split(a, "=", limit=2)[2]))
    end
    return default
end

function _parse_bool_arg(args, key::String, default::Bool)
    for a in args
        startswith(a, key * "=") || continue
        v = lowercase(strip(split(a, "=", limit=2)[2]))
        if v in ("1", "true", "yes", "y", "on")
            return true
        elseif v in ("0", "false", "no", "n", "off")
            return false
        else
            error("invalid boolean for $key: $v")
        end
    end
    return default
end

function _section_enabled(section::String, group::String)
    section == "all" && return true
    return section == group
end

function _bench(name::AbstractString, f::Function; reps::Int=7, setup::Union{Nothing,Function}=nothing)
    GC.gc()
    setup === nothing || setup()
    f() # warmup
    GC.gc()
    times_ms = Vector{Float64}(undef, reps)
    bytes = Vector{Int}(undef, reps)
    for i in 1:reps
        setup === nothing || setup()
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
    return (probe=String(name), median_ms=med_ms, median_kib=med_kib)
end

function _field_from_name(name::String)
    name == "qq" && return CM.QQField()
    name == "f2" && return CM.F2()
    name == "f3" && return CM.F3()
    name == "f5" && return CM.Fp(5)
    error("unknown field '$name' (supported: qq,f2,f3,f5)")
end

function _rand_coeff(rng::AbstractRNG, field::CM.AbstractCoeffField)
    v = rand(rng, -3:3)
    v == 0 && (v = 1)
    return CM.coerce(field, v)
end

function _random_pmodule(Q::FF.AbstractPoset, field::CM.AbstractCoeffField;
                         dmin::Int=1, dmax::Int=4, density::Float64=0.35, seed::Int=0xA0A1)
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

function _grid_finite_poset(nx::Int, ny::Int)
    (nx >= 1 && ny >= 1) || error("_grid_finite_poset: nx, ny must be >= 1")
    n = nx * ny
    rel = falses(n, n)
    @inline idx(ix, iy) = (iy - 1) * nx + ix
    @inbounds for y1 in 1:ny, x1 in 1:nx
        i = idx(x1, y1)
        for y2 in y1:ny, x2 in x1:nx
            j = idx(x2, y2)
            rel[i, j] = true
        end
    end
    return FF.FinitePoset(rel; check=false)
end

function _fan_cover_poset(width::Int)
    width >= 2 || error("_fan_cover_poset: width must be >= 2")
    n = width + 2
    rel = falses(n, n)
    @inbounds begin
        for i in 1:n
            rel[i, i] = true
        end
        top = n
        for mid in 2:(n - 1)
            rel[1, mid] = true
            rel[mid, top] = true
            rel[1, top] = true
        end
    end
    return FF.FinitePoset(rel; check=false)
end

function _random_fringe(Q::FF.AbstractPoset, field::CM.AbstractCoeffField;
                        nups::Int=24, ndowns::Int=24, density::Float64=0.4, seed::Int=0xB1B1)
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
        # For principal upset/downset summands, monomial condition requires
        # Up(up_verts[i]) intersect Down(down_verts[j]) is nonempty, i.e. up_verts[i] <= down_verts[j].
        FF.leq(Q, up_verts[i], down_verts[j]) || continue
        rand(rng) < density || continue
        phi[j, i] = _rand_coeff(rng, field)
    end

    return FF.FringeModule{K}(Q, U, D, phi; field=field)
end

@inline _digest_pmodule(M::MD.PModule) = sum(M.dims)

@inline function _digest_resolution_pair(F, dF)
    acc = length(F) + length(dF)
    @inbounds for A in dF
        acc += size(A, 1) + size(A, 2) + nnz(A)
    end
    return acc
end

@inline function _digest_upset_presentation(F)
    return length(F.U0) + length(F.U1) + size(F.delta, 1) + size(F.delta, 2) + nnz(F.delta)
end

@inline function _digest_downset_copresentation(E)
    return length(E.D0) + length(E.D1) + size(E.rho, 1) + size(E.rho, 2) + nnz(E.rho)
end

@inline function _digest_fringe(H)
    return length(H.U) + length(H.D) + size(H.phi, 1) + size(H.phi, 2) + nnz(H.phi)
end

function _write_csv(path::AbstractString, rows)
    open(path, "w") do io
        println(io, "probe,median_ms,median_kib")
        for r in rows
            println(io, string(r.probe, ",", r.median_ms, ",", r.median_kib))
        end
    end
end

function _with_ref!(f::Function, ref, val)
    old = ref[]
    ref[] = val
    try
        return f()
    finally
        ref[] = old
    end
end

function _clear_indicator_bench_state!(mods::MD.PModule...)
    IR._clear_indicator_prefix_caches!()
    @inbounds for M in mods
        MD._clear_map_leq_memo!(M)
        MD._clear_map_leq_many_plan_cache!(M)
        FF._clear_chain_parent_cache!(MD._get_cover_cache(M.Q))
    end
    return nothing
end

function main(; reps::Int=7, nx::Int=8, ny::Int=8, maxlen::Int=3,
              density::Float64=0.35, field_name::String="qq",
              verify::Bool=false,
              section::String="all",
              threads::Bool=(Threads.nthreads() > 1),
              out::String=joinpath(@__DIR__, "_tmp_indicator_resolutions_microbench.csv"))
    field = _field_from_name(field_name)
    K = CM.coeff_type(field)
    Q = _grid_finite_poset(nx, ny)
    FF.build_cache!(Q; cover=true, updown=true)

    M = _random_pmodule(Q, field; density=density, seed=Int(0xA0A1))
    res_opts = OPT.ResolutionOptions(maxlen=maxlen)
    Hm = _random_fringe(Q, field; density=0.40, seed=Int(0xB1B1))
    Hn = _random_fringe(Q, field; density=0.45, seed=Int(0xB2B2))
    Mm = IR.pmodule_from_fringe(Hm)
    Mn = IR.pmodule_from_fringe(Hn)
    Qfan = _fan_cover_poset(12)
    FF.build_cache!(Qfan; cover=true, updown=true)
    Mfan = _random_pmodule(Qfan, field; dmin=3, dmax=5, density=density, seed=Int(0xC3C3))
    fan_top = FF.nvertices(Qfan)
    rc = CM.ResolutionCache()
    strict_setup_M = () -> _clear_indicator_bench_state!(M)
    strict_setup_fan = () -> _clear_indicator_bench_state!(Mfan)
    strict_setup_indicator = () -> begin
        _clear_indicator_bench_state!(Mm, Mn)
        CM._clear_resolution_cache!(rc)
    end

    println("IndicatorResolutions micro-benchmark")
    println("reps=$(reps), grid=($(nx),$(ny)), maxlen=$(maxlen), field=$(field_name), threads=$(threads), section=$(section)")
    println("timing_policy=strict-cold except explicit cache-hit probes")
    println("upset_auto_profile=", IR._indicator_upset_auto_profile(M; maxlen=maxlen))
    println("downset_auto_profile=", IR._indicator_downset_auto_profile(M; maxlen=maxlen))

    rows = NamedTuple[]

    if _section_enabled(section, "core")
        # Warm the structural plan cache once, then measure the steady-state hit path.
        IR._upset_birth_block_plan(Q)
        IR._downset_birth_block_plan(Q)
        push!(rows, _bench("ir upset_birth_plan cache_hit", () -> begin
            sum(length, IR._upset_birth_block_plan(Q))
        end; reps=reps))
        push!(rows, _bench("ir downset_birth_plan cache_hit", () -> begin
            sum(length, IR._downset_birth_block_plan(Q))
        end; reps=reps))

        push!(rows, _bench("ir pmodule_from_fringe", () -> begin
            _digest_pmodule(IR.pmodule_from_fringe(Hm))
        end; reps=reps))

        push!(rows, _bench("ir pmodule_from_fringe forced_dict", () -> begin
            _with_ref!(IR.INDICATOR_PMODULE_DIRECT_STORE_MIN_EDGES, typemax(Int)) do
                _with_ref!(IR.INDICATOR_PMODULE_DIRECT_STORE_MIN_WORK, typemax(Int)) do
                    _digest_pmodule(IR.pmodule_from_fringe(Hm))
                end
            end
        end; reps=reps))

        push!(rows, _bench("ir pmodule_from_fringe forced_store", () -> begin
            _with_ref!(IR.INDICATOR_PMODULE_DIRECT_STORE_MIN_EDGES, 0) do
                _with_ref!(IR.INDICATOR_PMODULE_DIRECT_STORE_MIN_WORK, 0) do
                    _digest_pmodule(IR.pmodule_from_fringe(Hm))
                end
            end
        end; reps=reps))

        push!(rows, _bench("ir incoming_image_basis forced_dense", () -> begin
            _with_ref!(IR.INDICATOR_INCREMENTAL_LINALG, false) do
                size(IR._incoming_image_basis(Mfan, fan_top), 2)
            end
        end; reps=reps, setup=strict_setup_fan))

        push!(rows, _bench("ir incoming_image_basis forced_incremental", () -> begin
            _with_ref!(IR.INDICATOR_INCREMENTAL_LINALG, true) do
                _with_ref!(IR.INDICATOR_INCREMENTAL_LINALG_MIN_MAPS, 1) do
                    _with_ref!(IR.INDICATOR_INCREMENTAL_LINALG_MIN_ENTRIES, 0) do
                        size(IR._incoming_image_basis(Mfan, fan_top), 2)
                    end
                end
            end
        end; reps=reps, setup=strict_setup_fan))

        push!(rows, _bench("ir outgoing_span_basis forced_dense", () -> begin
            _with_ref!(IR.INDICATOR_DOWNSET_INCREMENTAL_LINALG_MIN_MAPS, typemax(Int)) do
                _with_ref!(IR.INDICATOR_DOWNSET_INCREMENTAL_LINALG_MIN_ENTRIES, typemax(Int)) do
                size(IR._outgoing_span_basis(Mfan, 1), 1)
                end
            end
        end; reps=reps, setup=strict_setup_fan))

        push!(rows, _bench("ir outgoing_span_basis forced_incremental", () -> begin
            _with_ref!(IR.INDICATOR_DOWNSET_INCREMENTAL_LINALG_MIN_MAPS, 1) do
                _with_ref!(IR.INDICATOR_DOWNSET_INCREMENTAL_LINALG_MIN_ENTRIES, 0) do
                    size(IR._outgoing_span_basis(Mfan, 1), 1)
                end
            end
        end; reps=reps, setup=strict_setup_fan))

        push!(rows, _bench("ir projective_cover", () -> begin
            F0, pi0, gens = IR.projective_cover(M; threads=threads)
            sum(F0.dims) + length(pi0.comps) + sum(length(g) for g in gens)
        end; reps=reps, setup=strict_setup_M))

        push!(rows, _bench("ir projective_cover packed_plan", () -> begin
            F0, pi0, plan = IR.projective_cover(M; materialize_gens=false, threads=threads)
            sum(F0.dims) + length(pi0.comps) + plan.total
        end; reps=reps, setup=strict_setup_M))

        push!(rows, _bench("ir upset_presentation_one_step", () -> begin
            _digest_upset_presentation(IR.upset_presentation_one_step(Hm))
        end; reps=reps))
    end

    if _section_enabled(section, "resolutions")
        push!(rows, _bench("ir upset_resolution pmodule", () -> begin
            F, dF = IR.upset_resolution(M; maxlen=maxlen, threads=threads)
            _digest_resolution_pair(F, dF)
        end; reps=reps, setup=strict_setup_M))

        push!(rows, _bench("ir upset_resolution pmodule birth_plan_delta", () -> begin
            _with_ref!(IR._INDICATOR_UPSET_ACTIVE_SOURCE_DELTA, false) do
                F, dF = IR.upset_resolution(M; maxlen=maxlen, threads=threads)
                _digest_resolution_pair(F, dF)
            end
        end; reps=reps, setup=strict_setup_M))

        push!(rows, _bench("ir upset_resolution pmodule no_vertex_incremental_cache", () -> begin
            _with_ref!(IR.INDICATOR_INCREMENTAL_VERTEX_CACHE, false) do
                F, dF = IR.upset_resolution(M; maxlen=maxlen, threads=threads)
                _digest_resolution_pair(F, dF)
            end
        end; reps=reps, setup=strict_setup_M))

        push!(rows, _bench("df projective_resolution pmodule", () -> begin
            R = DF.projective_resolution(M, res_opts; threads=threads)
            length(R.Pmods) + length(R.d_mor) + length(R.d_mat)
        end; reps=reps, setup=strict_setup_M))

        push!(rows, _bench("df injective_resolution pmodule", () -> begin
            R = DF.injective_resolution(M, res_opts; threads=threads)
            length(R.Emods) + length(R.d_mor)
        end; reps=reps, setup=strict_setup_M))
    end

    if maxlen >= 2 && _section_enabled(section, "resolutions")
        push!(rows, _bench("ir upset_resolution maxlen2 no_prefix", () -> begin
            old = IR.INDICATOR_PREFIX_CACHE_ENABLED[]
            IR.INDICATOR_PREFIX_CACHE_ENABLED[] = false
            try
                F, dF = IR.upset_resolution(M; maxlen=2, threads=threads)
                _digest_resolution_pair(F, dF)
            finally
                IR.INDICATOR_PREFIX_CACHE_ENABLED[] = old
            end
        end; reps=reps, setup=strict_setup_M))

        push!(rows, _bench("ir upset_resolution maxlen2 after_maxlen1_prefix", () -> begin
            old = IR.INDICATOR_PREFIX_CACHE_ENABLED[]
            IR.INDICATOR_PREFIX_CACHE_ENABLED[] = true
            try
                IR._clear_indicator_prefix_caches!()
                IR.upset_resolution(M; maxlen=1, threads=threads)
                F, dF = IR.upset_resolution(M; maxlen=2, threads=threads)
                _digest_resolution_pair(F, dF)
            finally
                IR.INDICATOR_PREFIX_CACHE_ENABLED[] = old
            end
        end; reps=reps, setup=strict_setup_M))

        push!(rows, _bench("ir upset_resolution maxlen2 no_incremental", () -> begin
            old_inc = IR.INDICATOR_INCREMENTAL_LINALG[]
            old_vertex = IR.INDICATOR_INCREMENTAL_VERTEX_CACHE[]
            old_pref = IR.INDICATOR_PREFIX_CACHE_ENABLED[]
            IR.INDICATOR_INCREMENTAL_LINALG[] = false
            IR.INDICATOR_INCREMENTAL_VERTEX_CACHE[] = false
            IR.INDICATOR_PREFIX_CACHE_ENABLED[] = false
            try
                F, dF = IR.upset_resolution(M; maxlen=2, threads=threads)
                _digest_resolution_pair(F, dF)
            finally
                IR.INDICATOR_INCREMENTAL_LINALG[] = old_inc
                IR.INDICATOR_INCREMENTAL_VERTEX_CACHE[] = old_vertex
                IR.INDICATOR_PREFIX_CACHE_ENABLED[] = old_pref
            end
        end; reps=reps, setup=strict_setup_M))

        push!(rows, _bench("ir upset_resolution maxlen2 birth_plan_delta", () -> begin
            _with_ref!(IR._INDICATOR_UPSET_ACTIVE_SOURCE_DELTA, false) do
                F, dF = IR.upset_resolution(M; maxlen=2, threads=threads)
                _digest_resolution_pair(F, dF)
            end
        end; reps=reps, setup=strict_setup_M))

    end

    if _section_enabled(section, "downset")
        push!(rows, _bench("ir injective_hull", () -> begin
            E0, iota, gens = IR.injective_hull(M; threads=threads)
            sum(E0.dims) + length(iota.comps) + sum(length(g) for g in gens)
        end; reps=reps, setup=strict_setup_M))

        push!(rows, _bench("ir injective_hull packed_plan", () -> begin
            E0, iota, plan = IR._injective_hull(M; materialize_gens=false, threads=threads)
            sum(E0.dims) + length(iota.comps) + plan.total
        end; reps=reps, setup=strict_setup_M))

        push!(rows, _bench("ir downset_copresentation_one_step", () -> begin
            _digest_downset_copresentation(IR.downset_copresentation_one_step(Hm))
        end; reps=reps))

        push!(rows, _bench("ir fringe_presentation", () -> begin
            _digest_fringe(IR.fringe_presentation(Mm))
        end; reps=reps))

        stage1_cokernel_ref = Ref{Any}(nothing)
        stage1_cache_ref = Ref{Any}(nothing)
        stage1_workspace_ref = Ref{Any}(nothing)
        stage1_memo_ref = Ref{Any}(nothing)
        stage1_support_ref = Ref(Vector{Int}())
        stage_graph_lists_ref = Ref{Any}(nothing)
        stage1_hull_setup = () -> begin
            strict_setup_M()
            cc = MD._get_cover_cache(M.Q)
            graph_lists = AB._cover_graph_lists(cc)
            memo_pool = IdDict{Any,Vector{Union{Nothing,Matrix{K}}}}()
            map_memo_local = IR._indicator_memo_for_module!(memo_pool, M)
            ws = IR._new_resolution_workspace(K, FF.nvertices(M.Q))
            IR._workspace_prepare!(ws, FF.nvertices(M.Q))
            initial_support = IR._nonzero_dim_vertices(M.dims)
            E0, iota0, _ = IR._injective_hull(
                M;
                cache=cc,
                map_memo=map_memo_local,
                workspace=ws,
                support_vertices=initial_support,
                graph_lists=graph_lists,
                threads=threads,
            )
            C0, _ = AB._cokernel_module(
                iota0;
                cache=cc,
                active_vertices=initial_support,
                graph_lists=graph_lists,
            )
            stage1_cokernel_ref[] = C0
            stage1_cache_ref[] = cc
            stage1_workspace_ref[] = ws
            stage1_memo_ref[] = IR._indicator_memo_for_module!(memo_pool, C0)
            stage1_support_ref[] = IR._nonzero_dim_vertices(C0.dims)
            stage_graph_lists_ref[] = graph_lists
            return nothing
        end
        push!(rows, _bench("ir downset stage1 _injective_hull", () -> begin
            C0 = stage1_cokernel_ref[]::MD.PModule{K}
            cc = stage1_cache_ref[]::MD.CoverCache
            ws = stage1_workspace_ref[]::IR._ResolutionWorkspace{K}
            map_memo = stage1_memo_ref[]::Vector{Union{Nothing,Matrix{K}}}
            support = stage1_support_ref[]
            graph_lists = stage_graph_lists_ref[]::AB._CoverGraphLists
            E1, j, gens = IR._injective_hull(
                C0;
                cache=cc,
                map_memo=map_memo,
                workspace=ws,
                support_vertices=support,
                graph_lists=graph_lists,
                threads=threads,
            )
            sum(E1.dims) + length(j.comps) + sum(length(g) for g in gens)
        end; reps=reps, setup=stage1_hull_setup))

        push!(rows, _bench("ir downset stage1 _injective_hull packed_plan_forced", () -> begin
            old_min_socle = IR._INDICATOR_INJECTIVE_PACKED_PLAN_MIN_SOCLE_VERTICES[]
            old_min_gens = IR._INDICATOR_INJECTIVE_PACKED_PLAN_MIN_TOTAL_GENS[]
            old_min_hull = IR._INDICATOR_INJECTIVE_PACKED_PLAN_MIN_TOTAL_HULL_DIMS[]
            try
                IR._INDICATOR_INJECTIVE_PACKED_PLAN_MIN_SOCLE_VERTICES[] = 0
                IR._INDICATOR_INJECTIVE_PACKED_PLAN_MIN_TOTAL_GENS[] = 0
                IR._INDICATOR_INJECTIVE_PACKED_PLAN_MIN_TOTAL_HULL_DIMS[] = 0
                C0 = stage1_cokernel_ref[]::MD.PModule{K}
                cc = stage1_cache_ref[]::MD.CoverCache
                ws = stage1_workspace_ref[]::IR._ResolutionWorkspace{K}
                map_memo = stage1_memo_ref[]::Vector{Union{Nothing,Matrix{K}}}
                support = stage1_support_ref[]
                graph_lists = stage_graph_lists_ref[]::AB._CoverGraphLists
                E1, j, gens = IR._injective_hull(
                    C0;
                    cache=cc,
                    map_memo=map_memo,
                    workspace=ws,
                    support_vertices=support,
                    graph_lists=graph_lists,
                    threads=threads,
                )
                sum(E1.dims) + length(j.comps) + sum(length(g) for g in gens)
            finally
                IR._INDICATOR_INJECTIVE_PACKED_PLAN_MIN_SOCLE_VERTICES[] = old_min_socle
                IR._INDICATOR_INJECTIVE_PACKED_PLAN_MIN_TOTAL_GENS[] = old_min_gens
                IR._INDICATOR_INJECTIVE_PACKED_PLAN_MIN_TOTAL_HULL_DIMS[] = old_min_hull
            end
        end; reps=reps, setup=stage1_hull_setup))

        stage1_morphism_ref = Ref{Any}(nothing)
        stage1_cokernel_setup = () -> begin
            stage1_hull_setup()
            C0 = stage1_cokernel_ref[]::MD.PModule{K}
            cc = stage1_cache_ref[]::MD.CoverCache
            ws = stage1_workspace_ref[]::IR._ResolutionWorkspace{K}
            map_memo = stage1_memo_ref[]::Vector{Union{Nothing,Matrix{K}}}
            support = stage1_support_ref[]
            graph_lists = stage_graph_lists_ref[]::AB._CoverGraphLists
            _, j, _ = IR._injective_hull(
                C0;
                cache=cc,
                map_memo=map_memo,
                workspace=ws,
                support_vertices=support,
                graph_lists=graph_lists,
                threads=threads,
            )
            stage1_morphism_ref[] = j
            return nothing
        end
        push!(rows, _bench("ir downset stage1 _cokernel_module", () -> begin
            j = stage1_morphism_ref[]::MD.PMorphism{K}
            cc = stage1_cache_ref[]::MD.CoverCache
            graph_lists = stage_graph_lists_ref[]::AB._CoverGraphLists
            C1, q1 = AB._cokernel_module(j; cache=cc, active_vertices=stage1_support_ref[], graph_lists=graph_lists)
            _digest_pmodule(C1) + length(q1.comps)
        end; reps=reps, setup=stage1_cokernel_setup))

        stage2_cokernel_ref = Ref{Any}(nothing)
        stage2_memo_ref = Ref{Any}(nothing)
        stage2_support_ref = Ref(Vector{Int}())
        stage2_hull_setup = () -> begin
            stage1_cokernel_setup()
            j = stage1_morphism_ref[]::MD.PMorphism{K}
            cc = stage1_cache_ref[]::MD.CoverCache
            graph_lists = stage_graph_lists_ref[]::AB._CoverGraphLists
            C1, _ = AB._cokernel_module(j; cache=cc, active_vertices=stage1_support_ref[], graph_lists=graph_lists)
            stage2_cokernel_ref[] = C1
            memo_pool = IdDict{Any,Vector{Union{Nothing,Matrix{K}}}}()
            stage2_memo_ref[] = IR._indicator_memo_for_module!(memo_pool, C1)
            stage2_support_ref[] = IR._nonzero_dim_vertices(C1.dims)
            return nothing
        end

        push!(rows, _bench("ir downset stage2 _injective_hull", () -> begin
            C1 = stage2_cokernel_ref[]::MD.PModule{K}
            cc = stage1_cache_ref[]::MD.CoverCache
            ws = stage1_workspace_ref[]::IR._ResolutionWorkspace{K}
            map_memo = stage2_memo_ref[]::Vector{Union{Nothing,Matrix{K}}}
            support = stage2_support_ref[]
            graph_lists = stage_graph_lists_ref[]::AB._CoverGraphLists
            E2, k, gens = IR._injective_hull(
                C1;
                cache=cc,
                map_memo=map_memo,
                workspace=ws,
                support_vertices=support,
                graph_lists=graph_lists,
                threads=threads,
            )
            sum(E2.dims) + length(k.comps) + sum(length(g) for g in gens)
        end; reps=reps, setup=stage2_hull_setup))

        stage2_morphism_ref = Ref{Any}(nothing)
        stage2_cokernel_setup = () -> begin
            stage2_hull_setup()
            C1 = stage2_cokernel_ref[]::MD.PModule{K}
            cc = stage1_cache_ref[]::MD.CoverCache
            ws = stage1_workspace_ref[]::IR._ResolutionWorkspace{K}
            map_memo = stage2_memo_ref[]::Vector{Union{Nothing,Matrix{K}}}
            support = stage2_support_ref[]
            graph_lists = stage_graph_lists_ref[]::AB._CoverGraphLists
            _, k, _ = IR._injective_hull(
                C1;
                cache=cc,
                map_memo=map_memo,
                workspace=ws,
                support_vertices=support,
                graph_lists=graph_lists,
                threads=threads,
            )
            stage2_morphism_ref[] = k
            return nothing
        end

        push!(rows, _bench("ir downset stage2 _cokernel_module", () -> begin
            k = stage2_morphism_ref[]::MD.PMorphism{K}
            cc = stage1_cache_ref[]::MD.CoverCache
            graph_lists = stage_graph_lists_ref[]::AB._CoverGraphLists
            C2, q2 = AB._cokernel_module(k; cache=cc, active_vertices=stage2_support_ref[], graph_lists=graph_lists)
            _digest_pmodule(C2) + length(q2.comps)
        end; reps=reps, setup=stage2_cokernel_setup))

        push!(rows, _bench("ir downset_resolution pmodule", () -> begin
            E, dE = IR.downset_resolution(M; maxlen=maxlen, threads=threads)
            _digest_resolution_pair(E, dE)
        end; reps=reps, setup=strict_setup_M))

        if maxlen >= 2
            push!(rows, _bench("ir downset_resolution maxlen2 no_incremental", () -> begin
                _with_ref!(IR.INDICATOR_DOWNSET_INCREMENTAL_LINALG_MIN_MAPS, typemax(Int)) do
                    _with_ref!(IR.INDICATOR_DOWNSET_INCREMENTAL_LINALG_MIN_ENTRIES, typemax(Int)) do
                        E, dE = IR.downset_resolution(M; maxlen=2, threads=threads)
                        _digest_resolution_pair(E, dE)
                    end
                end
            end; reps=reps, setup=strict_setup_M))
        end
    end

    if _section_enabled(section, "cache")
        push!(rows, _bench("ir indicator_resolutions uncached_compute", () -> begin
            F, dF, E, dE = IR._indicator_resolutions_from_pmodules(Mm, Mn; maxlen=maxlen, threads=threads)
            _digest_resolution_pair(F, dF) + _digest_resolution_pair(E, dE)
        end; reps=reps, setup=strict_setup_indicator))

        push!(rows, _bench("ir indicator_resolutions uncached", () -> begin
            F, dF, E, dE = IR.indicator_resolutions(Hm, Hn; maxlen=maxlen, threads=threads)
            _digest_resolution_pair(F, dF) + _digest_resolution_pair(E, dE)
        end; reps=reps, setup=() -> begin
            IR._clear_indicator_prefix_caches!()
            CM._clear_resolution_cache!(rc)
        end))

        PT = typeof(P)
        UP = TamerOp.IndicatorTypes.UpsetPresentation{K,PT,Nothing,SparseMatrixCSC{K,Int}}
        DP = TamerOp.IndicatorTypes.DownsetCopresentation{K,PT,Nothing,SparseMatrixCSC{K,Int}}
        cache_val_type = Tuple{
            Vector{UP},
            Vector{SparseMatrixCSC{K,Int}},
            Vector{DP},
            Vector{SparseMatrixCSC{K,Int}},
        }
        maxkey = maxlen === nothing ? -1 : Int(maxlen)
        key = CM._resolution_key3(Hm, Hn, maxkey)
        push!(rows, _bench("ir indicator_cache lookup_miss", () -> begin
            IR._resolution_cache_indicator_get(rc, key, cache_val_type)
        end; reps=reps, setup=() -> CM._clear_resolution_cache!(rc)))

        payload = IR.indicator_resolutions(Hm, Hn; maxlen=maxlen, threads=threads)
        push!(rows, _bench("ir indicator_cache store_only", () -> begin
            rc_local = CM.ResolutionCache()
            IR._resolution_cache_indicator_store!(rc_local, key, payload)
            IR._resolution_cache_indicator_store!(rc_local, key, payload)
        end; reps=reps))

        push!(rows, _bench("ir indicator_resolutions cache_miss_fresh", () -> begin
            rc_local = CM.ResolutionCache()
            F, dF, E, dE = IR._indicator_resolutions_from_pmodules(Mm, Mn; maxlen=maxlen, threads=threads)
            IR._resolution_cache_indicator_store!(rc_local, key, (F, dF, E, dE))
            _digest_resolution_pair(F, dF) + _digest_resolution_pair(E, dE)
        end; reps=reps, setup=strict_setup_indicator))

        push!(rows, _bench("ir indicator_resolutions cache_miss_after_clear", () -> begin
            F, dF, E, dE = IR._indicator_resolutions_from_pmodules(Mm, Mn; maxlen=maxlen, threads=threads)
            IR._resolution_cache_indicator_store!(rc, key, (F, dF, E, dE))
            _digest_resolution_pair(F, dF) + _digest_resolution_pair(E, dE)
        end; reps=reps, setup=strict_setup_indicator))

        # Warm cache hit path should measure mostly lookup/return overhead.
        payload_hit = IR._indicator_resolutions_from_pmodules(Mm, Mn; maxlen=maxlen, threads=threads)
        IR._resolution_cache_indicator_store!(rc, key, payload_hit)
        IR._resolution_cache_indicator_store!(rc, key, payload_hit)
        push!(rows, _bench("ir indicator_resolutions cache_hit", () -> begin
            F, dF, E, dE = IR._resolution_cache_indicator_get(rc, key, cache_val_type)
            _digest_resolution_pair(F, dF) + _digest_resolution_pair(E, dE)
        end; reps=reps))
    end

    if verify && _section_enabled(section, "verify")
        Fv, dFv = IR.upset_resolution(M; maxlen=maxlen, threads=threads)
        Ev, dEv = IR.downset_resolution(M; maxlen=maxlen, threads=threads)
        push!(rows, _bench("ir verify_upset_resolution", () -> begin
            IR.verify_upset_resolution(Fv, dFv; check_d2=true, check_exactness=true, check_connected=true)
        end; reps=reps))
        push!(rows, _bench("ir verify_downset_resolution", () -> begin
            IR.verify_downset_resolution(Ev, dEv; check_d2=true, check_exactness=true, check_connected=true)
        end; reps=reps))
    end

    _write_csv(out, rows)
    uncached_compute_ms = nothing
    miss_clear_ms = nothing
    for r in rows
        if r.probe == "ir indicator_resolutions uncached_compute"
            uncached_compute_ms = r.median_ms
        elseif r.probe == "ir indicator_resolutions cache_miss_after_clear"
            miss_clear_ms = r.median_ms
        end
    end
    if uncached_compute_ms !== nothing && miss_clear_ms !== nothing && miss_clear_ms > 1.10 * uncached_compute_ms
        println("GUARD WARN: cache_miss_after_clear > 1.10 * uncached_compute.")
        println("            miss-lifecycle/cache plumbing overhead is still too high.")
    end
    println("Wrote ", out)
end

if abspath(PROGRAM_FILE) == @__FILE__
    reps = _parse_int_arg(ARGS, "--reps", 7)
    nx = _parse_int_arg(ARGS, "--nx", 8)
    ny = _parse_int_arg(ARGS, "--ny", 8)
    maxlen = _parse_int_arg(ARGS, "--maxlen", 3)
    density = _parse_float_arg(ARGS, "--density", 0.35)
    field_name = _parse_string_arg(ARGS, "--field", "qq")
    section = _parse_string_arg(ARGS, "--section", "all")
    verify = _parse_bool_arg(ARGS, "--verify", false)
    threads = _parse_bool_arg(ARGS, "--threads", Threads.nthreads() > 1)
    out = _parse_path_arg(ARGS, "--out", joinpath(@__DIR__, "_tmp_indicator_resolutions_microbench.csv"))
    main(; reps=reps, nx=nx, ny=ny, maxlen=maxlen, density=density, field_name=field_name, verify=verify, section=section, threads=threads, out=out)
end
