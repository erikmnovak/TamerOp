#!/usr/bin/env julia
#
# finitefringe_microbench.jl
#
# Purpose
# - Benchmark core FiniteFringe kernels where recent optimizations were applied:
#   1) Uptight region partitioning (`_uptight_regions`)
#   2) Model-independent Hom dimension assembly (`hom_dimension`)
#
# What this compares
# - For each kernel, this script runs:
#   - a naive/baseline implementation (local copy of the previous algorithmic shape),
#   - the current optimized library implementation (including `hom_dimension`
#     path variants `:sparse_path`, `:dense_idx_internal`, and production `:auto`).
# - It also includes a targeted cold-start route-selection block comparing:
#   - heuristic-first `:auto` route selection,
#   - timing-only baseline route selection (`_hom_dimension_auto_timing_baseline`).
#
# Bench methodology
# - Deterministic synthetic fixtures.
# - Warmup + repeated timed runs.
# - Median wall-time and median allocation reported.
# - Functional parity checks (old vs new) before timing.
#
# Usage
#   julia --project=. benchmark/finitefringe_microbench.jl
#   julia --project=. benchmark/finitefringe_microbench.jl --reps=9 --n=180 --y=96 --nu=24 --nd=24
#   julia --project=. benchmark/finitefringe_microbench.jl --reps=9 --n=180 --y_cases=32,96,160
#   julia --project=. benchmark/finitefringe_microbench.jl --hom_dense_case=true
#   julia --project=. benchmark/finitefringe_microbench.jl --fiber_queries=2000
#

using Random
using SparseArrays

try
    using TamerOp
catch
    include(joinpath(@__DIR__, "..", "src", "TamerOp.jl"))
    using .TamerOp
end

const TO = TamerOp.Advanced
const FF = TO.FiniteFringe
const EN = TO.Encoding
const CM = TO.CoreModules
const FL = TamerOp.FieldLinAlg

function _parse_int_arg(args, key::String, default::Int)
    for a in args
        startswith(a, key * "=") || continue
        return max(1, parse(Int, split(a, "=", limit=2)[2]))
    end
    return default
end

function _parse_int_list_arg(args, key::String, default::Vector{Int})
    for a in args
        startswith(a, key * "=") || continue
        raw = split(a, "=", limit=2)[2]
        items = split(raw, ",")
        vals = Int[]
        for it in items
            s = strip(it)
            isempty(s) && continue
            push!(vals, max(1, parse(Int, s)))
        end
        return isempty(vals) ? default : vals
    end
    return default
end

function _parse_bool_arg(args, key::String, default::Bool)
    for a in args
        startswith(a, key * "=") || continue
        raw = lowercase(strip(split(a, "=", limit=2)[2]))
        if raw in ("1", "true", "yes", "on")
            return true
        elseif raw in ("0", "false", "no", "off")
            return false
        end
    end
    return default
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
    println(rpad(name, 42), " median_time=", round(med_ms, digits=3),
            " ms  median_alloc=", round(med_kib, digits=1), " KiB")
    return (ms=med_ms, kib=med_kib)
end

function _random_poset(n::Int; p::Float64=0.03, seed::Int=0xF1F1)
    rng = Random.MersenneTwister(seed)
    leq = falses(n, n)
    @inbounds for i in 1:n
        leq[i, i] = true
        for j in (i + 1):n
            leq[i, j] = rand(rng) < p
        end
    end
    # Boolean transitive closure.
    @inbounds for k in 1:n, i in 1:n, j in 1:n
        leq[i, j] = leq[i, j] || (leq[i, k] && leq[k, j])
    end
    return FF.FinitePoset(leq; check=false)
end

function _fixture_uptight(P::FF.AbstractPoset, ny::Int; seed::Int=0x5151)
    rng = Random.MersenneTwister(seed)
    n = TO.nvertices(P)
    Y = Vector{FF.Upset}(undef, ny)
    @inbounds for i in 1:ny
        mask = BitVector(rand(rng, Bool, n))
        Y[i] = FF.upset_closure(P, mask)
    end
    return Y
end

function _uptight_regions_old(Q::FF.AbstractPoset, Y::Vector{FF.Upset})
    sigs = Dict{Tuple{Vararg{Bool}}, Vector{Int}}()
    @inbounds for q in 1:TO.nvertices(Q)
        key = ntuple(i -> Y[i].mask[q], length(Y))
        vec = get!(sigs, key) do
            Int[]
        end
        push!(vec, q)
    end
    return collect(values(sigs))
end

function _canon_regions(regs::Vector{Vector{Int}})
    out = [sort(copy(r)) for r in regs]
    sort!(out; by = x -> (length(x), x))
    return out
end

@inline function _is_subset_mask(a::BitVector, b::BitVector)
    @inbounds for i in eachindex(a)
        if a[i] && !b[i]
            return false
        end
    end
    return true
end

function _random_fringe_module(P::FF.AbstractPoset, field::CM.AbstractCoeffField;
                               nu::Int=24, nd::Int=24, density::Float64=0.15, seed::Int=0x6161)
    rng = Random.MersenneTwister(seed)
    K = CM.coeff_type(field)
    n = TO.nvertices(P)

    U = Vector{FF.Upset}(undef, nu)
    D = Vector{FF.Downset}(undef, nd)
    @inbounds for i in 1:nu
        U[i] = FF.upset_closure(P, BitVector(rand(rng, Bool, n)))
    end
    @inbounds for j in 1:nd
        D[j] = FF.downset_closure(P, BitVector(rand(rng, Bool, n)))
    end

    phi = spzeros(K, nd, nu)
    @inbounds for j in 1:nd, i in 1:nu
        if FF.intersects(U[i], D[j]) && rand(rng) < density
            v = rand(rng, -3:3)
            if v == 0
                v = 1
            end
            phi[j, i] = CM.coerce(field, v)
        end
    end

    return FF.FringeModule{K}(P, U, D, phi; field=field)
end

function _clear_hom_route_memo!(M::FF.FringeModule)
    FF._clear_hom_route_choice!(M)
    return nothing
end

function _fiber_dimension_old(M::FF.FringeModule{K}, q::Int) where {K}
    cols = findall(U -> U.mask[q], M.U)
    rows = findall(D -> D.mask[q], M.D)
    if isempty(cols) || isempty(rows)
        return 0
    end
    return FL.rank_restricted(M.field, M.phi, rows, cols)
end

function _fiber_dimension_batch_old(M::FF.FringeModule, queries::Vector{Int})
    acc = 0
    @inbounds for q in queries
        acc += _fiber_dimension_old(M, q)
    end
    return acc
end

function _fiber_dimension_batch_new(M::FF.FringeModule, queries::Vector{Int})
    acc = 0
    @inbounds for q in queries
        acc += FF.fiber_dimension(M, q)
    end
    return acc
end

function _hom_dimension_old(M::FF.FringeModule{K}, N::FF.FringeModule{K}) where {K}
    M.P === N.P || error("Posets must match")
    P = M.P
    adj = FF._cover_undirected_adjacency(P)

    nUM = length(M.U); nDM = length(M.D)
    nUN = length(N.U); nDN = length(N.D)

    Ucomp_id_M    = Vector{Vector{Int}}(undef, nUM)
    Ucomp_masks_M = Vector{Vector{BitVector}}(undef, nUM)
    Ucomp_n_M     = Vector{Int}(undef, nUM)
    for i in 1:nUM
        comp_id, ncomp, comp_masks, _ = FF._component_data(adj, M.U[i].mask)
        Ucomp_id_M[i] = comp_id
        Ucomp_masks_M[i] = comp_masks
        Ucomp_n_M[i] = ncomp
    end

    Dcomp_id_N    = Vector{Vector{Int}}(undef, nDN)
    Dcomp_masks_N = Vector{Vector{BitVector}}(undef, nDN)
    Dcomp_n_N     = Vector{Int}(undef, nDN)
    for t in 1:nDN
        comp_id, ncomp, comp_masks, _ = FF._component_data(adj, N.D[t].mask)
        Dcomp_id_N[t] = comp_id
        Dcomp_masks_N[t] = comp_masks
        Dcomp_n_N[t] = ncomp
    end

    w_index = Dict{Tuple{Int,Int},Int}()
    w_rows_u = Vector{Vector{Vector{Int}}}()
    w_rows_d = Vector{Vector{Vector{Int}}}()
    W_dim = 0

    for iM in 1:nUM
        for tN in 1:nDN
            mask_int = M.U[iM].mask .& N.D[tN].mask
            if any(mask_int)
                _, ncomp_int, _, reps_int = FF._component_data(adj, mask_int)
                base = W_dim
                W_dim += ncomp_int

                rows_by_u = [Int[] for _ in 1:Ucomp_n_M[iM]]
                rows_by_d = [Int[] for _ in 1:Dcomp_n_N[tN]]
                for c in 1:ncomp_int
                    row = base + c
                    v = reps_int[c]
                    cu = Ucomp_id_M[iM][v]
                    cd = Dcomp_id_N[tN][v]
                    push!(rows_by_u[cu], row)
                    push!(rows_by_d[cd], row)
                end

                push!(w_rows_u, rows_by_u)
                push!(w_rows_d, rows_by_d)
                w_index[(iM, tN)] = length(w_rows_u)
            end
        end
    end

    V1 = Tuple{Int,Int,Int}[]
    for iM in 1:nUM
        for jN in 1:nUN
            for cU in 1:Ucomp_n_M[iM]
                if _is_subset_mask(Ucomp_masks_M[iM][cU], N.U[jN].mask)
                    push!(V1, (iM, jN, cU))
                end
            end
        end
    end
    V1_dim = length(V1)

    V2 = Tuple{Int,Int,Int}[]
    for sM in 1:nDM
        for tN in 1:nDN
            for cD in 1:Dcomp_n_N[tN]
                if _is_subset_mask(Dcomp_masks_N[tN][cD], M.D[sM].mask)
                    push!(V2, (sM, tN, cD))
                end
            end
        end
    end
    V2_dim = length(V2)

    T = zeros(K, W_dim, V1_dim)
    for (col, (iM, jN, cU)) in enumerate(V1)
        for tN in 1:nDN
            val = N.phi[tN, jN]
            if val != zero(K)
                pid = get(w_index, (iM, tN), 0)
                if pid != 0
                    rows = w_rows_u[pid][cU]
                    for r in rows
                        T[r, col] += val
                    end
                end
            end
        end
    end

    S = zeros(K, W_dim, V2_dim)
    for (col, (sM, tN, cD)) in enumerate(V2)
        for iM in 1:nUM
            val = M.phi[sM, iM]
            if val != zero(K)
                pid = get(w_index, (iM, tN), 0)
                if pid != 0
                    rows = w_rows_d[pid][cD]
                    for r in rows
                        S[r, col] += val
                    end
                end
            end
        end
    end

    big = hcat(T, -S)
    rT = FL.rank(M.field, T)
    rS = FL.rank(M.field, S)
    rBig = FL.rank(M.field, big)
    dimKer_big = (V1_dim + V2_dim) - rBig
    dimKer_T = V1_dim - rT
    dimKer_S = V2_dim - rS
    return dimKer_big - (dimKer_T + dimKer_S)
end

function main(args=ARGS)
    reps = _parse_int_arg(args, "--reps", 7)
    n = _parse_int_arg(args, "--n", 180)
    ny = _parse_int_arg(args, "--y", 96)
    y_cases = _parse_int_list_arg(args, "--y_cases", Int[32, 96, 160])
    nu = _parse_int_arg(args, "--nu", 24)
    nd = _parse_int_arg(args, "--nd", 24)
    fiber_queries = _parse_int_arg(args, "--fiber_queries", max(800, 8n))
    hom_dense_case = _parse_bool_arg(args, "--hom_dense_case", false)

    println("FiniteFringe microbench")
    println("reps=$(reps), poset_n=$(n), y=$(ny), y_cases=$(y_cases), nu=$(nu), nd=$(nd), fiber_queries=$(fiber_queries), hom_dense_case=$(hom_dense_case)\n")

    P = _random_poset(n; p=0.03, seed=Int(0x7010))
    println("== Uptight region partition ==")
    for (k, yk) in enumerate(y_cases)
        Yk = _fixture_uptight(P, yk; seed=Int(0x7011 + k))

        # Parity checks for each Y regime.
        r_old = _canon_regions(_uptight_regions_old(P, Yk))
        r_new = _canon_regions(EN._uptight_regions(P, Yk))
        r_old == r_new || error("_uptight_regions parity failed for y=$(yk)")

        regime = yk <= 64 ? "small <=64" : (yk <= 128 ? "medium <=128" : "large >128")
        println("  -- y=$(yk) ($regime)")
        b_old_up = _bench("uptight old (tuple keys)", () -> _uptight_regions_old(P, Yk); reps=reps)
        b_new_up = _bench("uptight new (packed)", () -> EN._uptight_regions(P, Yk); reps=reps)
        println("  speedup(new/old): ", round(b_old_up.ms / b_new_up.ms, digits=2), "x")
    end
    println()

    field = CM.QQField()
    K = CM.coeff_type(field)
    Mf = _random_fringe_module(P, field; nu=max(nu, 28), nd=max(nd, 28), density=0.16, seed=Int(0x7019))
    rng = MersenneTwister(0x7031)
    fiber_qs = rand(rng, 1:TO.nvertices(P), fiber_queries)
    sum_old_fiber = _fiber_dimension_batch_old(Mf, fiber_qs)
    sum_new_fiber = _fiber_dimension_batch_new(Mf, fiber_qs)
    sum_old_fiber == sum_new_fiber || error("fiber_dimension parity failed: old=$(sum_old_fiber), new=$(sum_new_fiber)")

    println("== Fiber dimension repeated query ==")
    b_old_f = _bench("fiber old (scan/findall)", () -> _fiber_dimension_batch_old(Mf, fiber_qs); reps=reps)
    b_new_f_cold = _bench("fiber new cold (build+memo)", () -> begin
        Mc = FF.FringeModule{K}(P, Mf.U, Mf.D, Mf.phi; field=field)
        _fiber_dimension_batch_new(Mc, fiber_qs)
    end; reps=reps)
    b_new_f = _bench("fiber new warm (cached)", () -> _fiber_dimension_batch_new(Mf, fiber_qs); reps=reps)
    println("speedup(new/old) cold: ", round(b_old_f.ms / b_new_f_cold.ms, digits=2), "x")
    println("speedup(new/old) warm: ", round(b_old_f.ms / b_new_f.ms, digits=2), "x")
    println()

    M = _random_fringe_module(P, field; nu=nu, nd=nd, density=0.15, seed=Int(0x7021))
    N = _random_fringe_module(P, field; nu=nu, nd=nd, density=0.15, seed=Int(0x7022))

    # Parity checks for hom dimension.
    h_old = _hom_dimension_old(M, N)
    h_sparse_path = FF._hom_dimension_with_path(M, N, :sparse_path)
    h_denseidx = FF._hom_dimension_with_path(M, N, :dense_idx_internal)
    h_auto = FF.hom_dimension(M, N)
    h_old == h_sparse_path == h_denseidx == h_auto ||
        error("hom_dimension parity failed: old=$(h_old), sparse_path=$(h_sparse_path), dense_idx=$(h_denseidx), auto=$(h_auto)")

    println("== Hom dimension assembly ==")
    b_old_h = _bench("hom_dimension old (dict+mask)", () -> _hom_dimension_old(M, N); reps=reps)
    b_sparse_h = _bench("hom_dimension path=:sparse_path", () -> FF._hom_dimension_with_path(M, N, :sparse_path); reps=reps)
    b_dense_h = _bench("hom_dimension path=:dense_idx_internal", () -> FF._hom_dimension_with_path(M, N, :dense_idx_internal); reps=reps)
    b_auto_h = _bench("hom_dimension auto", () -> FF.hom_dimension(M, N); reps=reps)
    b_auto_cold_first_h = _bench("hom_dimension auto cold first", () -> begin
        _clear_hom_route_memo!(M)
        Mc = FF.FringeModule{K}(P, M.U, M.D, M.phi; field=field)
        Nc = FF.FringeModule{K}(P, N.U, N.D, N.phi; field=field)
        FF.hom_dimension(Mc, Nc)
    end; reps=reps)
    b_auto_cold_h = _bench("hom_dimension auto cold memo", () -> begin
        Mc = FF.FringeModule{K}(P, M.U, M.D, M.phi; field=field)
        Nc = FF.FringeModule{K}(P, N.U, N.D, N.phi; field=field)
        FF.hom_dimension(Mc, Nc)
    end; reps=reps)
    has_timing_baseline = isdefined(FF, :_hom_dimension_auto_timing_baseline)
    b_auto_cold_timing_h = if has_timing_baseline
        _bench("hom_dimension auto cold timing-baseline", () -> begin
            _clear_hom_route_memo!(M)
            Mc = FF.FringeModule{K}(P, M.U, M.D, M.phi; field=field)
            Nc = FF.FringeModule{K}(P, N.U, N.D, N.phi; field=field)
            getfield(FF, :_hom_dimension_auto_timing_baseline)(Mc, Nc)
        end; reps=reps)
    else
        nothing
    end
    println("speedup(sparse_path/old): ", round(b_old_h.ms / b_sparse_h.ms, digits=2), "x")
    println("speedup(dense_idx/old): ", round(b_old_h.ms / b_dense_h.ms, digits=2), "x")
    println("speedup(auto/old): ", round(b_old_h.ms / b_auto_h.ms, digits=2), "x")
    println("warmup gain(auto cold-first/auto warm): ", round(b_auto_cold_first_h.ms / b_auto_h.ms, digits=2), "x")
    println("warmup gain(auto cold-memo/auto warm): ", round(b_auto_cold_h.ms / b_auto_h.ms, digits=2), "x")
    if b_auto_cold_timing_h === nothing
        println("cold gain(heuristic/timing-baseline): n/a (timing-baseline helper unavailable)")
    else
        println("cold gain(heuristic/timing-baseline): ", round(b_auto_cold_timing_h.ms / b_auto_cold_first_h.ms, digits=2), "x")
    end
    _clear_hom_route_memo!(M)
    _ = FF.hom_dimension(M, N)
    selected_sparse = FF._select_hom_internal_path!(M, N)
    best_h_ms = min(b_sparse_h.ms, b_dense_h.ms)
    auto_delta_pct = 100.0 * (b_auto_h.ms / best_h_ms - 1.0)
    println("auto selected path: ", selected_sparse, " | auto vs best path delta: ",
            round(auto_delta_pct, digits=2), "%")

    if hom_dense_case
        M_dense = FF.FringeModule{K}(P, M.U, M.D, Matrix(M.phi); field=field)
        N_dense = FF.FringeModule{K}(P, N.U, N.D, Matrix(N.phi); field=field)
        h_old_dense = _hom_dimension_old(M_dense, N_dense)
        h_sparse_path_dense = FF._hom_dimension_with_path(M_dense, N_dense, :sparse_path)
        h_denseidx_dense = FF._hom_dimension_with_path(M_dense, N_dense, :dense_idx_internal)
        h_auto_dense = FF.hom_dimension(M_dense, N_dense)
        h_old_dense == h_sparse_path_dense == h_denseidx_dense == h_auto_dense ||
            error("hom_dimension dense parity failed: old=$(h_old_dense), sparse_path=$(h_sparse_path_dense), dense_idx=$(h_denseidx_dense), auto=$(h_auto_dense)")

        println()
        println("== Hom dimension assembly (dense phi storage) ==")
        b_old_hd = _bench("hom_dimension old dense", () -> _hom_dimension_old(M_dense, N_dense); reps=reps)
        b_sparse_hd = _bench("hom_dimension sparse_path dense", () -> FF._hom_dimension_with_path(M_dense, N_dense, :sparse_path); reps=reps)
        b_dense_hd = _bench("hom_dimension dense_idx dense", () -> FF._hom_dimension_with_path(M_dense, N_dense, :dense_idx_internal); reps=reps)
        b_auto_hd = _bench("hom_dimension auto dense", () -> FF.hom_dimension(M_dense, N_dense); reps=reps)
        b_auto_cold_first_hd = _bench("hom_dimension auto dense cold first", () -> begin
            _clear_hom_route_memo!(M_dense)
            Mcd = FF.FringeModule{K}(P, M_dense.U, M_dense.D, M_dense.phi; field=field)
            Ncd = FF.FringeModule{K}(P, N_dense.U, N_dense.D, N_dense.phi; field=field)
            FF.hom_dimension(Mcd, Ncd)
        end; reps=reps)
        b_auto_cold_hd = _bench("hom_dimension auto dense cold memo", () -> begin
            Mcd = FF.FringeModule{K}(P, M_dense.U, M_dense.D, M_dense.phi; field=field)
            Ncd = FF.FringeModule{K}(P, N_dense.U, N_dense.D, N_dense.phi; field=field)
            FF.hom_dimension(Mcd, Ncd)
        end; reps=reps)
        b_auto_cold_timing_hd = if has_timing_baseline
            _bench("hom_dimension auto dense cold timing-baseline", () -> begin
                _clear_hom_route_memo!(M_dense)
                Mcd = FF.FringeModule{K}(P, M_dense.U, M_dense.D, M_dense.phi; field=field)
                Ncd = FF.FringeModule{K}(P, N_dense.U, N_dense.D, N_dense.phi; field=field)
                getfield(FF, :_hom_dimension_auto_timing_baseline)(Mcd, Ncd)
            end; reps=reps)
        else
            nothing
        end
        println("speedup(sparse_path/old): ", round(b_old_hd.ms / b_sparse_hd.ms, digits=2), "x")
        println("speedup(dense_idx/old): ", round(b_old_hd.ms / b_dense_hd.ms, digits=2), "x")
        println("speedup(auto/old): ", round(b_old_hd.ms / b_auto_hd.ms, digits=2), "x")
        println("warmup gain(auto dense cold-first/auto warm): ", round(b_auto_cold_first_hd.ms / b_auto_hd.ms, digits=2), "x")
        println("warmup gain(auto dense cold-memo/auto warm): ", round(b_auto_cold_hd.ms / b_auto_hd.ms, digits=2), "x")
        if b_auto_cold_timing_hd === nothing
            println("cold gain dense(heuristic/timing-baseline): n/a (timing-baseline helper unavailable)")
        else
            println("cold gain dense(heuristic/timing-baseline): ", round(b_auto_cold_timing_hd.ms / b_auto_cold_first_hd.ms, digits=2), "x")
        end
        _clear_hom_route_memo!(M_dense)
        _ = FF.hom_dimension(M_dense, N_dense)
        selected_dense = FF._select_hom_internal_path!(M_dense, N_dense)
        d_dense_min = min(FF._matrix_density(M_dense.phi), FF._matrix_density(N_dense.phi))
        println("effective dense-storage density dmin=", round(d_dense_min, digits=4))
        best_hd_ms = min(b_sparse_hd.ms, b_dense_hd.ms)
        auto_dense_delta_pct = 100.0 * (b_auto_hd.ms / best_hd_ms - 1.0)
        println("auto selected path (dense fixture): ", selected_dense,
                " | auto vs best path delta: ", round(auto_dense_delta_pct, digits=2), "%")

        println()
        println("== Dense-storage sparse-effective routing probe ==")
        b_probe_auto = _bench("probe auto dense", () -> FF.hom_dimension(M_dense, N_dense); reps=reps)
        b_probe_sparse = _bench("probe forced sparse dense", () -> FF._hom_dimension_with_path(M_dense, N_dense, :sparse_path); reps=reps)
        probe_gain = b_probe_auto.ms / b_probe_sparse.ms
        println("probe gain(forced sparse / auto): ", round(probe_gain, digits=2), "x")
    end
end

main()
