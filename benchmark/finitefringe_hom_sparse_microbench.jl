#!/usr/bin/env julia

using Random
using SparseArrays

try
    using TamerOp
catch
    include(joinpath(@__DIR__, "..", "src", "TamerOp.jl"))
    using .TamerOp
end

const FF = TamerOp.FiniteFringe
const CM = TamerOp.CoreModules
const FL = TamerOp.FieldLinAlg

function _bench(name::AbstractString, f::Function; reps::Int=7)
    GC.gc()
    f() # warm
    GC.gc()
    times_ms = Float64[]
    allocs = Int[]
    for _ in 1:reps
        m = @timed f()
        push!(times_ms, 1000 * m.time)
        push!(allocs, m.bytes)
    end
    sort!(times_ms)
    sort!(allocs)
    med_ms = times_ms[cld(reps, 2)]
    med_mib = allocs[cld(reps, 2)] / 1024^2
    println(name, ",median_ms=", round(med_ms, digits=3), ",median_alloc_mib=", round(med_mib, digits=3))
    return (ms=med_ms, mib=med_mib)
end

function _random_poset(n::Int; p::Float64=0.03, seed::Int=Int(0xBEEF))
    rng = MersenneTwister(seed)
    leq = falses(n, n)
    @inbounds for i in 1:n
        leq[i, i] = true
        for j in (i + 1):n
            leq[i, j] = rand(rng) < p
        end
    end
    @inbounds for k in 1:n, i in 1:n, j in 1:n
        leq[i, j] = leq[i, j] || (leq[i, k] && leq[k, j])
    end
    return FF.FinitePoset(leq; check=false)
end

function _random_module(P::FF.AbstractPoset, field::CM.AbstractCoeffField;
                        nu::Int=24, nd::Int=24, density::Float64=0.15, seed::Int=Int(0x1234))
    rng = MersenneTwister(seed)
    K = CM.coeff_type(field)
    n = FF.nvertices(P)

    U = [FF.upset_closure(P, BitVector(rand(rng, Bool, n))) for _ in 1:nu]
    D = [FF.downset_closure(P, BitVector(rand(rng, Bool, n))) for _ in 1:nd]

    phi = spzeros(K, nd, nu)
    @inbounds for j in 1:nd, i in 1:nu
        FF.intersects(U[i], D[j]) || continue
        rand(rng) < density || continue
        v = rand(rng, -2:2)
        v == 0 && continue
        phi[j, i] = CM.coerce(field, v)
    end
    return FF.FringeModule{K}(P, U, D, phi; field=field)
end

function _legacy_sparse_reference(M::FF.FringeModule{K}, N::FF.FringeModule{K}) where {K}
    M.P === N.P || error("Posets must match")
    adj = FF._cover_undirected_adjacency(M.P)

    nUM = length(M.U); nDM = length(M.D)
    nUN = length(N.U); nDN = length(N.D)

    Ucomp_id_M = Vector{Vector{Int}}(undef, nUM)
    Ucomp_masks_M = Vector{Vector{BitVector}}(undef, nUM)
    Ucomp_n_M = Vector{Int}(undef, nUM)
    for i in 1:nUM
        comp_id, ncomp, comp_masks, _ = FF._component_data(adj, M.U[i].mask)
        Ucomp_id_M[i] = comp_id
        Ucomp_masks_M[i] = comp_masks
        Ucomp_n_M[i] = ncomp
    end

    Dcomp_id_N = Vector{Vector{Int}}(undef, nDN)
    Dcomp_masks_N = Vector{Vector{BitVector}}(undef, nDN)
    Dcomp_n_N = Vector{Int}(undef, nDN)
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
                FF.is_subset(Ucomp_masks_M[iM][cU], N.U[jN].mask) || continue
                push!(V1, (iM, jN, cU))
            end
        end
    end

    V2 = Tuple{Int,Int,Int}[]
    for sM in 1:nDM
        for tN in 1:nDN
            for cD in 1:Dcomp_n_N[tN]
                FF.is_subset(Dcomp_masks_N[tN][cD], M.D[sM].mask) || continue
                push!(V2, (sM, tN, cD))
            end
        end
    end

    V1_dim = length(V1)
    V2_dim = length(V2)

    T = zeros(K, W_dim, V1_dim)
    for (col, (iM, jN, cU)) in enumerate(V1)
        for tN in 1:nDN
            val = N.phi[tN, jN]
            val == zero(K) && continue
            pid = get(w_index, (iM, tN), 0)
            pid == 0 && continue
            for r in w_rows_u[pid][cU]
                T[r, col] += val
            end
        end
    end

    S = zeros(K, W_dim, V2_dim)
    for (col, (sM, tN, cD)) in enumerate(V2)
        for iM in 1:nUM
            val = M.phi[sM, iM]
            val == zero(K) && continue
            pid = get(w_index, (iM, tN), 0)
            pid == 0 && continue
            for r in w_rows_d[pid][cD]
                S[r, col] += val
            end
        end
    end

    rT = FL.rank(M.field, T)
    rS = FL.rank(M.field, S)
    rBig = FL.rank(M.field, hcat(T, -S))
    return (V1_dim + V2_dim - rBig) - ((V1_dim - rT) + (V2_dim - rS))
end

P = _random_poset(170; p=0.028, seed=Int(0x2222))
field = CM.QQField()
M = _random_module(P, field; nu=24, nd=24, density=0.16, seed=Int(0x1111))
N = _random_module(P, field; nu=24, nd=24, density=0.14, seed=Int(0x3333))

v_ref = _legacy_sparse_reference(M, N)
v_new = FF._hom_dimension_with_path(M, N, :sparse_path)
v_auto = FF.hom_dimension(M, N)
(v_ref == v_new == v_auto) || error("benchmark fixture parity failed")

println("fixture_n=", P.n, ",nu=", length(M.U), ",nd=", length(M.D))
_bench("legacy_sparse_reference", () -> _legacy_sparse_reference(M, N); reps=7)
_bench("sparse_path", () -> FF._hom_dimension_with_path(M, N, :sparse_path); reps=7)
_bench("hom_auto", () -> FF.hom_dimension(M, N); reps=7)

# Targeted cache-reuse probe: first call builds sparse plan, second reuses it.
Mc = _random_module(P, field; nu=24, nd=24, density=0.16, seed=Int(0x4444))
Nc = _random_module(P, field; nu=24, nd=24, density=0.14, seed=Int(0x5555))
_ = FF._hom_dimension_with_path(Mc, Nc, :dense_idx_internal) # compile guard
cold_stats = @timed FF._hom_dimension_with_path(Mc, Nc, :sparse_path)
warm_stats = @timed FF._hom_dimension_with_path(Mc, Nc, :sparse_path)
println("sparse_path_cache_probe,cold_ms=", round(1000 * cold_stats.time, digits=3),
        ",cold_alloc_mib=", round(cold_stats.bytes / 1024^2, digits=3),
        ",warm_ms=", round(1000 * warm_stats.time, digits=3),
        ",warm_alloc_mib=", round(warm_stats.bytes / 1024^2, digits=3))
