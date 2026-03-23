#!/usr/bin/env julia
#
# abelian_categories_microbench.jl
#
# Purpose
# - Benchmark core AbelianCategories kernels touched by recent optimizations:
#   1) `kernel_with_inclusion`
#   2) `image_with_inclusion`
#   3) `_cokernel_module` / `cokernel_with_projection`
#
# What this compares
# - For each kernel, run:
#   - a local baseline copy of the previous algorithmic shape
#     (solve-per-edge and tuple-keyed dict edge assembly),
#   - the current implementation (store-aligned edge assembly + O(1) cover-slot writes).
#
# Notes
# - The current Abelian hot paths prioritize stable, low-overhead behavior
#   across small and medium workloads.
# - Cover-edge slot writes use precomputed cache slots (no per-edge binary
#   predecessor search in hot loops).
#
# Scope
# - Deterministic chain-poset fixtures.
# - Median wall-time and allocation over repeated runs.
# - Optional high-fanout fixture to expose per-vertex batched cokernel solves.
# - Functional parity checks before timing.
#
# Usage
#   julia --project=. benchmark/abelian_categories_microbench.jl
#   julia --project=. benchmark/abelian_categories_microbench.jl --reps=7 --n=28 --rank=5 --ker=5 --coker=5
#   julia --project=. benchmark/abelian_categories_microbench.jl --reps=5 --n=64 --n_small=24
#   julia --project=. benchmark/abelian_categories_microbench.jl --reps=5 --n=64 --fan_sweep_left=8 --fan_sweep_low=2 --fan_sweep_mid=8 --fan_sweep_high=16
#

using Random
using SparseArrays
using LinearAlgebra

module _AbelianBenchEnv
include(joinpath(@__DIR__, "..", "src", "CoreModules.jl"))
include(joinpath(@__DIR__, "..", "src", "Options.jl"))
include(joinpath(@__DIR__, "..", "src", "EncodingCore.jl"))
include(joinpath(@__DIR__, "..", "src", "RegionGeometry.jl"))
include(joinpath(@__DIR__, "..", "src", "FieldLinAlg.jl"))
include(joinpath(@__DIR__, "..", "src", "FiniteFringe.jl"))
include(joinpath(@__DIR__, "..", "src", "Modules.jl"))
include(joinpath(@__DIR__, "..", "src", "AbelianCategories.jl"))
end

const TO = _AbelianBenchEnv
const AC = TO.AbelianCategories
const MD = TO.Modules
const FF = TO.FiniteFringe
const CM = TO.CoreModules
const FL = TO.FieldLinAlg

function _parse_int_arg(args, key::String, default::Int)
    for a in args
        startswith(a, key * "=") || continue
        return max(1, parse(Int, split(a, "=", limit=2)[2]))
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
    println(rpad(name, 46), " median_time=", round(med_ms, digits=3),
            " ms  median_alloc=", round(med_kib, digits=1), " KiB")
    return (ms=med_ms, kib=med_kib)
end

@inline function _rand_coeff(rng::AbstractRNG, field::CM.AbstractCoeffField)
    v = rand(rng, -3:3)
    v == 0 && (v = 1)
    return CM.coerce(field, v)
end

function _rand_dense(rng::AbstractRNG, field::CM.AbstractCoeffField, m::Int, n::Int)
    K = CM.coeff_type(field)
    A = zeros(K, m, n)
    @inbounds for i in 1:m, j in 1:n
        A[i, j] = _rand_coeff(rng, field)
    end
    return A
end

function _chain_poset(n::Int)
    leq = falses(n, n)
    @inbounds for i in 1:n, j in i:n
        leq[i, j] = true
    end
    return FF.FinitePoset(leq; check=false)
end

function _two_layer_poset(nleft::Int, nright::Int)
    (nleft > 0 && nright > 0) || error("_two_layer_poset: need positive layer sizes")
    n = nleft + nright
    L = falses(n, n)
    @inbounds for i in 1:n
        L[i, i] = true
    end
    @inbounds for u in 1:nleft
        for v in (nleft + 1):n
            L[u, v] = true
        end
    end
    return FF.FinitePoset(L; check=false)
end

function _build_morphism_fixture(P::FF.AbstractPoset,
                                 field::CM.AbstractCoeffField;
                                 rank_part::Int=5,
                                 ker_part::Int=5,
                                 coker_part::Int=5,
                                 seed::Int=Int(0xAB31))
    rng = MersenneTwister(seed)
    K = CM.coeff_type(field)

    r = rank_part
    k = ker_part
    c = coker_part
    da = r + k
    db = r + c
    n = FF.nvertices(P)

    dimsA = fill(da, n)
    dimsB = fill(db, n)

    edgeA = Dict{Tuple{Int,Int}, Matrix{K}}()
    edgeB = Dict{Tuple{Int,Int}, Matrix{K}}()
    for (u, v) in FF.cover_edges(P)
        Ruv = _rand_dense(rng, field, r, r)
        Kuv = _rand_dense(rng, field, k, k)
        Yuv = _rand_dense(rng, field, c, c)
        Xuv = _rand_dense(rng, field, r, c)

        Auv = zeros(K, da, da)
        Buv = zeros(K, db, db)
        @inbounds begin
            copyto!(view(Auv, 1:r, 1:r), Ruv)
            copyto!(view(Auv, r+1:da, r+1:da), Kuv)
            copyto!(view(Buv, 1:r, 1:r), Ruv)
            copyto!(view(Buv, 1:r, r+1:db), Xuv)
            copyto!(view(Buv, r+1:db, r+1:db), Yuv)
        end

        edgeA[(u, v)] = Auv
        edgeB[(u, v)] = Buv
    end

    A = MD.PModule{K}(P, dimsA, edgeA; field=field)
    B = MD.PModule{K}(P, dimsB, edgeB; field=field)

    F = zeros(K, db, da)
    @inbounds for i in 1:r
        F[i, i] = CM.coerce(field, 1)
    end
    comps = [copy(F) for _ in 1:n]
    f = MD.PMorphism(A, B, comps)
    return A, B, f
end

function _kernel_with_inclusion_old(f::MD.PMorphism{K}; cache::Union{Nothing,MD.CoverCache}=nothing) where {K}
    M = f.dom
    n = FF.nvertices(M.Q)

    basisK = Vector{Matrix{K}}(undef, n)
    K_dims = zeros(Int, n)
    for i in 1:n
        B = FL.nullspace(f.dom.field, f.comps[i])
        basisK[i] = B
        K_dims[i] = size(B, 2)
    end

    cc = (cache === nothing ? MD._get_cover_cache(M.Q) : cache)
    preds = cc.preds
    succs = cc.succs

    maps_from_pred = [Vector{Matrix{K}}(undef, length(preds[v])) for v in 1:n]
    maps_to_succ   = [Vector{Matrix{K}}(undef, length(succs[u])) for u in 1:n]

    @inbounds for u in 1:n
        su = succs[u]
        maps_u_M = M.edge_maps.maps_to_succ[u]
        outu = maps_to_succ[u]
        for j in eachindex(su)
            v = su[j]
            if K_dims[u] == 0 || K_dims[v] == 0
                X = zeros(K, K_dims[v], K_dims[u])
                outu[j] = X
                ip = MD._find_sorted_index(preds[v], u)
                maps_from_pred[v][ip] = X
                continue
            end
            Im = maps_u_M[j] * basisK[u]
            X = FL.solve_fullcolumn(f.dom.field, basisK[v], Im; check_rhs=false)
            outu[j] = X
            ip = MD._find_sorted_index(preds[v], u)
            maps_from_pred[v][ip] = X
        end
    end

    storeK = MD.CoverEdgeMapStore{K,Matrix{K}}(preds, succs, maps_from_pred, maps_to_succ, cc.nedges)
    Kmod = MD.PModule{K}(M.Q, K_dims, storeK; field=M.field)
    iota = MD.PMorphism{K}(Kmod, M, [basisK[i] for i in 1:n])
    return Kmod, iota
end

function _image_with_inclusion_old(f::MD.PMorphism{K}; cache::Union{Nothing,MD.CoverCache}=nothing) where {K}
    N = f.cod
    Q = N.Q
    n = FF.nvertices(Q)

    bases = Vector{Matrix{K}}(undef, n)
    dims  = zeros(Int, n)
    for i in 1:n
        B = FL.colspace(f.dom.field, f.comps[i])
        bases[i] = B
        dims[i]  = size(B, 2)
    end

    storeN = N.edge_maps
    preds = storeN.preds
    succs = storeN.succs
    maps_from_pred = [Vector{Matrix{K}}(undef, length(preds[v])) for v in 1:n]
    maps_to_succ   = [Vector{Matrix{K}}(undef, length(succs[u])) for u in 1:n]

    @inbounds for u in 1:n
        su = succs[u]
        Nu = storeN.maps_to_succ[u]
        outu = maps_to_succ[u]
        Bu = bases[u]
        du = size(Bu, 2)
        for j in eachindex(su)
            v = su[j]
            Bv = bases[v]
            dv = size(Bv, 2)
            Auv = if du == 0
                zeros(K, dv, 0)
            elseif dv == 0
                zeros(K, 0, du)
            else
                T = Nu[j] * Bu
                FL.solve_fullcolumn(f.dom.field, Bv, T)
            end
            outu[j] = Auv
            ip = MD._find_sorted_index(preds[v], u)
            maps_from_pred[v][ip] = Auv
        end
    end

    storeIm = MD.CoverEdgeMapStore{K,Matrix{K}}(preds, succs, maps_from_pred, maps_to_succ, storeN.nedges)
    Im = MD.PModule{K}(Q, dims, storeIm; field=N.field)
    iota = MD.PMorphism(Im, N, [bases[i] for i in 1:n])
    return Im, iota
end

function _cokernel_module_old(iota::MD.PMorphism{K}; cache::Union{Nothing,MD.CoverCache}=nothing) where {K}
    E = iota.cod
    Q = E.Q
    n = FF.nvertices(Q)
    field = E.field

    Cdims = zeros(Int, n)
    qcomps = Vector{Matrix{K}}(undef, n)
    for i in 1:n
        Bi = FL.colspace(field, iota.comps[i])
        Ni = FL.nullspace(field, transpose(Bi))
        Cdims[i] = size(Ni, 2)
        qcomps[i] = transpose(Ni)
    end

    Cedges = Dict{Tuple{Int,Int}, Matrix{K}}()
    cc = (cache === nothing ? MD._get_cover_cache(Q) : cache)
    @inbounds for u in 1:n
        su = cc.succs[u]
        maps_u = E.edge_maps.maps_to_succ[u]
        for j in eachindex(su)
            v = su[j]
            if Cdims[u] > 0 && Cdims[v] > 0
                T = maps_u[j]
                X = FL.solve_fullcolumn(field, transpose(qcomps[u]), transpose(qcomps[v] * T))
                Cedges[(u, v)] = transpose(X)
            else
                Cedges[(u, v)] = zeros(K, Cdims[v], Cdims[u])
            end
        end
    end

    Cmod = MD.PModule{K}(Q, Cdims, Cedges; field=field)
    q = MD.PMorphism(E, Cmod, qcomps)
    return Cmod, q
end

function _pushout_old(f::MD.PMorphism{K}, g::MD.PMorphism{K};
                      cache::Union{Nothing,MD.CoverCache}=nothing) where {K}
    A = f.dom
    B = f.cod
    C = g.cod
    S = MD.direct_sum(B, C)
    Q = S.Q
    phi_comps = Vector{Matrix{K}}(undef, FF.nvertices(Q))
    @inbounds for u in 1:FF.nvertices(Q)
        fu = f.comps[u]
        gu = g.comps[u]
        b = size(fu, 1)
        c = size(gu, 1)
        a = size(fu, 2)
        M = Matrix{K}(undef, b + c, a)
        if b > 0
            copyto!(view(M, 1:b, :), fu)
        end
        if c > 0
            @inbounds for i in 1:c, j in 1:a
                M[b + i, j] = -gu[i, j]
            end
        end
        phi_comps[u] = M
    end
    phi = MD.PMorphism{K}(A, S, phi_comps)
    P, q = _cokernel_module_old(phi; cache=cache)
    inB_comps = Vector{Matrix{K}}(undef, FF.nvertices(Q))
    inC_comps = Vector{Matrix{K}}(undef, FF.nvertices(Q))
    @inbounds for u in 1:FF.nvertices(Q)
        qu = q.comps[u]
        b = B.dims[u]
        c = C.dims[u]
        inB_comps[u] = copy(view(qu, :, 1:b))
        inC_comps[u] = copy(view(qu, :, (b + 1):(b + c)))
    end
    inB = MD.PMorphism{K}(B, P, inB_comps)
    inC = MD.PMorphism{K}(C, P, inC_comps)
    return P, inB, inC, q, phi
end

function _pullback_old(f::MD.PMorphism{K}, g::MD.PMorphism{K};
                       cache::Union{Nothing,MD.CoverCache}=nothing) where {K}
    B = f.dom
    C = g.dom
    D = f.cod
    S = MD.direct_sum(B, C)
    Q = S.Q
    psi_comps = Vector{Matrix{K}}(undef, FF.nvertices(Q))
    @inbounds for u in 1:FF.nvertices(Q)
        fu = f.comps[u]
        gu = g.comps[u]
        d = size(fu, 1)
        b = size(fu, 2)
        c = size(gu, 2)
        M = Matrix{K}(undef, d, b + c)
        if b > 0
            copyto!(view(M, :, 1:b), fu)
        end
        if c > 0
            @inbounds for i in 1:d, j in 1:c
                M[i, b + j] = -gu[i, j]
            end
        end
        psi_comps[u] = M
    end
    psi = MD.PMorphism{K}(S, D, psi_comps)
    P, iota = _kernel_with_inclusion_old(psi; cache=cache)
    prB_comps = Vector{Matrix{K}}(undef, FF.nvertices(Q))
    prC_comps = Vector{Matrix{K}}(undef, FF.nvertices(Q))
    @inbounds for u in 1:FF.nvertices(Q)
        iu = iota.comps[u]
        b = B.dims[u]
        c = C.dims[u]
        prB_comps[u] = copy(view(iu, 1:b, :))
        prC_comps[u] = copy(view(iu, (b + 1):(b + c), :))
    end
    prB = MD.PMorphism{K}(P, B, prB_comps)
    prC = MD.PMorphism{K}(P, C, prC_comps)
    return P, prB, prC, iota, psi
end

function _equalizer_old(f::MD.PMorphism{K}, g::MD.PMorphism{K};
                        cache::Union{Nothing,MD.CoverCache}=nothing) where {K}
    h = AC._difference_morphism(f, g)
    return _kernel_with_inclusion_old(h; cache=cache)
end

function _coequalizer_old(f::MD.PMorphism{K}, g::MD.PMorphism{K};
                          cache::Union{Nothing,MD.CoverCache}=nothing) where {K}
    h = AC._difference_morphism(f, g)
    return _cokernel_module_old(h; cache=cache)
end

function _morphism_equal(field::CM.AbstractCoeffField, a::MD.PMorphism, b::MD.PMorphism)
    a.dom.dims == b.dom.dims || return false
    a.cod.dims == b.cod.dims || return false
    for i in eachindex(a.comps)
        A = a.comps[i]
        B = b.comps[i]
        if field isa CM.RealField
            isapprox(A, B; rtol=field.rtol, atol=field.atol) || return false
        else
            A == B || return false
        end
    end
    return true
end

function _pmodule_equal(field::CM.AbstractCoeffField, A::MD.PModule, B::MD.PModule)
    A.dims == B.dims || return false
    for ((u, v), Muv) in A.edge_maps
        Nuv = B.edge_maps[u, v]
        if field isa CM.RealField
            isapprox(Muv, Nuv; rtol=field.rtol, atol=field.atol) || return false
        else
            Muv == Nuv || return false
        end
    end
    return true
end

function main(args=ARGS)
    reps = _parse_int_arg(args, "--reps", 7)
    n = _parse_int_arg(args, "--n", 28)
    r = _parse_int_arg(args, "--rank", 5)
    k = _parse_int_arg(args, "--ker", 5)
    c = _parse_int_arg(args, "--coker", 5)
    n_small = _parse_int_arg(args, "--n_small", 0)
    fan_sweep_left = _parse_int_arg(args, "--fan_sweep_left", 0)
    fan_sweep_low = _parse_int_arg(args, "--fan_sweep_low", 2)
    fan_sweep_mid = _parse_int_arg(args, "--fan_sweep_mid", 8)
    fan_sweep_high = _parse_int_arg(args, "--fan_sweep_high", 16)

    field = CM.QQField()
    P = _chain_poset(n)
    A, B, f = _build_morphism_fixture(P, field; rank_part=r, ker_part=k, coker_part=c, seed=Int(0xA110))
    cc = MD._get_cover_cache(P)

    println("AbelianCategories microbench")
    println("reps=$(reps), n=$(n), rank=$(r), ker=$(k), coker=$(c), field=QQ\n")

    # Parity checks before timing.
    K_new, iK_new = AC.kernel_with_inclusion(f; cache=cc)
    K_old, iK_old = _kernel_with_inclusion_old(f; cache=cc)
    Im_new, iIm_new = AC.image_with_inclusion(f; cache=cc)
    Im_old, iIm_old = _image_with_inclusion_old(f; cache=cc)
    C_new, q_new = AC._cokernel_module(f; cache=cc)
    C_old, q_old = _cokernel_module_old(f; cache=cc)

    _pmodule_equal(field, K_new, K_old) || error("kernel parity failed")
    _morphism_equal(field, iK_new, iK_old) || error("kernel inclusion parity failed")
    _pmodule_equal(field, Im_new, Im_old) || error("image parity failed")
    _morphism_equal(field, iIm_new, iIm_old) || error("image inclusion parity failed")
    _pmodule_equal(field, C_new, C_old) || error("cokernel parity failed")
    _morphism_equal(field, q_new, q_old) || error("cokernel projection parity failed")

    println("== Kernel with inclusion ==")
    b_old_k = _bench("kernel old (solve per edge)", () -> _kernel_with_inclusion_old(f; cache=cc); reps=reps)
    b_new_k = _bench("kernel new (slot-aligned path)", () -> AC.kernel_with_inclusion(f; cache=cc); reps=reps)
    println("speedup(new/old): ", round(b_old_k.ms / b_new_k.ms, digits=2), "x")
    println()

    println("== Image with inclusion ==")
    b_old_i = _bench("image old (solve per edge)", () -> _image_with_inclusion_old(f; cache=cc); reps=reps)
    b_new_i = _bench("image new (slot-aligned path)", () -> AC.image_with_inclusion(f; cache=cc); reps=reps)
    println("speedup(new/old): ", round(b_old_i.ms / b_new_i.ms, digits=2), "x")
    println()

    println("== Cokernel with projection ==")
    b_old_c = _bench("cokernel old (dict + edge solve)", () -> _cokernel_module_old(f; cache=cc); reps=reps)
    b_new_c = _bench("cokernel new (slot-aligned path)", () -> AC._cokernel_module(f; cache=cc); reps=reps)
    println("speedup(new/old): ", round(b_old_c.ms / b_new_c.ms, digits=2), "x")

    println()
    println("== Pushout / pullback ==")
    z = MD.zero_morphism(f.dom, f.cod)
    Ppo_new, inB_new, inC_new, qpo_new, phi_new = AC.pushout(f, z; cache=cc)
    Ppo_old, inB_old, inC_old, qpo_old, phi_old = _pushout_old(f, z; cache=cc)
    _pmodule_equal(field, Ppo_new, Ppo_old) || error("pushout parity failed")
    _morphism_equal(field, inB_new, inB_old) || error("pushout inB parity failed")
    _morphism_equal(field, inC_new, inC_old) || error("pushout inC parity failed")
    _morphism_equal(field, qpo_new, qpo_old) || error("pushout q parity failed")
    _morphism_equal(field, phi_new, phi_old) || error("pushout phi parity failed")
    b_old_po = _bench("pushout old (generic block path)", () -> _pushout_old(f, z; cache=cc); reps=reps)
    b_new_po = _bench("pushout new (block-aware path)", () -> AC.pushout(f, z; cache=cc); reps=reps)
    println("speedup(new/old): ", round(b_old_po.ms / b_new_po.ms, digits=2), "x")

    PB_new, prB_new, prC_new, iota_new, psi_new = AC.pullback(f, z; cache=cc)
    PB_old, prB_old, prC_old, iota_old, psi_old = _pullback_old(f, z; cache=cc)
    _pmodule_equal(field, PB_new, PB_old) || error("pullback parity failed")
    _morphism_equal(field, prB_new, prB_old) || error("pullback prB parity failed")
    _morphism_equal(field, prC_new, prC_old) || error("pullback prC parity failed")
    _morphism_equal(field, iota_new, iota_old) || error("pullback iota parity failed")
    _morphism_equal(field, psi_new, psi_old) || error("pullback psi parity failed")
    b_old_pb = _bench("pullback old (generic block path)", () -> _pullback_old(f, z; cache=cc); reps=reps)
    b_new_pb = _bench("pullback new (block-aware path)", () -> AC.pullback(f, z; cache=cc); reps=reps)
    println("speedup(new/old): ", round(b_old_pb.ms / b_new_pb.ms, digits=2), "x")

    println()
    println("== Equalizer / coequalizer (self-map) ==")
    E_new, e_new = AC.equalizer(f, f; cache=cc)
    E_old, e_old = _equalizer_old(f, f; cache=cc)
    _pmodule_equal(field, E_new, E_old) || error("equalizer parity failed")
    _morphism_equal(field, e_new, e_old) || error("equalizer inclusion parity failed")
    b_old_eq = _bench("equalizer old (difference + kernel)", () -> _equalizer_old(f, f; cache=cc); reps=reps)
    b_new_eq = _bench("equalizer new (self-map fast path)", () -> AC.equalizer(f, f; cache=cc); reps=reps)
    println("speedup(new/old): ", round(b_old_eq.ms / b_new_eq.ms, digits=2), "x")

    Q_new, q_new = AC.coequalizer(f, f; cache=cc)
    Q_old, q_old = _coequalizer_old(f, f; cache=cc)
    _pmodule_equal(field, Q_new, Q_old) || error("coequalizer parity failed")
    _morphism_equal(field, q_new, q_old) || error("coequalizer projection parity failed")
    b_old_coeq = _bench("coequalizer old (difference + cokernel)", () -> _coequalizer_old(f, f; cache=cc); reps=reps)
    b_new_coeq = _bench("coequalizer new (self-map fast path)", () -> AC.coequalizer(f, f; cache=cc); reps=reps)
    println("speedup(new/old): ", round(b_old_coeq.ms / b_new_coeq.ms, digits=2), "x")

    println()
    println("== Pushout / pullback (self-map) ==")
    Ppo_self_new, inB_self_new, inC_self_new, qpo_self_new, phi_self_new = AC.pushout(f, f; cache=cc)
    Cok_self = AC.cokernel(f; cache=cc)
    Ppo_self_new.dims == (f.cod.dims .+ Cok_self.dims) || error("self pushout dims failed")
    for u in eachindex(f.comps)
        lhs = inB_self_new.comps[u] * f.comps[u]
        rhs = inC_self_new.comps[u] * f.comps[u]
        zero_block = zeros(eltype(f.comps[u]), size(qpo_self_new.comps[u], 1), size(phi_self_new.comps[u], 2))
        if field isa CM.RealField
            isapprox(lhs, rhs; rtol=field.rtol, atol=field.atol) || error("self pushout relation failed")
            isapprox(qpo_self_new.comps[u] * phi_self_new.comps[u], zero_block; rtol=field.rtol, atol=field.atol) || error("self pushout quotient failed")
        else
            lhs == rhs || error("self pushout relation failed")
            qpo_self_new.comps[u] * phi_self_new.comps[u] == zero_block || error("self pushout quotient failed")
        end
    end
    b_old_po_self = _bench("pushout old (same map)", () -> _pushout_old(f, f; cache=cc); reps=reps)
    b_new_po_self = _bench("pushout new (same-map fast path)", () -> AC.pushout(f, f; cache=cc); reps=reps)
    println("speedup(new/old): ", round(b_old_po_self.ms / b_new_po_self.ms, digits=2), "x")

    PB_self_new, prB_self_new, prC_self_new, iota_self_new, psi_self_new = AC.pullback(f, f; cache=cc)
    Ker_self = AC.kernel(f; cache=cc)
    PB_self_new.dims == (f.dom.dims .+ Ker_self.dims) || error("self pullback dims failed")
    for u in eachindex(f.comps)
        lhs = f.comps[u] * prB_self_new.comps[u]
        rhs = f.comps[u] * prC_self_new.comps[u]
        zero_block = zeros(eltype(f.comps[u]), size(psi_self_new.comps[u], 1), size(iota_self_new.comps[u], 2))
        if field isa CM.RealField
            isapprox(lhs, rhs; rtol=field.rtol, atol=field.atol) || error("self pullback relation failed")
            isapprox(psi_self_new.comps[u] * iota_self_new.comps[u], zero_block; rtol=field.rtol, atol=field.atol) || error("self pullback kernel failed")
        else
            lhs == rhs || error("self pullback relation failed")
            psi_self_new.comps[u] * iota_self_new.comps[u] == zero_block || error("self pullback kernel failed")
        end
    end
    b_old_pb_self = _bench("pullback old (same map)", () -> _pullback_old(f, f; cache=cc); reps=reps)
    b_new_pb_self = _bench("pullback new (same-map fast path)", () -> AC.pullback(f, f; cache=cc); reps=reps)
    println("speedup(new/old): ", round(b_old_pb_self.ms / b_new_pb_self.ms, digits=2), "x")

    println()
    println("== Cache-routed composites ==")
    b_coim_cache = _bench("coimage_with_projection (cache)", () -> AC.coimage_with_projection(f; cache=cc); reps=reps)
    b_coim_auto = _bench("coimage_with_projection (auto)", () -> AC.coimage_with_projection(f); reps=reps)
    println("cache/auto ratio: ", round(b_coim_cache.ms / b_coim_auto.ms, digits=2), "x")

    P1 = _chain_poset(1)
    cc1 = MD._get_cover_cache(P1)
    K1 = CM.coeff_type(field)
    empty_edges = Dict{Tuple{Int,Int}, Matrix{K1}}()
    A1 = MD.PModule{K1}(P1, [1], empty_edges; field=field)
    B1 = MD.PModule{K1}(P1, [2], empty_edges; field=field)
    C1 = MD.PModule{K1}(P1, [1], empty_edges; field=field)
    i1 = MD.PMorphism(A1, B1, [reshape(K1[CM.coerce(field, 1), CM.coerce(field, 0)], 2, 1)])
    p1 = MD.PMorphism(B1, C1, [reshape(K1[CM.coerce(field, 0), CM.coerce(field, 1)], 1, 2)])

    function _ses_check_cached()
        ses = AC.ShortExactSequence(i1, p1; check=false)
        AC.is_exact(ses; cache=cc1)
    end
    function _ses_check_auto()
        ses = AC.ShortExactSequence(i1, p1; check=false)
        AC.is_exact(ses)
    end
    b_ses_cache = _bench("short_exact_sequence check (cache)", _ses_check_cached; reps=reps)
    b_ses_auto = _bench("short_exact_sequence check (auto)", _ses_check_auto; reps=reps)
    println("cache/auto ratio: ", round(b_ses_cache.ms / b_ses_auto.ms, digits=2), "x")

    if n_small > 0
        println()
        println("== Cokernel small-case overhead check ==")
        Ps = _chain_poset(n_small)
        _, _, fs = _build_morphism_fixture(Ps, field; rank_part=r, ker_part=k, coker_part=c, seed=Int(0xA111))
        ccs = MD._get_cover_cache(Ps)
        # Warm/parity
        Cs_new, qs_new = AC._cokernel_module(fs; cache=ccs)
        Cs_old, qs_old = _cokernel_module_old(fs; cache=ccs)
        _pmodule_equal(field, Cs_new, Cs_old) || error("small-case cokernel parity failed")
        _morphism_equal(field, qs_new, qs_old) || error("small-case projection parity failed")

        bs_old = _bench("cokernel old (small case)", () -> _cokernel_module_old(fs; cache=ccs); reps=reps)
        bs_new = _bench("cokernel new (small case)", () -> AC._cokernel_module(fs; cache=ccs); reps=reps)
        println("speedup(new/old): ", round(bs_old.ms / bs_new.ms, digits=2), "x")
    end

    if fan_sweep_left > 0
        function _run_fanout_case(label::AbstractString, right::Int, seed::Int)
            rr = max(1, right)
            println()
            println("== ", label, " ==")
            Pf = _two_layer_poset(fan_sweep_left, rr)
            _, _, ff = _build_morphism_fixture(Pf, field;
                                               rank_part=r,
                                               ker_part=k,
                                               coker_part=c,
                                               seed=seed)
            ccf = MD._get_cover_cache(Pf)
            Cf_new, qf_new = AC._cokernel_module(ff; cache=ccf)
            Cf_old, qf_old = _cokernel_module_old(ff; cache=ccf)
            _pmodule_equal(field, Cf_new, Cf_old) || error("fanout cokernel parity failed ($label)")
            _morphism_equal(field, qf_new, qf_old) || error("fanout projection parity failed ($label)")

            bf_old = _bench("cokernel old ($label)", () -> _cokernel_module_old(ff; cache=ccf); reps=reps)
            bf_new = _bench("cokernel new ($label)", () -> AC._cokernel_module(ff; cache=ccf); reps=reps)
            println("speedup(new/old): ", round(bf_old.ms / bf_new.ms, digits=2), "x")
        end
        _run_fanout_case("fanout low", fan_sweep_low, Int(0xA112))
        _run_fanout_case("fanout mid", fan_sweep_mid, Int(0xA113))
        _run_fanout_case("fanout high", fan_sweep_high, Int(0xA114))
    end
end

main()
