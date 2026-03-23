#!/usr/bin/env julia

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
        return parse(Float64, split(a, "=", limit=2)[2])
    end
    return default
end

function _bench(name::AbstractString, f::Function; reps::Int=7)
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
    println(rpad(name, 34), " median_time=", round(med_ms, digits=3),
            " ms  median_alloc=", round(med_kib, digits=1), " KiB")
    return (ms=med_ms, kib=med_kib)
end

function _random_poset(n::Int; p::Float64=0.03, seed::Int=0xC0DE)
    rng = Random.MersenneTwister(seed)
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

function _random_fringe_module(P::FF.AbstractPoset, field::CM.AbstractCoeffField;
                               nu::Int=48, nd::Int=48, density::Float64=0.15, seed::Int=0xBEEF)
    rng = Random.MersenneTwister(seed)
    K = CM.coeff_type(field)
    n = FF.nvertices(P)
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
            phi[j, i] = CM.coerce(field, rand(rng, 1:3))
        end
    end
    return FF.FringeModule{K}(P, U, D, phi; field=field)
end

function _uptight_poset_dense_old(Q::FF.AbstractPoset, regions::Vector{Vector{Int}})
    r = length(regions)
    rel = falses(r, r)
    for A in 1:r, B in 1:r
        if A == B
            rel[A, B] = true
            continue
        end
        found = false
        for a in regions[A], b in regions[B]
            if FF.leq(Q, a, b)
                found = true
                break
            end
        end
        rel[A, B] = found
    end
    for k in 1:r, i in 1:r, j in 1:r
        rel[i, j] = rel[i, j] || (rel[i, k] && rel[k, j])
    end
    return FF.FinitePoset(rel; check=false)
end

function _image_upset_old(pi::EN.EncodingMap, U::FF.Upset)
    maskP = falses(FF.nvertices(pi.P))
    for q in 1:FF.nvertices(pi.Q)
        if U.mask[q]
            maskP[pi.pi_of_q[q]] = true
        end
    end
    return FF.upset_closure(pi.P, maskP)
end

function _image_downset_old(pi::EN.EncodingMap, D::FF.Downset)
    maskP = falses(FF.nvertices(pi.P))
    for q in 1:FF.nvertices(pi.Q)
        if D.mask[q]
            maskP[pi.pi_of_q[q]] = true
        end
    end
    return FF.downset_closure(pi.P, maskP)
end

function _preimage_upset_old(pi::EN.EncodingMap, Uhat::FF.Upset)
    maskQ = falses(FF.nvertices(pi.Q))
    for q in 1:FF.nvertices(pi.Q)
        if Uhat.mask[pi.pi_of_q[q]]
            maskQ[q] = true
        end
    end
    return FF.upset_closure(pi.Q, maskQ)
end

function _preimage_downset_old(pi::EN.EncodingMap, Dhat::FF.Downset)
    maskQ = falses(FF.nvertices(pi.Q))
    for q in 1:FF.nvertices(pi.Q)
        if Dhat.mask[pi.pi_of_q[q]]
            maskQ[q] = true
        end
    end
    return FF.downset_closure(pi.Q, maskQ)
end

function _pushforward_old(H::FF.FringeModule, pi::EN.EncodingMap)
    Uhat = [_image_upset_old(pi, U) for U in H.U]
    Dhat = [_image_downset_old(pi, D) for D in H.D]
    return FF.FringeModule{eltype(H.phi)}(pi.P, Uhat, Dhat, H.phi; field=H.field)
end

function _pullback_old(Hhat::FF.FringeModule, pi::EN.EncodingMap)
    UQ = [_preimage_upset_old(pi, Uhat) for Uhat in Hhat.U]
    DQ = [_preimage_downset_old(pi, Dhat) for Dhat in Hhat.D]
    return FF.FringeModule{eltype(Hhat.phi)}(pi.Q, UQ, DQ, Hhat.phi; field=Hhat.field)
end

function _repeat_labels(H::FF.FringeModule{K}; mult::Int=8) where {K}
    U = Vector{eltype(H.U)}(undef, length(H.U) * mult)
    D = Vector{eltype(H.D)}(undef, length(H.D) * mult)
    @inbounds for rep in 1:mult
        offu = (rep - 1) * length(H.U)
        offd = (rep - 1) * length(H.D)
        for i in eachindex(H.U)
            U[offu + i] = H.U[i]
        end
        for j in eachindex(H.D)
            D[offd + j] = H.D[j]
        end
    end
    phi = kron(spdiagm(0 => ones(Int, mult)), H.phi)
    return FF.FringeModule{K}(H.P, U, D, SparseMatrixCSC{K,Int}(phi); field=H.field)
end

function main(args=ARGS)
    reps = _parse_int_arg(args, "--reps", 7)
    n = _parse_int_arg(args, "--n", 120)
    nu = _parse_int_arg(args, "--nu", 48)
    nd = _parse_int_arg(args, "--nd", 48)
    calls = _parse_int_arg(args, "--calls", 32)
    density = _parse_float_arg(args, "--density", 0.15)

    println("Encoding microbench")
    println("reps=$(reps), n=$(n), nu=$(nu), nd=$(nd), calls=$(calls), density=$(density)\n")

    P = _random_poset(n; p=0.03, seed=Int(0x9101))
    field = CM.QQField()
    H = _random_fringe_module(P, field; nu=nu, nd=nd, density=density, seed=Int(0x9102))

    upt_regions = EN.build_uptight_encoding_from_fringe(H; poset_kind=:regions)
    upt_dense = EN.build_uptight_encoding_from_fringe(H; poset_kind=:dense)
    pi = upt_regions.pi
    Hhat = EN.pushforward_fringe_along_encoding(H, pi)
    Hrep = _repeat_labels(H; mult=8)
    Hrep_hat = EN.pushforward_fringe_along_encoding(Hrep, pi)

    # Parity checks
    @assert eltype(upt_regions.Y) === FF.Upset{typeof(P)}
    dense_old = _uptight_poset_dense_old(P, EN._uptight_regions(P, upt_regions.Y))
    @assert FF.leq_matrix(dense_old) == FF.leq_matrix(upt_dense.pi.P)

    Hhat_old = _pushforward_old(H, pi)
    @assert [u.mask for u in Hhat_old.U] == [u.mask for u in Hhat.U]
    @assert [d.mask for d in Hhat_old.D] == [d.mask for d in Hhat.D]
    @assert Hhat_old.phi == Hhat.phi

    Hpb_old = _pullback_old(Hhat, pi)
    Hpb_new = EN.pullback_fringe_along_encoding(Hhat, pi)
    @assert [u.mask for u in Hpb_old.U] == [u.mask for u in Hpb_new.U]
    @assert [d.mask for d in Hpb_old.D] == [d.mask for d in Hpb_new.D]
    @assert Hpb_old.phi == Hpb_new.phi

    println("== Push/Pull along finite encoding ==")
    b_push_old = _bench("pushforward old (scan)", () -> _pushforward_old(H, pi); reps=reps)
    b_push_new = _bench("pushforward new (fibers)", () -> EN.pushforward_fringe_along_encoding(H, pi); reps=reps)
    b_pull_old = _bench("pullback old (scan)", () -> _pullback_old(Hhat, pi); reps=reps)
    b_pull_new = _bench("pullback new (fibers)", () -> EN.pullback_fringe_along_encoding(Hhat, pi); reps=reps)
    println("speedup push new/old: ", round(b_push_old.ms / b_push_new.ms, digits=2), "x")
    println("speedup pull new/old: ", round(b_pull_old.ms / b_pull_new.ms, digits=2), "x")
    println()

    println("== Push/Pull with repeated labels ==")
    b_push_old_rep = _bench("pushforward old repeated", () -> _pushforward_old(Hrep, pi); reps=reps)
    b_push_new_rep = _bench("pushforward new repeated", () -> EN.pushforward_fringe_along_encoding(Hrep, pi); reps=reps)
    b_pull_old_rep = _bench("pullback old repeated", () -> _pullback_old(Hrep_hat, pi); reps=reps)
    b_pull_new_rep = _bench("pullback new repeated", () -> EN.pullback_fringe_along_encoding(Hrep_hat, pi); reps=reps)
    println("speedup push repeated new/old: ", round(b_push_old_rep.ms / b_push_new_rep.ms, digits=2), "x")
    println("speedup pull repeated new/old: ", round(b_pull_old_rep.ms / b_pull_new_rep.ms, digits=2), "x")
    println()

    println("== Repeated calls on same encoding ==")
    EN._clear_encoding_label_cache!(pi)
    b_push_old_calls = _bench("pushforward old calls", () -> begin
        for _ in 1:calls
            _pushforward_old(H, pi)
        end
    end; reps=reps)
    EN._clear_encoding_label_cache!(pi)
    b_push_new_calls = _bench("pushforward new calls", () -> begin
        for _ in 1:calls
            EN.pushforward_fringe_along_encoding(H, pi)
        end
    end; reps=reps)
    EN._clear_encoding_label_cache!(pi)
    b_pull_old_calls = _bench("pullback old calls", () -> begin
        for _ in 1:calls
            _pullback_old(Hhat, pi)
        end
    end; reps=reps)
    EN._clear_encoding_label_cache!(pi)
    b_pull_new_calls = _bench("pullback new calls", () -> begin
        for _ in 1:calls
            EN.pullback_fringe_along_encoding(Hhat, pi)
        end
    end; reps=reps)
    println("speedup push calls new/old: ", round(b_push_old_calls.ms / b_push_new_calls.ms, digits=2), "x")
    println("speedup pull calls new/old: ", round(b_pull_old_calls.ms / b_pull_new_calls.ms, digits=2), "x")
    println()

    regions = EN._uptight_regions(P, upt_regions.Y)
    println("== Dense uptight poset build ==")
    b_dense_old = _bench("dense poset old", () -> _uptight_poset_dense_old(P, regions); reps=reps)
    b_dense_new = _bench("dense poset new", () -> EN._uptight_poset(P, regions; poset_kind=:dense); reps=reps)
    println("speedup dense new/old: ", round(b_dense_old.ms / b_dense_new.ms, digits=2), "x")
end

main()
