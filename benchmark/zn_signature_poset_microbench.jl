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
const OPT = TO.Options
const DT = TO.DataTypes
const EC = TO.EncodingCore
const RES = TO.Results
const FF = TO.FiniteFringe
const FZ = TO.FlangeZn
const ZE = TO.ZnEncoding

function _parse_arg(args, key::String, default::Int)
    for a in args
        startswith(a, key * "=") || continue
        return max(1, parse(Int, split(a, "=", limit=2)[2]))
    end
    return default
end

function _bench(name::AbstractString, f::Function; reps::Int=8)
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
    println(rpad(name, 50), " median_time=", round(med_ms, digits=3),
            " ms  median_alloc=", round(med_kib, digits=1), " KiB")
    return (ms=med_ms, kib=med_kib)
end

function _random_face(n::Int, rng::AbstractRNG)
    coords = Vector{Bool}(undef, n)
    @inbounds for i in 1:n
        coords[i] = rand(rng) < 0.30
    end
    return FZ.face(n, coords)
end

function _rand_phi(field::CM.AbstractCoeffField, nrows::Int, ncols::Int, rng::AbstractRNG)
    K = CM.coeff_type(field)
    out = Matrix{K}(undef, nrows, ncols)
    @inbounds for i in 1:nrows, j in 1:ncols
        out[i, j] = CM.coerce(field, rand(rng, -2:2))
    end
    return out
end

function _fixture(; n::Int=3, nflats::Int=16, ninj::Int=16, seed::UInt=0x5a4e5f534947)
    rng = Random.MersenneTwister(seed)
    field = CM.QQField()
    K = CM.coeff_type(field)

    flats = Vector{FZ.IndFlat{n}}(undef, nflats)
    injectives = Vector{FZ.IndInj{n}}(undef, ninj)
    @inbounds for i in 1:nflats
        b = [rand(rng, -3:4) for _ in 1:n]
        flats[i] = FZ.IndFlat(_random_face(n, rng), b; id=Symbol(:F, i))
    end
    @inbounds for j in 1:ninj
        b = [rand(rng, -2:5) for _ in 1:n]
        injectives[j] = FZ.IndInj(_random_face(n, rng), b; id=Symbol(:E, j))
    end
    phi = _rand_phi(field, ninj, nflats, rng)
    FG = FZ.Flange{K}(n, flats, injectives, phi; field=field)

    opts = OPT.EncodingOptions(backend=:zn, max_regions=400_000)
    P, pi = ZE.encode_poset_from_flanges(FG, opts; poset_kind=:signature)

    flat_index, inj_index = ZE._generator_index_dicts(pi.flats, pi.injectives)
    flat_idxs = [flat_index[ZE._flat_key(F)] for F in FG.flats]
    inj_idxs = [inj_index[ZE._inj_key(E)] for E in FG.injectives]
    query_idxs = [rand(rng, 1:FF.nvertices(P)) for _ in 1:min(300, FF.nvertices(P))]
    return (FG=FG, P=P, pi=pi, flat_idxs=flat_idxs, inj_idxs=inj_idxs, query_idxs=query_idxs)
end

function _images_on_P_closure_baseline(P::FF.AbstractPoset,
                                       sig_y::AbstractVector{<:AbstractVector{Bool}},
                                       sig_z::AbstractVector{<:AbstractVector{Bool}},
                                       flat_idxs::AbstractVector{<:Integer},
                                       inj_idxs::AbstractVector{<:Integer})
    m = length(flat_idxs)
    r = length(inj_idxs)
    n = FF.nvertices(P)
    Uhat = Vector{FF.Upset}(undef, m)
    Dhat = Vector{FF.Downset}(undef, r)

    @inbounds for (loc, i0) in enumerate(flat_idxs)
        i = Int(i0)
        mask = falses(n)
        for t in 1:n
            mask[t] = sig_y[t][i]
        end
        Uhat[loc] = FF.upset_closure(P, mask)
    end
    @inbounds for (loc, j0) in enumerate(inj_idxs)
        j = Int(j0)
        mask = falses(n)
        for t in 1:n
            mask[t] = !sig_z[t][j]
        end
        Dhat[loc] = FF.downset_closure(P, mask)
    end
    return Uhat, Dhat
end

function _cover_edges_dense_baseline(P::ZE.SignaturePoset)
    L = FF.leq_matrix(P)
    Pdense = FF.FinitePoset(L; check=false)
    return FF.cover_edges(Pdense; cached=false)
end

function _updown_scan_baseline(P::ZE.SignaturePoset, idxs::Vector{Int})
    n = FF.nvertices(P)
    s = 0
    for i in idxs
        up = 0
        down = 0
        for j in 1:n
            FF.leq(P, i, j) && (up += 1)
            FF.leq(P, j, i) && (down += 1)
        end
        s += up + down
    end
    return s
end

function _updown_specialized(P::ZE.SignaturePoset, idxs::Vector{Int})
    s = 0
    for i in idxs
        for _ in FF.upset_indices(P, i)
            s += 1
        end
        for _ in FF.downset_indices(P, i)
            s += 1
        end
    end
    return s
end

@inline function _sig_subset_naive(a::AbstractVector{Bool}, b::AbstractVector{Bool})
    length(a) == length(b) || error("_sig_subset_naive: signature length mismatch")
    @inbounds for i in eachindex(a, b)
        a[i] && !b[i] && return false
    end
    return true
end

function _uptight_from_signatures_old(sig_y::AbstractVector{<:AbstractVector{Bool}},
                                      sig_z::AbstractVector{<:AbstractVector{Bool}})
    n = length(sig_y)
    L = falses(n, n)
    @inbounds for i in 1:n, j in 1:n
        L[i, j] = _sig_subset_naive(sig_y[i], sig_y[j]) &&
                  _sig_subset_naive(sig_z[i], sig_z[j])
    end
    @inbounds for k in 1:n, i in 1:n, j in 1:n
        L[i, j] = L[i, j] || (L[i, k] && L[k, j])
    end
    return FF.FinitePoset(L; check=false)
end

function _pushforward_strict_baseline(P::FF.AbstractPoset, pi::ZE.ZnEncodingMap, FG::FZ.Flange)
    flat_index, inj_index = ZE._generator_index_dicts(pi.flats, pi.injectives)

    flat_idxs = Vector{Int}(undef, length(FG.flats))
    @inbounds for i in 1:length(FG.flats)
        flat_idxs[i] = flat_index[ZE._flat_key(FG.flats[i])]
    end

    inj_idxs = Vector{Int}(undef, length(FG.injectives))
    @inbounds for j in 1:length(FG.injectives)
        inj_idxs[j] = inj_index[ZE._inj_key(FG.injectives[j])]
    end

    Uhat, Dhat = ZE._images_on_P(P, pi.sig_y, pi.sig_z, flat_idxs, inj_idxs)
    Phi = ZE._monomialize_phi(FG.phi, Uhat, Dhat)
    return FF.FringeModule{CM.coeff_type(FG.field)}(P, Uhat, Dhat, Phi; field=FG.field)
end

function _repeat_pushforward_new(P::FF.AbstractPoset, pi::ZE.ZnEncodingMap, FG::FZ.Flange, reps::Int)
    s = 0
    for _ in 1:reps
        H = ZE._pushforward_flange_to_fringe(P, pi, FG; strict=true)
        s += length(H.U) + length(H.D)
    end
    return s
end

function _repeat_pushforward_baseline(P::FF.AbstractPoset, pi::ZE.ZnEncodingMap, FG::FZ.Flange, reps::Int)
    s = 0
    for _ in 1:reps
        H = _pushforward_strict_baseline(P, pi, FG)
        s += length(H.U) + length(H.D)
    end
    return s
end

function _repeat_pmodule_direct(P::FF.AbstractPoset, pi::ZE.ZnEncodingMap, FG::FZ.Flange, reps::Int)
    s = 0
    for _ in 1:reps
        plan, flat_masks, inj_masks = ZE._strict_pushforward_plan_and_masks(pi, FG)
        M = ZE._pmodule_from_pushforward_plan(P, FG, plan, flat_masks, inj_masks)
        s += sum(M.dims)
    end
    return s
end

function _repeat_pmodule_via_fringe(P::FF.AbstractPoset, pi::ZE.ZnEncodingMap, FG::FZ.Flange, reps::Int)
    s = 0
    for _ in 1:reps
        H = ZE._pushforward_flange_to_fringe(P, pi, FG; strict=true)
        M = TamerOp.IndicatorResolutions.pmodule_from_fringe(H)
        s += sum(M.dims)
    end
    return s
end

function _repeat_workflow_encode(FG::FZ.Flange, reps::Int; use_session::Bool, warm_session::Bool=false)
    cache = use_session ? CM.SessionCache() : nothing
    if use_session && warm_session
        TamerOp.encode(FG; backend=:zn, cache=cache)
    end
    total = 0
    for _ in 1:reps
        enc = TamerOp.encode(FG; backend=:zn, cache=cache)
        total += length(enc.H.U) + length(enc.H.D)
    end
    return total
end

function _repeat_plan_build_fresh_maps(FG::FZ.Flange,
                                       opts::OPT.EncodingOptions,
                                       reps::Int;
                                       use_session::Bool)
    cache = use_session ? CM.SessionCache() : nothing
    total = 0
    for _ in 1:reps
        P, pi = ZE.encode_poset_from_flanges((FG,), opts;
                                             poset_kind=:signature,
                                             session_cache=cache)
        plan = ZE._get_or_build_pushforward_plan!(pi, FG; session_cache=cache)
        total += length(plan.zero_pairs) + FF.nvertices(P)
    end
    return total
end

function main(; reps::Int=8, nflats::Int=16, ninj::Int=16)
    fx = _fixture(; nflats=nflats, ninj=ninj)
    P = fx.P
    pi = fx.pi
    flat_idxs = fx.flat_idxs
    inj_idxs = fx.inj_idxs
    idxs = fx.query_idxs
    n = FF.nvertices(P)

    println("Zn SignaturePoset micro-benchmark")
    println("reps=$(reps), nflats=$(nflats), ninj=$(ninj), nregions=$(n)\n")

    c_new = _bench("cover_edges(SignaturePoset)", () -> FF.cover_edges(P; cached=false); reps=reps)
    c_old = _bench("cover_edges dense baseline", () -> _cover_edges_dense_baseline(P); reps=reps)
    println("  speedup baseline/new: ", round(c_old.ms / max(1e-9, c_new.ms), digits=3), "x")
    println("  alloc ratio baseline/new: ", round(c_old.kib / max(1e-9, c_new.kib), digits=3), "x\n")

    u_new = _bench("up/down specialized methods", () -> _updown_specialized(P, idxs); reps=reps)
    u_old = _bench("up/down leq-scan baseline", () -> _updown_scan_baseline(P, idxs); reps=reps)
    println("  speedup baseline/new: ", round(u_old.ms / max(1e-9, u_new.ms), digits=3), "x")
    println("  alloc ratio baseline/new: ", round(u_old.kib / max(1e-9, u_new.kib), digits=3), "x\n")

    i_new = _bench("_images_on_P direct masks", () -> ZE._images_on_P(P, pi.sig_y, pi.sig_z, flat_idxs, inj_idxs); reps=reps)
    i_old = _bench("_images_on_P closure baseline", () -> _images_on_P_closure_baseline(P, pi.sig_y, pi.sig_z, flat_idxs, inj_idxs); reps=reps)
    println("  speedup baseline/new: ", round(i_old.ms / max(1e-9, i_new.ms), digits=3), "x")
    println("  alloc ratio baseline/new: ", round(i_old.kib / max(1e-9, i_new.kib), digits=3), "x\n")

    # Isolate the monomialization change: precomputed zero pairs vs per-call intersects loop.
    Hmono = _pushforward_strict_baseline(P, pi, fx.FG)
    plan = ZE._get_or_build_pushforward_plan!(pi, fx.FG)
    m_new = _bench("monomialize precomputed zeros", () -> ZE._monomialize_phi_with_zero_pairs(fx.FG.phi, plan.zero_pairs); reps=reps)
    m_old = _bench("monomialize intersects loop", () -> ZE._monomialize_phi(fx.FG.phi, Hmono.U, Hmono.D); reps=reps)
    println("  speedup baseline/new: ", round(m_old.ms / max(1e-9, m_new.ms), digits=3), "x")
    println("  alloc ratio baseline/new: ", round(m_old.kib / max(1e-9, m_new.kib), digits=3), "x\n")

    p_new = _bench("strict pushforward cached", () -> _repeat_pushforward_new(P, pi, fx.FG, 20); reps=reps)
    p_old = _bench("strict pushforward baseline", () -> _repeat_pushforward_baseline(P, pi, fx.FG, 20); reps=reps)
    println("  speedup baseline/new: ", round(p_old.ms / max(1e-9, p_new.ms), digits=3), "x")
    println("  alloc ratio baseline/new: ", round(p_old.kib / max(1e-9, p_new.kib), digits=3), "x\n")

    d_new = _bench("dense uptight new (no closure)", () -> ZE._uptight_from_signatures(pi.sig_y, pi.sig_z); reps=reps)
    d_old = _bench("dense uptight old closure", () -> _uptight_from_signatures_old(pi.sig_y, pi.sig_z); reps=reps)
    println("  speedup baseline/new: ", round(d_old.ms / max(1e-9, d_new.ms), digits=3), "x")
    println("  alloc ratio baseline/new: ", round(d_old.kib / max(1e-9, d_new.kib), digits=3), "x")

    pm_new = _bench("pmodule direct strict plan", () -> _repeat_pmodule_direct(P, pi, fx.FG, 8); reps=reps)
    pm_old = _bench("pmodule via fringe baseline", () -> _repeat_pmodule_via_fringe(P, pi, fx.FG, 8); reps=reps)
    println("  speedup baseline/new: ", round(pm_old.ms / max(1e-9, pm_new.ms), digits=3), "x")
    println("  alloc ratio baseline/new: ", round(pm_old.kib / max(1e-9, pm_new.kib), digits=3), "x\n")

    opts = OPT.EncodingOptions(backend=:zn, max_regions=400_000)
    s_new = _bench("session encode+plan reuse", () -> _repeat_plan_build_fresh_maps(fx.FG, opts, 6; use_session=true); reps=reps)
    s_old = _bench("no session encode+plan reuse", () -> _repeat_plan_build_fresh_maps(fx.FG, opts, 6; use_session=false); reps=reps)
    println("  speedup baseline/new: ", round(s_old.ms / max(1e-9, s_new.ms), digits=3), "x")
    println("  alloc ratio baseline/new: ", round(s_old.kib / max(1e-9, s_new.kib), digits=3), "x\n")

    w_new = _bench("workflow encode session warm hits", () -> _repeat_workflow_encode(fx.FG, 6; use_session=true, warm_session=true); reps=reps)
    w_old = _bench("workflow encode no shared cache", () -> _repeat_workflow_encode(fx.FG, 6; use_session=false); reps=reps)
    println("  speedup baseline/new: ", round(w_old.ms / max(1e-9, w_new.ms), digits=3), "x")
    println("  alloc ratio baseline/new: ", round(w_old.kib / max(1e-9, w_new.kib), digits=3), "x")
end

if abspath(PROGRAM_FILE) == @__FILE__
    reps = _parse_arg(ARGS, "--reps", 8)
    nflats = _parse_arg(ARGS, "--nflats", 16)
    ninj = _parse_arg(ARGS, "--ninj", 16)
    main(; reps=reps, nflats=nflats, ninj=ninj)
end
