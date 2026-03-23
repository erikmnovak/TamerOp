#!/usr/bin/env julia
#
# abelian_categories_scorecard.jl
#
# Purpose
# - Provide a practical speed + robustness scorecard for `src/AbelianCategories.jl`.
#
# Coverage
# - Core kernels:
#   - kernel_with_inclusion
#   - image_with_inclusion
#   - _cokernel_module
# - Derived categorical operations:
#   - equalizer / coequalizer
#   - pushout / pullback
#   - short exact sequence checks
# - Robustness checks:
#   - explicit-cache vs auto-cache parity
#   - field sweep (QQ/F2/F3/Fp/RealField)
#   - fast-solve toggle parity for cokernel solves
#   - fanout regime for batched cokernel path
#
# Usage
#   julia --project=. benchmark/abelian_categories_scorecard.jl
#   julia --project=. benchmark/abelian_categories_scorecard.jl --reps=7 --n=32 --fields=qq,f2,f3,real
#   julia --project=. benchmark/abelian_categories_scorecard.jl --fan_left=12 --fan_right=24
#

using Random
using LinearAlgebra
using Dates

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

function _parse_str_arg(args, key::String, default::String)
    for a in args
        startswith(a, key * "=") || continue
        return String(split(a, "=", limit=2)[2])
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
    println(rpad(name, 44), " median_time=", round(med_ms, digits=3),
            " ms  median_alloc=", round(med_kib, digits=1), " KiB")
    return (ms=med_ms, kib=med_kib)
end

function _bench_per_call(name::AbstractString, f::Function; reps::Int=7, inner::Int=2048)
    GC.gc()
    for _ in 1:inner
        f()
    end
    GC.gc()
    times_us = Vector{Float64}(undef, reps)
    bytes_kib = Vector{Float64}(undef, reps)
    for i in 1:reps
        m = @timed begin
            for _ in 1:inner
                f()
            end
        end
        times_us[i] = 1.0e6 * m.time / inner
        bytes_kib[i] = (m.bytes / 1024.0) / inner
    end
    sort!(times_us)
    sort!(bytes_kib)
    med_us = times_us[cld(reps, 2)]
    med_kib = bytes_kib[cld(reps, 2)]
    println(rpad(name, 44), " median_time=", round(med_us, digits=3),
            " us/call  median_alloc=", round(med_kib, digits=3), " KiB/call")
    return (ms=med_us / 1000.0, kib=med_kib)
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
    L = falses(n, n)
    @inbounds for i in 1:n, j in i:n
        L[i, j] = true
    end
    return FF.FinitePoset(L; check=false)
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

function _routing_fixture(field::CM.AbstractCoeffField)
    K = CM.coeff_type(field)
    c(x) = CM.coerce(field, x)
    P = _chain_poset(1)
    empty_edges = Dict{Tuple{Int,Int}, Matrix{K}}()

    A = MD.PModule{K}(P, [2], empty_edges; field=field)
    B = MD.PModule{K}(P, [3], empty_edges; field=field)
    C = MD.PModule{K}(P, [1], empty_edges; field=field)

    f = MD.PMorphism(A, B, [K[c(1) c(0);
                            c(0) c(0);
                            c(0) c(0)]])
    z = MD.zero_morphism(A, B)
    cc = MD._get_cover_cache(P)
    im_sub = AC.image_submodule(f)
    ker_sub = AC.kernel_submodule(f; cache=cc)
    discrete = AC.DiscretePairDiagram(A, B)
    parallel = AC.ParallelPairDiagram(f, z)
    span = AC.SpanDiagram(f, z)
    cospan = AC.CospanDiagram(f, z)

    topA = MD.PModule{K}(P, [1], empty_edges; field=field)
    topB = MD.PModule{K}(P, [2], empty_edges; field=field)
    topC = MD.PModule{K}(P, [1], empty_edges; field=field)
    it = MD.PMorphism(topA, topB, [reshape(K[c(1), c(0)], 2, 1)])
    pt = MD.PMorphism(topB, topC, [reshape(K[c(0), c(1)], 1, 2)])
    top = AC.ShortExactSequence(it, pt)

    botA = MD.PModule{K}(P, [2], empty_edges; field=field)
    botB = MD.PModule{K}(P, [3], empty_edges; field=field)
    botC = MD.PModule{K}(P, [1], empty_edges; field=field)
    ib = MD.PMorphism(botA, botB, [K[c(1) c(0);
                                   c(0) c(1);
                                   c(0) c(0)]])
    pb = MD.PMorphism(botB, botC, [reshape(K[c(0), c(0), c(1)], 1, 3)])
    bottom = AC.ShortExactSequence(ib, pb)

    alpha = MD.PMorphism(topA, botA, [reshape(K[c(1), c(0)], 2, 1)])
    beta = MD.PMorphism(topB, botB, [K[c(1) c(0);
                                     c(0) c(1);
                                     c(0) c(0)]])
    gamma = MD.PMorphism(topC, botC, [reshape(K[c(0)], 1, 1)])

    return (; P, A, B, C, f, z, cc, im_sub, ker_sub, discrete, parallel, span, cospan,
            top, bottom, it, pt, ib, pb, alpha, beta, gamma)
end

function _field_from_token(tok::AbstractString)
    t = lowercase(strip(tok))
    if t == "qq"
        return CM.QQField()
    elseif t == "f2"
        return CM.F2()
    elseif t == "f3"
        return CM.F3()
    elseif startswith(t, "fp")
        p = parse(Int, t[3:end])
        return CM.Fp(p)
    elseif t == "real"
        return CM.RealField(Float64)
    end
    error("unknown field token '$tok' (use qq,f2,f3,fp7,real,...)")
end

function _parse_fields(args)
    spec = _parse_str_arg(args, "--fields", "qq,f3,real")
    toks = split(spec, ',')
    isempty(toks) && error("empty --fields specification")
    return [_field_from_token(tok) for tok in toks]
end

@inline function _matrix_equal(field::CM.AbstractCoeffField, A::AbstractMatrix, B::AbstractMatrix)
    size(A) == size(B) || return false
    if field isa CM.RealField
        return isapprox(A, B; rtol=field.rtol, atol=field.atol)
    end
    return A == B
end

function _morphism_equal(field::CM.AbstractCoeffField, a::MD.PMorphism, b::MD.PMorphism)
    a.dom.dims == b.dom.dims || return false
    a.cod.dims == b.cod.dims || return false
    for i in eachindex(a.comps)
        _matrix_equal(field, a.comps[i], b.comps[i]) || return false
    end
    return true
end

function _pmodule_equal(field::CM.AbstractCoeffField, A::MD.PModule, B::MD.PModule)
    A.dims == B.dims || return false
    for ((u, v), Muv) in A.edge_maps
        Nuv = B.edge_maps[u, v]
        _matrix_equal(field, Muv, Nuv) || return false
    end
    return true
end

function _check_cache_parity(field, f::MD.PMorphism, cc::MD.CoverCache)
    K1, i1 = AC.kernel_with_inclusion(f; cache=cc)
    K2, i2 = AC.kernel_with_inclusion(f; cache=:auto)
    _pmodule_equal(field, K1, K2) || error("kernel cache parity failed")
    _morphism_equal(field, i1, i2) || error("kernel inclusion cache parity failed")

    Im1, j1 = AC.image_with_inclusion(f; cache=cc)
    Im2, j2 = AC.image_with_inclusion(f; cache=:auto)
    _pmodule_equal(field, Im1, Im2) || error("image cache parity failed")
    _morphism_equal(field, j1, j2) || error("image inclusion cache parity failed")

    C1, q1 = AC._cokernel_module(f; cache=cc)
    C2, q2 = AC._cokernel_module(f; cache=:auto)
    _pmodule_equal(field, C1, C2) || error("cokernel cache parity failed")
    _morphism_equal(field, q1, q2) || error("cokernel projection cache parity failed")
end

function _check_equalizer_coequalizer_identities(field, f::MD.PMorphism, cc::MD.CoverCache)
    E, e = AC.equalizer(f, f; cache=cc)
    E.dims == f.dom.dims || error("equalizer(f,f) should match dom dimensions")
    for i in eachindex(E.dims)
        di = f.dom.dims[i]
        ri = FL.rank(field, e.comps[i])
        ri == di || error("equalizer inclusion should be an isomorphism at vertex $i")
    end

    Q, q = AC.coequalizer(f, f; cache=cc)
    Q.dims == f.cod.dims || error("coequalizer(f,f) should match cod dimensions")
    for i in eachindex(Q.dims)
        di = f.cod.dims[i]
        ri = FL.rank(field, q.comps[i])
        ri == di || error("coequalizer projection should be an isomorphism at vertex $i")
    end
end

function _check_pushout_pullback_commutativity(field, f::MD.PMorphism, cc::MD.CoverCache)
    Pout, inB, inC, _, _ = AC.pushout(f, f; cache=cc)
    n = FF.nvertices(Pout.Q)
    for u in 1:n
        lhs = inB.comps[u] * f.comps[u]
        rhs = inC.comps[u] * f.comps[u]
        _matrix_equal(field, lhs, rhs) || error("pushout commutativity failed at vertex $u")
    end

    Pin, prB, prC, _, _ = AC.pullback(f, f; cache=cc)
    n2 = FF.nvertices(Pin.Q)
    for u in 1:n2
        lhs = f.comps[u] * prB.comps[u]
        rhs = f.comps[u] * prC.comps[u]
        _matrix_equal(field, lhs, rhs) || error("pullback commutativity failed at vertex $u")
    end
end

function _check_short_exact_sequence(field, P, r::Int, k::Int, seed::Int)
    _, Bepi, fepi = _build_morphism_fixture(P, field;
                                            rank_part=r,
                                            ker_part=k,
                                            coker_part=1,
                                            seed=seed)
    # Make the map pointwise surjective by projecting onto full codomain rank.
    # Rebuild with coker_part=0 to force epi.
    Aepi, Bepi2, fepi2 = _build_morphism_fixture(P, field;
                                                 rank_part=r,
                                                 ker_part=k,
                                                 coker_part=0,
                                                 seed=seed + 1)
    cc = MD._get_cover_cache(P)
    Kmod, iK = AC.kernel_with_inclusion(fepi2; cache=cc)
    ses = AC.short_exact_sequence(iK, fepi2; check=true, cache=cc)
    AC.is_exact(ses; cache=cc) || error("short exact sequence check failed")
    # Sanity to avoid dead code elimination in benchmark harness setup.
    (length(Kmod.dims) == length(Aepi.dims) == length(Bepi2.dims)) || error("unexpected SES shape mismatch")
    return nothing
end

function _cokernel_with_flag(f::MD.PMorphism, cc::MD.CoverCache, fast::Bool)
    old = AC._FAST_SOLVE_NO_CHECK[]
    AC._FAST_SOLVE_NO_CHECK[] = fast
    try
        return AC._cokernel_module(f; cache=cc)
    finally
        AC._FAST_SOLVE_NO_CHECK[] = old
    end
end

function _append_row!(rows, field_name::String, probe::String, variant::String, b)
    push!(rows, (field=field_name, probe=probe, variant=variant,
                 median_ms=b.ms, median_kib=b.kib))
    return nothing
end

function _field_name(field::CM.AbstractCoeffField)
    if field isa CM.QQField
        return "qq"
    elseif field isa CM.RealField
        return "real"
    elseif field isa CM.PrimeField
        return "f$(field.p)"
    end
    return string(typeof(field))
end

function _write_rows(path::AbstractString, rows)
    open(path, "w") do io
        println(io, "timestamp,field,probe,variant,median_ms,median_kib")
        ts = Dates.format(Dates.now(Dates.UTC), dateformat"yyyy-mm-ddTHH:MM:SS")
        for r in rows
            println(io, string(ts, ",", r.field, ",", r.probe, ",", r.variant, ",",
                               r.median_ms, ",", r.median_kib))
        end
    end
end

function main(args=ARGS)
    reps = _parse_int_arg(args, "--reps", 5)
    route_inner = _parse_int_arg(args, "--route_inner", 2048)
    n = _parse_int_arg(args, "--n", 28)
    r = _parse_int_arg(args, "--rank", 5)
    k = _parse_int_arg(args, "--ker", 5)
    c = _parse_int_arg(args, "--coker", 5)
    robust_trials = _parse_int_arg(args, "--robust_trials", 3)
    fan_left = _parse_int_arg(args, "--fan_left", 8)
    fan_right = _parse_int_arg(args, "--fan_right", 16)
    out = _parse_str_arg(args, "--out", joinpath(@__DIR__, "_tmp_abelian_categories_scorecard.csv"))
    fields = _parse_fields(args)

    println("AbelianCategories scorecard")
    println("reps=$(reps), route_inner=$(route_inner), n=$(n), rank=$(r), ker=$(k), coker=$(c), robust_trials=$(robust_trials), fan_left=$(fan_left), fan_right=$(fan_right)")
    println("fields=", join((_field_name(f) for f in fields), ","))
    println()

    rows = NamedTuple[]

    for field in fields
        fname = _field_name(field)
        println("== field=", fname, " ==")
        P = _chain_poset(n)
        A, B, f = _build_morphism_fixture(P, field;
                                          rank_part=r,
                                          ker_part=k,
                                          coker_part=c,
                                          seed=Int(0xB001))
        cc = MD._get_cover_cache(P)

        # Robustness / parity checks over multiple deterministic seeds.
        for t in 1:robust_trials
            _, _, ft = _build_morphism_fixture(P, field;
                                               rank_part=r,
                                               ker_part=k,
                                               coker_part=c,
                                               seed=Int(0xB100 + t))
            cct = MD._get_cover_cache(P)
            _check_cache_parity(field, ft, cct)
            _check_equalizer_coequalizer_identities(field, ft, cct)
            _check_pushout_pullback_commutativity(field, ft, cct)
        end
        _check_short_exact_sequence(field, P, r, k, Int(0xB300))

        bk_cache = _bench("kernel_with_inclusion (cache)", () -> AC.kernel_with_inclusion(f; cache=cc); reps=reps)
        _append_row!(rows, fname, "kernel_with_inclusion", "cache", bk_cache)
        bk_auto = _bench("kernel_with_inclusion (auto)", () -> AC.kernel_with_inclusion(f; cache=:auto); reps=reps)
        _append_row!(rows, fname, "kernel_with_inclusion", "auto", bk_auto)

        bi_cache = _bench("image_with_inclusion (cache)", () -> AC.image_with_inclusion(f; cache=cc); reps=reps)
        _append_row!(rows, fname, "image_with_inclusion", "cache", bi_cache)
        bi_auto = _bench("image_with_inclusion (auto)", () -> AC.image_with_inclusion(f; cache=:auto); reps=reps)
        _append_row!(rows, fname, "image_with_inclusion", "auto", bi_auto)

        bc_cache = _bench("_cokernel_module (cache)", () -> AC._cokernel_module(f; cache=cc); reps=reps)
        _append_row!(rows, fname, "_cokernel_module", "cache", bc_cache)
        bc_auto = _bench("_cokernel_module (auto)", () -> AC._cokernel_module(f; cache=:auto); reps=reps)
        _append_row!(rows, fname, "_cokernel_module", "auto", bc_auto)

        beq = _bench("equalizer(f,f)", () -> AC.equalizer(f, f; cache=cc); reps=reps)
        _append_row!(rows, fname, "equalizer", "parallel_id", beq)
        bcoeq = _bench("coequalizer(f,f)", () -> AC.coequalizer(f, f; cache=cc); reps=reps)
        _append_row!(rows, fname, "coequalizer", "parallel_id", bcoeq)

        bpo = _bench("pushout(f,f)", () -> AC.pushout(f, f; cache=cc); reps=reps)
        _append_row!(rows, fname, "pushout", "parallel_same", bpo)
        bpb = _bench("pullback(f,f)", () -> AC.pullback(f, f; cache=cc); reps=reps)
        _append_row!(rows, fname, "pullback", "parallel_same", bpb)

        bses = _bench("short_exact_sequence check", () -> begin
            Aepi, Bepi, fepi = _build_morphism_fixture(P, field;
                                                       rank_part=r,
                                                       ker_part=k,
                                                       coker_part=0,
                                                       seed=Int(0xB200))
            ccepi = MD._get_cover_cache(P)
            Kmod, iK = AC.kernel_with_inclusion(fepi; cache=ccepi)
            ses = AC.short_exact_sequence(iK, fepi; check=true, cache=ccepi)
            AC.is_exact(ses; cache=ccepi)
            length(Kmod.dims) + length(Aepi.dims) + length(Bepi.dims)
        end; reps=reps)
        _append_row!(rows, fname, "short_exact_sequence", "check", bses)

        # Cokernel fast-solve toggle parity + timings.
        Cfast, qfast = _cokernel_with_flag(f, cc, true)
        Csafe, qsafe = _cokernel_with_flag(f, cc, false)
        _pmodule_equal(field, Cfast, Csafe) || error("cokernel parity failed between fast_solve flags")
        _morphism_equal(field, qfast, qsafe) || error("cokernel projection parity failed between fast_solve flags")

        bfast = _bench("_cokernel_module fast_solve=true", () -> _cokernel_with_flag(f, cc, true); reps=reps)
        _append_row!(rows, fname, "_cokernel_module", "fast_solve_on", bfast)
        bsafe = _bench("_cokernel_module fast_solve=false", () -> _cokernel_with_flag(f, cc, false); reps=reps)
        _append_row!(rows, fname, "_cokernel_module", "fast_solve_off", bsafe)

        # Fanout sweep: hits batched cokernel RHS path more aggressively.
        Pf = _two_layer_poset(fan_left, fan_right)
        _, _, ff = _build_morphism_fixture(Pf, field;
                                           rank_part=r,
                                           ker_part=k,
                                           coker_part=c,
                                           seed=Int(0xB500))
        ccf = MD._get_cover_cache(Pf)
        bfan = _bench("_cokernel_module fanout", () -> AC._cokernel_module(ff; cache=ccf); reps=reps)
        _append_row!(rows, fname, "_cokernel_module", "fanout_cache", bfan)

        rf = _routing_fixture(field)
        Ktiny, iKtiny = AC._kernel_with_inclusion_cached(rf.f, rf.cc)
        Imtiny, iImtiny = AC._image_with_inclusion_cached(rf.f, rf.cc)
        Ctiny, qtiny = AC._cokernel_module_cached(rf.f, rf.cc)
        Coimtiny, pctiny = AC._coimage_with_projection_cached(rf.f, rf.cc)
        Qtiny, qqtiny = AC._quotient_with_projection_cached(iImtiny, rf.cc)
        Epub, epub = AC.equalizer(rf.f, rf.z; cache=rf.cc)
        Ecore, ecore = AC._equalizer_cached(rf.f, rf.z, rf.cc)
        Qpub, qpub = AC.coequalizer(rf.f, rf.z; cache=rf.cc)
        Qcore, qcore = AC._coequalizer_cached(rf.f, rf.z, rf.cc)
        _pmodule_equal(field, Ktiny, AC.kernel_with_inclusion(rf.f; cache=:auto)[1]) || error("routing kernel parity failed")
        _morphism_equal(field, iKtiny, AC.kernel_with_inclusion(rf.f; cache=:auto)[2]) || error("routing kernel inclusion parity failed")
        _pmodule_equal(field, Imtiny, AC.image_with_inclusion(rf.f; cache=:auto)[1]) || error("routing image parity failed")
        _morphism_equal(field, iImtiny, AC.image_with_inclusion(rf.f; cache=:auto)[2]) || error("routing image inclusion parity failed")
        _pmodule_equal(field, Ctiny, AC._cokernel_module(rf.f; cache=:auto)[1]) || error("routing cokernel parity failed")
        _morphism_equal(field, qtiny, AC._cokernel_module(rf.f; cache=:auto)[2]) || error("routing cokernel projection parity failed")
        _pmodule_equal(field, Coimtiny, AC.coimage_with_projection(rf.f; cache=:auto)[1]) || error("routing coimage parity failed")
        _morphism_equal(field, pctiny, AC.coimage_with_projection(rf.f; cache=:auto)[2]) || error("routing coimage projection parity failed")
        _pmodule_equal(field, Qtiny, AC.quotient_with_projection(iImtiny; cache=:auto)[1]) || error("routing quotient parity failed")
        _morphism_equal(field, qqtiny, AC.quotient_with_projection(iImtiny; cache=:auto)[2]) || error("routing quotient projection parity failed")
        _pmodule_equal(field, Epub, AC.equalizer(rf.f, rf.z; cache=:auto)[1]) || error("routing equalizer parity failed")
        _morphism_equal(field, epub, AC.equalizer(rf.f, rf.z; cache=:auto)[2]) || error("routing equalizer inclusion parity failed")
        _pmodule_equal(field, Qpub, AC.coequalizer(rf.f, rf.z; cache=:auto)[1]) || error("routing coequalizer parity failed")
        _morphism_equal(field, qpub, AC.coequalizer(rf.f, rf.z; cache=:auto)[2]) || error("routing coequalizer projection parity failed")
        kernel_sub_pub = AC.kernel_submodule(rf.f; cache=:auto)
        kernel_sub_core = AC.Submodule{CM.coeff_type(field)}(AC._kernel_with_inclusion_cached(rf.f, rf.cc)[2])
        AC._ambient(kernel_sub_pub) === AC._ambient(kernel_sub_core) || error("routing kernel_submodule ambient parity failed")
        AC._sub(kernel_sub_pub).dims == AC._sub(kernel_sub_core).dims || error("routing kernel_submodule dims parity failed")
        image_sub_pub = AC.image_submodule(rf.f; cache=:auto)
        image_sub_core = AC.Submodule{CM.coeff_type(field)}(AC._image_with_inclusion_cached(rf.f, rf.cc)[2])
        AC._ambient(image_sub_pub) === AC._ambient(image_sub_core) || error("routing image_submodule ambient parity failed")
        AC._sub(image_sub_pub).dims == AC._sub(image_sub_core).dims || error("routing image_submodule dims parity failed")
        Itiny = AC._image_module_cached(rf.f, rf.cc)
        _pmodule_equal(field, Itiny, AC.image(rf.f; cache=:auto)) || error("routing image-only parity failed")
        Coonly = AC._coimage_only_cached(rf.f, rf.cc)
        _pmodule_equal(field, Coonly, AC.coimage(rf.f; cache=:auto)) || error("routing coimage-only parity failed")
        q_sub_pub = AC.quotient(rf.im_sub; cache=:auto)
        q_sub_core = AC._quotient_only_cached(AC._inclusion(rf.im_sub), rf.cc)
        _pmodule_equal(field, q_sub_pub, q_sub_core) || error("routing quotient(Submodule) parity failed")
        qwp_sub_pub = AC.quotient_with_projection(rf.im_sub; cache=:auto)
        qwp_sub_core = AC._quotient_with_projection_cached(AC._inclusion(rf.im_sub), rf.cc)
        _pmodule_equal(field, qwp_sub_pub[1], qwp_sub_core[1]) || error("routing quotient_with_projection(Submodule) parity failed")
        _morphism_equal(field, qwp_sub_pub[2], qwp_sub_core[2]) || error("routing quotient_with_projection(Submodule) projection parity failed")
        ses_ctor = AC.ShortExactSequence(rf.it, rf.pt; check=false, cache=rf.cc)
        ses_public = AC.short_exact_sequence(rf.it, rf.pt; check=false, cache=rf.cc)
        (ses_ctor.A === ses_public.A && ses_ctor.B === ses_public.B && ses_ctor.C === ses_public.C) || error("routing short_exact_sequence public parity failed")
        bip_pub = AC.biproduct(rf.A, rf.B)
        ds_core = MD.direct_sum_with_maps(rf.A, rf.B)
        bip_pub[1].dims == ds_core[1].dims || error("routing biproduct parity failed")
        prod_pub = AC.product(rf.A, rf.B)
        cop_pub = AC.coproduct(rf.A, rf.B)
        prod_pub[1].dims == ds_core[1].dims || error("routing product parity failed")
        cop_pub[1].dims == ds_core[1].dims || error("routing coproduct parity failed")
        lim_disc = AC.limit(rf.discrete; cache=rf.cc)
        col_disc = AC.colimit(rf.discrete; cache=rf.cc)
        lim_disc[1].dims == prod_pub[1].dims || error("routing discrete limit parity failed")
        col_disc[1].dims == cop_pub[1].dims || error("routing discrete colimit parity failed")
        lim_par = AC.limit(rf.parallel; cache=rf.cc)
        col_par = AC.colimit(rf.parallel; cache=rf.cc)
        _pmodule_equal(field, lim_par[1], Ecore) || error("routing parallel limit parity failed")
        _pmodule_equal(field, col_par[1], Qcore) || error("routing parallel colimit parity failed")
        lim_cos = AC.limit(rf.cospan; cache=rf.cc)
        col_span = AC.colimit(rf.span; cache=rf.cc)
        _pmodule_equal(field, lim_cos[1], AC.pullback(rf.f, rf.z; cache=rf.cc)[1]) || error("routing cospan limit parity failed")
        _pmodule_equal(field, col_span[1], AC.pushout(rf.f, rf.z; cache=rf.cc)[1]) || error("routing span colimit parity failed")

        println("== strict warm-cached routing ==")
        broute_k_pub = _bench_per_call("kernel_with_inclusion public", () -> AC.kernel_with_inclusion(rf.f; cache=:auto); reps=reps, inner=route_inner)
        _append_row!(rows, fname, "kernel_with_inclusion_routing", "public_cache", broute_k_pub)
        broute_k_core = _bench_per_call("kernel_with_inclusion core", () -> AC._kernel_with_inclusion_cached(rf.f, rf.cc); reps=reps, inner=route_inner)
        _append_row!(rows, fname, "kernel_with_inclusion_routing", "core_cached", broute_k_core)

        broute_i_pub = _bench_per_call("image_with_inclusion public", () -> AC.image_with_inclusion(rf.f; cache=:auto); reps=reps, inner=route_inner)
        _append_row!(rows, fname, "image_with_inclusion_routing", "public_cache", broute_i_pub)
        broute_i_core = _bench_per_call("image_with_inclusion core", () -> AC._image_with_inclusion_cached(rf.f, rf.cc); reps=reps, inner=route_inner)
        _append_row!(rows, fname, "image_with_inclusion_routing", "core_cached", broute_i_core)

        broute_c_pub = _bench_per_call("_cokernel_module public", () -> AC._cokernel_module(rf.f; cache=:auto); reps=reps, inner=route_inner)
        _append_row!(rows, fname, "_cokernel_module_routing", "public_cache", broute_c_pub)
        broute_c_core = _bench_per_call("_cokernel_module core", () -> AC._cokernel_module_cached(rf.f, rf.cc); reps=reps, inner=route_inner)
        _append_row!(rows, fname, "_cokernel_module_routing", "core_cached", broute_c_core)

        broute_kernel_pub = _bench_per_call("kernel public", () -> AC.kernel(rf.f; cache=:auto); reps=reps, inner=route_inner)
        _append_row!(rows, fname, "kernel_routing", "public_cache", broute_kernel_pub)
        broute_kernel_core = _bench_per_call("kernel core", () -> AC._kernel_module_cached(rf.f, rf.cc); reps=reps, inner=route_inner)
        _append_row!(rows, fname, "kernel_routing", "core_cached", broute_kernel_core)

        broute_image_pub = _bench_per_call("image public", () -> AC.image(rf.f; cache=:auto); reps=reps, inner=route_inner)
        _append_row!(rows, fname, "image_routing", "public_auto", broute_image_pub)
        broute_image_core = _bench_per_call("image core", () -> AC._image_module_cached(rf.f, rf.cc); reps=reps, inner=route_inner)
        _append_row!(rows, fname, "image_routing", "core_cached", broute_image_core)

        broute_cok_pub = _bench_per_call("cokernel public", () -> AC.cokernel(rf.f; cache=:auto); reps=reps, inner=route_inner)
        _append_row!(rows, fname, "cokernel_routing", "public_cache", broute_cok_pub)
        broute_cok_core = _bench_per_call("cokernel core", () -> AC._cokernel_only_cached(rf.f, rf.cc); reps=reps, inner=route_inner)
        _append_row!(rows, fname, "cokernel_routing", "core_cached", broute_cok_core)

        broute_coimage_pub = _bench_per_call("coimage public", () -> AC.coimage(rf.f; cache=:auto); reps=reps, inner=route_inner)
        _append_row!(rows, fname, "coimage_routing", "public_cache", broute_coimage_pub)
        broute_coimage_core = _bench_per_call("coimage core", () -> AC._coimage_only_cached(rf.f, rf.cc); reps=reps, inner=route_inner)
        _append_row!(rows, fname, "coimage_routing", "core_cached", broute_coimage_core)

        broute_coim_pub = _bench_per_call("coimage_with_projection public", () -> AC.coimage_with_projection(rf.f; cache=:auto); reps=reps, inner=route_inner)
        _append_row!(rows, fname, "coimage_with_projection_routing", "public_cache", broute_coim_pub)
        broute_coim_core = _bench_per_call("coimage_with_projection core", () -> AC._coimage_with_projection_cached(rf.f, rf.cc); reps=reps, inner=route_inner)
        _append_row!(rows, fname, "coimage_with_projection_routing", "core_cached", broute_coim_core)

        broute_eq_pub = _bench_per_call("equalizer public", () -> AC.equalizer(rf.f, rf.z; cache=:auto); reps=reps, inner=route_inner)
        _append_row!(rows, fname, "equalizer_routing", "public_cache", broute_eq_pub)
        broute_eq_core = _bench_per_call("equalizer core", () -> AC._equalizer_cached(rf.f, rf.z, rf.cc); reps=reps, inner=route_inner)
        _append_row!(rows, fname, "equalizer_routing", "core_cached", broute_eq_core)

        broute_coeq_pub = _bench_per_call("coequalizer public", () -> AC.coequalizer(rf.f, rf.z; cache=:auto); reps=reps, inner=route_inner)
        _append_row!(rows, fname, "coequalizer_routing", "public_cache", broute_coeq_pub)
        broute_coeq_core = _bench_per_call("coequalizer core", () -> AC._coequalizer_cached(rf.f, rf.z, rf.cc); reps=reps, inner=route_inner)
        _append_row!(rows, fname, "coequalizer_routing", "core_cached", broute_coeq_core)

        broute_sn_seq = _bench_per_call("snake_lemma sequence", () -> AC.snake_lemma(rf.top, rf.bottom, rf.alpha, rf.beta, rf.gamma; check=false, cache=rf.cc); reps=reps, inner=max(1, cld(route_inner, 8)))
        _append_row!(rows, fname, "snake_lemma_routing", "sequence_cache", broute_sn_seq)

        broute_ksub_pub = _bench_per_call("kernel_submodule public", () -> AC.kernel_submodule(rf.f; cache=:auto); reps=reps, inner=route_inner)
        _append_row!(rows, fname, "kernel_submodule_routing", "public_cache", broute_ksub_pub)
        broute_ksub_core = _bench_per_call("kernel_submodule core", () -> AC.Submodule{CM.coeff_type(field)}(AC._kernel_with_inclusion_cached(rf.f, rf.cc)[2]); reps=reps, inner=route_inner)
        _append_row!(rows, fname, "kernel_submodule_routing", "core_cached", broute_ksub_core)

        broute_isub_pub = _bench_per_call("image_submodule public", () -> AC.image_submodule(rf.f; cache=:auto); reps=reps, inner=route_inner)
        _append_row!(rows, fname, "image_submodule_routing", "public_auto", broute_isub_pub)
        broute_isub_core = _bench_per_call("image_submodule core", () -> AC.Submodule{CM.coeff_type(field)}(AC._image_with_inclusion_cached(rf.f, rf.cc)[2]); reps=reps, inner=route_inner)
        _append_row!(rows, fname, "image_submodule_routing", "core_cached", broute_isub_core)

        broute_qsub_pub = _bench_per_call("quotient(Submodule) public", () -> AC.quotient(rf.im_sub; cache=:auto); reps=reps, inner=route_inner)
        _append_row!(rows, fname, "quotient_submodule_routing", "public_cache", broute_qsub_pub)
        broute_qsub_core = _bench_per_call("quotient(Submodule) core", () -> AC._quotient_only_cached(AC._inclusion(rf.im_sub), rf.cc); reps=reps, inner=route_inner)
        _append_row!(rows, fname, "quotient_submodule_routing", "core_cached", broute_qsub_core)

        broute_qwps_pub = _bench_per_call("quotient_with_projection(Submodule) public", () -> AC.quotient_with_projection(rf.im_sub; cache=:auto); reps=reps, inner=route_inner)
        _append_row!(rows, fname, "quotient_with_projection_submodule_routing", "public_cache", broute_qwps_pub)
        broute_qwps_core = _bench_per_call("quotient_with_projection(Submodule) core", () -> AC._quotient_with_projection_cached(AC._inclusion(rf.im_sub), rf.cc); reps=reps, inner=route_inner)
        _append_row!(rows, fname, "quotient_with_projection_submodule_routing", "core_cached", broute_qwps_core)

        broute_ses_public = _bench_per_call("short_exact_sequence public", () -> AC.short_exact_sequence(rf.it, rf.pt; check=false, cache=rf.cc); reps=reps, inner=route_inner)
        _append_row!(rows, fname, "short_exact_sequence_routing", "public_cache", broute_ses_public)
        broute_ses_ctor = _bench_per_call("ShortExactSequence ctor", () -> AC.ShortExactSequence(rf.it, rf.pt; check=false, cache=rf.cc); reps=reps, inner=route_inner)
        _append_row!(rows, fname, "short_exact_sequence_routing", "ctor_cache", broute_ses_ctor)

        diagram_inner = max(1, cld(route_inner, 8))
        broute_bip_pub = _bench_per_call("biproduct public", () -> AC.biproduct(rf.A, rf.B); reps=reps, inner=diagram_inner)
        _append_row!(rows, fname, "biproduct_routing", "public", broute_bip_pub)
        broute_bip_core = _bench_per_call("biproduct core", () -> MD.direct_sum_with_maps(rf.A, rf.B); reps=reps, inner=diagram_inner)
        _append_row!(rows, fname, "biproduct_routing", "core", broute_bip_core)
        broute_prod_pub = _bench_per_call("product public", () -> AC.product(rf.A, rf.B); reps=reps, inner=diagram_inner)
        _append_row!(rows, fname, "product_routing", "public", broute_prod_pub)
        broute_prod_core = _bench_per_call("product core", () -> begin
            _, _, _, pA, pB = MD.direct_sum_with_maps(rf.A, rf.B)
            (pA, pB)
        end; reps=reps, inner=diagram_inner)
        _append_row!(rows, fname, "product_routing", "core", broute_prod_core)
        broute_cop_pub = _bench_per_call("coproduct public", () -> AC.coproduct(rf.A, rf.B); reps=reps, inner=diagram_inner)
        _append_row!(rows, fname, "coproduct_routing", "public", broute_cop_pub)
        broute_cop_core = _bench_per_call("coproduct core", () -> begin
            _, iA, iB, _, _ = MD.direct_sum_with_maps(rf.A, rf.B)
            (iA, iB)
        end; reps=reps, inner=diagram_inner)
        _append_row!(rows, fname, "coproduct_routing", "core", broute_cop_core)
        broute_lim_disc = _bench_per_call("limit discrete pair", () -> AC.limit(rf.discrete; cache=:auto); reps=reps, inner=diagram_inner)
        _append_row!(rows, fname, "limit_discrete_routing", "public_cache", broute_lim_disc)
        broute_lim_disc_core = _bench_per_call("product direct", () -> AC.product(rf.A, rf.B); reps=reps, inner=diagram_inner)
        _append_row!(rows, fname, "limit_discrete_routing", "direct", broute_lim_disc_core)
        broute_col_disc = _bench_per_call("colimit discrete pair", () -> AC.colimit(rf.discrete; cache=:auto); reps=reps, inner=diagram_inner)
        _append_row!(rows, fname, "colimit_discrete_routing", "public_cache", broute_col_disc)
        broute_col_disc_core = _bench_per_call("coproduct direct", () -> AC.coproduct(rf.A, rf.B); reps=reps, inner=diagram_inner)
        _append_row!(rows, fname, "colimit_discrete_routing", "direct", broute_col_disc_core)
        broute_lim_par = _bench_per_call("limit parallel pair", () -> AC.limit(rf.parallel; cache=:auto); reps=reps, inner=diagram_inner)
        _append_row!(rows, fname, "limit_parallel_routing", "public_cache", broute_lim_par)
        broute_eq_dir = _bench_per_call("equalizer direct", () -> AC.equalizer(rf.f, rf.z; cache=rf.cc); reps=reps, inner=diagram_inner)
        _append_row!(rows, fname, "limit_parallel_routing", "direct_cache", broute_eq_dir)
        broute_col_par = _bench_per_call("colimit parallel pair", () -> AC.colimit(rf.parallel; cache=:auto); reps=reps, inner=diagram_inner)
        _append_row!(rows, fname, "colimit_parallel_routing", "public_cache", broute_col_par)
        broute_coeq_dir = _bench_per_call("coequalizer direct", () -> AC.coequalizer(rf.f, rf.z; cache=rf.cc); reps=reps, inner=diagram_inner)
        _append_row!(rows, fname, "colimit_parallel_routing", "direct_cache", broute_coeq_dir)
        broute_lim_cos = _bench_per_call("limit cospan", () -> AC.limit(rf.cospan; cache=:auto); reps=reps, inner=max(1, cld(route_inner, 16)))
        _append_row!(rows, fname, "limit_cospan_routing", "public_cache", broute_lim_cos)
        broute_pb_dir = _bench_per_call("pullback direct", () -> AC.pullback(rf.f, rf.z; cache=rf.cc); reps=reps, inner=max(1, cld(route_inner, 16)))
        _append_row!(rows, fname, "limit_cospan_routing", "direct_cache", broute_pb_dir)
        broute_col_span = _bench_per_call("colimit span", () -> AC.colimit(rf.span; cache=:auto); reps=reps, inner=max(1, cld(route_inner, 16)))
        _append_row!(rows, fname, "colimit_span_routing", "public_cache", broute_col_span)
        broute_po_dir = _bench_per_call("pushout direct", () -> AC.pushout(rf.f, rf.z; cache=rf.cc); reps=reps, inner=max(1, cld(route_inner, 16)))
        _append_row!(rows, fname, "colimit_span_routing", "direct_cache", broute_po_dir)

        println()
    end

    _write_rows(out, rows)
    println("Wrote ", out)
    return nothing
end

main()
