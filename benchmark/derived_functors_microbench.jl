#!/usr/bin/env julia
#
# derived_functors_microbench.jl
#
# Purpose
# - Benchmark the main performance surfaces in `DerivedFunctors.jl`.
# - Separate direct resolution builders, end-to-end Ext/Tor calls, lower-level
#   algebra kernels with prebuilt resolutions, and Zn/PL wrapper paths.
#
# Coverage
# - `projective_resolution` / `injective_resolution` on P-modules and fringe modules
# - `Ext` / `Tor` in their common public models, with cold and warm-cache probes
# - `Ext(res, N)` / `ExtInjective(M, resN)` / `Tor(...; res=...)`
# - `ExtDoubleComplex` / `TorDoubleComplex`
# - `ExtZn`, `projective_resolution_Zn`, `injective_resolution_Zn`
# - `ExtRn`, `projective_resolution_Rn`, `injective_resolution_Rn` when the
#   polyhedral backend is available
#
# Timing policy
# - Warm-process microbenchmarking (`@timed` median over reps)
# - Cold vs warm behavior is encoded explicitly in probe names
#
# Usage
#   julia --project=. benchmark/derived_functors_microbench.jl
#   julia --project=. benchmark/derived_functors_microbench.jl --reps=5 --section=derived
#   julia --project=. benchmark/derived_functors_microbench.jl --field=f3 --nx=5 --ny=4 --maxlen=3
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
const FF = TamerOp.FiniteFringe
const MD = TamerOp.Modules
const IR = TamerOp.IndicatorResolutions
const DF = TamerOp.DerivedFunctors
const CH = TamerOp.ChainComplexes
const FZ = TamerOp.FlangeZn
const PLP = TamerOp.PLPolyhedra

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

function _parse_toggle_arg(args, key::String, default::Symbol)
    for a in args
        startswith(a, key * "=") || continue
        v = lowercase(strip(split(a, "=", limit=2)[2]))
        v in ("auto", "default") && return :auto
        v in ("on", "true", "1") && return :on
        v in ("off", "false", "0") && return :off
        error("invalid value '$v' for $key (expected auto|on|off)")
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

function _section_enabled(section::String, group::String)
    section == "all" && return true
    return section == group
end

function _bench(name::AbstractString, f::Function; reps::Int=7, setup::Union{Nothing,Function}=nothing)
    GC.gc()
    setup === nothing || setup()
    f()
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
    mid = cld(reps, 2)
    row = (probe=String(name), median_ms=times_ms[mid], median_kib=bytes[mid] / 1024.0)
    println(rpad(row.probe, 52),
            " median_time=", round(row.median_ms, digits=3), " ms",
            "  median_alloc=", round(row.median_kib, digits=1), " KiB")
    return row
end

function _write_csv(path::AbstractString, rows)
    open(path, "w") do io
        println(io, "probe,median_ms,median_kib")
        for r in rows
            println(io, string(r.probe, ",", r.median_ms, ",", r.median_kib))
        end
    end
end

function _bench_maybe!(rows, name::AbstractString, f::Function; reps::Int)
    try
        push!(rows, _bench(name, f; reps=reps))
    catch err
        println("Skipping ", name, ": ", sprint(showerror, err))
    end
    return rows
end

function _field_from_name(name::String)
    name == "qq" && return CM.QQField()
    name == "f2" && return CM.F2()
    name == "f3" && return CM.F3()
    name == "f5" && return CM.Fp(5)
    error("unknown field '$name' (supported: qq, f2, f3, f5)")
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

function _chain_poset(n::Int)
    rel = falses(n, n)
    @inbounds for i in 1:n, j in i:n
        rel[i, j] = true
    end
    return FF.FinitePoset(rel; check=false)
end

function _rand_coeff(rng::AbstractRNG, field::CM.AbstractCoeffField)
    v = rand(rng, -3:3)
    v == 0 && (v = 1)
    return CM.coerce(field, v)
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
        FF.leq(Q, up_verts[i], down_verts[j]) || continue
        rand(rng) < density || continue
        phi[j, i] = _rand_coeff(rng, field)
    end
    return FF.FringeModule{K}(Q, U, D, phi; field=field)
end

function _digest_projective_resolution(R)
    acc = length(R.Pmods) + length(R.d_mor)
    @inbounds for gens in R.gens
        acc += length(gens)
    end
    @inbounds for A in R.d_mat
        acc += size(A, 1) + size(A, 2) + nnz(A)
    end
    return acc
end

function _digest_injective_resolution(R)
    acc = length(R.Emods) + length(R.d_mor)
    @inbounds for gens in R.gens
        acc += length(gens)
    end
    return acc
end

function _digest_ext(E, maxdeg::Int)
    acc = 0
    @inbounds for t in 0:maxdeg
        acc += DF.dim(E, t)
    end
    return acc
end

function _digest_comparison(pair)
    P2I, I2P = pair
    acc = 0
    @inbounds for A in P2I
        acc += size(A, 1) + size(A, 2)
    end
    @inbounds for A in I2P
        acc += size(A, 1) + size(A, 2)
    end
    return acc
end

function _digest_tor(T, maxdeg::Int)
    acc = 0
    @inbounds for s in 0:maxdeg
        acc += DF.dim(T, s)
    end
    return acc
end

function _digest_doublecomplex(DC)
    acc = sum(DC.dims)
    @inbounds for A in DC.dv
        acc += nnz(A)
    end
    @inbounds for A in DC.dh
        acc += nnz(A)
    end
    return acc
end

function _digest_cochain_complex(C)
    acc = sum(C.dims)
    @inbounds for A in C.d
        acc += nnz(A)
    end
    return acc
end

function _core_fixture(field::CM.AbstractCoeffField; nx::Int=4, ny::Int=4, maxlen::Int=2, density::Float64=0.35)
    P = _grid_finite_poset(nx, ny)
    Pop = FF.FinitePoset(transpose(FF.leq_matrix(P)); check=false)
    FF.build_cache!(P; cover=true, updown=true)
    FF.build_cache!(Pop; cover=true, updown=true)
    K = CM.coeff_type(field)

    n = FF.nvertices(P)
    nups = max(12, min(24, 2n))
    ndowns = nups

    HM = _random_fringe(P, field; nups=nups, ndowns=ndowns, density=density, seed=Int(0xD1A1))
    HN = _random_fringe(P, field; nups=nups, ndowns=ndowns, density=density, seed=Int(0xD1A2))
    HRop = _random_fringe(Pop, field; nups=nups, ndowns=ndowns, density=density, seed=Int(0xD1A3))
    HL = _random_fringe(P, field; nups=nups, ndowns=ndowns, density=density, seed=Int(0xD1A4))

    M = IR.pmodule_from_fringe(HM)
    N = IR.pmodule_from_fringe(HN)
    Rop = IR.pmodule_from_fringe(HRop)
    L = IR.pmodule_from_fringe(HL)

    sparse_dims = [((i % 3) == 0 || i == n) ? 1 : 0 for i in 1:n]
    sparse_edges = Dict{Tuple{Int,Int},Matrix{K}}()
    for (u, v) in FF.cover_edges(P)
        sparse_edges[(u, v)] = CM.zeros(field, sparse_dims[v], sparse_dims[u])
    end
    Nsparse = MD.PModule{K}(P, sparse_dims, sparse_edges; field=field)

    resopt = OPT.ResolutionOptions(maxlen=maxlen, minimal=false, check=false)
    df_proj = OPT.DerivedFunctorOptions(maxdeg=maxlen, model=:projective)
    df_inj = OPT.DerivedFunctorOptions(maxdeg=maxlen, model=:injective)
    df_uni = OPT.DerivedFunctorOptions(maxdeg=maxlen, model=:unified)
    df_tor_first = OPT.DerivedFunctorOptions(maxdeg=maxlen, model=:first)
    df_tor_second = OPT.DerivedFunctorOptions(maxdeg=maxlen, model=:second)

    resM = DF.projective_resolution(M, resopt; threads=false)
    resNinj = DF.injective_resolution(N, resopt; threads=false)
    resRop = DF.projective_resolution(Rop, resopt; threads=false)
    resL = DF.projective_resolution(L, resopt; threads=false)

    cmpP = _chain_poset(3)
    cmpHm = FF.one_by_one_fringe(cmpP, FF.principal_upset(cmpP, 1), FF.principal_downset(cmpP, 2), one(K); field=field)
    cmpHn = FF.one_by_one_fringe(cmpP, FF.principal_upset(cmpP, 2), FF.principal_downset(cmpP, 3), one(K); field=field)
    cmpM = IR.pmodule_from_fringe(cmpHm)
    cmpN = IR.pmodule_from_fringe(cmpHn)
    cmp_df_uni = OPT.DerivedFunctorOptions(maxdeg=min(maxlen, 1), model=:unified)

    return (
        field=field,
        P=P,
        Pop=Pop,
        HM=HM,
        HN=HN,
        HRop=HRop,
        HL=HL,
        M=M,
        N=N,
        Nsparse=Nsparse,
        Rop=Rop,
        L=L,
        resopt=resopt,
        df_proj=df_proj,
        df_inj=df_inj,
        df_uni=df_uni,
        df_tor_first=df_tor_first,
        df_tor_second=df_tor_second,
        resM=resM,
        resNinj=resNinj,
        resRop=resRop,
        resL=resL,
        cmpM=cmpM,
        cmpN=cmpN,
        cmp_df_uni=cmp_df_uni,
        maxdeg=maxlen,
    )
end

function _zn_fixture(field::CM.AbstractCoeffField, maxlen::Int)
    K = CM.coeff_type(field)
    c(x) = CM.coerce(field, x)
    tau = FZ.face(1, [])

    F1 = FZ.IndFlat(tau, [0]; id=:F1)
    E1 = FZ.IndInj(tau, [5]; id=:E1)
    FG1 = FZ.Flange{K}(1, [F1], [E1], reshape(K[c(1)], 1, 1))

    F2 = FZ.IndFlat(tau, [2]; id=:F2)
    E2 = FZ.IndInj(tau, [7]; id=:E2)
    FG2 = FZ.Flange{K}(1, [F2], [E2], reshape(K[c(1)], 1, 1))

    enc = OPT.EncodingOptions(backend=:zn, max_regions=50_000, field=field)
    enc_dense = OPT.EncodingOptions(backend=:zn, max_regions=50_000, poset_kind=:dense, field=field)
    df = OPT.DerivedFunctorOptions(maxdeg=maxlen, model=:projective)
    res = OPT.ResolutionOptions(maxlen=maxlen, minimal=false, check=false)
    return (FG1=FG1, FG2=FG2, enc=enc, enc_dense=enc_dense, df=df, res=res)
end

function _pl_fixture(maxlen::Int)
    PLP.HAVE_POLY || return nothing
    field = CM.QQField()
    K = CM.coeff_type(field)
    c(x) = CM.coerce(field, x)

    U1_hp = PLP.make_hpoly(K[c(-1) c(0); c(0) c(-1)], K[c(0), c(0)])
    D1_hp = PLP.make_hpoly(K[c(1) c(0); c(0) c(1)], K[c(1), c(1)])
    U1 = PLP.PLUpset(PLP.PolyUnion(2, [U1_hp]))
    D1 = PLP.PLDownset(PLP.PolyUnion(2, [D1_hp]))
    F1 = PLP.PLFringe([U1], [D1], reshape(K[c(1)], 1, 1))

    U2_hp = PLP.make_hpoly(K[c(-1) c(0); c(0) c(-1)], K[c(-1), c(-1)])
    D2_hp = PLP.make_hpoly(K[c(1) c(0); c(0) c(1)], K[c(2), c(2)])
    U2 = PLP.PLUpset(PLP.PolyUnion(2, [U2_hp]))
    D2 = PLP.PLDownset(PLP.PolyUnion(2, [D2_hp]))
    F2 = PLP.PLFringe([U2], [D2], reshape(K[c(1)], 1, 1))

    enc = OPT.EncodingOptions(
        backend=:pl,
        max_regions=50_000,
        strict_eps=PLP.STRICT_EPS_QQ,
        field=field,
    )
    enc_dense = OPT.EncodingOptions(
        backend=:pl,
        max_regions=50_000,
        strict_eps=PLP.STRICT_EPS_QQ,
        poset_kind=:dense,
        field=field,
    )
    df = OPT.DerivedFunctorOptions(maxdeg=maxlen, model=:projective)
    res = OPT.ResolutionOptions(maxlen=maxlen, minimal=false, check=false)
    return (F1=F1, F2=F2, enc=enc, enc_dense=enc_dense, df=df, res=res)
end

function _push_resolution_rows!(rows, fx; reps::Int)
    proj_cache = Ref{CM.ResolutionCache}(CM.ResolutionCache())
    inj_cache = Ref{CM.ResolutionCache}(CM.ResolutionCache())

    push!(rows, _bench("res.projective pmodule cold", () -> begin
        cache = CM.ResolutionCache()
        _digest_projective_resolution(DF.projective_resolution(fx.M, fx.resopt; threads=false, cache=cache))
    end; reps=reps))

    push!(rows, _bench("res.projective pmodule warm_cache", () -> begin
        _digest_projective_resolution(DF.projective_resolution(fx.M, fx.resopt; threads=false, cache=proj_cache[]))
    end; reps=reps, setup=() -> begin
        proj_cache[] = CM.ResolutionCache()
        DF.projective_resolution(fx.M, fx.resopt; threads=false, cache=proj_cache[])
        nothing
    end))

    push!(rows, _bench("res.projective fringe cold", () -> begin
        cache = CM.ResolutionCache()
        _digest_projective_resolution(DF.projective_resolution(fx.HM, fx.resopt; threads=false, cache=cache))
    end; reps=reps))

    push!(rows, _bench("res.injective pmodule cold", () -> begin
        cache = CM.ResolutionCache()
        _digest_injective_resolution(DF.injective_resolution(fx.N, fx.resopt; threads=false, cache=cache))
    end; reps=reps))

    push!(rows, _bench("res.injective sparse_support pmodule cold", () -> begin
        cache = CM.ResolutionCache()
        _digest_injective_resolution(DF.injective_resolution(fx.Nsparse, fx.resopt; threads=false, cache=cache))
    end; reps=reps))

    push!(rows, _bench("res.injective pmodule warm_cache", () -> begin
        _digest_injective_resolution(DF.injective_resolution(fx.N, fx.resopt; threads=false, cache=inj_cache[]))
    end; reps=reps, setup=() -> begin
        inj_cache[] = CM.ResolutionCache()
        DF.injective_resolution(fx.N, fx.resopt; threads=false, cache=inj_cache[])
        nothing
    end))

    push!(rows, _bench("res.injective fringe cold", () -> begin
        cache = CM.ResolutionCache()
        _digest_injective_resolution(DF.injective_resolution(fx.HN, fx.resopt; threads=false, cache=cache))
    end; reps=reps))
    return rows
end

function _push_derived_rows!(rows, fx; reps::Int)
    ext_cache = Ref{CM.ResolutionCache}(CM.ResolutionCache())
    ext_inj_cache = Ref{CM.ResolutionCache}(CM.ResolutionCache())
    ext_uni_cache = Ref{CM.ResolutionCache}(CM.ResolutionCache())
    ext_uni_compare_cache = Ref{CM.ResolutionCache}(CM.ResolutionCache())
    tor_first_cache = Ref{CM.ResolutionCache}(CM.ResolutionCache())
    tor_second_cache = Ref{CM.ResolutionCache}(CM.ResolutionCache())

    push!(rows, _bench("derived.Ext projective cold", () -> begin
        cache = CM.ResolutionCache()
        _digest_ext(DF.Ext(fx.M, fx.N, fx.df_proj; cache=cache), fx.maxdeg)
    end; reps=reps))

    push!(rows, _bench("derived.Ext projective one_shot", () -> begin
        _digest_ext(DF.Ext(fx.M, fx.N, fx.df_proj; cache=nothing), fx.maxdeg)
    end; reps=reps))

    push!(rows, _bench("derived.Ext projective warm_cache", () -> begin
        _digest_ext(DF.Ext(fx.M, fx.N, fx.df_proj; cache=ext_cache[]), fx.maxdeg)
    end; reps=reps, setup=() -> begin
        ext_cache[] = CM.ResolutionCache()
        DF.Ext(fx.M, fx.N, fx.df_proj; cache=ext_cache[])
        nothing
    end))

    push!(rows, _bench("derived.Ext injective cold", () -> begin
        cache = CM.ResolutionCache()
        _digest_ext(DF.Ext(fx.M, fx.N, fx.df_inj; cache=cache), fx.maxdeg)
    end; reps=reps))

    push!(rows, _bench("derived.Ext injective one_shot", () -> begin
        _digest_ext(DF.Ext(fx.M, fx.N, fx.df_inj; cache=nothing), fx.maxdeg)
    end; reps=reps))

    push!(rows, _bench("derived.Ext injective warm_cache", () -> begin
        _digest_ext(DF.Ext(fx.M, fx.N, fx.df_inj; cache=ext_inj_cache[]), fx.maxdeg)
    end; reps=reps, setup=() -> begin
        ext_inj_cache[] = CM.ResolutionCache()
        DF.Ext(fx.M, fx.N, fx.df_inj; cache=ext_inj_cache[])
        nothing
    end))

    push!(rows, _bench("derived.Ext unified cold", () -> begin
        cache = CM.ResolutionCache()
        _digest_ext(DF.Ext(fx.M, fx.N, fx.df_uni; cache=cache), fx.maxdeg)
    end; reps=reps))

    push!(rows, _bench("derived.Ext unified warm_cache", () -> begin
        _digest_ext(DF.Ext(fx.M, fx.N, fx.df_uni; cache=ext_uni_cache[]), fx.maxdeg)
    end; reps=reps, setup=() -> begin
        ext_uni_cache[] = CM.ResolutionCache()
        DF.Ext(fx.M, fx.N, fx.df_uni; cache=ext_uni_cache[])
        nothing
    end))

    push!(rows, _bench("derived.Ext unified injective_model cold", () -> begin
        cache = CM.ResolutionCache()
        E = DF.Ext(fx.M, fx.N, fx.df_uni; cache=cache)
        _digest_ext(DF.injective_model(E), fx.maxdeg)
    end; reps=reps))

    push!(rows, _bench("derived.Ext unified comparison cold (chain fixture)", () -> begin
        cache = CM.ResolutionCache()
        E = DF.Ext(fx.cmpM, fx.cmpN, fx.cmp_df_uni; cache=cache)
        _digest_comparison(DF.comparison_isomorphisms(E))
    end; reps=reps))

    push!(rows, _bench("derived.Ext unified comparison warm_cache (chain fixture)", () -> begin
        E = DF.Ext(fx.cmpM, fx.cmpN, fx.cmp_df_uni; cache=ext_uni_compare_cache[])
        _digest_comparison(DF.comparison_isomorphisms(E))
    end; reps=reps, setup=() -> begin
        ext_uni_compare_cache[] = CM.ResolutionCache()
        DF.Ext(fx.cmpM, fx.cmpN, fx.cmp_df_uni; cache=ext_uni_compare_cache[])
        nothing
    end))

    push!(rows, _bench("derived.Ext fringe cold", () -> begin
        cache = CM.ResolutionCache()
        _digest_ext(DF.Ext(fx.HM, fx.HN, fx.df_proj; cache=cache), fx.maxdeg)
    end; reps=reps))

    push!(rows, _bench("derived.Tor first cold", () -> begin
        cache = CM.ResolutionCache()
        _digest_tor(DF.Tor(fx.Rop, fx.L, fx.df_tor_first; cache=cache), fx.maxdeg)
    end; reps=reps))

    push!(rows, _bench("derived.Tor first warm_cache", () -> begin
        _digest_tor(DF.Tor(fx.Rop, fx.L, fx.df_tor_first; cache=tor_first_cache[]), fx.maxdeg)
    end; reps=reps, setup=() -> begin
        tor_first_cache[] = CM.ResolutionCache()
        DF.Tor(fx.Rop, fx.L, fx.df_tor_first; cache=tor_first_cache[])
        nothing
    end))

    push!(rows, _bench("derived.Tor second cold", () -> begin
        cache = CM.ResolutionCache()
        _digest_tor(DF.Tor(fx.Rop, fx.L, fx.df_tor_second; cache=cache), fx.maxdeg)
    end; reps=reps))

    push!(rows, _bench("derived.Tor second warm_cache", () -> begin
        _digest_tor(DF.Tor(fx.Rop, fx.L, fx.df_tor_second; cache=tor_second_cache[]), fx.maxdeg)
    end; reps=reps, setup=() -> begin
        tor_second_cache[] = CM.ResolutionCache()
        DF.Tor(fx.Rop, fx.L, fx.df_tor_second; cache=tor_second_cache[])
        nothing
    end))

    push!(rows, _bench("derived.indicator_extdims fringe", () -> begin
        dims = DF.ext_dimensions_via_indicator_resolutions(fx.HM, fx.HN; maxlen=fx.maxdeg, verify=false)
        sum(values(dims))
    end; reps=reps))
    return rows
end

function _push_bicomplex_rows!(rows, fx; reps::Int)
    extdc_cache = Ref{CM.ResolutionCache}(CM.ResolutionCache())
    tordc_cache = Ref{CM.ResolutionCache}(CM.ResolutionCache())

    push!(rows, _bench("bicomplex.Ext from projective_resolution", () -> begin
        _digest_ext(DF.Ext(fx.resM, fx.N; threads=false), fx.maxdeg)
    end; reps=reps))

    proj_ext_cochain = Ref{Any}(nothing)
    push!(rows, _bench("bicomplex.Ext(res,N) hom_differential_build", () -> begin
        C, offs = DF.ExtTorSpaces._projective_ext_cochain_complex(fx.resM, fx.N; threads=false)
        _digest_cochain_complex(C) + sum(length, offs)
    end; reps=reps))

    push!(rows, _bench("bicomplex.Ext(res,N) cohomology_extraction", () -> begin
        H = CH.cohomology_data(proj_ext_cochain[]::CH.CochainComplex)
        sum(Ht.dimH for Ht in H)
    end; reps=reps, setup=() -> begin
        proj_ext_cochain[] = first(DF.ExtTorSpaces._projective_ext_cochain_complex(fx.resM, fx.N; threads=false))
        nothing
    end))

    push!(rows, _bench("bicomplex.Ext injective kernel", () -> begin
        _digest_ext(DF.ExtInjective(fx.M, fx.resNinj; threads=false), fx.maxdeg)
    end; reps=reps))

    push!(rows, _bench("bicomplex.Tor first with_res", () -> begin
        _digest_tor(DF.Tor(fx.Rop, fx.L, fx.df_tor_first; res=fx.resRop), fx.maxdeg)
    end; reps=reps))

    push!(rows, _bench("bicomplex.Tor second with_res", () -> begin
        _digest_tor(DF.Tor(fx.Rop, fx.L, fx.df_tor_second; res=fx.resL), fx.maxdeg)
    end; reps=reps))

    push!(rows, _bench("bicomplex.ExtDoubleComplex", () -> begin
        _digest_doublecomplex(DF.ExtDoubleComplex(fx.M, fx.N; maxlen=fx.maxdeg, threads=false))
    end; reps=reps))

    push!(rows, _bench("bicomplex.ExtDoubleComplex warm_cache", () -> begin
        _digest_doublecomplex(DF.ExtDoubleComplex(fx.M, fx.N; maxlen=fx.maxdeg, threads=false, cache=extdc_cache[]))
    end; reps=reps, setup=() -> begin
        extdc_cache[] = CM.ResolutionCache()
        DF.ExtDoubleComplex(fx.M, fx.N; maxlen=fx.maxdeg, threads=false, cache=extdc_cache[])
        nothing
    end))

    push!(rows, _bench("bicomplex.TorDoubleComplex", () -> begin
        _digest_doublecomplex(DF.TorDoubleComplex(fx.Rop, fx.L; maxlen=fx.maxdeg, threads=false))
    end; reps=reps))

    push!(rows, _bench("bicomplex.TorDoubleComplex warm_cache", () -> begin
        _digest_doublecomplex(DF.TorDoubleComplex(fx.Rop, fx.L; maxlen=fx.maxdeg, threads=false, cache=tordc_cache[]))
    end; reps=reps, setup=() -> begin
        tordc_cache[] = CM.ResolutionCache()
        DF.TorDoubleComplex(fx.Rop, fx.L; maxlen=fx.maxdeg, threads=false, cache=tordc_cache[])
        nothing
    end))
    return rows
end

function _push_wrapper_rows!(rows, zn_fx, pl_fx; reps::Int)
    push!(rows, _bench("wrappers.ExtZn", () -> begin
        _digest_ext(DF.ExtZn(zn_fx.FG1, zn_fx.FG2, zn_fx.enc, zn_fx.df), zn_fx.df.maxdeg)
    end; reps=reps))

    push!(rows, _bench("wrappers.projective_resolution_Zn", () -> begin
        _digest_projective_resolution(DF.projective_resolution_Zn(zn_fx.FG1, zn_fx.enc, zn_fx.res; threads=false))
    end; reps=reps))

    push!(rows, _bench("wrappers.injective_resolution_Zn", () -> begin
        _digest_injective_resolution(DF.injective_resolution_Zn(zn_fx.FG1, zn_fx.enc, zn_fx.res; threads=false))
    end; reps=reps))

    _bench_maybe!(rows, "wrappers.ExtDoubleComplex_Zn", () -> begin
        _digest_doublecomplex(DF.ExtDoubleComplex(zn_fx.FG1, zn_fx.FG2, zn_fx.enc_dense; maxlen=zn_fx.res.maxlen))
    end; reps=reps)

    if pl_fx === nothing
        println("Skipping PL wrapper benchmarks: Polyhedra backend unavailable or field is not qq.")
        return rows
    end

    push!(rows, _bench("wrappers.ExtRn", () -> begin
        _digest_ext(DF.ExtRn(pl_fx.F1, pl_fx.F2, pl_fx.enc, pl_fx.df), pl_fx.df.maxdeg)
    end; reps=reps))

    push!(rows, _bench("wrappers.projective_resolution_Rn", () -> begin
        _digest_projective_resolution(DF.projective_resolution_Rn(pl_fx.F1, pl_fx.enc, pl_fx.res; threads=false))
    end; reps=reps))

    push!(rows, _bench("wrappers.injective_resolution_Rn", () -> begin
        _digest_injective_resolution(DF.injective_resolution_Rn(pl_fx.F1, pl_fx.enc, pl_fx.res; threads=false))
    end; reps=reps))

    _bench_maybe!(rows, "wrappers.ExtDoubleComplex_Rn", () -> begin
        _digest_doublecomplex(DF.ExtDoubleComplex(pl_fx.F1, pl_fx.F2, pl_fx.enc_dense; maxlen=pl_fx.res.maxlen))
    end; reps=reps)
    return rows
end

function main(; reps::Int=5,
              section::String="all",
              field_name::String="qq",
              nx::Int=4,
              ny::Int=4,
              maxlen::Int=2,
              density::Float64=0.35,
              tor_direct_triplets::Symbol=:auto,
              out::String=joinpath(@__DIR__, "_tmp_derived_functors_microbench.csv"))
    field = _field_from_name(field_name)
    rows = NamedTuple[]
    old_tor_direct = DF.ExtTorSpaces._TOR_USE_DIRECT_MAP_TRIPLET_ASSEMBLY[]
    if tor_direct_triplets === :on
        DF.ExtTorSpaces._TOR_USE_DIRECT_MAP_TRIPLET_ASSEMBLY[] = true
    elseif tor_direct_triplets === :off
        DF.ExtTorSpaces._TOR_USE_DIRECT_MAP_TRIPLET_ASSEMBLY[] = false
    end

    println("DerivedFunctors microbenchmark")
    println("timing_policy=warm_process_median")
    println("section=", section,
            " field=", field_name,
            " nx=", nx,
            " ny=", ny,
            " maxlen=", maxlen,
            " density=", density,
            " tor_direct_triplets=", tor_direct_triplets,
            " threads=", Threads.nthreads())

    try
        need_core = _section_enabled(section, "resolutions") ||
                    _section_enabled(section, "derived") ||
                    _section_enabled(section, "bicomplex")
        fx = need_core ? _core_fixture(field; nx=nx, ny=ny, maxlen=maxlen, density=density) : nothing

        if _section_enabled(section, "resolutions")
            _push_resolution_rows!(rows, fx; reps=reps)
        end
        if _section_enabled(section, "derived")
            _push_derived_rows!(rows, fx; reps=reps)
        end
        if _section_enabled(section, "bicomplex")
            _push_bicomplex_rows!(rows, fx; reps=reps)
        end
        if _section_enabled(section, "wrappers")
            zn_fx = _zn_fixture(field, maxlen)
            pl_fx = field_name == "qq" ? _pl_fixture(maxlen) : nothing
            _push_wrapper_rows!(rows, zn_fx, pl_fx; reps=reps)
        end

        _write_csv(out, rows)
        println("wrote ", out)
        return rows
    finally
        DF.ExtTorSpaces._TOR_USE_DIRECT_MAP_TRIPLET_ASSEMBLY[] = old_tor_direct
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    args = copy(ARGS)
    main(
        reps=_parse_int_arg(args, "--reps", 5),
        section=_parse_string_arg(args, "--section", "all"),
        field_name=_parse_string_arg(args, "--field", "qq"),
        nx=_parse_int_arg(args, "--nx", 4),
        ny=_parse_int_arg(args, "--ny", 4),
        maxlen=_parse_int_arg(args, "--maxlen", 2),
        density=_parse_float_arg(args, "--density", 0.35),
        tor_direct_triplets=_parse_toggle_arg(args, "--tor_direct_triplets", :auto),
        out=_parse_path_arg(args, "--out", joinpath(@__DIR__, "_tmp_derived_functors_microbench.csv")),
    )
end
