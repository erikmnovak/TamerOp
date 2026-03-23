#!/usr/bin/env julia
#
# module_complexes_microbench.jl
#
# Purpose
# - Benchmark the main performance surfaces in `src/ModuleComplexes.jl`.
# - Separate basic complex/map construction from RHom/derived-tensor builders,
#   functoriality maps, and hyperExt/hyperTor public wrappers.
#
# Coverage
# - `ModuleCochainComplex`, `ModuleCochainMap`, `mapping_cone`
# - `cohomology_module`, `induced_map_on_cohomology_modules`
# - `RHomComplex` over P-modules and fringe wrappers
# - `rhom_map_first` / `rhom_map_second` on prebuilt RHom complexes and public wrappers
# - `DerivedTensorComplex`
# - `hyperExt`, `hyperTor`, and their induced maps
#
# Timing policy
# - Warm-process microbenchmarking (`@timed` median over reps)
# - Cache-sensitive probes encode cold/warm behavior explicitly in the probe name
#
# Usage
#   julia --project=. benchmark/module_complexes_microbench.jl
#   julia --project=. benchmark/module_complexes_microbench.jl --section=rhom --field=f3
#   julia --project=. benchmark/module_complexes_microbench.jl --profile=larger --reps=5
#

using Random
using SparseArrays
using Pkg

Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)

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
const MC = TamerOp.ModuleComplexes

function _parse_int_arg(args, key::String, default::Int)
    for a in args
        startswith(a, key * "=") || continue
        return max(1, parse(Int, split(a, "=", limit=2)[2]))
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

function _section_enabled(section::String, group::String)
    section == "all" && return true
    return section == group
end

function _profile_defaults(profile::String)
    if profile == "default"
        return (nx=3, ny=3, nups=8, ndowns=8, density=0.30, maxlen=2)
    elseif profile == "larger"
        return (nx=4, ny=4, nups=14, ndowns=14, density=0.35, maxlen=2)
    else
        error("unknown profile '$profile' (supported: default, larger)")
    end
end

function _bench(name::AbstractString, f::Function;
                reps::Int=7,
                setup::Union{Nothing,Function}=nothing)
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
    println(rpad(row.probe, 56),
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

function _field_from_name(name::String)
    name == "qq" && return CM.QQField()
    name == "f2" && return CM.F2()
    name == "f3" && return CM.F3()
    name == "f5" && return CM.Fp(5)
    error("unknown field '$name' (supported: qq, f2, f3, f5)")
end

function _grid_finite_poset(nx::Int, ny::Int)
    n = nx * ny
    rel = falses(n, n)
    @inline idx(ix, iy) = (iy - 1) * nx + ix
    @inbounds for y1 in 1:ny, x1 in 1:nx
        i = idx(x1, y1)
        for y2 in y1:ny, x2 in x1:nx
            rel[i, idx(x2, y2)] = true
        end
    end
    return FF.FinitePoset(rel; check=false)
end

function _rand_coeff(rng::AbstractRNG, field::CM.AbstractCoeffField)
    v = rand(rng, -3:3)
    v == 0 && (v = 1)
    return CM.coerce(field, v)
end

function _random_fringe(Q::FF.AbstractPoset, field::CM.AbstractCoeffField;
                        nups::Int, ndowns::Int, density::Float64, seed::Int)
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

function _scalar_morphism(M::MD.PModule{K}, a::Int) where {K}
    comps = Vector{Matrix{K}}(undef, M.Q.n)
    s = CM.coerce(M.field, a)
    @inbounds for u in 1:M.Q.n
        du = M.dims[u]
        comps[u] = du == 0 ? CM.zeros(M.field, 0, 0) : s * CM.eye(M.field, du)
    end
    return MD.PMorphism(M, M, comps)
end

function _nnzish(A::AbstractMatrix)
    return issparse(A) ? nnz(A) : count(!iszero, A)
end

function _digest_sparse_family(A)
    acc = 0
    @inbounds for M in A
        acc += size(M, 1) + size(M, 2) + nnz(M)
    end
    return acc
end

function _digest_pmodule(M::MD.PModule)
    acc = sum(M.dims)
    for v in 1:M.Q.n
        acc += M.dims[v]
    end
    for (_, A) in M.edge_maps
        acc += size(A, 1) + size(A, 2) + _nnzish(A)
    end
    return acc
end

function _digest_pmorphism(f::MD.PMorphism)
    acc = 0
    @inbounds for A in f.comps
        acc += size(A, 1) + size(A, 2) + _nnzish(A)
    end
    return acc
end

function _digest_module_complex(C::MC.ModuleCochainComplex)
    acc = 0
    @inbounds for M in C.terms
        acc += _digest_pmodule(M)
    end
    @inbounds for d in C.diffs
        acc += _digest_pmorphism(d)
    end
    return acc
end

function _digest_module_map(f::MC.ModuleCochainMap)
    acc = 0
    @inbounds for g in f.comps
        acc += _digest_pmorphism(g)
    end
    return acc
end

function _digest_chain_map(f)
    acc = 0
    @inbounds for A in f.maps
        acc += size(A, 1) + size(A, 2) + nnz(A)
    end
    return acc
end

function _digest_rhom(R::MC.RHomComplex)
    return sum(R.DC.dims) + _digest_sparse_family(R.DC.dv) + _digest_sparse_family(R.DC.dh) +
           sum(R.tot.dims) + _digest_sparse_family(R.tot.d)
end

function _digest_tensor(T::MC.DerivedTensorComplex)
    return sum(T.DC.dims) + _digest_sparse_family(T.DC.dv) + _digest_sparse_family(T.DC.dh) +
           sum(T.tot.dims) + _digest_sparse_family(T.tot.d)
end

function _digest_hyperext(H::MC.HyperExtSpace)
    acc = 0
    @inbounds for cd in H.cohom
        acc += cd.dimC + cd.dimZ + cd.dimB + cd.dimH
    end
    return acc
end

function _digest_hypertor(H::MC.HyperTorSpace)
    acc = 0
    @inbounds for cd in H.cohom
        acc += cd.dimC + cd.dimZ + cd.dimB + cd.dimH
    end
    return acc
end

function _first_nonzero_ext_degree(H::MC.HyperExtSpace)
    for t in MC.degree_range(H)
        MC.dim(H, t) > 0 && return t
    end
    return nothing
end

function _first_nonzero_tor_degree(H::MC.HyperTorSpace)
    for n in MC.degree_range(H)
        MC.dim(H, n) > 0 && return n
    end
    return nothing
end

function _fixture(; field::CM.AbstractCoeffField,
                  nx::Int,
                  ny::Int,
                  nups::Int,
                  ndowns::Int,
                  density::Float64,
                  maxlen::Int)
    K = CM.coeff_type(field)
    P = _grid_finite_poset(nx, ny)
    Pop = FF.FinitePoset(transpose(FF.leq_matrix(P)); check=false)
    FF.build_cache!(P; cover=true, updown=true)
    FF.build_cache!(Pop; cover=true, updown=true)

    Hm = _random_fringe(P, field; nups=nups, ndowns=ndowns, density=density, seed=Int(0xC011))
    Hn = _random_fringe(P, field; nups=nups, ndowns=ndowns, density=density, seed=Int(0xC012))
    Hr = _random_fringe(Pop, field; nups=nups, ndowns=ndowns, density=density, seed=Int(0xC013))
    M = IR.pmodule_from_fringe(Hm)
    N = IR.pmodule_from_fringe(Hn)
    Rop = IR.pmodule_from_fringe(Hr)

    Z = MD.zero_pmodule(M.Q; field=field)
    idM = MD.id_morphism(M)
    zero_MZ = MD.zero_morphism(M, Z)
    idZ = MD.id_morphism(Z)

    C0 = MC.ModuleCochainComplex([M], MD.PMorphism[]; tmin=0, check=true)
    C = MC.ModuleCochainComplex([M, M, Z], [idM, zero_MZ]; tmin=0, check=true)
    f2 = _scalar_morphism(M, 2)
    fmap0 = MC.ModuleCochainMap(C0, C0, [f2]; tmin=0, tmax=0, check=true)
    fmap = MC.ModuleCochainMap(C, C, [f2, f2, idZ]; tmin=0, tmax=2, check=true)
    gN = _scalar_morphism(N, 2)
    gR = _scalar_morphism(Rop, 2)

    resN = DF.injective_resolution(N, OPT.ResolutionOptions(maxlen=maxlen))
    hcache = DF.HomSystemCache{K}()

    R_serial = MC.RHomComplex(C, N; maxlen=maxlen, resN=resN, cache=hcache, threads=false)
    Hext = MC.hyperExt(C0, N; maxlen=maxlen, cache=hcache, threads=false)
    Htor = MC.hyperTor(Rop, C0; maxlen=maxlen, threads=false)
    t_ext = something(_first_nonzero_ext_degree(Hext), first(MC.degree_range(Hext)))
    t_tor = let rng = MC.degree_range(Htor)
        isempty(rng) ? 0 : something(_first_nonzero_tor_degree(Htor), first(rng))
    end

    T_serial = MC.DerivedTensorComplex(Rop, C; maxlen=maxlen, threads=false)
    return (
        field=field,
        Hm=Hm, Hn=Hn, Hr=Hr,
        M=M, N=N, Rop=Rop,
        Z=Z,
        C0=C0,
        C=C,
        fmap0=fmap0,
        fmap=fmap,
        gN=gN,
        gR=gR,
        maxlen=maxlen,
        resN=resN,
        hcache=hcache,
        R_serial=R_serial,
        Hext=Hext,
        Htor=Htor,
        t_ext=t_ext,
        t_tor=t_tor,
        T_serial=T_serial,
    )
end

function main(; reps::Int=5,
              field_name::String="qq",
              profile::String="default",
              section::String="all",
              out::String=joinpath(@__DIR__, "_tmp_module_complexes_microbench.csv"))
    cfg = _profile_defaults(profile)
    field = _field_from_name(field_name)
    fx = _fixture(; field=field, cfg...)
    rows = NamedTuple[]

    println("ModuleComplexes microbenchmark")
    println("timing_policy=warm_process_median reps=", reps,
            " field=", field_name, " profile=", profile,
            " threads=", Threads.nthreads())

    if _section_enabled(section, "core")
        push!(rows, _bench("mc complex construct_check", () -> begin
            C = MC.ModuleCochainComplex(fx.C.terms, fx.C.diffs; tmin=fx.C.tmin, check=true)
            _digest_module_complex(C)
        end; reps=reps))

        push!(rows, _bench("mc cochain_map construct_check", () -> begin
            f = MC.ModuleCochainMap(fx.C, fx.C, fx.fmap.comps; tmin=fx.fmap.tmin, tmax=fx.fmap.tmax, check=true)
            _digest_module_map(f)
        end; reps=reps))

        push!(rows, _bench("mc mapping_cone", () -> begin
            Cone = MC.mapping_cone(fx.fmap)
            _digest_module_complex(Cone)
        end; reps=reps))

        push!(rows, _bench("mc cohomology_module t0", () -> begin
            H0 = MC.cohomology_module(fx.C0, 0)
            _digest_pmodule(H0)
        end; reps=reps))

        push!(rows, _bench("mc induced_map_on_cohomology_modules t0", () -> begin
            hf = MC.induced_map_on_cohomology_modules(fx.fmap0, 0)
            _digest_pmorphism(hf)
        end; reps=reps))
    end

    if _section_enabled(section, "rhom")
        push!(rows, _bench("mc rhom_build pmodule serial", () -> begin
            R = MC.RHomComplex(fx.C, fx.N; maxlen=fx.maxlen, resN=fx.resN, threads=false)
            _digest_rhom(R)
        end; reps=reps))

        push!(rows, _bench("mc rhom_build fringe serial", () -> begin
            R = MC.RHomComplex(fx.C, fx.Hn; maxlen=fx.maxlen, resN=fx.resN, threads=false)
            _digest_rhom(R)
        end; reps=reps))

        push!(rows, _bench("mc rhom_build cache cold serial", () -> begin
            R = MC.RHomComplex(fx.C, fx.N; maxlen=fx.maxlen, resN=fx.resN, cache=fx.hcache, threads=false)
            _digest_rhom(R)
        end; reps=reps, setup=() -> DF.clear_hom_system_cache!(fx.hcache)))

        push!(rows, _bench("mc rhom_build cache warm serial", () -> begin
            R = MC.RHomComplex(fx.C, fx.N; maxlen=fx.maxlen, resN=fx.resN, cache=fx.hcache, threads=false)
            _digest_rhom(R)
        end; reps=reps))

        if Threads.nthreads() > 1
            push!(rows, _bench("mc rhom_build pmodule threaded", () -> begin
                R = MC.RHomComplex(fx.C, fx.N; maxlen=fx.maxlen, resN=fx.resN, cache=fx.hcache, threads=true)
                _digest_rhom(R)
            end; reps=reps))
        end
    end

    if _section_enabled(section, "maps")
        push!(rows, _bench("mc rhom_map_first prebuilt", () -> begin
            F = MC.rhom_map_first(fx.fmap, fx.R_serial, fx.R_serial; check=true, cache=fx.hcache, threads=false)
            _digest_chain_map(F)
        end; reps=reps))

        push!(rows, _bench("mc rhom_map_first wrapper", () -> begin
            F = MC.rhom_map_first(fx.fmap, fx.N; maxlen=fx.maxlen, resN=fx.resN, cache=fx.hcache, threads=false)
            _digest_chain_map(F)
        end; reps=reps))

        push!(rows, _bench("mc rhom_map_second prebuilt", () -> begin
            G = MC.rhom_map_second(fx.gN, fx.R_serial, fx.R_serial; check=true, cache=fx.hcache, threads=false)
            _digest_chain_map(G)
        end; reps=reps))

        push!(rows, _bench("mc rhom_map_second wrapper", () -> begin
            G = MC.rhom_map_second(fx.gN, fx.C, fx.Hn, fx.Hn; maxlen=fx.maxlen, cache=fx.hcache, threads=false)
            _digest_chain_map(G)
        end; reps=reps))

        push!(rows, _bench("mc derived_tensor_map_first prebuilt", () -> begin
            F = MC.derived_tensor_map_first(fx.gR, fx.T_serial, fx.T_serial; check=true)
            _digest_chain_map(F)
        end; reps=reps))

        push!(rows, _bench("mc derived_tensor_map_first prebuilt cached", () -> begin
            F = MC.derived_tensor_map_first(fx.gR, fx.T_serial, fx.T_serial; check=true, cache=fx.hcache)
            _digest_chain_map(F)
        end; reps=reps))

        push!(rows, _bench("mc derived_tensor_map_first coeff_lift", () -> begin
            upto = min(length(fx.T_serial.resR.Pmods), length(fx.T_serial.resR.Pmods)) - 1
            coeffs = MC._lift_projective_chainmap_coeff_uncached(
                fx.gR,
                fx.T_serial.resR,
                fx.T_serial.resR;
                upto=upto,
            )
            _digest_sparse_family(coeffs)
        end; reps=reps))

        push!(rows, _bench("mc derived_tensor_map_first assemble_maps", () -> begin
            upto = min(length(fx.T_serial.resR.Pmods), length(fx.T_serial.resR.Pmods)) - 1
            coeffs = MC._lift_projective_chainmap_coeff_uncached(
                fx.gR,
                fx.T_serial.resR,
                fx.T_serial.resR;
                upto=upto,
            )
            plan = MC._tensor_map_first_plan(fx.T_serial, fx.T_serial, nothing)
            maps = MC._assemble_derived_tensor_map_first_maps(coeffs, plan; threads=false)
            _digest_sparse_family(maps)
        end; reps=reps))

        push!(rows, _bench("mc tensor_coeff_block uncached", () -> begin
            upto = min(length(fx.T_serial.resR.Pmods), length(fx.T_serial.resR.Pmods)) - 1
            coeffs = MC._lift_projective_chainmap_coeff_uncached(
                fx.gR,
                fx.T_serial.resR,
                fx.T_serial.resR;
                upto=upto,
            )
            plan = MC._tensor_map_first_plan(fx.T_serial, fx.T_serial, nothing)
            idx = findfirst(d -> !isempty(d.adeg), plan.degrees)
            idx === nothing && return 0
            dplan = plan.degrees[idx]
            k = first(eachindex(dplan.adeg))
            block = MC._tensor_map_on_tor_chains_from_projective_coeff(
                dplan.terms[k],
                dplan.dom_gens[k],
                dplan.cod_gens[k],
                dplan.dom_offsets[k],
                dplan.cod_offsets[k],
                coeffs[dplan.adeg[k] + 1],
            )
            _digest_sparse_family((block,))
        end; reps=reps))

        push!(rows, _bench("mc tensor_coeff_block cached", () -> begin
            upto = min(length(fx.T_serial.resR.Pmods), length(fx.T_serial.resR.Pmods)) - 1
            coeffs = MC._lift_projective_chainmap_coeff_cached(
                fx.gR,
                fx.T_serial.resR,
                fx.T_serial.resR;
                upto=upto,
                cache=fx.hcache,
            )
            plan = MC._tensor_map_first_plan(fx.T_serial, fx.T_serial, fx.hcache)
            idx = findfirst(d -> !isempty(d.adeg), plan.degrees)
            idx === nothing && return 0
            dplan = plan.degrees[idx]
            k = first(eachindex(dplan.adeg))
            block = MC._tensor_map_on_tor_chains_from_projective_coeff(
                dplan.terms[k],
                dplan.dom_gens[k],
                dplan.cod_gens[k],
                dplan.dom_offsets[k],
                dplan.cod_offsets[k],
                coeffs[dplan.adeg[k] + 1];
                cache=fx.hcache,
            )
            _digest_sparse_family((block,))
        end; reps=reps))

        push!(rows, _bench("mc derived_tensor_map_first cochainmap_check", () -> begin
            upto = min(length(fx.T_serial.resR.Pmods), length(fx.T_serial.resR.Pmods)) - 1
            coeffs = MC._lift_projective_chainmap_coeff_uncached(
                fx.gR,
                fx.T_serial.resR,
                fx.T_serial.resR;
                upto=upto,
            )
            plan = MC._tensor_map_first_plan(fx.T_serial, fx.T_serial, nothing)
            maps = MC._assemble_derived_tensor_map_first_maps(coeffs, plan; threads=false)
            F = TamerOp.ChainComplexes.CochainMap(
                fx.T_serial.tot,
                fx.T_serial.tot,
                maps;
                tmin=plan.tmin,
                tmax=plan.tmax,
                check=true,
            )
            _digest_chain_map(F)
        end; reps=reps))

        push!(rows, _bench("mc derived_tensor_map_second prebuilt", () -> begin
            G = MC.derived_tensor_map_second(fx.fmap, fx.T_serial, fx.T_serial; check=true)
            _digest_chain_map(G)
        end; reps=reps))

        push!(rows, _bench("mc derived_tensor_map_second prebuilt cached", () -> begin
            G = MC.derived_tensor_map_second(fx.fmap, fx.T_serial, fx.T_serial; check=true, cache=fx.hcache)
            _digest_chain_map(G)
        end; reps=reps))

        if Threads.nthreads() > 1
            push!(rows, _bench("mc rhom_map_first wrapper threaded", () -> begin
                F = MC.rhom_map_first(fx.fmap, fx.N; maxlen=fx.maxlen, resN=fx.resN, cache=fx.hcache, threads=true)
                _digest_chain_map(F)
            end; reps=reps))
        end
    end

    if _section_enabled(section, "derived")
        push!(rows, _bench("mc hyperExt build pmodule", () -> begin
            H = MC.hyperExt(fx.C0, fx.N; maxlen=fx.maxlen, cache=fx.hcache, threads=false)
            _digest_hyperext(H)
        end; reps=reps))

        push!(rows, _bench("mc hyperExt build fringe", () -> begin
            H = MC.hyperExt(fx.C0, fx.Hn; maxlen=fx.maxlen, cache=fx.hcache, threads=false)
            _digest_hyperext(H)
        end; reps=reps))

        push!(rows, _bench("mc hyperExt_map_first", () -> begin
            A = MC.hyperExt_map_first(fx.fmap0, fx.Hext, fx.Hext; t=fx.t_ext, check=true, cache=fx.hcache)
            size(A, 1) + size(A, 2) + _nnzish(A)
        end; reps=reps))

        push!(rows, _bench("mc hyperExt_map_second", () -> begin
            A = MC.hyperExt_map_second(fx.gN, fx.Hext, fx.Hext; t=fx.t_ext, check=true, cache=fx.hcache)
            size(A, 1) + size(A, 2) + _nnzish(A)
        end; reps=reps))

        push!(rows, _bench("mc derived_tensor_build serial", () -> begin
            T = MC.DerivedTensorComplex(fx.Rop, fx.C; maxlen=fx.maxlen, threads=false)
            _digest_tensor(T)
        end; reps=reps))

        push!(rows, _bench("mc hyperTor build pmodule", () -> begin
            H = MC.hyperTor(fx.Rop, fx.C0; maxlen=fx.maxlen, threads=false)
            _digest_hypertor(H)
        end; reps=reps))

        push!(rows, _bench("mc hyperTor build fringe", () -> begin
            H = MC.hyperTor(fx.Hr, fx.C0; maxlen=fx.maxlen, threads=false)
            _digest_hypertor(H)
        end; reps=reps))

        push!(rows, _bench("mc hyperTor_map_first", () -> begin
            A = MC.hyperTor_map_first(fx.gR, fx.Htor, fx.Htor; n=fx.t_tor, check=true)
            size(A, 1) + size(A, 2) + _nnzish(A)
        end; reps=reps))

        push!(rows, _bench("mc hyperTor_map_first cached", () -> begin
            A = MC.hyperTor_map_first(fx.gR, fx.Htor, fx.Htor; n=fx.t_tor, check=true, cache=fx.hcache)
            size(A, 1) + size(A, 2) + _nnzish(A)
        end; reps=reps))

        push!(rows, _bench("mc hyperTor_map_second", () -> begin
            A = MC.hyperTor_map_second(fx.fmap0, fx.Htor, fx.Htor; n=fx.t_tor, check=true)
            size(A, 1) + size(A, 2) + _nnzish(A)
        end; reps=reps))

        if Threads.nthreads() > 1
            push!(rows, _bench("mc derived_tensor_build threaded", () -> begin
                T = MC.DerivedTensorComplex(fx.Rop, fx.C; maxlen=fx.maxlen, threads=true)
                _digest_tensor(T)
            end; reps=reps))
        end
    end

    _write_csv(out, rows)
    println("Wrote ", out)
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(
        reps=_parse_int_arg(ARGS, "--reps", 5),
        field_name=_parse_string_arg(ARGS, "--field", "qq"),
        profile=_parse_string_arg(ARGS, "--profile", "default"),
        section=_parse_string_arg(ARGS, "--section", "all"),
        out=_parse_path_arg(ARGS, "--out", joinpath(@__DIR__, "_tmp_module_complexes_microbench.csv")),
    )
end
