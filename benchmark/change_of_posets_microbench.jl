#!/usr/bin/env julia
#
# change_of_posets_microbench.jl
#
# Purpose
# - Benchmark the main performance surfaces in `ChangeOfPosets.jl`.
# - Separate refinement/common-poset construction, pullback, direct Kan
#   pushforwards, and derived pushforward paths.
#
# Coverage
# - `product_poset` on dense `FinitePoset` and structured `AbstractPoset`
# - `encode_pmodules_to_common_poset` on its same-object, equal-poset,
#   dense-product, and structured-product paths
# - `pullback` / `restriction` on modules and morphisms
# - `pushforward_left` / `pushforward_right` on modules and morphisms
# - `pushforward_left_complex` / `pushforward_right_complex`
# - `Lpushforward_left` / `Rpushforward_right`
#
# Timing policy
# - Warm-process microbenchmarking (`@timed` median over reps)
# - Cache-sensitive probes use explicit prewarm setup closures so cold and warm
#   behavior are measured separately
#
# Usage
#   julia --project=. benchmark/change_of_posets_microbench.jl
#   julia --project=. benchmark/change_of_posets_microbench.jl --section=refinement --reps=7
#   julia --project=. benchmark/change_of_posets_microbench.jl --field=f3 --n1=8 --n2=7 --nq=9 --np=4
#

using SparseArrays

try
    using TamerOp
catch
    try
        include(joinpath(@__DIR__, "..", "src", "TamerOp.jl"))
        using .TamerOp
    catch
        @info "Falling back to owner-module bootstrap for ChangeOfPosets benchmark"
        @eval module TamerOp
            include($(joinpath(@__DIR__, "..", "src", "CoreModules.jl")))
            include($(joinpath(@__DIR__, "..", "src", "Stats.jl")))
            include($(joinpath(@__DIR__, "..", "src", "Options.jl")))
            include($(joinpath(@__DIR__, "..", "src", "DataTypes.jl")))
            include($(joinpath(@__DIR__, "..", "src", "EncodingCore.jl")))
            include($(joinpath(@__DIR__, "..", "src", "Results.jl")))
            include($(joinpath(@__DIR__, "..", "src", "RegionGeometry.jl")))
            include($(joinpath(@__DIR__, "..", "src", "FieldLinAlg.jl")))
            include($(joinpath(@__DIR__, "..", "src", "FiniteFringe.jl")))
            include($(joinpath(@__DIR__, "..", "src", "IndicatorTypes.jl")))
            include($(joinpath(@__DIR__, "..", "src", "Encoding.jl")))
            include($(joinpath(@__DIR__, "..", "src", "Modules.jl")))
            include($(joinpath(@__DIR__, "..", "src", "AbelianCategories.jl")))
            include($(joinpath(@__DIR__, "..", "src", "IndicatorResolutions.jl")))
            include($(joinpath(@__DIR__, "..", "src", "FlangeZn.jl")))
            include($(joinpath(@__DIR__, "..", "src", "ZnEncoding.jl")))
            include($(joinpath(@__DIR__, "..", "src", "PLPolyhedra.jl")))
            include($(joinpath(@__DIR__, "..", "src", "PLBackend.jl")))
            include($(joinpath(@__DIR__, "..", "src", "ChainComplexes.jl")))
            include($(joinpath(@__DIR__, "..", "src", "DerivedFunctors.jl")))
            include($(joinpath(@__DIR__, "..", "src", "ModuleComplexes.jl")))
            include($(joinpath(@__DIR__, "..", "src", "ChangeOfPosets.jl")))
        end
        using .TamerOp
    end
end

const CM = TamerOp.CoreModules
const OPT = TamerOp.Options
const FF = TamerOp.FiniteFringe
const ENC = TamerOp.Encoding
const MD = TamerOp.Modules
const IR = TamerOp.IndicatorResolutions
const DF = TamerOp.DerivedFunctors
const MC = TamerOp.ModuleComplexes
const CO = TamerOp.ChangeOfPosets

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

@inline function _threads_flag(mode::Symbol)
    mode === :on && return true
    mode === :off && return false
    return Threads.nthreads() > 1
end

@inline function _section_enabled(section::String, group::String)
    section == "all" || return section == group
    return true
end

function _bench(name::AbstractString, section::AbstractString, f::Function;
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
    row = (
        section = String(section),
        probe = String(name),
        median_ms = times_ms[mid],
        median_kib = bytes[mid] / 1024.0,
    )
    println(rpad(string(row.section, ".", row.probe), 64),
            " median_time=", round(row.median_ms, digits=3), " ms",
            "  median_alloc=", round(row.median_kib, digits=1), " KiB")
    return row
end

function _write_csv(path::AbstractString, rows)
    open(path, "w") do io
        println(io, "section,probe,median_ms,median_kib")
        for r in rows
            println(io, string(r.section, ",", r.probe, ",", r.median_ms, ",", r.median_kib))
        end
    end
end

function _bench_maybe!(rows, name::AbstractString, section::AbstractString, f::Function;
                       reps::Int,
                       setup::Union{Nothing,Function}=nothing)
    try
        push!(rows, _bench(name, section, f; reps=reps, setup=setup))
    catch err
        println("Skipping ", section, ".", name, ": ", sprint(showerror, err))
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

function _dense_chain_poset(n::Int)
    rel = falses(n, n)
    @inbounds for i in 1:n, j in i:n
        rel[i, j] = true
    end
    return FF.FinitePoset(rel; check=false)
end

_structured_chain_poset(n::Int) = FF.ProductOfChainsPoset((n,))

@inline _grid2_index(row::Int, col::Int) = row + (col - 1) * 2

function _dense_grid_poset(nx::Int, ny::Int)
    n = nx * ny
    rel = falses(n, n)
    @inbounds for y1 in 1:ny, x1 in 1:nx
        i = x1 + (y1 - 1) * nx
        for y2 in y1:ny, x2 in x1:nx
            j = x2 + (y2 - 1) * nx
            rel[i, j] = true
        end
    end
    return FF.FinitePoset(rel; check=false)
end

function _principal_interval_module(P::FF.AbstractPoset, lo::Int, hi::Int, field::CM.AbstractCoeffField)
    U = FF.principal_upset(P, lo)
    D = FF.principal_downset(P, hi)
    H = FF.one_by_one_fringe(P, U, D, CM.coerce(field, 1); field=field)
    return IR.pmodule_from_fringe(H)
end

function _sum_modules(mods::Vector)
    isempty(mods) && error("_sum_modules: expected at least one module")
    M = mods[1]
    for i in 2:length(mods)
        M = MD.direct_sum(M, mods[i])
    end
    return M
end

function _chain_specs(n::Int, variant::Int)
    if variant == 1
        specs = Tuple{Int,Int}[
            (1, n),
            (1, max(1, n - 2)),
            (2, n),
            (cld(n, 2), n),
        ]
    else
        specs = Tuple{Int,Int}[
            (1, n),
            (2, max(2, n - 1)),
            (max(1, cld(n, 3)), max(cld(n, 2), max(1, cld(n, 3)))),
            (max(1, n - 2), n),
        ]
    end
    seen = Set{Tuple{Int,Int}}()
    out = Tuple{Int,Int}[]
    for spec in specs
        a, b = spec
        a = clamp(a, 1, n)
        b = clamp(b, a, n)
        spec2 = (a, b)
        spec2 in seen && continue
        push!(seen, spec2)
        push!(out, spec2)
    end
    return out
end

function _chain_module(P::FF.AbstractPoset, field::CM.AbstractCoeffField; variant::Int=1)
    mods = [_principal_interval_module(P, a, b, field) for (a, b) in _chain_specs(FF.nvertices(P), variant)]
    return _sum_modules(mods)
end

function _grid_specs(m::Int, variant::Int)
    if variant == 1
        return [
            ((_grid2_index(1, 1)), (_grid2_index(2, m))),
            ((_grid2_index(1, 1)), (_grid2_index(1, m))),
            ((_grid2_index(2, 1)), (_grid2_index(2, m))),
            ((_grid2_index(1, 2)), (_grid2_index(2, max(2, m - 1)))),
        ]
    end
    return [
        ((_grid2_index(1, 1)), (_grid2_index(2, m))),
        ((_grid2_index(1, max(1, cld(m, 2)))), (_grid2_index(2, m))),
        ((_grid2_index(1, 2)), (_grid2_index(1, m))),
        ((_grid2_index(2, 2)), (_grid2_index(2, max(2, m - 1)))),
    ]
end

function _grid_module(Q::FF.AbstractPoset, field::CM.AbstractCoeffField; variant::Int=1)
    n = FF.nvertices(Q)
    n % 2 == 0 || error("_grid_module: expected a 2-by-m grid-style indexing with an even number of vertices")
    m = div(n, 2)
    mods = [_principal_interval_module(Q, lo, hi, field) for (lo, hi) in _grid_specs(m, variant)]
    return _sum_modules(mods)
end

function _module_pair_with_maps(builder::Function, P, field::CM.AbstractCoeffField)
    A = builder(P, field; variant=1)
    B = builder(P, field; variant=2)
    S, iA, iB, pA, pB = MD.direct_sum_with_maps(A, B)
    return (A=A, B=B, S=S, iA=iA, iB=iB, pA=pA, pB=pB)
end

function _grouped_projection_encoding(nq_cols::Int, np::Int)
    Q = _dense_grid_poset(2, nq_cols)
    P = _dense_chain_poset(np)
    pi_of_q = Vector{Int}(undef, FF.nvertices(Q))
    @inbounds for col in 1:nq_cols
        p = min(np, max(1, cld(col * np, nq_cols)))
        pi_of_q[_grid2_index(1, col)] = p
        pi_of_q[_grid2_index(2, col)] = p
    end
    FF.build_cache!(Q; cover=true, updown=true)
    FF.build_cache!(P; cover=true, updown=true)
    return ENC.EncodingMap(Q, P, pi_of_q)
end

function _triangle_boundary_poset()
    n = 6
    leq = falses(n, n)
    for i in 1:n
        leq[i, i] = true
    end
    leq[1, 4] = true
    leq[1, 5] = true
    leq[2, 4] = true
    leq[2, 6] = true
    leq[3, 5] = true
    leq[3, 6] = true
    return FF.FinitePoset(n, leq)
end

@inline _digest_poset(P::FF.AbstractPoset) = FF.nvertices(P)

@inline function _digest_module(M::MD.PModule)
    return FF.nvertices(M.Q) + length(M.edge_maps) + sum(M.dims)
end

@inline function _digest_morphism(f::MD.PMorphism)
    acc = length(f.comps) + sum(f.dom.dims) + sum(f.cod.dims)
    @inbounds for A in f.comps
        acc += size(A, 1) + size(A, 2)
    end
    return acc
end

@inline function _digest_complex(C::MC.ModuleCochainComplex)
    acc = C.tmin + C.tmax
    @inbounds for M in C.terms
        acc += _digest_module(M)
    end
    @inbounds for f in C.diffs
        acc += _digest_morphism(f)
    end
    return acc
end

@inline function _digest_cochain_map(F::MC.ModuleCochainMap)
    acc = _digest_complex(F.C) + _digest_complex(F.D)
    @inbounds for f in F.comps
        acc += _digest_morphism(f)
    end
    return acc
end

@inline function _digest_left_complex_data(data)
    Cdom, Ccod, comps = data
    return _digest_complex(Cdom) + _digest_complex(Ccod) + _digest_morphism_vector(comps)
end

@inline _digest_module_vector(v::AbstractVector{<:MD.PModule}) = sum(_digest_module, v)
@inline _digest_morphism_vector(v::AbstractVector{<:MD.PMorphism}) = sum(_digest_morphism, v)
@inline _digest_sparse_matrix(A::SparseMatrixCSC) = nnz(A) + size(A, 1) + size(A, 2)
@inline _digest_sparse_matrix_vector(v) = sum(nnz, v)

@inline function _digest_product_out(out)
    return _digest_poset(out.P) + out.pi1.pi_of_q[end] + out.pi2.pi_of_q[end]
end

@inline function _digest_common_out(out)
    return _digest_poset(out.P) + _digest_module_vector(out.Ms) + out.pi1.pi_of_q[end] + out.pi2.pi_of_q[end]
end

@inline _digest_namedtuple(nt::NamedTuple) = sum(v isa Integer ? Int(v) : 0 for v in values(nt))

function main(args=ARGS)
    section = _parse_string_arg(args, "--section", "all")
    reps = _parse_int_arg(args, "--reps", 5)
    n1 = _parse_int_arg(args, "--n1", 7)
    n2 = _parse_int_arg(args, "--n2", 6)
    nq = _parse_int_arg(args, "--nq", 8)
    np = _parse_int_arg(args, "--np", 4)
    maxdeg = _parse_int_arg(args, "--maxdeg", 1)
    field_name = _parse_string_arg(args, "--field", "qq")
    thread_mode = _parse_toggle_arg(args, "--threads", :off)
    out_path = _parse_path_arg(args, "--out", joinpath(@__DIR__, "_tmp_change_of_posets_microbench.csv"))

    threads = _threads_flag(thread_mode)
    field = _field_from_name(field_name)
    np_effective = clamp(np, 2, max(2, nq))
    maxlen = maxdeg + 1
    res_opts = OPT.ResolutionOptions(maxlen=maxlen)
    df = OPT.DerivedFunctorOptions(maxdeg=maxdeg)

    println("timing_policy=warm_process_median reps=", reps,
            " threads=", threads,
            " field=", field_name,
            " section=", section)

    # Refinement fixtures.
    Psame = _dense_chain_poset(n1)
    Peq1 = _dense_chain_poset(n1)
    Peq2 = _dense_chain_poset(n1)
    Pd1 = _dense_chain_poset(n1)
    Pd2 = _dense_chain_poset(n2)
    Ps1 = _structured_chain_poset(n1)
    Ps2 = _structured_chain_poset(n2)

    for P in (Psame, Peq1, Peq2, Pd1, Pd2, Ps1, Ps2)
        FF.build_cache!(P; cover=true, updown=true)
    end

    same_pair = _module_pair_with_maps(_chain_module, Psame, field)
    eq_pair_1 = _module_pair_with_maps(_chain_module, Peq1, field)
    eq_pair_2 = _module_pair_with_maps(_chain_module, Peq2, field)
    dense_pair_1 = _module_pair_with_maps(_chain_module, Pd1, field)
    dense_pair_2 = _module_pair_with_maps(_chain_module, Pd2, field)
    struct_pair_1 = _module_pair_with_maps(_chain_module, Ps1, field)
    struct_pair_2 = _module_pair_with_maps(_chain_module, Ps2, field)

    pi = nothing
    pull_pair = nothing
    push_pair = nothing
    push_id = nothing
    proj_res_dom = nothing
    proj_res_cod = nothing
    proj_res_cod_alt = nothing
    inj_res_dom = nothing
    inj_res_cod = nothing

    if _section_enabled(section, "pullback") || _section_enabled(section, "kan") || _section_enabled(section, "derived")
        pi = _grouped_projection_encoding(nq, np_effective)
        pull_pair = _module_pair_with_maps(_chain_module, pi.P, field)
        push_pair = _module_pair_with_maps(_grid_module, pi.Q, field)
        push_id = MD.id_morphism(push_pair.S)
    end

    if _section_enabled(section, "derived")
        proj_res_dom = DF.projective_resolution(push_pair.A, res_opts; threads=threads)
        proj_res_cod = DF.projective_resolution(push_pair.S, res_opts; threads=threads)
        proj_res_cod_alt = DF.projective_resolution(push_pair.S, res_opts; threads=threads)
        inj_res_dom = DF.injective_resolution(push_pair.A, res_opts; threads=threads)
        inj_res_cod = DF.injective_resolution(push_pair.S, res_opts; threads=threads)
    end

    rows = NamedTuple[]

    if _section_enabled(section, "refinement")
        push!(rows, _bench("product_poset.dense_cold", "refinement",
            () -> _digest_product_out(CO.product_poset(Pd1, Pd2;
                                                       check=false,
                                                       cache_cover_edges=true,
                                                       use_cache=false,
                                                       session_cache=nothing));
            reps=reps))

        dense_sc_ref = Ref{Any}(nothing)
        dense_setup = function ()
            sc = CM.SessionCache()
            dense_sc_ref[] = sc
            CO.product_poset(Pd1, Pd2;
                             check=false,
                             cache_cover_edges=true,
                             use_cache=true,
                             session_cache=sc)
            return nothing
        end
        push!(rows, _bench("product_poset.dense_warm_cache", "refinement",
            () -> _digest_product_out(CO.product_poset(Pd1, Pd2;
                                                       check=false,
                                                       cache_cover_edges=true,
                                                       use_cache=true,
                                                       session_cache=dense_sc_ref[]));
            reps=reps, setup=dense_setup))

        push!(rows, _bench("product_poset.structured_cold", "refinement",
            () -> _digest_product_out(CO.product_poset(Ps1, Ps2;
                                                       check=false,
                                                       cache_cover_edges=true,
                                                       use_cache=false,
                                                       session_cache=nothing));
            reps=reps))

        struct_sc_ref = Ref{Any}(nothing)
        struct_setup = function ()
            sc = CM.SessionCache()
            struct_sc_ref[] = sc
            CO.product_poset(Ps1, Ps2;
                             check=false,
                             cache_cover_edges=true,
                             use_cache=true,
                             session_cache=sc)
            return nothing
        end
        push!(rows, _bench("product_poset.structured_warm_cache", "refinement",
            () -> _digest_product_out(CO.product_poset(Ps1, Ps2;
                                                       check=false,
                                                       cache_cover_edges=true,
                                                       use_cache=true,
                                                       session_cache=struct_sc_ref[]));
            reps=reps, setup=struct_setup))

        push!(rows, _bench("encode_common_poset.same_object", "refinement",
            () -> _digest_common_out(CO.encode_pmodules_to_common_poset(same_pair.A, same_pair.B;
                                                                        session_cache=nothing));
            reps=reps))

        push!(rows, _bench("encode_common_poset.poset_equal", "refinement",
            () -> _digest_common_out(CO.encode_pmodules_to_common_poset(eq_pair_1.A, eq_pair_2.A;
                                                                        session_cache=nothing));
            reps=reps))

        push!(rows, _bench("encode_common_poset.product_dense_cold", "refinement",
            () -> _digest_common_out(CO.encode_pmodules_to_common_poset(dense_pair_1.A, dense_pair_2.A;
                                                                        use_cache=false,
                                                                        session_cache=nothing));
            reps=reps))

        old_fast = CO._CHANGE_OF_POSETS_USE_PRODUCT_PROJECTION_FASTPATH[]
        old_fused = CO._CHANGE_OF_POSETS_USE_FUSED_PRODUCT_TRANSLATION[]
        dense_fallback_setup = function ()
            CO._CHANGE_OF_POSETS_USE_PRODUCT_PROJECTION_FASTPATH[] = false
            CO._CHANGE_OF_POSETS_USE_FUSED_PRODUCT_TRANSLATION[] = false
            return nothing
        end
        dense_fast_setup = function ()
            CO._CHANGE_OF_POSETS_USE_PRODUCT_PROJECTION_FASTPATH[] = true
            CO._CHANGE_OF_POSETS_USE_FUSED_PRODUCT_TRANSLATION[] = true
            return nothing
        end
        push!(rows, _bench("encode_common_poset.product_dense_cold_fallback", "refinement",
            () -> _digest_common_out(CO.encode_pmodules_to_common_poset(dense_pair_1.A, dense_pair_2.A;
                                                                        use_cache=false,
                                                                        session_cache=nothing));
            reps=reps, setup=dense_fallback_setup))
        push!(rows, _bench("encode_common_poset.product_dense_cold_fast", "refinement",
            () -> _digest_common_out(CO.encode_pmodules_to_common_poset(dense_pair_1.A, dense_pair_2.A;
                                                                        use_cache=false,
                                                                        session_cache=nothing));
            reps=reps, setup=dense_fast_setup))
        push!(rows, _bench("hom_common_refinement.product_dense_cold_fast", "refinement",
            () -> DF.dim(CO.hom_common_refinement(dense_pair_1.A, dense_pair_2.A;
                                                  use_cache=false,
                                                  session_cache=nothing));
            reps=reps, setup=dense_fast_setup))
        push!(rows, _bench("encode_common_poset.product_dense_cold_then_hom", "refinement",
            () -> begin
                out = CO.encode_pmodules_to_common_poset(dense_pair_1.A, dense_pair_2.A;
                                                         use_cache=false,
                                                         session_cache=nothing)
                DF.dim(DF.Hom(out.Ms[1], out.Ms[2]))
            end;
            reps=reps, setup=dense_fast_setup))
        direct_hom_off_setup = function ()
            CO._CHANGE_OF_POSETS_USE_DIRECT_HOM_COMMON_REFINEMENT[] = false
            CO._CHANGE_OF_POSETS_USE_PRODUCT_PROJECTION_FASTPATH[] = true
            CO._CHANGE_OF_POSETS_USE_FUSED_PRODUCT_TRANSLATION[] = true
            return nothing
        end
        direct_hom_on_setup = function ()
            CO._CHANGE_OF_POSETS_USE_DIRECT_HOM_COMMON_REFINEMENT[] = true
            CO._CHANGE_OF_POSETS_USE_PRODUCT_PROJECTION_FASTPATH[] = true
            CO._CHANGE_OF_POSETS_USE_FUSED_PRODUCT_TRANSLATION[] = true
            return nothing
        end
        push!(rows, _bench("hom_common_refinement.product_dense_cold_fallback", "refinement",
            () -> DF.dim(CO.hom_common_refinement(dense_pair_1.A, dense_pair_2.A;
                                                  use_cache=false,
                                                  session_cache=nothing));
            reps=reps, setup=direct_hom_off_setup))
        push!(rows, _bench("hom_common_refinement.product_dense_cold_direct", "refinement",
            () -> DF.dim(CO.hom_common_refinement(dense_pair_1.A, dense_pair_2.A;
                                                  use_cache=false,
                                                  session_cache=nothing));
            reps=reps, setup=direct_hom_on_setup))
        push!(rows, _bench("hom_common_refinement.product_dense_basis_fallback", "refinement",
            () -> length(DF.basis(CO.hom_common_refinement(dense_pair_1.A, dense_pair_2.A;
                                                           use_cache=false,
                                                           session_cache=nothing)));
            reps=reps, setup=direct_hom_off_setup))
        push!(rows, _bench("hom_common_refinement.product_dense_basis_direct", "refinement",
            () -> length(DF.basis(CO.hom_common_refinement(dense_pair_1.A, dense_pair_2.A;
                                                           use_cache=false,
                                                           session_cache=nothing)));
            reps=reps, setup=direct_hom_on_setup))
        push!(rows, _bench("hom_common_refinement.product_dense_basis_collect_fallback", "refinement",
            () -> length(collect(DF.basis(CO.hom_common_refinement(dense_pair_1.A, dense_pair_2.A;
                                                                   use_cache=false,
                                                                   session_cache=nothing))));
            reps=reps, setup=direct_hom_off_setup))
        push!(rows, _bench("hom_common_refinement.product_dense_basis_collect_direct", "refinement",
            () -> length(collect(DF.basis(CO.hom_common_refinement(dense_pair_1.A, dense_pair_2.A;
                                                                   use_cache=false,
                                                                   session_cache=nothing))));
            reps=reps, setup=direct_hom_on_setup))
        push!(rows, _bench("hom_dim_common_refinement.product_dense_direct", "refinement",
            () -> CO.hom_dim_common_refinement(dense_pair_1.A, dense_pair_2.A;
                                               use_cache=false,
                                               session_cache=nothing);
            reps=reps, setup=direct_hom_on_setup))
        push!(rows, _bench("has_nonzero_hom_common_refinement.product_dense_direct", "refinement",
            () -> CO.has_nonzero_hom_common_refinement(dense_pair_1.A, dense_pair_2.A;
                                                       use_cache=false,
                                                       session_cache=nothing);
            reps=reps, setup=direct_hom_on_setup))
        push!(rows, _bench("hom_bidim_common_refinement.product_dense_direct", "refinement",
            () -> _digest_namedtuple(CO.hom_bidim_common_refinement(dense_pair_1.A, dense_pair_2.A;
                                                                    use_cache=false,
                                                                    session_cache=nothing));
            reps=reps, setup=direct_hom_on_setup))
        CO._CHANGE_OF_POSETS_USE_PRODUCT_PROJECTION_FASTPATH[] = old_fast
        CO._CHANGE_OF_POSETS_USE_FUSED_PRODUCT_TRANSLATION[] = old_fused
        CO._CHANGE_OF_POSETS_USE_DIRECT_HOM_COMMON_REFINEMENT[] = true

        common_dense_sc_ref = Ref{Any}(nothing)
        common_dense_setup = function ()
            sc = CM.SessionCache()
            common_dense_sc_ref[] = sc
            CO.encode_pmodules_to_common_poset(dense_pair_1.A, dense_pair_2.A;
                                               use_cache=true,
                                               session_cache=sc)
            return nothing
        end
        push!(rows, _bench("encode_common_poset.product_dense_warm_cache", "refinement",
            () -> _digest_common_out(CO.encode_pmodules_to_common_poset(dense_pair_1.A, dense_pair_2.A;
                                                                        use_cache=true,
                                                                        session_cache=common_dense_sc_ref[]));
            reps=reps, setup=common_dense_setup))

        push!(rows, _bench("encode_common_poset.product_structured_cold", "refinement",
            () -> _digest_common_out(CO.encode_pmodules_to_common_poset(struct_pair_1.A, struct_pair_2.A;
                                                                        use_cache=false,
                                                                        session_cache=nothing));
            reps=reps))

        struct_fallback_setup = function ()
            CO._CHANGE_OF_POSETS_USE_PRODUCT_PROJECTION_FASTPATH[] = false
            CO._CHANGE_OF_POSETS_USE_FUSED_PRODUCT_TRANSLATION[] = false
            return nothing
        end
        struct_fast_setup = function ()
            CO._CHANGE_OF_POSETS_USE_PRODUCT_PROJECTION_FASTPATH[] = true
            CO._CHANGE_OF_POSETS_USE_FUSED_PRODUCT_TRANSLATION[] = true
            return nothing
        end
        push!(rows, _bench("encode_common_poset.product_structured_cold_fallback", "refinement",
            () -> _digest_common_out(CO.encode_pmodules_to_common_poset(struct_pair_1.A, struct_pair_2.A;
                                                                        use_cache=false,
                                                                        session_cache=nothing));
            reps=reps, setup=struct_fallback_setup))
        push!(rows, _bench("encode_common_poset.product_structured_cold_fast", "refinement",
            () -> _digest_common_out(CO.encode_pmodules_to_common_poset(struct_pair_1.A, struct_pair_2.A;
                                                                        use_cache=false,
                                                                        session_cache=nothing));
            reps=reps, setup=struct_fast_setup))
        push!(rows, _bench("hom_common_refinement.product_structured_cold_fast", "refinement",
            () -> DF.dim(CO.hom_common_refinement(struct_pair_1.A, struct_pair_2.A;
                                                  use_cache=false,
                                                  session_cache=nothing));
            reps=reps, setup=struct_fast_setup))
        push!(rows, _bench("encode_common_poset.product_structured_cold_then_hom", "refinement",
            () -> begin
                out = CO.encode_pmodules_to_common_poset(struct_pair_1.A, struct_pair_2.A;
                                                         use_cache=false,
                                                         session_cache=nothing)
                Qd = FF.FinitePoset(FF.leq_matrix(out.Ms[1].Q); check=false)
                dense1 = MD.PModule{CM.coeff_type(field)}(Qd, out.Ms[1].dims, out.Ms[1].edge_maps; field=out.Ms[1].field)
                dense2 = MD.PModule{CM.coeff_type(field)}(Qd, out.Ms[2].dims, out.Ms[2].edge_maps; field=out.Ms[2].field)
                DF.dim(DF.Hom(dense1, dense2))
            end;
            reps=reps, setup=struct_fast_setup))
        push!(rows, _bench("hom_common_refinement.product_structured_cold_fallback", "refinement",
            () -> DF.dim(CO.hom_common_refinement(struct_pair_1.A, struct_pair_2.A;
                                                  use_cache=false,
                                                  session_cache=nothing));
            reps=reps, setup=direct_hom_off_setup))
        push!(rows, _bench("hom_common_refinement.product_structured_cold_direct", "refinement",
            () -> DF.dim(CO.hom_common_refinement(struct_pair_1.A, struct_pair_2.A;
                                                  use_cache=false,
                                                  session_cache=nothing));
            reps=reps, setup=direct_hom_on_setup))
        push!(rows, _bench("hom_common_refinement.product_structured_basis_fallback", "refinement",
            () -> length(DF.basis(CO.hom_common_refinement(struct_pair_1.A, struct_pair_2.A;
                                                           use_cache=false,
                                                           session_cache=nothing)));
            reps=reps, setup=direct_hom_off_setup))
        push!(rows, _bench("hom_common_refinement.product_structured_basis_direct", "refinement",
            () -> length(DF.basis(CO.hom_common_refinement(struct_pair_1.A, struct_pair_2.A;
                                                           use_cache=false,
                                                           session_cache=nothing)));
            reps=reps, setup=direct_hom_on_setup))
        push!(rows, _bench("hom_common_refinement.product_structured_basis_collect_fallback", "refinement",
            () -> length(collect(DF.basis(CO.hom_common_refinement(struct_pair_1.A, struct_pair_2.A;
                                                                   use_cache=false,
                                                                   session_cache=nothing))));
            reps=reps, setup=direct_hom_off_setup))
        push!(rows, _bench("hom_common_refinement.product_structured_basis_collect_direct", "refinement",
            () -> length(collect(DF.basis(CO.hom_common_refinement(struct_pair_1.A, struct_pair_2.A;
                                                                   use_cache=false,
                                                                   session_cache=nothing))));
            reps=reps, setup=direct_hom_on_setup))
        push!(rows, _bench("hom_dim_common_refinement.product_structured_direct", "refinement",
            () -> CO.hom_dim_common_refinement(struct_pair_1.A, struct_pair_2.A;
                                               use_cache=false,
                                               session_cache=nothing);
            reps=reps, setup=direct_hom_on_setup))
        push!(rows, _bench("has_nonzero_hom_common_refinement.product_structured_direct", "refinement",
            () -> CO.has_nonzero_hom_common_refinement(struct_pair_1.A, struct_pair_2.A;
                                                       use_cache=false,
                                                       session_cache=nothing);
            reps=reps, setup=direct_hom_on_setup))
        push!(rows, _bench("hom_bidim_common_refinement.product_structured_direct", "refinement",
            () -> _digest_namedtuple(CO.hom_bidim_common_refinement(struct_pair_1.A, struct_pair_2.A;
                                                                    use_cache=false,
                                                                    session_cache=nothing));
            reps=reps, setup=direct_hom_on_setup))
        CO._CHANGE_OF_POSETS_USE_PRODUCT_PROJECTION_FASTPATH[] = old_fast
        CO._CHANGE_OF_POSETS_USE_FUSED_PRODUCT_TRANSLATION[] = old_fused
        CO._CHANGE_OF_POSETS_USE_DIRECT_HOM_COMMON_REFINEMENT[] = true

        common_struct_sc_ref = Ref{Any}(nothing)
        common_struct_setup = function ()
            sc = CM.SessionCache()
            common_struct_sc_ref[] = sc
            CO.encode_pmodules_to_common_poset(struct_pair_1.A, struct_pair_2.A;
                                               use_cache=true,
                                               session_cache=sc)
            return nothing
        end
        push!(rows, _bench("encode_common_poset.product_structured_warm_cache", "refinement",
            () -> _digest_common_out(CO.encode_pmodules_to_common_poset(struct_pair_1.A, struct_pair_2.A;
                                                                        use_cache=true,
                                                                        session_cache=common_struct_sc_ref[]));
            reps=reps, setup=common_struct_setup))
    end

    if _section_enabled(section, "pullback")
        push!(rows, _bench("pullback.module", "pullback",
            () -> _digest_module(CO.pullback(pi, pull_pair.A; check=true));
            reps=reps))

        pull_sc_ref = Ref{Any}(nothing)
        pull_module_setup = function ()
            sc = CM.SessionCache()
            pull_sc_ref[] = sc
            CO.pullback(pi, pull_pair.A; check=true, session_cache=sc)
            return nothing
        end
        push!(rows, _bench("pullback.module_warm_cache", "pullback",
            () -> _digest_module(CO.pullback(pi, pull_pair.A; check=true,
                                             session_cache=pull_sc_ref[]));
            reps=reps, setup=pull_module_setup))

        push!(rows, _bench("pullback.morphism", "pullback",
            () -> _digest_morphism(CO.pullback(pi, pull_pair.iA; check=true));
            reps=reps))

        pull_morphism_setup = function ()
            sc = CM.SessionCache()
            pull_sc_ref[] = sc
            CO.pullback(pi, pull_pair.iA; check=true, session_cache=sc)
            return nothing
        end
        push!(rows, _bench("pullback.morphism_warm_cache", "pullback",
            () -> _digest_morphism(CO.pullback(pi, pull_pair.iA; check=true,
                                               session_cache=pull_sc_ref[]));
            reps=reps, setup=pull_morphism_setup))

        push!(rows, _bench("restriction.module_alias", "pullback",
            () -> _digest_module(CO.restriction(pi, pull_pair.A; check=true));
            reps=reps))
    end

    if _section_enabled(section, "kan")
        push!(rows, _bench("pushforward_left.module", "kan",
            () -> _digest_module(CO.pushforward_left(pi, push_pair.A; check=true, threads=threads));
            reps=reps))
        push!(rows, _bench("pushforward_left.module_with_summary_quotient", "kan",
            () -> begin
                old = CO._CHANGE_OF_POSETS_USE_LEFT_KAN_QQ_SUMMARY_QUOTIENT[]
                CO._CHANGE_OF_POSETS_USE_LEFT_KAN_QQ_SUMMARY_QUOTIENT[] = true
                try
                    _digest_module(CO.pushforward_left(pi, push_pair.A; check=true, threads=threads))
                finally
                    CO._CHANGE_OF_POSETS_USE_LEFT_KAN_QQ_SUMMARY_QUOTIENT[] = old
                end
            end;
            reps=reps))

        kan_sc_ref = Ref{Any}(nothing)
        left_module_setup = function ()
            sc = CM.SessionCache()
            kan_sc_ref[] = sc
            CO.pushforward_left(pi, push_pair.A; check=true, threads=threads, session_cache=sc)
            return nothing
        end
        push!(rows, _bench("pushforward_left.module_warm_cache", "kan",
            () -> _digest_module(CO.pushforward_left(pi, push_pair.A;
                                                     check=true,
                                                     threads=threads,
                                                     session_cache=kan_sc_ref[]));
            reps=reps, setup=left_module_setup))

        push!(rows, _bench("pushforward_left.morphism", "kan",
            () -> _digest_morphism(CO.pushforward_left(pi, push_pair.iA; check=true, threads=threads));
            reps=reps))
        push!(rows, _bench("pushforward_left.morphism_with_summary_quotient", "kan",
            () -> begin
                old = CO._CHANGE_OF_POSETS_USE_LEFT_KAN_QQ_SUMMARY_QUOTIENT[]
                CO._CHANGE_OF_POSETS_USE_LEFT_KAN_QQ_SUMMARY_QUOTIENT[] = true
                try
                    _digest_morphism(CO.pushforward_left(pi, push_pair.iA; check=true, threads=threads))
                finally
                    CO._CHANGE_OF_POSETS_USE_LEFT_KAN_QQ_SUMMARY_QUOTIENT[] = old
                end
            end;
            reps=reps))
        push!(rows, _bench("pushforward_left.morphism_with_mul_fastpath", "kan",
            () -> begin
                old = CO._CHANGE_OF_POSETS_USE_LEFT_KAN_MORPHISM_MUL[]
                CO._CHANGE_OF_POSETS_USE_LEFT_KAN_MORPHISM_MUL[] = true
                try
                    _digest_morphism(CO.pushforward_left(pi, push_pair.iA; check=true, threads=threads))
                finally
                    CO._CHANGE_OF_POSETS_USE_LEFT_KAN_MORPHISM_MUL[] = old
                end
            end;
            reps=reps))
        left_morphism_data_ref = Ref{Any}(nothing)
        left_morphism_data_setup = function ()
            plan = CO._translation_plan(pi; session_cache=nothing)
            dom_out, data_dom = CO._left_kan_data(pi, push_pair.iA.dom;
                                                  check=false,
                                                  threads=threads,
                                                  session_cache=nothing,
                                                  plan=plan)
            cod_out, data_cod = CO._left_kan_data(pi, push_pair.iA.cod;
                                                  check=false,
                                                  threads=threads,
                                                  session_cache=nothing,
                                                  plan=plan)
            left_morphism_data_ref[] = (dom_out, cod_out, data_dom, data_cod, plan.left_fibers)
            return nothing
        end
        left_plan_ref = Ref{Any}(nothing)
        left_data_setup = function ()
            left_plan_ref[] = CO._translation_plan(pi; session_cache=nothing)
            return nothing
        end
        push!(rows, _bench("pushforward_left.data_prebuilt_plan", "kan",
            () -> begin
                plan = left_plan_ref[]
                _digest_module(first(CO._left_kan_data(pi, push_pair.A;
                                                       check=false,
                                                       threads=threads,
                                                       session_cache=nothing,
                                                       plan=plan)))
            end;
            reps=reps, setup=left_data_setup))
        push!(rows, _bench("pushforward_left.data_prebuilt_plan_with_summary_quotient", "kan",
            () -> begin
                old = CO._CHANGE_OF_POSETS_USE_LEFT_KAN_QQ_SUMMARY_QUOTIENT[]
                CO._CHANGE_OF_POSETS_USE_LEFT_KAN_QQ_SUMMARY_QUOTIENT[] = true
                try
                    plan = left_plan_ref[]
                    _digest_module(first(CO._left_kan_data(pi, push_pair.A;
                                                           check=false,
                                                           threads=threads,
                                                           session_cache=nothing,
                                                           plan=plan)))
                finally
                    CO._CHANGE_OF_POSETS_USE_LEFT_KAN_QQ_SUMMARY_QUOTIENT[] = old
                end
            end;
            reps=reps, setup=left_data_setup))
        push!(rows, _bench("pushforward_left.morphism_prebuilt_data", "kan",
            () -> begin
                dom_out, cod_out, data_dom, data_cod, fibers = left_morphism_data_ref[]
                _digest_morphism(CO._pushforward_left_morphism_from_data(
                    dom_out, cod_out, push_pair.iA, data_dom, data_cod, fibers; threads=threads
                ))
            end;
            reps=reps, setup=left_morphism_data_setup))
        push!(rows, _bench("pushforward_left.morphism_prebuilt_data_with_mul_fastpath", "kan",
            () -> begin
                old = CO._CHANGE_OF_POSETS_USE_LEFT_KAN_MORPHISM_MUL[]
                CO._CHANGE_OF_POSETS_USE_LEFT_KAN_MORPHISM_MUL[] = true
                try
                    dom_out, cod_out, data_dom, data_cod, fibers = left_morphism_data_ref[]
                    _digest_morphism(CO._pushforward_left_morphism_from_data(
                        dom_out, cod_out, push_pair.iA, data_dom, data_cod, fibers; threads=threads
                    ))
                finally
                    CO._CHANGE_OF_POSETS_USE_LEFT_KAN_MORPHISM_MUL[] = old
                end
            end;
            reps=reps, setup=left_morphism_data_setup))

        left_morphism_setup = function ()
            sc = CM.SessionCache()
            kan_sc_ref[] = sc
            CO.pushforward_left(pi, push_pair.iA; check=true, threads=threads, session_cache=sc)
            return nothing
        end
        push!(rows, _bench("pushforward_left.morphism_warm_cache", "kan",
            () -> _digest_morphism(CO.pushforward_left(pi, push_pair.iA;
                                                       check=true,
                                                       threads=threads,
                                                       session_cache=kan_sc_ref[]));
            reps=reps, setup=left_morphism_setup))

        push!(rows, _bench("pushforward_right.module", "kan",
            () -> _digest_module(CO.pushforward_right(pi, push_pair.A; check=true, threads=threads));
            reps=reps))
        push!(rows, _bench("pushforward_right.module_with_direct_csc", "kan",
            () -> begin
                old = CO._CHANGE_OF_POSETS_USE_RIGHT_KAN_DIRECT_CSC[]
                CO._CHANGE_OF_POSETS_USE_RIGHT_KAN_DIRECT_CSC[] = true
                try
                    _digest_module(CO.pushforward_right(pi, push_pair.A; check=true, threads=threads))
                finally
                    CO._CHANGE_OF_POSETS_USE_RIGHT_KAN_DIRECT_CSC[] = old
                end
            end;
            reps=reps))
        push!(rows, _bench("pushforward_right.module_no_selector_fastpath", "kan",
            () -> begin
                old = CO._CHANGE_OF_POSETS_USE_RIGHT_KAN_SELECTOR_FASTPATH[]
                CO._CHANGE_OF_POSETS_USE_RIGHT_KAN_SELECTOR_FASTPATH[] = false
                try
                    _digest_module(CO.pushforward_right(pi, push_pair.A; check=true, threads=threads))
                finally
                    CO._CHANGE_OF_POSETS_USE_RIGHT_KAN_SELECTOR_FASTPATH[] = old
                end
            end;
            reps=reps))
        push!(rows, _bench("pushforward_right.module_no_summary_selector", "kan",
            () -> begin
                old = CO._CHANGE_OF_POSETS_USE_RIGHT_KAN_QQ_SUMMARY_SELECTOR[]
                CO._CHANGE_OF_POSETS_USE_RIGHT_KAN_QQ_SUMMARY_SELECTOR[] = false
                try
                    _digest_module(CO.pushforward_right(pi, push_pair.A; check=true, threads=threads))
                finally
                    CO._CHANGE_OF_POSETS_USE_RIGHT_KAN_QQ_SUMMARY_SELECTOR[] = old
                end
            end;
            reps=reps))

        right_module_setup = function ()
            sc = CM.SessionCache()
            kan_sc_ref[] = sc
            CO.pushforward_right(pi, push_pair.A; check=true, threads=threads, session_cache=sc)
            return nothing
        end
        push!(rows, _bench("pushforward_right.module_warm_cache", "kan",
            () -> _digest_module(CO.pushforward_right(pi, push_pair.A;
                                                      check=true,
                                                      threads=threads,
                                                      session_cache=kan_sc_ref[]));
            reps=reps, setup=right_module_setup))

        push!(rows, _bench("pushforward_right.morphism", "kan",
            () -> _digest_morphism(CO.pushforward_right(pi, push_pair.iA; check=true, threads=threads));
            reps=reps))
        push!(rows, _bench("pushforward_right.morphism_with_direct_csc", "kan",
            () -> begin
                old = CO._CHANGE_OF_POSETS_USE_RIGHT_KAN_DIRECT_CSC[]
                CO._CHANGE_OF_POSETS_USE_RIGHT_KAN_DIRECT_CSC[] = true
                try
                    _digest_morphism(CO.pushforward_right(pi, push_pair.iA; check=true, threads=threads))
                finally
                    CO._CHANGE_OF_POSETS_USE_RIGHT_KAN_DIRECT_CSC[] = old
                end
            end;
            reps=reps))
        right_morphism_data_ref = Ref{Any}(nothing)
        right_morphism_data_setup = function ()
            plan = CO._translation_plan(pi; session_cache=nothing)
            dom_out, data_dom = CO._right_kan_data(pi, push_pair.iA.dom;
                                                   check=false,
                                                   threads=threads,
                                                   session_cache=nothing,
                                                   plan=plan)
            cod_out, data_cod = CO._right_kan_data(pi, push_pair.iA.cod;
                                                   check=false,
                                                   threads=threads,
                                                   session_cache=nothing,
                                                   plan=plan)
            right_morphism_data_ref[] = (dom_out, cod_out, data_dom, data_cod, plan.right_fibers)
            return nothing
        end
        push!(rows, _bench("pushforward_right.morphism_prebuilt_data", "kan",
            () -> begin
                dom_out, cod_out, data_dom, data_cod, fibers = right_morphism_data_ref[]
                _digest_morphism(CO._pushforward_right_morphism_from_data(
                    dom_out, cod_out, push_pair.iA, data_dom, data_cod, fibers; threads=threads
                ))
            end;
            reps=reps, setup=right_morphism_data_setup))
        right_general_ref = Ref{Any}(nothing)
        right_general_setup = function ()
            Qtri = _triangle_boundary_poset()
            Ppt = _dense_chain_poset(1)
            pitri = ENC.EncodingMap(Qtri, Ppt, fill(1, FF.nvertices(Qtri)))
            dims = ones(Int, FF.nvertices(Qtri))
            oneK = CM.coerce(field, 1)
            Ktri = typeof(oneK)
            edge_maps = Dict{Tuple{Int,Int}, SparseMatrixCSC{Ktri,Int}}()
            for (a, b) in FF.cover_edges(Qtri)
                edge_maps[(a, b)] = sparse(fill(oneK, 1, 1))
            end
            Mtri = MD.PModule{Ktri}(Qtri, dims, edge_maps; field=field)
            plan = CO._translation_plan(pitri; session_cache=nothing)
            fiber = only(plan.right_fibers)
            offp, _ = CO._offset_prefix(fiber.idxs, Mtri.dims)
            C = CO._right_kan_constraint_matrix(Mtri, fiber, offp)
            right_general_ref[] = (Mtri, fiber, offp, C)
            return nothing
        end
        push!(rows, _bench("pushforward_right.data_prebuilt_plan", "kan",
            () -> begin
                plan = CO._translation_plan(pi; session_cache=nothing)
                _digest_module(first(CO._right_kan_data(pi, push_pair.A;
                                                        check=false,
                                                        threads=threads,
                                                        session_cache=nothing,
                                                        plan=plan)))
            end;
            reps=reps))
        push!(rows, _bench("pushforward_right.data_prebuilt_plan_with_direct_csc", "kan",
            () -> begin
                old = CO._CHANGE_OF_POSETS_USE_RIGHT_KAN_DIRECT_CSC[]
                CO._CHANGE_OF_POSETS_USE_RIGHT_KAN_DIRECT_CSC[] = true
                try
                    plan = CO._translation_plan(pi; session_cache=nothing)
                    _digest_module(first(CO._right_kan_data(pi, push_pair.A;
                                                            check=false,
                                                            threads=threads,
                                                            session_cache=nothing,
                                                            plan=plan)))
                finally
                    CO._CHANGE_OF_POSETS_USE_RIGHT_KAN_DIRECT_CSC[] = old
                end
            end;
            reps=reps))
        push!(rows, _bench("pushforward_right.data_general_equation_build", "kan",
            () -> begin
                Mtri, fiber, offp, _ = right_general_ref[]
                _digest_sparse_matrix(CO._right_kan_constraint_matrix(Mtri, fiber, offp))
            end;
            reps=reps, setup=right_general_setup))
        push!(rows, _bench("pushforward_right.data_general_equation_build_with_direct_csc", "kan",
            () -> begin
                old = CO._CHANGE_OF_POSETS_USE_RIGHT_KAN_DIRECT_CSC[]
                CO._CHANGE_OF_POSETS_USE_RIGHT_KAN_DIRECT_CSC[] = true
                try
                    Mtri, fiber, offp, _ = right_general_ref[]
                    _digest_sparse_matrix(CO._right_kan_constraint_matrix(Mtri, fiber, offp))
                finally
                    CO._CHANGE_OF_POSETS_USE_RIGHT_KAN_DIRECT_CSC[] = old
                end
            end;
            reps=reps, setup=right_general_setup))
        push!(rows, _bench("pushforward_right.data_general_selector", "kan",
            () -> begin
                Mtri, _, _, C = right_general_ref[]
                Kp, Lp = CO._nullspace_selector_summary(Mtri.field, C)
                nnz(sparse(Kp)) + nnz(sparse(Lp)) + size(Kp, 1) + size(Kp, 2) + size(Lp, 1) + size(Lp, 2)
            end;
            reps=reps, setup=right_general_setup))
        push!(rows, _bench("pushforward_right.data_general_selector_no_summary_selector", "kan",
            () -> begin
                old = CO._CHANGE_OF_POSETS_USE_RIGHT_KAN_QQ_SUMMARY_SELECTOR[]
                CO._CHANGE_OF_POSETS_USE_RIGHT_KAN_QQ_SUMMARY_SELECTOR[] = false
                try
                    Mtri, _, _, C = right_general_ref[]
                    Kp, Lp = CO._nullspace_selector_summary(Mtri.field, C)
                    nnz(sparse(Kp)) + nnz(sparse(Lp)) + size(Kp, 1) + size(Kp, 2) + size(Lp, 1) + size(Lp, 2)
                finally
                    CO._CHANGE_OF_POSETS_USE_RIGHT_KAN_QQ_SUMMARY_SELECTOR[] = old
                end
            end;
            reps=reps, setup=right_general_setup))
        push!(rows, _bench("pushforward_right.data_general_selector_no_selector_fastpath", "kan",
            () -> begin
                old = CO._CHANGE_OF_POSETS_USE_RIGHT_KAN_SELECTOR_FASTPATH[]
                CO._CHANGE_OF_POSETS_USE_RIGHT_KAN_SELECTOR_FASTPATH[] = false
                try
                    Mtri, _, _, C = right_general_ref[]
                    Kp, Lp = CO._nullspace_selector_summary(Mtri.field, C)
                    nnz(sparse(Kp)) + nnz(sparse(Lp)) + size(Kp, 1) + size(Kp, 2) + size(Lp, 1) + size(Lp, 2)
                finally
                    CO._CHANGE_OF_POSETS_USE_RIGHT_KAN_SELECTOR_FASTPATH[] = old
                end
            end;
            reps=reps, setup=right_general_setup))
        push!(rows, _bench("pushforward_right.morphism_no_summary_selector", "kan",
            () -> begin
                old = CO._CHANGE_OF_POSETS_USE_RIGHT_KAN_QQ_SUMMARY_SELECTOR[]
                CO._CHANGE_OF_POSETS_USE_RIGHT_KAN_QQ_SUMMARY_SELECTOR[] = false
                try
                    _digest_morphism(CO.pushforward_right(pi, push_pair.iA; check=true, threads=threads))
                finally
                    CO._CHANGE_OF_POSETS_USE_RIGHT_KAN_QQ_SUMMARY_SELECTOR[] = old
                end
            end;
            reps=reps))
        push!(rows, _bench("pushforward_right.morphism_no_selector_fastpath", "kan",
            () -> begin
                old = CO._CHANGE_OF_POSETS_USE_RIGHT_KAN_SELECTOR_FASTPATH[]
                CO._CHANGE_OF_POSETS_USE_RIGHT_KAN_SELECTOR_FASTPATH[] = false
                try
                    _digest_morphism(CO.pushforward_right(pi, push_pair.iA; check=true, threads=threads))
                finally
                    CO._CHANGE_OF_POSETS_USE_RIGHT_KAN_SELECTOR_FASTPATH[] = old
                end
            end;
            reps=reps))

        right_morphism_setup = function ()
            sc = CM.SessionCache()
            kan_sc_ref[] = sc
            CO.pushforward_right(pi, push_pair.iA; check=true, threads=threads, session_cache=sc)
            return nothing
        end
        push!(rows, _bench("pushforward_right.morphism_warm_cache", "kan",
            () -> _digest_morphism(CO.pushforward_right(pi, push_pair.iA;
                                                        check=true,
                                                        threads=threads,
                                                        session_cache=kan_sc_ref[]));
            reps=reps, setup=right_morphism_setup))
    end

    if _section_enabled(section, "derived")
        _bench_maybe!(rows, "lift_chainmap.identity_distinct_res", "derived",
            () -> _digest_sparse_matrix_vector(DF.Functoriality.lift_chainmap(
                proj_res_cod, proj_res_cod_alt, push_id; maxlen=maxdeg + 1
            ));
            reps=reps)

        _bench_maybe!(rows, "pushforward_left_complex.module_reuse_res", "derived",
            () -> _digest_complex(CO.pushforward_left_complex(pi, push_pair.A, df;
                                                              check=true,
                                                              res=proj_res_dom,
                                                              threads=threads));
            reps=reps)

        _bench_maybe!(rows, "pushforward_left_complex.morphism_reuse_res", "derived",
            () -> _digest_cochain_map(CO.pushforward_left_complex(pi, push_id, df;
                                                                  check=true,
                                                                  res_dom=proj_res_cod,
                                                                  res_cod=proj_res_cod,
                                                                  threads=threads));
            reps=reps)
        _bench_maybe!(rows, "pushforward_left_complex.morphism_reuse_res_no_diffs_from_data", "derived",
            () -> begin
                old = CO._CHANGE_OF_POSETS_USE_LEFT_DERIVED_DIFFS_FROM_DATA[]
                CO._CHANGE_OF_POSETS_USE_LEFT_DERIVED_DIFFS_FROM_DATA[] = false
                try
                    _digest_cochain_map(CO.pushforward_left_complex(pi, push_id, df;
                                                                    check=true,
                                                                    res_dom=proj_res_cod,
                                                                    res_cod=proj_res_cod,
                                                                    threads=threads))
                finally
                    CO._CHANGE_OF_POSETS_USE_LEFT_DERIVED_DIFFS_FROM_DATA[] = old
                end
            end;
            reps=reps)
        _bench_maybe!(rows, "pushforward_left_complex.morphism_reuse_res_no_coeff_plan_cache", "derived",
            () -> begin
                old = CO._CHANGE_OF_POSETS_USE_LEFT_DERIVED_COEFF_PLAN_CACHE[]
                CO._CHANGE_OF_POSETS_USE_LEFT_DERIVED_COEFF_PLAN_CACHE[] = false
                try
                    _digest_cochain_map(CO.pushforward_left_complex(pi, push_id, df;
                                                                    check=true,
                                                                    res_dom=proj_res_cod,
                                                                    res_cod=proj_res_cod,
                                                                    threads=threads))
                finally
                    CO._CHANGE_OF_POSETS_USE_LEFT_DERIVED_COEFF_PLAN_CACHE[] = old
                end
            end;
            reps=reps)
        _bench_maybe!(rows, "pushforward_left_complex.morphism_reuse_res_no_coeff_fastpath", "derived",
            () -> begin
                old = CO._CHANGE_OF_POSETS_USE_DERIVED_LEFT_COEFF_FASTPATH[]
                CO._CHANGE_OF_POSETS_USE_DERIVED_LEFT_COEFF_FASTPATH[] = false
                try
                    _digest_cochain_map(CO.pushforward_left_complex(pi, push_id, df;
                                                                    check=true,
                                                                    res_dom=proj_res_cod,
                                                                    res_cod=proj_res_cod,
                                                                    threads=threads))
                finally
                    CO._CHANGE_OF_POSETS_USE_DERIVED_LEFT_COEFF_FASTPATH[] = old
                end
            end;
            reps=reps)
        _bench_maybe!(rows, "pushforward_left_complex.data_reuse_res", "derived",
            () -> _digest_left_complex_data(CO._pushforward_left_complex_data(
                pi, push_id, df;
                res_dom=proj_res_cod,
                res_cod=proj_res_cod,
                threads=threads,
                session_cache=nothing,
            ));
            reps=reps)

        left_complex_sc_ref = Ref{Any}(nothing)
        left_complex_setup = function ()
            sc = CM.SessionCache()
            left_complex_sc_ref[] = sc
            CO.pushforward_left_complex(pi, push_id, df;
                                        check=true,
                                        res_dom=proj_res_cod,
                                        res_cod=proj_res_cod,
                                        threads=threads,
                                        session_cache=sc)
            return nothing
        end
        _bench_maybe!(rows, "pushforward_left_complex.morphism_warm_cache", "derived",
            () -> _digest_cochain_map(CO.pushforward_left_complex(pi, push_id, df;
                                                                  check=true,
                                                                  res_dom=proj_res_cod,
                                                                  res_cod=proj_res_cod,
                                                                  threads=threads,
                                                                  session_cache=left_complex_sc_ref[]));
            reps=reps, setup=left_complex_setup)

        _bench_maybe!(rows, "pushforward_right_complex.module_reuse_res", "derived",
            () -> _digest_complex(CO.pushforward_right_complex(pi, push_pair.A, df;
                                                               check=true,
                                                               res=inj_res_dom,
                                                               threads=threads));
            reps=reps)

        _bench_maybe!(rows, "pushforward_right_complex.morphism_reuse_res", "derived",
            () -> _digest_cochain_map(CO.pushforward_right_complex(pi, push_id, df;
                                                                   check=true,
                                                                   res_dom=inj_res_cod,
                                                                   res_cod=inj_res_cod,
                                                                   threads=threads));
            reps=reps)
        _bench_maybe!(rows, "pushforward_right_complex.morphism_reuse_res_no_shared_resolution_fastpath", "derived",
            () -> begin
                old_shared = CO._CHANGE_OF_POSETS_USE_RIGHT_DERIVED_SHARED_RESOLUTION_FASTPATH[]
                old_id = CO._CHANGE_OF_POSETS_USE_RIGHT_DERIVED_IDENTITY_MAP_FASTPATH[]
                CO._CHANGE_OF_POSETS_USE_RIGHT_DERIVED_SHARED_RESOLUTION_FASTPATH[] = false
                CO._CHANGE_OF_POSETS_USE_RIGHT_DERIVED_IDENTITY_MAP_FASTPATH[] = false
                try
                    _digest_cochain_map(CO.pushforward_right_complex(pi, push_id, df;
                                                                     check=true,
                                                                     res_dom=inj_res_cod,
                                                                     res_cod=inj_res_cod,
                                                                     threads=threads))
                finally
                    CO._CHANGE_OF_POSETS_USE_RIGHT_DERIVED_SHARED_RESOLUTION_FASTPATH[] = old_shared
                    CO._CHANGE_OF_POSETS_USE_RIGHT_DERIVED_IDENTITY_MAP_FASTPATH[] = old_id
                end
            end;
            reps=reps)
        _bench_maybe!(rows, "pushforward_right_complex.morphism_reuse_res_no_diffs_from_data", "derived",
            () -> begin
                old = CO._CHANGE_OF_POSETS_USE_RIGHT_DERIVED_DIFFS_FROM_DATA[]
                CO._CHANGE_OF_POSETS_USE_RIGHT_DERIVED_DIFFS_FROM_DATA[] = false
                try
                    _digest_cochain_map(CO.pushforward_right_complex(pi, push_id, df;
                                                                     check=true,
                                                                     res_dom=inj_res_cod,
                                                                     res_cod=inj_res_cod,
                                                                     threads=threads))
                finally
                    CO._CHANGE_OF_POSETS_USE_RIGHT_DERIVED_DIFFS_FROM_DATA[] = old
                end
            end;
            reps=reps)
        _bench_maybe!(rows, "pushforward_right_complex.morphism_reuse_res_with_coeff_fastpath", "derived",
            () -> begin
                old = CO._CHANGE_OF_POSETS_USE_DERIVED_RIGHT_COEFF_FASTPATH[]
                CO._CHANGE_OF_POSETS_USE_DERIVED_RIGHT_COEFF_FASTPATH[] = true
                try
                    _digest_cochain_map(CO.pushforward_right_complex(pi, push_id, df;
                                                                     check=true,
                                                                     res_dom=inj_res_cod,
                                                                     res_cod=inj_res_cod,
                                                                     threads=threads))
                finally
                    CO._CHANGE_OF_POSETS_USE_DERIVED_RIGHT_COEFF_FASTPATH[] = old
                end
            end;
            reps=reps)
        _bench_maybe!(rows, "pushforward_right_complex.morphism_reuse_res_with_coeff_fastpath_no_plan_cache", "derived",
            () -> begin
                old_fast = CO._CHANGE_OF_POSETS_USE_DERIVED_RIGHT_COEFF_FASTPATH[]
                old_plan = CO._CHANGE_OF_POSETS_USE_RIGHT_DERIVED_COEFF_PLAN_CACHE[]
                CO._CHANGE_OF_POSETS_USE_DERIVED_RIGHT_COEFF_FASTPATH[] = true
                CO._CHANGE_OF_POSETS_USE_RIGHT_DERIVED_COEFF_PLAN_CACHE[] = false
                try
                    _digest_cochain_map(CO.pushforward_right_complex(pi, push_id, df;
                                                                     check=true,
                                                                     res_dom=inj_res_cod,
                                                                     res_cod=inj_res_cod,
                                                                     threads=threads))
                finally
                    CO._CHANGE_OF_POSETS_USE_DERIVED_RIGHT_COEFF_FASTPATH[] = old_fast
                    CO._CHANGE_OF_POSETS_USE_RIGHT_DERIVED_COEFF_PLAN_CACHE[] = old_plan
                end
            end;
            reps=reps)
        _bench_maybe!(rows, "pushforward_right_complex.morphism_reuse_res_with_coeff_fastpath_no_matrix_recurrence", "derived",
            () -> begin
                old_fast = CO._CHANGE_OF_POSETS_USE_DERIVED_RIGHT_COEFF_FASTPATH[]
                old_recur = CO._CHANGE_OF_POSETS_USE_RIGHT_DERIVED_COEFF_MATRIX_RECURRENCE[]
                CO._CHANGE_OF_POSETS_USE_DERIVED_RIGHT_COEFF_FASTPATH[] = true
                CO._CHANGE_OF_POSETS_USE_RIGHT_DERIVED_COEFF_MATRIX_RECURRENCE[] = false
                try
                    _digest_cochain_map(CO.pushforward_right_complex(pi, push_id, df;
                                                                     check=true,
                                                                     res_dom=inj_res_cod,
                                                                     res_cod=inj_res_cod,
                                                                     threads=threads))
                finally
                    CO._CHANGE_OF_POSETS_USE_DERIVED_RIGHT_COEFF_FASTPATH[] = old_fast
                    CO._CHANGE_OF_POSETS_USE_RIGHT_DERIVED_COEFF_MATRIX_RECURRENCE[] = old_recur
                end
            end;
            reps=reps)
        _bench_maybe!(rows, "pushforward_right_complex.morphism_reuse_res_no_coeff_fastpath", "derived",
            () -> begin
                old = CO._CHANGE_OF_POSETS_USE_DERIVED_RIGHT_COEFF_FASTPATH[]
                CO._CHANGE_OF_POSETS_USE_DERIVED_RIGHT_COEFF_FASTPATH[] = false
                try
                    _digest_cochain_map(CO.pushforward_right_complex(pi, push_id, df;
                                                                     check=true,
                                                                     res_dom=inj_res_cod,
                                                                     res_cod=inj_res_cod,
                                                                     threads=threads))
                finally
                    CO._CHANGE_OF_POSETS_USE_DERIVED_RIGHT_COEFF_FASTPATH[] = old
                end
            end;
            reps=reps)
        _bench_maybe!(rows, "pushforward_right_complex.morphism_reuse_res_no_selector_fastpath", "derived",
            () -> begin
                old = CO._CHANGE_OF_POSETS_USE_RIGHT_KAN_SELECTOR_FASTPATH[]
                CO._CHANGE_OF_POSETS_USE_RIGHT_KAN_SELECTOR_FASTPATH[] = false
                try
                    _digest_cochain_map(CO.pushforward_right_complex(pi, push_id, df;
                                                                     check=true,
                                                                     res_dom=inj_res_cod,
                                                                     res_cod=inj_res_cod,
                                                                     threads=threads))
                finally
                    CO._CHANGE_OF_POSETS_USE_RIGHT_KAN_SELECTOR_FASTPATH[] = old
                end
            end;
            reps=reps)

        right_complex_sc_ref = Ref{Any}(nothing)
        right_complex_setup = function ()
            sc = CM.SessionCache()
            right_complex_sc_ref[] = sc
            CO.pushforward_right_complex(pi, push_id, df;
                                         check=true,
                                         res_dom=inj_res_cod,
                                         res_cod=inj_res_cod,
                                         threads=threads,
                                         session_cache=sc)
            return nothing
        end
        _bench_maybe!(rows, "pushforward_right_complex.morphism_warm_cache", "derived",
            () -> _digest_cochain_map(CO.pushforward_right_complex(pi, push_id, df;
                                                                   check=true,
                                                                   res_dom=inj_res_cod,
                                                                   res_cod=inj_res_cod,
                                                                   threads=threads,
                                                                   session_cache=right_complex_sc_ref[]));
            reps=reps, setup=right_complex_setup)

        _bench_maybe!(rows, "Lpushforward_left.module_full", "derived",
            () -> _digest_module_vector(CO.Lpushforward_left(pi, push_pair.A, df;
                                                             check=true,
                                                             threads=threads));
            reps=reps)

        _bench_maybe!(rows, "Lpushforward_left.morphism_reuse_res", "derived",
            () -> _digest_morphism_vector(CO.Lpushforward_left(pi, push_id, df;
                                                                 check=true,
                                                                 res_dom=proj_res_cod,
                                                                 res_cod=proj_res_cod,
                                                                 threads=threads));
            reps=reps)
        _bench_maybe!(rows, "Lpushforward_left.morphism_reuse_res_no_cohomology_shortcut", "derived",
            () -> begin
                old = CO._CHANGE_OF_POSETS_USE_LEFT_DERIVED_COHOMOLOGY_SHORTCUT[]
                CO._CHANGE_OF_POSETS_USE_LEFT_DERIVED_COHOMOLOGY_SHORTCUT[] = false
                try
                    _digest_morphism_vector(CO.Lpushforward_left(pi, push_id, df;
                                                                   check=true,
                                                                   res_dom=proj_res_cod,
                                                                   res_cod=proj_res_cod,
                                                                   threads=threads))
                finally
                    CO._CHANGE_OF_POSETS_USE_LEFT_DERIVED_COHOMOLOGY_SHORTCUT[] = old
                end
            end;
            reps=reps)
        _bench_maybe!(rows, "Lpushforward_left.morphism_reuse_res_no_coeff_fastpath", "derived",
            () -> begin
                old = CO._CHANGE_OF_POSETS_USE_DERIVED_LEFT_COEFF_FASTPATH[]
                CO._CHANGE_OF_POSETS_USE_DERIVED_LEFT_COEFF_FASTPATH[] = false
                try
                    _digest_morphism_vector(CO.Lpushforward_left(pi, push_id, df;
                                                                   check=true,
                                                                   res_dom=proj_res_cod,
                                                                   res_cod=proj_res_cod,
                                                                   threads=threads))
                finally
                    CO._CHANGE_OF_POSETS_USE_DERIVED_LEFT_COEFF_FASTPATH[] = old
                end
            end;
            reps=reps)

        left_vec_sc_ref = Ref{Any}(nothing)
        left_vec_setup = function ()
            sc = CM.SessionCache()
            left_vec_sc_ref[] = sc
            CO.Lpushforward_left(pi, push_id, df;
                                 check=true,
                                 res_dom=proj_res_cod,
                                 res_cod=proj_res_cod,
                                 threads=threads,
                                 session_cache=sc)
            return nothing
        end
        _bench_maybe!(rows, "Lpushforward_left.morphism_warm_cache", "derived",
            () -> _digest_morphism_vector(CO.Lpushforward_left(pi, push_id, df;
                                                                 check=true,
                                                                 res_dom=proj_res_cod,
                                                                 res_cod=proj_res_cod,
                                                                 threads=threads,
                                                                 session_cache=left_vec_sc_ref[]));
            reps=reps, setup=left_vec_setup)

        _bench_maybe!(rows, "Rpushforward_right.module_full", "derived",
            () -> _digest_module_vector(CO.Rpushforward_right(pi, push_pair.A, df;
                                                              check=true,
                                                              threads=threads));
            reps=reps)

        _bench_maybe!(rows, "Rpushforward_right.morphism_reuse_res", "derived",
            () -> _digest_morphism_vector(CO.Rpushforward_right(pi, push_id, df;
                                                                  check=true,
                                                                  res_dom=inj_res_cod,
                                                                  res_cod=inj_res_cod,
                                                                  threads=threads));
            reps=reps)
        _bench_maybe!(rows, "Rpushforward_right.morphism_reuse_res_with_coeff_fastpath_no_matrix_recurrence", "derived",
            () -> begin
                old_fast = CO._CHANGE_OF_POSETS_USE_DERIVED_RIGHT_COEFF_FASTPATH[]
                old_recur = CO._CHANGE_OF_POSETS_USE_RIGHT_DERIVED_COEFF_MATRIX_RECURRENCE[]
                CO._CHANGE_OF_POSETS_USE_DERIVED_RIGHT_COEFF_FASTPATH[] = true
                CO._CHANGE_OF_POSETS_USE_RIGHT_DERIVED_COEFF_MATRIX_RECURRENCE[] = false
                try
                    _digest_morphism_vector(CO.Rpushforward_right(pi, push_id, df;
                                                                    check=true,
                                                                    res_dom=inj_res_cod,
                                                                    res_cod=inj_res_cod,
                                                                    threads=threads))
                finally
                    CO._CHANGE_OF_POSETS_USE_DERIVED_RIGHT_COEFF_FASTPATH[] = old_fast
                    CO._CHANGE_OF_POSETS_USE_RIGHT_DERIVED_COEFF_MATRIX_RECURRENCE[] = old_recur
                end
            end;
            reps=reps)
        _bench_maybe!(rows, "Rpushforward_right.morphism_reuse_res_no_coeff_fastpath", "derived",
            () -> begin
                old = CO._CHANGE_OF_POSETS_USE_DERIVED_RIGHT_COEFF_FASTPATH[]
                CO._CHANGE_OF_POSETS_USE_DERIVED_RIGHT_COEFF_FASTPATH[] = false
                try
                    _digest_morphism_vector(CO.Rpushforward_right(pi, push_id, df;
                                                                    check=true,
                                                                    res_dom=inj_res_cod,
                                                                    res_cod=inj_res_cod,
                                                                    threads=threads))
                finally
                    CO._CHANGE_OF_POSETS_USE_DERIVED_RIGHT_COEFF_FASTPATH[] = old
                end
            end;
            reps=reps)
        _bench_maybe!(rows, "Rpushforward_right.morphism_reuse_res_no_selector_fastpath", "derived",
            () -> begin
                old = CO._CHANGE_OF_POSETS_USE_RIGHT_KAN_SELECTOR_FASTPATH[]
                CO._CHANGE_OF_POSETS_USE_RIGHT_KAN_SELECTOR_FASTPATH[] = false
                try
                    _digest_morphism_vector(CO.Rpushforward_right(pi, push_id, df;
                                                                    check=true,
                                                                    res_dom=inj_res_cod,
                                                                    res_cod=inj_res_cod,
                                                                    threads=threads))
                finally
                    CO._CHANGE_OF_POSETS_USE_RIGHT_KAN_SELECTOR_FASTPATH[] = old
                end
            end;
            reps=reps)

        right_vec_sc_ref = Ref{Any}(nothing)
        right_vec_setup = function ()
            sc = CM.SessionCache()
            right_vec_sc_ref[] = sc
            CO.Rpushforward_right(pi, push_id, df;
                                  check=true,
                                  res_dom=inj_res_cod,
                                  res_cod=inj_res_cod,
                                  threads=threads,
                                  session_cache=sc)
            return nothing
        end
        _bench_maybe!(rows, "Rpushforward_right.morphism_warm_cache", "derived",
            () -> _digest_morphism_vector(CO.Rpushforward_right(pi, push_id, df;
                                                                  check=true,
                                                                  res_dom=inj_res_cod,
                                                                  res_cod=inj_res_cod,
                                                                  threads=threads,
                                                                  session_cache=right_vec_sc_ref[]));
            reps=reps, setup=right_vec_setup)
    end

    _write_csv(out_path, rows)
    println("wrote ", out_path)
    return nothing
end

main()
