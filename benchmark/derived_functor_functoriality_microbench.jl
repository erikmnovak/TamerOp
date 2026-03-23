#!/usr/bin/env julia

using SparseArrays

try
    using TamerOp
catch
    include(joinpath(@__DIR__, "..", "src", "TamerOp.jl"))
    using .TamerOp
end

const CM = TamerOp.CoreModules
const FF = TamerOp.FiniteFringe
const IR = TamerOp.IndicatorResolutions
const DF = TamerOp.DerivedFunctors
const OPT = TamerOp.Options
const MD = TamerOp.Modules

function _parse_int_arg(args, key::String, default::Int)
    prefix = key * "="
    for a in args
        startswith(a, prefix) || continue
        return max(1, parse(Int, split(a, "=", limit=2)[2]))
    end
    return default
end

function _median_stats(f::Function; reps::Int)
    GC.gc()
    f()
    GC.gc()
    times = Vector{Float64}(undef, reps)
    bytes = Vector{Int}(undef, reps)
    for i in 1:reps
        stats = @timed f()
        times[i] = 1000.0 * stats.time
        bytes[i] = stats.bytes
    end
    sort!(times)
    sort!(bytes)
    mid = cld(reps, 2)
    return (ms=times[mid], kib=bytes[mid] / 1024.0)
end

function _bench_ab(name::String, f_before::Function, f_after::Function; reps::Int)
    before = _median_stats(f_before; reps=reps)
    after = _median_stats(f_after; reps=reps)
    speedup = after.ms == 0.0 ? Inf : before.ms / after.ms
    alloc_ratio = before.kib == 0.0 ? Inf : after.kib / before.kib
    println(rpad(name, 44),
            " before=", round(before.ms, digits=3), " ms / ", round(before.kib, digits=1), " KiB",
            "  after=", round(after.ms, digits=3), " ms / ", round(after.kib, digits=1), " KiB",
            "  speedup=", round(speedup, digits=2), "x",
            "  alloc_ratio=", round(alloc_ratio, digits=2), "x")
    return (probe=name, before_ms=before.ms, before_kib=before.kib,
            after_ms=after.ms, after_kib=after.kib, speedup=speedup, alloc_ratio=alloc_ratio)
end

function _write_csv(path::AbstractString, rows)
    open(path, "w") do io
        println(io, "probe,before_ms,before_kib,after_ms,after_kib,speedup,alloc_ratio")
        for r in rows
            println(io, string(r.probe, ",", r.before_ms, ",", r.before_kib, ",",
                               r.after_ms, ",", r.after_kib, ",", r.speedup, ",", r.alloc_ratio))
        end
    end
end

function _diamond_poset()
    rel = falses(4, 4)
    for i in 1:4
        rel[i, i] = true
    end
    rel[1, 2] = true
    rel[1, 3] = true
    rel[2, 4] = true
    rel[3, 4] = true
    rel[1, 4] = true
    return FF.FinitePoset(rel; check=false)
end

function _chain_poset(n::Int)
    rel = falses(n, n)
    @inbounds for i in 1:n, j in i:n
        rel[i, j] = true
    end
    return FF.FinitePoset(rel; check=false)
end

function _simple_module(P, v, field)
    K = CM.coeff_type(field)
    H = FF.one_by_one_fringe(P, FF.principal_upset(P, v), FF.principal_downset(P, v), one(K);
                             field=field)
    return IR.pmodule_from_fringe(H)
end

function _fixture()
    field = CM.QQField()
    K = CM.coeff_type(field)
    c(x) = CM.coerce(field, x)
    P = _diamond_poset()
    Pop = FF.FinitePoset(transpose(FF.leq_matrix(P)); check=false)

    S1 = _simple_module(P, 1, field)
    S2 = _simple_module(P, 2, field)
    S4 = _simple_module(P, 4, field)
    Rop = _simple_module(Pop, 2, field)
    L = _simple_module(P, 1, field)

    E14 = DF.Ext(S1, S4, OPT.DerivedFunctorOptions(maxdeg=2, model=:projective))
    E24 = DF.Ext(S2, S4, OPT.DerivedFunctorOptions(maxdeg=1, model=:projective))
    E12 = DF.Ext(S1, S2, OPT.DerivedFunctorOptions(maxdeg=2, model=:projective))
    T = DF.Tor(Rop, L, OPT.DerivedFunctorOptions(maxdeg=1, model=:first))

    coeff_ext = DF.Functoriality._lift_pmodule_map_to_projective_resolution_chainmap_coeff(
        E14.res, E14.res, IR.id_morphism(S1); upto=1
    )[2]
    coeff_tor = DF.Functoriality._lift_pmodule_map_to_projective_resolution_chainmap_coeff(
        T.resRop, T.resRop, IR.id_morphism(Rop); upto=1
    )[2]

    return (
        field=field,
        K=K,
        c=c,
        E14=E14,
        E24=E24,
        E12=E12,
        T=T,
        coeff_ext=coeff_ext,
        coeff_tor=coeff_tor,
    )
end

function _fixture_moderate(; n::Int=8)
    field = CM.QQField()
    P = _chain_poset(n)
    Pop = FF.FinitePoset(transpose(FF.leq_matrix(P)); check=false)

    S1 = _simple_module(P, 1, field)
    Smid = _simple_module(P, max(2, cld(n, 2)), field)
    Send = _simple_module(P, n, field)
    Rop = _simple_module(Pop, max(2, cld(n, 2)), field)

    E1 = DF.Ext(S1, Send, OPT.DerivedFunctorOptions(maxdeg=2, model=:projective))
    T = DF.Tor(Rop, S1, OPT.DerivedFunctorOptions(maxdeg=2, model=:first))

    coeff_ext = DF.Functoriality._lift_pmodule_map_to_projective_resolution_chainmap_coeff(
        E1.res, E1.res, IR.id_morphism(S1); upto=1
    )[2]
    coeff_tor = DF.Functoriality._lift_pmodule_map_to_projective_resolution_chainmap_coeff(
        T.resRop, T.resRop, IR.id_morphism(Rop); upto=1
    )[2]

    return (E1=E1, T=T, coeff_ext=coeff_ext, coeff_tor=coeff_tor)
end

function main(; reps::Int=5,
              out::String=joinpath(@__DIR__, "_tmp_derived_functor_functoriality_microbench.csv"))
    fx = _fixture()
    rows = NamedTuple[]

    old_trip = DF.Functoriality._FUNCTORIALITY_USE_DIRECT_COEFF_TRIPLETS[]
    old_mv = DF.Functoriality._FUNCTORIALITY_USE_DIRECT_COEFF_MATVECS[]
    old_cache = DF.Functoriality._FUNCTORIALITY_USE_COEFF_PLAN_CACHE[]
    old_trip_nnz = DF.Functoriality._FUNCTORIALITY_DIRECT_COEFF_MIN_NNZ[]
    old_cocycle_nnz = DF.Functoriality._FUNCTORIALITY_DIRECT_COCYCLE_MIN_NNZ[]
    fxm = _fixture_moderate()

    println("DerivedFunctors functoriality microbenchmark")
    println("timing_policy=warm_process_median reps=", reps, " threads=", Threads.nthreads())

    try
        push!(rows, _bench_ab("precompose_hom_coeff",
            () -> begin
                DF.Functoriality._FUNCTORIALITY_USE_DIRECT_COEFF_TRIPLETS[] = false
                DF.Functoriality._FUNCTORIALITY_USE_COEFF_PLAN_CACHE[] = false
                DF.Functoriality._FUNCTORIALITY_DIRECT_COEFF_MIN_NNZ[] = typemax(Int)
                F = DF.Functoriality._precompose_on_hom_cochains_from_projective_coeff(
                    fx.E14.N,
                    fx.E14.res.gens[2],
                    fx.E14.res.gens[2],
                    fx.E14.offsets[2],
                    fx.E14.offsets[2],
                    fx.coeff_ext,
                )
                size(F, 1) + nnz(F)
            end,
            () -> begin
                DF.Functoriality._FUNCTORIALITY_USE_DIRECT_COEFF_TRIPLETS[] = true
                DF.Functoriality._FUNCTORIALITY_USE_COEFF_PLAN_CACHE[] = true
                DF.Functoriality._FUNCTORIALITY_DIRECT_COEFF_MIN_NNZ[] = 0
                F = DF.Functoriality._precompose_on_hom_cochains_from_projective_coeff(
                    fx.E14.N,
                    fx.E14.res.gens[2],
                    fx.E14.res.gens[2],
                    fx.E14.offsets[2],
                    fx.E14.offsets[2],
                    fx.coeff_ext,
                )
                size(F, 1) + nnz(F)
            end;
            reps=reps))

        push!(rows, _bench_ab("precompose_hom_coeff_moderate",
            () -> begin
                DF.Functoriality._FUNCTORIALITY_USE_DIRECT_COEFF_TRIPLETS[] = false
                DF.Functoriality._FUNCTORIALITY_USE_COEFF_PLAN_CACHE[] = false
                DF.Functoriality._FUNCTORIALITY_DIRECT_COEFF_MIN_NNZ[] = typemax(Int)
                F = DF.Functoriality._precompose_on_hom_cochains_from_projective_coeff(
                    fxm.E1.N,
                    fxm.E1.res.gens[2],
                    fxm.E1.res.gens[2],
                    fxm.E1.offsets[2],
                    fxm.E1.offsets[2],
                    fxm.coeff_ext,
                )
                size(F, 1) + nnz(F)
            end,
            () -> begin
                DF.Functoriality._FUNCTORIALITY_USE_DIRECT_COEFF_TRIPLETS[] = true
                DF.Functoriality._FUNCTORIALITY_USE_COEFF_PLAN_CACHE[] = true
                DF.Functoriality._FUNCTORIALITY_DIRECT_COEFF_MIN_NNZ[] = 0
                F = DF.Functoriality._precompose_on_hom_cochains_from_projective_coeff(
                    fxm.E1.N,
                    fxm.E1.res.gens[2],
                    fxm.E1.res.gens[2],
                    fxm.E1.offsets[2],
                    fxm.E1.offsets[2],
                    fxm.coeff_ext,
                )
                size(F, 1) + nnz(F)
            end;
            reps=reps))

        push!(rows, _bench_ab("tensor_tor_coeff",
            () -> begin
                DF.Functoriality._FUNCTORIALITY_USE_DIRECT_COEFF_TRIPLETS[] = false
                DF.Functoriality._FUNCTORIALITY_USE_COEFF_PLAN_CACHE[] = false
                DF.Functoriality._FUNCTORIALITY_DIRECT_COEFF_MIN_NNZ[] = typemax(Int)
                F = DF.Functoriality._tensor_map_on_tor_chains_from_projective_coeff(
                    fx.T.L,
                    fx.T.resRop.gens[2],
                    fx.T.resRop.gens[2],
                    fx.T.offsets[2],
                    fx.T.offsets[2],
                    fx.coeff_tor,
                )
                size(F, 1) + nnz(F)
            end,
            () -> begin
                DF.Functoriality._FUNCTORIALITY_USE_DIRECT_COEFF_TRIPLETS[] = true
                DF.Functoriality._FUNCTORIALITY_USE_COEFF_PLAN_CACHE[] = true
                DF.Functoriality._FUNCTORIALITY_DIRECT_COEFF_MIN_NNZ[] = 0
                F = DF.Functoriality._tensor_map_on_tor_chains_from_projective_coeff(
                    fx.T.L,
                    fx.T.resRop.gens[2],
                    fx.T.resRop.gens[2],
                    fx.T.offsets[2],
                    fx.T.offsets[2],
                    fx.coeff_tor,
                )
                size(F, 1) + nnz(F)
            end;
            reps=reps))

        push!(rows, _bench_ab("yoneda_product",
            () -> begin
                DF.Functoriality._FUNCTORIALITY_USE_DIRECT_COEFF_MATVECS[] = false
                DF.Functoriality._FUNCTORIALITY_USE_COEFF_PLAN_CACHE[] = false
                DF.Functoriality._FUNCTORIALITY_DIRECT_COCYCLE_MIN_NNZ[] = typemax(Int)
                _, coords = DF.yoneda_product(fx.E24, 1, [fx.c(1)], fx.E12, 1, [fx.c(1)]; ELN=fx.E14)
                sum(coords)
            end,
            () -> begin
                DF.Functoriality._FUNCTORIALITY_USE_DIRECT_COEFF_MATVECS[] = true
                DF.Functoriality._FUNCTORIALITY_USE_COEFF_PLAN_CACHE[] = true
                DF.Functoriality._FUNCTORIALITY_DIRECT_COCYCLE_MIN_NNZ[] = 0
                _, coords = DF.yoneda_product(fx.E24, 1, [fx.c(1)], fx.E12, 1, [fx.c(1)]; ELN=fx.E14)
                sum(coords)
            end;
            reps=reps))

        _write_csv(out, rows)
        println("wrote ", out)
        return rows
    finally
        DF.Functoriality._FUNCTORIALITY_USE_DIRECT_COEFF_TRIPLETS[] = old_trip
        DF.Functoriality._FUNCTORIALITY_USE_DIRECT_COEFF_MATVECS[] = old_mv
        DF.Functoriality._FUNCTORIALITY_USE_COEFF_PLAN_CACHE[] = old_cache
        DF.Functoriality._FUNCTORIALITY_DIRECT_COEFF_MIN_NNZ[] = old_trip_nnz
        DF.Functoriality._FUNCTORIALITY_DIRECT_COCYCLE_MIN_NNZ[] = old_cocycle_nnz
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(reps=_parse_int_arg(ARGS, "--reps", 5))
end
