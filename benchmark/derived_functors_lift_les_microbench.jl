#!/usr/bin/env julia

try
    using TamerOp
catch
    include(joinpath(@__DIR__, "..", "src", "TamerOp.jl"))
    using .TamerOp
end
using SparseArrays: nnz

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
        return parse(Int, split(a, "=", limit=2)[2])
    end
    return default
end

function _parse_string_arg(args, key::String, default::String)
    prefix = key * "="
    for a in args
        startswith(a, prefix) || continue
        return String(split(a, "=", limit=2)[2])
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

function _bench_ab(name::String, before::Function, after::Function; reps::Int)
    b = _median_stats(before; reps=reps)
    a = _median_stats(after; reps=reps)
    speedup = a.ms == 0.0 ? Inf : b.ms / a.ms
    alloc_ratio = b.kib == 0.0 ? Inf : a.kib / b.kib
    println(rpad(name, 38),
            " before=", round(b.ms, digits=3), " ms / ", round(b.kib, digits=1), " KiB",
            "  after=", round(a.ms, digits=3), " ms / ", round(a.kib, digits=1), " KiB",
            "  speedup=", round(speedup, digits=2), "x",
            "  alloc_ratio=", round(alloc_ratio, digits=2), "x")
    return (probe=name, before_ms=b.ms, before_kib=b.kib,
            after_ms=a.ms, after_kib=a.kib, speedup=speedup, alloc_ratio=alloc_ratio)
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

function _chain_poset(n::Int)
    rel = falses(n, n)
    @inbounds for i in 1:n, j in i:n
        rel[i, j] = true
    end
    return FF.FinitePoset(rel; check=false)
end

function _simple_module(P, v, field)
    K = CM.coeff_type(field)
    H = FF.one_by_one_fringe(P, FF.principal_upset(P, v), FF.principal_downset(P, v), one(K); field=field)
    return IR.pmodule_from_fringe(H)
end

function _direct_sum_with_split_sequence(A::MD.PModule{K}, C::MD.PModule{K}) where {K}
    Q = A.Q
    field = A.field
    dimsB = [A.dims[v] + C.dims[v] for v in 1:Q.n]
    edge_mapsB = Dict{Tuple{Int,Int}, Matrix{K}}()
    for (i, j) in FF.cover_edges(Q)
        Aij = A.edge_maps[i, j]
        Cij = C.edge_maps[i, j]
        top = hcat(Aij, CM.zeros(field, size(Aij, 1), size(Cij, 2)))
        bottom = hcat(CM.zeros(field, size(Cij, 1), size(Aij, 2)), Cij)
        edge_mapsB[(i, j)] = vcat(top, bottom)
    end
    B = MD.PModule{K}(Q, dimsB, edge_mapsB; field=field)
    comps_i = Vector{Matrix{K}}(undef, Q.n)
    comps_p = Vector{Matrix{K}}(undef, Q.n)
    for v in 1:Q.n
        comps_i[v] = vcat(CM.eye(field, A.dims[v]),
                          CM.zeros(field, C.dims[v], A.dims[v]))
        comps_p[v] = hcat(CM.zeros(field, C.dims[v], A.dims[v]),
                          CM.eye(field, C.dims[v]))
    end
    return B, MD.PMorphism(A, B, comps_i), MD.PMorphism(B, C, comps_p)
end

function _digest_sparse_family(F)
    total = 0
    @inbounds for A in F
        total += size(A, 1) + size(A, 2) + nnz(A)
    end
    return total
end

function _fixture(; n::Int=8)
    field = CM.QQField()
    P = _chain_poset(n)
    Sm = [_simple_module(P, v, field) for v in 1:n]
    pick(i) = Sm[clamp(i, 1, n)]
    S1 = Sm[1]
    S3 = pick(max(3, n - 4))
    S4 = pick(max(4, n - 3))
    S6 = pick(max(6, n - 1))
    S8 = Sm[end]

    B2, i2, p2 = _direct_sum_with_split_sequence(S8, S4)
    resM = DF.projective_resolution(S1, OPT.ResolutionOptions(maxlen=4))
    EMA = DF.Ext(resM, S8)
    EMB = DF.Ext(resM, B2)
    EMC = DF.Ext(resM, S4)

    B1, i1, p1 = _direct_sum_with_split_sequence(S4, S3)
    A1 = S4
    C1 = S3
    resN = DF.injective_resolution(S8, OPT.ResolutionOptions(maxlen=3))
    EA = DF.ExtInjective(S4, resN)
    EB = DF.ExtInjective(B1, resN)
    EC = DF.ExtInjective(S3, resN)

    resB = DF.projective_resolution(B2, OPT.ResolutionOptions(maxlen=4))

    return (
        field=field,
        S1=S1,
        S3=S3,
        S4=S4,
        S8=S8,
        B2=B2,
        i2=i2,
        p2=p2,
        A1=A1,
        B1=B1,
        C1=C1,
        EMA=EMA,
        EMB=EMB,
        EMC=EMC,
        EA=EA,
        EB=EB,
        EC=EC,
        i1=i1,
        p1=p1,
        resB=resB,
    )
end

function main(; reps::Int=5,
              n::Int=8,
              out::String=joinpath(@__DIR__, "_tmp_derived_functors_lift_les_microbench.csv"))
    fx = _fixture(n=n)
    rows = NamedTuple[]

    old_support = DF.Functoriality._FUNCTORIALITY_USE_SUPPORT_SOLVE_PLANS[]
    old_workspaces = DF.Functoriality._FUNCTORIALITY_USE_HOM_WORKSPACES[]
    old_inj_downset_cache = DF.Resolutions._RESOLUTIONS_USE_INJECTIVE_DOWNSET_SYSTEM_CACHE[]
    old_inj_downset_solve_plan = DF.Resolutions._RESOLUTIONS_USE_INJECTIVE_DOWNSET_SOLVE_PLAN_CACHE[]
    old_hom_solve_cache = DF.Functoriality._FUNCTORIALITY_USE_HOM_SOLVE_WORKSPACE_CACHE[]
    old_hom_basis_plan = DF.Functoriality._FUNCTORIALITY_USE_HOM_BASIS_SOLVE_PLAN_CACHE[]
    old_proj_lift_cache = DF.Resolutions._RESOLUTIONS_USE_PROJECTIVE_LIFT_STRUCTURE_CACHE[]

    println("DerivedFunctors lift/LES microbenchmark")
    println("timing_policy=warm_process_median reps=", reps, " n=", n, " threads=", Threads.nthreads())

    try
        push!(rows, _bench_ab("lift_chainmap_coeff",
            () -> begin
                DF.Functoriality._FUNCTORIALITY_USE_SUPPORT_SOLVE_PLANS[] = false
                DF.Resolutions._RESOLUTIONS_USE_PROJECTIVE_LIFT_STRUCTURE_CACHE[] = false
                _digest_sparse_family(DF.Functoriality._lift_pmodule_map_to_projective_resolution_chainmap_coeff(
                    fx.resB, fx.resB, IR.id_morphism(fx.B2); upto=3
                ))
            end,
            () -> begin
                DF.Functoriality._FUNCTORIALITY_USE_SUPPORT_SOLVE_PLANS[] = true
                DF.Resolutions._RESOLUTIONS_USE_PROJECTIVE_LIFT_STRUCTURE_CACHE[] = true
                _digest_sparse_family(DF.Functoriality._lift_pmodule_map_to_projective_resolution_chainmap_coeff(
                    fx.resB, fx.resB, IR.id_morphism(fx.B2); upto=3
                ))
            end; reps=reps))

        push!(rows, _bench_ab("connecting_hom_second",
            () -> begin
                DF.Functoriality._FUNCTORIALITY_USE_SUPPORT_SOLVE_PLANS[] = false
                DF.Resolutions._RESOLUTIONS_USE_INJECTIVE_DOWNSET_SYSTEM_CACHE[] = false
                DF.Resolutions._RESOLUTIONS_USE_INJECTIVE_DOWNSET_SOLVE_PLAN_CACHE[] = false
                sum(abs, DF.connecting_hom(fx.EMA, fx.EMB, fx.EMC, fx.i2, fx.p2; t=2))
            end,
            () -> begin
                DF.Functoriality._FUNCTORIALITY_USE_SUPPORT_SOLVE_PLANS[] = true
                DF.Resolutions._RESOLUTIONS_USE_INJECTIVE_DOWNSET_SYSTEM_CACHE[] = true
                DF.Resolutions._RESOLUTIONS_USE_INJECTIVE_DOWNSET_SOLVE_PLAN_CACHE[] = true
                sum(abs, DF.connecting_hom(fx.EMA, fx.EMB, fx.EMC, fx.i2, fx.p2; t=2))
            end; reps=reps))

        push!(rows, _bench_ab("connecting_hom_first",
            () -> begin
                DF.Functoriality._FUNCTORIALITY_USE_SUPPORT_SOLVE_PLANS[] = false
                DF.Functoriality._FUNCTORIALITY_USE_HOM_WORKSPACES[] = false
                DF.Resolutions._RESOLUTIONS_USE_INJECTIVE_DOWNSET_SYSTEM_CACHE[] = false
                DF.Resolutions._RESOLUTIONS_USE_INJECTIVE_DOWNSET_SOLVE_PLAN_CACHE[] = false
                DF.Functoriality._FUNCTORIALITY_USE_HOM_SOLVE_WORKSPACE_CACHE[] = false
                DF.Functoriality._FUNCTORIALITY_USE_HOM_BASIS_SOLVE_PLAN_CACHE[] = false
                sum(abs, DF.connecting_hom_first(fx.EA, fx.EB, fx.EC, fx.i1, fx.p1; t=1))
            end,
            () -> begin
                DF.Functoriality._FUNCTORIALITY_USE_SUPPORT_SOLVE_PLANS[] = true
                DF.Functoriality._FUNCTORIALITY_USE_HOM_WORKSPACES[] = true
                DF.Resolutions._RESOLUTIONS_USE_INJECTIVE_DOWNSET_SYSTEM_CACHE[] = true
                DF.Resolutions._RESOLUTIONS_USE_INJECTIVE_DOWNSET_SOLVE_PLAN_CACHE[] = true
                DF.Functoriality._FUNCTORIALITY_USE_HOM_SOLVE_WORKSPACE_CACHE[] = true
                DF.Functoriality._FUNCTORIALITY_USE_HOM_BASIS_SOLVE_PLAN_CACHE[] = true
                sum(abs, DF.connecting_hom_first(fx.EA, fx.EB, fx.EC, fx.i1, fx.p1; t=1))
            end; reps=reps))

        push!(rows, _bench_ab("ext_map_first_injective",
            () -> begin
                DF.Functoriality._FUNCTORIALITY_USE_HOM_WORKSPACES[] = false
                DF.Functoriality._FUNCTORIALITY_USE_HOM_SOLVE_WORKSPACE_CACHE[] = false
                DF.Functoriality._FUNCTORIALITY_USE_HOM_BASIS_SOLVE_PLAN_CACHE[] = false
                sum(abs, DF.ext_map_first(fx.EA, fx.EB, fx.i1; t=1))
            end,
            () -> begin
                DF.Functoriality._FUNCTORIALITY_USE_HOM_WORKSPACES[] = true
                DF.Functoriality._FUNCTORIALITY_USE_HOM_SOLVE_WORKSPACE_CACHE[] = true
                DF.Functoriality._FUNCTORIALITY_USE_HOM_BASIS_SOLVE_PLAN_CACHE[] = true
                sum(abs, DF.ext_map_first(fx.EA, fx.EB, fx.i1; t=1))
            end; reps=reps))

        push!(rows, _bench_ab("ext_map_second_injective",
            () -> begin
                DF.Functoriality._FUNCTORIALITY_USE_HOM_WORKSPACES[] = false
                DF.Functoriality._FUNCTORIALITY_USE_HOM_SOLVE_WORKSPACE_CACHE[] = false
                DF.Resolutions._RESOLUTIONS_USE_INJECTIVE_DOWNSET_SYSTEM_CACHE[] = false
                DF.Resolutions._RESOLUTIONS_USE_INJECTIVE_DOWNSET_SOLVE_PLAN_CACHE[] = false
                DF.Functoriality._FUNCTORIALITY_USE_HOM_BASIS_SOLVE_PLAN_CACHE[] = false
                sum(abs, DF.ext_map_second(fx.EC, fx.EC, IR.id_morphism(fx.S8); t=1))
            end,
            () -> begin
                DF.Functoriality._FUNCTORIALITY_USE_HOM_WORKSPACES[] = true
                DF.Functoriality._FUNCTORIALITY_USE_HOM_SOLVE_WORKSPACE_CACHE[] = true
                DF.Resolutions._RESOLUTIONS_USE_INJECTIVE_DOWNSET_SYSTEM_CACHE[] = true
                DF.Resolutions._RESOLUTIONS_USE_INJECTIVE_DOWNSET_SOLVE_PLAN_CACHE[] = true
                DF.Functoriality._FUNCTORIALITY_USE_HOM_BASIS_SOLVE_PLAN_CACHE[] = true
                sum(abs, DF.ext_map_second(fx.EC, fx.EC, IR.id_morphism(fx.S8); t=1))
            end; reps=reps))

        _write_csv(out, rows)
        println("wrote ", out)
        return rows
    finally
        DF.Functoriality._FUNCTORIALITY_USE_SUPPORT_SOLVE_PLANS[] = old_support
        DF.Functoriality._FUNCTORIALITY_USE_HOM_WORKSPACES[] = old_workspaces
        DF.Resolutions._RESOLUTIONS_USE_INJECTIVE_DOWNSET_SYSTEM_CACHE[] = old_inj_downset_cache
        DF.Resolutions._RESOLUTIONS_USE_INJECTIVE_DOWNSET_SOLVE_PLAN_CACHE[] = old_inj_downset_solve_plan
        DF.Functoriality._FUNCTORIALITY_USE_HOM_SOLVE_WORKSPACE_CACHE[] = old_hom_solve_cache
        DF.Functoriality._FUNCTORIALITY_USE_HOM_BASIS_SOLVE_PLAN_CACHE[] = old_hom_basis_plan
        DF.Resolutions._RESOLUTIONS_USE_PROJECTIVE_LIFT_STRUCTURE_CACHE[] = old_proj_lift_cache
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(reps=_parse_int_arg(ARGS, "--reps", 5),
         n=_parse_int_arg(ARGS, "--n", 8),
         out=_parse_string_arg(ARGS, "--out",
                               joinpath(@__DIR__, "_tmp_derived_functors_lift_les_microbench.csv")))
end
