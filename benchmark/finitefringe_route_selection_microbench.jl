#!/usr/bin/env julia
# FiniteFringe route-selection microbenchmarks.
# This script compares the legacy sparse-Hom route policy against the current
# structural predictor on fresh module pairs. It records explicit candidate path
# timings for the surviving internal routes, the legacy auto-policy latency,
# and the current auto-policy latency.

using Random
using SparseArrays
using Statistics

try
    using TamerOp
catch
    include(joinpath(@__DIR__, "..", "src", "TamerOp.jl"))
    using .TamerOp
end

const FF = TamerOp.FiniteFringe
const CM = TamerOp.CoreModules

function _parse_int_arg(args, key::String, default::Int)
    for arg in args
        startswith(arg, key * "=") || continue
        return parse(Int, split(arg, "=", limit=2)[2])
    end
    return default
end

function _random_poset(n::Int; p::Float64=0.035, seed::Int=0xAA01)
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

function _random_fringe_data(P::FF.AbstractPoset, field;
                             nu::Int, nd::Int, density::Float64, seed::Int)
    rng = Random.MersenneTwister(seed)
    K = CM.coeff_type(field)
    n = FF.nvertices(P)
    U = [copy(FF.upset_closure(P, BitVector(rand(rng, Bool, n))).mask) for _ in 1:nu]
    D = [copy(FF.downset_closure(P, BitVector(rand(rng, Bool, n))).mask) for _ in 1:nd]
    phi = spzeros(K, nd, nu)
    @inbounds for j in 1:nd, i in 1:nu
        any(U[i] .& D[j]) || continue
        rand(rng) < density || continue
        v = rand(rng, -3:3)
        v == 0 && continue
        phi[j, i] = CM.coerce(field, v)
    end
    return U, D, phi
end

function _make_module(P, field, Ubits, Dbits, phi)
    K = CM.coeff_type(field)
    U = [FF.Upset(P, copy(bits)) for bits in Ubits]
    D = [FF.Downset(P, copy(bits)) for bits in Dbits]
    return FF.FringeModule{K}(P, U, D, copy(phi); field=field)
end

@inline function _legacy_hom_work(nu::Int, nd::Int)
    return nu * nd + nu * nu + nd * nd
end

function _legacy_choice(M, N)
    dmin = min(M.phi_density, N.phi_density)
    dmax = max(M.phi_density, N.phi_density)
    work = _legacy_hom_work(length(M.U), length(M.D))
    tiny_work = 350
    sparse_dens = 0.24
    sparse_work = 80_000
    dense_work = 8_000
    work_band = 500
    dens_band = 0.020

    work <= tiny_work && return :dense_idx_internal
    if dmin <= sparse_dens - dens_band && work <= sparse_work + work_band
        return :sparse_path
    end
    if dmin >= sparse_dens + dens_band
        if work <= dense_work - work_band
            return :dense_idx_internal
        end
        return nothing
    end
    if work <= tiny_work + work_band
        return :dense_idx_internal
    elseif dmax <= sparse_dens && work <= sparse_work + work_band
        return :sparse_path
    end
    return nothing
end

function _median_ms(f::Function; reps::Int)
    GC.gc()
    f()
    GC.gc()
    vals = Vector{Float64}(undef, reps)
    for i in 1:reps
        vals[i] = 1000 * (@timed f()).time
    end
    sort!(vals)
    return vals[cld(reps, 2)]
end

function _candidate_times(P, field, U_M, D_M, phi_M, U_N, D_N, phi_N; reps::Int)
    times = Dict{Symbol,Float64}()
    vals = Dict{Symbol,Int}()
    for path in (:dense_idx_internal, :sparse_path)
        function run_path()
            M = _make_module(P, field, U_M, D_M, phi_M)
            N = _make_module(P, field, U_N, D_N, phi_N)
            return FF._hom_dimension_with_path(M, N, path)
        end
        vals[path] = run_path()
        times[path] = _median_ms(run_path; reps=reps)
    end
    length(unique(values(vals))) == 1 || error("route candidate mismatch")
    return times, first(values(vals))
end

function _current_auto(P, field, U_M, D_M, phi_M, U_N, D_N, phi_N; reps::Int)
    choices = Symbol[]
    function run_auto()
        M = _make_module(P, field, U_M, D_M, phi_M)
        N = _make_module(P, field, U_N, D_N, phi_N)
        v = FF.hom_dimension(M, N)
        push!(choices, FF._lookup_pair_cache(M.hom_cache[], N).route_choice)
        return v
    end
    val = run_auto()
    ms = _median_ms(run_auto; reps=reps)
    return ms, choices[end], val
end

function main(args=ARGS)
    reps = _parse_int_arg(args, "--reps", 1)
    field = CM.field_from_eltype(CM.QQ)
    outpath = joinpath(@__DIR__, "_tmp_finitefringe_route_selection.csv")

    # Warm method compilation before measuring route-policy latency.
    P_w = _random_poset(12; seed=Int(0xBEEF))
    U_w, D_w, phi_w = _random_fringe_data(P_w, field; nu=2, nd=2, density=0.3, seed=Int(0xCAFE))
    M_w = _make_module(P_w, field, U_w, D_w, phi_w)
    N_w = _make_module(P_w, field, U_w, D_w, phi_w)
    for path in (:dense_idx_internal, :sparse_path)
        FF._hom_dimension_with_path(M_w, N_w, path)
    end
    FF.hom_dimension(M_w, N_w)

    cases = [
        ("tiny_dense",         2,  2, 0.80, false),
        ("small_sparse",       8,  8, 0.18, false),
        ("medium_sparse",     12, 12, 0.12, false),
        ("medium_dense_store",12, 12, 0.20, true),
        ("larger_sparse",     12, 12, 0.30, false),
    ]

    open(outpath, "w") do io
        println(io, "label,nu,nd,dens,storage,W_dim,V1_dim,V2_dim,work_est,comp_total,dmin,legacy_choice,current_choice,legacy_auto_ms,current_auto_ms,di_ms,sp_ms")
        for (idx, (label, nu, nd, dens, dense_store)) in enumerate(cases)
            P = _random_poset(18; seed=0x8000 + idx)
            U_M, D_M, phi_M0 = _random_fringe_data(P, field; nu=nu, nd=nd, density=dens, seed=0x9000 + idx)
            U_N, D_N, phi_N0 = _random_fringe_data(P, field; nu=nu, nd=nd, density=dens, seed=0xA000 + idx)
            phi_M = dense_store ? Matrix(phi_M0) : phi_M0
            phi_N = dense_store ? Matrix(phi_N0) : phi_N0
            M0 = _make_module(P, field, U_M, D_M, phi_M)
            N0 = _make_module(P, field, U_N, D_N, phi_N)
            plan = FF._ensure_hom_layout_plan!(M0, N0)
            feat = FF._hom_route_features(M0, N0, plan)
            times, _ = _candidate_times(P, field, U_M, D_M, phi_M, U_N, D_N, phi_N; reps=reps)
            legacy_choice = _legacy_choice(M0, N0)
            if legacy_choice === nothing
                legacy_choice = (:dense_idx_internal, :sparse_path)[argmin((times[:dense_idx_internal], times[:sparse_path]))]
                legacy_auto_ms = times[:dense_idx_internal] + times[:sparse_path]
            else
                legacy_auto_ms = times[legacy_choice]
            end
            current_auto_ms, current_choice, _ = _current_auto(P, field, U_M, D_M, phi_M, U_N, D_N, phi_N; reps=reps)
            println(io, join((label, nu, nd, dens, dense_store ? "dense" : "sparse",
                              feat.W_dim, plan.sketch.V1_dim, plan.sketch.V2_dim,
                              feat.work_est, feat.comp_total, round(feat.dmin, digits=3),
                              legacy_choice, current_choice,
                              round(legacy_auto_ms, digits=3), round(current_auto_ms, digits=3),
                              round(times[:dense_idx_internal], digits=3), round(times[:sparse_path], digits=3)), ','))
            flush(io)
        end
    end

    println("wrote ", outpath)
end

main()
