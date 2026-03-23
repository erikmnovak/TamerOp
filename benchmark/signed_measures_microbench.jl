#!/usr/bin/env julia
# signed_measures_microbench.jl
#
# Purpose
# - Benchmark the current hot surfaces in `SignedMeasures.jl`.
# - Keep the benchmark sectional so point-measure, Euler, and rectangle-barcode
#   paths can be profiled independently.
#
# Benchmark-to-kernel mapping
# - `point` -> point signed-measure inversion, reconstruction, truncation, kernels
# - `euler` -> Euler surface, Euler signed measure, Euler distance
# - `rect`    -> rectangle signed barcode construction, cache reuse, rank reconstruction
# - `extract` -> isolated region-grid / fill / inversion / extraction lanes
# - `image` -> rectangle signed-barcode kernels and image vectorization
#
# Timing policy
# - Warm-process microbenchmarking (`@timed` median over reps)
# - Probe names encode cache state explicitly when relevant
#
# Usage
#   julia --project=. benchmark/signed_measures_microbench.jl
#   julia --project=. benchmark/signed_measures_microbench.jl --section=rect --reps=3
#   julia --project=. benchmark/signed_measures_microbench.jl --profile=larger --field=f3

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)

using Random
using Statistics

function _load_posetmodules_for_signed_bench()
    try
        @eval using TamerOp
        return :package
    catch
    end

    # Last-resort fallback when unrelated owner modules break root loading.
    @eval module TamerOp
        include(joinpath(@__DIR__, "..", "src", "CoreModules.jl"))
        include(joinpath(@__DIR__, "..", "src", "Stats.jl"))
        include(joinpath(@__DIR__, "..", "src", "Options.jl"))
        include(joinpath(@__DIR__, "..", "src", "DataTypes.jl"))
        include(joinpath(@__DIR__, "..", "src", "EncodingCore.jl"))
        include(joinpath(@__DIR__, "..", "src", "Results.jl"))
        module RegionGeometry
            let names = (
                :region_weights, :region_volume, :region_bbox, :region_widths,
                :region_centroid, :region_aspect_ratio, :region_diameter,
                :region_adjacency, :region_facet_count, :region_vertex_count,
                :region_boundary_measure, :region_boundary_measure_breakdown,
                :region_perimeter, :region_surface_area,
                :region_principal_directions,
                :region_chebyshev_ball, :region_chebyshev_center, :region_inradius,
                :region_circumradius,
                :region_boundary_to_volume_ratio, :region_isoperimetric_ratio,
                :region_mean_width, :region_minkowski_functionals,
                :region_covariance_anisotropy, :region_covariance_eccentricity, :region_anisotropy_scores,
            )
                for sym in names
                    Core.eval(@__MODULE__, :($(sym)(args...; kwargs...) = error("RegionGeometry stub unavailable in signed benchmark bootstrap")))
                end
            end
        end
        include(joinpath(@__DIR__, "..", "src", "FieldLinAlg.jl"))
        include(joinpath(@__DIR__, "..", "src", "FiniteFringe.jl"))
        include(joinpath(@__DIR__, "..", "src", "IndicatorTypes.jl"))
        include(joinpath(@__DIR__, "..", "src", "Encoding.jl"))
        include(joinpath(@__DIR__, "..", "src", "Modules.jl"))
        include(joinpath(@__DIR__, "..", "src", "AbelianCategories.jl"))
        include(joinpath(@__DIR__, "..", "src", "IndicatorResolutions.jl"))
        include(joinpath(@__DIR__, "..", "src", "FlangeZn.jl"))
        include(joinpath(@__DIR__, "..", "src", "ZnEncoding.jl"))
        module PLPolyhedra end
        module ModuleComplexes
            export ModuleCochainComplex
            struct ModuleCochainComplex{K}
                terms::Vector{Any}
                tmin::Int
            end
        end
        module ChangeOfPosets
            export pushforward_left, pushforward_right
            pushforward_left(args...; kwargs...) = error("ChangeOfPosets not available in signed benchmark bootstrap")
            pushforward_right(args...; kwargs...) = error("ChangeOfPosets not available in signed benchmark bootstrap")
        end
        module Serialization
            export save_mpp_decomposition_json, load_mpp_decomposition_json,
                   save_mpp_image_json, load_mpp_image_json
            save_mpp_decomposition_json(args...; kwargs...) = error("Serialization stub unavailable in signed benchmark bootstrap")
            load_mpp_decomposition_json(args...; kwargs...) = error("Serialization stub unavailable in signed benchmark bootstrap")
            save_mpp_image_json(args...; kwargs...) = error("Serialization stub unavailable in signed benchmark bootstrap")
            load_mpp_image_json(args...; kwargs...) = error("Serialization stub unavailable in signed benchmark bootstrap")
        end
        include(joinpath(@__DIR__, "..", "src", "InvariantCore.jl"))
        include(joinpath(@__DIR__, "..", "src", "SignedMeasures.jl"))

        DataTypes.nvertices(P::FiniteFringe.AbstractPoset) = FiniteFringe.nvertices(P)

        module Advanced
            const CoreModules = Main.TamerOp.CoreModules
            const Options = Main.TamerOp.Options
            const EncodingCore = Main.TamerOp.EncodingCore
            const FiniteFringe = Main.TamerOp.FiniteFringe
            const Modules = Main.TamerOp.Modules
            const IndicatorResolutions = Main.TamerOp.IndicatorResolutions
            const InvariantCore = Main.TamerOp.InvariantCore
            const SignedMeasures = Main.TamerOp.SignedMeasures
        end
    end
    @eval using .TamerOp
    return :source_signed_only
end

const _PM_LOAD_MODE = _load_posetmodules_for_signed_bench()
const TO = isdefined(TamerOp, :Advanced) ? TamerOp.Advanced : TamerOp
const CM = TO.CoreModules
const OPT = TO.Options
const FF = TO.FiniteFringe
const MD = TO.Modules
const IR = TO.IndicatorResolutions
const IC = TO.InvariantCore
const SM = TO.SignedMeasures
const FZ = TamerOp.FlangeZn
const ZE = TamerOp.ZnEncoding

function _parse_int_arg(args, key::String, default::Int)
    for a in args
        s = startswith(a, "--") ? a[3:end] : a
        startswith(s, key * "=") || continue
        return max(1, parse(Int, split(s, "=", limit=2)[2]))
    end
    return default
end

function _parse_string_arg(args, key::String, default::String)
    for a in args
        s = startswith(a, "--") ? a[3:end] : a
        startswith(s, key * "=") || continue
        return lowercase(strip(split(s, "=", limit=2)[2]))
    end
    return default
end

function _parse_bool_arg(args, key::String, default::Bool)
    for a in args
        s = startswith(a, "--") ? a[3:end] : a
        startswith(s, key * "=") || continue
        v = lowercase(strip(split(s, "=", limit=2)[2]))
        v in ("true", "1", "yes", "on") && return true
        v in ("false", "0", "no", "off") && return false
        error("invalid boolean for $key: $v")
    end
    return default
end

function _parse_path_arg(args, key::String, default::String)
    for a in args
        s = startswith(a, "--") ? a[3:end] : a
        startswith(s, key * "=") || continue
        return String(strip(split(s, "=", limit=2)[2]))
    end
    return default
end

_section_enabled(section::String, group::String) = section == "all" || section == group

function _field_from_name(name::String)
    name == "qq" && return CM.QQField()
    name == "f2" && return CM.F2()
    name == "f3" && return CM.F3()
    name == "f5" && return CM.Fp(5)
    error("unknown field '$name' (supported: qq, f2, f3, f5)")
end

function _profile_defaults(profile::String)
    if profile == "default"
        return (
            m23_copies = 3,
            m3_copies = 2,
            ncuts = 4,
            img_n = 21,
            trunc_terms = 6,
        )
    elseif profile == "larger"
        return (
            m23_copies = 6,
            m3_copies = 4,
            ncuts = 8,
            img_n = 31,
            trunc_terms = 10,
        )
    else
        error("unknown profile '$profile' (supported: default, larger)")
    end
end

function _bench(name::AbstractString, f::Function; reps::Int=7, setup::Union{Nothing,Function}=nothing)
    GC.gc()
    setup === nothing || setup()
    f()
    GC.gc()
    times_ms = Vector{Float64}(undef, reps)
    alloc_kib = Vector{Float64}(undef, reps)
    for i in 1:reps
        setup === nothing || setup()
        m = @timed f()
        times_ms[i] = 1000.0 * m.time
        alloc_kib[i] = m.bytes / 1024.0
    end
    sort!(times_ms)
    sort!(alloc_kib)
    mid = cld(reps, 2)
    row = (probe=String(name), median_ms=times_ms[mid], median_kib=alloc_kib[mid])
    println(rpad(row.probe, 52),
            " median_time=", round(row.median_ms, digits=3), " ms",
            " median_alloc=", round(row.median_kib, digits=1), " KiB")
    return row
end

_bench(f::Function, name::AbstractString; reps::Int=7, setup::Union{Nothing,Function}=nothing) =
    _bench(name, f; reps=reps, setup=setup)

function _write_csv(path::AbstractString, rows)
    open(path, "w") do io
        println(io, "probe,median_ms,median_kib")
        for r in rows
            println(io, string(r.probe, ",", r.median_ms, ",", r.median_kib))
        end
    end
end

function _repeat_direct_sum(ms::Vector{<:MD.PModule})
    isempty(ms) && error("_repeat_direct_sum: empty module list")
    cur = ms[1]
    for m in ms[2:end]
        cur = MD.direct_sum(cur, m)
    end
    return cur
end

function _mobius_tensor_from_signed_barcode(sb::SM.RectSignedBarcode{N,<:Integer}) where {N}
    dims = ntuple(k -> length(sb.axes[k]), N)
    w = zeros(Int, (dims..., dims...))
    axis_to_idx = ntuple(k -> Dict(v => i for (i, v) in pairs(sb.axes[k])), N)
    @inbounds for (rect, wt) in zip(sb.rects, sb.weights)
        p = ntuple(k -> axis_to_idx[k][rect.lo[k]], N)
        q = ntuple(k -> axis_to_idx[k][rect.hi[k]], N)
        w[p..., q...] += Int(wt)
    end
    return w
end

function _axis_encoding_flange(field::CM.AbstractCoeffField; ncuts::Int=8, seed::Integer=0x5352454e)
    N = 2
    tau_x = FZ.face(N, [false, true])
    tau_y = FZ.face(N, [true, false])
    thresholds = collect(-ncuts:ncuts)
    flats = FZ.IndFlat{N}[]
    injectives = FZ.IndInj{N}[]
    for (i, t) in enumerate(thresholds)
        push!(flats, FZ.IndFlat(tau_x, (t, 0); id=Symbol(:Fx, i)))
        push!(injectives, FZ.IndInj(tau_x, (t + 1, 0); id=Symbol(:Ex, i)))
        push!(flats, FZ.IndFlat(tau_y, (0, t); id=Symbol(:Fy, i)))
        push!(injectives, FZ.IndInj(tau_y, (0, t + 1); id=Symbol(:Ey, i)))
    end
    m = length(injectives)
    K = CM.coeff_type(field)
    phi = Matrix{K}(undef, m, m)
    rng = Random.MersenneTwister(UInt(seed))
    @inbounds for i in 1:m, j in 1:m
        phi[i, j] = CM.coerce(field, i == j ? 1 : rand(rng, -1:1))
    end
    return FZ.Flange{K}(N, flats, injectives, phi; field=field)
end

function _build_fixture(field::CM.AbstractCoeffField, profile::String; threads::Bool, section::String)
    cfg = _profile_defaults(profile)
    fgM = _axis_encoding_flange(field; ncuts=cfg.ncuts, seed=0x5352454e)
    fgN = _axis_encoding_flange(field; ncuts=cfg.ncuts, seed=0x53524550)
    enc_opts = OPT.EncodingOptions(backend=:zn, max_regions=500_000, field=field)
    P, Hs, pi = ZE.encode_from_flanges((fgM, fgN), enc_opts; poset_kind=:signature)
    M = IR.pmodule_from_fringe(Hs[1])
    N = IR.pmodule_from_fringe(Hs[2])

    opts = OPT.InvariantOptions(threads=threads)
    need_point = section in ("point", "all")
    need_euler = section in ("euler", "all")
    need_rect_hot = section in ("rect", "image", "all")
    need_extract = section in ("extract", "all")
    need_rect = need_rect_hot || need_extract
    need_image = section in ("image", "all")

    surfM = (need_point || need_euler) ? SM.euler_surface(M, pi, opts) : nothing
    surfN = (need_point || need_euler) ? SM.euler_surface(N, pi, opts) : nothing
    surf_axes = need_point ? ZE.axes_from_encoding(pi) : nothing
    pmM = need_point ? SM.point_signed_measure(surfM, surf_axes; drop_zeros=true) : nothing
    pmN = need_point ? SM.point_signed_measure(surfN, surf_axes; drop_zeros=true) : nothing
    pmM_trunc = need_point ? SM.truncate_point_signed_measure(pmM; max_terms=cfg.trunc_terms, min_abs_weight=1) : nothing

    rq_cache = need_rect ? IC.RankQueryCache(pi) : nothing
    cc = if need_rect
        FF.build_cache!(M.Q; cover=true, updown=false)
        FF._get_cover_cache(M.Q)
    else
        nothing
    end
    rank_ab = if need_rect
        let rq_cache = rq_cache, M = M, cc = cc
            (a::Int, b::Int) -> IC._rank_cache_get!(rq_cache, a, b, () -> IC.rank_map(M, a, b; cache=cc))
        end
    else
        nothing
    end
    sc_rect = need_rect ? CM.SessionCache() : nothing
    sc_euler = need_euler ? CM.SessionCache() : nothing
    sc_mma = (need_rect || need_euler) ? CM.SessionCache() : nothing
    axes_rect = need_rect ? ZE.axes_from_encoding(pi) : nothing
    sb_auto = need_rect_hot ? SM.rectangle_signed_barcode(M, pi, opts) : nothing
    sb_cached = need_rect_hot ? SM.rectangle_signed_barcode(M, pi, opts; rq_cache=rq_cache) : nothing
    sb_local = need_rect_hot ? SM.rectangle_signed_barcode(M, pi, opts; method=:local) : nothing
    sb_session = need_rect_hot ? SM.rectangle_signed_barcode(M, pi, opts; cache=sc_rect) : nothing
    sbN = need_image ? SM.rectangle_signed_barcode(N, pi, opts) : nothing
    sb_rank = need_rect_hot ? SM.rectangle_signed_barcode_rank(sb_cached) : nothing
    if need_extract
        SM._cached_rectangle_region_blocks_2d(pi, axes_rect, rq_cache, sc_rect; strict=false)
    end
    reg_full = need_rect ? SM._rectangle_region_grid(pi, axes_rect, rq_cache; strict=false) : nothing
    reg_comp, keep1, keep2, axes_comp = if need_rect
        SM._compress_region_grid_2d(reg_full, axes_rect)
    else
        (nothing, nothing, nothing, nothing)
    end
    r_dense_fill = need_rect ? SM._fill_rectangle_rank_tensor_dense_from_regions_2d(reg_full, rank_ab; threads=threads) : nothing
    r_packed_fill = need_rect ? SM._fill_rectangle_rank_tensor_packed_from_regions_2d(reg_comp, rank_ab; threads=threads) : nothing
    r_dense_inv = if need_rect
        r = copy(r_dense_fill)
        SM._mobius_inversion_interval_product!(r, 2)
        r
    else
        nothing
    end
    r_packed_inv = if need_rect
        r = copy(r_packed_fill)
        SM._mobius_inversion_interval_product_packed_2d!(r, size(reg_comp, 1), size(reg_comp, 2); threads=threads)
        r
    else
        nothing
    end
    p = need_rect_hot ? ntuple(i -> first(sb_cached.axes[i]), length(sb_cached.axes)) : nothing
    q = need_rect_hot ? ntuple(i -> last(sb_cached.axes[i]), length(sb_cached.axes)) : nothing
    xs = need_image ? collect(range(first(sb_cached.axes[1]), last(sb_cached.axes[1]), length=cfg.img_n)) : nothing
    ys = need_image ? collect(range(first(sb_cached.axes[2]), last(sb_cached.axes[2]), length=cfg.img_n)) : nothing
    axes_skew = need_rect ? (collect(0:8), collect(0:1)) : nothing
    sb_skew_true = need_rect ? SM.RectSignedBarcode{2,Int}(
        axes_skew,
        SM.Rect{2}[SM.Rect{2}((0, 0), (8, 1)), SM.Rect{2}((2, 0), (5, 0))],
        Int[1, -1],
    ) : nothing
    r_idx_skew = if need_rect
        let axes_skew = axes_skew, sb_skew_true = sb_skew_true
            (p, q) -> begin
                x = (axes_skew[1][p[1]], axes_skew[2][p[2]])
                y = (axes_skew[1][q[1]], axes_skew[2][q[2]])
                SM.rank_from_signed_barcode(sb_skew_true, x, y)
            end
        end
    else
        nothing
    end
    need_euler && SM.euler_signed_measure(M, pi, opts; cache=sc_euler)
    (need_rect || need_euler) && SM.mma_decomposition(M, pi, opts; method=:euler, cache=sc_mma)

    return (
        cfg = cfg,
        P = P,
        pi = pi,
        opts = opts,
        M = M,
        N = N,
        surfM = surfM,
        surfN = surfN,
        surf_axes = surf_axes,
        pmM = pmM,
        pmN = pmN,
        pmM_trunc = pmM_trunc,
        rq_cache = rq_cache,
        sc_rect = sc_rect,
        sc_euler = sc_euler,
        sc_mma = sc_mma,
        sb_auto = sb_auto,
        sb_cached = sb_cached,
        sb_local = sb_local,
        sb_session = sb_session,
        sbN = sbN,
        sb_rank = sb_rank,
        axes_rect = axes_rect,
        cc = cc,
        rank_ab = rank_ab,
        reg_full = reg_full,
        reg_comp = reg_comp,
        keep1 = keep1,
        keep2 = keep2,
        axes_comp = axes_comp,
        r_dense_fill = r_dense_fill,
        r_packed_fill = r_packed_fill,
        r_dense_inv = r_dense_inv,
        r_packed_inv = r_packed_inv,
        p = p,
        q = q,
        xs = xs,
        ys = ys,
        axes_skew = axes_skew,
        r_idx_skew = r_idx_skew,
    )
end

function main(args=ARGS)
    section = _parse_string_arg(args, "section", "all")
    field_name = _parse_string_arg(args, "field", "qq")
    profile = _parse_string_arg(args, "profile", "default")
    reps = _parse_int_arg(args, "reps", 5)
    threads = _parse_bool_arg(args, "threads", false)
    out = _parse_path_arg(args, "out", joinpath(@__DIR__, "_tmp_signed_measures_microbench.csv"))

    field = _field_from_name(field_name)
    fx = _build_fixture(field, profile; threads=threads, section=section)
    rows = NamedTuple[]

    if _section_enabled(section, "point")
        push!(rows, _bench("sm point point_signed_measure", reps=reps) do
            SM.point_signed_measure(fx.surfM, fx.surf_axes; drop_zeros=true)
        end)
        push!(rows, _bench("sm point surface_from_point_signed_measure", reps=reps) do
            SM.surface_from_point_signed_measure(fx.pmM)
        end)
        push!(rows, _bench("sm point truncate_point_signed_measure", reps=reps) do
            SM.truncate_point_signed_measure(fx.pmM; max_terms=fx.cfg.trunc_terms, min_abs_weight=1)
        end)
        push!(rows, _bench("sm point kernel gaussian", reps=reps) do
            SM.point_signed_measure_kernel(fx.pmM, fx.pmN; sigma=1.0, kind=:gaussian)
        end)
        push!(rows, _bench("sm point kernel laplacian", reps=reps) do
            SM.point_signed_measure_kernel(fx.pmM, fx.pmN; sigma=1.0, kind=:laplacian)
        end)
    end

    if _section_enabled(section, "euler")
        push!(rows, _bench("sm euler euler_surface", reps=reps) do
            SM.euler_surface(fx.M, fx.pi, fx.opts)
        end)
        push!(rows, _bench("sm euler euler_signed_measure", reps=reps) do
            SM.euler_signed_measure(fx.M, fx.pi, fx.opts)
        end)
        push!(rows, _bench("sm euler euler_signed_measure session warm", reps=reps) do
            SM.euler_signed_measure(fx.M, fx.pi, fx.opts; cache=fx.sc_euler)
        end)
        push!(rows, _bench("sm euler euler_signed_measure trunc", reps=reps) do
            SM.euler_signed_measure(fx.M, fx.pi, fx.opts; max_terms=fx.cfg.trunc_terms, min_abs_weight=1)
        end)
        push!(rows, _bench("sm euler euler_distance", reps=reps) do
            SM.euler_distance(fx.M, fx.N, fx.pi, fx.opts; ord=1)
        end)
        push!(rows, _bench("sm euler mma_decomposition euler warm", reps=reps) do
            SM.mma_decomposition(fx.M, fx.pi, fx.opts; method=:euler, cache=fx.sc_mma)
        end)
    end

    if _section_enabled(section, "rect")
        push!(rows, _bench("sm rect rectangle_signed_barcode auto", reps=reps) do
            SM.rectangle_signed_barcode(fx.M, fx.pi, fx.opts)
        end)
        push!(rows, _bench("sm rect rectangle_signed_barcode bulk", reps=reps) do
            SM.rectangle_signed_barcode(fx.M, fx.pi, fx.opts; method=:bulk)
        end)
        push!(rows, _bench("sm rect rectangle_signed_barcode auto dense2d", reps=reps,
                           setup=() -> (SM._USE_PACKED_RECTANGLE_BULK_2D[] = false)) do
            SM.rectangle_signed_barcode(fx.M, fx.pi, fx.opts; method=:auto)
        end)
        push!(rows, _bench("sm rect rectangle_signed_barcode auto packed2d", reps=reps,
                           setup=() -> (SM._USE_PACKED_RECTANGLE_BULK_2D[] = true)) do
            SM.rectangle_signed_barcode(fx.M, fx.pi, fx.opts; method=:auto)
        end)
        push!(rows, _bench("sm rect rectangle_signed_barcode bulk dense2d", reps=reps,
                           setup=() -> (SM._USE_PACKED_RECTANGLE_BULK_2D[] = false)) do
            SM.rectangle_signed_barcode(fx.M, fx.pi, fx.opts; method=:bulk)
        end)
        push!(rows, _bench("sm rect rectangle_signed_barcode bulk packed2d", reps=reps,
                           setup=() -> (SM._USE_PACKED_RECTANGLE_BULK_2D[] = true)) do
            SM.rectangle_signed_barcode(fx.M, fx.pi, fx.opts; method=:bulk)
        end)
        push!(rows, _bench("sm rect rectangle_signed_barcode cached", reps=reps) do
            SM.rectangle_signed_barcode(fx.M, fx.pi, fx.opts; rq_cache=fx.rq_cache)
        end)
        push!(rows, _bench("sm rect rectangle_signed_barcode session warm", reps=reps) do
            SM.rectangle_signed_barcode(fx.M, fx.pi, fx.opts; cache=fx.sc_rect)
        end)
        tensor_sc = Ref{Any}(nothing)
        push!(rows, _bench("sm rect rectangle_signed_barcode session tensor warm span", reps=reps,
                           setup=() -> begin
                               sc = CM.SessionCache()
                               rq = SM._signed_measures_rank_query_cache(fx.M, fx.pi, nothing, sc)
                               reg_comp, _, _, axes_comp = SM._cached_rectangle_region_blocks_2d(
                                   fx.pi, fx.axes_rect, rq, sc; strict=false
                               )
                               SM._cached_rectangle_packed_tensor_2d(
                                   fx.M, fx.pi, fx.axes_rect, reg_comp, axes_comp, rq, sc, fx.cc;
                                   strict=false,
                                   threads=threads,
                               )
                               tensor_sc[] = sc
                           end) do
            SM.rectangle_signed_barcode(fx.M, fx.pi, fx.opts; cache=tensor_sc[], method=:bulk, max_span=0)
        end)
        push!(rows, _bench("sm rect rectangle_signed_barcode local", reps=reps) do
            SM.rectangle_signed_barcode(fx.M, fx.pi, fx.opts; method=:local)
        end)
        push!(rows, _bench("sm rect synth rectangle_signed_barcode auto skewed", reps=reps) do
            SM.rectangle_signed_barcode(fx.r_idx_skew, fx.axes_skew; method=:auto)
        end)
        push!(rows, _bench("sm rect synth rectangle_signed_barcode bulk skewed", reps=reps) do
            SM.rectangle_signed_barcode(fx.r_idx_skew, fx.axes_skew; method=:bulk)
        end)
        push!(rows, _bench("sm rect synth rectangle_signed_barcode local skewed", reps=reps) do
            SM.rectangle_signed_barcode(fx.r_idx_skew, fx.axes_skew; method=:local)
        end)
        push!(rows, _bench("sm rect rectangle_signed_barcode_rank", reps=reps) do
            SM.rectangle_signed_barcode_rank(fx.sb_cached)
        end)
        push!(rows, _bench("sm rect rank_from_signed_barcode", reps=reps) do
            SM.rank_from_signed_barcode(fx.sb_cached, fx.p, fx.q)
        end)
        push!(rows, _bench("sm rect truncate_signed_barcode", reps=reps) do
            SM.truncate_signed_barcode(fx.sb_cached; max_terms=fx.cfg.trunc_terms, min_abs_weight=1)
        end)
        SM._USE_PACKED_RECTANGLE_BULK_2D[] = true
    end

    if _section_enabled(section, "extract")
        push!(rows, _bench("sm extract reg build raw2d", reps=reps) do
            SM._rectangle_region_grid(fx.pi, fx.axes_rect, fx.rq_cache; strict=false)
        end)
        push!(rows, _bench("sm extract reg compress2d", reps=reps) do
            SM._compress_region_grid_2d(fx.reg_full, fx.axes_rect)
        end)
        push!(rows, _bench("sm extract reg build block2d session warm", reps=reps) do
            SM._cached_rectangle_region_blocks_2d(fx.pi, fx.axes_rect, fx.rq_cache, fx.sc_rect; strict=false)
        end)
        push!(rows, _bench("sm extract bulk fill dense2d warmrank", reps=reps) do
            SM._fill_rectangle_rank_tensor_dense_from_regions_2d(fx.reg_full, fx.rank_ab; threads=threads)
        end)
        push!(rows, _bench("sm extract bulk fill packed2d warmrank", reps=reps) do
            SM._fill_rectangle_rank_tensor_packed_from_regions_2d(fx.reg_comp, fx.rank_ab; threads=threads)
        end)
        push!(rows, _bench("sm extract bulk invert dense2d", reps=reps) do
            r = copy(fx.r_dense_fill)
            SM._mobius_inversion_interval_product!(r, 2)
        end)
        push!(rows, _bench("sm extract bulk invert packed2d", reps=reps) do
            r = copy(fx.r_packed_fill)
            SM._mobius_inversion_interval_product_packed_2d!(r, size(fx.reg_comp, 1), size(fx.reg_comp, 2); threads=threads)
        end)
        push!(rows, _bench("sm extract bulk extract dense2d", reps=reps) do
            SM._extract_rectangles_from_mobius_tensor(fx.r_dense_inv, fx.axes_rect; drop_zeros=true)
        end)
        push!(rows, _bench("sm extract bulk extract packed2d", reps=reps) do
            SM._extract_rectangles_from_packed_mobius_2d(fx.r_packed_inv, fx.axes_comp; axes_out=fx.axes_rect, drop_zeros=true)
        end)
    end

    if _section_enabled(section, "image")
        push!(rows, _bench("sm image rectangle_kernel linear", reps=reps) do
            SM.rectangle_signed_barcode_kernel(fx.sb_cached, fx.sbN; kind=:linear)
        end)
        push!(rows, _bench("sm image rectangle_kernel gaussian", reps=reps) do
            SM.rectangle_signed_barcode_kernel(fx.sb_cached, fx.sbN; kind=:gaussian, sigma=1.0)
        end)
        push!(rows, _bench("sm image rectangle_image center", reps=reps) do
            SM.rectangle_signed_barcode_image(fx.sb_cached; xs=fx.xs, ys=fx.ys, sigma=1.0, mode=:center, threads=threads)
        end)
        push!(rows, _bench("sm image rectangle_image center cutoff", reps=reps) do
            SM.rectangle_signed_barcode_image(fx.sb_cached; xs=fx.xs, ys=fx.ys, sigma=1.0, mode=:center,
                                              cutoff_tol=1.0e-6, threads=threads)
        end)
        push!(rows, _bench("sm image rectangle_image lo", reps=reps) do
            SM.rectangle_signed_barcode_image(fx.sb_cached; xs=fx.xs, ys=fx.ys, sigma=1.0, mode=:lo, threads=threads)
        end)
    end

    _write_csv(out, rows)
    println("load_mode=", _PM_LOAD_MODE)
    println("wrote ", out)
end

main()
