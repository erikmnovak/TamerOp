#!/usr/bin/env julia
#
# serialization_microbench.jl
#
# Purpose
# - Benchmark Serialization.jl under multiple regimes:
#   * internal owned formats (flange, encoding, dataset, pipeline, PL fringe, MPPI)
#   * external adapters (interop fixture loaders)
# - Report runtime + allocations + serialized size where applicable.
#
# Timing policy
# - warm-cached local filesystem, median over repeated runs
# - each case warms once before timing
#
# Usage
#   julia --project=. benchmark/serialization_microbench.jl
#   julia --project=. benchmark/serialization_microbench.jl --profile=full --reps=7
#   julia --project=. benchmark/serialization_microbench.jl --external=0
#

using Random
using SparseArrays
using LinearAlgebra
using Logging

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

const SER = TamerOp.Serialization
const CM = TamerOp.CoreModules
const OPT = TamerOp.Options
const DT = TamerOp.DataTypes
const EC = TamerOp.EncodingCore
const RES = TamerOp.Results
const FF = TamerOp.FiniteFringe
const FZ = TamerOp.FlangeZn
const ZE = TamerOp.ZnEncoding
const IR = TamerOp.IndicatorResolutions
const PLB = TamerOp.PLBackend
const INV = TamerOp.Invariants

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
        return split(a, "=", limit=2)[2]
    end
    return default
end

function _parse_bool_arg(args, key::String, default::Bool)
    for a in args
        startswith(a, key * "=") || continue
        v = lowercase(split(a, "=", limit=2)[2])
        if v in ("1", "true", "yes", "y", "on")
            return true
        elseif v in ("0", "false", "no", "n", "off")
            return false
        else
            error("Invalid boolean for $key: $v")
        end
    end
    return default
end

function _bench(name::AbstractString, f::Function; reps::Int=5, warmup::Int=1)
    for _ in 1:warmup
        f()
    end
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
    p90_i = max(1, ceil(Int, 0.9 * reps))
    p90_ms = times_ms[p90_i]
    println(rpad(name, 58),
            " median_time=", round(med_ms, digits=3), " ms",
            "  p90=", round(p90_ms, digits=3), " ms",
            "  median_alloc=", round(med_kib, digits=1), " KiB")
    return (ms=med_ms, p90_ms=p90_ms, kib=med_kib)
end

@inline _mb_per_s(bytes::Integer, ms::Real) = ms <= 0 ? Inf : (bytes / 1024.0^2) / (ms / 1000.0)
@inline _valdim(::Val{N}) where {N} = N

@inline function _rand_scalar(field::CM.AbstractCoeffField, rng::AbstractRNG)
    if field isa CM.RealField
        return randn(rng)
    end
    return CM.coerce(field, rand(rng, -2:2))
end

@inline function _random_face(::Val{N}, rng::AbstractRNG; free_prob::Float64=0.35) where {N}
    return FZ.face(N, [rand(rng) < free_prob for _ in 1:N])
end

function _make_random_flange(field::CM.AbstractCoeffField, ::Val{N};
                             nflats::Int, ninj::Int, seed::UInt,
                             bmin::Int=-8, bmax::Int=12) where {N}
    rng = Random.MersenneTwister(seed)
    K = CM.coeff_type(field)
    flats = Vector{FZ.IndFlat{N}}(undef, nflats)
    injectives = Vector{FZ.IndInj{N}}(undef, ninj)
    @inbounds for i in 1:nflats
        b = ntuple(_ -> rand(rng, bmin:bmax), N)
        flats[i] = FZ.IndFlat(_random_face(Val(N), rng), b; id=Symbol(:F, i))
    end
    @inbounds for j in 1:ninj
        b = ntuple(_ -> rand(rng, bmin:bmax), N)
        injectives[j] = FZ.IndInj(_random_face(Val(N), rng), b; id=Symbol(:E, j))
    end
    phi = Matrix{K}(undef, ninj, nflats)
    @inbounds for i in 1:ninj, j in 1:nflats
        phi[i, j] = _rand_scalar(field, rng)
    end
    return FZ.Flange{K}(N, flats, injectives, phi; field=field)
end

function _axis_encoding_flange(field::CM.AbstractCoeffField; ncuts::Int=9, seed::UInt=0x53455249414c)
    N = 2
    K = CM.coeff_type(field)
    rng = Random.MersenneTwister(seed)
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
    phi = Matrix{K}(undef, m, m)
    @inbounds for i in 1:m, j in 1:m
        # Keep a stable low-rank-ish but nontrivial pattern.
        phi[i, j] = CM.coerce(field, ((i == j) ? 1 : rand(rng, -1:1)))
    end
    return FZ.Flange{K}(N, flats, injectives, phi; field=field)
end

function _random_fringe(P::FF.AbstractPoset, field::CM.AbstractCoeffField;
                        nU::Int=36, nD::Int=36, seed::UInt=0x52494e4745)
    rng = Random.MersenneTwister(seed)
    n = FF.nvertices(P)
    K = CM.coeff_type(field)
    U = Vector{FF.Upset}(undef, nU)
    D = Vector{FF.Downset}(undef, nD)
    @inbounds for i in 1:nU
        U[i] = FF.principal_upset(P, rand(rng, 1:n))
    end
    @inbounds for j in 1:nD
        D[j] = FF.principal_downset(P, rand(rng, 1:n))
    end
    phi = Matrix{K}(undef, nD, nU)
    @inbounds for i in 1:nD, j in 1:nU
        if FF.intersects(U[j], D[i])
            phi[i, j] = _rand_scalar(field, rng)
        else
            phi[i, j] = zero(K)
        end
    end
    return FF.FringeModule{K}(P, U, D, phi; field=field)
end

function _check_flange_roundtrip(FG, FG2)
    FG.n == FG2.n || error("Flange roundtrip mismatch: n")
    length(FG.flats) == length(FG2.flats) || error("Flange roundtrip mismatch: flats")
    length(FG.injectives) == length(FG2.injectives) || error("Flange roundtrip mismatch: injectives")
    size(FG.phi) == size(FG2.phi) || error("Flange roundtrip mismatch: phi size")
    return true
end

function _check_encoding_roundtrip(H1, H2)
    size(H1.phi) == size(H2.phi) || error("Encoding roundtrip mismatch: phi size")
    FF.nvertices(H1.P) == FF.nvertices(H2.P) || error("Encoding roundtrip mismatch: nvertices")
    return true
end

@inline function _parse_flange_quiet(payload::AbstractString, field)
    return Logging.with_logger(Logging.NullLogger()) do
        SER.parse_flange_json(payload; field=field)
    end
end

function _bench_flange_roundtrip(tmpdir::AbstractString; reps::Int, profile::Symbol)
    println("\n=== Section A: Flange JSON roundtrip ===")
    cases = profile == :full ?
        [(Val(2), 20, 20), (Val(3), 28, 28), (Val(3), 40, 40)] :
        [(Val(2), 16, 16), (Val(3), 24, 24)]
    fields = [("qq", CM.QQField()), ("f2", CM.F2()), ("real64", CM.RealField(Float64))]

    results = NamedTuple[]
    for (fname, field) in fields
        for (VN, nflats, ninj) in cases
            nd = _valdim(VN)
            FG = _make_random_flange(field, VN; nflats=nflats, ninj=ninj,
                                     seed=UInt(hash((fname, nflats, ninj))))
            path = joinpath(tmpdir, "flange_$(fname)_n$(nd)_$(nflats)x$(ninj).json")
            SER.save_flange_json(path, FG)
            FG2 = SER.load_flange_json(path; field=field)
            _check_flange_roundtrip(FG, FG2)
            payload = read(path, String)
            parsef = field isa CM.RealField ?
                (() -> _parse_flange_quiet(payload, field)) :
                (() -> SER.parse_flange_json(payload; field=field))
            FG3 = parsef()
            _check_flange_roundtrip(FG, FG3)
            bytes = filesize(path)

            bsave = _bench("flange save $(fname) n=$(nd) $(nflats)x$(ninj)",
                           () -> SER.save_flange_json(path, FG); reps=reps)
            bload = _bench("flange load $(fname) n=$(nd) $(nflats)x$(ninj)",
                           () -> SER.load_flange_json(path; field=field); reps=reps)
            bparse = _bench("flange parse(str) $(fname) n=$(nd) $(nflats)x$(ninj)",
                            parsef; reps=reps)
            println("  file_size=", bytes, " bytes",
                    "  save_throughput=", round(_mb_per_s(bytes, bsave.ms), digits=2), " MB/s",
                    "  load_throughput=", round(_mb_per_s(bytes, bload.ms), digits=2), " MB/s")
            push!(results, (section=:flange, name="save_$(fname)_$(nflats)x$(ninj)", ms=bsave.ms))
            push!(results, (section=:flange, name="load_$(fname)_$(nflats)x$(ninj)", ms=bload.ms))
            push!(results, (section=:flange, name="parse_$(fname)_$(nflats)x$(ninj)", ms=bparse.ms))
        end
    end
    return results
end

function _bench_encoding_roundtrip(tmpdir::AbstractString; reps::Int, profile::Symbol)
    println("\n=== Section B: Encoding JSON roundtrip ===")
    ncuts = profile == :full ? 12 : 9
    qf = CM.QQField()
    f2 = CM.F2()

    FGqq = _axis_encoding_flange(qf; ncuts=ncuts)
    FGf2 = _axis_encoding_flange(f2; ncuts=max(4, ncuts - 2))
    optsqq = OPT.EncodingOptions(backend=:zn, max_regions=500_000, field=qf)
    optsf2 = OPT.EncodingOptions(backend=:zn, max_regions=500_000, field=f2)

    P_sig, H_sig, pi_sig = ZE.encode_from_flange(FGqq, optsqq; poset_kind=:signature)
    P_den, H_den, pi_den = ZE.encode_from_flange(FGqq, optsqq; poset_kind=:dense)
    P_f2, H_f2, pi_f2 = ZE.encode_from_flange(FGf2, optsf2; poset_kind=:signature)

    # GridEncodingMap regime (different pi family).
    side = profile == :full ? 24 : 16
    axes = (collect(0.0:1.0:Float64(side - 1)), collect(0.0:1.0:Float64(side - 1)))
    P_grid = FF.GridPoset(axes)
    H_grid = _random_fringe(P_grid, qf; nU=48, nD=48)
    pi_grid = EC.GridEncodingMap(P_grid, axes; orientation=(1, 1))

    cases = [
        ("signature_no_leq_no_pi", P_sig, H_sig, nothing, false, nothing),
        ("signature_with_leq_no_pi", P_sig, H_sig, nothing, true, nothing),
        ("signature_no_leq_with_pi", P_sig, H_sig, pi_sig, false, nothing),
        ("signature_with_leq_with_pi", P_sig, H_sig, pi_sig, true, nothing),
        ("dense_with_leq_with_pi", P_den, H_den, pi_den, true, nothing),
        ("grid_with_leq_with_pi", P_grid, H_grid, pi_grid, true, nothing),
        ("signature_f2_with_pi_no_leq", P_f2, H_f2, pi_f2, false, f2),
    ]

    results = NamedTuple[]
    for (name, P, H, pi, include_leq, fld) in cases
        path = joinpath(tmpdir, "encoding_$(name).json")
        if pi === nothing
            SER.save_encoding_json(path, H; include_leq=include_leq)
            H2 = SER.load_encoding_json(path; output=:fringe)
            _check_encoding_roundtrip(H, H2)
            bsave = _bench("encoding save $(name)",
                           () -> SER.save_encoding_json(path, H; include_leq=include_leq); reps=reps)
            bload = _bench("encoding load $(name)",
                           () -> SER.load_encoding_json(path; output=:fringe); reps=reps)
            bytes = filesize(path)
            println("  file_size=", bytes, " bytes",
                    "  save_throughput=", round(_mb_per_s(bytes, bsave.ms), digits=2), " MB/s",
                    "  load_throughput=", round(_mb_per_s(bytes, bload.ms), digits=2), " MB/s")
            push!(results, (section=:encoding, name="save_$(name)", ms=bsave.ms))
            push!(results, (section=:encoding, name="load_$(name)", ms=bload.ms))
        else
            SER.save_encoding_json(path, P, H, pi; include_leq=include_leq)
            H2, pi2 = SER.load_encoding_json(path; output=:fringe_with_pi, field=fld)
            _check_encoding_roundtrip(H, H2)
            EC.dimension(pi2) == EC.dimension(pi) || error("Encoding roundtrip mismatch: pi dimension")

            bsave = _bench("encoding save $(name)",
                           () -> SER.save_encoding_json(path, P, H, pi; include_leq=include_leq); reps=reps)
            bload = _bench("encoding load+pi $(name)",
                           () -> SER.load_encoding_json(path; output=:fringe_with_pi, field=fld); reps=reps)
            bytes = filesize(path)
            println("  file_size=", bytes, " bytes",
                    "  save_throughput=", round(_mb_per_s(bytes, bsave.ms), digits=2), " MB/s",
                    "  load_throughput=", round(_mb_per_s(bytes, bload.ms), digits=2), " MB/s")
            push!(results, (section=:encoding, name="save_$(name)", ms=bsave.ms))
            push!(results, (section=:encoding, name="load_$(name)", ms=bload.ms))
        end
    end

    # Field-coercion regime (F2 payload loaded as QQ).
    path_coerce = joinpath(tmpdir, "encoding_signature_f2_coerce.json")
    SER.save_encoding_json(path_coerce, P_f2, H_f2, pi_f2; include_leq=false)
    _ = SER.load_encoding_json(path_coerce; output=:fringe, field=CM.QQField())
    bcoerce = _bench("encoding load field-coerce f2->qq",
                     () -> SER.load_encoding_json(path_coerce; output=:fringe, field=CM.QQField()); reps=reps)
    push!(results, (section=:encoding, name="load_coerce_f2_to_qq", ms=bcoerce.ms))
    return results
end

function _bench_dataset_pipeline(tmpdir::AbstractString; reps::Int, profile::Symbol)
    println("\n=== Section C: Dataset/Pipeline JSON ===")
    rng = Random.MersenneTwister(0x44534554)
    npts = profile == :full ? 8_000 : 3_000
    dim = 3
    pts = randn(rng, npts, dim)
    point_cloud = DT.PointCloud(pts)

    ngraph = profile == :full ? 8_000 : 3_000
    edges = Tuple{Int,Int}[]
    sizehint!(edges, 3 * ngraph)
    for i in 1:(ngraph - 1)
        push!(edges, (i, i + 1))
    end
    for _ in 1:(2 * ngraph)
        u = rand(rng, 1:ngraph)
        v = rand(rng, 1:ngraph)
        u == v && continue
        push!(edges, (min(u, v), max(u, v)))
    end
    coords = [randn(rng, 2) for _ in 1:ngraph]
    weights = rand(rng, length(edges))
    graph = DT.GraphData(ngraph, edges; coords=coords, weights=weights, T=Float64)

    side = profile == :full ? 320 : 192
    img = DT.ImageNd(rand(rng, Float64, side, side))

    spec = OPT.FiltrationSpec(kind=:rips_vr_filtration, max_dim=1, radius=0.35,
                             poset_kind=:signature, axes_policy=:encoding)
    popts = OPT.PipelineOptions(poset_kind=:signature, axes_policy=:encoding)

    cases = [
        ("dataset_pointcloud", point_cloud),
        ("dataset_graph", graph),
        ("dataset_image", img),
    ]

    results = NamedTuple[]
    for (name, data) in cases
        path = joinpath(tmpdir, "$(name).json")
        SER.save_dataset_json(path, data)
        _ = SER.load_dataset_json(path)
        bsave = _bench("save $(name)", () -> SER.save_dataset_json(path, data); reps=reps)
        bload_strict = _bench("load $(name) strict", () -> SER.load_dataset_json(path; validation=:strict); reps=reps)
        bload_trusted = _bench("load $(name) trusted", () -> SER.load_dataset_json(path; validation=:trusted); reps=reps)
        bytes = filesize(path)
        println("  file_size=", bytes, " bytes",
                "  save_throughput=", round(_mb_per_s(bytes, bsave.ms), digits=2), " MB/s",
                "  load_throughput=", round(_mb_per_s(bytes, bload_strict.ms), digits=2), " MB/s")
        push!(results, (section=:dataset, name="save_$(name)", ms=bsave.ms))
        push!(results, (section=:dataset, name="load_$(name)_strict", ms=bload_strict.ms))
        push!(results, (section=:dataset, name="load_$(name)_trusted", ms=bload_trusted.ms))
    end

    ppath = joinpath(tmpdir, "pipeline_pointcloud.json")
    SER.save_pipeline_json(ppath, point_cloud, spec; degree=1, pipeline_opts=popts)
    _ = SER.load_pipeline_json(ppath)
    bpsave = _bench("save pipeline(pointcloud)", () -> SER.save_pipeline_json(ppath, point_cloud, spec; degree=1, pipeline_opts=popts); reps=reps)
    bpload_strict = _bench("load pipeline(pointcloud) strict", () -> SER.load_pipeline_json(ppath; validation=:strict); reps=reps)
    bpload_trusted = _bench("load pipeline(pointcloud) trusted", () -> SER.load_pipeline_json(ppath; validation=:trusted); reps=reps)
    push!(results, (section=:pipeline, name="save_pipeline", ms=bpsave.ms))
    push!(results, (section=:pipeline, name="load_pipeline_strict", ms=bpload_strict.ms))
    push!(results, (section=:pipeline, name="load_pipeline_trusted", ms=bpload_trusted.ms))
    return results
end

function _bench_owned_aux(tmpdir::AbstractString; reps::Int, profile::Symbol)
    println("\n=== Section D: PL fringe + MPPI JSON ===")
    results = NamedTuple[]

    Aup = reshape(CM.QQ[-1 0], 1, 2)
    bup = CM.QQ[0]
    Adown = reshape(CM.QQ[1 0], 1, 2)
    bdown = CM.QQ[2]
    ups = [TamerOp.PLPolyhedra.PLUpset(TamerOp.PLPolyhedra.PolyUnion(2, [
        TamerOp.PLPolyhedra.HPoly(2, Aup, bup, nothing, falses(1), TamerOp.PLPolyhedra.STRICT_EPS_QQ),
    ]))]
    downs = [TamerOp.PLPolyhedra.PLDownset(TamerOp.PLPolyhedra.PolyUnion(2, [
        TamerOp.PLPolyhedra.HPoly(2, Adown, bdown, nothing, falses(1), TamerOp.PLPolyhedra.STRICT_EPS_QQ),
    ]))]
    Fpl = TamerOp.PLPolyhedra.PLFringe(2, ups, downs, reshape(CM.QQ[1], 1, 1))
    pl_path = joinpath(tmpdir, "pl_fringe.json")
    SER.save_pl_fringe_json(pl_path, Fpl)
    bpl_save = _bench("save pl_fringe", () -> SER.save_pl_fringe_json(pl_path, Fpl); reps=reps)
    bpl_load_strict = _bench("load pl_fringe strict", () -> SER.load_pl_fringe_json(pl_path; validation=:strict); reps=reps)
    bpl_load_trusted = _bench("load pl_fringe trusted", () -> SER.load_pl_fringe_json(pl_path; validation=:trusted); reps=reps)
    push!(results, (section=:pl_fringe, name="save_pl_fringe", ms=bpl_save.ms))
    push!(results, (section=:pl_fringe, name="load_pl_fringe_strict", ms=bpl_load_strict.ms))
    push!(results, (section=:pl_fringe, name="load_pl_fringe_trusted", ms=bpl_load_trusted.ms))

    Ups = [PLB.BoxUpset([0.0, 0.0], [1.0, 1.0])]
    Downs = [PLB.BoxDownset([2.0, 2.0], [3.0, 3.0])]
    _, Hbox, pibox = PLB.encode_fringe_boxes(Ups, Downs)
    Mbox = IR.pmodule_from_fringe(Hbox)
    resolution = profile == :full ? 16 : 8
    img = INV.mpp_image(Mbox, pibox; resolution=resolution, sigma=0.3, N=4)

    dpath = joinpath(tmpdir, "mpp_decomposition.json")
    ipath = joinpath(tmpdir, "mpp_image.json")
    INV.save_mpp_decomposition_json(dpath, img.decomp)
    INV.save_mpp_image_json(ipath, img)
    bd_save = _bench("save mpp_decomposition", () -> INV.save_mpp_decomposition_json(dpath, img.decomp); reps=reps)
    bd_load_strict = _bench("load mpp_decomposition strict", () -> INV.load_mpp_decomposition_json(dpath; validation=:strict); reps=reps)
    bd_load_trusted = _bench("load mpp_decomposition trusted", () -> INV.load_mpp_decomposition_json(dpath; validation=:trusted); reps=reps)
    bi_save = _bench("save mpp_image", () -> INV.save_mpp_image_json(ipath, img); reps=reps)
    bi_load_strict = _bench("load mpp_image strict", () -> INV.load_mpp_image_json(ipath; validation=:strict); reps=reps)
    bi_load_trusted = _bench("load mpp_image trusted", () -> INV.load_mpp_image_json(ipath; validation=:trusted); reps=reps)
    push!(results, (section=:mpp, name="save_decomposition", ms=bd_save.ms))
    push!(results, (section=:mpp, name="load_decomposition_strict", ms=bd_load_strict.ms))
    push!(results, (section=:mpp, name="load_decomposition_trusted", ms=bd_load_trusted.ms))
    push!(results, (section=:mpp, name="save_image", ms=bi_save.ms))
    push!(results, (section=:mpp, name="load_image_strict", ms=bi_load_strict.ms))
    push!(results, (section=:mpp, name="load_image_trusted", ms=bi_load_trusted.ms))
    return results
end

function _bench_external_adapters(; reps::Int, profile::Symbol)
    println("\n=== Section E: External adapter fixture loads ===")
    fixtures = joinpath(@__DIR__, "..", "test", "fixtures", "interop")
    isdir(fixtures) || error("Interop fixtures not found: $(fixtures)")

    cases = [
        ("load_gudhi_json", () -> SER.load_gudhi_json(joinpath(fixtures, "gudhi.json"))),
        ("load_ripserer_json", () -> SER.load_ripserer_json(joinpath(fixtures, "ripserer.json"))),
        ("load_eirene_json", () -> SER.load_eirene_json(joinpath(fixtures, "eirene.json"))),
        ("load_gudhi_txt", () -> SER.load_gudhi_txt(joinpath(fixtures, "gudhi.txt"))),
        ("load_ripserer_txt", () -> SER.load_ripserer_txt(joinpath(fixtures, "ripserer.txt"))),
        ("load_eirene_txt", () -> SER.load_eirene_txt(joinpath(fixtures, "eirene.txt"))),
    ]
    if profile == :full
        append!(cases, [
            ("load_ripser_point_cloud", () -> SER.load_ripser_point_cloud(joinpath(fixtures, "ripser_point_cloud.txt"))),
            ("load_ripser_distance", () -> SER.load_ripser_distance(joinpath(fixtures, "ripser_distance.txt"); max_dim=1)),
            ("load_ripser_lower_distance", () -> SER.load_ripser_lower_distance(joinpath(fixtures, "ripser_lower_distance.txt"); max_dim=1)),
            ("load_ripser_upper_distance", () -> SER.load_ripser_upper_distance(joinpath(fixtures, "ripser_upper_distance.txt"); max_dim=1)),
            ("load_ripser_sparse_triplet", () -> SER.load_ripser_sparse_triplet(joinpath(fixtures, "ripser_sparse_triplet.txt"); max_dim=1)),
            ("load_ripser_binary_lower_distance", () -> SER.load_ripser_binary_lower_distance(joinpath(fixtures, "ripser_binary_lower_distance.bin"); max_dim=1)),
            ("load_dipha_distance_matrix", () -> SER.load_dipha_distance_matrix(joinpath(fixtures, "dipha_distance_matrix.bin"); max_dim=1)),
        ])
    end

    results = NamedTuple[]
    for (name, f) in cases
        _ = f()
        b = _bench(name, f; reps=reps)
        push!(results, (section=:external, name=name, ms=b.ms))
    end
    return results
end

function _print_summary(all_results)
    println("\n=== Summary: slowest median-time cases ===")
    isempty(all_results) && return
    ord = sort(all_results; by=x -> x.ms, rev=true)
    topk = min(length(ord), 12)
    for i in 1:topk
        r = ord[i]
        println(lpad(i, 2), ". ", rpad(String(r.section) * "/" * r.name, 56),
                round(r.ms, digits=3), " ms")
    end
end

function main(args=ARGS)
    profile_str = lowercase(_parse_str_arg(args, "--profile", "quick"))
    profile = profile_str == "full" ? :full : :quick
    reps = _parse_int_arg(args, "--reps", profile == :full ? 6 : 4)
    include_external = _parse_bool_arg(args, "--external", true)

    println("Serialization micro-benchmark")
    println("profile=$(profile), reps=$(reps), external=$(include_external)")
    println("timing_policy=warm-cached filesystem, median over reps, warmup=1")

    tmpdir = mktempdir(prefix="serialization_microbench_")
    println("tmpdir=$(tmpdir)")

    all_results = NamedTuple[]
    try
        append!(all_results, _bench_flange_roundtrip(tmpdir; reps=reps, profile=profile))
        append!(all_results, _bench_encoding_roundtrip(tmpdir; reps=reps, profile=profile))
        append!(all_results, _bench_dataset_pipeline(tmpdir; reps=reps, profile=profile))
        append!(all_results, _bench_owned_aux(tmpdir; reps=reps, profile=profile))
        if include_external
            append!(all_results, _bench_external_adapters(; reps=reps, profile=profile))
        end
        _print_summary(all_results)
    finally
        rm(tmpdir; recursive=true, force=true)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
