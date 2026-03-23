#!/usr/bin/env julia

using Random
using JSON3

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

function _parse_int_arg(args, key::String, default::Int)
    for a in args
        startswith(a, key * "=") || continue
        return max(1, parse(Int, split(a, "=", limit=2)[2]))
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
            error("Invalid bool value for $key: $v")
        end
    end
    return default
end

function _bench(name::AbstractString, f::Function; reps::Int=5, warmup::Int=1)
    for _ in 1:warmup
        f()
    end
    GC.gc()
    ts = Vector{Float64}(undef, reps)
    bs = Vector{Int}(undef, reps)
    for i in 1:reps
        m = @timed f()
        ts[i] = 1000.0 * m.time
        bs[i] = m.bytes
    end
    sort!(ts)
    sort!(bs)
    med_ms = ts[cld(reps, 2)]
    med_kib = bs[cld(reps, 2)] / 1024.0
    println(rpad(name, 52), " median=", round(med_ms, digits=3), " ms",
            " alloc=", round(med_kib, digits=1), " KiB")
    return (ms=med_ms, kib=med_kib)
end

function _axis_encoding_flange(field::CM.AbstractCoeffField; ncuts::Int=8, seed::Integer=0x5352454e)
    N = 2
    rng = Random.MersenneTwister(UInt(seed))
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
    @inbounds for i in 1:m, j in 1:m
        phi[i, j] = CM.coerce(field, i == j ? 1 : rand(rng, -1:1))
    end
    return FZ.Flange{K}(N, flats, injectives, phi; field=field)
end

function _make_graph(n::Int, rng::AbstractRNG)
    edges = Tuple{Int,Int}[]
    sizehint!(edges, 3n)
    for i in 1:(n - 1)
        push!(edges, (i, i + 1))
    end
    for _ in 1:(2n)
        u = rand(rng, 1:n)
        v = rand(rng, 1:n)
        u == v && continue
        push!(edges, (min(u, v), max(u, v)))
    end
    coords = [randn(rng, 2) for _ in 1:n]
    weights = rand(rng, length(edges))
    return DT.GraphData(n, edges; coords=coords, weights=weights, T=Float64)
end

function _load_dataset_json_baseline(path::AbstractString)
    raw = read(path, String)
    kind = JSON3.read(raw, NamedTuple{(:kind,),Tuple{String}}).kind
    if kind == "PointCloud"
        obj = JSON3.read(raw, SER._PointCloudColumnarJSON)
        pts = Matrix{Float64}(undef, obj.n, obj.d)
        t = 1
        @inbounds for i in 1:obj.n, j in 1:obj.d
            pts[i, j] = obj.points_flat[t]
            t += 1
        end
        return DT.PointCloud(pts; copy=false)
    elseif kind == "GraphData"
        obj = JSON3.read(raw, SER._GraphDataColumnarJSON)
        coords = if obj.coords_dim === nothing || obj.coords_flat === nothing
            nothing
        else
            out = Matrix{Float64}(undef, obj.n, obj.coords_dim)
            t = 1
            @inbounds for i in 1:obj.n, j in 1:obj.coords_dim
                out[i, j] = obj.coords_flat[t]
                t += 1
            end
            out
        end
        return DT.GraphData(obj.n, obj.edges_u, obj.edges_v;
                            coords=coords,
                            weights=obj.weights,
                            T=Float64,
                            copy=false)
    end
    return SER._dataset_from_obj(JSON3.read(raw))
end

function _obj_from_dataset_baseline(data)
    if data isa DT.PointCloud
        pts = DT.point_matrix(data)
        npts, d = size(pts)
        T = eltype(pts)
        flat = Vector{T}(undef, npts * d)
        t = 1
        @inbounds for i in 1:npts, j in 1:d
            flat[t] = pts[i, j]
            t += 1
        end
        return Dict("kind" => "PointCloud",
                    "layout" => "columnar_v1",
                    "n" => npts,
                    "d" => d,
                    "points_flat" => flat)
    elseif data isa DT.GraphData
        edges_u, edges_v = DT.edge_columns(data)
        coords_dim = nothing
        coords_flat = nothing
        coords = DT.coord_matrix(data)
        if coords !== nothing
            ncoords, d = size(coords)
            Tcoord = eltype(coords)
            buf = Vector{Tcoord}(undef, ncoords * d)
            t = 1
            @inbounds for i in 1:ncoords, j in 1:d
                buf[t] = coords[i, j]
                t += 1
            end
            coords_dim = d
            coords_flat = buf
        end
        return Dict("kind" => "GraphData",
                    "layout" => "columnar_v1",
                    "n" => data.n,
                    "edges_u" => edges_u,
                    "edges_v" => edges_v,
                    "coords_dim" => coords_dim,
                    "coords_flat" => coords_flat,
                    "weights" => data.weights === nothing ? nothing : collect(data.weights))
    end
    return SER._obj_from_dataset(data)
end

function _save_dataset_json_baseline(path::AbstractString, data)
    return SER._json_write(path, _obj_from_dataset_baseline(data); pretty=false)
end

function _decode_phi_qqchunks_baseline(phi_obj::SER._PhiQQChunksJSON,
                                       m_expected::Int,
                                       k_expected::Int)
    m, k = SER._phi_dims(phi_obj)
    (m == m_expected && k == k_expected) || error("phi dimensions mismatch")
    len = m * k
    Phi = Matrix{CM.QQ}(undef, m, k)
    @inbounds for idx in 1:len
        s = phi_obj.num_sign[idx]
        num = SER._chunks_to_bigint(phi_obj.num_chunks, phi_obj.num_ptr, idx, "qq_chunks_v1.num")
        den = SER._chunks_to_bigint(phi_obj.den_chunks, phi_obj.den_ptr, idx, "qq_chunks_v1.den")
        den == 0 && error("qq_chunks_v1.den must be nonzero.")
        if s < 0
            num = -num
        elseif s == 0
            num = BigInt(0)
        end
        Phi[idx] = CM.QQ(num // den)
    end
    return Phi
end

function _load_encoding_json_strict_baseline(path::AbstractString)
    raw = read(path, String)
    obj = JSON3.read(raw, SER._FiniteEncodingFringeJSONV1)
    obj.kind == "FiniteEncodingFringe" || error("Unsupported encoding JSON kind: $(obj.kind)")
    obj.schema_version == SER.ENCODING_SCHEMA_VERSION ||
        error("Unsupported encoding JSON schema_version: $(obj.schema_version)")
    P = SER._parse_poset_from_typed(obj.poset)
    n = FF.nvertices(P)
    Umasks = SER._decode_masks(obj.U, "U", n)
    Dmasks = SER._decode_masks(obj.D, "D", n)
    U = SER._build_upsets(P, Umasks, true)
    D = SER._build_downsets(P, Dmasks, true)
    saved_field = SER._field_from_typed(obj.coeff_field)
    m = length(D)
    k = length(U)
    Phi = SER._decode_phi(obj.phi, saved_field, saved_field, m, k)
    return FF.FringeModule{CM.coeff_type(saved_field)}(P, U, D, Phi; field=saved_field)
end

function _load_encoding_json_trusted_baseline(path::AbstractString)
    raw = read(path, String)
    obj = JSON3.read(raw, SER._FiniteEncodingFringeJSONV1)
    P = SER._parse_poset_from_typed(obj.poset)
    n = FF.nvertices(P)
    U = SER._build_upsets(P, SER._decode_masks(obj.U, "U", n), false)
    D = SER._build_downsets(P, SER._decode_masks(obj.D, "D", n), false)
    m = length(D)
    k = length(U)
    Phi = if obj.phi isa SER._PhiQQChunksJSON
        _decode_phi_qqchunks_baseline(obj.phi, m, k)
    else
        saved_field = SER._field_from_typed(obj.coeff_field)
        SER._decode_phi(obj.phi, saved_field, saved_field, m, k)
    end
    return FF.FringeModule{CM.QQ}(P, U, D, Phi; field=CM.QQField())
end

function _parse_finite_fringe_json_baseline(json_src;
                                            field::Union{Nothing,CM.AbstractCoeffField}=nothing,
                                            validation::Symbol=:strict)
    obj = JSON3.read(json_src)
    validate_masks = SER._resolve_validation_mode(validation)
    P = SER._parse_poset_from_obj(obj["poset"])
    n = FF.nvertices(P)
    U = SER._build_upsets(P, SER._external_parse_mask_rows(obj["U"], "U", n), validate_masks)
    D = SER._build_downsets(P, SER._external_parse_mask_rows(obj["D"], "D", n), validate_masks)
    saved_field, target_field = SER._external_parse_field(obj, field)
    Phi = SER._external_parse_phi(obj["phi"], saved_field, target_field, length(D), length(U))
    return FF.FringeModule{CM.coeff_type(target_field)}(P, U, D, Phi; field=target_field)
end

function _parse_pl_fringe_json_baseline(json_src)
    return SER._parse_pl_fringe_obj(JSON3.read(json_src))
end

function _encoding_section(tmpdir::AbstractString; reps::Int)
    println("\n=== Encoding JSON fast paths ===")
    qf = CM.QQField()
    FG = _axis_encoding_flange(qf; ncuts=8)
    opts = OPT.EncodingOptions(backend=:zn, max_regions=500_000, field=qf)
    P, H, pi = ZE.encode_from_flange(FG, opts; poset_kind=:signature)
    path_auto = joinpath(tmpdir, "enc_auto.json")
    path_dense = joinpath(tmpdir, "enc_dense.json")
    path_pretty = joinpath(tmpdir, "enc_pretty.json")
    path_compact = joinpath(tmpdir, "enc_compact.json")

    SER.save_encoding_json(path_auto, P, H, pi; include_leq=:auto, pretty=false)
    SER.save_encoding_json(path_dense, P, H, pi; include_leq=true, pretty=false)
    SER.save_encoding_json(path_pretty, P, H, pi; include_leq=:auto, pretty=true)
    SER.save_encoding_json(path_compact, P, H, pi; include_leq=:auto, pretty=false)

    b_load_v1_old = _bench("load schema_v1 strict baseline",
                           () -> _load_encoding_json_strict_baseline(path_auto);
                           reps=reps)
    b_load_v1 = _bench("load schema_v1 (typed) strict",
                       () -> SER.load_encoding_json(path_auto; output=:fringe, validation=:strict);
                       reps=reps)
    b_load_trusted = _bench("load trusted",
                            () -> SER.load_encoding_json(path_auto; output=:fringe, validation=:trusted);
                            reps=reps)
    b_load_pi = _bench("load trusted + pi",
                       () -> SER.load_encoding_json(path_auto; output=:fringe_with_pi, validation=:trusted);
                       reps=reps)
    b_save_auto = _bench("save include_leq=:auto",
                         () -> SER.save_encoding_json(path_auto, P, H, pi; include_leq=:auto, pretty=false);
                         reps=reps)
    b_save_dense = _bench("save include_leq=true",
                          () -> SER.save_encoding_json(path_dense, P, H, pi; include_leq=true, pretty=false);
                          reps=reps)
    b_save_pretty = _bench("save pretty=true",
                           () -> SER.save_encoding_json(path_pretty, P, H, pi; include_leq=:auto, pretty=true);
                           reps=reps)
    b_save_compact = _bench("save pretty=false",
                            () -> SER.save_encoding_json(path_compact, P, H, pi; include_leq=:auto, pretty=false);
                            reps=reps)

    Kqq = CM.coeff_type(qf)
    mphi, kphi = size(H.phi)
    phi_big = Matrix{Kqq}(undef, mphi, kphi)
    @inbounds for i in 1:mphi, j in 1:kphi
        if H.phi[i, j] == 0
            phi_big[i, j] = 0
        else
            num = BigInt(10)^28 + BigInt(97 * i + j)
            den = BigInt(10)^19 + BigInt(31 * j + i + 1)
            phi_big[i, j] = num // den
        end
    end
    H_big = FF.FringeModule{Kqq}(P, H.U, H.D, phi_big; field=qf)
    path_big = joinpath(tmpdir, "enc_qq_big.json")
    SER.save_encoding_json(path_big, P, H_big, pi; include_leq=:auto, pretty=false)
    b_load_big_old = _bench("load qq_chunks heavy baseline",
                            () -> _load_encoding_json_trusted_baseline(path_big);
                            reps=reps)
    b_load_big = _bench("load qq_chunks heavy trusted",
                        () -> SER.load_encoding_json(path_big; output=:fringe, validation=:trusted);
                        reps=reps)

    ext_obj = JSON3.read(read(path_auto, String))
    finite_json = JSON3.write(Dict(
        "poset" => ext_obj["poset"],
        "U" => [collect(U.mask) for U in H.U],
        "D" => [collect(D.mask) for D in H.D],
        "coeff_field" => ext_obj["coeff_field"],
        "phi" => ext_obj["phi"],
    ))

    Aup = reshape(CM.QQ[-1 0], 1, 2)
    bup = CM.QQ[0]
    Adown = reshape(CM.QQ[1 0], 1, 2)
    bdown = CM.QQ[2]
    ups = [TamerOp.PLPolyhedra.PLUpset(TamerOp.PLPolyhedra.PolyUnion(2, [
        TamerOp.PLPolyhedra.HPoly(2, Aup, bup, nothing, falses(1), TamerOp.PLPolyhedra.STRICT_EPS_QQ),
        TamerOp.PLPolyhedra.HPoly(2, reshape(CM.QQ[0 -1], 1, 2), CM.QQ[i], nothing, falses(1), TamerOp.PLPolyhedra.STRICT_EPS_QQ),
    ])) for i in 0:15]
    downs = [TamerOp.PLPolyhedra.PLDownset(TamerOp.PLPolyhedra.PolyUnion(2, [
        TamerOp.PLPolyhedra.HPoly(2, Adown, bdown, nothing, falses(1), TamerOp.PLPolyhedra.STRICT_EPS_QQ),
        TamerOp.PLPolyhedra.HPoly(2, reshape(CM.QQ[0 1], 1, 2), CM.QQ[i + 3], nothing, falses(1), TamerOp.PLPolyhedra.STRICT_EPS_QQ),
    ])) for i in 0:15]
    phi_pl = Matrix{CM.QQ}(undef, length(downs), length(ups))
    @inbounds for i in axes(phi_pl, 1), j in axes(phi_pl, 2)
        phi_pl[i, j] = (i == j || (i + j) % 7 == 0) ? CM.QQ(1) : CM.QQ(0)
    end
    Fpl = TamerOp.PLPolyhedra.PLFringe(2, ups, downs, phi_pl)
    path_pl = joinpath(tmpdir, "pl_fringe.json")
    SER.save_pl_fringe_json(path_pl, Fpl; pretty=false)
    pl_json = read(path_pl, String)
    b_parse_finite_old = _bench("parse finite fringe baseline",
                                () -> _parse_finite_fringe_json_baseline(finite_json; validation=:strict);
                                reps=reps)
    b_parse_finite = _bench("parse finite fringe typed",
                            () -> SER.parse_finite_fringe_json(finite_json; validation=:strict);
                            reps=reps)
    b_parse_pl_old = _bench("parse PL fringe baseline",
                            () -> _parse_pl_fringe_json_baseline(pl_json);
                            reps=reps)
    b_parse_pl = _bench("parse PL fringe typed",
                        () -> SER.parse_pl_fringe_json(pl_json);
                        reps=reps)

    size_auto = filesize(path_auto)
    size_dense = filesize(path_dense)
    size_pretty = filesize(path_pretty)
    size_compact = filesize(path_compact)
    println("file size include_leq=:auto: ", size_auto, " bytes")
    println("file size include_leq=true:  ", size_dense, " bytes",
            "  (x", round(size_dense / max(1, size_auto), digits=2), ")")
    println("file size pretty=true:       ", size_pretty, " bytes")
    println("file size pretty=false:      ", size_compact, " bytes",
            "  (", round(100 * (1 - size_compact / max(1, size_pretty)), digits=1), "% smaller)")
    println("trusted load speedup (trusted vs strict): x",
            round(b_load_v1.ms / max(1e-9, b_load_trusted.ms), digits=2))
    println("pi decode overhead (fringe_with_pi vs fringe): x",
            round(b_load_pi.ms / max(1e-9, b_load_trusted.ms), digits=2))
    println("auto leq save speedup (auto vs dense): x",
            round(b_save_dense.ms / max(1e-9, b_save_auto.ms), digits=2))
    println("compact save speedup (pretty=false vs true): x",
            round(b_save_pretty.ms / max(1e-9, b_save_compact.ms), digits=2))
    println("qq heavy/load baseline ratio (heavy vs regular trusted): x",
            round(b_load_big.ms / max(1e-9, b_load_trusted.ms), digits=2))
    println("strict load speedup (new vs baseline): x",
            round(b_load_v1_old.ms / max(1e-9, b_load_v1.ms), digits=2))
    println("qq-heavy load speedup (new vs baseline): x",
            round(b_load_big_old.ms / max(1e-9, b_load_big.ms), digits=2))
    println("finite fringe parse speedup (typed vs baseline): x",
            round(b_parse_finite_old.ms / max(1e-9, b_parse_finite.ms), digits=2))
    println("PL fringe parse speedup (typed vs baseline): x",
            round(b_parse_pl_old.ms / max(1e-9, b_parse_pl.ms), digits=2))
end

function _dataset_section(tmpdir::AbstractString; reps::Int)
    println("\n=== Dataset/Pipeline JSON fast paths ===")
    rng = Random.MersenneTwister(0x51525354)

    pc = DT.PointCloud(randn(rng, 5000, 3))
    graph = _make_graph(5000, rng)
    spec = OPT.FiltrationSpec(kind=:rips, max_dim=1, radius=0.35,
                              poset_kind=:signature, axes_policy=:encoding)
    popts = OPT.PipelineOptions(poset_kind=:signature, axes_policy=:encoding)

    pc_old = joinpath(tmpdir, "pc_baseline.json")
    pc_col = joinpath(tmpdir, "pc_columnar.json")
    g_old = joinpath(tmpdir, "g_baseline.json")
    g_col = joinpath(tmpdir, "g_columnar.json")
    p_pc = joinpath(tmpdir, "pipeline_pointcloud.json")
    p_g = joinpath(tmpdir, "pipeline_graph.json")

    _save_dataset_json_baseline(pc_old, pc)
    SER.save_dataset_json(pc_col, pc; pretty=false)
    _save_dataset_json_baseline(g_old, graph)
    SER.save_dataset_json(g_col, graph; pretty=false)
    SER.save_pipeline_json(p_pc, pc, spec; degree=1, pipeline_opts=popts)
    SER.save_pipeline_json(p_g, graph, spec; degree=1, pipeline_opts=popts)

    b_save_pc_old = _bench("save PointCloud baseline",
                           () -> _save_dataset_json_baseline(pc_col, pc);
                           reps=reps)
    b_save_pc = _bench("save PointCloud columnar",
                       () -> SER.save_dataset_json(pc_col, pc; pretty=false);
                       reps=reps)
    b_save_g_old = _bench("save GraphData baseline",
                          () -> _save_dataset_json_baseline(g_col, graph);
                          reps=reps)
    b_save_g = _bench("save GraphData columnar",
                      () -> SER.save_dataset_json(g_col, graph; pretty=false);
                      reps=reps)
    b_pc_old = _bench("load PointCloud baseline", () -> _load_dataset_json_baseline(pc_old); reps=reps)
    b_pc_col_strict = _bench("load PointCloud columnar strict", () -> SER.load_dataset_json(pc_col; validation=:strict); reps=reps)
    b_pc_col_trusted = _bench("load PointCloud columnar trusted", () -> SER.load_dataset_json(pc_col; validation=:trusted); reps=reps)
    b_g_old = _bench("load GraphData baseline", () -> _load_dataset_json_baseline(g_old); reps=reps)
    b_g_col_strict = _bench("load GraphData columnar strict", () -> SER.load_dataset_json(g_col; validation=:strict); reps=reps)
    b_g_col_trusted = _bench("load GraphData columnar trusted", () -> SER.load_dataset_json(g_col; validation=:trusted); reps=reps)
    b_pipe_pc_strict = _bench("load pipeline(pointcloud) strict", () -> SER.load_pipeline_json(p_pc; validation=:strict); reps=reps)
    b_pipe_pc_trusted = _bench("load pipeline(pointcloud) trusted", () -> SER.load_pipeline_json(p_pc; validation=:trusted); reps=reps)
    b_pipe_g_strict = _bench("load pipeline(graph) strict", () -> SER.load_pipeline_json(p_g; validation=:strict); reps=reps)
    b_pipe_g_trusted = _bench("load pipeline(graph) trusted", () -> SER.load_pipeline_json(p_g; validation=:trusted); reps=reps)

    println("pointcloud file size columnar: ", filesize(pc_col), " bytes")
    println("graph file size columnar:      ", filesize(g_col), " bytes")
    println("PointCloud save speedup (new vs baseline): x",
            round(b_save_pc_old.ms / max(1e-9, b_save_pc.ms), digits=2))
    println("GraphData save speedup (new vs baseline): x",
            round(b_save_g_old.ms / max(1e-9, b_save_g.ms), digits=2))
    println("PointCloud strict load speedup (new vs baseline): x",
            round(b_pc_old.ms / max(1e-9, b_pc_col_strict.ms), digits=2))
    println("PointCloud trusted/strict speedup: x",
            round(b_pc_col_strict.ms / max(1e-9, b_pc_col_trusted.ms), digits=2))
    println("GraphData strict load speedup (new vs baseline): x",
            round(b_g_old.ms / max(1e-9, b_g_col_strict.ms), digits=2))
    println("GraphData trusted/strict speedup: x",
            round(b_g_col_strict.ms / max(1e-9, b_g_col_trusted.ms), digits=2))
    println("Pipeline(pointcloud) trusted/strict speedup: x",
            round(b_pipe_pc_strict.ms / max(1e-9, b_pipe_pc_trusted.ms), digits=2))
    println("Pipeline(graph) trusted/strict speedup: x",
            round(b_pipe_g_strict.ms / max(1e-9, b_pipe_g_trusted.ms), digits=2))
end

function main(args=ARGS)
    reps = _parse_int_arg(args, "--reps", 4)
    run_encoding = _parse_bool_arg(args, "--encoding", true)
    run_dataset = _parse_bool_arg(args, "--dataset", true)
    println("Serialization fast-path microbench (warm cached)")
    println("reps=", reps)
    println("encoding=", run_encoding, ", dataset=", run_dataset)
    tmpdir = mktempdir(prefix="serialization_fastpath_")
    println("tmpdir=", tmpdir)
    try
        run_encoding && _encoding_section(tmpdir; reps=reps)
        run_dataset && _dataset_section(tmpdir; reps=reps)
    finally
        rm(tmpdir; recursive=true, force=true)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
