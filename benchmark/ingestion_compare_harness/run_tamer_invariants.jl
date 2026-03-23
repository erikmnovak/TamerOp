#!/usr/bin/env julia

using DelimitedFiles
using Statistics
using TOML
using Dates

const _FALLBACK_SOURCE_MODE = let source_mode = false
    try
        @eval using TamerOp
    catch
        include(joinpath(@__DIR__, "..", "..", "src", "TamerOp.jl"))
        @eval using .TamerOp
        source_mode = true
    end
    source_mode
end

const _HAVE_NEARESTNEIGHBORS = let ok = false
    try
        @eval using NearestNeighbors
        ok = true
    catch
        ok = false
    end
    ok
end

if _FALLBACK_SOURCE_MODE && _HAVE_NEARESTNEIGHBORS
    include(joinpath(@__DIR__, "..", "..", "ext", "TamerOpNearestNeighborsExt.jl"))
end

const DI = TamerOp.DataIngestion
const EC = TamerOp.EncodingCore
const SM = TamerOp.SignedMeasures
const SI = TamerOp.SliceInvariants
const Opt = TamerOp.Options

const _DEFAULT_MANIFEST = joinpath(@__DIR__, "fixtures_invariants", "manifest.toml")
const _DEFAULT_OUT = joinpath(@__DIR__, "results_tamer_invariants.csv")
const _DEFAULT_INVARIANTS = (:euler_signed_measure,)

@inline function _arg(args::Vector{String}, key::String, default::String)
    prefix = key * "="
    for a in args
        startswith(a, prefix) || continue
        return split(a, "=", limit=2)[2]
    end
    return default
end

@inline _arg_int(args::Vector{String}, key::String, default::Int) = parse(Int, _arg(args, key, string(default)))

@inline function _profile_defaults(profile::Symbol)
    if profile == :desktop
        return (reps=4, trim_between_reps=true, trim_between_cases=true)
    elseif profile == :balanced
        return (reps=5, trim_between_reps=false, trim_between_cases=true)
    elseif profile == :stress
        return (reps=9, trim_between_reps=false, trim_between_cases=false)
    elseif profile == :probe
        return (reps=3, trim_between_reps=false, trim_between_cases=true)
    end
    error("--profile must be one of: desktop, balanced, stress, probe")
end

@inline function _memory_relief!()
    GC.gc()
    GC.gc(true)
    try
        Base.Libc.malloc_trim(0)
    catch
    end
    return nothing
end

function _load_point_cloud(path::String)
    raw = readdlm(path, ',', Float64)
    mat = if raw isa Matrix{Float64}
        raw
    elseif raw isa Vector{Float64}
        reshape(raw, :, 1)
    else
        Matrix{Float64}(raw)
    end
    pts = [Vector{Float64}(view(mat, i, :)) for i in 1:size(mat, 1)]
    return TamerOp.PointCloud(pts)
end

function _load_image(path::String)
    raw = readdlm(path, ',', Float64)
    mat = if raw isa Matrix{Float64}
        raw
    elseif raw isa Vector{Float64}
        reshape(raw, :, 1)
    else
        Matrix{Float64}(raw)
    end
    return TamerOp.ImageNd(Matrix{Float64}(mat))
end

function _load_case_data(case::AbstractDict{String,<:Any}, path::String)
    dataset = string(case["dataset"])
    if dataset == "gaussian_shell"
        return _load_point_cloud(path)
    elseif dataset == "image_sine"
        return _load_image(path)
    end
    error("Unsupported dataset: $(dataset)")
end

function _encoding_spec_for_case(case::AbstractDict{String,<:Any}; prefer_nn::Bool=true)
    regime = string(case["regime"])
    max_dim = Int(case["max_dim"])
    if regime == "degree_rips_parity"
        degree_radius = Float64(case["degree_radius"])
        return TamerOp.FiltrationSpec(
            kind=:degree_rips,
            max_dim=max_dim,
            radius=degree_radius,
            construction=TamerOp.ConstructionOptions(
                sparsify=:radius,
                collapse=:none,
                output_stage=:encoding_result,
            ),
        )
    elseif regime == "rips_codensity_parity"
        codensity_radius = Float64(case["codensity_radius"])
        codensity_dtm_mass = Float64(case["codensity_dtm_mass"])
        return TamerOp.FiltrationSpec(
            kind=:rips_codensity,
            max_dim=max_dim,
            radius=codensity_radius,
            dtm_mass=codensity_dtm_mass,
            construction=TamerOp.ConstructionOptions(
                sparsify=:radius,
                collapse=:none,
                output_stage=:encoding_result,
            ),
        )
    elseif regime == "rips_lowerstar_parity"
        lowerstar_radius = Float64(case["lowerstar_radius"])
        return TamerOp.FiltrationSpec(
            kind=:rips_lowerstar,
            max_dim=max_dim,
            radius=lowerstar_radius,
            coord=1,
            construction=TamerOp.ConstructionOptions(
                sparsify=:radius,
                collapse=:none,
                output_stage=:encoding_result,
            ),
        )
    elseif regime == "delaunay_lowerstar_parity"
        return TamerOp.FiltrationSpec(
            kind=:function_delaunay,
            max_dim=max_dim,
            vertex_function=(p, _)->Float64(p[1]),
            simplex_agg=:max,
            construction=TamerOp.ConstructionOptions(output_stage=:encoding_result),
        )
    elseif regime == "alpha_parity"
        return TamerOp.FiltrationSpec(
            kind=:alpha,
            max_dim=max_dim,
            construction=TamerOp.ConstructionOptions(output_stage=:encoding_result),
        )
    elseif regime == "core_delaunay_parity"
        return TamerOp.FiltrationSpec(
            kind=:core_delaunay,
            max_dim=max_dim,
            construction=TamerOp.ConstructionOptions(output_stage=:encoding_result),
        )
    elseif regime == "cubical_parity"
        return TamerOp.FiltrationSpec(
            kind=:cubical,
            construction=TamerOp.ConstructionOptions(output_stage=:encoding_result),
        )
    elseif regime == "claim_matching"
        claim_radius = Float64(case["claim_radius"])
        nn_backend = prefer_nn ? :nearestneighbors : :bruteforce
        return TamerOp.FiltrationSpec(
            kind=:rips,
            max_dim=max_dim,
            radius=claim_radius,
            nn_backend=nn_backend,
            construction=TamerOp.ConstructionOptions(
                sparsify=:radius,
                collapse=:none,
                output_stage=:encoding_result,
            ),
        )
    elseif regime == "rips_parity"
        parity_radius = Float64(case["parity_radius"])
        return TamerOp.FiltrationSpec(
            kind=:rips,
            max_dim=max_dim,
            radius=parity_radius,
            construction=TamerOp.ConstructionOptions(
                sparsify=:radius,
                collapse=:none,
                output_stage=:encoding_result,
            ),
        )
    elseif regime == "landmark_parity"
        landmarks = Int[Int(v) for v in case["landmarks"]]
        landmark_radius = Float64(case["landmark_radius"])
        return TamerOp.FiltrationSpec(
            kind=:landmark_rips,
            max_dim=max_dim,
            landmarks=landmarks,
            radius=landmark_radius,
            construction=TamerOp.ConstructionOptions(output_stage=:encoding_result),
        )
    else
        error("Unsupported regime: $(regime)")
    end
end

@inline function _parse_invariants(raw::AbstractString)
    s = lowercase(strip(raw))
    if s == "all" || isempty(s)
        return nothing
    end
    out = Symbol[]
    for part in split(s, ',')
        token = replace(lowercase(strip(part)), '-' => '_')
        if token == "rank" || token == "rank_signed_measure"
            push!(out, :rank_signed_measure)
        elseif token == "slice" || token == "slice_barcodes"
            push!(out, :slice_barcodes)
        elseif token == "euler" || token == "euler_signed_measure"
            push!(out, :euler_signed_measure)
        else
            error("--invariants contains unsupported token $(part)")
        end
    end
    isempty(out) && error("--invariants selected no invariants.")
    return unique(out)
end

function _approved_invariants_for_regime(raw::Dict, regime::AbstractString)
    tbl = get(raw, "invariant_eligibility", nothing)
    tbl === nothing && return nothing
    tbl isa Dict || error("manifest invariant_eligibility must be a table.")
    vals = get(tbl, regime, nothing)
    vals === nothing && return nothing
    vals isa AbstractVector || error("manifest invariant_eligibility[$regime] must be an array.")
    out = Symbol[]
    for v in vals
        token = replace(lowercase(strip(String(v))), '-' => '_')
        if token == "rank" || token == "rank_signed_measure"
            push!(out, :rank_signed_measure)
        elseif token == "slice" || token == "slice_barcodes"
            push!(out, :slice_barcodes)
        elseif token == "euler" || token == "euler_signed_measure"
            push!(out, :euler_signed_measure)
        else
            error("Unsupported manifest-approved invariant token $(v) for regime $(regime).")
        end
    end
    return unique(out)
end

function _selected_invariants_for_case(requested, approved)
    if approved === nothing
        return requested === nothing ? collect(_DEFAULT_INVARIANTS) : requested
    end
    if requested === nothing
        return approved
    end
    approved_set = Set(approved)
    return [inv for inv in requested if inv in approved_set]
end

@inline function _weight_token(w)
    if w isa Integer
        return string(w)
    elseif w isa Rational
        return string(numerator(w), "//", denominator(w))
    end
    return repr(w)
end

@inline _coord_token(x::Real) = repr(Float64(x))
@inline _coord_key(coords::Vector{Float64}) = join((_coord_token(x) for x in coords), "|")
@inline _coord_key_tuple(coords) = join((_coord_token(x) for x in coords), "|")

function _slice_query_from_case(case::AbstractDict{String,<:Any})
    haskey(case, "slice_directions") || error("slice_barcodes requires manifest slice_directions")
    haskey(case, "slice_offsets") || error("slice_barcodes requires manifest slice_offsets")
    dirs_raw = case["slice_directions"]
    offs_raw = case["slice_offsets"]
    dirs = [Float64[Float64(v) for v in dir] for dir in dirs_raw]
    offs = [Float64[Float64(v) for v in off] for off in offs_raw]
    return dirs, offs
end

function _canonical_measure_terms_raw(raw_coords::Vector{Vector{Float64}}, raw_wts)
    terms = [(raw_coords[i], raw_wts[i]) for i in eachindex(raw_wts)]
    sort!(terms; by=t -> _coord_key(t[1]))
    merged = Vector{Tuple{Vector{Float64},Any}}()
    for (coords, wt) in terms
        if !isempty(merged) && merged[end][1] == coords
            merged[end] = (coords, merged[end][2] + wt)
        else
            push!(merged, (coords, wt))
        end
    end
    filter!(t -> !iszero(t[2]), merged)
    return merged
end

function _canonical_measure_terms(out)
    if hasproperty(out, :barcode) && hasproperty(out, :semantic_axes_birth) && hasproperty(out, :semantic_axes_death)
        sb = getproperty(out, :barcode)
        sem_axes_birth = getproperty(out, :semantic_axes_birth)
        sem_axes_death = getproperty(out, :semantic_axes_death)
        rects = getproperty(sb, :rects)
        wts = getproperty(sb, :weights)
        grid_axes = getproperty(sb, :axes)
        axis_birth_maps = ntuple(d -> Dict{Int,Float64}(grid_axes[d][j] => Float64(sem_axes_birth[d][j]) for j in eachindex(grid_axes[d])), length(sem_axes_birth))
        axis_death_maps = ntuple(d -> Dict{Int,Float64}(grid_axes[d][j] => Float64(sem_axes_death[d][j]) for j in eachindex(grid_axes[d])), length(sem_axes_death))
        raw_coords = Vector{Vector{Float64}}(undef, length(wts))
        for i in eachindex(wts)
            rect = rects[i]
            raw_coords[i] = Float64[
                (axis_birth_maps[d][rect.lo[d]] for d in eachindex(sem_axes_birth))...;
                (axis_death_maps[d][rect.hi[d]] for d in eachindex(sem_axes_death))...;
            ]
        end
        return _canonical_measure_terms_raw(raw_coords, wts)
    elseif hasproperty(out, :barcode) && hasproperty(out, :semantic_axes)
        sb = getproperty(out, :barcode)
        sem_axes = getproperty(out, :semantic_axes)
        sem_axes === nothing && (sem_axes = getproperty(sb, :axes))
        rects = getproperty(sb, :rects)
        wts = getproperty(sb, :weights)
        grid_axes = getproperty(sb, :axes)
        axis_maps = ntuple(d -> Dict{Int,Float64}(grid_axes[d][j] => Float64(sem_axes[d][j]) for j in eachindex(grid_axes[d])), length(sem_axes))
        raw_coords = Vector{Vector{Float64}}(undef, length(wts))
        for i in eachindex(wts)
            rect = rects[i]
            raw_coords[i] = Float64[
                (axis_maps[d][rect.lo[d]] for d in eachindex(sem_axes))...;
                (axis_maps[d][rect.hi[d]] for d in eachindex(sem_axes))...;
            ]
        end
        return _canonical_measure_terms_raw(raw_coords, wts)
    elseif hasproperty(out, :axes) && hasproperty(out, :inds) && hasproperty(out, :wts)
        axes = getproperty(out, :axes)
        inds = getproperty(out, :inds)
        wts = getproperty(out, :wts)
        raw_coords = Vector{Vector{Float64}}(undef, length(wts))
        for i in eachindex(wts)
            raw_coords[i] = [Float64(axes[d][inds[i][d]]) for d in eachindex(axes)]
        end
        return _canonical_measure_terms_raw(raw_coords, wts)
    elseif hasproperty(out, :rects) && hasproperty(out, :weights)
        rects = getproperty(out, :rects)
        wts = getproperty(out, :weights)
        raw_coords = Vector{Vector{Float64}}(undef, length(wts))
        for i in eachindex(wts)
            rect = rects[i]
            raw_coords[i] = Float64[rect.lo...; rect.hi...]
        end
        return _canonical_measure_terms_raw(raw_coords, wts)
    end
    error("Unsupported invariant output type $(typeof(out))")
end

@inline function _serialize_measure_terms(terms)
    return join((_coord_key(coords) * "=>" * _weight_token(wt) for (coords, wt) in terms), ";")
end

@inline _abs_mass_terms(terms) = sum(Float64(abs(wt)) for (_, wt) in terms)

@inline function _slice_tail_value(tvals::AbstractVector{<:Real})
    m = length(tvals)
    m >= 1 || error("_slice_tail_value requires nonempty tvals")
    if m == 1
        return Float64(tvals[1] + one(tvals[1]))
    end
    step = tvals[end] - tvals[end - 1]
    iszero(step) && (step = one(step))
    return Float64(tvals[end] + step)
end

function _canonical_barcode_terms(bar, tail::Union{Nothing,Float64}=nothing)
    terms0 = Tuple{Tuple{Float64,Float64},Int}[]
    for (interval, mult) in pairs(bar)
        b = Float64(interval[1])
        d = Float64(interval[2])
        if tail !== nothing && isfinite(tail) && isfinite(d) && d == tail
            d = Inf
        end
        push!(terms0, ((b, d), Int(mult)))
    end
    sort!(terms0; by=t -> (_coord_token(t[1][1]), _coord_token(t[1][2])))
    merged = Tuple{Tuple{Float64,Float64},Int}[]
    for (interval, mult) in terms0
        if !isempty(merged) && merged[end][1] == interval
            merged[end] = (interval, merged[end][2] + mult)
        else
            push!(merged, (interval, mult))
        end
    end
    filter!(t -> t[2] != 0, merged)
    return merged
end

@inline function _serialize_barcode_terms(terms)
    return join((_coord_key_tuple((interval[1], interval[2])) * "=>" * _weight_token(mult)
                 for (interval, mult) in terms), ";")
end

@inline function _serialize_slice_record(dir, off, terms)
    return "dir=" * _coord_key_tuple(dir) * "@@off=" * _coord_key_tuple(off) * "@@bars=" * _serialize_barcode_terms(terms)
end

function _slice_output_contract(out)
    payload = hasproperty(out, :slice_result) ? getproperty(out, :slice_result) : out
    enc = hasproperty(out, :encoding_result) ? getproperty(out, :encoding_result) : nothing
    exact_line = hasproperty(out, :exact_line_persistence) && Bool(getproperty(out, :exact_line_persistence))
    pi0 = enc === nothing ? nothing : TamerOp.encoding_map(enc)
    raw_pi = pi0 isa EC.CompiledEncoding ? TamerOp.encoding_map(pi0) : pi0
    dirs = hasproperty(payload, :dirs) ? getproperty(payload, :dirs) : SI.slice_directions(payload)
    offs = hasproperty(payload, :offs) ? getproperty(payload, :offs) : SI.slice_offsets(payload)
    bars = SI.slice_barcodes(payload)
    opts = Opt.InvariantOptions(box=:auto)
    records = String[]
    term_count = 0
    abs_mass = 0.0
    for i in axes(bars, 1), j in axes(bars, 2)
        tail = nothing
        if !exact_line && raw_pi !== nothing
            chain, tvals = SI.slice_chain(raw_pi, offs[j], dirs[i], opts; drop_unknown=true)
            if !isempty(chain) && !isempty(tvals)
                tail = _slice_tail_value(tvals)
            end
        end
        terms = _canonical_barcode_terms(bars[i, j], tail)
        term_count += length(terms)
        abs_mass += isempty(terms) ? 0.0 : sum(abs(Float64(mult)) for (_, mult) in terms)
        push!(records, _serialize_slice_record(dirs[i], offs[j], terms))
    end
    return term_count, abs_mass, join(records, "###")
end

@inline function _slice_exact_line_endpoints_grid1d(
    case::AbstractDict{String,<:Any},
    axis::AbstractVector{<:Real},
    x0::Float64,
    dir::Float64,
)
    dir > 0.0 || error("slice_barcodes exact 1D path currently requires positive direction.")
    m = length(axis)
    m >= 1 || error("slice_barcodes exact 1D path requires a nonempty axis.")
    regime = String(case["regime"])
    lower0 = regime == "rips_parity" ? 0.0 : Float64(first(axis))
    vals = Vector{Float64}(undef, m + 1)
    vals[1] = (lower0 - x0) / dir
    @inbounds for i in 2:m
        vals[i] = (Float64(axis[i]) - x0) / dir
    end
    vals[end] = Inf
    return vals
end

@inline function _serialize_rank_axes(axes)
    return join((_coord_key_tuple(axis) for axis in axes), ";;")
end

function _serialize_rank_table(axes, table)
    nd = length(axes)
    dims = ntuple(i -> length(axes[i]), nd)
    entries = String[]
    for pCI in CartesianIndices(dims)
        p = pCI.I
        q_ranges = ntuple(k -> p[k]:dims[k], nd)
        for qCI in CartesianIndices(q_ranges)
            q = qCI.I
            val = @inbounds table[pCI, qCI]
            iszero(val) && continue
            pcoords = ntuple(k -> Float64(axes[k][p[k]]), nd)
            qcoords = ntuple(k -> Float64(axes[k][q[k]]), nd)
            push!(entries, _coord_key_tuple(pcoords) * "||" * _coord_key_tuple(qcoords) * "=>" * _weight_token(val))
        end
    end
    return join(entries, ";")
end

function _rank_output_contract(out)
    if hasproperty(out, :rank_query_axes) && hasproperty(out, :rank_table_canonical)
        axes = getproperty(out, :rank_query_axes)
        table = getproperty(out, :rank_table_canonical)
        return _serialize_rank_axes(axes), String(table)
    end
    if hasproperty(out, :axes) && hasproperty(out, :rects) && hasproperty(out, :weights)
        axes = getproperty(out, :axes)
        table = SM.rectangle_signed_barcode_rank(out; zero_noncomparable=true, threads=false)
        return _serialize_rank_axes(axes), _serialize_rank_table(axes, table)
    end
    return "", ""
end

@inline function _run_rank_signed_measure_uncached(data, spec, degree::Int)
    session_cache = TamerOp.SessionCache()
    enc = DI.encode(data, spec; degree=degree, stage=:encoding_result, cache=session_cache)
    sb = TamerOp.rectangle_signed_barcode(enc; opts=Opt.InvariantOptions(), cache=session_cache)
    pi0 = TamerOp.encoding_map(enc)
    raw_pi = pi0 isa EC.CompiledEncoding ? TamerOp.encoding_map(pi0) : pi0
    if raw_pi isa EC.GridEncodingMap
        birth_axes, death_axes = SM._rectangle_signed_barcode_grid_semantic_axes(raw_pi, Opt.InvariantOptions(); keep_endpoints=true)
        rank_table = SM.rectangle_signed_barcode_rank(sb; zero_noncomparable=true, threads=false)
        return (
            ; barcode=sb,
              semantic_axes_birth=birth_axes,
              semantic_axes_death=death_axes,
              rank_query_axes=birth_axes,
              rank_table_canonical=_serialize_rank_table(birth_axes, rank_table),
        )
    end
    return sb
end

function _run_slice_barcodes_exact_grid1d(enc, case)
    directions, offsets = _slice_query_from_case(case)
    pi0 = TamerOp.encoding_map(enc)
    raw_pi = pi0 isa EC.CompiledEncoding ? TamerOp.encoding_map(pi0) : pi0
    raw_pi isa EC.GridEncodingMap{1} || error("exact 1D slice path requires GridEncodingMap{1}")
    axis = raw_pi.coords[1]
    chain = collect(1:length(axis))
    M = TamerOp.Workflow.pmodule(enc)
    nd = length(directions)
    no = length(offsets)
    bars = Matrix{Dict{Tuple{Float64,Float64},Int}}(undef, nd, no)
    for i in 1:nd
        dir = directions[i]
        length(dir) == 1 || error("exact 1D slice path expects 1D directions.")
        d = Float64(dir[1])
        for j in 1:no
            off = offsets[j]
            length(off) == 1 || error("exact 1D slice path expects 1D offsets.")
            x0 = Float64(off[1])
            vals = _slice_exact_line_endpoints_grid1d(case, axis, x0, d)
            bars[i, j] = SI.slice_barcode(M, chain; values=vals, check_chain=false)
        end
    end
    weights = ones(Float64, nd, no)
    return SI.SliceBarcodesResult(bars, weights, directions, offsets)
end

function _run_slice_barcodes_exact_grid2d(enc, case)
    directions, offsets = _slice_query_from_case(case)
    pi0 = TamerOp.encoding_map(enc)
    raw_pi = pi0 isa EC.CompiledEncoding ? TamerOp.encoding_map(pi0) : pi0
    raw_pi isa EC.GridEncodingMap{2} || error("exact 2D slice path requires GridEncodingMap{2}")
    M = TamerOp.Workflow.pmodule(enc)
    nd = length(directions)
    no = length(offsets)
    bars = Matrix{Dict{Tuple{Float64,Float64},Int}}(undef, nd, no)
    for i in 1:nd
        dir = Float64[Float64(v) for v in directions[i]]
        length(dir) == 2 || error("exact 2D slice path expects 2D directions.")
        n1 = -dir[2]
        n2 = dir[1]
        for j in 1:no
            off_entry = offsets[j]
            off = if length(off_entry) == 1
                Float64(off_entry[1])
            elseif length(off_entry) == 2
                x0 = Float64[Float64(v) for v in off_entry]
                n1 * x0[1] + n2 * x0[2]
            else
                error("exact 2D slice path expects scalar offsets or 2D basepoints.")
            end
            chain, vals = TamerOp.Fibered2D.slice_chain_exact_2d(
                raw_pi,
                dir,
                off,
                Opt.InvariantOptions(box=:auto);
                normalize_dirs=:none,
            )
            bars[i, j] = isempty(chain) ? Dict{Tuple{Float64,Float64},Int}() :
                SI.slice_barcode(M, chain; values=vals, check_chain=false)
        end
    end
    weights = ones(Float64, nd, no)
    return SI.SliceBarcodesResult(bars, weights, directions, offsets)
end

@inline function _run_slice_barcodes_uncached(data, spec, case, degree::Int)
    session_cache = TamerOp.SessionCache()
    enc = DI.encode(data, spec; degree=degree, stage=:encoding_result, cache=session_cache)
    pi0 = TamerOp.encoding_map(enc)
    raw_pi = pi0 isa EC.CompiledEncoding ? TamerOp.encoding_map(pi0) : pi0
    if raw_pi isa EC.GridEncodingMap{1}
        res = _run_slice_barcodes_exact_grid1d(enc, case)
        return (; slice_result=res, encoding_result=enc, exact_line_persistence=true)
    elseif raw_pi isa EC.GridEncodingMap{2}
        res = _run_slice_barcodes_exact_grid2d(enc, case)
        return (; slice_result=res, encoding_result=enc, exact_line_persistence=true)
    end
    directions, offsets = _slice_query_from_case(case)
    res = TamerOp.slice_barcodes(
        enc;
        opts=Opt.InvariantOptions(),
        cache=session_cache,
        directions=directions,
        offsets=offsets,
        normalize_weights=false,
        threads=false,
        packed=false,
    )
    return (; slice_result=res, encoding_result=enc, exact_line_persistence=false)
end

@inline function _run_euler_signed_measure_uncached(data, spec, degree::Int)
    session_cache = TamerOp.SessionCache()
    enc = DI.encode(data, spec; degree=degree, stage=:encoded_complex, cache=session_cache)
    return TamerOp.euler_signed_measure(enc; opts=Opt.InvariantOptions(), cache=session_cache)
end

@inline function _invariant_stage_note(invariant_kind::Symbol)
    if invariant_kind === :rank_signed_measure
        return "cold_mode=warm_uncached;cache=per_call_session;stage=encoding_result_to_rectangle_signed_barcode"
    elseif invariant_kind === :slice_barcodes
        return "cold_mode=warm_uncached;cache=per_call_session;stage=encoding_result_to_slice_barcodes"
    elseif invariant_kind === :euler_signed_measure
        return "cold_mode=warm_uncached;cache=per_call_session;stage=encoded_complex_to_signed_measure"
    end
    error("Unsupported invariant kind $(invariant_kind)")
end

@inline function _run_invariant_uncached(data, spec, invariant_kind::Symbol, degree::Int)
    if invariant_kind === :rank_signed_measure
        return _run_rank_signed_measure_uncached(data, spec, degree)
    elseif invariant_kind === :slice_barcodes
        error("_run_invariant_uncached(slice_barcodes): case-specific query metadata required")
    elseif invariant_kind === :euler_signed_measure
        return _run_euler_signed_measure_uncached(data, spec, degree)
    end
    error("Unsupported invariant kind $(invariant_kind)")
end

function _timed_invariant(data, spec, case, invariant_kind::Symbol, degree::Int)
    runner = invariant_kind === :slice_barcodes ?
        () -> _run_slice_barcodes_uncached(data, spec, case, degree) :
        () -> _run_invariant_uncached(data, spec, invariant_kind, degree)
    m = @timed out = runner()
    return out, 1000.0 * m.time, m.bytes / 1024.0
end

@inline _p90(v::Vector{Float64}) = isempty(v) ? NaN : sort(v)[max(1, ceil(Int, 0.9 * length(v)))]

function _csv_escape(x)
    s = string(x)
    if occursin(',', s) || occursin('"', s)
        s = replace(s, '"' => "\"\"")
        return "\"" * s * "\""
    end
    return s
end

function _write_csv(path::AbstractString, rows::Vector{NamedTuple})
    open(path, "w") do io
        println(io, "tool,case_id,regime,invariant_kind,degree,n_points,ambient_dim,max_dim,cold_ms,cold_alloc_kib,warm_median_ms,warm_p90_ms,warm_alloc_median_kib,output_term_count,output_abs_mass,output_measure_canonical,output_rank_query_axes_canonical,output_rank_table_canonical,notes,timestamp_utc")
        for r in rows
            vals = (
                r.tool, r.case_id, r.regime, r.invariant_kind, r.degree, r.n_points,
                r.ambient_dim, r.max_dim, r.cold_ms, r.cold_alloc_kib,
                r.warm_median_ms, r.warm_p90_ms, r.warm_alloc_median_kib,
                r.output_term_count, r.output_abs_mass, r.output_measure_canonical,
                r.output_rank_query_axes_canonical, r.output_rank_table_canonical,
                r.notes, r.timestamp_utc,
            )
            println(io, join(_csv_escape.(vals), ","))
        end
    end
end

function main()
    args = copy(ARGS)
    profile = Symbol(lowercase(_arg(args, "--profile", "desktop")))
    defaults = _profile_defaults(profile)
    manifest_path = _arg(args, "--manifest", _DEFAULT_MANIFEST)
    out_path = _arg(args, "--out", _DEFAULT_OUT)
    reps = _arg_int(args, "--reps", defaults.reps)
    degree = _arg_int(args, "--degree", 0)
    regime_filter = _arg(args, "--regime", "all")
    case_filter = _arg(args, "--case", "")
    invariants = _parse_invariants(_arg(args, "--invariants", "all"))

    reps >= 1 || error("--reps must be >= 1.")

    println("[profile] ", profile, " (reps=", reps, ", trim_between_reps=", defaults.trim_between_reps, ", trim_between_cases=", defaults.trim_between_cases, ")")
    println("NearestNeighbors package loaded: ", _HAVE_NEARESTNEIGHBORS)
    println("Invariants: ", invariants === nothing ? :all : invariants)

    raw = TOML.parsefile(manifest_path)
    cases = get(raw, "cases", Any[])
    rows = NamedTuple[]

    for c in cases
        case_id = string(c["id"])
        regime = string(c["regime"])
        regime_filter == "all" || regime == regime_filter || continue
        isempty(case_filter) || case_id == case_filter || continue

        approved_invariants = _approved_invariants_for_regime(raw, regime)
        selected_invariants = _selected_invariants_for_case(invariants, approved_invariants)
        isempty(selected_invariants) && begin
            println("[skip] ", case_id, ": no benchmark-approved invariants for regime=", regime)
            continue
        end

        fixture = joinpath(dirname(manifest_path), string(c["path"]))
        n_points = Int(c["n_points"])
        ambient_dim = Int(c["ambient_dim"])
        max_dim = Int(c["max_dim"])
        data = _load_case_data(c, fixture)

        spec = _encoding_spec_for_case(c; prefer_nn=true)
        for invariant_kind in selected_invariants
            notes = _invariant_stage_note(invariant_kind)
            cold_out = nothing
            cold_ms = NaN
            cold_alloc = NaN
            try
                if invariant_kind === :slice_barcodes
                    _run_slice_barcodes_uncached(data, spec, c, degree)
                else
                    _run_invariant_uncached(data, spec, invariant_kind, degree)
                end
                _memory_relief!()
                cold_out, cold_ms, cold_alloc = _timed_invariant(data, spec, c, invariant_kind, degree)
            catch err
                if regime == "claim_matching"
                    spec = _encoding_spec_for_case(c; prefer_nn=false)
                    notes = string(notes, ";nearestneighbors_unavailable_fallback_bruteforce")
                    try
                        if invariant_kind === :slice_barcodes
                            _run_slice_barcodes_uncached(data, spec, c, degree)
                        else
                            _run_invariant_uncached(data, spec, invariant_kind, degree)
                        end
                        _memory_relief!()
                        cold_out, cold_ms, cold_alloc = _timed_invariant(data, spec, c, invariant_kind, degree)
                    catch inner_err
                        println("[skip] ", case_id, " ", invariant_kind, ": ", sprint(showerror, inner_err))
                        continue
                    end
                else
                    println("[skip] ", case_id, " ", invariant_kind, ": ", sprint(showerror, err))
                    continue
                end
            end

            warm_times = Float64[]
            warm_allocs = Float64[]
            for _ in 1:reps
                _, tms, akib = _timed_invariant(data, spec, c, invariant_kind, degree)
                push!(warm_times, tms)
                push!(warm_allocs, akib)
                defaults.trim_between_reps && _memory_relief!()
            end

            warm_median_ms = median(warm_times)
            warm_p90_ms = _p90(warm_times)
            warm_alloc_median = median(warm_allocs)
            if invariant_kind === :slice_barcodes
                output_term_count, output_abs_mass, output_measure_canonical = _slice_output_contract(cold_out)
            else
                output_terms = _canonical_measure_terms(cold_out)
                output_term_count = length(output_terms)
                output_abs_mass = _abs_mass_terms(output_terms)
                output_measure_canonical = _serialize_measure_terms(output_terms)
            end
            output_rank_query_axes_canonical, output_rank_table_canonical = invariant_kind === :rank_signed_measure ?
                _rank_output_contract(cold_out) : ("", "")

            println(
                rpad(case_id, 32),
                " inv=", invariant_kind,
                " cold_ms=", round(cold_ms, digits=3),
                " warm_med_ms=", round(warm_median_ms, digits=3),
                " terms=", output_term_count,
            )

            push!(rows, (
                tool="tamer_op",
                case_id=case_id,
                regime=regime,
                invariant_kind=String(invariant_kind),
                degree=degree,
                n_points=n_points,
                ambient_dim=ambient_dim,
                max_dim=max_dim,
                cold_ms=cold_ms,
                cold_alloc_kib=cold_alloc,
                warm_median_ms=warm_median_ms,
                warm_p90_ms=warm_p90_ms,
                warm_alloc_median_kib=warm_alloc_median,
                output_term_count=output_term_count,
                output_abs_mass=output_abs_mass,
                output_measure_canonical=output_measure_canonical,
                output_rank_query_axes_canonical=output_rank_query_axes_canonical,
                output_rank_table_canonical=output_rank_table_canonical,
                notes=notes,
                timestamp_utc=string(Dates.now(Dates.UTC)),
            ))
            defaults.trim_between_cases && _memory_relief!()
        end
    end

    isempty(rows) && error("No invariant benchmark rows were produced. manifest=$(manifest_path), regime=$(regime_filter), case=$(case_filter), invariants=$(invariants)")
    _write_csv(out_path, rows)
    println("Wrote tamer invariant results: $(out_path)")
end

main()
