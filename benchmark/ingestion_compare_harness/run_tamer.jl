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
    # In source-include mode Julia package extensions are not auto-activated.
    # Explicitly load the NN extension so :nearestneighbors backend is available.
    include(joinpath(@__DIR__, "..", "..", "ext", "TamerOpNearestNeighborsExt.jl"))
end

const _DEFAULT_MANIFEST = joinpath(@__DIR__, "fixtures", "manifest.toml")
const _DEFAULT_OUT = joinpath(@__DIR__, "results_tamer.csv")

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
        # Desktop-safe default: slightly lower reps + aggressive memory trimming.
        return (reps=4, trim_between_reps=true, trim_between_cases=true)
    elseif profile == :balanced
        # Balanced local runs with more stable medians.
        return (reps=5, trim_between_reps=false, trim_between_cases=true)
    elseif profile == :stress
        # Heavier sampling for publication-grade runs.
        return (reps=9, trim_between_reps=false, trim_between_cases=false)
    elseif profile == :probe
        # Quick probe for fast feedback.
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
        # Non-glibc platforms may not provide malloc_trim.
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

function _spec_for_case(case::AbstractDict{String,<:Any}; prefer_nn::Bool=true)
    regime = string(case["regime"])
    max_dim = Int(case["max_dim"])
    if regime == "claim_matching"
        haskey(case, "claim_radius") || error("claim_matching case requires claim_radius in manifest.")
        claim_radius = Float64(case["claim_radius"])
        nn_backend = prefer_nn ? :nearestneighbors : :bruteforce
        return TamerOp.FiltrationSpec(
            kind=:rips,
            max_dim=max_dim,
            radius=claim_radius,
            nn_backend=nn_backend,
            construction=TamerOp.ConstructionOptions(
                ;
                sparsify=:radius,
                collapse=:none,
                output_stage=:simplex_tree,
            ),
        )
    elseif regime == "rips_parity"
        if haskey(case, "parity_radius")
            parity_radius = Float64(case["parity_radius"])
            return TamerOp.FiltrationSpec(
                kind=:rips,
                max_dim=max_dim,
                radius=parity_radius,
                construction=TamerOp.ConstructionOptions(
                    ;
                    sparsify=:radius,
                    collapse=:none,
                    output_stage=:simplex_tree,
                ),
            )
        end
        return TamerOp.FiltrationSpec(
            kind=:rips,
            max_dim=max_dim,
            construction=TamerOp.ConstructionOptions(
                ;
                sparsify=:none,
                collapse=:none,
                output_stage=:simplex_tree,
            ),
        )
    elseif regime == "degree_rips_parity"
        haskey(case, "degree_radius") || error("degree_rips_parity case requires degree_radius in manifest.")
        degree_radius = Float64(case["degree_radius"])
        return TamerOp.FiltrationSpec(
            kind=:degree_rips,
            max_dim=max_dim,
            radius=degree_radius,
            construction=TamerOp.ConstructionOptions(
                ;
                sparsify=:radius,
                collapse=:none,
                output_stage=:simplex_tree,
            ),
        )
    elseif regime == "rips_codensity_parity"
        haskey(case, "codensity_radius") || error("rips_codensity_parity case requires codensity_radius in manifest.")
        codensity_radius = Float64(case["codensity_radius"])
        codensity_dtm_mass = Float64(case["codensity_dtm_mass"])
        return TamerOp.FiltrationSpec(
            kind=:rips_codensity,
            max_dim=max_dim,
            radius=codensity_radius,
            dtm_mass=codensity_dtm_mass,
            construction=TamerOp.ConstructionOptions(
                ;
                sparsify=:radius,
                collapse=:none,
                output_stage=:simplex_tree,
            ),
        )
    elseif regime == "rips_lowerstar_parity"
        haskey(case, "lowerstar_radius") || error("rips_lowerstar_parity case requires lowerstar_radius in manifest.")
        lowerstar_radius = Float64(case["lowerstar_radius"])
        return TamerOp.FiltrationSpec(
            kind=:rips_lowerstar,
            max_dim=max_dim,
            radius=lowerstar_radius,
            coord=1,
            construction=TamerOp.ConstructionOptions(
                ;
                sparsify=:radius,
                collapse=:none,
                output_stage=:simplex_tree,
            ),
        )
    elseif regime == "delaunay_lowerstar_parity"
        return TamerOp.FiltrationSpec(
            kind=:function_delaunay,
            max_dim=max_dim,
            vertex_function=(p, _)->Float64(p[1]),
            simplex_agg=:max,
            construction=TamerOp.ConstructionOptions(
                ;
                output_stage=:simplex_tree,
            ),
        )
    elseif regime == "alpha_parity"
        return TamerOp.FiltrationSpec(
            kind=:alpha,
            max_dim=max_dim,
            construction=TamerOp.ConstructionOptions(
                ;
                output_stage=:simplex_tree,
            ),
        )
    elseif regime == "landmark_parity"
        haskey(case, "landmarks") || error("landmark_parity case requires landmarks in manifest.")
        haskey(case, "landmark_radius") || error("landmark_parity case requires landmark_radius in manifest.")
        landmarks = Int[Int(v) for v in case["landmarks"]]
        length(landmarks) > 0 || error("landmark_parity landmarks cannot be empty.")
        landmark_radius = Float64(case["landmark_radius"])
        return TamerOp.FiltrationSpec(
            kind=:landmark_rips,
            max_dim=max_dim,
            landmarks=landmarks,
            radius=landmark_radius,
            construction=TamerOp.ConstructionOptions(
                ;
                output_stage=:simplex_tree,
            ),
        )
    elseif regime == "core_delaunay_parity"
        return TamerOp.FiltrationSpec(
            kind=:core_delaunay,
            max_dim=max_dim,
            construction=TamerOp.ConstructionOptions(
                ;
                output_stage=:simplex_tree,
            ),
        )
    elseif regime == "cubical_parity"
        return TamerOp.FiltrationSpec(
            kind=:cubical,
            construction=TamerOp.ConstructionOptions(
                ;
                output_stage=:graded_complex,
            ),
        )
    else
        error("Unsupported regime: $(regime)")
    end
end

@inline function _run_encode_uncached(data, spec)
    # Harness policy: measure algorithmic runtime, excluding startup/JIT and
    # excluding cross-call session caching.
    return TamerOp.encode(data, spec; degree=0, cache=nothing)
end

function _timed_encode(data, spec)
    m = @timed st = _run_encode_uncached(data, spec)
    return st, 1000.0 * m.time, m.bytes / 1024.0
end

@inline _p90(v::Vector{Float64}) = isempty(v) ? NaN : sort(v)[max(1, ceil(Int, 0.9 * length(v)))]

@inline function _simplex_count(st)
    if hasproperty(st, :simplex_dims)
        return length(getproperty(st, :simplex_dims))
    elseif hasproperty(st, :dims)
        return length(getproperty(st, :dims))
    elseif hasproperty(st, :simplices)
        return length(getproperty(st, :simplices))
    elseif hasproperty(st, :cells_by_dim)
        return sum(length, getproperty(st, :cells_by_dim))
    end
    return missing
end

@inline function _edge_count(st)
    if hasproperty(st, :simplex_dims)
        return count(==(1), getproperty(st, :simplex_dims))
    elseif hasproperty(st, :dims)
        return count(==(1), getproperty(st, :dims))
    elseif hasproperty(st, :cells_by_dim)
        cbd = getproperty(st, :cells_by_dim)
        return length(cbd) >= 2 ? length(cbd[2]) : 0
    end
    return missing
end

@inline function _max_simplex_dim(st)
    if hasproperty(st, :simplex_dims)
        d = getproperty(st, :simplex_dims)
        return isempty(d) ? -1 : maximum(d)
    elseif hasproperty(st, :dims)
        d = getproperty(st, :dims)
        return isempty(d) ? -1 : maximum(d)
    elseif hasproperty(st, :cells_by_dim)
        return length(getproperty(st, :cells_by_dim)) - 1
    end
    return missing
end

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
        println(io, "tool,case_id,regime,n_points,ambient_dim,max_dim,cold_ms,cold_alloc_kib,warm_median_ms,warm_p90_ms,warm_alloc_median_kib,simplex_count,edge_count,max_simplex_dim,notes,timestamp_utc")
        for r in rows
            vals = (
                r.tool, r.case_id, r.regime, r.n_points, r.ambient_dim, r.max_dim,
                r.cold_ms, r.cold_alloc_kib, r.warm_median_ms, r.warm_p90_ms, r.warm_alloc_median_kib,
                r.simplex_count, r.edge_count, r.max_simplex_dim, r.notes, r.timestamp_utc,
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
    regime_filter = _arg(args, "--regime", "all")
    case_filter = _arg(args, "--case", "")

    reps >= 1 || error("--reps must be >= 1.")

    println("[profile] ", profile, " (reps=", reps, ", trim_between_reps=", defaults.trim_between_reps, ", trim_between_cases=", defaults.trim_between_cases, ")")
    println("NearestNeighbors package loaded: ", _HAVE_NEARESTNEIGHBORS)

    raw = TOML.parsefile(manifest_path)
    cases = get(raw, "cases", Any[])
    rows = NamedTuple[]

    for c in cases
        case_id = string(c["id"])
        regime = string(c["regime"])
        regime_filter == "all" || regime == regime_filter || continue
        isempty(case_filter) || case_id == case_filter || continue

        fixture = joinpath(dirname(manifest_path), string(c["path"]))
        n_points = Int(c["n_points"])
        ambient_dim = Int(c["ambient_dim"])
        max_dim = Int(c["max_dim"])
        data = _load_case_data(c, fixture)

        notes = ""
        spec = _spec_for_case(c; prefer_nn=true)
        if regime == "alpha_parity"
            notes = "alpha_vs_alpha"
        elseif regime == "degree_rips_parity"
            notes = "degree_rips_vs_degree_rips"
        elseif regime == "rips_codensity_parity"
            notes = "rips_codensity_vs_rips_codensity"
        elseif regime == "rips_lowerstar_parity"
            notes = "rips_lowerstar_exact_parity"
        elseif regime == "delaunay_lowerstar_parity"
            notes = "function_delaunay_vs_delaunay_lowerstar"
        elseif regime == "landmark_parity"
            notes = "landmark_subset_radius_parity"
        elseif regime == "core_delaunay_parity"
            notes = "core_delaunay_vs_core_delaunay"
        elseif regime == "cubical_parity"
            notes = "cubical_vs_cubical"
        end
        cold_st = nothing
        cold_ms = NaN
        cold_alloc = NaN
        try
            _run_encode_uncached(data, spec)  # untimed prewarm
            _memory_relief!()
            cold_st, cold_ms, cold_alloc = _timed_encode(data, spec)
        catch err
            if regime == "claim_matching"
                notes = "nearestneighbors_unavailable_fallback_bruteforce"
                spec = _spec_for_case(c; prefer_nn=false)
                _run_encode_uncached(data, spec)  # untimed prewarm
                _memory_relief!()
                cold_st, cold_ms, cold_alloc = _timed_encode(data, spec)
            else
                rethrow(err)
            end
        end

        warm_times = Float64[]
        warm_allocs = Float64[]
        for _ in 1:reps
            _, tms, akib = _timed_encode(data, spec)
            push!(warm_times, tms)
            push!(warm_allocs, akib)
            defaults.trim_between_reps && _memory_relief!()
        end

        warm_median_ms = median(warm_times)
        warm_p90_ms = _p90(warm_times)
        warm_alloc_median = median(warm_allocs)
        simplex_count = _simplex_count(cold_st)
        edge_count = _edge_count(cold_st)
        max_simplex_dim = _max_simplex_dim(cold_st)
        notes = isempty(notes) ? "cold_mode=warm_uncached;cache=none" :
                string(notes, ";cold_mode=warm_uncached;cache=none")

        println(
            rpad(case_id, 28),
            " regime=", regime,
            " cold_ms=", round(cold_ms, digits=3),
            " warm_med_ms=", round(warm_median_ms, digits=3),
            " simplices=", simplex_count,
        )

        push!(
            rows,
            (
                tool="tamer_op",
                case_id=case_id,
                regime=regime,
                n_points=n_points,
                ambient_dim=ambient_dim,
                max_dim=max_dim,
                cold_ms=cold_ms,
                cold_alloc_kib=cold_alloc,
                warm_median_ms=warm_median_ms,
                warm_p90_ms=warm_p90_ms,
                warm_alloc_median_kib=warm_alloc_median,
                simplex_count=simplex_count,
                edge_count=edge_count,
                max_simplex_dim=max_simplex_dim,
                notes=notes,
                timestamp_utc=string(Dates.now(Dates.UTC)),
            ),
        )

        defaults.trim_between_cases && _memory_relief!()
    end

    isempty(rows) && error("No cases selected. manifest=$(manifest_path), regime=$(regime_filter), case=$(case_filter)")
    _write_csv(out_path, rows)
    println("Wrote tamer results: $(out_path)")
end

main()
