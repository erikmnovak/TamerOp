#!/usr/bin/env julia
#
# data_ingestion_microbench.jl
#
# Purpose
# - Benchmark data-ingestion throughput across the main ingestion families in
#   `DataIngestion.jl`, with deterministic synthetic fixtures and optional
#   external pipeline fixtures on disk.
#
# What this measures
# - `encode(data, filtration/spec; ...)` ingestion paths for point-cloud, graph,
#   and image-based
#   filtrations.
# - Two execution stages:
#   - `:graded_complex` (construction-only stage)
#   - `:encoding_result` (end-to-end ingestion to encoding output)
# - Cache behavior:
#   - `cache=:auto` (per-call cache scope)
#   - `cache=SessionCache()` (cross-call reuse)
#
# External fixture support
# - This script can write reusable external fixture artifacts (pipeline JSON +
#   raw dataset files) to a directory and benchmark those files later.
# - This is intended to support cross-tool comparisons (e.g., tamer-op vs
#   multipers/RIVET) on exactly the same raw inputs.
#
# Usage
#   julia --project=. benchmark/data_ingestion_microbench.jl
#   julia --project=. benchmark/data_ingestion_microbench.jl --reps=8 --stages=graded
#   julia --project=. benchmark/data_ingestion_microbench.jl --reps=8 --stages=both
#   julia --project=. benchmark/data_ingestion_microbench.jl --reps=8 --stages=tree
#   julia --project=. benchmark/data_ingestion_microbench.jl --quick=1
#   julia --project=. benchmark/data_ingestion_microbench.jl --profile=balanced
#   julia --project=. benchmark/data_ingestion_microbench.jl --write_fixtures=1
#   julia --project=. benchmark/data_ingestion_microbench.jl --include_external=1 --external_dir=benchmark/data_ingestion_fixtures
#   julia --project=. benchmark/data_ingestion_microbench.jl --csv_out=benchmark/data_ingestion_results.csv
#   julia --project=. benchmark/data_ingestion_microbench.jl --stages=full --preflight_probe=1
#   julia --project=. benchmark/data_ingestion_microbench.jl --dense_distance_probe=1
#   julia --project=. benchmark/data_ingestion_microbench.jl --stage_isolation_probe=1 --include_synthetic=0 --include_external=0
#   JULIA_NUM_THREADS=4 julia --project=. benchmark/data_ingestion_microbench.jl --lowdim_h0_probe=1 --lazy_diff_thread_probe=1 --include_synthetic=0 --include_external=0
#   julia --project=. benchmark/data_ingestion_microbench.jl --structural_inclusion_probe=1 --include_synthetic=0 --include_external=0
#   julia --project=. benchmark/data_ingestion_microbench.jl --term_materialization_probe=1 --include_synthetic=0 --include_external=0
#   julia --project=. benchmark/data_ingestion_microbench.jl --pointcloud_dense_stream_probe=1 --graph_clique_probe=1 --include_synthetic=0 --include_external=0
#   julia --project=. benchmark/data_ingestion_microbench.jl --pointcloud_dim2_kernel_probe=1 --include_synthetic=0 --include_external=0
#   julia --project=. benchmark/data_ingestion_microbench.jl --nn_backend_probe=1 --include_synthetic=0 --include_external=0
#   julia --project=. benchmark/data_ingestion_microbench.jl --h0_chain_sweep_probe=1 --include_synthetic=0 --include_external=0
#   julia --project=. benchmark/data_ingestion_microbench.jl --h0_active_chain_incremental_probe=1 --include_synthetic=0 --include_external=0
#   julia --project=. benchmark/data_ingestion_microbench.jl --h1_fastpath_probe=1 --include_synthetic=0 --include_external=0
#   julia --project=. benchmark/data_ingestion_microbench.jl --t2_degree_local_probe=1 --include_synthetic=0 --include_external=0
#   julia --project=. benchmark/data_ingestion_microbench.jl --dims_only_probe=1 --include_synthetic=0 --include_external=0
#   julia --project=. benchmark/data_ingestion_microbench.jl --solve_loop_probe=1 --include_synthetic=0 --include_external=0
#   julia --project=. benchmark/data_ingestion_microbench.jl --cache_hit_skip_lazy_probe=1 --include_synthetic=0 --include_external=0
#   julia --project=. benchmark/data_ingestion_microbench.jl --plan_norm_cache_probe=1 --include_synthetic=0 --include_external=0
#   julia --project=. benchmark/data_ingestion_microbench.jl --boundary_kernel_probe=1 --include_synthetic=0 --include_external=0
#   julia --project=. benchmark/data_ingestion_microbench.jl --pointcloud_lowdim_radius_stream_probe=1 --include_synthetic=0 --include_external=0
#   julia --project=. benchmark/data_ingestion_microbench.jl --monotone_rank_probe=1 --include_synthetic=0 --include_external=0
#   julia --project=. benchmark/data_ingestion_microbench.jl --monotone_incremental_probe=1 --include_synthetic=0 --include_external=0
#   julia --project=. benchmark/data_ingestion_microbench.jl --degree_local_all_t_probe=1 --include_synthetic=0 --include_external=0
#   julia --project=. benchmark/data_ingestion_microbench.jl --encoding_result_lazy_probe=1 --include_synthetic=0 --include_external=0
#   julia --project=. benchmark/data_ingestion_microbench.jl --structural_map_probe=1 --include_synthetic=0 --include_external=0
#   julia --project=. benchmark/data_ingestion_microbench.jl --packed_edgelist_probe=1 --include_synthetic=0 --include_external=0
#   julia --project=. benchmark/data_ingestion_microbench.jl --structural_kernel_probe=1 --include_synthetic=0 --include_external=0
#   julia --project=. benchmark/data_ingestion_microbench.jl --active_list_chain_probe=1 --include_synthetic=0 --include_external=0
#   julia --project=. benchmark/data_ingestion_microbench.jl --delaunay_cache_probe=1 --include_synthetic=0 --include_external=0
#   julia --project=. benchmark/data_ingestion_microbench.jl --delaunay_simplextree_fusion_probe=1 --include_synthetic=0 --include_external=0
#   julia --project=. benchmark/data_ingestion_microbench.jl --core_numbers_probe=1 --include_synthetic=0 --include_external=0
#   julia --project=. benchmark/data_ingestion_microbench.jl --landmark_radius_probe=1 --include_synthetic=0 --include_external=0
#   julia --project=. benchmark/data_ingestion_microbench.jl --cubical_2d_probe=1 --include_synthetic=0 --include_external=0

using Dates
using DelimitedFiles
using Random
using SparseArrays
using TOML

try
    using TamerOp
catch
    include(joinpath(@__DIR__, "..", "src", "TamerOp.jl"))
    using .TamerOp
end

const TO = TamerOp.Advanced
const CM = TO.CoreModules
const DI = TamerOp.DataIngestion
const AC = TamerOp.AbelianCategories
const FL = TamerOp.FieldLinAlg

# Ensure NearestNeighbors extension is loaded when available so nn_backend=:auto
# in benchmark cases reflects the intended fast CPU path.
const _HAVE_NEARESTNEIGHBORS = let ok = false
    try
        @eval using NearestNeighbors
        ok = true
    catch
        ok = false
    end
    ok
end
const _HAVE_POINTCLOUD_NN_BACKEND = try
    DI._have_pointcloud_nn_backend()
catch
    false
end

const _SYMBOL_PARAM_KEYS = Set{Symbol}((
    :simplex_agg,
    :centrality,
    :metric,
    :lift,
    :highdim_policy,
    :nn_backend,
    :axes_policy,
    :axis_kind,
    :poset_kind,
    :multicritical,
    :onecritical_selector,
))

struct IngestionBenchCase
    name::String
    family::Symbol
    data
    spec::TamerOp.FiltrationSpec
    degree::Int
end

@inline function _parse_arg(args, key::String, default::Int)
    for a in args
        startswith(a, key * "=") || continue
        return parse(Int, split(a, "=", limit=2)[2])
    end
    return default
end

@inline function _parse_str(args, key::String, default::String)
    for a in args
        startswith(a, key * "=") || continue
        return String(split(a, "=", limit=2)[2])
    end
    return default
end

@inline function _parse_bool(args, key::String, default::Bool)
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

@inline function _benchmark_profile_defaults(profile::Symbol)
    if profile == :balanced
        # Safer default profile that still gives stable relative performance signal.
        return (reps=8, quick=true, include_synthetic=true, include_external=false, full_all=false)
    elseif profile == :stress
        return (reps=6, quick=false, include_synthetic=true, include_external=false, full_all=false)
    elseif profile == :probe
        return (reps=10, quick=true, include_synthetic=false, include_external=false, full_all=false)
    else
        throw(ArgumentError("unknown benchmark profile $(profile); expected :balanced, :stress, or :probe"))
    end
end

function _bench(name::AbstractString, f::Function; reps::Int=8)
    GC.gc()
    f() # warmup
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
    med_idx = cld(reps, 2)
    p90_idx = min(reps, max(1, ceil(Int, 0.9 * reps)))
    med_ms = times_ms[med_idx]
    p90_ms = times_ms[p90_idx]
    med_kib = bytes[med_idx] / 1024.0
    println(rpad(name, 54),
            " median_time=", round(med_ms, digits=3), " ms",
            "  p90_time=", round(p90_ms, digits=3), " ms",
            "  median_alloc=", round(med_kib, digits=1), " KiB")
    return (med_ms=med_ms, p90_ms=p90_ms, med_kib=med_kib)
end

@inline function _safe_name(s::AbstractString)
    t = lowercase(strip(s))
    t = replace(t, r"[^a-z0-9]+" => "_")
    isempty(t) && return "case"
    return t
end

@inline function _get_key(x, key::Symbol, default=nothing)
    x === nothing && return default
    if x isa NamedTuple
        return haskey(x, key) ? getfield(x, key) : default
    elseif x isa AbstractDict
        haskey(x, key) && return x[key]
        sk = String(key)
        haskey(x, sk) && return x[sk]
        return default
    end
    return default
end

@inline function _as_symbol(x, default::Symbol)
    x === nothing && return default
    x isa Symbol && return x
    x isa AbstractString && return Symbol(x)
    return default
end

@inline function _normalize_param_value(key::Symbol, v)
    if key in _SYMBOL_PARAM_KEYS && v isa AbstractString
        return Symbol(v)
    end
    return v
end

function _construction_with_stage(raw, stage::Symbol)
    if raw isa TamerOp.ConstructionOptions
        return TamerOp.ConstructionOptions(;
            sparsify=raw.sparsify,
            collapse=raw.collapse,
            output_stage=stage,
            budget=raw.budget,
        )
    end

    sparsify = _as_symbol(_get_key(raw, :sparsify, :none), :none)
    collapse = _as_symbol(_get_key(raw, :collapse, :none), :none)

    budget_raw = _get_key(raw, :budget, nothing)
    max_simplices = _get_key(budget_raw, :max_simplices, nothing)
    max_edges = _get_key(budget_raw, :max_edges, nothing)
    memory_budget_bytes = _get_key(budget_raw, :memory_budget_bytes, nothing)

    return TamerOp.ConstructionOptions(;
        sparsify=sparsify,
        collapse=collapse,
        output_stage=stage,
        budget=(max_simplices, max_edges, memory_budget_bytes),
    )
end

function _spec_with_stage(spec::TamerOp.FiltrationSpec, stage::Symbol)
    ps = Pair{Symbol,Any}[]
    raw_construction = nothing
    for (k, v) in pairs(spec.params)
        if k === :construction
            raw_construction = v
        else
            push!(ps, k => _normalize_param_value(k, v))
        end
    end
    push!(ps, :construction => _construction_with_stage(raw_construction, stage))
    nt = (; (kv.first => kv.second for kv in ps)...)
    return TamerOp.FiltrationSpec(; kind=spec.kind, nt...)
end

function _point_cloud_fixture(n::Int; seed::Int=0xD471)
    rng = Random.MersenneTwister(seed)
    points = Vector{Vector{Float64}}(undef, n)
    for i in 1:n
        x = rand(rng)
        y = 0.35 * sin(8.0 * x) + 0.15 * randn(rng)
        points[i] = [x, y]
    end
    return TamerOp.PointCloud(points)
end

function _graph_fixture(n::Int; seed::Int=0xD472)
    rng = Random.MersenneTwister(seed)
    coords = [[rand(rng), rand(rng)] for _ in 1:n]
    edge_w = Dict{Tuple{Int,Int},Float64}()

    # Keep connected baseline.
    for i in 1:(n - 1)
        u, v = i, i + 1
        w = hypot(coords[u][1] - coords[v][1], coords[u][2] - coords[v][2]) + 1e-3
        edge_w[(u, v)] = w
    end

    # Add sparse random shortcuts.
    m_extra = 2 * n
    for _ in 1:m_extra
        u = rand(rng, 1:n)
        v = rand(rng, 1:n)
        u == v && continue
        a, b = min(u, v), max(u, v)
        w = hypot(coords[a][1] - coords[b][1], coords[a][2] - coords[b][2]) + 1e-3
        edge_w[(a, b)] = w
    end

    edges = collect(keys(edge_w))
    sort!(edges)
    weights = [edge_w[e] for e in edges]
    return TamerOp.GraphData(n, edges; coords=coords, weights=weights)
end

function _graph_clique_fixture(n::Int; seed::Int=0xD473)
    rng = Random.MersenneTwister(seed)
    coords = [[rand(rng), rand(rng)] for _ in 1:n]
    edge_w = Dict{Tuple{Int,Int},Float64}()

    # Ring + k-neighborhood edges; this creates many triangles.
    k = 3
    for i in 1:n
        for d in 1:k
            j = ((i - 1 + d) % n) + 1
            a, b = min(i, j), max(i, j)
            w = hypot(coords[a][1] - coords[b][1], coords[a][2] - coords[b][2]) + 1e-3
            edge_w[(a, b)] = w
        end
    end

    edges = collect(keys(edge_w))
    sort!(edges)
    weights = [edge_w[e] for e in edges]
    return TamerOp.GraphData(n, edges; coords=coords, weights=weights)
end

function _image_fixture(side::Int; seed::Int=0xD474)
    rng = Random.MersenneTwister(seed)
    img = Matrix{Float64}(undef, side, side)
    for i in 1:side, j in 1:side
        img[i, j] = 0.6 * sin(i / 9.0) + 0.5 * cos(j / 11.0) + 0.05 * randn(rng)
    end
    vals = sort(vec(copy(img)))
    thr = vals[clamp(cld(3 * length(vals), 5), 1, length(vals))]
    mask = img .> thr
    return TamerOp.ImageNd(img), mask
end

function _embedded_planar_fixture(n::Int; seed::Int=0xD475)
    rng = Random.MersenneTwister(seed)
    verts = [[rand(rng), rand(rng)] for _ in 1:n]
    edge_w = Dict{Tuple{Int,Int},Float64}()
    k = max(2, min(4, n - 1))
    for i in 1:n
        for d in 1:k
            j = ((i - 1 + d) % n) + 1
            a, b = min(i, j), max(i, j)
            w = hypot(verts[a][1] - verts[b][1], verts[a][2] - verts[b][2]) + 1e-3
            edge_w[(a, b)] = w
        end
    end
    edges = collect(keys(edge_w))
    sort!(edges)
    weights = [edge_w[e] for e in edges]
    xmin = minimum(v[1] for v in verts)
    xmax = maximum(v[1] for v in verts)
    ymin = minimum(v[2] for v in verts)
    ymax = maximum(v[2] for v in verts)
    data = TamerOp.EmbeddedPlanarGraph2D(verts, edges; bbox=(xmin, xmax, ymin, ymax))
    return data, weights
end

function _synthetic_cases(; point_n::Int, graph_n::Int, clique_n::Int, image_side::Int, seed::Int)
    cases = IngestionBenchCase[]

    point_data = _point_cloud_fixture(point_n; seed=seed + 1)
    pvals = [p[1] + 0.35 * p[2] for p in point_data.points]
    point_knn = max(1, min(12, point_n - 1))
    core_knn = max(1, min(10, point_n - 1))
    point_budget = (
        max_simplices = max(200_000, 60 * point_n),
        max_edges = max(200_000, 40 * point_n),
        memory_budget_bytes = 8_000_000_000,
    )
    landmark_target = min(point_n, max(16, ceil(Int, sqrt(point_n))))
    landmark_step = max(1, fld(point_n, landmark_target))
    landmarks = collect(1:landmark_step:point_n)
    length(landmarks) > landmark_target && resize!(landmarks, landmark_target)
    isempty(landmarks) && push!(landmarks, 1)
    large_point = point_n >= 5_000
    if !large_point
        point_d2_n = min(point_n, 42)
        point_data_d2 = TamerOp.PointCloud(point_data.points[1:point_d2_n])
        push!(cases, IngestionBenchCase(
            "point_rips_dense_d1",
            :point,
            point_data,
            TamerOp.FiltrationSpec(
                kind=:rips,
                max_dim=1,
                construction=TamerOp.ConstructionOptions(; sparsify=:none, budget=point_budget),
            ),
            0,
        ))
        push!(cases, IngestionBenchCase(
            "point_rips_dense_d2",
            :point,
            point_data_d2,
            TamerOp.FiltrationSpec(
                kind=:rips,
                max_dim=2,
                construction=TamerOp.ConstructionOptions(; sparsify=:none, budget=point_budget),
            ),
            0,
        ))
        push!(cases, IngestionBenchCase(
            "point_rips_density_dense_d1",
            :point,
            point_data,
            TamerOp.FiltrationSpec(
                kind=:rips_density,
                max_dim=1,
                density_k=point_knn,
                construction=TamerOp.ConstructionOptions(; sparsify=:none, budget=point_budget),
            ),
            0,
        ))
        push!(cases, IngestionBenchCase(
            "point_function_rips_dense_d1",
            :point,
            point_data,
            TamerOp.FiltrationSpec(
                kind=:function_rips,
                max_dim=1,
                vertex_values=pvals,
                simplex_agg=:max,
                construction=TamerOp.ConstructionOptions(; sparsify=:none, budget=point_budget),
            ),
            0,
        ))
    end
    push!(cases, IngestionBenchCase(
        "point_rips_knn_sparsify",
        :point,
        point_data,
        TamerOp.FiltrationSpec(
            kind=:rips,
            max_dim=1,
            knn=point_knn,
            nn_backend=:auto,
            construction=TamerOp.ConstructionOptions(; sparsify=:knn, budget=point_budget),
        ),
        0,
    ))
    push!(cases, IngestionBenchCase(
        "point_rips_greedy_landmarks",
        :point,
        point_data,
        TamerOp.FiltrationSpec(
            kind=:rips,
            max_dim=1,
            n_landmarks=min(point_n, point_n >= 50_000 ? 2_048 : max(512, ceil(Int, sqrt(point_n)))),
            construction=TamerOp.ConstructionOptions(; sparsify=:greedy_perm, budget=point_budget),
        ),
        0,
    ))
    push!(cases, IngestionBenchCase(
        "point_landmark_rips_d1",
        :point,
        point_data,
        TamerOp.FiltrationSpec(
            kind=:landmark_rips,
            max_dim=1,
            landmarks=landmarks,
            construction=TamerOp.ConstructionOptions(; budget=point_budget),
        ),
        0,
    ))
    push!(cases, IngestionBenchCase(
        "point_function_rips_d1",
        :point,
        point_data,
        TamerOp.FiltrationSpec(
            kind=:function_rips,
            max_dim=1,
            vertex_values=pvals,
            simplex_agg=:max,
            knn=point_knn,
            nn_backend=:auto,
            construction=TamerOp.ConstructionOptions(; sparsify=:knn, budget=point_budget),
        ),
        0,
    ))
    if _HAVE_POINTCLOUD_NN_BACKEND
        push!(cases, IngestionBenchCase(
            "point_rips_knn_nearestneighbors",
            :point,
            point_data,
            TamerOp.FiltrationSpec(
                kind=:rips,
                max_dim=1,
                knn=point_knn,
                nn_backend=:nearestneighbors,
                construction=TamerOp.ConstructionOptions(; sparsify=:knn, budget=point_budget),
            ),
            0,
        ))
        push!(cases, IngestionBenchCase(
            "point_rips_knn_approx",
            :point,
            point_data,
            TamerOp.FiltrationSpec(
                kind=:rips,
                max_dim=1,
                knn=point_knn,
                nn_backend=:approx,
                nn_approx_candidates=48,
                construction=TamerOp.ConstructionOptions(; sparsify=:knn, budget=point_budget),
            ),
            0,
        ))
    end

    delaunay_n = min(point_n, 96)
    point_delaunay = TamerOp.PointCloud(point_data.points[1:delaunay_n])
    pvals_d = pvals[1:delaunay_n]
    push!(cases, IngestionBenchCase(
        "point_delaunay_lower_star_d2",
        :point,
        point_delaunay,
        TamerOp.FiltrationSpec(kind=:delaunay_lower_star, max_dim=2, vertex_values=pvals_d, simplex_agg=:max),
        0,
    ))
    push!(cases, IngestionBenchCase(
        "point_function_delaunay_d2",
        :point,
        point_delaunay,
        TamerOp.FiltrationSpec(kind=:function_delaunay, max_dim=2, vertex_values=pvals_d, simplex_agg=:max),
        0,
    ))
    push!(cases, IngestionBenchCase(
        "point_alpha_d2",
        :point,
        point_delaunay,
        TamerOp.FiltrationSpec(kind=:alpha, max_dim=2),
        0,
    ))
    push!(cases, IngestionBenchCase(
        "point_core_delaunay_d2",
        :point,
        point_delaunay,
        TamerOp.FiltrationSpec(kind=:core_delaunay, max_dim=2),
        0,
    ))
    push!(cases, IngestionBenchCase(
        "point_degree_rips_knn_d1",
        :point,
        point_data,
        TamerOp.FiltrationSpec(
            kind=:degree_rips,
            max_dim=1,
            knn=point_knn,
            nn_backend=:auto,
            construction=TamerOp.ConstructionOptions(; sparsify=:knn, budget=point_budget),
        ),
        0,
    ))
    push!(cases, IngestionBenchCase(
        "point_core",
        :point,
        point_data,
        TamerOp.FiltrationSpec(kind=:core, knn=core_knn, vertex_values=pvals),
        0,
    ))
    push!(cases, IngestionBenchCase(
        "point_rhomboid_d1",
        :point,
        point_data,
        TamerOp.FiltrationSpec(kind=:rhomboid, max_dim=1, vertex_values=pvals),
        0,
    ))
    rhomboid2_n = min(point_n, 52)
    point_rhomboid2 = TamerOp.PointCloud(point_data.points[1:rhomboid2_n])
    pvals_r2 = pvals[1:rhomboid2_n]
    push!(cases, IngestionBenchCase(
        "point_rhomboid_d2",
        :point,
        point_rhomboid2,
        TamerOp.FiltrationSpec(kind=:rhomboid, max_dim=2, vertex_values=pvals_r2),
        0,
    ))
    point_tree_data = TamerOp.encode(
        point_data,
        TamerOp.FiltrationSpec(
            kind=:rips,
            max_dim=1,
            knn=point_knn,
            nn_backend=:auto,
            construction=TamerOp.ConstructionOptions(; sparsify=:knn, output_stage=:simplex_tree, budget=point_budget),
        );
        degree=0,
        cache=:auto,
    )
    push!(cases, IngestionBenchCase(
        "point_simplextree_input_rips",
        :point,
        point_tree_data,
        TamerOp.FiltrationSpec(kind=:rips, max_dim=1,
                                    construction=TamerOp.ConstructionOptions(; output_stage=:encoding_result, budget=point_budget)),
        0,
    ))
    push!(cases, IngestionBenchCase(
        "point_simplextree_input_rips_eps",
        :point,
        point_tree_data,
        TamerOp.FiltrationSpec(
            kind=:graded,
            eps=0.05,
            construction=TamerOp.ConstructionOptions(; output_stage=:encoding_result, budget=point_budget),
        ),
        0,
    ))
    push!(cases, IngestionBenchCase(
        "point_simplextree_input_rips_eps_via_graded",
        :point,
        TamerOp.DataIngestion._graded_complex_from_simplex_tree(point_tree_data),
        TamerOp.FiltrationSpec(
            kind=:graded,
            eps=0.05,
            construction=TamerOp.ConstructionOptions(; output_stage=:encoding_result, budget=point_budget),
        ),
        0,
    ))

    mc_cells = [Int[1, 2], Int[1]]
    mc_b1 = sparse([1, 2], [1, 1], [1, -1], 2, 1)
    mc_grades = [
        [Float64[0.0, 0.0]],
        [Float64[0.0, 0.0]],
        [Float64[1.0, 0.0], Float64[0.0, 1.0], Float64[1.0, 1.0]],
    ]
    mc_g = TamerOp.MultiCriticalGradedComplex(mc_cells, [mc_b1], mc_grades)
    mc_st = TamerOp.DataIngestion._simplex_tree_multi_from_complex(mc_g)
    mc_one_spec = TamerOp.FiltrationSpec(
        kind=:graded,
        multicritical=:one_critical,
        onecritical_selector=:lexmin,
        onecritical_enforce_boundary=true,
        construction=TamerOp.ConstructionOptions(; output_stage=:encoding_result, budget=point_budget),
    )
    push!(cases, IngestionBenchCase(
        "point_simplextree_multicritical_onecritical",
        :point,
        mc_st,
        mc_one_spec,
        0,
    ))
    push!(cases, IngestionBenchCase(
        "point_multicritical_onecritical_via_graded",
        :point,
        mc_g,
        mc_one_spec,
        0,
    ))

    graph_data = _graph_fixture(graph_n; seed=seed + 2)
    gvals = [graph_data.coords[i][1] + 0.25 * graph_data.coords[i][2] for i in 1:graph_data.n]
    push!(cases, IngestionBenchCase(
        "graph_lower_star",
        :graph,
        graph_data,
        TamerOp.FiltrationSpec(kind=:graph_lower_star, vertex_values=gvals, simplex_agg=:max),
        0,
    ))
    push!(cases, IngestionBenchCase(
        "graph_centrality_degree",
        :graph,
        graph_data,
        TamerOp.FiltrationSpec(kind=:graph_centrality, centrality=:degree, lift=:lower_star),
        0,
    ))
    push!(cases, IngestionBenchCase(
        "graph_geodesic_hop",
        :graph,
        graph_data,
        TamerOp.FiltrationSpec(kind=:graph_geodesic, sources=[1], metric=:hop, lift=:lower_star),
        0,
    ))
    push!(cases, IngestionBenchCase(
        "graph_function_geodesic_bi",
        :graph,
        graph_data,
        TamerOp.FiltrationSpec(
            kind=:graph_function_geodesic_bifiltration,
            sources=[1],
            metric=:hop,
            vertex_values=gvals,
            lift=:lower_star,
            simplex_agg=:max,
        ),
        0,
    ))
    push!(cases, IngestionBenchCase(
        "graph_weight_threshold_graph",
        :graph,
        graph_data,
        TamerOp.FiltrationSpec(kind=:graph_weight_threshold, lift=:graph, edge_weights=graph_data.weights),
        0,
    ))
    push!(cases, IngestionBenchCase(
        "graph_edge_weighted",
        :graph,
        graph_data,
        TamerOp.FiltrationSpec(kind=:edge_weighted, edge_weights=graph_data.weights),
        0,
    ))
    push!(cases, IngestionBenchCase(
        "graph_core",
        :graph,
        graph_data,
        TamerOp.FiltrationSpec(kind=:core, vertex_values=gvals),
        0,
    ))

    graph_clique = _graph_clique_fixture(clique_n; seed=seed + 3)
    gvals_clique = [graph_clique.coords[i][1] + 0.25 * graph_clique.coords[i][2] for i in 1:graph_clique.n]
    push!(cases, IngestionBenchCase(
        "graph_weight_threshold_clique_d2",
        :graph,
        graph_clique,
        TamerOp.FiltrationSpec(kind=:graph_weight_threshold, lift=:clique, max_dim=2, edge_weights=graph_clique.weights),
        0,
    ))
    push!(cases, IngestionBenchCase(
        "graph_clique_lower_star_d2",
        :graph,
        graph_clique,
        TamerOp.FiltrationSpec(kind=:clique_lower_star, max_dim=2, vertex_values=gvals_clique, simplex_agg=:max),
        0,
    ))

    image_data, image_mask = _image_fixture(image_side; seed=seed + 4)
    push!(cases, IngestionBenchCase(
        "image_lower_star",
        :image,
        image_data,
        TamerOp.FiltrationSpec(kind=:lower_star),
        0,
    ))
    push!(cases, IngestionBenchCase(
        "image_cubical",
        :image,
        image_data,
        TamerOp.FiltrationSpec(kind=:cubical),
        0,
    ))
    push!(cases, IngestionBenchCase(
        "image_distance_bifiltration",
        :image,
        image_data,
        TamerOp.FiltrationSpec(kind=:image_distance_bifiltration, mask=image_mask),
        0,
    ))

    embedded_n = max(24, min(clique_n, 64))
    embedded_data, embedded_weights = _embedded_planar_fixture(embedded_n; seed=seed + 5)
    push!(cases, IngestionBenchCase(
        "embedded_wing_vein_bifiltration",
        :image,
        embedded_data,
        TamerOp.FiltrationSpec(
            kind=:wing_vein_bifiltration,
            grid=(32, 32),
            bbox=embedded_data.bbox,
        ),
        0,
    ))
    push!(cases, IngestionBenchCase(
        "embedded_edge_weighted",
        :graph,
        TamerOp.GraphData(length(embedded_data.vertices), embedded_data.edges;
                               coords=embedded_data.vertices, weights=embedded_weights),
        TamerOp.FiltrationSpec(kind=:edge_weighted, edge_weights=embedded_weights),
        0,
    ))

    return cases
end

function _write_pointcloud_csv(path::AbstractString, data::TamerOp.PointCloud)
    open(path, "w") do io
        for p in data.points
            println(io, join(Float64[p...], ","))
        end
    end
    return nothing
end

function _write_graph_csv(path::AbstractString, data::TamerOp.GraphData)
    open(path, "w") do io
        println(io, "u,v,weight")
        has_weights = data.weights !== nothing && length(data.weights) == length(data.edges)
        for i in eachindex(data.edges)
            u, v = data.edges[i]
            w = has_weights ? Float64(data.weights[i]) : 1.0
            println(io, "$(u),$(v),$(w)")
        end
    end
    return nothing
end

function _write_image_csv(path::AbstractString, data::TamerOp.ImageNd)
    arr = data.data
    if ndims(arr) == 2
        writedlm(path, Matrix{Float64}(arr), ',')
        return nothing
    end
    open(path, "w") do io
        println(io, "i1,i2,...,value")
        for I in CartesianIndices(arr)
            idx = Tuple(I)
            println(io, string(join(idx, ","), ",", Float64(arr[I])))
        end
    end
    return nothing
end

function _write_fixture_pack(dir::AbstractString, cases::Vector{IngestionBenchCase})
    mkpath(dir)
    manifest_cases = NamedTuple[]
    for case in cases
        base = _safe_name(case.name)
        pipeline_file = base * ".pipeline.json"
        pipeline_path = joinpath(dir, pipeline_file)
        spec = _spec_with_stage(case.spec, :encoding_result)
        TamerOp.save_pipeline_json(pipeline_path, case.data, spec; degree=case.degree)

        raw_file = ""
        if case.data isa TamerOp.PointCloud
            raw_file = base * ".points.csv"
            _write_pointcloud_csv(joinpath(dir, raw_file), case.data)
        elseif case.data isa TamerOp.GraphData
            raw_file = base * ".edges.csv"
            _write_graph_csv(joinpath(dir, raw_file), case.data)
        elseif case.data isa TamerOp.ImageNd
            raw_file = base * ".image.csv"
            _write_image_csv(joinpath(dir, raw_file), case.data)
        end

        push!(manifest_cases, (; name=case.name, family=String(case.family), pipeline=pipeline_file, raw=raw_file, degree=case.degree))
    end

    manifest_path = joinpath(dir, "manifest.toml")
    open(manifest_path, "w") do io
        println(io, "version = 1")
        println(io, "generated_by = \"benchmark/data_ingestion_microbench.jl\"")
        println(io, "generated_at_utc = \"", Dates.format(Dates.now(Dates.UTC), dateformat"yyyy-mm-ddTHH:MM:SS"), "Z\"")
        for c in manifest_cases
            println(io)
            println(io, "[[cases]]")
            println(io, "name = \"", c.name, "\"")
            println(io, "family = \"", c.family, "\"")
            println(io, "pipeline = \"", c.pipeline, "\"")
            println(io, "raw = \"", c.raw, "\"")
            println(io, "degree = ", c.degree)
        end
    end
    return manifest_path
end

function _load_external_cases(dir::AbstractString)
    cases = IngestionBenchCase[]
    isdir(dir) || return cases

    manifest_path = joinpath(dir, "manifest.toml")
    if isfile(manifest_path)
        obj = TOML.parsefile(manifest_path)
        entries = get(obj, "cases", Any[])
        for e in entries
            pipe_rel = get(e, "pipeline", nothing)
            pipe_rel === nothing && continue
            pipe_path = isabspath(pipe_rel) ? pipe_rel : joinpath(dir, pipe_rel)
            if !isfile(pipe_path)
                @warn "Skipping missing pipeline fixture" path=pipe_path
                continue
            end
            data, spec, degree, _ = TamerOp.load_pipeline_json(pipe_path)
            name = get(e, "name", basename(pipe_path))
            fam = Symbol(get(e, "family", "external"))
            push!(cases, IngestionBenchCase(name, fam, data, spec, Int(degree)))
        end
        return cases
    end

    for path in sort(filter(p -> endswith(p, ".pipeline.json"), readdir(dir; join=true)))
        data, spec, degree, _ = TamerOp.load_pipeline_json(path)
        name = replace(basename(path), ".pipeline.json" => "")
        push!(cases, IngestionBenchCase(name, :external, data, spec, Int(degree)))
    end
    return cases
end

function _case_size_hint(case::IngestionBenchCase)
    d = case.data
    if d isa TamerOp.PointCloud
        n = length(d.points)
        m = n == 0 ? 0 : length(d.points[1])
        return "n=$(n), dim=$(m)"
    elseif d isa TamerOp.GraphData
        return "n=$(d.n), edges=$(length(d.edges))"
    elseif d isa TamerOp.ImageNd
        return "size=$(size(d.data))"
    end
    return "type=$(typeof(d))"
end

function _family_filter(cases::Vector{IngestionBenchCase}, family::Symbol)
    family === :all && return cases
    return [c for c in cases if c.family == family]
end

@inline function _full_safe_case(case::IngestionBenchCase)
    case.name in (
        "point_rips_knn_sparsify",
        "point_rips_greedy_landmarks",
        "point_landmark_rips_d1",
        "point_degree_rips_knn_d1",
        "point_rips_knn_nearestneighbors",
        "point_rips_knn_approx",
        "point_function_delaunay_d2",
        "point_alpha_d2",
        "point_core_delaunay_d2",
        "point_simplextree_input_rips",
        "point_simplextree_input_rips_eps",
        "point_simplextree_input_rips_eps_via_graded",
        "point_simplextree_multicritical_onecritical",
        "point_multicritical_onecritical_via_graded",
        "graph_lower_star",
        "graph_edge_weighted",
        "graph_clique_lower_star_d2",
        "image_lower_star",
        "image_cubical",
    )
end

@inline function _tree_safe_case(case::IngestionBenchCase)
    case.family != :image && case.spec.kind != :graded
end

function _write_results_csv(path::AbstractString, results)
    open(path, "w") do io
        println(io, "case,family,stage,cache_mode,median_ms,p90_ms,median_alloc_kib")
        for r in results
            println(io,
                    string(r.case, ",", r.family, ",", r.stage, ",", r.cache_mode, ",",
                           round(r.med_ms, digits=6), ",",
                           round(r.p90_ms, digits=6), ",",
                           round(r.med_kib, digits=6)))
        end
    end
    return nothing
end

function _run_case(case::IngestionBenchCase, stage::Symbol; reps::Int)
    spec = _spec_with_stage(case.spec, stage)
    stage_label = stage == :graded_complex ? "graded" : (stage == :simplex_tree ? "tree" : "full")
    println("\n[case] ", case.name, "  family=", case.family, "  ", _case_size_hint(case), "  stage=", stage_label)

    auto = _bench("  cache=:auto", () -> TamerOp.encode(case.data, spec; degree=case.degree, cache=:auto); reps=reps)
    sc = CM.SessionCache()
    sess = _bench("  cache=SessionCache()", () -> TamerOp.encode(case.data, spec; degree=case.degree, cache=sc); reps=reps)

    speedup = auto.med_ms / max(sess.med_ms, 1e-12)
    println("  speedup session/auto=", round(speedup, digits=3), "x")

    r_auto = (case=case.name, family=String(case.family), stage=stage_label, cache_mode="auto",
              med_ms=auto.med_ms, p90_ms=auto.p90_ms, med_kib=auto.med_kib)
    r_sess = (case=case.name, family=String(case.family), stage=stage_label, cache_mode="session",
              med_ms=sess.med_ms, p90_ms=sess.p90_ms, med_kib=sess.med_kib)
    return r_auto, r_sess
end

function _run_preflight_probe(; reps::Int, quick::Bool)
    n = quick ? 48 : 160
    data = _point_cloud_fixture(n; seed=Int(0xD499))
    knn = max(1, min(12, n - 1))
    budget = (
        max_simplices = max(200_000, 60 * n),
        max_edges = max(200_000, 40 * n),
        memory_budget_bytes = 8_000_000_000,
    )
    spec = TamerOp.FiltrationSpec(
        kind=:rips,
        max_dim=1,
        axes=(collect(range(0.0, stop=2.0, length=16)),),
        knn=knn,
        nn_backend=:auto,
        construction=TamerOp.ConstructionOptions(; sparsify=:knn, output_stage=:graded_complex, budget=budget),
    )
    println("\n[probe] preflight overhead  n=$(n), knn=$(knn)")
    off = _bench("  encode(...; preflight=false)",
                 () -> TamerOp.encode(data, spec; degree=0, cache=:auto, preflight=false);
                 reps=reps)
    on = _bench("  encode(...; preflight=true)",
                () -> TamerOp.encode(data, spec; degree=0, cache=:auto, preflight=true);
                reps=reps)
    strict = _bench("  encode(...; strict_preflight=true)",
                    () -> TamerOp.encode(data, spec; degree=0, cache=:auto, strict_preflight=true);
                    reps=reps)
    println("  overhead preflight/on vs off=", round(on.med_ms / max(off.med_ms, 1e-12), digits=3), "x")
    println("  overhead strict vs off=", round(strict.med_ms / max(off.med_ms, 1e-12), digits=3), "x")

    r_off = (case="preflight_probe_point_rips_knn", family="point", stage="graded", cache_mode="preflight_off",
             med_ms=off.med_ms, p90_ms=off.p90_ms, med_kib=off.med_kib)
    r_on = (case="preflight_probe_point_rips_knn", family="point", stage="graded", cache_mode="preflight_on",
            med_ms=on.med_ms, p90_ms=on.p90_ms, med_kib=on.med_kib)
    r_strict = (case="preflight_probe_point_rips_knn", family="point", stage="graded", cache_mode="preflight_strict",
                med_ms=strict.med_ms, p90_ms=strict.p90_ms, med_kib=strict.med_kib)
    return (r_off, r_on, r_strict)
end

function _dense_rips_simplices(n::Int)
    simplices = Vector{Vector{Vector{Int}}}(undef, 3)
    simplices[1] = [[i] for i in 1:n]
    simplices[2] = DI._combinations(n, 2)
    simplices[3] = DI._combinations(n, 3)
    return simplices
end

function _dense_rips_grades_matrix(points::Vector{Vector{Float64}}, simplices)
    dist = DI._point_cloud_distance_matrix(points)
    acc = 0.0
    for k in 2:length(simplices)
        for s in simplices[k]
            maxd = 0.0
            @inbounds for i in 1:length(s)
                for j in (i + 1):length(s)
                    d = dist[s[i], s[j]]
                    d > maxd && (maxd = d)
                end
            end
            acc += maxd
        end
    end
    return acc
end

function _dense_rips_grades_packed(points::Vector{Vector{Float64}}, simplices)
    packed = DI._point_cloud_pairwise_packed(points)
    n = length(points)
    acc = 0.0
    for k in 2:length(simplices)
        for s in simplices[k]
            maxd = 0.0
            @inbounds for i in 1:length(s)
                for j in (i + 1):length(s)
                    d = DI._packed_pair_distance(packed, n, s[i], s[j])
                    d > maxd && (maxd = d)
                end
            end
            acc += maxd
        end
    end
    return acc
end

function _run_dense_distance_probe(; reps::Int, quick::Bool)
    n = quick ? 28 : 42
    data = _point_cloud_fixture(n; seed=Int(0xD49A))
    points = Vector{Vector{Float64}}(undef, length(data.points))
    @inbounds for i in eachindex(data.points)
        points[i] = Float64[data.points[i]...]
    end
    simplices = _dense_rips_simplices(length(points))
    println("\n[probe] dense distance backend  n=$(n), max_dim=2")
    dense = _bench("  distance matrix kernel",
                   () -> _dense_rips_grades_matrix(points, simplices);
                   reps=reps)
    packed = _bench("  packed pairwise kernel",
                    () -> _dense_rips_grades_packed(points, simplices);
                    reps=reps)
    println("  packed vs dense time ratio=", round(packed.med_ms / max(dense.med_ms, 1e-12), digits=3), "x")
    println("  packed vs dense alloc ratio=", round(packed.med_kib / max(dense.med_kib, 1e-12), digits=3), "x")

    r_dense = (case="dense_distance_probe_rips_d2", family="point", stage="graded", cache_mode="dense_matrix",
               med_ms=dense.med_ms, p90_ms=dense.p90_ms, med_kib=dense.med_kib)
    r_packed = (case="dense_distance_probe_rips_d2", family="point", stage="graded", cache_mode="packed_pairwise",
                med_ms=packed.med_ms, p90_ms=packed.p90_ms, med_kib=packed.med_kib)
    return (r_dense, r_packed)
end

function _run_stage_isolation_probe(; reps::Int, quick::Bool)
    n = quick ? 80 : 240
    data = _point_cloud_fixture(n; seed=Int(0xD49B))
    knn = max(2, min(12, n - 1))
    budget = (
        max_simplices = max(500_000, 120 * n),
        max_edges = max(500_000, 80 * n),
        memory_budget_bytes = 8_000_000_000,
    )
    spec = TamerOp.FiltrationSpec(
        kind=:rips,
        max_dim=1,
        axes=(collect(range(0.0, stop=2.0, length=24)),),
        knn=knn,
        nn_backend=:auto,
        construction=TamerOp.ConstructionOptions(; sparsify=:knn, output_stage=:encoding_result, budget=budget),
    )

    println("\n[probe] stage isolation  n=$(n), knn=$(knn)")
    module_auto = _bench("  stage=:module  cache=:auto",
                         () -> TamerOp.encode(data, spec; degree=0, cache=:auto, stage=:module);
                         reps=reps)
    full_auto = _bench("  stage=:encoding_result  cache=:auto",
                       () -> TamerOp.encode(data, spec; degree=0, cache=:auto, stage=:encoding_result);
                       reps=reps)
    fringe_auto = _bench("  stage=:fringe  cache=:auto",
                         () -> TamerOp.encode(data, spec; degree=0, cache=:auto, stage=:fringe);
                         reps=reps)

    sc = CM.SessionCache()
    module_sess = _bench("  stage=:module  cache=SessionCache()",
                         () -> TamerOp.encode(data, spec; degree=0, cache=sc, stage=:module);
                         reps=reps)
    full_sess = _bench("  stage=:encoding_result  cache=SessionCache()",
                       () -> TamerOp.encode(data, spec; degree=0, cache=sc, stage=:encoding_result);
                       reps=reps)
    fringe_sess = _bench("  stage=:fringe  cache=SessionCache()",
                         () -> TamerOp.encode(data, spec; degree=0, cache=sc, stage=:fringe);
                         reps=reps)

    println("  auto: encoding_result/module ratio=", round(full_auto.med_ms / max(module_auto.med_ms, 1e-12), digits=3), "x")
    println("  auto: fringe/encoding_result ratio=", round(fringe_auto.med_ms / max(full_auto.med_ms, 1e-12), digits=3), "x")
    println("  session: encoding_result/module ratio=", round(full_sess.med_ms / max(module_sess.med_ms, 1e-12), digits=3), "x")
    println("  session: fringe/encoding_result ratio=", round(fringe_sess.med_ms / max(full_sess.med_ms, 1e-12), digits=3), "x")

    return (
        (case="stage_isolation_probe_point_rips_knn", family="point", stage="module", cache_mode="auto",
         med_ms=module_auto.med_ms, p90_ms=module_auto.p90_ms, med_kib=module_auto.med_kib),
        (case="stage_isolation_probe_point_rips_knn", family="point", stage="full", cache_mode="auto",
         med_ms=full_auto.med_ms, p90_ms=full_auto.p90_ms, med_kib=full_auto.med_kib),
        (case="stage_isolation_probe_point_rips_knn", family="point", stage="fringe", cache_mode="auto",
         med_ms=fringe_auto.med_ms, p90_ms=fringe_auto.p90_ms, med_kib=fringe_auto.med_kib),
        (case="stage_isolation_probe_point_rips_knn", family="point", stage="module", cache_mode="session",
         med_ms=module_sess.med_ms, p90_ms=module_sess.p90_ms, med_kib=module_sess.med_kib),
        (case="stage_isolation_probe_point_rips_knn", family="point", stage="full", cache_mode="session",
         med_ms=full_sess.med_ms, p90_ms=full_sess.p90_ms, med_kib=full_sess.med_kib),
        (case="stage_isolation_probe_point_rips_knn", family="point", stage="fringe", cache_mode="session",
         med_ms=fringe_sess.med_ms, p90_ms=fringe_sess.p90_ms, med_kib=fringe_sess.med_kib),
    )
end

function _build_lazy_lowdim_h0_fixture(; quick::Bool)
    n = quick ? 90 : 220
    data = _point_cloud_fixture(n; seed=Int(0xD49C))
    spec = TamerOp.FiltrationSpec(
        kind=:rips,
        max_dim=1,
        axes=(collect(range(0.0, stop=2.0, length=24)),),
        knn=min(12, max(2, n - 1)),
        nn_backend=:auto,
        construction=TamerOp.ConstructionOptions(;
            sparsify=:knn,
            output_stage=:graded_complex,
            budget=(max_simplices=2_000_000, max_edges=2_000_000, memory_budget_bytes=8_000_000_000),
        ),
    )
    G = TamerOp.encode(data, spec; degree=0, cache=:auto, stage=:graded_complex)
    axes = spec.params[:axes]
    P = DI.poset_from_axes(axes)
    mkL() = DI._lazy_cochain_complex_from_graded_complex(G, P, axes; field=CM.QQField())
    return mkL, n
end

function _build_lazy_diff_fixture(; quick::Bool)
    n = quick ? 28 : 44
    data = _point_cloud_fixture(n; seed=Int(0xD49D))
    vals = [sin(2.0 * pi * (i - 1) / max(n - 1, 1)) for i in 1:n]
    spec = TamerOp.FiltrationSpec(
        kind=:function_rips,
        max_dim=2,
        vertex_values=vals,
        simplex_agg=:max,
        axes=(
            collect(range(0.0, stop=2.0, length=14)),
            collect(range(-1.0, stop=1.0, length=11)),
        ),
        construction=TamerOp.ConstructionOptions(;
            sparsify=:none,
            output_stage=:graded_complex,
            budget=(max_simplices=3_000_000, max_edges=3_000_000, memory_budget_bytes=8_000_000_000),
        ),
    )
    G = TamerOp.encode(data, spec; degree=0, cache=:auto, stage=:graded_complex)
    axes = spec.params[:axes]
    P = DI.poset_from_axes(axes)
    mkL() = DI._lazy_cochain_complex_from_graded_complex(G, P, axes; field=CM.QQField())
    return mkL, n
end

function _run_lowdim_h0_probe(; reps::Int, quick::Bool)
    mkL, n = _build_lazy_lowdim_h0_fixture(; quick=quick)
    println("\n[probe] low-dim H0 fast path  n=$(n), max_dim=1")
    old_min_pos = DI._H0_UNIONFIND_MIN_POS_VERTICES[]
    old_min_v = DI._H0_UNIONFIND_MIN_TOTAL_ACTIVE_VERTICES[]
    old_min_e = DI._H0_UNIONFIND_MIN_TOTAL_ACTIVE_EDGES[]
    fast = generic_via_dispatch = generic_direct = nothing
    try
        DI._H0_UNIONFIND_MIN_POS_VERTICES[] = 0
        DI._H0_UNIONFIND_MIN_TOTAL_ACTIVE_VERTICES[] = 0
        DI._H0_UNIONFIND_MIN_TOTAL_ACTIVE_EDGES[] = 0
        fast = _bench("  _cohomology_module_from_lazy (union-find forced)",
                      () -> begin
                          L = mkL()
                          DI._cohomology_module_from_lazy(L, 0)
                      end;
                      reps=reps)

        DI._H0_UNIONFIND_MIN_POS_VERTICES[] = typemax(Int)
        DI._H0_UNIONFIND_MIN_TOTAL_ACTIVE_VERTICES[] = typemax(Int)
        DI._H0_UNIONFIND_MIN_TOTAL_ACTIVE_EDGES[] = typemax(Int)
        generic_via_dispatch = _bench("  _cohomology_module_from_lazy (union-find disabled)",
                                      () -> begin
                                          L = mkL()
                                          DI._cohomology_module_from_lazy(L, 0)
                                      end;
                                      reps=reps)
        generic_direct = _bench("  _cohomology_module_from_lazy_generic",
                                () -> begin
                                    L = mkL()
                                    DI._cohomology_module_from_lazy_generic(L, 0)
                                end;
                                reps=reps)
    finally
        DI._H0_UNIONFIND_MIN_POS_VERTICES[] = old_min_pos
        DI._H0_UNIONFIND_MIN_TOTAL_ACTIVE_VERTICES[] = old_min_v
        DI._H0_UNIONFIND_MIN_TOTAL_ACTIVE_EDGES[] = old_min_e
    end
    println("  forced-fast/disabled-dispatch time ratio=", round(fast.med_ms / max(generic_via_dispatch.med_ms, 1e-12), digits=3), "x")
    println("  forced-fast/disabled-dispatch alloc ratio=", round(fast.med_kib / max(generic_via_dispatch.med_kib, 1e-12), digits=3), "x")
    println("  forced-fast/generic-direct time ratio=", round(fast.med_ms / max(generic_direct.med_ms, 1e-12), digits=3), "x")
    println("  forced-fast/generic-direct alloc ratio=", round(fast.med_kib / max(generic_direct.med_kib, 1e-12), digits=3), "x")
    return (
        (case="lowdim_h0_probe_rips_d1", family="point", stage="module", cache_mode="fastpath_forced",
         med_ms=fast.med_ms, p90_ms=fast.p90_ms, med_kib=fast.med_kib),
        (case="lowdim_h0_probe_rips_d1", family="point", stage="module", cache_mode="fastpath_disabled_dispatch",
         med_ms=generic_via_dispatch.med_ms, p90_ms=generic_via_dispatch.p90_ms, med_kib=generic_via_dispatch.med_kib),
        (case="lowdim_h0_probe_rips_d1", family="point", stage="module", cache_mode="generic_direct",
         med_ms=generic_direct.med_ms, p90_ms=generic_direct.p90_ms, med_kib=generic_direct.med_kib),
    )
end

function _run_lazy_diff_thread_probe(; reps::Int, quick::Bool)
    mkL, n = _build_lazy_diff_fixture(; quick=quick)
    println("\n[probe] lazy diff assembly threaded vs serial  n=$(n), max_dim=2, threads=$(Threads.nthreads())")
    serial = _bench("  _lazy_diff_components(threaded=false)",
                    () -> begin
                        L = mkL()
                        DI._lazy_ensure_active!(L, 1)
                        DI._lazy_ensure_active!(L, 2)
                        DI._lazy_diff_components(L, 1; threaded=false)
                    end;
                    reps=reps)
    threaded = _bench("  _lazy_diff_components(threaded=true)",
                      () -> begin
                          L = mkL()
                          DI._lazy_ensure_active!(L, 1)
                          DI._lazy_ensure_active!(L, 2)
                          DI._lazy_diff_components(L, 1; threaded=true)
                      end;
                      reps=reps)
    println("  threaded/serial time ratio=", round(threaded.med_ms / max(serial.med_ms, 1e-12), digits=3), "x")
    println("  threaded/serial alloc ratio=", round(threaded.med_kib / max(serial.med_kib, 1e-12), digits=3), "x")
    return (
        (case="lazy_diff_thread_probe_rips_d2", family="point", stage="cochain", cache_mode="serial",
         med_ms=serial.med_ms, p90_ms=serial.p90_ms, med_kib=serial.med_kib),
        (case="lazy_diff_thread_probe_rips_d2", family="point", stage="cochain", cache_mode="threaded",
         med_ms=threaded.med_ms, p90_ms=threaded.p90_ms, med_kib=threaded.med_kib),
    )
end

function _run_structural_inclusion_probe(; reps::Int, quick::Bool)
    n = quick ? 90 : 220
    data = _point_cloud_fixture(n; seed=Int(0xD4A2))
    spec = TamerOp.FiltrationSpec(
        kind=:rips,
        max_dim=1,
        axes=(collect(range(0.0, stop=2.2, length=30)),),
        knn=min(16, max(2, n - 1)),
        nn_backend=:auto,
        construction=TamerOp.ConstructionOptions(;
            sparsify=:knn,
            output_stage=:encoding_result,
            budget=(max_simplices=4_500_000, max_edges=4_000_000, memory_budget_bytes=8_000_000_000),
        ),
    )
    println("\n[probe] structural inclusion maps in full encode path  n=$(n), max_dim=1, degree=0")
    old_min_pos = DI._H0_UNIONFIND_MIN_POS_VERTICES[]
    old_min_v = DI._H0_UNIONFIND_MIN_TOTAL_ACTIVE_VERTICES[]
    old_min_e = DI._H0_UNIONFIND_MIN_TOTAL_ACTIVE_EDGES[]
    structural = nothing
    try
        DI._H0_UNIONFIND_MIN_POS_VERTICES[] = 0
        DI._H0_UNIONFIND_MIN_TOTAL_ACTIVE_VERTICES[] = 0
        DI._H0_UNIONFIND_MIN_TOTAL_ACTIVE_EDGES[] = 0
        structural = _bench("  encode(... structural inclusion maps)",
                            () -> begin
                                TamerOp.encode(data, spec; degree=0, cache=:auto, stage=:encoding_result)
                            end;
                            reps=reps)
    finally
        DI._H0_UNIONFIND_MIN_POS_VERTICES[] = old_min_pos
        DI._H0_UNIONFIND_MIN_TOTAL_ACTIVE_VERTICES[] = old_min_v
        DI._H0_UNIONFIND_MIN_TOTAL_ACTIVE_EDGES[] = old_min_e
    end
    return (
        (case="structural_inclusion_probe_rips_d1", family="point", stage="full", cache_mode="structural",
         med_ms=structural.med_ms, p90_ms=structural.p90_ms, med_kib=structural.med_kib),
    )
end

function _run_term_materialization_probe(; reps::Int, quick::Bool)
    n = quick ? 120 : 260
    data = _point_cloud_fixture(n; seed=Int(0xD4A3))
    spec = TamerOp.FiltrationSpec(
        kind=:rips,
        max_dim=1,
        axes=(collect(range(0.0, stop=2.0, length=24)),),
        knn=min(12, max(2, n - 1)),
        nn_backend=:auto,
        construction=TamerOp.ConstructionOptions(;
            sparsify=:knn,
            output_stage=:graded_complex,
            budget=(max_simplices=2_000_000, max_edges=2_000_000, memory_budget_bytes=8_000_000_000),
        ),
    )
    G = TamerOp.encode(data, spec; degree=0, cache=:auto, stage=:graded_complex)
    axes = spec.params[:axes]
    P = DI.poset_from_axes(axes)
    println("\n[probe] term materialization structural maps  n=$(n), max_dim=1")
    structural = _bench("  _lazy_term_idx!(structural inclusion)",
                        () -> begin
                            L = DI._lazy_cochain_complex_from_graded_complex(G, P, axes; field=CM.QQField())
                            DI._lazy_term_idx!(L, 1)
                        end;
                        reps=reps)
    return (
        (case="term_materialization_probe_rips_d1", family="point", stage="module", cache_mode="structural",
         med_ms=structural.med_ms, p90_ms=structural.p90_ms, med_kib=structural.med_kib),
    )
end

function _run_pointcloud_dense_stream_probe(; reps::Int, quick::Bool)
    n = quick ? 32 : 44
    data = _point_cloud_fixture(n; seed=Int(0xD4A4))
    spec = TamerOp.FiltrationSpec(
        kind=:rips,
        max_dim=2,
        construction=TamerOp.ConstructionOptions(;
            sparsify=:none,
            output_stage=:graded_complex,
            budget=(max_simplices=2_000_000, max_edges=2_000_000, memory_budget_bytes=8_000_000_000),
        ),
    )
    println("\n[probe] point-cloud non-sparse distance streaming  n=$(n), max_dim=2")
    old_stream = DI._POINTCLOUD_STREAM_DIST_NONSPARSE[]
    streamed = packed = nothing
    try
        DI._POINTCLOUD_STREAM_DIST_NONSPARSE[] = true
        streamed = _bench("  encode(... streamed simplex distances)",
                          () -> begin
                              TamerOp.encode(data, spec; degree=0, cache=:auto, stage=:graded_complex)
                          end;
                          reps=reps)
        DI._POINTCLOUD_STREAM_DIST_NONSPARSE[] = false
        packed = _bench("  encode(... packed pairwise fallback)",
                        () -> begin
                            TamerOp.encode(data, spec; degree=0, cache=:auto, stage=:graded_complex)
                        end;
                        reps=reps)
    finally
        DI._POINTCLOUD_STREAM_DIST_NONSPARSE[] = old_stream
    end
    println("  streamed/packed time ratio=", round(streamed.med_ms / max(packed.med_ms, 1e-12), digits=3), "x")
    println("  streamed/packed alloc ratio=", round(streamed.med_kib / max(packed.med_kib, 1e-12), digits=3), "x")
    return (
        (case="pointcloud_dense_stream_probe_rips_d2", family="point", stage="graded_complex", cache_mode="stream_on",
         med_ms=streamed.med_ms, p90_ms=streamed.p90_ms, med_kib=streamed.med_kib),
        (case="pointcloud_dense_stream_probe_rips_d2", family="point", stage="graded_complex", cache_mode="stream_off_packed",
         med_ms=packed.med_ms, p90_ms=packed.p90_ms, med_kib=packed.med_kib),
    )
end

function _run_pointcloud_lowdim_radius_stream_probe(; reps::Int, quick::Bool)
    n = quick ? 900 : 2000
    radius = quick ? 1.5 : 2.0
    density_k = 16
    budget = (max_simplices=6_000_000, max_edges=6_000_000, memory_budget_bytes=8_000_000_000)

    function _run_dim_probe(d::Int)
        rng = MersenneTwister(Int(0xD4A4F + 31 * d))
        pts = [randn(rng, d) for _ in 1:n]
        data = TamerOp.PointCloud(pts)
        spec = TamerOp.FiltrationSpec(
            kind=:rips_density,
            max_dim=1,
            radius=radius,
            density_k=density_k,
            nn_backend=:auto,
            construction=TamerOp.ConstructionOptions(;
                sparsify=:none,
                output_stage=:simplex_tree,
                budget=budget,
            ),
        )
        old_stream = DI._POINTCLOUD_LOWDIM_RADIUS_STREAMING[]
        streamed = dense = nothing
        try
            DI._POINTCLOUD_LOWDIM_RADIUS_STREAMING[] = true
            streamed = _bench("  lowdim radius stream on  (d=$(d))",
                              () -> TamerOp.encode(data, spec; degree=0, cache=:auto, stage=:simplex_tree);
                              reps=reps)
            DI._POINTCLOUD_LOWDIM_RADIUS_STREAMING[] = false
            dense = _bench("  lowdim radius stream off (d=$(d))",
                           () -> TamerOp.encode(data, spec; degree=0, cache=:auto, stage=:simplex_tree);
                           reps=reps)
        finally
            DI._POINTCLOUD_LOWDIM_RADIUS_STREAMING[] = old_stream
        end
        println("  stream_on/stream_off time ratio (d=$(d))=", round(streamed.med_ms / max(dense.med_ms, 1e-12), digits=3), "x")
        println("  stream_on/stream_off alloc ratio (d=$(d))=", round(streamed.med_kib / max(dense.med_kib, 1e-12), digits=3), "x")
        return (
            (case="pointcloud_lowdim_radius_stream_probe_d$(d)", family="point", stage="simplex_tree", cache_mode="stream_on",
             med_ms=streamed.med_ms, p90_ms=streamed.p90_ms, med_kib=streamed.med_kib),
            (case="pointcloud_lowdim_radius_stream_probe_d$(d)", family="point", stage="simplex_tree", cache_mode="stream_off_densefilter",
             med_ms=dense.med_ms, p90_ms=dense.p90_ms, med_kib=dense.med_kib),
        )
    end

    println("\n[probe] point-cloud lowdim finite-radius streaming  n=$(n), radius=$(radius), k=$(density_k)")
    out2 = _run_dim_probe(2)
    out8 = _run_dim_probe(8)
    return (out2..., out8...)
end

function _run_pointcloud_dim2_kernel_probe(; reps::Int, quick::Bool)
    n = quick ? 72 : 120
    data = _point_cloud_fixture(n; seed=Int(0xD4A4C))
    spec = TamerOp.FiltrationSpec(
        kind=:rips,
        max_dim=2,
        radius=0.36,
        construction=TamerOp.ConstructionOptions(;
            sparsify=:none,
            collapse=:none,
            output_stage=:simplex_tree,
            budget=(max_simplices=4_000_000, max_edges=2_000_000, memory_budget_bytes=8_000_000_000),
        ),
    )
    println("\n[probe] point-cloud dim2 packed kernel  n=$(n), max_dim=2")
    old_kernel = DI._POINTCLOUD_DIM2_PACKED_KERNEL[]
    packed_on = packed_off = nothing
    try
        DI._POINTCLOUD_DIM2_PACKED_KERNEL[] = true
        packed_on = _bench("  encode(... dim2 packed kernel ON)",
                           () -> begin
                               TamerOp.encode(data, spec; degree=0, cache=:auto, stage=:simplex_tree)
                           end;
                           reps=reps)
        DI._POINTCLOUD_DIM2_PACKED_KERNEL[] = false
        packed_off = _bench("  encode(... dim2 packed kernel OFF)",
                            () -> begin
                                TamerOp.encode(data, spec; degree=0, cache=:auto, stage=:simplex_tree)
                            end;
                            reps=reps)
    finally
        DI._POINTCLOUD_DIM2_PACKED_KERNEL[] = old_kernel
    end
    println("  on/off time ratio=", round(packed_on.med_ms / max(packed_off.med_ms, 1e-12), digits=3), "x")
    println("  on/off alloc ratio=", round(packed_on.med_kib / max(packed_off.med_kib, 1e-12), digits=3), "x")
    return (
        (case="pointcloud_dim2_kernel_probe_rips_d2", family="point", stage="simplex_tree", cache_mode="packed_on",
         med_ms=packed_on.med_ms, p90_ms=packed_on.p90_ms, med_kib=packed_on.med_kib),
        (case="pointcloud_dim2_kernel_probe_rips_d2", family="point", stage="simplex_tree", cache_mode="packed_off",
         med_ms=packed_off.med_ms, p90_ms=packed_off.p90_ms, med_kib=packed_off.med_kib),
    )
end

function _run_delaunay_cache_probe(; reps::Int, quick::Bool)
    n = quick ? 320 : 960
    data = _point_cloud_fixture(n; seed=Int(0xD4B0))
    spec = TamerOp.FiltrationSpec(
        kind=:alpha,
        max_dim=2,
        delaunay_backend=:auto,
        construction=TamerOp.ConstructionOptions(;
            output_stage=:simplex_tree,
            budget=(max_simplices=4_000_000, max_edges=4_000_000, memory_budget_bytes=8_000_000_000),
        ),
    )
    println("\n[probe] Delaunay packed cache behavior (alpha, n=$(n))")
    old_cache = DI._POINTCLOUD_DELAUNAY_CACHE_ENABLED[]
    miss = hit = uncached = nothing
    try
        DI._POINTCLOUD_DELAUNAY_CACHE_ENABLED[] = true
        DI._clear_pointcloud_delaunay_cache!()
        miss = _bench("  packed Delaunay cache miss",
                      () -> begin
                          DI._clear_pointcloud_delaunay_cache!()
                          DI._packed_delaunay_simplices(data.points, spec; max_dim=2)
                          nothing
                      end;
                      reps=reps)

        DI._clear_pointcloud_delaunay_cache!()
        DI._packed_delaunay_simplices(data.points, spec; max_dim=2)
        hit = _bench("  packed Delaunay cache hit",
                     () -> begin
                         DI._packed_delaunay_simplices(data.points, spec; max_dim=2)
                         nothing
                     end;
                     reps=reps)

        DI._POINTCLOUD_DELAUNAY_CACHE_ENABLED[] = false
        uncached = _bench("  packed Delaunay cache disabled",
                          () -> begin
                              DI._packed_delaunay_simplices(data.points, spec; max_dim=2)
                              nothing
                          end;
                          reps=reps)
    finally
        DI._POINTCLOUD_DELAUNAY_CACHE_ENABLED[] = old_cache
        DI._clear_pointcloud_delaunay_cache!()
    end
    println("  hit/miss time ratio=", round(hit.med_ms / max(miss.med_ms, 1e-12), digits=4), "x")
    println("  hit/uncached time ratio=", round(hit.med_ms / max(uncached.med_ms, 1e-12), digits=4), "x")
    return (
        (case="delaunay_cache_probe_alpha_d2", family="point", stage="simplex_tree", cache_mode="cache_miss",
         med_ms=miss.med_ms, p90_ms=miss.p90_ms, med_kib=miss.med_kib),
        (case="delaunay_cache_probe_alpha_d2", family="point", stage="simplex_tree", cache_mode="cache_hit",
         med_ms=hit.med_ms, p90_ms=hit.p90_ms, med_kib=hit.med_kib),
        (case="delaunay_cache_probe_alpha_d2", family="point", stage="simplex_tree", cache_mode="cache_disabled",
         med_ms=uncached.med_ms, p90_ms=uncached.p90_ms, med_kib=uncached.med_kib),
    )
end

function _run_delaunay_simplextree_fusion_probe(; reps::Int, quick::Bool)
    n = quick ? 1200 : 3200
    data = _point_cloud_fixture(n; seed=Int(0xD4B2))
    pts = data.points
    vvals = [p[1] + 2p[2] for p in pts]
    aspec = TamerOp.FiltrationSpec(
        kind=:alpha,
        max_dim=2,
        delaunay_backend=:auto,
        construction=TamerOp.ConstructionOptions(;
            output_stage=:simplex_tree,
            budget=(max_simplices=4_000_000, max_edges=4_000_000, memory_budget_bytes=8_000_000_000),
        ),
    )
    dspec = TamerOp.FiltrationSpec(
        kind=:delaunay_lower_star,
        max_dim=2,
        delaunay_backend=:auto,
        vertex_values=vvals,
        simplex_agg=:max,
        construction=TamerOp.ConstructionOptions(;
            output_stage=:simplex_tree,
            budget=(max_simplices=4_000_000, max_edges=4_000_000, memory_budget_bytes=8_000_000_000),
        ),
    )

    old_alpha = () -> begin
        packed = DI._packed_delaunay_simplices(pts, aspec; max_dim=2)
        ne = length(packed.edges)
        nt = length(packed.triangles)
        grades = Vector{NTuple{1,Float64}}(undef, n + ne + nt)
        t = 1
        @inbounds for _ in 1:n
            grades[t] = (0.0,)
            t += 1
        end
        @inbounds for idx in 1:ne
            grades[t] = (packed.edge_radius[idx],)
            t += 1
        end
        @inbounds for idx in 1:nt
            grades[t] = (packed.tri_radius[idx],)
            t += 1
        end
        DI._materialize_point_cloud_dim012(n, packed.edges, packed.triangles, grades, aspec; return_simplex_tree=true)
    end
    new_alpha = () -> TamerOp.encode(data, aspec; degree=0, stage=:simplex_tree)

    old_delaunay = () -> begin
        packed = DI._packed_delaunay_simplices(pts, dspec; max_dim=2)
        ne = length(packed.edges)
        nt = length(packed.triangles)
        grades = Vector{NTuple{1,Float64}}(undef, n + ne + nt)
        t = 1
        @inbounds for v in 1:n
            grades[t] = (vvals[v],)
            t += 1
        end
        @inbounds for idx in 1:ne
            i, j = packed.edges[idx]
            grades[t] = (DI._aggregate_pair(vvals[i], vvals[j], :max),)
            t += 1
        end
        @inbounds for idx in 1:nt
            i, j, k = packed.triangles[idx]
            grades[t] = (DI._aggregate_triple(vvals[i], vvals[j], vvals[k], :max),)
            t += 1
        end
        DI._materialize_point_cloud_dim012(n, packed.edges, packed.triangles, grades, dspec; return_simplex_tree=true)
    end
    new_delaunay = () -> TamerOp.encode(data, dspec; degree=0, stage=:simplex_tree)

    println("\n[probe] Delaunay simplex-tree fused path (n=$(n))")
    alpha_old = _bench("  alpha simplex_tree old-emulated", old_alpha; reps=reps)
    alpha_new = _bench("  alpha simplex_tree fused", new_alpha; reps=reps)
    println("  alpha fused/old ratio=", round(alpha_new.med_ms / max(alpha_old.med_ms, 1e-12), digits=3), "x")
    del_old = _bench("  delaunay simplex_tree old-emulated", old_delaunay; reps=reps)
    del_new = _bench("  delaunay simplex_tree fused", new_delaunay; reps=reps)
    println("  delaunay fused/old ratio=", round(del_new.med_ms / max(del_old.med_ms, 1e-12), digits=3), "x")
    return (
        (case="delaunay_simplextree_fusion_probe_alpha", family="point", stage="simplex_tree", cache_mode="old_emulated",
         med_ms=alpha_old.med_ms, p90_ms=alpha_old.p90_ms, med_kib=alpha_old.med_kib),
        (case="delaunay_simplextree_fusion_probe_alpha", family="point", stage="simplex_tree", cache_mode="fused",
         med_ms=alpha_new.med_ms, p90_ms=alpha_new.p90_ms, med_kib=alpha_new.med_kib),
        (case="delaunay_simplextree_fusion_probe_delaunay", family="point", stage="simplex_tree", cache_mode="old_emulated",
         med_ms=del_old.med_ms, p90_ms=del_old.p90_ms, med_kib=del_old.med_kib),
        (case="delaunay_simplextree_fusion_probe_delaunay", family="point", stage="simplex_tree", cache_mode="fused",
         med_ms=del_new.med_ms, p90_ms=del_new.p90_ms, med_kib=del_new.med_kib),
    )
end

function _run_core_numbers_probe(; reps::Int, quick::Bool)
    function _core_numbers_naive(n::Int, edges::Vector{NTuple{2,Int}})
        adj = [Int[] for _ in 1:n]
        for (u, v) in edges
            u == v && continue
            push!(adj[u], v)
            push!(adj[v], u)
        end
        deg = [length(adj[v]) for v in 1:n]
        alive = trues(n)
        core = zeros(Int, n)
        remaining = n
        k = 0
        while remaining > 0
            peeled = false
            for v in 1:n
                if alive[v] && deg[v] <= k
                    alive[v] = false
                    core[v] = k
                    remaining -= 1
                    peeled = true
                    for w in adj[v]
                        alive[w] && (deg[w] -= 1)
                    end
                end
            end
            peeled || (k += 1)
        end
        return core
    end

    n = quick ? 6000 : 12000
    m_extra = quick ? 20_000 : 60_000
    rng = MersenneTwister(Int(0xD4B3))
    edges = NTuple{2,Int}[]
    seen = Set{NTuple{2,Int}}()
    sizehint!(edges, n + m_extra)
    sizehint!(seen, n + m_extra)
    for i in 1:(n - 1)
        e = (i, i + 1)
        push!(edges, e)
        push!(seen, e)
    end
    while length(edges) < (n - 1 + m_extra)
        u = rand(rng, 1:n)
        v = rand(rng, 1:n)
        u == v && continue
        a, b = min(u, v), max(u, v)
        e = (a, b)
        e in seen && continue
        push!(seen, e)
        push!(edges, e)
    end

    println("\n[probe] core-number kernel (n=$(n), m=$(length(edges)))")
    naive = _bench("  core_numbers naive",
                   () -> _core_numbers_naive(n, edges);
                   reps=reps)
    linear = _bench("  core_numbers linear",
                    () -> DI._core_numbers(n, edges);
                    reps=reps)
    println("  linear/naive time ratio=", round(linear.med_ms / max(naive.med_ms, 1e-12), digits=3), "x")
    return (
        (case="core_numbers_probe", family="graph", stage="graded_complex", cache_mode="naive",
         med_ms=naive.med_ms, p90_ms=naive.p90_ms, med_kib=naive.med_kib),
        (case="core_numbers_probe", family="graph", stage="graded_complex", cache_mode="linear",
         med_ms=linear.med_ms, p90_ms=linear.p90_ms, med_kib=linear.med_kib),
    )
end

function _run_landmark_radius_probe(; reps::Int, quick::Bool)
    n = quick ? 10_000 : 50_000
    m = quick ? 160 : 224
    radius = quick ? 0.24 : 0.24
    data = _point_cloud_fixture(n; seed=Int(0xD4B5))
    rng = MersenneTwister(Int(0xD4B6))
    landmarks = sort!(randperm(rng, n)[1:m])

    spec_auto = TamerOp.FiltrationSpec(
        kind=:landmark_rips,
        max_dim=1,
        landmarks=landmarks,
        radius=radius,
        construction=TamerOp.ConstructionOptions(;
            output_stage=:simplex_tree,
        ),
    )
    spec_explicit = TamerOp.FiltrationSpec(
        kind=:landmark_rips,
        max_dim=1,
        landmarks=landmarks,
        radius=radius,
        construction=TamerOp.ConstructionOptions(;
            sparsify=:radius,
            output_stage=:simplex_tree,
        ),
    )

    # Contract parity: default landmark+radius normalization should match explicit sparse radius.
    st_auto = TamerOp.encode(data, spec_auto; degree=0, stage=:simplex_tree)
    st_explicit = TamerOp.encode(data, spec_explicit; degree=0, stage=:simplex_tree)
    DI.simplex_count(st_auto) == DI.simplex_count(st_explicit) ||
        error("landmark_radius_probe parity mismatch: normalized vs explicit sparse radius simplex counts differ.")

    println("\n[probe] landmark+radii normalization + subgraph cache (n=$(n), m=$(m), radius=$(radius))")
    auto = _bench("  landmark auto-normalized",
                  () -> TamerOp.encode(data, spec_auto; degree=0, stage=:simplex_tree, cache=:auto);
                  reps=reps)
    explicit = _bench("  landmark explicit sparse radius",
                      () -> TamerOp.encode(data, spec_explicit; degree=0, stage=:simplex_tree, cache=:auto);
                      reps=reps)
    println("  auto/explicit time ratio=", round(auto.med_ms / max(explicit.med_ms, 1e-12), digits=3), "x")

    sc = CM.SessionCache()
    miss = _bench("  landmark subgraph cache miss",
                  () -> begin
                      CM._clear_session_cache!(sc)
                      TamerOp.encode(data, spec_auto; degree=0, stage=:simplex_tree, cache=sc)
                      nothing
                  end;
                  reps=reps)
    TamerOp.encode(data, spec_auto; degree=0, stage=:simplex_tree, cache=sc)
    hit = _bench("  landmark subgraph cache hit",
                 () -> begin
                     TamerOp.encode(data, spec_auto; degree=0, stage=:simplex_tree, cache=sc)
                     nothing
                 end;
                 reps=reps)
    println("  cache hit/miss time ratio=", round(hit.med_ms / max(miss.med_ms, 1e-12), digits=3), "x")

    return (
        (case="landmark_radius_probe", family="point", stage="simplex_tree", cache_mode="auto_normalized",
         med_ms=auto.med_ms, p90_ms=auto.p90_ms, med_kib=auto.med_kib),
        (case="landmark_radius_probe", family="point", stage="simplex_tree", cache_mode="explicit_sparse_radius",
         med_ms=explicit.med_ms, p90_ms=explicit.p90_ms, med_kib=explicit.med_kib),
        (case="landmark_radius_probe", family="point", stage="simplex_tree", cache_mode="session_cache_miss",
         med_ms=miss.med_ms, p90_ms=miss.p90_ms, med_kib=miss.med_kib),
        (case="landmark_radius_probe", family="point", stage="simplex_tree", cache_mode="session_cache_hit",
         med_ms=hit.med_ms, p90_ms=hit.p90_ms, med_kib=hit.med_kib),
    )
end

function _run_cubical_2d_probe(; reps::Int, quick::Bool)
    side = quick ? 80 : 128
    data, mask = _image_fixture(side; seed=Int(0xD4C1))
    spec_cub = TamerOp.FiltrationSpec(
        kind=:cubical,
        construction=TamerOp.ConstructionOptions(; output_stage=:graded_complex),
    )
    spec_bi = TamerOp.FiltrationSpec(
        kind=:image_distance_bifiltration,
        mask=mask,
        construction=TamerOp.ConstructionOptions(; output_stage=:graded_complex),
    )

    old_fast = DI._CUBICAL_2D_FASTPATH[]
    fast_cub = slow_cub = fast_bi = slow_bi = nothing
    try
        DI._CUBICAL_2D_FASTPATH[] = false
        G_cub_slow = TamerOp.encode(data, spec_cub; degree=0, stage=:graded_complex, cache=:auto)
        G_bi_slow = TamerOp.encode(data, spec_bi; degree=0, stage=:graded_complex, cache=:auto)

        DI._CUBICAL_2D_FASTPATH[] = true
        G_cub_fast = TamerOp.encode(data, spec_cub; degree=0, stage=:graded_complex, cache=:auto)
        G_bi_fast = TamerOp.encode(data, spec_bi; degree=0, stage=:graded_complex, cache=:auto)

        (G_cub_fast.cells_by_dim == G_cub_slow.cells_by_dim &&
         G_cub_fast.boundaries == G_cub_slow.boundaries &&
         G_cub_fast.grades == G_cub_slow.grades) ||
            error("cubical_2d_probe parity mismatch: cubical fast path output differs from generic.")
        (G_bi_fast.cells_by_dim == G_bi_slow.cells_by_dim &&
         G_bi_fast.boundaries == G_bi_slow.boundaries &&
         G_bi_fast.grades == G_bi_slow.grades) ||
            error("cubical_2d_probe parity mismatch: image_distance fast path output differs from generic.")

        DI._CUBICAL_2D_FASTPATH[] = true
        fast_cub = _bench("  cubical 2D fast path on",
                          () -> TamerOp.encode(data, spec_cub; degree=0, stage=:graded_complex, cache=:auto);
                          reps=reps)
        fast_bi = _bench("  image_distance 2D fast path on",
                         () -> TamerOp.encode(data, spec_bi; degree=0, stage=:graded_complex, cache=:auto);
                         reps=reps)

        DI._CUBICAL_2D_FASTPATH[] = false
        slow_cub = _bench("  cubical 2D fast path off (generic)",
                          () -> TamerOp.encode(data, spec_cub; degree=0, stage=:graded_complex, cache=:auto);
                          reps=reps)
        slow_bi = _bench("  image_distance 2D fast path off (generic)",
                         () -> TamerOp.encode(data, spec_bi; degree=0, stage=:graded_complex, cache=:auto);
                         reps=reps)
    finally
        DI._CUBICAL_2D_FASTPATH[] = old_fast
    end

    println("\n[probe] cubical/image 2D fast path (side=$(side))")
    println("  cubical on/off time ratio=", round(fast_cub.med_ms / max(slow_cub.med_ms, 1e-12), digits=3), "x")
    println("  image_distance on/off time ratio=", round(fast_bi.med_ms / max(slow_bi.med_ms, 1e-12), digits=3), "x")

    return (
        (case="cubical_2d_probe", family="image", stage="graded_complex", cache_mode="fast_on_cubical",
         med_ms=fast_cub.med_ms, p90_ms=fast_cub.p90_ms, med_kib=fast_cub.med_kib),
        (case="cubical_2d_probe", family="image", stage="graded_complex", cache_mode="fast_off_cubical",
         med_ms=slow_cub.med_ms, p90_ms=slow_cub.p90_ms, med_kib=slow_cub.med_kib),
        (case="cubical_2d_probe", family="image", stage="graded_complex", cache_mode="fast_on_image_distance",
         med_ms=fast_bi.med_ms, p90_ms=fast_bi.p90_ms, med_kib=fast_bi.med_kib),
        (case="cubical_2d_probe", family="image", stage="graded_complex", cache_mode="fast_off_image_distance",
         med_ms=slow_bi.med_ms, p90_ms=slow_bi.p90_ms, med_kib=slow_bi.med_kib),
    )
end

function _run_graph_clique_probe(; reps::Int, quick::Bool)
    n = quick ? 24 : 34
    rng = MersenneTwister(Int(0xD4A5))
    edges = Tuple{Int,Int}[]
    for i in 1:(n - 1), j in (i + 1):n
        rand(rng) <= (quick ? 0.23 : 0.19) && push!(edges, (i, j))
    end
    isempty(edges) && push!(edges, (1, 2))
    data = TamerOp.GraphData(n, edges)
    vertex_vals = [sin(0.13 * i) for i in 1:n]
    edge_weights = [1.0 + abs(cos(0.17 * i)) for i in 1:length(edges)]
    cons = TamerOp.ConstructionOptions(;
        output_stage=:graded_complex,
        budget=(max_simplices=2_000_000, max_edges=500_000, memory_budget_bytes=8_000_000_000),
    )
    spec_clique = TamerOp.FiltrationSpec(
        kind=:clique_lower_star,
        max_dim=2,
        vertex_values=vertex_vals,
        simplex_agg=:max,
        construction=cons,
    )
    spec_w = TamerOp.FiltrationSpec(
        kind=:graph_weight_threshold,
        lift=:clique,
        max_dim=2,
        edge_weights=edge_weights,
        construction=cons,
    )
    println("\n[probe] graph clique enumeration (auto vs forced modes)  n=$(n), |E|=$(length(edges)), max_dim=2")
    old = DI._GRAPH_CLIQUE_ENUM_MODE[]
    c_auto = c_inter = c_comb = nothing
    w_auto = w_inter = w_comb = nothing
    try
        DI._GRAPH_CLIQUE_ENUM_MODE[] = :auto
        c_auto = _bench("  clique_lower_star (auto)",
                        () -> TamerOp.encode(data, spec_clique; degree=0, cache=:auto, stage=:graded_complex);
                        reps=reps)
        w_auto = _bench("  graph_weight_threshold:lq (auto)",
                        () -> TamerOp.encode(data, spec_w; degree=0, cache=:auto, stage=:graded_complex);
                        reps=reps)

        DI._GRAPH_CLIQUE_ENUM_MODE[] = :intersection
        c_inter = _bench("  clique_lower_star (intersection)",
                        () -> TamerOp.encode(data, spec_clique; degree=0, cache=:auto, stage=:graded_complex);
                        reps=reps)
        w_inter = _bench("  graph_weight_threshold:lq (intersection)",
                        () -> TamerOp.encode(data, spec_w; degree=0, cache=:auto, stage=:graded_complex);
                        reps=reps)
        DI._GRAPH_CLIQUE_ENUM_MODE[] = :combinations
        c_comb = _bench("  clique_lower_star (combinations)",
                        () -> TamerOp.encode(data, spec_clique; degree=0, cache=:auto, stage=:graded_complex);
                        reps=reps)
        w_comb = _bench("  graph_weight_threshold:lq (combinations)",
                        () -> TamerOp.encode(data, spec_w; degree=0, cache=:auto, stage=:graded_complex);
                        reps=reps)
    finally
        DI._GRAPH_CLIQUE_ENUM_MODE[] = old
    end
    println("  clique auto/intersection time ratio=", round(c_auto.med_ms / max(c_inter.med_ms, 1e-12), digits=3), "x")
    println("  clique auto/combinations time ratio=", round(c_auto.med_ms / max(c_comb.med_ms, 1e-12), digits=3), "x")
    println("  weight-threshold auto/intersection time ratio=", round(w_auto.med_ms / max(w_inter.med_ms, 1e-12), digits=3), "x")
    println("  weight-threshold auto/combinations time ratio=", round(w_auto.med_ms / max(w_comb.med_ms, 1e-12), digits=3), "x")
    return (
        (case="graph_clique_probe_lower_star", family="graph", stage="graded_complex", cache_mode="auto",
         med_ms=c_auto.med_ms, p90_ms=c_auto.p90_ms, med_kib=c_auto.med_kib),
        (case="graph_clique_probe_lower_star", family="graph", stage="graded_complex", cache_mode="intersection",
         med_ms=c_inter.med_ms, p90_ms=c_inter.p90_ms, med_kib=c_inter.med_kib),
        (case="graph_clique_probe_lower_star", family="graph", stage="graded_complex", cache_mode="combinations",
         med_ms=c_comb.med_ms, p90_ms=c_comb.p90_ms, med_kib=c_comb.med_kib),
        (case="graph_clique_probe_weight_threshold", family="graph", stage="graded_complex", cache_mode="auto",
         med_ms=w_auto.med_ms, p90_ms=w_auto.p90_ms, med_kib=w_auto.med_kib),
        (case="graph_clique_probe_weight_threshold", family="graph", stage="graded_complex", cache_mode="intersection",
         med_ms=w_inter.med_ms, p90_ms=w_inter.p90_ms, med_kib=w_inter.med_kib),
        (case="graph_clique_probe_weight_threshold", family="graph", stage="graded_complex", cache_mode="combinations",
         med_ms=w_comb.med_ms, p90_ms=w_comb.p90_ms, med_kib=w_comb.med_kib),
    )
end

function _run_nn_backend_probe(; reps::Int, quick::Bool)
    if !_HAVE_POINTCLOUD_NN_BACKEND
        println("\n[probe] nn backend parity/speed: skipped (NearestNeighbors extension unavailable)")
        return NamedTuple[]
    end
    n = quick ? 320 : 1200
    d = quick ? 18 : 28
    rng = MersenneTwister(Int(0xD4A9))
    pts = [randn(rng, d) for _ in 1:n]
    data = TamerOp.PointCloud(pts)
    k = min(12, max(2, n - 1))
    budget = (max_simplices=4_000_000, max_edges=4_000_000, memory_budget_bytes=8_000_000_000)
    mk_spec(backend::Symbol; approx_candidates::Int=0) = TamerOp.FiltrationSpec(
        kind=:rips,
        max_dim=1,
        knn=k,
        nn_backend=backend,
        nn_approx_candidates=approx_candidates,
        construction=TamerOp.ConstructionOptions(; sparsify=:knn, output_stage=:graded_complex, budget=budget),
    )

    println("\n[probe] nn backend speed/parity (kNN sparse graph)  n=$(n), d=$(d), k=$(k)")
    bf = _bench("  nn_backend=:bruteforce",
                () -> TamerOp.encode(data, mk_spec(:bruteforce); degree=0, cache=:auto, stage=:graded_complex);
                reps=reps)
    nn = _bench("  nn_backend=:nearestneighbors",
                () -> TamerOp.encode(data, mk_spec(:nearestneighbors); degree=0, cache=:auto, stage=:graded_complex);
                reps=reps)
    ap = _bench("  nn_backend=:approx",
                () -> TamerOp.encode(data, mk_spec(:approx; approx_candidates=max(64, 4k)); degree=0, cache=:auto, stage=:graded_complex);
                reps=reps)
    au = _bench("  nn_backend=:auto",
                () -> TamerOp.encode(data, mk_spec(:auto); degree=0, cache=:auto, stage=:graded_complex);
                reps=reps)
    println("  nearestneighbors/bruteforce time ratio=", round(nn.med_ms / max(bf.med_ms, 1e-12), digits=3), "x")
    println("  approx/bruteforce time ratio=", round(ap.med_ms / max(bf.med_ms, 1e-12), digits=3), "x")
    println("  auto/bruteforce time ratio=", round(au.med_ms / max(bf.med_ms, 1e-12), digits=3), "x")
    return (
        (case="nn_backend_probe_rips_knn", family="point", stage="graded_complex", cache_mode="bruteforce",
         med_ms=bf.med_ms, p90_ms=bf.p90_ms, med_kib=bf.med_kib),
        (case="nn_backend_probe_rips_knn", family="point", stage="graded_complex", cache_mode="nearestneighbors",
         med_ms=nn.med_ms, p90_ms=nn.p90_ms, med_kib=nn.med_kib),
        (case="nn_backend_probe_rips_knn", family="point", stage="graded_complex", cache_mode="approx",
         med_ms=ap.med_ms, p90_ms=ap.p90_ms, med_kib=ap.med_kib),
        (case="nn_backend_probe_rips_knn", family="point", stage="graded_complex", cache_mode="auto",
         med_ms=au.med_ms, p90_ms=au.p90_ms, med_kib=au.med_kib),
    )
end

function _run_h0_chain_sweep_probe(; reps::Int, quick::Bool)
    n = quick ? 120 : 260
    data = _point_cloud_fixture(n; seed=Int(0xD4A6))
    spec = TamerOp.FiltrationSpec(
        kind=:rips,
        max_dim=1,
        axes=(collect(range(0.0, stop=2.4, length=36)),),
        knn=min(16, max(2, n - 1)),
        nn_backend=:auto,
        construction=TamerOp.ConstructionOptions(;
            sparsify=:knn,
            output_stage=:encoding_result,
            budget=(max_simplices=5_000_000, max_edges=5_000_000, memory_budget_bytes=8_000_000_000),
        ),
    )

    g_n = quick ? 140 : 320
    rng = MersenneTwister(Int(0xD4A7))
    g_edges = Tuple{Int,Int}[]
    for i in 1:(g_n - 1), j in (i + 1):g_n
        rand(rng) <= (quick ? 0.022 : 0.0135) && push!(g_edges, (i, j))
    end
    isempty(g_edges) && push!(g_edges, (1, 2))
    g_w = [0.01 + rand(rng) for _ in 1:length(g_edges)]
    g_data = TamerOp.GraphData(g_n, g_edges; weights=g_w)
    g_spec = TamerOp.FiltrationSpec(
        kind=:graph_weight_threshold,
        max_dim=1,
        construction=TamerOp.ConstructionOptions(;
            output_stage=:encoding_result,
            budget=(max_simplices=5_000_000, max_edges=5_000_000, memory_budget_bytes=8_000_000_000),
        ),
    )

    println("\n[probe] H0 chain-sweep fast path (point + graph-weight-threshold)")
    old_chain = DI._H0_CHAIN_SWEEP_FASTPATH[]
    old_min_pos = DI._H0_UNIONFIND_MIN_POS_VERTICES[]
    old_min_v = DI._H0_UNIONFIND_MIN_TOTAL_ACTIVE_VERTICES[]
    old_min_e = DI._H0_UNIONFIND_MIN_TOTAL_ACTIVE_EDGES[]
    p_fast = p_base = g_fast = g_base = nothing
    try
        DI._H0_UNIONFIND_MIN_POS_VERTICES[] = 0
        DI._H0_UNIONFIND_MIN_TOTAL_ACTIVE_VERTICES[] = 0
        DI._H0_UNIONFIND_MIN_TOTAL_ACTIVE_EDGES[] = 0

        DI._H0_CHAIN_SWEEP_FASTPATH[] = true
        p_fast = _bench("  point rips d1 (chain-sweep on)",
                        () -> TamerOp.encode(data, spec; degree=0, cache=:auto, stage=:encoding_result);
                        reps=reps)
        g_fast = _bench("  graph weight-threshold d1 (chain-sweep on)",
                        () -> TamerOp.encode(g_data, g_spec; degree=0, cache=:auto, stage=:encoding_result);
                        reps=reps)

        DI._H0_CHAIN_SWEEP_FASTPATH[] = false
        p_base = _bench("  point rips d1 (chain-sweep off)",
                        () -> TamerOp.encode(data, spec; degree=0, cache=:auto, stage=:encoding_result);
                        reps=reps)
        g_base = _bench("  graph weight-threshold d1 (chain-sweep off)",
                        () -> TamerOp.encode(g_data, g_spec; degree=0, cache=:auto, stage=:encoding_result);
                        reps=reps)
    finally
        DI._H0_CHAIN_SWEEP_FASTPATH[] = old_chain
        DI._H0_UNIONFIND_MIN_POS_VERTICES[] = old_min_pos
        DI._H0_UNIONFIND_MIN_TOTAL_ACTIVE_VERTICES[] = old_min_v
        DI._H0_UNIONFIND_MIN_TOTAL_ACTIVE_EDGES[] = old_min_e
    end
    println("  point chain-on/off time ratio=", round(p_fast.med_ms / max(p_base.med_ms, 1e-12), digits=3), "x")
    println("  graph-threshold chain-on/off time ratio=", round(g_fast.med_ms / max(g_base.med_ms, 1e-12), digits=3), "x")
    return (
        (case="h0_chain_sweep_probe_point_rips_d1", family="point", stage="full", cache_mode="chain_on",
         med_ms=p_fast.med_ms, p90_ms=p_fast.p90_ms, med_kib=p_fast.med_kib),
        (case="h0_chain_sweep_probe_point_rips_d1", family="point", stage="full", cache_mode="chain_off",
         med_ms=p_base.med_ms, p90_ms=p_base.p90_ms, med_kib=p_base.med_kib),
        (case="h0_chain_sweep_probe_graph_weight_threshold_d1", family="graph", stage="full", cache_mode="chain_on",
         med_ms=g_fast.med_ms, p90_ms=g_fast.p90_ms, med_kib=g_fast.med_kib),
        (case="h0_chain_sweep_probe_graph_weight_threshold_d1", family="graph", stage="full", cache_mode="chain_off",
         med_ms=g_base.med_ms, p90_ms=g_base.p90_ms, med_kib=g_base.med_kib),
    )
end

function _run_h0_active_chain_incremental_probe(; reps::Int, quick::Bool)
    n = quick ? 110 : 260
    data = _point_cloud_fixture(n; seed=Int(0xD4AE))
    spec_gc = TamerOp.FiltrationSpec(
        kind=:rips,
        max_dim=1,
        axes=(collect(range(0.0, stop=2.4, length=(quick ? 26 : 40))),),
        knn=min(16, max(2, n - 1)),
        nn_backend=:auto,
        construction=TamerOp.ConstructionOptions(;
            sparsify=:knn,
            output_stage=:graded_complex,
            budget=(max_simplices=5_000_000, max_edges=5_000_000, memory_budget_bytes=8_000_000_000),
        ),
    )
    G = TamerOp.encode(data, spec_gc; degree=0)
    axes = spec_gc.params[:axes]
    P = DI.poset_from_axes(axes)
    println("\n[probe] H0 active-chain incremental union-find path")
    old_inc = DI._H0_ACTIVE_CHAIN_INCREMENTAL[]
    old_min_pos = DI._H0_UNIONFIND_MIN_POS_VERTICES[]
    old_min_v = DI._H0_UNIONFIND_MIN_TOTAL_ACTIVE_VERTICES[]
    old_min_e = DI._H0_UNIONFIND_MIN_TOTAL_ACTIVE_EDGES[]
    inc_on = inc_off = nothing
    try
        DI._H0_UNIONFIND_MIN_POS_VERTICES[] = 0
        DI._H0_UNIONFIND_MIN_TOTAL_ACTIVE_VERTICES[] = 0
        DI._H0_UNIONFIND_MIN_TOTAL_ACTIVE_EDGES[] = 0
        DI._H0_ACTIVE_CHAIN_INCREMENTAL[] = true
        inc_on = _bench("  _cohomology_module_from_lazy(t=0, incremental on)",
                        () -> begin
                            L = DI._lazy_cochain_complex_from_graded_complex(G, P, axes; field=CM.F2())
                            DI._cohomology_module_from_lazy(L, 0)
                        end;
                        reps=reps)
        DI._H0_ACTIVE_CHAIN_INCREMENTAL[] = false
        inc_off = _bench("  _cohomology_module_from_lazy(t=0, incremental off)",
                         () -> begin
                             L = DI._lazy_cochain_complex_from_graded_complex(G, P, axes; field=CM.F2())
                             DI._cohomology_module_from_lazy(L, 0)
                         end;
                         reps=reps)
    finally
        DI._H0_ACTIVE_CHAIN_INCREMENTAL[] = old_inc
        DI._H0_UNIONFIND_MIN_POS_VERTICES[] = old_min_pos
        DI._H0_UNIONFIND_MIN_TOTAL_ACTIVE_VERTICES[] = old_min_v
        DI._H0_UNIONFIND_MIN_TOTAL_ACTIVE_EDGES[] = old_min_e
    end
    println("  incremental-on/off time ratio=", round(inc_on.med_ms / max(inc_off.med_ms, 1e-12), digits=3), "x")
    return (
        (case="h0_active_chain_incremental_probe_rips_d1", family="point", stage="module", cache_mode="incremental_on",
         med_ms=inc_on.med_ms, p90_ms=inc_on.p90_ms, med_kib=inc_on.med_kib),
        (case="h0_active_chain_incremental_probe_rips_d1", family="point", stage="module", cache_mode="incremental_off",
         med_ms=inc_off.med_ms, p90_ms=inc_off.p90_ms, med_kib=inc_off.med_kib),
    )
end

function _run_t2_degree_local_probe(; reps::Int, quick::Bool)
    nverts = quick ? 36 : 96
    verts = collect(1:nverts)
    edges = Tuple{Int,Int}[]
    for i in 2:nverts
        push!(edges, (1, i))
    end
    for i in 2:(nverts - 1)
        push!(edges, (i, i + 1))
    end
    triangles = Tuple{Int,Int,Int}[]
    for i in 2:(nverts - 1)
        push!(triangles, (1, i, i + 1))
    end
    I1 = Int[]; J1 = Int[]; V1 = Int[]
    sizehint!(I1, 2length(edges)); sizehint!(J1, 2length(edges)); sizehint!(V1, 2length(edges))
    for (j, (a, b)) in enumerate(edges)
        push!(I1, a); push!(J1, j); push!(V1, 1)
        push!(I1, b); push!(J1, j); push!(V1, -1)
    end
    B1 = sparse(I1, J1, V1, nverts, length(edges))
    B2 = spzeros(Int, length(edges), length(triangles))
    cells = [collect(1:nverts), collect(1:length(edges)), collect(1:length(triangles))]
    boundaries = [B1, B2]
    grades = vcat([Float64[0.0] for _ in verts],
                  [Float64[0.35] for _ in edges],
                  [Float64[0.75] for _ in triangles])
    data = TamerOp.GradedComplex(cells, boundaries, grades)
    axes = (collect(range(0.0, stop=1.0, length=(quick ? 24 : 40))),)
    P = DI.poset_from_axes(axes)
    println("\n[probe] degree-local t=2 fast path (graded fan complex)")
    old_local = DI._COHOMOLOGY_DEGREE_LOCAL_FASTPATH[]
    fast = base = nothing
    try
        DI._COHOMOLOGY_DEGREE_LOCAL_FASTPATH[] = true
        fast = _bench("  _cohomology_module_from_lazy(t=2, local on)",
                      () -> begin
                          L = DI._lazy_cochain_complex_from_graded_complex(data, P, axes; field=CM.F2())
                          DI._cohomology_module_from_lazy(L, 2)
                      end;
                      reps=reps)
        DI._COHOMOLOGY_DEGREE_LOCAL_FASTPATH[] = false
        base = _bench("  _cohomology_module_from_lazy(t=2, local off)",
                      () -> begin
                          L = DI._lazy_cochain_complex_from_graded_complex(data, P, axes; field=CM.F2())
                          DI._cohomology_module_from_lazy(L, 2)
                      end;
                      reps=reps)
    finally
        DI._COHOMOLOGY_DEGREE_LOCAL_FASTPATH[] = old_local
    end
    println("  local-on/off time ratio=", round(fast.med_ms / max(base.med_ms, 1e-12), digits=3), "x")
    return (
        (case="t2_degree_local_probe_graded_fan", family="point", stage="module", cache_mode="local_on",
         med_ms=fast.med_ms, p90_ms=fast.p90_ms, med_kib=fast.med_kib),
        (case="t2_degree_local_probe_graded_fan", family="point", stage="module", cache_mode="local_off",
         med_ms=base.med_ms, p90_ms=base.p90_ms, med_kib=base.med_kib),
    )
end

function _run_monotone_rank_probe(; reps::Int, quick::Bool)
    n = quick ? 72 : 220
    cells0 = collect(1:n)
    cells1 = collect(1:n)
    I = Int[]
    J = Int[]
    V = Int[]
    sizehint!(I, 2n)
    sizehint!(J, 2n)
    sizehint!(V, 2n)
    @inbounds for e in 1:(n - 1)
        push!(I, e); push!(J, e); push!(V, 1)
        push!(I, e + 1); push!(J, e); push!(V, -1)
    end
    push!(I, 1); push!(J, n); push!(V, 1)
    push!(I, n); push!(J, n); push!(V, -1)
    B = sparse(I, J, V, n, n)
    grades = [i <= n ? [0.0] : [0.5] for i in 1:(n + n)]
    data = TamerOp.GradedComplex([cells0, cells1], [B], grades)
    axes = (collect(range(0.0, stop=1.0, length=(quick ? 64 : 220))),)
    spec = TamerOp.FiltrationSpec(
        kind=:graded,
        axes=axes,
        construction=TamerOp.ConstructionOptions(;
            output_stage=:encoding_result,
            budget=(max_simplices=4_000_000, max_edges=4_000_000, memory_budget_bytes=8_000_000_000),
        ),
    )
    println("\n[probe] monotone rank-update dims path (cohomology_dims, t=1)")
    old_flag = DI._COHOMOLOGY_DIMS_MONOTONE_RANK_FASTPATH[]
    fast = base = nothing
    try
        DI._COHOMOLOGY_DIMS_MONOTONE_RANK_FASTPATH[] = true
        fast = _bench("  encode(:cohomology_dims,t=1) monotone-rank on",
                      () -> TamerOp.encode(data, spec; degree=1, cache=:auto, stage=:cohomology_dims);
                      reps=reps)
        DI._COHOMOLOGY_DIMS_MONOTONE_RANK_FASTPATH[] = false
        base = _bench("  encode(:cohomology_dims,t=1) monotone-rank off",
                      () -> TamerOp.encode(data, spec; degree=1, cache=:auto, stage=:cohomology_dims);
                      reps=reps)
    finally
        DI._COHOMOLOGY_DIMS_MONOTONE_RANK_FASTPATH[] = old_flag
    end
    println("  monotone-on/off time ratio=", round(fast.med_ms / max(base.med_ms, 1e-12), digits=3), "x")
    return (
        (case="monotone_rank_probe_graded_t1", family="point", stage="cohomology_dims", cache_mode="monotone_on",
         med_ms=fast.med_ms, p90_ms=fast.p90_ms, med_kib=fast.med_kib),
        (case="monotone_rank_probe_graded_t1", family="point", stage="cohomology_dims", cache_mode="monotone_off",
         med_ms=base.med_ms, p90_ms=base.p90_ms, med_kib=base.med_kib),
    )
end

function _run_monotone_incremental_probe(; reps::Int, quick::Bool)
    n = quick ? 96 : 220
    cells0 = collect(1:n)
    cells1 = collect(1:n)
    I = Int[]
    J = Int[]
    V = Int[]
    sizehint!(I, 2n)
    sizehint!(J, 2n)
    sizehint!(V, 2n)
    @inbounds for e in 1:(n - 1)
        push!(I, e); push!(J, e); push!(V, 1)
        push!(I, e + 1); push!(J, e); push!(V, -1)
    end
    B = sparse(I, J, V, n, n)
    grades = [i <= n ? [0.0] : [0.5] for i in 1:(n + n)]
    data = TamerOp.GradedComplex([cells0, cells1], [B], grades)
    axes = (collect(range(0.0, stop=1.0, length=(quick ? 84 : 240))),)
    spec = TamerOp.FiltrationSpec(
        kind=:graded,
        axes=axes,
        construction=TamerOp.ConstructionOptions(;
            output_stage=:encoding_result,
            budget=(max_simplices=4_000_000, max_edges=4_000_000, memory_budget_bytes=8_000_000_000),
        ),
    )
    println("\n[probe] monotone incremental-rank engine (cohomology_dims, t=1)")
    old_fast = DI._COHOMOLOGY_DIMS_MONOTONE_RANK_FASTPATH[]
    old_inc = DI._COHOMOLOGY_DIMS_MONOTONE_INCREMENTAL_RANK[]
    fast = base = nothing
    try
        DI._COHOMOLOGY_DIMS_MONOTONE_RANK_FASTPATH[] = true
        DI._COHOMOLOGY_DIMS_MONOTONE_INCREMENTAL_RANK[] = true
        fast = _bench("  encode(:cohomology_dims,t=1) monotone-incremental on",
                      () -> TamerOp.encode(data, spec; degree=1, cache=:auto, stage=:cohomology_dims);
                      reps=reps)
        DI._COHOMOLOGY_DIMS_MONOTONE_INCREMENTAL_RANK[] = false
        base = _bench("  encode(:cohomology_dims,t=1) monotone-incremental off",
                      () -> TamerOp.encode(data, spec; degree=1, cache=:auto, stage=:cohomology_dims);
                      reps=reps)
    finally
        DI._COHOMOLOGY_DIMS_MONOTONE_RANK_FASTPATH[] = old_fast
        DI._COHOMOLOGY_DIMS_MONOTONE_INCREMENTAL_RANK[] = old_inc
    end
    println("  monotone-incremental on/off time ratio=", round(fast.med_ms / max(base.med_ms, 1e-12), digits=3), "x")
    return (
        (case="monotone_incremental_probe_graded_t1", family="graded", stage="cohomology_dims", cache_mode="incremental_on",
         med_ms=fast.med_ms, p90_ms=fast.p90_ms, med_kib=fast.med_kib),
        (case="monotone_incremental_probe_graded_t1", family="graded", stage="cohomology_dims", cache_mode="incremental_off",
         med_ms=base.med_ms, p90_ms=base.p90_ms, med_kib=base.med_kib),
    )
end

function _run_structural_kernel_probe(; reps::Int, quick::Bool)
    K = CM.coeff_type(CM.F2())
    nrows = quick ? 256 : 768
    ncols = quick ? 220 : 640
    row_for_col = Vector{Int}(undef, ncols)
    @inbounds for j in 1:ncols
        row_for_col[j] = j % 7 == 0 ? 0 : 1 + ((3 * j + 11) % nrows)
    end
    A = DI._StructuralInclusionMap{K}(nrows, ncols, row_for_col)
    rows = collect(1:min(nrows, quick ? 160 : 420))
    cols = collect(1:min(ncols, quick ? 180 : 520))
    field = CM.F2()
    println("\n[probe] structural-map kernels (rank_restricted + nullspace + colspace)")
    old_struct = DI._STRUCTURAL_MAP_FAST_KERNELS[]
    fast = base = nothing
    try
        DI._STRUCTURAL_MAP_FAST_KERNELS[] = true
        fast = _bench("  structural kernels fast on",
                      () -> begin
                          r = FL.rank_restricted(field, A, rows, cols)
                          Z = FL.nullspace(field, A)
                          C = FL.colspace(field, A)
                          r + size(Z, 2) + size(C, 2)
                      end;
                      reps=reps)
        DI._STRUCTURAL_MAP_FAST_KERNELS[] = false
        base = _bench("  structural kernels fast off",
                      () -> begin
                          r = FL.rank_restricted(field, A, rows, cols)
                          Z = FL.nullspace(field, A)
                          C = FL.colspace(field, A)
                          r + size(Z, 2) + size(C, 2)
                      end;
                      reps=reps)
    finally
        DI._STRUCTURAL_MAP_FAST_KERNELS[] = old_struct
    end
    println("  structural kernels on/off time ratio=", round(fast.med_ms / max(base.med_ms, 1e-12), digits=3), "x")
    return (
        (case="structural_kernel_probe", family="graded", stage="module", cache_mode="struct_fast_on",
         med_ms=fast.med_ms, p90_ms=fast.p90_ms, med_kib=fast.med_kib),
        (case="structural_kernel_probe", family="graded", stage="module", cache_mode="struct_fast_off",
         med_ms=base.med_ms, p90_ms=base.p90_ms, med_kib=base.med_kib),
    )
end

function _run_active_list_chain_probe(; reps::Int, quick::Bool)
    nverts = quick ? 900 : 3_600
    ncells = quick ? 14_000 : 56_000
    births = Vector{NTuple{1,Int}}(undef, ncells)
    @inbounds for i in 1:ncells
        births[i] = (1 + ((37 * i + 17) % nverts),)
    end
    vertex_idxs = [(i,) for i in 1:nverts]
    orientation = (1,)
    println("\n[probe] active-list 1d chain fast path")
    old_active = DI._ACTIVE_LISTS_CHAIN_FASTPATH[]
    fast = base = nothing
    try
        DI._ACTIVE_LISTS_CHAIN_FASTPATH[] = true
        fast = _bench("  _active_lists chain fast on",
                      () -> DI._active_lists(births, vertex_idxs, orientation; multicritical=:union);
                      reps=reps)
        DI._ACTIVE_LISTS_CHAIN_FASTPATH[] = false
        base = _bench("  _active_lists chain fast off",
                      () -> DI._active_lists(births, vertex_idxs, orientation; multicritical=:union);
                      reps=reps)
    finally
        DI._ACTIVE_LISTS_CHAIN_FASTPATH[] = old_active
    end
    println("  active-list fast on/off time ratio=", round(fast.med_ms / max(base.med_ms, 1e-12), digits=3), "x")
    return (
        (case="active_list_chain_probe", family="graded", stage="graded_complex", cache_mode="active_fast_on",
         med_ms=fast.med_ms, p90_ms=fast.p90_ms, med_kib=fast.med_kib),
        (case="active_list_chain_probe", family="graded", stage="graded_complex", cache_mode="active_fast_off",
         med_ms=base.med_ms, p90_ms=base.p90_ms, med_kib=base.med_kib),
    )
end

function _run_degree_local_all_t_probe(; reps::Int, quick::Bool)
    nverts = quick ? 24 : 96
    verts = collect(1:nverts)
    edges = Tuple{Int,Int}[]
    for i in 2:nverts
        push!(edges, (1, i))
    end
    for i in 2:(nverts - 1)
        push!(edges, (i, i + 1))
    end
    triangles = Tuple{Int,Int,Int}[]
    for i in 2:(nverts - 1)
        push!(triangles, (1, i, i + 1))
    end
    I1 = Int[]; J1 = Int[]; V1 = Int[]
    sizehint!(I1, 2length(edges)); sizehint!(J1, 2length(edges)); sizehint!(V1, 2length(edges))
    for (j, (a, b)) in enumerate(edges)
        push!(I1, a); push!(J1, j); push!(V1, 1)
        push!(I1, b); push!(J1, j); push!(V1, -1)
    end
    B1 = sparse(I1, J1, V1, nverts, length(edges))
    B2 = spzeros(Int, length(edges), length(triangles))
    cells = [collect(1:nverts), collect(1:length(edges)), collect(1:length(triangles))]
    boundaries = [B1, B2]
    grades = vcat([Float64[0.0] for _ in verts],
                  [Float64[0.35] for _ in edges],
                  [Float64[0.75] for _ in triangles])
    data = TamerOp.GradedComplex(cells, boundaries, grades)
    axes = (collect(range(0.0, stop=1.0, length=(quick ? 16 : 40))),)
    spec = TamerOp.FiltrationSpec(
        kind=:graded,
        axes=axes,
        construction=TamerOp.ConstructionOptions(;
            output_stage=:encoding_result,
            budget=(max_simplices=4_000_000, max_edges=4_000_000, memory_budget_bytes=8_000_000_000),
        ),
    )
    pipeline = TamerOp.PipelineOptions(field=CM.F2())
    println("\n[probe] degree-local all-t path (module, t=1)")
    old_local = DI._COHOMOLOGY_DEGREE_LOCAL_FASTPATH[]
    old_all_t = DI._COHOMOLOGY_DEGREE_LOCAL_ALL_T[]
    old_h1 = DI._H1_COKERNEL_FASTPATH[]
    old_min_pos = DI._COHOMOLOGY_DEGREE_LOCAL_T1_MIN_POS_VERTICES[]
    old_min_d1 = DI._COHOMOLOGY_DEGREE_LOCAL_T1_MIN_TOTAL_ACTIVE_DIM1[]
    old_min_d2 = DI._COHOMOLOGY_DEGREE_LOCAL_T1_MIN_TOTAL_ACTIVE_DIM2[]
    old_lazy = DI._ENCODING_RESULT_LAZY_MODULE[]
    fast = base = nothing
    try
        DI._ENCODING_RESULT_LAZY_MODULE[] = false
        DI._H1_COKERNEL_FASTPATH[] = false
        DI._COHOMOLOGY_DEGREE_LOCAL_FASTPATH[] = true
        DI._COHOMOLOGY_DEGREE_LOCAL_T1_MIN_POS_VERTICES[] = typemax(Int)
        DI._COHOMOLOGY_DEGREE_LOCAL_T1_MIN_TOTAL_ACTIVE_DIM1[] = typemax(Int)
        DI._COHOMOLOGY_DEGREE_LOCAL_T1_MIN_TOTAL_ACTIVE_DIM2[] = typemax(Int)
        DI._COHOMOLOGY_DEGREE_LOCAL_ALL_T[] = true
        fast = _bench("  encode(:module,t=1) local-all-t on",
                      () -> TamerOp.encode(data, spec; degree=1, cache=:auto, stage=:module, pipeline=pipeline);
                      reps=reps)
        DI._COHOMOLOGY_DEGREE_LOCAL_ALL_T[] = false
        base = _bench("  encode(:module,t=1) local-all-t off",
                      () -> TamerOp.encode(data, spec; degree=1, cache=:auto, stage=:module, pipeline=pipeline);
                      reps=reps)
    finally
        DI._ENCODING_RESULT_LAZY_MODULE[] = old_lazy
        DI._H1_COKERNEL_FASTPATH[] = old_h1
        DI._COHOMOLOGY_DEGREE_LOCAL_FASTPATH[] = old_local
        DI._COHOMOLOGY_DEGREE_LOCAL_ALL_T[] = old_all_t
        DI._COHOMOLOGY_DEGREE_LOCAL_T1_MIN_POS_VERTICES[] = old_min_pos
        DI._COHOMOLOGY_DEGREE_LOCAL_T1_MIN_TOTAL_ACTIVE_DIM1[] = old_min_d1
        DI._COHOMOLOGY_DEGREE_LOCAL_T1_MIN_TOTAL_ACTIVE_DIM2[] = old_min_d2
    end
    println("  local-all-t on/off time ratio=", round(fast.med_ms / max(base.med_ms, 1e-12), digits=3), "x")
    return (
        (case="degree_local_all_t_probe_graded_t1", family="graded", stage="module", cache_mode="all_t_on",
         med_ms=fast.med_ms, p90_ms=fast.p90_ms, med_kib=fast.med_kib),
        (case="degree_local_all_t_probe_graded_t1", family="graded", stage="module", cache_mode="all_t_off",
         med_ms=base.med_ms, p90_ms=base.p90_ms, med_kib=base.med_kib),
    )
end

function _run_encoding_result_lazy_probe(; reps::Int, quick::Bool)
    n = quick ? 22 : 60
    cells0 = collect(1:n)
    cells1 = collect(1:(n - 1))
    I = Int[]
    J = Int[]
    V = Int[]
    sizehint!(I, 2 * (n - 1))
    sizehint!(J, 2 * (n - 1))
    sizehint!(V, 2 * (n - 1))
    @inbounds for e in 1:(n - 1)
        push!(I, e); push!(J, e); push!(V, 1)
        push!(I, e + 1); push!(J, e); push!(V, -1)
    end
    B = sparse(I, J, V, n, n - 1)
    grades = vcat([Float64[0.0] for _ in 1:n], [Float64[0.45] for _ in 1:(n - 1)])
    data = TamerOp.GradedComplex([cells0, cells1], [B], grades)
    axes = (collect(range(0.0, stop=1.0, length=(quick ? 14 : 32))),)
    spec = TamerOp.FiltrationSpec(
        kind=:graded,
        max_dim=1,
        axes=axes,
        construction=TamerOp.ConstructionOptions(;
            output_stage=:encoding_result,
            budget=(max_simplices=6_000_000, max_edges=6_000_000, memory_budget_bytes=8_000_000_000),
        ),
    )
    pipeline = TamerOp.PipelineOptions(field=CM.F2())

    println("\n[probe] encoding_result lazy module path")
    old_lazy = DI._ENCODING_RESULT_LAZY_MODULE[]
    on_encode = off_encode = nothing
    on_force = off_force = nothing
    try
        DI._ENCODING_RESULT_LAZY_MODULE[] = true
        on_encode = _bench("  encode(:encoding_result,t=1) lazy-module on",
                           () -> TamerOp.encode(data, spec; degree=1, cache=:auto, stage=:encoding_result, pipeline=pipeline);
                           reps=reps)
        on_force = _bench("  encode(:encoding_result,t=1)+pmodule lazy-module on",
                          () -> begin
                              enc = TamerOp.encode(data, spec; degree=1, cache=:auto, stage=:encoding_result, pipeline=pipeline)
                              TamerOp.Workflow.pmodule(enc)
                          end;
                          reps=reps)

        DI._ENCODING_RESULT_LAZY_MODULE[] = false
        off_encode = _bench("  encode(:encoding_result,t=1) lazy-module off",
                            () -> TamerOp.encode(data, spec; degree=1, cache=:auto, stage=:encoding_result, pipeline=pipeline);
                            reps=reps)
        off_force = _bench("  encode(:encoding_result,t=1)+pmodule lazy-module off",
                           () -> begin
                               enc = TamerOp.encode(data, spec; degree=1, cache=:auto, stage=:encoding_result, pipeline=pipeline)
                               TamerOp.Workflow.pmodule(enc)
                           end;
                           reps=reps)
    finally
        DI._ENCODING_RESULT_LAZY_MODULE[] = old_lazy
    end

    println("  lazy encode on/off time ratio=", round(on_encode.med_ms / max(off_encode.med_ms, 1e-12), digits=3), "x")
    println("  lazy+pmodule on/off time ratio=", round(on_force.med_ms / max(off_force.med_ms, 1e-12), digits=3), "x")
    return (
        (case="encoding_result_lazy_probe_graded_t1", family="graded", stage="encoding_result", cache_mode="lazy_on_encode",
         med_ms=on_encode.med_ms, p90_ms=on_encode.p90_ms, med_kib=on_encode.med_kib),
        (case="encoding_result_lazy_probe_graded_t1", family="graded", stage="encoding_result", cache_mode="lazy_off_encode",
         med_ms=off_encode.med_ms, p90_ms=off_encode.p90_ms, med_kib=off_encode.med_kib),
        (case="encoding_result_lazy_probe_graded_t1", family="graded", stage="encoding_result", cache_mode="lazy_on_plus_pmodule",
         med_ms=on_force.med_ms, p90_ms=on_force.p90_ms, med_kib=on_force.med_kib),
        (case="encoding_result_lazy_probe_graded_t1", family="graded", stage="encoding_result", cache_mode="lazy_off_plus_pmodule",
         med_ms=off_force.med_ms, p90_ms=off_force.p90_ms, med_kib=off_force.med_kib),
    )
end

function _run_structural_map_probe(; reps::Int, quick::Bool)
    n = quick ? 84 : 220
    cells0 = collect(1:n)
    cells1 = collect(1:n)
    I = Int[]
    J = Int[]
    V = Int[]
    sizehint!(I, 2n)
    sizehint!(J, 2n)
    sizehint!(V, 2n)
    @inbounds for e in 1:(n - 1)
        push!(I, e); push!(J, e); push!(V, 1)
        push!(I, e + 1); push!(J, e); push!(V, -1)
    end
    push!(I, 1); push!(J, n); push!(V, 1)
    push!(I, n); push!(J, n); push!(V, -1)
    B = sparse(I, J, V, n, n)
    grades = [i <= n ? [0.0] : [0.5] for i in 1:(n + n)]
    data = TamerOp.GradedComplex([cells0, cells1], [B], grades)
    axes = (collect(range(0.0, stop=1.0, length=(quick ? 72 : 240))),)
    spec = TamerOp.FiltrationSpec(
        kind=:graded,
        axes=axes,
        construction=TamerOp.ConstructionOptions(;
            output_stage=:encoding_result,
            budget=(max_simplices=4_000_000, max_edges=4_000_000, memory_budget_bytes=8_000_000_000),
        ),
    )
    println("\n[probe] structural-map restricted-rank kernel (cohomology_dims, t=1)")
    old_mon = DI._COHOMOLOGY_DIMS_MONOTONE_RANK_FASTPATH[]
    old_direct = DI._COHOMOLOGY_DIMS_USE_DIRECT_RESTRICTED_RANK[]
    fast = base = nothing
    try
        DI._COHOMOLOGY_DIMS_MONOTONE_RANK_FASTPATH[] = true
        DI._COHOMOLOGY_DIMS_USE_DIRECT_RESTRICTED_RANK[] = true
        fast = _bench("  encode(:cohomology_dims,t=1) structural-direct on",
                      () -> TamerOp.encode(data, spec; degree=1, cache=:auto, stage=:cohomology_dims);
                      reps=reps)
        DI._COHOMOLOGY_DIMS_USE_DIRECT_RESTRICTED_RANK[] = false
        base = _bench("  encode(:cohomology_dims,t=1) structural-direct off",
                      () -> TamerOp.encode(data, spec; degree=1, cache=:auto, stage=:cohomology_dims);
                      reps=reps)
    finally
        DI._COHOMOLOGY_DIMS_MONOTONE_RANK_FASTPATH[] = old_mon
        DI._COHOMOLOGY_DIMS_USE_DIRECT_RESTRICTED_RANK[] = old_direct
    end
    println("  structural-direct on/off time ratio=", round(fast.med_ms / max(base.med_ms, 1e-12), digits=3), "x")
    return (
        (case="structural_map_probe_graded_t1", family="point", stage="cohomology_dims", cache_mode="structural_on",
         med_ms=fast.med_ms, p90_ms=fast.p90_ms, med_kib=fast.med_kib),
        (case="structural_map_probe_graded_t1", family="point", stage="cohomology_dims", cache_mode="structural_off",
         med_ms=base.med_ms, p90_ms=base.p90_ms, med_kib=base.med_kib),
    )
end

function _run_packed_edgelist_probe(; reps::Int, quick::Bool)
    n = quick ? 160 : 360
    raw_edges = Tuple{Int,Int}[]
    @inbounds for i in 1:n
        j1 = (i % n) + 1
        j2 = ((i + 1) % n) + 1
        push!(raw_edges, (min(i, j1), max(i, j1)))
        push!(raw_edges, (min(i, j2), max(i, j2)))
        if (i % 7) == 0
            j3 = ((i + 19) % n) + 1
            push!(raw_edges, (min(i, j3), max(i, j3)))
        end
    end
    uniq = Dict{Tuple{Int,Int},Bool}()
    for e in raw_edges
        uniq[e] = true
    end
    edges = collect(keys(uniq))
    sort!(edges)
    vals = [sin(0.07 * i) + 0.01 * (i % 5) for i in 1:n]
    data = TamerOp.GraphData(n, edges)
    gate_on = DI._use_packed_edge_list_backend(n, length(edges), 3)
    spec = TamerOp.FiltrationSpec(
        kind=:clique_lower_star,
        max_dim=3,
        vertex_values=vals,
        construction=TamerOp.ConstructionOptions(;
            output_stage=:graded_complex,
            budget=(max_simplices=8_000_000, max_edges=8_000_000, memory_budget_bytes=8_000_000_000),
        ),
    )

    println("\n[probe] packed edge-list backend (graph clique lower-star)")
    println("  fixture gate decision: ", gate_on, " (n=", n, ", m=", length(edges), ")")
    old_flag = DI._GRAPH_PACKED_EDGELIST_BACKEND[]
    old_cache = DI._GRAPH_BACKEND_WINNER_CACHE_ENABLED[]
    old_probe = DI._GRAPH_BACKEND_WINNER_CACHE_PROBE[]
    fast = base = nothing
    auto_warm = nothing
    try
        # Forced on/off comparison: disable bucket winner cache to isolate backend cost.
        DI._GRAPH_BACKEND_WINNER_CACHE_ENABLED[] = false
        DI._GRAPH_BACKEND_WINNER_CACHE_PROBE[] = false
        DI._clear_graph_backend_winner_cache!()
        DI._GRAPH_PACKED_EDGELIST_BACKEND[] = true
        fast = _bench("  encode(:graded_complex) packed-edgelist on",
                      () -> TamerOp.encode(data, spec; degree=0, cache=:auto, stage=:graded_complex);
                      reps=reps)
        DI._GRAPH_PACKED_EDGELIST_BACKEND[] = false
        base = _bench("  encode(:graded_complex) packed-edgelist off",
                      () -> TamerOp.encode(data, spec; degree=0, cache=:auto, stage=:graded_complex);
                      reps=reps)

        # Auto mode with winner-cache enabled (cold fill + warm measure).
        DI._GRAPH_PACKED_EDGELIST_BACKEND[] = true
        DI._GRAPH_BACKEND_WINNER_CACHE_ENABLED[] = true
        DI._GRAPH_BACKEND_WINNER_CACHE_PROBE[] = true
        DI._clear_graph_backend_winner_cache!()
        TamerOp.encode(data, spec; degree=0, cache=:auto, stage=:graded_complex) # cold cache fill
        auto_warm = _bench("  encode(:graded_complex) packed-auto warm",
                           () -> TamerOp.encode(data, spec; degree=0, cache=:auto, stage=:graded_complex);
                           reps=reps)
    finally
        DI._GRAPH_PACKED_EDGELIST_BACKEND[] = old_flag
        DI._GRAPH_BACKEND_WINNER_CACHE_ENABLED[] = old_cache
        DI._GRAPH_BACKEND_WINNER_CACHE_PROBE[] = old_probe
    end
    println("  packed-edgelist on/off time ratio=", round(fast.med_ms / max(base.med_ms, 1e-12), digits=3), "x")
    println("  packed-auto/off time ratio=", round(auto_warm.med_ms / max(base.med_ms, 1e-12), digits=3), "x")
    return (
        (case="packed_edgelist_probe_graph_clique", family="graph", stage="graded_complex", cache_mode="packed_on",
         med_ms=fast.med_ms, p90_ms=fast.p90_ms, med_kib=fast.med_kib),
        (case="packed_edgelist_probe_graph_clique", family="graph", stage="graded_complex", cache_mode="packed_off",
         med_ms=base.med_ms, p90_ms=base.p90_ms, med_kib=base.med_kib),
        (case="packed_edgelist_probe_graph_clique", family="graph", stage="graded_complex", cache_mode="packed_auto_warm",
         med_ms=auto_warm.med_ms, p90_ms=auto_warm.p90_ms, med_kib=auto_warm.med_kib),
    )
end

function _run_dims_only_probe(; reps::Int, quick::Bool)
    n = quick ? 110 : 260
    data = _point_cloud_fixture(n; seed=Int(0xD4AF))
    spec = TamerOp.FiltrationSpec(
        kind=:rips,
        max_dim=1,
        axes=(collect(range(0.0, stop=2.4, length=(quick ? 28 : 40))),),
        knn=min(16, max(2, n - 1)),
        nn_backend=:auto,
        construction=TamerOp.ConstructionOptions(;
            sparsify=:knn,
            output_stage=:encoding_result,
            budget=(max_simplices=5_000_000, max_edges=5_000_000, memory_budget_bytes=8_000_000_000),
        ),
    )
    println("\n[probe] dims-only invariant shortcut (encode full vs cohomology_dims)")
    full = _bench("  encode(:encoding_result) + restricted_hilbert",
                  () -> begin
                      enc = TamerOp.encode(data, spec; degree=0, cache=:auto, stage=:encoding_result)
                      TamerOp.Invariants.restricted_hilbert(enc.M)
                  end;
                  reps=reps)
    dims_only = _bench("  encode(:cohomology_dims) + restricted_hilbert(dims)",
                       () -> begin
                           d = TamerOp.encode(data, spec; degree=0, cache=:auto, stage=:cohomology_dims)
                           TamerOp.Invariants.restricted_hilbert(d.dims)
                       end;
                       reps=reps)
    println("  dims-only/full time ratio=", round(dims_only.med_ms / max(full.med_ms, 1e-12), digits=3), "x")
    return (
        (case="dims_only_probe_rips_d1", family="point", stage="encoding_result", cache_mode="full_module",
         med_ms=full.med_ms, p90_ms=full.p90_ms, med_kib=full.med_kib),
        (case="dims_only_probe_rips_d1", family="point", stage="cohomology_dims", cache_mode="dims_only",
         med_ms=dims_only.med_ms, p90_ms=dims_only.p90_ms, med_kib=dims_only.med_kib),
    )
end

function _run_solve_loop_probe(; reps::Int, quick::Bool)
    n = quick ? 30 : 80
    cells0 = collect(1:n)
    cells1 = collect(1:n)
    I = Int[]
    J = Int[]
    V = Int[]
    sizehint!(I, 2n)
    sizehint!(J, 2n)
    sizehint!(V, 2n)
    @inbounds for e in 1:(n - 1)
        push!(I, e); push!(J, e); push!(V, 1)
        push!(I, e + 1); push!(J, e); push!(V, -1)
    end
    push!(I, 1); push!(J, n); push!(V, 1)
    push!(I, n); push!(J, n); push!(V, -1)
    B = sparse(I, J, V, n, n)
    grades = [i <= n ? [0.0] : [0.5] for i in 1:(n + n)]
    data = TamerOp.GradedComplex([cells0, cells1], [B], grades)
    axes = (collect(range(0.0, stop=1.0, length=(quick ? 20 : 30))),)
    spec = TamerOp.FiltrationSpec(
        kind=:graded,
        axes=axes,
        construction=TamerOp.ConstructionOptions(;
            output_stage=:encoding_result,
            budget=(max_simplices=4_000_000, max_edges=4_000_000, memory_budget_bytes=8_000_000_000),
        ),
    )
    pipeline = TamerOp.PipelineOptions(field=CM.F2())
    println("\n[probe] solve-loop checks in image/cokernel (generic H1 path)")
    old_h1 = DI._H1_COKERNEL_FASTPATH[]
    old_solve = AC._FAST_SOLVE_NO_CHECK[]
    fast = base = nothing
    try
        DI._H1_COKERNEL_FASTPATH[] = false
        AC._FAST_SOLVE_NO_CHECK[] = true
        fast = _bench("  generic H1 with no RHS checks",
                      () -> TamerOp.encode(data, spec; degree=1, cache=:auto, stage=:encoding_result, pipeline=pipeline);
                      reps=reps)
        AC._FAST_SOLVE_NO_CHECK[] = false
        base = _bench("  generic H1 with RHS checks",
                      () -> TamerOp.encode(data, spec; degree=1, cache=:auto, stage=:encoding_result, pipeline=pipeline);
                      reps=reps)
    finally
        DI._H1_COKERNEL_FASTPATH[] = old_h1
        AC._FAST_SOLVE_NO_CHECK[] = old_solve
    end
    println("  no-check/check time ratio=", round(fast.med_ms / max(base.med_ms, 1e-12), digits=3), "x")
    return (
        (case="solve_loop_probe_h1_generic", family="point", stage="encoding_result", cache_mode="no_rhs_check",
         med_ms=fast.med_ms, p90_ms=fast.p90_ms, med_kib=fast.med_kib),
        (case="solve_loop_probe_h1_generic", family="point", stage="encoding_result", cache_mode="with_rhs_check",
         med_ms=base.med_ms, p90_ms=base.p90_ms, med_kib=base.med_kib),
    )
end

function _run_cache_hit_skip_lazy_probe(; reps::Int, quick::Bool)
    n = quick ? 90 : 220
    data = _point_cloud_fixture(n; seed=Int(0xD4A8))
    spec = TamerOp.FiltrationSpec(
        kind=:rips,
        max_dim=1,
        axes=(collect(range(0.0, stop=2.0, length=28)),),
        knn=min(16, max(2, n - 1)),
        nn_backend=:auto,
        construction=TamerOp.ConstructionOptions(;
            sparsify=:knn,
            output_stage=:encoding_result,
            budget=(max_simplices=4_000_000, max_edges=4_000_000, memory_budget_bytes=8_000_000_000),
        ),
    )
    println("\n[probe] cache-hit lazy-build skip in full encode path")
    old_skip = DI._INGESTION_SKIP_LAZY_ON_MODULE_CACHE_HIT[]
    old_chain = DI._H0_CHAIN_SWEEP_FASTPATH[]
    old_min_pos = DI._H0_UNIONFIND_MIN_POS_VERTICES[]
    old_min_v = DI._H0_UNIONFIND_MIN_TOTAL_ACTIVE_VERTICES[]
    old_min_e = DI._H0_UNIONFIND_MIN_TOTAL_ACTIVE_EDGES[]
    skip_on = skip_off = nothing
    try
        DI._H0_CHAIN_SWEEP_FASTPATH[] = true
        DI._H0_UNIONFIND_MIN_POS_VERTICES[] = 0
        DI._H0_UNIONFIND_MIN_TOTAL_ACTIVE_VERTICES[] = 0
        DI._H0_UNIONFIND_MIN_TOTAL_ACTIVE_EDGES[] = 0

        DI._INGESTION_SKIP_LAZY_ON_MODULE_CACHE_HIT[] = true
        sc_on = CM.SessionCache()
        TamerOp.encode(data, spec; degree=0, cache=sc_on, stage=:encoding_result)
        skip_on = _bench("  cache hit with lazy skip",
                         () -> TamerOp.encode(data, spec; degree=0, cache=sc_on, stage=:encoding_result);
                         reps=reps)

        DI._INGESTION_SKIP_LAZY_ON_MODULE_CACHE_HIT[] = false
        sc_off = CM.SessionCache()
        TamerOp.encode(data, spec; degree=0, cache=sc_off, stage=:encoding_result)
        skip_off = _bench("  cache hit without lazy skip",
                          () -> TamerOp.encode(data, spec; degree=0, cache=sc_off, stage=:encoding_result);
                          reps=reps)
    finally
        DI._INGESTION_SKIP_LAZY_ON_MODULE_CACHE_HIT[] = old_skip
        DI._H0_CHAIN_SWEEP_FASTPATH[] = old_chain
        DI._H0_UNIONFIND_MIN_POS_VERTICES[] = old_min_pos
        DI._H0_UNIONFIND_MIN_TOTAL_ACTIVE_VERTICES[] = old_min_v
        DI._H0_UNIONFIND_MIN_TOTAL_ACTIVE_EDGES[] = old_min_e
    end
    println("  skip-on/skip-off time ratio=", round(skip_on.med_ms / max(skip_off.med_ms, 1e-12), digits=3), "x")
    return (
        (case="cache_hit_skip_lazy_probe_rips_d1", family="point", stage="full", cache_mode="skip_on",
         med_ms=skip_on.med_ms, p90_ms=skip_on.p90_ms, med_kib=skip_on.med_kib),
        (case="cache_hit_skip_lazy_probe_rips_d1", family="point", stage="full", cache_mode="skip_off",
         med_ms=skip_off.med_ms, p90_ms=skip_off.p90_ms, med_kib=skip_off.med_kib),
    )
end

function _run_h1_fastpath_probe(; reps::Int, quick::Bool)
    n = quick ? 30 : 80
    cells0 = collect(1:n)
    cells1 = collect(1:n) # path edges + one closing edge
    I = Int[]
    J = Int[]
    V = Int[]
    sizehint!(I, 2n)
    sizehint!(J, 2n)
    sizehint!(V, 2n)
    @inbounds for e in 1:(n - 1)
        push!(I, e); push!(J, e); push!(V, 1)
        push!(I, e + 1); push!(J, e); push!(V, -1)
    end
    push!(I, 1); push!(J, n); push!(V, 1)
    push!(I, n); push!(J, n); push!(V, -1)
    B = sparse(I, J, V, n, n)
    grades = [i <= n ? [0.0] : [0.5] for i in 1:(n + n)]
    data = TamerOp.GradedComplex([cells0, cells1], [B], grades)
    axes = (collect(range(0.0, stop=1.0, length=(quick ? 20 : 30))),)
    spec = TamerOp.FiltrationSpec(
        kind=:graded,
        axes=axes,
        construction=TamerOp.ConstructionOptions(;
            output_stage=:encoding_result,
            budget=(max_simplices=4_000_000, max_edges=4_000_000, memory_budget_bytes=8_000_000_000),
        ),
    )
    pipeline = TamerOp.PipelineOptions(field=CM.F2())
    println("\n[probe] H1 cokernel fast path (degree=1, graded input, n=$(n))")
    old_h1 = DI._H1_COKERNEL_FASTPATH[]
    fast = base = nothing
    try
        DI._H1_COKERNEL_FASTPATH[] = true
        fast = _bench("  encode(... degree=1, h1 fastpath on)",
                      () -> TamerOp.encode(data, spec; degree=1, cache=:auto, stage=:encoding_result, pipeline=pipeline);
                      reps=reps)
        DI._H1_COKERNEL_FASTPATH[] = false
        base = _bench("  encode(... degree=1, h1 fastpath off)",
                      () -> TamerOp.encode(data, spec; degree=1, cache=:auto, stage=:encoding_result, pipeline=pipeline);
                      reps=reps)
    finally
        DI._H1_COKERNEL_FASTPATH[] = old_h1
    end
    println("  fast-on/off time ratio=", round(fast.med_ms / max(base.med_ms, 1e-12), digits=3), "x")
    return (
        (case="h1_fastpath_probe_graded_d1", family="point", stage="encoding_result", cache_mode="fast_on",
         med_ms=fast.med_ms, p90_ms=fast.p90_ms, med_kib=fast.med_kib),
        (case="h1_fastpath_probe_graded_d1", family="point", stage="encoding_result", cache_mode="fast_off",
         med_ms=base.med_ms, p90_ms=base.p90_ms, med_kib=base.med_kib),
    )
end

function _run_plan_norm_cache_probe(; reps::Int, quick::Bool)
    n = quick ? 60 : 160
    nbatch = quick ? 12 : 36
    d = quick ? 3 : 5
    rng = MersenneTwister(Int(0xD4AA))
    batch = Vector{TamerOp.PointCloud}(undef, nbatch)
    for i in 1:nbatch
        pts = [randn(rng, d) .+ 0.02 * i for _ in 1:n]
        batch[i] = TamerOp.PointCloud(pts)
    end
    spec = TamerOp.FiltrationSpec(
        kind=:rips,
        max_dim=1,
        knn=min(10, max(2, n - 1)),
        nn_backend=:auto,
        construction=TamerOp.ConstructionOptions(;
            sparsify=:knn,
            output_stage=:simplex_tree,
            budget=(max_simplices=4_000_000, max_edges=4_000_000, memory_budget_bytes=8_000_000_000),
        ),
    )
    println("\n[probe] plan normalization cache reuse across batch  n=$(n), batch=$(nbatch), d=$(d)")
    old_norm = DI._INGESTION_PLAN_NORM_CACHE[]
    norm_on = norm_off = nothing
    try
        DI._INGESTION_PLAN_NORM_CACHE[] = true
        norm_on = _bench("  plan cache enabled",
                         () -> begin
                             sc = CM.SessionCache()
                             for x in batch
                                 p = DI.plan_ingestion(x, spec; cache=sc, stage=:simplex_tree)
                                 DI.run_ingestion(p; stage=:simplex_tree, degree=0)
                             end
                             nothing
                         end;
                         reps=reps)
        DI._INGESTION_PLAN_NORM_CACHE[] = false
        norm_off = _bench("  plan cache disabled",
                          () -> begin
                              sc = CM.SessionCache()
                              for x in batch
                                  p = DI.plan_ingestion(x, spec; cache=sc, stage=:simplex_tree)
                                  DI.run_ingestion(p; stage=:simplex_tree, degree=0)
                              end
                              nothing
                          end;
                          reps=reps)
    finally
        DI._INGESTION_PLAN_NORM_CACHE[] = old_norm
    end
    println("  plan-cache-on/off time ratio=", round(norm_on.med_ms / max(norm_off.med_ms, 1e-12), digits=3), "x")
    return (
        (case="plan_norm_cache_probe_batch_rips_d1", family="point", stage="simplex_tree", cache_mode="norm_on",
         med_ms=norm_on.med_ms, p90_ms=norm_on.p90_ms, med_kib=norm_on.med_kib),
        (case="plan_norm_cache_probe_batch_rips_d1", family="point", stage="simplex_tree", cache_mode="norm_off",
         med_ms=norm_off.med_ms, p90_ms=norm_off.p90_ms, med_kib=norm_off.med_kib),
    )
end

function _run_boundary_kernel_probe(; reps::Int, quick::Bool)
    nverts = quick ? 120 : 260
    nsimp = quick ? 1_200 : 4_800
    rng = MersenneTwister(Int(0xD4AB))
    simplices2 = Vector{Vector{Int}}(undef, nsimp)
    edge_keys = UInt64[]
    sizehint!(edge_keys, 3 * nsimp)
    for i in 1:nsimp
        a = rand(rng, 1:nverts)
        b = rand(rng, 1:nverts)
        c = rand(rng, 1:nverts)
        while b == a
            b = rand(rng, 1:nverts)
        end
        while c == a || c == b
            c = rand(rng, 1:nverts)
        end
        tri = sort!([a, b, c])
        simplices2[i] = tri
        push!(edge_keys, DI._pack_edge_key(tri[1], tri[2]))
        push!(edge_keys, DI._pack_edge_key(tri[1], tri[3]))
        push!(edge_keys, DI._pack_edge_key(tri[2], tri[3]))
    end
    sort!(edge_keys)
    unique!(edge_keys)
    faces1 = Vector{Vector{Int}}(undef, length(edge_keys))
    @inbounds for i in eachindex(edge_keys)
        a, b = DI._unpack_edge_key(edge_keys[i])
        faces1[i] = [a, b]
    end

    println("\n[probe] simplicial boundary kernel (K=3) specialized vs hash  nverts=$(nverts), nsimp=$(nsimp), nfaces=$(length(faces1))")
    old_specialized = DI._SIMPLICIAL_BOUNDARY_SPECIALIZED[]
    spec = hash = nothing
    try
        DI._SIMPLICIAL_BOUNDARY_SPECIALIZED[] = true
        spec = _bench("  _simplicial_boundary specialized",
                      () -> DI._simplicial_boundary(simplices2, faces1);
                      reps=reps)
        DI._SIMPLICIAL_BOUNDARY_SPECIALIZED[] = false
        hash = _bench("  _simplicial_boundary hash",
                      () -> DI._simplicial_boundary(simplices2, faces1);
                      reps=reps)
    finally
        DI._SIMPLICIAL_BOUNDARY_SPECIALIZED[] = old_specialized
    end
    println("  specialized/hash time ratio=", round(spec.med_ms / max(hash.med_ms, 1e-12), digits=3), "x")
    return (
        (case="boundary_kernel_probe_k3", family="point", stage="graded_complex", cache_mode="specialized_on",
         med_ms=spec.med_ms, p90_ms=spec.p90_ms, med_kib=spec.med_kib),
        (case="boundary_kernel_probe_k3", family="point", stage="graded_complex", cache_mode="specialized_off",
         med_ms=hash.med_ms, p90_ms=hash.p90_ms, med_kib=hash.med_kib),
    )
end

function main(; reps::Int=6,
              quick::Bool=false,
              stages::Symbol=:both,
              family::Symbol=:all,
              full_all::Bool=false,
              include_synthetic::Bool=true,
              include_external::Bool=true,
              external_dir::String="benchmark/data_ingestion_fixtures",
              write_fixtures::Bool=false,
              fixtures_dir::String="benchmark/data_ingestion_fixtures",
              csv_out::String="",
              preflight_probe::Bool=false,
              dense_distance_probe::Bool=false,
              stage_isolation_probe::Bool=false,
              lowdim_h0_probe::Bool=false,
              lazy_diff_thread_probe::Bool=false,
              structural_inclusion_probe::Bool=false,
              term_materialization_probe::Bool=false,
              pointcloud_dense_stream_probe::Bool=false,
              pointcloud_dim2_kernel_probe::Bool=false,
              graph_clique_probe::Bool=false,
              nn_backend_probe::Bool=false,
              h0_chain_sweep_probe::Bool=false,
              h0_active_chain_incremental_probe::Bool=false,
              h1_fastpath_probe::Bool=false,
              t2_degree_local_probe::Bool=false,
              monotone_rank_probe::Bool=false,
              monotone_incremental_probe::Bool=false,
              degree_local_all_t_probe::Bool=false,
              encoding_result_lazy_probe::Bool=false,
              structural_map_probe::Bool=false,
              packed_edgelist_probe::Bool=false,
              structural_kernel_probe::Bool=false,
              active_list_chain_probe::Bool=false,
              delaunay_cache_probe::Bool=false,
              delaunay_simplextree_fusion_probe::Bool=false,
              core_numbers_probe::Bool=false,
              landmark_radius_probe::Bool=false,
              cubical_2d_probe::Bool=false,
              dims_only_probe::Bool=false,
              solve_loop_probe::Bool=false,
              cache_hit_skip_lazy_probe::Bool=false,
              plan_norm_cache_probe::Bool=false,
              boundary_kernel_probe::Bool=false,
              pointcloud_lowdim_radius_stream_probe::Bool=false)
    stage_list = if stages == :both
        [:graded_complex, :encoding_result]
    elseif stages == :graded
        [:graded_complex]
    elseif stages == :tree
        [:simplex_tree]
    elseif stages == :full
        [:encoding_result]
    else
        error("Unsupported --stages=$(stages). Expected: both|graded|tree|full.")
    end

    includes_full = any(==(:encoding_result), stage_list)
    if quick
        if includes_full
            point_n = 12
            graph_n = 16
            clique_n = 24
            image_side = 16
        else
            point_n = 80
            graph_n = 90
            clique_n = 48
            image_side = 48
        end
    else
        if includes_full
            point_n = 16
            graph_n = 24
            clique_n = 36
            image_side = 24
        else
            point_n = 220
            graph_n = 180
            clique_n = 72
            image_side = 96
        end
    end

    synthetic_cases = include_synthetic ?
        _synthetic_cases(point_n=point_n, graph_n=graph_n, clique_n=clique_n, image_side=image_side, seed=Int(0xD470)) :
        IngestionBenchCase[]

    if write_fixtures
        path = _write_fixture_pack(fixtures_dir, synthetic_cases)
        println("Wrote ingestion fixture pack: ", path)
    end

    external_cases = include_external ? _load_external_cases(external_dir) : IngestionBenchCase[]
    cases = vcat(synthetic_cases, external_cases)
    cases = _family_filter(cases, family)
    isempty(cases) && !preflight_probe && !dense_distance_probe && !stage_isolation_probe &&
        !lowdim_h0_probe && !lazy_diff_thread_probe && !structural_inclusion_probe &&
        !term_materialization_probe && !pointcloud_dense_stream_probe && !pointcloud_dim2_kernel_probe &&
        !graph_clique_probe &&
        !nn_backend_probe &&
        !h0_chain_sweep_probe && !h0_active_chain_incremental_probe &&
        !h1_fastpath_probe && !t2_degree_local_probe &&
        !monotone_rank_probe && !monotone_incremental_probe &&
        !degree_local_all_t_probe &&
        !encoding_result_lazy_probe &&
        !structural_map_probe && !packed_edgelist_probe &&
        !structural_kernel_probe && !active_list_chain_probe &&
        !delaunay_cache_probe &&
        !delaunay_simplextree_fusion_probe &&
        !core_numbers_probe &&
        !landmark_radius_probe &&
        !cubical_2d_probe &&
        !dims_only_probe &&
        !solve_loop_probe &&
        !cache_hit_skip_lazy_probe && !plan_norm_cache_probe &&
        !boundary_kernel_probe &&
        !pointcloud_lowdim_radius_stream_probe &&
        error("No benchmark cases selected. family=$(family), include_synthetic=$(include_synthetic), include_external=$(include_external).")

    println("Data ingestion micro-benchmark")
    println("cases=", length(cases), ", reps=", reps, ", stages=", [s == :graded_complex ? "graded" : (s == :simplex_tree ? "tree" : "full") for s in stage_list])
    println("julia_threads=", Threads.nthreads())
    println("include_synthetic=", include_synthetic, ", include_external=", include_external, ", family=", family, ", full_all=", full_all)
    println("nearestneighbors_pkg=", _HAVE_NEARESTNEIGHBORS, ", pointcloud_nn_backend=", _HAVE_POINTCLOUD_NN_BACKEND)

    results = NamedTuple[]
    if !isempty(cases)
        for stage in stage_list
            stage_cases = if stage == :encoding_result && !full_all
                [c for c in cases if _full_safe_case(c)]
            elseif stage == :simplex_tree
                [c for c in cases if _tree_safe_case(c)]
            else
                cases
            end
            if stage == :encoding_result && !full_all && length(stage_cases) != length(cases)
                println("\n[info] full stage uses conservative case subset (use --full_all=1 to run all): ", length(stage_cases), "/", length(cases))
            elseif stage == :simplex_tree && length(stage_cases) != length(cases)
                println("\n[info] tree stage skips non-simplicial cases: ", length(stage_cases), "/", length(cases))
            end
            for case in stage_cases
                r_auto, r_sess = _run_case(case, stage; reps=reps)
                push!(results, r_auto)
                push!(results, r_sess)
            end
        end
    end

    if preflight_probe
        append!(results, _run_preflight_probe(; reps=reps, quick=quick))
    end
    if dense_distance_probe
        append!(results, _run_dense_distance_probe(; reps=reps, quick=quick))
    end
    if stage_isolation_probe
        append!(results, _run_stage_isolation_probe(; reps=reps, quick=quick))
    end
    if lowdim_h0_probe
        append!(results, _run_lowdim_h0_probe(; reps=reps, quick=quick))
    end
    if lazy_diff_thread_probe
        append!(results, _run_lazy_diff_thread_probe(; reps=reps, quick=quick))
    end
    if structural_inclusion_probe
        append!(results, _run_structural_inclusion_probe(; reps=reps, quick=quick))
    end
    if term_materialization_probe
        append!(results, _run_term_materialization_probe(; reps=reps, quick=quick))
    end
    if pointcloud_dense_stream_probe
        append!(results, _run_pointcloud_dense_stream_probe(; reps=reps, quick=quick))
    end
    if pointcloud_dim2_kernel_probe
        append!(results, _run_pointcloud_dim2_kernel_probe(; reps=reps, quick=quick))
    end
    if graph_clique_probe
        append!(results, _run_graph_clique_probe(; reps=reps, quick=quick))
    end
    if nn_backend_probe
        append!(results, _run_nn_backend_probe(; reps=reps, quick=quick))
    end
    if h0_chain_sweep_probe
        append!(results, _run_h0_chain_sweep_probe(; reps=reps, quick=quick))
    end
    if h0_active_chain_incremental_probe
        append!(results, _run_h0_active_chain_incremental_probe(; reps=reps, quick=quick))
    end
    if h1_fastpath_probe
        append!(results, _run_h1_fastpath_probe(; reps=reps, quick=quick))
    end
    if t2_degree_local_probe
        append!(results, _run_t2_degree_local_probe(; reps=reps, quick=quick))
    end
    if monotone_rank_probe
        append!(results, _run_monotone_rank_probe(; reps=reps, quick=quick))
    end
    if monotone_incremental_probe
        append!(results, _run_monotone_incremental_probe(; reps=reps, quick=quick))
    end
    if degree_local_all_t_probe
        append!(results, _run_degree_local_all_t_probe(; reps=reps, quick=quick))
    end
    if encoding_result_lazy_probe
        append!(results, _run_encoding_result_lazy_probe(; reps=reps, quick=quick))
    end
    if structural_map_probe
        append!(results, _run_structural_map_probe(; reps=reps, quick=quick))
    end
    if packed_edgelist_probe
        append!(results, _run_packed_edgelist_probe(; reps=reps, quick=quick))
    end
    if structural_kernel_probe
        append!(results, _run_structural_kernel_probe(; reps=reps, quick=quick))
    end
    if active_list_chain_probe
        append!(results, _run_active_list_chain_probe(; reps=reps, quick=quick))
    end
    if delaunay_cache_probe
        append!(results, _run_delaunay_cache_probe(; reps=reps, quick=quick))
    end
    if delaunay_simplextree_fusion_probe
        append!(results, _run_delaunay_simplextree_fusion_probe(; reps=reps, quick=quick))
    end
    if core_numbers_probe
        append!(results, _run_core_numbers_probe(; reps=reps, quick=quick))
    end
    if landmark_radius_probe
        append!(results, _run_landmark_radius_probe(; reps=reps, quick=quick))
    end
    if cubical_2d_probe
        append!(results, _run_cubical_2d_probe(; reps=reps, quick=quick))
    end
    if dims_only_probe
        append!(results, _run_dims_only_probe(; reps=reps, quick=quick))
    end
    if solve_loop_probe
        append!(results, _run_solve_loop_probe(; reps=reps, quick=quick))
    end
    if cache_hit_skip_lazy_probe
        append!(results, _run_cache_hit_skip_lazy_probe(; reps=reps, quick=quick))
    end
    if plan_norm_cache_probe
        append!(results, _run_plan_norm_cache_probe(; reps=reps, quick=quick))
    end
    if boundary_kernel_probe
        append!(results, _run_boundary_kernel_probe(; reps=reps, quick=quick))
    end
    if pointcloud_lowdim_radius_stream_probe
        append!(results, _run_pointcloud_lowdim_radius_stream_probe(; reps=reps, quick=quick))
    end

    println("\nSummary (slowest median times):")
    by_time = sort(results; by=r -> r.med_ms, rev=true)
    for r in by_time[1:min(end, 14)]
        println(rpad("  " * r.case * " [" * r.stage * "/" * r.cache_mode * "]", 58),
                round(r.med_ms, digits=3), " ms  ",
                round(r.med_kib, digits=1), " KiB")
    end

    if !isempty(csv_out)
        _write_results_csv(csv_out, results)
        println("\nWrote CSV results: ", csv_out)
    end

    return nothing
end

let
    args = copy(ARGS)
    profile = Symbol(lowercase(_parse_str(args, "--profile", "balanced")))
    defaults = _benchmark_profile_defaults(profile)
    reps = max(1, _parse_arg(args, "--reps", defaults.reps))
    quick = _parse_bool(args, "--quick", defaults.quick)
    stages_raw = lowercase(_parse_str(args, "--stages", "graded"))
    stages = stages_raw == "both" ? :both :
             (stages_raw == "graded" ? :graded :
             (stages_raw == "tree" ? :tree :
             (stages_raw == "full" ? :full : Symbol(stages_raw))))
    family = Symbol(lowercase(_parse_str(args, "--family", "all")))
    full_all = _parse_bool(args, "--full_all", defaults.full_all)
    include_synthetic = _parse_bool(args, "--include_synthetic", defaults.include_synthetic)
    include_external = _parse_bool(args, "--include_external", defaults.include_external)
    external_dir = _parse_str(args, "--external_dir", "benchmark/data_ingestion_fixtures")
    write_fixtures = _parse_bool(args, "--write_fixtures", false)
    fixtures_dir = _parse_str(args, "--fixtures_dir", "benchmark/data_ingestion_fixtures")
    csv_out = _parse_str(args, "--csv_out", "")
    preflight_probe = _parse_bool(args, "--preflight_probe", false)
    dense_distance_probe = _parse_bool(args, "--dense_distance_probe", false)
    stage_isolation_probe = _parse_bool(args, "--stage_isolation_probe", false)
    lowdim_h0_probe = _parse_bool(args, "--lowdim_h0_probe", false)
    lazy_diff_thread_probe = _parse_bool(args, "--lazy_diff_thread_probe", false)
    structural_inclusion_probe = _parse_bool(args, "--structural_inclusion_probe", false)
    term_materialization_probe = _parse_bool(args, "--term_materialization_probe", false)
    pointcloud_dense_stream_probe = _parse_bool(args, "--pointcloud_dense_stream_probe", false)
    pointcloud_dim2_kernel_probe = _parse_bool(args, "--pointcloud_dim2_kernel_probe", false)
    graph_clique_probe = _parse_bool(args, "--graph_clique_probe", false)
    nn_backend_probe = _parse_bool(args, "--nn_backend_probe", false)
    h0_chain_sweep_probe = _parse_bool(args, "--h0_chain_sweep_probe", false)
    h0_active_chain_incremental_probe = _parse_bool(args, "--h0_active_chain_incremental_probe", false)
    h1_fastpath_probe = _parse_bool(args, "--h1_fastpath_probe", false)
    t2_degree_local_probe = _parse_bool(args, "--t2_degree_local_probe", false)
    monotone_rank_probe = _parse_bool(args, "--monotone_rank_probe", false)
    monotone_incremental_probe = _parse_bool(args, "--monotone_incremental_probe", false)
    degree_local_all_t_probe = _parse_bool(args, "--degree_local_all_t_probe", false)
    encoding_result_lazy_probe = _parse_bool(args, "--encoding_result_lazy_probe", false)
    structural_map_probe = _parse_bool(args, "--structural_map_probe", false)
    packed_edgelist_probe = _parse_bool(args, "--packed_edgelist_probe", false)
    structural_kernel_probe = _parse_bool(args, "--structural_kernel_probe", false)
    active_list_chain_probe = _parse_bool(args, "--active_list_chain_probe", false)
    delaunay_cache_probe = _parse_bool(args, "--delaunay_cache_probe", false)
    delaunay_simplextree_fusion_probe = _parse_bool(args, "--delaunay_simplextree_fusion_probe", false)
    core_numbers_probe = _parse_bool(args, "--core_numbers_probe", false)
    landmark_radius_probe = _parse_bool(args, "--landmark_radius_probe", false)
    cubical_2d_probe = _parse_bool(args, "--cubical_2d_probe", false)
    dims_only_probe = _parse_bool(args, "--dims_only_probe", false)
    solve_loop_probe = _parse_bool(args, "--solve_loop_probe", false)
    cache_hit_skip_lazy_probe = _parse_bool(args, "--cache_hit_skip_lazy_probe", false)
    plan_norm_cache_probe = _parse_bool(args, "--plan_norm_cache_probe", false)
    boundary_kernel_probe = _parse_bool(args, "--boundary_kernel_probe", false)
    pointcloud_lowdim_radius_stream_probe = _parse_bool(args, "--pointcloud_lowdim_radius_stream_probe", false)

    println("[profile] ", profile, " (reps=", reps, ", quick=", quick,
            ", include_synthetic=", include_synthetic, ", include_external=", include_external,
            ", full_all=", full_all, ")")

    main(; reps=reps,
         quick=quick,
         stages=stages,
         family=family,
         full_all=full_all,
         include_synthetic=include_synthetic,
         include_external=include_external,
         external_dir=external_dir,
         write_fixtures=write_fixtures,
         fixtures_dir=fixtures_dir,
         csv_out=csv_out,
         preflight_probe=preflight_probe,
         dense_distance_probe=dense_distance_probe,
         stage_isolation_probe=stage_isolation_probe,
         lowdim_h0_probe=lowdim_h0_probe,
         lazy_diff_thread_probe=lazy_diff_thread_probe,
         structural_inclusion_probe=structural_inclusion_probe,
         term_materialization_probe=term_materialization_probe,
         pointcloud_dense_stream_probe=pointcloud_dense_stream_probe,
         pointcloud_dim2_kernel_probe=pointcloud_dim2_kernel_probe,
         graph_clique_probe=graph_clique_probe,
         nn_backend_probe=nn_backend_probe,
         h0_chain_sweep_probe=h0_chain_sweep_probe,
         h0_active_chain_incremental_probe=h0_active_chain_incremental_probe,
         h1_fastpath_probe=h1_fastpath_probe,
         t2_degree_local_probe=t2_degree_local_probe,
         monotone_rank_probe=monotone_rank_probe,
         monotone_incremental_probe=monotone_incremental_probe,
         degree_local_all_t_probe=degree_local_all_t_probe,
         encoding_result_lazy_probe=encoding_result_lazy_probe,
         structural_map_probe=structural_map_probe,
         packed_edgelist_probe=packed_edgelist_probe,
         structural_kernel_probe=structural_kernel_probe,
         active_list_chain_probe=active_list_chain_probe,
         delaunay_cache_probe=delaunay_cache_probe,
         delaunay_simplextree_fusion_probe=delaunay_simplextree_fusion_probe,
         core_numbers_probe=core_numbers_probe,
         landmark_radius_probe=landmark_radius_probe,
         cubical_2d_probe=cubical_2d_probe,
         dims_only_probe=dims_only_probe,
         solve_loop_probe=solve_loop_probe,
         cache_hit_skip_lazy_probe=cache_hit_skip_lazy_probe,
         plan_norm_cache_probe=plan_norm_cache_probe,
         boundary_kernel_probe=boundary_kernel_probe,
         pointcloud_lowdim_radius_stream_probe=pointcloud_lowdim_radius_stream_probe)
end
