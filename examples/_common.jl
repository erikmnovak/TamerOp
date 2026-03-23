# =============================================================================
# examples/_common.jl
#
# Shared helpers for onboarding examples.
#
# Design goals:
# - Keep examples deterministic and reproducible.
# - Use canonical library IO paths (save_features/save_dataset_json/save_pipeline_json).
# - Make each script runnable as a plain Julia file.
# =============================================================================

using Random
using Printf
using Dates
using SparseArrays
using LinearAlgebra
using Statistics

let local_src = normpath(joinpath(@__DIR__, "..", "src", "TamerOp.jl"))
    if isfile(local_src)
        if isdefined(Main, :TamerOp)
            loaded_path = try
                pathof(Main.TamerOp)
            catch
                nothing
            end
            if loaded_path !== nothing && normpath(loaded_path) != local_src
                error("examples/_common.jl expected the repo-local TamerOp at $(local_src), but this session already loaded $(loaded_path). Restart the kernel or Julia session and rerun.")
            end
        else
            include(local_src)
        end
    else
        using TamerOp
    end
end

const TO = TamerOp

"Print a consistent banner so examples are easy to scan in terminal output."
function example_header(tag::AbstractString,
                        title::AbstractString;
                        theme::AbstractString,
                        teaches::Vector{String}=String[])
    println("\n", "="^88)
    println("Example $(tag): $(title)")
    println("="^88)
    println("Theme: ", theme)
    if !isempty(teaches)
        println("Teaches:")
        for t in teaches
            println("  - ", t)
        end
    end
end

"Print a compact stage marker in each script."
function stage(msg::AbstractString)
    println("\n", "-"^72)
    println(msg)
    println("-"^72)
end

"Create a stable per-example output directory under examples/_outputs/."
function example_outdir(name::AbstractString)
    out = joinpath(@__DIR__, "_outputs", name)
    mkpath(out)
    return out
end

"Save canonical feature artifacts (CSV wide/long) plus optional NPZ."
function save_feature_bundle(outdir::AbstractString,
                             stem::AbstractString,
                             fs::TO.FeatureSet)
    mkpath(outdir)

    csv_wide = joinpath(outdir, stem * "__wide.csv")
    csv_long = joinpath(outdir, stem * "__long.csv")
    TO.save_features(csv_wide, fs; format=:csv, mode=:wide, metadata=true)
    TO.save_features(csv_long, fs; format=:csv, mode=:long, metadata=true)

    native_paths = Dict{Symbol,String}()

    native_paths[:csv_wide] = csv_wide
    native_paths[:csv_long] = csv_long

    # Optional NPZ extension path.
    try
        p = joinpath(outdir, stem * ".npz")
        TO.save_features(p, fs; format=:npz, mode=:wide, metadata=true)
        native_paths[:npz] = p
    catch
    end

    return (csv_wide=csv_wide, csv_long=csv_long, native=native_paths)
end

"Check deterministic feature reproducibility and return max absolute deviation."
function assert_feature_sets_match(fs_ref::TO.FeatureSet,
                                   fs_new::TO.FeatureSet;
                                   atol::Float64=1e-10,
                                   rtol::Float64=1e-8)
    size(fs_ref.X) == size(fs_new.X) ||
        throw(ArgumentError("Feature matrix shape mismatch: $(size(fs_ref.X)) vs $(size(fs_new.X))"))
    fs_ref.names == fs_new.names ||
        throw(ArgumentError("Feature names mismatch"))
    fs_ref.ids == fs_new.ids ||
        throw(ArgumentError("Feature ids mismatch"))

    maxabs = maximum(abs.(fs_ref.X .- fs_new.X))
    all(isapprox.(fs_ref.X, fs_new.X; atol=atol, rtol=rtol)) ||
        throw(ArgumentError("Feature values are not reproducible within tolerances (maxabs=$(maxabs), atol=$(atol), rtol=$(rtol))."))
    return maxabs
end

"Return true when an optional extension module is loaded."
@inline function extension_loaded(ext_name::Symbol)::Bool
    return Base.get_extension(TO, ext_name) !== nothing
end

"Pick experiment output formats that are currently available in-process."
function available_experiment_formats()
    out = Symbol[]
    extension_loaded(:TamerOpArrowExt) && push!(out, :arrow)
    extension_loaded(:TamerOpParquet2Ext) && push!(out, :parquet)
    extension_loaded(:TamerOpNPZExt) && push!(out, :npz)
    if extension_loaded(:TamerOpCSVExt)
        push!(out, :csv_wide)
        push!(out, :csv_long)
    end
    return out
end

"Create EncodingResult-backed samples with stable ids and labels for batch APIs."
function to_encoding_samples(encodings::AbstractVector,
                             labels::AbstractVector{<:AbstractString};
                             prefix::AbstractString="sample")
    length(encodings) == length(labels) ||
        throw(ArgumentError("encodings/labels length mismatch"))
    out = Vector{NamedTuple}(undef, length(encodings))
    @inbounds for i in eachindex(encodings)
        enc = encodings[i]
        out[i] = (;
            M=enc.M,
            pi=enc.pi,
            label=String(labels[i]),
            id=@sprintf("%s_%03d", prefix, i),
        )
    end
    return out
end

# -----------------------------------------------------------------------------
# Deterministic synthetic datasets used by the examples.
# -----------------------------------------------------------------------------

function make_noisy_circle_cloud(n::Int;
                                 radius::Float64=1.0,
                                 noise::Float64=0.03,
                                 seed::Int=0)
    rng = MersenneTwister(seed)
    pts = Vector{Vector{Float64}}(undef, n)
    @inbounds for i in 1:n
        th = 2pi * (i - 1) / n
        x = radius * cos(th) + noise * randn(rng)
        y = radius * sin(th) + noise * randn(rng)
        pts[i] = [x, y]
    end
    return TO.PointCloud(pts)
end

function make_noisy_figure_eight_cloud(n::Int;
                                       scale::Float64=1.0,
                                       noise::Float64=0.03,
                                       seed::Int=1)
    rng = MersenneTwister(seed)
    pts = Vector{Vector{Float64}}(undef, n)
    @inbounds for i in 1:n
        th = 2pi * (i - 1) / n
        x = scale * sin(th)
        y = scale * sin(th) * cos(th)
        pts[i] = [x + noise * randn(rng), y + noise * randn(rng)]
    end
    return TO.PointCloud(pts)
end

function make_pointcloud_dataset(; n_per_class::Int=12,
                                  n_points::Int=64,
                                  seed::Int=20260217)
    samples = TO.PointCloud[]
    labels = String[]
    for i in 1:n_per_class
        push!(samples, make_noisy_circle_cloud(n_points; seed=seed + i))
        push!(labels, "circle")
    end
    for i in 1:n_per_class
        push!(samples, make_noisy_figure_eight_cloud(n_points; seed=seed + 10_000 + i))
        push!(labels, "figure8")
    end
    return samples, labels
end

function make_image_distance_dataset(; n_per_class::Int=6,
                                      side::Int=48,
                                      seed::Int=20260217)
    rng = MersenneTwister(seed)
    imgs = TO.ImageNd[]
    masks = BitMatrix[]
    labels = String[]

    # Class A: annulus-like binary support with slight jitter.
    for i in 1:n_per_class
        img = Matrix{Float64}(undef, side, side)
        cx = side / 2 + 0.8 * randn(rng)
        cy = side / 2 + 0.8 * randn(rng)
        @inbounds for y in 1:side, x in 1:side
            dx = (x - cx) / side
            dy = (y - cy) / side
            r = sqrt(dx * dx + dy * dy)
            img[y, x] = exp(-((r - 0.22)^2) / 0.004) + 0.03 * randn(rng)
        end
        mask = img .> quantile(vec(img), 0.62)
        push!(imgs, TO.ImageNd(img))
        push!(masks, mask)
        push!(labels, "annulus")
    end

    # Class B: two-blob support.
    for i in 1:n_per_class
        img = Matrix{Float64}(undef, side, side)
        c1x = 0.34 * side + 0.8 * randn(rng)
        c1y = 0.50 * side + 0.8 * randn(rng)
        c2x = 0.66 * side + 0.8 * randn(rng)
        c2y = 0.50 * side + 0.8 * randn(rng)
        @inbounds for y in 1:side, x in 1:side
            g1 = exp(-(((x - c1x)^2 + (y - c1y)^2) / (2 * 3.5^2)))
            g2 = exp(-(((x - c2x)^2 + (y - c2y)^2) / (2 * 3.5^2)))
            img[y, x] = g1 + g2 + 0.03 * randn(rng)
        end
        mask = img .> quantile(vec(img), 0.70)
        push!(imgs, TO.ImageNd(img))
        push!(masks, mask)
        push!(labels, "two_blobs")
    end

    return imgs, masks, labels
end

function _graph_edges_random_geometric(coords::Vector{Vector{Float64}}, r::Float64)
    n = length(coords)
    edges = Tuple{Int,Int}[]
    for i in 1:n
        for j in (i + 1):n
            dx = coords[i][1] - coords[j][1]
            dy = coords[i][2] - coords[j][2]
            if sqrt(dx * dx + dy * dy) <= r
                push!(edges, (i, j))
            end
        end
    end
    return edges
end

function _graph_edges_sbm(n::Int, p_in::Float64, p_out::Float64, rng)
    edges = Tuple{Int,Int}[]
    half = div(n, 2)
    for i in 1:n
        for j in (i + 1):n
            same = (i <= half && j <= half) || (i > half && j > half)
            p = same ? p_in : p_out
            rand(rng) <= p && push!(edges, (i, j))
        end
    end
    return edges
end

function make_graph_dataset(; n_per_class::Int=8,
                             n_vertices::Int=48,
                             seed::Int=20260217)
    rng = MersenneTwister(seed)
    graphs = TO.GraphData[]
    labels = String[]

    # Class A: random geometric graph.
    for _ in 1:n_per_class
        coords = [[rand(rng), rand(rng)] for _ in 1:n_vertices]
        edges = _graph_edges_random_geometric(coords, 0.24)
        isempty(edges) && push!(edges, (1, 2))
        weights = [hypot(coords[u][1] - coords[v][1], coords[u][2] - coords[v][2]) for (u, v) in edges]
        push!(graphs, TO.GraphData(n_vertices, edges; coords=coords, weights=weights))
        push!(labels, "geometric")
    end

    # Class B: two-community SBM.
    for _ in 1:n_per_class
        coords = [[rand(rng), rand(rng)] for _ in 1:n_vertices]
        edges = _graph_edges_sbm(n_vertices, 0.20, 0.03, rng)
        isempty(edges) && push!(edges, (1, 2))
        weights = fill(1.0, length(edges))
        push!(graphs, TO.GraphData(n_vertices, edges; coords=coords, weights=weights))
        push!(labels, "sbm")
    end

    return graphs, labels
end
