using Random
using Printf
using SparseArrays

try
    using TamerOp
catch
    include(joinpath(@__DIR__, "..", "src", "TamerOp.jl"))
    using .TamerOp
end

const DT = TamerOp.DataTypes
const DI = TamerOp.DataIngestion

function _parse_flag(args::Vector{String}, key::String, default)
    prefix = key * "="
    for arg in args
        startswith(arg, prefix) || continue
        value = split(arg, "=", limit=2)[2]
        return default isa Int ? parse(Int, value) : value
    end
    return default
end

function _median_stats(f::Function; reps::Int)
    times = Vector{Float64}(undef, reps)
    allocs = Vector{Int}(undef, reps)
    for i in 1:reps
        stats = @timed f()
        times[i] = stats.time
        allocs[i] = stats.bytes
    end
    p = sortperm(times)
    mid = p[cld(reps, 2)]
    return (time=times[mid], bytes=allocs[mid])
end

function _point_fixture(n::Int, d::Int; seed::Int=Int(0xD771))
    rng = MersenneTwister(seed)
    A = randn(rng, n, d)
    rows = [Vector{Float64}(A[i, :]) for i in 1:n]
    return (; A, rows)
end

function _graph_fixture(n::Int, m::Int; seed::Int=Int(0xD772))
    rng = MersenneTwister(seed)
    coords = randn(rng, n, 2)
    edge_u = Vector{Int}(undef, m)
    edge_v = Vector{Int}(undef, m)
    for i in 1:m
        u = rand(rng, 1:n)
        v = rand(rng, 1:n)
        u == v && (v = (v % n) + 1)
        edge_u[i] = min(u, v)
        edge_v[i] = max(u, v)
    end
    edges = [(edge_u[i], edge_v[i]) for i in 1:m]
    weights = rand(rng, m)
    return (; coords, edge_u, edge_v, edges, weights)
end

function _embedded_fixture(nv::Int, ne::Int; seed::Int=Int(0xD773))
    rng = MersenneTwister(seed)
    verts = rand(rng, nv, 2)
    edge_u = Vector{Int}(undef, ne)
    edge_v = Vector{Int}(undef, ne)
    for i in 1:ne
        u = rand(rng, 1:nv)
        v = rand(rng, 1:nv)
        u == v && (v = (v % nv) + 1)
        edge_u[i] = min(u, v)
        edge_v[i] = max(u, v)
    end
    edges = [(edge_u[i], edge_v[i]) for i in 1:ne]
    poly_pts = rand(rng, 2 * ne, 2)
    poly_offsets = collect(1:2:(2 * ne + 1))
    polylines = [[Vector{Float64}(poly_pts[j, :]), Vector{Float64}(poly_pts[j + 1, :])] for j in 1:2:(2 * ne)]
    return (; verts, edge_u, edge_v, edges, poly_pts, poly_offsets, polylines)
end

function _complex_fixture(nv::Int, ne::Int; seed::Int=Int(0xD774))
    rng = MersenneTwister(seed)
    cells = [collect(1:nv), collect(1:ne)]
    I = Vector{Int}(undef, 2 * ne)
    J = Vector{Int}(undef, 2 * ne)
    V = Vector{Int}(undef, 2 * ne)
    t = 1
    for e in 1:ne
        u = rand(rng, 1:nv)
        v = rand(rng, 1:nv)
        u == v && (v = (v % nv) + 1)
        I[t] = u
        J[t] = e
        V[t] = -1
        t += 1
        I[t] = v
        J[t] = e
        V[t] = 1
        t += 1
    end
    boundaries = [sparse(I, J, V, nv, ne)]
    grades = [(rand(rng),) for _ in 1:(nv + ne)]
    multigrades = [[g] for g in grades]
    return (; cells, boundaries, grades, multigrades)
end

function _old_cell_dims_from_cells(cells_by_dim)
    out = Int[]
    for (d, cells) in enumerate(cells_by_dim)
        for _ in cells
            push!(out, d - 1)
        end
    end
    return out
end

function _emulate_old_graded_complex_ctor(cells_by_dim, boundaries, grades)
    total = sum(length, cells_by_dim)
    cell_dims = length(grades) == total ? _old_cell_dims_from_cells(cells_by_dim) : fill(0, length(grades))
    N = length(grades[1])
    T = eltype(grades[1])
    ng = Vector{NTuple{N,T}}(undef, length(grades))
    for i in eachindex(grades)
        ng[i] = ntuple(j -> T(grades[i][j]), N)
    end
    return (cells_by_dim, boundaries, ng, cell_dims)
end

function _emulate_old_multicritical_ctor(cells_by_dim, boundaries, grades)
    total = sum(length, cells_by_dim)
    length(grades) == total || error("multicritical benchmark fixture mismatch")
    cell_dims = _old_cell_dims_from_cells(cells_by_dim)
    first_cell = findfirst(!isempty, grades)
    N = length(grades[first_cell][1])
    T = eltype(grades[first_cell][1])
    ng = Vector{Vector{NTuple{N,T}}}(undef, length(grades))
    for i in eachindex(grades)
        gi = grades[i]
        out = Vector{NTuple{N,T}}(undef, length(gi))
        for j in eachindex(gi)
            out[j] = ntuple(k -> T(gi[j][k]), N)
        end
        ng[i] = unique(out)
    end
    return (cells_by_dim, boundaries, ng, cell_dims)
end

function main(args)
    reps = _parse_flag(args, "--reps", 7)
    n = _parse_flag(args, "--n", 4096)
    d = _parse_flag(args, "--d", 3)
    section = _parse_flag(args, "--section", "all")
    out = _parse_flag(args, "--out", joinpath(@__DIR__, "_tmp_data_types_layout_microbench.csv"))

    println("Timing policy: warm, same-process A/B baseline")
    println("reps=$reps, n=$n, d=$d, section=$section")

    point = _point_fixture(n, d)
    graph = _graph_fixture(max(64, fld(n, 8)), max(256, fld(n, 2)))
    embedded = _embedded_fixture(max(64, fld(n, 16)), max(128, fld(n, 16)))
    complex = _complex_fixture(max(64, fld(n, 8)), max(64, fld(n, 8)))
    packed_pc = DT.PointCloud(point.A)
    k = max(1, min(8, n - 1))
    radius = d <= 1 ? 0.6 : (d == 2 ? 0.8 : 1.0)
    idxs = collect(1:clamp(fld(n, 8), 4, 256))
    spec_knn = TamerOp.FiltrationSpec(
        kind=:rips,
        knn=k,
        nn_backend=:bruteforce,
        construction=TamerOp.ConstructionOptions(; sparsify=:knn, output_stage=:simplex_tree),
    )
    construction_knn = DI._construction_from_params(spec_knn.params)
    spec_radius = TamerOp.FiltrationSpec(
        kind=:rips,
        radius=radius,
        nn_backend=:bruteforce,
        construction=TamerOp.ConstructionOptions(; sparsify=:radius, output_stage=:simplex_tree),
    )
    construction_radius = DI._construction_from_params(spec_radius.params)

    cases = [
        ("pointcloud", "pointcloud_ctor_from_matrix", () -> begin
            before = _median_stats(() -> begin
                pts = [Vector{Float64}(point.A[i, :]) for i in 1:size(point.A, 1)]
                return nothing
            end; reps=reps)
            after = _median_stats(() -> DT.PointCloud(point.A); reps=reps)
            return before, after
        end),
        ("pointcloud", "pointcloud_pairwise_packed", () -> begin
            before = _median_stats(() -> DI._point_cloud_pairwise_packed(point.rows); reps=reps)
            after = _median_stats(() -> DI._point_cloud_pairwise_packed(DT.point_matrix(packed_pc)); reps=reps)
            return before, after
        end),
        ("pointcloud", "pointcloud_knn_bruteforce_graph", () -> begin
            before = _median_stats(() -> DI._point_cloud_knn_graph(packed_pc.points, k; backend=:bruteforce, approx_candidates=0); reps=reps)
            after = _median_stats(() -> DI._point_cloud_knn_graph(DT.point_matrix(packed_pc), k; backend=:bruteforce, approx_candidates=0); reps=reps)
            return before, after
        end),
        ("pointcloud", "pointcloud_radius_bruteforce_graph", () -> begin
            before = _median_stats(() -> DI._point_cloud_radius_graph(packed_pc.points, radius; backend=:bruteforce, approx_candidates=0); reps=reps)
            after = _median_stats(() -> DI._point_cloud_radius_graph(DT.point_matrix(packed_pc), radius; backend=:bruteforce, approx_candidates=0); reps=reps)
            return before, after
        end),
        ("pointcloud", "pointcloud_sparsify_knn_bruteforce", () -> begin
            before = _median_stats(() -> DI._point_cloud_sparsify_edge_driven(packed_pc.points, spec_knn, construction_knn); reps=reps)
            after = _median_stats(() -> DI._point_cloud_sparsify_edge_driven(DT.point_matrix(packed_pc), spec_knn, construction_knn); reps=reps)
            return before, after
        end),
        ("pointcloud", "pointcloud_sparsify_radius_bruteforce", () -> begin
            before = _median_stats(() -> DI._point_cloud_sparsify_edge_driven(packed_pc.points, spec_radius, construction_radius); reps=reps)
            after = _median_stats(() -> DI._point_cloud_sparsify_edge_driven(DT.point_matrix(packed_pc), spec_radius, construction_radius); reps=reps)
            return before, after
        end),
        ("pointcloud", "pointcloud_radius_indexed", () -> begin
            before = _median_stats(() -> DI._point_cloud_edges_within_radius_indexed(packed_pc.points, idxs, radius); reps=reps)
            after = _median_stats(() -> DI._point_cloud_edges_within_radius_indexed(DT.point_matrix(packed_pc), idxs, radius); reps=reps)
            return before, after
        end),
        ("graph", "graph_ctor_columnar", () -> begin
            before = _median_stats(() -> begin
                coords = [Vector{Float64}(graph.coords[i, :]) for i in 1:size(graph.coords, 1)]
                edges = graph.edges
                weights = graph.weights
                return nothing
            end; reps=reps)
            after = _median_stats(() -> DT.GraphData(size(graph.coords, 1), graph.edge_u, graph.edge_v;
                coords=graph.coords, weights=graph.weights, copy=false); reps=reps)
            return before, after
        end),
        ("embedded", "embedded_ctor_columnar", () -> begin
            before = _median_stats(() -> begin
                verts = [Vector{Float64}(embedded.verts[i, :]) for i in 1:size(embedded.verts, 1)]
                edges = embedded.edges
                polylines = embedded.polylines
                bbox = (0.0, 1.0, 0.0, 1.0)
                return nothing
            end; reps=reps)
            after = _median_stats(() -> DT.EmbeddedPlanarGraph2D(
                embedded.verts, embedded.edge_u, embedded.edge_v;
                polyline_offsets=embedded.poly_offsets,
                polyline_points=embedded.poly_pts,
                bbox=(0.0, 1.0, 0.0, 1.0),
                copy=false,
            ); reps=reps)
            return before, after
        end),
        ("complex", "graded_complex_ctor_packed_metadata", () -> begin
            before = _median_stats(() -> _emulate_old_graded_complex_ctor(complex.cells, complex.boundaries, complex.grades); reps=reps)
            after = _median_stats(() -> DT.GradedComplex(complex.cells, complex.boundaries, complex.grades); reps=reps)
            return before, after
        end),
        ("complex", "multicritical_ctor_packed_metadata", () -> begin
            before = _median_stats(() -> _emulate_old_multicritical_ctor(complex.cells, complex.boundaries, complex.multigrades); reps=reps)
            after = _median_stats(() -> DT.MultiCriticalGradedComplex(complex.cells, complex.boundaries, complex.multigrades); reps=reps)
            return before, after
        end),
    ]

    rows = String[]
    push!(rows, "case,variant,time_ms,alloc_kib")
    for (case_section, label, run_case) in cases
        (section == "all" || case_section == section) || continue
        before, after = run_case()
        push!(rows, @sprintf("%s,before,%.6f,%.3f", label, 1.0e3 * before.time, before.bytes / 1024))
        push!(rows, @sprintf("%s,after,%.6f,%.3f", label, 1.0e3 * after.time, after.bytes / 1024))
        ratio = before.time / max(after.time, eps())
        println(@sprintf("%-28s  %.3f ms -> %.3f ms  (%.2fx),  %.1f KiB -> %.1f KiB",
            label, 1.0e3 * before.time, 1.0e3 * after.time, ratio,
            before.bytes / 1024, after.bytes / 1024))
    end

    open(out, "w") do io
        write(io, join(rows, "\n"))
        write(io, "\n")
    end
    println("wrote ", out)
end

main(ARGS)
