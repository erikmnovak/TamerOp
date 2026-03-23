# Owned dataset and pipeline JSON formats.

# -----------------------------------------------------------------------------
# A0) Datasets + pipeline specs (Workflow)
# -----------------------------------------------------------------------------

function _obj_from_dataset(data)
    if data isa PointCloud
        pts = point_matrix(data)
        npts, d = size(pts)
        return Dict("kind" => "PointCloud",
                    "layout" => _DATASET_COLUMN_LAYOUT,
                    "n" => npts,
                    "d" => d,
                    "points_flat" => vec(pts))
    elseif data isa ImageNd
        return Dict("kind" => "ImageNd",
                    "size" => collect(size(data.data)),
                    "data" => collect(vec(data.data)))
    elseif data isa GraphData
        edges_u, edges_v = edge_columns(data)
        coords_dim = nothing
        coords_flat = nothing
        coords = coord_matrix(data)
        if coords !== nothing
            ncoords, d = size(coords)
            ncoords == data.n || error("GraphData coords row count must equal n for columnar serialization.")
            coords_dim = d
            coords_flat = vec(coords)
        end
        return Dict("kind" => "GraphData",
                    "layout" => _DATASET_COLUMN_LAYOUT,
                    "n" => data.n,
                    "edges_u" => edges_u,
                    "edges_v" => edges_v,
                    "coords_dim" => coords_dim,
                    "coords_flat" => coords_flat,
                    "weights" => getfield(data, :weights))
    elseif data isa EmbeddedPlanarGraph2D
        return Dict("kind" => "EmbeddedPlanarGraph2D",
                    "vertices" => [collect(v) for v in data.vertices],
                    "edges" => [collect(e) for e in data.edges],
                    "polylines" => data.polylines === nothing ? nothing : [[collect(p) for p in poly] for poly in data.polylines],
                    "bbox" => data.bbox === nothing ? nothing : collect(data.bbox))
    elseif data isa GradedComplex
        bnds = Any[]
        for B in data.boundaries
            Ii, Jj, Vv = findnz(B)
            push!(bnds, Dict(
                "m" => size(B, 1),
                "n" => size(B, 2),
                "I" => collect(Ii),
                "J" => collect(Jj),
                "V" => collect(Vv),
            ))
        end
        return Dict("kind" => "GradedComplex",
                    "cells_by_dim" => [collect(c) for c in data.cells_by_dim],
                    "boundaries" => bnds,
                    "grades" => [collect(g) for g in data.grades],
                    "cell_dims" => collect(data.cell_dims))
    elseif data isa MultiCriticalGradedComplex
        bnds = Any[]
        for B in data.boundaries
            Ii, Jj, Vv = findnz(B)
            push!(bnds, Dict(
                "m" => size(B, 1),
                "n" => size(B, 2),
                "I" => collect(Ii),
                "J" => collect(Jj),
                "V" => collect(Vv),
            ))
        end
        return Dict("kind" => "MultiCriticalGradedComplex",
                    "cells_by_dim" => [collect(c) for c in data.cells_by_dim],
                    "boundaries" => bnds,
                    "grades" => [[collect(g) for g in gs] for gs in data.grades],
                    "cell_dims" => collect(data.cell_dims))
    elseif data isa SimplexTreeMulti
        return Dict("kind" => "SimplexTreeMulti",
                    "simplex_offsets" => collect(data.simplex_offsets),
                    "simplex_vertices" => collect(data.simplex_vertices),
                    "simplex_dims" => collect(data.simplex_dims),
                    "dim_offsets" => collect(data.dim_offsets),
                    "grade_offsets" => collect(data.grade_offsets),
                    "grade_data" => [collect(g) for g in data.grade_data])
    else
        error("Unsupported dataset type for serialization.")
    end
end

function _dataset_from_obj(obj)
    kind = String(obj["kind"])
    if kind == "PointCloud"
        haskey(obj, "points_flat") || error("PointCloud JSON missing canonical `points_flat` payload.")
        haskey(obj, "layout") || error("PointCloud JSON missing canonical `layout` payload.")
        haskey(obj, "n") || error("PointCloud JSON missing canonical `n` payload.")
        haskey(obj, "d") || error("PointCloud JSON missing canonical `d` payload.")
        _require_dataset_layout(String(obj["layout"]), kind)
        n = Int(obj["n"])
        d = Int(obj["d"])
        return _pointcloud_from_flat(n, d, Vector{Float64}(obj["points_flat"]))
    elseif kind == "ImageNd"
        sz = Vector{Int}(obj["size"])
        flat = Vector{Float64}(obj["data"])
        data = reshape(flat, Tuple(sz))
        return ImageNd(data)
    elseif kind == "GraphData"
        haskey(obj, "n") || error("GraphData JSON missing canonical `n` payload.")
        n = Int(obj["n"])
        weights = obj["weights"] === nothing ? nothing : Vector{Float64}(obj["weights"])
        haskey(obj, "edges_u") || error("GraphData JSON missing canonical `edges_u` payload.")
        haskey(obj, "edges_v") || error("GraphData JSON missing canonical `edges_v` payload.")
        haskey(obj, "layout") || error("GraphData JSON missing canonical `layout` payload.")
        _require_dataset_layout(String(obj["layout"]), kind)
        coords_dim = haskey(obj, "coords_dim") && obj["coords_dim"] !== nothing ? Int(obj["coords_dim"]) : nothing
        coords_flat = haskey(obj, "coords_flat") && obj["coords_flat"] !== nothing ?
            Vector{Float64}(obj["coords_flat"]) : nothing
        return _graph_from_columns(n,
                                   Vector{Int}(obj["edges_u"]),
                                   Vector{Int}(obj["edges_v"]);
                                   coords_dim=coords_dim,
                                   coords_flat=coords_flat,
                                   weights=weights)
    elseif kind == "EmbeddedPlanarGraph2D"
        verts = [Vector{Float64}(v) for v in obj["vertices"]]
        edges = [ (Int(e[1]), Int(e[2])) for e in obj["edges"] ]
        polylines = obj["polylines"] === nothing ? nothing :
            [[Vector{Float64}(p) for p in poly] for poly in obj["polylines"]]
        bbox = obj["bbox"] === nothing ? nothing : (Float64(obj["bbox"][1]),
                                                   Float64(obj["bbox"][2]),
                                                   Float64(obj["bbox"][3]),
                                                   Float64(obj["bbox"][4]))
        return EmbeddedPlanarGraph2D(verts, edges; polylines=polylines, bbox=bbox)
    elseif kind == "GradedComplex"
        cells = [Vector{Int}(c) for c in obj["cells_by_dim"]]
        boundaries = SparseMatrixCSC{Int,Int}[]
        for b in obj["boundaries"]
            m = Int(b["m"]); n = Int(b["n"])
            I = Vector{Int}(b["I"])
            J = Vector{Int}(b["J"])
            V = Vector{Int}(b["V"])
            push!(boundaries, sparse(I, J, V, m, n))
        end
        grades = [Vector{Float64}(g) for g in obj["grades"]]
        cell_dims = Vector{Int}(obj["cell_dims"])
        return GradedComplex(cells, boundaries, grades; cell_dims=cell_dims)
    elseif kind == "MultiCriticalGradedComplex"
        cells = [Vector{Int}(c) for c in obj["cells_by_dim"]]
        boundaries = SparseMatrixCSC{Int,Int}[]
        for b in obj["boundaries"]
            m = Int(b["m"]); n = Int(b["n"])
            I = Vector{Int}(b["I"])
            J = Vector{Int}(b["J"])
            V = Vector{Int}(b["V"])
            push!(boundaries, sparse(I, J, V, m, n))
        end
        grades = [[Vector{Float64}(g) for g in gs] for gs in obj["grades"]]
        cell_dims = Vector{Int}(obj["cell_dims"])
        return MultiCriticalGradedComplex(cells, boundaries, grades; cell_dims=cell_dims)
    elseif kind == "SimplexTreeMulti"
        simplex_offsets = Vector{Int}(obj["simplex_offsets"])
        simplex_vertices = Vector{Int}(obj["simplex_vertices"])
        simplex_dims = Vector{Int}(obj["simplex_dims"])
        dim_offsets = Vector{Int}(obj["dim_offsets"])
        grade_offsets = Vector{Int}(obj["grade_offsets"])
        raw_grades = obj["grade_data"]
        isempty(raw_grades) && error("SimplexTreeMulti JSON payload has empty grade_data.")
        N = length(raw_grades[1])
        grade_data = Vector{NTuple{N,Float64}}(undef, length(raw_grades))
        for i in eachindex(raw_grades)
            g = raw_grades[i]
            length(g) == N || error("SimplexTreeMulti JSON grade arity mismatch at index $i.")
            grade_data[i] = ntuple(k -> Float64(g[k]), N)
        end
        return SimplexTreeMulti(simplex_offsets, simplex_vertices, simplex_dims,
                                dim_offsets, grade_offsets, grade_data)
    else
        error("Unknown dataset kind: $kind")
    end
end

@inline function _construction_budget_obj(b::ConstructionBudget)
    return Dict(
        "max_simplices" => b.max_simplices,
        "max_edges" => b.max_edges,
        "memory_budget_bytes" => b.memory_budget_bytes,
    )
end

@inline function _construction_options_obj(c::ConstructionOptions)
    return Dict(
        "sparsify" => String(c.sparsify),
        "collapse" => String(c.collapse),
        "output_stage" => String(c.output_stage),
        "budget" => _construction_budget_obj(c.budget),
    )
end

function _spec_obj(spec::FiltrationSpec)
    params = Dict{String,Any}()
    for (k, v) in pairs(spec.params)
        if k == :construction
            if v isa ConstructionOptions
                params["construction"] = _construction_options_obj(v)
            elseif v isa ConstructionBudget
                params["construction"] = Dict("budget" => _construction_budget_obj(v))
            else
                params["construction"] = v
            end
        else
            params[String(k)] = v
        end
    end
    return Dict("kind" => String(spec.kind), "params" => params)
end

function _spec_from_obj(obj)
    kind = Symbol(String(obj["kind"]))
    params_obj = obj["params"]
    params = (; (Symbol(k) => params_obj[k] for k in keys(params_obj))...)
    return FiltrationSpec(; kind=kind, params...)
end

function _pipeline_options_from_spec(spec::FiltrationSpec)
    p = spec.params
    return PipelineOptions(;
        orientation = get(p, :orientation, nothing),
        axes_policy = Symbol(get(p, :axes_policy, :encoding)),
        axis_kind = get(p, :axis_kind, nothing),
        eps = get(p, :eps, nothing),
        poset_kind = Symbol(get(p, :poset_kind, :signature)),
        field = get(p, :field, nothing),
        max_axis_len = get(p, :max_axis_len, nothing),
    )
end

function _pipeline_options_from_any(spec::FiltrationSpec, x)
    if x === nothing
        return _pipeline_options_from_spec(spec)
    elseif x isa PipelineOptions
        return x
    elseif x isa NamedTuple
        return PipelineOptions(; x...)
    elseif x isa AbstractDict
        vals = (; (Symbol(k) => x[k] for k in keys(x))...)
        return PipelineOptions(; vals...)
    end
    throw(ArgumentError("pipeline_opts must be nothing, PipelineOptions, NamedTuple, or AbstractDict."))
end

function _pipeline_options_obj(opts::PipelineOptions)
    return Dict(
        "orientation" => opts.orientation,
        "axes_policy" => String(opts.axes_policy),
        "axis_kind" => opts.axis_kind,
        "eps" => opts.eps,
        "poset_kind" => String(opts.poset_kind),
        "field" => opts.field,
        "max_axis_len" => opts.max_axis_len,
    )
end

function _pipeline_options_from_obj(obj)::PipelineOptions
    orient_raw = get(obj, "orientation", nothing)
    orientation = if orient_raw isa AbstractVector
        ntuple(i -> Int(orient_raw[i]), length(orient_raw))
    else
        orient_raw
    end
    axis_kind_raw = get(obj, "axis_kind", nothing)
    axis_kind = axis_kind_raw isa AbstractString ? Symbol(axis_kind_raw) : axis_kind_raw
    field_raw = get(obj, "field", nothing)
    field = field_raw isa AbstractString ? Symbol(field_raw) : field_raw
    return PipelineOptions(;
        orientation = orientation,
        axes_policy = Symbol(get(obj, "axes_policy", "encoding")),
        axis_kind = axis_kind,
        eps = get(obj, "eps", nothing),
        poset_kind = Symbol(get(obj, "poset_kind", "signature")),
        field = field,
        max_axis_len = get(obj, "max_axis_len", nothing),
    )
end

"""
    save_dataset_json(path, data; profile=:compact, pretty=nothing)

Serialize a dataset into the stable TamerOp-owned dataset schema.

This is the canonical owned write path for datasets such as `PointCloud`,
`GraphData`, `ImageNd`, `EmbeddedPlanarGraph2D`, `GradedComplex`,
`MultiCriticalGradedComplex`, and `SimplexTreeMulti`.

This is not the cheap inspection path. Use [`dataset_json_summary`](@ref) or
[`inspect_json`](@ref) to inspect an existing artifact cheaply, and use
[`check_dataset_json`](@ref) when you need strict schema validation before
calling [`load_dataset_json`](@ref).

Use `profile=:compact` (default) for compact writes and `profile=:debug` for a
pretty-printed artifact.
"""
function save_dataset_json(path::AbstractString, data;
                           profile::Symbol=:compact,
                           pretty::Union{Nothing,Bool}=nothing)
    return _json_write(path, _obj_from_dataset(data);
                       pretty=_resolve_owned_json_pretty(profile, pretty))
end

"""
    load_dataset_json(path; validation=:strict)

Load a dataset serialized by [`save_dataset_json`](@ref).

This is a strict owned-schema loader. Prefer [`dataset_json_summary`](@ref) for
cheap-first inspection and [`check_dataset_json`](@ref) when you need explicit
validation before loading an existing artifact.

Use `validation=:trusted` only for TamerOp-produced files that already
passed schema validation and need the lighter hot load path.
"""
function load_dataset_json(path::AbstractString; validation::Symbol=:strict)
    validation === :strict && return _load_dataset_json_strict(path)
    validation === :trusted && return _load_dataset_json_trusted(path)
    _resolve_validation_mode(validation)
    error("unreachable validation mode")
end

"""
    save_pipeline_json(path, data, spec; degree=nothing, pipeline_opts=nothing, profile=:compact, pretty=nothing)

Serialize a dataset, filtration spec, degree, and structured `PipelineOptions`
into the stable TamerOp-owned pipeline schema.

This is the canonical owned artifact for replaying a workflow setup. Use
[`pipeline_json_summary`](@ref) or [`inspect_json`](@ref) to inspect an
existing artifact cheaply before deciding whether to validate or load it.

Use `profile=:compact` (default) for compact writes and `profile=:debug` for a
pretty-printed artifact.
"""
function save_pipeline_json(path::AbstractString, data, spec::FiltrationSpec;
                            degree=nothing,
                            pipeline_opts=nothing,
                            profile::Symbol=:compact,
                            pretty::Union{Nothing,Bool}=nothing)
    popts = _pipeline_options_from_any(spec, pipeline_opts)
    obj = Dict(
        "schema_version" => PIPELINE_SCHEMA_VERSION,
        "dataset" => _obj_from_dataset(data),
        "spec" => _spec_obj(spec),
        "degree" => degree,
        "pipeline_options" => _pipeline_options_obj(popts),
    )
    return _json_write(path, obj; pretty=_resolve_owned_json_pretty(profile, pretty))
end

"""
    load_pipeline_json(path; validation=:strict) -> (data, spec, degree, pipeline_opts)

Load a pipeline artifact written by [`save_pipeline_json`](@ref).

This is a strict owned-schema loader returning the dataset, filtration spec,
degree, and `PipelineOptions`. Prefer [`pipeline_json_summary`](@ref) for a
cheap-first family check and [`check_pipeline_json`](@ref) for explicit schema
validation before loading.

Use `validation=:trusted` only for TamerOp-produced artifacts when you
want the lighter replay path and are willing to trust the stored schema.
"""
function load_pipeline_json(path::AbstractString; validation::Symbol=:strict)
    validation === :strict && return _load_pipeline_json_strict(path)
    validation === :trusted && return _load_pipeline_json_trusted(path)
    _resolve_validation_mode(validation)
    error("unreachable validation mode")
end

