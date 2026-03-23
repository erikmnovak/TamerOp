# =============================================================================
# DataTypes.jl
#
# Shared typed ingestion/data containers used by ingestion, serialization, and
# workflow layers.
# =============================================================================
module DataTypes

using SparseArrays

function npoints end
function ambient_dim end
function nedges end
function nvertices end
function max_dim end
function cell_counts end
function parameter_dim end
function edge_list end
function vertex_positions end
function vertex_coordinates end
function edge_weights end
function polylines end
function bounding_box end
function cell_grades end
function boundary_maps end
function cells end
function cell_grade end
function boundary_map end
function cell_grade_set end
function nsimplices end
function simplices end
function simplex_dimension end
function simplex_grade_set end
function data_validation_summary end
function data_summary end

"""
    DataValidationSummary

Display wrapper for validation reports returned by the `check_*` helpers in
`DataTypes`.

The canonical data validators still return structured `NamedTuple`s for easy
programmatic inspection. Wrap one of those reports with
[`data_validation_summary`](@ref) when you want a compact notebook/REPL
presentation of the same information.
"""
struct DataValidationSummary{R}
    report::R
end

"""
    data_validation_summary(report) -> DataValidationSummary

Wrap a data-container validation report in a display-oriented object.

Use this when you want the report returned by helpers such as
[`check_point_cloud`](@ref), [`check_graph_data`](@ref), or
[`check_graded_complex`](@ref) to print as a compact mathematical summary with
issues laid out line by line.
"""
@inline data_validation_summary(report::NamedTuple) = DataValidationSummary(report)

@inline function _validation_report(kind::Symbol, valid::Bool; kwargs...)
    return (; kind, valid, kwargs...)
end

@inline function _throw_invalid_data(kind::AbstractString, issues)
    msg = isempty(issues) ? "invalid $(kind)" : join(string.(issues), "; ")
    throw(ArgumentError("$(kind): $(msg)"))
end

struct MatrixRowsView{T,M<:AbstractMatrix{T}} <:
       AbstractVector{SubArray{T,1,M,Tuple{Int,Base.Slice{Base.OneTo{Int}}},true}}
    mat::M
end

Base.IndexStyle(::Type{<:MatrixRowsView}) = IndexLinear()
Base.size(v::MatrixRowsView) = (size(v.mat, 1),)
Base.axes(v::MatrixRowsView) = (Base.OneTo(size(v.mat, 1)),)
@inline Base.getindex(v::MatrixRowsView, i::Int) = @view v.mat[i, :]
@inline function Base.iterate(v::MatrixRowsView, state::Int=1)
    state > size(v.mat, 1) && return nothing
    return (@view(v.mat[state, :]), state + 1)
end

struct EdgePairsView{U<:AbstractVector{Int},V<:AbstractVector{Int}} <: AbstractVector{Tuple{Int,Int}}
    u::U
    v::V
end

Base.IndexStyle(::Type{<:EdgePairsView}) = IndexLinear()
Base.size(v::EdgePairsView) = (length(v.u),)
Base.axes(v::EdgePairsView) = (Base.OneTo(length(v.u)),)
@inline Base.getindex(v::EdgePairsView, i::Int) = (v.u[i], v.v[i])
@inline function Base.iterate(v::EdgePairsView, state::Int=1)
    state > length(v.u) && return nothing
    return ((v.u[state], v.v[state]), state + 1)
end

struct PolylineRowsView{T,M<:AbstractMatrix{T}} <:
       AbstractVector{SubArray{T,1,M,Tuple{Int,Base.Slice{Base.OneTo{Int}}},true}}
    mat::M
    lo::Int
    hi::Int
end

Base.IndexStyle(::Type{<:PolylineRowsView}) = IndexLinear()
Base.size(v::PolylineRowsView) = (max(0, v.hi - v.lo + 1),)
Base.axes(v::PolylineRowsView) = (Base.OneTo(max(0, v.hi - v.lo + 1)),)
@inline Base.getindex(v::PolylineRowsView, i::Int) = @view v.mat[v.lo + i - 1, :]
@inline function Base.iterate(v::PolylineRowsView, state::Int=1)
    state > max(0, v.hi - v.lo + 1) && return nothing
    return (@view(v.mat[v.lo + state - 1, :]), state + 1)
end

struct PolylinesView{T,M<:AbstractMatrix{T},O<:AbstractVector{Int}} <: AbstractVector{PolylineRowsView{T,M}}
    offsets::O
    points::M
end

Base.IndexStyle(::Type{<:PolylinesView}) = IndexLinear()
Base.size(v::PolylinesView) = (length(v.offsets) - 1,)
Base.axes(v::PolylinesView) = (Base.OneTo(length(v.offsets) - 1),)
@inline function Base.getindex(v::PolylinesView{T,M}, i::Int) where {T,M}
    lo = v.offsets[i]
    hi = v.offsets[i + 1] - 1
    return PolylineRowsView{T,M}(v.points, lo, hi)
end
@inline function Base.iterate(v::PolylinesView{T,M}, state::Int=1) where {T,M}
    state > length(v.offsets) - 1 && return nothing
    return (v[state], state + 1)
end

struct SegmentedSlicesView{T,V<:AbstractVector{T},O<:AbstractVector{Int}} <:
       AbstractVector{SubArray{T,1,V,Tuple{UnitRange{Int}},true}}
    data::V
    offsets::O
end

Base.IndexStyle(::Type{<:SegmentedSlicesView}) = IndexLinear()
Base.size(v::SegmentedSlicesView) = (length(v.offsets) - 1,)
Base.axes(v::SegmentedSlicesView) = (Base.OneTo(length(v.offsets) - 1),)
@inline function Base.getindex(v::SegmentedSlicesView{T,V}, i::Int) where {T,V}
    lo = v.offsets[i]
    hi = v.offsets[i + 1] - 1
    return @view v.data[lo:hi]
end
@inline function Base.iterate(v::SegmentedSlicesView{T,V}, state::Int=1) where {T,V}
    state > length(v.offsets) - 1 && return nothing
    return (v[state], state + 1)
end

struct RepeatedDimsView{O<:AbstractVector{Int}} <: AbstractVector{Int}
    offsets::O
end

Base.IndexStyle(::Type{<:RepeatedDimsView}) = IndexLinear()
Base.size(v::RepeatedDimsView) = (last(v.offsets) - 1,)
Base.axes(v::RepeatedDimsView) = (Base.OneTo(last(v.offsets) - 1),)
@inline function Base.getindex(v::RepeatedDimsView, i::Int)
    1 <= i <= last(v.offsets) - 1 || throw(BoundsError(v, i))
    return searchsortedlast(v.offsets, i) - 1
end
@inline function Base.iterate(v::RepeatedDimsView, state::Int=1)
    state > last(v.offsets) - 1 && return nothing
    return (v[state], state + 1)
end

@inline function _rows_to_matrix(::Type{T}, rows::AbstractVector{<:AbstractVector}) where {T}
    n = length(rows)
    d = n == 0 ? 0 : length(rows[1])
    A = Matrix{T}(undef, n, d)
    @inbounds for i in 1:n
        row = rows[i]
        length(row) == d || error("row data must have uniform dimension.")
        for j in 1:d
            A[i, j] = T(row[j])
        end
    end
    return A
end

@inline function _rows_to_matrix(::Type{T}, rows::AbstractVector{<:Tuple}) where {T}
    n = length(rows)
    d = n == 0 ? 0 : length(rows[1])
    A = Matrix{T}(undef, n, d)
    @inbounds for i in 1:n
        row = rows[i]
        length(row) == d || error("row data must have uniform dimension.")
        for j in 1:d
            A[i, j] = T(row[j])
        end
    end
    return A
end

@inline function _split_edges(edges::AbstractVector{<:Tuple{Int,Int}})
    m = length(edges)
    u = Vector{Int}(undef, m)
    v = Vector{Int}(undef, m)
    @inbounds for i in 1:m
        u[i], v[i] = edges[i]
    end
    return u, v
end

@inline function _flatten_polylines(::Type{T}, polylines::AbstractVector) where {T}
    nlines = length(polylines)
    offsets = Vector{Int}(undef, nlines + 1)
    offsets[1] = 1
    total = 0
    d = 0
    if nlines > 0 && !isempty(polylines[1])
        d = length(polylines[1][1])
    end
    @inbounds for i in 1:nlines
        poly = polylines[i]
        if !isempty(poly)
            d == 0 && (d = length(poly[1]))
            for p in poly
                length(p) == d || error("polylines must have uniform point dimension.")
            end
        end
        total += length(poly)
        offsets[i + 1] = total + 1
    end
    pts = Matrix{T}(undef, total, d)
    t = 1
    @inbounds for poly in polylines
        for p in poly
            for j in 1:d
                pts[t, j] = T(p[j])
            end
            t += 1
        end
    end
    return offsets, pts
end

@inline function _pack_cells(cells_by_dim::AbstractVector{<:AbstractVector{<:Integer}})
    nd = length(cells_by_dim)
    total = sum(length, cells_by_dim)
    cell_ids = Vector{Int}(undef, total)
    dim_offsets = Vector{Int}(undef, nd + 1)
    dim_offsets[1] = 1
    t = 1
    @inbounds for d in 1:nd
        cells = cells_by_dim[d]
        for cell in cells
            cell_ids[t] = Int(cell)
            t += 1
        end
        dim_offsets[d + 1] = t
    end
    return cell_ids, dim_offsets
end

@inline function _validate_cell_dims(cell_dims::AbstractVector{<:Integer},
                                     dim_offsets::AbstractVector{Int},
                                     label::AbstractString)
    length(cell_dims) == last(dim_offsets) - 1 || error("$(label): cell_dims length mismatch.")
    idx = 1
    @inbounds for d in 1:(length(dim_offsets) - 1)
        for _ in dim_offsets[d]:(dim_offsets[d + 1] - 1)
            Int(cell_dims[idx]) == d - 1 || error("$(label): cell_dims mismatch at cell $idx.")
            idx += 1
        end
    end
    return nothing
end

"""
    PointCloud(points)

Minimal point cloud container. `points` is an n-by-d matrix (rows are points),
or a vector of coordinate vectors.
"""
struct PointCloud{T}
    coords::Matrix{T}
    PointCloud{T}(coords::Matrix{T}) where {T} = new{T}(coords)
end

function PointCloud(points::Matrix{T}; copy::Bool=false) where {T}
    return PointCloud{T}(copy ? copy(points) : points)
end

function PointCloud(points::AbstractMatrix{T}) where {T}
    return PointCloud{T}(Matrix{T}(points))
end

PointCloud(points::AbstractVector{<:AbstractVector{T}}) where {T} =
    PointCloud{T}(_rows_to_matrix(T, points))

PointCloud(points::AbstractVector{<:Tuple{Vararg{T}}}) where {T} =
    PointCloud{T}(_rows_to_matrix(T, points))

Base.propertynames(::PointCloud, private::Bool=false) =
    private ? (:coords, :points) : (:coords, :points)

function Base.getproperty(data::PointCloud, s::Symbol)
    s === :points && return MatrixRowsView(getfield(data, :coords))
    return getfield(data, s)
end

"""
    ImageNd(data)

Minimal N-dim image/scalar field container. `data` is an N-dim array.
"""
struct ImageNd{T,N}
    data::Array{T,N}
    ImageNd{T,N}(data::Array{T,N}) where {T,N} = new{T,N}(data)
end

ImageNd(data::Array{T,N}) where {T,N} = ImageNd{T,N}(data)

"""
    GraphData(n, edges; coords=nothing, weights=nothing)

Minimal graph container. `edges` is a vector of (u,v) pairs (1-based).
Optional `coords` can store embeddings, and `weights` can store edge weights.
"""
struct GraphData{T}
    n::Int
    edge_u::Vector{Int}
    edge_v::Vector{Int}
    coord_matrix::Union{Nothing, Matrix{T}}
    weights::Union{Nothing, Vector{T}}
end

function GraphData(n::Integer, edges::AbstractVector{<:Tuple{Int,Int}};
                   coords=nothing,
                   weights::Union{Nothing, AbstractVector}=nothing,
                   T::Type=Float64,
                   copy::Bool=false)
    edge_u, edge_v = _split_edges(edges)
    coords_mat = if coords === nothing
        nothing
    elseif coords isa AbstractMatrix
        (!copy && coords isa Matrix{T}) ? coords : Matrix{T}(coords)
    elseif coords isa AbstractVector{<:AbstractVector}
        _rows_to_matrix(T, coords)
    elseif coords isa AbstractVector{<:Tuple}
        _rows_to_matrix(T, coords)
    else
        error("GraphData: unsupported coords container $(typeof(coords)).")
    end
    coords_mat === nothing || size(coords_mat, 1) == Int(n) || error("GraphData: coords row count must equal n.")
    weights_vec = weights === nothing ? nothing : (copy || !(weights isa Vector{T}) ? T[weights...] : weights)
    weights_vec === nothing || length(weights_vec) == length(edge_u) || error("GraphData: weights length must equal edge count.")
    return GraphData{T}(Int(n), edge_u, edge_v, coords_mat, weights_vec)
end

function GraphData(n::Integer, edge_u::AbstractVector{<:Integer}, edge_v::AbstractVector{<:Integer};
                   coords::Union{Nothing, AbstractMatrix}=nothing,
                   weights::Union{Nothing, AbstractVector}=nothing,
                   T::Type=Float64,
                   copy::Bool=false)
    length(edge_u) == length(edge_v) || error("GraphData: edge column lengths must match.")
    u = (copy || !(edge_u isa Vector{Int})) ? Int[edge_u...] : edge_u
    v = (copy || !(edge_v isa Vector{Int})) ? Int[edge_v...] : edge_v
    coords_mat = if coords === nothing
        nothing
    elseif !copy && coords isa Matrix{T}
        coords
    else
        Matrix{T}(coords)
    end
    coords_mat === nothing || size(coords_mat, 1) == Int(n) || error("GraphData: coords row count must equal n.")
    weights_vec = weights === nothing ? nothing : (copy || !(weights isa Vector{T}) ? T[weights...] : weights)
    weights_vec === nothing || length(weights_vec) == length(u) || error("GraphData: weights length must equal edge count.")
    return GraphData{T}(Int(n), u, v, coords_mat, weights_vec)
end

Base.propertynames(::GraphData, private::Bool=false) =
    private ? (:n, :edges, :coords, :weights, :edge_u, :edge_v, :coord_matrix) :
              (:n, :edges, :coords, :weights)

function Base.getproperty(data::GraphData, s::Symbol)
    if s === :edges
        return EdgePairsView(getfield(data, :edge_u), getfield(data, :edge_v))
    elseif s === :coords
        coords = getfield(data, :coord_matrix)
        return coords === nothing ? nothing : MatrixRowsView(coords)
    end
    return getfield(data, s)
end

"""
    EmbeddedPlanarGraph2D(vertices, edges; polylines=nothing, bbox=nothing)

Embedded planar graph container for 2D applications.
"""
struct EmbeddedPlanarGraph2D{T}
    vertex_matrix::Matrix{T}
    edge_u::Vector{Int}
    edge_v::Vector{Int}
    polyline_offsets::Union{Nothing, Vector{Int}}
    polyline_points::Union{Nothing, Matrix{T}}
    bbox::Union{Nothing, NTuple{4,T}}
end

function EmbeddedPlanarGraph2D(vertices::AbstractVector{<:AbstractVector{T}},
                               edges::AbstractVector{<:Tuple{Int,Int}};
                               polylines::Union{Nothing, AbstractVector}=nothing,
                               bbox::Union{Nothing, NTuple{4,T}}=nothing) where {T}
    verts = _rows_to_matrix(T, vertices)
    edge_u, edge_v = _split_edges(edges)
    offsets, points = polylines === nothing ? (nothing, nothing) : _flatten_polylines(T, polylines)
    return EmbeddedPlanarGraph2D{T}(verts, edge_u, edge_v, offsets, points, bbox)
end

function EmbeddedPlanarGraph2D(vertices::Matrix{T},
                               edges::AbstractVector{<:Tuple{Int,Int}};
                               polylines::Union{Nothing, AbstractVector}=nothing,
                               bbox::Union{Nothing, NTuple{4,T}}=nothing,
                               copy::Bool=false) where {T}
    edge_u, edge_v = _split_edges(edges)
    offsets, points = polylines === nothing ? (nothing, nothing) : _flatten_polylines(T, polylines)
    verts = copy ? copy(vertices) : vertices
    return EmbeddedPlanarGraph2D{T}(verts, edge_u, edge_v, offsets, points, bbox)
end

function EmbeddedPlanarGraph2D(vertices::Matrix{T},
                               edge_u::AbstractVector{<:Integer},
                               edge_v::AbstractVector{<:Integer};
                               polyline_offsets::Union{Nothing, AbstractVector{<:Integer}}=nothing,
                               polyline_points::Union{Nothing, AbstractMatrix}=nothing,
                               bbox::Union{Nothing, NTuple{4,T}}=nothing,
                               copy::Bool=false) where {T}
    length(edge_u) == length(edge_v) || error("EmbeddedPlanarGraph2D: edge column lengths must match.")
    u = (copy || !(edge_u isa Vector{Int})) ? Int[edge_u...] : edge_u
    v = (copy || !(edge_v isa Vector{Int})) ? Int[edge_v...] : edge_v
    offs = polyline_offsets === nothing ? nothing :
        ((copy || !(polyline_offsets isa Vector{Int})) ? Int[polyline_offsets...] : polyline_offsets)
    pts = if polyline_points === nothing
        nothing
    elseif !copy && polyline_points isa Matrix{T}
        polyline_points
    else
        Matrix{T}(polyline_points)
    end
    verts = copy ? copy(vertices) : vertices
    offs === nothing || pts !== nothing || error("EmbeddedPlanarGraph2D: polyline_offsets requires polyline_points.")
    pts === nothing || offs !== nothing || error("EmbeddedPlanarGraph2D: polyline_points requires polyline_offsets.")
    if offs !== nothing
        length(offs) >= 1 || error("EmbeddedPlanarGraph2D: polyline_offsets cannot be empty.")
        first(offs) == 1 || error("EmbeddedPlanarGraph2D: polyline_offsets must start at 1.")
        last(offs) == size(pts, 1) + 1 || error("EmbeddedPlanarGraph2D: polyline_offsets terminator mismatch.")
    end
    return EmbeddedPlanarGraph2D{T}(verts, u, v, offs, pts, bbox)
end

Base.propertynames(::EmbeddedPlanarGraph2D, private::Bool=false) =
    private ? (:vertices, :edges, :polylines, :bbox, :vertex_matrix, :edge_u, :edge_v, :polyline_offsets, :polyline_points) :
              (:vertices, :edges, :polylines, :bbox)

function Base.getproperty(data::EmbeddedPlanarGraph2D, s::Symbol)
    if s === :vertices
        return MatrixRowsView(getfield(data, :vertex_matrix))
    elseif s === :edges
        return EdgePairsView(getfield(data, :edge_u), getfield(data, :edge_v))
    elseif s === :polylines
        offsets = getfield(data, :polyline_offsets)
        points = getfield(data, :polyline_points)
        return offsets === nothing ? nothing : PolylinesView(offsets, points)
    end
    return getfield(data, s)
end

@inline point_matrix(data::PointCloud) = getfield(data, :coords)
@inline coord_matrix(data::GraphData) = getfield(data, :coord_matrix)
@inline edge_columns(data::Union{GraphData,EmbeddedPlanarGraph2D}) =
    (getfield(data, :edge_u), getfield(data, :edge_v))
@inline vertex_matrix(data::EmbeddedPlanarGraph2D) = getfield(data, :vertex_matrix)
@inline polyline_storage(data::EmbeddedPlanarGraph2D) =
    (getfield(data, :polyline_offsets), getfield(data, :polyline_points))

"""
    npoints(pc::PointCloud) -> Int

Number of points in the point cloud.
"""
@inline npoints(data::PointCloud) = size(point_matrix(data), 1)

"""
    ambient_dim(x) -> Int or nothing

Ambient geometric dimension of a data container when that notion is available.

- `PointCloud` returns the column dimension of its coordinate matrix.
- `GraphData` returns the embedding dimension when coordinates are present, and
  `nothing` otherwise.
- `EmbeddedPlanarGraph2D` returns `2`.
"""
@inline ambient_dim(data::PointCloud) = size(point_matrix(data), 2)
@inline ambient_dim(data::GraphData) = let coords = coord_matrix(data)
    coords === nothing ? nothing : size(coords, 2)
end
@inline ambient_dim(::EmbeddedPlanarGraph2D) = 2

"""
    nvertices(data) -> Int

Number of graph vertices in a graph-like data container.

Use this instead of reaching into storage fields such as `n` or `vertex_matrix`
when you want the vertex count as a mathematical quantity.
"""
@inline nvertices(data::GraphData) = getfield(data, :n)
@inline nvertices(data::EmbeddedPlanarGraph2D) = size(vertex_matrix(data), 1)

"""
    nedges(data) -> Int

Number of graph edges in a graph-like data container.
"""
@inline nedges(data::Union{GraphData,EmbeddedPlanarGraph2D}) = length(getfield(data, :edge_u))

"""
    edge_list(data) -> AbstractVector{Tuple{Int,Int}}

Return the graph edge set of a graph-like data container as an iterable
collection of `(u,v)` pairs.

This is the canonical semantic accessor for graph connectivity data. Prefer it
over reading raw storage like `edge_u` / `edge_v`.
"""
@inline edge_list(data::Union{GraphData,EmbeddedPlanarGraph2D}) =
    EdgePairsView(getfield(data, :edge_u), getfield(data, :edge_v))

"""
    vertex_positions(data::EmbeddedPlanarGraph2D) -> AbstractVector

Return the planar vertex coordinates of an embedded planar graph.

The return value behaves like a vector of 2-vectors, one per vertex, and is the
canonical semantic accessor for geometric vertex positions.
"""
@inline vertex_positions(data::EmbeddedPlanarGraph2D) = MatrixRowsView(vertex_matrix(data))

"""
    vertex_coordinates(data) -> AbstractVector or nothing

Return the geometric vertex coordinates attached to a graph-like container when
that notion is available.

- For [`GraphData`](@ref), this returns the embedded vertex coordinates when the
  graph was constructed with `coords=...`, and `nothing` otherwise.
- For [`EmbeddedPlanarGraph2D`](@ref), this returns the planar vertex
  coordinates and is equivalent to [`vertex_positions`](@ref).
"""
@inline vertex_coordinates(data::GraphData) = let coords = coord_matrix(data)
    coords === nothing ? nothing : MatrixRowsView(coords)
end
@inline vertex_coordinates(data::EmbeddedPlanarGraph2D) = vertex_positions(data)

"""
    edge_weights(data) -> AbstractVector or nothing

Return the edge weights attached to a graph container, when present.

For [`GraphData`](@ref), this returns the stored edge-weight vector or `nothing`
if the graph is unweighted.
"""
@inline edge_weights(data::GraphData) = getfield(data, :weights)

"""
    polylines(data::EmbeddedPlanarGraph2D) -> AbstractVector or nothing

Return the embedded polyline geometry attached to an embedded planar graph.

The return value behaves like a vector of polylines, where each polyline is
itself a vector-like sequence of planar points. Returns `nothing` when the
graph was built without polyline geometry.
"""
@inline function polylines(data::EmbeddedPlanarGraph2D)
    offsets, points = polyline_storage(data)
    return offsets === nothing ? nothing : PolylinesView(offsets, points)
end

"""
    bounding_box(data::EmbeddedPlanarGraph2D) -> NTuple or nothing

Return the stored bounding box of an embedded planar graph, when present.

Use this accessor instead of reading the raw `bbox` field directly.
"""
@inline bounding_box(data::EmbeddedPlanarGraph2D) = getfield(data, :bbox)

@inline function _datatype_describe(data::PointCloud)
    return (kind=:point_cloud,
            npoints=npoints(data),
            ambient_dim=ambient_dim(data),
            eltype=eltype(point_matrix(data)))
end

@inline function _datatype_describe(data::GraphData)
    return (kind=:graph_data,
            nvertices=nvertices(data),
            nedges=nedges(data),
            ambient_dim=ambient_dim(data),
            weighted=getfield(data, :weights) !== nothing,
            embedded=coord_matrix(data) !== nothing)
end

@inline function _datatype_describe(data::EmbeddedPlanarGraph2D)
    offs, _ = polyline_storage(data)
    return (kind=:embedded_planar_graph_2d,
            nvertices=nvertices(data),
            nedges=nedges(data),
            npolylines=offs === nothing ? 0 : length(offs) - 1,
            has_bbox=getfield(data, :bbox) !== nothing)
end

function Base.show(io::IO, data::PointCloud)
    d = _datatype_describe(data)
    print(io, "PointCloud(npoints=", d.npoints, ", ambient_dim=", d.ambient_dim, ")")
end
function Base.show(io::IO, ::MIME"text/plain", data::PointCloud)
    d = _datatype_describe(data)
    print(io, "PointCloud\n  npoints: ", d.npoints,
          "\n  ambient_dim: ", d.ambient_dim,
          "\n  eltype: ", d.eltype)
end

function Base.show(io::IO, data::GraphData)
    d = _datatype_describe(data)
    print(io, "GraphData(nvertices=", d.nvertices, ", nedges=", d.nedges, ")")
end
function Base.show(io::IO, ::MIME"text/plain", data::GraphData)
    d = _datatype_describe(data)
    print(io, "GraphData\n  nvertices: ", d.nvertices,
          "\n  nedges: ", d.nedges,
          "\n  ambient_dim: ", repr(d.ambient_dim),
          "\n  weighted: ", d.weighted,
          "\n  embedded: ", d.embedded)
end

function Base.show(io::IO, data::EmbeddedPlanarGraph2D)
    d = _datatype_describe(data)
    print(io, "EmbeddedPlanarGraph2D(nvertices=", d.nvertices,
          ", nedges=", d.nedges, ", npolylines=", d.npolylines, ")")
end
function Base.show(io::IO, ::MIME"text/plain", data::EmbeddedPlanarGraph2D)
    d = _datatype_describe(data)
    print(io, "EmbeddedPlanarGraph2D\n  nvertices: ", d.nvertices,
          "\n  nedges: ", d.nedges,
          "\n  ambient_dim: 2",
          "\n  npolylines: ", d.npolylines,
          "\n  has_bbox: ", d.has_bbox)
end

"""
    check_point_cloud(data::PointCloud; throw=false) -> NamedTuple

Validate a hand-built point cloud container.

Wrap the returned report with [`data_validation_summary`](@ref) when you want a
readable notebook or REPL summary.

This validator currently checks:
- the coordinate storage is a 2-dimensional matrix,
- the reported point count and ambient dimension are well-defined.
"""
function check_point_cloud(data::PointCloud; throw::Bool=false)
    issues = String[]
    A = point_matrix(data)
    ndims(A) == 2 || push!(issues, "coords must be a 2-dimensional matrix")
    valid = isempty(issues)
    throw && !valid && _throw_invalid_data("PointCloud", issues)
    return _validation_report(:point_cloud, valid;
                              npoints=npoints(data),
                              ambient_dim=ambient_dim(data),
                              issues=issues)
end

"""
    check_graph_data(data::GraphData; throw=false) -> NamedTuple

Validate a hand-built graph container.

Wrap the returned report with [`data_validation_summary`](@ref) when you want a
readable notebook or REPL summary.

This validator currently checks:
- `edge_u` and `edge_v` have the same length,
- every edge endpoint lies in `1:nvertices`,
- embedded coordinate storage, when present, has one row per vertex,
- edge weights, when present, have one entry per edge.
"""
function check_graph_data(data::GraphData; throw::Bool=false)
    issues = String[]
    n = nvertices(data)
    u, v = edge_columns(data)
    length(u) == length(v) || push!(issues, "edge_u and edge_v must have the same length")
    @inbounds for i in eachindex(u)
        1 <= u[i] <= n || push!(issues, "edge_u[$i]=$(u[i]) is outside 1:$n")
        1 <= v[i] <= n || push!(issues, "edge_v[$i]=$(v[i]) is outside 1:$n")
    end
    coords = coord_matrix(data)
    coords === nothing || size(coords, 1) == n || push!(issues, "coordinate row count must equal nvertices")
    weights = getfield(data, :weights)
    weights === nothing || length(weights) == length(u) || push!(issues, "weights length must equal edge count")
    valid = isempty(issues)
    throw && !valid && _throw_invalid_data("GraphData", issues)
    return _validation_report(:graph_data, valid;
                              nvertices=n,
                              nedges=nedges(data),
                              ambient_dim=ambient_dim(data),
                              issues=issues)
end

"""
    check_embedded_planar_graph(data::EmbeddedPlanarGraph2D; throw=false) -> NamedTuple

Validate a hand-built embedded planar graph container.

Wrap the returned report with [`data_validation_summary`](@ref) when you want a
readable notebook or REPL summary.

This validator currently checks:
- the vertex matrix has exactly two columns,
- every edge endpoint lies in `1:nvertices`,
- polyline offsets/points are either both present or both absent,
- polyline offsets start at `1` and have a consistent terminator when present.
"""
function check_embedded_planar_graph(data::EmbeddedPlanarGraph2D; throw::Bool=false)
    issues = String[]
    verts = vertex_matrix(data)
    size(verts, 2) == 2 || push!(issues, "vertex matrix must have exactly 2 columns")
    n = nvertices(data)
    u, v = edge_columns(data)
    @inbounds for i in eachindex(u)
        1 <= u[i] <= n || push!(issues, "edge_u[$i]=$(u[i]) is outside 1:$n")
        1 <= v[i] <= n || push!(issues, "edge_v[$i]=$(v[i]) is outside 1:$n")
    end
    offs, pts = polyline_storage(data)
    if offs === nothing
        pts === nothing || push!(issues, "polyline_points requires polyline_offsets")
    else
        pts === nothing && push!(issues, "polyline_offsets requires polyline_points")
        first(offs) == 1 || push!(issues, "polyline_offsets must start at 1")
        pts === nothing || last(offs) == size(pts, 1) + 1 ||
            push!(issues, "polyline_offsets terminator mismatch")
    end
    valid = isempty(issues)
    throw && !valid && _throw_invalid_data("EmbeddedPlanarGraph2D", issues)
    return _validation_report(:embedded_planar_graph_2d, valid;
                              nvertices=n,
                              nedges=nedges(data),
                              issues=issues)
end

"""
    GradedComplex(cells_by_dim, boundaries, grades; cell_dims=nothing)

Generic graded cell complex container.
"""
struct GradedComplex{N,T}
    cell_ids::Vector{Int}
    dim_offsets::Vector{Int}
    boundaries::Vector{SparseMatrixCSC{Int,Int}}
    grades::Vector{NTuple{N,T}}
end

Base.propertynames(::GradedComplex, private::Bool=false) =
    private ? (:cells_by_dim, :cell_dims, :cell_ids, :dim_offsets, :boundaries, :grades) :
              (:cells_by_dim, :cell_dims, :boundaries, :grades)

function Base.getproperty(data::GradedComplex, s::Symbol)
    if s === :cells_by_dim
        return SegmentedSlicesView(getfield(data, :cell_ids), getfield(data, :dim_offsets))
    elseif s === :cell_dims
        return RepeatedDimsView(getfield(data, :dim_offsets))
    end
    return getfield(data, s)
end

function GradedComplex(cells_by_dim::AbstractVector{<:AbstractVector{<:Integer}},
                       boundaries::Vector{SparseMatrixCSC{Int,Int}},
                       grades::AbstractVector{<:AbstractVector{T}};
                       cell_dims::Union{Nothing,AbstractVector{<:Integer}}=nothing) where {T}
    cell_ids, dim_offsets = _pack_cells(cells_by_dim)
    total = last(dim_offsets) - 1
    length(grades) == total || error("GradedComplex: grades length mismatch.")
    cell_dims === nothing || _validate_cell_dims(cell_dims, dim_offsets, "GradedComplex")
    N = length(grades[1])
    ng = Vector{NTuple{N,T}}(undef, length(grades))
    for i in eachindex(grades)
        length(grades[i]) == N || error("GradedComplex: grade $i has wrong length.")
        ng[i] = ntuple(j -> T(grades[i][j]), N)
    end
    return GradedComplex{N,T}(cell_ids, dim_offsets, boundaries, ng)
end

function GradedComplex(cells_by_dim::AbstractVector{<:AbstractVector{<:Integer}},
                       boundaries::Vector{SparseMatrixCSC{Int,Int}},
                       grades::AbstractVector{<:Tuple};
                       cell_dims::Union{Nothing,AbstractVector{<:Integer}}=nothing)
    cell_ids, dim_offsets = _pack_cells(cells_by_dim)
    total = last(dim_offsets) - 1
    length(grades) == total || error("GradedComplex: grades length mismatch.")
    cell_dims === nothing || _validate_cell_dims(cell_dims, dim_offsets, "GradedComplex")
    N = length(grades[1])
    T = eltype(grades[1])
    ng = Vector{NTuple{N,T}}(undef, length(grades))
    for i in eachindex(grades)
        length(grades[i]) == N || error("GradedComplex: grade $i has wrong length.")
        ng[i] = ntuple(j -> T(grades[i][j]), N)
    end
    return GradedComplex{N,T}(cell_ids, dim_offsets, boundaries, ng)
end

"""
    MultiCriticalGradedComplex(cells_by_dim, boundaries, grades; cell_dims=nothing)

Graded cell complex where each cell can carry multiple minimal grades.
"""
struct MultiCriticalGradedComplex{N,T}
    cell_ids::Vector{Int}
    dim_offsets::Vector{Int}
    boundaries::Vector{SparseMatrixCSC{Int,Int}}
    grade_offsets::Vector{Int}
    grade_data::Vector{NTuple{N,T}}
end

Base.propertynames(::MultiCriticalGradedComplex, private::Bool=false) =
    private ? (:cells_by_dim, :cell_dims, :grades, :cell_ids, :dim_offsets, :boundaries, :grade_offsets, :grade_data) :
              (:cells_by_dim, :cell_dims, :boundaries, :grades)

function Base.getproperty(data::MultiCriticalGradedComplex, s::Symbol)
    if s === :cells_by_dim
        return SegmentedSlicesView(getfield(data, :cell_ids), getfield(data, :dim_offsets))
    elseif s === :cell_dims
        return RepeatedDimsView(getfield(data, :dim_offsets))
    elseif s === :grades
        return SegmentedSlicesView(getfield(data, :grade_data), getfield(data, :grade_offsets))
    end
    return getfield(data, s)
end

function MultiCriticalGradedComplex(cells_by_dim::AbstractVector{<:AbstractVector{<:Integer}},
                                    boundaries::Vector{SparseMatrixCSC{Int,Int}},
                                    grades::AbstractVector{<:AbstractVector{<:Tuple}};
                                    cell_dims::Union{Nothing,AbstractVector{<:Integer}}=nothing)
    cell_ids, dim_offsets = _pack_cells(cells_by_dim)
    total = last(dim_offsets) - 1
    length(grades) == total || error("MultiCriticalGradedComplex: grades length mismatch.")
    cell_dims === nothing || _validate_cell_dims(cell_dims, dim_offsets, "MultiCriticalGradedComplex")

    first_cell = findfirst(!isempty, grades)
    first_cell === nothing && error("MultiCriticalGradedComplex: each cell must have at least one grade.")
    first_grade = grades[first_cell][1]
    N = length(first_grade)
    T = eltype(first_grade)

    total_grades = sum(length, grades)
    grade_offsets = Vector{Int}(undef, length(grades) + 1)
    grade_offsets[1] = 1
    grade_data = Vector{NTuple{N,T}}(undef, total_grades)
    t = 1
    for i in eachindex(grades)
        gi = grades[i]
        isempty(gi) && error("MultiCriticalGradedComplex: cell $i has empty grade set.")
        for j in eachindex(gi)
            g = gi[j]
            length(g) == N || error("MultiCriticalGradedComplex: grade length mismatch at cell $i.")
            grade = ntuple(k -> T(g[k]), N)
            @inbounds for prev in grade_offsets[i]:(t - 1)
                grade_data[prev] == grade && error("MultiCriticalGradedComplex: duplicate grade at cell $i.")
            end
            grade_data[t] = grade
            t += 1
        end
        grade_offsets[i + 1] = t
    end
    return MultiCriticalGradedComplex{N,T}(cell_ids, dim_offsets, boundaries, grade_offsets, grade_data)
end

function MultiCriticalGradedComplex(cells_by_dim::AbstractVector{<:AbstractVector{<:Integer}},
                                    boundaries::Vector{SparseMatrixCSC{Int,Int}},
                                    grades::AbstractVector{<:AbstractVector{<:AbstractVector{T}}};
                                    cell_dims::Union{Nothing,AbstractVector{<:Integer}}=nothing) where {T}
    cell_ids, dim_offsets = _pack_cells(cells_by_dim)
    total = last(dim_offsets) - 1
    length(grades) == total || error("MultiCriticalGradedComplex: grades length mismatch.")
    cell_dims === nothing || _validate_cell_dims(cell_dims, dim_offsets, "MultiCriticalGradedComplex")

    first_cell = findfirst(!isempty, grades)
    first_cell === nothing && error("MultiCriticalGradedComplex: each cell must have at least one grade.")
    N = length(grades[first_cell][1])
    total_grades = sum(length, grades)
    grade_offsets = Vector{Int}(undef, length(grades) + 1)
    grade_offsets[1] = 1
    grade_data = Vector{NTuple{N,T}}(undef, total_grades)
    t = 1
    for i in eachindex(grades)
        gi = grades[i]
        isempty(gi) && error("MultiCriticalGradedComplex: cell $i has empty grade set.")
        for j in eachindex(gi)
            g = gi[j]
            length(g) == N || error("MultiCriticalGradedComplex: grade length mismatch at cell $i.")
            grade = ntuple(k -> T(g[k]), N)
            @inbounds for prev in grade_offsets[i]:(t - 1)
                grade_data[prev] == grade && error("MultiCriticalGradedComplex: duplicate grade at cell $i.")
            end
            grade_data[t] = grade
            t += 1
        end
        grade_offsets[i + 1] = t
    end
    return MultiCriticalGradedComplex{N,T}(cell_ids, dim_offsets, boundaries, grade_offsets, grade_data)
end

@inline _packed_cell_storage(data::Union{GradedComplex,MultiCriticalGradedComplex}) =
    (getfield(data, :cell_ids), getfield(data, :dim_offsets))

@inline _packed_multigrade_storage(data::MultiCriticalGradedComplex) =
    (getfield(data, :grade_offsets), getfield(data, :grade_data))

@inline _packed_dim_count(dim_offsets::AbstractVector{Int}, slot::Int) =
    dim_offsets[slot + 1] - dim_offsets[slot]

@inline function _packed_dim_counts(dim_offsets::AbstractVector{Int})
    nd = length(dim_offsets) - 1
    counts = Vector{Int}(undef, nd)
    @inbounds for slot in 1:nd
        counts[slot] = _packed_dim_count(dim_offsets, slot)
    end
    return counts
end

@inline _packed_total_cells(dim_offsets::AbstractVector{Int}) = last(dim_offsets) - 1

@inline function _rebuild_graded_complex(data::Union{GradedComplex,MultiCriticalGradedComplex},
                                         grades::Vector{NTuple{N,T}}) where {N,T}
    return GradedComplex{N,T}(getfield(data, :cell_ids), getfield(data, :dim_offsets),
                              data.boundaries, grades)
end

@inline function _rebuild_multicritical_complex(data::MultiCriticalGradedComplex,
                                                grade_offsets::Vector{Int},
                                                grade_data::Vector{NTuple{N,T}}) where {N,T}
    return MultiCriticalGradedComplex{N,T}(getfield(data, :cell_ids), getfield(data, :dim_offsets),
                                           data.boundaries, grade_offsets, grade_data)
end

"""
    cell_grades(data)

Return the grading data attached to each cell of a graded complex-like
container.

- For [`GradedComplex`](@ref), this is a vector of one grade tuple per cell.
- For [`MultiCriticalGradedComplex`](@ref), this is a grouped view giving the
  minimal grade set attached to each cell.

Prefer this accessor over direct field inspection (`grades`, `grade_data`,
`grade_offsets`) when writing user-facing code.
"""
@inline cell_grades(data::GradedComplex) = data.grades
@inline cell_grades(data::MultiCriticalGradedComplex) = data.grades

"""
    boundary_maps(data)

Return the boundary maps of a graded complex-like container.

The result is the vector of boundary matrices ordered by topological degree.
Use this accessor instead of reading the `boundaries` field directly in
user-facing code and examples.
"""
@inline boundary_maps(data::Union{GradedComplex,MultiCriticalGradedComplex}) = data.boundaries

"""
    cells(data; dim=nothing)

Return the cells of a graded complex-like container.

- With `dim=nothing`, this returns the grouped cells-by-dimension view.
- With `dim=d`, this returns the cells in topological dimension `d`.

Dimensions are zero-based in the mathematical sense: `dim=0` returns
0-cells/vertices, `dim=1` returns 1-cells/edges, and so on.
"""
@inline function cells(data::Union{GradedComplex,MultiCriticalGradedComplex}; dim=nothing)
    grouped = data.cells_by_dim
    dim === nothing && return grouped
    d = Int(dim)
    0 <= d <= max_dim(data) || throw(BoundsError(grouped, d))
    return grouped[d + 1]
end

"""
    cell_grade(data::GradedComplex, i) -> NTuple

Return the grade attached to cell `i` of a singly graded complex.

This is the direct single-cell accessor corresponding to [`cell_grades`](@ref).
"""
@inline function cell_grade(data::GradedComplex, i::Integer)
    ii = Int(i)
    1 <= ii <= length(data.grades) || throw(BoundsError(data.grades, ii))
    return data.grades[ii]
end

"""
    cell_grade_set(data, i)

Return the full grade set attached to cell `i`.

- For [`GradedComplex`](@ref), this is a singleton tuple containing the unique
  grade of the cell.
- For [`MultiCriticalGradedComplex`](@ref), this is the minimal grade set of
  the cell.
"""
@inline function cell_grade_set(data::GradedComplex, i::Integer)
    return (cell_grade(data, i),)
end
@inline function cell_grade_set(data::MultiCriticalGradedComplex, i::Integer)
    ii = Int(i)
    grouped = data.grades
    1 <= ii <= length(grouped) || throw(BoundsError(grouped, ii))
    return grouped[ii]
end

"""
    boundary_map(data; dim=d) -> SparseMatrixCSC

Return the boundary map `del_d : C_d -> C_{d-1}` of a graded complex-like
container.

Use this when you want one specific boundary map by mathematical degree rather
than the whole vector returned by [`boundary_maps`](@ref).
"""
@inline function boundary_map(data::Union{GradedComplex,MultiCriticalGradedComplex}; dim)
    d = Int(dim)
    1 <= d <= max_dim(data) || throw(BoundsError(data.boundaries, d))
    return data.boundaries[d]
end

"""
    max_dim(data) -> Int

Topological dimension of a graded complex-like container.
"""
@inline max_dim(data::Union{GradedComplex,MultiCriticalGradedComplex}) =
    length(getfield(data, :dim_offsets)) - 2

"""
    cell_counts(data) -> Vector{Int}

Cell counts by topological dimension.
"""
@inline cell_counts(data::Union{GradedComplex,MultiCriticalGradedComplex}) =
    _packed_dim_counts(getfield(data, :dim_offsets))

"""
    parameter_dim(data) -> Int

Number of grade coordinates carried by each cell/simplex.
"""
@inline parameter_dim(::GradedComplex{N}) where {N} = N
@inline parameter_dim(::MultiCriticalGradedComplex{N}) where {N} = N

"""
    SimplexTreeMulti(simplex_offsets, simplex_vertices, simplex_dims,
                     dim_offsets, grade_offsets, grade_data)

Compact simplicial multifiltration container with packed storage.
"""
struct SimplexTreeMulti{N,T}
    simplex_offsets::Vector{Int}
    simplex_vertices::Vector{Int}
    simplex_dims::Vector{Int}
    dim_offsets::Vector{Int}
    grade_offsets::Vector{Int}
    grade_data::Vector{NTuple{N,T}}
    function SimplexTreeMulti{N,T}(simplex_offsets::Vector{Int},
                                   simplex_vertices::Vector{Int},
                                   simplex_dims::Vector{Int},
                                   dim_offsets::Vector{Int},
                                   grade_offsets::Vector{Int},
                                   grade_data::Vector{NTuple{N,T}}) where {N,T}
        return new{N,T}(simplex_offsets, simplex_vertices, simplex_dims,
                        dim_offsets, grade_offsets, grade_data)
    end
end

function SimplexTreeMulti(simplex_offsets::Vector{Int},
                          simplex_vertices::Vector{Int},
                          simplex_dims::Vector{Int},
                          dim_offsets::Vector{Int},
                          grade_offsets::Vector{Int},
                          grade_data::Vector{NTuple{N,T}}) where {N,T}
    ns = length(simplex_dims)
    length(simplex_offsets) == ns + 1 ||
        error("SimplexTreeMulti: simplex_offsets must have length nsimplices+1.")
    length(grade_offsets) == ns + 1 ||
        error("SimplexTreeMulti: grade_offsets must have length nsimplices+1.")
    !isempty(dim_offsets) || error("SimplexTreeMulti: dim_offsets cannot be empty.")
    first(simplex_offsets) == 1 || error("SimplexTreeMulti: simplex_offsets must start at 1.")
    first(grade_offsets) == 1 || error("SimplexTreeMulti: grade_offsets must start at 1.")
    last(simplex_offsets) == length(simplex_vertices) + 1 ||
        error("SimplexTreeMulti: simplex_offsets terminator mismatch.")
    last(grade_offsets) == length(grade_data) + 1 ||
        error("SimplexTreeMulti: grade_offsets terminator mismatch.")
    last(dim_offsets) == ns + 1 ||
        error("SimplexTreeMulti: dim_offsets terminator mismatch.")
    for i in 1:ns
        simplex_offsets[i] <= simplex_offsets[i + 1] ||
            error("SimplexTreeMulti: simplex_offsets must be nondecreasing.")
        grade_offsets[i] < grade_offsets[i + 1] ||
            error("SimplexTreeMulti: each simplex must have at least one grade.")
    end
    return SimplexTreeMulti{N,T}(simplex_offsets, simplex_vertices, simplex_dims,
                                 dim_offsets, grade_offsets, grade_data)
end

@inline simplex_count(ST::SimplexTreeMulti) = length(ST.simplex_dims)
@inline max_simplex_dim(ST::SimplexTreeMulti) = isempty(ST.simplex_dims) ? -1 : maximum(ST.simplex_dims)
@inline max_dim(ST::SimplexTreeMulti) = max_simplex_dim(ST)
@inline nsimplices(ST::SimplexTreeMulti) = simplex_count(ST)
@inline parameter_dim(::SimplexTreeMulti{N}) where {N} = N
@inline cell_counts(ST::SimplexTreeMulti) =
    [ST.dim_offsets[i + 1] - ST.dim_offsets[i] for i in 1:(length(ST.dim_offsets) - 1)]

"""
    simplex_dimension(ST, i) -> Int

Return the topological dimension of simplex `i`.
"""
@inline function simplex_dimension(ST::SimplexTreeMulti, i::Integer)
    ii = Int(i)
    1 <= ii <= simplex_count(ST) || throw(BoundsError(ST.simplex_dims, ii))
    return ST.simplex_dims[ii]
end

@inline function simplex_vertices(ST::SimplexTreeMulti, i::Integer)
    ii = Int(i)
    1 <= ii <= simplex_count(ST) || throw(BoundsError(ST.simplex_dims, ii))
    lo = ST.simplex_offsets[ii]
    hi = ST.simplex_offsets[ii + 1] - 1
    return @view ST.simplex_vertices[lo:hi]
end

@inline function simplex_grades(ST::SimplexTreeMulti, i::Integer)
    ii = Int(i)
    1 <= ii <= simplex_count(ST) || throw(BoundsError(ST.simplex_dims, ii))
    lo = ST.grade_offsets[ii]
    hi = ST.grade_offsets[ii + 1] - 1
    return @view ST.grade_data[lo:hi]
end

"""
    simplex_grade_set(ST, i)

Return the full grade set attached to simplex `i`.
"""
@inline simplex_grade_set(ST::SimplexTreeMulti, i::Integer) = simplex_grades(ST, i)

"""
    simplex_grades(ST::SimplexTreeMulti)

Return the multifiltration grades attached to each simplex in a packed simplex
tree.

The result is a grouped view indexed by simplex number, where each entry is the
grade set carried by that simplex. Prefer this semantic accessor over direct
inspection of `grade_offsets` / `grade_data`.
"""
@inline simplex_grades(ST::SimplexTreeMulti) =
    SegmentedSlicesView(getfield(ST, :grade_data), getfield(ST, :grade_offsets))

"""
    simplices(ST; dim=nothing)

Return the simplices of a packed simplex tree.

- With `dim=nothing`, this returns a grouped view indexed by simplex number.
- With `dim=d`, this returns a vector of simplices in topological dimension `d`.

This accessor is intended for semantic inspection and examples. For
allocation-sensitive code, prefer lower-level packed traversal.
"""
@inline function simplices(ST::SimplexTreeMulti; dim=nothing)
    grouped = SegmentedSlicesView(getfield(ST, :simplex_vertices), getfield(ST, :simplex_offsets))
    dim === nothing && return grouped
    d = Int(dim)
    0 <= d <= max_dim(ST) || throw(BoundsError(ST.simplex_dims, d))
    lo = ST.dim_offsets[d + 1]
    hi = ST.dim_offsets[d + 2] - 1
    return [simplex_vertices(ST, i) for i in lo:hi]
end

@inline function _datatype_describe(data::GradedComplex)
    return (kind=:graded_complex,
            max_dim=max_dim(data),
            cell_counts=cell_counts(data),
            parameter_dim=parameter_dim(data),
            ncells=last(getfield(data, :dim_offsets)) - 1)
end

@inline function _datatype_describe(data::MultiCriticalGradedComplex)
    return (kind=:multicritical_graded_complex,
            max_dim=max_dim(data),
            cell_counts=cell_counts(data),
            parameter_dim=parameter_dim(data),
            ncells=last(getfield(data, :dim_offsets)) - 1,
            total_grades=last(getfield(data, :grade_offsets)) - 1)
end

@inline function _datatype_describe(ST::SimplexTreeMulti)
    return (kind=:simplex_tree_multi,
            max_dim=max_dim(ST),
            simplex_counts=cell_counts(ST),
            parameter_dim=parameter_dim(ST),
            nsimplices=simplex_count(ST))
end

"""
    data_summary(data) -> NamedTuple

Return a compact semantic summary of a `DataTypes` container.

This is the owner-module inspection entrypoint for point clouds, graphs,
embedded planar graphs, graded complexes, and packed simplex trees. It mirrors
the shared `describe(...)` surface without requiring users to know that the
generic is owned elsewhere.

Best practices
- use `data_summary(...)` when you are already working inside `DataTypes` or
  `TamerOp.Advanced`;
- use the semantic accessors in this module when you need specific content such
  as edges, grades, or boundary maps;
- keep the `check_*` helpers for structural validation of hand-built objects.
"""
data_summary(data::PointCloud) = _datatype_describe(data)
data_summary(data::GraphData) = _datatype_describe(data)
data_summary(data::EmbeddedPlanarGraph2D) = _datatype_describe(data)
data_summary(data::GradedComplex) = _datatype_describe(data)
data_summary(data::MultiCriticalGradedComplex) = _datatype_describe(data)
data_summary(data::SimplexTreeMulti) = _datatype_describe(data)

function Base.show(io::IO, data::GradedComplex)
    d = _datatype_describe(data)
    print(io, "GradedComplex(max_dim=", d.max_dim,
          ", ncells=", d.ncells, ", parameter_dim=", d.parameter_dim, ")")
end
function Base.show(io::IO, ::MIME"text/plain", data::GradedComplex)
    d = _datatype_describe(data)
    print(io, "GradedComplex\n  max_dim: ", d.max_dim,
          "\n  cell_counts: ", repr(d.cell_counts),
          "\n  parameter_dim: ", d.parameter_dim,
          "\n  ncells: ", d.ncells)
end

function Base.show(io::IO, data::MultiCriticalGradedComplex)
    d = _datatype_describe(data)
    print(io, "MultiCriticalGradedComplex(max_dim=", d.max_dim,
          ", ncells=", d.ncells, ", parameter_dim=", d.parameter_dim, ")")
end
function Base.show(io::IO, ::MIME"text/plain", data::MultiCriticalGradedComplex)
    d = _datatype_describe(data)
    print(io, "MultiCriticalGradedComplex\n  max_dim: ", d.max_dim,
          "\n  cell_counts: ", repr(d.cell_counts),
          "\n  parameter_dim: ", d.parameter_dim,
          "\n  ncells: ", d.ncells,
          "\n  total_grades: ", d.total_grades)
end

function Base.show(io::IO, ST::SimplexTreeMulti)
    d = _datatype_describe(ST)
    print(io, "SimplexTreeMulti(max_dim=", d.max_dim,
          ", nsimplices=", d.nsimplices, ", parameter_dim=", d.parameter_dim, ")")
end
function Base.show(io::IO, ::MIME"text/plain", ST::SimplexTreeMulti)
    d = _datatype_describe(ST)
    print(io, "SimplexTreeMulti\n  max_dim: ", d.max_dim,
          "\n  simplex_counts: ", repr(d.simplex_counts),
          "\n  parameter_dim: ", d.parameter_dim,
          "\n  nsimplices: ", d.nsimplices)
end

"""
    check_graded_complex(data::GradedComplex; throw=false) -> NamedTuple

Validate a hand-built graded complex container.

Wrap the returned report with [`data_validation_summary`](@ref) when you want a
readable notebook or REPL summary.

This validator currently checks:
- the number of boundary maps matches `max_dim(data)`,
- each boundary matrix has the expected row/column counts by cell dimension,
- the number of grades matches the total number of cells.
"""
function check_graded_complex(data::GradedComplex; throw::Bool=false)
    issues = String[]
    counts = cell_counts(data)
    bounds = data.boundaries
    expected = max(length(counts) - 1, 0)
    length(bounds) == expected || push!(issues, "expected $(expected) boundary matrices, got $(length(bounds))")
    @inbounds for i in 1:min(length(bounds), expected)
        size(bounds[i], 1) == counts[i] || push!(issues, "boundary[$i] row count must equal cell count in dimension $(i - 1)")
        size(bounds[i], 2) == counts[i + 1] || push!(issues, "boundary[$i] column count must equal cell count in dimension $(i)")
    end
    length(data.grades) == last(getfield(data, :dim_offsets)) - 1 || push!(issues, "grades length must equal total cell count")
    valid = isempty(issues)
    throw && !valid && _throw_invalid_data("GradedComplex", issues)
    return _validation_report(:graded_complex, valid;
                              max_dim=max_dim(data),
                              cell_counts=counts,
                              parameter_dim=parameter_dim(data),
                              issues=issues)
end

"""
    check_multicritical_complex(data::MultiCriticalGradedComplex; throw=false) -> NamedTuple

Validate a hand-built multicritical graded complex container.

Wrap the returned report with [`data_validation_summary`](@ref) when you want a
readable notebook or REPL summary.

This validator currently checks:
- the number of boundary maps matches `max_dim(data)`,
- each boundary matrix has the expected row/column counts by cell dimension,
- `grade_offsets` starts at `1` and has the expected terminator,
- every cell has at least one stored grade.
"""
function check_multicritical_complex(data::MultiCriticalGradedComplex; throw::Bool=false)
    issues = String[]
    counts = cell_counts(data)
    bounds = data.boundaries
    expected = max(length(counts) - 1, 0)
    length(bounds) == expected || push!(issues, "expected $(expected) boundary matrices, got $(length(bounds))")
    @inbounds for i in 1:min(length(bounds), expected)
        size(bounds[i], 1) == counts[i] || push!(issues, "boundary[$i] row count must equal cell count in dimension $(i - 1)")
        size(bounds[i], 2) == counts[i + 1] || push!(issues, "boundary[$i] column count must equal cell count in dimension $(i)")
    end
    offsets = getfield(data, :grade_offsets)
    first(offsets) == 1 || push!(issues, "grade_offsets must start at 1")
    last(offsets) == length(getfield(data, :grade_data)) + 1 || push!(issues, "grade_offsets terminator mismatch")
    @inbounds for i in 1:length(offsets) - 1
        offsets[i] < offsets[i + 1] || push!(issues, "each cell must have at least one grade")
    end
    valid = isempty(issues)
    throw && !valid && _throw_invalid_data("MultiCriticalGradedComplex", issues)
    return _validation_report(:multicritical_graded_complex, valid;
                              max_dim=max_dim(data),
                              cell_counts=counts,
                              parameter_dim=parameter_dim(data),
                              issues=issues)
end

"""
    check_simplex_tree_multi(data::SimplexTreeMulti; throw=false) -> NamedTuple

Validate a hand-built packed simplicial multifiltration container.

Wrap the returned report with [`data_validation_summary`](@ref) when you want a
readable notebook or REPL summary.

This validator currently checks:
- simplex and grade offsets start at `1`,
- simplex and grade offsets have consistent terminators,
- `dim_offsets` has the expected terminator,
- each simplex stores exactly `dim+1` vertices,
- each simplex has at least one grade.
"""
function check_simplex_tree_multi(data::SimplexTreeMulti; throw::Bool=false)
    issues = String[]
    ns = simplex_count(data)
    first(data.simplex_offsets) == 1 || push!(issues, "simplex_offsets must start at 1")
    first(data.grade_offsets) == 1 || push!(issues, "grade_offsets must start at 1")
    last(data.simplex_offsets) == length(data.simplex_vertices) + 1 || push!(issues, "simplex_offsets terminator mismatch")
    last(data.grade_offsets) == length(data.grade_data) + 1 || push!(issues, "grade_offsets terminator mismatch")
    last(data.dim_offsets) == ns + 1 || push!(issues, "dim_offsets terminator mismatch")
    @inbounds for i in 1:ns
        nverts = data.simplex_offsets[i + 1] - data.simplex_offsets[i]
        nverts == data.simplex_dims[i] + 1 || push!(issues, "simplex $i stores $(nverts) vertices but has dim $(data.simplex_dims[i])")
        data.grade_offsets[i] < data.grade_offsets[i + 1] || push!(issues, "simplex $i must have at least one grade")
    end
    valid = isempty(issues)
    throw && !valid && _throw_invalid_data("SimplexTreeMulti", issues)
    return _validation_report(:simplex_tree_multi, valid;
                              max_dim=max_dim(data),
                              simplex_counts=cell_counts(data),
                              parameter_dim=parameter_dim(data),
                              issues=issues)
end

"""
    Base.show(io::IO, summary::DataValidationSummary)

Compact one-line summary for a wrapped data-container validation report.
"""
function Base.show(io::IO, summary::DataValidationSummary)
    report = summary.report
    kind = get(report, :kind, :data_validation)
    valid = get(report, :valid, false)
    issues = get(report, :issues, String[])
    print(io, "DataValidationSummary(kind=", kind,
          ", valid=", valid,
          ", issues=", length(issues), ")")
end

"""
    Base.show(io::IO, ::MIME\"text/plain\", summary::DataValidationSummary)

Verbose multi-line summary for a wrapped data-container validation report.
"""
function Base.show(io::IO, ::MIME"text/plain", summary::DataValidationSummary)
    report = summary.report
    kind = get(report, :kind, :data_validation)
    valid = get(report, :valid, false)
    issues = get(report, :issues, String[])
    println(io, "DataValidationSummary")
    println(io, "  kind = ", kind)
    println(io, "  valid = ", valid)
    for key in (:npoints, :ambient_dim, :nvertices, :nedges, :max_dim, :cell_counts, :simplex_counts, :parameter_dim)
        haskey(report, key) || continue
        println(io, "  ", key, " = ", getfield(report, key))
    end
    if isempty(issues)
        println(io, "  issues = none")
    else
        println(io, "  issues:")
        for msg in issues
            println(io, "    - ", msg)
        end
    end
end

end # module DataTypes
