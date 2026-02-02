module DataPipeline
# =============================================================================
# DataPipeline.jl
#
# Minimal data-ingestion surface (Step 1) and lightweight grid encoding map
# (Step 2). This file defines dataset/filtration structs and a small encoding
# map that plugs into existing invariant utilities via `locate` and
# `axes_from_encoding`.
# =============================================================================

using SparseArrays
using ..CoreModules: AbstractPLikeEncodingMap
import ..CoreModules: locate, dimension, representatives, axes_from_encoding
using ..FiniteFringe: AbstractPoset, FinitePoset, GridPoset, ProductOfChainsPoset, nvertices

export PointCloud, ImageNd, GraphData, EmbeddedPlanarGraph2D, GradedComplex,
       FiltrationSpec, GridEncodingMap, poset_from_axes, grid_index,
       encode_from_data, ingest

# -----------------------------------------------------------------------------
# Step 1: dataset and filtration specs
# -----------------------------------------------------------------------------

"""
    PointCloud(points)

Minimal point cloud container. `points` is an n-by-d matrix (rows are points),
or a vector of coordinate vectors.
"""
struct PointCloud{T}
    points::Vector{Vector{T}}
end

function PointCloud(points::AbstractMatrix{T}) where {T}
    pts = [Vector{T}(points[i, :]) for i in 1:size(points, 1)]
    return PointCloud{T}(pts)
end

PointCloud(points::AbstractVector{<:AbstractVector{T}}) where {T} =
    PointCloud{T}([Vector{T}(p) for p in points])

"""
    ImageNd(data)

Minimal N-dim image/scalar field container. `data` is an N-dim array.
"""
struct ImageNd{T,N}
    data::Array{T,N}
end

ImageNd(data::Array{T,N}) where {T,N} = ImageNd{T,N}(data)

"""
    GraphData(n, edges; coords=nothing, weights=nothing)

Minimal graph container. `edges` is a vector of (u,v) pairs (1-based).
Optional `coords` can store embeddings, and `weights` can store edge weights.
"""
struct GraphData{T}
    n::Int
    edges::Vector{Tuple{Int,Int}}
    coords::Union{Nothing, Vector{Vector{T}}}
    weights::Union{Nothing, Vector{T}}
end

function GraphData(n::Integer, edges::AbstractVector{<:Tuple{Int,Int}};
                   coords::Union{Nothing, AbstractVector{<:AbstractVector}}=nothing,
                   weights::Union{Nothing, AbstractVector}=nothing,
                   T::Type=Float64)
    coords_vec = coords === nothing ? nothing : [Vector{T}(c) for c in coords]
    weights_vec = weights === nothing ? nothing : Vector{T}(weights)
    return GraphData{T}(Int(n), Vector{Tuple{Int,Int}}(edges), coords_vec, weights_vec)
end

"""
    EmbeddedPlanarGraph2D(vertices, edges; polylines=nothing, bbox=nothing)

Embedded planar graph container for 2D applications (e.g. wing veins).
`vertices` is a vector of 2D coordinate vectors, `edges` are vertex index pairs.
`polylines` can store per-edge piecewise-linear geometry.
"""
struct EmbeddedPlanarGraph2D{T}
    vertices::Vector{Vector{T}}
    edges::Vector{Tuple{Int,Int}}
    polylines::Union{Nothing, Vector{Vector{Vector{T}}}}
    bbox::Union{Nothing, NTuple{4,T}}
end

function EmbeddedPlanarGraph2D(vertices::AbstractVector{<:AbstractVector{T}},
                               edges::AbstractVector{<:Tuple{Int,Int}};
                               polylines::Union{Nothing, AbstractVector}=nothing,
                               bbox::Union{Nothing, NTuple{4,T}}=nothing) where {T}
    verts = [Vector{T}(v) for v in vertices]
    polys = polylines === nothing ? nothing : [ [Vector{T}(p) for p in poly] for poly in polylines ]
    return EmbeddedPlanarGraph2D{T}(verts, Vector{Tuple{Int,Int}}(edges), polys, bbox)
end

"""
    GradedComplex(cells_by_dim, boundaries, grades; cell_dims=nothing)

Generic graded cell complex container ("escape hatch").

Fields:
`cells_by_dim`: Vector of cell indices grouped by dimension, e.g. cells_by_dim[d+1]
                is a vector of cell ids in dimension d.
`boundaries`: Vector of sparse boundary matrices between dimensions.
`grades`: Vector of grade vectors (same order as cells concatenated by dimension).
`cell_dims`: Explicit dimension for each cell (same order as `grades`).
"""
struct GradedComplex{N,T}
    cells_by_dim::Vector{Vector{Int}}
    boundaries::Vector{SparseMatrixCSC{Int,Int}}
    grades::Vector{NTuple{N,T}}
    cell_dims::Vector{Int}
end

function _cell_dims_from_cells(cells_by_dim::Vector{Vector{Int}})
    out = Int[]
    for (d, cells) in enumerate(cells_by_dim)
        for _ in cells
            push!(out, d - 1)
        end
    end
    return out
end

function GradedComplex(cells_by_dim::Vector{Vector{Int}},
                       boundaries::Vector{SparseMatrixCSC{Int,Int}},
                       grades::Vector{<:AbstractVector{T}};
                       cell_dims::Union{Nothing,Vector{Int}}=nothing) where {T}
    total = sum(length.(cells_by_dim))
    if cell_dims === nothing
        if length(grades) == total
            cell_dims = _cell_dims_from_cells(cells_by_dim)
        else
            # Keep construction permissive; downstream validators can reject.
            cell_dims = fill(0, length(grades))
        end
    end
    N = length(grades[1])
    ng = Vector{NTuple{N,T}}(undef, length(grades))
    for i in eachindex(grades)
        length(grades[i]) == N || error("GradedComplex: grade $i has wrong length.")
        ng[i] = ntuple(j -> T(grades[i][j]), N)
    end
    return GradedComplex{N,T}(cells_by_dim, boundaries, ng, cell_dims)
end

function GradedComplex(cells_by_dim::Vector{Vector{Int}},
                       boundaries::Vector{SparseMatrixCSC{Int,Int}},
                       grades::Vector{<:Tuple};
                       cell_dims::Union{Nothing,Vector{Int}}=nothing)
    total = sum(length.(cells_by_dim))
    if cell_dims === nothing
        if length(grades) == total
            cell_dims = _cell_dims_from_cells(cells_by_dim)
        else
            # Keep construction permissive; downstream validators can reject.
            cell_dims = fill(0, length(grades))
        end
    end
    N = length(grades[1])
    T = eltype(grades[1])
    ng = Vector{NTuple{N,T}}(undef, length(grades))
    for i in eachindex(grades)
        length(grades[i]) == N || error("GradedComplex: grade $i has wrong length.")
        ng[i] = ntuple(j -> T(grades[i][j]), N)
    end
    return GradedComplex{N,T}(cells_by_dim, boundaries, ng, cell_dims)
end

"""
    FiltrationSpec(; kind, params...)

Lightweight filtration specification container. This will grow as the ingestion
layer is implemented; for now it stores a `kind` symbol and a `params` named tuple.
"""
struct FiltrationSpec
    kind::Symbol
    params::NamedTuple
end

FiltrationSpec(; kind::Symbol, params...) = FiltrationSpec(kind, NamedTuple(params))

# -----------------------------------------------------------------------------
# Step 2: lightweight grid encoding map
# -----------------------------------------------------------------------------

"""
    GridEncodingMap(P, coords; orientation=ntuple(_->1, N))

Axis-aligned grid encoding map for a product-of-chains poset.

`coords` is an N-tuple of sorted coordinate vectors. `orientation[i]` is +1
for sublevel-style axes and -1 for superlevel-style axes (applied by negation
before indexing).
"""
struct GridEncodingMap{N,T,P<:AbstractPoset} <: AbstractPLikeEncodingMap
    P::P
    coords::NTuple{N,Vector{T}}
    orientation::NTuple{N,Int}
    sizes::NTuple{N,Int}
    strides::NTuple{N,Int}
end

function _grid_strides(sizes::NTuple{N,Int}) where {N}
    strides = Vector{Int}(undef, N)
    strides[1] = 1
    for i in 2:N
        strides[i] = strides[i-1] * sizes[i-1]
    end
    return ntuple(i -> strides[i], N)
end

"""
    grid_index(idxs, sizes) -> Int

Convert an N-tuple of 1-based indices into a linear index using mixed radix
ordering with the first axis varying fastest.
"""
function grid_index(idxs::NTuple{N,Int}, sizes::NTuple{N,Int}) where {N}
    strides = _grid_strides(sizes)
    lin = 1
    for i in 1:N
        lin += (idxs[i] - 1) * strides[i]
    end
    return lin
end

"""
    poset_from_axes(axes; orientation=ntuple(_->1, N), kind=:grid) -> AbstractPoset

Build the product-of-chains poset on a grid defined by `axes`.
`axes` is an N-tuple of sorted coordinate vectors. `orientation[i]` is +1 for
sublevel-style order (increasing) and -1 for superlevel-style order (decreasing).
"""
function poset_from_axes(axes::NTuple{N,Vector{T}};
                         orientation::NTuple{N,Int}=ntuple(_ -> 1, N),
                         kind::Symbol = :grid) where {N,T}
    sizes = ntuple(i -> length(axes[i]), N)
    total = 1
    for i in 1:N
        total *= sizes[i]
    end
    for i in 1:N
        o = orientation[i]
        (o == 1 || o == -1) || error("poset_from_axes: orientation[$i] must be +1 or -1.")
    end

    if kind == :grid
        if any(o -> o == -1, orientation)
            return ProductOfChainsPoset(sizes)
        else
            return GridPoset(axes)
        end
    elseif kind == :dense
        # Enumerate all multi-indices in mixed radix order.
        idxs = Vector{NTuple{N,Int}}(undef, total)
        cur = ones(Int, N)
        for lin in 1:total
            idxs[lin] = ntuple(i -> cur[i], N)
            for i in 1:N
                cur[i] += 1
                if cur[i] <= sizes[i]
                    break
                else
                    cur[i] = 1
                end
            end
        end

        leq = falses(total, total)
        for i in 1:total
            ai = idxs[i]
            for j in 1:total
                bj = idxs[j]
                ok = true
                for k in 1:N
                    if orientation[k] == 1
                        if ai[k] > bj[k]
                            ok = false
                            break
                        end
                    else
                        if ai[k] < bj[k]
                            ok = false
                            break
                        end
                    end
                end
                leq[i, j] = ok
            end
        end

        return FinitePoset(leq; check=false)
    else
        error("poset_from_axes: kind must be :grid or :dense")
    end
end

function GridEncodingMap(P::AbstractPoset, coords::NTuple{N,Vector{T}};
                         orientation::NTuple{N,Int}=ntuple(_ -> 1, N)) where {N,T}
    sizes = ntuple(i -> length(coords[i]), N)
    total = 1
    for i in 1:N
        total *= sizes[i]
    end
    total == nvertices(P) || error("GridEncodingMap: grid size $(total) does not match nvertices(P)=$(nvertices(P)).")
    for i in 1:N
        o = orientation[i]
        (o == 1 || o == -1) || error("GridEncodingMap: orientation[$i] must be +1 or -1.")
    end
    return GridEncodingMap{N,T,typeof(P)}(P, coords, orientation, sizes, _grid_strides(sizes))
end

dimension(pi::GridEncodingMap{N}) where {N} = N
axes_from_encoding(pi::GridEncodingMap) = pi.coords

function locate(pi::GridEncodingMap{N,T}, x::AbstractVector{<:Real}) where {N,T}
    length(x) == N || error("GridEncodingMap.locate: expected vector length $(N), got $(length(x)).")
    idxs = Vector{Int}(undef, N)
    for i in 1:N
        xi = pi.orientation[i] == 1 ? x[i] : -x[i]
        idx = searchsortedlast(pi.coords[i], xi)
        if idx < 1
            return 0
        end
        idxs[i] = idx
    end
    lin = 1
    for i in 1:N
        lin += (idxs[i] - 1) * pi.strides[i]
    end
    return lin
end

function representatives(pi::GridEncodingMap{N,T}) where {N,T}
    # Cartesian product of coordinate axes (grid points).
    reps = Vector{Vector{T}}(undef, nvertices(pi.P))
    idxs = ones(Int, N)
    for lin in 1:nvertices(pi.P)
        reps[lin] = [pi.coords[i][idxs[i]] for i in 1:N]
        # advance mixed radix counter
        for i in 1:N
            idxs[i] += 1
            if idxs[i] <= pi.sizes[i]
                break
            else
                idxs[i] = 1
            end
        end
    end
    return reps
end

# -----------------------------------------------------------------------------
# Narrative entrypoints (declared now; implemented later)
# -----------------------------------------------------------------------------

"""
    encode_from_data(data, spec; kwargs...)

Planned high-level ingestion entrypoint. This will turn a dataset + filtration
spec into a finite-encoded fringe module plus encoding map.
"""
function encode_from_data end

"""
    ingest(data, spec; kwargs...)

Alias for `encode_from_data` to support narrative workflows.
"""
function ingest end

end
