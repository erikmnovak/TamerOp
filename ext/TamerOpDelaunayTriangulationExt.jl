module TamerOpDelaunayTriangulationExt

import DelaunayTriangulation as DT

const TO = let pm = nothing
    if isdefined(Main, :TamerOp)
        pm = getfield(Main, :TamerOp)
    else
        @eval import TamerOp
        pm = TamerOp
    end
    pm
end

const DI = TO.DataIngestion

@inline function _points2d(points)
    n = length(points)
    coords = Vector{NTuple{2,Float64}}(undef, n)
    @inbounds for i in 1:n
        p = points[i]
        length(p) >= 2 || throw(ArgumentError("PointCloud point dimension mismatch in Delaunay backend."))
        coords[i] = (Float64(p[1]), Float64(p[2]))
    end
    return coords
end

@inline function _edge_radius_from_coords(coords::Vector{NTuple{2,Float64}}, i::Int, j::Int)
    xi, yi = coords[i]
    xj, yj = coords[j]
    dx = xi - xj
    dy = yi - yj
    return 0.5 * sqrt(dx * dx + dy * dy)
end

@inline function _circumradius2_from_coords(a::NTuple{2,Float64},
                                            b::NTuple{2,Float64},
                                            c::NTuple{2,Float64};
                                            atol::Float64=1e-12)
    ax, ay = a
    bx, by = b
    cx, cy = c
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    abs(d) <= atol && return nothing
    aa = ax * ax + ay * ay
    bb = bx * bx + by * by
    cc = cx * cx + cy * cy
    ux = (aa * (by - cy) + bb * (cy - ay) + cc * (ay - by)) / d
    uy = (aa * (cx - bx) + bb * (ax - cx) + cc * (bx - ax)) / d
    r2 = (ux - ax)^2 + (uy - ay)^2
    return r2 <= 0.0 ? 0.0 : r2
end

function _packed_delaunay_2d(points; max_dim::Int=2)
    n = length(points)
    n == 0 && return DI._PackedDelaunay2D(NTuple{2,Int}[], Float64[], NTuple{3,Int}[], Float64[])
    coords = _points2d(points)
    tri = DT.triangulate(coords)

    edges = NTuple{2,Int}[]
    sizehint!(edges, max(0, 3n))
    @inbounds for e in DT.each_solid_edge(tri)
        i, j = e
        i == j && continue
        a, b = i < j ? (i, j) : (j, i)
        push!(edges, (a, b))
    end
    edge_radius = Float64[]
    sizehint!(edge_radius, length(edges))
    @inbounds for (a, b) in edges
        push!(edge_radius, _edge_radius_from_coords(coords, a, b))
    end

    triangles = NTuple{3,Int}[]
    tri_radius = Float64[]
    if max_dim >= 2
        sizehint!(triangles, max(0, 2n))
        sizehint!(tri_radius, max(0, 2n))
        @inbounds for t in DT.each_solid_triangle(tri)
            i, j, k = t
            (i == j || i == k || j == k) && continue
            a, b, c = DI._sort_triplet(i, j, k)
            r2 = _circumradius2_from_coords(coords[a], coords[b], coords[c]; atol=1e-12)
            r2 === nothing && continue
            push!(triangles, (a, b, c))
            push!(tri_radius, sqrt(r2))
        end
    end

    return DI._PackedDelaunay2D(edges, edge_radius, triangles, tri_radius)
end

DI._set_pointcloud_delaunay_2d_impl!(_packed_delaunay_2d)

end # module
