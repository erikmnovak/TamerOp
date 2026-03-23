module TamerOpNearestNeighborsExt

using NearestNeighbors
using Random

import TamerOp
const TO = TamerOp

const DI = TO.DataIngestion

@inline function _points_to_matrix(points)
    n = length(points)
    n > 0 || return zeros(Float64, 0, 0)
    d = length(points[1])
    X = Matrix{Float64}(undef, d, n)
    @inbounds for j in 1:n
        p = points[j]
        length(p) == d || throw(ArgumentError("PointCloud point dimension mismatch in NN backend."))
        for i in 1:d
            X[i, j] = Float64(p[i])
        end
    end
    return X
end

@inline function _build_exact_tree(X::Matrix{Float64})
    d = size(X, 1)
    return d <= 12 ? KDTree(X) : BallTree(X)
end

function _projected_matrix(X::Matrix{Float64}; outdim::Int)
    d, _ = size(X)
    dproj = max(1, min(outdim, d))
    rng = MersenneTwister(0xC0FFEE + 257 * d + 911 * dproj)
    R = Matrix{Float64}(undef, dproj, d)
    @inbounds for j in 1:d, i in 1:dproj
        R[i, j] = randn(rng)
    end
    Y = R * X
    return Y
end

function _knn_graph(points, k::Int; backend::Symbol=:auto, approx_candidates::Int=0)
    n = length(points)
    n > 0 || return NTuple{2,Int}[], Float64[], Float64[]
    k_eff = min(k, n - 1)
    k_eff > 0 || throw(ArgumentError("kNN k=$(k) exceeds number of neighbors."))

    if backend == :bruteforce
        return nothing
    end

    X = _points_to_matrix(points)
    d = size(X, 1)
    use_approx = backend == :approx || (backend == :auto && d > 20 && n > 1500)
    edge_keys = Vector{UInt64}(undef, n * k_eff)
    edge_vals = Vector{Float64}(undef, n * k_eff)
    edge_count = 0
    kdist = Vector{Float64}(undef, n)
    idxbuf = fill(0, k_eff)
    distbuf = fill(Inf, k_eff)

    if use_approx
        cand = approx_candidates > 0 ? approx_candidates : max(4 * k_eff, k_eff + 8)
        cand = max(k_eff + 1, min(n, cand))
        Y = _projected_matrix(X; outdim=min(16, d))
        tree = _build_exact_tree(Y)
        @inbounds for i in 1:n
            fill!(idxbuf, 0)
            fill!(distbuf, Inf)
            neigh, _ = knn(tree, view(Y, :, i), cand, true)
            for j in neigh
                j == i && continue
                DI._insert_neighbor_sorted!(idxbuf, distbuf, j, DI._euclidean_distance(points[i], points[j]))
            end
            # Guard: if approximate candidates underfilled, finish with exact scan.
            if idxbuf[k_eff] == 0
                for j in 1:n
                    i == j && continue
                    DI._insert_neighbor_sorted!(idxbuf, distbuf, j, DI._euclidean_distance(points[i], points[j]))
                end
            end
            kdist[i] = distbuf[k_eff]
            for t in 1:k_eff
                j = idxbuf[t]
                j == 0 && continue
                u, v = DI._ordered_pair(i, j)
                edge_count += 1
                edge_keys[edge_count] = DI._pack_edge_key(u, v)
                edge_vals[edge_count] = distbuf[t]
            end
        end
    else
        nn = min(n, k_eff + 1)
        tree = _build_exact_tree(X)
        @inbounds for i in 1:n
            fill!(idxbuf, 0)
            fill!(distbuf, Inf)
            neigh, _ = knn(tree, view(X, :, i), nn, true)
            for j in neigh
                j == i && continue
                DI._insert_neighbor_sorted!(idxbuf, distbuf, j, DI._euclidean_distance(points[i], points[j]))
            end
            kdist[i] = distbuf[k_eff]
            for t in 1:k_eff
                j = idxbuf[t]
                j == 0 && continue
                u, v = DI._ordered_pair(i, j)
                edge_count += 1
                edge_keys[edge_count] = DI._pack_edge_key(u, v)
                edge_vals[edge_count] = distbuf[t]
            end
        end
    end

    edges, dists = DI._finalize_edge_pairs(edge_keys, edge_vals, edge_count)
    return edges, dists, kdist
end

function _knn_graph_edges(points, k::Int; backend::Symbol=:auto, approx_candidates::Int=0)
    n = length(points)
    n > 0 || return NTuple{2,Int}[]
    k_eff = min(k, n - 1)
    k_eff > 0 || throw(ArgumentError("kNN k=$(k) exceeds number of neighbors."))

    if backend == :bruteforce
        return nothing
    end

    X = _points_to_matrix(points)
    d = size(X, 1)
    use_approx = backend == :approx || (backend == :auto && d > 20 && n > 1500)
    edge_keys = Vector{UInt64}(undef, n * k_eff)
    edge_count = 0
    idxbuf = fill(0, k_eff)
    distbuf = fill(Inf, k_eff)

    if use_approx
        cand = approx_candidates > 0 ? approx_candidates : max(4 * k_eff, k_eff + 8)
        cand = max(k_eff + 1, min(n, cand))
        Y = _projected_matrix(X; outdim=min(16, d))
        tree = _build_exact_tree(Y)
        @inbounds for i in 1:n
            fill!(idxbuf, 0)
            fill!(distbuf, Inf)
            neigh, _ = knn(tree, view(Y, :, i), cand, true)
            for j in neigh
                j == i && continue
                DI._insert_neighbor_sorted!(idxbuf, distbuf, j, DI._euclidean_distance(points[i], points[j]))
            end
            if idxbuf[k_eff] == 0
                for j in 1:n
                    i == j && continue
                    DI._insert_neighbor_sorted!(idxbuf, distbuf, j, DI._euclidean_distance(points[i], points[j]))
                end
            end
            for t in 1:k_eff
                j = idxbuf[t]
                j == 0 && continue
                u, v = DI._ordered_pair(i, j)
                edge_count += 1
                edge_keys[edge_count] = DI._pack_edge_key(u, v)
            end
        end
    else
        nn = min(n, k_eff + 1)
        tree = _build_exact_tree(X)
        @inbounds for i in 1:n
            fill!(idxbuf, 0)
            fill!(distbuf, Inf)
            neigh, _ = knn(tree, view(X, :, i), nn, true)
            for j in neigh
                j == i && continue
                DI._insert_neighbor_sorted!(idxbuf, distbuf, j, DI._euclidean_distance(points[i], points[j]))
            end
            for t in 1:k_eff
                j = idxbuf[t]
                j == 0 && continue
                u, v = DI._ordered_pair(i, j)
                edge_count += 1
                edge_keys[edge_count] = DI._pack_edge_key(u, v)
            end
        end
    end

    return DI._finalize_edge_pairs_edges_only(edge_keys, edge_count)
end

function _radius_graph(points, r::Float64; backend::Symbol=:auto, approx_candidates::Int=0)
    if backend == :bruteforce || backend == :approx
        return nothing
    end
    n = length(points)
    n > 0 || return NTuple{2,Int}[], Float64[]
    X = _points_to_matrix(points)
    tree = _build_exact_tree(X)
    edge_keys = UInt64[]
    edge_vals = Float64[]
    sizehint!(edge_keys, min(max(0, 6 * n), 250_000))
    sizehint!(edge_vals, min(max(0, 6 * n), 250_000))
    @inbounds for i in 1:n
        neigh = inrange(tree, view(X, :, i), r, false)
        for j in neigh
            j <= i && continue
            d_ij = DI._euclidean_distance(points[i], points[j])
            if d_ij <= r
                push!(edge_keys, DI._pack_edge_key(i, j))
                push!(edge_vals, d_ij)
            end
        end
    end
    return DI._finalize_edge_pairs(edge_keys, edge_vals, length(edge_keys))
end

function _radius_graph_edges(points, r::Float64; backend::Symbol=:auto, approx_candidates::Int=0)
    if backend == :bruteforce || backend == :approx
        return nothing
    end
    n = length(points)
    n > 0 || return NTuple{2,Int}[]
    X = _points_to_matrix(points)
    tree = _build_exact_tree(X)
    edge_keys = UInt64[]
    sizehint!(edge_keys, min(max(0, 6 * n), 250_000))
    @inbounds for i in 1:n
        neigh = inrange(tree, view(X, :, i), r, false)
        for j in neigh
            j <= i && continue
            DI._euclidean_distance(points[i], points[j]) <= r || continue
            push!(edge_keys, DI._pack_edge_key(i, j))
        end
    end
    return DI._finalize_edge_pairs_edges_only(edge_keys, length(edge_keys))
end

function _knn_distances(points, k::Int; backend::Symbol=:auto, approx_candidates::Int=0)
    out = _knn_graph(points, k; backend=backend, approx_candidates=approx_candidates)
    out === nothing && return nothing
    return out[3]
end

function _dtm_values(points, dtm_mass::Real; backend::Symbol=:auto, approx_candidates::Int=0)
    backend == :bruteforce && return nothing
    n = length(points)
    n > 1 || throw(ArgumentError("DTM codensity requires at least 2 points."))
    k_eff = DI._dtm_neighbor_count(n, dtm_mass)
    X = _points_to_matrix(points)
    tree = _build_exact_tree(X)
    vals = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        _, dists = knn(tree, view(X, :, i), k_eff, true)
        vals[i] = sqrt(sum(abs2, dists) / k_eff)
    end
    return vals
end

DI._set_pointcloud_nn_impl!(;
    knn_graph=_knn_graph,
    radius_graph=_radius_graph,
    knn_distances=_knn_distances,
    dtm_values=_dtm_values,
    knn_graph_edges=_knn_graph_edges,
    radius_graph_edges=_radius_graph_edges,
)

end # module
