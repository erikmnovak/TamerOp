module Invariants

using LinearAlgebra
using ..CoreModules: QQ, PLikeEncodingMap, locate, region_weights
using ..Encoding: EncodingMap
using ..PLPolyhedra
using JSON3

import ..ExactQQ: rankQQ
import ..FiniteFringe: FinitePoset, FringeModule, Upset, Downset, fiber_dimension
import ..IndicatorTypes: UpsetPresentation, DownsetCopresentation
import ..ModuleComplexes: ModuleCochainComplex
import ..ChangeOfPosets: pushforward_left, pushforward_right
import Base.Threads

# We implement invariants primarily for the internal exact type QQ.
import ..IndicatorResolutions: PModule, map_leq, pmodule_from_fringe, CoverCache, _cover_cache, _chosen_predecessor

import ..ZnEncoding: ZnEncodingMap


export rank_map, rank_invariant, rank_invariant_tame, ecc,
       restricted_hilbert, hilbert_distance,
       slice_chain, restrict_to_chain, slice_barcode,
       bottleneck_distance, matching_distance_approx,
       matching_distance_exact_2d, matching_distance_exact_slices_2d,
       slice_chain_exact_2d,
       sample_directions_2d, default_directions, default_offsets, 
       encoding_box, 
       PersistenceLandscape1D, persistence_landscape, landscape_value,
       MPLandscape, mp_landscape, mp_landscape_distance, mp_landscape_inner_product, mp_landscape_kernel,
       PersistenceImage1D, persistence_image, persistence_silhouette,
       barcode_entropy, barcode_summary, slice_barcode,
       slice_barcodes, precompute_slice_barcode_map, slice_features, slice_kernel,
       integrated_hilbert_mass, measure_by_dimension, support_measure,
       measure_by_value, region_values,
       support_measure, vertex_set_measure,
       dim_stats, dim_norm, region_weight_entropy, aspect_ratio_stats,
       module_size_summary, sliced_bottleneck_distance, sliced_wasserstein_distance_approx,
       interface_measure, interface_measure_by_dim_pair, interface_measure_dim_changes,
       Rect, RectSignedBarcode, axes_from_encoding, rectangle_signed_barcode, rectangle_signed_barcode_rank, rectangles_from_grid, 
       rank_from_signed_barcode, truncate_signed_barcode, rectangle_signed_barcode_image, 
       rectangle_signed_barcode_kernel, mma_decomposition,
       wasserstein_distance, wasserstein_kernel, sliced_wasserstein_distance, sliced_wasserstein_kernel,
       RankQueryCache, rank_query, coarsen_axis, coarsen_axes, restrict_axes_to_encoding,
       FiberedArrangement2D, fibered_arrangement_2d,
       FiberedBarcodeCache2D, fibered_barcode_cache_2d,
       fibered_cell_id, fibered_chain, fibered_values,
       fibered_barcode, fibered_barcode_index, fibered_slice,
       fibered_barcode_cache_stats,
       region_volume_samples_by_dim, region_volume_histograms_by_dim,
       region_boundary_to_volume_samples_by_dim, region_boundary_to_volume_histograms_by_dim,
       graph_degrees, graph_connected_components, graph_modularity,
       region_adjacency_graph_stats,
       ProjectedArrangement1D, ProjectedArrangement, ProjectedBarcodeCache,
       projected_arrangement, projected_barcode_cache, projected_barcodes,
       projected_distances, projected_distance,
       projected_kernel,
       feature_map, feature_vector,
       module_geometry_summary,
       betti_support_measures, bass_support_measures,
       PrettyPrinter, pretty,
       PointSignedMeasure,
       point_signed_measure,
       surface_from_point_signed_measure,
       truncate_point_signed_measure,
       point_signed_measure_kernel,
       euler_characteristic_surface,
       euler_surface,
       euler_signed_measure,
       euler_distance,
       MPPLineSpec, MPPDecomposition, MPPImage,
       mpp_decomposition, mpp_image,
       mpp_image_distance, mpp_image_inner_product, mpp_image_kernel,
       save_mpp_decomposition_json, load_mpp_decomposition_json,
       save_mpp_image_json, load_mpp_image_json,
       rectangle_measure, rectangles,
       upsets_covering, downsets_covering,
       minimal_primes, socle, associated_primes,
       multiparameter_landscape, multiparameter_landscape_standard,
       matching_distance, matching_distance_approx_2d,
       FiberedSliceFamily2D, fibered_slice_family_2d
    


# ----- Rank invariant ----------------------------------------------------------

# Internal helper: compute M(u->v) with memoization shared across many calls.
function _map_leq_cached(
    M::PModule{QQ},
    u::Int,
    v::Int,
    cc,
    memo::Dict{Tuple{Int,Int}, Matrix{QQ}}
)::Matrix{QQ}
    key = (u, v)
    if haskey(memo, key)
        return memo[key]
    end
    if u == v
        X = Matrix{QQ}(I, M.dims[v], M.dims[u])
        memo[key] = X
        return X
    end

    # Fast path: cover edge.
    # NOTE: This may alias internal storage of the module; treat as read-only.
    if cc.C[u, v]
        X = M.edge_maps[u, v]
        memo[key] = X
        return X
    end

    # Choose (and cache) a predecessor w of v that is still >= u.
    # This uses the same parent-pointer cache as `IndicatorResolutions.map_leq`.
    w = _chosen_predecessor(cc, u, v)
    X = _map_leq_cached(M, u, w, cc, memo)
    Y = M.edge_maps[w, v]
    memo[key] = Y * X
    return memo[key]
end


"""
    rank_map(M, a, b; cache=nothing, memo=nothing) -> Int

Return the rank of the structure map `M(a <= b)` for comparable `a <= b`.

This is a "finite encoding" rank invariant query:
- `M` is a `PModule{QQ}` (or a `FringeModule{QQ}`; see method below)
- `a` and `b` are vertex indices in the underlying finite poset

Keyword arguments:
- `cache`: internal cover-cache object (advanced; used by `rank_invariant`)
- `memo`: dictionary used to memoize computed maps (advanced)

Notes for performance:
If you will make many rank queries for a `FringeModule`, convert once:
`Mp = pmodule_from_fringe(H)`, then call `rank_map(Mp, ...)`.
"""
function rank_map(M::PModule{QQ}, a::Int, b::Int; cache=nothing, memo=nothing)::Int
    Q = M.Q
    (1 <= a <= Q.n) || error("rank_map: a out of range")
    (1 <= b <= Q.n) || error("rank_map: b out of range")
    Q.leq[a, b] || error("rank_map: a and b are not comparable (need a <= b)")

    if a == b
        return M.dims[a]
    end

    # Use shared memoization if provided; otherwise fall back to map_leq.
    if memo !== nothing
        cc = cache === nothing ? _cover_cache(Q) : cache
        A = _map_leq_cached(M, a, b, cc, memo)
        return rankQQ(A)
    end

    A = map_leq(M, a, b; cache=cache)
    return rankQQ(A)
end

function rank_map(H::FringeModule{QQ}, a::Int, b::Int; kwargs...)::Int
    Mp = pmodule_from_fringe(H)
    return rank_map(Mp, a, b; kwargs...)
end


"""
    rank_map(M, pi, x, y; strict=true) -> Int

Convenience wrapper: compute `rank_map(M, a, b)` where
`a = locate(pi, x)` and `b = locate(pi, y)`.

This is meant for finite encodings coming from:
- Z^n (ZnEncodingMap)
- PL encodings (PLEncodingMap / PLEncodingMapBoxes)
- any other encoding map type implementing `locate(pi, x)`

If `strict=true` and either point maps to 0 (unknown region), an error is thrown.
If `strict=false`, unknown regions yield rank 0.
"""
function rank_map(M::PModule{QQ}, pi, x, y; strict::Bool=true)::Int
    a = locate(pi, x)
    b = locate(pi, y)
    if (a == 0 || b == 0)
        strict && error("rank_map: locate(pi, x) or locate(pi, y) returned 0 (unknown region)")
        return 0
    end
    return rank_map(M, a, b)
end

function rank_map(H::FringeModule{QQ}, pi, x, y; strict::Bool=true)::Int
    Mp = pmodule_from_fringe(H)
    return rank_map(Mp, pi, x, y; strict=strict)
end


"""
    rank_invariant(M::PModule{QQ}; store_zeros=false, threads=(Threads.nthreads() > 1))

Compute the rank invariant of a module `M`, returning a dictionary mapping
`(a, b)` with `a <= b` to the rank of the structure map `M(a <= b)`.

Keyword arguments:

- `store_zeros`: if `true`, include all comparable pairs `(a, b)` including rank 0.
  If `false`, store only positive ranks (sparser and usually faster).
- `threads`: if `true` and Julia has more than one thread, parallelize over the
  outer vertex index `a`. This is safe: `CoverCache` is thread-safe and each
  thread uses its own map memo and output dictionary.
"""
function rank_invariant(
    M::PModule{QQ};
    store_zeros::Bool = false,
    threads::Bool = (Threads.nthreads() > 1),
)
    Q = M.Q
    cc = _cover_cache(Q)

    if threads && Threads.nthreads() > 1
        nT = Threads.nthreads()
        memo_by_thread = [Dict{Tuple{Int, Int}, Matrix{QQ}}() for _ in 1:nT]
        out_by_thread = [Dict{Tuple{Int, Int}, Int}() for _ in 1:nT]

        Threads.@threads for a in 1:Q.n
            tid = Threads.threadid()
            memo = memo_by_thread[tid]
            out = out_by_thread[tid]
            for b in 1:Q.n
                r = rank_map(M, a, b; cache = cc, memo = memo)
                if store_zeros || r > 0
                    out[(a, b)] = r
                end
            end
        end

        ranks = Dict{Tuple{Int, Int}, Int}()
        for d in out_by_thread
            merge!(ranks, d)
        end
        return ranks
    end

    # Serial fallback (unchanged logic).
    memo = Dict{Tuple{Int, Int}, Matrix{QQ}}()
    ranks = Dict{Tuple{Int, Int}, Int}()
    for a in 1:Q.n, b in 1:Q.n
        r = rank_map(M, a, b; cache = cc, memo = memo)
        if store_zeros || r > 0
            ranks[(a, b)] = r
        end
    end
    return ranks
end


function rank_invariant(H::FringeModule{QQ}; kwargs...)
    Mp = pmodule_from_fringe(H)
    return rank_invariant(Mp; kwargs...)
end


#--------------------------------------------------------------------
# Signed barcodes / rectangle measures (Mobius inversion of rank).
#--------------------------------------------------------------------

"""
    Rect{N}

An axis-aligned hyperrectangle in Z^N, represented by two corners `lo` and `hi`
with `lo <= hi` coordinatewise.

This type is used to represent "rectangle signed barcodes" (also called
"signed rectangle measures") obtained by Mobius inversion of the rank invariant.
"""
struct Rect{N}
    lo::NTuple{N,Int}
    hi::NTuple{N,Int}
    function Rect{N}(lo::NTuple{N,Int}, hi::NTuple{N,Int}) where {N}
        _tuple_leq(lo, hi) || error("Rect: expected lo <= hi coordinatewise")
        return new{N}(lo, hi)
    end
end

# Coordinatewise partial order on integer tuples.
_tuple_leq(a::NTuple{N,Int}, b::NTuple{N,Int}) where {N} = all(a[i] <= b[i] for i in 1:N)

function Base.show(io::IO, r::Rect{N}) where {N}
    print(io, "Rect", r.lo, " => ", r.hi)
end

"""
    RectSignedBarcode{N,T}

A finite signed multiset of axis-aligned hyperrectangles in Z^N.

Fields:
- `axes`: the coordinate grids used for the inversion, one sorted vector per axis.
- `rects`: the rectangles (elements of the free abelian group on rectangles).
- `weights`: the integer weights (can be negative).

Interpretation:
Given a rank invariant function `r(p,q)` on comparable pairs `p <= q` in the grid,
Mobius inversion produces weights `w(lo,hi)` such that

    r(p,q) = sum_{lo <= p, q <= hi} w(lo,hi)

for all grid pairs `p <= q`. For N=1 this recovers the usual barcode multiplicities.
For N>1 the result is typically signed, hence the name "signed barcode".
"""
struct RectSignedBarcode{N,T<:Integer}
    axes::NTuple{N,Vector{Int}}
    rects::Vector{Rect{N}}
    weights::Vector{T}
end

Base.length(sb::RectSignedBarcode) = length(sb.rects)


# =============================================================================
# Signed point measures (Euler signed-measure / Mobius inversion on a grid)
# =============================================================================

"""
    PointSignedMeasure(axes, inds, wts)

A sparse signed measure supported on a rectangular grid.

- `axes` is an `N`-tuple of coordinate vectors `(a1, ..., aN)` describing the grid.
- `inds` is a vector of `N`-tuples of *indices* into those axes.
- `wts` is the corresponding vector of signed weights.

This is the point-measure analogue of `RectSignedBarcode`.  It is the natural
output type for an Euler signed-measure decomposition, but it can also be used
for any grid function whose Mobius inversion (on the product-of-chains poset)
is desired.

Notes for mathematicians:
- If `f : A1 x ... x AN -> Z` is a function on a finite product of chains,
  then the Mobius inversion `mu` satisfies

      f(x) = sum_{y <= x} mu(y),

  where `<=` is the product order.  `PointSignedMeasure` stores `mu`.
"""
struct PointSignedMeasure{N,T,W}
    axes::NTuple{N,Vector{T}}
    inds::Vector{NTuple{N,Int}}
    wts::Vector{W}
end

Base.length(pm::PointSignedMeasure) = length(pm.wts)

function Base.show(io::IO, pm::PointSignedMeasure)
    N = length(pm.axes)
    print(io, "PointSignedMeasure{", N, "}(")
    print(io, length(pm), " points)")
end

"""
    truncate_point_signed_measure(pm; max_terms=0, min_abs_weight=0)

Return a new `PointSignedMeasure` keeping only the "largest" terms.

- Drops all terms with `abs(w) < min_abs_weight`.
- If `max_terms > 0`, keeps only the top `max_terms` by `abs(w)`.

This mirrors `truncate_signed_barcode` for rectangles.
"""
function truncate_point_signed_measure(pm::PointSignedMeasure;
                                       max_terms::Int=0,
                                       min_abs_weight::Real=0)
    n = length(pm)
    if n == 0
        return pm
    end

    keep = Int[]
    sizehint!(keep, n)
    @inbounds for i in 1:n
        if abs(pm.wts[i]) >= min_abs_weight
            push!(keep, i)
        end
    end

    if max_terms > 0 && length(keep) > max_terms
        # sort kept indices by descending abs(weight)
        sort!(keep, by=i -> -abs(pm.wts[i]))
        keep = keep[1:max_terms]
        sort!(keep)  # restore increasing index order for stability
    end

    new_inds = Vector{NTuple{length(pm.axes),Int}}(undef, length(keep))
    new_wts  = Vector{eltype(pm.wts)}(undef, length(keep))
    @inbounds for (j,i) in enumerate(keep)
        new_inds[j] = pm.inds[i]
        new_wts[j]  = pm.wts[i]
    end

    return PointSignedMeasure(pm.axes, new_inds, new_wts)
end

# -----------------------------------------------------------------------------
# Internal: in-place Mobius inversion on product of chains via iterated diffs
# -----------------------------------------------------------------------------

# Apply 1D "difference" operator along each axis in-place.
# For 1D: w[i] <- f[i] - f[i-1] (with f[0]=0)
# For ND: iterated differences gives the product-poset Mobius inversion.
function _mobius_inversion_product_chains!(w::AbstractArray)
    N = ndims(w)
    for d in 1:N
        # eachslice keeps dimension d and fixes all others, producing 1D views
        for sl in eachslice(w; dims=d)
            @inbounds for k in length(sl):-1:2
                sl[k] -= sl[k-1]
            end
        end
    end
    return w
end

# Inverse of Mobius inversion: prefix sums along each axis in-place.
function _prefix_sum_product_chains!(f::AbstractArray)
    N = ndims(f)
    for d in 1:N
        for sl in eachslice(f; dims=d)
            @inbounds for k in 2:length(sl)
                sl[k] += sl[k-1]
            end
        end
    end
    return f
end

# -----------------------------------------------------------------------------
# Mixed-orientation Mobius inversion on products of chains
#
# Rectangle signed barcodes live on "interval posets":
# pairs (p,q) with p <= q, ordered by inclusion:
#   (lo,hi) <= (p,q)  iff  lo <= p and q <= hi.
#
# This is a product of chains in the "lo/p" coordinates and REVERSED chains in
# the "hi/q" coordinates.  Computationally, Mobius inversion is still just an
# iterated 1D difference along each axis; reversed axes use a forward difference.
# -----------------------------------------------------------------------------

# In-place Mobius inversion on a product of chains where some axes are reversed.
#
# For a non-reversed axis:
#   w[k] <- f[k] - f[k-1]  (with f[0] = 0)
# For a reversed axis:
#   w[k] <- f[k] - f[k+1]  (with f[end+1] = 0)
#
# This is the natural transform for interval-posets / rectangle measures.
function _mobius_inversion_product_chains_mixed!(w::AbstractArray,
                                                reverse_axis::NTuple{N,Bool}) where {N}
    ndims(w) == N || error("_mobius_inversion_product_chains_mixed!: reverse_axis must have length ndims(w)")
    for d in 1:N
        if reverse_axis[d]
            # Forward difference (uses the next entry).
            for sl in eachslice(w; dims=d)
                @inbounds for k in 1:(length(sl)-1)
                    sl[k] -= sl[k+1]
                end
            end
        else
            # Backward difference (uses the previous entry).
            for sl in eachslice(w; dims=d)
                @inbounds for k in length(sl):-1:2
                    sl[k] -= sl[k-1]
                end
            end
        end
    end
    return w
end

# Inverse transform to `_mobius_inversion_product_chains_mixed!`:
# in-place "mixed prefix sums".
#
# For a non-reversed axis:
#   f[k] <- f[k] + f[k-1]  (prefix sum)
# For a reversed axis:
#   f[k] <- f[k] + f[k+1]  (reverse prefix sum / suffix sum)
function _prefix_sum_product_chains_mixed!(f::AbstractArray,
                                          reverse_axis::NTuple{N,Bool}) where {N}
    ndims(f) == N || error("_prefix_sum_product_chains_mixed!: reverse_axis must have length ndims(f)")
    for d in 1:N
        if reverse_axis[d]
            for sl in eachslice(f; dims=d)
                @inbounds for k in (length(sl)-1):-1:1
                    sl[k] += sl[k+1]
                end
            end
        else
            for sl in eachslice(f; dims=d)
                @inbounds for k in 2:length(sl)
                    sl[k] += sl[k-1]
                end
            end
        end
    end
    return f
end


"""
    point_signed_measure(surface, axes; drop_zeros=true)

Compute the Mobius inversion of `surface` on the product-of-chains grid given by `axes`.

- `surface` is an `N`-dimensional array of values on the grid.
- `axes` is an `N`-tuple of coordinate vectors, whose lengths match `size(surface)`.

Returns a `PointSignedMeasure` supported on grid points.

Performance notes:
- Uses in-place iterated differences (O(N * prod(size))) rather than O(2^N) inclusion-exclusion.
"""
function point_signed_measure(surface::AbstractArray{<:Integer,N},
                              axes::NTuple{N,AbstractVector};
                              drop_zeros::Bool=true) where {N}
    # Normalize axes to concrete vectors (no views/ranges stored in the measure)
    ax = ntuple(i -> collect(axes[i]), N)
    size(surface) == ntuple(i -> length(ax[i]), N) ||
        throw(ArgumentError("axes lengths must match surface size"))

    w = copy(surface)
    _mobius_inversion_product_chains!(w)

    inds = Vector{NTuple{N,Int}}()
    wts  = Vector{eltype(w)}()
    sizehint!(inds, length(w))
    sizehint!(wts,  length(w))

    @inbounds for I in CartesianIndices(w)
        val = w[I]
        if !drop_zeros || val != 0
            push!(inds, ntuple(k -> I[k], N))
            push!(wts, val)
        end
    end

    return PointSignedMeasure(ax, inds, wts)
end

"""
    surface_from_point_signed_measure(pm)

Reconstruct the grid function `surface` from its Mobius inversion stored in `pm`.

This is the inverse operation to `point_signed_measure` (up to dropped zeros).
"""
function surface_from_point_signed_measure(pm::PointSignedMeasure{N,T,W}) where {N,T,W}
    dims = ntuple(i -> length(pm.axes[i]), N)
    f = zeros(Int, dims...)
    @inbounds for i in 1:length(pm)
        I = pm.inds[i]
        f[I...] += pm.wts[i]
    end
    _prefix_sum_product_chains!(f)
    return f
end

"""
    point_signed_measure_kernel(pm1, pm2; sigma=1.0, kind=:gaussian)

A simple weighted point-cloud kernel for signed point measures.

- `kind=:gaussian` uses `exp(-||x-y||^2 / (2*sigma^2))`
- `kind=:laplacian` uses `exp(-||x-y|| / sigma)`

This is optional but convenient for ML-style pipelines on Euler signed measures.
"""
function point_signed_measure_kernel(pm1::PointSignedMeasure{N},
                                     pm2::PointSignedMeasure{N};
                                     sigma::Real=1.0,
                                     kind::Symbol=:gaussian) where {N}
    sigma > 0 || throw(ArgumentError("sigma must be positive"))
    s2 = float(sigma*sigma)
    acc = 0.0
    @inbounds for i in 1:length(pm1)
        I = pm1.inds[i]
        w1 = float(pm1.wts[i])
        x = ntuple(d -> pm1.axes[d][I[d]], N)
        for j in 1:length(pm2)
            J = pm2.inds[j]
            w2 = float(pm2.wts[j])
            y = ntuple(d -> pm2.axes[d][J[d]], N)

            # Euclidean distance in coordinate space
            dsq = 0.0
            for d in 1:N
                t = float(x[d] - y[d])
                dsq += t*t
            end

            k = if kind === :gaussian
                exp(-dsq/(2*s2))
            elseif kind === :laplacian
                exp(-sqrt(dsq)/float(sigma))
            else
                throw(ArgumentError("kind must be :gaussian or :laplacian"))
            end

            acc += w1*w2*k
        end
    end
    return acc
end



"""
    axes_from_encoding(pi::ZnEncodingMap)

Infer a coordinate grid along each axis from a finite encoding map `pi`.

The resulting axes are sorted and unique. In typical finite-encoding constructions,
these are the "critical coordinates" along each axis (possibly including one
representative below the first critical coordinate).
"""
function axes_from_encoding(pi::ZnEncodingMap)
    # NOTE:
    # For ZnEncodingMap, `pi.coords` is a vector of *axis-wise* critical coordinates:
    #     pi.coords[i] = sorted unique Int breakpoints along axis i
    # so:
    #   - the ambient dimension is pi.n == length(pi.coords)
    #   - the natural "encoding grid" along axis i is given by representatives of the
    #     slabs determined by these breakpoints.
    #
    # If the breakpoints are c1 < c2 < ... < ck, then Z is partitioned into slabs
    #   (-inf, c1-1], [c1, c2-1], ..., [c_{k-1}, c_k-1], [c_k, +inf)
    # and convenient integer representatives are
    #   c1-1, c1, c2, ..., ck.
    #
    # If an axis has no breakpoints (fully free), we use [0] as a harmless representative.
    n = pi.n
    length(pi.coords) == n || error("axes_from_encoding: expected length(pi.coords) == pi.n")

    return ntuple(i -> begin
        ci = pi.coords[i]
        if isempty(ci)
            [0]
        else
            ax = Vector{Int}(undef, length(ci) + 1)
            ax[1] = ci[1] - 1
            @inbounds for j in 1:length(ci)
                ax[j + 1] = ci[j]
            end
            sort!(ax)
            unique!(ax)
            ax
        end
    end, n)
end

# Build the nonzero Mobius coefficients for a product of chains.
# For a chain, mu(i,j) is nonzero only when j in {i, i+1}, with values {+1, -1}.
# For a product of N chains, mu is the product, hence 2^N patterns.
function _mobius_eps(N::Int)
    m = 1 << N
    eps = Vector{NTuple{N,Int}}(undef, m)
    sgn = Vector{Int}(undef, m)
    for mask in 0:(m-1)
        eps[mask+1] = ntuple(i -> Int((mask >> (i-1)) & 1), N)
        sgn[mask+1] = isodd(Base.count_ones(mask)) ? -1 : 1
    end
    return eps, sgn
end

_shift_plus(t::NTuple{N,Int}, eps::NTuple{N,Int}) where {N} = ntuple(i -> t[i] + eps[i], N)
_shift_minus(t::NTuple{N,Int}, eps::NTuple{N,Int}) where {N} = ntuple(i -> t[i] - eps[i], N)


###############################################################################
# Performance helpers (memoization and grid restriction) for rectangle signed
# barcodes on large grids.
###############################################################################

"""
    RankQueryCache(pi::ZnEncodingMap)

Cache object used by `rectangle_signed_barcode(M, pi, ...)` and related routines.

The cache contains:
- `loc_cache`: memoizes `locate(pi, g)` for lattice points `g` (g is an `NTuple{N,Int}`),
- `rank_cache`: memoizes `rank_map(M, a, b)` for region indices `a,b` in the encoded poset.

This is a pure performance tool: it does not change mathematical results. The cache
is safe to reuse across calls as long as you keep the same encoding map `pi`. The
`rank_cache` is specific to the module `M`; do not reuse it across different modules.
"""
mutable struct RankQueryCache{N}
    loc_cache::Dict{NTuple{N,Int},Int}
    rank_cache::Dict{Tuple{Int,Int},Int}
end

function RankQueryCache(pi::ZnEncodingMap)
    # For ZnEncodingMap, the dimension is pi.n (not length(pi.coords[1])).
    N = pi.n
    length(pi.coords) == N || error("RankQueryCache: expected length(pi.coords) == pi.n")

    return RankQueryCache{N}(
        Dict{NTuple{N,Int},Int}(),
        Dict{Tuple{Int,Int},Int}(),
    )
end

"""
    coarsen_axis(axis; max_len, method=:uniform)

Heuristic grid restriction: downsample a sorted axis to have at most `max_len`
values (always keeping the first and last element). This is intended for very
large grids where enumerating all rectangles is too expensive.

If `max_len <= 0` or `length(axis) <= max_len`, returns a sorted unique copy of
`axis`.

Currently supported methods:
- `:uniform`: uniform downsampling in index space.
"""
function coarsen_axis(axis::AbstractVector{<:Integer}; max_len::Int, method::Symbol=:uniform)
    ax = sort(unique(Int.(axis)))
    L = length(ax)
    if max_len <= 0 || L <= max_len
        return ax
    end
    if method != :uniform
        error("coarsen_axis: unsupported method = $(method); use :uniform")
    end
    # Uniformly sample indices in [1, L], always including endpoints.
    idxs = unique(round.(Int, range(1, L; length=max_len)))
    idxs = sort(unique(vcat(1, idxs, L)))
    return ax[idxs]
end

"""
    coarsen_axes(axes; max_len, method=:uniform)

Apply `coarsen_axis` to each axis in an `N`-tuple `axes`.
"""
function coarsen_axes(axes::NTuple{N,<:AbstractVector{<:Integer}}; max_len::Int, method::Symbol=:uniform) where {N}
    return ntuple(k -> coarsen_axis(axes[k]; max_len=max_len, method=method), N)
end

"""
    restrict_axes_to_encoding(axes, pi; keep_endpoints=true)

Grid restriction heuristic for `ZnEncodingMap`.

Given user-provided axes and an encoding map `pi`, keep only those axis values
that appear in the encoding-derived axes `axes_from_encoding(pi)` and lie between
the user axis endpoints. If `keep_endpoints=true`, also keep the user endpoints.

For typical `ZnEncodingMap`s, the rank invariant is constant on the axis-aligned
cells determined by the encoding, so this restriction often removes redundant grid
points without changing nonzero signed weights.
"""
function restrict_axes_to_encoding(axes::NTuple{N,<:AbstractVector{<:Integer}}, pi::ZnEncodingMap; keep_endpoints::Bool=true) where {N}
    enc = axes_from_encoding(pi)
    return ntuple(k -> _restrict_axis_to_encoding(axes[k], enc[k]; keep_endpoints=keep_endpoints), N)
end

function _restrict_axis_to_encoding(axis::AbstractVector{<:Integer}, enc_axis::Vector{Int}; keep_endpoints::Bool=true)
    ax = sort(unique(Int.(axis)))
    if isempty(ax)
        return copy(enc_axis)
    end
    lo, hi = first(ax), last(ax)
    vals = Int[]
    for v in enc_axis
        if lo <= v <= hi
            push!(vals, v)
        end
    end
    if keep_endpoints
        push!(vals, lo)
        push!(vals, hi)
    end
    return sort(unique(vals))
end

# Internal: normalize an axis tuple to an `NTuple{N,Vector{Int}}` of sorted unique axes.
function _normalize_axes(axes::NTuple{N,<:AbstractVector{<:Integer}}) where {N}
    return ntuple(k -> sort(unique(Int.(axes[k]))), N)
end


# -----------------------------------------------------------------------------
# Real-valued axes support (needed for PL/box encodings and general R^n grids)
# -----------------------------------------------------------------------------

function _normalize_axes_real(axes::NTuple{N,AbstractVector}) where {N}
    ax = ntuple(i -> sort(collect(axes[i])), N)
    for i in 1:N
        length(ax[i]) > 0 || throw(ArgumentError("axis cannot be empty"))
    end
    return ax
end

# Intersection of two sorted vectors (generic Real). Used by :encoding axis policy.
function _axis_intersection_real(a::AbstractVector{<:Real},
                                 b::AbstractVector{<:Real})
    i = 1
    j = 1
    out = eltype(a)[]
    sizehint!(out, min(length(a), length(b)))
    while i <= length(a) && j <= length(b)
        if a[i] == b[j]
            push!(out, a[i]); i += 1; j += 1
        elseif a[i] < b[j]
            i += 1
        else
            j += 1
        end
    end
    return out
end

# Restrict a proposed axis to one supported by the encoding axis (generic Real).
# Keeps only values that appear in the encoding axis.
function _restrict_axis_to_encoding(axis::AbstractVector{<:Real},
                                   enc_axis::AbstractVector{<:Real})
    axis2 = sort(collect(axis))
    enc2  = sort(collect(enc_axis))
    out = _axis_intersection_real(axis2, enc2)
    length(out) > 0 || throw(ArgumentError("axis restriction is empty"))
    return out
end

# Generic fallback: if an encoding supports axes_from_encoding, we can restrict.
function restrict_axes_to_encoding(axes::NTuple{N,AbstractVector}, pi) where {N}
    enc_axes = axes_from_encoding(pi)  # requires a method; may be provided by Zn or PLBackend
    length(enc_axes) == N || throw(ArgumentError("axes dimension mismatch"))
    return ntuple(i -> _restrict_axis_to_encoding(axes[i], enc_axes[i]), N)
end

# Coarsen any sorted axis (Integer or Real) by downsampling every other point.
function coarsen_axis(axis::AbstractVector{<:Real})
    length(axis) <= 2 && return collect(axis)
    out = eltype(axis)[]
    sizehint!(out, (length(axis)+1) >>> 1)
    @inbounds for i in 1:2:length(axis)
        push!(out, axis[i])
    end
    # Ensure last point included (preserve extent)
    if out[end] != axis[end]
        push!(out, axis[end])
    end
    return out
end



"""
    _rectangle_signed_barcode_local(rank_idx, axes; drop_zeros=true, tol=0, max_span=nothing)

Compute the rectangle signed barcode by N-dimensional Mobius inversion of a rank invariant
sampled on a finite grid.

Arguments:
- `axes` is a tuple of sorted coordinate lists `(a1, ..., aN)`. The grid points are the
  cartesian product of these axes.
- `rank_idx(p, q)` must accept index tuples `p::NTuple{N,Int}`, `q::NTuple{N,Int}` with
  `p <= q` coordinatewise (indices into `axes`), and return the rank invariant value at the
  corresponding grid points.

Keyword arguments:
- `drop_zeros`: if true, return only rectangles with `|w| > tol`; otherwise include all
  enumerated rectangles and store `0` when `|w| <= tol`.
- `tol`: integer threshold for treating small weights as zero.
- `max_span`: optional grid restriction heuristic. If `max_span` is an `Int` or an `N`-tuple,
  only rectangles with coordinate spans `axes[k][q[k]] - axes[k][p[k]] <= max_span[k]` are
  enumerated. This can dramatically reduce runtime on very large grids.

Returns:
A `RectSignedBarcode{N,Int}`.

Notes:
This is the higher-dimensional analog of recovering a 1-parameter barcode from the rank function.
For `N > 1` the output is a signed measure on rectangles (not, in general, a literal decomposition
of the module as a direct sum of rectangle modules).
"""
function _rectangle_signed_barcode_local(rank_idx::Function, axes::NTuple{N,Vector{Int}};
    drop_zeros::Bool=true,
    tol::Int=0,
    max_span=nothing) where {N}

    @assert tol >= 0 "rectangle_signed_barcode: tol must be nonnegative"

    rects = Rect{N}[]
    weights = Int[]

    dims = ntuple(k -> length(axes[k]), N)
    CI = CartesianIndices(dims)

    # Precompute epsilon-shifts and signs for Mobius inversion.
    eps_list, eps_sign = _mobius_eps(N)

    # Normalize max_span to an NTuple{N,Int} or `nothing`.
    span = nothing
    if max_span !== nothing
        if max_span isa Integer
            s = Int(max_span)
            @assert s >= 0 "rectangle_signed_barcode: max_span must be nonnegative"
            span = ntuple(_ -> s, N)
        else
            @assert length(max_span) == N "rectangle_signed_barcode: max_span must be an Int or an N-tuple"
            span = ntuple(k -> Int(max_span[k]), N)
            @assert all(span[k] >= 0 for k in 1:N) "rectangle_signed_barcode: max_span must be nonnegative"
        end
    end

    for pCI in CI
        p = pCI.I

        # Enumerate q indices with p <= q coordinatewise, optionally restricting spans.
        ranges = ntuple(k -> begin
            start = p[k]
            stop = dims[k]
            if span !== nothing
                pval = @inbounds axes[k][start]
                qmax = pval + span[k]
                qstop = searchsortedlast(axes[k], qmax)
                stop = min(stop, qstop)
            end
            start:stop
        end, N)

        for qCI in CartesianIndices(ranges)
            q = qCI.I

            w = 0
            @inbounds for (i, eps_p) in enumerate(eps_list), (j, eps_q) in enumerate(eps_list)
                p2 = _shift_minus(p, eps_p)
                q2 = _shift_plus(q, eps_q)

                # Ensure shifted indices still satisfy 1 <= p2 <= q2 <= dims.
                ok = true
                for k in 1:N
                    if p2[k] < 1 || q2[k] > dims[k] || p2[k] > q2[k]
                        ok = false
                        break
                    end
                end
                if ok
                    w += eps_sign[i] * eps_sign[j] * rank_idx(p2, q2)
                end
            end

            if abs(w) > tol
                # NOTE (parsing pitfall):
                # Avoid writing `ntuple(k -> @inbounds axes[k][p[k]], N)` because `@inbounds`
                # can accidentally capture the trailing `, N`, leading to a one-argument call
                # `ntuple(closure)` and a MethodError at runtime.
                lo = ntuple(Val(N)) do k
                    @inbounds axes[k][p[k]]
                end
                hi = ntuple(Val(N)) do k
                    @inbounds axes[k][q[k]]
                end
                push!(rects, Rect{N}(lo, hi))
                push!(weights, w)
            elseif !drop_zeros
               lo = ntuple(Val(N)) do k
                    @inbounds axes[k][p[k]]
                end
                hi = ntuple(Val(N)) do k
                    @inbounds axes[k][q[k]]
                end
                push!(rects, Rect{N}(lo, hi))
                push!(weights, 0)
            end
        end
    end

    return RectSignedBarcode{N,Int}(axes, rects, weights)
end

# -----------------------------------------------------------------------------
# Fast rectangle signed barcode computation (bulk rank + Mobius inversion)
# -----------------------------------------------------------------------------

function _estimate_rectangle_rank_array_elems(axes::NTuple{N,Vector{Int}}) where {N}
    # We allocate an Int array of size prod(length.(axes))^2.
    # Use overflow-safe arithmetic to avoid nonsense sizes on large grids.
    m = 1
    for k in 1:N
        mk, of = Base.mul_with_overflow(m, length(axes[k]))
        if of
            return typemax(Int)
        end
        m = mk
    end
    mm, of = Base.mul_with_overflow(m, m)
    return of ? typemax(Int) : mm
end

function _choose_rectangle_signed_barcode_method(method::Symbol,
                                                 axes::NTuple{N,Vector{Int}};
                                                 max_span,
                                                 bulk_max_elems::Int) where {N}
    if method == :auto
        if max_span === nothing
            ne = _estimate_rectangle_rank_array_elems(axes)
            return (ne <= bulk_max_elems) ? :bulk : :local
        else
            # Span truncation typically makes local enumeration preferable.
            return :local
        end
    elseif method == :local
        return :local
    elseif method == :bulk
        ne = _estimate_rectangle_rank_array_elems(axes)
        if ne > bulk_max_elems
            throw(ArgumentError(
                "method=:bulk would allocate an array with $ne Int entries " *
                "(> bulk_max_elems=$bulk_max_elems). " *
                "Increase bulk_max_elems, coarsen the axes, or use method=:local."
            ))
        end
        return :bulk
    else
        throw(ArgumentError("Unknown method=$(method). Use :auto, :bulk, or :local."))
    end
end

function _rectangle_signed_barcode_bulk(rank_idx::Function,
                                        axes::NTuple{N,Vector{Int}};
                                        drop_zeros::Bool=true,
                                        tol::Int=0,
                                        max_span=nothing) where {N}
    span = max_span === nothing ? nothing :
        (max_span isa Integer ? ntuple(_ -> Int(max_span), N) :
                               ntuple(i -> Int(max_span[i]), N))

    dims = ntuple(i -> length(axes[i]), N)

    # Store the rank function r(p,q) on the whole grid (p,q), with the convention
    # r(p,q) = 0 when p is not <= q.  We only evaluate rank_idx on comparable
    # pairs and leave the rest at 0.
    r = zeros(Int, (dims..., dims...))

    if N == 1
        d1 = dims[1]
        for p in 1:d1
            for q in p:d1
                @inbounds r[p,q] = rank_idx((p,), (q,))
            end
        end
    else
        CI = CartesianIndices(dims)
        for pCI in CI
            p = pCI.I
            q_ranges = ntuple(k -> p[k]:dims[k], N)
            for qCI in CartesianIndices(q_ranges)
                q = qCI.I
                @inbounds r[p..., q...] = rank_idx(p, q)
            end
        end
    end

    # Mobius inversion on the interval-poset: p axes normal, q axes reversed.
    reverse_axis = ntuple(d -> d > N, 2N)
    _mobius_inversion_product_chains_mixed!(r, reverse_axis)

    rects = Rect{N}[]
    weights = Int[]
    CI = CartesianIndices(dims)
    for pCI in CI
        p = pCI.I
        q_ranges = ntuple(k -> p[k]:dims[k], N)
        for qCI in CartesianIndices(q_ranges)
            q = qCI.I
            w = @inbounds r[p..., q...]

            if span !== nothing
                bad = false
                @inbounds for k in 1:N
                    if axes[k][q[k]] - axes[k][p[k]] > span[k]
                        bad = true
                        break
                    end
                end
                bad && continue
            end

            if drop_zeros && (abs(w) <= tol)
                continue
            end

            lo = ntuple(k -> axes[k][p[k]], N)
            hi = ntuple(k -> axes[k][q[k]], N)
            push!(rects, Rect{N}(lo, hi))
            push!(weights, w)
        end
    end

    return RectSignedBarcode{N,Int}(axes, rects, weights)
end

"""
    rectangle_signed_barcode(rank_idx, axes; drop_zeros=true, tol=0, max_span=nothing,
                             method=:auto, bulk_max_elems=20_000_000)

Compute the rectangle signed barcode (a signed decomposition into axis-aligned
rectangles) induced by a rank function on a finite grid.

The rank function is provided as `rank_idx(p,q)`, where `p` and `q` are
N-tuples of *indices* into the coordinate axes, and should be interpreted as
a rank invariant on comparable pairs `p <= q` (coordinatewise).

Algorithm choices:

- `method=:local` reproduces the original inclusion-exclusion formula, evaluating
  `rank_idx` up to `2^(2N)` times per rectangle (but allowing early truncation by
  `max_span`).
- `method=:bulk` evaluates `rank_idx` once per comparable pair `(p,q)` and then
  performs a single mixed-orientation Mobius inversion (finite differences) on a
  `prod(length.(axes))^2` array. This is typically much faster when you want the
  full barcode and rank queries are cheap enough to make overhead matter.
- `method=:auto` (default) uses `:bulk` when `max_span === nothing` and the
  required rank array is at most `bulk_max_elems` entries; otherwise it uses
  `:local`.

The output is a `RectSignedBarcode`, a list of rectangles with signed integer
weights. Zero-weight rectangles are omitted by default.
"""
function rectangle_signed_barcode(rank_idx::Function,
                                  axes::NTuple{N,<:AbstractVector{<:Integer}};
                                  drop_zeros::Bool=true,
                                  tol::Int=0,
                                  max_span=nothing,
                                  method::Symbol=:auto,
                                  bulk_max_elems::Int=20_000_000) where {N}
    ax = ntuple(i -> collect(Int, axes[i]), N)
    meth = _choose_rectangle_signed_barcode_method(
        method, ax; max_span=max_span, bulk_max_elems=bulk_max_elems
    )
    if meth == :bulk
        return _rectangle_signed_barcode_bulk(
            rank_idx, ax; drop_zeros=drop_zeros, tol=tol, max_span=max_span
        )
    else
        return _rectangle_signed_barcode_local(
            rank_idx, ax; drop_zeros=drop_zeros, tol=tol, max_span=max_span
        )
    end
end

"""
    rectangle_signed_barcode(M, pi; axes=nothing, strict=false, drop_zeros=true,
                             tol=0, max_span=nothing, axes_policy=:as_given,
                             max_axis_len=0, rq_cache=nothing, keep_endpoints=true,
                             method=:auto, bulk_max_elems=20_000_000)

Compute the rectangle signed barcode of a QQ-valued Z^n module `M` using a `ZnEncodingMap` `pi`.

The barcode is the Mobius inversion of the rank invariant restricted to the lattice grid
`axes` (one integer coordinate vector per dimension).

Axis selection:
- If `axes === nothing`, use `axes_from_encoding(pi)` (encoding-derived "critical" values).
- Otherwise `axes` is normalized to sorted unique integer vectors, then optionally modified by
  `axes_policy`:
    * `:as_given`  keep the provided axes.
    * `:encoding`  intersect with encoding-derived axes (and keep endpoints if requested).
    * `:coarsen`   uniformly downsample each axis to length <= `max_axis_len`.

Caching:
- Pass `rq_cache = RankQueryCache(pi)` to reuse cached `locate(pi, x)` values and cached region-pair
  ranks `rank_map(M, a, b)` across calls.

Algorithm choices:
- `method=:bulk` evaluates the rank function once per comparable pair on the grid, stores it in a
  `prod(length.(axes))^2` array, and performs one mixed Mobius inversion (finite differences).
  This avoids the `2^(2n)` rank evaluations per rectangle of the local formula and is usually
  much faster on moderate grids.
- `method=:local` uses the original inclusion-exclusion formula and can be preferable when `max_span`
  is set (because it avoids building the full rank array).
- `method=:auto` (default) uses `:bulk` when `max_span === nothing` and the rank array would have at
  most `bulk_max_elems` entries.

`strict=false` treats grid points not found in the encoding as rank 0.
"""
function rectangle_signed_barcode(M::PModule, pi::ZnEncodingMap;
                                  axes::Union{Nothing,Tuple}=nothing,
                                  strict::Bool=false,
                                  drop_zeros::Bool=true,
                                  tol::Int=0,
                                  max_span=nothing,
                                  axes_policy::Symbol=:as_given,
                                  max_axis_len::Int=0,
                                  rq_cache::Union{Nothing,RankQueryCache}=nothing,
                                  keep_endpoints::Bool=true,
                                  method::Symbol=:auto,
                                  bulk_max_elems::Int=20_000_000)

    axesN = axes === nothing ? axes_from_encoding(pi) : _normalize_axes(axes)

    if axes !== nothing && axes_policy != :as_given
        enc_axes = axes_from_encoding(pi)
        if axes_policy == :encoding
            axesN = _axes_policy_encoding(axesN, enc_axes; keep_endpoints=keep_endpoints)
        elseif axes_policy == :coarsen
            axesN = ntuple(i -> coarsen_axis(enc_axes[i], max_axis_len; keep_endpoints=keep_endpoints),
                           pi.n)
        else
            error("Unknown axes_policy=$(axes_policy)")
        end
    end

    if rq_cache === nothing
        rq_cache = RankQueryCache(pi)
    else
        rq_cache.pi === pi || error("rectangle_signed_barcode: rq_cache.pi must match pi")
        rq_cache.n == pi.n || error("rectangle_signed_barcode: rq_cache has wrong dimension")
    end

    cc = _cover_cache(M.Q)

    region_idx(g) = get!(rq_cache.loc_cache, g) do
        locate(pi, g)
    end

    function rank_ab(a::Int, b::Int)
        return get!(rq_cache.rank_cache, (a, b)) do
            rank_map(M, a, b; cache=cc)
        end
    end

    axesN isa NTuple{pi.n,Vector{Int}} || error("axes length mismatch: expected pi.n axes")

    # Decide whether to use the bulk (dense rank array + Mobius) or local (per-rectangle) method.
    meth = _choose_rectangle_signed_barcode_method(
        method, axesN; max_span=max_span, bulk_max_elems=bulk_max_elems
    )

    # -------------------------------------------------------------------------
    # Bulk method: precompute locate(pi, x) on the grid once, then compute ranks
    # on all comparable pairs, then perform one mixed Mobius inversion.
    # -------------------------------------------------------------------------
    if meth == :bulk
        span = max_span === nothing ? nothing :
            (max_span isa Integer ? ntuple(_ -> Int(max_span), pi.n) :
                                   ntuple(i -> Int(max_span[i]), pi.n))

        dims = ntuple(i -> length(axesN[i]), pi.n)
        CI = CartesianIndices(dims)

        # Grid point -> region index a = locate(pi, x).  This saves two Dict lookups
        # per rank query in the O(M^2) bulk fill loop.
        reg = Array{Int,pi.n}(undef, dims...)
        for pCI in CI
            p = pCI.I
            x = ntuple(Val(pi.n)) do k
                @inbounds axesN[k][p[k]]
            end
            a = region_idx(x)
            if strict && a == 0
                error("rectangle_signed_barcode: point not found in encoding")
            end
            @inbounds reg[pCI] = a
        end

        # Dense rank array r[p,q] for comparable (p <= q) pairs.
        r = zeros(Int, (dims..., dims...))
        for pCI in CI
            a = @inbounds reg[pCI]
            p = pCI.I
            q_ranges = ntuple(k -> p[k]:dims[k], pi.n)
            for qCI in CartesianIndices(q_ranges)
                b = @inbounds reg[qCI]
                if a == 0 || b == 0
                    @inbounds r[pCI, qCI] = 0
                else
                    @inbounds r[pCI, qCI] = rank_ab(a, b)
                end
            end
        end

        # Interval-poset Mobius inversion: p dims normal, q dims reversed.
        reverse_axis = ntuple(d -> d > pi.n, 2 * pi.n)
        _mobius_inversion_product_chains_mixed!(r, reverse_axis)

        rects = Rect{pi.n}[]
        weights = Int[]

        for pCI in CI
            p = pCI.I
            q_ranges = ntuple(k -> p[k]:dims[k], pi.n)
            for qCI in CartesianIndices(q_ranges)
                q = qCI.I
                w = @inbounds r[pCI, qCI]

                if span !== nothing
                    bad = false
                    @inbounds for k in 1:pi.n
                        if axesN[k][q[k]] - axesN[k][p[k]] > span[k]
                            bad = true
                            break
                        end
                    end
                    bad && continue
                end

                if drop_zeros && (abs(w) <= tol)
                    continue
                end

                lo = ntuple(k -> axesN[k][p[k]], pi.n)
                hi = ntuple(k -> axesN[k][q[k]], pi.n)
                push!(rects, Rect{pi.n}(lo, hi))
                push!(weights, w)
            end
        end

        return RectSignedBarcode{pi.n,Int}(axesN, rects, weights)
    end

    # -------------------------------------------------------------------------
    # Local method: keep the original inclusion-exclusion logic (but cached).
    # -------------------------------------------------------------------------
    function rank_idx_cached(p::NTuple{N,Int}, q::NTuple{N,Int}) where {N}
        x = ntuple(Val(N)) do i
            @inbounds axesN[i][p[i]]
        end
        y = ntuple(Val(N)) do i
            @inbounds axesN[i][q[i]]
        end
        a = region_idx(x)
        b = region_idx(y)
        if a == 0 || b == 0
            if strict
                error("Point not found in encoding")
            end
            return 0
        end
        return rank_ab(a, b)
    end

    return _rectangle_signed_barcode_local(
        rank_idx_cached, axesN; drop_zeros=drop_zeros, tol=tol, max_span=max_span
    )
end



"""
    rank_from_signed_barcode(sb, p, q)

Evaluate the rank function reconstructed from a rectangle signed barcode.

Inputs `p` and `q` are points in Z^N (represented as NTuples) with `p <= q`.
The value returned is

    sum_{rect in sb} weight(rect) * 1[ rect.lo <= p and q <= rect.hi ].

Warning:
This reconstruction is exact on the grid used to compute `sb`. Off-grid evaluation is meaningful
when the underlying rank function is constant on the corresponding encoded cells.
"""
function rank_from_signed_barcode(sb::RectSignedBarcode{N}, p::NTuple{N,Int}, q::NTuple{N,Int}) where {N}
    _tuple_leq(p, q) || error("rank_from_signed_barcode: expected p <= q")
    r = 0
    for (rect, w) in zip(sb.rects, sb.weights)
        if _tuple_leq(rect.lo, p) && _tuple_leq(q, rect.hi)
            r += w
        end
    end
    return r
end

"""
    rectangles_from_grid(axes; max_span=nothing)

Enumerate all grid-aligned rectangles with corners in the given axes.

The order matches the iteration order used by `rectangle_signed_barcode`:
lexicographic in the lower corner, and lexicographic in the upper corner
restricted by `lo <= hi` (coordinatewise). This is useful for debugging and for
building feature matrices indexed by rectangles.
"""
function rectangles_from_grid(axes::NTuple{N,<:AbstractVector{<:Integer}}; max_span=nothing) where {N}
    ax = ntuple(i -> collect(Int, axes[i]), N)
    span = max_span === nothing ? nothing :
        (max_span isa Integer ? ntuple(_ -> Int(max_span), N) :
                               ntuple(i -> Int(max_span[i]), N))
    dims = ntuple(i -> length(ax[i]), N)
    rects = Rect{N}[]
    CI = CartesianIndices(dims)
    for pCI in CI
        p = pCI.I
        q_ranges = ntuple(k -> p[k]:dims[k], N)
        for qCI in CartesianIndices(q_ranges)
            q = qCI.I
            if span !== nothing
                bad = false
                @inbounds for k in 1:N
                    if ax[k][q[k]] - ax[k][p[k]] > span[k]
                        bad = true
                        break
                    end
                end
                bad && continue
            end
            lo = ntuple(k -> ax[k][p[k]], N)
            hi = ntuple(k -> ax[k][q[k]], N)
            push!(rects, Rect{N}(lo, hi))
        end
    end
    return rects
end

"""
    rectangle_signed_barcode_rank(sb; zero_noncomparable=true)

Reconstruct the rank function on the underlying grid from a rectangle signed barcode.

Returns a 2N-dimensional Int array `R` with indices `(p..., q...)`, where `p` and `q`
range over grid indices. For comparable pairs `p <= q`, `R[p..., q...]` equals the
rank invariant reconstructed from `sb`.

Implementation note:
This is the inverse of the Mobius inversion used by `rectangle_signed_barcode`, and is
implemented via a mixed-orientation prefix-sum transform (no per-rectangle scanning).
"""
function rectangle_signed_barcode_rank(sb::RectSignedBarcode{N};
                                       zero_noncomparable::Bool=true) where {N}
    axes = sb.axes
    dims = ntuple(i -> length(axes[i]), N)

    # Coordinate-to-index maps for each axis (axes are assumed unique and sorted).
    idx = ntuple(i -> Dict{Int,Int}(axes[i][j] => j for j in 1:dims[i]), N)

    # Put weights onto a dense (lo,hi) array indexed by grid indices.
    w = zeros(Int, (dims..., dims...))
    for (rect, wt) in zip(sb.rects, sb.weights)
        p = ntuple(i -> idx[i][rect.lo[i]], N)
        q = ntuple(i -> idx[i][rect.hi[i]], N)
        pCI = CartesianIndex(p)
        qCI = CartesianIndex(q)
        @inbounds w[pCI, qCI] += Int(wt)
    end

    # Invert Mobius inversion: prefix sums in p, reverse prefix sums in q.
    reverse_axis = ntuple(d -> d > N, 2N)
    _prefix_sum_product_chains_mixed!(w, reverse_axis)

    if zero_noncomparable
        # Convention: rank invariant is only meaningful for comparable pairs p <= q.
        # We zero out noncomparable index pairs for a cleaner array representation.
        CI = CartesianIndices(dims)
        for pCI in CI
            p = pCI.I
            for qCI in CI
                q = qCI.I
                comparable = true
                @inbounds for k in 1:N
                    if p[k] > q[k]
                        comparable = false
                        break
                    end
                end
                if !comparable
                    @inbounds w[pCI, qCI] = 0
                end
            end
        end
    end

    return w
end



"""
    truncate_signed_barcode(sb; max_terms=nothing, min_abs_weight=1)

Return a truncated signed barcode by discarding rectangles with small weights.

- Keep only rectangles with `abs(weight) >= min_abs_weight`.
- If `max_terms` is set, keep at most that many rectangles, chosen by decreasing `abs(weight)`.

This is a simple "module approximation (MMA) style" step: the full signed barcode is exact
(on the chosen grid), while truncation yields a controlled-size approximation suitable for
feature extraction pipelines.
"""
function truncate_signed_barcode(sb::RectSignedBarcode{N}; max_terms=nothing, min_abs_weight::Int=1) where {N}
    keep = [i for i in eachindex(sb.weights) if abs(sb.weights[i]) >= min_abs_weight]
    if max_terms !== nothing && length(keep) > max_terms
        sort!(keep, by=i -> abs(sb.weights[i]), rev=true)
        keep = keep[1:max_terms]
        sort!(keep)
    end
    rects = sb.rects[keep]
    weights = sb.weights[keep]
    return RectSignedBarcode{N,eltype(weights)}(sb.axes, rects, weights)
end

"""
    rectangle_signed_barcode_kernel(sb1, sb2; kind=:linear, sigma=1.0)

Kernels for rectangle signed barcodes.

- `kind=:linear` computes the dot product of rectangle weights on exactly matching rectangles.
- `kind=:gaussian` uses a Gaussian kernel on rectangle endpoints in R^(2N):

      k(sb1,sb2) = sum_{i,j} w1[i] * w2[j] * exp(-||emb(rect1[i]) - emb(rect2[j])||^2 / (2*sigma^2))

  where `emb(rect) = (lo..., hi...)`.

The Gaussian version is a standard RKHS embedding of signed measures.
"""
function rectangle_signed_barcode_kernel(sb1::RectSignedBarcode{N}, sb2::RectSignedBarcode{N}; kind::Symbol=:linear, sigma::Real=1.0) where {N}
    if kind == :linear
        d2 = Dict{Rect{N},Int}()
        for (r,w) in zip(sb2.rects, sb2.weights)
            d2[r] = get(d2, r, 0) + w
        end
        s = 0.0
        for (r,w1) in zip(sb1.rects, sb1.weights)
            w2 = get(d2, r, 0)
            s += w1 * w2
        end
        return s
    elseif kind == :gaussian
        sig2 = float(sigma)^2
        sig2 > 0 || error("rectangle_signed_barcode_kernel: sigma must be > 0")
        s = 0.0
        for (r1,w1) in zip(sb1.rects, sb1.weights)
            emb1 = (r1.lo..., r1.hi...)
            for (r2,w2) in zip(sb2.rects, sb2.weights)
                emb2 = (r2.lo..., r2.hi...)
                d2 = 0.0
                for k in 1:(2*N)
                    dk = float(emb1[k] - emb2[k])
                    d2 += dk * dk
                end
                s += float(w1) * float(w2) * exp(-d2 / (2 * sig2))
            end
        end
        return s
    else
        error("rectangle_signed_barcode_kernel: unknown kind $(kind)")
    end
end

"""
    rectangle_signed_barcode_image(sb; xs=nothing, ys=nothing, sigma=1.0, mode=:center)

A simple 2D "image-like" vectorization of a rectangle signed barcode.

Each rectangle contributes its signed weight to a 2D grid via a Gaussian bump centered at:
- `mode=:center` (default): the rectangle center (average of `lo` and `hi`)
- `mode=:lo`: the lower corner `lo`
- `mode=:hi`: the upper corner `hi`

This is intentionally lightweight and meant for data-analysis pipelines.
For N != 2, this function errors (a higher-dimensional tensorization would be needed).
"""
function rectangle_signed_barcode_image(sb::RectSignedBarcode{2}; xs=nothing, ys=nothing, sigma::Real=1.0, mode::Symbol=:center)
    xs === nothing && (xs = sb.axes[1])
    ys === nothing && (ys = sb.axes[2])
    sig2 = float(sigma)^2
    sig2 > 0 || error("rectangle_signed_barcode_image: sigma must be > 0")

    img = zeros(Float64, length(xs), length(ys))

    for (rect, w) in zip(sb.rects, sb.weights)
        cx, cy = if mode == :center
            ((rect.lo[1] + rect.hi[1]) / 2, (rect.lo[2] + rect.hi[2]) / 2)
        elseif mode == :lo
            (rect.lo[1], rect.lo[2])
        elseif mode == :hi
            (rect.hi[1], rect.hi[2])
        else
            error("rectangle_signed_barcode_image: unknown mode $(mode)")
        end

        for (ix, x) in pairs(xs)
            dx = float(x - cx)
            for (iy, y) in pairs(ys)
                dy = float(y - cy)
                img[ix, iy] += float(w) * exp(-(dx * dx + dy * dy) / (2 * sig2))
            end
        end
    end

    return img
end

"""
    mma_decomposition(M, pi; method=:rectangles,
                      rect_kwargs=NamedTuple(), slice_kwargs=NamedTuple(), mpp_kwargs=NamedTuple(),
                      truncate=true, max_terms=nothing, min_abs_weight=1,
                      axes=nothing, axes_policy=:encoding, max_axis_len=64,
                      euler_drop_zeros=true, euler_max_terms=0, euler_min_abs_weight=0)

A small "MMA style" front-end.

This is not a full MMA solver. It provides the core algebraic objects commonly used in
multi-parameter matching distance (MMA) pipelines:
  - rectangle signed measures (Mobius inversion of the rank invariant),
  - directional slice barcodes,
  - optional Euler characteristic objects on a grid, and
  - optional multiparameter persistence images (MPPI; Carriere et al.).

Methods

- `method=:rectangles` returns a `RectSignedBarcode` obtained by Mobius inversion of the rank
  invariant, optionally truncated via `truncate_signed_barcode`.
- `method=:slices` returns the `NamedTuple` produced by `slice_barcodes(M, pi; ...)`.
  Note: `slice_barcodes` requires `directions` and `offsets`; pass them via `slice_kwargs`.
- `method=:both` returns `(rectangles=..., slices=...)`.
- `method=:euler` returns `(euler_surface=..., euler_signed_measure=...)`.
- `method=:mpp_image` returns a Carriere multiparameter persistence image (`MPPImage`).
  Note: this requires a 2-parameter (2D) encoding and a QQ-valued module.
- `method=:all` returns `(rectangles=..., slices=..., euler_surface=..., euler_signed_measure=...)`.

Keyword argument routing

- `rect_kwargs` is forwarded to `rectangle_signed_barcode(M, pi; ...)`.
- `slice_kwargs` is forwarded to `slice_barcodes(M, pi; ...)`.
- `mpp_kwargs` is forwarded to `mpp_image(M, pi; ...)` (only used for `method=:mpp_image`).

Euler-related keywords

The Euler outputs are computed on a rectangular grid controlled by `(axes, axes_policy, max_axis_len)`.
Truncation of the Euler signed measure is controlled separately via the `euler_*` keywords.

If you want Euler output for an object that is not a `PModule` or for an encoding that is not
a `ZnEncodingMap`, use the generic method

    mma_decomposition(obj, pi; method=:euler, axes=..., axes_policy=..., max_axis_len=...)

which returns only the Euler outputs.
"""
function mma_decomposition(M::PModule, pi::ZnEncodingMap;
        method::Symbol=:rectangles,
        rect_kwargs::NamedTuple=NamedTuple(),
        slice_kwargs::NamedTuple=NamedTuple(),
        mpp_kwargs::NamedTuple=NamedTuple(),
        truncate::Bool=true,
        max_terms=nothing,
        min_abs_weight::Int=1,
        axes=nothing,
        axes_policy::Symbol=:encoding,
        max_axis_len::Int=64,
        euler_drop_zeros::Bool=true,
        euler_max_terms::Int=0,
        euler_min_abs_weight::Real=0)

    # Compute only what is requested. This keeps the wrapper cheap for the
    # common case `method=:rectangles`.
    want_rect = (method === :rectangles) || (method === :both) || (method === :all)
    want_slices = (method === :slices) || (method === :both) || (method === :all)
    want_euler = (method === :euler) || (method === :all)
    want_mpp = (method === :mpp_image)

    if want_mpp && !(M isa PModule{QQ})
        throw(ArgumentError("method=:mpp_image requires a QQ-valued module (PModule{QQ})."))
    end

    if want_mpp && pi.n != 2
        throw(ArgumentError("method=:mpp_image requires a 2-parameter (2D) encoding; got n=$(pi.n)."))
    end

    rect_sb = nothing
    if want_rect
        rect_sb = rectangle_signed_barcode(M, pi; rect_kwargs...)
        if truncate
            rect_sb = truncate_signed_barcode(rect_sb; max_terms=max_terms, min_abs_weight=min_abs_weight)
        end
    end

    slice_data = nothing
    if want_slices
        slice_data = slice_barcodes(M, pi; slice_kwargs...)
    end

    euler_surf = nothing
    euler_pm = nothing
    if want_euler
        euler_surf = euler_surface(M, pi; axes=axes, axes_policy=axes_policy, max_axis_len=max_axis_len)
        euler_pm = euler_signed_measure(M, pi;
                                        axes=axes,
                                        axes_policy=axes_policy,
                                        max_axis_len=max_axis_len,
                                        drop_zeros=euler_drop_zeros,
                                        max_terms=euler_max_terms,
                                        min_abs_weight=euler_min_abs_weight)
    end

    mpp_img = nothing
    if want_mpp
        mpp_img = mpp_image(M, pi; mpp_kwargs...)
    end

    if method === :rectangles
        return rect_sb
    elseif method === :slices
        return slice_data
    elseif method === :both
        return (rectangles=rect_sb, slices=slice_data)
    elseif method === :euler
        return (euler_surface=euler_surf, euler_signed_measure=euler_pm)
    elseif method === :mpp_image
        return mpp_img
    elseif method === :all
        return (rectangles=rect_sb,
                slices=slice_data,
                euler_surface=euler_surf,
                euler_signed_measure=euler_pm)
    else
        throw(ArgumentError("Unknown method: $method. Use :rectangles, :slices, :both, :euler, :mpp_image, or :all."))
    end
end

"""
    mma_decomposition(M::PModule, pi; method=:euler, mpp_kwargs=NamedTuple(),
                      axes=nothing, axes_policy=:encoding, max_axis_len=64,
                      euler_drop_zeros=true, euler_max_terms=0, euler_min_abs_weight=0)

A lightweight wrapper for Euler and MPPI output when `pi` is not a `ZnEncodingMap`.

This method exists so that you can write

    mma_decomposition(M, pi; method=:mpp_image, mpp_kwargs=(...))

for 2D encodings coming from backends such as `PLBackend`.

Supported methods

- `method=:euler` returns `(euler_surface=..., euler_signed_measure=...)`.
- `method=:mpp_image` returns an `MPPImage` via `mpp_image(M, pi; mpp_kwargs...)`.
  Note: this requires a 2-parameter (2D) encoding and a `PModule{QQ}`.

Other `method` values are not supported here (use the `ZnEncodingMap` method if you
need rectangles or slice barcodes).
"""
function mma_decomposition(M::PModule, pi;
        method::Symbol=:euler,
        mpp_kwargs::NamedTuple=NamedTuple(),
        axes=nothing,
        axes_policy::Symbol=:encoding,
        max_axis_len::Int=64,
        euler_drop_zeros::Bool=true,
        euler_max_terms::Int=0,
        euler_min_abs_weight::Real=0)

    if method === :euler
        euler_surf = euler_surface(M, pi; axes=axes, axes_policy=axes_policy, max_axis_len=max_axis_len)
        euler_pm = euler_signed_measure(M, pi;
                                        axes=axes,
                                        axes_policy=axes_policy,
                                        max_axis_len=max_axis_len,
                                        drop_zeros=euler_drop_zeros,
                                        max_terms=euler_max_terms,
                                        min_abs_weight=euler_min_abs_weight)
        return (euler_surface=euler_surf, euler_signed_measure=euler_pm)
    elseif method === :mpp_image
        if !(M isa PModule{QQ})
            throw(ArgumentError("method=:mpp_image requires a QQ-valued module (PModule{QQ})."))
        end
        if hasproperty(pi, :n)
            n = getproperty(pi, :n)
            if n != 2
                throw(ArgumentError("method=:mpp_image requires a 2-parameter (2D) encoding; got n=$n."))
            end
        end
        return mpp_image(M, pi; mpp_kwargs...)
    else
        throw(ArgumentError("Unknown method: $method. Use :euler or :mpp_image."))
    end
end


"""
    mma_decomposition(obj, pi; method=:euler, axes=nothing, axes_policy=:encoding, max_axis_len=64,
                      euler_drop_zeros=true, euler_max_terms=0, euler_min_abs_weight=0)

Generic Euler-only front-end.

This method is useful for objects such as `ModuleCochainComplex` (Euler surface = alternating sum of
chain-group dimensions) or for encodings `pi` that are not `ZnEncodingMap`s.

Only `method=:euler` is supported for this signature.
"""
function mma_decomposition(obj, pi;
        method::Symbol=:euler,
        axes=nothing,
        axes_policy::Symbol=:encoding,
        max_axis_len::Int=64,
        euler_drop_zeros::Bool=true,
        euler_max_terms::Int=0,
        euler_min_abs_weight::Real=0)

    method === :euler || throw(ArgumentError("mma_decomposition(obj,pi): only method=:euler is supported for this signature"))

    euler_surf = euler_surface(obj, pi; axes=axes, axes_policy=axes_policy, max_axis_len=max_axis_len)
    euler_pm = euler_signed_measure(obj, pi;
                                    axes=axes,
                                    axes_policy=axes_policy,
                                    max_axis_len=max_axis_len,
                                    drop_zeros=euler_drop_zeros,
                                    max_terms=euler_max_terms,
                                    min_abs_weight=euler_min_abs_weight)
    return (euler_surface=euler_surf, euler_signed_measure=euler_pm)
end



# ----- Restricted Hilbert function --------------------------------------------

"""
    restricted_hilbert(M) -> Vector{Int}

Restricted Hilbert function on a finite poset: the dimension surface
`q |-> dim(M_q)` represented as a vector indexed by poset vertices.

Methods:
- `restricted_hilbert(M::PModule{QQ})` returns `copy(M.dims)`
- `restricted_hilbert(H::FringeModule{QQ})` computes fiber dimensions via
  `fiber_dimension(H, q)` for each vertex
"""
restricted_hilbert(M::PModule{QQ}) = copy(M.dims)

function restricted_hilbert(H::FringeModule{QQ})
    return [fiber_dimension(H, q) for q in 1:H.P.n]
end


"""
    restricted_hilbert(M, pi, x; strict=true) -> Int

Evaluate the restricted Hilbert function at a point `x` in the original domain,
by first locating its region via `locate(pi, x)`.

If `strict=true` and `locate` returns 0, an error is thrown.
If `strict=false`, unknown regions return 0.
"""
function restricted_hilbert(M::PModule{QQ}, pi, x; strict::Bool=true)::Int
    p = locate(pi, x)
    if p == 0
        strict && error("restricted_hilbert: locate(pi, x) returned 0 (unknown region)")
        return 0
    end
    (1 <= p <= length(M.dims)) || error("restricted_hilbert: locate returned out-of-range index")
    return M.dims[p]
end

function restricted_hilbert(H::FringeModule{QQ}, pi, x; strict::Bool=true)::Int
    Mp = pmodule_from_fringe(H)
    return restricted_hilbert(Mp, pi, x; strict=strict)
end

"""
    restricted_hilbert(dims::AbstractVector{<:Integer}) -> Vector{Int}

Convenience overload: interpret `dims` as an already computed restricted Hilbert
function (dimension per region). This makes it easy to use the "module-level"
summary/statistics functions without constructing a module object.
"""
function restricted_hilbert(dims::AbstractVector{<:Integer})
    return [Int(d) for d in dims]
end


# ----- Distances between restricted Hilbert functions --------------------------

function _normalize_weights(n::Int, weights)
    if weights === nothing
        return nothing
    elseif weights isa AbstractVector
        length(weights) == n || error("weights must have length $n")
        return weights
    elseif weights isa AbstractDict
        w = ones(Float64, n)
        for (k, v) in weights
            (1 <= k <= n) || error("weight key out of range: $k")
            w[Int(k)] = float(v)
        end
        return w
    else
        error("weights must be nothing, a vector, or a dictionary")
    end
end


"""
    hilbert_distance(M, N; norm=:L1, weights=nothing) -> Real

Compute a fast distance between two modules based on their restricted Hilbert
functions (dimension surfaces) on a fixed finite encoding.

Inputs:
* `M`, `N`: `PModule{QQ}` or `FringeModule{QQ}` on the same finite poset.
* `norm`: one of `:L1`, `:L2`, `:Linf`.
* `weights`: optional per-region weights.

If `weights` is provided, it should be either:
* a vector `w` of length `P.n`, or
* a dictionary mapping region indices to weights (others default to 1).

The weighted norms are computed as:
* `L1`: sum_i w[i] * abs(hM[i] - hN[i])
* `L2`: sqrt( sum_i w[i] * (hM[i] - hN[i])^2 )
* `Linf`: max_i w[i] * abs(hM[i] - hN[i])

To weight by region size for a particular encoding map `pi`, use
`weights = region_weights(pi; box=...)` when available.
"""
function hilbert_distance(M, N; norm::Symbol=:L1, weights=nothing)
    hM = restricted_hilbert(M)
    hN = restricted_hilbert(N)
    length(hM) == length(hN) || error("hilbert_distance: modules must live on the same finite poset")
    n = length(hM)

    w = _normalize_weights(n, weights)

    if norm == :L1
        if w === nothing
            s = 0
            for i in 1:n
                s += abs(hM[i] - hN[i])
            end
            return s
        end
        s = zero(promote_type(eltype(w), Int))
        for i in 1:n
            s += w[i] * abs(hM[i] - hN[i])
        end
        return s
    elseif norm == :L2
        acc = 0.0
        if w === nothing
            for i in 1:n
                d = float(hM[i] - hN[i])
                acc += d * d
            end
        else
            for i in 1:n
                d = float(hM[i] - hN[i])
                acc += float(w[i]) * d * d
            end
        end
        return sqrt(acc)
    elseif norm == :Linf
        if w === nothing
            m = 0
            for i in 1:n
                m = max(m, abs(hM[i] - hN[i]))
            end
            return m
        end
        m = 0.0
        for i in 1:n
            m = max(m, float(w[i]) * float(abs(hM[i] - hN[i])))
        end
        return m
    else
        error("hilbert_distance: norm must be :L1, :L2, or :Linf")
    end
end


# Convenience wrapper: compute weights from an encoding map when available.
function hilbert_distance(M, N, pi; norm::Symbol=:L1, box=nothing, kwargs...)
    w = region_weights(pi; box=box)
    return hilbert_distance(M, N; norm=norm, weights=w, kwargs...)
end


# =============================================================================
# Euler characteristic surface and Euler signed-measure pipeline
# =============================================================================

"""
    euler_characteristic_surface(obj, pi; axes=nothing, axes_policy=:as_given,
                                 max_axis_len=64, threads=(Threads.nthreads() > 1))

Compute the Euler characteristic surface of `obj` on a rectangular grid.

- If `obj` is a `PModule`, this is the same as the restricted Hilbert surface
  (dimension surface) on the grid.
- If `obj` is a `ModuleCochainComplex`, this is the alternating sum of the
  chain-group dimensions:

      chi(x) = sum_t (-1)^t dim(C^t_x),

  evaluated on the grid points, where `x` is located into a constant region
  using `locate(pi, x)`.

Threading:
- The surface computation is embarrassingly parallel over grid points.
- If `threads=true` and Julia has > 1 thread, the surface fill loop uses
  `Threads.@threads`.
- When `locate` requires a vector input, this routine uses per-thread scratch
  vectors to avoid races.

Axes policies mirror `rectangle_signed_barcode`:
- `:as_given`  use `axes` directly (must be provided)
- `:encoding`  restrict/intersect `axes` with `axes_from_encoding(pi)` (or use encoding axes if axes=nothing)
- `:coarsen`   downsample axes until each axis length <= max_axis_len

Example (Euler surface -> Euler signed measure -> distance/kernel)


# Suppose `obj` is a `PModule` (degree-0 Euler = dimension) or a `ModuleCochainComplex`
# (Euler = alternating sum of chain-group dimensions), and `pi` is an encoding map.
#
# You must choose a grid (a tuple of coordinate axes). In practice, a good default is
# `axes_policy=:encoding` with `axes=nothing`, which uses `axes_from_encoding(pi)`.
axes = ([1, 2, 3],)  # toy 1D grid for illustration; in 2D use (xs, ys)

# 1) Evaluate Euler characteristic on the grid.
surf = euler_surface(obj, pi; axes=axes, axes_policy=:as_given)

# 2) Mobius inversion on the product-of-chains grid yields a sparse signed point measure.
pm = euler_signed_measure(obj, pi; axes=axes, axes_policy=:as_given)

# 3) Inversion is exact on the chosen grid:
surf2 = surface_from_point_signed_measure(pm)  # should equal `surf`

# 4) A cheap distance between objects can be computed from the surfaces.
d = euler_distance(obj1, obj2, pi; axes=axes, axes_policy=:as_given, ord=1)

# 5) Or turn signed measures into a kernel value.
pm1 = euler_signed_measure(obj1, pi; axes=axes, axes_policy=:as_given)
pm2 = euler_signed_measure(obj2, pi; axes=axes, axes_policy=:as_given)
k = point_signed_measure_kernel(pm1, pm2; kind=:gaussian, sigma=1.0)

# 6) The MMA-style wrapper can compute Euler outputs too.
out = mma_decomposition(obj1, pi; method=:euler, axes=axes, axes_policy=:as_given)
surf1 = out.euler_surface
pm1b  = out.euler_signed_measure
"""
function euler_characteristic_surface(
    obj,
    pi::EncodingMap;
    axes = nothing,
    axes_policy = :as_given,
    max_axis_len = 64,
    threads::Bool = (Threads.nthreads() > 1),
)
    N = dim(pi)
    if N != 2
        throw(ArgumentError("euler_characteristic_surface currently expects dim(pi) == 2"))
    end

    # Get module and precompute chi per vertex.
    M = obj isa PModule{QQ} ? obj : euler_characteristic(obj)
    chi_dims = [dim_at(M, u) for u in 1:M.Q.n]

    ax = if axes === nothing
        default_axes(pi; axes_policy = axes_policy, max_axis_len = max_axis_len)
    else
        axes
    end

    surf = zeros(Int, length(ax[1]), length(ax[2]))

    # We can pass tuples to locate on some maps (fast path).
    use_tuple = locate_supports_tuple(pi)

    CI = CartesianIndices(surf)

    if use_tuple
        if threads && Threads.nthreads() > 1
            Threads.@threads for k in eachindex(surf)
                I = CI[k]
                x = (ax[1][I[1]], ax[2][I[2]])
                u = locate(pi, x)
                @inbounds surf[I] = (u == 0) ? 0 : chi_dims[u]
            end
        else
            @inbounds for I in CI
                x = (ax[1][I[1]], ax[2][I[2]])
                u = locate(pi, x)
                surf[I] = (u == 0) ? 0 : chi_dims[u]
            end
        end
        return surf
    end

    # Vector locate path: needs scratch space. Make it per-thread for safety.
    if threads && Threads.nthreads() > 1
        xv_by_thread = [Vector{Float64}(undef, N) for _ in 1:Threads.nthreads()]
        Threads.@threads for k in eachindex(surf)
            I = CI[k]
            xv = xv_by_thread[Threads.threadid()]
            @inbounds begin
                xv[1] = float(ax[1][I[1]])
                xv[2] = float(ax[2][I[2]])
            end
            u = locate(pi, xv)
            @inbounds surf[I] = (u == 0) ? 0 : chi_dims[u]
        end
        return surf
    end

    xv = Vector{Float64}(undef, N)
    @inbounds for I in CI
        xv[1] = float(ax[1][I[1]])
        xv[2] = float(ax[2][I[2]])
        u = locate(pi, xv)
        surf[I] = (u == 0) ? 0 : chi_dims[u]
    end
    return surf
end


# Short alias (common terminology in multipers literature).
euler_surface(obj, pi; kwargs...) = euler_characteristic_surface(obj, pi; kwargs...)

# Internal: Euler dims on encoding elements.
_euler_dims(M::PModule) = M.dims

function _euler_dims(C::ModuleCochainComplex)
    # alternating sum over degrees t = C.tmin + (i-1)
    # Result is a vector indexed by encoding poset elements (same indexing as module dims).
    # We assume all terms live over the same encoding poset.
    n = length(C.terms[1].dims)
    chi = zeros(Int, n)
    for (i, M) in enumerate(C.terms)
        t = C.tmin + (i - 1)
        sgn = isodd(t) ? -1 : 1
        @inbounds for u in 1:n
            chi[u] += sgn * M.dims[u]
        end
    end
    return chi
end

"""
    euler_signed_measure(obj, pi; axes, axes_policy=:encoding, max_axis_len=64,
                         drop_zeros=true, max_terms=0, min_abs_weight=0)

Compute the Euler signed-measure decomposition:

1. Compute Euler characteristic surface on a grid.
2. Mobius-invert it (product-of-chains) to get a signed point measure.

Returns a `PointSignedMeasure`.
"""
function euler_signed_measure(obj, pi;
                             axes=nothing,
                             axes_policy::Symbol=:encoding,
                             max_axis_len::Int=64,
                             drop_zeros::Bool=true,
                             max_terms::Int=0,
                             min_abs_weight::Real=0)

    surf = euler_characteristic_surface(obj, pi;
                                        axes=axes,
                                        axes_policy=axes_policy,
                                        max_axis_len=max_axis_len)

    # Determine axes used (must match surface dimensions).
    ax = if axes_policy === :as_given
        _normalize_axes_real(axes)
    elseif axes_policy === :encoding
        axes === nothing ? axes_from_encoding(pi) : restrict_axes_to_encoding(axes, pi)
    else
        # for :coarsen, recompute same selection logic as in surface function
        if axes === nothing
            a0 = axes_from_encoding(pi)
        else
            a0 = _normalize_axes_real(axes)
        end
        while maximum(length.(a0)) > max_axis_len
            a0 = ntuple(i -> coarsen_axis(a0[i]), length(a0))
        end
        a0
    end
    ax = _normalize_axes_real(ax)

    pm = point_signed_measure(surf, ax; drop_zeros=drop_zeros)
    if max_terms > 0 || min_abs_weight > 0
        pm = truncate_point_signed_measure(pm; max_terms=max_terms, min_abs_weight=min_abs_weight)
    end
    return pm
end

"""
    euler_distance(A, B; ord=1)

Lp distance between two Euler characteristic surfaces (arrays on the same grid).

If `ord == Inf`, returns the max norm.
"""
function euler_distance(A::AbstractArray, B::AbstractArray; ord::Real=1)
    size(A) == size(B) || throw(ArgumentError("surface sizes must match"))
    if ord == Inf
        m = 0.0
        @inbounds for I in eachindex(A)
            v = abs(float(A[I] - B[I]))
            if v > m
                m = v
            end
        end
        return m
    else
        ord >= 1 || throw(ArgumentError("ord must be >= 1 or Inf"))
        s = 0.0
        @inbounds for I in eachindex(A)
            s += abs(float(A[I] - B[I]))^ord
        end
        return s^(1/ord)
    end
end

"""
    euler_distance(obj1, obj2, pi; axes, axes_policy=:encoding, max_axis_len=64, ord=1)

Convenience wrapper: compute Euler surfaces on a grid and then take Lp distance.
"""
function euler_distance(obj1, obj2, pi;
                        axes=nothing,
                        axes_policy::Symbol=:encoding,
                        max_axis_len::Int=64,
                        ord::Real=1)
    A = euler_characteristic_surface(obj1, pi; axes=axes, axes_policy=axes_policy, max_axis_len=max_axis_len)
    B = euler_characteristic_surface(obj2, pi; axes=axes, axes_policy=axes_policy, max_axis_len=max_axis_len)
    return euler_distance(A, B; ord=ord)
end



# =============================================================================
# (3b) Global "module size" summaries built from region sizes
# =============================================================================

# Internal helper: interpret box=:auto uniformly.
@inline _resolve_box(pi, box) = (box === :auto ? window_box(pi) : box)

"""
    integrated_hilbert_mass(M, pi; box=nothing, weights=nothing, kwargs...) -> Real

Approximate the integral of the Hilbert function over a window.

In a finite encoding, the Hilbert function is constant on each region. If
`w_r = vol(R_r \\cap W)` denotes the region weight inside window W, then:

    \\int_W dim M(x) dx  \\approx  \\sum_r w_r * dim(M|_{R_r})

Arguments:
- `M`: typically a `PModule{QQ}` or `FringeModule{QQ}`.
- `pi`: an encoding map (any object implementing `region_weights(pi; ...)`).
- `box`: optional window `(a,b)` forwarded to `region_weights`.
- `weights`: optional precomputed weights vector.

`kwargs...` are forwarded to `region_weights` when `weights` is not supplied.
"""
function integrated_hilbert_mass(M, pi; box=nothing, weights=nothing, kwargs...)
    h = restricted_hilbert(M)
    bb = _resolve_box(pi, box)
    w = weights === nothing ? region_weights(pi; box=bb, kwargs...) : weights
    length(w) == length(h) || error("integrated_hilbert_mass: weights length does not match restricted_hilbert length")
    T = promote_type(eltype(w), Int)
    s = zero(T)
    for i in eachindex(h)
        s += w[i] * h[i]
    end
    return s
end

"""
    measure_by_dimension(M, pi; box=nothing, weights=nothing, kwargs...) -> Dict{Int,T}

"Histogram" of region sizes by module dimension.

Returns a dictionary mapping d -> total measure of {x in box : dim M(x) == d}.
Values have type `T = promote_type(eltype(weights), Int)`.
"""
function measure_by_dimension(M, pi; box=nothing, weights=nothing, kwargs...)
    h = restricted_hilbert(M)
    bb = _resolve_box(pi, box)
    w = weights === nothing ? region_weights(pi; box=bb, kwargs...) : weights
    length(w) == length(h) || error("measure_by_dimension: weights length does not match restricted_hilbert length")
    T = promote_type(eltype(w), Int)
    out = Dict{Int,T}()
    zT = zero(T)
    for i in eachindex(h)
        d = h[i]
        out[d] = get(out, d, zT) + w[i]
    end
    return out
end

"""
    support_measure(M, pi; box=nothing, weights=nothing, min_dim=1, kwargs...) -> Real

Total measure of the support (or high-rank) region:

    measure({x in box : dim M(x) >= min_dim}).

Default `min_dim=1` corresponds to the usual support {dim > 0}.
"""
function support_measure(M, pi; box=nothing, weights=nothing, min_dim::Int=1, kwargs...)
    h = restricted_hilbert(M)
    bb = _resolve_box(pi, box)
    w = weights === nothing ? region_weights(pi; box=bb, kwargs...) : weights
    length(w) == length(h) || error("support_measure: weights length does not match restricted_hilbert length")
    T = promote_type(eltype(w), Int)
    s = zero(T)
    for i in eachindex(h)
        if h[i] >= min_dim
            s += w[i]
        end
    end
    return s
end

"""
    dim_stats(M, pi; box=nothing, weights=nothing, kwargs...) -> NamedTuple

Basic region-weighted statistics of the dimension surface.

Returns:
- `total_measure`: sum of weights
- `integrated_mass`: sum(w_r * dim_r)
- `mean`: integrated_mass / total_measure
- `var`: weighted variance of dim_r

If `total_measure == 0`, `mean` and `var` are `NaN`.
"""
function dim_stats(M, pi; box=nothing, weights=nothing, kwargs...)
    h = restricted_hilbert(M)
    bb = _resolve_box(pi, box)
    w = weights === nothing ? region_weights(pi; box=bb, kwargs...) : weights
    length(w) == length(h) || error("dim_stats: weights length does not match restricted_hilbert length")
    T = promote_type(eltype(w), Int)
    total = zero(T)
    mass = zero(T)
    mass2 = zero(T)
    for i in eachindex(h)
        wi = w[i]
        di = h[i]
        total += wi
        mass += wi * di
        mass2 += wi * (di * di)
    end
    if total == zero(T)
        return (total_measure=total, integrated_mass=mass, mean=NaN, var=NaN)
    end
    total_f = float(total)
    mean = float(mass) / total_f
    mean2 = float(mass2) / total_f
    var = mean2 - mean * mean
    return (total_measure=total, integrated_mass=mass, mean=mean, var=var)
end

"""
    dim_norm(M, pi; p=2, box=nothing, weights=nothing, kwargs...) -> Float64

Compute an L^p-type norm of the dimension surface over the window.

For finite p:
    ( sum_r w_r * |dim_r|^p )^(1/p)

For p == Inf:
    max_{r : w_r > 0} |dim_r|
"""
function dim_norm(M, pi; p::Real=2, box=nothing, weights=nothing, kwargs...)
    h = restricted_hilbert(M)
    bb = _resolve_box(pi, box)
    w = weights === nothing ? region_weights(pi; box=bb, kwargs...) : weights
    length(w) == length(h) || error("dim_norm: weights length does not match restricted_hilbert length")
    if p == Inf
        m = 0.0
        for i in eachindex(h)
            if float(w[i]) > 0.0
                m = max(m, abs(float(h[i])))
            end
        end
        return m
    end
    p > 0 || error("dim_norm: p must be positive or Inf")
    s = 0.0
    for i in eachindex(h)
        wi = float(w[i])
        wi == 0.0 && continue
        s += wi * (abs(float(h[i]))^p)
    end
    return s^(1.0 / p)
end

"""
    region_weight_entropy(pi; box=nothing, weights=nothing, base=exp(1), kwargs...) -> Float64

Shannon entropy of the region-size distribution (weights normalized internally).
"""
function region_weight_entropy(pi; box=nothing, weights=nothing, base::Real=exp(1), kwargs...)
    bb = _resolve_box(pi, box)
    w = weights === nothing ? region_weights(pi; box=bb, kwargs...) : weights
    total = 0.0
    for wi in w
        total += float(wi)
    end
    total > 0.0 || return 0.0
    invtotal = 1.0 / total
    logbase = log(float(base))
    H = 0.0
    for wi in w
        p = float(wi) * invtotal
        if p > 0.0
            H -= p * log(p)
        end
    end
    return H / logbase
end

"""
    aspect_ratio_stats(pi; box, weights=nothing, kwargs...) -> NamedTuple

Volume-weighted summary statistics of `region_aspect_ratio(pi, r; box=box)`.
"""
function aspect_ratio_stats(pi; box, weights=nothing, kwargs...)
    bb = _resolve_box(pi, box)
    w = weights === nothing ? region_weights(pi; box=bb, kwargs...) : weights
    total = 0.0
    mean_num = 0.0
    amin = Inf
    amax = 0.0
    for r in 1:length(w)
        wi = float(w[r])
        wi == 0.0 && continue
        ar = region_aspect_ratio(pi, r; box=bb, kwargs...)
        total += wi
        mean_num += wi * float(ar)
        amin = min(amin, float(ar))
        amax = max(amax, float(ar))
    end
    if total == 0.0
        return (total_measure=0.0, mean=NaN, min=NaN, max=NaN)
    end
    return (total_measure=total, mean=mean_num / total, min=amin, max=amax)
end

"""
    module_size_summary(M, pi; box=nothing, weights=nothing, kwargs...) -> NamedTuple

Convenience wrapper collecting:
- total_measure
- integrated_hilbert_mass
- support_measure
- measure_by_dimension
- mean_dim, var_dim
"""
function module_size_summary(M, pi; box=nothing, weights=nothing, kwargs...)
    bb = _resolve_box(pi, box)
    w = weights === nothing ? region_weights(pi; box=bb, kwargs...) : weights
    stats = dim_stats(M, pi; weights=w)
    return (
        total_measure = stats.total_measure,
        integrated_hilbert_mass = integrated_hilbert_mass(M, pi; weights=w),
        support_measure = support_measure(M, pi; weights=w),
        measure_by_dimension = measure_by_dimension(M, pi; weights=w),
        mean_dim = stats.mean,
        var_dim = stats.var
    )
end

# =============================================================================
# (3c) Interface measures from the region adjacency graph
# =============================================================================

"""
    interface_measure(pi; box, adjacency=nothing, kwargs...) -> Float64

Total (n-1)-dimensional interface measure between distinct regions inside `box`.

This is the sum of all values in `region_adjacency(pi; box=box)`.
"""
function interface_measure(pi; box, adjacency=nothing, kwargs...)
    edges = adjacency === nothing ? region_adjacency(pi; box=box, kwargs...) : adjacency
    total = 0.0
    for v in values(edges)
        total += float(v)
    end
    return total
end

"""
    interface_measure_by_dim_pair(M, pi; box, adjacency=nothing, kwargs...) -> Dict{Tuple{Int,Int},Float64}

Aggregate interface measures by the pair of module dimensions on the two sides.

Keys are ordered pairs (d1,d2) with d1 <= d2.
"""
function interface_measure_by_dim_pair(M, pi; box, adjacency=nothing, kwargs...)
    dims = restricted_hilbert(M)
    edges = adjacency === nothing ? region_adjacency(pi; box=box, kwargs...) : adjacency
    out = Dict{Tuple{Int,Int},Float64}()
    for (e, m) in edges
        r, s = e
        d1 = dims[r]
        d2 = dims[s]
        key = d1 <= d2 ? (d1, d2) : (d2, d1)
        out[key] = get(out, key, 0.0) + float(m)
    end
    return out
end

"""
    interface_measure_dim_changes(M, pi; box, adjacency=nothing, kwargs...) -> Float64

Total interface measure between regions whose module dimensions differ.
"""
function interface_measure_dim_changes(M, pi; box, adjacency=nothing, kwargs...)
    dims = restricted_hilbert(M)
    edges = adjacency === nothing ? region_adjacency(pi; box=box, kwargs...) : adjacency
    total = 0.0
    for (e, m) in edges
        r, s = e
        dims[r] == dims[s] && continue
        total += float(m)
    end
    return total
end


# ------------------------------------------------------------------------------
# (3b) Additional module-level geometry statistics
#
# These build on:
# - region_weights (volumes)
# - region_boundary_measure (boundary measures)
# - region_adjacency (adjacency graph, weighted by boundary measure)
#
# and stratify by the restricted Hilbert function (dimension per region).
# ------------------------------------------------------------------------------

# Internal histogram helper (no StatsBase dependency).
# Returns (edges, counts) where edges has length nbins+1.
function _histogram_1d(values::Vector{Float64}; nbins::Integer=10, range=nothing)
    nbins >= 1 || error("histogram: nbins must be >= 1")
    v = [x for x in values if isfinite(x)]
    isempty(v) && return (edges=Float64[], counts=Int[])

    lo = range === nothing ? minimum(v) : float(range[1])
    hi = range === nothing ? maximum(v) : float(range[2])

    if hi == lo
        hi = lo + 1.0
    end

    edges = collect(range(lo, hi; length=nbins + 1))
    counts = zeros(Int, nbins)

    for x in v
        # Place x into a bin; include hi in the last bin.
        if x <= lo
            counts[1] += 1
        elseif x >= hi
            counts[end] += 1
        else
            t = (x - lo) / (hi - lo)
            b = Int(floor(t * nbins)) + 1
            b = clamp(b, 1, nbins)
            counts[b] += 1
        end
    end

    return (edges=edges, counts=counts)
end

"""
    region_volume_samples_by_dim(M, pi; box=nothing, weights=nothing) -> Dict{Int,Vector{Float64}}

Group region volumes by the module dimension on each region.

Arguments:
- `M` can be a module, or simply a vector of dimensions (see `restricted_hilbert` overload).
- Provide `weights` (region volumes) to avoid recomputing them; otherwise `box` is required.
"""
function region_volume_samples_by_dim(M, pi; box=nothing, weights=nothing)
    dims = restricted_hilbert(M)

    if weights === nothing
        box === nothing && error("region_volume_samples_by_dim: provide box or weights")
        weights = region_weights(pi; box=box)
    end

    length(weights) == length(dims) || error("region_volume_samples_by_dim: size mismatch")

    out = Dict{Int,Vector{Float64}}()
    for i in eachindex(weights)
        d = dims[i]
        push!(get!(out, d, Float64[]), float(weights[i]))
    end
    return out
end

"""
    region_volume_histograms_by_dim(M, pi; box=nothing, weights=nothing, nbins=10, range=nothing)
        -> Dict{Int,NamedTuple}

Histograms of region volumes stratified by region dimension.

Returns a dictionary mapping `dim => (edges, counts)`.
"""
function region_volume_histograms_by_dim(M, pi; box=nothing, weights=nothing,
    nbins::Integer=10, range=nothing)
    samples = region_volume_samples_by_dim(M, pi; box=box, weights=weights)
    out = Dict{Int,NamedTuple}()
    for (d, vals) in samples
        out[d] = _histogram_1d(vals; nbins=nbins, range=range)
    end
    return out
end

"""
    region_boundary_to_volume_samples_by_dim(M, pi; box=nothing, weights=nothing,
                                            boundary_measures=nothing, strict=true)
        -> Dict{Int,Vector{Float64}}

Compute boundary_measure / volume for each region, stratified by module dimension.

Provide `weights` and/or `boundary_measures` to avoid recomputation.
If `boundary_measures` is not provided, `box` is required and we compute
`region_boundary_measure(pi, i; box=box)`.
"""
function region_boundary_to_volume_samples_by_dim(M, pi; box=nothing,
    weights=nothing, boundary_measures=nothing, strict::Bool=true)

    dims = restricted_hilbert(M)

    if weights === nothing
        box === nothing && error("region_boundary_to_volume_samples_by_dim: provide box or weights")
        weights = region_weights(pi; box=box)
    end

    n = length(weights)
    length(dims) == n || error("region_boundary_to_volume_samples_by_dim: size mismatch")

    if boundary_measures === nothing
        box === nothing && error("region_boundary_to_volume_samples_by_dim: provide box or boundary_measures")
        boundary_measures = [region_boundary_measure(pi, i; box=box, strict=strict) for i in 1:n]
    end
    length(boundary_measures) == n || error("region_boundary_to_volume_samples_by_dim: boundary size mismatch")

    out = Dict{Int,Vector{Float64}}()
    for i in 1:n
        V = float(weights[i])
        S = float(boundary_measures[i])
        ratio = (V == 0.0) ? ((S == 0.0) ? NaN : Inf) : (S / V)
        push!(get!(out, dims[i], Float64[]), ratio)
    end
    return out
end

"""
    region_boundary_to_volume_histograms_by_dim(M, pi; box=nothing, weights=nothing,
                                               boundary_measures=nothing, nbins=10, range=nothing, strict=true)
        -> Dict{Int,NamedTuple}

Histograms of boundary_measure / volume stratified by region dimension.

Returns a dictionary mapping `dim => (edges, counts)`.
"""
function region_boundary_to_volume_histograms_by_dim(M, pi; box=nothing,
    weights=nothing, boundary_measures=nothing, nbins::Integer=10, range=nothing,
    strict::Bool=true)

    samples = region_boundary_to_volume_samples_by_dim(M, pi; box=box, weights=weights,
        boundary_measures=boundary_measures, strict=strict)
    out = Dict{Int,NamedTuple}()
    for (d, vals) in samples
        out[d] = _histogram_1d(vals; nbins=nbins, range=range)
    end
    return out
end

# ------------------------------------------------------------------------------
# Graph statistics for the region adjacency graph
# The adjacency is assumed undirected with each edge appearing once as (i,j), i<j.
# Edge weights are typically boundary measures (interface sizes).
# ------------------------------------------------------------------------------

"""
    graph_degrees(adjacency, nregions) -> (degrees, weighted_degrees)

Compute (unweighted) degrees and weighted degrees for an undirected graph.

- `adjacency` should be a Dict{Tuple{Int,Int},<:Real} with edges stored once.
- `degrees[i]` is the number of incident edges.
- `weighted_degrees[i]` is the sum of incident edge weights.
"""
function graph_degrees(adjacency, nregions::Integer)
    deg = zeros(Int, nregions)
    wdeg = zeros(Float64, nregions)
    for ((i, j), w) in adjacency
        wi = float(w)
        deg[i] += 1
        deg[j] += 1
        wdeg[i] += wi
        wdeg[j] += wi
    end
    return (degrees=deg, weighted_degrees=wdeg)
end

"""
    graph_connected_components(adjacency, nregions) -> Vector{Vector{Int}}

Connected components of an undirected graph.
"""
function graph_connected_components(adjacency, nregions::Integer)
    nbrs = [Int[] for _ in 1:nregions]
    for ((i, j), _) in adjacency
        push!(nbrs[i], j)
        push!(nbrs[j], i)
    end

    visited = falses(nregions)
    comps = Vector{Vector{Int}}()

    for v in 1:nregions
        if visited[v]
            continue
        end
        stack = [v]
        visited[v] = true
        comp = Int[]
        while !isempty(stack)
            u = pop!(stack)
            push!(comp, u)
            for w in nbrs[u]
                if !visited[w]
                    visited[w] = true
                    push!(stack, w)
                end
            end
        end
        push!(comps, comp)
    end

    return comps
end

"""
    graph_modularity(labels, adjacency; nregions=length(labels)) -> Float64

Weighted Newman-Girvan modularity for an undirected graph.

Let m = total edge weight (edges counted once). Let k_i be weighted degree of node i.
Then:

    Q = sum_c (w_in(c)/m - (k_tot(c)/(2m))^2),

where w_in(c) is total weight of edges with both endpoints in community c, and
k_tot(c) = sum_{i in c} k_i.

Returns 0 when m == 0.
"""
function graph_modularity(labels::AbstractVector{<:Integer}, adjacency; nregions::Integer=length(labels))
    length(labels) == nregions || error("graph_modularity: label length mismatch")

    m = 0.0
    wdeg = zeros(Float64, nregions)
    for ((i, j), w) in adjacency
        wi = float(w)
        m += wi
        wdeg[i] += wi
        wdeg[j] += wi
    end
    if m == 0.0
        return 0.0
    end

    # k_tot per community
    ktot = Dict{Int,Float64}()
    for i in 1:nregions
        lab = Int(labels[i])
        ktot[lab] = get(ktot, lab, 0.0) + wdeg[i]
    end

    # w_in per community (edges counted once)
    win = Dict{Int,Float64}()
    for ((i, j), w) in adjacency
        li = Int(labels[i])
        lj = Int(labels[j])
        if li == lj
            win[li] = get(win, li, 0.0) + float(w)
        end
    end

    Q = 0.0
    for (lab, ksum) in ktot
        Q += get(win, lab, 0.0) / m - (ksum / (2.0 * m))^2
    end
    return Q
end

"""
    region_adjacency_graph_stats(M, pi; box=nothing, adjacency=nothing, weights=nothing, strict=true) -> NamedTuple

Compute graph statistics for the region adjacency graph, including modularity stratified by
module dimension and by connected components.

- If `adjacency` is not provided, `box` is required and we call `region_adjacency(pi; box=box)`.
- `M` can be a module or a dims vector (see `restricted_hilbert` overload).
"""
function region_adjacency_graph_stats(M, pi; box=nothing, adjacency=nothing, weights=nothing, strict::Bool=true)
    dims = restricted_hilbert(M)
    nregions = length(dims)

    if adjacency === nothing
        box === nothing && error("region_adjacency_graph_stats: provide box or adjacency")
        adjacency = region_adjacency(pi; box=box, strict=strict)
    end

    degs = graph_degrees(adjacency, nregions)
    comps = graph_connected_components(adjacency, nregions)
    comp_sizes = [length(c) for c in comps]

    total_w = 0.0
    for w in values(adjacency)
        total_w += float(w)
    end

    # Partition by dimension and by connected components.
    Q_dim = graph_modularity(dims, adjacency; nregions=nregions)

    comp_label = zeros(Int, nregions)
    for (k, comp) in enumerate(comps)
        for v in comp
            comp_label[v] = k
        end
    end
    Q_comp = graph_modularity(comp_label, adjacency; nregions=nregions)

    return (
        nregions=nregions,
        nedges=length(adjacency),
        total_edge_weight=total_w,
        degrees=degs.degrees,
        weighted_degrees=degs.weighted_degrees,
        ncomponents=length(comps),
        component_sizes=comp_sizes,
        components=comps,
        modularity_by_dim=Q_dim,
        modularity_by_component=Q_comp,
    )
end

"""
    module_geometry_summary(M, pi; box=nothing, weights=nothing, adjacency=nothing,
                            boundary_measures=nothing, nbins=10, range=nothing, strict=true) -> NamedTuple

One-stop "geometry + module" summary, bundling:
- existing size/interface summaries
- new volume and boundary/volume distributions by dimension
- adjacency graph statistics (degrees, components, modularity)

This is intentionally a NamedTuple so mathematicians can access fields by name.
"""
function module_geometry_summary(M, pi; box=nothing, weights=nothing, adjacency=nothing,
    boundary_measures=nothing, nbins::Integer=10, range=nothing, strict::Bool=true)

    bb = _resolve_box(pi, box)

    if weights === nothing
        box === nothing && error("module_geometry_summary: provide box or weights")
        weights = region_weights(pi; box=bb, strict=strict)
    end

    if adjacency === nothing
        box === nothing && error("module_geometry_summary: provide box or adjacency")
        adjacency = region_adjacency(pi; box=bb, strict=strict)
    end

    if boundary_measures === nothing
        box === nothing && error("module_geometry_summary: provide box or boundary_measures")
        boundary_measures = [region_boundary_measure(pi, i; box=bb, strict=strict) for i in 1:length(weights)]
    end

    size_sum = module_size_summary(M, pi; weights=weights)
    iface = interface_measure(pi; box=bb, adjacency=adjacency)
    iface_pairs = interface_measure_by_dim_pair(M, pi; box=bb, adjacency=adjacency)
    iface_changes = interface_measure_dim_changes(M, pi; box=bb, adjacency=adjacency)

    vol_samples = region_volume_samples_by_dim(M, pi; weights=weights)
    vol_hists = region_volume_histograms_by_dim(M, pi; weights=weights, nbins=nbins, range=range)

    b2v_samples = region_boundary_to_volume_samples_by_dim(M, pi; weights=weights,
        boundary_measures=boundary_measures, strict=strict)
    b2v_hists = region_boundary_to_volume_histograms_by_dim(M, pi; weights=weights,
        boundary_measures=boundary_measures, nbins=nbins, range=range, strict=strict)

    gstats = region_adjacency_graph_stats(M, pi; adjacency=adjacency, strict=strict)

    return (
        size_summary=size_sum,
        interface_measure=iface,
        interface_by_dim_pair=iface_pairs,
        interface_dim_changes=iface_changes,
        volume_samples_by_dim=vol_samples,
        volume_histograms_by_dim=vol_hists,
        boundary_to_volume_samples_by_dim=b2v_samples,
        boundary_to_volume_histograms_by_dim=b2v_hists,
        graph_stats=gstats,
    )
end

# -----------------------------------------------------------------------------
# Asymptotic growth summaries (expanding windows)
# -----------------------------------------------------------------------------

# small internal helper: scale a box about its center and (optionally) integerize outward
function _scale_box_about_center(ell0, u0, s::Real, padding::Real, integerize::Bool)
    n = length(ell0)
    if integerize
        ell = Vector{Int}(undef, n)
        u   = Vector{Int}(undef, n)
        for i in 1:n
            lo0 = float(ell0[i])
            hi0 = float(u0[i])
            c = (lo0 + hi0) / 2
            h = (hi0 - lo0) / 2
            lo = c - float(s) * h - padding
            hi = c + float(s) * h + padding
            ell[i] = floor(Int, lo)
            u[i]   = ceil(Int,  hi)
        end
        return (ell, u)
    else
        ell = Vector{Float64}(undef, n)
        u   = Vector{Float64}(undef, n)
        for i in 1:n
            lo0 = float(ell0[i])
            hi0 = float(u0[i])
            c = (lo0 + hi0) / 2
            h = (hi0 - lo0) / 2
            ell[i] = c - float(s) * h - padding
            u[i]   = c + float(s) * h + padding
        end
        return (ell, u)
    end
end

# log-log linear regression fit of log(y) vs log(scale)
function _loglog_fit(scales, ys; strict::Bool=false)
    xs = Float64[]
    zs = Float64[]
    used = Int[]
    for i in eachindex(scales)
        s = float(scales[i])
        y = float(ys[i])
        if isfinite(s) && isfinite(y) && s > 0 && y > 0
            push!(xs, log(s))
            push!(zs, log(y))
            push!(used, i)
        end
    end
    if length(xs) < 2
        strict && error("loglog fit needs at least two positive points")
        return (exponent=NaN, intercept=NaN, r2=NaN, used_indices=used)
    end

    # compute means manually (no Statistics dependency)
    mx = 0.0
    mz = 0.0
    for i in eachindex(xs)
        mx += xs[i]
        mz += zs[i]
    end
    mx /= length(xs)
    mz /= length(xs)

    vx = 0.0
    cov = 0.0
    for i in eachindex(xs)
        dx = xs[i] - mx
        vx += dx * dx
        cov += dx * (zs[i] - mz)
    end
    b = cov / vx
    a = mz - b * mx

    sse = 0.0
    sst = 0.0
    for i in eachindex(xs)
        pred = a + b * xs[i]
        err = zs[i] - pred
        sse += err * err
        dz = zs[i] - mz
        sst += dz * dz
    end
    r2 = (sst == 0.0 ? 1.0 : 1.0 - sse / sst)

    return (exponent=b, intercept=a, r2=r2, used_indices=used)
end

# Polynomial least squares fit y ~ sum_{k=0}^deg c[k+1] * x^k.
function _polyfit(xs::AbstractVector{<:Real}, ys::AbstractVector{<:Real}, deg::Int)
    m = length(xs)
    A = Matrix{Float64}(undef, m, deg + 1)
    for i in 1:m
        x = float(xs[i])
        A[i, 1] = 1.0
        for k in 2:(deg + 1)
            A[i, k] = A[i, k - 1] * x
        end
    end
    y = Float64.(ys)
    c = A \ y
    yhat = A * c
    mu = sum(y) / m
    sst = sum((y[i] - mu)^2 for i in 1:m)
    sse = sum((y[i] - yhat[i])^2 for i in 1:m)
    r2 = sst == 0.0 ? NaN : 1.0 - sse/sst
    return (degree=deg, coeffs=c, r2=r2)
end

# Quasi-polynomial fit: separate polynomial fits on each residue class mod period.
function _quasipolyfit(xs::AbstractVector{<:Integer}, ys::AbstractVector{<:Real}, deg::Int, period::Int)
    fits = Vector{Any}(undef, period)
    for r in 0:(period - 1)
        idx = [i for i in eachindex(xs) if mod(xs[i], period) == r]
        if length(idx) < deg + 1
            fits[r + 1] = nothing
            continue
        end
        xsr = [xs[i] for i in idx]
        ysr = [ys[i] for i in idx]
        fits[r + 1] = merge(_polyfit(xsr, ysr, deg), (residue=r, npoints=length(idx)))
    end
    return (period=period, degree=deg, fits=fits)
end


"""
    module_geometry_asymptotics(M, pi;
        base_box=:auto,
        scales=[1,2,4,8],
        padding=0.0,
        fit=:loglog,
        include_interface=true,
        strict=true,
        include_ehrhart=false,
        ehrhart_period=1,
        ehrhart_degree=:auto
    ) -> NamedTuple

Compute window-dependent size/geometry summaries as the window expands.

If `base_box == :auto`, uses `window_box(pi)`.

For each `s in scales`, expands the base window about its center by factor `s`,
then adds absolute `padding`.

Returns a NamedTuple containing:
  * `windows`: per-scale windows used
  * `total_measure`: sum of region weights (volume in R^n or lattice count in Z^n)
  * `integrated_hilbert_mass`: integrated mass over the window
  * `interface_measure`: total adjacency weight, if supported
  * fitted log-log growth exponents
"""
function module_geometry_asymptotics(M, pi;
    base_box=:auto,
    scales=[1,2,4,8],
    padding=0.0,
    fit::Symbol=:loglog,
    include_interface::Bool=true,
    strict::Bool=true,
    include_ehrhart::Bool=false,
    ehrhart_period::Int=1,
    ehrhart_degree::Union{Symbol,Int}=:auto
    )

    fit == :loglog || error("module_geometry_asymptotics: only fit=:loglog is implemented")

    bb = (base_box === :auto ? window_box(pi) : base_box)
    ell0, u0 = bb
    length(ell0) == length(u0) || error("module_geometry_asymptotics: base_box has mismatched endpoints")

    integerize = (eltype(ell0) <: Integer) && (eltype(u0) <: Integer)

    # Dimension of the ambient parameter space (length of the window vectors).
    # Needed for Ehrhart-degree heuristics.
    dim = length(ell0)

    windows = Vector{typeof(bb)}()
    totals  = Float64[]
    masses  = Float64[]

    do_iface = include_interface && hasmethod(region_adjacency, Tuple{typeof(pi)})
    ifaces = do_iface ? Float64[] : nothing

    for s in scales
        float(s) > 0 || error("module_geometry_asymptotics: scales must be positive")
        win = _scale_box_about_center(ell0, u0, s, padding, integerize)
        push!(windows, win)

        w = region_weights(pi; box=win, strict=strict)
        push!(totals, sum(float, w))

        mh = integrated_hilbert_mass(M, pi; weights=w)
        push!(masses, float(mh))

        if do_iface
            adj = region_adjacency(pi; box=win, strict=strict)
            push!(ifaces, sum(float, values(adj)))
        end
    end

    fit_total = _loglog_fit(scales, totals)
    fit_mass  = _loglog_fit(scales, masses)
    fit_iface = do_iface ? _loglog_fit(scales, ifaces) : nothing

    ehrhart_total = nothing
    ehrhart_iface = nothing

    if include_ehrhart
        deg_total = ehrhart_degree === :auto ? min(dim, length(scales) - 1) : Int(ehrhart_degree)
        ehrhart_total = _quasipolyfit(Int.(scales), totals, deg_total, ehrhart_period)
        if do_iface
            deg_iface = ehrhart_degree === :auto ? min(max(dim - 1, 0), length(scales) - 1) : max(Int(ehrhart_degree) - 1, 0)
            ehrhart_iface = _quasipolyfit(Int.(scales), ifaces, deg_iface, ehrhart_period)
        end
    end

    return (
        base_box=bb,
        scales=scales,
        windows=windows,
        total_measure=totals,
        integrated_hilbert_mass=masses,
        interface_measure=ifaces,
        exponent_total_measure=fit_total.exponent,
        exponent_integrated_hilbert_mass=fit_mass.exponent,
        exponent_interface_measure=(do_iface ? fit_iface.exponent : NaN),
        fit_total=fit_total,
        fit_integrated_hilbert_mass=fit_mass,
        fit_interface=fit_iface,
        ehrhart_total_measure=ehrhart_total,
        ehrhart_interface_measure=ehrhart_iface,
        ehrhart_period=ehrhart_period,
        ehrhart_degree=ehrhart_degree
    )
end



# =============================================================================
# (3d) ASCII pretty printing for NamedTuple / Dict summaries
# =============================================================================

# This is intentionally lightweight and explicit:
# - We do NOT overload Base.show(::NamedTuple) globally.
# - We provide:
#     * pretty(x; ...) -> String
#     * pretty(io::IO, x; ...) -> nothing
#     * PrettyPrinter(x; ...) wrapper so `display(PrettyPrinter(x))` "just works".
#
# The output is ASCII-only and uses "..." (not Unicode ellipses).

# Small internal context storing formatting options and a cache of indentation strings.
struct _PrettyCtx
    indent::Int
    max_depth::Int
    max_items::Int
    max_list::Int
    sort_keys::Bool
    indents::Vector{String}
end

function _PrettyCtx(indent::Int, max_depth::Int, max_items::Int, max_list::Int, sort_keys::Bool)
    step = repeat(" ", indent)
    indents = Vector{String}(undef, max_depth + 2)
    indents[1] = ""
    for k in 2:length(indents)
        indents[k] = indents[k - 1] * step
    end
    return _PrettyCtx(indent, max_depth, max_items, max_list, sort_keys, indents)
end

# State to avoid trailing newlines while still printing line-oriented output.
mutable struct _PPState
    first_line::Bool
end
_PPState() = _PPState(true)

# Start a new output line, inserting '\n' *between* lines but not at the end.
function _pp_line(io::IO, st::_PPState, indent_str::AbstractString)
    if st.first_line
        st.first_line = false
    else
        write(io, '\n')
    end
    write(io, indent_str)
    return nothing
end

# Scalar-ness predicates: used to decide when vectors are safe to show inline.
_pp_is_scalar_type(T) = (T <: Number) || (T <: Bool) || (T <: Char) || (T <: Symbol) || (T <: AbstractString)
_pp_is_scalar(x) = (x === nothing) || (x isa Number) || (x isa Bool) || (x isa Char) || (x isa Symbol) || (x isa AbstractString)

# Inline representation:
# - returns a String for values that should be printed on one line
# - returns nothing for containers we want to expand (NamedTuple, Dict, and non-scalar vectors)
function _pp_inline_string(x, ctx::_PrettyCtx)::Union{Nothing,String}
    # Expand these (do not inline).
    if x isa NamedTuple || x isa AbstractDict
        return nothing
    end

    # Simple scalars.
    if x === nothing
        return "nothing"
    elseif x isa Number || x isa Bool || x isa Char || x isa Symbol
        return string(x)
    elseif x isa AbstractString
        # repr(...) adds quotes and escaping, still ASCII if the string is ASCII.
        return repr(x)
    elseif x isa Tuple
        # Tuples are typically small in this codebase (e.g. dimension pairs).
        io = IOBuffer()
        write(io, '(')
        for (i, y) in enumerate(x)
            if i > 1
                write(io, ", ")
            end
            if y isa AbstractString
                write(io, repr(y))
            else
                print(io, y)
            end
        end
        write(io, ')')
        return String(take!(io))
    elseif x isa AbstractVector
        # Inline only if the element type is scalar-ish. This avoids scanning huge vectors.
        if !_pp_is_scalar_type(eltype(x))
            return nothing
        end
        n = length(x)
        if n == 0
            return "[]"
        end
        m = min(n, ctx.max_list)
        io = IOBuffer()
        write(io, '[')

        # If max_list==0, show only an ellipsis summary.
        if m == 0
            write(io, "...] (len=")
            print(io, n)
            write(io, ')')
            return String(take!(io))
        end

        for i in 1:m
            if i > 1
                write(io, ", ")
            end
            print(io, x[i])
        end
        if n > m
            write(io, ", ...] (len=")
            print(io, n)
            write(io, ')')
        else
            write(io, ']')
        end
        return String(take!(io))
    elseif x isa AbstractMatrix
        m, n = size(x)
        return "$(m)x$(n) " * string(typeof(x))
    elseif x isa AbstractArray
        # Higher-dimensional arrays: just show a small summary.
        return "Array" * string(size(x)) * " " * string(typeof(x))
    else
        # Fallback: treat as scalar-ish, inline.
        # This is mainly used for small objects (e.g. Sets, small structs) that show on one line.
        return string(x)
    end
end

# Forward decl: container printer.
function _pp_container(io::IO, st::_PPState, ctx::_PrettyCtx, x, level::Int)
    if x isa NamedTuple
        return _pp_namedtuple(io, st, ctx, x, level)
    elseif x isa AbstractDict
        return _pp_dict(io, st, ctx, x, level)
    elseif x isa AbstractVector
        return _pp_vector(io, st, ctx, x, level)
    else
        # Fallback: print inline on its own line.
        ind = ctx.indents[level + 1]
        _pp_line(io, st, ind)
        inl = _pp_inline_string(x, ctx)
        if inl === nothing
            print(io, string(x))
        else
            print(io, inl)
        end
        return nothing
    end
end

# Print one "field-like" entry:
# - inline values use "key = value"
# - container values use "key:" then expand at the next indentation level
function _pp_field(io::IO, st::_PPState, ctx::_PrettyCtx, level::Int, key::AbstractString, val)
    ind = ctx.indents[level + 1]
    inl = _pp_inline_string(val, ctx)
    if inl !== nothing
        _pp_line(io, st, ind)
        print(io, key, " = ", inl)
        return nothing
    end

    # Non-inline value (likely a container).
    if level >= ctx.max_depth
        _pp_line(io, st, ind)
        print(io, key, " = ...")
        return nothing
    end

    _pp_line(io, st, ind)
    print(io, key, ":")
    _pp_container(io, st, ctx, val, level + 1)
    return nothing
end

function _pp_namedtuple(io::IO, st::_PPState, ctx::_PrettyCtx, nt::NamedTuple, level::Int)
    names = propertynames(nt)
    n = length(names)
    shown = min(n, ctx.max_items)
    for i in 1:shown
        nm = names[i]
        _pp_field(io, st, ctx, level, string(nm), getproperty(nt, nm))
    end
    if shown < n
        ind = ctx.indents[level + 1]
        _pp_line(io, st, ind)
        print(io, "... (", n - shown, " more fields)")
    end
    return nothing
end

function _pp_sorted_keys(d::AbstractDict, ctx::_PrettyCtx)
    ks = collect(keys(d))
    if !ctx.sort_keys
        return ks
    end

    # Preserve numeric ordering for Integer keys.
    if all(k -> k isa Integer, ks)
        sort!(ks)
        return ks
    end

    # Otherwise, try Julia's default ordering and fall back to string-based ordering.
    try
        sort!(ks)
    catch
        sort!(ks, by=k -> string(k))
    end
    return ks
end

function _pp_dict_entry(io::IO, st::_PPState, ctx::_PrettyCtx, level::Int, key, val)
    ind = ctx.indents[level + 1]
    kstr = repr(key)

    inl = _pp_inline_string(val, ctx)
    if inl !== nothing
        _pp_line(io, st, ind)
        print(io, kstr, " => ", inl)
        return nothing
    end

    if level >= ctx.max_depth
        _pp_line(io, st, ind)
        print(io, kstr, " => ...")
        return nothing
    end

    _pp_line(io, st, ind)
    print(io, kstr, " =>")
    _pp_container(io, st, ctx, val, level + 1)
    return nothing
end

function _pp_dict(io::IO, st::_PPState, ctx::_PrettyCtx, d::AbstractDict, level::Int)
    ks = _pp_sorted_keys(d, ctx)
    n = length(ks)
    shown = min(n, ctx.max_items)
    for i in 1:shown
        k = ks[i]
        _pp_dict_entry(io, st, ctx, level, k, d[k])
    end
    if shown < n
        ind = ctx.indents[level + 1]
        _pp_line(io, st, ind)
        print(io, "... (", n - shown, " more entries)")
    end
    return nothing
end

function _pp_vector(io::IO, st::_PPState, ctx::_PrettyCtx, v::AbstractVector, level::Int)
    n = length(v)
    shown = min(n, ctx.max_items)
    for i in 1:shown
        key = "[" * string(i) * "]"
        _pp_field(io, st, ctx, level, key, v[i])
    end
    if shown < n
        ind = ctx.indents[level + 1]
        _pp_line(io, st, ind)
        print(io, "... (", n - shown, " more items)")
    end
    return nothing
end

"""
    pretty(x; name=nothing, indent=2, max_depth=6, max_items=20, max_list=12, sort_keys=true) -> String
    pretty(io::IO, x; name=nothing, indent=2, max_depth=6, max_items=20, max_list=12, sort_keys=true)

ASCII pretty-printer for nested summaries, especially `NamedTuple`s returned by:
- `module_geometry_summary`
- `region_adjacency_graph_stats`
- `region_minkowski_functionals`
- `region_anisotropy_scores`

Conventions:
- Inline values print as `key = value`.
- Containers print as `key:` followed by an indented block.
- Dict entries print as `key => value`.
- Truncation uses ASCII `"..."` and reports remaining counts.

Keyword arguments:
- `name`: optional label for the root (useful for REPL reports).
- `indent`: spaces per indentation level.
- `max_depth`: maximum nesting depth; deeper containers are replaced by `...`.
- `max_items`: maximum fields/entries/items shown per container.
- `max_list`: maximum scalar entries shown inline for scalar vectors.
- `sort_keys`: stable ordering for Dicts (numeric for Integer keys; otherwise best-effort).
"""
function pretty(io::IO, x;
    name=nothing,
    indent::Integer=2,
    max_depth::Integer=6,
    max_items::Integer=20,
    max_list::Integer=12,
    sort_keys::Bool=true)

    indent >= 0 || error("pretty: indent must be >= 0")
    max_depth >= 0 || error("pretty: max_depth must be >= 0")
    max_items >= 0 || error("pretty: max_items must be >= 0")
    max_list >= 0 || error("pretty: max_list must be >= 0")

    ctx = _PrettyCtx(Int(indent), Int(max_depth), Int(max_items), Int(max_list), Bool(sort_keys))
    st = _PPState()

    if name !== nothing
        nm = string(name)
        inl = _pp_inline_string(x, ctx)
        if inl !== nothing
            _pp_line(io, st, ctx.indents[1])
            print(io, nm, " = ", inl)
        else
            _pp_line(io, st, ctx.indents[1])
            print(io, nm, ":")
            _pp_container(io, st, ctx, x, 1)
        end
    else
        inl = _pp_inline_string(x, ctx)
        if inl !== nothing
            _pp_line(io, st, ctx.indents[1])
            print(io, inl)
        else
            _pp_container(io, st, ctx, x, 0)
        end
    end

    return nothing
end

function pretty(x; kwargs...)
    io = IOBuffer()
    pretty(io, x; kwargs...)
    return String(take!(io))
end

"""
    PrettyPrinter(x; name=nothing, indent=2, max_depth=6, max_items=20, max_list=12, sort_keys=true)

A small wrapper type so that `display(PrettyPrinter(x))` (or REPL evaluation of the wrapper)
prints an ASCII pretty representation.

This is deliberately non-intrusive: it does NOT change how `NamedTuple` prints globally.

Example:
summ = module_geometry_summary(H, pi; box=box)
display(PrettyPrinter(summ; name="module_geometry_summary"))
"""
struct PrettyPrinter{T}
    x::T
    name::Union{Nothing,String}
    indent::Int
    max_depth::Int
    max_items::Int
    max_list::Int
    sort_keys::Bool
end

function PrettyPrinter(x;
    name=nothing,
    indent::Integer=2,
    max_depth::Integer=6,
    max_items::Integer=20,
    max_list::Integer=12,
    sort_keys::Bool=true)

    return PrettyPrinter{typeof(x)}(
        x,
        name === nothing ? nothing : string(name),
        Int(indent),
        Int(max_depth),
        Int(max_items),
        Int(max_list),
        Bool(sort_keys),
    )
end

Base.show(io::IO, ::MIME"text/plain", P::PrettyPrinter) =
pretty(io, P.x; 
       name=P.name,
       indent=P.indent,
       max_depth=P.max_depth,
       max_items=P.max_items,
       max_list=P.max_list,
       sort_keys=P.sort_keys,
)

Base.show(io::IO, P::PrettyPrinter) = Base.show(io, MIME"text/plain"(), P)

# =============================================================================
# (4) Slice restrictions and 1-parameter barcodes
#     + (5) Approximate matching distance via bottleneck distances
# =============================================================================

# ----- Slice restrictions and 1-parameter barcodes ----------------------------

# Internal: normalize an axis-aligned box argument.
# We accept boxes as (lo, hi), where lo and hi are vectors of the same length.
# Returns Float64 vectors for robust numeric clipping.
function _normalize_axis_aligned_box(box)
    (box isa Tuple && length(box) == 2) || error("expected box=(lo, hi)")
    lo, hi = box
    lo_v = Float64[float(x) for x in lo]
    hi_v = Float64[float(x) for x in hi]
    length(lo_v) == length(hi_v) || error("box endpoints must have the same length")
    @inbounds for i in 1:length(lo_v)
        lo_v[i] <= hi_v[i] || error("box must satisfy lo[i] <= hi[i] for all i")
    end
    return lo_v, hi_v
end

# Internal: intersect two axis-aligned boxes (either may be `nothing`).
# If both are provided, we use their intersection.
function _intersect_axis_aligned_boxes(box, box2)
    if box === nothing
        return _normalize_axis_aligned_box(box2)
    elseif box2 === nothing
        return _normalize_axis_aligned_box(box)
    end
    lo1, hi1 = _normalize_axis_aligned_box(box)
    lo2, hi2 = _normalize_axis_aligned_box(box2)
    length(lo1) == length(lo2) || error("box and box2 must have the same dimension")

    lo = similar(lo1)
    hi = similar(hi1)
    @inbounds for i in 1:length(lo)
        lo[i] = max(lo1[i], lo2[i])
        hi[i] = min(hi1[i], hi2[i])
    end
    return lo, hi
end

# Internal: parameter interval [tlo, thi] for which x(t)=x0+t*dir lies in an axis-aligned box.
# Returns (Inf, -Inf) if the line misses the box.
function _line_param_range_in_box_nd(
    x0::AbstractVector,
    dir::AbstractVector,
    box::Tuple{Vector{Float64},Vector{Float64}};
    atol::Float64=1e-12
)
    lo, hi = box
    d = length(lo)
    length(hi) == d || error("_line_param_range_in_box_nd: box endpoint length mismatch")
    length(x0) == d || error("_line_param_range_in_box_nd: x0 length mismatch")
    length(dir) == d || error("_line_param_range_in_box_nd: dir length mismatch")

    tlo = -Inf
    thi =  Inf

    @inbounds for i in 1:d
        a = lo[i]
        b = hi[i]
        a <= b || return (Inf, -Inf)

        xi = float(x0[i])
        di = float(dir[i])

        if di == 0.0
            # Coordinate is constant along the line; must already lie in [a,b].
            (xi < a - atol || xi > b + atol) && return (Inf, -Inf)
            continue
        end

        t1 = (a - xi) / di
        t2 = (b - xi) / di
        lo_i = min(t1, t2)
        hi_i = max(t1, t2)

        tlo = max(tlo, lo_i)
        thi = min(thi, hi_i)

        (tlo <= thi + atol) || return (Inf, -Inf)
    end

    return (tlo, thi)
end

"""
    slice_chain(pi, x0, dir;
        ts=nothing,
        tmin=nothing,
        tmax=nothing,
        nsteps::Int=1001,
        strict::Bool=true,
        drop_unknown::Bool=true,
        dedup::Bool=true,
        check_chain::Bool=false,
        box=nothing,
        box2=nothing,
    ) -> (chain::Vector{Int}, tvals::Vector)

Sample the encoding map `pi` along the affine line x(t) = x0 + t*dir and return
the sequence of region indices visited by the samples, together with the
parameter values at which those regions were recorded.

Parameter grids:
- If `ts` is provided, it is used verbatim (after optional filtering to a window).
- Otherwise the routine samples `nsteps` equally spaced points in an interval
  determined by `tmin` and `tmax`.

Window semantics (`box` / `box2`):
- A "window" is an axis-aligned bounding box, represented in the same format used
  elsewhere in this codebase (typically `([lo...], [hi...])`).
- If both `box` and `box2` are provided, the window is their intersection.
- If a window is present and a bound is omitted, the omitted bound is chosen
  from the full line-window intersection interval (semantic default).
- If no window is present and a bound is omitted, the default interval is `[0, 1]`.

Clipping rule:
- Whenever a window is present, any user-specified interval `[tmin, tmax]` is
  clipped to the line-window intersection interval.
- If the line does not intersect the window, the returned chain is empty.

Unknown regions:
- If `strict=true`, encountering `locate(pi, x) == 0` throws an error.
- If `strict=false`, unknown samples are either dropped (`drop_unknown=true`)
  or kept as region index 0.

If `dedup=true`, consecutive repeats of the same region index are removed.

If `check_chain=true`, the routine verifies that the resulting chain is monotone
in the region poset of `pi`.
"""
function slice_chain(
    pi,
    x0::AbstractVector,
    dir::AbstractVector;
    ts=nothing,
    tmin::Union{Real,Nothing}=nothing,
    tmax::Union{Real,Nothing}=nothing,
    nsteps::Int=1001,
    strict::Bool=true,
    drop_unknown::Bool=true,
    dedup::Bool=true,
    check_chain::Bool=false,
    box=nothing,
    box2=nothing
)
    length(x0) == length(dir) || error("slice_chain: x0 and dir must have same length")
    nsteps >= 2 || error("slice_chain: nsteps must be >= 2")

    # If a window is present, we clip/filter the parameter grid to the portion
    # of the line that lies inside the window.
    clip = (box !== nothing) || (box2 !== nothing)
    atol = 1e-12

    # If both windows are provided, interpret this as box intersection.
    bx = clip ? _intersect_axis_aligned_boxes(box, box2) : nothing

    # Choose parameter grid.
    if ts === nothing
        # Determine the default bounds.
        #
        # Semantic default:
        # - If a window is present and a bound is omitted, use the full intersection
        #   interval of the line with the window.
        # - If no window is present and a bound is omitted, use [0, 1].
        if clip
            tlo_box, thi_box = _line_param_range_in_box_nd(x0, dir, bx; atol=atol)
            tmin_eff = (tmin === nothing) ? tlo_box : float(tmin)
            tmax_eff = (tmax === nothing) ? thi_box : float(tmax)

            # Always clip any user-specified interval to the window intersection.
            tmin_eff = max(tmin_eff, tlo_box)
            tmax_eff = min(tmax_eff, thi_box)
        else
            tmin_eff = (tmin === nothing) ? 0.0 : float(tmin)
            tmax_eff = (tmax === nothing) ? 1.0 : float(tmax)
        end

        tgrid = (tmin_eff > tmax_eff) ? Float64[] : collect(range(tmin_eff, tmax_eff; length=nsteps))
    else
        tg = collect(ts)
        if clip && !isempty(tg)
            tlo_box, thi_box = _line_param_range_in_box_nd(x0, dir, bx; atol=atol)
            tg = [t for t in tg if float(t) >= tlo_box - atol && float(t) <= thi_box + atol]
        end
        tgrid = tg
    end

    chain = Int[]
    tvals = eltype(tgrid)[]

    for t in tgrid
        x = x0 .+ float(t) .* dir
        r = locate(pi, x)

        if r == 0
            if strict
                error("slice_chain: point left encoding domain (locate==0); pass strict=false to drop/keep unknown")
            end
            if drop_unknown
                continue
            end
        end

        if dedup && !isempty(chain) && last(chain) == r
            continue
        end

        push!(chain, r)
        push!(tvals, t)
    end

    if check_chain && !isempty(chain)
        P = region_poset(pi)
        _is_monotone_chain(P, chain) || error("slice_chain: chain is not monotone in region poset")
    end

    return chain, tvals
end

# Z^n wrapper: integer steps (useful because locate(pi, x) expects integer points).
function slice_chain(pi::ZnEncodingMap,
    x0::AbstractVector{<:Integer},
    dir::AbstractVector{<:Integer};
    kmin::Int=0,
    kmax::Int=100,
    kwargs...
)
    ts = kmin:kmax
    # Force dispatch to the generic slice_chain(pi, x0::AbstractVector, dir::AbstractVector; ...)
    return invoke(slice_chain, Tuple{Any, AbstractVector, AbstractVector},
                  pi, x0, dir; ts=ts, kwargs...)
end


# Helper: assert that `chain` is a chain in the poset Q (consecutive comparability suffices).
function _assert_chain(Q, chain::AbstractVector{Int})
    m = length(chain)
    m >= 1 || error("expected a nonempty chain")
    for q in chain
        (1 <= q <= Q.n) || error("chain vertex out of range: $q")
    end
    for i in 1:m-1
        Q.leq[chain[i], chain[i+1]] || error("not a chain: chain[$i]=$(chain[i]) is not <= chain[$(i+1)]=$(chain[i+1])")
    end
    return nothing
end

# Extend a length-m vector of parameter values to length m+1 by adding one extra endpoint.
# This lets us encode half-open intervals [birth, death) with death == m+1.
function _extend_values(values::AbstractVector)
    m = length(values)
    m >= 1 || error("values must be nonempty")
    if m == 1
        step = one(values[1])
    else
        step = values[end] - values[end-1]
        step == zero(step) && (step = one(step))
    end
    return vcat(values, values[end] + step)
end

"""
    restrict_to_chain(M, chain) -> PModule{QQ}

Restrict a finite-poset module `M` to a chain of vertices.

The returned module lives on the chain poset {1,...,m} with arrows i -> i+1,
where m = length(chain). Its vector spaces are `M_{chain[i]}` and its structure
maps are the corresponding maps in `M`.

This is useful if you want to inspect the actual matrices along the slice.
For interval decomposition / barcodes, `slice_barcode` is usually the better entry point.
"""
function restrict_to_chain(M::PModule{QQ}, chain::AbstractVector{Int})::PModule{QQ}
    _assert_chain(M.Q, chain)
    m = length(chain)

    # Chain poset: i <= j iff i <= j.
    leq = falses(m, m)
    for i in 1:m
        for j in i:m
            leq[i, j] = true
        end
    end
    Qc = FinitePoset(leq; check=false)

    dims = [M.dims[chain[i]] for i in 1:m]

    # Only need cover edges (i,i+1).
    edge_maps = Dict{Tuple{Int,Int}, Matrix{QQ}}()
    cc = _cover_cache(M.Q)
    memo = Dict{Tuple{Int,Int}, Matrix{QQ}}()
    for i in 1:m-1
        edge_maps[(i, i+1)] = _map_leq_cached(M, chain[i], chain[i+1], cc, memo)
    end
    return PModule{QQ}(Qc, dims, edge_maps)
end

"""
    slice_barcode(M, chain; values=nothing, check_chain=true) -> Dict{Tuple{T,T},Int}

Compute the interval decomposition (1-parameter barcode) of the restriction of `M`
to a chain `chain = [q1, ..., qm]` with q1 <= ... <= qm.

The result is returned as a sparse dictionary encoding a multiset of half-open intervals:
    (birth, death) => multiplicity

Conventions:
- Births are indexed by chain position 1..m.
- Deaths are indexed by 2..m+1, where death == m+1 means "persists to the end".
- Intervals are half-open: [birth, death).

If `values` is provided, endpoints are taken from `values` instead of integer positions.
Acceptable lengths:
- length(values) == m: we extend by one extra endpoint to represent death == m+1.
- length(values) == m+1: used as-is.

Implementation note:
We compute the A_m-interval multiplicities by inclusion-exclusion on the 1D rank invariant
along the chain:
  mult(b,d) = r[b, d-1] - r[b-1, d-1] - r[b, d] + r[b-1, d]
where r[i,j] = rank(M(q_i -> q_j)) and out-of-range r is treated as 0.
"""
function slice_barcode(M::PModule{QQ}, chain::AbstractVector{Int};
    values=nothing,
    check_chain::Bool=true
)
    check_chain && _assert_chain(M.Q, chain)
    m = length(chain)

    # Endpoint coordinates along the chain.
    if values === nothing
        endpoints = collect(1:(m+1))
    else
        length(values) == m || length(values) == m + 1 || error("slice_barcode: values must have length m or m+1")
        endpoints = length(values) == m ? _extend_values(values) : values
    end

    # Compute rank matrix on the chain: r[i,j] = rank(M(q_i -> q_j)).
    cc = _cover_cache(M.Q)
    memo = Dict{Tuple{Int,Int}, Matrix{QQ}}()
    R = zeros(Int, m, m)
    for i in 1:m
        for j in i:m
            R[i, j] = rank_map(M, chain[i], chain[j]; cache=cc, memo=memo)
        end
    end

    getR(i, j) = (i < 1 || j < 1 || i > m || j > m || i > j) ? 0 : R[i, j]

    out = Dict{Tuple{eltype(endpoints), eltype(endpoints)}, Int}()

    # Interval multiplicities via inclusion-exclusion on the rank invariant.
    # For half-open intervals [b,d) with 1 <= b <= m and b+1 <= d <= m+1:
    #   mult(b,d) = r[b, d-1] - r[b-1, d-1] - r[b, d] + r[b-1, d]
    for b in 1:m
        for d in (b+1):(m+1)
            mult = getR(b, d-1) - getR(b-1, d-1) - getR(b, d) + getR(b-1, d)
            mult < 0 && error("slice_barcode: negative multiplicity detected at (b,d)=($b,$d)")
            if mult > 0
                out[(endpoints[b], endpoints[d])] = mult
            end
        end
    end

    return out
end

# Convenience wrapper: build the chain from a slice in the original domain first.
function slice_barcode(M::PModule{QQ}, pi, x0::AbstractVector, dir::AbstractVector; kwargs...)
    chain, tvals = slice_chain(pi, x0, dir; kwargs...)
    return slice_barcode(M, chain; values=tvals)
end

# ----- Bottleneck distance on 1D barcodes -------------------------------------

# Expand a sparse barcode dictionary to a multiset of diagram points.
function _barcode_points(bar)::Vector{Tuple{Float64,Float64}}
    if bar isa AbstractVector
        pts = Tuple{Float64,Float64}[]
        for I in bar
            b, d = I
            push!(pts, (float(b), float(d)))
        end
        return pts
    elseif bar isa AbstractDict
        pts = Tuple{Float64,Float64}[]
        for (I, mult) in bar
            b, d = I
            for _ in 1:Int(mult)
                push!(pts, (float(b), float(d)))
            end
        end
        return pts
    else
        error("barcode_points: expected a vector of intervals or a dictionary")
    end
end

#--------------------------------------------------------------------
# Wasserstein distance and kernels for 1-parameter barcodes.
#--------------------------------------------------------------------

function _point_distance(a::Tuple{<:Real,<:Real}, b::Tuple{<:Real,<:Real}, q::Real)
    dx = abs(float(a[1] - b[1]))
    dy = abs(float(a[2] - b[2]))
    if q == Inf
        return max(dx, dy)
    elseif q == 2
        return sqrt(dx * dx + dy * dy)
    elseif q == 1
        return dx + dy
    else
        error("_point_distance: supported q are 1, 2, Inf")
    end
end

function _diag_distance(a::Tuple{<:Real,<:Real}, q::Real)
    d = abs(float(a[2] - a[1]))
    if q == Inf
        return d / 2
    elseif q == 2
        return d / sqrt(2)
    elseif q == 1
        return d
    else
        error("_diag_distance: supported q are 1, 2, Inf")
    end
end

function _hungarian(cost::Matrix{Float64})
    n, m = size(cost)
    n == m || error("_hungarian: cost matrix must be square")

    u = zeros(Float64, n + 1)
    v = zeros(Float64, m + 1)
    p = zeros(Int, m + 1)
    way = zeros(Int, m + 1)

    for i in 1:n
        p[1] = i
        j0 = 1
        minv = fill(Inf, m + 1)
        used = fill(false, m + 1)
        way .= 0

        while true
            used[j0] = true
            i0 = p[j0]
            delta = Inf
            j1 = 1

            for j in 2:m+1
                if !used[j]
                    cur = cost[i0, j - 1] - u[i0 + 1] - v[j]
                    if cur < minv[j]
                        minv[j] = cur
                        way[j] = j0
                    end
                    if minv[j] < delta
                        delta = minv[j]
                        j1 = j
                    end
                end
            end

            for j in 1:m+1
                if used[j]
                    u[p[j] + 1] += delta
                    v[j] -= delta
                else
                    minv[j] -= delta
                end
            end

            j0 = j1
            if p[j0] == 0
                break
            end
        end

        while true
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 1
                break
            end
        end
    end

    assignment = zeros(Int, n)
    for j in 2:m+1
        i = p[j]
        if i != 0
            assignment[i] = j - 1
        end
    end

    total = 0.0
    for i in 1:n
        total += cost[i, assignment[i]]
    end
    return assignment, total
end

"""
    wasserstein_distance(bar1, bar2; p=2, q=Inf)

Compute the p-Wasserstein distance between two 1-parameter barcodes (persistence diagrams).

Barcodes use the same representation as `slice_barcode`, namely a dictionary
`Dict{Tuple{T,T},Int}` mapping (birth, death) to multiplicity.

Parameters:
- `p >= 1` is the Wasserstein exponent.
- `q` selects the ground metric on points in the plane: `q=Inf` (default) is L_infty,
  `q=2` is Euclidean, and `q=1` is L1.

Diagonal matching is included in the standard way. The return value is a Float64.
"""
function wasserstein_distance(bar1, bar2; p::Real=2, q::Real=Inf)
    p >= 1 || error("wasserstein_distance: expected p >= 1")

    P = _barcode_points(bar1)
    Q = _barcode_points(bar2)
    m = length(P)
    n = length(Q)
    N = m + n

    N == 0 && return 0.0

    C = zeros(Float64, N, N)

    for i in 1:m
        for j in 1:n
            C[i, j] = _point_distance(P[i], Q[j], q)^p
        end
        for j in (n+1):N
            C[i, j] = _diag_distance(P[i], q)^p
        end
    end

    for i in (m+1):N
        for j in 1:n
            C[i, j] = _diag_distance(Q[j], q)^p
        end
    end

    _, cost = _hungarian(C)
    return cost^(1 / p)
end

"""
    wasserstein_kernel(bar1, bar2; p=2, q=Inf, sigma=1.0, kind=:gaussian)

Kernels derived from the Wasserstein distance:

- `kind=:gaussian`  -> exp(-d^2/(2*sigma^2))
- `kind=:laplacian` -> exp(-d/sigma)
"""
function wasserstein_kernel(bar1, bar2; p::Real=2, q::Real=Inf, sigma::Real=1.0, kind::Symbol=:gaussian)
    sigma > 0 || error("wasserstein_kernel: sigma must be > 0")
    d = wasserstein_distance(bar1, bar2; p=p, q=q)
    if kind == :gaussian
        return exp(-(d * d) / (2 * float(sigma)^2))
    elseif kind == :laplacian
        return exp(-d / float(sigma))
    else
        error("wasserstein_kernel: unknown kind $(kind)")
    end
end

"""
    sliced_wasserstein_kernel(M, N, pi; ...)

Kernelized sliced Wasserstein invariant computed from 1D slice barcodes.

This wrapper exists for convenience. It constructs default directions and offsets
when omitted, and then calls `slice_kernel`.
"""
function sliced_wasserstein_kernel(M::PModule{QQ}, N::PModule{QQ}, pi;
    offsets=nothing,
    n_dirs::Integer=16,
    n_offsets::Integer=9,
    p=2,
    q=1,
    agg=mean,
    dir_weights=nothing,
    offset_weights=nothing,
    lambda=1.0,
    max_den=8,
    include_axes=false,
    strict=true,
    box=nothing,
    box2=nothing)

    if dirs === nothing
        dirs = default_directions(pi; n_dirs=n_dirs, max_den=max_den, include_axes=include_axes, normalize=:L1)
    end
    if offsets === nothing
        offsets = default_offsets(pi; n_offsets=n_offsets, box=box, margin=0.05)
    end

    return slice_kernel(M, N, pi;
        
        offsets=offsets,
        p=p,
        q=q,
        agg=agg,
        dir_weights=dir_weights,
        offset_weights=offset_weights,
        kernel="wasserstein_gaussian",
        lambda=lambda,
        strict=strict,
        box=box,
        box2=box2)
end

# ---------------------------------------------------------------------------
# Internal helper: unify all slice-based distances in one implementation.
#
# There are two distinct ways "weights" are used in the literature/code:
#
#   (A) weight_mode = :integrate
#       W is treated as an integration weight matrix. Then:
#         agg = mean   -> sum(W .* d)   (a weighted mean if normalize_weights=true)
#         agg = :pmean -> (sum(W .* d.^p))^(1/p)   (Lp-type aggregate, if weights are normalized)
#
#   (B) weight_mode = :scale
#       W is treated as a multiplicative scaling matrix (matching-distance style).
#       In this mode the natural reduction is agg = maximum, giving max(W .* d).
#
# This helper is intentionally internal (leading underscore) to keep the public API
# small while avoiding code duplication across sliced and matching-style distances.
# ---------------------------------------------------------------------------
function _slice_based_barcode_distance(
    bcs1::Matrix{Dict{Tuple{Float64, Float64}, Int}},
    bcs2::Matrix{Dict{Tuple{Float64, Float64}, Int}};
    weights1 = uniform2d,
    weights2 = uniform2d,
    weight = (d, o) -> weights1(d) * weights2(o),
    dirs::AbstractVector{Tuple{Float64, Float64}},
    offs::AbstractVector{Tuple{Float64, Float64}},
    dist = :bottleneck,
    agg = :mean,
    agg_p = 1.0,
    agg_norm::Real = 1.0,
    threads::Bool = (Threads.nthreads() > 1),
)
    # Build separable weights and outer product.
    wdir = [direction_weight(d, weights1) for d in dirs]
    woff = [offset_weight(o, weights2) for o in offs]
    W = wdir .* transpose(woff)

    dist_fn = dist == :bottleneck ? bottleneck_distance :
              dist == :wasserstein ? wasserstein_distance :
              throw(ArgumentError("Unknown dist=$dist"))

    agg_mode = agg
    if !(agg_mode in (:mean, :pmean, :max)) && !(agg_mode isa Function)
        throw(ArgumentError("Unknown agg=$agg"))
    end

    nslices = length(bcs1)

    if threads && Threads.nthreads() > 1
        nT = Threads.nthreads()

        if agg_mode == :max
            best_by_slot = fill(0.0, nT)
            Threads.@threads for slot in 1:nT
                best = 0.0
                for idx in slot:nT:nslices
                    w = W[idx]
                    if w > 0
                        d = w * dist_fn(bcs1[idx], bcs2[idx])
                        if d > best
                            best = d
                        end
                    end
                end
                best_by_slot[slot] = best
            end
            return maximum(best_by_slot)
        end

        if agg_mode == :mean
            acc_by_slot = fill(0.0, nT)
            sumw_by_slot = fill(0.0, nT)
            Threads.@threads for slot in 1:nT
                acc = 0.0
                sumw = 0.0
                for idx in slot:nT:nslices
                    w = W[idx]
                    if w > 0
                        acc += w * dist_fn(bcs1[idx], bcs2[idx])
                        sumw += w
                    end
                end
                acc_by_slot[slot] = acc
                sumw_by_slot[slot] = sumw
            end
            acc = sum(acc_by_slot)
            sumw = sum(sumw_by_slot)
            return (sumw == 0.0) ? 0.0 : acc / sumw / float(agg_norm)
        end

        if agg_mode == :pmean
            p = float(agg_p)
            acc_by_slot = fill(0.0, nT)
            sumw_by_slot = fill(0.0, nT)
            Threads.@threads for slot in 1:nT
                acc = 0.0
                sumw = 0.0
                for idx in slot:nT:nslices
                    w = W[idx]
                    if w > 0
                        d = dist_fn(bcs1[idx], bcs2[idx])
                        acc += w * d^p
                        sumw += w
                    end
                end
                acc_by_slot[slot] = acc
                sumw_by_slot[slot] = sumw
            end
            acc = sum(acc_by_slot)
            sumw = sum(sumw_by_slot)
            if sumw == 0.0
                return 0.0
            end
            return (acc / sumw)^(1 / p) / float(agg_norm)
        end

        # Custom aggregator: compute all scaled distances in parallel, then call `agg`.
        vals = Vector{Float64}(undef, nslices)
        Threads.@threads for idx in 1:nslices
            vals[idx] = W[idx] * dist_fn(bcs1[idx], bcs2[idx])
        end
        return agg_mode(vals)
    end

    # Serial implementation (existing behavior).
    if agg_mode == :max
        best = 0.0
        for i in eachindex(bcs1)
            w = W[i]
            if w > 0
                best = max(best, w * dist_fn(bcs1[i], bcs2[i]))
            end
        end
        return best
    end

    if agg_mode == :mean
        acc = 0.0
        sumw = 0.0
        for i in eachindex(bcs1)
            w = W[i]
            if w > 0
                acc += w * dist_fn(bcs1[i], bcs2[i])
                sumw += w
            end
        end
        return (sumw == 0.0) ? 0.0 : acc / sumw / float(agg_norm)
    end

    if agg_mode == :pmean
        p = float(agg_p)
        acc = 0.0
        sumw = 0.0
        for i in eachindex(bcs1)
            w = W[i]
            if w > 0
                d = dist_fn(bcs1[i], bcs2[i])
                acc += w * d^p
                sumw += w
            end
        end
        return (sumw == 0.0) ? 0.0 : (acc / sumw)^(1 / p) / float(agg_norm)
    end

    vals = Float64[]
    for i in eachindex(bcs1)
        w = W[i]
        if w > 0
            push!(vals, w * dist_fn(bcs1[i], bcs2[i]))
        end
    end
    return agg_mode(vals)
end


"""
    sliced_wasserstein_distance(M, N, pi; ...)

A slice-based distance computed from 1D barcodes of line restrictions.

Per-slice metric:
- Uses `wasserstein_distance` on the barcode for each slice.

Aggregation (key point for "Lp generalizations"):
- `agg = mean` or `agg = :mean` returns `sum(W .* d)` where `W` is the weight matrix from
  `slice_barcodes`. If `normalize_weights=true` (default), `sum(W) == 1`, so this is a
  genuine weighted mean.
- `agg = :pmean` returns `(sum(W .* d.^agg_p))^(1/agg_p)` (with `agg_p=Inf` giving a max).
  This is the clean "Lp-family" slice aggregate, matching the intention in your screenshot.
- `agg = maximum` or `agg = :maximum` returns `maximum(d)` across slices (unweighted).
- Any other callable `agg` is applied to the unweighted vector of per-slice distances.

Keywords `dirs`, `offsets`, `n_dirs`, `n_offsets`, `max_den`, `include_axes`,
`normalize_dirs`, `offset_weights`, `normalize_weights`, `offset_margin`, `weight`,
and `strict/box/box2` mirror the slicing machinery.

Wasserstein parameters:
- `p` is the Wasserstein exponent (p >= 1).
- `q` selects the ground metric on points (q=Inf is L_infty; q=2 Euclidean; q=1 L1).
"""
function sliced_wasserstein_distance(
    M::PModule{QQ},
    N::PModule{QQ},
    pi::PLikeEncodingMap;
    offsets=nothing,
    n_dirs::Integer=16,
    n_offsets::Integer=9,
    max_den=8,
    include_axes=false,
    normalize_dirs::Symbol=:L1,
    offset_weights=:cosine,
    normalize_weights=true,
    offset_margin=0.05,
    p=2,
    q=1,
    agg=mean,
    agg_p::Real=2,               # <-- new: Lp aggregator exponent for agg=:pmean
    weight::Symbol=:lesnick_l1,
    strict=true,
    box=nothing,
    box2=nothing,
    threads::Bool = (Threads.nthreads() > 1),
    slice_kwargs...
)::Float64

    dirs, offs = get_slices(dirs, offs; ndirs = ndirs, noff = noff)
    Bs1 = slice_barcodes(M, pi, dirs, offs; weights1 = weights1, weights2 = weights2, weight = weight, values = :t, threads = threads, slice_kwargs... )
    Bs2 = slice_barcodes(N, pi, dirs, offs; weights1 = weights1, weights2 = weights2, weight = weight, values = :t, threads = threads, slice_kwargs...)
    return _slice_based_barcode_distance(
        Bs1.barcodes, Bs2.barcodes;
        weights1 = weights1, weights2 = weights2, weight = weight,
        dirs = Bs1.dirs, offs = Bs1.offs,
        dist = :wasserstein,
        agg = agg, agg_p = agg_p, agg_norm = agg_norm,
        threads = threads,
    )
end

"""
    sliced_bottleneck_distance(M, N, pi; ...)

Exactly like `sliced_wasserstein_distance`, but the per-slice metric is
`bottleneck_distance` instead of `wasserstein_distance`.

This is the "sliced bottleneck" member of the same slice-based family.

Aggregation:
- `agg = mean` or `agg = :mean` returns `sum(W .* d)` (weighted mean if weights are normalized).
- `agg = :pmean` returns `(sum(W .* d.^agg_p))^(1/agg_p)` with `agg_p=Inf` giving a max.
- `agg = maximum` or `agg = :maximum` returns `maximum(d)` across slices (unweighted).
- Any other callable `agg` is applied to the unweighted vector of per-slice distances.
"""
function sliced_bottleneck_distance(
    M::PModule{QQ},
    N::PModule{QQ},
    pi::PLikeEncodingMap;
    ndirs::Int = 100,
    noff::Int = 50,
    dirs = :auto,
    offsets = :auto,
    dist_fn::Function = bottleneck_distance,
    dist_kwargs = NamedTuple(),
    agg = :mean,
    agg_p = 1.0,
    agg_norm::Real = 1.0,
    threads::Bool = (Threads.nthreads() > 1),
    slice_kwargs...
)
    dirs, offsets = get_slices(dirs, offsets; ndirs = ndirs, noff = noff)

    Bs1 = slice_barcodes(M, pi, dirs, offsets; values = :t, threads = threads, slice_kwargs...)
    Bs2 = slice_barcodes(N, pi, dirs, offsets; values = :t, threads = threads, slice_kwargs...)

    return _slice_based_barcode_distance(
        Bs1.barcodes, Bs2.barcodes;
        dirs = Bs1.dirs, offs = Bs1.offsets,
        dist_fn = dist_fn, agg = agg, agg_p = agg_p, agg_norm = agg_norm,
        threads = threads,
        dist_kwargs...
    )
end


_diag_cost(p::Tuple{Float64,Float64}) = 0.5 * abs(p[2] - p[1])
_linf_dist(p::Tuple{Float64,Float64}, q::Tuple{Float64,Float64}) = max(abs(p[1] - q[1]), abs(p[2] - q[2]))

# Hopcroft-Karp maximum matching for bipartite graphs given by adjacency lists.
function _hopcroft_karp(adj::Vector{Vector{Int}}, n_left::Int, n_right::Int)::Int
    pairU = fill(0, n_left)
    pairV = fill(0, n_right)
    dist = fill(0, n_left)
    INF = typemax(Int)

    function bfs()::Bool
        q = Int[]
        for u in 1:n_left
            if pairU[u] == 0
                dist[u] = 0
                push!(q, u)
            else
                dist[u] = INF
            end
        end

        found = false
        qi = 1
        while qi <= length(q)
            u = q[qi]
            qi += 1
            for v in adj[u]
                u2 = pairV[v]
                if u2 == 0
                    found = true
                elseif dist[u2] == INF
                    dist[u2] = dist[u] + 1
                    push!(q, u2)
                end
            end
        end
        return found
    end

    function dfs(u::Int)::Bool
        for v in adj[u]
            u2 = pairV[v]
            if u2 == 0 || (dist[u2] == dist[u] + 1 && dfs(u2))
                pairU[u] = v
                pairV[v] = u
                return true
            end
        end
        dist[u] = INF
        return false
    end

    matching = 0
    while bfs()
        for u in 1:n_left
            if pairU[u] == 0 && dfs(u)
                matching += 1
            end
        end
    end
    return matching
end

function _bottleneck_leq_eps(A::Vector{Tuple{Float64,Float64}}, B::Vector{Tuple{Float64,Float64}}, eps::Float64)::Bool
    m = length(A)
    n = length(B)

    # Left side: A points + n diagonal copies.
    # Right side: B points + m diagonal copies.
    n_left = m + n
    n_right = n + m

    diag_nodes = n+1:n+m  # right diagonal indices (may be empty)

    # Precompute which B points can be sent to the diagonal within eps.
    Bok = Int[]
    for j in 1:n
        if _diag_cost(B[j]) <= eps
            push!(Bok, j)
        end
    end

    adj = Vector{Vector{Int}}(undef, n_left)
    for u in 1:n_left
        adj[u] = Int[]
    end

    # A points.
    for i in 1:m
        p = A[i]
        neigh = adj[i]
        # to B points
        for j in 1:n
            if _linf_dist(p, B[j]) <= eps
                push!(neigh, j)
            end
        end
        # to diagonal (any right diagonal copy)
        if _diag_cost(p) <= eps
            for v in diag_nodes
                push!(neigh, v)
            end
        end
    end

    # Left diagonal copies (one per B point, but all identical as diagonal points).
    for k in 1:n
        neigh = adj[m + k]
        # match B points to diagonal when allowed
        for j in Bok
            push!(neigh, j)
        end
        # diagonal to diagonal always allowed (cost 0)
        for v in diag_nodes
            push!(neigh, v)
        end
    end

    match_size = _hopcroft_karp(adj, n_left, n_right)
    return match_size == n_left
end

"""
    bottleneck_distance(barA, barB) -> Float64

Compute the bottleneck distance between two 1D barcodes.

Inputs can be:
- a dictionary `(birth, death) => multiplicity` as returned by `slice_barcode`, or
- a vector of intervals `[(birth, death), ...]` (multiplicity by repetition).

We use the standard persistence-diagram L_infinity metric:
- cost((b,d),(b',d')) = max(|b-b'|, |d-d'|)
- cost((b,d), diagonal) = (d-b)/2

This implementation is exact for the given finite barcodes (up to floating-point).
"""
function bottleneck_distance(barA, barB)::Float64
    A = _barcode_points(barA)
    B = _barcode_points(barB)

    if isempty(A) && isempty(B)
        return 0.0
    end

    # Candidate eps values: all pairwise distances and diagonal costs.
    epss = Float64[0.0]
    for p in A
        push!(epss, _diag_cost(p))
    end
    for q in B
        push!(epss, _diag_cost(q))
    end
    for p in A
        for q in B
            push!(epss, _linf_dist(p, q))
        end
    end
    sort!(epss)
    epss = unique(epss)

    lo = 1
    hi = length(epss)
    while lo < hi
        mid = (lo + hi) >>> 1
        if _bottleneck_leq_eps(A, B, epss[mid])
            hi = mid
        else
            lo = mid + 1
        end
    end
    return epss[lo]
end

# Convenience: bottleneck distance between slice barcodes of two modules on the same chain.
function bottleneck_distance(M::PModule{QQ}, N::PModule{QQ}, chain::AbstractVector{Int}; kwargs...)::Float64
    bM = slice_barcode(M, chain; kwargs...)
    bN = slice_barcode(N, chain; kwargs...)
    return bottleneck_distance(bM, bN)
end

# ----- Approximate matching distance ------------------------------------------

function _direction_weight(dir::AbstractVector; scheme::Symbol=:lesnick_l1)::Float64
    if scheme == :none
        return 1.0
    elseif scheme == :lesnick_l1
        # Normalize to L1 = 1 and take min coordinate.
        s = 0.0
        for x in dir
            x < 0 && error("direction_weight: expected nonnegative direction entries")
            s += float(x)
        end
        s == 0.0 && error("direction_weight: zero direction vector")
        mn = Inf
        for x in dir
            mn = min(mn, float(x) / s)
        end
        return float(mn)
    elseif scheme == :lesnick_linf
        # Normalize so max = 1 and take min coordinate.
        mx = 0.0
        for x in dir
            x < 0 && error("direction_weight: expected nonnegative direction entries")
            mx = max(mx, float(x))
        end
        mx == 0.0 && error("direction_weight: zero direction vector")
        mn = Inf
        for x in dir
            mn = min(mn, float(x) / mx)
        end
        return float(mn)
    else
        error("direction_weight: unknown scheme $scheme")
    end
end

# --------------------------------------------------------------------------
# Lightweight weight helpers used by slice-based invariants.
# These are tiny and allocation-free for inner loops.
# --------------------------------------------------------------------------

"""
    uniform2d(::Any) -> Float64

Trivial weight function returning 1.0. Used as the default for unweighted
direction/offset sampling.
"""
@inline uniform2d(::Any)::Float64 = 1.0

"""
    direction_weight(dir, spec) -> Float64

Evaluate a direction weight.

`spec` can be:
- a `Symbol` (e.g. `:lesnick_l1`, `:lesnick_linf`, `:none`)
- a function `w(dir)::Real`
- a real scalar (constant weight)
"""
@inline function direction_weight(dir::AbstractVector{<:Real}, spec)::Float64
    if spec === :none || spec === :uniform
        return 1.0
    elseif spec isa Symbol
        return _direction_weight(dir; scheme=spec)
    elseif spec isa Function
        return Float64(spec(dir))
    elseif spec isa Real
        return Float64(spec)
    else
        throw(ArgumentError("direction_weight: unsupported spec type $(typeof(spec))"))
    end
end

"""
    offset_weight((c0,c1), spec) -> Float64

Weight for an *interval* of offsets (used for arrangement-cell weighting).

`spec` can be:
- `:uniform` / `:none`  -> 1.0
- `:length`             -> c1 - c0
- a function `w((c0,c1))::Real`
- a real scalar
"""
@inline function offset_weight(interval::Tuple{<:Real,<:Real}, spec)::Float64
    if spec === :none || spec === :uniform
        return 1.0
    elseif spec === :length
        return Float64(interval[2] - interval[1])
    elseif spec isa Function
        return Float64(spec(interval))
    elseif spec isa Real
        return Float64(spec)
    else
        throw(ArgumentError("offset_weight: unsupported spec $(spec)"))
    end
end

# For finite sampled offsets (slice_barcodes sampling). Distinct from interval weights.
function _offset_sample_weights(offs::AbstractVector{<:Real}, spec)
    n = length(offs)
    w = ones(Float64, n)
    if spec === nothing || spec === :none || spec === :uniform
        return w
    elseif spec === :cosine
        # Smooth window; normalize afterwards if requested.
        for i in 1:n
            t = (i - 0.5) / n
            w[i] = sin(pi * t)
        end
        return w
    elseif spec isa Function
        for i in 1:n
            w[i] = Float64(spec(offs[i]))
        end
        return w
    else
        throw(ArgumentError("_offset_sample_weights: unsupported offset_weights $(spec)"))
    end
end


"""
    sample_directions_2d(; max_den=8, include_axes=false, normalize=:L1) -> Vector

Return a deterministic list of "standard" direction vectors in 2D.

This helper is intended for slice-based workflows such as:
* approximating the matching distance via `matching_distance_approx`,
* sampling fibered barcodes (RIVET-style) by sweeping a set of slopes.

Directions are generated from primitive integer pairs `(a,b)` with
`0 <= a,b <= max_den`, `gcd(a,b)=1`, and `(a,b) != (0,0)`.

By default we *exclude* the coordinate axes (require `a>0` and `b>0`), because
matching-distance slices typically assume strictly positive directions.
Set `include_axes=true` to also include axis directions.

The list is sorted by increasing slope `b/a` (with slope = +Inf when `a==0`).

Normalization:
* `normalize=:L1` (default): return Float64 directions with `d[1]+d[2]==1`.
* `normalize=:Linf`: return Float64 directions with `max(d)==1`.
* `normalize=:none`: return integer directions `Vector{Int}` (useful for Z^2).

Examples
--------

R^2 / PL encodings:

    dirs = sample_directions_2d(max_den=5)
    d = matching_distance_approx(M, N, pi; directions=dirs, offsets=[x0], tmin=0, tmax=1)

Z^2 encodings (integer directions and integer steps):

    dirsZ = sample_directions_2d(max_den=5; normalize=:none)
    d = matching_distance_approx(M, N, pi; directions=dirsZ, offsets=[g0], kmin=0, kmax=50)
"""
function sample_directions_2d(; max_den::Int=8, include_axes::Bool=false, normalize::Symbol=:L1)
    max_den >= 1 || error("sample_directions_2d: max_den must be >= 1")

    pairs = Tuple{Int,Int}[]
    for a in 0:max_den, b in 0:max_den
        (a == 0 && b == 0) && continue
        if !include_axes && (a == 0 || b == 0)
            continue
        end
        gcd(a, b) == 1 || continue
        push!(pairs, (a, b))
    end

    # Sort by slope b/a (a==0 treated as +Inf).
    slope(p::Tuple{Int,Int}) = p[1] == 0 ? Inf : (float(p[2]) / float(p[1]))
    sort!(pairs; by=slope)

    if normalize == :none
        return [Int[p[1], p[2]] for p in pairs]
    elseif normalize == :L1
        return [Float64[float(p[1]) / float(p[1] + p[2]),
                        float(p[2]) / float(p[1] + p[2])] for p in pairs]
    elseif normalize == :Linf
        return [Float64[float(p[1]) / float(max(p[1], p[2])),
                        float(p[2]) / float(max(p[1], p[2]))] for p in pairs]
    else
        error("sample_directions_2d: normalize must be :L1, :Linf, or :none")
    end
end


# ----- Defaults for sliced invariants ------------------------------------------

"""
    encoding_box(pi; margin=0.05, box=nothing) -> (lo, hi)

Return an axis-aligned bounding box for the parameter space of an encoding map `pi`.

The box is inferred from a finite set of representative points stored in the
encoding map:

- If `pi` has a non-empty field `reps`, use `pi.reps`.
- Else if `pi` has a non-empty field `witnesses`, use `pi.witnesses`.
- Else if `pi` has a non-empty field `coords` that looks like a list of points
  (as in `ZnEncodingMap`), use `pi.coords`.

The return value is a pair `(lo, hi)` where `lo[i] <= hi[i]` and both are
`Vector{Float64}`.

If `box` is provided, it should be a pair `(lo, hi)` and is returned after
conversion to `Float64`.

`margin` enlarges the box by a relative amount in each coordinate. For example,
`margin=0.05` enlarges each side length by 5 percent on both ends. Set
`margin=0.0` to disable enlargement.

This helper exists mainly to support `default_offsets` and out-of-the-box
behavior of `matching_distance_approx` and the sliced Wasserstein wrappers.
"""
function encoding_box(pi; margin::Real=0.05, box=nothing)
    if box !== nothing
        lo, hi = box
        lo_v = Float64[float(x) for x in lo]
        hi_v = Float64[float(x) for x in hi]
        length(lo_v) == length(hi_v) || error("encoding_box: box endpoints must have same dimension")
        for i in 1:length(lo_v)
            lo_v[i] <= hi_v[i] || error("encoding_box: expected lo[i] <= hi[i] for all i")
        end
        return lo_v, hi_v
    end

    pts = _encoding_points(pi)
    d = length(pts[1])
    lo = fill(Inf, d)
    hi = fill(-Inf, d)
    for p in pts
        length(p) == d || error("encoding_box: inconsistent point dimensions in encoding map")
        for i in 1:d
            x = float(p[i])
            if x < lo[i]; lo[i] = x; end
            if x > hi[i]; hi[i] = x; end
        end
    end

    margin < 0 && error("encoding_box: margin must be nonnegative")
    if margin > 0
        for i in 1:d
            w = hi[i] - lo[i]
            lo[i] -= margin * w
            hi[i] += margin * w
        end
    end
    return lo, hi
end

"""
    window_box(pi;
        padding=0.0,
        margin=0.05,
        integerize=:auto,
        method::Symbol=:reps
    ) -> (ell, u)

Return a finite axis-aligned window (ell, u) for the parameter space of an
encoding map `pi`.

This is a "mathematician-friendly" helper used by window-dependent invariants.
It is especially useful when regions are unbounded.

Heuristics (data-driven):
  * Start with `encoding_box(pi; margin=0.0)` (representative points / witnesses).
  * Optionally incorporate per-axis coordinate lists `pi.coords` (if present),
    especially to avoid degenerate (zero-width) directions.
  * Apply relative enlargement `margin` and absolute `padding`.
  * Optionally round outward to integers for lattice encodings (`Z^n`).

Keywords:
  * `padding`: absolute padding added to each coordinate.
  * `margin`: relative padding based on (u-ell).
  * `integerize`:
        - :auto   => integerize if the encoding data is integer-valued
        - :always => always integerize outward
        - :never  => never integerize
  * `method`:
        - :reps   => primarily use reps/witnesses, fix degenerate axes using coords if possible
        - :coords => use per-axis coords if available, otherwise fall back to reps
        - :mix    => take the hull of reps-box and coords-box when both exist

Returns:
  `(ell, u)` with `Vector`s of length `pi.n`. The element type is `Int` if
  integerized, otherwise `Float64`.
"""
function window_box(pi; padding=0.0, margin=0.05, integerize=:auto, method::Symbol=:reps)
    padding < 0 && error("window_box: padding must be nonnegative")
    margin < 0 && error("window_box: margin must be nonnegative")

    # --- candidate from reps/witnesses via encoding_box ---
    reps_box = nothing
    try
        reps_box = encoding_box(pi; margin=0.0)
    catch
        reps_box = nothing
    end

    # --- candidate from coords if coords is per-axis list (common in PLBackend / ZnEncoding) ---
    coords_box = nothing
    if hasproperty(pi, :coords)
        c = getproperty(pi, :coords)
        if c isa AbstractVector && !isempty(c) && c[1] isa AbstractVector
            # interpret as per-axis coordinate lists iff length(c) == pi.n when available
            ok = true
            if hasproperty(pi, :n)
                ok = (length(c) == getproperty(pi, :n))
            end
            if ok
                ellc = Vector{Float64}(undef, length(c))
                uc   = Vector{Float64}(undef, length(c))
                for i in 1:length(c)
                    ci = c[i]
                    if isempty(ci)
                        # free coordinate: give a small symmetric default window
                        ellc[i] = -1.0
                        uc[i]   =  1.0
                    else
                        # include one step beyond extremes
                        mn = float(minimum(ci))
                        mx = float(maximum(ci))
                        ellc[i] = mn - 1.0
                        uc[i]   = mx + 1.0
                    end
                end
                coords_box = (ellc, uc)
            end
        end
    end

    # --- select base box ---
    base = nothing
    if method == :reps
        base = reps_box === nothing ? coords_box : reps_box
        base === nothing && error("window_box: cannot infer window (no reps and no usable coords)")
        # repair degenerate axes if possible
        if coords_box !== nothing
            ell, u = base
            ellc, uc = coords_box
            for i in 1:length(ell)
                if u[i] == ell[i]
                    ell[i] = ellc[i]
                    u[i]   = uc[i]
                end
            end
        end
    elseif method == :coords
        base = coords_box === nothing ? reps_box : coords_box
        base === nothing && error("window_box: cannot infer window (no usable coords and no reps)")
    elseif method == :mix
        if reps_box === nothing
            base = coords_box
        elseif coords_box === nothing
            base = reps_box
        else
            ellr, ur = reps_box
            ellc, uc = coords_box
            ell = similar(ellr)
            u   = similar(ur)
            for i in 1:length(ell)
                ell[i] = min(ellr[i], ellc[i])
                u[i]   = max(ur[i],  uc[i])
            end
            base = (ell, u)
        end
        base === nothing && error("window_box: cannot infer window (no reps and no usable coords)")
    else
        error("window_box: unknown method=$(method)")
    end

    ell, u = base

    # ensure nondegenerate axes even if both sources are degenerate
    for i in 1:length(ell)
        if u[i] == ell[i]
            ell[i] -= 1.0
            u[i]   += 1.0
        end
    end

    # apply relative margin
    for i in 1:length(ell)
        w = u[i] - ell[i]
        ell[i] -= margin * w
        u[i]   += margin * w
    end

    # apply absolute padding
    if padding != 0.0
        for i in 1:length(ell)
            ell[i] -= padding
            u[i]   += padding
        end
    end

    # decide integerization
    do_int = false
    if integerize == :always
        do_int = true
    elseif integerize == :never
        do_int = false
    elseif integerize == :auto
        # prefer reps/witness data type if available
        do_int = false
        try
            pts = _encoding_points(pi)
            if !isempty(pts)
                do_int = (eltype(first(pts)) <: Integer)
            end
        catch
            # fallback: coords element type
            if hasproperty(pi, :coords)
                c = getproperty(pi, :coords)
                if c isa AbstractVector && !isempty(c) && c[1] isa AbstractVector
                    do_int = (eltype(c[1]) <: Integer)
                end
            end
        end
    else
        error("window_box: integerize must be :auto, :always, or :never")
    end

    if do_int
        elli = Vector{Int}(undef, length(ell))
        ui   = Vector{Int}(undef, length(u))
        for i in 1:length(ell)
            elli[i] = floor(Int, ell[i])
            ui[i]   = ceil(Int,  u[i])
        end
        return (elli, ui)
    else
        return (ell, u)
    end
end


# Internal helper: choose a finite set of points that represent the parameter space.
function _encoding_points(pi)
    if hasproperty(pi, :reps)
        reps = getproperty(pi, :reps)
        if reps !== nothing && !isempty(reps)
            return reps
        end
    end
    if hasproperty(pi, :witnesses)
        w = getproperty(pi, :witnesses)
        if w !== nothing && !isempty(w)
            return w
        end
    end
    if hasproperty(pi, :coords)
        c = getproperty(pi, :coords)
        # Heuristic: treat `coords` as a point cloud only when it is "tall".
        # This avoids misclassifying PL grid coordinate arrays as a point cloud.
        if c !== nothing && !isempty(c) && c[1] isa AbstractVector{<:Real} &&
           length(c) > length(c[1])
            return c
        end
    end
    error("cannot infer defaults: encoding map has no non-empty `reps`, `witnesses`, or point-like `coords` field")
end

# Subsample a vector deterministically by taking approximately evenly spaced indices.
function _subsample_evenly(v::AbstractVector, n::Integer)
    n <= 0 && error("expected n >= 1")
    m = length(v)
    if m <= n
        return collect(v)
    end
    idxs = round.(Int, range(1, m, length=n))
    idxs = unique(idxs)
    return [v[i] for i in idxs]
end

# Decide whether to default to integer directions/offsets (lattice-style slicing).
_is_lattice_encoding(pi) = eltype(_encoding_points(pi)) <: Integer

"""
    default_directions(pi; n_dirs=16, max_den=8, include_axes=false, normalize=:L1)

Return a deterministic list of slice directions suitable for `pi`.

- For 2-parameter encodings, directions are generated from primitive integer
  pairs `(a,b)` with `1 <= a,b <= max_den` (and optionally axis directions).
- For higher dimension, a small collection of primitive integer vectors with
  entries in `1:max_den` is used.

If the encoding points of `pi` are integer-valued (lattice encodings), the
returned directions are integer vectors. Otherwise, directions are returned as
`Float64` vectors, normalized according to `normalize` (one of `:L1`, `:Linf`,
or `:none`).

This function is intended to provide sensible defaults; it is not a substitute
for problem-specific direction sampling.
"""
function default_directions(pi; n_dirs::Integer=16, max_den::Integer=8,
                           include_axes::Bool=false, normalize::Symbol=:L1)
    d = length(_encoding_points(pi)[1])
    integer_dirs = _is_lattice_encoding(pi)
    return default_directions(d; n_dirs=n_dirs, max_den=max_den,
                              include_axes=include_axes, normalize=normalize,
                              integer=integer_dirs)
end

"""
    default_directions(d::Integer; n_dirs=16, max_den=8, include_axes=false,
                       normalize=:L1, integer=false)

Low-level direction generator used by `default_directions(pi)`.
"""
function default_directions(d::Integer; n_dirs::Integer=16, max_den::Integer=8,
                           include_axes::Bool=false, normalize::Symbol=:L1,
                           integer::Bool=false)
    d <= 0 && error("default_directions: dimension must be positive")
    max_den <= 0 && error("default_directions: max_den must be positive")
    n_dirs <= 0 && error("default_directions: n_dirs must be positive")

    if d == 1
        return integer ? [Int[1]] : [Float64[1.0]]
    elseif d == 2
        dirs_all = sample_directions_2d(max_den=max_den, include_axes=include_axes,
                                        normalize=(integer ? :none : normalize))
        return _subsample_evenly(dirs_all, n_dirs)
    end

    lo = include_axes ? 0 : 1
    dirs_int = Vector{Vector{Int}}()
    ranges = ntuple(_ -> lo:max_den, d)
    for tup in Iterators.product(ranges...)
        v = collect(tup)
        all(x -> x == 0, v) && continue
        g = 0
        for x in v
            g = gcd(g, x)
        end
        g == 1 || continue
        push!(dirs_int, v)
    end

    sort!(dirs_int; by=v -> begin
        s = sum(v)
        return Tuple(float(x) / float(s) for x in v)
    end)

    dirs_int = _subsample_evenly(dirs_int, n_dirs)

    if integer || normalize == :none
        return dirs_int
    elseif normalize == :L1
        return [Float64[float(x) / float(sum(v)) for x in v] for v in dirs_int]
    elseif normalize == :Linf
        return [Float64[float(x) / float(maximum(v)) for x in v] for v in dirs_int]
    else
        error("default_directions: normalize must be :L1, :Linf, or :none")
    end
end

"""
    default_offsets(pi; n_offsets=9, box=nothing, margin=0.05)

Return a deterministic list of slice offsets (basepoints) for `pi`.

The offsets are points in the parameter space. For a given direction `dir`,
each offset `x0` defines a slice line `x(t) = x0 + t * dir`.

This method does not depend on `dir` and is designed to work with the current
`sliced_*` APIs, which use a common set of offsets for all directions.

If the encoding points of `pi` are integer-valued (lattice encodings), the
returned offsets are integer vectors.

See also `default_offsets(pi, dir; ...)` for a direction-aware construction in 2D.
"""
function default_offsets(pi; n_offsets::Integer=9, box=nothing, margin::Real=0.05)
    n_offsets <= 0 && error("default_offsets: n_offsets must be positive")
    lo, hi = encoding_box(pi; box=box, margin=margin)
    d = length(lo)

    if _is_lattice_encoding(pi)
        lo_i = ceil.(Int, lo)
        hi_i = floor.(Int, hi)
        any(lo_i .> hi_i) && error("default_offsets: empty integer box after rounding")
        if n_offsets == 1
            return [lo_i]
        end
        pts = Vector{Vector{Int}}()
        for k in 0:(n_offsets - 1)
            t = float(k) / float(n_offsets - 1)
            p = [round(Int, lo_i[i] + t * (hi_i[i] - lo_i[i])) for i in 1:d]
            push!(pts, p)
        end
        return unique(pts)
    end

    m = ceil(Int, n_offsets^(1 / d))
    coords = [collect(range(lo[i], hi[i], length=max(m, 1))) for i in 1:d]
    pts = Vector{Vector{Float64}}()
    for tup in Iterators.product(coords...)
        push!(pts, Float64[tup...])
    end
    return _subsample_evenly(pts, n_offsets)
end

"""
    default_offsets(pi, dir; n_offsets=9, box=nothing, margin=0.05)

Direction-aware offset construction for 2-parameter (2D) encodings.

When `pi` is not a lattice encoding and the ambient dimension is 2, this chooses
offsets along a segment orthogonal to `dir` so that the resulting family of
parallel lines intersects the bounding box of `pi`.

For other cases, this falls back to `default_offsets(pi; ...)`.
"""
function default_offsets(pi, dir::AbstractVector;
                         n_offsets::Integer=9,
                         box=nothing,
                         margin::Real=0.05)
    n_offsets <= 0 && error("default_offsets: n_offsets must be positive")
    _is_lattice_encoding(pi) && return default_offsets(pi; n_offsets=n_offsets, box=box, margin=margin)

    lo, hi = encoding_box(pi; box=box, margin=margin)
    d = length(lo)
    length(dir) == d || error("default_offsets: direction dimension mismatch")
    if d != 2
        return default_offsets(pi; n_offsets=n_offsets, box=(lo, hi), margin=0.0)
    end

    dirf = Float64[float(dir[1]), float(dir[2])]
    n = Float64[-dirf[2], dirf[1]]
    nn = norm(n)
    if nn == 0.0
        return default_offsets(pi; n_offsets=n_offsets, box=(lo, hi), margin=0.0)
    end
    n ./= nn

    corners = (Float64[lo[1], lo[2]],
               Float64[lo[1], hi[2]],
               Float64[hi[1], lo[2]],
               Float64[hi[1], hi[2]])
    projs = [dot(c, n) for c in corners]
    smin, smax = minimum(projs), maximum(projs)

    c = (lo .+ hi) ./ 2
    cproj = dot(c, n)
    svals = range(smin, smax, length=n_offsets)
    return [c .+ (s - cproj) .* n for s in svals]
end



"""
    matching_distance_approx(M, N, slices; default_weight=1.0) -> Float64
    matching_distance_approx(M, N, pi; directions, offsets, weight=:lesnick_l1, ...) -> Float64

Approximate the (2D/ND) matching distance by taking a maximum of bottleneck
distances over a finite family of 1D slices.

Two ways to call this:

1) Provide slices explicitly:
    slices = [
        [q1,q2,...,qm],                                # a chain in M.Q
        (chain=[...], values=[...], weight=1.0),       # richer spec (NamedTuple)
        (chain, values, weight)                        # tuple form
    ]

2) Provide an encoding map and sample geometric slices:
    matching_distance_approx(M, N, pi;
        directions=[v1,v2,...],
        offsets=[x01,x02,...],
        tmin=..., tmax=..., nsteps=..., strict=true,
        weight=:lesnick_l1
    )

For each slice we:
- build a chain in the finite encoding (via `slice_chain` if using (2)),
- compute slice barcodes for M and N,
- compute bottleneck distance between those barcodes,
- multiply by a slice weight,
- take the maximum.

Notes:
- This is an approximation: increasing the number of sampled slices improves it.
- The default `weight=:lesnick_l1` matches the common Lesnick-Wright style scaling
  in 2D after L1 normalization; set `weight=:none` to disable weighting.
"""
function matching_distance_approx(
    M::PModule{QQ},
    N::PModule{QQ},
    slices::AbstractVector;
    default_weight=1.0
)::Float64
    best = 0.0

    for spec in slices
        chain = nothing
        values = nothing
        w = float(default_weight)

        if spec isa AbstractVector{Int}
            chain = spec
        elseif spec isa NamedTuple
            chain = spec.chain
            values = get(spec, :values, nothing)
            w = float(get(spec, :weight, default_weight))
        elseif spec isa Tuple
            length(spec) >= 1 || error("matching_distance_approx: empty tuple slice spec")
            chain = spec[1]
            values = length(spec) >= 2 ? spec[2] : nothing
            w = length(spec) >= 3 ? float(spec[3]) : float(default_weight)
        else
            error("matching_distance_approx: unknown slice spec type $(typeof(spec))")
        end

        bM = slice_barcode(M, chain; values=values)
        bN = slice_barcode(N, chain; values=values)
        best = max(best, w * bottleneck_distance(bM, bN))
    end

    return best
end

matching_distance_approx(
    M::PModule{QQ},
    N::PModule{QQ},
    chain::AbstractVector{Int};
    default_weight=1.0
)::Float64 = matching_distance_approx(M, N, [chain]; default_weight=default_weight)

"""
    matching_distance_approx(M, N, pi; ...)

Approximate matching distance by sampling directions and offsets, and taking a supremum
over (direction, offset) of a direction-weighted bottleneck distance between slice barcodes.

This uses a matching-style reduction:
    max_{dir, off} ( w(dir) * bottleneck_distance(slice(M), slice(N)) )
"""
function matching_distance_approx(
    M::PModule{QQ},
    N::PModule{QQ},
    pi::PLikeEncodingMap;
    ndirs::Int = 100,
    noff::Int = 50,
    dirs = :auto,
    offs = :auto,
    weights1 = uniform2d,
    weights2 = uniform2d,
    weight = (d, o) -> weights1(d) * weights2(o),
    agg = :max,
    agg_p = 1.0,
    agg_norm::Real = 1.0,
    dist_kwargs = (p = 1.0, q = Inf),
    threads::Bool = (Threads.nthreads() > 1),
    slice_kwargs...
)
    return _slice_based_barcode_distance(
        M, N, pi;
        dist_fn = matching_distance,
        dist_kwargs = dist_kwargs,
        weight_mode = :scale,
        dirs = dirs,
        offs = offs,
        ndirs = ndirs,
        noff = noff,
        weights1 = weights1,
        weights2 = weights2,
        weight = weight,
        agg = agg,
        agg_p = agg_p,
        agg_norm = agg_norm,
        threads = threads,
        slice_kwargs...
    )
end


"""
    matching_wasserstein_distance_approx(M, N, pi; p=2, q=1, ...)

A Wasserstein analog of `matching_distance_approx`:

    max_{dir, off} ( w(dir) * wasserstein_distance(slice(M), slice(N); p, q) )

This is not the classical matching distance (which uses bottleneck), but it is exactly
the "matching_wasserstein_distance_approx" slice-based family member requested in your
screenshot: same structure, Wasserstein per slice, supremum reduction.

Keywords mirror `matching_distance_approx`, plus Wasserstein parameters `p` and `q`.
"""
function matching_wasserstein_distance_approx(
    M::PModule{QQ},
    N::PModule{QQ},
    pi::PLikeEncodingMap;
    ndirs::Int = 100,
    noff::Int = 50,
    dirs = :auto,
    offs = :auto,
    weights1 = uniform2d,
    weights2 = uniform2d,
    weight = (d, o) -> weights1(d) * weights2(o),
    agg = :max,
    agg_p = 1.0,
    agg_norm::Real = 1.0,
    dist_kwargs = (p = 1.0, q = Inf),
    threads::Bool = (Threads.nthreads() > 1),
    slice_kwargs...
)
    return _slice_based_barcode_distance(
        M, N, pi;
        dist_fn = matching_wasserstein_distance,
        dist_kwargs = dist_kwargs,
        weight_mode = :scale,
        dirs = dirs,
        offs = offs,
        ndirs = ndirs,
        noff = noff,
        weights1 = weights1,
        weights2 = weights2,
        weight = weight,
        agg = agg,
        agg_p = agg_p,
        agg_norm = agg_norm,
        threads = threads,
        slice_kwargs...
    )
end


"""
    matching_wasserstein_distance_approx(M, N, pi, slices; p=2, q=1)

Slice-explicit variant (mirrors `matching_distance_approx(M,N,pi,slices)`):
each element of `slices` can be either:

- a chain `Vector{Int}` (interpreted as (chain, weights=nothing, t=0, w=1))
- or a tuple `(chain, weights, t, w)` where `w` scales the per-slice distance

The output is:
    max_slice ( w * wasserstein_distance(barcode_slice_M, barcode_slice_N; p, q) )
"""
function matching_wasserstein_distance_approx(
    M::PModule{QQ},
    N::PModule{QQ},
    pi::PLikeEncodingMap,
    slices::Vector;
    p=2,
    q=1
)::Float64
    d = 0.0
    for slice in slices
        chain = slice isa Vector ? slice : slice[1]
        weights = slice isa Vector ? nothing : slice[2]
        t = slice isa Vector ? 0 : slice[3]
        w = slice isa Vector ? 1 : slice[4]

        bcM = slice_barcode(M, pi, chain, weights, t)
        bcN = slice_barcode(N, pi, chain, weights, t)
        ds = w * wasserstein_distance(bcM, bcN; p=p, q=q)
        if ds > d
            d = ds
        end
    end
    return d
end


# -----------------------------------------------------------------------------
# Exact 2D matching distance (deterministic, no RNG)
#
# This implements the screenshot algorithm:
#   - Collect critical points (grid vertices or polyhedral vertices) + box corners.
#   - Build critical slopes from dy/dx over pairs, then use midpoints between slopes
#     as representative directions (optionally include axes).
#   - For each direction, compute critical offsets c_i = dot(n, p_i) with
#     n = [-dir[2], dir[1]], and use midpoints between consecutive c_i as
#     representative offsets.
#   - For each (dir, offset) representative, extract the slice chain exactly:
#       * coords-based backend: intersect with all vertical/horizontal grid lines.
#       * polyhedral backend: compute all region t-interval endpoints, then label
#         segments via locate at midpoints.
#   - Compute max over slices of w(dir) * bottleneck_distance(slice_barcode(...)).
#
# Notes:
#   - Values passed to slice_barcode are boundary parameter values (length n+1).
#   - Determinism is enforced by sorting and stable de-duplication.
#   - For speed, we avoid sampling and use O(K^2) slope enumeration over critical points,
#     which is feasible for modest K (typical in exact use).
# -----------------------------------------------------------------------------

# In-place unique of a sorted float vector using an absolute tolerance.
function _unique_sorted_floats!(v::Vector{Float64}; atol::Float64=1e-12)
    sort!(v)
    n = length(v)
    if n <= 1
        return v
    end
    j = 1
    last = v[1]
    @inbounds for i in 2:n
        x = v[i]
        if abs(x - last) > atol
            j += 1
            v[j] = x
            last = x
        end
    end
    resize!(v, j)
    return v
end

# In-place unique of a sorted vector of points (x,y) with lexicographic sorting.
function _unique_points_2d!(pts::Vector{NTuple{2,Float64}}; atol::Float64=1e-12)
    sort!(pts, by=p->(p[1], p[2]))
    n = length(pts)
    if n <= 1
        return pts
    end
    j = 1
    last = pts[1]
    @inbounds for i in 2:n
        p = pts[i]
        if (abs(p[1]-last[1]) > atol) || (abs(p[2]-last[2]) > atol)
            j += 1
            pts[j] = p
            last = p
        end
    end
    resize!(pts, j)
    return pts
end

# Basepoint x0 on the line with normal offset:
#   n = [-dir[2], dir[1]]
#   dot(n, x0) = offset
# We choose x0 = (offset / ||n||^2) * n, the projection of the origin onto the line.
function _line_basepoint_from_normal_offset_2d(dir::Vector{Float64}, offset::Float64)
    n1 = -dir[2]
    n2 =  dir[1]
    nn = n1*n1 + n2*n2
    if nn == 0.0
        error("slice_chain_exact_2d: direction vector must be nonzero")
    end
    s = offset / nn
    return [s*n1, s*n2]
end

# Parameter interval [tmin, tmax] for which x(t)=x0+t*dir lies inside axis-aligned box (a,b).
function _line_param_range_in_box_2d(x0::Vector{Float64}, dir::Vector{Float64},
                                     box::Tuple{Vector{Float64},Vector{Float64}};
                                     atol::Float64=1e-12)
    a, b = box
    tmin = -Inf
    tmax =  Inf
    @inbounds for k in 1:2
        dk = dir[k]
        ak = a[k]
        bk = b[k]
        if dk == 0.0
            # Coordinate is constant: must already lie in [a,b].
            if (x0[k] < min(ak,bk) - atol) || (x0[k] > max(ak,bk) + atol)
                return (Inf, -Inf)  # empty
            end
        else
            t1 = (ak - x0[k]) / dk
            t2 = (bk - x0[k]) / dk
            lo = min(t1, t2)
            hi = max(t1, t2)
            tmin = max(tmin, lo)
            tmax = min(tmax, hi)
        end
    end
    return (tmin, tmax)
end

# Convert a list of segment labels and boundary times into a compressed chain and boundary list.
# boundaries has length = length(labels)+1.
function _chain_values_from_boundaries(labels::Vector{Int}, boundaries::Vector{Float64};
                                       strict::Bool=true, atol::Float64=1e-12)
    if isempty(labels)
        return Int[], Float64[]
    end
    chain = Int[]
    values = Float64[]
    sizehint!(chain, length(labels))
    sizehint!(values, length(labels)+1)

    push!(values, boundaries[1])
    @inbounds for i in 1:length(labels)
        q = labels[i]
        if q == 0
            if strict
                error("slice_chain_exact_2d: encountered region label 0 inside the requested box; " *
                      "shrink box or pass strict=false")
            else
                # Non-strict behavior: skip. Note that interior 0-segments will be glued.
                continue
            end
        end
        if isempty(chain)
            push!(chain, q)
        elseif q != chain[end]
            push!(values, boundaries[i])   # boundary between segments i-1 and i
            push!(chain, q)
        end
    end
    push!(values, boundaries[end])

    return chain, values
end

# Critical points for coords-based (axis-aligned) backend:
# all grid vertices coords[1] x coords[2] (finite, inside box) + box corners.
function _critical_points_boxes_2d(pi, box::Tuple{Vector{Float64},Vector{Float64}}; atol::Float64=1e-12)
    a, b = box
    coords = getproperty(pi, :coords)
    xs_all = coords[1]
    ys_all = coords[2]

    xmin = min(a[1], b[1]) - atol
    xmax = max(a[1], b[1]) + atol
    ymin = min(a[2], b[2]) - atol
    ymax = max(a[2], b[2]) + atol

    xs = Float64[]
    ys = Float64[]
    for c in xs_all
        if isfinite(c)
            cf = float(c)
            if (cf >= xmin) && (cf <= xmax)
                push!(xs, cf)
            end
        end
    end
    for c in ys_all
        if isfinite(c)
            cf = float(c)
            if (cf >= ymin) && (cf <= ymax)
                push!(ys, cf)
            end
        end
    end

    pts = NTuple{2,Float64}[]
    sizehint!(pts, length(xs)*length(ys) + 4)
    for x in xs, y in ys
        push!(pts, (x,y))
    end
    # box corners
    push!(pts, (a[1],a[2]))
    push!(pts, (a[1],b[2]))
    push!(pts, (b[1],a[2]))
    push!(pts, (b[1],b[2]))

    _unique_points_2d!(pts; atol=atol)
    return pts
end

# Critical points for polyhedral backend:
# all vertices of each region inside box (when enumerable) + box corners.
function _critical_points_poly_2d(pi::PLPolyhedra.PLEncodingMap,
                                  box::Tuple{Vector{Float64},Vector{Float64}};
                                  max_combinations::Int=200000,
                                  max_vertices::Int=20000,
                                  atol::Float64=1e-12)
    a, b = box
    pts = NTuple{2,Float64}[]
    sizehint!(pts, 32)

    # Collect region vertices in the box. This is deterministic up to sorting below.
    for hp in pi.regions
        verts = PLPolyhedra._vertices_of_hpoly_in_box(hp, (a,b);
                                                     max_combinations=max_combinations,
                                                     max_vertices=max_vertices)
        if verts !== nothing
            for v in verts
                # v is a tuple of QQs (dimension 2 here).
                push!(pts, (float(v[1]), float(v[2])))
            end
        else
            # Fallback: include witness points if enumeration fails.
            # This does not guarantee exactness, but avoids crashes.
            # The exact algorithm is intended for cases where vertex enumeration succeeds.
            # (Witnesses are Float64 already.)
        end
    end

    # Add corners to ensure offset range covers box.
    push!(pts, (a[1],a[2]))
    push!(pts, (a[1],b[2]))
    push!(pts, (b[1],a[2]))
    push!(pts, (b[1],b[2]))

    _unique_points_2d!(pts; atol=atol)
    return pts
end

# Representative slopes: given sorted unique slopes s1<s2<...<sk, return midpoints of
# intervals (0,s1), (s1,s2), ..., (sk, +inf) as finite representatives.
function _representative_slopes(slopes::Vector{Float64}; atol::Float64=1e-12)
    _unique_sorted_floats!(slopes; atol=atol)
    if isempty(slopes)
        return [1.0]  # degenerate fallback
    end
    reps = Float64[]
    sizehint!(reps, length(slopes)+1)

    # (0, s1)
    push!(reps, 0.5*slopes[1])

    # (si, s(i+1))
    for i in 1:length(slopes)-1
        push!(reps, 0.5*(slopes[i] + slopes[i+1]))
    end

    # (sk, +inf) -> pick 2*sk
    push!(reps, 2.0*slopes[end])

    return reps
end

# Compute representative directions (normalized) from critical points.
function _critical_directions_2d(points::Vector{NTuple{2,Float64}};
                                 normalize_dirs::Symbol=:L1,
                                 include_axes::Bool=false,
                                 atol::Float64=1e-12)
    # Sort points by x so dx>=0 for j>i.
    pts = copy(points)
    sort!(pts, by=p->(p[1], p[2]))

    slopes = Float64[]
    n = length(pts)
    sizehint!(slopes, max(0, n*(n-1)>>>2))
    @inbounds for i in 1:n-1
        xi, yi = pts[i]
        for j in i+1:n
            xj, yj = pts[j]
            dx = xj - xi
            dy = yj - yi
            if dx > atol && dy > atol
                push!(slopes, dy/dx)
            end
        end
    end

    reps = _representative_slopes(slopes; atol=atol)

    dirs = Vector{Vector{Float64}}()
    sizehint!(dirs, length(reps) + (include_axes ? 2 : 0))

    for s in reps
        # direction with slope s in positive quadrant
        d = [1.0, s]
        d = _normalize_dir(d, normalize_dirs)
        # skip zero directions
        if (d[1] > 0.0) && (d[2] > 0.0)
            push!(dirs, d)
        end
    end

    if include_axes
        # normalized axis directions (robustly handled in slice extraction)
        push!(dirs, _normalize_dir([1.0, 0.0], normalize_dirs))
        push!(dirs, _normalize_dir([0.0, 1.0], normalize_dirs))
    end

    return dirs
end

# For a fixed direction, compute representative offsets (midpoints between consecutive dot products).
function _critical_offsets_2d(points::Vector{NTuple{2,Float64}}, dir::Vector{Float64};
                              atol::Float64=1e-12)
    n1 = -dir[2]
    n2 =  dir[1]
    cs = Float64[]
    sizehint!(cs, length(points))
    for p in points
        push!(cs, n1*p[1] + n2*p[2])
    end
    _unique_sorted_floats!(cs; atol=atol)
    if length(cs) < 2
        return Float64[]
    end
    offs = Float64[]
    sizehint!(offs, length(cs)-1)
    for i in 1:length(cs)-1
        push!(offs, 0.5*(cs[i] + cs[i+1]))
    end
    return offs
end

# Exact slice-chain extraction for coords-based backend.
function _slice_chain_exact_boxes_2d(pi, dir::Vector{Float64}, offset::Float64;
                                     box::Tuple{Vector{Float64},Vector{Float64}},
                                     strict::Bool=true, atol::Float64=1e-12)
    x0 = _line_basepoint_from_normal_offset_2d(dir, offset)
    tmin, tmax = _line_param_range_in_box_2d(x0, dir, box; atol=atol)
    if !(tmin < tmax)
        return Int[], Float64[]
    end

    a, b = box
    xmin = min(a[1], b[1])
    xmax = max(a[1], b[1])
    ymin = min(a[2], b[2])
    ymax = max(a[2], b[2])

    coords = getproperty(pi, :coords)
    xs = coords[1]
    ys = coords[2]

    ts = Float64[tmin, tmax]

    # Intersections with vertical grid lines x=c.
    if dir[1] != 0.0
        for c in xs
            if isfinite(c)
                cf = float(c)
                if (cf > xmin + atol) && (cf < xmax - atol)
                    t = (cf - x0[1]) / dir[1]
                    if (t > tmin + atol) && (t < tmax - atol)
                        push!(ts, t)
                    end
                end
            end
        end
    end

    # Intersections with horizontal grid lines y=c.
    if dir[2] != 0.0
        for c in ys
            if isfinite(c)
                cf = float(c)
                if (cf > ymin + atol) && (cf < ymax - atol)
                    t = (cf - x0[2]) / dir[2]
                    if (t > tmin + atol) && (t < tmax - atol)
                        push!(ts, t)
                    end
                end
            end
        end
    end

    _unique_sorted_floats!(ts; atol=atol)

    # Label each open segment by a midpoint query.
    labels = Vector{Int}(undef, length(ts)-1)
    x = [0.0, 0.0]
    @inbounds for i in 1:length(labels)
        tmid = 0.5*(ts[i] + ts[i+1])
        x[1] = x0[1] + tmid*dir[1]
        x[2] = x0[2] + tmid*dir[2]
        labels[i] = locate(pi, x)
    end

    return _chain_values_from_boundaries(labels, ts; strict=strict, atol=atol)
end

# Compute line-interval intersection with a single convex polyhedron Ax<=b.
function _line_interval_in_hpoly_2d(A::Matrix{Float64}, b::Vector{Float64},
                                    x0::Vector{Float64}, dir::Vector{Float64};
                                    atol::Float64=1e-12)
    tlo = -Inf
    thi =  Inf
    m = size(A,1)
    @inbounds for i in 1:m
        a1 = A[i,1]
        a2 = A[i,2]
        alpha = a1*dir[1] + a2*dir[2]
        beta  = b[i] - (a1*x0[1] + a2*x0[2])

        if abs(alpha) <= atol
            if beta < -atol
                return (Inf, -Inf)
            end
        elseif alpha > 0.0
            thi = min(thi, beta/alpha)
        else
            tlo = max(tlo, beta/alpha)
        end

        if tlo >= thi - atol
            return (Inf, -Inf)
        end
    end
    return (tlo, thi)
end

# Exact slice-chain extraction for polyhedral backend.
# We compute all region interval endpoints, sort them, then label segments by locate at midpoints.
function _slice_chain_exact_poly_2d(pi::PLPolyhedra.PLEncodingMap,
                                    dir::Vector{Float64}, offset::Float64;
                                    box::Tuple{Vector{Float64},Vector{Float64}},
                                    strict::Bool=true, atol::Float64=1e-12,
                                    region_cache=nothing)
    x0 = _line_basepoint_from_normal_offset_2d(dir, offset)
    tmin, tmax = _line_param_range_in_box_2d(x0, dir, box; atol=atol)
    if !(tmin < tmax)
        return Int[], Float64[]
    end

    # Precompute float A,b per region for speed if provided.
    A_list, b_list = if region_cache === nothing
        ([Float64.(hp.A) for hp in pi.regions],
         [Float64.(hp.b) for hp in pi.regions])
    else
        region_cache
    end

    ts = Float64[tmin, tmax]
    for r in 1:length(pi.regions)
        A = A_list[r]
        b = b_list[r]
        tlo, thi = _line_interval_in_hpoly_2d(A, b, x0, dir; atol=atol)
        tlo = max(tlo, tmin)
        thi = min(thi, tmax)
        if thi > tlo + atol
            push!(ts, tlo)
            push!(ts, thi)
        end
    end

    _unique_sorted_floats!(ts; atol=atol)

    labels = Vector{Int}(undef, length(ts)-1)
    x = [0.0, 0.0]
    @inbounds for i in 1:length(labels)
        tmid = 0.5*(ts[i] + ts[i+1])
        x[1] = x0[1] + tmid*dir[1]
        x[2] = x0[2] + tmid*dir[2]
        labels[i] = locate(pi, x)
    end

    return _chain_values_from_boundaries(labels, ts; strict=strict, atol=atol)
end

"""
    slice_chain_exact_2d(pi, dir, offset; box=nothing, normalize_dirs=:L1, strict=true, atol=1e-12)

Exact slice-chain extraction in 2D.

We consider the line with direction `dir` and normal offset `offset`:
  - Normalize `dir` according to `normalize_dirs`.
  - Define the normal `n = [-dir[2], dir[1]]`.
  - Choose basepoint `x0 = (offset/||n||^2) * n` so that `dot(n, x0) = offset`.
  - Consider the line segment inside `box` (an axis-aligned box given as `(a,b)`).

Return:
  - `chain::Vector{Int}`: region labels along the segment, merged by constant regions.
  - `values::Vector{Float64}`: boundary parameter values of length `length(chain)+1`.

Backends:
  - If `pi` has a `coords` field, we treat regions as an axis-aligned grid and intersect
    the line with all vertical/horizontal grid lines.
  - If `pi isa PLPolyhedra.PLEncodingMap`, we intersect the line with each region polyhedron.
"""
function slice_chain_exact_2d(pi, dir::AbstractVector{<:Real}, offset::Real;
                             box=nothing, normalize_dirs::Symbol=:L1,
                             strict::Bool=true, atol::Real=1e-12)
    if length(dir) != 2
        error("slice_chain_exact_2d is only defined for 2D directions")
    end

    bx = box === nothing ? encoding_box(pi) : box
    a, b = bx
    a = Float64.(a)
    b = Float64.(b)
    bx = (a,b)

    d = [float(dir[1]), float(dir[2])]
    d = _normalize_dir(d, normalize_dirs)

    off = float(offset)
    atol_f = float(atol)

    if pi isa PLPolyhedra.PLEncodingMap
        return _slice_chain_exact_poly_2d(pi, d, off; box=bx, strict=strict, atol=atol_f)
    elseif hasproperty(pi, :coords)
        return _slice_chain_exact_boxes_2d(pi, d, off; box=bx, strict=strict, atol=atol_f)
    else
        error("slice_chain_exact_2d: unsupported encoding map type; expected coords-based or PLPolyhedra.PLEncodingMap")
    end
end

"""
    matching_distance_exact_slices_2d(pi; box=nothing, weight=:lesnick_l1, normalize_dirs=:L1,
                                      include_axes=false, strict=true,
                                      max_combinations=200000, max_vertices=20000, atol=1e-12)

Deterministically enumerate the slice representatives used in the exact 2D matching distance.

Returns a vector of NamedTuples with fields:
  - `dir`    (Vector{Float64}, length 2)
  - `offset` (Float64), the normal offset c = dot(n, x) with n = [-dir[2], dir[1]]
  - `chain`  (Vector{Int})
  - `values` (Vector{Float64}, length chain+1)
  - `weight` (Float64)

One representative is produced per arrangement cell determined by the critical points.
"""
function matching_distance_exact_slices_2d(pi;
        box=nothing,
        weight::Symbol=:lesnick_l1,
        normalize_dirs::Symbol=:L1,
        include_axes::Bool=false,
        strict::Bool=true,
        max_combinations::Int=200000,
        max_vertices::Int=20000,
        atol::Real=1e-12)

    bx = box === nothing ? encoding_box(pi) : box
    a, b = bx
    a = Float64.(a)
    b = Float64.(b)
    bx = (a,b)
    atol_f = float(atol)

    # Collect critical points.
    points = if pi isa PLPolyhedra.PLEncodingMap
        _critical_points_poly_2d(pi, bx;
                                 max_combinations=max_combinations,
                                 max_vertices=max_vertices,
                                 atol=atol_f)
    elseif hasproperty(pi, :coords)
        _critical_points_boxes_2d(pi, bx; atol=atol_f)
    else
        error("matching_distance_exact_slices_2d: unsupported encoding map type")
    end

    # Directions and offsets.
    dirs = _critical_directions_2d(points;
                                   normalize_dirs=normalize_dirs,
                                   include_axes=include_axes,
                                   atol=atol_f)

    # For polyhedra, cache float A,b per region for speed across many offsets.
    region_cache = nothing
    if pi isa PLPolyhedra.PLEncodingMap
        region_cache = ([Float64.(hp.A) for hp in pi.regions],
                        [Float64.(hp.b) for hp in pi.regions])
    end

    slices = NamedTuple[]
    for d in dirs
        w = _direction_weight(d; scheme=weight)

        # If weight is 0, the slice cannot affect the maximum, but we still
        # generate it for completeness.
        offs = _critical_offsets_2d(points, d; atol=atol_f)

        for off in offs
            chain, values = if pi isa PLPolyhedra.PLEncodingMap
                _slice_chain_exact_poly_2d(pi, d, off; box=bx, strict=strict, atol=atol_f,
                                           region_cache=region_cache)
            else
                _slice_chain_exact_boxes_2d(pi, d, off; box=bx, strict=strict, atol=atol_f)
            end

            if !isempty(chain)
                push!(slices, (dir=d, offset=off, chain=chain, values=values, weight=w))
            end
        end
    end

    return slices
end

"""
    matching_distance_exact_2d(M, N, pi; box=nothing, weight=:lesnick_l1, normalize_dirs=:L1,
                              include_axes=false, strict=true, atol=1e-12,
                              max_combinations=200000, max_vertices=20000)

Exact 2D matching distance via deterministic arrangement-cell enumeration.

This computes:
  max_{dir,offset}  w(dir) * bottleneck_distance( B(M|slice), B(N|slice) )

where the maximum is taken over one representative slice per arrangement cell
determined by the critical points of `pi`.

For speed, this avoids any sampling along slices.
"""
function matching_distance_exact_2d(
    M::PModule{QQ},
    N::PModule{QQ},
    pi::EncodingMap;
    box = nothing,
    weight::Symbol = :lesnick_l1,
    normalize_dirs::Symbol = :L1,
    include_axes::Bool = false,
    strict::Bool = true,
    atol::Float64 = 1e-12,
    max_combinations::Int = 200000,
    max_vertices::Int = 20000,
    max_cells::Int = 5000000,
    precompute::Symbol = :cells_barcodes,
    store_values::Bool = true,
    threads::Bool = (Threads.nthreads() > 1),
)
    arr = fibered_arrangement_2d(pi;
        box = box,
        normalize_dirs = normalize_dirs,
        include_axes = include_axes,
        strict = strict,
        atol = atol,
        max_combinations = max_combinations,
        max_vertices = max_vertices,
        max_cells = max_cells,
        precompute = (precompute in (:cells, :cells_barcodes, :all) ? :cells : :none),
    )

    fam = fibered_slice_family_2d(arr; direction_weight = weight, store_values = store_values)

    cacheM = fibered_barcode_cache_2d(M, arr; precompute = precompute)
    cacheN = fibered_barcode_cache_2d(N, arr; precompute = precompute)

    return matching_distance_exact_2d(cacheM, cacheN;
        weight = weight,
        family = fam,
        threads = threads,
    )
end


# Typed barcode representations for fibered 2D computations.
# IndexBarcode: endpoints stored as integer indices into a values vector.
# FloatBarcode: endpoints stored as Float64 boundary parameter values.
const IndexBarcode = Dict{Tuple{Int,Int},Int}
const FloatBarcode = Dict{Tuple{Float64,Float64},Int}

# ----- Fast fibered-barcode queries in 2D: RIVET-style augmented arrangement -----
#
# RIVET's key idea:
#   In 2D, the slice chain (and therefore the index barcode) is constant on the
#   2-cells of a line arrangement in (direction, offset) space.
#
# This section implements a mathematician-friendly and reusable cache:
#   - FiberedArrangement2D: geometry-only point-location structure + per-cell chain/events
#   - FiberedBarcodeCache2D: module-augmented cache of index barcodes per chain
#
# Query time (typical, nondegenerate):
#   O(log N + m + |barcode|)
# where m = chain length of the slice (usually small/moderate).
#
# This provides the "fast repeated queries" capability requested in the screenshot.

const _AxisCoordIndex2D = NTuple{2,Int}

"""
    FiberedArrangement2D

Geometry-only cache for fast point-location of 2D lines in the space of
(direction, normal offset).  It stores:

- critical slopes (direction changes where order of offsets changes)
- for each direction cell, a fixed ordering of critical points by normal offset
- for each (direction cell, offset cell), a cached slice chain (computed lazily or eagerly)
- for the boxes backend, a compact event description for reconstructing boundary parameter
  values quickly for any query line in the same cell

This object is independent of any module `M` and can be shared between many modules.

Construct with [`fibered_arrangement_2d`](@ref).
Augment with [`fibered_barcode_cache_2d`](@ref).
"""
mutable struct FiberedArrangement2D
    pi::Any
    box::Tuple{Vector{Float64},Vector{Float64}}
    normalize_dirs::Symbol
    include_axes::Bool
    strict::Bool
    atol::Float64

    backend::Symbol
    points::Vector{NTuple{2,Float64}}
    slope_breaks::Vector{Float64}
    dir_reps::Vector{Vector{Float64}}

    # For each direction cell:
    # - orders[k] is a permutation of critical points sorted by normal offset
    # - unique_pos[k] are positions in that ordering representing unique offset levels
    orders::Vector{Vector{Int}}
    unique_pos::Vector{Vector{Int}}

    # number of offset cells per direction cell
    noff::Vector{Int}

    # prefix starts for flattened storage
    start::Vector{Int}
    total_cells::Int

    # Per-cell cache:
    #  0 = not computed yet
    # -1 = empty slice
    # >0 = chain_id
    cell_chain_id::Vector{Int}

    # Boxes backend events are stored in a pool; each cell stores (start,len)
    cell_event_start::Vector{Int}
    cell_event_len::Vector{Int}
    event_pool::Vector{_AxisCoordIndex2D}

    # coordinate lists for boxes backend (filtered to box)
    xs::Vector{Float64}
    ys::Vector{Float64}

    # poly backend cache: (A_list, b_list)
    region_cache::Any

    # chain registry
    chain_key_to_id::Dict{Vector{Int},Int}
    chains::Vector{Vector{Int}}

    # stats: how many cells have been computed
    n_cell_computed::Int

    # Cache of precomputed slice families keyed by (direction_weight_scheme, store_values).
    # Stored as Any to avoid forward-type dependency in this struct definition.
    slice_family_cache::Dict{Tuple{Symbol,Bool},Any}
end

"""
    FiberedBarcodeCache2D

Module-augmented cache on top of [`FiberedArrangement2D`](@ref).
It memoizes the *index barcode* for each chain encountered so that repeated
fibered barcode queries are fast.

Construct with [`fibered_barcode_cache_2d`](@ref).
Query with [`fibered_barcode`](@ref).
"""
mutable struct FiberedBarcodeCache2D
    arrangement::FiberedArrangement2D
    M::PModule{QQ}

    # index_barcodes[i] is either nothing or an IndexBarcode
    index_barcodes::Vector{Union{Nothing,IndexBarcode}}
    n_barcode_computed::Int
end

# ------------------------ internal helpers ------------------------

@inline function _as_float2(v)::Vector{Float64}
    if length(v) != 2
        throw(ArgumentError("expected a 2-vector"))
    end
    return [Float64(v[1]), Float64(v[2])]
end

@inline function _normal_from_dir_2d(d::Vector{Float64})
    return (-d[2], d[1])
end

function _critical_slopes_2d(points::Vector{NTuple{2,Float64}}; atol::Float64=1e-12)
    n = length(points)
    slopes = Float64[]
    for i in 1:n-1
        xi, yi = points[i]
        for j in i+1:n
            xj, yj = points[j]
            dx = xj - xi
            dy = yj - yi
            if dx > atol && dy > atol
                push!(slopes, dy/dx)
            end
        end
    end
    sort!(slopes)
    _unique_sorted_floats!(slopes; atol=atol)
    return slopes
end

function _fibered_dir_cell_index(arr::FiberedArrangement2D, d::Vector{Float64})::Int
    if d[1] < -arr.atol || d[2] < -arr.atol
        throw(ArgumentError("direction must lie in the nonnegative quadrant"))
    end
    if abs(d[1]) <= arr.atol && abs(d[2]) <= arr.atol
        throw(ArgumentError("direction must be nonzero"))
    end

    ndir_pos = length(arr.slope_breaks) + 1

    if arr.include_axes
        if abs(d[2]) <= arr.atol && d[1] > arr.atol
            return ndir_pos + 1
        elseif abs(d[1]) <= arr.atol && d[2] > arr.atol
            return ndir_pos + 2
        end
    else
        if abs(d[1]) <= arr.atol || abs(d[2]) <= arr.atol
            throw(ArgumentError("axis directions require include_axes=true"))
        end
    end

    if d[1] <= arr.atol || d[2] <= arr.atol
        throw(ArgumentError("direction must have strictly positive entries unless include_axes=true"))
    end

    slope = d[2] / d[1]
    k = searchsortedfirst(arr.slope_breaks, slope)
    return k
end

function _fibered_offset_cell_index(
    arr::FiberedArrangement2D,
    dir_idx::Int,
    d::Vector{Float64},
    off::Float64;
    tie_break::Symbol = :up,
)
    pos = arr.unique_pos[dir_idx]
    order = arr.orders[dir_idx]
    pts = arr.points
    (n1, n2) = _normal_from_dir_2d(d)

    nu = length(pos)
    if nu < 2
        return 0
    end

    pmin = pts[order[pos[1]]]
    pmax = pts[order[pos[end]]]
    cmin = n1*pmin[1] + n2*pmin[2]
    cmax = n1*pmax[1] + n2*pmax[2]

    if off <= cmin + arr.atol || off >= cmax - arr.atol
        return 0
    end

    lo = 1
    hi = nu
    while lo + 1 < hi
        mid = (lo + hi) >>> 1
        pmid = pts[order[pos[mid]]]
        cmid = n1*pmid[1] + n2*pmid[2]

        if tie_break === :up
            if cmid <= off + arr.atol
                lo = mid
            else
                hi = mid
            end
        elseif tie_break === :down
            if cmid < off - arr.atol
                lo = mid
            else
                hi = mid
            end
        else
            throw(ArgumentError("tie_break must be :up or :down"))
        end
    end

    return lo
end

@inline function _arr2d_cell_linear_index(arr::FiberedArrangement2D, dir_idx::Int, off_idx::Int)
    return arr.start[dir_idx] + off_idx - 1
end

function _arr2d_chain_id!(arr::FiberedArrangement2D, chain::Vector{Int})::Int
    # IMPORTANT: `chain` is used as a Dict key. Do not mutate it after insertion.
    id = get(arr.chain_key_to_id, chain, 0)
    if id == 0
        id = length(arr.chains) + 1
        arr.chain_key_to_id[chain] = id
        push!(arr.chains, chain)
    end
    return id
end

function _nearest_coord_index(coords::Vector{Float64}, v::Float64; atol::Float64=1e-8)
    n = length(coords)
    if n == 0
        return 0
    end
    i = searchsortedfirst(coords, v)
    best = 0
    bestd = Inf

    for j in (i-1, i, i+1)
        if 1 <= j <= n
            d = abs(coords[j] - v)
            if d < bestd
                bestd = d
                best = j
            end
        end
    end

    return (bestd <= atol) ? best : 0
end

function _slice_chain_exact_boxes_2d_fast(
    pi,
    dir::Vector{Float64},
    offset::Float64;
    box,
    xs::Vector{Float64},
    ys::Vector{Float64},
    strict::Bool,
    atol::Float64,
)
    x0 = _line_basepoint_from_normal_offset_2d(dir, offset)
    tmin, tmax = _line_param_range_in_box_2d(x0, dir, box; atol=atol)
    if tmax <= tmin + atol
        return Int[], Float64[]
    end

    a, b = box
    xmin, ymin = a
    xmax, ymax = b

    ts = Float64[tmin, tmax]

    if abs(dir[1]) > atol
        for x in xs
            if x > xmin + atol && x < xmax - atol
                t = (x - x0[1]) / dir[1]
                if t > tmin + atol && t < tmax - atol
                    push!(ts, t)
                end
            end
        end
    end
    if abs(dir[2]) > atol
        for y in ys
            if y > ymin + atol && y < ymax - atol
                t = (y - x0[2]) / dir[2]
                if t > tmin + atol && t < tmax - atol
                    push!(ts, t)
                end
            end
        end
    end

    sort!(ts)
    _unique_sorted_floats!(ts; atol=atol)

    labels = Vector{Int}(undef, length(ts)-1)
    for i in 1:length(labels)
        tm = 0.5*(ts[i] + ts[i+1])
        xm = x0 .+ tm .* dir
        labels[i] = locate(pi, xm)
    end

    chain, values = _chain_values_from_boundaries(labels, ts; strict=strict, atol=atol)
    return chain, values
end

function _arr2d_slice_chain_and_values(arr::FiberedArrangement2D, dir::Vector{Float64}, off::Float64)
    if arr.backend === :boxes
        return _slice_chain_exact_boxes_2d_fast(
            arr.pi, dir, off;
            box=arr.box,
            xs=arr.xs,
            ys=arr.ys,
            strict=arr.strict,
            atol=arr.atol,
        )
    else
        return _slice_chain_exact_poly_2d(
            arr.pi, dir, off;
            box=arr.box,
            strict=arr.strict,
            atol=arr.atol,
            region_cache=arr.region_cache,
        )
    end
end

function _events_from_values_boxes(arr::FiberedArrangement2D, dir::Vector{Float64}, off::Float64, values::Vector{Float64})
    if length(values) <= 2
        return _AxisCoordIndex2D[]
    end
    x0 = _line_basepoint_from_normal_offset_2d(dir, off)
    events = Vector{_AxisCoordIndex2D}(undef, length(values)-2)

    snap_tol = max(1e-8, 100*arr.atol)

    for k in 1:length(events)
        t = values[k+1]
        x = x0[1] + t*dir[1]
        y = x0[2] + t*dir[2]

        ix = _nearest_coord_index(arr.xs, x; atol=snap_tol)
        iy = _nearest_coord_index(arr.ys, y; atol=snap_tol)

        if ix != 0
            events[k] = (1, ix)
        elseif iy != 0
            events[k] = (2, iy)
        else
            throw(ErrorException("could not snap boundary point to a grid line"))
        end
    end

    return events
end

function _arr2d_compute_cell!(arr::FiberedArrangement2D, dir_idx::Int, off_idx::Int)
    cell = _arr2d_cell_linear_index(arr, dir_idx, off_idx)
    cached = arr.cell_chain_id[cell]
    if cached != 0
        return cached
    end

    # Use representative direction for the direction cell.
    drep = arr.dir_reps[dir_idx]

    # Representative offset: midpoint between consecutive unique levels for drep.
    order = arr.orders[dir_idx]
    pos = arr.unique_pos[dir_idx]
    pL = arr.points[order[pos[off_idx]]]
    pR = arr.points[order[pos[off_idx+1]]]
    (n1, n2) = _normal_from_dir_2d(drep)
    cL = n1*pL[1] + n2*pL[2]
    cR = n1*pR[1] + n2*pR[2]
    off_mid = 0.5*(cL + cR)

    chain, vals = _arr2d_slice_chain_and_values(arr, drep, off_mid)
    arr.n_cell_computed += 1

    if isempty(chain)
        arr.cell_chain_id[cell] = -1
        return -1
    end

    cid = _arr2d_chain_id!(arr, chain)
    arr.cell_chain_id[cell] = cid

    if arr.backend === :boxes
        ev = _events_from_values_boxes(arr, drep, off_mid, vals)
        s = length(arr.event_pool) + 1
        append!(arr.event_pool, ev)
        arr.cell_event_start[cell] = s
        arr.cell_event_len[cell] = length(ev)
    end

    return cid
end

function _arr2d_values_from_cell_boxes(
    arr::FiberedArrangement2D,
    cell::Int,
    d::Vector{Float64},
    off::Float64,
    chain_len::Int,
)
    x0 = _line_basepoint_from_normal_offset_2d(d, off)
    tmin, tmax = _line_param_range_in_box_2d(x0, d, arr.box; atol=arr.atol)
    if tmax <= tmin + arr.atol
        return Float64[]
    end

    nvals = chain_len + 1
    values = Vector{Float64}(undef, nvals)
    values[1] = tmin
    values[end] = tmax

    s = arr.cell_event_start[cell]
    l = arr.cell_event_len[cell]
    if l != nvals - 2
        throw(ErrorException("event count does not match chain length"))
    end

    for k in 1:l
        axis, idx = arr.event_pool[s + k - 1]
        if axis == 1
            x = arr.xs[idx]
            values[k+1] = (x - x0[1]) / d[1]
        else
            y = arr.ys[idx]
            values[k+1] = (y - x0[2]) / d[2]
        end
    end

    # Enforce monotonicity against numerical noise.
    for i in 2:length(values)
        if values[i] < values[i-1]
            values[i] = values[i-1]
        end
    end

    return values
end

function _arr2d_values_for_chain_poly(arr::FiberedArrangement2D, d::Vector{Float64}, off::Float64, chain::Vector{Int})
    x0 = _line_basepoint_from_normal_offset_2d(d, off)
    tmin, tmax = _line_param_range_in_box_2d(x0, d, arr.box; atol=arr.atol)
    if tmax <= tmin + arr.atol
        return Float64[]
    end

    A_list, b_list = arr.region_cache
    m = length(chain)
    intervals = Vector{Tuple{Float64,Float64}}(undef, m)
    mids = Vector{Float64}(undef, m)

    for i in 1:m
        r = chain[i]
        tlo, thi = _line_interval_in_hpoly_2d(A_list[r], b_list[r], x0, d; atol=arr.atol)
        tlo = max(tlo, tmin)
        thi = min(thi, tmax)
        if thi <= tlo + arr.atol
            throw(ErrorException("empty region-line intersection; likely degenerate query"))
        end
        intervals[i] = (tlo, thi)
        mids[i] = 0.5*(tlo + thi)
    end

    perm = sortperm(mids)
    values = Vector{Float64}(undef, m+1)
    values[1] = tmin
    for k in 1:m-1
        values[k+1] = intervals[perm[k]][2]
    end
    values[end] = tmax

    for i in 2:length(values)
        if values[i] < values[i-1]
            values[i] = values[i-1]
        end
    end

    return values
end

function _sync_index_barcodes!(cache::FiberedBarcodeCache2D)
    n = length(cache.arrangement.chains)
    while length(cache.index_barcodes) < n
        push!(cache.index_barcodes, nothing)
    end
    return nothing
end

function _index_barcode_for_chain!(cache::FiberedBarcodeCache2D, chain_id::Int)
    _sync_index_barcodes!(cache)
    b = cache.index_barcodes[chain_id]
    if b !== nothing
        return b
    end
    chain = cache.arrangement.chains[chain_id]
    bidx = slice_barcode(cache.M, chain; values=nothing, check_chain=false)
    cache.index_barcodes[chain_id] = bidx
    cache.n_barcode_computed += 1
    return bidx
end

function _precompute_cells!(arr::FiberedArrangement2D)
    for dir_idx in 1:length(arr.dir_reps)
        for off_idx in 1:arr.noff[dir_idx]
            _arr2d_compute_cell!(arr, dir_idx, off_idx)
        end
    end
    return nothing
end

function _precompute_index_barcodes!(cache::FiberedBarcodeCache2D)
    arr = cache.arrangement
    _sync_index_barcodes!(cache)
    for i in 1:length(arr.chains)
        _index_barcode_for_chain!(cache, i)
    end
    return nothing
end

# ------------------------ public constructors ------------------------

"""
    fibered_arrangement_2d(pi; box=nothing, normalize_dirs=:L1, include_axes=false,
                           strict=true, atol=1e-12,
                           max_combinations=200000, max_vertices=20000,
                           max_cells=5_000_000,
                           precompute=:none)

Build a geometry-only arrangement cache for fast fibered-barcode queries in 2D.

`precompute`:
- `:none`  : build point-location structure; compute cells lazily on demand
- `:cells` : compute all cells immediately

This object can be shared across modules.
"""
function fibered_arrangement_2d(
    pi;
    box=nothing,
    normalize_dirs::Symbol=:L1,
    include_axes::Bool=false,
    strict::Bool=true,
    atol::Float64=1e-12,
    max_combinations::Int=200_000,
    max_vertices::Int=20_000,
    max_cells::Int=5_000_000,
    precompute::Symbol=:none,
)
    bx_raw = box === nothing ? encoding_box(pi) : box
    bx = ([Float64(bx_raw[1][1]), Float64(bx_raw[1][2])],
          [Float64(bx_raw[2][1]), Float64(bx_raw[2][2])])

    backend = :boxes
    xs = Float64[]
    ys = Float64[]
    region_cache = nothing

    if pi isa PLPolyhedra.PLEncodingMap
        backend = :poly
        points = _critical_points_poly_2d(pi, bx;
            max_combinations=max_combinations,
            max_vertices=max_vertices,
            atol=atol,
        )
        region_cache = ([Float64.(hp.A) for hp in pi.regions],
                        [Float64.(hp.b) for hp in pi.regions])
        xs = Float64[]
        ys = Float64[]
    else
        if !hasproperty(pi, :coords)
            throw(ArgumentError("fibered_arrangement_2d requires either PLEncodingMap or a boxes backend with coords"))
        end
        points = _critical_points_boxes_2d(pi, bx; atol=atol)

        a, b = bx
        xmin, ymin = a
        xmax, ymax = b

        xs = [Float64(x) for x in pi.coords[1] if isfinite(x) && x >= xmin - atol && x <= xmax + atol]
        ys = [Float64(y) for y in pi.coords[2] if isfinite(y) && y >= ymin - atol && y <= ymax + atol]
        sort!(xs); _unique_sorted_floats!(xs; atol=atol)
        sort!(ys); _unique_sorted_floats!(ys; atol=atol)
    end

    slopes = _critical_slopes_2d(points; atol=atol)
    reps = _representative_slopes(slopes)

    dir_reps = Vector{Vector{Float64}}()
    for s in reps
        push!(dir_reps, _normalize_dir([1.0, s], normalize_dirs))
    end

    if include_axes
        push!(dir_reps, _normalize_dir([1.0, 0.0], normalize_dirs))
        push!(dir_reps, _normalize_dir([0.0, 1.0], normalize_dirs))
    end

    orders = Vector{Vector{Int}}(undef, length(dir_reps))
    unique_pos = Vector{Vector{Int}}(undef, length(dir_reps))
    noff = Vector{Int}(undef, length(dir_reps))

    npts = length(points)

    for k in 1:length(dir_reps)
        d = dir_reps[k]
        (n1, n2) = _normal_from_dir_2d(d)
        cs = Vector{Float64}(undef, npts)
        for i in 1:npts
            p = points[i]
            cs[i] = n1*p[1] + n2*p[2]
        end
        ord = sortperm(cs)
        orders[k] = ord

        pos = Int[]
        if npts > 0
            push!(pos, 1)
            prev = cs[ord[1]]
            for j in 2:npts
                c = cs[ord[j]]
                if abs(c - prev) > atol
                    push!(pos, j)
                    prev = c
                end
            end
        end
        unique_pos[k] = pos
        noff[k] = max(0, length(pos) - 1)
    end

    start = Vector{Int}(undef, length(dir_reps))
    acc = 1
    for k in 1:length(dir_reps)
        start[k] = acc
        acc += noff[k]
    end
    total_cells = acc - 1

    if total_cells > max_cells
        throw(ArgumentError("arrangement too large: total_cells=$total_cells exceeds max_cells=$max_cells"))
    end

    cell_chain_id = fill(0, total_cells)
    cell_event_start = fill(0, total_cells)
    cell_event_len = fill(0, total_cells)
    event_pool = _AxisCoordIndex2D[]

    arr = FiberedArrangement2D(
        pi, bx, normalize_dirs, include_axes, strict, atol,
        backend, points, slopes, dir_reps,
        orders, unique_pos,
        noff, start, total_cells,
        cell_chain_id,
        cell_event_start, cell_event_len, event_pool,
        xs, ys,
        region_cache,
        Dict{Vector{Int},Int}(),
        Vector{Vector{Int}}(),
        0,
        Dict{Tuple{Symbol,Bool},Any}()
    )

    if precompute === :cells
        _precompute_cells!(arr)
    elseif precompute !== :none
        throw(ArgumentError("precompute must be :none or :cells"))
    end

    return arr
end

"""
    fibered_barcode_cache_2d(M, pi; arrangement=nothing, precompute=:none, kwargs...)
    fibered_barcode_cache_2d(M, arrangement::FiberedArrangement2D; precompute=:none)

Create a module-augmented cache for fast fibered barcode queries.

This is the "augmented arrangement" layer (RIVET-style caching):

- A `FiberedArrangement2D` (geometry only) partitions the space of slice parameters
  (direction, normal offset) into 2-cells on which the *slice chain* is constant.
- A `FiberedBarcodeCache2D` then augments that arrangement with a module `M` and memoizes
  the 1D *index barcode* for each slice chain as it is encountered.

The recommended workflow is to build *one* arrangement and then build *many* module caches
on top of it. This makes repeated slicing, matching distances, and kernels fast because
the expensive geometric bookkeeping is shared.

Precomputation options
- `:none`  : lazy cells and lazy index barcodes (default)
- `:cells` : precompute all arrangement cells; index barcodes are still lazy
- `:full`  : precompute all arrangement cells and all index barcodes

Example: shared arrangement workflow

    using PosetModules
    const PM  = PosetModules
    const Inv = PM.Invariants

    # Inputs:
    #   pi : an encoding map in R^2 (boxes backend or PLEncodingMap)
    #   M,N: 2-parameter persistence modules over the same base field
    pi = ...
    M  = ...
    N  = ...

    # 1. Build ONE geometry-only arrangement (share across many modules).
    #
    # The "arrangement" is the expensive part: it organizes the (dir, offset) plane
    # into cells where the combinatorics of the slice restriction is constant.
    arr = Inv.fibered_arrangement_2d(pi;
        box = Inv.encoding_box(pi),   # optional bounding box
        normalize_dirs = :L1,         # must match later queries
        include_axes = false,         # set true to include axis directions
        precompute = :cells,          # optional: eager cell computation
    )

    # 2. Build MANY module caches on the SAME arrangement object.
    #
    # Each cache stores only the module `M` plus memoized barcodes for slice chains.
    cacheM = Inv.fibered_barcode_cache_2d(M, arr)
    cacheN = Inv.fibered_barcode_cache_2d(N, arr)

    # 3a. Single slice query (direction + normal offset).
    # This returns a 1D barcode as a Dict((birth, death) => multiplicity).
    bar = Inv.fibered_barcode(cacheM, [1.0, 1.0], 0.25)

    # 3b. Many slice queries at once (matrix of barcodes).
    #
    # `offsets` can be either:
    #   - normal offsets (Real), or
    #   - basepoints in R^2 (length-2 vector/tuple), which are converted to normal offsets.
    out = Inv.slice_barcodes(cacheM;
        dirs = [[1.0, 1.0], [1.0, 2.0]],
        offsets = [0.0, 0.25],        # normal offsets; or basepoints like [x0, y0]
        values = :t,                  # barcode endpoints in the 1D parameter t
        direction_weight = :lesnick_l1,
    )
    barcodes = out.barcodes  # ndirs x noffsets matrix of Dicts
    weights  = out.weights   # ndirs x noffsets matrix (normalized by default)

    # 4. Higher-level computations that reuse the same arrangement.
    #
    # Exact matching distance by enumerating one representative slice per arrangement cell.
    d_match = Inv.matching_distance_exact_2d(cacheM, cacheN; weight = :lesnick_l1)

    # Kernel computed by integrating a per-slice kernel over arrangement cells.
    k = Inv.slice_kernel(cacheM, cacheN;
        kind = :gaussian,
        direction_weight = :lesnick_l1,
        cell_weight = :uniform,
    )

Notes
- `matching_distance_exact_2d(cacheM, cacheN)` and `slice_kernel(cacheM, cacheN)` require
  `cacheM.arrangement === cacheN.arrangement` (the same arrangement object).
- If you do not pass an existing arrangement, `fibered_barcode_cache_2d(M, pi; ...)` will
  build a fresh one (convenient, but not shared across modules unless you pass it around).
"""
function fibered_barcode_cache_2d(
    M::PModule{QQ},
    pi;
    arrangement=nothing,
    precompute::Symbol=:none,
    box=nothing,
    normalize_dirs::Symbol=:L1,
    include_axes::Bool=false,
    strict::Bool=true,
    atol::Float64=1e-12,
    max_combinations::Int=200_000,
    max_vertices::Int=20_000,
    max_cells::Int=5_000_000,
)
    arr = arrangement === nothing ? fibered_arrangement_2d(
        pi;
        box=box,
        normalize_dirs=normalize_dirs,
        include_axes=include_axes,
        strict=strict,
        atol=atol,
        max_combinations=max_combinations,
        max_vertices=max_vertices,
        max_cells=max_cells,
        precompute=(precompute === :none ? :none : :cells),
    ) : arrangement

    cache = FiberedBarcodeCache2D(arr, M, Any[], 0)

    if precompute === :full
        _precompute_cells!(arr)
        _precompute_index_barcodes!(cache)
    elseif precompute === :cells
        _precompute_cells!(arr)
    elseif precompute !== :none
        throw(ArgumentError("precompute must be :none, :cells, or :full"))
    end

    return cache
end

function fibered_barcode_cache_2d(M::PModule{QQ}, arr::FiberedArrangement2D; precompute::Symbol=:none)
    cache = FiberedBarcodeCache2D(arr, M, Any[], 0)
    if precompute === :full
        _precompute_cells!(arr)
        _precompute_index_barcodes!(cache)
    elseif precompute !== :none
        throw(ArgumentError("precompute must be :none or :full for this constructor"))
    end
    return cache
end

"""
    fibered_barcode_cache_2d(M, arr; precompute=:none)

Build a module-specific cache over a shared arrangement.

`precompute` can include:
- :none
- :cells
- :barcodes
- :cells_barcodes (synonym: :full)
"""
function fibered_barcode_cache_2d(
    M::PModule{QQ},
    arr::FiberedArrangement2D;
    precompute::Symbol = :none,
)
    # normalize synonyms
    if precompute == :full
        precompute = :cells_barcodes
    end

    cache = FiberedBarcodeCache2D(arr, M, Union{Nothing,IndexBarcode}[], 0)

    if precompute in (:cells, :cells_barcodes, :all)
        _precompute_cells!(arr)
    end
    if precompute in (:barcodes, :cells_barcodes, :all)
        _precompute_index_barcodes!(cache)
    end
    return cache
end


# ------------------------ public query API ------------------------

"""
    fibered_cell_id(arr, dir, offset; tie_break=:up)
    fibered_cell_id(arr, dir, x0::AbstractVector; tie_break=:up)

Return the arrangement cell id `(dir_cell, offset_cell)` for the line determined by:

- `dir` and normal-offset `offset`
- `dir` and basepoint `x0` (offset is computed as dot(n, x0))

Returns `nothing` if the line misses the arrangement box or lies on its boundary.
"""
function fibered_cell_id(arr::FiberedArrangement2D, dir, offset::Real; tie_break::Symbol=:up)
    d = _normalize_dir(_as_float2(dir), arr.normalize_dirs)
    dir_idx = _fibered_dir_cell_index(arr, d)
    off_idx = _fibered_offset_cell_index(arr, dir_idx, d, Float64(offset); tie_break=tie_break)
    if off_idx == 0
        return nothing
    end
    return (dir_idx, off_idx)
end

function fibered_cell_id(arr::FiberedArrangement2D, dir, x0::AbstractVector{<:Real}; tie_break::Symbol=:up)
    d = _normalize_dir(_as_float2(dir), arr.normalize_dirs)
    (n1, n2) = _normal_from_dir_2d(d)
    off = n1*Float64(x0[1]) + n2*Float64(x0[2])
    return fibered_cell_id(arr, d, off; tie_break=tie_break)
end

"""
    fibered_chain(arr, dir, offset; tie_break=:up, copy=true)

Return the slice chain (as a vector of region labels) for the given line.
The chain is computed lazily and then cached in the arrangement cell.
"""
function fibered_chain(arr::FiberedArrangement2D, dir, offset::Real; tie_break::Symbol=:up, copy::Bool=true)
    cid = fibered_cell_id(arr, dir, offset; tie_break=tie_break)
    if cid === nothing
        return Int[]
    end
    d = _normalize_dir(_as_float2(dir), arr.normalize_dirs)
    dir_idx, off_idx = cid
    chain_id = _arr2d_compute_cell!(arr, dir_idx, off_idx)
    if chain_id <= 0
        return Int[]
    end
    return copy ? Base.copy(arr.chains[chain_id]) : arr.chains[chain_id]
end

"""
    fibered_values(arr, dir, offset; tie_break=:up)

Return boundary parameter values along the line segment inside `arr.box`.
These are the `t` values used by `slice_barcode` to map index endpoints to real endpoints.
"""
function fibered_values(arr::FiberedArrangement2D, dir, offset::Real; tie_break::Symbol=:up)
    cid = fibered_cell_id(arr, dir, offset; tie_break=tie_break)
    if cid === nothing
        return Float64[]
    end
    d = _normalize_dir(_as_float2(dir), arr.normalize_dirs)
    off = Float64(offset)
    dir_idx, off_idx = cid
    cell = _arr2d_cell_linear_index(arr, dir_idx, off_idx)
    chain_id = _arr2d_compute_cell!(arr, dir_idx, off_idx)
    if chain_id <= 0
        return Float64[]
    end
    chain = arr.chains[chain_id]

    if arr.backend === :boxes
        return _arr2d_values_from_cell_boxes(arr, cell, d, off, length(chain))
    else
        # poly: try chain-only reconstruction; fallback to exact if degenerate
        try
            return _arr2d_values_for_chain_poly(arr, d, off, chain)
        catch
            _, vals2 = slice_chain_exact_2d(arr.pi, d, off;
                box=arr.box, normalize_dirs=:none, strict=arr.strict, atol=arr.atol)
            return vals2
        end
    end
end

"""
    fibered_barcode(cache, dir, offset; values=:t, tie_break=:up, verify=false)
    fibered_barcode(cache, dir, x0::AbstractVector; values=:t, tie_break=:up, verify=false)

Return the 1D barcode of the slice of `cache.M` along the line determined by:

- `dir` and normal-offset `offset`
- `dir` and basepoint `x0`

`values=:t` returns a barcode whose endpoints are real parameter values.
`values=:index` returns the index barcode.

If `verify=true`, cross-check with `slice_chain_exact_2d` (slow, debug only).
"""
function fibered_barcode(
    cache::FiberedBarcodeCache2D,
    dir,
    offset::Real;
    values::Symbol = :t,
    tie_break::Symbol = :up,
    verify::Bool = false,
)
    arr = cache.arrangement
    d = _normalize_dir(_as_float2(dir), arr.normalize_dirs)
    off = Float64(offset)

    cid = fibered_cell_id(arr, d, off; tie_break=tie_break)
    if cid === nothing
        return Dict{Tuple{Float64,Float64},Int}()
    end
    dir_idx, off_idx = cid
    cell = _arr2d_cell_linear_index(arr, dir_idx, off_idx)
    chain_id = _arr2d_compute_cell!(arr, dir_idx, off_idx)
    if chain_id <= 0
        return Dict{Tuple{Float64,Float64},Int}()
    end

    bidx = _index_barcode_for_chain!(cache, chain_id)
    if values === :index
        return Base.copy(bidx)
    elseif values !== :t
        throw(ArgumentError("values must be :t or :index"))
    end

    chain = arr.chains[chain_id]
    vals = Float64[]
    ok = true

    if arr.backend === :boxes
        try
            vals = _arr2d_values_from_cell_boxes(arr, cell, d, off, length(chain))
        catch
            ok = false
        end
    else
        try
            vals = _arr2d_values_for_chain_poly(arr, d, off, chain)
        catch
            ok = false
        end
    end

    if !ok || isempty(vals)
        # fallback exact (handles degenerate/boundary queries robustly)
        chain2, vals2 = slice_chain_exact_2d(arr.pi, d, off;
            box=arr.box, normalize_dirs=:none, strict=arr.strict, atol=arr.atol)
        if isempty(chain2)
            return Dict{Tuple{Float64,Float64},Int}()
        end
        cid2 = _arr2d_chain_id!(arr, chain2)
        bidx2 = _index_barcode_for_chain!(cache, cid2)
        b = Dict{Tuple{Float64,Float64},Int}()
        sizehint!(b, length(bidx2))
        for ((i,j), mult) in bidx2
            b[(vals2[i], vals2[j])] = mult
        end
        return b
    end

    b = Dict{Tuple{Float64,Float64},Int}()
    sizehint!(b, length(bidx))
    for ((i,j), mult) in bidx
        b[(vals[i], vals[j])] = mult
    end

    if verify
        chain2, vals2 = slice_chain_exact_2d(arr.pi, d, off;
            box=arr.box, normalize_dirs=:none, strict=arr.strict, atol=arr.atol)
        if chain2 != chain
            throw(ErrorException("verification failed: cached chain differs from exact chain"))
        end
        if length(vals2) == length(vals)
            for k in 1:length(vals)
                if abs(vals2[k] - vals[k]) > 1e-8
                    throw(ErrorException("verification failed: cached values differ from exact values"))
                end
            end
        end
    end

    return b
end

function fibered_barcode(cache::FiberedBarcodeCache2D, dir, x0::AbstractVector{<:Real}; kwargs...)
    arr = cache.arrangement
    d = _normalize_dir(_as_float2(dir), arr.normalize_dirs)
    (n1, n2) = _normal_from_dir_2d(d)
    off = n1*Float64(x0[1]) + n2*Float64(x0[2])
    return fibered_barcode(cache, d, off; kwargs...)
end

"""
    fibered_barcode_index(cache, dir, offset; tie_break=:up)

Convenience wrapper for `fibered_barcode(...; values=:index)`.
"""
function fibered_barcode_index(cache::FiberedBarcodeCache2D, dir, offset::Real; tie_break::Symbol=:up)
    return fibered_barcode(cache, dir, offset; values=:index, tie_break=tie_break)
end

"""
    fibered_slice(cache, dir, offset; tie_break=:up)

Return `(chain, values, barcode)` consistently, including robust fallback for
degenerate queries (so that chain/values match barcode).
"""
function fibered_slice(
    cache::FiberedBarcodeCache2D,
    dir,
    offset::Real;
    tie_break::Symbol = :up,
)
    arr = cache.arrangement
    d = _normalize_dir(_as_float2(dir), arr.normalize_dirs)
    off = Float64(offset)

    cid = fibered_cell_id(arr, d, off; tie_break=tie_break)
    if cid === nothing
        return (chain=Int[], values=Float64[], barcode=Dict{Tuple{Float64,Float64},Int}())
    end

    dir_idx, off_idx = cid
    cell = _arr2d_cell_linear_index(arr, dir_idx, off_idx)
    chain_id = _arr2d_compute_cell!(arr, dir_idx, off_idx)

    if chain_id <= 0
        return (chain=Int[], values=Float64[], barcode=Dict{Tuple{Float64,Float64},Int}())
    end

    bidx = _index_barcode_for_chain!(cache, chain_id)
    chain = arr.chains[chain_id]
    vals = Float64[]
    ok = true

    if arr.backend === :boxes
        try
            vals = _arr2d_values_from_cell_boxes(arr, cell, d, off, length(chain))
        catch
            ok = false
        end
    else
        try
            vals = _arr2d_values_for_chain_poly(arr, d, off, chain)
        catch
            ok = false
        end
    end

    if !ok || isempty(vals)
        chain2, vals2 = slice_chain_exact_2d(arr.pi, d, off;
            box=arr.box, normalize_dirs=:none, strict=arr.strict, atol=arr.atol)
        if isempty(chain2)
            return (chain=Int[], values=Float64[], barcode=Dict{Tuple{Float64,Float64},Int}())
        end
        cid2 = _arr2d_chain_id!(arr, chain2)
        bidx2 = _index_barcode_for_chain!(cache, cid2)
        b = Dict{Tuple{Float64,Float64},Int}()
        for ((i,j), mult) in bidx2
            b[(vals2[i], vals2[j])] = mult
        end
        return (chain=chain2, values=vals2, barcode=b)
    end

    b = Dict{Tuple{Float64,Float64},Int}()
    for ((i,j), mult) in bidx
        b[(vals[i], vals[j])] = mult
    end

    return (chain=Base.copy(chain), values=vals, barcode=b)
end

"""
    fibered_barcode_cache_stats(cache)

Return a NamedTuple with basic statistics describing the cache state.
Useful to confirm caching/precomputation.
"""
function fibered_barcode_cache_stats(cache::FiberedBarcodeCache2D)
    arr = cache.arrangement
    return (
        n_points = length(arr.points),
        n_dir_cells = length(arr.dir_reps),
        total_cells = arr.total_cells,
        n_cells_computed = arr.n_cell_computed,
        n_chains = length(arr.chains),
        n_index_barcodes_computed = cache.n_barcode_computed,
        normalize_dirs = arr.normalize_dirs,
        include_axes = arr.include_axes,
        strict = arr.strict,
        atol = arr.atol,
    )
end


# -----------------------------------------------------------------------------
# Convenience wrappers and higher-level integrations for the 2D cache
# -----------------------------------------------------------------------------

# Internal helper: representative normal-offset for a given arrangement cell.
# Returns (off_mid, off_left, off_right), where off_mid is the midpoint
# between consecutive distinct critical offsets in that direction cell.
@inline function _arr2d_cell_representative_offset(
    arr::FiberedArrangement2D,
    dir_idx::Int,
    off_idx::Int,
)
    drep = arr.dir_reps[dir_idx]
    order = arr.orders[dir_idx]
    pos = arr.unique_pos[dir_idx]

    pL = arr.points[order[pos[off_idx]]]
    pR = arr.points[order[pos[off_idx + 1]]]

    (n1, n2) = _normal_from_dir_2d(drep)
    cL = n1 * pL[1] + n2 * pL[2]
    cR = n1 * pR[1] + n2 * pR[2]
    off_mid = 0.5 * (cL + cR)

    return off_mid, cL, cR
end

# Internal helper: angular width (in radians) of a direction cell, using the
# slope parameter s = d2/d1 and theta = atan(s).
#
# If include_axes=true, the axis directions are represented explicitly and have
# measure zero in theta, so we return 0.0 for those extra cells.
function _arr2d_direction_cell_theta_width(arr::FiberedArrangement2D, dir_idx::Int)
    n_base = length(arr.slopes) + 1
    if dir_idx > n_base
        return 0.0
    end
    if isempty(arr.slopes)
        return 0.5 * Base.MathConstants.pi
    end

    if dir_idx == 1
        sL = 0.0
        sR = arr.slopes[1]
    elseif dir_idx == n_base
        sL = arr.slopes[end]
        sR = Inf
    else
        sL = arr.slopes[dir_idx - 1]
        sR = arr.slopes[dir_idx]
    end

    thL = atan(sL)
    thR = isfinite(sR) ? atan(sR) : (0.5 * Base.MathConstants.pi)
    return thR - thL
end

# Internal helper: compute a safe global (tmin,tmax) range for representative
# slices over all nonempty arrangement cells. This is mainly used to produce a
# default tgrid for landscape kernels without materializing all barcodes.
function _arr2d_representative_tmin_tmax(arr::FiberedArrangement2D)
    tmin = Inf
    tmax = -Inf

    ndir = length(arr.dir_reps)
    for dir_idx in 1:ndir
        d = arr.dir_reps[dir_idx]
        for off_idx in 1:arr.noff[dir_idx]
            chain_id = _arr2d_compute_cell!(arr, dir_idx, off_idx)
            chain_id <= 0 && continue

            off_mid, _, _ = _arr2d_cell_representative_offset(arr, dir_idx, off_idx)
            chain = arr.chains[chain_id]
            cell = _arr2d_cell_linear_index(arr, dir_idx, off_idx)

            vals = Float64[]
            ok = true
            if arr.backend === :boxes
                try
                    vals = _arr2d_values_from_cell_boxes(arr, cell, d, off_mid, length(chain))
                catch
                    ok = false
                end
            else
                try
                    vals = _arr2d_values_for_chain_poly(arr, d, off_mid, chain)
                catch
                    ok = false
                end
            end

            if !ok || isempty(vals)
                _, vals2 = slice_chain_exact_2d(arr.pi, d, off_mid;
                    box=arr.box, normalize_dirs=:none, strict=arr.strict, atol=arr.atol)
                vals = vals2
            end
            isempty(vals) && continue

            tmin = min(tmin, vals[1])
            tmax = max(tmax, vals[end])
        end
    end

    if !isfinite(tmin) || !isfinite(tmax)
        return 0.0, 1.0
    end
    if tmin == tmax
        return tmin - 0.5, tmax + 0.5
    end
    return tmin, tmax
end

@inline function _barcode_from_index_and_values!(
    out::FloatBarcode,
    bidx::IndexBarcode,
    vals::AbstractVector{<:Real},
)::FloatBarcode
    empty!(out)
    sizehint!(out, length(bidx))
    @inbounds for ((i, j), m) in bidx
        out[(Float64(vals[i]), Float64(vals[j]))] = m
    end
    return out
end

@inline function _barcode_from_index_and_values!(
    out::FloatBarcode,
    bidx::IndexBarcode,
    pool::Vector{Float64},
    start::Int,
)::FloatBarcode
    empty!(out)
    sizehint!(out, length(bidx))
    @inbounds for ((i, j), m) in bidx
        out[(pool[start + i - 1], pool[start + j - 1])] = m
    end
    return out
end

function _barcode_from_index_and_values(bidx::IndexBarcode, vals::AbstractVector{<:Real})::FloatBarcode
    out = FloatBarcode()
    return _barcode_from_index_and_values!(out, bidx, vals)
end

"""
    FiberedSliceFamily2D

Geometry-only precomputation for arrangement-exact slice families in R^2.

A `FiberedArrangement2D` partitions (direction, offset) space into finitely many 2-cells.
On each nonempty cell the sliced module is constant. A `FiberedSliceFamily2D` chooses
one representative slice per nonempty cell and (optionally) stores the boundary
parameter values for that slice.

This object is intended for many-pair workloads: build once per dataset, reuse across
many module pairs.
"""
struct FiberedSliceFamily2D
    arrangement::FiberedArrangement2D

    # one entry per nonempty arrangement cell
    dir_idx::Vector{Int}
    off_idx::Vector{Int}
    cell_id::Vector{Int}
    chain_id::Vector{Int}

    # representative offsets and interval endpoints
    off_mid::Vector{Float64}
    off0::Vector{Float64}
    off1::Vector{Float64}

    # concatenated boundary values storage
    vals_pool::Vector{Float64}
    vals_start::Vector{Int}
    vals_len::Vector{Int}

    # per-direction precomputations
    dir_weight::Vector{Float64}
    theta_width::Vector{Float64}

    direction_weight_scheme::Symbol
    store_values::Bool
    unique_chain_ids::Vector{Int}
end

@inline nslices(fam::FiberedSliceFamily2D)::Int = length(fam.cell_id)

@inline function fibered_values(fam::FiberedSliceFamily2D, k::Int)
    s = fam.vals_start[k]
    l = fam.vals_len[k]
    if s == 0 || l == 0
        return Float64[]
    end
    return @view fam.vals_pool[s:(s + l - 1)]
end

@inline function _arr2d_cell_offset_interval(arr::FiberedArrangement2D, dir_idx::Int, off_idx::Int)
    d = arr.dir_reps[dir_idx]
    order = arr.orders[dir_idx]
    pos = arr.unique_pos[dir_idx]
    @inbounds begin
        pL = arr.points[order[pos[off_idx]]]
        pR = arr.points[order[pos[off_idx + 1]]]
        n1 = -d[2]
        n2 =  d[1]
        cL = n1 * pL[1] + n2 * pL[2]
        cR = n1 * pR[1] + n2 * pR[2]
    end
    return cL, cR, 0.5 * (cL + cR)
end

"""
    fibered_slice_family_2d(arr; direction_weight=:lesnick_l1, store_values=true)

Build and cache a `FiberedSliceFamily2D` for a given arrangement.

The result is cached in `arr.slice_family_cache[(direction_weight, store_values)]`.
"""
function fibered_slice_family_2d(
    arr::FiberedArrangement2D;
    direction_weight::Symbol = :lesnick_l1,
    store_values::Bool = true,
)
    key = (direction_weight, store_values)
    cached = get(arr.slice_family_cache, key, nothing)
    if cached !== nothing
        return cached::FiberedSliceFamily2D
    end

    if arr.n_cell_computed != arr.total_cells
        _precompute_cells!(arr)
    end

    ndirs = length(arr.dir_reps)
    dir_w = Vector{Float64}(undef, ndirs)
    theta_w = Vector{Float64}(undef, ndirs)
    for i in 1:ndirs
        d = arr.dir_reps[i]
        dir_w[i] = _direction_weight(d; scheme=direction_weight)
        theta_w[i] = _arr2d_direction_cell_theta_width(arr, i)
    end

    dir_idx = Int[]
    off_idx = Int[]
    cell_id = Int[]
    chain_id = Int[]
    off_mid = Float64[]
    off0 = Float64[]
    off1 = Float64[]

    vals_pool = Float64[]
    vals_start = Int[]
    vals_len = Int[]

    for di in 1:ndirs
        d = arr.dir_reps[di]
        for oi in 1:arr.noff[di]
            cell = _arr2d_cell_linear_index(arr, di, oi)
            cid = arr.cell_chain_id[cell]
            cid > 0 || continue

            c0, c1, cmid = _arr2d_cell_offset_interval(arr, di, oi)

            push!(dir_idx, di)
            push!(off_idx, oi)
            push!(cell_id, cell)
            push!(chain_id, cid)
            push!(off_mid, cmid)
            push!(off0, c0)
            push!(off1, c1)

            if store_values
                chain = arr.chains[cid]
                m = length(chain)

                vals = if arr.backend == :boxes
                    _arr2d_values_from_cell_boxes(arr, cell, d, cmid, m)
                else
                    try
                        _arr2d_values_for_chain_poly(arr, d, cmid, chain)
                    catch
                        _, v = _arr2d_slice_chain_and_values(arr, d, cmid)
                        v
                    end
                end

                s = length(vals_pool) + 1
                append!(vals_pool, vals)
                push!(vals_start, s)
                push!(vals_len, length(vals))
            else
                push!(vals_start, 0)
                push!(vals_len, 0)
            end
        end
    end

    uniq = sort!(unique(chain_id))

    fam = FiberedSliceFamily2D(
        arr, dir_idx, off_idx, cell_id, chain_id,
        off_mid, off0, off1,
        vals_pool, vals_start, vals_len,
        dir_w, theta_w,
        direction_weight, store_values, uniq,
    )

    arr.slice_family_cache[key] = fam
    return fam
end

"""
    slice_barcodes(cache::FiberedBarcodeCache2D; dirs, offsets, ...) -> NamedTuple

Convenience wrapper to compute a matrix of fibered barcodes using the augmented
arrangement cache.

This is analogous to `slice_barcodes(M, pi; ...)`, but it uses `fibered_barcode`
and therefore benefits from the arrangement and barcode cache.

Inputs
- `dirs` (or `directions`): iterable of 2-vectors specifying slice directions.
  Directions are interpreted using the arrangement normalization stored in
  `cache.arrangement.normalize_dirs`.
- `offsets`: iterable where each element is either
    * a `Real` normal-offset `off = dot(n, x)` with `n = (-d2, d1)`, or
    * a length-2 basepoint (vector or 2-tuple) lying on the slice line.

Keywords
- `values`: `:t` (default) or `:index` (index barcodes).
- `tie_break`: forwarded to `fibered_barcode` for boundary cases.
- `verify`: forwarded to `fibered_barcode` (slow; checks against exact slicing).
- `direction_weight`: one of `:none`, `:lesnick_l1`, `:lesnick_linf`.
- `offset_weights`: `nothing` (default), a vector of length `length(offsets)`,
  a function `w(off)` (called on each offset element as provided), or a scalar.
- `normalize_weights`: normalize the weight matrix to sum to 1.

Returns a NamedTuple:
- `barcodes`: `nd x no` matrix of barcodes (each a Dict).
- `weights`:  `nd x no` matrix of weights.
- `directions`: normalized directions used (Float64 vectors).
- `offsets`: collected offsets as provided.
"""
function slice_barcodes(
    cache::FiberedBarcodeCache2D;
    dirs,
    offsets,
    values::Symbol = :t,
    direction_weight::Union{Symbol,Function,Real} = :none,
    offset_weights = nothing,
    normalize_weights::Bool = true,
    tie_break::Symbol = :up,
    threads::Bool = (Threads.nthreads() > 1),
)
    nd = length(dirs)
    no = length(offsets)

    wdir = Vector{Float64}(undef, nd)
    for i in 1:nd
        wdir[i] = direction_weight(dirs[i], direction_weight)
    end
    woff = _offset_sample_weights(offsets, offset_weights)

    weights = wdir * woff'
    if normalize_weights
        s = sum(weights)
        if s > 0
            weights ./= s
        end
    end

    barcodes = Matrix{Any}(undef, nd, no)  # remains Any because values=:t returns Dict{Tuple{Float64,Float64},Int}
    if threads
        Threads.@threads for idx in 1:(nd * no)
            i = div((idx - 1), no) + 1
            j = (idx - 1) % no + 1
            barcodes[i, j] = fibered_barcode(cache, dirs[i], offsets[j]; values=values, tie_break=tie_break)
        end
    else
        for i in 1:nd, j in 1:no
            barcodes[i, j] = fibered_barcode(cache, dirs[i], offsets[j]; values=values, tie_break=tie_break)
        end
    end

    return (barcodes=barcodes, weights=weights, offs=offsets)
end




"""
    matching_distance_exact_2d(cacheM::FiberedBarcodeCache2D,
                              cacheN::FiberedBarcodeCache2D; weight=:lesnick_l1) -> Float64

Exact 2D matching distance computed by enumerating *all* arrangement cells
(one representative slice per cell), reusing the same augmented arrangement
across many module pairs.

This overload is intended for the workflow:
1. Build one `FiberedArrangement2D` once (from `pi` and `box`).
2. Build one `FiberedBarcodeCache2D` per module on that same arrangement.
3. Call `matching_distance_exact_2d(cacheM, cacheN)` repeatedly.

Requirements
- `cacheM.arrangement === cacheN.arrangement` (same arrangement object)
- same base field `Q` in both modules.

The `weight` keyword matches `matching_distance_exact_2d(M,N,pi; weight=...)`.
"""
function matching_distance_exact_2d(
    cacheM::FiberedBarcodeCache2D,
    cacheN::FiberedBarcodeCache2D;
    weight::Symbol = :lesnick_l1,
    family::Union{Nothing,FiberedSliceFamily2D} = nothing,
    store_values::Bool = true,
    threads::Bool = (Threads.nthreads() > 1),
)
    arr = cacheM.arrangement
    arr === cacheN.arrangement || error("matching_distance_exact_2d: caches must share the same arrangement")

    fam = family === nothing ? fibered_slice_family_2d(arr; direction_weight=weight, store_values=store_values) : family
    fam.arrangement === arr || error("matching_distance_exact_2d: family must belong to cache arrangement")

    # Thread-safe: ensure needed barcodes exist before threading.
    # (No locks in inner loops.)
    if threads
        _precompute_index_barcodes!(cacheM)
        _precompute_index_barcodes!(cacheN)
    end

    # Per-thread scratch dictionaries to avoid allocations.
    nt = threads ? Threads.nthreads() : 1
    tmpM = [FloatBarcode() for _ in 1:nt]
    tmpN = [FloatBarcode() for _ in 1:nt]
    best_by_thread = fill(0.0, nt)

    ns = nslices(fam)

    if threads
        Threads.@threads for k in 1:ns
            tid = Threads.threadid()
            di = fam.dir_idx[k]
            cid = fam.chain_id[k]
            bidxM = cacheM.index_barcodes[cid]::IndexBarcode
            bidxN = cacheN.index_barcodes[cid]::IndexBarcode

            wdir = fam.dir_weight[di]

            # use precomputed values pool (fast)
            s = fam.vals_start[k]
            bM = _barcode_from_index_and_values!(tmpM[tid], bidxM, fam.vals_pool, s)
            bN = _barcode_from_index_and_values!(tmpN[tid], bidxN, fam.vals_pool, s)

            d = wdir * bottleneck_distance(bM, bN)
            if d > best_by_thread[tid]
                best_by_thread[tid] = d
            end
        end
        return maximum(best_by_thread)
    else
        best = 0.0
        bMtmp = tmpM[1]
        bNtmp = tmpN[1]
        for k in 1:ns
            di = fam.dir_idx[k]
            cid = fam.chain_id[k]
            bidxM = _index_barcode_for_chain!(cacheM, cid)
            bidxN = _index_barcode_for_chain!(cacheN, cid)

            wdir = fam.dir_weight[di]
            s = fam.vals_start[k]
            bM = _barcode_from_index_and_values!(bMtmp, bidxM, fam.vals_pool, s)
            bN = _barcode_from_index_and_values!(bNtmp, bidxN, fam.vals_pool, s)

            d = wdir * bottleneck_distance(bM, bN)
            if d > best
                best = d
            end
        end
        return best
    end
end



"""
    slice_kernel(cacheM::FiberedBarcodeCache2D, cacheN::FiberedBarcodeCache2D; ...) -> Float64

Compute a sliced kernel by integrating over *all* arrangement cells (one
representative slice per cell), using the shared augmented arrangement.

This is the arrangement-exact analogue of sampling-based `slice_kernel(M,N,pi; ...)`.

Keywords
- `kind`, `sigma`, `gamma`, `tgrid`, `kmax`: forwarded to `_barcode_kernel`.
- `direction_weight`: direction weighting scheme (`:none`, `:lesnick_l1`, `:lesnick_linf`).
- `cell_weight`: how to weight arrangement cells. Supported symbols:
    * `:uniform`      (each nonempty cell weight 1)
    * `:offset_length` (weight by the length of the offset interval in normal-offset)
    * `:theta`        (weight by angular width of the direction cell)
    * `:theta_offset` (product of `:theta` and `:offset_length`)
  or a function `w(dir, off, dir_cell, off_cell, arr)::Real`.
- `normalize_weights`: kept for API symmetry; the result is always a
  weighted average as in `slice_kernel` for explicit slice lists.
- `tgrid_nsteps`: if `tgrid` is not provided and `kind` is landscape-based,
  we build a default grid with this many points.
"""
function slice_kernel(
    cacheM::FiberedBarcodeCache2D,
    cacheN::FiberedBarcodeCache2D;
    kind::Symbol = :bottleneck_gaussian,
    sigma::Real = 1.0,
    direction_weight::Symbol = :lesnick_l1,
    cell_weight::Symbol = :uniform,
    normalize_weights::Bool = true,
    family::Union{Nothing,FiberedSliceFamily2D} = nothing,
    store_values::Bool = true,
    threads::Bool = (Threads.nthreads() > 1),
)
    arr = cacheM.arrangement
    arr === cacheN.arrangement || error("slice_kernel: caches must share the same arrangement")

    fam = family === nothing ? fibered_slice_family_2d(arr; direction_weight=direction_weight, store_values=store_values) : family

    # Kernel on barcodes (reuse existing kernel definitions)
    kernel_fn = _barcode_kernel(:bottleneck, kind, sigma)

    if threads
        _precompute_index_barcodes!(cacheM)
        _precompute_index_barcodes!(cacheN)
    end

    nt = threads ? Threads.nthreads() : 1
    tmpM = [FloatBarcode() for _ in 1:nt]
    tmpN = [FloatBarcode() for _ in 1:nt]
    acc_by_thread = fill(0.0, nt)
    sumw_by_thread = fill(0.0, nt)

    ns = nslices(fam)

    # helper: cell weight
    @inline function _cell_w(k::Int)::Float64
        if cell_weight === :uniform || cell_weight === :none
            return 1.0
        elseif cell_weight === :offset_length
            return fam.off1[k] - fam.off0[k]
        elseif cell_weight === :theta
            return fam.theta_width[fam.dir_idx[k]]
        elseif cell_weight === :theta_offset
            return fam.theta_width[fam.dir_idx[k]] * (fam.off1[k] - fam.off0[k])
        else
            throw(ArgumentError("slice_kernel: unknown cell_weight=$(cell_weight)"))
        end
    end

    if threads
        Threads.@threads for k in 1:ns
            tid = Threads.threadid()
            di = fam.dir_idx[k]
            cid = fam.chain_id[k]
            w = fam.dir_weight[di] * _cell_w(k)

            bidxM = cacheM.index_barcodes[cid]::IndexBarcode
            bidxN = cacheN.index_barcodes[cid]::IndexBarcode
            s = fam.vals_start[k]
            bM = _barcode_from_index_and_values!(tmpM[tid], bidxM, fam.vals_pool, s)
            bN = _barcode_from_index_and_values!(tmpN[tid], bidxN, fam.vals_pool, s)

            acc_by_thread[tid] += w * kernel_fn(bM, bN)
            sumw_by_thread[tid] += w
        end
        acc = sum(acc_by_thread)
        sumw = sum(sumw_by_thread)
        return normalize_weights && sumw > 0 ? acc / sumw : acc
    else
        acc = 0.0
        sumw = 0.0
        bMtmp = tmpM[1]
        bNtmp = tmpN[1]
        for k in 1:ns
            di = fam.dir_idx[k]
            cid = fam.chain_id[k]
            w = fam.dir_weight[di] * _cell_w(k)

            bidxM = _index_barcode_for_chain!(cacheM, cid)
            bidxN = _index_barcode_for_chain!(cacheN, cid)
            s = fam.vals_start[k]
            bM = _barcode_from_index_and_values!(bMtmp, bidxM, fam.vals_pool, s)
            bN = _barcode_from_index_and_values!(bNtmp, bidxN, fam.vals_pool, s)

            acc += w * kernel_fn(bM, bN)
            sumw += w
        end
        return normalize_weights && sumw > 0 ? acc / sumw : acc
    end
end




# -----------------------------------------------------------------------------
# Projected distances / projected invariants (pushforward along projections)
# -----------------------------------------------------------------------------

"""
    ProjectedArrangement1D

A single monotone projection from a finite region poset `Q` to a chain `C`.

Fields:
- `Q`: source region poset
- `C`: target chain poset
- `f`: EncodingMap Q -> C
- `chain`: the chain indices 1:C.n (stored as a UnitRange for zero allocation)
- `values`: endpoint coordinates for the chain (length C.n+1), suitable for `slice_barcode`
- `dir`: the direction vector used to define this projection (for bookkeeping)
"""
struct ProjectedArrangement1D
    Q::FinitePoset
    C::FinitePoset
    f::EncodingMap
    chain::UnitRange{Int}
    values::Vector{Float64}
    dir::Vector{Float64}
end

"""
    ProjectedArrangement

A family of 1D projections sharing the same source poset `Q`.
"""
struct ProjectedArrangement
    Q::FinitePoset
    projections::Vector{ProjectedArrangement1D}
end

# Internal: build the chain poset {1,...,k} with i <= j ordering.
function _chain_poset(k::Int)
    leq = BitMatrix(undef, k, k)
    @inbounds for i in 1:k, j in 1:k
        leq[i,j] = (i <= j)
    end
    return FinitePoset(k, leq; check=false)
end

# Internal: minimal isotone majorant (upper closure) on a finite poset.
# t(p) = max_{q <= p} s(q). This guarantees monotonicity.
function _monotone_upper_closure(Q::FinitePoset, s::AbstractVector{<:Real})
    n = Q.n
    leq = Q.leq
    t = Vector{Float64}(undef, n)
    @inbounds for p in 1:n
        m = -Inf
        for q in 1:n
            if leq[q,p]
                v = float(s[q])
                if v > m
                    m = v
                end
            end
        end
        t[p] = m
    end
    return t
end

"""
    projected_arrangement(Q::FinitePoset, values; enforce_monotone=:upper)

Build a `ProjectedArrangement` containing a single projection map determined by a
real-valued function on the elements of `Q`.

If `enforce_monotone=:upper`, we replace `values` by its minimal isotone majorant
so that the resulting map `Q -> chain` is guaranteed monotone.
"""
function projected_arrangement(Q::FinitePoset, values::AbstractVector{<:Real};
                               enforce_monotone::Symbol = :upper,
                               dir = nothing)
    length(values) == Q.n || error("values must have length Q.n")

    t = enforce_monotone == :upper ? _monotone_upper_closure(Q, values) :
        Float64.(values)

    uvals = sort(unique(t))
    k = length(uvals)
    C = _chain_poset(k)

    # Map each region value to its rank in the sorted unique list.
    pi_of_q = Vector{Int}(undef, Q.n)
    @inbounds for i in 1:Q.n
        pi_of_q[i] = searchsortedfirst(uvals, t[i])
    end

    f = EncodingMap(Q, C, pi_of_q)
    vals_ext = _extend_values(uvals)

    d = dir === nothing ? Float64[] : Float64.(collect(dir))
    proj = ProjectedArrangement1D(Q, C, f, 1:k, vals_ext, d)
    return ProjectedArrangement(Q, [proj])
end

# Dot product helper (works for tuples or vectors).
@inline function _dot(dir::AbstractVector{<:Real}, x)
    s = 0.0
    @inbounds for i in eachindex(dir)
        s += float(dir[i]) * float(x[i])
    end
    return s
end

"""
    projected_arrangement(pi::PLikeEncodingMap; dirs=nothing, n_dirs=32, normalize=:L1, ...)

Build a family of monotone projections from the encoding geometry `pi`.

Each direction `dir` defines a linear functional x -> dot(dir, x) evaluated on a
representative point for each region. The resulting real values are then
monotone-closed on the region poset, collapsed to a chain, and stored for reuse.
"""
function projected_arrangement(pi::PLikeEncodingMap;
                               dirs::Union{Nothing,AbstractVector}=nothing,
                               n_dirs::Int = 32,
                               max_den::Int = 8,
                               include_axes::Bool = false,
                               normalize::Symbol = :L1,
                               enforce_monotone::Symbol = :upper,
                                Q::Union{Nothing,FinitePoset}=nothing)

    # Determine the region poset Q (either provided or reconstructed from pi).
    Qposet = (Q === nothing) ? region_poset(pi) : Q
    Qposet.n == _nregions_encoding(pi) || error(
        "projected_arrangement: incompatible Q; Q.n=$(Qposet.n) but encoding has $(_nregions_encoding(pi)) regions"
    )

    dirs === nothing && (dirs = default_directions(dim(pi); n_dirs=n_dirs,
                                                   max_den=max_den,
                                                   include_axes=include_axes,
                                                   normalize=normalize))

    arrs = Vector{ProjectedArrangement1D}(undef, length(dirs))
    for (j,dir) in enumerate(dirs)
        vals = region_values(pi, x -> _dot(dir, x))
        tmp = projected_arrangement(Qposet, vals; enforce_monotone=enforce_monotone, dir=dir)
        arrs[j] = tmp.projections[1]
    end
    return ProjectedArrangement(Qposet, arrs)
end

"""
    ProjectedBarcodeCache

Cache the 1D barcodes obtained by pushing a module forward along each projection
in a fixed `ProjectedArrangement`.

This is the recommended workflow:
- build `arr = projected_arrangement(pi; ...)` once,
- build caches for many modules,
- compare rapidly via `projected_distance` / `projected_kernel`.
"""
mutable struct ProjectedBarcodeCache
    arrangement::ProjectedArrangement
    M::PModule{QQ}
    side::Symbol
    barcodes::Vector{Any}
    n_computed::Int
end

function projected_barcode_cache(M::PModule{QQ}, arr::ProjectedArrangement;
                                 side::Symbol = :left,
                                 precompute::Bool = false)
    side in (:left, :right) || error("side must be :left or :right")
    barcodes = fill(nothing, length(arr.projections))
    cache = ProjectedBarcodeCache(arr, M, side, barcodes, 0)
    precompute && projected_barcodes(cache)
    return cache
end

# Lazy compute ith barcode.
function _projected_barcode(cache::ProjectedBarcodeCache, i::Int)
    bc = cache.barcodes[i]
    if bc === nothing
        proj = cache.arrangement.projections[i]
        Mp = cache.side == :left ? pushforward_left(proj.f, cache.M; check=false) :
                                   pushforward_right(proj.f, cache.M; check=false)
        bc = slice_barcode(Mp, proj.chain; values=proj.values, check_chain=false)
        cache.barcodes[i] = bc
        cache.n_computed += 1
    end
    return bc
end

"""
    projected_barcodes(cache; inds=nothing)

Return all (or selected) cached projected barcodes.
"""
function projected_barcodes(cache::ProjectedBarcodeCache; inds=nothing)
    n = length(cache.arrangement.projections)
    inds === nothing && (inds = 1:n)
    out = Vector{Any}(undef, length(inds))
    for (k,i) in enumerate(inds)
        out[k] = _projected_barcode(cache, i)
    end
    return out
end

"""
    projected_distances(cacheM, cacheN; dist=:bottleneck, p=1, q=1)

Return the vector of per-projection barcode distances.
"""
function projected_distances(cacheM::ProjectedBarcodeCache,
                             cacheN::ProjectedBarcodeCache;
                             dist::Symbol = :bottleneck,
                             p::Real = 1,
                             q::Real = 1)
    n = length(cacheM.arrangement.projections)
    n == length(cacheN.arrangement.projections) || error("different projection families")

    out = Vector{Float64}(undef, n)
    for i in 1:n
        b1 = _projected_barcode(cacheM, i)
        b2 = _projected_barcode(cacheN, i)
        out[i] = dist == :bottleneck ? bottleneck_distance(b1, b2) :
                 dist == :wasserstein ? wasserstein_distance(b1, b2; p=p, q=q) :
                 error("unknown dist=$dist")
    end
    return out
end

"""
    projected_distance(cacheM, cacheN; dist=:bottleneck, agg=:mean, dir_weights=nothing, ...)

Aggregate per-projection distances (mean or maximum by default).
Weights are normalized to sum 1 when using mean/sum aggregation.
"""
function projected_distance(cacheM::ProjectedBarcodeCache,
                            cacheN::ProjectedBarcodeCache;
                            dist::Symbol = :bottleneck,
                            p::Real = 1,
                            q::Real = 1,
                            agg::Symbol = :mean,
                            dir_weights = nothing)
    dvec = projected_distances(cacheM, cacheN; dist=dist, p=p, q=q)
    n = length(dvec)

    w = dir_weights === nothing ? fill(1.0/n, n) : Float64.(collect(dir_weights))
    length(w) == n || error("dir_weights must have length $n")
    s = sum(w)
    s > 0 || error("dir_weights must sum positive")
    w ./= s

    if agg == :mean
        return sum(w .* dvec)
    elseif agg == :sum
        return sum(w .* dvec)
    elseif agg == :maximum
        return maximum(dvec)
    else
        error("unknown agg=$agg")
    end
end

"""
    projected_kernel(cacheM, cacheN; kind=:wasserstein_gaussian, sigma=1.0, agg=:mean, dir_weights=nothing)

Aggregate per-projection barcode kernels using `_barcode_kernel`.
"""
function projected_kernel(cacheM::ProjectedBarcodeCache,
                          cacheN::ProjectedBarcodeCache;
                          kind::Symbol = :wasserstein_gaussian,
                          sigma::Real = 1.0,
                          p::Real = 1,
                          q::Real = 1,
                          agg::Symbol = :mean,
                          dir_weights = nothing)
    n = length(cacheM.arrangement.projections)
    n == length(cacheN.arrangement.projections) || error("different projection families")

    w = dir_weights === nothing ? fill(1.0/n, n) : Float64.(collect(dir_weights))
    length(w) == n || error("dir_weights must have length $n")
    s = sum(w)
    s > 0 || error("dir_weights must sum positive")
    w ./= s

    vals = Vector{Float64}(undef, n)
    for i in 1:n
        b1 = _projected_barcode(cacheM, i)
        b2 = _projected_barcode(cacheN, i)
        vals[i] = _barcode_kernel(b1, b2; kind=kind, sigma=sigma, p=p, q=q)
    end

    if agg == :mean
        return sum(w .* vals)
    elseif agg == :sum
        return sum(w .* vals)
    elseif agg == :maximum
        return maximum(vals)
    else
        error("unknown agg=$agg")
    end
end



# -----------------------------------------------------------------------------
# ----- Multiparameter persistence images (Carriere et al. construction) -------
# -----------------------------------------------------------------------------
#
# This section implements the "Multiparameter Persistence Image" (MPPI)
# construction described by Carriere et al.
#
# The goal is NOT "just take 1D persistence images on slices".
# Instead, the construction:
#   1) chooses specific families of slices (lines) in the parameter plane,
#   2) computes fibered barcodes along these slices,
#   3) connects bars across neighboring slices by optimal matchings ("vineyards"),
#   4) treats each connected track as a summand I,
#   5) weights each summand by a geometric hull weight w(I),
#   6) produces an image by evaluating the Carriere kernel
#        k(x, I) = omega(l*) * exp(-d(x,I)^2 / sigma^2)
#      where l* is the slice containing the closest segment in I.
#
# This produces a stable, low-dimensional representation that explicitly
# accounts for multiparameter coherence across slices.

"""
    MPPLineSpec

A single slice line for the Carriere MPPI construction.

Fields
- `dir`: direction vector (Float64, length 2), normalized according to the
         arrangement's direction normalization convention (usually L1).
- `off`: normal offset (Float64), so the line is { x : n . x = off } where
         n is a unit normal associated to dir.
- `x0`: a basepoint on the line (Float64, length 2), compatible with (dir, off).
- `omega`: the Carriere direction weight omega(dir) = min(cos(theta), sin(theta))
           where theta is the direction angle in [0, pi/2]. Equivalently,
           if `u = dir / norm(dir,2)` then omega = min(u[1], u[2]).
"""
struct MPPLineSpec
    dir::Vector{Float64}
    off::Float64
    x0::Vector{Float64}
    omega::Float64
end

"""
    MPPDecomposition

A decomposition of a module into "summands" I_k as used by Carriere MPPI.

Each summand is represented as a list of segments in R^2, where each segment
lies on one of the chosen slice lines and corresponds to one barcode interval
(birth -> death mapped into the plane).

Fields
- `lines`: vector of `MPPLineSpec` used.
- `summands`: vector of summands; each summand is a vector of segments.
  A segment is stored as `(p, q, omega)` where p,q are endpoints (length-2 Float64)
  and omega is the direction weight of the slice line containing that segment.
- `weights`: w(I_k) geometric weights for each summand (Float64).
- `box`: the ambient bounding box (a,b) in R^2 used for normalization.
"""
struct MPPDecomposition
    lines::Vector{MPPLineSpec}
    summands::Vector{Vector{Tuple{Vector{Float64},Vector{Float64},Float64}}}
    weights::Vector{Float64}
    box::Tuple{Vector{Float64},Vector{Float64}}
end

"""
    MPPImage

A Carriere multiparameter persistence image.

Fields
- `xgrid`, `ygrid`: coordinate grids for evaluation.
- `img`: matrix of values, size (length(ygrid), length(xgrid)).
- `sigma`: Gaussian scale used in kernel exp(-d^2/sigma^2).
- `decomp`: the underlying `MPPDecomposition` (stored for reproducibility).
"""
struct MPPImage
    xgrid::Vector{Float64}
    ygrid::Vector{Float64}
    img::Matrix{Float64}
    sigma::Float64
    decomp::MPPDecomposition
end

# ------------------------- internal geometry helpers --------------------------

# Squared Euclidean distance from point (x,y) to segment with endpoints (p,q).
#
# Important for performance:
# The MPPI image evaluation loops over many grid points, so we provide a scalar
# version to avoid allocations in the innermost loops.
@inline function _dist2_point_segment_xy(x::Float64, y::Float64,
                                        px::Float64, py::Float64,
                                        qx::Float64, qy::Float64)::Float64
    vx = qx - px
    vy = qy - py
    wx = x - px
    wy = y - py
    vv = vx*vx + vy*vy
    if vv == 0.0
        dx = x - px
        dy = y - py
        return dx*dx + dy*dy
    end
    t = (wx*vx + wy*vy) / vv
    if t <= 0.0
        dx = x - px
        dy = y - py
        return dx*dx + dy*dy
    elseif t >= 1.0
        dx = x - qx
        dy = y - qy
        return dx*dx + dy*dy
    else
        projx = px + t*vx
        projy = py + t*vy
        dx = x - projx
        dy = y - projy
        return dx*dx + dy*dy
    end
end

# Backwards compatible vector form (convenient in some internal helpers).
@inline function _dist2_point_segment(x::Vector{Float64},
                                      p::Vector{Float64},
                                      q::Vector{Float64})::Float64
    return _dist2_point_segment_xy(x[1], x[2], p[1], p[2], q[1], q[2])
end

# Squared distance from point (x,y) to an axis-aligned bounding box.
# Returns 0.0 if (x,y) lies inside the box.
@inline function _bbox_dist2_xy(x::Float64, y::Float64,
                               xmin::Float64, xmax::Float64,
                               ymin::Float64, ymax::Float64)::Float64
    dx = 0.0
    if x < xmin
        dx = xmin - x
    elseif x > xmax
        dx = x - xmax
    end
    dy = 0.0
    if y < ymin
        dy = ymin - y
    elseif y > ymax
        dy = y - ymax
    end
    return dx*dx + dy*dy
end


# Convex hull (monotone chain) and area for weight w(I).
@inline _cross2(o::NTuple{2,Float64}, a::NTuple{2,Float64}, b::NTuple{2,Float64}) =
    (a[1]-o[1])*(b[2]-o[2]) - (a[2]-o[2])*(b[1]-o[1])

function _convex_hull_2d(pts::Vector{NTuple{2,Float64}}; atol::Float64=0.0)
    n = length(pts)
    n <= 1 && return pts
    sort!(pts, by=p->(p[1],p[2]))
    _unique_points_2d!(pts; atol=atol)
    length(pts) <= 1 && return pts

    lower = NTuple{2,Float64}[]
    for p in pts
        while length(lower) >= 2 &&
              _cross2(lower[end-1], lower[end], p) <= atol
            pop!(lower)
        end
        push!(lower, p)
    end

    upper = NTuple{2,Float64}[]
    for p in reverse(pts)
        while length(upper) >= 2 &&
              _cross2(upper[end-1], upper[end], p) <= atol
            pop!(upper)
        end
        push!(upper, p)
    end

    return vcat(lower[1:end-1], upper[1:end-1])
end

function _polygon_area_2d(poly::Vector{NTuple{2,Float64}})::Float64
    n = length(poly)
    n < 3 && return 0.0
    s = 0.0
    @inbounds for i in 1:n
        x1,y1 = poly[i]
        x2,y2 = poly[i == n ? 1 : i+1]
        s += x1*y2 - y1*x2
    end
    return 0.5 * abs(s)
end

# Carriere omega(dir). Uses unit L2 direction.
@inline function _omega_from_dir(d::Vector{Float64})::Float64
    nrm = sqrt(d[1]*d[1] + d[2]*d[2])
    nrm == 0.0 && return 0.0
    u1 = d[1]/nrm
    u2 = d[2]/nrm
    return min(u1, u2)
end

# Convert a barcode dict to an explicit multiset of (b,d) points.
function _barcode_points(bc::Dict{Tuple{Float64,Float64},Int})
    pts = Tuple{Float64,Float64}[]
    for (k,m) in bc
        for _ in 1:m
            push!(pts, k)
        end
    end
    return pts
end

# Bottleneck matching between two multisets of points.
# Returns a vector `match` of length length(A) with entries in 1:length(B) or 0 (diagonal).
#
# We reuse the internal 1D bottleneck machinery already present in this file:
# `bottleneck_distance(barA, barB)` exists, but it does not expose matching.
# Therefore, we implement a small exact bipartite matching specialized to the
# Carriere construction where sizes are typically modest.
#
# This is O(n^3) Hungarian-style on the max metric, but n is usually small
# (number of bars per slice).
function _bottleneck_matching_points(A::Vector{Tuple{Float64,Float64}},
                                    B::Vector{Tuple{Float64,Float64}})
    n = length(A)
    m = length(B)
    # Cost to diagonal for a point (b,d): half persistence in L_infty.
    diagcost(p) = 0.5*abs(p[2]-p[1])

    # Build square cost matrix by padding with diagonal nodes.
    N = max(n,m)
    C = fill(0.0, N, N)
    for i in 1:N
        for j in 1:N
            if i <= n && j <= m
                # L_infty distance between points.
                C[i,j] = max(abs(A[i][1]-B[j][1]), abs(A[i][2]-B[j][2]))
            elseif i <= n
                C[i,j] = diagcost(A[i])
            elseif j <= m
                C[i,j] = diagcost(B[j])
            else
                C[i,j] = 0.0
            end
        end
    end

    # Find minimal bottleneck threshold t such that a perfect matching exists.
    vals = unique(vec(C))
    sort!(vals)
    function feasible(t)
        # Build adjacency: i connects j if C[i,j] <= t.
        adj = [Int[] for _ in 1:N]
        for i in 1:N
            for j in 1:N
                C[i,j] <= t && push!(adj[i], j)
            end
        end
        # Standard bipartite matching (DFS augment).
        matchR = fill(0, N)
        function dfs(u, seen)
            for v in adj[u]
                seen[v] && continue
                seen[v] = true
                if matchR[v] == 0 || dfs(matchR[v], seen)
                    matchR[v] = u
                    return true
                end
            end
            return false
        end
        for u in 1:N
            seen = falses(N)
            dfs(u, seen) || return false
        end
        return true
    end

    lo = 1
    hi = length(vals)
    while lo < hi
        mid = (lo+hi) >>> 1
        feasible(vals[mid]) ? (hi = mid) : (lo = mid+1)
    end
    thr = vals[lo]

    # Extract one matching at threshold thr.
    adj = [Int[] for _ in 1:N]
    for i in 1:N
        for j in 1:N
            C[i,j] <= thr && push!(adj[i], j)
        end
    end
    matchR = fill(0, N)
    function dfs(u, seen)
        for v in adj[u]
            seen[v] && continue
            seen[v] = true
            if matchR[v] == 0 || dfs(matchR[v], seen)
                matchR[v] = u
                return true
            end
        end
        return false
    end
    for u in 1:N
        seen = falses(N)
        dfs(u, seen) || error("internal error: expected feasible matching")
    end

    # Convert matchingR (right->left) into left->right for original sizes.
    matchL = fill(0, n)
    for j in 1:N
        i = matchR[j]
        if i >= 1 && i <= n && j >= 1 && j <= m
            matchL[i] = j
        end
    end
    return matchL
end

# -------------------- line families L_m^N, L_M^N, L_delta ---------------------

function _box_corners_2d(box::Tuple{AbstractVector{<:Real},AbstractVector{<:Real}})
    a_in,b_in = box
    a = Float64[float(a_in[1]), float(a_in[2])]
    b = Float64[float(b_in[1]), float(b_in[2])]
    lo = Float64[min(a[1],b[1]), min(a[2],b[2])]
    hi = Float64[max(a[1],b[1]), max(a[2],b[2])]
    return lo, hi
end

function _make_line_spec(arr::FiberedArrangement2D,
                         dir_in::AbstractVector{<:Real},
                         off_in::Real;
                         tie_break::Symbol=:center)
    d = _normalize_dir(_as_float2(dir_in), arr.normalize_dirs)
    off = Float64(off_in)
    # Nudge offsets away from boundaries if needed.
    cid = fibered_cell_id(arr, d, off; tie_break=tie_break)
    if cid === nothing
        eps = max(10.0*arr.atol, 1e-12)
        cid2 = fibered_cell_id(arr, d, off+eps; tie_break=tie_break)
        if cid2 !== nothing
            off += eps
        else
            cid2 = fibered_cell_id(arr, d, off-eps; tie_break=tie_break)
            if cid2 !== nothing
                off -= eps
            end
        end
    end
    x0 = _line_basepoint_from_normal_offset_2d(d, off)
    omega = _omega_from_dir(d)
    return MPPLineSpec(d, off, x0, omega)
end

function _line_families_carriere(arr::FiberedArrangement2D;
                                 N::Int=16,
                                 delta::Union{Real,Symbol}=:auto,
                                 tie_break::Symbol=:center)
    N > 1 || error("mpp_image: N must be >= 2")
    lo, hi = _box_corners_2d(arr.box)
    mpt = lo
    Mpt = hi

    # Directions in (0,pi/2). We avoid axes by default; axes contributions are
    # automatically damped since omega=0 there anyway.
    dirs = Vector{Vector{Float64}}()
    for i in 1:(N-1)
        th = (i*pi) / (2.0*N)
        push!(dirs, Float64[cos(th), sin(th)])
    end

    lines = MPPLineSpec[]

    # L_m^N: lines through bottom-left corner m.
    for d in dirs
        (n1,n2) = _normal_from_dir_2d(d)
        off = n1*mpt[1] + n2*mpt[2]
        push!(lines, _make_line_spec(arr, d, off; tie_break=tie_break))
    end

    # L_M^N: lines through top-right corner M.
    for d in dirs
        (n1,n2) = _normal_from_dir_2d(d)
        off = n1*Mpt[1] + n2*Mpt[2]
        push!(lines, _make_line_spec(arr, d, off; tie_break=tie_break))
    end

    # L_delta: slope-1 lines sweeping across the box.
    d45 = Float64[1.0, 1.0]
    (n1,n2) = _normal_from_dir_2d(d45)
    # compute min/max normal offsets over corners
    corners = [
        (lo[1], lo[2]),
        (lo[1], hi[2]),
        (hi[1], lo[2]),
        (hi[1], hi[2]),
    ]
    offs = Float64[n1*c[1] + n2*c[2] for c in corners]
    omin = minimum(offs)
    omax = maximum(offs)

    dstep = 0.0
    if delta === :auto
        # automatic: about N steps across the span
        dstep = (omax-omin)/max(1, N)
    else
        dstep = Float64(delta)
    end
    dstep > 0.0 || error("mpp_image: delta must be positive")

    off = omin
    while off <= omax + 1e-12
        push!(lines, _make_line_spec(arr, d45, off; tie_break=tie_break))
        off += dstep
    end

    return lines
end

# ---------------------- vineyard-style decomposition --------------------------

"""
    mpp_decomposition(cache::FiberedBarcodeCache2D; N=16, delta=:auto, q=1.0, tie_break=:center)

Compute the Carriere MPPI summand decomposition from a fibered barcode cache.

Returns an `MPPDecomposition`.
"""
function mpp_decomposition(cache::FiberedBarcodeCache2D;
                           N::Int=16,
                           delta::Union{Real,Symbol}=:auto,
                           q::Real=1.0,
                           tie_break::Symbol=:center)

    arr = cache.arrangement
    lines = _line_families_carriere(arr; N=N, delta=delta, tie_break=tie_break)

    # Compute barcodes per line.
    bcs = Vector{Vector{Tuple{Float64,Float64}}}(undef, length(lines))
    for i in 1:length(lines)
        spec = lines[i]
        bc = fibered_barcode(cache, spec.dir, spec.off; values=:t, tie_break=:center)
        bcs[i] = _barcode_points(bc)
    end

    # Union-find for tracking connected components across matchings.
    counts = [length(bcs[i]) for i in 1:length(lines)]
    offsets = cumsum(vcat(0, counts[1:end-1]))
    total = sum(counts)

    parent = collect(1:total)
    rankv = fill(0, total)
    findp(x) = (parent[x] == x ? x : (parent[x] = findp(parent[x])))
    function unite(a,b)
        ra = findp(a)
        rb = findp(b)
        ra == rb && return
        if rankv[ra] < rankv[rb]
            parent[ra] = rb
        elseif rankv[ra] > rankv[rb]
            parent[rb] = ra
        else
            parent[rb] = ra
            rankv[ra] += 1
        end
    end

    # Match successive barcodes and union matched bars.
    for i in 1:(length(lines)-1)
        A = bcs[i]
        B = bcs[i+1]
        isempty(A) && continue
        isempty(B) && continue
        match = _bottleneck_matching_points(A,B)
        for a in 1:length(match)
            b = match[a]
            b == 0 && continue
            ida = offsets[i] + a
            idb = offsets[i+1] + b
            unite(ida, idb)
        end
    end

    # Gather nodes by component.
    comps = Dict{Int, Vector{Tuple{Int,Int}}}()
    for i in 1:length(lines)
        for j in 1:counts[i]
            id = offsets[i] + j
            r = findp(id)
            if !haskey(comps, r)
                comps[r] = Tuple{Int,Int}[]
            end
            push!(comps[r], (i,j))
        end
    end

    # Build summands as lists of geometric segments (p,q,omega).
    summands = Vector{Vector{Tuple{Vector{Float64},Vector{Float64},Float64}}}()
    weights = Float64[]
    lo, hi = _box_corners_2d(arr.box)
    areaR = (hi[1]-lo[1])*(hi[2]-lo[2])
    areaR > 0.0 || error("mpp_decomposition: box has zero area")

    for (_, nodes) in comps
        segs = Tuple{Vector{Float64},Vector{Float64},Float64}[]
        pts_for_hull = NTuple{2,Float64}[]
        for (li, bj) in nodes
            spec = lines[li]
            (b,d) = bcs[li][bj]
            pb = Float64[spec.x0[1] + b*spec.dir[1], spec.x0[2] + b*spec.dir[2]]
            pd = Float64[spec.x0[1] + d*spec.dir[1], spec.x0[2] + d*spec.dir[2]]
            push!(segs, (pb, pd, spec.omega))
            push!(pts_for_hull, (pb[1],pb[2]))
            push!(pts_for_hull, (pd[1],pd[2]))
        end

        hull = _convex_hull_2d(pts_for_hull; atol=arr.atol)
        areaI = _polygon_area_2d(hull)

        w = 1.0
        if q != 0.0
            w = (areaI / areaR)^Float64(q)
        end
        push!(summands, segs)
        push!(weights, w)
    end

    return MPPDecomposition(lines, summands, weights, (lo,hi))
end

# -------------------------- internal evaluation cache -------------------------

# For fast MPPI image evaluation, we precompute:
# - a per-summand bounding box (used for optional cutoff pruning), and
# - per-segment scalar data including a segment-level bounding box (used for exact pruning).
#
# Each segment is stored as a tuple
#   (px, py, qx, qy, omega, xmin, xmax, ymin, ymax)
# where (xmin,xmax,ymin,ymax) is the segment's axis-aligned bounding box.
struct _MPPImageEvalCache
    segdata::Vector{Vector{NTuple{9,Float64}}}
    bboxes::Vector{NTuple{4,Float64}}  # (xmin, xmax, ymin, ymax) per summand
end

function _mpp_image_eval_cache(decomp::MPPDecomposition)::_MPPImageEvalCache
    ns = length(decomp.summands)
    segdata = Vector{Vector{NTuple{9,Float64}}}(undef, ns)
    bboxes = Vector{NTuple{4,Float64}}(undef, ns)

    for k in 1:ns
        segs = decomp.summands[k]
        m = length(segs)

        v = Vector{NTuple{9,Float64}}(undef, m)

        xmin = Inf
        xmax = -Inf
        ymin = Inf
        ymax = -Inf

        for i in 1:m
            p, q, om = segs[i]
            px = p[1]; py = p[2]
            qx = q[1]; qy = q[2]

            sxmin = (px < qx ? px : qx)
            sxmax = (px < qx ? qx : px)
            symin = (py < qy ? py : qy)
            symax = (py < qy ? qy : py)

            v[i] = (px, py, qx, qy, om, sxmin, sxmax, symin, symax)

            # Update summand bounding box.
            if px < xmin; xmin = px; end
            if qx < xmin; xmin = qx; end
            if px > xmax; xmax = px; end
            if qx > xmax; xmax = qx; end
            if py < ymin; ymin = py; end
            if qy < ymin; ymin = qy; end
            if py > ymax; ymax = py; end
            if qy > ymax; ymax = qy; end
        end

        segdata[k] = v
        if m == 0
            # Should not happen for MPPI summands, but keep a sensible default.
            bboxes[k] = (0.0, 0.0, 0.0, 0.0)
        else
            bboxes[k] = (xmin, xmax, ymin, ymax)
        end
    end

    return _MPPImageEvalCache(segdata, bboxes)
end

# Determine an effective cutoff radius from either an explicit radius or a kernel-value tolerance.
#
# - If both are omitted, returns Inf (no cutoff).
# - If cutoff_tol is provided, we use exp(-d^2/sigma^2) <= cutoff_tol to define the radius.
function _mpp_effective_cutoff_radius(sig::Float64,
                                      cutoff_radius::Union{Nothing,Real},
                                      cutoff_tol::Union{Nothing,Real})::Float64
    if cutoff_radius !== nothing && cutoff_tol !== nothing
        throw(ArgumentError("mpp_image: use at most one of cutoff_radius and cutoff_tol"))
    end

    if cutoff_radius !== nothing
        r = Float64(cutoff_radius)
        r >= 0.0 || throw(ArgumentError("mpp_image: cutoff_radius must be >= 0"))
        return r
    end

    if cutoff_tol !== nothing
        tol = Float64(cutoff_tol)
        (0.0 < tol < 1.0) || throw(ArgumentError("mpp_image: cutoff_tol must satisfy 0 < cutoff_tol < 1"))
        # exp(-d^2/sigma^2) <= tol  <=>  d >= sigma * sqrt(log(1/tol))
        return sig * sqrt(log(1.0 / tol))
    end

    return Inf
end


# ---------------------------- MPPI evaluation --------------------------------

"""
    mpp_image(decomp::MPPDecomposition;
              resolution=32, xgrid=nothing, ygrid=nothing,
              sigma=0.05, cutoff_radius=nothing, cutoff_tol=nothing,
              segment_prune=true)

Evaluate a Carriere multiparameter persistence image from a precomputed vineyard decomposition.

This method does not recompute barcodes or matchings; it only evaluates the kernel sum
on a grid in R^2.

Keyword arguments
- `resolution`: number of grid points per axis (used when `xgrid` or `ygrid` is `nothing`).
- `xgrid`, `ygrid`: explicit grids. If provided, they are converted to `Vector{Float64}`.
- `sigma`: Gaussian scale in exp(-d^2/sigma^2).
- `cutoff_radius`: optional distance cutoff. If set, summands whose distance to the query
  point exceeds this cutoff are skipped. This is an approximation.
- `cutoff_tol`: alternative to `cutoff_radius`. Defines the cutoff radius by
  `exp(-d^2/sigma^2) <= cutoff_tol`. Requires `0 < cutoff_tol < 1`.
- `segment_prune`: if true, use exact bounding-box pruning inside the nearest-segment search
  (this does not change the result; it only reduces work).

Returns an `MPPImage`.
"""
function mpp_image(decomp::MPPDecomposition;
                   resolution::Int=32,
                   xgrid=nothing,
                   ygrid=nothing,
                   sigma::Real=0.05,
                   cutoff_radius::Union{Nothing,Real}=nothing,
                   cutoff_tol::Union{Nothing,Real}=nothing,
                   segment_prune::Bool=true)

    sig = Float64(sigma)
    sig > 0.0 || error("mpp_image: sigma must be positive")

    # If at least one axis uses `resolution`, enforce it.
    if xgrid === nothing || ygrid === nothing
        resolution > 1 || error("mpp_image: resolution must be >= 2")
    end

    lo, hi = decomp.box

    xg = if xgrid === nothing
        collect(range(lo[1], hi[1], length=resolution))
    else
        xgtmp = Float64.(collect(xgrid))
        length(xgtmp) > 1 || error("mpp_image: xgrid must have length >= 2")
        xgtmp
    end

    yg = if ygrid === nothing
        collect(range(lo[2], hi[2], length=resolution))
    else
        ygtmp = Float64.(collect(ygrid))
        length(ygtmp) > 1 || error("mpp_image: ygrid must have length >= 2")
        ygtmp
    end

    cutoff = _mpp_effective_cutoff_radius(sig, cutoff_radius, cutoff_tol)
    cutoff2 = isfinite(cutoff) ? cutoff*cutoff : Inf

    eval_cache = _mpp_image_eval_cache(decomp)

    img = zeros(Float64, length(yg), length(xg))
    invsig2 = 1.0/(sig*sig)

    ns = length(decomp.weights)
    weights = decomp.weights
    segdata = eval_cache.segdata
    bboxes = eval_cache.bboxes

    # Loop order:
    # - ix outer, iy inner writes contiguous memory in Julia's column-major layout
    #   because img[iy, ix] has the first index varying fastest.
    @inbounds for ix in 1:length(xg)
        x = xg[ix]
        for iy in 1:length(yg)
            y = yg[iy]

            acc = 0.0

            for k in 1:ns
                wI = weights[k]
                wI == 0.0 && continue

                # Optional approximate pruning: skip summands whose bounding box is farther
                # than the cutoff radius from the query point.
                if cutoff2 < Inf
                    (xmin, xmax, ymin, ymax) = bboxes[k]
                    if _bbox_dist2_xy(x, y, xmin, xmax, ymin, ymax) > cutoff2
                        continue
                    end
                end

                bestd2 = Inf
                bestomega = 0.0

                segs = segdata[k]
                @inbounds for j in 1:length(segs)
                    seg = segs[j]
                    (px, py, qx, qy, om, sxmin, sxmax, symin, symax) = seg

                    # Exact pruning: if this segment's bounding box is already farther
                    # than the best distance found so far, it cannot improve the minimum.
                    if segment_prune && bestd2 < Inf
                        if _bbox_dist2_xy(x, y, sxmin, sxmax, symin, symax) >= bestd2
                            continue
                        end
                    end

                    d2 = _dist2_point_segment_xy(x, y, px, py, qx, qy)
                    if d2 < bestd2
                        bestd2 = d2
                        bestomega = om
                        if bestd2 == 0.0
                            break
                        end
                    end
                end

                # Optional cutoff: if the nearest segment is still beyond the cutoff,
                # the kernel value is below exp(-cutoff^2/sigma^2).
                if cutoff2 < Inf && bestd2 > cutoff2
                    continue
                end

                acc += wI * bestomega * exp(-bestd2 * invsig2)
            end

            img[iy, ix] = acc
        end
    end

    return MPPImage(xg, yg, img, sig, decomp)
end

"""
    mpp_image(cache::FiberedBarcodeCache2D;
              resolution=32, xgrid=nothing, ygrid=nothing,
              sigma=0.05, N=16, delta=:auto, q=1.0,
              tie_break=:center,
              cutoff_radius=nothing, cutoff_tol=nothing, segment_prune=true)

Compute a Carriere multiparameter persistence image (MPPI) from a fibered-barcode cache.

This does two stages:
1) `mpp_decomposition(cache; N, delta, q, tie_break)`  (barcodes + matchings + vineyard tracks)
2) `mpp_image(decomp; resolution, xgrid, ygrid, sigma, cutoff_radius, cutoff_tol, segment_prune)`

Keyword arguments
- `resolution`, `xgrid`, `ygrid`, `sigma`: image evaluation parameters.
- `N`, `delta`, `q`, `tie_break`: Carriere decomposition parameters (see `mpp_decomposition`).
- `cutoff_radius` / `cutoff_tol`: optional approximate cutoff for faster evaluation.
- `segment_prune`: exact bounding-box pruning inside each summand's nearest-segment search.

Returns an `MPPImage`.
"""
function mpp_image(cache::FiberedBarcodeCache2D;
                   resolution::Int=32,
                   xgrid=nothing,
                   ygrid=nothing,
                   sigma::Real=0.05,
                   N::Int=16,
                   delta::Union{Real,Symbol}=:auto,
                   q::Real=1.0,
                   tie_break::Symbol=:center,
                   cutoff_radius::Union{Nothing,Real}=nothing,
                   cutoff_tol::Union{Nothing,Real}=nothing,
                   segment_prune::Bool=true)

    decomp = mpp_decomposition(cache; N=N, delta=delta, q=q, tie_break=tie_break)
    return mpp_image(decomp;
                     resolution=resolution,
                     xgrid=xgrid,
                     ygrid=ygrid,
                     sigma=sigma,
                     cutoff_radius=cutoff_radius,
                     cutoff_tol=cutoff_tol,
                     segment_prune=segment_prune)
end

# Convenience overloads: build a fibered cache from (M,pi) directly.
#
# These constructors mirror `fibered_barcode_cache_2d` so that MPPI can be used
# with the same "shared arrangement" workflow as other 2D slice-based invariants.

function mpp_decomposition(M::PModule{QQ}, pi;
                           arrangement=nothing,
                           precompute::Symbol=:none,
                           box=nothing,
                           normalize_dirs::Symbol=:L1,
                           include_axes::Bool=false,
                           strict::Bool=true,
                           atol::Real=1e-12,
                           max_combinations::Int=200_000,
                           max_vertices::Int=20_000,
                           max_cells::Int=5_000_000,
                           kwargs...)
    cache = fibered_barcode_cache_2d(M, pi;
                                    arrangement=arrangement,
                                    precompute=precompute,
                                    box=box,
                                    normalize_dirs=normalize_dirs,
                                    include_axes=include_axes,
                                    strict=strict,
                                    atol=Float64(atol),
                                    max_combinations=max_combinations,
                                    max_vertices=max_vertices,
                                    max_cells=max_cells)
    return mpp_decomposition(cache; kwargs...)
end

function mpp_image(M::PModule{QQ}, pi;
                   arrangement=nothing,
                   precompute::Symbol=:none,
                   box=nothing,
                   normalize_dirs::Symbol=:L1,
                   include_axes::Bool=false,
                   strict::Bool=true,
                   atol::Real=1e-12,
                   max_combinations::Int=200_000,
                   max_vertices::Int=20_000,
                   max_cells::Int=5_000_000,
                   kwargs...)
    cache = fibered_barcode_cache_2d(M, pi;
                                    arrangement=arrangement,
                                    precompute=precompute,
                                    box=box,
                                    normalize_dirs=normalize_dirs,
                                    include_axes=include_axes,
                                    strict=strict,
                                    atol=Float64(atol),
                                    max_combinations=max_combinations,
                                    max_vertices=max_vertices,
                                    max_cells=max_cells)
    return mpp_image(cache; kwargs...)
end


"""
    mpp_image_inner_product(A::MPPImage, B::MPPImage)

L2 inner product of two MPPI images (same grid required).
"""
function mpp_image_inner_product(A::MPPImage, B::MPPImage)
    A.xgrid == B.xgrid || error("mpp_image_inner_product: xgrid mismatch")
    A.ygrid == B.ygrid || error("mpp_image_inner_product: ygrid mismatch")
    size(A.img) == size(B.img) || error("mpp_image_inner_product: img size mismatch")
    return sum(A.img .* B.img)
end

"""
    mpp_image_distance(A::MPPImage, B::MPPImage)

L2 distance between two MPPI images (same grid required).
"""
function mpp_image_distance(A::MPPImage, B::MPPImage)::Float64
    ipAA = mpp_image_inner_product(A,A)
    ipBB = mpp_image_inner_product(B,B)
    ipAB = mpp_image_inner_product(A,B)
    val = ipAA + ipBB - 2.0*ipAB
    return sqrt(max(val, 0.0))
end

"""
    mpp_image_kernel(A::MPPImage, B::MPPImage; sigma=1.0)

Gaussian kernel on MPPI images:
    exp(-||A-B||^2 / sigma^2)
"""
function mpp_image_kernel(A::MPPImage, B::MPPImage; sigma::Real=1.0)::Float64
    s = Float64(sigma)
    s > 0.0 || error("mpp_image_kernel: sigma must be positive")
    d = mpp_image_distance(A,B)
    return exp(-(d*d)/(s*s))
end


# ---------------------------- MPPI JSON serialization -------------------------

# The package already contains JSON serialization helpers for other objects (see Serialization.jl).
# For MPPI we keep the schema intentionally simple and self-contained, so that users can:
#   - cache expensive decompositions (barcodes + matchings),
#   - cache full images (for downstream ML / distance computations),
#   - reload them without needing access to the original module.

function _mpp_floatvec2(x)::Vector{Float64}
    length(x) == 2 || error("MPPI JSON: expected a length-2 vector")
    return Float64[Float64(x[1]), Float64(x[2])]
end

function _mpp_decomposition_to_dict(decomp::MPPDecomposition)
    lines = Vector{Any}(undef, length(decomp.lines))
    for (i, ls) in enumerate(decomp.lines)
        lines[i] = Dict(
            "dir" => ls.dir,
            "off" => ls.off,
            "x0" => ls.x0,
            "omega" => ls.omega,
        )
    end

    summands = Vector{Any}(undef, length(decomp.summands))
    for k in 1:length(decomp.summands)
        segs = decomp.summands[k]
        arr = Vector{Any}(undef, length(segs))
        for j in 1:length(segs)
            (p, q, om) = segs[j]
            arr[j] = Dict("p" => p, "q" => q, "omega" => om)
        end
        summands[k] = arr
    end

    lo, hi = decomp.box

    return Dict(
        "kind" => "MPPDecomposition",
        "version" => 1,
        "lines" => lines,
        "summands" => summands,
        "weights" => decomp.weights,
        "box" => Dict("lo" => lo, "hi" => hi),
    )
end

function _mpp_decomposition_from_dict(obj)::MPPDecomposition
    if !haskey(obj, "kind") || String(obj["kind"]) != "MPPDecomposition"
        error("MPPI JSON: expected kind == 'MPPDecomposition'")
    end

    lines_obj = obj["lines"]
    lines = Vector{MPPLineSpec}(undef, length(lines_obj))
    for (i, l) in enumerate(lines_obj)
        dir = _mpp_floatvec2(l["dir"])
        off = Float64(l["off"])
        x0 = _mpp_floatvec2(l["x0"])
        omega = Float64(l["omega"])
        lines[i] = MPPLineSpec(dir, off, x0, omega)
    end

    summands_obj = obj["summands"]
    summands = Vector{Vector{Tuple{Vector{Float64},Vector{Float64},Float64}}}(undef, length(summands_obj))
    for (k, s) in enumerate(summands_obj)
        segs = Vector{Tuple{Vector{Float64},Vector{Float64},Float64}}(undef, length(s))
        for (j, seg) in enumerate(s)
            p = _mpp_floatvec2(seg["p"])
            q = _mpp_floatvec2(seg["q"])
            om = Float64(seg["omega"])
            segs[j] = (p, q, om)
        end
        summands[k] = segs
    end

    weights_obj = obj["weights"]
    weights = Float64[Float64(w) for w in weights_obj]

    box_obj = obj["box"]
    lo = _mpp_floatvec2(box_obj["lo"])
    hi = _mpp_floatvec2(box_obj["hi"])

    return MPPDecomposition(lines, summands, weights, (lo, hi))
end

function _mpp_image_to_dict(img::MPPImage; include_decomp::Bool=true)
    ny, nx = size(img.img)
    mat = Vector{Any}(undef, ny)
    for i in 1:ny
        # Store as row-major arrays (each JSON row is a Julia row iy).
        mat[i] = [img.img[i, j] for j in 1:nx]
    end

    d = Dict(
        "kind" => "MPPImage",
        "version" => 1,
        "sigma" => img.sigma,
        "xgrid" => img.xgrid,
        "ygrid" => img.ygrid,
        "img" => mat,
    )

    if include_decomp
        d["decomp"] = _mpp_decomposition_to_dict(img.decomp)
    end

    return d
end

function _mpp_image_from_dict(obj)::MPPImage
    if !haskey(obj, "kind") || String(obj["kind"]) != "MPPImage"
        error("MPPI JSON: expected kind == 'MPPImage'")
    end

    sig = Float64(obj["sigma"])
    xgrid = Float64[Float64(x) for x in obj["xgrid"]]
    ygrid = Float64[Float64(y) for y in obj["ygrid"]]

    rows = obj["img"]
    length(rows) == length(ygrid) || error("MPPI JSON: img row count does not match ygrid")
    imgmat = zeros(Float64, length(ygrid), length(xgrid))
    for i in 1:length(ygrid)
        row = rows[i]
        length(row) == length(xgrid) || error("MPPI JSON: img column count does not match xgrid")
        for j in 1:length(xgrid)
            imgmat[i, j] = Float64(row[j])
        end
    end

    if !haskey(obj, "decomp")
        error("MPPI JSON: missing field 'decomp' (cannot reconstruct MPPImage without it)")
    end
    decomp = _mpp_decomposition_from_dict(obj["decomp"])

    return MPPImage(xgrid, ygrid, imgmat, sig, decomp)
end

"""
    save_mpp_decomposition_json(path, decomp::MPPDecomposition)

Save an `MPPDecomposition` to a JSON file.

This is a good cache point: the decomposition contains the vineyard tracks and weights,
but not the full image grid. After loading, evaluate images via `mpp_image(decomp; ...)`.

Returns `path`.
"""
function save_mpp_decomposition_json(path::AbstractString, decomp::MPPDecomposition)
    obj = _mpp_decomposition_to_dict(decomp)
    open(path, "w") do io
        JSON3.write(io, obj; allow_inf=true, indent=2)
    end
    return path
end

"""
    load_mpp_decomposition_json(path) -> MPPDecomposition

Load an `MPPDecomposition` written by `save_mpp_decomposition_json`.
"""
function load_mpp_decomposition_json(path::AbstractString)::MPPDecomposition
    obj = open(JSON3.read, path)
    return _mpp_decomposition_from_dict(obj)
end

"""
    save_mpp_image_json(path, img::MPPImage; include_decomp=true)

Save an `MPPImage` (including its decomposition by default) to a JSON file.

Returns `path`.
"""
function save_mpp_image_json(path::AbstractString, img::MPPImage; include_decomp::Bool=true)
    obj = _mpp_image_to_dict(img; include_decomp=include_decomp)
    open(path, "w") do io
        JSON3.write(io, obj; allow_inf=true, indent=2)
    end
    return path
end

"""
    load_mpp_image_json(path) -> MPPImage

Load an `MPPImage` written by `save_mpp_image_json`.
"""
function load_mpp_image_json(path::AbstractString)::MPPImage
    obj = open(JSON3.read, path)
    return _mpp_image_from_dict(obj)
end



# ----- Multiparameter persistence landscapes ----------------------------------
#
# We implement a practical, finite-sampled version of Vipond's multiparameter
# persistence landscapes. The key computational primitive is already in place:
# for a chosen geometric slice (line) we can restrict a multiparameter module to
# a chain in a finite encoding, compute its 1D barcode, and then convert that
# barcode into an ordinary 1D persistence landscape (Bubenik).
#
# The resulting "mp landscape" is represented here as a family of 1D landscapes
# indexed by a finite list of slice directions and offsets, together with a
# common evaluation grid and slice weights. This makes it directly usable as a
# stable feature map for statistics and kernel methods.

"""
    PersistenceLandscape1D

A sampled 1D persistence landscape.

Fields
------
- `tgrid::Vector{Float64}`: sorted evaluation points.
- `values::Matrix{Float64}`: a `kmax x length(tgrid)` matrix where
  `values[k,i]` is the k-th landscape layer evaluated at `tgrid[i]`.

Notes
-----
This is a discrete representation. Use `landscape_value(pl, k, t)` to evaluate
by piecewise-linear interpolation on the stored grid.
"""
struct PersistenceLandscape1D
    tgrid::Vector{Float64}
    values::Matrix{Float64}
end

Base.size(pl::PersistenceLandscape1D) = size(pl.values)
Base.getindex(pl::PersistenceLandscape1D, k::Int, i::Int) = pl.values[k, i]
Base.getindex(pl::PersistenceLandscape1D, k::Int) = view(pl.values, k, :)

"""
    landscape_value(pl, k, t; extrapolate=false) -> Float64

Evaluate the k-th landscape layer at parameter value `t` by piecewise-linear
interpolation on the stored grid `pl.tgrid`.

If `t` lies outside the grid and `extrapolate=false` (default), this returns 0.0.
If `extrapolate=true`, it clamps to the nearest endpoint value.
"""
function landscape_value(pl::PersistenceLandscape1D, k::Int, t::Real; extrapolate::Bool=false)::Float64
    1 <= k <= size(pl.values, 1) || error("landscape_value: k out of range")
    tt = float(t)
    tg = pl.tgrid
    nt = length(tg)

    if nt == 0
        return 0.0
    elseif nt == 1
        return extrapolate ? pl.values[k, 1] : 0.0
    end

    # Outside the sampled window.
    if tt <= tg[1]
        return extrapolate ? pl.values[k, 1] : 0.0
    elseif tt >= tg[end]
        return extrapolate ? pl.values[k, end] : 0.0
    end

    # Find i with tg[i] <= tt < tg[i+1].
    i = searchsortedlast(tg, tt)
    i == nt && (i = nt - 1)

    t0 = tg[i]
    t1 = tg[i + 1]
    v0 = pl.values[k, i]
    v1 = pl.values[k, i + 1]

    # Linear interpolation (t0 < t1 is guaranteed by construction).
    alpha = (tt - t0) / (t1 - t0)
    return (1.0 - alpha) * v0 + alpha * v1
end

# Internal: clean and validate a grid.
function _clean_tgrid(tgrid)::Vector{Float64}
    tg = Float64[float(t) for t in collect(tgrid)]
    sort!(tg)
    unique!(tg)

    # We need at least two points for integration; fall back to a trivial grid.
    if length(tg) < 2
        tg = Float64[0.0, 1.0]
    end
    return tg
end

# Internal: the tent function associated to an interval (b,d).
# This is max(0, min(t-b, d-t)), i.e. a triangle with peak at (b+d)/2.
@inline function _tent_value(b::Float64, d::Float64, t::Float64)::Float64
    v = min(t - b, d - t)
    return v > 0 ? v : 0.0
end

"""
    persistence_landscape(bar; kmax=5, tgrid=nothing, nsteps=401) -> PersistenceLandscape1D

Compute the first `kmax` layers of the 1D persistence landscape associated to a
barcode `bar`.

Input formats for `bar`
-----------------------
- `Dict((birth, death) => multiplicity)`
- `Vector{Tuple{birth, death}}` (multiplicity 1)

Evaluation
----------
If `tgrid` is provided, the landscape is sampled on that grid (after sorting and
de-duplication). If `tgrid=nothing`, a default uniform grid with `nsteps` points
is chosen from `min(birth)` to `max(death)`.

The output stores a matrix `values` of size `(kmax, length(tgrid))` where
`values[k,i]` equals the k-th landscape layer evaluated at `tgrid[i]`.
"""
function persistence_landscape(
    bar;
    kmax::Int=5,
    tgrid=nothing,
    nsteps::Int=401
)::PersistenceLandscape1D
    kmax >= 1 || error("persistence_landscape: kmax must be >= 1")

    pts = _barcode_points(bar)  # Vector{Tuple{Float64,Float64}} (expanded multiplicities)

    # Choose a default grid if needed.
    if tgrid === nothing
        if isempty(pts)
            tg = Float64[0.0, 1.0]
        else
            bmin = minimum(p[1] for p in pts)
            dmax = maximum(p[2] for p in pts)
            (bmin < dmax) || (dmax = bmin + 1.0)
            tg = collect(range(bmin, dmax; length=nsteps))
        end
    else
        tg = collect(tgrid)
    end

    tg = _clean_tgrid(tg)
    nt = length(tg)

    # Early exit: empty barcode.
    if isempty(pts)
        return PersistenceLandscape1D(tg, zeros(Float64, kmax, nt))
    end

    # Validate intervals.
    for (b, d) in pts
        b < d || error("persistence_landscape: invalid interval with birth >= death: ($b, $d)")
    end

    # Evaluate the landscape: at each t we take the k-th order statistic of tent values.
    vals = zeros(Float64, kmax, nt)
    tmp = Float64[]

    for (j, t) in enumerate(tg)
        empty!(tmp)
        for (b, d) in pts
            v = _tent_value(b, d, t)
            v > 0 && push!(tmp, v)
        end
        if !isempty(tmp)
            sort!(tmp; rev=true)
            m = min(kmax, length(tmp))
            for k in 1:m
                vals[k, j] = tmp[k]
            end
        end
    end

    return PersistenceLandscape1D(tg, vals)
end


"""
    MPLandscape

A finite-sampled approximation of a multiparameter persistence landscape.

This stores a family of 1D persistence landscapes indexed by a finite list of
slice directions and offsets.

Fields
------
- `kmax`: number of landscape layers stored per slice.
- `tgrid`: common evaluation grid for all slices.
- `values`: an array of size `(ndirs, noffsets, kmax, length(tgrid))`.
  `values[i,j,k,l]` is the k-th landscape layer at `tgrid[l]` for slice
  `(direction[i], offset[j])`.
- `weights`: an `(ndirs, noffsets)` matrix of nonnegative slice weights.
- `directions`, `offsets`: metadata as passed/used in construction.

See also `mp_landscape_distance` and `mp_landscape_kernel`.
"""
struct MPLandscape
    kmax::Int
    tgrid::Vector{Float64}
    values::Array{Float64,4}
    weights::Matrix{Float64}
    directions::Vector{Any}
    offsets::Vector{Any}
end

"""
    L[idir, ioff]

Return a (kmax x length(tgrid)) view of the stored 1D landscape values for the
slice with direction index `idir` and offset index `ioff`.
"""
Base.getindex(L::MPLandscape, idir::Int, ioff::Int) = view(L.values, idir, ioff, :, :)

# -----------------------------------------------------------------------------
# Pretty-printing / ergonomics
# -----------------------------------------------------------------------------

"""
    Base.show(io::IO, L::MPLandscape)

Print a compact one-line summary for a sampled multiparameter persistence
landscape. The summary is meant to be "mathematician-friendly" and includes:

- number of directions (ndirs)
- number of offsets (noffsets)
- kmax (number of landscape layers stored)
- nt (length of the common t-grid)
- whether the stored slice weights appear normalized (sum(weights) ~= 1)

For a more verbose multi-line summary, use:

    show(io, MIME("text/plain"), L)
"""
function Base.show(io::IO, L::MPLandscape)
    nd = size(L.values, 1)
    no = size(L.values, 2)
    nt = length(L.tgrid)

    wsum = sum(L.weights)
    # Heuristic: consider weights "normalized" if they sum to 1 within tolerance.
    wnorm = abs(wsum - 1.0) < 1e-8

    print(io,
          "MPLandscape(",
          "ndirs=", nd,
          ", noffsets=", no,
          ", kmax=", L.kmax,
          ", nt=", nt,
          ", weights_normalized=", wnorm,
          ")")
end

"""
    Base.show(io::IO, ::MIME"text/plain", L::MPLandscape)

Verbose multi-line summary for `MPLandscape`.
"""
function Base.show(io::IO, ::MIME"text/plain", L::MPLandscape)
    nd = size(L.values, 1)
    no = size(L.values, 2)
    nt = length(L.tgrid)

    wsum = sum(L.weights)
    wnorm = abs(wsum - 1.0) < 1e-8

    println(io, "MPLandscape")
    println(io, "  ndirs = ", nd)
    println(io, "  noffsets = ", no)
    println(io, "  kmax = ", L.kmax)
    println(io, "  nt = ", nt)

    if nt > 0
        println(io, "  tmin = ", L.tgrid[1])
        println(io, "  tmax = ", L.tgrid[end])
    end

    println(io, "  weights_sum = ", wsum)
    println(io, "  weights_normalized = ", wnorm)
end


# Internal: normalize a direction vector.
function _normalize_dir(dir::AbstractVector, normalize::Symbol)
    normalize == :none && return dir
    v = Float64[float(x) for x in dir]
    if normalize == :L1
        s = sum(abs.(v))
        s > 0 || error("_normalize_dir: zero direction vector")
        return v ./ s
    elseif normalize == :Linf
        m = maximum(abs.(v))
        m > 0 || error("_normalize_dir: zero direction vector")
        return v ./ m
    else
        error("_normalize_dir: normalize must be :none, :L1, or :Linf")
    end
end

# Internal: trapezoidal rule on a common grid.
function _trapz(tg::Vector{Float64}, f::AbstractVector{Float64})::Float64
    nt = length(tg)
    length(f) == nt || error("_trapz: length mismatch")
    nt < 2 && return 0.0

    s = 0.0
    for i in 1:nt-1
        dt = tg[i + 1] - tg[i]
        dt >= 0 || error("_trapz: tgrid must be sorted increasing")
        s += 0.5 * (f[i] + f[i + 1]) * dt
    end
    return s
end

# Internal: choose weights according to a policy.
function _combine_weights(LA::MPLandscape, LB::MPLandscape, mode::Symbol)::Matrix{Float64}
    if mode == :check
        maximum(abs.(LA.weights .- LB.weights)) < 1e-12 || error("mp_landscape_*: weight mismatch between inputs")
        return LA.weights
    elseif mode == :left
        return LA.weights
    elseif mode == :right
        return LB.weights
    elseif mode == :average
        return 0.5 .* (LA.weights .+ LB.weights)
    else
        error("_combine_weights: mode must be :check, :left, :right, or :average")
    end
end

"""
    mp_landscape(M, slices; kmax=5, tgrid, default_weight=1.0, normalize_weights=true) -> MPLandscape
    mp_landscape(M, pi; directions, offsets, kmax=5, ...) -> MPLandscape

Compute a sampled multiparameter persistence landscape.

Two calling patterns are supported.

1) Explicit slices (purely combinatorial):
    slices = [
        [q1,q2,...,qm],                                # a chain in M.Q
        (chain=[...], values=[...], weight=1.0),       # richer spec (NamedTuple)
        (chain, values, weight)                        # tuple form
    ]
    L = mp_landscape(M, slices; kmax=5, tgrid=...)

2) Geometric slices via a finite encoding map `pi`:
    L = mp_landscape(M, pi;
        directions=[v1,v2,...],
        offsets=[x01,x02,...],
        tmin=..., tmax=..., nsteps=..., strict=true,
        normalize_dirs=:none,
        direction_weight=:none,
        normalize_weights=true
    )

The geometric version builds chains via `slice_chain`, computes 1D barcodes via
`slice_barcode`, and converts them to 1D persistence landscapes via
`persistence_landscape`.

Remarks for Z^n
---------------
For `pi::ZnEncodingMap` you can specify the integer sampling range with
`kmin=...` and `kmax_param=...` (note the name `kmax_param` to avoid conflict
with the landscape layer parameter `kmax`).

See also `mp_landscape_distance` and `mp_landscape_kernel`.
"""
function mp_landscape(
    M::PModule{QQ},
    slices::AbstractVector;
    kmax::Int=5,
    tgrid,
    default_weight=1.0,
    normalize_weights::Bool=true
)::MPLandscape
    tg = _clean_tgrid(tgrid)
    nt = length(tg)

    nslices = length(slices)
    nslices > 0 || error("mp_landscape: slices list is empty")

    vals = zeros(Float64, nslices, 1, kmax, nt)
    W = zeros(Float64, nslices, 1)
    dirs = Vector{Any}(undef, nslices)
    offs = Any[nothing]

    for (i, spec) in enumerate(slices)
        chain = nothing
        v = nothing
        w = float(default_weight)

        if spec isa AbstractVector{Int}
            chain = spec
        elseif spec isa NamedTuple
            chain = spec.chain
            v = get(spec, :values, nothing)
            w = float(get(spec, :weight, default_weight))
        elseif spec isa Tuple
            length(spec) >= 1 || error("mp_landscape: empty tuple slice spec")
            chain = spec[1]
            v = length(spec) >= 2 ? spec[2] : nothing
            w = length(spec) >= 3 ? float(spec[3]) : float(default_weight)
        else
            error("mp_landscape: unknown slice spec type $(typeof(spec))")
        end

        bc = slice_barcode(M, chain; values=v)
        pl = persistence_landscape(bc; kmax=kmax, tgrid=tg)
        vals[i, 1, :, :] = pl.values
        W[i, 1] = w
        dirs[i] = spec
    end

    if normalize_weights
        s = sum(W)
        s > 0 || error("mp_landscape: total slice weight is zero")
        W ./= s
    end

    return MPLandscape(kmax, tg, vals, W, dirs, offs)
end

mp_landscape(
    M::PModule{QQ},
    chain::AbstractVector{Int};
    kwargs...
)::MPLandscape = mp_landscape(M, [chain]; kwargs...)

function mp_landscape(
    M::PModule{QQ},
    pi;
    directions=nothing,
    offsets=nothing,
    kmax::Int=5,
    tgrid=nothing,
    tmin::Union{Real,Nothing}=nothing,
    tmax::Union{Real,Nothing}=nothing,
    nsteps::Int=401,
    kmin=nothing,
    kmax_param=nothing,
    strict::Bool=true,
    drop_unknown::Bool=true,
    dedup::Bool=true,
    normalize_dirs::Symbol=:none,
    direction_weight::Symbol=:none,
    normalize_weights::Bool=true,
    slice_kwargs...
)::MPLandscape
    directions === nothing && error("mp_landscape: provide directions=...")
    offsets === nothing && error("mp_landscape: provide offsets=...")

    # Normalize directions if requested.
    dirs_in = Any[_normalize_dir(dir, normalize_dirs) for dir in directions]

    nd = length(dirs_in)
    no = length(offsets)
    nd > 0 || error("mp_landscape: directions is empty")
    no > 0 || error("mp_landscape: offsets is empty")

    # Choose the landscape evaluation grid.
    if tgrid === nothing
        if (kmin !== nothing) && (kmax_param !== nothing)
            tmin_eff = float(kmin)
            tmax_eff = float(kmax_param)
            tg = collect(range(tmin_eff, tmax_eff; length=nsteps))
        else
            # Semantic default for the landscape t-grid:
            # If a window is present and a bound is omitted, use the union of
            # line-window intersection intervals across all requested slices.
            #
            # If no window is present (or no slice intersects), fall back to [0, 1].
            box = haskey(slice_kwargs, :box) ? slice_kwargs[:box] : nothing
            box2 = haskey(slice_kwargs, :box2) ? slice_kwargs[:box2] : nothing
            clip = (box !== nothing) || (box2 !== nothing)

            tlo_glob = 0.0
            thi_glob = 1.0

            if clip && (tmin === nothing || tmax === nothing)
                bx = _intersect_axis_aligned_boxes(box, box2)
                tlo_tmp = Inf
                thi_tmp = -Inf
                atol = 1e-12

                for dir in dirs_in
                    for x0 in offsets
                        tlo, thi = _line_param_range_in_box_nd(x0, dir, bx; atol=atol)
                        if tlo <= thi
                            tlo_tmp = min(tlo_tmp, tlo)
                            thi_tmp = max(thi_tmp, thi)
                        end
                    end
                end

                if tlo_tmp != Inf
                    tlo_glob = tlo_tmp
                    thi_glob = thi_tmp
                end
            end

            tmin_eff = (tmin === nothing) ? tlo_glob : float(tmin)
            tmax_eff = (tmax === nothing) ? thi_glob : float(tmax)
            tg = collect(range(tmin_eff, tmax_eff; length=nsteps))
        end
    else
        tg = collect(tgrid)
    end

    tg = _clean_tgrid(tg)

    nt = length(tg)

    # Weights: one per (direction, offset).
    W = zeros(Float64, nd, no)
    for i in 1:nd
        wdir = _direction_weight(dirs_in[i]; scheme=direction_weight)
        for j in 1:no
            W[i, j] = wdir
        end
    end
    if normalize_weights
        s = sum(W)
        s > 0 || error("mp_landscape: total slice weight is zero")
        W ./= s
    end

    vals = zeros(Float64, nd, no, kmax, nt)

    for i in 1:nd
        dir = dirs_in[i]
        for j in 1:no
            x0 = offsets[j]

            chain = Int[]
            tvals = Any[]

            if (kmin !== nothing) && (kmax_param !== nothing)
                chain, tvals = slice_chain(pi, x0, dir;
                                           kmin=Int(kmin), kmax=Int(kmax_param),
                                           strict=strict,
                                           drop_unknown=drop_unknown,
                                           dedup=dedup,
                                           slice_kwargs...)
            else
                chain, tvals = slice_chain(pi, x0, dir;
                                           tmin=tmin, tmax=tmax, nsteps=nsteps,
                                           strict=strict,
                                           drop_unknown=drop_unknown,
                                           dedup=dedup,
                                           slice_kwargs...)
            end

            if isempty(chain)
                # Keep the all-zero landscape for this slice.
                continue
            end

            bc = slice_barcode(M, chain; values=tvals)
            pl = persistence_landscape(bc; kmax=kmax, tgrid=tg)
            vals[i, j, :, :] = pl.values
        end
    end

    return MPLandscape(kmax, tg, vals, W, collect(dirs_in), collect(offsets))
end


"""
    mp_landscape_distance(LM, LN; p=2, weight_mode=:check) -> Float64
    mp_landscape_distance(M, N, slices; p=2, ...) -> Float64
    mp_landscape_distance(M, N, pi; p=2, ...) -> Float64

Compute an L^p distance between multiparameter persistence landscapes.

- If `LM` and `LN` are `MPLandscape` objects, this compares them directly.
- If modules `M` and `N` are provided, the landscapes are constructed using the
  provided slice data (explicit `slices` or geometric `pi`) and then compared.

The integral in the t-parameter is approximated by the trapezoidal rule on the
common grid stored in the landscape object(s). The sum over k is truncated at
the stored `kmax`.

`p` may be any positive real number or `Inf`.
"""
function mp_landscape_distance(
    LA::MPLandscape,
    LB::MPLandscape;
    p::Real=2,
    weight_mode::Symbol=:check
)::Float64
    LA.kmax == LB.kmax || error("mp_landscape_distance: kmax mismatch")
    size(LA.values) == size(LB.values) || error("mp_landscape_distance: values array shape mismatch")
    length(LA.tgrid) == length(LB.tgrid) || error("mp_landscape_distance: tgrid length mismatch")
    maximum(abs.(LA.tgrid .- LB.tgrid)) < 1e-12 || error("mp_landscape_distance: tgrid mismatch")

    W = _combine_weights(LA, LB, weight_mode)

    nd, no, kmax, nt = size(LA.values)
    tg = LA.tgrid
    pp = float(p)

    if isinf(pp)
        best = 0.0
        for i in 1:nd, j in 1:no
            w = W[i, j]
            for k in 1:kmax
                dv = maximum(abs.(view(LA.values, i, j, k, :) .- view(LB.values, i, j, k, :)))
                best = max(best, w * dv)
            end
        end
        return best
    end

    pp > 0 || error("mp_landscape_distance: p must be > 0")
    acc = 0.0

    for i in 1:nd, j in 1:no
        w = W[i, j]
        w == 0.0 && continue

        for k in 1:kmax
            # Integrate |lambda_M - lambda_N|^p over t using trapezoid rule.
            s = 0.0
            for it in 1:nt-1
                dt = tg[it + 1] - tg[it]
                a1 = abs(LA.values[i, j, k, it] - LB.values[i, j, k, it])
                a2 = abs(LA.values[i, j, k, it + 1] - LB.values[i, j, k, it + 1])
                s += 0.5 * (a1^pp + a2^pp) * dt
            end
            acc += w * s
        end
    end

    return acc^(1 / pp)
end

function mp_landscape_distance(
    M::PModule{QQ},
    N::PModule{QQ},
    slices::AbstractVector;
    p::Real=2,
    weight_mode::Symbol=:check,
    mp_kwargs...
)::Float64
    LM = mp_landscape(M, slices; mp_kwargs...)
    LN = mp_landscape(N, slices; mp_kwargs...)
    return mp_landscape_distance(LM, LN; p=p, weight_mode=weight_mode)
end

function mp_landscape_distance(
    M::PModule{QQ},
    N::PModule{QQ},
    pi;
    p::Real=2,
    weight_mode::Symbol=:check,
    mp_kwargs...
)::Float64
    LM = mp_landscape(M, pi; mp_kwargs...)
    LN = mp_landscape(N, pi; mp_kwargs...)
    return mp_landscape_distance(LM, LN; p=p, weight_mode=weight_mode)
end


"""
    mp_landscape_inner_product(LM, LN; weight_mode=:check) -> Float64

Approximate the L^2 inner product between two multiparameter persistence landscapes:

    <LM, LN> = sum_k int LM_k(t) * LN_k(t) dt,

with the integral approximated by the trapezoidal rule on the common grid and
with slice weights applied.
"""
function mp_landscape_inner_product(
    LA::MPLandscape,
    LB::MPLandscape;
    weight_mode::Symbol=:check
)::Float64
    LA.kmax == LB.kmax || error("mp_landscape_inner_product: kmax mismatch")
    size(LA.values) == size(LB.values) || error("mp_landscape_inner_product: values array shape mismatch")
    length(LA.tgrid) == length(LB.tgrid) || error("mp_landscape_inner_product: tgrid length mismatch")
    maximum(abs.(LA.tgrid .- LB.tgrid)) < 1e-12 || error("mp_landscape_inner_product: tgrid mismatch")

    W = _combine_weights(LA, LB, weight_mode)

    nd, no, kmax, nt = size(LA.values)
    tg = LA.tgrid
    acc = 0.0

    for i in 1:nd, j in 1:no
        w = W[i, j]
        w == 0.0 && continue

        for k in 1:kmax
            s = 0.0
            for it in 1:nt-1
                dt = tg[it + 1] - tg[it]
                a1 = LA.values[i, j, k, it] * LB.values[i, j, k, it]
                a2 = LA.values[i, j, k, it + 1] * LB.values[i, j, k, it + 1]
                s += 0.5 * (a1 + a2) * dt
            end
            acc += w * s
        end
    end

    return acc
end


"""
    mp_landscape_kernel(LM, LN; kind=:gaussian, sigma=1.0, p=2, gamma=nothing) -> Float64
    mp_landscape_kernel(M, N, slices; ...) -> Float64
    mp_landscape_kernel(M, N, pi; ...) -> Float64

Kernels derived from the multiparameter persistence landscape.

Supported kinds
---------------
- `kind=:gaussian` (default): `exp(-gamma * d^2)` where `d` is the `mp_landscape_distance`
  with exponent `p` (default `p=2`). If `gamma` is not provided, it uses
  `gamma = 1/(2*sigma^2)`.
- `kind=:laplacian`: `exp(-d/sigma)`.
- `kind=:linear`: L^2 inner product (only meaningful with `p=2`).

These are intended as practical building blocks for statistics and kernel
methods using multiparameter persistence.
"""
function mp_landscape_kernel(
    LA::MPLandscape,
    LB::MPLandscape;
    kind::Symbol=:gaussian,
    sigma::Real=1.0,
    p::Real=2,
    gamma=nothing,
    weight_mode::Symbol=:check
)::Float64
    if kind == :linear
        return mp_landscape_inner_product(LA, LB; weight_mode=weight_mode)
    end

    d = mp_landscape_distance(LA, LB; p=p, weight_mode=weight_mode)

    if kind == :gaussian
        if gamma === nothing
            sigma > 0 || error("mp_landscape_kernel: sigma must be > 0")
            gamma = 1.0 / (2.0 * float(sigma)^2)
        end
        return exp(-float(gamma) * d^2)
    elseif kind == :laplacian
        sigma > 0 || error("mp_landscape_kernel: sigma must be > 0")
        return exp(-d / float(sigma))
    else
        error("mp_landscape_kernel: kind must be :gaussian, :laplacian, or :linear")
    end
end

function mp_landscape_kernel(
    M::PModule{QQ},
    N::PModule{QQ},
    slices::AbstractVector;
    kind::Symbol=:gaussian,
    sigma::Real=1.0,
    p::Real=2,
    gamma=nothing,
    weight_mode::Symbol=:check,
    mp_kwargs...
)::Float64
    LM = mp_landscape(M, slices; mp_kwargs...)
    LN = mp_landscape(N, slices; mp_kwargs...)
    return mp_landscape_kernel(LM, LN;
                               kind=kind,
                               sigma=sigma,
                               p=p,
                               gamma=gamma,
                               weight_mode=weight_mode)
end

function mp_landscape_kernel(
    M::PModule{QQ},
    N::PModule{QQ},
    pi;
    directions=nothing,
    offsets=nothing,
    kind::Symbol=:gaussian,
    sigma::Real=1.0,
    p::Real=2,
    gamma=nothing,
    weight_mode::Symbol=:check,
    mp_kwargs...
)::Float64
    directions === nothing && error("mp_landscape_kernel: provide directions=...")
    offsets === nothing && error("mp_landscape_kernel: provide offsets=...")

    LM = mp_landscape(M, pi; directions=directions, offsets=offsets, mp_kwargs...)
    LN = mp_landscape(N, pi; directions=directions, offsets=offsets, mp_kwargs...)
    return mp_landscape_kernel(LM, LN;
                               kind=kind,
                               sigma=sigma,
                               p=p,
                               gamma=gamma,
                               weight_mode=weight_mode)
end

# ----- Slice vectorizations and sliced kernels -------------------------------

"""
    PersistenceImage1D

A persistence image for a 1D barcode (equivalently, a 1D persistence diagram),
represented on a rectangular grid.

We use the common (birth, persistence) coordinate system by default, where each
interval (b, d) is mapped to the point (b, d-b).

The `values` matrix is indexed as `values[iy, ix]` with:
- `ix` indexing `xgrid` (birth),
- `iy` indexing `ygrid` (persistence or death, depending on `coords`).
"""
struct PersistenceImage1D
    xgrid::Vector{Float64}
    ygrid::Vector{Float64}
    values::Matrix{Float64}
end

function Base.show(io::IO, PI::PersistenceImage1D)
    print(io, "PersistenceImage1D(nx=", length(PI.xgrid), ", ny=", length(PI.ygrid), ")")
end

function Base.show(io::IO, ::MIME"text/plain", PI::PersistenceImage1D)
    println(io, "PersistenceImage1D")
    println(io, "  nx = ", length(PI.xgrid))
    println(io, "  ny = ", length(PI.ygrid))
    if !isempty(PI.xgrid)
        println(io, "  birth_range = [", PI.xgrid[1], ", ", PI.xgrid[end], "]")
    end
    if !isempty(PI.ygrid)
        println(io, "  y_range     = [", PI.ygrid[1], ", ", PI.ygrid[end], "]")
    end
end

# Build a Float64 grid from a (lo, hi) range and a requested number of points.
function _grid_from_range(rng, n::Int)::Vector{Float64}
    n >= 1 || error("_grid_from_range: n must be >= 1")
    lo, hi = rng
    lo = float(lo)
    hi = float(hi)
    if !(isfinite(lo) && isfinite(hi))
        error("_grid_from_range: range endpoints must be finite")
    end
    if hi < lo
        error("_grid_from_range: need lo <= hi")
    end
    if lo == hi
        # Expand a degenerate range so range(...) makes sense.
        lo -= 0.5
        hi += 0.5
    end
    return collect(range(lo, hi; length=n))
end

# Extract global bounds from a single barcode:
# returns (bmin, dmax, pmax, n_intervals).
function _barcode_bounds_birth_death(bar)
    bmin = Inf
    dmax = -Inf
    pmax = 0.0
    n = 0
    for (bd, mult) in bar
        mult <= 0 && continue
        b0, d0 = bd
        b = float(b0)
        d = float(d0)
        d > b || continue
        bmin = min(bmin, b)
        dmax = max(dmax, d)
        pmax = max(pmax, d - b)
        n += mult
    end
    if bmin == Inf
        # Empty barcode: choose a harmless default.
        bmin = 0.0
        dmax = 1.0
        pmax = 1.0
        n = 0
    end
    return bmin, dmax, pmax, n
end

# Weight assigned to a single interval.
# This is used for persistence images, silhouettes, and entropy.
function _interval_weight(weighting, b::Float64, d::Float64; p::Real=1)::Float64
    if weighting isa Function
        return float(weighting(b, d))
    end
    if weighting == :none
        return 1.0
    elseif weighting == :persistence
        pers = d - b
        pers < 0 && return 0.0
        return pers^float(p)
    else
        error("_interval_weight: unknown weighting $(weighting); supported: :none, :persistence, or a function")
    end
end

# Normalize a Float64 matrix in-place.
function _normalize_matrix!(A::AbstractMatrix{Float64}, mode::Symbol)
    mode == :none && return A
    if mode == :l1
        s = sum(abs.(A))
        s > 0 && (A ./= s)
    elseif mode == :l2
        s = norm(A)
        s > 0 && (A ./= s)
    elseif mode == :max
        m = maximum(A)
        m > 0 && (A ./= m)
    else
        error("_normalize_matrix!: normalize must be :none, :l1, :l2, or :max")
    end
    return A
end

"""
    persistence_image(bar; ... ) -> PersistenceImage1D

Compute a persistence image for a 1D barcode `bar` (a dictionary mapping
interval endpoints `(b, d)` to multiplicity).

This is a standard, ML-friendly vectorization of a persistence diagram: each
interval contributes a Gaussian "blob" centered at (birth, persistence) by
default.

Keywords:
- `xgrid`, `ygrid`: explicit grids (centers) for birth and persistence/death.
- `birth_range`, `pers_range`: ranges used if grids are not provided.
- `nbirth`, `npers`: grid sizes used if grids are not provided.
- `sigma`: Gaussian bandwidth (> 0).
- `coords`: `:birth_persistence` (default) or `:birth_death`.
- `weighting`: `:persistence` (default), `:none`, or a function `(b,d)->w`.
- `p`: exponent used when `weighting=:persistence`.
- `normalize`: `:none` (default), `:l1`, `:l2`, or `:max`.

Notes:
- The discretization is by evaluation at grid centers (not exact pixel integrals).
- Empty barcodes yield an all-zero image.
"""
function persistence_image(bar;
                           xgrid=0:0.1:1,
                           ygrid=0:0.1:1,
                           sigma=0.1,
                           coords=:birth_persistence,
                           weighting=:persistence,
                           p=1,
                           normalize=:none,
                           differentiable::Bool=false)

    xg = collect(xgrid)
    yg = collect(ygrid)
    inv2sig2 = 1.0 / (2*sigma^2)

    if !differentiable
        # Fast in-place version (existing behavior)
        img = zeros(Float64, length(yg), length(xg))
        for (bd, mult) in bar
            b = bd[1]
            d = bd[2]

            x, y = coords == :birth_persistence ? (b, d-b) :
                   coords == :birth_death ? (b, d) :
                   coords == :midlife_persistence ? (0.5*(b+d), d-b) :
                   error("unknown coords=$coords")

            w = mult * _interval_weight(weighting, b, d; p=p)

            for ix in eachindex(xg)
                dx2 = (x - xg[ix])^2
                for iy in eachindex(yg)
                    dy2 = (y - yg[iy])^2
                    img[iy,ix] += w * exp(-(dx2+dy2)*inv2sig2)
                end
            end
        end

        _normalize_matrix!(img, normalize)
        return PersistenceImage1D(xg, yg, img)
    end

    # Pure/non-mutating version for AD friendliness.
    terms = [begin
                b = bd[1]
                d = bd[2]
                x, y = coords == :birth_persistence ? (b, d-b) :
                       coords == :birth_death ? (b, d) :
                       coords == :midlife_persistence ? (0.5*(b+d), d-b) :
                       error("unknown coords=$coords")
                w = mult * _interval_weight(weighting, b, d; p=p)
                (x, y, w)
            end for (bd, mult) in bar]

    img = [sum(t[3] * exp(-(((t[1]-xg[ix])^2 + (t[2]-yg[iy])^2))*inv2sig2) for t in terms)
           for iy in eachindex(yg), ix in eachindex(xg)]

    # Non-mutating normalization
    img = normalize == :none ? img :
          normalize == :l1  ? (sum(abs, img) > 0 ? img ./ sum(abs, img) : img) :
          normalize == :l2  ? (sqrt(sum(abs2, img)) > 0 ? img ./ sqrt(sum(abs2, img)) : img) :
          normalize == :max ? (maximum(abs, img) > 0 ? img ./ maximum(abs, img) : img) :
          error("unknown normalize=$normalize")

    return PersistenceImage1D(xg, yg, img)
end

"""
    feature_map(x; kind=:persistence_image, flatten=true, differentiable=false, kwargs...)

Compute a fixed-size numeric representation suitable for ML.

Currently supported:
- `x` is a barcode dictionary and `kind=:persistence_image`
- `x` is a `PersistenceImage1D`

If `flatten=true`, returns a vector in column-major order.
If `differentiable=true` and kind supports it, avoids mutation (Zygote-friendly).
"""
function feature_map(x;
                     kind::Symbol = :persistence_image,
                     flatten::Bool = true,
                     differentiable::Bool = false,
                     kwargs...)
    mat = if x isa PersistenceImage1D
        x.values
    elseif kind == :persistence_image
        persistence_image(x; differentiable=differentiable, kwargs...).values
    else
        error("unsupported feature_map kind=$kind for input $(typeof(x))")
    end
    return flatten ? vec(mat) : mat
end

"""
    feature_vector(x; kwargs...)

Alias for `feature_map(x; flatten=true, ...)`.
"""
feature_vector(x; kwargs...) = feature_map(x; flatten=true, kwargs...)


"""
    persistence_silhouette(bar; tgrid, weighting=:persistence, p=1, normalize=true)

Compute a persistence silhouette for a 1D barcode.

A silhouette is the weighted average (by default, persistence weights) of the
tent functions associated to each interval. If `normalize=true`, the output is
divided by the total weight, so the result is scale-stable.

This is a standard 1D functional summary, distinct from landscapes:
- landscapes take successive maxima of tents (layered max),
- silhouettes average tents (weighted sum / average).
"""
function persistence_silhouette(
    bar;
    tgrid,
    weighting=:persistence,
    p::Real=1,
    normalize::Bool=true
)::Vector{Float64}
    tg = _clean_tgrid(tgrid)
    out = zeros(Float64, length(tg))
    denom = 0.0

    for (bd, mult) in bar
        mult <= 0 && continue
        b0, d0 = bd
        b = float(b0)
        d = float(d0)
        d > b || continue

        w = float(mult) * _interval_weight(weighting, b, d; p=p)
        w == 0.0 && continue

        denom += w
        for it in eachindex(tg)
            out[it] += w * _tent_value(b, d, tg[it])
        end
    end

    if normalize && denom > 0.0
        out ./= denom
    end
    return out
end

"""
    barcode_entropy(bar; normalize=true, base=exp(1), weighting=:persistence, p=1) -> Float64

Compute "persistent entropy" for a barcode.

We interpret the barcode as a multiset of intervals. Each interval gets a
nonnegative weight `w_i` (by default, persistence^p). We normalize to a
probability distribution and return Shannon entropy.

- `normalize=true` divides by `log(n_intervals)` (common in the literature),
  yielding values in [0, 1] when there are at least 2 intervals.
- `base` controls the logarithm base (default: natural).
"""
function barcode_entropy(
    bar;
    normalize::Bool=true,
    base::Real=exp(1),
    weighting=:persistence,
    p::Real=1
)::Float64
    base = float(base)
    base > 0 || error("barcode_entropy: base must be > 0")

    tot = 0.0
    n = 0
    ws = Vector{Tuple{Float64, Int}}()

    for (bd, mult) in bar
        mult <= 0 && continue
        b0, d0 = bd
        b = float(b0)
        d = float(d0)
        d > b || continue

        w = _interval_weight(weighting, b, d; p=p)
        w = max(w, 0.0)
        w == 0.0 && continue

        push!(ws, (w, mult))
        tot += w * float(mult)
        n += mult
    end

    (tot > 0.0 && n > 0) || return 0.0

    invlogbase = 1.0 / log(base)
    H = 0.0
    for (w, mult) in ws
        p_i = w / tot
        H -= float(mult) * p_i * (log(p_i) * invlogbase)
    end

    if normalize && n > 1
        H /= (log(float(n)) * invlogbase)
    end
    return H
end

"""
    barcode_summary(bar; normalize_entropy=true) -> NamedTuple

Summary statistics for a 1D barcode. Intended for quick diagnostics and
lightweight ML features.

Returns a NamedTuple with fields:
- `n_intervals`
- `total_persistence`
- `max_persistence`
- `mean_persistence`
- `l2_persistence`
- `entropy`
"""
function barcode_summary(bar; normalize_entropy::Bool=true)
    n = 0
    total = 0.0
    maxp = 0.0
    sumsq = 0.0

    for (bd, mult) in bar
        mult <= 0 && continue
        b0, d0 = bd
        b = float(b0)
        d = float(d0)
        d > b || continue

        pers = d - b
        n += mult
        total += float(mult) * pers
        maxp = max(maxp, pers)
        sumsq += float(mult) * pers * pers
    end

    meanp = (n > 0) ? (total / float(n)) : 0.0
    l2p = sqrt(sumsq)
    ent = barcode_entropy(bar; normalize=normalize_entropy)

    return (n_intervals=n,
            total_persistence=total,
            max_persistence=maxp,
            mean_persistence=meanp,
            l2_persistence=l2p,
            entropy=ent)
end

const _DEFAULT_BARCODE_SUMMARY_FIELDS =
    (:n_intervals, :total_persistence, :max_persistence, :mean_persistence, :l2_persistence, :entropy)

function _barcode_summary_vector(bar; fields=_DEFAULT_BARCODE_SUMMARY_FIELDS, normalize_entropy::Bool=true)
    nt = barcode_summary(bar; normalize_entropy=normalize_entropy)
    return Float64[float(getproperty(nt, f)) for f in fields]
end

# Flatten a PersistenceLandscape1D to a single feature vector.
# Convention: k-major ordering, i.e.
#   [L_1(t1), ..., L_1(tN), L_2(t1), ..., L_2(tN), ...]
function _landscape_feature_vector(pl::PersistenceLandscape1D)::Vector{Float64}
    kmax, nt = size(pl.values)
    out = Vector{Float64}(undef, kmax * nt)
    idx = 1
    for k in 1:kmax
        for it in 1:nt
            out[idx] = pl.values[k, it]
            idx += 1
        end
    end
    return out
end

# Flatten a PersistenceImage1D to a single feature vector.
# Convention: y-major ordering, with x varying fastest:
#   [row1(x1..xN), row2(x1..xN), ...]
function _image_feature_vector(PI::PersistenceImage1D)::Vector{Float64}
    ny, nx = size(PI.values)
    out = Vector{Float64}(undef, nx * ny)
    idx = 1
    for iy in 1:ny
        for ix in 1:nx
            out[idx] = PI.values[iy, ix]
            idx += 1
        end
    end
    return out
end

# Parse slice specs (same conventions as mp_landscape):
# - chain::Vector{Int}
# - NamedTuple (chain=..., values=..., weight=...)
# - Tuple (chain, values, weight)
function _parse_slice_spec(spec; default_weight::Real=1.0)
    chain = nothing
    values = nothing
    w = float(default_weight)

    if spec isa AbstractVector{Int}
        chain = spec
    elseif spec isa NamedTuple
        chain = spec.chain
        values = get(spec, :values, nothing)
        w = float(get(spec, :weight, default_weight))
    elseif spec isa Tuple
        length(spec) >= 1 || error("_parse_slice_spec: tuple slice spec must have at least a chain")
        chain = spec[1]
        values = (length(spec) >= 2) ? spec[2] : nothing
        w = (length(spec) >= 3) ? float(spec[3]) : float(default_weight)
    else
        error("_parse_slice_spec: unsupported slice spec type $(typeof(spec))")
    end

    chain isa AbstractVector{Int} || error("_parse_slice_spec: chain must be a vector of Int")
    return chain, values, w
end

"""
    slice_barcodes(M, slices; default_weight=1.0, normalize_weights=true) -> NamedTuple

Compute the 1D slice barcodes of a (multi-parameter) module `M` for an explicit
collection of slice specs.

Each slice spec can be:
- a chain `Vector{Int}`,
- a NamedTuple `(chain=..., values=..., weight=...)`,
- a Tuple `(chain, values, weight)`.

Returns `(barcodes, weights, slices)` where:
- `barcodes[i]` is the barcode dict for the i-th slice,
- `weights[i]` is its (optionally normalized) weight.
"""
function slice_barcodes(
    M::PModule{QQ},
    slices::AbstractVector;
    weight = uniform2d,
    weight_fn = slice -> weight((slice.dir[1], slice.dir[2])),
    values = nothing,
    threads::Bool = (Threads.nthreads() > 1),
)
    ns = length(slices)
    bcs = Vector{Any}(undef, ns)
    W = Vector{Float64}(undef, ns)

    if threads && Threads.nthreads() > 1
        Threads.@threads for i in 1:ns
            slice = _parse_slice_spec(slices[i]; weight_fn = weight_fn)
            ch = slice.chain
            vals = values === nothing ? slice.values : values
            bcs[i] = slice_barcode(M, ch; values = vals, check_chain = false)
            W[i] = slice.weight
        end
        return (barcodes = bcs, weights = W)
    end

    for i in 1:ns
        slice = _parse_slice_spec(slices[i]; weight_fn = weight_fn)
        ch = slice.chain
        vals = values === nothing ? slice.values : values
        bcs[i] = slice_barcode(M, ch; values = vals, check_chain = false)
        W[i] = slice.weight
    end

    return (barcodes = bcs, weights = W)
end


slice_barcodes(M::PModule{QQ}, chain::AbstractVector{Int}; kwargs...) =
    slice_barcodes(M, [chain]; kwargs...)

"""
    slice_barcodes(M, pi; directions, offsets, ...) -> NamedTuple

Compute slice barcodes via a finite encoding map `pi` (Rn or Zn style).

This mirrors the "geometric slicing" API of `mp_landscape`:

- `directions`: list of direction vectors.
- `offsets`: list of basepoints (one per offset).
- For Rn/PL encodings: provide `tmin, tmax, nsteps`.
- For Zn encodings: provide `kmin, kmax_param`.

Weights:
- `direction_weight` applies `_direction_weight` (e.g. :lesnick_l1).
- `offset_weights` can be:
  * `nothing` (uniform), or
  * a vector of length `length(offsets)`, or
  * a function `x0 -> weight`.

Returns `(barcodes, weights, directions, offsets)` where:
- `barcodes[i,j]` is the barcode for direction i and offset j,
- `weights[i,j]` is the corresponding slice weight.
"""
function slice_barcodes(
    M::PModule{QQ},
    pi::EncodingMap;
    directions = :auto,
    offsets = :auto,
    weight = uniform2d,
    weight_fn = (d, o) -> weight((d[1], d[2])) * weight((o[1], o[2])),
    values = nothing,
    threads::Bool = (Threads.nthreads() > 1),
    slice_kwargs...
)
    # Get default directions/offsets if needed.
    dirs, offs = get_slices(directions, offsets)

    nd = length(dirs)
    no = length(offs)

    bcs = Matrix{Dict{Tuple{Float64, Float64}, Int}}(undef, nd, no)
    W = zeros(Float64, nd, no)

    for i in 1:nd
        for j in 1:no
            W[i, j] = weight_fn(dirs[i], offs[j])
        end
    end

    if threads && Threads.nthreads() > 1
        Threads.@threads for k in 1:(nd * no)
            i = div((k - 1), no) + 1
            j = (k - 1) % no + 1
            # Slice barcode is independent per (dir, off).
            chain, tvals = slice_chain(pi, dirs[i], offs[j]; slice_kwargs...)
            vals_use = values === nothing ? tvals : values
            bcs[i, j] = slice_barcode(M, chain; values = vals_use, check_chain = false)
        end
        return (barcodes = bcs, weights = W, dirs = dirs, offs = offs)
    end

    for i in 1:nd, j in 1:no
        chain, tvals = slice_chain(pi, dirs[i], offs[j]; slice_kwargs...)
        vals_use = values === nothing ? tvals : values
        bcs[i, j] = slice_barcode(M, chain; values = vals_use, check_chain = false)
    end

    return (barcodes = bcs, weights = W, dirs = dirs, offs = offs)
end


# Bounds from a collection of barcodes.
function _barcode_bounds_collection(bcs)
    bmin = Inf
    dmax = -Inf
    pmax = 0.0
    for bc in bcs
        for (bd, mult) in bc
            mult <= 0 && continue
            b0, d0 = bd
            b = float(b0)
            d = float(d0)
            d > b || continue
            bmin = min(bmin, b)
            dmax = max(dmax, d)
            pmax = max(pmax, d - b)
        end
    end
    if bmin == Inf
        bmin = 0.0
        dmax = 1.0
        pmax = 1.0
    end
    return bmin, dmax, pmax
end

# Convert a user featurizer output to a numeric vector.
function _as_feature_vector(x)::Vector{Float64}
    if x isa Real
        return Float64[float(x)]
    elseif x isa AbstractVector
        return Float64[float(v) for v in x]
    else
        error("featurizer must return a real or a vector of reals; got $(typeof(x))")
    end
end

# Default t-grid (global, so features align across slices).
function _default_tgrid_from_barcodes(bcs; nsteps::Int=401)::Vector{Float64}
    bmin, dmax, _ = _barcode_bounds_collection(bcs)
    nsteps < 2 && (nsteps = 2)
    if bmin == dmax
        bmin -= 0.5
        dmax += 0.5
    end
    return collect(range(bmin, dmax; length=nsteps))
end

# Default image grids (global, so features align across slices).
function _default_image_grids_from_barcodes(
    bcs;
    img_xgrid=nothing,
    img_ygrid=nothing,
    img_birth_range=nothing,
    img_pers_range=nothing,
    img_nbirth::Int=20,
    img_npers::Int=20
)
    bmin, dmax, pmax = _barcode_bounds_collection(bcs)

    xg = if img_xgrid === nothing
        br = (img_birth_range === nothing) ? (bmin, dmax) : img_birth_range
        _grid_from_range(br, img_nbirth)
    else
        Float64[float(x) for x in collect(img_xgrid)]
    end

    yg = if img_ygrid === nothing
        pr = (img_pers_range === nothing) ? (0.0, pmax) : img_pers_range
        _grid_from_range(pr, img_npers)
    else
        Float64[float(y) for y in collect(img_ygrid)]
    end

    return xg, yg
end

# Aggregate per-slice feature vectors (explicit slices).
function _aggregate_feature_vectors(
    feats::AbstractVector{<:AbstractVector{Float64}},
    W::AbstractVector{Float64};
    aggregate::Symbol=:mean,
    unwrap_scalar::Bool=true
)
    ns = length(feats)
    ns > 0 || error("_aggregate_feature_vectors: empty feature list")
    d = length(feats[1])

    # Shape check
    for i in 2:ns
        length(feats[i]) == d || error("_aggregate_feature_vectors: feature length mismatch")
    end

    if aggregate == :stack
        A = zeros(Float64, ns, d)
        for i in 1:ns
            for j in 1:d
                A[i, j] = feats[i][j]
            end
        end
        if unwrap_scalar && d == 1
            return vec(A)
        end
        return A
    elseif aggregate == :sum || aggregate == :mean
        acc = zeros(Float64, d)
        sumw = sum(W)
        for i in 1:ns
            w = W[i]
            w == 0.0 && continue
            for j in 1:d
                acc[j] += w * feats[i][j]
            end
        end
        if aggregate == :mean
            sumw > 0.0 || error("slice_features: total weight is zero")
            acc ./= sumw
        end
        if unwrap_scalar && d == 1
            return acc[1]
        end
        return acc
    else
        error("slice_features: aggregate must be :mean, :sum, or :stack")
    end
end

# Aggregate per-slice feature vectors (geometric slices, 2D array of slices).
function _aggregate_feature_vectors(
    feats::Array{<:AbstractVector{Float64}, 2},
    W::AbstractMatrix{Float64};
    aggregate::Symbol=:mean,
    unwrap_scalar::Bool=true
)
    nd, no = size(feats)
    d = length(feats[1, 1])

    # Shape check
    for i in 1:nd, j in 1:no
        length(feats[i, j]) == d || error("_aggregate_feature_vectors: feature length mismatch")
    end

    if aggregate == :stack
        A = zeros(Float64, nd, no, d)
        for i in 1:nd, j in 1:no
            for k in 1:d
                A[i, j, k] = feats[i, j][k]
            end
        end
        if unwrap_scalar && d == 1
            B = zeros(Float64, nd, no)
            for i in 1:nd, j in 1:no
                B[i, j] = A[i, j, 1]
            end
            return B
        end
        return A
    elseif aggregate == :sum || aggregate == :mean
        acc = zeros(Float64, d)
        sumw = sum(W)
        for i in 1:nd, j in 1:no
            w = W[i, j]
            w == 0.0 && continue
            for k in 1:d
                acc[k] += w * feats[i, j][k]
            end
        end
        if aggregate == :mean
            sumw > 0.0 || error("slice_features: total weight is zero")
            acc ./= sumw
        end
        if unwrap_scalar && d == 1
            return acc[1]
        end
        return acc
    else
        error("slice_features: aggregate must be :mean, :sum, or :stack")
    end
end

"""
    slice_features(M, slices; featurizer=:landscape, aggregate=:mean, ...) -> features
    slice_features(M, pi; directions, offsets, featurizer=:landscape, aggregate=:mean, ...) -> features

Compute multiparameter "slice features" by:
1) slicing a multi-parameter module into many 1D modules (fibers/lines),
2) computing a 1D featurization per slice,
3) aggregating across slices.

This supports the standard ML pipeline described in the multiparameter
literature (slice barcodes -> aggregate 1D vectorizations).

Supported `featurizer` values:
- `:landscape`  -> flatten persistence landscapes on a common `tgrid`
- `:image`      -> flatten persistence images on common grids
- `:silhouette` -> persistence silhouette on a common `tgrid`
- `:entropy`    -> persistent entropy (scalar)
- `:summary`    -> a small vector of summary stats
- or any function `bc -> real_or_vector`

Aggregation modes (`aggregate`):
- `:mean`  -> weighted average
- `:sum`   -> weighted sum
- `:stack` -> return per-slice features (matrix for explicit slices, 3D array for geometric slices)

Key options:
- `normalize_weights`: normalize slice weights (default true)
- `unwrap_scalar`: return scalar when the feature dimension is 1 (default true)
"""
function slice_features(
    M::PModule{QQ},
    slices::AbstractVector;
    featurizer=:landscape,
    aggregate::Symbol=:mean,
    default_weight::Real=1.0,
    normalize_weights::Bool=true,
    unwrap_scalar::Bool=true,
    # Landscape / silhouette defaults:
    kmax::Int=5,
    tgrid=nothing,
    tgrid_nsteps::Int=401,
    # Silhouette options:
    sil_weighting=:persistence,
    sil_p::Real=1,
    sil_normalize::Bool=true,
    # Image options:
    img_xgrid=nothing,
    img_ygrid=nothing,
    img_birth_range=nothing,
    img_pers_range=nothing,
    img_nbirth::Int=20,
    img_npers::Int=20,
    img_sigma::Real=1.0,
    img_coords::Symbol=:birth_persistence,
    img_weighting=:persistence,
    img_p::Real=1,
    img_normalize::Symbol=:none,
    # Entropy options:
    entropy_normalize::Bool=true,
    entropy_weighting=:persistence,
    entropy_p::Real=1,
    # Summary options:
    summary_fields=_DEFAULT_BARCODE_SUMMARY_FIELDS,
    summary_normalize_entropy::Bool=true
)
    data = slice_barcodes(M, slices; default_weight=default_weight, normalize_weights=normalize_weights)
    bcs = data.barcodes
    W = data.weights

    # Global grids if needed
    tg = tgrid
    if (featurizer == :landscape || featurizer == :silhouette) && tg === nothing
        tg = _default_tgrid_from_barcodes(bcs; nsteps=tgrid_nsteps)
    end
    if tg !== nothing
        tg = _clean_tgrid(tg)
    end

    xg = nothing
    yg = nothing
    if featurizer == :image
        xg, yg = _default_image_grids_from_barcodes(
            bcs;
            img_xgrid=img_xgrid,
            img_ygrid=img_ygrid,
            img_birth_range=img_birth_range,
            img_pers_range=img_pers_range,
            img_nbirth=img_nbirth,
            img_npers=img_npers
        )
    end

    feats = Vector{Vector{Float64}}(undef, length(bcs))
    for i in 1:length(bcs)
        bc = bcs[i]
        if featurizer == :landscape
            pl = persistence_landscape(bc; kmax=kmax, tgrid=tg)
            feats[i] = _landscape_feature_vector(pl)
        elseif featurizer == :silhouette
            s = persistence_silhouette(bc; tgrid=tg, weighting=sil_weighting, p=sil_p, normalize=sil_normalize)
            feats[i] = Float64[s...]
        elseif featurizer == :image
            PI = persistence_image(bc;
                                   xgrid=xg,
                                   ygrid=yg,
                                   sigma=img_sigma,
                                   coords=img_coords,
                                   weighting=img_weighting,
                                   p=img_p,
                                   normalize=img_normalize)
            feats[i] = _image_feature_vector(PI)
        elseif featurizer == :entropy
            e = barcode_entropy(bc; normalize=entropy_normalize, weighting=entropy_weighting, p=entropy_p)
            feats[i] = Float64[float(e)]
        elseif featurizer == :summary
            feats[i] = _barcode_summary_vector(bc; fields=summary_fields, normalize_entropy=summary_normalize_entropy)
        elseif featurizer isa Function
            feats[i] = _as_feature_vector(featurizer(bc))
        else
            error("slice_features: unknown featurizer $(featurizer)")
        end
    end

    return _aggregate_feature_vectors(feats, W; aggregate=aggregate, unwrap_scalar=unwrap_scalar)
end

slice_features(M::PModule{QQ}, chain::AbstractVector{Int}; kwargs...) =
    slice_features(M, [chain]; kwargs...)

function slice_features(
    M::PModule{QQ},
    pi;
    directions=nothing,
    offsets=nothing,
    featurizer=:landscape,
    aggregate::Symbol=:mean,
    normalize_weights::Bool=true,
    unwrap_scalar::Bool=true,
    # Geometric slice parameters:
    tmin::Union{Real,Nothing}=nothing,
    tmax::Union{Real,Nothing}=nothing,
    nsteps::Int=401,
    kmin=nothing,
    kmax_param=nothing,
    strict::Bool=true,
    drop_unknown::Bool=true,
    dedup::Bool=true,
    normalize_dirs::Symbol=:none,
    direction_weight::Symbol=:none,
    offset_weights=nothing,
    # Landscape / silhouette defaults:
    kmax::Int=5,
    tgrid=nothing,
    tgrid_nsteps::Int=401,
    # Silhouette options:
    sil_weighting=:persistence,
    sil_p::Real=1,
    sil_normalize::Bool=true,
    # Image options:
    img_xgrid=nothing,
    img_ygrid=nothing,
    img_birth_range=nothing,
    img_pers_range=nothing,
    img_nbirth::Int=20,
    img_npers::Int=20,
    img_sigma::Real=1.0,
    img_coords::Symbol=:birth_persistence,
    img_weighting=:persistence,
    img_p::Real=1,
    img_normalize::Symbol=:none,
    # Entropy options:
    entropy_normalize::Bool=true,
    entropy_weighting=:persistence,
    entropy_p::Real=1,
    # Summary options:
    summary_fields=_DEFAULT_BARCODE_SUMMARY_FIELDS,
    summary_normalize_entropy::Bool=true,
    slice_kwargs...
)
    data = slice_barcodes(M, pi;
                          directions=directions,
                          
                          offsets=offsets,
                          tmin=tmin,
                          tmax=tmax,
                          nsteps=nsteps,
                          kmin=kmin,
                          kmax_param=kmax_param,
                          strict=strict,
                          drop_unknown=drop_unknown,
                          dedup=dedup,
                          normalize_dirs=normalize_dirs,
                          direction_weight=direction_weight,
                          offset_weights=offset_weights,
                          normalize_weights=normalize_weights,
                          slice_kwargs...)

    bcs = data.barcodes
    W = data.weights

    # Global grids if needed
    tg = tgrid
    if (featurizer == :landscape || featurizer == :silhouette) && tg === nothing
        tg = _default_tgrid_from_barcodes(bcs; nsteps=tgrid_nsteps)
    end
    if tg !== nothing
        tg = _clean_tgrid(tg)
    end

    xg = nothing
    yg = nothing
    if featurizer == :image
        xg, yg = _default_image_grids_from_barcodes(
            bcs;
            img_xgrid=img_xgrid,
            img_ygrid=img_ygrid,
            img_birth_range=img_birth_range,
            img_pers_range=img_pers_range,
            img_nbirth=img_nbirth,
            img_npers=img_npers
        )
    end

    nd, no = size(bcs)
    feats = Array{Vector{Float64}}(undef, nd, no)
    for i in 1:nd, j in 1:no
        bc = bcs[i, j]
        if featurizer == :landscape
            pl = persistence_landscape(bc; kmax=kmax, tgrid=tg)
            feats[i, j] = _landscape_feature_vector(pl)
        elseif featurizer == :silhouette
            s = persistence_silhouette(bc; tgrid=tg, weighting=sil_weighting, p=sil_p, normalize=sil_normalize)
            feats[i, j] = Float64[s...]
        elseif featurizer == :image
            PI = persistence_image(bc;
                                   xgrid=xg,
                                   ygrid=yg,
                                   sigma=img_sigma,
                                   coords=img_coords,
                                   weighting=img_weighting,
                                   p=img_p,
                                   normalize=img_normalize)
            feats[i, j] = _image_feature_vector(PI)
        elseif featurizer == :entropy
            e = barcode_entropy(bc; normalize=entropy_normalize, weighting=entropy_weighting, p=entropy_p)
            feats[i, j] = Float64[float(e)]
        elseif featurizer == :summary
            feats[i, j] = _barcode_summary_vector(bc; fields=summary_fields, normalize_entropy=summary_normalize_entropy)
        elseif featurizer isa Function
            feats[i, j] = _as_feature_vector(featurizer(bc))
        else
            error("slice_features: unknown featurizer $(featurizer)")
        end
    end

    return _aggregate_feature_vectors(feats, W; aggregate=aggregate, unwrap_scalar=unwrap_scalar)
end

# Kernel between 1D barcodes (used by slice_kernel).
function _barcode_kernel(
    bA,
    bB;
    kind=:bottleneck_gaussian,
    sigma::Real=1.0,
    gamma=nothing,

    # Wasserstein kernel parameters:
    p::Real=2,
    q::Real=Inf,

    # Landscape kernel parameters:
    tgrid=nothing,
    kmax::Int=5
)::Float64
    if kind isa Function
        return float(kind(bA, bB))
    end

    if kind == :bottleneck_gaussian
        d = bottleneck_distance(bA, bB)
        g = (gamma === nothing) ? (1.0 / (2.0 * float(sigma)^2)) : float(gamma)
        float(sigma) > 0 || error("_barcode_kernel: sigma must be > 0 for gaussian kernels")
        return exp(-g * d * d)
    elseif kind == :bottleneck_laplacian
        d = bottleneck_distance(bA, bB)
        float(sigma) > 0 || error("_barcode_kernel: sigma must be > 0 for laplacian kernels")
        return exp(-d / float(sigma))
    elseif kind == :landscape_gaussian
        tgrid === nothing && error("_barcode_kernel: tgrid required for landscape kernels")
        plA = persistence_landscape(bA; kmax=kmax, tgrid=tgrid)
        plB = persistence_landscape(bB; kmax=kmax, tgrid=tgrid)
        vA = _landscape_feature_vector(plA)
        vB = _landscape_feature_vector(plB)
        d = norm(vA - vB)
        g = (gamma === nothing) ? (1.0 / (2.0 * float(sigma)^2)) : float(gamma)
        float(sigma) > 0 || error("_barcode_kernel: sigma must be > 0 for gaussian kernels")
        return exp(-g * d * d)
    elseif kind == :landscape_laplacian
        tgrid === nothing && error("_barcode_kernel: tgrid required for landscape kernels")
        plA = persistence_landscape(bA; kmax=kmax, tgrid=tgrid)
        plB = persistence_landscape(bB; kmax=kmax, tgrid=tgrid)
        vA = _landscape_feature_vector(plA)
        vB = _landscape_feature_vector(plB)
        d = norm(vA - vB)
        float(sigma) > 0 || error("_barcode_kernel: sigma must be > 0 for laplacian kernels")
        return exp(-d / float(sigma))
    elseif kind == :landscape_linear
        tgrid === nothing && error("_barcode_kernel: tgrid required for landscape kernels")
        plA = persistence_landscape(bA; kmax=kmax, tgrid=tgrid)
        plB = persistence_landscape(bB; kmax=kmax, tgrid=tgrid)
        vA = _landscape_feature_vector(plA)
        vB = _landscape_feature_vector(plB)
        return dot(vA, vB)
    elseif kind == :wasserstein_gaussian
        sigma > 0 || error("_barcode_kernel: sigma must be > 0")
        d = wasserstein_distance(bA, bB; p=p, q=q)
        return exp(-(d * d) / (2 * float(sigma)^2))
    elseif kind == :wasserstein_laplacian
        sigma > 0 || error("_barcode_kernel: sigma must be > 0")
        d = wasserstein_distance(bA, bB; p=p, q=q)
        return exp(-d / float(sigma))
    else
        error("_barcode_kernel: unknown kind $(kind)")
    end
end

"""
    slice_kernel(M, N, slices; kind=:bottleneck_gaussian, ...) -> Float64
    slice_kernel(M, N, pi; directions, offsets, kind=:bottleneck_gaussian, ...) -> Float64

Compute a sliced kernel by:
- restricting `M` and `N` to each slice,
- computing a 1D kernel per slice,
- averaging (weighted) across slices.

This matches the "sliced-kernel" pattern common in the multiparameter
literature (integrate 1-parameter kernels over lines).

Supported `kind` values:
- `:bottleneck_gaussian`  (Gaussian kernel on bottleneck distance)
- `:bottleneck_laplacian` (Laplacian kernel on bottleneck distance)
- `:landscape_gaussian`   (Gaussian kernel on L2 distance between landscape vectors)
- `:landscape_laplacian`
- `:landscape_linear`     (linear kernel on landscape vectors)
- `:wasserstein_gaussian` (Gaussian kernel on p-Wasserstein distance)
- `:wasserstein_laplacian`
- or any function `(bcM, bcN) -> Float64`
"""
function slice_kernel(
    M::PModule{QQ},
    N::PModule{QQ},
    slices::AbstractVector;
    kind=:bottleneck_gaussian,
    sigma::Real=1.0,
    gamma=nothing,
    default_weight::Real=1.0,
    normalize_weights::Bool=true,
    # Landscape kernel parameters:
    tgrid=nothing,
    tgrid_nsteps::Int=401,
    kmax::Int=5
)::Float64
    dataM = slice_barcodes(M, slices; default_weight=default_weight, normalize_weights=normalize_weights)
    dataN = slice_barcodes(N, slices; default_weight=default_weight, normalize_weights=normalize_weights)

    bM = dataM.barcodes
    bN = dataN.barcodes
    W = dataM.weights

    length(bM) == length(bN) || error("slice_kernel: slice list length mismatch")

    tg = tgrid
    if (kind == :landscape_gaussian || kind == :landscape_laplacian || kind == :landscape_linear) && tg === nothing
        # Choose a common global grid based on both families of barcodes.
        tg = _default_tgrid_from_barcodes(vcat(bM, bN); nsteps=tgrid_nsteps)
    end
    if tg !== nothing
        tg = _clean_tgrid(tg)
    end

    sumw = sum(W)
    sumw > 0.0 || error("slice_kernel: total weight is zero")

    acc = 0.0
    for i in 1:length(bM)
        w = W[i]
        w == 0.0 && continue
        acc += w * _barcode_kernel(bM[i], bN[i]; kind=kind, sigma=sigma, gamma=gamma, tgrid=tg, kmax=kmax)
    end

    # Always return the weighted average (not the raw weighted sum).
    return acc / sumw
end

slice_kernel(M::PModule{QQ}, N::PModule{QQ}, chain::AbstractVector{Int}; kwargs...) =
    slice_kernel(M, N, [chain]; kwargs...)

function slice_kernel(
    M::PModule{QQ},
    N::PModule{QQ},
    pi;
    directions=nothing,
    offsets=nothing,
    kind=:bottleneck_gaussian,
    sigma::Real=1.0,
    gamma=nothing,
    normalize_weights::Bool=true,
    # Geometric slice parameters:
    tmin::Union{Real,Nothing}=nothing,
    tmax::Union{Real,Nothing}=nothing,
    nsteps::Int=401,
    kmin=nothing,
    kmax_param=nothing,
    strict::Bool=true,
    drop_unknown::Bool=true,
    dedup::Bool=true,
    normalize_dirs::Symbol=:none,
    direction_weight::Symbol=:none,
    offset_weights=nothing,
    # Landscape kernel parameters:
    tgrid=nothing,
    tgrid_nsteps::Int=401,
    kmax::Int=5,
    slice_kwargs...
)::Float64
    dataM = slice_barcodes(M, pi;
                           directions=directions,
                           
                           offsets=offsets,
                           tmin=tmin,
                           tmax=tmax,
                           nsteps=nsteps,
                           kmin=kmin,
                           kmax_param=kmax_param,
                           strict=strict,
                           drop_unknown=drop_unknown,
                           dedup=dedup,
                           normalize_dirs=normalize_dirs,
                           direction_weight=direction_weight,
                           offset_weights=offset_weights,
                           normalize_weights=normalize_weights,
                           slice_kwargs...)

    dataN = slice_barcodes(N, pi;
                           directions=directions,
                           
                           offsets=offsets,
                           tmin=tmin,
                           tmax=tmax,
                           nsteps=nsteps,
                           kmin=kmin,
                           kmax_param=kmax_param,
                           strict=strict,
                           drop_unknown=drop_unknown,
                           dedup=dedup,
                           normalize_dirs=normalize_dirs,
                           direction_weight=direction_weight,
                           offset_weights=offset_weights,
                           normalize_weights=normalize_weights,
                           slice_kwargs...)

    bM = dataM.barcodes
    bN = dataN.barcodes
    W = dataM.weights

    size(bM) == size(bN) || error("slice_kernel: barcode grid shape mismatch")

    tg = tgrid
    if (kind == :landscape_gaussian || kind == :landscape_laplacian || kind == :landscape_linear) && tg === nothing
        tg = _default_tgrid_from_barcodes(vcat(vec(bM), vec(bN)); nsteps=tgrid_nsteps)
    end
    if tg !== nothing
        tg = _clean_tgrid(tg)
    end

    sumw = sum(W)
    sumw > 0.0 || error("slice_kernel: total weight is zero")

    acc = 0.0
    nd, no = size(bM)
    for i in 1:nd, j in 1:no
        w = W[i, j]
        w == 0.0 && continue
        acc += w * _barcode_kernel(bM[i, j], bN[i, j]; kind=kind, sigma=sigma, gamma=gamma, tgrid=tg, kmax=kmax)
    end

    return acc / sumw
end

# Convenience wrappers: allow working directly with presentations.
slice_barcodes(H::FringeModule{QQ}, args...; kwargs...) = slice_barcodes(pmodule_from_fringe(H), args...; kwargs...)
slice_features(H::FringeModule{QQ}, args...; kwargs...) = slice_features(pmodule_from_fringe(H), args...; kwargs...)
slice_kernel(H::FringeModule{QQ}, K::FringeModule{QQ}, args...; kwargs...) =
    slice_kernel(pmodule_from_fringe(H), pmodule_from_fringe(K), args...; kwargs...)


# ----- Betti/Bass tables for indicator resolutions ------------------------------

# Helper for principal upset summands: unique minimal element.
function _unique_minimal_vertex(U::Upset)::Int
    P = U.P
    mins = Int[]
    for q in 1:P.n
        U.mask[q] || continue
        # q is minimal in U if there is no p < q also in U.
        ismin = true
        for p in 1:P.n
            (p == q) && continue
            if U.mask[p] && P.leq[p, q]
                ismin = false
                break
            end
        end
        ismin && push!(mins, q)
    end
    length(mins) == 1 || error("expected a principal upset; found $(length(mins)) minimal vertices")
    return mins[1]
end

# Helper for principal downset summands: unique maximal element.
function _unique_maximal_vertex(D::Downset)::Int
    P = D.P
    maxs = Int[]
    for q in 1:P.n
        D.mask[q] || continue
        # q is maximal in D if there is no r > q also in D.
        ismax = true
        for r in 1:P.n
            (r == q) && continue
            if D.mask[r] && P.leq[q, r]
                ismax = false
                break
            end
        end
        ismax && push!(maxs, q)
    end
    length(maxs) == 1 || error("expected a principal downset; found $(length(maxs)) maximal vertices")
    return maxs[1]
end


# -----------------------------------------------------------------------------
# IMPORTANT:
# PosetModules exports betti/betti_table/bass/bass_table from DerivedFunctors.
# So we must EXTEND those functions here (not define new Invariants.betti, etc).
# -----------------------------------------------------------------------------
import ..DerivedFunctors: betti, betti_table, bass, bass_table

"""
    betti(F) -> Dict{Tuple{Int,Int},Int}

Compute multigraded Betti numbers from an upset indicator resolution.

The input `F` is the `F` returned by `upset_resolution`, i.e. a vector of
`UpsetPresentation{QQ}` objects. Each term is a direct sum of principal upsets.

The returned dictionary is keyed by `(a, p)` where:
* `a` is the homological degree (starting at 0)
* `p` is a vertex of the underlying poset

The value is the multiplicity of the principal upset based at `p` in term `a`.

This is the same data returned by `betti(projective_resolution(...))`, but is
available directly from the indicator-resolution output.
"""
function betti(F::Vector{UpsetPresentation{QQ}})
    length(F) > 0 || return Dict{Tuple{Int,Int},Int}()
    out = Dict{Tuple{Int,Int}, Int}()
    for i in 1:length(F)
        a = i - 1
        for U in F[i].U0
            p = _unique_minimal_vertex(U)
            out[(a, p)] = get(out, (a, p), 0) + 1
        end
    end
    return out
end

"""
    betti_table(F) -> Matrix{Int}

Dense Betti table for an upset indicator resolution.

Row `a+1` and column `p` stores the multiplicity of the principal upset based at
vertex `p` in homological degree `a`.
"""
function betti_table(F::Vector{UpsetPresentation{QQ}}; pad_to::Union{Nothing,Int}=nothing)
    length(F) > 0 || return zeros(Int, 0, 0)
    P = F[1].P
    B = zeros(Int, length(F), P.n)
    for ((a, p), m) in betti(F)
        B[a + 1, p] = m
    end

    if pad_to === nothing
        return B
    else
        pad_to >= 0 || error("betti_table: pad_to must be >= 0")
        r = pad_to + 1
        if size(B, 1) == r
            return B
        elseif size(B, 1) > r
            return B[1:r, :]
        else
            C = zeros(Int, r, P.n)
            C[1:size(B,1), :] .= B
            return C
        end
    end
end

"""
    bass(E) -> Dict{Tuple{Int,Int},Int}

Compute multigraded Bass numbers from a downset indicator resolution.

The input `E` is the `E` returned by `downset_resolution`, i.e. a vector of
`DownsetCopresentation{QQ}` objects. Each term is a direct sum of principal
downsets.

The returned dictionary is keyed by `(b, p)` where:
* `b` is the cohomological degree (starting at 0)
* `p` is a vertex of the underlying poset

The value is the multiplicity of the principal downset with top element `p` in
term `b`.
"""
function bass(E::Vector{DownsetCopresentation{QQ}})
    length(E) > 0 || return Dict{Tuple{Int,Int},Int}()
    out = Dict{Tuple{Int,Int}, Int}()
    for i in 1:length(E)
        b = i - 1
        for D in E[i].D0
            p = _unique_maximal_vertex(D)
            out[(b, p)] = get(out, (b, p), 0) + 1
        end
    end
    return out
end

"""
    bass_table(E) -> Matrix{Int}

Dense Bass table for a downset indicator resolution.

Row `b+1` and column `p` stores the multiplicity of the principal downset with
top vertex `p` in cohomological degree `b`.
"""
function bass_table(E::Vector{DownsetCopresentation{QQ}}; pad_to::Union{Nothing,Int}=nothing)
    length(E) > 0 || return zeros(Int, 0, 0)
    P = E[1].P
    B = zeros(Int, length(E), P.n)
    for ((b, p), m) in bass(E)
        B[b + 1, p] = m
    end

    if pad_to === nothing
        return B
    else
        pad_to >= 0 || error("bass_table: pad_to must be >= 0")
        r = pad_to + 1
        if size(B, 1) == r
            return B
        elseif size(B, 1) > r
            return B[1:r, :]
        else
            C = zeros(Int, r, P.n)
            C[1:size(B,1), :] .= B
            return C
        end
    end
end


# =============================================================================
# (5) Linking region size measures back to algebraic / derived invariants
# =============================================================================
#
# The main idea is simple: many algebraic quantities (or quantities derived from
# algebraic computations) are constant on encoding regions. Once you can label
# each region by such an invariant, you can measure the "locus" where it takes
# a given value by summing region weights.
#
# This section provides:
#   - `region_values(pi, f; ...)` evaluate a region-constant function to get a vector.
#   - `measure_by_value(values, pi; ...)` stratify a window by an arbitrary label.
#   - `support_measure(mask, pi; ...)` measure a subset of regions given as a mask.
#   - `vertex_set_measure(vertices, pi; ...)` measure a subset given as indices.
#   - `betti_support_measures` / `bass_support_measures` compute size summaries
#     of where Betti/Bass numbers are supported.


"""
    _nregions_encoding(pi) -> Int

Internal helper: determine the number of regions for common encoding map types.

This is used by `region_values(pi, f; arg=:index)` to allow value evaluation by
region index even when representatives are not available.

We try (in this order):
- `length(pi.sig_y)` if `pi.sig_y` exists,
- `length(pi.regions)` if `pi.regions` exists,
- `length(pi.reps)` if `pi.reps` exists.
"""
function _nregions_encoding(pi)
    if hasproperty(pi, :sig_y)
        sy = getproperty(pi, :sig_y)
        sy !== nothing && return length(sy)
    end
    if hasproperty(pi, :regions)
        rg = getproperty(pi, :regions)
        rg !== nothing && return length(rg)
    end
    if hasproperty(pi, :reps)
        reps = getproperty(pi, :reps)
        reps !== nothing && return length(reps)
    end
    error("_nregions_encoding: cannot determine number of regions; expected pi.sig_y, pi.regions, or pi.reps")
end


"""
    region_values(pi, f; arg=:rep) -> AbstractVector

Evaluate a region-constant function on each region of an encoding.

This is a lightweight helper for building "stratifications" of parameter space
by any quantity that is constant on regions.

Arguments
---------
- `pi`: an encoding map. If `arg` is `:rep` or `:both`, then `pi` must provide
  representatives as `pi.reps`. If `arg` is `:index`, representatives are not
  needed, but `pi` must expose the number of regions via one of the common
  fields `pi.sig_y`, `pi.regions`, or `pi.reps`.
- `f`: function used to label regions.

Keyword arguments
-----------------
- `arg` controls how `f` is called:
  - `:rep`   calls `f(rep)` where `rep = pi.reps[r]` is a representative point.
  - `:index` calls `f(r)` where `r` is the region index (1-based).
  - `:both`  calls `f(r, rep)`.

Returns
-------
A vector `vals` with `vals[r] = f(...)` for each region.

Notes
-----
This is intended for derived or external invariants that you can compute on a
single representative point per region.
"""
function region_values(pi, f; arg::Symbol=:rep)
    if arg === :index
        n = _nregions_encoding(pi)
        n == 0 && return Any[]
        v1 = f(1)
        T = typeof(v1)
        out = Vector{T}(undef, n)
        out[1] = v1
        @inbounds for r in 2:n
            out[r] = f(r)
        end
        return out
    end

    # arg == :rep or :both require representatives.
    if !hasproperty(pi, :reps)
        error("region_values: arg=$arg requires pi.reps, but pi has no property reps")
    end
    reps = getproperty(pi, :reps)
    reps === nothing && error("region_values: pi.reps is nothing")
    n = length(reps)
    n == 0 && return Any[]

    # Compute the first value to infer element type.
    v1 = if arg === :rep
        f(reps[1])
    elseif arg === :both
        f(1, reps[1])
    else
        error("region_values: arg must be :rep, :index, or :both")
    end

    T = typeof(v1)
    out = Vector{T}(undef, n)
    out[1] = v1

    if n == 1
        return out
    end

    if arg === :rep
        @inbounds for r in 2:n
            out[r] = f(reps[r])
        end
    else
        @inbounds for r in 2:n
            out[r] = f(r, reps[r])
        end
    end
    return out
end


# ------------------------------ Region poset access ------------------------------
#
# For "PL-like" encodings (ZnEncoding, PLPolyhedra, PLBackend), the encoder returns
# a region poset P together with an encoding map pi. The encoding map typically
# stores only per-region signatures (sig_y, sig_z) and representative points.
#
# The projected-invariant machinery (and slice_chain sanity checks) needs access
# to this region poset. We therefore reconstruct it from signatures when it is
# not stored explicitly, and cache the result keyed by the encoding map object.

# Cache for region_poset(pi) when the encoding map does not store P explicitly.
#
# IMPORTANT:
# `WeakKeyDict` keys must be *mutable* objects (Julia attaches a finalizer to the key).
# Most encoding maps in this repo are immutable structs containing mutable fields.
# Therefore we key the weak cache by the mutable signature container `pi.sig_y`,
# not by `pi` itself.
#
# We also store `sig_z` (by identity) in the cache entry and verify it on lookup
# to avoid collisions if two encodings ever share the same `sig_y` object.
struct _RegionPosetCacheEntry
    sig_z::Any
    P::FinitePoset
end

const _REGION_POSET_CACHE = Base.WeakKeyDict{Any, _RegionPosetCacheEntry}()
const _REGION_POSET_CACHE_LOCK = Base.ReentrantLock()

@inline function _sig_leq(a::AbstractVector{Bool}, b::AbstractVector{Bool})::Bool
    # Componentwise order on {0,1}^m: a <= b iff whenever a[k] is true, b[k] is true.
    length(a) == length(b) || error("_sig_leq: signature length mismatch")
    @inbounds for k in eachindex(a, b)
        if a[k] && !b[k]
            return false
        end
    end
    return true
end

function _uptight_poset_from_signatures(
    sig_y::AbstractVector{<:AbstractVector{Bool}},
    sig_z::AbstractVector{<:AbstractVector{Bool}},
)::FinitePoset
    rN = length(sig_y)
    length(sig_z) == rN || error("_uptight_poset_from_signatures: sig_y and sig_z length mismatch")

    leq = falses(rN, rN)
    @inbounds for i in 1:rN
        yi = sig_y[i]
        zi = sig_z[i]
        for j in 1:rN
            if _sig_leq(yi, sig_y[j]) && _sig_leq(zi, sig_z[j])
                leq[i, j] = true
            end
        end
    end

    # Signature inclusion is reflexive and transitive, so leq is already a preorder.
    # FinitePoset(leq) validates antisymmetry/transitivity, giving a true poset.
    #
    # Note: this avoids the extra O(r^3) transitive-closure pass used in some encoders,
    # while still keeping FinitePoset's correctness checks.
    return FinitePoset(leq)
end

"""
    region_poset(pi) -> FinitePoset

Return the finite "region poset" underlying a `PLikeEncodingMap` `pi`.

In this codebase, the region poset is the *uptight poset on signatures*:
for regions r and s we declare r <= s iff both signature bitvectors satisfy

    sig_y[r] <= sig_y[s]   and   sig_z[r] <= sig_z[s]

componentwise (i.e. every upset/downset membership bit that is true in r is also
true in s). This is exactly the poset P constructed by the encoders.

Implementation notes
--------------------
* If the encoding object stores a region poset directly as a field/property `P`,
  we return it.
* Otherwise we reconstruct P from the stored signatures `pi.sig_y` and `pi.sig_z`,
  and cache the result in a `WeakKeyDict` keyed by `pi.sig_y` (a mutable vector).
  This avoids repeating the O(r^3) validation work inside `FinitePoset(...)` when
  region_poset is queried many times for the same encoding map.
"""
function region_poset(pi::PLikeEncodingMap)
    # Fast path: the encoding itself stores P.
    if hasproperty(pi, :P)
        P = getproperty(pi, :P)
        if P isa FinitePoset
            return P
        end
    end

    # Signature-based reconstruction (the standard situation for encoders in this repo).
    if !(hasproperty(pi, :sig_y) && hasproperty(pi, :sig_z))
        error("region_poset: pi has no property P and no (sig_y, sig_z); cannot reconstruct for type $(typeof(pi))")
    end
    sig_y = getproperty(pi, :sig_y)
    sig_z = getproperty(pi, :sig_z)

    # Cached reconstruction path.
    #
    # Key the WeakKeyDict by `sig_y` (mutable), not by `pi` (often immutable).
    lock(_REGION_POSET_CACHE_LOCK)
    entry = get(_REGION_POSET_CACHE, sig_y, nothing)
    unlock(_REGION_POSET_CACHE_LOCK)

    if entry !== nothing
        e = entry::_RegionPosetCacheEntry
        if e.sig_z === sig_z
            return e.P
        end
        # If sig_y is shared but sig_z differs, treat as a cache miss.
        # (This should be very rare; correctness beats caching here.)
    end

    Pnew = _uptight_poset_from_signatures(sig_y, sig_z)

    lock(_REGION_POSET_CACHE_LOCK)
    _REGION_POSET_CACHE[sig_y] = _RegionPosetCacheEntry(sig_z, Pnew)
    unlock(_REGION_POSET_CACHE_LOCK)
    return Pnew
end



"""
    measure_by_value(values, pi; box=nothing, weights=nothing, kwargs...) -> Dict

Given a vector `values` indexed by regions, return the total region-weight
(measure) for each distinct value.

This is a generalization of `measure_by_dimension`, and is meant to support
"locus size" questions such as:
- "how much parameter space has invariant == v?"
- "how much space supports Betti generators in degree a with multiplicity m?"
- "how much space lies in each stratum of a region-constant derived invariant?"

Keyword arguments
-----------------
- `box`: window used when computing region weights (defaults to `:auto` behavior
  used elsewhere if you pass `box=:auto`).
- `weights`: optionally provide precomputed region weights to avoid recomputation.
- `kwargs...`: forwarded to `region_weights(pi; ...)` when weights are not given.

Returns
-------
A dictionary `Dict{V,T}` where `V = eltype(values)` and `T = promote_type(eltype(weights), Int)`.
"""
function measure_by_value(values::AbstractVector, pi; box=nothing, weights=nothing, kwargs...)
    w = if weights === nothing
        bb = _resolve_box(pi, box)
        region_weights(pi; box=bb, kwargs...)
    else
        weights
    end
    length(w) == length(values) || error("measure_by_value: weights length does not match values length")

    T = promote_type(eltype(w), Int)
    out = Dict{eltype(values), T}()
    zT = zero(T)
    @inbounds for i in eachindex(values)
        v = values[i]
        out[v] = get(out, v, zT) + w[i]
    end
    return out
end


"""
    measure_by_value(f, pi; arg=:rep, box=nothing, weights=nothing, kwargs...) -> Dict

Convenience wrapper: compute `values = region_values(pi, f; arg=arg)` and then
call `measure_by_value(values, pi; ...)`.
"""
function measure_by_value(f::Function, pi; arg::Symbol=:rep, box=nothing, weights=nothing, kwargs...)
    vals = region_values(pi, f; arg=arg)
    return measure_by_value(vals, pi; box=box, weights=weights, kwargs...)
end


"""
    support_measure(mask, pi; box=nothing, weights=nothing, kwargs...) -> Real

Measure the union of a subset of regions, given as a boolean mask.

This complements `support_measure(M, pi)` which measures the parameter-space
subset {dim M(x) >= 1}. Here, you provide the subset directly.
"""
function support_measure(mask::AbstractVector{Bool}, pi; box=nothing, weights=nothing, kwargs...)
    w = if weights === nothing
        bb = _resolve_box(pi, box)
        region_weights(pi; box=bb, kwargs...)
    else
        weights
    end
    length(w) == length(mask) || error("support_measure(mask): weights length does not match mask length")
    T = promote_type(eltype(w), Int)
    s = zero(T)
    @inbounds for i in eachindex(mask)
        mask[i] && (s += w[i])
    end
    return s
end


"""
    vertex_set_measure(vertices, pi; box=nothing, weights=nothing, kwargs...) -> Real

Measure the union of a subset of regions, given as a list of region indices.

This is intentionally *not* called `support_measure`, because `support_measure`
already supports passing a vector of Hilbert values (dimensions) in place of a
module. Those Hilbert vectors are also integer vectors, so we avoid dispatch
ambiguity by using a distinct name.

Duplicate indices are ignored.
"""
function vertex_set_measure(vertices::AbstractVector{<:Integer}, pi; box=nothing, weights=nothing, kwargs...)
    w = if weights === nothing
        bb = _resolve_box(pi, box)
        region_weights(pi; box=bb, kwargs...)
    else
        weights
    end
    n = length(w)
    T = promote_type(eltype(w), Int)
    s = zero(T)
    seen = falses(n)
    @inbounds for p in vertices
        1 <= p <= n || error("vertex_set_measure: vertex index $p out of range 1:$n")
        if !seen[p]
            s += w[p]
            seen[p] = true
        end
    end
    return s
end


# Internal helper: support and mass measures for multigraded tables indexed by (deg, vertex).
function _multigraded_support_measures(tbl::Dict{Tuple{Int,Int},<:Integer}, w)
    n = length(w)
    T = promote_type(eltype(w), Int)

    # Determine degrees present.
    dmax = -1
    for (k, _) in tbl
        d = k[1]
        d > dmax && (dmax = d)
    end
    if dmax < 0
        return (support_by_degree=zeros(T, 0), mass_by_degree=zeros(T, 0), support_union=zero(T), mass_total=zero(T))
    end

    support_by = zeros(T, dmax + 1)
    mass_by = zeros(T, dmax + 1)
    seen_union = falses(n)
    su = zero(T)

    @inbounds for ((d, p), m) in tbl
        0 <= d <= dmax || continue
        1 <= p <= n || error("multigraded_support: vertex index $p out of range 1:$n")
        wp = w[p]
        if !seen_union[p]
            seen_union[p] = true
            su += wp
        end
        if m != 0
            support_by[d + 1] += wp
            mass_by[d + 1] += wp * T(m)
        end
    end

    return (support_by_degree=support_by, mass_by_degree=mass_by, support_union=su, mass_total=sum(mass_by))
end

function _multigraded_support_measures(B::AbstractMatrix{<:Integer}, w)
    n = length(w)
    size(B, 2) == n || error("multigraded_support: matrix columns must match length(weights)")
    r = size(B, 1)
    T = promote_type(eltype(w), Int)
    support_by = zeros(T, r)
    mass_by = zeros(T, r)
    su = zero(T)

    # For each vertex p, scan all degrees and accumulate.
    @inbounds for p in 1:n
        wp = w[p]
        anynz = false
        for a in 1:r
            m = B[a, p]
            if m != 0
                anynz = true
                support_by[a] += wp
                mass_by[a] += wp * T(m)
            end
        end
        anynz && (su += wp)
    end

    return (support_by_degree=support_by, mass_by_degree=mass_by, support_union=su, mass_total=sum(mass_by))
end


"""
    betti_support_measures(B, pi; box=nothing, weights=nothing, kwargs...) -> NamedTuple

Compute region-size measures of where Betti numbers are supported.

Input `B` may be:
- a `Dict{Tuple{Int,Int},Int}` as returned by `betti(res)`,
- a dense matrix as returned by `betti_table(res)`,
- any object accepted by `betti` (e.g. a `ProjectiveResolution`).

The returned `NamedTuple` contains:
- `support_by_degree`: vector where entry `a+1` is the measure of regions p
  with a nonzero Betti number in homological degree `a`.
- `mass_by_degree`: vector where entry `a+1` is the multiplicity-weighted measure
  sum_p beta_{a,p} * w[p].
- `support_union`: measure of the union of all Betti-support vertices.
- `mass_total`: total multiplicity-weighted measure across all degrees.

This is one concrete realization of the "bridge" idea: it quantifies where
algebraic generators in a resolution live in parameter space.
"""
function betti_support_measures(B, pi; box=nothing, weights=nothing, kwargs...)
    w = if weights === nothing
        bb = _resolve_box(pi, box)
        region_weights(pi; box=bb, kwargs...)
    else
        weights
    end

    if B isa Dict{Tuple{Int,Int},<:Integer}
        return _multigraded_support_measures(B, w)
    elseif B isa AbstractMatrix{<:Integer}
        return _multigraded_support_measures(B, w)
    else
        return betti_support_measures(betti(B), pi; box=box, weights=weights, kwargs...)
    end
end


"""
    bass_support_measures(B, pi; box=nothing, weights=nothing, kwargs...) -> NamedTuple

Analog of `betti_support_measures` for multigraded Bass numbers.

Input `B` may be:
- a `Dict{Tuple{Int,Int},Int}` as returned by `bass(res)`,
- a dense matrix as returned by `bass_table(res)`,
- any object accepted by `bass` (e.g. an `InjectiveResolution`).
"""
function bass_support_measures(B, pi; box=nothing, weights=nothing, kwargs...)
    w = if weights === nothing
        bb = _resolve_box(pi, box)
        region_weights(pi; box=bb, kwargs...)
    else
        weights
    end

    if B isa Dict{Tuple{Int,Int},<:Integer}
        return _multigraded_support_measures(B, w)
    elseif B isa AbstractMatrix{<:Integer}
        return _multigraded_support_measures(B, w)
    else
        return bass_support_measures(bass(B), pi; box=box, weights=weights, kwargs...)
    end
end


# -----------------------------------------------------------------------------
# Support geometry: unions of regions with dim(M_x) >= k
# -----------------------------------------------------------------------------

"""
    support_mask(H; min_dim=1)

Return a BitVector indicating which regions are in the support:
mask[r] == true iff H[r] >= min_dim.
"""
function support_mask(H::AbstractVector{<:Integer}; min_dim::Integer=1)
    m = BitVector(undef, length(H))
    @inbounds for i in eachindex(H)
        m[i] = (H[i] >= min_dim)
    end
    return m
end

"""
    support_vertices(H; min_dim=1)

Return the list of region indices where H[r] >= min_dim.
"""
support_vertices(H::AbstractVector{<:Integer}; min_dim::Integer=1) =
    findall(support_mask(H; min_dim=min_dim))

"""
    support_vertices(M, pi; min_dim=1)

Compute restricted Hilbert function on pi and return its support vertices.
"""
function support_vertices(M, pi; min_dim::Integer=1)
    H = restricted_hilbert(M; pi=pi)
    return support_vertices(H; min_dim=min_dim)
end

# Internal: induced components on a mask from an edge dictionary adjacency.
function _masked_components_from_edges(n::Int, edges::Dict{Tuple{Int,Int},<:Real}, mask::BitVector)
    nbrs = Dict{Int, Vector{Int}}()
    for ((a,b), _) in edges
        a == b && continue
        if mask[a] && mask[b]
            push!(get!(nbrs, a, Int[]), b)
            push!(get!(nbrs, b, Int[]), a)
        end
    end

    visited = falses(n)
    comps = Vector{Vector{Int}}()

    for v in 1:n
        if mask[v] && !visited[v]
            stack = Int[v]
            visited[v] = true
            comp = Int[]
            while !isempty(stack)
                u = pop!(stack)
                push!(comp, u)
                for w in get(nbrs, u, Int[])
                    if mask[w] && !visited[w]
                        visited[w] = true
                        push!(stack, w)
                    end
                end
            end
            sort!(comp)
            push!(comps, comp)
        end
    end
    return comps
end

"""
    support_components(H, pi; box=:auto, min_dim=1, adjacency=nothing, strict=true)

Connected components of the support (regions where H[r] >= min_dim) under region
adjacency. Returns a vector of components, each a vector of region indices.

If `adjacency` is not provided, this calls `region_adjacency(pi; box=...)`.
"""
function support_components(H::AbstractVector{<:Integer}, pi;
    box=:auto,
    min_dim::Integer=1,
    adjacency=nothing,
    strict::Bool=true
)
    mask = support_mask(H; min_dim=min_dim)
    bb = _resolve_box(pi, box)

    edges = adjacency === nothing ? region_adjacency(pi; box=bb, strict=strict) : adjacency
    return _masked_components_from_edges(length(H), edges, mask)
end

"""
    support_components(M, pi; box=:auto, min_dim=1, adjacency=nothing, strict=true)

Same as support_components(H,pi), but computes H = restricted_hilbert(M; pi=pi).
"""
function support_components(M, pi; box=:auto, min_dim::Integer=1, adjacency=nothing, strict::Bool=true)
    H = restricted_hilbert(M; pi=pi)
    return support_components(H, pi; box=box, min_dim=min_dim, adjacency=adjacency, strict=strict)
end

# Graph diameter helpers (exact for small components, double-sweep approx for large).
function _bfs_eccentricity(nbrs::Dict{Int,Vector{Int}}, start::Int, allowed::BitVector)
    q = Int[start]
    dist = Dict{Int,Int}(start => 0)
    head = 1
    maxd = 0
    while head <= length(q)
        v = q[head]; head += 1
        dv = dist[v]
        maxd = max(maxd, dv)
        for w in get(nbrs, v, Int[])
            if allowed[w] && !haskey(dist, w)
                dist[w] = dv + 1
                push!(q, w)
            end
        end
    end
    return maxd
end

"""
    support_graph_diameter(H, pi; box=:auto, min_dim=1, strict=true, exact_limit=300)

Graph diameter of each support component and the overall support graph.
Returns (component_diameters, overall_diameter).
"""
function support_graph_diameter(H::AbstractVector{<:Integer}, pi;
    box=:auto,
    min_dim::Integer=1,
    strict::Bool=true,
    exact_limit::Int=300
)
    mask = support_mask(H; min_dim=min_dim)
    bb = _resolve_box(pi, box)
    edges = region_adjacency(pi; box=bb, strict=strict)

    nbrs = Dict{Int,Vector{Int}}()
    for ((a,b), _) in edges
        a == b && continue
        if mask[a] && mask[b]
            push!(get!(nbrs, a, Int[]), b)
            push!(get!(nbrs, b, Int[]), a)
        end
    end

    comps = support_components(H, pi; box=bb, min_dim=min_dim, adjacency=edges, strict=strict)

    diameters = Int[]
    for comp in comps
        if length(comp) <= 1
            push!(diameters, 0)
            continue
        end
        allowed = falses(length(H))
        for v in comp
            allowed[v] = true
        end
        if length(comp) <= exact_limit
            dmax = 0
            for v in comp
                dmax = max(dmax, _bfs_eccentricity(nbrs, v, allowed))
            end
            push!(diameters, dmax)
        else
            # Approx diameter via double sweep.
            v0 = comp[1]
            d0 = _bfs_eccentricity(nbrs, v0, allowed)
            push!(diameters, d0)
        end
    end

    overall = isempty(diameters) ? 0 : maximum(diameters)
    return (diameters, overall)
end

"""
    support_bbox(H, pi; box=:auto, min_dim=1, strict=true)

Bounding box of the union of supported regions, computed by unioning per-region bboxes.
Requires region_bbox support in the backend. Uses the resolved finite `box` to clip.
"""
function support_bbox(H::AbstractVector{<:Integer}, pi;
    box=:auto,
    min_dim::Integer=1,
    strict::Bool=true
)
    verts = support_vertices(H; min_dim=min_dim)
    bb = _resolve_box(pi, box)
    isempty(verts) && return bb

    ell = copy(bb[1])
    u = copy(bb[2])
    first = true
    for r in verts
        rb = region_bbox(pi, r; box=bb, strict=strict)
        if first
            ell .= rb[1]
            u .= rb[2]
            first = false
        else
            @inbounds for i in eachindex(ell)
                ell[i] = min(ell[i], rb[1][i])
                u[i] = max(u[i], rb[2][i])
            end
        end
    end
    return (ell, u)
end

"""
    support_geometric_diameter(H, pi; box=:auto, min_dim=1, metric=:L2)

Geometric diameter of the support union, upper-bounded by its bounding box diameter.

metric:
- :L2   sqrt(sum((u-ell)^2))
- :Linf max(u-ell)
- :L1   sum(u-ell)
"""
function support_geometric_diameter(H::AbstractVector{<:Integer}, pi;
    box=:auto,
    min_dim::Integer=1,
    metric::Symbol=:L2,
    strict::Bool=true
)
    bb = support_bbox(H, pi; box=box, min_dim=min_dim, strict=strict)
    ell, u = bb
    if metric == :L2
        s = 0.0
        for i in eachindex(ell)
            d = float(u[i] - ell[i])
            s += d*d
        end
        return sqrt(s)
    elseif metric == :Linf
        m = 0.0
        for i in eachindex(ell)
            m = max(m, float(u[i] - ell[i]))
        end
        return m
    elseif metric == :L1
        s = 0.0
        for i in eachindex(ell)
            s += float(u[i] - ell[i])
        end
        return s
    else
        error("Unknown metric=$metric; expected :L2, :Linf, or :L1.")
    end
end

"""
    support_measure_stats(H, pi; box=:auto, min_dim=1, strict=true, alpha=0.05)

Support measure with uncertainty: calls `region_weights(...; return_info=true)` if
available and returns (estimate, stderr, ci, info).

For exact backends, stderr=0 and ci=(estimate,estimate).
For Monte Carlo/sample backends, uses binomial Wilson interval on the *subset* count.
"""
function support_measure_stats(H::AbstractVector{<:Integer}, pi;
    box=:auto,
    min_dim::Integer=1,
    strict::Bool=true,
    alpha::Real=0.05
)
    bb = _resolve_box(pi, box)
    rw = region_weights(pi; box=bb, strict=strict, return_info=true, alpha=alpha)

    mask = support_mask(H; min_dim=min_dim)
    weights = rw.weights

    est = 0.0
    for i in eachindex(weights)
        if mask[i]
            est += float(weights[i])
        end
    end

    # Exact: counts missing or nsamples==0 => degenerate CI.
    if !haskey(rw, :counts) || rw.counts === nothing || rw.nsamples == 0
        return (estimate=est, stderr=0.0, ci=(est, est), info=rw)
    end

    # Subset count is binomial on "in support" vs "not in support".
    c = 0
    for i in eachindex(rw.counts)
        if mask[i]
            c += rw.counts[i]
        end
    end

    n = rw.nsamples
    total = haskey(rw, :total_volume) ? rw.total_volume : float(rw.total_points)
    p = c / n
    se = total * sqrt(p * (1.0 - p) / n)
    (plo, phi) = _wilson_interval(c, n; alpha=float(alpha))
    ci = (total * plo, total * phi)

    return (estimate=est, stderr=se, ci=ci, info=rw)
end



end # module
