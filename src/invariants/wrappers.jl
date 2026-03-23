# This fragment owns opts-default public wrappers for TamerOp.Invariants.
# It should stay thin and only forward to the canonical owner methods defined in
# earlier invariant fragments or sibling invariant families.

# -----------------------------------------------------------------------------
# Public opts-default wrappers (keyword opts)
# -----------------------------------------------------------------------------

rank_invariant(H::FringeModule{K}; opts::InvariantOptions=InvariantOptions(), kwargs...) where {K} =
    rank_invariant(H, opts; kwargs...)

rectangle_signed_barcode(M, pi; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    rectangle_signed_barcode(M, pi, opts; kwargs...)

restricted_hilbert(M::PModule{K}, pi, x; opts::InvariantOptions=InvariantOptions()) where {K} =
    restricted_hilbert(M, pi, x, opts)
restricted_hilbert(H::FringeModule{K}, pi, x; opts::InvariantOptions=InvariantOptions()) where {K} =
    restricted_hilbert(H, pi, x, opts)
rank_map(M::PModule{K}, pi, x, y; opts::InvariantOptions=InvariantOptions(), cache=nothing, memo=nothing) where {K} =
    rank_map(M, pi, x, y, opts; cache = cache, memo = memo)
rank_map(H::FringeModule{K}, pi, x, y; opts::InvariantOptions=InvariantOptions(), cache=nothing, memo=nothing) where {K} =
    rank_map(H, pi, x, y, opts; cache = cache, memo = memo)

rank_query(M::PModule{K}, pi, x, y; opts::InvariantOptions=InvariantOptions(), kwargs...) where {K} =
    rank_query(M, pi, x, y, opts; kwargs...)
rank_query(H::FringeModule{K}, pi, x, y; opts::InvariantOptions=InvariantOptions(), kwargs...) where {K} =
    rank_query(H, pi, x, y, opts; kwargs...)

hilbert_distance(M, N, pi; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    hilbert_distance(M, N, pi, opts; kwargs...)

integrated_hilbert_mass(M, pi; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    integrated_hilbert_mass(M, pi, opts; kwargs...)
measure_by_dimension(M, pi; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    measure_by_dimension(M, pi, opts; kwargs...)
support_measure(M, pi; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    support_measure(M, pi, opts; kwargs...)
dim_stats(M, pi; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    dim_stats(M, pi, opts; kwargs...)
dim_norm(M, pi; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    dim_norm(M, pi, opts; kwargs...)
region_weight_entropy(pi; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    region_weight_entropy(pi, opts; kwargs...)
aspect_ratio_stats(pi; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    aspect_ratio_stats(pi, opts; kwargs...)
module_size_summary(M, pi; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    module_size_summary(M, pi, opts; kwargs...)
interface_measure(pi; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    interface_measure(pi, opts; kwargs...)
interface_measure_by_dim_pair(M, pi; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    interface_measure_by_dim_pair(M, pi, opts; kwargs...)
interface_measure_dim_changes(M, pi; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    interface_measure_dim_changes(M, pi, opts; kwargs...)
region_volume_samples_by_dim(M, pi; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    region_volume_samples_by_dim(M, pi, opts; kwargs...)
region_volume_histograms_by_dim(M, pi; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    region_volume_histograms_by_dim(M, pi, opts; kwargs...)
region_boundary_to_volume_samples_by_dim(M, pi; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    region_boundary_to_volume_samples_by_dim(M, pi, opts; kwargs...)
region_boundary_to_volume_histograms_by_dim(M, pi; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    region_boundary_to_volume_histograms_by_dim(M, pi, opts; kwargs...)
region_adjacency_graph_stats(M, pi; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    region_adjacency_graph_stats(M, pi, opts; kwargs...)
module_geometry_summary(M, pi; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    module_geometry_summary(M, pi, opts; kwargs...)
module_geometry_asymptotics(M, pi; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    module_geometry_asymptotics(M, pi, opts; kwargs...)

slice_chain(pi, x0, dir; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    slice_chain(pi, x0, dir, opts; kwargs...)

sliced_wasserstein_kernel(M, N, pi; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    sliced_wasserstein_kernel(M, N, pi, opts; kwargs...)
sliced_wasserstein_distance(M, N, pi; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    sliced_wasserstein_distance(M, N, pi, opts; kwargs...)
sliced_bottleneck_distance(M, N, pi; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    sliced_bottleneck_distance(M, N, pi, opts; kwargs...)

encoding_box(pi::PLikeEncodingMap; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    encoding_box(pi, opts; kwargs...)
encoding_box(axes::Tuple{Vararg{<:AbstractVector}}; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    encoding_box(axes, opts; kwargs...)

default_offsets(pi::PLikeEncodingMap; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    default_offsets(pi, opts; kwargs...)
default_offsets(pi::PLikeEncodingMap, dir::AbstractVector{<:Real}; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    default_offsets(pi, dir, opts; kwargs...)

matching_distance_approx(M, N, pi; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    matching_distance_approx(M, N, pi, opts; kwargs...)
matching_wasserstein_distance_approx(M, N, pi; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    matching_wasserstein_distance_approx(M, N, pi, opts; kwargs...)

slice_chain_exact_2d(pi, dir, offset; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    slice_chain_exact_2d(pi, dir, offset, opts; kwargs...)
matching_distance_exact_slices_2d(pi; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    matching_distance_exact_slices_2d(pi, opts; kwargs...)

fibered_arrangement_2d(pi; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    fibered_arrangement_2d(pi, opts; kwargs...)
fibered_barcode_cache_2d(M, pi; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    fibered_barcode_cache_2d(M, pi, opts; kwargs...)

matching_distance_exact_2d(M, N, pi; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    matching_distance_exact_2d(M, N, pi, opts; kwargs...)

measure_by_value(values::AbstractVector, pi; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    measure_by_value(values, pi, opts; kwargs...)
measure_by_value(values::AbstractVector, v, pi; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    measure_by_value(values, v, pi, opts; kwargs...)
measure_by_value(f::Function, pi; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    measure_by_value(f, pi, opts; kwargs...)

support_measure(mask::AbstractVector{Bool}, pi; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    support_measure(mask, pi, opts; kwargs...)
vertex_set_measure(vertices::AbstractVector{<:Integer}, pi; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    vertex_set_measure(vertices, pi, opts; kwargs...)
betti_support_measures(B, pi; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    betti_support_measures(B, pi, opts; kwargs...)
bass_support_measures(B, pi; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    bass_support_measures(B, pi, opts; kwargs...)

support_components(H, pi; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    support_components(H, pi, opts; kwargs...)
support_graph_diameter(H, pi; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    support_graph_diameter(H, pi, opts; kwargs...)
support_bbox(H, pi; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    support_bbox(H, pi, opts; kwargs...)
support_geometric_diameter(H, pi; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    support_geometric_diameter(H, pi, opts; kwargs...)
support_measure_stats(H, pi; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    support_measure_stats(H, pi, opts; kwargs...)


