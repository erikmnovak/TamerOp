module Invariants

"""
    TamerOp.Invariants

Owner module for the basic invariant surface built on encoded multiparameter
modules. `Invariants` now owns the core rank/Hilbert/statistics/support-query
family, while distinct invariant families such as `SignedMeasures`,
`SliceInvariants`, `Fibered2D`, and `MultiparameterImages` live in sibling owner
modules.

Private fragments loaded here:
- `src/invariants/basics.jl`: rank/Hilbert primitives
- `src/invariants/module_stats.jl`: module-size summaries and interface measures
- `src/invariants/geometry_stats.jl`: geometry/asymptotic summaries
- `src/invariants/algebraic_support.jl`: pretty printing, Betti/Bass support,
  region-value stratifications
- `src/invariants/support_geometry.jl`: support-geometry queries
- `src/invariants/wrappers.jl`: opts-default public wrappers
"""

using LinearAlgebra
using JSON3
using ..CoreModules: EncodingCache, AbstractCoeffField, RegionPosetCachePayload,
                     AbstractSlicePlanCache
using ..Options: InvariantOptions
using ..EncodingCore: PLikeEncodingMap, CompiledEncoding, locate, axes_from_encoding, dimension, representatives,
                       check_query_point
using Statistics: mean
using ..Stats: _wilson_interval
using ..Encoding: EncodingMap
import ..Encoding: source_poset
import ..Encoding: source_poset
using ..PLPolyhedra
using ..RegionGeometry: region_weights, region_volume, region_bbox, region_widths,
                        region_centroid, region_aspect_ratio, region_diameter,
                        region_adjacency, region_facet_count, region_vertex_count,
                        region_boundary_measure, region_boundary_measure_breakdown,
                        region_perimeter, region_surface_area,
                        region_principal_directions,
                        region_chebyshev_ball, region_chebyshev_center, region_inradius,
                        region_circumradius,
                        region_boundary_to_volume_ratio, region_isoperimetric_ratio,
                        region_mean_width, region_minkowski_functionals,
                        region_covariance_anisotropy, region_covariance_eccentricity, region_anisotropy_scores

using ..FieldLinAlg
using ..InvariantCore: SliceSpec, RankQueryCache,
                       _unwrap_compiled,
                       _default_strict, _default_threads, _drop_keys,
                       orthant_directions, _selection_kwargs_from_opts, _axes_kwargs_from_opts,
                       _eye,
                       FloatBarcode, IndexBarcode, PackedBarcodeGrid, PackedIndexBarcode, PackedFloatBarcode,
                       RANK_INVARIANT_MEMO_THRESHOLD, RECTANGLE_LOC_LINEAR_CACHE_THRESHOLD,
                       _use_array_memo, _new_array_memo, _grid_cache_index,
                       _memo_index, _memo_get, _memo_set!, _map_leq_cached,
                       _rank_cache_get!, _resolve_rank_query_cache,
                       _rank_query_point_tuple, _rank_query_locate!
import ..InvariantCore: rank_map
import ..FiniteFringe: AbstractPoset, FinitePoset, FringeModule, Upset, Downset, fiber_dimension,
                       leq, leq_matrix, upset_indices, downset_indices, leq_col, nvertices, build_cache!,
                       _preds
import ..ZnEncoding
import ..IndicatorTypes: UpsetPresentation, DownsetCopresentation
import ..ChangeOfPosets: pushforward_left, pushforward_right
import ..DerivedFunctors: source_module
import Base.Threads

import ..Serialization: save_mpp_decomposition_json, load_mpp_decomposition_json,
                        save_mpp_image_json, load_mpp_image_json

# Invariants are field-generic; some workflows are only exact over QQ.
import ..Modules: PModule, map_leq, CoverCache, _get_cover_cache
import ..IndicatorResolutions: pmodule_from_fringe

import ..ZnEncoding: ZnEncodingMap
import ..SignedMeasures: Rect, RectSignedBarcode, PointSignedMeasure,
                           rectangle_signed_barcode, rank_from_signed_barcode, rectangles_from_grid,
                           rectangle_signed_barcode_rank, truncate_signed_barcode,
                           rectangle_signed_barcode_kernel, rectangle_signed_barcode_image,
                           mma_decomposition, coarsen_axis, coarsen_axes,
                           restrict_axes_to_encoding
import ..SliceInvariants: _normalize_box, encoding_box, window_box,
                          slice_chain, restrict_to_chain, slice_barcode,
                          bottleneck_distance, SliceDistanceTask,
                          wasserstein_distance, wasserstein_kernel, sliced_wasserstein_kernel,
                          sliced_wasserstein_distance, sliced_bottleneck_distance,
                          sample_directions_2d, default_directions, default_offsets,
                          matching_distance_approx, matching_wasserstein_distance_approx,
                          PersistenceLandscape1D, landscape_value, persistence_landscape,
                          PersistenceImage1D, persistence_image, feature_map, feature_vector,
                          persistence_silhouette, barcode_entropy, barcode_summary,
                          collect_slices, save_slices_json, load_slices_json,
                          CompiledSlicePlan, SlicePlanCacheKey, SlicePlanCache,
                          clear_slice_plan_cache!, clear_slice_module_cache!,
                          SliceModuleCache, SliceModulePairCache,
                          compile_slice_plan, compile_slices, slice_barcodes, run_invariants,
                          slice_features, slice_kernel, module_cache,
                          SliceBarcodesTask, SliceKernelTask
import ..Fibered2D: slice_chain_exact_2d, matching_distance_exact_slices_2d,
                    FiberedArrangement2D, FiberedBarcodeCache2D,
                    FiberedSliceResult,
                    ProjectedArrangement1D, ProjectedArrangement, ProjectedBarcodeCache,
                    ProjectedBarcodesResult, ProjectedDistancesResult,
                    Fibered2DValidationSummary,
                    fibered_arrangement_2d, fibered_barcode_cache_2d, fibered_cell_id,
                    fibered_chain, fibered_values, fibered_barcode, fibered_barcode_index,
                    fibered_slice, fibered_barcode_cache_stats, FiberedSliceFamily2D, nslices,
                    source_encoding, working_box, direction_representatives, slope_breaks,
                    ncells, computed_cell_count, chain_count, backend, shared_arrangement,
                    cached_barcode_count, source_arrangement, unique_chain_count,
                    direction_weight_scheme, stores_values, slice_direction,
                    slice_offset, slice_offset_interval, slice_chain_id, slice_values,
                    projections, nprojections, projection_direction, projection_values,
                    projection_chain, projection_indices, projection_directions,
                    computed_projection_count, distance_metric,
                    fibered2d_validation_summary,
                    check_fibered_arrangement_2d, check_fibered_barcode_cache_2d,
                    check_fibered_slice_family_2d, check_projected_arrangement,
                    check_projected_barcode_cache, check_fibered_cache_pair,
                    check_projected_cache_pair, check_fibered_direction,
                    check_fibered_offset, check_fibered_basepoint,
                    check_fibered_query, check_projected_direction,
                    fibered_arrangement_summary, fibered_cache_summary,
                    fibered_family_summary, projected_arrangement_summary,
                    projected_cache_summary, fibered_query_summary,
                    fibered_slice_family_2d, matching_distance_exact_2d,
                    projected_arrangement, projected_barcode_cache, projected_barcodes,
                    projected_distances, projected_distance, projected_kernel
import ..MultiparameterImages: MPPLineSpec, MPPDecomposition, MPPImage, MPLandscape,
                               mpp_decomposition, mpp_image, mpp_image_inner_product,
                               mpp_image_distance, mpp_image_kernel, mp_landscape,
                               mp_landscape_distance, mp_landscape_inner_product, mp_landscape_kernel
import ..ChainComplexes: describe

include("invariants/basics.jl")
include("invariants/summary_types.jl")
include("invariants/module_stats.jl")
include("invariants/geometry_stats.jl")
include("invariants/algebraic_support.jl")
include("invariants/support_geometry.jl")
include("invariants/wrappers.jl")

end # module
