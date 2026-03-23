"""
TamerOp.Visualization

Backend-agnostic visualization owner for encoding and invariant objects.

Design
------
- Core visualization data lives here as inspectable `VisualizationSpec` objects.
- Rendering backends live in `ext/` and register themselves on demand.
- The canonical simple-surface workflow is:

  `visualize(obj)` -> backend-native figure/widget
  `save_visual(outdir, stem, obj)` -> choose an export target automatically
  `save_visuals(outdir, requests)` -> batch export without backend bookkeeping
  `available_visuals(obj)` -> supported visualization kinds

- The advanced exact-control export path remains available as:

  `save_visual(path, obj)` -> write to an explicit filename/extension

Cheap-first workflow
--------------------
- start with `available_visuals(obj)` to see the supported kinds,
- build an inspectable spec with `visual_spec(obj; kind=...)`,
- inspect it via `describe(spec)` / `visual_summary(spec)`,
- validate via `check_visual_spec(spec)` when you hand-build a spec,
- export with `save_visual(outdir, stem, obj)` or `save_visuals(outdir, requests)`
  when you want files without manual backend bookkeeping,
- use `save_visual(path, obj)` only when you want exact filename/backend control,
- render with `visualize(obj)` or `render(spec)` only when you actually want
  a backend figure.

This owner intentionally keeps rendering out of `src/` so notebook and script
UX can stay stable even when no Makie backend is installed.
"""
module Visualization

using Statistics

import ..ChainComplexes: describe, cohomology_dims
using ..CoreModules: SessionCache
using ..FiniteFringe: AbstractPoset, ProductOfChainsPoset, GridPoset, ProductPoset, nvertices, leq
import ..DataTypes
import ..DataIngestion
using ..EncodingCore: AbstractPLikeEncodingMap, CompiledEncoding, GridEncodingMap,
                      compile_encoding, encoding_axes, encoding_map,
                      encoding_representatives, locate
using ..Encoding: EncodingMap, source_poset, target_poset, region_map
using ..Results: EncodingResult, CohomologyDimsResult, ModuleTranslationResult, InvariantResult,
                 invariant_value, translation_map, translation_kind, source_result
using ..ChangeOfPosets: CommonRefinementTranslationResult, common_poset, projection_maps
using ..FlangeZn: Flange, active_flats, active_injectives, flats, injectives
using ..FieldLinAlg: rank_restricted
using ..Invariants: RankInvariantResult, value_at, source_poset, check_rank_query_points
using ..PLBackend: PLEncodingMapBoxes, region_bbox, nregions, _cells_in_region_in_box
using ..ZnEncoding: ZnEncodingMap, region_representatives
using ..SliceInvariants: SliceBarcodesResult, slice_barcodes, slice_weights, slice_directions, slice_offsets,
                         bottleneck_distance
using ..Fibered2D: FiberedArrangement2D, FiberedBarcodeCache2D, FiberedSliceFamily2D, FiberedSliceResult,
                   ProjectedArrangement, ProjectedBarcodesResult, ProjectedDistancesResult,
                   source_encoding, working_box, direction_representatives, slope_breaks,
                   ncells, computed_cell_count, source_arrangement, slice_direction,
                   slice_offset_interval, slice_offset, slice_chain_id, fibered_values, stores_values,
                   projections, projection_indices,
                   projection_directions, projected_distances, shared_arrangement,
                   fibered_query_summary, backend, fibered_barcode_cache_2d,
                   slice_chain, slice_values, slice_barcode, fibered_slice,
                   fibered_slice_family_2d, _arr2d_cell_offset_interval, _fibered_dir_cell_index, _arr2d_compute_cell!
using ..SignedMeasures: Rect, RectSignedBarcode, PointSignedMeasure, SignedMeasureDecomposition,
                        rectangles, weights, points, total_variation, total_mass,
                        components, component_names, has_rectangles,
                        has_euler_signed_measure, has_mpp_image,
                        euler_signed_measure, mpp_image
using ..MultiparameterImages: MPPLineSpec, MPPDecomposition, MPPImage, MPLandscape,
                              line_direction, line_basepoint, line_offset, line_omega,
                              line_specs, summand_segments, nsummands, nlines, bounding_box,
                              summand_weights, image_xgrid, image_ygrid, image_values,
                              decomposition, landscape_grid, landscape_values,
                              landscape_layers, slice_weights, slice_directions,
                              slice_offsets, ndirections, noffsets, landscape_slice
include("visualization/types.jl")
include("visualization/validation.jl")
include("visualization/rendering.jl")
include("visualization/builders_encoding.jl")
include("visualization/builders_invariants.jl")
include("visualization/builders_ingestion.jl")

end # module Visualization
