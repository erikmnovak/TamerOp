module PosetModules
# =============================================================================
# Umbrella module for the library.
# Include order matters:
#   1) Core helpers (numeric aliases, feature flags, thin wrappers)
#   2) Finite poset layer (Fringe + indicator sets)
#   3) Presentation/copresentation record types used across subsystems
#   4) Exact rational linear algebra (exact RREF, rank, etc.)  <-- self-contained
#   5) Finite encodings (uptight poset)
#   6) Indicator resolutions and Hom/Ext assembly
#   7) Zn flange layer (flats/injectives) and CAS bridge
#   8) IO (JSON serializations) and 2D visualization
#   9) PL polyhedra backend (+ cross-validation against flange)
# =============================================================================

# 1) Core helpers and feature flags
include("CoreModules.jl")

# 2) Exact rational linear algebra (Nemo if available, fallback otherwise)
#    IMPORTANT: our ExactQQ fallback is fully self-contained; no import from IndicatorResolutions.
include("ExactQQ.jl")

# 3) Finite poset + indicator sets + fringe presentations
include("FiniteFringe.jl")

# 4) Shared record types for one-step indicator (co)presentations
#    NOTE: these are now defined in FiniteFringe.jl as `module IndicatorTypes`
#    to keep include-order constraints simple while preserving the public module
#    name `PosetModules.IndicatorTypes`.

# 5) Finite encodings (uptight poset; image/preimage of up/downsets)
include("Encoding.jl")

# 6) Indicator resolutions + Hom/Ext (first page + general Tot^*)
include("HomExt.jl")
include("IndicatorResolutions.jl")

# 7) Zn flange data structure and CAS bridge
include("FlangeZn.jl")
include("M2SingularBridge.jl")

# 8) Serialization and lightweight visualization
include("Serialization.jl")
include("Viz2D.jl")

# 9) PL backend (general H-rep) and cross-validation helpers
include("PLPolyhedra.jl")
include("CrossValidateFlangePL.jl")

# 10) Derived-functor layer (depends on FiniteFringe + IndicatorResolutions + FlangeZn)
include("ChainComplexes.jl")
include("ZnEncoding.jl")
include("DerivedFunctors.jl")
include("ModuleComplexes.jl")
include("ChangeOfPosets.jl")

# 11) Rank invariant, restricted Hilbert function, and resolution tables
include("Invariants.jl")


# Optional: axis-aligned PL backend (kept separate; off by default)
# Enable with ENV: POSETMODULES_ENABLE_PL_AXIS=true
if CoreModules.ENABLE_PL_AXIS
    @info "Including experimental PLBackend (axis-aligned) because ENABLE_PL_AXIS=true"
    include("PLBackend.jl")
end

# ----------------------- Re-exports for a clean user-facing API ----------------
using .CoreModules
using .FiniteFringe
using .IndicatorTypes
using .ExactQQ
using .Encoding
using .IndicatorResolutions
using .HomExt
using .FlangeZn
using .M2SingularBridge
using .Serialization
using .Viz2D
using .PLPolyhedra
using .CrossValidateFlangePL

if CoreModules.ENABLE_PL_AXIS
    using .PLBackend
end

# Bring specific symbols into our namespace so we can re-export them.
import .HomExt: build_hom_tot_complex, ext_dims_via_resolutions

# Miller-style compressed encodings for Z^n flanges
import .ZnEncoding: ZnEncodingMap,
                    encode_poset_from_flanges,
                    fringe_from_flange,
                    encode_from_flange,
                    encode_from_flanges

# Bring derived-functor API into umbrella namespace for a user-facing API.
using .DerivedFunctors:
    Hom, HomSpace,
    ProjectiveResolution, InjectiveResolution,
    projective_resolution, injective_resolution,
    betti, betti_table, bass, bass_table,

    # Ext API (bind into PosetModules so PM.Ext, PM.ExtInjective, PM.yoneda_product, etc. work)
    ExtSpaceProjective, ExtSpaceInjective, Ext, ExtInjective,
    ExtSpace, dim, basis,
    representative, cycles, boundaries, coordinates,
    ext_map_first, ext_map_second,
    connecting_hom, connecting_hom_first,
    yoneda_product,

    # Algebra structures / caching
    ExtLongExactSequenceSecond, ExtAlgebra, ExtElement, multiply, element, unit, precompute!,

    # Tor API
    tor_map_second, Tor,
    TorSpace, TorSpaceSecond, tor_map_first, tor_map_second,
    TorLongExactSequenceFirst, TorLongExactSequenceSecond,
    TorDoubleComplex, TorSpectralSequence, TorSpectralPage,
    ext_action_on_tor,
    TorAlgebra, TorElement, set_chain_product!, multiplication_matrix,

    # Wrappers for Z^n / R^n layers and boxing utilities
    ExtZn, ExtRn, pmodule_on_box,

    # The rest of the existing imports in this list:
    ExtDoubleComplex, ExtSpectralSequence,
    lift_injective_chainmap,
    ExtLongExactSequenceSecond, ExtSpace, ExtLongExactSequenceFirst,
    comparison_isomorphism, comparison_isomorphisms, 
    minimality_report,
    projective_model, injective_model,
    ProjectiveMinimalityReport, InjectiveMinimalityReport,
    minimality_report, is_minimal, assert_minimal,
    minimal_projective_resolution, minimal_injective_resolution,
    minimal_betti, minimal_bass

using .ModuleComplexes:
    ModuleCochainComplex, ModuleCochainMap, ModuleCochainHomotopy, 
    is_cochain_homotopy,
    ModuleDistinguishedTriangle, mapping_cone, mapping_cone_triangle,
    cohomology_module, cohomology_module_data, induced_map_on_cohomology_modules,
    is_quasi_isomorphism, RHomComplex, RHom, HyperExtSpace, hyperExt,
    DerivedTensorComplex, DerivedTensor, HyperTorSpace, hyperTor,
    rhom_map_first, rhom_map_second,
    hyperExt_map_first, hyperExt_map_second,
    DerivedTensorComplex, DerivedTensor, HyperTorSpace, hyperTor,
    derived_tensor_map_first, derived_tensor_map_second,
    hyperTor_map_first, hyperTor_map_second

# Invariants on finite encodings
using .Invariants


using .ChainComplexes: CochainComplex, CochainMap, shift, extend_range,
                       mapping_cone, mapping_cone_triangle,
                       DistinguishedTriangle, LongExactSequence, long_exact_sequence,
                       DoubleComplex, total_complex,
                       SpectralSequence, spectral_sequence, page, E_r, page_terms, differential, total_cohomology_dims,
                       SubquotientData, term,
                       dr_target, dr_source,
                       filtration_dims, filtration_basis, filtration_subquotient,
                       edge_inclusion, edge_projection, split_total_cohomology,
                       product_matrix, product_coords,
                       collapse_page, convergence_report,
                       cohomology_dims, homology_dims,
                       SpectralTermsPage,
                       E_r_terms, E2_terms,
                       page_terms_dict, page_dims_dict,
                       diagonal_criterion,
                       FiltrationData, filtration_data,
                       collapse_data,
                       ExtensionProblem, extension_problem


using .ChangeOfPosets

# Finite poset + fringe presentation
export FinitePoset, Upset, Downset, FringeModule,
       upset_from_generators, downset_from_generators, cover_edges, fiber_dimension,
       hom_dimension, dense_to_sparse_K

# Encoding / uptight posets
export EncodingMap, UptightEncoding,
       build_uptight_encoding_from_fringe, pullback_fringe_along_encoding,
       image_upset, image_downset, preimage_upset, preimage_downset

# Indicator one-step data (types)
export UpsetPresentation, DownsetCopresentation

# Indicator resolutions + Hom/Ext (API)
export upset_presentation_one_step, downset_copresentation_one_step,
       hom_ext_first_page, build_hom_tot_complex, ext_dims_via_resolutions,
       verify_upset_resolution, verify_downset_resolution,
       product, coproduct, biproduct,
       equalizer, coequalizer,
       AbstractDiagram, DiscretePairDiagram, ParallelPairDiagram, SpanDiagram, CospanDiagram,
       limit, colimit


# Zn flange + query + bridge
export Face, face, IndFlat, IndInj, Flat, Injective, Flange, canonical_matrix, dim_at, bounding_box,
       parse_flange_json, flange_from_m2

# Serialization and 2D viz
export save_flange_json, load_flange_json,
       save_encoding_json, load_encoding_json,
       draw_flange_regions2D, draw_constant_subdivision2D

# PL encoders + cross validation
export HPoly, PolyUnion, PLUpset, PLDownset, PLFringe, PLEncodingMap, 
       locate, region_weights, region_bbox, region_volume, region_diameter, make_hpoly,
       encode_from_PL_fringe, encode_from_PL_fringes, cross_validate,
       region_widths, region_centroid, region_aspect_ratio, region_adjacency,
       region_facet_count, region_vertex_count,
       region_boundary_measure, region_boundary_measure_breakdown,
       region_perimeter, region_surface_area,
       region_principal_directions,
       region_chebyshev_ball, region_chebyshev_center, region_inradius,
       region_circumradius,
       region_boundary_to_volume_ratio, region_isoperimetric_ratio,
       region_mean_width, region_minkowski_functionals,
       region_covariance_anisotropy, region_covariance_eccentricity, region_anisotropy_scores,
       PolyInBoxCache, poly_in_box_cache

export Hom, Ext, ExtInjective,
       ProjectiveResolution, InjectiveResolution,
       projective_resolution, injective_resolution,
       dim, basis, representative, cycles, boundaries, coordinates,
       ext_map_second, ext_map_first, connecting_hom,
       connecting_hom_first,
       yoneda_product,
       betti, betti_table,
       bass, bass_table,
       Tor, TorSpace,
       tor_map_first, tor_map_second,
       ExtZn, ExtRn, pmodule_on_box,
       encode_pmodule_from_flange, encode_pmodules_from_flanges,
       projective_resolution_Zn, injective_resolution_Zn,
       minimal_projective_resolution_Zn, minimal_injective_resolution_Zn,
       minimal_betti_Zn, minimal_bass_Zn,
       encode_pmodule_from_PL_fringe, encode_pmodules_from_PL_fringes,
       projective_resolution_Rn, injective_resolution_Rn,
       minimal_projective_resolution_Rn, minimal_injective_resolution_Rn,
       minimal_betti_Rn, minimal_bass_Rn,
       ExtAlgebra, ExtElement, lift_injective_chainmap, element, unit, multiply, precompute!,
       ProjectiveMinimalityReport, InjectiveMinimalityReport,
       minimality_report, is_minimal, assert_minimal,
       minimal_projective_resolution, minimal_injective_resolution,
       minimal_betti, minimal_bass,
       ExtDoubleComplex, ExtSpectralSequence, ExtLongExactSequenceSecond,
       ExtSpace, ExtLongExactSequenceFirst, comparison_isomorphism, 
       comparison_isomorphisms, projective_model, injective_model,
       SpectralTermsPage,
       E_r_terms, E2_terms,
       page_terms_dict, page_dims_dict,
       diagonal_criterion,
       FiltrationData, filtration_data,
       collapse_data,
       ExtensionProblem, extension_problem


export TorSpaceSecond,
       TorLongExactSequenceFirst, TorLongExactSequenceSecond,
       TorDoubleComplex, TorSpectralSequence,
       ext_action_on_tor,
       TorAlgebra, TorElement, set_chain_product!, multiplication_matrix

export zero_pmodule, zero_morphism, direct_sum, direct_sum_with_maps,
       kernel_with_inclusion, kernel,
       image_with_inclusion, image,
       cokernel_with_projection, cokernel,
       coimage_with_projection, coimage,
       quotient_with_projection, quotient,
       is_zero_morphism, is_monomorphism, is_epimorphism,
       Submodule, submodule, sub, ambient, inclusion,
       kernel_submodule, image_submodule,
       pushout, pullback,
       ShortExactSequence, short_exact_sequence, is_exact, assert_exact,
       snake_lemma, SnakeLemmaResult

export ModuleCochainComplex, ModuleCochainMap, ModuleCochainHomotopy, 
       is_cochain_homotopy,
       ModuleDistinguishedTriangle, mapping_cone, mapping_cone_triangle,
       cohomology_module, cohomology_module_data, induced_map_on_cohomology_modules,
       is_quasi_isomorphism, RHomComplex, RHom, HyperExtSpace, hyperExt,
       DerivedTensorComplex, DerivedTensor, HyperTorSpace, hyperTor,
       rhom_map_first, rhom_map_second,
       hyperExt_map_first, hyperExt_map_second,
       DerivedTensorComplex, DerivedTensor, HyperTorSpace, hyperTor,
       derived_tensor_map_first, derived_tensor_map_second,
       hyperTor_map_first, hyperTor_map_second


# Miller-style finite encodings for Z^n flanges (do not enumerate a box)
export ZnEncodingMap,
       encode_poset_from_flanges,
       fringe_from_flange,
       encode_from_flange,
       encode_from_flanges

# Finite-encoding invariants (exports from Invariants.jl)
for sym in names(Invariants; all=false, imported=false)
    sym === :Invariants && continue
    @eval export $sym
end

export CochainComplex, CochainMap, shift, extend_range,
       mapping_cone, mapping_cone_triangle,
       DistinguishedTriangle, LongExactSequence, long_exact_sequence,
       DoubleComplex, total_complex,
       SpectralSequence, spectral_sequence, page, E_r, page_terms, differential,
       SubquotientData, term,
       dr_target, dr_source,
       filtration_dims, filtration_basis, filtration_subquotient,
       edge_inclusion, edge_projection, split_total_cohomology,
       product_matrix, product_coords,
       collapse_page, convergence_report,
       total_cohomology_dims,
       cohomology_dims, homology_dims

export restriction,
       pushforward_left, pushforward_right,
       left_kan_extension, right_kan_extension,
       Lpushforward_left, Rpushforward_right,
       derived_pushforward_left, derived_pushforward_right,
       pushforward_left_complex, pushforward_right_complex,
       restriction, pushforward_left, pushforward_right,
       Lpushforward_left, Rpushforward_right,
       pushforward_left_complex, pushforward_right_complex,
       product_poset, encode_pmodules_to_common_poset

# Optional experimental (axis-aligned PL) exports
if CoreModules.ENABLE_PL_AXIS
    export BoxUpset, BoxDownset, PLEncodingMapBoxes, encode_fringe_boxes
end

end # module
