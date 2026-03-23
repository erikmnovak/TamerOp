# public_api.jl -- owner-level reexports, caches, and convenience wrappers

# -----------------------------------------------------------------------------
# Public surface reexports (parent module)
# -----------------------------------------------------------------------------

using .Utils: compose
import ..IndicatorResolutions: resolution_length
import ..ChainComplexes: describe, sequence_dimensions, sequence_maps, sequence_entry, spectral_sequence_summary

import .Resolutions:
    ProjectiveResolution, InjectiveResolution,
    projective_resolution, injective_resolution,
    betti, betti_table, bass, bass_table,
    minimality_report,
    ProjectiveMinimalityReport, InjectiveMinimalityReport,
    is_minimal, assert_minimal,
    lift_injective_chainmap,
    _coeff_matrix_upsets,
    _flatten_gens_at,
    _solve_downset_postcompose_coeff

import .ExtTorSpaces:
    Hom, HomSpace,
    degree_range,
    ExtSpaceProjective, ExtSpaceInjective, ExtSpace,
    Ext, ExtInjective,
    Tor, TorSpace, TorSpaceSecond,
    dim, basis, representative, cycles, boundaries, coordinates,
    comparison_isomorphism, comparison_isomorphisms,
    projective_model, injective_model,
    hom_ext_first_page, ext_dimensions_via_indicator_resolutions

import .Functoriality:
    ext_map_first, ext_map_second,
    tor_map_first, tor_map_second,
    connecting_hom, connecting_hom_first,
    ExtLongExactSequenceSecond, ExtLongExactSequenceFirst,
    TorLongExactSequenceFirst, TorLongExactSequenceSecond,
    _FUNCTORIALITY_USE_HOM_WORKSPACES,
    _PrecomposeWorkspace,
    _PostcomposeWorkspace,
    _precompose_matrix,
    _postcompose_matrix,
    _precompose_on_hom_cochains_from_projective_coeff,
    _tensor_map_on_tor_chains_from_projective_coeff,
    _tor_blockdiag_map_on_chains

import .Algebras:
    yoneda_product,
    ExtAlgebra, ExtElement,
    multiply, element, unit, precompute!,
    TorAlgebra, TorElement,
    set_chain_product!, set_chain_product_generator!,
    multiplication_matrix,
    trivial_tor_product_generator,
    ext_action_on_tor

import .SpectralSequences:
    ExtDoubleComplex, ExtSpectralSequence,
    TorDoubleComplex, TorSpectralSequence,
    TorSpectralPage

import .Backends:
    ExtZn, ExtRn,
    pmodule_on_box,
    projective_resolution_Zn, injective_resolution_Zn,
    projective_resolution_Rn, injective_resolution_Rn

using .HomExtEngine:
    build_hom_tot_complex,
    build_hom_bicomplex_data,
    ext_dims_via_resolutions, pi0_count

@inline function HomSystemCache(::Type{K}) where {K}
    MT = SparseMatrixCSC{K,Int}
    return HomSystemCache(HomSpace{K}, MT, MT)
end

HomSystemCache{K}() where {K} = HomSystemCache(K)

@inline _hom_with_cache(M::PModule{K}, N::PModule{K}, ::Nothing) where {K} = Hom(M, N)

function _hom_with_cache(
    M::PModule{K},
    N::PModule{K},
    cache::HomSystemCache{HomSpace{K},SparseMatrixCSC{K,Int},SparseMatrixCSC{K,Int}},
) where {K}
    key = _cache_key2(M, N)
    cached = _cache_lookup(cache.hom, key)
    cached === nothing || return cached
    H = Hom(M, N)
    return _cache_store_or_get!(cache.hom, key, H)
end

function _hom_with_cache(M::PModule{K}, N::PModule{K}, ::HomSystemCache) where {K}
    error("hom_with_cache: cache scalar type mismatch for coefficient type $(K).")
end

function hom_with_cache(M::PModule{K}, N::PModule{K}; cache::Union{Nothing,HomSystemCache}=nothing) where {K}
    return _hom_with_cache(M, N, cache)
end

@inline function _precompose_cached(Hdom::HomSpace{K}, Hcod::HomSpace{K}, f::PMorphism{K}, ::Nothing) where {K}
    return sparse(_precompose_matrix(Hdom, Hcod, f))
end

function _precompose_cached(
    Hdom::HomSpace{K},
    Hcod::HomSpace{K},
    f::PMorphism{K},
    cache::HomSystemCache{HomSpace{K},SparseMatrixCSC{K,Int},SparseMatrixCSC{K,Int}},
) where {K}
    key = _cache_key3(Hdom, Hcod, f)
    cached = _cache_lookup(cache.precompose, key)
    cached === nothing || return cached
    F = sparse(_precompose_matrix(Hdom, Hcod, f))
    return _cache_store_or_get!(cache.precompose, key, F)
end

function _precompose_cached(Hdom::HomSpace{K}, Hcod::HomSpace{K}, f::PMorphism{K}, ::HomSystemCache) where {K}
    error("precompose_matrix_cached: cache scalar type mismatch for coefficient type $(K).")
end

function precompose_matrix_cached(Hdom::HomSpace{K}, Hcod::HomSpace{K}, f::PMorphism{K}; cache::Union{Nothing,HomSystemCache}=nothing) where {K}
    return _precompose_cached(Hdom, Hcod, f, cache)
end

@inline function _postcompose_cached(Hdom::HomSpace{K}, Hcod::HomSpace{K}, g::PMorphism{K}, ::Nothing) where {K}
    return sparse(_postcompose_matrix(Hdom, Hcod, g))
end

function _postcompose_cached(
    Hdom::HomSpace{K},
    Hcod::HomSpace{K},
    g::PMorphism{K},
    cache::HomSystemCache{HomSpace{K},SparseMatrixCSC{K,Int},SparseMatrixCSC{K,Int}},
) where {K}
    key = _cache_key3(Hdom, Hcod, g)
    cached = _cache_lookup(cache.postcompose, key)
    cached === nothing || return cached
    F = sparse(_postcompose_matrix(Hdom, Hcod, g))
    return _cache_store_or_get!(cache.postcompose, key, F)
end

function _postcompose_cached(Hdom::HomSpace{K}, Hcod::HomSpace{K}, g::PMorphism{K}, ::HomSystemCache) where {K}
    error("postcompose_matrix_cached: cache scalar type mismatch for coefficient type $(K).")
end

function postcompose_matrix_cached(Hdom::HomSpace{K}, Hcod::HomSpace{K}, g::PMorphism{K}; cache::Union{Nothing,HomSystemCache}=nothing) where {K}
    return _postcompose_cached(Hdom, Hcod, g, cache)
end

# -----------------------------------------------------------------------------
# Public opts-default wrappers
# -----------------------------------------------------------------------------

Hom(M, N; cache::Union{Nothing,HomSystemCache}=nothing) =
    hom_with_cache(M, N; cache=cache)

projective_resolution(M; opts::ResolutionOptions=ResolutionOptions(), cache=nothing) =
    projective_resolution(M, opts; cache=cache)
injective_resolution(M; opts::ResolutionOptions=ResolutionOptions(), cache=nothing) =
    injective_resolution(M, opts; cache=cache)
betti(M; opts::ResolutionOptions=ResolutionOptions()) =
    betti(M, opts)
bass(M; opts::ResolutionOptions=ResolutionOptions()) =
    bass(M, opts)

Ext(M, N; opts::DerivedFunctorOptions=DerivedFunctorOptions(), cache=nothing) =
    Ext(M, N, opts; cache=cache)
ExtInjective(M, N; opts::DerivedFunctorOptions=DerivedFunctorOptions(), cache=nothing) =
    ExtInjective(M, N, opts; cache=cache)
ExtSpace(M, N; opts::DerivedFunctorOptions=DerivedFunctorOptions(), check::Bool=true, cache=nothing) =
    ExtSpace(M, N, opts; check=check, cache=cache)
Tor(Rop, L; opts::DerivedFunctorOptions=DerivedFunctorOptions(), res=nothing, cache=nothing) =
    Tor(Rop, L, opts; res=res, cache=cache)
ExtAlgebra(M; opts::DerivedFunctorOptions=DerivedFunctorOptions()) =
    ExtAlgebra(M, opts)
ext_action_on_tor(A, T, x; opts::DerivedFunctorOptions=DerivedFunctorOptions()) =
    ext_action_on_tor(A, T, x, opts)

ExtDoubleComplex(M, N; opts::ResolutionOptions=ResolutionOptions(), cache=nothing) =
    ExtDoubleComplex(M, N, opts; cache=cache)
ExtSpectralSequence(M, N; opts::ResolutionOptions=ResolutionOptions(), first::Symbol=:vertical, cache=nothing) =
    ExtSpectralSequence(M, N, opts; first=first, cache=cache)
TorSpectralSequence(Rop, L; maxlen=nothing, maxlenR=nothing, maxlenL=nothing,
                    first::Symbol=:vertical, cache=nothing) =
    TorSpectralSequence(Rop, L; maxlen=maxlen, maxlenR=maxlenR, maxlenL=maxlenL,
                        first=first, cache=cache)

ExtZn(FG1, FG2; enc::EncodingOptions=EncodingOptions(), df::DerivedFunctorOptions=DerivedFunctorOptions(), kwargs...) =
    ExtZn(FG1, FG2, enc, df; kwargs...)
ExtRn(F1, F2; enc::EncodingOptions=EncodingOptions(), df::DerivedFunctorOptions=DerivedFunctorOptions()) =
    ExtRn(F1, F2, enc, df)

ExtLongExactSequenceSecond(M, A, B, C, i, p; opts::DerivedFunctorOptions=DerivedFunctorOptions()) =
    ExtLongExactSequenceSecond(M, A, B, C, i, p, opts)
ExtLongExactSequenceSecond(M, ses; opts::DerivedFunctorOptions=DerivedFunctorOptions()) =
    ExtLongExactSequenceSecond(M, ses, opts)

ExtLongExactSequenceFirst(A, B, C, N, i, p; opts::DerivedFunctorOptions=DerivedFunctorOptions()) =
    ExtLongExactSequenceFirst(A, B, C, N, i, p, opts)
ExtLongExactSequenceFirst(ses, N; opts::DerivedFunctorOptions=DerivedFunctorOptions()) =
    ExtLongExactSequenceFirst(ses, N, opts)

TorLongExactSequenceSecond(Rop, i, p; opts::DerivedFunctorOptions=DerivedFunctorOptions()) =
    TorLongExactSequenceSecond(Rop, i, p, opts)
TorLongExactSequenceSecond(Rop, ses; opts::DerivedFunctorOptions=DerivedFunctorOptions()) =
    TorLongExactSequenceSecond(Rop, ses, opts)

TorLongExactSequenceFirst(L, i, p; opts::DerivedFunctorOptions=DerivedFunctorOptions()) =
    TorLongExactSequenceFirst(L, i, p, opts)
TorLongExactSequenceFirst(L, ses; opts::DerivedFunctorOptions=DerivedFunctorOptions()) =
    TorLongExactSequenceFirst(L, ses, opts)

projective_resolution_Zn(FG; enc::EncodingOptions=EncodingOptions(), res::ResolutionOptions=ResolutionOptions(), return_encoding::Bool=false,
                         threads::Bool = (Threads.nthreads() > 1)) =
    projective_resolution_Zn(FG, enc, res;
                             return_encoding=return_encoding, threads=threads)
injective_resolution_Zn(FG; enc::EncodingOptions=EncodingOptions(), res::ResolutionOptions=ResolutionOptions(), return_encoding::Bool=false,
                        threads::Bool = (Threads.nthreads() > 1)) =
    injective_resolution_Zn(FG, enc, res;
                            return_encoding=return_encoding, threads=threads)

projective_resolution_Rn(FG; enc::EncodingOptions=EncodingOptions(), res::ResolutionOptions=ResolutionOptions(), return_encoding::Bool=false,
                         threads::Bool = (Threads.nthreads() > 1)) =
    projective_resolution_Rn(FG, enc, res;
                             return_encoding=return_encoding, threads=threads)
injective_resolution_Rn(FG; enc::EncodingOptions=EncodingOptions(), res::ResolutionOptions=ResolutionOptions(), return_encoding::Bool=false,
                        threads::Bool = (Threads.nthreads() > 1)) =
    injective_resolution_Rn(FG, enc, res;
                            return_encoding=return_encoding, threads=threads)

# -----------------------------------------------------------------------------
# Shared describe(...) bridge into ChainComplexes
# -----------------------------------------------------------------------------

@inline function _describe_derived_element(kind::Symbol, x)
    coords = element_coordinates(x)
    return (
        kind=kind,
        field=algebra_field(parent_algebra(x)),
        degree=element_degree(x),
        coordinate_length=length(coords),
        nonzero_coordinates=count(y -> !iszero(y), coords),
        is_zero=all(iszero, coords),
    )
end

describe(res::ProjectiveResolution) = resolution_summary(res)
describe(res::InjectiveResolution) = resolution_summary(res)
describe(H::HomSpace) = hom_summary(H)
describe(E::ExtSpaceProjective) = ext_summary(E)
describe(E::ExtSpaceInjective) = ext_summary(E)
describe(E::ExtSpace) = ext_summary(E)
describe(T::TorSpace) = tor_summary(T)
describe(T::TorSpaceSecond) = tor_summary(T)
describe(A::ExtAlgebra) = algebra_summary(A)
describe(A::TorAlgebra) = algebra_summary(A)
describe(x::ExtElement) = _describe_derived_element(:ext_element, x)
describe(x::TorElement) = _describe_derived_element(:tor_element, x)
describe(les::ExtLongExactSequenceSecond) = derived_les_summary(les)
describe(les::ExtLongExactSequenceFirst) = derived_les_summary(les)
describe(les::TorLongExactSequenceSecond) = derived_les_summary(les)
describe(les::TorLongExactSequenceFirst) = derived_les_summary(les)
describe(TSS::TorSpectralSequence) = spectral_sequence_summary(TSS)
