"""
    ChangeOfPosets

Change-of-poset functors for modules on finite posets.

This owner module sits one layer above finite encodings and generic module
algebra. It owns:
- pullback / restriction along monotone maps,
- left and right Kan extension along monotone maps,
- common-refinement Hom spaces used to compare modules on different posets,
- derived change-of-poset object-level translations implemented in this owner.

It does not own:
- finite encodings themselves (`Encoding`, `ZnEncoding`, `PLPolyhedra`,
  `PLBackend`),
- generic module algebra (`Modules`, `AbelianCategories`),
- generic derived-functor infrastructure (`DerivedFunctors`,
  `ModuleComplexes`).

Canonical user entrypoints:
- [`restriction`](@ref) / [`pullback`](@ref),
- [`pushforward_left`](@ref) / [`left_kan_extension`](@ref),
- [`pushforward_right`](@ref) / [`right_kan_extension`](@ref),
- [`product_poset`](@ref),
- [`encode_pmodules_to_common_poset`](@ref),
- [`hom_common_refinement`](@ref),
- [`common_refinement_summary`](@ref),
- [`check_monotone_map`](@ref) / [`check_common_refinement_hom`](@ref),
- [`derived_pushforward_left`](@ref),
- [`derived_pushforward_right`](@ref).

Notation policy:
- prefer [`restriction`](@ref), [`pushforward_left`](@ref),
  [`pushforward_right`](@ref), [`derived_pushforward_left`](@ref), and
  [`derived_pushforward_right`](@ref) in notebook-facing code,
- treat [`pullback`](@ref), [`left_kan_extension`](@ref), and
  [`right_kan_extension`](@ref) as the categorical aliases for the same
  operations.

Contributor note:
- monotone-map utilities live near the top of this file,
- pullback/restriction owns the translation of modules and morphisms along
  monotone maps,
- left/right Kan code owns the pushforward kernels,
- common-refinement Hom machinery owns the product-poset refinement and lazy Hom
  wrappers,
- derived wrappers own the object-level derived pushforward translations built
  on top of the non-derived kernels.
"""
module ChangeOfPosets

using LinearAlgebra
using SparseArrays

import Base.Threads
import ..FiniteFringe
using ..FiniteFringe: AbstractPoset, FinitePoset, ProductPoset, cover_edges, leq, leq_matrix, nvertices, poset_equal,
                      downset_indices, upset_indices, _preds, _succs, _pred_slots_of_succ
using ..Encoding: EncodingMap
using ..CoreModules: AbstractCoeffField, QQ, QQField, RealField, SessionCache, ProductPosetCacheEntry, _SessionProductKey,
                     EncodingCache, GeometryCachePayload, _encoding_cache!
using ..Options: ResolutionOptions, DerivedFunctorOptions
using ..FieldLinAlg
import ..FieldLinAlg: _SparseRREF, SparseRow, _SparseRowAccumulator,
                      _reset_sparse_row_accumulator!, _push_sparse_row_entry!,
                      _materialize_sparse_row!, _sparse_rref_push_homogeneous!,
                      _nullspace_from_pivots, _sparse_rows

const _CHANGE_OF_POSETS_USE_PRODUCT_PROJECTION_FASTPATH = Ref(true)
const _CHANGE_OF_POSETS_USE_FUSED_PRODUCT_TRANSLATION = Ref(true)
const _CHANGE_OF_POSETS_USE_DIRECT_HOM_COMMON_REFINEMENT = Ref(true)
const _CHANGE_OF_POSETS_USE_DERIVED_LEFT_COEFF_FASTPATH = Ref(true)
const _CHANGE_OF_POSETS_USE_DERIVED_RIGHT_COEFF_FASTPATH = Ref(false)
const _CHANGE_OF_POSETS_USE_RIGHT_KAN_SELECTOR_FASTPATH = Ref(false)
const _CHANGE_OF_POSETS_USE_RIGHT_KAN_QQ_SUMMARY_SELECTOR = Ref(true)
const _CHANGE_OF_POSETS_USE_RIGHT_KAN_DIRECT_CSC = Ref(true)
const _CHANGE_OF_POSETS_USE_LEFT_KAN_QQ_SUMMARY_QUOTIENT = Ref(false)
const _CHANGE_OF_POSETS_USE_LEFT_KAN_MORPHISM_MUL = Ref(false)
const _CHANGE_OF_POSETS_USE_LEFT_DERIVED_COEFF_PLAN_CACHE = Ref(true)
const _CHANGE_OF_POSETS_USE_LEFT_DERIVED_DIFFS_FROM_DATA = Ref(true)
const _CHANGE_OF_POSETS_USE_LEFT_DERIVED_COHOMOLOGY_SHORTCUT = Ref(true)
const _CHANGE_OF_POSETS_USE_RIGHT_DERIVED_COEFF_PLAN_CACHE = Ref(true)
const _CHANGE_OF_POSETS_USE_RIGHT_DERIVED_DIFFS_FROM_DATA = Ref(true)
const _CHANGE_OF_POSETS_USE_RIGHT_DERIVED_COEFF_MATRIX_RECURRENCE = Ref(true)
const _CHANGE_OF_POSETS_USE_RIGHT_DERIVED_SHARED_RESOLUTION_FASTPATH = Ref(true)
const _CHANGE_OF_POSETS_USE_RIGHT_DERIVED_IDENTITY_MAP_FASTPATH = Ref(true)

@inline function _eye(::Type{K}, n::Int) where {K}
    M = zeros(K, n, n)
    for i in 1:n
        M[i, i] = one(K)
    end
    return M
end

import ..IndicatorResolutions
using ..Modules: PModule, PMorphism, CoverEdgeMapStore, MapLeqQueryBatch, id_morphism,
                 map_leq, map_leq_many, _prepare_map_leq_batch_owned, _get_cover_cache
import ..AbelianCategories: pullback, _right_inverse_full_row

import ..DerivedFunctors
using ..DerivedFunctors: projective_resolution, injective_resolution,
                         lift_injective_chainmap, Hom, HomSpace

# lift_chainmap lives in DerivedFunctors.Functoriality.
using ..DerivedFunctors.Functoriality: lift_chainmap

import ..ModuleComplexes
using ..ModuleComplexes: ModuleCochainComplex, ModuleCochainMap,
                         cohomology_module, induced_map_on_cohomology_modules
import ..ChainComplexes: describe, source, target

# -----------------------------------------------------------------------------
# Validation summaries
# -----------------------------------------------------------------------------

"""
    MonotoneMapValidationSummary

Display-oriented validation summary for [`check_monotone_map`](@ref).

This wraps the monotone-map validation report so it prints compactly in the
REPL or notebooks while still exposing the underlying report fields through
property access such as `summary.valid` and `summary.issues`.
"""
struct MonotoneMapValidationSummary{R}
    report::R
end

"""
    CommonRefinementHomValidationSummary

Display-oriented validation summary for [`check_common_refinement_hom`](@ref).

This wraps the common-refinement Hom validation report so it prints compactly
without forcing users to inspect raw cache fields directly.
"""
struct CommonRefinementHomValidationSummary{R}
    report::R
end

"""
    monotone_map_validation_summary(report) -> MonotoneMapValidationSummary

Wrap a raw monotone-map validation report in a display-oriented summary object.
"""
@inline monotone_map_validation_summary(report::NamedTuple) = MonotoneMapValidationSummary(report)
@inline monotone_map_validation_summary(summary::MonotoneMapValidationSummary) = summary

"""
    common_refinement_hom_validation_summary(report) -> CommonRefinementHomValidationSummary

Wrap a raw common-refinement Hom validation report in a display-oriented
summary object.
"""
@inline common_refinement_hom_validation_summary(report::NamedTuple) = CommonRefinementHomValidationSummary(report)
@inline common_refinement_hom_validation_summary(summary::CommonRefinementHomValidationSummary) = summary

@inline Base.getproperty(summary::MonotoneMapValidationSummary, s::Symbol) =
    s === :report ? getfield(summary, :report) : getproperty(getfield(summary, :report), s)
@inline Base.getproperty(summary::CommonRefinementHomValidationSummary, s::Symbol) =
    s === :report ? getfield(summary, :report) : getproperty(getfield(summary, :report), s)

@inline Base.propertynames(summary::MonotoneMapValidationSummary, private::Bool=false) =
    private ? (:report, propertynames(getfield(summary, :report), true)...) : propertynames(getfield(summary, :report))
@inline Base.propertynames(summary::CommonRefinementHomValidationSummary, private::Bool=false) =
    private ? (:report, propertynames(getfield(summary, :report), true)...) : propertynames(getfield(summary, :report))

@inline describe(summary::MonotoneMapValidationSummary) = getfield(summary, :report)
@inline describe(summary::CommonRefinementHomValidationSummary) = getfield(summary, :report)

function Base.show(io::IO, summary::MonotoneMapValidationSummary)
    r = getfield(summary, :report)
    print(io, "MonotoneMapValidationSummary(valid=", r.valid,
          ", nsource=", r.nsource,
          ", ntarget=", r.ntarget,
          ", nissues=", length(r.issues), ")")
end

function Base.show(io::IO, summary::CommonRefinementHomValidationSummary)
    r = getfield(summary, :report)
    print(io, "CommonRefinementHomValidationSummary(valid=", r.valid,
          ", dimension=", r.dimension,
          ", translated=", r.translated_materialized,
          ", nissues=", length(r.issues), ")")
end

function Base.show(io::IO, ::MIME"text/plain", summary::MonotoneMapValidationSummary)
    r = getfield(summary, :report)
    print(io, "MonotoneMapValidationSummary",
          "\n  valid: ", r.valid,
          "\n  nsource: ", r.nsource,
          "\n  ntarget: ", r.ntarget,
          "\n  issues: ", isempty(r.issues) ? "none" : join(r.issues, "\n          "))
end

function Base.show(io::IO, ::MIME"text/plain", summary::CommonRefinementHomValidationSummary)
    r = getfield(summary, :report)
    print(io, "CommonRefinementHomValidationSummary",
          "\n  valid: ", r.valid,
          "\n  dimension: ", r.dimension,
          "\n  translated_materialized: ", r.translated_materialized,
          "\n  basis_matrix_materialized: ", r.basis_matrix_materialized,
          "\n  hom_materialized: ", r.hom_materialized,
          "\n  issues: ", isempty(r.issues) ? "none" : join(r.issues, "\n          "))
end

# -----------------------------------------------------------------------------
# Utilities: compatibility, monotonicity, terminal/initial detection
# -----------------------------------------------------------------------------

# Compare posets structurally.
@inline _same_poset(A::AbstractPoset, B::AbstractPoset)::Bool = poset_equal(A, B)

"""
    _check_monotone(pi)

Throw an error if `pi::EncodingMap` is not order-preserving.
It is enough to check cover edges, since they generate the order.
"""
function _check_monotone(pi::EncodingMap)
    Q = pi.Q
    P = pi.P
    f = pi.pi_of_q

    if length(f) != nvertices(Q)
        error("EncodingMap.pi_of_q must have length nvertices(Q)")
    end
    for q in 1:nvertices(Q)
        if f[q] < 1 || f[q] > nvertices(P)
            error("EncodingMap.pi_of_q values must lie in 1..nvertices(P)")
        end
    end

    for (u,v) in cover_edges(Q)
        if !leq(P, f[u], f[v])
            error("EncodingMap is not monotone: $u <= $v in Q but pi($u) !<= pi($v) in P")
        end
    end
    return nothing
end

"""
    check_monotone_map(pi::EncodingMap; throw=false) -> MonotoneMapValidationSummary

Validate that an [`EncodingMap`](@ref) is order-preserving.

This is the notebook-friendly validation surface for hand-built monotone maps
used by change-of-poset routines. It checks:
- `length(pi.pi_of_q) == nvertices(pi.Q)`,
- all image labels lie in `1:nvertices(pi.P)`,
- every cover edge of `pi.Q` maps to a comparable pair in `pi.P`.

Use `describe(report)` when you want the raw structured report back.
"""
function check_monotone_map(pi::EncodingMap; throw::Bool=false)
    issues = String[]
    Q = pi.Q
    P = pi.P
    f = pi.pi_of_q

    length(f) == nvertices(Q) ||
        push!(issues, "pi_of_q has length $(length(f)); expected $(nvertices(Q)) = nvertices(Q).")
    for q in 1:min(length(f), nvertices(Q))
        1 <= f[q] <= nvertices(P) || push!(issues, "pi($q) = $(f[q]) lies outside 1:$(nvertices(P)).")
    end
    if length(f) == nvertices(Q)
        for (u, v) in cover_edges(Q)
            if !(1 <= f[u] <= nvertices(P) && 1 <= f[v] <= nvertices(P))
                continue
            end
            leq(P, f[u], f[v]) || push!(issues, "cover edge $u <= $v in Q is not order-preserving under pi.")
        end
    end

    report = (;
        kind=:monotone_map,
        valid=isempty(issues),
        nsource=nvertices(Q),
        ntarget=nvertices(P),
        issues=issues,
    )
    if throw && !report.valid
        Base.throw(ArgumentError("check_monotone_map: " * join(report.issues, " ")))
    end
    return monotone_map_validation_summary(report)
end

"""
    _maximum_element(Q, S)

Return the maximum element of `S` (viewed as a subset of poset `Q`)
if it exists, otherwise return `nothing`.
"""
function _maximum_element(Q::AbstractPoset, S::Vector{Int})::Union{Int,Nothing}
    isempty(S) && return nothing
    cand = S[1]
    @inbounds for k in 2:length(S)
        s = S[k]
        if leq(Q, cand, s) && !leq(Q, s, cand)
            cand = s
        end
    end
    @inbounds for s in S
        if !leq(Q, s, cand)
            return nothing
        end
    end
    return cand
end

"""
    _minimum_element(Q, S)

Return the minimum element of `S` if it exists, else `nothing`.
"""
function _minimum_element(Q::AbstractPoset, S::Vector{Int})::Union{Int,Nothing}
    isempty(S) && return nothing
    cand = S[1]
    @inbounds for k in 2:length(S)
        s = S[k]
        if leq(Q, s, cand) && !leq(Q, cand, s)
            cand = s
        end
    end
    @inbounds for s in S
        if !leq(Q, cand, s)
            return nothing
        end
    end
    return cand
end

# -----------------------------------------------------------------------------
# Product posets and "common refinement" encoding for modules on different posets
# -----------------------------------------------------------------------------
#
# Motivation (Case C):
# If M1 is a PModule on P1 and M2 is a PModule on P2, the library's Hom(M1,M2)
# requires the SAME poset object. Mathematically, a standard way to compare them
# is to choose a common refinement poset and pull both modules back.
#
# The standard refinement is the cartesian product poset P = P1 x P2 with:
#   (i1,j1) <= (i2,j2)  iff  i1 <= i2 in P1  AND  j1 <= j2 in P2.
#
# Then we pull back M1 along pr1 : P -> P1 and M2 along pr2 : P -> P2, and compute
# Hom on the resulting common-poset modules.
#
# Performance notes:
#   * Constructing the product leq matrix is inherently O((n1*n2)^2) bits.
#   * Computing cover edges from scratch on the product is very expensive; we
#     pre-populate the per-poset cover cache using the known cover-edge structure of a
#     cartesian product poset:
#         covers are exactly "change one coordinate by a cover edge, keep the other fixed".
#   * Pullback along projections is implemented without calling map_leq on every edge
#     (which would allocate tons of identity matrices). We reuse identity matrices
#     and reuse the original cover-edge maps by reference.
#
# This is meant as a convenience layer for common-poset workflows:
#     out = encode_pmodules_to_common_poset(M1, M2)
#     H = Hom(translated_modules(out).left, translated_modules(out).right)

"""
    ProductPosetResult

Typed result object returned by [`product_poset`](@ref).

Mathematically, this stores the common product poset `P1 x P2` together with
its two projection maps back to the original factors.

Canonical data:
- `P` is the product poset,
- `pi1 : P -> P1` is the left projection,
- `pi2 : P -> P2` is the right projection.

This object stores only the mathematical output of [`product_poset`](@ref).
It does not expose the internal session-cache entries used to reuse product
constructions across calls.

Inspect this result via `describe(res)`, [`common_poset`](@ref), and
[`projection_maps`](@ref) rather than unpacking fields manually.
"""
struct ProductPosetResult{PType,Pi1Type,Pi2Type}
    P::PType
    pi1::Pi1Type
    pi2::Pi2Type
end

"""
    CommonRefinementTranslationResult

Typed result object returned by [`encode_pmodules_to_common_poset`](@ref).

Mathematically, this packages the common refinement poset, the translated
modules on that common poset, and the projection maps used to translate them.

Canonical data:
- `P` is the common refinement poset,
- `Ms` stores the two translated modules in the same order as the input
  modules,
- `pi1` and `pi2` are the refinement maps from `P` back to the original
  ambient posets.

This object stores only the canonical refinement data returned to the caller.
Translation plans, cache entries, and temporary product-translation kernels
remain internal implementation details.

Inspect this result via `describe(res)`, [`common_poset`](@ref),
[`projection_maps`](@ref), and [`translated_modules`](@ref).
"""
struct CommonRefinementTranslationResult{PType,MsType,Pi1Type,Pi2Type}
    P::PType
    Ms::MsType
    pi1::Pi1Type
    pi2::Pi2Type
end

"""
    common_poset(res::ProductPosetResult)
    common_poset(res::CommonRefinementTranslationResult)

Return the common finite poset carried by a `ChangeOfPosets` result object.
"""
@inline common_poset(res::ProductPosetResult) = res.P
@inline common_poset(res::CommonRefinementTranslationResult) = res.P

"""
    projection_maps(res::ProductPosetResult) -> NamedTuple
    projection_maps(res::CommonRefinementTranslationResult) -> NamedTuple

Return the two refinement/projection maps carried by a `ChangeOfPosets` result
object as `(left=..., right=...)`.
"""
@inline projection_maps(res::ProductPosetResult) = (; left=res.pi1, right=res.pi2)
@inline projection_maps(res::CommonRefinementTranslationResult) = (; left=res.pi1, right=res.pi2)

"""
    translated_modules(res::CommonRefinementTranslationResult) -> NamedTuple

Return the two translated modules produced by
[`encode_pmodules_to_common_poset`](@ref) as `(left=..., right=...)`.
"""
@inline translated_modules(res::CommonRefinementTranslationResult) = (; left=res.Ms[1], right=res.Ms[2])

@inline function _product_poset_result_describe(res::ProductPosetResult)
    return (;
        kind=:product_poset_result,
        common_nvertices=nvertices(res.P),
        left_nvertices=nvertices(res.pi1.P),
        right_nvertices=nvertices(res.pi2.P),
        common_poset_type=nameof(typeof(res.P)),
        left_poset_type=nameof(typeof(res.pi1.P)),
        right_poset_type=nameof(typeof(res.pi2.P)),
    )
end

@inline function _common_refinement_translation_describe(res::CommonRefinementTranslationResult)
    return (;
        kind=:common_refinement_translation_result,
        field=res.Ms[1].field,
        common_nvertices=nvertices(res.P),
        left_nvertices=nvertices(res.pi1.P),
        right_nvertices=nvertices(res.pi2.P),
        left_total_dim=_module_total_dim(res.Ms[1]),
        right_total_dim=_module_total_dim(res.Ms[2]),
        translated_on_common_poset=(res.Ms[1].Q === res.P) && (res.Ms[2].Q === res.P),
    )
end

describe(res::ProductPosetResult) = _product_poset_result_describe(res)
describe(res::CommonRefinementTranslationResult) = _common_refinement_translation_describe(res)

function Base.show(io::IO, res::ProductPosetResult)
    d = describe(res)
    print(io, "ProductPosetResult(common_nvertices=", d.common_nvertices,
          ", left_nvertices=", d.left_nvertices,
          ", right_nvertices=", d.right_nvertices, ")")
end

function Base.show(io::IO, res::CommonRefinementTranslationResult)
    d = describe(res)
    print(io, "CommonRefinementTranslationResult(field=", d.field,
          ", common_nvertices=", d.common_nvertices,
          ", left_total_dim=", d.left_total_dim,
          ", right_total_dim=", d.right_total_dim, ")")
end

function Base.show(io::IO, ::MIME"text/plain", res::ProductPosetResult)
    d = describe(res)
    print(io, "ProductPosetResult",
          "\n  common_nvertices: ", d.common_nvertices,
          "\n  left_nvertices: ", d.left_nvertices,
          "\n  right_nvertices: ", d.right_nvertices,
          "\n  common_poset_type: ", d.common_poset_type)
end

function Base.show(io::IO, ::MIME"text/plain", res::CommonRefinementTranslationResult)
    d = describe(res)
    print(io, "CommonRefinementTranslationResult",
          "\n  field: ", d.field,
          "\n  common_nvertices: ", d.common_nvertices,
          "\n  left_nvertices: ", d.left_nvertices,
          "\n  right_nvertices: ", d.right_nvertices,
          "\n  left_total_dim: ", d.left_total_dim,
          "\n  right_total_dim: ", d.right_total_dim,
          "\n  translated_on_common_poset: ", d.translated_on_common_poset)
end

Base.length(::ProductPosetResult) = 3
Base.length(::CommonRefinementTranslationResult) = 4

function Base.iterate(res::ProductPosetResult, state::Int=1)
    state == 1 && return (res.P, 2)
    state == 2 && return (res.pi1, 3)
    state == 3 && return (res.pi2, 4)
    return nothing
end

function Base.iterate(res::CommonRefinementTranslationResult, state::Int=1)
    state == 1 && return (res.P, 2)
    state == 2 && return (res.Ms, 3)
    state == 3 && return (res.pi1, 4)
    state == 4 && return (res.pi2, 5)
    return nothing
end

@inline _product_dense_cache(session_cache::Union{Nothing,SessionCache}) =
    session_cache === nothing ? nothing : session_cache.product_dense

@inline _product_obj_cache(session_cache::Union{Nothing,SessionCache}) =
    session_cache === nothing ? nothing : session_cache.product_obj

@inline _translation_cache(session_cache::Union{Nothing,SessionCache}, pi::EncodingMap) =
    session_cache === nothing ? nothing : _encoding_cache!(session_cache, UInt(objectid(pi)))

@inline function _translation_cache_get(cache::Union{Nothing,EncodingCache}, key)
    cache === nothing && return nothing
    Base.lock(cache.lock)
    try
        entry = get(cache.geometry, key, nothing)
        return entry === nothing ? nothing : entry.value
    finally
        Base.unlock(cache.lock)
    end
end

@inline function _translation_cache_set!(cache::Union{Nothing,EncodingCache}, key, value)
    cache === nothing && return value
    Base.lock(cache.lock)
    try
        cache.geometry[key] = GeometryCachePayload(value)
    finally
        Base.unlock(cache.lock)
    end
    return value
end

@inline _translation_module_key(tag::Symbol, pi::EncodingMap, M) =
    (tag, UInt(objectid(M)), UInt(objectid(pi.Q)), UInt(objectid(pi.P)))

@inline _translation_morphism_key(tag::Symbol, pi::EncodingMap, f) =
    (tag, UInt(objectid(f)), UInt(objectid(pi.Q)), UInt(objectid(pi.P)))

@inline _translation_complex_key(tag::Symbol, obj, res, maxdeg::Int) =
    (tag, UInt(objectid(obj)), UInt(objectid(res)), maxdeg)

@inline _translation_derived_map_key(tag::Symbol, f, res_dom, res_cod, maxdeg::Int) =
    (tag, UInt(objectid(f)), UInt(objectid(res_dom)), UInt(objectid(res_cod)), maxdeg)

@inline _translation_left_coeff_key(pi::EncodingMap, dom_bases, cod_bases) =
    (:left_derived_coeff_plan, UInt(objectid(dom_bases)), UInt(objectid(cod_bases)),
     UInt(objectid(pi.Q)), UInt(objectid(pi.P)))

@inline _translation_right_coeff_key(pi::EncodingMap, dom_bases, cod_bases) =
    (:right_derived_coeff_plan, UInt(objectid(dom_bases)), UInt(objectid(cod_bases)),
     UInt(objectid(pi.Q)), UInt(objectid(pi.P)))

# Pre-populate the FiniteFringe cover-edges cache for the cartesian product poset.
# This avoids the expensive generic cover-edge computation on P1 x P2.
function _cache_product_cover_edges!(Pprod::FinitePoset, P1::FinitePoset, P2::FinitePoset)
    n1, n2 = P1.n, P2.n
    n = n1 * n2

    C1 = cover_edges(P1)
    C2 = cover_edges(P2)

    # cover adjacency matrix + edge list
    mat = falses(n, n)
    edges = Vector{Tuple{Int,Int}}()
    sizehint!(edges, length(C1) * n2 + n1 * length(C2))

    @inbounds begin
        # Horizontal covers: (i1,j) <. (i2,j) for each cover i1<.i2 in P1, for each j.
        for (i1, i2) in C1
            base1 = (i1 - 1) * n2
            base2 = (i2 - 1) * n2
            for j in 1:n2
                u = base1 + j
                v = base2 + j
                mat[u, v] = true
                push!(edges, (u, v))
            end
        end

        # Vertical covers: (i,j1) <. (i,j2) for each cover j1<.j2 in P2, for each i.
        for i in 1:n1
            base = (i - 1) * n2
            for (j1, j2) in C2
                u = base + j1
                v = base + j2
                mat[u, v] = true
                push!(edges, (u, v))
            end
        end
    end

    # Pre-populate per-poset cover-edge cache (lazy cover-cache builder reuses this).
    FiniteFringe._set_cover_edges_cache!(Pprod, FiniteFringe.CoverEdges(mat, edges))
    return nothing
end

"""
    product_poset(P1::FinitePoset, P2::FinitePoset;
                  check=false, cache_cover_edges=true, use_cache=true,
                  session_cache=nothing) -> ProductPosetResult

Construct the cartesian product poset `P = P1 x P2` and package it in a
[`ProductPosetResult`](@ref).

Use [`common_poset`](@ref) and [`projection_maps`](@ref) for semantic
inspection of the result.

Index convention:
  Vertex `(i,j)` with `i in 1:P1.n`, `j in 1:P2.n` is stored at linear index
  `k = (i-1)*P2.n + j`.

Performance:
  - If `use_cache=true` and `check=false`, repeated calls with the same `P1` and `P2`
    objects reuse the previously constructed product poset when `session_cache` is set.
  - If `cache_cover_edges=true`, we also pre-fill the cover-edge cache for the product,
    which makes downstream homological algebra much faster.

# Examples

```jldoctest
julia> using TamerOp

julia> const CO = TamerOp.ChangeOfPosets;

julia> const FF = TamerOp.FiniteFringe;

julia> P1 = FF.FinitePoset(Bool[1 1; 0 1]);

julia> P2 = FF.FinitePoset(Bool[1]);

julia> out = CO.product_poset(P1, P2);

julia> TamerOp.describe(out).kind
:product_poset_result

julia> CO.common_poset(out) === out.P
true
```
"""
function product_poset(
    P1::FinitePoset,
    P2::FinitePoset;
    check::Bool = false,
    cache_cover_edges::Bool = true,
    use_cache::Bool = true,
    session_cache::Union{Nothing,SessionCache}=nothing,
)
    dense_cache = _product_dense_cache(session_cache)
    cache_enabled = use_cache && !check && cache_cover_edges && dense_cache !== nothing
    L1 = leq_matrix(P1)
    L2 = leq_matrix(P2)
    # Fast cache path (only for the common "production" settings).
    if cache_enabled
        cache_key = _SessionProductKey(L1, L2)
        entry = get(dense_cache, cache_key, nothing)
        if entry !== nothing
            return ProductPosetResult(entry.P, entry.pi1, entry.pi2)
        end
    end

    n1, n2 = P1.n, P2.n
    n = n1 * n2

    # Build leq matrix as a block matrix:
    #   block(i1,i2) = leq_matrix(P2) if leq(P1,i1,i2) else 0.
    L = falses(n, n)

    @views @inbounds for i1 in 1:n1
        rr = ((i1 - 1) * n2 + 1):(i1 * n2)
        row1 = L1[i1, :]
        i2 = findnext(row1, 1)
        while i2 !== nothing
            cc = ((i2 - 1) * n2 + 1):(i2 * n2)
            copyto!(L[rr, cc], L2)
            i2 = findnext(row1, i2 + 1)
        end
    end

    P = FinitePoset(L; check = check)

    # Projections P -> P1 and P -> P2 as EncodingMap objects.
    pi1_of_q = Vector{Int}(undef, n)
    pi2_of_q = Vector{Int}(undef, n)
    @inbounds for k in 1:n
        pi1_of_q[k] = div((k - 1), n2) + 1
        pi2_of_q[k] = (k - 1) % n2 + 1
    end

    pi1 = EncodingMap(P, P1, pi1_of_q)
    pi2 = EncodingMap(P, P2, pi2_of_q)

    if cache_cover_edges
        _cache_product_cover_edges!(P, P1, P2)
    end

    out = ProductPosetResult(P, pi1, pi2)

    if cache_enabled
        cache_key = _SessionProductKey(L1, L2)
        dense_cache[cache_key] = ProductPosetCacheEntry{Any,Any,Any,Any,Any}(L1, L2, out.P, out.pi1, out.pi2)
    end

    return out
end

function product_poset(
    P1::AbstractPoset,
    P2::AbstractPoset;
    check::Bool = false,
    cache_cover_edges::Bool = true,
    use_cache::Bool = true,
    session_cache::Union{Nothing,SessionCache}=nothing,
)
    obj_cache = _product_obj_cache(session_cache)
    cache_enabled = use_cache && obj_cache !== nothing
    if cache_enabled
        cache_key = _SessionProductKey(P1, P2)
        entry = get(obj_cache, cache_key, nothing)
        if entry !== nothing
            return ProductPosetResult(entry.P, entry.pi1, entry.pi2)
        end
    end

    n1, n2 = nvertices(P1), nvertices(P2)
    n = n1 * n2

    # Structured fallback: avoid materializing leq unless requested elsewhere.
    P = ProductPoset(P1, P2)

    # Projections P -> P1 and P -> P2 as EncodingMap objects.
    pi1_of_q = Vector{Int}(undef, n)
    pi2_of_q = Vector{Int}(undef, n)
    @inbounds for k in 1:n
        pi1_of_q[k] = ((k - 1) % n1) + 1
        pi2_of_q[k] = div((k - 1), n1) + 1
    end

    pi1 = EncodingMap(P, P1, pi1_of_q)
    pi2 = EncodingMap(P, P2, pi2_of_q)

    out = ProductPosetResult(P, pi1, pi2)
    if cache_enabled
        cache_key = _SessionProductKey(P1, P2)
        obj_cache[cache_key] = ProductPosetCacheEntry{Any,Any,Any,Any,Any}(P1, P2, out.P, out.pi1, out.pi2)
    end

    return out
end

@inline _dense_prod_decode(idx::Int, n2::Int) = (div(idx - 1, n2) + 1, ((idx - 1) % n2) + 1)
@inline _structured_prod_decode(idx::Int, n1::Int) = (((idx - 1) % n1) + 1, div(idx - 1, n1) + 1)

@inline function _identity_map_cached(
    ::Type{MatT},
    ::Type{K},
    cache::Dict{Int,MatT},
    d::Int,
) where {K,MatT<:AbstractMatrix{K}}
    return get!(cache, d) do
        convert(MatT, _eye(K, d))
    end
end

function _identity_maps_per_vertex(
    ::Type{MatT},
    ::Type{K},
    dims::Vector{Int},
) where {K,MatT<:AbstractMatrix{K}}
    cache = Dict{Int,MatT}()
    ids = Vector{MatT}(undef, length(dims))
    @inbounds for i in eachindex(dims)
        d = dims[i]
        ids[i] = get!(cache, d) do
            convert(MatT, _eye(K, d))
        end
    end
    return ids
end

function _cover_store_layout(Q::AbstractPoset)
    cc = _get_cover_cache(Q)
    n = nvertices(Q)
    preds = Vector{Vector{Int}}(undef, n)
    succs = Vector{Vector{Int}}(undef, n)
    pred_slots = Vector{Vector{Int}}(undef, n)
    @inbounds for v in 1:n
        preds[v] = collect(_preds(cc, v))
    end
    @inbounds for u in 1:n
        succs[u] = collect(_succs(cc, u))
        pred_slots[u] = collect(_pred_slots_of_succ(cc, u))
    end
    return preds, succs, pred_slots
end

function _build_product_projection_pair_dense(
    M1::PModule{K,F1,MatT1},
    M2::PModule{K,F2,MatT2},
    Pprod::FinitePoset,
    P1::FinitePoset,
    P2::FinitePoset,
) where {K,F1,F2,MatT1<:AbstractMatrix{K},MatT2<:AbstractMatrix{K}}
    n1, n2 = P1.n, P2.n
    n = n1 * n2
    dims1 = Vector{Int}(undef, n)
    dims2 = Vector{Int}(undef, n)
    @inbounds for i in 1:n1
        d1 = M1.dims[i]
        base = (i - 1) * n2
        for j in 1:n2
            idx = base + j
            dims1[idx] = d1
            dims2[idx] = M2.dims[j]
        end
    end

    preds, succs, pred_slots = _cover_store_layout(Pprod)
    maps_from_pred1 = [Vector{MatT1}(undef, length(preds[v])) for v in 1:n]
    maps_from_pred2 = [Vector{MatT2}(undef, length(preds[v])) for v in 1:n]
    maps_to_succ1 = [Vector{MatT1}(undef, length(succs[u])) for u in 1:n]
    maps_to_succ2 = [Vector{MatT2}(undef, length(succs[u])) for u in 1:n]
    id_maps1 = _identity_maps_per_vertex(MatT1, K, M1.dims)
    id_maps2 = _identity_maps_per_vertex(MatT2, K, M2.dims)

    @inbounds for u in 1:n
        iu, ju = _dense_prod_decode(u, n2)
        su = succs[u]
        slots = pred_slots[u]
        mu1 = maps_to_succ1[u]
        mu2 = maps_to_succ2[u]
        for j in eachindex(su)
            v = su[j]
            slot = slots[j]
            iv, jv = _dense_prod_decode(v, n2)
            if ju == jv
                A1 = M1.edge_maps[iu, iv]
                A2 = id_maps2[ju]
                mu1[j] = A1
                mu2[j] = A2
                maps_from_pred1[v][slot] = A1
                maps_from_pred2[v][slot] = A2
            else
                A1 = id_maps1[iu]
                A2 = M2.edge_maps[ju, jv]
                mu1[j] = A1
                mu2[j] = A2
                maps_from_pred1[v][slot] = A1
                maps_from_pred2[v][slot] = A2
            end
        end
    end

    nedges = sum(length, succs)
    store1 = CoverEdgeMapStore{K,MatT1}(preds, succs, maps_from_pred1, maps_to_succ1, nedges)
    store2 = CoverEdgeMapStore{K,MatT2}(preds, succs, maps_from_pred2, maps_to_succ2, nedges)
    return (
        PModule{K}(Pprod, dims1, store1; field=M1.field),
        PModule{K}(Pprod, dims2, store2; field=M2.field),
    )
end

function _build_product_projection_pair_structured(
    M1::PModule{K,F1,MatT1},
    M2::PModule{K,F2,MatT2},
    Pprod::ProductPoset,
) where {K,F1,F2,MatT1<:AbstractMatrix{K},MatT2<:AbstractMatrix{K}}
    P1 = Pprod.P1
    P2 = Pprod.P2
    n1, n2 = nvertices(P1), nvertices(P2)
    n = n1 * n2
    dims1 = Vector{Int}(undef, n)
    dims2 = Vector{Int}(undef, n)
    @inbounds for j in 1:n2
        d2 = M2.dims[j]
        base = (j - 1) * n1
        for i in 1:n1
            idx = base + i
            dims1[idx] = M1.dims[i]
            dims2[idx] = d2
        end
    end

    preds, succs, pred_slots = _cover_store_layout(Pprod)
    maps_from_pred1 = [Vector{MatT1}(undef, length(preds[v])) for v in 1:n]
    maps_from_pred2 = [Vector{MatT2}(undef, length(preds[v])) for v in 1:n]
    maps_to_succ1 = [Vector{MatT1}(undef, length(succs[u])) for u in 1:n]
    maps_to_succ2 = [Vector{MatT2}(undef, length(succs[u])) for u in 1:n]
    id_maps1 = _identity_maps_per_vertex(MatT1, K, M1.dims)
    id_maps2 = _identity_maps_per_vertex(MatT2, K, M2.dims)

    @inbounds for u in 1:n
        iu, ju = _structured_prod_decode(u, n1)
        su = succs[u]
        slots = pred_slots[u]
        mu1 = maps_to_succ1[u]
        mu2 = maps_to_succ2[u]
        for j in eachindex(su)
            v = su[j]
            slot = slots[j]
            iv, jv = _structured_prod_decode(v, n1)
            if ju == jv
                A1 = M1.edge_maps[iu, iv]
                A2 = id_maps2[ju]
                mu1[j] = A1
                mu2[j] = A2
                maps_from_pred1[v][slot] = A1
                maps_from_pred2[v][slot] = A2
            else
                A1 = id_maps1[iu]
                A2 = M2.edge_maps[ju, jv]
                mu1[j] = A1
                mu2[j] = A2
                maps_from_pred1[v][slot] = A1
                maps_from_pred2[v][slot] = A2
            end
        end
    end

    nedges = sum(length, succs)
    store1 = CoverEdgeMapStore{K,MatT1}(preds, succs, maps_from_pred1, maps_to_succ1, nedges)
    store2 = CoverEdgeMapStore{K,MatT2}(preds, succs, maps_from_pred2, maps_to_succ2, nedges)
    return (
        PModule{K}(Pprod, dims1, store1; field=M1.field),
        PModule{K}(Pprod, dims2, store2; field=M2.field),
    )
end

function _pullback_product_projection_dense(
    M::PModule{K,F,MatT},
    Pprod::FinitePoset,
    P1::FinitePoset,
    P2::FinitePoset,
    kind::Int,
) where {K,F,MatT<:AbstractMatrix{K}}
    n1, n2 = P1.n, P2.n
    n = n1 * n2
    dims_out = Vector{Int}(undef, n)
    if kind == 1
        @inbounds for i in 1:n1
            di = M.dims[i]
            base = (i - 1) * n2
            for j in 1:n2
                dims_out[base + j] = di
            end
        end
    else
        @inbounds for i in 1:n1
            base = (i - 1) * n2
            for j in 1:n2
                dims_out[base + j] = M.dims[j]
            end
        end
    end

    preds, succs, pred_slots = _cover_store_layout(Pprod)
    maps_from_pred = [Vector{MatT}(undef, length(preds[v])) for v in 1:n]
    maps_to_succ = [Vector{MatT}(undef, length(succs[u])) for u in 1:n]
    id_maps = _identity_maps_per_vertex(MatT, K, M.dims)
    @inbounds for u in 1:n
        iu, ju = _dense_prod_decode(u, n2)
        su = succs[u]
        slots = pred_slots[u]
        mu = maps_to_succ[u]
        for j in eachindex(su)
            v = su[j]
            slot = slots[j]
            iv, jv = _dense_prod_decode(v, n2)
            A = if kind == 1
                ju == jv ? M.edge_maps[iu, iv] : id_maps[iu]
            else
                ju == jv ? id_maps[ju] : M.edge_maps[ju, jv]
            end
            mu[j] = A
            maps_from_pred[v][slot] = A
        end
    end
    nedges = sum(length, succs)
    store = CoverEdgeMapStore{K,MatT}(preds, succs, maps_from_pred, maps_to_succ, nedges)
    return PModule{K}(Pprod, dims_out, store; field=M.field)
end

function _pullback_to_product_pr1(
    M::PModule{K,F,MatT},
    Pprod::FinitePoset,
    P1::FinitePoset,
    P2::FinitePoset,
    C1,
    C2,
) where {K,F,MatT<:AbstractMatrix{K}}
    return _pullback_product_projection_dense(M, Pprod, P1, P2, 1)
end

function _pullback_to_product_pr2(
    M::PModule{K,F,MatT},
    Pprod::FinitePoset,
    P1::FinitePoset,
    P2::FinitePoset,
    C1,
    C2,
) where {K,F,MatT<:AbstractMatrix{K}}
    return _pullback_product_projection_dense(M, Pprod, P1, P2, 2)
end

@inline function _product_projection_kind(pi::EncodingMap)
    Q = pi.Q
    Q isa ProductPoset || return 0
    n1 = nvertices(Q.P1)
    n = nvertices(Q)
    if pi.P === Q.P1
        @inbounds for q in 1:n
            pi.pi_of_q[q] == ((q - 1) % n1) + 1 || return 0
        end
        return 1
    elseif pi.P === Q.P2
        @inbounds for q in 1:n
            pi.pi_of_q[q] == div(q - 1, n1) + 1 || return 0
        end
        return 2
    end
    return 0
end

function _pullback_product_projection_structured(
    pi::EncodingMap,
    M::PModule{K,F,MatT},
    kind::Int,
) where {K,F,MatT<:AbstractMatrix{K}}
    Q = pi.Q::ProductPoset
    n1 = nvertices(Q.P1)
    n2 = nvertices(Q.P2)
    n = n1 * n2
    dims_out = Vector{Int}(undef, n)
    if kind == 1
        @inbounds for j in 1:n2
            base = (j - 1) * n1
            for i in 1:n1
                dims_out[base + i] = M.dims[i]
            end
        end
    else
        @inbounds for j in 1:n2
            dj = M.dims[j]
            base = (j - 1) * n1
            for i in 1:n1
                dims_out[base + i] = dj
            end
        end
    end

    preds, succs, pred_slots = _cover_store_layout(Q)
    maps_from_pred = [Vector{MatT}(undef, length(preds[v])) for v in 1:n]
    maps_to_succ = [Vector{MatT}(undef, length(succs[u])) for u in 1:n]
    id_maps = _identity_maps_per_vertex(MatT, K, M.dims)
    @inbounds for u in 1:n
        iu, ju = _structured_prod_decode(u, n1)
        su = succs[u]
        slots = pred_slots[u]
        mu = maps_to_succ[u]
        for j in eachindex(su)
            v = su[j]
            slot = slots[j]
            iv, jv = _structured_prod_decode(v, n1)
            A = if kind == 1
                ju == jv ? M.edge_maps[iu, iv] : id_maps[iu]
            else
                ju == jv ? id_maps[ju] : M.edge_maps[ju, jv]
            end
            mu[j] = A
            maps_from_pred[v][slot] = A
        end
    end
    nedges = sum(length, succs)
    store = CoverEdgeMapStore{K,MatT}(preds, succs, maps_from_pred, maps_to_succ, nedges)
    return PModule{K}(Q, dims_out, store; field=M.field)
end

"""
    encode_pmodules_to_common_poset(M1::PModule, M2::PModule;
        method=:product, check_poset=false, cache_cover_edges=true, use_cache=true) -> CommonRefinementTranslationResult

Turn "Case C" into the same UX pattern as Cases A/B.

Given:
  - `M1` a poset module on poset `P1 = M1.Q`,
  - `M2` a poset module on poset `P2 = M2.Q`,

return a [`CommonRefinementTranslationResult`](@ref) whose semantic accessors
are:
- [`common_poset`](@ref) for the refinement poset,
- [`translated_modules`](@ref) for the two translated modules,
- [`projection_maps`](@ref) for the refinement maps.

Default refinement (`method=:product`):
  P = P1 x P2 (cartesian product poset).

Special case (performance + usability):
  If P1 and P2 have identical leq matrices (even if they are different objects),
  this function rebases M2 onto P1 so that Hom works without blowing up to a product.

Typical usage:
    out = encode_pmodules_to_common_poset(M1, M2)
    mods = translated_modules(out)
    H = Hom(mods.left, mods.right)
    d = dim(H)

# Examples

```jldoctest
julia> using TamerOp

julia> const CO = TamerOp.ChangeOfPosets;

julia> const FF = TamerOp.FiniteFringe;

julia> const IR = TamerOp.IndicatorResolutions;

julia> field = TamerOp.CoreModules.QQField();

julia> P1 = FF.FinitePoset(Bool[1 1; 0 1]);

julia> P2 = FF.FinitePoset(Bool[1]);

julia> M1 = IR.pmodule_from_fringe(FF.one_by_one_fringe(P1,
                                                       FF.principal_upset(P1, 1),
                                                       FF.principal_downset(P1, 2),
                                                       1//1; field=field));

julia> M2 = IR.pmodule_from_fringe(FF.one_by_one_fringe(P2,
                                                       FF.principal_upset(P2, 1),
                                                       FF.principal_downset(P2, 1),
                                                       1//1; field=field));

julia> out = CO.encode_pmodules_to_common_poset(M1, M2);

julia> TamerOp.describe(out).kind
:common_refinement_translation_result

julia> CO.translated_modules(out).left.Q === CO.common_poset(out)
true
```
"""
function encode_pmodules_to_common_poset(
    M1::PModule{K},
    M2::PModule{K};
    method::Symbol = :product,
    check_poset::Bool = false,
    cache_cover_edges::Bool = true,
    use_cache::Bool = true,
    session_cache::Union{Nothing,SessionCache}=nothing,
) where {K}
    P1 = M1.Q
    P2 = M2.Q

    # Already on the same poset object: nothing to do.
    if P1 === P2
        n = nvertices(P1)
        id = collect(1:n)
        pi1 = EncodingMap(P1, P1, id)
        pi2 = EncodingMap(P1, P2, id)
        return CommonRefinementTranslationResult(P1, (M1, M2), pi1, pi2)
    end

    # Structural equality: same leq, different objects. Avoid P1 x P2 blowup.
    if nvertices(P1) == nvertices(P2) && poset_equal(P1, P2)
        n = nvertices(P1)
        id = collect(1:n)
        pi1 = EncodingMap(P1, P1, id)
        pi2 = EncodingMap(P1, P2, id)

        # Rebase M2 onto P1 (indices match because leq matrices match).
        M2b = PModule{K}(P1, M2.dims, M2.edge_maps; field=M2.field)
        return CommonRefinementTranslationResult(P1, (M1, M2b), pi1, pi2)
    end

    if method != :product
        error("encode_pmodules_to_common_poset: only method=:product is implemented.")
    end

    prod = product_poset(P1, P2;
                         check = check_poset,
                         cache_cover_edges = cache_cover_edges,
                         use_cache = use_cache,
                         session_cache = session_cache)
    P = prod.P
    pi1 = prod.pi1
    pi2 = prod.pi2

    if _CHANGE_OF_POSETS_USE_FUSED_PRODUCT_TRANSLATION[] &&
       P1 isa FinitePoset && P2 isa FinitePoset && P isa FinitePoset
        cache1 = _translation_cache(session_cache, pi1)
        cache2 = _translation_cache(session_cache, pi2)
        key1 = _translation_module_key(:pullback_product_pr1, pi1, M1)
        key2 = _translation_module_key(:pullback_product_pr2, pi2, M2)
        M1p = _translation_cache_get(cache1, key1)
        M2p = _translation_cache_get(cache2, key2)
        if M1p === nothing || M2p === nothing
            M1pb, M2pb = _build_product_projection_pair_dense(M1, M2, P, P1, P2)
            M1p === nothing && _translation_cache_set!(cache1, key1, M1pb)
            M2p === nothing && _translation_cache_set!(cache2, key2, M2pb)
            M1p = M1p === nothing ? M1pb : M1p
            M2p = M2p === nothing ? M2pb : M2p
        end
        return CommonRefinementTranslationResult(P, (M1p, M2p), pi1, pi2)
    end

    if _CHANGE_OF_POSETS_USE_FUSED_PRODUCT_TRANSLATION[] && P isa ProductPoset
        cache1 = _translation_cache(session_cache, pi1)
        cache2 = _translation_cache(session_cache, pi2)
        key1 = _translation_module_key(:pullback_product_structured_pr1, pi1, M1)
        key2 = _translation_module_key(:pullback_product_structured_pr2, pi2, M2)
        M1p = _translation_cache_get(cache1, key1)
        M2p = _translation_cache_get(cache2, key2)
        if M1p === nothing || M2p === nothing
            M1pb, M2pb = _build_product_projection_pair_structured(M1, M2, P)
            M1p === nothing && _translation_cache_set!(cache1, key1, M1pb)
            M2p === nothing && _translation_cache_set!(cache2, key2, M2pb)
            M1p = M1p === nothing ? M1pb : M1p
            M2p = M2p === nothing ? M2pb : M2p
        end
        return CommonRefinementTranslationResult(P, (M1p, M2p), pi1, pi2)
    end

    if P1 isa FinitePoset && P2 isa FinitePoset && P isa FinitePoset
        # Dense fallback when fused product translation is disabled.
        C1 = cover_edges(P1)
        C2 = cover_edges(P2)
        cache1 = _translation_cache(session_cache, pi1)
        cache2 = _translation_cache(session_cache, pi2)
        key1 = _translation_module_key(:pullback_product_pr1, pi1, M1)
        key2 = _translation_module_key(:pullback_product_pr2, pi2, M2)
        M1p = _translation_cache_get(cache1, key1)
        if M1p === nothing
            M1p = _pullback_to_product_pr1(M1, P, P1, P2, C1, C2)
            _translation_cache_set!(cache1, key1, M1p)
        end
        M2p = _translation_cache_get(cache2, key2)
        if M2p === nothing
            M2p = _pullback_to_product_pr2(M2, P, P1, P2, C1, C2)
            _translation_cache_set!(cache2, key2, M2p)
        end
        return CommonRefinementTranslationResult(P, (M1p, M2p), pi1, pi2)
    end

    M1p = pullback(pi1, M1; check = check_poset, session_cache=session_cache)
    M2p = pullback(pi2, M2; check = check_poset, session_cache=session_cache)

    return CommonRefinementTranslationResult(P, (M1p, M2p), pi1, pi2)
end

"""
    CommonRefinementHomSpace{K}

Lazy common-refinement Hom space for modules on different ambient posets.

Mathematically, this represents `Hom(M, N)` after translating `M` and `N` to a
shared refinement poset, currently a product-poset refinement.

Source/target conventions:
- `dom0` is the original source module before translation,
- `cod0` is the original target module before translation.

Cached versus canonical data:
- `dim_cached` is the cached dimension of the degree-0 Hom space,
- `translated`, `basis_matrix`, and `hom` are lazy caches for the translated
  refinement modules, dense basis matrix, and materialized `HomSpace`,
- users should inspect this object via `describe(H)`, `source(H)`, `target(H)`,
  `dim(H)`, `basis(H)`, and [`basis_matrix`](@ref) rather than by reading cache
  fields directly.

# Examples

```jldoctest
julia> using TamerOp

julia> const FF = TamerOp.FiniteFringe;

julia> const IR = TamerOp.IndicatorResolutions;

julia> field = TamerOp.CoreModules.QQField();

julia> P1 = FF.FinitePoset(Bool[1 1; 0 1]);

julia> P2 = FF.FinitePoset(Bool[1]);

julia> M1 = IR.pmodule_from_fringe(FF.one_by_one_fringe(P1,
                                                       FF.principal_upset(P1, 1),
                                                       FF.principal_downset(P1, 2),
                                                       1//1; field=field));

julia> M2 = IR.pmodule_from_fringe(FF.one_by_one_fringe(P2,
                                                       FF.principal_upset(P2, 1),
                                                       FF.principal_downset(P2, 1),
                                                       1//1; field=field));

julia> H = TamerOp.ChangeOfPosets.hom_common_refinement(M1, M2);

julia> TamerOp.describe(H).kind
:common_refinement_hom_space

julia> TamerOp.dim(H) >= 0
true
```
"""
mutable struct CommonRefinementHomSpace{K}
    dom0::PModule{K}
    cod0::PModule{K}
    dim_cached::Int
    method::Symbol
    check_poset::Bool
    cache_cover_edges::Bool
    use_cache::Bool
    session_cache::Union{Nothing,SessionCache}
    rref::Union{Nothing,_SparseRREF{K}}
    offsets::Union{Nothing,Vector{Int}}
    basis_matrix::Union{Nothing,Matrix{K}}
    translated::Any
    hom::Union{Nothing,HomSpace{K}}
end

"""
    CommonRefinementHomBasis{K}

Lazy basis view for a [`CommonRefinementHomSpace`](@ref).

This behaves like a vector of `PMorphism`s. `length(B)` is cheap, while indexed
access or `collect(B)` may materialize the translated refinement modules and the
underlying `HomSpace` basis.

Inspect this wrapper via `describe(B)`, `length(B)`, `source(B)`, and
`target(B)` before collecting explicit morphisms.
"""
struct CommonRefinementHomBasis{K} <: AbstractVector{PMorphism{K}}
    parent::CommonRefinementHomSpace{K}
end

function Base.getproperty(H::CommonRefinementHomSpace, s::Symbol)
    if s === :basis
        return DerivedFunctors.basis(H)
    end
    return getfield(H, s)
end

function Base.show(io::IO, H::CommonRefinementHomSpace{K}) where {K}
    d = describe(H)
    print(io, "CommonRefinementHomSpace(field=", d.field,
          ", dimension=", d.dimension,
          ", translated=", d.translated_materialized, ")")
end

function Base.show(io::IO, B::CommonRefinementHomBasis{K}) where {K}
    d = describe(B)
    print(io, "CommonRefinementHomBasis(field=", d.field,
          ", dimension=", d.dimension,
          ", hom_materialized=", d.hom_materialized, ")")
end

function Base.show(io::IO, ::MIME"text/plain", H::CommonRefinementHomSpace)
    d = describe(H)
    print(io, "CommonRefinementHomSpace",
          "\n  field: ", d.field,
          "\n  dimension: ", d.dimension,
          "\n  source_nvertices: ", d.source_nvertices,
          "\n  target_nvertices: ", d.target_nvertices,
          "\n  source_total_dim: ", d.source_total_dim,
          "\n  target_total_dim: ", d.target_total_dim,
          "\n  translated_materialized: ", d.translated_materialized,
          "\n  basis_matrix_materialized: ", d.basis_matrix_materialized,
          "\n  hom_materialized: ", d.hom_materialized)
end

function Base.show(io::IO, ::MIME"text/plain", B::CommonRefinementHomBasis)
    d = describe(B)
    print(io, "CommonRefinementHomBasis",
          "\n  field: ", d.field,
          "\n  dimension: ", d.dimension,
          "\n  source_nvertices: ", d.source_nvertices,
          "\n  target_nvertices: ", d.target_nvertices,
          "\n  hom_materialized: ", d.hom_materialized,
          "\n  basis_matrix_materialized: ", d.basis_matrix_materialized)
end

Base.IndexStyle(::Type{<:CommonRefinementHomBasis}) = IndexLinear()
Base.size(B::CommonRefinementHomBasis) = (getfield(B.parent, :dim_cached),)
Base.length(B::CommonRefinementHomBasis) = getfield(B.parent, :dim_cached)
Base.axes(B::CommonRefinementHomBasis) = (Base.OneTo(length(B)),)
Base.eltype(::Type{CommonRefinementHomBasis{K}}) where {K} = PMorphism{K}

DerivedFunctors.degree_range(::CommonRefinementHomSpace) = 0:0
DerivedFunctors.dim(H::CommonRefinementHomSpace) = H.dim_cached
function DerivedFunctors.dim(H::CommonRefinementHomSpace, t::Int)
    t == 0 || throw(DomainError(t, "CommonRefinementHomSpace is concentrated in degree 0"))
    return H.dim_cached
end

function _densify_module_if_needed(M::PModule{K}) where {K}
    P = M.Q
    if P isa FinitePoset
        return M
    end
    Pd = FinitePoset(leq_matrix(P); check=false)
    return PModule{K}(Pd, M.dims, M.edge_maps; field=M.field)
end

function _coerce_hom_modules_to_same_poset(M::PModule{K}, N::PModule{K}) where {K}
    M.Q === N.Q && return M, N
    poset_equal(M.Q, N.Q) || error("hom_common_refinement: translated modules landed on non-equal posets.")
    return M, PModule{K}(M.Q, N.dims, N.edge_maps; field=N.field)
end

function _ensure_common_refinement_translation!(H::CommonRefinementHomSpace{K}) where {K}
    cached = getfield(H, :translated)
    cached === nothing || return cached
    M1 = _densify_module_if_needed(getfield(H, :dom0))
    M2 = _densify_module_if_needed(getfield(H, :cod0))
    out = encode_pmodules_to_common_poset(M1, M2;
                                          method=getfield(H, :method),
                                          check_poset=getfield(H, :check_poset),
                                          cache_cover_edges=getfield(H, :cache_cover_edges),
                                          use_cache=getfield(H, :use_cache),
                                          session_cache=getfield(H, :session_cache))
    T1, T2 = _coerce_hom_modules_to_same_poset(out.Ms[1], out.Ms[2])
    data = (out = out, dom = T1, cod = T2)
    setfield!(H, :translated, data)
    return data
end

function _ensure_common_refinement_basis_matrix!(H::CommonRefinementHomSpace{K}) where {K}
    cached = getfield(H, :basis_matrix)
    cached === nothing || return cached
    offs = getfield(H, :offsets)
    R = getfield(H, :rref)
    B = if R === nothing || offs === nothing
        B0, offs0 = _hom_basis_matrix_product_common_refinement(getfield(H, :dom0), getfield(H, :cod0); dense=true)
        setfield!(H, :offsets, offs0)
        B0
    else
        nvars = offs[end]
        length(R.pivot_cols) == nvars ? zeros(K, nvars, 0) : _nullspace_from_pivots(R, nvars)
    end
    setfield!(H, :basis_matrix, B)
    return B
end

function _materialize_common_refinement_hom!(H::CommonRefinementHomSpace{K}) where {K}
    cached = getfield(H, :hom)
    cached === nothing || return cached
    data = _ensure_common_refinement_translation!(H)
    T1 = data.dom
    T2 = data.cod
    # The materialized basis path densifies the refinement modules so the vectorization
    # order must match the dense product order used by `Hom(...)`.
    offs = getfield(H, :offsets)
    B = _ensure_common_refinement_basis_matrix!(H)
    hs = HomSpace{K}(T1, T2, nothing, B, offs)
    setfield!(H, :hom, hs)
    return hs
end

function DerivedFunctors.basis(H::CommonRefinementHomSpace, t::Int)
    t == 0 || throw(DomainError(t, "CommonRefinementHomSpace is concentrated in degree 0"))
    return CommonRefinementHomBasis(H)
end
DerivedFunctors.basis(H::CommonRefinementHomSpace) = CommonRefinementHomBasis(H)

@inline source(H::CommonRefinementHomSpace) = H.dom0
@inline target(H::CommonRefinementHomSpace) = H.cod0
@inline source(B::CommonRefinementHomBasis) = source(B.parent)
@inline target(B::CommonRefinementHomBasis) = target(B.parent)

@inline function _module_total_dim(M::PModule)
    s = 0
    @inbounds for d in M.dims
        s += d
    end
    return s
end

@inline function _common_refinement_describe(H::CommonRefinementHomSpace)
    translated = getfield(H, :translated)
    return (;
        kind=:common_refinement_hom_space,
        field=H.dom0.field,
        method=H.method,
        dimension=H.dim_cached,
        source_nvertices=nvertices(H.dom0.Q),
        target_nvertices=nvertices(H.cod0.Q),
        source_total_dim=_module_total_dim(H.dom0),
        target_total_dim=_module_total_dim(H.cod0),
        translated_materialized=translated !== nothing,
        refinement_nvertices=translated === nothing ? nothing : nvertices(translated.dom.Q),
        basis_matrix_materialized=getfield(H, :basis_matrix) !== nothing,
        hom_materialized=getfield(H, :hom) !== nothing,
    )
end

@inline function _common_refinement_basis_describe(B::CommonRefinementHomBasis)
    H = B.parent
    return (;
        kind=:common_refinement_hom_basis,
        field=source(H).field,
        dimension=length(B),
        source_nvertices=nvertices(source(H).Q),
        target_nvertices=nvertices(target(H).Q),
        source_total_dim=_module_total_dim(source(H)),
        target_total_dim=_module_total_dim(target(H)),
        basis_matrix_materialized=getfield(H, :basis_matrix) !== nothing,
        hom_materialized=getfield(H, :hom) !== nothing,
    )
end

describe(H::CommonRefinementHomSpace) = _common_refinement_describe(H)
describe(B::CommonRefinementHomBasis) = _common_refinement_basis_describe(B)

"""
    common_refinement_summary(H) -> NamedTuple

Owner-local summary alias for common-refinement Hom objects.

Use this as the obvious `ChangeOfPosets` entrypoint for cheap-first inspection
before materializing explicit basis morphisms.
"""
@inline common_refinement_summary(H::CommonRefinementHomSpace) = describe(H)
@inline common_refinement_summary(B::CommonRefinementHomBasis) = describe(B)

"""
    basis_matrix(H::CommonRefinementHomSpace) -> Matrix

Return the dense basis matrix whose columns encode a basis of the common-
refinement Hom space.

This is a heavier accessor than `dim(H)` or `basis(H)`. Prefer those cheap
inspection surfaces first.
"""
@inline basis_matrix(H::CommonRefinementHomSpace) = _ensure_common_refinement_basis_matrix!(H)
@inline basis_matrix(B::CommonRefinementHomBasis) = basis_matrix(B.parent)

"""
    hom_dimension(H::CommonRefinementHomSpace) -> Int

Owner-local scalar alias for `dim(H)`.
"""
@inline hom_dimension(H::CommonRefinementHomSpace) = DerivedFunctors.dim(H)

"""
    check_common_refinement_hom(H; throw=false) -> CommonRefinementHomValidationSummary

Validate the cached state and basic mathematical contract of a
[`CommonRefinementHomSpace`](@ref).
"""
function check_common_refinement_hom(H::CommonRefinementHomSpace; throw::Bool=false)
    issues = String[]
    source(H).field == target(H).field ||
        push!(issues, "source and target modules must live over the same coefficient field.")
    H.dim_cached >= 0 || push!(issues, "cached dimension must be nonnegative.")
    H.method == :product || push!(issues, "unsupported common-refinement method $(H.method).")

    offs = getfield(H, :offsets)
    R = getfield(H, :rref)
    B = getfield(H, :basis_matrix)
    hs = getfield(H, :hom)
    translated = getfield(H, :translated)

    (R === nothing) == (offs === nothing) || push!(issues, "rref and offsets caches should be populated together.")
    if B !== nothing
        offs === nothing && push!(issues, "basis_matrix is populated but offsets are missing.")
        offs !== nothing && size(B, 1) == offs[end] ||
            offs === nothing || push!(issues, "basis_matrix row count does not match the vectorization offsets.")
        size(B, 2) == H.dim_cached || push!(issues, "basis_matrix column count does not match the cached Hom dimension.")
    end
    if translated !== nothing
        translated.dom.field == source(H).field ||
            push!(issues, "translated source module changed coefficient field unexpectedly.")
        translated.cod.field == target(H).field ||
            push!(issues, "translated target module changed coefficient field unexpectedly.")
        poset_equal(translated.dom.Q, translated.cod.Q) ||
            push!(issues, "translated common-refinement modules do not share the same ambient poset.")
    end
    if hs !== nothing
        DerivedFunctors.dim(hs) == H.dim_cached ||
            push!(issues, "materialized HomSpace dimension does not match the cached dimension.")
    end

    report = (;
        kind=:common_refinement_hom_space,
        valid=isempty(issues),
        method=H.method,
        dimension=H.dim_cached,
        source_nvertices=nvertices(source(H).Q),
        target_nvertices=nvertices(target(H).Q),
        translated_materialized=translated !== nothing,
        basis_matrix_materialized=B !== nothing,
        hom_materialized=hs !== nothing,
        issues=issues,
    )
    if throw && !report.valid
        Base.throw(ArgumentError("check_common_refinement_hom: " * join(report.issues, " ")))
    end
    return common_refinement_hom_validation_summary(report)
end

@inline check_common_refinement_hom(B::CommonRefinementHomBasis; throw::Bool=false) =
    check_common_refinement_hom(B.parent; throw=throw)

function Base.getindex(B::CommonRefinementHomBasis{K}, j::Int) where {K}
    @boundscheck checkbounds(B, j)
    hs = _materialize_common_refinement_hom!(getfield(B, :parent))
    return DerivedFunctors.basis(hs)[j]
end

function Base.iterate(B::CommonRefinementHomBasis, state::Int=1)
    state > length(B) && return nothing
    return B[state], state + 1
end

function Base.collect(B::CommonRefinementHomBasis{K}) where {K}
    hs = _materialize_common_refinement_hom!(getfield(B, :parent))
    return copy(DerivedFunctors.basis(hs))
end

@inline _prod_linear_index(i::Int, j::Int, n1::Int, n2::Int, dense::Bool) =
    dense ? ((i - 1) * n2 + j) : (i + (j - 1) * n1)

function _hom_product_common_refinement_rref(
    M1::PModule{K},
    M2::PModule{K};
    dense::Bool,
) where {K}
    P1 = M1.Q
    P2 = M2.Q
    n1 = nvertices(P1)
    n2 = nvertices(P2)
    n = n1 * n2
    offs = zeros(Int, n + 1)
    idx = 0
    if dense
        @inbounds for i in 1:n1
            di = M1.dims[i]
            for j in 1:n2
                idx += 1
                offs[idx + 1] = offs[idx] + M2.dims[j] * di
            end
        end
    else
        @inbounds for j in 1:n2
            dj = M2.dims[j]
            for i in 1:n1
                idx += 1
                offs[idx + 1] = offs[idx] + dj * M1.dims[i]
            end
        end
    end
    nvars = offs[end]
    nvars == 0 && return nothing, offs, nvars

    C1 = cover_edges(P1).edges
    C2 = cover_edges(P2).edges
    R = _SparseRREF{K}(nvars)
    row = SparseRow{K}()
    acc = _SparseRowAccumulator{K}(nvars)
    fullrank = false

    @inbounds for (i1, i2) in C1
        A = M1.edge_maps[i1, i2]
        du = M1.dims[i1]
        dv = M1.dims[i2]
        du == 0 && continue
        for j in 1:n2
            dN = M2.dims[j]
            dN == 0 && continue
            u = _prod_linear_index(i1, j, n1, n2, dense)
            v = _prod_linear_index(i2, j, n1, n2, dense)
            offu = offs[u]
            offv = offs[v]
            for ii in 1:dN, jj in 1:du
                _reset_sparse_row_accumulator!(acc)
                _push_sparse_row_entry!(acc, offu + ii + (jj - 1) * dN, one(K))
                if A isa SparseMatrixCSC
                    for ptr in A.colptr[jj]:(A.colptr[jj + 1] - 1)
                        l = A.rowval[ptr]
                        c = A.nzval[ptr]
                        _push_sparse_row_entry!(acc, offv + ii + (l - 1) * dN, -c)
                    end
                else
                    for l in 1:dv
                        c = A[l, jj]
                        iszero(c) && continue
                        _push_sparse_row_entry!(acc, offv + ii + (l - 1) * dN, -c)
                    end
                end
                _materialize_sparse_row!(row, acc)
                isempty(row.idx) && continue
                _sparse_rref_push_homogeneous!(R, row)
                if length(R.pivot_cols) == nvars
                    fullrank = true
                    break
                end
            end
            fullrank && break
        end
        fullrank && break
    end

    if !fullrank
        @inbounds for (j1, j2) in C2
            B = M2.edge_maps[j1, j2]
            dNu = M2.dims[j1]
            dNv = M2.dims[j2]
            dNv == 0 && continue
            for i in 1:n1
                du = M1.dims[i]
                du == 0 && continue
                u = _prod_linear_index(i, j1, n1, n2, dense)
                v = _prod_linear_index(i, j2, n1, n2, dense)
                offu = offs[u]
                offv = offs[v]
                for ii in 1:dNv, jj in 1:du
                    _reset_sparse_row_accumulator!(acc)
                    if B isa SparseMatrixCSC
                        for k in 1:dNu
                            for ptr in B.colptr[k]:(B.colptr[k + 1] - 1)
                                r = B.rowval[ptr]
                                r == ii || continue
                                c = B.nzval[ptr]
                                _push_sparse_row_entry!(acc, offu + k + (jj - 1) * dNu, c)
                            end
                        end
                    else
                        for k in 1:dNu
                            c = B[ii, k]
                            iszero(c) && continue
                            _push_sparse_row_entry!(acc, offu + k + (jj - 1) * dNu, c)
                        end
                    end
                    _push_sparse_row_entry!(acc, offv + ii + (jj - 1) * dNv, -one(K))
                    _materialize_sparse_row!(row, acc)
                    isempty(row.idx) && continue
                    _sparse_rref_push_homogeneous!(R, row)
                    if length(R.pivot_cols) == nvars
                        fullrank = true
                        break
                    end
                end
                fullrank && break
            end
            fullrank && break
        end
    end

    return R, offs, nvars
end

function _hom_dim_product_common_refinement(
    M1::PModule{K},
    M2::PModule{K};
    dense::Bool,
) where {K}
    R, _, nvars = _hom_product_common_refinement_rref(M1, M2; dense=dense)
    R === nothing && return 0
    return nvars - length(R.pivot_cols)
end

function _hom_basis_matrix_product_common_refinement(
    M1::PModule{K},
    M2::PModule{K};
    dense::Bool,
) where {K}
    R, offs, nvars = _hom_product_common_refinement_rref(M1, M2; dense=dense)
    if R === nothing
        return zeros(K, 0, 0), offs
    end
    B = length(R.pivot_cols) == nvars ? zeros(K, nvars, 0) : _nullspace_from_pivots(R, nvars)
    return B, offs
end

"""
    hom_common_refinement(M1::PModule, M2::PModule; kwargs...)

Compute `Hom(M1, M2)` after moving both modules to a common refinement poset.

For product refinements this returns a lazy Hom-like object whose `dim` is
computed directly from the product constraints without eagerly materializing the
two translated modules. `basis(...)` returns a lazy vector-like view; taking
its length is cheap, while indexing/collecting basis elements materializes the
translated common-refinement modules as needed.

Cheap-first workflow:
- start with `common_refinement_summary(H)` or `describe(H)`,
- use `dim(H)` / `hom_dimension(H)` before asking for an explicit basis,
- materialize `basis_matrix(H)` or `collect(basis(H))` only when you need
  coordinates or explicit morphisms.
"""
function hom_common_refinement(
    M1::PModule{K},
    M2::PModule{K};
    method::Symbol = :product,
    check_poset::Bool = false,
    cache_cover_edges::Bool = true,
    use_cache::Bool = true,
    session_cache::Union{Nothing,SessionCache}=nothing,
) where {K}
    P1 = M1.Q
    P2 = M2.Q
    if P1 === P2
        return Hom(M1, M2)
    end
    if nvertices(P1) == nvertices(P2) && poset_equal(P1, P2)
        M1d = _densify_module_if_needed(M1)
        M2b = PModule{K}(M1d.Q, M2.dims, M2.edge_maps; field=M2.field)
        return Hom(M1d, M2b)
    end
    method == :product || error("hom_common_refinement: only method=:product is implemented.")
    if _CHANGE_OF_POSETS_USE_DIRECT_HOM_COMMON_REFINEMENT[]
        # Use the dense product ordering even for structured inputs so the cached
        # constraint basis can be reused if `basis(H)` later materializes to a
        # dense common-refinement HomSpace.
        R, offs, nvars = _hom_product_common_refinement_rref(M1, M2; dense=true)
        d = R === nothing ? 0 : (nvars - length(R.pivot_cols))
        return CommonRefinementHomSpace{K}(M1, M2, d, method, check_poset, cache_cover_edges,
                                           use_cache, session_cache, R, offs, nothing, nothing, nothing)
    end
    M1d = _densify_module_if_needed(M1)
    M2d = _densify_module_if_needed(M2)
    out = encode_pmodules_to_common_poset(M1d, M2d;
                                          method=method,
                                          check_poset=check_poset,
                                          cache_cover_edges=cache_cover_edges,
                                          use_cache=use_cache,
                                          session_cache=session_cache)
    T1, T2 = _coerce_hom_modules_to_same_poset(out.Ms[1], out.Ms[2])
    return Hom(T1, T2)
end

function hom_dim_common_refinement(
    M1::PModule{K},
    M2::PModule{K};
    method::Symbol = :product,
    check_poset::Bool = false,
    cache_cover_edges::Bool = true,
    use_cache::Bool = true,
    session_cache::Union{Nothing,SessionCache}=nothing,
) where {K}
    P1 = M1.Q
    P2 = M2.Q
    if P1 === P2
        return DerivedFunctors.dim(Hom(M1, M2))
    end
    if nvertices(P1) == nvertices(P2) && poset_equal(P1, P2)
        M1d = _densify_module_if_needed(M1)
        M2b = PModule{K}(M1d.Q, M2.dims, M2.edge_maps; field=M2.field)
        return DerivedFunctors.dim(Hom(M1d, M2b))
    end
    method == :product || error("hom_dim_common_refinement: only method=:product is implemented.")
    if _CHANGE_OF_POSETS_USE_DIRECT_HOM_COMMON_REFINEMENT[]
        return _hom_dim_product_common_refinement(M1, M2; dense=true)
    end
    return DerivedFunctors.dim(hom_common_refinement(M1, M2;
                                                     method=method,
                                                     check_poset=check_poset,
                                                     cache_cover_edges=cache_cover_edges,
                                                     use_cache=use_cache,
                                                     session_cache=session_cache))
end

has_nonzero_hom_common_refinement(M1::PModule, M2::PModule; kwargs...) =
    hom_dim_common_refinement(M1, M2; kwargs...) > 0

function hom_bidim_common_refinement(M1::PModule, M2::PModule; kwargs...)
    return (forward = hom_dim_common_refinement(M1, M2; kwargs...),
            reverse = hom_dim_common_refinement(M2, M1; kwargs...))
end


# -----------------------------------------------------------------------------
# Pullback (restriction)
# -----------------------------------------------------------------------------

# Pullback / restriction of modules along a monotone map

@inline function _pullback_pair_data(pi::EncodingMap, C)
    edge_keys = Vector{Tuple{Int,Int}}(undef, length(C))
    pairs = Vector{Tuple{Int,Int}}(undef, length(C))
    @inbounds for i in eachindex(C)
        u, v = C[i]
        iu = pi.pi_of_q[u]
        iv = pi.pi_of_q[v]
        edge_keys[i] = (u, v)
        pairs[i] = (iu, iv)
    end
    return edge_keys, _prepare_map_leq_batch_owned(pairs)
end

# Internal helper: compute pullback module along pi without re-checking monotonicity.
@inline function _pullback_module_no_check(
    pi::EncodingMap,
    M::PModule{K},
    C,
    prepared=nothing,
) where {K}
    Q = pi.Q
    P = pi.P
    @assert M.Q === P

    dims_out = Vector{Int}(undef, nvertices(Q))
    @inbounds for q in 1:nvertices(Q)
        dims_out[q] = M.dims[pi.pi_of_q[q]]
    end

    edge_maps_out = Dict{Tuple{Int,Int}, AbstractMatrix{K}}()
    sizehint!(edge_maps_out, length(C))

    if prepared === nothing
        edge_keys = Tuple{Int,Int}[]
        pairs = Tuple{Int,Int}[]
        sizehint!(edge_keys, length(C))
        sizehint!(pairs, length(C))
        @inbounds for (u, v) in C
            iu = pi.pi_of_q[u]
            iv = pi.pi_of_q[v]
            push!(edge_keys, (u, v))
            # If pi is monotone, iu <= iv in P and map_leq is defined.
            push!(pairs, (iu, iv))
        end
        maps = map_leq_many(M, pairs)
        @inbounds for i in eachindex(edge_keys)
            edge_maps_out[edge_keys[i]] = maps[i]
        end
    else
        edge_keys, pair_batch = prepared
        maps = map_leq_many(M, pair_batch)
        @inbounds for i in eachindex(edge_keys)
            edge_maps_out[edge_keys[i]] = maps[i]
        end
    end

    return PModule{K}(Q, dims_out, edge_maps_out; field=M.field)
end

"""
    pullback(pi::EncodingMap, M::PModule; check=true, session_cache=nothing) -> PModule
    pullback(pi::EncodingMap, f::PMorphism; check=true, session_cache=nothing) -> PMorphism

Categorical pullback of a module or morphism along a monotone map `pi : Q -> P`.

Source/target convention
- `pi.Q` is the source poset of the monotone map,
- `pi.P` is the target poset of the monotone map,
- `M` or `f` must live on `pi.P`,
- the returned module or morphism lives on `pi.Q`.

Notation policy
- For notebook-facing code, prefer [`restriction`](@ref) as the canonical name.
- `pullback` is the categorical alias and calls the same kernel.

Cache semantics
- with `session_cache=nothing`, this is a one-shot translation,
- with `session_cache=sc::SessionCache`, repeated pullbacks along the same `pi`
  reuse translation plans and cached translated objects.

Cheap-first workflow
- call `describe(restriction(pi, M))` for a compact summary of the pulled-back
  module,
- materialize larger algebraic constructions only after inspecting that module.
"""
function pullback(
    pi::EncodingMap,
    M::PModule{K};
    check::Bool = true,
    session_cache::Union{Nothing,SessionCache}=nothing,
    plan=nothing,
) where {K}
    check && _check_monotone(pi)
    cache = _translation_cache(session_cache, pi)
    key = _translation_module_key(:pullback_module, pi, M)
    cached = _translation_cache_get(cache, key)
    cached !== nothing && return cached
    if _CHANGE_OF_POSETS_USE_PRODUCT_PROJECTION_FASTPATH[]
        kind = _product_projection_kind(pi)
        if kind != 0
            return _translation_cache_set!(cache, key,
                                           _pullback_product_projection_structured(pi, M, kind))
        end
    end
    if plan === nothing && session_cache === nothing
        C = cover_edges(pi.Q)
        return _pullback_module_no_check(pi, M, C)
    end
    plan === nothing && (plan = _translation_plan(pi; session_cache=session_cache))
    return _translation_cache_set!(cache, key,
                                   _pullback_module_no_check(pi, M, plan.coverQ_edges, plan.pullback_prepared))
end

function pullback(
    pi::EncodingMap,
    f::PMorphism{K};
    check::Bool = true,
    session_cache::Union{Nothing,SessionCache}=nothing,
    plan=nothing,
) where {K}
    check && _check_monotone(pi)
    cache = _translation_cache(session_cache, pi)
    key = _translation_morphism_key(:pullback_morphism, pi, f)
    cached = _translation_cache_get(cache, key)
    cached !== nothing && return cached
    if plan === nothing && session_cache === nothing
        C = cover_edges(pi.Q)
        prepared = _pullback_pair_data(pi, C)
    else
        plan === nothing && (plan = _translation_plan(pi; session_cache=session_cache))
        C = plan.coverQ_edges
        prepared = plan.pullback_prepared
    end

    # Pull back domain and codomain modules.
    dom_pb = pullback(pi, f.dom; check=false, session_cache=session_cache, plan=plan)
    cod_pb = pullback(pi, f.cod; check=false, session_cache=session_cache, plan=plan)

    # Pull back components pointwise along pi: (f_pb)_q = f_{pi(q)}.
    comps_pb = Vector{Matrix{K}}(undef, nvertices(pi.Q))
    @inbounds for q in 1:nvertices(pi.Q)
        comps_pb[q] = f.comps[pi.pi_of_q[q]]
    end

    return _translation_cache_set!(cache, key, PMorphism(dom_pb, cod_pb, comps_pb))
end


"""
    restriction(pi::EncodingMap, N::PModule; check=true, session_cache=nothing) -> PModule
    restriction(pi::EncodingMap, f::PMorphism; check=true, session_cache=nothing) -> PMorphism

Canonical notebook-facing name for restriction along a monotone map.

This is the recommended user entrypoint for change-of-poset restriction. It is
mathematically the same operation as [`pullback`](@ref), and it forwards
directly to that implementation.

# Examples

```jldoctest
julia> using TamerOp

julia> const CO = TamerOp.ChangeOfPosets;

julia> const EN = TamerOp.Encoding;

julia> const FF = TamerOp.FiniteFringe;

julia> const IR = TamerOp.IndicatorResolutions;

julia> field = TamerOp.CoreModules.QQField();

julia> P = FF.FinitePoset(Bool[1]);

julia> Q = FF.FinitePoset(Bool[1 1; 0 1]);

julia> pi = EN.EncodingMap(Q, P, [1, 1]);

julia> M = IR.pmodule_from_fringe(FF.one_by_one_fringe(P,
                                                      FF.principal_upset(P, 1),
                                                      FF.principal_downset(P, 1),
                                                      1//1; field=field));

julia> R = CO.restriction(pi, M);

julia> TamerOp.describe(R).vertices
2
```
"""
@inline restriction(pi::EncodingMap, N::PModule;
                    check::Bool=true,
                    session_cache::Union{Nothing,SessionCache}=nothing) =
    pullback(pi, N; check=check, session_cache=session_cache)
@inline restriction(pi::EncodingMap, f::PMorphism;
                    check::Bool=true,
                    session_cache::Union{Nothing,SessionCache}=nothing) =
    pullback(pi, f; check=check, session_cache=session_cache)

# -----------------------------------------------------------------------------
# Left Kan extension (left pushforward)
# -----------------------------------------------------------------------------

struct _KanFiberPlan
    idxs::Vector{Int}                     # q's in canonical fiber order
    edge_u::Vector{Int}                   # global source vertex of each active cover edge in the fiber
    edge_v::Vector{Int}                   # global target vertex of each active cover edge in the fiber
    edge_u_local::Vector{Int}             # source local index in idxs
    edge_v_local::Vector{Int}             # target local index in idxs
    edge_slot::Vector{Int}                # slot into store.maps_to_succ[edge_u]
    out_edge_ptr::Vector{Int}             # local source -> incident outgoing edge indices
    out_edge_idx::Vector{Int}
    in_edge_ptr::Vector{Int}              # local target -> incident incoming edge indices
    in_edge_idx::Vector{Int}
    extremum::Int                         # terminal/minimum object in the fiber, or 0
    extremum_local::Int                   # local position of the extremum in idxs, or 0
    extremal_batch::Union{Nothing,MapLeqQueryBatch}
end

# LeftKanData stores enough to compute maps functorially (including morphisms).
struct LeftKanData{K}
    offsets::Vector{Vector{Int}}          # local offset prefixes into the ambient direct sum
    dimS::Vector{Int}                     # ambient direct sum dimension
    W::Vector{Matrix{K}}                  # section V_p -> S_p
    L::Vector{Matrix{K}}                  # quotient map S_p -> V_p (L*W = I)
    dimV::Vector{Int}                     # dim V_p
end

struct RightKanData{K}
    offsets::Vector{Vector{Int}}          # local offset prefixes into the ambient direct product
    dimS::Vector{Int}
    Ksec::Vector{Matrix{K}}               # section V_p -> S_p (kernel basis)
    L::Vector{Matrix{K}}                  # coordinate map S_p -> V_p (L*K = I)
    dimV::Vector{Int}
end

@inline function _offset_prefix(idxs::Vector{Int}, d::Vector{Int})
    offs = Vector{Int}(undef, length(idxs) + 1)
    offs[1] = 0
    @inbounds for i in eachindex(idxs)
        offs[i + 1] = offs[i] + d[idxs[i]]
    end
    return offs, offs[end]
end

@inline function _packed_local_edge_lists(nlocals::Int, edge_local::Vector{Int})
    ptr = zeros(Int, nlocals + 1)
    @inbounds for loc in edge_local
        ptr[loc + 1] += 1
    end
    running = 1
    @inbounds for loc in 1:nlocals
        count = ptr[loc + 1]
        ptr[loc] = running
        running += count
    end
    ptr[end] = running
    data = Vector{Int}(undef, length(edge_local))
    next = copy(ptr)
    @inbounds for edge_idx in eachindex(edge_local)
        loc = edge_local[edge_idx]
        pos = next[loc]
        data[pos] = edge_idx
        next[loc] = pos + 1
    end
    return ptr, data
end

@inline function _cover_store_from_succ_maps(
    ::Type{K},
    preds::Vector{Vector{Int}},
    succs::Vector{Vector{Int}},
    pred_slots::Vector{Vector{Int}},
    maps_to_succ::Vector{Vector{SparseMatrixCSC{K,Int}}},
) where {K}
    n = length(preds)
    maps_from_pred = [Vector{SparseMatrixCSC{K,Int}}(undef, length(preds[v])) for v in 1:n]
    @inbounds for u in 1:n
        su = succs[u]
        slots = pred_slots[u]
        mu = maps_to_succ[u]
        for j in eachindex(su)
            v = su[j]
            maps_from_pred[v][slots[j]] = mu[j]
        end
    end
    return CoverEdgeMapStore{K,SparseMatrixCSC{K,Int}}(preds, succs, maps_from_pred, maps_to_succ, sum(length, succs))
end

# Left inverse for a full-column-rank matrix using exact field linear algebra.
function _left_inverse_full_column(field::AbstractCoeffField, A::AbstractMatrix{K}) where {K}
    A0 = Matrix(A)
    m,n = size(A0)
    if n == 0
        return zeros(K, 0, m)
    end
    if field isa RealField
        return A0 \ Matrix{K}(I, m, m)
    end

    Aug = hcat(A0, Matrix{K}(I, m, m))
    R, pivs_all = FieldLinAlg.rref(field, Aug)
    pivs = Int[]
    for p in pivs_all
        p <= n && push!(pivs, p)
    end
    length(pivs) == n || error("left_inverse_full_column: expected full column rank, got rank $(length(pivs)) < $n")

    L = zeros(K, n, m)
    @inbounds for (row, pcol) in enumerate(pivs)
        L[pcol, :] = R[row, n+1:end]
    end
    return L
end

@inline function _selector_left_inverse(::Type{K}, free_cols::Vector{Int}, nvars::Int) where {K}
    L = zeros(K, length(free_cols), nvars)
    @inbounds for (k, j) in enumerate(free_cols)
        L[k, j] = one(K)
    end
    return L
end

function _nullspace_selector_summary(field::AbstractCoeffField, C::SparseMatrixCSC{K,Int}) where {K}
    nvars = size(C, 2)
    if nvars == 0
        return zeros(K, 0, 0), zeros(K, 0, 0)
    end
    if field isa RealField || !_CHANGE_OF_POSETS_USE_RIGHT_KAN_SELECTOR_FASTPATH[]
        Kp = FieldLinAlg.nullspace(field, C)
        return Kp, _left_inverse_full_column(field, Kp)
    end

    rows = _sparse_rows(C)
    R = _SparseRREF{K}(nvars)
    @inbounds for row in rows
        isempty(row.idx) && continue
        _sparse_rref_push_homogeneous!(R, row)
        length(R.pivot_cols) == nvars && break
    end

    if length(R.pivot_cols) == nvars
        return zeros(K, nvars, 0), zeros(K, 0, nvars)
    end

    Kp = _nullspace_from_pivots(R, nvars)
    free_cols = Int[]
    sizehint!(free_cols, nvars - length(R.pivot_cols))
    @inbounds for j in 1:nvars
        R.pivot_pos[j] == 0 && push!(free_cols, j)
    end
    return Kp, _selector_left_inverse(K, free_cols, nvars)
end

function _nullspace_selector_summary(field::QQField, C::SparseMatrixCSC{QQ,Int})
    nvars = size(C, 2)
    if nvars == 0
        return zeros(QQ, 0, 0), zeros(QQ, 0, 0)
    end
    if _CHANGE_OF_POSETS_USE_RIGHT_KAN_QQ_SUMMARY_SELECTOR[]
        S = FieldLinAlg.elimination_summary(field, C)
        R = S.rref
        Kp = FieldLinAlg.nullspace(S)
        free_cols = Int[]
        sizehint!(free_cols, nvars - length(R.pivot_cols))
        @inbounds for j in 1:nvars
            R.pivot_pos[j] == 0 && push!(free_cols, j)
        end
        return Kp, _selector_left_inverse(QQ, free_cols, nvars)
    end
    if !_CHANGE_OF_POSETS_USE_RIGHT_KAN_SELECTOR_FASTPATH[]
        Kp = FieldLinAlg.nullspace(field, C)
        return Kp, _left_inverse_full_column(field, Kp)
    end

    rows = _sparse_rows(C)
    R = _SparseRREF{QQ}(nvars)
    @inbounds for row in rows
        isempty(row.idx) && continue
        _sparse_rref_push_homogeneous!(R, row)
        length(R.pivot_cols) == nvars && break
    end

    if length(R.pivot_cols) == nvars
        return zeros(QQ, nvars, 0), zeros(QQ, 0, nvars)
    end

    Kp = _nullspace_from_pivots(R, nvars)
    free_cols = Int[]
    sizehint!(free_cols, nvars - length(R.pivot_cols))
    @inbounds for j in 1:nvars
        R.pivot_pos[j] == 0 && push!(free_cols, j)
    end
    return Kp, _selector_left_inverse(QQ, free_cols, nvars)
end

function _left_kan_relation_transpose(
    ::Type{K},
    fiber::_KanFiberPlan,
    offp::Vector{Int},
    d::Vector{Int},
    maps_to_succ,
) where {K}
    Sp = offp[end]
    nrows = 0
    nnz_est = 0
    @inbounds for edge_idx in eachindex(fiber.edge_u)
        u = fiber.edge_u[edge_idx]
        du = d[u]
        du == 0 && continue
        nrows += du
        A = maps_to_succ[u][fiber.edge_slot[edge_idx]]
        nnz_est += du
        nnz_est += A isa SparseMatrixCSC ? nnz(A) : du * size(A, 1)
    end

    Irows = Int[]
    Jcols = Int[]
    Vvals = K[]
    sizehint!(Irows, nnz_est)
    sizehint!(Jcols, nnz_est)
    sizehint!(Vvals, nnz_est)

    row0 = 0
    @inbounds for edge_idx in eachindex(fiber.edge_u)
        u = fiber.edge_u[edge_idx]
        du = d[u]
        du == 0 && continue

        v = fiber.edge_v[edge_idx]
        ou = offp[fiber.edge_u_local[edge_idx]]
        ov = offp[fiber.edge_v_local[edge_idx]]
        A = maps_to_succ[u][fiber.edge_slot[edge_idx]]

        for j in 1:du
            row = row0 + j
            push!(Irows, row)
            push!(Jcols, ou + j)
            push!(Vvals, one(K))

            if A isa SparseMatrixCSC
                @inbounds for ptr in A.colptr[j]:(A.colptr[j + 1] - 1)
                    push!(Irows, row)
                    push!(Jcols, ov + A.rowval[ptr])
                    push!(Vvals, -A.nzval[ptr])
                end
            else
                @inbounds for r in 1:size(A, 1)
                    val = A[r, j]
                    iszero(val) && continue
                    push!(Irows, row)
                    push!(Jcols, ov + r)
                    push!(Vvals, -val)
                end
            end
        end

        row0 += du
    end

    return sparse(Irows, Jcols, Vvals, nrows, Sp)
end

function _left_kan_quotient_summary(field::AbstractCoeffField, RelT::AbstractMatrix{K}) where {K}
    Wp = FieldLinAlg.nullspace(field, RelT)
    return Wp, _left_inverse_full_column(field, Wp)
end

function _left_kan_quotient_summary(field::QQField, RelT::SparseMatrixCSC{QQ,Int})
    if _CHANGE_OF_POSETS_USE_LEFT_KAN_QQ_SUMMARY_QUOTIENT[]
        S = FieldLinAlg.elimination_summary(field, RelT)
        Wp = FieldLinAlg.nullspace(S)
        R = S.rref
        free_cols = Int[]
        sizehint!(free_cols, size(RelT, 2) - length(R.pivot_cols))
        @inbounds for j in 1:size(RelT, 2)
            R.pivot_pos[j] == 0 && push!(free_cols, j)
        end
        return Wp, _selector_left_inverse(QQ, free_cols, size(RelT, 2))
    end
    Wp = FieldLinAlg.nullspace(field, RelT)
    return Wp, _left_inverse_full_column(field, Wp)
end

@inline function _right_kan_row_offsets(fiber::_KanFiberPlan, d::Vector{Int})
    edge_rows = Vector{Int}(undef, length(fiber.edge_v) + 1)
    edge_rows[1] = 0
    @inbounds for edge_idx in eachindex(fiber.edge_v)
        edge_rows[edge_idx + 1] = edge_rows[edge_idx] + d[fiber.edge_v[edge_idx]]
    end
    return edge_rows
end

function _right_kan_constraint_matrix_triplet(
    ::Type{K},
    fiber::_KanFiberPlan,
    offp::Vector{Int},
    d::Vector{Int},
    maps_to_succ,
) where {K}
    nrows = 0
    @inbounds for edge_idx in eachindex(fiber.edge_v)
        nrows += d[fiber.edge_v[edge_idx]]
    end
    Sp = offp[end]
    Irows = Int[]
    Jcols = Int[]
    Vvals = K[]
    row0 = 0

    @inbounds for edge_idx in eachindex(fiber.edge_u)
        u = fiber.edge_u[edge_idx]
        du = d[u]
        dv = d[fiber.edge_v[edge_idx]]
        dv == 0 && continue

        ou = offp[fiber.edge_u_local[edge_idx]]
        ov = offp[fiber.edge_v_local[edge_idx]]
        A = maps_to_succ[u][fiber.edge_slot[edge_idx]]

        for k in 1:dv
            push!(Irows, row0 + k)
            push!(Jcols, ov + k)
            push!(Vvals, one(K))
        end

        for j in 1:du
            if A isa SparseMatrixCSC
                @inbounds for ptr in A.colptr[j]:(A.colptr[j + 1] - 1)
                    k = A.rowval[ptr]
                    val = A.nzval[ptr]
                    push!(Irows, row0 + k)
                    push!(Jcols, ou + j)
                    push!(Vvals, -val)
                end
            else
                @inbounds for k in 1:size(A, 1)
                    val = A[k, j]
                    iszero(val) && continue
                    push!(Irows, row0 + k)
                    push!(Jcols, ou + j)
                    push!(Vvals, -val)
                end
            end
        end

        row0 += dv
    end

    return sparse(Irows, Jcols, Vvals, nrows, Sp)
end

function _right_kan_constraint_matrix_direct_csc(
    ::Type{K},
    fiber::_KanFiberPlan,
    offp::Vector{Int},
    d::Vector{Int},
    maps_to_succ,
) where {K}
    Sp = offp[end]
    edge_rows = _right_kan_row_offsets(fiber, d)
    nrows = edge_rows[end]
    colptr = Vector{Int}(undef, Sp + 1)
    colptr[1] = 1
    nnz = 0

    @inbounds for loc in eachindex(fiber.idxs)
        q = fiber.idxs[loc]
        dq = d[q]
        in_lo = fiber.in_edge_ptr[loc]
        in_hi = fiber.in_edge_ptr[loc + 1] - 1
        out_lo = fiber.out_edge_ptr[loc]
        out_hi = fiber.out_edge_ptr[loc + 1] - 1
        for j in 1:dq
            col_nnz = in_hi >= in_lo ? (in_hi - in_lo + 1) : 0
            if out_hi >= out_lo
                for pos in out_lo:out_hi
                    edge_idx = fiber.out_edge_idx[pos]
                    u = fiber.edge_u[edge_idx]
                    A = maps_to_succ[u][fiber.edge_slot[edge_idx]]
                    if !(A isa SparseMatrixCSC)
                        return _right_kan_constraint_matrix_triplet(K, fiber, offp, d, maps_to_succ)
                    end
                    col_nnz += A.colptr[j + 1] - A.colptr[j]
                end
            end
            nnz += col_nnz
            colptr[offp[loc] + j + 1] = nnz + 1
        end
    end

    rowval = Vector{Int}(undef, nnz)
    nzval = Vector{K}(undef, nnz)
    next = copy(colptr)

    @inbounds for loc in eachindex(fiber.idxs)
        q = fiber.idxs[loc]
        dq = d[q]
        in_lo = fiber.in_edge_ptr[loc]
        in_hi = fiber.in_edge_ptr[loc + 1] - 1
        out_lo = fiber.out_edge_ptr[loc]
        out_hi = fiber.out_edge_ptr[loc + 1] - 1
        for j in 1:dq
            col = offp[loc] + j
            pos = next[col]
            if in_hi >= in_lo
                for ipos in in_lo:in_hi
                    edge_idx = fiber.in_edge_idx[ipos]
                    rowval[pos] = edge_rows[edge_idx] + j
                    nzval[pos] = one(K)
                    pos += 1
                end
            end
            if out_hi >= out_lo
                for opos in out_lo:out_hi
                    edge_idx = fiber.out_edge_idx[opos]
                    u = fiber.edge_u[edge_idx]
                    A = maps_to_succ[u][fiber.edge_slot[edge_idx]]
                    for ptr in A.colptr[j]:(A.colptr[j + 1] - 1)
                        rowval[pos] = edge_rows[edge_idx] + A.rowval[ptr]
                        nzval[pos] = -A.nzval[ptr]
                        pos += 1
                    end
                end
            end
            next[col] = pos
        end
    end

    return SparseMatrixCSC(nrows, Sp, colptr, rowval, nzval)
end

@inline function _right_kan_constraint_matrix(
    ::Type{K},
    fiber::_KanFiberPlan,
    offp::Vector{Int},
    d::Vector{Int},
    maps_to_succ,
) where {K}
    if _CHANGE_OF_POSETS_USE_RIGHT_KAN_DIRECT_CSC[]
        return _right_kan_constraint_matrix_direct_csc(K, fiber, offp, d, maps_to_succ)
    end
    return _right_kan_constraint_matrix_triplet(K, fiber, offp, d, maps_to_succ)
end

@inline _right_kan_constraint_matrix(M::PModule{K}, fiber::_KanFiberPlan, offp::Vector{Int}) where {K} =
    _right_kan_constraint_matrix(K, fiber, offp, M.dims, M.edge_maps.maps_to_succ)

function _right_kan_edge_map_from_data(
    ::Type{K},
    Vu::Int,
    Vv::Int,
    Ku::AbstractMatrix{K},
    Lv::AbstractMatrix{K},
    offu::Vector{Int},
    offv::Vector{Int},
    fiberv::_KanFiberPlan,
    embed::Vector{Int},
    d::Vector{Int},
) where {K}
    if Vu == 0 || Vv == 0
        return spzeros(K, Vv, Vu)
    end

    Fuv = zeros(K, Vv, Vu)
    @inbounds for loc in eachindex(fiberv.idxs)
        q = fiberv.idxs[loc]
        dq = d[q]
        dq == 0 && continue
        src_local = embed[loc]
        ru = (offu[src_local] + 1):offu[src_local + 1]
        cv = (offv[loc] + 1):offv[loc + 1]
        mul!(Fuv, @view(Lv[:, cv]), @view(Ku[ru, :]), one(K), one(K))
    end
    return sparse(Fuv)
end

# Fiber downset index sets: I_p = { q | pi(q) <= p }
function _index_sets_left(pi::EncodingMap)
    Q = pi.Q
    P = pi.P
    f = pi.pi_of_q

    by_base = [Int[] for _ in 1:nvertices(P)]
    for q in 1:nvertices(Q)
        push!(by_base[f[q]], q)
    end

    idxs = Vector{Vector{Int}}(undef, nvertices(P))
    for p in 1:nvertices(P)
        lst = Int[]
        for v in downset_indices(P, p)
            append!(lst, by_base[v])
        end
        idxs[p] = lst
    end
    return idxs
end

function _left_kan_data(pi::EncodingMap, M::PModule{K};
                        check::Bool=true,
                        threads::Bool = (Threads.nthreads() > 1),
                        session_cache::Union{Nothing,SessionCache}=nothing,
                        plan=nothing) where {K}
    if check
        _check_monotone(pi)
        if !_same_poset(pi.Q, M.Q)
            error("pushforward_left(pi, M): M must be a module on the domain poset pi.Q")
        end
    end

    cache = _translation_cache(session_cache, pi)
    cache_key = _translation_complex_key(:left_kan_data, M, pi, 0)
    cached = _translation_cache_get(cache, cache_key)
    cached !== nothing && return cached

    Q = pi.Q
    P = pi.P
    d = M.dims
    field = M.field

    # Use store-aligned cover-edge traversal to avoid keyed edge lookups in hot loops.
    store = M.edge_maps
    maps_to_succ = store.maps_to_succ

    # Build once; reused many times in the terminal-object fast path.
    cacheQ = _get_cover_cache(Q)

    if plan === nothing && session_cache !== nothing
        plan = _translation_plan(pi; session_cache=session_cache)
    end
    plan === nothing && (plan = _build_pi_translation_plan(pi))
    fibers = plan.left_fibers

    offsets = Vector{Vector{Int}}(undef, nvertices(P))
    dimS = Vector{Int}(undef, nvertices(P))
    W = Vector{Matrix{K}}(undef, nvertices(P))
    L = Vector{Matrix{K}}(undef, nvertices(P))
    dimV = Vector{Int}(undef, nvertices(P))

    @inline function _left_kan_at_p(p::Int)
        fiber = fibers[p]
        ip = fiber.idxs
        offp, Sp = _offset_prefix(ip, d)
        offsets[p] = offp
        dimS[p] = Sp

        if isempty(ip)
            W[p] = zeros(K, 0, 0)
            L[p] = zeros(K, 0, 0)
            dimV[p] = 0
            return
        end

        # Fast path: if fiber has a terminal object qmax, colim = M(qmax).
        qmax = fiber.extremum
        if qmax != 0
            Vp = d[qmax]
            Wp = zeros(K, Sp, Vp)
            Lp = zeros(K, Vp, Sp)

            # W: include the qmax summand (identity block)
            oq = offp[fiber.extremum_local]
            for j in 1:Vp
                Wp[oq + j, j] = one(K)
            end

            # L: send each summand M(q) to M(qmax) via map_leq(q -> qmax)
            maps = SparseMatrixCSC{K,Int}[]
            if fiber.extremal_batch !== nothing
                maps = map_leq_many(M, fiber.extremal_batch; cache=cacheQ)
            end
            midx = 1
            for loc in eachindex(ip)
                q = ip[loc]
                dq = d[q]
                if q == qmax
                    oq = offp[loc]
                    for j in 1:Vp
                        Lp[j, oq + j] = one(K)
                    end
                else
                    A = maps[midx]
                    midx += 1
                    dq == 0 && continue
                    oq = offp[loc]
                    @inbounds for i in 1:Vp
                        for j in 1:dq
                            v = A[i, j]
                            if v != 0
                                Lp[i, oq + j] = v
                            end
                        end
                    end
                end
            end

            W[p] = Wp
            L[p] = Lp
            dimV[p] = Vp
            return
        end

        RelT = _left_kan_relation_transpose(K, fiber, offp, d, maps_to_succ)
        Wp, Lp = _left_kan_quotient_summary(field, RelT)
        Vp = size(Wp, 2)

        W[p] = Wp
        L[p] = Lp
        dimV[p] = Vp
        return
    end

    # Build each colimit space V_p as quotient of direct sum by relations.
    if threads
        Threads.@threads for p in 1:nvertices(P)
            _left_kan_at_p(p)
        end
    else
        for p in 1:nvertices(P)
            _left_kan_at_p(p)
        end
    end

    maps_to_succP = [Vector{SparseMatrixCSC{K,Int}}(undef, length(plan.succsP[u])) for u in 1:nvertices(P)]
    for u in 1:nvertices(P)
        Vu = dimV[u]
        Wu = W[u]
        offu = offsets[u]
        fiberu = fibers[u]
        su = plan.succsP[u]
        embeds_u = plan.left_edge_embeds[u]
        mu = maps_to_succP[u]
        for j in eachindex(su)
            v = su[j]
            Vv = dimV[v]
            if Vu == 0 || Vv == 0
                mu[j] = spzeros(K, Vv, Vu)
                continue
            end

            Fuv = zeros(K, Vv, Vu)
            offv = offsets[v]
            Lv = L[v]
            embed = embeds_u[j]
            for loc in eachindex(fiberu.idxs)
                q = fiberu.idxs[loc]
                dq = d[q]
                dq == 0 && continue
                dst_local = embed[loc]
                ru = (offu[loc] + 1):offu[loc + 1]
                cv = (offv[dst_local] + 1):offv[dst_local + 1]
                @views Fuv .+= Lv[:, cv] * Wu[ru, :]
            end
            mu[j] = sparse(Fuv)
        end
    end

    store_out = _cover_store_from_succ_maps(K, plan.predsP, plan.succsP, plan.pred_slotsP, maps_to_succP)
    Mout = PModule{K}(P, dimV, store_out; field=M.field)
    data = LeftKanData(offsets, dimS, W, L, dimV)
    return _translation_cache_set!(cache, cache_key, (Mout, data))
end

"""
    pushforward_left(pi::EncodingMap, M::PModule; check=true, session_cache=nothing) -> PModule

Left Kan extension (left pushforward) of a module along `pi: Q -> P`.

At `p in P` this is the colimit over the fiber downset:

    I_p = { q in Q | pi(q) <= p }.

This is left adjoint to `pullback(pi, -)`.

Notation policy
- `pushforward_left` is the canonical notebook-facing name.
- [`left_kan_extension`](@ref) is the categorical alias for the same kernel.

Cache semantics
- with `session_cache=nothing`, this is a one-shot left Kan computation,
- with `session_cache=sc::SessionCache`, translation plans and intermediate
  colimit data are reused across repeated calls.

Cheap-first workflow
- start with `describe(pushforward_left(pi, M))`,
- only then feed the result into heavier derived or Hom calculations.

Fast path: if `I_p` has a maximum element, the colimit equals `M(max)`.

# Examples

```jldoctest
julia> using TamerOp

julia> const CO = TamerOp.ChangeOfPosets;

julia> const EN = TamerOp.Encoding;

julia> const FF = TamerOp.FiniteFringe;

julia> const IR = TamerOp.IndicatorResolutions;

julia> field = TamerOp.CoreModules.QQField();

julia> Q = FF.FinitePoset(Bool[1 1; 0 1]);

julia> P = FF.FinitePoset(Bool[1]);

julia> pi = EN.EncodingMap(Q, P, [1, 1]);

julia> M = IR.pmodule_from_fringe(FF.one_by_one_fringe(Q,
                                                      FF.principal_upset(Q, 1),
                                                      FF.principal_downset(Q, 2),
                                                      1//1; field=field));

julia> L = CO.pushforward_left(pi, M);

julia> TamerOp.describe(L).vertices
1
```
"""
function pushforward_left(pi::EncodingMap, M::PModule{K};
                          check::Bool=true,
                          threads::Bool = (Threads.nthreads() > 1),
                          session_cache::Union{Nothing,SessionCache}=nothing)::PModule{K} where {K}
    Mout, _ = _left_kan_data(pi, M; check=check, threads=threads, session_cache=session_cache)
    return Mout
end

"""
    pushforward_left(pi::EncodingMap, f::PMorphism; check=true, session_cache=nothing) -> PMorphism

Push forward a morphism of modules along `pi` using left Kan extension.

The returned morphism lives between `pushforward_left(pi, f.dom)` and
`pushforward_left(pi, f.cod)`.

Source/target convention
- `pi.Q` is the source poset of the monotone map,
- `pi.P` is the target poset,
- `f` must be a morphism of modules on `pi.Q`,
- the returned morphism lives on `pi.P`.

Notation policy
- `pushforward_left` is the canonical notebook-facing name,
- [`left_kan_extension`](@ref) is the categorical alias for the same
  underlying module-level construction.

Cache semantics
- with `session_cache=sc::SessionCache`, repeated left-pushforward morphism
  computations reuse translation plans and pushed-forward domain/codomain
  modules.
"""
function pushforward_left(pi::EncodingMap, f::PMorphism{K};
                          check::Bool=true,
                          threads::Bool = (Threads.nthreads() > 1),
                          session_cache::Union{Nothing,SessionCache}=nothing)::PMorphism{K} where {K}
    if check
        _check_monotone(pi)
        if !_same_poset(pi.Q, f.dom.Q) || !_same_poset(pi.Q, f.cod.Q)
            error("pushforward_left(pi, f): f must be a morphism on the domain poset pi.Q")
        end
    end

    cache = _translation_cache(session_cache, pi)
    key = _translation_morphism_key(:pushforward_left_morphism, pi, f)
    cached = _translation_cache_get(cache, key)
    cached !== nothing && return cached
    plan = _translation_plan(pi; session_cache=session_cache)
    dom_out, data_dom = _left_kan_data(pi, f.dom; check=false, threads=threads,
                                       session_cache=session_cache, plan=plan)
    cod_out, data_cod = _left_kan_data(pi, f.cod; check=false, threads=threads,
                                       session_cache=session_cache, plan=plan)
    return _translation_cache_set!(
        cache,
        key,
        _pushforward_left_morphism_from_data(dom_out, cod_out, f, data_dom, data_cod, plan.left_fibers; threads=threads),
    )
end

function _pushforward_left_morphism_from_data(
    dom_out::PModule{K},
    cod_out::PModule{K},
    f::PMorphism{K},
    data_dom::LeftKanData{K},
    data_cod::LeftKanData{K},
    fibers::Vector{_KanFiberPlan};
    threads::Bool,
) where {K}
    comps = Vector{Matrix{K}}(undef, length(fibers))

    @inline function _left_pushforward_comp(p::Int)
        Vd = data_dom.dimV[p]
        Vc = data_cod.dimV[p]
        if Vd == 0 || Vc == 0
            comps[p] = zeros(K, Vc, Vd)
            return
        end

        Fp = zeros(K, Vc, Vd)
        fiber = fibers[p]
        offs_dom = data_dom.offsets[p]
        offs_cod = data_cod.offsets[p]
        Lp = data_cod.L[p]
        Wp = data_dom.W[p]

        if !_CHANGE_OF_POSETS_USE_LEFT_KAN_MORPHISM_MUL[]
            @inbounds for loc in eachindex(fiber.idxs)
                q = fiber.idxs[loc]
                dd = f.dom.dims[q]
                dc = f.cod.dims[q]
                (dd == 0 || dc == 0) && continue

                rd = (offs_dom[loc] + 1):offs_dom[loc + 1]
                cc = (offs_cod[loc] + 1):offs_cod[loc + 1]
                @views Fp .+= Lp[:, cc] * (f.comps[q] * Wp[rd, :])
            end
            comps[p] = Fp
            return
        end

        max_dc = 0
        @inbounds for q in fiber.idxs
            max_dc = max(max_dc, f.cod.dims[q])
        end
        tmp = zeros(K, max_dc, Vd)

        @inbounds for loc in eachindex(fiber.idxs)
            q = fiber.idxs[loc]
            dd = f.dom.dims[q]
            dc = f.cod.dims[q]
            (dd == 0 || dc == 0) && continue

            rd = (offs_dom[loc] + 1):offs_dom[loc + 1]
            cc = (offs_cod[loc] + 1):offs_cod[loc + 1]
            tmpv = @view tmp[1:dc, :]
            Lblock = @view Lp[:, cc]
            Wblock = @view Wp[rd, :]
            mul!(tmpv, f.comps[q], Wblock)
            mul!(Fp, Lblock, tmpv, one(K), one(K))
        end
        comps[p] = Fp
        return
    end

    if threads
        Threads.@threads for p in eachindex(fibers)
            _left_pushforward_comp(p)
        end
    else
        for p in eachindex(fibers)
            _left_pushforward_comp(p)
        end
    end

    return PMorphism(dom_out, cod_out, comps)
end

"""
    left_kan_extension(pi::EncodingMap, M::PModule; check=true, session_cache=nothing) -> PModule

Categorical alias for [`pushforward_left`](@ref).

Prefer `pushforward_left` in notebook-facing code when you want the operation
to read as a left pushforward rather than as a Kan-extension phrase.

Source/target convention
- `pi : Q -> P`,
- `M` must live on `Q = pi.Q`,
- the returned module lives on `P = pi.P`.

Cache semantics
- `left_kan_extension(...; session_cache=sc)` uses the same cached left Kan
  plans and translated modules as `pushforward_left`.
"""
left_kan_extension(pi::EncodingMap, M::PModule{K};
                   check::Bool=true,
                   threads::Bool = (Threads.nthreads() > 1),
                   session_cache::Union{Nothing,SessionCache}=nothing) where {K} =
    pushforward_left(pi, M; check=check, threads=threads, session_cache=session_cache)

# -----------------------------------------------------------------------------
# Right Kan extension (right pushforward)
# -----------------------------------------------------------------------------

# Fiber upset index sets: J_p = { q | p <= pi(q) }
function _index_sets_right(pi::EncodingMap)
    Q = pi.Q
    P = pi.P
    f = pi.pi_of_q

    by_base = [Int[] for _ in 1:nvertices(P)]
    for q in 1:nvertices(Q)
        push!(by_base[f[q]], q)
    end

    idxs = Vector{Vector{Int}}(undef, nvertices(P))
    for p in 1:nvertices(P)
        lst = Int[]
        for v in upset_indices(P, p)
            append!(lst, by_base[v])
        end
        idxs[p] = lst
    end
    return idxs
end

function _build_kan_fiber_plans(
    idxs::Vector{Vector{Int}},
    extremum::Vector{Int},
    succsQ::Vector{Vector{Int}},
)
    nP = length(idxs)
    nQ = isempty(succsQ) ? 0 : length(succsQ)
    local_pos = zeros(Int, nQ)
    plans = Vector{_KanFiberPlan}(undef, nP)
    @inbounds for p in 1:nP
        idxp = idxs[p]
        for loc in eachindex(idxp)
            q = idxp[loc]
            local_pos[q] = loc
        end

        edge_u = Int[]
        edge_v = Int[]
        edge_u_local = Int[]
        edge_v_local = Int[]
        edge_slot = Int[]
        for u_local in eachindex(idxp)
            u = idxp[u_local]
            su = succsQ[u]
            for slot in eachindex(su)
                v = su[slot]
                v_local = local_pos[v]
                v_local == 0 && continue
                push!(edge_u, u)
                push!(edge_v, v)
                push!(edge_u_local, u_local)
                push!(edge_v_local, v_local)
                push!(edge_slot, slot)
            end
        end

        ext = extremum[p]
        ext_local = ext == 0 ? 0 : local_pos[ext]
        batch = nothing
        if ext != 0
            pairs = Tuple{Int,Int}[]
            sizehint!(pairs, max(length(idxp) - 1, 0))
            for q in idxp
                q == ext && continue
                push!(pairs, (q, ext))
            end
            isempty(pairs) || (batch = _prepare_map_leq_batch_owned(pairs))
        end
        out_ptr, out_idx = _packed_local_edge_lists(length(idxp), edge_u_local)
        in_ptr, in_idx = _packed_local_edge_lists(length(idxp), edge_v_local)
        plans[p] = _KanFiberPlan(idxp, edge_u, edge_v, edge_u_local, edge_v_local, edge_slot,
                                 out_ptr, out_idx, in_ptr, in_idx,
                                 ext, ext_local, batch)

        for q in idxp
            local_pos[q] = 0
        end
    end
    return plans
end

function _build_right_kan_fiber_plans(
    idxs::Vector{Vector{Int}},
    minimum::Vector{Int},
    succsQ::Vector{Vector{Int}},
)
    nP = length(idxs)
    nQ = isempty(succsQ) ? 0 : length(succsQ)
    local_pos = zeros(Int, nQ)
    plans = Vector{_KanFiberPlan}(undef, nP)
    @inbounds for p in 1:nP
        idxp = idxs[p]
        for loc in eachindex(idxp)
            q = idxp[loc]
            local_pos[q] = loc
        end

        edge_u = Int[]
        edge_v = Int[]
        edge_u_local = Int[]
        edge_v_local = Int[]
        edge_slot = Int[]
        for u_local in eachindex(idxp)
            u = idxp[u_local]
            su = succsQ[u]
            for slot in eachindex(su)
                v = su[slot]
                v_local = local_pos[v]
                v_local == 0 && continue
                push!(edge_u, u)
                push!(edge_v, v)
                push!(edge_u_local, u_local)
                push!(edge_v_local, v_local)
                push!(edge_slot, slot)
            end
        end

        ext = minimum[p]
        ext_local = ext == 0 ? 0 : local_pos[ext]
        batch = nothing
        if ext != 0
            pairs = Tuple{Int,Int}[]
            sizehint!(pairs, max(length(idxp) - 1, 0))
            for q in idxp
                q == ext && continue
                push!(pairs, (ext, q))
            end
            isempty(pairs) || (batch = _prepare_map_leq_batch_owned(pairs))
        end
        out_ptr, out_idx = _packed_local_edge_lists(length(idxp), edge_u_local)
        in_ptr, in_idx = _packed_local_edge_lists(length(idxp), edge_v_local)
        plans[p] = _KanFiberPlan(idxp, edge_u, edge_v, edge_u_local, edge_v_local, edge_slot,
                                 out_ptr, out_idx, in_ptr, in_idx,
                                 ext, ext_local, batch)

        for q in idxp
            local_pos[q] = 0
        end
    end
    return plans
end

function _build_left_edge_embeds(
    left_fibers::Vector{_KanFiberPlan},
    succsP::Vector{Vector{Int}},
)
    nQ = isempty(left_fibers) ? 0 : maximum((isempty(fp.idxs) ? 0 : maximum(fp.idxs) for fp in left_fibers))
    local_pos = zeros(Int, nQ)
    embeds = [Vector{Vector{Int}}(undef, length(succsP[u])) for u in 1:length(succsP)]
    @inbounds for u in 1:length(succsP)
        su = succsP[u]
        for j in eachindex(su)
            v = su[j]
            idxv = left_fibers[v].idxs
            for loc in eachindex(idxv)
                q = idxv[loc]
                local_pos[q] = loc
            end
            idxu = left_fibers[u].idxs
            mapuv = Vector{Int}(undef, length(idxu))
            for i in eachindex(idxu)
                mapuv[i] = local_pos[idxu[i]]
            end
            embeds[u][j] = mapuv
            for q in idxv
                local_pos[q] = 0
            end
        end
    end
    return embeds
end

function _build_right_edge_embeds(
    right_fibers::Vector{_KanFiberPlan},
    succsP::Vector{Vector{Int}},
)
    nQ = isempty(right_fibers) ? 0 : maximum((isempty(fp.idxs) ? 0 : maximum(fp.idxs) for fp in right_fibers))
    local_pos = zeros(Int, nQ)
    embeds = [Vector{Vector{Int}}(undef, length(succsP[u])) for u in 1:length(succsP)]
    @inbounds for u in 1:length(succsP)
        su = succsP[u]
        idxu = right_fibers[u].idxs
        for loc in eachindex(idxu)
            q = idxu[loc]
            local_pos[q] = loc
        end
        for j in eachindex(su)
            v = su[j]
            idxv = right_fibers[v].idxs
            mapvu = Vector{Int}(undef, length(idxv))
            for i in eachindex(idxv)
                mapvu[i] = local_pos[idxv[i]]
            end
            embeds[u][j] = mapvu
        end
        for q in idxu
            local_pos[q] = 0
        end
    end
    return embeds
end

struct _PiTranslationPlan{PB}
    coverQ_edges::Vector{Tuple{Int,Int}}
    pullback_prepared::Tuple{Vector{Tuple{Int,Int}},PB}
    coverP_edges::Vector{Tuple{Int,Int}}
    predsP::Vector{Vector{Int}}
    succsP::Vector{Vector{Int}}
    pred_slotsP::Vector{Vector{Int}}
    left_idxs::Vector{Vector{Int}}
    left_terminal::Vector{Int}
    left_fibers::Vector{_KanFiberPlan}
    left_edge_embeds::Vector{Vector{Vector{Int}}}
    right_idxs::Vector{Vector{Int}}
    right_minimum::Vector{Int}
    right_fibers::Vector{_KanFiberPlan}
    right_edge_embeds::Vector{Vector{Vector{Int}}}
end

function _build_pi_translation_plan(pi::EncodingMap)
    succsQ = [collect(_succs(_get_cover_cache(pi.Q), u)) for u in 1:nvertices(pi.Q)]
    coverQ_edges = copy(cover_edges(pi.Q).edges)
    predsP, succsP, pred_slotsP = _cover_store_layout(pi.P)
    coverP_edges = copy(cover_edges(pi.P).edges)
    left_idxs = _index_sets_left(pi)
    left_terminal = Vector{Int}(undef, nvertices(pi.P))
    @inbounds for p in 1:nvertices(pi.P)
        q = _maximum_element(pi.Q, left_idxs[p])
        left_terminal[p] = q === nothing ? 0 : q
    end
    left_fibers = _build_kan_fiber_plans(left_idxs, left_terminal, succsQ)
    left_edge_embeds = _build_left_edge_embeds(left_fibers, succsP)
    right_idxs = _index_sets_right(pi)
    right_minimum = Vector{Int}(undef, nvertices(pi.P))
    @inbounds for p in 1:nvertices(pi.P)
        q = _minimum_element(pi.Q, right_idxs[p])
        right_minimum[p] = q === nothing ? 0 : q
    end
    right_fibers = _build_right_kan_fiber_plans(right_idxs, right_minimum, succsQ)
    right_edge_embeds = _build_right_edge_embeds(right_fibers, succsP)
    return _PiTranslationPlan(
        coverQ_edges,
        _pullback_pair_data(pi, coverQ_edges),
        coverP_edges,
        predsP,
        succsP,
        pred_slotsP,
        left_idxs,
        left_terminal,
        left_fibers,
        left_edge_embeds,
        right_idxs,
        right_minimum,
        right_fibers,
        right_edge_embeds,
    )
end

@inline function _translation_plan(pi::EncodingMap;
                                   session_cache::Union{Nothing,SessionCache}=nothing)
    cache = _translation_cache(session_cache, pi)
    key = (:pi_translation_plan,)
    cached = _translation_cache_get(cache, key)
    cached !== nothing && return cached
    return _translation_cache_set!(cache, key, _build_pi_translation_plan(pi))
end

function _right_kan_data(pi::EncodingMap, M::PModule{K};
                         check::Bool=true,
                         threads::Bool = (Threads.nthreads() > 1),
                         session_cache::Union{Nothing,SessionCache}=nothing,
                         plan=nothing) where {K}
    if check
        _check_monotone(pi)
        if !_same_poset(pi.Q, M.Q)
            error("pushforward_right(pi, M): M must be a module on the domain poset pi.Q")
        end
    end

    cache = _translation_cache(session_cache, pi)
    cache_key = _translation_complex_key(:right_kan_data, M, pi, 0)
    cached = _translation_cache_get(cache, cache_key)
    cached !== nothing && return cached

    Q = pi.Q
    P = pi.P
    d = M.dims
    field = M.field

    # Use store-aligned cover-edge traversal to avoid keyed edge lookups in hot loops.
    store = M.edge_maps
    maps_to_succ = store.maps_to_succ

    # Build once; reused many times in the terminal-object fast path.
    cacheQ = _get_cover_cache(Q)

    if plan === nothing && session_cache !== nothing
        plan = _translation_plan(pi; session_cache=session_cache)
    end
    plan === nothing && (plan = _build_pi_translation_plan(pi))
    fibers = plan.right_fibers

    offsets = Vector{Vector{Int}}(undef, nvertices(P))
    dimS = Vector{Int}(undef, nvertices(P))
    Ksec = Vector{Matrix{K}}(undef, nvertices(P))
    L = Vector{Matrix{K}}(undef, nvertices(P))
    dimV = Vector{Int}(undef, nvertices(P))

    @inline function _right_kan_at_p(p::Int)
        fiber = fibers[p]
        jp = fiber.idxs
        offp, Sp = _offset_prefix(jp, d)
        offsets[p] = offp
        dimS[p] = Sp

        if isempty(jp)
            Ksec[p] = zeros(K, 0, 0)
            L[p] = zeros(K, 0, 0)
            dimV[p] = 0
            return
        end

        # Fast path: if fiber has an initial object qmin, limit = M(qmin).
        qmin = fiber.extremum
        if qmin != 0
            Vp = d[qmin]
            Kp = zeros(K, Sp, Vp)
            Lp = zeros(K, Vp, Sp)

            # K: cone embedding M(qmin) -> oplus_{q in jp} M(q), q component is map(qmin->q).
            maps = SparseMatrixCSC{K,Int}[]
            if fiber.extremal_batch !== nothing
                maps = map_leq_many(M, fiber.extremal_batch; cache=cacheQ)
            end
            midx = 1
            for loc in eachindex(jp)
                q = jp[loc]
                dq = d[q]
                dq == 0 && continue
                oq = offp[loc]
                if q == qmin
                    for j in 1:Vp
                        Kp[oq + j, j] = one(K)
                    end
                else
                    A = maps[midx]
                    midx += 1
                    dq == 0 && continue
                    nr = min(dq, size(A, 1))
                    nc = min(Vp, size(A, 2))
                    (nr == 0 || nc == 0) && continue
                    @inbounds for i in 1:nr
                        for j in 1:nc
                            v = A[i, j]
                            if v != 0
                                Kp[oq + i, j] = v
                            end
                        end
                    end
                end
            end

            # L: projection to qmin component
            oq = offp[fiber.extremum_local]
            for j in 1:Vp
                Lp[j, oq + j] = one(K)
            end

            Ksec[p] = Kp
            L[p] = Lp
            dimV[p] = Vp
            return
        end

        C = _right_kan_constraint_matrix(K, fiber, offp, d, maps_to_succ)
        Kp, Lp = _nullspace_selector_summary(field, C)
        Vp = size(Kp, 2)

        Ksec[p] = Kp
        L[p] = Lp
        dimV[p] = Vp
        return
    end

    if threads
        Threads.@threads for p in 1:nvertices(P)
            _right_kan_at_p(p)
        end
    else
        for p in 1:nvertices(P)
            _right_kan_at_p(p)
        end
    end

    maps_to_succP = [Vector{SparseMatrixCSC{K,Int}}(undef, length(plan.succsP[u])) for u in 1:nvertices(P)]
    for u in 1:nvertices(P)
        Vu = dimV[u]
        Ku = Ksec[u]
        offu = offsets[u]
        su = plan.succsP[u]
        embeds_u = plan.right_edge_embeds[u]
        mu = maps_to_succP[u]
        for j in eachindex(su)
            v = su[j]
            Vv = dimV[v]
            if Vu == 0 || Vv == 0
                mu[j] = spzeros(K, Vv, Vu)
                continue
            end

            offv = offsets[v]
            fiberv = fibers[v]
            Lv = L[v]
            embed = embeds_u[j]
            mu[j] = _right_kan_edge_map_from_data(K, Vu, Vv, Ku, Lv, offu, offv, fiberv, embed, d)
        end
    end

    store_out = _cover_store_from_succ_maps(K, plan.predsP, plan.succsP, plan.pred_slotsP, maps_to_succP)
    Mout = PModule{K}(P, dimV, store_out; field=M.field)
    data = RightKanData(offsets, dimS, Ksec, L, dimV)
    return _translation_cache_set!(cache, cache_key, (Mout, data))
end

"""
    pushforward_right(pi::EncodingMap, M::PModule; check=true, session_cache=nothing) -> PModule

Right Kan extension (right pushforward) along `pi: Q -> P`.

At `p in P` this is the limit over the fiber upset:

    J_p = { q in Q | p <= pi(q) }.

This is right adjoint to `pullback(pi, -)`.

Notation policy
- `pushforward_right` is the canonical notebook-facing name.
- [`right_kan_extension`](@ref) is the categorical alias for the same kernel.

Cache semantics
- with `session_cache=nothing`, this is a one-shot right Kan computation,
- with `session_cache=sc::SessionCache`, translation plans and intermediate
  limit data are reused across repeated calls.

Cheap-first workflow
- start with `describe(pushforward_right(pi, M))`,
- then move on to heavier derived constructions only if needed.

Fast path: if `J_p` has a minimum element, the limit equals `M(min)`.

# Examples

```jldoctest
julia> using TamerOp

julia> const CO = TamerOp.ChangeOfPosets;

julia> const EN = TamerOp.Encoding;

julia> const FF = TamerOp.FiniteFringe;

julia> const IR = TamerOp.IndicatorResolutions;

julia> field = TamerOp.CoreModules.QQField();

julia> Q = FF.FinitePoset(Bool[1 1; 0 1]);

julia> P = FF.FinitePoset(Bool[1]);

julia> pi = EN.EncodingMap(Q, P, [1, 1]);

julia> M = IR.pmodule_from_fringe(FF.one_by_one_fringe(Q,
                                                      FF.principal_upset(Q, 1),
                                                      FF.principal_downset(Q, 2),
                                                      1//1; field=field));

julia> R = CO.pushforward_right(pi, M);

julia> TamerOp.describe(R).vertices
1
```
"""
function pushforward_right(pi::EncodingMap, M::PModule{K};
                           check::Bool=true,
                           threads::Bool = (Threads.nthreads() > 1),
                           session_cache::Union{Nothing,SessionCache}=nothing)::PModule{K} where {K}
    Mout, _ = _right_kan_data(pi, M; check=check, threads=threads, session_cache=session_cache)
    return Mout
end

"""
    pushforward_right(pi::EncodingMap, f::PMorphism; check=true, session_cache=nothing) -> PMorphism

Push forward a morphism of modules along `pi` using right Kan extension.

The returned morphism lives between `pushforward_right(pi, f.dom)` and
`pushforward_right(pi, f.cod)`.

Source/target convention
- `pi.Q` is the source poset of the monotone map,
- `pi.P` is the target poset,
- `f` must be a morphism of modules on `pi.Q`,
- the returned morphism lives on `pi.P`.

Notation policy
- `pushforward_right` is the canonical notebook-facing name,
- [`right_kan_extension`](@ref) is the categorical alias for the same
  underlying module-level construction.

Cache semantics
- with `session_cache=sc::SessionCache`, repeated right-pushforward morphism
  computations reuse translation plans and pushed-forward domain/codomain
  modules.
"""
function pushforward_right(pi::EncodingMap, f::PMorphism{K};
                           check::Bool=true,
                           threads::Bool = (Threads.nthreads() > 1),
                           session_cache::Union{Nothing,SessionCache}=nothing)::PMorphism{K} where {K}
    if check
        _check_monotone(pi)
        if !_same_poset(pi.Q, f.dom.Q) || !_same_poset(pi.Q, f.cod.Q)
            error("pushforward_right(pi, f): f must be a morphism on the domain poset pi.Q")
        end
    end

    cache = _translation_cache(session_cache, pi)
    key = _translation_morphism_key(:pushforward_right_morphism, pi, f)
    cached = _translation_cache_get(cache, key)
    cached !== nothing && return cached
    plan = _translation_plan(pi; session_cache=session_cache)
    dom_out, data_dom = _right_kan_data(pi, f.dom; check=false, threads=threads,
                                        session_cache=session_cache, plan=plan)
    cod_out, data_cod = _right_kan_data(pi, f.cod; check=false, threads=threads,
                                        session_cache=session_cache, plan=plan)
    return _translation_cache_set!(
        cache,
        key,
        _pushforward_right_morphism_from_data(dom_out, cod_out, f, data_dom, data_cod, plan.right_fibers; threads=threads),
    )
end

"""
    right_kan_extension(pi::EncodingMap, M::PModule; check=true, session_cache=nothing) -> PModule

Categorical alias for [`pushforward_right`](@ref).

Prefer `pushforward_right` in notebook-facing code when you want the operation
to read as a right pushforward rather than as a Kan-extension phrase.

Source/target convention
- `pi : Q -> P`,
- `M` must live on `Q = pi.Q`,
- the returned module lives on `P = pi.P`.

Cache semantics
- `right_kan_extension(...; session_cache=sc)` uses the same cached right Kan
  plans and translated modules as `pushforward_right`.
"""
right_kan_extension(pi::EncodingMap, M::PModule{K};
                    check::Bool=true,
                    threads::Bool = (Threads.nthreads() > 1),
                    session_cache::Union{Nothing,SessionCache}=nothing) where {K} =
    pushforward_right(pi, M; check=check, threads=threads, session_cache=session_cache)

function _pushforward_right_morphism_from_data(
    dom_out::PModule{K},
    cod_out::PModule{K},
    f::PMorphism{K},
    data_dom::RightKanData{K},
    data_cod::RightKanData{K},
    fibers::Vector{_KanFiberPlan};
    threads::Bool,
) where {K}
    comps = Vector{Matrix{K}}(undef, length(fibers))

    @inline function _right_pushforward_comp(p::Int)
        Vd = data_dom.dimV[p]
        Vc = data_cod.dimV[p]
        if Vd == 0 || Vc == 0
            comps[p] = zeros(K, Vc, Vd)
            return
        end

        Fp = zeros(K, Vc, Vd)
        fiber = fibers[p]
        offs_dom = data_dom.offsets[p]
        offs_cod = data_cod.offsets[p]
        Lp = data_cod.L[p]
        Kp = data_dom.Ksec[p]

        max_dc = 0
        @inbounds for q in fiber.idxs
            max_dc = max(max_dc, f.cod.dims[q])
        end
        tmp = zeros(K, max_dc, Vd)

        @inbounds for loc in eachindex(fiber.idxs)
            q = fiber.idxs[loc]
            dd = f.dom.dims[q]
            dc = f.cod.dims[q]
            (dd == 0 || dc == 0) && continue

            rd = (offs_dom[loc] + 1):offs_dom[loc + 1]
            cc = (offs_cod[loc] + 1):offs_cod[loc + 1]
            tmpv = @view tmp[1:dc, :]
            Lblock = @view Lp[:, cc]
            Kblock = @view Kp[rd, :]
            mul!(tmpv, f.comps[q], Kblock)
            mul!(Fp, Lblock, tmpv, one(K), one(K))
        end
        comps[p] = Fp
        return
    end

    if threads
        Threads.@threads for p in eachindex(fibers)
            _right_pushforward_comp(p)
        end
    else
        for p in eachindex(fibers)
            _right_pushforward_comp(p)
        end
    end

    return PMorphism(dom_out, cod_out, comps)
end

function _pushforward_left_resolution_terms_data(
    pi::EncodingMap,
    pmods::AbstractVector{<:PModule{K}};
    threads::Bool,
    session_cache::Union{Nothing,SessionCache},
    plan,
) where {K}
    n = length(pmods)
    terms = Vector{PModule{K}}(undef, n)
    data = Vector{LeftKanData{K}}(undef, n)
    @inbounds for k in 1:n
        term_k, data_k = _left_kan_data(pi, pmods[k];
                                        check=false,
                                        threads=threads,
                                        session_cache=session_cache,
                                        plan=plan)
        idx = n - k + 1
        terms[idx] = term_k
        data[idx] = data_k
    end
    return terms, data
end

function _pushforward_left_resolution_diffs_from_data(
    pmods::AbstractVector{<:PModule{K}},
    diffs_in::AbstractVector{<:PMorphism{K}},
    terms::Vector{PModule{K}},
    data::Vector{LeftKanData{K}},
    fibers::Vector{_KanFiberPlan};
    threads::Bool,
) where {K}
    maxlen = length(diffs_in)
    diffs = Vector{PMorphism{K}}(undef, maxlen)
    idx_by_module = IdDict{PModule{K},Int}()
    @inbounds for k in eachindex(pmods)
        idx_by_module[pmods[k]] = length(pmods) - k + 1
    end
    @inbounds for k in 1:maxlen
        slot = maxlen - k + 1
        dom_idx = idx_by_module[diffs_in[k].dom]
        cod_idx = idx_by_module[diffs_in[k].cod]
        diffs[slot] = _pushforward_left_morphism_from_data(
            terms[dom_idx],
            terms[cod_idx],
            diffs_in[k],
            data[dom_idx],
            data[cod_idx],
            fibers;
            threads=threads,
        )
    end
    return diffs
end

function _pushforward_left_complex_data(
    pi::EncodingMap,
    f::PMorphism{K},
    df::DerivedFunctorOptions;
    res_dom,
    res_cod,
    threads::Bool,
    session_cache::Union{Nothing,SessionCache},
) where {K}
    maxlen = df.maxdeg + 1
    cache = _translation_cache(session_cache, pi)
    key = _translation_derived_map_key(:pushforward_left_complex_data, f, res_dom, res_cod, df.maxdeg)
    cached = _translation_cache_get(cache, key)
    cached !== nothing && return cached

    plan = _translation_plan(pi; session_cache=session_cache)
    terms_dom, data_dom = _pushforward_left_resolution_terms_data(
        pi, res_dom.Pmods;
        threads=threads,
        session_cache=session_cache,
        plan=plan,
    )
    terms_cod, data_cod = _pushforward_left_resolution_terms_data(
        pi, res_cod.Pmods;
        threads=threads,
        session_cache=session_cache,
        plan=plan,
    )

    if _CHANGE_OF_POSETS_USE_LEFT_DERIVED_DIFFS_FROM_DATA[]
        diffs_dom = _pushforward_left_resolution_diffs_from_data(
            res_dom.Pmods, res_dom.d_mor, terms_dom, data_dom, plan.left_fibers; threads=threads,
        )
        diffs_cod = _pushforward_left_resolution_diffs_from_data(
            res_cod.Pmods, res_cod.d_mor, terms_cod, data_cod, plan.left_fibers; threads=threads,
        )
    else
        diffs_dom = Vector{PMorphism{K}}(undef, maxlen)
        diffs_cod = Vector{PMorphism{K}}(undef, maxlen)
        for k in 1:maxlen
            idx = maxlen - k + 1
            diffs_dom[idx] = pushforward_left(pi, res_dom.d_mor[k];
                                              check=false,
                                              threads=threads,
                                              session_cache=session_cache)
            diffs_cod[idx] = pushforward_left(pi, res_cod.d_mor[k];
                                              check=false,
                                              threads=threads,
                                              session_cache=session_cache)
        end
    end

    Cdom = ModuleCochainComplex(terms_dom, diffs_dom; tmin=-maxlen, check=false)
    Ccod = ModuleCochainComplex(terms_cod, diffs_cod; tmin=-maxlen, check=false)

    H = lift_chainmap(res_dom, res_cod, f; maxlen=maxlen)
    comps = Vector{PMorphism{K}}(undef, maxlen + 1)
    if _CHANGE_OF_POSETS_USE_DERIVED_LEFT_COEFF_FASTPATH[]
        coeff_plans = Vector{_LeftCoeffPlan}(undef, maxlen + 1)
        for k in 0:maxlen
            idx = maxlen - k + 1
            coeff_plans[idx] = _left_pushforward_coeff_plan(
                pi, res_dom.gens[k + 1], res_cod.gens[k + 1], plan;
                session_cache=session_cache,
            )
        end
        for k in 0:maxlen
            idx = maxlen - k + 1
            comps[idx] = _pushforward_left_from_upset_coeff(
                pi,
                terms_dom[idx],
                terms_cod[idx],
                data_dom[idx],
                data_cod[idx],
                res_dom.gens[k + 1],
                res_cod.gens[k + 1],
                H[k + 1],
                plan;
                coeff_plan=coeff_plans[idx],
                session_cache=session_cache,
            )
        end
    else
        phi = Vector{PMorphism{K}}(undef, maxlen + 1)
        for k in 0:maxlen
            phi[k + 1] = _pmorphism_from_upset_coeff(res_dom.Pmods[k + 1], res_cod.Pmods[k + 1],
                                                     res_dom.gens[k + 1], res_cod.gens[k + 1],
                                                     H[k + 1])
        end
        for k in 0:maxlen
            idx = maxlen - k + 1
            comps[idx] = _pushforward_left_morphism_from_data(
                terms_dom[idx],
                terms_cod[idx],
                phi[k + 1],
                data_dom[idx],
                data_cod[idx],
                plan.left_fibers;
                threads=threads,
            )
        end
    end

    return _translation_cache_set!(cache, key, (Cdom, Ccod, comps))
end

function _pushforward_right_resolution_terms_data(
    pi::EncodingMap,
    emods::AbstractVector{<:PModule{K}};
    threads::Bool,
    session_cache::Union{Nothing,SessionCache},
    plan,
) where {K}
    n = length(emods)
    terms = Vector{PModule{K}}(undef, n)
    data = Vector{RightKanData{K}}(undef, n)
    @inbounds for k in 1:n
        terms[k], data[k] = _right_kan_data(pi, emods[k];
                                            check=false,
                                            threads=threads,
                                            session_cache=session_cache,
                                            plan=plan)
    end
    return terms, data
end

function _pushforward_right_resolution_diffs_from_data(
    emods::AbstractVector{<:PModule{K}},
    diffs_in::AbstractVector{<:PMorphism{K}},
    terms::Vector{PModule{K}},
    data::Vector{RightKanData{K}},
    fibers::Vector{_KanFiberPlan};
    threads::Bool,
) where {K}
    maxlen = length(diffs_in)
    diffs = Vector{PMorphism{K}}(undef, maxlen)
    idx_by_module = IdDict{PModule{K},Int}()
    @inbounds for k in eachindex(emods)
        idx_by_module[emods[k]] = k
    end
    @inbounds for k in 1:maxlen
        diff = diffs_in[k]
        dom_idx = idx_by_module[diff.dom]
        cod_idx = idx_by_module[diff.cod]
        diffs[k] = _pushforward_right_morphism_from_data(
            terms[dom_idx],
            terms[cod_idx],
            diff,
            data[dom_idx],
            data[cod_idx],
            fibers;
            threads=threads,
        )
    end
    return diffs
end

@inline function _is_identity_matrix(A::AbstractMatrix{K}) where {K}
    m, n = size(A)
    m == n || return false
    @inbounds for j in 1:n
        for i in 1:m
            a = A[i, j]
            if i == j
                a == one(K) || return false
            else
                iszero(a) || return false
            end
        end
    end
    return true
end

function _is_identity_pmodule_morphism(f::PMorphism{K}) where {K}
    f.dom === f.cod || return false
    @inbounds for u in eachindex(f.comps)
        _is_identity_matrix(f.comps[u]) || return false
    end
    return true
end

function _pushforward_right_complex_data(
    pi::EncodingMap,
    f::PMorphism{K},
    df::DerivedFunctorOptions;
    res_dom,
    res_cod,
    threads::Bool,
    session_cache::Union{Nothing,SessionCache},
) where {K}
    maxlen = df.maxdeg + 1
    cache = _translation_cache(session_cache, pi)
    key = _translation_derived_map_key(:pushforward_right_complex_data, f, res_dom, res_cod, df.maxdeg)
    cached = _translation_cache_get(cache, key)
    cached !== nothing && return cached

    plan = _translation_plan(pi; session_cache=session_cache)
    same_res = _CHANGE_OF_POSETS_USE_RIGHT_DERIVED_SHARED_RESOLUTION_FASTPATH[] && (res_dom === res_cod)
    terms_dom, data_dom = _pushforward_right_resolution_terms_data(
        pi, res_dom.Emods;
        threads=threads,
        session_cache=session_cache,
        plan=plan,
    )
    if same_res
        terms_cod = terms_dom
        data_cod = data_dom
    else
        terms_cod, data_cod = _pushforward_right_resolution_terms_data(
            pi, res_cod.Emods;
            threads=threads,
            session_cache=session_cache,
            plan=plan,
        )
    end

    if _CHANGE_OF_POSETS_USE_RIGHT_DERIVED_DIFFS_FROM_DATA[]
        diffs_dom = _pushforward_right_resolution_diffs_from_data(
            res_dom.Emods, res_dom.d_mor, terms_dom, data_dom, plan.right_fibers; threads=threads,
        )
        diffs_cod = same_res ? diffs_dom : _pushforward_right_resolution_diffs_from_data(
            res_cod.Emods, res_cod.d_mor, terms_cod, data_cod, plan.right_fibers; threads=threads,
        )
    else
        diffs_dom = Vector{PMorphism{K}}(undef, maxlen)
        for k in 1:maxlen
            diffs_dom[k] = pushforward_right(pi, res_dom.d_mor[k];
                                             check=false,
                                             threads=threads,
                                             session_cache=session_cache)
        end
        if same_res
            diffs_cod = diffs_dom
        else
            diffs_cod = Vector{PMorphism{K}}(undef, maxlen)
            for k in 1:maxlen
                diffs_cod[k] = pushforward_right(pi, res_cod.d_mor[k];
                                                 check=false,
                                                 threads=threads,
                                                 session_cache=session_cache)
            end
        end
    end

    Cdom = ModuleCochainComplex(terms_dom, diffs_dom; tmin=0, check=false)
    Ccod = same_res ? Cdom : ModuleCochainComplex(terms_cod, diffs_cod; tmin=0, check=false)

    if same_res && _CHANGE_OF_POSETS_USE_RIGHT_DERIVED_IDENTITY_MAP_FASTPATH[] &&
       _is_identity_pmodule_morphism(f)
        comps = Vector{PMorphism{K}}(undef, maxlen + 1)
        @inbounds for k in 1:(maxlen + 1)
            comps[k] = id_morphism(terms_dom[k])
        end
    elseif _CHANGE_OF_POSETS_USE_DERIVED_RIGHT_COEFF_FASTPATH[]
        comps = Vector{PMorphism{K}}(undef, maxlen + 1)
        coeffs = _lift_injective_chainmap_coeffs(f, res_dom, res_cod; upto=maxlen, check=false)
        coeff_plans = Vector{_RightCoeffPlan}(undef, maxlen + 1)
        @inbounds for k in 0:maxlen
            coeff_plans[k + 1] = _right_pushforward_coeff_plan(
                pi, res_dom.gens[k + 1], res_cod.gens[k + 1], plan;
                session_cache=session_cache,
            )
        end
        @inbounds for k in 0:maxlen
            comps[k + 1] = _pushforward_right_from_downset_coeff(
                pi,
                terms_dom[k + 1],
                terms_cod[k + 1],
                data_dom[k + 1],
                data_cod[k + 1],
                res_dom.gens[k + 1],
                res_cod.gens[k + 1],
                coeffs[k + 1],
                plan;
                coeff_plan=coeff_plans[k + 1],
                session_cache=session_cache,
            )
        end
    else
        comps = Vector{PMorphism{K}}(undef, maxlen + 1)
        phi = lift_injective_chainmap(f, res_dom, res_cod; upto=maxlen, check=false)
        @inbounds for k in 0:maxlen
            comps[k + 1] = _pushforward_right_morphism_from_data(
                terms_dom[k + 1],
                terms_cod[k + 1],
                phi[k + 1],
                data_dom[k + 1],
                data_cod[k + 1],
                plan.right_fibers;
                threads=threads,
            )
        end
    end

    return _translation_cache_set!(cache, key, (Cdom, Ccod, comps))
end

# -----------------------------------------------------------------------------
# Derived functors (object-level)
# -----------------------------------------------------------------------------

"""
    pushforward_left_complex(pi, M, df; check=true, res=nothing)

Compute a cochain complex whose cohomology in degree -i is
`L_i pushforward_left(pi, M)` for i = 0..df.maxdeg.

Derived degree is controlled by `df.maxdeg`.

Speed: pass a precomputed `ProjectiveResolution` via `res` to avoid recomputation.
"""
function pushforward_left_complex(pi::EncodingMap, M::PModule{K}, df::DerivedFunctorOptions;
                                  check::Bool=true,
                                  res=nothing,
                                  threads::Bool = (Threads.nthreads() > 1),
                                  session_cache::Union{Nothing,SessionCache}=nothing) where {K}
    if check
        _check_monotone(pi)
        if !_same_poset(pi.Q, M.Q)
            error("pushforward_left_complex: M must be on pi.Q")
        end
    end

    maxlen = df.maxdeg + 1
    if res === nothing
        res = projective_resolution(M, ResolutionOptions(maxlen=maxlen); threads=threads)
    else
        @assert res.M === M
        @assert length(res.Pmods) >= maxlen + 1
        @assert length(res.d_mor) >= maxlen
    end

    cache = _translation_cache(session_cache, pi)
    key = _translation_complex_key(:pushforward_left_complex_module, M, res, df.maxdeg)
    cached = _translation_cache_get(cache, key)
    cached !== nothing && return cached

    plan = _translation_plan(pi; session_cache=session_cache)
    terms, _ = _pushforward_left_resolution_terms_data(
        pi, res.Pmods;
        threads=threads,
        session_cache=session_cache,
        plan=plan,
    )
    diffs = Vector{PMorphism{K}}(undef, maxlen)

    for k in 1:maxlen
        diffs[maxlen - k + 1] = pushforward_left(pi, res.d_mor[k];
                                                 check=false,
                                                 threads=threads,
                                                 session_cache=session_cache)
    end

    return _translation_cache_set!(cache, key,
                                   ModuleCochainComplex(terms, diffs; tmin=-maxlen, check=check))
end

pushforward_left_complex(pi::EncodingMap, M::PModule{K};
                         opts::DerivedFunctorOptions=DerivedFunctorOptions(),
                         check::Bool=true,
                         res=nothing,
                         threads::Bool = (Threads.nthreads() > 1),
                         session_cache::Union{Nothing,SessionCache}=nothing) where {K} =
    pushforward_left_complex(pi, M, opts;
                             check=check, res=res, threads=threads, session_cache=session_cache)

"""
    Lpushforward_left(pi, M, df; check=true)

Return `[L_0, L_1, ..., L_df.maxdeg]` where `L_i = L_i pushforward_left(pi, M)`.

Derived degree is controlled by `df.maxdeg`.
"""
function Lpushforward_left(pi::EncodingMap, M::PModule{K}, df::DerivedFunctorOptions;
                           check::Bool=true,
                           threads::Bool = (Threads.nthreads() > 1),
                           session_cache::Union{Nothing,SessionCache}=nothing) where {K}
    maxdeg = df.maxdeg
    cache = _translation_cache(session_cache, pi)
    key = _translation_complex_key(:Lpushforward_left_module, M, pi, maxdeg)
    cached = _translation_cache_get(cache, key)
    cached !== nothing && return cached
    C = pushforward_left_complex(pi, M, df;
                                 check=check,
                                 threads=threads,
                                 session_cache=session_cache)
    out = Vector{PModule{K}}(undef, maxdeg + 1)
    for i in 0:maxdeg
        out[i + 1] = cohomology_module(C, -i)
    end
    return _translation_cache_set!(cache, key, out)
end

Lpushforward_left(pi::EncodingMap, M::PModule{K};
                  opts::DerivedFunctorOptions=DerivedFunctorOptions(),
                  check::Bool=true,
                  threads::Bool = (Threads.nthreads() > 1),
                  session_cache::Union{Nothing,SessionCache}=nothing) where {K} =
    Lpushforward_left(pi, M, opts;
                      check=check, threads=threads, session_cache=session_cache)

"""
    derived_pushforward_left(pi::EncodingMap, M::PModule, opts=DerivedFunctorOptions();
                             check=true, session_cache=nothing) -> Vector

Canonical notebook-facing entrypoint for the left-derived pushforward of a
module along `pi : Q -> P`.

This returns the sequence `[L_0, L_1, ..., L_n]` up to `opts.maxdeg`, with each
term a module on `pi.P`.

Notation policy
- prefer `derived_pushforward_left` in user-facing code,
- keep `Lpushforward_left` as the internal categorical/derived spelling.

Cache semantics
- with `session_cache=sc::SessionCache`, this reuses translation plans,
  translated resolutions, and induced cohomology computations across calls.

Cheap-first workflow
- inspect `first(derived_pushforward_left(pi, M; opts=DerivedFunctorOptions(maxdeg=0)))`
  or `map(describe, derived_pushforward_left(pi, M; opts=DerivedFunctorOptions(maxdeg=0)))`
  before asking for higher derived degrees.
"""
derived_pushforward_left(pi::EncodingMap, M::PModule{K}, df::DerivedFunctorOptions;
                         check::Bool=true,
                         threads::Bool = (Threads.nthreads() > 1),
                         session_cache::Union{Nothing,SessionCache}=nothing) where {K} =
    Lpushforward_left(pi, M, df; check=check, threads=threads, session_cache=session_cache)

derived_pushforward_left(pi::EncodingMap, M::PModule{K};
                         opts::DerivedFunctorOptions=DerivedFunctorOptions(),
                         check::Bool=true,
                         threads::Bool = (Threads.nthreads() > 1),
                         session_cache::Union{Nothing,SessionCache}=nothing) where {K} =
    derived_pushforward_left(pi, M, opts;
                             check=check, threads=threads, session_cache=session_cache)

"""
    pushforward_left_complex(pi, f, df; check=true, res_dom=nothing, res_cod=nothing)

Return a cochain map between the complexes `pushforward_left_complex(pi, f.dom, df)` and
`pushforward_left_complex(pi, f.cod, df)`.

The induced map on cohomology in degree `-i` is the map on left-derived functors
`L_i pushforward_left(pi, f)` for i = 0..df.maxdeg.

Implementation:
1. take projective resolutions of dom and cod
2. lift `f` canonically to a chain map between projective resolutions (coefficient form)
3. apply `pushforward_left` termwise
4. package as a `ModuleCochainMap`
"""
function pushforward_left_complex(pi::EncodingMap, f::PMorphism{K}, df::DerivedFunctorOptions;
                                  check::Bool=true,
                                  res_dom=nothing,
                                  res_cod=nothing,
                                  threads::Bool = (Threads.nthreads() > 1),
                                  session_cache::Union{Nothing,SessionCache}=nothing) where {K}
    if check
        _check_monotone(pi)
        @assert _same_poset(pi.Q, f.dom.Q)
        @assert _same_poset(pi.Q, f.cod.Q)
    end

    maxlen = df.maxdeg + 1

    if res_dom === nothing
        res_dom = projective_resolution(f.dom, ResolutionOptions(maxlen=maxlen); threads=threads)
    end
    if res_cod === nothing
        res_cod = projective_resolution(f.cod, ResolutionOptions(maxlen=maxlen); threads=threads)
    end

    cache = _translation_cache(session_cache, pi)
    key = _translation_derived_map_key(:pushforward_left_complex_morphism, f, res_dom, res_cod, df.maxdeg)
    cached = _translation_cache_get(cache, key)
    cached !== nothing && return cached

    Cdom, Ccod, comps = _pushforward_left_complex_data(
        pi, f, df;
        res_dom=res_dom,
        res_cod=res_cod,
        threads=threads,
        session_cache=session_cache,
    )

    return _translation_cache_set!(cache, key,
                                   ModuleCochainMap(Cdom, Ccod, comps; check=check))
end

pushforward_left_complex(pi::EncodingMap, f::PMorphism{K};
                         opts::DerivedFunctorOptions=DerivedFunctorOptions(),
                         check::Bool=true,
                         res_dom=nothing,
                         res_cod=nothing,
                         threads::Bool = (Threads.nthreads() > 1),
                         session_cache::Union{Nothing,SessionCache}=nothing) where {K} =
    pushforward_left_complex(pi, f, opts;
                             check=check, res_dom=res_dom, res_cod=res_cod,
                             threads=threads, session_cache=session_cache)

function _induced_maps_on_cohomology_modules_from_data(
    Cdom::ModuleCochainComplex{K},
    Ccod::ModuleCochainComplex{K},
    comps::Vector{PMorphism{K}},
    maxdeg::Int,
) where {K}
    field = Cdom.terms[1].field
    out = Vector{PMorphism{K}}(undef, maxdeg + 1)
    @inbounds for i in 0:maxdeg
        t = -i
        Cd = ModuleComplexes.cohomology_module_data(Cdom, t)
        Dd = ModuleComplexes.cohomology_module_data(Ccod, t)
        ft = comps[t - Cdom.tmin + 1]
        Q = Cdom.terms[1].Q
        compsH = Vector{Matrix{K}}(undef, nvertices(Q))
        for u in 1:nvertices(Q)
            rhsZ = ft.comps[u] * Cd.iZ.comps[u]
            if Dd.Z.dims[u] == 0
                compsH[u] = zeros(K, Dd.H.dims[u], Cd.H.dims[u])
            else
                z_to_z = FieldLinAlg.solve_fullcolumn(field, Dd.iZ.comps[u], rhsZ)
                rhsH = Dd.q.comps[u] * z_to_z
                qC = Cd.q.comps[u]
                if size(qC, 1) == 0
                    compsH[u] = zeros(K, size(rhsH, 1), 0)
                else
                    rinv = _right_inverse_full_row(field, qC)
                    compsH[u] = rhsH * rinv
                end
            end
        end
        out[i + 1] = PMorphism{K}(Cd.H, Dd.H, compsH)
    end
    return out
end

"""
    Lpushforward_left(pi, f, df; check=true, res_dom=nothing, res_cod=nothing)

Return the induced maps on left-derived pushforward:

    out[i+1] : L_i pushforward_left(pi, f.dom) -> L_i pushforward_left(pi, f.cod)

for i = 0..df.maxdeg.

Derived degree is controlled by `df.maxdeg`.
"""
function Lpushforward_left(pi::EncodingMap, f::PMorphism{K}, df::DerivedFunctorOptions;
                           check::Bool=true,
                           res_dom=nothing,
                           res_cod=nothing,
                           threads::Bool = (Threads.nthreads() > 1),
                           session_cache::Union{Nothing,SessionCache}=nothing) where {K}
    maxdeg = df.maxdeg
    if res_dom === nothing
        res_dom = projective_resolution(f.dom, ResolutionOptions(maxlen=maxdeg + 1); threads=threads)
    end
    if res_cod === nothing
        res_cod = projective_resolution(f.cod, ResolutionOptions(maxlen=maxdeg + 1); threads=threads)
    end
    cache = _translation_cache(session_cache, pi)
    key = _translation_derived_map_key(:Lpushforward_left_morphism, f, res_dom, res_cod, maxdeg)
    cached = _translation_cache_get(cache, key)
    cached !== nothing && return cached
    if _CHANGE_OF_POSETS_USE_LEFT_DERIVED_COHOMOLOGY_SHORTCUT[]
        Cdom, Ccod, comps = _pushforward_left_complex_data(
            pi, f, df;
            res_dom=res_dom,
            res_cod=res_cod,
            threads=threads,
            session_cache=session_cache,
        )
        out = _induced_maps_on_cohomology_modules_from_data(Cdom, Ccod, comps, maxdeg)
    else
        F = pushforward_left_complex(pi, f, df;
                                     check=check, res_dom=res_dom, res_cod=res_cod,
                                     threads=threads, session_cache=session_cache)
        out = Vector{PMorphism{K}}(undef, maxdeg + 1)
        for i in 0:maxdeg
            out[i + 1] = induced_map_on_cohomology_modules(F, -i)
        end
    end
    return _translation_cache_set!(cache, key, out)
end

Lpushforward_left(pi::EncodingMap, f::PMorphism{K};
                  opts::DerivedFunctorOptions=DerivedFunctorOptions(),
                  check::Bool=true,
                  res_dom=nothing,
                  res_cod=nothing,
                  threads::Bool = (Threads.nthreads() > 1),
                  session_cache::Union{Nothing,SessionCache}=nothing) where {K} =
    Lpushforward_left(pi, f, opts;
                      check=check, res_dom=res_dom, res_cod=res_cod,
                      threads=threads, session_cache=session_cache)

"""
    derived_pushforward_left(pi::EncodingMap, f::PMorphism, opts=DerivedFunctorOptions();
                             check=true, session_cache=nothing) -> Vector

Canonical notebook-facing entrypoint for the induced maps on left-derived
pushforward along `pi : Q -> P`.

The returned vector stores maps

`L_i pushforward_left(pi, f.dom) -> L_i pushforward_left(pi, f.cod)`

for `i = 0, ..., opts.maxdeg`.

Source/target convention
- `f` must be a morphism on `pi.Q`,
- each returned map lives on `pi.P`.

Cache semantics
- with `session_cache=sc::SessionCache`, this reuses translated resolutions,
  lifted chain maps, and induced cohomology computations across calls.

Cheap-first workflow
- start with `opts=DerivedFunctorOptions(maxdeg=0)` unless higher derived maps
  are explicitly needed.
"""
derived_pushforward_left(pi::EncodingMap, f::PMorphism{K}, df::DerivedFunctorOptions;
                         check::Bool=true,
                         threads::Bool = (Threads.nthreads() > 1),
                         session_cache::Union{Nothing,SessionCache}=nothing) where {K} =
    Lpushforward_left(pi, f, df; check=check, threads=threads, session_cache=session_cache)

derived_pushforward_left(pi::EncodingMap, f::PMorphism{K};
                         opts::DerivedFunctorOptions=DerivedFunctorOptions(),
                         check::Bool=true,
                         threads::Bool = (Threads.nthreads() > 1),
                         session_cache::Union{Nothing,SessionCache}=nothing) where {K} =
    derived_pushforward_left(pi, f, opts;
                             check=check, threads=threads, session_cache=session_cache)


"""
        pushforward_right_complex(pi, M, df; check=true, res=nothing)

Compute a cochain complex whose cohomology in degree i is
`R^i pushforward_right(pi, M)` for i = 0..df.maxdeg.

Derived degree is controlled by `df.maxdeg`

Speed: pass a precomputed `InjectiveResolution` via `res` to avoid recomputation.
"""
function pushforward_right_complex(pi::EncodingMap, M::PModule{K}, df::DerivedFunctorOptions;
                                   check::Bool=true,
                                   res=nothing,
                                   threads::Bool = (Threads.nthreads() > 1),
                                   session_cache::Union{Nothing,SessionCache}=nothing) where {K}
    if check
        _check_monotone(pi)
        if !_same_poset(pi.Q, M.Q)
            error("pushforward_right_complex: M must be on pi.Q")
        end
    end

    maxlen = df.maxdeg + 1
    if res === nothing
        res = injective_resolution(M, ResolutionOptions(maxlen=maxlen); threads=threads)
    else
        @assert res.N === M
        @assert length(res.Emods) >= maxlen + 1
        @assert length(res.d_mor) >= maxlen
    end

    cache = _translation_cache(session_cache, pi)
    key = _translation_complex_key(:pushforward_right_complex_module, M, res, df.maxdeg)
    cached = _translation_cache_get(cache, key)
    cached !== nothing && return cached

    plan = _translation_plan(pi; session_cache=session_cache)
    terms, _ = _pushforward_right_resolution_terms_data(
        pi, res.Emods;
        threads=threads,
        session_cache=session_cache,
        plan=plan,
    )
    diffs = Vector{PMorphism{K}}(undef, maxlen)

    for k in 1:maxlen
        diffs[k] = pushforward_right(pi, res.d_mor[k];
                                     check=false,
                                     threads=threads,
                                     session_cache=session_cache)
    end

    return _translation_cache_set!(cache, key,
                                   ModuleCochainComplex(terms, diffs; tmin=0, check=check))
end

pushforward_right_complex(pi::EncodingMap, M::PModule{K};
                          opts::DerivedFunctorOptions=DerivedFunctorOptions(),
                          check::Bool=true,
                          res=nothing,
                          threads::Bool = (Threads.nthreads() > 1),
                          session_cache::Union{Nothing,SessionCache}=nothing) where {K} =
    pushforward_right_complex(pi, M, opts;
                              check=check, res=res, threads=threads, session_cache=session_cache)

"""
    Rpushforward_right(pi, M, df; check=true)

Return `[R^0, R^1, ..., R^df.maxdeg]` where `R^i = R^i pushforward_right(pi, M)`.

Derived degree is controlled by `df.maxdeg`.
"""
function Rpushforward_right(pi::EncodingMap, M::PModule{K}, df::DerivedFunctorOptions;
                            check::Bool=true,
                            threads::Bool = (Threads.nthreads() > 1),
                            session_cache::Union{Nothing,SessionCache}=nothing) where {K}
    maxdeg = df.maxdeg
    cache = _translation_cache(session_cache, pi)
    key = _translation_complex_key(:Rpushforward_right_module, M, pi, maxdeg)
    cached = _translation_cache_get(cache, key)
    cached !== nothing && return cached
    C = pushforward_right_complex(pi, M, df;
                                  check=check,
                                  threads=threads,
                                  session_cache=session_cache)
    out = Vector{PModule{K}}(undef, maxdeg + 1)
    for i in 0:maxdeg
        out[i + 1] = cohomology_module(C, i)
    end
    return _translation_cache_set!(cache, key, out)
end

Rpushforward_right(pi::EncodingMap, M::PModule{K};
                   opts::DerivedFunctorOptions=DerivedFunctorOptions(),
                   check::Bool=true,
                   threads::Bool = (Threads.nthreads() > 1),
                   session_cache::Union{Nothing,SessionCache}=nothing) where {K} =
    Rpushforward_right(pi, M, opts;
                       check=check, threads=threads, session_cache=session_cache)

"""
    derived_pushforward_right(pi::EncodingMap, M::PModule, opts=DerivedFunctorOptions();
                              check=true, session_cache=nothing) -> Vector

Canonical notebook-facing entrypoint for the right-derived pushforward of a
module along `pi : Q -> P`.

This returns the sequence `[R^0, R^1, ..., R^n]` up to `opts.maxdeg`, with each
term a module on `pi.P`.

Notation policy
- prefer `derived_pushforward_right` in user-facing code,
- keep `Rpushforward_right` as the internal categorical/derived spelling.

Cache semantics
- with `session_cache=sc::SessionCache`, this reuses translation plans,
  translated injective resolutions, and induced cohomology computations across
  calls.

Cheap-first workflow
- inspect `first(derived_pushforward_right(pi, M; opts=DerivedFunctorOptions(maxdeg=0)))`
  or `map(describe, derived_pushforward_right(pi, M; opts=DerivedFunctorOptions(maxdeg=0)))`
  before asking for higher derived degrees.
"""
derived_pushforward_right(pi::EncodingMap, M::PModule{K}, df::DerivedFunctorOptions;
                          check::Bool=true,
                          threads::Bool = (Threads.nthreads() > 1),
                          session_cache::Union{Nothing,SessionCache}=nothing) where {K} =
    Rpushforward_right(pi, M, df; check=check, threads=threads, session_cache=session_cache)

derived_pushforward_right(pi::EncodingMap, M::PModule{K};
                          opts::DerivedFunctorOptions=DerivedFunctorOptions(),
                          check::Bool=true,
                          threads::Bool = (Threads.nthreads() > 1),
                          session_cache::Union{Nothing,SessionCache}=nothing) where {K} =
    derived_pushforward_right(pi, M, opts;
                              check=check, threads=threads, session_cache=session_cache)
    
"""
    pushforward_right_complex(pi, f, df; check=true, res_dom=nothing, res_cod=nothing)

Return the induced cochain map between `pushforward_right_complex(pi, f.dom, df)` and
`pushforward_right_complex(pi, f.cod, df)`.

Cohomology in degree i gives the induced map:

    R^i pushforward_right(pi, f)
    
for i = 0..df.maxdeg.

Implementation:
1. take injective resolutions of dom and cod
2. canonically lift `f` to a chain map between injective resolutions
3. apply `pushforward_right` termwise
4. package as a `ModuleCochainMap`
"""
function pushforward_right_complex(pi::EncodingMap, f::PMorphism{K}, df::DerivedFunctorOptions;
                                   check::Bool=true,
                                   res_dom=nothing,
                                   res_cod=nothing,
                                   threads::Bool = (Threads.nthreads() > 1),
                                   session_cache::Union{Nothing,SessionCache}=nothing) where {K}
    if check
        _check_monotone(pi)
        @assert _same_poset(pi.Q, f.dom.Q)
        @assert _same_poset(pi.Q, f.cod.Q)
    end

    maxlen = df.maxdeg + 1

    if res_dom === nothing
        res_dom = injective_resolution(f.dom, ResolutionOptions(maxlen=maxlen); threads=threads)
    end
    if res_cod === nothing
        res_cod = injective_resolution(f.cod, ResolutionOptions(maxlen=maxlen); threads=threads)
    end

    cache = _translation_cache(session_cache, pi)
    key = _translation_derived_map_key(:pushforward_right_complex_morphism, f, res_dom, res_cod, df.maxdeg)
    cached = _translation_cache_get(cache, key)
    cached !== nothing && return cached
    identity_same_res = _CHANGE_OF_POSETS_USE_RIGHT_DERIVED_SHARED_RESOLUTION_FASTPATH[] &&
                        _CHANGE_OF_POSETS_USE_RIGHT_DERIVED_IDENTITY_MAP_FASTPATH[] &&
                        (res_dom === res_cod) &&
                        _is_identity_pmodule_morphism(f)
    Cdom, Ccod, comps = _pushforward_right_complex_data(
        pi, f, df;
        res_dom=res_dom,
        res_cod=res_cod,
        threads=threads,
        session_cache=session_cache,
    )
    return _translation_cache_set!(cache, key,
                                   ModuleCochainMap(Cdom, Ccod, comps; check=(check && !identity_same_res)))
end

pushforward_right_complex(pi::EncodingMap, f::PMorphism{K};
                          opts::DerivedFunctorOptions=DerivedFunctorOptions(),
                          check::Bool=true,
                          res_dom=nothing,
                          res_cod=nothing,
                          threads::Bool = (Threads.nthreads() > 1),
                          session_cache::Union{Nothing,SessionCache}=nothing) where {K} =
    pushforward_right_complex(pi, f, opts;
                              check=check, res_dom=res_dom, res_cod=res_cod,
                              threads=threads, session_cache=session_cache)


"""
    Rpushforward_right(pi, f, df; check=true, res_dom=nothing, res_cod=nothing)

Return the induced maps on right-derived pushforward:

    out[i+1] : R^i pushforward_right(pi, f.dom) -> R^i pushforward_right(pi, f.cod)

for i = 0..df.maxdeg.

Derived degree is controlled by `df.maxdeg`.
"""
function Rpushforward_right(pi::EncodingMap, f::PMorphism{K}, df::DerivedFunctorOptions;
                            check::Bool=true,
                            res_dom=nothing,
                            res_cod=nothing,
                            threads::Bool = (Threads.nthreads() > 1),
                            session_cache::Union{Nothing,SessionCache}=nothing) where {K}
    maxdeg = df.maxdeg
    if res_dom === nothing
        res_dom = injective_resolution(f.dom, ResolutionOptions(maxlen=maxdeg + 1); threads=threads)
    end
    if res_cod === nothing
        res_cod = injective_resolution(f.cod, ResolutionOptions(maxlen=maxdeg + 1); threads=threads)
    end
    cache = _translation_cache(session_cache, pi)
    key = _translation_derived_map_key(:Rpushforward_right_morphism, f, res_dom, res_cod, maxdeg)
    cached = _translation_cache_get(cache, key)
    cached !== nothing && return cached
    F = pushforward_right_complex(pi, f, df;
                                  check=check, res_dom=res_dom, res_cod=res_cod,
                                  threads=threads, session_cache=session_cache)
    out = Vector{PMorphism{K}}(undef, maxdeg + 1)
    for i in 0:maxdeg
        out[i + 1] = induced_map_on_cohomology_modules(F, i)
    end
    return _translation_cache_set!(cache, key, out)
end

Rpushforward_right(pi::EncodingMap, f::PMorphism{K};
                   opts::DerivedFunctorOptions=DerivedFunctorOptions(),
                   check::Bool=true,
                   res_dom=nothing,
                   res_cod=nothing,
                   threads::Bool = (Threads.nthreads() > 1),
                   session_cache::Union{Nothing,SessionCache}=nothing) where {K} =
    Rpushforward_right(pi, f, opts;
                       check=check, res_dom=res_dom, res_cod=res_cod,
                       threads=threads, session_cache=session_cache)

"""
    derived_pushforward_right(pi::EncodingMap, f::PMorphism, opts=DerivedFunctorOptions();
                              check=true, session_cache=nothing) -> Vector

Canonical notebook-facing entrypoint for the induced maps on right-derived
pushforward along `pi : Q -> P`.

The returned vector stores maps

`R^i pushforward_right(pi, f.dom) -> R^i pushforward_right(pi, f.cod)`

for `i = 0, ..., opts.maxdeg`.

Source/target convention
- `f` must be a morphism on `pi.Q`,
- each returned map lives on `pi.P`.

Cache semantics
- with `session_cache=sc::SessionCache`, this reuses translated injective
  resolutions, lifted chain maps, and induced cohomology computations across
  calls.

Cheap-first workflow
- start with `opts=DerivedFunctorOptions(maxdeg=0)` unless higher derived maps
  are explicitly needed.
"""
derived_pushforward_right(pi::EncodingMap, f::PMorphism{K}, df::DerivedFunctorOptions;
                          check::Bool=true,
                          threads::Bool = (Threads.nthreads() > 1),
                          session_cache::Union{Nothing,SessionCache}=nothing) where {K} =
    Rpushforward_right(pi, f, df; check=check, threads=threads, session_cache=session_cache)

derived_pushforward_right(pi::EncodingMap, f::PMorphism{K};
                          opts::DerivedFunctorOptions=DerivedFunctorOptions(),
                          check::Bool=true,
                          threads::Bool = (Threads.nthreads() > 1),
                          session_cache::Union{Nothing,SessionCache}=nothing) where {K} =
    derived_pushforward_right(pi, f, opts;
                              check=check, threads=threads, session_cache=session_cache)

# -----------------------------------------------------------------------------
# Helpers: morphisms between direct sums of principal upsets (projectives)
# -----------------------------------------------------------------------------

# Active summand indices at each vertex u, respecting the canonical ordering used in projective covers.
function _active_upset_indices_from_bases(Q::AbstractPoset, bases::Vector{Int})
    n = nvertices(Q)
    by_base = [Int[] for _ in 1:n]
    for (j, b) in enumerate(bases)
        push!(by_base[b], j)
    end

    out = Vector{Vector{Int}}(undef, n)
    for u in 1:n
        idxs = Int[]
        for b in downset_indices(Q, u)
            append!(idxs, by_base[b])
        end
        out[u] = idxs
    end
    return out
end

function _active_downset_indices_from_bases(Q::AbstractPoset, bases::Vector{Int})
    n = nvertices(Q)
    by_base = [Int[] for _ in 1:n]
    for (j, b) in enumerate(bases)
        push!(by_base[b], j)
    end

    out = Vector{Vector{Int}}(undef, n)
    for u in 1:n
        idxs = Int[]
        for b in upset_indices(Q, u)
            append!(idxs, by_base[b])
        end
        out[u] = idxs
    end
    return out
end

struct _LeftCoeffPlan
    act_dom::Vector{Vector{Int}}
    act_cod::Vector{Vector{Int}}
    ptr::Vector{Int}
    block_q::Vector{Int}
    block_loc::Vector{Int}
    max_rows::Vector{Int}
    max_cols::Vector{Int}
end

struct _RightCoeffPlan
    act_dom::Vector{Vector{Int}}
    act_cod::Vector{Vector{Int}}
    ptr::Vector{Int}
    block_q::Vector{Int}
    block_loc::Vector{Int}
    max_rows::Vector{Int}
    max_cols::Vector{Int}
end

function _left_pushforward_coeff_plan(
    pi::EncodingMap,
    dom_bases::Vector{Int},
    cod_bases::Vector{Int},
    plan::_PiTranslationPlan;
    session_cache::Union{Nothing,SessionCache}=nothing,
)
    cache = _translation_cache(session_cache, pi)
    if _CHANGE_OF_POSETS_USE_LEFT_DERIVED_COEFF_PLAN_CACHE[]
        key = _translation_left_coeff_key(pi, dom_bases, cod_bases)
        cached = _translation_cache_get(cache, key)
        cached !== nothing && return cached
    end

    Q = pi.Q
    nP = nvertices(pi.P)
    act_dom = _active_upset_indices_from_bases(Q, dom_bases)
    act_cod = _active_upset_indices_from_bases(Q, cod_bases)

    counts = zeros(Int, nP)
    max_rows = zeros(Int, nP)
    max_cols = zeros(Int, nP)
    total = 0
    @inbounds for p in 1:nP
        fiber = plan.left_fibers[p]
        count = 0
        mr = 0
        mc = 0
        for loc in eachindex(fiber.idxs)
            q = fiber.idxs[loc]
            nr = length(act_cod[q])
            nc = length(act_dom[q])
            (nr == 0 || nc == 0) && continue
            count += 1
            mr = max(mr, nr)
            mc = max(mc, nc)
        end
        counts[p] = count
        max_rows[p] = mr
        max_cols[p] = mc
        total += count
    end

    ptr = Vector{Int}(undef, nP + 1)
    ptr[1] = 1
    @inbounds for p in 1:nP
        ptr[p + 1] = ptr[p] + counts[p]
    end
    block_q = Vector{Int}(undef, total)
    block_loc = Vector{Int}(undef, total)
    next = copy(ptr)
    @inbounds for p in 1:nP
        fiber = plan.left_fibers[p]
        for loc in eachindex(fiber.idxs)
            q = fiber.idxs[loc]
            isempty(act_cod[q]) && continue
            isempty(act_dom[q]) && continue
            pos = next[p]
            block_q[pos] = q
            block_loc[pos] = loc
            next[p] = pos + 1
        end
    end

    coeff_plan = _LeftCoeffPlan(act_dom, act_cod, ptr, block_q, block_loc, max_rows, max_cols)
    if _CHANGE_OF_POSETS_USE_LEFT_DERIVED_COEFF_PLAN_CACHE[]
        key = _translation_left_coeff_key(pi, dom_bases, cod_bases)
        return _translation_cache_set!(cache, key, coeff_plan)
    end
    return coeff_plan
end

function _right_pushforward_coeff_plan(
    pi::EncodingMap,
    dom_bases::Vector{Int},
    cod_bases::Vector{Int},
    plan::_PiTranslationPlan;
    session_cache::Union{Nothing,SessionCache}=nothing,
)
    cache = _translation_cache(session_cache, pi)
    if _CHANGE_OF_POSETS_USE_RIGHT_DERIVED_COEFF_PLAN_CACHE[]
        key = _translation_right_coeff_key(pi, dom_bases, cod_bases)
        cached = _translation_cache_get(cache, key)
        cached !== nothing && return cached
    end

    Q = pi.Q
    nP = nvertices(pi.P)
    act_dom = _active_downset_indices_from_bases(Q, dom_bases)
    act_cod = _active_downset_indices_from_bases(Q, cod_bases)

    counts = zeros(Int, nP)
    max_rows = zeros(Int, nP)
    max_cols = zeros(Int, nP)
    total = 0
    @inbounds for p in 1:nP
        fiber = plan.right_fibers[p]
        count = 0
        mr = 0
        mc = 0
        for loc in eachindex(fiber.idxs)
            q = fiber.idxs[loc]
            nr = length(act_cod[q])
            nc = length(act_dom[q])
            (nr == 0 || nc == 0) && continue
            count += 1
            mr = max(mr, nr)
            mc = max(mc, nc)
        end
        counts[p] = count
        max_rows[p] = mr
        max_cols[p] = mc
        total += count
    end

    ptr = Vector{Int}(undef, nP + 1)
    ptr[1] = 1
    @inbounds for p in 1:nP
        ptr[p + 1] = ptr[p] + counts[p]
    end
    block_q = Vector{Int}(undef, total)
    block_loc = Vector{Int}(undef, total)
    next = copy(ptr)
    @inbounds for p in 1:nP
        fiber = plan.right_fibers[p]
        for loc in eachindex(fiber.idxs)
            q = fiber.idxs[loc]
            isempty(act_cod[q]) && continue
            isempty(act_dom[q]) && continue
            pos = next[p]
            block_q[pos] = q
            block_loc[pos] = loc
            next[p] = pos + 1
        end
    end

    coeff_plan = _RightCoeffPlan(act_dom, act_cod, ptr, block_q, block_loc, max_rows, max_cols)
    if _CHANGE_OF_POSETS_USE_RIGHT_DERIVED_COEFF_PLAN_CACHE[]
        key = _translation_right_coeff_key(pi, dom_bases, cod_bases)
        return _translation_cache_set!(cache, key, coeff_plan)
    end
    return coeff_plan
end

@inline function _gather_component_matrix!(
    buf::AbstractMatrix{K},
    C::AbstractMatrix{K},
    rows::Vector{Int},
    cols::Vector{Int},
) where {K}
    nr = length(rows)
    nc = length(cols)
    @inbounds for j in 1:nc
        cj = cols[j]
        for i in 1:nr
            buf[i, j] = C[rows[i], cj]
        end
    end
    return nothing
end

@inline function _gather_component_matrix(
    ::Type{K},
    C::AbstractMatrix{K},
    rows::Vector{Int},
    cols::Vector{Int},
) where {K}
    nr = length(rows)
    nc = length(cols)
    M = Matrix{K}(undef, nr, nc)
    @inbounds for j in 1:nc
        cj = cols[j]
        for i in 1:nr
            M[i, j] = C[rows[i], cj]
        end
    end
    return M
end

function _pushforward_left_from_upset_coeff(
    pi::EncodingMap,
    dom_out::PModule{K},
    cod_out::PModule{K},
    data_dom::LeftKanData{K},
    data_cod::LeftKanData{K},
    dom_bases::Vector{Int},
    cod_bases::Vector{Int},
    C::AbstractMatrix{K},
    plan::_PiTranslationPlan;
    coeff_plan=nothing,
    session_cache::Union{Nothing,SessionCache}=nothing,
) where {K}
    coeff_plan === nothing && (coeff_plan = _left_pushforward_coeff_plan(
        pi, dom_bases, cod_bases, plan; session_cache=session_cache,
    ))
    act_dom = coeff_plan.act_dom
    act_cod = coeff_plan.act_cod
    comps = Vector{Matrix{K}}(undef, nvertices(pi.P))
    @inbounds for p in 1:nvertices(pi.P)
        Vd = data_dom.dimV[p]
        Vc = data_cod.dimV[p]
        if Vd == 0 || Vc == 0
            comps[p] = zeros(K, Vc, Vd)
            continue
        end
        lo = coeff_plan.ptr[p]
        hi = coeff_plan.ptr[p + 1] - 1
        if hi < lo
            comps[p] = zeros(K, Vc, Vd)
            continue
        end
        offs_dom = data_dom.offsets[p]
        offs_cod = data_cod.offsets[p]
        Wp = data_dom.W[p]
        Lp = data_cod.L[p]
        Fp = zeros(K, Vc, Vd)
        scratch = Matrix{K}(undef, coeff_plan.max_rows[p], coeff_plan.max_cols[p])
        tmp = Matrix{K}(undef, coeff_plan.max_rows[p], Vd)
        for pos in lo:hi
            q = coeff_plan.block_q[pos]
            loc = coeff_plan.block_loc[pos]
            rows = act_cod[q]
            cols = act_dom[q]
            nr = length(rows)
            nc = length(cols)
            rd = (offs_dom[loc] + 1):offs_dom[loc + 1]
            cc = (offs_cod[loc] + 1):offs_cod[loc + 1]
            Cq = @view scratch[1:nr, 1:nc]
            tmpv = @view tmp[1:nr, :]
            _gather_component_matrix!(Cq, C, rows, cols)
            mul!(tmpv, Cq, @view(Wp[rd, :]))
            mul!(Fp, @view(Lp[:, cc]), tmpv, one(K), one(K))
        end
        comps[p] = Fp
    end
    return PMorphism(dom_out, cod_out, comps)
end

function _lift_injective_chainmap_coeffs(
    g::PMorphism{K},
    res_dom,
    res_cod;
    upto::Int,
    check::Bool=true,
) where {K}
    if check
        @assert g.dom === res_dom.N
        @assert g.cod === res_cod.N
    end
    Q = g.dom.Q
    coeffs = Vector{Matrix{K}}(undef, upto + 1)

    dom_bases0 = res_dom.gens[1]
    cod_bases0 = res_cod.gens[1]
    structure0 = DerivedFunctors.Resolutions._cached_downset_hom_structure(Q, dom_bases0, cod_bases0)
    rhs0 = PMorphism{K}(g.dom, res_cod.Emods[1], [res_cod.iota0.comps[u] * g.comps[u] for u in 1:nvertices(Q)])
    coeffs[1] = DerivedFunctors.Resolutions._solve_downset_postcompose_coeff(
        res_dom.iota0, rhs0, dom_bases0, cod_bases0, structure0.act_dom, structure0.act_cod; check=check
    )
    upto == 0 && return coeffs

    if _CHANGE_OF_POSETS_USE_RIGHT_DERIVED_COEFF_MATRIX_RECURRENCE[]
        cod_diff_coeffs = Vector{SparseMatrixCSC{K,Int}}(undef, upto)
        rhs_structures = Vector{typeof(structure0)}(undef, upto)
        @inbounds for k in 1:upto
            cod_diff_coeffs[k] = DerivedFunctors.Resolutions._coeff_matrix_downsets(
                Q, res_cod.gens[k], res_cod.gens[k + 1], res_cod.d_mor[k]
            )
            rhs_structures[k] = DerivedFunctors.Resolutions._cached_downset_hom_structure(
                Q, res_dom.gens[k], res_cod.gens[k + 1]
            )
        end

        @inbounds for k in 1:upto
            dom_bases = res_dom.gens[k + 1]
            cod_bases = res_cod.gens[k + 1]
            structure = DerivedFunctors.Resolutions._cached_downset_hom_structure(Q, dom_bases, cod_bases)
            rhs_structure = rhs_structures[k]
            rhs_coeff = zeros(K, size(cod_diff_coeffs[k], 1), size(coeffs[k], 2))
            mul!(rhs_coeff, cod_diff_coeffs[k], coeffs[k])
            rhs = DerivedFunctors.Resolutions._pmorphism_from_downset_coeff(
                res_dom.Emods[k], res_cod.Emods[k + 1],
                rhs_structure.act_dom, rhs_structure.act_cod, rhs_coeff
            )
            coeffs[k + 1] = DerivedFunctors.Resolutions._solve_downset_postcompose_coeff(
                res_dom.d_mor[k], rhs, dom_bases, cod_bases, structure.act_dom, structure.act_cod; check=check
            )
        end
        return coeffs
    end

    phis = Vector{PMorphism{K}}(undef, upto + 1)
    phis[1] = DerivedFunctors.Resolutions._pmorphism_from_downset_coeff(
        res_dom.Emods[1], res_cod.Emods[1], structure0.act_dom, structure0.act_cod, coeffs[1]
    )

    for k in 1:upto
        dom_bases = res_dom.gens[k + 1]
        cod_bases = res_cod.gens[k + 1]
        structure = DerivedFunctors.Resolutions._cached_downset_hom_structure(Q, dom_bases, cod_bases)
        rhs = PMorphism{K}(res_dom.Emods[k], res_cod.Emods[k + 1],
                           [res_cod.d_mor[k].comps[u] * phis[k].comps[u] for u in 1:nvertices(Q)])
        coeffs[k + 1] = DerivedFunctors.Resolutions._solve_downset_postcompose_coeff(
            res_dom.d_mor[k], rhs, dom_bases, cod_bases, structure.act_dom, structure.act_cod; check=check
        )
        phis[k + 1] = DerivedFunctors.Resolutions._pmorphism_from_downset_coeff(
            res_dom.Emods[k + 1], res_cod.Emods[k + 1], structure.act_dom, structure.act_cod, coeffs[k + 1]
        )
    end
    return coeffs
end

function _pushforward_right_from_downset_coeff(
    pi::EncodingMap,
    dom_out::PModule{K},
    cod_out::PModule{K},
    data_dom::RightKanData{K},
    data_cod::RightKanData{K},
    dom_bases::Vector{Int},
    cod_bases::Vector{Int},
    C::AbstractMatrix{K},
    plan::_PiTranslationPlan,
    ;
    coeff_plan=nothing,
    session_cache::Union{Nothing,SessionCache}=nothing,
) where {K}
    coeff_plan === nothing && (coeff_plan = _right_pushforward_coeff_plan(
        pi, dom_bases, cod_bases, plan; session_cache=session_cache,
    ))
    act_dom = coeff_plan.act_dom
    act_cod = coeff_plan.act_cod
    comps = Vector{Matrix{K}}(undef, nvertices(pi.P))
    @inbounds for p in 1:nvertices(pi.P)
        Vd = data_dom.dimV[p]
        Vc = data_cod.dimV[p]
        if Vd == 0 || Vc == 0
            comps[p] = zeros(K, Vc, Vd)
            continue
        end
        lo = coeff_plan.ptr[p]
        hi = coeff_plan.ptr[p + 1] - 1
        if hi < lo
            comps[p] = zeros(K, Vc, Vd)
            continue
        end
        offs_dom = data_dom.offsets[p]
        offs_cod = data_cod.offsets[p]
        Kp = data_dom.Ksec[p]
        Lp = data_cod.L[p]
        Fp = zeros(K, Vc, Vd)
        scratch = Matrix{K}(undef, coeff_plan.max_rows[p], coeff_plan.max_cols[p])
        tmp = Matrix{K}(undef, coeff_plan.max_rows[p], Vd)
        for pos in lo:hi
            q = coeff_plan.block_q[pos]
            loc = coeff_plan.block_loc[pos]
            rows = act_cod[q]
            cols = act_dom[q]
            nr = length(rows)
            nc = length(cols)
            rd = (offs_dom[loc] + 1):offs_dom[loc + 1]
            cc = (offs_cod[loc] + 1):offs_cod[loc + 1]
            Cq = @view scratch[1:nr, 1:nc]
            tmpv = @view tmp[1:nr, :]
            _gather_component_matrix!(Cq, C, rows, cols)
            mul!(tmpv, Cq, @view(Kp[rd, :]))
            mul!(Fp, @view(Lp[:, cc]), tmpv, one(K), one(K))
        end
        comps[p] = Fp
    end
    return PMorphism(dom_out, cod_out, comps)
end


"""
    _pmorphism_from_upset_coeff(dom, cod, dom_bases, cod_bases, C)

Internal: build a PMorphism between direct sums of principal upsets from a global coefficient matrix.

`C` has size (#cod_summands) x (#dom_summands).  At vertex `u`, the component is the restriction
to the summands active at `u`, in canonical order.
"""
function _pmorphism_from_upset_coeff(dom::PModule{K}, cod::PModule{K},
                                    dom_bases::Vector{Int}, cod_bases::Vector{Int},
                                    C::AbstractMatrix{K})::PMorphism{K} where {K}
    Q = dom.Q
    @assert Q === cod.Q

    act_dom = _active_upset_indices_from_bases(Q, dom_bases)
    act_cod = _active_upset_indices_from_bases(Q, cod_bases)

    comps = Vector{Matrix{K}}(undef, nvertices(Q))
    for u in 1:nvertices(Q)
        rows = act_cod[u]
        cols = act_dom[u]
        comps[u] = _gather_component_matrix(K, C, rows, cols)
    end
    return PMorphism(dom, cod, comps)
end


end # module ChangeOfPosets
