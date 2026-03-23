# shared.jl -- owner-level cache/shared contracts for DerivedFunctors

using ..CoreModules: AbstractCoeffField, RealField, coeff_type, field_from_eltype, coerce,
                     AbstractHomSystemCache
using ..Options: EncodingOptions, ResolutionOptions, DerivedFunctorOptions
using ..FieldLinAlg
import ..Modules
import ..AbelianCategories
using ..Modules: PModule, PMorphism
using SparseArrays: sparse, SparseMatrixCSC
import Base.Threads

"""
    HomSystemCache{K}()
    HomSystemCache(K::Type)

Thread-local sharded cache for expensive Hom-system setup reused across derived pipelines.

Stored entries:
- `hom`: `HomSpace` objects keyed by `(objectid(dom), objectid(cod))`
- `precompose`: precompose coordinate matrices keyed by `(objectid(Hdom), objectid(Hcod), objectid(f))`
- `postcompose`: postcompose coordinate matrices keyed by `(objectid(Hdom), objectid(Hcod), objectid(g))`
"""
struct _HomKey2
    a::UInt
    b::UInt
end

struct _HomKey3
    a::UInt
    b::UInt
    c::UInt
end

mutable struct HomSystemCache{HV,PV,QV} <: AbstractHomSystemCache
    hom::Vector{Dict{_HomKey2,HV}}
    precompose::Vector{Dict{_HomKey3,PV}}
    postcompose::Vector{Dict{_HomKey3,QV}}
end

function HomSystemCache(::Type{HV}, ::Type{PV}, ::Type{QV}; shard_capacity::Int=256) where {HV,PV,QV}
    nshards = max(1, Threads.maxthreadid())
    hom = [Dict{_HomKey2,HV}() for _ in 1:nshards]
    pre = [Dict{_HomKey3,PV}() for _ in 1:nshards]
    post = [Dict{_HomKey3,QV}() for _ in 1:nshards]
    if shard_capacity > 0
        for d in hom
            sizehint!(d, shard_capacity)
        end
        for d in pre
            sizehint!(d, shard_capacity)
        end
        for d in post
            sizehint!(d, shard_capacity)
        end
    end
    return HomSystemCache(hom, pre, post)
end

@inline _cache_tid_index(shards::AbstractVector) =
    min(length(shards), max(1, Threads.threadid()))

@inline _cache_shard(shards::AbstractVector) = shards[_cache_tid_index(shards)]

function clear_hom_system_cache!(cache::HomSystemCache)
    for d in cache.hom
        empty!(d)
    end
    for d in cache.precompose
        empty!(d)
    end
    for d in cache.postcompose
        empty!(d)
    end
    return nothing
end

@inline _cache_key2(a, b) = _HomKey2(UInt(objectid(a)), UInt(objectid(b)))
@inline _cache_key3(a, b, c) = _HomKey3(UInt(objectid(a)), UInt(objectid(b)), UInt(objectid(c)))

@inline function _cache_lookup(shards::AbstractVector{<:AbstractDict{K,V}}, key::K) where {K,V}
    d = _cache_shard(shards)
    return get(d, key, nothing)::Union{Nothing,V}
end

@inline function _cache_store_or_get!(shards::AbstractVector{<:AbstractDict{K,V}}, key::K, value::V) where {K,V}
    d = _cache_shard(shards)
    existing = get(d, key, nothing)::Union{Nothing,V}
    if existing === nothing
        d[key] = value
        return value
    end
    return existing
end

# -----------------------------------------------------------------------------
# Owner-level semantic accessor generics
# -----------------------------------------------------------------------------

function resolution_terms end
function resolution_differentials end
function augmentation_map end
function coaugmentation_map end
function source_module end
function target_module end
function nonzero_degrees end
function degree_dimensions end
function total_dimension end
function page_dimensions end
function generator_degrees end
function algebra_field end
function resolution_summary end
function hom_summary end
function ext_summary end
function tor_summary end
function double_complex_summary end
function derived_les_summary end
function algebra_summary end
function parent_algebra end
function element_degree end
function element_coordinates end
function wrapped_spectral_sequence end
function underlying_ext_space end
function underlying_tor_space end
function cached_product_degrees end
function check_projective_resolution end
function check_injective_resolution end
function check_ext_spectral_sequence end
function check_tor_spectral_sequence end
function check_ext_algebra end
function check_tor_algebra end

"""
    DerivedFunctorValidationSummary

Compact wrapper around a validation report produced by the `DerivedFunctors`
UX-layer `check_*` helpers.

The wrapped report is a `NamedTuple` whose exact auxiliary fields depend on the
validated object, but every report contains at least:
- `kind`: symbolic object kind
- `valid`: overall validation result
- `issues`: a tuple of human-readable validation issues
"""
struct DerivedFunctorValidationSummary{R}
    report::R
end

@inline derived_functor_validation_summary(report::NamedTuple) = DerivedFunctorValidationSummary(report)

@inline function _derived_validation_report(
    kind::Symbol,
    valid::Bool;
    issues::AbstractVector{<:AbstractString}=String[],
    kwargs...,
)
    return (; kind, valid, issues=Tuple(String.(issues)), kwargs...)
end

@inline function _throw_invalid_derived_functor(fname::Symbol, issues::AbstractVector{<:AbstractString})
    msg = isempty(issues) ? "invalid object" : " - " * join(issues, "\n - ")
    Base.throw(ArgumentError(string(fname, ": validation failed\n", msg)))
end

function Base.show(io::IO, summary::DerivedFunctorValidationSummary)
    r = summary.report
    print(io, "DerivedFunctorValidationSummary(kind=", r.kind,
          ", valid=", r.valid,
          ", issues=", length(r.issues), ")")
end

function Base.show(io::IO, ::MIME"text/plain", summary::DerivedFunctorValidationSummary)
    r = summary.report
    println(io, "DerivedFunctorValidationSummary")
    println(io, "  kind: ", r.kind)
    println(io, "  valid: ", r.valid)
    println(io, "  issues: ", length(r.issues))
    if !isempty(r.issues)
        println(io, "  first_issue: ", first(r.issues))
    end
end

@inline _total_offset_aidx(a::Int, amin::Int) = a - amin + 1

function _build_total_offsets_grid(
    amin::Int, amax::Int,
    bmin::Int, bmax::Int,
    dims::AbstractMatrix{Int},
)
    tmin = amin + bmin
    tmax = amax + bmax
    offsets = [fill(-1, amax - amin + 1) for _ in tmin:tmax]
    dimsCt = zeros(Int, tmax - tmin + 1)

    for t in tmin:tmax
        off = 0
        row = offsets[t - tmin + 1]
        alo = max(amin, t - bmax)
        ahi = min(amax, t - bmin)
        for a in alo:ahi
            ai = _total_offset_aidx(a, amin)
            b = t - a
            bi = b - bmin + 1
            row[ai] = off
            off += dims[ai, bi]
        end
        dimsCt[t - tmin + 1] = off
    end

    return offsets, dimsCt, tmin, tmax
end

@inline function _total_offset_get(
    offsets::Vector{Vector{Int}},
    t::Int,
    tmin::Int,
    amin::Int,
    a::Int,
)
    v = offsets[t - tmin + 1][_total_offset_aidx(a, amin)]
    v >= 0 || error("_total_offset_get: invalid (t,a)=($t,$a)")
    return v
end

"""
Utils: shared low-level utilities for the derived-functors layer.

Intended contents (move here incrementally):
- small linear algebra helpers
- sparse-matrix manipulation helpers
- caching/memoization helpers local to DerivedFunctors
- generic composition and indexing helpers

Design rule:
- keep this dependency-light; higher-level constructions should depend on Utils,
  not the other way around.
"""
