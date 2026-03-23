# =============================================================================
# Encoding.jl
#
# Finite encodings built from fringe presentations and their associated image /
# preimage operations.
#
# Ownership guide
# - finite encoding records: `EncodingMap`, `UptightEncoding`
# - postcomposition with ambient encodings: `PostcomposedEncodingMap`
# - uptight-region construction: `_uptight_regions`, `_uptight_poset`,
#   `build_uptight_encoding_from_fringe`
# - finite image/preimage and fringe push/pull helpers:
#   `_image_*`, `_preimage_*`, `pullback_fringe_along_encoding`,
#   `pushforward_fringe_along_encoding`
#
# This sibling module depends on `FiniteFringe` and `EncodingCore`, but is
# intentionally separate from the hot finite-poset / Hom / fiber kernels and
# from the generic ambient encoding interface.
# =============================================================================

"""
    Encoding

Finite-poset encodings built from fringe presentations.

# What this module owns

This subsystem owns the finite encoding objects and finite-poset translation
helpers used to compress or transfer fringe data:

- [`EncodingMap`](@ref), a finite encoding `pi : Q -> P`,
- [`UptightEncoding`](@ref), which packages an encoding together with the
  constant-upset family used to build it,
- [`PostcomposedEncodingMap`](@ref), which composes a finite encoding with an
  ambient `EncodingCore`-style map,
- uptight-region construction from a fringe module,
- finite encoding image/preimage helpers and fringe pushforward/pullback along
  a finite encoding.

# What this module does not own

`Encoding` does **not** own:

- generic ambient encoding interfaces, compiled ambient maps, or generic
  inspection/query APIs (`TamerOp.EncodingCore`);
- finite-poset/fringe kernels such as `Hom`, `fiber_dimension`, or cover-cache
  logic (`TamerOp.FiniteFringe`).

# Canonical entrypoints

The canonical public entrypoints owned here are:

- [`EncodingMap(...)`](@ref),
- [`UptightEncoding(...)`](@ref),
- [`PostcomposedEncodingMap(...)`](@ref),
- [`build_uptight_encoding_from_fringe(...)`](@ref),
- [`pullback_fringe_along_encoding(...)`](@ref),
- [`pushforward_fringe_along_encoding(...)`](@ref).

Use `EncodingCore` for generic ambient-map inspection and query operations such
as `locate`, `representatives`, and compiled encoding summaries. Use
`FiniteFringe` for the ambient finite-poset and fringe-module objects that the
finite encoding acts on.

# Example workflow

```jldoctest
julia> P = TamerOp.FiniteFringe.FinitePoset(Bool[1 1; 0 1]);

julia> U = [TamerOp.FiniteFringe.principal_upset(P, 1)];

julia> D = [TamerOp.FiniteFringe.principal_downset(P, 2)];

julia> M = TamerOp.FiniteFringe.FringeModule{TamerOp.QQ}(P, U, D, reshape(TamerOp.QQ[1], 1, 1); field=TamerOp.QQ);

julia> enc = TamerOp.Encoding.build_uptight_encoding_from_fringe(M);

julia> TamerOp.ChainComplexes.describe(enc).kind
:uptight_encoding

julia> TamerOp.Encoding.check_uptight_encoding(enc).valid
true

julia> Hpush = TamerOp.Encoding.pushforward_fringe_along_encoding(M, TamerOp.Encoding.encoding_map(enc));

julia> TamerOp.FiniteFringe.ambient_poset(Hpush) === TamerOp.Encoding.target_poset(enc)
true
```
"""
module Encoding
# =============================================================================
# Finite encodings ("uptight posets") from a finite family of constant upsets.
#
# References: Miller section 4 (Defs. 4.12 - 4.18 and Thm. 4.19 - 4.22).
# =============================================================================

using SparseArrays
import ..FiniteFringe
import ..FiniteFringe: AbstractPoset, FringeModule, RegionsPoset, nvertices, leq, upset_indices, downset_indices
import ..FiniteFringe: ambient_poset
import ..EncodingCore: encoding_map, check_encoding_map

# This module defines methods on EncodingCore.AbstractPLikeEncodingMap and
# calls EncodingCore.locate/dimension/etc., so we must import the sibling module
# binding into scope (not just individual names).
import ..EncodingCore

# ----------------------------- Data structures -------------------------------

"""
    EncodingMap
    EncodingMap(Q, P, pi_of_q) -> EncodingMap

A finite encoding `pi : Q -> P` between finite posets.

# Mathematical meaning

`EncodingMap` represents a finite map from a source poset `Q` to a target
finite poset `P`. The map is stored vertexwise: for each source vertex `q`, the
entry `pi_of_q[q]` is the target vertex `pi(q)`.

This is the canonical finite encoding object used when a finite fringe model on
`Q` is compressed or transferred to a smaller poset `P`. The constructor
`EncodingMap(Q, P, pi_of_q)` always means a map `pi : Q -> P`, and the entry
`pi_of_q[q]` is the label of the target region containing the source vertex
`q`.

# Inputs

- `Q::AbstractPoset`: source finite poset.
- `P::AbstractPoset`: target finite poset.
- `pi_of_q::AbstractVector{<:Integer}`: vector of length `nvertices(Q)` with
  image labels in `1:nvertices(P)`.

# Output

- An `EncodingMap{typeof(Q),typeof(P)}` representing `pi : Q -> P`.

# Stored representation

- `Q`, `P`: source and target posets.
- `pi_of_q[q]`: the target-region label assigned to source vertex `q`.
- `inverse_fibers`: lazily built inverse-fiber plan used by internal image and
  preimage routines.
- `label_cache`: lazily built cache for pushed-forward and pulled-back upset and
  downset labels.

The cache fields are implementation details, not part of the intended user API.
Inspect the mathematical object through the semantic interface (`Q`, `P`,
`pi_of_q`, `describe(...)`, and the image/pullback helpers) rather than by
relying on cache internals.

# Invariants

- `length(pi_of_q) == nvertices(Q)`.
- Every entry satisfies `1 <= pi_of_q[q] <= nvertices(P)`.

# Failure / contract behavior

- Throws if the image vector length does not match the source poset size.
- Throws if any image label lies outside the target vertex range.

# Best practices

- Use `EncodingMap` for finite-poset encodings only; it is not the generic
  ambient encoding interface.
- Use `EncodingCore` types for ambient geometric/continuous encodings.
- Use [`build_uptight_encoding_from_fringe`](@ref) when the encoding should be
  derived canonically from a fringe module rather than assembled by hand.

# Examples

```jldoctest
julia> P = TamerOp.FiniteFringe.FinitePoset(Bool[1 1; 0 1]);

julia> pi = TamerOp.Encoding.EncodingMap(P, P, [1, 2]);

julia> TamerOp.ChainComplexes.describe(pi).image_size
2

julia> TamerOp.Encoding.check_encoding_map(pi).valid
true
```
"""
struct _EncodingFiberPlan
    ptr::Vector{Int}
    data::Vector{Int}
end

struct _EncodingLabelCache{QPoset<:AbstractPoset,PPoset<:AbstractPoset}
    image_upsets::IdDict{FiniteFringe.Upset{QPoset},FiniteFringe.Upset{PPoset}}
    image_downsets::IdDict{FiniteFringe.Downset{QPoset},FiniteFringe.Downset{PPoset}}
    preimage_upsets::IdDict{FiniteFringe.Upset{PPoset},FiniteFringe.Upset{QPoset}}
    preimage_downsets::IdDict{FiniteFringe.Downset{PPoset},FiniteFringe.Downset{QPoset}}
end

@inline function _EncodingLabelCache(::Type{QPoset}, ::Type{PPoset}) where {QPoset<:AbstractPoset,PPoset<:AbstractPoset}
    return _EncodingLabelCache{QPoset,PPoset}(
        IdDict{FiniteFringe.Upset{QPoset},FiniteFringe.Upset{PPoset}}(),
        IdDict{FiniteFringe.Downset{QPoset},FiniteFringe.Downset{PPoset}}(),
        IdDict{FiniteFringe.Upset{PPoset},FiniteFringe.Upset{QPoset}}(),
        IdDict{FiniteFringe.Downset{PPoset},FiniteFringe.Downset{QPoset}}(),
    )
end

struct EncodingMap{QPoset<:AbstractPoset,PPoset<:AbstractPoset}
    Q::QPoset
    P::PPoset
    pi_of_q::Vector{Int}
    inverse_fibers::Base.RefValue{Union{Nothing,_EncodingFiberPlan}}
    label_cache::Base.RefValue{Union{Nothing,_EncodingLabelCache{QPoset,PPoset}}}

    function EncodingMap(Qobj::QPoset,
                         Pobj::PPoset,
                         pi_of_q::Vector{Int},
                         inverse_fibers::Base.RefValue{Union{Nothing,_EncodingFiberPlan}},
                         label_cache::Base.RefValue{Union{Nothing,_EncodingLabelCache{QPoset,PPoset}}}) where {QPoset<:AbstractPoset,PPoset<:AbstractPoset}
        length(pi_of_q) == nvertices(Qobj) || error("EncodingMap: pi_of_q length must equal nvertices(Q).")
        np = nvertices(Pobj)
        @inbounds for q in eachindex(pi_of_q)
            p = pi_of_q[q]
            1 <= p <= np || error("EncodingMap: pi_of_q[$q] must lie in 1:nvertices(P).")
        end
        return new{QPoset,PPoset}(Qobj, Pobj, pi_of_q, inverse_fibers, label_cache)
    end
end

function EncodingMap(Qobj::QPoset, Pobj::PPoset, pi_of_q::AbstractVector{<:Integer}) where {QPoset<:AbstractPoset,PPoset<:AbstractPoset}
    return EncodingMap(
        Qobj,
        Pobj,
        Int[pi_of_q...],
        Ref{Union{Nothing,_EncodingFiberPlan}}(nothing),
        Ref{Union{Nothing,_EncodingLabelCache{QPoset,PPoset}}}(nothing),
    )
end

"""
    source_poset(pi::EncodingMap) -> AbstractPoset
    target_poset(pi::EncodingMap) -> AbstractPoset
    source_poset(enc::UptightEncoding) -> AbstractPoset
    target_poset(enc::UptightEncoding) -> AbstractPoset

Return the source and target finite posets of a finite encoding object.

These are the canonical semantic accessors for the finite-poset direction of an
encoding. Prefer them over reading `Q` and `P` fields directly in ordinary user
code.
"""
@inline source_poset(pi::EncodingMap) = pi.Q
@inline target_poset(pi::EncodingMap) = pi.P

"""
    region_map(pi::EncodingMap) -> Vector{Int}

Return the source-to-target region map of a finite encoding.

The returned vector is the canonical semantic view of `pi_of_q`: the entry at
index `q` is the target region label `pi(q)`.
"""
@inline region_map(pi::EncodingMap) = pi.pi_of_q

@inline ambient_poset(pi::EncodingMap) = target_poset(pi)

@inline function _encoding_image_size(pi::EncodingMap)
    return isempty(pi.pi_of_q) ? 0 : length(unique(pi.pi_of_q))
end

@inline function _encoding_sibling_describe(pi::EncodingMap)
    return (kind=:finite_encoding_map,
            source_poset_kind=Symbol(nameof(typeof(pi.Q))),
            target_poset_kind=Symbol(nameof(typeof(pi.P))),
            nsource=nvertices(pi.Q),
            ntarget=nvertices(pi.P),
            image_size=_encoding_image_size(pi),
            inverse_fibers_cached=pi.inverse_fibers[] !== nothing,
            label_cache_cached=pi.label_cache[] !== nothing)
end

"""
    UptightEncoding
    UptightEncoding(pi, Y) -> UptightEncoding

Finite encoding together with the constant-upset family used to build it.

# Mathematical meaning

`UptightEncoding` packages a finite encoding `pi : Q -> P` together with the
family `Y` of constant upsets on `Q` from which the uptight target poset `P`
was constructed. This is the canonical inspection/debugging object for
uptight-region encodings derived from fringe modules.

# Inputs

- `pi::EncodingMap`: the induced finite encoding `Q -> P`.
- `Y::Vector{<:Upset}`: the constant-upset family on `Q` used to define the
  uptight regions.

# Output

- An `UptightEncoding` carrying both the encoding and its defining family.

# Invariants

- `Y` is a family of upsets on the source poset of `pi`.
- `pi` and `Y` refer to the same ambient source poset `Q`.

# Failure / contract behavior

- The outer constructor assumes `Y` is already type-consistent with `pi.Q`.
- This object does not recompute or validate the entire uptight construction;
  use the canonical builder when you want a guaranteed consistent encoding.

# Best practices

- Prefer [`build_uptight_encoding_from_fringe`](@ref) for ordinary workflows.
- Use `UptightEncoding(pi, Y)` when you deliberately want the finite encoding
  together with its defining family for inspection, debugging, or experiments.

# Examples

```jldoctest
julia> P = TamerOp.FiniteFringe.FinitePoset(Bool[1 1; 0 1]);

julia> U = [TamerOp.FiniteFringe.principal_upset(P, 1),
            TamerOp.FiniteFringe.principal_upset(P, 2)];

julia> pi = TamerOp.Encoding.EncodingMap(P, P, [1, 2]);

julia> enc = TamerOp.Encoding.UptightEncoding(pi, U);

julia> TamerOp.ChainComplexes.describe(enc).nconstant_upsets
2
```
"""
struct UptightEncoding{QPoset<:AbstractPoset,PPoset<:AbstractPoset,U<:FiniteFringe.Upset{QPoset}}
    pi::EncodingMap{QPoset,PPoset}
    Y::Vector{U}
    @inline function UptightEncoding{QPoset,PPoset,U}(pi::EncodingMap{QPoset,PPoset}, Y::Vector{U}) where {QPoset<:AbstractPoset,PPoset<:AbstractPoset,U<:FiniteFringe.Upset{QPoset}}
        return new{QPoset,PPoset,U}(pi, Y)
    end
end

@inline UptightEncoding(pi::EncodingMap{QPoset,PPoset}, Y::Vector{U}) where {QPoset<:AbstractPoset,PPoset<:AbstractPoset,U<:FiniteFringe.Upset{QPoset}} =
    UptightEncoding{QPoset,PPoset,U}(pi, Y)

"""
    constant_upsets(enc::UptightEncoding) -> Vector{Upset}

Return the constant-upset family used to define the uptight encoding.
"""
@inline constant_upsets(enc::UptightEncoding) = enc.Y

"""
    encoding_map(enc::UptightEncoding) -> EncodingMap

Return the underlying finite encoding of an uptight encoding bundle.
"""
@inline encoding_map(enc::UptightEncoding) = enc.pi

@inline source_poset(enc::UptightEncoding) = source_poset(enc.pi)
@inline target_poset(enc::UptightEncoding) = target_poset(enc.pi)
@inline ambient_poset(enc::UptightEncoding) = target_poset(enc.pi)

@inline function _encoding_sibling_describe(enc::UptightEncoding)
    pi = enc.pi
    return (kind=:uptight_encoding,
            source_poset_kind=Symbol(nameof(typeof(pi.Q))),
            target_poset_kind=Symbol(nameof(typeof(pi.P))),
            nsource=nvertices(pi.Q),
            ntarget=nvertices(pi.P),
            image_size=_encoding_image_size(pi),
            nconstant_upsets=length(enc.Y),
            inverse_fibers_cached=pi.inverse_fibers[] !== nothing,
            label_cache_cached=pi.label_cache[] !== nothing)
end

# ------------------ Postcomposition with a finite encoding map ---------------

"""
    PostcomposedEncodingMap(pi0, pi)

Postcompose an *ambient* encoding map `pi0` (e.g. Z^n- or R^n-encoding) with a *finite*
encoding map `pi : Q -> P` (an `EncodingMap`). Conceptually:

    x mapsto q = locate(pi0, x) mapsto pi(q)

This is used by `Workflow.coarsen` to keep user-facing `encode(...)` semantics stable
while compressing the *finite* encoding poset.

# Mathematical meaning

If `pi0 : X -> Q` is an ambient encoding and `pi : Q -> P` is a finite encoding,
then `PostcomposedEncodingMap(pi0, pi)` represents the composite

`X -> Q -> P`.

# Stored representation

- `pi0`: the ambient encoding map.
- `pi_of_q`: the finite encoding labels of `pi`.
- `Pn`: target finite-poset size `nvertices(P)`.
- `reps_cache`: lazily built representative cache for the postcomposed target
  regions.

The representative cache is an implementation detail rather than a public
contract surface.

# Failure / contract behavior

- Construction assumes `pi` is compatible with the codomain regions of `pi0`.
- Representative construction throws if some target region of the finite map is
  not hit by the postcomposition.

# Best practices

- Use this helper when you want to keep an ambient `EncodingCore`-style user
  surface while replacing the finite target poset by a coarser one.
- For purely finite-poset work, use [`EncodingMap`](@ref) directly.
- `PostcomposedEncodingMap` is the canonical bridge from a generic ambient
  `EncodingCore` map to a finite encoding map.
"""
struct PostcomposedEncodingMap{PI<:EncodingCore.AbstractPLikeEncodingMap,R<:Tuple} <: EncodingCore.AbstractPLikeEncodingMap
    pi0::PI
    pi_of_q::Vector{Int}
    Pn::Int
    reps_cache::Base.RefValue{Union{Nothing,Vector{R}}}
end

@inline function _postcomposed_rep_type(pi0::EncodingCore.AbstractPLikeEncodingMap)
    reps0 = EncodingCore.representatives(pi0)
    isempty(reps0) && return Tuple{}
    r1 = reps0[1]
    return r1 isa Tuple ? typeof(r1) : typeof(Tuple(r1))
end

@inline function PostcomposedEncodingMap(pi0::EncodingCore.AbstractPLikeEncodingMap, pi::EncodingMap)
    R = _postcomposed_rep_type(pi0)
    return PostcomposedEncodingMap{typeof(pi0),R}(pi0, pi.pi_of_q, nvertices(pi.P), Ref{Union{Nothing,Vector{R}}}(nothing))
end

@inline function _postcomposed_source_poset(pi::PostcomposedEncodingMap)
    if applicable(EncodingCore.encoding_poset, pi.pi0)
        return EncodingCore.encoding_poset(pi.pi0)
    end
    return nothing
end

@inline source_poset(pi::PostcomposedEncodingMap) = _postcomposed_source_poset(pi)
@inline target_poset(pi::PostcomposedEncodingMap) = nothing
@inline ambient_poset(pi::PostcomposedEncodingMap) = target_poset(pi)

@inline function _encoding_sibling_describe(pi::PostcomposedEncodingMap)
    srcP = _postcomposed_source_poset(pi)
    return (kind=:postcomposed_encoding_map,
            source_poset_kind=srcP === nothing ? nothing : Symbol(nameof(typeof(srcP))),
            target_poset_kind=nothing,
            nsource=length(pi.pi_of_q),
            ntarget=pi.Pn,
            image_size=isempty(pi.pi_of_q) ? 0 : length(unique(pi.pi_of_q)),
            representatives_cached=pi.reps_cache[] !== nothing,
            ambient_map_type=typeof(pi.pi0))
end

@inline function EncodingCore.locate(pi::PostcomposedEncodingMap, x::AbstractVector; kwargs...)
    q = EncodingCore.locate(pi.pi0, x; kwargs...)
    q == 0 && return 0
    @inbounds return pi.pi_of_q[q]
end

@inline function EncodingCore.locate(pi::PostcomposedEncodingMap, x::NTuple{N,T}; kwargs...) where {N,T<:Real}
    q = EncodingCore.locate(pi.pi0, x; kwargs...)
    q == 0 && return 0
    @inbounds return pi.pi_of_q[q]
end

@inline EncodingCore.dimension(pi::PostcomposedEncodingMap) = EncodingCore.dimension(pi.pi0)
@inline EncodingCore.axes_from_encoding(pi::PostcomposedEncodingMap) = EncodingCore.axes_from_encoding(pi.pi0)

function EncodingCore.representatives(pi::PostcomposedEncodingMap{PI,R}) where {PI,R}
    cached = pi.reps_cache[]
    cached !== nothing && return cached

    reps0 = EncodingCore.representatives(pi.pi0)
    repsP = Vector{R}(undef, pi.Pn)
    filled = falses(pi.Pn)

    @inbounds for q in eachindex(pi.pi_of_q)
        p = pi.pi_of_q[q]
        if !filled[p]
            repsP[p] = reps0[q] isa Tuple ? reps0[q] : Tuple(reps0[q])
            filled[p] = true
        end
    end

    all(filled) || error("PostcomposedEncodingMap: could not build representatives for all regions.")

    pi.reps_cache[] = repsP
    return repsP
end

@inline function _encoding_show(io::IO, pi::EncodingMap)
    print(io, "EncodingMap(nQ=", nvertices(pi.Q),
          ", nP=", nvertices(pi.P),
          ", image_size=", _encoding_image_size(pi), ")")
end

@inline function _encoding_show(io::IO, enc::UptightEncoding)
    pi = enc.pi
    print(io, "UptightEncoding(nQ=", nvertices(pi.Q),
          ", nP=", nvertices(pi.P),
          ", nconstant_upsets=", length(enc.Y), ")")
end

@inline function _encoding_show(io::IO, pi::PostcomposedEncodingMap)
    print(io, "PostcomposedEncodingMap(dim=", EncodingCore.dimension(pi),
          ", nregions=", pi.Pn, ")")
end

Base.show(io::IO, pi::EncodingMap) = _encoding_show(io, pi)
Base.show(io::IO, enc::UptightEncoding) = _encoding_show(io, enc)
Base.show(io::IO, pi::PostcomposedEncodingMap) = _encoding_show(io, pi)

function Base.show(io::IO, ::MIME"text/plain", pi::EncodingMap)
    println(io, "EncodingMap")
    println(io, "  source: ", typeof(pi.Q), " with ", nvertices(pi.Q), " vertices")
    println(io, "  target: ", typeof(pi.P), " with ", nvertices(pi.P), " vertices")
    println(io, "  image_size: ", _encoding_image_size(pi))
    println(io, "  inverse_fibers_cached: ", pi.inverse_fibers[] !== nothing)
    print(io,   "  label_cache_cached: ", pi.label_cache[] !== nothing)
end

function Base.show(io::IO, ::MIME"text/plain", enc::UptightEncoding)
    pi = enc.pi
    println(io, "UptightEncoding")
    println(io, "  source: ", typeof(pi.Q), " with ", nvertices(pi.Q), " vertices")
    println(io, "  target: ", typeof(pi.P), " with ", nvertices(pi.P), " vertices")
    println(io, "  image_size: ", _encoding_image_size(pi))
    println(io, "  nconstant_upsets: ", length(enc.Y))
    println(io, "  inverse_fibers_cached: ", pi.inverse_fibers[] !== nothing)
    print(io,   "  label_cache_cached: ", pi.label_cache[] !== nothing)
end

function Base.show(io::IO, ::MIME"text/plain", pi::PostcomposedEncodingMap)
    println(io, "PostcomposedEncodingMap")
    println(io, "  ambient_map_type: ", typeof(pi.pi0))
    println(io, "  parameter_dim: ", EncodingCore.dimension(pi))
    println(io, "  nregions: ", pi.Pn)
    println(io, "  image_size: ", isempty(pi.pi_of_q) ? 0 : length(unique(pi.pi_of_q)))
    print(io,   "  representatives_cached: ", pi.reps_cache[] !== nothing)
end

"""
    check_encoding_map(pi::EncodingMap; throw=false) -> NamedTuple

Validate a finite encoding map `pi : Q -> P`.

The returned report checks:

- `length(region_map(pi)) == nvertices(source_poset(pi))`,
- every image label lies in `1:nvertices(target_poset(pi))`,
- the image actually hits at most the declared target range.

Set `throw=true` to raise an `ArgumentError` instead of returning an invalid
report.

```jldoctest
julia> P = TamerOp.FiniteFringe.FinitePoset(Bool[1 1; 0 1]);

julia> pi = TamerOp.Encoding.EncodingMap(P, P, [1, 2]);

julia> TamerOp.Encoding.check_encoding_map(pi).valid
true
```
"""
function check_encoding_map(pi::EncodingMap; throw::Bool=false)
    issues = String[]
    nQ = nvertices(pi.Q)
    nP = nvertices(pi.P)
    length(pi.pi_of_q) == nQ || push!(issues, "pi_of_q length $(length(pi.pi_of_q)) must equal nvertices(Q)=$nQ")
    @inbounds for q in eachindex(pi.pi_of_q)
        p = pi.pi_of_q[q]
        1 <= p <= nP || push!(issues, "pi_of_q[$q]=$p must lie in 1:nvertices(P)=$nP")
    end
    report = (kind=:finite_encoding_map,
              valid=isempty(issues),
              source_poset_kind=Symbol(nameof(typeof(pi.Q))),
              target_poset_kind=Symbol(nameof(typeof(pi.P))),
              nsource=nQ,
              ntarget=nP,
              image_size=_encoding_image_size(pi),
              issues=issues)
    if throw && !report.valid
        throw(ArgumentError("check_encoding_map: invalid finite encoding map: " * join(report.issues, "; ")))
    end
    return report
end

"""
    check_uptight_encoding(enc::UptightEncoding; throw=false) -> NamedTuple

Validate an uptight encoding bundle.

The returned report checks:

- the underlying finite encoding map is valid,
- all constant upsets belong to the source poset of the encoding,
- the constant-upset family and the stored region map are structurally
  compatible,
- every target label is hit and is represented by exactly one signature class.

Set `throw=true` to raise an `ArgumentError` instead of returning an invalid
report.

```jldoctest
julia> P = TamerOp.FiniteFringe.FinitePoset(Bool[1 1; 0 1]);

julia> Y = [TamerOp.FiniteFringe.principal_upset(P, 1),
            TamerOp.FiniteFringe.principal_upset(P, 2)];

julia> enc = TamerOp.Encoding.UptightEncoding(TamerOp.Encoding.EncodingMap(P, P, [1, 2]), Y);

julia> TamerOp.Encoding.check_uptight_encoding(enc).valid
true
```
"""
function check_uptight_encoding(enc::UptightEncoding; throw::Bool=false)
    issues = String[]
    pi = enc.pi
    map_report = check_encoding_map(pi)
    map_report.valid || append!(issues, String.(map_report.issues))
    Q = pi.Q
    P = pi.P
    Y = enc.Y
    @inbounds for (i, U) in enumerate(Y)
        U.P === Q || push!(issues, "constant upset $i does not belong to the source poset")
    end
    sig_by_label = Dict{Int,BitVector}()
    label_by_sig = Dict{BitVector,Int}()
    @inbounds for q in 1:nvertices(Q)
        sig = falses(length(Y))
        for i in eachindex(Y)
            sig[i] = Y[i].mask[q]
        end
        p = pi.pi_of_q[q]
        prevsig = get(sig_by_label, p, nothing)
        if prevsig === nothing
            sig_by_label[p] = copy(sig)
        elseif prevsig != sig
            push!(issues, "target label $p is assigned to multiple signature classes")
            break
        end
        prevp = get(label_by_sig, sig, 0)
        if prevp == 0
            label_by_sig[copy(sig)] = p
        elseif prevp != p
            push!(issues, "equal signatures map to different target labels $prevp and $p")
            break
        end
    end
    length(sig_by_label) == nvertices(P) || push!(issues, "image size $(length(sig_by_label)) must equal nvertices(P)=$(nvertices(P))")
    report = (kind=:uptight_encoding,
              valid=isempty(issues),
              source_poset_kind=Symbol(nameof(typeof(Q))),
              target_poset_kind=Symbol(nameof(typeof(P))),
              nsource=nvertices(Q),
              ntarget=nvertices(P),
              nconstant_upsets=length(Y),
              issues=issues)
    if throw && !report.valid
        throw(ArgumentError("check_uptight_encoding: invalid uptight encoding: " * join(report.issues, "; ")))
    end
    return report
end

"""
    check_postcomposed_encoding(pi::PostcomposedEncodingMap; throw=false) -> NamedTuple

Validate a postcomposed ambient-plus-finite encoding map.

The returned report checks:

- the ambient map `pi.pi0` satisfies the generic `EncodingCore` map contract,
- the finite label map length matches the number of ambient source regions,
- every target label lies in `1:pi.Pn`,
- the cached representatives, when populated, have the correct length.

Set `throw=true` to raise an `ArgumentError` instead of returning an invalid
report.
"""
function check_postcomposed_encoding(pi::PostcomposedEncodingMap; throw::Bool=false)
    issues = String[]
    upstream = EncodingCore.check_encoding_map(pi.pi0)
    upstream.valid || append!(issues, String.(upstream.issues))
    reps0 = try
        EncodingCore.representatives(pi.pi0)
    catch err
        push!(issues, "representatives(pi0) failed: $(sprint(showerror, err))")
        Tuple[]
    end
    length(pi.pi_of_q) == length(reps0) || push!(issues, "pi_of_q length $(length(pi.pi_of_q)) must equal number of source regions $(length(reps0))")
    @inbounds for q in eachindex(pi.pi_of_q)
        p = pi.pi_of_q[q]
        1 <= p <= pi.Pn || push!(issues, "pi_of_q[$q]=$p must lie in 1:$(pi.Pn)")
    end
    cached = pi.reps_cache[]
    cached === nothing || length(cached) == pi.Pn || push!(issues, "cached representatives length $(length(cached)) must equal target size $(pi.Pn)")
    report = (kind=:postcomposed_encoding_map,
              valid=isempty(issues),
              nsource=length(pi.pi_of_q),
              ntarget=pi.Pn,
              image_size=isempty(pi.pi_of_q) ? 0 : length(unique(pi.pi_of_q)),
              representatives_cached=cached !== nothing,
              issues=issues)
    if throw && !report.valid
        throw(ArgumentError("check_postcomposed_encoding: invalid postcomposed encoding: " * join(report.issues, "; ")))
    end
    return report
end

check_encoding_map(pi::PostcomposedEncodingMap; throw::Bool=false) = check_postcomposed_encoding(pi; throw=throw)

# ----------------------- Uptight regions from a family Y ---------------------

function _uptight_regions(Q::FiniteFringe.AbstractPoset, Y::AbstractVector{<:FiniteFringe.Upset})
    m = length(Y)
    if m <= 64
        sigs = Dict{UInt64, Vector{Int}}()
        @inbounds for q in 1:nvertices(Q)
            key = zero(UInt64)
            for i in 1:m
                if Y[i].mask[q]
                    key |= (UInt64(1) << (i - 1))
                end
            end
            vec = get!(sigs, key) do
                Int[]
            end
            push!(vec, q)
        end
        return collect(values(sigs))
    elseif m <= 128
        sigs = Dict{Tuple{UInt64,UInt64}, Vector{Int}}()
        @inbounds for q in 1:nvertices(Q)
            lo = zero(UInt64)
            hi = zero(UInt64)
            for i in 1:64
                if Y[i].mask[q]
                    lo |= (UInt64(1) << (i - 1))
                end
            end
            for i in 65:m
                if Y[i].mask[q]
                    hi |= (UInt64(1) << (i - 65))
                end
            end
            key = (lo, hi)
            vec = get!(sigs, key) do
                Int[]
            end
            push!(vec, q)
        end
        return collect(values(sigs))
    else
        n = nvertices(Q)
        nchunks = cld(m, 64)
        scratch = zeros(UInt64, nchunks)

        heads = Dict{UInt, Int}()
        next_samehash = Int[]
        signatures = Vector{Vector{UInt64}}()
        classes = Vector{Vector{Int}}()

        sizehint!(next_samehash, max(1, min(n, 4096)))
        sizehint!(signatures, max(1, min(n, 4096)))
        sizehint!(classes, max(1, min(n, 4096)))

        @inbounds for q in 1:n
            fill!(scratch, 0x0)
            for i in 1:m
                if Y[i].mask[q]
                    c = ((i - 1) >>> 6) + 1
                    bit = (i - 1) & 0x3f
                    scratch[c] |= UInt64(1) << bit
                end
            end

            h = UInt(0x9e3779b97f4a7c15)
            for c in 1:nchunks
                h = hash(scratch[c], h)
            end

            idx = get(heads, h, 0)
            found = 0
            while idx != 0
                sig = signatures[idx]
                same = true
                for c in 1:nchunks
                    if sig[c] != scratch[c]
                        same = false
                        break
                    end
                end
                if same
                    found = idx
                    break
                end
                idx = next_samehash[idx]
            end

            if found == 0
                push!(signatures, copy(scratch))
                push!(classes, Int[q])
                prev_head = get(heads, h, 0)
                push!(next_samehash, prev_head)
                heads[h] = length(classes)
            else
                push!(classes[found], q)
            end
        end
        return classes
    end
end

@inline function _or_chunks!(dest::BitVector, src::BitVector)
    dc = dest.chunks
    sc = src.chunks
    @inbounds for w in eachindex(dc)
        dc[w] |= sc[w]
    end
    r = length(dest) & 63
    if r != 0
        dc[end] &= (UInt64(1) << r) - 1
    end
    return dest
end

@inline function _is_upset_mask(P::AbstractPoset, mask::BitVector)
    @inbounds for i in 1:nvertices(P)
        mask[i] || continue
        for j in upset_indices(P, i)
            mask[j] || return false
        end
    end
    return true
end

@inline function _is_downset_mask(P::AbstractPoset, mask::BitVector)
    @inbounds for j in 1:nvertices(P)
        mask[j] || continue
        for i in downset_indices(P, j)
            mask[i] || return false
        end
    end
    return true
end

function _uptight_dense_relation(Q::FiniteFringe.AbstractPoset, regions::Vector{Vector{Int}})
    r = length(regions)
    nq = nvertices(Q)
    region_of_q = zeros(Int, nq)
    @inbounds for (idx, R) in enumerate(regions)
        for q in R
            region_of_q[q] = idx
        end
    end

    rows = [falses(r) for _ in 1:r]
    @inbounds for A in 1:r
        row = rows[A]
        row[A] = true
        for a in regions[A]
            for b in upset_indices(Q, a)
                row[region_of_q[b]] = true
            end
        end
    end

    @inbounds for k in 1:r
        rk = rows[k]
        for i in 1:r
            rows[i][k] || continue
            _or_chunks!(rows[i], rk)
        end
    end

    rel = falses(r, r)
    @inbounds for i in 1:r
        rel[i, :] .= rows[i]
    end
    return rel
end

function _uptight_poset(Q::FiniteFringe.AbstractPoset, regions::Vector{Vector{Int}};
                        poset_kind::Symbol = :regions)
    r = length(regions)
    if poset_kind == :regions
        return RegionsPoset(Q, regions)
    elseif poset_kind == :dense
        rel = _uptight_dense_relation(Q, regions)
        return FiniteFringe.FinitePoset(rel; check=false)
    else
        error("_uptight_poset: poset_kind must be :regions or :dense")
    end
end

function _encoding_map(Q::FiniteFringe.AbstractPoset,
                       P::FiniteFringe.AbstractPoset,
                       regions::Vector{Vector{Int}})
    pi_of_q = zeros(Int, nvertices(Q))
    for (idx, R) in enumerate(regions)
        for q in R
            pi_of_q[q] = idx
        end
    end
    EncodingMap(Q, P, pi_of_q)
end

function _build_inverse_fiber_plan(pi::EncodingMap)
    np = nvertices(pi.P)
    nq = nvertices(pi.Q)
    counts = zeros(Int, np)
    @inbounds for q in 1:nq
        counts[pi.pi_of_q[q]] += 1
    end
    ptr = Vector{Int}(undef, np + 1)
    ptr[1] = 1
    @inbounds for p in 1:np
        ptr[p + 1] = ptr[p] + counts[p]
    end
    next = copy(ptr)
    data = Vector{Int}(undef, nq)
    @inbounds for q in 1:nq
        p = pi.pi_of_q[q]
        idx = next[p]
        data[idx] = q
        next[p] = idx + 1
    end
    return _EncodingFiberPlan(ptr, data)
end

@inline function _ensure_inverse_fiber_plan!(pi::EncodingMap)
    plan = pi.inverse_fibers[]
    plan === nothing || return plan
    plan = _build_inverse_fiber_plan(pi)
    pi.inverse_fibers[] = plan
    return plan
end

@inline function _ensure_label_cache!(pi::EncodingMap{QPoset,PPoset}) where {QPoset<:AbstractPoset,PPoset<:AbstractPoset}
    cache = pi.label_cache[]
    cache === nothing || return cache
    cache = _EncodingLabelCache(QPoset, PPoset)
    pi.label_cache[] = cache
    return cache
end

@inline function _clear_encoding_label_cache!(pi::EncodingMap)
    pi.label_cache[] = nothing
    return pi
end

@inline function _pushforward_upset_mask(pi::EncodingMap, U::FiniteFringe.Upset)
    maskP = falses(nvertices(pi.P))
    @inbounds for q in 1:nvertices(pi.Q)
        U.mask[q] && (maskP[pi.pi_of_q[q]] = true)
    end
    return maskP
end

@inline function _pushforward_downset_mask(pi::EncodingMap, D::FiniteFringe.Downset)
    maskP = falses(nvertices(pi.P))
    @inbounds for q in 1:nvertices(pi.Q)
        D.mask[q] && (maskP[pi.pi_of_q[q]] = true)
    end
    return maskP
end

@inline function _pullback_mask(pi::EncodingMap, maskP::BitVector)
    maskQ = falses(nvertices(pi.Q))
    plan = _ensure_inverse_fiber_plan!(pi)
    @inbounds for p in 1:nvertices(pi.P)
        maskP[p] || continue
        for idx in plan.ptr[p]:(plan.ptr[p + 1] - 1)
            maskQ[plan.data[idx]] = true
        end
    end
    return maskQ
end

"""
    _image_upset(pi::EncodingMap, U::Upset) -> Upset

Return the image of a source-poset upset under the finite encoding `pi : Q -> P`
as a `FiniteFringe.Upset` on the target poset `P`.

The result is always returned as a closed upset on `P`; if the pushed-forward
mask is not already upward closed, the function takes the upset closure. Results
are cached by identity in the finite encoding's internal label cache.
"""
function _image_upset(pi::EncodingMap, U::FiniteFringe.Upset)
    cache = _ensure_label_cache!(pi)
    return get!(cache.image_upsets, U) do
        maskP = _pushforward_upset_mask(pi, U)
        _is_upset_mask(pi.P, maskP) ? FiniteFringe.Upset(pi.P, maskP) :
                                      FiniteFringe.upset_closure(pi.P, maskP)
    end
end

"""
    _image_downset(pi::EncodingMap, D::Downset) -> Downset

Return the image of a source-poset downset under the finite encoding `pi : Q -> P`
as a `FiniteFringe.Downset` on the target poset `P`.

The result is always returned as a closed downset on `P`; if the pushed-forward
mask is not already downward closed, the function takes the downset closure.
Results are cached by identity in the finite encoding's internal label cache.
"""
function _image_downset(pi::EncodingMap, D::FiniteFringe.Downset)
    cache = _ensure_label_cache!(pi)
    return get!(cache.image_downsets, D) do
        maskP = _pushforward_downset_mask(pi, D)
        _is_downset_mask(pi.P, maskP) ? FiniteFringe.Downset(pi.P, maskP) :
                                        FiniteFringe.downset_closure(pi.P, maskP)
    end
end

"""
    _preimage_upset(pi::EncodingMap, Uhat::Upset) -> Upset

Return the preimage of a target-poset upset under the finite encoding
`pi : Q -> P` as a `FiniteFringe.Upset` on the source poset `Q`.

Because inverse images of upsets under finite encodings are already upward
closed, the result is returned directly as an upset on `Q`. Results are cached
by identity in the finite encoding's internal label cache.
"""
function _preimage_upset(pi::EncodingMap, Uhat::FiniteFringe.Upset)
    cache = _ensure_label_cache!(pi)
    return get!(cache.preimage_upsets, Uhat) do
        maskQ = _pullback_mask(pi, Uhat.mask)
        FiniteFringe.Upset(pi.Q, maskQ)
    end
end

"""
    _preimage_downset(pi::EncodingMap, Dhat::Downset) -> Downset

Return the preimage of a target-poset downset under the finite encoding
`pi : Q -> P` as a `FiniteFringe.Downset` on the source poset `Q`.

Because inverse images of downsets under finite encodings are already downward
closed, the result is returned directly as a downset on `Q`. Results are cached
by identity in the finite encoding's internal label cache.
"""
function _preimage_downset(pi::EncodingMap, Dhat::FiniteFringe.Downset)
    cache = _ensure_label_cache!(pi)
    return get!(cache.preimage_downsets, Dhat) do
        maskQ = _pullback_mask(pi, Dhat.mask)
        FiniteFringe.Downset(pi.Q, maskQ)
    end
end

"""
    build_uptight_encoding_from_fringe(M::FringeModule; poset_kind=:regions) -> UptightEncoding

Build the canonical finite uptight encoding associated to a fringe module.

Given a fringe presentation on `Q` with upsets `U_i` (births) and downsets
`D_j` (deaths), this constructs the finite family

`Y = {U_i} union {complement(D_j)}`

of constant upsets, forms the uptight regions, builds the induced finite target
poset `P_Y`, and returns an [`UptightEncoding`](@ref) carrying both the finite
encoding `pi : Q -> P_Y` and the constant-upset family `Y`.

The returned `Y` is part of the canonical object, not a debug-only side
attachment: it is the finite family used to define the uptight regions.

`poset_kind=:regions` returns a structured `RegionsPoset`, while `:dense`
materializes a `FinitePoset`.

```jldoctest
julia> P = TamerOp.FiniteFringe.FinitePoset(Bool[1 1; 0 1]);

julia> U = [TamerOp.FiniteFringe.principal_upset(P, 1)];

julia> D = [TamerOp.FiniteFringe.principal_downset(P, 2)];

julia> M = TamerOp.FiniteFringe.FringeModule{TamerOp.QQ}(P, U, D, reshape(TamerOp.QQ[1], 1, 1); field=TamerOp.QQ);

julia> enc = TamerOp.Encoding.build_uptight_encoding_from_fringe(M);

julia> TamerOp.ChainComplexes.describe(enc).kind
:uptight_encoding
```
"""
function build_uptight_encoding_from_fringe(M::FiniteFringe.FringeModule;
                                            poset_kind::Symbol = :regions)
    Q = M.P
    PT = typeof(Q)
    nu = length(M.U)
    nd = length(M.D)
    Y = Vector{FiniteFringe.Upset{PT}}(undef, nu + nd)
    @inbounds for i in 1:nu
        Y[i] = M.U[i]
    end
    @inbounds for j in 1:nd
        comp = BitVector(.!M.D[j].mask)
        Y[nu + j] = FiniteFringe.upset_closure(Q, comp)
    end
    regions = _uptight_regions(Q, Y)
    P = _uptight_poset(Q, regions; poset_kind=poset_kind)
    pi = _encoding_map(Q, P, regions)
    UptightEncoding(pi, Y)
end

"""
    pullback_fringe_along_encoding(H_hat::FringeModule_on_P, pi::EncodingMap) -> FringeModule_on_Q

Prop. 4.11 (used in the proof of Thm. 6.12): pull back a monomial matrix for a module on `P`
by replacing row labels `D_hat_j` with `pi^{-1}(D_hat_j)` and column labels `U_hat_i` with `pi^{-1}(U_hat_i)`.
The scalar matrix is unchanged.

This acts on finite `FiniteFringe` upset/downset labels, not on generic
`EncodingCore` compiled encodings. The returned object is a
`FiniteFringe.FringeModule` on the source poset `Q`, with all upset/downset
labels returned as closed finite-fringe objects. Label translations are cached
by identity through the finite encoding's internal label cache.

```jldoctest
julia> P = TamerOp.FiniteFringe.FinitePoset(Bool[1 1; 0 1]);

julia> pi = TamerOp.Encoding.EncodingMap(P, P, [1, 2]);

julia> H = TamerOp.FiniteFringe.one_by_one_fringe(P,
           TamerOp.FiniteFringe.principal_upset(P, 1),
           TamerOp.FiniteFringe.principal_downset(P, 2),
           TamerOp.QQ(1); field=TamerOp.QQ);

julia> Hpb = TamerOp.Encoding.pullback_fringe_along_encoding(H, pi);

julia> TamerOp.FiniteFringe.ambient_poset(Hpb) === P
true
```
"""
function pullback_fringe_along_encoding(Hhat::FiniteFringe.FringeModule, pi::EncodingMap)
    UQ = Vector{FiniteFringe.Upset{typeof(pi.Q)}}(undef, length(Hhat.U))
    DQ = Vector{FiniteFringe.Downset{typeof(pi.Q)}}(undef, length(Hhat.D))
    @inbounds for i in eachindex(Hhat.U)
        UQ[i] = _preimage_upset(pi, Hhat.U[i])
    end
    @inbounds for j in eachindex(Hhat.D)
        DQ[j] = _preimage_downset(pi, Hhat.D[j])
    end
    FiniteFringe.FringeModule{eltype(Hhat.phi)}(pi.Q, UQ, DQ, Hhat.phi; field=Hhat.field)
end

"""
    pushforward_fringe_along_encoding(H::FringeModule_on_Q, pi::EncodingMap) -> FringeModule_on_P

Push a fringe presentation forward along a finite encoding map `pi : Q -> P`.

This sends each upset generator `U_i` of `H` to its image under `pi` and each
downset generator `D_j` to its image under `pi`, while keeping the scalar matrix
`phi` unchanged.

This acts on finite `FiniteFringe` upset/downset labels, not on generic
compiled ambient encodings. The returned object is a `FiniteFringe.FringeModule`
on the target poset `P`; image labels are returned as closed finite-fringe
upsets/downsets and are cached by identity through the finite encoding's
internal label cache.
"""
function pushforward_fringe_along_encoding(H::FiniteFringe.FringeModule, pi::EncodingMap)
    Uhat = Vector{FiniteFringe.Upset{typeof(pi.P)}}(undef, length(H.U))
    Dhat = Vector{FiniteFringe.Downset{typeof(pi.P)}}(undef, length(H.D))
    @inbounds for i in eachindex(H.U)
        Uhat[i] = _image_upset(pi, H.U[i])
    end
    @inbounds for j in eachindex(H.D)
        Dhat[j] = _image_downset(pi, H.D[j])
    end
    FiniteFringe.FringeModule{eltype(H.phi)}(pi.P, Uhat, Dhat, H.phi; field=H.field)
end

end # module Encoding
