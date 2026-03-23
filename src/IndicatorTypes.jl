# =============================================================================
# IndicatorTypes.jl
#
# One-step indicator presentation and copresentation containers.
#
# This sibling module owns the small algebraic record types used by indicator
# resolutions and related derived-functor code. It depends on `FiniteFringe`
# for the underlying poset and label objects, but does not own fringe-module
# kernels, encoding machinery, or derived-functor algorithms.
# =============================================================================

"""
    IndicatorTypes

One-step indicator presentations and copresentations over finite posets.

# Mathematical model

This subsystem packages the two basic algebraic containers used throughout the
indicator-resolution and derived-functor code:

- [`UpsetPresentation`](@ref), representing a one-step presentation
  `F_1 -> F_0 -> M` with summands labeled by upsets;
- [`DownsetCopresentation`](@ref), representing a one-step copresentation
  `M -> E^0 -> E^1` with summands labeled by downsets.

These objects record:

- the ambient finite poset,
- the label families,
- the coefficient matrix of the presentation or copresentation map,
- and, optionally, an attached fringe module `H` when an exact source/target
  object is available.

# Canonical public entrypoints

The canonical public entrypoints owned by this module are:

- [`UpsetPresentation(...)`](@ref),
- [`UpsetPresentation{K}(...)`](@ref),
- [`DownsetCopresentation(...)`](@ref),
- [`DownsetCopresentation{K}(...)`](@ref).

For inspection and validation, use:

- `describe(X)` for compact semantic summaries,
- [`ambient_poset`](@ref) and [`field`](@ref),
- [`generator_labels`](@ref), [`relation_labels`](@ref),
  [`cogenerator_labels`](@ref), [`corelation_labels`](@ref),
- [`presentation_matrix`](@ref), [`copresentation_matrix`](@ref),
- [`attached_fringe`](@ref),
- [`check_upset_presentation`](@ref), [`check_downset_copresentation`](@ref).

# Relationship to `IndicatorResolutions`

`IndicatorResolutions` builds and consumes these containers as the lightweight
algebraic interface between fringe modules and derived-functor code. In
practice:

- use `IndicatorResolutions.fringe_presentation(H)` when you want a canonical
  `UpsetPresentation` or `DownsetCopresentation` from a fringe module,
- inspect the result with `describe(...)`, semantic accessors, and `check_*`,
- pass the resulting objects into resolution or Ext/Tor workflows rather than
  unpacking their raw fields manually.

# Ownership boundary

`IndicatorTypes` owns the indicator presentation/copresentation containers
only. It does **not** own:

- finite-poset or fringe-module kernels (`TamerOp.FiniteFringe`),
- indicator-resolution algorithms (`TamerOp.IndicatorResolutions`),
- Ext/Tor assembly and derived-functor computations (`TamerOp.DerivedFunctors`),
- encoding or polyhedral translation layers (`TamerOp.Encoding`,
  `TamerOp.EncodingCore`).

In particular, there are no hidden encode-vs-pmodule bridges here: these types
are algebraic containers with explicit labels and matrices.
"""
module IndicatorTypes

using SparseArrays
import Base: show

using ..FiniteFringe: AbstractPoset, Upset, Downset, FringeModule,
                      check_poset, check_upset, check_downset, check_fringe_module,
                      nvertices
import ..FiniteFringe: ambient_poset, base_poset, field
using ..CoreModules: AbstractCoeffField, QQField, RealField, PrimeField,
                     coeff_type, field_from_eltype

@inline function _coerce_indicator_matrix(::Type{K}, A::SparseMatrixCSC) where {K}
    return SparseMatrixCSC{K,Int}(A)
end

@inline function _coerce_indicator_matrix(::Type{K}, A::AbstractMatrix) where {K}
    return Matrix{K}(A)
end

function _coerce_upset_labels(P::PT, U::AbstractVector{<:Upset}) where {PT<:AbstractPoset}
    out = Vector{Upset{PT}}(undef, length(U))
    @inbounds for i in eachindex(U)
        Ui = U[i]
        Ui.P === P || error("UpsetPresentation: upset label poset must match the presentation poset.")
        out[i] = Ui
    end
    return out
end

function _coerce_downset_labels(P::PT, D::AbstractVector{<:Downset}) where {PT<:AbstractPoset}
    out = Vector{Downset{PT}}(undef, length(D))
    @inbounds for i in eachindex(D)
        Di = D[i]
        Di.P === P || error("DownsetCopresentation: downset label poset must match the copresentation poset.")
        out[i] = Di
    end
    return out
end

@inline function _check_indicator_fringe(::Type{K}, H) where {K}
    H === nothing && return nothing
    H isa FringeModule{K} || error("indicator presentation fringe module must have coefficient type $K")
    return H
end

@inline _indicator_field_label(::QQField) = "QQ"
@inline _indicator_field_label(field::PrimeField) = "Fp($(field.p))"
@inline _indicator_field_label(field::RealField{T}) where {T<:AbstractFloat} = "RealField($(T))"
@inline _indicator_field_label(field::AbstractCoeffField) = string(nameof(typeof(field)))

@inline _indicator_field(::Type{K}, H) where {K} = H === nothing ? field_from_eltype(K) : H.field

"""
    UpsetPresentation{K}

One-step upset presentation `F_1 --delta--> F_0 -> M` over a finite poset.

# Mathematical meaning

`UpsetPresentation{K}` records a presentation of a module by direct sums of
principal upset summands. The direct-summand labels live in the ambient poset
`P`, and the matrix `delta` represents the presentation map from the relation
object `F_1` to the generator object `F_0`.

# Stored data

- `P`: the ambient finite poset.
- `U0`: labels for the summands of `F_0`; these are the generator labels.
- `U1`: labels for the summands of `F_1`; these are the relation labels.
- `delta`: the matrix of `F_1 -> F_0` with
  `size(delta) == (length(U1), length(U0))`.
- `H`: an attached source fringe module when available, or `nothing`.

# Invariants

- `size(delta, 1) == length(U1)`.
- `size(delta, 2) == length(U0)`.
- all labels belong to the ambient poset `P`.
- if `H !== nothing`, it should be a fringe module over the same ambient poset
  and coefficient type.

# Notes

- `UpsetPresentation{K}` is a container type. It does not perform hidden
  conversions to pmodules or encodings.
- Use [`generator_labels`](@ref), [`relation_labels`](@ref),
  [`presentation_matrix`](@ref), [`attached_fringe`](@ref), and `describe(...)`
  instead of field access in ordinary user code.
"""
struct UpsetPresentation{K,PT<:AbstractPoset,H,MAT<:AbstractMatrix{K}}
    P::PT
    U0::Vector{Upset{PT}}
    U1::Vector{Upset{PT}}
    delta::MAT
    H::H

    function UpsetPresentation{K,PT,H,MAT}(P::PT,
                                           U0::Vector{Upset{PT}},
                                           U1::Vector{Upset{PT}},
                                           delta::MAT,
                                           Hobj::H) where {K,PT<:AbstractPoset,H,MAT<:AbstractMatrix{K}}
        size(delta, 1) == length(U1) || error("UpsetPresentation: row count must equal length(U1).")
        size(delta, 2) == length(U0) || error("UpsetPresentation: column count must equal length(U0).")
        return new{K,PT,H,MAT}(P, U0, U1, delta, Hobj)
    end
end

"""
    DownsetCopresentation{K}

One-step downset copresentation `M -> E^0 --rho--> E^1` over a finite poset.

# Mathematical meaning

`DownsetCopresentation{K}` records a copresentation of a module by direct sums
of principal downset summands. The matrix `rho` represents the copresentation
map from the cogenerator object `E^0` to the corelation object `E^1`.

# Stored data

- `P`: the ambient finite poset.
- `D0`: labels for the summands of `E^0`; these are the cogenerator labels.
- `D1`: labels for the summands of `E^1`; these are the corelation labels.
- `rho`: the matrix of `E^0 -> E^1` with
  `size(rho) == (length(D1), length(D0))`.
- `H`: an attached target fringe module when available, or `nothing`.

# Invariants

- `size(rho, 1) == length(D1)`.
- `size(rho, 2) == length(D0)`.
- all labels belong to the ambient poset `P`.
- if `H !== nothing`, it should be a fringe module over the same ambient poset
  and coefficient type.

# Notes

- `DownsetCopresentation{K}` is a container type. It does not perform hidden
  conversions to pmodules or encodings.
- Use [`cogenerator_labels`](@ref), [`corelation_labels`](@ref),
  [`copresentation_matrix`](@ref), [`attached_fringe`](@ref), and
  `describe(...)` instead of field access in ordinary user code.
"""
struct DownsetCopresentation{K,PT<:AbstractPoset,H,MAT<:AbstractMatrix{K}}
    P::PT
    D0::Vector{Downset{PT}}
    D1::Vector{Downset{PT}}
    rho::MAT
    H::H

    function DownsetCopresentation{K,PT,H,MAT}(P::PT,
                                               D0::Vector{Downset{PT}},
                                               D1::Vector{Downset{PT}},
                                               rho::MAT,
                                               Hobj::H) where {K,PT<:AbstractPoset,H,MAT<:AbstractMatrix{K}}
        size(rho, 1) == length(D1) || error("DownsetCopresentation: row count must equal length(D1).")
        size(rho, 2) == length(D0) || error("DownsetCopresentation: column count must equal length(D0).")
        return new{K,PT,H,MAT}(P, D0, D1, rho, Hobj)
    end
end

"""
    UpsetPresentation(P, U0, U1, delta, H; field) -> UpsetPresentation
    UpsetPresentation(P, U0, U1, delta, H) -> UpsetPresentation
    UpsetPresentation{K}(P, U0, U1, delta, H) -> UpsetPresentation{K}

Construct a one-step upset presentation over the ambient poset `P`.

# Inputs

- `P::AbstractPoset`: ambient finite poset.
- `U0`: generator labels, one upset per column of `delta`.
- `U1`: relation labels, one upset per row of `delta`.
- `delta`: coefficient matrix of the map `F_1 -> F_0`.
- `H`: attached fringe module or `nothing`.
- `field`: optional coefficient field when `delta` should be coerced.

# Output

- An [`UpsetPresentation`](@ref) with matrix orientation
  `size(delta) == (length(U1), length(U0))`.

# Contract behavior

- `U0` and `U1` are expected to contain labels already attached to `P`.
- The constructor does **not** close generator labels for you.
- There are no legacy aliases and no hidden encode-vs-pmodule bridges here.
- `field=...` is only for coefficient coercion; use `UpsetPresentation{K}(...)`
  when you want the coefficient type to be explicit in the call signature.

# Best practices

- Use [`generator_labels`](@ref), [`relation_labels`](@ref), and
  [`presentation_matrix`](@ref) after construction.
- Use [`check_upset_presentation`](@ref) when you hand-build one of these
  objects and want a structured validation report.

# Examples

```jldoctest
julia> using TamerOp

julia> const FF = TamerOp.FiniteFringe;

julia> const IT = TamerOp.IndicatorTypes;

julia> P = FF.FinitePoset(Bool[1 1; 0 1]);

julia> U0 = FF.principal_upset(P, 1);

julia> U1 = FF.principal_upset(P, 2);

julia> F = IT.UpsetPresentation{QQ}(P, [U0], [U1], reshape(QQ[1], 1, 1), nothing);

julia> describe(F).ngenerators
1

julia> IT.check_upset_presentation(F).valid
true
```
"""
function UpsetPresentation(
    P::PT,
    U0::AbstractVector{<:Upset},
    U1::AbstractVector{<:Upset},
    delta::AbstractMatrix,
    H;
    field::Union{Nothing,AbstractCoeffField}=nothing,
) where {PT<:AbstractPoset}
    K = field === nothing ? eltype(delta) : coeff_type(field)
    return IndicatorTypes.UpsetPresentation{K}(P, U0, U1, delta, H)
end

function UpsetPresentation{K}(
    P::PT,
    U0::AbstractVector{<:Upset},
    U1::AbstractVector{<:Upset},
    delta::AbstractMatrix,
    H,
) where {K,PT<:AbstractPoset}
    U0c = _coerce_upset_labels(P, U0)
    U1c = _coerce_upset_labels(P, U1)
    deltaK = _coerce_indicator_matrix(K, delta)
    Hc = _check_indicator_fringe(K, H)
    return IndicatorTypes.UpsetPresentation{K,PT,typeof(Hc),typeof(deltaK)}(P, U0c, U1c, deltaK, Hc)
end

"""
    DownsetCopresentation(P, D0, D1, rho, H; field) -> DownsetCopresentation
    DownsetCopresentation(P, D0, D1, rho, H) -> DownsetCopresentation
    DownsetCopresentation{K}(P, D0, D1, rho, H) -> DownsetCopresentation{K}

Construct a one-step downset copresentation over the ambient poset `P`.

# Inputs

- `P::AbstractPoset`: ambient finite poset.
- `D0`: cogenerator labels, one downset per column of `rho`.
- `D1`: corelation labels, one downset per row of `rho`.
- `rho`: coefficient matrix of the map `E^0 -> E^1`.
- `H`: attached fringe module or `nothing`.
- `field`: optional coefficient field when `rho` should be coerced.

# Output

- A [`DownsetCopresentation`](@ref) with matrix orientation
  `size(rho) == (length(D1), length(D0))`.

# Contract behavior

- `D0` and `D1` are expected to contain labels already attached to `P`.
- The constructor does **not** close generator labels for you.
- There are no legacy aliases and no hidden encode-vs-pmodule bridges here.
- `field=...` is only for coefficient coercion; use
  `DownsetCopresentation{K}(...)` when you want the coefficient type to be
  explicit in the call signature.

# Best practices

- Use [`cogenerator_labels`](@ref), [`corelation_labels`](@ref), and
  [`copresentation_matrix`](@ref) after construction.
- Use [`check_downset_copresentation`](@ref) when you hand-build one of these
  objects and want a structured validation report.

# Examples

```jldoctest
julia> using TamerOp

julia> const FF = TamerOp.FiniteFringe;

julia> const IT = TamerOp.IndicatorTypes;

julia> P = FF.FinitePoset(Bool[1 1; 0 1]);

julia> D0 = FF.principal_downset(P, 1);

julia> D1 = FF.principal_downset(P, 2);

julia> E = IT.DownsetCopresentation{QQ}(P, [D0], [D1], reshape(QQ[1], 1, 1), nothing);

julia> describe(E).ncogenerators
1

julia> IT.check_downset_copresentation(E).valid
true
```
"""
function DownsetCopresentation(
    P::PT,
    D0::AbstractVector{<:Downset},
    D1::AbstractVector{<:Downset},
    rho::AbstractMatrix,
    H;
    field::Union{Nothing,AbstractCoeffField}=nothing,
) where {PT<:AbstractPoset}
    K = field === nothing ? eltype(rho) : coeff_type(field)
    return IndicatorTypes.DownsetCopresentation{K}(P, D0, D1, rho, H)
end

function DownsetCopresentation{K}(
    P::PT,
    D0::AbstractVector{<:Downset},
    D1::AbstractVector{<:Downset},
    rho::AbstractMatrix,
    H,
) where {K,PT<:AbstractPoset}
    D0c = _coerce_downset_labels(P, D0)
    D1c = _coerce_downset_labels(P, D1)
    rhoK = _coerce_indicator_matrix(K, rho)
    Hc = _check_indicator_fringe(K, H)
    return IndicatorTypes.DownsetCopresentation{K,PT,typeof(Hc),typeof(rhoK)}(P, D0c, D1c, rhoK, Hc)
end

"""
    ambient_poset(X::Union{UpsetPresentation,DownsetCopresentation}) -> AbstractPoset
    base_poset(X::Union{UpsetPresentation,DownsetCopresentation}) -> AbstractPoset

Return the ambient finite poset of an indicator presentation or copresentation.
"""
@inline ambient_poset(F::UpsetPresentation) = F.P
@inline ambient_poset(E::DownsetCopresentation) = E.P
@inline base_poset(F::UpsetPresentation) = F.P
@inline base_poset(E::DownsetCopresentation) = E.P

"""
    field(X::Union{UpsetPresentation,DownsetCopresentation}) -> AbstractCoeffField

Return the coefficient field of an indicator presentation or copresentation.

If an attached fringe module is present, its field is returned. Otherwise the
field is reconstructed from the coefficient element type.
"""
@inline field(F::UpsetPresentation{K}) where {K} = _indicator_field(K, F.H)
@inline field(E::DownsetCopresentation{K}) where {K} = _indicator_field(K, E.H)

"""
    generator_labels(F::UpsetPresentation) -> Vector{Upset}
    relation_labels(F::UpsetPresentation) -> Vector{Upset}

Return the generator and relation labels of an upset presentation.
"""
@inline generator_labels(F::UpsetPresentation) = F.U0
@inline relation_labels(F::UpsetPresentation) = F.U1

"""
    cogenerator_labels(E::DownsetCopresentation) -> Vector{Downset}
    corelation_labels(E::DownsetCopresentation) -> Vector{Downset}

Return the cogenerator and corelation labels of a downset copresentation.

`cogenerator_labels(E)` are the labels on `E^0`, which are also the natural
death labels of the copresentation.
"""
@inline cogenerator_labels(E::DownsetCopresentation) = E.D0
@inline corelation_labels(E::DownsetCopresentation) = E.D1

"""
    presentation_matrix(F::UpsetPresentation) -> AbstractMatrix
    copresentation_matrix(E::DownsetCopresentation) -> AbstractMatrix

Return the coefficient matrix of an indicator presentation or copresentation.
"""
@inline presentation_matrix(F::UpsetPresentation) = F.delta
@inline copresentation_matrix(E::DownsetCopresentation) = E.rho

"""
    attached_fringe(X::Union{UpsetPresentation,DownsetCopresentation}) -> Union{FringeModule,Nothing}

Return the attached fringe module when one is stored on the container, or
`nothing` otherwise.
"""
@inline attached_fringe(F::UpsetPresentation) = F.H
@inline attached_fringe(E::DownsetCopresentation) = E.H

@inline function _indicator_describe(F::UpsetPresentation)
    return (kind=:upset_presentation,
            field=field(F),
            poset_kind=Symbol(nameof(typeof(F.P))),
            nvertices=nvertices(F.P),
            ngenerators=length(F.U0),
            nrelations=length(F.U1),
            matrix_size=size(F.delta),
            has_attached_fringe=F.H !== nothing)
end

@inline function _indicator_describe(E::DownsetCopresentation)
    return (kind=:downset_copresentation,
            field=field(E),
            poset_kind=Symbol(nameof(typeof(E.P))),
            nvertices=nvertices(E.P),
            ncogenerators=length(E.D0),
            ncorelations=length(E.D1),
            matrix_size=size(E.rho),
            has_attached_fringe=E.H !== nothing)
end

"""
    check_upset_presentation(F; throw=false) -> NamedTuple

Validate a hand-built [`UpsetPresentation`](@ref).

The returned report checks:

- the ambient poset is itself valid,
- the matrix shape matches the label lengths,
- all labels belong to the ambient poset,
- each label is a valid upset,
- the attached fringe module, when present, matches the ambient poset and
  coefficient type.

Set `throw=true` to raise an `ArgumentError` instead of returning an invalid
report.

# Examples

```jldoctest
julia> using TamerOp

julia> const FF = TamerOp.FiniteFringe;

julia> const IT = TamerOp.IndicatorTypes;

julia> P = FF.FinitePoset(Bool[1 1; 0 1]);

julia> F = IT.UpsetPresentation{QQ}(P,
                                   [FF.principal_upset(P, 1)],
                                   [FF.principal_upset(P, 2)],
                                   reshape(QQ[1], 1, 1),
                                   nothing);

julia> IT.check_upset_presentation(F).valid
true
```
"""
function check_upset_presentation(F::UpsetPresentation{K}; throw::Bool=false) where {K}
    issues = String[]
    P = F.P
    poset_report = check_poset(P)
    poset_report.valid || append!(issues, "ambient poset: " .* poset_report.issues)

    size(F.delta, 1) == length(F.U1) || push!(issues, "matrix row count $(size(F.delta, 1)) must equal number of relation labels $(length(F.U1))")
    size(F.delta, 2) == length(F.U0) || push!(issues, "matrix column count $(size(F.delta, 2)) must equal number of generator labels $(length(F.U0))")

    @inbounds for (i, U) in enumerate(F.U0)
        U.P === P || push!(issues, "generator label $i does not belong to the ambient poset")
        rep = check_upset(U)
        rep.valid || append!(issues, "generator label $i: " .* rep.issues)
    end
    @inbounds for (i, U) in enumerate(F.U1)
        U.P === P || push!(issues, "relation label $i does not belong to the ambient poset")
        rep = check_upset(U)
        rep.valid || append!(issues, "relation label $i: " .* rep.issues)
    end

    H = F.H
    if H !== nothing
        H isa FringeModule || push!(issues, "attached fringe must be a FringeModule or nothing")
        if H isa FringeModule
            H.P === P || push!(issues, "attached fringe module must belong to the ambient poset")
            coeff_type(field(H)) == K || push!(issues, "attached fringe coefficient field must match the presentation coefficient type")
            rep = check_fringe_module(H)
            rep.valid || append!(issues, "attached fringe: " .* rep.issues)
        end
    end

    report = (kind=:upset_presentation,
              valid=isempty(issues),
              field=field(F),
              nvertices=nvertices(P),
              ngenerators=length(F.U0),
              nrelations=length(F.U1),
              matrix_size=size(F.delta),
              has_attached_fringe=H !== nothing,
              issues=issues)
    if throw && !report.valid
        Base.throw(ArgumentError("check_upset_presentation: invalid upset presentation: " * join(report.issues, "; ")))
    end
    return report
end

"""
    check_downset_copresentation(E; throw=false) -> NamedTuple

Validate a hand-built [`DownsetCopresentation`](@ref).

The returned report checks:

- the ambient poset is itself valid,
- the matrix shape matches the label lengths,
- all labels belong to the ambient poset,
- each label is a valid downset,
- the attached fringe module, when present, matches the ambient poset and
  coefficient type.

Set `throw=true` to raise an `ArgumentError` instead of returning an invalid
report.

# Examples

```jldoctest
julia> using TamerOp

julia> const FF = TamerOp.FiniteFringe;

julia> const IT = TamerOp.IndicatorTypes;

julia> P = FF.FinitePoset(Bool[1 1; 0 1]);

julia> E = IT.DownsetCopresentation{QQ}(P,
                                       [FF.principal_downset(P, 1)],
                                       [FF.principal_downset(P, 2)],
                                       reshape(QQ[1], 1, 1),
                                       nothing);

julia> IT.check_downset_copresentation(E).valid
true
```
"""
function check_downset_copresentation(E::DownsetCopresentation{K}; throw::Bool=false) where {K}
    issues = String[]
    P = E.P
    poset_report = check_poset(P)
    poset_report.valid || append!(issues, "ambient poset: " .* poset_report.issues)

    size(E.rho, 1) == length(E.D1) || push!(issues, "matrix row count $(size(E.rho, 1)) must equal number of corelation labels $(length(E.D1))")
    size(E.rho, 2) == length(E.D0) || push!(issues, "matrix column count $(size(E.rho, 2)) must equal number of cogenerator labels $(length(E.D0))")

    @inbounds for (i, D) in enumerate(E.D0)
        D.P === P || push!(issues, "cogenerator label $i does not belong to the ambient poset")
        rep = check_downset(D)
        rep.valid || append!(issues, "cogenerator label $i: " .* rep.issues)
    end
    @inbounds for (i, D) in enumerate(E.D1)
        D.P === P || push!(issues, "corelation label $i does not belong to the ambient poset")
        rep = check_downset(D)
        rep.valid || append!(issues, "corelation label $i: " .* rep.issues)
    end

    H = E.H
    if H !== nothing
        H isa FringeModule || push!(issues, "attached fringe must be a FringeModule or nothing")
        if H isa FringeModule
            H.P === P || push!(issues, "attached fringe module must belong to the ambient poset")
            coeff_type(field(H)) == K || push!(issues, "attached fringe coefficient field must match the copresentation coefficient type")
            rep = check_fringe_module(H)
            rep.valid || append!(issues, "attached fringe: " .* rep.issues)
        end
    end

    report = (kind=:downset_copresentation,
              valid=isempty(issues),
              field=field(E),
              nvertices=nvertices(P),
              ncogenerators=length(E.D0),
              ncorelations=length(E.D1),
              matrix_size=size(E.rho),
              has_attached_fringe=H !== nothing,
              issues=issues)
    if throw && !report.valid
        Base.throw(ArgumentError("check_downset_copresentation: invalid downset copresentation: " * join(report.issues, "; ")))
    end
    return report
end

function show(io::IO, F::UpsetPresentation)
    d = _indicator_describe(F)
    print(io, "UpsetPresentation(field=", _indicator_field_label(d.field),
          ", nvertices=", d.nvertices,
          ", ngenerators=", d.ngenerators,
          ", nrelations=", d.nrelations, ")")
end

function show(io::IO, ::MIME"text/plain", F::UpsetPresentation)
    d = _indicator_describe(F)
    print(io, "UpsetPresentation\n",
          "  field: ", _indicator_field_label(d.field), "\n",
          "  nvertices: ", d.nvertices, "\n",
          "  ngenerators: ", d.ngenerators, "\n",
          "  nrelations: ", d.nrelations, "\n",
          "  matrix_size: ", d.matrix_size, "\n",
          "  attached_fringe: ", d.has_attached_fringe)
end

function show(io::IO, E::DownsetCopresentation)
    d = _indicator_describe(E)
    print(io, "DownsetCopresentation(field=", _indicator_field_label(d.field),
          ", nvertices=", d.nvertices,
          ", ncogenerators=", d.ncogenerators,
          ", ncorelations=", d.ncorelations, ")")
end

function show(io::IO, ::MIME"text/plain", E::DownsetCopresentation)
    d = _indicator_describe(E)
    print(io, "DownsetCopresentation\n",
          "  field: ", _indicator_field_label(d.field), "\n",
          "  nvertices: ", d.nvertices, "\n",
          "  ncogenerators: ", d.ncogenerators, "\n",
          "  ncorelations: ", d.ncorelations, "\n",
          "  matrix_size: ", d.matrix_size, "\n",
          "  attached_fringe: ", d.has_attached_fringe)
end

end # module IndicatorTypes
