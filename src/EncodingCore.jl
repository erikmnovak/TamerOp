"""
    EncodingCore

Shared encoding-map abstractions, compiled encoding wrappers, and query-shape
validation for the finite-encoding layer.

Ownership map
- `AbstractPLikeEncodingMap` defines the shared contract for encoding-map
  objects that support `dimension`, `locate`, and batched query classification.
- [`CompiledEncoding`](@ref) owns the inspectable wrapper that attaches a
  backend encoding map to its finite encoding poset.
- [`GridEncodingMap`](@ref) owns the lightweight axis-aligned grid encoding
  implementation used in tests, examples, and backend-adjacent workflows.
- `check_encoding_map`, `check_compiled_encoding`, `check_query_point`, and
  `check_query_matrix` own the canonical notebook-friendly validation surface
  for encoding objects and query shapes.
- [`encoding_summary`](@ref) and the semantic accessors
  `encoding_poset`/`encoding_map`/`encoding_axes`/`encoding_representatives`
  own cheap inspection of compiled/query-facing encoding objects.

What this module does not own
- finite encoding records such as `EncodingMap`, `UptightEncoding`, and
  postcomposition helpers; those belong to `Encoding`,
- backend-specific finite encodings such as `ZnEncoding` and PL encodings,
- geometry ownership for polyhedral or region-level exact queries; those belong
  to `PLBackend` and `PLPolyhedra`,
- workflow orchestration and invariant algorithms; those belong to `Results`,
  `Workflow`, and the invariant-family owners.

Contributor note
- put shared query-shape contracts, compiled-wrapper semantics, and lightweight
  encoding inspection here,
- keep finite-poset encoding semantics in `Encoding`,
- keep backend-specific region geometry and exact-membership machinery in
  `ZnEncoding`, `PLBackend`, or `PLPolyhedra`,
- keep workflow result wrappers and user-facing orchestration out of this owner.
"""
module EncodingCore

using ..CoreModules: EncodingCache, SessionCache, _encoding_cache!, _WORKFLOW_ENCODING_CACHE_KEY

"""
    AbstractPLikeEncodingMap

Abstract supertype for encoding-map objects used throughout the finite-encoding
and PL backends.
"""
abstract type AbstractPLikeEncodingMap end

const PLikeEncodingMap = AbstractPLikeEncodingMap

function check_encoding_map end
function check_compiled_encoding end
function check_query_point end
function check_query_matrix end
function encoding_summary end
function encoding_poset end
function encoding_map end
function encoding_axes end
function encoding_representatives end

"""
    EncodingValidationSummary

Display wrapper for validation reports returned by the `EncodingCore`
`check_*` helpers.

The canonical encoding validators still return structured `NamedTuple`s for
easy programmatic inspection. Wrap one of those reports with
[`encoding_validation_summary`](@ref) when you want a compact notebook/REPL
presentation of the same information.
"""
struct EncodingValidationSummary{R}
    report::R
end

"""
    encoding_validation_summary(report) -> EncodingValidationSummary

Wrap an encoding validation report in a display-oriented object.

Use this when you want the report returned by
[`check_encoding_map`](@ref), [`check_compiled_encoding`](@ref),
[`check_query_point`](@ref), or [`check_query_matrix`](@ref) to print as a
compact mathematical summary with issues laid out line by line.
"""
@inline encoding_validation_summary(report::NamedTuple) = EncodingValidationSummary(report)

@inline _encoding_issue_report(kind::Symbol, valid::Bool; kwargs...) =
    (; kind, valid, kwargs...)

function _throw_invalid_encoding(kind::Symbol, issues::Vector{String})
    throw(ArgumentError(string(kind, ": ", isempty(issues) ? "invalid encoding object." : join(issues, " "))))
end

@inline _representative_dim(x::NTuple{N,<:Any}) where {N} = N
@inline _representative_dim(x::AbstractVector) = length(x)
@inline _representative_dim(::Any) = nothing

function _check_axes_contract(axes, dim::Int, issues::Vector{String})
    axes === nothing && return nothing
    axes isa Tuple || push!(issues, "axes must be a tuple of coordinate axes.")
    axes isa Tuple || return nothing
    length(axes) == dim || push!(issues, "axes tuple length $(length(axes)) does not match encoding dimension $dim.")
    for (i, axis) in pairs(axes)
        axis isa AbstractVector || push!(issues, "axis $i must be an abstract vector of coordinates.")
    end
    return nothing
end

function _check_representative_contract(reps, dim::Int, issues::Vector{String})
    reps === nothing && return nothing
    reps isa AbstractVector || reps isa Tuple || push!(issues, "representatives must be a finite tuple/vector of sample points.")
    (reps isa AbstractVector || reps isa Tuple) || return nothing
    for (i, rep) in pairs(reps)
        rdim = _representative_dim(rep)
        rdim === nothing && begin
            push!(issues, "representative $i does not expose a coordinate dimension.")
            continue
        end
        rdim == dim || push!(issues, "representative $i has dimension $rdim, expected $dim.")
    end
    return nothing
end

function _encoding_describe(pi::AbstractPLikeEncodingMap)
    axes = encoding_axes(pi)
    reps = encoding_representatives(pi)
    return (;
        kind=:encoding_map,
        map_type=typeof(pi),
        parameter_dim=dimension(pi),
        has_axes=axes !== nothing,
        has_representatives=reps !== nothing,
    )
end

"""
    encoding_summary(enc) -> NamedTuple

Return a compact semantic summary of an encoding object.

This is the owner-module inspection entrypoint for [`CompiledEncoding`](@ref)
and encoding-map objects such as [`GridEncodingMap`](@ref). It mirrors the
shared `describe(...)` surface without requiring users to know that the generic
is owned elsewhere.

Best practices
- use `encoding_summary(...)` when you are already working inside
  `EncodingCore` or `TamerOp.Advanced`;
- use the semantic accessors [`encoding_poset`](@ref), [`encoding_map`](@ref),
  [`encoding_axes`](@ref), and [`encoding_representatives`](@ref) for specific
  payloads;
- keep `check_encoding_map(...)` and `check_compiled_encoding(...)` for
  contract validation.
"""
encoding_summary(pi::AbstractPLikeEncodingMap) = _encoding_describe(pi)

"""
    CompiledEncoding(P, pi; axes=nothing, reps=nothing, meta=NamedTuple())

Primary wrapper for an encoding map together with its finite encoding poset.

`CompiledEncoding` is the canonical object to pass around once an encoding map
`pi` has been attached to the finite poset `P` it indexes. It stores optional
axes and representative points so high-level code can inspect the encoding
without reaching into backend-specific fields.

Mathematical/contract meaning
- `P` is the finite encoding poset whose region labels are returned by
  `locate(pi, x)`,
- `pi` is the backend encoding map that classifies query points,
- `axes` and `reps` are optional semantic metadata for inspection and plotting,
- `meta` is backend/cache payload and should be treated as an implementation
  detail rather than ordinary mathematical data.

Query contract
- `locate(enc, x)` forwards to the wrapped encoding map,
- `locate_many(enc, X)` and `locate_many!(dest, enc, X)` forward likewise,
- the default point contract is one real coordinate vector per query, with
  outside queries conventionally returning `0`.

Best practice:
- construct with [`compile_encoding`](@ref) instead of calling the struct
  constructor directly;
- inspect with [`encoding_summary`](@ref) or `describe(enc)`,
  [`encoding_poset`](@ref),
  [`encoding_map`](@ref), [`encoding_axes`](@ref), and
  [`encoding_representatives`](@ref);
- validate hand-built objects with [`check_compiled_encoding`](@ref) before
  using them in workflow code.

This owner does not decide what the underlying finite encoding is; that belongs
to `Encoding`, `ZnEncoding`, `PLBackend`, or `PLPolyhedra`. `CompiledEncoding`
only standardizes the inspectable/query-facing wrapper surface.
"""
struct CompiledEncoding{PiType,PType,AxesType,RepsType,MetaType} <: AbstractPLikeEncodingMap
    P::PType
    pi::PiType
    axes::AxesType
    reps::RepsType
    meta::MetaType
end

"""
    compile_encoding(P, pi; axes=nothing, reps=nothing, meta=NamedTuple()) -> CompiledEncoding

Attach an encoding map `pi` to the finite encoding poset `P`.

The returned [`CompiledEncoding`](@ref) is the canonical advanced-user wrapper
for encoding maps. It records:
- the finite encoding poset `P`,
- the backend map object `pi`,
- optional coordinate axes,
- optional representative points,
- backend metadata/cache payload in `meta`.

Defaults:
- `axes=nothing` means "ask the encoding map whether axes are available";
- `reps=nothing` means "ask the encoding map whether representatives are available";
- `meta=NamedTuple()` keeps the wrapper lightweight unless a backend/cache
  layer explicitly needs metadata.

Use this when you want an inspectable encoding object. Simple workflow code
should usually rely on higher-level `encode(...)` entrypoints instead. For
inspection inside `EncodingCore`, prefer [`encoding_summary`](@ref) over direct
field access.
"""
function compile_encoding(P, pi; axes=nothing, reps=nothing, meta=NamedTuple())
    axes_val = axes === nothing ? _axes_or_nothing(pi) : axes
    reps_val = reps === nothing ? _reps_or_nothing(pi) : reps
    return CompiledEncoding(P, pi, axes_val, reps_val, meta)
end

compile_encoding(::Any, ::Nothing; kwargs...) = nothing

function Base.getproperty(enc::CompiledEncoding, name::Symbol)
    if name === :P || name === :pi || name === :axes || name === :reps || name === :meta
        return getfield(enc, name)
    end
    return getproperty(getfield(enc, :pi), name)
end

function Base.propertynames(enc::CompiledEncoding, private::Bool=false)
    return (fieldnames(typeof(enc))..., propertynames(enc.pi, private)...)
end

function Base.show(io::IO, enc::CompiledEncoding)
    d = _encoding_describe(enc)
    print(io, "CompiledEncoding(map=", nameof(d.map_type),
          ", parameter_dim=", d.parameter_dim,
          ", has_axes=", d.has_axes,
          ", has_representatives=", d.has_representatives, ")")
end

function Base.show(io::IO, ::MIME"text/plain", enc::CompiledEncoding)
    d = _encoding_describe(enc)
    print(io, "CompiledEncoding",
          "\n  poset_type: ", d.poset_type,
          "\n  map_type: ", d.map_type,
          "\n  parameter_dim: ", d.parameter_dim,
          "\n  has_axes: ", d.has_axes,
          "\n  has_representatives: ", d.has_representatives,
          "\n  meta_type: ", d.meta_type)
end

"""
    locate(pi, x)

Classify a query point `x` into the region index of an encoding map.

Public contract:
- `x` is normally an `AbstractVector{<:Real}` of length equal to
  [`dimension(pi)`](@ref);
- tuple queries are only supported when an encoding-map type explicitly defines
  them;
- outside the encoding domain, the conventional return value is `0`.

For user-facing code, prefer vector queries unless the owner encoding-map type
documents tuple support explicitly. Use [`check_query_point`](@ref) when you
want to validate query shape before calling `locate`.
"""
function locate end
"""
    locate_many!(dest, pi, X; kwargs...) -> dest

Classify many query points at once.

`X` is expected to store one query per column, so `size(X, 1)` is the ambient
 parameter dimension and `size(X, 2)` is the number of queries. `dest` must
 have length `size(X, 2)`.

Use `locate_many!` for repeated classification work; it is the canonical
 batched contract for backend implementations. For a fresh output vector, use
 [`locate_many`](@ref). Use [`check_query_matrix`](@ref) when you want to
 validate a batched query shape before calling `locate_many!`.
"""
function locate_many! end
"""
    locate_many(pi, X; kwargs...) -> Vector{Int}

Allocate an output vector and classify all query columns of `X`.

This is a convenience wrapper around [`locate_many!`](@ref). Advanced users who
 care about allocations should prefer `locate_many!`.
"""
function locate_many end

const _ENCODINGCORE_CACHED_LOCATE_STYLE = Ref(true)
const _LOCATE_STYLE_LOCK = Base.ReentrantLock()
const _LOCATE_STYLE_CACHE = Dict{Tuple{DataType,Bool,Bool,Bool},UInt8}()

const _LOCATE_STYLE_PLAIN = UInt8(0)
const _LOCATE_STYLE_STRICT_ONLY = UInt8(1)
const _LOCATE_STYLE_STRICT_CLOSURE = UInt8(2)

@inline _geometry_fingerprint(pi) = UInt(objectid(pi))
@inline _geometry_fingerprint(enc::CompiledEncoding) = _geometry_fingerprint(enc.pi)
@inline _locate_style_owner(pi) = typeof(pi)
@inline _locate_style_owner(enc::CompiledEncoding) = _locate_style_owner(enc.pi)

"""
    check_query_point(pi, x; throw=false) -> NamedTuple

Validate the shape of a single query point against the default public
`EncodingCore` locate contract.

The canonical public contract expects `x` to be an `AbstractVector{<:Real}` of
length equal to [`dimension(pi)`](@ref). Tuple queries are intentionally not
treated as valid here, because tuple support is owner-specific and not part of
the default contract.

Use this helper when validating user input before calling [`locate`](@ref).
Wrap the returned report with [`encoding_validation_summary`](@ref) when you
want a readable notebook or REPL summary.
"""
function check_query_point(pi::AbstractPLikeEncodingMap, x; throw::Bool=false)
    issues = String[]
    dim = try
        dimension(pi)
    catch err
        push!(issues, "dimension(pi) failed: $(sprint(showerror, err))")
        0
    end
    dim = dim isa Integer ? Int(dim) : 0
    if !(x isa AbstractVector{<:Real})
        push!(issues, "query point must be an AbstractVector{<:Real} of length $dim.")
    elseif dim > 0 && length(x) != dim
        push!(issues, "query point length $(length(x)) does not match encoding dimension $dim.")
    end
    valid = isempty(issues)
    throw && !valid && _throw_invalid_encoding(:encoding_query_point, issues)
    return _encoding_issue_report(:encoding_query_point, valid;
                                  map_type=typeof(pi),
                                  parameter_dim=dim,
                                  query_type=typeof(x),
                                  query_length=(x isa AbstractVector ? length(x) : nothing),
                                  issues=issues)
end

check_query_point(enc::CompiledEncoding, x; throw::Bool=false) =
    check_query_point(enc.pi, x; throw=throw)

"""
    check_query_matrix(pi, X; throw=false) -> NamedTuple

Validate the shape of a batched query matrix against the default public
`EncodingCore` locate-many contract.

The canonical public contract expects `X` to be an `AbstractMatrix{<:Real}`
with one query per column, so `size(X, 1)` must equal
[`dimension(pi)`](@ref).

Use this helper when validating user input before calling
[`locate_many!`](@ref) or [`locate_many`](@ref). Wrap the returned report with
[`encoding_validation_summary`](@ref) when you want a readable notebook or REPL
summary.
"""
function check_query_matrix(pi::AbstractPLikeEncodingMap, X; throw::Bool=false)
    issues = String[]
    dim = try
        dimension(pi)
    catch err
        push!(issues, "dimension(pi) failed: $(sprint(showerror, err))")
        0
    end
    dim = dim isa Integer ? Int(dim) : 0
    if !(X isa AbstractMatrix{<:Real})
        push!(issues, "query matrix must be an AbstractMatrix{<:Real} with one query per column.")
    elseif dim > 0 && size(X, 1) != dim
        push!(issues, "query matrix row count $(size(X, 1)) does not match encoding dimension $dim.")
    end
    valid = isempty(issues)
    throw && !valid && _throw_invalid_encoding(:encoding_query_matrix, issues)
    return _encoding_issue_report(:encoding_query_matrix, valid;
                                  map_type=typeof(pi),
                                  parameter_dim=dim,
                                  matrix_type=typeof(X),
                                  size=(X isa AbstractMatrix ? size(X) : nothing),
                                  issues=issues)
end

check_query_matrix(enc::CompiledEncoding, X; throw::Bool=false) =
    check_query_matrix(enc.pi, X; throw=throw)

@inline function _locate_style_tag(style::UInt8)
    style == _LOCATE_STYLE_STRICT_CLOSURE && return Val(:strict_closure)
    style == _LOCATE_STYLE_STRICT_ONLY && return Val(:strict_only)
    return Val(:plain)
end

function _probe_locate_style(pi, x0::AbstractVector{<:Real}; strict::Bool, closure::Bool)
    try
        locate(pi, x0; strict=strict, closure=closure)
        return _LOCATE_STYLE_STRICT_CLOSURE
    catch err
        err isa MethodError || rethrow()
    end
    try
        locate(pi, x0; strict=strict)
        return _LOCATE_STYLE_STRICT_ONLY
    catch err
        err isa MethodError || rethrow()
    end
    locate(pi, x0)
    return _LOCATE_STYLE_PLAIN
end

function _probe_locate_many_style(pi, x0::AbstractVector{<:Real}; strict::Bool, closure::Bool)
    probe = Matrix{Float64}(undef, length(x0), 1)
    @inbounds for i in eachindex(x0)
        probe[i, 1] = float(x0[i])
    end
    dest = Vector{Int}(undef, 1)
    try
        locate_many!(dest, pi, probe; strict=strict, closure=closure)
        return _LOCATE_STYLE_STRICT_CLOSURE
    catch err
        err isa MethodError || rethrow()
    end
    try
        locate_many!(dest, pi, probe; strict=strict)
        return _LOCATE_STYLE_STRICT_ONLY
    catch err
        err isa MethodError || rethrow()
    end
    locate_many!(dest, pi, probe)
    return _LOCATE_STYLE_PLAIN
end

function _locate_call_style(pi, x0::AbstractVector{<:Real}; strict::Bool, closure::Bool, batched::Bool=false)
    if !_ENCODINGCORE_CACHED_LOCATE_STYLE[]
        style = batched ?
            _probe_locate_many_style(pi, x0; strict=strict, closure=closure) :
            _probe_locate_style(pi, x0; strict=strict, closure=closure)
        return _locate_style_tag(style)
    end
    key = (_locate_style_owner(pi), strict, closure, batched)
    Base.lock(_LOCATE_STYLE_LOCK)
    try
        cached = get(_LOCATE_STYLE_CACHE, key, typemin(UInt8))
        if cached != typemin(UInt8)
            return _locate_style_tag(cached)
        end
    finally
        Base.unlock(_LOCATE_STYLE_LOCK)
    end
    style = batched ?
        _probe_locate_many_style(pi, x0; strict=strict, closure=closure) :
        _probe_locate_style(pi, x0; strict=strict, closure=closure)
    Base.lock(_LOCATE_STYLE_LOCK)
    try
        _LOCATE_STYLE_CACHE[key] = style
    finally
        Base.unlock(_LOCATE_STYLE_LOCK)
    end
    return _locate_style_tag(style)
end

function _clear_locate_style_cache!()
    Base.lock(_LOCATE_STYLE_LOCK)
    try
        empty!(_LOCATE_STYLE_CACHE)
    finally
        Base.unlock(_LOCATE_STYLE_LOCK)
    end
    return nothing
end

function locate(pi::AbstractPLikeEncodingMap, x::NTuple{N,<:Real}; kwargs...) where {N}
    _ = kwargs
    throw(ArgumentError("locate($(typeof(pi)), tuple query) is not part of the default public contract. Pass a real vector of length $N, or define an explicit tuple overload for this encoding-map type."))
end

locate(enc::CompiledEncoding, x; kwargs...) = locate(enc.pi, x; kwargs...)
locate(enc::CompiledEncoding, x::NTuple{N,<:Real}; kwargs...) where {N} = locate(enc.pi, x; kwargs...)
locate(enc::CompiledEncoding, x::AbstractVector; kwargs...) = locate(enc.pi, x; kwargs...)

function locate_many!(dest::AbstractVector{<:Integer}, pi::AbstractPLikeEncodingMap,
                      X::AbstractMatrix{<:Real}; kwargs...)
    length(dest) == size(X, 2) || throw(ArgumentError("locate_many!: destination length $(length(dest)) must equal the number of query columns $(size(X, 2))."))
    @inbounds for j in 1:size(X, 2)
        dest[j] = locate(pi, view(X, :, j); kwargs...)
    end
    return dest
end

function locate_many(pi::AbstractPLikeEncodingMap, X::AbstractMatrix{<:Real}; kwargs...)
    out = Vector{Int}(undef, size(X, 2))
    return locate_many!(out, pi, X; kwargs...)
end

locate_many!(dest::AbstractVector{<:Integer}, enc::CompiledEncoding, X::AbstractMatrix{<:Real}; kwargs...) =
    locate_many!(dest, enc.pi, X; kwargs...)
locate_many!(dest::AbstractVector{<:Integer}, enc::CompiledEncoding, X::AbstractMatrix{<:AbstractFloat}; kwargs...) =
    locate_many!(dest, enc.pi, X; kwargs...)
locate_many(enc::CompiledEncoding, X::AbstractMatrix{<:Real}; kwargs...) =
    locate_many(enc.pi, X; kwargs...)
locate_many(enc::CompiledEncoding, X::AbstractMatrix{<:AbstractFloat}; kwargs...) =
    locate_many(enc.pi, X; kwargs...)

"""
    dimension(pi) -> Int

Return the ambient parameter-space dimension of an encoding map.
"""
function dimension end

dimension(enc::CompiledEncoding) = dimension(enc.pi)

"""
    representatives(pi)

Return representative points for the regions of `pi`, when available.

Representatives are optional metadata. Use them for inspection, plotting, and
 lightweight semantic summaries, not as a substitute for the actual encoding
 map.
"""
function representatives end

@inline representatives(::AbstractPLikeEncodingMap) = nothing
@inline _reps_or_nothing(::Any) = nothing
@inline _reps_or_nothing(pi::AbstractPLikeEncodingMap) = representatives(pi)

function representatives(enc::CompiledEncoding)
    enc.reps === nothing ? representatives(enc.pi) : enc.reps
end

"""
    axes_from_encoding(pi)

Return coordinate axes derived from the encoding, when available.

Axes are optional semantic metadata. When present, they are useful for
 inspection and plotting of grid/product-style encodings.
"""
function axes_from_encoding end

@inline axes_from_encoding(::AbstractPLikeEncodingMap) = nothing
@inline _axes_or_nothing(::Any) = nothing
@inline _axes_or_nothing(pi::AbstractPLikeEncodingMap) = axes_from_encoding(pi)

function axes_from_encoding(enc::CompiledEncoding)
    enc.axes === nothing ? axes_from_encoding(enc.pi) : enc.axes
end

@inline function _compile_encoding_without_reps(P, pi, meta)
    return CompiledEncoding(P, pi, _axes_or_nothing(pi), nothing, meta)
end

@inline function _compile_encoding_cached(P, pi, session_cache::Union{Nothing,SessionCache};
                                          include_reps::Bool=true)
    if session_cache === nothing
        ec = EncodingCache()
        return include_reps ? compile_encoding(P, pi; meta=ec) :
               _compile_encoding_without_reps(P, pi, ec)
    end
    ec = _encoding_cache!(session_cache, _WORKFLOW_ENCODING_CACHE_KEY)
    return include_reps ? compile_encoding(P, pi; meta=ec) :
           _compile_encoding_without_reps(P, pi, ec)
end

"""
    GridEncodingMap(P, coords; orientation=ntuple(_->1, N))

Axis-aligned grid encoding map for a product-style grid of coordinates.

`coords[i]` stores the sorted coordinates along axis `i`, and `orientation[i]`
 indicates whether the axis is used directly (`+1`) or with reversed sign
 (`-1`).

Contract
- `P` is the finite region poset indexed by the grid,
- `coords[i]` is the ordered axis used to classify coordinate `i`,
- `locate(pi, x)` expects a real vector of length `N` and returns either a
  1-based region label in `P` or `0` for an out-of-domain query,
- `locate_many!` expects one query per column.

Best practice:
- use sorted, duplicate-free coordinate vectors;
- inspect with `describe(pi)`, [`dimension`](@ref), and
  [`axes_from_encoding`](@ref);
- validate hand-built objects with [`check_encoding_map`](@ref) before wiring
  them into higher-level code.
"""
struct GridEncodingMap{N,T,P} <: AbstractPLikeEncodingMap
    P::P
    coords::NTuple{N,Vector{T}}
    orientation::NTuple{N,Int}
    sizes::NTuple{N,Int}
    strides::NTuple{N,Int}
end

function _grid_strides(sizes::NTuple{N,Int}) where {N}
    strides = Vector{Int}(undef, N)
    strides[1] = 1
    for i in 2:N
        strides[i] = strides[i - 1] * sizes[i - 1]
    end
    return ntuple(i -> strides[i], N)
end

function GridEncodingMap(P, coords::NTuple{N,Vector{T}};
                         orientation::NTuple{N,Int}=ntuple(_ -> 1, N)) where {N,T}
    sizes = ntuple(i -> length(coords[i]), N)
    for i in 1:N
        o = orientation[i]
        (o == 1 || o == -1) || throw(ArgumentError("GridEncodingMap: orientation[$i] must be +1 or -1."))
    end
    return GridEncodingMap{N,T,typeof(P)}(P, coords, orientation, sizes, _grid_strides(sizes))
end

dimension(pi::GridEncodingMap{N}) where {N} = N
axes_from_encoding(pi::GridEncodingMap) = pi.coords

"""
    encoding_poset(enc) -> P

Return the finite encoding poset attached to an encoding object when that notion
is available.

For [`CompiledEncoding`](@ref), this is the canonical accessor for the stored
poset `P`. Use it instead of reaching into `enc.P` directly.
"""
encoding_poset(enc::CompiledEncoding) = enc.P
encoding_poset(pi::GridEncodingMap) = pi.P

"""
    encoding_map(enc) -> pi

Return the underlying encoding-map object.

For [`CompiledEncoding`](@ref), this is the backend map object that actually
implements classification. Use it instead of reading `enc.pi` directly.
"""
encoding_map(enc::CompiledEncoding) = enc.pi
encoding_map(pi::AbstractPLikeEncodingMap) = pi

"""
    encoding_axes(enc) -> axes or nothing

Return semantic coordinate axes for an encoding object, when available.

Axes are optional metadata. They are useful for inspection and plotting, but
not required for the core classification contract.
"""
encoding_axes(enc::CompiledEncoding) = axes_from_encoding(enc)
encoding_axes(pi::AbstractPLikeEncodingMap) = axes_from_encoding(pi)

"""
    encoding_representatives(enc) -> reps or nothing

Return representative points for the regions of an encoding object, when
available.

Representatives are intended for inspection and diagnostics. They should not be
treated as a substitute for the underlying encoding map.
"""
encoding_representatives(enc::CompiledEncoding) = representatives(enc)
encoding_representatives(pi::AbstractPLikeEncodingMap) = representatives(pi)

function _encoding_describe(enc::CompiledEncoding)
    return (;
        kind=:compiled_encoding,
        poset_type=typeof(enc.P),
        map_type=typeof(enc.pi),
        parameter_dim=dimension(enc),
        has_axes=encoding_axes(enc) !== nothing,
        has_representatives=encoding_representatives(enc) !== nothing,
        meta_type=typeof(enc.meta),
    )
end

function _encoding_describe(pi::GridEncodingMap)
    return (;
        kind=:grid_encoding_map,
        poset_type=typeof(pi.P),
        parameter_dim=dimension(pi),
        axis_sizes=Tuple(pi.sizes),
        orientation=pi.orientation,
        nregions=prod(pi.sizes),
    )
end

function Base.show(io::IO, pi::GridEncodingMap)
    d = _encoding_describe(pi)
    print(io, "GridEncodingMap(parameter_dim=", d.parameter_dim,
          ", axis_sizes=", d.axis_sizes, ")")
end

function Base.show(io::IO, ::MIME"text/plain", pi::GridEncodingMap)
    d = _encoding_describe(pi)
    print(io, "GridEncodingMap",
          "\n  poset_type: ", d.poset_type,
          "\n  parameter_dim: ", d.parameter_dim,
          "\n  axis_sizes: ", repr(d.axis_sizes),
          "\n  orientation: ", repr(d.orientation),
          "\n  nregions: ", d.nregions)
end

function locate(pi::GridEncodingMap{N,T}, x::AbstractVector{<:Real}) where {N,T}
    length(x) == N || throw(ArgumentError("GridEncodingMap.locate: expected a query vector of length $N, got length $(length(x))."))
    lin = 1
    for i in 1:N
        xi = pi.orientation[i] == 1 ? x[i] : -x[i]
        if xi == 0
            xi = zero(xi)
        end
        idx = searchsortedlast(pi.coords[i], xi)
        if idx < 1
            return 0
        end
        lin += (idx - 1) * pi.strides[i]
    end
    return lin
end

function locate(pi::GridEncodingMap{N,T}, x::NTuple{N,<:Real}) where {N,T}
    lin = 1
    @inbounds for i in 1:N
        xi = pi.orientation[i] == 1 ? x[i] : -x[i]
        if xi == 0
            xi = zero(xi)
        end
        idx = searchsortedlast(pi.coords[i], xi)
        if idx < 1
            return 0
        end
        lin += (idx - 1) * pi.strides[i]
    end
    return lin
end

function locate_many!(dest::AbstractVector{<:Integer}, pi::GridEncodingMap{N,T},
                      X::AbstractMatrix{<:AbstractFloat}; kwargs...) where {N,T}
    size(X, 1) == N || throw(ArgumentError("locate_many!: expected query matrix with $N rows, got $(size(X, 1))."))
    length(dest) == size(X, 2) || throw(ArgumentError("locate_many!: destination length $(length(dest)) must equal the number of query columns $(size(X, 2))."))
    _ = kwargs
    @inbounds for j in 1:size(X, 2)
        lin = 1
        for i in 1:N
            xi = pi.orientation[i] == 1 ? X[i, j] : -X[i, j]
            if xi == 0
                xi = zero(xi)
            end
            idx = searchsortedlast(pi.coords[i], xi)
            if idx < 1
                lin = 0
                break
            end
            lin += (idx - 1) * pi.strides[i]
        end
        dest[j] = lin
    end
    return dest
end

function locate_many!(dest::AbstractVector{<:Integer}, pi::GridEncodingMap{N,T},
                      X::AbstractMatrix{<:Real}; kwargs...) where {N,T}
    Xf = Matrix{Float64}(undef, size(X, 1), size(X, 2))
    @inbounds for j in axes(X, 2), i in axes(X, 1)
        Xf[i, j] = float(X[i, j])
    end
    return locate_many!(dest, pi, Xf; kwargs...)
end

function representatives(pi::GridEncodingMap{N,T}) where {N,T}
    total = prod(pi.sizes)
    reps = Vector{NTuple{N,T}}(undef, total)
    idxs = ones(Int, N)
    for lin in 1:total
        reps[lin] = ntuple(i -> pi.coords[i][idxs[i]], N)
        for i in 1:N
            idxs[i] += 1
            if idxs[i] <= pi.sizes[i]
                break
            else
                idxs[i] = 1
            end
        end
    end
    return reps
end

"""
    check_encoding_map(pi; throw=false) -> NamedTuple

Validate an encoding-map object against the canonical `EncodingCore` contract.

This helper is meant for advanced users constructing encoding maps by hand. It
 checks the ambient dimension, optional axes/representative metadata, and any
 owner-specific structural invariants.

Use `throw=true` when invalid data should raise immediately instead of
 returning a report. Wrap the returned report with
 [`encoding_validation_summary`](@ref) when you want a readable notebook or
 REPL summary.
"""
function check_encoding_map(pi::AbstractPLikeEncodingMap; throw::Bool=false)
    issues = String[]
    dim = try
        dimension(pi)
    catch err
        push!(issues, "dimension(pi) failed: $(sprint(showerror, err))")
        0
    end
    dim isa Integer || push!(issues, "dimension(pi) must return an integer.")
    dim isa Integer && dim >= 1 || push!(issues, "dimension(pi) must be at least 1.")
    dim = dim isa Integer ? Int(dim) : 0
    if dim > 0
        try
            _check_axes_contract(encoding_axes(pi), dim, issues)
        catch err
            push!(issues, "axes_from_encoding(pi) failed: $(sprint(showerror, err))")
        end
        try
            _check_representative_contract(encoding_representatives(pi), dim, issues)
        catch err
            push!(issues, "representatives(pi) failed: $(sprint(showerror, err))")
        end
    end
    valid = isempty(issues)
    throw && !valid && _throw_invalid_encoding(:encoding_map, issues)
    return _encoding_issue_report(:encoding_map, valid;
                                  map_type=typeof(pi),
                                  parameter_dim=dim,
                                  issues=issues)
end

function check_encoding_map(pi::GridEncodingMap; throw::Bool=false)
    issues = String[]
    for i in 1:dimension(pi)
        axis = pi.coords[i]
        issorted(axis) || push!(issues, "axis $i must be sorted.")
        allunique(axis) || push!(issues, "axis $i must not contain duplicate coordinates.")
        (pi.orientation[i] == 1 || pi.orientation[i] == -1) ||
            push!(issues, "orientation[$i] must be +1 or -1.")
        length(axis) == pi.sizes[i] || push!(issues, "sizes[$i] does not match axis $i length.")
    end
    pi.strides == _grid_strides(pi.sizes) || push!(issues, "stored strides do not match axis sizes.")
    _check_axes_contract(pi.coords, dimension(pi), issues)
    _check_representative_contract(representatives(pi), dimension(pi), issues)
    valid = isempty(issues)
    throw && !valid && _throw_invalid_encoding(:encoding_map, issues)
    return _encoding_issue_report(:encoding_map, valid;
                                  map_type=typeof(pi),
                                  parameter_dim=dimension(pi),
                                  axis_sizes=Tuple(pi.sizes),
                                  issues=issues)
end

"""
    check_compiled_encoding(enc; throw=false) -> NamedTuple

Validate a [`CompiledEncoding`](@ref) against the canonical wrapper contract.

This checks the wrapped encoding map, optional axes/representatives, and basic
 wrapper consistency. Use it when building `CompiledEncoding` objects manually
 or when debugging encoding-map plumbing. Wrap the returned report with
 [`encoding_validation_summary`](@ref) when you want a readable notebook or
 REPL summary.
"""
function check_compiled_encoding(enc::CompiledEncoding; throw::Bool=false)
    issues = String[]
    report = check_encoding_map(enc.pi)
    report.valid || append!(issues, String.(report.issues))
    dim = try
        dimension(enc)
    catch err
        push!(issues, "dimension(enc) failed: $(sprint(showerror, err))")
        0
    end
    dim = dim isa Integer ? Int(dim) : 0
    enc.P === nothing && push!(issues, "encoding poset must not be nothing.")
    dim > 0 && _check_axes_contract(enc.axes, dim, issues)
    dim > 0 && _check_representative_contract(enc.reps, dim, issues)
    valid = isempty(issues)
    throw && !valid && _throw_invalid_encoding(:compiled_encoding, issues)
    return _encoding_issue_report(:compiled_encoding, valid;
                                  poset_type=typeof(enc.P),
                                  map_type=typeof(enc.pi),
                                  parameter_dim=dim,
                                  has_axes=enc.axes !== nothing,
                                  has_representatives=enc.reps !== nothing,
                                  issues=issues)
end

"""
    Base.show(io::IO, summary::EncodingValidationSummary)

Compact one-line summary for a wrapped encoding validation report.
"""
function Base.show(io::IO, summary::EncodingValidationSummary)
    report = summary.report
    kind = get(report, :kind, :encoding_validation)
    valid = get(report, :valid, false)
    issues = get(report, :issues, String[])
    print(io, "EncodingValidationSummary(kind=", kind,
          ", valid=", valid,
          ", issues=", length(issues), ")")
end

"""
    Base.show(io::IO, ::MIME\"text/plain\", summary::EncodingValidationSummary)

Verbose multi-line summary for a wrapped encoding validation report.
"""
function Base.show(io::IO, ::MIME"text/plain", summary::EncodingValidationSummary)
    report = summary.report
    kind = get(report, :kind, :encoding_validation)
    valid = get(report, :valid, false)
    issues = get(report, :issues, String[])
    println(io, "EncodingValidationSummary")
    println(io, "  kind = ", kind)
    println(io, "  valid = ", valid)
    for key in (:map_type, :poset_type, :parameter_dim, :axis_sizes, :has_axes, :has_representatives, :query_type, :query_length, :matrix_type, :size)
        haskey(report, key) || continue
        println(io, "  ", key, " = ", getfield(report, key))
    end
    if isempty(issues)
        println(io, "  issues = none")
    else
        println(io, "  issues:")
        for msg in issues
            println(io, "    - ", msg)
        end
    end
end

end # module EncodingCore
