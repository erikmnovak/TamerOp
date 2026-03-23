"""
    Results

Workflow-facing result wrappers for encoding, cohomology-dimension,
resolution, and invariant computations.

Ownership map
- [`EncodingResult`](@ref) owns the inspectable wrapper returned by
  user-facing encoding workflows.
- [`CohomologyDimsResult`](@ref) owns the cheap dims-only workflow result for
  cohomology-first exploration.
- [`EncodedComplexResult`](@ref) owns the encoded-complex workflow result for
  downstream invariants that need the full cochain complex rather than one
  encoded module.
- [`ModuleTranslationResult`](@ref) owns workflow-facing change-of-poset module
  translations together with provenance from the source workflow object.
- [`ResolutionResult`](@ref) owns resolution outputs together with optional
  provenance and cheap summaries.
- [`InvariantResult`](@ref) owns computed invariant values together with the
  source workflow object they came from.
- `check_*_result` and [`result_validation_summary`](@ref) own the canonical
  notebook-friendly validation surface for hand-built result wrappers.

What this module does not own
- encoding algorithms or compiled encoding internals; those belong to
  `Workflow`, `DataIngestion`, `EncodingCore`, and the backend owners,
- algebraic resolution or invariant algorithms; those belong to
  `IndicatorResolutions`, `DerivedFunctors`, and the invariant-family owners,
- low-level module algebra; that belongs to `Modules` and
  `AbelianCategories`.

Cheap-first workflow
- start with `describe(result)` or the owner-local alias
  [`result_summary`](@ref),
- then use semantic accessors such as [`encoding_module`](@ref),
  [`resolution_object`](@ref), [`invariant_value`](@ref), and
  [`source_result`](@ref),
- call `unwrap(result)` only when you explicitly want the raw underlying
  payload.
"""
module Results

using ..CoreModules: AbstractCoeffField, QQField, SessionCache
using ..Options: EncodingOptions, ResolutionOptions, InvariantOptions
using ..EncodingCore: CompiledEncoding, _compile_encoding_cached
import ..CoreModules: change_field
import ..EncodingCore: compile_encoding, encoding_poset, encoding_map, encoding_axes,
                       encoding_representatives, check_encoding_map, check_compiled_encoding

function encoding_module end
function encoding_complex end
function translated_module end
function translation_map end
function translation_kind end
function resolution_object end
function invariant_value end
function source_result end
function result_summary end
function result_validation_summary end
function check_encoding_result end
function check_cohomology_dims_result end
function check_encoded_complex_result end
function check_resolution_result end
function check_invariant_result end
function _materialize_complex end

"""
    ResultValidationSummary

Display wrapper for validation reports returned by the workflow-result
`check_*_result(...)` helpers.

The canonical result validators still return structured `NamedTuple`s for easy
programmatic inspection. Wrap one of those reports with
[`result_validation_summary`](@ref) when you want a compact notebook/REPL
presentation of the same information.
"""
struct ResultValidationSummary{R}
    report::R
end

"""
    result_validation_summary(report) -> ResultValidationSummary

Wrap a workflow-result validation report in a display-oriented object.

Use this when you want the report returned by
[`check_encoding_result`](@ref), [`check_cohomology_dims_result`](@ref),
[`check_resolution_result`](@ref), or [`check_invariant_result`](@ref) to print
as a compact mathematical summary with issues laid out line by line.
"""
@inline result_validation_summary(report::NamedTuple) = ResultValidationSummary(report)

@inline _result_report(kind::Symbol, valid::Bool; kwargs...) = (; kind, valid, kwargs...)

function _throw_invalid_result(kind::Symbol, issues::Vector{String})
    throw(ArgumentError(string(kind, ": ", isempty(issues) ? "invalid workflow result." : join(issues, " "))))
end

@inline _try_length(x) = try length(x) catch; nothing end
@inline _try_module_dims(x) = try module_dims(materialize_module(x)) catch; nothing end
@inline _try_complex_term_count(x) = try length(getproperty(x, :terms)) catch; nothing end
@inline _try_complex_degree_range(x) = try getproperty(x, :tmin):getproperty(x, :tmax) catch; nothing end
@inline _materialize_complex(C) = C
@inline _try_encoding_check(pi) = try
    pi isa CompiledEncoding ? check_compiled_encoding(pi) : check_encoding_map(pi)
catch
    nothing
end

"""
    EncodingResult(P, M, pi; H=nothing, presentation=nothing,
                  opts=EncodingOptions(), backend=opts.backend, meta=NamedTuple())

Workflow-facing wrapper for the output of a user-facing `encode(...)`.

`EncodingResult` stores:
- the finite encoding poset `P`,
- the encoded module `M`,
- the encoding map `pi`,
- optional cohomological provenance `H`,
- optional presentation/provenance data,
- the encoding options used to produce the result.

Use this object when you want an inspectable workflow result. Prefer semantic
accessors such as [`encoding_poset`](@ref), [`encoding_module`](@ref), and
[`encoding_map`](@ref) over direct field access. Prefer `describe(enc)` or
[`result_summary`](@ref) plus `dimensions(enc)` for inspection before falling
back to `unwrap(enc)`. Use [`unwrap`](@ref) only when you explicitly want the
low-level raw tuple `(P, M, pi)`.
"""
struct EncodingResult{PType,MType,PiType,HType,PresType,MetaType}
    P::PType
    M::MType
    pi::PiType
    H::HType
    presentation::PresType
    opts::EncodingOptions
    backend::Symbol
    meta::MetaType
end

EncodingResult(P, M, pi;
               H=nothing,
               presentation=nothing,
               opts::EncodingOptions=EncodingOptions(),
               backend::Symbol=opts.backend,
               meta=NamedTuple()) =
    EncodingResult(P, M, pi, H, presentation, opts, backend, meta)

"""
    CohomologyDimsResult(P, dims, pi; degree=0, field=QQField(), meta=NamedTuple())

Workflow-facing wrapper for dims-only cohomology output on an encoding poset.

This is the cheap summary result returned by dims-first encoding workflows. It
stores the encoding poset `P`, the degree-wise dimension data `dims`, the
encoding map `pi`, and the chosen cohomological degree `degree`.

Use this when downstream work only needs cohomology dimensions. Prefer
[`cohomology_dims`](@ref), [`result_summary`](@ref), and [`encoding_map`](@ref)
over direct field access. Use [`unwrap`](@ref) only when you explicitly want
the raw tuple `(P, dims, pi)`.
"""
struct CohomologyDimsResult{PType,DType,PiType,FType,MetaType}
    P::PType
    dims::DType
    pi::PiType
    degree::Int
    field::FType
    meta::MetaType
end

CohomologyDimsResult(P, dims, pi;
                     degree::Int=0,
                     field::AbstractCoeffField=QQField(),
                     meta=NamedTuple()) =
    CohomologyDimsResult(P, dims, pi, degree, field, meta)

"""
    EncodedComplexResult(P, C, pi; field=QQField(), meta=NamedTuple())

Workflow-facing wrapper for encoded-complex outputs.

`EncodedComplexResult` stores:
- the finite encoding poset `P`,
- the encoded cochain-complex payload `C`,
- the encoding map `pi`,
- the ground field used for the encoded complex,
- lightweight metadata/provenance.

Use this when downstream work needs the full encoded complex rather than a
single encoded module. Prefer semantic accessors such as
[`encoding_complex`](@ref), [`encoding_poset`](@ref), and [`encoding_map`](@ref)
over direct field access. Prefer `describe(enc)` or [`result_summary`](@ref)
for inspection before falling back to [`unwrap`](@ref).

The complex payload may be either a materialized `ModuleCochainComplex` or a
lazy encoded-cochain storage object that is materialized only when a
downstream workflow actually needs full term/differential data.
"""
struct EncodedComplexResult{PType,CType,PiType,FType,MetaType}
    P::PType
    C::CType
    pi::PiType
    field::FType
    meta::MetaType
end

EncodedComplexResult(P, C, pi;
                     field::AbstractCoeffField=QQField(),
                     meta=NamedTuple()) =
    EncodedComplexResult(P, C, pi, field, meta)

"""
    ModuleTranslationResult(kind, M, map; classifier=nothing, source=nothing, meta=NamedTuple())

Workflow-facing wrapper for change-of-poset translations of modules.

`ModuleTranslationResult` stores:
- the translated module `M`,
- the finite monotone map used for the translation,
- an optional postcomposed ambient classifier when the translated object still
  has a meaningful ambient encoding map,
- the source workflow object when one exists,
- lightweight provenance metadata.

Use [`translated_module`](@ref), [`translation_map`](@ref), and
[`source_result`](@ref) for inspection before falling back to raw fields.
Pushforward-style workflow translations usually populate the optional ambient
classifier, while restriction-style translations may legitimately leave it as
`nothing`.
"""
struct ModuleTranslationResult{MType,MapType,ClsType,SrcType,MetaType}
    kind::Symbol
    M::MType
    map::MapType
    classifier::ClsType
    source::SrcType
    meta::MetaType
end

ModuleTranslationResult(kind::Symbol, M, map;
                        classifier=nothing,
                        source=nothing,
                        meta=NamedTuple()) =
    ModuleTranslationResult(kind, M, map, classifier, source, meta)

materialize_module(M) = M
module_dims(M) = M.dims

"""
    result_summary(result) -> NamedTuple

Owner-local summary alias for workflow result wrappers.

Use this when you are already working in `Results` or `TamerOp.Advanced`
and want the same cheap semantic summary returned by `describe(result)` without
leaving the owner-module vocabulary.
"""
result_summary(enc::EncodingResult) = _result_describe(enc)
result_summary(enc::CohomologyDimsResult) = _result_describe(enc)
result_summary(enc::EncodedComplexResult) = _result_describe(enc)
result_summary(res::ModuleTranslationResult) = _result_describe(res)

"""
    encoding_module(enc::EncodingResult)

Return the encoded module stored in an [`EncodingResult`](@ref).

This is the canonical semantic accessor for the module payload. Use it instead
of reading `enc.M` directly.
"""
encoding_module(enc::EncodingResult) = materialize_module(enc.M)

"""
    encoding_complex(enc::EncodedComplexResult)

Return the encoded cochain complex stored in an [`EncodedComplexResult`](@ref).

Use this instead of reading `enc.C` directly.
"""
encoding_complex(enc::EncodedComplexResult) = enc.C

"""
    translated_module(res::ModuleTranslationResult)

Return the translated module stored in a workflow change-of-poset result.

Use this instead of reading `res.M` directly.
"""
translated_module(res::ModuleTranslationResult) = materialize_module(res.M)

"""
    translation_map(res::ModuleTranslationResult)

Return the finite monotone map used to build a workflow change-of-poset result.
"""
translation_map(res::ModuleTranslationResult) = res.map

"""
    translation_kind(res::ModuleTranslationResult)

Return the change-of-poset workflow kind recorded in `res`.

Typical values are `:restriction`, `:pushforward_left`, `:pushforward_right`,
`:derived_pushforward_left`, and `:derived_pushforward_right`.
"""
translation_kind(res::ModuleTranslationResult) = res.kind

compile_encoding(enc::EncodingResult; kwargs...) =
    enc.pi isa CompiledEncoding ? enc.pi : compile_encoding(enc.P, enc.pi; kwargs...)

compile_encoding(enc::CohomologyDimsResult; kwargs...) =
    enc.pi isa CompiledEncoding ? enc.pi : compile_encoding(enc.P, enc.pi; kwargs...)

compile_encoding(enc::EncodedComplexResult; kwargs...) =
    enc.pi isa CompiledEncoding ? enc.pi : compile_encoding(enc.P, enc.pi; kwargs...)

function compile_encoding(res::ModuleTranslationResult; kwargs...)
    cls = res.classifier
    cls === nothing && throw(ArgumentError("compile_encoding: this ModuleTranslationResult does not carry an ambient classifier."))
    return cls isa CompiledEncoding ? cls : compile_encoding(encoding_poset(res), cls; kwargs...)
end

"""
    encoding_poset(result)

Return the finite encoding poset carried by a workflow result.

For `EncodingResult` and `CohomologyDimsResult`, this is the original encoding
poset. For `ModuleTranslationResult`, it is the poset of the translated module.
For `ResolutionResult` and `InvariantResult`, it is forwarded from the stored
provenance object when available.
"""
encoding_poset(enc::EncodingResult) = enc.P
encoding_poset(enc::CohomologyDimsResult) = enc.P
encoding_poset(enc::EncodedComplexResult) = enc.P

"""
    encoding_map(result)

Return the encoding map carried by a workflow result.

Use this instead of reading `result.pi` or `result.enc.pi` directly.
For `ModuleTranslationResult`, this returns the optional ambient classifier
carried by pushforward-style workflow translations; restriction-style results
may legitimately return `nothing`.
"""
encoding_map(enc::EncodingResult) = enc.pi
encoding_map(enc::CohomologyDimsResult) = enc.pi
encoding_map(enc::EncodedComplexResult) = enc.pi

encoding_axes(enc::EncodingResult) = encoding_axes(compile_encoding(enc))
encoding_axes(enc::CohomologyDimsResult) = encoding_axes(compile_encoding(enc))
encoding_axes(enc::EncodedComplexResult) = encoding_axes(compile_encoding(enc))

encoding_representatives(enc::EncodingResult) = encoding_representatives(compile_encoding(enc))
encoding_representatives(enc::CohomologyDimsResult) = encoding_representatives(compile_encoding(enc))
encoding_representatives(enc::EncodedComplexResult) = encoding_representatives(compile_encoding(enc))

function Base.show(io::IO, enc::EncodingResult)
    d = _result_describe(enc)
    print(io, "EncodingResult(backend=", d.backend,
          ", module_type=", nameof(d.module_type),
          ", compiled=", d.compiled, ")")
end

function Base.show(io::IO, ::MIME"text/plain", enc::EncodingResult)
    d = _result_describe(enc)
    print(io, "EncodingResult",
          "\n  poset_type: ", d.poset_type,
          "\n  module_type: ", d.module_type,
          "\n  encoding_map_type: ", d.encoding_map_type,
          "\n  backend: ", d.backend,
          "\n  compiled: ", d.compiled,
          "\n  has_cohomology: ", d.has_cohomology,
          "\n  has_presentation: ", d.has_presentation,
          "\n  module_dims: ", repr(d.module_dims))
end

function Base.show(io::IO, enc::CohomologyDimsResult)
    d = _result_describe(enc)
    print(io, "CohomologyDimsResult(degree=", d.degree,
          ", compiled=", d.compiled,
          ", dims_type=", nameof(d.dims_type), ")")
end

function Base.show(io::IO, ::MIME"text/plain", enc::CohomologyDimsResult)
    d = _result_describe(enc)
    print(io, "CohomologyDimsResult",
          "\n  degree: ", d.degree,
          "\n  field_type: ", d.field_type,
          "\n  poset_type: ", d.poset_type,
          "\n  dims_type: ", d.dims_type,
          "\n  dims_length: ", repr(d.dims_length),
          "\n  encoding_map_type: ", d.encoding_map_type,
          "\n  compiled: ", d.compiled)
end

function Base.show(io::IO, enc::EncodedComplexResult)
    d = _result_describe(enc)
    print(io, "EncodedComplexResult(field_type=", nameof(d.field_type),
          ", complex_type=", nameof(d.complex_type),
          ", compiled=", d.compiled, ")")
end

function Base.show(io::IO, ::MIME"text/plain", enc::EncodedComplexResult)
    d = _result_describe(enc)
    print(io, "EncodedComplexResult",
          "\n  poset_type: ", d.poset_type,
          "\n  complex_type: ", d.complex_type,
          "\n  encoding_map_type: ", d.encoding_map_type,
          "\n  compiled: ", d.compiled,
          "\n  field_type: ", d.field_type,
          "\n  degree_range: ", repr(d.degree_range),
          "\n  nterms: ", repr(d.nterms))
end

function Base.show(io::IO, res::ModuleTranslationResult)
    d = _result_describe(res)
    print(io, "ModuleTranslationResult(kind=", d.translation_kind,
          ", module_type=", nameof(d.module_type),
          ", has_classifier=", d.has_classifier, ")")
end

function Base.show(io::IO, ::MIME"text/plain", res::ModuleTranslationResult)
    d = _result_describe(res)
    print(io, "ModuleTranslationResult",
          "\n  translation_kind: ", d.translation_kind,
          "\n  module_type: ", d.module_type,
          "\n  map_type: ", d.map_type,
          "\n  has_classifier: ", d.has_classifier,
          "\n  source_type: ", d.source_type,
          "\n  module_dims: ", repr(d.module_dims))
end

"""
    Base.show(io::IO, summary::ResultValidationSummary)

Compact one-line summary for a wrapped workflow-result validation report.
"""
function Base.show(io::IO, summary::ResultValidationSummary)
    report = summary.report
    kind = get(report, :kind, :result_validation)
    valid = get(report, :valid, false)
    issues = get(report, :issues, String[])
    print(io, "ResultValidationSummary(kind=", kind,
          ", valid=", valid,
          ", issues=", length(issues), ")")
end

"""
    Base.show(io::IO, ::MIME\"text/plain\", summary::ResultValidationSummary)

Verbose multi-line summary for a wrapped workflow-result validation report.
"""
function Base.show(io::IO, ::MIME"text/plain", summary::ResultValidationSummary)
    report = summary.report
    kind = get(report, :kind, :result_validation)
    valid = get(report, :valid, false)
    issues = get(report, :issues, String[])
    println(io, "ResultValidationSummary")
    println(io, "  kind = ", kind)
    println(io, "  valid = ", valid)
    for key in (:backend, :degree, :dims_length, :has_cohomology, :has_betti, :has_minimality, :invariant)
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

Base.length(::EncodingResult) = 3
Base.IteratorSize(::Type{<:EncodingResult}) = Base.HasLength()

function Base.iterate(enc::EncodingResult, state::Int=1)
    state == 1 && return (enc.P, 2)
    state == 2 && return (enc.M, 3)
    state == 3 && return (enc.pi, 4)
    return nothing
end

Base.length(::CohomologyDimsResult) = 3
Base.IteratorSize(::Type{<:CohomologyDimsResult}) = Base.HasLength()

function Base.iterate(enc::CohomologyDimsResult, state::Int=1)
    state == 1 && return (enc.P, 2)
    state == 2 && return (enc.dims, 3)
    state == 3 && return (enc.pi, 4)
    return nothing
end

Base.length(::EncodedComplexResult) = 3
Base.IteratorSize(::Type{<:EncodedComplexResult}) = Base.HasLength()

function Base.iterate(enc::EncodedComplexResult, state::Int=1)
    state == 1 && return (enc.P, 2)
    state == 2 && return (enc.C, 3)
    state == 3 && return (enc.pi, 4)
    return nothing
end

@inline function _encoding_with_session_cache(enc::EncodingResult,
                                              session_cache::Union{Nothing,SessionCache})
    session_cache === nothing && return enc
    raw_pi = enc.pi isa CompiledEncoding ? enc.pi.pi : enc.pi
    pi2 = _compile_encoding_cached(enc.P, raw_pi, session_cache)
    return EncodingResult(enc.P, enc.M, pi2;
                          H=enc.H,
                          presentation=enc.presentation,
                          opts=enc.opts,
                          backend=enc.backend,
                          meta=enc.meta)
end

@inline function _encoding_with_session_cache(enc::CohomologyDimsResult,
                                              session_cache::Union{Nothing,SessionCache})
    session_cache === nothing && return enc
    raw_pi = enc.pi isa CompiledEncoding ? enc.pi.pi : enc.pi
    pi2 = _compile_encoding_cached(enc.P, raw_pi, session_cache)
    return CohomologyDimsResult(enc.P, enc.dims, pi2;
                                degree=enc.degree,
                                field=enc.field,
                                meta=enc.meta)
end

@inline function _encoding_with_session_cache(enc::EncodedComplexResult,
                                              session_cache::Union{Nothing,SessionCache})
    session_cache === nothing && return enc
    raw_pi = enc.pi isa CompiledEncoding ? enc.pi.pi : enc.pi
    pi2 = _compile_encoding_cached(enc.P, raw_pi, session_cache; include_reps=false)
    return EncodedComplexResult(enc.P, enc.C, pi2;
                                field=enc.field,
                                meta=enc.meta)
end

function change_field(enc::EncodingResult, field::AbstractCoeffField)
    M2 = change_field(enc.M, field)
    H2 = enc.H === nothing ? nothing : change_field(enc.H, field)
    pres2 = enc.presentation
    if pres2 !== nothing && hasmethod(change_field, (typeof(pres2), AbstractCoeffField))
        pres2 = change_field(pres2, field)
    end
    return EncodingResult(enc.P, M2, enc.pi;
                          H=H2,
                          presentation=pres2,
                          opts=enc.opts,
                          backend=enc.backend,
                          meta=enc.meta)
end

function change_field(enc::CohomologyDimsResult, field::AbstractCoeffField)
    return CohomologyDimsResult(enc.P, copy(enc.dims), enc.pi;
                                degree=enc.degree,
                                field=field,
                                meta=enc.meta)
end

function change_field(enc::EncodedComplexResult, field::AbstractCoeffField)
    return EncodedComplexResult(enc.P, change_field(enc.C, field), enc.pi;
                                field=field,
                                meta=enc.meta)
end

unwrap(enc::EncodingResult) = (enc.P, enc.M, enc.pi)
unwrap(enc::CohomologyDimsResult) = (enc.P, enc.dims, enc.pi)
unwrap(enc::EncodedComplexResult) = (enc.P, enc.C, enc.pi)
unwrap(res::ModuleTranslationResult) = translated_module(res)

"""
    check_encoding_result(enc; throw=false) -> NamedTuple

Validate a hand-built [`EncodingResult`](@ref).

This checks the stored encoding map, module-dimension accessibility, and basic
workflow-result consistency. Use it when constructing result objects manually
for tests, examples, or advanced workflows. Wrap the returned report with
[`result_validation_summary`](@ref) when you want a readable notebook or REPL
summary.
"""
function check_encoding_result(enc::EncodingResult; throw::Bool=false)
    issues = String[]
    enc.P === nothing && push!(issues, "encoding poset must not be nothing.")
    enc.backend isa Symbol || push!(issues, "backend must be a Symbol.")
    _try_module_dims(enc.M) === nothing && push!(issues, "encoded module must support dimension inspection.")
    report = _try_encoding_check(enc.pi)
    report === nothing || report.valid || append!(issues, String.(report.issues))
    valid = isempty(issues)
    throw && !valid && _throw_invalid_result(:encoding_result, issues)
    return _result_report(:encoding_result, valid;
                          backend=enc.backend,
                          has_cohomology=enc.H !== nothing,
                          issues=issues)
end

"""
    check_cohomology_dims_result(enc; throw=false) -> NamedTuple

Validate a hand-built [`CohomologyDimsResult`](@ref).

Wrap the returned report with [`result_validation_summary`](@ref) when you want
a readable notebook or REPL summary.
"""
function check_cohomology_dims_result(enc::CohomologyDimsResult; throw::Bool=false)
    issues = String[]
    enc.P === nothing && push!(issues, "encoding poset must not be nothing.")
    enc.degree isa Int || push!(issues, "degree must be an Int.")
    enc.field isa AbstractCoeffField || push!(issues, "field must be an AbstractCoeffField.")
    _try_length(enc.dims) === nothing && push!(issues, "dims payload should expose a length for inspection.")
    report = _try_encoding_check(enc.pi)
    report === nothing || report.valid || append!(issues, String.(report.issues))
    valid = isempty(issues)
    throw && !valid && _throw_invalid_result(:cohomology_dims_result, issues)
    return _result_report(:cohomology_dims_result, valid;
                          degree=enc.degree,
                          dims_length=_try_length(enc.dims),
                          issues=issues)
end

"""
    check_encoded_complex_result(enc; throw=false) -> NamedTuple

Validate a hand-built [`EncodedComplexResult`](@ref).

Wrap the returned report with [`result_validation_summary`](@ref) when you want
a readable notebook or REPL summary.
"""
function check_encoded_complex_result(enc::EncodedComplexResult; throw::Bool=false)
    issues = String[]
    enc.P === nothing && push!(issues, "encoding poset must not be nothing.")
    enc.C === nothing && push!(issues, "encoded complex must not be nothing.")
    enc.field isa AbstractCoeffField || push!(issues, "field must be an AbstractCoeffField.")
    _try_complex_term_count(enc.C) === nothing && push!(issues, "encoded complex should expose term storage for inspection.")
    report = _try_encoding_check(enc.pi)
    report === nothing || report.valid || append!(issues, String.(report.issues))
    valid = isempty(issues)
    throw && !valid && _throw_invalid_result(:encoded_complex_result, issues)
    return _result_report(:encoded_complex_result, valid;
                          degree_range=_try_complex_degree_range(enc.C),
                          nterms=_try_complex_term_count(enc.C),
                          issues=issues)
end

"""
    ResolutionResult(res; enc=nothing, betti=nothing, minimality=nothing,
                     opts=ResolutionOptions(), meta=NamedTuple())

Workflow-facing wrapper for a resolution computation plus provenance.

`ResolutionResult` stores the underlying resolution object together with the
encoding result it came from, optional Betti/minimality summaries, and the
resolution options used to produce it.

Prefer [`resolution_object`](@ref), [`encoding_poset`](@ref), and
[`result_summary`](@ref) over direct field access. Prefer
[`source_result`](@ref) when you want provenance. Use [`unwrap`](@ref) only
when you explicitly want the raw resolution object.
"""
struct ResolutionResult{ResType,EncType,BettiType,MinType,MetaType}
    res::ResType
    enc::EncType
    betti::BettiType
    minimality::MinType
    opts::ResolutionOptions
    meta::MetaType
end

ResolutionResult(res;
                 enc=nothing,
                 betti=nothing,
                 minimality=nothing,
                 opts::ResolutionOptions=ResolutionOptions(),
                 meta=NamedTuple()) =
    ResolutionResult(res, enc, betti, minimality, opts, meta)

unwrap(res::ResolutionResult) = res.res

"""
    check_resolution_result(res; throw=false) -> NamedTuple

Validate a hand-built [`ResolutionResult`](@ref).

Wrap the returned report with [`result_validation_summary`](@ref) when you want
a readable notebook or REPL summary.
"""
function check_resolution_result(res::ResolutionResult; throw::Bool=false)
    issues = String[]
    res.res === nothing && push!(issues, "resolution object must not be nothing.")
    if res.enc !== nothing
        if res.enc isa EncodingResult
            report = check_encoding_result(res.enc)
            report.valid || append!(issues, String.(report.issues))
        else
            push!(issues, "resolution provenance must be an EncodingResult or nothing.")
        end
    end
    valid = isempty(issues)
    throw && !valid && _throw_invalid_result(:resolution_result, issues)
    return _result_report(:resolution_result, valid;
                          has_betti=res.betti !== nothing,
                          has_minimality=res.minimality !== nothing,
                          issues=issues)
end

"""
    InvariantResult(enc, which, value; opts=InvariantOptions(), meta=NamedTuple())

Workflow-facing wrapper for a computed invariant plus provenance.

`InvariantResult` stores the source encoding/dims result, the invariant name,
the computed value, and the invariant options used to produce it.

Prefer [`invariant_value`](@ref), [`encoding_map`](@ref), and `describe(...)`
or [`result_summary`](@ref) over direct field access. Prefer
[`source_result`](@ref) when you want the provenance object. Use [`unwrap`](@ref)
only when you explicitly want the raw invariant value.
"""
struct InvariantResult{EncType,WhichType,ValType,MetaType}
    enc::EncType
    which::WhichType
    value::ValType
    opts::InvariantOptions
    meta::MetaType
end

InvariantResult(enc, which, value;
                opts::InvariantOptions=InvariantOptions(),
                meta=NamedTuple()) =
    InvariantResult(enc, which, value, opts, meta)

function _result_describe(enc::EncodingResult)
    dims = _try_module_dims(enc.M)
    return (;
        kind=:encoding_result,
        poset_type=typeof(enc.P),
        module_type=typeof(enc.M),
        encoding_map_type=typeof(enc.pi),
        compiled=enc.pi isa CompiledEncoding,
        backend=enc.backend,
        has_cohomology=enc.H !== nothing,
        has_presentation=enc.presentation !== nothing,
        module_dims=dims,
    )
end

function _result_describe(enc::CohomologyDimsResult)
    return (;
        kind=:cohomology_dims_result,
        poset_type=typeof(enc.P),
        dims_type=typeof(enc.dims),
        dims_length=_try_length(enc.dims),
        encoding_map_type=typeof(enc.pi),
        compiled=enc.pi isa CompiledEncoding,
        degree=enc.degree,
        field_type=typeof(enc.field),
    )
end

function _result_describe(enc::EncodedComplexResult)
    return (;
        kind=:encoded_complex_result,
        poset_type=typeof(enc.P),
        complex_type=typeof(enc.C),
        encoding_map_type=typeof(enc.pi),
        compiled=enc.pi isa CompiledEncoding,
        field_type=typeof(enc.field),
        degree_range=_try_complex_degree_range(enc.C),
        nterms=_try_complex_term_count(enc.C),
    )
end

function _result_describe(res::ModuleTranslationResult)
    return (;
        kind=:module_translation_result,
        translation_kind=res.kind,
        module_type=typeof(res.M),
        map_type=typeof(res.map),
        has_classifier=res.classifier !== nothing,
        source_type=typeof(res.source),
        module_dims=_try_module_dims(res.M),
    )
end

function _result_describe(res::ResolutionResult)
    return (;
        kind=:resolution_result,
        resolution_type=typeof(res.res),
        source_type=typeof(res.enc),
        has_betti=res.betti !== nothing,
        has_minimality=res.minimality !== nothing,
        opts_type=typeof(res.opts),
    )
end

function _result_describe(inv::InvariantResult)
    return (;
        kind=:invariant_result,
        source_type=typeof(inv.enc),
        invariant=inv.which,
        value_type=typeof(inv.value),
        opts_type=typeof(inv.opts),
    )
end

_encoding_result_dimensions(enc::EncodingResult) = _try_module_dims(enc.M)
_cohomology_dims_payload(enc::CohomologyDimsResult) = enc.dims

unwrap(inv::InvariantResult) = inv.value

result_summary(res::ResolutionResult) = _result_describe(res)
result_summary(inv::InvariantResult) = _result_describe(inv)

"""
    resolution_object(res::ResolutionResult)

Return the underlying resolution/provenance object stored in a
[`ResolutionResult`](@ref).
"""
resolution_object(res::ResolutionResult) = res.res

"""
    invariant_value(inv::InvariantResult)

Return the invariant value stored in an [`InvariantResult`](@ref).
"""
invariant_value(inv::InvariantResult) = inv.value

"""
    source_result(result)

Return the immediate workflow provenance object carried by a result wrapper.

- For [`ResolutionResult`](@ref), this is the source [`EncodingResult`](@ref)
  or `nothing`.
- For [`InvariantResult`](@ref), this is the source
  [`EncodingResult`](@ref) or [`CohomologyDimsResult`](@ref).
- For [`ModuleTranslationResult`](@ref), this is the source
  [`EncodingResult`](@ref) when the translation originated from the workflow
  surface.
- For [`EncodingResult`](@ref) and [`CohomologyDimsResult`](@ref), this returns
  `nothing`.

Use this instead of reading `res.enc` or `inv.enc` directly.
"""
source_result(::EncodingResult) = nothing
source_result(::CohomologyDimsResult) = nothing
source_result(::EncodedComplexResult) = nothing
source_result(res::ModuleTranslationResult) = res.source
source_result(res::ResolutionResult) = res.enc
source_result(inv::InvariantResult) = inv.enc

encoding_poset(res::ModuleTranslationResult) = materialize_module(res.M).Q
encoding_poset(res::ResolutionResult) = res.enc === nothing ? nothing : encoding_poset(res.enc)
encoding_poset(inv::InvariantResult) = encoding_poset(inv.enc)

encoding_map(res::ModuleTranslationResult) = res.classifier
encoding_map(res::ResolutionResult) = res.enc === nothing ? nothing : encoding_map(res.enc)
encoding_map(inv::InvariantResult) = encoding_map(inv.enc)

encoding_axes(res::ModuleTranslationResult) = encoding_axes(compile_encoding(res))
encoding_axes(res::ResolutionResult) = res.enc === nothing ? nothing : encoding_axes(res.enc)
encoding_axes(inv::InvariantResult) = encoding_axes(inv.enc)

encoding_representatives(res::ModuleTranslationResult) = encoding_representatives(compile_encoding(res))
encoding_representatives(res::ResolutionResult) = res.enc === nothing ? nothing : encoding_representatives(res.enc)
encoding_representatives(inv::InvariantResult) = encoding_representatives(inv.enc)

function Base.show(io::IO, res::ResolutionResult)
    d = _result_describe(res)
    print(io, "ResolutionResult(resolution_type=", nameof(d.resolution_type),
          ", has_betti=", d.has_betti,
          ", has_minimality=", d.has_minimality, ")")
end

function Base.show(io::IO, ::MIME"text/plain", res::ResolutionResult)
    d = _result_describe(res)
    print(io, "ResolutionResult",
          "\n  resolution_type: ", d.resolution_type,
          "\n  source_type: ", d.source_type,
          "\n  has_betti: ", d.has_betti,
          "\n  has_minimality: ", d.has_minimality,
          "\n  opts_type: ", d.opts_type)
end

function Base.show(io::IO, inv::InvariantResult)
    d = _result_describe(inv)
    print(io, "InvariantResult(which=", repr(d.invariant),
          ", value_type=", nameof(d.value_type), ")")
end

function Base.show(io::IO, ::MIME"text/plain", inv::InvariantResult)
    d = _result_describe(inv)
    print(io, "InvariantResult",
          "\n  invariant: ", repr(d.invariant),
          "\n  source_type: ", d.source_type,
          "\n  value_type: ", d.value_type,
          "\n  opts_type: ", d.opts_type)
end

"""
    check_invariant_result(inv; throw=false) -> NamedTuple

Validate a hand-built [`InvariantResult`](@ref).

Wrap the returned report with [`result_validation_summary`](@ref) when you want
a readable notebook or REPL summary.
"""
function check_invariant_result(inv::InvariantResult; throw::Bool=false)
    issues = String[]
    inv.which === nothing && push!(issues, "invariant selector `which` must not be nothing.")
    inv.value === nothing && push!(issues, "invariant value must not be nothing.")
    if inv.enc isa EncodingResult
        report = check_encoding_result(inv.enc)
        report.valid || append!(issues, String.(report.issues))
    elseif inv.enc isa CohomologyDimsResult
        report = check_cohomology_dims_result(inv.enc)
        report.valid || append!(issues, String.(report.issues))
    elseif inv.enc isa EncodedComplexResult
        report = check_encoded_complex_result(inv.enc)
        report.valid || append!(issues, String.(report.issues))
    else
        push!(issues, "source object must be an EncodingResult, CohomologyDimsResult, or EncodedComplexResult.")
    end
    valid = isempty(issues)
    throw && !valid && _throw_invalid_result(:invariant_result, issues)
    return _result_report(:invariant_result, valid;
                          invariant=inv.which,
                          issues=issues)
end

function change_field(res::ResolutionResult, field::AbstractCoeffField)
    enc2 = res.enc === nothing ? nothing : change_field(res.enc, field)
    res2 = res.res
    if res2 !== nothing && hasmethod(change_field, (typeof(res2), AbstractCoeffField))
        res2 = change_field(res2, field)
    end
    return ResolutionResult(res2;
                            enc=enc2,
                            betti=res.betti,
                            minimality=res.minimality,
                            opts=res.opts,
                            meta=res.meta)
end

function change_field(inv::InvariantResult, field::AbstractCoeffField)
    enc2 = change_field(inv.enc, field)
    return InvariantResult(enc2, inv.which, inv.value; opts=inv.opts, meta=inv.meta)
end

end # module Results
