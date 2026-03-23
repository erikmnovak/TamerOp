# Shared serialization infrastructure: summaries, validators, JSON helpers,
# typed schemas, feature metadata, and common parsing/build helpers.

"""
    JSONArtifactSummary

Typed summary object returned by [`inspect_json`](@ref) and the owner-local
serialization summary helpers.

This is the cheap-first inspection surface for TamerOp-owned JSON
artifacts. Use [`artifact_kind`](@ref), [`schema_version`](@ref),
[`artifact_field`](@ref), [`artifact_poset_kind`](@ref),
[`has_encoding_map`](@ref), [`has_dense_leq`](@ref), and
[`artifact_size_bytes`](@ref) instead of reading raw JSON fields by hand.
"""
struct JSONArtifactSummary{P,R}
    path::P
    report::R
end

"""
    SerializationValidationSummary

Typed wrapper around raw reports returned by `check_*_json`.

Use [`serialization_validation_summary`](@ref) for compact notebook/REPL
display of validation reports.
"""
struct SerializationValidationSummary{R}
    report::R
end

function artifact_kind end
function schema_version end
function artifact_field end
function artifact_poset_kind end
function artifact_path end
function artifact_profile_hint end
function artifact_data_kind end
function has_encoding_map end
function has_dense_leq end
function artifact_size_bytes end

"""
    serialization_validation_summary(report) -> SerializationValidationSummary

Wrap a raw serialization-validation report in a compact display-oriented object.
"""
@inline serialization_validation_summary(report::NamedTuple) = SerializationValidationSummary(report)

@inline function _serialization_report(
    kind::Symbol,
    valid::Bool;
    issues::AbstractVector{<:AbstractString}=String[],
    kwargs...,
)
    return (; kind, valid, issues=Tuple(String.(issues)), kwargs...)
end

@inline function _throw_invalid_serialization(fname::Symbol, issues::AbstractVector{<:AbstractString})
    msg = isempty(issues) ? "invalid artifact" : " - " * join(issues, "\n - ")
    Base.throw(ArgumentError(string(fname, ": validation failed\n", msg)))
end

function Base.propertynames(summary::JSONArtifactSummary, private::Bool=false)
    keys = Tuple(propertynames(getfield(summary, :report)))
    return private ? (:path, :report, keys...) : (:path, keys...)
end

function Base.getproperty(summary::JSONArtifactSummary, sym::Symbol)
    if sym === :path || sym === :report
        return getfield(summary, sym)
    end
    report = getfield(summary, :report)
    if sym in propertynames(report)
        return getproperty(report, sym)
    end
    return getfield(summary, sym)
end

@inline describe(summary::JSONArtifactSummary) = merge((; path=summary.path), summary.report)
@inline describe(summary::SerializationValidationSummary) = summary.report

@inline artifact_kind(summary::JSONArtifactSummary) = get(summary.report, :kind, nothing)
@inline schema_version(summary::JSONArtifactSummary) = get(summary.report, :schema_version, nothing)
@inline artifact_field(summary::JSONArtifactSummary) = get(summary.report, :field, nothing)
@inline artifact_poset_kind(summary::JSONArtifactSummary) = get(summary.report, :poset_kind, nothing)
@inline artifact_path(summary::JSONArtifactSummary) = summary.path
@inline artifact_profile_hint(summary::JSONArtifactSummary) = get(summary.report, :profile_hint, nothing)
@inline artifact_data_kind(summary::JSONArtifactSummary) = get(summary.report, :data_kind, nothing)
@inline has_encoding_map(summary::JSONArtifactSummary) = get(summary.report, :has_pi, nothing)
@inline has_dense_leq(summary::JSONArtifactSummary) = get(summary.report, :has_dense_leq, nothing)
@inline artifact_size_bytes(summary::JSONArtifactSummary) = get(summary.report, :size_bytes, nothing)

function Base.show(io::IO, summary::JSONArtifactSummary)
    print(io, "JSONArtifactSummary(kind=", repr(artifact_kind(summary)),
          ", schema_version=", repr(schema_version(summary)),
          ", path=", repr(summary.path),
          ", size_bytes=", repr(artifact_size_bytes(summary)), ")")
end

function Base.show(io::IO, ::MIME"text/plain", summary::JSONArtifactSummary)
    d = describe(summary)
    println(io, "JSONArtifactSummary")
    for key in propertynames(d)
        println(io, "  ", key, ": ", repr(getproperty(d, key)))
    end
end

function Base.show(io::IO, summary::SerializationValidationSummary)
    r = summary.report
    print(io, "SerializationValidationSummary(kind=", r.kind,
          ", valid=", r.valid,
          ", issues=", length(r.issues), ")")
end

function Base.show(io::IO, ::MIME"text/plain", summary::SerializationValidationSummary)
    r = summary.report
    println(io, "SerializationValidationSummary")
    println(io, "  kind: ", r.kind)
    println(io, "  valid: ", r.valid)
    for key in propertynames(r)
        key in (:kind, :valid, :issues) && continue
        println(io, "  ", key, ": ", repr(getproperty(r, key)))
    end
    if isempty(r.issues)
        print(io, "  issues: []")
    else
        println(io, "  issues:")
        for issue in r.issues
            println(io, "    - ", issue)
        end
    end
end

"""
    artifact_kind(summary::JSONArtifactSummary)
    schema_version(summary::JSONArtifactSummary)
    artifact_field(summary::JSONArtifactSummary)
    artifact_poset_kind(summary::JSONArtifactSummary)
    artifact_path(summary::JSONArtifactSummary)
    artifact_profile_hint(summary::JSONArtifactSummary)
    artifact_data_kind(summary::JSONArtifactSummary)
    has_encoding_map(summary::JSONArtifactSummary)
    has_dense_leq(summary::JSONArtifactSummary)
    artifact_size_bytes(summary::JSONArtifactSummary)

Semantic accessors for JSON artifact summaries.

These are the preferred accessors for the common inspection questions answered
by [`inspect_json`](@ref) and the owner-local summary helpers.
"""

# =============================================================================
# 1) Shared helpers
# =============================================================================

"""
    feature_schema_header(; format=nothing) -> Dict{String,Any}

Canonical schema header for feature artifacts owned by TamerOp.
"""
function feature_schema_header(; format::Union{Nothing,Symbol}=nothing)
    hdr = Dict{String,Any}(
        "kind" => "features",
        "schema_version" => string(TAMER_FEATURE_SCHEMA_VERSION),
    )
    format === nothing || (hdr["format"] = String(format))
    return hdr
end

"""
    validate_feature_metadata_schema(meta; max_version=TAMER_FEATURE_SCHEMA_VERSION)

Validate a feature metadata object against the canonical feature schema header.
Returns `true` on success and throws on invalid/unsupported schema tags.
"""
function validate_feature_metadata_schema(meta; max_version::VersionNumber=TAMER_FEATURE_SCHEMA_VERSION)
    kind = haskey(meta, "kind") ? String(meta["kind"]) : ""
    kind == "features" || error("Feature metadata has unsupported kind: $(kind)")
    haskey(meta, "schema_version") || error("Feature metadata missing schema_version")
    ver = try
        VersionNumber(String(meta["schema_version"]))
    catch
        bad = haskey(meta, "schema_version") ? meta["schema_version"] : missing
        error("Feature metadata has invalid schema_version: $(bad)")
    end
    ver <= max_version || error("Unsupported feature metadata schema_version: $(ver)")
    return true
end

function _field_to_obj(field::AbstractCoeffField)
    if field isa QQField
        return Dict("kind" => "qq")
    elseif field isa RealField
        T = coeff_type(field)
        return Dict("kind" => "real",
                    "T" => string(T),
                    "rtol" => field.rtol,
                    "atol" => field.atol)
    elseif field isa PrimeField
        return Dict("kind" => "fp", "p" => field.p)
    end
    error("Unsupported coefficient field for JSON serialization: $(typeof(field))")
end

function _field_from_obj(obj)
    kind = lowercase(String(obj["kind"]))
    if kind == "qq"
        return QQField()
    elseif kind == "real"
        Tname = String(obj["T"])
        T = Tname == "Float64" ? Float64 :
            Tname == "Float32" ? Float32 :
            error("Unsupported real field type in JSON: $(Tname)")
        rtol = haskey(obj, "rtol") ? T(obj["rtol"]) : sqrt(eps(T))
        atol = haskey(obj, "atol") ? T(obj["atol"]) : zero(T)
        return RealField(T; rtol=rtol, atol=atol)
    elseif kind == "fp"
        p = Int(obj["p"])
        return PrimeField(p)
    end
    error("Unsupported coeff_field kind: $(kind)")
end

function _scalar_to_json(field::AbstractCoeffField, x)
    if field isa QQField
        return rational_to_string(QQ(x))
    elseif field isa RealField
        return Float64(x)
    elseif field isa PrimeField
        return Int(coerce(field, x).val)
    end
    error("Unsupported coefficient field for scalar serialization: $(typeof(field))")
end

function _scalar_from_json(field::AbstractCoeffField, val)
    if field isa QQField
        if val isa Integer
            return QQ(BigInt(val))
        end
        s = String(val)
        if occursin("/", s)
            return string_to_rational(s)
        end
        return QQ(parse(BigInt, strip(s)))
    elseif field isa RealField
        T = coeff_type(field)
        return val isa AbstractString ? T(parse(Float64, val)) : T(val)
    elseif field isa PrimeField
        return coerce(field, val isa AbstractString ? parse(Int, val) : Int(val))
    end
    error("Unsupported coefficient field for scalar parsing: $(typeof(field))")
end

@inline function _json_write(path::AbstractString, obj; pretty::Bool=false, indent::Int=2)
    open(path, "w") do io
        if pretty && indent > 0
            ac = JSON3.AlignmentContext(:Left, UInt8(clamp(indent, 0, 255)), UInt8(0), UInt8(0))
            JSON3.pretty(io, obj, ac; allow_inf=true)
        else
            JSON3.write(io, obj; allow_inf=true)
        end
    end
    return path
end

@inline _json_read(path::AbstractString) = open(JSON3.read, path)

@inline function _resolve_validation_mode(validation::Symbol)::Bool
    validation === :strict && return true
    validation === :trusted && return false
    error("validation must be :strict or :trusted. Use :strict for external/untrusted files and :trusted for TamerOp-produced files you trust.")
end

@inline function _resolve_owned_json_save_profile(profile::Symbol)
    profile === :compact && return (pretty=false,)
    profile === :debug && return (pretty=true,)
    throw(ArgumentError("profile must be :compact or :debug."))
end

@inline function _resolve_owned_json_pretty(profile::Symbol,
                                            pretty::Union{Nothing,Bool})
    defaults = _resolve_owned_json_save_profile(profile)
    return pretty === nothing ? defaults.pretty : pretty
end

@inline function _resolve_encoding_output_mode(output::Symbol)::Symbol
    output === :fringe && return :fringe
    output === :fringe_with_pi && return :fringe_with_pi
    output === :encoding_result && return :encoding_result
    error("output must be one of :fringe, :fringe_with_pi, :encoding_result.")
end

@inline function _resolve_encoding_save_profile(profile::Symbol)
    profile === :compact && return (include_pi=true, include_leq=:auto, pretty=false)
    profile === :portable && return (include_pi=true, include_leq=true, pretty=false)
    profile === :debug && return (include_pi=true, include_leq=true, pretty=true)
    throw(ArgumentError("profile must be :compact, :portable, or :debug."))
end

@inline function _artifact_size_or_nothing(path::AbstractString)
    try
        return filesize(path)
    catch
        return nothing
    end
end

@inline function _json_pretty_hint(path::AbstractString)
    try
        open(path, "r") do io
            head = read(io, min(filesize(path), 256))
            return any(==(UInt8('\n')), head)
        end
    catch
        return nothing
    end
end

@inline function _field_kind_from_obj(coeff_obj)
    if coeff_obj === nothing
        return nothing
    end
    return haskey(coeff_obj, "kind") ? String(coeff_obj["kind"]) : nothing
end

@inline function _poset_kind_from_obj(poset_obj)
    poset_obj === nothing && return nothing
    return haskey(poset_obj, "kind") ? String(poset_obj["kind"]) : nothing
end

function _poset_has_dense_leq(poset_obj)
    poset_obj === nothing && return nothing
    haskey(poset_obj, "leq") && return true
    if haskey(poset_obj, "left")
        left = _poset_has_dense_leq(poset_obj["left"])
        right = _poset_has_dense_leq(poset_obj["right"])
        return something(left, false) || something(right, false)
    end
    return false
end

@inline function _encoding_profile_hint(has_pi::Bool, has_dense::Union{Nothing,Bool}, pretty_hint::Union{Nothing,Bool})
    has_dense === nothing && return nothing
    pretty_hint === nothing && return nothing
    if has_pi && !has_dense && !pretty_hint
        return :compact
    elseif has_pi && has_dense && !pretty_hint
        return :portable
    elseif has_pi && has_dense && pretty_hint
        return :debug
    end
    return nothing
end

@inline function _owned_profile_hint(pretty_hint::Union{Nothing,Bool})
    pretty_hint === nothing && return nothing
    return pretty_hint ? :debug : :compact
end

@inline function _artifact_summary(path::Union{Nothing,AbstractString}, report::NamedTuple)
    return JSONArtifactSummary(path === nothing ? nothing : String(path), report)
end

function _expect_artifact_kind(summary::JSONArtifactSummary,
                               expected::Union{AbstractString,Tuple},
                               fname::Symbol)
    expected_set = expected isa Tuple ? expected : (expected,)
    kind = artifact_kind(summary)
    kind in expected_set && return summary
    throw(ArgumentError(string(fname, ": expected artifact kind ",
                               join(repr.(expected_set), " or "),
                               ", got ", repr(kind), ".")))
end

"""
    inspect_json(path) -> JSONArtifactSummary

Cheap-first metadata probe for TamerOp-owned JSON artifacts.

This is the canonical inspection path when you want to understand what a file
contains before paying the cost of a full load. The returned
[`JSONArtifactSummary`](@ref) records artifact kind, schema version, file path,
file size, and format-specific hints such as encoding-map presence, poset kind,
dense-`leq` inclusion, or an inferred encoding save profile.

`inspect_json` does not fully validate the artifact schema. Use the
corresponding `check_*_json` helper when you need strict contract validation
before loading, and call the matching `load_*_json` routine only after the file
looks like the artifact family you expect.
"""
function inspect_json(path::AbstractString)
    obj = _json_read(path)
    kind = haskey(obj, "kind") ? String(obj["kind"]) : "unknown"
    schema = if haskey(obj, "schema_version")
        obj["schema_version"] isa Integer ? Int(obj["schema_version"]) : obj["schema_version"]
    elseif haskey(obj, "version")
        obj["version"] isa Integer ? Int(obj["version"]) : obj["version"]
    else
        nothing
    end
    size_bytes = _artifact_size_or_nothing(path)
    pretty_hint = _json_pretty_hint(path)

    report = if kind == "FiniteEncodingFringe"
        poset = obj["poset"]
        coeff = obj["coeff_field"]
        has_pi = haskey(obj, "pi")
        dense_leq = _poset_has_dense_leq(poset)
        (
            kind = kind,
            schema_version = schema,
            field = _field_kind_from_obj(coeff),
            poset_kind = _poset_kind_from_obj(poset),
            nvertices = haskey(poset, "n") ? Int(poset["n"]) : missing,
            n_upsets = haskey(obj, "U") && haskey(obj["U"], "nrows") ? Int(obj["U"]["nrows"]) : missing,
            n_downsets = haskey(obj, "D") && haskey(obj["D"], "nrows") ? Int(obj["D"]["nrows"]) : missing,
            has_pi = has_pi,
            has_dense_leq = dense_leq,
            profile_hint = _encoding_profile_hint(has_pi, dense_leq, pretty_hint),
            size_bytes = size_bytes,
        )
    elseif kind == "FlangeZn"
        (
            kind = kind,
            schema_version = schema,
            field = haskey(obj, "coeff_field") ? _field_kind_from_obj(obj["coeff_field"]) : nothing,
            n = haskey(obj, "n") ? Int(obj["n"]) : missing,
            n_flats = haskey(obj, "flats") ? length(obj["flats"]) : missing,
            n_injectives = haskey(obj, "injectives") ? length(obj["injectives"]) : missing,
            has_phi = haskey(obj, "phi"),
            profile_hint = _owned_profile_hint(pretty_hint),
            size_bytes = size_bytes,
        )
    elseif kind == "PLFringe"
        (
            kind = kind,
            schema_version = schema,
            field = haskey(obj, "coeff_field") ? _field_kind_from_obj(obj["coeff_field"]) : nothing,
            n = haskey(obj, "n") ? Int(obj["n"]) : missing,
            n_upsets = haskey(obj, "ups") ? length(obj["ups"]) : missing,
            n_downsets = haskey(obj, "downs") ? length(obj["downs"]) : missing,
            has_phi = haskey(obj, "phi"),
            profile_hint = _owned_profile_hint(pretty_hint),
            size_bytes = size_bytes,
        )
    elseif kind == "MPPDecomposition"
        (
            kind = kind,
            schema_version = schema,
            n_lines = haskey(obj, "lines") ? length(obj["lines"]) : missing,
            n_summands = haskey(obj, "summands") ? length(obj["summands"]) : missing,
            has_weights = haskey(obj, "weights"),
            box_dim = haskey(obj, "box") && haskey(obj["box"], "lo") ? length(obj["box"]["lo"]) : missing,
            profile_hint = _owned_profile_hint(pretty_hint),
            size_bytes = size_bytes,
        )
    elseif kind == "MPPImage"
        (
            kind = kind,
            schema_version = schema,
            nx = haskey(obj, "xgrid") ? length(obj["xgrid"]) : missing,
            ny = haskey(obj, "ygrid") ? length(obj["ygrid"]) : missing,
            sigma = haskey(obj, "sigma") ? Float64(obj["sigma"]) : missing,
            has_decomp = haskey(obj, "decomp"),
            profile_hint = _owned_profile_hint(pretty_hint),
            size_bytes = size_bytes,
        )
    elseif kind == "PointCloud" || kind == "GraphData" || kind == "ImageNd" ||
           kind == "EmbeddedPlanarGraph2D" || kind == "GradedComplex" ||
           kind == "MultiCriticalGradedComplex" || kind == "SimplexTreeMulti"
        (
            kind = kind,
            schema_version = schema,
            data_kind = kind,
            profile_hint = _owned_profile_hint(pretty_hint),
            size_bytes = size_bytes,
        )
    elseif haskey(obj, "dataset") && haskey(obj, "spec")
        dataset = obj["dataset"]
        (
            kind = "PipelineJSON",
            schema_version = schema,
            data_kind = haskey(dataset, "kind") ? String(dataset["kind"]) : "unknown",
            has_pipeline_options = haskey(obj, "pipeline_options"),
            has_degree = haskey(obj, "degree"),
            profile_hint = _owned_profile_hint(pretty_hint),
            size_bytes = size_bytes,
        )
    else
        (
            kind = kind,
            schema_version = schema,
            size_bytes = size_bytes,
        )
    end

    return _artifact_summary(path, report)
end

"""
    json_artifact_summary(path) -> JSONArtifactSummary

Owner-local alias for [`inspect_json`](@ref).

Use this when you want the discoverable Serialization-owned summary entrypoint
rather than the generic inspection name. This remains a cheap-first inspection
helper: it does not perform full schema validation.
"""
@inline json_artifact_summary(path::AbstractString) = inspect_json(path)

"""
    encoding_json_summary(path) -> JSONArtifactSummary
    dataset_json_summary(path) -> JSONArtifactSummary
    pipeline_json_summary(path) -> JSONArtifactSummary
    flange_json_summary(path) -> JSONArtifactSummary
    pl_fringe_json_summary(path) -> JSONArtifactSummary
    mpp_decomposition_json_summary(path) -> JSONArtifactSummary
    mpp_image_json_summary(path) -> JSONArtifactSummary

Owner-local summary helpers for the main TamerOp-owned JSON artifact
families.

These are cheap inspection helpers: they validate only the artifact family
contract (by kind) and otherwise return the same summary object as
[`inspect_json`](@ref). Use them when you already know which owned family you
expect and want a notebook-friendly summary before deciding whether a strict
`check_*_json` or a full `load_*_json` call is warranted.
"""
@inline encoding_json_summary(path::AbstractString) =
    _expect_artifact_kind(inspect_json(path), "FiniteEncodingFringe", :encoding_json_summary)
@inline dataset_json_summary(path::AbstractString) =
    _expect_artifact_kind(inspect_json(path),
                          ("PointCloud", "GraphData", "ImageNd", "EmbeddedPlanarGraph2D", "GradedComplex", "MultiCriticalGradedComplex", "SimplexTreeMulti"),
                          :dataset_json_summary)
@inline pipeline_json_summary(path::AbstractString) =
    _expect_artifact_kind(inspect_json(path), "PipelineJSON", :pipeline_json_summary)
@inline flange_json_summary(path::AbstractString) =
    _expect_artifact_kind(inspect_json(path), "FlangeZn", :flange_json_summary)
@inline pl_fringe_json_summary(path::AbstractString) =
    _expect_artifact_kind(inspect_json(path), "PLFringe", :pl_fringe_json_summary)
@inline mpp_decomposition_json_summary(path::AbstractString) =
    _expect_artifact_kind(inspect_json(path), "MPPDecomposition", :mpp_decomposition_json_summary)
@inline mpp_image_json_summary(path::AbstractString) =
    _expect_artifact_kind(inspect_json(path), "MPPImage", :mpp_image_json_summary)

"""
    feature_metadata_summary(meta_or_path) -> JSONArtifactSummary

Summarize a TamerOp feature-metadata header, either from a metadata object
or from a JSON file containing the header directly (or under a top-level
`"metadata"` key).

This is the cheap-first inspection path for feature headers. It reports the
declared feature schema and format without loading any downstream feature data.
Use this before trusting a feature artifact, and call
[`validate_feature_metadata_schema`](@ref) when you need strict header
validation.
"""
function feature_metadata_summary(meta_or_path)
    if meta_or_path isa AbstractString
        obj = _json_read(meta_or_path)
        meta = haskey(obj, "metadata") ? obj["metadata"] : obj
        report = (
            kind = haskey(meta, "kind") ? String(meta["kind"]) : "unknown",
            schema_version = haskey(meta, "schema_version") ? String(meta["schema_version"]) : nothing,
            format = haskey(meta, "format") ? String(meta["format"]) : nothing,
            size_bytes = _artifact_size_or_nothing(meta_or_path),
        )
        return _artifact_summary(meta_or_path, report)
    end
    meta = meta_or_path
    report = (
        kind = haskey(meta, "kind") ? String(meta["kind"]) : "unknown",
        schema_version = haskey(meta, "schema_version") ? String(meta["schema_version"]) : nothing,
        format = haskey(meta, "format") ? String(meta["format"]) : nothing,
        size_bytes = nothing,
    )
    return _artifact_summary(nothing, report)
end

"""
    check_feature_metadata_json(path_or_meta; throw=false) -> NamedTuple

Validate a TamerOp feature-metadata header, either from a metadata object
or from a JSON file containing the header directly (or under a top-level
`"metadata"` key).

This is the explicit validation companion to [`feature_metadata_summary`](@ref).
Use the summary helper for cheap-first inspection and call this validator when
you need strict confirmation that the feature header satisfies the owned schema
contract.
"""
function check_feature_metadata_json(path_or_meta; throw::Bool=false)
    issues = String[]
    summary = nothing
    path = path_or_meta isa AbstractString ? String(path_or_meta) : nothing
    try
        summary = feature_metadata_summary(path_or_meta)
        if path_or_meta isa AbstractString
            obj = _json_read(path_or_meta)
            meta = haskey(obj, "metadata") ? obj["metadata"] : obj
            validate_feature_metadata_schema(meta)
        else
            validate_feature_metadata_schema(path_or_meta)
        end
    catch e
        push!(issues, sprint(showerror, e))
    end
    valid = isempty(issues)
    throw && !valid && _throw_invalid_serialization(:check_feature_metadata_json, issues)
    return _serialization_report(
        :check_feature_metadata_json,
        valid;
        path=path,
        summary=summary,
        _validation_summary_fields(summary)...,
        issues=issues,
    )
end

@inline function _validation_summary_fields(summary::Union{Nothing,JSONArtifactSummary})
    if summary === nothing
        return (; artifact_kind=nothing,
                schema_version=nothing,
                artifact_field=nothing,
                artifact_poset_kind=nothing,
                has_encoding_map=nothing,
                has_dense_leq=nothing,
                size_bytes=nothing)
    end
    return (; artifact_kind=artifact_kind(summary),
            schema_version=schema_version(summary),
            artifact_field=artifact_field(summary),
            artifact_poset_kind=artifact_poset_kind(summary),
            has_encoding_map=has_encoding_map(summary),
            has_dense_leq=has_dense_leq(summary),
            size_bytes=artifact_size_bytes(summary))
end

function _check_owned_json(path::AbstractString,
                           expected_kinds,
                           loader::Function,
                           fname::Symbol;
                           throw::Bool=false)
    issues = String[]
    summary = nothing
    try
        summary = inspect_json(path)
    catch e
        push!(issues, sprint(showerror, e))
    end
    if summary !== nothing
        kinds = expected_kinds isa Tuple ? expected_kinds : (expected_kinds,)
        kinds_str = join(repr.(kinds), " or ")
        artifact_kind(summary) in kinds ||
            push!(issues, "expected artifact kind $(kinds_str), got $(repr(artifact_kind(summary))).")
    end
    if isempty(issues)
        try
            loader()
        catch e
            push!(issues, sprint(showerror, e))
        end
    end
    valid = isempty(issues)
    throw && !valid && _throw_invalid_serialization(fname, issues)
    return _serialization_report(fname, valid;
                                 path=String(path),
                                 summary=summary,
                                 _validation_summary_fields(summary)...,
                                 issues=issues)
end

"""
    check_dataset_json(path; throw=false) -> NamedTuple
    check_pipeline_json(path; throw=false) -> NamedTuple
    check_encoding_json(path; throw=false) -> NamedTuple
    check_flange_json(path; throw=false) -> NamedTuple
    check_pl_fringe_json(path; throw=false) -> NamedTuple
    check_mpp_decomposition_json(path; throw=false) -> NamedTuple
    check_mpp_image_json(path; throw=false) -> NamedTuple

Validate the stable TamerOp-owned JSON artifact families.

These helpers are the preferred notebook-friendly validation surface for
artifact schemas and required-key contracts. They return structured reports by
default and raise `ArgumentError` only when `throw=true`.

Use the corresponding `*_json_summary` helper when you want a cheap family
check first, and use these validators when you want strict confirmation that an
artifact satisfies the owned schema before calling `load_*_json`.
"""
@inline function check_dataset_json(path::AbstractString; throw::Bool=false)
    return _check_owned_json(path,
                             ("PointCloud", "GraphData", "ImageNd", "EmbeddedPlanarGraph2D", "GradedComplex", "MultiCriticalGradedComplex", "SimplexTreeMulti"),
                             () -> load_dataset_json(path),
                             :check_dataset_json;
                             throw=throw)
end

@inline function _check_serialization_option(value,
                                             validator::Function,
                                             fname::Symbol;
                                             normalized=nothing,
                                             throw::Bool=false)
    issues = String[]
    try
        validator(value)
    catch e
        push!(issues, sprint(showerror, e))
    end
    valid = isempty(issues)
    throw && !valid && _throw_invalid_serialization(fname, issues)
    return _serialization_report(fname, valid; input=value, normalized=normalized, issues=issues)
end

"""
    check_json_save_profile(profile; throw=false) -> NamedTuple
    check_encoding_save_profile(profile; throw=false) -> NamedTuple
    check_include_leq_option(x; throw=false) -> NamedTuple
    check_serialization_validation_mode(x; throw=false) -> NamedTuple
    check_encoding_output_mode(x; throw=false) -> NamedTuple

Validate the strict option contracts used by Serialization save/load entrypoints.

These helpers are the cheap-first way to confirm keyword values before they are
threaded into heavier save/load calls. They return structured reports by
default and raise `ArgumentError` only when `throw=true`.

`check_json_save_profile` validates the generic owned-artifact save contract
(`:compact` / `:debug`). `check_encoding_save_profile` validates the richer
encoding-only profile contract (`:compact` / `:portable` / `:debug`).
"""
@inline function check_json_save_profile(profile; throw::Bool=false)
    normalized = let
        try
            _resolve_owned_json_save_profile(Symbol(profile))
        catch
            nothing
        end
    end
    return _check_serialization_option(
        profile,
        x -> _resolve_owned_json_save_profile(Symbol(x)),
        :check_json_save_profile;
        normalized=normalized,
        throw=throw,
    )
end

@inline function check_encoding_save_profile(profile; throw::Bool=false)
    normalized = let
        try
            _resolve_encoding_save_profile(Symbol(profile))
        catch
            nothing
        end
    end
    return _check_serialization_option(
        profile,
        x -> _resolve_encoding_save_profile(Symbol(x)),
        :check_encoding_save_profile;
        normalized=normalized,
        throw=throw,
    )
end

@inline function check_include_leq_option(x; throw::Bool=false)
    normalized = (x === :auto || x isa Bool) ? x : nothing
    return _check_serialization_option(
        x,
        value -> ((value === :auto || value isa Bool) ||
                  error("include_leq must be :auto, true, or false.")),
        :check_include_leq_option;
        normalized=normalized,
        throw=throw,
    )
end

@inline function check_serialization_validation_mode(x; throw::Bool=false)
    normalized = let
        try
            _resolve_validation_mode(Symbol(x))
        catch
            nothing
        end
    end
    return _check_serialization_option(
        x,
        value -> _resolve_validation_mode(Symbol(value)),
        :check_serialization_validation_mode;
        normalized=normalized,
        throw=throw,
    )
end

@inline function check_encoding_output_mode(x; throw::Bool=false)
    normalized = let
        try
            _resolve_encoding_output_mode(Symbol(x))
        catch
            nothing
        end
    end
    return _check_serialization_option(
        x,
        value -> _resolve_encoding_output_mode(Symbol(value)),
        :check_encoding_output_mode;
        normalized=normalized,
        throw=throw,
    )
end

@inline function check_pipeline_json(path::AbstractString; throw::Bool=false)
    return _check_owned_json(path,
                             "PipelineJSON",
                             () -> load_pipeline_json(path),
                             :check_pipeline_json;
                             throw=throw)
end

@inline function check_encoding_json(path::AbstractString; throw::Bool=false)
    return _check_owned_json(path,
                             "FiniteEncodingFringe",
                             () -> _load_encoding_json_v1(read(path); output=:fringe, validation=:strict),
                             :check_encoding_json;
                             throw=throw)
end

@inline function check_flange_json(path::AbstractString; throw::Bool=false)
    return _check_owned_json(path,
                             "FlangeZn",
                             () -> load_flange_json(path),
                             :check_flange_json;
                             throw=throw)
end

@inline function check_pl_fringe_json(path::AbstractString; throw::Bool=false)
    return _check_owned_json(path,
                             "PLFringe",
                             () -> load_pl_fringe_json(path),
                             :check_pl_fringe_json;
                             throw=throw)
end

@inline function check_mpp_decomposition_json(path::AbstractString; throw::Bool=false)
    return _check_owned_json(path,
                             "MPPDecomposition",
                             () -> load_mpp_decomposition_json(path),
                             :check_mpp_decomposition_json;
                             throw=throw)
end

@inline function check_mpp_image_json(path::AbstractString; throw::Bool=false)
    return _check_owned_json(path,
                             "MPPImage",
                             () -> load_mpp_image_json(path),
                             :check_mpp_image_json;
                             throw=throw)
end

struct _PointCloudColumnarJSON
    kind::String
    layout::String
    n::Int
    d::Int
    points_flat::Vector{Float64}
end
JSON3.StructTypes.StructType(::Type{_PointCloudColumnarJSON}) = JSON3.StructTypes.Struct()

struct _GraphDataColumnarJSON
    kind::String
    layout::String
    n::Int
    edges_u::Vector{Int}
    edges_v::Vector{Int}
    coords_dim::Union{Nothing,Int}
    coords_flat::Union{Nothing,Vector{Float64}}
    weights::Union{Nothing,Vector{Float64}}
end
JSON3.StructTypes.StructType(::Type{_GraphDataColumnarJSON}) = JSON3.StructTypes.Struct()

const _DATASET_COLUMN_LAYOUT = "column_major_v2"

@inline function _require_dataset_layout(layout::AbstractString, kind::AbstractString)
    layout == _DATASET_COLUMN_LAYOUT ||
        error("$kind JSON missing canonical `layout=\"$(_DATASET_COLUMN_LAYOUT)\"` payload.")
    return nothing
end

@inline function _pointcloud_from_flat(n::Int, d::Int, flat::Vector{Float64})
    length(flat) == n * d || error("PointCloud points_flat length mismatch.")
    pts = reshape(flat, n, d)
    return pts isa Matrix{Float64} ? PointCloud(pts; copy=false) : PointCloud(Matrix{Float64}(pts))
end

@inline function _coords_from_flat(n::Int, d::Int, flat::Vector{Float64})
    d >= 0 || error("GraphData coords_dim must be nonnegative.")
    d == 0 && return Matrix{Float64}(undef, n, 0)
    length(flat) == n * d || error("GraphData coords_flat length mismatch.")
    coords = reshape(flat, n, d)
    return coords isa Matrix{Float64} ? coords : Matrix{Float64}(coords)
end

@inline function _coords_from_flat_rowmajor(n::Int, d::Int, flat::Vector{Float64})
    d >= 0 || error("row-major coords_dim must be nonnegative.")
    d == 0 && return Matrix{Float64}(undef, n, 0)
    length(flat) == n * d || error("row-major coords_flat length mismatch.")
    out = Matrix{Float64}(undef, n, d)
    t = 1
    @inbounds for i in 1:n, j in 1:d
        out[i, j] = flat[t]
        t += 1
    end
    return out
end

@inline function _skip_json_ws(bytes, i::Int, n::Int)
    @inbounds while i <= n
        b = bytes[i]
        if b == 0x20 || b == 0x09 || b == 0x0a || b == 0x0d
            i += 1
        else
            break
        end
    end
    return i
end

@inline function _bytes_match(bytes, i::Int, lit::String)
    j = i
    @inbounds for c in codeunits(lit)
        j > length(bytes) && return false
        bytes[j] == c || return false
        j += 1
    end
    return true
end

@inline function _dataset_kind_from_raw(raw)
    bytes = raw isa AbstractVector{UInt8} ? raw : codeunits(raw)
    n = length(bytes)
    i = 1
    @inbounds while i <= n - 5
        if bytes[i] == 0x22 &&
           bytes[i + 1] == 0x6b &&
           bytes[i + 2] == 0x69 &&
           bytes[i + 3] == 0x6e &&
           bytes[i + 4] == 0x64 &&
           bytes[i + 5] == 0x22
            j = _skip_json_ws(bytes, i + 6, n)
            j <= n && bytes[j] == 0x3a || return nothing
            j = _skip_json_ws(bytes, j + 1, n)
            j <= n && bytes[j] == 0x22 || return nothing
            j += 1
            if _bytes_match(bytes, j, "PointCloud")
                return :PointCloud
            elseif _bytes_match(bytes, j, "GraphData")
                return :GraphData
            else
                return nothing
            end
        end
        i += 1
    end
    return nothing
end

@inline function _graph_from_columns(n::Int,
                                     edges_u::Vector{Int},
                                     edges_v::Vector{Int};
                                     coords_dim::Union{Nothing,Int}=nothing,
                                     coords_flat::Union{Nothing,Vector{Float64}}=nothing,
                                     weights::Union{Nothing,Vector{Float64}}=nothing)
    length(edges_u) == length(edges_v) || error("GraphData edge column lengths mismatch.")
    coords = if coords_dim === nothing || coords_flat === nothing
        nothing
    else
        _coords_from_flat(n, coords_dim, coords_flat)
    end
    return GraphData(n, edges_u, edges_v; coords=coords, weights=weights, T=Float64, copy=false)
end

@inline function _dataset_from_raw(raw; validation::Bool=true)
    kind = _dataset_kind_from_raw(raw)
    if kind === :PointCloud
        obj = try
            JSON3.read(raw, _PointCloudColumnarJSON)
        catch err
            err isa MethodError || rethrow()
            return _dataset_from_obj(JSON3.read(raw))
        end
        validation && _require_dataset_layout(obj.layout, obj.kind)
        return _pointcloud_from_flat(obj.n, obj.d, obj.points_flat)
    elseif kind === :GraphData
        obj = try
            JSON3.read(raw, _GraphDataColumnarJSON)
        catch err
            err isa MethodError || rethrow()
            return _dataset_from_obj(JSON3.read(raw))
        end
        validation && _require_dataset_layout(obj.layout, obj.kind)
        return _graph_from_columns(obj.n, obj.edges_u, obj.edges_v;
                                   coords_dim=obj.coords_dim,
                                   coords_flat=obj.coords_flat,
                                   weights=obj.weights)
    end
    return _dataset_from_obj(JSON3.read(raw))
end

@inline function _load_dataset_json_strict(path::AbstractString)
    return _dataset_from_raw(read(path); validation=true)
end

@inline function _load_dataset_json_trusted(path::AbstractString)
    return _dataset_from_raw(read(path); validation=false)
end

@inline function _load_pipeline_json_obj(obj; validation::Bool=true)
    if validation
        version = haskey(obj, "schema_version") ? Int(obj["schema_version"]) : 0
        version == PIPELINE_SCHEMA_VERSION || error("Unsupported pipeline JSON schema_version: $(version). Expected $(PIPELINE_SCHEMA_VERSION).")
        haskey(obj, "pipeline_options") || error("pipeline_options field is required in pipeline JSON.")
    end
    data = _dataset_from_obj(obj["dataset"])
    spec = _spec_from_obj(obj["spec"])
    degree = haskey(obj, "degree") ? obj["degree"] : nothing
    pipeline_opts = haskey(obj, "pipeline_options") ?
        _pipeline_options_from_obj(obj["pipeline_options"]) :
        _pipeline_options_from_spec(spec)
    return data, spec, degree, pipeline_opts
end

@inline function _load_pipeline_json_strict(path::AbstractString)
    return _load_pipeline_json_obj(_json_read(path); validation=true)
end

@inline function _load_pipeline_json_trusted(path::AbstractString)
    return _load_pipeline_json_obj(_json_read(path); validation=false)
end

@inline function _resolve_include_leq(P::AbstractPoset, include_leq::Union{Bool,Symbol})
    if include_leq === :auto
        return P isa FinitePoset
    end
    include_leq isa Bool || error("include_leq must be Bool or :auto.")
    return include_leq
end

# Typed encoding JSON schema (v1) for fast load paths.
abstract type _CoeffFieldJSON end
abstract type _PosetJSON end
abstract type _MaskJSON end
abstract type _PhiJSON end
abstract type _PiJSON end

Base.@kwdef mutable struct _QQFieldJSON <: _CoeffFieldJSON
    kind::String = "qq"
end

Base.@kwdef mutable struct _RealFieldJSON <: _CoeffFieldJSON
    kind::String = "real"
    T::String = "Float64"
    rtol::Union{Nothing,Float64} = nothing
    atol::Union{Nothing,Float64} = nothing
end

Base.@kwdef mutable struct _FpFieldJSON <: _CoeffFieldJSON
    kind::String = "fp"
    p::Int = 2
end

Base.@kwdef mutable struct _MaskPackedWordsJSON <: _MaskJSON
    kind::String = "packed_words_v1"
    nrows::Int = 0
    ncols::Int = 0
    words_per_row::Int = 0
    words::Vector{UInt64} = UInt64[]
end

Base.@kwdef mutable struct _FinitePosetJSON <: _PosetJSON
    kind::String = "FinitePoset"
    n::Int = 0
    leq::_MaskPackedWordsJSON = _MaskPackedWordsJSON()
end

Base.@kwdef mutable struct _ProductOfChainsPosetJSON <: _PosetJSON
    kind::String = "ProductOfChainsPoset"
    n::Int = 0
    sizes::Vector{Int} = Int[]
    leq::Union{Nothing,_MaskPackedWordsJSON} = nothing
end

Base.@kwdef mutable struct _GridPosetJSON <: _PosetJSON
    kind::String = "GridPoset"
    n::Int = 0
    coords::Vector{Vector{Float64}} = Vector{Vector{Float64}}()
    leq::Union{Nothing,_MaskPackedWordsJSON} = nothing
end

Base.@kwdef mutable struct _ProductPosetJSON <: _PosetJSON
    kind::String = "ProductPoset"
    n::Int = 0
    left::Union{Nothing,_PosetJSON} = nothing
    right::Union{Nothing,_PosetJSON} = nothing
    leq::Union{Nothing,_MaskPackedWordsJSON} = nothing
end

Base.@kwdef mutable struct _SignaturePosetJSON <: _PosetJSON
    kind::String = "SignaturePoset"
    n::Int = 0
    sig_y::_MaskPackedWordsJSON = _MaskPackedWordsJSON()
    sig_z::_MaskPackedWordsJSON = _MaskPackedWordsJSON()
    leq::Union{Nothing,_MaskPackedWordsJSON} = nothing
end

Base.@kwdef mutable struct _PhiQQChunksJSON <: _PhiJSON
    kind::String = "qq_chunks_v1"
    m::Int = 0
    k::Int = 0
    base::Int = 1_000_000_000
    num_sign::Vector{Int8} = Int8[]
    num_ptr::Vector{Int} = Int[]
    num_chunks::Vector{UInt32} = UInt32[]
    den_ptr::Vector{Int} = Int[]
    den_chunks::Vector{UInt32} = UInt32[]
end

Base.@kwdef mutable struct _PhiFpFlatJSON <: _PhiJSON
    kind::String = "fp_flat_v1"
    m::Int = 0
    k::Int = 0
    data::Vector{Int} = Int[]
end

Base.@kwdef mutable struct _PhiRealFlatJSON <: _PhiJSON
    kind::String = "real_flat_v1"
    m::Int = 0
    k::Int = 0
    data::Vector{Float64} = Float64[]
end

Base.@kwdef mutable struct _FaceGeneratorJSON
    b::Vector{Int} = Int[]
    tau::Vector{Int} = Int[]
end

Base.@kwdef mutable struct _GridEncodingMapJSON <: _PiJSON
    kind::String = "GridEncodingMap"
    coords::Vector{Vector{Float64}} = Vector{Vector{Float64}}()
    orientation::Vector{Int} = Int[]
end

Base.@kwdef mutable struct _ZnEncodingMapJSON <: _PiJSON
    kind::String = "ZnEncodingMap"
    n::Int = 0
    coords::Vector{Vector{Int}} = Vector{Vector{Int}}()
    sig_y::_MaskPackedWordsJSON = _MaskPackedWordsJSON()
    sig_z::_MaskPackedWordsJSON = _MaskPackedWordsJSON()
    reps::Vector{Vector{Int}} = Vector{Vector{Int}}()
    flats::Vector{_FaceGeneratorJSON} = _FaceGeneratorJSON[]
    injectives::Vector{_FaceGeneratorJSON} = _FaceGeneratorJSON[]
    cell_shape::Union{Nothing,Vector{Int}} = nothing
    cell_strides::Union{Nothing,Vector{Int}} = nothing
    cell_to_region::Union{Nothing,Vector{Int}} = nothing
end

Base.@kwdef mutable struct _PLEncodingMapBoxesJSON <: _PiJSON
    kind::String = "PLEncodingMapBoxes"
    n::Int = 0
    coords::Vector{Vector{Float64}} = Vector{Vector{Float64}}()
    sig_y::_MaskPackedWordsJSON = _MaskPackedWordsJSON()
    sig_z::_MaskPackedWordsJSON = _MaskPackedWordsJSON()
    reps::Vector{Vector{Float64}} = Vector{Vector{Float64}}()
    Ups::Vector{Vector{Float64}} = Vector{Vector{Float64}}()
    Downs::Vector{Vector{Float64}} = Vector{Vector{Float64}}()
    cell_shape::Vector{Int} = Int[]
    cell_strides::Vector{Int} = Int[]
    cell_to_region::Vector{Int} = Int[]
    coord_flags::Vector{Vector{UInt8}} = Vector{Vector{UInt8}}()
    axis_is_uniform::Vector{Bool} = Bool[]
    axis_step::Vector{Float64} = Float64[]
    axis_min::Vector{Float64} = Float64[]
end

Base.@kwdef mutable struct _FiniteEncodingFringeJSONV1
    kind::String = ""
    schema_version::Int = 0
    poset::_PosetJSON = _FinitePosetJSON()
    U::_MaskJSON = _MaskPackedWordsJSON()
    D::_MaskJSON = _MaskPackedWordsJSON()
    coeff_field::_CoeffFieldJSON = _QQFieldJSON()
    phi::_PhiJSON = _PhiQQChunksJSON()
    pi::Union{Nothing,_PiJSON} = nothing
end

Base.@kwdef mutable struct _CanonicalPLHPolyJSON
    A::Any = nothing
    b::Any = nothing
    strict_mask::Any = nothing
    strict_eps::Any = nothing
end

Base.@kwdef mutable struct _CanonicalPLUnionJSON
    n::Int = 0
    parts::Vector{_CanonicalPLHPolyJSON} = _CanonicalPLHPolyJSON[]
end

Base.@kwdef mutable struct _CanonicalPLFringeJSON
    kind::String = ""
    schema_version::Int = 0
    n::Int = 0
    ups::Vector{_CanonicalPLUnionJSON} = _CanonicalPLUnionJSON[]
    downs::Vector{_CanonicalPLUnionJSON} = _CanonicalPLUnionJSON[]
    phi::Any = nothing
end

JSON3.StructTypes.StructType(::Type{_CoeffFieldJSON}) = JSON3.StructTypes.AbstractType()
JSON3.StructTypes.subtypekey(::Type{_CoeffFieldJSON}) = :kind
JSON3.StructTypes.subtypes(::Type{_CoeffFieldJSON}) = (
    qq = _QQFieldJSON,
    real = _RealFieldJSON,
    fp = _FpFieldJSON,
)
JSON3.StructTypes.StructType(::Type{_QQFieldJSON}) = JSON3.StructTypes.Mutable()
JSON3.StructTypes.StructType(::Type{_RealFieldJSON}) = JSON3.StructTypes.Mutable()
JSON3.StructTypes.StructType(::Type{_FpFieldJSON}) = JSON3.StructTypes.Mutable()

JSON3.StructTypes.StructType(::Type{_PosetJSON}) = JSON3.StructTypes.AbstractType()
JSON3.StructTypes.subtypekey(::Type{_PosetJSON}) = :kind
JSON3.StructTypes.subtypes(::Type{_PosetJSON}) = (
    FinitePoset = _FinitePosetJSON,
    ProductOfChainsPoset = _ProductOfChainsPosetJSON,
    GridPoset = _GridPosetJSON,
    ProductPoset = _ProductPosetJSON,
    SignaturePoset = _SignaturePosetJSON,
)
JSON3.StructTypes.StructType(::Type{_FinitePosetJSON}) = JSON3.StructTypes.Mutable()
JSON3.StructTypes.StructType(::Type{_ProductOfChainsPosetJSON}) = JSON3.StructTypes.Mutable()
JSON3.StructTypes.StructType(::Type{_GridPosetJSON}) = JSON3.StructTypes.Mutable()
JSON3.StructTypes.StructType(::Type{_ProductPosetJSON}) = JSON3.StructTypes.Mutable()
JSON3.StructTypes.StructType(::Type{_SignaturePosetJSON}) = JSON3.StructTypes.Mutable()

JSON3.StructTypes.StructType(::Type{_MaskJSON}) = JSON3.StructTypes.AbstractType()
JSON3.StructTypes.subtypekey(::Type{_MaskJSON}) = :kind
JSON3.StructTypes.subtypes(::Type{_MaskJSON}) = (
    packed_words_v1 = _MaskPackedWordsJSON,
)
JSON3.StructTypes.StructType(::Type{_MaskPackedWordsJSON}) = JSON3.StructTypes.Mutable()

JSON3.StructTypes.StructType(::Type{_PhiJSON}) = JSON3.StructTypes.AbstractType()
JSON3.StructTypes.subtypekey(::Type{_PhiJSON}) = :kind
JSON3.StructTypes.subtypes(::Type{_PhiJSON}) = (
    qq_chunks_v1 = _PhiQQChunksJSON,
    fp_flat_v1 = _PhiFpFlatJSON,
    real_flat_v1 = _PhiRealFlatJSON,
)
JSON3.StructTypes.StructType(::Type{_PhiQQChunksJSON}) = JSON3.StructTypes.Mutable()
JSON3.StructTypes.StructType(::Type{_PhiFpFlatJSON}) = JSON3.StructTypes.Mutable()
JSON3.StructTypes.StructType(::Type{_PhiRealFlatJSON}) = JSON3.StructTypes.Mutable()

JSON3.StructTypes.StructType(::Type{_PiJSON}) = JSON3.StructTypes.AbstractType()
JSON3.StructTypes.subtypekey(::Type{_PiJSON}) = :kind
JSON3.StructTypes.subtypes(::Type{_PiJSON}) = (
    GridEncodingMap = _GridEncodingMapJSON,
    ZnEncodingMap = _ZnEncodingMapJSON,
    PLEncodingMapBoxes = _PLEncodingMapBoxesJSON,
)
JSON3.StructTypes.StructType(::Type{_FaceGeneratorJSON}) = JSON3.StructTypes.Mutable()
JSON3.StructTypes.StructType(::Type{_GridEncodingMapJSON}) = JSON3.StructTypes.Mutable()
JSON3.StructTypes.StructType(::Type{_ZnEncodingMapJSON}) = JSON3.StructTypes.Mutable()
JSON3.StructTypes.StructType(::Type{_PLEncodingMapBoxesJSON}) = JSON3.StructTypes.Mutable()
JSON3.StructTypes.StructType(::Type{_FiniteEncodingFringeJSONV1}) = JSON3.StructTypes.Mutable()
JSON3.StructTypes.StructType(::Type{_CanonicalPLHPolyJSON}) = JSON3.StructTypes.Mutable()
JSON3.StructTypes.StructType(::Type{_CanonicalPLUnionJSON}) = JSON3.StructTypes.Mutable()
JSON3.StructTypes.StructType(::Type{_CanonicalPLFringeJSON}) = JSON3.StructTypes.Mutable()

@inline _field_from_typed(obj::_QQFieldJSON) = QQField()
@inline function _field_from_typed(obj::_RealFieldJSON)
    Tname = obj.T
    T = Tname == "Float64" ? Float64 :
        Tname == "Float32" ? Float32 :
        error("Unsupported real field type in JSON: $(Tname)")
    rtol = obj.rtol === nothing ? sqrt(eps(T)) : T(obj.rtol)
    atol = obj.atol === nothing ? zero(T) : T(obj.atol)
    return RealField(T; rtol=rtol, atol=atol)
end
@inline _field_from_typed(obj::_FpFieldJSON) = PrimeField(obj.p)

@inline function _mask_lastword(ncols::Int)::UInt64
    rem = ncols & 63
    return rem == 0 ? typemax(UInt64) : (UInt64(1) << rem) - 1
end

@inline function _csv_escape(x)
    s = string(x)
    if occursin(',', s) || occursin('"', s)
        s = replace(s, '"' => "\"\"")
        return "\"" * s * "\""
    end
    return s
end

"""
    _write_feature_csv_wide(path, X, names, ids; ids_col=:id, include_ids=true)

Internal CSV fallback writer for wide feature tables.
Rows are samples and columns are features.
"""
function _write_feature_csv_wide(path::AbstractString,
                                 X::AbstractMatrix,
                                 names::AbstractVector,
                                 ids::AbstractVector{<:AbstractString};
                                 ids_col::Symbol=:id,
                                 include_ids::Bool=true)
    nfeat = size(X, 2)
    length(names) == nfeat || throw(ArgumentError("_write_feature_csv_wide: feature-name count mismatch"))
    length(ids) == size(X, 1) || throw(ArgumentError("_write_feature_csv_wide: id count mismatch"))
    open(path, "w") do io
        hdr = include_ids ? Any[ids_col; names] : Any[names...]
        println(io, join(_csv_escape.(hdr), ","))
        @inbounds for i in 1:size(X, 1)
            row = Vector{Any}(undef, nfeat + (include_ids ? 1 : 0))
            t = 1
            if include_ids
                row[t] = ids[i]
                t += 1
            end
            for j in 1:nfeat
                row[t] = X[i, j]
                t += 1
            end
            println(io, join(_csv_escape.(row), ","))
        end
    end
    return path
end

"""
    _write_feature_csv_long(path, X, names, ids; include_sample_index=true)

Internal CSV fallback writer for long feature tables.
"""
function _write_feature_csv_long(path::AbstractString,
                                 X::AbstractMatrix,
                                 names::AbstractVector,
                                 ids::AbstractVector{<:AbstractString};
                                 include_sample_index::Bool=true)
    nfeat = size(X, 2)
    length(names) == nfeat || throw(ArgumentError("_write_feature_csv_long: feature-name count mismatch"))
    length(ids) == size(X, 1) || throw(ArgumentError("_write_feature_csv_long: id count mismatch"))
    open(path, "w") do io
        if include_sample_index
            println(io, "id,feature,value,sample_index")
        else
            println(io, "id,feature,value")
        end
        @inbounds for i in 1:size(X, 1)
            idi = ids[i]
            for j in 1:nfeat
                if include_sample_index
                    vals = (idi, names[j], X[i, j], i)
                else
                    vals = (idi, names[j], X[i, j])
                end
                println(io, join(_csv_escape.(vals), ","))
            end
        end
    end
    return path
end

