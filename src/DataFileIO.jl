# =============================================================================
# DataFileIO.jl
#
# File-oriented dataset ingestion adapters.
# Converts on-disk formats into typed ingestion objects used by DataIngestion.
# =============================================================================

module DataFileIO

using ..DataTypes: PointCloud, GraphData, ImageNd, GradedComplex
using ..Options: ConstructionOptions, DataFileOptions
import ..ChainComplexes: describe
import ..Serialization

const _MISSING_STRINGS = Set(["", "na", "nan", "null", "missing"])
const _TABLE_FORMATS = (:csv, :tsv, :txt)
const _RIPSER_FORMATS = (
    :ripser_point_cloud,
    :ripser_distance,
    :ripser_lower_distance,
    :ripser_upper_distance,
    :ripser_sparse_triplet,
    :ripser_binary_lower_distance,
    :ripser_lower_distance_streaming,
)

function data_file_summary end
function path end
function format end
function kind end
function candidate_kinds end
function schema_version end
function header_used end
function nrows end
function ncols end
function columns end
function sample_rows end
function detail end
function resolved_format end
function resolved_kind end
function is_table_file end
function is_dataset_json end
function is_ripser_file end
function issues end
function resolved_columns end
function ok end
function inspection end
function validation_kind end
function artifact end
function data_file_validation_summary end
function load_data_validation_summary end
function table_column_validation_summary end
function is_ambiguous end
function requires_explicit_kind end
function suggested_kind end
function check_data_file end
function check_load_data end
function check_table_columns end

"""
    DelimitedTableInspection

Typed detail object attached to [`DataFileInspectionSummary`](@ref) for
delimited-table files.

This stores the canonical table facts produced by [`inspect_data_file`](@ref):
header usage, row/column counts, column names, and the sampled raw rows.

Example
-------
```julia
info = inspect_data_file("points.csv")
tbl = detail(info)
describe(tbl)
columns(tbl)
```
"""
struct DelimitedTableInspection
    header_used::Bool
    nrows::Int
    ncols::Int
    columns::Tuple
    sample_rows::Vector{Vector{String}}
end

"""
    DatasetFileInspection

Typed detail object attached to [`DataFileInspectionSummary`](@ref) for
canonical dataset JSON files.

Use [`schema_version`](@ref) on the parent summary for the primary schema
contract, and inspect `detail(summary)` when you need the richer serialization
artifact metadata returned by [`Serialization.inspect_json`](@ref).

Example
-------
```julia
info = inspect_data_file("points.json")
json_detail = detail(info)
describe(json_detail)
artifact(json_detail)
```
"""
struct DatasetFileInspection{D}
    schema_version
    artifact::D
end

"""
    DataFileInspectionSummary

Typed summary returned by [`inspect_data_file`](@ref).

This is the preferred inspection object for file-based dataset loading. It
replaces the previous shape-varying `NamedTuple` surface with one stable owner
container.

Use the semantic accessors
[`path`](@ref), [`format`](@ref), [`kind`](@ref),
[`candidate_kinds`](@ref), [`schema_version`](@ref),
[`header_used`](@ref), [`nrows`](@ref), [`ncols`](@ref),
[`columns`](@ref), [`sample_rows`](@ref), [`detail`](@ref),
[`resolved_format`](@ref), [`resolved_kind`](@ref),
[`is_table_file`](@ref), [`is_dataset_json`](@ref), and
[`is_ripser_file`](@ref) instead of unpacking raw tuples.

Example
-------
```julia
info = inspect_data_file("points.csv")
describe(info)
candidate_kinds(info)
sample_rows(info)
```
"""
struct DataFileInspectionSummary{D}
    path::String
    format::Symbol
    kind::Symbol
    candidate_kinds::Tuple
    schema_version
    header_used::Union{Nothing,Bool}
    nrows::Union{Nothing,Int}
    ncols::Union{Nothing,Int}
    columns::Union{Nothing,Tuple}
    sample_rows::Vector{Vector{String}}
    detail::D
end

"""
    DataFileValidationSummary

Typed report returned by [`check_data_file`](@ref).

This validator checks the file-level inspection contract without materializing
the dataset itself. It is the notebook-friendly preflight companion to
[`inspect_data_file`](@ref).
"""
struct DataFileValidationSummary{I}
    ok::Bool
    path::String
    format
    kind
    issues::Vector{String}
    candidate_kinds::Tuple
    inspection::Union{Nothing,I}
    DataFileValidationSummary{I}(ok::Bool,
                                 path::String,
                                 format,
                                 kind,
                                 issues::Vector{String},
                                 candidate_kinds::Tuple,
                                 inspection::Union{Nothing,I}) where {I} =
        new{I}(ok, path, format, kind, issues, candidate_kinds, inspection)
end

DataFileValidationSummary(ok::Bool,
                          path::String,
                          format,
                          kind,
                          issues::Vector{String},
                          candidate_kinds::Tuple,
                          inspection) =
    inspection === nothing ?
    DataFileValidationSummary{Nothing}(ok, path, format, kind, issues, candidate_kinds, nothing) :
    DataFileValidationSummary{typeof(inspection)}(ok, path, format, kind, issues, candidate_kinds, inspection)

"""
    LoadDataValidationSummary

Typed report returned by [`check_load_data`](@ref).

This validator checks the canonical [`load_data`](@ref) call surface, including
table ambiguity and kind/format mismatches.
"""
struct LoadDataValidationSummary{I,C}
    ok::Bool
    path::String
    format
    kind
    issues::Vector{String}
    candidate_kinds::Tuple
    resolved_columns::C
    inspection::Union{Nothing,I}
    LoadDataValidationSummary{I,C}(ok::Bool,
                                   path::String,
                                   format,
                                   kind,
                                   issues::Vector{String},
                                   candidate_kinds::Tuple,
                                   resolved_columns::C,
                                   inspection::Union{Nothing,I}) where {I,C} =
        new{I,C}(ok, path, format, kind, issues, candidate_kinds, resolved_columns, inspection)
end

LoadDataValidationSummary(ok::Bool,
                          path::String,
                          format,
                          kind,
                          issues::Vector{String},
                          candidate_kinds::Tuple,
                          resolved_columns,
                          inspection) =
    inspection === nothing ?
    LoadDataValidationSummary{Nothing,typeof(resolved_columns)}(ok, path, format, kind, issues, candidate_kinds, resolved_columns, nothing) :
    LoadDataValidationSummary{typeof(inspection),typeof(resolved_columns)}(ok, path, format, kind, issues, candidate_kinds, resolved_columns, inspection)

"""
    TableColumnValidationSummary

Typed report returned by [`check_table_columns`](@ref).

Use this when you want to validate a CSV/TSV/TXT file's column interpretation
before calling [`load_data`](@ref).
"""
struct TableColumnValidationSummary{I,C}
    ok::Bool
    path::String
    format
    kind
    issues::Vector{String}
    candidate_kinds::Tuple
    resolved_columns::C
    inspection::Union{Nothing,I}
    TableColumnValidationSummary{I,C}(ok::Bool,
                                      path::String,
                                      format,
                                      kind,
                                      issues::Vector{String},
                                      candidate_kinds::Tuple,
                                      resolved_columns::C,
                                      inspection::Union{Nothing,I}) where {I,C} =
        new{I,C}(ok, path, format, kind, issues, candidate_kinds, resolved_columns, inspection)
end

TableColumnValidationSummary(ok::Bool,
                             path::String,
                             format,
                             kind,
                             issues::Vector{String},
                             candidate_kinds::Tuple,
                             resolved_columns,
                             inspection) =
    inspection === nothing ?
    TableColumnValidationSummary{Nothing,typeof(resolved_columns)}(ok, path, format, kind, issues, candidate_kinds, resolved_columns, nothing) :
    TableColumnValidationSummary{typeof(inspection),typeof(resolved_columns)}(ok, path, format, kind, issues, candidate_kinds, resolved_columns, inspection)

@inline path(summary::Union{
    DataFileInspectionSummary,
    DataFileValidationSummary,
    LoadDataValidationSummary,
    TableColumnValidationSummary,
}) = summary.path
@inline format(summary::Union{
    DataFileInspectionSummary,
    DataFileValidationSummary,
    LoadDataValidationSummary,
    TableColumnValidationSummary,
}) = summary.format
@inline kind(summary::Union{
    DataFileInspectionSummary,
    DataFileValidationSummary,
    LoadDataValidationSummary,
    TableColumnValidationSummary,
}) = summary.kind
@inline candidate_kinds(summary::Union{
    DataFileInspectionSummary,
    DataFileValidationSummary,
    LoadDataValidationSummary,
    TableColumnValidationSummary,
}) = summary.candidate_kinds
@inline schema_version(summary::DataFileInspectionSummary) = summary.schema_version
@inline header_used(summary::DataFileInspectionSummary) = summary.header_used
@inline nrows(summary::DataFileInspectionSummary) = summary.nrows
@inline ncols(summary::DataFileInspectionSummary) = summary.ncols
@inline columns(summary::DataFileInspectionSummary) = summary.columns
@inline sample_rows(summary::DataFileInspectionSummary) = summary.sample_rows
@inline detail(summary::DataFileInspectionSummary) = summary.detail
@inline resolved_format(summary::DataFileInspectionSummary) = summary.format
@inline header_used(info::DelimitedTableInspection) = info.header_used
@inline nrows(info::DelimitedTableInspection) = info.nrows
@inline ncols(info::DelimitedTableInspection) = info.ncols
@inline columns(info::DelimitedTableInspection) = info.columns
@inline sample_rows(info::DelimitedTableInspection) = info.sample_rows
@inline schema_version(info::DatasetFileInspection) = info.schema_version

"""
    artifact(info::DatasetFileInspection)

Return the underlying serialization-artifact inspection object attached to a
dataset JSON inspection detail.
"""
@inline artifact(info::DatasetFileInspection) = info.artifact

"""
    ok(summary)

Return whether a `DataFileIO` validation summary reports a valid contract.
"""
@inline ok(summary::Union{
    DataFileValidationSummary,
    LoadDataValidationSummary,
    TableColumnValidationSummary,
}) = summary.ok

"""
    inspection(summary)

Return the typed inspection summary attached to a `DataFileIO` validation
report, or `nothing` if validation failed before inspection completed.
"""
@inline inspection(summary::Union{
    DataFileValidationSummary,
    LoadDataValidationSummary,
    TableColumnValidationSummary,
}) = summary.inspection
@inline issues(summary::Union{
    DataFileValidationSummary,
    LoadDataValidationSummary,
    TableColumnValidationSummary,
}) = summary.issues
@inline resolved_columns(summary::Union{LoadDataValidationSummary,TableColumnValidationSummary}) =
    summary.resolved_columns

"""
    validation_kind(summary)

Return the stable validation-report kind tag for a `DataFileIO` validation
summary.
"""
@inline validation_kind(::DataFileValidationSummary) = :data_file_validation
@inline validation_kind(::LoadDataValidationSummary) = :load_data_validation
@inline validation_kind(::TableColumnValidationSummary) = :table_column_validation
@inline is_table_file(summary::DataFileInspectionSummary) = summary.format in _TABLE_FORMATS
@inline is_dataset_json(summary::DataFileInspectionSummary) = summary.format == :dataset_json
@inline is_ripser_file(summary::DataFileInspectionSummary) = summary.format in _RIPSER_FORMATS

"""
    is_ambiguous(summary::DataFileInspectionSummary)

Return `true` when a file inspection leaves more than one plausible table data
kind.
"""
@inline is_ambiguous(summary::DataFileInspectionSummary) =
    is_table_file(summary) && length(summary.candidate_kinds) > 1

"""
    requires_explicit_kind(summary::DataFileInspectionSummary)

Return whether the inspected file requires an explicit `kind=...` for
[`load_data`](@ref).
"""
@inline requires_explicit_kind(summary::DataFileInspectionSummary) = is_table_file(summary)
@inline function resolved_kind(summary::DataFileInspectionSummary)
    if summary.kind != :table
        return summary.kind
    elseif length(summary.candidate_kinds) == 1
        return summary.candidate_kinds[1]
    end
    return nothing
end

"""
    suggested_kind(summary::DataFileInspectionSummary)

Return the unambiguous suggested data kind from an inspection summary, or
`nothing` when no honest single-kind suggestion is available.
"""
@inline suggested_kind(summary::DataFileInspectionSummary) = resolved_kind(summary)

@inline function _table_candidate_kinds(nrows::Int, ncols::Int)
    cands = Symbol[:point_cloud]
    ncols >= 2 && push!(cands, :graph)
    push!(cands, :image)
    nrows == ncols && push!(cands, :distance_matrix)
    return Tuple(cands)
end

@inline function _throw_invalid_datafile(kind_name::AbstractString, issues_vec::Vector{String})
    msg = isempty(issues_vec) ? "invalid $(kind_name)" : join(issues_vec, "; ")
    throw(ArgumentError("$(kind_name): $(msg)"))
end

@inline _error_issue(err) = sprint(showerror, err)

@inline function _validation_issue_fields(summary)
    isempty(summary.issues) && return ""
    return string("\n  issues: ", join(summary.issues, "\n          "))
end

function describe(info::DataFileInspectionSummary)
    return (
        kind = :data_file_inspection,
        path = info.path,
        format = info.format,
        file_kind = info.kind,
        resolved_kind = resolved_kind(info),
        candidate_kinds = info.candidate_kinds,
        is_ambiguous = is_ambiguous(info),
        requires_explicit_kind = requires_explicit_kind(info),
        suggested_kind = suggested_kind(info),
        schema_version = info.schema_version,
        header_used = info.header_used,
        nrows = info.nrows,
        ncols = info.ncols,
        columns = info.columns,
        sample_rows = info.sample_rows,
        is_table_file = is_table_file(info),
        is_dataset_json = is_dataset_json(info),
        is_ripser_file = is_ripser_file(info),
    )
end

function describe(info::DelimitedTableInspection)
    return (
        kind = :delimited_table_inspection,
        header_used = info.header_used,
        nrows = info.nrows,
        ncols = info.ncols,
        columns = info.columns,
        sample_rows = info.sample_rows,
    )
end

function describe(info::DatasetFileInspection)
    return (
        kind = :dataset_file_inspection,
        schema_version = info.schema_version,
        artifact = info.artifact,
    )
end

function describe(summary::DataFileValidationSummary)
    return (
        kind = validation_kind(summary),
        ok = summary.ok,
        path = summary.path,
        format = summary.format,
        file_kind = summary.kind,
        candidate_kinds = summary.candidate_kinds,
        issues = summary.issues,
    )
end

function describe(summary::LoadDataValidationSummary)
    return (
        kind = validation_kind(summary),
        ok = summary.ok,
        path = summary.path,
        format = summary.format,
        file_kind = summary.kind,
        candidate_kinds = summary.candidate_kinds,
        resolved_columns = summary.resolved_columns,
        issues = summary.issues,
    )
end

function describe(summary::TableColumnValidationSummary)
    return (
        kind = validation_kind(summary),
        ok = summary.ok,
        path = summary.path,
        format = summary.format,
        file_kind = summary.kind,
        candidate_kinds = summary.candidate_kinds,
        resolved_columns = summary.resolved_columns,
        issues = summary.issues,
    )
end

function Base.show(io::IO, info::DelimitedTableInspection)
    print(io, "DelimitedTableInspection(nrows=", info.nrows,
          ", ncols=", info.ncols,
          ", header_used=", info.header_used, ")")
end

function Base.show(io::IO, info::DatasetFileInspection)
    print(io, "DatasetFileInspection(schema_version=", repr(info.schema_version), ")")
end

function Base.show(io::IO, ::MIME"text/plain", info::DelimitedTableInspection)
    print(io, "DelimitedTableInspection\n  header_used: ", info.header_used,
          "\n  nrows: ", info.nrows,
          "\n  ncols: ", info.ncols,
          "\n  columns: ", repr(info.columns))
    isempty(info.sample_rows) || print(io, "\n  sample_rows: ", repr(info.sample_rows))
end

function Base.show(io::IO, ::MIME"text/plain", info::DatasetFileInspection)
    print(io, "DatasetFileInspection\n  schema_version: ", repr(info.schema_version),
          "\n  artifact: ", repr(info.artifact))
end

function Base.show(io::IO, info::DataFileInspectionSummary)
    print(io, "DataFileInspectionSummary(format=", repr(info.format),
          ", kind=", repr(info.kind),
          ", candidate_kinds=", repr(info.candidate_kinds))
    info.nrows === nothing || print(io, ", nrows=", info.nrows)
    info.ncols === nothing || print(io, ", ncols=", info.ncols)
    print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", info::DataFileInspectionSummary)
    println(io, "DataFileInspectionSummary")
    println(io, "  path: ", info.path)
    println(io, "  format: ", info.format)
    println(io, "  kind: ", info.kind)
    println(io, "  resolved_kind: ", repr(resolved_kind(info)))
    println(io, "  candidate_kinds: ", repr(info.candidate_kinds))
    info.schema_version === nothing || println(io, "  schema_version: ", repr(info.schema_version))
    info.header_used === nothing || println(io, "  header_used: ", info.header_used)
    info.nrows === nothing || println(io, "  nrows: ", info.nrows)
    info.ncols === nothing || println(io, "  ncols: ", info.ncols)
    info.columns === nothing || println(io, "  columns: ", repr(info.columns))
    isempty(info.sample_rows) || println(io, "  sample_rows: ", repr(info.sample_rows))
end

function Base.show(io::IO, summary::DataFileValidationSummary)
    print(io, "DataFileValidationSummary(ok=", summary.ok,
          ", format=", repr(summary.format),
          ", issues=", length(summary.issues), ")")
end

function Base.show(io::IO, ::MIME"text/plain", summary::DataFileValidationSummary)
    print(io, "DataFileValidationSummary\n  ok: ", summary.ok,
          "\n  path: ", summary.path,
          "\n  format: ", repr(summary.format),
          "\n  kind: ", repr(summary.kind),
          "\n  candidate_kinds: ", repr(summary.candidate_kinds))
    isempty(summary.issues) || print(io, "\n  issues: ", join(summary.issues, "\n          "))
end

function Base.show(io::IO, summary::LoadDataValidationSummary)
    print(io, "LoadDataValidationSummary(ok=", summary.ok,
          ", format=", repr(summary.format),
          ", kind=", repr(summary.kind),
          ", issues=", length(summary.issues), ")")
end

function Base.show(io::IO, ::MIME"text/plain", summary::LoadDataValidationSummary)
    print(io, "LoadDataValidationSummary\n  ok: ", summary.ok,
          "\n  path: ", summary.path,
          "\n  format: ", repr(summary.format),
          "\n  kind: ", repr(summary.kind),
          "\n  candidate_kinds: ", repr(summary.candidate_kinds),
          "\n  resolved_columns: ", repr(summary.resolved_columns))
    isempty(summary.issues) || print(io, "\n  issues: ", join(summary.issues, "\n          "))
end

function Base.show(io::IO, summary::TableColumnValidationSummary)
    print(io, "TableColumnValidationSummary(ok=", summary.ok,
          ", format=", repr(summary.format),
          ", kind=", repr(summary.kind),
          ", issues=", length(summary.issues), ")")
end

function Base.show(io::IO, ::MIME"text/plain", summary::TableColumnValidationSummary)
    print(io, "TableColumnValidationSummary\n  ok: ", summary.ok,
          "\n  path: ", summary.path,
          "\n  format: ", repr(summary.format),
          "\n  kind: ", repr(summary.kind),
          "\n  candidate_kinds: ", repr(summary.candidate_kinds),
          "\n  resolved_columns: ", repr(summary.resolved_columns))
    isempty(summary.issues) || print(io, "\n  issues: ", join(summary.issues, "\n          "))
end

@inline function _resolve_file_kind(kind::Symbol, opts::DataFileOptions)::Symbol
    if kind == :auto
        return opts.kind
    end
    return kind
end

@inline function _normalize_datafile_kind(kind::Symbol)::Symbol
    if kind === :PointCloud || kind === :point_cloud
        return :point_cloud
    elseif kind === :GraphData || kind === :graph
        return :graph
    elseif kind === :ImageNd || kind === :image
        return :image
    elseif kind === :GradedComplex || kind === :distance_matrix
        return :distance_matrix
    end
    return kind
end

@inline function _infer_file_format(path::AbstractString)::Symbol
    ext = lowercase(splitext(path)[2])
    if ext == ".json"
        return :dataset_json
    elseif ext == ".csv"
        return :csv
    elseif ext == ".tsv"
        return :tsv
    elseif ext == ".txt"
        return :txt
    elseif ext == ".bin"
        return :ripser_binary_lower_distance
    end
    throw(ArgumentError("load_data: could not infer format from extension $(ext). Pass format explicitly."))
end

@inline function _resolve_file_format(format::Symbol, path::AbstractString, opts::DataFileOptions)::Symbol
    f = format == :auto ? opts.format : format
    return f == :auto ? _infer_file_format(path) : f
end

@inline function _delimiter_for_format(fmt::Symbol, opts::DataFileOptions)
    if opts.delimiter !== nothing
        return opts.delimiter
    elseif fmt == :csv
        return ','
    elseif fmt == :tsv
        return '\t'
    elseif fmt == :txt
        return nothing
    end
    return nothing
end

@inline function _is_missing_token(tok::AbstractString)::Bool
    return lowercase(strip(tok)) in _MISSING_STRINGS
end

@inline function _split_line(line::AbstractString, delim)
    if delim === nothing
        return split(strip(line))
    end
    return [strip(t) for t in split(line, delim)]
end

@inline function _all_numeric_tokens(tokens::Vector{String})::Bool
    @inbounds for t in tokens
        tryparse(Float64, t) === nothing && return false
    end
    return true
end

@inline function _sanitize_name(s::AbstractString)::Symbol
    str = strip(String(s))
    isempty(str) && return Symbol("_")
    str = replace(str, r"\s+" => "_")
    return Symbol(str)
end

@inline function _uniquify_names(names::Vector{Symbol})
    seen = Dict{Symbol,Int}()
    out = Vector{Symbol}(undef, length(names))
    @inbounds for i in eachindex(names)
        nm = names[i]
        k = get!(seen, nm, 0) + 1
        seen[nm] = k
        out[i] = k == 1 ? nm : Symbol(string(nm), "_", k)
    end
    return out
end

@inline function _parse_delimited_table(path::AbstractString,
                                        fmt::Symbol,
                                        opts::DataFileOptions)
    delim = _delimiter_for_format(fmt, opts)
    rows = Vector{Vector{String}}()
    open(path, "r") do io
        for raw in eachline(io)
            line = strip(raw)
            isempty(line) && continue
            if opts.comment_prefix !== nothing
                startswith(line, string(opts.comment_prefix)) && continue
            end
            toks = _split_line(line, delim)
            isempty(toks) && continue
            push!(rows, toks)
        end
    end
    isempty(rows) && throw(ArgumentError("load_data: file has no data rows after filtering comments/empty lines: $(path)"))
    width = length(rows[1])
    @inbounds for i in 2:length(rows)
        length(rows[i]) == width || throw(ArgumentError("load_data: non-rectangular delimited table at row $(i)."))
    end

    header_used = if opts.header === nothing
        !_all_numeric_tokens(rows[1])
    else
        Bool(opts.header)
    end

    names = if header_used
        _uniquify_names([_sanitize_name(t) for t in rows[1]])
    else
        [Symbol("x", i) for i in 1:width]
    end
    data_rows = header_used ? rows[2:end] : rows
    isempty(data_rows) && throw(ArgumentError("load_data: no data rows after header in $(path)."))
    return names, data_rows, header_used
end

@inline function _col_selector_vec(x)
    if x === nothing
        return nothing
    elseif x isa Tuple
        return collect(x)
    elseif x isa AbstractVector
        return collect(x)
    else
        return Any[x]
    end
end

@inline function _column_index(names::Vector{Symbol}, key)::Int
    if key isa Integer
        idx = Int(key)
        1 <= idx <= length(names) || throw(ArgumentError("column index $(idx) is out of bounds 1:$(length(names))."))
        return idx
    elseif key isa Symbol
        idx = findfirst(==(key), names)
        idx === nothing && throw(ArgumentError("column $(key) not found. Available columns: $(names)."))
        return idx
    elseif key isa AbstractString
        return _column_index(names, Symbol(key))
    end
    throw(ArgumentError("unsupported column selector type $(typeof(key)); use Int/Symbol/String."))
end

@inline function _parse_float_token(tok::AbstractString)
    _is_missing_token(tok) && return nothing
    v = tryparse(Float64, tok)
    v === nothing && throw(ArgumentError("non-numeric token encountered: '$(tok)'"))
    return v
end

@inline function _parse_int_token(tok::AbstractString)
    _is_missing_token(tok) && return nothing
    vi = tryparse(Int, tok)
    if vi !== nothing
        return vi
    end
    vf = tryparse(Float64, tok)
    vf === nothing && throw(ArgumentError("non-integer token encountered: '$(tok)'"))
    isinteger(vf) || throw(ArgumentError("expected integer token but found non-integer value: '$(tok)'"))
    return Int(round(vf))
end

@inline function _drop_row_on_missing(policy::Symbol)::Bool
    policy == :drop_rows && return true
    policy == :error && return false
    throw(ArgumentError("unsupported missing_policy $(policy)."))
end

@inline function _table_inspection(path::AbstractString,
                                   fmt::Symbol,
                                   opts::DataFileOptions;
                                   sample_rows::Int=10)
    names, rows, header_used0 = _parse_delimited_table(path, fmt, opts)
    nr = length(rows)
    nc = length(names)
    nshow = min(sample_rows, nr)
    sample = nshow == 0 ? Vector{Vector{String}}() : copy(rows[1:nshow])
    detail = DelimitedTableInspection(header_used0, nr, nc, Tuple(names), sample)
    return DataFileInspectionSummary(
        abspath(path),
        fmt,
        :table,
        _table_candidate_kinds(nr, nc),
        nothing,
        header_used0,
        nr,
        nc,
        Tuple(names),
        sample,
        detail,
    )
end

@inline function _resolve_graph_column_indices(names::Vector{Symbol}, opts::DataFileOptions)
    length(names) >= 2 || throw(ArgumentError("graph parsing requires at least two columns for u/v."))
    u_idx = if opts.u_col == :u && !(:u in names)
        1
    else
        _column_index(names, opts.u_col)
    end
    v_idx = if opts.v_col == :v && !(:v in names)
        2
    else
        _column_index(names, opts.v_col)
    end
    w_idx = opts.weight_col === nothing ? nothing : _column_index(names, opts.weight_col)
    return u_idx, v_idx, w_idx
end

function _validate_distance_table(rows::Vector{Vector{String}},
                                  opts::DataFileOptions;
                                  check_symmetric::Bool=true,
                                  symmetry_tol::Real=1.0e-10)
    n = length(rows)
    n > 0 || throw(ArgumentError("distance_matrix parsing expects at least one row."))
    all(length(r) == n for r in rows) ||
        throw(ArgumentError("distance_matrix parsing expects an n x n numeric table; got $(n) rows with non-square width."))
    _drop_row_on_missing(opts.missing_policy) &&
        throw(ArgumentError("distance_matrix parsing does not support missing_policy=:drop_rows."))
    dist = Matrix{Float64}(undef, n, n)
    @inbounds for i in 1:n
        for j in 1:n
            v = _parse_float_token(rows[i][j])
            v === nothing && throw(ArgumentError("distance_matrix parsing encountered missing values."))
            dist[i, j] = v
        end
    end
    if check_symmetric
        tol = Float64(symmetry_tol)
        @inbounds for i in 1:n
            for j in (i + 1):n
                abs(dist[i, j] - dist[j, i]) <= tol ||
                    throw(ArgumentError("distance matrix is not symmetric within tolerance $(tol) at ($(i),$(j))."))
            end
        end
    end
    return nothing
end

function _resolve_table_columns(names::Vector{Symbol},
                                rows::Vector{Vector{String}},
                                kind::Symbol,
                                opts::DataFileOptions)
    if kind == :point_cloud
        cols_raw = _col_selector_vec(opts.cols)
        col_idx = cols_raw === nothing ? collect(1:length(names)) : [_column_index(names, c) for c in cols_raw]
        isempty(col_idx) && throw(ArgumentError("point-cloud parsing requires at least one coordinate column."))
        return (; coords=Tuple(names[col_idx]))
    elseif kind == :graph
        u_idx, v_idx, w_idx = _resolve_graph_column_indices(names, opts)
        return (; u=names[u_idx], v=names[v_idx], weight=(w_idx === nothing ? nothing : names[w_idx]))
    elseif kind == :image
        return (; values=Tuple(names))
    elseif kind == :distance_matrix
        _validate_distance_table(rows, opts)
        return (; values=Tuple(names))
    end
    throw(ArgumentError("unsupported kind=$(kind) for delimited-table parsing."))
end

function _load_point_cloud_table(path::AbstractString, fmt::Symbol, opts::DataFileOptions)
    names, rows, _ = _parse_delimited_table(path, fmt, opts)
    cols_raw = _col_selector_vec(opts.cols)
    col_idx = if cols_raw === nothing
        collect(1:length(names))
    else
        [_column_index(names, c) for c in cols_raw]
    end
    isempty(col_idx) && throw(ArgumentError("point-cloud parsing requires at least one coordinate column."))
    points = Matrix{Float64}(undef, length(rows), length(col_idx))
    drop_missing = _drop_row_on_missing(opts.missing_policy)
    kept = 0
    @inbounds for row in rows
        keep = true
        for j in eachindex(col_idx)
            v = _parse_float_token(row[col_idx[j]])
            if v === nothing
                if drop_missing
                    keep = false
                    break
                end
                throw(ArgumentError("missing value in point-cloud row with missing_policy=:error."))
            end
            points[kept + 1, j] = v
        end
        keep && (kept += 1)
    end
    kept == 0 && throw(ArgumentError("no point rows survived parsing."))
    return PointCloud(kept == size(points, 1) ? points : copy(@view points[1:kept, :]))
end

function _load_graph_table(path::AbstractString, fmt::Symbol, opts::DataFileOptions)
    names, rows, _ = _parse_delimited_table(path, fmt, opts)
    u_idx, v_idx, w_idx = _resolve_graph_column_indices(names, opts)
    edge_u = Int[]
    edge_v = Int[]
    weights = w_idx === nothing ? nothing : Float64[]
    drop_missing = _drop_row_on_missing(opts.missing_policy)
    maxv = 0
    @inbounds for row in rows
        u = _parse_int_token(row[u_idx])
        v = _parse_int_token(row[v_idx])
        if u === nothing || v === nothing
            if drop_missing
                continue
            end
            throw(ArgumentError("missing u/v in graph row with missing_policy=:error."))
        end
        u >= 1 || throw(ArgumentError("graph edges must be 1-based positive indices (got u=$(u))."))
        v >= 1 || throw(ArgumentError("graph edges must be 1-based positive indices (got v=$(v))."))
        push!(edge_u, u)
        push!(edge_v, v)
        if w_idx !== nothing
            w = _parse_float_token(row[w_idx])
            if w === nothing
                if drop_missing
                    pop!(edge_u)
                    pop!(edge_v)
                    continue
                end
                throw(ArgumentError("missing weight in graph row with missing_policy=:error."))
            end
            push!(weights, w)
        end
        maxv = max(maxv, u, v)
    end
    isempty(edge_u) && throw(ArgumentError("no graph edges survived parsing."))
    return GraphData(maxv, edge_u, edge_v; weights=weights, T=Float64, copy=false)
end

function _load_image_table(path::AbstractString, fmt::Symbol, opts::DataFileOptions)
    _, rows, _ = _parse_delimited_table(path, fmt, opts)
    m = length(rows)
    n = length(rows[1])
    A = Matrix{Float64}(undef, m, n)
    drop_missing = _drop_row_on_missing(opts.missing_policy)
    kept = 0
    @inbounds for i in 1:m
        if drop_missing
            has_missing = false
            for j in 1:n
                _is_missing_token(rows[i][j]) && (has_missing = true; break)
            end
            has_missing && continue
        end
        kept += 1
        for j in 1:n
            v = _parse_float_token(rows[i][j])
            if v === nothing
                throw(ArgumentError("missing value in image row with missing_policy=:error."))
            end
            A[kept, j] = v
        end
    end
    kept > 0 || throw(ArgumentError("no image rows survived parsing."))
    return ImageNd(A[1:kept, :])
end

function _load_distance_matrix_table(path::AbstractString, fmt::Symbol, opts::DataFileOptions;
                                     max_dim::Int=1,
                                     radius::Union{Nothing,Real}=nothing,
                                     knn::Union{Nothing,Int}=nothing,
                                     construction::ConstructionOptions=ConstructionOptions(),
                                     check_symmetric::Bool=true,
                                     symmetry_tol::Real=1.0e-10)
    _, rows, _ = _parse_delimited_table(path, fmt, opts)
    n = length(rows)
    _validate_distance_table(rows, opts; check_symmetric=check_symmetric, symmetry_tol=symmetry_tol)
    dist = Matrix{Float64}(undef, n, n)
    @inbounds for i in 1:n
        for j in 1:n
            dist[i, j] = Float64(_parse_float_token(rows[i][j]))
        end
    end
    return Serialization._graded_complex_from_distance_matrix(
        dist;
        max_dim=max_dim,
        radius=radius,
        knn=knn,
        construction=construction,
    )
end

@inline function _kind_matches_data(data, kind::Symbol)::Bool
    kind == :auto && return true
    if kind == :point_cloud
        return data isa PointCloud
    elseif kind == :graph
        return data isa GraphData
    elseif kind == :image
        return data isa ImageNd
    elseif kind == :distance_matrix
        return data isa GradedComplex
    end
    return false
end

"""
    load_data(path; kind=:auto, format=:auto, opts=DataFileOptions(), kwargs...)

Load a dataset-like object from a file path into canonical typed ingestion data.

Typical workflow
- call [`inspect_data_file`](@ref) first when the physical format or table kind
  is not obvious;
- use [`check_load_data`](@ref) for notebook-friendly validation of the
  canonical `load_data(...)` call;
- then call `load_data(path; kind=..., format=...)` once the contract is clear.

For delimited tables (`.csv`, `.tsv`, `.txt`), `kind=:auto` is intentionally
rejected because the same table shape can encode several mathematical objects.
"""
function load_data(path::AbstractString;
                   kind::Symbol=:auto,
                   format::Symbol=:auto,
                   opts::DataFileOptions=DataFileOptions(),
                   kwargs...)
    k = _resolve_file_kind(kind, opts)
    f = _resolve_file_format(format, path, opts)

    if f == :dataset_json
        if !isempty(kwargs)
            bad = join(string.(keys(kwargs)), ", ")
            throw(ArgumentError("load_data: dataset_json does not accept extra kwargs ($(bad))."))
        end
        data = Serialization.load_dataset_json(path)
        _kind_matches_data(data, k) || throw(ArgumentError("load_data: expected kind=$(k), but dataset JSON decoded to $(typeof(data))."))
        return data
    elseif f == :ripser_point_cloud
        data = Serialization.load_ripser_point_cloud(path)
        _kind_matches_data(data, k) || throw(ArgumentError("load_data: expected kind=$(k), but ripser point-cloud loader returns PointCloud."))
        return data
    elseif f == :ripser_distance
        data = Serialization.load_ripser_distance(path; kwargs...)
        _kind_matches_data(data, k == :auto ? :distance_matrix : k) ||
            throw(ArgumentError("load_data: expected kind=$(k), but ripser distance loader returns GradedComplex."))
        return data
    elseif f == :ripser_lower_distance
        data = Serialization.load_ripser_lower_distance(path; kwargs...)
        _kind_matches_data(data, k == :auto ? :distance_matrix : k) ||
            throw(ArgumentError("load_data: expected kind=$(k), but ripser lower-distance loader returns GradedComplex."))
        return data
    elseif f == :ripser_upper_distance
        data = Serialization.load_ripser_upper_distance(path; kwargs...)
        _kind_matches_data(data, k == :auto ? :distance_matrix : k) ||
            throw(ArgumentError("load_data: expected kind=$(k), but ripser upper-distance loader returns GradedComplex."))
        return data
    elseif f == :ripser_sparse_triplet
        data = Serialization.load_ripser_sparse_triplet(path; kwargs...)
        _kind_matches_data(data, k == :auto ? :distance_matrix : k) ||
            throw(ArgumentError("load_data: expected kind=$(k), but ripser sparse-triplet loader returns GradedComplex."))
        return data
    elseif f == :ripser_binary_lower_distance
        data = Serialization.load_ripser_binary_lower_distance(path; kwargs...)
        _kind_matches_data(data, k == :auto ? :distance_matrix : k) ||
            throw(ArgumentError("load_data: expected kind=$(k), but ripser binary loader returns GradedComplex."))
        return data
    elseif f == :ripser_lower_distance_streaming
        data = Serialization.load_ripser_lower_distance_streaming(path; kwargs...)
        _kind_matches_data(data, k == :auto ? :distance_matrix : k) ||
            throw(ArgumentError("load_data: expected kind=$(k), but ripser streaming loader returns GradedComplex."))
        return data
    elseif f == :csv || f == :tsv || f == :txt
        k == :auto &&
            throw(ArgumentError("load_data: kind=:auto is ambiguous for $(f). Pass kind=:point_cloud, :graph, :image, or :distance_matrix."))
        if k == :point_cloud
            if !isempty(kwargs)
                bad = join(string.(keys(kwargs)), ", ")
                throw(ArgumentError("load_data: point_cloud table parsing does not accept extra kwargs ($(bad))."))
            end
            return _load_point_cloud_table(path, f, opts)
        elseif k == :graph
            if !isempty(kwargs)
                bad = join(string.(keys(kwargs)), ", ")
                throw(ArgumentError("load_data: graph table parsing does not accept extra kwargs ($(bad))."))
            end
            return _load_graph_table(path, f, opts)
        elseif k == :image
            if !isempty(kwargs)
                bad = join(string.(keys(kwargs)), ", ")
                throw(ArgumentError("load_data: image table parsing does not accept extra kwargs ($(bad))."))
            end
            return _load_image_table(path, f, opts)
        elseif k == :distance_matrix
            return _load_distance_matrix_table(path, f, opts; kwargs...)
        end
        throw(ArgumentError("load_data: unsupported kind=$(k) for format=$(f)."))
    end
    throw(ArgumentError("load_data: unsupported format $(f)."))
end

"""
    inspect_data_file(path; format=:auto, sample_rows=10, opts=DataFileOptions())

Lightweight inspector for file-ingestion planning.

This is the canonical preflight entrypoint for `DataFileIO`. It returns a typed
[`DataFileInspectionSummary`](@ref) with stable accessors rather than a
shape-varying raw tuple.

Examples
--------
```julia
info = inspect_data_file("points.csv")
describe(info)
is_ambiguous(info)

check_load_data("points.csv"; kind=:point_cloud)
load_data("points.csv"; kind=:point_cloud)
```
"""
function inspect_data_file(path::AbstractString;
                           format::Symbol=:auto,
                           sample_rows::Int=10,
                           opts::DataFileOptions=DataFileOptions())
    sample_rows >= 0 || throw(ArgumentError("inspect_data_file: sample_rows must be >= 0."))
    f = _resolve_file_format(format, path, opts)
    if f == :dataset_json
        info = Serialization.inspect_json(path)
        raw_kind = Serialization.artifact_data_kind(info)
        raw_kind === nothing && (raw_kind = Serialization.artifact_kind(info))
        kind = raw_kind === nothing ? :unknown : _normalize_datafile_kind(Symbol(raw_kind))
        schema = Serialization.schema_version(info)
        detail0 = DatasetFileInspection(schema, info)
        return DataFileInspectionSummary(
            abspath(path),
            f,
            kind,
            (kind,),
            schema,
            nothing,
            nothing,
            nothing,
            nothing,
            Vector{Vector{String}}(),
            detail0,
        )
    elseif f == :ripser_point_cloud
        return DataFileInspectionSummary(
            abspath(path),
            f,
            :point_cloud,
            (:point_cloud,),
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            Vector{Vector{String}}(),
            nothing,
        )
    elseif f == :ripser_distance || f == :ripser_lower_distance || f == :ripser_upper_distance ||
           f == :ripser_sparse_triplet || f == :ripser_binary_lower_distance || f == :ripser_lower_distance_streaming
        return DataFileInspectionSummary(
            abspath(path),
            f,
            :distance_matrix,
            (:distance_matrix,),
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            Vector{Vector{String}}(),
            nothing,
        )
    elseif f == :csv || f == :tsv || f == :txt
        return _table_inspection(path, f, opts; sample_rows=sample_rows)
    end
    throw(ArgumentError("inspect_data_file: unsupported format $(f)."))
end

"""
    data_file_summary(path; format=:auto, sample_rows=10, opts=DataFileOptions())
    data_file_summary(info::DataFileInspectionSummary)

Owner-local summary alias for [`inspect_data_file`](@ref).

Use this when you are already working inside `DataFileIO` or
`TamerOp.Advanced.DataFileIO` and want the structured summary directly.

Example
-------
```julia
data_file_summary("points.csv")
```
"""
@inline function data_file_summary(path::AbstractString;
                                   format::Symbol=:auto,
                                   sample_rows::Int=10,
                                   opts::DataFileOptions=DataFileOptions())
    return describe(inspect_data_file(path; format=format, sample_rows=sample_rows, opts=opts))
end
@inline data_file_summary(info::DataFileInspectionSummary) = describe(info)

"""
    data_file_validation_summary(x)
    load_data_validation_summary(x)
    table_column_validation_summary(x)

Owner-local validation-summary aliases for the `check_*` helpers in
`DataFileIO`.

These accept either an already-built typed validation summary or the
corresponding path/keyword arguments for the underlying checker.
"""
@inline data_file_validation_summary(summary::DataFileValidationSummary) = summary
@inline function data_file_validation_summary(path::AbstractString;
                                              format::Symbol=:auto,
                                              opts::DataFileOptions=DataFileOptions(),
                                              throw::Bool=false)
    return check_data_file(path; format=format, opts=opts, throw=throw)
end

@inline load_data_validation_summary(summary::LoadDataValidationSummary) = summary
@inline function load_data_validation_summary(path::AbstractString;
                                              kind::Symbol=:auto,
                                              format::Symbol=:auto,
                                              opts::DataFileOptions=DataFileOptions(),
                                              throw::Bool=false)
    return check_load_data(path; kind=kind, format=format, opts=opts, throw=throw)
end

@inline table_column_validation_summary(summary::TableColumnValidationSummary) = summary
@inline function table_column_validation_summary(path::AbstractString;
                                                 kind::Symbol,
                                                 format::Symbol=:auto,
                                                 opts::DataFileOptions=DataFileOptions(),
                                                 throw::Bool=false)
    return check_table_columns(path; kind=kind, format=format, opts=opts, throw=throw)
end

"""
    check_data_file(path; format=:auto, opts=DataFileOptions(), throw=false) -> DataFileValidationSummary

Validate the file-level inspection contract for a data file.

This helper is lightweight: it checks format inference and the canonical
inspection path without materializing the loaded dataset.

Example
-------
```julia
report = check_data_file("points.csv")
ok(report)
```
"""
function check_data_file(path::AbstractString;
                         format::Symbol=:auto,
                         opts::DataFileOptions=DataFileOptions(),
                         throw::Bool=false)
    info = nothing
    fmt = nothing
    issues0 = String[]
    kind0 = nothing
    cands = ()
    try
        fmt = _resolve_file_format(format, path, opts)
        info = inspect_data_file(path; format=fmt, opts=opts)
        kind0 = kind(info)
        cands = candidate_kinds(info)
    catch err
        push!(issues0, _error_issue(err))
    end
    ok = isempty(issues0)
    summary = DataFileValidationSummary(ok, abspath(path), fmt, kind0, issues0, cands, info)
    throw && !ok && _throw_invalid_datafile("check_data_file", issues0)
    return summary
end

"""
    check_table_columns(path; kind, format=:auto, opts=DataFileOptions(), throw=false) -> TableColumnValidationSummary

Validate the interpretation of a delimited table under a concrete data `kind`.

Use this before `load_data(...; kind=:point_cloud|:graph|:image|:distance_matrix)`
when a CSV/TSV/TXT file could plausibly represent several mathematical objects.

Examples
--------
Graph table:
```julia
check_table_columns("graph.tsv";
    kind=:graph,
    format=:tsv,
    opts=DataFileOptions(; header=true, u_col=:u, v_col=:v, weight_col=:w))
```

Distance matrix:
```julia
check_table_columns("dist.csv"; kind=:distance_matrix, format=:csv)
```
"""
function check_table_columns(path::AbstractString;
                             kind::Symbol,
                             format::Symbol=:auto,
                             opts::DataFileOptions=DataFileOptions(),
                             throw::Bool=false)
    info = nothing
    fmt = nothing
    cands = ()
    issues0 = String[]
    cols0 = nothing
    try
        kind == :auto && Base.throw(ArgumentError("check_table_columns: pass an explicit table kind."))
        fmt = _resolve_file_format(format, path, opts)
        fmt in _TABLE_FORMATS || throw(ArgumentError("check_table_columns: format=$(fmt) is not a delimited table format."))
        names, rows, _ = _parse_delimited_table(path, fmt, opts)
        info = _table_inspection(path, fmt, opts; sample_rows=0)
        cands = candidate_kinds(info)
        cols0 = _resolve_table_columns(names, rows, kind, opts)
    catch err
        push!(issues0, _error_issue(err))
    end
    ok = isempty(issues0)
    summary = TableColumnValidationSummary(ok, abspath(path), fmt, kind, issues0, cands, cols0, info)
    throw && !ok && _throw_invalid_datafile("check_table_columns", issues0)
    return summary
end

"""
    check_load_data(path; kind=:auto, format=:auto, opts=DataFileOptions(), throw=false) -> LoadDataValidationSummary

Validate the canonical [`load_data`](@ref) call contract without requiring the
caller to interpret raw exceptions.

This is the preferred notebook/REPL validator before loading ambiguous table
files.

Examples
--------
Ambiguous CSV:
```julia
info = inspect_data_file("points.csv")
describe(info)

check_load_data("points.csv"; kind=:point_cloud,
                opts=DataFileOptions(; header=true, cols=(:x, :y)))
load_data("points.csv"; kind=:point_cloud,
          opts=DataFileOptions(; header=true, cols=(:x, :y)))
```

Dataset JSON:
```julia
check_load_data("points.json")
load_data("points.json")
```
"""
function check_load_data(path::AbstractString;
                         kind::Symbol=:auto,
                         format::Symbol=:auto,
                         opts::DataFileOptions=DataFileOptions(),
                         throw::Bool=false)
    info = nothing
    fmt = nothing
    issues0 = String[]
    cands = ()
    cols0 = nothing
    try
        fmt = _resolve_file_format(format, path, opts)
        info = inspect_data_file(path; format=fmt, opts=opts)
        cands = candidate_kinds(info)
        if is_table_file(info)
            kind == :auto &&
                Base.throw(ArgumentError("load_data: kind=:auto is ambiguous for $(fmt). Pass kind=:point_cloud, :graph, :image, or :distance_matrix."))
            table_summary = check_table_columns(path; kind=kind, format=fmt, opts=opts, throw=true)
            cols0 = resolved_columns(table_summary)
        end
        load_data(path; kind=kind, format=fmt, opts=opts)
    catch err
        push!(issues0, _error_issue(err))
    end
    ok = isempty(issues0)
    summary = LoadDataValidationSummary(ok, abspath(path), fmt, kind, issues0, cands, cols0, info)
    throw && !ok && _throw_invalid_datafile("check_load_data", issues0)
    return summary
end

end # module DataFileIO
