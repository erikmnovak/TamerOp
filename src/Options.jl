# =============================================================================
# Options.jl
#
# Structured option/specification types used across ingestion, encoding,
# workflow, module, and derived-functor layers.
# =============================================================================
module Options

using ..CoreModules: AbstractCoeffField, QQField

const _CONSTRUCTION_SPARSIFY_MODES = (:none, :radius, :knn, :greedy_perm)
const _CONSTRUCTION_COLLAPSE_MODES = (:none, :dominated_edges, :acyclic)
const _CONSTRUCTION_OUTPUT_STAGES = (:simplex_tree, :graded_complex, :cochain, :module,
                                     :fringe, :flange, :encoding_result)
const _DATAFILE_KINDS = (:auto, :point_cloud, :graph, :image, :distance_matrix)
const _DATAFILE_FORMATS = (:auto, :dataset_json, :csv, :tsv, :txt,
                           :ripser_point_cloud, :ripser_distance, :ripser_lower_distance,
                           :ripser_upper_distance, :ripser_sparse_triplet,
                           :ripser_binary_lower_distance, :ripser_lower_distance_streaming)
const _DATAFILE_MISSING_POLICIES = (:error, :drop_rows)
const _AXES_POLICIES = (:encoding, :as_given, :coarsen)
const _POSET_KINDS = (:signature, :dense, :regions)
const _ENCODING_BACKENDS = (:auto, :zn, :pl, :pl_backend, :pl_backend_boxes, :boxes, :axis,
                            :data, :serialization)
const _DERIVED_FUNCTOR_MODELS = (:auto, :projective, :injective, :unified, :first, :second)
const _DERIVED_FUNCTOR_CANONS = (:auto, :projective, :injective, :none)
const _FINITE_FRINGE_POSET_KINDS = (:regions, :dense)

@inline function _allowed_symbol_list(allowed::Tuple{Vararg{Symbol}})
    return join(string.(allowed), ", ")
end

@inline function _validate_symbol_choice(owner::AbstractString,
                                         field::AbstractString,
                                         value::Symbol,
                                         allowed::Tuple{Vararg{Symbol}})::Symbol
    value in allowed && return value
    throw(ArgumentError("$(owner): $(field) must be one of $(_allowed_symbol_list(allowed)) (got $(value))."))
end

@inline _validate_construction_sparsify(value::Symbol) =
    _validate_symbol_choice("ConstructionOptions", "sparsify", value, _CONSTRUCTION_SPARSIFY_MODES)
@inline _validate_construction_collapse(value::Symbol) =
    _validate_symbol_choice("ConstructionOptions", "collapse", value, _CONSTRUCTION_COLLAPSE_MODES)
@inline _validate_construction_output_stage(value::Symbol) =
    _validate_symbol_choice("ConstructionOptions", "output_stage", value, _CONSTRUCTION_OUTPUT_STAGES)
@inline _validate_datafile_kind(value::Symbol) =
    _validate_symbol_choice("DataFileOptions", "kind", value, _DATAFILE_KINDS)
@inline _validate_datafile_format(value::Symbol) =
    _validate_symbol_choice("DataFileOptions", "format", value, _DATAFILE_FORMATS)
@inline _validate_datafile_missing_policy(value::Symbol) =
    _validate_symbol_choice("DataFileOptions", "missing_policy", value, _DATAFILE_MISSING_POLICIES)
@inline _validate_axes_policy(owner::AbstractString, value::Symbol) =
    _validate_symbol_choice(owner, "axes_policy", value, _AXES_POLICIES)
@inline _validate_poset_kind(owner::AbstractString, value::Symbol) =
    _validate_symbol_choice(owner, "poset_kind", value, _POSET_KINDS)
@inline _validate_encoding_backend(value::Symbol) =
    _validate_symbol_choice("EncodingOptions", "backend", value, _ENCODING_BACKENDS)
@inline _validate_derived_functor_model(value::Symbol) =
    _validate_symbol_choice("DerivedFunctorOptions", "model", value, _DERIVED_FUNCTOR_MODELS)
@inline _validate_derived_functor_canon(value::Symbol) =
    _validate_symbol_choice("DerivedFunctorOptions", "canon", value, _DERIVED_FUNCTOR_CANONS)
@inline _validate_finite_fringe_poset_kind(value::Symbol) =
    _validate_symbol_choice("FiniteFringeOptions", "poset_kind", value, _FINITE_FRINGE_POSET_KINDS)

@inline function _render_option_namedtuple(io::IO, name::AbstractString, nt::NamedTuple)
    print(io, name, "(")
    first = true
    for (k, v) in pairs(nt)
        first || print(io, ", ")
        print(io, k, "=", repr(v))
        first = false
    end
    print(io, ")")
end

@inline function _render_option_namedtuple_pretty(io::IO, name::AbstractString, nt::NamedTuple)
    print(io, name)
    for (k, v) in pairs(nt)
        print(io, "\n  ", k, ": ", repr(v))
    end
end

"""
    FiltrationSpec(; kind, params...)

Lightweight serialized filtration specification.

Use `FiltrationSpec` when you want a stable, serializable description of an
ingestion family and its parameters. The `kind` field names the filtration
family; `params` stores the user-facing keyword parameters that define that
family.

Best practices:
- prefer typed filtration constructors internally when you are writing owner
  code for a filtration family,
- prefer `FiltrationSpec` at workflow/serialization boundaries,
- inspect with `describe(spec)` or plain-text display rather than field
  archaeology.
"""
struct FiltrationSpec
    kind::Symbol
    params::NamedTuple
end

FiltrationSpec(; kind::Symbol, params...) = FiltrationSpec(kind, NamedTuple(params))

"""
    ConstructionBudget(; max_simplices=nothing, max_edges=nothing, memory_budget_bytes=nothing)

Construction budget for data-ingestion enumeration.

These limits are operational guardrails, not mathematical parameters. They let
users cap edge/simplex growth or approximate memory use before a construction
path expands combinatorially.

Best practices:
- leave entries as `nothing` when no explicit cap is wanted,
- set at least one cap for large point-cloud or graph workflows,
- treat this as a safety contract, not a soft hint.
"""
struct ConstructionBudget
    max_simplices::Union{Nothing,Int}
    max_edges::Union{Nothing,Int}
    memory_budget_bytes::Union{Nothing,Int}
end

ConstructionBudget(; max_simplices::Union{Nothing,Integer}=nothing,
                   max_edges::Union{Nothing,Integer}=nothing,
                   memory_budget_bytes::Union{Nothing,Integer}=nothing) =
    ConstructionBudget(max_simplices === nothing ? nothing : Int(max_simplices),
                       max_edges === nothing ? nothing : Int(max_edges),
                       memory_budget_bytes === nothing ? nothing : Int(memory_budget_bytes))

"""
    ConstructionOptions(; sparsify=:none, collapse=:none, output_stage=:encoding_result,
                         budget=(nothing, nothing, nothing))

Canonical data-ingestion construction controls.

These options govern how a filtration is built, not what mathematical family
is being built:
- `sparsify` controls whether the input graph is dense or radius/knn/pruning
  driven,
- `collapse` controls simplification before downstream encoding,
- `output_stage` chooses how far the construction proceeds,
- `budget` adds explicit combinatorial guardrails.

Defaults are chosen for ordinary user workflows:
- `output_stage=:encoding_result` because that is the canonical high-level goal,
- `sparsify=:none` and `collapse=:none` preserve the full construction unless
  the user asks for a cheaper approximation/simplification.
"""
struct ConstructionOptions
    sparsify::Symbol
    collapse::Symbol
    output_stage::Symbol
    budget::ConstructionBudget
end

function ConstructionOptions(; sparsify::Symbol=:none,
                             collapse::Symbol=:none,
                             output_stage::Symbol=:encoding_result,
                             budget=(nothing, nothing, nothing))
    sp = _validate_construction_sparsify(sparsify)
    co = _validate_construction_collapse(collapse)
    os = _validate_construction_output_stage(output_stage)
    b = if budget isa ConstructionBudget
        budget
    elseif budget isa Tuple
        length(budget) == 3 || throw(ArgumentError("ConstructionOptions: budget tuple must be (max_simplices, max_edges, memory_budget_bytes)."))
        ConstructionBudget(; max_simplices=budget[1], max_edges=budget[2], memory_budget_bytes=budget[3])
    elseif budget isa NamedTuple
        ConstructionBudget(; budget...)
    else
        throw(ArgumentError("ConstructionOptions: budget must be a ConstructionBudget, 3-tuple, or NamedTuple."))
    end
    return ConstructionOptions(sp, co, os, b)
end

"""
    DataFileOptions(; kind=:auto, format=:auto, header=nothing, delimiter=nothing,
                    comment_prefix='#', missing_policy=:error, cols=nothing,
                    u_col=:u, v_col=:v, weight_col=nothing)

Canonical parsing controls for `load_data`.

These options describe how a file should be *interpreted*:
- `kind` chooses the intended mathematical data type,
- `format` chooses the physical on-disk schema,
- `missing_policy` determines whether malformed rows are rejected or dropped,
- `cols`/`u_col`/`v_col`/`weight_col` select tabular columns when applicable.

Defaults are intentionally conservative:
- `kind=:auto`, `format=:auto` let the loader infer common cases,
- `missing_policy=:error` avoids silently changing the mathematical dataset.
"""
struct DataFileOptions{H,D,C,ColsT,UColT,VColT,WColT}
    kind::Symbol
    format::Symbol
    header::H
    delimiter::D
    comment_prefix::C
    missing_policy::Symbol
    cols::ColsT
    u_col::UColT
    v_col::VColT
    weight_col::WColT
end

function DataFileOptions(;
    kind::Symbol=:auto,
    format::Symbol=:auto,
    header=nothing,
    delimiter=nothing,
    comment_prefix='#',
    missing_policy::Symbol=:error,
    cols=nothing,
    u_col=:u,
    v_col=:v,
    weight_col=nothing,
)
    kind2 = _validate_datafile_kind(kind)
    format2 = _validate_datafile_format(format)
    missing_policy2 = _validate_datafile_missing_policy(missing_policy)
    return DataFileOptions{
        typeof(header),
        typeof(delimiter),
        typeof(comment_prefix),
        typeof(cols),
        typeof(u_col),
        typeof(v_col),
        typeof(weight_col),
    }(kind2, format2, header, delimiter, comment_prefix, missing_policy2, cols, u_col, v_col, weight_col)
end

"""
    PipelineOptions(; orientation=nothing, axes_policy=:encoding, axis_kind=nothing,
                     eps=nothing, poset_kind=:signature, field=nothing, max_axis_len=nothing)

Structured pipeline controls that materially affect data-ingestion encodings and
their reproducibility in serialized pipeline artifacts.

- `axes_policy` controls whether downstream axes come from the encoding, are
  taken exactly as supplied, or are coarsened,
- `poset_kind` chooses the representation of the finite encoding poset,
- `orientation`, `axis_kind`, and `eps` thread geometric/ordering choices
  through ingestion and serialization.

Use this object when you need reproducible pipeline configuration rather than
ad-hoc keyword bundles.
"""
struct PipelineOptions{OrientationT,AxisKindT,EpsT,FieldT}
    orientation::OrientationT
    axes_policy::Symbol
    axis_kind::AxisKindT
    eps::EpsT
    poset_kind::Symbol
    field::FieldT
    max_axis_len::Union{Nothing,Int}
end

PipelineOptions(; orientation=nothing,
                axes_policy::Symbol=:encoding,
                axis_kind=nothing,
                eps=nothing,
                poset_kind::Symbol=:signature,
                field=nothing,
                max_axis_len::Union{Nothing,Int}=nothing) =
    PipelineOptions(orientation,
                    _validate_axes_policy("PipelineOptions", axes_policy),
                    axis_kind,
                    eps,
                    _validate_poset_kind("PipelineOptions", poset_kind),
                    field,
                    max_axis_len)

"""
    EncodingOptions(; backend=:auto, max_regions=nothing, strict_eps=nothing,
                    poset_kind=:signature, field=QQField())

Options controlling finite encodings.

- `backend` chooses the encoding engine (`:auto` is the canonical user default),
- `max_regions` caps region explosion in region-based encoders,
- `strict_eps` threads exactness/tolerance information to encoders that support
  it,
- `poset_kind` chooses whether the output poset stays structured or is
  materialized densely,
- `field` chooses the coefficient field of the resulting algebraic object.

Best practice:
- simple users should usually keep `backend=:auto`,
- advanced users should set `backend` only when they have a concrete algorithmic
  reason,
- use `describe(opts)` or plain-text display to inspect the effective contract.
"""
struct EncodingOptions{S,F<:AbstractCoeffField}
    backend::Symbol
    max_regions::Union{Nothing,Int}
    strict_eps::S
    poset_kind::Symbol
    field::F
end

EncodingOptions(; backend::Symbol=:auto,
                max_regions=nothing,
                strict_eps=nothing,
                poset_kind::Symbol=:signature,
                field::AbstractCoeffField=QQField()) =
    EncodingOptions{typeof(strict_eps),typeof(field)}(
        _validate_encoding_backend(backend),
        max_regions === nothing ? nothing : Int(max_regions),
        strict_eps,
        _validate_poset_kind("EncodingOptions", poset_kind),
        field,
    )

"""
    ResolutionOptions(; maxlen=3, minimal=false, check=true)

Options controlling projective/injective resolution construction.

- `maxlen` truncates the computed resolution length,
- `minimal` requests minimality when supported,
- `check` enables correctness validation of the constructed resolution.

Defaults favor safety and ordinary exploratory workloads.
"""
struct ResolutionOptions
    maxlen::Int
    minimal::Bool
    check::Bool
end

ResolutionOptions(; maxlen::Int=3, minimal::Bool=false, check::Bool=true) =
    ResolutionOptions(maxlen, minimal, check)

@inline function validate_pl_mode(mode::Symbol)::Symbol
    if mode === :fast
        return :fast
    elseif mode === :verified
        return :verified
    end
    throw(ArgumentError("pl_mode must be exactly :fast or :verified. Use :fast for the normal high-performance path or :verified when you want stricter geometric checking. Got $(mode)."))
end

"""
    InvariantOptions(; axes=nothing, axes_policy=:encoding, max_axis_len=256,
                     box=nothing, threads=nothing, strict=nothing, pl_mode=:fast)

Options controlling invariant computations.

- `axes_policy` controls how evaluation axes are chosen,
- `max_axis_len` caps coarsened axis resolution,
- `box` restricts geometric support when the invariant supports it,
- `threads` and `strict` control execution/contract behavior,
- `pl_mode` chooses the PL geometry contract (`:fast` or `:verified`).

Defaults are intentionally cheap and workflow-oriented:
- `axes_policy=:encoding` follows the encoding naturally,
- `pl_mode=:fast` is the canonical public default.
"""
struct InvariantOptions{A,B}
    axes::A
    axes_policy::Symbol
    max_axis_len::Int
    box::B
    threads::Union{Nothing,Bool}
    strict::Union{Nothing,Bool}
    pl_mode::Symbol
end

InvariantOptions(; axes=nothing,
                 axes_policy::Symbol=:encoding,
                 max_axis_len::Int=256,
                 box=nothing,
                 threads=nothing,
                 strict=nothing,
                 pl_mode::Symbol=:fast) =
    InvariantOptions{typeof(axes),typeof(box)}(
        axes,
        _validate_axes_policy("InvariantOptions", axes_policy),
        max_axis_len,
        box,
        threads,
        strict,
        validate_pl_mode(pl_mode),
    )

InvariantOptions(axes, axes_policy::Symbol, max_axis_len::Int, box, threads, strict) =
    InvariantOptions{typeof(axes),typeof(box)}(
        axes,
        _validate_axes_policy("InvariantOptions", axes_policy),
        max_axis_len,
        box,
        threads,
        strict,
        :fast,
    )

"""
    DerivedFunctorOptions(; maxdeg=3, model=:auto, canon=:auto)

Options controlling Ext/Tor and related derived-functor computations.

- `maxdeg` truncates the computed homological/cohomological degree range,
- `model` chooses the computational model (`:projective`, `:injective`,
  `:unified`, `:first`, `:second`, or `:auto` depending on the functor),
- `canon` chooses the preferred canonical representation when a symmetric model
  admits multiple reasonable choices.

These symbols are operational choices, not mathematical invariants; `:auto`
remains the best default unless a workflow genuinely needs a specific model.
"""
struct DerivedFunctorOptions
    maxdeg::Int
    model::Symbol
    canon::Symbol
end

DerivedFunctorOptions(; maxdeg::Int=3, model::Symbol=:auto, canon::Symbol=:auto) =
    DerivedFunctorOptions(maxdeg,
                          _validate_derived_functor_model(model),
                          _validate_derived_functor_canon(canon))

"""
    FiniteFringeOptions(; check=true, cached=true, store_sparse=false, scalar=1, poset_kind=:regions)

Options for `FiniteFringe` convenience entrypoints.

- `check` validates hand-built inputs,
- `cached` enables reuse of cached combinatorial helpers,
- `store_sparse` prefers sparse storage for the resulting data,
- `scalar` sets the scalar used in one-by-one fringe construction helpers,
- `poset_kind` chooses the representation of the resulting finite poset.
"""
struct FiniteFringeOptions{S}
    check::Bool
    cached::Bool
    store_sparse::Bool
    scalar::S
    poset_kind::Symbol
end

FiniteFringeOptions(; check::Bool=true,
                    cached::Bool=true,
                    store_sparse::Bool=false,
                    scalar=1,
                    poset_kind::Symbol=:regions) =
    FiniteFringeOptions{typeof(scalar)}(check, cached, store_sparse, scalar,
                                        _validate_finite_fringe_poset_kind(poset_kind))

"""
    ModuleOptions(; check_sizes=true, cache=nothing)

Options for `Modules` convenience entrypoints.

- `check_sizes=true` keeps constructor/query contracts strict,
- `cache` threads a cover/session cache into hot module-map paths.

This object is intentionally small. Simple users usually rely on the default.
"""
struct ModuleOptions{C}
    check_sizes::Bool
    cache::C
end

ModuleOptions(; check_sizes::Bool=true, cache=nothing) =
    ModuleOptions{typeof(cache)}(check_sizes, cache)

@inline _option_describe(spec::FiltrationSpec) =
    (kind=:filtration_spec,
     filtration_kind=spec.kind,
     parameters=spec.params,
     nparameters=length(spec.params))
@inline _option_describe(b::ConstructionBudget) =
    (kind=:construction_budget,
     max_simplices=b.max_simplices,
     max_edges=b.max_edges,
     memory_budget_bytes=b.memory_budget_bytes)
@inline _option_describe(opts::ConstructionOptions) =
    (kind=:construction_options,
     sparsify=opts.sparsify,
     collapse=opts.collapse,
     output_stage=opts.output_stage,
     budget=_option_describe(opts.budget))
@inline _option_describe(opts::DataFileOptions) =
    (kind=:datafile_options,
     data_kind=opts.kind,
     format=opts.format,
     missing_policy=opts.missing_policy,
     header=opts.header,
     delimiter=opts.delimiter,
     comment_prefix=opts.comment_prefix,
     cols=opts.cols,
     u_col=opts.u_col,
     v_col=opts.v_col,
     weight_col=opts.weight_col)
@inline _option_describe(opts::PipelineOptions) =
    (kind=:pipeline_options,
     orientation=opts.orientation,
     axes_policy=opts.axes_policy,
     axis_kind=opts.axis_kind,
     eps=opts.eps,
     poset_kind=opts.poset_kind,
     field=opts.field,
     max_axis_len=opts.max_axis_len)
@inline _option_describe(opts::EncodingOptions) =
    (kind=:encoding_options,
     backend=opts.backend,
     max_regions=opts.max_regions,
     strict_eps=opts.strict_eps,
     poset_kind=opts.poset_kind,
     field=opts.field)
@inline _option_describe(opts::ResolutionOptions) =
    (kind=:resolution_options,
     maxlen=opts.maxlen,
     minimal=opts.minimal,
     check=opts.check)
@inline _option_describe(opts::InvariantOptions) =
    (kind=:invariant_options,
     axes_policy=opts.axes_policy,
     max_axis_len=opts.max_axis_len,
     box=opts.box,
     threads=opts.threads,
     strict=opts.strict,
     pl_mode=opts.pl_mode,
     axes=opts.axes)
@inline _option_describe(opts::DerivedFunctorOptions) =
    (kind=:derived_functor_options,
     maxdeg=opts.maxdeg,
     model=opts.model,
     canon=opts.canon)
@inline _option_describe(opts::FiniteFringeOptions) =
    (kind=:finite_fringe_options,
     check=opts.check,
     cached=opts.cached,
     store_sparse=opts.store_sparse,
     scalar=opts.scalar,
     poset_kind=opts.poset_kind)
@inline _option_describe(opts::ModuleOptions) =
    (kind=:module_options,
     check_sizes=opts.check_sizes,
     cache=opts.cache)

function Base.show(io::IO, spec::FiltrationSpec)
    _render_option_namedtuple(io, "FiltrationSpec", (; kind=spec.kind, params=spec.params))
end
function Base.show(io::IO, ::MIME"text/plain", spec::FiltrationSpec)
    _render_option_namedtuple_pretty(io, "FiltrationSpec", (; kind=spec.kind, params=spec.params))
end

function Base.show(io::IO, b::ConstructionBudget)
    _render_option_namedtuple(io, "ConstructionBudget",
                              (; max_simplices=b.max_simplices,
                                 max_edges=b.max_edges,
                                 memory_budget_bytes=b.memory_budget_bytes))
end
function Base.show(io::IO, ::MIME"text/plain", b::ConstructionBudget)
    _render_option_namedtuple_pretty(io, "ConstructionBudget",
                                     (; max_simplices=b.max_simplices,
                                        max_edges=b.max_edges,
                                        memory_budget_bytes=b.memory_budget_bytes))
end

function Base.show(io::IO, opts::ConstructionOptions)
    _render_option_namedtuple(io, "ConstructionOptions",
                              (; sparsify=opts.sparsify,
                                 collapse=opts.collapse,
                                 output_stage=opts.output_stage,
                                 budget=opts.budget))
end
function Base.show(io::IO, ::MIME"text/plain", opts::ConstructionOptions)
    _render_option_namedtuple_pretty(io, "ConstructionOptions",
                                     (; sparsify=opts.sparsify,
                                        collapse=opts.collapse,
                                        output_stage=opts.output_stage,
                                        budget=opts.budget))
end

function Base.show(io::IO, opts::DataFileOptions)
    _render_option_namedtuple(io, "DataFileOptions",
                              (; kind=opts.kind, format=opts.format,
                                 missing_policy=opts.missing_policy))
end
function Base.show(io::IO, ::MIME"text/plain", opts::DataFileOptions)
    _render_option_namedtuple_pretty(io, "DataFileOptions",
                                     (; kind=opts.kind, format=opts.format,
                                        missing_policy=opts.missing_policy,
                                        cols=opts.cols,
                                        u_col=opts.u_col,
                                        v_col=opts.v_col,
                                        weight_col=opts.weight_col))
end

function Base.show(io::IO, opts::PipelineOptions)
    _render_option_namedtuple(io, "PipelineOptions",
                              (; axes_policy=opts.axes_policy,
                                 poset_kind=opts.poset_kind,
                                 max_axis_len=opts.max_axis_len))
end
function Base.show(io::IO, ::MIME"text/plain", opts::PipelineOptions)
    _render_option_namedtuple_pretty(io, "PipelineOptions",
                                     (; orientation=opts.orientation,
                                        axes_policy=opts.axes_policy,
                                        axis_kind=opts.axis_kind,
                                        eps=opts.eps,
                                        poset_kind=opts.poset_kind,
                                        field=opts.field,
                                        max_axis_len=opts.max_axis_len))
end

function Base.show(io::IO, opts::EncodingOptions)
    _render_option_namedtuple(io, "EncodingOptions",
                              (; backend=opts.backend, poset_kind=opts.poset_kind,
                                 max_regions=opts.max_regions))
end
function Base.show(io::IO, ::MIME"text/plain", opts::EncodingOptions)
    _render_option_namedtuple_pretty(io, "EncodingOptions",
                                     (; backend=opts.backend,
                                        max_regions=opts.max_regions,
                                        strict_eps=opts.strict_eps,
                                        poset_kind=opts.poset_kind,
                                        field=opts.field))
end

function Base.show(io::IO, opts::ResolutionOptions)
    _render_option_namedtuple(io, "ResolutionOptions",
                              (; maxlen=opts.maxlen, minimal=opts.minimal, check=opts.check))
end
function Base.show(io::IO, ::MIME"text/plain", opts::ResolutionOptions)
    _render_option_namedtuple_pretty(io, "ResolutionOptions",
                                     (; maxlen=opts.maxlen, minimal=opts.minimal, check=opts.check))
end

function Base.show(io::IO, opts::InvariantOptions)
    _render_option_namedtuple(io, "InvariantOptions",
                              (; axes_policy=opts.axes_policy,
                                 max_axis_len=opts.max_axis_len,
                                 pl_mode=opts.pl_mode))
end
function Base.show(io::IO, ::MIME"text/plain", opts::InvariantOptions)
    _render_option_namedtuple_pretty(io, "InvariantOptions",
                                     (; axes=opts.axes,
                                        axes_policy=opts.axes_policy,
                                        max_axis_len=opts.max_axis_len,
                                        box=opts.box,
                                        threads=opts.threads,
                                        strict=opts.strict,
                                        pl_mode=opts.pl_mode))
end

function Base.show(io::IO, opts::DerivedFunctorOptions)
    _render_option_namedtuple(io, "DerivedFunctorOptions",
                              (; maxdeg=opts.maxdeg, model=opts.model, canon=opts.canon))
end
function Base.show(io::IO, ::MIME"text/plain", opts::DerivedFunctorOptions)
    _render_option_namedtuple_pretty(io, "DerivedFunctorOptions",
                                     (; maxdeg=opts.maxdeg, model=opts.model, canon=opts.canon))
end

function Base.show(io::IO, opts::FiniteFringeOptions)
    _render_option_namedtuple(io, "FiniteFringeOptions",
                              (; check=opts.check, cached=opts.cached,
                                 store_sparse=opts.store_sparse, poset_kind=opts.poset_kind))
end
function Base.show(io::IO, ::MIME"text/plain", opts::FiniteFringeOptions)
    _render_option_namedtuple_pretty(io, "FiniteFringeOptions",
                                     (; check=opts.check,
                                        cached=opts.cached,
                                        store_sparse=opts.store_sparse,
                                        scalar=opts.scalar,
                                        poset_kind=opts.poset_kind))
end

function Base.show(io::IO, opts::ModuleOptions)
    _render_option_namedtuple(io, "ModuleOptions",
                              (; check_sizes=opts.check_sizes, cache=opts.cache))
end
function Base.show(io::IO, ::MIME"text/plain", opts::ModuleOptions)
    _render_option_namedtuple_pretty(io, "ModuleOptions",
                                     (; check_sizes=opts.check_sizes, cache=opts.cache))
end

end # module Options
