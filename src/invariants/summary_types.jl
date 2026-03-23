# This fragment owns typed UX wrappers for TamerOp.Invariants:
# typed summary/result objects for notebook-facing invariant aggregators, plus
# lightweight query validators for rank/support surfaces. Keep the wrappers
# thin: they should preserve programmatic access to the underlying report while
# offering better `show`, `describe`, and semantic accessors.

abstract type _InvariantReportWrapper end
abstract type _InvariantValidationWrapper end

struct RankInvariantResult{P,D<:AbstractDict{Tuple{Int,Int},Int}} <: AbstractDict{Tuple{Int,Int},Int}
    Q::P
    data::D
    include_zeros::Bool
end

struct SupportComponentsSummary{V<:AbstractVector{Vector{Int}}} <: AbstractVector{Vector{Int}}
    data::V
    component_sizes::Vector{Int}
end

struct ModuleSizeSummary{R} <: _InvariantReportWrapper
    report::R
end

struct ModuleGeometrySummary{R} <: _InvariantReportWrapper
    report::R
end

struct ModuleGeometryAsymptoticsSummary{R} <: _InvariantReportWrapper
    report::R
end

struct SupportMeasureSummary{R} <: _InvariantReportWrapper
    report::R
end

struct SupportBoundingBox{R} <: _InvariantReportWrapper
    report::R
end

struct SupportGraphDiameterSummary{R} <: _InvariantReportWrapper
    report::R
end

struct BettiSupportMeasuresSummary{R} <: _InvariantReportWrapper
    report::R
end

struct BassSupportMeasuresSummary{R} <: _InvariantReportWrapper
    report::R
end

struct RankQueryValidationSummary{R} <: _InvariantValidationWrapper
    report::R
end

struct SupportGeometryValidationSummary{R} <: _InvariantValidationWrapper
    report::R
end

@inline _report(x::_InvariantReportWrapper) = getfield(x, :report)
@inline _report(x::_InvariantValidationWrapper) = getfield(x, :report)

Base.:(==)(a::_InvariantReportWrapper, b::_InvariantReportWrapper) =
    typeof(a) === typeof(b) && _report(a) == _report(b)
Base.isequal(a::_InvariantReportWrapper, b::_InvariantReportWrapper) =
    typeof(a) === typeof(b) && isequal(_report(a), _report(b))
Base.hash(x::_InvariantReportWrapper, h::UInt) = hash((typeof(x), _report(x)), h)

Base.:(==)(a::_InvariantValidationWrapper, b::_InvariantValidationWrapper) =
    typeof(a) === typeof(b) && _report(a) == _report(b)
Base.isequal(a::_InvariantValidationWrapper, b::_InvariantValidationWrapper) =
    typeof(a) === typeof(b) && isequal(_report(a), _report(b))
Base.hash(x::_InvariantValidationWrapper, h::UInt) = hash((typeof(x), _report(x)), h)

function Base.getproperty(x::_InvariantReportWrapper, name::Symbol)
    name === :report && return getfield(x, :report)
    return getproperty(getfield(x, :report), name)
end

function Base.propertynames(x::_InvariantReportWrapper, private::Bool=false)
    names = propertynames(getfield(x, :report))
    return private ? (:report, names...) : names
end

function Base.getproperty(x::_InvariantValidationWrapper, name::Symbol)
    name === :report && return getfield(x, :report)
    return getproperty(getfield(x, :report), name)
end

function Base.propertynames(x::_InvariantValidationWrapper, private::Bool=false)
    names = propertynames(getfield(x, :report))
    return private ? (:report, names...) : names
end

Base.haskey(x::_InvariantReportWrapper, key::Symbol) = hasproperty(x, key)
Base.haskey(x::_InvariantValidationWrapper, key::Symbol) = hasproperty(x, key)
Base.getindex(x::_InvariantReportWrapper, key::Symbol) = getproperty(x, key)
Base.getindex(x::_InvariantValidationWrapper, key::Symbol) = getproperty(x, key)

Base.length(result::RankInvariantResult) = length(result.data)
Base.iterate(result::RankInvariantResult, state...) = iterate(result.data, state...)
Base.getindex(result::RankInvariantResult, key::Tuple{Int,Int}) = result.data[key]
Base.haskey(result::RankInvariantResult, key::Tuple{Int,Int}) = haskey(result.data, key)
Base.keys(result::RankInvariantResult) = keys(result.data)
Base.values(result::RankInvariantResult) = values(result.data)
Base.pairs(result::RankInvariantResult) = pairs(result.data)
Base.get(result::RankInvariantResult, key::Tuple{Int,Int}, default) = get(result.data, key, default)
Base.eltype(::Type{<:RankInvariantResult}) = Pair{Tuple{Int,Int},Int}
Base.copy(result::RankInvariantResult) = copy(result.data)

Base.IndexStyle(::Type{<:SupportComponentsSummary}) = IndexLinear()
Base.eltype(::Type{<:SupportComponentsSummary}) = Vector{Int}
Base.length(summary::SupportComponentsSummary) = length(summary.data)
Base.size(summary::SupportComponentsSummary) = (length(summary.data),)
Base.iterate(summary::SupportComponentsSummary, state...) = iterate(summary.data, state...)
Base.getindex(summary::SupportComponentsSummary, i::Int) = summary.data[i]

Base.length(::SupportBoundingBox) = 2
Base.length(::SupportGraphDiameterSummary) = 2

function Base.iterate(summary::SupportBoundingBox, state::Int=1)
    state == 1 && return (summary.lo, 2)
    state == 2 && return (summary.hi, 3)
    return nothing
end

function Base.iterate(summary::SupportGraphDiameterSummary, state::Int=1)
    state == 1 && return (summary.component_diameters, 2)
    state == 2 && return (summary.overall_diameter, 3)
    return nothing
end

function Base.getindex(summary::SupportBoundingBox, i::Int)
    i == 1 && return summary.lo
    i == 2 && return summary.hi
    throw(BoundsError(summary, i))
end

function Base.getindex(summary::SupportGraphDiameterSummary, i::Int)
    i == 1 && return summary.component_diameters
    i == 2 && return summary.overall_diameter
    throw(BoundsError(summary, i))
end

@doc raw"""
    RankInvariantResult

Typed result returned by [`rank_invariant`](@ref).

This is the preferred inspection object for the rank invariant of a finite
poset module. It preserves dictionary-like behavior:
- `result[(a, b)]` for stored entries,
- `get(result, (a, b), 0)` for zero-default lookup,
- iteration over `(a, b) => rank` pairs.

Use the semantic accessors
[`nentries`](@ref), [`nonzero_pairs`](@ref), [`value_at`](@ref),
[`store_zeros`](@ref), and [`source_poset`](@ref) instead of relying on raw
dictionary internals.

Example
-------
```julia
ri = rank_invariant(M)
describe(ri)
value_at(ri, 2, 5)
nonzero_pairs(ri)
```
""" RankInvariantResult

@doc raw"""
    SupportComponentsSummary

Typed result returned by [`support_components`](@ref).

This is the preferred inspection object for support connected components. It
preserves vector-like behavior:
- `length(summary)` gives the number of components,
- `summary[i]` returns the `i`th component,
- iteration yields the component vertex lists in order.

Use the semantic accessors [`ncomponents`](@ref),
[`component_sizes`](@ref), [`components`](@ref), and
[`largest_component`](@ref) for notebook-facing exploration.

Example
-------
```julia
comps = support_components(M, pi; opts=opts)
describe(comps)
component_sizes(comps)
largest_component(comps)
```
""" SupportComponentsSummary

"""
    ModuleSizeSummary

Typed summary returned by [`module_size_summary`](@ref).

Use semantic accessors such as [`total_measure`](@ref),
[`integrated_hilbert_mass`](@ref), and [`measure_by_dimension`](@ref) instead
of inspecting the raw report directly.
"""

"""
    ModuleGeometrySummary

Typed summary returned by [`module_geometry_summary`](@ref).

This is the preferred owner-local wrapper for combined support-size, interface,
histogram, and adjacency-graph statistics.
"""

"""
    ModuleGeometryAsymptoticsSummary

Typed summary returned by [`module_geometry_asymptotics`](@ref).

Use semantic accessors such as [`base_box`](@ref), [`windows`](@ref),
[`fit_total`](@ref), and [`ehrhart_total_measure`](@ref) for notebook-facing
inspection.
"""

"""
    SupportMeasureSummary

Typed support-measure summary returned by [`support_measure_stats`](@ref).
"""

"""
    SupportBoundingBox

Typed support bounding box returned by [`support_bbox`](@ref).

This remains tuple-compatible for light backwards compatibility:
`lo, hi = support_bbox(...)` still works.
"""

"""
    SupportGraphDiameterSummary

Typed support-graph diameter summary returned by [`support_graph_diameter`](@ref).

This remains tuple-compatible for light backwards compatibility:
`component_diameters, overall = support_graph_diameter(...)` still works.
"""

"""
    BettiSupportMeasuresSummary

Typed support summary returned by [`betti_support_measures`](@ref).
"""

"""
    BassSupportMeasuresSummary

Typed support summary returned by [`bass_support_measures`](@ref).
"""

"""
    RankQueryValidationSummary

Typed validation summary returned by [`check_rank_query_points`](@ref).
"""

"""
    SupportGeometryValidationSummary

Typed validation summary returned by [`check_support_box`](@ref),
[`check_support_window`](@ref), and [`check_support_measure_query`](@ref).
"""

"""
    nentries(result::RankInvariantResult)

Return the number of stored rank-invariant entries.
"""
@inline nentries(result::RankInvariantResult) = length(result.data)

"""
    nonzero_pairs(result::RankInvariantResult)

Return the comparable pairs `(a, b)` whose rank value is nonzero.
"""
@inline nonzero_pairs(result::RankInvariantResult) =
    [pair for pair in keys(result.data) if result.data[pair] != 0]

"""
    value_at(result::RankInvariantResult, a, b)

Return the rank-invariant value at `(a, b)`, using `0` for unstored pairs.
"""
@inline value_at(result::RankInvariantResult, a::Integer, b::Integer) =
    get(result.data, (Int(a), Int(b)), 0)

"""
    store_zeros(result::RankInvariantResult)

Return whether the rank-invariant result stores explicit zero entries.
"""
@inline store_zeros(result::RankInvariantResult) = result.include_zeros

"""
    source_poset(result::RankInvariantResult)

Return the finite poset on which the rank-invariant result is indexed.
"""
@inline source_poset(result::RankInvariantResult) = result.Q

"""
    ncomponents(summary::SupportComponentsSummary)

Return the number of support connected components.
"""
@inline ncomponents(summary::SupportComponentsSummary) = length(summary.data)

"""
    component_sizes(summary::SupportComponentsSummary)

Return the sizes of the support connected components.
"""
@inline component_sizes(summary::SupportComponentsSummary) = summary.component_sizes

"""
    components(summary::SupportComponentsSummary)

Return the support connected components as a vector of region-index vectors.
"""
@inline components(summary::SupportComponentsSummary) = summary.data

"""
    largest_component(summary::SupportComponentsSummary)

Return the largest support connected component, or `Int[]` if the support is
empty.
"""
@inline largest_component(summary::SupportComponentsSummary) =
    isempty(summary.data) ? Int[] : summary.data[argmax(summary.component_sizes)]

@inline total_measure(summary::Union{ModuleSizeSummary, SupportMeasureSummary}) = summary.total_measure
@inline total_measure(summary::ModuleGeometryAsymptoticsSummary) = summary.total_measure
@inline integrated_hilbert_mass(summary::ModuleSizeSummary) = summary.integrated_hilbert_mass
@inline integrated_hilbert_mass(summary::ModuleGeometryAsymptoticsSummary) = summary.integrated_hilbert_mass
@inline support_measure(summary::Union{ModuleSizeSummary, SupportMeasureSummary}) = summary.support_measure
@inline measure_by_dimension(summary::ModuleSizeSummary) = summary.measure_by_dimension
@inline mean_dim(summary::ModuleSizeSummary) = summary.mean_dim
@inline var_dim(summary::ModuleSizeSummary) = summary.var_dim

@inline size_summary(summary::ModuleGeometrySummary) = summary.size_summary
@inline interface_measure(summary::ModuleGeometrySummary) = summary.interface_measure
@inline interface_measure(summary::ModuleGeometryAsymptoticsSummary) = summary.interface_measure
@inline interface_by_dim_pair(summary::ModuleGeometrySummary) = summary.interface_by_dim_pair
@inline graph_stats(summary::ModuleGeometrySummary) = summary.graph_stats
@inline volume_histograms_by_dim(summary::ModuleGeometrySummary) = summary.volume_histograms_by_dim
@inline boundary_to_volume_histograms_by_dim(summary::ModuleGeometrySummary) = summary.boundary_to_volume_histograms_by_dim
@inline boundary_to_volume_samples_by_dim(summary::ModuleGeometrySummary) = summary.boundary_to_volume_samples_by_dim
@inline component_sizes(summary::ModuleGeometrySummary) = summary.graph_stats.component_sizes

@inline base_box(summary::ModuleGeometryAsymptoticsSummary) = summary.base_box
@inline windows(summary::ModuleGeometryAsymptoticsSummary) = summary.windows
@inline scales(summary::ModuleGeometryAsymptoticsSummary) = summary.scales
@inline fit_total(summary::ModuleGeometryAsymptoticsSummary) = summary.fit_total
@inline fit_integrated_hilbert_mass(summary::ModuleGeometryAsymptoticsSummary) = summary.fit_integrated_hilbert_mass
@inline fit_interface(summary::ModuleGeometryAsymptoticsSummary) = summary.fit_interface
@inline ehrhart_total_measure(summary::ModuleGeometryAsymptoticsSummary) = summary.ehrhart_total_measure

@inline estimate(summary::SupportMeasureSummary) = summary.estimate
@inline stderr(summary::SupportMeasureSummary) = summary.stderr
@inline ci(summary::SupportMeasureSummary) = summary.ci
@inline support_fraction(summary::SupportMeasureSummary) = summary.support_fraction
@inline support_bbox(summary::SupportMeasureSummary) = summary.support_bbox
@inline support_bbox(summary::SupportBoundingBox) = summary
@inline support_bbox_diameter(summary::SupportMeasureSummary) = summary.support_bbox_diameter
@inline component_sizes(summary::SupportGraphDiameterSummary) = summary.component_sizes
@inline overall_graph_diameter(summary::SupportGraphDiameterSummary) = summary.overall_diameter

@inline support_by_degree(summary::Union{BettiSupportMeasuresSummary, BassSupportMeasuresSummary}) = summary.support_by_degree
@inline mass_by_degree(summary::Union{BettiSupportMeasuresSummary, BassSupportMeasuresSummary}) = summary.mass_by_degree
@inline support_union(summary::Union{BettiSupportMeasuresSummary, BassSupportMeasuresSummary}) = summary.support_union
@inline support_total(summary::Union{BettiSupportMeasuresSummary, BassSupportMeasuresSummary}) = summary.support_total
@inline mass_total(summary::Union{BettiSupportMeasuresSummary, BassSupportMeasuresSummary}) = summary.mass_total

function describe(result::RankInvariantResult)
    return (
        kind = :rank_invariant,
        nentries = nentries(result),
        nnonzero = length(nonzero_pairs(result)),
        store_zeros = store_zeros(result),
        nvertices = nvertices(source_poset(result)),
        poset_type = typeof(source_poset(result)),
    )
end

function describe(summary::SupportComponentsSummary)
    return (
        kind = :support_components,
        ncomponents = ncomponents(summary),
        component_sizes = component_sizes(summary),
        largest_component = largest_component(summary),
    )
end

function describe(summary::ModuleSizeSummary)
    return (
        kind = :module_size_summary,
        total_measure = summary.total_measure,
        integrated_hilbert_mass = summary.integrated_hilbert_mass,
        support_measure = summary.support_measure,
        mean_dim = summary.mean_dim,
        var_dim = summary.var_dim,
        ndimension_strata = length(summary.measure_by_dimension),
    )
end

function describe(summary::ModuleGeometrySummary)
    gs = summary.graph_stats
    return (
        kind = :module_geometry_summary,
        total_measure = summary.size_summary.total_measure,
        support_measure = summary.size_summary.support_measure,
        interface_measure = summary.interface_measure,
        ncomponents = gs.ncomponents,
        component_sizes = gs.component_sizes,
        nedges = gs.nedges,
        volume_histogram_degrees = collect(sort!(collect(keys(summary.volume_histograms_by_dim)))),
        boundary_to_volume_histogram_degrees = collect(sort!(collect(keys(summary.boundary_to_volume_histograms_by_dim)))),
    )
end

function describe(summary::ModuleGeometryAsymptoticsSummary)
    return (
        kind = :module_geometry_asymptotics,
        nscales = length(summary.scales),
        scales = summary.scales,
        exponent_total_measure = summary.exponent_total_measure,
        exponent_integrated_hilbert_mass = summary.exponent_integrated_hilbert_mass,
        exponent_interface_measure = summary.exponent_interface_measure,
        has_interface = summary.interface_measure !== nothing,
        has_ehrhart_total_measure = summary.ehrhart_total_measure !== nothing,
    )
end

function describe(summary::SupportMeasureSummary)
    return (
        kind = :support_measure_summary,
        estimate = summary.estimate,
        stderr = summary.stderr,
        ci = summary.ci,
        total_measure = summary.total_measure,
        support_measure = summary.support_measure,
        support_fraction = summary.support_fraction,
        support_bbox = summary.support_bbox,
        support_bbox_diameter = summary.support_bbox_diameter,
    )
end

function describe(summary::SupportBoundingBox)
    return (
        kind = :support_bounding_box,
        ambient_dimension = length(summary.lo),
        lo = summary.lo,
        hi = summary.hi,
        widths = summary.hi .- summary.lo,
    )
end

function describe(summary::SupportGraphDiameterSummary)
    return (
        kind = :support_graph_diameter,
        ncomponents = length(summary.component_diameters),
        component_sizes = summary.component_sizes,
        component_diameters = summary.component_diameters,
        overall_graph_diameter = summary.overall_diameter,
    )
end

function describe(summary::BettiSupportMeasuresSummary)
    return (
        kind = :betti_support_measures,
        ndegrees = length(summary.support_by_degree),
        support_union = summary.support_union,
        support_total = summary.support_total,
        mass_total = summary.mass_total,
    )
end

function describe(summary::BassSupportMeasuresSummary)
    return (
        kind = :bass_support_measures,
        ndegrees = length(summary.support_by_degree),
        support_union = summary.support_union,
        support_total = summary.support_total,
        mass_total = summary.mass_total,
    )
end

function Base.show(io::IO, summary::ModuleSizeSummary)
    print(io,
        "ModuleSizeSummary(total_measure=", summary.total_measure,
        ", support_measure=", summary.support_measure,
        ", mean_dim=", summary.mean_dim, ")")
end

function Base.show(io::IO, result::RankInvariantResult)
    d = describe(result)
    print(io,
        "RankInvariantResult(nentries=", d.nentries,
        ", nnonzero=", d.nnonzero,
        ", store_zeros=", d.store_zeros, ")")
end

function Base.show(io::IO, ::MIME"text/plain", result::RankInvariantResult)
    d = describe(result)
    print(io,
        "RankInvariantResult",
        "\n  nentries: ", d.nentries,
        "\n  nnonzero: ", d.nnonzero,
        "\n  store_zeros: ", d.store_zeros,
        "\n  nvertices: ", d.nvertices,
        "\n  poset_type: ", d.poset_type)
end

function Base.show(io::IO, summary::SupportComponentsSummary)
    d = describe(summary)
    print(io,
        "SupportComponentsSummary(ncomponents=", d.ncomponents,
        ", largest_component_size=", isempty(d.largest_component) ? 0 : length(d.largest_component), ")")
end

function Base.show(io::IO, ::MIME"text/plain", summary::SupportComponentsSummary)
    d = describe(summary)
    print(io,
        "SupportComponentsSummary",
        "\n  ncomponents: ", d.ncomponents,
        "\n  component_sizes: ", repr(d.component_sizes),
        "\n  largest_component: ", repr(d.largest_component))
end

function Base.show(io::IO, ::MIME"text/plain", summary::ModuleSizeSummary)
    d = describe(summary)
    print(io,
        "ModuleSizeSummary",
        "\n  total_measure: ", d.total_measure,
        "\n  integrated_hilbert_mass: ", d.integrated_hilbert_mass,
        "\n  support_measure: ", d.support_measure,
        "\n  mean_dim: ", d.mean_dim,
        "\n  var_dim: ", d.var_dim,
        "\n  ndimension_strata: ", d.ndimension_strata)
end

function Base.show(io::IO, summary::ModuleGeometrySummary)
    d = describe(summary)
    print(io,
        "ModuleGeometrySummary(total_measure=", d.total_measure,
        ", interface_measure=", d.interface_measure,
        ", ncomponents=", d.ncomponents, ")")
end

function Base.show(io::IO, ::MIME"text/plain", summary::ModuleGeometrySummary)
    d = describe(summary)
    print(io,
        "ModuleGeometrySummary",
        "\n  total_measure: ", d.total_measure,
        "\n  support_measure: ", d.support_measure,
        "\n  interface_measure: ", d.interface_measure,
        "\n  ncomponents: ", d.ncomponents,
        "\n  component_sizes: ", repr(d.component_sizes),
        "\n  nedges: ", d.nedges,
        "\n  volume_histogram_degrees: ", repr(d.volume_histogram_degrees),
        "\n  boundary_to_volume_histogram_degrees: ", repr(d.boundary_to_volume_histogram_degrees))
end

function Base.show(io::IO, summary::ModuleGeometryAsymptoticsSummary)
    d = describe(summary)
    print(io,
        "ModuleGeometryAsymptoticsSummary(nscales=", d.nscales,
        ", exponent_total_measure=", d.exponent_total_measure,
        ", exponent_interface_measure=", d.exponent_interface_measure, ")")
end

function Base.show(io::IO, ::MIME"text/plain", summary::ModuleGeometryAsymptoticsSummary)
    d = describe(summary)
    print(io,
        "ModuleGeometryAsymptoticsSummary",
        "\n  scales: ", repr(d.scales),
        "\n  exponent_total_measure: ", d.exponent_total_measure,
        "\n  exponent_integrated_hilbert_mass: ", d.exponent_integrated_hilbert_mass,
        "\n  exponent_interface_measure: ", repr(d.exponent_interface_measure),
        "\n  has_interface: ", d.has_interface,
        "\n  has_ehrhart_total_measure: ", d.has_ehrhart_total_measure)
end

function Base.show(io::IO, summary::SupportMeasureSummary)
    print(io,
        "SupportMeasureSummary(estimate=", summary.estimate,
        ", support_fraction=", summary.support_fraction,
        ", bbox_diameter=", summary.support_bbox_diameter, ")")
end

function Base.show(io::IO, ::MIME"text/plain", summary::SupportMeasureSummary)
    d = describe(summary)
    print(io,
        "SupportMeasureSummary",
        "\n  estimate: ", d.estimate,
        "\n  stderr: ", d.stderr,
        "\n  ci: ", repr(d.ci),
        "\n  total_measure: ", d.total_measure,
        "\n  support_measure: ", d.support_measure,
        "\n  support_fraction: ", d.support_fraction,
        "\n  support_bbox: ", repr(d.support_bbox),
        "\n  support_bbox_diameter: ", d.support_bbox_diameter)
end

function Base.show(io::IO, summary::SupportBoundingBox)
    print(io, "SupportBoundingBox(dim=", length(summary.lo), ")")
end

function Base.show(io::IO, ::MIME"text/plain", summary::SupportBoundingBox)
    d = describe(summary)
    print(io,
        "SupportBoundingBox",
        "\n  ambient_dimension: ", d.ambient_dimension,
        "\n  lo: ", repr(d.lo),
        "\n  hi: ", repr(d.hi),
        "\n  widths: ", repr(d.widths))
end

function Base.show(io::IO, summary::SupportGraphDiameterSummary)
    print(io,
        "SupportGraphDiameterSummary(ncomponents=", length(summary.component_diameters),
        ", overall=", summary.overall_diameter, ")")
end

function Base.show(io::IO, ::MIME"text/plain", summary::SupportGraphDiameterSummary)
    d = describe(summary)
    print(io,
        "SupportGraphDiameterSummary",
        "\n  ncomponents: ", d.ncomponents,
        "\n  component_sizes: ", repr(d.component_sizes),
        "\n  component_diameters: ", repr(d.component_diameters),
        "\n  overall_graph_diameter: ", d.overall_graph_diameter)
end

function Base.show(io::IO, summary::BettiSupportMeasuresSummary)
    print(io,
        "BettiSupportMeasuresSummary(ndegrees=", length(summary.support_by_degree),
        ", support_total=", summary.support_total,
        ", mass_total=", summary.mass_total, ")")
end

function Base.show(io::IO, ::MIME"text/plain", summary::BettiSupportMeasuresSummary)
    d = describe(summary)
    print(io,
        "BettiSupportMeasuresSummary",
        "\n  ndegrees: ", d.ndegrees,
        "\n  support_union: ", d.support_union,
        "\n  support_total: ", d.support_total,
        "\n  mass_total: ", d.mass_total)
end

function Base.show(io::IO, summary::BassSupportMeasuresSummary)
    print(io,
        "BassSupportMeasuresSummary(ndegrees=", length(summary.support_by_degree),
        ", support_total=", summary.support_total,
        ", mass_total=", summary.mass_total, ")")
end

function Base.show(io::IO, ::MIME"text/plain", summary::BassSupportMeasuresSummary)
    d = describe(summary)
    print(io,
        "BassSupportMeasuresSummary",
        "\n  ndegrees: ", d.ndegrees,
        "\n  support_union: ", d.support_union,
        "\n  support_total: ", d.support_total,
        "\n  mass_total: ", d.mass_total)
end

function Base.show(io::IO, summary::RankQueryValidationSummary)
    print(io, "RankQueryValidationSummary(valid=", summary.valid,
        ", issues=", length(summary.issues), ")")
end

function Base.show(io::IO, ::MIME"text/plain", summary::RankQueryValidationSummary)
    println(io, "RankQueryValidationSummary")
    println(io, "  valid: ", summary.valid)
    println(io, "  strict: ", summary.strict)
    println(io, "  parameter_dim: ", summary.parameter_dim)
    println(io, "  x_region: ", repr(summary.x_region))
    println(io, "  y_region: ", repr(summary.y_region))
    if isempty(summary.issues)
        print(io, "  issues: none")
    else
        println(io, "  issues:")
        for issue in summary.issues
            println(io, "    - ", issue)
        end
    end
end

function Base.show(io::IO, summary::SupportGeometryValidationSummary)
    print(io, "SupportGeometryValidationSummary(kind=", summary.kind,
        ", valid=", summary.valid,
        ", issues=", length(summary.issues), ")")
end

function Base.show(io::IO, ::MIME"text/plain", summary::SupportGeometryValidationSummary)
    println(io, "SupportGeometryValidationSummary")
    println(io, "  kind: ", summary.kind)
    println(io, "  valid: ", summary.valid)
    if hasproperty(summary, :parameter_dim)
        println(io, "  parameter_dim: ", summary.parameter_dim)
    end
    if hasproperty(summary, :box_source)
        println(io, "  box_source: ", summary.box_source)
    end
    if hasproperty(summary, :resolved_box)
        println(io, "  resolved_box: ", repr(summary.resolved_box))
    end
    if hasproperty(summary, :min_dim)
        println(io, "  min_dim: ", repr(summary.min_dim))
    end
    if hasproperty(summary, :strict)
        println(io, "  strict: ", repr(summary.strict))
    end
    if isempty(summary.issues)
        print(io, "  issues: none")
    else
        println(io, "  issues:")
        for issue in summary.issues
            println(io, "    - ", issue)
        end
    end
end

function _validation_report(::Type{S}, kind::Symbol, issues::Vector{String}; throw::Bool, kwargs...) where {S}
    valid = isempty(issues)
    if throw && !valid
        Base.throw(ArgumentError(string(kind, ": ", join(issues, " "))))
    end
    report = merge((kind=kind, valid=valid, issues=issues), NamedTuple(kwargs))
    return S(report)
end

"""
    check_rank_query_points(pi, x, y; opts=InvariantOptions(), throw=false)

Validate a pair of rank-query points against the default `rank_query` contract.

This checks:
- point shapes against `dimension(pi)`,
- whether `locate(pi, x)` / `locate(pi, y)` succeed,
- whether unknown regions are acceptable under `opts.strict`.
"""
function check_rank_query_points(pi::PLikeEncodingMap, x, y;
    opts::InvariantOptions = InvariantOptions(),
    throw::Bool = false)
    issues = String[]
    xrep = check_query_point(pi, x; throw=false)
    yrep = check_query_point(pi, y; throw=false)
    !xrep.valid && append!(issues, xrep.issues)
    !yrep.valid && append!(issues, yrep.issues)

    strict0 = opts.strict === nothing ? true : opts.strict
    xr = nothing
    yr = nothing
    if xrep.valid
        try
            xr = locate(pi, x)
        catch err
            push!(issues, "locate(pi, x) failed: $(sprint(showerror, err))")
        end
    end
    if yrep.valid
        try
            yr = locate(pi, y)
        catch err
            push!(issues, "locate(pi, y) failed: $(sprint(showerror, err))")
        end
    end
    if strict0
        xr == 0 && push!(issues, "locate(pi, x) returned 0 under strict rank-query semantics.")
        yr == 0 && push!(issues, "locate(pi, y) returned 0 under strict rank-query semantics.")
    end

    return _validation_report(RankQueryValidationSummary, :rank_query_points, issues;
        throw=throw,
        parameter_dim=xrep.parameter_dim,
        strict=strict0,
        x_region=xr,
        y_region=yr,
        x_valid=xrep.valid,
        y_valid=yrep.valid)
end

check_rank_query_points(enc::CompiledEncoding, x, y; kwargs...) =
    check_rank_query_points(_unwrap_compiled(enc), x, y; kwargs...)

"""
    check_support_box(pi; opts=InvariantOptions(), throw=false)

Validate the support-window box used by the support-geometry helpers.
"""
function check_support_box(pi::PLikeEncodingMap;
    opts::InvariantOptions = InvariantOptions(),
    throw::Bool = false)
    issues = String[]
    dim = try
        Int(dimension(pi))
    catch err
        push!(issues, "dimension(pi) failed: $(sprint(showerror, err))")
        0
    end

    box_source = opts.box === nothing ? :auto : :explicit
    box0 = opts.box === nothing ? :auto : opts.box
    resolved = nothing
    try
        resolved = _resolve_box(pi, box0)
    catch err
        push!(issues, "support box resolution failed: $(sprint(showerror, err))")
    end

    if resolved !== nothing
        lo, hi = resolved
        length(lo) == length(hi) || push!(issues, "support box endpoints have mismatched lengths.")
        if dim > 0 && length(lo) != dim
            push!(issues, "support box dimension $(length(lo)) does not match encoding dimension $dim.")
        end
        if length(lo) == length(hi)
            for i in eachindex(lo)
                float(lo[i]) <= float(hi[i]) || push!(issues, "support box endpoint pair $i is invalid: lo > hi.")
            end
        end
    end

    return _validation_report(SupportGeometryValidationSummary, :support_box, issues;
        throw=throw,
        parameter_dim=dim,
        box_source=box_source,
        resolved_box=resolved)
end

check_support_box(enc::CompiledEncoding; kwargs...) =
    check_support_box(_unwrap_compiled(enc); kwargs...)

"""
    check_support_window(pi; opts=InvariantOptions(), throw=false)

Owner-local synonym for [`check_support_box`](@ref).
"""
check_support_window(pi::Union{PLikeEncodingMap, CompiledEncoding}; kwargs...) =
    check_support_box(pi; kwargs...)

"""
    check_support_measure_query(pi; opts=InvariantOptions(), min_dim=1, throw=false)

Validate the input contract used by the support-measure and support-geometry
summary routines.
"""
function check_support_measure_query(pi::PLikeEncodingMap;
    opts::InvariantOptions = InvariantOptions(),
    min_dim = 1,
    throw::Bool = false)
    issues = String[]
    box_report = check_support_box(pi; opts=opts, throw=false)
    !box_report.valid && append!(issues, box_report.issues)
    min_dim isa Integer || push!(issues, "min_dim must be an integer threshold.")
    strict0 = opts.strict === nothing ? true : opts.strict
    return _validation_report(SupportGeometryValidationSummary, :support_measure_query, issues;
        throw=throw,
        parameter_dim=box_report.parameter_dim,
        box_source=box_report.box_source,
        resolved_box=box_report.resolved_box,
        min_dim=min_dim,
        strict=strict0)
end

check_support_measure_query(enc::CompiledEncoding; kwargs...) =
    check_support_measure_query(_unwrap_compiled(enc); kwargs...)
