# Visualization core types, summaries, and notebook-facing display wrappers.

function visualize end
function save_visual end
function save_visuals end
function available_visuals end
function visual_spec end
function render end
function check_visual_spec end
function check_visual_request end
function visual_summary end

abstract type AbstractVisualizationLayer end

struct HeatmapLayer <: AbstractVisualizationLayer
    x::Vector{Float64}
    y::Vector{Float64}
    values::Matrix{Float64}
    colormap::Symbol
    alpha::Float64
    colorbar_label::String
    show_colorbar::Bool
end

HeatmapLayer(x::Vector{Float64},
             y::Vector{Float64},
             values::Matrix{Float64},
             colormap::Symbol,
             alpha::Float64,
             colorbar_label::AbstractString) =
    HeatmapLayer(x, y, values, colormap, alpha, String(colorbar_label), !isempty(colorbar_label))

struct RectLayer <: AbstractVisualizationLayer
    rects::Vector{NTuple{4,Float64}}
    fill_color::Symbol
    stroke_color::Symbol
    alpha::Float64
    linewidth::Float64
end

struct SegmentLayer <: AbstractVisualizationLayer
    segments::Vector{NTuple{4,Float64}}
    color::Symbol
    alpha::Float64
    linewidth::Float64
end

struct PolylineLayer <: AbstractVisualizationLayer
    paths::Vector{Vector{NTuple{2,Float64}}}
    color::Symbol
    alpha::Float64
    linewidth::Float64
    closed::Bool
end

struct PointLayer <: AbstractVisualizationLayer
    points::Vector{NTuple{2,Float64}}
    color::Union{Symbol,Vector{Float64}}
    alpha::Float64
    markersize::Float64
    colormap::Symbol
end

PointLayer(points::Vector{NTuple{2,Float64}},
           color::Symbol,
           alpha::Float64,
           markersize::Float64) =
    PointLayer(points, color, alpha, markersize, :viridis)

PointLayer(points::Vector{NTuple{2,Float64}},
           color::Vector{Float64},
           alpha::Float64,
           markersize::Float64) =
    PointLayer(points, color, alpha, markersize, :viridis)

struct Point3Layer <: AbstractVisualizationLayer
    points::Vector{NTuple{3,Float64}}
    color::Union{Symbol,Vector{Float64}}
    alpha::Float64
    markersize::Float64
    colormap::Symbol
end

Point3Layer(points::Vector{NTuple{3,Float64}},
            color::Symbol,
            alpha::Float64,
            markersize::Float64) =
    Point3Layer(points, color, alpha, markersize, :viridis)

Point3Layer(points::Vector{NTuple{3,Float64}},
            color::Vector{Float64},
            alpha::Float64,
            markersize::Float64) =
    Point3Layer(points, color, alpha, markersize, :viridis)

struct TextLayer <: AbstractVisualizationLayer
    labels::Vector{String}
    positions::Vector{NTuple{2,Float64}}
    color::Symbol
    textsize::Float64
end

struct Segment3Layer <: AbstractVisualizationLayer
    segments::Vector{NTuple{6,Float64}}
    color::Symbol
    alpha::Float64
    linewidth::Float64
end

struct BarcodeLayer <: AbstractVisualizationLayer
    intervals::Vector{NTuple{2,Float64}}
    multiplicities::Vector{Int}
    color::Symbol
    linewidth::Float64
    ystart::Float64
    ystep::Float64
end

struct VisualizationSpec
    kind::Symbol
    title::String
    subtitle::String
    layers::Vector{AbstractVisualizationLayer}
    panels::Vector{VisualizationSpec}
    axes::NamedTuple
    legend::NamedTuple
    interaction::NamedTuple
    metadata::NamedTuple
end

struct VisualizationValidationSummary{R}
    report::R
end

"""
    VisualExportResult

Summary of one completed visualization export.

This is the canonical return type for the simple export surface
`save_visual(outdir, stem, obj; ...)` and `save_visuals(outdir, requests; ...)`.
It records which file was written, which backend actually handled the export,
which on-disk format was produced, and which visualization family was saved.

Use `describe(result)` or the cheap scalar accessors `export_path`,
`export_backend`, `export_format`, `export_kind`, and `export_stem` when you
want to inspect an export from a notebook without digging through raw fields.
"""
struct VisualExportResult
    path::String
    backend::Symbol
    format::Symbol
    kind::Symbol
    stem::String
end

@inline export_path(result::VisualExportResult) = result.path
@inline export_backend(result::VisualExportResult) = result.backend
@inline export_format(result::VisualExportResult) = result.format
@inline export_kind(result::VisualExportResult) = result.kind
@inline export_stem(result::VisualExportResult) = result.stem

function describe(result::VisualExportResult)
    return (
        kind = :visual_export_result,
        path = export_path(result),
        backend = export_backend(result),
        format = export_format(result),
        visual_kind = export_kind(result),
        stem = export_stem(result),
    )
end

function Base.show(io::IO, result::VisualExportResult)
    d = describe(result)
    print(io,
          "VisualExportResult(",
          "stem=", repr(d.stem),
          ", visual_kind=", d.visual_kind,
          ", format=", d.format,
          ", backend=", d.backend,
          ")")
end

function Base.show(io::IO, ::MIME"text/plain", result::VisualExportResult)
    d = describe(result)
    print(io,
          "VisualExportResult",
          "\n  stem: ", repr(d.stem),
          "\n  visual_kind: ", d.visual_kind,
          "\n  format: ", d.format,
          "\n  backend: ", d.backend,
          "\n  path: ", repr(d.path))
end

function VisualizationSpec(kind::Symbol;
                           title::AbstractString="",
                           subtitle::AbstractString="",
                           layers::Vector{AbstractVisualizationLayer}=AbstractVisualizationLayer[],
                           panels::Vector{VisualizationSpec}=VisualizationSpec[],
                           axes::NamedTuple=(; xlabel="x", ylabel="y", xlimits=nothing, ylimits=nothing,
                                             zlabel="z", zlimits=nothing,
                                             aspect=:auto, xticks=nothing, yticks=nothing),
                           legend::NamedTuple=(; visible=false, title="", entries=NamedTuple()),
                           interaction::NamedTuple=(; hover=false, labels=false, clicks=false,
                                                    widgets=(), notebook=:summary_card),
                           metadata::NamedTuple=NamedTuple())
    return VisualizationSpec(kind, String(title), String(subtitle), layers, panels, axes, legend, interaction, metadata)
end

@inline visual_kind(spec::VisualizationSpec) = spec.kind
@inline visual_layers(spec::VisualizationSpec) = spec.layers
@inline visual_panels(spec::VisualizationSpec) = spec.panels
@inline visual_axes(spec::VisualizationSpec) = spec.axes
@inline visual_metadata(spec::VisualizationSpec) = spec.metadata
@inline visual_legend(spec::VisualizationSpec) = spec.legend
@inline visual_interaction(spec::VisualizationSpec) = spec.interaction

@inline visual_summary(spec::VisualizationSpec) = describe(spec)

function describe(spec::VisualizationSpec)
    return (
        kind = :visualization_spec,
        visual_kind = visual_kind(spec),
        title = spec.title,
        subtitle = spec.subtitle,
        nlayers = length(spec.layers),
        npanels = length(spec.panels),
        layer_types = Tuple(Symbol(nameof(typeof(layer))) for layer in spec.layers),
        axes = spec.axes,
        metadata = spec.metadata,
        legend_visible = get(spec.legend, :visible, false),
        interaction = spec.interaction,
    )
end

function Base.show(io::IO, spec::VisualizationSpec)
    d = describe(spec)
    print(io,
          "VisualizationSpec(",
          "kind=", d.visual_kind,
          ", nlayers=", d.nlayers,
          ", npanels=", d.npanels,
          ", title=", repr(d.title),
          ")")
end

function Base.show(io::IO, ::MIME"text/plain", spec::VisualizationSpec)
    d = describe(spec)
    print(io,
          "VisualizationSpec",
          "\n  visual_kind: ", d.visual_kind,
          "\n  title: ", repr(d.title),
          "\n  subtitle: ", repr(d.subtitle),
          "\n  nlayers: ", d.nlayers,
          "\n  npanels: ", d.npanels,
          "\n  layer_types: ", repr(d.layer_types),
          "\n  axes: ", repr(d.axes),
          "\n  metadata: ", repr(d.metadata))
end

function Base.show(io::IO, ::MIME"text/html", spec::VisualizationSpec)
    d = describe(spec)
    print(io,
          "<div style='font-family:monospace;border:1px solid #d0d0d0;padding:0.75rem;border-radius:0.5rem'>",
          "<div><strong>VisualizationSpec</strong></div>",
          "<div>kind: <code>", d.visual_kind, "</code></div>",
          "<div>title: ", escape_string(d.title), "</div>",
          "<div>layers: ", d.nlayers, "</div>",
          "<div>panels: ", d.npanels, "</div>",
          "<div>layer types: <code>", join(string.(d.layer_types), ", "), "</code></div>",
          "</div>")
end

@inline Base.length(spec::VisualizationSpec) = length(spec.layers)

function Base.show(io::IO, summary::VisualizationValidationSummary)
    report = summary.report
    print(io, "VisualizationValidationSummary(valid=", get(report, :valid, false),
          ", issues=", length(get(report, :issues, String[])), ")")
end

function Base.show(io::IO, ::MIME"text/plain", summary::VisualizationValidationSummary)
    report = summary.report
    print(io,
          "VisualizationValidationSummary",
          "\n  valid: ", get(report, :valid, false),
          "\n  kind: ", get(report, :kind, :unknown),
          "\n  issues: ", repr(get(report, :issues, String[])))
end

@inline _visual_validation_summary(report::NamedTuple) = VisualizationValidationSummary(report)

@inline function _default_axes_2d(; xlabel::AbstractString="x", ylabel::AbstractString="y",
                                   xlimits=nothing, ylimits=nothing,
                                   aspect=:data, xticks=nothing, yticks=nothing)
    return (; xlabel=String(xlabel), ylabel=String(ylabel),
            xlimits=xlimits, ylimits=ylimits,
            zlabel="z", zlimits=nothing,
            aspect=aspect, xticks=xticks, yticks=yticks)
end

@inline function _default_axes_3d(; xlabel::AbstractString="x", ylabel::AbstractString="y",
                                   zlabel::AbstractString="z",
                                   xlimits=nothing, ylimits=nothing, zlimits=nothing,
                                   aspect=:data, xticks=nothing, yticks=nothing)
    return (; xlabel=String(xlabel), ylabel=String(ylabel), zlabel=String(zlabel),
            xlimits=xlimits, ylimits=ylimits, zlimits=zlimits,
            aspect=aspect, xticks=xticks, yticks=yticks)
end

@inline function _default_interaction(; hover::Bool=false, labels::Bool=false,
                                       clicks::Bool=false, widgets=(),
                                       notebook=:summary_card)
    return (; hover, labels, clicks, widgets, notebook)
end

@inline function _default_legend(; visible::Bool=false, title::AbstractString="", entries=NamedTuple())
    return (; visible, title=String(title), entries)
end

function _finite_extrema(values)
    vals = Float64[]
    for x in values
        xf = float(x)
        isfinite(xf) && push!(vals, xf)
    end
    isempty(vals) && return nothing
    return (minimum(vals), maximum(vals))
end

@inline _midpoint(rect::NTuple{4,Float64}) = (0.5 * (rect[1] + rect[3]), 0.5 * (rect[2] + rect[4]))

function _color_from_sign(weight::Real; positive::Symbol=:crimson, negative::Symbol=:royalblue, zero::Symbol=:gray50)
    w = float(weight)
    return w > 0 ? positive : (w < 0 ? negative : zero)
end

function _bbox_from_points(points::AbstractVector{<:NTuple{2,<:Real}})
    isempty(points) && return nothing
    xs = Float64[float(p[1]) for p in points]
    ys = Float64[float(p[2]) for p in points]
    return ((minimum(xs), maximum(xs)), (minimum(ys), maximum(ys)))
end
