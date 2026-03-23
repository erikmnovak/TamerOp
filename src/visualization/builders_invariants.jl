# Invariant visualization builders.

function _expand_barcode_intervals(bar)
    if bar isa AbstractDict{<:Tuple{<:Real,<:Real},<:Integer}
        intervals = NTuple{2,Float64}[]
        multiplicities = Int[]
        for (iv, mult) in pairs(bar)
            push!(intervals, (float(iv[1]), float(iv[2])))
            push!(multiplicities, Int(mult))
        end
        sort!(eachindex(intervals), by=i -> (intervals[i][1], intervals[i][2]))
        return intervals, multiplicities
    elseif bar isa AbstractVector{<:Tuple{<:Real,<:Real}}
        return NTuple{2,Float64}[(float(iv[1]), float(iv[2])) for iv in bar], ones(Int, length(bar))
    else
        throw(ArgumentError("Unsupported barcode payload $(typeof(bar)); use unpacked slice barcodes for visualization."))
    end
end

function _barcode_count(bar)
    ivs, mults = _expand_barcode_intervals(bar)
    return sum(mults)
end

const _INVARIANT_VALUE_COLORS = (
    :gray80,
    :lightskyblue2,
    :cornflowerblue,
    :royalblue3,
    :seagreen3,
    :goldenrod2,
    :tomato2,
    :orchid3,
)

@inline _invariant_value_color(i::Int) =
    _INVARIANT_VALUE_COLORS[1 + mod(i - 1, length(_INVARIANT_VALUE_COLORS))]

function _integer_value_palette(values::AbstractVector{<:Integer})
    uniq = sort!(unique(Int.(values)))
    cmap = Dict{Int,Symbol}()
    for (i, value) in enumerate(uniq)
        cmap[value] = _invariant_value_color(i)
    end
    return uniq, cmap
end

function _integer_value_legend(values::AbstractVector{<:Integer}; title::AbstractString)
    uniq, cmap = _integer_value_palette(values)
    entries = (; (Symbol("v" * string(i)) => (; label=string(v), color=cmap[v], style=:patch)
                   for (i, v) in enumerate(uniq))...)
    return _default_legend(visible=!isempty(uniq), title=title, entries=entries)
end

function _rank_matrix(result::RankInvariantResult)
    Q = source_poset(result)
    n = nvertices(Q)
    vals = fill(NaN, n, n)
    @inbounds for a in 1:n, b in 1:n
        leq(Q, a, b) || continue
        vals[a, b] = float(value_at(result, a, b))
    end
    return vals
end

function _rank_tile_geometry(result::RankInvariantResult)
    Q = source_poset(result)
    n = nvertices(Q)
    rects = NTuple{4,Float64}[]
    values = Int[]
    label_pts = NTuple{2,Float64}[]
    labels = String[]
    @inbounds for a in 1:n, b in 1:n
        leq(Q, a, b) || continue
        push!(rects, (float(b) - 0.45, float(a) - 0.45, float(b) + 0.45, float(a) + 0.45))
        rank_ab = Int(value_at(result, a, b))
        push!(values, rank_ab)
        if rank_ab != 0
            push!(label_pts, (float(b), float(a)))
            push!(labels, string(rank_ab))
        end
    end
    return rects, values, label_pts, labels
end

function _hilbert_bar_rects(dims::AbstractVector{<:Integer})
    vals = Int.(dims)
    rects = NTuple{4,Float64}[]
    label_pts = NTuple{2,Float64}[]
    labels = String[]
    ymax = isempty(vals) ? 1.0 : max(1.0, float(maximum(vals)))
    ypad = 0.04 * ymax
    for i in eachindex(vals)
        h = float(vals[i])
        push!(rects, (float(i) - 0.35, 0.0, float(i) + 0.35, h))
        push!(label_pts, (float(i), h + ypad))
        push!(labels, string(vals[i]))
    end
    return rects, label_pts, labels
end

function _hilbert_curve_path(dims::AbstractVector{<:Integer})
    vals = Int.(dims)
    pts = NTuple{2,Float64}[(float(i), float(vals[i])) for i in eachindex(vals)]
    return pts
end

_has_hilbert_heatmap_geometry(::GridEncodingMap{2}) = true
_has_hilbert_heatmap_geometry(::PLEncodingMapBoxes) = true
_has_hilbert_heatmap_geometry(::ZnEncodingMap) = true
_has_hilbert_heatmap_geometry(enc::CompiledEncoding) = _has_hilbert_heatmap_geometry(encoding_map(enc))
_has_hilbert_heatmap_geometry(_) = false

_has_rank_query_geometry(::AbstractPLikeEncodingMap) = true
_has_rank_query_geometry(enc::CompiledEncoding) = _has_rank_query_geometry(encoding_map(enc))
_has_rank_query_geometry(_) = false

function _hilbert_heatmap_spec(pi::GridEncodingMap{2}, dims::AbstractVector{<:Integer})
    rects, _, axes = _grid_rectangles_2d(pi)
    length(rects) == length(dims) || throw(ArgumentError("hilbert_heatmap expects one Hilbert value per grid region."))
    values = Int.(dims)
    uniq, cmap = _integer_value_palette(values)
    layers = AbstractVisualizationLayer[]
    for v in uniq
        idx = findall(==(v), values)
        isempty(idx) || push!(layers, RectLayer(rects[idx], cmap[v], :black, 0.48, 0.8))
    end
    push!(layers, _text_layer_from_labels(_rect_text_centers(rects), string.(values); color=:black, textsize=10.0))
    return VisualizationSpec(:hilbert_heatmap;
                             title="Restricted Hilbert heatmap",
                             subtitle="piecewise-constant dimensions on encoding regions",
                             layers=layers,
                             axes=axes,
                             metadata=(; object=:restricted_hilbert, nregions=length(values), max_dim=maximum(values),
                                        figure_size=(760, 620), legend_position=:right),
                             legend=_integer_value_legend(values; title="dim"),
                             interaction=_default_interaction(hover=true, labels=true))
end

function _hilbert_heatmap_spec(pi::PLEncodingMapBoxes, dims::AbstractVector{<:Integer}; box=nothing)
    region_rects, labels, axes, label_positions = _pl_rectangles_2d(pi; box=box)
    region_ids = parse.(Int, labels)
    values = Int[dims[r] for r in region_ids]
    uniq, cmap = _integer_value_palette(values)
    layers = AbstractVisualizationLayer[]
    for (rects_r, v) in zip(region_rects, values)
        color = cmap[v]
        push!(layers, RectLayer(rects_r, color, :black, 0.28, 0.8))
        push!(layers, SegmentLayer(_region_boundary_segments(rects_r), :black, 0.7, 1.0))
    end
    label_text_positions = _offset_label_positions(label_positions, axes)
    push!(layers, _text_layer_from_labels(label_text_positions, string.(values); color=:black, textsize=10.0))
    return VisualizationSpec(:hilbert_heatmap;
                             title="Restricted Hilbert heatmap",
                             subtitle="piecewise-constant dimensions on encoding regions",
                             layers=layers,
                             axes=axes,
                             metadata=(; object=:restricted_hilbert, nregions=length(values), max_dim=maximum(values),
                                        figure_size=(860, 620), legend_position=:right),
                             legend=_integer_value_legend(values; title="dim"),
                             interaction=_default_interaction(hover=true, labels=true))
end

function _hilbert_heatmap_spec(pi::ZnEncodingMap, dims::AbstractVector{<:Integer})
    rects, _, axes = _zn_rectangles_2d(pi)
    length(rects) == length(dims) || throw(ArgumentError("hilbert_heatmap expects one Hilbert value per Zn region."))
    values = Int.(dims)
    uniq, cmap = _integer_value_palette(values)
    layers = AbstractVisualizationLayer[]
    for v in uniq
        idx = findall(==(v), values)
        isempty(idx) || push!(layers, RectLayer(rects[idx], cmap[v], :black, 0.48, 0.8))
    end
    push!(layers, _text_layer_from_labels(_rect_text_centers(rects), string.(values); color=:black, textsize=10.0))
    return VisualizationSpec(:hilbert_heatmap;
                             title="Restricted Hilbert heatmap",
                             subtitle="dimension values on the Zn encoding support",
                             layers=layers,
                             axes=axes,
                             metadata=(; object=:restricted_hilbert, nregions=length(values), max_dim=maximum(values),
                                        figure_size=(760, 620), legend_position=:right),
                             legend=_integer_value_legend(values; title="dim"),
                             interaction=_default_interaction(hover=true, labels=true))
end

_hilbert_heatmap_spec(enc::CompiledEncoding, dims::AbstractVector{<:Integer}; kwargs...) =
    _hilbert_heatmap_spec(encoding_map(enc), dims; kwargs...)

@inline function _cohomology_support_legend()
    return _default_legend(visible=true,
                           title="support",
                           entries=(;
                               supported=(; label="nonzero", color=:chartreuse3, style=:patch),
                               unsupported=(; label="zero", color=:gray90, style=:patch),
                           ))
end

@inline _axes_with_auto_aspect(axes::NamedTuple) = merge(axes, (; aspect=:auto))

function _cohomology_support_spec(pi::GridEncodingMap{2}, dims::AbstractVector{<:Integer}; degree::Int)
    rects, _, axes = _grid_rectangles_2d(pi)
    length(rects) == length(dims) || throw(ArgumentError("cohomology_support expects one dimension value per grid region."))
    values = Int.(dims)
    supported = values .> 0
    layers = AbstractVisualizationLayer[
        RectLayer(rects, :gray90, :gray60, 0.92, 0.8),
    ]
    if any(supported)
        push!(layers, RectLayer(rects[supported], :chartreuse3, :black, 0.62, 1.0))
        push!(layers, _text_layer_from_labels(_rect_text_centers(rects[supported]),
                                              string.(values[supported]);
                                              color=:black,
                                              textsize=10.0))
    end
    return VisualizationSpec(:cohomology_support;
                             title="H^$(degree) region support",
                             subtitle="nonzero cohomology dimensions on encoding regions",
                             layers=layers,
                             axes=axes,
                             metadata=(; object=:cohomology_dims,
                                        degree,
                                        nregions=length(values),
                                        support_count=count(supported),
                                        max_dim=isempty(values) ? 0 : maximum(values),
                                        figure_size=(760, 620),
                                        legend_position=:right),
                             legend=_cohomology_support_legend(),
                             interaction=_default_interaction(hover=true, labels=any(supported)))
end

function _cohomology_support_spec(pi::PLEncodingMapBoxes, dims::AbstractVector{<:Integer}; degree::Int, box=nothing)
    region_rects, labels, axes, label_positions = _pl_rectangles_2d(pi; box=box)
    region_ids = parse.(Int, labels)
    values = Int[dims[r] for r in region_ids]
    supported = values .> 0
    layers = AbstractVisualizationLayer[]
    for (rects_r, keep) in zip(region_rects, supported)
        push!(layers, RectLayer(rects_r, keep ? :chartreuse3 : :gray90, :black, keep ? 0.42 : 0.18, 0.8))
        push!(layers, SegmentLayer(_region_boundary_segments(rects_r), :black, 0.7, 1.0))
    end
    if any(supported)
        text_positions = _offset_label_positions(label_positions[supported], axes)
        push!(layers, _text_layer_from_labels(text_positions,
                                              string.(values[supported]);
                                              color=:black,
                                              textsize=10.0))
    end
    return VisualizationSpec(:cohomology_support;
                             title="H^$(degree) region support",
                             subtitle="nonzero cohomology dimensions on encoding regions",
                             layers=layers,
                             axes=axes,
                             metadata=(; object=:cohomology_dims,
                                        degree,
                                        nregions=length(values),
                                        support_count=count(supported),
                                        max_dim=isempty(values) ? 0 : maximum(values),
                                        figure_size=(860, 620),
                                        legend_position=:right),
                             legend=_cohomology_support_legend(),
                             interaction=_default_interaction(hover=true, labels=any(supported)))
end

function _cohomology_support_spec(pi::ZnEncodingMap, dims::AbstractVector{<:Integer}; degree::Int)
    rects, _, axes = _zn_rectangles_2d(pi)
    length(rects) == length(dims) || throw(ArgumentError("cohomology_support expects one dimension value per Zn region."))
    values = Int.(dims)
    supported = values .> 0
    layers = AbstractVisualizationLayer[
        RectLayer(rects, :gray90, :gray60, 0.92, 0.8),
    ]
    if any(supported)
        push!(layers, RectLayer(rects[supported], :chartreuse3, :black, 0.62, 1.0))
        push!(layers, _text_layer_from_labels(_rect_text_centers(rects[supported]),
                                              string.(values[supported]);
                                              color=:black,
                                              textsize=10.0))
    end
    return VisualizationSpec(:cohomology_support;
                             title="H^$(degree) region support",
                             subtitle="nonzero cohomology dimensions on the Zn encoding support",
                             layers=layers,
                             axes=axes,
                             metadata=(; object=:cohomology_dims,
                                        degree,
                                        nregions=length(values),
                                        support_count=count(supported),
                                        max_dim=isempty(values) ? 0 : maximum(values),
                                        figure_size=(760, 620),
                                        legend_position=:right),
                             legend=_cohomology_support_legend(),
                             interaction=_default_interaction(hover=true, labels=any(supported)))
end

_cohomology_support_spec(enc::CompiledEncoding, dims::AbstractVector{<:Integer}; kwargs...) =
    _cohomology_support_spec(encoding_map(enc), dims; kwargs...)

function _cohomology_support_plane_spec(pi::GridEncodingMap{2}, dims::AbstractVector{<:Integer}; degree::Int)
    rects, _, axes = _grid_rectangles_2d(pi)
    length(rects) == length(dims) || throw(ArgumentError("cohomology_support_plane expects one dimension value per grid region."))
    values = Int.(dims)
    supported = values .> 0
    layers = AbstractVisualizationLayer[
        RectLayer(rects, :gray90, :gray90, 0.98, 0.2),
    ]
    any(supported) && push!(layers, RectLayer(rects[supported], :chartreuse3, :chartreuse3, 0.92, 0.2))
    return VisualizationSpec(:cohomology_support_plane;
                             title="H^$(degree) support plane",
                             subtitle="cohomology support on the bifiltration parameter plane",
                             layers=layers,
                             axes=_axes_with_auto_aspect(axes),
                             metadata=(; object=:cohomology_dims,
                                        degree,
                                        nregions=length(values),
                                        support_count=count(supported),
                                        max_dim=isempty(values) ? 0 : maximum(values),
                                        figure_size=(820, 520),
                                        legend_position=:right),
                             legend=_cohomology_support_legend(),
                             interaction=_default_interaction(hover=true, labels=false))
end

function _cohomology_support_plane_spec(pi::PLEncodingMapBoxes, dims::AbstractVector{<:Integer}; degree::Int, box=nothing)
    region_rects, labels, axes, _ = _pl_rectangles_2d(pi; box=box)
    region_ids = parse.(Int, labels)
    values = Int[dims[r] for r in region_ids]
    supported = values .> 0
    layers = AbstractVisualizationLayer[]
    for (rects_r, keep) in zip(region_rects, supported)
        push!(layers, RectLayer(rects_r, keep ? :chartreuse3 : :gray90, keep ? :chartreuse3 : :gray90, keep ? 0.90 : 0.98, 0.2))
    end
    return VisualizationSpec(:cohomology_support_plane;
                             title="H^$(degree) support plane",
                             subtitle="cohomology support on the bifiltration parameter plane",
                             layers=layers,
                             axes=_axes_with_auto_aspect(axes),
                             metadata=(; object=:cohomology_dims,
                                        degree,
                                        nregions=length(values),
                                        support_count=count(supported),
                                        max_dim=isempty(values) ? 0 : maximum(values),
                                        figure_size=(880, 520),
                                        legend_position=:right),
                             legend=_cohomology_support_legend(),
                             interaction=_default_interaction(hover=true, labels=false))
end

function _cohomology_support_plane_spec(pi::ZnEncodingMap, dims::AbstractVector{<:Integer}; degree::Int)
    rects, _, axes = _zn_rectangles_2d(pi)
    length(rects) == length(dims) || throw(ArgumentError("cohomology_support_plane expects one dimension value per Zn region."))
    values = Int.(dims)
    supported = values .> 0
    layers = AbstractVisualizationLayer[
        RectLayer(rects, :gray90, :gray90, 0.98, 0.2),
    ]
    any(supported) && push!(layers, RectLayer(rects[supported], :chartreuse3, :chartreuse3, 0.92, 0.2))
    return VisualizationSpec(:cohomology_support_plane;
                             title="H^$(degree) support plane",
                             subtitle="cohomology support on the bifiltration parameter plane",
                             layers=layers,
                             axes=_axes_with_auto_aspect(axes),
                             metadata=(; object=:cohomology_dims,
                                        degree,
                                        nregions=length(values),
                                        support_count=count(supported),
                                        max_dim=isempty(values) ? 0 : maximum(values),
                                        figure_size=(820, 520),
                                        legend_position=:right),
                             legend=_cohomology_support_legend(),
                             interaction=_default_interaction(hover=true, labels=false))
end

_cohomology_support_plane_spec(enc::CompiledEncoding, dims::AbstractVector{<:Integer}; kwargs...) =
    _cohomology_support_plane_spec(encoding_map(enc), dims; kwargs...)

function _hilbert_bar_spec(dims::AbstractVector{<:Integer})
    vals = Int.(dims)
    rects, label_pts, labels = _hilbert_bar_rects(vals)
    ymax = isempty(vals) ? 1.0 : max(1.0, float(maximum(vals)))
    return VisualizationSpec(:hilbert_bars;
                             title="Restricted Hilbert bars",
                             subtitle="dimension by encoding-region index",
                             layers=AbstractVisualizationLayer[
                                 RectLayer(rects, :cornflowerblue, :navy, 0.55, 1.0),
                                 _text_layer_from_labels(label_pts, labels; color=:black, textsize=9.0),
                             ],
                             axes=_default_axes_2d(xlabel="region index", ylabel="dimension",
                                                   xlimits=(0.5, float(length(vals)) + 0.5),
                                                   ylimits=(0.0, 1.12 * ymax),
                                                   aspect=:auto,
                                                   xticks=(collect(1:length(vals)), string.(collect(1:length(vals))))),
                             metadata=(; object=:restricted_hilbert, nregions=length(vals), max_dim=isempty(vals) ? 0 : maximum(vals),
                                        figure_size=(760, 460), legend_position=:none))
end

function _hilbert_curve_spec(dims::AbstractVector{<:Integer})
    vals = Int.(dims)
    path = _hilbert_curve_path(vals)
    ymax = isempty(vals) ? 1.0 : max(1.0, float(maximum(vals)))
    return VisualizationSpec(:restricted_hilbert_curve;
                             title="Restricted Hilbert curve",
                             subtitle="dimension along encoding-region index order",
                             layers=AbstractVisualizationLayer[
                                 PolylineLayer([path], :royalblue3, 1.0, 2.5, false),
                                 PointLayer(path, :royalblue3, 1.0, 10.0),
                             ],
                             axes=_default_axes_2d(xlabel="region index", ylabel="dimension",
                                                   xlimits=(0.5, float(length(vals)) + 0.5),
                                                   ylimits=(0.0, 1.12 * ymax),
                                                   aspect=:auto,
                                                   xticks=(collect(1:length(vals)), string.(collect(1:length(vals))))),
                             metadata=(; object=:restricted_hilbert, nregions=length(vals), max_dim=isempty(vals) ? 0 : maximum(vals),
                                        figure_size=(760, 460), legend_position=:none))
end

function _as_query_pair(pair)
    if pair isa Tuple || pair isa AbstractVector
        length(pair) == 2 || throw(ArgumentError("rank_query_overlay expects each query pair to contain exactly two points."))
        return (only(_as_points2(pair[1])), only(_as_points2(pair[2])))
    end
    throw(ArgumentError("rank_query_overlay expects query pairs of the form (x, y)."))
end

function _collect_rank_query_pairs(; pair=nothing, pairs=nothing)
    out = Tuple{NTuple{2,Float64},NTuple{2,Float64}}[]
    pair === nothing || push!(out, _as_query_pair(pair))
    if pairs !== nothing
        for qpair in pairs
            push!(out, _as_query_pair(qpair))
        end
    end
    return out
end

function _barcode_spec(bar; title::AbstractString="Barcode")
    intervals, multiplicities = _expand_barcode_intervals(bar)
    isempty(intervals) && (intervals = NTuple{2,Float64}[(0.0, 0.0)]; multiplicities = [0])
    births = [iv[1] for iv in intervals]
    deaths = [iv[2] for iv in intervals]
    ycount = max(1, sum(multiplicities))
    return VisualizationSpec(:barcode;
                             title=title,
                             subtitle="1-parameter interval decomposition",
                             layers=AbstractVisualizationLayer[
                                 BarcodeLayer(intervals, multiplicities, :navy, 2.0, 1.0, 1.0),
                             ],
                             axes=_default_axes_2d(xlabel="birth", ylabel="interval index",
                                                   xlimits=(minimum(births), maximum(deaths)),
                                                   ylimits=(0.0, float(ycount + 1)),
                                                   aspect=:auto),
                             metadata=(; barcode_count=length(intervals), total_multiplicity=sum(multiplicities)))
end

available_visuals(::AbstractDict{<:Tuple{<:Real,<:Real},<:Integer}) = (:barcode,)
available_visuals(::AbstractVector{<:Tuple{<:Real,<:Real}}) = (:barcode,)
_visual_spec(bar::AbstractDict{<:Tuple{<:Real,<:Real},<:Integer}, kind::Symbol; kwargs...) = kind === :barcode ? _barcode_spec(bar; title="Barcode") : throw(ArgumentError("Unsupported barcode visualization kind $(kind)."))
_visual_spec(bar::AbstractVector{<:Tuple{<:Real,<:Real}}, kind::Symbol; kwargs...) = kind === :barcode ? _barcode_spec(bar; title="Barcode") : throw(ArgumentError("Unsupported barcode visualization kind $(kind)."))

available_visuals(::RankInvariantResult) = (:rank_heatmap, :rank_rectangles)
available_visuals(res::CohomologyDimsResult) =
    _has_hilbert_heatmap_geometry(encoding_map(res)) ? (:cohomology_support_plane, :cohomology_support) : ()

function available_visuals(inv::InvariantResult)
    val = invariant_value(inv)
    if inv.which === :rank_invariant && val isa RankInvariantResult
        kinds = Symbol[:rank_heatmap, :rank_rectangles]
        _has_rank_query_geometry(encoding_map(inv)) && push!(kinds, :rank_query_overlay)
        return Tuple(kinds)
    elseif inv.which === :restricted_hilbert && val isa AbstractVector{<:Integer}
        kinds = Symbol[]
        _has_hilbert_heatmap_geometry(encoding_map(inv)) && push!(kinds, :hilbert_heatmap)
        append!(kinds, [:hilbert_bars, :restricted_hilbert_curve])
        return Tuple(kinds)
    end
    return ()
end

function _visual_spec(res::CohomologyDimsResult, kind::Symbol; kwargs...)
    _ = kwargs
    kind in (:cohomology_support_plane, :cohomology_support) ||
        throw(ArgumentError("CohomologyDimsResult supports kind=:cohomology_support_plane or :cohomology_support only."))
    pi = encoding_map(res)
    _has_hilbert_heatmap_geometry(pi) ||
        throw(ArgumentError("cohomology support visuals require a 2D encoding with region geometry in its provenance."))
    if kind === :cohomology_support_plane
        return _cohomology_support_plane_spec(pi, cohomology_dims(res); degree=res.degree)
    end
    return _cohomology_support_spec(pi, cohomology_dims(res); degree=res.degree)
end

function _visual_spec(result::RankInvariantResult, kind::Symbol; kwargs...)
    _ = kwargs
    Q = source_poset(result)
    n = nvertices(Q)
    if kind === :rank_heatmap
        vals = _rank_matrix(result)
        ticks = (collect(1:n), string.(collect(1:n)))
        return VisualizationSpec(:rank_heatmap;
                                 title="Rank invariant heatmap",
                                 subtitle="rank values on comparable poset pairs",
                                 layers=AbstractVisualizationLayer[
                                     # Makie heatmaps interpret the first matrix axis along x,
                                     # so transpose to keep x=b, y=a consistent with the tile view.
                                     HeatmapLayer(Float64[1:n;], Float64[1:n;], permutedims(vals), :viridis, 1.0, "rank"),
                                 ],
                                 axes=_default_axes_2d(xlabel="target region b", ylabel="source region a",
                                                       xlimits=(0.5, float(n) + 0.5),
                                                       ylimits=(0.5, float(n) + 0.5),
                                                       aspect=:auto,
                                                       xticks=ticks, yticks=ticks),
                                 metadata=(; object=:rank_invariant, nvertices=n, nstored=length(result),
                                            figure_size=(720, 620), legend_position=:none))
    elseif kind === :rank_rectangles
        rects, values, label_pts, labels = _rank_tile_geometry(result)
        uniq, cmap = _integer_value_palette(values)
        layers = AbstractVisualizationLayer[]
        for v in uniq
            idx = findall(==(v), values)
            isempty(idx) || push!(layers, RectLayer(rects[idx], cmap[v], :black, 0.52, 0.8))
        end
        isempty(labels) || push!(layers, _text_layer_from_labels(label_pts, labels; color=:black, textsize=9.0))
        ticks = (collect(1:n), string.(collect(1:n)))
        return VisualizationSpec(:rank_rectangles;
                                 title="Rank invariant rectangles",
                                 subtitle="comparable region pairs shown as labeled tiles",
                                 layers=layers,
                                 axes=_default_axes_2d(xlabel="target region b", ylabel="source region a",
                                                       xlimits=(0.5, float(n) + 0.5),
                                                       ylimits=(0.5, float(n) + 0.5),
                                                       aspect=:equal,
                                                       xticks=ticks, yticks=ticks),
                                 metadata=(; object=:rank_invariant, nvertices=n, nstored=length(result),
                                            figure_size=(760, 620), legend_position=:right),
                                 legend=_integer_value_legend(values; title="rank"),
                                 interaction=_default_interaction(hover=true, labels=!isempty(labels)))
    end
    throw(ArgumentError("RankInvariantResult supports kind=:rank_heatmap or :rank_rectangles only."))
end

function _visual_spec(inv::InvariantResult, kind::Symbol; pair=nothing, pairs=nothing, kwargs...)
    _ = kwargs
    val = invariant_value(inv)
    if inv.which === :rank_invariant && val isa RankInvariantResult
        if kind === :rank_query_overlay
            pi = encoding_map(inv)
            _has_rank_query_geometry(pi) || throw(ArgumentError("rank_query_overlay requires an invariant result with 2D query geometry in its encoding provenance."))
            base = _visual_spec(pi, :regions)
            query_pairs = _collect_rank_query_pairs(pair=pair, pairs=pairs)
            isempty(query_pairs) && throw(ArgumentError("rank_query_overlay requires keyword pair or pairs."))
            segments = NTuple{4,Float64}[]
            xpts = NTuple{2,Float64}[]
            ypts = NTuple{2,Float64}[]
            label_pts = NTuple{2,Float64}[]
            labels = String[]
            Q = source_poset(val)
            for (i, (x, y)) in enumerate(query_pairs)
                report = check_rank_query_points(pi, collect(x), collect(y); throw=true)
                a = Int(report.x_region)
                b = Int(report.y_region)
                leq(Q, a, b) || throw(ArgumentError("rank_query_overlay requires located regions to satisfy a <= b; got a=$(a), b=$(b)."))
                rank_ab = value_at(val, a, b)
                push!(segments, (x[1], x[2], y[1], y[2]))
                push!(xpts, x)
                push!(ypts, y)
                mid = (0.5 * (x[1] + y[1]), 0.5 * (x[2] + y[2]))
                push!(label_pts, mid)
                push!(labels, "r$(i) = $(rank_ab)")
            end
            label_positions = _offset_label_positions(label_pts, base.axes; dx_frac=0.008, dy_frac=0.008)
            layers = copy(base.layers)
            push!(layers, SegmentLayer(segments, :orange3, 0.9, 2.0))
            push!(layers, PointLayer(xpts, :royalblue3, 0.95, 12.0))
            push!(layers, PointLayer(ypts, :crimson, 0.95, 12.0))
            push!(layers, _text_layer_from_labels(label_positions, labels; color=:black, textsize=10.0))
            base_entries = get(base.legend, :entries, NamedTuple())
            query_entries = (; query_start=(; label="x", color=:royalblue3, style=:marker),
                              query_end=(; label="y", color=:crimson, style=:marker),
                              rank_pair=(; label="query pair", color=:orange3, style=:line))
            legend = merge(base.legend, (; visible=true, entries=merge(base_entries, query_entries)))
            return VisualizationSpec(:rank_query_overlay;
                                     title="Rank query overlay",
                                     subtitle="query pairs on the encoding with rank labels",
                                     layers=layers,
                                     axes=base.axes,
                                     metadata=merge(base.metadata, (; object=:rank_invariant, npairs=length(query_pairs))),
                                     legend=legend,
                                     interaction=_default_interaction(hover=true, labels=true))
        end
        return _visual_spec(val, kind)
    elseif inv.which === :restricted_hilbert && val isa AbstractVector{<:Integer}
        if kind === :hilbert_heatmap
            pi = encoding_map(inv)
            _has_hilbert_heatmap_geometry(pi) || throw(ArgumentError("hilbert_heatmap requires an invariant result whose encoding provenance has 2D region geometry."))
            return _hilbert_heatmap_spec(pi, val)
        elseif kind === :hilbert_bars
            return _hilbert_bar_spec(val)
        elseif kind === :restricted_hilbert_curve
            return _hilbert_curve_spec(val)
        end
    end
    throw(ArgumentError("Unsupported invariant visualization kind $(kind) for InvariantResult(which=$(inv.which))."))
end

available_visuals(::SliceBarcodesResult) = (:barcode, :barcode_bank, :slice_family)

function _slice_barcode_at(result::SliceBarcodesResult, index)
    bars = slice_barcodes(result)
    if index === nothing
        if length(bars) != 1
            throw(ArgumentError("SliceBarcodesResult contains multiple barcodes; pass index=(i,j) or index=i."))
        end
        return first(bars)
    end
    return bars[index]
end

function _visual_spec(result::SliceBarcodesResult, kind::Symbol; index=nothing, kwargs...)
    if kind === :barcode
        bar = _slice_barcode_at(result, index)
        return _barcode_spec(bar; title="Slice barcode")
    elseif kind === :barcode_bank
        bars = slice_barcodes(result)
        counts = map(_barcode_count, bars)
        if ndims(counts) == 2
            vals = Float64.(counts)
            x = Float64[1:size(vals, 2);]
            y = Float64[1:size(vals, 1);]
        else
            vals = reshape(Float64.(collect(counts)), 1, :)
            x = Float64[1:size(vals, 2);]
            y = [1.0]
        end
        return VisualizationSpec(:barcode_bank;
                                 title="Barcode bank",
                                 subtitle="interval counts across the sampled slice family",
                                 layers=AbstractVisualizationLayer[
                                     HeatmapLayer(x, y, vals, :viridis, 1.0, "interval count"),
                                 ],
                                 axes=_default_axes_2d(xlabel="slice column", ylabel="slice row",
                                                       xlimits=(minimum(x), maximum(x)),
                                                       ylimits=(minimum(y), maximum(y))),
                                 metadata=(; barcode_shape=size(bars), total_weight=sum(slice_weights(result))))
    elseif kind === :slice_family
        dirs = slice_directions(result)
        offs = slice_offsets(result)
        if !isempty(dirs) && !isempty(offs)
            pts = NTuple{2,Float64}[]
            labels = String[]
            for (j, dir) in enumerate(dirs), (i, off) in enumerate(offs)
                length(dir) == 2 || continue
                push!(pts, (float(dir[1]), float(dir[2])))
                push!(labels, string(off))
            end
            bbox = _bbox_from_points(pts)
            return VisualizationSpec(:slice_family;
                                     title="Slice family",
                                     subtitle="direction representatives with offset labels",
                                     layers=AbstractVisualizationLayer[
                                         PointLayer(pts, :darkgreen, 0.9, 12.0),
                                         _text_layer_from_labels(pts, labels; color=:black, textsize=9.0),
                                     ],
                                     axes=_default_axes_2d(xlabel="dir_x", ylabel="dir_y",
                                                           xlimits=bbox === nothing ? nothing : bbox[1],
                                                           ylimits=bbox === nothing ? nothing : bbox[2],
                                                           aspect=:equal),
                                     metadata=(; ndirections=length(dirs), noffsets=length(offs), total_weight=sum(slice_weights(result))))
        end
        weights_vec = vec(Float64.(slice_weights(result)))
        pts = [(float(i), weights_vec[i]) for i in eachindex(weights_vec)]
        return VisualizationSpec(:slice_family;
                                 title="Slice family weights",
                                 subtitle="weights for explicitly provided slices",
                                 layers=AbstractVisualizationLayer[
                                     PointLayer(pts, :darkgreen, 0.9, 10.0),
                                 ],
                                 axes=_default_axes_2d(xlabel="slice index", ylabel="weight",
                                                       xlimits=(1.0, float(length(weights_vec))),
                                                       ylimits=_finite_extrema(weights_vec)))
    end
    throw(ArgumentError("Unsupported SliceBarcodesResult visualization kind $(kind)."))
end

available_visuals(::FiberedArrangement2D) = (:fibered_arrangement, :fibered_query, :fibered_cell_highlight, :fibered_tie_break,
                                          :fibered_offset_intervals, :fibered_projected_comparison)
available_visuals(::FiberedBarcodeCache2D) = (:fibered_arrangement, :fibered_query, :fibered_cell_highlight, :fibered_tie_break,
                                             :fibered_offset_intervals, :fibered_projected_comparison,
                                             :fibered_family, :fibered_chain_cells, :fibered_family_contributions,
                                             :fibered_distance_diagnostic, :fibered_query_barcode)
available_visuals(::FiberedSliceFamily2D) = (:fibered_family, :fibered_chain_cells, :fibered_family_contributions, :fibered_distance_diagnostic)
available_visuals(::FiberedSliceResult) = (:fibered_slice, :fibered_slice_overlay, :barcode)
available_visuals(::ProjectedArrangement) = (:projected_arrangement,)
available_visuals(::ProjectedBarcodesResult) = (:barcode_bank,)
available_visuals(::ProjectedDistancesResult) = (:projected_distances,)

@inline function _line_basepoint_from_offset_2d(dir::AbstractVector{<:Real}, off::Real)
    d1 = float(dir[1])
    d2 = float(dir[2])
    n1, n2 = -d2, d1
    nn = n1 * n1 + n2 * n2
    nn == 0.0 && throw(ArgumentError("fibered visualization requires a nonzero direction vector."))
    s = float(off) / nn
    return (s * n1, s * n2)
end

function _slice_segments_from_values(dir::AbstractVector{<:Real}, off::Real, vals::AbstractVector{<:Real})
    x0 = _line_basepoint_from_offset_2d(dir, off)
    d1 = float(dir[1])
    d2 = float(dir[2])
    segments = NTuple{4,Float64}[]
    mids = NTuple{2,Float64}[]
    for i in 1:max(length(vals) - 1, 0)
        t0 = float(vals[i])
        t1 = float(vals[i + 1])
        p = (x0[1] + t0 * d1, x0[2] + t0 * d2)
        q = (x0[1] + t1 * d1, x0[2] + t1 * d2)
        push!(segments, (p[1], p[2], q[1], q[2]))
        push!(mids, (0.5 * (p[1] + q[1]), 0.5 * (p[2] + q[2])))
    end
    return segments, mids
end

function _float_bucket_style(values::AbstractVector{<:Real}; title::AbstractString="value", nbuckets::Int=5, digits::Int=3)
    vals = Float64[]
    for v in values
        vf = float(v)
        isfinite(vf) && push!(vals, vf)
    end
    if isempty(vals)
        return fill(1, length(values)), Symbol[_invariant_value_color(1)], _default_legend(visible=false)
    end
    lo, hi = extrema(vals)
    if hi <= lo + 1e-12
        bins = fill(1, length(values))
        colors = Symbol[_invariant_value_color(1)]
        entries = [(; label=string(round(hi; digits=digits)), color=colors[1], style=:patch)]
        return bins, colors, _default_legend(visible=true, title=title, entries=entries)
    end
    nb = min(nbuckets, max(2, length(unique(round.(vals; digits=6)))))
    edges = collect(range(lo, hi; length=nb + 1))
    colors = Symbol[_invariant_value_color(i + 1) for i in 1:nb]
    bins = Int[]
    sizehint!(bins, length(values))
    for value in values
        b = searchsortedlast(edges, float(value)) - 1
        push!(bins, clamp(b, 1, nb))
    end
    entries = NamedTuple[]
    for i in 1:nb
        a = round(edges[i]; digits=digits)
        b = round(edges[i + 1]; digits=digits)
        label = i < nb ? "[$a, $b)" : "[$a, $b]"
        push!(entries, (; label, color=colors[i], style=:patch))
    end
    return bins, colors, _default_legend(visible=true, title=title, entries=entries)
end

function _resolve_fibered_caches(owner, caches)
    (caches isa Tuple || caches isa AbstractVector) ||
        throw(ArgumentError("fibered exact-family visuals require keyword caches=(cacheA, cacheB)."))
    length(caches) == 2 || throw(ArgumentError("caches must contain exactly two FiberedBarcodeCache2D objects."))
    cacheA, cacheB = caches
    cacheA isa FiberedBarcodeCache2D || throw(ArgumentError("first cache in caches must be a FiberedBarcodeCache2D."))
    cacheB isa FiberedBarcodeCache2D || throw(ArgumentError("second cache in caches must be a FiberedBarcodeCache2D."))
    arr = owner isa FiberedSliceFamily2D ? source_arrangement(owner) :
          owner isa FiberedArrangement2D ? owner :
          owner isa FiberedBarcodeCache2D ? shared_arrangement(owner) :
          nothing
    arr === nothing && throw(ArgumentError("unsupported fibered cache owner $(typeof(owner))."))
    shared_arrangement(cacheA) === arr || throw(ArgumentError("first cache in caches must share the visualized arrangement/family."))
    shared_arrangement(cacheB) === arr || throw(ArgumentError("second cache in caches must share the visualized arrangement/family."))
    return cacheA, cacheB
end

function _fibered_cell_axes(arr::FiberedArrangement2D, maxoff::Int; ylabel::AbstractString="offset cell")
    ndir = length(direction_representatives(arr))
    ymax = max(1, maxoff)
    xticks = ndir <= 16 ? (collect(1:ndir), string.(collect(1:ndir))) : nothing
    yticks = ymax <= 16 ? (collect(1:ymax), string.(collect(1:ymax))) : nothing
    return _default_axes_2d(xlabel="direction cell", ylabel=ylabel,
                            xlimits=(0.5, float(ndir) + 0.5),
                            ylimits=(0.5, float(ymax) + 0.5),
                            aspect=:auto, xticks=xticks, yticks=yticks)
end

function _fibered_family_metrics(fam::FiberedSliceFamily2D, caches)
    cacheA, cacheB = _resolve_fibered_caches(fam, caches)
    chain_counts = Dict{Int,Int}()
    for cid in fam.chain_id
        chain_counts[cid] = get(chain_counts, cid, 0) + 1
    end
    ns = length(fam.cell_id)
    contributions = Vector{Float64}(undef, ns)
    distances = Vector{Float64}(undef, ns)
    weights = Vector{Float64}(undef, ns)
    multiplicities = Vector{Int}(undef, ns)
    for k in 1:ns
        dir = slice_direction(fam, k)
        off = slice_offset(fam, k)
        bcA = slice_barcode(fibered_slice(cacheA, dir, off))
        bcB = slice_barcode(fibered_slice(cacheB, dir, off))
        d = bottleneck_distance(bcA, bcB)
        w = fam.dir_weight[fam.dir_idx[k]]
        distances[k] = d
        weights[k] = w
        contributions[k] = w * d
        multiplicities[k] = get(chain_counts, fam.chain_id[k], 1)
    end
    argmax_index = isempty(contributions) ? nothing : argmax(contributions)
    return (;
        contributions,
        distances,
        weights,
        multiplicities,
        chain_counts,
        max_contribution=isempty(contributions) ? 0.0 : maximum(contributions),
        argmax_index,
    )
end

function _fibered_cell_rects(fam::FiberedSliceFamily2D)
    return NTuple{4,Float64}[
        (float(fam.dir_idx[k]) - 0.45, float(fam.off_idx[k]) - 0.45,
         float(fam.dir_idx[k]) + 0.45, float(fam.off_idx[k]) + 0.45)
        for k in eachindex(fam.cell_id)
    ]
end

function _fibered_chain_cells_spec(fam::FiberedSliceFamily2D)
    rects = _fibered_cell_rects(fam)
    chain_ids = Int.(fam.chain_id)
    uniq, cmap = _integer_value_palette(chain_ids)
    layers = AbstractVisualizationLayer[]
    for cid in uniq
        idx = findall(==(cid), chain_ids)
        isempty(idx) || push!(layers, RectLayer(rects[idx], cmap[cid], :black, 0.72, 0.8))
    end
    if length(chain_ids) <= 48
        pts = NTuple{2,Float64}[(float(fam.dir_idx[k]), float(fam.off_idx[k])) for k in eachindex(fam.cell_id)]
        push!(layers, _text_layer_from_labels(pts, string.(chain_ids); color=:black, textsize=8.0))
    end
    legend = length(uniq) <= 8 ? _integer_value_legend(chain_ids; title="chain") : _default_legend(visible=false)
    arr = source_arrangement(fam)
    return VisualizationSpec(:fibered_chain_cells;
                             title="Fibered chain cells",
                             subtitle="nonempty arrangement cells colored by cached chain id",
                             layers=layers,
                             axes=_fibered_cell_axes(arr, isempty(fam.off_idx) ? 0 : maximum(fam.off_idx)),
                             metadata=(; object=:fibered_slice_family, nslices=length(fam.cell_id),
                                        unique_chains=length(fam.unique_chain_ids), figure_size=(780, 620),
                                        legend_position=length(uniq) <= 8 ? :right : :none),
                             legend=legend,
                             interaction=_default_interaction(hover=true, labels=length(chain_ids) <= 48))
end

function _fibered_contribution_cell_spec(fam::FiberedSliceFamily2D, metrics;
                                         title::AbstractString,
                                         subtitle::AbstractString,
                                         highlight_argmax::Bool=false)
    rects = _fibered_cell_rects(fam)
    bins, colors, legend = _float_bucket_style(metrics.contributions; title="w*d")
    layers = AbstractVisualizationLayer[]
    for b in sort!(unique(bins))
        idx = findall(==(b), bins)
        isempty(idx) || push!(layers, RectLayer(rects[idx], colors[b], :black, 0.78, 0.8))
    end
    top = isempty(metrics.contributions) ? Int[] : sortperm(metrics.contributions; rev=true)[1:min(6, length(metrics.contributions))]
    if !isempty(top)
        pts = NTuple{2,Float64}[(float(fam.dir_idx[k]), float(fam.off_idx[k])) for k in top]
        lbls = ["s$(k)=$(round(metrics.contributions[k]; digits=3))" for k in top]
        push!(layers, _text_layer_from_labels(pts, lbls; color=:black, textsize=8.0))
    end
    if highlight_argmax && metrics.argmax_index !== nothing
        push!(layers, SegmentLayer(_region_boundary_segments([rects[metrics.argmax_index]]), :black, 1.0, 2.4))
    end
    arr = source_arrangement(fam)
    return VisualizationSpec(:fibered_family_contributions;
                             title=title,
                             subtitle=subtitle,
                             layers=layers,
                             axes=_fibered_cell_axes(arr, isempty(fam.off_idx) ? 0 : maximum(fam.off_idx)),
                             metadata=(; object=:fibered_slice_family, nslices=length(fam.cell_id),
                                        unique_chains=length(fam.unique_chain_ids), matching_distance=metrics.max_contribution,
                                        argmax_index=metrics.argmax_index, figure_size=(820, 620), legend_position=:right),
                             legend=legend,
                             interaction=_default_interaction(hover=true, labels=true))
end

function _fibered_chain_multiplicity_panel(fam::FiberedSliceFamily2D, metrics)
    chain_ids = sort!(collect(keys(metrics.chain_counts)))
    counts = [metrics.chain_counts[cid] for cid in chain_ids]
    ymax = isempty(counts) ? 1.0 : max(1.0, float(maximum(counts)))
    rects = NTuple{4,Float64}[]
    labels = String[]
    label_pts = NTuple{2,Float64}[]
    avg_contrib = Float64[]
    for (j, cid) in enumerate(chain_ids)
        count = counts[j]
        push!(rects, (float(j) - 0.35, 0.0, float(j) + 0.35, float(count)))
        push!(labels, string(count))
        push!(label_pts, (float(j), float(count) + 0.05 * ymax))
        idx = findall(==(cid), fam.chain_id)
        push!(avg_contrib, isempty(idx) ? 0.0 : (sum(metrics.contributions[idx]) / length(idx)))
    end
    bins, colors, legend = _float_bucket_style(avg_contrib; title="avg w*d")
    layers = AbstractVisualizationLayer[]
    for b in sort!(unique(bins))
        idx = findall(==(b), bins)
        isempty(idx) || push!(layers, RectLayer(rects[idx], colors[b], :black, 0.72, 0.8))
    end
    isempty(labels) || push!(layers, _text_layer_from_labels(label_pts, labels; color=:black, textsize=8.0))
    return VisualizationSpec(:fibered_family_contributions;
                             title="Cached chain multiplicities",
                             subtitle="how many arrangement cells reuse each exact chain",
                             layers=layers,
                             axes=_default_axes_2d(xlabel="chain id", ylabel="cell count",
                                                   xlimits=(0.5, float(max(length(chain_ids), 1)) + 0.5),
                                                   ylimits=(0.0, 1.15 * ymax),
                                                   aspect=:auto,
                                                   xticks=(collect(1:length(chain_ids)), string.(chain_ids))),
                             metadata=(; object=:fibered_slice_family, unique_chains=length(chain_ids),
                                        figure_size=(760, 420), legend_position=:right),
                             legend=legend)
end

function _fibered_family_overlay_panel(fam::FiberedSliceFamily2D;
                                       values=nothing,
                                       highlight_index=nothing,
                                       title::AbstractString="Fibered slice family",
                                       subtitle::AbstractString="one representative slice per nonempty arrangement cell")
    arr = source_arrangement(fam)
    outline = _box_outline(working_box(arr))
    paths = Vector{Vector{NTuple{2,Float64}}}()
    mids = NTuple{2,Float64}[]
    kept = Int[]
    for k in 1:length(fam.cell_id)
        line = _line_through_box(slice_direction(fam, k), slice_offset(fam, k), working_box(arr))
        length(line) == 2 || continue
        push!(paths, [line[1], line[2]])
        push!(mids, _line_midpoint(line))
        push!(kept, k)
    end
    bbox = _bbox_from_points(vcat(outline, mids))
    layers = AbstractVisualizationLayer[
        PolylineLayer([outline], :black, 1.0, 1.5, true),
        PolylineLayer(paths, :gray70, 0.4, 1.0, false),
    ]
    legend = _default_legend(visible=false)
    if values === nothing
        if length(kept) <= 24
            labels = [string(round(slice_offset(fam, k); digits=3)) for k in kept]
            push!(layers, _text_layer_from_labels(mids, labels; color=:black, textsize=8.0))
        end
    else
        vals_kept = Float64[float(values[k]) for k in kept]
        bins, colors, legend = _float_bucket_style(vals_kept; title="w*d")
        for b in sort!(unique(bins))
            idx = findall(==(b), bins)
            isempty(idx) || push!(layers, PointLayer(mids[idx], colors[b], 0.95, 12.0))
        end
        if highlight_index !== nothing && highlight_index in kept
            local_idx = findfirst(==(highlight_index), kept)
            push!(layers, PolylineLayer([paths[local_idx]], :orange3, 1.0, 2.8, false))
        end
        top = isempty(vals_kept) ? Int[] : sortperm(vals_kept; rev=true)[1:min(5, length(vals_kept))]
        if !isempty(top)
            label_pts = [mids[i] for i in top]
            labels = ["s$(kept[i])=$(round(vals_kept[i]; digits=3))" for i in top]
            push!(layers, _text_layer_from_labels(_offset_label_positions(label_pts,
                                                                          _default_axes_2d(xlabel="x1", ylabel="x2",
                                                                                           xlimits=bbox === nothing ? nothing : bbox[1],
                                                                                           ylimits=bbox === nothing ? nothing : bbox[2],
                                                                                           aspect=:equal)),
                                                   labels; color=:black, textsize=8.0))
        end
    end
    return VisualizationSpec(:fibered_family;
                             title=title,
                             subtitle=subtitle,
                             layers=layers,
                             axes=_default_axes_2d(xlabel="x1", ylabel="x2",
                                                   xlimits=bbox === nothing ? nothing : bbox[1],
                                                   ylimits=bbox === nothing ? nothing : bbox[2],
                                                   aspect=:equal),
                             metadata=(; object=:fibered_slice_family, nslices=length(fam.cell_id),
                                        unique_chains=length(fam.unique_chain_ids), figure_size=(860, 620),
                                        legend_position=get(legend, :visible, false) ? :right : :none),
                             legend=legend,
                             interaction=_default_interaction(hover=true, labels=values !== nothing))
end

function _fibered_top_contributions_panel(fam::FiberedSliceFamily2D, metrics; ntop::Int=8)
    idxs = isempty(metrics.contributions) ? Int[] : sortperm(metrics.contributions; rev=true)[1:min(ntop, length(metrics.contributions))]
    vals = [metrics.contributions[k] for k in idxs]
    ymax = isempty(vals) ? 1.0 : max(1.0, maximum(vals))
    rects = NTuple{4,Float64}[]
    labels = String[]
    label_pts = NTuple{2,Float64}[]
    for (j, k) in enumerate(idxs)
        v = vals[j]
        push!(rects, (float(j) - 0.35, 0.0, float(j) + 0.35, v))
        push!(labels, string(round(v; digits=3)))
        push!(label_pts, (float(j), v + 0.05 * ymax))
    end
    bins, colors, legend = _float_bucket_style(vals; title="w*d")
    layers = AbstractVisualizationLayer[]
    for b in sort!(unique(bins))
        idx = findall(==(b), bins)
        isempty(idx) || push!(layers, RectLayer(rects[idx], colors[b], :black, 0.78, 0.8))
    end
    isempty(labels) || push!(layers, _text_layer_from_labels(label_pts, labels; color=:black, textsize=8.0))
    xticks = (collect(1:length(idxs)), ["s$(k)" for k in idxs])
    return VisualizationSpec(:fibered_distance_diagnostic;
                             title="Top exact contributions",
                             subtitle="largest weighted bottleneck terms in the exact family",
                             layers=layers,
                             axes=_default_axes_2d(xlabel="family slice", ylabel="w*d",
                                                   xlimits=(0.5, float(max(length(idxs), 1)) + 0.5),
                                                   ylimits=(0.0, 1.15 * ymax),
                                                   aspect=:auto, xticks=xticks),
                             metadata=(; object=:fibered_slice_family, ntop=length(idxs), figure_size=(780, 440), legend_position=:right),
                             legend=legend)
end

function _fibered_offset_interval_spec(arr::FiberedArrangement2D, dir)
    d = Float64[float(dir[1]), float(dir[2])]
    dir_idx = _fibered_dir_cell_index(arr, d)
    noff = arr.noff[dir_idx]
    rects = NTuple{4,Float64}[]
    label_pts = NTuple{2,Float64}[]
    labels = String[]
    chain_ids = Int[]
    for off_idx in 1:noff
        c0, c1, _ = _arr2d_cell_offset_interval(arr, dir_idx, off_idx)
        cid = _arr2d_compute_cell!(arr, dir_idx, off_idx)
        push!(rects, (float(c0), 0.68, float(c1), 1.32))
        push!(label_pts, (0.5 * (float(c0) + float(c1)), 1.0))
        push!(labels, cid > 0 ? string(cid) : "empty")
        push!(chain_ids, max(cid, 0))
    end
    uniq, cmap = _integer_value_palette(chain_ids)
    layers = AbstractVisualizationLayer[]
    for cid in uniq
        idx = findall(==(cid), chain_ids)
        color = cid == 0 ? :gray85 : cmap[cid]
        isempty(idx) || push!(layers, RectLayer(rects[idx], color, :black, 0.74, 0.8))
    end
    if noff <= 24
        push!(layers, _text_layer_from_labels(label_pts, labels; color=:black, textsize=8.0))
    end
    legend_entries = NamedTuple[(; label=(cid == 0 ? "empty" : string(cid)), color=(cid == 0 ? :gray85 : cmap[cid]), style=:patch) for cid in uniq]
    legend = length(uniq) <= 8 ? _default_legend(visible=true, title="chain", entries=legend_entries) : _default_legend(visible=false)
    return VisualizationSpec(:fibered_offset_intervals;
                             title="Fibered offset intervals",
                             subtitle="offset-cell intervals inside one direction cell",
                             layers=layers,
                             axes=_default_axes_2d(xlabel="offset", ylabel="selected direction cell",
                                                   xlimits=_finite_extrema(vcat([r[1] for r in rects], [r[3] for r in rects])),
                                                   ylimits=(0.5, 1.5), aspect=:auto,
                                                   yticks=([1.0], ["dir $(dir_idx)"])),
                             metadata=(; object=:fibered_arrangement, direction_cell=dir_idx,
                                        offset_cells=noff, figure_size=(860, 420),
                                        legend_position=get(legend, :visible, false) ? :right : :none),
                             legend=legend,
                             interaction=_default_interaction(hover=true, labels=noff <= 24))
end

function _fibered_slice_overlay_spec(result::FiberedSliceResult;
                                     arrangement::FiberedArrangement2D,
                                     dir,
                                     offset=nothing,
                                     basepoint=nothing,
                                     tie_break::Symbol=:up)
    arg = offset === nothing ? basepoint : offset
    report = fibered_query_summary(arrangement, dir, arg; tie_break=tie_break)
    report.valid || throw(ArgumentError("fibered_slice_overlay requires a valid arrangement query; got issues $(report.issues)."))
    chain = Int.(slice_chain(result))
    vals = Float64.(slice_values(result))
    length(vals) == length(chain) + 1 || throw(ArgumentError("fibered_slice_overlay requires aligned chain/value data."))
    outline = _box_outline(working_box(arrangement))
    pts = NTuple{2,Float64}[(float(p[1]), float(p[2])) for p in arrangement.points]
    segments, mids = _slice_segments_from_values(report.normalized_direction, report.offset, vals)
    bbox = _bbox_from_points(vcat(outline, pts, mids))
    axes = _default_axes_2d(xlabel="x1", ylabel="x2",
                            xlimits=bbox === nothing ? nothing : _pad_limits(bbox[1]),
                            ylimits=bbox === nothing ? nothing : _pad_limits(bbox[2]),
                            aspect=:equal)
    layers = AbstractVisualizationLayer[
        PolylineLayer([outline], :black, 1.0, 1.5, true),
        PointLayer(pts, :gray55, 0.55, 8.0),
    ]
    whole_line = _line_through_box(report.normalized_direction, report.offset, working_box(arrangement))
    isempty(whole_line) || push!(layers, PolylineLayer([collect(whole_line)], :gray65, 0.6, 1.2, false))
    colors = Symbol[_invariant_value_color(i) for i in eachindex(segments)]
    for color in unique(colors)
        idx = findall(==(color), colors)
        isempty(idx) || push!(layers, SegmentLayer(segments[idx], color, 0.95, 3.0))
    end
    if !isempty(chain)
        push!(layers, _text_layer_from_labels(_offset_label_positions(mids, axes), string.(chain); color=:black, textsize=8.5))
    end
    return VisualizationSpec(:fibered_slice_overlay;
                             title="Fibered slice overlay",
                             subtitle="the exact chain segments drawn back inside the 2D arrangement",
                             layers=layers,
                             axes=axes,
                             metadata=merge((; object=:fibered_slice, chain_length=length(chain), value_count=length(vals),
                                               figure_size=(860, 620), legend_position=:none), report),
                             interaction=_default_interaction(hover=true, labels=true))
end

function _visual_spec(result::FiberedSliceResult, kind::Symbol; arrangement=nothing, dir=nothing, offset=nothing, basepoint=nothing, tie_break::Symbol=:up, kwargs...)
    _ = kwargs
    if kind === :barcode
        return _barcode_spec(slice_barcode(result); title="Fibered slice barcode")
    elseif kind === :fibered_slice_overlay
        arrangement isa FiberedArrangement2D || throw(ArgumentError("fibered_slice_overlay requires keyword arrangement=<FiberedArrangement2D>."))
        dir === nothing && throw(ArgumentError("fibered_slice_overlay requires keyword dir."))
        return _fibered_slice_overlay_spec(result; arrangement=arrangement, dir=dir, offset=offset, basepoint=basepoint, tie_break=tie_break)
    elseif kind === :fibered_slice
        chain = Int.(slice_chain(result))
        vals = Float64.(slice_values(result))
        bc = slice_barcode(result)
        if isempty(chain) || length(vals) < 2
            return VisualizationSpec(:fibered_slice;
                                     title="Fibered slice",
                                     subtitle="empty exact slice result",
                                     layers=AbstractVisualizationLayer[
                                         TextLayer(["empty slice"], [(0.5, 0.5)], :black, 12.0),
                                     ],
                                     axes=_default_axes_2d(xlabel="t", ylabel="chain region",
                                                           xlimits=(0.0, 1.0), ylimits=(0.0, 1.0), aspect=:auto),
                                     metadata=(; object=:fibered_slice, chain_length=0, value_count=length(vals),
                                                barcode_intervals=_barcode_count(bc), figure_size=(860, 520),
                                                legend_position=:none))
        end
        length(vals) == length(chain) + 1 ||
            throw(ArgumentError("fibered_slice visualization requires length(slice_values(result)) == length(slice_chain(result)) + 1."))
        layers = AbstractVisualizationLayer[]
        rects = NTuple{4,Float64}[]
        fill_colors = Symbol[]
        label_pts = NTuple{2,Float64}[]
        labels = String[]
        for i in eachindex(chain)
            y = float(i)
            push!(rects, (vals[i], y - 0.34, vals[i + 1], y + 0.34))
            push!(fill_colors, _invariant_value_color(i))
            push!(label_pts, (0.5 * (vals[i] + vals[i + 1]), y))
            push!(labels, string(chain[i]))
        end
        for color in unique(fill_colors)
            idx = findall(==(color), fill_colors)
            isempty(idx) || push!(layers, RectLayer(rects[idx], color, :black, 0.5, 0.8))
        end
        break_segments = NTuple{4,Float64}[]
        ymin = 0.5
        ymax = float(length(chain)) + 0.5
        for t in vals
            push!(break_segments, (t, ymin, t, ymax))
        end
        push!(layers, SegmentLayer(break_segments, :gray45, 0.55, 1.0))
        push!(layers, _text_layer_from_labels(label_pts, labels; color=:black, textsize=9.0))
        ticks = (collect(1:length(chain)), string.(chain))
        return VisualizationSpec(:fibered_slice;
                                 title="Fibered slice",
                                 subtitle="exact chain intervals along the selected slice",
                                 layers=layers,
                                 axes=_default_axes_2d(xlabel="t", ylabel="chain region",
                                                       xlimits=(minimum(vals), maximum(vals)),
                                                       ylimits=(0.5, float(length(chain)) + 0.5),
                                                       aspect=:auto,
                                                       yticks=ticks),
                                 metadata=(; object=:fibered_slice, chain_length=length(chain), value_count=length(vals),
                                            barcode_intervals=_barcode_count(bc), total_multiplicity=_barcode_count(bc),
                                            figure_size=(860, 520), legend_position=:none),
                                 interaction=_default_interaction(hover=true, labels=true))
    end
    throw(ArgumentError("FiberedSliceResult supports kind=:fibered_slice, :fibered_slice_overlay, or :barcode only."))
end

function _box_outline(box)
    lo, hi = box
    return [(float(lo[1]), float(lo[2])), (float(hi[1]), float(lo[2])), (float(hi[1]), float(hi[2])), (float(lo[1]), float(hi[2])), (float(lo[1]), float(lo[2]))]
end

@inline function _pad_limits(lims::Tuple{<:Real,<:Real}; frac::Float64=0.05, minpad::Float64=0.05)
    lo = float(lims[1])
    hi = float(lims[2])
    pad = max(minpad, frac * abs(hi - lo))
    return (lo - pad, hi + pad)
end

function _line_through_box(dir::AbstractVector{<:Real}, off::Real, box)
    lo, hi = box
    xlo, ylo = float(lo[1]), float(lo[2])
    xhi, yhi = float(hi[1]), float(hi[2])
    d1, d2 = float(dir[1]), float(dir[2])
    n1, n2 = -d2, d1
    pts = NTuple{2,Float64}[]
    if abs(n2) > 1e-12
        y = (float(off) - n1 * xlo) / n2
        ylo - 1e-9 <= y <= yhi + 1e-9 && push!(pts, (xlo, y))
        y = (float(off) - n1 * xhi) / n2
        ylo - 1e-9 <= y <= yhi + 1e-9 && push!(pts, (xhi, y))
    end
    if abs(n1) > 1e-12
        x = (float(off) - n2 * ylo) / n1
        xlo - 1e-9 <= x <= xhi + 1e-9 && push!(pts, (x, ylo))
        x = (float(off) - n2 * yhi) / n1
        xlo - 1e-9 <= x <= xhi + 1e-9 && push!(pts, (x, yhi))
    end
    uniq = unique(pts)
    length(uniq) >= 2 || return NTuple{2,Float64}[]
    return uniq[1:2]
end

@inline function _line_midpoint(seg::AbstractVector{<:NTuple{2,<:Real}})
    isempty(seg) && return (0.0, 0.0)
    p = seg[1]
    q = seg[end]
    return (0.5 * (float(p[1]) + float(q[1])), 0.5 * (float(p[2]) + float(q[2])))
end

function _fibered_cell_offsets(arr::FiberedArrangement2D, cell_id::Tuple{Int,Int})
    c0, c1, cmid = _arr2d_cell_offset_interval(arr, cell_id[1], cell_id[2])
    return (float(c0), float(c1), float(cmid))
end

function _fibered_strip_paths(arr::FiberedArrangement2D, dir::AbstractVector{<:Real}, c0::Real, c1::Real; nsamples::Int=7)
    width = float(c1) - float(c0)
    width >= 0 || return Vector{Vector{NTuple{2,Float64}}}()
    if width <= 1e-10
        line = _line_through_box(dir, 0.5 * (float(c0) + float(c1)), working_box(arr))
        return isempty(line) ? Vector{Vector{NTuple{2,Float64}}}() : [collect(line)]
    end
    lo = float(c0) + 0.1 * width
    hi = float(c1) - 0.1 * width
    offs = lo <= hi ? collect(range(lo, hi; length=max(nsamples, 2))) : [0.5 * (float(c0) + float(c1))]
    paths = Vector{Vector{NTuple{2,Float64}}}()
    for off in offs
        seg = _line_through_box(dir, off, working_box(arr))
        isempty(seg) || push!(paths, collect(seg))
    end
    return paths
end

function _fibered_cell_visual(arr::FiberedArrangement2D, report;
                              primary_cell::Union{Nothing,Tuple{Int,Int}}=report.cell_id,
                              primary_color::Symbol=:teal,
                              primary_tag::String="selected",
                              secondary_cell::Union{Nothing,Tuple{Int,Int}}=nothing,
                              secondary_color::Symbol=:crimson,
                              secondary_tag::String="alternate",
                              query_line_color::Symbol=:orange3,
                              query_label::Union{Nothing,String}=nothing,
                              show_boundary::Bool=false)
    report.valid || throw(ArgumentError("fibered exact-cell visual requires a valid query; got issues $(report.issues)."))
    d = report.normalized_direction
    box = working_box(arr)
    outline = _box_outline(box)
    pts = NTuple{2,Float64}[(float(p[1]), float(p[2])) for p in arr.points]
    bbox = _bbox_from_points(vcat(outline, pts))
    axes = _default_axes_2d(xlabel="x1", ylabel="x2",
                            xlimits=bbox === nothing ? nothing : _pad_limits(bbox[1]),
                            ylimits=bbox === nothing ? nothing : _pad_limits(bbox[2]),
                            aspect=:equal)
    layers = AbstractVisualizationLayer[
        PolylineLayer([outline], :black, 1.0, 1.5, true),
        PointLayer(pts, :gray55, 0.55, 8.0),
    ]
    labels = String[]
    label_pts = NTuple{2,Float64}[]

    function add_cell!(cell_id, color::Symbol, tag::String; alpha::Float64=0.42)
        cell_id === nothing && return nothing
        c0, c1, cmid = _fibered_cell_offsets(arr, cell_id)
        hatch = _fibered_strip_paths(arr, d, c0, c1)
        isempty(hatch) || push!(layers, PolylineLayer(hatch, color, alpha, 1.1, false))
        boundaries = Vector{Vector{NTuple{2,Float64}}}()
        for off in (c0, c1)
            seg = _line_through_box(d, off, box)
            isempty(seg) || push!(boundaries, collect(seg))
        end
        isempty(boundaries) || push!(layers, PolylineLayer(boundaries, color, 0.9, 1.8, false))
        mid = _line_through_box(d, cmid, box)
        if !isempty(mid)
            push!(label_pts, _line_midpoint(mid))
            push!(labels, string(tag, " cell ", cell_id[1], "/", cell_id[2]))
        end
        return nothing
    end

    add_cell!(primary_cell, primary_color, primary_tag)
    secondary_cell == primary_cell || add_cell!(secondary_cell, secondary_color, secondary_tag)

    query_seg = _line_through_box(d, report.offset, box)
    isempty(query_seg) || push!(layers, PolylineLayer([collect(query_seg)], query_line_color, 1.0, 2.6, false))
    report.basepoint === nothing || push!(layers, PointLayer(_as_points2(report.basepoint), query_line_color, 1.0, 13.0))

    if show_boundary && !isempty(query_seg)
        push!(label_pts, _line_midpoint(query_seg))
        push!(labels, "boundary")
    elseif query_label !== nothing && !isempty(query_seg)
        push!(label_pts, _line_midpoint(query_seg))
        push!(labels, query_label)
    end

    isempty(labels) || push!(layers, _text_layer_from_labels(_offset_label_positions(label_pts, axes), labels; color=:black, textsize=9.0))

    return layers, axes
end

function _visual_spec(arr::FiberedArrangement2D, kind::Symbol; dir=nothing, offset=nothing, basepoint=nothing, tie_break::Symbol=:up, projected=nothing, kwargs...)
    _ = kwargs
    box = working_box(arr)
    outline = _box_outline(box)
    if kind === :fibered_arrangement
        pts = NTuple{2,Float64}[(float(p[1]), float(p[2])) for p in arr.points]
        bbox = _bbox_from_points(vcat(outline, pts))
        return VisualizationSpec(:fibered_arrangement;
                                 title="Fibered arrangement",
                                 subtitle="working box with critical points",
                                 layers=AbstractVisualizationLayer[
                                     PolylineLayer([outline], :black, 1.0, 1.5, true),
                                     PointLayer(pts, :purple3, 0.9, 10.0),
                                 ],
                                 axes=_default_axes_2d(xlabel="x1", ylabel="x2",
                                                       xlimits=bbox === nothing ? nothing : bbox[1],
                                                       ylimits=bbox === nothing ? nothing : bbox[2],
                                                       aspect=:equal),
                                 metadata=(; object=:fibered_arrangement, direction_cells=length(direction_representatives(arr)),
                                            total_cells=ncells(arr), computed_cells=computed_cell_count(arr),
                                            backend=backend(arr), figure_size=(780, 620), legend_position=:none))
    elseif kind === :fibered_query
        report = fibered_query_summary(arr, dir, offset === nothing ? basepoint : offset; tie_break=tie_break)
        d = report.normalized_direction
        off = report.offset
        line = _line_through_box(d, off, box)
        pts = NTuple{2,Float64}[(float(p[1]), float(p[2])) for p in arr.points]
        layers = AbstractVisualizationLayer[
            PolylineLayer([outline], :black, 1.0, 1.5, true),
            PointLayer(pts, :gray55, 0.65, 8.0),
        ]
        isempty(line) || push!(layers, PolylineLayer([collect(line)], :orange3, 1.0, 2.5, false))
        if basepoint !== nothing
            bp = _as_points2(basepoint)
            append!(layers, AbstractVisualizationLayer[PointLayer(bp, :orange3, 1.0, 14.0)])
        end
        bbox = _bbox_from_points(vcat(outline, pts))
        return VisualizationSpec(:fibered_query;
                                 title="Fibered query",
                                 subtitle="selected exact slice inside the arrangement",
                                 layers=layers,
                                 axes=_default_axes_2d(xlabel="x1", ylabel="x2",
                                                       xlimits=bbox === nothing ? nothing : _pad_limits(bbox[1]),
                                                       ylimits=bbox === nothing ? nothing : _pad_limits(bbox[2]),
                                                       aspect=:equal),
                                 metadata=merge((; figure_size=(780, 620), legend_position=:none), report))
    elseif kind === :fibered_cell_highlight
        report = fibered_query_summary(arr, dir, offset === nothing ? basepoint : offset; tie_break=tie_break)
        layers, axes = _fibered_cell_visual(arr, report;
                                            primary_cell=report.cell_id,
                                            primary_color=:teal,
                                            primary_tag="selected",
                                            query_label="query")
        return VisualizationSpec(:fibered_cell_highlight;
                                 title="Fibered cell highlight",
                                 subtitle="selected arrangement cell shown as a strip of exact slices",
                                 layers=layers,
                                 axes=axes,
                                 metadata=merge((; figure_size=(860, 620), legend_position=:none), report))
    elseif kind === :fibered_tie_break
        report = fibered_query_summary(arr, dir, offset === nothing ? basepoint : offset; tie_break=tie_break)
        report.tie_break_relevant || throw(ArgumentError("fibered_tie_break requires a boundary query where tie_break changes the selected cell."))
        layers, axes = _fibered_cell_visual(arr, report;
                                            primary_cell=report.cell_up,
                                            primary_color=:royalblue3,
                                            primary_tag="up",
                                            secondary_cell=report.cell_down,
                                            secondary_color=:crimson,
                                            secondary_tag="down",
                                            query_line_color=:black,
                                            show_boundary=true)
        return VisualizationSpec(:fibered_tie_break;
                                 title="Fibered tie-break diagnostic",
                                 subtitle="adjacent arrangement cells on either side of a boundary query",
                                 layers=layers,
                                 axes=axes,
                                 metadata=merge((; figure_size=(860, 620), legend_position=:none), report))
    elseif kind === :fibered_offset_intervals
        dir === nothing && throw(ArgumentError("fibered_offset_intervals requires keyword dir."))
        return _fibered_offset_interval_spec(arr, dir)
    elseif kind === :fibered_projected_comparison
        projected isa ProjectedArrangement || throw(ArgumentError("fibered_projected_comparison requires keyword projected=<ProjectedArrangement>."))
        fam = fibered_slice_family_2d(arr)
        exact_panel = _fibered_family_overlay_panel(fam; title="Exact fibered family",
                                                    subtitle="one representative exact slice per nonempty arrangement cell")
        proj_panel = _visual_spec(projected, :projected_arrangement)
        return VisualizationSpec(:fibered_projected_comparison;
                                 title="Fibered vs projected comparison",
                                 subtitle="exact arrangement slices beside the projected 1D direction family",
                                 panels=[exact_panel, proj_panel],
                                 metadata=(; object=:fibered_arrangement, nslices=length(fam.cell_id),
                                            nprojections=length(projections(projected)), panel_columns=2,
                                            figure_size=(1180, 620)))
    end
    throw(ArgumentError("Unsupported FiberedArrangement2D visualization kind $(kind)."))
end

function _visual_spec(cache::FiberedBarcodeCache2D, kind::Symbol; caches=nothing, dir=nothing, offset=nothing, basepoint=nothing, projected=nothing, kwargs...)
    if kind in (:fibered_family, :fibered_chain_cells, :fibered_family_contributions, :fibered_distance_diagnostic)
        fam = fibered_slice_family_2d(shared_arrangement(cache))
        return _visual_spec(fam, kind; caches=(caches === nothing ? nothing : caches), kwargs...)
    elseif kind === :fibered_query_barcode
        arg = offset === nothing ? basepoint : offset
        report = fibered_query_summary(cache, dir, arg; tie_break=get(kwargs, :tie_break, :up))
        slice = fibered_slice(cache, dir, arg; tie_break=get(kwargs, :tie_break, :up))
        overlay = _visual_spec(slice, :fibered_slice_overlay;
                               arrangement=shared_arrangement(cache), dir=dir, offset=offset,
                               basepoint=basepoint, tie_break=get(kwargs, :tie_break, :up))
        bar = _visual_spec(slice, :barcode)
        return VisualizationSpec(:fibered_query_barcode;
                                 title="Fibered query + barcode",
                                 subtitle="exact slice geometry paired with the resulting 1-parameter barcode",
                                 panels=[overlay, bar],
                                 metadata=merge((; figure_size=(1180, 560), panel_columns=2), report))
    elseif kind === :fibered_projected_comparison
        return _visual_spec(shared_arrangement(cache), kind; projected=projected)
    end
    return _visual_spec(shared_arrangement(cache), kind; dir=dir, offset=offset, basepoint=basepoint, projected=projected, kwargs...)
end

function _visual_spec(fam::FiberedSliceFamily2D, kind::Symbol; caches=nothing, kwargs...)
    _ = kwargs
    if kind === :fibered_family
        return _fibered_family_overlay_panel(fam)
    elseif kind === :fibered_chain_cells
        return _fibered_chain_cells_spec(fam)
    elseif kind === :fibered_family_contributions
        caches === nothing && throw(ArgumentError("fibered_family_contributions requires keyword caches=(cacheA, cacheB)."))
        metrics = _fibered_family_metrics(fam, caches)
        cell_panel = _fibered_contribution_cell_spec(fam, metrics;
                                                     title="Exact family contributions",
                                                     subtitle=stores_values(fam) ?
                                                         "weighted exact contributions by arrangement cell; boundary values cached" :
                                                         "weighted exact contributions by arrangement cell; boundary values recomputed")
        mult_panel = _fibered_chain_multiplicity_panel(fam, metrics)
        return VisualizationSpec(:fibered_family_contributions;
                                 title="Fibered family contributions",
                                 subtitle="exact per-cell contributions and chain reuse across the cached family",
                                 panels=[cell_panel, mult_panel],
                                 metadata=(; object=:fibered_slice_family, nslices=length(fam.cell_id),
                                            unique_chains=length(fam.unique_chain_ids), matching_distance=metrics.max_contribution,
                                            panel_columns=2, figure_size=(1260, 620)))
    elseif kind === :fibered_distance_diagnostic
        caches === nothing && throw(ArgumentError("fibered_distance_diagnostic requires keyword caches=(cacheA, cacheB)."))
        metrics = _fibered_family_metrics(fam, caches)
        cell_panel = _fibered_contribution_cell_spec(fam, metrics;
                                                     title="Exact contribution map",
                                                     subtitle="arrangement cells colored by weighted bottleneck contribution",
                                                     highlight_argmax=true)
        overlay_panel = _fibered_family_overlay_panel(fam;
                                                      values=metrics.contributions,
                                                      highlight_index=metrics.argmax_index,
                                                      title="Extremizing exact slice",
                                                      subtitle=metrics.argmax_index === nothing ?
                                                          "no nonempty family slices" :
                                                          "argmax slice s$(metrics.argmax_index) at dir cell $(fam.dir_idx[metrics.argmax_index]), offset cell $(fam.off_idx[metrics.argmax_index])")
        top_panel = _fibered_top_contributions_panel(fam, metrics)
        return VisualizationSpec(:fibered_distance_diagnostic;
                                 title="Exact fibered distance diagnostic",
                                 subtitle="dominant exact-family cells and the slice attaining the matching distance",
                                 panels=[cell_panel, overlay_panel, top_panel],
                                 metadata=(; object=:fibered_slice_family, nslices=length(fam.cell_id),
                                            unique_chains=length(fam.unique_chain_ids), matching_distance=metrics.max_contribution,
                                            argmax_index=metrics.argmax_index, panel_columns=2,
                                            figure_size=(1260, 900)))
    end
    throw(ArgumentError("Unsupported FiberedSliceFamily2D visualization kind $(kind)."))
end

function _visual_spec(arr::ProjectedArrangement, kind::Symbol; kwargs...)
    kind === :projected_arrangement || throw(ArgumentError("Unsupported ProjectedArrangement visualization kind $(kind)."))
    _ = kwargs
    dirs = NTuple{2,Float64}[]
    labels = String[]
    for (j, proj) in enumerate(projections(arr))
        if length(proj.dir) == 2
            push!(dirs, (float(proj.dir[1]), float(proj.dir[2])))
            push!(labels, string(j))
        end
    end
    bbox = _bbox_from_points(dirs)
    return VisualizationSpec(:projected_arrangement;
                             title="Projected arrangement",
                             subtitle="projection directions used for 1D pushforwards",
                             layers=AbstractVisualizationLayer[
                                 PointLayer(dirs, :darkgreen, 0.95, 12.0),
                                 _text_layer_from_labels(dirs, labels; color=:black, textsize=9.0),
                             ],
                             axes=_default_axes_2d(xlabel="dir_x", ylabel="dir_y",
                                                   xlimits=bbox === nothing ? nothing : bbox[1],
                                                   ylimits=bbox === nothing ? nothing : bbox[2],
                                                   aspect=:equal),
                             metadata=(; object=:projected_arrangement, nprojections=length(dirs)))
end

function _visual_spec(result::ProjectedBarcodesResult, kind::Symbol; kwargs...)
    kind === :barcode_bank || throw(ArgumentError("ProjectedBarcodesResult supports kind=:barcode_bank only."))
    _ = kwargs
    vals = reshape(Float64.([_barcode_count(bar) for bar in result.barcodes]), 1, :)
    x = Float64[1:size(vals, 2);]
    return VisualizationSpec(:barcode_bank;
                             title="Projected barcode bank",
                             subtitle="interval counts by projection",
                             layers=AbstractVisualizationLayer[
                                 HeatmapLayer(x, [1.0], vals, :viridis, 1.0, "interval count"),
                             ],
                             axes=_default_axes_2d(xlabel="projection index", ylabel="row", xlimits=(1.0, float(length(result.barcodes))), ylimits=(1.0, 1.0)),
                             metadata=(; nprojections=length(result.barcodes), projection_indices=Tuple(projection_indices(result))))
end

function _visual_spec(result::ProjectedDistancesResult, kind::Symbol; kwargs...)
    kind === :projected_distances || throw(ArgumentError("ProjectedDistancesResult supports kind=:projected_distances only."))
    _ = kwargs
    dists = Float64.(projected_distances(result))
    pts = [(float(i), dists[i]) for i in eachindex(dists)]
    return VisualizationSpec(:projected_distances;
                             title="Projected distances",
                             subtitle="per-projection distance values before aggregation",
                             layers=AbstractVisualizationLayer[
                                 PolylineLayer([pts], :navy, 1.0, 2.0, false),
                                 PointLayer(pts, :navy, 1.0, 9.0),
                             ],
                             axes=_default_axes_2d(xlabel="projection index", ylabel="distance",
                                                   xlimits=(1.0, float(length(dists))),
                                                   ylimits=_finite_extrema(dists)),
                             metadata=(; nprojections=length(dists), projection_indices=Tuple(projection_indices(result))))
end

available_visuals(::RectSignedBarcode{2}) = (:rectangles, :density_image)
available_visuals(::PointSignedMeasure{2}) = (:signed_atoms,)
function available_visuals(out::SignedMeasureDecomposition)
    kinds = Symbol[]
    has_rectangles(out) && append!(kinds, [:rectangles, :density_image])
    has_euler_signed_measure(out) && push!(kinds, :signed_atoms)
    has_mpp_image(out) && (:density_image in kinds || push!(kinds, :density_image))
    return Tuple(kinds)
end

function _rect_geometry(sb::RectSignedBarcode{2})
    rects = NTuple{4,Float64}[]
    labels = String[]
    colors = Symbol[]
    for (rect, wt) in zip(rectangles(sb), weights(sb))
        push!(rects, (float(rect.lo[1]), float(rect.lo[2]), float(rect.hi[1]), float(rect.hi[2])))
        push!(labels, string(wt))
        push!(colors, _color_from_sign(wt))
    end
    return rects, labels, colors
end

function _rect_density(sb::RectSignedBarcode{2})
    xs = Float64[float(x) for x in sb.axes[1]]
    ys = Float64[float(y) for y in sb.axes[2]]
    vals = zeros(Float64, length(ys), length(xs))
    for (rect, wt) in zip(rectangles(sb), weights(sb))
        for j in rect.lo[2]:rect.hi[2], i in rect.lo[1]:rect.hi[1]
            vals[j, i] += float(wt)
        end
    end
    return xs, ys, vals
end

function _visual_spec(sb::RectSignedBarcode{2}, kind::Symbol; kwargs...)
    _ = kwargs
    if kind === :rectangles
        rects, labels, colors = _rect_geometry(sb)
        centers = _rect_text_centers(rects)
        layers = AbstractVisualizationLayer[]
        for color in (:crimson, :royalblue, :gray50)
            idx = findall(==(color), colors)
            isempty(idx) || push!(layers, RectLayer(rects[idx], color, color, 0.28, 1.0))
        end
        push!(layers, _text_layer_from_labels(centers, labels; color=:black, textsize=9.0))
        bbox = _bbox_from_points(centers)
        return VisualizationSpec(:rectangles;
                                 title="Rectangle signed barcode",
                                 subtitle="signed rectangles colored by coefficient sign",
                                 layers=layers,
                                 axes=_default_axes_2d(xlabel="x1", ylabel="x2",
                                                       xlimits=bbox === nothing ? nothing : bbox[1],
                                                       ylimits=bbox === nothing ? nothing : bbox[2],
                                                       aspect=:equal),
                                 metadata=(; nrectangles=length(rects), total_mass=total_mass(sb), total_variation=total_variation(sb)),
                                 legend=_default_legend(visible=true, entries=(; positive=:crimson, negative=:royalblue)))
    elseif kind === :density_image
        xs, ys, vals = _rect_density(sb)
        return VisualizationSpec(:density_image;
                                 title="Rectangle signed density",
                                 subtitle="grid-sampled accumulation of signed rectangle weights",
                                 layers=AbstractVisualizationLayer[
                                     HeatmapLayer(xs, ys, vals, :balance, 1.0, "signed mass"),
                                 ],
                                 axes=_default_axes_2d(xlabel="x1", ylabel="x2",
                                                       xlimits=(minimum(xs), maximum(xs)),
                                                       ylimits=(minimum(ys), maximum(ys))),
                                 metadata=(; nrectangles=length(rectangles(sb)), total_mass=total_mass(sb)))
    end
    throw(ArgumentError("Unsupported RectSignedBarcode visualization kind $(kind)."))
end

function _visual_spec(pm::PointSignedMeasure{2}, kind::Symbol; kwargs...)
    kind === :signed_atoms || throw(ArgumentError("PointSignedMeasure supports kind=:signed_atoms only."))
    _ = kwargs
    pts = NTuple{2,Float64}[(float(p[1]), float(p[2])) for p in points(pm)]
    pos = NTuple{2,Float64}[]
    neg = NTuple{2,Float64}[]
    labels = String[]
    label_pts = NTuple{2,Float64}[]
    for (p, wt) in zip(pts, weights(pm))
        push!(wt > 0 ? pos : neg, p)
        push!(labels, string(wt))
        push!(label_pts, p)
    end
    bbox = _bbox_from_points(pts)
    return VisualizationSpec(:signed_atoms;
                             title="Signed atoms",
                             subtitle="point-measure support with signed coefficients",
                             layers=AbstractVisualizationLayer[
                                 PointLayer(pos, :crimson, 0.95, 12.0),
                                 PointLayer(neg, :royalblue, 0.95, 12.0),
                                 _text_layer_from_labels(label_pts, labels; color=:black, textsize=9.0),
                             ],
                             axes=_default_axes_2d(xlabel="x1", ylabel="x2",
                                                   xlimits=bbox === nothing ? nothing : bbox[1],
                                                   ylimits=bbox === nothing ? nothing : bbox[2],
                                                   aspect=:equal),
                             metadata=(; nterms=length(pts), total_mass=total_mass(pm), total_variation=total_variation(pm)),
                             legend=_default_legend(visible=true, entries=(; positive=:crimson, negative=:royalblue)))
end

function _visual_spec(out::SignedMeasureDecomposition, kind::Symbol; kwargs...)
    if kind === :rectangles
        has_rectangles(out) || throw(ArgumentError("SignedMeasureDecomposition has no rectangles component."))
        return _visual_spec(rectangles(out), :rectangles; kwargs...)
    elseif kind === :signed_atoms
        has_euler_signed_measure(out) || throw(ArgumentError("SignedMeasureDecomposition has no euler_signed_measure component."))
        return _visual_spec(euler_signed_measure(out), :signed_atoms; kwargs...)
    elseif kind === :density_image
        if has_mpp_image(out)
            return _visual_spec(mpp_image(out), :mpp_image; kwargs...)
        elseif has_rectangles(out)
            return _visual_spec(rectangles(out), :density_image; kwargs...)
        elseif has_euler_signed_measure(out)
            return _visual_spec(euler_signed_measure(out), :signed_atoms; kwargs...)
        end
    end
    throw(ArgumentError("SignedMeasureDecomposition does not support kind=$(kind) with its present components $(component_names(out))."))
end

available_visuals(::MPPLineSpec) = (:mpp_line_spec,)
available_visuals(::MPPDecomposition) = (:mpp_decomposition,)
available_visuals(::MPPImage) = (:mpp_image,)
available_visuals(::MPLandscape) = (:mp_landscape, :landscape_slices)

const _MPP_SUMMAND_COLORS = (
    :darkorange2,
    :royalblue3,
    :seagreen3,
    :orchid3,
    :goldenrod2,
    :tomato2,
    :slateblue3,
    :cadetblue3,
)

@inline _mpp_summand_color(i::Int) = _MPP_SUMMAND_COLORS[1 + mod(i - 1, length(_MPP_SUMMAND_COLORS))]

function _mpp_weight_style(weights::AbstractVector{<:Real}, i::Int)
    isempty(weights) && return (0.7, 2.0)
    vals = Float64[float(w) for w in weights]
    w = vals[i]
    wlo = minimum(vals)
    whi = maximum(vals)
    t = whi > wlo ? (w - wlo) / (whi - wlo) : 0.5
    alpha = 0.38 + 0.47 * t
    linewidth = 1.4 + 1.8 * t
    return (alpha, linewidth)
end

function _visual_spec(line::MPPLineSpec, kind::Symbol; box=nothing, kwargs...)
    kind === :mpp_line_spec || throw(ArgumentError("MPPLineSpec supports kind=:mpp_line_spec only."))
    _ = kwargs
    b = box === nothing ? ([minimum((line_basepoint(line)[1], 0.0)) - 1, minimum((line_basepoint(line)[2], 0.0)) - 1],
                           [maximum((line_basepoint(line)[1], 1.0)) + 1, maximum((line_basepoint(line)[2], 1.0)) + 1]) : box
    line_pts = _line_through_box(line_direction(line), line_offset(line), b)
    base_pt = _as_points2(line_basepoint(line))
    bbox = _bbox_from_points(vcat(line_pts, base_pt))
    return VisualizationSpec(:mpp_line_spec;
                             title="MPPI line specification",
                             subtitle="one weighted slice line used in the decomposition",
                             layers=AbstractVisualizationLayer[
                                 PolylineLayer([_box_outline(b)], :black, 1.0, 1.0, true),
                                 PolylineLayer([collect(line_pts)], :darkorange2, 1.0, 2.0, false),
                                 PointLayer(base_pt, :darkorange2, 1.0, 12.0),
                             ],
                             axes=_default_axes_2d(xlabel="x1", ylabel="x2",
                                                   xlimits=bbox === nothing ? nothing : bbox[1],
                                                   ylimits=bbox === nothing ? nothing : bbox[2],
                                                   aspect=:equal),
                             metadata=(; offset=line_offset(line), omega=line_omega(line)))
end

function _visual_spec(decomp::MPPDecomposition, kind::Symbol; layout::Symbol=:overlay, kwargs...)
    kind === :mpp_decomposition || throw(ArgumentError("MPPDecomposition supports kind=:mpp_decomposition only."))
    _ = kwargs
    layout in (:overlay, :summands) || throw(ArgumentError("mpp_decomposition layout must be :overlay or :summands."))
    box = bounding_box(decomp)
    outline = _box_outline(box)
    axes = _default_axes_2d(xlabel="x1", ylabel="x2",
                            xlimits=(float(box[1][1]), float(box[2][1])),
                            ylimits=(float(box[1][2]), float(box[2][2])),
                            aspect=:equal)
    weights = summand_weights(decomp)
    if layout === :overlay
        layers = AbstractVisualizationLayer[
            PolylineLayer([outline], :black, 1.0, 1.0, true),
        ]
        for k in 1:nsummands(decomp)
            paths_k = Vector{Vector{NTuple{2,Float64}}}()
            for (p, q, _) in summand_segments(decomp, k)
                push!(paths_k, [(float(p[1]), float(p[2])), (float(q[1]), float(q[2]))])
            end
            isempty(paths_k) && continue
            alpha_k, linewidth_k = _mpp_weight_style(weights, k)
            push!(layers, PolylineLayer(paths_k, _mpp_summand_color(k), alpha_k, linewidth_k, false))
        end
        return VisualizationSpec(:mpp_decomposition;
                                 title="MPP decomposition",
                                 subtitle="summand segments in the ambient box",
                                 layers=layers,
                                 axes=axes,
                                 metadata=(; nsummands=nsummands(decomp), nlines=nlines(decomp), weight_sum=sum(weights),
                                            layout=:overlay, figure_size=(920, 620), legend_position=:right),
                                 legend=_default_legend(visible=true,
                                                        entries=(; (Symbol("S" * string(i)) => _mpp_summand_color(i) for i in 1:nsummands(decomp))...)))
    end

    panels = VisualizationSpec[]
    for k in 1:nsummands(decomp)
        paths_k = Vector{Vector{NTuple{2,Float64}}}()
        for (p, q, _) in summand_segments(decomp, k)
            push!(paths_k, [(float(p[1]), float(p[2])), (float(q[1]), float(q[2]))])
        end
        alpha_k, linewidth_k = _mpp_weight_style(weights, k)
        push!(panels, VisualizationSpec(:mpp_decomposition;
                                        title="Summand $(k)",
                                        subtitle="weight=$(round(float(weights[k]); digits=3))",
                                        layers=AbstractVisualizationLayer[
                                            PolylineLayer([outline], :black, 1.0, 1.0, true),
                                            PolylineLayer(paths_k, _mpp_summand_color(k), alpha_k, linewidth_k, false),
                                        ],
                                        axes=axes,
                                        metadata=(; summand=k, weight=float(weights[k]), nsegments=length(paths_k))))
    end
    return VisualizationSpec(:mpp_decomposition;
                             title="MPP decomposition",
                             subtitle="one panel per summand",
                             panels=panels,
                             axes=axes,
                             metadata=(; nsummands=nsummands(decomp), nlines=nlines(decomp), weight_sum=sum(weights),
                                        layout=:summands, panel_columns=min(3, max(1, nsummands(decomp))),
                                        figure_size=(1120, 560)))
end

function _visual_spec(img::MPPImage, kind::Symbol; kwargs...)
    kind === :mpp_image || throw(ArgumentError("MPPImage supports kind=:mpp_image only."))
    _ = kwargs
    xg = Float64.(image_xgrid(img))
    yg = Float64.(image_ygrid(img))
    return VisualizationSpec(:mpp_image;
                             title="Multiparameter persistence image",
                             subtitle="Gaussian-smoothed MPPI image",
                             layers=AbstractVisualizationLayer[
                                 HeatmapLayer(xg, yg, Float64.(image_values(img)), :magma, 1.0, "intensity"),
                             ],
                             axes=_default_axes_2d(xlabel="x1", ylabel="x2",
                                                   xlimits=(minimum(xg), maximum(xg)),
                                                   ylimits=(minimum(yg), maximum(yg))),
                             metadata=(; image_shape=size(image_values(img)), sigma=img.sigma, nsummands=nsummands(decomposition(img))))
end

function _landscape_aggregate(L::MPLandscape)
    vals = landscape_values(L)
    W = Float64.(slice_weights(L))
    kmax = size(vals, 3)
    nt = size(vals, 4)
    out = zeros(Float64, kmax, nt)
    denom = sum(W)
    denom == 0 && (denom = 1.0)
    for idir in axes(vals, 1), ioff in axes(vals, 2), k in axes(vals, 3), t in axes(vals, 4)
        out[k, t] += W[idir, ioff] * vals[idir, ioff, k, t]
    end
    out ./= denom
    return out
end

@inline function _landscape_color(k::Integer)
    palette = (:royalblue3, :darkorange2, :seagreen4, :mediumorchid3, :firebrick3, :goldenrod3)
    return palette[mod1(k, length(palette))]
end

@inline function _landscape_ylim(vals::AbstractArray{<:Real})
    ymax = maximum(float.(vals))
    ymax <= 0 && return (0.0, 1.0)
    return (0.0, 1.08 * ymax)
end

function _visual_spec(L::MPLandscape, kind::Symbol; idir=nothing, ioff=nothing, layer::Int=1, kwargs...)
    _ = kwargs
    tg = Float64.(landscape_grid(L))
    if kind === :mp_landscape
        vals = _landscape_aggregate(L)
        nlayers = size(vals, 1)
        if nlayers <= 5
            layers = AbstractVisualizationLayer[]
            legend_entries = Pair{Symbol,NamedTuple}[]
            for k in 1:nlayers
                path = NTuple{2,Float64}[(tg[j], vals[k, j]) for j in eachindex(tg)]
                color = _landscape_color(k)
                push!(layers, PolylineLayer([path], color, 1.0, 2.5, false))
                push!(legend_entries, Symbol("layer_" * string(k)) => (; color, style=:line))
            end
            return VisualizationSpec(:mp_landscape;
                                     title="Multiparameter landscape",
                                     subtitle="slice-weighted mean landscape curves",
                                     layers=layers,
                                     axes=_default_axes_2d(xlabel="t", ylabel="landscape value",
                                                           xlimits=(minimum(tg), maximum(tg)),
                                                           ylimits=_landscape_ylim(vals)),
                                     metadata=(; ndirections=ndirections(L), noffsets=noffsets(L),
                                                kmax=landscape_layers(L), render_mode=:curves,
                                                figure_size=(860, 520), legend_position=:right),
                                     legend=_default_legend(visible=(nlayers > 1),
                                                            title="layers",
                                                            entries=(; legend_entries...)))
        end
        y = Float64[1:nlayers;]
        return VisualizationSpec(:mp_landscape;
                                 title="Multiparameter landscape",
                                 subtitle="slice-weighted mean landscape values",
                                 layers=AbstractVisualizationLayer[
                                     HeatmapLayer(tg, y, vals, :viridis, 1.0, "landscape value"),
                                 ],
                                 axes=_default_axes_2d(xlabel="t", ylabel="layer",
                                                       xlimits=(minimum(tg), maximum(tg)),
                                                       ylimits=(minimum(y), maximum(y))),
                                 metadata=(; ndirections=ndirections(L), noffsets=noffsets(L),
                                            kmax=landscape_layers(L), render_mode=:heatmap,
                                            figure_size=(820, 520), legend_position=:none))
    elseif kind === :landscape_slices
        ls = Float64.(landscape_slice(L, idir, ioff))
        layer_idx = min(layer, size(ls, 1))
        vals = ls[layer_idx, :]
        path = NTuple{2,Float64}[(tg[j], vals[j]) for j in eachindex(tg)]
        return VisualizationSpec(:landscape_slices;
                                 title="Landscape slice",
                                 subtitle="one slice/layer from the sampled landscape family",
                                 layers=AbstractVisualizationLayer[
                                     PolylineLayer([path], _landscape_color(layer_idx), 1.0, 2.5, false),
                                 ],
                                 axes=_default_axes_2d(xlabel="t", ylabel="landscape value",
                                                       xlimits=(minimum(tg), maximum(tg)),
                                                       ylimits=_landscape_ylim(vals)),
                                 metadata=(; idir=idir, ioff=ioff, requested_layer=layer,
                                            direction=slice_directions(L)[idir], offset=slice_offsets(L)[ioff]))
    end
    throw(ArgumentError("Unsupported MPLandscape visualization kind $(kind)."))
end
