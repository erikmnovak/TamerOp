# Data-ingestion and graded-complex visualization builders.

@inline _image_array(img::DataTypes.ImageNd) = getfield(img, :data)

function _variance_order(A::AbstractMatrix{<:Real})
    d = size(A, 2)
    n = size(A, 1)
    n == 0 && return collect(1:d)
    means = Vector{Float64}(undef, d)
    vars = Vector{Float64}(undef, d)
    @inbounds for j in 1:d
        s = 0.0
        for i in 1:n
            s += float(A[i, j])
        end
        mu = s / n
        means[j] = mu
        ss = 0.0
        for i in 1:n
            delta = float(A[i, j]) - mu
            ss += delta * delta
        end
        vars[j] = ss / max(n - 1, 1)
    end
    return sortperm(vars; rev=true)
end

function _project_coords(A::AbstractMatrix{<:Real}, target_dim::Int; dims=nothing)
    d = size(A, 2)
    chosen = if dims === nothing
        if d <= target_dim
            collect(1:d)
        else
            order = _variance_order(A)
            sort(order[1:min(target_dim, length(order))])
        end
    else
        collect(Int, dims)
    end
    isempty(chosen) && (chosen = collect(1:min(target_dim, d)))
    P = Matrix{Float64}(undef, size(A, 1), length(chosen))
    @inbounds for j in eachindex(chosen), i in 1:size(A, 1)
        P[i, j] = float(A[i, chosen[j]])
    end
    return P, chosen
end

function _bbox_from_matrix(A::AbstractMatrix{<:Real})
    size(A, 1) == 0 && return nothing
    mins = [minimum(Float64.(A[:, j])) for j in 1:size(A, 2)]
    maxs = [maximum(Float64.(A[:, j])) for j in 1:size(A, 2)]
    return mins, maxs
end

function _pad_limits_1d(lims::Tuple{<:Real,<:Real}; frac::Float64=0.06, minpad::Float64=0.05)
    lo = float(lims[1])
    hi = float(lims[2])
    pad = max(minpad, frac * max(abs(hi - lo), 1.0))
    return (lo - pad, hi + pad)
end

function _axes_from_matrix_2d(A::AbstractMatrix{<:Real}; labels=("x1", "x2"))
    bbox = _bbox_from_matrix(A)
    bbox === nothing && return _default_axes_2d(xlabel=labels[1], ylabel=labels[2])
    return _default_axes_2d(xlabel=labels[1], ylabel=labels[2],
                            xlimits=_pad_limits_1d((bbox[1][1], bbox[2][1])),
                            ylimits=_pad_limits_1d((bbox[1][2], bbox[2][2])),
                            aspect=:equal)
end

function _axes_from_matrix_3d(A::AbstractMatrix{<:Real}; labels=("x1", "x2", "x3"))
    bbox = _bbox_from_matrix(A)
    bbox === nothing && return _default_axes_3d(xlabel=labels[1], ylabel=labels[2], zlabel=labels[3])
    return _default_axes_3d(xlabel=labels[1], ylabel=labels[2], zlabel=labels[3],
                            xlimits=_pad_limits_1d((bbox[1][1], bbox[2][1])),
                            ylimits=_pad_limits_1d((bbox[1][2], bbox[2][2])),
                            zlimits=_pad_limits_1d((bbox[1][3], bbox[2][3])),
                            aspect=:data)
end

function _collect_point_labels(labels, n::Int)
    labels === nothing && return String[]
    out = String[]
    if labels isa AbstractVector
        length(labels) == n || throw(ArgumentError("labels length must equal the number of plotted points."))
        sizehint!(out, n)
        for lbl in labels
            push!(out, string(lbl))
        end
        return out
    end
    throw(ArgumentError("labels must be an AbstractVector or nothing."))
end

function _density_heatmap(points::AbstractMatrix{<:Real}; nbins::Int=24)
    size(points, 2) == 2 || throw(ArgumentError("density overlay requires 2-dimensional points."))
    size(points, 1) == 0 && return HeatmapLayer([0.0, 1.0], [0.0, 1.0], zeros(2, 2), :magma, 0.65, "point density")
    xs = Float64.(points[:, 1])
    ys = Float64.(points[:, 2])
    xlo, xhi = minimum(xs), maximum(xs)
    ylo, yhi = minimum(ys), maximum(ys)
    xspan = max(xhi - xlo, 1e-9)
    yspan = max(yhi - ylo, 1e-9)
    vals = zeros(Float64, nbins, nbins)
    @inbounds for i in eachindex(xs)
        bx = clamp(1 + floor(Int, ((xs[i] - xlo) / xspan) * (nbins - 1)), 1, nbins)
        by = clamp(1 + floor(Int, ((ys[i] - ylo) / yspan) * (nbins - 1)), 1, nbins)
        vals[by, bx] += 1.0
    end
    xgrid = collect(range(xlo, xhi; length=nbins))
    ygrid = collect(range(ylo, yhi; length=nbins))
    return HeatmapLayer(xgrid, ygrid, vals, :magma, 0.65, "point density")
end

@inline function _dist2(A::AbstractMatrix{<:Real}, i::Int, j::Int)
    s = 0.0
    @inbounds for k in 1:size(A, 2)
        delta = float(A[i, k]) - float(A[j, k])
        s += delta * delta
    end
    return s
end

function _knn_edges(A::AbstractMatrix{<:Real}, k::Int)
    n = size(A, 1)
    0 < k < n || throw(ArgumentError("k must satisfy 0 < k < npoints."))
    seen = Set{Tuple{Int,Int}}()
    edges = Tuple{Int,Int}[]
    @inbounds for i in 1:n
        dists = Tuple{Float64,Int}[]
        sizehint!(dists, n - 1)
        for j in 1:n
            i == j && continue
            push!(dists, (_dist2(A, i, j), j))
        end
        sort!(dists, by=first)
        for (_, j) in dists[1:min(k, length(dists))]
            e = i < j ? (i, j) : (j, i)
            if !(e in seen)
                push!(seen, e)
                push!(edges, e)
            end
        end
    end
    return edges
end

function _radius_edges(A::AbstractMatrix{<:Real}, radius::Real)
    n = size(A, 1)
    r2 = float(radius)^2
    edges = Tuple{Int,Int}[]
    @inbounds for i in 1:(n - 1), j in (i + 1):n
        _dist2(A, i, j) <= r2 && push!(edges, (i, j))
    end
    return edges
end

function _segments_from_edges_2d(points::AbstractMatrix{<:Real}, edges)
    segs = NTuple{4,Float64}[]
    @inbounds for (u, v) in edges
        push!(segs, (float(points[u, 1]), float(points[u, 2]),
                     float(points[v, 1]), float(points[v, 2])))
    end
    return segs
end

function _segments_from_edges_3d(points::AbstractMatrix{<:Real}, edges)
    segs = NTuple{6,Float64}[]
    @inbounds for (u, v) in edges
        push!(segs, (float(points[u, 1]), float(points[u, 2]), float(points[u, 3]),
                     float(points[v, 1]), float(points[v, 2]), float(points[v, 3])))
    end
    return segs
end

function _graph_layout_2d(data::DataTypes.GraphData; dims=nothing)
    coords = DataTypes.coord_matrix(data)
    if coords === nothing
        n = DataTypes.nvertices(data)
        pts = Matrix{Float64}(undef, n, 2)
        @inbounds for i in 1:n
            theta = 2 * pi * (i - 1) / max(n, 1)
            pts[i, 1] = cos(theta)
            pts[i, 2] = sin(theta)
        end
        return pts, [1, 2], :circular
    end
    P, chosen = _project_coords(coords, 2; dims=dims)
    return P, chosen, :embedded
end

function _graph_layout_3d(data::DataTypes.GraphData; dims=nothing)
    coords = DataTypes.coord_matrix(data)
    if coords === nothing
        n = DataTypes.nvertices(data)
        pts = Matrix{Float64}(undef, n, 3)
        @inbounds for i in 1:n
            theta = 2 * pi * (i - 1) / max(n, 1)
            pts[i, 1] = cos(theta)
            pts[i, 2] = sin(theta)
            pts[i, 3] = 0.25 * sin(2 * theta)
        end
        return pts, [1, 2, 3], :circular
    end
    P, chosen = _project_coords(coords, 3; dims=dims)
    return P, chosen, :embedded
end

function _weighted_edge_layers_2d(points::AbstractMatrix{<:Real}, edges, weights::AbstractVector{<:Real})
    layers = AbstractVisualizationLayer[]
    legend_entries = NamedTuple[]
    isempty(edges) && return layers, legend_entries
    wmin = minimum(Float64.(weights))
    wmax = maximum(Float64.(weights))
    span = max(wmax - wmin, 1e-9)
    palette = (:steelblue3, :darkorange2, :crimson)
    bins = [NTuple{4,Float64}[] for _ in 1:3]
    @inbounds for (edge, w) in zip(edges, weights)
        q = clamp(1 + floor(Int, 2 * (float(w) - wmin) / span), 1, 3)
        push!(bins[q], (float(points[edge[1], 1]), float(points[edge[1], 2]),
                        float(points[edge[2], 1]), float(points[edge[2], 2])))
    end
    for i in 1:3
        isempty(bins[i]) && continue
        push!(layers, SegmentLayer(bins[i], palette[i], 0.7, 1.0 + 0.8 * (i - 1)))
        lo = wmin + (i - 1) * span / 3
        hi = wmin + i * span / 3
        push!(legend_entries, (; label="$(round(lo; digits=3)) - $(round(hi; digits=3))", color=palette[i], style=:line))
    end
    return layers, legend_entries
end

function _weighted_edge_layers_3d(points::AbstractMatrix{<:Real}, edges, weights::AbstractVector{<:Real})
    layers = AbstractVisualizationLayer[]
    legend_entries = NamedTuple[]
    isempty(edges) && return layers, legend_entries
    wmin = minimum(Float64.(weights))
    wmax = maximum(Float64.(weights))
    span = max(wmax - wmin, 1e-9)
    palette = (:steelblue3, :darkorange2, :crimson)
    bins = [NTuple{6,Float64}[] for _ in 1:3]
    @inbounds for (edge, w) in zip(edges, weights)
        q = clamp(1 + floor(Int, 2 * (float(w) - wmin) / span), 1, 3)
        push!(bins[q], (float(points[edge[1], 1]), float(points[edge[1], 2]), float(points[edge[1], 3]),
                        float(points[edge[2], 1]), float(points[edge[2], 2]), float(points[edge[2], 3])))
    end
    for i in 1:3
        isempty(bins[i]) && continue
        push!(layers, Segment3Layer(bins[i], palette[i], 0.7, 1.0 + 0.8 * (i - 1)))
        lo = wmin + (i - 1) * span / 3
        hi = wmin + i * span / 3
        push!(legend_entries, (; label="$(round(lo; digits=3)) - $(round(hi; digits=3))", color=palette[i], style=:line))
    end
    return layers, legend_entries
end

function _text_panel_spec(kind::Symbol, title::AbstractString, lines::AbstractVector{<:AbstractString}; metadata::NamedTuple=NamedTuple())
    n = max(length(lines), 1)
    positions = [(0.02, float(n - i + 1)) for i in 1:n]
    labels = n == 0 ? [""] : String[string(line) for line in lines]
    return VisualizationSpec(kind;
                             title=title,
                             layers=AbstractVisualizationLayer[
                                 TextLayer(labels, positions, :black, 12.0),
                             ],
                             axes=_default_axes_2d(xlabel="", ylabel="",
                                                   xlimits=(0.0, 1.0),
                                                   ylimits=(0.0, float(n + 1)),
                                                   aspect=:auto),
                             metadata=(; metadata..., panel_style=:text_only))
end

function _bar_spec(kind::Symbol, title::AbstractString, labels::AbstractVector{<:AbstractString},
                   values::AbstractVector{<:Real}; ylabel::AbstractString="value",
                   fill_color::Symbol=:dodgerblue, metadata::NamedTuple=NamedTuple())
    n = length(values)
    n == 0 && return _text_panel_spec(kind, title, ["No values are available for this diagnostic."];
                                      metadata=(; metadata..., empty=true))
    rects = NTuple{4,Float64}[]
    positions = NTuple{2,Float64}[]
    text_labels = String[]
    ymax = max(maximum(Float64.(values)), 1.0)
    @inbounds for i in 1:n
        xlo = float(i) - 0.4
        xhi = float(i) + 0.4
        yhi = max(0.0, float(values[i]))
        push!(rects, (xlo, 0.0, xhi, yhi))
        push!(positions, (float(i), yhi + 0.03 * ymax))
        push!(text_labels, string(round(float(values[i]); digits=3)))
    end
    return VisualizationSpec(kind;
                             title=title,
                             layers=AbstractVisualizationLayer[
                                 RectLayer(rects, fill_color, :black, 0.45, 1.0),
                                 TextLayer(text_labels, positions, :black, 10.0),
                             ],
                             axes=_default_axes_2d(xlabel="", ylabel=ylabel,
                                                   xlimits=(0.25, float(n) + 0.75),
                                                   ylimits=(0.0, 1.15 * ymax),
                                                   xticks=(Float64.(1:n), collect(labels))),
                             metadata=metadata)
end

function available_visuals(pc::DataTypes.PointCloud)
    d = DataTypes.ambient_dim(pc)
    if d >= 3
        return (:points_3d, :points_2d, :point_density, :knn_graph, :radius_graph)
    end
    return (:points_2d, :point_density, :knn_graph, :radius_graph)
end

function available_visuals(res::DataIngestion.PointCodensityResult)
    base = collect(available_visuals(DataIngestion.source_data(res)))
    DataTypes.ambient_dim(DataIngestion.source_data(res)) == 2 &&
        push!(base, :codensity_radius_snapshots)
    return Tuple(base)
end

@inline function _point_layer_from_matrix(A::AbstractMatrix{<:Real}; color=:dodgerblue, alpha::Float64=1.0,
                                          markersize::Float64=8.0, colormap::Symbol=:viridis)
    pts = NTuple{2,Float64}[(float(A[i, 1]), float(A[i, 2])) for i in 1:size(A, 1)]
    if color isa AbstractVector
        return PointLayer(pts, Float64[float(v) for v in color], alpha, markersize, colormap)
    end
    return PointLayer(pts, color, alpha, markersize, colormap)
end

@inline function _codensity_snapshot_cutoffs(vals::AbstractVector{<:Real}, levels)
    if levels isa Symbol
        levels === :quantiles ||
            throw(ArgumentError("codensity_radius_snapshots codensity_levels must be :quantiles or an explicit vector of cutoffs."))
        return Float64.(quantile(Float64.(vals), [0.38, 0.62, 0.84]))
    elseif levels isa AbstractVector
        isempty(levels) && throw(ArgumentError("codensity_radius_snapshots requires at least one codensity cutoff."))
        return Float64[float(v) for v in levels]
    end
    throw(ArgumentError("codensity_radius_snapshots codensity_levels must be :quantiles or an explicit vector of cutoffs."))
end

function _codensity_snapshot_panel_spec(coords::AbstractMatrix{<:Real},
                                        vals::AbstractVector{<:Real},
                                        cutoff::Real,
                                        radius::Real)
    cutoff64 = float(cutoff)
    radius64 = float(radius)
    keep = Float64.(vals) .<= cutoff64
    retained = coords[keep, :]
    retained_vals = Float64.(vals[keep])
    layers = AbstractVisualizationLayer[
        _point_layer_from_matrix(coords; color=:gray88, alpha=0.95, markersize=5.0),
        _point_layer_from_matrix(retained; color=retained_vals, alpha=0.18, markersize=340.0 * radius64, colormap=:viridis),
        _point_layer_from_matrix(retained; color=retained_vals, alpha=0.95, markersize=8.0, colormap=:viridis),
    ]
    return VisualizationSpec(:codensity_radius_snapshot;
                             title="c <= $(round(cutoff64; digits=3)), r = $(round(radius64; digits=2))",
                             subtitle="$(count(identity, keep)) retained points",
                             layers=layers,
                             axes=_axes_from_matrix_2d(coords),
                             metadata=(; cutoff=cutoff64, radius=radius64, nretained=count(identity, keep)))
end

function _visual_spec(res::DataIngestion.PointCodensityResult, kind::Symbol;
                      radii::AbstractVector{<:Real}=[0.07, 0.16, 0.30],
                      codensity_levels=:quantiles,
                      kwargs...)
    vals = DataIngestion.codensity_values(res)
    data = DataIngestion.source_data(res)
    if kind === :codensity_radius_snapshots
        DataTypes.ambient_dim(data) == 2 ||
            throw(ArgumentError("codensity_radius_snapshots requires a 2-dimensional point cloud."))
        isempty(radii) &&
            throw(ArgumentError("codensity_radius_snapshots requires at least one radius."))
        all(float(r) > 0.0 for r in radii) ||
            throw(ArgumentError("codensity_radius_snapshots radii must be positive."))
        coords = Matrix{Float64}(DataTypes.point_matrix(data))
        cutoffs = _codensity_snapshot_cutoffs(vals, codensity_levels)
        panels = VisualizationSpec[]
        sizehint!(panels, length(cutoffs) * length(radii))
        for cutoff in reverse(cutoffs), radius in radii
            push!(panels, _codensity_snapshot_panel_spec(coords, vals, cutoff, radius))
        end
        return VisualizationSpec(:codensity_radius_snapshots;
                                 title="Codensity/radius snapshots",
                                 subtitle="rows filter by codensity cutoff; columns increase proxy radius",
                                 panels=panels,
                                 metadata=(;
                                     object=:point_codensity_result,
                                     dtm_mass=DataIngestion.codensity_mass(res),
                                     neighbor_count=DataIngestion.neighbor_count(res),
                                     nlevels=length(cutoffs),
                                     nradii=length(radii),
                                     codensity_levels=Tuple(cutoffs),
                                     radii=Tuple(Float64.(radii)),
                                     panel_columns=max(length(radii), 1),
                                     figure_size=(980, 760),
                                 ))
    end
    base = _visual_spec(data, kind; color_values=Float64.(vals), kwargs...)
    return VisualizationSpec(base.kind;
                             title=base.title,
                             subtitle=base.subtitle,
                             layers=base.layers,
                             panels=base.panels,
                             axes=base.axes,
                             legend=base.legend,
                             interaction=base.interaction,
                             metadata=(;
                                 base.metadata...,
                                 object=:point_codensity_result,
                                 dtm_mass=DataIngestion.codensity_mass(res),
                                 neighbor_count=DataIngestion.neighbor_count(res),
                                 value_range=isempty(vals) ? nothing : extrema(vals),
                             ))
end

function _visual_spec(pc::DataTypes.PointCloud, kind::Symbol; dims=nothing, labels=nothing,
                      color_values=nothing, density::Bool=false, k::Int=6,
                      radius::Union{Nothing,Real}=nothing, kwargs...)
    _ = kwargs
    A = DataTypes.point_matrix(pc)
    if kind === :points_3d
        P, chosen = _project_coords(A, min(size(A, 2), 3); dims=dims)
        size(P, 2) == 3 || return _visual_spec(pc, :points_2d; dims=chosen, labels=labels, color_values=color_values, density=density)
        pts = NTuple{3,Float64}[(P[i, 1], P[i, 2], P[i, 3]) for i in 1:size(P, 1)]
        ax = _axes_from_matrix_3d(P; labels=Tuple("x$(d)" for d in chosen))
        layers = AbstractVisualizationLayer[Point3Layer(pts,
                                                        color_values === nothing ? :dodgerblue3 : Float64.(color_values),
                                                        0.85, 10.0,
                                                        :viridis)]
        metadata = (; object=:point_cloud,
                    projected_dims=Tuple(chosen),
                    ambient_dim=DataTypes.ambient_dim(pc),
                    figure_size=(760, 620))
        return VisualizationSpec(:points_3d;
                                 title="Point cloud preview",
                                 subtitle="3D point-cloud view",
                                 layers=layers,
                                 axes=ax,
                                 legend=_default_legend(visible=false, title="layers",
                                                        entries=[(; label="points", color=:dodgerblue3, style=:marker)]),
                                 metadata=metadata,
                                 interaction=_default_interaction(hover=true))
    elseif kind === :points_2d || kind === :point_density || kind === :knn_graph || kind === :radius_graph
        P, chosen = _project_coords(A, min(size(A, 2), 2); dims=dims)
        ax = _axes_from_matrix_2d(P; labels=Tuple("x$(d)" for d in chosen))
        layers = AbstractVisualizationLayer[]
        legend_entries = NamedTuple[]
        show_density = density || kind === :point_density
        show_density && push!(layers, _density_heatmap(P))
        if kind === :knn_graph
            edges = size(A, 1) <= 1 ? Tuple{Int,Int}[] : _knn_edges(A, min(k, size(A, 1) - 1))
            push!(layers, SegmentLayer(_segments_from_edges_2d(P, edges), :gray35, 0.62, 1.3))
            push!(legend_entries, (; label="kNN edges", color=:gray55, style=:line))
        elseif kind === :radius_graph
            rr = radius === nothing ? sqrt(maximum(_dist2(A, 1, j) for j in 2:size(A, 1))) / 2 : float(radius)
            edges = size(A, 1) <= 1 ? Tuple{Int,Int}[] : _radius_edges(A, rr)
            push!(layers, SegmentLayer(_segments_from_edges_2d(P, edges), :gray35, 0.62, 1.3))
            push!(legend_entries, (; label="radius edges", color=:gray55, style=:line))
        end
        push!(layers, PointLayer(NTuple{2,Float64}[(P[i, 1], P[i, 2]) for i in 1:size(P, 1)],
                                 color_values === nothing ? :dodgerblue3 : Float64.(color_values),
                                 0.82, kind === :points_2d || kind === :point_density ? 9.5 : 7.5,
                                 :viridis))
        push!(legend_entries, (; label="points", color=:dodgerblue3, style=:marker))
        labvec = _collect_point_labels(labels, size(P, 1))
        isempty(labvec) || push!(layers, TextLayer(labvec,
                                                   [(P[i, 1] + 0.02, P[i, 2] + 0.02) for i in 1:size(P, 1)],
                                                   :black, 9.0))
        metadata = (; object=:point_cloud,
                    projected_dims=Tuple(chosen),
                    ambient_dim=DataTypes.ambient_dim(pc),
                    npoints=DataTypes.npoints(pc),
                    overlay_density=show_density,
                    legend_position=kind === :knn_graph || kind === :radius_graph ? :right : :none,
                    figure_size=kind === :point_density ? (760, 520) :
                                kind === :points_2d ? (720, 520) : (780, 560))
        return VisualizationSpec(kind;
                                 title="Point cloud preview",
                                 subtitle=kind === :point_density ? "point-density overlay" :
                                          kind === :knn_graph ? "point cloud with kNN preview graph" :
                                          kind === :radius_graph ? "point cloud with radius preview graph" :
                                          "2D point-cloud view",
                                 layers=layers,
                                 axes=ax,
                                 legend=_default_legend(visible=kind === :knn_graph || kind === :radius_graph,
                                                        title="layers",
                                                        entries=legend_entries),
                                 metadata=metadata,
                                 interaction=_default_interaction(hover=true, labels=!isempty(labvec)))
    end
    throw(ArgumentError("Unsupported PointCloud visualization kind $(kind)."))
end

function available_visuals(data::DataTypes.GraphData)
    d = DataTypes.ambient_dim(data)
    if d !== nothing && d >= 3
        return (:graph_3d, :graph, :weighted_graph)
    end
    return (:graph, :weighted_graph)
end

function _visual_spec(data::DataTypes.GraphData, kind::Symbol; dims=nothing, labels=nothing, kwargs...)
    _ = kwargs
    if kind === :graph_3d
        pts, chosen, layout = _graph_layout_3d(data; dims=dims)
        edges = collect(DataTypes.edge_list(data))
        layers = AbstractVisualizationLayer[Segment3Layer(_segments_from_edges_3d(pts, edges), :gray55, 0.55, 1.2)]
        w = DataTypes.edge_weights(data)
        legend_entries = [(; label="edges", color=:gray55, style=:line)]
        if w !== nothing
            weight_layers, extra_entries = _weighted_edge_layers_3d(pts, edges, w)
            layers = isempty(weight_layers) ? AbstractVisualizationLayer[] : AbstractVisualizationLayer[weight_layers...]
            legend_entries = isempty(extra_entries) ? legend_entries : extra_entries
        else
            layers = AbstractVisualizationLayer[layers...]
        end
        push!(layers, Point3Layer([tuple(pts[i, 1], pts[i, 2], pts[i, 3]) for i in 1:size(pts, 1)], :black, 0.9, 9.0))
        return VisualizationSpec(:graph_3d;
                                 title="Graph preview",
                                 subtitle="3D graph view ($(layout))",
                                 layers=layers,
                                 axes=_axes_from_matrix_3d(pts; labels=Tuple("x$(d)" for d in chosen)),
                                 legend=_default_legend(visible=true, title="edges", entries=legend_entries),
                                 metadata=(; object=:graph_data, projected_dims=Tuple(chosen), layout))
    elseif kind === :graph || kind === :weighted_graph
        pts, chosen, layout = _graph_layout_2d(data; dims=dims)
        edges = collect(DataTypes.edge_list(data))
        layers = AbstractVisualizationLayer[]
        legend_entries = NamedTuple[]
        weights = DataTypes.edge_weights(data)
        if kind === :weighted_graph && weights !== nothing
            edge_layers, legend_entries = _weighted_edge_layers_2d(pts, edges, weights)
            append!(layers, edge_layers)
        else
            push!(layers, SegmentLayer(_segments_from_edges_2d(pts, edges), :gray55, 0.55, 1.2))
            push!(legend_entries, (; label="edges", color=:gray55, style=:line))
        end
        push!(layers, PointLayer([tuple(pts[i, 1], pts[i, 2]) for i in 1:size(pts, 1)], :black, 0.9, 9.0))
        push!(legend_entries, (; label="vertices", color=:black, style=:marker))
        labvec = _collect_point_labels(labels, size(pts, 1))
        isempty(labvec) || push!(layers, TextLayer(labvec,
                                                   [(pts[i, 1] + 0.02, pts[i, 2] + 0.02) for i in 1:size(pts, 1)],
                                                   :black, 9.0))
        return VisualizationSpec(kind;
                                 title="Graph preview",
                                 subtitle=kind === :weighted_graph && weights !== nothing ? "weighted-edge view" : "node-edge view",
                                 layers=layers,
                                 axes=_axes_from_matrix_2d(pts; labels=Tuple("x$(d)" for d in chosen)),
                                 legend=_default_legend(visible=true, title="layers", entries=legend_entries),
                                 metadata=(; object=:graph_data, projected_dims=Tuple(chosen), layout, nvertices=DataTypes.nvertices(data), nedges=DataTypes.nedges(data)),
                                 interaction=_default_interaction(hover=true, labels=!isempty(labvec)))
    end
    throw(ArgumentError("Unsupported GraphData visualization kind $(kind)."))
end

available_visuals(::DataTypes.EmbeddedPlanarGraph2D) = (:embedded_planar_graph,)

function _visual_spec(data::DataTypes.EmbeddedPlanarGraph2D, kind::Symbol; labels=nothing, kwargs...)
    _ = kwargs
    kind === :embedded_planar_graph || throw(ArgumentError("Unsupported EmbeddedPlanarGraph2D visualization kind $(kind)."))
    verts = reduce(vcat, [[float(v[1]) float(v[2])] for v in DataTypes.vertex_positions(data)])
    pts = size(verts, 1) == 0 ? zeros(0, 2) : reshape(verts, :, 2)
    poly = DataTypes.polylines(data)
    layers = AbstractVisualizationLayer[]
    if poly === nothing
        push!(layers, SegmentLayer(_segments_from_edges_2d(pts, collect(DataTypes.edge_list(data))), :gray40, 0.7, 1.3))
    else
        paths = Vector{Vector{NTuple{2,Float64}}}()
        for line in poly
            push!(paths, [tuple(float(p[1]), float(p[2])) for p in line])
        end
        push!(layers, PolylineLayer(paths, :gray40, 0.8, 1.6, false))
    end
    push!(layers, PointLayer([tuple(pts[i, 1], pts[i, 2]) for i in 1:size(pts, 1)], :black, 0.9, 8.5))
    labvec = _collect_point_labels(labels, size(pts, 1))
    isempty(labvec) || push!(layers, TextLayer(labvec, [(pts[i, 1] + 0.02, pts[i, 2] + 0.02) for i in 1:size(pts, 1)], :black, 9.0))
    bbox = DataTypes.bounding_box(data)
    axes = if bbox === nothing
        _axes_from_matrix_2d(pts; labels=("x", "y"))
    else
        _default_axes_2d(xlabel="x", ylabel="y",
                         xlimits=_pad_limits_1d((bbox[1], bbox[3])),
                         ylimits=_pad_limits_1d((bbox[2], bbox[4])),
                         aspect=:equal)
    end
    return VisualizationSpec(:embedded_planar_graph;
                             title="Embedded planar graph",
                             subtitle="planar graph view",
                             layers=layers,
                             axes=axes,
                             legend=_default_legend(visible=true, title="layers",
                                                    entries=[(; label="edges", color=:gray40, style=:line),
                                                             (; label="vertices", color=:black, style=:marker)]),
                             metadata=(; object=:embedded_planar_graph_2d, nvertices=DataTypes.nvertices(data), nedges=DataTypes.nedges(data)),
                             interaction=_default_interaction(hover=true, labels=!isempty(labvec)))
end

function available_visuals(img::DataTypes.ImageNd{T,N}) where {T,N}
    if N == 2
        return (:image,)
    elseif N == 3
        if size(_image_array(img), 3) <= 4
            return (:image, :channels, :slice_viewer)
        end
        return (:image, :slice_viewer)
    end
    return (:image, :slice_viewer)
end

function _image_slice_matrix(img::DataTypes.ImageNd, view_dims::Tuple{Int,Int}, fixed::Dict{Int,Int})
    A = _image_array(img)
    idx = Any[Colon() for _ in 1:ndims(A)]
    for (dim, pos) in fixed
        idx[dim] = clamp(Int(pos), 1, size(A, dim))
    end
    idx[view_dims[1]] = Colon()
    idx[view_dims[2]] = Colon()
    slice = Float64.(A[idx...])
    ndims(slice) == 2 || (slice = reshape(slice, size(A, view_dims[1]), size(A, view_dims[2])))
    return slice
end

function _image_default_view(img::DataTypes.ImageNd)
    A = _image_array(img)
    nd = ndims(A)
    if nd == 2
        return (1, 2), Dict{Int,Int}()
    elseif nd == 3 && size(A, 3) <= 4
        return (1, 2), Dict(3 => clamp(1, 1, size(A, 3)))
    else
        fixed = Dict{Int,Int}()
        for d in 3:nd
            fixed[d] = cld(size(A, d), 2)
        end
        return (1, 2), fixed
    end
end

function _image_spec(img::DataTypes.ImageNd; kind::Symbol=:image, view_dims=nothing,
                     slice_indices=nothing, colormap::Symbol=:magma, kwargs...)
    _ = kwargs
    A = _image_array(img)
    vd, fixed = _image_default_view(img)
    view_dims === nothing || (vd = (Int(view_dims[1]), Int(view_dims[2])))
    if slice_indices !== nothing
        fixed = Dict{Int,Int}(Int(k) => Int(v) for (k, v) in pairs(slice_indices))
    end
    vals = if kind === :channels && ndims(A) == 3 && size(A, 3) <= 4
        mats = [Float64.(A[:, :, c]) for c in 1:size(A, 3)]
        x = Float64[1:size(A, 2);]
        y = Float64[1:size(A, 1);]
        panels = VisualizationSpec[]
        for c in 1:length(mats)
            push!(panels, VisualizationSpec(:channels;
                                            title="channel $c",
                                            layers=AbstractVisualizationLayer[
                                                HeatmapLayer(x, y, mats[c], colormap, 1.0, "intensity"),
                                            ],
                                            axes=_default_axes_2d(xlabel="x$(vd[1])", ylabel="x$(vd[2])",
                                                                  xlimits=(minimum(x), maximum(x)),
                                                                  ylimits=(minimum(y), maximum(y)))))
        end
        return VisualizationSpec(:channels;
                                 title="Image channels",
                                 subtitle="one panel per stored channel",
                                 panels=panels,
                                 metadata=(; shape=size(A), nchannels=size(A, 3)))
    else
        _image_slice_matrix(img, vd, fixed)
    end
    x = Float64[1:size(vals, 2);]
    y = Float64[1:size(vals, 1);]
    widgets = kind === :slice_viewer ? (:slice_index,) : ()
    subtitle = kind === :slice_viewer ? "interactive slice preview" : "heatmap preview"
    return VisualizationSpec(kind;
                             title="Image preview",
                             subtitle=subtitle,
                             layers=AbstractVisualizationLayer[
                                 HeatmapLayer(x, y, vals, colormap, 1.0, "intensity"),
                             ],
                             axes=_default_axes_2d(xlabel="axis $(vd[1])", ylabel="axis $(vd[2])",
                                                   xlimits=(minimum(x), maximum(x)),
                                                   ylimits=(minimum(y), maximum(y))),
                             metadata=(; shape=size(A), view_dims=vd, fixed_indices=Dict(fixed), volume=A, colormap),
                             interaction=_default_interaction(hover=true, widgets=widgets, notebook=kind === :slice_viewer ? :widget_viewer : :summary_card))
end

function _visual_spec(img::DataTypes.ImageNd, kind::Symbol; kwargs...)
    kind in (:image, :channels, :slice_viewer) || throw(ArgumentError("Unsupported ImageNd visualization kind $(kind)."))
    return _image_spec(img; kind=kind, kwargs...)
end

available_visuals(::DataIngestion.IngestionEstimate) = (:simplex_counts,)
available_visuals(::DataIngestion.IngestionPlan) = ()
available_visuals(::DataIngestion.GradedComplexBuildResult) = ()
available_visuals(::DataTypes.GradedComplex) = ()
available_visuals(::DataTypes.MultiCriticalGradedComplex) = ()
available_visuals(::DataTypes.SimplexTreeMulti) = (:simplex_counts,)

function _visual_spec(est::DataIngestion.IngestionEstimate, kind::Symbol; kwargs...)
    _ = kwargs
    if kind === :simplex_counts
        counts = DataIngestion.cell_counts_by_dim(est)
        return _bar_spec(:simplex_counts, "estimated cells by dimension",
                         ["dim $(d - 1)" for d in 1:length(counts)], counts;
                         ylabel="estimated cells",
                         fill_color=:steelblue3,
                         metadata=(; counts))
    end
    throw(ArgumentError("Unsupported IngestionEstimate visualization kind $(kind)."))
end

function _visual_spec(res::DataIngestion.GradedComplexBuildResult, kind::Symbol; kwargs...)
    _ = kwargs
    throw(ArgumentError("GradedComplexBuildResult has no visualization kinds. Use describe(buildres) or graded_complex_build_summary(buildres) for diagnostics."))
end

function _visual_spec(G::Union{DataTypes.GradedComplex,DataTypes.MultiCriticalGradedComplex}, kind::Symbol; kwargs...)
    _ = kwargs
    throw(ArgumentError("GradedComplex has no visualization kinds. Use describe(graded_complex) or data_summary(graded_complex) for diagnostics."))
end

function _visual_spec(ST::DataTypes.SimplexTreeMulti, kind::Symbol; kwargs...)
    _ = kwargs
    if kind === :simplex_counts
        counts = DataTypes.cell_counts(ST)
        return _bar_spec(:simplex_counts, "simplex counts by dimension",
                         ["dim $(d - 1)" for d in 1:length(counts)], counts;
                         ylabel="simplices",
                         fill_color=:teal,
                         metadata=(; counts, nsimplices=DataTypes.simplex_count(ST)))
    end
    throw(ArgumentError("Unsupported SimplexTreeMulti visualization kind $(kind)."))
end
