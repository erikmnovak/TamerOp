# Encoding and translation visualization builders, including migrated flange views.

available_visuals(::Any) = ()

function visual_spec(obj; kind::Symbol=:auto, cache=:auto, kwargs...)
    report = check_visual_request(obj; kind=kind, kwargs..., throw=true)
    chosen = report.requested_kind
    return _visual_spec(obj, chosen; cache=cache, kwargs...)
end

_visual_spec(obj, kind::Symbol; kwargs...) =
    throw(ArgumentError("No visualization builder is registered for $(nameof(typeof(obj))) with kind=$(kind)."))

function _as_points2(point)
    if point === nothing
        return NTuple{2,Float64}[]
    elseif point isa NTuple{2,<:Real}
        return [(float(point[1]), float(point[2]))]
    elseif point isa AbstractVector{<:Real}
        length(point) == 2 || throw(ArgumentError("expected a 2-vector point"))
        return [(float(point[1]), float(point[2]))]
    else
        throw(ArgumentError("unsupported point representation $(typeof(point))"))
    end
end

function _collect_query_points(; point=nothing, points=nothing)
    out = NTuple{2,Float64}[]
    point === nothing || append!(out, _as_points2(point))
    if points !== nothing
        if points isa AbstractMatrix{<:Real}
            size(points, 1) == 2 || throw(ArgumentError("points matrix must have 2 rows."))
            @inbounds for j in 1:size(points, 2)
                push!(out, (float(points[1, j]), float(points[2, j])))
            end
        else
            for p in points
                append!(out, _as_points2(p))
            end
        end
    end
    return out
end

function _text_layer_from_labels(points::AbstractVector{<:NTuple{2,<:Real}}, labels::AbstractVector{<:AbstractString}; color::Symbol=:black, textsize::Float64=10.0)
    return TextLayer(String[String(lbl) for lbl in labels],
                     NTuple{2,Float64}[(float(p[1]), float(p[2])) for p in points],
                     color,
                     textsize)
end

function _offset_label_positions(points::AbstractVector{<:NTuple{2,<:Real}}, axes::NamedTuple;
                                 dx_frac::Float64=0.014, dy_frac::Float64=0.010)
    isempty(points) && return NTuple{2,Float64}[]
    xlimits = get(axes, :xlimits, nothing)
    ylimits = get(axes, :ylimits, nothing)
    xspan = xlimits === nothing ? 1.0 : abs(float(xlimits[2]) - float(xlimits[1]))
    yspan = ylimits === nothing ? 1.0 : abs(float(ylimits[2]) - float(ylimits[1]))
    dx = max(0.03, dx_frac * xspan)
    dy = max(0.03, dy_frac * yspan)
    return NTuple{2,Float64}[(float(p[1]) + dx, float(p[2]) + dy) for p in points]
end

function _rect_text_centers(rects::Vector{NTuple{4,Float64}})
    return NTuple{2,Float64}[_midpoint(rect) for rect in rects]
end

function _expanded_axis(axis::AbstractVector{<:Real})
    isempty(axis) && return Float64[]
    if length(axis) == 1
        a = float(axis[1])
        return [a - 0.5, a + 0.5]
    end
    out = Vector{Float64}(undef, length(axis) + 1)
    @inbounds begin
        out[1] = float(axis[1]) - 0.5 * (float(axis[2]) - float(axis[1]))
        for i in 2:length(axis)
            out[i] = 0.5 * (float(axis[i - 1]) + float(axis[i]))
        end
        out[end] = float(axis[end]) + 0.5 * (float(axis[end]) - float(axis[end - 1]))
    end
    return out
end

function _grid_rectangles_2d(pi::GridEncodingMap{2}; box=nothing)
    xedges = _expanded_axis(pi.coords[1])
    yedges = _expanded_axis(pi.coords[2])
    rects = NTuple{4,Float64}[]
    labels = String[]
    for j in 1:(length(yedges) - 1), i in 1:(length(xedges) - 1)
        xlo, xhi = xedges[i], xedges[i + 1]
        ylo, yhi = yedges[j], yedges[j + 1]
        push!(rects, (xlo, ylo, xhi, yhi))
        idx = 1 + (i - 1) * pi.strides[1] + (j - 1) * pi.strides[2]
        push!(labels, string(idx))
    end
    axes = _default_axes_2d(xlabel="x1", ylabel="x2",
                            xlimits=(xedges[1], xedges[end]),
                            ylimits=(yedges[1], yedges[end]))
    return rects, labels, axes
end

function _pl_view_box_2d(pi::PLEncodingMapBoxes)
    reps = region_representatives(pi)
    xr = _finite_extrema((rep[1] for rep in reps if length(rep) == 2))
    yr = _finite_extrema((rep[2] for rep in reps if length(rep) == 2))

    if xr === nothing || yr === nothing
        xr = _finite_extrema(pi.coords[1])
        yr = _finite_extrema(pi.coords[2])
    end

    xr === nothing && (xr = (-1.0, 1.0))
    yr === nothing && (yr = (-1.0, 1.0))

    xpad = xr[1] == xr[2] ? 1.0 : max(1.0, 0.1 * (xr[2] - xr[1]))
    ypad = yr[1] == yr[2] ? 1.0 : max(1.0, 0.1 * (yr[2] - yr[1]))
    return ([xr[1] - xpad, yr[1] - ypad], [xr[2] + xpad, yr[2] + ypad])
end

function _pl_rectangles_2d(pi::PLEncodingMapBoxes; box=nothing)
    view_box = box === nothing ? _pl_view_box_2d(pi) : box
    region_rects = Vector{Vector{NTuple{4,Float64}}}()
    labels = String[]
    label_positions = NTuple{2,Float64}[]
    reps = region_representatives(pi)
    nreg = nregions(pi)
    for r in 1:nreg
        lows, highs = _cells_in_region_in_box(pi, r, view_box; strict=false)
        isempty(lows) && continue
        rects_r = NTuple{4,Float64}[]
        for (lo, hi) in zip(lows, highs)
            if length(lo) == 2 && length(hi) == 2
                push!(rects_r, (float(lo[1]), float(lo[2]), float(hi[1]), float(hi[2])))
            end
        end
        isempty(rects_r) && continue
        push!(region_rects, rects_r)
        push!(labels, string(r))
        rep = reps[r]
        rep_pt = (float(rep[1]), float(rep[2]))
        if view_box[1][1] <= rep_pt[1] <= view_box[2][1] && view_box[1][2] <= rep_pt[2] <= view_box[2][2]
            push!(label_positions, rep_pt)
        else
            rect0 = first(rects_r)
            push!(label_positions, _midpoint(rect0))
        end
    end
    xlimits = (float(view_box[1][1]), float(view_box[2][1]))
    ylimits = (float(view_box[1][2]), float(view_box[2][2]))
    axes = _default_axes_2d(xlabel="x1", ylabel="x2",
                            xlimits=xlimits,
                            ylimits=ylimits)
    return region_rects, labels, axes, label_positions
end

const _BOX_REGION_COLORS = (
    :seagreen3,
    :cornflowerblue,
    :goldenrod2,
    :orchid3,
    :tomato2,
    :slateblue3,
    :darkkhaki,
    :cadetblue3,
)

@inline _box_region_color(i::Int) = _BOX_REGION_COLORS[1 + mod(i - 1, length(_BOX_REGION_COLORS))]

@inline function _segment_key(seg::NTuple{4,Float64})
    x1, y1, x2, y2 = seg
    return (x1 < x2 || (x1 == x2 && y1 <= y2)) ? seg : (x2, y2, x1, y1)
end

function _region_boundary_segments(rects::Vector{NTuple{4,Float64}})
    counts = Dict{NTuple{4,Float64},Int}()
    for rect in rects
        xlo, ylo, xhi, yhi = rect
        for edge in ((xlo, ylo, xhi, ylo),
                     (xhi, ylo, xhi, yhi),
                     (xhi, yhi, xlo, yhi),
                     (xlo, yhi, xlo, ylo))
            key = _segment_key(edge)
            counts[key] = get(counts, key, 0) + 1
        end
    end
    segments = NTuple{4,Float64}[]
    for (seg, count) in counts
        count == 1 && push!(segments, seg)
    end
    return segments
end

function _zn_rectangles_2d(pi::ZnEncodingMap)
    rects = NTuple{4,Float64}[]
    labels = String[]
    reps = region_representatives(pi)
    for (r, rep) in enumerate(reps)
        length(rep) == 2 || continue
        x = float(rep[1])
        y = float(rep[2])
        push!(rects, (x - 0.5, y - 0.5, x + 0.5, y + 0.5))
        push!(labels, string(r))
    end
    lims = _bbox_from_points(_rect_text_centers(rects))
    axes = _default_axes_2d(xlabel="g1", ylabel="g2",
                            xlimits=lims === nothing ? nothing : lims[1],
                            ylimits=lims === nothing ? nothing : lims[2],
                            aspect=:equal)
    return rects, labels, axes
end

function _encoding_region_spec(rects::Vector{NTuple{4,Float64}}, labels::Vector{String};
                               title::AbstractString, kind::Symbol,
                               show_labels::Bool=false, points=NTuple{2,Float64}[], point_labels=String[],
                               label_positions=nothing,
                               metadata::NamedTuple=NamedTuple(), axes::NamedTuple=_default_axes_2d(),
                               fill_color::Symbol=:dodgerblue)
    layers = AbstractVisualizationLayer[
        RectLayer(rects, fill_color, :black, 0.22, 1.0),
    ]
    if show_labels && !isempty(labels)
        positions = label_positions === nothing ? _rect_text_centers(rects) : label_positions
        push!(layers, _text_layer_from_labels(positions, labels; color=:black, textsize=10.0))
    end
    if !isempty(points)
        push!(layers, PointLayer(points, :orange3, 0.95, 14.0))
        isempty(point_labels) || push!(layers, _text_layer_from_labels(points, point_labels; color=:black, textsize=10.0))
    end
    return VisualizationSpec(kind; title=title,
                             layers=layers,
                             axes=axes,
                             metadata=merge((; figure_size=(760, 620), legend_position=:none), metadata),
                             interaction=_default_interaction(hover=true, labels=show_labels || !isempty(point_labels)))
end

function _query_overlay_labels(pi::AbstractPLikeEncodingMap, pts::Vector{NTuple{2,Float64}})
    labels = String[]
    for (j, p) in enumerate(pts)
        region = locate(pi, collect(p))
        push!(labels, string("q", j, " -> ", region))
    end
    return labels
end

function _flange_default_box(FG::Flange)
    xs = Int[]
    ys = Int[]
    for F in flats(FG)
        push!(xs, F.b[1]); push!(ys, F.b[2])
    end
    for E in injectives(FG)
        push!(xs, E.b[1]); push!(ys, E.b[2])
    end
    isempty(xs) && return ([0.0, 0.0], [5.0, 5.0])
    lo = [float(minimum(xs) - 1), float(minimum(ys) - 1)]
    hi = [float(maximum(xs) + 2), float(maximum(ys) + 2)]
    return (lo, hi)
end

function _clip_rect_2d(ell::NTuple{2,Float64}, u::NTuple{2,Float64}; L1=-Inf, U1=Inf, L2=-Inf, U2=Inf)
    xlo = max(ell[1], L1); xhi = min(u[1], U1)
    ylo = max(ell[2], L2); yhi = min(u[2], U2)
    xlo <= xhi && ylo <= yhi || return nothing
    return (xlo, ylo, xhi, yhi)
end

available_visuals(::Flange) = (:regions, :constant_subdivision)

function _visual_spec(FG::Flange, kind::Symbol; box=nothing, alpha_up::Real=0.25, alpha_dn::Real=0.25, kwargs...)
    FG.n == 2 || throw(ArgumentError("Visualization v1 only supports 2D flanges."))
    _ = kwargs
    b = box === nothing ? _flange_default_box(FG) : box
    ell = (float(b[1][1]), float(b[1][2]))
    u = (float(b[2][1]), float(b[2][2]))
    if kind === :regions
        up_rects = NTuple{4,Float64}[]
        dn_rects = NTuple{4,Float64}[]
        up_labels = String[]
        dn_labels = String[]
        for (j, F) in enumerate(flats(FG))
            L1 = F.tau.coords[1] ? -Inf : float(F.b[1])
            L2 = F.tau.coords[2] ? -Inf : float(F.b[2])
            rect = _clip_rect_2d(ell, u; L1=L1, L2=L2)
            rect === nothing && continue
            push!(up_rects, rect)
            push!(up_labels, "U$(j)")
        end
        for (i, E) in enumerate(injectives(FG))
            U1 = E.tau.coords[1] ? Inf : float(E.b[1])
            U2 = E.tau.coords[2] ? Inf : float(E.b[2])
            rect = _clip_rect_2d(ell, u; U1=U1, U2=U2)
            rect === nothing && continue
            push!(dn_rects, rect)
            push!(dn_labels, "D$(i)")
        end
        layers = AbstractVisualizationLayer[
            RectLayer(up_rects, :dodgerblue, :navy, float(alpha_up), 0.0),
            RectLayer(dn_rects, :crimson, :darkred, float(alpha_dn), 0.0),
            _text_layer_from_labels(_rect_text_centers(up_rects), up_labels; color=:navy, textsize=10.0),
            _text_layer_from_labels(_rect_text_centers(dn_rects), dn_labels; color=:darkred, textsize=10.0),
        ]
        return VisualizationSpec(:regions;
                                 title="Flange regions",
                                 subtitle="flats and injectives clipped to a viewing box",
                                 layers=layers,
                                 axes=_default_axes_2d(xlabel="x1", ylabel="x2", xlimits=(ell[1], u[1]), ylimits=(ell[2], u[2]), aspect=:equal),
                                 metadata=(; object=:flange, nflats=length(up_rects), ninjectives=length(dn_rects), box=b),
                                 legend=_default_legend(visible=true, entries=(; flats=:dodgerblue, injectives=:crimson)),
                                 interaction=_default_interaction(hover=true, labels=true))
    elseif kind === :constant_subdivision
        xlo = ceil(Int, ell[1]); ylo = ceil(Int, ell[2])
        xhi = floor(Int, u[1]); yhi = floor(Int, u[2])
        nx = max(0, xhi - xlo)
        ny = max(0, yhi - ylo)
        nx > 0 && ny > 0 || throw(ArgumentError("constant_subdivision requires a box with at least one interior unit cell."))
        vals = Matrix{Float64}(undef, ny, nx)
        for iy in 1:ny, ix in 1:nx
            g = [xlo + ix - 1, ylo + iy - 1]
            cols = active_flats(FG, g)
            rows = active_injectives(FG, g)
            vals[iy, ix] = (isempty(cols) || isempty(rows)) ? 0.0 : float(rank_restricted(FG.field, FG.phi, rows, cols))
        end
        return VisualizationSpec(:constant_subdivision;
                                 title="Flange constant subdivision",
                                 subtitle="heatmap of fiber dimensions on unit cells",
                                 layers=AbstractVisualizationLayer[
                                     HeatmapLayer(Float64[xlo + i - 1 for i in 1:nx], Float64[ylo + j - 1 for j in 1:ny], vals, :viridis, 1.0, "dim"),
                                 ],
                                 axes=_default_axes_2d(xlabel="x1", ylabel="x2", xlimits=(xlo, xhi), ylimits=(ylo, yhi), aspect=:equal),
                                 metadata=(; object=:flange, matrix_size=size(FG.phi), box=b))
    end
    throw(ArgumentError("Unsupported flange visualization kind $(kind)."))
end

available_visuals(::GridEncodingMap{2}) = (:regions, :region_labels, :query_overlay)
available_visuals(::PLEncodingMapBoxes) = (:regions, :region_labels, :query_overlay)
available_visuals(::ZnEncodingMap) = (:regions, :region_labels, :query_overlay)

function _visual_spec(pi::GridEncodingMap{2}, kind::Symbol; point=nothing, points=nothing, kwargs...)
    _ = kwargs
    rects, labels, axes = _grid_rectangles_2d(pi)
    pts = kind === :query_overlay ? _collect_query_points(point=point, points=points) : NTuple{2,Float64}[]
    point_labels = kind === :query_overlay ? _query_overlay_labels(pi, pts) : String[]
    return _encoding_region_spec(rects, labels;
                                 title="Grid encoding",
                                 kind=kind,
                                 show_labels=(kind === :region_labels),
                                 points=pts,
                                 point_labels=point_labels,
                                 axes=axes,
                                 metadata=(; object=:grid_encoding_map, nregions=length(rects), axis_sizes=pi.sizes),
                                 fill_color=:cornflowerblue)
end

function _visual_spec(pi::PLEncodingMapBoxes, kind::Symbol; point=nothing, points=nothing, box=nothing, kwargs...)
    _ = kwargs
    region_rects, labels, axes, label_positions = _pl_rectangles_2d(pi; box=box)
    label_text_positions = _offset_label_positions(label_positions, axes)
    pts = kind === :query_overlay ? _collect_query_points(point=point, points=points) : NTuple{2,Float64}[]
    point_labels = kind === :query_overlay ? _query_overlay_labels(pi, pts) : String[]
    layers = AbstractVisualizationLayer[]
    for (idx, rects_r) in enumerate(region_rects)
        color = _box_region_color(idx)
        push!(layers, RectLayer(rects_r, color, :black, 0.18, 0.8))
        push!(layers, SegmentLayer(_region_boundary_segments(rects_r), color, 0.95, 1.6))
    end
    isempty(label_positions) || push!(layers, PointLayer(label_positions, :black, 0.9, 5.5))
    if kind === :region_labels && !isempty(labels)
        push!(layers, _text_layer_from_labels(label_text_positions, labels; color=:black, textsize=10.0))
    end
    if !isempty(pts)
        push!(layers, PointLayer(pts, :orange3, 0.95, 14.0))
        isempty(point_labels) || push!(layers, _text_layer_from_labels(pts, point_labels; color=:black, textsize=10.0))
    end
    return VisualizationSpec(kind;
                             title="Box encoding",
                             layers=layers,
                             axes=axes,
                             metadata=(; object=:pl_boxes, nregions=length(labels), ncells=sum(length, region_rects),
                                        query_count=length(pts), figure_size=(860, 620), legend_position=:right),
                             legend=_default_legend(visible=true,
                                                    entries=(; (Symbol("R" * string(i)) => _box_region_color(i) for i in 1:length(region_rects))...)),
                             interaction=_default_interaction(hover=true, labels=(kind === :region_labels) || !isempty(point_labels)))
end

function _visual_spec(pi::ZnEncodingMap, kind::Symbol; point=nothing, points=nothing, kwargs...)
    _ = kwargs
    rects, labels, axes = _zn_rectangles_2d(pi)
    pts = kind === :query_overlay ? _collect_query_points(point=point, points=points) : NTuple{2,Float64}[]
    point_labels = kind === :query_overlay ? _query_overlay_labels(pi, pts) : String[]
    return _encoding_region_spec(rects, labels;
                                 title="Zn encoding",
                                 kind=kind,
                                 show_labels=(kind === :region_labels),
                                 points=pts,
                                 point_labels=point_labels,
                                 axes=axes,
                                 metadata=(; object=:zn_encoding_map, nregions=length(rects), query_count=length(pts)),
                                 fill_color=:darkorange2)
end

function _visual_spec(enc::CompiledEncoding, kind::Symbol; kwargs...)
    return _visual_spec(encoding_map(enc), kind; kwargs...)
end

function available_visuals(enc::CompiledEncoding)
    return available_visuals(encoding_map(enc))
end

function _visual_spec(res::EncodingResult, kind::Symbol; cache=:auto, kwargs...)
    _ = cache
    return _visual_spec(compile_encoding(res), kind; kwargs...)
end

available_visuals(res::EncodingResult) = available_visuals(compile_encoding(res))

function _poset_coordinates_2d(P::ProductOfChainsPoset{2})
    pts = Vector{NTuple{2,Float64}}(undef, nvertices(P))
    idx = 1
    for j in 1:P.sizes[2], i in 1:P.sizes[1]
        pts[idx] = (float(i), float(j))
        idx += 1
    end
    return pts
end

function _poset_coordinates_2d(P::GridPoset{2})
    pts = Vector{NTuple{2,Float64}}(undef, nvertices(P))
    idx = 1
    for y in P.coords[2], x in P.coords[1]
        pts[idx] = (float(x), float(y))
        idx += 1
    end
    return pts
end

function _poset_coordinates_2d(P::ProductPoset)
    n1 = nvertices(P.P1)
    pts = Vector{NTuple{2,Float64}}(undef, nvertices(P))
    for idx in 1:nvertices(P)
        i1 = ((idx - 1) % n1) + 1
        i2 = div(idx - 1, n1) + 1
        pts[idx] = (float(i1), float(i2))
    end
    return pts
end

_poset_coordinates_2d(P::AbstractPoset) = nothing

available_visuals(pi::EncodingMap) = let tgt = _poset_coordinates_2d(target_poset(pi)); src = _poset_coordinates_2d(source_poset(pi))
    if tgt === nothing
        ()
    elseif src === nothing
        (:regions, :region_labels)
    else
        (:regions, :region_labels, :pushforward_overlay)
    end
end

function _visual_spec(pi::EncodingMap, kind::Symbol; kwargs...)
    _ = kwargs
    tgt_pts = _poset_coordinates_2d(target_poset(pi))
    tgt_pts === nothing && throw(ArgumentError("Visualization v1 needs a 2D-embeddable target poset for EncodingMap visuals."))
    if kind === :regions || kind === :region_labels
        counts = zeros(Int, length(tgt_pts))
        for p in region_map(pi)
            counts[p] += 1
        end
        point_layer = PointLayer(tgt_pts, :royalblue, 0.9, 12.0)
        layers = AbstractVisualizationLayer[point_layer]
        if kind === :region_labels
            labels = [string(i, ":", counts[i]) for i in eachindex(counts)]
            push!(layers, _text_layer_from_labels(tgt_pts, labels; color=:black, textsize=10.0))
        end
        bbox = _bbox_from_points(tgt_pts)
        return VisualizationSpec(kind;
                                 title="Finite encoding map",
                                 subtitle="target-region occupancy counts",
                                 layers=layers,
                                 axes=_default_axes_2d(xlabel="target x", ylabel="target y",
                                                       xlimits=bbox === nothing ? nothing : bbox[1],
                                                       ylimits=bbox === nothing ? nothing : bbox[2],
                                                       aspect=:equal),
                                 metadata=(; object=:encoding_map, nsource=nvertices(source_poset(pi)), ntarget=nvertices(target_poset(pi))))
    elseif kind === :pushforward_overlay
        src_pts = _poset_coordinates_2d(source_poset(pi))
        src_pts === nothing && throw(ArgumentError("pushforward_overlay requires a 2D-embeddable source poset."))
        segments = NTuple{4,Float64}[]
        for (q, p) in enumerate(region_map(pi))
            s = src_pts[q]
            t = tgt_pts[p]
            push!(segments, (s[1], s[2], t[1], t[2]))
        end
        bbox = _bbox_from_points(vcat(src_pts, tgt_pts))
        return VisualizationSpec(:pushforward_overlay;
                                 title="Finite map overlay",
                                 subtitle="segments join source vertices to their targets",
                                 layers=AbstractVisualizationLayer[
                                     SegmentLayer(segments, :gray50, 0.6, 1.0),
                                     PointLayer(src_pts, :royalblue, 0.95, 10.0),
                                     PointLayer(tgt_pts, :crimson, 0.95, 10.0),
                                 ],
                                 axes=_default_axes_2d(xlabel="x", ylabel="y",
                                                       xlimits=bbox === nothing ? nothing : bbox[1],
                                                       ylimits=bbox === nothing ? nothing : bbox[2],
                                                       aspect=:equal),
                                 metadata=(; object=:encoding_map, nsource=nvertices(source_poset(pi)), ntarget=nvertices(target_poset(pi))),
                                 legend=_default_legend(visible=true, entries=(; source=:royalblue, target=:crimson, map=:gray50)))
    end
    throw(ArgumentError("Unsupported EncodingMap visualization kind $(kind)."))
end

available_visuals(::CommonRefinementTranslationResult) = (:common_refinement,)

function _visual_spec(res::CommonRefinementTranslationResult, kind::Symbol; kwargs...)
    _ = kwargs
    kind === :common_refinement || throw(ArgumentError("Unsupported kind $(kind) for CommonRefinementTranslationResult."))
    P = common_poset(res)
    pts = _poset_coordinates_2d(P)
    pts === nothing && throw(ArgumentError("common_refinement visualization requires a 2D-embeddable common poset."))
    proj = projection_maps(res)
    left_map = region_map(proj.left)
    right_map = region_map(proj.right)
    labels = [string("(", left_map[i], ",", right_map[i], ")") for i in eachindex(left_map)]
    bbox = _bbox_from_points(pts)
    return VisualizationSpec(:common_refinement;
                             title="Common refinement",
                             subtitle="common regions labeled by left/right projection indices",
                             layers=AbstractVisualizationLayer[
                                 PointLayer(pts, :purple3, 0.95, 12.0),
                                 _text_layer_from_labels(pts, labels; color=:black, textsize=9.0),
                             ],
                             axes=_default_axes_2d(xlabel="common x", ylabel="common y",
                                                   xlimits=bbox === nothing ? nothing : bbox[1],
                                                   ylimits=bbox === nothing ? nothing : bbox[2],
                                                   aspect=:equal),
                             metadata=(; object=:common_refinement, ncommon=nvertices(P), left_target=nvertices(target_poset(proj.left)), right_target=nvertices(target_poset(proj.right))))
end

available_visuals(res::ModuleTranslationResult) = begin
    map = translation_map(res)
    map isa EncodingMap && _poset_coordinates_2d(source_poset(map)) !== nothing && _poset_coordinates_2d(target_poset(map)) !== nothing ?
        (:pushforward_overlay,) : ()
end

function _visual_spec(res::ModuleTranslationResult, kind::Symbol; kwargs...)
    kind === :pushforward_overlay || throw(ArgumentError("Unsupported kind $(kind) for ModuleTranslationResult."))
    return _visual_spec(translation_map(res), :pushforward_overlay; kwargs...)
end
