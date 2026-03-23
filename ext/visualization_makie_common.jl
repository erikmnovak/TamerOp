function __register_visual_makie_backend!(TO, MakieMod, backend::Symbol; allow_save::Bool=true)
    Viz = TO.Visualization

    Point2 = if isdefined(MakieMod, :Point2f)
        MakieMod.Point2f
    elseif isdefined(MakieMod, :Point2f0)
        MakieMod.Point2f0
    else
        nothing
    end

    to_point2(p) = Point2 === nothing ? p : Point2(float(p[1]), float(p[2]))

    _uses_axis3(spec) = any(layer -> layer isa Viz.Point3Layer || layer isa Viz.Segment3Layer,
                            Viz.visual_layers(spec))

    function _legend_entries(spec)
        legend = Viz.visual_legend(spec)
        entries = get(legend, :entries, NamedTuple())
        entries isa AbstractVector && return entries
        entries isa Tuple && return collect(entries)
        if entries isa NamedTuple
            isempty(keys(entries)) && return NamedTuple[]
            out = NamedTuple[]
            for (label, entry) in pairs(entries)
                if entry isa NamedTuple
                    push!(out, merge((; label=String(label)), entry))
                else
                    push!(out, (; label=String(label), color=entry, style=:patch))
                end
            end
            return out
        end
        return NamedTuple[]
    end

    function _colorbar_layers(layers)
        return [layer for layer in layers if layer isa Viz.HeatmapLayer && layer.show_colorbar]
    end

    function _apply_axis_limits!(ax, spec)
        spec.axes.xlimits === nothing || MakieMod.xlims!(ax, spec.axes.xlimits...)
        spec.axes.ylimits === nothing || MakieMod.ylims!(ax, spec.axes.ylimits...)
        if _uses_axis3(spec)
            get(spec.axes, :zlimits, nothing) === nothing || MakieMod.zlims!(ax, spec.axes.zlimits...)
        end
        get(spec.axes, :xticks, nothing) === nothing || (ax.xticks = spec.axes.xticks)
        get(spec.axes, :yticks, nothing) === nothing || (ax.yticks = spec.axes.yticks)
        spec.axes.aspect === :equal && (ax.aspect = MakieMod.DataAspect())
        spec.axes.aspect === :data && (ax.aspect = MakieMod.DataAspect())
        return ax
    end

    function _render_text_panel!(fig, grid, spec)
        lines = String[]
        for layer in Viz.visual_layers(spec)
            layer isa Viz.TextLayer || continue
            append!(lines, layer.labels)
        end
        title_row = 1
        if !isempty(spec.title)
            MakieMod.Label(grid[title_row, 1], spec.title;
                           fontsize=18, tellwidth=false, halign=:left)
            title_row += 1
        end
        if !isempty(spec.subtitle)
            MakieMod.Label(grid[title_row, 1], spec.subtitle;
                           fontsize=12, tellwidth=false, halign=:left)
            title_row += 1
        end
        for (i, line) in enumerate(lines)
            MakieMod.Label(grid[title_row + i - 1, 1], line;
                           fontsize=13, tellwidth=false, halign=:left)
        end
        return nothing
    end

    function _draw_layers!(ax, spec)
        colorbars = NamedTuple[]
        for layer in Viz.visual_layers(spec)
            if layer isa Viz.HeatmapLayer
                hm = MakieMod.heatmap!(ax, layer.x, layer.y, layer.values;
                                       colormap=layer.colormap,
                                       alpha=layer.alpha)
                layer.show_colorbar && push!(colorbars, (; plot=hm, label=layer.colorbar_label))
            elseif layer isa Viz.RectLayer
                for rect in layer.rects
                    poly = [to_point2((rect[1], rect[2])), to_point2((rect[3], rect[2])),
                            to_point2((rect[3], rect[4])), to_point2((rect[1], rect[4]))]
                    MakieMod.poly!(ax, poly;
                                   color=(layer.fill_color, layer.alpha),
                                   strokecolor=layer.stroke_color,
                                   strokewidth=layer.linewidth)
                end
            elseif layer isa Viz.SegmentLayer
                for seg in layer.segments
                    MakieMod.lines!(ax, [seg[1], seg[3]], [seg[2], seg[4]];
                                    color=(layer.color, layer.alpha), linewidth=layer.linewidth)
                end
            elseif layer isa Viz.Segment3Layer
                for seg in layer.segments
                    MakieMod.lines!(ax, [seg[1], seg[4]], [seg[2], seg[5]], [seg[3], seg[6]];
                                    color=(layer.color, layer.alpha), linewidth=layer.linewidth)
                end
            elseif layer isa Viz.PolylineLayer
                for path in layer.paths
                    xs = [p[1] for p in path]
                    ys = [p[2] for p in path]
                    MakieMod.lines!(ax, xs, ys; color=(layer.color, layer.alpha), linewidth=layer.linewidth)
                    if layer.closed && !isempty(path)
                        MakieMod.lines!(ax, [path[end][1], path[1][1]], [path[end][2], path[1][2]];
                                        color=(layer.color, layer.alpha), linewidth=layer.linewidth)
                    end
                end
            elseif layer isa Viz.PointLayer
                isempty(layer.points) || begin
                    scatter_color = layer.color isa AbstractVector ? layer.color : (layer.color, layer.alpha)
                    kwargs = layer.color isa AbstractVector ?
                             (; color=layer.color, colormap=layer.colormap, alpha=layer.alpha, markersize=layer.markersize) :
                             (; color=(layer.color, layer.alpha), markersize=layer.markersize)
                    MakieMod.scatter!(ax,
                                      [p[1] for p in layer.points],
                                      [p[2] for p in layer.points];
                                      kwargs...)
                end
            elseif layer isa Viz.Point3Layer
                isempty(layer.points) || begin
                    kwargs = layer.color isa AbstractVector ?
                             (; color=layer.color, colormap=layer.colormap, alpha=layer.alpha, markersize=layer.markersize) :
                             (; color=(layer.color, layer.alpha), markersize=layer.markersize)
                    MakieMod.scatter!(ax,
                                      [p[1] for p in layer.points],
                                      [p[2] for p in layer.points],
                                      [p[3] for p in layer.points];
                                      kwargs...)
                end
            elseif layer isa Viz.TextLayer
                for (lbl, pos) in zip(layer.labels, layer.positions)
                    MakieMod.text!(ax, lbl; position=(pos[1], pos[2]), color=layer.color, fontsize=layer.textsize)
                end
            elseif layer isa Viz.BarcodeLayer
                y = layer.ystart
                for (iv, mult) in zip(layer.intervals, layer.multiplicities)
                    for _ in 1:max(mult, 1)
                        MakieMod.lines!(ax, [iv[1], iv[2]], [y, y];
                                        color=layer.color, linewidth=layer.linewidth)
                        y += layer.ystep
                    end
                end
            else
                error("Unsupported visualization layer $(typeof(layer)) for Makie rendering.")
            end
        end
        _apply_axis_limits!(ax, spec)
        return colorbars
    end

    function _render_legend!(fig, slot, spec)
        legend = Viz.visual_legend(spec)
        get(legend, :visible, false) || return nothing
        entries = _legend_entries(spec)
        isempty(entries) && return nothing
        elements = Any[]
        labels = String[]
        for entry in entries
            style = get(entry, :style, :patch)
            color = get(entry, :color, :black)
            if style === :line && isdefined(MakieMod, :LineElement)
                push!(elements, MakieMod.LineElement(color=color))
            elseif style === :marker && isdefined(MakieMod, :MarkerElement)
                push!(elements, MakieMod.MarkerElement(color=color, marker=:circle))
            elseif isdefined(MakieMod, :PolyElement)
                push!(elements, MakieMod.PolyElement(color=color))
            elseif isdefined(MakieMod, :MarkerElement)
                push!(elements, MakieMod.MarkerElement(color=color, marker=:circle))
            else
                continue
            end
            push!(labels, String(get(entry, :label, string(style))))
        end
        isempty(elements) && return nothing
        title = get(legend, :title, "")
        return MakieMod.Legend(slot, elements, labels; title=title)
    end

    function _make_axis(figslot, spec)
        if _uses_axis3(spec)
            return MakieMod.Axis3(figslot;
                                  xlabel=spec.axes.xlabel,
                                  ylabel=spec.axes.ylabel,
                                  zlabel=get(spec.axes, :zlabel, "z"),
                                  title=spec.title)
        end
        return MakieMod.Axis(figslot;
                             xlabel=spec.axes.xlabel,
                             ylabel=spec.axes.ylabel,
                             title=spec.title,
                             subtitle=spec.subtitle)
    end

    function _render_spec_into_grid!(fig, grid, spec)
        if get(Viz.visual_metadata(spec), :panel_style, nothing) === :text_only
            return _render_text_panel!(fig, grid, spec)
        end
        axis_slot = grid[1, 1]
        ax = _make_axis(axis_slot, spec)
        colorbars = _draw_layers!(ax, spec)
        offset = 2
        for cb in colorbars
            MakieMod.Colorbar(grid[1, offset], cb.plot; label=cb.label)
            offset += 1
        end
        legend_position = get(Viz.visual_metadata(spec), :legend_position, :bottom)
        if legend_position === :right
            _render_legend!(fig, grid[1, offset], spec)
        elseif legend_position !== :none
            _render_legend!(fig, grid[2, 1], spec)
        end
        return ax
    end

    function _slice_from_volume(volume, view_dims::Tuple{Int,Int}, fixed::Dict{Int,Int})
        idx = Any[Colon() for _ in 1:ndims(volume)]
        for (dim, pos) in fixed
            idx[dim] = Int(pos)
        end
        idx[view_dims[1]] = Colon()
        idx[view_dims[2]] = Colon()
        slice = Float64.(volume[idx...])
        ndims(slice) == 2 && return slice
        return reshape(slice, size(volume, view_dims[1]), size(volume, view_dims[2]))
    end

    function _render_slice_viewer(spec; figure=nothing)
        volume = get(spec.metadata, :volume, nothing)
        volume === nothing && return nothing
        view_dims = Tuple(get(spec.metadata, :view_dims, (1, 2)))
        fixed = Dict{Int,Int}(get(spec.metadata, :fixed_indices, Dict{Int,Int}()))
        control_dims = sort!(collect(setdiff(1:ndims(volume), collect(view_dims))))
        isempty(control_dims) && return nothing
        fig = figure === nothing ? MakieMod.Figure() : figure
        ax = MakieMod.Axis(fig[1, 1];
                           xlabel=spec.axes.xlabel,
                           ylabel=spec.axes.ylabel,
                           title=spec.title,
                           subtitle=spec.subtitle)
        current = copy(fixed)
        zobs = MakieMod.Observable(_slice_from_volume(volume, view_dims, current))
        hm = MakieMod.heatmap!(ax,
                               1:size(zobs[], 2),
                               1:size(zobs[], 1),
                               zobs;
                               colormap=get(spec.metadata, :colormap, :magma),
                               alpha=1.0)
        _apply_axis_limits!(ax, spec)
        MakieMod.Colorbar(fig[1, 2], hm; label="intensity")
        for (row, dim) in enumerate(control_dims)
            MakieMod.Label(fig[row + 1, 1], "slice dim $dim"; tellwidth=false)
            slider = MakieMod.Slider(fig[row + 1, 2];
                                     range=1:size(volume, dim),
                                     startvalue=get(current, dim, cld(size(volume, dim), 2)))
            MakieMod.Label(fig[row + 1, 3], MakieMod.lift(v -> string(Int(round(v))), slider.value))
            MakieMod.on(slider.value) do v
                current[dim] = Int(round(v))
                zobs[] = _slice_from_volume(volume, view_dims, current)
            end
        end
        return fig
    end

    function render_spec(spec; display::Symbol=:inline, figure=nothing, kwargs...)
        spec isa Viz.VisualizationSpec || throw(ArgumentError("render_spec expected a VisualizationSpec, got $(typeof(spec))."))
        _ = (display, kwargs)
        get(Viz.visual_interaction(spec), :notebook, :summary_card) === :widget_viewer && begin
            widget_fig = _render_slice_viewer(spec; figure=figure)
            widget_fig === nothing || return widget_fig
        end
        panels = Viz.visual_panels(spec)
        ncols = isempty(panels) ? 1 : Int(clamp(get(Viz.visual_metadata(spec), :panel_columns, min(3, length(panels))), 1, max(length(panels), 1)))
        nrows = isempty(panels) ? 1 : cld(length(panels), ncols)
        fig = if figure === nothing
            fig_size = get(Viz.visual_metadata(spec), :figure_size, nothing)
            fig_size === nothing ? MakieMod.Figure() : MakieMod.Figure(size=fig_size)
        else
            figure
        end
        if isempty(panels)
            grid = fig[1, 1] = MakieMod.GridLayout()
            _render_spec_into_grid!(fig, grid, spec)
            return fig
        end

        first_panel_row = 1
        if !isempty(spec.title)
            MakieMod.Label(fig[first_panel_row, 1:ncols], spec.title; fontsize=22, tellwidth=false)
            first_panel_row += 1
        end
        if !isempty(spec.subtitle)
            MakieMod.Label(fig[first_panel_row, 1:ncols], spec.subtitle; fontsize=14, tellwidth=false)
            first_panel_row += 1
        end
        for (idx, panel) in enumerate(panels)
            row = first_panel_row + div(idx - 1, ncols)
            col = 1 + mod(idx - 1, ncols)
            grid = fig[row, col] = MakieMod.GridLayout()
            _render_spec_into_grid!(fig, grid, panel)
        end
        return fig
    end

    save_spec = if allow_save
        function(path::AbstractString, spec; kwargs...)
            spec isa Viz.VisualizationSpec || throw(ArgumentError("save_spec expected a VisualizationSpec, got $(typeof(spec))."))
            fig = render_spec(spec; kwargs...)
            MakieMod.save(path, fig)
            return path
        end
    else
        nothing
    end

    return Viz._register_visual_backend!(backend; render=render_spec, save=save_spec)
end
