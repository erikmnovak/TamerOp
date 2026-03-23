# Validation helpers for visualization requests and backend-agnostic specs.

@inline function _visual_issue_report(kind::Symbol, valid::Bool; kwargs...)
    return (; kind, valid, kwargs...)
end

function _throw_invalid_visual(kind::Symbol, issues::Vector{String})
    msg = isempty(issues) ? "invalid visualization request." : join(issues, " ")
    throw(ArgumentError(string(kind, ": ", msg)))
end

function check_visual_spec(spec::VisualizationSpec; throw::Bool=false)
    issues = String[]
    String(spec.kind) == "" && push!(issues, "kind must be a nonempty symbol")
    isempty(spec.layers) && isempty(spec.panels) &&
        push!(issues, "spec must contain at least one layer or one panel")
    !isempty(spec.layers) && !isempty(spec.panels) &&
        push!(issues, "spec cannot mix direct layers and panels")
    haskey(spec.axes, :xlabel) || push!(issues, "axes must define :xlabel")
    haskey(spec.axes, :ylabel) || push!(issues, "axes must define :ylabel")
    haskey(spec.axes, :xlimits) || push!(issues, "axes must define :xlimits")
    haskey(spec.axes, :ylimits) || push!(issues, "axes must define :ylimits")

    for (idx, layer) in enumerate(spec.layers)
        if layer isa HeatmapLayer
            (size(layer.values, 2) == length(layer.x) && size(layer.values, 1) == length(layer.y)) ||
                push!(issues, "HeatmapLayer $idx values shape must match x/y lengths.")
        elseif layer isa RectLayer
            for rect in layer.rects
                rect[1] <= rect[3] || push!(issues, "RectLayer $idx has xlo > xhi.")
                rect[2] <= rect[4] || push!(issues, "RectLayer $idx has ylo > yhi.")
            end
        elseif layer isa TextLayer
            length(layer.labels) == length(layer.positions) ||
                push!(issues, "TextLayer $idx labels/positions length mismatch.")
        elseif layer isa BarcodeLayer
            length(layer.intervals) == length(layer.multiplicities) ||
                push!(issues, "BarcodeLayer $idx intervals/multiplicities length mismatch.")
        elseif layer isa PointLayer
            layer.color isa AbstractVector && length(layer.color) != length(layer.points) &&
                push!(issues, "PointLayer $idx color_values length must match the number of points.")
        elseif layer isa Point3Layer
            isempty(layer.points) || all(length(p) == 3 for p in layer.points) ||
                push!(issues, "Point3Layer $idx points must be 3-vectors.")
            layer.color isa AbstractVector && length(layer.color) != length(layer.points) &&
                push!(issues, "Point3Layer $idx color_values length must match the number of points.")
        elseif layer isa Segment3Layer
            all(seg -> length(seg) == 6, layer.segments) ||
                push!(issues, "Segment3Layer $idx segments must store (x1,y1,z1,x2,y2,z2).")
        end
    end

    for (idx, panel) in enumerate(spec.panels)
        isempty(panel.panels) || push!(issues, "panel $idx cannot contain nested panels.")
        panel_report = check_visual_spec(panel; throw=false)
        get(panel_report, :valid, false) || begin
            for issue in get(panel_report, :issues, String[])
                push!(issues, "panel $idx: " * issue)
            end
        end
    end

    valid = isempty(issues)
    throw && !valid && _throw_invalid_visual(:check_visual_spec, issues)
    return _visual_issue_report(:visual_spec, valid;
                                visual_kind=visual_kind(spec),
                                nlayers=length(spec.layers),
                                npanels=length(spec.panels),
                                issues=issues)
end

function check_visual_request(obj; kind::Symbol=:auto, throw::Bool=false, kwargs...)
    supported = available_visuals(obj)
    issues = String[]
    isempty(supported) && push!(issues, "no visualization kinds are registered for $(nameof(typeof(obj))).")
    requested = kind === :auto ? (isempty(supported) ? :auto : supported[1]) : kind
    if kind !== :auto && !(kind in supported)
        push!(issues, "kind=$(kind) is unsupported for $(nameof(typeof(obj))); supported kinds are $(supported).")
    end
    _append_visual_request_issues!(issues, obj, requested; kwargs...)
    valid = isempty(issues)
    throw && !valid && _throw_invalid_visual(:check_visual_request, issues)
    return _visual_issue_report(:visual_request, valid;
                                object_type=Symbol(nameof(typeof(obj))),
                                requested_kind=requested,
                                supported_kinds=supported,
                                issues=issues)
end

_append_visual_request_issues!(issues::Vector{String}, obj, kind::Symbol; kwargs...) = issues

function _require_one_of!(issues::Vector{String}, names::Tuple, kwargs::NamedTuple, context::AbstractString)
    any(name -> get(kwargs, name, nothing) !== nothing, names) ||
        push!(issues, context * " requires one of " * join(string.(names), ", ") * ".")
    return issues
end

function _require_all!(issues::Vector{String}, names::Tuple, kwargs::NamedTuple, context::AbstractString)
    for name in names
        get(kwargs, name, nothing) !== nothing || push!(issues, context * " requires keyword " * string(name) * ".")
    end
    return issues
end

function _append_visual_request_issues!(issues::Vector{String}, obj::Union{AbstractPLikeEncodingMap,CompiledEncoding,EncodingResult}, kind::Symbol; kwargs...)
    params = (; kwargs...)
    if kind === :query_overlay
        _require_one_of!(issues, (:point, :points), params, "query_overlay")
    end
    return issues
end

function _append_visual_request_issues!(issues::Vector{String}, obj::DataTypes.PointCloud, kind::Symbol; kwargs...)
    params = (; kwargs...)
    if kind === :knn_graph
        k = get(params, :k, nothing)
        k === nothing || Int(k) > 0 || push!(issues, "knn_graph k must be positive.")
    elseif kind === :radius_graph
        radius = get(params, :radius, nothing)
        radius === nothing || float(radius) > 0 || push!(issues, "radius_graph radius must be positive.")
    end
    return issues
end

function _append_visual_request_issues!(issues::Vector{String}, obj::DataIngestion.PointCodensityResult, kind::Symbol; kwargs...)
    params = (; kwargs...)
    if kind === :codensity_radius_snapshots
        DataTypes.ambient_dim(DataIngestion.source_data(obj)) == 2 ||
            push!(issues, "codensity_radius_snapshots requires a 2-dimensional point cloud.")
        radii = get(params, :radii, nothing)
        if radii !== nothing
            try
                isempty(radii) &&
                    push!(issues, "codensity_radius_snapshots requires at least one radius.")
                all(float(r) > 0.0 for r in radii) ||
                    push!(issues, "codensity_radius_snapshots radii must be positive.")
            catch err
                push!(issues, sprint(showerror, err))
            end
        end
        levels = get(params, :codensity_levels, nothing)
        if levels isa Symbol
            levels === :quantiles ||
                push!(issues, "codensity_radius_snapshots codensity_levels must be :quantiles or an explicit vector of cutoffs.")
        elseif levels !== nothing
            try
                isempty(levels) &&
                    push!(issues, "codensity_radius_snapshots requires at least one codensity cutoff.")
                all(isfinite(float(v)) for v in levels) ||
                    push!(issues, "codensity_radius_snapshots cutoffs must be finite real values.")
            catch err
                push!(issues, sprint(showerror, err))
            end
        end
        return issues
    end
    return _append_visual_request_issues!(issues, DataIngestion.source_data(obj), kind; kwargs...)
end

function _append_visual_request_issues!(issues::Vector{String}, obj::Union{FiberedArrangement2D,FiberedBarcodeCache2D}, kind::Symbol; kwargs...)
    params = (; kwargs...)
    if kind in (:fibered_query, :fibered_cell_highlight, :fibered_tie_break, :fibered_query_barcode)
        _require_all!(issues, (:dir,), params, string(kind))
        _require_one_of!(issues, (:offset, :basepoint), params, string(kind))
        if kind === :fibered_tie_break && isempty(issues)
            try
                arg = get(params, :offset, nothing)
                arg === nothing && (arg = get(params, :basepoint, nothing))
                report = fibered_query_summary(obj, params.dir, arg; tie_break=get(params, :tie_break, :up))
                report.valid || append!(issues, String.(report.issues))
                report.valid && !report.tie_break_relevant &&
                    push!(issues, "fibered_tie_break requires a boundary query where cell_up != cell_down.")
            catch err
                push!(issues, sprint(showerror, err))
            end
        end
    elseif kind === :fibered_offset_intervals
        _require_all!(issues, (:dir,), params, string(kind))
    elseif kind === :fibered_projected_comparison
        _require_all!(issues, (:projected,), params, string(kind))
        projected = get(params, :projected, nothing)
        projected isa ProjectedArrangement || push!(issues, "fibered_projected_comparison requires projected to be a ProjectedArrangement.")
    elseif kind in (:fibered_family_contributions, :fibered_distance_diagnostic)
        _require_all!(issues, (:caches,), params, string(kind))
        if isempty(issues)
            try
                _resolve_fibered_caches(obj, params.caches)
            catch err
                push!(issues, sprint(showerror, err))
            end
        end
    end
    return issues
end

function _append_visual_request_issues!(issues::Vector{String}, obj::FiberedSliceFamily2D, kind::Symbol; kwargs...)
    params = (; kwargs...)
    if kind in (:fibered_family_contributions, :fibered_distance_diagnostic)
        _require_all!(issues, (:caches,), params, string(kind))
        if isempty(issues)
            try
                _resolve_fibered_caches(obj, params.caches)
            catch err
                push!(issues, sprint(showerror, err))
            end
        end
    end
    return issues
end

function _append_visual_request_issues!(issues::Vector{String}, obj::FiberedSliceResult, kind::Symbol; kwargs...)
    params = (; kwargs...)
    if kind === :fibered_slice_overlay
        _require_all!(issues, (:arrangement, :dir), params, string(kind))
        _require_one_of!(issues, (:offset, :basepoint), params, string(kind))
        arr = get(params, :arrangement, nothing)
        arr isa FiberedArrangement2D || push!(issues, "fibered_slice_overlay requires arrangement to be a FiberedArrangement2D.")
    end
    return issues
end

function _append_visual_request_issues!(issues::Vector{String}, obj::SliceBarcodesResult, kind::Symbol; kwargs...)
    params = (; kwargs...)
    if kind === :barcode
        bars = slice_barcodes(obj)
        count = length(bars)
        count == 1 || get(params, :index, nothing) !== nothing ||
            push!(issues, "barcode view on a SliceBarcodesResult with multiple barcodes requires keyword index.")
    end
    return issues
end

function _append_visual_request_issues!(issues::Vector{String}, obj::MPLandscape, kind::Symbol; kwargs...)
    params = (; kwargs...)
    if kind === :landscape_slices
        get(params, :idir, nothing) !== nothing || push!(issues, "landscape_slices requires keyword idir.")
        get(params, :ioff, nothing) !== nothing || push!(issues, "landscape_slices requires keyword ioff.")
    end
    return issues
end

function _append_visual_request_issues!(issues::Vector{String}, obj::MPPDecomposition, kind::Symbol; kwargs...)
    params = (; kwargs...)
    if kind === :mpp_decomposition
        layout = get(params, :layout, :overlay)
        layout in (:overlay, :summands) ||
            push!(issues, "mpp_decomposition layout must be :overlay or :summands.")
    end
    return issues
end

function _append_visual_request_issues!(issues::Vector{String}, obj::InvariantResult, kind::Symbol; kwargs...)
    params = (; kwargs...)
    if kind === :rank_query_overlay
        _require_one_of!(issues, (:pair, :pairs), params, "rank_query_overlay")
    end
    return issues
end

function _append_visual_request_issues!(issues::Vector{String}, obj::ModuleTranslationResult, kind::Symbol; kwargs...)
    if kind === :pushforward_overlay
        map = translation_map(obj)
        map isa EncodingMap || push!(issues, "pushforward_overlay currently requires translation_map(res) to be an EncodingMap.")
    end
    return issues
end
