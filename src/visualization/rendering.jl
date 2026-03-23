# Renderer registry and backend selection. Concrete Makie code lives in ext/.

const _VISUAL_RENDERERS = Dict{Symbol,Function}()
const _VISUAL_SAVERS = Dict{Symbol,Function}()
const _VISUAL_SPEC_HTML_BACKEND = :spec_html
const _VISUAL_BACKEND_HELP = Dict{Symbol,String}(
    :cairomakie => "Install CairoMakie.jl for deterministic static rendering/export.",
    :wglmakie => "Install WGLMakie.jl for notebook-friendly interactive rendering.",
)

function _register_visual_backend!(backend::Symbol; render::Function, save::Union{Nothing,Function}=nothing)
    _VISUAL_RENDERERS[backend] = render
    save === nothing || (_VISUAL_SAVERS[backend] = save)
    return backend
end

@inline _visual_backend_available(backend::Symbol) = haskey(_VISUAL_RENDERERS, backend)
@inline _visual_save_available(backend::Symbol) = haskey(_VISUAL_SAVERS, backend)

function _in_notebook_context()
    return isdefined(Main, :IJulia) || isdefined(Main, :PlutoRunner) ||
           haskey(ENV, "JPY_PARENT_PID") || haskey(ENV, "COLAB_RELEASE_TAG")
end

function _try_load_visual_backend!(backend::Symbol)
    _visual_backend_available(backend) && return true
    root = parentmodule(@__MODULE__)
    spec = if backend === :cairomakie
        (; dep=:CairoMakie, extmod=:TamerOpCairoMakieExt, extfile="TamerOpCairoMakieExt.jl")
    elseif backend === :wglmakie
        (; dep=:WGLMakie, extmod=:TamerOpWGLMakieExt, extfile="TamerOpWGLMakieExt.jl")
    else
        return false
    end
    isdefined(root, spec.extmod) && return true
    try
        Base.require(Main, spec.dep)
    catch
        return false
    end
    isdefined(root, spec.extmod) && return true
    path = normpath(joinpath(@__DIR__, "..", "..", "ext", spec.extfile))
    isfile(path) || return false
    try
        Base.include(root, path)
    catch
        return false
    end
    return _visual_backend_available(backend)
end

function _ensure_any_visual_backend!()
    _try_load_visual_backend!(:wglmakie)
    _try_load_visual_backend!(:cairomakie)
    return nothing
end

function _resolve_visual_backend(requested::Symbol; for_save::Bool=false, display::Symbol=:inline)
    requested === :auto || begin
        _try_load_visual_backend!(requested)
        if for_save
            _visual_save_available(requested) || throw(ArgumentError("No save-capable visualization backend registered for $(requested). $(get(_VISUAL_BACKEND_HELP, requested, ""))"))
        else
            _visual_backend_available(requested) || throw(ArgumentError("No visualization backend registered for $(requested). $(get(_VISUAL_BACKEND_HELP, requested, ""))"))
        end
        return requested
    end

    _ensure_any_visual_backend!()
    if for_save
        if _visual_save_available(:cairomakie)
            return :cairomakie
        elseif _visual_save_available(:wglmakie)
            return :wglmakie
        end
    else
        if display === :inline && _in_notebook_context() && _visual_backend_available(:wglmakie)
            return :wglmakie
        elseif _visual_backend_available(:cairomakie)
            return :cairomakie
        elseif _visual_backend_available(:wglmakie)
            return :wglmakie
        end
    end

    msg = string(
        "No visualization renderer is available. Install CairoMakie.jl for static figures ",
        "or WGLMakie.jl for notebook rendering. `visual_spec(...)` remains available without a renderer."
    )
    throw(ArgumentError(msg))
end

@inline function _normalize_visual_export_preference(prefer)
    prefer isa Symbol || throw(ArgumentError("`prefer` must be :static or :interactive, got $(typeof(prefer))."))
    prefer in (:static, :interactive) ||
        throw(ArgumentError("`prefer` must be :static or :interactive, got $(prefer)."))
    return prefer
end

@inline function _normalize_visual_export_format(format)
    format isa Symbol || throw(ArgumentError("`format` must be :auto or a file-format symbol, got $(typeof(format))."))
    format === :auto && return format
    isempty(String(format)) && throw(ArgumentError("`format` must be :auto or a nonempty file-format symbol."))
    return format
end

function _check_visual_export_stem(stem::AbstractString)
    isempty(stem) && throw(ArgumentError("`stem` must be a nonempty basename without an extension."))
    basename(stem) == stem || throw(ArgumentError("`stem` must be a basename without directory separators, got $(repr(stem))."))
    isempty(splitext(stem)[2]) || throw(ArgumentError("`stem` must omit the file extension; pass `format=...` instead."))
    return String(stem)
end

@inline _visual_default_format_for_backend(backend::Symbol) = backend === :wglmakie ? :html : :png

function _visual_export_format_from_path(path::AbstractString)
    ext = lowercase(splitext(path)[2])
    isempty(ext) && throw(ArgumentError("`save_visual(path, ...)` requires a filename with an extension."))
    return Symbol(ext[2:end])
end

function _choose_visual_export_target(outdir::AbstractString,
                                      stem::AbstractString;
                                      prefer::Symbol=:static,
                                      format::Symbol=:auto,
                                      backend::Symbol=:auto)
    prefer = _normalize_visual_export_preference(prefer)
    format = _normalize_visual_export_format(format)
    stem_s = _check_visual_export_stem(stem)
    mkpath(outdir)

    if backend !== :auto
        chosen_format = format === :auto ? _visual_default_format_for_backend(backend) : format
        return (; path=joinpath(outdir, stem_s * "." * String(chosen_format)),
                backend=backend,
                format=chosen_format,
                stem=stem_s)
    end

    _ensure_any_visual_backend!()
    if format !== :auto
        chosen_backend = if format === :html
            _visual_save_available(:wglmakie) ? :wglmakie : _VISUAL_SPEC_HTML_BACKEND
        else
            _resolve_visual_backend(:auto; for_save=true)
        end
        return (; path=joinpath(outdir, stem_s * "." * String(format)),
                backend=chosen_backend,
                format=format,
                stem=stem_s)
    end

    chosen_backend, chosen_format = if prefer === :static
        if _visual_save_available(:cairomakie)
            (:cairomakie, :png)
        elseif _visual_save_available(:wglmakie)
            (:wglmakie, :html)
        else
            (_VISUAL_SPEC_HTML_BACKEND, :html)
        end
    else
        if _visual_save_available(:wglmakie)
            (:wglmakie, :html)
        elseif _visual_save_available(:cairomakie)
            (:cairomakie, :png)
        else
            (_VISUAL_SPEC_HTML_BACKEND, :html)
        end
    end

    return (; path=joinpath(outdir, stem_s * "." * String(chosen_format)),
            backend=chosen_backend,
            format=chosen_format,
            stem=stem_s)
end

available_visuals(spec::VisualizationSpec) = (visual_kind(spec),)

function render(spec::VisualizationSpec; backend::Symbol=:auto, display::Symbol=:inline, kwargs...)
    report = check_visual_spec(spec; throw=true)
    _ = report
    chosen = _resolve_visual_backend(backend; for_save=false, display=display)
    return Base.invokelatest(_VISUAL_RENDERERS[chosen], spec; display=display, kwargs...)
end

function visualize(obj; kind::Symbol=:auto, backend::Symbol=:auto, display::Symbol=:inline,
                   cache=:auto, kwargs...)
    spec = obj isa VisualizationSpec ? obj : visual_spec(obj; kind=kind, cache=cache, kwargs...)
    return render(spec; backend=backend, display=display, kwargs...)
end

function _write_visual_html(path::AbstractString, spec::VisualizationSpec)
    open(path, "w") do io
        show(io, MIME("text/html"), spec)
    end
    return path
end

function _save_visual_internal(path::AbstractString, spec::VisualizationSpec; backend::Symbol=:auto, kwargs...)
    report = check_visual_spec(spec; throw=true)
    _ = report
    format = _visual_export_format_from_path(path)
    if format === :html
        if backend !== :cairomakie
            _try_load_visual_backend!(:wglmakie)
            if _visual_save_available(:wglmakie)
                saved_path = Base.invokelatest(_VISUAL_SAVERS[:wglmakie], path, spec; kwargs...)
                return (; path=String(saved_path), backend=:wglmakie, format=:html)
            end
        end
        saved_path = _write_visual_html(path, spec)
        return (; path=String(saved_path), backend=_VISUAL_SPEC_HTML_BACKEND, format=:html)
    end
    chosen = _resolve_visual_backend(backend; for_save=true)
    saved_path = Base.invokelatest(_VISUAL_SAVERS[chosen], path, spec; kwargs...)
    return (; path=String(saved_path), backend=chosen, format=format)
end

"""
    save_visual(outdir, stem, obj; kind=:auto, prefer=:static, format=:auto,
                backend=:auto, cache=:auto, kwargs...)
    save_visual(path, obj; kind=:auto, backend=:auto, cache=:auto, kwargs...)

Save a visualization to disk.

The canonical notebook-facing form is `save_visual(outdir, stem, obj; ...)`.
It chooses a concrete export target for you: by default it prefers a static PNG
through CairoMakie, falls back to interactive HTML through WGLMakie, and then
falls back to spec HTML when no renderer is installed.

Use the path-based form only when you want exact control over the filename,
extension, or backend. That advanced form preserves the older contract and
returns the written path string.
"""
save_visual

function save_visual(path::AbstractString, spec::VisualizationSpec; backend::Symbol=:auto, kwargs...)
    return _save_visual_internal(path, spec; backend=backend, kwargs...).path
end

function save_visual(path::AbstractString, obj; kind::Symbol=:auto, backend::Symbol=:auto, cache=:auto, kwargs...)
    spec = obj isa VisualizationSpec ? obj : visual_spec(obj; kind=kind, cache=cache, kwargs...)
    return save_visual(path, spec; backend=backend, kwargs...)
end

function save_visual(outdir::AbstractString,
                     stem::AbstractString,
                     spec::VisualizationSpec;
                     prefer::Symbol=:static,
                     format::Symbol=:auto,
                     backend::Symbol=:auto,
                     kwargs...)
    target = _choose_visual_export_target(outdir, stem; prefer=prefer, format=format, backend=backend)
    saved = _save_visual_internal(target.path, spec; backend=target.backend, kwargs...)
    return VisualExportResult(saved.path, saved.backend, saved.format, visual_kind(spec), target.stem)
end

function save_visual(outdir::AbstractString,
                     stem::AbstractString,
                     obj;
                     kind::Symbol=:auto,
                     prefer::Symbol=:static,
                     format::Symbol=:auto,
                     backend::Symbol=:auto,
                     cache=:auto,
                     kwargs...)
    spec = obj isa VisualizationSpec ? obj : visual_spec(obj; kind=kind, cache=cache, kwargs...)
    return save_visual(outdir, stem, spec; prefer=prefer, format=format, backend=backend, kwargs...)
end

@inline _visual_request_missing(field::Symbol) = throw(ArgumentError("Each visualization export request must include `$(field)`."))

function _save_visual_request(outdir::AbstractString,
                              request::NamedTuple;
                              prefer::Symbol=:static,
                              format::Symbol=:auto,
                              backend::Symbol=:auto,
                              cache=:auto)
    hasproperty(request, :stem) || _visual_request_missing(:stem)
    hasproperty(request, :obj) || _visual_request_missing(:obj)
    stem = getproperty(request, :stem)
    obj = getproperty(request, :obj)
    overrides = Base.structdiff(request, (; stem=nothing, obj=nothing))
    opts = merge((; prefer=prefer, format=format, backend=backend, cache=cache), overrides)
    return save_visual(outdir, stem, obj; opts...)
end

"""
    save_visuals(outdir, requests; prefer=:static, format=:auto,
                 backend=:auto, cache=:auto)

Save a batch of visualization requests into `outdir`.

Each request is a named tuple with at least `stem` and `obj`. Any additional
keys are forwarded as keyword arguments to `save_visual(outdir, stem, obj; ...)`.
This keeps notebook code focused on the mathematical objects being exported
instead of manual backend/extension bookkeeping.
"""
function save_visuals(outdir::AbstractString,
                      requests::AbstractVector{<:NamedTuple};
                      prefer::Symbol=:static,
                      format::Symbol=:auto,
                      backend::Symbol=:auto,
                      cache=:auto)
    return [
        _save_visual_request(outdir, request; prefer=prefer, format=format, backend=backend, cache=cache)
        for request in requests
    ]
end
