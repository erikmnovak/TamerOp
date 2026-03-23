module TamerOpWGLMakieExt

using WGLMakie
import TamerOp

const TO = TamerOp

include("visualization_makie_common.jl")

__register_visual_makie_backend!(TO, WGLMakie, :wglmakie; allow_save=false)

const VIZ = TO.Visualization
const _BASE_WGL_RENDER = VIZ._VISUAL_RENDERERS[:wglmakie]

function _save_wgl_visual(path::AbstractString, spec; kwargs...)
    spec isa VIZ.VisualizationSpec || throw(ArgumentError("save_spec expected a VisualizationSpec, got $(typeof(spec))."))
    WGLMakie.activate!(; use_html_widgets=true)
    fig = Base.invokelatest(_BASE_WGL_RENDER, spec; kwargs...)
    ext = lowercase(splitext(path)[2])
    if ext == ".html"
        app = WGLMakie.Bonito.App(fig)
        WGLMakie.Bonito.export_static(path, app)
    else
        WGLMakie.save(path, fig)
    end
    return path
end

VIZ._register_visual_backend!(:wglmakie;
                              render=(spec; kwargs...) -> begin
                                  WGLMakie.activate!(; use_html_widgets=true)
                                  Base.invokelatest(_BASE_WGL_RENDER, spec; kwargs...)
                              end,
                              save=_save_wgl_visual)

end # module TamerOpWGLMakieExt
