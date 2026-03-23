module TamerOpCairoMakieExt

using CairoMakie
import TamerOp

const TO = TamerOp

include("visualization_makie_common.jl")

__register_visual_makie_backend!(TO, CairoMakie, :cairomakie; allow_save=true)

end # module TamerOpCairoMakieExt
