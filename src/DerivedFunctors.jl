module DerivedFunctors

"""
DerivedFunctors: owner module for Hom/Ext/Tor, resolutions, functoriality,
spectral sequences, and workflow wrappers built on finite-poset modules.

This file stays thin on purpose: it owns include order and the public surface,
while implementation lives in `src/derived_functors/`.
"""

include("derived_functors/shared.jl")
include("derived_functors/utils.jl")
include("derived_functors/graded_spaces.jl")
include("derived_functors/hom_ext_engine.jl")
include("derived_functors/resolutions.jl")
include("derived_functors/ext_tor_spaces.jl")
include("derived_functors/functoriality.jl")
include("derived_functors/algebras.jl")
include("derived_functors/spectral_sequences.jl")
include("derived_functors/backends.jl")
include("derived_functors/public_api.jl")

end
