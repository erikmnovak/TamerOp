module FieldLinAlg
"""
    FieldLinAlg

Owner module for coefficient-field-aware linear algebra in TamerOp.

Scope:
- backend routing and threshold state,
- exact and floating-point kernels,
- sparse elimination utilities,
- public rank/nullspace/solve APIs.

Implementation is split across private include files in `src/field_linalg/`.
"""

# Ownership map for the private FieldLinAlg fragments.
#
# - `thresholds.jl` owns threshold state, persistence, backend traits, and
#   autotune-facing configuration knobs.
# - `autotune.jl` owns benchmark-driven threshold fitting and probe generation.
# - `backend_routing.jl` owns backend choice, conversion, and routing helpers.
# - `qq_engine.jl` owns exact rational dense/sparse kernels, modular fallbacks,
#   and QQ-specific solve/rank/nullspace implementations.
# - `nonqq_engines.jl` owns F2/F3/Fp/RealField kernels and their solve/rref
#   variants.
# - `sparse_rref.jl` owns shared sparse elimination primitives used by both QQ
#   and non-QQ engines.
# - `public_api.jl` owns the public entrypoints and restricted helper surface;
#   it does not own backend policy or kernel internals.

import Nemo
using LinearAlgebra
using SparseArrays
using Random
using TOML
using Dates
using Statistics

using ..CoreModules: AbstractCoeffField, QQField, PrimeField, RealField, FpElem,
                     BackendMatrix, coeff_type, eye, QQ,
                     _unwrap_backend_matrix, _backend_kind, _backend_payload,
                     _set_backend_payload!

include("field_linalg/thresholds.jl")
include("field_linalg/autotune.jl")
include("field_linalg/backend_routing.jl")
include("field_linalg/nonqq_engines.jl")
include("field_linalg/sparse_rref.jl")
include("field_linalg/qq_engine.jl")
include("field_linalg/public_api.jl")

end # module FieldLinAlg
