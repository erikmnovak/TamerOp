"""
    FiniteFringe

Finite-poset and finite-fringe algebra for the `TamerOp` package.

# Mathematical model

`FiniteFringe` owns the canonical implementation of the following closely related
objects and computations:

- finite posets represented by explicit order data,
- upsets and downsets inside an ambient finite poset,
- fringe modules presented by upset generators, downset relations, and a
  coefficient matrix,
- fiber dimensions `dim_k M_q` at vertices `q` of the ambient poset,
- dimensions of morphism spaces `dim_k Hom(M, N)` between fringe modules.

The intended model is the standard finite-fringe presentation:

- a finite poset `P`,
- a list of birth upsets `U_1, ..., U_n`,
- a list of death downsets `D_1, ..., D_m`,
- a matrix `phi` of size `m x n` over a ground field `k`,

subject to the usual support condition that a nonzero entry `phi[j,i]` is only
allowed when `U_i  intersect  D_j` is nonempty.

# Canonical user entrypoints

For ordinary user-facing work in this subsystem, the canonical entrypoints are:

- `FinitePoset(...)`
- `principal_upset`, `principal_downset`
- `upset_closure`, `downset_closure`
- `upset_from_generators`, `downset_from_generators`
- `FringeModule{K}(...)`
- `one_by_one_fringe(...)`
- `fiber_dimension(M, q)`
- `hom_dimension(M, N)`

These are the mathematically meaningful entrypoints. Users should not need to
inspect internal cache or plan objects for ordinary work.

# Support-closure contract

The finite-fringe API makes one strict distinction:

- `principal_upset`, `principal_downset`, `upset_closure`, `downset_closure`,
  `upset_from_generators`, and `downset_from_generators` *construct closed
  supports for you*.
- `Upset`, `Downset`, `FringeModule{K}(...)`, and `one_by_one_fringe(...)`
  *expect already-closed supports*.

In other words, generator-based helper functions perform closure, while the
presentation-level constructors take mathematically finalized supports as input.

# Module ownership

`FiniteFringe` owns the hot finite-poset / upset-downset / fringe-module /
fiber / `Hom` subsystem.

The following *do not* belong to this owner anymore:

- `TamerOp.IndicatorTypes`: indicator presentation/container types,
- `TamerOp.Encoding`: finite-fringe encoding and push/pull helpers.

Those are sibling owner modules, not nested submodules of `FiniteFringe`.
"""
module FiniteFringe

using SparseArrays, LinearAlgebra
using ..CoreModules: QQ, QQField, AbstractCoeffField, coeff_type, field_from_eltype, coerce
using ..Options: FiniteFringeOptions
import ..CoreModules: change_field
import ..FieldLinAlg

# Private implementation files. FiniteFringe remains the owner module; these
# fragments separate posets, fringe-module/fiber code, and Hom kernels for readability.
include("finite_fringe/posets.jl")
include("finite_fringe/sets_and_types.jl")
include("finite_fringe/fringe_module_and_fiber.jl")
include("finite_fringe/hom.jl")

end # module
