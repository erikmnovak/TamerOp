"""
    InvariantCore

Shared invariant-side plumbing reused across multiple invariant families.

`InvariantCore` is an owner module for backend-facing types and helpers that are
shared by more than one invariant family, but do not themselves form a coherent
mathematical subsystem for ordinary users.

Ownership map:
- `shared_types.jl` owns [`SliceSpec`](@ref) and compiled-encoding unwrapping.
- `options_helpers.jl` owns invariant-option normalization and keyword
  projection helpers.
- `packed_barcodes.jl` owns packed barcode and packed barcode-grid storage used
  by slice and fibered pipelines.
- `rank_api.jl` owns the shared [`rank_map`](@ref) entrypoints reused by
  multiple invariant families.
- `rank_cache.jl` owns [`RankQueryCache`](@ref), map-leq memo plumbing, and the
  hot-path cache helpers used by rank-style invariant queries.

This module does not own:
- invariant-family result objects,
- slice/barcode/distance algorithms,
- public workflow wrappers.

Contributor note:
- Put code here only when it is genuine shared invariant plumbing reused across
  multiple owner modules such as `Invariants`, `SliceInvariants`,
  `SignedMeasures`, `Fibered2D`, or `MultiparameterImages`.
- Keep mathematically coherent family logic in the family owner that exposes the
  public result surface.
- Prefer keeping backend helpers internal here rather than adding broad public
  wrappers or APISurface bindings.
"""
module InvariantCore

using LinearAlgebra
import Base.Threads
import Base: show, values

using ..Options: InvariantOptions
using ..EncodingCore: CompiledEncoding, locate
using ..FieldLinAlg
import ..FiniteFringe: FringeModule, nvertices, build_cache!, _preds, leq
import ..Modules: PModule, map_leq,
                 _get_cover_cache, _chosen_predecessor
import ..IndicatorResolutions: pmodule_from_fringe
import ..ZnEncoding: ZnEncodingMap
import ..ChainComplexes: describe

include("invariant_core/shared_types.jl")
include("invariant_core/options_helpers.jl")
include("invariant_core/packed_barcodes.jl")
include("invariant_core/rank_api.jl")
include("invariant_core/rank_cache.jl")

end # module InvariantCore
