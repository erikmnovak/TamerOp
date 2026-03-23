# File: Serialization.jl

"""
TamerOp.Serialization

All JSON-facing I/O lives here.

Separation of concerns
----------------------
A) Internal formats (owned/stable):
   - `save_*_json` / `load_*_json`
   - Schemas are controlled by TamerOp. Loaders are intentionally strict.
   - Cheap-first UX: `inspect_json`, `*_json_summary`, and `check_*_json`
     should be the normal first stop before a full load.

B) External adapters (CAS ingestion):
   - `parse_*_json` / `*_from_*`
   - Strict parsers for canonical exchange schemas emitted by external tools.
     These adapters are explicit contracts, not compatibility shims.
   - These are not TamerOp-owned storage formats; they should be documented
     and reasoned about separately from the owned `save_*_json` / `load_*_json`
     families.
   - External adapters do not participate in the owned cheap-first JSON UX
     (`inspect_json` / `*_json_summary` / `check_*_json`) unless the artifact
     itself is a TamerOp-owned format.

C) Invariant caches (MPPI):
   - `save_mpp_*_json` / `load_mpp_*_json`
   - Convenience cache formats for expensive derived objects defined in `TamerOp.Invariants`.

Private fragment layout
-----------------------
1) `serialization/shared.jl`
2) `serialization/owned_datasets.jl`
3) `serialization/owned_encodings.jl`
4) `serialization/external_interop.jl`
5) `serialization/external_cas.jl`
6) `serialization/owned_mppi.jl`

Keep the owner file thin and place subsystem logic in the private fragments.
"""
module Serialization

using JSON3
using SparseArrays

import ..CoreModules
using ..CoreModules: QQ, AbstractCoeffField, QQField, RealField, PrimeField,
    coeff_type, coerce, FpElem, rational_to_string, string_to_rational
import ..FlangeZn: Face, IndFlat, IndInj, Flange, canonical_matrix
import ..FiniteFringe: AbstractPoset, FinitePoset, ProductOfChainsPoset, GridPoset, ProductPoset,
                       FringeModule, nvertices, leq_matrix
import ..ZnEncoding: SignaturePoset, PackedSignatureRows
using ..FiniteFringe
using ..Modules: PModule, _clear_cover_cache!
using ..DataTypes: PointCloud, ImageNd, GraphData, EmbeddedPlanarGraph2D, GradedComplex,
                   MultiCriticalGradedComplex, SimplexTreeMulti, point_matrix, coord_matrix,
                   edge_columns
using ..Options: FiltrationSpec, ConstructionBudget, ConstructionOptions, PipelineOptions, EncodingOptions
using ..EncodingCore: GridEncodingMap, CompiledEncoding
using ..Results: EncodingResult
import ..Results: materialize_module
import ..ChainComplexes: describe
import ..ZnEncoding
import ..PLPolyhedra
import ..PLBackend
import ..IndicatorResolutions: pmodule_from_fringe, fringe_presentation

const PIPELINE_SCHEMA_VERSION = 2
const ENCODING_SCHEMA_VERSION = 1
const PLFRINGE_SCHEMA_VERSION = 1
const TAMER_FEATURE_SCHEMA_VERSION = v"0.2.0"

include("serialization/shared.jl")
include("serialization/owned_datasets.jl")
include("serialization/owned_encodings.jl")
include("serialization/external_interop.jl")
include("serialization/external_cas.jl")
include("serialization/owned_mppi.jl")

end # module Serialization
