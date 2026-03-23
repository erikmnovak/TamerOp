using Test
using Random
using LinearAlgebra
using SparseArrays

# -----------------------------------------------------------------------------
# Test bootstrap
#
# These tests can be run in two modes:
#
# 1) Package mode (recommended):
#    From the repo root:
#        julia --project=. -e 'import Pkg; Pkg.test()'
#
# 2) Script mode (convenient during development):
#        julia test/runtests.jl
#
# In script mode, `using TamerOp` fails unless the repo is on LOAD_PATH.
# So we fall back to a direct include of src/TamerOp.jl.
# -----------------------------------------------------------------------------

const _TO_SRC_DIR = let here = @__DIR__
    direct = normpath(joinpath(here, "src"))
    parent = normpath(joinpath(here, "..", "src"))
    isdir(direct) ? direct : parent
end

try
    using TamerOp
catch
    include(joinpath(_TO_SRC_DIR, "TamerOp.jl"))
    using .TamerOp
end

# Keep API-surface contract checks explicit.
const TOA = TamerOp.Advanced

# Test-facing namespace:
# resolve symbols directly from owner modules to avoid coupling correctness tests
# to APISurface curation decisions.
struct _TestSurface end

const _TEST_SURFACE_MODULES = (
    TamerOp,
    TamerOp.CoreModules,
    TamerOp.Stats,
    TamerOp.Options,
    TamerOp.DataTypes,
    TamerOp.EncodingCore,
    TamerOp.Results,
    TamerOp.RegionGeometry,
    TamerOp.FiniteFringe,
    TamerOp.IndicatorTypes,
    TamerOp.Encoding,
    TamerOp.Modules,
    TamerOp.AbelianCategories,
    TamerOp.IndicatorResolutions,
    TamerOp.FlangeZn,
    TamerOp.ZnEncoding,
    TamerOp.PLPolyhedra,
    TamerOp.PLBackend,
    TamerOp.ChainComplexes,
    TamerOp.DerivedFunctors,
    TamerOp.ModuleComplexes,
    TamerOp.ChangeOfPosets,
    TamerOp.Serialization,
    TamerOp.SyntheticData,
    TamerOp.DataFileIO,
    TamerOp.InvariantCore,
    TamerOp.SignedMeasures,
    TamerOp.SliceInvariants,
    TamerOp.Fibered2D,
    TamerOp.MultiparameterImages,
    TamerOp.Visualization,
    TamerOp.Invariants,
    TamerOp.Workflow,
    TamerOp.DataIngestion,
    TamerOp.Featurizers,
)

@inline function Base.getproperty(::_TestSurface, s::Symbol)
    for m in _TEST_SURFACE_MODULES
        if isdefined(m, s)
            return getfield(m, s)
        end
    end
    throw(UndefVarError(s))
end

const TO = _TestSurface()

# Convenient aliases used throughout the test suite.
const DF  = TamerOp.DerivedFunctors
const FF  = TamerOp.FiniteFringe
const EN  = TamerOp.Encoding
const HE  = DF.HomExtEngine
const MD  = TamerOp.Modules
const IR  = TamerOp.IndicatorResolutions
const FZ  = TamerOp.FlangeZn
const SER = TamerOp.Serialization
const PLP = TamerOp.PLPolyhedra
const PLB = TamerOp.PLBackend
const CC  = TamerOp.ChainComplexes
const OPT = TamerOp.Options
const DT  = TamerOp.DataTypes
const EC  = TamerOp.EncodingCore
const RES = TamerOp.Results
const QQ  = TamerOp.CoreModules.QQ
const CM  = TamerOp.CoreModules
const IC  = TamerOp.InvariantCore
const Inv = TamerOp.Invariants
const SM  = TamerOp.SignedMeasures
const SD  = TamerOp.SyntheticData

using SparseArrays

# NOTE:
# PLBackend is now always loaded by src/TamerOp.jl, so the historical test-time
# "force include PLBackend.jl regardless of ENV toggles" hack is no longer needed.

# ---------------- Helpers used by multiple test files -------------------------

"""
    chain_poset(n::Integer; check::Bool=false) -> FF.FinitePoset

Return the chain poset on `n` elements labeled `1:n`, ordered by
`i <= j` iff `i <= j`.

Notes
-----
- This constructor is deterministic and the relation is known to be a valid
  partial order, so `check=false` is the default for speed.
- Set `check=true` if you want to force validation (useful when debugging).
- `n == 0` returns the empty poset.
"""
function chain_poset(n::Integer; check::Bool=false)::FF.FinitePoset
    n < 0 && throw(ArgumentError("chain_poset: n must be >= 0, got $n"))
    nn = Int(n)

    # BitMatrix with L[i,j] = true iff i <= j.
    L = falses(nn, nn)
    @inbounds for i in 1:nn
        for j in i:nn
            L[i, j] = true
        end
    end

    return FF.FinitePoset(L; check=check)
end

"Disjoint union of two chains of lengths m and n, with no relations across components."
function disjoint_two_chains_poset(m::Int, n::Int)
    (m >= 1 && n >= 1) || error("disjoint_two_chains_poset: need m >= 1 and n >= 1")
    N = m + n
    leq = falses(N, N)

    # First chain: 1 <= 2 <= ... <= m (store full transitive closure).
    for i in 1:m
        for j in i:m
            leq[i, j] = true
        end
    end

    # Second chain: (m+1) <= ... <= (m+n), again full transitive closure.
    off = m
    for i in 1:n
        for j in i:n
            leq[off + i, off + j] = true
        end
    end

    return FF.FinitePoset(leq)
end

"Default: two chains of length 2 (vertices {1,2} and {3,4})."
disjoint_two_chains_poset() = disjoint_two_chains_poset(2, 2)


"""
Diamond poset on {1,2,3,4} with relations

    1 < 2 < 4
    1 < 3 < 4

and with 2 incomparable to 3.

This is the smallest non-chain poset where "two different length-2 paths"
exist (1->2->4 and 1->3->4), which is exactly the situation where indicator
resolutions can have length > 1 and Ext^2 can be nonzero.
"""
function diamond_poset()
    leq = falses(4, 4)
    for i in 1:4
        leq[i, i] = true
    end
    leq[1, 2] = true
    leq[1, 3] = true
    leq[2, 4] = true
    leq[3, 4] = true
    leq[1, 4] = true  # transitive closure needed explicitly for FinitePoset
    return FF.FinitePoset(leq)
end

"""
Boolean lattice B3 on subsets of {1,2,3} ordered by inclusion.

Element numbering is by bitmask order:
1: {}
2: {1}
3: {2}
4: {3}
5: {1,2}
6: {1,3}
7: {2,3}
8: {1,2,3}
"""
function boolean_lattice_B3_poset()
    masks = Int[0, 1, 2, 4, 3, 5, 6, 7]
    n = length(masks)
    leq = falses(n, n)
    for i in 1:n
        mi = masks[i]
        for j in 1:n
            mj = masks[j]
            leq[i, j] = (mi & mj) == mi
        end
    end
    return FF.FinitePoset(leq)
end

"Convenience: 1x1 fringe module with scalar on the unique entry."
one_by_one_fringe(P::FF.AbstractPoset, U::FF.Upset, D::FF.Downset;
                  scalar=CM.QQ(1), field=CM.QQField()) =
    FF.one_by_one_fringe(P, U, D, scalar; field=field)

"Convenience: 1x1 fringe module with a specified scalar (positional)."
one_by_one_fringe(P::FF.AbstractPoset, U::FF.Upset, D::FF.Downset, scalar;
                  field=CM.QQField()) =
    FF.one_by_one_fringe(P, U, D, scalar; field=field)

"Simple modules on the chain 1 < 2: S1 supported at 1, S2 supported at 2."
function simple_modules_chain2()
    P = chain_poset(2)
    S1 = one_by_one_fringe(P, FF.principal_upset(P, 1), FF.principal_downset(P, 1))
    S2 = one_by_one_fringe(P, FF.principal_upset(P, 2), FF.principal_downset(P, 2))
    return P, S1, S2
end

# ---------------- Field test harness ----------------------------------------

const FIELD_QQ = CM.QQField()
const FIELD_F2 = CM.F2()
const FIELD_F3 = CM.F3()
const FIELD_F5 = CM.Fp(5)
const FIELD_R64 = CM.RealField(Float64; rtol=1e-10, atol=1e-12)

const FIELDS_FULL = (FIELD_QQ, FIELD_F2, FIELD_F3, FIELD_F5, FIELD_R64)

with_fields(fields, f::Function) = foreach(f, fields)
with_fields(f::Function, fields) = foreach(f, fields)

# ---------------- ASCII-only source tree test -------------------------------

@testset "ASCII-only source tree" begin
    function jl_files_under(dir::AbstractString)
        files = String[]
        for (root, _, fs) in walkdir(dir)
            for f in fs
                endswith(f, ".jl") || continue
                push!(files, joinpath(root, f))
            end
        end
        sort!(files)
        return files
    end

    function first_nonascii_byte(path::AbstractString)
        data = read(path)
        for (i, b) in enumerate(data)
            if b > 0x7f
                return (i, b)
            end
        end
        return nothing
    end

    src_guess = _TO_SRC_DIR
    src_dir  = isdir(src_guess) ? src_guess : normpath(@__DIR__)
    test_dir = normpath(@__DIR__)

    for f in vcat(jl_files_under(src_dir), jl_files_under(test_dir))
        bad = first_nonascii_byte(f)
        if bad !== nothing
            pos, byte = bad
            @info "Non-ASCII byte detected" file=f pos=pos byte=byte
        end
        @test bad === nothing
    end
end

# ---------------- Export hygiene ----------------------------------------------

@testset "No submodule export blocks" begin
    src_dir = _TO_SRC_DIR

    function jl_files_under(dir::AbstractString)
        files = String[]
        for (root, _, fs) in walkdir(dir)
            for f in fs
                endswith(f, ".jl") || continue
                push!(files, normpath(joinpath(root, f)))
            end
        end
        sort!(files)
        return files
    end

    allowed_exports = Set([normpath(joinpath(src_dir, "TamerOp.jl"))])
    pat = r"(?m)^\s*export\b"

    for f in jl_files_under(src_dir)
        f in allowed_exports && continue
        has_export = occursin(pat, read(f, String))
        if has_export
            @info "Unexpected export statement outside TamerOp.jl" file=f
        end
        @test !has_export
    end
end

# ---------------- Public API smoke test --------------------------------------

@testset "Public API smoke test" begin
    # Finite-poset primitives
    @test isdefined(TOA, :FinitePoset)
    @test isdefined(TOA, :Upset)
    @test isdefined(TOA, :Downset)
    @test isdefined(TOA, :FringeModule)
    @test isdefined(TOA, :principal_upset)
    @test isdefined(TOA, :principal_downset)
    @test isdefined(TOA, :upset_from_generators)
    @test isdefined(TOA, :downset_from_generators)
    @test isdefined(TOA, :one_by_one_fringe)
    @test isdefined(TOA, :cover_edges)

    # Encoding-map layer
    @test isdefined(TOA, :EncodingMap)
    @test isdefined(TOA, :UptightEncoding)
    @test isdefined(TOA.Encoding, :build_uptight_encoding_from_fringe)
    @test isdefined(TOA.Encoding, :pullback_fringe_along_encoding)
    @test isdefined(TOA.Encoding, :pushforward_fringe_along_encoding)

    # JSON IO helpers
    @test isdefined(TOA, :parse_finite_fringe_json)
    @test isdefined(TOA, :finite_fringe_from_m2)
    @test isdefined(TOA, :save_flange_json)
    @test isdefined(TOA, :load_flange_json)
    @test isdefined(TOA, :parse_flange_json)
    @test isdefined(TOA, :save_pl_fringe_json)
    @test isdefined(TOA, :load_pl_fringe_json)
    @test isdefined(TOA, :parse_pl_fringe_json)
    @test isdefined(TOA.Serialization, :parse_finite_fringe_json)
    @test isdefined(TOA.Serialization, :finite_fringe_from_m2)
    @test isdefined(TOA.Serialization, :save_encoding_json)
    @test isdefined(TOA.Serialization, :load_encoding_json)
    @test isdefined(TOA.Serialization, :save_mpp_decomposition_json)
    @test isdefined(TOA.Serialization, :load_mpp_decomposition_json)
    @test isdefined(TOA.Serialization, :save_mpp_image_json)
    @test isdefined(TOA.Serialization, :load_mpp_image_json)
    @test isdefined(TOA, :save_dataset_json)
    @test isdefined(TOA, :load_dataset_json)
    @test isdefined(TOA, :save_pipeline_json)
    @test isdefined(TOA, :load_pipeline_json)

    # Data ingestion entrypoints
    @test isdefined(TOA, :encode)
    @test isdefined(TOA, :hom_dimension)
    @test !isdefined(TOA, :encode_from_data)
    @test !isdefined(TOA, :ingest)
    @test isdefined(TOA, :one_criticalify)
    @test isdefined(TOA, :criticality)
    @test isdefined(TOA, :normalize_multicritical)
    @test isdefined(TOA, :fringe_presentation)
    @test isdefined(TOA, :PipelineOptions)
    @test isdefined(TOA, :DataFileOptions)
    @test isdefined(TOA, :load_data)
    @test isdefined(TOA, :inspect_data_file)
    @test isdefined(TOA, :DataIngestion)
    @test isdefined(TOA, :DataFileIO)
    @test isdefined(TOA.DataIngestion, :AbstractFiltration)
    @test isdefined(TOA.DataIngestion, :RipsFiltration)
    @test isdefined(TOA.DataIngestion, :LandmarkRipsFiltration)
    @test isdefined(TOA.DataIngestion, :GraphLowerStarFiltration)
    @test isdefined(TOA.DataIngestion, :DelaunayLowerStarFiltration)
    @test isdefined(TOA.DataIngestion, :FunctionDelaunayFiltration)
    @test isdefined(TOA.DataIngestion, :CoreFiltration)
    @test isdefined(TOA.DataIngestion, :RhomboidFiltration)
    @test isdefined(TOA.DataIngestion, :to_filtration)
    @test isdefined(TOA.DataIngestion, :estimate_ingestion)
    @test isdefined(TOA, :IngestionPlan)
    @test isdefined(TOA, :IngestionEstimate)
    @test isdefined(TOA, :GradedComplexBuildResult)
    @test isdefined(TOA, :estimate_ingestion)
    @test isdefined(TOA, :plan_ingestion)
    @test isdefined(TOA, :run_ingestion)
    @test isdefined(TOA, :check_data_filtration)
    @test isdefined(TOA, :ingestion_plan_summary)

    # Indicator-resolution and module hot-path entrypoints.
    @test isdefined(TOA, :pmodule_from_fringe)
    @test isdefined(TOA, :projective_cover)
    @test isdefined(TOA, :injective_hull)
    @test isdefined(TOA, :upset_resolution)
    @test isdefined(TOA, :downset_resolution)
    @test isdefined(TOA, :indicator_resolutions)
    @test isdefined(TOA, :verify_upset_resolution)
    @test isdefined(TOA, :verify_downset_resolution)
    @test isdefined(TOA, :map_leq)
    @test isdefined(TOA, :map_leq_many)
    @test isdefined(TOA, :map_leq_many!)
    @test isdefined(TOA, :direct_sum_with_maps)

    # Core advanced options and deeper change-of-poset hooks.
    @test isdefined(TOA, :EncodingOptions)
    @test isdefined(TOA, :ResolutionOptions)
    @test isdefined(TOA, :InvariantOptions)
    @test isdefined(TOA, :DerivedFunctorOptions)
    @test isdefined(TOA, :left_kan_extension)
    @test isdefined(TOA, :right_kan_extension)
    @test isdefined(TOA, :derived_pushforward_left)
    @test isdefined(TOA, :derived_pushforward_right)

    # Resolution tables
    @test isdefined(TOA, :betti_table)
    @test isdefined(TOA, :bass_table)
end

@testset "API surface contracts" begin
    root_exports = Set(names(TamerOp; all=false, imported=false))
    adv_exports = Set(names(TamerOp.Advanced; all=false, imported=false))

    # Root exports are strictly the curated simple surface.
    for sym in TamerOp.SIMPLE_API
        @test sym in root_exports
    end
    for sym in TamerOp.ADVANCED_ONLY_API
        @test !(sym in root_exports)
    end

    # Advanced exports the full curated power-user superset.
    for sym in TamerOp.ADVANCED_API
        @test sym in adv_exports
    end
end

# ---------------- Run test files ---------------------------------------------
# Linear algebra engine
include("test/test_invariants.jl")
include("test/test_data_pipeline.jl")
