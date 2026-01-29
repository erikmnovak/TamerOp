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
# In script mode, `using PosetModules` fails unless the repo is on LOAD_PATH.
# So we fall back to a direct include of src/PosetModules.jl.
# -----------------------------------------------------------------------------

try
    using PosetModules
catch
    include(joinpath(@__DIR__, "..", "src", "PosetModules.jl"))
    using .PosetModules
end

# Tests intentionally target the "broad" surface under PosetModules.Advanced,
# because most tests exercise internal submodules and non-curated utilities.
const PM = PosetModules.Advanced


# Convenient aliases used throughout the test suite.
const DF  = PM.DerivedFunctors
const FF  = PM.FiniteFringe
const EN  = PM.Encoding
const HE  = DF.HomExtEngine
const MD  = PM.Modules
const IR  = PM.IndicatorResolutions
const FZ  = PM.FlangeZn
const SER = PM.Serialization
const PLP = PM.PLPolyhedra
const PLB = PM.PLBackend
const EX  = PM.ExactQQ
const CC  = PM.ChainComplexes
const QQ  = PM.CoreModules.QQ
const CM  = PM.CoreModules

using SparseArrays

# NOTE:
# PLBackend is now always loaded by src/PosetModules.jl, so the historical test-time
# "force include PLBackend.jl regardless of ENV toggles" hack is no longer needed.

# ---------------- Helpers used by multiple test files -------------------------

"""
    chain_poset(n::Integer; check::Bool=false) -> FinitePoset

Return the chain poset on `n` elements labeled `1:n`, ordered by
`i <= j` iff `i <= j`.

Notes
-----
- This constructor is deterministic and the relation is known to be a valid
  partial order, so `check=false` is the default for speed.
- Set `check=true` if you want to force validation (useful when debugging).
- `n == 0` returns the empty poset.
"""
function chain_poset(n::Integer; check::Bool=false)::FinitePoset
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
one_by_one_fringe(P::FF.FinitePoset, U::FF.Upset, D::FF.Downset; scalar=QQ(1)) =
    FF.one_by_one_fringe(P, U, D; scalar=scalar)

"Convenience: 1x1 fringe module with a specified scalar (positional)."
one_by_one_fringe(P::FF.FinitePoset, U::FF.Upset, D::FF.Downset, scalar) =
    FF.one_by_one_fringe(P, U, D, scalar)

"Simple modules on the chain 1 < 2: S1 supported at 1, S2 supported at 2."
function simple_modules_chain2()
    P = chain_poset(2)
    S1 = one_by_one_fringe(P, FF.principal_upset(P, 1), FF.principal_downset(P, 1))
    S2 = one_by_one_fringe(P, FF.principal_upset(P, 2), FF.principal_downset(P, 2))
    return P, S1, S2
end

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

    src_guess = normpath(joinpath(@__DIR__, "..", "src"))
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

# ---------------- Public API smoke test --------------------------------------

@testset "Public API smoke test" begin
    # Finite-poset primitives
    @test isdefined(PM, :FinitePoset)
    @test isdefined(PM, :Upset)
    @test isdefined(PM, :Downset)
    @test isdefined(PM, :FringeModule)
    @test isdefined(PM, :principal_upset)
    @test isdefined(PM, :principal_downset)
    @test isdefined(PM, :upset_from_generators)
    @test isdefined(PM, :downset_from_generators)
    @test isdefined(PM, :one_by_one_fringe)
    @test isdefined(PM, :cover_edges)

    # Encoding-map layer
    @test isdefined(PM, :EncodingMap)
    @test isdefined(PM, :UptightEncoding)
    @test isdefined(PM, :build_uptight_encoding_from_fringe)
    @test isdefined(PM, :pullback_fringe_along_encoding)
    @test isdefined(PM, :pushforward_fringe_along_encoding)

    # JSON IO helpers
    @test isdefined(PM, :save_flange_json)
    @test isdefined(PM, :load_flange_json)
    @test isdefined(PM, :save_encoding_json)
    @test isdefined(PM, :load_encoding_json)
    @test isdefined(PM, :save_mpp_decomposition_json)
    @test isdefined(PM, :load_mpp_decomposition_json)
    @test isdefined(PM, :save_mpp_image_json)
    @test isdefined(PM, :load_mpp_image_json)

    # Resolution tables
    @test isdefined(PM, :betti_table)
    @test isdefined(PM, :bass_table)
end

# ---------------- Run test files ---------------------------------------------

include("test_finite_fringe.jl")
include("test_encoding.jl")

# Backends + geometry
include("test_pl_backend.jl")
include("test_zn_backend.jl")
include("test_geometry.jl")

# Homological algebra
include("test_indicator_resolutions.jl")
include("test_derived_functors.jl")
include("test_model_independent_ext_layer.jl")
include("test_chain_complexes_homology.jl")
include("test_functoriality_ext_tor_maps.jl")

# Invariants
include("test_invariants.jl")

# Stress tests last
include("test_random_stress.jl")