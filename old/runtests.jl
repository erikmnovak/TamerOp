using Test

# Load the library either as an installed package or via local include.
# This makes "julia --project tests/runtests.jl" work even without Pkg setup.
try
    using PosetModules
catch
    include(joinpath(@__DIR__, "..", "src", "PosetModules.jl"))
    using .PosetModules
end

const PM  = PosetModules
const FF  = PM.FiniteFringe
const EN  = PM.Encoding
const HE  = PM.HomExt
const IR  = PM.IndicatorResolutions
const FZ  = PM.FlangeZn
const SER = PM.Serialization
const BR  = PM.M2SingularBridge
const CV  = PM.CrossValidateFlangePL
const PLP = PM.PLPolyhedra
const EX  = PM.ExactQQ
const CC  = PM.ChainComplexes
const QQ  = PM.CoreModules.QQ

using SparseArrays

# ---------------------------------------------------------------------
# Optional modules needed by the test suite
# ---------------------------------------------------------------------
#
# In the library, PLBackend (axis-aligned PL "boxes") is optional at load time:
# it is gated behind CoreModules.ENABLE_PL_AXIS.
#
# Several test files expect PM.PLBackend to exist, so we explicitly include it
# into the PosetModules namespace for the test suite when it was not loaded by
# default.
if !isdefined(PosetModules, :PLBackend)
    @eval PosetModules include(joinpath(@__DIR__, "..", "src", "PLBackend.jl"))
end

# ---------------- Helpers used by multiple test files -------------------------

"Chain poset on {1,...,n} with i <= j iff i <= j as integers."
function chain_poset(n::Int)
    leq = falses(n, n)
    for i in 1:n
        for j in i:n
            leq[i, j] = true
        end
    end
    return FF.FinitePoset(leq)
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

    src_dir  = normpath(joinpath(@__DIR__, "..", "src"))
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

# ---------------- Run test files ---------------------------------------------

include("test_finite_fringe.jl")
include("test_encoding.jl")

# Backends + geometry
include("test_pl_backend.jl")
include("test_zn_backend.jl")
include("test_geometry.jl")

# Homological algebra and invariants
include("test_indicator_resolutions.jl")
include("test_derived_functors.jl")
include("test_model_independent_ext_layer.jl")
include("test_chain_complexes_homology.jl")
include("test_functoriality_ext_tor_maps.jl")
include("test_invariants.jl")

# Stress tests last
include("test_random_stress.jl")