try
    using TamerOp
catch
    include(joinpath(@__DIR__, "..", "src", "TamerOp.jl"))
    using .TamerOp
end
const TO = TamerOp
const FZ = TO.FlangeZn

# -----------------------------------------------------------------------------
# 1) INPUT (presentation layer): two toy Z^2 fringe presentations (flanges)
#
# We build two "rectangle indicator" modules:
#   M = 1 on [0,2] x [0,2], else 0
#   N = 1 on [1,3] x [1,3], else 0
#
# For a rectangle indicator, you can use:
#   - one flat at the lower-left corner (principal upset),
#   - one injective at the upper-right corner (principal downset),
#   - Phi = [1] as the single map from flat -> injective.
# (This pattern appears in the Zn backend tests.)
# -----------------------------------------------------------------------------

tau = FZ.face(2, [])  # "vertex face": no free coordinates

F_M = FZ.IndFlat(tau, [0, 0]; id=:F_M)
E_M = FZ.IndInj(tau, [2, 2]; id=:E_M)
Phi_M = reshape([TO.QQ(1)], 1, 1)
FM = FZ.Flange(2, [F_M], [E_M], Phi_M)

F_N = FZ.IndFlat(tau, [1, 1]; id=:F_N)
E_N = FZ.IndInj(tau, [3, 3]; id=:E_N)
Phi_N = reshape([TO.QQ(1)], 1, 1)
FN = FZ.Flange(2, [F_N], [E_N], Phi_N)

# Sanity check directly on the flange presentation (before encoding).
@assert FZ.dim_at(FM, [0, 0]) == 1
@assert FZ.dim_at(FM, [3, 3]) == 0

# -----------------------------------------------------------------------------
# 2) ENCODE (finite encoding layer): encode both modules on a COMMON poset P
#
# This is the key Workflow convenience: encode(A, B; ...) common-encodes and
# returns two EncodingResults sharing the same P and classifier pi.
# -----------------------------------------------------------------------------

enc_opts = TO.EncodingOptions(backend=:zn, max_regions=50_000, poset_kind=:signature)
encM, encN = TO.encode(FM, FN; opts=enc_opts)

P  = TO.poset(encM)
pi = TO.classifier(encM)   # shared classifier (encM.pi === encN.pi)
M  = TO.pmodule(encM)
N  = TO.pmodule(encN)

# Point query example: map a Z^2-grade g to a vertex q in P, then read dim.
# (You can do this kind of query anywhere downstream without re-deriving geometry.)
q11 = TO.locate(pi, [1, 1])
println("dim M(1,1) = ", TO.dim_at(M, q11))
println("dim N(1,1) = ", TO.dim_at(N, q11))

# -----------------------------------------------------------------------------
# 3) (Optional) COARSEN (single-encoding speed hack)
#
# coarsen(enc) merges regions/labels for a single module. This is good for
# single-module invariants, but do not coarsen two modules independently and
# then try to compare them: the encodings may no longer share (P, pi).
# -----------------------------------------------------------------------------

encM_small = TO.coarsen(encM; method=:uptight)

# -----------------------------------------------------------------------------
# 4) INVARIANTS (curated stable wrappers from Workflow)
# -----------------------------------------------------------------------------

ri_M = TO.rank_invariant(encM_small)     # rank invariant object (dictionary-like)
surf_M = TO.euler_surface(encM_small)    # Euler characteristic surface as an array
img_M = TO.mpp_image(encM_small; sigma=0.35, resolution=32)

println("Euler surface array size: ", size(surf_M))
println("MPP image pixel grid: ", size(img_M.img))

# -----------------------------------------------------------------------------
# 5) DISTANCES between TWO modules (needs common encoding)
#
# matching_distance checks that encM and encN share (P, pi) and then calls the
# slice-based approximation by default. You can control sampling with n_dirs,
# n_offsets, etc.
# -----------------------------------------------------------------------------

dmatch = TO.matching_distance(encM, encN;
                              method=:approx,
                              n_dirs=24,
                              n_offsets=12,
                              max_den=8)

println("Approx matching distance(M, N) = ", dmatch)

# -----------------------------------------------------------------------------
# 6) DERIVED FUNCTORS (Ext as a graded space with a standard interface)
#
# ext(encM, encN; maxdeg=k) returns a graded-vector-space-like object.
# Use degree_range/dim/basis, and optionally representative/coordinates.
# -----------------------------------------------------------------------------

E = TO.ext(encM, encN; maxdeg=2)

println("Ext degrees available: ", TO.degree_range(E))
for t in TO.degree_range(E)
    println("dim Ext^", t, "(M,N) = ", TO.dim(E, t))
end

# Demonstrate representative/coordinates interface in degree 1, if nonzero.
t = 1
d1 = TO.dim(E, t)
if d1 > 0
    # Coordinate vector for the first basis class:
    c = zeros(TO.QQ, d1)
    c[1] = TO.QQ(1)

    # Turn coordinates into a cocycle representative, then recover coordinates.
    z = TO.representative(E, t, c)
    c_back = TO.coordinates(E, t, z)

    println("Coordinates round-trip ok: ", c_back == c)
end

# -----------------------------------------------------------------------------
# 7) RESOLUTION (projective, with Betti + minimality report)
# -----------------------------------------------------------------------------

res_opts = TO.ResolutionOptions(maxlen=3, minimal=true, check=true)
res = TO.resolve(encM; kind=:projective, opts=res_opts, minimality=true)

println("Betti data:")
println(TO.betti(res))
println("Is minimal (reported)? ", TO.is_minimal(res))
