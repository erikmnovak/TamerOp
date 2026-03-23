# =============================================================================
# Example 01: hand-built GradedComplex
#
# What this teaches
# -----------------
# 1) How to build a tiny graded complex directly (cells, boundary, grades).
# 2) How explicit axes control the encoded poset shape.
# 3) How to compute core summaries and export one feature row.
#
# This is the algebra-first example: no geometry ingestion, no hidden builders.
#
# Terminology note:
# - We start from a `GradedComplex`: each cell carries a multi-grade (here 2D).
# - "Graded" means indexed by filtration parameters (not by homological degree).
# - After `encode(...; degree=t)`, the output module `enc.M` is the
#   cohomology module H^t, graded by the finite poset induced from those
#   filtration parameters/axes.
# =============================================================================

include(joinpath(@__DIR__, "_common.jl"))

outdir = example_outdir("01_graded_complex")
println("\n[01] Hand-built GradedComplex -> encoded module -> feature row")

# -----------------------------------------------------------------------------
# 1) Build a tiny 1D complex with 2-parameter grades
# -----------------------------------------------------------------------------

# Cell ids by dimension:
# - dim 0: vertices 1,2,3
# - dim 1: edges 1,2
cells_by_dim = [Int[1, 2, 3], Int[1, 2]]

# Boundary d1: C1 -> C0 (3x2).
# Edge 1 = v1 - v2, edge 2 = v2 - v3.
# (Signs matter over QQ/Real; over F2 they reduce mod 2.)
B1 = sparse(
    [1, 2, 2, 3],
    [1, 1, 2, 2],
    [1, -1, 1, -1],
    3,
    2,
)

# One grade per cell in dim-order [v1, v2, v3, e1, e2].
grades = [
    [0.0, 0.0],
    [0.6, 0.0],
    [1.2, 0.0],
    [0.6, 0.4],
    [1.2, 0.8],
]

G = TO.GradedComplex(cells_by_dim, [B1], grades)

# -----------------------------------------------------------------------------
# 2) Encode with explicit axes
# -----------------------------------------------------------------------------

axes = (
    # Axis 1 samples the first grade coordinate.
    collect(range(0.0, stop=1.6, length=5)),
    # Axis 2 samples the second grade coordinate.
    collect(range(0.0, stop=1.2, length=4)),
)

spec = TO.FiltrationSpec(
    ;
    # `:graded` means "ingest this object as an already graded complex".
    kind=:graded,
    axes=axes,
)
println("spec type: ", typeof(spec))

# Same degree convention as quickstart:
# degree=t asks ingestion to return the module H^t.
# We choose F2 for fast, deterministic linear algebra in this tutorial.
enc = TO.encode(G, spec; degree=0, field=TO.CoreModules.F2(), cache=:auto)
println("enc type: ", typeof(enc))
println("enc.M type (module): ", typeof(enc.M))
println("enc.pi type (classifier): ", typeof(enc.pi))
println("Encoded poset vertices: ", TO.nvertices(enc.P))

# -----------------------------------------------------------------------------
# 3) Invariant sanity checks
# -----------------------------------------------------------------------------

opts = TO.InvariantOptions(
    ;
    # Keep axes fixed exactly as provided above (no auto-derived axis changes).
    axes=axes,
    axes_policy=:as_given,
    max_axis_len=64,
    threads=false,
    strict=true,
    pl_mode=:fast,
)

rh = TO.restricted_hilbert(enc.M)
rank_tbl = TO.rank_invariant(enc; opts=opts, store_zeros=true)

println("restricted_hilbert length: ", length(rh))
println("rank table entries: ", length(rank_tbl))

# -----------------------------------------------------------------------------
# 4) Feature extraction
# -----------------------------------------------------------------------------

# Use rank grid here so output dimension is explicit and easy to explain.
# For n vertices, RankGridSpec produces n^2 coordinates in deterministic order.
rank_spec = TO.RankGridSpec(nvertices=TO.nvertices(enc.P), store_zeros=true, threads=false)

# `batch_transform` is used even for one sample so this file mirrors dataset workflows.
fs = TO.batch_transform(
    [enc],
    rank_spec;
    opts=opts,
    batch=TO.BatchOptions(threaded=false, backend=:threads, progress=false),
    cache=:auto,
    idfun=_ -> "graded_hand_001",
)

println("nfeatures(rank_spec): ", TO.nfeatures(rank_spec))
println("feature matrix shape: ", size(fs.X))

# -----------------------------------------------------------------------------
# 5) Save outputs
# -----------------------------------------------------------------------------

paths = save_feature_bundle(outdir, "graded_hand", fs)
println("CSV wide: ", paths.csv_wide)
println("CSV long: ", paths.csv_long)
println("Native optional outputs: ", paths.native)

g_path = joinpath(outdir, "graded_complex.json")
TO.save_dataset_json(g_path, G)
println("Saved graded complex JSON: ", g_path)

println("\nDone. Output directory: ", outdir)
