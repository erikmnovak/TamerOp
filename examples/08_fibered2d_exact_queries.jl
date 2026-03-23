# =============================================================================
# Example 08: Exact 2D fibered queries and projected comparisons
#
# Theme
# -----
# "Build one exact 2D arrangement, inspect it first, then reuse it for exact
# slice queries and projected comparisons."
# =============================================================================

include(joinpath(@__DIR__, "_common.jl"))

example_header(
    "08",
    "Exact 2D fibered queries";
    theme="Inspect exact arrangement/cache objects before reusing them across repeated slice queries.",
    teaches=[
        "How to build and inspect a FiberedArrangement2D",
        "How FiberedSliceResult and fibered_query_summary expose one exact query",
        "How the projected workflow compares to the exact fibered workflow",
    ],
)

outdir = example_outdir("08_fibered2d_exact_queries")

const Inv = TO.Invariants
const TOA = TO.Advanced
const PLB = TO.PLBackend
const FF = TO.FiniteFringe
const IR = TO.IndicatorResolutions
const CM = TO.CoreModules

stage("1) Build a tiny 2D encoding and two modules on it")

Ups = [
    PLB.BoxUpset([0.0, -10.0]),
    PLB.BoxUpset([1.0, -10.0]),
]
Downs = PLB.BoxDownset[]
P, _, pi = PLB.encode_fringe_boxes(Ups, Downs, TOA.EncodingOptions())

r_left = TOA.locate(pi, [0.5, 0.0])
r_right = TOA.locate(pi, [2.0, 0.0])
field = CM.QQField()

M_left = IR.pmodule_from_fringe(
    FF.one_by_one_fringe(
        P,
        FF.principal_upset(P, r_left),
        FF.principal_downset(P, r_right),
        1;
        field=field,
    ),
)
M_right = IR.pmodule_from_fringe(
    FF.one_by_one_fringe(
        P,
        FF.principal_upset(P, r_right),
        FF.principal_downset(P, r_right),
        1;
        field=field,
    ),
)

println("Poset size: ", P.n)
println("Located regions: left=", r_left, ", right=", r_right)

stage("2) Build one exact fibered arrangement and inspect it")

opts = TOA.InvariantOptions(box=([-1.0, -1.0], [2.0, 1.0]), strict=true)
arr = Inv.fibered_arrangement_2d(pi, opts; normalize_dirs=:L1, include_axes=true, precompute=:cells)

println("Arrangement summary: ", Inv.fibered_arrangement_summary(arr))
println("Arrangement validator: ", Inv.check_fibered_arrangement_2d(arr; throw=false))

stage("3) Build caches and inspect one exact slice query")

cache_left = Inv.fibered_barcode_cache_2d(M_left, arr; precompute=:none)
cache_right = Inv.fibered_barcode_cache_2d(M_right, arr; precompute=:none)

println("Left cache summary:  ", Inv.fibered_cache_summary(cache_left))
println("Right cache summary: ", Inv.fibered_cache_summary(cache_right))

slice_res = Inv.fibered_slice(cache_left, (1.0, 1.0), 0.0)
query_res = Inv.fibered_query_summary(cache_left, (1.0, 1.0), 0.0)

println("Slice summary: ", TO.describe(slice_res))
println("Query summary: ", query_res)
println("Query barcode: ", Inv.slice_barcode(slice_res))

stage("4) Build a reusable slice family and compute one exact distance")

fam = Inv.fibered_slice_family_2d(arr; direction_weight=:lesnick_l1, store_values=true)
d_exact = Inv.matching_distance_exact_2d(cache_left, cache_right; family=fam, threads=false)

println("Family summary: ", Inv.fibered_family_summary(fam))
println("Exact matching distance: ", d_exact)

stage("5) Compare with the projected workflow")

parr = Inv.projected_arrangement(pi; dirs=[(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)], threads=false)
proj_left = Inv.projected_barcode_cache(M_left, parr; precompute=true)
proj_right = Inv.projected_barcode_cache(M_right, parr; precompute=true)

pb = Inv.projected_barcodes(proj_left; threads=false)
pd = Inv.projected_distances(proj_left, proj_right; dist=:bottleneck, threads=false)
d_proj = Inv.projected_distance(proj_left, proj_right; dist=:bottleneck, agg=:mean, threads=false)

println("Projected arrangement summary: ", Inv.projected_arrangement_summary(parr))
println("Projected cache summary: ", Inv.projected_cache_summary(proj_left))
println("Projected barcode batch summary: ", TO.describe(pb))
println("Projected distance batch summary: ", TO.describe(pd))
println("Projected mean distance: ", d_proj)

stage("6) Save a compact report")

report_path = joinpath(outdir, "fibered2d_exact_report.txt")
open(report_path, "w") do io
    println(io, "example=08_fibered2d_exact_queries")
    println(io, "arrangement_summary=", Inv.fibered_arrangement_summary(arr))
    println(io, "cache_summary=", Inv.fibered_cache_summary(cache_left))
    println(io, "slice_summary=", TO.describe(slice_res))
    println(io, "query_summary=", query_res)
    println(io, "family_summary=", Inv.fibered_family_summary(fam))
    println(io, "exact_matching_distance=", d_exact)
    println(io, "projected_arrangement_summary=", Inv.projected_arrangement_summary(parr))
    println(io, "projected_cache_summary=", Inv.projected_cache_summary(proj_left))
    println(io, "projected_barcode_batch_summary=", TO.describe(pb))
    println(io, "projected_distance_batch_summary=", TO.describe(pd))
    println(io, "projected_mean_distance=", d_proj)
end

println("Saved report: ", report_path)
println("\nDone. Output directory: ", outdir)
