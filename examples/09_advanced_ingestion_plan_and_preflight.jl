# =============================================================================
# Example 09: Advanced ingestion planning and preflight
#
# Theme
# -----
# "Estimate first, inspect the normalized plan second, then choose whether to
# stop at the graded-complex stage or run all the way to an encoding result."
# =============================================================================

include(joinpath(@__DIR__, "_common.jl"))

example_header(
    "09",
    "Advanced ingestion planning and preflight";
    theme="Use typed filtrations, preflight estimates, and staged execution before paying for the full encoding pipeline.",
    teaches=[
        "How to inspect a typed ingestion filtration before execution",
        "How estimate_ingestion(...) and plan_ingestion(...) expose cheap-first planning data",
        "How to stop at :graded_complex before running to :encoding_result",
    ],
)

outdir = example_outdir("09_advanced_ingestion_plan_and_preflight")

const DI = TO.DataIngestion
const CM = TO.CoreModules
const EC = TO.EncodingCore

stage("1) Build typed data and a typed filtration")

data = TO.PointCloud([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.2, 0.9],
    [1.1, 0.8],
])

construction = TO.ConstructionOptions(;
    sparsify=:knn,
    collapse=:none,
    output_stage=:graded_complex,
    budget=(max_simplices=64, max_edges=24, memory_budget_bytes=10_000_000),
)

filtration = DI.RipsDensityFiltration(;
    max_dim=1,
    knn=2,
    density_k=2,
    construction=construction,
)

println("Filtration summary: ", DI.filtration_summary(filtration))
println("Filtration validator: ", DI.check_filtration(filtration; throw=false))

stage("2) Estimate ingestion cost before building anything heavy")

est = DI.estimate_ingestion(data, filtration)

println("Estimate summary: ", DI.ingestion_estimate_summary(est))
println("Estimated cells: ", DI.estimated_cells(est))
println("Estimated axis sizes: ", DI.estimated_axis_sizes(est))
println("Estimate warnings: ", DI.estimate_warnings(est))

stage("3) Build and inspect a normalized ingestion plan")

plan = DI.plan_ingestion(
    data,
    filtration;
    field=CM.QQField(),
    cache=:auto,
    preflight=true,
)

println("Plan summary: ", DI.ingestion_plan_summary(plan))
println("Plan validator: ", DI.check_ingestion_plan(plan; throw=false))
println("Route hint: ", DI.route_hint(plan))
println("Preflight attached: ", DI.has_preflight(plan))

stage("4) Stop at the graded-complex stage")

G = DI.run_ingestion(plan; stage=:graded_complex)

println("Graded complex summary: ", TO.describe(G))

buildres = DI.build_graded_complex(data, filtration)
println("Direct build summary: ", TO.describe(buildres))
println("Build axis sizes: ", ntuple(i -> length(DI.grade_axes(buildres)[i]), length(DI.grade_axes(buildres))))

stage("5) Run the same plan to the final encoding result")

enc = DI.run_ingestion(plan; stage=:encoding_result)

println("Encoding summary: ", TO.result_summary(enc))
println("Encoding axes: ", EC.axes_from_encoding(enc.pi))

stage("6) Save a compact report")

report_path = joinpath(outdir, "ingestion_plan_report.txt")
open(report_path, "w") do io
    println(io, "example=09_advanced_ingestion_plan_and_preflight")
    println(io, "filtration_summary=", DI.filtration_summary(filtration))
    println(io, "estimate_summary=", DI.ingestion_estimate_summary(est))
    println(io, "plan_summary=", DI.ingestion_plan_summary(plan))
    println(io, "graded_complex_summary=", TO.describe(G))
    println(io, "build_result_summary=", TO.describe(buildres))
    println(io, "encoding_summary=", TO.result_summary(enc))
end

println("Saved report: ", report_path)
println("\nDone. Output directory: ", outdir)
