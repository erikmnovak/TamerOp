# =============================================================================
# Example 02: Point-cloud bifiltration with Rips-density
#
# Theme
# -----
# "Multipersistence from practical point-cloud data, end-to-end on a dataset."
#
# We build two synthetic classes (noisy circles vs noisy figure-eights), encode
# each sample with a sparse Rips-density filtration, and export batch features.
# =============================================================================

include(joinpath(@__DIR__, "_common.jl"))

example_header(
    "02",
    "Point-cloud bifiltration with :rips_density";
    theme="Dataset workflow: many samples -> encodings -> one feature table.",
    teaches=[
        "How to set sparse construction + budget guardrails",
        "How to run deterministic batch featurization",
        "How to emit wide/long feature tables for downstream statistics",
    ],
)

Random.seed!(20260217)
outdir = example_outdir("02_point_cloud")

stage("1) Build a reproducible two-class point-cloud dataset")

clouds, labels = make_pointcloud_dataset(n_per_class=10, n_points=64, seed=2002)
println("Dataset size: ", length(clouds), " point clouds")
println("Class counts: circle=", count(==("circle"), labels), ", figure8=", count(==("figure8"), labels))

stage("2) Define filtration and encode each sample")

construction = TO.ConstructionOptions(
    ;
    sparsify=:knn,
    collapse=:none,
    budget=(max_simplices=140_000, max_edges=90_000, memory_budget_bytes=1_000_000_000),
)

filtration = TO.RipsDensityFiltration(
    ;
    max_dim=1,
    knn=10,
    density_k=10,
    nn_backend=:auto,
    construction=construction,
)
field = TO.CoreModules.F2()

encodings = Vector{TO.EncodingResult}(undef, length(clouds))
for i in eachindex(clouds)
    encodings[i] = TO.encode(clouds[i], filtration; degree=0, field=field, cache=:auto)
end
println("Encoded ", length(encodings), " samples.")

samples = to_encoding_samples(encodings, labels; prefix="pc")

stage("3) Build featurizer specs")

dirs = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]]
offs = [
    collect(range(-0.6, stop=0.6, length=5)),
    collect(range(-0.6, stop=0.6, length=5)),
    collect(range(-0.8, stop=0.8, length=5)),
    collect(range(-0.8, stop=0.8, length=5)),
]

sbar_spec = TO.SlicedBarcodeSpec(
    ;
    directions=dirs,
    offsets=offs,
    featurizer=:summary,
    aggregate=:mean,
    normalize_weights=true,
    threads=true,
)

land_spec = TO.LandscapeSpec(
    ;
    directions=dirs,
    offsets=offs,
    tgrid=collect(range(0.0, stop=1.4, length=56)),
    kmax=3,
    aggregate=:mean,
    normalize_weights=true,
    threads=true,
)

spec = TO.CompositeSpec((sbar_spec, land_spec))

opts = TO.InvariantOptions(
    ;
    axes_policy=:encoding,
    max_axis_len=64,
    threads=true,
    strict=true,
    pl_mode=:fast,
)

stage("4) Batch transform with SessionCache reuse")

sc = TO.SessionCache()
fs = TO.batch_transform(
    samples,
    spec;
    opts=opts,
    idfun=s -> s.id,
    labelfun=s -> s.label,
    batch=TO.BatchOptions(threaded=true, backend=:threads, progress=false, deterministic=true),
    cache=sc,
    on_unsupported=:error,
)

println("Feature matrix shape: ", size(fs.X))
println("nfeatures(spec): ", TO.nfeatures(spec))
println("First 5 feature names: ", TO.feature_names(spec)[1:5])

stage("5) Save outputs")

paths = save_feature_bundle(outdir, "point_cloud_rips_density", fs)
println("CSV wide: ", paths.csv_wide)
println("CSV long: ", paths.csv_long)
println("Native optional outputs: ", paths.native)

# Save one canonical pipeline artifact for exact reruns.
pipe_path = joinpath(outdir, "pipeline.json")
TO.save_pipeline_json(
    pipe_path,
    clouds[1],
    filtration;
    degree=0,
    pipeline_opts=TO.PipelineOptions(axes_policy=:encoding, poset_kind=:signature, field=:F2),
)
println("Saved pipeline JSON (sample schema): ", pipe_path)

println("\nDone. Output directory: ", outdir)
