# =============================================================================
# Example 06: Batch experiment orchestration and exports
#
# Theme
# -----
# "I have many encoded samples and I want one reproducible pipeline run."
#
# This example uses ExperimentSpec/run_experiment to execute multiple featurizers,
# write artifacts, and emit a manifest suitable for reproducible workflows.
# =============================================================================

include(joinpath(@__DIR__, "_common.jl"))

example_header(
    "06",
    "Batch experiment spec + artifacts";
    theme="Pipeline definition separated from execution; reproducible run manifest.",
    teaches=[
        "How to define ExperimentSpec with multiple featurizers",
        "How to configure output formats without hidden defaults",
        "How to inspect run artifacts and load the manifest",
    ],
)

Random.seed!(20260217)
outdir = example_outdir("06_experiment_batch")

stage("1) Build encoded sample set")

clouds, labels = make_pointcloud_dataset(n_per_class=8, n_points=56, seed=6006)
filtration = TO.RipsDensityFiltration(
    ;
    max_dim=1,
    knn=8,
    density_k=8,
    nn_backend=:auto,
    construction=TO.ConstructionOptions(
        ;
        sparsify=:knn,
        budget=(max_simplices=120_000, max_edges=80_000, memory_budget_bytes=900_000_000),
    ),
)
field = TO.CoreModules.F2()
encodings = [TO.encode(clouds[i], filtration; degree=0, field=field, cache=:auto) for i in eachindex(clouds)]
samples = to_encoding_samples(encodings, labels; prefix="exp")

opts = TO.InvariantOptions(
    ;
    axes_policy=:encoding,
    max_axis_len=64,
    threads=true,
    strict=true,
    pl_mode=:fast,
)

stage("2) Define featurizers and experiment spec")

dirs = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
offs = [
    collect(range(-0.5, stop=0.5, length=5)),
    collect(range(-0.5, stop=0.5, length=5)),
    collect(range(-0.7, stop=0.7, length=5)),
]

land_spec = TO.LandscapeSpec(
    ;
    directions=dirs,
    offsets=offs,
    tgrid=collect(range(0.0, stop=1.2, length=48)),
    kmax=2,
    aggregate=:mean,
    normalize_weights=true,
    threads=true,
)

euler_spec = TO.EulerSurfaceSpec(
    ;
    axes_policy=:encoding,
    max_axis_len=64,
    threads=true,
)

# Use only actually-loaded extension formats.
formats = available_experiment_formats()
println("Available extension-backed formats: ", formats)

io = TO.ExperimentIOConfig(
    ;
    outdir=outdir,
    prefix="pointcloud_batch",
    format=:wide,
    formats=formats,
    write_metadata=true,
    overwrite=true,
)

exp = TO.ExperimentSpec(
    (land_spec, euler_spec);
    name="pointcloud_batch_demo",
    opts=opts,
    batch=TO.BatchOptions(threaded=true, backend=:threads, progress=false, deterministic=true),
    cache=:auto,
    on_unsupported=:error,
    idfun=s -> s.id,
    labelfun=s -> s.label,
    io=io,
    metadata=(purpose="onboarding_example", dataset="synthetic_point_cloud"),
)

stage("3) Run experiment")

res = TO.run_experiment(exp, samples)
println("Total elapsed seconds: ", round(res.total_elapsed_seconds, digits=3))
println("Run directory: ", res.run_dir)
println("Manifest path: ", res.manifest_path)

for art in res.artifacts
    println("Artifact $(art.key): elapsed=$(round(art.elapsed_seconds, digits=3))s, n_features=$(TO.nfeatures(art.features))")
    println("  feature_paths: ", art.feature_paths)

    # If no extension-backed format was written, still emit manual CSV for usability.
    if isempty(art.feature_paths)
        fallback = save_feature_bundle(outdir, String(art.key), art.features)
        println("  CSV wide: ", fallback.csv_wide)
    end
end

stage("4) Load manifest and inspect metadata-only view")

loaded = TO.load_experiment(res.run_dir; load_features=false, prefer=:none, strict=true)
println("Loaded manifest keys: ", sort!(collect(keys(loaded.manifest))))
println("Loaded artifacts: ", [a.key for a in loaded.artifacts])

println("\nDone. Output directory: ", outdir)
