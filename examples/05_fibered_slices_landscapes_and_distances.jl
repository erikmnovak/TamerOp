# =============================================================================
# Example 05: Fibered slices, landscapes, and projected distances
#
# Theme
# -----
# "RIVET-adjacent workflow: many slices, cache reuse, and distance features."
#
# This example emphasizes computational structure:
# - compile slice geometry once,
# - reuse caches across many transforms,
# - build distance-to-reference features.
# =============================================================================

include(joinpath(@__DIR__, "_common.jl"))

example_header(
    "05",
    "Fibered slices + projected distances";
    theme="Fit-once/transform-many workflow with explicit cache reuse.",
    teaches=[
        "How to compile and reuse slice plans",
        "How to combine landscapes with distance-to-reference features",
        "How SessionCache helps repeated batch transforms",
    ],
)

Random.seed!(20260217)
outdir = example_outdir("05_fibered_slices")

stage("1) Prepare encoded samples")

clouds, labels = make_pointcloud_dataset(n_per_class=6, n_points=60, seed=5005)
filtration = TO.RipsDensityFiltration(
    ;
    max_dim=1,
    knn=9,
    density_k=9,
    nn_backend=:auto,
    construction=TO.ConstructionOptions(
        ;
        sparsify=:knn,
        budget=(max_simplices=120_000, max_edges=80_000, memory_budget_bytes=900_000_000),
    ),
)
field = TO.CoreModules.F2()

encodings = [TO.encode(clouds[i], filtration; degree=0, field=field, cache=:auto) for i in eachindex(clouds)]
samples = to_encoding_samples(encodings, labels; prefix="fiber")

opts = TO.InvariantOptions(
    ;
    axes_policy=:encoding,
    max_axis_len=64,
    threads=true,
    strict=true,
    pl_mode=:fast,
)

stage("2) Compile slice plan once and reuse it")

dirs = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]]
offs = [
    collect(range(-0.5, stop=0.5, length=5)),
    collect(range(-0.5, stop=0.5, length=5)),
    collect(range(-0.7, stop=0.7, length=5)),
    collect(range(-0.7, stop=0.7, length=5)),
]

plan = TO.Invariants.compile_slices(encodings[1].pi, opts; directions=dirs, offsets=offs, threads=true)
slice0 = TO.slice_barcodes(encodings[1].M, plan, opts; packed=true, threaded=true)
println("Compiled plan type: ", typeof(plan))
println("slice_barcodes(plan) keys: ", collect(keys(slice0)))

stage("3) Build featurizers (landscapes + projected distances)")

land_spec = TO.LandscapeSpec(
    ;
    directions=dirs,
    offsets=offs,
    tgrid=collect(range(0.0, stop=1.3, length=56)),
    kmax=3,
    aggregate=:mean,
    normalize_weights=true,
    threads=true,
)

# References can be any samples carrying (M, pi).
refs = samples[1:3]
proj_spec = TO.ProjectedDistancesSpec(
    refs;
    reference_names=[:ref_circle_1, :ref_circle_2, :ref_figure8_1],
    dist=:bottleneck,
    n_dirs=24,
    precompute=true,
    threads=true,
)

spec = TO.CompositeSpec((land_spec, proj_spec))

stage("4) Batch transform with a shared SessionCache")

sc = TO.SessionCache()
fs = TO.batch_transform(
    samples,
    spec;
    opts=opts,
    idfun=s -> s.id,
    labelfun=s -> s.label,
    batch=TO.BatchOptions(threaded=true, backend=:threads, progress=false, deterministic=true),
    cache=sc,
)

println("Feature matrix shape: ", size(fs.X))
println("nfeatures(spec): ", TO.nfeatures(spec))

# Inspect cache footprint for one representative sample/spec.
cache_obj = TO.build_cache(samples[1], spec; opts=opts, threaded=true)
println("Cache stats (single-sample cache object): ", TO.cache_stats(cache_obj))

stage("5) Save outputs")

paths = save_feature_bundle(outdir, "fibered_landscape_projected", fs)
println("CSV wide: ", paths.csv_wide)
println("CSV long: ", paths.csv_long)
println("Native optional outputs: ", paths.native)

println("\nDone. Output directory: ", outdir)
