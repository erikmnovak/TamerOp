# =============================================================================
# Example 03: Image bifiltration (distance + intensity)
#
# Theme
# -----
# "Use image data directly, without converting to point clouds first."
#
# We build a synthetic image dataset, ingest via ImageDistanceBifiltration, and
# featurize with Euler + slice summaries.
# =============================================================================

include(joinpath(@__DIR__, "_common.jl"))

example_header(
    "03",
    "Image bifiltration via distance/intensity";
    theme="Non-point-cloud ingestion with the same encoding + featurization pipeline.",
    teaches=[
        "How to provide image masks for distance bifiltration",
        "How to inspect Euler surfaces on image-derived modules",
        "How to export image-derived feature tables",
    ],
)

Random.seed!(20260217)
outdir = example_outdir("03_image_bifiltration")

stage("1) Build deterministic synthetic images")

imgs, masks, labels = make_image_distance_dataset(n_per_class=5, side=40, seed=3003)
println("Dataset size: ", length(imgs), " images")
println("Mask size: ", size(masks[1]))

stage("2) Encode each image with ImageDistanceBifiltration")

field = TO.CoreModules.F2()
encodings = Vector{TO.EncodingResult}(undef, length(imgs))
for i in eachindex(imgs)
    filt = TO.ImageDistanceBifiltration(; mask=masks[i])
    encodings[i] = TO.encode(imgs[i], filt; degree=0, field=field, cache=:auto)
end

samples = to_encoding_samples(encodings, labels; prefix="img")

stage("3) Direct invariant sanity checks on the first sample")

opts = TO.InvariantOptions(
    ;
    axes_policy=:encoding,
    max_axis_len=64,
    threads=true,
    strict=true,
    pl_mode=:fast,
)

surf0 = TO.euler_surface(encodings[1]; opts=opts)
println("euler_surface(first image) size: ", size(surf0))

dirs = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
offs = [
    collect(range(-0.5, stop=0.5, length=5)),
    collect(range(-0.5, stop=0.5, length=5)),
    collect(range(-0.7, stop=0.7, length=5)),
]
sb0 = TO.slice_barcodes(encodings[1]; opts=opts, directions=dirs, offsets=offs, threads=true)
println("slice_barcodes(first image) keys: ", collect(keys(sb0)))

stage("4) Featurize dataset")

euler_spec = TO.EulerSurfaceSpec(
    ;
    axes_policy=:encoding,
    max_axis_len=64,
    threads=true,
)

sbar_spec = TO.SlicedBarcodeSpec(
    ;
    directions=dirs,
    offsets=offs,
    featurizer=:summary,
    aggregate=:mean,
    normalize_weights=true,
    threads=true,
)

spec = TO.CompositeSpec((euler_spec, sbar_spec))

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
println("First 5 feature names: ", TO.feature_names(spec)[1:5])

stage("5) Save outputs")

paths = save_feature_bundle(outdir, "image_distance_bifiltration", fs)
println("CSV wide: ", paths.csv_wide)
println("CSV long: ", paths.csv_long)
println("Native optional outputs: ", paths.native)

println("\nDone. Output directory: ", outdir)
