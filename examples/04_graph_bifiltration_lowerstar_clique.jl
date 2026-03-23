# =============================================================================
# Example 04: Graph filtration families (lower-star and clique lift)
#
# Theme
# -----
# "Graphs are first-class ingestion targets; filtration choice changes features."
#
# We compare lower-star vs clique-lift behavior, then run a dataset-level batch
# transform with graph-derived encodings.
# =============================================================================

include(joinpath(@__DIR__, "_common.jl"))

example_header(
    "04",
    "Graph filtrations: lower-star vs clique";
    theme="Graph ingestion parity with point-cloud/image workflows.",
    teaches=[
        "How to use typed graph filtration families",
        "How lift=:lower_star vs lift=:clique changes encoded complexity",
        "How to export graph-derived feature matrices",
    ],
)

Random.seed!(20260217)
outdir = example_outdir("04_graph_filtrations")

stage("1) Build deterministic graph dataset")

graphs, labels = make_graph_dataset(n_per_class=6, n_vertices=42, seed=4004)
println("Dataset size: ", length(graphs), " graphs")
println("Class counts: geometric=", count(==("geometric"), labels), ", sbm=", count(==("sbm"), labels))

stage("2) Compare lower-star and clique lifts on one representative graph")

g0 = graphs[1]
field = TO.CoreModules.F2()
f_lower = TO.GraphCentralityFiltration(
    ;
    centrality=:degree,
    metric=:hop,
    lift=:lower_star,
    max_dim=2,
)
f_clique = TO.GraphCentralityFiltration(
    ;
    centrality=:degree,
    metric=:hop,
    lift=:clique,
    max_dim=2,
)

enc_lower = TO.encode(g0, f_lower; degree=0, field=field, cache=:auto)
enc_clique = TO.encode(g0, f_clique; degree=0, field=field, cache=:auto)

println("lower-star poset size: ", TO.nvertices(enc_lower.P))
println("clique-lift poset size: ", TO.nvertices(enc_clique.P))
println("lower-star Hilbert nnz: ", count(!iszero, TO.restricted_hilbert(enc_lower.M)))
println("clique-lift Hilbert nnz: ", count(!iszero, TO.restricted_hilbert(enc_clique.M)))

stage("3) Encode full dataset with lower-star lift")

encodings = Vector{TO.EncodingResult}(undef, length(graphs))
for i in eachindex(graphs)
    encodings[i] = TO.encode(graphs[i], f_lower; degree=0, field=field, cache=:auto)
end
samples = to_encoding_samples(encodings, labels; prefix="graph")

opts = TO.InvariantOptions(
    ;
    axes_policy=:encoding,
    max_axis_len=64,
    threads=true,
    strict=true,
    pl_mode=:fast,
)

stage("4) Featurize (slice summaries + landscapes)")

dirs = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
offs = [
    collect(range(-0.5, stop=0.5, length=5)),
    collect(range(-0.5, stop=0.5, length=5)),
    collect(range(-0.7, stop=0.7, length=5)),
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
    tgrid=collect(range(0.0, stop=1.2, length=48)),
    kmax=2,
    aggregate=:mean,
    normalize_weights=true,
    threads=true,
)

spec = TO.CompositeSpec((sbar_spec, land_spec))

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

stage("5) Save outputs")

paths = save_feature_bundle(outdir, "graph_lowerstar_features", fs)
println("CSV wide: ", paths.csv_wide)
println("CSV long: ", paths.csv_long)
println("Native optional outputs: ", paths.native)

println("\nDone. Output directory: ", outdir)
