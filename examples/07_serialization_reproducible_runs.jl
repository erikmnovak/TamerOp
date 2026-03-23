# =============================================================================
# Example 07: Serialization, reproducibility, and interchange
#
# Theme
# -----
# "I can save pipeline artifacts, reload them in a clean session, and verify
# feature outputs are reproducible."
#
# This example focuses on operational discipline:
#   1) persist dataset + typed filtration contract,
#   2) rerun from serialized artifacts,
#   3) verify feature identities and values.
# =============================================================================

include(joinpath(@__DIR__, "_common.jl"))

example_header(
    "07",
    "Serialization and reproducible reruns";
    theme="Persist-and-reload workflow with explicit reproducibility assertions.",
    teaches=[
        "How to save and reload dataset/pipeline JSON artifacts",
        "How to convert FiltrationSpec back to typed filtration via to_filtration",
        "How to assert feature reproducibility numerically",
        "How to emit portable feature files for external stats workflows",
    ],
)

Random.seed!(20260217)
outdir = example_outdir("07_serialization_repro")

stage("1) Build a compact deterministic GradedComplex")

# Keep the object intentionally small so this example executes quickly.
cells_by_dim = [Int[1, 2, 3], Int[1, 2]]
B1 = sparse(
    [1, 2, 2, 3],
    [1, 1, 2, 2],
    [1, -1, 1, -1],
    3,
    2,
)
grades = [
    [0.0, 0.0],
    [0.6, 0.0],
    [1.2, 0.0],
    [0.6, 0.4],
    [1.2, 0.8],
]
data = TO.GradedComplex(cells_by_dim, [B1], grades)
filtration = TO.GradedFiltration()

opts = TO.InvariantOptions(
    ;
    axes=(collect(range(0.0, stop=1.6, length=5)), collect(range(0.0, stop=1.2, length=4))),
    axes_policy=:as_given,
    max_axis_len=64,
    threads=false,
    strict=true,
    pl_mode=:fast,
)

stage("2) Baseline encode + featurize")

enc_ref = TO.encode(data, filtration; degree=0, field=TO.CoreModules.F2(), cache=:auto)
spec = TO.EulerSurfaceSpec(axes=opts.axes, axes_policy=:as_given, threads=false)

vec_ref = TO.transform(spec, enc_ref; opts=opts, threaded=false)
fs_ref = TO.FeatureSet(
    reshape(vec_ref, 1, :),
    TO.feature_names(spec),
    ["repro_001"],
    (label=["graded"], source="baseline"),
)

println("Baseline feature shape: ", size(fs_ref.X))
println("Baseline nfeatures(spec): ", TO.nfeatures(spec))

stage("3) Save reproducibility artifacts")

dataset_path = joinpath(outdir, "dataset.json")
pipeline_path = joinpath(outdir, "pipeline.json")

TO.save_dataset_json(dataset_path, data)
TO.save_pipeline_json(
    pipeline_path,
    data,
    filtration;
    degree=0,
    pipeline_opts=TO.PipelineOptions(
        orientation=(1, 1),
        axes_policy=:as_given,
        axis_kind=:explicit,
        poset_kind=:signature,
        field=:F2,
    ),
)

println("Saved dataset JSON: ", dataset_path)
println("Saved pipeline JSON: ", pipeline_path)

# Save features with extension-backed paths when available, plus manual CSV fallback.
paths = save_feature_bundle(outdir, "repro_features", fs_ref)
println("CSV wide: ", paths.csv_wide)
println("CSV long: ", paths.csv_long)
println("Native optional outputs: ", paths.native)

stage("4) Reload artifacts and rerun")

data_reloaded = TO.load_dataset_json(dataset_path)
data_from_pipe, spec_from_pipe, degree_from_pipe, popts_from_pipe = TO.load_pipeline_json(pipeline_path)
filtration_reloaded = TO.to_filtration(spec_from_pipe)

println("Loaded degree: ", degree_from_pipe)
println("Loaded pipeline options axes_policy: ", popts_from_pipe.axes_policy)
println("Loaded pipeline options field: ", popts_from_pipe.field)

field_reloaded = if popts_from_pipe.field === nothing || popts_from_pipe.field === :F2
    TO.CoreModules.F2()
elseif popts_from_pipe.field === :F3
    TO.CoreModules.F3()
elseif popts_from_pipe.field === :QQ
    TO.CoreModules.QQ
else
    throw(ArgumentError("Unsupported pipeline field in this example: $(popts_from_pipe.field)"))
end

enc_reloaded = TO.encode(data_reloaded, filtration_reloaded; degree=Int(degree_from_pipe), field=field_reloaded, cache=:auto)
vec_reloaded = TO.transform(spec, enc_reloaded; opts=opts, threaded=false)
fs_reloaded = TO.FeatureSet(
    reshape(vec_reloaded, 1, :),
    TO.feature_names(spec),
    ["repro_001"],
    (label=["graded"], source="reloaded"),
)

stage("5) Verify reproducibility")

# Dataset payload check (structural, not object-identity).
@assert typeof(data_from_pipe) == typeof(data_reloaded)
@assert data_from_pipe.cells_by_dim == data_reloaded.cells_by_dim
@assert data_from_pipe.cell_dims == data_reloaded.cell_dims
@assert data_from_pipe.grades == data_reloaded.grades
@assert length(data_from_pipe.boundaries) == length(data_reloaded.boundaries)
for i in eachindex(data_from_pipe.boundaries)
    @assert findnz(data_from_pipe.boundaries[i]) == findnz(data_reloaded.boundaries[i])
end
maxabs = assert_feature_sets_match(fs_ref, fs_reloaded; atol=1e-10, rtol=1e-8)
println("Feature reproducibility check passed.")
println("Max |baseline - reloaded|: ", maxabs)

# Optional extension-backed load check if a native feature file exists.
if haskey(paths.native, :npz)
    fs_npz = TO.load_features(paths.native[:npz]; format=:npz)
    maxabs_npz = assert_feature_sets_match(fs_ref, fs_npz; atol=1e-10, rtol=1e-8)
    println("NPZ round-trip reproducibility check passed (maxabs=$(maxabs_npz)).")
elseif haskey(paths.native, :csv_wide)
    fs_csv = TO.load_features(paths.native[:csv_wide]; format=:csv, mode=:wide)
    maxabs_csv = assert_feature_sets_match(fs_ref, fs_csv; atol=1e-10, rtol=1e-8)
    println("CSV round-trip reproducibility check passed (maxabs=$(maxabs_csv)).")
else
    println("No extension-backed feature loader available; manual CSV outputs are still written.")
end

println("\nDone. Output directory: ", outdir)
