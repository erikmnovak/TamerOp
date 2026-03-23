# Onboarding Examples

This directory is an ordered, didactic onboarding suite for `TamerOp`.

Each example includes:
- a theme header,
- explicit pipeline stages,
- deterministic seeds,
- guardrails for combinatorial blowup,
- concrete output artifacts under `examples/_outputs/<example_name>/`.

Notebook variants are available for the first four onboarding examples and the
visualization walk-throughs:
- `00_quickstart_module_to_features.ipynb`
- `01_graded_complex_by_hand.ipynb`
- `02_point_cloud_bifiltration_rips_density.ipynb`
- `03_image_bifiltration_distance_intensity.ipynb`
- `10_visualization_engine_basics.ipynb`
- `11_ingestion_visualization_pipeline.ipynb`
- `12_graph_ingestion_visualization_pipeline.ipynb`
- `13_synthetic_families_and_coupled_fringes.ipynb`

Workshop tutorial suite:
- `combinatorial_cowork_2026/README.md`
- `combinatorial_cowork_2026/00_kernel_smoke_and_workflow_contract.ipynb`
- `combinatorial_cowork_2026/01_presentations_to_encodings.ipynb`
- `combinatorial_cowork_2026/02_resolutions_hom_ext_tor.ipynb`
- `combinatorial_cowork_2026/03_change_of_posets_and_common_refinement.ipynb`
- `combinatorial_cowork_2026/04_invariants_and_visual_payoff.ipynb`
- `combinatorial_cowork_2026/05_noisy_annulus_rips_codensity_h1_2persistence.ipynb`
- `combinatorial_cowork_2026/90_speaker_appendix_exact_queries_and_extra_derived.ipynb`

Custom-filtration starter template:
- `99_custom_trigrade_filtration.ipynb` (register a user-defined `P=3` filtration family)

## Run order

1. `00_quickstart_module_to_features.ipynb`
2. `01_graded_complex_by_hand.jl` (or `01_graded_complex_by_hand.ipynb`)
3. `02_point_cloud_bifiltration_rips_density.jl` (or `02_point_cloud_bifiltration_rips_density.ipynb`)
4. `03_image_bifiltration_distance_intensity.jl` (or `03_image_bifiltration_distance_intensity.ipynb`)
5. `04_graph_bifiltration_lowerstar_clique.jl`
6. `05_fibered_slices_landscapes_and_distances.jl`
7. `06_batch_experiment_spec_and_exports.jl`
8. `07_serialization_reproducible_runs.jl`
9. `08_fibered2d_exact_queries.jl`
10. `09_advanced_ingestion_plan_and_preflight.jl`
11. `10_visualization_engine_basics.jl` (or `10_visualization_engine_basics.ipynb`)
12. `11_ingestion_visualization_pipeline.jl` (or `11_ingestion_visualization_pipeline.ipynb`)
13. `12_graph_ingestion_visualization_pipeline.jl` (or `12_graph_ingestion_visualization_pipeline.ipynb`)
14. `13_synthetic_families_and_coupled_fringes.jl` (or `13_synthetic_families_and_coupled_fringes.ipynb`)
15. `90_advanced_indicator_resolutions_and_ext.jl` (optional advanced appendix)

## Running examples

From repo root:

```bash
jupyter lab examples/00_quickstart_module_to_features.ipynb
```

To run the notebook variants, open:
- `examples/00_quickstart_module_to_features.ipynb`
- `examples/01_graded_complex_by_hand.ipynb`
- `examples/02_point_cloud_bifiltration_rips_density.ipynb`
- `examples/03_image_bifiltration_distance_intensity.ipynb`
- `examples/10_visualization_engine_basics.ipynb`
- `examples/11_ingestion_visualization_pipeline.ipynb`
- `examples/12_graph_ingestion_visualization_pipeline.ipynb`
- `examples/99_custom_trigrade_filtration.ipynb`

or run all in order:

```bash
for f in \
  examples/01_graded_complex_by_hand.jl \
  examples/02_point_cloud_bifiltration_rips_density.jl \
  examples/03_image_bifiltration_distance_intensity.jl \
  examples/04_graph_bifiltration_lowerstar_clique.jl \
  examples/05_fibered_slices_landscapes_and_distances.jl \
  examples/06_batch_experiment_spec_and_exports.jl \
  examples/07_serialization_reproducible_runs.jl \
  examples/08_fibered2d_exact_queries.jl \
  examples/09_advanced_ingestion_plan_and_preflight.jl \
  examples/10_visualization_engine_basics.jl \
  examples/11_ingestion_visualization_pipeline.jl \
  examples/12_graph_ingestion_visualization_pipeline.jl \
  examples/13_synthetic_families_and_coupled_fringes.jl \
  examples/90_advanced_indicator_resolutions_and_ext.jl
do
  julia --project "$f"
done
```

## Notes

- Feature exports use the canonical `save_features(...)` API.
- CSV output (wide + long) is written for all examples; optional Arrow/Parquet/NPZ
  outputs depend on loaded extensions.
- Data-ingestion API is canonical:
  - use `encode(data, filtration_or_spec; ...)`
  - use `stage=...` to request intermediate objects
    (`:simplex_tree`, `:graded_complex`, `:cochain`, `:module`, `:fringe`, `:flange`, `:encoding_result`)
  - default is `stage=:auto`, which follows `ConstructionOptions.output_stage`
- All examples use the canonical Workflow cache contract:
  - `cache=:auto` for per-call automatic caching
  - `cache=SessionCache()` for explicit cross-call reuse
- PL mode uses strict canonical values only: `:fast` / `:verified`.

## Simple Serialization

For onboarding and reproducible runs, use these as the canonical encoding I/O calls:

```julia
using TamerOp

enc = encode(data, spec; degree=0)

# Default save profile is :compact (small file, includes pi, omits dense leq for structured posets).
save_encoding_json("encoding.json", enc)

# Quick metadata probe without full object reconstruction.
meta = inspect_json("encoding.json")

# Default load now returns EncodingResult for simple users.
enc2 = load_encoding_json("encoding.json")

# If you only need fringe data:
H = load_encoding_json("encoding.json"; output=:fringe)

# If file is trusted and you want fastest load path:
H_fast = load_encoding_json("encoding.json"; output=:fringe, validation=:trusted)

# Explicit profile presets when needed:
save_encoding_json("encoding_portable.json", enc; profile=:portable)  # include dense leq
save_encoding_json("encoding_debug.json", enc; profile=:debug)        # pretty JSON + dense leq
```

## File Ingestion Quickstart

For files on disk, use the canonical `load_data` + `encode(path, ...)` path:

```julia
using TamerOp

spec = FiltrationSpec(kind=:rips, max_dim=1)

# 1) Inspect first (recommended for CSV/TSV/TXT).
info = inspect_data_file("points.csv")

# 2) Parse file into typed ingestion data.
file_opts = DataFileOptions(; header=true, cols=(:x, :y))
data = load_data("points.csv"; kind=:point_cloud, opts=file_opts)

# 3) Encode directly from path (one-liner).
enc = encode("points.csv", spec; kind=:point_cloud, file_opts=file_opts, degree=0)
```

Notes:
- For table files (`.csv`, `.tsv`, `.txt`), pass `kind=...` explicitly.
- For dataset JSON created with `save_dataset_json`, `kind=:auto` works.
- Graph tables can use edge columns via `DataFileOptions(; u_col=:u, v_col=:v, weight_col=:w)`.
