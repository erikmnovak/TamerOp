# Ingestion Comparison Harness (tamer-op vs multipers)

This harness provides a reproducible, case-based comparison for point-cloud and image ingestion.

## What it benchmarks

Primary regimes:

- `claim_matching`
  - Radius-threshold parity regime with shared per-case radius from fixture manifest.
  - tamer-op: Rips + `sparsify=:radius`, `radius=claim_radius`, `collapse=:none`, `output_stage=:simplex_tree`.
  - multipers: `gudhi.RipsComplex(..., max_edge_length=claim_radius)` wrapped as `multipers.simplex_tree_multi.SimplexTreeMulti`.

- `rips_parity`
  - Shared-threshold parity regime for direct algorithm comparison.
  - fixture generation writes `parity_radius` (radius-k policy) into the manifest.
  - tamer-op: Rips + `radius=parity_radius`, `sparsify=:none`, `collapse=:none`, `output_stage=:simplex_tree`.
  - multipers: `gudhi.RipsComplex(..., max_edge_length=parity_radius)` wrapped as `SimplexTreeMulti`.
  - fallback path (no `parity_radius` in manifest): `PointCloud2FilteredComplex(..., complex="rips", output_type="simplextree", num_collapses=0)`.

- `landmark_parity`
  - Shared deterministic landmark subset + shared radius threshold on that subset.
  - tamer-op: `kind=:landmark_rips` with manifest landmarks + `radius=landmark_radius`.
  - multipers: GUDHI Rips on the exact same landmark subset with `max_edge_length=landmark_radius`, wrapped as `SimplexTreeMulti`.
  - This is strict apples-to-apples for landmark-only Rips construction.

- `degree_rips_parity`
  - Shared radius-threshold parity for degree-Rips bifiltration.
  - tamer-op: `kind=:degree_rips`, `radius=degree_radius`, `output_stage=:simplex_tree`.
  - multipers: `filtrations.DegreeRips(points, threshold_radius=degree_radius)`.

- `rips_codensity_parity`
  - Shared radius + density-policy parity for codensity-style Rips bifiltration.
  - tamer-op: `kind=:rips_codensity`, `radius=codensity_radius`, `dtm_mass=codensity_dtm_mass`.
  - multipers: `filtrations.RipsCodensity(points, dtm_mass=codensity_dtm_mass, threshold_radius=codensity_radius)`.

- `rips_lowerstar_parity`
  - Shared radius + shared vertex function parity (`coord1`).
  - tamer-op: `kind=:function_rips`, `vertex_function=(p,_)->p[1]`, `radius=lowerstar_radius`.
  - multipers: `filtrations.RipsLowerstar(points=..., function=coord1, threshold_radius=lowerstar_radius)`.

- `delaunay_lowerstar_parity`
  - Shared Delaunay lower-star parity with the same vertex function (`coord1`).
  - tamer-op: `kind=:function_delaunay`, `vertex_function=(p,_)->p[1]`.
  - multipers: `filtrations.DelaunayLowerstar(points, function=coord1)`.
  - Some multipers builds may not include `function_delaunay`; those cases are skipped in `run_multipers.py`.

- `alpha_parity` (alpha parity)
  - Non-Rips alpha-complex parity comparison for geometric filtrations in 2D.
  - tamer-op: `kind=:alpha`, `output_stage=:simplex_tree`.
  - multipers: `PointCloud2FilteredComplex(complex="alpha", output_type="simplextree")`.

- `core_delaunay_parity`
  - Non-Rips core-Delaunay parity comparison in 2D.
  - tamer-op: `kind=:core_delaunay`, `max_dim=...`, `output_stage=:simplex_tree`.
  - multipers: `filtrations.CoreDelaunay(...)`, then `prune_above_dimension(max_dim)`.

- `cubical_parity`
  - Image cubical filtration parity comparison.
  - tamer-op: `kind=:cubical`, `output_stage=:graded_complex` (cubical cells).
  - multipers: `filtrations.Cubical(image[...,None])`.

Outputs include:
- `cold_ms`: **warm-uncached single-shot** time (one untimed prewarm call, then one timed uncached call).
- `warm_*`: repeated uncached timings (median/p90).
This isolates algorithmic runtime and avoids startup/JIT process costs in the reported `cold_ms`.

## End-to-end invariant harness

There is a companion harness for measuring the cost of going from in-memory raw
data all the way to a requested invariant, rather than stopping at an
intermediate build stage.

Current invariant coverage:

- `euler_signed_measure`
  - tamer-op: build the canonical `:encoded_complex` workflow object, then evaluate `euler_signed_measure(encoded_complex_result)`
  - multipers: build filtered complex then `signed_measure(..., invariant="euler")`
- `slice_barcodes`
  - experimental harness only
  - tamer-op: build the canonical `:encoding_result` workflow object, then evaluate `slice_barcodes(encoding_result; directions=..., offsets=...)`
  - multipers: build filtered complex, then run `Slicer(...).persistence_on_line(...)` on the shared manifest slice family
  - not benchmark-approved yet; current raw-line contract is not parity-clean
- `rank_signed_measure`
  - still available as an explicit opt-in
  - not part of the first reportable matrix, because the current tamer-op raw-data
    route is not yet stable enough for parity-grade comparison there

First reportable benchmark contract:

- invariant target: `euler_signed_measure` only
- fixture scale: `0.5`
- profile: `desktop`
- case matrix: `benchmark/ingestion_compare_harness/cases_invariants_representative.toml`
- output comparison policy:
  - write all successful case rows,
  - canonicalize signed-measure outputs,
  - record `parity_status` / `parity_reason` per case,
  - include only parity-matched rows in the geomean summary

Representative case matrix:

- `inv_alpha_n150_d2_md2_s11`
- `inv_alpha_n1500_d2_md2_s11`
- `inv_alpha_n15000_d2_md2_s11`
- `inv_cubical_side24_s11`
- `inv_cubical_side80_s11`
- `inv_cubical_side253_s11`

Invariant eligibility is declared in the case TOML itself:

- top-level `[invariant_eligibility]`
- keys are regime names
- values are the benchmark-approved invariant names for that regime
- `--invariants all` expands only to those approved pairs

`degree_rips_parity` and `core_delaunay_parity` are still intentionally
excluded from this first presentation-grade matrix because the local multipers
build does not expose the needed exact invariant route for those regimes.
`rips_parity`, `rips_codensity_parity`, `rips_lowerstar_parity`, and `landmark_parity` now
have small-case Euler parity approval in the full invariant matrix, but they
are not yet part of the presentation-grade representative set.

Invariant outputs include:

- `cold_ms`: one untimed prewarm, then one timed uncached end-to-end run
- `warm_*`: repeated uncached end-to-end timings
- `output_term_count`: number of signed-measure atoms/rectangles returned
- `output_abs_mass`: sum of absolute output weights
- `output_measure_canonical`: canonicalized support/weight serialization used for parity checks

Invariant comparison outputs include:

- `parity_status`
  - `matched` or `mismatched`
- `parity_reason`
  - reason for mismatch when outputs differ
- `summary_included`
  - `yes` only for parity-matched rows

The comparison summary is headline-only: mismatches and runner failures are kept
in the raw outputs and failure report, but excluded from summary geomeans.

## Files

- `cases.toml`: case matrix (sizes, dimensions, seeds, regimes).
- `generate_fixtures.py`: generates deterministic CSV point-cloud/image fixtures and manifest.
- `run_tamer.jl`: runs tamer-op ingestion on the fixture manifest.
- `run_multipers.py`: runs multipers ingestion on the same fixtures.
- `compare.py`: joins result CSVs and computes slowdown tables.
- `run_all.sh`: one-shot runner.

## Usage

From repo root:

```bash
# Quick sanity pass (scaled down fixtures)
SCALE=0.01 PROFILE=probe REGIME=all bash benchmark/ingestion_compare_harness/run_all.sh

# Desktop-safe default profile (recommended local default)
SCALE=1.0 PROFILE=desktop REGIME=all bash benchmark/ingestion_compare_harness/run_all.sh

# Balanced profile (more repetitions)
SCALE=1.0 PROFILE=balanced REGIME=all bash benchmark/ingestion_compare_harness/run_all.sh

# Heavier stress profile
SCALE=1.0 PROFILE=stress REGIME=all bash benchmark/ingestion_compare_harness/run_all.sh

# Explicitly override profile reps
SCALE=1.0 PROFILE=desktop REPS=6 REGIME=all bash benchmark/ingestion_compare_harness/run_all.sh

# Run an expanded apples-to-apples matrix (different n / ambient_dim / max_dim)
CASES_FILE=benchmark/ingestion_compare_harness/cases_matrix.toml \
FIXTURES_DIR=benchmark/ingestion_compare_harness/fixtures_matrix \
TAMER_OUT=benchmark/ingestion_compare_harness/results_tamer_matrix.csv \
MULTIPERS_OUT=benchmark/ingestion_compare_harness/results_multipers_matrix.csv \
COMPARE_OUT=benchmark/ingestion_compare_harness/comparison_matrix.csv \
SUMMARY_OUT=benchmark/ingestion_compare_harness/comparison_summary_matrix.csv \
SCALE=1.0 PROFILE=desktop REGIME=all \
bash benchmark/ingestion_compare_harness/run_all.sh

# Run non-Rips matrix (alpha/core_delaunay/cubical)
CASES_FILE=benchmark/ingestion_compare_harness/cases_nonrips.toml \
FIXTURES_DIR=benchmark/ingestion_compare_harness/fixtures_nonrips \
TAMER_OUT=benchmark/ingestion_compare_harness/results_tamer_nonrips.csv \
MULTIPERS_OUT=benchmark/ingestion_compare_harness/results_multipers_nonrips.csv \
COMPARE_OUT=benchmark/ingestion_compare_harness/comparison_nonrips.csv \
SUMMARY_OUT=benchmark/ingestion_compare_harness/comparison_summary_nonrips.csv \
SCALE=1.0 PROFILE=desktop REGIME=all \
bash benchmark/ingestion_compare_harness/run_all.sh

# Run landmark-subset parity matrix
CASES_FILE=benchmark/ingestion_compare_harness/cases_matrix.toml \
FIXTURES_DIR=benchmark/ingestion_compare_harness/fixtures_matrix \
TAMER_OUT=benchmark/ingestion_compare_harness/results_tamer_matrix.csv \
MULTIPERS_OUT=benchmark/ingestion_compare_harness/results_multipers_matrix.csv \
COMPARE_OUT=benchmark/ingestion_compare_harness/comparison_matrix.csv \
SUMMARY_OUT=benchmark/ingestion_compare_harness/comparison_summary_matrix.csv \
SCALE=1.0 PROFILE=desktop REGIME=landmark_parity \
bash benchmark/ingestion_compare_harness/run_all.sh

# Generate the representative invariant fixtures
python benchmark/ingestion_compare_harness/generate_fixtures.py \
  --cases benchmark/ingestion_compare_harness/cases_invariants_representative.toml \
  --out_dir benchmark/ingestion_compare_harness/fixtures_invariants_representative_scale05 \
  --scale 0.5 \
  --force

# Run the representative matrix in fresh per-case processes for both tools
python benchmark/ingestion_compare_harness/run_invariant_supervisor.py \
  --manifest benchmark/ingestion_compare_harness/fixtures_invariants_representative_scale05/manifest.toml \
  --tools both \
  --profile desktop \
  --invariants all \
  --degree 0 \
  --work_dir benchmark/ingestion_compare_harness/_run_invariant_supervisor_representative_scale05 \
  --tamer_out benchmark/ingestion_compare_harness/results_tamer_invariants_representative_scale05.csv \
  --multipers_out benchmark/ingestion_compare_harness/results_multipers_invariants_representative_scale05.csv \
  --comparison_out benchmark/ingestion_compare_harness/comparison_invariants_representative_scale05.csv \
  --summary_out benchmark/ingestion_compare_harness/comparison_summary_invariants_representative_scale05.csv \
  --comparison_failures_out benchmark/ingestion_compare_harness/comparison_failures_invariants_representative_scale05.csv
```

If you want the manual three-step route instead of the supervisor:

```bash
julia --project=. benchmark/ingestion_compare_harness/run_tamer_invariants.jl \
  --manifest=benchmark/ingestion_compare_harness/fixtures_invariants_representative_scale05/manifest.toml \
  --out=benchmark/ingestion_compare_harness/results_tamer_invariants_representative_scale05.csv \
  --profile=desktop \
  --invariants=all \
  --degree=0

python benchmark/ingestion_compare_harness/run_multipers_invariants.py \
  --manifest benchmark/ingestion_compare_harness/fixtures_invariants_representative_scale05/manifest.toml \
  --out benchmark/ingestion_compare_harness/results_multipers_invariants_representative_scale05.csv \
  --profile desktop \
  --invariants all \
  --degree 0

python benchmark/ingestion_compare_harness/compare_invariants.py \
  --tamer benchmark/ingestion_compare_harness/results_tamer_invariants_representative_scale05.csv \
  --multipers benchmark/ingestion_compare_harness/results_multipers_invariants_representative_scale05.csv \
  --out benchmark/ingestion_compare_harness/comparison_invariants_representative_scale05.csv \
  --summary_out benchmark/ingestion_compare_harness/comparison_summary_invariants_representative_scale05.csv \
  --failures_out benchmark/ingestion_compare_harness/comparison_failures_invariants_representative_scale05.csv
```

Profiles:

- `desktop` (default): `reps=4`, trims memory between cases/reps
- `balanced`: `reps=5`, trims memory between cases
- `stress`: `reps=9`
- `probe`: `reps=3`, trims memory between cases

Both runners accept explicit `--reps` to override the profile default.

## Robust Single-Shot Number (single metric)

For a single robust algorithmic single-shot metric, run fresh-process repeats per case and aggregate medians:

```bash
python benchmark/ingestion_compare_harness/cold_robust.py \
  --manifest benchmark/ingestion_compare_harness/fixtures_50k/manifest.toml \
  --regime claim_matching \
  --restarts 3 \
  --profile desktop

# Run robust cold on the expanded matrix
python benchmark/ingestion_compare_harness/cold_robust.py \
  --manifest benchmark/ingestion_compare_harness/fixtures_matrix/manifest.toml \
  --regime all \
  --restarts 3 \
  --profile desktop \
  --raw_out benchmark/ingestion_compare_harness/cold_robust_raw_matrix.csv \
  --summary_out benchmark/ingestion_compare_harness/cold_robust_summary_matrix.csv
```

Outputs:
- `cold_robust_raw.csv`: all per-repeat cold measurements
- `cold_robust_summary.csv`: per-case median cold times + robust geomean slowdown

To rank where tamer is slower/faster:

```bash
python benchmark/ingestion_compare_harness/where_slower.py \
  --comparison benchmark/ingestion_compare_harness/comparison_matrix.csv \
  --manifest benchmark/ingestion_compare_harness/fixtures_matrix/manifest.toml \
  --out benchmark/ingestion_compare_harness/where_slower_matrix.csv
```

## Result files

Generated in this directory:

- `results_tamer.csv`
- `results_multipers.csv`
- `comparison.csv`
- `comparison_summary.csv`

`comparison_summary.csv` reports geometric-mean slowdown:

- `warm_geomean_slowdown_tamer_over_multipers`
- `cold_geomean_slowdown_tamer_over_multipers`

Values > 1.0 mean tamer-op is slower than multipers.

## Notes

- Use the same thread settings for both tools when comparing (e.g. single-thread vs all-core).
- `run_multipers.py` requires `multipers` installed in the active Python environment.
- `claim_matching` now aligns construction policy knobs via shared `claim_radius` and no collapse on both tools.
- `alpha_parity` now runs alpha-vs-alpha parity (canonical alpha parity regime).
