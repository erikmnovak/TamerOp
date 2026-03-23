# AGENTS.md

Guidance for coding agents working in `tamer-op`.

## Core intent
- Optimize for performance and clean architecture.
- Keep APIs ergonomic for new users.
- Primary user base is mathematicians (not software developers); optimize naming, defaults, and docs/comments for mathematical readability first.
- Prefer clarity over compatibility shims (project is pre-release).
- Keep `Workflow` thin orchestration; place subsystem logic in dedicated files/modules.

## Project-specific principles
- No legacy-compatibility layers unless explicitly requested.
- No alias-heavy APIs in performance-critical surfaces.
- Keep canonical mode/value contracts strict and explicit.
- Avoid hidden behavior switches; backend/mode choice should be inspectable.
- Do not keep "legacy"/"shim" naming in production paths (`legacy_*`, alias symbols, old keyword bridges).
- Favor signature cleanup over overload accretion: remove convenience arities that do not add real capability.
- Keep internal plumbing internal (`_name` + non-export) when it is not part of the intended user surface.
- Internal function policy (practical):
  - Keep internal functions only if they do at least one of:
    - remove duplication in real call paths,
    - isolate hot-path logic for performance,
    - enforce invariants/contracts centrally,
    - provide reusable plumbing used in multiple places.
  - Remove internal functions that are:
    - dead (unused),
    - one-line wrappers with no semantic boundary,
    - internal knobs not consumed by runtime behavior,
    - diagnostic-only helpers that do not support active tests or critical debugging.
  - Do not keep "switchable dead paths":
    - an internal fast path that stays off by default because it is not a consistent win should be treated as temporary tuning scaffolding, not permanent library structure,
    - either retune/narrow it until it is the canonical default in its winning regime, or delete it,
    - do not leave long-lived off-by-default branches in production code merely because they can still be toggled manually.
  - Apply the same cleanup rule to thresholded/gated paths:
    - if a gate never activates on realistic workloads, or the activated regime does not benchmark as a clear win, remove the path rather than keeping dormant branch/plumbing complexity,
    - remove associated dead workspaces/caches/tests/bench lanes when a path is deleted.

## API policy
- Breaking API changes are acceptable when they improve clarity/performance.
- Still keep convenience defaults (users should not need to pass empty `Options()` everywhere).
- Prefer one canonical public path over multiple near-duplicate wrappers.
- For mathematical subsystems, make the canonical public path task-oriented:
  - users should ask for the mathematical object/result they want,
  - not assemble it from internal storage-shaped helpers.
- Treat `Workflow` as the main notebook-facing task surface for cross-owner operations.
  - High-level tasks such as change-of-poset workflows on `EncodingResult`, common-refinement translation, cheap resolution tables, and exact 2D distance belong on `Workflow` when they materially improve discoverability.
- Default public wrappers to the cheapest mathematically meaningful output.
  - Prefer dims/summaries/lazy views by default.
  - Force explicit opt-in for heavy objects (full bases, representatives, page terms, quotient data, etc.).
- Advanced workflows must still be easy to discover and use without deep software-engineering context.
- If adding new options/fields, thread them through all relevant call paths (do not silently drop).
- Prefer a single cache keyword surface in Workflow APIs; avoid multi-keyword cache controls.
- When a Workflow wrapper becomes the curated root binding for a name that also exists on an owner module, preserve the owner-native capability on the Workflow generic.
  - Example pattern: if `betti_table` or `matching_distance_exact_2d` are rebound through `Workflow`, keep forwarding methods for resolution objects / cache objects so the root surface does not lose advanced functionality.
- Avoid adding new public wrappers unless they provide clear UX or measurable performance benefit.
- Add public wrappers when they materially improve mathematical readability, standardize a subsystem surface, or keep simple users off storage-shaped internals.
- Standardize UX across analogous ingestion families (point cloud / graph / image): same option names and semantics where possible.
- Prefer named-keyword forms for mathematically indexed queries when positional arguments are easy to misread.
  - Example pattern: `term(ss; page=2, p=a, q=b)` rather than positional page/bidegree arguments.
  - Keep positional forms for internal/advanced hot paths only when they add real value.
- Prefer one notation convention per public subsystem and use it consistently in docs and helpers.
  - If a subsystem uses cohomology degree `t`, homology degree `s`, and spectral bidegree `(p,q)`, keep that convention everywhere user-facing.
- Avoid forcing users to inspect raw struct fields for ordinary workflows.
  - Prefer typed accessors (`dimensions`, `basis`, `coordinates`, `component`, `structure_map`, etc.) and semantic object accessors over field archaeology.
- Result/container objects that are meant to be user-visible should usually come with:
  - a compact `show`,
  - a structured `describe(...)`,
  - semantic accessors for the key mathematical pieces,
  - validation helpers where hand-built inputs are plausible.
- If a semantically meaningful component type appears inside a user-visible mathematical object, do not leave it half-polished.
  - Either keep it internal,
  - or keep it public and give it the same UX treatment (`show`, `describe`, semantic accessors, `check_*`).
- If a serialized/config boundary object is itself a common direct user object (for example `FiltrationSpec`), treat it like the rest of the UX surface:
  - give it `show`,
  - `describe(...)`,
  - an owner-local summary helper,
  - and explicit validation when malformed hand-built specs are plausible.
- For backend-heavy owner modules, keep the UX pass selective:
  - polish the genuinely user-facing mathematical objects and query workflows,
  - keep planners, caches, packed rows, bucket tables, and other internal machinery internal unless users are expected to inspect them directly.
- User-facing owner modules should usually provide an owner-local summary alias in addition to shared `describe(...)`.
  - Examples: `fringe_summary`, `flange_summary`, `zn_encoding_summary`, `pl_encoding_summary`, `box_encoding_summary`.
  - Prefer one obvious owner-local summary entrypoint per subsystem rather than many near-duplicate wrappers.
- For owner UX curation, prefer a clear split:
  - simple surface exposes the canonical task entrypoints plus cheap summary helpers,
  - advanced surface exposes heavier object types, validators, and detailed accessors.
- Treat shared `describe(...)` as one generic UX hook, not as a place for repeated owner-specific generic doc attachments.
  - Keep at most one generic-level doc on the shared `describe` function object.
  - Document owner-local summary aliases, user-visible result types, and specific `describe(::T)` methods instead of re-attaching generic-level docs in each owner.
- Validation helpers should be notebook-friendly and explicit.
  - Prefer `check_*` helpers with `throw=false` / `throw=true` over relying on downstream method or shape errors.
  - When a validation family is part of the user surface, also add a typed `*ValidationSummary` wrapper with compact `show`.
  - Validation helpers should report malformed hand-built storage/contracts as invalid summaries when possible rather than leaking incidental `BoundsError` / `DimensionMismatch` exceptions.
- Query-shaped subsystems should usually expose dedicated query validators.
  - Prefer explicit helpers like `check_*_point`, `check_*_points`, `check_*_box`, `check_*_region` when shape/box/index contracts are easy to misuse.
  - Validation reports should state when a finite box is required, when a query is outside the represented region set, and other mathematically important query contracts.
- For notebook/REPL ergonomics, add cheap scalar helpers before heavier materialization helpers.
  - Prefer helpers like `nregions`, `generator_counts`, `critical_coordinate_counts`, `resolution_length`, `matrix_size`, `axes_uniformity`, etc. when they answer common exploratory questions cheaply.
- When an owner-level encode/build routine is intended for direct use, prefer a typed owner-level result object over a raw tuple.
  - If the owner surface still returns raw tuples for now, treat that as UX debt rather than the desired end state.
- When a subsystem exposes both typed runtime objects and serialized/spec objects for the same concept, keep the cheap semantic accessors parallel across both surfaces when that is mathematically honest.
  - Example pattern: `filtration_kind`, `filtration_parameters`, and `construction_mode` should work on both typed filtrations and `FiltrationSpec`.
- Registry/extensibility surfaces should be inspectable in their own right.
  - Prefer helpers like `registered_*`, `*_family_summary`, and `check_*_spec` over forcing users to inspect raw schema tuples or internal registries.
- If a typed build-result wrapper becomes user-visible, give it the full UX treatment:
  - summary helper,
  - semantic accessors,
  - and a `check_*` validator when users might construct or edit it by hand.
- Do not add owner-local `pretty(...)` / tree-printer layers when typed `show`, `describe(...)`, and semantic accessors already cover the intended UX.
  - Keep deep tree printers only when they provide a distinct capability that users actually need.
  - If a pretty-printer is only used in tests/debugging after a UX pass, remove it rather than keeping a second inspection surface.
- At owner boundaries, keep canonical user-facing names local to the owning subsystem and translate explicitly when calling another owner.
  - Do not depend on another owner's `_internal` constants, `_internal` helper names, or internal field naming as part of a public/spec contract.
  - Example pattern: if one owner's public feature spec uses `:count` but another owner's internal summary object uses `:n_intervals`, map explicitly at the boundary.
- When a subsystem owns a typed options object for query/build contracts, thread user-facing controls through that object rather than forwarding ad hoc duplicate keywords into downstream internals.
  - Example pattern: pass `box` / `strict` through `InvariantOptions` when calling slice-compilation/query routines instead of restating them as direct compile kwargs.
- For fringe construction, keep strict canonical signatures (no legacy aliases):
  - `one_by_one_fringe(P, U, D, scalar; field=...)` (+ explicit mask variant),
  - `FringeModule{K}(P, U, D, phi; field=...)` with explicit `field`.
- Remove forwarding wrappers that only restate existing behavior and blur the encode vs pmodule boundary.

## Module organization policy
- Keep `src/TamerOp.jl` as include-order + API binding/export map only.
- `src/DataIngestion.jl` and `src/Featurizers.jl` are standalone sibling modules (`TamerOp.DataIngestion`, `TamerOp.Featurizers`), not nested under `Workflow`.
- Do not include `DataIngestion.jl`/`Featurizers.jl` from `Workflow.jl`; load them from `TamerOp.jl`.
- Keep shared workflow cache/session helper plumbing in `CoreModules` (not duplicated across Workflow/DataIngestion/Featurizers).
- Keep APISurface contracts authoritative; bind root/advanced symbols from `APISurface.jl` lists rather than ad-hoc `using`/`export` accretion.
- Remember that simple API bindings win first.
  - If a symbol already exists in advanced owner bindings and is promoted into the simple Workflow surface, the root binding will resolve to `Workflow`; adjust methods accordingly instead of assuming the old owner binding still reaches root users.
- Split by ownership/cohesion, not by line count alone:
  - if a file is one coherent subsystem that is just too large, keep one owner module in `src/<Owner>.jl` and move private implementation fragments into `src/<owner_snake_case>/`,
  - if a file mixes unrelated concepts, split those concepts into sibling owner modules instead of hiding them behind one catch-all owner.
- Prefer semantically honest owners over convenience buckets:
  - `FieldLinAlg` should remain the owner module for field-aware linear algebra, with private fragments under `src/field_linalg/`,
  - `FiniteFringe` should remain the owner module for the finite-poset / fringe / Hom / fiber subsystem, with private fragments under `src/finite_fringe/`,
  - `CoreModules` should contain only true low-level shared runtime, not unrelated data/options/results/stats concepts.
- For mixed cases, split in two stages:
  - first promote genuinely separate nested concepts into sibling owner modules,
  - then thin the remaining coherent owner into a private-folder split if readability still warrants it.
  - Current example: `IndicatorTypes` and `Encoding` are sibling owner modules; `FiniteFringe` remains the owner for the hot fringe subsystem.
- When using the owner-module + private-folder pattern:
  - keep the top-level owner file thin (module docstring, imports, ordered `include`s, public ownership),
  - keep private fragments non-exporting and scoped to that owner only,
  - do not partition further than the boundaries are operationally useful; coarse splits are preferred over premature micro-fragmentation.
- For `Serialization`, keep one public owner module in `src/Serialization.jl` and place private implementation fragments under `src/serialization/`.
  - Keep owned-format logic and external/adaptor logic in separate private fragments even when they stay under the same public owner.
- Do not create sibling public modules merely to reduce file size when the code still belongs to one coherent owner.
- Do not preemptively split large but still-coherent owner modules like `Modules.jl` or `AbelianCategories.jl`; split only when the maintenance/performance boundary is real.

## Performance policy
- Prioritize hot paths in:
  - `src/FieldLinAlg.jl`
  - `src/Invariants.jl`
  - `src/ZnEncoding.jl`
  - `src/PLBackend.jl`
  - `src/PLPolyhedra.jl`
  - `src/DerivedFunctors.jl`
  - `src/IndicatorResolutions.jl`
- Avoid `Any`, tuple-key dict churn, and repeated materialization in inner loops.
- Prefer typed arrays, linear-index caches, packed representations, and preallocated outputs.
- Prefer sparse-native operations; avoid densification unless explicitly justified.
- Use two-phase threaded design: sequential compile/build cache, threaded read-only compute.
- Threaded loops should write by deterministic index, not shared mutable dictionaries.
- In threaded kernels, avoid relying on `threadid()`-indexed mutable vectors for `push!` accumulation; prefer per-work-item shards plus deterministic reduction.
- Do not keep complexity that does not benchmark as a win; revert or simplify if gains are noise/negative.
- Treat cold and warm behavior separately; optimize both and report both.
- For new batched/vectorized kernels, keep a correctness-equivalent scalar fallback and gate activation by problem size/shape.
- Prefer conservative heuristic gates first; then tune thresholds with benchmark evidence (do not force fast paths globally).
- When a generic fast path is reused in a new owner kernel, give it an owner-specific gate if the cost regime differs; do not assume one subsystem's threshold transfers cleanly to another.
- When a subsystem has asymmetric dual families (e.g. upset/downset, projective/injective), tune and gate them separately by default; do not assume one side's thresholds transfer to the other.
- Keep fast-path switches inspectable with internal (non-exported) knobs so A/B checks are possible without invasive code edits.
- Fast-path knobs are for tuning, not indefinite coexistence.
  - If a path does not become a default-worthy win after benchmarking, remove it rather than keeping dormant branch complexity in production code.
  - If a path only wins in a narrower regime, split the implementation/gate by regime and enable it only there; do not keep a broad off-by-default branch for hypothetical future use.
- For threshold retunes, verify both sides of the gate explicitly:
  - benchmark a representative fixture that should stay off,
  - benchmark a representative fixture where the gate turns on,
  - keep the thresholded path only if the activated regime is a clear win.
- Validate performance across both cache and non-cache call paths; avoid optimizing one while regressing the other.
- Validate serial and threaded regimes separately; a threaded win can hide a serial regression (and vice versa).
- For batched query kernels, validate both uniform and skewed query distributions; a grouped/bucketed win on uniform clouds can regress badly on skewed buckets.
- For thread-sensitive heuristic gates, treat moderate-thread and high-thread regimes separately; a gate that is good at modest thread counts may need a stricter cap on high-thread runs.
- For sparse/functoriality kernels, benchmark both tiny and moderate-sparsity regimes before broadening a fast path; tiny fixtures alone are often a poor guide for threshold tuning.
- For cache-bearing encoding/translation kernels, optimize and benchmark all three regimes separately:
  - one-shot calls,
  - repeated labels within one call,
  - repeated calls on the same compiled/cached map.
- For cohomology/exact-algebra kernels, treat whole-complex and single-degree paths as separate performance surfaces; a lazy/full-data optimization can help one while regressing the other.
- Wire performance gains through canonical high-level entrypoints so simple users benefit without changing call patterns.

## Field and backend expectations
- Ground field support includes `QQ`, `F2`, `F3`, `Fp(p)`, and `RealField`.
- Keep field-generic behavior through `CoreModules` field APIs and `FieldLinAlg`.
- Preserve operation-specific backend selection (rank/nullspace/solve may choose differently).
- Maintain autotune + threshold persistence behavior (`linalg_thresholds.toml`) when touching linalg heuristics.
- The canonical linalg-threshold persistence file lives at repo root (`linalg_thresholds.toml`), not under `src/`.

## PL geometry mode contract
- Canonical PL mode is strict:
  - `:fast`
  - `:verified`
- Validate via `validate_pl_mode` in `CoreModules`.
- Do not reintroduce alias symbols (`:hybrid_fast`, `:hybrid_verified`, `:hybrid`, `:exact`) for mode selection.

## PL geometry cache contract
- Keep cache levels explicit and strict:
  - `:light` (locate/prefilter only),
  - `:geometry` (exact region geometry on-demand),
  - `:full` (precomputed heavy geometry/facets/adjacency support).
- Auto-cache paths should default to `:light` and promote by call intent; avoid eager full-cache work for one-off exact queries.
- Keep a one-off no-cache exact route for small/single-region queries when global cache build cost dominates.
- Ensure compiled/forwarded entrypoints preserve the same cache-level/intent behavior as direct owner-module calls.

## Data ingestion policy
- Primary ingestion implementation lives in `src/DataIngestion.jl`; featurizer API lives in `src/Featurizers.jl`.
- Keep typed filtration dispatch + `FiltrationSpec` conversion model:
  - typed filtrations for extensibility,
  - `FiltrationSpec` as canonical serialized boundary.
- Keep typed filtration UX and `FiltrationSpec` UX parallel when both are part of normal user workflows.
  - `FiltrationSpec` should be directly inspectable and validatable, not treated as an opaque transport object.
- Custom filtration registration is part of the user-facing extensibility surface.
  - Expose family-level inspection helpers for the registry (`registered_filtration_families`, family summaries, spec validators) so users do not need raw schema archaeology.
- For large point clouds, prefer sparse/edge-driven construction with explicit budgets:
  - `sparsify=:knn|:radius|:greedy_perm`,
  - strict `ConstructionBudget` checks (`max_edges`, `max_simplices`, memory).
- Prefer `nn_backend=:auto` for user-facing defaults:
  - route to NN extension when available,
  - deterministic fallback to bruteforce when unavailable.
- Keep hot sparse graph builders free of Dict churn in inner loops (use packed keys / typed buffers / deterministic finalize).
- Canonical in-memory storage for large raw datasets should be packed/columnar:
  - point clouds as dense coordinate matrices,
  - graph edges as columnar endpoint arrays,
  - hot consumers should use packed helpers rather than row-view convenience accessors.
- Prefer `SimplexTreeMulti` and graded/lazy stages for scale-sensitive paths; avoid eager full `|P|` expansion unless explicitly requested by output stage.
- Maintain preflight guardrails (`estimate_ingestion`) and early combinatorial checks before enumeration.
- Ingestion is keyword-contract heavy; provide explicit validators for stage/option/config surfaces when users may drive them from notebooks, JSON, or UI state.
  - Prefer helpers like `check_ingestion_stage`, `check_construction_options`, and `check_preflight_mode` over letting bad values fail deep inside planning/execution.
- Preserve multipers/RIVET comparison harness under `benchmark/ingestion_compare_harness/` and keep regimes apples-to-apples.

## Serialization policy
- Keep internal JSON save paths performance-first by default:
  - default to compact writes (`pretty=false`),
  - keep pretty-printing as an explicit opt-in only.
- Generic owned JSON save profiles are strict:
  - `:compact` is the default canonical profile,
  - `:debug` is the human-readable opt-in profile.
- Do not keep a fake generic `:portable` profile.
  - `:portable` is currently meaningful only for encoding-family artifacts,
  - keep it encoding-only unless another owned family gains a real portable contract with different emitted content.
- For structured posets, avoid dense relation emission by default:
  - use `include_leq=:auto`,
  - require explicit `include_leq=true` when dense `leq` materialization is actually needed.
- Keep strict load validation as the default contract; when needed, expose explicit trusted fast paths (e.g. `validate_masks=false`) for internally produced artifacts only.
- Keep the owned-format UX consistent:
  - inspect/summary first,
  - then `check_*`,
  - then strict `load_*`,
  - with trusted fast paths only as explicit opt-ins for internally produced artifacts.
- For large homogeneous datasets (point clouds, graphs, etc.), prefer columnar schemas + typed decoding on hot load paths; do not keep pre-release compatibility loaders for prior schemas unless explicitly requested.
- When downstream modules consume typed serialization summaries, use semantic accessors rather than raw field assumptions.
  - If `inspect_json(...)` or a similar surface graduates from raw tuples to a typed summary object, update downstream consumers to use the new accessors immediately.
- External/adaptor parsers should be canonical-only:
  - no alias keys,
  - no best-effort top-level fallbacks,
  - no legacy headers/shorthand forms once the canonical schema exists.
- Keep the owned/adaptor boundary explicit in code and docs.
  - Do not let external compatibility parsing blur the contract of owned JSON formats.
- In external/adaptor parsers, avoid per-entry warning spam in hot loops; aggregate repeated warnings to one warning per file/payload when possible.

## Testing policy
- Do **not** run the full suite unless explicitly requested by the user.
- Run focused tests for touched modules first (targeted harnesses are fine).
- Add/adjust targeted tests with each behavioral change, especially for:
  - mode/backends contracts
  - threaded vs serial parity
  - cache/no-cache parity
  - field parity
  - stage parity (`:simplex_tree`, `:graded_complex`, full pipeline where relevant)
- User expectation: tests must validate **algorithmic correctness**, not just smoke/regression/contracts.
  - Prefer hand-computable oracle fixtures where feasible.
  - If answers are known a priori, assert exact expected outputs.
  - For floating-point/RealField paths, use explicit tolerance-based oracle checks.
- Add negative-contract tests for new API contracts (`@test_throws` on invalid modes/options/shapes).
- Keep optional-dependency tests extension-aware and deterministic (skip only when package is missing).
- For heuristic-gated fast paths, add explicit parity tests that force both gate states (`off` vs `on`) and compare outputs.
- For geometry/invariant outputs with non-semantic ordering differences, canonicalize before comparison (stable sort + explicit rounding/tolerances).
- Include both tiny and moderate-size oracle tests when adding gates, so the gate boundary itself is validated.
- For threaded cache/build/read kernels, include `nthreads()>1` stress/parity tests that can catch write-contention/concurrency violations.
- Keep tests resilient to API-surface curation: correctness tests should call owner modules directly, not rely on curated binding layers.
- Keep shared test aliases centralized in `test/runtests.jl` when possible, and avoid shadowing those aliases with local variables inside test blocks.
- When running a single test file outside `test/runtests.jl`, reuse the centralized alias/helper prelude rather than inventing a divergent local bootstrap.
- For long owner-file runs in constrained environments, it is acceptable to run:
  - one field per fresh Julia process,
  - and, if necessary, one top-level `@testset` chunk at a time,
  provided the entire owner file is covered and field parity is preserved.
- When building a focused owner-test bootstrap from `test/runtests.jl`, stop before suite-global guard tests that are not part of the owner itself (for example ASCII-tree audits).
- For persistent cache surfaces, add lifecycle tests for:
  - lazy population on first use,
  - second-call reuse,
  - explicit cache clear/reset behavior when such a helper exists.
- For serialization/schema changes, add both:
  - strict canonical-contract tests (default path),
  - trusted fast-path parity tests where applicable.
- For removed legacy/compatibility surfaces, add negative tests that the old schema/aliases now throw.
- For module-level kernel changes (`Modules.jl`), include all of:
  - negative API-contract tests (`@test_throws` for invalid indices/comparability/shape conflicts),
  - `ModuleOptions` behavior parity tests,
  - direct non-QQ oracle tests (`F2`, `F3`, `Fp`, `RealField`),
  - randomized differential tests vs a naive oracle (chain + branching posets),
  - threaded contention parity tests for batched calls when `nthreads()>1`,
  - backendized-storage oracle checks (e.g. `BackendMatrix`/Nemo path when available),
  - lightweight perf-regression guard tests for core hot kernels.
- Remove temporary harness files after use.
- Keep test files coherent by subsystem; avoid creating one-off detached test files when an existing subsystem file is the right home.
- Avoid one-off test patterns used once only (e.g., bespoke skip helpers) when plain conditional tests are clearer.
- For UX/API work, add targeted tests for:
  - canonical public wrappers and their default outputs,
  - named-keyword and positional parity when both exist,
  - accessor parity vs underlying objects,
  - `describe(...)`/`show` shape on key result objects,
  - validation helper success/failure paths with `throw=false` and `throw=true`,
  - owner-local summary helpers,
  - compiled/cache-bearing parity when helpers are intended to accept wrapped objects,
  - advanced-binding exposure when the helper is intended to be discoverable there.

## Benchmark policy
- For performance work, update or add microbenchmarks under `benchmark/`.
- Keep benchmarks focused and reproducible (warmup + comparable cases).
- Report speed deltas and note fidelity/correctness tradeoffs, if any.
- When a speedup targets a specific workload shape, add that workload as a first-class benchmark lane rather than relying on a nearby proxy benchmark.
  - Examples:
    - neighboring-box walks for `ZnEncoding.pmodule_on_box(...)`,
    - boundary-heavy grouped locate batches for `PLPolyhedra`,
    - slab-heavy batched queries for `FlangeZn`.
- If introducing a new fast path, add a benchmark block that isolates that path (e.g., tiny-query/short-chain cases), and compare against a meaningful baseline.
- For heuristic-gated algebra/functoriality kernels, include at least:
  - one tiny fixture that should stay on the fallback path,
  - one moderate-sparsity fixture that is a plausible fast-path regime,
  - explicit threshold-comparison rows when retuning the gate.
- For thresholded cleanup decisions, keep the benchmark honest:
  - if the activated regime does not finish or does not clearly beat forced-off, delete the path rather than preserving speculative complexity.
- For cacheable translation/encoding kernels, add explicit benchmark lanes for:
  - one-shot calls,
  - repeated-label workloads,
  - repeated-call workloads on the same cached object.
- For ingestion and backend work, report both:
  - subsystem microbench deltas,
  - workflow-level impact (`encode(...)` path) when applicable.
- Keep external comparisons (e.g. multipers) regime-matched; explicitly label directional vs apples-to-apples results.
- Benchmark scripts should resolve symbols from owner modules (e.g. `CoreModules`, `RegionGeometry`) rather than curated API bindings.
- Audit manual source bootstraps in `benchmark/` whenever module ownership changes:
  - partial `include(...)` bootstraps are a real churn surface,
  - moving a nested module to a sibling owner can break those scripts even when package-mode loading is fine.
- Prefer built-in A/B probes in the same benchmark script (toggle-based before/after) for deterministic local comparisons.
- Report allocations alongside runtime; GC/allocation regressions are first-class performance regressions.
- For geometry locate kernels, benchmark both:
  - uniform/random query clouds,
  - boundary-heavy or candidate-heavy batches,
  because exact-membership reuse can be invisible on random clouds but decisive on boundary-heavy workloads.
- For serialization benchmarks, report file-size deltas alongside runtime/allocations; schema and pretty/compact choices are performance features.
- State timing policy explicitly in benchmark outputs (strict-cold vs warm-uncached vs warm-cached) and do not mix them silently.
- Before performance runs, ensure no stale heavy Julia jobs are competing for CPU/RAM; process contention can masquerade as regressions or crashes.
- For large architectural refactors, validate the owner module directly first (parse/smoke load/focused benchmark) before trusting root-module timings from a noisy environment.
- For noisy kernels, run at least two independent benchmark passes and keep both raw outputs before calling wins/regressions.
- Prefer benchmark harnesses with sectional switches (e.g. run only encoding or only dataset blocks) so long runs remain operable in constrained/dev environments.

### Benchmark run checklist (operational)
1. Ensure a clean benchmark environment:
   - stop stale heavy Julia processes,
   - avoid background benchmark/test runs,
   - note machine/thread settings used.
2. Declare timing policy up front:
   - strict-cold, warm-uncached, or warm-cached,
   - use the same policy for all compared variants/tools.
3. Run the baseline first and persist raw outputs (CSV/text) with clear filenames.
4. Run the candidate with identical inputs/flags, then compute explicit deltas:
   - median runtime ratio,
   - allocation/GC delta.
5. Validate correctness parity after performance changes:
   - targeted oracle/parity tests,
   - gate off/on parity when heuristic gates are involved.
6. Check both cache and non-cache paths, and serial vs threaded paths when relevant.
7. Record conclusions with caveats:
   - where wins occur,
   - where regressions/noise remain,
   - next tuning target (if any).

## Code hygiene
- Keep `src/TamerOp.jl` as an API map (avoid dumping implementation helpers there).
- Place logic in the most specific module that owns it.
- Internal-only functions must always use a leading underscore and remain non-exported.
- Internal performance tuning controls (feature flags, thresholds, gates) should remain non-exported and documented near the owning kernel.
- In owner modules, do not define methods specialized on a concrete type before that type is defined in the file; method/type order matters for direct source loading and benchmark harnesses.
- When splitting an owner module across private include files, give the owner file a module docstring and give each fragment a short header comment/docstring stating what that file owns and its scope.
- Keep comments concise and technical; avoid stale docs.
- Keep `TamerOp.jl` comments synchronized with real structure (no stale references like `PublicAPI.jl`/`AdvancedAPI.jl` when those files do not exist).
- If a constant/config appears dead, remove it or wire it fully.
- When deleting a dead or legacy path, also scrub stale docstrings/comments/benchmark labels so the removed route is not mentioned anywhere user- or maintainer-facing.
- Prefer explicit, canonical names over aliases in all new code.
- Prefer semantic accessor names over exposing raw storage fields as the de facto user API.
- When a user-visible struct becomes important enough for notebook/REPL use, add:
  - object docstring,
  - compact `show`,
  - structured `describe(...)`,
  - semantic accessors,
  - validation helper if users may construct it by hand.
- Typed report/validation wrappers that are part of the user surface should usually have structural equality semantics (`==`, `isequal`, `hash`) so tests and notebooks can compare them meaningfully.
- When tightening API contracts, update tests/examples/docs in the same sweep so user-facing guidance stays consistent.
- Keep submodule files free of `export` blocks; only top-level `TamerOp.jl` should export curated API symbols.

## Examples and docs policy
- Examples are onboarding assets: optimize for didactic clarity over showcasing every internal stage.
- Examples should be notebook-first when the primary user workflow is exploratory/visual.
  - Treat `.ipynb` as the canonical form for user-facing visualization and ingestion tutorials.
  - Add a `.jl` companion only when it provides clear extra value for automation, CI-style execution, or deterministic artifact generation.
  - Do not keep parallel notebook/script examples in sync by default if the script is not serving a distinct purpose.
  - Current policy example: visualization tutorials `10`, `11`, and `12` are notebook-first and notebook-only unless a real non-notebook use case reappears.
- Example notebooks should serve as both:
  - a didactic tutorial for one coherent workflow, and
  - a curated showcase of what the subsystem can do nearby.
- Keep one clear narrative spine per notebook.
  - Teach the canonical task from start to finish without forcing users to infer the intended workflow.
- For tutorial notebooks, keep the cell granularity fine and explicit.
  - Prefer one mathematical idea or UX concept per code cell.
  - If a cell introduces a new conceptual distinction (for example type vs `kind`, summary vs validation, or spec vs export), give that distinction its own short markdown cell near the first use.
  - Prefer a slightly longer notebook with clean conceptual steps over denser expert-style batching.
  - Do not hide canonical workflow steps behind notebook-local helper wrappers unless teaching the helper is itself part of the lesson.
  - In particular, prefer explicit public calls for visualization and export (`visual_spec(...)`, `visual_summary(...)`, `save_visuals(...)`) over notebook-scoped convenience functions when the notebook is meant to teach those actions.
- Around that spine, include a small number of deliberately chosen side views that expose adjacent capability.
  - Good pattern: show the main workflow first, then add “other useful views” or “what else this object supports” cells that use the same object/result.
  - Avoid turning notebooks into a grab-bag of unrelated features.
- Be explicit about what is canonical versus exploratory.
  - Label cells/sections so users can tell which calls are the main recommended workflow and which are broader showcase material.
- In notebook examples, prefer the canonical public workflow entrypoint even when an owner-level helper is shorter.
  - If a top-level workflow exists, use that in the main narrative spine.
  - Example: prefer `encode(data, filtration; stage=:graded_complex)` over `DataIngestion.build_graded_complex(...)` in user-facing notebooks.
  - Example: prefer `encode(Ups, Downs)` over `PLBackend.encode_fringe_boxes(...)` in user-facing notebooks, then use semantic accessors like `encoding_poset(...)`, `encoding_map(...)`, and `encoding_module(...)` instead of unpacking raw owner tuples.
  - Use owner-level builders/executors only in notebooks that are explicitly about that owner module or advanced internals.
- End example notebooks with a short recap when useful.
  - Summarize what mathematical task was completed and what additional capabilities the user just saw.
- Prefer one coherent story per file with clear section headers and minimal control-flow clutter.
- Use comments to explain *semantics* (what object is returned, what key kwargs mean, what mathematical object is being computed), not just mechanics.
- For beginner examples, explicitly annotate return-object semantics (e.g., what `EncodingResult` contains and what key kwargs like `degree` mean).
- Keep beginner examples on canonical public entrypoints; reserve internal/plumbing paths for advanced examples only.
- Public docstrings should be detailed enough that REPL help is genuinely useful:
  - include signatures,
  - describe the mathematical meaning of the function/object,
  - state best-practice usage,
  - point users to the cheap/default path first,
  - explain when heavier outputs/materialization are appropriate.
- For query/encoding/geometry helpers, docstrings should also state the concrete query contract.
  - Explain accepted point/matrix/box/region shapes,
  - state whether finite boxes are required,
  - document sentinel meanings such as `0` from `locate`,
  - and say when direct lookup or cache-bearing paths are present.
- For result objects and categorical/algebraic containers, docstrings should answer:
  - what mathematical object this represents,
  - what the canonical accessors are,
  - what `describe(...)`/`dimensions(...)`/related helpers return.
- Owner-local summary helpers should usually include a short doc example showing the intended cheap-first workflow.
  - Typical pattern: construct object -> `describe(...)` / `*_summary(...)` -> validate with `check_*` -> run one query -> inspect a region/result summary.
- Prefer cheap-first exploration helpers in docs and examples.
  - Show `page_dims(...)` before `page_terms(...)`,
  - summaries before full representatives,
  - validation helpers before manual debugging of malformed hand-built inputs.

## Extensions and optional deps
- Keep optional ecosystem integrations in `ext/` (Tables, Arrow, CSV, NPZ, Parquet2, Folds, ProgressLogging, etc.).
- Core API should provide clear actionable errors if an extension-backed format/backend is requested but unavailable.

## Workflow cache contract
- Top-level Workflow entrypoints should expose one cache contract:
  - `cache=:auto` for per-call automatic caching.
  - `cache=sc::SessionCache` for explicit cross-call reuse.
- Do not expose `session_cache` as a parallel public keyword in top-level Workflow entrypoints.
- Specialized caches (`ResolutionCache`, `HomSystemCache`, `SlicePlanCache`) should be derived internally from the session cache path, not required from simple users.
- When a workflow task depends on heavy reusable geometry or planning state, derive that reuse from the session cache internally instead of exposing a second public cache surface.
  - Example: exact 2D matching should reuse arrangements / slice families through workflow geometry caches behind `cache=sc`.
- Tests and examples should use `cache=sc` (not `session_cache=sc`) when demonstrating reuse.
- Reuse expectation should be preserved across calls like:
  - `enc = encode(...; cache=sc)`
  - `E = ext(enc1, enc2; cache=sc)` (and similarly `tor`, `resolve`, `invariant`, `slice_barcodes`, `mp_landscape`).

## Test guardrails for API structure
- Keep a regression test that enforces no `export` statements outside `src/TamerOp.jl` (currently in `test/runtests.jl`).
- When a symbol is promoted into the simple Workflow surface, add targeted API tests for:
  - root binding exposure,
  - parity against the owner-level implementation,
  - negative contract paths that were previously enforced only at the owner level.

## Doc discoverability
- Owner modules should normally carry their own module docstring in the owner file.
- If `@doc TamerOp.<Owner>` does not reliably surface that narrative through the umbrella binding, attach an explicit binding doc in `src/TamerOp.jl`.
- Use that umbrella-binding doc fix selectively for discoverability problems; do not replace normal owner-file docstrings with root-binding docs by default.

## When uncertain
- Choose the path that is:
  1. faster in hot workloads,
  2. cleaner/smaller in API surface,
  3. stricter in contracts,
  4. easier to benchmark and test.
