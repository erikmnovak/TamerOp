# Combinatorial CoWork 2026 Tutorial Suite

This subfolder contains the notebook-only tutorial material for a 1-hour
hands-on presentation aimed at commutative algebraists who are comfortable with
computation.

The suite uses one narrative throughout:

synthetic presentation -> encode once -> do algebra -> transport across posets ->
use invariants and visuals as summaries of that algebra.

## Core notebooks

1. `00_kernel_smoke_and_workflow_contract.ipynb`
   - 5 min target
   - environment check + core workflow contract
2. `01_presentations_to_encodings.ipynb`
   - 10 min target
   - synthetic source families and their common `encode(...)` path
3. `02_resolutions_hom_ext_tor.ipynb`
   - 15 min target
   - common encodings, resolutions, Betti/Bass tables, `Hom`, `Ext`, `Tor`
4. `03_change_of_posets_and_common_refinement.ipynb`
   - 15 min target
   - common refinement, restriction, pushforwards, derived pushforwards
5. `04_invariants_and_visual_payoff.ipynb`
   - 10 min target
   - invariant, signed-measure, image, landscape, and exact-distance summaries

Buffer / Q&A: 5 minutes.

## Optional extensions

- `05_noisy_annulus_rips_codensity_h1_2persistence.ipynb`
  - optional point-cloud follow-up
  - noisy annulus with interior noise, codensity/radius snapshot panel, and a degree-1 support-plane picture for a `RipsCodensityFiltration`

## Optional appendix

- `90_speaker_appendix_exact_queries_and_extra_derived.ipynb`
  - keep this for speaker-side exploration or follow-up discussion; it is not
    part of the live hour.

## Runtime and workflow conventions

- All notebooks use synthetic data only; there are no external datasets.
- All notebooks are standalone but sequential: each one rebuilds its own tiny
  examples instead of relying on hidden state from previous notebooks.
- Default field choice is `QQField()` to match the workshop audience.
- The notebooks are designed to stay within a 1-3 minute runtime budget on a
  participant laptop.
- The canonical cache contract is used throughout:
  - `cache=:auto` for one-shot work,
  - `cache=sc::SessionCache` for repeated queries inside one notebook.
- Visuals are notebook-first:
  - explicit visual-spec summaries appear inline in the notebooks,
  - each notebook also includes a guaranteed `save_visuals(...)` export cell.
- Most export cells write lightweight HTML visual-spec artifacts to
  `examples/_outputs/combinatorial_cowork_2026/<notebook>/`.
- Notebook `05_noisy_annulus_rips_codensity_h1_2persistence.ipynb` is the one
  exception: it exports PNGs because the codensity/radius panel is a custom
  CairoMakie figure rather than a library-owned visual spec.
- If you want heavier rendered figures elsewhere, edit the export cell locally
  to request a different `format` / `backend`.

## UX contract reference

The library is designed so that ordinary notebook work does not require field
archaeology or internal storage knowledge. Across the tutorial suite, six UX
surfaces appear repeatedly. Participants should learn these as the main
inspection/validation vocabulary.

### 1. `describe(...)`

`describe(...)` is the generic cross-library inspection hook.

Use it when you want a first answer to:

- what object did this computation return?
- what are its main mathematical sizes, ranges, or dimensions?
- what is the cheapest useful summary of this object?

This is the best default habit for notebook work because it is uniform across
owners. If a participant only remembers one inspection tool, it should be
`describe(...)`.

Typical examples in the suite:

- `describe(box)` for a synthetic source object
- `describe(enc)` for an `EncodingResult`
- `describe(H)` for a `Hom` object
- `describe(E)` for `Ext`
- `describe(mpp)` for an image-style invariant

The design intent is:

- users should not need to remember owner-specific summary names first,
- users should be able to ask for an inspection summary before they know the
  exact runtime type they are holding.

### 2. Owner-local `*_summary(...)`

Owner-local summary helpers are the subsystem-native inspection surface.

Examples include:

- `box_fringe_summary(...)`
- `result_summary(...)`
- other owner-specific `*_summary(...)` helpers in the broader library

These are useful when the user already knows which subsystem they are working
in and wants the owner's preferred vocabulary. They often present the same
object a bit more locally and mathematically than generic `describe(...)`.

Practical rule:

- use `describe(...)` by default,
- use `*_summary(...)` when you are working deeply within one owner and want
  that owner's canonical local summary.

In this workshop, `*_summary(...)` is present to show that local subsystem
surfaces exist, but the main habit we teach is still `describe(...)`.

### 3. `check_*(...; throw=false/true)`

`check_*` helpers are explicit validators.

They answer a different question from `describe(...)`:

- `describe(...)` asks: what do I have?
- `check_*(...)` asks: is it valid?

These helpers matter most when a user:

- hand-constructs an object,
- edits one interactively in a notebook,
- or receives data from a less-trusted boundary and wants an explicit contract
  check before running heavier computations.

Typical examples:

- `check_synthetic_box_fringe(...)`
- subsystem-specific query validators elsewhere in the library

In contrast, objects produced by trusted constructors usually do not need
immediate validation in routine use, but the validator exists when a notebook
needs an explicit correctness check.

### 4. Semantic accessors

Semantic accessors are the main replacement for raw field inspection.

Instead of asking users to remember internal field names, the library prefers
named accessors that expose the key mathematical pieces directly.

Examples include:

- `encoding_map(...)`
- `encoding_poset(...)`
- `common_poset(...)`
- `projection_maps(...)`
- `translated_module(...)`
- `translated_modules(...)`
- `betti_table(...)`
- `bass_table(...)`
- `export_path(...)`
- `export_stem(...)`

These accessors matter because many user-visible objects are wrappers or result
containers. The accessor tells the user what the relevant mathematical
component is without forcing them to inspect struct fields directly.

Practical rule:

- if there is a semantic accessor, prefer it over field access in notebooks.

### 5. Compact `show`

Compact `show` is the automatic display behavior in the REPL or notebook.

This is the surface users meet before they call anything explicitly. A good
compact display should:

- identify the kind of object,
- show the most important sizes or state briefly,
- avoid dumping storage-heavy internals,
- encourage the next inspection step rather than overwhelming the user.

For user-visible result and container objects, compact `show` is part of the
UX contract, not an incidental convenience. It should make a notebook output
cell readable even when the user simply leaves the object as the final
expression.

In practice, compact `show` and `describe(...)` work together:

- `show` gives the quick glance,
- `describe(...)` gives the structured explanation.

### 6. Typed result objects and typed validation summaries

The library prefers typed result/container objects over raw tuples.

Examples include:

- `EncodingResult`
- `ResolutionResult`
- `InvariantResult`
- `ModuleTranslationResult`
- `CommonRefinementTranslationResult`
- `VisualExportResult`

When validation itself is part of the user surface, the library also prefers
typed validation-summary wrappers rather than bare booleans or ad hoc strings.

Why this matters:

- typed result objects preserve provenance,
- they make semantic accessors possible,
- they support compact `show`,
- they support `describe(...)`,
- they keep notebook outputs stable and inspectable,
- they avoid storage-shaped tuple contracts leaking into user workflows.

This is especially important in `Workflow`, where the goal is to let users ask
for mathematical tasks directly and receive inspectable mathematical objects in
return.

## How to use these six surfaces in practice

A good notebook workflow is:

1. build or compute an object,
2. inspect it with `describe(...)`,
3. use owner-local `*_summary(...)` only when you want subsystem-native detail,
4. use semantic accessors to pull out the key mathematical pieces,
5. run `check_*(...)` only when you need an explicit validation step,
6. rely on compact `show` and typed result objects to keep ordinary output
   readable.

That is the inspection pattern used throughout this tutorial suite.

## Suggested prep order for the speaker

1. Run notebook 00 once to confirm the Julia kernel and environment.
2. Rehearse notebooks 02 and 03 twice; they carry the densest algebraic story.
3. Keep notebook 04 open for the closing payoff and optional exact-distance demo.
4. Use notebook 90 only if time remains or the audience pushes deeper.

## Source material reused from the main examples directory

- `examples/13_synthetic_families_and_coupled_fringes.*`
- `examples/zn2_flange.jl`
- `examples/90_advanced_indicator_resolutions_and_ext.jl`
- `examples/10_visualization_engine_basics.ipynb`
