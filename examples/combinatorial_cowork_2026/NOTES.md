# Speaker Notes — Combinatorial CoWork 2026

This file is speaker-only. It is not part of the participant-facing notebook flow.

## Pacing target

- `00_kernel_smoke_and_workflow_contract.ipynb`: 5 min
- `01_presentations_to_encodings.ipynb`: 10 min
- `02_resolutions_hom_ext_tor.ipynb`: 15 min
- `03_change_of_posets_and_common_refinement.ipynb`: 15 min
- `04_invariants_and_visual_payoff.ipynb`: 10 min
- buffer / Q&A: 5 min
- optional notebook `05_noisy_annulus_rips_codensity_h1_2persistence.ipynb`: only as a point-cloud follow-up

Keep the hour centered on notebooks `02` and `03`. Those carry the algebraic payoff for this audience.

## Global advice

- Start every notebook from a fresh Julia kernel if time permits.
- Run the setup cell first, then pause and confirm the audience sees:
  - `TO`
  - `sc = SessionCache()`
  - one deterministic synthetic source object
- Prefer `describe(...)`, `result_summary(...)`, `betti_table(...)`, `bass_table(...)`, and small scalar queries over raw field inspection.
- Keep the export cells short: run them, show the returned paths, then move on.
- The notebooks mostly export lightweight HTML visual-spec artifacts by default, so no render-toggle setup is needed for the talk.
- Notebook `05` is the one exception: it renders PNGs through CairoMakie because the snapshot panel is a custom figure.

## Cell-skipping policy

If the room machine is slow or compilation drags:

- still do notebook `00`
- keep notebook `01` short
- do notebooks `02` and `03` in full
- in notebook `04`, show only the first invariant block and one image/landscape block
- skip notebook `05` unless the audience explicitly wants a point-cloud / multipers-style detour
- skip notebook `90` entirely unless Q&A asks for it

## Notebook-by-notebook notes

### `00_kernel_smoke_and_workflow_contract.ipynb`

Goal:
- establish the workflow contract once:
  - synthetic source
  - `encode(...)`
  - `EncodingResult`
  - `SessionCache`
  - one invariant

If time is tight:
- run the export cell but do not discuss it beyond the returned paths
- show only the encoding summary and the rank-invariant cell

### `01_presentations_to_encodings.ipynb`

Goal:
- make the provenance point clearly:
  - box-fringe source
  - staircase source
  - PL fringe
  - flange
  - same `encode(...)` workflow surface

Live emphasis:
- do not explain every constructor in detail
- explain why the sources differ mathematically, then show that the workflow contract does not

If time is tight:
- compare only one box-style source and one non-box source
- keep only one visual summary cell

### `02_resolutions_hom_ext_tor.ipynb`

Goal:
- this is the first major notebook
- show that after encoding, standard commutative-algebra style queries are natural:
  - resolutions
  - Betti/Bass tables
  - `Hom`
  - `Ext`
  - `Tor`

Live emphasis:
- `betti_table(...)` and `bass_table(...)` first
- then `hom_dimension(...)`
- then `ext(...)`
- mention that the `Tor` cell uses the valid opposite-poset setup on purpose

If time is tight:
- skip the representative/coordinates round-trip cell
- keep the exercise cell as spoken homework rather than executing it live

### `03_change_of_posets_and_common_refinement.ipynb`

Goal:
- second major notebook
- this is where functoriality becomes concrete

Live emphasis:
- first `common_refinement(...)`
- then `hom(...; transport=:common_refinement)`
- then `restriction(...)`
- then one pushforward comparison

Advanced aside:
- keep the explicit `EncodingMap` cell short and clearly labeled as the one advanced detour

If time is tight:
- do not dwell on both derived pushforward cells
- show one and summarize the other

### `04_invariants_and_visual_payoff.ipynb`

Goal:
- close the story by treating invariants as summaries of algebra already seen

Live emphasis:
- `rank_invariant`
- `restricted_hilbert`
- `euler_surface`
- one signed-measure block
- one `mpp_image` / `mp_landscape` block

If time is tight:
- skip `matching_distance_exact_2d`
- run the export cell but do not discuss it

### `05_noisy_annulus_rips_codensity_h1_2persistence.ipynb`

Goal:
- give a concrete point-cloud-side codensity example that connects the workshop visuals back to a familiar multipers-style picture

Live emphasis:
- the cloud is genuinely noisy, including interior points
- the snapshot panel is notebook-local scaffolding, not a new library API
- the canonical library object is still the `stage=:cohomology_dims` result and its `H^1` support plane

If time is tight:
- show the raw colored point cloud and the final `H^1` support plane
- skip the full codensity/radius snapshot panel discussion

### `90_speaker_appendix_exact_queries_and_extra_derived.ipynb`

Use only if:
- the audience asks about exact 2D reuse,
- or someone asks what lies beyond `Ext` / `Tor` in the current workflow.

Do not open this during the main hour unless you are ahead of schedule.

## Recommended live order inside the hour

1. notebook `00` quickly
2. notebook `01` selectively
3. notebook `02` carefully
4. notebook `03` carefully
5. notebook `04` as payoff
6. notebook `05` only for a point-cloud / multipers-style extension
7. notebook `90` only in Q&A

## Environment reminders

The safe default for the talk is:

```bash
julia --project=.
```

and then run the notebooks as written. The default export cells already produce lightweight HTML artifacts without requiring Makie setup.
