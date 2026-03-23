# =============================================================================
# Example 13: Synthetic families and diagonal-vs-coupled fringe visuals
#
# Theme
# -----
# "Build parameter sweeps from the new coupled synthetic generators, then render
# one direct comparison between a diagonal box fringe and a coupled staircase
# box fringe."
# =============================================================================

include(joinpath(@__DIR__, "_common.jl"))

example_header(
    "13",
    "Synthetic families and coupled fringe visuals";
    theme="Use sweep_family(...) for coupled synthetic generators, then compare diagonal and coupled fringe geometry/rank visuals.",
    teaches=[
        "How to build SyntheticFamily sweeps from the new coupled generators",
        "How to inspect family members before doing any heavier encoding work",
        "How to compare a diagonal synthetic fringe and a coupled synthetic fringe with saved visuals",
    ],
)

outdir = example_outdir("13_synthetic_families_and_coupled_fringes")

const SD = TO.SyntheticData
const FZ = TO.FlangeZn
const TOA = TO.Advanced

function maybe_save_visual(outdir::AbstractString,
                           stem::AbstractString,
                           obj;
                           kind::Symbol,
                           enabled::Bool=false,
                           kwargs...)
    enabled || return nothing
    path = joinpath(outdir, stem * ".html")
    TO.save_visual(path, obj; kind=kind, kwargs...)
    return (; path=path, format=:html)
end

function print_family_items(title::AbstractString,
                            fam::SD.SyntheticFamily,
                            describe_item::Function)
    println(title, ": ", SD.synthetic_family_summary(fam))
    for (label, item) in zip(SD.synthetic_labels(fam), SD.synthetic_items(fam))
        println("  - ", label, " -> ", describe_item(item))
    end
end

stage("1) Build sweep families from the new coupled generators")

chain_bars_a = [(1, 4), (2, 5), (3, 6)]
chain_bars_b = [(1, 5), (3, 6), (4, 7)]
chain_bars_c = [(1, 6), (2, 6), (3, 7)]

chain_family = SD.sweep_family(
    SD.coupled_chain_fringe;
    sweep=(bars=[chain_bars_a, chain_bars_b, chain_bars_c],),
    n=7,
    generator=:coupled_chain_sweep,
)

stairs_a = [([0.0, 0.0], [1.5, 1.0]), ([0.75, 0.5], [2.25, 1.75]), ([1.5, 1.0], [3.0, 2.5])]
stairs_b = [([0.0, 0.0], [2.0, 1.25]), ([0.75, 0.5], [2.75, 2.25]), ([1.5, 1.0], [3.75, 3.25])]

box_family = SD.sweep_family(
    SD.staircase_box_fringe;
    sweep=(bars=[stairs_a, stairs_b],),
    generator=:staircase_box_sweep,
)

phi_default = [1 1 0; 1 0 1; 0 1 1]
phi_sparse = [1 1 0; 0 1 1; 0 0 1]

flange_family = SD.sweep_family(
    SD.mixed_face_flange;
    sweep=(phi=[phi_default, phi_sparse],),
    generator=:mixed_face_flange_sweep,
)

print_family_items("Coupled chain family", chain_family, M -> TO.describe(M))
print_family_items("Staircase box family", box_family, B -> SD.box_fringe_summary(B))
print_family_items(
    "Mixed-face flange family",
    flange_family,
    FG -> (; summary=TO.describe(FG), dim_at_2_2=FZ.dim_at(FG, (2, 2))),
)

stage("2) Compare one diagonal box fringe and one coupled staircase box fringe")

diagonal_boxes = [
    ([0.0, 0.0], [1.0, 1.0]),
    ([1.4, 0.2], [2.4, 1.2]),
    ([2.7, 0.8], [3.7, 1.8]),
]

box_diag = SD.box_bar_fringe(bars=diagonal_boxes)
box_coupled = SD.staircase_box_fringe(bars=stairs_b)

println("Diagonal box summary: ", SD.box_fringe_summary(box_diag))
println("Coupled box summary:  ", SD.box_fringe_summary(box_coupled))

enc_diag = TO.encode(box_diag)
enc_coupled = TO.encode(box_coupled)

pi_diag = TO.encoding_map(enc_diag)
pi_coupled = TO.encoding_map(enc_coupled)
rank_diag = TO.rank_invariant(enc_diag)
rank_coupled = TO.rank_invariant(enc_coupled)

println("Diagonal encoding summary: ", TO.describe(enc_diag))
println("Coupled encoding summary:  ", TO.describe(enc_coupled))
println("Diagonal rank visuals: ", TO.available_visuals(rank_diag))
println("Coupled rank visuals:  ", TO.available_visuals(rank_coupled))

stage("3) Build inspectable visual specs")

regions_diag_spec = TOA.visual_spec(pi_diag; kind=:regions)
regions_coupled_spec = TOA.visual_spec(pi_coupled; kind=:regions)
rank_diag_spec = TOA.visual_spec(rank_diag; kind=:rank_heatmap)
rank_coupled_spec = TOA.visual_spec(rank_coupled; kind=:rank_heatmap)

println("Diagonal regions visual summary: ", TOA.visual_summary(regions_diag_spec))
println("Coupled regions visual summary:  ", TOA.visual_summary(regions_coupled_spec))
println("Diagonal rank visual summary:    ", TOA.visual_summary(rank_diag_spec))
println("Coupled rank visual summary:     ", TOA.visual_summary(rank_coupled_spec))

render_artifacts = get(ENV, "TO_EXAMPLE_RENDER", "0") == "1"
if render_artifacts
    println("TO_EXAMPLE_RENDER=1, so renderer-backed HTML exports are enabled.")
else
    println("Skipping renderer-backed exports by default. Set TO_EXAMPLE_RENDER=1 to save HTML visuals too.")
end

regions_diag = maybe_save_visual(outdir, "diagonal_box_regions", pi_diag; kind=:regions, enabled=render_artifacts)
regions_coupled = maybe_save_visual(outdir, "coupled_box_regions", pi_coupled; kind=:regions, enabled=render_artifacts)
rank_diag_path = maybe_save_visual(outdir, "diagonal_box_rank_heatmap", rank_diag; kind=:rank_heatmap, enabled=render_artifacts)
rank_coupled_path = maybe_save_visual(outdir, "coupled_box_rank_heatmap", rank_coupled; kind=:rank_heatmap, enabled=render_artifacts)

stage("4) Save a compact report")

report_path = joinpath(outdir, "synthetic_coupled_fringes_report.txt")
open(report_path, "w") do io
    println(io, "example=13_synthetic_families_and_coupled_fringes")
    println(io, "chain_family_summary=", SD.synthetic_family_summary(chain_family))
    println(io, "box_family_summary=", SD.synthetic_family_summary(box_family))
    println(io, "flange_family_summary=", SD.synthetic_family_summary(flange_family))
    println(io, "diagonal_box_summary=", SD.box_fringe_summary(box_diag))
    println(io, "coupled_box_summary=", SD.box_fringe_summary(box_coupled))
    println(io, "diagonal_encoding_summary=", TO.describe(enc_diag))
    println(io, "coupled_encoding_summary=", TO.describe(enc_coupled))
    println(io, "diagonal_rank_summary=", TO.describe(rank_diag))
    println(io, "coupled_rank_summary=", TO.describe(rank_coupled))
    println(io, "diagonal_regions_visual_summary=", TOA.visual_summary(regions_diag_spec))
    println(io, "coupled_regions_visual_summary=", TOA.visual_summary(regions_coupled_spec))
    println(io, "diagonal_rank_visual_summary=", TOA.visual_summary(rank_diag_spec))
    println(io, "coupled_rank_visual_summary=", TOA.visual_summary(rank_coupled_spec))
    println(io, "diagonal_regions_visual_artifact=", regions_diag)
    println(io, "coupled_regions_visual_artifact=", regions_coupled)
    println(io, "diagonal_rank_visual_artifact=", rank_diag_path)
    println(io, "coupled_rank_visual_artifact=", rank_coupled_path)
    println(io, "diagonal_poset_vertices=", TO.FiniteFringe.nvertices(TO.encoding_poset(enc_diag)))
    println(io, "coupled_poset_vertices=", TO.FiniteFringe.nvertices(TO.encoding_poset(enc_coupled)))
end

println("Saved report: ", report_path)
println("\nDone. Output directory: ", outdir)
