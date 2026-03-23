# =============================================================================
# Example 90 (optional): Indicator resolutions and Ext diagnostics
#
# Theme
# -----
# "The advanced algebraic layer is available when you want homological
# diagnostics, while keeping the same Workflow cache contract."
#
# This example is intentionally separate from the main onboarding track.
# It demonstrates how to:
#   - compute projective/injective resolutions,
#   - compute low-degree Ext dimensions,
#   - reuse SessionCache across repeated derived-functor calls.
# =============================================================================

include(joinpath(@__DIR__, "_common.jl"))

example_header(
    "90",
    "Advanced: resolutions and Ext diagnostics";
    theme="Optional algebraic workflow built on top of encoded modules.",
    teaches=[
        "How to run resolve(...) from a standard EncodingResult",
        "How to compute Ext^t(M, M) dimensions for small t",
        "How SessionCache reuse percolates into derived-functor calls",
    ],
)

Random.seed!(20260217)
outdir = example_outdir("90_advanced_ext")

CM = TO.CoreModules
OPT = TO.Options
DT = TO.DataTypes
EC = TO.EncodingCore
RES = TO.Results
FF = TO.FiniteFringe
IR = TO.IndicatorResolutions

stage("1) Build a tiny finite-fringe module and wrap as EncodingResult")

# Chain poset with two elements: 1 <= 2.
leq = Bool[
    true  true
    false true
]
P = FF.FinitePoset(leq; check=false)

field = CM.QQField()
K = CM.coeff_type(field)

S1 = FF.one_by_one_fringe(
    P,
    FF.principal_upset(P, 1),
    FF.principal_downset(P, 1),
    one(K);
    field=field,
)

M = IR.pmodule_from_fringe(S1)
enc = RES.EncodingResult(P, M, nothing; H=S1, opts=OPT.EncodingOptions(field=S1.field), backend=:example)
println("Encoded poset vertices: ", enc.P.n)

stage("2) Resolve (projective + injective)")

sc = CM.SessionCache()
ropts = TO.ResolutionOptions(maxlen=2, minimal=false, check=true)

res_proj = TO.resolve(enc; kind=:projective, opts=ropts, cache=sc)
res_inj = TO.resolve(enc; kind=:injective, opts=ropts, cache=sc)

println("Projective resolution type: ", typeof(res_proj.res))
println("Injective resolution type: ", typeof(res_inj.res))
println("Projective Betti table available: ", res_proj.betti !== nothing)
println("Injective Bass table available: ", haskey(res_inj.meta, :bass))

stage("3) Compute Ext dimensions and show cache reuse")

E1 = TO.ext(enc, enc; maxdeg=2, model=:projective, cache=sc)
E2 = TO.ext(enc, enc; maxdeg=2, model=:projective, cache=sc)

# With shared SessionCache, repeated calls should reuse the cached resolution path.
println("Ext cache reuse (E1.res === E2.res): ", E1.res === E2.res)

ext_dims = [TO.dim(E1, t) for t in 0:2]
println("Ext dimensions [t=0,1,2]: ", ext_dims)

stage("4) Save a compact advanced report")

report_path = joinpath(outdir, "advanced_ext_report.txt")
open(report_path, "w") do io
    println(io, "example=90_advanced_indicator_resolutions_and_ext")
    println(io, "timestamp_utc=", Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS"))
    println(io, "poset_vertices=", enc.P.n)
    println(io, "projective_resolution_type=", typeof(res_proj.res))
    println(io, "injective_resolution_type=", typeof(res_inj.res))
    println(io, "ext_dims=", join(string.(ext_dims), ","))
    println(io, "ext_cache_reuse=", E1.res === E2.res)
end
println("Saved report: ", report_path)

println("\nDone. Output directory: ", outdir)
