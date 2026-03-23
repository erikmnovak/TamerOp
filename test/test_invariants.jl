using Test
using Random
using JSON3
import Base.Threads

function _median_elapsed(f::Function; warmup::Int=1, reps::Int=5)
    for _ in 1:warmup
        f()
    end
    ts = Vector{Float64}(undef, reps)
    for i in 1:reps
        ts[i] = @elapsed f()
    end
    sort!(ts)
    return ts[cld(reps, 2)]
end

struct ToyPi <: TO.EncodingCore.PLikeEncodingMap end
EC.dimension(::ToyPi) = 1
function EC.locate(::ToyPi, x::AbstractVector)
    length(x) == 1 || error("ToyPi expects a 1D point")
    return round(Int, float(x[1]))
end
function EC.locate(::ToyPi, x::NTuple{1,<:Real})
    return round(Int, float(x[1]))
end

struct ToyPi1DThresholds <: TO.EncodingCore.PLikeEncodingMap end
EC.dimension(::ToyPi1DThresholds) = 1
EC.representatives(::ToyPi1DThresholds) = [(0.5,), (1.5,), (2.5,)]
EC.axes_from_encoding(::ToyPi1DThresholds) = ([0.0, 1.0, 2.0, 3.0],)
function EC.locate(::ToyPi1DThresholds, x::AbstractVector)
    length(x) == 1 || error("ToyPi1DThresholds expects a 1D point")
    t = float(x[1])
    if t < 1.0
        return 1
    elseif t < 2.0
        return 2
    elseif t < 3.0
        return 3
    else
        return 0
    end
end

const SI = TamerOp.SliceInvariants
function EC.locate(::ToyPi1DThresholds, x::NTuple{1,<:Real})
    t = float(x[1])
    if t < 1.0
        return 1
    elseif t < 2.0
        return 2
    elseif t < 3.0
        return 3
    else
        return 0
    end
end

struct ToyPi1DIntervals <: TO.EncodingCore.PLikeEncodingMap end
EC.dimension(::ToyPi1DIntervals) = 1
EC.representatives(::ToyPi1DIntervals) = [(0.5,), (1.5,), (2.5,)]
EC.axes_from_encoding(::ToyPi1DIntervals) = ([0.0, 1.0, 2.0, 3.0],)
function EC.locate(::ToyPi1DIntervals, x::AbstractVector)
    t = x[1]
    if 0.0 <= t < 1.0
        return 1
    elseif 1.0 <= t < 2.0
        return 2
    elseif 2.0 <= t < 3.0
        return 3
    else
        return 0
    end
end
function EC.locate(::ToyPi1DIntervals, x::NTuple{1,<:Real})
    t = x[1]
    if 0.0 <= t < 1.0
        return 1
    elseif 1.0 <= t < 2.0
        return 2
    elseif 2.0 <= t < 3.0
        return 3
    else
        return 0
    end
end

struct ToyPi2D <: TO.EncodingCore.PLikeEncodingMap end
EC.dimension(::ToyPi2D) = 2
function EC.locate(::ToyPi2D, x::AbstractVector)
    length(x) == 2 || error("ToyPi2D expects a 2D point")
    eps = 1e-9
    i = floor(Int, float(x[1]) + eps)
    j = floor(Int, float(x[2]) + eps)
    return 1 + i + 10 * j
end

struct ToyBoxes2D <: TO.EncodingCore.PLikeEncodingMap
    coords::NTuple{2,Vector{Float64}}
    reps::Vector{NTuple{2,Float64}}
end

EC.dimension(::ToyBoxes2D) = 2
EC.representatives(pi::ToyBoxes2D) = pi.reps
EC.axes_from_encoding(pi::ToyBoxes2D) = pi.coords
function EC.locate(pi::ToyBoxes2D, x::AbstractVector{<:Real}; strict::Bool=true, closure::Bool=true)
    x1 = x[1]
    x2 = x[2]
    if (x1 < 0.0) || (x1 > 3.0) || (x2 < 0.0) || (x2 > 3.0)
        return 0
    elseif x1 < 1.0
        return 1
    elseif x1 < 2.0
        return 2
    else
        return 3
    end
end
function EC.locate(pi::ToyBoxes2D, x::NTuple{2,<:Real}; strict::Bool=true, closure::Bool=true)
    x1 = x[1]
    x2 = x[2]
    if (x1 < 0.0) || (x1 > 3.0) || (x2 < 0.0) || (x2 > 3.0)
        return 0
    elseif x1 < 1.0
        return 1
    elseif x1 < 2.0
        return 2
    else
        return 3
    end
end

with_fields(FIELDS_FULL) do field
K = CM.coeff_type(field)
@inline cf(x) = CM.coerce(field, x)

@testset "PLikeEncodingMap dispatch hook" begin
    @test TO.ZnEncoding.ZnEncodingMap <: TO.EncodingCore.PLikeEncodingMap
    @test TO.PLPolyhedra.PLEncodingMap    <: TO.EncodingCore.PLikeEncodingMap

    # PLBackend is optional at load time; only test if present.
    if isdefined(TamerOp, :PLBackend)
        @test TamerOp.PLBackend.PLEncodingMapBoxes <: TO.EncodingCore.PLikeEncodingMap
    end
end

@testset "Finite-encoding invariants: rank and restricted Hilbert" begin
    P = chain_poset(3)
    MD._clear_cover_cache!(P)

    # Interval module supported on {2,3} for the chain 1 < 2 < 3.
    H23 = one_by_one_fringe(P, FF.principal_upset(P, 2), FF.principal_downset(P, 3), cf(1); field=field)
    M23 = IR.pmodule_from_fringe(H23)

    @test Inv.rank_map(M23, 2, 3) == 1
    @test Inv.rank_map(M23, 1, 1) == 0
    @test Inv.rank_map(M23, 1, 2) == 0

    rinv = TO.rank_invariant(M23)
    @test rinv isa Inv.RankInvariantResult
    @test length(rinv) == 3
    @test rinv[(2, 2)] == 1
    @test rinv[(2, 3)] == 1
    @test rinv[(3, 3)] == 1
    @test !haskey(rinv, (1, 1))
    @test Inv.nentries(rinv) == 3
    @test Inv.value_at(rinv, 1, 1) == 0
    @test Inv.value_at(rinv, 2, 3) == 1
    @test !Inv.store_zeros(rinv)
    @test Inv.source_poset(rinv) === P
    @test describe(rinv).kind == :rank_invariant

    rinv_all = TO.rank_invariant(M23; store_zeros=true)
    @test rinv_all isa Inv.RankInvariantResult
    @test length(rinv_all) == 6
    @test rinv_all[(1, 1)] == 0
    @test Inv.store_zeros(rinv_all)
    @test length(Inv.nonzero_pairs(rinv_all)) == 3

    # Noncomparable pair should error.
    Pd = diamond_poset()
    Hd2 = one_by_one_fringe(Pd, FF.principal_upset(Pd, 2), FF.principal_downset(Pd, 2); field=field)
    Md2 = IR.pmodule_from_fringe(Hd2)
    @test_throws ErrorException Inv.rank_map(Md2, 2, 3)

    if Threads.nthreads() > 1
        MD._clear_cover_cache!(P)
        rinv_serial = TO.rank_invariant(M23; threads = false)
        rinv_thread = TO.rank_invariant(M23; threads = true)
        @test rinv_thread == rinv_serial

        rinv_all_serial = TO.rank_invariant(M23; store_zeros = true, threads = false)
        rinv_all_thread = TO.rank_invariant(M23; store_zeros = true, threads = true)
        @test rinv_all_thread == rinv_all_serial
    end
    
end


@testset "Hilbert distances" begin
    P = chain_poset(3)

    H23 = one_by_one_fringe(P, FF.principal_upset(P, 2), FF.principal_downset(P, 3); field=field)
    M23 = IR.pmodule_from_fringe(H23)

    H3 = one_by_one_fringe(P, FF.principal_upset(P, 3), FF.principal_downset(P, 3); field=field)
    M3 = IR.pmodule_from_fringe(H3)

    @test TO.restricted_hilbert(M23) == [0, 1, 1]
    @test TO.restricted_hilbert(M3) == [0, 0, 1]

    @test Inv.hilbert_distance(M23, M3; norm=:L1) == 1
    @test Inv.hilbert_distance(M23, M3; norm=:Linf) == 1
    @test isapprox(Inv.hilbert_distance(M23, M3; norm=:L2), 1.0)

    w = [2, 3, 5]
    @test Inv.hilbert_distance(M23, M3; norm=:L1, weights=w) == 3
    @test Inv.hilbert_distance(M23, M3; norm=:Linf, weights=w) == 3
end

@testset "Euler characteristic surface and Euler signed-measure" begin
    P = chain_poset(3)

    # Interval module supported on {2,3} for chain 1<2<3.
    H23 = one_by_one_fringe(P, FF.principal_upset(P, 2), FF.principal_downset(P, 3), cf(1); field=field)
    M23 = IR.pmodule_from_fringe(H23)

    # Module supported on {3}.
    H3  = one_by_one_fringe(P, FF.principal_upset(P, 3), FF.principal_downset(P, 3), cf(1); field=field)
    M3  = IR.pmodule_from_fringe(H3)

    pi = ToyPi()
    axes = ([1,2,3],)
    opts_axes = TO.InvariantOptions(axes=axes, axes_policy=:as_given)

    # Euler surface of a module should equal its restricted Hilbert values on this encoding.
    surf23 = TO.euler_surface(M23, pi, opts_axes)
    @test surf23 == reshape([0,1,1], 3)

    if Threads.nthreads() > 1
        opts_serial = TO.InvariantOptions(axes=axes, axes_policy=:as_given, threads=false)
        opts_thread = TO.InvariantOptions(axes=axes, axes_policy=:as_given, threads=true)
        surf_serial = TO.euler_surface(M23, pi, opts_serial)
        surf_thread = TO.euler_surface(M23, pi, opts_thread)
        @test surf_thread == surf_serial
    end

    # Mobius inversion in 1D is discrete derivative: weights [0,1,0].
    pm23   = SM.euler_signed_measure(M23, pi, opts_axes)
    @test length(pm23) == 1
    @test pm23.inds[1] == (2,)
    @test pm23.wts[1]  == 1

    # Reconstruction from point measure recovers surface.
    rec23 = SM.surface_from_point_signed_measure(pm23)
    @test rec23 == surf23

    # Test raw point_signed_measure on a known surface.
    surf = reshape([0,1,1], 3)
    pm = SM.point_signed_measure(surf, axes; drop_zeros=true)
    @test length(pm) == 1
    @test pm.inds[1] == (2,)
    @test pm.wts[1] == 1
    @test SM.surface_from_point_signed_measure(pm) == surf

    surf2 = reshape([2,2,1,3], 4)
    axes2 = (collect(1.0:1.0:4.0),)
    pm2 = SM.point_signed_measure(surf2, axes2; drop_zeros=true)
    @test pm2.inds == [(1,), (3,), (4,)]
    @test pm2.wts == [2, -1, 2]
    @test SM.surface_from_point_signed_measure(pm2) == surf2

    # Euler distance between M23 and M3 on this grid.
    surf3 = TO.euler_surface(M3, pi, opts_axes)
    @test surf3 == reshape([0,0,1], 3)
    @test SM.euler_distance(surf23, surf3; ord=1) == 1
    @test SM.euler_distance(M23, M3, pi, opts_axes; ord=Inf) == 1.0

    # Euler for a 2-term cochain complex: chi = dim(C^0) - dim(C^1) (tmin=0)
    C = TO.ModuleCochainComplex([M23, M3], [TO.zero_morphism(M23, M3)]; tmin=0)
    surfC = TO.euler_surface(C, pi, opts_axes)
    # reuse the opts already defined earlier in this testset
    @test surfC == reshape([0,1,0], 3)

    pmC = SM.euler_signed_measure(C, pi, opts_axes)
    # Mobius derivative of [0,1,0] is [0,1,-1] (drop_zeros keeps 2 points)
    @test length(pmC) == 2
    @test pmC.inds[1] == (2,)
    @test pmC.wts[1]  == 1
    @test pmC.inds[2] == (3,)
    @test pmC.wts[2]  == -1
    @test SM.surface_from_point_signed_measure(pmC) == surfC

    # mma_decomposition integration: generic Euler-only front-end.
    outM = SM.mma_decomposition(M23, pi, opts_axes; method=:euler)
    @test outM isa SM.SignedMeasureDecomposition
    @test SM.has_euler_surface(outM)
    @test SM.has_euler_signed_measure(outM)
    @test TO.describe(outM).kind == :signed_measure_decomposition
    @test outM.euler_surface == surf23
    @test SM.surface_from_point_signed_measure(outM.euler_signed_measure) == surf23

    outC = SM.mma_decomposition(C, pi, opts_axes; method=:euler)
    @test outC isa SM.SignedMeasureDecomposition
    @test SM.has_euler_surface(outC)
    @test SM.has_euler_signed_measure(outC)
    @test outC.euler_surface == surfC
    @test SM.surface_from_point_signed_measure(outC.euler_signed_measure) == surfC

    # The generic signature only supports method=:euler.
    @test_throws ArgumentError SM.mma_decomposition(M23, pi, opts_axes; method=:all)
end


@testset "Slice restrictions and 1D barcodes" begin
    P = chain_poset(3)

    H23 = one_by_one_fringe(P, FF.principal_upset(P, 2), FF.principal_downset(P, 3); field=field)
    M23 = IR.pmodule_from_fringe(H23)

    H3 = one_by_one_fringe(P, FF.principal_upset(P, 3), FF.principal_downset(P, 3); field=field)
    M3 = IR.pmodule_from_fringe(H3)

    chain = [1, 2, 3]

    b23 = TO.slice_barcode(M23, chain)
    @test b23 == Dict((2, 4) => 1)

    b3 = TO.slice_barcode(M3, chain)
    @test b3 == Dict((3, 4) => 1)

    pb23 = TamerOp.SliceInvariants._slice_barcode_packed(M23, chain; values=nothing)
    @test pb23 isa TamerOp.SliceInvariants.PackedIndexBarcode
    @test TamerOp.SliceInvariants._barcode_from_packed(pb23) == b23

    pb23t = TamerOp.SliceInvariants._slice_barcode_packed(M23, chain; values=[0.0, 1.0, 2.0, 3.0])
    @test pb23t isa TamerOp.SliceInvariants.PackedFloatBarcode
    @test TamerOp.SliceInvariants._barcode_from_packed(pb23t) == Dict((1.0, 3.0) => 1)

    b0 = Dict{Tuple{Int,Int},Int}()

    @test Inv.bottleneck_distance(b23, b23) == 0.0
    @test Inv.bottleneck_distance(b23, b3) == 1.0
    @test isapprox(Inv.bottleneck_distance(b3, b0), 0.5)

    # Approx matching distance over a single slice is just the bottleneck distance.
    @test Inv.matching_distance_approx(M23, M3, [chain]) == 1.0

    # Approx matching distance over a single slice is just the bottleneck distance.
    @test Inv.matching_distance_approx(M23, M3, [chain]) == 1.0

    # Restriction to a chain should produce a 1D module with the expected dims.
    Mc = Inv.restrict_to_chain(M23, chain)
    @test Mc.dims == [0, 1, 1]
    @test Inv.rank_map(Mc, 2, 3) == 1

end

@testset "Compiled slice plan parity + cache reuse" begin
    P = chain_poset(3)
    H23 = one_by_one_fringe(P, FF.principal_upset(P, 2), FF.principal_downset(P, 3); field=field)
    M23 = IR.pmodule_from_fringe(H23)
    H3 = one_by_one_fringe(P, FF.principal_upset(P, 3), FF.principal_downset(P, 3); field=field)
    M3 = IR.pmodule_from_fringe(H3)

    pi = ToyPi1DThresholds()
    dirs = [[1.0]]
    offs = [[0.0]]

    cache = Inv.SlicePlanCache()
    plan1 = Inv.compile_slice_plan(pi;
                                  directions=dirs, offsets=offs,
                                  tmin=0.0, tmax=3.0, nsteps=121,
                                  threads=false, cache=cache)
    plan2 = Inv.compile_slice_plan(pi;
                                  directions=dirs, offsets=offs,
                                  tmin=0.0, tmax=3.0, nsteps=121,
                                  threads=false, cache=cache)
    @test plan1 === plan2

    plan_api = Inv.compile_slices(pi, TO.InvariantOptions();
                                 directions=dirs, offsets=offs,
                                 tmin=0.0, tmax=3.0, nsteps=121,
                                 threads=false, cache=nothing)
    @test plan_api.nd == plan1.nd
    @test plan_api.no == plan1.no

    Inv.clear_slice_module_cache!()
    cacheM = Inv.module_cache(M23)
    @test cacheM === Inv.module_cache(M23)
    @test isempty(cacheM.packed_plan_barcodes)

    data_plan = TO.slice_barcodes(M23, plan1; packed=true, threads=false)
    @test data_plan isa SI.SliceBarcodesResult
    @test TO.describe(data_plan).kind == :slice_barcodes_result
    @test size(data_plan.barcodes) == (1, 1)
    @test data_plan.barcodes isa SI.PackedBarcodeGrid{SI.PackedFloatBarcode}
    @test SI.slice_barcodes(data_plan) === data_plan.barcodes
    @test SI.slice_weights(data_plan) === data_plan.weights
    @test SI.slice_directions(data_plan) === data_plan.dirs
    @test SI.slice_offsets(data_plan) === data_plan.offs
    @test SI.packed_barcodes(data_plan) === data_plan.barcodes
    @test occursin("SliceBarcodesResult", sprint(show, data_plan))
    data_plan_uncached = SI._slice_barcodes_plan_packed_uncached(M23, plan1; threads=false)
    @test SI._barcode_from_packed(data_plan_uncached[1, 1]) ==
          SI._barcode_from_packed(data_plan.barcodes[1, 1])
    @test length(cacheM.packed_plan_barcodes) == 1
    data_plan_run = Inv.run_invariants(plan1, cacheM, Inv.SliceBarcodesTask(; packed=true, threads=false))
    data_plan_run2 = Inv.run_invariants(plan1, cacheM, Inv.SliceBarcodesTask(; packed=true, threads=false))
    @test data_plan_run.barcodes === data_plan.barcodes
    @test data_plan_run2.barcodes === data_plan.barcodes
    @test SI._barcode_from_packed(data_plan_run.barcodes[1, 1]) ==
          SI._barcode_from_packed(data_plan.barcodes[1, 1])

    # Build explicit slice specs from the compiled plan and check exact parity.
    slices = NamedTuple{(:chain, :values, :weight),Tuple{Vector{Int},Vector{Float64},Float64}}[]
    for i in 1:plan1.nd, j in 1:plan1.no
        idx = (i - 1) * plan1.no + j
        s = plan1.vals_start[idx]
        l = plan1.vals_len[idx]
        vals = l == 0 ? Float64[] : collect(@view(plan1.vals_pool[s:s + l - 1]))
        push!(slices, (chain = plan1.chains[idx], values = vals, weight = plan1.weights[i, j]))
    end
    data_explicit = TO.slice_barcodes(M23, slices; packed=true, threads=false)
    @test data_explicit isa SI.SliceBarcodesResult
    @test SI._barcode_from_packed(data_plan.barcodes[1, 1]) ==
          SI._barcode_from_packed(data_explicit.barcodes[1])
    @test isapprox(data_plan.weights[1, 1], data_explicit.weights[1]; atol=1e-12)

    # Typed SliceSpec collection helpers (plan + generator + JSON round-trip).
    typed_t = TO.collect_slices(plan1; values=:t)
    @test eltype(typed_t) == TO.SliceSpec{Float64,Vector{Float64}}
    data_typed = TO.slice_barcodes(M23, typed_t; packed=true, threads=false)
    @test SI._barcode_from_packed(data_typed.barcodes[1]) ==
          SI._barcode_from_packed(data_explicit.barcodes[1])
    @test isapprox(data_typed.weights[1], data_explicit.weights[1]; atol=1e-12)

    typed_index = TO.collect_slices(plan1; values=:index)
    @test eltype(typed_index) == TO.SliceSpec{Float64,Nothing}
    data_typed_idx = TO.slice_barcodes(M23, typed_index; packed=true, threads=false)
    @test data_typed_idx.barcodes[1] isa SI.PackedIndexBarcode

    typed_from_generator = TO.collect_slices((c for c in plan1.chains))
    @test eltype(typed_from_generator) == TO.SliceSpec{Float64,Nothing}
    @test length(typed_from_generator) == length(plan1.chains)

    spec_ux = IC.SliceSpec([1, 3, 4]; values=[0.0, 0.5, 1.0], weight=0.25)
    spec_desc = TO.describe(spec_ux)
    @test spec_desc.kind == :slice_spec
    @test spec_desc.chain_length == 3
    @test spec_desc.values_mode == :real
    @test spec_desc.values_length == 3
    @test isapprox(spec_desc.weight, 0.25; atol=1e-12)
    @test IC.describe(spec_ux) == spec_desc
    @test IC.chain(spec_ux) == [1, 3, 4]
    @test Base.values(spec_ux) == [0.0, 0.5, 1.0]
    @test isapprox(IC.weight(spec_ux), 0.25; atol=1e-12)
    @test occursin("SliceSpec(", sprint(show, spec_ux))
    @test occursin("chain_length", sprint(show, MIME"text/plain"(), spec_ux))
    @test IC.check_slice_spec(spec_ux).valid
    @test occursin("check_slice_spec", string(@doc TamerOp.InvariantCore.SliceSpec))
    @test occursin("inspectable object", lowercase(string(@doc TamerOp.ChainComplexes.describe)))

    spec_bad = IC.SliceSpec{Float64,Vector{Float64}}([1, 2], [0.0], -1.0)
    spec_bad_report = IC.check_slice_spec(spec_bad)
    @test !spec_bad_report.valid
    @test !isempty(spec_bad_report.issues)
    @test_throws ArgumentError IC.check_slice_spec(spec_bad; throw=true)

    tmpd = mktempdir()
    slices_path = joinpath(tmpd, "slice_specs.json")
    TO.save_slices_json(slices_path, typed_t)
    loaded_slices = TO.load_slices_json(slices_path)
    @test eltype(loaded_slices) == TO.SliceSpec{Float64,Vector{Float64}}
    data_loaded = TO.slice_barcodes(M23, loaded_slices; packed=true, threads=false)
    @test SI._barcode_from_packed(data_loaded.barcodes[1]) ==
          SI._barcode_from_packed(data_typed.barcodes[1])
    @test isapprox(data_loaded.weights[1], data_typed.weights[1]; atol=1e-12)

    f_plan = Inv.slice_features(M23, plan1; featurizer=:summary, threads=false)
    f_explicit = Inv.slice_features(M23, typed_t; featurizer=:summary, threads=false)
    @test f_plan isa SI.SliceFeaturesResult
    @test f_explicit isa SI.SliceFeaturesResult
    @test Inv.slice_features(f_plan) == Inv.slice_features(f_explicit)
    @test sprint(show, f_plan) |> x -> occursin("SliceFeaturesResult", x)
    @test length(cacheM.packed_plan_barcodes) == 1

    tgrid_land = collect(range(0.0, 3.0; length=9))
    old_landscape_cache = SI._SLICE_USE_LANDSCAPE_FEATURE_CACHE[]
    try
        SI._SLICE_USE_LANDSCAPE_FEATURE_CACHE[] = false
        empty!(cacheM.landscape_plan_features)
        f_land_off = Inv.slice_features(M23, plan1;
                                        featurizer=:landscape,
                                        kmax=2,
                                        tgrid=tgrid_land,
                                        threads=false)
        @test isempty(cacheM.landscape_plan_features)

        SI._SLICE_USE_LANDSCAPE_FEATURE_CACHE[] = true
        f_land_on = Inv.slice_features(M23, plan1;
                                       featurizer=:landscape,
                                       kmax=2,
                                       tgrid=tgrid_land,
                                       threads=false)
        @test isapprox.(Inv.slice_features(f_land_on), Inv.slice_features(f_land_off); atol=1e-12) |> all
        @test length(cacheM.landscape_plan_features) == 1

        f_land_on2 = Inv.slice_features(M23, plan1;
                                        featurizer=:landscape,
                                        kmax=2,
                                        tgrid=tgrid_land,
                                        threads=false)
        @test isapprox.(Inv.slice_features(f_land_on2), Inv.slice_features(f_land_on); atol=1e-12) |> all
        @test length(cacheM.landscape_plan_features) == 1
    finally
        SI._SLICE_USE_LANDSCAPE_FEATURE_CACHE[] = old_landscape_cache
    end

    k_plan = Inv.slice_kernel(M23, M3, pi;
                             directions=dirs, offsets=offs,
                             tmin=0.0, tmax=3.0, nsteps=121,
                             kind=:bottleneck_gaussian, sigma=1.0,
                             threads=false)
    k_explicit = Inv.slice_kernel(M23, M3, slices;
                                 kind=:bottleneck_gaussian, sigma=1.0,
                                 normalize_weights=true,
                                 threads=false)
    @test isapprox(k_plan, k_explicit; atol=1e-12)

    k_run = Inv.run_invariants(
        plan1,
        Inv.module_cache(M23, M3),
        Inv.SliceKernelTask(; kind=:bottleneck_gaussian, sigma=1.0, threads=false),
    )
    @test isapprox(k_run, k_plan; atol=1e-12)

    pair_cache = Inv.module_cache(M23, M3)
    data_plan_M3 = TO.slice_barcodes(M3, plan1; packed=true, threads=false)
    countA_before = length(pair_cache.A.packed_plan_barcodes)
    countB_before = length(pair_cache.B.packed_plan_barcodes)
    @test countA_before >= 1
    @test countB_before >= 1
    d_ref = Inv.bottleneck_distance(data_plan.barcodes[1, 1], data_plan_M3.barcodes[1, 1])
    d_run = Inv.run_invariants(
        plan1,
        pair_cache,
        Inv.SliceDistanceTask(; dist_fn=Inv.bottleneck_distance, dist_kwargs=NamedTuple(), threads=false),
    )
    d_run2 = Inv.run_invariants(
        plan1,
        pair_cache,
        Inv.SliceDistanceTask(; dist_fn=Inv.bottleneck_distance, dist_kwargs=NamedTuple(), threads=false),
    )
    @test isapprox(d_run, d_ref; atol=1e-12)
    @test isapprox(d_run2, d_ref; atol=1e-12)
    @test length(pair_cache.A.packed_plan_barcodes) == countA_before
    @test length(pair_cache.B.packed_plan_barcodes) == countB_before

    old_landscape_cache = SI._SLICE_USE_LANDSCAPE_FEATURE_CACHE[]
    try
        SI._SLICE_USE_LANDSCAPE_FEATURE_CACHE[] = false
        empty!(pair_cache.A.landscape_plan_features)
        empty!(pair_cache.B.landscape_plan_features)
        k_land_off = Inv.slice_kernel(M23, M3, plan1;
                                      kind=:landscape_linear,
                                      tgrid=tgrid_land,
                                      kmax=2,
                                      threads=false)
        @test isempty(pair_cache.A.landscape_plan_features)
        @test isempty(pair_cache.B.landscape_plan_features)

        SI._SLICE_USE_LANDSCAPE_FEATURE_CACHE[] = true
        k_land_on = Inv.slice_kernel(M23, M3, plan1;
                                     kind=:landscape_linear,
                                     tgrid=tgrid_land,
                                     kmax=2,
                                     threads=false)
        @test isapprox(k_land_on, k_land_off; atol=1e-12)
        @test length(pair_cache.A.landscape_plan_features) == 1
        @test length(pair_cache.B.landscape_plan_features) == 1
    finally
        SI._SLICE_USE_LANDSCAPE_FEATURE_CACHE[] = old_landscape_cache
    end

    alloc_plan = @allocated begin
        p = Inv.compile_slice_plan(pi;
                                  directions=dirs, offsets=offs,
                                  tmin=0.0, tmax=3.0, nsteps=121,
                                  threads=false, cache=nothing)
        TO.slice_barcodes(M23, p; packed=true, threads=false)
    end
    @test alloc_plan < 3_000_000

    # Runtime budgets (warmup + median over deterministic tiny fixture).
    t_slice_plan = _median_elapsed(; warmup=1, reps=5) do
        TO.slice_barcodes(M23, plan1; packed=true, threads=false)
    end
    @test t_slice_plan < 0.75

    task_dist = Inv.SliceDistanceTask(; dist_fn=Inv.bottleneck_distance, dist_kwargs=NamedTuple(), threads=false)
    t_dist_plan = _median_elapsed(; warmup=1, reps=5) do
        Inv.run_invariants(plan1, Inv.module_cache(M23, M3), task_dist)
    end
    @test t_dist_plan < 0.75

    task_kernel = Inv.SliceKernelTask(; kind=:bottleneck_gaussian, sigma=1.0, threads=false)
    t_kernel_plan = _median_elapsed(; warmup=1, reps=5) do
        Inv.run_invariants(plan1, Inv.module_cache(M23, M3), task_kernel)
    end
    @test t_kernel_plan < 0.75
end

@testset "SliceInvariants UX surface" begin
    P = chain_poset(3)
    H23 = one_by_one_fringe(P, FF.principal_upset(P, 2), FF.principal_downset(P, 3); field=field)
    M23 = IR.pmodule_from_fringe(H23)
    H3 = one_by_one_fringe(P, FF.principal_upset(P, 3), FF.principal_downset(P, 3); field=field)
    M3 = IR.pmodule_from_fringe(H3)

    bar = Dict((0.0, 1.0) => 1, (0.25, 0.75) => 1)
    pl = SI.persistence_landscape(bar; kmax=2, tgrid=[0.0, 0.5, 1.0])
    img = SI.persistence_image(bar; xgrid=[0.0, 0.5, 1.0], ygrid=[0.0, 0.5, 1.0], sigma=0.2)

    pi = ToyPi1DThresholds()
    opts = TO.InvariantOptions(box=([0.0], [3.0]))
    plan_cache = SI.SlicePlanCache()
    plan = SI.compile_slice_plan(
        pi,
        opts;
        directions=[[1.0]],
        offsets=[[0.0], [1.0]],
        tmin=0.0,
        tmax=3.0,
        nsteps=33,
        threads=false,
        cache=plan_cache,
    )

    Inv.clear_slice_module_cache!()
    cacheM = SI.module_cache(M23)
    pair = SI.module_cache(M23, M3)
    data = SI.slice_barcodes(M23, plan; packed=true, threads=false)
    feat_data = SI.slice_features(M23, plan; featurizer=:landscape, kmax=2, tgrid=[0.0, 0.5, 1.0], threads=false)

    task_b = SI.SliceBarcodesTask(; packed=true, threads=false)
    task_d = SI.SliceDistanceTask(; dist_fn=SI.bottleneck_distance, weight_mode=:integrate, agg=:mean, threads=false)
    task_k = SI.SliceKernelTask(; kind=:landscape_linear, tgrid=[0.0, 0.5, 1.0], kmax=2, threads=false)

    d_pl = TO.describe(pl)
    @test d_pl.kind == :persistence_landscape_1d
    @test d_pl.landscape_layers == 2
    @test d_pl.feature_dimension == 6
    @test SI.landscape_grid(pl) == [0.0, 0.5, 1.0]
    @test size(SI.landscape_values(pl)) == (2, 3)
    @test SI.landscape_layers(pl) == 2
    @test SI.feature_dimension(pl) == 6
    @test SI.landscape_summary(pl) == d_pl
    @test SI.check_persistence_landscape(pl).valid
    @test occursin("PersistenceLandscape1D", sprint(show, pl))
    @test occursin("feature_dimension", sprint(show, MIME"text/plain"(), pl))

    d_img = TO.describe(img)
    @test d_img.kind == :persistence_image_1d
    @test d_img.image_shape == (3, 3)
    @test SI.image_xgrid(img) == [0.0, 0.5, 1.0]
    @test SI.image_ygrid(img) == [0.0, 0.5, 1.0]
    @test size(SI.image_values(img)) == (3, 3)
    @test SI.image_shape(img) == (3, 3)
    @test SI.persistence_image_summary(img) == d_img
    @test SI.check_persistence_image(img).valid
    @test occursin("PersistenceImage1D", sprint(show, img))
    @test occursin("image_shape", sprint(show, MIME"text/plain"(), img))

    d_data = TO.describe(data)
    @test d_data.kind == :slice_barcodes_result
    @test d_data.packed
    @test d_data.barcode_shape == (1, 2)
    @test SI.slice_barcodes(data) === data.barcodes
    @test SI.slice_weights(data) === data.weights
    @test SI.slice_directions(data) === data.dirs
    @test SI.slice_offsets(data) === data.offs
    @test SI.packed_barcodes(data) === data.barcodes
    @test occursin("SliceBarcodesResult", sprint(show, data))
    @test occursin("slice_count", sprint(show, MIME"text/plain"(), data))

    d_feat = TO.describe(feat_data)
    @test feat_data isa SI.SliceFeaturesResult
    @test d_feat.kind == :slice_features_result
    @test d_feat.feature_kind == :landscape
    @test d_feat.aggregate == :mean
    @test SI.slice_features(feat_data) isa AbstractVector
    @test SI.slice_weights(feat_data) === feat_data.weights
    @test SI.feature_kind(feat_data) == :landscape
    @test SI.feature_aggregate(feat_data) == :mean
    @test occursin("SliceFeaturesResult", sprint(show, feat_data))
    @test occursin("feature_kind", sprint(show, MIME"text/plain"(), feat_data))

    d_plan = TO.describe(plan)
    @test d_plan.kind == :compiled_slice_plan
    @test d_plan.ambient_dim == 1
    @test d_plan.nslices == 2
    @test SI.plan_directions(plan) == [[1.0]]
    @test SI.plan_offsets(plan) == [[0.0], [1.0]]
    @test size(SI.plan_weights(plan)) == (1, 2)
    @test SI.nslices(plan) == 2
    @test isapprox(SI.total_weight(plan), 1.0; atol=1e-12)
    @test SI.plan_has_values(plan)
    @test SI.plan_value_mode(plan) == :t
    spec1 = SI.slice_spec(plan, 1)
    @test spec1 isa TO.SliceSpec{Float64,Vector{Float64}}
    @test TO.describe(spec1).kind == :slice_spec
    @test SI.slice_plan_summary(plan) == d_plan
    @test occursin("CompiledSlicePlan", sprint(show, plan))
    @test occursin("nslices", sprint(show, MIME"text/plain"(), plan))

    d_plan_cache = TO.describe(plan_cache)
    @test d_plan_cache.kind == :slice_plan_cache
    @test d_plan_cache.cached_plans == 1
    @test SI.slice_cache_summary(plan_cache) == d_plan_cache
    @test occursin("SlicePlanCache", sprint(show, plan_cache))

    d_cache = TO.describe(cacheM)
    @test d_cache.kind == :slice_module_cache
    @test TOA.source_module(cacheM) === M23
    @test d_cache.cached_barcode_plans == SI.cached_barcode_plan_count(cacheM)
    @test d_cache.cached_barcode_plans >= 1
    @test d_cache.cached_landscape_plans == SI.cached_landscape_plan_count(cacheM)
    @test d_cache.cached_landscape_plans >= 1
    @test SI.slice_cache_summary(cacheM) == d_cache
    @test occursin("SliceModuleCache", sprint(show, cacheM))

    d_pair = TO.describe(pair)
    @test d_pair.kind == :slice_module_pair_cache
    @test SI.left_cache(pair) === SI.module_cache(M23)
    @test SI.right_cache(pair) === SI.module_cache(M3)
    @test SI.slice_pair_cache_summary(pair) == d_pair
    @test occursin("SliceModulePairCache", sprint(show, pair))

    d_task_b = TO.describe(task_b)
    @test d_task_b.kind == :slice_barcodes_task
    @test SI.task_kind(task_b) == :barcodes
    @test SI.task_threads(task_b) == false
    @test SI.slice_task_summary(task_b) == d_task_b

    d_task_d = TO.describe(task_d)
    @test d_task_d.kind == :slice_distance_task
    @test SI.task_kind(task_d) == :distance
    @test SI.task_threads(task_d) == false
    @test SI.distance_function(task_d) === SI.bottleneck_distance
    @test SI.slice_task_summary(task_d) == d_task_d

    d_task_k = TO.describe(task_k)
    @test d_task_k.kind == :slice_kernel_task
    @test SI.task_kind(task_k) == :kernel
    @test SI.task_threads(task_k) == false
    @test SI.kernel_kind(task_k) == :landscape_linear
    @test SI.kernel_sigma(task_k) == 1.0
    @test SI.slice_task_summary(task_k) == d_task_k

    specs = TO.collect_slices(plan; values=:t)
    scs = SI.slice_collection_summary(specs)
    @test scs.kind == :slice_collection
    @test scs.nslices == 2
    @test scs.values_mode == :real

    valid_plan = SI.check_compiled_slice_plan(plan)
    @test valid_plan.valid
    wrapped_report = SI.slice_invariant_validation_summary(valid_plan)
    @test wrapped_report.report === valid_plan
    @test occursin("SliceInvariantValidationSummary", sprint(show, wrapped_report))

    bad_plan = SI.CompiledSlicePlan(
        plan.dirs,
        plan.offs,
        plan.weights,
        plan.chain_pool,
        plan.chain_start,
        plan.chain_len,
        plan.chains,
        plan.vals_pool,
        plan.vals_start,
        fill(0, length(plan.vals_len)),
        plan.nd,
        plan.no,
    )
    @test !SI.check_compiled_slice_plan(bad_plan).valid
    @test_throws ArgumentError SI.check_compiled_slice_plan(bad_plan; throw=true)

    @test SI.check_slice_plan_cache(plan_cache).valid
    @test SI.check_slice_module_cache(cacheM).valid
    @test SI.check_slice_module_pair_cache(pair).valid
    @test SI.check_slice_barcodes_task(task_b).valid
    @test SI.check_slice_distance_task(task_d).valid
    bad_task_d = SI.SliceDistanceTask(; weight_mode=:bad, threads=false)
    @test !SI.check_slice_distance_task(bad_task_d).valid
    @test_throws ArgumentError SI.check_slice_distance_task(bad_task_d; throw=true)
    @test SI.check_slice_kernel_task(task_k).valid
    bad_task_k = SI.SliceKernelTask(; kind=:bad, threads=false)
    @test !SI.check_slice_kernel_task(bad_task_k).valid
    @test_throws ArgumentError SI.check_slice_kernel_task(bad_task_k; throw=true)

    @test SI.check_slice_direction(pi, [1.0]).valid
    @test !SI.check_slice_direction(pi, [1.0, 0.0]).valid
    @test_throws ArgumentError SI.check_slice_direction(pi, [1.0, 0.0]; throw=true)
    @test SI.check_slice_basepoint(pi, [0.0]).valid
    @test !SI.check_slice_basepoint(pi, [0.0, 1.0]).valid
    @test_throws ArgumentError SI.check_slice_basepoint(pi, [0.0, 1.0]; throw=true)
    @test SI.check_slice_request(pi, [0.0], [1.0], opts).valid
    bad_opts = TO.InvariantOptions(box=([2.0], [1.0]))
    @test !SI.check_slice_request(pi, [0.0], [1.0], bad_opts).valid
    @test_throws ArgumentError SI.check_slice_request(pi, [0.0], [1.0], bad_opts; throw=true)

    @test SI.check_slice_specs(specs).valid
    bad_specs = [TO.SliceSpec([1, 1]; values=[0.0, 1.0], weight=1.0)]
    @test !SI.check_slice_specs(bad_specs).valid
    @test_throws ArgumentError SI.check_slice_specs(bad_specs; throw=true)

    bad_pl = SI.PersistenceLandscape1D([1.0, 0.0], zeros(2, 2))
    @test !SI.check_persistence_landscape(bad_pl).valid
    @test_throws ArgumentError SI.check_persistence_landscape(bad_pl; throw=true)

    bad_img = SI.PersistenceImage1D([0.0, 0.5], [0.0], zeros(2, 2))
    @test !SI.check_persistence_image(bad_img).valid
    @test_throws ArgumentError SI.check_persistence_image(bad_img; throw=true)

    @test TOA.PersistenceLandscape1D === SI.PersistenceLandscape1D
    @test TOA.SliceBarcodesResult === SI.SliceBarcodesResult
    @test TOA.SliceFeaturesResult === SI.SliceFeaturesResult
    @test TOA.CompiledSlicePlan === SI.CompiledSlicePlan
    @test TOA.SliceInvariantValidationSummary === SI.SliceInvariantValidationSummary
    @test TOA.compile_slice_plan === SI.compile_slice_plan
    @test TOA.module_cache === SI.module_cache
    @test TOA.run_invariants === SI.run_invariants
    @test TOA.landscape_summary === SI.landscape_summary
    @test TOA.slice_plan_summary === SI.slice_plan_summary
    @test TOA.slice_task_summary === SI.slice_task_summary
    @test TOA.slice_weights(data) === data.weights
    @test TOA.packed_barcodes(data) === data.barcodes
    @test TOA.slice_features(feat_data) === feat_data.features
    @test TOA.feature_kind(feat_data) == :landscape
    @test TOA.feature_aggregate(feat_data) == :mean
    @test TOA.check_persistence_landscape(pl).valid
    @test TOA.check_persistence_image(img).valid
    @test TOA.check_compiled_slice_plan === SI.check_compiled_slice_plan
    @test TOA.check_slice_request === SI.check_slice_request
    @test TOA.nslices(plan) == 2
    @test TOA.source_module(cacheM) === M23
end


@testset "slice_chain wrapper for ZnEncodingMap" begin
    n = 1
    b = [0]
    c = [5]
    I = FZ.Face(n, Int[])

    flats = [FZ.IndFlat(I, b; id=:F)]
    injectives = [FZ.IndInj(I, c; id=:E)]

    Phi = spzeros(K, 1, 1)
    Phi[1, 1] = 1

    FG = FZ.Flange{K}(n, flats, injectives, Phi)

    enc = TO.EncodingOptions(backend=:zn, max_regions=1000)
    Penc, Henc, pi = TamerOp.ZnEncoding.encode_from_flange(FG, enc)

    @test TO.nvertices(Penc) == 3

    # `ZnEncodingMap`s are defined on the integer lattice Z^n, but generic
    # helpers may represent an integer lattice point as a float (e.g. `2.0`).
    # Such points should be accepted if they are integer-valued.
    @test EC.locate(pi, [-2.0]) == EC.locate(pi, [-2])

    opts = TO.InvariantOptions()
    chain, tvals = TO.slice_chain(pi, [-2], [1], opts; kmin=0, kmax=9)
    @test chain == [1, 2, 3]
    @test tvals == [0, 2, 8]

    Menc = IR.pmodule_from_fringe(Henc)
    bM = TO.slice_barcode(Menc, chain)
    @test bM == Dict((2, 3) => 1)
end



@testset "Wrappers using locate(pi, x)" begin
    P = chain_poset(3)
    H23 = one_by_one_fringe(P, FF.principal_upset(P, 2), FF.principal_downset(P, 3); field=field)
    M23 = IR.pmodule_from_fringe(H23)

    pi = ToyPi()
    opts = TO.InvariantOptions()
    @test Inv.rank_map(M23, pi, [2], [3], opts) == 1
    @test TO.restricted_hilbert(M23, pi, [2], opts) == 1

    # rank_query generic fallback path for non-Zn encoding maps.
    pi_thr = ToyPi1DThresholds()
    @test Inv.rank_query(M23, pi_thr, [1], [2], opts) == Inv.rank_map(M23, pi_thr, [1], [2], opts)
    @test Inv.rank_query(M23, pi_thr, [1], [2]) == 1
    @test_throws ErrorException Inv.rank_query(M23, pi_thr, [4], [2], opts)
    @test Inv.rank_query(M23, pi_thr, [4], [2], TO.InvariantOptions(strict=false)) == 0
end


@testset "Betti/Bass tables from indicator resolutions" begin
    # Use the diamond poset because it supports nontrivial length-2 resolutions.
    P = diamond_poset()
    S1 = one_by_one_fringe(P, FF.principal_upset(P, 1), FF.principal_downset(P, 1); field=field)
    M1 = IR.pmodule_from_fringe(S1)

    # Upset indicator resolution
    F, _dF = IR.upset_resolution(M1; maxlen=2)
    resP = DF.projective_resolution(M1, TO.ResolutionOptions(maxlen=2))

    @test DF.betti(F) == DF.betti(resP)
    @test DF.betti_table(F) == DF.betti_table(resP)

    # Downset indicator resolution
    E, _dE = IR.downset_resolution(M1; maxlen=2)
    resI = DF.injective_resolution(M1, TO.ResolutionOptions(maxlen=2))

    @test DF.bass(E) == DF.bass(resI)
    @test DF.bass_table(E) == DF.bass_table(resI)
end


@testset "region_weights for ZnEncodingMap (1D sanity check)" begin
    # Reproduce the 1D Z-encoding test and additionally check region size weights.
    n = 1
    b = [0]
    c = [5]
    I = FZ.Face(n, Int[])

    flats = [FZ.IndFlat(I, b; id=:F)]
    injectives = [FZ.IndInj(I, c; id=:E)]

    Phi = spzeros(K, 1, 1)
    Phi[1, 1] = 1

    FG = FZ.Flange{K}(n, flats, injectives, Phi)

    enc = TO.EncodingOptions(backend=:zn, max_regions=1000)
    Penc, _Henc, pi = TamerOp.ZnEncoding.encode_from_flange(FG, enc)

    @test TO.nvertices(Penc) == 3

    # Box [b-2, c+2] = [-2, 7].
    w = TO.RegionGeometry.region_weights(pi; box=([b[1] - 2], [c[1] + 2]))
    @test w == [2, c[1] - b[1] + 1, 2]

        # mma_decomposition integration: PModule + ZnEncodingMap signature.
    # This should work out-of-the-box for method=:euler because Euler surfaces can be
    # evaluated on the encoding grid (axes_policy=:encoding).
    Menc = IR.pmodule_from_fringe(_Henc)
    opts_enc = TO.InvariantOptions(axes_policy=:encoding)
    outE = SM.mma_decomposition(Menc, pi, opts_enc; method=:euler)

    surfE = TO.euler_surface(Menc, pi, opts_enc)
    @test outE isa SM.SignedMeasureDecomposition
    @test SM.has_euler_surface(outE)
    @test SM.has_euler_signed_measure(outE)
    @test outE.euler_surface == surfE
    @test SM.surface_from_point_signed_measure(outE.euler_signed_measure) == surfE

    # method=:all combines rectangles + slices + Euler. For slices we must provide
    # directions and offsets; we keep this tiny so the test stays cheap.
    outAll = SM.mma_decomposition(
        Menc,
        pi;
        method=:all,
        slice_kwargs=(
            directions=[[1.0]],
            offsets=[[0.0]],
            tmin=-2.0,
            tmax=7.0,
            nsteps=11,
            strict=false,
            drop_unknown=true,
            dedup=true,
        ),
    )

    @test outAll isa SM.SignedMeasureDecomposition
    @test SM.has_rectangles(outAll)
    @test SM.has_slices(outAll)
    @test SM.has_euler_surface(outAll)
    @test SM.has_euler_signed_measure(outAll)
    @test !SM.has_mpp_image(outAll)
    @test SM.has_euler(outAll)
    @test !SM.has_image(outAll)
    @test SM.nrectangles(outAll) == SM.nterms(SM.rectangles(outAll))
    @test SM.nslices(outAll) == length(SM.slices(outAll).barcodes)
    @test SM.ncomponents(outAll) == 4
    @test !isempty(SM.components(outAll))
    @test SM.component_names(outAll) == (:rectangles, :slices, :euler_surface, :euler_signed_measure)
    @test SM.signed_measure_decomposition_summary(outAll).kind == :signed_measure_decomposition
    @test SM.surface_from_point_signed_measure(outAll.euler_signed_measure) == outAll.euler_surface
    @test hasproperty(SM.slices(outAll), :barcodes)
    @test hasproperty(SM.slices(outAll), :weights)
    @test SM.rectangles(outAll) isa SM.RectSignedBarcode
    @test occursin("SignedMeasureDecomposition", sprint(show, outAll))
    decomp_report = SM.check_signed_measure_decomposition(outAll; throw=false)
    @test decomp_report isa SM.SignedMeasureDecompositionValidationSummary
    @test decomp_report.valid

    out_bad = SM.SignedMeasureDecomposition(
        euler_surface = zeros(Int, 2),
        euler_signed_measure = SM.PointSignedMeasure((Float64[0.0, 1.0, 2.0],), [(2,)], [1]),
    )
    bad_decomp = SM.check_signed_measure_decomposition(out_bad; throw=false)
    @test !bad_decomp.valid
    @test !isempty(bad_decomp.errors)
    @test_throws ArgumentError SM.check_signed_measure_decomposition(out_bad; throw=true)

    @test TOA.Rect === SM.Rect
    @test TOA.RectSignedBarcode === SM.RectSignedBarcode
    @test TOA.PointSignedMeasure === SM.PointSignedMeasure
    @test TOA.SignedMeasureDecomposition === SM.SignedMeasureDecomposition
    @test TOA.RectValidationSummary === SM.RectValidationSummary
    @test TOA.RectSignedBarcodeValidationSummary === SM.RectSignedBarcodeValidationSummary
    @test TOA.PointSignedMeasureValidationSummary === SM.PointSignedMeasureValidationSummary
    @test TOA.SignedMeasureDecompositionValidationSummary === SM.SignedMeasureDecompositionValidationSummary
    @test TOA.term === SM.term
    @test TOA.largest_terms === SM.largest_terms
    @test TOA.check_rect === SM.check_rect
    @test TOA.check_rect_signed_barcode === SM.check_rect_signed_barcode
    @test TOA.check_point_signed_measure === SM.check_point_signed_measure
    @test TOA.check_signed_measure_decomposition === SM.check_signed_measure_decomposition
    @test TOA.signed_measure_summary === SM.signed_measure_summary
    @test TOA.rect_signed_barcode_summary === SM.rect_signed_barcode_summary
    @test TOA.point_signed_measure_summary === SM.point_signed_measure_summary
    @test TOA.signed_measure_decomposition_summary === SM.signed_measure_decomposition_summary

end


@testset "Extra slice + bottleneck tests (optional edge cases)" begin

    @testset "Non-chain input rejection" begin
        P = diamond_poset()
        H = one_by_one_fringe(P, FF.principal_upset(P, 2), FF.principal_downset(P, 2); field=field)
        M = IR.pmodule_from_fringe(H)

        # 2 and 3 are incomparable in the diamond poset.
        chain_bad = [2, 3]
        @test_throws ErrorException TO.slice_barcode(M, chain_bad)
        @test_throws ErrorException Inv.restrict_to_chain(M, chain_bad)
    end

    @testset "Geometric slicing wrapper for a toy locate" begin
        # Toy encoding map for R^2: regions are determined by (floor(x1), floor(x2)).
        # Along the diagonal line x(t) = (t,t), this yields a strictly increasing chain.
        pi = ToyPi2D()
        chain, tvals = TO.slice_chain(pi, [0.0, 0.0], [1.0, 1.0], TO.InvariantOptions(strict=true);
                                      tmin=0.0, tmax=3.0, nsteps=301,
                                      drop_unknown=true, dedup=true)

        @test chain == [1, 12, 23, 34]
        @test length(chain) == length(tvals)
    @test all(diff(chain) .> 0)               # monotone and de-duplicated
    @test all(diff(tvals) .> 0)               # parameter values strictly increasing
    end

    @testset "slice_chain batched locate parity" begin
        Pgrid = chain_poset(3)
        pi = TO.EncodingCore.GridEncodingMap(Pgrid, ([0.0, 1.0, 2.0],))
        opts = TO.InvariantOptions(strict=true, threads=false)
        x0 = [0.0]
        dir = [1.0]

        old_batch = SI._SLICE_CHAIN_USE_BATCHED_LOCATE[]
        old_batch_min = SI._SLICE_CHAIN_BATCHED_LOCATE_MIN_SAMPLES[]
        try
            SI._SLICE_CHAIN_USE_BATCHED_LOCATE[] = false
            chain_off, tvals_off = TO.slice_chain(pi, x0, dir, opts;
                                                  tmin=0.0, tmax=2.0, nsteps=301,
                                                  drop_unknown=true, dedup=true)
            plan_off = Inv.compile_slice_plan(pi;
                                              directions=[[1.0]], offsets=[[0.0]],
                                              tmin=0.0, tmax=2.0, nsteps=301,
                                              threads=false, cache=nothing)
            SI._SLICE_CHAIN_USE_BATCHED_LOCATE[] = true
            SI._SLICE_CHAIN_BATCHED_LOCATE_MIN_SAMPLES[] = 1
            chain_on, tvals_on = TO.slice_chain(pi, x0, dir, opts;
                                                tmin=0.0, tmax=2.0, nsteps=301,
                                                drop_unknown=true, dedup=true)
            plan_on = Inv.compile_slice_plan(pi;
                                             directions=[[1.0]], offsets=[[0.0]],
                                             tmin=0.0, tmax=2.0, nsteps=301,
                                             threads=false, cache=nothing)
            @test chain_off == chain_on
            @test tvals_off == tvals_on
            slices_off = Inv.collect_slices(plan_off; values=:t)
            slices_on = Inv.collect_slices(plan_on; values=:t)
            @test [s.chain for s in slices_off] == [s.chain for s in slices_on]
            @test [s.values for s in slices_off] == [s.values for s in slices_on]
        finally
            SI._SLICE_CHAIN_USE_BATCHED_LOCATE[] = old_batch
            SI._SLICE_CHAIN_BATCHED_LOCATE_MIN_SAMPLES[] = old_batch_min
        end
    end

    @testset "slice_chain clips to box window" begin
        PLB = TO.PLBackend

        Ups = [PLB.BoxUpset([0.0, 0.0], [1.0, 1.0])]
        Downs = [PLB.BoxDownset([2.0, 2.0], [3.0, 3.0])]
        _P, _H, pi = PLB.encode_fringe_boxes(Ups, Downs)

        x0 = [0.5, 0.5]
        dir = [1.0, 0.0]

        # Clip to an interior window. Along x(t) = (0.5+t, 0.5), staying in x in [0.2,0.8]
        # means t in [-0.3, 0.3], and with tmin=0 we expect max t <= 0.3.
        box = ([0.2, 0.2], [0.8, 0.8])
        opts_strict = TO.InvariantOptions(strict=true, box = box)
        chain, tvals = TO.slice_chain(pi, x0, dir, opts_strict;
                                    tmin=0.0, tmax=100.0, nsteps=11,
                                    drop_unknown=true, dedup=true)
        @test !isempty(chain)
        @test maximum(tvals) <= 0.3 + 1e-12

        # With box2, use intersection. Shrink x-high from 0.8 to 0.6 => tmax from 0.3 to 0.1.
        box2 = ([0.2, 0.2], [0.6, 0.8])
        chain2, tvals2 = TO.slice_chain(pi, x0, dir, opts_strict;
                                        tmin=0.0, tmax=100.0, nsteps=11,
                                        drop_unknown=true, dedup=true, box2=box2)
        @test !isempty(chain2)
        @test maximum(tvals2) <= 0.1 + 1e-12

        # Explicit ts: should filter to those integers in [0, 0.1], i.e. just 0.
        chain3, tvals3 = TO.slice_chain(pi, x0, dir, opts_strict;
                                        ts=0:10,
                                        drop_unknown=true, dedup=true, box2=box2)
        @test tvals3 == [0]
        @test length(chain3) == 1
        
        # Semantic default:
        # If a window is present and the caller omits tmin/tmax, slice_chain should
        # sample the full line-window intersection interval (not [0,1] intersected).
        chain4, tvals4 = TO.slice_chain(pi, x0, dir, opts_strict;
                                        nsteps=11,
                                        drop_unknown=true, dedup=false)
        @test length(tvals4) == 11
        @test isapprox(first(tvals4), -0.3; atol=1e-12)
        @test isapprox(last(tvals4), 0.3; atol=1e-12)

        # With two windows, the effective window is their intersection.
        chain5, tvals5 = TO.slice_chain(pi, x0, dir, opts_strict;
                                        nsteps=11,
                                        drop_unknown=true, dedup=false, box2=box2)
        @test length(tvals5) == 11
        @test isapprox(first(tvals5), -0.3; atol=1e-12)
        @test isapprox(last(tvals5), 0.1; atol=1e-12)

    end


    @testset "Bottleneck triangle inequality sanity check" begin
        # A has one interval; B and C add a short interval that changes length.
        A = Dict((0.0, 2.0) => 1)
        B = Dict((0.0, 2.0) => 1, (3.0, 3.2) => 1)
        C = Dict((0.0, 2.0) => 1, (3.0, 3.4) => 1)

        dAB = Inv.bottleneck_distance(A, B)
        dBC = Inv.bottleneck_distance(B, C)
        dAC = Inv.bottleneck_distance(A, C)

        @test isapprox(dAB, 0.1; atol=1e-12)
        @test isapprox(dBC, 0.2; atol=1e-12)
        @test isapprox(dAC, 0.2; atol=1e-12)

        # Triangle inequality (allow a tiny floating tolerance).
        @test dAC <= dAB + dBC + 1e-12
    end
end


@testset "sample_directions_2d helper" begin
    dirs = Inv.sample_directions_2d(max_den=3)
    @test !isempty(dirs)

    # All directions are 2D, strictly positive, and L1-normalized.
    @test all(length(d) == 2 for d in dirs)
    @test all(d[1] > 0 && d[2] > 0 for d in dirs)
    @test all(isapprox(d[1] + d[2], 1.0; atol=1e-12) for d in dirs)

    # The diagonal direction (1,1) must be present.
    @test any(d -> isapprox(d[1], 0.5; atol=1e-12) && isapprox(d[2], 0.5; atol=1e-12), dirs)

    # Integer directions are useful for Z^2 encodings.
    dirsZ = Inv.sample_directions_2d(max_den=3; normalize=:none)
    @test any(d -> d == (1, 1), dirsZ)
end

@testset "Multiparameter persistence landscapes" begin
    P = chain_poset(3)

    # Module supported on 2 -> 3 (so barcode is (1,3) in the slice values below).
    H23 = one_by_one_fringe(P, FF.principal_upset(P, 2), FF.principal_downset(P, 3); field=field)
    M23 = IR.pmodule_from_fringe(H23)

    # Module supported only at 3 (barcode (2,3) in the slice values below).
    H3 = one_by_one_fringe(P, FF.principal_upset(P, 3), FF.principal_downset(P, 3); field=field)
    M3 = IR.pmodule_from_fringe(H3)

    # A simple ground-truth barcode and its 1D persistence landscape.
    bar = Dict((1.0, 3.0) => 1)
    tgrid = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    pl = Inv.persistence_landscape(bar; kmax=3, tgrid=tgrid)

    # lambda_1 is a tent with peak 1 at t=2.
    @test isapprox(pl.values[1, 1], 0.0; atol=1e-12)
    @test isapprox(pl.values[1, 3], 0.0; atol=1e-12)  # t = 1.0 (birth)
    @test isapprox(pl.values[1, 4], 0.5; atol=1e-12)  # t = 1.5
    @test isapprox(pl.values[1, 5], 1.0; atol=1e-12)  # t = 2.0 (midpoint)
    @test isapprox(pl.values[1, 6], 0.5; atol=1e-12)  # t = 2.5
    @test isapprox(pl.values[1, 7], 0.0; atol=1e-12)  # t = 3.0 (death)

    # Higher layers vanish for a single interval barcode.
    @test maximum(abs.(pl.values[2:end, :])) == 0.0

    packed_bar = SI._slice_barcode_packed(M23, [1, 2, 3]; values=[0.0, 1.0, 2.0])
    vals_fill = zeros(Float64, 3, length(tgrid))
    SI._persistence_landscape_values!(
        vals_fill,
        packed_bar,
        tgrid;
        points_scratch=Tuple{Float64,Float64}[],
        tent_scratch=Float64[],
    )
    @test vals_fill == pl.values

    # Build a multiparameter landscape from an explicit chain slice.
    slice = (chain=[1, 2, 3], values=[0.0, 1.0, 2.0], weight=1.0)
    L23 = TO.mp_landscape(M23, [slice]; kmax=2, tgrid=tgrid)
    L3  = TO.mp_landscape(M3,  [slice]; kmax=2, tgrid=tgrid)

    @test size(L23.values) == (1, 1, 2, length(tgrid))
    @test isapprox(sum(L23.weights), 1.0; atol=1e-12)

    # The first layer for M23 should match the tent coming from (1,3).
    @test isapprox(L23.values[1, 1, 1, 5], 1.0; atol=1e-12)
    @test isapprox(L23.values[1, 1, 1, 4], 0.5; atol=1e-12)
    @test isapprox(L23.values[1, 1, 1, 6], 0.5; atol=1e-12)
    @test isapprox(L23.values[1, 1, 2, 5], 0.0; atol=1e-12)

    # Distance and kernel sanity checks.
    d_same = Inv.mp_landscape_distance(L23, L23; p=2)
    @test isapprox(d_same, 0.0; atol=1e-12)

    d_23_3 = Inv.mp_landscape_distance(L23, L3; p=2)
    d_3_23 = Inv.mp_landscape_distance(L3, L23; p=2)
    @test d_23_3 >= 0.0
    @test isapprox(d_23_3, d_3_23; atol=1e-12)
    @test d_23_3 > 0.0

    k_same = Inv.mp_landscape_kernel(L23, L23; kind=:gaussian, sigma=1.0)
    k_diff = Inv.mp_landscape_kernel(L23, L3; kind=:gaussian, sigma=1.0)
    @test isapprox(k_same, 1.0; atol=1e-12)
    @test k_diff < 1.0

    # Convenience wrappers: construct landscapes internally.
    d_wrap = Inv.mp_landscape_distance(M23, M3, [slice]; p=2, kmax=2, tgrid=tgrid)
    @test isapprox(d_wrap, d_23_3; atol=1e-12)

    k_wrap = Inv.mp_landscape_kernel(M23, M3, [slice]; kind=:gaussian, sigma=1.0, p=2, kmax=2, tgrid=tgrid)
    @test isapprox(k_wrap, k_diff; atol=1e-12)

    # Geometric slicing wrapper sanity check with a toy locate() on R^1.
    pi = ToyPi1DThresholds()
    dirs = [[1.0]]
    offs = [[0.0]]

    L23_geo = TO.mp_landscape(M23, pi;
                              directions=dirs,
                              offsets=offs,
                              kmax=2,
                              tgrid=tgrid,
                              tmin=0.0,
                              tmax=3.0,
                              nsteps=301,
                              strict=false,
                              drop_unknown=true,
                              dedup=true)

    @test isapprox(L23_geo.values[1, 1, 1, 5], 1.0; atol=1e-12)
    L23_ts = TamerOp.MultiparameterImages.mp_landscape(M23, pi;
                                                            directions=dirs,
                                                            offsets=offs,
                                                            kmax=2,
                                                            tgrid=tgrid,
                                                            ts=0.0:0.5:3.0,
                                                            drop_unknown=true,
                                                            dedup=true)
    @test L23_ts isa TO.MPLandscape
    plan_ts = TO.SliceInvariants.compile_slice_plan(
        pi,
        TO.Options.InvariantOptions(box=([0.0], [3.0]), strict=false);
        directions=dirs,
        offsets=offs,
        tmin=0.0,
        tmax=3.0,
        nsteps=301,
        threads=false,
        drop_unknown=true,
        dedup=true,
        ts=0.0:0.5:3.0,
        cache=TO.SliceInvariants.SlicePlanCache(),
    )
    L23_plan = TamerOp.MultiparameterImages.mp_landscape(M23, plan_ts;
                                                              kmax=2,
                                                              tgrid=tgrid,
                                                              threads=false)
    @test L23_plan.values == L23_ts.values
    @test L23_plan.weights == L23_ts.weights
    @test_throws ArgumentError TamerOp.MultiparameterImages.mp_landscape(M23, pi;
                                                                              directions=dirs,
                                                                              offsets=offs,
                                                                              kmax=2,
                                                                              tgrid=tgrid,
                                                                              kmin=0,
                                                                              kmax_param=3)

    # -------------------------------------------------------------------------
    # show(::MPLandscape) should give a compact, informative summary.
    # -------------------------------------------------------------------------

    s = sprint(show, L23)
    @test occursin("MPLandscape(", s)
    @test occursin("ndirs=1", s)
    @test occursin("noffsets=1", s)
    @test occursin("kmax=2", s)
    @test occursin("nt=$(length(tgrid))", s)
    @test occursin("weights_normalized=true", s)

    # If we intentionally skip normalization and use a non-unit weight,
    # the summary should report "weights_normalized=false".
    slice2 = (chain=[1, 2, 3], values=[0.0, 1.0, 2.0], weight=2.0)
    L_un = TO.mp_landscape(M23, [slice2]; kmax=2, tgrid=tgrid, normalize_weights=false)
    s_un = sprint(show, L_un)
    @test occursin("weights_normalized=false", s_un)

    # The text/plain show should include the same core metadata in a readable form.
    s_plain = sprint(show, MIME("text/plain"), L23)
    @test occursin("MPLandscape", s_plain)
    @test occursin("ndirs = 1", s_plain)
    @test occursin("noffsets = 1", s_plain)
    @test occursin("kmax = 2", s_plain)
    @test occursin("weights_normalized = true", s_plain)

    dL = TO.describe(L23)
    @test dL.kind == :mp_landscape
    @test dL.ndirections == 1
    @test dL.noffsets == 1
    @test dL.kmax == 2
    @test dL.grid_length == length(tgrid)
    @test TO.mp_landscape_summary(L23) == dL
    @test TO.landscape_grid(L23) == L23.tgrid
    @test TO.landscape_values(L23) === L23.values
    @test TO.landscape_layers(L23) == 2
    @test TO.slice_weights(L23) === L23.weights
    @test TO.slice_directions(L23) === L23.directions
    @test TO.slice_offsets(L23) === L23.offsets
    @test TO.ndirections(L23) == 1
    @test TO.noffsets(L23) == 1
    @test isapprox(TO.weight_sum(L23), 1.0)
    @test TO.landscape_slice(L23, 1, 1) == L23[1, 1]
    @test TO.feature_dimension(L23) == length(L23.values)

    Lreport = TO.check_mp_landscape(L23)
    @test Lreport isa TO.MPLandscapeValidationSummary
    @test Lreport.valid

    badL = TO.MPLandscape(0, [0.0, 0.0], zeros(1, 1, 1, 2), reshape([-1.0], 1, 1), [[1.0]], [nothing])
    badL_report = TO.check_mp_landscape(badL)
    @test !badL_report.valid
    @test !isempty(badL_report.errors)
    @test_throws ArgumentError TO.check_mp_landscape(badL; throw=true)
    @test TOA.mp_landscape_summary === TamerOp.MultiparameterImages.mp_landscape_summary
    @test TOA.check_mp_landscape === TamerOp.MultiparameterImages.check_mp_landscape
    @test TOA.ndirections === TamerOp.MultiparameterImages.ndirections
    @test TOA.noffsets === TamerOp.MultiparameterImages.noffsets

end

@testset "Slice vectorizations and sliced kernels" begin
    # ---------------------------------------------------------------------
    # Barcode-level vectorizations.
    # ---------------------------------------------------------------------
    bar = Dict((1.0, 3.0) => 1)
    tgrid = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    pl = Inv.persistence_landscape(bar; kmax=1, tgrid=tgrid)
    sil = Inv.persistence_silhouette(bar; tgrid=tgrid, weighting=:persistence, p=1, normalize=true)

    # For a single interval, silhouette (with normalize=true) equals the tent itself,
    # i.e. the first landscape layer.
    @test maximum(abs.(sil .- pl.values[1, :])) < 1e-12

    img = Inv.persistence_image(bar;
                               xgrid=[0.0, 1.0, 2.0, 3.0],
                               ygrid=[0.0, 1.0, 2.0],
                               sigma=0.2,
                               weighting=:none,
                               normalize=:none)
    @test size(img.values) == (3, 4)
    ind = argmax(img.values)
    iy, ix = Tuple(CartesianIndices(img.values)[ind])
    @test ix == 2   # birth = 1.0 (xgrid[2])
    @test iy == 3   # pers  = 2.0 (ygrid[3])

    imgN = Inv.persistence_image(bar;
                                xgrid=[0.0, 1.0, 2.0, 3.0],
                                ygrid=[0.0, 1.0, 2.0],
                                sigma=0.2,
                                weighting=:none,
                                normalize=:max)
    @test isapprox(maximum(imgN.values), 1.0; atol=1e-12)

    bar2 = Dict((0.0, 2.0) => 1, (0.0, 1.0) => 1)
    H = Inv.barcode_entropy(bar2; normalize=false)
    expected = -(2 / 3 * log(2 / 3) + 1 / 3 * log(1 / 3))
    @test isapprox(H, expected; atol=1e-12)

    Hn = Inv.barcode_entropy(bar2; normalize=true)
    @test isapprox(Hn, H / log(2); atol=1e-12)

    summ = Inv.barcode_summary(bar2; normalize_entropy=false)
    @test summ.n_intervals == 2
    @test isapprox(summ.total_persistence, 3.0; atol=1e-12)
    @test isapprox(summ.max_persistence, 2.0; atol=1e-12)
    @test isapprox(summ.mean_persistence, 1.5; atol=1e-12)

    # ---------------------------------------------------------------------
    # Module-level slice_features and slice_kernel.
    # Build a small direct sum module with two intervals along a chain slice.
    # ---------------------------------------------------------------------
    P = chain_poset(3)
    U1 = FF.principal_upset(P, 2)
    D1 = FF.principal_downset(P, 3)
    U2 = FF.principal_upset(P, 3)
    D2 = FF.principal_downset(P, 3)

    Phi = spzeros(K, 2, 2)
    Phi[1, 1] = 1
    Phi[2, 2] = 1

    Hsum = FF.FringeModule{K}(P, [U1, U2], [D1, D2], Phi; field=field)
    Msum = IR.pmodule_from_fringe(Hsum)

    slice = (chain=[1, 2, 3], values=[0.0, 1.0, 2.0], weight=1.0)
    bc_sum = TO.slice_barcode(Msum, slice.chain; values=slice.values)

    @test bc_sum == Dict((1.0, 3.0) => 1, (2.0, 3.0) => 1)

    sb_float = TO.slice_barcodes(Msum, [slice]; threads=false)
    @test eltype(sb_float.barcodes) == Inv.FloatBarcode

    sb_index = TO.slice_barcodes(Msum, [slice.chain]; threads=false)
    @test eltype(sb_index.barcodes) == Inv.IndexBarcode

    # slice_features uses normalized persistent entropy by default (entropy_normalize=true),
    # so it should match Hn computed above.
    ent_norm = Inv.slice_features(Msum, [slice]; featurizer=:entropy, aggregate=:mean)
    @test ent_norm isa SI.SliceFeaturesResult
    @test isapprox(Inv.slice_features(ent_norm), Hn; atol=1e-12)

    # Also test the raw (unnormalized) entropy path explicitly.
    ent_raw = Inv.slice_features(Msum, [slice];
                                featurizer=:entropy,
                                aggregate=:mean,
                                entropy_normalize=false)
    @test isapprox(Inv.slice_features(ent_raw), expected; atol=1e-12)

    # Also check landscape features shape and a known value for M23.
    H23 = FF.one_by_one_fringe(P, U1, D1, cf(1); field=field)
    M23 = IR.pmodule_from_fringe(H23)

    f_land = Inv.slice_features(M23, [slice];
                               featurizer=:landscape,
                               kmax=2,
                               tgrid=tgrid,
                               aggregate=:mean)
    @test f_land isa SI.SliceFeaturesResult
    @test length(Inv.slice_features(f_land)) == 2 * length(tgrid)
    # For barcode (1,3) and tgrid above, layer 1 at t=2.0 is 1.0.
    @test isapprox(Inv.slice_features(f_land)[5], 1.0; atol=1e-12)

    # Geometric version: compare to explicit slice using a tiny toy encoding map.
    pi = ToyPi1DIntervals()
    f_geo = Inv.slice_features(M23, pi;
                              opts=TO.InvariantOptions(strict=false),
                              directions=[[1.0]],
                              offsets=[[0.0]],
                              tmin=0.0,
                              tmax=3.0,
                              nsteps=301,
                              drop_unknown=true,
                              dedup=true,
                              featurizer=:landscape,
                              kmax=2,
                              tgrid=tgrid,
                              aggregate=:mean)
    @test isapprox(Inv.slice_features(f_geo)[5], Inv.slice_features(f_land)[5]; atol=1e-12)

    sb_geo = TO.slice_barcodes(M23, pi;
                               opts=TO.InvariantOptions(strict=false),
                               directions=[[1.0]],
                               offsets=[[0.0]],
                               tmin=0.0,
                               tmax=3.0,
                               nsteps=101,
                               drop_unknown=true,
                               dedup=true,
                               threads=false)
    @test eltype(sb_geo.barcodes) == Inv.FloatBarcode

    # Sliced kernels: identical inputs should give kernel value 1 for gaussian kinds.
    k_same = Inv.slice_kernel(M23, M23, [slice]; kind=:bottleneck_gaussian, sigma=1.0)
    @test isapprox(k_same, 1.0; atol=1e-12)

    # Symmetry and strict inequality for different modules.
    H3 = FF.one_by_one_fringe(P, U2, D2, cf(1); field=field)
    M3 = IR.pmodule_from_fringe(H3)
    k_diff1 = Inv.slice_kernel(M23, M3, [slice]; kind=:bottleneck_gaussian, sigma=1.0)
    k_diff2 = Inv.slice_kernel(M3, M23, [slice]; kind=:bottleneck_gaussian, sigma=1.0)
    @test isapprox(k_diff1, k_diff2; atol=1e-12)
    @test k_diff1 < 1.0

    # Landscape kernel (gaussian) on identical inputs is also 1.
    k_land = Inv.slice_kernel(M23, M23, [slice];
                             kind=:landscape_gaussian,
                             sigma=1.0,
                             tgrid=tgrid,
                             kmax=2)
    @test isapprox(k_land, 1.0; atol=1e-12)
end

@testset "Signed rectangles / Mobius inversion" begin
    axes = ([0, 1, 2], [0, 1])

    rects = SM.Rect{2}[
        SM.Rect{2}((0, 0), (2, 1)),
        SM.Rect{2}((1, 0), (2, 0)),
        SM.Rect{2}((0, 1), (1, 1)),
    ]
    weights = [2, -1, 3]
    sb_true = SM.RectSignedBarcode{2,Int}(axes, rects, weights)
    @test SM.axes(sb_true) == axes
    @test SM.rectangles(sb_true) == rects
    @test SM.weights(sb_true) == weights
    @test SM.ambient_dimension(sb_true) == 2
    @test SM.nterms(sb_true) == 3
    @test SM.positive_terms(sb_true) == 2
    @test SM.negative_terms(sb_true) == 1
    @test SM.total_mass(sb_true) == 4
    @test SM.total_variation(sb_true) == 6
    @test SM.axis_lengths(sb_true) == (3, 2)
    @test SM.weight_range(sb_true) == (-1, 3)
    @test SM.max_abs_weight(sb_true) == 3
    @test SM.support_size(sb_true) == 3
    @test SM.lower_corner(rects[1]) == (0, 0)
    @test SM.upper_corner(rects[1]) == (2, 1)
    @test SM.side_lengths(rects[1]) == (2, 1)
    @test SM.grid_span(rects[1]) == (3, 2)
    @test SM.center(rects[1]) == (1.0, 0.5)
    @test SM.contains(rects[1], (1, 1))
    @test !SM.contains(rects[2], (0, 1))
    @test SM.intersects(rects[1], rects[2])
    @test SM.intersection(rects[1], rects[2]) == rects[2]
    @test !SM.is_degenerate(rects[1])
    @test SM.is_degenerate(SM.Rect{2}((1, 1), (1, 2)))
    @test SM.term(sb_true, 2).weight == -1
    @test SM.coefficient(sb_true, rects[2]) == -1
    @test SM.support_bbox(sb_true) == (lo = (0, 0), hi = (2, 1))
    @test length(SM.largest_terms(sb_true; n=2)) == 2
    @test SM.describe(sb_true).kind == :rect_signed_barcode
    @test SM.signed_measure_summary(sb_true) == SM.describe(sb_true)
    @test SM.rect_signed_barcode_summary(sb_true) == SM.describe(sb_true)
    @test TO.describe(sb_true).kind == :rect_signed_barcode
    @test occursin("RectSignedBarcode", sprint(show, sb_true))
    rect_report = SM.check_rect(rects[1]; axes=axes, throw=false)
    @test rect_report isa SM.RectValidationSummary
    @test rect_report.valid
    @test occursin("RectValidationSummary", sprint(show, rect_report))
    sb_report = SM.check_rect_signed_barcode(sb_true; throw=false)
    @test sb_report isa SM.RectSignedBarcodeValidationSummary
    @test sb_report.valid
    @test SM.validate(sb_true; throw=false) isa SM.RectSignedBarcodeValidationSummary
    @test SM.validate(sb_true; throw=false).valid

    function r_idx(p, q)
        x = (axes[1][p[1]], axes[2][p[2]])
        y = (axes[1][q[1]], axes[2][q[2]])
        return SM.rank_from_signed_barcode(sb_true, x, y)
    end

    sb_est = SM.rectangle_signed_barcode(r_idx, axes)

    d_true = Dict(zip(sb_true.rects, sb_true.weights))
    d_est = Dict(zip(sb_est.rects, sb_est.weights))
    @test d_est == d_true

    dims = (length(axes[1]), length(axes[2]))
    for pCI in CartesianIndices(dims)
        p = Tuple(pCI)
        for qCI in CartesianIndices((p[1]:dims[1], p[2]:dims[2]))
            q = Tuple(qCI)
            x = (axes[1][p[1]], axes[2][p[2]])
            y = (axes[1][q[1]], axes[2][q[2]])
            @test SM.rank_from_signed_barcode(sb_est, x, y) == r_idx(p, q)
        end
    end

    sb_trunc = SM.truncate_signed_barcode(sb_est; max_terms=1)
    @test length(sb_trunc) == 1
    @test abs(sb_trunc.weights[1]) == maximum(abs.(sb_est.weights))

    sb_bad = SM.RectSignedBarcode{1,Int}(([0, 2],), [SM.Rect{1}((0,), (1,))], [1])
    bad_report = SM.validate(sb_bad; throw=false)
    @test !bad_report.valid
    @test !isempty(bad_report.errors)
    @test_throws ArgumentError SM.validate(sb_bad)
    bad_rect_report = SM.check_rect(SM.Rect{1}((0,), (1,)); axes=([0, 2],), throw=false)
    @test !bad_rect_report.valid
    @test !isempty(bad_rect_report.errors)
    @test_throws ArgumentError SM.check_rect(SM.Rect{1}((0,), (1,)); axes=([0, 2],), throw=true)

    k_lin = SM.rectangle_signed_barcode_kernel(sb_est, sb_est; kind=:linear)
    @test isapprox(k_lin, sum(float(w)^2 for w in sb_est.weights); atol=1e-12)

    k_gauss = SM.rectangle_signed_barcode_kernel(sb_est, sb_est; kind=:gaussian, sigma=1.0)
    k_gauss_dense = 0.0
    for (r1, w1) in zip(sb_est.rects, sb_est.weights)
        emb1 = (r1.lo..., r1.hi...)
        for (r2, w2) in zip(sb_est.rects, sb_est.weights)
            emb2 = (r2.lo..., r2.hi...)
            d2 = 0.0
            for k in 1:length(emb1)
                dk = float(emb1[k] - emb2[k])
                d2 += dk * dk
            end
            k_gauss_dense += float(w1) * float(w2) * exp(-d2 / 2)
        end
    end
    @test isapprox(k_gauss, k_gauss_dense; atol=1e-12)
    @test SM._rectangle_signed_barcode_embedding_cached(sb_est) ===
          SM._rectangle_signed_barcode_embedding_cached(sb_est)

    img = SM.rectangle_signed_barcode_image(sb_est; sigma=1.0, mode=:center, cutoff_tol=0.0)
    @test size(img) == (length(axes[1]), length(axes[2]))

    img_dense = zeros(Float64, length(axes[1]), length(axes[2]))
    for (rect, w) in zip(sb_est.rects, sb_est.weights)
        cx = (rect.lo[1] + rect.hi[1]) / 2
        cy = (rect.lo[2] + rect.hi[2]) / 2
        for (ix, x) in pairs(axes[1]), (iy, y) in pairs(axes[2])
            img_dense[ix, iy] += float(w) * exp(-((x - cx)^2 + (y - cy)^2) / 2)
        end
    end
    @test isapprox(img, img_dense; atol=1e-12, rtol=0.0)

    img_cut = SM.rectangle_signed_barcode_image(sb_est; sigma=1.0, mode=:center, cutoff_tol=1.0e-6)
    @test isapprox(img_cut, img_dense; atol=1.0e-5, rtol=0.0)

    wtensor = zeros(Int, 2, 2, 2, 2)
    wtensor[1, 1, 1, 1] = 2
    wtensor[1, 1, 2, 2] = -1
    axes2 = ([0, 1], [0, 1])
    sb_tensor = SM._extract_rectangles_from_mobius_tensor(wtensor, axes2; drop_zeros=true)
    @test Dict(zip(sb_tensor.rects, sb_tensor.weights)) == Dict(
        SM.Rect{2}((0, 0), (0, 0)) => 2,
        SM.Rect{2}((0, 0), (1, 1)) => -1,
    )
end

@testset "Wasserstein on barcodes" begin
    bar1 = Dict((0, 2) => 1)
    bar2 = Dict((0, 3) => 1)

    d12 = Inv.wasserstein_distance(bar1, bar2; p=1, q=Inf)
    @test isapprox(d12, 1.0; atol=1e-12)

    dempty = Inv.wasserstein_distance(Dict{Tuple{Int,Int},Int}(), bar1; p=2, q=Inf)
    @test isapprox(dempty, 1.0; atol=1e-12)

    k = Inv.wasserstein_kernel(bar1, bar2; p=1, q=Inf, sigma=1.0, kind=:gaussian)
    @test isapprox(k, exp(-0.5); atol=1e-12)
end

@testset "Rectangle signed barcode speed options" begin
    # -------------------------------------------------------------------------
    # 1) max_span restriction should equal filtering the full result by span.
    # -------------------------------------------------------------------------
    axes = ([0, 1, 2], [0, 1, 2])

    rects_true = TO.Invariants.Rect{2}[
        TO.Invariants.Rect{2}((0, 0), (2, 2)),  # span 2 in both directions
        TO.Invariants.Rect{2}((0, 0), (1, 1)),  # span 1
        TO.Invariants.Rect{2}((1, 0), (2, 1)),  # span 1
    ]
    weights_true = [1, -1, 2]
    sb_true = TO.Invariants.RectSignedBarcode{2,Int}(axes, rects_true, weights_true)

    r_idx(pI, qI) = TO.Invariants.rank_from_signed_barcode(
        sb_true,
        ntuple(k -> axes[k][pI[k]], 2),
        ntuple(k -> axes[k][qI[k]], 2),
    )

    sb_full = SM.rectangle_signed_barcode(r_idx, axes)
    sb_span = SM.rectangle_signed_barcode(r_idx, axes; max_span=(1, 1))

    function sb_to_dict(sb)
        d = Dict{Tuple{NTuple{2,Int},NTuple{2,Int}},Int}()
        for (r, w) in zip(sb.rects, sb.weights)
            d[(r.lo, r.hi)] = w
        end
        return d
    end

    d_full = sb_to_dict(sb_full)
    d_span = sb_to_dict(sb_span)

    d_full_small = Dict{Tuple{NTuple{2,Int},NTuple{2,Int}},Int}()
    for (k, w) in d_full
        lo, hi = k
        if (hi[1] - lo[1] <= 1) && (hi[2] - lo[2] <= 1)
            d_full_small[k] = w
        end
    end

    @test d_span == d_full_small

    # -------------------------------------------------------------------------
    # 1b) Bulk vs local algorithm parity + rank_idx call counts.
    # -------------------------------------------------------------------------
    sb_local = SM.rectangle_signed_barcode(r_idx, axes; method=:local)
    sb_bulk = SM.rectangle_signed_barcode(r_idx, axes; method=:bulk)

    @test Dict(zip(sb_local.rects, sb_local.weights)) ==
          Dict(zip(sb_bulk.rects, sb_bulk.weights))

    old_bulk2d = SM._USE_PACKED_RECTANGLE_BULK_2D[]
    try
        SM._USE_PACKED_RECTANGLE_BULK_2D[] = false
        sb_bulk_dense = SM.rectangle_signed_barcode(r_idx, axes; method=:bulk)
        SM._USE_PACKED_RECTANGLE_BULK_2D[] = true
        sb_bulk_packed = SM.rectangle_signed_barcode(r_idx, axes; method=:bulk)
        @test Dict(zip(sb_bulk_dense.rects, sb_bulk_dense.weights)) ==
              Dict(zip(sb_bulk_packed.rects, sb_bulk_packed.weights))
    finally
        SM._USE_PACKED_RECTANGLE_BULK_2D[] = old_bulk2d
    end

    # Bulk calls rank_idx exactly once per comparable pair (p <= q).
    dims = map(length, axes)
    n_pairs = prod(div(d * (d + 1), 2) for d in dims)

    c_bulk = Ref(0)
    r_idx_count_bulk(pI, qI) = (c_bulk[] += 1; r_idx(pI, qI))
    _ = SM.rectangle_signed_barcode(r_idx_count_bulk, axes; method=:bulk)
    @test c_bulk[] == n_pairs

    # Auto should use the bulk path when the grid is moderate and not highly skewed.
    axes_auto = (collect(0:3), collect(0:3))
    rects_auto = TO.Invariants.Rect{2}[
        TO.Invariants.Rect{2}((0, 0), (3, 3)),
        TO.Invariants.Rect{2}((1, 1), (2, 3)),
    ]
    weights_auto = [1, -1]
    sb_true_auto = TO.Invariants.RectSignedBarcode{2,Int}(axes_auto, rects_auto, weights_auto)
    r_idx_auto(pI, qI) = TO.Invariants.rank_from_signed_barcode(
        sb_true_auto,
        ntuple(k -> axes_auto[k][pI[k]], 2),
        ntuple(k -> axes_auto[k][qI[k]], 2),
    )
    sb_auto = SM.rectangle_signed_barcode(r_idx_auto, axes_auto; method=:auto)
    sb_auto_bulk = SM.rectangle_signed_barcode(r_idx_auto, axes_auto; method=:bulk)
    @test Dict(zip(sb_auto.rects, sb_auto.weights)) ==
          Dict(zip(sb_auto_bulk.rects, sb_auto_bulk.weights))

    c_auto = Ref(0)
    r_idx_count_auto(pI, qI) = (c_auto[] += 1; r_idx_auto(pI, qI))
    dims_auto = map(length, axes_auto)
    n_pairs_auto = prod(div(d * (d + 1), 2) for d in dims_auto)
    _ = SM.rectangle_signed_barcode(r_idx_count_auto, axes_auto; method=:auto)
    @test c_auto[] == n_pairs_auto
    @test SM._choose_rectangle_signed_barcode_method(:auto, map(collect, axes_auto);
                                                     max_span=nothing,
                                                     bulk_max_elems=10_000) == :bulk

    reg_blocks = [
        1 1 1 2 2
        1 1 1 2 2
        1 1 1 2 2
        3 3 3 4 4
        3 3 3 4 4
    ]
    axes_blocks = (collect(0:4), collect(0:4))
    reg_comp, keep1, keep2, axes_comp = SM._compress_region_grid_2d(reg_blocks, axes_blocks)
    @test keep1 == [1, 3, 4, 5]
    @test keep2 == [1, 3, 4, 5]
    @test axes_comp == ([0, 2, 3, 4], [0, 2, 3, 4])
    @test reg_comp == reg_blocks[keep1, keep2]

    # Auto should stay on the local path for highly skewed axes.
    axes_skew = (collect(0:8), collect(0:1))
    rects_skew = TO.Invariants.Rect{2}[
        TO.Invariants.Rect{2}((0, 0), (8, 1)),
        TO.Invariants.Rect{2}((2, 0), (5, 0)),
    ]
    weights_skew = [1, -1]
    sb_true_skew = TO.Invariants.RectSignedBarcode{2,Int}(axes_skew, rects_skew, weights_skew)
    r_idx_skew(pI, qI) = TO.Invariants.rank_from_signed_barcode(
        sb_true_skew,
        ntuple(k -> axes_skew[k][pI[k]], 2),
        ntuple(k -> axes_skew[k][qI[k]], 2),
    )
    @test SM._choose_rectangle_signed_barcode_method(:auto, map(collect, axes_skew);
                                                     max_span=nothing,
                                                     bulk_max_elems=10_000) == :local
    sb_auto_skew = SM.rectangle_signed_barcode(r_idx_skew, axes_skew; method=:auto)
    sb_local_skew = SM.rectangle_signed_barcode(r_idx_skew, axes_skew; method=:local)
    @test Dict(zip(sb_auto_skew.rects, sb_auto_skew.weights)) ==
          Dict(zip(sb_local_skew.rects, sb_local_skew.weights))

    # Local method should make strictly more rank_idx calls on the same problem.
    c_local = Ref(0)
    r_idx_count_local(pI, qI) = (c_local[] += 1; r_idx(pI, qI))
    _ = SM.rectangle_signed_barcode(r_idx_count_local, axes; method=:local)
    @test c_local[] > n_pairs

    # With a restrictive span cutoff, auto should stay on the local path.
    @test SM._choose_rectangle_signed_barcode_method(:auto, map(collect, axes);
                                                     max_span=(1, 1),
                                                     bulk_max_elems=10_000) == :local
    sb_local_span = SM.rectangle_signed_barcode(r_idx, axes; method=:local, max_span=(1, 1))
    @test Dict(zip(sb_span.rects, sb_span.weights)) ==
          Dict(zip(sb_local_span.rects, sb_local_span.weights))

    # Fast inverse transform: barcode -> dense rank array matches rank_idx on p <= q.
    R = SM.rectangle_signed_barcode_rank(sb_bulk)
    for p1 in 1:dims[1], p2 in 1:dims[2]
        for q1 in p1:dims[1], q2 in p2:dims[2]
            @test R[p1, p2, q1, q2] == r_idx((p1, p2), (q1, q2))
        end
    end


    # -------------------------------------------------------------------------
    # 2) Axis coarsening helper keeps endpoints and respects max_len.
    # -------------------------------------------------------------------------
    ax = collect(0:9)
    axc = TO.Invariants.coarsen_axis(ax; max_len=5)
    @test length(axc) <= 5
    @test first(axc) == 0
    @test last(axc) == 9

    # -------------------------------------------------------------------------
    # 3) Wrapper-level memoization and axes_policy smoke tests (N = 1).
    #    QQ-only: relies on ZnEncoding defaults.
    # -------------------------------------------------------------------------
    if field isa CM.QQField
        R = TO.QQ
        n = 1
        flats = [
            FZ.IndFlat(TO.face(n, []), [0]),
            FZ.IndFlat(TO.face(n, []), [2]),
        ]
        injectives = [
            FZ.IndInj(TO.face(n, []), [1]),
            FZ.IndInj(TO.face(n, []), [3]),
        ]
        # Phi must be (#injectives x #flats): rows index injectives, cols index flats.
        # Here we just take the 2x2 identity (a convenient "direct sum" style choice).
        Phi = zeros(R, length(injectives), length(flats))
        Phi[1, 1] = one(R)
        Phi[2, 2] = one(R)
        F = TO.Flange{R}(n, flats, injectives, Phi)

        enc = TO.EncodingOptions(backend=:zn, max_regions=100)
        (Penc, Henc, pi) = TamerOp.ZnEncoding.encode_from_flange(F, enc)
        Menc = IR.pmodule_from_fringe(Henc)

        axes_user = (collect(-2:6),)

        # Regression test: ZnEncodingMap.coords is axis-wise critical coordinates, so
        # axes_from_encoding(pi) must return an N-tuple where N == pi.n.
        enc_axes = EC.axes_from_encoding(pi)
        @test length(enc_axes) == n
        @test enc_axes[1] == [-1, 0, 2, 4]

        cache = TamerOp.InvariantCore.RankQueryCache(pi)
        @test typeof(cache).parameters[1] == n

        cache = TamerOp.InvariantCore.RankQueryCache(pi)
        cache_desc = TO.describe(cache)
        @test cache_desc.kind == :rank_query_cache
        @test cache_desc.ambient_dim == n
        @test cache_desc.nregions == IC.nregions(cache)
        @test cache_desc.cache_layout == IC.cache_layout(cache)
        @test cache_desc.loc_cache_size == 0
        @test cache_desc.rank_cache_size == 0
        @test !cache_desc.warm
        @test IC.describe(cache) == cache_desc
        @test IC.encoding(cache) === pi
        @test IC.loc_cache_size(cache) == length(cache.loc_cache)
        @test IC.rank_cache_size(cache) == 0
        @test occursin("RankQueryCache(", sprint(show, cache))
        @test occursin("cache_layout", sprint(show, MIME"text/plain"(), cache))
        @test IC.check_rank_query_cache(cache).valid
        @test occursin("reuse one cache per encoding", lowercase(string(@doc TamerOp.InvariantCore.RankQueryCache)))

        # rank_query on ZnEncodingMap should match rank_map and reuse RankQueryCache.
        rq_opts = TO.InvariantOptions()
        qx = [-2]
        qy = [8]
        rq_xy = Inv.rank_query(Menc, pi, qx, qy, rq_opts; rq_cache=cache)
        @test rq_xy == Inv.rank_map(Menc, pi, qx, qy, rq_opts)
        @test Inv.rank_query(Menc, pi, qx, qy; opts=rq_opts, rq_cache=cache) == rq_xy

        a = EC.locate(pi, qx)
        b = EC.locate(pi, qy)
        @test Inv.rank_query(Menc, pi, a, b; rq_cache=cache) == Inv.rank_map(Menc, a, b)
        @test IC.loc_cache_size(cache) > 0
        _ = IC._rank_cache_get!(cache, a, b) do
            7
        end
        warm_desc = TO.describe(cache)
        @test warm_desc.warm
        @test warm_desc.loc_cache_size == IC.loc_cache_size(cache)
        @test warm_desc.rank_cache_size == IC.rank_cache_size(cache)

        pi_comp = TO.EncodingCore.compile_encoding(Penc, pi)
        @test Inv.rank_query(Menc, pi_comp, qx, qy, rq_opts; rq_cache=cache) == rq_xy
        @test Inv.rank_query(Menc, pi_comp, a, b; rq_cache=cache) == Inv.rank_map(Menc, a, b)

        cache_bad = TamerOp.InvariantCore.RankQueryCache(pi)
        cache_bad.n_regions += 1
        cache_bad_report = IC.check_rank_query_cache(cache_bad)
        @test !cache_bad_report.valid
        @test !isempty(cache_bad_report.issues)
        @test_throws ArgumentError IC.check_rank_query_cache(cache_bad; throw=true)

        sb_enc_1 = SM.rectangle_signed_barcode(Menc, pi;
            axes=axes_user,
            axes_policy=:encoding,
            rq_cache=cache)

        sb_enc_2 = SM.rectangle_signed_barcode(Menc, pi;
            axes=axes_user,
            axes_policy=:encoding,
            rq_cache=cache)
        sb_enc_span_1 = SM.rectangle_signed_barcode(Menc, pi;
            axes=axes_user,
            axes_policy=:encoding,
            rq_cache=cache,
            max_span=0)
        sb_enc_span_2 = SM.rectangle_signed_barcode(Menc, pi;
            axes=axes_user,
            axes_policy=:encoding,
            rq_cache=cache,
            max_span=0)

        # Cache reuse must not change the output.
        @test sb_enc_1.rects == sb_enc_2.rects
        @test sb_enc_1.weights == sb_enc_2.weights
        @test sb_enc_1.axes == sb_enc_2.axes
        @test sb_enc_1 === sb_enc_2
        @test sb_enc_span_1 === sb_enc_span_2
        @test sb_enc_span_1 !== sb_enc_1

        sc = CM.SessionCache()
        sb_sc_1 = SM.rectangle_signed_barcode(Menc, pi, rq_opts; cache=sc)
        sb_sc_2 = SM.rectangle_signed_barcode(Menc, pi, rq_opts; cache=sc)
        sb_sc_span_1 = SM.rectangle_signed_barcode(Menc, pi, rq_opts; cache=sc, max_span=0)
        sb_sc_span_2 = SM.rectangle_signed_barcode(Menc, pi, rq_opts; cache=sc, max_span=0)
        @test sb_sc_1.rects == sb_sc_2.rects
        @test sb_sc_1.weights == sb_sc_2.weights
        @test sb_sc_1.axes == sb_sc_2.axes
        @test sb_sc_1 === sb_sc_2
        @test sb_sc_span_1 === sb_sc_span_2
        @test sb_sc_span_1 !== sb_sc_1
        @test SM._signed_measures_rank_query_cache(Menc, pi, nothing, sc) ===
              SM._signed_measures_rank_query_cache(Menc, pi, nothing, sc)

        surf_sc_1 = SM.euler_surface(Menc, pi, rq_opts; cache=sc)
        surf_sc_2 = SM.euler_surface(Menc, pi, rq_opts; cache=sc)
        @test surf_sc_1 == surf_sc_2
        surf_cached_1, _ = SM._cached_euler_characteristic_surface(Menc, pi, rq_opts, sc)
        surf_cached_2, _ = SM._cached_euler_characteristic_surface(Menc, pi, rq_opts, sc)
        @test surf_cached_1 === surf_cached_2

        outE_sc = SM.mma_decomposition(Menc, pi, rq_opts; method=:euler, cache=sc)
        @test outE_sc.euler_surface == surf_sc_1
        @test SM.surface_from_point_signed_measure(outE_sc.euler_signed_measure) == surf_sc_1

        # encoding restriction should keep endpoints and only include encoding-axis points in between.
        enc_axes = EC.axes_from_encoding(pi)[1]
        @test first(sb_enc_1.axes[1]) == first(axes_user[1])
        @test last(sb_enc_1.axes[1]) == last(axes_user[1])
        @test all(v == first(axes_user[1]) || v == last(axes_user[1]) || (v in enc_axes) for v in sb_enc_1.axes[1])

        # coarsen policy must reduce axis length to max_axis_len.
        sb_coarse = SM.rectangle_signed_barcode(Menc, pi;
            axes=axes_user,
            axes_policy=:coarsen,
            max_axis_len=4)
        @test length(sb_coarse.axes[1]) <= 4

        # Bulk and local algorithms must agree (modulo ordering and zero pruning).
        sb_enc_bulk = SM.rectangle_signed_barcode(Menc, pi; axes=axes_user, axes_policy=:encoding,
                                                  rq_cache=cache, method=:bulk)
        sb_enc_local = SM.rectangle_signed_barcode(Menc, pi; axes=axes_user, axes_policy=:encoding,
                                                   rq_cache=cache, method=:local)
        @test Dict(zip(sb_enc_bulk.rects, sb_enc_bulk.weights)) ==
              Dict(zip(sb_enc_local.rects, sb_enc_local.weights))

        reg_enc = SM._rectangle_region_grid(pi, sb_enc_bulk.axes, cache; strict=false)

        sc_blocks = CM.SessionCache()
        reg_cached_1 = SM._cached_rectangle_region_grid(pi, sb_enc_bulk.axes, cache, sc_blocks; strict=false)
        reg_cached_2 = SM._cached_rectangle_region_grid(pi, sb_enc_bulk.axes, cache, sc_blocks; strict=false)
        @test reg_cached_1 === reg_cached_2
        if length(sb_enc_bulk.axes) == 2
            reg_comp_enc, keep1_enc, keep2_enc, axes_comp_enc =
                SM._compress_region_grid_2d(reg_enc, sb_enc_bulk.axes)
            blocks_cached_1 = SM._cached_rectangle_region_blocks_2d(pi, sb_enc_bulk.axes, cache, sc_blocks; strict=false)
            blocks_cached_2 = SM._cached_rectangle_region_blocks_2d(pi, sb_enc_bulk.axes, cache, sc_blocks; strict=false)
            @test blocks_cached_1 === blocks_cached_2
            @test blocks_cached_1 == (reg_comp_enc, keep1_enc, keep2_enc, axes_comp_enc)
        end

        old_bulk2d = SM._USE_PACKED_RECTANGLE_BULK_2D[]
        try
            SM._USE_PACKED_RECTANGLE_BULK_2D[] = false
            sb_enc_bulk_dense = SM.rectangle_signed_barcode(Menc, pi; axes=axes_user, axes_policy=:encoding,
                                                            rq_cache=cache, method=:bulk)
            SM._USE_PACKED_RECTANGLE_BULK_2D[] = true
            sb_enc_bulk_packed = SM.rectangle_signed_barcode(Menc, pi; axes=axes_user, axes_policy=:encoding,
                                                             rq_cache=cache, method=:bulk)
            @test Dict(zip(sb_enc_bulk_dense.rects, sb_enc_bulk_dense.weights)) ==
                  Dict(zip(sb_enc_bulk_packed.rects, sb_enc_bulk_packed.weights))
            @test sb_enc_bulk_packed.axes == sb_enc_bulk_dense.axes == sb_enc_bulk.axes
        finally
            SM._USE_PACKED_RECTANGLE_BULK_2D[] = old_bulk2d
        end

        grid_pts = ((1, 1), (2, 1), (1, 2), (2, 2))
        grid_leq = falses(4, 4)
        for i in eachindex(grid_pts), j in eachindex(grid_pts)
            grid_leq[i, j] = grid_pts[i][1] <= grid_pts[j][1] && grid_pts[i][2] <= grid_pts[j][2]
        end
        Pgrid = FF.FinitePoset(grid_leq)
        pi_grid = EC.GridEncodingMap(Pgrid, ([0.0, 1.0], [0.0, 2.0]))
        Hgrid = FF.one_by_one_fringe(
            Pgrid,
            FF.principal_upset(Pgrid, 2),
            FF.principal_downset(Pgrid, 4),
            CM.QQ(1);
            field=field,
        )
        Mgrid = IR.pmodule_from_fringe(Hgrid)
        grid_opts = TO.InvariantOptions()

        sb_grid = SM.rectangle_signed_barcode(Mgrid, pi_grid, grid_opts)
        sb_grid_comp = SM.rectangle_signed_barcode(Mgrid, EC.compile_encoding(Pgrid, pi_grid), grid_opts)
        @test sb_grid.axes == ([1, 2], [1, 2])
        @test Dict(zip(sb_grid.rects, sb_grid.weights)) ==
              Dict(zip(sb_grid_comp.rects, sb_grid_comp.weights))

        grid_lin(t) = IC._grid_cache_index(t, pi_grid.sizes)
        for p in grid_pts, q in grid_pts
            if p[1] <= q[1] && p[2] <= q[2]
                @test SM.rank_from_signed_barcode(sb_grid, p, q) ==
                      Inv.rank_map(Mgrid, grid_lin(p), grid_lin(q))
            end
        end

        sc_grid = CM.SessionCache()
        sb_grid_sc_1 = SM.rectangle_signed_barcode(Mgrid, pi_grid, grid_opts; cache=sc_grid)
        sb_grid_sc_2 = SM.rectangle_signed_barcode(Mgrid, pi_grid, grid_opts; cache=sc_grid)
        @test sb_grid_sc_1 === sb_grid_sc_2

        enc_grid = RES.EncodingResult(Pgrid, Mgrid, pi_grid; backend=:test)
        sb_grid_workflow = TamerOp.rectangle_signed_barcode(enc_grid; opts=grid_opts, cache=sc_grid)
        @test Dict(zip(sb_grid_workflow.rects, sb_grid_workflow.weights)) ==
              Dict(zip(sb_grid.rects, sb_grid.weights))
        @test TamerOp.encoding_axes(enc_grid) == pi_grid.coords

        @testset "Workflow rank_map oracle on shared-grid benchmark pairs" begin
            _point_cloud(mat::Matrix{Float64}) =
                TO.PointCloud([Vector{Float64}(view(mat, i, :)) for i in 1:size(mat, 1)])

            function _direct_algebraic_rank(enc::RES.EncodingResult, p::AbstractVector{<:Real}, q::AbstractVector{<:Real})
                Mloc = TO.pmodule(enc)
                piloc = TO.classifier(enc)
                a = EC.locate(piloc, p)
                b = EC.locate(piloc, q)
                a != 0 || error("direct oracle: p mapped outside the encoding")
                b != 0 || error("direct oracle: q mapped outside the encoding")
                FF.leq(Mloc.Q, a, b) || error("direct oracle: located regions are not comparable")
                FF.build_cache!(Mloc.Q; cover=true, updown=false)
                A = MD.map_leq(Mloc, a, b)
                return TamerOp.FieldLinAlg.rank(Mloc.field, A)
            end

            codensity_pts = Float64[
                -0.43936001 -0.76934895;
                -0.94057973 -0.06897480;
                 0.41005922 -0.86576552;
                -0.89062056 -0.30745033;
                 0.91480650 -0.12215502;
                 0.30342685  0.94477498;
                -0.99848830 -0.38373306;
                -0.99706795  0.10379965;
            ]
            codensity_spec = TO.FiltrationSpec(
                kind=:rips_codensity,
                max_dim=1,
                radius=1.86814163652,
                dtm_mass=0.5,
                construction=TO.ConstructionOptions(
                    sparsify=:radius,
                    collapse=:none,
                    output_stage=:encoding_result,
                ),
            )
            codensity_enc = TO.encode(_point_cloud(codensity_pts), codensity_spec; degree=0)
            codensity_p = [0.0, 0.4046961549013076]
            codensity_q = [0.13211549579425005, 0.4046961549013076]
            codensity_workflow = TO.rank_map(TO.pmodule(codensity_enc), TO.classifier(codensity_enc), codensity_p, codensity_q, TO.InvariantOptions())
            codensity_direct = _direct_algebraic_rank(codensity_enc, codensity_p, codensity_q)
            @test codensity_workflow == codensity_direct == 1

            lowerstar_pts = Float64[
                 1.05321915 -0.09836355;
                 0.28295657  0.96244730;
                -0.87947945 -0.30389624;
                -1.04338067  0.11311428;
                -0.64354448  0.85015295;
                -0.84057703 -0.56110545;
                 0.06055744  1.04055769;
                 0.34483588 -0.99386999;
                 0.67444726 -0.76649590;
                 0.64174181  0.69523216;
            ]
            lowerstar_spec = TO.FiltrationSpec(
                kind=:rips_lowerstar,
                max_dim=1,
                radius=2.0700079356,
                coord=1,
                construction=TO.ConstructionOptions(
                    sparsify=:radius,
                    collapse=:none,
                    output_stage=:encoding_result,
                ),
            )
            lowerstar_enc = TO.encode(_point_cloud(lowerstar_pts), lowerstar_spec; degree=0)
            lowerstar_p = [0.0, 0.06055744]
            lowerstar_q = [0.23571721627982334, 0.28295657]
            lowerstar_workflow = TO.rank_map(TO.pmodule(lowerstar_enc), TO.classifier(lowerstar_enc), lowerstar_p, lowerstar_q, TO.InvariantOptions())
            lowerstar_direct = _direct_algebraic_rank(lowerstar_enc, lowerstar_p, lowerstar_q)
            lowerstar_vals = lowerstar_pts[:, 1]
            lowerstar_radius = 2.0700079356
            lowerstar_edges = Tuple{Int,Int,Float64}[]
            for i in 1:size(lowerstar_pts, 1), j in (i + 1):size(lowerstar_pts, 1)
                dij = norm(view(lowerstar_pts, i, :) .- view(lowerstar_pts, j, :))
                dij <= lowerstar_radius || continue
                push!(lowerstar_edges, (i, j, dij))
            end

            function _lowerstar_active_vertices(x::AbstractVector{<:Real})
                x[1] >= -1e-12 || return Int[]
                out = Int[]
                for i in eachindex(lowerstar_vals)
                    lowerstar_vals[i] <= x[2] + 1e-12 || continue
                    push!(out, i)
                end
                return out
            end

            function _lowerstar_active_components(x::AbstractVector{<:Real})
                verts = _lowerstar_active_vertices(x)
                present = falses(length(lowerstar_vals))
                for v in verts
                    present[v] = true
                end
                adj = [Int[] for _ in eachindex(lowerstar_vals)]
                for (i, j, dij) in lowerstar_edges
                    dij <= x[1] + 1e-12 || continue
                    max(lowerstar_vals[i], lowerstar_vals[j]) <= x[2] + 1e-12 || continue
                    present[i] && present[j] || continue
                    push!(adj[i], j)
                    push!(adj[j], i)
                end
                seen = falses(length(lowerstar_vals))
                comps = Vector{Vector{Int}}()
                for v in verts
                    seen[v] && continue
                    stack = [v]
                    seen[v] = true
                    comp = Int[]
                    while !isempty(stack)
                        u = pop!(stack)
                        push!(comp, u)
                        for w in adj[u]
                            seen[w] && continue
                            seen[w] = true
                            push!(stack, w)
                        end
                    end
                    push!(comps, sort(comp))
                end
                return comps
            end

            lowerstar_q_components = _lowerstar_active_components(lowerstar_q)
            lowerstar_p_vertices = Set(_lowerstar_active_vertices(lowerstar_p))
            lowerstar_oracle = count(comp -> any(v -> v in lowerstar_p_vertices, comp), lowerstar_q_components)

            @test lowerstar_oracle == 5
            @test lowerstar_workflow == lowerstar_direct == lowerstar_oracle == 5
        end
    end

end

@testset "Defaults that unblock usability (matching distance and sliced Wasserstein)" begin

    # -------------------------
    # PLBackend: defaults from pi.reps
    # -------------------------
    Ups = [PLB.BoxUpset([0.0, 0.0], [1.0, 1.0])]
    Downs = [PLB.BoxDownset([2.0, 2.0], [3.0, 3.0])]

    P, H, pi = PLB.encode_fringe_boxes(Ups, Downs)
    M = IR.pmodule_from_fringe(H)
    opts = TO.InvariantOptions()

    # Out-of-the-box: no directions, no offsets.
    dmatch = Inv.matching_distance_approx(M, M, pi, opts)
    @test isapprox(dmatch, 0.0; atol=1e-12)

    dsw = Inv.sliced_wasserstein_distance(M, M, pi, opts)
    @test isapprox(dsw, 0.0; atol=1e-12)

    # Bounding box extraction: derived from pi.reps (finite points).
    lo, hi = Inv.encoding_box(pi, opts; margin=0.0)
    reps = pi.reps
    d = length(reps[1])
    lo_expected = [minimum(float(r[i]) for r in reps) for i in 1:d]
    hi_expected = [maximum(float(r[i]) for r in reps) for i in 1:d]
    @test lo == lo_expected
    @test hi == hi_expected

    # default_offsets: correct count and dimension in float case
    opts_box = TO.InvariantOptions(box=(lo, hi))
    offs = Inv.default_offsets(pi, opts_box; n_offsets=7, margin=0.0)
    @test length(offs) == 7
    @test all(length(x) == d for x in offs)

    # default_directions: correct count and dimension
    dirs = Inv.default_directions(pi; n_dirs=5, max_den=4, include_axes=false, normalize=:L1)
    @test length(dirs) == 5
    @test all(length(v) == d for v in dirs)

    # -------------------------
    # PLPolyhedra: defaults from pi.witnesses
    # -------------------------
    if PLP.HAVE_POLY && (field isa CM.QQField)
        # Lightweight PLPolyhedra smoke path:
        # avoid full polyhedral encoding solve in this cross-field defaults test.
        hp = PLP.make_hpoly([-1.0 0.0; 0.0 -1.0; 1.0 0.0; 0.0 1.0], [0.0, 0.0, 2.0, 2.0])
        pi2 = PLP.PLEncodingMap(2,
            [BitVector([false])],
            [BitVector([false])],
            [hp],
            [(1.0, 1.0)])

        P2 = chain_poset(1)
        H2 = one_by_one_fringe(P2, FF.principal_upset(P2, 1), FF.principal_downset(P2, 1); field=field)
        M2 = IR.pmodule_from_fringe(H2)
        opts2 = TO.InvariantOptions()

        # Keep this branch lightweight: default witness/box behavior is what we
        # validate here; heavy distance kernels are covered elsewhere.
        @test EC.dimension(pi2) == 2
        @test length(M2.dims) == 1

        lo2, hi2 = Inv.encoding_box(pi2, opts2; margin=0.0)
        wit = pi2.witnesses
        d2 = length(wit[1])
        lo2_expected = [minimum(float(w[i]) for w in wit) for i in 1:d2]
        hi2_expected = [maximum(float(w[i]) for w in wit) for i in 1:d2]
        @test lo2 == lo2_expected
        @test hi2 == hi2_expected
    end

    @test Inv.matching_distance_approx(M, M, pi, opts) == 0.0
    @test Inv.sliced_wasserstein_distance(M, M, pi, opts) == 0.0

    # New: Lp-generalization family members should also work out-of-the-box.
    @test Inv.sliced_bottleneck_distance(M, M, pi, opts) == 0.0
    @test Inv.matching_wasserstein_distance_approx(M, M, pi, opts) == 0.0

    if PLP.HAVE_POLY && (field isa CM.QQField)
        @test EC.dimension(pi2) == 2
    end

    @testset "Exact 2D matching distance: deterministic and correct on a toy example" begin
        # Toy coords-based encoding map:
        # Box [0,3]x[0,3] partitioned into three vertical stripes:
        #   region 1: x1 in [0,1), region 2: x1 in [1,2), region 3: x1 in [2,3]
        #
        # For modules M23 vs M3 on the chain 1<=2<=3, the exact Lesnick matching distance
        # over this box is 1.0 (achieved for sufficiently steep directions where weight=dir1
        # and bottleneck scales as 1/dir1).
        reps = [(0.5, 1.5), (1.5, 1.5), (2.5, 1.5)]
        pi = ToyBoxes2D(([0.0, 1.0, 2.0, 3.0], [0.0, 3.0]), reps)
        box = ([0.0, 0.0], [3.0, 3.0])
        opts_exact = TO.InvariantOptions(box=box)

        P = chain_poset(3)
        M23 = IR.pmodule_from_fringe(one_by_one_fringe(P, FF.principal_upset(P, 2), FF.principal_downset(P, 3); field=field))
        M3  = IR.pmodule_from_fringe(one_by_one_fringe(P, FF.principal_upset(P, 3), FF.principal_downset(P, 3); field=field))

        # Slice-chain exactness sanity check at one noncritical line:
        # dir = (1,1) normalized L1 -> (0.5,0.5).
        # offset chosen so y-x = 0.5 (i.e. dot([-0.5,0.5], x) = 0.25).
        chain, vals = TO.slice_chain_exact_2d(pi, [1.0, 1.0], 0.25, opts_exact; normalize_dirs=:L1)
        @test chain == [1,2,3]
        @test isapprox(vals[1], 0.5; atol=1e-12)
        @test isapprox(vals[2], 2.5; atol=1e-12)
        @test isapprox(vals[3], 4.5; atol=1e-12)
        @test isapprox(vals[4], 5.5; atol=1e-12)

        # Exact distance should be 1.0 on this toy configuration.
        d1 = Inv.matching_distance_exact_2d(M23, M3, pi, opts_exact; weight=:lesnick_l1, normalize_dirs=:L1)
        d2 = Inv.matching_distance_exact_2d(M23, M3, pi, opts_exact; weight=:lesnick_l1, normalize_dirs=:L1)
        @test d1 == d2  # determinism
        @test isapprox(d1, 1.0; atol=1e-10)

        # Agreement with the slice-based evaluator using the exact slice list.
        fam = Inv.matching_distance_exact_slices_2d(pi, opts_exact; normalize_dirs=:L1)
        d3 = Inv.matching_distance_approx(M23, M3, fam.slices)
        @test isapprox(d3, d1; atol=1e-12)

        # Toy polyhedral encoding: same three vertical stripes, expressed as HPolys.
        # This exercises the polyhedral exact slice extraction path.
        if field isa CM.QQField
            function hpoly_box(xlo, xhi, ylo, yhi)
                A = Matrix{TO.QQ}(undef, 4, 2)
                b = Vector{TO.QQ}(undef, 4)
                # x >= xlo  <=>  -x <= -xlo
                A[1,1] = -1; A[1,2] =  0; b[1] = -cf(xlo)
                # x <= xhi
                A[2,1] =  1; A[2,2] =  0; b[2] =  cf(xhi)
                # y >= ylo  <=>  -y <= -ylo
                A[3,1] =  0; A[3,2] = -1; b[3] = -cf(ylo)
                # y <= yhi
                A[4,1] =  0; A[4,2] =  1; b[4] =  cf(yhi)
                # Canonical HPoly constructor: always pass strictness data explicitly.
                strict_mask = falses(size(A, 1))
                return TO.PLPolyhedra.HPoly(2, A, b, nothing, strict_mask, TO.PLPolyhedra.STRICT_EPS_QQ)
            end

            hp1 = hpoly_box(0.0, 1.0, 0.0, 3.0)
            hp2 = hpoly_box(1.0, 2.0, 0.0, 3.0)
            hp3 = hpoly_box(2.0, 3.0, 0.0, 3.0)

            sigy = [BitVector([false]), BitVector([false]), BitVector([false])]
            sigz = [BitVector([false]), BitVector([false]), BitVector([false])]
            witnesses = [(0.5, 1.5), (1.5, 1.5), (2.5, 1.5)]

            pi_poly = TO.PLPolyhedra.PLEncodingMap(2, sigy, sigz, [hp1, hp2, hp3], witnesses)

            d_poly = Inv.matching_distance_exact_2d(M23, M3, pi_poly, opts_exact; weight=:lesnick_l1, normalize_dirs=:L1)
            @test isapprox(d_poly, 1.0; atol=1e-10)
        end

        # Shared cache for downstream tests.
        cache_full = Inv.fibered_barcode_cache_2d(M23, pi, opts_exact;
            normalize_dirs=:L1, precompute=:full)

        @testset "Fibered barcode cache 2D: augmented arrangement" begin
            # Reuse cache_full built above.

            bar = Inv.fibered_barcode(cache_full, [1.0, 1.0], 0.25; values=:t)
            @test bar == Dict((2.5, 5.5) => 1)

            bar_idx = Inv.fibered_barcode_index(cache_full, [1.0, 1.0], 0.25)
            @test bar_idx == Dict((2, 4) => 1)

            # basepoint query matches offset query
            x0 = [-0.25, 0.25]
            @test Inv.fibered_barcode(cache_full, [1.0, 1.0], x0; values=:t) == bar

            # scaling direction does not change the slice (after normalization)
            @test Inv.fibered_barcode(cache_full, [2.0, 2.0], 0.25; values=:t) == bar

            st = Inv.fibered_barcode_cache_stats(cache_full)
            @test st.total_cells == st.n_cells_computed
            @test st.n_chains > 0
            @test st.n_index_barcodes_computed == st.n_chains

            # Lazy cache: ensure first query computes exactly one cell
            cache_lazy = Inv.fibered_barcode_cache_2d(M23, pi, opts_exact;
                normalize_dirs=:L1, precompute=:none)

            st0 = Inv.fibered_barcode_cache_stats(cache_lazy)
            @test st0.n_cells_computed == 0

            bar_lazy = Inv.fibered_barcode(cache_lazy, [1.0, 1.0], 0.25; values=:t)
            @test bar_lazy == bar

            st1 = Inv.fibered_barcode_cache_stats(cache_lazy)
            @test st1.n_cells_computed == 1

            _ = Inv.fibered_barcode(cache_lazy, [1.0, 1.0], 0.25; values=:t)
            st2 = Inv.fibered_barcode_cache_stats(cache_lazy)
            @test st2.n_cells_computed == 1

            fam_default = Inv.fibered_slice_family_2d(cache_full.arrangement;
                direction_weight=:lesnick_l1, store_values=true)
            ns_default = length(fam_default.cell_id)

            cache_family = Inv.fibered_barcode_cache_2d(M23, cache_full.arrangement;
                precompute=:family)
            st_family = Inv.fibered_barcode_cache_stats(cache_family)
            @test st_family.n_cells_computed == st_family.total_cells
            @test st_family.n_slice_families_cached >= 1
            @test st_family.n_index_barcodes_computed == 0
            @test st_family.n_family_barcode_slices_computed == 0
            @test st_family.n_distance_slices_computed == 0

            cache_barcodes = Inv.fibered_barcode_cache_2d(M23, cache_full.arrangement;
                precompute=:barcodes)
            st_barcodes = Inv.fibered_barcode_cache_stats(cache_barcodes)
            @test st_barcodes.n_family_barcode_payloads >= 1
            @test st_barcodes.n_family_barcode_slices_computed == ns_default
            @test st_barcodes.n_distance_slices_computed == 0
            @test Inv.fibered_barcode(cache_barcodes, [1.0, 1.0], 0.25; values=:t) == bar

            cache_distance = Inv.fibered_barcode_cache_2d(M23, cache_full.arrangement;
                precompute=:distance)
            st_distance = Inv.fibered_barcode_cache_stats(cache_distance)
            @test st_distance.n_family_barcode_slices_computed == 0
            @test st_distance.n_distance_slices_computed == ns_default
            @test Inv.fibered_barcode(cache_distance, [1.0, 1.0], 0.25; values=:t) == bar

            # Polyhedral backend: index barcodes must agree exactly
            if field isa CM.QQField
                cache_poly = Inv.fibered_barcode_cache_2d(M23, pi_poly, opts_exact;
                    normalize_dirs=:L1, precompute=:full)

                bar_poly_idx = Inv.fibered_barcode_index(cache_poly, [1.0, 1.0], 0.25)
                @test bar_poly_idx == Dict((2, 4) => 1)

                bar_poly = Inv.fibered_barcode(cache_poly, [1.0, 1.0], 0.25; values=:t)
                kv = collect(bar_poly)
                @test length(kv) == 1
                ((b, dth), mult) = kv[1]
                @test mult == 1
                @test isapprox(b, 2.5; atol=1e-8)
                @test isapprox(dth, 5.5; atol=1e-8)
            end

            # Convenience wrapper: batched barcode queries on the cache.
            #
            # We support scalar (normal-offset) queries and basepoint queries.
            dirs = [[1.0, 1.0], [1.0, 2.0]]
            offs = Any[0.25, x0]
            sb = Inv.slice_barcodes(cache_full; directions=dirs, offsets=offs,
                                    values=:t, direction_weight=:none,
                                    normalize_weights=true)
            @test size(sb.barcodes) == (2, 2)
            @test size(sb.weights) == (2, 2)
            @test eltype(sb.barcodes) == Inv.FloatBarcode
            @test isapprox(sum(sb.weights), 1.0; atol=1e-12)
            @test all(isapprox.(sb.weights, 0.25; atol=1e-12))

            sb_packed = Inv.slice_barcodes(cache_full; directions=dirs, offsets=offs,
                                           values=:t, packed=true, direction_weight=:none,
                                           normalize_weights=true)
            @test sb_packed.barcodes isa Inv.PackedBarcodeGrid{Inv.PackedFloatBarcode}
            @test size(sb_packed.barcodes) == size(sb.barcodes)
            @test SI._barcode_from_packed(sb_packed.barcodes[1, 1]) == sb.barcodes[1, 1]

            sb_idx = Inv.slice_barcodes(cache_full; directions=dirs, offsets=offs,
                                        values=:index, direction_weight=:none,
                                        normalize_weights=true)
            @test eltype(sb_idx.barcodes) == Inv.IndexBarcode
            sb_idx_packed = Inv.slice_barcodes(cache_full; directions=dirs, offsets=offs,
                                               values=:index, packed=true, direction_weight=:none,
                                               normalize_weights=true)
            @test sb_idx_packed.barcodes isa Inv.PackedBarcodeGrid{Inv.PackedIndexBarcode}
            @test SI._barcode_from_packed(sb_idx_packed.barcodes[1, 1]) == sb_idx.barcodes[1, 1]

            if Threads.nthreads() > 1
                S_serial = TO.slice_barcodes(cache_full, dirs, offs; threads = false)
                S_thread = TO.slice_barcodes(cache_full, dirs, offs; threads = true)
                @test S_thread.barcodes == S_serial.barcodes
                @test S_thread.weights == S_serial.weights
            end

            # Each entry agrees with the single-query API.
            @test sb.barcodes[1, 1] == Inv.fibered_barcode(cache_full, dirs[1], 0.25; values=:t)
            @test sb.barcodes[1, 2] == Inv.fibered_barcode(cache_full, dirs[1], x0; values=:t)

            # Cache-aware exact matching distance: reuse a shared arrangement.
            arr_shared = Inv.fibered_arrangement_2d(pi, opts_exact; normalize_dirs=:L1, precompute=:cells)
            cache23 = Inv.fibered_barcode_cache_2d(M23, arr_shared; precompute=:full)
            cache3  = Inv.fibered_barcode_cache_2d(M3,  arr_shared; precompute=:full)

            d_cache = Inv.matching_distance_exact_2d(cache23, cache3; weight=:lesnick_l1)
            @test isapprox(d_cache, d1; atol=1e-12)

            fam_shared = Inv.fibered_slice_family_2d(arr_shared;
                direction_weight=:lesnick_l1, store_values=true)
            ns_shared = length(fam_shared.cell_id)
            d_direct_reuse = Inv.matching_distance_exact_2d(M23, M3, pi, opts_exact;
                weight=:lesnick_l1,
                normalize_dirs=:L1,
                arrangement=arr_shared,
                family=fam_shared,
            )
            @test isapprox(d_direct_reuse, d1; atol=1e-12)
            cache23_family = Inv.fibered_barcode_cache_2d(M23, arr_shared; precompute=:family)
            cache3_family = Inv.fibered_barcode_cache_2d(M3, arr_shared; precompute=:family)
            st_dist0 = Inv.fibered_barcode_cache_stats(cache23_family)
            @test st_dist0.n_distance_slices_computed == 0

            d_cache_first = Inv.matching_distance_exact_2d(cache23_family, cache3_family;
                weight=:lesnick_l1, family=fam_shared, threads=false)
            st_dist1 = Inv.fibered_barcode_cache_stats(cache23_family)
            @test isapprox(d_cache_first, d1; atol=1e-12)
            @test st_dist1.n_distance_slices_computed == 0

            d_cache_second = Inv.matching_distance_exact_2d(cache23_family, cache3_family;
                weight=:lesnick_l1, family=fam_shared, threads=false)
            st_dist2 = Inv.fibered_barcode_cache_stats(cache23_family)
            @test isapprox(d_cache_second, d1; atol=1e-12)
            @test st_dist2.n_distance_slices_computed == st_dist1.n_distance_slices_computed

            k_cache_first = Inv.slice_kernel(cache23_family, cache3_family;
                kind=:bottleneck_gaussian, sigma=1.0,
                direction_weight=:lesnick_l1, family=fam_shared, threads=false)
            k_ref = Inv.slice_kernel(cache23, cache3;
                kind=:bottleneck_gaussian, sigma=1.0,
                direction_weight=:lesnick_l1, family=fam_shared, threads=false)
            st_kernel1 = Inv.fibered_barcode_cache_stats(cache23_family)
            @test isapprox(k_cache_first, k_ref; atol=1e-12)
            @test st_kernel1.n_distance_slices_computed == ns_shared
            @test st_kernel1.n_family_barcode_slices_computed == 0

            k_cache_second = Inv.slice_kernel(cache23_family, cache3_family;
                kind=:bottleneck_gaussian, sigma=1.0,
                direction_weight=:lesnick_l1, family=fam_shared, threads=false)
            st_kernel2 = Inv.fibered_barcode_cache_stats(cache23_family)
            @test isapprox(k_cache_second, k_cache_first; atol=1e-12)
            @test st_kernel2.n_distance_slices_computed == st_kernel1.n_distance_slices_computed
            @test st_kernel2.n_family_barcode_slices_computed == 0

            cache23_distance = Inv.fibered_barcode_cache_2d(M23, arr_shared; precompute=:distance)
            cache3_distance = Inv.fibered_barcode_cache_2d(M3, arr_shared; precompute=:distance)
            st_distance0 = Inv.fibered_barcode_cache_stats(cache23_distance)
            @test st_distance0.n_family_barcode_slices_computed == 0
            @test st_distance0.n_distance_slices_computed == ns_shared

            k_distance_pre = Inv.slice_kernel(cache23_distance, cache3_distance;
                kind=:bottleneck_gaussian, sigma=1.0,
                direction_weight=:lesnick_l1, family=fam_shared, threads=false)
            st_distance1 = Inv.fibered_barcode_cache_stats(cache23_distance)
            @test isapprox(k_distance_pre, k_ref; atol=1e-12)
            @test st_distance1.n_family_barcode_slices_computed == 0
            @test st_distance1.n_distance_slices_computed == ns_shared

            if Threads.nthreads() > 1
                d_serial = Inv.matching_distance_exact_2d(cache23, cache3; threads = false)
                d_thread = Inv.matching_distance_exact_2d(cache23, cache3; threads = true)
                @test d_thread == d_serial
            end

            if Threads.nthreads() > 1
                # Threading parity: precompute paths
                arr_s = Inv.fibered_arrangement_2d(pi, opts_exact;
                                                   normalize_dirs=:L1,
                                                   precompute=:cells,
                                                   threads=false)
                arr_t = Inv.fibered_arrangement_2d(pi, opts_exact;
                                                   normalize_dirs=:L1,
                                                   precompute=:cells,
                                                   threads=true)

                cache_s = Inv.fibered_barcode_cache_2d(M23, arr_s; precompute=:full, threads=false)
                cache_t = Inv.fibered_barcode_cache_2d(M23, arr_t; precompute=:full, threads=true)

                st_s = Inv.fibered_barcode_cache_stats(cache_s)
                st_t = Inv.fibered_barcode_cache_stats(cache_t)
                @test st_s.total_cells == st_t.total_cells
                @test st_s.n_cells_computed == st_t.n_cells_computed
                @test st_s.n_index_barcodes_computed == st_t.n_index_barcodes_computed
            end

            # Arrangement-exact sliced kernel: compare to the slice-list backend.
            fam2 = Inv.matching_distance_exact_slices_2d(pi, opts_exact; normalize_dirs=:L1)
            slices2 = fam2.slices
            k_slices = Inv.slice_kernel(M23, M3, slices2; kind=:bottleneck_gaussian, sigma=1.0)
            k_cache  = Inv.slice_kernel(cache23, cache3;
                                        kind=:bottleneck_gaussian, sigma=1.0,
                                        direction_weight=:lesnick_l1, cell_weight=:uniform)
            @test isapprox(k_cache, k_slices; atol=1e-12)

            if Threads.nthreads() > 1
                k_serial = Inv.slice_kernel(cache23, cache3; kind = :wasserstein_gaussian, sigma = 1.0, threads = false)
                k_thread = Inv.slice_kernel(cache23, cache3; kind = :wasserstein_gaussian, sigma = 1.0, threads = true)
                @test isapprox(k_thread, k_serial; rtol = 1e-12, atol = 1e-12)
            end
        end

        @testset "Multiparameter persistence images (Carriere MPPI)" begin
            # Reuse cache_full built above in this testset.

            imgA = Inv.mpp_image(cache_full; resolution=16, sigma=0.1, N=6, delta=:auto, q=1.0)
            imgB = Inv.mpp_image(cache_full; resolution=16, sigma=0.1, N=6, delta=:auto, q=1.0)

            @test size(imgA.img) == (16, 16)
            @test all(isfinite, imgA.img)
            @test all(x -> x >= 0.0, imgA.img)

            # Deterministic for same inputs
            @test imgA.img == imgB.img

            # Self-distance is zero
            @test isapprox(Inv.mpp_image_distance(imgA, imgA), 0.0; atol=1e-12)

            # Symmetry
            dAB = Inv.mpp_image_distance(imgA, imgB)
            dBA = Inv.mpp_image_distance(imgB, imgA)
            @test isapprox(dAB, dBA; atol=1e-12)

            # Kernel is 1 on identical inputs
            @test isapprox(Inv.mpp_image_kernel(imgA, imgA), 1.0; atol=1e-12)
        end


    end

end


@testset "Lp generalizations of slice-based distances" begin

    # Build a small 2D PLBackend encoding and a fringe module on its poset.
    Ups = [PLB.BoxUpset([0.0, 0.0], [1.0, 1.0])]
    Downs = [PLB.BoxDownset([2.0, 2.0], [3.0, 3.0])]

    P, H, pi = PLB.encode_fringe_boxes(Ups, Downs)
    M = IR.pmodule_from_fringe(H)
    Z = TO.zero_pmodule(M.Q; field=M.field)
    opts_lp = TO.InvariantOptions()

    # Deterministic directions/offsets so the test is stable.
    dirs = Inv.default_directions(pi; n_dirs=3, max_den=3, include_axes=true, normalize=:L1)
    offs = Inv.default_offsets(pi, opts_lp; n_offsets=3)

    # -----------------------------------------------------------------------
    # Manual weighted p-mean check for sliced Wasserstein.
    # -----------------------------------------------------------------------
    Inv.clear_slice_module_cache!()
    outM = Inv.slice_barcodes(M, pi;
        directions=dirs,
        offsets=offs,
        direction_weight=:none,
        offset_weights=nothing,
        normalize_weights=true
    )
    @test isempty(SI._GLOBAL_SLICE_MODULE_CACHE)
    @test eltype(outM.barcodes) == Inv.FloatBarcode
    bcsM = outM.barcodes
    W    = outM.weights

    outZ = Inv.slice_barcodes(Z, pi;
        directions=dirs,
        offsets=offs,
        direction_weight=:none,
        offset_weights=nothing,
        normalize_weights=true
    )
    @test eltype(outZ.barcodes) == Inv.FloatBarcode
    bcsZ = outZ.barcodes

    plan_none = Inv.compile_slice_plan(
        pi,
        opts_lp;
        directions=dirs,
        offsets=offs,
        normalize_dirs=:none,
        direction_weight=:none,
        offset_weights=nothing,
        normalize_weights=true,
        threads=false,
    )
    outM_plan = SI._slice_barcodes_plan_result_uncached(M, plan_none; packed=false, threads=false)
    @test outM.weights == outM_plan.weights
    @test outM.barcodes == outM_plan.barcodes

    agg_p = 2.0
    s = 0.0
    for i in axes(bcsM, 1), j in axes(bcsM, 2)
        d = Inv.wasserstein_distance(bcsM[i, j], bcsZ[i, j]; p=2, q=1)
        s += W[i, j] * d^agg_p
    end
    manual_sw = s^(1 / agg_p)

    got_sw = Inv.sliced_wasserstein_distance(
        M, Z, pi, opts_lp;
        directions=dirs,
        offsets=offs,
        weight=:none,
        offset_weights=:none,
        normalize_weights=true,
        p=2,
        q=1,
        agg=:pmean,
        agg_p=agg_p,
        strict=true
    )
    @test isapprox(got_sw, manual_sw; atol=1e-12, rtol=1e-12)

    # -----------------------------------------------------------------------
    # Manual weighted p-mean check for sliced bottleneck.
    # -----------------------------------------------------------------------
    s2 = 0.0
    for i in axes(bcsM, 1), j in axes(bcsM, 2)
        d = Inv.bottleneck_distance(bcsM[i, j], bcsZ[i, j])
        s2 += W[i, j] * d^agg_p
    end
    manual_sb = s2^(1 / agg_p)

    got_sb = Inv.sliced_bottleneck_distance(
        M, Z, pi, opts_lp;
        directions=dirs,
        offsets=offs,
        weight=:none,
        offset_weights=:none,
        normalize_weights=true,
        agg=:pmean,
        agg_p=agg_p,
        strict=true
    )
    @test isapprox(got_sb, manual_sb; atol=1e-12, rtol=1e-12)

    # -----------------------------------------------------------------------
    # Manual matching-style max check for matching_wasserstein_distance_approx.
    # -----------------------------------------------------------------------
    outM2 = Inv.slice_barcodes(M, pi;
        directions=dirs,
        offsets=offs,
        direction_weight=:lesnick_linf,
        offset_weights=nothing,
        normalize_weights=false
    )
    bcsM2 = outM2.barcodes
    W2    = outM2.weights

    outZ2 = Inv.slice_barcodes(Z, pi;
        directions=dirs,
        offsets=offs,
        direction_weight=:lesnick_linf,
        offset_weights=nothing,
        normalize_weights=false
    )
    bcsZ2 = outZ2.barcodes

    manual_mw = 0.0
    for i in axes(bcsM2, 1), j in axes(bcsM2, 2)
        d = Inv.wasserstein_distance(bcsM2[i, j], bcsZ2[i, j]; p=2, q=1)
        val = W2[i, j] * d
        manual_mw = max(manual_mw, val)
    end

    Inv.clear_slice_module_cache!()
    got_mw = Inv.matching_wasserstein_distance_approx(
        M, Z, pi, opts_lp;
        directions=dirs,
        offsets=offs,
        weight=:lesnick_linf,
        p=2,
        q=1,
        strict=true
    )
    @test isempty(SI._GLOBAL_SLICE_MODULE_CACHE)
    @test isapprox(got_mw, manual_mw; atol=1e-12, rtol=1e-12)

    old_packed_fast = SI._SLICE_USE_PACKED_DISTANCE_FASTPATH[]
    try
        SI._SLICE_USE_PACKED_DISTANCE_FASTPATH[] = false
        got_md_off = Inv.matching_distance_approx(
            M, Z, pi, opts_lp;
            directions=dirs,
            offsets=offs,
            weight=:lesnick_linf,
            strict=true,
        )
        got_mw_off = Inv.matching_wasserstein_distance_approx(
            M, Z, pi, opts_lp;
            directions=dirs,
            offsets=offs,
            weight=:lesnick_linf,
            p=2,
            q=1,
            strict=true,
        )

        SI._SLICE_USE_PACKED_DISTANCE_FASTPATH[] = true
        got_md_on = Inv.matching_distance_approx(
            M, Z, pi, opts_lp;
            directions=dirs,
            offsets=offs,
            weight=:lesnick_linf,
            strict=true,
        )
        got_mw_on = Inv.matching_wasserstein_distance_approx(
            M, Z, pi, opts_lp;
            directions=dirs,
            offsets=offs,
            weight=:lesnick_linf,
            p=2,
            q=1,
            strict=true,
        )

        @test isapprox(got_md_on, got_md_off; atol=1e-12, rtol=1e-12)
        @test isapprox(got_mw_on, got_mw_off; atol=1e-12, rtol=1e-12)
    finally
        SI._SLICE_USE_PACKED_DISTANCE_FASTPATH[] = old_packed_fast
    end
end

# Included from test/runtests.jl; uses shared aliases (TO, QQ, PLB, ...).
#
# PLBackend is now always loaded by src/TamerOp.jl, so the historical
# force-load hack is no longer needed here.

@testset "Derived-invariant size measures: strata + Betti/Bass support" begin
    # Build the same simple 1D PLBackend encoding:
    # thresholds at 0 and 2 produce 3 regions with reps near -1, 1, 3.
    Ups = [PLB.BoxUpset([0.0])]
    Downs = [PLB.BoxDownset([2.0])]
    opts_enc = TO.EncodingOptions()
    P, Hhat, pi = PLB.encode_fringe_boxes(Ups, Downs, opts_enc)

    box = ([-1.0], [3.0])
    w = TO.RegionGeometry.region_weights(pi; box=box)
    opts = TO.InvariantOptions(box=box)

    # Identify regions robustly using locate (avoid relying on ordering).
    rL = EC.locate(pi, [-0.5])
    rM = EC.locate(pi, [1.0])
    rR = EC.locate(pi, [2.5])

    @test isapprox(w[rL], 1.0; atol=1e-9)
    @test isapprox(w[rM], 2.0; atol=1e-9)
    @test isapprox(w[rR], 1.0; atol=1e-9)

    # Value strata (e.g. where some invariant is constant)
    vals = zeros(Int, P.n)
    vals[rM] = 1
    vals[rR] = 1

    m0 = Inv.measure_by_value(vals, 0, pi, opts)
    m1 = Inv.measure_by_value(vals, 1, pi, opts)
    @test isapprox(m0, 1.0; atol=1e-9)
    @test isapprox(m1, 3.0; atol=1e-9)

    mask = falses(P.n)
    mask[rM] = true
    mask[rR] = true
    sm = Inv.support_measure(mask, pi, opts)
    @test isapprox(sm, 3.0; atol=1e-9)

    # Betti support by degree (toy numeric example)
    B = zeros(Int, 2, P.n)
    B[1, rL] = 1
    B[1, rR] = 2
    B[2, rM] = 1

    bs = Inv.betti_support_measures(B, pi, opts)
    @test isapprox(bs.support_total, 1.0 + 2.0 + 2.0; atol=1e-9)
    @test length(bs.support_by_degree) == 2

    # Same in dictionary form (multigraded)
    Bd = Dict{Tuple{Int,Int}, Int}(
        (0, rL) => 1,
        (0, rR) => 2,
        (1, rM) => 1
    )
    bs2 = Inv.betti_support_measures(Bd, pi, opts)
    @test isapprox(bs2.support_total, bs.support_total; atol=1e-9)
end

@testset "Invariants typed summaries and validators" begin
    Ups = [PLB.BoxUpset([0.0])]
    Downs = [PLB.BoxDownset([2.0])]
    P, H, pi = PLB.encode_fringe_boxes(Ups, Downs, TO.EncodingOptions())
    opts = TO.InvariantOptions(box=([-1.0], [3.0]))

    size_sum = Inv.module_size_summary(H, pi, opts)
    @test size_sum isa Inv.ModuleSizeSummary
    @test describe(size_sum).kind == :module_size_summary
    @test Inv.integrated_hilbert_mass(size_sum) == size_sum.integrated_hilbert_mass
    @test Inv.measure_by_dimension(size_sum) == size_sum.measure_by_dimension
    @test occursin("ModuleSizeSummary", repr("text/plain", size_sum))

    geom_sum = Inv.module_geometry_summary(H, pi, opts; nbins=4)
    @test geom_sum isa Inv.ModuleGeometrySummary
    @test geom_sum.size_summary isa Inv.ModuleSizeSummary
    @test describe(geom_sum).kind == :module_geometry_summary
    @test Inv.interface_measure(geom_sum) == geom_sum.interface_measure
    @test Inv.graph_stats(geom_sum) == geom_sum.graph_stats
    @test Inv.component_sizes(geom_sum) == geom_sum.graph_stats.component_sizes
    @test occursin("ModuleGeometrySummary", repr("text/plain", geom_sum))
    @test haskey(Inv.describe(geom_sum), :component_sizes)
    @test haskey(geom_sum.report, :size_summary)

    asym = Inv.module_geometry_asymptotics(ones(Int, P.n), pi, opts; scales=[1, 2], include_interface=true)
    @test asym isa Inv.ModuleGeometryAsymptoticsSummary
    @test describe(asym).kind == :module_geometry_asymptotics
    @test Inv.base_box(asym) == asym.base_box
    @test Inv.scales(asym) == [1, 2]
    @test haskey(asym, :ehrhart_total_measure)

    bbox = Inv.support_bbox([1, 0, 1], pi, opts; min_dim=1)
    @test bbox isa Inv.SupportBoundingBox
    lo, hi = bbox
    @test lo == bbox.lo
    @test hi == bbox.hi
    @test bbox[1] == bbox.lo
    @test bbox[2] == bbox.hi
    @test describe(bbox).kind == :support_bounding_box

    graph_diam = Inv.support_graph_diameter([1, 1, 0], pi, opts; min_dim=1)
    @test graph_diam isa Inv.SupportGraphDiameterSummary
    diams, overall = graph_diam
    @test diams == graph_diam.component_diameters
    @test overall == graph_diam.overall_diameter
    @test Inv.overall_graph_diameter(graph_diam) == overall
    @test Inv.component_sizes(graph_diam) == [2]

    supp_sum = Inv.support_measure_stats([0, 1, 0], pi, opts; min_dim=1)
    @test supp_sum isa Inv.SupportMeasureSummary
    @test supp_sum.support_bbox isa Inv.SupportBoundingBox
    @test describe(supp_sum).kind == :support_measure_summary
    @test Inv.support_measure(supp_sum) == supp_sum.support_measure
    @test Inv.support_fraction(supp_sum) == supp_sum.support_fraction
    @test Inv.support_bbox(supp_sum) === supp_sum.support_bbox
    @test Inv.support_bbox_diameter(supp_sum) == supp_sum.support_bbox_diameter

    B = zeros(Int, 2, P.n)
    B[1, 1] = 1
    B[2, 2] = 1
    betti_sum = Inv.betti_support_measures(B, pi, opts)
    @test betti_sum isa Inv.BettiSupportMeasuresSummary
    @test describe(betti_sum).kind == :betti_support_measures
    @test Inv.support_by_degree(betti_sum) == betti_sum.support_by_degree
    @test occursin("BettiSupportMeasuresSummary", repr("text/plain", betti_sum))

    bass_sum = Inv.bass_support_measures(B, pi, opts)
    @test bass_sum isa Inv.BassSupportMeasuresSummary
    @test describe(bass_sum).kind == :bass_support_measures
    @test Inv.mass_by_degree(bass_sum) == bass_sum.mass_by_degree
    @test occursin("BassSupportMeasuresSummary", repr("text/plain", bass_sum))

    rq_ok = Inv.check_rank_query_points(pi, [-0.5], [1.0]; opts=opts)
    @test rq_ok isa Inv.RankQueryValidationSummary
    @test rq_ok.valid

    rq_bad = Inv.check_rank_query_points(pi, [-0.5, 0.0], [1.0]; opts=opts)
    @test !rq_bad.valid
    @test_throws ArgumentError Inv.check_rank_query_points(pi, [-0.5, 0.0], [1.0]; opts=opts, throw=true)

    box_ok = Inv.check_support_box(pi; opts=opts)
    @test box_ok isa Inv.SupportGeometryValidationSummary
    @test box_ok.valid
    @test Inv.check_support_window(pi; opts=opts).valid

    box_bad = Inv.check_support_box(pi; opts=TO.InvariantOptions(box=([0.0, 1.0], [2.0])))
    @test !box_bad.valid
    @test_throws ArgumentError Inv.check_support_box(pi; opts=TO.InvariantOptions(box=([0.0, 1.0], [2.0])), throw=true)

    measure_bad = Inv.check_support_measure_query(pi; opts=opts, min_dim=1.5)
    @test !measure_bad.valid
    @test_throws ArgumentError Inv.check_support_measure_query(pi; opts=opts, min_dim=1.5, throw=true)

    @test TOA.ModuleSizeSummary === Inv.ModuleSizeSummary
    @test TOA.check_rank_query_points === Inv.check_rank_query_points
    @test TOA.RankInvariantResult === Inv.RankInvariantResult
    @test TOA.SupportComponentsSummary === Inv.SupportComponentsSummary
end

@testset "MPPI: cutoffs, pruning, serialization, and mma wrapper" begin
    IR = TO.IndicatorResolutions
    Inv = TO.Invariants

    # A tiny 2D PL backend encoding to keep the test fast.
    Ups = [PLB.BoxUpset([0.0, 0.0], [1.0, 1.0])]
    Downs = [PLB.BoxDownset([2.0, 2.0], [3.0, 3.0])]
    P, H, pi = PLB.encode_fringe_boxes(Ups, Downs)
    M = IR.pmodule_from_fringe(H)

    img_exact = Inv.mpp_image(M, pi; resolution=8, sigma=0.3, N=4)
    @test img_exact isa Inv.MPPImage
    @test length(img_exact.xgrid) == 8
    @test length(img_exact.ygrid) == 8
    @test size(img_exact.img) == (8, 8)
    @test isapprox(img_exact.sigma, 0.3; atol=0.0, rtol=0.0)

    d_decomp = TO.describe(img_exact.decomp)
    @test d_decomp.kind == :mpp_decomposition
    @test TO.mpp_decomposition_summary(img_exact.decomp) == d_decomp
    @test TO.nlines(img_exact.decomp) == length(img_exact.decomp.lines)
    @test TO.nsummands(img_exact.decomp) == length(img_exact.decomp.summands)
    @test TO.line_specs(img_exact.decomp) === img_exact.decomp.lines
    @test TO.summand_weights(img_exact.decomp) === img_exact.decomp.weights
    @test TO.total_segments(img_exact.decomp) == sum(length, img_exact.decomp.summands)
    @test TO.bounding_box(img_exact.decomp) == img_exact.decomp.box
    @test isapprox(TO.weight_sum(img_exact.decomp), sum(img_exact.decomp.weights))
    if TO.nsummands(img_exact.decomp) > 0
        @test TO.summand_segments(img_exact.decomp, 1) === img_exact.decomp.summands[1]
        @test first(TO.summand_segments(img_exact.decomp, 1)) isa TO.MultiparameterImages.MPPSegment
    end

    line1 = TO.line_specs(img_exact.decomp)[1]
    d_line = TO.describe(line1)
    @test d_line.kind == :mpp_line_spec
    @test TO.mpp_line_spec_summary(line1) == d_line
    @test TO.line_direction(line1) === line1.dir
    @test TO.line_offset(line1) == line1.off
    @test TO.line_basepoint(line1) === line1.x0
    @test TO.line_omega(line1) == line1.omega
    s_line = sprint(show, line1)
    @test occursin("MPPLineSpec(", s_line)
    s_line_plain = sprint(show, MIME("text/plain"), line1)
    @test occursin("MPPLineSpec", s_line_plain)
    @test occursin("direction", s_line_plain)
    line_report = TO.check_mpp_line_spec(line1)
    @test line_report isa TO.MPPLineSpecValidationSummary
    @test line_report.valid
    bad_line = TO.MPPLineSpec([0.0, 0.0], Inf, [1.0], 2.0)
    bad_line_report = TO.check_mpp_line_spec(bad_line)
    @test !bad_line_report.valid
    @test !isempty(bad_line_report.errors)
    @test_throws ArgumentError TO.check_mpp_line_spec(bad_line; throw=true)

    d_img = TO.describe(img_exact)
    @test d_img.kind == :mpp_image
    @test TO.mpp_image_summary(img_exact) == d_img
    @test TO.image_xgrid(img_exact) === img_exact.xgrid
    @test TO.image_ygrid(img_exact) === img_exact.ygrid
    @test TO.image_values(img_exact) === img_exact.img
    @test TO.image_shape(img_exact) == size(img_exact.img)
    @test TO.decomposition(img_exact) === img_exact.decomp

    s_decomp = sprint(show, img_exact.decomp)
    @test occursin("MPPDecomposition(", s_decomp)
    s_decomp_plain = sprint(show, MIME("text/plain"), img_exact.decomp)
    @test occursin("MPPDecomposition", s_decomp_plain)
    @test occursin("nsummands", s_decomp_plain)
    s_img = sprint(show, img_exact)
    @test occursin("MPPImage(", s_img)
    s_img_plain = sprint(show, MIME("text/plain"), img_exact)
    @test occursin("MPPImage", s_img_plain)
    @test occursin("image_shape", s_img_plain)

    decomp_report = TO.check_mpp_decomposition(img_exact.decomp)
    @test decomp_report isa TO.MPPDecompositionValidationSummary
    @test decomp_report.valid
    img_report = TO.check_mpp_image(img_exact)
    @test img_report isa TO.MPPImageValidationSummary
    @test img_report.valid

    bad_decomp = TO.MPPDecomposition(
        img_exact.decomp.lines,
        img_exact.decomp.summands[1:end-1],
        img_exact.decomp.weights,
        img_exact.decomp.box,
    )
    bad_decomp_report = TO.check_mpp_decomposition(bad_decomp)
    @test !bad_decomp_report.valid
    @test !isempty(bad_decomp_report.errors)
    @test_throws ArgumentError TO.check_mpp_decomposition(bad_decomp; throw=true)

    bad_img = TO.MPPImage(
        img_exact.xgrid,
        img_exact.ygrid,
        zeros(length(img_exact.ygrid) - 1, length(img_exact.xgrid)),
        -1.0,
        img_exact.decomp,
    )
    bad_img_report = TO.check_mpp_image(bad_img)
    @test !bad_img_report.valid
    @test !isempty(bad_img_report.errors)
    @test_throws ArgumentError TO.check_mpp_image(bad_img; throw=true)

    compat_decomp = TO.MPPDecomposition(
        img_exact.decomp.lines,
        [[([0.0, 0.0], [1.0, 1.0], 0.5)]],
        [1.0],
        ([0.0, 0.0], [1.0, 1.0]),
    )
    @test TO.check_mpp_decomposition(compat_decomp).valid
    @test first(compat_decomp.summands[1]) isa TO.MultiparameterImages.MPPSegment

    A_match = Tuple{Float64,Float64}[(0.0, 1.0), (0.2, 0.8), (0.55, 0.9)]
    B_match = Tuple{Float64,Float64}[(0.0, 1.0), (0.25, 0.75)]
    pool_match = vcat(A_match, B_match)
    generic_match = TO.MultiparameterImages._bottleneck_matching_points(A_match, B_match)
    flat_match = TO.MultiparameterImages._bottleneck_matching_points_flat(
        pool_match,
        1,
        length(A_match),
        length(A_match) + 1,
        length(B_match),
    )
    @test flat_match == generic_match
    @test TO.MultiparameterImages._bottleneck_matching_points_flat(pool_match, 1, length(A_match), length(pool_match) + 1, 0) == zeros(Int, length(A_match))

    # Re-evaluation from the stored decomposition should match.
    img2 = Inv.mpp_image(img_exact.decomp; xgrid=img_exact.xgrid, ygrid=img_exact.ygrid, sigma=img_exact.sigma)
    @test img2.img == img_exact.img

    # segment_prune toggles an exact optimization.
    img_np = Inv.mpp_image(img_exact.decomp;
                           xgrid=img_exact.xgrid,
                           ygrid=img_exact.ygrid,
                           sigma=img_exact.sigma,
                           segment_prune=false)
    @test img_np.img == img_exact.img

    # Large cutoff should be exact (no pruning).
    img_big = Inv.mpp_image(img_exact.decomp;
                            xgrid=img_exact.xgrid,
                            ygrid=img_exact.ygrid,
                            sigma=img_exact.sigma,
                            cutoff_radius=1e6)
    @test img_big.img == img_exact.img

    # cutoff_tol is an optional approximation; it can only decrease the image (nonnegative kernel).
    img_cut = Inv.mpp_image(img_exact.decomp;
                            xgrid=img_exact.xgrid,
                            ygrid=img_exact.ygrid,
                            sigma=img_exact.sigma,
                            cutoff_tol=1e-6)
    @test all(img_cut.img .<= img_exact.img .+ 1e-12)

    # Cutoff arguments are mutually exclusive.
    @test_throws ArgumentError Inv.mpp_image(img_exact.decomp;
                                            xgrid=img_exact.xgrid,
                                            ygrid=img_exact.ygrid,
                                            sigma=img_exact.sigma,
                                            cutoff_radius=1.0,
                                            cutoff_tol=1e-3)
    @test_throws ArgumentError Inv.mpp_image(img_exact.decomp;
                                            xgrid=img_exact.xgrid,
                                            ygrid=img_exact.ygrid,
                                            sigma=img_exact.sigma,
                                            cutoff_tol=1.0)

    # Inner product should require identical grids (not just the same lengths).
    img_shift = Inv.mpp_image(img_exact.decomp;
                              xgrid=img_exact.xgrid .+ 0.1,
                              ygrid=img_exact.ygrid,
                              sigma=img_exact.sigma)
    @test_throws ErrorException Inv.mpp_image_inner_product(img_exact, img_shift)

    # JSON round-trip: decomposition.
    mktemp() do path, io
        close(io)
        Inv.save_mpp_decomposition_json(path, img_exact.decomp)
        info = SER.inspect_json(path)
        @test info isa SER.JSONArtifactSummary
        @test SER.artifact_kind(info) == "MPPDecomposition"
        @test SER.artifact_profile_hint(info) == :compact
        @test SER.artifact_size_bytes(info) isa Integer
        @test SER.mpp_decomposition_json_summary(path).kind == "MPPDecomposition"
        @test SER.check_mpp_decomposition_json(path).valid
        decomp_rt = Inv.load_mpp_decomposition_json(path)
        decomp_trusted = Inv.load_mpp_decomposition_json(path; validation=:trusted)
        img_rt = Inv.mpp_image(decomp_rt; xgrid=img_exact.xgrid, ygrid=img_exact.ygrid, sigma=img_exact.sigma)
        @test isapprox(img_rt.img, img_exact.img; atol=1e-12, rtol=0.0)
        @test decomp_trusted.lines == decomp_rt.lines
        @test decomp_trusted.weights == decomp_rt.weights

        debug_path = path * ".debug"
        Inv.save_mpp_decomposition_json(debug_path, img_exact.decomp; profile=:debug)
        @test filesize(path) < filesize(debug_path)
        @test SER.inspect_json(debug_path).profile_hint == :debug
    end

    # JSON round-trip: full image.
    mktemp() do path, io
        close(io)
        Inv.save_mpp_image_json(path, img_exact)
        info = SER.inspect_json(path)
        @test info isa SER.JSONArtifactSummary
        @test SER.artifact_kind(info) == "MPPImage"
        @test SER.artifact_profile_hint(info) == :compact
        @test SER.mpp_image_json_summary(path).kind == "MPPImage"
        @test SER.check_mpp_image_json(path).valid
        img_rt = Inv.load_mpp_image_json(path)
        img_trusted = Inv.load_mpp_image_json(path; validation=:trusted)
        @test isapprox(img_rt.xgrid, img_exact.xgrid; atol=1e-12, rtol=0.0)
        @test isapprox(img_rt.ygrid, img_exact.ygrid; atol=1e-12, rtol=0.0)
        @test isapprox(img_rt.sigma, img_exact.sigma; atol=0.0, rtol=0.0)
        @test isapprox(img_rt.img, img_exact.img; atol=1e-12, rtol=0.0)
        @test isapprox(img_trusted.img, img_rt.img; atol=1e-12, rtol=0.0)

        debug_path = path * ".debug"
        Inv.save_mpp_image_json(debug_path, img_exact; profile=:debug)
        @test filesize(path) < filesize(debug_path)
        @test SER.inspect_json(debug_path).profile_hint == :debug
    end

    mktemp() do path, io
        close(io)
        Inv.save_mpp_image_json(path, img_exact)
        obj = JSON3.read(read(path, String))
        bad = Dict{String,Any}()
        for (k, v) in pairs(obj)
            bad[String(k)] = v
        end
        pop!(bad, "decomp", nothing)
        write(path, JSON3.write(bad))
        report = SER.check_mpp_image_json(path)
        @test !report.valid
        @test !isempty(report.issues)
        @test_throws ArgumentError SER.check_mpp_image_json(path; throw=true)
    end

    @test TOA.check_mpp_decomposition_json === SER.check_mpp_decomposition_json
    @test TOA.check_mpp_image_json === SER.check_mpp_image_json
    @test TOA.mpp_decomposition_json_summary === SER.mpp_decomposition_json_summary
    @test TOA.mpp_image_json_summary === SER.mpp_image_json_summary
    @test TamerOp.mpp_decomposition_summary === TamerOp.MultiparameterImages.mpp_decomposition_summary
    @test TamerOp.mpp_image_summary === TamerOp.MultiparameterImages.mpp_image_summary
    @test TamerOp.mp_landscape_summary === TamerOp.MultiparameterImages.mp_landscape_summary
    @test TOA.mpp_decomposition_summary === TamerOp.MultiparameterImages.mpp_decomposition_summary
    @test TOA.mpp_image_summary === TamerOp.MultiparameterImages.mpp_image_summary
    @test TOA.mpp_line_spec_summary === TamerOp.MultiparameterImages.mpp_line_spec_summary
    @test TOA.check_mpp_line_spec === TamerOp.MultiparameterImages.check_mpp_line_spec
    @test TOA.check_mpp_decomposition === TamerOp.MultiparameterImages.check_mpp_decomposition
    @test TOA.check_mpp_image === TamerOp.MultiparameterImages.check_mpp_image
    @test TOA.line_direction === TamerOp.MultiparameterImages.line_direction
    @test TOA.line_offset === TamerOp.MultiparameterImages.line_offset
    @test TOA.line_basepoint === TamerOp.MultiparameterImages.line_basepoint
    @test TOA.line_omega === TamerOp.MultiparameterImages.line_omega
    @test TOA.nlines === TamerOp.MultiparameterImages.nlines
    @test TOA.nsummands === TamerOp.MultiparameterImages.nsummands
    @test TOA.decomposition === TamerOp.MultiparameterImages.decomposition

    # mma_decomposition wrapper.
    out = SM.mma_decomposition(M, pi; method=:mpp_image, mpp_kwargs=(resolution=6, N=3, sigma=0.25))
    @test out isa SM.SignedMeasureDecomposition
    @test SM.has_mpp_image(out)
    @test !SM.has_rectangles(out)
    @test !SM.has_slices(out)
    @test !SM.has_euler(out)
    @test SM.has_image(out)
    @test SM.component_names(out) == (:mpp_image,)
    @test SM.ncomponents(out) == 1
    img = SM.mpp_image(out)
    @test img isa Inv.MPPImage
    @test length(img.xgrid) == 6
    @test length(img.ygrid) == 6
    out_report = SM.check_signed_measure_decomposition(out; throw=false)
    @test out_report.valid
    @test TO.describe(out).kind == :signed_measure_decomposition
end

@testset "Projected distances and differentiable feature maps" begin
    Inv = TO.Invariants
    IR = TO.IndicatorResolutions

    # Simple chain module tests
    P = chain_poset(3)
    M23 = IR.pmodule_from_fringe(one_by_one_fringe(P, FF.principal_upset(P, 2), FF.principal_downset(P, 3); field=field))
    M3  = IR.pmodule_from_fringe(one_by_one_fringe(P, FF.principal_upset(P, 3), FF.principal_downset(P, 3); field=field))

    arr = Inv.projected_arrangement(P, [0.0, 1.0, 2.0])
    c23 = Inv.projected_barcode_cache(M23, arr)
    c3  = Inv.projected_barcode_cache(M3, arr)

    b23_res = Inv.projected_barcodes(c23)
    b3_res  = Inv.projected_barcodes(c3)
    b23 = Inv.projected_barcodes(b23_res)[1]
    b3  = Inv.projected_barcodes(b3_res)[1]
    @test !isnothing(c23.packed_barcodes[1])
    @test !isnothing(c3.packed_barcodes[1])

    @test b23 == Dict((1.0,3.0)=>1)
    @test b3  == Dict((2.0,3.0)=>1)

    d = Inv.projected_distance(c23, c3; dist=:bottleneck, agg=:mean)
    @test isapprox(d, 1.0; atol=1e-12)

    ksame = Inv.projected_kernel(c23, c23; kind=:wasserstein_gaussian, sigma=1.0, agg=:mean)
    @test isapprox(ksame, 1.0; atol=1e-12)

    # Allocation regression (warm, then measure steady-state).
    Inv.projected_distance(c23, c3; dist=:bottleneck, agg=:mean, threads=false)
    alloc_proj_dist = @allocated Inv.projected_distance(c23, c3; dist=:bottleneck, agg=:mean, threads=false)
    @test alloc_proj_dist < 2_500_000

    Inv.projected_kernel(c23, c23; kind=:wasserstein_gaussian, sigma=1.0, agg=:mean, threads=false)
    alloc_proj_kernel = @allocated Inv.projected_kernel(c23, c23; kind=:wasserstein_gaussian, sigma=1.0, agg=:mean, threads=false)
    @test alloc_proj_kernel < 2_500_000

    t_proj_dist = _median_elapsed(; warmup=1, reps=5) do
        Inv.projected_distance(c23, c3; dist=:bottleneck, agg=:mean, threads=false)
    end
    @test t_proj_dist < 0.75

    t_proj_kernel = _median_elapsed(; warmup=1, reps=5) do
        Inv.projected_kernel(c23, c23; kind=:wasserstein_gaussian, sigma=1.0, agg=:mean, threads=false)
    end
    @test t_proj_kernel < 0.75

    if Threads.nthreads() > 1
        # Threading parity: projected_* family
        c23_s = Inv.projected_barcode_cache(M23, arr)
        c3_s = Inv.projected_barcode_cache(M3, arr)
        c23_t = Inv.projected_barcode_cache(M23, arr)
        c3_t = Inv.projected_barcode_cache(M3, arr)

        b23_s = Inv.projected_barcodes(Inv.projected_barcodes(c23_s; threads=false))[1]
        b23_t = Inv.projected_barcodes(Inv.projected_barcodes(c23_t; threads=true))[1]
        @test b23_s == b23_t

        d_s = Inv.projected_distance(c23_s, c3_s; dist=:bottleneck, agg=:mean, threads=false)
        d_t = Inv.projected_distance(c23_t, c3_t; dist=:bottleneck, agg=:mean, threads=true)
        @test isapprox(d_s, d_t; atol=1e-12, rtol=0.0)

        k_s = Inv.projected_kernel(c23_s, c23_s; kind=:wasserstein_gaussian, sigma=1.0, agg=:mean, threads=false)
        k_t = Inv.projected_kernel(c23_t, c23_t; kind=:wasserstein_gaussian, sigma=1.0, agg=:mean, threads=true)
        @test isapprox(k_s, k_t; atol=1e-12, rtol=0.0)
    end

    # Differentiable persistence image agrees with fast version
    bar = Dict((0.0,1.0)=>1, (0.5,1.5)=>1)
    xg = 0.0:0.25:2.0
    yg = 0.0:0.25:2.0

    pi_fast = Inv.persistence_image(bar; xgrid=xg, ygrid=yg, sigma=0.3,
                                   coords=:birth_death, weighting=:none, normalize=:none)

    pi_diff = Inv.persistence_image(bar; xgrid=xg, ygrid=yg, sigma=0.3,
                                   coords=:birth_death, weighting=:none, normalize=:none,
                                   differentiable=true)

    @test isapprox(pi_fast.values, pi_diff.values)

    fv = Inv.feature_vector(bar; kind=:persistence_image, xgrid=xg, ygrid=yg, sigma=0.3,
                            coords=:birth_death, weighting=:none, normalize=:none,
                            differentiable=true)

    @test length(fv) == length(xg)*length(yg)
    @test isapprox(reshape(fv, length(yg), length(xg)), pi_diff.values)

    if Threads.nthreads() > 1
        # Threading parity: persistence image
        pi_serial = Inv.persistence_image(bar; xgrid=xg, ygrid=yg, sigma=0.3,
                                          coords=:birth_death, weighting=:none, normalize=:none,
                                          threads=false)
        pi_thread = Inv.persistence_image(bar; xgrid=xg, ygrid=yg, sigma=0.3,
                                          coords=:birth_death, weighting=:none, normalize=:none,
                                          threads=true)
        @test isapprox(pi_thread.values, pi_serial.values; atol=1e-12, rtol=0.0)
    end
end

@testset "PD distance backends" begin
    Inv = TO.Invariants
    Random.seed!(1234)

    function rand_barcode(n::Int)
        pts = Dict{Tuple{Float64,Float64},Int}()
        for _ in 1:n
            b = rand()
            d = b + rand()
            key = (b, d)
            pts[key] = get(pts, key, 0) + 1
        end
        return pts
    end

    for n1 in 0:5, n2 in 0:5
        b1 = rand_barcode(n1)
        b2 = rand_barcode(n2)
        d_h = Inv.wasserstein_distance(b1, b2; p=2, q=2, backend=:hungarian)
        d_a = Inv.wasserstein_distance(b1, b2; p=2, q=2, backend=:auction)
        @test isapprox(d_h, d_a; atol=0.3, rtol=0.3)
    end

    b1 = rand_barcode(5)
    b2 = rand_barcode(4)
    d_hk = Inv.bottleneck_distance(b1, b2; backend=:hk)
    d_auto = Inv.bottleneck_distance(b1, b2; backend=:auto)
    @test d_hk == d_auto
end

@testset "fibered slice family + typed caches" begin
    Inv = TO.Invariants
    # Build a simple 2D PLBackend encoding with three vertical stripes (chain of length 3).
    Ups = [PLB.BoxUpset([0.0, -10.0]), PLB.BoxUpset([1.0, -10.0])]
    Downs = PLB.BoxDownset[]
    opts_enc = TO.EncodingOptions()
    P, H, pi = PLB.encode_fringe_boxes(Ups, Downs, opts_enc)

    r2 = EC.locate(pi, [0.5, 0.0])
    r3 = EC.locate(pi, [2.0, 0.0])

    M23 = IR.pmodule_from_fringe(one_by_one_fringe(P, FF.principal_upset(P, r2), FF.principal_downset(P, r3); field=field))
    M3  = IR.pmodule_from_fringe(one_by_one_fringe(P, FF.principal_upset(P, r3), FF.principal_downset(P, r3); field=field))

    box = ([-1.0, -1.0], [2.0, 1.0])
    opts = TO.InvariantOptions(box=box)
    arr_shared = Inv.fibered_arrangement_2d(pi, opts; normalize_dirs=:L1, precompute=:cells)

    fam = Inv.fibered_slice_family_2d(arr_shared; direction_weight=:lesnick_l1, store_values=true)
    @test fam.arrangement === arr_shared
    @test Inv.nslices(fam) > 0

    cache23 = Inv.fibered_barcode_cache_2d(M23, arr_shared; precompute=:full)
    cache3  = Inv.fibered_barcode_cache_2d(M3,  arr_shared; precompute=:full)

    # typed cache: packed barcodes are the internal representation.
    @test eltype(cache23.index_barcodes_packed) <: Union{Nothing,Inv.PackedIndexBarcode}

    # exact distance matches slice-list backend
    d_slices = Inv.matching_distance_exact_2d(M23, M3, pi, opts; weight=:lesnick_l1, normalize_dirs=:L1)
    d_cache  = Inv.matching_distance_exact_2d(cache23, cache3; weight=:lesnick_l1, family=fam, threads=false)
    @test isapprox(d_cache, d_slices; atol=1e-12)

    # thread determinism
    d_thr = Inv.matching_distance_exact_2d(cache23, cache3; weight=:lesnick_l1, family=fam, threads=true)
    @test isapprox(d_thr, d_cache; atol=1e-12)

    # Allocation regression for fibered exact matching + slice extraction.
    Inv.slice_barcodes(cache23; dirs=[[1.0, 1.0]], offsets=[0.0], values=:t, threads=false)
    alloc_slice = @allocated Inv.slice_barcodes(cache23; dirs=[[1.0, 1.0]], offsets=[0.0], values=:t, threads=false)
    @test alloc_slice < 2_000_000

    Inv.matching_distance_exact_2d(cache23, cache3; weight=:lesnick_l1, family=fam, threads=false)
    alloc_exact = @allocated Inv.matching_distance_exact_2d(cache23, cache3; weight=:lesnick_l1, family=fam, threads=false)
    @test alloc_exact < 2_500_000

    t_exact = _median_elapsed(; warmup=1, reps=5) do
        Inv.matching_distance_exact_2d(cache23, cache3; weight=:lesnick_l1, family=fam, threads=false)
    end
    @test t_exact < 1.0

    # kernel matches slice-list backend (uniform cell weighting)
    fam2 = Inv.matching_distance_exact_slices_2d(pi, opts; normalize_dirs=:L1)
    slices2 = fam2.slices
    k_slices = Inv.slice_kernel(M23, M3, slices2; kind=:bottleneck_gaussian, sigma=1.0, normalize_weights=true)
    k_cache  = Inv.slice_kernel(cache23, cache3; kind=:bottleneck_gaussian, sigma=1.0,
                               direction_weight=:lesnick_l1, cell_weight=:uniform,
                               family=fam, normalize_weights=true, threads=false)
    @test isapprox(k_cache, k_slices; atol=1e-12)

    axes = ([1, 2, 3], [1, 2, 3])
    opts_bulk = TO.InvariantOptions(axes=axes, axes_policy=:as_given)
    if pi isa TamerOp.ZnEncoding.ZnEncodingMap
        rq_cache_rect = IC.RankQueryCache(pi)
        FF.build_cache!(M23.Q; cover=true, updown=false)
        cc_rect = FF._get_cover_cache(M23.Q)
        reg_rect = SM._rectangle_region_grid(pi, axes, rq_cache_rect; strict=false)
        reg_comp_rect, keep1_rect, keep2_rect, axes_comp_rect =
            SM._compress_region_grid_2d(reg_rect, axes)

        old_comp_cap = SM._RECTANGLE_BULK_AUTO_MAX_COMPRESSED_COMPARABLE_PAIRS[]
        try
            rq_cache_gate = IC.RankQueryCache(pi)
            SM._RECTANGLE_BULK_AUTO_MAX_COMPRESSED_COMPARABLE_PAIRS[] = typemax(Int)
            @test SM._choose_rectangle_signed_barcode_method_module(
                M23, pi, axes, rq_cache_gate, nothing;
                strict=false,
                method=:auto,
                max_span=nothing,
                bulk_max_elems=typemax(Int),
            ) == :bulk

            rq_cache_gate = IC.RankQueryCache(pi)
            SM._RECTANGLE_BULK_AUTO_MAX_COMPRESSED_COMPARABLE_PAIRS[] = 0
            @test SM._choose_rectangle_signed_barcode_method_module(
                M23, pi, axes, rq_cache_gate, nothing;
                strict=false,
                method=:auto,
                max_span=nothing,
                bulk_max_elems=typemax(Int),
            ) == :local
        finally
            SM._RECTANGLE_BULK_AUTO_MAX_COMPRESSED_COMPARABLE_PAIRS[] = old_comp_cap
        end

        sc_tensor = CM.SessionCache()
        packed_tensor_1 = SM._cached_rectangle_packed_tensor_2d(
            M23, pi, axes, reg_comp_rect, axes_comp_rect, rq_cache_rect, sc_tensor, cc_rect;
            strict=false,
            threads=false,
        )
        packed_tensor_2 = SM._cached_rectangle_packed_tensor_2d(
            M23, pi, axes, reg_comp_rect, axes_comp_rect, rq_cache_rect, sc_tensor, cc_rect;
            strict=false,
            threads=false,
        )
        @test packed_tensor_1 === packed_tensor_2

        sb_tensor_bulk = SM.rectangle_signed_barcode(M23, pi, opts_bulk; cache=sc_tensor, method=:bulk)
        sb_tensor_span = SM.rectangle_signed_barcode(M23, pi, opts_bulk; cache=sc_tensor, method=:bulk, max_span=(0, 0))
        packed_tensor_3 = SM._cached_rectangle_packed_tensor_2d(
            M23, pi, axes, reg_comp_rect, axes_comp_rect, rq_cache_rect, sc_tensor, cc_rect;
            strict=false,
            threads=false,
        )
        @test packed_tensor_3 === packed_tensor_1
        @test sb_tensor_bulk !== sb_tensor_span
        @test first(sb_tensor_span.axes) == first(sb_tensor_bulk.axes)

        if Threads.nthreads() > 1
            # Threading parity: rectangle signed barcode (bulk path)
            sb_serial = SM.rectangle_signed_barcode(M23, pi, opts_bulk; method=:bulk, threads=false)
            sb_thread = SM.rectangle_signed_barcode(M23, pi, opts_bulk; method=:bulk, threads=true)
            @test sb_thread.rects == sb_serial.rects
            @test sb_thread.weights == sb_serial.weights

            # Threading parity: rectangle signed barcode rank reconstruction
            rk_serial = SM.rectangle_signed_barcode_rank(sb_serial; threads=false)
            rk_thread = SM.rectangle_signed_barcode_rank(sb_serial; threads=true)
            @test rk_thread == rk_serial

            # Threading parity: rectangle signed barcode image
            img_serial = SM.rectangle_signed_barcode_image(sb_serial; threads=false)
            img_thread = SM.rectangle_signed_barcode_image(sb_serial; threads=true)
            @test isapprox(img_thread, img_serial; atol=1e-12, rtol=0.0)
            img_cut_serial = SM.rectangle_signed_barcode_image(sb_serial; cutoff_tol=1.0e-6, threads=false)
            img_cut_thread = SM.rectangle_signed_barcode_image(sb_serial; cutoff_tol=1.0e-6, threads=true)
            @test isapprox(img_cut_thread, img_cut_serial; atol=1e-12, rtol=0.0)

            # Threading parity: MPPI image (same cache, same grids)
            img_serial2 = Inv.mpp_image(cache23; resolution=6, N=3, sigma=0.25, threads=false)
            img_thread2 = Inv.mpp_image(cache23; resolution=6, N=3, sigma=0.25, threads=true)
            @test isapprox(img_thread2.img, img_serial2.img; atol=1e-12, rtol=0.0)
        end
    else
        @test pi isa PLB.PLEncodingMapBoxes
    end
end

@testset "Exported invariant utility APIs: direct correctness coverage" begin
    # ------------------------------------------------------------------
    # Region-label / support utilities
    # ------------------------------------------------------------------
    pi1 = ToyPi1DThresholds()
    opts1 = TO.InvariantOptions(box=([0.0], [3.0]), strict=true)

    vals_rep = Inv.region_values(pi1, x -> (x[1] < 2.0 ? :left : :right); arg=:rep)
    @test vals_rep == [:left, :left, :right]

    pi_idx = ToyBoxes2D((Float64[0.0, 1.0], Float64[0.0, 1.0]), [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)])
    vals_idx = Inv.region_values(pi_idx, r -> r * r; arg=:index)
    @test vals_idx == [1, 4, 9]

    vals_both = Inv.region_values(pi1, (r, x) -> r + Int(floor(x[1])); arg=:both)
    @test vals_both == [1, 3, 5]

    w = [2.0, 3.0, 5.0]
    @test Inv.support_measure(Bool[true, false, true], pi1, opts1; weights=w) == 7.0
    @test Inv.vertex_set_measure([1, 3, 3], pi1, opts1; weights=w) == 7.0
    @test_throws ErrorException Inv.vertex_set_measure([4], pi1, opts1; weights=w)

    # ------------------------------------------------------------------
    # Axis coarsening / restriction helpers
    # ------------------------------------------------------------------
    axes2 = ([1, 2, 3, 4, 5], [10, 20, 30, 40, 50])
    coarse = Inv.coarsen_axes(axes2; max_len=3)
    @test length(coarse[1]) <= 3
    @test length(coarse[2]) <= 3
    @test first(coarse[1]) == 1 && last(coarse[1]) == 5
    @test first(coarse[2]) == 10 && last(coarse[2]) == 50

    axes_restricted = Inv.restrict_axes_to_encoding(([0, 2, 3],), pi1)
    @test axes_restricted == ([0.0, 2.0, 3.0],)

    # Specialized ZnEncodingMap axis restriction semantics.
    I = FZ.Face(1, Int[])
    flats = [FZ.IndFlat(I, [1]; id=:F)]
    injectives = [FZ.IndInj(I, [3]; id=:E)]
    FG = FZ.Flange{K}(1, flats, injectives, reshape([cf(1)], 1, 1))
    _, _, zpi = TamerOp.ZnEncoding.encode_from_flange(FG, TO.EncodingOptions(backend=:zn, max_regions=1000))

    zax_keep = Inv.restrict_axes_to_encoding(([0, 2, 4],), zpi; keep_endpoints=true)
    zax_drop = Inv.restrict_axes_to_encoding(([0, 2, 4],), zpi; keep_endpoints=false)
    @test 0 in zax_keep[1] && 4 in zax_keep[1]
    @test issubset(Set(zax_drop[1]), Set(zax_keep[1]))
    @test all(x -> x in EC.axes_from_encoding(zpi)[1], zax_drop[1])

    # ------------------------------------------------------------------
    # Graph primitives
    # ------------------------------------------------------------------
    adj = Dict{Tuple{Int,Int},Float64}((1,2)=>2.0, (2,3)=>1.0)
    gd = Inv.graph_degrees(adj, 4)
    @test gd.degrees == [1, 2, 1, 0]
    @test gd.weighted_degrees == [2.0, 3.0, 1.0, 0.0]

    comps = Inv.graph_connected_components(adj, 4)
    comp_sets = Set(Set(c) for c in comps)
    @test comp_sets == Set([Set([1,2,3]), Set([4])])

    modq = Inv.graph_modularity([1, 1, 2, 3], adj; nregions=4)
    @test isapprox(modq, -1.0/18.0; atol=1e-12)
    @test Inv.graph_modularity([1, 1], Dict{Tuple{Int,Int},Float64}(); nregions=2) == 0.0

    # ------------------------------------------------------------------
    # Point signed measure helpers
    # ------------------------------------------------------------------
    pm = SM.PointSignedMeasure((Float64[0.0, 1.0, 2.0],), [(1,), (2,), (3,)], [1, -5, 2])
    @test axes(pm) == (Float64[0.0, 1.0, 2.0],)
    @test SM.support_indices(pm) == [(1,), (2,), (3,)]
    @test SM.weights(pm) == [1, -5, 2]
    @test SM.points(pm) == [(0.0,), (1.0,), (2.0,)]
    @test SM.ambient_dimension(pm) == 1
    @test SM.nterms(pm) == 3
    @test SM.positive_terms(pm) == 2
    @test SM.negative_terms(pm) == 1
    @test SM.total_mass(pm) == -2
    @test SM.total_variation(pm) == 8
    @test SM.axis_lengths(pm) == (3,)
    @test SM.weight_range(pm) == (-5, 2)
    @test SM.max_abs_weight(pm) == 5
    @test SM.support_size(pm) == 3
    @test SM.term(pm, 2).point == (1.0,)
    @test SM.coefficient(pm, (1.0,)) == -5
    @test SM.support_bbox(pm) == (lo = (0.0,), hi = (2.0,))
    @test length(SM.largest_terms(pm; n=2)) == 2
    @test SM.describe(pm).kind == :point_signed_measure
    @test SM.signed_measure_summary(pm) == SM.describe(pm)
    @test SM.point_signed_measure_summary(pm) == SM.describe(pm)
    @test TO.describe(pm).kind == :point_signed_measure
    @test occursin("PointSignedMeasure", sprint(show, pm))
    pm_report = SM.check_point_signed_measure(pm; throw=false)
    @test pm_report isa SM.PointSignedMeasureValidationSummary
    @test pm_report.valid
    @test occursin("PointSignedMeasureValidationSummary", sprint(show, pm_report))
    @test SM.validate(pm; throw=false).valid
    pm_trunc = SM.truncate_point_signed_measure(pm; max_terms=2, min_abs_weight=2)
    @test pm_trunc.inds == [(2,), (3,)]
    @test pm_trunc.wts == [-5, 2]

    pm_bad = SM.PointSignedMeasure((Float64[0.0, 1.0],), [(3,)], [1])
    pm_bad_report = SM.validate(pm_bad; throw=false)
    @test !pm_bad_report.valid
    @test !isempty(pm_bad_report.errors)
    @test_throws ArgumentError SM.validate(pm_bad)

    pm1 = SM.PointSignedMeasure((Float64[0.0],), [(1,)], [2])
    pm2 = SM.PointSignedMeasure((Float64[0.0],), [(1,)], [3])
    @test SM.point_signed_measure_kernel(pm1, pm2; sigma=1.0, kind=:gaussian) == 6.0
    @test SM.point_signed_measure_kernel(pm1, pm2; sigma=1.0, kind=:laplacian) == 6.0

    # ------------------------------------------------------------------
    # Euler alias + Bass support summaries
    # ------------------------------------------------------------------
    Pch = chain_poset(3)
    H23 = one_by_one_fringe(Pch, FF.principal_upset(Pch, 2), FF.principal_downset(Pch, 3), cf(1); field=field)
    M23 = IR.pmodule_from_fringe(H23)
    opts_axes = TO.InvariantOptions(axes=([1, 2, 3],), axes_policy=:as_given)

    @test SM.euler_characteristic_surface(M23, pi1, opts_axes) == TO.euler_surface(M23, pi1, opts_axes)

    Bdict = Dict((0, 1) => 1, (1, 3) => 2)
    bs = Inv.bass_support_measures(Bdict, pi1, opts1; weights=w)
    @test bs.support_by_degree == [2.0, 5.0]
    @test bs.mass_by_degree == [2.0, 10.0]
    @test bs.support_union == 7.0

    Bmat = [1 0 0; 0 0 2]
    bs_mat = Inv.bass_support_measures(Bmat, pi1, opts1; weights=w)
    @test bs_mat == bs

    # ------------------------------------------------------------------
    # Fibered low-level accessors + projected distances
    # ------------------------------------------------------------------
    Ups = [PLB.BoxUpset([0.0, -10.0]), PLB.BoxUpset([1.0, -10.0])]
    Downs = PLB.BoxDownset[]
    P2, _, pi2 = PLB.encode_fringe_boxes(Ups, Downs, TO.EncodingOptions())
    r2 = EC.locate(pi2, [0.5, 0.0])
    r3 = EC.locate(pi2, [2.0, 0.0])

    M2 = IR.pmodule_from_fringe(one_by_one_fringe(P2, FF.principal_upset(P2, r2), FF.principal_downset(P2, r3); field=field))
    M3 = IR.pmodule_from_fringe(one_by_one_fringe(P2, FF.principal_upset(P2, r3), FF.principal_downset(P2, r3); field=field))

    opts2 = TO.InvariantOptions(box=([-1.0, -1.0], [2.0, 1.0]))
    arr2 = Inv.fibered_arrangement_2d(pi2, opts2; normalize_dirs=:L1, precompute=:cells)

    cid = Inv.fibered_cell_id(arr2, (1.0, 1.0), 0.0)
    @test cid !== nothing
    cid2 = Inv.fibered_cell_id(arr2, (1.0, 1.0), [0.0, 0.0])
    @test cid2 !== nothing

    ch = Inv.fibered_chain(arr2, (1.0, 1.0), 0.0; copy=false)
    vals = Inv.fibered_values(arr2, (1.0, 1.0), 0.0)
    @test length(vals) == length(ch) + 1

    fbc = Inv.fibered_barcode_cache_2d(M2, arr2; precompute=:none)
    fs = Inv.fibered_slice(fbc, (1.0, 1.0), 0.0)
    @test fs isa Inv.FiberedSliceResult
    @test Inv.slice_chain(fs) == collect(ch)
    @test length(Inv.slice_values(fs)) == length(Inv.slice_chain(fs)) + 1
    @test Inv.slice_barcode(fs) == Inv.fibered_barcode(fbc, (1.0, 1.0), 0.0)

    parr = Inv.projected_arrangement(pi2; dirs=[(1.0, 0.0), (0.0, 1.0)], threads=false)
    pc2 = Inv.projected_barcode_cache(M2, parr; precompute=true)
    pc3 = Inv.projected_barcode_cache(M3, parr; precompute=true)
    dvec = Inv.projected_distances(pc2, pc3; dist=:bottleneck, threads=false)
    @test dvec isa Inv.ProjectedDistancesResult
    @test length(Inv.projected_distances(dvec)) == 2
    dmean = Inv.projected_distance(pc2, pc3; dist=:bottleneck, agg=:mean, threads=false)
    @test isapprox(dmean, sum(Inv.projected_distances(dvec)) / length(Inv.projected_distances(dvec)); atol=1e-12)

    # ------------------------------------------------------------------
    # Direct exported sliced kernel + MPP decomposition APIs
    # ------------------------------------------------------------------
    kself = Inv.sliced_wasserstein_kernel(
        M2, M2, pi2, opts2;
        n_dirs=8, n_offsets=5, sigma=1.0,
        normalize_weights=true, normalize_dirs=:L1,
        threads=false
    )
    kcross = Inv.sliced_wasserstein_kernel(
        M2, M3, pi2, opts2;
        n_dirs=8, n_offsets=5, sigma=1.0,
        normalize_weights=true, normalize_dirs=:L1,
        threads=false
    )
    kcross_sym = Inv.sliced_wasserstein_kernel(
        M3, M2, pi2, opts2;
        n_dirs=8, n_offsets=5, sigma=1.0,
        normalize_weights=true, normalize_dirs=:L1,
        threads=false
    )
    @test isapprox(kself, 1.0; atol=1e-12)
    @test 0.0 <= kcross <= 1.0
    @test isapprox(kcross, kcross_sym; atol=1e-12)

    decomp_from_wrapper = Inv.mpp_decomposition(M2, pi2, opts2)
    cpi2 = EC.compile_encoding(P2, pi2)
    decomp_from_compiled = Inv.mpp_decomposition(M2, cpi2, opts2)
    decomp_root_from_wrapper = TO.mpp_decomposition(M2, pi2; opts=opts2)
    decomp_root_from_compiled = TO.mpp_decomposition(M2, cpi2; opts=opts2)
    arr_mpp = Inv.fibered_arrangement_2d(pi2, opts2; include_axes=true)
    cache_mpp = Inv.fibered_barcode_cache_2d(M2, arr_mpp)
    decomp_from_cache = Inv.mpp_decomposition(cache_mpp)
    @test length(decomp_from_wrapper.summands) == length(decomp_from_cache.summands)
    @test length(decomp_from_wrapper.weights) == length(decomp_from_cache.weights)
    @test isapprox(sum(decomp_from_wrapper.weights), sum(decomp_from_cache.weights); atol=1e-12)
    @test decomp_from_wrapper.box == decomp_from_cache.box
    @test length(decomp_from_compiled.summands) == length(decomp_from_wrapper.summands)
    @test decomp_from_compiled.lines == decomp_from_wrapper.lines
    @test isapprox(sum(decomp_from_compiled.weights), sum(decomp_from_wrapper.weights); atol=1e-12)
    @test decomp_root_from_wrapper.lines == decomp_from_wrapper.lines
    @test decomp_root_from_compiled.lines == decomp_from_wrapper.lines

    img_from_wrapper = Inv.mpp_image(M2, pi2, opts2; resolution=8, sigma=0.3, N=4)
    img_from_compiled = Inv.mpp_image(M2, cpi2, opts2; resolution=8, sigma=0.3, N=4)
    img_root_from_wrapper = TO.mpp_image(M2, pi2; opts=opts2, resolution=8, sigma=0.3, N=4)
    img_root_from_compiled = TO.mpp_image(M2, cpi2; opts=opts2, resolution=8, sigma=0.3, N=4)
    land_root_from_wrapper = TO.mp_landscape(
        M2,
        pi2;
        opts=opts2,
        directions=[(1.0, 1.0)],
        offsets=[(0.0, 0.0)],
        tgrid=collect(range(0.0, stop=1.5, length=16)),
        kmax=2,
    )
    land_root_from_compiled = TO.mp_landscape(
        M2,
        cpi2;
        opts=opts2,
        directions=[(1.0, 1.0)],
        offsets=[(0.0, 0.0)],
        tgrid=collect(range(0.0, stop=1.5, length=16)),
        kmax=2,
    )
    @test TOA.image_values(img_from_compiled) == TOA.image_values(img_from_wrapper)
    @test img_from_compiled.xgrid == img_from_wrapper.xgrid
    @test img_from_compiled.ygrid == img_from_wrapper.ygrid
    @test TOA.image_values(img_root_from_wrapper) == TOA.image_values(img_from_wrapper)
    @test TOA.image_values(img_root_from_compiled) == TOA.image_values(img_from_wrapper)
    @test land_root_from_wrapper.values == land_root_from_compiled.values

    decomp_threaded_from_wrapper = Inv.mpp_decomposition(M2, pi2, opts2; N=7, delta=0.2, q=0.5, tie_break=:center, precompute=:barcodes)
    decomp_threaded_from_cache = Inv.mpp_decomposition(Inv.fibered_barcode_cache_2d(M2, arr_mpp; precompute=:barcodes); N=7, delta=0.2, q=0.5, tie_break=:center)
    @test decomp_threaded_from_wrapper.lines == decomp_threaded_from_cache.lines
    @test decomp_threaded_from_wrapper.box == decomp_threaded_from_cache.box
    @test length(decomp_threaded_from_wrapper.summands) == length(decomp_threaded_from_cache.summands)
    @test isapprox(sum(decomp_threaded_from_wrapper.weights), sum(decomp_threaded_from_cache.weights); atol=1e-12)

    # ------------------------------------------------------------------
    # Feature-map direct API
    # ------------------------------------------------------------------
    bar = Dict((0.0, 1.0)=>1, (0.25, 0.75)=>1)
    pim = Inv.persistence_image(bar; xgrid=0.0:0.5:1.0, ygrid=0.0:0.5:1.0, sigma=0.2)
    @test Inv.feature_map(pim; flatten=false) == pim.values
    @test Inv.feature_map(pim; flatten=true) == vec(pim.values)

    # ------------------------------------------------------------------
    # Type-level API contracts for exported invariant data structures
    # ------------------------------------------------------------------
    @testset "Exported invariant type-level contracts" begin
        # Point signed measure
        @test pm isa SM.PointSignedMeasure
        @test length(pm.inds) == length(pm.wts)
        @test length(pm.axes) == 1

        # Fibered arrangement
        @test arr2 isa Inv.FiberedArrangement2D
        @test arr2.total_cells >= 0
        @test length(arr2.dir_reps) == length(arr2.orders) == length(arr2.unique_pos) == length(arr2.noff)
        @test length(arr2.start) == length(arr2.noff)
        @test arr2.total_cells == sum(arr2.noff)
        @test length(arr2.cell_chain_id) == arr2.total_cells
        @test arr2.n_cell_computed <= arr2.total_cells

        # Fibered slice family
        fam_types = Inv.fibered_slice_family_2d(arr2; direction_weight=:lesnick_l1, store_values=true)
        @test fam_types isa Inv.FiberedSliceFamily2D
        @test fam_types.arrangement === arr2
        @test Inv.nslices(fam_types) == length(fam_types.dir_idx) == length(fam_types.off_idx) ==
              length(fam_types.cell_id) == length(fam_types.chain_id) ==
              length(fam_types.off_mid) == length(fam_types.off0) == length(fam_types.off1) ==
              length(fam_types.vals_start) == length(fam_types.vals_len)
        @test fam_types.direction_weight_scheme == :lesnick_l1
        @test fam_types.store_values
        for k in 1:Inv.nslices(fam_types)
            cid = fam_types.chain_id[k]
            @test cid > 0
            vv = Inv.fibered_values(fam_types, k)
            @test length(vv) == fam_types.vals_len[k]
            if fam_types.vals_len[k] > 0
                @test length(vv) == length(arr2.chains[cid]) + 1
            end
        end

        # Fibered barcode cache
        @test fbc isa Inv.FiberedBarcodeCache2D
        @test length(fbc.index_barcodes_packed) == length(arr2.chains)
        @test fbc.n_barcode_computed >= 0
        @test fbc.M === M2
        @test fbc.arrangement === arr2

        # Projected arrangements + caches
        @test parr isa Inv.ProjectedArrangement
        @test length(parr.projections) == 2
        for pr in parr.projections
            @test pr isa Inv.ProjectedArrangement1D
            @test first(pr.chain) == 1
            @test last(pr.chain) == TO.nvertices(pr.C)
            @test length(pr.values) == length(pr.chain) + 1
            @test length(pr.dir) == 2
        end

        @test pc2 isa Inv.ProjectedBarcodeCache
        @test pc3 isa Inv.ProjectedBarcodeCache
        @test pc2.arrangement === parr
        @test length(pc2.packed_barcodes) == length(parr.projections)
        @test pc2.n_computed >= 0

        # Persistence landscape/image
        pl = Inv.persistence_landscape(bar; kmax=3, tgrid=[0.0, 0.5, 1.0])
        @test pl isa Inv.PersistenceLandscape1D
        @test issorted(pl.tgrid)
        @test size(pl.values, 1) == 3
        @test size(pl.values, 2) == length(pl.tgrid)
        @test all(isfinite, pl.values)
        @test all(x -> x >= 0.0, pl.values)

        @test pim isa Inv.PersistenceImage1D
        @test size(pim.values, 1) == length(pim.ygrid)
        @test size(pim.values, 2) == length(pim.xgrid)
        @test all(isfinite, pim.values)
        @test all(x -> x >= 0.0, pim.values)

        # MPP decomposition + line specs
        @test decomp_from_wrapper isa Inv.MPPDecomposition
        @test all(l -> l isa Inv.MPPLineSpec, decomp_from_wrapper.lines)
        @test length(decomp_from_wrapper.summands) == length(decomp_from_wrapper.weights)
        @test all(w -> w >= 0.0, decomp_from_wrapper.weights)
        for line in decomp_from_wrapper.lines
            @test length(line.dir) == 2
            @test length(line.x0) == 2
            @test isfinite(line.off)
            @test isfinite(line.omega)
            @test 0.0 <= line.omega <= 1.0 + 1e-12
        end
    end

    @testset "Fibered2D UX surface" begin
        fam = Inv.fibered_slice_family_2d(arr2; direction_weight=:lesnick_l1, store_values=true)

        d_arr = TO.describe(arr2)
        @test d_arr.kind == :fibered_arrangement_2d
        @test d_arr.backend == :boxes
        @test d_arr.ambient_dim == 2
        @test d_arr.total_cells == Inv.ncells(arr2)
        @test Inv.fibered_arrangement_summary(arr2) == d_arr
        @test occursin("FiberedArrangement2D", sprint(show, arr2))

        @test Inv.source_encoding(arr2) === pi2
        @test Inv.working_box(arr2) == arr2.box
        @test Inv.direction_representatives(arr2) === arr2.dir_reps
        @test Inv.slope_breaks(arr2) === arr2.slope_breaks
        @test Inv.ncells(arr2) == arr2.total_cells
        @test Inv.computed_cell_count(arr2) == arr2.n_cell_computed
        @test Inv.chain_count(arr2) == length(arr2.chains)
        @test Inv.backend(arr2) == :boxes

        d_cache = TO.describe(fbc)
        @test d_cache.kind == :fibered_barcode_cache_2d
        @test d_cache.cached_barcodes == Inv.cached_barcode_count(fbc)
        @test Inv.fibered_cache_summary(fbc) == d_cache
        @test occursin("FiberedBarcodeCache2D", sprint(show, fbc))

        @test Inv.source_module(fbc) === M2
        @test Inv.shared_arrangement(fbc) === arr2
        @test Inv.cached_barcode_count(fbc) == fbc.n_barcode_computed

        d_fs = TO.describe(fs)
        @test d_fs.kind == :fibered_slice_result
        @test d_fs.chain_length == length(Inv.slice_chain(fs))
        @test occursin("FiberedSliceResult", sprint(show, fs))

        d_fam = TO.describe(fam)
        @test d_fam.kind == :fibered_slice_family_2d
        @test d_fam.nslices == Inv.nslices(fam)
        @test d_fam.unique_chain_count == Inv.unique_chain_count(fam)
        @test Inv.fibered_family_summary(fam) == d_fam
        @test occursin("FiberedSliceFamily2D", sprint(show, fam))

        @test Inv.source_arrangement(fam) === arr2
        @test Inv.unique_chain_count(fam) == length(fam.unique_chain_ids)
        @test Inv.direction_weight_scheme(fam) == :lesnick_l1
        @test Inv.stores_values(fam)
        @test Inv.slice_direction(fam, 1) == arr2.dir_reps[fam.dir_idx[1]]
        @test Inv.slice_offset(fam, 1) == fam.off_mid[1]
        @test Inv.slice_offset_interval(fam, 1) == (fam.off0[1], fam.off1[1])
        @test Inv.slice_chain_id(fam, 1) == fam.chain_id[1]

        d_proj1 = TO.describe(parr.projections[1])
        @test d_proj1.kind == :projected_arrangement_1d
        @test Inv.projected_arrangement_summary(parr.projections[1]) == d_proj1
        @test occursin("ProjectedArrangement1D", sprint(show, parr.projections[1]))

        d_parr = TO.describe(parr)
        @test d_parr.kind == :projected_arrangement
        @test d_parr.nprojections == 2
        @test Inv.projected_arrangement_summary(parr) == d_parr
        @test occursin("ProjectedArrangement", sprint(show, parr))

        @test Inv.source_poset(parr) === parr.Q
        @test Inv.source_poset(parr.projections[1]) === parr.Q
        @test Inv.projections(parr) === parr.projections
        @test Inv.nprojections(parr) == 2
        @test Inv.projection_direction(parr.projections[1]) === parr.projections[1].dir
        @test Inv.projection_values(parr.projections[1]) === parr.projections[1].values
        @test Inv.projection_chain(parr.projections[1]) == parr.projections[1].chain

        d_pc = TO.describe(pc2)
        @test d_pc.kind == :projected_barcode_cache
        @test d_pc.computed_projections == Inv.computed_projection_count(pc2)
        @test Inv.projected_cache_summary(pc2) == d_pc
        @test occursin("ProjectedBarcodeCache", sprint(show, pc2))

        @test Inv.source_module(pc2) === M2
        @test Inv.computed_projection_count(pc2) == pc2.n_computed

        pb_res = Inv.projected_barcodes(pc2; inds=[1], threads=false)
        @test pb_res isa Inv.ProjectedBarcodesResult
        @test TO.describe(pb_res).kind == :projected_barcodes_result
        @test Inv.projection_indices(pb_res) == [1]
        @test length(Inv.projection_directions(pb_res)) == 1
        @test length(Inv.projected_barcodes(pb_res)) == 1
        @test occursin("ProjectedBarcodesResult", sprint(show, pb_res))

        pd_res = Inv.projected_distances(pc2, pc3; dist=:bottleneck, threads=false)
        @test pd_res isa Inv.ProjectedDistancesResult
        @test TO.describe(pd_res).kind == :projected_distances_result
        @test Inv.distance_metric(pd_res) == :bottleneck
        @test length(Inv.projected_distances(pd_res)) == 2
        @test length(Inv.projection_directions(pd_res)) == 2
        @test occursin("ProjectedDistancesResult", sprint(show, pd_res))

        fibered_grid = Inv.slice_barcodes(
            fbc;
            dirs=[(1.0, 1.0), (1.0, 2.0)],
            offsets=[0.0],
            packed=true,
            normalize_weights=false,
            threads=false,
        )
        @test fibered_grid isa SI.SliceBarcodesResult
        @test size(SI.slice_weights(fibered_grid)) == (2, 1)
        @test length(SI.slice_directions(fibered_grid)) == 2
        @test SI.slice_offsets(fibered_grid) == [0.0]
        @test !isnothing(SI.packed_barcodes(fibered_grid))
        @test TO.describe(fibered_grid).kind == :slice_barcodes_result

        qsum_arr = Inv.fibered_query_summary(arr2, (1.0, 1.0), 0.0)
        @test qsum_arr.kind == :fibered_query_summary
        @test qsum_arr.valid
        @test qsum_arr.normalized_direction == [0.5, 0.5]
        @test qsum_arr.chain_length == length(ch)
        @test qsum_arr.barcode_intervals === nothing

        qsum_cache = Inv.fibered_query_summary(fbc, (1.0, 1.0), [0.0, 0.0])
        @test qsum_cache.valid
        @test qsum_cache.basepoint == [0.0, 0.0]
        @test qsum_cache.offset == 0.0
        @test qsum_cache.barcode_intervals == length(Inv.slice_barcode(fs))
        @test qsum_cache.barcode_total_multiplicity == sum(values(Inv.slice_barcode(fs)))

        arr_report = Inv.check_fibered_arrangement_2d(arr2; throw=false)
        cache_report = Inv.check_fibered_barcode_cache_2d(fbc; throw=false)
        fam_report = Inv.check_fibered_slice_family_2d(fam; throw=false)
        parr_report = Inv.check_projected_arrangement(parr; throw=false)
        pc_report = Inv.check_projected_barcode_cache(pc2; throw=false)
        @test arr_report.valid
        @test cache_report.valid
        @test fam_report.valid
        @test parr_report.valid
        @test pc_report.valid

        wrap = Inv.fibered2d_validation_summary(arr_report)
        @test wrap isa Inv.Fibered2DValidationSummary
        @test occursin("Fibered2DValidationSummary", sprint(show, wrap))
        @test occursin("Fibered2DValidationSummary", sprint(show, MIME"text/plain"(), wrap))

        @test Inv.check_fibered_direction((1.0, 1.0); throw=false).valid
        @test !Inv.check_fibered_direction((-1.0, 1.0); throw=false).valid
        @test Inv.check_fibered_offset(0.0; throw=false).valid
        @test !Inv.check_fibered_offset(Inf; throw=false).valid
        @test Inv.check_fibered_basepoint([0.0, 0.0]; throw=false).valid
        @test !Inv.check_fibered_basepoint([0.0]; throw=false).valid
        @test Inv.check_projected_direction((1.0, -1.0); throw=false).valid
        @test !Inv.check_projected_direction((0.0, 0.0); throw=false).valid

        q_report = Inv.check_fibered_query(arr2, (1.0, 1.0), 0.0; throw=false)
        @test q_report.valid
        @test hasproperty(q_report, :tie_break_relevant)
        q_bad = Inv.check_fibered_query(arr2, (1.0, 0.0), 0.0; throw=false)
        @test !q_bad.valid
        @test any(occursin("include_axes", msg) for msg in q_bad.issues)

        arr_bad = deepcopy(arr2)
        push!(arr_bad.cell_chain_id, 1)
        @test !Inv.check_fibered_arrangement_2d(arr_bad; throw=false).valid
        @test_throws ArgumentError Inv.check_fibered_arrangement_2d(arr_bad; throw=true)

        cache_bad = deepcopy(fbc)
        push!(cache_bad.index_barcodes_packed, nothing)
        @test !Inv.check_fibered_barcode_cache_2d(cache_bad; throw=false).valid
        @test_throws ArgumentError Inv.check_fibered_barcode_cache_2d(cache_bad; throw=true)

        fam_bad = deepcopy(fam)
        pop!(fam_bad.off1)
        @test !Inv.check_fibered_slice_family_2d(fam_bad; throw=false).valid
        @test_throws ArgumentError Inv.check_fibered_slice_family_2d(fam_bad; throw=true)

        parr_bad = deepcopy(parr)
        pop!(parr_bad.projections[1].values)
        @test !Inv.check_projected_arrangement(parr_bad; throw=false).valid
        @test_throws ArgumentError Inv.check_projected_arrangement(parr_bad; throw=true)

        pc_bad = deepcopy(pc2)
        push!(pc_bad.packed_barcodes, nothing)
        @test !Inv.check_projected_barcode_cache(pc_bad; throw=false).valid
        @test_throws ArgumentError Inv.check_projected_barcode_cache(pc_bad; throw=true)

        arr_other = Inv.fibered_arrangement_2d(pi2, opts2; normalize_dirs=:L1, precompute=:none)
        cache_other = Inv.fibered_barcode_cache_2d(M3, arr_other; precompute=:none)
        pair_report = Inv.check_fibered_cache_pair(fbc, cache_other; throw=false)
        @test !pair_report.valid
        @test_throws ArgumentError Inv.check_fibered_cache_pair(fbc, cache_other; throw=true)

        parr_other = Inv.projected_arrangement(pi2; dirs=[(1.0, 0.0), (0.0, 1.0)], threads=false)
        pc_other = Inv.projected_barcode_cache(M3, parr_other; precompute=false)
        proj_pair_report = Inv.check_projected_cache_pair(pc2, pc_other; throw=false)
        @test !proj_pair_report.valid
        @test_throws ArgumentError Inv.check_projected_cache_pair(pc2, pc_other; throw=true)

        @test TOA.FiberedArrangement2D === Inv.FiberedArrangement2D
        @test TOA.ProjectedBarcodeCache === Inv.ProjectedBarcodeCache
        @test TOA.Fibered2DValidationSummary === Inv.Fibered2DValidationSummary
        @test TOA.FiberedSliceResult === Inv.FiberedSliceResult
        @test TOA.ProjectedBarcodesResult === Inv.ProjectedBarcodesResult
        @test TOA.ProjectedDistancesResult === Inv.ProjectedDistancesResult
        @test TOA.source_encoding(arr2) === pi2
        @test TOA.source_module(fbc) === M2
        @test TOA.source_poset(parr) === parr.Q
        @test TOA.nslices(fam) == Inv.nslices(fam)
        @test TOA.projections(parr) === parr.projections
        @test TOA.nprojections(parr) == 2
        @test TOA.slice_values(fs) == Inv.slice_values(fs)
        @test TOA.projection_indices(pb_res) == [1]
        @test TOA.distance_metric(pd_res) == :bottleneck
        @test TOA.cached_barcode_count(fbc) == Inv.cached_barcode_count(fbc)
        @test TOA.fibered_arrangement_summary(arr2) == d_arr
        @test TOA.projected_cache_summary(pc2) == d_pc
        @test TOA.fibered_query_summary(fbc, (1.0, 1.0), 0.0).valid
        @test TOA.fibered2d_validation_summary(arr_report) isa Inv.Fibered2DValidationSummary
        @test TOA.check_fibered_arrangement_2d(arr2; throw=false).valid
    end
end
end # with_fields
