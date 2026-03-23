using Test

const FEA = TamerOp.Featurizers

# In include-based test harnesses (no package-extension autoload), manually
# load optional table/IO extensions when their deps are present.
if Base.find_package("Tables") !== nothing && !isdefined(Main, :TamerOpTablesExt)
    include(joinpath(@__DIR__, "..", "ext", "TamerOpTablesExt.jl"))
end
if Base.find_package("Arrow") !== nothing && !isdefined(Main, :TamerOpArrowExt)
    include(joinpath(@__DIR__, "..", "ext", "TamerOpArrowExt.jl"))
end
if Base.find_package("Parquet2") !== nothing && !isdefined(Main, :TamerOpParquet2Ext)
    include(joinpath(@__DIR__, "..", "ext", "TamerOpParquet2Ext.jl"))
end
if Base.find_package("NPZ") !== nothing && !isdefined(Main, :TamerOpNPZExt)
    include(joinpath(@__DIR__, "..", "ext", "TamerOpNPZExt.jl"))
end
if Base.find_package("CSV") !== nothing && !isdefined(Main, :TamerOpCSVExt)
    include(joinpath(@__DIR__, "..", "ext", "TamerOpCSVExt.jl"))
end
if Base.find_package("Folds") !== nothing && !isdefined(Main, :TamerOpFoldsExt)
    include(joinpath(@__DIR__, "..", "ext", "TamerOpFoldsExt.jl"))
end
if Base.find_package("KernelFunctions") !== nothing && !isdefined(Main, :TamerOpKernelFunctionsExt)
    include(joinpath(@__DIR__, "..", "ext", "TamerOpKernelFunctionsExt.jl"))
end
if Base.find_package("Distances") !== nothing && !isdefined(Main, :TamerOpDistancesExt)
    include(joinpath(@__DIR__, "..", "ext", "TamerOpDistancesExt.jl"))
end

@testset "Workflow featurizer specs and batch featurize" begin
    field = CM.QQField()

    Ups = [PLB.BoxUpset([0.0, -10.0]), PLB.BoxUpset([1.0, -10.0])]
    Downs = PLB.BoxDownset[]
    opts_enc = TO.EncodingOptions(field=field)
    P, H, pi = PLB.encode_fringe_boxes(Ups, Downs, opts_enc)

    r2 = EC.locate(pi, [0.5, 0.0])
    r3 = EC.locate(pi, [2.0, 0.0])

    M23 = IR.pmodule_from_fringe(one_by_one_fringe(P, FF.principal_upset(P, r2), FF.principal_downset(P, r3), 1; field=field))
    M3  = IR.pmodule_from_fringe(one_by_one_fringe(P, FF.principal_upset(P, r3), FF.principal_downset(P, r3), 1; field=field))

    enc23 = TO.EncodingResult(P, M23, pi)
    enc3 = TO.EncodingResult(P, M3, pi)
    samples = [enc23, enc3]
    opts_inv = TO.InvariantOptions(box=([-1.0, -1.0], [2.0, 1.0]), strict=false)

    function manual_barcode_window(bar, window)
        window !== nothing && return (Float64(window[1]), Float64(window[2]))
        lo = Inf
        hi = -Inf
        seen = false
        for ((b0, d0), mult) in bar
            mult <= 0 && continue
            b = Float64(b0)
            d = Float64(d0)
            if isfinite(b)
                lo = min(lo, b)
                hi = max(hi, b)
                seen = true
            end
            if isfinite(d)
                lo = min(lo, d)
                hi = max(hi, d)
                seen = true
            end
        end
        return seen ? (lo, hi) : nothing
    end

    function clipped_barcode(bar; window=nothing, infinite_policy=:clip_to_window)
        work = manual_barcode_window(bar, window)
        if infinite_policy == :clip_to_window && work === nothing
            throw(ArgumentError("test oracle needs a finite window for unbounded intervals"))
        end
        lo = work === nothing ? 0.0 : work[1]
        hi = work === nothing ? 0.0 : work[2]
        out = Dict{Tuple{Float64,Float64},Int}()
        for ((b0, d0), mult) in bar
            mult <= 0 && continue
            b = Float64(b0)
            d = Float64(d0)
            if !isfinite(b) || !isfinite(d)
                infinite_policy == :error && throw(ArgumentError("test oracle hit unbounded interval"))
                b = isfinite(b) ? b : (b > 0 ? hi : lo)
                d = isfinite(d) ? d : (d > 0 ? hi : lo)
            end
            d > b || continue
            out[(b, d)] = get(out, (b, d), 0) + mult
        end
        return out
    end

    function manual_topk_vector(bar, k; window=nothing, infinite_policy=:clip_to_window)
        bc = clipped_barcode(bar; window=window, infinite_policy=infinite_policy)
        entries = collect(bc)
        sort!(entries; by=x -> (-(x[1][2] - x[1][1]), x[1][1], x[1][2]))
        out = zeros(Float64, 1 + 3 * k)
        out[1] = Float64(sum(values(bc)))
        idx = 2
        remaining = k
        for ((b, d), mult) in entries
            remaining == 0 && break
            copies = min(mult, remaining)
            for _ in 1:copies
                out[idx] = 1.0
                out[idx + 1] = b
                out[idx + 2] = d - b
                idx += 3
            end
            remaining -= copies
        end
        return out
    end

    function manual_topk_vector_packed(pb, k; window=nothing, infinite_policy=:clip_to_window)
        work = window
        if work === nothing
            lo = Inf
            hi = -Inf
            seen = false
            for p in pb.pairs
                b = Float64(p.b)
                d = Float64(p.d)
                if isfinite(b)
                    lo = min(lo, b)
                    hi = max(hi, b)
                    seen = true
                end
                if isfinite(d)
                    lo = min(lo, d)
                    hi = max(hi, d)
                    seen = true
                end
            end
            work = seen ? (lo, hi) : nothing
        end
        if infinite_policy == :clip_to_window && work === nothing
            throw(ArgumentError("test oracle needs a finite window for unbounded packed intervals"))
        end
        lo = work === nothing ? 0.0 : work[1]
        hi = work === nothing ? 0.0 : work[2]
        entries = Tuple{Float64,Float64,Float64,Int}[]
        count = 0
        for idx in eachindex(pb.pairs)
            mult = pb.mults[idx]
            mult <= 0 && continue
            p = pb.pairs[idx]
            b = Float64(p.b)
            d = Float64(p.d)
            if !isfinite(b) || !isfinite(d)
                infinite_policy == :error && throw(ArgumentError("test oracle hit unbounded packed interval"))
                b = isfinite(b) ? b : (b > 0 ? hi : lo)
                d = isfinite(d) ? d : (d > 0 ? hi : lo)
            end
            d > b || continue
            push!(entries, (d - b, b, d, mult))
            count += mult
        end
        sort!(entries; by=x -> (-x[1], x[2], x[3]))
        out = zeros(Float64, 1 + 3 * k)
        out[1] = Float64(count)
        idx = 2
        remaining = k
        for (pers, birth, _, mult) in entries
            remaining == 0 && break
            copies = min(mult, remaining)
            for _ in 1:copies
                out[idx] = 1.0
                out[idx + 1] = birth
                out[idx + 2] = pers
                idx += 3
            end
            remaining -= copies
        end
        return out
    end

    function manual_summary_vector(bar, fields; window=nothing, infinite_policy=:clip_to_window, normalize_entropy=true)
        bc = clipped_barcode(bar; window=window, infinite_policy=infinite_policy)
        nt = Inv.barcode_summary(bc; normalize_entropy=normalize_entropy)
        lookup = Dict(
            :count => Float64(nt.n_intervals),
            :sum_persistence => Float64(nt.total_persistence),
            :mean_persistence => Float64(nt.mean_persistence),
            :max_persistence => Float64(nt.max_persistence),
            :entropy => Float64(nt.entropy),
        )
        return Float64[lookup[f] for f in fields]
    end

    function aggregate_barcode_oracle(barcodes, weights, builder, aggregate)
        nd, no = size(barcodes)
        feats = [builder(barcodes[i, j]) for i in 1:nd, j in 1:no]
        if aggregate == :stack
            out = Float64[]
            for i in 1:nd, j in 1:no
                append!(out, feats[i, j])
            end
            return out
        end
        out = zeros(Float64, length(feats[1, 1]))
        sumw = sum(weights)
        for i in 1:nd, j in 1:no
            out .+= weights[i, j] .* feats[i, j]
        end
        aggregate == :mean && (out ./= sumw)
        return out
    end

    lspec = TO.LandscapeSpec(
        directions=[[1.0, 1.0]],
        offsets=[[0.0, 0.0]],
        tgrid=collect(0.0:0.5:3.0),
        kmax=2,
        strict=false,
    )
    lv = TO.transform(lspec, enc23; opts=opts_inv, threaded=false)
    @test length(lv) == TO.nfeatures(lspec)
    @test length(TO.feature_names(lspec)) == TO.nfeatures(lspec)
    lax = TO.feature_axes(lspec)
    @test lax.k == collect(1:lspec.kmax)
    @test lax.t == lspec.tgrid
    @test lax.aggregate == lspec.aggregate
    @test FEA.supports(lspec, enc23)
    @test !FEA.supports(lspec, M23)

    pispec = TO.PersistenceImageSpec(
        directions=[[1.0, 1.0]],
        offsets=[[0.0, 0.0]],
        xgrid=collect(0.0:0.5:2.0),
        ygrid=collect(0.0:0.5:2.0),
        sigma=0.35,
        strict=false,
    )
    piv = TO.transform(pispec, enc23; opts=opts_inv, threaded=false)
    @test length(piv) == TO.nfeatures(pispec)
    piax = TO.feature_axes(pispec)
    @test piax.x == pispec.xgrid
    @test piax.y == pispec.ygrid

    mlspec = TO.MPLandscapeSpec(
        directions=[[1.0, 1.0]],
        offsets=[[0.0, 0.0]],
        tgrid=collect(0.0:0.5:3.0),
        kmax=2,
        strict=false,
        direction_weight=:uniform,
    )
    mlv = TO.transform(mlspec, enc23; opts=opts_inv, threaded=false)
    @test length(mlv) == TO.nfeatures(mlspec)
    mlax = TO.feature_axes(mlspec)
    @test mlax.k == collect(1:mlspec.kmax)
    @test mlax.t == mlspec.tgrid
    mlopts = OPT.InvariantOptions(
        axes=opts_inv.axes,
        axes_policy=opts_inv.axes_policy,
        max_axis_len=opts_inv.max_axis_len,
        box=opts_inv.box,
        threads=opts_inv.threads,
        strict=opts_inv.strict,
        pl_mode=opts_inv.pl_mode,
    )
    mlplan = Inv.compile_slices(
        enc23.pi,
        mlopts;
        directions=mlspec.directions,
        offsets=mlspec.offsets,
        offset_weights=mlspec.offset_weights,
        normalize_weights=mlspec.normalize_weights,
        tmin=mlspec.tmin,
        tmax=mlspec.tmax,
        nsteps=mlspec.nsteps,
        drop_unknown=mlspec.drop_unknown,
        dedup=mlspec.dedup,
        normalize_dirs=mlspec.normalize_dirs,
        direction_weight=mlspec.direction_weight,
        threads=false,
    )
    mlobj = Inv.mp_landscape(M23, mlplan; kmax=mlspec.kmax, tgrid=mlspec.tgrid, normalize_weights=mlspec.normalize_weights, threads=false)
    ml_expected = Float64[mlobj.values[i, j, k, t]
                          for t in 1:size(mlobj.values, 4)
                          for k in 1:size(mlobj.values, 3)
                          for j in 1:size(mlobj.values, 2)
                          for i in 1:size(mlobj.values, 1)]
    @test isapprox(mlv, ml_expected; atol=1e-12, rtol=0.0)

    espec = TO.EulerSurfaceSpec(
        axes=([-1.0, 0.0, 1.0, 2.0], [-1.0, 0.0, 1.0]),
        axes_policy=:as_given,
        strict=false,
    )
    ev = TO.transform(espec, enc23; opts=opts_inv, threaded=false)
    @test length(ev) == TO.nfeatures(espec)
    eax = TO.feature_axes(espec)
    @test eax.axis_1 == espec.axes[1]
    @test eax.axis_2 == espec.axes[2]
    @test eax.axes_policy == espec.axes_policy

    rspec = TO.RankGridSpec(nvertices=TO.nvertices(P), store_zeros=true)
    rv = TO.transform(rspec, enc23; opts=opts_inv, threaded=false)
    @test length(rv) == TO.nfeatures(rspec)
    rax = TO.feature_axes(rspec)
    @test length(rax.a) == TO.nvertices(P)
    @test length(rax.b) == TO.nvertices(P)

    rhspec = TO.RestrictedHilbertSpec(nvertices=TO.nvertices(P))
    @test FEA.supports(rhspec, enc23)
    rhv = TO.transform(rhspec, enc23; opts=opts_inv, threaded=false)
    @test length(rhv) == TO.nfeatures(rhspec)
    @test rhv == Float64.(Inv.restricted_hilbert(M23))
    rhax = TO.feature_axes(rhspec)
    @test rhax.vertex == collect(1:TO.nvertices(P))

    sspec = TO.SlicedBarcodeSpec(
        directions=[[1.0, 1.0]],
        offsets=[[0.0, 0.0]],
        featurizer=:summary,
        strict=false,
    )
    sv = TO.transform(sspec, enc23; opts=opts_inv, threaded=false)
    @test length(sv) == TO.nfeatures(sspec)
    sax = TO.feature_axes(sspec)
    @test sax.aggregate == sspec.aggregate
    @test sax.stat == Symbol.(sspec.summary_fields)

    topkspec = TO.BarcodeTopKSpec(
        directions=[[1.0, 1.0], [1.0, 0.0]],
        offsets=[[0.0, 0.0], [0.5, 0.0]],
        k=2,
        aggregate=:stack,
        source=:slice,
        strict=false,
    )
    @test FEA.supports(topkspec, enc23)
    topkv = TO.transform(topkspec, enc23; opts=opts_inv, threaded=false)
    topk_data = Inv.slice_barcodes(
        M23,
        enc23.pi;
        opts=opts_inv,
        directions=topkspec.directions,
        offsets=topkspec.offsets,
        normalize_dirs=topkspec.normalize_dirs,
        direction_weight=topkspec.direction_weight,
        offset_weights=topkspec.offset_weights,
        normalize_weights=topkspec.normalize_weights,
        drop_unknown=topkspec.drop_unknown,
        tmin=topkspec.tmin,
        tmax=topkspec.tmax,
        nsteps=topkspec.nsteps,
        packed=false,
        threads=false,
    )
    topk_expected = aggregate_barcode_oracle(
        topk_data.barcodes,
        topk_data.weights,
        bc -> manual_topk_vector(bc, topkspec.k; window=topkspec.window, infinite_policy=topkspec.infinite_policy),
        topkspec.aggregate,
    )
    @test topkv == topk_expected
    @test length(topkv) == TO.nfeatures(topkspec)
    @test TO.feature_axes(topkspec).source == :slice

    topkspec_fibered = TO.BarcodeTopKSpec(
        directions=topkspec.directions,
        offsets=topkspec.offsets,
        k=topkspec.k,
        aggregate=:stack,
        source=:fibered,
        window=(-0.5, 2.0),
        strict=false,
    )
    @test FEA.supports(topkspec_fibered, enc23)
    topkv_fibered = TO.transform(topkspec_fibered, enc23; opts=opts_inv, threaded=false)
    fcache = Inv.fibered_barcode_cache_2d(
        M23,
        enc23.pi,
        opts_inv;
        include_axes=true,
        normalize_dirs=topkspec_fibered.normalize_dirs,
        threads=false,
    )
    topk_fibered_data = Inv.slice_barcodes(
        fcache;
        dirs=topkspec_fibered.directions,
        offsets=topkspec_fibered.offsets,
        values=:t,
        packed=true,
        direction_weight=topkspec_fibered.direction_weight,
        offset_weights=topkspec_fibered.offset_weights,
        normalize_weights=topkspec_fibered.normalize_weights,
        threads=false,
    )
    topk_expected_fibered = aggregate_barcode_oracle(
        topk_fibered_data.barcodes,
        topk_fibered_data.weights,
        bc -> manual_topk_vector_packed(bc, topkspec_fibered.k; window=topkspec_fibered.window, infinite_policy=topkspec_fibered.infinite_policy),
        topkspec_fibered.aggregate,
    )
    @test topkv_fibered == topk_expected_fibered

    bsumspec = TO.BarcodeSummarySpec(
        directions=topkspec.directions,
        offsets=topkspec.offsets,
        aggregate=:mean,
        source=:slice,
        strict=false,
    )
    @test FEA.supports(bsumspec, enc23)
    bsumv = TO.transform(bsumspec, enc23; opts=opts_inv, threaded=false)
    bsum_expected = aggregate_barcode_oracle(
        topk_data.barcodes,
        topk_data.weights,
        bc -> manual_summary_vector(
            bc,
            bsumspec.fields;
            window=bsumspec.window,
            infinite_policy=bsumspec.infinite_policy,
            normalize_entropy=bsumspec.normalize_entropy,
        ),
        bsumspec.aggregate,
    )
    @test isapprox(bsumv, bsum_expected; atol=1e-12, rtol=0.0)
    @test TO.feature_axes(bsumspec).field == collect(bsumspec.fields)

    pmspec = TO.PointSignedMeasureSpec(ndims=2, k=2, coords=:values)
    pm_direct = SM.PointSignedMeasure(
        (Float64[0.0, 1.0], Float64[10.0, 20.0]),
        [(2, 1), (1, 2), (2, 2)],
        [3.0, -5.0, 1.0],
    )
    @test FEA.supports(pmspec, pm_direct)
    pmv = TO.transform(pmspec, pm_direct; threaded=false)
    @test pmv == [3.0, 1.0, -5.0, 0.0, 20.0, 1.0, 3.0, 1.0, 10.0]
    pmcache = TO.build_cache(pm_direct, pmspec; threaded=false)
    @test TO.cache_stats(pmcache).kind == :point_measure
    @test TO.transform(pmspec, pmcache; threaded=false) == pmv

    eulerspec = TO.EulerSignedMeasureSpec(ndims=2, k=4, coords=:values, strict=false)
    @test FEA.supports(eulerspec, enc23)
    eulerv = TO.transform(eulerspec, enc23; opts=opts_inv, threaded=false)
    eopts = OPT.InvariantOptions(
        axes=eulerspec.axes === nothing ? opts_inv.axes : eulerspec.axes,
        axes_policy=eulerspec.axes_policy,
        max_axis_len=eulerspec.max_axis_len,
        box=opts_inv.box,
        threads=false,
        strict=eulerspec.strict === nothing ? opts_inv.strict : eulerspec.strict,
        pl_mode=opts_inv.pl_mode,
    )
    epm = SM.euler_signed_measure(M23, enc23.pi, eopts;
                                   drop_zeros=eulerspec.drop_zeros,
                                   max_terms=eulerspec.max_terms > 0 ? max(eulerspec.max_terms, eulerspec.k) : 0,
                                   min_abs_weight=eulerspec.min_abs_weight)
    euler_expected = FEA._point_measure_topk_vector(epm, eulerspec.ndims, eulerspec.k, eulerspec.coords)
    @test eulerv == euler_expected
    @test length(eulerv) == TO.nfeatures(eulerspec)

    mpphistspec = TO.MPPDecompositionHistogramSpec(
        orientation_bins=4,
        scale_bins=3,
        scale_range=(0.0, 1.0),
        weight=:mass,
        normalize=:l1,
        N=8,
        q=1.0,
    )
    @test FEA.supports(mpphistspec, enc23)
    mpphistv = TO.transform(mpphistspec, enc23; opts=opts_inv, threaded=false)
    @test length(mpphistv) == TO.nfeatures(mpphistspec)
    @test isapprox(sum(mpphistv[1:(mpphistspec.orientation_bins * mpphistspec.scale_bins)]), 1.0; atol=1e-12, rtol=0.0)
    @test all(isfinite, mpphistv)

    synthetic_decomp = Inv.MPPDecomposition(
        Inv.MPPLineSpec[],
        [
            [([0.0, 0.0], [1.0, 0.0], 1.0), ([0.0, 0.0], [0.0, 1.0], 1.0)],
            [([0.0, 0.0], [1.0, 1.0], 1.0)],
        ],
        [2.0, 1.0],
        ([0.0, 0.0], [1.0, 1.0]),
    )
    synthetic_hist = TO.MPPDecompositionHistogramSpec(
        orientation_bins=4,
        scale_bins=3,
        scale_range=(0.0, 1.0),
        weight=:mass,
        normalize=:l1,
    )
    synthetic_hist_v = FEA._mpp_histogram_vector_from_decomp(synthetic_hist, synthetic_decomp)
    @test synthetic_hist_v[1:8] == zeros(8)
    @test isapprox(synthetic_hist_v[9:12], [1/3, 1/3, 1/3, 0.0]; atol=1e-12, rtol=0.0)
    @test isapprox(synthetic_hist_v[13], 3.0; atol=1e-12, rtol=0.0)
    @test synthetic_hist_v[14] == 2.0
    @test synthetic_hist_v[15] == 3.0
    @test isapprox(synthetic_hist_v[16], 1.0; atol=1e-12, rtol=0.0)
    @test isapprox(synthetic_hist_v[17], 0.9182958340544894; atol=1e-12, rtol=0.0)

    mpphistspec2 = TO.MPPDecompositionHistogramSpec(
        orientation_bins=6,
        scale_bins=2,
        scale_range=(0.0, 1.0),
        weight=:count,
        normalize=:none,
        N=mpphistspec.N,
        q=mpphistspec.q,
    )
    mpphist_pair = TO.CompositeSpec((mpphistspec, mpphistspec2), namespacing=true)
    mpphist_cache = TO.build_cache(enc23, mpphist_pair; opts=opts_inv, threaded=false, level=:fibered)
    mpphist_pair_v = TO.transform(mpphist_pair, mpphist_cache; opts=opts_inv, threaded=false)
    @test isapprox(
        mpphist_pair_v,
        vcat(
            TO.transform(mpphistspec, enc23; opts=opts_inv, threaded=false),
            TO.transform(mpphistspec2, enc23; opts=opts_inv, threaded=false),
        );
        atol=1e-12,
        rtol=0.0,
    )
    @test TO.cache_stats(mpphist_cache).n_mpp_decompositions == 1

    sbispec = TO.SignedBarcodeImageSpec(
        xs=collect(0.0:0.5:2.0),
        ys=collect(0.0:0.5:2.0),
        sigma=0.35,
        axes=([-1.0, 0.0, 1.0, 2.0], [-1.0, 0.0, 1.0]),
        axes_policy=:as_given,
        strict=false,
    )
    @test TO.nfeatures(sbispec) == length(sbispec.xs) * length(sbispec.ys)
    sbax = TO.feature_axes(sbispec)
    @test sbax.x == sbispec.xs
    @test sbax.y == sbispec.ys
    @test sbax.mode == sbispec.mode
    @test !FEA.supports(sbispec, enc23)
    @test_throws ArgumentError TO.transform(sbispec, enc23; opts=opts_inv, threaded=false)

    dspec = TO.ProjectedDistancesSpec(
        [enc3];
        reference_names=[:M3],
        n_dirs=4,
        normalize=:L1,
        precompute=true,
    )
    dv = TO.transform(dspec, enc23; opts=opts_inv, threaded=false)
    @test length(dv) == TO.nfeatures(dspec)
    @test dv[1] >= 0.0
    dax = TO.feature_axes(dspec)
    @test dax.reference == dspec.reference_names
    @test dax.dist == dspec.dist

    mppspec = TO.MPPImageSpec(
        resolution=6,
        sigma=0.15,
        N=8,
        q=1.0,
        segment_prune=true,
    )
    @test FEA.supports(mppspec, enc23)
    mppv = TO.transform(mppspec, enc23; opts=opts_inv, threaded=false)
    @test length(mppv) == TO.nfeatures(mppspec)
    mppax = TO.feature_axes(mppspec)
    @test length(mppax.x) == mppspec.resolution
    @test length(mppax.y) == mppspec.resolution
    mppobj = TO.mpp_image(
        enc23;
        opts=opts_inv,
        resolution=mppspec.resolution,
        sigma=mppspec.sigma,
        N=mppspec.N,
        delta=mppspec.delta,
        q=mppspec.q,
        tie_break=mppspec.tie_break,
        cutoff_radius=mppspec.cutoff_radius,
        cutoff_tol=mppspec.cutoff_tol,
        segment_prune=mppspec.segment_prune,
        threads=false,
    )
    mpp_expected = Float64[mppobj.img[iy, ix]
                           for ix in 1:length(mppobj.xgrid)
                           for iy in 1:length(mppobj.ygrid)]
    @test isapprox(mppv, mpp_expected; atol=1e-12, rtol=0.0)

    mlspec2 = TO.MPLandscapeSpec(
        directions=[[1.0, 1.0]],
        offsets=[[0.0, 0.0]],
        tgrid=collect(0.0:0.5:3.0),
        kmax=1,
        strict=false,
        direction_weight=:uniform,
    )
    mlv2 = TO.transform(mlspec2, enc23; opts=opts_inv, threaded=false)
    mlpair = TO.CompositeSpec((mlspec, mlspec2), namespacing=true)
    mlpair_cache = TO.build_cache(enc23, mlpair; opts=opts_inv, threaded=false)
    mlpair_v = TO.transform(mlpair, mlpair_cache; opts=opts_inv, threaded=false)
    @test isapprox(mlpair_v, vcat(mlv, mlv2); atol=1e-12, rtol=0.0)
    @test TO.cache_stats(mlpair_cache).n_slice_plans == 1

    mppspec2 = TO.MPPImageSpec(
        resolution=5,
        sigma=0.2,
        N=mppspec.N,
        q=mppspec.q,
        tie_break=mppspec.tie_break,
        segment_prune=false,
    )
    mppv2 = TO.transform(mppspec2, enc23; opts=opts_inv, threaded=false)
    mpppair = TO.CompositeSpec((mppspec, mppspec2), namespacing=true)
    mpppair_cache = TO.build_cache(enc23, mpppair; opts=opts_inv, threaded=false, level=:fibered)
    mpppair_v = TO.transform(mpppair, mpppair_cache; opts=opts_inv, threaded=false)
    @test isapprox(mpppair_v, vcat(mppv, mppv2); atol=1e-12, rtol=0.0)
    @test TO.cache_stats(mpppair_cache).n_mpp_decompositions == 1

    bettispec = TO.BettiTableSpec(nvertices=TO.nvertices(P), pad_to=2, resolution_maxlen=2)
    @test FEA.supports(bettispec, enc23)
    betti_v = TO.transform(bettispec, enc23; threaded=false)
    @test length(betti_v) == TO.nfeatures(bettispec)
    betti_res = TO.resolve(enc23; kind=:projective, opts=OPT.ResolutionOptions(maxlen=2), cache=:auto)
    betti_tbl = DF.betti_table(betti_res.res; pad_to=bettispec.pad_to)
    betti_expected = Float64[betti_tbl[a + 1, p] for a in 0:bettispec.pad_to for p in 1:bettispec.nvertices]
    @test betti_v == betti_expected
    @test TO.transform(bettispec, betti_res; threaded=false) == betti_expected

    bassspec = TO.BassTableSpec(nvertices=TO.nvertices(P), pad_to=2, resolution_maxlen=2)
    @test FEA.supports(bassspec, enc23)
    bass_v = TO.transform(bassspec, enc23; threaded=false)
    @test length(bass_v) == TO.nfeatures(bassspec)
    bass_res = TO.resolve(enc23; kind=:injective, opts=OPT.ResolutionOptions(maxlen=2), cache=:auto)
    bass_tbl = DF.bass_table(bass_res.res; pad_to=bassspec.pad_to)
    bass_expected = Float64[bass_tbl[b + 1, p] for b in 0:bassspec.pad_to for p in 1:bassspec.nvertices]
    @test bass_v == bass_expected
    @test TO.transform(bassspec, bass_res; threaded=false) == bass_expected
    @test_throws ArgumentError TO.transform(bettispec, bass_res; threaded=false)
    @test_throws ArgumentError TO.transform(bassspec, betti_res; threaded=false)

    betti_support_spec = TO.BettiSupportMeasuresSpec(pad_to=2, resolution_maxlen=2)
    @test FEA.supports(betti_support_spec, enc23)
    betti_support_v = TO.transform(betti_support_spec, enc23; opts=opts_inv, threaded=false)
    betti_support_nt = Inv.betti_support_measures(betti_tbl, enc23.pi, opts_inv)
    betti_support_expected = Float64[
        betti_support_nt.support_by_degree[a + 1] for a in 0:betti_support_spec.pad_to
    ]
    append!(betti_support_expected, Float64[
        betti_support_nt.mass_by_degree[a + 1] for a in 0:betti_support_spec.pad_to
    ])
    append!(betti_support_expected, Float64[
        betti_support_nt.support_union,
        betti_support_nt.support_total,
        betti_support_nt.mass_total,
    ])
    @test betti_support_v == betti_support_expected
    @test TO.transform(betti_support_spec, betti_res; opts=opts_inv, threaded=false) == betti_support_expected

    bass_support_spec = TO.BassSupportMeasuresSpec(pad_to=2, resolution_maxlen=2)
    @test FEA.supports(bass_support_spec, enc23)
    bass_support_v = TO.transform(bass_support_spec, enc23; opts=opts_inv, threaded=false)
    bass_support_nt = Inv.bass_support_measures(bass_tbl, enc23.pi, opts_inv)
    bass_support_expected = Float64[
        bass_support_nt.support_by_degree[b + 1] for b in 0:bass_support_spec.pad_to
    ]
    append!(bass_support_expected, Float64[
        bass_support_nt.mass_by_degree[b + 1] for b in 0:bass_support_spec.pad_to
    ])
    append!(bass_support_expected, Float64[
        bass_support_nt.support_union,
        bass_support_nt.support_total,
        bass_support_nt.mass_total,
    ])
    @test bass_support_v == bass_support_expected
    @test TO.transform(bass_support_spec, bass_res; opts=opts_inv, threaded=false) == bass_support_expected
    @test_throws ArgumentError TO.transform(betti_support_spec, bass_res; opts=opts_inv, threaded=false)
    @test_throws ArgumentError TO.transform(bass_support_spec, betti_res; opts=opts_inv, threaded=false)

    betti_pair = TO.CompositeSpec((bettispec, bettispec), namespacing=true)
    @test FEA.supports(betti_pair, betti_res)
    @test TO.transform(betti_pair, betti_res; threaded=false) == vcat(betti_expected, betti_expected)
    @test !FEA.supports(TO.CompositeSpec((bettispec, bassspec), namespacing=true), enc23)

    proj_pair = TO.CompositeSpec((bettispec, betti_support_spec), namespacing=true)
    @test FEA.supports(proj_pair, enc23)
    proj_pair_cache = TO.build_cache(enc23, proj_pair; opts=opts_inv, threaded=false)
    @test proj_pair_cache isa FEA.ResolutionMeasureInvariantCache
    @test TO.transform(proj_pair, proj_pair_cache; opts=opts_inv, threaded=false) ==
          vcat(betti_expected, betti_support_expected)

    cspec = TO.CompositeSpec((lspec, pispec), namespacing=true)
    cv = TO.transform(cspec, enc23; opts=opts_inv, threaded=false)
    @test length(cv) == TO.nfeatures(cspec)
    @test length(TO.feature_names(cspec)) == TO.nfeatures(cspec)
    cax = TO.feature_axes(cspec)
    @test cax.namespacing == cspec.namespacing
    @test length(cax.components) == 2
    @test cax.components[1].start == 1
    @test cax.components[2].start == TO.nfeatures(lspec) + 1

    # Composite fusion for lower-cost families (Euler + rank grid).
    csmall = TO.CompositeSpec((espec, rspec), namespacing=true)
    csmall_v = TO.transform(csmall, enc23; opts=opts_inv, threaded=false)
    @test isapprox(csmall_v, vcat(ev, rv); atol=1e-12, rtol=0.0)

    # Invariant cache protocol: build once, transform many.
    ccache = TO.build_cache(enc23, cspec; opts=opts_inv, threaded=false)
    @test ccache isa FEA.EncodingInvariantCache
    cv_cached = TO.transform(cspec, ccache; threaded=false)
    @test isapprox(cv_cached, cv; atol=1e-12, rtol=0.0)
    cstats = TO.cache_stats(ccache)
    @test cstats.kind == :encoding
    @test cstats.n_slice_plans == 1
    @test cstats.n_fibered == 0

    # Explicit cache-level control.
    ccache_slice = TO.build_cache(enc23, cspec; opts=opts_inv, threaded=false, level=:slice)
    cstats_slice = TO.cache_stats(ccache_slice)
    @test cstats_slice.n_slice_plans == 1
    @test cstats_slice.n_fibered == 0
    cv_slice = TO.transform(cspec, ccache_slice; threaded=false)
    @test isapprox(cv_slice, cv; atol=1e-12, rtol=0.0)
    lv_slice = TO.transform(lspec, ccache_slice; threaded=false)
    piv_slice = TO.transform(pispec, ccache_slice; threaded=false)
    @test isapprox(cv_slice, vcat(lv_slice, piv_slice); atol=1e-12, rtol=0.0)

    ccache_fibered = TO.build_cache(enc23, cspec; opts=opts_inv, threaded=false, level=:fibered)
    cstats_fibered = TO.cache_stats(ccache_fibered)
    @test cstats_fibered.n_fibered >= 1
    @test cstats_fibered.n_slice_plans == 0
    cv_fibered = TO.transform(cspec, ccache_fibered; threaded=false)
    @test length(cv_fibered) == length(cv)
    @test all(isfinite, cv_fibered)

    csmall_cache = TO.build_cache(enc23, csmall; opts=opts_inv, threaded=false)
    csmall_cached = TO.transform(csmall, csmall_cache; threaded=false)
    @test isapprox(csmall_cached, csmall_v; atol=1e-12, rtol=0.0)

    lcache = TO.build_cache(enc23, lspec; opts=opts_inv, threaded=false)
    lv_cached = TO.transform(lspec, lcache; threaded=false)
    @test isapprox(lv_cached, lv; atol=1e-12, rtol=0.0)

    rhcache = TO.build_cache(enc23, rhspec; opts=opts_inv, threaded=false)
    @test rhcache isa FEA.RestrictedHilbertInvariantCache
    @test TO.transform(rhspec, rhcache; threaded=false) == rhv

    dcache = TO.build_cache(enc23, dspec; opts=opts_inv, threaded=false)
    dv_cached = TO.transform(dspec, dcache; threaded=false)
    @test isapprox(dv_cached, dv; atol=1e-12, rtol=0.0)

    mlcache = TO.build_cache(enc23, mlspec; opts=opts_inv, threaded=false)
    mlv_cached = TO.transform(mlspec, mlcache; threaded=false)
    @test isapprox(mlv_cached, mlv; atol=1e-12, rtol=0.0)

    mppcache = TO.build_cache(enc23, mppspec; opts=opts_inv, threaded=false, level=:fibered)
    @test TO.cache_stats(mppcache).n_fibered >= 1
    mppv_cached = TO.transform(mppspec, mppcache; threaded=false)
    @test isapprox(mppv_cached, mppv; atol=1e-12, rtol=0.0)
    @test TO.cache_stats(mppcache).n_mpp_decompositions == 1

    betti_cache = TO.build_cache(enc23, bettispec; threaded=false)
    @test TO.cache_stats(betti_cache).kind == :resolution
    @test TO.cache_stats(betti_cache).resolution_kind == :projective
    @test TO.transform(bettispec, betti_cache; threaded=false) == betti_expected

    bass_cache = TO.build_cache(enc23, bassspec; threaded=false)
    @test TO.cache_stats(bass_cache).kind == :resolution
    @test TO.cache_stats(bass_cache).resolution_kind == :injective
    @test TO.transform(bassspec, bass_cache; threaded=false) == bass_expected

    mcache = TO.build_cache(M23; opts=opts_inv, threaded=false)
    @test mcache isa FEA.ModuleInvariantCache
    @test TO.cache_stats(mcache).kind == :module
    rv_cached = TO.transform(rspec, mcache; threaded=false)
    @test isapprox(rv_cached, rv; atol=1e-12, rtol=0.0)
    @test_throws ArgumentError TO.build_cache(M23, lspec; opts=opts_inv, threaded=false)
    @test_throws ArgumentError TO.build_cache(enc23, lspec; opts=opts_inv, threaded=false, level=:bad_level)

    eh_cache = TO.build_cache(enc23, espec; opts=opts_inv, threaded=false)
    @test eh_cache isa FEA.RestrictedHilbertInvariantCache
    @test TO.cache_stats(eh_cache).kind == :restricted_hilbert
    ev_cached = TO.transform(espec, eh_cache; threaded=false)
    @test isapprox(ev_cached, ev; atol=1e-12, rtol=0.0)

    # Rank-only composite should fuse on module cache as well.
    rdouble = TO.CompositeSpec((rspec, rspec), namespacing=true)
    rdouble_v = TO.transform(rdouble, M23; opts=opts_inv, threaded=false)
    @test isapprox(rdouble_v, vcat(rv, rv); atol=1e-12, rtol=0.0)
    rdouble_cache = TO.build_cache(M23, rdouble; opts=opts_inv, threaded=false)
    rdouble_cached = TO.transform(rdouble, rdouble_cache; threaded=false)
    @test isapprox(rdouble_cached, rdouble_v; atol=1e-12, rtol=0.0)

    # Signed-barcode cache path should materialize rank-query cache for Zn backends.
    F = TO.FlangeZn.face(2, [1, 2])
    flats = [TO.FlangeZn.IndFlat(F, [0, 0]; id=:F)]
    injectives = [TO.FlangeZn.IndInj(F, [0, 0]; id=:E)]
    FG = FZ.Flange{CM.QQ}(2, flats, injectives, reshape([CM.QQ(1)], 1, 1); field=field)
    enc_zn = TamerOp.encode(FG; backend=:zn)
    sbispec_zn = TO.SignedBarcodeImageSpec(
        xs=collect(-1.0:0.5:1.0),
        ys=collect(-1.0:0.5:1.0),
        sigma=0.35,
        strict=false,
    )
    @test FEA.supports(sbispec_zn, enc_zn)
    opts_zn = TO.InvariantOptions(strict=false)
    sbcache = TO.build_cache(enc_zn, sbispec_zn; opts=opts_zn, threaded=false, level=:all)
    @test TO.cache_stats(sbcache).has_rank_query
    sbvals = TO.transform(sbispec_zn, sbcache; opts=opts_zn, threaded=false)
    @test length(sbvals) == TO.nfeatures(sbispec_zn)

    rectspec = TO.RectangleSignedBarcodeTopKSpec(ndims=2, k=2, coords=:indices, strict=false)
    @test FEA.supports(rectspec, enc_zn)
    rectv = TO.transform(rectspec, enc_zn; opts=opts_zn, threaded=false)
    sb_zn = Inv.rectangle_signed_barcode(enc_zn.M, enc_zn.pi, opts_zn; threads=false)
    rect_expected = FEA._rect_signed_barcode_topk_vector(sb_zn, rectspec.ndims, rectspec.k, rectspec.coords)
    @test rectv == rect_expected
    rectcache = TO.build_cache(enc_zn, rectspec; opts=opts_zn, threaded=false, level=:all)
    @test TO.cache_stats(rectcache).has_rank_query
    @test TO.transform(rectspec, rectcache; opts=opts_zn, threaded=false) == rect_expected

    sb_direct = SM.RectSignedBarcode(
        (Int[0, 1, 2], Int[0, 1, 2]),
        [SM.Rect{2}((1, 1), (2, 2)), SM.Rect{2}((2, 1), (2, 2))],
        Int[3, -1],
    )
    rectspec_direct = TO.RectangleSignedBarcodeTopKSpec(ndims=2, k=2, coords=:values)
    @test TO.transform(rectspec_direct, sb_direct; threaded=false) ==
          [2.0, 1.0, 3.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, 1.0, 0.0, 1.0, 1.0]

    mdspec = TO.MatchingDistanceBankSpec([enc3]; reference_names=[:M3], method=:approx, n_dirs=8, n_offsets=4)
    @test FEA.supports(mdspec, enc23)
    mdv = TO.transform(mdspec, enc23; opts=opts_inv, threaded=false)
    @test length(mdv) == TO.nfeatures(mdspec)
    @test isapprox(mdv[1], TO.matching_distance(enc23, enc3; method=:approx, opts=opts_inv, n_dirs=8, n_offsets=4); atol=1e-12, rtol=0.0)

    bserial = TO.BatchOptions(threaded=false, backend=:serial, deterministic=true)
    fs = TO.featurize(samples, lspec;
                      opts=opts_inv,
                      batch=bserial,
                      idfun=s -> string(TO.nvertices(s.P)),
                      labelfun=s -> Inv.rank_map(s.M, r3, r3))
    @test size(fs.X, 1) == length(samples)
    @test size(fs.X, 2) == TO.nfeatures(lspec)
    @test length(fs.names) == TO.nfeatures(lspec)
    @test length(fs.ids) == length(samples)
    @test haskey(fs.meta, :labels)
    @test fs.meta.labels == [1, 1]
    @test haskey(fs.meta, :feature_axes)
    @test TO.feature_axes(fs) == TO.feature_axes(lspec)

    fs_batch_serial = TO.featurize(samples, lspec; opts=opts_inv, batch=bserial)
    @test size(fs_batch_serial.X) == size(fs.X)
    @test fs_batch_serial.X == fs.X

    bthreads = TO.BatchOptions(threaded=true, backend=:threads, deterministic=true, chunk_size=1)
    fs_batch_threads = TO.featurize(samples, lspec; opts=opts_inv, batch=bthreads)
    @test fs_batch_threads.X == fs_batch_serial.X
    @test fs_batch_threads.names == fs_batch_serial.names
    @test fs_batch_threads.ids == fs_batch_serial.ids

    bfolds = TO.BatchOptions(threaded=true, backend=:folds, deterministic=true)
    fs_batch_folds = TO.featurize(samples, lspec; opts=opts_inv, batch=bfolds)
    @test fs_batch_folds.X == fs_batch_serial.X

    bprogress = TO.BatchOptions(threaded=false, backend=:serial, progress=true, deterministic=true)
    fs_batch_progress = TO.featurize(samples, lspec; opts=opts_inv, batch=bprogress)
    @test fs_batch_progress.X == fs_batch_serial.X
    @test fs_batch_progress.meta.batch.progress

    @test_throws ArgumentError TO.BatchOptions(threaded=true, backend=:bad_backend)

    fs_bt = TO.batch_transform(samples, lspec; opts=opts_inv, batch=bthreads)
    @test fs_bt.X == fs_batch_threads.X
    @test fs_bt.names == fs_batch_threads.names
    @test fs_bt.ids == fs_batch_threads.ids

    # Repeated deterministic runs should preserve exact row order/content.
    fs_det_ref = TO.featurize(samples, lspec; opts=opts_inv, batch=bthreads, cache=:auto)
    for _ in 1:3
        fs_det = TO.featurize(samples, lspec; opts=opts_inv, batch=bthreads, cache=:auto)
        @test fs_det.ids == fs_det_ref.ids
        @test fs_det.names == fs_det_ref.names
        @test fs_det.X == fs_det_ref.X
    end

    # Cache policy parity in threaded batch mode.
    session_cache = CM.SessionCache()
    fs_cache_auto = TO.featurize(samples, lspec; opts=opts_inv, batch=bthreads, cache=:auto)
    fs_cache_explicit_1 = TO.featurize(samples, lspec; opts=opts_inv, batch=bthreads, cache=session_cache)
    fs_cache_explicit_2 = TO.featurize(samples, lspec; opts=opts_inv, batch=bthreads, cache=session_cache)
    @test fs_cache_explicit_1.X == fs_cache_auto.X
    @test fs_cache_explicit_2.X == fs_cache_auto.X
    @test fs_cache_explicit_1.ids == fs_cache_auto.ids
    @test fs_cache_explicit_1.names == fs_cache_auto.names

    Xbuf = Matrix{Float64}(undef, size(fs_batch_threads.X, 1), size(fs_batch_threads.X, 2))
    fs_bang = TO.batch_transform!(Xbuf, samples, lspec; opts=opts_inv, batch=bthreads)
    @test fs_bang.X === Xbuf
    @test fs_bang.X == fs_batch_threads.X

    Xm = Matrix{Union{Missing,Float64}}(undef, length(samples), TO.nfeatures(lspec))
    fs_bang_missing = TO.batch_transform!(Xm, samples, lspec; opts=opts_inv, on_unsupported=:missing, batch=bserial)
    @test fs_bang_missing.X === Xm
    @test all(!ismissing, fs_bang_missing.X)

    buff = zeros(Float64, TO.nfeatures(lspec))
    TO.transform!(buff, lspec, enc23; opts=opts_inv, threaded=false)
    @test isapprox(buff, lv; atol=1e-12, rtol=0.0)

    @test FEA.asrowmajor(fs) === fs.X
    @test size(FEA.ascolmajor(fs)) == (size(fs.X, 2), size(fs.X, 1))
    @test FEA.feature_table(fs; format=:wide) isa FEA.FeatureSetWideTable
    @test FEA.feature_table(fs; format=:long) isa FEA.FeatureSetLongTable
    @test_throws ArgumentError FEA.feature_table(fs; format=:bad)
    @test TO.Serialization.TAMER_FEATURE_SCHEMA_VERSION == v"0.2.0"

    md = FEA.feature_metadata(fs; format=:wide)
    @test md["kind"] == "features"
    @test md["schema_version"] == "0.2.0"
    @test md["n_samples"] == size(fs.X, 1)
    @test md["n_features"] == size(fs.X, 2)
    @test md["layout"] == "rows=samples, cols=features"
    @test md["format"] == "wide"
    @test haskey(md, "feature_axes")
    @test string(md["feature_axes"]["aggregate"]) == string(lspec.aggregate)
    hdr = TO.Serialization.feature_schema_header(format=:wide)
    @test hdr["kind"] == "features"
    @test hdr["schema_version"] == "0.2.0"
    @test TO.Serialization.validate_feature_metadata_schema(md)

    mpath = FEA.default_feature_metadata_path(joinpath(mktempdir(), "features.arrow"))
    @test endswith(mpath, "features.arrow.meta.json")

    tmpd = mktempdir()
    meta_path = joinpath(tmpd, "features.meta.json")
    FEA.save_metadata_json(meta_path, md)
    md2 = FEA.load_metadata_json(meta_path; validate_feature_schema=true)
    @test string(md2["schema_version"]) == "0.2.0"
    @test Int(md2["n_features"]) == size(fs.X, 2)
    md_typed = FEA.load_metadata_json(meta_path; validate_feature_schema=true, typed=true)
    @test md_typed.spec isa TO.LandscapeSpec
    @test md_typed.opts isa TO.InvariantOptions
    @test md_typed.spec.kmax == lspec.kmax
    @test md_typed.spec.aggregate == lspec.aggregate
    @test md_typed.opts.axes_policy == opts_inv.axes_policy
    @test md_typed.opts.max_axis_len == opts_inv.max_axis_len
    @test md_typed.opts.strict == opts_inv.strict

    # Explicit typed round-trip from metadata fragments.
    lspec2 = FEA.spec_from_metadata(md["spec"])
    @test lspec2 isa TO.LandscapeSpec
    @test TO.feature_names(lspec2) == TO.feature_names(lspec)
    @test TO.nfeatures(lspec2) == TO.nfeatures(lspec)

    opts2 = FEA.invariant_options_from_metadata(md["opts"])
    @test opts2 isa TO.InvariantOptions
    @test opts2.axes_policy == opts_inv.axes_policy
    @test opts2.max_axis_len == opts_inv.max_axis_len
    @test opts2.strict == opts_inv.strict

    # Nested spec round-trip (composite + nested specs).
    mdc = FEA.feature_metadata(TO.FeatureSet(fs.X, fs.names, fs.ids, (spec=cspec, opts=opts_inv)); format=:wide)
    cspec2 = FEA.spec_from_metadata(mdc["spec"])
    @test cspec2 isa TO.CompositeSpec
    @test length(cspec2.specs) == length(cspec.specs)
    @test TO.feature_names(cspec2) == TO.feature_names(cspec)

    mdd = FEA.feature_metadata(TO.FeatureSet(fs.X, fs.names, fs.ids, (spec=dspec, opts=opts_inv)); format=:wide)
    dspec2 = FEA.spec_from_metadata(mdd["spec"])
    @test dspec2 isa TO.ProjectedDistancesSpec
    @test dspec2.reference_names == dspec.reference_names
    @test dspec2.dist == dspec.dist
    @test haskey(mdd["spec"]["fields"], "reference_ids")
    @test collect(String.(mdd["spec"]["fields"]["reference_ids"])) == ["M3"]

    resolve_ref = id -> id == "M3" ? enc3 : nothing
    dspec3 = FEA.load_spec_with_resolver(mdd["spec"], resolve_ref)
    @test dspec3 isa TO.ProjectedDistancesSpec
    @test length(dspec3.references) == 1
    @test dspec3.references[1] === enc3
    dv3 = TO.transform(dspec3, enc23; opts=opts_inv, threaded=false)
    @test isapprox(dv3, dv; atol=1e-12, rtol=0.0)
    @test_throws ArgumentError FEA.load_spec_with_resolver(mdd["spec"], _ -> nothing; require_all=true)

    mdm = FEA.feature_metadata(TO.FeatureSet(zeros(Float64, 1, TO.nfeatures(mdspec)),
                                            TO.feature_names(mdspec),
                                            ["sample"],
                                            (spec=mdspec, opts=opts_inv)); format=:wide)
    mdspec2 = FEA.load_spec_with_resolver(mdm["spec"], resolve_ref)
    @test mdspec2 isa TO.MatchingDistanceBankSpec
    @test mdspec2.references[1] === enc3
    @test mdspec2.reference_names == mdspec.reference_names

    pmm = FEA.feature_metadata(TO.FeatureSet(zeros(Float64, 1, TO.nfeatures(pmspec)),
                                            TO.feature_names(pmspec),
                                            ["sample"],
                                            (spec=pmspec, opts=opts_inv)); format=:wide)
    pmspec2 = FEA.spec_from_metadata(pmm["spec"])
    @test pmspec2 isa TO.PointSignedMeasureSpec
    @test TO.feature_names(pmspec2) == TO.feature_names(pmspec)

    em = FEA.feature_metadata(TO.FeatureSet(zeros(Float64, 1, TO.nfeatures(eulerspec)),
                                           TO.feature_names(eulerspec),
                                           ["sample"],
                                           (spec=eulerspec, opts=opts_inv)); format=:wide)
    eulerspec2 = FEA.spec_from_metadata(em["spec"])
    @test eulerspec2 isa TO.EulerSignedMeasureSpec
    @test TO.feature_names(eulerspec2) == TO.feature_names(eulerspec)

    rm = FEA.feature_metadata(TO.FeatureSet(zeros(Float64, 1, TO.nfeatures(rectspec)),
                                           TO.feature_names(rectspec),
                                           ["sample"],
                                           (spec=rectspec, opts=opts_zn)); format=:wide)
    rectspec2 = FEA.spec_from_metadata(rm["spec"])
    @test rectspec2 isa TO.RectangleSignedBarcodeTopKSpec
    @test TO.feature_names(rectspec2) == TO.feature_names(rectspec)

    extended_spec = TO.CompositeSpec((
        rhspec,
        topkspec,
        bsumspec,
        pmspec,
        eulerspec,
        rectspec,
        mdspec,
        mlspec,
        mppspec,
        mpphistspec,
        bettispec,
        bassspec,
        betti_support_spec,
        bass_support_spec,
    ), namespacing=true)
    mdx = FEA.feature_metadata(
        TO.FeatureSet(zeros(Float64, 1, TO.nfeatures(extended_spec)),
                      TO.feature_names(extended_spec),
                      ["sample"],
                      (spec=extended_spec, opts=opts_inv));
        format=:wide,
    )
    extended_spec2 = FEA.spec_from_metadata(mdx["spec"])
    @test extended_spec2 isa TO.CompositeSpec
    @test length(extended_spec2.specs) == 14
    @test extended_spec2.specs[1] isa TO.RestrictedHilbertSpec
    @test extended_spec2.specs[2] isa TO.BarcodeTopKSpec
    @test extended_spec2.specs[3] isa TO.BarcodeSummarySpec
    @test extended_spec2.specs[4] isa TO.PointSignedMeasureSpec
    @test extended_spec2.specs[5] isa TO.EulerSignedMeasureSpec
    @test extended_spec2.specs[6] isa TO.RectangleSignedBarcodeTopKSpec
    @test extended_spec2.specs[7] isa TO.MatchingDistanceBankSpec
    @test extended_spec2.specs[8] isa TO.MPLandscapeSpec
    @test extended_spec2.specs[9] isa TO.MPPImageSpec
    @test extended_spec2.specs[10] isa TO.MPPDecompositionHistogramSpec
    @test extended_spec2.specs[11] isa TO.BettiTableSpec
    @test extended_spec2.specs[12] isa TO.BassTableSpec
    @test extended_spec2.specs[13] isa TO.BettiSupportMeasuresSpec
    @test extended_spec2.specs[14] isa TO.BassSupportMeasuresSpec
    @test TO.feature_names(extended_spec2) == TO.feature_names(extended_spec)

    proj_meta_path = joinpath(tmpd, "projected_features.meta.json")
    FEA.save_metadata_json(proj_meta_path, mdd)
    proj_typed = FEA.load_metadata_json(proj_meta_path;
                                       validate_feature_schema=true,
                                       typed=true,
                                       resolve_ref=resolve_ref,
                                       require_resolved_refs=true)
    @test proj_typed.spec isa TO.ProjectedDistancesSpec
    @test proj_typed.spec.references[1] === enc3

    # Internal wide/long reconstruction helpers (no optional deps required).
    cols_wide = (id=["s1", "s2"], f1=[1.0, 2.0], f2=[3.0, 4.0])
    fs_wide = FEA._featureset_from_columntable(cols_wide; format=:wide)
    @test fs_wide.ids == ["s1", "s2"]
    @test fs_wide.names == [:f1, :f2]
    @test all(fs_wide.X .== [1.0 3.0; 2.0 4.0])

    cols_long = (id=["s1", "s1", "s2", "s2"],
                 feature=[:f1, :f2, :f1, :f2],
                 value=[1.0, 3.0, 2.0, 4.0])
    fs_long = FEA._featureset_from_columntable(cols_long; format=:long)
    @test fs_long.ids == ["s1", "s2"]
    @test fs_long.names == [:f1, :f2]
    @test all(fs_long.X .== [1.0 3.0; 2.0 4.0])

    # Arrow IO (optional extension)
    arrow_path = joinpath(tmpd, "features.arrow")
    if Base.find_package("Arrow") === nothing
        @test_throws ArgumentError TO.save_features_arrow(arrow_path, fs)
        @test_throws ArgumentError TO.load_features_arrow(arrow_path)
        @test_throws ArgumentError TO.save_features(arrow_path, fs; format=:arrow)
        @test_throws ArgumentError TO.load_features(arrow_path; format=:arrow)
        @test_throws ArgumentError TO.save_features(arrow_path, fs; format=:auto)
        @test_throws ArgumentError TO.load_features(arrow_path; format=:auto)
    else
        @eval using Arrow
        TO.save_features_arrow(arrow_path, fs; format=:wide)
        @test isfile(arrow_path)
        @test isfile(FEA.default_feature_metadata_path(arrow_path))
        fs_arrow = TO.load_features_arrow(arrow_path; format=:wide)
        @test fs_arrow.names == fs.names
        @test fs_arrow.ids == fs.ids
        @test size(fs_arrow.X) == size(fs.X)
        @test all(fs_arrow.X .== fs.X)

        TO.save_features(arrow_path, fs; format=:arrow, mode=:wide, metadata=true)
        fs_arrow2 = TO.load_features(arrow_path; format=:arrow, mode=:wide)
        @test fs_arrow2.names == fs.names
        @test fs_arrow2.ids == fs.ids
        @test size(fs_arrow2.X) == size(fs.X)
        @test all(fs_arrow2.X .== fs.X)

        TO.save_features(arrow_path, fs; format=:auto, mode=:wide, metadata=true)
        fs_arrow3 = TO.load_features(arrow_path; format=:auto, mode=:wide)
        @test fs_arrow3.names == fs.names
        @test fs_arrow3.ids == fs.ids
        @test size(fs_arrow3.X) == size(fs.X)
        @test all(fs_arrow3.X .== fs.X)

        @test_throws ArgumentError TO.save_features(arrow_path, fs; format=:arrow, layout=:features_by_samples)
        @test_throws ArgumentError TO.load_features(arrow_path; format=:arrow, layout=:features_by_samples)
    end

    # Parquet IO (optional extension)
    parq_path = joinpath(tmpd, "features.parquet")
    if Base.find_package("Parquet2") === nothing
        @test_throws ArgumentError TO.save_features_parquet(parq_path, fs)
        @test_throws ArgumentError TO.load_features_parquet(parq_path)
        @test_throws ArgumentError TO.save_features(parq_path, fs; format=:parquet)
        @test_throws ArgumentError TO.load_features(parq_path; format=:parquet)
        @test_throws ArgumentError TO.save_features(parq_path, fs; format=:auto)
        @test_throws ArgumentError TO.load_features(parq_path; format=:auto)
    else
        @eval using Parquet2
        TO.save_features_parquet(parq_path, fs; format=:long)
        @test isfile(parq_path)
        @test isfile(FEA.default_feature_metadata_path(parq_path))
        fs_parq = TO.load_features_parquet(parq_path; format=:long)
        @test fs_parq.names == fs.names
        @test fs_parq.ids == fs.ids
        @test size(fs_parq.X) == size(fs.X)
        @test all(fs_parq.X .== fs.X)

        TO.save_features(parq_path, fs; format=:parquet, mode=:long, metadata=true)
        fs_parq2 = TO.load_features(parq_path; format=:parquet, mode=:long)
        @test fs_parq2.names == fs.names
        @test fs_parq2.ids == fs.ids
        @test size(fs_parq2.X) == size(fs.X)
        @test all(fs_parq2.X .== fs.X)

        TO.save_features(parq_path, fs; format=:auto, mode=:long, metadata=true)
        fs_parq3 = TO.load_features(parq_path; format=:auto, mode=:long)
        @test fs_parq3.names == fs.names
        @test fs_parq3.ids == fs.ids
        @test size(fs_parq3.X) == size(fs.X)
        @test all(fs_parq3.X .== fs.X)

        @test_throws ArgumentError TO.save_features(parq_path, fs; format=:parquet, layout=:features_by_samples)
        @test_throws ArgumentError TO.load_features(parq_path; format=:parquet, layout=:features_by_samples)
    end

    # NPZ is extension-backed; CSV save has a built-in fallback, while CSV load
    # still requires the CSV extension.
    npz_path = joinpath(tmpd, "features.npz")
    csv_path = joinpath(tmpd, "features.csv")
    if Base.find_package("NPZ") === nothing
        @test_throws ArgumentError TO.save_features_npz(npz_path, fs)
        @test_throws ArgumentError TO.load_features_npz(npz_path)
        @test_throws ArgumentError TO.save_features(npz_path, fs; format=:npz)
        @test_throws ArgumentError TO.load_features(npz_path; format=:npz)
    else
        @eval using NPZ
        TO.save_features_npz(npz_path, fs; format=:wide, layout=:samples_by_features)
        @test isfile(npz_path)
        @test isfile(FEA.default_feature_metadata_path(npz_path))
        fs_npz = TO.load_features_npz(npz_path)
        @test fs_npz.names == fs.names
        @test fs_npz.ids == fs.ids
        @test size(fs_npz.X) == size(fs.X)
        @test all(fs_npz.X .== fs.X)
        md_npz = FEA.load_metadata_json(FEA.default_feature_metadata_path(npz_path); validate_feature_schema=true)
        @test md_npz["kind"] == "features"
        @test md_npz["schema_version"] == string(TO.Serialization.TAMER_FEATURE_SCHEMA_VERSION)

        # Generic interop entrypoints should route to NPZ extension.
        TO.save_features(npz_path, fs; format=:auto, mode=:wide, layout=:samples_by_features, metadata=true)
        fs_npz2 = TO.load_features(npz_path; format=:auto, mode=:wide)
        @test fs_npz2.names == fs.names
        @test fs_npz2.ids == fs.ids
        @test size(fs_npz2.X) == size(fs.X)
        @test all(fs_npz2.X .== fs.X)
        @test haskey(fs_npz2.meta, :metadata)
        @test fs_npz2.meta.metadata["schema_version"] == string(TO.Serialization.TAMER_FEATURE_SCHEMA_VERSION)

        # Optional transposed storage layout round-trip.
        TO.save_features(npz_path, fs; format=:npz, mode=:wide, layout=:features_by_samples, metadata=true)
        fs_npz3 = TO.load_features(npz_path; format=:npz, mode=:wide)
        @test fs_npz3.names == fs.names
        @test fs_npz3.ids == fs.ids
        @test size(fs_npz3.X) == size(fs.X)
        @test all(fs_npz3.X .== fs.X)
    end
    if Base.find_package("CSV") === nothing
        TO.save_features_csv(csv_path, fs)
        @test isfile(csv_path)
        @test isfile(FEA.default_feature_metadata_path(csv_path))
        @test_throws ArgumentError TO.load_features_csv(csv_path)
        TO.save_features(csv_path, fs; format=:csv)
        @test isfile(csv_path)
        @test isfile(FEA.default_feature_metadata_path(csv_path))
        @test_throws ArgumentError TO.load_features(csv_path; format=:csv)
    else
        @eval using CSV
        TO.save_features_csv(csv_path, fs; format=:wide, layout=:samples_by_features)
        @test isfile(csv_path)
        @test isfile(FEA.default_feature_metadata_path(csv_path))
        fs_csv = TO.load_features_csv(csv_path; format=:wide)
        @test fs_csv.names == fs.names
        @test fs_csv.ids == fs.ids
        @test size(fs_csv.X) == size(fs.X)
        @test all(fs_csv.X .== fs.X)

        TO.save_features(csv_path, fs; format=:auto, mode=:wide, layout=:samples_by_features, metadata=true)
        fs_csv2 = TO.load_features(csv_path; format=:auto, mode=:wide)
        @test fs_csv2.names == fs.names
        @test fs_csv2.ids == fs.ids
        @test size(fs_csv2.X) == size(fs.X)
        @test all(fs_csv2.X .== fs.X)

        # Deterministic feature order in wide mode should match FeatureSet.names.
        header = split(chomp(readline(csv_path)), ',')
        @test header == ["id"; String.(fs.names)]

        # Long mode write/load support.
        TO.save_features_csv(csv_path, fs; format=:long, layout=:samples_by_features)
        fs_csv_long = TO.load_features_csv(csv_path; format=:long)
        @test fs_csv_long.names == fs.names
        @test fs_csv_long.ids == fs.ids
        @test size(fs_csv_long.X) == size(fs.X)
        @test all(fs_csv_long.X .== fs.X)
    end
    @test_throws ArgumentError TO.save_features(joinpath(tmpd, "features.unknown"), fs; format=:auto)
    @test_throws ArgumentError TO.load_features(joinpath(tmpd, "features.unknown"); format=:auto)

    mixed = Any[enc23, M23, enc3]
    @test_throws ArgumentError TO.featurize(mixed, lspec; opts=opts_inv, batch=bserial, on_unsupported=:error)

    fs_skip = TO.featurize(mixed, lspec; opts=opts_inv, batch=bserial, on_unsupported=:skip)
    @test size(fs_skip.X, 1) == 2
    @test size(fs_skip.X, 2) == TO.nfeatures(lspec)
    @test fs_skip.meta.skipped_indices == [2]

    fs_missing = TO.featurize(mixed, lspec; opts=opts_inv, batch=bserial, on_unsupported=:missing)
    @test size(fs_missing.X, 1) == 3
    @test size(fs_missing.X, 2) == TO.nfeatures(lspec)
    @test all(ismissing, view(fs_missing.X, 2, :))
    @test !any(ismissing, view(fs_missing.X, 1, :))
    @test !any(ismissing, view(fs_missing.X, 3, :))

    Xmixed = Matrix{Float64}(undef, 3, TO.nfeatures(lspec))
    @test_throws ArgumentError TO.batch_transform!(Xmixed, mixed, lspec; opts=opts_inv, batch=bserial, on_unsupported=:missing)
    Xmixed_m = Matrix{Union{Missing,Float64}}(undef, 3, TO.nfeatures(lspec))
    fs_missing_bang = TO.batch_transform!(Xmixed_m, mixed, lspec; opts=opts_inv, batch=bserial, on_unsupported=:missing)
    @test all(ismissing, view(fs_missing_bang.X, 2, :))

    # Unsupported-sample policy parity across deterministic backends.
    fs_skip_serial = TO.featurize(mixed, lspec; opts=opts_inv, batch=bserial, on_unsupported=:skip)
    fs_skip_threads = TO.featurize(mixed, lspec; opts=opts_inv, batch=bthreads, on_unsupported=:skip)
    @test fs_skip_threads.X == fs_skip_serial.X
    @test fs_skip_threads.ids == fs_skip_serial.ids
    @test fs_skip_threads.names == fs_skip_serial.names

    fs_missing_serial = TO.featurize(mixed, lspec; opts=opts_inv, batch=bserial, on_unsupported=:missing)
    fs_missing_threads = TO.featurize(mixed, lspec; opts=opts_inv, batch=bthreads, on_unsupported=:missing)
    @test isequal(fs_missing_threads.X, fs_missing_serial.X)
    @test fs_missing_threads.ids == fs_missing_serial.ids
    @test fs_missing_threads.names == fs_missing_serial.names

    if Base.find_package("Folds") !== nothing
        fs_skip_folds = TO.featurize(mixed, lspec; opts=opts_inv, batch=bfolds, on_unsupported=:skip)
        fs_missing_folds = TO.featurize(mixed, lspec; opts=opts_inv, batch=bfolds, on_unsupported=:missing)
        @test fs_skip_folds.X == fs_skip_serial.X
        @test isequal(fs_missing_folds.X, fs_missing_serial.X)
    end

    if Threads.nthreads() > 1
        fs_thr_1 = TO.featurize(samples, lspec; opts=opts_inv, batch=bthreads, cache=:auto)
        fs_thr_2 = TO.featurize(samples, lspec; opts=opts_inv, batch=bthreads, cache=:auto)
        @test fs_thr_1.X == fs_thr_2.X
        @test fs_thr_1.ids == fs_thr_2.ids
    end

    # All unsupported samples: skip => empty matrix, missing => all-missing rows.
    fs_sb_skip = TO.featurize(samples, sbispec; opts=opts_inv, batch=bserial, on_unsupported=:skip)
    @test size(fs_sb_skip.X, 1) == 0
    @test size(fs_sb_skip.X, 2) == TO.nfeatures(sbispec)

    fs_sb_missing = TO.featurize(samples, sbispec; opts=opts_inv, batch=bserial, on_unsupported=:missing)
    @test size(fs_sb_missing.X, 1) == length(samples)
    @test size(fs_sb_missing.X, 2) == TO.nfeatures(sbispec)
    @test all(ismissing, fs_sb_missing.X)

    # Long-form wrappers for optional Tables.jl integration.
    es = TO.euler_surface(M23, pi, opts_inv)
    est = FEA.euler_surface_table(es; id="m23")
    @test est isa FEA.EulerSurfaceLongTable

    bar = Dict((0.0, 1.0) => 1, (0.5, 1.5) => 1)
    PI = Inv.persistence_image(bar; xgrid=0.0:0.5:1.5, ygrid=0.0:0.5:1.5)
    pit = FEA.persistence_image_table(PI; id="pi")
    @test pit isa FEA.PersistenceImageLongTable

    L = TO.mp_landscape(M23, [Int[r2, r3]]; kmax=2, tgrid=collect(0.0:0.5:3.0))
    lt = FEA.mp_landscape_table(L; id="land")
    @test lt isa FEA.MPLandscapeLongTable

    pm = SM.euler_signed_measure(M23, pi, opts_inv)
    pmt = FEA.point_signed_measure_table(pm; id="pm")
    @test pmt isa FEA.PointSignedMeasureLongTable

    @testset "KernelFunctions and Distances integration" begin
        if Base.find_package("KernelFunctions") === nothing
            @test_throws ArgumentError FEA.mp_landscape_kernel_object()
            @test_throws ArgumentError FEA.projected_kernel_object()
        else
            @eval using KernelFunctions
            chain = Int[r2, r3]
            L23_small = TO.mp_landscape(M23, [chain]; kmax=2, tgrid=collect(0.0:0.5:3.0))
            L3_small = TO.mp_landscape(M3, [chain]; kmax=2, tgrid=collect(0.0:0.5:3.0))

            k_land = FEA.mp_landscape_kernel_object(kind=:gaussian, sigma=1.0, p=2)
            @test isapprox(KernelFunctions.kappa(k_land, L23_small, L23_small), 1.0; atol=1e-12)
            K_land = KernelFunctions.kernelmatrix(k_land, Any[L23_small, L3_small])
            @test size(K_land) == (2, 2)
            @test isapprox(K_land[1, 1], 1.0; atol=1e-12)
            @test K_land[1, 2] <= K_land[1, 1]
            @test length(KernelFunctions.kernelmatrix_diag(k_land, Any[L23_small, L3_small])) == 2

            arr_proj = Inv.projected_arrangement(pi; n_dirs=4, normalize=:L1, threads=false)
            cache23_proj = Inv.projected_barcode_cache(M23, arr_proj; precompute=true)
            cache3_proj = Inv.projected_barcode_cache(M3, arr_proj; precompute=true)
            k_proj = FEA.projected_kernel_object(kind=:bottleneck_gaussian, sigma=1.0, threads=false)
            @test KernelFunctions.kappa(k_proj, cache23_proj, cache23_proj) >= KernelFunctions.kappa(k_proj, cache23_proj, cache3_proj)

            k_pm = FEA.point_signed_measure_kernel_object(sigma=1.0)
            @test isapprox(KernelFunctions.kappa(k_pm, pm, pm),
                           SM.point_signed_measure_kernel(pm, pm; sigma=1.0);
                           atol=1e-12)

            sb_zn = Inv.rectangle_signed_barcode(enc_zn.M, enc_zn.pi, opts_zn)
            k_sb = FEA.rectangle_signed_barcode_kernel_object(kind=:linear, sigma=1.0)
            @test isapprox(KernelFunctions.kappa(k_sb, sb_zn, sb_zn),
                           Inv.rectangle_signed_barcode_kernel(sb_zn, sb_zn; kind=:linear, sigma=1.0);
                           atol=1e-12)
        end

        if Base.find_package("Distances") === nothing
            @test_throws ArgumentError FEA.matching_distance_metric()
            @test_throws ArgumentError FEA.bottleneck_distance_metric()
        else
            @eval using Distances

            m_match = FEA.matching_distance_metric(method=:approx, opts=opts_inv)
            d_match = Distances.evaluate(m_match, enc23, enc3)
            @test isapprox(d_match, TO.matching_distance(enc23, enc3; method=:approx, opts=opts_inv); atol=1e-12)
            D_match = Distances.pairwise(m_match, Any[enc23, enc3])
            @test size(D_match) == (2, 2)
            @test isapprox(D_match[1, 1], 0.0; atol=1e-12)
            @test isapprox(D_match[1, 2], D_match[2, 1]; atol=1e-12)

            chain = Int[r2, r3]
            L23_small = TO.mp_landscape(M23, [chain]; kmax=2, tgrid=collect(0.0:0.5:3.0))
            L3_small = TO.mp_landscape(M3, [chain]; kmax=2, tgrid=collect(0.0:0.5:3.0))
            m_land = FEA.mp_landscape_distance_metric(p=2)
            @test isapprox(Distances.evaluate(m_land, L23_small, L23_small), 0.0; atol=1e-12)
            @test Distances.evaluate(m_land, L23_small, L3_small) >= 0.0

            bar = Dict((0.0, 1.0) => 1, (0.5, 1.5) => 1)
            m_bott = FEA.bottleneck_distance_metric()
            @test isapprox(Distances.evaluate(m_bott, bar, bar), 0.0; atol=1e-12)
            m_wass = FEA.wasserstein_distance_metric(p=2, q=1)
            @test isapprox(Distances.evaluate(m_wass, bar, bar), 0.0; atol=1e-12)

            arr_proj = Inv.projected_arrangement(pi; n_dirs=4, normalize=:L1, threads=false)
            cache23_proj = Inv.projected_barcode_cache(M23, arr_proj; precompute=true)
            cache3_proj = Inv.projected_barcode_cache(M3, arr_proj; precompute=true)
            m_proj = FEA.projected_distance_metric(dist=:bottleneck, agg=:mean, threads=false)
            @test Distances.evaluate(m_proj, cache23_proj, cache23_proj) <= Distances.evaluate(m_proj, cache23_proj, cache3_proj) + 1e-12
            D_proj = Distances.pairwise(m_proj, Any[cache23_proj, cache3_proj])
            @test size(D_proj) == (2, 2)
            @test isapprox(D_proj[1, 1], 0.0; atol=1e-12)
        end
    end

    @testset "Experiment runner" begin
        io_formats = FEA.ExperimentIOConfig(outdir=nothing,
                                           prefix="exp_formats",
                                           format=:wide,
                                           formats=[:csv_long, :npz, :csv_wide, :npz],
                                           write_metadata=false)
        @test io_formats.formats == [:npz, :csv_wide, :csv_long]
        io_ctor = FEA.ExperimentIOConfig(outdir=nothing,
                                        prefix="exp_ctor",
                                        format=:wide,
                                        formats=[:arrow, :parquet, :arrow],
                                        write_metadata=false)
        @test io_ctor.formats == [:arrow, :parquet]

        io_nowrite = FEA.ExperimentIOConfig(outdir=nothing,
                                           prefix="exp_nowrite",
                                           format=:wide,
                                           formats=Symbol[],
                                           write_metadata=false,
                                           overwrite=true)
        exp_nowrite = FEA.ExperimentSpec((lspec,);
                                        name="exp_nowrite",
                                        opts=opts_inv,
                                        batch=bserial,
                                        cache=:auto,
                                        io=io_nowrite,
                                        metadata=(tag="nowrite",))
        res_nowrite = TO.run_experiment(exp_nowrite, samples)
        @test length(res_nowrite.artifacts) == 1
        @test res_nowrite.run_dir === nothing
        @test res_nowrite.manifest_path === nothing
        @test isempty(res_nowrite.artifacts[1].feature_paths)
        @test res_nowrite.artifacts[1].metadata_path === nothing
        @test res_nowrite.total_elapsed_seconds >= 0.0

        tmp_exp = mktempdir()
        has_arrow = Base.find_package("Arrow") !== nothing
        has_parquet = Base.find_package("Parquet2") !== nothing
        io_out = FEA.ExperimentIOConfig(outdir=tmp_exp,
                                       prefix="exp_out",
                                       format=:wide,
                                       formats=(has_arrow ? [:arrow] : Symbol[]),
                                       write_metadata=true,
                                       overwrite=true)
        exp_out = FEA.ExperimentSpec((lspec, pispec);
                                    name="exp_out",
                                    opts=opts_inv,
                                    batch=bserial,
                                    cache=:auto,
                                    io=io_out,
                                    metadata=(purpose="featurizer_smoke",))
        res_out = TO.run_experiment(exp_out, samples)
        @test length(res_out.artifacts) == 2
        @test res_out.run_dir !== nothing
        @test isdir(res_out.run_dir)
        @test res_out.manifest_path !== nothing
        @test isfile(res_out.manifest_path)
        @test res_out.metadata["experiment_name"] == "exp_out"
        @test res_out.metadata["n_featurizers"] == 2
        for art in res_out.artifacts
            @test art.elapsed_seconds >= 0.0
            @test art.metadata_path !== nothing
            @test isfile(art.metadata_path)
            if has_arrow
                @test haskey(art.feature_paths, :arrow)
                @test isfile(art.feature_paths[:arrow])
            else
                @test !haskey(art.feature_paths, :arrow)
            end
        end
        @test res_out.artifacts[1].features.X == TO.featurize(samples, lspec; opts=opts_inv, batch=bserial, cache=:auto).X

        loaded_meta = TO.load_experiment(res_out.manifest_path; load_features=false)
        @test loaded_meta isa FEA.LoadedExperimentResult
        @test length(loaded_meta.artifacts) == 2
        @test loaded_meta.total_elapsed_seconds >= 0.0
        @test all(a -> a.features === nothing, loaded_meta.artifacts)
        @test all(a -> a.metadata !== nothing, loaded_meta.artifacts)

        loaded_none = TO.load_experiment(res_out.run_dir; load_features=true, prefer=:none)
        @test all(a -> a.features === nothing, loaded_none.artifacts)

        if has_arrow
            loaded_arrow = TO.load_experiment(res_out.run_dir; load_features=true, prefer=:arrow)
            @test length(loaded_arrow.artifacts) == 2
            @test loaded_arrow.artifacts[1].features !== nothing
            @test loaded_arrow.artifacts[1].features.X == res_out.artifacts[1].features.X
            @test loaded_arrow.artifacts[1].spec isa TO.AbstractFeaturizerSpec
            @test loaded_arrow.artifacts[1].opts isa TO.InvariantOptions
        end

        if has_parquet
            io_parq = FEA.ExperimentIOConfig(outdir=mktempdir(),
                                            prefix="exp_parq",
                                            format=:long,
                                            formats=[:parquet],
                                            write_metadata=true,
                                            overwrite=true)
            res_parq = TO.run_experiment((lspec,), samples;
                                         name="exp_parq",
                                         opts=opts_inv,
                                         batch=bserial,
                                         cache=:auto,
                                         io=io_parq)
            @test length(res_parq.artifacts) == 1
            @test haskey(res_parq.artifacts[1].feature_paths, :parquet)
            @test isfile(res_parq.artifacts[1].feature_paths[:parquet])
            @test isfile(res_parq.manifest_path)
            loaded_parq = TO.load_experiment(res_parq.manifest_path; prefer=:parquet)
            @test length(loaded_parq.artifacts) == 1
            @test loaded_parq.artifacts[1].features !== nothing
            @test loaded_parq.artifacts[1].features.X == res_parq.artifacts[1].features.X
        end

        has_npz = Base.find_package("NPZ") !== nothing
        has_csv = Base.find_package("CSV") !== nothing
        fmts = Symbol[]
        has_arrow && push!(fmts, :arrow)
        has_parquet && push!(fmts, :parquet)
        has_npz && push!(fmts, :npz)
        has_csv && append!(fmts, [:csv_wide, :csv_long])
        if !isempty(fmts)
            io_multi = FEA.ExperimentIOConfig(outdir=mktempdir(),
                                             prefix="exp_multi",
                                             format=:wide,
                                             formats=fmts,
                                             write_metadata=true,
                                             overwrite=true)
            res_multi = TO.run_experiment((lspec,), samples;
                                          name="exp_multi",
                                          opts=opts_inv,
                                          batch=bserial,
                                          cache=:auto,
                                          io=io_multi)
            @test length(res_multi.artifacts) == 1
            art = res_multi.artifacts[1]
            for f in fmts
                @test haskey(art.feature_paths, f)
                @test isfile(art.feature_paths[f])
            end
            @test art.metadata_path !== nothing
            @test isfile(art.metadata_path)
            md_multi = FEA.load_metadata_json(art.metadata_path; validate_feature_schema=false)
            @test String(md_multi["kind"]) == "experiment_feature"
            @test haskey(md_multi, "artifact_formats")
            @test Set(Symbol.(String.(md_multi["artifact_formats"]))) == Set(fmts)
            @test String(md_multi["artifact_layout"]) == "samples_by_features"

            man_multi = FEA.load_metadata_json(res_multi.manifest_path; validate_feature_schema=false)
            @test Set(Symbol.(String.(man_multi["artifact_formats"]))) == Set(fmts)
            @test haskey(man_multi["artifacts"][1]["feature_paths"], string(first(fmts)))

            if has_npz
                loaded_npz = TO.load_experiment(res_multi.manifest_path; prefer=:npz)
                @test loaded_npz.artifacts[1].features !== nothing
                @test loaded_npz.artifacts[1].features.X == res_multi.artifacts[1].features.X
            end
            if has_csv
                loaded_csv = TO.load_experiment(res_multi.manifest_path; prefer=:csv)
                @test loaded_csv.artifacts[1].features !== nothing
                @test loaded_csv.artifacts[1].features.X == res_multi.artifacts[1].features.X
            end
        end
    end

    @testset "Featurizers UX summaries and validators" begin
        fs_ux = TO.featurize(samples, lspec; opts=opts_inv, batch=bserial, cache=:auto)
        @test FEA.feature_names(fs_ux) == fs_ux.names
        @test FEA.sample_ids(fs_ux) == fs_ux.ids
        @test FEA.feature_matrix(fs_ux) === fs_ux.X
        @test FEA.feature_values(fs_ux) === fs_ux.X
        @test FEA.sample_labels(fs_ux) === nothing
        @test FEA.metadata(fs_ux) === fs_ux.meta
        @test TO.describe(fs_ux).kind == :feature_set
        @test FEA.feature_set_summary(fs_ux) == TO.describe(fs_ux)
        @test occursin("FeatureSet(", sprint(show, fs_ux))
        @test occursin("FeatureSet", sprint(show, MIME"text/plain"(), fs_ux))

        fs_ok = FEA.check_feature_set(fs_ux)
        @test fs_ok.valid
        @test TO.describe(fs_ok).kind == :feature_set_validation
        @test occursin("FeatureSetValidationSummary", sprint(show, fs_ok))

        bad_fs = FEA.FeatureSet(zeros(Float64, 2, 1), [:f1, :f1], ["s1"], (labels=["a"],))
        bad_fs_summary = FEA.check_feature_set(bad_fs)
        @test !bad_fs_summary.valid
        @test !isempty(bad_fs_summary.issues)
        @test_throws ArgumentError FEA.check_feature_set(bad_fs; throw=true)

        @test FEA.featurizer_summary(lspec).spec_type == :LandscapeSpec
        @test occursin("LandscapeSpec", sprint(show, lspec))
        @test occursin("LandscapeSpec", sprint(show, MIME"text/plain"(), lspec))
        good_spec_summary = FEA.check_featurizer_spec(lspec)
        @test good_spec_summary.valid

        bad_lspec = FEA.LandscapeSpec(
            [[0.0, 0.0]],
            [[0.0]],
            nothing,
            0,
            Float64[],
            :bad,
            true,
            nothing,
            nothing,
            0,
            true,
            true,
            true,
            :bad,
            :bad,
            nothing,
        )
        bad_spec_summary = FEA.check_featurizer_spec(bad_lspec)
        @test !bad_spec_summary.valid
        @test !isempty(bad_spec_summary.issues)
        @test_throws ArgumentError FEA.check_featurizer_spec(bad_lspec; throw=true)

        tmp_ux = mktempdir()
        io_ux = FEA.ExperimentIOConfig(outdir=tmp_ux,
                                       prefix="exp_ux",
                                       format=:wide,
                                       formats=Symbol[],
                                       write_metadata=true,
                                       overwrite=true)
        exp_ux = FEA.ExperimentSpec((lspec,);
                                    name="exp_ux",
                                    opts=opts_inv,
                                    batch=bserial,
                                    cache=:auto,
                                    io=io_ux,
                                    metadata=(tag="ux",))
        exp_summary = FEA.experiment_summary(exp_ux)
        @test exp_summary.kind == :experiment_spec
        @test FEA.experiment_name(exp_ux) == "exp_ux"
        @test occursin("ExperimentSpec", sprint(show, exp_ux))
        @test occursin("ExperimentSpec", sprint(show, MIME"text/plain"(), exp_ux))

        exp_valid = FEA.check_experiment_spec(exp_ux)
        @test exp_valid.valid
        @test occursin("ExperimentSpecValidationSummary", sprint(show, exp_valid))

        bad_rank_spec = FEA.RankGridSpec(-1, true, nothing)
        bad_exp = FEA.ExperimentSpec((bad_rank_spec,);
                                     name="",
                                     opts=opts_inv,
                                     batch=bserial,
                                     cache=:auto,
                                     io=FEA.ExperimentIOConfig(outdir=nothing,
                                                               prefix="bad_exp",
                                                               format=:wide,
                                                               formats=Symbol[],
                                                               write_metadata=false,
                                                               overwrite=true))
        bad_exp_summary = FEA.check_experiment_spec(bad_exp)
        @test !bad_exp_summary.valid
        @test !isempty(bad_exp_summary.invalid_specs)
        @test_throws ArgumentError FEA.check_experiment_spec(bad_exp; throw=true)

        res_ux = TO.run_experiment(exp_ux, samples)
        art_key = only(FEA.artifact_keys(res_ux))
        art_ux = FEA.artifact(res_ux, art_key)
        @test FEA.experiment_name(res_ux) == "exp_ux"
        @test FEA.artifacts(res_ux) === res_ux.artifacts
        @test FEA.nartifacts(res_ux) == 1
        @test FEA.artifact_spec(art_ux) === lspec
        @test FEA.artifact_features(art_ux) isa FEA.FeatureSet
        @test FEA.artifact_elapsed_seconds(art_ux) >= 0.0
        @test FEA.feature_paths(art_ux) == Dict{Symbol,String}()
        @test FEA.run_dir(res_ux) == res_ux.run_dir
        @test FEA.manifest_path(res_ux) == res_ux.manifest_path
        @test FEA.total_elapsed_seconds(res_ux) == res_ux.total_elapsed_seconds
        @test FEA.has_features(art_ux)
        @test FEA.has_features(res_ux)
        @test FEA.experiment_artifact_summary(art_ux).kind == :experiment_artifact
        @test occursin("ExperimentResult", sprint(show, res_ux))
        @test occursin("ExperimentArtifact", sprint(show, art_ux))

        loaded_ux = TO.load_experiment(res_ux.manifest_path; load_features=false)
        loaded_art = FEA.artifact(loaded_ux, art_key)
        @test FEA.experiment_name(loaded_ux) == "exp_ux"
        @test FEA.nartifacts(loaded_ux) == 1
        @test !FEA.has_features(loaded_art)
        @test !FEA.has_features(loaded_ux)
        @test FEA.artifact_spec(loaded_art) isa FEA.AbstractFeaturizerSpec
        @test FEA.artifact_features(loaded_art) === nothing
        @test FEA.experiment_artifact_summary(loaded_art).kind == :loaded_experiment_artifact
        @test occursin("LoadedExperimentResult", sprint(show, loaded_ux))
        @test occursin("LoadedExperimentArtifact", sprint(show, loaded_art))

        loaded_valid = FEA.check_loaded_experiment(loaded_ux)
        @test loaded_valid.valid
        @test occursin("LoadedExperimentValidationSummary", sprint(show, loaded_valid))

        bad_loaded = FEA.LoadedExperimentResult(
            Dict("kind" => "bad_manifest"),
            FEA.LoadedExperimentArtifact[
                FEA.LoadedExperimentArtifact(:dup, nothing, nothing, nothing, nothing, 0.0, Dict{String,Any}(), Dict{Symbol,String}(), nothing),
                FEA.LoadedExperimentArtifact(:dup, nothing, nothing, nothing, nothing, 0.0, Dict{String,Any}(), Dict{Symbol,String}(), nothing),
            ],
            nothing,
            "manifest.json",
            0.0,
        )
        bad_loaded_summary = FEA.check_loaded_experiment(bad_loaded)
        @test !bad_loaded_summary.valid
        @test !isempty(bad_loaded_summary.issues)
        @test_throws ArgumentError FEA.check_loaded_experiment(bad_loaded; throw=true)

        @test TOA.ExperimentSpec === FEA.ExperimentSpec
        @test TOA.ExperimentResult === FEA.ExperimentResult
        @test TOA.FeatureSetValidationSummary === FEA.FeatureSetValidationSummary
        @test TOA.feature_set_summary === FEA.feature_set_summary
        @test TOA.experiment_summary === FEA.experiment_summary
        @test TOA.experiment_artifact_summary === FEA.experiment_artifact_summary
        @test TOA.featurizer_summary === FEA.featurizer_summary
        @test TOA.check_feature_set === FEA.check_feature_set
        @test TOA.check_experiment_spec === FEA.check_experiment_spec
        @test TOA.check_loaded_experiment === FEA.check_loaded_experiment
        @test TOA.check_featurizer_spec === FEA.check_featurizer_spec
        @test TOA.sample_ids === FEA.sample_ids
        @test TOA.feature_matrix === FEA.feature_matrix
        @test TOA.feature_values === FEA.feature_values
        @test TOA.sample_labels === FEA.sample_labels
        @test TOA.metadata === FEA.metadata
        @test TOA.experiment_name === FEA.experiment_name
        @test TOA.artifact === FEA.artifact
    end

    @testset "Batch perf guards (lightweight)" begin
        perf_samples = vcat(fill(enc23, 16), fill(enc3, 16))
        perf_batch = TO.BatchOptions(threaded=false, backend=:serial, deterministic=true)
        perf_run(cache_mode) = TO.featurize(perf_samples, lspec; opts=opts_inv, batch=perf_batch, cache=cache_mode)

        # Warmup before timing/allocation checks.
        perf_run(nothing)
        perf_run(:auto)

        alloc_raw = @allocated perf_run(nothing)
        alloc_auto = @allocated perf_run(:auto)
        @test alloc_raw < 25_000_000
        @test alloc_auto < 25_000_000

        median_ms(f, reps::Int) = begin
            ts = Vector{Float64}(undef, reps)
            for i in 1:reps
                GC.gc()
                t0 = time_ns()
                f()
                ts[i] = (time_ns() - t0) / 1.0e6
            end
            sort!(ts)
            ts[cld(reps, 2)]
        end

        t_raw = median_ms(() -> perf_run(nothing), 3)
        t_auto = median_ms(() -> perf_run(:auto), 3)
        @test t_raw < 4000.0
        @test t_auto < 4000.0
        @test t_auto <= (3.0 * t_raw + 1.0)
    end
end
