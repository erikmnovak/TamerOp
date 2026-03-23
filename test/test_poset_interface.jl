using Test

# Included from test/runtests.jl; uses shared aliases (TO, FF, ...).

struct DummyEncodingMapForTupleContract <: EC.AbstractPLikeEncodingMap end
EC.dimension(::DummyEncodingMapForTupleContract) = 1
EC.locate(::DummyEncodingMapForTupleContract, x::AbstractVector{<:Real}) = (length(x) == 1 && x[1] >= 0 ? 1 : 0)

with_fields(FIELDS_FULL) do field
K = CM.coeff_type(field)
@inline c(x) = CM.coerce(field, x)

@testset "Structured posets: leq and cover_edges" begin
    coords = (collect(1:2), collect(10:10:30))
    P = FF.GridPoset(coords)

    # Index mapping: first axis varies fastest.
    idx(i, j) = i + (j - 1) * length(coords[1])

    @test FF.leq(P, idx(1, 1), idx(2, 3))
    @test !FF.leq(P, idx(2, 3), idx(1, 1))
    @test FF.leq(P, idx(2, 2), idx(2, 3))

    C = FF.cover_edges(P)
    @test length(C) == 7
    @test C[idx(1, 1), idx(2, 1)]
    @test C[idx(2, 1), idx(2, 2)]
    @test !C[idx(1, 1), idx(2, 2)]

    P1 = chain_poset(2)
    P2 = chain_poset(2)
    Pprod = FF.ProductPoset(P1, P2)
    idxp(i, j) = i + (j - 1) * P1.n

    @test FF.leq(Pprod, idxp(1, 1), idxp(2, 2))
    @test !FF.leq(Pprod, idxp(2, 2), idxp(1, 1))

    Cprod = FF.cover_edges(Pprod)
    @test length(Cprod) == 4
    @test Cprod[idxp(1, 1), idxp(2, 1)]
    @test Cprod[idxp(1, 1), idxp(1, 2)]
    @test !Cprod[idxp(1, 1), idxp(2, 2)]
end

@testset "upset_indices/downset_indices contract" begin
    P = chain_poset(5)
    for i in 1:FF.nvertices(P)
        @test collect(FF.upset_indices(P, i)) == collect(i:FF.nvertices(P))
        @test collect(FF.downset_indices(P, i)) == collect(1:i)
    end
end

@testset "GridPoset rejects duplicate or unsorted coords" begin
    @test_throws ErrorException FF.GridPoset((Float64[0.0, 0.0, 1.0], Float64[0.0, 1.0]))
    @test_throws ErrorException FF.GridPoset((Float64[0.0, 2.0, 1.0], Float64[0.0, 1.0]))
end

@testset "GridPoset size does not scale like n^2" begin
    coords = (collect(1:200), collect(1:200))
    P = FF.GridPoset(coords)

    # A dense leq matrix would be enormous; GridPoset should be tiny.
    nverts = length(coords[1]) * length(coords[2])
    dense_bytes = div(nverts * nverts, 8)  # BitMatrix uses 1 bit per entry.

    @test Base.summarysize(P) < div(dense_bytes, 100)

    # leq should be allocation-free in a tight loop.
    alloc = @allocated begin
        for _ in 1:10_000
            FF.leq(P, 1, 1)
        end
    end
    @test alloc < 1_000_000
end

@testset "monotone_upper_closure works on structured posets" begin
    # 2x2 grid, indices (1,1)=1, (2,1)=2, (1,2)=3, (2,2)=4
    coords = (collect(1:2), collect(1:2))
    P = FF.GridPoset(coords)
    s = [1.0, 3.0, 2.0, 4.0]
    t = TO.Invariants._monotone_upper_closure(P, s)
    @test t == [1.0, 3.0, 2.0, 4.0]

    # Non-monotone input should be corrected to the minimal isotone majorant.
    s2 = [1.0, 0.0, 2.0, 3.0]
    t2 = TO.Invariants._monotone_upper_closure(P, s2)
    @test t2 == [1.0, 1.0, 2.0, 3.0]
end

@testset "encode_pmodules_to_common_poset works for structured posets" begin
    P1 = FF.GridPoset((collect(1:2), collect(1:2)))
    P2 = FF.ProductPoset(chain_poset(2), chain_poset(3))

    dims1 = ones(Int, FF.nvertices(P1))
    dims2 = ones(Int, FF.nvertices(P2))

    edge_maps1 = Dict{Tuple{Int,Int}, Matrix{K}}()
    for (u, v) in FF.cover_edges(P1)
        edge_maps1[(u, v)] = Matrix{K}(I, 1, 1)
    end
    edge_maps2 = Dict{Tuple{Int,Int}, Matrix{K}}()
    for (u, v) in FF.cover_edges(P2)
        edge_maps2[(u, v)] = Matrix{K}(I, 1, 1)
    end

    M1 = MD.PModule{K}(P1, dims1, edge_maps1; field=field)
    M2 = MD.PModule{K}(P2, dims2, edge_maps2; field=field)

    out = TO.ChangeOfPosets.encode_pmodules_to_common_poset(M1, M2; method=:product, check_poset=false)
    @test TOA.CommonRefinementTranslationResult === TO.ChangeOfPosets.CommonRefinementTranslationResult
    @test TOA.common_poset === TO.ChangeOfPosets.common_poset
    @test TOA.projection_maps === TO.ChangeOfPosets.projection_maps
    @test TOA.translated_modules === TO.ChangeOfPosets.translated_modules
    @test out isa TO.ChangeOfPosets.CommonRefinementTranslationResult
    @test TO.describe(out).kind == :common_refinement_translation_result
    @test TO.ChangeOfPosets.common_poset(out) === out.P
    @test TO.ChangeOfPosets.projection_maps(out).left === out.pi1
    @test TO.ChangeOfPosets.projection_maps(out).right === out.pi2
    @test TO.ChangeOfPosets.translated_modules(out).left === out.Ms[1]
    @test TO.ChangeOfPosets.translated_modules(out).right === out.Ms[2]
    @test FF.nvertices(out.P) == FF.nvertices(P1) * FF.nvertices(P2)
    @test out.Ms[1].Q === out.P
    @test out.Ms[2].Q === out.P
    @test length(out.Ms[1].dims) == FF.nvertices(out.P)
    @test length(out.Ms[2].dims) == FF.nvertices(out.P)

    n1 = FF.nvertices(P1)
    @test out.pi1.Q === out.P
    @test out.pi2.Q === out.P
    @test out.pi1.P === P1
    @test out.pi2.P === P2
    @test out.pi1.pi_of_q[1] == 1
    @test out.pi2.pi_of_q[1] == 1
    @test out.pi1.pi_of_q[2] == 2
    @test out.pi2.pi_of_q[2] == 1
    @test out.pi1.pi_of_q[n1 + 1] == 1
    @test out.pi2.pi_of_q[n1 + 1] == 2
    Pout, Ms, pi1, pi2 = out
    @test Pout === out.P
    @test Ms === out.Ms
    @test pi1 === out.pi1
    @test pi2 === out.pi2
    @test occursin("CommonRefinementTranslationResult", sprint(show, out))
    @test occursin("common_nvertices:", sprint(io -> show(io, MIME"text/plain"(), out)))
end

@testset "SignaturePoset leq and cover_edges" begin
    sig_y = BitVector[
        BitVector([false, false]),
        BitVector([true,  false]),
        BitVector([true,  true]),
    ]
    sig_z = BitVector[
        BitVector([false]),
        BitVector([false]),
        BitVector([false]),
    ]
    P = TO.ZnEncoding.SignaturePoset(sig_y, sig_z)
    @test FF.leq(P, 1, 2)
    @test FF.leq(P, 2, 3)
    @test !FF.leq(P, 3, 1)

    C = FF.cover_edges(P)
    @test length(C) == 2
    @test C[1, 2]
    @test C[2, 3]
    @test !C[1, 3]
end

@testset "SignaturePoset upset/downset and branching covers" begin
    sig_y = BitVector[
        BitVector([false, false]),
        BitVector([true,  false]),
        BitVector([false, true]),
        BitVector([true,  true]),
    ]
    sig_z = [BitVector([false]) for _ in 1:4]
    P = TO.ZnEncoding.SignaturePoset(sig_y, sig_z)

    @test collect(FF.upset_indices(P, 1)) == [1, 2, 3, 4]
    @test collect(FF.upset_indices(P, 2)) == [2, 4]
    @test collect(FF.upset_indices(P, 3)) == [3, 4]
    @test collect(FF.upset_indices(P, 4)) == [4]

    @test collect(FF.downset_indices(P, 1)) == [1]
    @test collect(FF.downset_indices(P, 2)) == [1, 2]
    @test collect(FF.downset_indices(P, 3)) == [1, 3]
    @test collect(FF.downset_indices(P, 4)) == [1, 2, 3, 4]

    C = FF.cover_edges(P)
    @test Set(C.edges) == Set([(1, 2), (1, 3), (2, 4), (3, 4)])
    @test C === FF.cover_edges(P)
end

@testset "SignaturePoset up/down iterators match leq scans" begin
    sig_y = BitVector[
        BitVector([false, false, false]),
        BitVector([true,  false, false]),
        BitVector([false, true,  false]),
        BitVector([true,  true,  false]),
        BitVector([false, false, true]),
        BitVector([true,  false, true]),
        BitVector([false, true,  true]),
        BitVector([true,  true,  true]),
    ]
    sig_z = [BitVector([false, false]) for _ in eachindex(sig_y)]
    P = TO.ZnEncoding.SignaturePoset(sig_y, sig_z)

    n = FF.nvertices(P)
    for i in 1:n
        up_ref = [j for j in 1:n if FF.leq(P, i, j)]
        down_ref = [j for j in 1:n if FF.leq(P, j, i)]
        @test collect(FF.upset_indices(P, i)) == up_ref
        @test collect(FF.downset_indices(P, i)) == down_ref
    end

    # SignaturePoset avoids automatic dense up/down cache materialization.
    @test P.cache.upsets === nothing
    @test P.cache.downsets === nothing
end

@testset "Dense uptight construction skips redundant closure" begin
    sig_y = BitVector[
        BitVector([false, false, false]),
        BitVector([true,  false, false]),
        BitVector([true,  true,  false]),
        BitVector([true,  true,  true]),
    ]
    sig_z = [BitVector([false]) for _ in 1:4]

    function old_uptight(sig_y, sig_z)
        n = length(sig_y)
        L = falses(n, n)
        for i in 1:n, j in 1:n
            L[i, j] = TO.ZnEncoding._sig_subset(sig_y[i], sig_y[j]) &&
                      TO.ZnEncoding._sig_subset(sig_z[i], sig_z[j])
        end
        for k in 1:n, i in 1:n, j in 1:n
            L[i, j] = L[i, j] || (L[i, k] && L[k, j])
        end
        return L
    end

    Pdense = TO.ZnEncoding._uptight_from_signatures(sig_y, sig_z)
    @test FF.leq_matrix(Pdense) == old_uptight(sig_y, sig_z)
end

@testset "encode_from_flanges supports poset_kind=:signature" begin
    tau = FZ.face(1, [false])
    flat = FZ.IndFlat(tau, [0])
    inj = FZ.IndInj(tau, [0])
    FG = FZ.Flange{K}(1, [flat], [inj], reshape([c(1)], 1, 1); field=field)
    opts = OPT.EncodingOptions(backend = :zn, max_regions = 16)

    P, Hs, pi = TO.ZnEncoding.encode_from_flanges([FG], opts; poset_kind = :signature)
    @test P isa TO.ZnEncoding.SignaturePoset
    @test Hs[1].P === P
    @test length(pi.sig_y) == FF.nvertices(P)
end

@testset "uptight encoding supports poset_kind=:regions" begin
    Q = chain_poset(3)
    U1 = FF.principal_upset(Q, 1)
    U2 = FF.principal_upset(Q, 2)
    D1 = FF.principal_downset(Q, 2)
    M = FF.FringeModule{K}(Q, [U1, U2], [D1], reshape([c(1), c(0)], 1, 2); field=field)

    upt = TO.Encoding.build_uptight_encoding_from_fringe(M; poset_kind = :regions)
    P = upt.pi.P
    @test P isa FF.RegionsPoset

    upt_dense = TO.Encoding.build_uptight_encoding_from_fringe(M; poset_kind = :dense)
    @test FF.nvertices(P) == FF.nvertices(upt_dense.pi.P)
    @test FF.leq_matrix(P) == FF.leq_matrix(upt_dense.pi.P)
end

@testset "Structured poset caches reuse by identity" begin
    P = FF.GridPoset((collect(1:3), collect(1:3)))
    C1 = FF.cover_edges(P)
    C2 = FF.cover_edges(P)
    @test C1 === C2

    P2 = FF.GridPoset((collect(1:2), collect(1:2)))
    Q1 = FF.ProductPoset(P, P2)
    Q2 = FF.ProductPoset(P, P2)
    sc = CM.SessionCache()
    prod1 = TO.ChangeOfPosets.product_poset(Q1, Q2; use_cache=true, session_cache=sc)
    prod2 = TO.ChangeOfPosets.product_poset(Q1, Q2; use_cache=true, session_cache=sc)
    @test TOA.ProductPosetResult === TO.ChangeOfPosets.ProductPosetResult
    @test prod1 isa TO.ChangeOfPosets.ProductPosetResult
    @test TO.describe(prod1).kind == :product_poset_result
    @test TO.ChangeOfPosets.common_poset(prod1) === prod1.P
    @test TO.ChangeOfPosets.projection_maps(prod1).left === prod1.pi1
    @test TO.ChangeOfPosets.projection_maps(prod1).right === prod1.pi2
    Pprod, pi1, pi2 = prod1
    @test Pprod === prod1.P
    @test pi1 === prod1.pi1
    @test pi2 === prod1.pi2
    @test prod1.P === prod2.P
    _ = FF.cover_edges(prod1.P)
    @test prod1.P.cache.cover_edges !== nothing
    @test occursin("ProductPosetResult", sprint(show, prod1))
    @test occursin("common_nvertices:", sprint(io -> show(io, MIME"text/plain"(), prod1)))
end

@testset "Strict tuple locate contract and compile_encoding hooks" begin
    pi = DummyEncodingMapForTupleContract()
    @test EC.locate(pi, [1.0]) == 1
    @test_throws ArgumentError EC.locate(pi, (1.0,))

    P = chain_poset(1)
    enc = EC.compile_encoding(P, pi)
    @test enc.axes === nothing
    @test enc.reps === nothing
end

@testset "EncodingCore UX surface" begin
    P = chain_poset(4)
    pi = EC.GridEncodingMap(P, ([0.0, 1.0], [0.0, 2.0]))
    enc = EC.compile_encoding(P, pi)

    @test CC.describe(pi).kind == :grid_encoding_map
    @test CC.describe(enc).kind == :compiled_encoding
    @test CC.describe(enc).parameter_dim == 2
    @test CC.describe(enc).has_axes
    @test CC.describe(enc).has_representatives
    @test EC.encoding_summary(pi).kind == :grid_encoding_map
    @test EC.encoding_summary(enc).kind == :compiled_encoding

    @test EC.encoding_poset(enc) === P
    @test EC.encoding_map(enc) === pi
    @test EC.encoding_axes(enc) == ([0.0, 1.0], [0.0, 2.0])
    @test length(EC.encoding_representatives(enc)) == 4

    @test TOA.encoding_poset(enc) === P
    @test TOA.encoding_map(enc) === pi
    @test TOA.encoding_axes(enc) == ([0.0, 1.0], [0.0, 2.0])
    @test length(TOA.encoding_representatives(enc)) == 4

    @test EC.check_encoding_map(pi).valid
    @test EC.check_compiled_encoding(enc).valid
    @test EC.check_query_point(pi, [0.0, 1.0]).valid
    @test EC.check_query_matrix(pi, zeros(2, 3)).valid
    @test TOA.check_encoding_map(pi).valid
    @test TOA.check_compiled_encoding(enc).valid
    @test TOA.check_query_point(pi, [0.0, 1.0]).valid
    @test TOA.check_query_matrix(pi, zeros(2, 3)).valid
    @test TOA.encoding_summary(enc).kind == :compiled_encoding
    @test TOA.encoding_validation_summary(EC.check_encoding_map(pi)) isa EC.EncodingValidationSummary

    @test occursin("GridEncodingMap(", sprint(show, pi))
    @test occursin("axis_sizes:", sprint(show, MIME"text/plain"(), pi))
    @test occursin("CompiledEncoding(", sprint(show, enc))
    @test occursin("map_type:", sprint(show, MIME"text/plain"(), enc))
    @test occursin("EncodingValidationSummary(", sprint(show, EC.encoding_validation_summary(EC.check_query_point(pi, [0.0, 1.0]))))
    @test occursin("issues = none", sprint(show, MIME"text/plain"(), EC.encoding_validation_summary(EC.check_query_matrix(pi, zeros(2, 1)))))

    bad_pi = EC.GridEncodingMap{2,Float64,typeof(P)}(P, ([1.0, 0.0], [0.0, 2.0]), (1, 1), (2, 2), (1, 2))
    bad_report = EC.check_encoding_map(bad_pi)
    @test !bad_report.valid
    @test !isempty(bad_report.issues)
    @test any(s -> occursin("axis", s) || occursin("strides", s), String.(bad_report.issues))
    @test_throws ArgumentError EC.check_encoding_map(bad_pi; throw=true)
    bad_point = EC.check_query_point(pi, (0.0, 1.0))
    @test !bad_point.valid
    @test occursin("AbstractVector", join(bad_point.issues, "; "))
    @test_throws ArgumentError EC.check_query_point(pi, (0.0, 1.0); throw=true)
    bad_matrix = EC.check_query_matrix(pi, zeros(1, 3))
    @test !bad_matrix.valid
    @test occursin("row count 1", join(bad_matrix.issues, "; "))
    @test_throws ArgumentError EC.check_query_matrix(pi, zeros(1, 3); throw=true)

    err = try
        EC.locate(pi, [0.0])
        nothing
    catch e
        sprint(showerror, e)
    end
    @test err !== nothing
    @test occursin("expected a query vector of length 2", err)

    err_many = try
        EC.locate_many!(zeros(Int, 1), pi, zeros(2, 2))
        nothing
    catch e
        sprint(showerror, e)
    end
    @test err_many !== nothing
    @test occursin("destination length 1 must equal the number of query columns 2", err_many)

    @test occursin("canonical advanced-user wrapper", string(@doc EC.compile_encoding))
    @test occursin("Classify a query point", string(@doc EC.locate))
    @test occursin("Validate an encoding-map object", string(@doc EC.check_encoding_map))
    @test occursin("owner-module inspection", lowercase(string(@doc EC.encoding_summary)))
    @test occursin("default public", lowercase(string(@doc EC.check_query_point)))
    @test occursin("canonical object to pass around", lowercase(string(@doc EC.CompiledEncoding)))
    @test occursin("returns either a", lowercase(string(@doc EC.GridEncodingMap)))
end

@testset "Encoding UX surface" begin
    EN = TamerOp.Encoding
    Q = chain_poset(2)
    P = chain_poset(2)
    pi = EN.EncodingMap(Q, P, [1, 2])
    Y = [FF.principal_upset(Q, 1), FF.principal_upset(Q, 2)]
    enc = EN.UptightEncoding(pi, Y)
    pi0 = EC.GridEncodingMap(Q, ([0.0, 1.0],))
    post = EN.PostcomposedEncodingMap(pi0, pi)

    @test CC.describe(pi).kind == :finite_encoding_map
    @test CC.describe(pi).nsource == 2
    @test CC.describe(pi).image_size == 2
    @test CC.describe(enc).kind == :uptight_encoding
    @test CC.describe(enc).nconstant_upsets == 2
    @test CC.describe(enc).image_size == 2
    @test CC.describe(post).kind == :postcomposed_encoding_map
    @test CC.describe(post).ntarget == 2
    @test !CC.describe(post).representatives_cached
    @test TOA.describe(pi).kind == :finite_encoding_map
    @test TOA.describe(enc).nconstant_upsets == 2

    @test occursin("EncodingMap(", sprint(show, pi))
    @test occursin("image_size=", sprint(show, pi))
    @test occursin("source:", sprint(show, MIME"text/plain"(), pi))
    @test occursin("UptightEncoding(", sprint(show, enc))
    @test occursin("nconstant_upsets=", sprint(show, enc))
    @test occursin("nconstant_upsets:", sprint(show, MIME"text/plain"(), enc))
    @test occursin("PostcomposedEncodingMap(", sprint(show, post))
    @test occursin("nregions=", sprint(show, post))
    @test occursin("ambient_map_type:", sprint(show, MIME"text/plain"(), post))

    @test EN.source_poset(pi) === Q
    @test EN.target_poset(pi) === P
    @test EN.region_map(pi) == [1, 2]
    @test TOA.source_poset(pi) === Q
    @test TOA.target_poset(pi) === P
    @test TOA.region_map(pi) == [1, 2]
    @test TOA.ambient_poset(pi) === P

    @test EN.encoding_map(enc) === pi
    @test EN.constant_upsets(enc) === Y
    @test EN.source_poset(enc) === Q
    @test EN.target_poset(enc) === P
    @test TO.encoding_map(enc) === pi
    @test TOA.constant_upsets(enc) === Y
    @test TOA.source_poset(enc) === Q
    @test TOA.target_poset(enc) === P

    @test EN.source_poset(post) === Q
    @test EN.target_poset(post) === nothing
    @test TOA.source_poset(post) === Q
    @test TOA.target_poset(post) === nothing

    @test EN.check_encoding_map(pi).valid
    @test TO.check_encoding_map(pi).valid
    @test TOA.check_encoding_map(pi).valid
    @test EN.check_uptight_encoding(enc).valid
    @test TOA.check_uptight_encoding(enc).valid
    @test EN.check_postcomposed_encoding(post).valid
    @test TOA.check_postcomposed_encoding(post).valid

    @test occursin("finite encoding", lowercase(string(@doc EN.EncodingMap)))
    @test occursin("uptight encoding", lowercase(string(@doc EN.UptightEncoding)))
    @test occursin("build the canonical finite uptight encoding", lowercase(string(@doc EN.build_uptight_encoding_from_fringe)))
    @test occursin("finite-fringe upset/downset labels", lowercase(string(@doc EN.pullback_fringe_along_encoding)))

    badY = [FF.principal_upset(Q, 1), FF.principal_upset(chain_poset(2), 2)]
    bad_enc = EN.UptightEncoding(pi, badY)
    bad_report = EN.check_uptight_encoding(bad_enc)
    @test !bad_report.valid
    @test occursin("source poset", join(bad_report.issues, "; "))
    @test_throws ArgumentError EN.check_uptight_encoding(bad_enc; throw=true)
end

@testset "Option structs are concretely typed" begin
    enc_opts = OPT.EncodingOptions()
    inv_opts = OPT.InvariantOptions()
    ff_opts = OPT.FiniteFringeOptions()
    mod_opts = OPT.ModuleOptions()

    @test fieldtype(typeof(enc_opts), :strict_eps) !== Any
    @test fieldtype(typeof(inv_opts), :axes) !== Any
    @test fieldtype(typeof(inv_opts), :box) !== Any
    @test fieldtype(typeof(ff_opts), :scalar) !== Any
    @test fieldtype(typeof(mod_opts), :cache) !== Any
end

@testset "Options UX surface" begin
    fs = OPT.FiltrationSpec(kind=:rips, radius=1.0, max_dim=2)
    copt = OPT.ConstructionOptions()
    dopt = OPT.DataFileOptions()
    popt = OPT.PipelineOptions()
    eopt = OPT.EncodingOptions()
    ropt = OPT.ResolutionOptions()
    iopt = OPT.InvariantOptions()
    fopt = OPT.DerivedFunctorOptions()
    ffopt = OPT.FiniteFringeOptions()
    mopt = OPT.ModuleOptions()

    @test CC.describe(fs).kind == :filtration_spec
    @test CC.describe(fs).filtration_kind == :rips
    @test CC.describe(copt).kind == :construction_options
    @test CC.describe(copt).output_stage == :encoding_result
    @test CC.describe(dopt).kind == :datafile_options
    @test CC.describe(popt).kind == :pipeline_options
    @test CC.describe(eopt).kind == :encoding_options
    @test CC.describe(ropt).kind == :resolution_options
    @test CC.describe(iopt).kind == :invariant_options
    @test CC.describe(fopt).kind == :derived_functor_options
    @test CC.describe(ffopt).kind == :finite_fringe_options
    @test CC.describe(mopt).kind == :module_options

    @test TOA.describe(eopt).backend == :auto
    @test TOA.describe(iopt).pl_mode == :fast
    @test occursin("operational contract", string(@doc CC.describe(eopt)))
    @test occursin("how a filtration is built", string(@doc OPT.ConstructionOptions))
    @test occursin("chooses the encoding engine", string(@doc OPT.EncodingOptions))

    compact = sprint(show, eopt)
    pretty = sprint(show, MIME"text/plain"(), iopt)
    @test occursin("EncodingOptions(", compact)
    @test occursin("backend=:auto", compact)
    @test occursin("InvariantOptions", pretty)
    @test occursin("axes_policy:", pretty)
    @test occursin("pl_mode:", pretty)

    err = try
        OPT.ConstructionOptions(sparsify=:bad)
        ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("ConstructionOptions: sparsify must be one of", err)

    err = try
        OPT.DataFileOptions(kind=:bad)
        ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("DataFileOptions: kind must be one of", err)

    err = try
        OPT.EncodingOptions(backend=:bad)
        ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("EncodingOptions: backend must be one of", err)

    err = try
        OPT.DerivedFunctorOptions(model=:bad)
        ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("DerivedFunctorOptions: model must be one of", err)

    plmsg = try
        OPT.validate_pl_mode(:bad)
        ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("pl_mode must be exactly :fast or :verified", plmsg)
    @test occursin("Use :fast", plmsg)
end

@testset "DataTypes UX surface" begin
    pc = DT.PointCloud([0.0 1.0; 2.0 3.0])
    gd = DT.GraphData(3, [(1, 2), (2, 3)])
    epg = DT.EmbeddedPlanarGraph2D([0.0 0.0; 1.0 0.0; 1.0 1.0], [(1, 2), (2, 3)])
    b1 = sparse([1, 2], [1, 1], [1, -1], 2, 1)
    gc = DT.GradedComplex([[1, 2], [1]], [b1], [(0.0,), (1.0,), (2.0,)])
    mgc = DT.MultiCriticalGradedComplex([[1, 2], [1]], [b1],
                                        [[(0.0,)], [(1.0,)], [(2.0,), (3.0,)]])
    st = DT.SimplexTreeMulti([1, 2, 3, 5], [1, 2, 1, 2], [0, 0, 1], [1, 3, 4],
                             [1, 2, 3, 4], [(0.0,), (1.0,), (2.0,)])

    @test TOA.npoints(pc) == 2
    @test TOA.ambient_dim(pc) == 2
    @test TOA.nedges(gd) == 2
    @test DT.nvertices(gd) == 3
    @test TOA.max_dim(gc) == 1
    @test TOA.cell_counts(gc) == [2, 1]
    @test TOA.parameter_dim(gc) == 1
    @test TOA.max_dim(st) == 1
    @test TOA.cell_counts(st) == [2, 1]
    @test TOA.parameter_dim(st) == 1
    @test collect(TOA.edge_list(gd)) == [(1, 2), (2, 3)]
    @test collect(TOA.edge_list(epg)) == [(1, 2), (2, 3)]
    @test length(TOA.vertex_positions(epg)) == 3
    @test collect(TOA.vertex_coordinates(epg)) == collect(TOA.vertex_positions(epg))
    @test TOA.vertex_coordinates(gd) === nothing
    @test TOA.edge_weights(gd) === nothing
    @test TOA.polylines(epg) === nothing
    @test TOA.bounding_box(epg) === nothing
    @test TOA.cells(gc; dim=0) == [1, 2]
    @test TOA.cells(gc; dim=1) == [1]
    @test TOA.cell_grades(gc) == gc.grades
    @test TOA.cell_grade(gc, 2) == (1.0,)
    @test TOA.cell_grade_set(gc, 2) == ((1.0,),)
    @test length(TOA.cell_grades(mgc)) == 3
    @test collect(TOA.cell_grade_set(mgc, 3)) == [(2.0,), (3.0,)]
    @test TOA.boundary_maps(gc) === gc.boundaries
    @test TOA.boundary_maps(mgc) === mgc.boundaries
    @test TOA.boundary_map(gc; dim=1) === gc.boundaries[1]
    @test TOA.nsimplices(st) == 3
    @test TOA.simplex_dimension(st, 3) == 1
    @test collect(TOA.simplices(st; dim=1)[1]) == [1, 2]
    @test collect(TOA.simplex_grades(st)[3]) == [(2.0,)]
    @test collect(TOA.simplex_grade_set(st, 3)) == [(2.0,)]
    @test DT.data_summary(st).kind == :simplex_tree_multi

    @test CC.describe(pc).kind == :point_cloud
    @test CC.describe(gd).kind == :graph_data
    @test CC.describe(epg).kind == :embedded_planar_graph_2d
    @test CC.describe(gc).kind == :graded_complex
    @test CC.describe(mgc).kind == :multicritical_graded_complex
    @test CC.describe(st).kind == :simplex_tree_multi

    @test TOA.check_point_cloud(pc).valid
    @test TOA.check_graph_data(gd).valid
    @test TOA.check_embedded_planar_graph(epg).valid
    @test TOA.check_graded_complex(gc).valid
    @test TOA.check_multicritical_complex(mgc).valid
    @test TOA.check_simplex_tree_multi(st).valid
    @test TOA.data_validation_summary(DT.check_graph_data(gd)) isa DT.DataValidationSummary

    compact = sprint(show, gc)
    pretty = sprint(show, MIME"text/plain"(), st)
    @test occursin("GradedComplex(", compact)
    @test occursin("SimplexTreeMulti", pretty)
    @test occursin("simplex_counts:", pretty)
    @test occursin("DataValidationSummary(", sprint(show, DT.data_validation_summary(DT.check_graded_complex(gc))))
    @test occursin("issues = none", sprint(show, MIME"text/plain"(), DT.data_validation_summary(DT.check_point_cloud(pc))))

    gd_bad = DT.GraphData{Float64}(2, [1, 3], [2, 1], nothing, nothing)
    bad_report = DT.check_graph_data(gd_bad)
    @test !bad_report.valid
    @test occursin("outside 1:2", join(bad_report.issues, "; "))
    @test_throws ArgumentError DT.check_graph_data(gd_bad; throw=true)
    @test occursin("readable notebook", lowercase(string(@doc DT.check_graph_data)))
    @test occursin("graph edge set", lowercase(string(@doc DT.edge_list)))
    @test occursin("grading data", lowercase(string(@doc DT.cell_grades)))
    @test occursin("owner-module inspection", lowercase(string(@doc DT.data_summary)))
    @test occursin("validator currently checks", lowercase(string(@doc DT.check_simplex_tree_multi)))

    st_bad = DT.SimplexTreeMulti{1,Float64}([1, 2, 4], [1, 2, 3], [0, 0], [1, 3], [1, 2, 3],
                                            [(0.0,), (1.0,)])
    st_bad_report = DT.check_simplex_tree_multi(st_bad)
    @test !st_bad_report.valid
    @test_throws ArgumentError DT.check_simplex_tree_multi(st_bad; throw=true)
end

@testset "Point codensity UX surface" begin
    pc = DT.PointCloud([0.0 0.0; 1.0 0.0; 2.0 0.0])
    filt = DI.RipsCodensityFiltration(max_dim=1, knn=2, dtm_mass=0.5, nn_backend=:bruteforce)

    cod = TO.point_codensity(pc, filt)
    @test cod isa DI.PointCodensityResult
    @test TO.codensity_mass(cod) == 0.5
    @test TO.neighbor_count(cod) == 2
    @test DI.source_data(cod) === pc

    vals = TO.codensity_values(cod)
    @test length(vals) == 3
    @test all(v -> v ≈ inv(sqrt(2)), vals)

    ds = TO.describe(cod)
    @test ds.kind == :point_codensity_result
    @test ds.npoints == 3
    @test ds.ambient_dim == 2
    @test ds.dtm_mass == 0.5
    @test ds.neighbor_count == 2
    @test ds.value_range[1] ≈ inv(sqrt(2))
    @test ds.value_range[2] ≈ inv(sqrt(2))

    @test occursin("PointCodensityResult(", sprint(show, cod))
    @test occursin("neighbor_count:", sprint(show, MIME"text/plain"(), cod))

    spec = OPT.FiltrationSpec(kind=:rips_codensity, max_dim=1, knn=2, dtm_mass=0.5, nn_backend=:bruteforce)
    cod_spec = TO.point_codensity(pc, spec)
    @test TO.codensity_values(cod_spec) ≈ vals

    @test_throws ArgumentError TO.point_codensity(pc, DI.RipsFiltration(max_dim=1, nn_backend=:bruteforce))
    @test_throws ArgumentError TO.point_codensity(pc, OPT.FiltrationSpec(kind=:rips, max_dim=1, nn_backend=:bruteforce))
end

@testset "Results UX surface" begin
    P = chain_poset(2)
    edge_maps = Dict((1, 2) => sparse([1], [1], [c(1)], 1, 1))
    M = MD.PModule(P, [1, 1], edge_maps)
    pi = EC.GridEncodingMap(P, ([0.0, 1.0],))

    enc = RES.EncodingResult(P, M, pi; backend=:test)
    C = TO.ModuleCochainComplex([M], MD.PMorphism{K}[]; tmin=0, check=true)
    enc_complex = RES.EncodedComplexResult(P, C, pi; field=field)
    dims_res = RES.CohomologyDimsResult(P, [1, 0], pi; degree=1, field=field)
    res = RES.ResolutionResult(:dummy_resolution; enc=enc, betti=[1, 0], minimality=(checked=true,))
    inv = RES.InvariantResult(dims_res, :restricted_hilbert, Dict((1,) => 1))
    inv_complex = RES.InvariantResult(enc_complex, :euler_surface, fill(c(0), 2))

    @test TO.encoding_poset(enc) === P
    @test TO.encoding_module(enc) === M
    @test TO.encoding_map(enc) === pi
    @test TO.encoding_poset(enc_complex) === P
    @test TO.encoding_complex(enc_complex) === C
    @test TO.encoding_map(enc_complex) === pi
    @test TO.encoding_axes(enc) == ([0.0, 1.0],)
    @test length(TO.encoding_representatives(enc)) == 2
    @test TO.cohomology_dims(dims_res) == [1, 0]
    @test TO.resolution_object(res) == :dummy_resolution
    @test TO.invariant_value(inv) == Dict((1,) => 1)
    @test TO.source_result(enc) === nothing
    @test TO.source_result(enc_complex) === nothing
    @test TO.source_result(dims_res) === nothing
    @test TO.source_result(res) === enc
    @test TO.source_result(inv) === dims_res
    @test RES.result_summary(enc).kind == :encoding_result
    @test RES.result_summary(enc_complex).kind == :encoded_complex_result
    @test RES.result_summary(dims_res).kind == :cohomology_dims_result
    @test RES.result_summary(res).kind == :resolution_result
    @test RES.result_summary(inv).kind == :invariant_result
    @test TOA.result_summary(enc).kind == :encoding_result

    @test TO.dimensions(enc) == [1, 1]
    @test TO.dimensions(dims_res) == [1, 0]
    @test TO.describe(enc).kind == :encoding_result
    @test TO.describe(enc_complex).kind == :encoded_complex_result
    @test CC.dimensions(enc) == [1, 1]
    @test CC.dimensions(dims_res) == [1, 0]

    @test CC.describe(enc).kind == :encoding_result
    @test CC.describe(enc_complex).kind == :encoded_complex_result
    @test CC.describe(enc).backend == :test
    @test CC.describe(dims_res).kind == :cohomology_dims_result
    @test CC.describe(res).kind == :resolution_result
    @test CC.describe(inv).kind == :invariant_result

    @test RES.check_encoding_result(enc).valid
    @test RES.check_encoded_complex_result(enc_complex).valid
    @test RES.check_cohomology_dims_result(dims_res).valid
    @test RES.check_resolution_result(res).valid
    @test RES.check_invariant_result(inv).valid
    @test RES.check_invariant_result(inv_complex).valid

    @test TOA.check_encoding_result(enc).valid
    @test TOA.check_encoded_complex_result(enc_complex).valid
    @test TOA.check_cohomology_dims_result(dims_res).valid
    @test TOA.check_resolution_result(res).valid
    @test TOA.check_invariant_result(inv).valid
    @test TOA.result_validation_summary(RES.check_encoding_result(enc)) isa RES.ResultValidationSummary

    vcompact = sprint(show, RES.result_validation_summary(RES.check_encoding_result(enc)))
    @test occursin("ResultValidationSummary(", vcompact)
    vpretty = sprint(show, MIME"text/plain"(), RES.result_validation_summary(RES.check_invariant_result(inv)))
    @test occursin("issues = none", vpretty)

    @test occursin("EncodingResult(", sprint(show, enc))
    @test occursin("backend:", sprint(show, MIME"text/plain"(), enc))
    @test occursin("EncodedComplexResult(", sprint(show, enc_complex))
    @test occursin("complex_type:", sprint(show, MIME"text/plain"(), enc_complex))
    @test occursin("CohomologyDimsResult(", sprint(show, dims_res))
    @test occursin("degree:", sprint(show, MIME"text/plain"(), dims_res))
    @test occursin("ResolutionResult(", sprint(show, res))
    @test occursin("has_betti:", sprint(show, MIME"text/plain"(), res))
    @test occursin("InvariantResult(", sprint(show, inv))
    @test occursin("invariant:", sprint(show, MIME"text/plain"(), inv))

    bad_enc = RES.EncodingResult(nothing, M, pi; backend=:test)
    @test !RES.check_encoding_result(bad_enc).valid
    @test_throws ArgumentError RES.check_encoding_result(bad_enc; throw=true)
    bad_enc_complex = RES.EncodedComplexResult(nothing, C, pi; field=field)
    @test !RES.check_encoded_complex_result(bad_enc_complex).valid
    @test_throws ArgumentError RES.check_encoded_complex_result(bad_enc_complex; throw=true)

    bad_inv = RES.InvariantResult(:bad_source, nothing, nothing)
    @test !RES.check_invariant_result(bad_inv).valid
    @test_throws ArgumentError RES.check_invariant_result(bad_inv; throw=true)

    @test occursin("workflow-facing wrapper", lowercase(string(@doc RES.EncodingResult)))
    @test occursin("encoded cochain complex", lowercase(string(@doc RES.EncodedComplexResult)))
    @test occursin("semantic accessor", lowercase(string(@doc RES.encoding_module)))
    @test occursin("provenance", lowercase(string(@doc RES.source_result)))
    @test occursin("display-oriented", lowercase(string(@doc RES.result_validation_summary)))
    @test occursin("owner-local summary alias", lowercase(string(@doc RES.result_summary)))
    @test occursin("result_summary", string(@doc RES.ResolutionResult))
end

@testset "Workflow UX surface" begin
    Ups = [PLB.BoxUpset([0.0])]
    Downs = [PLB.BoxDownset([2.0])]
    sc = CM.SessionCache()

    enc = TO.encode(Ups, Downs; backend=:pl_backend, cache=sc)
    @test enc isa RES.EncodingResult

    box_syn = SD.box_bar_fringe(
        bars=[([0.0, 0.0], [1.0, 1.0]), ([1.25, 0.25], [2.25, 1.5])];
        field=field,
    )
    enc_syn = TO.encode(box_syn; cache=sc)
    @test enc_syn isa RES.EncodingResult
    @test enc_syn.presentation === box_syn
    @test TO.describe(enc_syn).kind == :encoding_result

    enc_syn_direct = TO.encode(FF.birth_upsets(box_syn),
                               FF.death_downsets(box_syn),
                               FZ.coefficient_matrix(box_syn);
                               field=field,
                               cache=sc)
    @test FF.nvertices(TO.encoding_poset(enc_syn)) == FF.nvertices(TO.encoding_poset(enc_syn_direct))

    ri_syn = TO.rank_invariant(enc_syn; opts=OPT.InvariantOptions(threads=false))
    ri_syn_direct = TO.rank_invariant(enc_syn_direct; opts=OPT.InvariantOptions(threads=false))
    @test Dict(ri_syn) == Dict(ri_syn_direct)

    enc_syn_stage = TO.encode(box_syn; output=:encoding_result, cache=sc)
    @test enc_syn_stage isa RES.EncodingResult

    P_syn_raw, H_syn_raw, pi_syn_raw = TO.encode(box_syn; output=:raw, cache=sc)
    @test FF.nvertices(P_syn_raw) == FF.nvertices(TO.encoding_poset(enc_syn))
    @test H_syn_raw isa FF.FringeModule
    @test TO.encoding_axes(pi_syn_raw) == TO.encoding_axes(TO.encoding_map(enc_syn))
    @test TOA.locate(pi_syn_raw, [0.5, 0.5]) == TOA.locate(TO.encoding_map(enc_syn), [0.5, 0.5])

    enc2 = TO.coarsen(enc; cache=sc)
    @test enc2 isa RES.EncodingResult

    res = TO.resolve(enc; cache=sc)
    @test res isa RES.ResolutionResult

    inv = TO.invariant(enc; which=:restricted_hilbert, cache=sc)
    @test inv isa RES.InvariantResult

    @test TO.hom_dimension(enc, enc; cache=sc) >= 1

    Psmall = chain_poset(1)
    Qbig = chain_poset(2)
    pi_q_to_p = EN.EncodingMap(Qbig, Psmall, [1, 1])
    id_P = EN.EncodingMap(Psmall, Psmall, [1])
    id_Q = EN.EncodingMap(Qbig, Qbig, collect(1:FF.nvertices(Qbig)))

    MP = IR.pmodule_from_fringe(FF.one_by_one_fringe(Psmall,
                                                     FF.principal_upset(Psmall, 1),
                                                     FF.principal_downset(Psmall, 1),
                                                     c(1); field=field))
    MQ = IR.pmodule_from_fringe(FF.one_by_one_fringe(Qbig,
                                                     FF.principal_upset(Qbig, 1),
                                                     FF.principal_downset(Qbig, 2),
                                                     c(1); field=field))

    encP = RES.EncodingResult(Psmall, MP, id_P; backend=:test)
    encQ = RES.EncodingResult(Qbig, MQ, id_Q; backend=:test)

    @test TamerOp.restriction === TamerOp.Workflow.restriction
    @test TamerOp.common_refinement === TamerOp.Workflow.common_refinement
    @test TamerOp.pushforward_left === TamerOp.Workflow.pushforward_left
    @test TamerOp.pushforward_right === TamerOp.Workflow.pushforward_right
    @test TamerOp.derived_pushforward_left === TamerOp.Workflow.derived_pushforward_left
    @test TamerOp.derived_pushforward_right === TamerOp.Workflow.derived_pushforward_right
    @test TamerOp.betti_table === TamerOp.Workflow.betti_table
    @test TamerOp.bass_table === TamerOp.Workflow.bass_table
    @test TamerOp.matching_distance_exact_2d === TamerOp.Workflow.matching_distance_exact_2d
    @test TamerOp.CommonRefinementTranslationResult === TO.ChangeOfPosets.CommonRefinementTranslationResult
    @test TamerOp.ModuleTranslationResult === RES.ModuleTranslationResult
    @test TamerOp.common_poset === TO.ChangeOfPosets.common_poset
    @test TamerOp.projection_maps === TO.ChangeOfPosets.projection_maps
    @test TamerOp.translated_modules === TO.ChangeOfPosets.translated_modules
    @test TamerOp.translated_module === RES.translated_module
    @test TamerOp.translation_map === RES.translation_map
    @test TamerOp.translation_kind === RES.translation_kind

    cref = TamerOp.common_refinement(encP, encQ; cache=sc)
    @test cref isa TO.ChangeOfPosets.CommonRefinementTranslationResult
    @test TamerOp.describe(cref).kind == :common_refinement_translation_result
    @test TamerOp.common_poset(cref) === cref.P
    @test TamerOp.projection_maps(cref).left === cref.pi1
    @test TamerOp.projection_maps(cref).right === cref.pi2
    @test TamerOp.translated_modules(cref).left === cref.Ms[1]
    @test TamerOp.translated_modules(cref).right === cref.Ms[2]

    rest = TamerOp.restriction(pi_q_to_p, encP; cache=sc)
    @test rest isa RES.ModuleTranslationResult
    @test TamerOp.translation_kind(rest) == :restriction
    @test TamerOp.translation_map(rest) === pi_q_to_p
    @test TamerOp.source_result(rest) === encP
    @test TamerOp.translated_module(rest) isa MD.PModule
    @test TamerOp.translated_module(rest).Q === Qbig
    @test TamerOp.encoding_poset(rest) === Qbig
    @test TamerOp.encoding_map(rest) === nothing
    @test TamerOp.result_summary(rest).kind == :module_translation_result
    @test TamerOp.describe(rest).translation_kind == :restriction
    @test TamerOp.unwrap(rest) === TamerOp.translated_module(rest)
    @test occursin("ModuleTranslationResult(", sprint(show, rest))
    @test occursin("translation_kind:", sprint(show, MIME"text/plain"(), rest))

    pi_enc_to_small = EN.EncodingMap(enc.P, Psmall, fill(1, FF.nvertices(enc.P)))

    left = TamerOp.pushforward_left(pi_enc_to_small, enc; cache=sc)
    @test left isa RES.ModuleTranslationResult
    @test TamerOp.translation_kind(left) == :pushforward_left
    @test TamerOp.source_result(left) === enc
    @test TamerOp.translated_module(left).Q === Psmall
    @test TamerOp.encoding_poset(left) === Psmall
    @test TamerOp.encoding_map(left) !== nothing
    @test TamerOp.describe(left).has_classifier

    right = TamerOp.pushforward_right(pi_enc_to_small, enc; cache=sc)
    @test right isa RES.ModuleTranslationResult
    @test TamerOp.translation_kind(right) == :pushforward_right
    @test TamerOp.source_result(right) === enc
    @test TamerOp.translated_module(right).Q === Psmall
    @test TamerOp.encoding_map(right) !== nothing

    df_opts = OPT.DerivedFunctorOptions(maxdeg=0)
    dleft = TamerOp.derived_pushforward_left(pi_q_to_p, encQ; opts=df_opts, cache=sc)
    @test dleft isa Vector{RES.ModuleTranslationResult}
    @test !isempty(dleft)
    @test TamerOp.translation_kind(first(dleft)) == :derived_pushforward_left
    @test first(dleft).meta.derived_degree == 0

    dright = TamerOp.derived_pushforward_right(pi_q_to_p, encQ; opts=df_opts, cache=sc)
    @test dright isa Vector{RES.ModuleTranslationResult}
    @test !isempty(dright)
    @test TamerOp.translation_kind(first(dright)) == :derived_pushforward_right
    @test first(dright).meta.derived_degree == 0

    @test_throws ErrorException TamerOp.hom_dimension(encP, encQ; cache=sc)
    @test_throws ErrorException TamerOp.hom(encP, encQ; cache=sc)
    @test_throws ArgumentError TamerOp.hom_dimension(encP, encQ; cache=sc, transport=:bad)
    @test_throws ArgumentError TamerOp.hom(encP, encQ; cache=sc, transport=:bad)

    common_dim = TamerOp.hom_dimension(encP, encQ; cache=sc, transport=:common_refinement)
    @test common_dim == TamerOp.ChangeOfPosets.hom_dim_common_refinement(MP, MQ; session_cache=sc)

    common_hom = TamerOp.hom(encP, encQ; cache=sc, transport=:common_refinement)
    @test DF.dim(common_hom) == common_dim

    psm = TamerOp.point_signed_measure(enc; opts=OPT.InvariantOptions(threads=false), cache=sc)
    esm = TamerOp.euler_signed_measure(enc; opts=OPT.InvariantOptions(threads=false), cache=sc)
    @test TamerOp.describe(psm).kind == :point_signed_measure
    @test TamerOp.describe(esm).kind == :point_signed_measure
    @test TamerOp.describe(psm).total_mass == TamerOp.describe(esm).total_mass
    @test TamerOp.describe(psm).nterms == TamerOp.describe(esm).nterms

    betti_enc = TamerOp.betti_table(enc; cache=sc)
    bass_enc = TamerOp.bass_table(enc; cache=sc)
    @test betti_enc isa Matrix{Int}
    @test bass_enc isa Matrix{Int}
    @test betti_enc == DF.betti_table(TamerOp.resolution_object(TamerOp.resolve(enc; kind=:projective, cache=sc)))
    @test bass_enc == DF.bass_table(TamerOp.resolution_object(TamerOp.resolve(enc; kind=:injective, cache=sc)))

    res_proj = TamerOp.resolve(enc; kind=:projective, cache=sc)
    res_inj = TamerOp.resolve(enc; kind=:injective, cache=sc)
    @test TamerOp.betti_table(res_proj) == betti_enc
    @test TamerOp.bass_table(res_inj) == bass_enc
    @test_throws ErrorException TamerOp.betti_table(res_inj)
    @test_throws ErrorException TamerOp.bass_table(res_proj)

    tau = FZ.face(2, [])
    FG = FZ.Flange(2,
                   [FZ.IndFlat(tau, [0, 0]; id=:F_workflow)],
                   [FZ.IndInj(tau, [2, 2]; id=:E_workflow)],
                   reshape([c(1)], 1, 1))
    enc_zn = TamerOp.encode(FG; backend=:zn, cache=sc)
    rect_opts = OPT.InvariantOptions(threads=false, axes_policy=:encoding, max_axis_len=8)
    sb = TamerOp.rectangle_signed_barcode(enc_zn; opts=rect_opts, cache=sc)
    @test TamerOp.describe(sb).kind == :rect_signed_barcode
    img = TamerOp.rectangle_signed_barcode_image(
        enc_zn;
        opts=rect_opts,
        cache=sc,
        rect_kwargs=(drop_zeros=true,),
        sigma=0.75,
    )
    @test img isa Matrix{Float64}
    @test size(img, 1) > 0
    @test size(img, 2) > 0

    Ups2d = [
        PLB.BoxUpset([0.0, -10.0]),
        PLB.BoxUpset([1.0, -10.0]),
    ]
    Downs2d = PLB.BoxDownset[]
    P2d, _, pi2d = PLB.encode_fringe_boxes(Ups2d, Downs2d, OPT.EncodingOptions())
    r_left = TOA.locate(pi2d, [0.5, 0.0])
    r_right = TOA.locate(pi2d, [2.0, 0.0])

    M_left = IR.pmodule_from_fringe(
        FF.one_by_one_fringe(
            P2d,
            FF.principal_upset(P2d, r_left),
            FF.principal_downset(P2d, r_right),
            c(1);
            field=field,
        ),
    )
    M_right = IR.pmodule_from_fringe(
        FF.one_by_one_fringe(
            P2d,
            FF.principal_upset(P2d, r_right),
            FF.principal_downset(P2d, r_right),
            c(1);
            field=field,
        ),
    )
    enc_left = RES.EncodingResult(P2d, M_left, pi2d; backend=:test)
    enc_right = RES.EncodingResult(P2d, M_right, pi2d; backend=:test)
    opts_exact = OPT.InvariantOptions(box=([-1.0, -1.0], [2.0, 1.0]), strict=true, threads=false)

    d_exact = TamerOp.matching_distance_exact_2d(
        enc_left,
        enc_right;
        opts=opts_exact,
        cache=sc,
        weight=:lesnick_l1,
        normalize_dirs=:L1,
        precompute=:none,
        store_values=true,
    )
    d_exact_direct = TO.Fibered2D.matching_distance_exact_2d(
        M_left,
        M_right,
        pi2d,
        opts_exact;
        weight=:lesnick_l1,
        normalize_dirs=:L1,
        precompute=:none,
        store_values=true,
    )
    @test isapprox(d_exact, d_exact_direct; atol=1e-12, rtol=0.0)
    @test TamerOp.matching_distance_exact_2d(enc_left, enc_right;
                                                  opts=opts_exact,
                                                  cache=sc,
                                                  weight=:lesnick_l1,
                                                  normalize_dirs=:L1,
                                                  precompute=:none,
                                                  store_values=true) == d_exact
    @test_throws ErrorException TamerOp.matching_distance_exact_2d(encP, encQ; opts=opts_exact, cache=sc)
    @test_throws ErrorException TamerOp.matching_distance_exact_2d(enc, enc; cache=sc)

    workflow_doc = string(@doc TO.Workflow)
    encode_doc = string(@doc TO.encode)
    coarsen_doc = string(@doc TO.coarsen)
    commonref_doc = string(@doc TamerOp.common_refinement)
    restriction_doc = string(@doc TamerOp.restriction)
    pushleft_doc = string(@doc TamerOp.pushforward_left)
    pushright_doc = string(@doc TamerOp.pushforward_right)
    dpushleft_doc = string(@doc TamerOp.derived_pushforward_left)
    dpushright_doc = string(@doc TamerOp.derived_pushforward_right)
    resolve_doc = string(@doc TO.resolve)
    betti_doc = string(@doc TamerOp.betti_table)
    bass_doc = string(@doc TamerOp.bass_table)
    invariant_doc = string(@doc TO.invariant)
    homdim_doc = string(@doc TO.hom_dimension)
    hom_doc = string(@doc TO.hom)
    ext_doc = string(@doc TO.ext)
    tor_doc = string(@doc TO.tor)
    rhom_doc = string(@doc TO.rhom)
    hyperext_doc = string(@doc TO.hyperext)
    hypertor_doc = string(@doc TO.hypertor)
    extalg_doc = string(@doc TO.ext_algebra)
    invariants_doc = string(@doc TO.invariants)
    rankinv_doc = string(@doc TO.rank_invariant)
    hilbert_doc = string(@doc TO.restricted_hilbert)
    pointsigned_doc = string(@doc TamerOp.point_signed_measure)
    euler_doc = string(@doc TO.euler_surface)
    eulersigned_doc = string(@doc TamerOp.euler_signed_measure)
    rectbarcode_doc = string(@doc TamerOp.rectangle_signed_barcode)
    rectimage_doc = string(@doc TamerOp.rectangle_signed_barcode_image)
    slicebar_doc = string(@doc TO.slice_barcode)
    slice_doc = string(@doc TO.slice_barcodes)
    exact2d_doc = string(@doc TamerOp.matching_distance_exact_2d)
    landscape_doc = string(@doc TO.mp_landscape)
    mppdec_doc = string(@doc TO.mpp_decomposition)
    mppimg_doc = string(@doc TO.mpp_image)
    translation_result_doc = string(@doc TamerOp.ModuleTranslationResult)

    @test occursin("Workflow output policy", workflow_doc)
    @test occursin("typed wrappers", workflow_doc)
    @test occursin("task-oriented workflow surface", workflow_doc)

    @test occursin("Presentation encoding", encode_doc)
    @test occursin("Data-ingestion execution", encode_doc)
    @test occursin("SyntheticBoxFringe", encode_doc)
    @test occursin("`output` vs `stage`", encode_doc)
    @test occursin("output=:result", encode_doc)
    @test occursin("output=:encoding_result", encode_doc)
    @test occursin("stage=:encoding_result", encode_doc)
    @test occursin("cache=:auto", coarsen_doc)
    @test occursin("EncodingResult", coarsen_doc)
    @test occursin("CommonRefinementTranslationResult", commonref_doc)
    @test occursin("ModuleTranslationResult", restriction_doc)
    @test occursin("ModuleTranslationResult", pushleft_doc)
    @test occursin("ModuleTranslationResult", pushright_doc)
    @test occursin("ModuleTranslationResult", dpushleft_doc)
    @test occursin("ModuleTranslationResult", dpushright_doc)

    @test occursin("cache=:auto", resolve_doc)
    @test occursin("SessionCache", resolve_doc)
    @test occursin("ResolutionResult", resolve_doc)
    @test occursin("dense Betti table", betti_doc)
    @test occursin("cache=:auto", betti_doc)
    @test occursin("dense Bass table", bass_doc)
    @test occursin("cache=:auto", bass_doc)

    @test occursin("cache=:auto", invariant_doc)
    @test occursin("InvariantResult", invariant_doc)

    @test occursin("cache=:auto", homdim_doc)
    @test occursin("transport=:common_refinement", homdim_doc)
    @test occursin("SessionCache", homdim_doc)
    @test occursin("cache=:auto", hom_doc)
    @test occursin("transport=:common_refinement", hom_doc)
    @test occursin("cache=:auto", ext_doc)
    @test occursin("cache=:auto", tor_doc)
    @test occursin("Hom-system cache", rhom_doc)
    @test occursin("algebraic HyperExt object directly", hyperext_doc)
    @test occursin("HyperTor", hypertor_doc)
    @test occursin("Ext-algebra", extalg_doc)
    @test occursin("Batch convenience wrapper", invariants_doc)
    @test occursin("bare rank-invariant value", rankinv_doc)
    @test occursin("bare restricted-Hilbert value", hilbert_doc)
    @test occursin("point-signed measure", lowercase(pointsigned_doc))
    @test occursin("bare Euler-surface value", euler_doc)
    @test occursin("Euler signed measure", eulersigned_doc)
    @test occursin("rectangle signed barcode", lowercase(rectbarcode_doc))
    @test occursin("rect_kwargs", rectimage_doc)
    @test occursin("one slice-barcode value", slicebar_doc)
    @test occursin("cache=:auto", slice_doc)
    @test occursin("exact 2D", exact2d_doc)
    @test occursin("common classifier map", exact2d_doc)
    @test occursin("cache=sc::SessionCache", exact2d_doc)
    @test occursin("cache=:auto", landscape_doc)
    @test occursin("bare multiparameter decomposition", mppdec_doc)
    @test occursin("bare multiparameter image", mppimg_doc)
    @test occursin("workflow-facing wrapper", lowercase(translation_result_doc))
end

@testset "SessionCache uses sharded encoding/module stores" begin
    sc = CM.SessionCache()
    P = chain_poset(2)
    ec = CM._encoding_cache!(sc, P)
    mc = CM._module_cache!(sc, TO.zero_pmodule(P; field=field))
    @test ec isa CM.EncodingCache
    @test mc isa CM.ModuleCache
    @test CM._session_encoding_bucket_count(sc) >= 1
    @test CM._session_module_bucket_count(sc) >= 1
    @test length(sc.zn_pushforward_plan) >= 1
    if Threads.nthreads() == 1
        @test length(sc.zn_pushforward_plan) == 1
    end

    compact = sprint(show, sc)
    @test occursin("SessionCache(", compact)
    @test occursin("encoding=", compact)
    @test occursin("modules=", compact)

    pretty = sprint(show, MIME"text/plain"(), sc)
    @test occursin("SessionCache", pretty)
    @test occursin("encoding buckets:", pretty)
    @test occursin("module buckets:", pretty)
    @test occursin("shard layout:", pretty)
end

@testset "Cache payload maps use typed wrappers" begin
    @test fieldtype(CM.EncodingCache, :posets) == Dict{Tuple{Tuple,Tuple{Vararg{Int}}},CM.PosetCachePayload}
    @test fieldtype(CM.EncodingCache, :cubical) == Dict{Tuple{Vararg{Int}},CM.CubicalCachePayload}
    @test fieldtype(CM.EncodingCache, :geometry) == Dict{Tuple,CM.GeometryCachePayload}
    @test fieldtype(CM.ModuleCache, :payload) == Dict{Symbol,CM.ModulePayload}
    @test fieldtype(CM.ResolutionCache, :projective) == Dict{CM.ResolutionKey2,CM.ProjectiveResolutionPayload}
    @test fieldtype(CM.ResolutionCache, :injective) == Dict{CM.ResolutionKey2,CM.InjectiveResolutionPayload}
    @test fieldtype(CM.ResolutionCache, :indicator) == Dict{CM.ResolutionKey3,CM.IndicatorResolutionPayload}
    @test TO.DerivedFunctors.HomSystemCache <: CM.AbstractHomSystemCache
    @test TamerOp.SliceInvariants.SlicePlanCache <: CM.AbstractSlicePlanCache
    @test fieldtype(CM.SessionCache, :hom_system) == Union{Nothing,CM.AbstractHomSystemCache}
    @test fieldtype(CM.SessionCache, :slice_plan) == Union{Nothing,CM.AbstractSlicePlanCache}
    @test fieldtype(CM.SessionCache, :zn_pushforward_fringe) ==
          Vector{Dict{Tuple{UInt64,Symbol,UInt64,UInt},CM.ZnPushforwardFringeArtifact{Any}}}
    @test fieldtype(CM.SessionCache, :product_dense) ==
          Dict{CM._SessionProductKey,CM.ProductPosetCacheEntry{Any,Any,Any,Any,Any}}
    @test fieldtype(CM.SessionCache, :product_obj) ==
          Dict{CM._SessionProductKey,CM.ProductPosetCacheEntry{Any,Any,Any,Any,Any}}

    sc = CM.SessionCache()
    P = chain_poset(2)
    H = FF.one_by_one_fringe(P, FF.principal_upset(P, 1), FF.principal_downset(P, 1), c(1); field=field)
    fid = CM._field_cache_key(field)
    efp = UInt64(0x11)
    ffp = UInt64(0x22)
    CM._session_set_zn_pushforward_fringe!(sc, efp, :signature, ffp, fid, H)
    @test CM._session_get_zn_pushforward_fringe(sc, efp, :signature, ffp, fid) === H
    @test CM._session_zn_pushforward_fringe_count(sc) == 1
end

@testset "Workflow cache validator gives actionable errors" begin
    msg = ""
    try
        CM._resolve_workflow_session_cache(:bad_mode)
    catch err
        @test err isa ArgumentError
        msg = sprint(showerror, err)
    end
    @test occursin("cache must be one of :auto, nothing, or SessionCache()", msg)
    @test occursin("cache=:auto", msg)
    @test occursin("cache=SessionCache()", msg)
    @test occursin("cache=nothing", msg)
end
end # with_fields
