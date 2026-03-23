using Test
using Random
using SparseArrays

const FZ = TO.FlangeZn
const ZE = TO.ZnEncoding
const IR = TO.IndicatorResolutions
with_fields(FIELDS_FULL) do field
    K = CM.coeff_type(field)
    @inline c(x) = CM.coerce(field, x)

# ---------------------------------------------------------------------------
# Local helpers for the updated FlangeZn indicator API.
#
# The library intentionally removed the old Flat/Injective and flat/inj shims
# (and also disallows ambiguous reordered IndFlat/IndInj constructors) because
# both inputs are vectors and mixups were a frequent source of subtle bugs.
# ---------------------------------------------------------------------------
    mk_face(n, coords) = FZ.face(n, coords)
    mk_flat(b, coords; id=:F) = FZ.IndFlat(mk_face(length(b), coords), b; id=id)
    mk_inj(b, coords; id=:E) = FZ.IndInj(mk_face(length(b), coords), b; id=id)

    function _gather_projection_rows_naive(Bu::Matrix{K},
                                           rows_u::Vector{Int},
                                           rows_v::Vector{Int}) where {K}
        Ph = length(rows_v)
        du = size(Bu, 2)
        out = Matrix{K}(undef, Ph, du)
        Pg = length(rows_u)
        j = 1
        @inbounds for i in 1:Ph
            target = rows_v[i]
            while j <= Pg && rows_u[j] < target
                j += 1
            end
            if j > Pg || rows_u[j] != target
                error("naive box reference: projection mismatch; expected rows_v subset rows_u")
            end
            for c in 1:du
                out[i, c] = Bu[j, c]
            end
        end
        return out
    end

    function _naive_pmodule_on_box(FG::FZ.Flange{K},
                                   a::NTuple{N,Int},
                                   b::NTuple{N,Int},
                                   field::CM.AbstractCoeffField) where {K,N}
        Q, coords = ZE.grid_poset(a, b)
        ncoords = length(coords)
        rows_at = Vector{Vector{Int}}(undef, ncoords)
        B = Vector{Matrix{K}}(undef, ncoords)
        dims = Vector{Int}(undef, ncoords)

        @inbounds for u in 1:ncoords
            A, rows, cols = FZ.degree_matrix(FG, coords[u])
            rows_at[u] = rows
            if isempty(rows) || isempty(cols)
                B[u] = zeros(K, length(rows), 0)
                dims[u] = 0
            else
                Bu = TamerOp.FieldLinAlg.colspace(field, A)
                B[u] = Bu
                dims[u] = size(Bu, 2)
            end
        end

        edge_maps = Dict{Tuple{Int,Int}, Matrix{K}}()
        for (u, v) in FF.cover_edges(Q)
            du = dims[u]
            dv = dims[v]
            if du == 0 || dv == 0
                edge_maps[(u, v)] = zeros(K, dv, du)
                continue
            end
            Im = _gather_projection_rows_naive(B[u], rows_at[u], rows_at[v])
            edge_maps[(u, v)] = TamerOp.FieldLinAlg.solve_fullcolumn(field, B[v], Im)
        end

        return TO.PModule{K}(Q, dims, edge_maps; field=field)
    end

    function _sample_comparable_pairs(Q; rng::AbstractRNG, nsample::Int=64)
        pairs = Tuple{Int,Int}[]
        n = TO.nvertices(Q)
        for u in 1:n, v in 1:n
            FF.leq(Q, u, v) || continue
            push!(pairs, (u, v))
        end
        if length(pairs) <= nsample
            return pairs
        end
        order = collect(1:length(pairs))
        Random.shuffle!(rng, order)
        return [pairs[order[i]] for i in 1:nsample]
    end

    function _images_on_P_closure_baseline(P::FF.AbstractPoset,
                                           sig_y::AbstractVector{<:AbstractVector{Bool}},
                                           sig_z::AbstractVector{<:AbstractVector{Bool}},
                                           flat_idxs::AbstractVector{<:Integer},
                                           inj_idxs::AbstractVector{<:Integer})
        m = length(flat_idxs)
        r = length(inj_idxs)
        n = FF.nvertices(P)
        Uhat = Vector{FF.Upset}(undef, m)
        Dhat = Vector{FF.Downset}(undef, r)

        @inbounds for (loc, i0) in enumerate(flat_idxs)
            i = Int(i0)
            mask = falses(n)
            for t in 1:n
                mask[t] = sig_y[t][i]
            end
            Uhat[loc] = FF.upset_closure(P, mask)
        end
        @inbounds for (loc, j0) in enumerate(inj_idxs)
            j = Int(j0)
            mask = falses(n)
            for t in 1:n
                mask[t] = !sig_z[t][j]
            end
            Dhat[loc] = FF.downset_closure(P, mask)
        end
        return Uhat, Dhat
    end

    function _module_equal_on_covers(M1, M2, field)
        @test FF.poset_equal(M1.Q, M2.Q)
        @test M1.dims == M2.dims
        for (u, v) in FF.cover_edges(M1.Q)
            A = M1.edge_maps[u, v]
            B = M2.edge_maps[u, v]
            if field isa CM.RealField
                @test isapprox(A, B; rtol=1e-8, atol=1e-10)
            else
                @test A == B
            end
        end
    end

    @testset "ZnEncoding _images_on_P matches closure baseline" begin
        tau = FZ.face(2, [false, false])
        flats = [
            FZ.IndFlat(tau, [0, 0]; id=:F1),
            FZ.IndFlat(tau, [1, 0]; id=:F2),
            FZ.IndFlat(tau, [0, 1]; id=:F3),
        ]
        injectives = [
            FZ.IndInj(tau, [1, 1]; id=:E1),
            FZ.IndInj(tau, [2, 0]; id=:E2),
        ]
        phi = Matrix{K}(I, length(injectives), length(flats))
        FG = FZ.Flange{K}(2, flats, injectives, phi; field=field)
        opts = TO.EncodingOptions(backend=:zn, max_regions=512)
        P, pi = ZE.encode_poset_from_flanges(FG, opts; poset_kind=:signature)

        flat_index, inj_index = ZE._generator_index_dicts(pi.flats, pi.injectives)
        flat_idxs = [flat_index[ZE._flat_key(F)] for F in FG.flats]
        inj_idxs = [inj_index[ZE._inj_key(E)] for E in FG.injectives]

        Unew, Dnew = ZE._images_on_P(P, pi.sig_y, pi.sig_z, flat_idxs, inj_idxs)
        Uold, Dold = _images_on_P_closure_baseline(P, pi.sig_y, pi.sig_z, flat_idxs, inj_idxs)

        @test length(Unew) == length(Uold)
        @test length(Dnew) == length(Dold)
        @test [U.mask for U in Unew] == [U.mask for U in Uold]
        @test [D.mask for D in Dnew] == [D.mask for D in Dold]

        for U in Unew
            @test U.mask == FF.upset_closure(P, copy(U.mask)).mask
        end
        for D in Dnew
            @test D.mask == FF.downset_closure(P, copy(D.mask)).mask
        end
    end

    @testset "ZnEncoding strict pushforward cache is built and reused" begin
        tau = FZ.face(2, [false, false])
        flats = [
            FZ.IndFlat(tau, [0, 0]; id=:F1),
            FZ.IndFlat(tau, [1, 0]; id=:F2),
        ]
        injectives = [
            FZ.IndInj(tau, [1, 1]; id=:E1),
            FZ.IndInj(tau, [2, 0]; id=:E2),
        ]
        phi = reshape([c(1), c(0), c(0), c(1)], 2, 2)
        FG = FZ.Flange{K}(2, flats, injectives, phi; field=field)
        opts = TO.EncodingOptions(backend=:zn, max_regions=512)
        P, pi = ZE.encode_poset_from_flanges(FG, opts; poset_kind=:signature)

        @test pi.pushforward_cache.flat_index === nothing
        @test pi.pushforward_cache.inj_index === nothing
        @test pi.pushforward_cache.flat_masks === nothing
        @test pi.pushforward_cache.inj_masks === nothing
        @test isempty(pi.pushforward_cache.plan_by_flange)

        H1 = ZE._pushforward_flange_to_fringe(P, pi, FG; strict=true)
        @test pi.pushforward_cache.flat_index !== nothing
        @test pi.pushforward_cache.inj_index !== nothing
        @test pi.pushforward_cache.flat_masks !== nothing
        @test pi.pushforward_cache.inj_masks !== nothing
        fkey = ZE._flange_fingerprint(FG)
        @test haskey(pi.pushforward_cache.plan_by_flange, fkey)
        fi_ref = pi.pushforward_cache.flat_index
        ji_ref = pi.pushforward_cache.inj_index
        fm_ref = pi.pushforward_cache.flat_masks
        im_ref = pi.pushforward_cache.inj_masks
        plan_ref = pi.pushforward_cache.plan_by_flange[fkey]
        @test H1.phi == ZE._monomialize_phi(FG.phi, H1.U, H1.D)
        @test H1.phi == ZE._monomialize_phi_with_zero_pairs(FG.phi, plan_ref.zero_pairs)

        H2 = ZE._pushforward_flange_to_fringe(P, pi, FG; strict=true)
        @test pi.pushforward_cache.flat_index === fi_ref
        @test pi.pushforward_cache.inj_index === ji_ref
        @test pi.pushforward_cache.flat_masks === fm_ref
        @test pi.pushforward_cache.inj_masks === im_ref
        @test pi.pushforward_cache.plan_by_flange[fkey] === plan_ref
        @test length(H1.U) == length(H2.U)
        @test length(H1.D) == length(H2.D)
        @test H1.phi == H2.phi
    end

    @testset "ZnEncoding direct strict plan -> PModule matches fringe path" begin
        tau = FZ.face(2, [false, false])
        flats = [
            FZ.IndFlat(tau, [0, 0]; id=:F1),
            FZ.IndFlat(tau, [1, 0]; id=:F2),
            FZ.IndFlat(tau, [0, 1]; id=:F3),
        ]
        injectives = [
            FZ.IndInj(tau, [1, 1]; id=:E1),
            FZ.IndInj(tau, [2, 0]; id=:E2),
        ]
        phi = reshape([c(1), c(0), c(1), c(0), c(1), c(0)], 2, 3)
        FG = FZ.Flange{K}(2, flats, injectives, phi; field=field)
        opts = TO.EncodingOptions(backend=:zn, max_regions=512, field=field)
        P, pi = ZE.encode_poset_from_flanges(FG, opts; poset_kind=:signature)

        plan, flat_masks, inj_masks = ZE._strict_pushforward_plan_and_masks(pi, FG)
        M_direct = ZE._pmodule_from_pushforward_plan(P, FG, plan, flat_masks, inj_masks)
        H = ZE._strict_pushforward_fringe_from_plan(P, FG, plan, flat_masks, inj_masks)
        M_via_fringe = IR.pmodule_from_fringe(H)
        _module_equal_on_covers(M_direct, M_via_fringe, field)
    end

    @testset "ZnEncoding signatures are packed with Bool-vector views" begin
        tau = FZ.face(2, [false, false])
        flats = [
            FZ.IndFlat(tau, [0, 0]; id=:F1),
            FZ.IndFlat(tau, [1, 0]; id=:F2),
            FZ.IndFlat(tau, [0, 1]; id=:F3),
        ]
        injectives = [
            FZ.IndInj(tau, [1, 1]; id=:E1),
            FZ.IndInj(tau, [2, 0]; id=:E2),
        ]
        phi = reshape([c(1), c(0), c(0), c(1), c(1), c(0)], 2, 3)
        FG = FZ.Flange{K}(2, flats, injectives, phi; field=field)
        opts = TO.EncodingOptions(backend=:zn, max_regions=512)
        _, pi = ZE.encode_poset_from_flanges(FG, opts; poset_kind=:signature)

        @test pi.sig_y isa ZE.PackedSignatureRows
        @test pi.sig_z isa ZE.PackedSignatureRows
        @test pi.sig_y.bitlen == length(pi.flats)
        @test pi.sig_z.bitlen == length(pi.injectives)

        rid = 1
        yrow = collect(Bool, pi.sig_y[rid])
        zrow = collect(Bool, pi.sig_z[rid])
        yexp = [FZ.in_flat(pi.flats[i], pi.reps[rid]) for i in 1:length(pi.flats)]
        zexp = [!FZ.in_inj(pi.injectives[j], pi.reps[rid]) for j in 1:length(pi.injectives)]
        @test yrow == yexp
        @test zrow == zexp
    end

    @testset "Workflow SessionCache reuses Zn pushforward plans across encodes" begin
        tau = FZ.face(2, [false, false])
        flats = [
            FZ.IndFlat(tau, [0, 0]; id=:F1),
            FZ.IndFlat(tau, [1, 0]; id=:F2),
        ]
        injectives = [
            FZ.IndInj(tau, [1, 1]; id=:E1),
            FZ.IndInj(tau, [2, 0]; id=:E2),
        ]
        phi = reshape([c(1), c(0), c(0), c(1)], 2, 2)
        FG = FZ.Flange{K}(2, flats, injectives, phi; field=field)
        opts = TO.EncodingOptions(backend=:zn, max_regions=512, field=field)
        sc = CM.SessionCache()

        enc1 = TO.encode(FG, opts; cache=sc)
        enc2 = TO.encode(FG, opts; cache=sc)
        pi1 = enc1.pi.pi
        pi2 = enc2.pi.pi

        fkey = ZE._flange_fingerprint(FG)
        ekey = pi1.encoding_fingerprint
        plan1 = pi1.pushforward_cache.plan_by_flange[fkey]
        plan2 = pi2.pushforward_cache.plan_by_flange[fkey]
        cached_payload = CM._session_get_zn_pushforward_plan(sc, ekey, fkey)

        @test cached_payload !== nothing
        @test cached_payload isa NamedTuple{(:flat_idxs,:inj_idxs,:zero_pairs)}
        @test cached_payload.flat_idxs === plan1.flat_idxs
        @test cached_payload.inj_idxs === plan1.inj_idxs
        @test cached_payload.zero_pairs === plan1.zero_pairs
        @test CM._session_get_zn_pushforward_plan(sc, ekey, fkey) === cached_payload
        @test plan2 === plan1
        @test CM._session_zn_pushforward_plan_count(sc) == 1
    end

    @testset "Zn locate tuple-float path avoids tuple->vector fallback allocation" begin
        tau = FZ.face(2, [false, false])
        flats = [
            FZ.IndFlat(tau, [0, 0]; id=:F1),
            FZ.IndFlat(tau, [1, 0]; id=:F2),
        ]
        injectives = [
            FZ.IndInj(tau, [1, 1]; id=:E1),
            FZ.IndInj(tau, [2, 0]; id=:E2),
        ]
        phi = reshape([c(1), c(0), c(0), c(1)], 2, 2)
        FG = FZ.Flange{K}(2, flats, injectives, phi; field=field)
        opts = TO.EncodingOptions(backend=:zn, max_regions=512, field=field)
        _, pi = ZE.encode_poset_from_flanges((FG,), opts; poset_kind=:signature)

        q1 = EC.locate(pi, (0.25, 0.75))
        q2 = EC.locate(pi, [0.25, 0.75])
        @test q1 == q2

        # Warmup once so @allocated captures runtime allocations only.
        EC.locate(pi, (0.25, 0.75))
        alloc_tuple = @allocated EC.locate(pi, (0.25, 0.75))
        @test alloc_tuple <= 32
    end

    @testset "Workflow SessionCache reuses Zn encoding artifacts and pushed fringes" begin
        tau = FZ.face(2, [false, false])
        flats = [
            FZ.IndFlat(tau, [0, 0]; id=:F1),
            FZ.IndFlat(tau, [1, 0]; id=:F2),
        ]
        injectives = [
            FZ.IndInj(tau, [1, 1]; id=:E1),
            FZ.IndInj(tau, [2, 0]; id=:E2),
        ]
        phi = reshape([c(1), c(0), c(0), c(1)], 2, 2)
        FG = FZ.Flange{K}(2, flats, injectives, phi; field=field)
        opts = TO.EncodingOptions(backend=:zn, max_regions=512, field=field)
        sc = CM.SessionCache()

        enc1 = TO.encode(FG, opts; cache=sc)
        enc2 = TO.encode(FG, opts; cache=sc)

        @test enc1.P === enc2.P
        @test enc1.pi.pi === enc2.pi.pi
        @test enc1.H === enc2.H
        @test enc1.M === enc2.M
        @test CM._session_zn_encoding_artifact_count(sc) == 1
        @test CM._session_zn_pushforward_fringe_count(sc) == 1
        @test CM._session_zn_pushforward_module_count(sc) == 1
    end

    @testset "ZnEncoding session cache reuses encode_poset_from_flanges artifacts" begin
        tau = FZ.face(2, [false, false])
        flats = [
            FZ.IndFlat(tau, [0, 0]; id=:F1),
            FZ.IndFlat(tau, [1, 0]; id=:F2),
        ]
        injectives = [
            FZ.IndInj(tau, [1, 1]; id=:E1),
            FZ.IndInj(tau, [2, 0]; id=:E2),
        ]
        phi = reshape([c(1), c(0), c(0), c(1)], 2, 2)
        FG = FZ.Flange{K}(2, flats, injectives, phi; field=field)
        opts = TO.EncodingOptions(backend=:zn, max_regions=512, field=field)
        sc = CM.SessionCache()

        P1, pi1 = ZE.encode_poset_from_flanges((FG,), opts;
                                               poset_kind=:signature,
                                               session_cache=sc)
        P2, pi2 = ZE.encode_poset_from_flanges((FG,), opts;
                                               poset_kind=:signature,
                                               session_cache=sc)
        @test P1 === P2
        @test pi1 === pi2
        @test CM._session_zn_encoding_artifact_count(sc) == 1
    end

    @testset "Workflow SessionCache reuses pushed fringes per flange in common encode" begin
        tau = FZ.face(1, [])
        FG1 = FZ.Flange{K}(1,
                           [FZ.IndFlat(tau, [0]; id=:F1)],
                           [FZ.IndInj(tau, [5]; id=:E1)],
                           reshape([c(1)], 1, 1);
                           field=field)
        FG2 = FZ.Flange{K}(1,
                           [FZ.IndFlat(tau, [2]; id=:F2)],
                           [FZ.IndInj(tau, [7]; id=:E2)],
                           reshape([c(1)], 1, 1);
                           field=field)
        opts = TO.EncodingOptions(backend=:zn, max_regions=512, field=field)
        sc = CM.SessionCache()

        out1 = TO.encode((FG1, FG2), opts; cache=sc)
        out2 = TO.encode((FG1, FG2), opts; cache=sc)

        @test out1[1].P === out2[1].P
        @test out1[1].P === out1[2].P
        @test out1[1].P === out2[2].P
        @test out1[1].H === out2[1].H
        @test out1[2].H === out2[2].H
        @test out1[1].M === out2[1].M
        @test out1[2].M === out2[2].M
        @test CM._session_zn_encoding_artifact_count(sc) == 1
        @test CM._session_zn_pushforward_fringe_count(sc) == 2
        @test CM._session_zn_pushforward_module_count(sc) == 2
    end


    @testset "Zn wrappers: common encoding matches explicit encoding route" begin
    # Two 1D flanges representing interval modules [0,5] and [2,7].
    tau = FZ.face(1, [])  # no free coordinates for n=1 (i.e. an interval-type flange)

    F1 = FZ.IndFlat(tau, [0]; id=:F1)
    E1 = FZ.IndInj(tau, [5]; id=:E1)
    FG1 = FZ.Flange{K}(1, [F1], [E1], reshape([c(1)], 1, 1))

    F2 = FZ.IndFlat(tau, [2]; id=:F2)
    E2 = FZ.IndInj(tau, [7]; id=:E2)
    FG2 = FZ.Flange{K}(1, [F2], [E2], reshape([c(1)], 1, 1))

    enc = TO.EncodingOptions(backend=:zn, max_regions=50_000)
    df = TO.DerivedFunctorOptions(maxdeg=2)
    res = TO.ResolutionOptions(maxlen=3)

    enc_out = TO.encode((FG1, FG2), enc)
    P = enc_out[1].P
    Ms = [enc_out[1].M, enc_out[2].M]

    @test length(Ms) == 2
    @test Ms[1].Q === P
    @test Ms[2].Q === P

    E_explicit = DF.Ext(Ms[1], Ms[2], df)
    E_wrap = DF.ExtZn(FG1, FG2, enc, df)

    @test [TO.dim(E_explicit, t) for t in 0:2] == [TO.dim(E_wrap, t) for t in 0:2]

    # Resolution wrappers: compare against "encode + resolution" directly.
    enc1 = TO.encode(FG1, enc)
    res_wrap = DF.projective_resolution_Zn(FG1, enc, res; return_encoding=true)
    @test FF.poset_equal(res_wrap.P, enc1.P)
    @test length(res_wrap.res.Pmods) == res.maxlen + 1
    @test length(res_wrap.res.d_mor) == res.maxlen
    @test DF.betti_table(res_wrap.res) == DF.betti_table(DF.projective_resolution(enc1.M, res))

    res_min = TO.ResolutionOptions(maxlen=3, minimal=true, check=true)

    bt_wrap = DF.betti(DF.projective_resolution_Zn(FG1, enc, res_min))
    bt_explicit = DF.betti(DF.projective_resolution(enc1.M, res_min))
    @test bt_wrap == bt_explicit
    end

    @testset "Wrappers for Z^n: injective resolutions and minimal Bass data" begin
    n = 1
    flats = [mk_flat([0],[false]), mk_flat([2],[false])]
    inj   = [mk_inj([4],[false])]
    Phi = [c(1); c(1)] |> x -> reshape(x, 1, 2)
    FG = FZ.Flange{K}(n, flats, inj, Phi)

    enc = TO.EncodingOptions(backend=:zn, max_regions=50_000)
    res = TO.ResolutionOptions(maxlen=3, check=true)

    enc_fg = TO.encode(FG, enc)
    P, M, pi = enc_fg.P, enc_fg.M, enc_fg.pi

    # Compare injective resolutions (wrapper vs explicit encode-then-resolve)
    resI = DF.injective_resolution(M, res)
    resI_Z = DF.injective_resolution_Zn(FG, enc, res)
    @test DF.bass_table(resI_Z) == DF.bass_table(resI)

    # Minimal injective resolutions and minimal Bass invariants
    res_min = TO.ResolutionOptions(maxlen=3, minimal=true, check=true)

    resMinI = DF.injective_resolution(M, res_min)
    resMinI_Z = DF.injective_resolution_Zn(FG, enc, res_min)
    @test DF.bass_table(resMinI_Z) == DF.bass_table(resMinI)

    @test DF.bass(resMinI_Z) == DF.bass(resMinI)
    end

    @testset "FlangeZn: IndFlat/IndInj constructors (new API)" begin
    tau = FZ.face(2, [2])

    # Canonical constructor order: (tau, b; id=...)
    F = FZ.IndFlat(tau, [1, 2])
    @test F isa FZ.IndFlat
    @test F.id == :F
    @test F.b == (1, 2)
    @test F.tau == tau

    E = FZ.IndInj(tau, [3, 4]; id=:E)
    @test E isa FZ.IndInj
    @test E.id == :E
    @test E.b == (3, 4)
    @test E.tau == tau

    # The library intentionally does NOT support reordered positional arguments
    # (both inputs are vectors, so such constructors were a source of subtle bugs).
    @test_throws MethodError FZ.IndFlat([1, 2], tau, :F2)
    @test_throws MethodError FZ.IndInj([3, 4], tau, :E2)
    end

    @testset "FlangeZn UX surface" begin
    tau = FZ.face(2, [1])
    F = FZ.IndFlat(tau, [1, 2]; id=:F1)
    E = FZ.IndInj(tau, [3, 4]; id=:E1)
    FG = FZ.Flange{K}(2, [F], [E], reshape(K[c(1)], 1, 1); field=field)

    tau_desc = TamerOp.describe(tau)
    @test tau_desc.kind == :face
    @test tau_desc.ambient_dim == 2
    @test tau_desc.free_coordinates == [1]

    flat_desc = TamerOp.describe(F)
    @test flat_desc.kind == :flat
    @test flat_desc.translation == (1, 2)
    @test flat_desc.free_coordinates == [1]

    inj_desc = TamerOp.describe(E)
    @test inj_desc.kind == :injective
    @test inj_desc.translation == (3, 4)
    @test inj_desc.free_coordinates == [1]

    flange_desc = TamerOp.describe(FG)
    @test flange_desc.kind == :flange
    @test flange_desc.ambient_dim == 2
    @test flange_desc.nflats == 1
    @test flange_desc.ninjectives == 1
    @test flange_desc.matrix_size == (1, 1)

    @test FZ.matrix_size(FG) == (1, 1)
    @test FZ.generator_counts(FG) == (; flats=1, injectives=1)
    @test FZ.is_minimized(FG)
    @test FZ.active_counts(FG, (2, 3)) == (; injectives=1, flats=1)
    @test FZ.field(FG) === field

    summary = FZ.flange_summary(FG)
    @test summary.kind == :flange
    @test summary.generator_counts == (; flats=1, injectives=1)
    @test summary.is_minimized == true

    qsummary = FZ.flange_query_summary(FG, (2, 3))
    @test qsummary.kind == :flange_query
    @test qsummary.point == (2, 3)
    @test qsummary.active_counts == (; injectives=1, flats=1)
    @test qsummary.local_matrix_size == (1, 1)
    @test qsummary.dimension == 1

    point_ok = FZ.check_flange_point(FG, (2, 3))
    @test point_ok.valid
    @test point_ok.ambient_dim == 2
    point_bad = FZ.check_flange_point(FG, [2])
    @test !point_bad.valid
    @test occursin("expected ambient dimension 2", join(point_bad.issues, " "))
    @test_throws ArgumentError FZ.check_flange_point(FG, [2]; throw=true)

    points_ok = FZ.check_flange_points(FG, [(1, 2), (2, 3)]; sweep=:auto)
    @test points_ok.valid
    points_box = FZ.check_flange_points(FG, [(1, 2), (2, 2), (1, 3), (2, 3)]; sweep=:box)
    @test points_box.valid
    points_bad = FZ.check_flange_points(FG, [(1, 2), (3, 5)]; sweep=:box)
    @test !points_bad.valid
    @test occursin("full axis-aligned integer box", join(points_bad.issues, " "))
    @test_throws ArgumentError FZ.check_flange_points(FG, [(1, 2), (3, 5)]; sweep=:box, throw=true)

    @test FZ.check_face(tau).valid
    @test FZ.check_indflat(F).valid
    @test FZ.check_indinj(E).valid
    flange_ok = FZ.check_flange(FG)
    @test flange_ok.valid
    @test flange_ok.matrix_size == (1, 1)
    @test flange_ok.generator_counts == (; flats=1, injectives=1)
    @test sprint(show, FZ.flange_validation_summary(flange_ok)) ==
          "FlangeValidationSummary(kind=flange, valid=true, issues=0)"
    @test occursin("valid: true", sprint(show, MIME("text/plain"), FZ.flange_validation_summary(flange_ok)))

    tau_bad = FZ.Face(2, BitVector([true]))
    @test !FZ.check_face(tau_bad).valid
    @test_throws ArgumentError FZ.check_face(tau_bad; throw=true)

    @test TOA.Face === FZ.Face
    @test TOA.IndFlat === FZ.IndFlat
    @test TOA.IndInj === FZ.IndInj
    @test TOA.Flange === FZ.Flange
    @test TOA.ambient_dim(tau) == 2
    @test TOA.ambient_dim(F) == 2
    @test TOA.ambient_dim(E) == 2
    @test TOA.ambient_dim(FG) == 2
    @test TOA.flats(FG) === FG.flats
    @test TOA.injectives(FG) === FG.injectives
    @test TOA.coefficient_matrix(FG) === FG.phi
    @test TOA.free_coordinates(tau) == [1]
    @test TOA.translation(F) == (1, 2)
    @test TOA.translation(E) == (3, 4)
    @test TOA.face(F) === tau
    @test TOA.face(E) === tau
    @test TOA.nflats(FG) == 1
    @test TOA.ninjectives(FG) == 1
    @test TOA.matrix_size(FG) == (1, 1)
    @test TOA.generator_counts(FG) == (; flats=1, injectives=1)
    @test TOA.is_minimized(FG)
    @test TOA.active_counts(FG, (2, 3)) == (; injectives=1, flats=1)
    @test TOA.field(FG) === field
    @test TOA.check_face(tau).valid
    @test TOA.check_indflat(F).valid
    @test TOA.check_indinj(E).valid
    @test TOA.check_flange(FG).valid
    @test TOA.check_flange_point(FG, (2, 3)).valid
    @test TOA.check_flange_points(FG, [(1, 2), (2, 2), (1, 3), (2, 3)]; sweep=:box).valid
    @test TOA.flange_summary(FG).kind == :flange
    @test TOA.flange_query_summary(FG, (2, 3)).dimension == 1
    @test TOA.flange_validation_summary(flange_ok) isa FZ.FlangeValidationSummary

    @test occursin("Face(n=2", sprint(show, tau))
    @test occursin("IndFlat(", sprint(show, F))
    @test occursin("IndInj(", sprint(show, E))
    @test occursin("Flange(n=2", sprint(show, FG))
    @test occursin("ambient_dim = 2", sprint(show, MIME("text/plain"), FG))
    @test occursin("matrix_size = (1, 1)", sprint(show, MIME("text/plain"), FG))
    end

    @testset "ZnEncoding UX surface" begin
    tau0 = FZ.Face(1, [false])
    F = FZ.IndFlat(tau0, [0]; id=:F1)
    E = FZ.IndInj(tau0, [2]; id=:E1)
    FG = FZ.Flange{K}(1, [F], [E], reshape(K[c(1)], 1, 1); field=field)
    opts = TO.EncodingOptions(backend=:zn, max_regions=128)

    Psig, _, pi = ZE.encode_from_flange(FG, opts; poset_kind=:signature)
    Pdense, _, pidense = ZE.encode_from_flange(FG, opts; poset_kind=:dense)
    cache_sig = ZE.compile_zn_cache(Psig, pi)
    cache_dense = ZE.compile_zn_cache(Pdense, pidense)

    pi_desc = TamerOp.describe(pi)
    @test pi_desc.kind == :zn_encoding_map
    @test pi_desc.ambient_dim == 1
    @test pi_desc.nregions == FF.nvertices(Psig)
    @test pi_desc.generator_counts == (; flats=1, injectives=1)
    @test pi_desc.poset_kind == :signature
    @test pi_desc.critical_coordinate_counts == (1,)

    P_desc = TamerOp.describe(Psig)
    @test P_desc.kind == :signature_poset
    @test P_desc.nregions == FF.nvertices(Psig)
    @test P_desc.generator_counts == (; flats=1, injectives=1)
    @test P_desc.poset_kind == :signature

    cache_desc = TamerOp.describe(cache_sig)
    @test cache_desc.kind == :zn_encoding_cache
    @test cache_desc.poset_kind == :signature
    dense_cache_desc = TamerOp.describe(cache_dense)
    @test dense_cache_desc.poset_kind == :dense

    @test FF.poset_equal(ZE.region_poset(pi), Psig)
    @test ZE.region_poset(cache_sig) === Psig
    @test ZE.region_poset(cache_dense) === Pdense
    @test ZE.nregions(pi) == length(pi.reps)
    @test ZE.critical_coordinates(pi) == pi.coords
    @test ZE.critical_coordinate_counts(pi) == (length(pi.coords[1]),)
    @test ZE.region_representatives(pi) === pi.reps
    @test ZE.region_representative(pi, 1) == pi.reps[1]
    sig1 = ZE.region_signature(pi, 1)
    @test sig1.y == collect(Bool, pi.sig_y[1])
    @test sig1.z == collect(Bool, pi.sig_z[1])
    @test ZE.flats(pi) === pi.flats
    @test ZE.injectives(pi) === pi.injectives
    @test ZE.generator_counts(pi) == (; flats=1, injectives=1)
    @test ZE.cell_shape(pi) == pi.cell_shape
    @test ZE.has_direct_lookup(pi) == (pi.cell_to_region !== nothing)
    @test ZE.poset_kind(pi) == :signature
    @test ZE.poset_kind(cache_sig) == :signature
    @test ZE.poset_kind(cache_dense) == :dense

    zsummary = ZE.zn_encoding_summary(pi)
    @test zsummary.kind == :zn_encoding_map
    @test zsummary.nregions == length(pi.reps)
    qsummary = ZE.zn_query_summary(pi, (0,))
    @test qsummary.kind == :zn_query
    @test qsummary.region >= 1
    @test qsummary.representative == ZE.region_representative(pi, qsummary.region)
    @test qsummary.signature_support_sizes isa NamedTuple
    @test qsummary.direct_lookup_enabled == ZE.has_direct_lookup(pi)
    qoutside = ZE.zn_query_summary(pi, (-10,))
    @test qoutside.outside
    rsummary = ZE.zn_region_summary(pi, qsummary.region)
    @test rsummary.kind == :zn_region
    @test rsummary.region == qsummary.region
    @test rsummary.representative == qsummary.representative
    @test rsummary.signature_support_sizes == qsummary.signature_support_sizes

    box_ok = ZE.check_zn_box(pi, ([-1], [3]))
    @test box_ok.valid
    @test box_ok.total_points == BigInt(5)
    box_bad = ZE.check_zn_box(pi, ([-1, 0], [3, 0]))
    @test !box_bad.valid
    @test occursin("ambient dimension 1", join(box_bad.issues, " "))
    @test_throws ArgumentError ZE.check_zn_box(pi, ([-1, 0], [3, 0]); throw=true)

    map_report = ZE.check_zn_encoding_map(pi)
    @test map_report.valid
    @test map_report.nregions == ZE.nregions(pi)
    @test ZE.check_signature_poset(Psig).valid
    cache_report = ZE.check_zn_cache(cache_sig)
    @test cache_report.valid
    @test cache_report.poset_kind == :signature
    @test sprint(show, ZE.zn_encoding_validation_summary(cache_report)) ==
          "ZnEncodingValidationSummary(kind=zn_encoding_cache, valid=true, issues=0)"
    @test occursin("valid: true", sprint(show, MIME("text/plain"), ZE.zn_encoding_validation_summary(cache_report)))

    @test ZE.check_zn_query_point(pi, (0,)).valid
    @test ZE.check_zn_query_point(pi, [0.0]).valid
    point_bad = ZE.check_zn_query_point(pi, (0, 1))
    @test !point_bad.valid
    @test occursin("ambient dimension 1", join(point_bad.issues, " "))
    @test_throws ArgumentError ZE.check_zn_query_point(pi, (0, 1); throw=true)

    Xint = reshape(Int[0, 1], 1, :)
    Xfloat = reshape([0.0, 1.0], 1, :)
    Xbad = reshape(Rational{Int}[0//1, 1//1], 1, :)
    @test ZE.check_zn_query_matrix(pi, Xint).valid
    @test ZE.check_zn_query_matrix(cache_sig, Xfloat).valid
    @test !ZE.check_zn_query_matrix(pi, Xbad).valid
    @test_throws ArgumentError ZE.check_zn_query_matrix(pi, Xbad; throw=true)

    @test TOA.ZnEncodingMap === ZE.ZnEncodingMap
    @test TOA.SignaturePoset === ZE.SignaturePoset
    @test TOA.ZnEncodingCache === ZE.ZnEncodingCache
    @test TOA.compile_zn_cache === ZE.compile_zn_cache
    @test FF.poset_equal(TOA.region_poset(pi), Psig)
    @test TOA.nregions(pi) == ZE.nregions(pi)
    @test TOA.critical_coordinates(pi) == pi.coords
    @test TOA.critical_coordinate_counts(pi) == ZE.critical_coordinate_counts(pi)
    @test TOA.region_representatives(pi) === pi.reps
    @test TOA.region_representative(pi, 1) == pi.reps[1]
    @test TOA.region_signature(pi, 1) == sig1
    @test TOA.flats(pi) === pi.flats
    @test TOA.injectives(pi) === pi.injectives
    @test TOA.generator_counts(pi) == ZE.generator_counts(pi)
    @test TOA.cell_shape(pi) == pi.cell_shape
    @test TOA.has_direct_lookup(pi) == ZE.has_direct_lookup(pi)
    @test TOA.poset_kind(cache_dense) == :dense
    @test TOA.zn_encoding_summary(pi).kind == :zn_encoding_map
    @test TOA.zn_query_summary(pi, (0,)).region == qsummary.region
    @test TOA.zn_region_summary(pi, qsummary.region).region == qsummary.region
    @test TOA.check_zn_box(pi, ([-1], [3])).valid
    @test TOA.check_zn_encoding_map(pi).valid
    @test TOA.check_signature_poset(Psig).valid
    @test TOA.check_zn_cache(cache_sig).valid
    @test TOA.check_zn_query_point(pi, (0,)).valid
    @test TOA.check_zn_query_matrix(pi, Xint).valid
    @test TOA.zn_encoding_validation_summary(cache_report) isa ZE.ZnEncodingValidationSummary

    @test occursin("ZnEncodingMap(n=1", sprint(show, pi))
    @test occursin("nregions = ", sprint(show, MIME("text/plain"), pi))
    @test occursin("SignaturePoset(", sprint(show, Psig))
    @test occursin("ZnEncodingCache(", sprint(show, cache_sig))
    @test occursin("poset_kind = :dense", sprint(show, MIME("text/plain"), cache_dense))
    end

    @testset "FlangeZn dim_at + minimize invariance" begin
    # n = 1 interval [b,c] via flat (>= b) and injective (<= c)
    n = 1
    tau0 = FZ.Face(n, [false])
    b = 1
    cmax = 3
    F1 = FZ.IndFlat(tau0, [b]; id=:F1)
    E1 = FZ.IndInj(tau0, [cmax]; id=:E1)
    Phi = reshape(K[c(1)], 1, 1)
    FG = FZ.Flange{K}(n, [F1], [E1], Phi)

    # dim is 1 on [b,c], 0 otherwise
    for g in (b-2):(cmax+2)
        d_fast = FZ.dim_at(FG, (g,))
        d = FZ.dim_at(FG, [g]; rankfun=A -> TamerOp.FieldLinAlg.rank(field, A))
        expected = (b <= g <= cmax) ? 1 : 0
        @test d_fast == expected
        @test d == expected
        @test d_fast == d
    end

    @test (@inferred FZ.dim_at(FG, (b,))) isa Int

    # intersects should detect empty intersection when b > c
    F_bad = FZ.IndFlat(tau0, [5]; id=:Fbad)
    E_bad = FZ.IndInj(tau0, [2]; id=:Ebad)
    @test FZ.intersects(F_bad, E_bad) == false
    FG_bad = FZ.Flange{K}(n, [F_bad], [E_bad], reshape(K[c(1)], 1, 1))
    @test FG_bad.phi[1, 1] == zero(K)

    # Minimize should merge proportional duplicate columns without changing dim_at
    F2 = FZ.IndFlat(tau0, [b]; id=:F2)
    Phi2 = reshape(K[c(1), c(2)], 1, 2)    # second column is 2x the first
    FG2 = FZ.Flange{K}(n, [F1, F2], [E1], Phi2)
    FG2m = FZ.minimize(FG2)
    @test (@inferred FZ.minimize(FG2)) isa FZ.Flange

    for g in (b-1):(cmax+1)
        d0 = FZ.dim_at(FG2, (g,))
        d1 = FZ.dim_at(FG2, [g];  rankfun=A -> TamerOp.FieldLinAlg.rank(field, A))
        d2 = FZ.dim_at(FG2m, [g]; rankfun=A -> TamerOp.FieldLinAlg.rank(field, A))
        @test d0 == d1
        @test d1 == d2
    end

    @testset "degree_matrix! parity and buffer reuse" begin
    n = 2
    tau_x = mk_face(n, [false, true])
    tau_y = mk_face(n, [true, false])
    flats = [
        FZ.IndFlat(tau_x, [0, 0]; id=:Fx0),
        FZ.IndFlat(tau_x, [2, 0]; id=:Fx2),
        FZ.IndFlat(tau_y, [0, 1]; id=:Fy1),
    ]
    injectives = [
        FZ.IndInj(tau_x, [4, 0]; id=:Ex4),
        FZ.IndInj(tau_x, [3, 0]; id=:Ex3),
        FZ.IndInj(tau_y, [0, 3]; id=:Ey3),
    ]
    Phi = reshape([c(1), c(2), c(0),
                   c(0), c(1), c(1),
                   c(1), c(0), c(2)], 3, 3)
    H = FZ.Flange{K}(n, flats, injectives, Phi)

    g1 = (1, 1)
    g2 = (3, 2)

    # degree_matrix! with caller-owned buffers matches allocating degree_matrix.
    rows_buf = Int[]
    cols_buf = Int[]
    A1, r1, c1 = FZ.degree_matrix!(rows_buf, cols_buf, H, g1)
    A1_alloc, r1_alloc, c1_alloc = FZ.degree_matrix(H, g1)
    @test r1 === rows_buf
    @test c1 === cols_buf
    @test r1 == r1_alloc
    @test c1 == c1_alloc
    @test Matrix(A1) == Matrix(A1_alloc)
    @test r1_alloc !== rows_buf
    @test c1_alloc !== cols_buf

    # A second call reuses the same buffers and refreshes contents.
    A2, r2, c2 = FZ.degree_matrix!(rows_buf, cols_buf, H, g2)
    A2_alloc, r2_alloc, c2_alloc = FZ.degree_matrix(H, g2)
    @test r2 === rows_buf
    @test c2 === cols_buf
    @test r2 == r2_alloc
    @test c2 == c2_alloc
    @test Matrix(A2) == Matrix(A2_alloc)

    # Scratch variant reuses per-thread buffers and stays parity-correct.
    A3, r3, c3 = FZ.degree_matrix!(H, g1)
    A3_alloc, r3_alloc, c3_alloc = FZ.degree_matrix(H, g1)
    @test r3 == r3_alloc
    @test c3 == c3_alloc
    @test Matrix(A3) == Matrix(A3_alloc)

    A4, r4, c4 = FZ.degree_matrix!(H, g2)
    A4_alloc, r4_alloc, c4_alloc = FZ.degree_matrix(H, g2)
    @test r4 == r4_alloc
    @test c4 == c4_alloc
    @test Matrix(A4) == Matrix(A4_alloc)
    @test r4 === r3
    @test c4 === c3
    end

    @testset "FlangeDimCache + dim_at_many parity" begin
    n = 2
    tau_x = mk_face(n, [false, true])
    tau_y = mk_face(n, [true, false])
    flats = [
        FZ.IndFlat(tau_x, [0, 0]; id=:Fx0),
        FZ.IndFlat(tau_x, [2, 0]; id=:Fx2),
        FZ.IndFlat(tau_y, [0, 1]; id=:Fy1),
    ]
    injectives = [
        FZ.IndInj(tau_x, [4, 0]; id=:Ex4),
        FZ.IndInj(tau_x, [3, 0]; id=:Ex3),
        FZ.IndInj(tau_y, [0, 3]; id=:Ey3),
    ]
    Phi = reshape([c(1), c(2), c(0),
                   c(0), c(1), c(1),
                   c(1), c(0), c(2)], 3, 3)
    H = FZ.Flange{K}(n, flats, injectives, Phi)

    points = [(0, 0), (1, 1), (2, 1), (1, 1), (3, 2), (0, 0), (3, 2), (2, 1)]
    d_ref = [FZ.dim_at(H, p) for p in points]

    cache = FZ.FlangeDimCache(H)
    d_cached = [FZ.dim_at(H, p; cache=cache) for p in points]
    @test d_cached == d_ref
    @test cache.hits > 0
    @test cache.misses > 0
    @test cache.nentries > 0

    empty!(cache)
    @test cache.nentries == 0
    @test cache.hits == 0
    @test cache.misses == 0
    @test [FZ.dim_at(H, p; cache=cache) for p in points] == d_ref
    @test cache.hits > 0
    @test cache.misses > 0

    bounded_cache = FZ.FlangeDimCache(H; max_entries=1)
    @test [FZ.dim_at(H, p; cache=bounded_cache) for p in points] == d_ref
    @test bounded_cache.nentries <= 1

    probe_cache = FZ.FlangeDimCache(H)
    rows = Int[]
    cols = Int[]
    sizehint!(rows, probe_cache.kernel.ninj)
    sizehint!(cols, probe_cache.kernel.nflat)
    for p in ((0, 0), (2, 1), (3, 2))
        nr_old = length(FZ._fill_active_injectives_kernel!(rows, probe_cache.row_words, probe_cache.kernel, p))
        old_rows = copy(rows)
        nc_old = length(FZ._fill_active_flats_kernel!(cols, probe_cache.col_words, probe_cache.kernel, p))
        old_cols = copy(cols)

        nr_new = FZ._fill_active_injectives_words_kernel!(probe_cache.row_words, probe_cache.kernel, p)
        nc_new = FZ._fill_active_flats_words_kernel!(probe_cache.col_words, probe_cache.kernel, p)
        FZ._decode_active_words!(probe_cache.rows, probe_cache.row_words, probe_cache.kernel.ninj)
        FZ._decode_active_words!(probe_cache.cols, probe_cache.col_words, probe_cache.kernel.nflat)

        @test nr_new == nr_old
        @test nc_new == nc_old
        @test probe_cache.rows == old_rows
        @test probe_cache.cols == old_cols
    end

    out_unsorted = Vector{Int}(undef, length(points))
    out_sorted = Vector{Int}(undef, length(points))
    FZ.dim_at_many!(out_unsorted, H, points; cache=FZ.FlangeDimCache(H), sort_points=false)
    FZ.dim_at_many!(out_sorted, H, points; cache=FZ.FlangeDimCache(H), sort_points=true)
    @test out_unsorted == d_ref
    @test out_sorted == d_ref
    @test FZ.dim_at_many(H, points; cache=FZ.FlangeDimCache(H), sort_points=true) == d_ref

    points_vec = [[p[1], p[2]] for p in points]
    @test FZ.dim_at_many(H, points_vec; cache=FZ.FlangeDimCache(H), sort_points=true) == d_ref
    @test FZ.dim_at_many(H, points; dedup=false, sort_points=false, sweep=:none) == d_ref
    @test FZ.dim_at_many(H, points; dedup=true, sort_points=false, sweep=:none) == d_ref
    @test_throws ErrorException FZ.dim_at_many(H, points; sweep=:box)

    box_points = [(x, y) for y in 0:2 for x in 0:3]
    d_box_ref = [FZ.dim_at(H, p) for p in box_points]
    @test FZ.dim_at_many(H, box_points; sweep=:box, dedup=true, sort_points=false) == d_box_ref

    box_dup = vcat(box_points, box_points[3:6], box_points[1:2])
    d_box_dup_ref = [FZ.dim_at(H, p) for p in box_dup]
    @test FZ.dim_at_many(H, box_dup; sweep=:box, dedup=true, sort_points=false) == d_box_dup_ref

    slab_points = Tuple{Int,Int}[]
    for y in (0, 2, 4, 6), x in 0:255
        push!(slab_points, (x, y))
    end
    @test FZ._dense_slab_groups(unique(slab_points)) !== nothing
    d_slab_ref = FZ.dim_at_many(H, slab_points; sweep=:none, dedup=true, sort_points=true)
    @test FZ.dim_at_many(H, slab_points; sweep=:auto, dedup=true, sort_points=true) == d_slab_ref

    line_points = [(x, 2) for x in 0:63]
    line_ref = [FZ.dim_at(H, p) for p in line_points]
    line_cache = FZ.FlangeDimCache(H)
    line_vals = fill(-1, length(line_points))
    line_scratch = FZ._box_sweep_scratch(line_cache.kernel, length(line_points))
    @inbounds for i in eachindex(line_points)
        line_scratch.line_uids[i] = i
    end
    FZ._eval_unique_line_sweep!(line_vals, H, line_cache, line_scratch,
                                line_scratch.line_uids, 0, 63, (2,))
    @test line_vals == line_ref

    slab_points_y = Tuple{Int,Int}[]
    for x in (0, 2, 4, 6), y in 0:255
        push!(slab_points_y, (x, y))
    end
    @test FZ._dense_slab_groups(unique(slab_points_y); axis=1) === nothing
    @test FZ._dense_slab_groups(unique(slab_points_y); axis=2) !== nothing
    @test 2 in FZ._sweep_axes_by_cost(line_cache.kernel)
    d_slab_y_ref = FZ.dim_at_many(H, slab_points_y; sweep=:none, dedup=true, sort_points=true)
    @test FZ.dim_at_many(H, slab_points_y; sweep=:auto, dedup=true, sort_points=true) == d_slab_y_ref

    line_points_y = [(2, y) for y in 0:63]
    line_y_ref = [FZ.dim_at(H, p) for p in line_points_y]
    line_y_vals = fill(-1, length(line_points_y))
    line_y_scratch = FZ._box_sweep_scratch(line_cache.kernel, length(line_points_y))
    @inbounds for i in eachindex(line_points_y)
        line_y_scratch.line_uids[i] = i
    end
    FZ._eval_unique_line_sweep!(line_y_vals, H, line_cache, line_y_scratch,
                                line_y_scratch.line_uids, 0, 63, (2,); axis=2)
    @test line_y_vals == line_y_ref

    if Threads.nthreads() > 1
        @test FZ.dim_at_many(H, points; threaded=true, dedup=true, sweep=:none) == d_ref
        @test FZ.dim_at_many(H, box_dup; threaded=true, dedup=true, sweep=:box) == d_box_dup_ref
        @test FZ.dim_at_many(H, slab_points; threaded=true, dedup=true, sweep=:auto, sort_points=true) == d_slab_ref
        @test FZ.dim_at_many(H, slab_points_y; threaded=true, dedup=true, sweep=:auto, sort_points=true) == d_slab_y_ref
    end

    H_other = FZ.Flange{K}(n, flats, injectives, Phi .+ c(1))
    @test_throws ErrorException FZ.dim_at(H_other, points[1]; cache=cache)
    end

        if field isa CM.QQField
        @testset "canonical_matrix / degree_matrix / bounding_box" begin
        # 1D: flat is x >= 1, inj is x <= 3.
        flats = [FZ.IndFlat(mk_face(length([1]), [false]), [1]; id=:F)]
        injectives = [FZ.IndInj(mk_face(length([3]), [false]), [3]; id=:E)]
        Phi = FZ.canonical_matrix(flats, injectives)
        @test Phi == reshape(K[1], 1, 1)

        # Non-intersecting pair: x >= 5 and x <= 3.
        flats_bad = [FZ.IndFlat(mk_face(length([5]), [false]), [5]; id=:Fbad)]
        injectives_bad = [FZ.IndInj(mk_face(length([3]), [false]), [3]; id=:Ebad)]
        Phibad = FZ.canonical_matrix(flats_bad, injectives_bad)
        @test Phibad == reshape(K[0], 1, 1)

        # degree_matrix should pick out exactly the active row/col at a given degree.
        Phi2 = reshape(K[2], 1, 1)
        H = FZ.Flange{K}(1, flats, injectives, Phi2)
        Phi_sub, rows, cols = FZ.degree_matrix(H, [2])
        @test rows == [1]
        @test cols == [1]
        @test Phi_sub == reshape(K[2], 1, 1)

        # Outside the intersection, injectives can be inactive while flats remain active.
        Phi_sub2, rows2, cols2 = FZ.degree_matrix(H, [10])
        @test rows2 == Int[]
        @test cols2 == [1]
        @test size(Phi_sub2) == (0, 1)

        # bounding_box in 1D with margin 1:
        #   flats force a >= (b_flat - margin) = 0
        #   injectives force b <= (b_inj + margin) = 4
        a_box, b_box = FZ.bounding_box(H; margin=1)
        @test a_box == [0]
        @test b_box == [4]
        end
        end

        @testset "minimize: do not merge different labels, but merge proportional duplicates" begin
        # Two proportional columns with different underlying flats must not be merged.
        F1 = FZ.IndFlat(mk_face(length([0]), [false]), [0]; id=:F1)
        F2 = FZ.IndFlat(mk_face(length([1]), [false]), [1]; id=:F2)  # different threshold => different upset
        E1 = FZ.IndInj(mk_face(length([2]), [false]), [2]; id=:E1)

        # Columns are proportional but flats differ.
        Phi = K[1 2]
        H = FZ.Flange{K}(1, [F1, F2], [E1], Phi)
        Hmin = FZ.minimize(H)
        @test length(Hmin.flats) == 2

        # Two proportional rows with identical injectives should be merged.
        Fin = [FZ.IndFlat(mk_face(length([0]), [false]), [0]; id=:F)]
        Einj1 = FZ.IndInj(mk_face(length([0]), [false]), [0]; id=:E)
        Einj2 = FZ.IndInj(mk_face(length([0]), [false]), [0]; id=:Edup) # same underlying downset as Einj1
        Phi_rows = reshape(K[1, 2], 2, 1)        # 2x1, proportional rows
        H2 = FZ.Flange{K}(1, Fin, [Einj1, Einj2], Phi_rows)
        H2min = FZ.minimize(H2)
        @test length(H2min.injectives) == 1

        # Rank at degree 0 should be unchanged by minimization.
        @test FZ.dim_at(H2, [0]) == FZ.dim_at(H2min, [0])
        end

        @testset "ZnEncoding: region encoding without enumerating a box" begin
        # FG, b, cmax are in scope here because this testset is nested.
        enc = TO.EncodingOptions(backend=:zn, max_regions=1000)
        P, Henc, pi = TO.ZnEncoding.encode_from_flange(FG, enc)

        # The encoding poset should be a 3-chain:
        #   left of b  <  between  <  right of cmax.
        @test TO.nvertices(P) == 3
        @test Set(FF.cover_edges(P)) == Set([(1,2),(2,3)])
        @test Set(FF.cover_edges(P; cached=false)) == Set(FF.cover_edges(P))

        # In 1D, critical coordinates come from:
        #   flat threshold b
        #   complement(injective) threshold cmax+1
        @test pi.coords[1] == [b, cmax+1]

        # Spot-check the encoded fiber dimensions via locate.
        @test FF.fiber_dimension(Henc, EC.locate(pi, [b-5])) == 0
        @test FF.fiber_dimension(Henc, EC.locate(pi, [b])) == 1
        @test FF.fiber_dimension(Henc, EC.locate(pi, [cmax])) == 1
        @test FF.fiber_dimension(Henc, EC.locate(pi, [cmax+1])) == 0
        end

    @testset "ZnEncoding: direct flange -> fringe (Remark 6.14 bridge)" begin
        # Build the encoding only, then push FG down to a fringe presentation.
        enc = TO.EncodingOptions(backend=:zn, max_regions=1000)
        P, pi = TO.ZnEncoding.encode_poset_from_flanges(FG, enc)
        H = ZE._pushforward_flange_to_fringe(P, pi, FG)   # strict=true by default

        # Compare with the convenience wrapper.
        P2, H2, pi2 = TO.ZnEncoding.encode_from_flange(FG, enc)
        @test TO.nvertices(P) == TO.nvertices(P2)
        @test Set(FF.cover_edges(P)) == Set(FF.cover_edges(P2))

        # Fiber dimensions should match on all sampled degrees.
        for g in (b-5):(cmax+5)
            t  = EC.locate(pi,  [g])
            t2 = EC.locate(pi2, [g])
            @test t == t2
            if t != 0
                @test FF.fiber_dimension(H,  t)  == FF.fiber_dimension(H2, t2)
            end
        end

        # strictness: labels not present in the encoding must be rejected.
        F_extra = FZ.IndFlat(mk_face(length([b + 1]), [false]), [b + 1]; id=:Fextra)
        E = FZ.IndInj(mk_face(length([cmax]), [false]), [cmax]; id=:E)
        FG_extra = FZ.Flange{K}(1, [FZ.IndFlat(mk_face(length([b]), [false]), [b]; id=:F), F_extra], [E], K[1 1])

        @test_throws ErrorException ZE._pushforward_flange_to_fringe(P, pi, FG_extra)
    end
    end

@testset "CrossValidateFlangePL smoke test" begin
    n = 1
    tau0 = FZ.Face(n, [false])
    F1 = FZ.IndFlat(tau0, [1]; id=:F1)
    E1 = FZ.IndInj(tau0, [3]; id=:E1)
    Phi = reshape(K[c(1)], 1, 1)
    FG = FZ.Flange{K}(n, [F1], [E1], Phi)

    ok, report = FZ.cross_validate(FG; margin=1, rankfun=A -> TamerOp.FieldLinAlg.rank(field, A))
    @test ok == true
    @test haskey(report, "mismatches")
    @test isempty(report["mismatches"])
end

@testset "ZnEncoding 2D: free coordinate compression and correctness" begin
    # A 2D flange that depends only on coordinate 1; coordinate 2 is free everywhere.
    flats = [mk_flat([0, 0], [false, true])]
    inj   = [mk_inj([1, 0], [false, true])]
    Phi   = reshape(K[1], 1, 1)
    FG = FZ.Flange{K}(2, flats, inj, Phi)

    enc = TO.EncodingOptions(backend=:zn, max_regions=100)
    enc_fg = TO.encode(FG, enc)
    P, M, pi = enc_fg.P, enc_fg.M, enc_fg.pi

    # Expected: only 3 regions along coordinate 1 (below, inside, above), and 1 slab along coord 2.
    @test TO.nvertices(P) == 3

    # pi should ignore coordinate 2
    for g1 in -3:4
        u = EC.locate(pi, [g1, -10])
        v = EC.locate(pi, [g1,  10])
        @test u == v
    end

    # Monotonicity: g <= h implies pi(g) <= pi(h)
    for g1 in -3:3, h1 in g1:4
        ug = EC.locate(pi, [g1, 0])
        uh = EC.locate(pi, [h1, 0])
        @test FF.leq(P, ug, uh)
    end

    # Dimension consistency on a representative grid of lattice points
    for g1 in -3:4, g2 in (-2, 0, 7)
        g = [g1, g2]
        @test FZ.dim_at(FG, g) == M.dims[EC.locate(pi, g)]
    end
end

@testset "ZnEncoding 2D: common encoding for multiple flanges" begin
    # Two slabs along coordinate 1, coordinate 2 free.
    FG1 = FZ.Flange{K}(2,
        [mk_flat([0,0],[false,true])],
        [mk_inj([1,0],[false,true])],
        reshape(K[1], 1, 1)
    )
    FG2 = FZ.Flange{K}(2,
        [mk_flat([1,0],[false,true])],
        [mk_inj([2,0],[false,true])],
        reshape(K[1], 1, 1)
    )

    enc = TO.EncodingOptions(backend=:zn, max_regions=200)
    encs = TO.encode((FG1, FG2), enc)
    P = encs[1].P
    pi = encs[1].pi
    M1, M2 = encs[1].M, encs[2].M

    # Critical coordinates along g1 are {0,1,2,3} giving <= 5 slabs => P.n <= 5.
    @test TO.nvertices(P) <= 5

    for g1 in -1:4, g2 in (-5, 0, 5)
        g = [g1, g2]
        u = EC.locate(pi, g)
        @test u != 0
        @test FZ.dim_at(FG1, g) == M1.dims[u]
        @test FZ.dim_at(FG2, g) == M2.dims[u]
    end
end

@testset "ZnEncoding advanced cache API: compile_zn_cache + locate_many + box cache" begin
    FG = FZ.Flange{K}(2,
        [mk_flat([0, 0], [false, true])],
        [mk_inj([1, 0], [false, true])],
        reshape(K[1], 1, 1)
    )

    enc = TO.EncodingOptions(backend=:zn, max_regions=200)
    enc_fg = TO.encode(FG, enc)
    P, M, pi_comp = enc_fg.P, enc_fg.M, enc_fg.pi
    pi = pi_comp.pi
    zcache = ZE.compile_zn_cache(P, pi)
    cenc = pi_comp

    Xint = Int[
        -2 -1 0 1 2 3
         7 -3 5 0 9 1
    ]

    # Integer batched locate parity.
    d1 = ZE.locate_many(pi, Xint; threaded=false)
    d2 = ZE.locate_many(zcache, Xint; threaded=true)
    d3 = ZE.locate_many(cenc, Xint; threaded=true)
    @test d1 == d2 == d3
    @test d1 == [EC.locate(pi, Xint[:, j]) for j in 1:size(Xint, 2)]
    @test pi.cell_to_region !== nothing

    # Fast locate path should not depend on signature dictionary lookups when the
    # slab-cell lookup table is available.
    sig_copy = copy(pi.sig_to_region)
    empty!(pi.sig_to_region)
    @test ZE.locate_many(pi, Xint; threaded=false) == d1
    merge!(pi.sig_to_region, sig_copy)

    # Float batched locate parity (round-to-nearest lattice point).
    Xflt = Float64.(Xint) .+ 0.24
    d4 = ZE.locate_many(pi, Xflt; threaded=true)
    @test d4 == [EC.locate(pi, Xflt[:, j]) for j in 1:size(Xflt, 2)]

    # In-place API with destination buffer.
    d5 = fill(-1, size(Xint, 2))
    ZE.locate_many!(d5, zcache, Xint; threaded=true)
    @test d5 == d1

    # Reusable pmodule_on_box basis cache.
    bcache = ZE.compile_zn_box_cache(FG)
    Mbox1 = ZE.pmodule_on_box(FG; a=(-2, -2), b=(3, 3), cache=bcache)
    nentries1 = sum(length(v) for v in values(bcache.basis_cache))
    ntrans1 = length(bcache.transition_cache)
    Mbox2 = ZE.pmodule_on_box(FG; a=(-2, -2), b=(3, 3), cache=bcache)
    nentries2 = sum(length(v) for v in values(bcache.basis_cache))
    ntrans2 = length(bcache.transition_cache)
    @test Mbox1.dims == Mbox2.dims
    @test nentries2 == nentries1
    @test ntrans2 == ntrans1
    @test all(k -> k isa UInt64, keys(bcache.transition_cache))

    st = ZE.box_cache_stats(bcache)
    @test st.basis_entries == nentries2
    @test st.transition_entries == ntrans2
    @test st.basis_cache_hits >= 0
    @test st.transition_cache_misses >= 0

    # Neighboring box should reuse the same cache surface and remain oracle-equivalent.
    Mbox3 = ZE.pmodule_on_box(FG; a=(-1, -2), b=(4, 3), cache=bcache)
    Mbox3_ref = _naive_pmodule_on_box(FG, (-1, -2), (4, 3), field)
    _module_equal_on_covers(Mbox3, Mbox3_ref, field)
end

@testset "ZnEncoding large oracle fixtures (all fields)" begin
    # Fixture 1 (1D): independent interval summands with diagonal Phi.
    # Oracle: dim(g) = number of intervals [b_i, c_i] containing g.
    bs = [0, 25, 50]
    cs = [40, 70, 100]
    flats = [mk_flat([bs[i]], [false]; id=Symbol(:CF, i)) for i in eachindex(bs)]
    injs  = [mk_inj([cs[i]], [false]; id=Symbol(:CE, i)) for i in eachindex(cs)]
    Phi = Matrix{K}(I, length(injs), length(flats))
    FG = FZ.Flange{K}(1, flats, injs, Phi)
    M = ZE.pmodule_on_box(FG; a=(-10,), b=(120,), cache=ZE.compile_zn_box_cache(FG))

    gvals = -10:120
    expected = [count(i -> (bs[i] <= g <= cs[i]), eachindex(bs)) for g in gvals]
    @test M.dims == expected

    # Fixture 2 (2D free-axis): dims are invariant in y for fixed x.
    FG2 = FZ.Flange{K}(2,
        [mk_flat([0, 0], [false, true]), mk_flat([30, 0], [false, true]), mk_flat([60, 0], [false, true])],
        [mk_inj([20, 0], [false, true]), mk_inj([50, 0], [false, true]), mk_inj([90, 0], [false, true])],
        Matrix{K}(I, 3, 3)
    )
    Mxy = ZE.pmodule_on_box(FG2; a=(-5, -40), b=(95, 40), cache=ZE.compile_zn_box_cache(FG2))
    lensx = 95 - (-5) + 1
    lensy = 40 - (-40) + 1
    @test lensx * lensy == length(Mxy.dims)
    for x in (1, div(lensx, 2), lensx)
        d0 = Mxy.dims[x]
        d1 = Mxy.dims[x + div(lensy, 2) * lensx]
        d2 = Mxy.dims[x + (lensy - 1) * lensx]
        @test d0 == d1 == d2
    end
end

@testset "ZnEncoding optimization-branch counters (all fields)" begin
    FG = FZ.Flange{K}(2,
        [mk_flat([0, 0], [false, true]), mk_flat([35, 0], [false, true]), mk_flat([70, 0], [false, true])],
        [mk_inj([25, 0], [false, true]), mk_inj([55, 0], [false, true]), mk_inj([95, 0], [false, true])],
        Matrix{K}(I, 3, 3)
    )
    bcache = ZE.compile_zn_box_cache(FG)
    M1 = ZE.pmodule_on_box(FG; a=(-10, -50), b=(110, 50), cache=bcache)

    @test bcache.basis_cache_misses > 0
    @test bcache.full_basis_recomputes > 0
    @test bcache.incremental_refinements > 0
    @test bcache.transition_cache_misses > 0
    @test length(bcache.transition_cache) > 0

    hits_before = bcache.basis_cache_hits
    thits_before = bcache.transition_cache_hits
    misses_before = bcache.transition_cache_misses
    M2 = ZE.pmodule_on_box(FG; a=(-10, -50), b=(110, 50), cache=bcache)

    @test M2.dims == M1.dims
    @test bcache.basis_cache_hits > hits_before
    @test bcache.transition_cache_hits > thits_before
    @test bcache.transition_cache_misses == misses_before

    nedges = length(collect(FF.cover_edges(M1.Q)))
    @test length(bcache.transition_cache) <= nedges
    @test bcache.full_basis_recomputes + bcache.incremental_refinements >= bcache.basis_cache_misses
end

@testset "ZnEncoding normalized perf guards (QQ)" begin
    if !(field isa CM.QQField)
        @test true
    else
        @inline function _median_elapsed(f::Function; reps::Int=5)
            ts = Vector{Float64}(undef, reps)
            for i in 1:reps
                ts[i] = @elapsed f()
            end
            return sort(ts)[cld(reps, 2)]
        end
        @inline _ns_per_item(t::Float64, n::Int) = (t * 1.0e9) / max(1, n)
        # Local dev runs default to relaxed perf guards unless explicitly forced.
        strict_default = (get(ENV, "CI", "false") == "true") ? "1" : "0"
        strict_ci = get(ENV, "TAMER_STRICT_PERF_CI", strict_default) == "1"

        # --- dim_at kernel guard: auto path should keep allocations and throughput
        # at least on par with explicit submatrix materialization baseline.
        m = 40
        flats_perf = [mk_flat([i - 12, 0], [false, true]; id=Symbol(:PF, i)) for i in 1:m]
        injs_perf  = [mk_inj([i + 18, 0], [false, true]; id=Symbol(:PE, i)) for i in 1:m]
        Phi_perf = Matrix{K}(I, m, m)
        FG_perf = FZ.Flange{K}(2, flats_perf, injs_perf, Phi_perf)
        q_perf = [(x, y) for x in -16:70 for y in -6:6]

        @inline function _dim_at_submatrix_old(g::NTuple{2,Int})
            rows = FZ.active_injectives(FG_perf.injectives, g)
            cols = FZ.active_flats(FG_perf.flats, g)
            if isempty(rows) || isempty(cols)
                return 0
            end
            return TamerOp.FieldLinAlg.rank(field, FG_perf.phi[rows, cols])
        end

        # Warmup compile/runtime paths.
        let s = 0
            @inbounds for g in q_perf
                s += FZ.dim_at(FG_perf, g)
                s += _dim_at_submatrix_old(g)
            end
            @test s >= 0
        end

        alloc_auto = @allocated begin
            s = 0
            @inbounds for g in q_perf
                s += FZ.dim_at(FG_perf, g)
            end
            s
        end
        alloc_old = @allocated begin
            s = 0
            @inbounds for g in q_perf
                s += _dim_at_submatrix_old(g)
            end
            s
        end

        t_auto = _median_elapsed(reps=5) do
            s = 0
            @inbounds for g in q_perf
                s += FZ.dim_at(FG_perf, g)
            end
            @test s >= 0
        end
        t_old = _median_elapsed(reps=5) do
            s = 0
            @inbounds for g in q_perf
                s += _dim_at_submatrix_old(g)
            end
            @test s >= 0
        end

        ns_auto = _ns_per_item(t_auto, length(q_perf))
        ns_old = _ns_per_item(t_old, length(q_perf))
        if strict_ci
            @test ns_auto <= 1.90 * ns_old + 500.0
            @test alloc_auto <= 0.95 * alloc_old + 16_384
        else
            @test ns_auto <= 2.10 * ns_old + 700.0
            @test alloc_auto <= 1.05 * alloc_old + 32_768
        end

        # Buffer-reuse degree_matrix! should be effectively allocation-free post warmup.
        rows_buf = Int[]
        cols_buf = Int[]
        FZ.degree_matrix!(rows_buf, cols_buf, FG_perf, q_perf[1])
        alloc_alloc = @allocated begin
            s = 0
            @inbounds for g in q_perf
                A, rows, cols = FZ.degree_matrix(FG_perf, g)
                s += size(A, 1) + size(A, 2) + length(rows) + length(cols)
            end
            s
        end
        alloc_reuse = @allocated begin
            s = 0
            @inbounds for g in q_perf
                A, rows, cols = FZ.degree_matrix!(rows_buf, cols_buf, FG_perf, g)
                s += size(A, 1) + size(A, 2) + length(rows) + length(cols)
            end
            s
        end
        if strict_ci
            @test alloc_reuse <= 0.35 * alloc_alloc + 4_096
        else
            @test alloc_reuse <= 0.55 * alloc_alloc + 8_192
        end

        # --- Box-encoding kernel guard: cache reuse should not regress vs uncached.
        FG_box = FZ.Flange{K}(2,
            [mk_flat([0, 0], [false, true]), mk_flat([20, 0], [false, true]), mk_flat([40, 0], [false, true])],
            [mk_inj([15, 0], [false, true]), mk_inj([35, 0], [false, true]), mk_inj([55, 0], [false, true])],
            Matrix{K}(I, 3, 3)
        )
        a_box = (-8, -12)
        b_box = (24, 12)
        bcache_perf = ZE.compile_zn_box_cache(FG_box)
        # Warmup
        M_uncached = ZE.pmodule_on_box(FG_box; a=a_box, b=b_box, cache=nothing)
        M_cached = ZE.pmodule_on_box(FG_box; a=a_box, b=b_box, cache=bcache_perf)
        @test M_cached.dims == M_uncached.dims
        ncells = length(M_cached.dims)
        iters = strict_ci ? 16 : 12
        reps_box = strict_ci ? 7 : 5

        t_uncached = _median_elapsed(reps=reps_box) do
            s = 0
            for _ in 1:iters
                M = ZE.pmodule_on_box(FG_box; a=a_box, b=b_box, cache=nothing)
                s += sum(M.dims)
            end
            @test s >= 0
        end
        t_cached = _median_elapsed(reps=reps_box) do
            s = 0
            for _ in 1:iters
                M = ZE.pmodule_on_box(FG_box; a=a_box, b=b_box, cache=bcache_perf)
                s += sum(M.dims)
            end
            @test s >= 0
        end
        ns_uncached = _ns_per_item(t_uncached, iters * ncells)
        ns_cached = _ns_per_item(t_cached, iters * ncells)

        if strict_ci
            @test ns_cached <= 1.50 * ns_uncached + 220.0
        else
            @test ns_cached <= 1.75 * ns_uncached + 320.0
        end
    end
end

@testset "ZnEncoding edge-map matrix oracle fixtures" begin
    # 1D hand-computable oracle with two overlapping intervals:
    # I1=[0,2], I2=[1,3], Phi=I2.
    FG = FZ.Flange{K}(1,
        [mk_flat([0], [false]; id=:F1), mk_flat([1], [false]; id=:F2)],
        [mk_inj([2], [false]; id=:E1), mk_inj([3], [false]; id=:E2)],
        Matrix{K}(I, 2, 2)
    )
    M = ZE.pmodule_on_box(FG; a=(0,), b=(3,), cache=ZE.compile_zn_box_cache(FG))
    @test M.dims == [1, 2, 2, 1]
    @test Set(FF.cover_edges(M.Q)) == Set([(1, 2), (2, 3), (3, 4)])

    A12 = M.edge_maps[1, 2]
    A23 = M.edge_maps[2, 3]
    A34 = M.edge_maps[3, 4]
    E12 = reshape(K[c(1), c(0)], 2, 1)
    E23 = Matrix{K}(I, 2, 2)
    E34 = reshape(K[c(0), c(1)], 1, 2)

    if field isa CM.RealField
        @test isapprox(A12, E12; rtol=1e-10, atol=1e-12)
        @test isapprox(A23, E23; rtol=1e-10, atol=1e-12)
        @test isapprox(A34, E34; rtol=1e-10, atol=1e-12)
    else
        @test A12 == E12
        @test A23 == E23
        @test A34 == E34
    end

    # Composition oracle: map 1->4 should be zero (rank 0).
    @test Inv.rank_map(M, 1, 4) == 0
end

@testset "ZnEncoding randomized differential vs naive reference (medium grids)" begin
    function _assert_fast_vs_naive(FG::FZ.Flange{K},
                                   a::NTuple{2,Int},
                                   b::NTuple{2,Int},
                                   rng::AbstractRNG;
                                   nsample::Int=72) where {K}
        M_fast = ZE.pmodule_on_box(FG; a=a, b=b, cache=ZE.compile_zn_box_cache(FG))
        M_ref  = _naive_pmodule_on_box(FG, a, b, field)

        @test FF.poset_equal(M_fast.Q, M_ref.Q)
        @test M_fast.dims == M_ref.dims

        for (u, v) in FF.cover_edges(M_fast.Q)
            rf = TamerOp.FieldLinAlg.rank(field, M_fast.edge_maps[u, v])
            rr = TamerOp.FieldLinAlg.rank(field, M_ref.edge_maps[u, v])
            @test rf == rr
        end

        for (u, v) in _sample_comparable_pairs(M_fast.Q; rng=rng, nsample=nsample)
            @test Inv.rank_map(M_fast, u, v) == Inv.rank_map(M_ref, u, v)
        end
    end

    if field isa CM.RealField
        # Real-field differential checks are deterministic and numerically stable.
        rng = MersenneTwister(0xA11CE)

        FG1 = FZ.Flange{K}(2,
            [mk_flat([0, 0], [false, true]), mk_flat([20, 0], [false, true]), mk_flat([40, 0], [false, true])],
            [mk_inj([10, 0], [false, true]), mk_inj([30, 0], [false, true]), mk_inj([50, 0], [false, true])],
            Matrix{K}(I, 3, 3)
        )
        _assert_fast_vs_naive(FG1, (-5, -20), (55, 20), rng; nsample=144)

        FG2 = FZ.Flange{K}(2,
            [mk_flat([-3, -2], [false, false]; id=:RF1),
             mk_flat([ 0, -1], [false, false]; id=:RF2),
             mk_flat([ 2,  1], [false, false]; id=:RF3)],
            [mk_inj([ 4,  2], [false, false]; id=:RE1),
             mk_inj([ 5,  3], [false, false]; id=:RE2),
             mk_inj([ 6,  4], [false, false]; id=:RE3)],
            Matrix{K}(I, 3, 3)
        )
        _assert_fast_vs_naive(FG2, (-4, -4), (6, 6), rng; nsample=128)
    else
        seeds = (0xC0FFEE, 0xBAD5EED, 0x1234567)
        for seed in seeds
            rng = MersenneTwister(seed)

            for trial in 1:4
                n = 2
                m = 8
                r = 8

                flats = Vector{FZ.IndFlat{2}}(undef, m)
                injs  = Vector{FZ.IndInj{2}}(undef, r)
                for i in 1:m
                    bvec = [rand(rng, -6:6), rand(rng, -6:6)]
                    cvec = [rand(rng, Bool), rand(rng, Bool)]
                    flats[i] = mk_flat(bvec, cvec; id=Symbol("RF$(seed)_$(trial)_$i"))
                end
                for j in 1:r
                    bvec = [rand(rng, -6:6), rand(rng, -6:6)]
                    cvec = [rand(rng, Bool), rand(rng, Bool)]
                    injs[j] = mk_inj(bvec, cvec; id=Symbol("RE$(seed)_$(trial)_$j"))
                end

                Phi = Matrix{K}(undef, r, m)
                @inbounds for i in 1:r, j in 1:m
                    Phi[i, j] = c(rand(rng, -2:2))
                end
                FG = FZ.Flange{K}(n, flats, injs, Phi)
                _assert_fast_vs_naive(FG, (-5, -5), (5, 5), rng; nsample=128)
            end

            # Adversarial fixture A: repeated/near-repeated columns with shifted windows.
            flatsA = [mk_flat([i - 3, 0], [false, true]; id=Symbol(:AF, i)) for i in 1:8]
            injsA  = [mk_inj([i + 1, 0], [false, true]; id=Symbol(:AE, i)) for i in 1:8]
            PhiA = zeros(K, 8, 8)
            @inbounds for i in 1:8
                PhiA[i, i] = c(1)
                i < 8 && (PhiA[i, i + 1] = c(1))
            end
            FGA = FZ.Flange{K}(2, flatsA, injsA, PhiA)
            _assert_fast_vs_naive(FGA, (-6, -4), (8, 4), rng; nsample=144)

            # Adversarial fixture B: mixed free/locked coordinates creating abrupt signature flips.
            flatsB = [
                mk_flat([-4, -1], [false, false]; id=:BF1),
                mk_flat([-2,  0], [false, true];  id=:BF2),
                mk_flat([ 0, -2], [true,  false]; id=:BF3),
                mk_flat([ 2,  1], [false, false]; id=:BF4),
                mk_flat([ 4,  0], [false, true];  id=:BF5),
                mk_flat([ 1,  3], [true,  false]; id=:BF6),
            ]
            injsB = [
                mk_inj([ 3,  2], [false, false]; id=:BE1),
                mk_inj([ 1,  4], [false, true];  id=:BE2),
                mk_inj([ 5,  1], [true,  false]; id=:BE3),
                mk_inj([ 7,  3], [false, false]; id=:BE4),
                mk_inj([ 6,  0], [false, true];  id=:BE5),
                mk_inj([ 2,  5], [true,  false]; id=:BE6),
            ]
            PhiB = Matrix{K}(undef, 6, 6)
            @inbounds for i in 1:6, j in 1:6
                PhiB[i, j] = c((i == j) ? 1 : ((i + j) % 3 == 0 ? -1 : 0))
            end
            FGB = FZ.Flange{K}(2, flatsB, injsB, PhiB)
            _assert_fast_vs_naive(FGB, (-6, -6), (8, 8), rng; nsample=160)
        end
    end
end

if field isa CM.QQField
@testset "ZnEncoding large oracle fixtures + optimization-branch assertions" begin
    # Fixture 1 (1D, large): independent interval summands with diagonal Phi.
    # Oracle: dim(g) = number of intervals [b_i, c_i] containing g.
    n = 1
    bs = [0, 40, 80, 120]
    cs = [60, 100, 140, 180]
    flats = [mk_flat([bs[i]], [false]; id=Symbol(:F, i)) for i in eachindex(bs)]
    injs  = [mk_inj([cs[i]], [false]; id=Symbol(:E, i)) for i in eachindex(cs)]
    Phi = Matrix{K}(I, length(injs), length(flats))
    FG = FZ.Flange{K}(n, flats, injs, Phi)

    bcache = ZE.compile_zn_box_cache(FG)
    M = ZE.pmodule_on_box(FG; a=(-20,), b=(220,), cache=bcache)

    gvals = -20:220
    expected = [count(i -> (bs[i] <= g <= cs[i]), eachindex(bs)) for g in gvals]
    @test M.dims == expected

    # Branch assertions after first run.
    @test bcache.basis_cache_misses > 0
    @test bcache.full_basis_recomputes > 0
    @test bcache.incremental_refinements > 0
    @test bcache.transition_cache_misses > 0

    hits_before = bcache.basis_cache_hits
    thits_before = bcache.transition_cache_hits
    misses_before = bcache.transition_cache_misses
    M2 = ZE.pmodule_on_box(FG; a=(-20,), b=(220,), cache=bcache)
    @test M2.dims == expected
    @test bcache.basis_cache_hits > hits_before
    @test bcache.transition_cache_hits > thits_before
    @test bcache.transition_cache_misses == misses_before

    # Fixture 2 (2D free-axis): transition-slot reuse should collapse far below #edges.
    FG2 = FZ.Flange{K}(2,
        [mk_flat([0, 0], [false, true]), mk_flat([60, 0], [false, true]), mk_flat([120, 0], [false, true])],
        [mk_inj([40, 0], [false, true]), mk_inj([100, 0], [false, true]), mk_inj([160, 0], [false, true])],
        Matrix{K}(I, 3, 3)
    )
    bcache2 = ZE.compile_zn_box_cache(FG2)
    Mxy = ZE.pmodule_on_box(FG2; a=(-10, -60), b=(190, 60), cache=bcache2)
    nedges = length(collect(FF.cover_edges(Mxy.Q)))
    @test length(bcache2.transition_cache) < max(10, div(nedges, 25))
    @test bcache2.incremental_refinements > 0
    @test bcache2.identity_transition_fastpaths > 0

    # Oracle: because axis 2 is free, dims are invariant along y for fixed x.
    lensx = 190 - (-10) + 1
    lensy = 60 - (-60) + 1
    @test lensx * lensy == length(Mxy.dims)
    for x in (1, div(lensx, 2), lensx)
        d0 = Mxy.dims[x]                            # y = 1
        d1 = Mxy.dims[x + div(lensy, 2) * lensx]   # y ~ middle
        d2 = Mxy.dims[x + (lensy - 1) * lensx]     # y = last
        @test d0 == d1 == d2
    end

    # Fixture 3 (2D, sparse-first solve): high-dimensional sparse identity-like regime.
    k = 96
    flats3 = [mk_flat([0, 0], [false, true]; id=Symbol(:SF, i)) for i in 1:k]
    injs3  = [mk_inj([180, 0], [false, true]; id=Symbol(:SE, i)) for i in 1:k]
    Phi3 = Matrix{K}(I, k, k)
    FG3 = FZ.Flange{K}(2, flats3, injs3, Phi3)
    bcache3 = ZE.compile_zn_box_cache(FG3)
    M3 = ZE.pmodule_on_box(FG3; a=(-5, -20), b=(185, 20), cache=bcache3)

    @test maximum(M3.dims) == k
    @test bcache3.transition_cache_misses > 0
    @test bcache3.sparse_transition_solves +
          bcache3.dense_transition_solves +
          bcache3.identity_transition_fastpaths +
          bcache3.coeff_transition_fastpaths +
          bcache3.basis_change_transition_fastpaths > 0

    # Fixture 4 (1D, fixed rows): coefficient fast path should fire when only columns change.
    flats4 = [mk_flat([0], [false]; id=:CF1), mk_flat([60], [false]; id=:CF2), mk_flat([120], [false]; id=:CF3)]
    injs4  = [mk_inj([10_000], [false]; id=:CE1), mk_inj([10_000], [false]; id=:CE2), mk_inj([10_000], [false]; id=:CE3)]
    Phi4 = Matrix{K}(I, 3, 3)
    FG4 = FZ.Flange{K}(1, flats4, injs4, Phi4)
    bcache4 = ZE.compile_zn_box_cache(FG4)
    M4 = ZE.pmodule_on_box(FG4; a=(-5,), b=(185,), cache=bcache4)
    @test maximum(M4.dims) == 3
    @test bcache4.coeff_transition_fastpaths > 0

    # Fixture 5 (1D, same rows, one-column basis swap): basis-change fast path should fire.
    flats5 = [
        mk_flat([2], [false]; id=:BF1),  # active for x >= 2
        mk_flat([2], [true];  id=:BF2),  # active for x <= 2
        mk_flat([3], [false]; id=:BF3),  # active for x >= 3
    ]
    injs5 = [mk_inj([10_000], [false]; id=:BE1), mk_inj([10_000], [false]; id=:BE2)]
    Phi5 = K[1 0 1;
             0 1 1]
    FG5 = FZ.Flange{K}(1, flats5, injs5, Phi5)
    bcache5 = ZE.compile_zn_box_cache(FG5)
    M5 = ZE.pmodule_on_box(FG5; a=(2,), b=(3,), cache=bcache5)
    @test M5.dims == [2, 2]
    @test bcache5.basis_change_transition_fastpaths > 0
    old_mode = ZE._ZN_BASIS_CHANGE_FASTPATH[]
    try
        ZE._ZN_BASIS_CHANGE_FASTPATH[] = false
        M5slow = ZE.pmodule_on_box(FG5; a=(2,), b=(3,))
        @test M5slow.dims == M5.dims
    finally
        ZE._ZN_BASIS_CHANGE_FASTPATH[] = old_mode
    end
end
end

if field == CM.F2()
@testset "ZnEncoding transition routing is field-sensitive for F2" begin
    k = 48
    flats = [mk_flat([0, 0], [false, true]; id=Symbol(:FF, i)) for i in 1:k]
    injs  = [mk_inj([120, 0], [false, true]; id=Symbol(:FE, i)) for i in 1:k]
    Phi = Matrix{K}(I, k, k)
    FG = FZ.Flange{K}(2, flats, injs, Phi)
    M = ZE.pmodule_on_box(FG; a=(-5, -15), b=(125, 15))
    @test maximum(M.dims) == k
    old_mode = ZE._ZN_TRANSITION_SPARSE_OVERRIDE[]
    try
        ZE._ZN_TRANSITION_SPARSE_OVERRIDE[] = :never
        Mdense = ZE.pmodule_on_box(FG; a=(-5, -15), b=(125, 15))
        @test Mdense.dims == M.dims
    finally
        ZE._ZN_TRANSITION_SPARSE_OVERRIDE[] = old_mode
    end
end
end

@testset "ZnEncoding 2D: strict _pushforward_flange_to_fringe rejects missing generators" begin
    FG1 = FZ.Flange{K}(2,
        [mk_flat([0,0],[false,true])],
        [mk_inj([1,0],[false,true])],
        reshape(K[1], 1, 1)
    )
    FG2 = FZ.Flange{K}(2,
        [mk_flat([10,0],[false,true])],  # new flat label not present in FG1 encoding
        [mk_inj([11,0],[false,true])],
        reshape(K[1], 1, 1)
    )

    enc = TO.EncodingOptions(backend=:zn, max_regions=200)
    P, pi = ZE.encode_poset_from_flanges(FG1, enc)

    @test_throws ErrorException ZE._pushforward_flange_to_fringe(P, pi, FG2; strict=true)

    # Non-strict mode should still push forward, dropping unmatched generators safely.
    H2 = ZE._pushforward_flange_to_fringe(P, pi, FG2; strict=false)
    M2 = IR.pmodule_from_fringe(H2)

    # Sanity: the pushed module is defined on P
    @test length(M2.dims) == TO.nvertices(P)
end

@testset "ZnEncodingMap region_weights: exact methods agree" begin
    # Same 2D setup used in test_znencoding_2d_regions.jl:
    # - one flat at x1 >= 0 (x2 free)
    # - one injective at x1 <= 1 (x2 free)
    flats = [mk_flat([0, 0], [false, true])]
    injs  = [mk_inj([1, 0], [false, true])]
    Phi   = reshape(K[1], 1, 1)
    FG    = FZ.Flange{K}(2, flats, injs, Phi)

    enc = TO.EncodingOptions(backend=:zn, max_regions=100)
    enc_fg = TO.encode(FG, enc)
    P, Henc, pi = enc_fg.P, enc_fg.M, enc_fg.pi

    a = [-2, -3]
    b = [ 3,  4]
    len2 = b[2] - a[2] + 1  # length in free coordinate

    # Determine region indices by locating representative points.
    rid_left  = EC.locate(pi, [-1, 0])  # x1 < 0
    rid_mid   = EC.locate(pi, [ 0, 0])  # 0 <= x1 <= 1
    rid_right = EC.locate(pi, [ 2, 0])  # x1 > 1

    expected = zeros(Int, length(pi.sig_y))
    expected[rid_left]  = 2 * len2  # x1 = -2,-1
    expected[rid_mid]   = 2 * len2  # x1 = 0,1
    expected[rid_right] = 2 * len2  # x1 = 2,3

    w_cells   = TO.RegionGeometry.region_weights(pi; box=(a, b), method=:cells)
    w_points  = TO.RegionGeometry.region_weights(pi; box=(a, b), method=:points)
    w_auto    = TO.RegionGeometry.region_weights(pi; box=(a, b), method=:auto)

    @test w_cells == expected
    @test w_points == expected
    @test w_auto == expected
end

@testset "ZnEncodingMap region_weights: Monte Carlo sampling is close (reproducible)" begin
    flats = [mk_flat([0, 0], [false, true])]
    injs  = [mk_inj([1, 0], [false, true])]
    Phi   = reshape(K[1], 1, 1)
    FG    = FZ.Flange{K}(2, flats, injs, Phi)

    enc = TO.EncodingOptions(backend=:zn, max_regions=100)
    enc_fg = TO.encode(FG, enc)
    P, Henc, pi = enc_fg.P, enc_fg.M, enc_fg.pi

    a = [-50, -500]
    b = [ 49,  499]
    len2 = b[2] - a[2] + 1  # 1000

    rid_left  = EC.locate(pi, [-1, 0])
    rid_mid   = EC.locate(pi, [ 0, 0])
    rid_right = EC.locate(pi, [ 2, 0])

    expected = zeros(Int, length(pi.sig_y))
    expected[rid_left]  = 50 * len2
    expected[rid_mid]   =  2 * len2
    expected[rid_right] = 48 * len2

    # Monte Carlo estimate
    rng = MersenneTwister(0)
    info = TO.RegionGeometry.region_weights(pi; box=(a, b), method=:sample, nsamples=20_000, rng=rng, return_info=true)

    @test info.method == :sample
    @test eltype(info.weights) == Float64
    @test length(info.weights) == length(expected)
    @test info.stderr !== nothing
    @test length(info.stderr) == length(expected)

    # Total points in the box:
    total_points = (b[1] - a[1] + 1) * (b[2] - a[2] + 1)
    @test isapprox(sum(info.weights), total_points; rtol=0.03)

    # Each bin should be reasonably close. Tolerances chosen for test stability.
    for j in eachindex(expected)
        @test isapprox(info.weights[j], expected[j]; rtol=0.06, atol=5_000.0)
    end
end

@testset "ZnEncodingMap region_weights: auto can be forced to sampling" begin
    flats = [mk_flat([0, 0], [false, true])]
    injs  = [mk_inj([1, 0], [false, true])]
    Phi   = reshape(K[1], 1, 1)
    FG    = FZ.Flange{K}(2, flats, injs, Phi)

    enc = TO.EncodingOptions(backend=:zn, max_regions=100)
    enc_fg = TO.encode(FG, enc)
    P, Henc, pi = enc_fg.P, enc_fg.M, enc_fg.pi

    a = [-50, -500]
    b = [ 49,  499]

    rng = MersenneTwister(1)
    info = TO.RegionGeometry.region_weights(pi; box=(a, b), method=:auto, max_cells=0, max_points=0,
                             nsamples=1_000, rng=rng, return_info=true)

    @test info.method == :sample
    @test eltype(info.weights) == Float64
end

@testset "ZnEncodingMap region_weights: :auto count_type promotes to BigInt on overflow" begin
    flats = [mk_flat([0, 0], [false, true])]
    injs  = [mk_inj([1, 0], [false, true])]
    Phi   = reshape(K[1], 1, 1)
    FG    = FZ.Flange{K}(2, flats, injs, Phi)

    enc = TO.EncodingOptions(backend=:zn, max_regions=100)
    enc_fg = TO.encode(FG, enc)
    P, Henc, pi = enc_fg.P, enc_fg.M, enc_fg.pi

    # Choose an interval length > typemax(Int) while endpoints still fit in Int64.
    a = [-9_000_000_000_000_000_000, 0]
    b = [ 9_000_000_000_000_000_000, 0]

    rid_left  = EC.locate(pi, [-1, 0])
    rid_mid   = EC.locate(pi, [ 0, 0])
    rid_right = EC.locate(pi, [ 2, 0])

    expected = zeros(BigInt, length(pi.sig_y))
    expected[rid_left]  = BigInt(-1) - BigInt(a[1]) + 1                 # a1..-1
    expected[rid_mid]   = BigInt(2)                                     # 0..1
    expected[rid_right] = BigInt(b[1]) - BigInt(2) + 1                  # 2..b1

    w = TO.RegionGeometry.region_weights(pi; box=(a, b), method=:cells, count_type=:auto)
    @test eltype(w) == BigInt
    @test w == expected
    @test sum(w) == (BigInt(b[1]) - BigInt(a[1]) + 1) * (BigInt(b[2]) - BigInt(a[2]) + 1)
end

@testset "region_poset(pi) reconstructs the encoder region poset" begin
    # -------------------------------------------------------------------------
    # 1) ZnEncoding: pi has (sig_y, sig_z) but no pi.P; region_poset should match P
    # -------------------------------------------------------------------------
    let
        n = 1
        b = [0]
        c = [5]
        I = FZ.Face(n, Int[])
        flats = [FZ.IndFlat(I, b; id=:F)]
        injectives = [FZ.IndInj(I, c; id=:E)]
        Phi = spzeros(K, 1, 1); Phi[1, 1] = 1
        FG = FZ.Flange{K}(n, flats, injectives, Phi)

        enc = TO.EncodingOptions(backend=:zn, max_regions=1000)
        Penc, Henc, pi = TO.ZnEncoding.encode_from_flange(FG, enc)
        rpcache = CM.EncodingCache()

        Q = TO.Invariants.region_poset(pi; poset_kind = :signature, cache=rpcache)
        @test TO.nvertices(Q) == TO.nvertices(Penc)
        @test FF.poset_equal(Q, Penc)

        # Cached repeat call should return the exact same poset object.
        Q2 = TO.Invariants.region_poset(pi; poset_kind = :signature, cache=rpcache)
        @test Q2 === Q

        Qdense = TO.Invariants.region_poset(pi; poset_kind = :dense, cache=rpcache)
        @test FF.leq_matrix(Q) == FF.leq_matrix(Qdense)

        # Projected arrangement should work without requiring pi.P.
        arr = Inv.projected_arrangement(pi; dirs=[[1.0]])
        @test FF.poset_equal(arr.Q, Penc)

        # And should accept a provided Q for maximum speed.
        arr2 = Inv.projected_arrangement(pi; dirs=[[1.0]], Q=Penc)
        @test FF.poset_equal(arr2.Q, Penc)
    end

    # -------------------------------------------------------------------------
    # 2) PLPolyhedra: PLEncodingMap has (sig_y, sig_z) but no pi.P
    # -------------------------------------------------------------------------
    let
        PLP = TO.PLPolyhedra
        if !PLP.HAVE_POLY
            @test true
        else
            n = 1

            # PLPolyhedra uses H-polytopes { x : A*x <= b }.
            # Upset: x >= 0  <=>  (-x <= 0)
            Uhp = PLP.make_hpoly([-1.0], 0.0)

            # Downset: x <= 2 <=>  (x <= 2)
            Dhp = PLP.make_hpoly([1.0], 2.0)

            U = PLP.PLUpset(PLP.PolyUnion(n, [Uhp]))
            D = PLP.PLDownset(PLP.PolyUnion(n, [Dhp]))

            # PLFringe requires an explicit Phi of size (#Downs) x (#Ups).
            F1 = PLP.PLFringe([U], PLP.PLDownset[], zeros(K, 0, 1))
            F2 = PLP.PLFringe(PLP.PLUpset[], [D], zeros(K, 1, 0))

            enc = TO.EncodingOptions(backend=:pl, max_regions=10_000)
            Ppl, Hpl, pipl = PLP.encode_from_PL_fringes(F1, F2, enc; poset_kind = :signature)
            out_tuple = TO.encode((F1, F2); enc=enc)
            out_vec = TO.encode(PLP.PLFringe[F1, F2]; enc=enc)
            @test length(out_tuple) == 2
            @test length(out_vec) == 2
            @test_throws MethodError TO.encode(F1, F2; enc=enc)
            @test_throws MethodError TO.encode(F1, F2, F1; enc=enc)
            @test Ppl isa TO.ZnEncoding.SignaturePoset
            rpcache = CM.EncodingCache()

            Qpl = TO.Invariants.region_poset(pipl; poset_kind = :signature, cache=rpcache)
            @test TO.nvertices(Qpl) == TO.nvertices(Ppl)
            @test FF.poset_equal(Qpl, Ppl)

            Qpl2 = TO.Invariants.region_poset(pipl; poset_kind = :signature, cache=rpcache)
            @test Qpl2 === Qpl

            Qpl_dense = TO.Invariants.region_poset(pipl; poset_kind = :dense, cache=rpcache)
            @test FF.leq_matrix(Qpl) == FF.leq_matrix(Qpl_dense)

            arr = Inv.projected_arrangement(pipl; dirs=[[1.0]])
            @test FF.poset_equal(arr.Q, Ppl)

            arr2 = Inv.projected_arrangement(pipl; dirs=[[1.0]], Q=Ppl)
            @test FF.poset_equal(arr2.Q, Ppl)
        end
    end


    # -------------------------------------------------------------------------
    # 3) PLBackend: PLEncodingMapBoxes has (sig_y, sig_z) but no pi.P
    # -------------------------------------------------------------------------
    if isdefined(TO, :PLBackend)
        let
            PB = TO.PLBackend
            n = 1
            U = PB.BoxUpset([0.0])
            D = PB.BoxDownset([2.0])

            Ups = [U]
            Downs = [D]

            enc = TO.EncodingOptions(backend=:pl_backend, max_regions=10_000)
            Pbx, Hbx, pibx = PB.encode_fringe_boxes(Ups, Downs, enc; poset_kind = :signature)
            @test Pbx isa TO.ZnEncoding.SignaturePoset
            rpcache = CM.EncodingCache()

            Qbx = TO.Invariants.region_poset(pibx; poset_kind = :signature, cache=rpcache)
            @test TO.nvertices(Qbx) == TO.nvertices(Pbx)
            @test FF.poset_equal(Qbx, Pbx)

            Qbx2 = TO.Invariants.region_poset(pibx; poset_kind = :signature, cache=rpcache)
            @test Qbx2 === Qbx

            Qbx_dense = TO.Invariants.region_poset(pibx; poset_kind = :dense, cache=rpcache)
            @test FF.leq_matrix(Qbx) == FF.leq_matrix(Qbx_dense)

            arr = Inv.projected_arrangement(pibx; dirs=[[1.0]])
            @test FF.poset_equal(arr.Q, Pbx)

            arr2 = Inv.projected_arrangement(pibx; dirs=[[1.0]], Q=Pbx)
            @test FF.poset_equal(arr2.Q, Pbx)

            Mbx = IR.pmodule_from_fringe(Hbx)
            @test Inv.rank_map(Mbx, 1, 1) >= 0
        end
    end
end

if field isa CM.QQField
@testset "encode poset_kind=:dense yields FinitePoset" begin
    if isdefined(TO, :PLBackend)
        PB = TO.PLBackend
        Ups = [PB.BoxUpset([0.0])]
        Downs = [PB.BoxDownset([2.0])]
        enc = TO.EncodingOptions(backend=:pl_backend, max_regions=10_000, poset_kind=:dense)
        P, _H, _pi = TO.encode(Ups, Downs, enc)
        @test P isa TO.FinitePoset
    end

    PLP = TO.PLPolyhedra
    if PLP.HAVE_POLY
        n = 1
        Uhp = PLP.make_hpoly([-1.0], 0.0)
        Dhp = PLP.make_hpoly([1.0], 2.0)
        U = PLP.PLUpset(PLP.PolyUnion(n, [Uhp]))
        D = PLP.PLDownset(PLP.PolyUnion(n, [Dhp]))
        F = PLP.PLFringe([U], [D], ones(K, 1, 1))
        enc = TO.EncodingOptions(backend=:pl, max_regions=10_000, poset_kind=:dense)
        P, _H, _pi = TO.encode(F, enc)
        @test P isa TO.FinitePoset
    end
end
end
end # with_fields
