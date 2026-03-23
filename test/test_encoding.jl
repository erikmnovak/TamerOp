using Test
using SparseArrays
using LinearAlgebra
using Random
using JSON3

# Included from test/runtests.jl.
# Uses shared aliases defined there (TO, FF, EN, IR, DF, MD, QQ, ...).

# A minimal ambient encoding map for workflow coarsen tests.
struct DummyEncodingMap <: EC.AbstractPLikeEncodingMap
    n::Int
end
EC.locate(pi::DummyEncodingMap, x::NTuple{1,Int}) =
    (1 <= x[1] <= pi.n ? x[1] : 0)
EC.locate(pi::DummyEncodingMap, x::AbstractVector{<:Integer}) =
    (length(x) >= 1 && 1 <= x[1] <= pi.n ? Int(x[1]) : 0)
EC.dimension(::DummyEncodingMap) = 1
EC.axes_from_encoding(pi::DummyEncodingMap) = (collect(1:pi.n),)
EC.representatives(pi::DummyEncodingMap) = [(i,) for i in 1:pi.n]

with_fields(FIELDS_FULL) do field
K = CM.coeff_type(field)
@inline c(x) = CM.coerce(field, x)


@testset "Finite encoding from fringe (Defs 4.12-4.18)" begin
    P = chain_poset(3)
    U2 = FF.principal_upset(P, 2)
    D2 = FF.principal_downset(P, 2)

    M = one_by_one_fringe(P, U2, D2)

    enc = EN.build_uptight_encoding_from_fringe(M; poset_kind = :dense)
    pi = enc.pi
    @test enc isa EN.UptightEncoding
    @test eltype(enc.Y) <: FF.Upset{typeof(P)}

    # pi should be order-preserving: i <= j in Q => pi(i) <= pi(j) in P_Y
    for i in 1:pi.Q.n, j in 1:pi.Q.n
        if FF.leq(pi.Q, i, j)
            @test FF.leq(pi.P, pi.pi_of_q[i], pi.pi_of_q[j])
        end
    end

    # Build the induced module on the encoded poset by pushing U,D forward.
    Hhat = EN.pushforward_fringe_along_encoding(M, pi)

    # Pull back and recover the original U,D exactly. This exercises the
    # internal image/preimage transport helpers through public constructors.
    M2 = EN.pullback_fringe_along_encoding(Hhat, pi)
    @test M2.U[1].mask == M.U[1].mask
    @test M2.D[1].mask == M.D[1].mask
    for q in 1:P.n
        @test FF.fiber_dimension(M2, q) == FF.fiber_dimension(M, q)
    end
end

@testset "Encoding push/pull memoization and typed postcomposition cache" begin
    P = chain_poset(4)
    field = CM.QQField()
    K = CM.coeff_type(field)
    U = FF.principal_upset(P, 2)
    D = FF.principal_downset(P, 3)
    Phi = sparse([1, 2], [1, 2], [CM.coerce(field, 1), CM.coerce(field, 2)], 2, 2)
    H = FF.FringeModule{K}(P, [U, U], [D, D], Phi; field=field)

    upt = EN.build_uptight_encoding_from_fringe(H; poset_kind=:regions)
    @test upt.pi.label_cache[] === nothing
    Hpush = EN.pushforward_fringe_along_encoding(H, upt.pi)
    @test Hpush.U[1] === Hpush.U[2]
    @test Hpush.D[1] === Hpush.D[2]
    @test upt.pi.label_cache[] !== nothing

    Hpush2 = EN.pushforward_fringe_along_encoding(H, upt.pi)
    @test Hpush2.U[1] === Hpush.U[1]
    @test Hpush2.D[1] === Hpush.D[1]

    Hpull = EN.pullback_fringe_along_encoding(Hpush, upt.pi)
    @test Hpull.U[1] === Hpull.U[2]
    @test Hpull.D[1] === Hpull.D[2]

    Hpull2 = EN.pullback_fringe_along_encoding(Hpush, upt.pi)
    @test Hpull2.U[1] === Hpull.U[1]
    @test Hpull2.D[1] === Hpull.D[1]

    pi0 = DummyEncodingMap(FF.nvertices(P))
    pic = EN.PostcomposedEncodingMap(pi0, upt.pi)
    reps = EC.representatives(pic)
    @test reps isa Vector{eltype(reps)}
    @test eltype(reps) <: Tuple
end

@testset "Uptight poset uses transitive closure (Example 4.16)" begin
    # Miller Example 4.16 exhibits non-transitivity of the raw "exists a<=b" relation
    # between uptight regions; Definition 4.17 then defines the uptight poset as its
    # transitive closure. We build a finite truncation of the N^2 example to test that
    # Encoding._uptight_poset really closes transitively.

    # Q = {0..3} x {0..2} with product order.
    function grid_poset(ax::Int, by::Int)
        coords = Tuple{Int,Int}[]
        for a in 0:ax, b in 0:by
            push!(coords, (a, b))
        end
        nQ = length(coords)
        leq = falses(nQ, nQ)
        for i in 1:nQ, j in 1:nQ
            ai, bi = coords[i]
            aj, bj = coords[j]
            leq[i, j] = (ai <= aj) && (bi <= bj)
        end
        Q = FF.FinitePoset(leq)
        idx = Dict{Tuple{Int,Int}, Int}()
        for (i, c) in enumerate(coords)
            idx[c] = i
        end
        return Q, coords, idx
    end

    Q, coords, idx = grid_poset(3, 2)

    # Upsets corresponding to the monomial ideals in Example 4.16:
    #   U1 = <x^2, y>    => (a>=2) or (b>=1)
    #   U2 = <x^3, y>    => (a>=3) or (b>=1)
    #   U3 = <x*y>       => (a>=1) and (b>=1)
    #   U4 = <x^2*y>     => (a>=2) and (b>=1)
    U1 = FF.upset_closure(Q, BitVector([(a >= 2) || (b >= 1) for (a, b) in coords]))
    U2 = FF.upset_closure(Q, BitVector([(a >= 3) || (b >= 1) for (a, b) in coords]))
    U3 = FF.upset_closure(Q, BitVector([(a >= 1) && (b >= 1) for (a, b) in coords]))
    U4 = FF.upset_closure(Q, BitVector([(a >= 2) && (b >= 1) for (a, b) in coords]))

    # No deaths: we only need the upsets to form Y for uptight signatures.
    phi0 = spzeros(K, 0, 4)
    M = FF.FringeModule{K}(Q, [U1, U2, U3, U4], FF.Downset[], phi0; field=field)

    enc = EN.build_uptight_encoding_from_fringe(M; poset_kind = :dense)
    pi = enc.pi
    P = pi.P

    # Identify the regions by representative lattice points (degrees):
    #   A: x^2  = (2,0) has signature {U1}
    #   B: x^3  = (3,0) and y = (0,1) share signature {U1,U2}
    #   C: x*y  = (1,1) has signature {U1,U2,U3}
    q_x2 = idx[(2, 0)]
    q_x3 = idx[(3, 0)]
    q_y  = idx[(0, 1)]
    q_xy = idx[(1, 1)]

    A = pi.pi_of_q[q_x2]
    B = pi.pi_of_q[q_x3]
    B2 = pi.pi_of_q[q_y]
    C = pi.pi_of_q[q_xy]

    @test B == B2
    @test length(Set([A, B, C])) == 3

    # In the transitive closure P, we must have A <= B <= C, hence A <= C.
    @test FF.leq(P, A, B)
    @test FF.leq(P, B, C)
    @test FF.leq(P, A, C)

    # But in the underlying "exists a<=c" relation on regions, there is no witness for A <= C:
    # the only point in A is (2,0), and every point in C has x=1, so (2,0) is not <= any c in C.
    has_witness = false
    for a in 1:Q.n, c in 1:Q.n
        if pi.pi_of_q[a] == A && pi.pi_of_q[c] == C && FF.leq(Q, a, c)
            has_witness = true
            break
        end
    end
    @test !has_witness
end

@testset "Encoding push/pull and dense builder parity against scan baseline" begin
    P = chain_poset(5)
    U1 = FF.principal_upset(P, 2)
    U2 = FF.principal_upset(P, 4)
    D1 = FF.principal_downset(P, 3)
    D2 = FF.principal_downset(P, 5)
    Phi = sparse([1, 2], [1, 2], [c(1), c(2)], 2, 2)
    H = FF.FringeModule{K}(P, [U1, U2], [D1, D2], Phi; field=field)

    upt = EN.build_uptight_encoding_from_fringe(H; poset_kind=:regions)
    pi = upt.pi
    regions = EN._uptight_regions(P, upt.Y)

    function pushforward_old(H, pi)
        Uhat = map(H.U) do U
            maskP = falses(FF.nvertices(pi.P))
            for q in 1:FF.nvertices(pi.Q)
                U.mask[q] && (maskP[pi.pi_of_q[q]] = true)
            end
            FF.upset_closure(pi.P, maskP)
        end
        Dhat = map(H.D) do D
            maskP = falses(FF.nvertices(pi.P))
            for q in 1:FF.nvertices(pi.Q)
                D.mask[q] && (maskP[pi.pi_of_q[q]] = true)
            end
            FF.downset_closure(pi.P, maskP)
        end
        FF.FringeModule{eltype(H.phi)}(pi.P, Uhat, Dhat, H.phi; field=H.field)
    end

    function pullback_old(Hhat, pi)
        UQ = map(Hhat.U) do Uhat
            maskQ = falses(FF.nvertices(pi.Q))
            for q in 1:FF.nvertices(pi.Q)
                Uhat.mask[pi.pi_of_q[q]] && (maskQ[q] = true)
            end
            FF.upset_closure(pi.Q, maskQ)
        end
        DQ = map(Hhat.D) do Dhat
            maskQ = falses(FF.nvertices(pi.Q))
            for q in 1:FF.nvertices(pi.Q)
                Dhat.mask[pi.pi_of_q[q]] && (maskQ[q] = true)
            end
            FF.downset_closure(pi.Q, maskQ)
        end
        FF.FringeModule{eltype(Hhat.phi)}(pi.Q, UQ, DQ, Hhat.phi; field=Hhat.field)
    end

    function dense_old(Q, regions)
        r = length(regions)
        rel = falses(r, r)
        for A in 1:r, B in 1:r
            if A == B
                rel[A, B] = true
                continue
            end
            found = false
            for a in regions[A], b in regions[B]
                if FF.leq(Q, a, b)
                    found = true
                    break
                end
            end
            rel[A, B] = found
        end
        for k in 1:r, i in 1:r, j in 1:r
            rel[i, j] = rel[i, j] || (rel[i, k] && rel[k, j])
        end
        FF.FinitePoset(rel; check=false)
    end

    Hpush_old = pushforward_old(H, pi)
    Hpush_new = EN.pushforward_fringe_along_encoding(H, pi)
    @test [u.mask for u in Hpush_old.U] == [u.mask for u in Hpush_new.U]
    @test [d.mask for d in Hpush_old.D] == [d.mask for d in Hpush_new.D]
    @test Hpush_old.phi == Hpush_new.phi

    Hpull_old = pullback_old(Hpush_new, pi)
    Hpull_new = EN.pullback_fringe_along_encoding(Hpush_new, pi)
    @test [u.mask for u in Hpull_old.U] == [u.mask for u in Hpull_new.U]
    @test [d.mask for d in Hpull_old.D] == [d.mask for d in Hpull_new.D]
    @test Hpull_old.phi == Hpull_new.phi

    Pold = dense_old(P, regions)
    Pnew = EN._uptight_poset(P, regions; poset_kind=:dense)
    @test FF.leq_matrix(Pold) == FF.leq_matrix(Pnew)
end


@testset "Workflow coarsen (uptight compression)" begin
    P = chain_poset(4)

    # A tiny 1 times 1 fringe module on the chain.
    U = FF.principal_upset(P, 2)
    D = FF.principal_downset(P, 3)
    Phi = sparse([1], [1], [c(1)], 1, 1)
    H = FF.FringeModule{K}(P, [U], [D], Phi; field=field)

    pi0 = DummyEncodingMap(FF.nvertices(P))

    # Construct an EncodingResult manually.
    enc = RES.EncodingResult(P, IR.pmodule_from_fringe(H), pi0;
        H = H,
        backend = :dummy,
        meta = Dict{Symbol,Any}(),
    )

    enc2 = TO.coarsen(enc)
    enc_no_h = RES.EncodingResult(P, IR.pmodule_from_fringe(H), pi0;
        H = nothing,
        backend = :dummy,
        meta = Dict{Symbol,Any}(),
    )
    enc2_no_h = TO.coarsen(enc_no_h)

    upt = EN.build_uptight_encoding_from_fringe(H)
    pi = upt.pi

    # New region poset size is the (uptight) compression of the old one.
    @test FF.nvertices(enc2.P) == FF.nvertices(pi.P)

    # Classifier should be postcomposition: (q,) <-> pi(q).
    for q in 1:FF.nvertices(P)
        @test EC.locate(enc2.pi, (q,)) == pi.pi_of_q[q]
    end

    # Representatives are forwarded to the coarse regions (and cached).
    reps2 = EC.representatives(enc2.pi)
    @test length(reps2) == FF.nvertices(enc2.P)
    for p in 1:FF.nvertices(enc2.P)
        @test EC.locate(enc2.pi, reps2[p]) == p
    end

    # Fringe module was pushed forward along the coarsening map.
    H2_expected = EN.pushforward_fringe_along_encoding(H, pi)
    @test [u.mask for u in enc2.H.U] == [u.mask for u in H2_expected.U]
    @test [d.mask for d in enc2.H.D] == [d.mask for d in H2_expected.D]
    @test enc2.H.phi == H2_expected.phi
    @test [u.mask for u in enc2_no_h.H.U] == [u.mask for u in H2_expected.U]
    @test [d.mask for d in enc2_no_h.H.D] == [d.mask for d in H2_expected.D]
    @test enc2_no_h.H.phi == H2_expected.phi
end


@testset "JSON serialization round-trips" begin
    expected_field_kind = field isa CM.QQField ? "qq" :
                          field isa CM.RealField ? "real" : "fp"

    # Flange round-trip
    n = 1
    tau0 = FZ.Face(n, [false])
    F1 = FZ.IndFlat(tau0, [1]; id=:F1)
    E1 = FZ.IndInj(tau0, [3]; id=:E1)
    Phi = reshape(K[c(1)], 1, 1)
    FG = FZ.Flange{K}(n, [F1], [E1], Phi)

    mktempdir() do dir
        path = joinpath(dir, "flange.json")
        SER.save_flange_json(path, FG)
        info = SER.inspect_json(path)
        @test info isa SER.JSONArtifactSummary
        @test SER.artifact_kind(info) == "FlangeZn"
        @test SER.artifact_path(info) == path
        @test SER.artifact_field(info) == expected_field_kind
        @test SER.artifact_profile_hint(info) == :compact
        @test SER.artifact_size_bytes(info) isa Integer
        @test SER.flange_json_summary(path).kind == "FlangeZn"
        @test SER.check_flange_json(path).valid
        FG2 = SER.load_flange_json(path)
        FG3 = SER.load_flange_json(path; validation=:trusted)

        @test FG2.n == FG.n
        @test length(FG2.flats) == length(FG.flats)
        @test length(FG2.injectives) == length(FG.injectives)
        @test FG2.phi == FG.phi
        @test FG2.flats[1].b == FG.flats[1].b
        @test FG2.injectives[1].b == FG.injectives[1].b
        @test FG2.flats[1].tau.coords == FG.flats[1].tau.coords
        @test FG2.injectives[1].tau.coords == FG.injectives[1].tau.coords
        @test FG3.phi == FG2.phi
        @test FG3.field == FG2.field

        debug_path = joinpath(dir, "flange_debug.json")
        SER.save_flange_json(debug_path, FG; profile=:debug)
        @test filesize(path) < filesize(debug_path)
        @test SER.inspect_json(debug_path).profile_hint == :debug
    end

    # Flange round-trip with non-QQ field + override coercion
    field2 = CM.F2()
    K2 = CM.coeff_type(field2)
    Phi2 = reshape(K2[CM.coerce(field2, 1)], 1, 1)
    FGf2 = FZ.Flange{K2}(n, [F1], [E1], Phi2; field=field2)
    mktempdir() do dir
        path = joinpath(dir, "flange_f2.json")
        SER.save_flange_json(path, FGf2)
        FGf2_loaded = SER.load_flange_json(path)
        @test FGf2_loaded.field == field2
        @test eltype(FGf2_loaded.phi) == K2
        @test FGf2_loaded.phi[1, 1] == CM.coerce(field2, 1)

        FGf2_asQQ = SER.load_flange_json(path; field=CM.QQField())
        if field isa CM.QQField
            @test eltype(FGf2_asQQ.phi) == QQ
            @test FGf2_asQQ.phi[1, 1] == CM.coerce(CM.QQField(), 1)
        end
    end

    # PL fringe round-trip
    Aup = reshape(QQ[-1], 1, 1)
    bup = QQ[0]
    hup = PLP.HPoly(1, Aup, bup, nothing, falses(1), PLP.STRICT_EPS_QQ)
    Upl = PLP.PLUpset(PLP.PolyUnion(1, [hup]))

    Adown = reshape(QQ[1], 1, 1)
    bdown = QQ[2]
    hdown = PLP.HPoly(1, Adown, bdown, nothing, falses(1), PLP.STRICT_EPS_QQ)
    Dpl = PLP.PLDownset(PLP.PolyUnion(1, [hdown]))

    Fpl = PLP.PLFringe(1, [Upl], [Dpl], reshape(QQ[1], 1, 1))

    mktempdir() do dir
        path = joinpath(dir, "pl_fringe.json")
        SER.save_pl_fringe_json(path, Fpl)
        meta = SER.inspect_json(path)
        @test meta isa SER.JSONArtifactSummary
        @test meta.kind == "PLFringe"
        @test meta.n == 1
        @test meta.n_upsets == 1
        @test meta.n_downsets == 1
        @test meta.has_phi == true
        @test meta.profile_hint == :compact
        @test SER.pl_fringe_json_summary(path).kind == "PLFringe"
        @test SER.check_pl_fringe_json(path).valid

        Fpl2 = SER.load_pl_fringe_json(path)
        Fpl3 = SER.load_pl_fringe_json(path; validation=:trusted)
        @test Fpl2.n == 1
        @test length(Fpl2.Ups) == 1
        @test length(Fpl2.Downs) == 1
        @test Fpl2.Phi == Fpl.Phi
        @test Fpl2.Ups[1].U.parts[1].A == Aup
        @test Fpl2.Ups[1].U.parts[1].b == bup
        @test Fpl2.Downs[1].D.parts[1].A == Adown
        @test Fpl2.Downs[1].D.parts[1].b == bdown
        @test Fpl3.Phi == Fpl2.Phi

        debug_path = joinpath(dir, "pl_fringe_debug.json")
        SER.save_pl_fringe_json(debug_path, Fpl; profile=:debug)
        @test filesize(path) < filesize(debug_path)
        @test SER.inspect_json(debug_path).profile_hint == :debug
    end

    mktempdir() do dir
        path = joinpath(dir, "bad_flange.json")
        write(path, JSON3.write(Dict("kind" => "FlangeZn")))
        report = SER.check_flange_json(path)
        @test !report.valid
        @test !isempty(report.issues)
        @test_throws ArgumentError SER.check_flange_json(path; throw=true)
    end

    mktempdir() do dir
        path = joinpath(dir, "bad_pl_fringe.json")
        write(path, JSON3.write(Dict("kind" => "PLFringe", "schema_version" => SER.PLFRINGE_SCHEMA_VERSION)))
        report = SER.check_pl_fringe_json(path)
        @test !report.valid
        @test !isempty(report.issues)
        @test_throws ArgumentError SER.check_pl_fringe_json(path; throw=true)
    end

    @test TOA.check_flange_json === SER.check_flange_json
    @test TOA.check_pl_fringe_json === SER.check_pl_fringe_json
    @test TOA.flange_json_summary === SER.flange_json_summary
    @test TOA.pl_fringe_json_summary === SER.pl_fringe_json_summary

    # Canonical PL fringe parser.
    function _parse_finite_fringe_json_baseline(json_src;
                                                field::Union{Nothing,CM.AbstractCoeffField}=nothing,
                                                validation::Symbol=:strict)
        obj = JSON3.read(json_src)
        validate_masks = SER._resolve_validation_mode(validation)
        P = SER._parse_poset_from_obj(obj["poset"])
        n = FF.nvertices(P)
        U = SER._build_upsets(P, SER._external_parse_mask_rows(obj["U"], "U", n), validate_masks)
        D = SER._build_downsets(P, SER._external_parse_mask_rows(obj["D"], "D", n), validate_masks)
        saved_field, target_field = SER._external_parse_field(obj, field)
        Phi = SER._external_parse_phi(obj["phi"], saved_field, target_field, length(D), length(U))
        return FF.FringeModule{CM.coeff_type(target_field)}(P, U, D, Phi; field=target_field)
    end

    _parse_pl_fringe_json_baseline(json_src) = SER._parse_pl_fringe_obj(JSON3.read(json_src))

    pl_json = """
    {
      "kind": "PLFringe",
      "schema_version": $(SER.PLFRINGE_SCHEMA_VERSION),
      "n": 1,
      "ups": [ { "n": 1, "parts": [ { "A": [["-1/1"]], "b": ["0/1"] } ] } ],
      "downs": [ { "n": 1, "parts": [ { "A": [["1/1"]], "b": ["2/1"], "strict_mask": [1] } ] } ],
      "phi": [ ["1/1"] ]
    }
    """
    Fpl3 = SER.parse_pl_fringe_json(pl_json)
    Fpl3_old = _parse_pl_fringe_json_baseline(pl_json)
    @test Fpl3.n == 1
    @test length(Fpl3.Ups) == 1
    @test length(Fpl3.Downs) == 1
    @test Fpl3.Phi == reshape(QQ[1], 1, 1)
    @test Fpl3.Downs[1].D.parts[1].strict_mask == BitVector([true])
    @test Fpl3.Phi == Fpl3_old.Phi
    @test Fpl3.Ups[1].U.parts[1].A == Fpl3_old.Ups[1].U.parts[1].A
    @test Fpl3.Downs[1].D.parts[1].b == Fpl3_old.Downs[1].D.parts[1].b

    pl_legacy_json = """
    {
      "n": 1,
      "upsets": [ { "A": [["-1/1"]], "b": ["0/1"] } ],
      "downsets": [ { "parts": [ { "A": [["1/1"]], "b": ["2/1"] } ] } ],
      "phi": [ ["1/1"] ]
    }
    """
    @test_throws ErrorException SER.parse_pl_fringe_json(pl_legacy_json)

    pl_shorthand_json = """
    {
      "kind": "PLFringe",
      "schema_version": $(SER.PLFRINGE_SCHEMA_VERSION),
      "n": 1,
      "ups": [ { "A": [["-1/1"]], "b": ["0/1"] } ],
      "downs": [ { "n": 1, "parts": [ { "A": [["1/1"]], "b": ["2/1"] } ] } ],
      "phi": [ ["1/1"] ]
    }
    """
    @test_throws ErrorException SER.parse_pl_fringe_json(pl_shorthand_json)

    # Canonical finite fringe parser.
    finite_json = """
    {
      "poset": {
        "kind": "FinitePoset",
        "n": 3,
        "leq": [
          [true, true, true],
          [false, true, true],
          [false, false, true]
        ]
      },
      "U": [[2, 3]],
      "D": [[true, true, false]],
      "phi": [["2/1"]]
    }
    """
    Hext = SER.parse_finite_fringe_json(finite_json)
    Hext_old = _parse_finite_fringe_json_baseline(finite_json)
    @test Hext.P isa FF.FinitePoset
    @test Hext.U[1].mask == BitVector([false, true, true])
    @test Hext.D[1].mask == BitVector([true, true, false])
    @test Hext.phi[1, 1] == CM.coerce(Hext.field, 2)
    @test FF.fiber_dimension(Hext, 1) == 0
    @test FF.fiber_dimension(Hext, 2) == 1
    @test FF.fiber_dimension(Hext, 3) == 0
    @test [U.mask for U in Hext.U] == [U.mask for U in Hext_old.U]
    @test [D.mask for D in Hext.D] == [D.mask for D in Hext_old.D]
    @test Hext.phi == Hext_old.phi

    finite_legacy_json = """
    {
      "n": 3,
      "leq": [
        [true, true, true],
        [false, true, true],
        [false, false, true]
      ],
      "upsets": [[2, 3]],
      "downsets": [[true, true, false]],
      "phi": [["2/1"]]
    }
    """
    @test_throws ErrorException SER.parse_finite_fringe_json(finite_legacy_json)

    finite_legacy_field_json = """
    {
      "poset": {
        "kind": "FinitePoset",
        "n": 3,
        "leq": [
          [true, true, true],
          [false, true, true],
          [false, false, true]
        ]
      },
      "U": [[2, 3]],
      "D": [[true, true, false]],
      "field": "qq",
      "phi": [["2/1"]]
    }
    """
    @test_throws ErrorException SER.parse_finite_fringe_json(finite_legacy_field_json)

    # Validation mode symmetry with load_encoding_json.
    invalid_masks_json = """
    {
      "poset": {
        "kind": "FinitePoset",
        "n": 2,
        "leq": [
          [true, true],
          [false, true]
        ]
      },
      "U": [[true, false]],
      "D": [[true, true]],
      "phi": [[1]]
    }
    """
    @test_throws ErrorException SER.parse_finite_fringe_json(invalid_masks_json; validation=:strict)
    Htrusted = SER.parse_finite_fringe_json(invalid_masks_json; validation=:trusted)
    @test Htrusted.U[1].mask == BitVector([true, false])

    # Finite encoding fringe round-trip
    P = chain_poset(3)
    M = one_by_one_fringe(P, FF.principal_upset(P, 2), FF.principal_downset(P, 2))

    mktempdir() do dir
        path = joinpath(dir, "encoding.json")
        SER.save_encoding_json(path, M)
        obj = JSON3.read(read(path, String))
        @test Int(obj["schema_version"]) == SER.ENCODING_SCHEMA_VERSION
        @test obj["poset"]["leq"]["kind"] == "packed_words_v1"
        @test obj["U"]["kind"] == "packed_words_v1"
        @test obj["D"]["kind"] == "packed_words_v1"
        @test obj["phi"]["kind"] == "qq_chunks_v1"
        M_loaded = SER.load_encoding_json(path; output=:fringe)

        @test M_loaded.P.n == M.P.n
        @test FF.poset_equal(M_loaded.P, M.P)
        @test M_loaded.U[1].mask == M.U[1].mask
        @test M_loaded.D[1].mask == M.D[1].mask
        @test Matrix(M_loaded.phi) == Matrix(M.phi)
        for q in 1:M.P.n
            @test FF.fiber_dimension(M_loaded, q) == FF.fiber_dimension(M, q)
        end
    end

    # Finite encoding fringe round-trip with non-QQ field + override coercion
    M2 = one_by_one_fringe(P, FF.principal_upset(P, 2), FF.principal_downset(P, 2);
                           scalar=1, field=field2)
    mktempdir() do dir
        path = joinpath(dir, "encoding_f2.json")
        SER.save_encoding_json(path, M2)
        obj = JSON3.read(read(path, String))
        @test obj["phi"]["kind"] == "fp_flat_v1"
        M2_loaded = SER.load_encoding_json(path; output=:fringe)
        @test M2_loaded.field == field2
        @test eltype(M2_loaded.phi) == K2
        @test M2_loaded.phi[1, 1] == CM.coerce(field2, 1)

        M2_asQQ = SER.load_encoding_json(path; output=:fringe, field=CM.QQField())
        if field isa CM.QQField
            @test eltype(M2_asQQ.phi) == QQ
            @test M2_asQQ.phi[1, 1] == CM.coerce(CM.QQField(), 1)
        end
    end


    # M2/Singular bridge parser (pure JSON input; QQ-encoded scalar)
    json = """
    {
      "n": 1,
      "field": "QQ",
      "flats":      [ {"b":[1], "tau":[false], "id":"F1"} ],
      "injectives": [ {"b":[3], "tau":[false], "id":"E1"} ],
      "phi": [ ["1/1"] ]
    }
    """
    FG3 = SER.parse_flange_json(json)
    @test FG3.n == 1
    @test length(FG3.flats) == 1
    @test length(FG3.injectives) == 1
    @test FG3.phi[1,1] == CM.coerce(CM.QQField(), 1)
end

if field isa CM.QQField
@testset "M2SingularBridge.parse_flange_json edge cases" begin
    # tau given as index list (1-based) rather than Bool vector; phi omitted -> canonical_matrix
    json1 = """
    {
      "n": 1,
      "flats": [ { "id": "F", "b": [0], "tau": [1] } ],
      "injectives": [ { "id": "E", "b": [0], "tau": [1] } ]
    }
    """
    H1 = SER.parse_flange_json(json1)
    @test H1.n == 1
    @test length(H1.flats) == 1
    @test length(H1.injectives) == 1
    @test Matrix(H1.phi) == reshape(K[1], 1, 1)

    # Explicit phi entries can be rationals in string form.
    json2 = """
    {
      "n": 1,
      "flats": [ { "id": "F", "b": [0], "tau": [false] } ],
      "injectives": [ { "id": "E", "b": [0], "tau": [false] } ],
      "phi": [ [ "-2/3" ] ]
    }
    """
    H2 = SER.parse_flange_json(json2)
    @test Matrix(H2.phi)[1, 1] == (-c(2) / c(3))

    # Non-intersecting flat/injective pairs must force Phi entries to 0 (monomial condition).
    json3 = """
    {
      "n": 1,
      "flats": [ { "id": "F", "b": [5], "tau": [false] } ],
      "injectives": [ { "id": "E", "b": [3], "tau": [false] } ],
      "phi": [ [ 1 ] ]
    }
    """
    H3 = SER.parse_flange_json(json3)
    @test Matrix(H3.phi) == reshape(K[0], 1, 1)
    @test FZ.dim_at(H3, [0]) == 0

    json4 = """
    {
      "n": 1,
      "coeff_field": { "kind": "real", "T": "Float64" },
      "flats": [ { "id": "F1", "b": [0], "tau": [false] },
                 { "id": "F2", "b": [0], "tau": [false] } ],
      "injectives": [ { "id": "E1", "b": [0], "tau": [false] } ],
      "phi": [ [ 0.5, 1.5 ] ]
    }
    """
    @test_logs (:warn, r"phi has 2 non-integer numeric entries") SER.parse_flange_json(json4)
end
end

if field isa CM.QQField
@testset "Serialization.load_encoding_json strict schema" begin
    json = """
    {
      "kind": "FiniteEncodingFringe",
      "schema_version": 1,
      "poset": {
        "kind": "FinitePoset",
        "n": 3,
        "leq": {"kind":"packed_words_v1","nrows":3,"ncols":3,"words_per_row":1,"words":[7,6,4]}
      },
      "U": {"kind":"packed_words_v1","nrows":1,"ncols":3,"words_per_row":1,"words":[6]},
      "D": {"kind":"packed_words_v1","nrows":1,"ncols":3,"words_per_row":1,"words":[3]},
      "coeff_field": {"kind":"qq"},
      "phi": {
        "kind":"qq_chunks_v1",
        "m":1,
        "k":1,
        "base":1000000000,
        "num_sign":[-1],
        "num_ptr":[1,2],
        "num_chunks":[2],
        "den_ptr":[1,2],
        "den_chunks":[3]
      }
    }
    """
    path, io = mktemp()
    write(io, json)
    close(io)

    M = SER.load_encoding_json(path; output=:fringe)
    @test M.P.n == 3
    @test FF.leq_matrix(M.P) == BitMatrix([1 1 1; 0 1 1; 0 0 1])
    @test length(M.U) == 1
    @test length(M.D) == 1
    @test M.U[1].mask == BitVector([false, true, true])
    @test M.D[1].mask == BitVector([true, true, false])
    @test Matrix(M.phi) == reshape(K[c(-2//3)], 1, 1)
end

@testset "Serialization.load_encoding_json trusted mask fast path (schema v1)" begin
    json = """
    {
      "kind": "FiniteEncodingFringe",
      "schema_version": 1,
      "poset": {
        "kind": "FinitePoset",
        "n": 3,
        "leq": {"kind":"packed_words_v1","nrows":3,"ncols":3,"words_per_row":1,"words":[7,6,4]}
      },
      "U": {"kind":"packed_words_v1","nrows":1,"ncols":3,"words_per_row":1,"words":[5]},
      "D": {"kind":"packed_words_v1","nrows":1,"ncols":3,"words_per_row":1,"words":[2]},
      "coeff_field": {"kind":"qq"},
      "phi": {
        "kind":"qq_chunks_v1",
        "m":1,
        "k":1,
        "base":1000000000,
        "num_sign":[0],
        "num_ptr":[1,2],
        "num_chunks":[0],
        "den_ptr":[1,2],
        "den_chunks":[1]
      }
    }
    """
    path, io = mktemp()
    write(io, json)
    close(io)

    err = nothing
    try
        SER.load_encoding_json(path; output=:fringe)
    catch e
        err = e
    end
    @test err isa ErrorException
    @test occursin("validation=:trusted", sprint(showerror, err))
    H = SER.load_encoding_json(path; output=:fringe, validation=:trusted)
    @test H.U[1].mask == BitVector([true, false, true])
    @test H.D[1].mask == BitVector([false, true, false])
end

@testset "Serialization.load_encoding_json rejects non-v1 schema" begin
    json_old = """
    {
      "kind": "FiniteEncodingFringe",
      "schema_version": 3,
      "poset": {
        "kind": "FinitePoset",
        "n": 3,
        "leq": {"kind":"packed_words_v1","nrows":3,"ncols":3,"words_per_row":1,"words":[7,6,4]}
      },
      "U": {"kind":"packed_words_v1","nrows":1,"ncols":3,"words_per_row":1,"words":[6]},
      "D": {"kind":"packed_words_v1","nrows":1,"ncols":3,"words_per_row":1,"words":[3]},
      "coeff_field": {"kind":"qq"},
      "phi": {
        "kind":"qq_chunks_v1",
        "m":1,
        "k":1,
        "base":1000000000,
        "num_sign":[-1],
        "num_ptr":[1,2],
        "num_chunks":[2],
        "den_ptr":[1,2],
        "den_chunks":[3]
      }
    }
    """
    path, io = mktemp()
    write(io, json_old)
    close(io)

    @test_throws ErrorException SER.load_encoding_json(path; output=:fringe)
end

@testset "Serialization.load_encoding_json output modes + inspect" begin
    coords = (Float64[0.0, 1.0],)
    P = FF.GridPoset(coords)
    U = FF.principal_upset(P, 2)
    D = FF.principal_downset(P, 2)
    H = one_by_one_fringe(P, U, D)
    pi = EC.GridEncodingMap(P, coords)
    M = IR.pmodule_from_fringe(H)
    enc = RES.EncodingResult(P, M, pi; H=H, backend=:zn)

    mktemp() do path, io
        close(io)
        SER.save_encoding_json(path, enc)
        info = SER.inspect_json(path)
        @test info isa SER.JSONArtifactSummary
        @test info.kind == "FiniteEncodingFringe"
        @test info.nvertices == 2
        @test info.has_pi
        @test SER.artifact_kind(info) == "FiniteEncodingFringe"
        @test SER.artifact_field(info) == "qq"
        @test SER.artifact_poset_kind(info) == "GridPoset"
        @test SER.artifact_profile_hint(info) == :compact
        @test SER.has_encoding_map(info)
        @test !SER.has_dense_leq(info)
        @test SER.artifact_size_bytes(info) isa Integer
        @test SER.encoding_json_summary(path).kind == "FiniteEncodingFringe"
        @test SER.check_encoding_json(path).valid

        H2 = SER.load_encoding_json(path; output=:fringe)
        @test H2 isa FF.FringeModule

        H3, pi3 = SER.load_encoding_json(path; output=:fringe_with_pi)
        @test H3 isa FF.FringeModule
        @test pi3 isa EC.GridEncodingMap

        enc_default = SER.load_encoding_json(path)
        @test enc_default isa RES.EncodingResult

        enc2 = SER.load_encoding_json(path; output=:encoding_result)
        @test enc2 isa RES.EncodingResult
        @test FF.nvertices(enc2.P) == 2
        @test enc2.M isa MD.PModule

        SER.save_encoding_json(path, enc; include_pi=false)
        err = nothing
        try
            SER.load_encoding_json(path)
        catch e
            err = e
        end
        @test err isa ErrorException
        @test occursin("include_pi=true", sprint(showerror, err))
    end

    mktemp() do path, io
        close(io)
        write(path, JSON3.write(Dict("kind" => "FiniteEncodingFringe")))
        report = SER.check_encoding_json(path)
        @test !report.valid
        @test !isempty(report.issues)
        @test_throws ArgumentError SER.check_encoding_json(path; throw=true)
    end

    @test TOA.encoding_json_summary === SER.encoding_json_summary
    @test TOA.check_encoding_json === SER.check_encoding_json
end

@testset "Serialization.save_encoding_json profile presets" begin
    coords = (Float64[0.0, 1.0],)
    P = FF.GridPoset(coords)
    U = FF.principal_upset(P, 2)
    D = FF.principal_downset(P, 2)
    H = one_by_one_fringe(P, U, D)
    M = IR.pmodule_from_fringe(H)
    pi = EC.GridEncodingMap(P, coords)
    enc = RES.EncodingResult(P, M, pi; H=H, backend=:zn)

    mktemp() do path, io
        close(io)
        SER.save_encoding_json(path, enc; profile=:compact)
        obj = JSON3.read(read(path, String))
        @test haskey(obj, "pi")
        @test !haskey(obj["poset"], "leq")
        @test SER.inspect_json(path).profile_hint == :compact

        SER.save_encoding_json(path, enc; profile=:portable)
        obj = JSON3.read(read(path, String))
        @test haskey(obj["poset"], "leq")
        @test SER.inspect_json(path).profile_hint == :portable

        SER.save_encoding_json(path, enc; profile=:debug)
        raw = read(path, String)
        @test occursin("\n", raw)
        @test occursin("  \"kind\"", raw)
        @test SER.inspect_json(path).profile_hint == :debug

        SER.save_encoding_json(path, enc; profile=:portable, include_pi=false)
        obj = JSON3.read(read(path, String))
        @test !haskey(obj, "pi")
    end
end
end


# Helper: build an antichain poset with n elements
function antichain_poset(n::Int)
    leq = falses(n, n)
    for i in 1:n
        leq[i,i] = true
    end
    return FF.FinitePoset(leq)
end

# Helper: "V" poset with 1<2 and 1<3
function v_poset()
    leq = falses(3,3)
    for i in 1:3
        leq[i,i] = true
    end
    leq[1,2] = true
    leq[1,3] = true
    return FF.FinitePoset(leq)
end

# Poset of nonempty proper faces of a triangle (order complex = S^1):
# {1},{2},{3},{1,2},{1,3},{2,3} ordered by inclusion.
function triangle_boundary_poset()
    n = 6
    leq = falses(n, n)
    for i in 1:n
        leq[i,i] = true
    end
    leq[1,4] = true
    leq[1,5] = true
    leq[2,4] = true
    leq[2,6] = true
    leq[3,5] = true
    leq[3,6] = true
    return FF.FinitePoset(n, leq)
end

@testset "Change-of-posets: pullback / Kan extensions / derived" begin

    # -------------------------------------------------------------------------
    # Pullback along collapse chain2 -> point
    # -------------------------------------------------------------------------
    Q = chain_poset(2)
    P = chain_poset(1)
    pi = EN.EncodingMap(Q, P, [1,1])

    N = MD.PModule{K}(P, [3], Dict{Tuple{Int,Int},SparseMatrixCSC{K,Int}}())
    pbN = TO.restriction(pi, N)

    @test pbN.Q.n == 2
    @test pbN.dims == [3,3]
    @test Matrix(pbN.edge_maps[1, 2]) == Matrix{K}(I, 3, 3)

    # Pullback on morphisms should reuse the same pointwise component at each q in Q.
    fN = MD.PMorphism(N, N, [K[c(2) c(0) c(0); c(0) c(3) c(0); c(0) c(0) c(5)]])
    pbfN = TO.restriction(pi, fN)
    @test pbfN.dom.dims == pbN.dims
    @test pbfN.cod.dims == pbN.dims
    @test Matrix(pbfN.comps[1]) == Matrix(fN.comps[1])
    @test Matrix(pbfN.comps[2]) == Matrix(fN.comps[1])

    # -------------------------------------------------------------------------
    # Left/right Kan extension collapse chain2 -> point (terminal/initial fast path)
    # -------------------------------------------------------------------------
    A = spzeros(K, 2, 1)
    A[1,1] = c(1)  # 1 -> first coordinate

    M = MD.PModule{K}(Q, [1,2], Dict((1,2)=>A))

    Lan = TO.pushforward_left(pi, M)
    Ran = TO.pushforward_right(pi, M)

    @test Lan.Q.n == 1
    @test Lan.dims == [2]   # terminal object is vertex 2
    @test Ran.dims == [1]   # initial object is vertex 1

    # -------------------------------------------------------------------------
    # Functoriality on morphisms (collapse)
    # -------------------------------------------------------------------------
    # Choose a module endomorphism commuting with A.
    # Use coefficients that are nonzero in small prime fields.
    f1 = K[1]
    f2 = [c(1) c(0);
          c(0) c(1)]
    f = MD.PMorphism(M, M, [reshape(f1,1,1), f2])

    Lan_f = TO.pushforward_left(pi, f)
    Ran_f = TO.pushforward_right(pi, f)

    @test Matrix(Lan_f.comps[1]) == f2
    @test Matrix(Ran_f.comps[1]) == reshape(f1,1,1)

    # -------------------------------------------------------------------------
    # Identity encoding map should give identity functors (fast paths guarantee exact equality)
    # -------------------------------------------------------------------------
    pid = EN.EncodingMap(Q, Q, [1,2])
    pb_id = TO.restriction(pid, M)
    Lan_id = TO.pushforward_left(pid, M)
    Ran_id = TO.pushforward_right(pid, M)

    @test pb_id.dims == M.dims
    @test Matrix(pb_id.edge_maps[1, 2]) == Matrix(M.edge_maps[1, 2])

    @test Lan_id.dims == M.dims
    @test Matrix(Lan_id.edge_maps[1, 2]) == Matrix(M.edge_maps[1, 2])

    @test Ran_id.dims == M.dims
    @test Matrix(Ran_id.edge_maps[1, 2]) == Matrix(M.edge_maps[1, 2])

    # -------------------------------------------------------------------------
    # Nontrivial colimit: V-poset collapsed to point with identity maps gives dim 1
    # -------------------------------------------------------------------------
    Qv = v_poset()
    Pv = chain_poset(1)
    piv = EN.EncodingMap(Qv, Pv, [1,1,1])

    # dims all 1, maps 1->2 and 1->3 are identity
    Ev = Dict{Tuple{Int,Int},SparseMatrixCSC{K,Int}}()
    for (u,v) in FF.cover_edges(Qv).edges
        mat = spzeros(K, 1, 1)
        mat[1,1] = c(1)
        Ev[(u,v)] = mat
    end
    Mv = MD.PModule{K}(Qv, [1,1,1], Ev)

    Lan_v = TO.pushforward_left(piv, Mv)
    @test Lan_v.dims == [1]   # pushout of k <- k -> k is k

    # -------------------------------------------------------------------------
    # General-case limit/colimit: antichain of 2 collapsed to point -> direct sum (dim 5)
    # -------------------------------------------------------------------------
    Qa = antichain_poset(2)
    Pa = chain_poset(1)
    pia = EN.EncodingMap(Qa, Pa, [1,1])

    Ma = MD.PModule{K}(Qa, [2,3], Dict{Tuple{Int,Int},SparseMatrixCSC{K,Int}}())
    Lan_a = TO.pushforward_left(pia, Ma)
    Ran_a = TO.pushforward_right(pia, Ma)

    @test Lan_a.dims == [5]
    @test Ran_a.dims == [5]

    # -------------------------------------------------------------------------
    # Derived functors vanish for collapse with terminal/initial objects
    # -------------------------------------------------------------------------
    if !(field isa CM.RealField)
        Lmods = TO.derived_pushforward_left(pi, M, OPT.DerivedFunctorOptions(maxdeg=2))
        Rmods = TO.derived_pushforward_right(pi, M, OPT.DerivedFunctorOptions(maxdeg=2))

        @test Lmods[1].dims == Lan.dims
        @test Lmods[2].dims == [0]
        @test Lmods[3].dims == [0]

        @test Rmods[1].dims == Ran.dims
        @test Rmods[2].dims == [0]
        @test Rmods[3].dims == [0]
    end

end

@testset "Change-of-posets threading parity" begin
    Q = chain_poset(2)
    P = chain_poset(1)
    pi = EN.EncodingMap(Q, P, [1,1])

    A = spzeros(K, 2, 1)
    A[1,1] = c(1)
    M = MD.PModule{K}(Q, [1,2], Dict((1,2)=>A))

    Lan_s = TO.pushforward_left(pi, M; threads=false)
    Lan_t = TO.pushforward_left(pi, M; threads=true)
    @test Lan_s.dims == Lan_t.dims

    Ran_s = TO.pushforward_right(pi, M; threads=false)
    Ran_t = TO.pushforward_right(pi, M; threads=true)
    @test Ran_s.dims == Ran_t.dims

    # Identity encoding to exercise edge map equality.
    pid = EN.EncodingMap(Q, Q, [1,2])
    Lan_id_s = TO.pushforward_left(pid, M; threads=false)
    Lan_id_t = TO.pushforward_left(pid, M; threads=true)
    @test Lan_id_s.dims == Lan_id_t.dims
    @test Matrix(Lan_id_s.edge_maps[1, 2]) == Matrix(Lan_id_t.edge_maps[1, 2])

    Ran_id_s = TO.pushforward_right(pid, M; threads=false)
    Ran_id_t = TO.pushforward_right(pid, M; threads=true)
    @test Ran_id_s.dims == Ran_id_t.dims
    @test Matrix(Ran_id_s.edge_maps[1, 2]) == Matrix(Ran_id_t.edge_maps[1, 2])

    # Morphism parity on identity encoding.
    f1 = K[1]
    f2 = [c(1) c(0);
          c(0) c(1)]
    f = MD.PMorphism(M, M, [reshape(f1,1,1), f2])

    Lf_s = TO.pushforward_left(pid, f; threads=false)
    Lf_t = TO.pushforward_left(pid, f; threads=true)
    @test Matrix(Lf_s.comps[1]) == Matrix(Lf_t.comps[1])
    @test Matrix(Lf_s.comps[2]) == Matrix(Lf_t.comps[2])

    Rf_s = TO.pushforward_right(pid, f; threads=false)
    Rf_t = TO.pushforward_right(pid, f; threads=true)
    @test Matrix(Rf_s.comps[1]) == Matrix(Rf_t.comps[1])
    @test Matrix(Rf_s.comps[2]) == Matrix(Rf_t.comps[2])

    # Derived functors parity on collapse (uses resolutions internally).
    if !(field isa CM.RealField)
        df = OPT.DerivedFunctorOptions(maxdeg=1)
        Ls = TO.derived_pushforward_left(pi, M, df; threads=false)
        Lt = TO.derived_pushforward_left(pi, M, df; threads=true)
        @test [L.dims for L in Ls] == [L.dims for L in Lt]

        Rs = TO.derived_pushforward_right(pi, M, df; threads=false)
        Rt = TO.derived_pushforward_right(pi, M, df; threads=true)
        @test [R.dims for R in Rs] == [R.dims for R in Rt]
    end
end

if !(field isa CM.RealField)
@testset "ChangeOfPosets right-Kan selector fast path parity" begin
    Qtri = triangle_boundary_poset()
    Ppt = chain_poset(1)
    pi = EN.EncodingMap(Qtri, Ppt, fill(1, Qtri.n))

    dims = ones(Int, Qtri.n)
    edge_maps = Dict{Tuple{Int,Int}, SparseMatrixCSC{K,Int}}()
    for (a,b) in FF.cover_edges(Qtri)
        edge_maps[(a,b)] = sparse(fill(c(1), 1, 1))
    end
    M = MD.PModule{K}(Qtri, dims, edge_maps)
    f = MD.PMorphism(M, M, [fill(c(2), 1, 1) for _ in 1:Qtri.n])

    function _same_module(A::MD.PModule, B::MD.PModule)
        @test A.dims == B.dims
        for (u, v) in FF.cover_edges(A.Q)
            @test Matrix(A.edge_maps[u, v]) == Matrix(B.edge_maps[u, v])
        end
    end

    function _same_morphism(g::MD.PMorphism, h::MD.PMorphism)
        @test g.dom.dims == h.dom.dims
        @test g.cod.dims == h.cod.dims
        for u in eachindex(g.comps)
            @test Matrix(g.comps[u]) == Matrix(h.comps[u])
        end
    end

    old_selector = TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_RIGHT_KAN_SELECTOR_FASTPATH[]
    old_summary = TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_RIGHT_KAN_QQ_SUMMARY_SELECTOR[]
    old_direct_csc = TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_RIGHT_KAN_DIRECT_CSC[]
    try
        TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_RIGHT_KAN_SELECTOR_FASTPATH[] = false
        TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_RIGHT_KAN_QQ_SUMMARY_SELECTOR[] = false
        TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_RIGHT_KAN_DIRECT_CSC[] = false
        Rmod_fallback = TO.pushforward_right(pi, M; threads=false)
        Rmor_fallback = TO.pushforward_right(pi, f; threads=false)

        TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_RIGHT_KAN_SELECTOR_FASTPATH[] = true
        TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_RIGHT_KAN_QQ_SUMMARY_SELECTOR[] = true
        TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_RIGHT_KAN_DIRECT_CSC[] = true
        Rmod_fast = TO.pushforward_right(pi, M; threads=false)
        Rmor_fast = TO.pushforward_right(pi, f; threads=false)

        _same_module(Rmod_fallback, Rmod_fast)
        _same_morphism(Rmor_fallback, Rmor_fast)
    finally
        TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_RIGHT_KAN_SELECTOR_FASTPATH[] = old_selector
        TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_RIGHT_KAN_QQ_SUMMARY_SELECTOR[] = old_summary
        TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_RIGHT_KAN_DIRECT_CSC[] = old_direct_csc
    end
end
end

@testset "ChangeOfPosets left-Kan morphism prebuilt-data parity" begin
    Qtri = triangle_boundary_poset()
    Ppt = chain_poset(1)
    pi = EN.EncodingMap(Qtri, Ppt, fill(1, Qtri.n))

    dims = ones(Int, Qtri.n)
    edge_maps = Dict{Tuple{Int,Int}, SparseMatrixCSC{K,Int}}()
    for (a,b) in FF.cover_edges(Qtri)
        edge_maps[(a,b)] = sparse(fill(c(1), 1, 1))
    end
    M = MD.PModule{K}(Qtri, dims, edge_maps)
    f = MD.PMorphism(M, M, [fill(c(2), 1, 1) for _ in 1:Qtri.n])

    plan = TO.ChangeOfPosets._translation_plan(pi; session_cache=nothing)
    dom_out, data_dom = TO.ChangeOfPosets._left_kan_data(pi, f.dom;
                                                         check=false,
                                                         threads=false,
                                                         session_cache=nothing,
                                                         plan=plan)
    cod_out, data_cod = TO.ChangeOfPosets._left_kan_data(pi, f.cod;
                                                         check=false,
                                                         threads=false,
                                                         session_cache=nothing,
                                                         plan=plan)

    old_mul = TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_LEFT_KAN_MORPHISM_MUL[]
    try
        TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_LEFT_KAN_MORPHISM_MUL[] = false
        public_slow = TO.pushforward_left(pi, f; threads=false)
        prebuilt_slow = TO.ChangeOfPosets._pushforward_left_morphism_from_data(
            dom_out, cod_out, f, data_dom, data_cod, plan.left_fibers; threads=false
        )

        TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_LEFT_KAN_MORPHISM_MUL[] = true
        public_fast = TO.pushforward_left(pi, f; threads=false)
        prebuilt_fast = TO.ChangeOfPosets._pushforward_left_morphism_from_data(
            dom_out, cod_out, f, data_dom, data_cod, plan.left_fibers; threads=false
        )

        for pair in ((public_slow, prebuilt_slow), (public_fast, prebuilt_fast), (public_slow, public_fast))
            g, h = pair
            @test g.dom.dims == h.dom.dims
            @test g.cod.dims == h.cod.dims
            for u in eachindex(g.comps)
                @test Matrix(g.comps[u]) == Matrix(h.comps[u])
            end
        end
    finally
        TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_LEFT_KAN_MORPHISM_MUL[] = old_mul
    end
end

if field isa CM.QQField
@testset "ChangeOfPosets left-Kan quotient fast path parity" begin
    Qtri = triangle_boundary_poset()
    Ppt = chain_poset(1)
    pi = EN.EncodingMap(Qtri, Ppt, fill(1, Qtri.n))

    dims = ones(Int, Qtri.n)
    edge_maps = Dict{Tuple{Int,Int}, SparseMatrixCSC{K,Int}}()
    for (a,b) in FF.cover_edges(Qtri)
        edge_maps[(a,b)] = sparse(fill(c(1), 1, 1))
    end
    M = MD.PModule{K}(Qtri, dims, edge_maps)
    f = MD.PMorphism(M, M, [fill(c(2), 1, 1) for _ in 1:Qtri.n])

    old_summary = TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_LEFT_KAN_QQ_SUMMARY_QUOTIENT[]
    old_mul = TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_LEFT_KAN_MORPHISM_MUL[]
    try
        TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_LEFT_KAN_QQ_SUMMARY_QUOTIENT[] = false
        TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_LEFT_KAN_MORPHISM_MUL[] = false
        Lmod_slow = TO.pushforward_left(pi, M; threads=false)
        Lmor_slow = TO.pushforward_left(pi, f; threads=false)

        TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_LEFT_KAN_QQ_SUMMARY_QUOTIENT[] = true
        TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_LEFT_KAN_MORPHISM_MUL[] = false
        Lmod_fast = TO.pushforward_left(pi, M; threads=false)
        Lmor_fast = TO.pushforward_left(pi, f; threads=false)

        @test Lmod_slow.dims == Lmod_fast.dims
        for (u, v) in FF.cover_edges(Lmod_slow.Q)
            @test Matrix(Lmod_slow.edge_maps[u, v]) == Matrix(Lmod_fast.edge_maps[u, v])
        end
        @test Lmor_slow.dom.dims == Lmor_fast.dom.dims
        @test Lmor_slow.cod.dims == Lmor_fast.cod.dims
        for u in eachindex(Lmor_slow.comps)
            @test Matrix(Lmor_slow.comps[u]) == Matrix(Lmor_fast.comps[u])
        end
    finally
        TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_LEFT_KAN_QQ_SUMMARY_QUOTIENT[] = old_summary
        TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_LEFT_KAN_MORPHISM_MUL[] = old_mul
    end
end
end

if field isa CM.QQField
@testset "Derived pushforward maps (morphism action)" begin
    Qtri = triangle_boundary_poset()
    Ppt = chain_poset(1)
    # Collapse Qtri -> *.
    # EncodingMap(Q, P, pi_of_q) stores the domain poset first.
    pi = EN.EncodingMap(Qtri, Ppt, fill(1, Qtri.n))

    # Constant 1D module on Qtri.
    dims = ones(Int, Qtri.n)
    edge_maps = Dict{Tuple{Int,Int}, SparseMatrixCSC{K,Int}}()
    for (a,b) in FF.cover_edges(Qtri)
        edge_maps[(a,b)] = sparse(fill(c(1), 1, 1))
    end
    M = MD.PModule{K}(Qtri, dims, edge_maps)

    # Scalar endomorphisms of M.
    function scalar_endomorphism(M::MD.PModule{K}, a::K) where {K}
        comps = [fill(a, 1, 1) for _ in 1:M.Q.n]
        return MD.PMorphism(M, M, comps)
    end

    f2 = scalar_endomorphism(M, c(1))
    f3 = scalar_endomorphism(M, c(1))

    # Compose fiberwise.
    function compose_morphism(g::MD.PMorphism, f::MD.PMorphism)
        @assert f.cod === g.dom
        n = f.dom.Q.n
        comps = [g.comps[u] * f.comps[u] for u in 1:n]
        return MD.PMorphism(f.dom, g.cod, comps)
    end

    f6 = compose_morphism(f3, f2)

    # Higher derived functors are nontrivial (S^1):
    Lmods = TO.derived_pushforward_left(pi, M, OPT.DerivedFunctorOptions(maxdeg=1))
    Rmods = TO.derived_pushforward_right(pi, M, OPT.DerivedFunctorOptions(maxdeg=1))

    @test FZ.dim_at(Lmods[1], 1) == 1   # L_0
    @test FZ.dim_at(Lmods[2], 1) == 1   # L_1
    @test FZ.dim_at(Rmods[1], 1) == 1   # R^0
    @test FZ.dim_at(Rmods[2], 1) == 1   # R^1

    # Induced derived maps: should act by scalar multiplication in all degrees.
    Lf2 = TO.derived_pushforward_left(pi, f2, OPT.DerivedFunctorOptions(maxdeg=1))
    Lf3 = TO.derived_pushforward_left(pi, f3, OPT.DerivedFunctorOptions(maxdeg=1))
    Lf6 = TO.derived_pushforward_left(pi, f6, OPT.DerivedFunctorOptions(maxdeg=1))

    @test Lf2[1].comps[1] == fill(c(1), 1, 1)
    @test Lf2[2].comps[1] == fill(c(1), 1, 1)

    # Functoriality in each degree: L(f3 o f2) = L(f3) o L(f2)
    @test Lf6[1].comps[1] == Lf3[1].comps[1] * Lf2[1].comps[1]
    @test Lf6[2].comps[1] == Lf3[2].comps[1] * Lf2[2].comps[1]

    # Right-derived maps.
    Rf2 = TO.derived_pushforward_right(pi, f2, OPT.DerivedFunctorOptions(maxdeg=1))
    Rf3 = TO.derived_pushforward_right(pi, f3, OPT.DerivedFunctorOptions(maxdeg=1))
    Rf6 = TO.derived_pushforward_right(pi, f6, OPT.DerivedFunctorOptions(maxdeg=1))

    @test Rf2[1].comps[1] == fill(c(1), 1, 1)
    @test Rf2[2].comps[1] == fill(c(1), 1, 1)

    # Functoriality: R(f3 o f2) = R(f3) o R(f2)
    @test Rf6[1].comps[1] == Rf3[1].comps[1] * Rf2[1].comps[1]
    @test Rf6[2].comps[1] == Rf3[2].comps[1] * Rf2[2].comps[1]
end
end

if field isa CM.QQField
@testset "Sparse naturality systems (no dense QQ matrices)" begin
    @testset "ChainComplexes.solve_particular works on SparseMatrixCSC{QQ}" begin
        # A is 3x3, consistent system with 2 RHS columns.
        A = sparse([1, 2, 2, 3],
                   [1, 1, 3, 2],
                   K[1, 1, 1, 1],
                   3, 3)

        B = K[1 0;
               0 1;
               1 1]

        Xs = CC.solve_particular(A, B)
        @test A * Xs == B

        # Dense version should also solve.
        Xd = CC.solve_particular(CM.QQField(), Matrix(A), B)
        @test A * Xd == B

        # Inconsistent system: 0*x = 1.
        A0 = sparse(Int[], Int[], K[], 1, 1)
        B0 = reshape(K[1], 1, 1)
        @test CC.solve_particular(A0, B0) === nothing
    end

    @testset "DerivedFunctors.Hom produces correct basis (sanity cases)" begin
        Q = chain_poset(2)

        # Case 1: identity edge maps -> Hom is 1-dimensional.
        edge_id = Dict((1, 2) => Matrix{K}(I, 1, 1))
        M = MD.PModule{K}(Q, [1, 1], edge_id)
        N = MD.PModule{K}(Q, [1, 1], edge_id)

        H = DF.Hom(M, N)
        @test length(H.basis) == 1
        for f in H.basis
            for (u, v) in FF.cover_edges(Q)
                @test N.edge_maps[u, v] * f.comps[u] == f.comps[v] * M.edge_maps[u, v]
            end
        end

        # Case 2: zero edge maps -> no coupling between vertices -> dimension 2.
        edge_zero = Dict((1, 2) => zeros(K, 1, 1))
        M0 = MD.PModule{K}(Q, [1, 1], edge_zero)
        N0 = MD.PModule{K}(Q, [1, 1], edge_zero)

        H0 = DF.Hom(M0, N0)
        @test length(H0.basis) == 2
        for f in H0.basis
            for (u, v) in FF.cover_edges(Q)
                @test N0.edge_maps[u, v] * f.comps[u] == f.comps[v] * M0.edge_maps[u, v]
            end
        end
    end
end
end

@testset "Common refinement encoding for different posets" begin
    # A small helper for a constant 1-dimensional module on a finite poset:
    # dims[v] = 1 for all v, and every cover-edge map is the 1x1 identity.
    function constant_1_module(P::FF.FinitePoset)
        dims = fill(1, P.n)
        edge_maps = Dict{Tuple{Int,Int}, Matrix{K}}()
        C = FF.cover_edges(P)
        sizehint!(edge_maps, length(C))
        for (u, v) in C
            edge_maps[(u, v)] = Matrix{K}(I, 1, 1)
        end
        return MD.PModule{K}(P, dims, edge_maps)
    end

    P1 = chain_poset(2)
    P2 = chain_poset(3)

    M1 = constant_1_module(P1)
    M2 = constant_1_module(P2)

    # The new API: build a common refinement automatically.
    P, Ms, pi1, pi2 = TO.encode_pmodules_to_common_poset(M1, M2)

    @test P.n == P1.n * P2.n
    @test Ms[1].Q === P
    @test Ms[2].Q === P
    @test pi1.Q === P && pi1.P === P1
    @test pi2.Q === P && pi2.P === P2

    # Sanity: product cover-edge count should be |E1|*n2 + n1*|E2|
    C1 = FF.cover_edges(P1)
    C2 = FF.cover_edges(P2)
    C  = FF.cover_edges(P)  # should be fast because product_poset pre-caches it
    @test length(C) == length(C1) * P2.n + P1.n * length(C2)

    # Hom between constant rank-1 modules should be 1-dimensional.
    H = DF.Hom(Ms[1], Ms[2])
    @test TO.dim(H) == 1

    # Compare against the generic restriction-based pullback (correctness check).
    # This also ensures restriction(pi, M) works concretely (not just defined).
    M1r = TO.restriction(pi1, M1)
    M2r = TO.restriction(pi2, M2)

    @test M1r.dims == Ms[1].dims
    @test M2r.dims == Ms[2].dims

    for e in FF.cover_edges(P)
        u, v = e
        @test Ms[1].edge_maps[u, v] == M1r.edge_maps[u, v]
        @test Ms[2].edge_maps[u, v] == M2r.edge_maps[u, v]
    end

    # A nontrivial example where commutativity forces Hom = 0.
    # S1: simple at vertex 1 on P1=chain(2).
    # S2: simple at vertex 2 on P2=chain(3).
    # On the product, vertical identity maps in S1 and zero maps in S2
    # force any would-be morphism to vanish.
    S1 = IR.pmodule_from_fringe(one_by_one_fringe(P1, FF.principal_upset(P1, 1), FF.principal_downset(P1, 1)))
    S2 = IR.pmodule_from_fringe(one_by_one_fringe(P2, FF.principal_upset(P2, 2), FF.principal_downset(P2, 2)))

    _, Ms2, _, _ = TO.encode_pmodules_to_common_poset(S1, S2)
    H2 = DF.Hom(Ms2[1], Ms2[2])
    @test TO.dim(H2) == 0
end

@testset "Identical leq matrices avoid product blowup" begin
    function constant_1_module(P::FF.FinitePoset)
        dims = fill(1, P.n)
        edge_maps = Dict{Tuple{Int,Int}, Matrix{K}}()
        C = FF.cover_edges(P)
        sizehint!(edge_maps, length(C))
        for (u, v) in C
            edge_maps[(u, v)] = Matrix{K}(I, 1, 1)
        end
        return MD.PModule{K}(P, dims, edge_maps)
    end

    P  = chain_poset(3)
    Pc = FF.FinitePoset(BitMatrix(FF.leq_matrix(P)); check = false)  # structurally identical, different object

    M  = constant_1_module(P)
    Mc = constant_1_module(Pc)

    Pout, Ms, pi1, pi2 = TO.encode_pmodules_to_common_poset(M, Mc)

    # Should NOT return a 9-vertex product. It should rebase Mc onto P.
    @test Pout === P
    @test Ms[1].Q === P
    @test Ms[2].Q === P

    H = DF.Hom(Ms[1], Ms[2])
    @test TO.dim(H) == 1
end

@testset "product_poset caching" begin
    P1 = chain_poset(2)
    P2 = chain_poset(3)
    sc = CM.SessionCache()

    prod1 = TO.product_poset(P1, P2; session_cache=sc)
    prod2 = TO.product_poset(P1, P2; session_cache=sc)

    # With an explicit session cache, this should hit the cache.
    @test prod1.P === prod2.P
end

@testset "ChangeOfPosets translation caches" begin
    Q = chain_poset(4)
    P = chain_poset(2)
    pi = EN.EncodingMap(Q, P, [1, 1, 2, 2])

    MQ = IR.pmodule_from_fringe(one_by_one_fringe(Q,
                                                  FF.principal_upset(Q, 2),
                                                  FF.principal_downset(Q, 4);
                                                  scalar=one(K), field=field))
    MP = IR.pmodule_from_fringe(one_by_one_fringe(P,
                                                  FF.principal_upset(P, 1),
                                                  FF.principal_downset(P, 2);
                                                  scalar=one(K), field=field))
    idQ = MD.id_morphism(MQ)
    sc = CM.SessionCache()

    pb1 = TO.pullback(pi, MP; session_cache=sc)
    pb2 = TO.pullback(pi, MP; session_cache=sc)
    @test pb1 === pb2

    left1 = TO.pushforward_left(pi, MQ; threads=false, session_cache=sc)
    left2 = TO.pushforward_left(pi, MQ; threads=false, session_cache=sc)
    @test left1 === left2

    right1 = TO.pushforward_right(pi, MQ; threads=false, session_cache=sc)
    right2 = TO.pushforward_right(pi, MQ; threads=false, session_cache=sc)
    @test right1 === right2

    leftf1 = TO.pushforward_left(pi, idQ; threads=false, session_cache=sc)
    leftf2 = TO.pushforward_left(pi, idQ; threads=false, session_cache=sc)
    @test leftf1 === leftf2

    rightf1 = TO.pushforward_right(pi, idQ; threads=false, session_cache=sc)
    rightf2 = TO.pushforward_right(pi, idQ; threads=false, session_cache=sc)
    @test rightf1 === rightf2
end

@testset "encode_pmodules_to_common_poset caches translated modules" begin
    P1 = chain_poset(2)
    P2 = chain_poset(3)
    M1 = IR.pmodule_from_fringe(one_by_one_fringe(P1,
                                                  FF.principal_upset(P1, 1),
                                                  FF.principal_downset(P1, 2);
                                                  scalar=one(K), field=field))
    M2 = IR.pmodule_from_fringe(one_by_one_fringe(P2,
                                                  FF.principal_upset(P2, 2),
                                                  FF.principal_downset(P2, 3);
                                                  scalar=one(K), field=field))
    sc = CM.SessionCache()

    out1 = TO.encode_pmodules_to_common_poset(M1, M2; session_cache=sc)
    out2 = TO.encode_pmodules_to_common_poset(M1, M2; session_cache=sc)

    @test out1.P === out2.P
    @test out1.pi1 === out2.pi1
    @test out1.pi2 === out2.pi2
    @test out1.Ms[1] === out2.Ms[1]
    @test out1.Ms[2] === out2.Ms[2]
end

@testset "ChangeOfPosets product refinement fast paths match generic translation" begin
    CO = TamerOp.ChangeOfPosets

    function _same_module_on_same_poset(A::MD.PModule, B::MD.PModule)
        @test FF.poset_equal(A.Q, B.Q)
        @test A.dims == B.dims
        for (u, v) in FF.cover_edges(A.Q)
            @test A.edge_maps[u, v] == B.edge_maps[u, v]
        end
    end

    function _hom_dim_same_q(A::MD.PModule{K}, B::MD.PModule{K}) where {K}
        A.Q === B.Q && return DF.dim(DF.Hom(A, B))
        Q = A.Q
        @test FF.poset_equal(Q, B.Q)
        Bb = MD.PModule{K}(Q, B.dims, B.edge_maps; field=B.field)
        return DF.dim(DF.Hom(A, Bb))
    end

    dense1 = chain_poset(2)
    dense2 = chain_poset(3)
    struct1 = FF.ProductOfChainsPoset((2,))
    struct2 = FF.ProductOfChainsPoset((3,))
    FF.build_cache!(struct1; cover=true, updown=true)
    FF.build_cache!(struct2; cover=true, updown=true)

    Md1 = IR.pmodule_from_fringe(one_by_one_fringe(dense1,
                                                   FF.principal_upset(dense1, 1),
                                                   FF.principal_downset(dense1, 2),
                                                   one(K); field=field))
    Md2 = IR.pmodule_from_fringe(one_by_one_fringe(dense2,
                                                   FF.principal_upset(dense2, 2),
                                                   FF.principal_downset(dense2, 3),
                                                   one(K); field=field))
    Ms1 = IR.pmodule_from_fringe(one_by_one_fringe(struct1,
                                                   FF.principal_upset(struct1, 1),
                                                   FF.principal_downset(struct1, 2),
                                                   one(K); field=field))
    Ms2 = IR.pmodule_from_fringe(one_by_one_fringe(struct2,
                                                   FF.principal_upset(struct2, 2),
                                                   FF.principal_downset(struct2, 3),
                                                   one(K); field=field))

    old_fast = CO._CHANGE_OF_POSETS_USE_PRODUCT_PROJECTION_FASTPATH[]
    old_fused = CO._CHANGE_OF_POSETS_USE_FUSED_PRODUCT_TRANSLATION[]
    old_direct_hom = CO._CHANGE_OF_POSETS_USE_DIRECT_HOM_COMMON_REFINEMENT[]
    try
        CO._CHANGE_OF_POSETS_USE_PRODUCT_PROJECTION_FASTPATH[] = false
        CO._CHANGE_OF_POSETS_USE_FUSED_PRODUCT_TRANSLATION[] = false
        dense_generic = TO.encode_pmodules_to_common_poset(Md1, Md2; use_cache=false, session_cache=nothing)
        struct_generic = TO.encode_pmodules_to_common_poset(Ms1, Ms2; use_cache=false, session_cache=nothing)

        CO._CHANGE_OF_POSETS_USE_PRODUCT_PROJECTION_FASTPATH[] = true
        CO._CHANGE_OF_POSETS_USE_FUSED_PRODUCT_TRANSLATION[] = true
        dense_fast = TO.encode_pmodules_to_common_poset(Md1, Md2; use_cache=false, session_cache=nothing)
        struct_fast = TO.encode_pmodules_to_common_poset(Ms1, Ms2; use_cache=false, session_cache=nothing)

        _same_module_on_same_poset(dense_fast.Ms[1], dense_generic.Ms[1])
        _same_module_on_same_poset(dense_fast.Ms[2], dense_generic.Ms[2])
        _same_module_on_same_poset(struct_fast.Ms[1], struct_generic.Ms[1])
        _same_module_on_same_poset(struct_fast.Ms[2], struct_generic.Ms[2])

        Hdense = TO.hom_common_refinement(Md1, Md2; use_cache=false, session_cache=nothing)
        Hstruct = TO.hom_common_refinement(Ms1, Ms2; use_cache=false, session_cache=nothing)
        @test TO.dim(Hdense) == _hom_dim_same_q(dense_fast.Ms[1], dense_fast.Ms[2])
        @test TO.dim(Hstruct) == _hom_dim_same_q(struct_fast.Ms[1], struct_fast.Ms[2])
        Bdense = DF.basis(Hdense)
        Bstruct = DF.basis(Hstruct)
        @test length(Bdense) == DF.dim(Hdense)
        @test length(Bstruct) == DF.dim(Hstruct)
        @test getfield(Hdense, :hom) === nothing
        @test getfield(Hstruct, :hom) === nothing
        @test TO.hom_dim_common_refinement(Md1, Md2; use_cache=false, session_cache=nothing) == DF.dim(Hdense)
        @test TO.hom_dim_common_refinement(Ms1, Ms2; use_cache=false, session_cache=nothing) == DF.dim(Hstruct)
        @test TO.has_nonzero_hom_common_refinement(Md1, Md2; use_cache=false, session_cache=nothing) ==
              (TO.hom_dim_common_refinement(Md1, Md2; use_cache=false, session_cache=nothing) > 0)
        @test TO.has_nonzero_hom_common_refinement(Ms1, Ms2; use_cache=false, session_cache=nothing) ==
              (TO.hom_dim_common_refinement(Ms1, Ms2; use_cache=false, session_cache=nothing) > 0)
        @test TO.hom_bidim_common_refinement(Md1, Md2; use_cache=false, session_cache=nothing).forward ==
              TO.hom_dim_common_refinement(Md1, Md2; use_cache=false, session_cache=nothing)
        if !isempty(Bdense)
            @test Bdense[1] isa MD.PMorphism{K}
            @test getfield(Hdense, :hom) === nothing
            @test getfield(Hdense, :translated) !== nothing
            @test getfield(Hdense, :basis_matrix) !== nothing
        end
        if !isempty(Bstruct)
            @test Bstruct[1] isa MD.PMorphism{K}
            @test getfield(Hstruct, :hom) === nothing
            @test getfield(Hstruct, :translated) !== nothing
            @test getfield(Hstruct, :basis_matrix) !== nothing
        end

        CO._CHANGE_OF_POSETS_USE_DIRECT_HOM_COMMON_REFINEMENT[] = false
        Hdense_fallback = TO.hom_common_refinement(Md1, Md2; use_cache=false, session_cache=nothing)
        Hstruct_fallback = TO.hom_common_refinement(Ms1, Ms2; use_cache=false, session_cache=nothing)
        CO._CHANGE_OF_POSETS_USE_DIRECT_HOM_COMMON_REFINEMENT[] = true
        Hdense_direct = TO.hom_common_refinement(Md1, Md2; use_cache=false, session_cache=nothing)
        Hstruct_direct = TO.hom_common_refinement(Ms1, Ms2; use_cache=false, session_cache=nothing)
        @test length(DF.basis(Hdense_direct)) == length(DF.basis(Hdense_fallback))
        @test length(DF.basis(Hstruct_direct)) == length(DF.basis(Hstruct_fallback))

        @test length(collect(DF.basis(Hdense_direct))) == length(DF.basis(Hdense_direct))
        @test length(collect(DF.basis(Hstruct_direct))) == length(DF.basis(Hstruct_direct))

        prod_struct = TO.product_poset(struct1, struct2; use_cache=false, session_cache=nothing)
        pb_fast = TO.pullback(prod_struct.pi1, Ms1; session_cache=nothing)
        CO._CHANGE_OF_POSETS_USE_PRODUCT_PROJECTION_FASTPATH[] = false
        pb_generic = TO.pullback(prod_struct.pi1, Ms1; session_cache=nothing)
        _same_module_on_same_poset(pb_fast, pb_generic)
    finally
        CO._CHANGE_OF_POSETS_USE_PRODUCT_PROJECTION_FASTPATH[] = old_fast
        CO._CHANGE_OF_POSETS_USE_FUSED_PRODUCT_TRANSLATION[] = old_fused
        CO._CHANGE_OF_POSETS_USE_DIRECT_HOM_COMMON_REFINEMENT[] = old_direct_hom
    end
end

@testset "ChangeOfPosets common-refinement UX surfaces" begin
    CO = TamerOp.ChangeOfPosets

    P1 = chain_poset(2)
    P2 = chain_poset(3)
    M1 = IR.pmodule_from_fringe(one_by_one_fringe(P1,
                                                  FF.principal_upset(P1, 1),
                                                  FF.principal_downset(P1, 2),
                                                  one(K); field=field))
    M2 = IR.pmodule_from_fringe(one_by_one_fringe(P2,
                                                  FF.principal_upset(P2, 2),
                                                  FF.principal_downset(P2, 3),
                                                  one(K); field=field))

    H = TO.hom_common_refinement(M1, M2; use_cache=false, session_cache=nothing)
    B = DF.basis(H)

    dH = TO.describe(H)
    dB = TO.describe(B)
    @test dH.kind == :common_refinement_hom_space
    @test dH.dimension == TO.dim(H)
    @test dH.source_nvertices == FF.nvertices(P1)
    @test dH.target_nvertices == FF.nvertices(P2)
    @test dH.source_total_dim == sum(M1.dims)
    @test dH.target_total_dim == sum(M2.dims)
    @test dB.kind == :common_refinement_hom_basis
    @test dB.dimension == TO.dim(H)
    @test dB.source_total_dim == dH.source_total_dim
    @test dB.target_total_dim == dH.target_total_dim

    @test TOA.CommonRefinementHomSpace === CO.CommonRefinementHomSpace
    @test TOA.CommonRefinementHomBasis === CO.CommonRefinementHomBasis
    @test TOA.MonotoneMapValidationSummary === CO.MonotoneMapValidationSummary
    @test TOA.CommonRefinementHomValidationSummary === CO.CommonRefinementHomValidationSummary
    @test TOA.common_refinement_summary === CO.common_refinement_summary
    @test TOA.common_refinement_hom_validation_summary === CO.common_refinement_hom_validation_summary
    @test TOA.monotone_map_validation_summary === CO.monotone_map_validation_summary
    @test TOA.check_common_refinement_hom === CO.check_common_refinement_hom
    @test TOA.check_monotone_map === CO.check_monotone_map
    @test TOA.basis_matrix === CO.basis_matrix

    @test CO.common_refinement_summary(H) == dH
    @test CO.common_refinement_summary(B) == dB
    @test TOA.source(H) === M1
    @test TOA.target(H) === M2
    @test TOA.source(B) === M1
    @test TOA.target(B) === M2

    Bmat = CO.basis_matrix(H)
    @test size(Bmat, 2) == TO.dim(H)
    @test CO.basis_matrix(B) === Bmat
    @test TO.describe(H).basis_matrix_materialized

    report = CO.check_common_refinement_hom(H)
    @test report isa CO.CommonRefinementHomValidationSummary
    @test report.valid
    @test CO.check_common_refinement_hom(B).valid

    Hbad = deepcopy(H)
    setfield!(Hbad, :dim_cached, -1)
    bad_report = CO.check_common_refinement_hom(Hbad)
    @test bad_report isa CO.CommonRefinementHomValidationSummary
    @test !bad_report.valid
    @test !isempty(bad_report.issues)
    @test_throws ArgumentError CO.check_common_refinement_hom(Hbad; throw=true)

    pi_valid = EN.EncodingMap(P1, P1, [1, 2])
    pi_bad = EN.EncodingMap(P1, P1, [2, 1])
    @test CO.check_monotone_map(pi_valid).valid
    monotone_report = CO.check_monotone_map(pi_bad)
    @test monotone_report isa CO.MonotoneMapValidationSummary
    @test !monotone_report.valid
    @test !isempty(monotone_report.issues)
    @test_throws ArgumentError CO.check_monotone_map(pi_bad; throw=true)
    @test CO.common_refinement_hom_validation_summary(report) === report
    @test CO.monotone_map_validation_summary(monotone_report) === monotone_report

    show_H = sprint(show, H)
    show_B = sprint(show, B)
    show_report = sprint(show, report)
    show_monotone = sprint(show, monotone_report)
    plain_H = sprint(io -> show(io, MIME"text/plain"(), H))
    plain_B_before = sprint(io -> show(io, MIME"text/plain"(), B))
    plain_report = sprint(io -> show(io, MIME"text/plain"(), report))
    plain_monotone = sprint(io -> show(io, MIME"text/plain"(), monotone_report))
    @test occursin("CommonRefinementHomSpace", show_H)
    @test occursin("dimension=", show_H)
    @test occursin("source_nvertices:", plain_H)
    @test occursin("basis_matrix_materialized:", plain_H)
    @test occursin("CommonRefinementHomBasis", show_B)
    @test occursin("hom_materialized: false", plain_B_before)
    @test occursin("CommonRefinementHomValidationSummary", show_report)
    @test occursin("MonotoneMapValidationSummary", show_monotone)
    @test occursin("valid: true", plain_report)
    @test occursin("valid: false", plain_monotone)

    collected_basis = collect(B)
    @test length(collected_basis) == TO.dim(H)
    plain_B_after = sprint(io -> show(io, MIME"text/plain"(), B))
    @test occursin("hom_materialized: true", plain_B_after)

    describe_doc = repr(MIME"text/plain"(), @doc TamerOp.ChainComplexes.describe)
    homspace_doc = repr(MIME"text/plain"(), @doc CO.CommonRefinementHomSpace)
    restriction_doc = repr(MIME"text/plain"(), @doc CO.restriction)
    left_doc = repr(MIME"text/plain"(), @doc CO.pushforward_left)
    right_doc = repr(MIME"text/plain"(), @doc CO.pushforward_right)
    translate_doc = repr(MIME"text/plain"(), @doc CO.encode_pmodules_to_common_poset)
    @test occursin("CommonRefinementHomSpace", describe_doc)
    @test occursin("Lazy common-refinement Hom space", homspace_doc)
    @test occursin("basis_matrix", homspace_doc)
    @test occursin("canonical notebook-facing name", restriction_doc)
    @test occursin("canonical notebook-facing name", left_doc)
    @test occursin("canonical notebook-facing name", right_doc)
    @test occursin("CommonRefinementTranslationResult", translate_doc)
end

@testset "Sparse triplet assembly helper" begin
    # Dense block, offset placement
    A = K[0 2;
           3 0]
    I = Int[]; J = Int[]; V = K[]
    CM._append_scaled_triplets!(I, J, V, A, 0, 0)
    S = sparse(I, J, V, 2, 2)
    @test S == sparse(A)

    # Scaled block into larger matrix
    I = Int[]; J = Int[]; V = K[]
    CM._append_scaled_triplets!(I, J, V, A, 1, 2; scale = -c(2))
    S = sparse(I, J, V, 3, 4)
    R = spzeros(K, 3, 4)
    R[2:3, 3:4] = -c(2) * A
    @test S == R

    # Indexed (noncontiguous) placement
    rows = [2, 4]
    cols = [1, 3]
    B = K[1 0;
           0 2]
    I = Int[]; J = Int[]; V = K[]
    CM._append_scaled_triplets!(I, J, V, B, rows, cols)
    S = sparse(I, J, V, 5, 5)
    R = spzeros(K, 5, 5)
    R[rows, cols] = B
    @test S == R
end

@testset "Hom differential assembly matches dense reference" begin
    # poset 1 <= 2
    leq = Bool[true true;
               false true]
    P = FF.FinitePoset(leq; check=true)

    dims = [1, 1]
    edge_maps = Dict((1,2) => K[1;;])  # 1x1 matrix
    N = MD.PModule(P, dims, edge_maps)

    # dummy resolution data for _build_hom_differential
    gens = Vector{Vector{Int}}(undef, 2)
    gens[1] = [1]      # degree 0 gens
    gens[2] = [1, 2]   # degree 1 gens

    # Coefficient matrix of d_1 : P_1 -> P_0 (rows = cod summands, cols = dom summands)
    # delta is stored with rows = codomain summands (P_0) and cols = domain summands (P_1).
    # Here P_0 has 1 summand (base vertex 1) and P_1 has 2 summands (base vertices 1,2).
    delta = sparse([1,1], [1,2], [c(1), c(1)], 1, 2)

    # Minimal resolution object: only `gens` and `d_mat` are used by _build_hom_differential,
    # but the struct requires the full field list.
    Pmods = [N, N]  # placeholders (not used in this test)

    # dummy morphisms (not used by _build_hom_differential; only needed to build the struct)
    dummy_pm = MD.id_morphism(N)

    # Canonical ProjectiveResolution API: (M, Pmods, gens, d_mor, d_mat, aug)
    res = DF.ProjectiveResolution(N, Pmods, gens, [dummy_pm], [delta], dummy_pm)

    offs_cod = [0, N.dims[1]]                       # [0,1]
    offs_dom = [0, N.dims[1], N.dims[1]+N.dims[2]]  # [0,1,2]

    Dnew = DF.ExtTorSpaces._build_hom_differential(res, N, 1, offs_cod, offs_dom)
    @test issparse(Dnew)

    # dense reference:
    Dref = zeros(K, 2, 1)
    Dref[1,1] = 1   # map_leq(N,1,1)
    Dref[2,1] = 1   # map_leq(N,1,2)
    @test Matrix(Dnew) == Dref
end

@testset "CoverEdgeMapStore equality and rebasing" begin
    P = chain_poset(3)
    dims = [1,1,1]
    edge_maps = Dict{Tuple{Int,Int}, Matrix{K}}()
    for (u,v) in IR.cover_edges(P)
        edge_maps[(u,v)] = K[1;;]
    end

    M1 = MD.PModule{K}(P, dims, edge_maps)
    M2 = MD.PModule{K}(P, dims, edge_maps)

    # Structural equality of stores (not pointer equality)
    @test M1.edge_maps == M2.edge_maps

    # Rebase onto an "equivalent" poset object (same leq matrix, new instance)
    P2 = FF.FinitePoset(copy(FF.leq_matrix(P)))
    M3 = MD.PModule{K}(P2, M1.dims, M1.edge_maps)  # should not error
    @test M3.dims == M1.dims
    @test M3.edge_maps == M1.edge_maps
end

@testset "Store-aligned cover-edge iteration" begin
    # Small non-chain poset with branching.
    Q = diamond_poset()

    # Deterministic module on Q.
    dims = [2, 1, 1, 2]

    # Cover maps: (1,2), (1,3), (2,4), (3,4).
    edge = Dict{Tuple{Int,Int}, Matrix{K}}()
    edge[(1,2)] = K[1 0]          # 1x2
    edge[(1,3)] = K[0 1]          # 1x2
    # NOTE: use 2x1 matrices (not length-2 vectors) for cover-edge maps.
    edge[(2,4)] = K[1; 0;;]       # 2x1
    edge[(3,4)] = K[0; 1;;]       # 2x1

    M = MD.PModule{K}(Q, dims, edge)

    store = M.edge_maps
    succs = store.succs
    preds = store.preds

    # (1) Store-aligned traversal agrees with keyed lookup.
    @testset "maps_to_succ agrees with getindex" begin
        for u in 1:Q.n
            su = succs[u]
            Mu = store.maps_to_succ[u]
            for j in eachindex(su)
                v = su[j]
                A1 = Mu[j]
                A2 = M.edge_maps[u, v]
                A3 = M.edge_maps[u, v]
                @test A1 == A2
                @test A1 == A3
            end
        end
    end

    # (2) maps_to_succ and maps_from_pred are consistent.
    @testset "maps_from_pred consistency" begin
        for u in 1:Q.n
            su = succs[u]
            Mu = store.maps_to_succ[u]
            for j in eachindex(su)
                v = su[j]
                ip = MD._find_sorted_index(preds[v], u)
                @test store.maps_from_pred[v][ip] == Mu[j]
            end
        end
    end

    # (3) Kernel/image routines still commute on edges after the store-direct rewrite.
    @testset "kernel/image commuting diagrams" begin
        # Simple chain so we can reason about results.
        Qc = chain_poset(3)

        dimsC = [1, 1, 1]
        edgeC = Dict{Tuple{Int,Int}, Matrix{K}}()
        edgeC[(1,2)] = K[1;;]
        edgeC[(2,3)] = K[1;;]
        Mc = MD.PModule{K}(Qc, dimsC, edgeC)

        # Zero map Mc -> Mc.
        fcomps = [zeros(K, 1, 1) for _ in 1:3]
        f = MD.PMorphism(Mc, Mc, fcomps)

        K, iotaK = TO.kernel_with_inclusion(f)
        Im, iotaIm = TO.image_with_inclusion(f)

        # Kernel of zero map should be the whole module.
        @test K.dims == Mc.dims

        # Image of zero map should be the zero module.
        @test all(Im.dims .== 0)

        # Commutativity on cover edges:
        # Mc(u->v) * iotaK[u] == iotaK[v] * K(u->v)
        for (u, v) in FF.cover_edges(Qc)
            @test Mc.edge_maps[u, v] * iotaK.comps[u] == iotaK.comps[v] * K.edge_maps[u, v]
            @test Mc.edge_maps[u, v] * iotaIm.comps[u] == iotaIm.comps[v] * Im.edge_maps[u, v]
        end
    end
end
end # with_fields
