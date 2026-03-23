with_fields(FIELDS_FULL) do field
K = CM.coeff_type(field)
@inline c(x) = CM.coerce(field, x)

@testset "FiniteFringe basics" begin
    P = chain_poset(3)

    # cover edges on a 3-chain should be 1->2 and 2->3 only
    C = FF.cover_edges(P)
    @test C[1,2] == true
    @test C[2,3] == true
    @test C[1,3] == false

    # principal sets
    U2 = FF.principal_upset(P, 2)      # {2,3}
    D2 = FF.principal_downset(P, 2)    # {1,2}
    @test U2.mask == BitVector([false, true,  true])
    @test D2.mask == BitVector([true,  true,  false])
    @test_throws ErrorException FF.Upset(P, BitVector([true, false]))
    @test_throws ErrorException FF.Downset(P, BitVector([true, false]))

    # 1x1 interval module supported at {2}
    phi = spzeros(K, 1, 1)
    phi[1,1] = c(1)
    M = FF.FringeModule{K}(P, [U2], [D2], phi; field=field)
    @test_throws UndefKeywordError FF.FringeModule{K}(P, [U2], [D2], phi)
    @test_throws MethodError FF.FringeModule(P, [U2], [D2], phi; field=field)
    @test_throws MethodError FF.FringeModule(P, [U2], [D2], phi; store_sparse=true, field=field)

    @test FF.fiber_dimension(M, 1) == 0
    @test FF.fiber_dimension(M, 2) == 1
    @test FF.fiber_dimension(M, 3) == 0

    # Endomorphisms of a connected indicator should be 1-dimensional
    @test FF.hom_dimension(M, M) == 1

    # Monomial condition should reject nonzero entry when U cap D is empty
    U3 = FF.principal_upset(P, 3)      # {3}
    D1 = FF.principal_downset(P, 1)    # {1}
    phi_bad = spzeros(K, 1, 1)
    phi_bad[1,1] = c(1)
    @test_throws AssertionError FF.FringeModule{K}(P, [U3], [D1], phi_bad; field=field)


        @testset "FinitePoset validation errors" begin
        # Not reflexive.
        leq1 = trues(2, 2)
        leq1[1, 1] = false
        @test_throws ErrorException FF.FinitePoset(leq1)

        # Violates antisymmetry: 1<=2 and 2<=1.
        leq2 = trues(2, 2)
        @test_throws ErrorException FF.FinitePoset(leq2)

        # Violates transitivity: 1<=2, 2<=3, but not 1<=3.
        leq3 = falses(3, 3)
        for i in 1:3
            leq3[i, i] = true
        end
        leq3[1, 2] = true
        leq3[2, 3] = true
        @test_throws ErrorException FF.FinitePoset(leq3)
    end

    @testset "FinitePoset unchecked constructor" begin
        # Same leq3 as above: non-transitive but reflexive/antisymmetric.
        leq3 = falses(3, 3)
        for i in 1:3
            leq3[i, i] = true
        end
        leq3[1, 2] = true
        leq3[2, 3] = true

        # With check=false we should be able to build it (unsafe).
        P_bad = FF.FinitePoset(leq3; check=false)
        @test P_bad.n == 3
    end

    @testset "cover_edges correctness and caching" begin
        Random.seed!(12345)

        function transitive_closure!(R::BitMatrix)
            n = size(R, 1)
            @inbounds for k in 1:n
                for i in 1:n
                    if R[i, k]
                        for j in 1:n
                            R[i, j] |= R[k, j]
                        end
                    end
                end
            end
            return R
        end

        function naive_cover_edges(leq::BitMatrix)
            n = size(leq, 1)
            mat = falses(n, n)
            edges = Tuple{Int,Int}[]
            @inbounds for i in 1:n
                for j in 1:n
                    if i != j && leq[i, j]
                        covered = true
                        for k in 1:n
                            if k != i && k != j && leq[i, k] && leq[k, j]
                                covered = false
                                break
                            end
                        end
                        if covered
                            mat[i, j] = true
                            push!(edges, (i, j))
                        end
                    end
                end
            end
            return mat, edges
        end

        for n in (5, 8)
            # Random strict order edges (upper triangular), then take transitive closure.
            R = falses(n, n)
            @inbounds for i in 1:n
                R[i, i] = true
                for j in (i + 1):n
                    if rand() < 0.25
                        R[i, j] = true
                    end
                end
            end
            transitive_closure!(R)

            P = FF.FinitePoset(R; check=false)
            C1 = FF.cover_edges(P)

            # Regression tests: CoverEdges is iterable over edges, so BitMatrix(C) and
            # Matrix(C) must be explicitly specialized (otherwise Base tries the
            # generic iterator constructor and errors).
            @test BitMatrix(C1) === convert(BitMatrix, C1)
            @test Matrix(C1) == Matrix(BitMatrix(C1))

            C2 = FF.cover_edges(P)
            @test C1 === C2  # cached
            @test P.cache.cover_edges === C1

            # Packed cover-cache adjacency and slot maps should be mutually consistent.
            FF.build_cache!(P; cover=true, updown=false)
            cc = P.cache.cover
            @test cc !== nothing
            @test length(cc.succ_ptr) == n + 1
            @test length(cc.pred_ptr) == n + 1
            @test length(cc.succ_idx) == cc.nedges
            @test length(cc.pred_idx) == cc.nedges
            @test length(cc.succ_pred_slot) == cc.nedges
            @test length(cc.pred_succ_slot) == cc.nedges
            for u in 1:n
                su = FF._succs(cc, u)
                ps = FF._pred_slots_of_succ(cc, u)
                @test length(ps) == length(su)
                for j in eachindex(su)
                    v = su[j]
                    i = ps[j]
                    pv = FF._preds(cc, v)
                    sv = FF._succ_slots_of_pred(cc, v)
                    @test 1 <= i <= length(pv)
                    @test pv[i] == u
                    @test sv[i] == j
                end
            end

            # cached=false should force recomputation (new object) but same result.
            C3 = FF.cover_edges(P; cached=false)
            @test C3 !== C1
            @test BitMatrix(C3) == BitMatrix(C1)
            @test findall(C3) == findall(C1)

            FF._clear_cover_cache!(P)
            @test P.cache.cover_edges === nothing
            C4 = FF.cover_edges(P)
            @test C4 !== C1
            @test BitMatrix(C4) == BitMatrix(C1)

            mat_ref, edges_ref = naive_cover_edges(R)
            @test BitMatrix(C1) == mat_ref
            @test findall(C1) == edges_ref
        end
    end


    @testset "Upset/downset closure and generators" begin
        P4 = chain_poset(4)
        S = falses(P4.n)
        S[2] = true

        Ucl = FF.upset_closure(P4, S)
        Dcl = FF.downset_closure(P4, S)

        @test Ucl.mask == BitVector([false, true, true, true])
        @test Dcl.mask == BitVector([true, true, false, false])

        Pdia = diamond_poset()
        Ugen = FF.upset_from_generators(Pdia, [2, 3])
        Dgen = FF.downset_from_generators(Pdia, [2, 3])
        Ugen_mask = FF.upset_from_generators(Pdia, BitVector([false, true, true, false]))
        Dgen_mask = FF.downset_from_generators(Pdia, BitVector([false, true, true, false]))

        @test Ugen.mask == BitVector([false, true, true, true])  # {2,3,4}
        @test Dgen.mask == BitVector([true, true, true, false])  # {1,2,3}
        @test Ugen_mask.mask == Ugen.mask
        @test Dgen_mask.mask == Dgen.mask
        @test_throws ErrorException FF.upset_from_generators(Pdia, [0, 2])
        @test_throws ErrorException FF.downset_from_generators(Pdia, [2, 99])
        @test_throws ErrorException FF.upset_from_generators(Pdia, BitVector([true, false]))
        @test_throws ErrorException FF.downset_from_generators(Pdia, BitVector([true, false]))
    end

    @testset "cover_edges on a non-chain (diamond) poset" begin
        Pdia = diamond_poset()
        C = FF.cover_edges(Pdia)
        edges = Set([(I[1], I[2]) for I in findall(C)])
        @test edges == Set([(1, 2), (1, 3), (2, 4), (3, 4)])
    end


    @testset "Dense phi branch and _dense_to_sparse_K" begin
        P3 = chain_poset(3)
        U2 = FF.principal_upset(P3, 2)
        D2 = FF.principal_downset(P3, 2)

        # Dense 1x1 phi exercises the non-sparse path in _check_monomial_condition.
        phi_dense = reshape(K[c(1)], 1, 1)
        Mdense = FF.FringeModule{K}(P3, [U2], [D2], phi_dense; field=field)
        @test FF.fiber_dimension(Mdense, 2) == 1

        # Converting dense -> sparse should preserve values.
        phi_sparse = FF._dense_to_sparse_K(phi_dense)
        @test phi_sparse isa SparseMatrixCSC{K, Int}
        @test Matrix(phi_sparse) == phi_dense

        # Sparse phi should behave the same.
        Msparse = FF.FringeModule{K}(P3, [U2], [D2], phi_sparse; field=field)
        @test FF.fiber_dimension(Msparse, 2) == 1

        # Dense constructor should reject nonzero phi when U cap D is empty.
        U3 = FF.principal_upset(P3, 3)
        D1 = FF.principal_downset(P3, 1)
        @test_throws AssertionError FF.FringeModule{K}(P3, [U3], [D1], reshape(K[c(1)], 1, 1); field=field)
    end

    @testset "FiniteFringe exported API direct coverage" begin
        # FiniteFringeOptions direct constructor/field contract.
        opts = FF.FiniteFringeOptions()
        @test opts.check === true
        @test opts.cached === true
        @test opts.store_sparse === false
        @test opts.scalar == 1
        @test opts.poset_kind == :regions
        opts2 = FF.FiniteFringeOptions(check=false, cached=false, store_sparse=true, scalar=QQ(3), poset_kind=:dense)
        @test opts2.check === false
        @test opts2.cached === false
        @test opts2.store_sparse === true
        @test opts2.scalar == QQ(3)
        @test opts2.poset_kind == :dense

        # leq_row / leq_col direct API (FinitePoset specialization).
        P = chain_poset(4)
        L = FF.leq_matrix(P)
        row2 = FF.leq_row(P, 2)
        col3 = FF.leq_col(P, 3)
        @test length(row2) == P.n
        @test length(col3) == P.n
        @test all(Bool(row2[j]) == L[2, j] for j in 1:P.n)
        @test all(Bool(col3[i]) == L[i, 3] for i in 1:P.n)

        # leq_row / leq_col on structured posets go through upset/downset APIs.
        Ps = FF.ProductOfChainsPoset((3, 2))
        @test FF.leq_row(Ps, 2) == FF.upset_indices(Ps, 2)
        @test FF.leq_col(Ps, 2) == FF.downset_indices(Ps, 2)

        # poset_equal_opposite direct contract.
        Pop = FF.FinitePoset(transpose(FF.leq_matrix(P)); check=false)
        @test FF.poset_equal_opposite(P, Pop)
        @test !FF.poset_equal_opposite(P, P)
        P1 = chain_poset(1)
        @test FF.poset_equal_opposite(P1, P1)

        # FF-level change_field (direct API), including semantic change mod 2.
        Pone = chain_poset(1)
        U = FF.principal_upset(Pone, 1)
        D = FF.principal_downset(Pone, 1)
        Hqq = one_by_one_fringe(Pone, U, D, QQ(2); field=CM.QQField())
        Hf2 = FF.change_field(Hqq, CM.F2())
        @test Hf2.field isa typeof(CM.F2())
        @test Hf2.P === Hqq.P
        @test Hf2.U === Hqq.U
        @test Hf2.D === Hqq.D
        @test FF.fiber_dimension(Hqq, 1) == 1
        @test FF.fiber_dimension(Hf2, 1) == 0
    end

    @testset "FiniteFringe UX surface" begin
        Pux = chain_poset(3)
        Uux = FF.principal_upset(Pux, 2)
        Dux = FF.principal_downset(Pux, 2)
        phiux = spzeros(K, 1, 1)
        phiux[1, 1] = c(1)
        Mux = FF.FringeModule{K}(Pux, [Uux], [Dux], phiux; field=field)

        @test TO.describe(Pux).kind == :finite_poset
        @test TO.describe(Uux).kind == :upset
        @test TO.describe(Dux).kind == :downset
        @test TO.describe(Mux).kind == :fringe_module
        @test TO.describe(Mux).ngenerators == 1
        @test TO.describe(Mux).nrelations == 1
        @test FF.fringe_summary(Pux).kind == :finite_poset
        @test FF.fringe_summary(Uux).support == [2, 3]
        @test FF.fringe_summary(Dux).support == [1, 2]
        @test FF.fringe_summary(Mux).matrix_size == (1, 1)

        @test FF.check_poset(Pux).valid
        @test FF.check_upset(Uux).valid
        @test FF.check_downset(Dux).valid
        @test FF.check_fringe_module(Mux).valid
        @test FF.check_fringe_data(Pux, [Uux], [Dux], phiux; field=field).valid
        @test FF.finite_fringe_validation_summary(FF.check_fringe_module(Mux)).report.valid

        Ubad = FF.Upset(Pux, BitVector([false, true, false]))
        Dbad = FF.Downset(Pux, BitVector([false, true, false]))
        @test !FF.check_upset(Ubad).valid
        @test !FF.check_downset(Dbad).valid
        @test_throws ArgumentError FF.check_upset(Ubad; throw=true)
        @test_throws ArgumentError FF.check_downset(Dbad; throw=true)

        U3 = FF.principal_upset(Pux, 3)
        D1 = FF.principal_downset(Pux, 1)
        phibad = spzeros(K, 1, 1)
        phibad[1, 1] = c(1)
        @test !FF.check_fringe_data(Pux, [U3], [D1], phibad; field=field).valid

        @test FF.base_poset(Uux) === Pux
        @test FF.base_poset(Dux) === Pux
        @test FF.base_poset(Mux) === Pux
        @test FF.ambient_poset(Pux) === Pux
        @test FF.ambient_poset(Uux) === Pux
        @test FF.ambient_poset(Dux) === Pux
        @test FF.ambient_poset(Mux) === Pux
        @test FF.field(Mux) === field
        @test FF.support(Uux) == [2, 3]
        @test FF.support(Dux) == [1, 2]
        @test FF.contains(Uux, 2)
        @test !FF.contains(Uux, 1)
        @test FF.contains(Dux, 1)
        @test !FF.contains(Dux, 3)
        @test FF.birth_upsets(Mux) === Mux.U
        @test FF.death_downsets(Mux) === Mux.D
        @test FF.fringe_coefficients(Mux) === Mux.phi
        @test FF.ngenerators(Mux) == 1
        @test FF.nrelations(Mux) == 1
        @test TO.dimensions(Mux).fibers == [0, 1, 0]
        @test TO.dimensions(Mux).total == 1
        @test TO.dimensions(Mux, 2).fiber == 1

        @test TOA.base_poset(Mux) === Pux
        @test TOA.ambient_poset(Mux) === Pux
        @test TOA.field(Mux) === field
        @test TOA.ngenerators(Mux) == 1
        @test TOA.nrelations(Mux) == 1
        @test TOA.check_fringe_module(Mux).valid
        @test TOA.fringe_summary(Mux).kind == :fringe_module

        @test occursin("FinitePoset", sprint(show, Pux))
        @test occursin("Upset", sprint(show, Uux))
        @test occursin("Downset", sprint(show, Dux))
        @test occursin("FringeModule", sprint(show, Mux))
        @test occursin("nvertices", sprint(show, MIME("text/plain"), Pux))
        @test occursin("support", sprint(show, MIME("text/plain"), Uux))
        @test occursin("matrix_size", sprint(show, MIME("text/plain"), Mux))
        @test occursin("FiniteFringeValidationSummary", sprint(show, FF.finite_fringe_validation_summary(FF.check_fringe_module(Mux))))
        @test occursin("issues", sprint(show, MIME("text/plain"), FF.finite_fringe_validation_summary(FF.check_fringe_module(Mux))))
    end

    @testset "FiniteFringe threaded cache/query parity under contention" begin
        if !(field isa CM.QQField) || Threads.nthreads() <= 1
            @test true
        else
            rng = MersenneTwister(0xCACE)
            P = chain_poset(96)
            FF.build_cache!(P; cover=true, updown=true)

            nu, nd = 24, 23
            U = [FF.principal_upset(P, rand(rng, 1:P.n)) for _ in 1:nu]
            D = [FF.principal_downset(P, rand(rng, 1:P.n)) for _ in 1:nd]
            phi = spzeros(K, nd, nu)
            @inbounds for j in 1:nd, i in 1:nu
                FF.intersects(U[i], D[j]) || continue
                rand(rng) < 0.22 || continue
                v = rand(rng, -2:2)
                v == 0 && continue
                phi[j, i] = c(v)
            end
            M = FF.FringeModule{K}(P, U, D, phi; field=field)

            # Two-phase warmup: all caches built sequentially, then read-only threaded queries.
            for q in 1:P.n
                FF.upset_indices(P, q)
                FF.downset_indices(P, q)
                FF.fiber_dimension(M, q)
            end

            qcount = 4_000
            queries = [rand(rng, 1:P.n) for _ in 1:qcount]
            expected = Vector{Int}(undef, qcount)
            @inbounds for i in 1:qcount
                q = queries[i]
                expected[i] = 10_000 * FF.fiber_dimension(M, q) +
                              100 * length(FF.upset_indices(P, q)) +
                              length(FF.downset_indices(P, q))
            end

            nruns = max(8, 4 * Threads.nthreads())
            got = Vector{Vector{Int}}(undef, nruns)
            Threads.@threads :static for r in 1:nruns
                out = Vector{Int}(undef, qcount)
                @inbounds for i in 1:qcount
                    q = queries[i]
                    out[i] = 10_000 * FF.fiber_dimension(M, q) +
                             100 * length(FF.upset_indices(P, q)) +
                             length(FF.downset_indices(P, q))
                end
                got[r] = out
            end

            for r in 1:nruns
                @test got[r] == expected
            end
        end
    end
end

@testset "build_cache! two-phase warmup" begin
    P = chain_poset(6)
    @test P.cache.cover === nothing
    @test P.cache.upsets === nothing
    @test P.cache.downsets === nothing

    FF.build_cache!(P; cover=true, updown=true)
    @test P.cache.cover !== nothing
    @test P.cache.upsets !== nothing
    @test P.cache.downsets !== nothing

    cc1 = P.cache.cover
    FF.build_cache!(P; cover=true, updown=true)
    @test P.cache.cover === cc1
end

if field isa CM.QQField
@testset "FiniteFringe normalized performance envelopes (QQ)" begin
    field = CM.QQField()
    K = CM.coeff_type(field)
    c(x) = CM.coerce(field, x)
    strict_ci = get(ENV, "TAMER_STRICT_PERF_CI", "1") == "1"
    reps = strict_ci ? 5 : 3

    @inline function _median_elapsed(f::Function; reps::Int=5)
        ts = Vector{Float64}(undef, reps)
        for i in 1:reps
            ts[i] = @elapsed f()
        end
        sort!(ts)
        return ts[cld(reps, 2)]
    end
    @inline _ns_per_item(t::Float64, n::Int) = (1.0e9 * t) / max(1, n)

    # 1) fiber_dimension query index path vs old scan baseline.
    P = chain_poset(72)
    rng = MersenneTwister(0x51F1E)
    nu, nd = 26, 25
    U = [FF.principal_upset(P, rand(rng, 1:P.n)) for _ in 1:nu]
    D = [FF.principal_downset(P, rand(rng, 1:P.n)) for _ in 1:nd]
    phi = spzeros(K, nd, nu)
    @inbounds for j in 1:nd, i in 1:nu
        FF.intersects(U[i], D[j]) || continue
        rand(rng) < 0.20 || continue
        v = rand(rng, -2:2)
        v == 0 && continue
        phi[j, i] = c(v)
    end
    M = FF.FringeModule{K}(P, U, D, phi; field=field)
    queries = [rand(rng, 1:P.n) for _ in 1:6_000]

    fiber_scan_ref(M::FF.FringeModule{K}, q::Int) where {K} = begin
        cols = findall(U0 -> U0.mask[q], M.U)
        rows = findall(D0 -> D0.mask[q], M.D)
        (isempty(cols) || isempty(rows)) && return 0
        FF.FieldLinAlg.rank_restricted(M.field, M.phi, rows, cols)
    end
    old_vals = Vector{Int}(undef, length(queries))
    @inbounds for i in eachindex(queries)
        old_vals[i] = fiber_scan_ref(M, queries[i])
    end
    new_vals = [FF.fiber_dimension(M, q) for q in queries]
    @test new_vals == old_vals

    t_old = _median_elapsed(reps=reps) do
        s = 0
        @inbounds for q in queries
            s += fiber_scan_ref(M, q)
        end
        @test s >= 0
    end
    t_new = _median_elapsed(reps=reps) do
        s = 0
        @inbounds for q in queries
            s += FF.fiber_dimension(M, q)
        end
        @test s >= 0
    end
    old_ns = _ns_per_item(t_old, length(queries))
    new_ns = _ns_per_item(t_new, length(queries))
    if strict_ci
        @test new_ns <= 0.95 * old_ns + 120.0
    else
        @test new_ns <= 1.15 * old_ns + 250.0
    end

    # 2) _uptight_regions fast grouping vs tuple-dict baseline.
    Q = chain_poset(120)
    Y = Vector{FF.Upset}(undef, 160)
    @inbounds for i in eachindex(Y)
        Y[i] = FF.principal_upset(Q, rand(rng, 1:Q.n))
    end
    naive_uptight_regions(Q, Y) = begin
        sigs = Dict{Tuple{Vararg{Bool}}, Vector{Int}}()
        @inbounds for q in 1:FF.nvertices(Q)
            key = ntuple(i -> Y[i].mask[q], length(Y))
            vec = get!(sigs, key) do
                Int[]
            end
            push!(vec, q)
        end
        collect(values(sigs))
    end
    canon(regs) = sort([sort(copy(r)) for r in regs], by = v -> (length(v), first(v)))
    regs_old = naive_uptight_regions(Q, Y)
    regs_new = EN._uptight_regions(Q, Y)
    @test canon(regs_new) == canon(regs_old)

    t_uptight_old = _median_elapsed(reps=reps) do
        r = naive_uptight_regions(Q, Y)
        @test !isempty(r)
    end
    t_uptight_new = _median_elapsed(reps=reps) do
        r = EN._uptight_regions(Q, Y)
        @test !isempty(r)
    end
    old_cell_ns = _ns_per_item(t_uptight_old, Q.n)
    new_cell_ns = _ns_per_item(t_uptight_new, Q.n)
    if strict_ci
        @test new_cell_ns <= 0.95 * old_cell_ns + 150.0
    else
        @test new_cell_ns <= 1.2 * old_cell_ns + 300.0
    end

    # 3) hom_dimension auto route should match best internal mode on sparse fixtures.
    P_h = chain_poset(36)
    rng_h = MersenneTwister(0x7A11A)
    nu_h, nd_h = 20, 19
    U_h = [FF.principal_upset(P_h, rand(rng_h, 1:P_h.n)) for _ in 1:nu_h]
    D_h = [FF.principal_downset(P_h, rand(rng_h, 1:P_h.n)) for _ in 1:nd_h]

    phi_hM = spzeros(K, nd_h, nu_h)
    phi_hN = spzeros(K, nd_h, nu_h)
    @inbounds for j in 1:nd_h, i in 1:nu_h
        FF.intersects(U_h[i], D_h[j]) || continue
        rand(rng_h) < 0.12 || continue
        v = rand(rng_h, -2:2)
        v == 0 && continue
        phi_hM[j, i] = c(v)
        rand(rng_h) < 0.7 && (phi_hN[j, i] = c(v == 0 ? 1 : v))
    end
    M_h = FF.FringeModule{K}(P_h, U_h, D_h, phi_hM; field=field)
    N_h = FF.FringeModule{K}(P_h, U_h, D_h, phi_hN; field=field)

    h_legacy = FF._hom_dimension_with_path(M_h, N_h, :sparse_path)
    h_denseidx = FF._hom_dimension_with_path(M_h, N_h, :dense_idx_internal)
    h_auto = FF.hom_dimension(M_h, N_h)
    @test h_auto == h_legacy == h_denseidx
    chosen_h = FF._select_hom_internal_path!(M_h, N_h)
    @test chosen_h == :sparse_path

    t_h_legacy = _median_elapsed(reps=reps) do
        FF._hom_dimension_with_path(M_h, N_h, :sparse_path)
    end
    t_h_dense = _median_elapsed(reps=reps) do
        FF._hom_dimension_with_path(M_h, N_h, :dense_idx_internal)
    end
    t_h_auto = _median_elapsed(reps=reps) do
        FF.hom_dimension(M_h, N_h)
    end

    best_ns = min(_ns_per_item(t_h_legacy, 1), _ns_per_item(t_h_dense, 1))
    auto_ns = _ns_per_item(t_h_auto, 1)
    if strict_ci
        @test auto_ns <= 2.15 * best_ns + 320_000.0
    else
        @test auto_ns <= 1.85 * best_ns + 320_000.0
    end

    # 4) tiny dense-ish fixtures may still use the dense-index path.
    P_d = chain_poset(12)
    rng_d = MersenneTwister(0xD3E511)
    nu_d, nd_d = 2, 2
    U_d = [FF.principal_upset(P_d, rand(rng_d, 1:P_d.n)) for _ in 1:nu_d]
    D_d = [FF.principal_downset(P_d, rand(rng_d, 1:P_d.n)) for _ in 1:nd_d]
    phi_dM = Matrix{K}(undef, nd_d, nu_d)
    phi_dN = Matrix{K}(undef, nd_d, nu_d)
    @inbounds for j in 1:nd_d, i in 1:nu_d
        if FF.intersects(U_d[i], D_d[j]) && rand(rng_d) < 0.75
            v = rand(rng_d, 1:3)
            phi_dM[j, i] = c(v)
            phi_dN[j, i] = c(v + 1)
        else
            phi_dM[j, i] = zero(K)
            phi_dN[j, i] = zero(K)
        end
    end
    M_d = FF.FringeModule{K}(P_d, U_d, D_d, phi_dM; field=field)
    N_d = FF.FringeModule{K}(P_d, U_d, D_d, phi_dN; field=field)

    h_dense_idx = FF._hom_dimension_with_path(M_d, N_d, :dense_idx_internal)
    h_sparse_path = FF._hom_dimension_with_path(M_d, N_d, :sparse_path)
    h_dense_auto = FF.hom_dimension(M_d, N_d)
    @test h_dense_auto == h_dense_idx == h_sparse_path
    @test FF._select_hom_internal_path!(M_d, N_d) == :dense_idx_internal

    # Warm one-shot selector before timing (we budget steady-state runtime).
    _ = FF.hom_dimension(M_d, N_d)
    t_dense_idx = _median_elapsed(reps=reps) do
        FF._hom_dimension_with_path(M_d, N_d, :dense_idx_internal)
    end
    t_sparse_path = _median_elapsed(reps=reps) do
        FF._hom_dimension_with_path(M_d, N_d, :sparse_path)
    end
    t_dense_auto = _median_elapsed(reps=reps) do
        FF.hom_dimension(M_d, N_d)
    end

    best_dense_ns = min(_ns_per_item(t_dense_idx, 1),
                        _ns_per_item(t_sparse_path, 1))
    auto_dense_ns = _ns_per_item(t_dense_auto, 1)
    if strict_ci
        @test auto_dense_ns <= 1.25 * best_dense_ns + 120_000.0
    else
        @test auto_dense_ns <= 1.40 * best_dense_ns + 200_000.0
    end
end
end

@testset "HomExt pi0_count (Prop 3.10 sanity)" begin
    # Disjoint union of two chains: intersection has two connected components.
    P = disjoint_two_chains_poset()

    U = FF.upset_from_generators(P, [2, 4])      # {2,4}
    D = FF.downset_from_generators(P, [2, 4])    # {1,2,3,4}

    @test HE.pi0_count(P, U, D) == 2

    # Connected case: chain
    Q = chain_poset(3)
    U2 = FF.principal_upset(Q, 2)
    D2 = FF.principal_downset(Q, 2)
    @test HE.pi0_count(Q, U2, D2) == 1
end

if field isa CM.QQField
@testset "Encoding._uptight_regions signature grouping oracle" begin
    P = chain_poset(48)
    rng = MersenneTwister(0xC0FFEE)

    normalize_partition(regs) = sort([sort(copy(r)) for r in regs], by = v -> (length(v), first(v)))

    function naive_uptight_regions(Q, Y)
        m = length(Y)
        sigs = Dict{Tuple{Vararg{Bool}}, Vector{Int}}()
        for q in 1:FF.nvertices(Q)
            key = ntuple(i -> Y[i].mask[q], m)
            vec = get!(sigs, key) do
                Int[]
            end
            push!(vec, q)
        end
        return collect(values(sigs))
    end

    # Hit all three implementation branches: m<=64, m<=128, and m>128.
    for m in (32, 96, 160)
        Y = Vector{FF.Upset}(undef, m)
        for i in 1:m
            g = rand(rng, 1:FF.nvertices(P))
            Y[i] = FF.principal_upset(P, g)
        end

        fast_regs = EN._uptight_regions(P, Y)
        ref_regs = naive_uptight_regions(P, Y)
        @test normalize_partition(fast_regs) == normalize_partition(ref_regs)
    end
end
end


@testset "By-hand theory checks (Miller Prop 3.10 and chain intervals)" begin

    @testset "Prop 3.10: Hom for indicator modules uses pi0 on Hasse graph" begin
        # Disjoint union of two chains: 1<2 and 3<4 (no relations between components).
        P = disjoint_two_chains_poset()

        allU = FF.upset_closure(P, trues(P.n))
        allD = FF.downset_closure(P, trues(P.n))

        # U = upset generated by {2,4} is just {2,4}.
        # D = downset generated by {2,4} is the whole poset.
        U = FF.upset_from_generators(P, [2, 4])
        D = FF.downset_from_generators(P, [2, 4])

        # k[U] as a fringe image: k[U] -> k[Q] (death = all).
        kU = one_by_one_fringe(P, U, allD; field=field)

        # k[D] as a fringe image: k[Q] -> k[D] (birth = all).
        kD = one_by_one_fringe(P, allU, D; field=field)

        # Prop 3.10(1): dim Hom_Q(k[U], k[D]) = number of connected components of U cap D.
        # Here U cap D = U = {2,4}, which has 2 components.
        @test HE.pi0_count(P, U, D) == 2
        @test FF.hom_dimension(kU, kD) == 2
        @test FF.hom_dimension(kU, kD) == HE.pi0_count(P, U, D)

        # Prop 3.10(2): For upsets U',U, dim Hom(k[U'], k[U]) = number of components of U'
        # that lie inside U.
        Uprime = FF.upset_from_generators(P, [2, 4])  # {2,4} (2 components)
        Usmall = FF.upset_from_generators(P, [2])     # {2} (1 component)
        kUprime = one_by_one_fringe(P, Uprime, allD; field=field)
        kUsmall = one_by_one_fringe(P, Usmall, allD; field=field)

        @test FF.hom_dimension(kUprime, kUsmall) == 1
        @test FF.hom_dimension(kUsmall, kUprime) == 1

        # End(k[Uprime]) has one scalar per connected component of Uprime.
        @test FF.hom_dimension(kUprime, kUprime) == 2

        # Prop 3.10(3): For downsets D,D', dim Hom(k[D], k[D']) = number of components of D'
        # that lie inside D (note the direction).
        Dall = allD
        Dsmall = FF.downset_from_generators(P, [2])   # {1,2}
        kDall = one_by_one_fringe(P, allU, Dall; field=field)
        kDsmall = one_by_one_fringe(P, allU, Dsmall; field=field)

        @test FF.hom_dimension(kDall, kDsmall) == 1
        @test FF.hom_dimension(kDsmall, kDall) == 1
        @test FF.hom_dimension(kDall, kDall) == 2
    end

    @testset "Chain intervals: Hom and Ext^1 formula for type A_n" begin
        # On a finite chain, an interval module I[a,b] is indecomposable.
        # For the equioriented A_n quiver (a chain poset), the category is hereditary,
        # so Ext^t = 0 for t >= 2. Ext^1 is 1 exactly for the "one-step overlap" pattern:
        # Ext^1(I[a,b], I[c,d]) = 1 iff a < c <= b+1 <= d (and b < n so b+1 exists).
        #
        # We realize I[a,b] as the image of k[U_a] -> k[D_b] with phi = 1, where:
        #   U_a = principal upset at a  (degrees >= a)
        #   D_b = principal downset at b (degrees <= b)
        if field isa CM.RealField
            # This theorem-level integer-dimension characterization is exact-field specific.
            @test true
        else
        n = 4
        P = chain_poset(n)

        function interval_module(a::Int, b::Int)
            @assert 1 <= a <= b <= n
            U = FF.principal_upset(P, a)
            D = FF.principal_downset(P, b)
            return one_by_one_fringe(P, U, D; field=field)
        end

        intervals = [(a, b) for a in 1:n for b in a:n]
        mods = Dict{Tuple{Int,Int}, FF.FringeModule{K}}()
        for (a, b) in intervals
            mods[(a, b)] = interval_module(a, b)
        end

        for (a, b) in intervals, (c, d) in intervals
            M = mods[(a, b)]
            N = mods[(c, d)]

            verify_exact = !(field isa CM.RealField)
            ext = DF.ext_dimensions_via_indicator_resolutions(M, N; maxlen=5, verify=verify_exact)

            # Hom(I[a,b], I[c,d]) is 1 iff c <= a <= d <= b, else 0.
            expected_hom = (c <= a && a <= d && d <= b) ? 1 : 0

            # Ext^1(I[a,b], I[c,d]) is 1 iff a < c <= b+1 <= d, else 0.
            expected_ext1 = (b < n && a < c && c <= b + 1 && b + 1 <= d) ? 1 : 0

            @test get(ext, 0, 0) == expected_hom
            @test get(ext, 1, 0) == expected_ext1
            @test get(ext, 2, 0) == 0
        end

        # Sanity-check the total complex sign convention: d_{t+1} * d_t == 0.
        # Pick a pair where Ext^1 is expected nonzero to ensure differentials are not trivial.
        M = interval_module(1, 2)   # I[1,2]
        N = interval_module(2, 3)   # I[2,3]

        F, dF = IR.upset_resolution(M; maxlen=5)
        E, dE = IR.downset_resolution(N; maxlen=5)

        dimsCt, dts = HE.build_hom_tot_complex(F, dF, E, dE)
        for t in 1:(length(dts) - 1)
            @test nnz(dts[t+1] * dts[t]) == 0
        end
        end
    end

    @testset "Zero module edge case" begin
        # A fringe module with empty intersection but zero phi should be the zero module.
        P = chain_poset(3)
        U = FF.principal_upset(P, 3)      # {3}
        D = FF.principal_downset(P, 1)    # {1}
        phi0 = spzeros(K, 1, 1)          # must be zero since U cap D is empty
        M0 = FF.FringeModule{K}(P, [U], [D], phi0; field=field)

        for q in 1:P.n
            @test FF.fiber_dimension(M0, q) == 0
        end

        # Hom from/to zero should be 0.
        U2 = FF.principal_upset(P, 2)
        D2 = FF.principal_downset(P, 2)
        M = one_by_one_fringe(P, U2, D2; field=field)

        @test FF.hom_dimension(M0, M) == 0
        @test FF.hom_dimension(M, M0) == 0

        ext = DF.ext_dimensions_via_indicator_resolutions(M0, M; maxlen=5)
        @test get(ext, 0, 0) == 0
        @test get(ext, 1, 0) == 0
    end
end

@testset "hom_dimension sparse-vs-dense assembly oracle parity" begin
    P = chain_poset(18)
    rng = MersenneTwister(0x5EED123 + Int(mod(hash(string(K)), 10_000)))

    function random_upset()
        g = rand(rng, 1:P.n)
        return FF.principal_upset(P, g)
    end
    function random_downset()
        g = rand(rng, 1:P.n)
        return FF.principal_downset(P, g)
    end

    function random_module(nu::Int, nd::Int; density::Float64=0.18)
        U = [random_upset() for _ in 1:nu]
        D = [random_downset() for _ in 1:nd]
        phi = spzeros(K, nd, nu)
        @inbounds for j in 1:nd, i in 1:nu
            FF.intersects(U[i], D[j]) || continue
            rand(rng) < density || continue
            v = rand(rng, -2:2)
            v == 0 && continue
            phi[j, i] = c(v)
        end
        return FF.FringeModule{K}(P, U, D, phi; field=field)
    end

    function hom_dimension_reference_old(M::FF.FringeModule{K}, N::FF.FringeModule{K}) where {K}
        M.P === N.P || error("Posets must match")
        P = M.P
        adj = FF._cover_undirected_adjacency(P)

        nUM = length(M.U); nDM = length(M.D)
        nUN = length(N.U); nDN = length(N.D)

        Ucomp_id_M    = Vector{Vector{Int}}(undef, nUM)
        Ucomp_masks_M = Vector{Vector{BitVector}}(undef, nUM)
        Ucomp_n_M     = Vector{Int}(undef, nUM)
        for i in 1:nUM
            comp_id, ncomp, comp_masks, _ = FF._component_data(adj, M.U[i].mask)
            Ucomp_id_M[i] = comp_id
            Ucomp_masks_M[i] = comp_masks
            Ucomp_n_M[i] = ncomp
        end

        Dcomp_id_N    = Vector{Vector{Int}}(undef, nDN)
        Dcomp_masks_N = Vector{Vector{BitVector}}(undef, nDN)
        Dcomp_n_N     = Vector{Int}(undef, nDN)
        for t in 1:nDN
            comp_id, ncomp, comp_masks, _ = FF._component_data(adj, N.D[t].mask)
            Dcomp_id_N[t] = comp_id
            Dcomp_masks_N[t] = comp_masks
            Dcomp_n_N[t] = ncomp
        end

        w_index = Dict{Tuple{Int,Int},Int}()
        w_rows_u = Vector{Vector{Vector{Int}}}()
        w_rows_d = Vector{Vector{Vector{Int}}}()
        W_dim = 0

        for iM in 1:nUM
            for tN in 1:nDN
                mask_int = M.U[iM].mask .& N.D[tN].mask
                if any(mask_int)
                    _, ncomp_int, _, reps_int = FF._component_data(adj, mask_int)
                    base = W_dim
                    W_dim += ncomp_int

                    rows_by_u = [Int[] for _ in 1:Ucomp_n_M[iM]]
                    rows_by_d = [Int[] for _ in 1:Dcomp_n_N[tN]]
                    for c in 1:ncomp_int
                        row = base + c
                        v = reps_int[c]
                        cu = Ucomp_id_M[iM][v]
                        cd = Dcomp_id_N[tN][v]
                        push!(rows_by_u[cu], row)
                        push!(rows_by_d[cd], row)
                    end

                    push!(w_rows_u, rows_by_u)
                    push!(w_rows_d, rows_by_d)
                    w_index[(iM, tN)] = length(w_rows_u)
                end
            end
        end

        V1 = Tuple{Int,Int,Int}[]
        for iM in 1:nUM
            for jN in 1:nUN
                for cU in 1:Ucomp_n_M[iM]
                    if FF.is_subset(Ucomp_masks_M[iM][cU], N.U[jN].mask)
                        push!(V1, (iM, jN, cU))
                    end
                end
            end
        end
        V1_dim = length(V1)

        V2 = Tuple{Int,Int,Int}[]
        for sM in 1:nDM
            for tN in 1:nDN
                for cD in 1:Dcomp_n_N[tN]
                    if FF.is_subset(Dcomp_masks_N[tN][cD], M.D[sM].mask)
                        push!(V2, (sM, tN, cD))
                    end
                end
            end
        end
        V2_dim = length(V2)

        T = zeros(K, W_dim, V1_dim)
        for (col, (iM, jN, cU)) in enumerate(V1)
            for tN in 1:nDN
                val = N.phi[tN, jN]
                if val != zero(K)
                    pid = get(w_index, (iM, tN), 0)
                    if pid != 0
                        rows = w_rows_u[pid][cU]
                        for r in rows
                            T[r, col] += val
                        end
                    end
                end
            end
        end

        S = zeros(K, W_dim, V2_dim)
        for (col, (sM, tN, cD)) in enumerate(V2)
            for iM in 1:nUM
                val = M.phi[sM, iM]
                if val != zero(K)
                    pid = get(w_index, (iM, tN), 0)
                    if pid != 0
                        rows = w_rows_d[pid][cD]
                        for r in rows
                            S[r, col] += val
                        end
                    end
                end
            end
        end

        rT = FF.FieldLinAlg.rank(field, T)
        rS = FF.FieldLinAlg.rank(field, S)
        rBig = FF.FieldLinAlg.rank(field, hcat(T, -S))
        dimKer_big = (V1_dim + V2_dim) - rBig
        dimKer_T = V1_dim - rT
        dimKer_S = V2_dim - rS
        return dimKer_big - (dimKer_T + dimKer_S)
    end

    M = random_module(14, 13; density=0.22)
    N = random_module(13, 14; density=0.20)
    M_dense = FF.FringeModule{K}(P, M.U, M.D, Matrix(M.phi); field=field)
    N_dense = FF.FringeModule{K}(P, N.U, N.D, Matrix(N.phi); field=field)

    @test M.hom_cache[].upset === nothing
    @test N.hom_cache[].downset === nothing

    ref_sparse = hom_dimension_reference_old(M, N)
    ref_dense = hom_dimension_reference_old(M_dense, N_dense)
    @test FF._resolve_hom_dimension_path(M, N) == :sparse_path
    @test FF._resolve_hom_dimension_path(M_dense, N_dense) == :sparse_path
    @test ref_sparse == ref_dense
    @test FF.hom_dimension(M, N) == ref_sparse
    @test FF._hom_dimension_with_path(M, N, :sparse_path) == ref_sparse
    @test FF._hom_dimension_with_path(M, N, :dense_idx_internal) == ref_sparse
    @test FF.hom_dimension(M_dense, N_dense) == ref_dense
    @test FF._hom_dimension_with_path(M_dense, N_dense, :sparse_path) == ref_dense
    @test FF._hom_dimension_with_path(M_dense, N_dense, :dense_idx_internal) == ref_dense
    @test FF.hom_dimension(M, N) == FF.hom_dimension(M_dense, N_dense)
    @test_throws MethodError FF.hom_dimension(M, N; mode=:unsupported)
    @test_throws MethodError FF.hom_dimension(M, N; mode=:sparse_path)
    @test_throws MethodError FF.hom_dimension(M, N; mode=:dense_idx_internal)

    @test M.hom_cache[].adj !== nothing
    @test M.hom_cache[].upset !== nothing
    @test N.hom_cache[].downset !== nothing
    cache_u = M.hom_cache[].upset
    cache_d = N.hom_cache[].downset
    _ = FF.hom_dimension(M, N)
    @test M.hom_cache[].upset === cache_u
    @test N.hom_cache[].downset === cache_d
    entry_sparse = FF._lookup_pair_cache(M.hom_cache[], N)
    @test entry_sparse !== nothing
    chosen_sparse = entry_sparse.route_choice
    @test chosen_sparse in (:sparse_path, :dense_idx_internal)
    fp_sparse = FF._hom_route_fingerprint(M, N, :internal_choice)
    @test FF._route_fingerprint_choice_get(M.hom_cache[], fp_sparse) == chosen_sparse
    @test haskey(M.P.cache.hom_route_choice, fp_sparse)
    @test M.P.cache.hom_route_choice[fp_sparse] == chosen_sparse
    _ = FF.hom_dimension(M, N)
    @test FF._lookup_pair_cache(M.hom_cache[], N).route_choice == chosen_sparse

    # Dense-storage sparse-effective fixtures should route through the sparse path.
    FF._clear_hom_route_choice!(M_dense)
    _ = FF.hom_dimension(M_dense, N_dense)
    @test M_dense.hom_cache[].route_timing_fallbacks == 0
    @test FF._resolve_hom_dimension_path(M_dense, N_dense) == :sparse_path

    entry_dense = FF._lookup_pair_cache(M_dense.hom_cache[], N_dense)
    @test entry_dense !== nothing
    chosen = entry_dense.route_choice
    @test chosen == :sparse_path
    fp_dense = FF._hom_route_fingerprint(M_dense, N_dense, :internal_choice)
    @test FF._route_fingerprint_choice_get(M_dense.hom_cache[], fp_dense) == chosen
    @test haskey(M_dense.P.cache.hom_route_choice, fp_dense)
    @test M_dense.P.cache.hom_route_choice[fp_dense] == chosen
    _ = FF.hom_dimension(M_dense, N_dense)
    @test FF._lookup_pair_cache(M_dense.hom_cache[], N_dense).route_choice == chosen

    # Shape/density route memo should be reusable across new modules on the same poset.
    M2 = random_module(14, 13; density=0.22)
    N2 = random_module(13, 14; density=0.20)
    h2 = FF.hom_dimension(M2, N2)
    @test h2 == hom_dimension_reference_old(M2, N2)
    fp_sparse2 = FF._hom_route_fingerprint(M2, N2, :internal_choice)
    @test haskey(M2.P.cache.hom_route_choice, fp_sparse2)
    @test FF._route_fingerprint_choice_get(M2.hom_cache[], fp_sparse2) !== nothing

    M2_dense = FF.FringeModule{K}(P, M2.U, M2.D, Matrix(M2.phi); field=field)
    N2_dense = FF.FringeModule{K}(P, N2.U, N2.D, Matrix(N2.phi); field=field)
    h2d = FF.hom_dimension(M2_dense, N2_dense)
    @test h2d == hom_dimension_reference_old(M2_dense, N2_dense)
    fp_dense2 = FF._hom_route_fingerprint(M2_dense, N2_dense, :internal_choice)
    @test haskey(M2_dense.P.cache.hom_route_choice, fp_dense2)
    @test FF._route_fingerprint_choice_get(M2_dense.hom_cache[], fp_dense2) !== nothing

    # Sparse moderate-work fixtures should prefer sparse path directly.
    M_mid = random_module(20, 19; density=0.14)
    N_mid = random_module(20, 19; density=0.13)
    @test FF._resolve_hom_dimension_path(M_mid, N_mid) == :sparse_path

    # Sparse low-work fixtures should also route without timing fallback.
    M_small = random_module(8, 8; density=0.18)
    N_small = random_module(8, 8; density=0.16)
    @test FF._resolve_hom_dimension_path(M_small, N_small) == :sparse_path
    @test FF._heuristic_hom_internal_choice(M_small, N_small) == :sparse_path
    FF._clear_hom_route_choice!(M_small)
    ref_small = hom_dimension_reference_old(M_small, N_small)
    @test FF.hom_dimension(M_small, N_small) == ref_small
    @test M_small.hom_cache[].route_timing_fallbacks == 0
    @test FF._lookup_pair_cache(M_small.hom_cache[], N_small).route_choice == :sparse_path

    # Force one-shot timed fallback directly and verify parity.
    FF._clear_hom_route_choice!(M_small)
    hc_small = FF._ensure_hom_cache!(M_small)
    entry_small = FF._ensure_pair_cache!(hc_small, N_small)
    fkey_small = FF._hom_route_fingerprint(M_small, N_small, :internal_choice)
    path_small = FF._select_hom_internal_path_timed!(M_small, N_small, hc_small, entry_small, fkey_small)
    @test path_small in (:sparse_path, :dense_idx_internal)
    @test FF._hom_dimension_with_path(M_small, N_small, path_small) == ref_small
    @test hc_small.route_timing_fallbacks >= 1
end

@testset "FringeModule typed-poset and sparse-path contracts" begin
    P = chain_poset(16)
    rng = MersenneTwister(0x7A11 + Int(mod(hash(string(K)), 10_000)))

    function random_module(nu::Int, nd::Int; density::Float64=0.16)
        U = [FF.principal_upset(P, rand(rng, 1:P.n)) for _ in 1:nu]
        D = [FF.principal_downset(P, rand(rng, 1:P.n)) for _ in 1:nd]
        phi = spzeros(K, nd, nu)
        @inbounds for j in 1:nd, i in 1:nu
            FF.intersects(U[i], D[j]) || continue
            rand(rng) < density || continue
            v = rand(rng, -2:2)
            v == 0 && continue
            phi[j, i] = c(v)
        end
        return FF.FringeModule{K}(P, U, D, phi; field=field)
    end

    U2 = FF.principal_upset(P, 2)
    D2 = FF.principal_downset(P, 2)
    @test U2 isa FF.Upset{typeof(P)}
    @test D2 isa FF.Downset{typeof(P)}

    M = random_module(12, 11; density=0.14)
    N = random_module(11, 12; density=0.13)
    @test M isa FF.FringeModule{K,typeof(P)}
    @test eltype(M.U) == FF.Upset{typeof(P)}
    @test eltype(M.D) == FF.Downset{typeof(P)}

    P2 = chain_poset(16)
    @test_throws ErrorException FF.FringeModule{K}(P2, M.U, M.D, M.phi; field=field)

    FF._clear_hom_route_choice!(M)
    h_sparse = FF._hom_dimension_with_path(M, N, :sparse_path)
    sparse_entry = FF._lookup_pair_cache(M.hom_cache[], N)
    @test sparse_entry !== nothing
    sparse_plan = sparse_entry.sparse_plan
    @test sparse_plan.W_dim == size(sparse_plan.T, 1)
    @test sparse_plan.W_dim == size(sparse_plan.S, 1)
    @test length(sparse_plan.t_tN) == nnz(sparse_plan.T)
    @test length(sparse_plan.s_sM) == nnz(sparse_plan.S)
    @test sparse_plan.t_nzptr !== nothing
    @test sparse_plan.s_nzptr !== nothing
    @test sparse_plan.t_nzptr_max >= 0
    @test sparse_plan.s_nzptr_max >= 0
    @test length(sparse_plan.t_rows.ptr) == size(sparse_plan.T, 1) + 1
    @test length(sparse_plan.s_rows.ptr) == size(sparse_plan.S, 1) + 1
    @test length(sparse_plan.t_rows.row_nnz) == size(sparse_plan.T, 1)
    @test length(sparse_plan.s_rows.row_nnz) == size(sparse_plan.S, 1)
    @test sparse_plan.t_rows.max_row_nnz >= 0
    @test sparse_plan.s_rows.max_row_nnz >= 0
    @test size(sparse_plan.hcat_buf, 2) == size(sparse_plan.T, 2) + size(sparse_plan.S, 2)
    @test size(sparse_plan.hcat_buf_rev, 2) == size(sparse_plan.T, 2) + size(sparse_plan.S, 2)
    if sparse_plan.t_nzptr !== nothing
        for i in 1:min(8, length(sparse_plan.t_nzptr))
            p = sparse_plan.t_nzptr[i]
            @test p >= 1
            @test p <= length(N.phi.nzval)
            @test N.phi.nzval[p] == N.phi[sparse_plan.t_tN[i], sparse_plan.t_jN[i]]
        end
    end
    if sparse_plan.s_nzptr !== nothing
        for i in 1:min(8, length(sparse_plan.s_nzptr))
            p = sparse_plan.s_nzptr[i]
            @test p >= 1
            @test p <= length(M.phi.nzval)
            @test M.phi.nzval[p] == M.phi[sparse_plan.s_sM[i], sparse_plan.s_iM[i]]
        end
    end
    r_union_ws = FF._rank_hcat_signed_sparse_workspace!(field, sparse_plan.hcat_buf,
                                                         sparse_plan.T, sparse_plan.S,
                                                         sparse_plan.nnzT)
    r_union_ref = FF.FieldLinAlg.rank(field, hcat(sparse_plan.T, sparse_plan.S))
    @test r_union_ws == r_union_ref
    r_union_pref, rT_pref = FF._rank_hcat_signed_sparse_workspace_with_prefix_rank!(
        field, sparse_plan.hcat_buf, sparse_plan.T, sparse_plan.S, sparse_plan.nnzT, size(sparse_plan.T, 2)
    )
    @test r_union_pref == r_union_ref
    if rT_pref >= 0
        @test rT_pref == FF.FieldLinAlg.rank(field, sparse_plan.T)
    end
    r_union_onepass, rT_onepass, rS_onepass = FF._rank_hcat_signed_sparse_rowplans_with_side_rank!(
        sparse_plan.union_red, sparse_plan.s_red, sparse_plan.union_row, sparse_plan.side_row,
        sparse_plan.T, sparse_plan.t_rows, sparse_plan.S, sparse_plan.s_rows, size(sparse_plan.T, 2)
    )
    @test r_union_onepass == r_union_ref
    @test rT_onepass == FF.FieldLinAlg.rank(field, sparse_plan.T)
    @test rS_onepass == FF.FieldLinAlg.rank(field, sparse_plan.S)
    r_union_ws_rev = FF._rank_hcat_signed_sparse_workspace!(field, sparse_plan.hcat_buf_rev,
                                                             sparse_plan.S, sparse_plan.T,
                                                             sparse_plan.nnzS)
    @test r_union_ws_rev == r_union_ref
    r_union_pref_rev, rS_pref = FF._rank_hcat_signed_sparse_workspace_with_prefix_rank!(
        field, sparse_plan.hcat_buf_rev, sparse_plan.S, sparse_plan.T, sparse_plan.nnzS, size(sparse_plan.S, 2)
    )
    @test r_union_pref_rev == r_union_ref
    if rS_pref >= 0
        @test rS_pref == FF.FieldLinAlg.rank(field, sparse_plan.S)
    end
    r_union_onepass_rev, rS_onepass_rev, rT_onepass_rev = FF._rank_hcat_signed_sparse_rowplans_with_side_rank!(
        sparse_plan.union_red, sparse_plan.t_red, sparse_plan.union_row, sparse_plan.side_row,
        sparse_plan.S, sparse_plan.s_rows, sparse_plan.T, sparse_plan.t_rows, size(sparse_plan.S, 2)
    )
    @test r_union_onepass_rev == r_union_ref
    @test rS_onepass_rev == FF.FieldLinAlg.rank(field, sparse_plan.S)
    @test rT_onepass_rev == FF.FieldLinAlg.rank(field, sparse_plan.T)
    if !isempty(sparse_plan.w_data)
        w0 = sparse_plan.w_data[1]
        @test length(w0.u_ptr) >= 2
        @test length(w0.d_ptr) >= 2
    end
    h_sparse2 = FF._hom_dimension_with_path(M, N, :sparse_path)
    @test h_sparse2 == h_sparse
    @test FF._lookup_pair_cache(M.hom_cache[], N).sparse_plan === sparse_plan
    if size(sparse_plan.S, 2) <= size(sparse_plan.T, 2)
        @test sparse_plan.cached_S_valid
        @test !sparse_plan.cached_T_valid
        @test sparse_plan.cached_rS == FF.FieldLinAlg.rank(field, sparse_plan.S)
    else
        @test sparse_plan.cached_T_valid
        @test !sparse_plan.cached_S_valid
        @test sparse_plan.cached_rT == FF.FieldLinAlg.rank(field, sparse_plan.T)
    end

    h_dense_idx = FF._hom_dimension_with_path(M, N, :dense_idx_internal)
    @test h_sparse == h_dense_idx

    M2 = random_module(12, 11; density=0.15)
    N2 = random_module(11, 12; density=0.14)
    alloc_first = @allocated FF._hom_dimension_with_path(M2, N2, :sparse_path)
    alloc_second = @allocated FF._hom_dimension_with_path(M2, N2, :sparse_path)
    @test alloc_second < alloc_first

    M3 = random_module(12, 11; density=0.16)
    N3 = random_module(11, 12; density=0.15)
    h3 = FF._hom_dimension_with_path(M3, N3, :sparse_path)
    plan3 = FF._lookup_pair_cache(M3.hom_cache[], N3).sparse_plan
    @test plan3 !== nothing
    if size(plan3.S, 2) <= size(plan3.T, 2)
        @test plan3.cached_S_valid
        p = findfirst(v -> !iszero(v), M3.phi.nzval)
        @test p !== nothing
        old = M3.phi.nzval[p]
        M3.phi.nzval[p] = old + c(1)
        h3_mut = FF._hom_dimension_with_path(M3, N3, :sparse_path)
        @test h3_mut == FF._hom_dimension_with_path(M3, N3, :dense_idx_internal)
        @test plan3.cached_S_valid
        @test plan3.cached_rS == FF.FieldLinAlg.rank(field, plan3.S)
    else
        @test plan3.cached_T_valid
        p = findfirst(v -> !iszero(v), N3.phi.nzval)
        @test p !== nothing
        old = N3.phi.nzval[p]
        N3.phi.nzval[p] = old + c(1)
        h3_mut = FF._hom_dimension_with_path(M3, N3, :sparse_path)
        @test h3_mut == FF._hom_dimension_with_path(M3, N3, :dense_idx_internal)
        @test plan3.cached_T_valid
        @test plan3.cached_rT == FF.FieldLinAlg.rank(field, plan3.T)
    end
end

@testset "fiber_dimension cached query index parity" begin
    P = chain_poset(24)
    rng = MersenneTwister(0xA11CE + Int(mod(hash(string(K)), 10_000)))

    function random_module(nu::Int, nd::Int; density::Float64=0.2)
        U = [FF.principal_upset(P, rand(rng, 1:P.n)) for _ in 1:nu]
        D = [FF.principal_downset(P, rand(rng, 1:P.n)) for _ in 1:nd]
        phi = spzeros(K, nd, nu)
        @inbounds for j in 1:nd, i in 1:nu
            FF.intersects(U[i], D[j]) || continue
            rand(rng) < density || continue
            v = rand(rng, -3:3)
            v == 0 && continue
            phi[j, i] = c(v)
        end
        return FF.FringeModule{K}(P, U, D, phi; field=field)
    end

    function fiber_dimension_scan_reference(M::FF.FringeModule{K}, q::Int) where {K}
        cols = findall(U -> U.mask[q], M.U)
        rows = findall(D -> D.mask[q], M.D)
        isempty(cols) || isempty(rows) && return 0
        return FF.FieldLinAlg.rank_restricted(M.field, M.phi, rows, cols)
    end

    M = random_module(16, 15; density=0.24)
    @test M.fiber_index[] !== nothing
    @test M.fiber_dims[] === nothing
    expected = [fiber_dimension_scan_reference(M, q) for q in 1:P.n]

    got = Int[]
    for q in 1:P.n
        push!(got, FF.fiber_dimension(M, q))
    end
    @test got == expected
    @test M.fiber_index[] !== nothing
    @test M.fiber_dims[] !== nothing
    @test all(v != typemin(Int) for v in M.fiber_dims[])

    for _ in 1:50
        q = rand(rng, 1:P.n)
        @test FF.fiber_dimension(M, q) == expected[q]
    end

    P_big = chain_poset(128)
    U_big = [FF.principal_upset(P_big, rand(rng, 1:P_big.n)) for _ in 1:300]
    D_big = [FF.principal_downset(P_big, rand(rng, 1:P_big.n)) for _ in 1:300]
    phi_big = spzeros(K, length(D_big), length(U_big))
    @inbounds for j in 1:length(D_big), i in 1:length(U_big)
        FF.intersects(U_big[i], D_big[j]) || continue
        rand(rng) < 0.04 || continue
        v = rand(rng, -1:1)
        v == 0 && continue
        phi_big[j, i] = c(v)
    end
    M_big = FF.FringeModule{K}(P_big, U_big, D_big, phi_big; field=field)
    @test M_big.fiber_index[] === nothing
    q_big = rand(rng, 1:P_big.n)
    expected_big = fiber_dimension_scan_reference(M_big, q_big)
    rows_big, cols_big = FF._build_fiber_query_slice(M_big.U, M_big.D, q_big)
    roww_big, colw_big, nr_big, nc_big = FF._build_fiber_query_slice_words(M_big.U, M_big.D, q_big)
    @test nr_big == length(rows_big)
    @test nc_big == length(cols_big)
    @test FF.FieldLinAlg.rank_restricted_words(M_big.field, M_big.phi, roww_big, colw_big, nr_big, nc_big;
                                               nrows=size(M_big.phi, 1), ncols=size(M_big.phi, 2)) ==
          FF.FieldLinAlg.rank_restricted(M_big.field, M_big.phi, rows_big, cols_big)
    @test FF.fiber_dimension(M_big, q_big) == expected_big
    @test M_big.fiber_index[] === nothing

    seen = Set([q_big])
    target_queries = FF._fiber_lazy_full_index_after(M_big)
    @test target_queries > 1
    while length(seen) < target_queries
        q = rand(rng, 1:P_big.n)
        q in seen && continue
        push!(seen, q)
        @test FF.fiber_dimension(M_big, q) == fiber_dimension_scan_reference(M_big, q)
    end
    @test M_big.fiber_index[] !== nothing
end

@testset "one_by_one_fringe accepts Bool masks" begin
    P = chain_poset(5)

    # Upset {3,4,5} and downset {1,2,3} given as membership masks.
    U_mask = BitVector([i >= 3 for i in 1:P.n])
    D_mask = BitVector([i <= 3 for i in 1:P.n])

    H_mask  = TO.one_by_one_fringe(P, U_mask, D_mask, c(1); field=field)
    H_typed = TO.one_by_one_fringe(P, FF.Upset(P, U_mask), FF.Downset(P, D_mask), c(1); field=field)

    for q in 1:P.n
        @test FF.fiber_dimension(H_mask, q) == FF.fiber_dimension(H_typed, q)
    end

    # Also accept plain Vector{Bool} (not just BitVector).
    H_vec = TO.one_by_one_fringe(P, collect(U_mask), collect(D_mask), c(1); field=field)
    for q in 1:P.n
        @test FF.fiber_dimension(H_vec, q) == FF.fiber_dimension(H_typed, q)
    end

    # Strict contract: scalar argument is required (typed upset/downset and mask forms).
    @test_throws MethodError TO.one_by_one_fringe(P, FF.Upset(P, U_mask), FF.Downset(P, D_mask))
    @test_throws MethodError TO.one_by_one_fringe(P, FF.Upset(P, U_mask), FF.Downset(P, D_mask); scalar=c(1), field=field)
    @test_throws MethodError TO.one_by_one_fringe(P, U_mask, D_mask)
    @test_throws MethodError TO.one_by_one_fringe(P, U_mask, D_mask; scalar=c(1), field=field)
end

end # with_fields

@testset "Up/down cache default behavior" begin
    P = chain_poset(5)
    _ = FF.upset_indices(P, 1)
    @test P.cache.upsets !== nothing
    @test P.cache.downsets !== nothing

    FF._clear_cover_cache!(P)
    @test P.cache.upsets === nothing
    @test P.cache.downsets === nothing

    Pc = FF.ProductOfChainsPoset((4, 4))
    _ = FF.upset_indices(Pc, 1)
    @test Pc.cache.upsets === nothing
    @test Pc.cache.downsets === nothing
end
