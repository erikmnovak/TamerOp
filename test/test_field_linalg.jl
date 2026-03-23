using Random
using LinearAlgebra
using SparseArrays
using TOML

@testset "FieldLinAlg engines" begin
    FL = TamerOp.FieldLinAlg
    F2 = CM.F2()
    F2Elem = CM.FpElem{2}

    function f2mat(A::AbstractMatrix{<:Integer})
        return F2Elem.(A .% 2)
    end

    function f2mat(v::AbstractVector{<:Integer})
        return F2Elem.(v .% 2)
    end

    function f2rank_naive(A::Matrix{Int})
        M = copy(A .% 2)
        m, n = size(M)
        r = 0
        c = 1
        while r < m && c <= n
            piv = 0
            for i in r+1:m
                if M[i, c] == 1
                    piv = i
                    break
                end
            end
            if piv == 0
                c += 1
                continue
            end
            r += 1
            if piv != r
                M[r, :], M[piv, :] = M[piv, :], M[r, :]
            end
            for i in 1:m
                i == r && continue
                if M[i, c] == 1
                    @inbounds for j in c:n
                        M[i, j] = xor(M[i, j], M[r, j])
                    end
                end
            end
            c += 1
        end
        return r
    end

    function _selection_words(idxs::Vector{Int}, nmax::Int)
        words = zeros(UInt64, cld(nmax, 64))
        for idx in idxs
            wd = ((idx - 1) >>> 6) + 1
            words[wd] |= UInt64(1) << ((idx - 1) & 63)
        end
        return words
    end

    function _tol(field, A)
        return (field isa CM.RealField) ? (field.atol + field.rtol * opnorm(Matrix(A), 2)) : 0.0
    end

    @testset "F2 rank + rank_dim (dense)" begin
        Aint = [
            1 0 1 1 0;
            1 1 0 1 1;
            0 1 1 0 1;
            1 1 1 0 0
        ]
        A = f2mat(Aint)
        r1 = FL.rank(F2, A)
        r2 = FL.rank_dim(F2, A)
        @test r1 == r2
        @test r1 == f2rank_naive(Aint)
    end

    @testset "F2 rank + rank_dim (sparse)" begin
        Aint = [
            1 0 1 0 0;
            0 1 1 0 1;
            1 1 0 1 0;
            0 0 1 1 1
        ]
        A = sparse(f2mat(Aint))
        r1 = FL.rank(F2, A)
        r2 = FL.rank_dim(F2, A)
        @test r1 == r2
        @test r1 == f2rank_naive(Aint)
    end

    @testset "F2 nullspace + rref" begin
        Aint = [
            1 0 1 0;
            0 1 1 1;
            1 1 0 1
        ]
        A = f2mat(Aint)
        N = FL.nullspace(F2, A)
        Z = A * N
        @test all(x -> x.val == 0, Z)

        R, pivs = FL.rref(F2, A; pivots=true)
        @test length(pivs) == FL.rank(F2, A)
        @test all(p -> 1 <= p <= size(A, 2), pivs)
    end

    @testset "F2 solve_fullcolumn (dense + sparse)" begin
        n = 6
        m = 10
        B = vcat(f2mat(Matrix{Int}(I, n, n)), f2mat(rand(0:1, m - n, n)))
        X = f2mat(rand(0:1, n, 3))
        Y = B * X
        Xhat = FL.solve_fullcolumn(F2, B, Y)
        @test Xhat == X

        Bs = sparse(B)
        Y2 = Bs * X
        Xhat2 = FL.solve_fullcolumn(F2, Bs, Y2)
        @test Xhat2 == X
    end

    @testset "F2 rank_restricted + colspace sparse" begin
        Aint = [
            1 0 1 1 0;
            0 1 1 0 1;
            1 1 0 1 1;
            0 0 1 1 0
        ]
        A = sparse(f2mat(Aint))
        rows = [1, 3, 4]
        cols = [2, 3, 5]
        sub = Matrix{Int}(Aint[rows, cols])
        @test FL.rank_restricted(F2, A, rows, cols) == f2rank_naive(sub)

        C = FL.colspace(F2, A)
        @test C isa SparseMatrixCSC
    end

    @testset "F2 rank_restricted dense" begin
        Aint = [
            1 0 1 1 0;
            0 1 1 0 1;
            1 1 0 1 1;
            0 0 1 1 0
        ]
        A = f2mat(Aint)
        rows = [1, 3, 4]
        cols = [2, 3, 5]
        sub = Matrix{Int}(Aint[rows, cols])
        r = FL.rank(F2, A[rows, cols])
        @test r == f2rank_naive(sub)
    end

    @testset "F2 rref sparse pivot consistency" begin
        Aint = [
            1 0 1 0 1;
            0 1 1 1 0;
            1 1 0 1 1;
            0 0 1 1 0
        ]
        A = sparse(f2mat(Aint))
        R, pivs = FL.rref(F2, A; pivots=true)
        @test length(pivs) == FL.rank(F2, A)
        @test all(p -> 1 <= p <= size(A, 2), pivs)
        @test R == FL.rref(F2, Matrix(A); pivots=false)
    end

    @testset "F2 solve_fullcolumn RHS check" begin
        B = f2mat([1 0 0;
                   0 1 0;
                   1 1 0;
                   0 0 1])
        y_bad = f2mat([1, 0, 0, 0])
        @test_throws ErrorException FL.solve_fullcolumn(F2, B, y_bad; check_rhs=true)
        x = FL.solve_fullcolumn(F2, B, y_bad; check_rhs=false)
        @test B * x != y_bad
    end

    @testset "F2 solve_fullcolumn cache parity" begin
        B = vcat(f2mat(Matrix{Int}(I, 3, 3)), f2mat([1 1 0; 0 1 1]))
        X = f2mat([1 0; 0 1; 1 1])
        Y = B * X

        FL._clear_f2_fullcolumn_cache!()
        x1 = FL.solve_fullcolumn(F2, B, Y; cache=true)
        @test x1 == X
        @test haskey(FL._F2_FULLCOLUMN_FACTOR_CACHE, B)

        x2 = FL.solve_fullcolumn(F2, B, Y; cache=true)
        @test x2 == X

        # Factor reuse via explicit factor argument
        fact = FL._F2_FULLCOLUMN_FACTOR_CACHE[B]
        x3 = FL.solve_fullcolumn(F2, B, Y; cache=false, factor=fact)
        @test x3 == X
    end

    @testset "F2 edge cases" begin
        A0 = f2mat(zeros(Int, 0, 5))
        @test FL.rank(F2, A0) == 0
        N0 = FL.nullspace(F2, A0)
        @test size(N0) == (5, 5)

        A1 = f2mat(zeros(Int, 4, 0))
        @test FL.rank(F2, A1) == 0
        N1 = FL.nullspace(F2, A1)
        @test size(N1) == (0, 0)

        Z = f2mat(zeros(Int, 3, 4))
        N = FL.nullspace(F2, Z)
        @test size(N) == (4, 4)
        @test Z * N == f2mat(zeros(Int, 3, 4))
    end

    @testset "F2 randomized properties" begin
        rng = MersenneTwister(20240202)

        for _ in 1:20
            m = rand(rng, 2:8)
            n = rand(rng, 2:8)
            A = f2mat(rand(rng, 0:1, m, n))

            r = FL.rank(F2, A)
            N = FL.nullspace(F2, A)
            @test size(N, 1) == n
            @test size(N, 2) == n - r
            @test A * N == f2mat(zeros(Int, m, size(N, 2)))

            C = FL.colspace(F2, A)
            @test FL.rank(F2, C) == r
        end
    end

    @testset "F2 sparse nullspace agrees with dense" begin
        rng = MersenneTwister(4242)
        for _ in 1:10
            m = rand(rng, 3:8)
            n = rand(rng, 3:8)
            nnz_target = rand(rng, 3:(m * n))
            I = rand(rng, 1:m, nnz_target)
            J = rand(rng, 1:n, nnz_target)
            V = [F2Elem(rand(rng, 0:1)) for _ in 1:nnz_target]
            A = sparse(I, J, V, m, n)
            dropzeros!(A)

            Ns = FL.nullspace(F2, A)
            Nd = FL.nullspace(F2, Matrix(A))

            @test size(Ns, 1) == n
            @test size(Ns, 2) == size(Nd, 2)
            @test A * Ns == f2mat(zeros(Int, m, size(Ns, 2)))
            @test FL.rank(F2, Ns) == size(Ns, 2)
        end
    end

    @testset "FpElem scalar/coercion properties" begin
        @testset "F2 arithmetic" begin
            K2 = CM.FpElem{2}
            a = K2(1)
            b = K2(1)
            @test a + b == K2(0)
            @test a * b == K2(1)
            @test a / b == K2(1)
        end

        @testset "F3 arithmetic + inverses" begin
            K3 = CM.FpElem{3}
            a = K3(2)
            @test a + K3(2) == K3(1)
            @test a * K3(2) == K3(1)
            @test a / K3(2) == K3(1)
            @test inv(a) * a == K3(1)
        end

        @testset "Fp(5) coercion + inverses" begin
            F5 = CM.Fp(5)
            K5 = CM.FpElem{5}
            @test CM.coerce(F5, 7) == K5(2)
            @test CM.coerce(F5, -1) == K5(4)
            @test inv(K5(2)) * K5(2) == K5(1)
        end
    end

    @testset "change_field helpers" begin
        FQ = CM.QQField()
        F2 = CM.F2()
        K = CM.coeff_type(FQ)
        @inline c(x) = CM.coerce(FQ, x)

        # --- Flange ---
        flats = [FZ.IndFlat(FZ.face(1, [1]), [0])]
        inj = [FZ.IndInj(FZ.face(1, [1]), [0])]
        phi = reshape(K[c(2)], 1, 1)
        FG = FZ.Flange(1, flats, inj, phi; field=FQ)
        FG2 = CM.change_field(FG, F2)
        @test FG2.field == F2
        @test eltype(FG2.phi) == CM.FpElem{2}
        @test FG2.phi[1, 1] == CM.FpElem{2}(0)

        # --- FringeModule ---
        P = chain_poset(2)
        U = FF.principal_upset(P, 1)
        D = FF.principal_downset(P, 1)
        phiH = spzeros(K, 1, 1)
        phiH[1, 1] = c(1)
        H = FF.FringeModule{K}(P, [U], [D], phiH; field=FQ)
        H2 = CM.change_field(H, F2)
        @test H2.field == F2
        @test eltype(H2.phi) == CM.FpElem{2}
        @test H2.phi[1, 1] == CM.FpElem{2}(1)

        # --- PModule + PMorphism ---
        edge = Dict{Tuple{Int,Int}, Matrix{K}}()
        edge[(1, 2)] = reshape(K[c(2)], 1, 1)
        M = MD.PModule{K}(P, [1, 1], edge)
        M2 = CM.change_field(M, F2)
        @test M2.field == F2
        @test eltype(M2.edge_maps.maps_to_succ[1][1]) == CM.FpElem{2}
        @test M2.edge_maps.maps_to_succ[1][1][1, 1] == CM.FpElem{2}(0)

        comps = [reshape(K[c(1)], 1, 1), reshape(K[c(2)], 1, 1)]
        f = MD.PMorphism(M, M, comps)
        f2 = CM.change_field(f, F2)
        @test f2.dom.field == F2
        @test f2.cod.field == F2
        @test f2.comps[1][1, 1] == CM.FpElem{2}(1)
        @test f2.comps[2][1, 1] == CM.FpElem{2}(0)

        # --- EncodingResult ---
        enc = RES.EncodingResult(P, M, nothing; H=H, presentation=FG)
        enc2 = CM.change_field(enc, F2)
        @test enc2.M.field == F2
        @test enc2.H.field == F2
        @test enc2.presentation.field == F2

        # --- ResolutionResult ---
        res = RES.ResolutionResult(M; enc=enc)
        res2 = CM.change_field(res, F2)
        @test res2.enc.M.field == F2
        @test res2.res.field == F2

        # --- InvariantResult ---
        inv = RES.InvariantResult(enc, :dummy, 7)
        inv2 = CM.change_field(inv, F2)
        @test inv2.enc.M.field == F2
        @test inv2.value == 7
    end

    @testset "_SparseRowAccumulator correctness" begin
        K = QQ
        acc = FL._SparseRowAccumulator{K}(12)
        row = FL.SparseRow{K}()

        FL._reset_sparse_row_accumulator!(acc)
        FL._push_sparse_row_entry!(acc, 4, QQ(2))
        FL._push_sparse_row_entry!(acc, 2, QQ(3))
        FL._push_sparse_row_entry!(acc, 4, QQ(-2)) # cancellation
        FL._push_sparse_row_entry!(acc, 7, QQ(5))
        FL._push_sparse_row_entry!(acc, 2, QQ(1))
        FL._materialize_sparse_row!(row, acc)
        @test row.idx == [2, 7]
        @test row.val == QQ[4, 5]

        FL._reset_sparse_row_accumulator!(acc)
        FL._push_sparse_row_entry!(acc, 1, QQ(1))
        FL._materialize_sparse_row!(row, acc)
        @test row.idx == [1]
        @test row.val == QQ[1]
    end

    @testset "_row_axpy! merge correctness" begin
        row = FL.SparseRow{QQ}(Int[1, 3, 6], QQ[2, 5, -1])
        other = FL.SparseRow{QQ}(Int[2, 3, 5], QQ[4, -5, 7])
        tmp_idx = Int[]
        tmp_val = QQ[]

        tmp_idx, tmp_val = FL._row_axpy!(row, QQ(2), other, tmp_idx, tmp_val)
        @test row.idx == [1, 2, 3, 5, 6]
        @test row.val == QQ[2, 8, -5, 14, -1]
        @test isempty(tmp_idx)
        @test isempty(tmp_val)

        tmp_idx, tmp_val = FL._row_axpy!(row, QQ(-1), FL.SparseRow{QQ}(Int[1, 5], QQ[2, 9]), tmp_idx, tmp_val)
        @test row.idx == [2, 3, 5, 6]
        @test row.val == QQ[8, -5, 5, -1]
        @test isempty(tmp_idx)
        @test isempty(tmp_val)

        row_one = FL.SparseRow{QQ}(Int[2, 4], QQ[3, -2])
        tmp_idx, tmp_val = FL._row_axpy!(row_one, one(QQ), FL.SparseRow{QQ}(Int[1, 4], QQ[5, 2]), tmp_idx, tmp_val)
        @test row_one.idx == [1, 2]
        @test row_one.val == QQ[5, 3]
        @test isempty(tmp_idx)
        @test isempty(tmp_val)

        row_empty = FL.SparseRow{QQ}()
        tmp_idx, tmp_val = FL._row_axpy!(row_empty, -one(QQ), FL.SparseRow{QQ}(Int[2, 5], QQ[4, -7]), tmp_idx, tmp_val)
        @test row_empty.idx == [2, 5]
        @test row_empty.val == QQ[-4, 7]
        @test isempty(tmp_idx)
        @test isempty(tmp_val)
    end

    @testset "_SparseRREF pivot incidence correctness" begin
        R = FL._SparseRREF{QQ}(6)

        @test FL._sparse_rref_push_homogeneous!(R, FL.SparseRow{QQ}(Int[1, 2], QQ[1, 1]))
        @test FL._sparse_rref_push_homogeneous!(R, FL.SparseRow{QQ}(Int[2, 5], QQ[1, 1]))
        @test FL._sparse_rref_push_homogeneous!(R, FL.SparseRow{QQ}(Int[5], QQ[1]))

        @test R.pivot_cols == [1, 2, 5]
        @test FL._row_coeff(R.pivot_rows[1], 5) == QQ(0)
        @test FL._row_coeff(R.pivot_rows[2], 5) == QQ(0)
    end

    @testset "_SparseRREF exact incidence" begin
        function exact_incidence_ok(R)
            for j in 1:R.nvars
                expected = Int[]
                for (pos, row) in pairs(R.pivot_rows)
                    if j in row.idx[2:end]
                        push!(expected, pos)
                    end
                end
                sort!(expected)
                actual = sort!(collect(R.col_rows[j]))
                actual == expected || return false
            end
            return true
        end

        rows = [
            FL.SparseRow{QQ}(Int[1, 2, 4], QQ[1, 1, 1]),
            FL.SparseRow{QQ}(Int[2, 4, 5], QQ[1, -1, 1]),
            FL.SparseRow{QQ}(Int[3, 4, 6], QQ[1, 1, 1]),
            FL.SparseRow{QQ}(Int[1, 3, 5], QQ[1, -1, 1]),
        ]

        R = FL._SparseRREF{QQ}(6)
        for row in rows
            FL._sparse_rref_push_homogeneous!(R, copy(row))
        end

        @test R.pivot_cols == [1, 2, 3]
        @test exact_incidence_ok(R)
    end

    @testset "_SparseRREF recursive pivot elimination" begin
        R = FL._SparseRREF{QQ}(3)
        @test FL._sparse_rref_push_homogeneous!(R, FL.SparseRow{QQ}(Int[1, 2], QQ[1, 1]))
        @test FL._sparse_rref_push_homogeneous!(R, FL.SparseRow{QQ}(Int[2], QQ[1]))
        @test !FL._sparse_rref_push_homogeneous!(R, FL.SparseRow{QQ}(Int[1], QQ[1]))
        @test R.pivot_cols == [1, 2]
    end

    @testset "_SparseREF rank parity" begin
        rows = [
            FL.SparseRow{QQ}(Int[1, 3, 6], QQ[2, 1, -1]),
            FL.SparseRow{QQ}(Int[2, 4], QQ[3, 1]),
            FL.SparseRow{QQ}(Int[1, 2, 5], QQ[1, -2, 4]),
            FL.SparseRow{QQ}(Int[3, 5, 6], QQ[5, 2, 1]),
            FL.SparseRow{QQ}(Int[4, 6], QQ[1, -3]),
        ]
        RR = FL._SparseRREF{QQ}(6)
        RE = FL._SparseREF{QQ}(6)
        rank_rref = 0
        rank_ref = 0
        for row in rows
            rank_rref += FL._sparse_rref_push_homogeneous!(RR, copy(row)) ? 1 : 0
            rank_ref += FL._sparse_ref_push_homogeneous!(RE, copy(row)) ? 1 : 0
        end
        @test rank_ref == rank_rref == 5
        @test RE.pivot_cols == RR.pivot_cols
        @test length(RE.pivot_inv) == length(RE.pivot_cols)
    end

    @testset "backend matrix storage hooks" begin
        FQ = CM.QQField()
        K = CM.coeff_type(FQ)
        P = chain_poset(2)
        old_nemo = FL._NEMO_ENABLED[]
        try
            FL._NEMO_ENABLED[] = true
            A = Matrix{K}(I, 256, 256) # above default Nemo threshold (50_000 entries)
            edge = Dict{Tuple{Int,Int}, Matrix{K}}((1, 2) => A)
            M = MD.PModule{K}(P, [256, 256], edge; field=FQ)
            @test M.edge_maps[1, 2] isa CM.BackendMatrix{K}
            @test Matrix(M.edge_maps[1, 2]) == A

            comps = [Matrix{K}(I, 256, 256), Matrix{K}(I, 256, 256)]
            f = MD.PMorphism(M, M, comps)
            @test f.comps[1] isa CM.BackendMatrix{K}
            @test f.comps[2] isa CM.BackendMatrix{K}
            @test Matrix(f.comps[1]) == comps[1]
        finally
            FL._NEMO_ENABLED[] = old_nemo
        end
    end

    @testset "F3 rank + rank_dim (dense)" begin
        F3 = CM.F3()
        F3Elem = CM.FpElem{3}
        Aint = [
            1 2 0 1;
            2 1 1 0;
            1 1 2 2;
            0 2 1 1
        ]
        A = F3Elem.(Aint .% 3)
        r1 = FL.rank(F3, A)
        r2 = FL.rank_dim(F3, A)
        @test r1 == r2
        @test 0 <= r1 <= min(size(A)...)
    end

    @testset "F3 nullspace + rref" begin
        F3 = CM.F3()
        F3Elem = CM.FpElem{3}
        Aint = [
            1 2 0 1;
            0 1 1 2;
            2 1 1 0
        ]
        A = F3Elem.(Aint .% 3)
        N = FL.nullspace(F3, A)
        @test A * N == zeros(F3Elem, size(A, 1), size(N, 2))

        R, pivs = FL.rref(F3, A; pivots=true)
        @test length(pivs) == FL.rank(F3, A)
        @test all(p -> 1 <= p <= size(A, 2), pivs)
        @test size(R) == size(A)
    end

    @testset "F3 solve_fullcolumn (dense + sparse)" begin
        F3 = CM.F3()
        F3Elem = CM.FpElem{3}
        n = 4
        m = 7
        B = vcat(F3Elem.(Matrix{Int}(I, n, n)), F3Elem.(rand(0:2, m - n, n)))
        X = F3Elem.(rand(0:2, n, 2))
        Y = B * X
        Xhat = FL.solve_fullcolumn(F3, B, Y)
        @test Xhat == X

        Bs = sparse(B)
        Y2 = Bs * X
        Xhat2 = FL.solve_fullcolumn(F3, Bs, Y2)
        @test Xhat2 == X
    end

    @testset "F3 rank_restricted + colspace sparse" begin
        F3 = CM.F3()
        F3Elem = CM.FpElem{3}
        Aint = [
            1 2 0 1 2;
            2 1 1 0 1;
            1 1 2 2 0;
            0 2 1 1 2
        ]
        A = sparse(F3Elem.(Aint .% 3))
        rows = [1, 3, 4]
        cols = [2, 3, 5]
        r1 = FL.rank_restricted(F3, A, rows, cols)
        r2 = FL.rank(F3, Matrix(A)[rows, cols])
        @test r1 == r2

        C = FL.colspace(F3, A)
        @test size(C, 1) == size(A, 1)
        @test FL.rank(F3, C) == FL.rank(F3, A)
    end

    @testset "F3 edge cases" begin
        F3 = CM.F3()
        F3Elem = CM.FpElem{3}
        A0 = F3Elem.(zeros(Int, 0, 5))
        @test FL.rank(F3, A0) == 0
        N0 = FL.nullspace(F3, A0)
        @test size(N0) == (5, 5)

        A1 = F3Elem.(zeros(Int, 4, 0))
        @test FL.rank(F3, A1) == 0
        N1 = FL.nullspace(F3, A1)
        @test size(N1) == (0, 0)
    end

    @testset "F3 solve_fullcolumn cache parity" begin
        F3 = CM.F3()
        F3Elem = CM.FpElem{3}
        B = vcat(F3Elem.(Matrix{Int}(I, 3, 3)), F3Elem.([1 2 0; 2 1 1]))
        X = F3Elem.([1 0; 2 1; 1 2])
        Y = B * X

        FL._clear_f3_fullcolumn_cache!()
        x1 = FL.solve_fullcolumn(F3, B, Y; cache=true)
        @test x1 == X
        @test haskey(FL._F3_FULLCOLUMN_FACTOR_CACHE, B)

        x2 = FL.solve_fullcolumn(F3, B, Y; cache=true)
        @test x2 == X

        fact = FL._F3_FULLCOLUMN_FACTOR_CACHE[B]
        x3 = FL.solve_fullcolumn(F3, B, Y; cache=false, factor=fact)
        @test x3 == X
    end

    @testset "QQ vs F2 vs F3 cache parity (basic behavior)" begin
        FL._clear_fullcolumn_cache!()
        FL._clear_f2_fullcolumn_cache!()
        FL._clear_f3_fullcolumn_cache!()
        @test isempty(FL._FULLCOLUMN_FACTOR_CACHE)
        @test isempty(FL._F2_FULLCOLUMN_FACTOR_CACHE)
        @test isempty(FL._F3_FULLCOLUMN_FACTOR_CACHE)
    end

    @testset "F3 randomized properties" begin
        F3 = CM.F3()
        F3Elem = CM.FpElem{3}
        rng = MersenneTwister(20240203)

        for _ in 1:20
            m = rand(rng, 2:8)
            n = rand(rng, 2:8)
            A = F3Elem.(rand(rng, 0:2, m, n))

            r = FL.rank(F3, A)
            N = FL.nullspace(F3, A)
            @test size(N, 1) == n
            @test size(N, 2) == n - r
            @test A * N == zeros(F3Elem, m, size(N, 2))

            C = FL.colspace(F3, A)
            @test FL.rank(F3, C) == r
        end
    end

    @testset "F3 sparse nullspace agrees with dense" begin
        F3 = CM.F3()
        F3Elem = CM.FpElem{3}
        rng = MersenneTwister(4243)
        for _ in 1:10
            m = rand(rng, 3:8)
            n = rand(rng, 3:8)
            nnz_target = rand(rng, 3:(m * n))
            I = rand(rng, 1:m, nnz_target)
            J = rand(rng, 1:n, nnz_target)
            V = [F3Elem(rand(rng, 0:2)) for _ in 1:nnz_target]
            A = sparse(I, J, V, m, n)
            dropzeros!(A)

            Ns = FL.nullspace(F3, A)
            Nd = FL.nullspace(F3, Matrix(A))

            @test size(Ns, 1) == n
            @test size(Ns, 2) == size(Nd, 2)
            @test A * Ns == zeros(F3Elem, m, size(Ns, 2))
            @test FL.rank(F3, Ns) == size(Ns, 2)
        end
    end

    @testset "FieldLinAlg contract tests (all fields)" begin
        rng = MersenneTwister(20250202)

        function _is_real(field)
            return field isa CM.RealField
        end

        function _contract_tol(field, A)
            field isa CM.RealField || return zero(eltype(A))
            return field.atol + field.rtol * opnorm(Matrix(A), 1)
        end

        with_fields(FIELDS_FULL) do field
            K = CM.coeff_type(field)
            @testset "contracts over $(field)" begin
                m = rand(rng, 4:9)
                n = rand(rng, 4:9)
                A = CM.rand(field, m, n)

                r = FL.rank(field, A)
                @test r == FL.rank_dim(field, A)

                N = FL.nullspace(field, A)
                @test size(N, 1) == n
                @test size(N, 2) == n - r
                if _is_real(field)
                    @test norm(Matrix(A) * N) <= _contract_tol(field, A) + 1e-8
                else
                    @test A * N == zeros(K, m, size(N, 2))
                end

                C = FL.colspace(field, A)
                @test FL.rank(field, C) == r

                # solve_fullcolumn for full column rank
                ncol = rand(rng, 2:5)
                nrow = ncol + rand(rng, 1:5)
                B = vcat(CM.eye(field, ncol), CM.rand(field, nrow - ncol, ncol))
                X = CM.rand(field, ncol, 2)
                Y = B * X
                Xhat = FL.solve_fullcolumn(field, B, Y)
                if _is_real(field)
                    @test norm(Matrix(B) * Xhat - Matrix(Y)) <= _contract_tol(field, B) + 1e-8
                else
                    @test B * Xhat == Y
                end
            end
        end
    end

    @testset "rank_restricted API parity (dense + sparse)" begin
        with_fields(FIELDS_FULL) do field
            @testset "restricted rank over $(field)" begin
                Aint = [
                    1 0 2 1 0 3 1;
                    0 1 1 0 2 1 0;
                    1 1 0 2 1 0 2;
                    0 0 1 1 1 2 0;
                    2 1 0 1 0 1 1;
                    1 0 1 0 1 0 1
                ]
                A = CM.coerce.(Ref(field), Aint)
                As = sparse(A)
                rows = [1, 3, 4, 6]
                cols = [1, 2, 5, 7]
                row_words = _selection_words(rows, size(A, 1))
                col_words = _selection_words(cols, size(A, 2))

                rd = FL.rank_restricted(field, A, rows, cols)
                rd_ref = FL.rank(field, A[rows, cols])
                @test rd == rd_ref

                rd_words = FL.rank_restricted_words(field, A, row_words, col_words, length(rows), length(cols);
                                                   nrows=size(A, 1), ncols=size(A, 2))
                @test rd_words == rd_ref

                Cd_words, cpiv_words = FL.colspace_restricted_words(field, A, row_words, col_words,
                                                                    length(rows), length(cols);
                                                                    nrows=size(A, 1), ncols=size(A, 2),
                                                                    pivots=true)
                @test size(Cd_words, 1) == length(rows)
                @test size(Cd_words, 2) == rd_ref
                ref_pivots = cols[collect(last(FL.rref(field, A[rows, cols]; pivots=true)))]
                if field isa CM.RealField
                    @test length(cpiv_words) == rd_ref
                    @test FL.rank(field, A[rows, cpiv_words]) == rd_ref
                    @test norm(Matrix(Cd_words) - Matrix(A[rows, cpiv_words])) <= _tol(field, A[rows, cpiv_words]) + 1e-8
                else
                    @test cpiv_words == ref_pivots
                end

                tiny_rows = [2, 5]
                tiny_cols = [3, 6]
                tiny_row_words = _selection_words(tiny_rows, size(A, 1))
                tiny_col_words = _selection_words(tiny_cols, size(A, 2))
                tiny_ref = FL.rank(field, A[tiny_rows, tiny_cols])
                @test FL.rank_restricted_words(field, A, tiny_row_words, tiny_col_words,
                                               length(tiny_rows), length(tiny_cols);
                                               nrows=size(A, 1), ncols=size(A, 2)) == tiny_ref
                if field isa CM.QQField
                    @test FL.rank_restricted_words(field, A, row_words, col_words, length(rows), length(cols);
                                                   nrows=size(A, 1), ncols=size(A, 2),
                                                   backend=:exact) == rd_ref
                    @test FL.rank_restricted_words(field, A, row_words, col_words, length(rows), length(cols);
                                                   nrows=size(A, 1), ncols=size(A, 2),
                                                   backend=:modular) == rd_ref
                    @test FL.rank_restricted_words(field, A, row_words, col_words, length(rows), length(cols);
                                                   nrows=size(A, 1), ncols=size(A, 2),
                                                   backend=:nemo) == rd_ref
                end

                rs = FL.rank_restricted(field, As, rows, cols)
                rs_ref = FL.rank(field, Matrix(As)[rows, cols])
                @test rs == rs_ref

                rs_words = FL.rank_restricted_words(field, As, row_words, col_words, length(rows), length(cols);
                                                   nrows=size(A, 1), ncols=size(A, 2))
                @test rs_words == rs_ref

                Cs_words, cspiv_words = FL.colspace_restricted_words(field, As, row_words, col_words,
                                                                     length(rows), length(cols);
                                                                     nrows=size(A, 1), ncols=size(A, 2),
                                                                     pivots=true)
                @test size(Cs_words, 1) == length(rows)
                @test size(Cs_words, 2) == rs_ref
                ref_pivots_s = cols[collect(last(FL.rref(field, A[rows, cols]; pivots=true)))]
                if field isa CM.RealField
                    @test length(cspiv_words) == rs_ref
                    @test FL.rank(field, A[rows, cspiv_words]) == rs_ref
                    @test norm(Matrix(Cs_words) - Matrix(A[rows, cspiv_words])) <= _tol(field, A[rows, cspiv_words]) + 1e-8
                else
                    @test cspiv_words == ref_pivots_s
                end

                @test FL.rank_restricted_words(field, A, row_words, zeros(UInt64, length(col_words)), length(rows), 0;
                                               nrows=size(A, 1), ncols=size(A, 2)) == 0
                @test FL.rank_restricted_words(field, A, zeros(UInt64, length(row_words)), col_words, 0, length(cols);
                                               nrows=size(A, 1), ncols=size(A, 2)) == 0
                @test FL.rank_restricted(field, A, Int[], cols) == 0
                @test FL.rank_restricted(field, A, rows, Int[]) == 0
                @test FL.rank_restricted(field, As, Int[], cols) == 0
                @test FL.rank_restricted(field, As, rows, Int[]) == 0
            end
        end
    end

    @testset "restricted nullspace/solve API parity (dense + sparse)" begin
        rng = MersenneTwister(20260206)
        with_fields(FIELDS_FULL) do field
            K = CM.coeff_type(field)
            is_real = field isa CM.RealField
            tol = is_real ? (field.atol + field.rtol + 1e-8) : 0.0

            @testset "restricted nullspace/solve over $(field)" begin
                Aint = [
                    1 0 2 1 0;
                    0 1 1 0 2;
                    1 1 0 2 1;
                    0 0 1 1 1;
                    2 1 0 1 0
                ]
                A = CM.coerce.(Ref(field), Aint)
                As = sparse(A)
                rows = [1, 3, 4]
                cols = [1, 2, 5]

                Nd = FL.nullspace_restricted(field, A, rows, cols)
                Ns = FL.nullspace_restricted(field, As, rows, cols)
                Aref = A[rows, cols]
                if is_real
                    @test norm(Matrix(Aref) * Nd) <= tol
                    @test norm(Matrix(Aref) * Ns) <= tol
                else
                    @test Aref * Nd == zeros(K, length(rows), size(Nd, 2))
                    @test Aref * Ns == zeros(K, length(rows), size(Ns, 2))
                end
                @test size(Nd, 1) == length(cols)
                @test size(Ns, 1) == length(cols)

                Bn = 3
                Bm = 6
                Bfull = vcat(CM.eye(field, Bn), CM.rand(field, Bm - Bn, Bn))
                Xtrue = CM.rand(field, Bn, 2)
                Yfull = Bfull * Xtrue
                srows = [1, 2, 3, 5]
                scols = [1, 2, 3]
                ycols = collect(axes(Yfull, 2))
                Xd = FL.solve_fullcolumn_restricted(field, Bfull, srows, scols, Yfull)
                Xs = FL.solve_fullcolumn_restricted(field, sparse(Bfull), srows, scols, Yfull)
                row_words = _selection_words(srows, size(Bfull, 1))
                col_words = _selection_words(ycols, size(Yfull, 2))
                Xdw = FL.solve_fullcolumn_restricted_words(field, Bfull[srows, scols], Yfull,
                                                          row_words, col_words, length(srows), length(ycols);
                                                          nrows=size(Bfull, 1), ncols=size(Yfull, 2))
                Xsw = FL.solve_fullcolumn_restricted_words(field, sparse(Bfull[srows, scols]), Yfull,
                                                          row_words, col_words, length(srows), length(ycols);
                                                          nrows=size(Bfull, 1), ncols=size(Yfull, 2))
                if is_real
                    @test norm(Matrix(Bfull[srows, scols]) * Xd - Matrix(Yfull[srows, :])) <= tol
                    @test norm(Matrix(Bfull[srows, scols]) * Xs - Matrix(Yfull[srows, :])) <= tol
                    @test norm(Matrix(Bfull[srows, scols]) * Xdw - Matrix(Yfull[srows, :])) <= tol
                    @test norm(Matrix(Bfull[srows, scols]) * Xsw - Matrix(Yfull[srows, :])) <= tol
                else
                    @test Bfull[srows, scols] * Xd == Yfull[srows, :]
                    @test Bfull[srows, scols] * Xs == Yfull[srows, :]
                    @test Bfull[srows, scols] * Xdw == Yfull[srows, :]
                    @test Bfull[srows, scols] * Xsw == Yfull[srows, :]
                end

                @test size(FL.nullspace_restricted(field, A, Int[], cols), 2) == length(cols)
                @test size(FL.nullspace_restricted(field, As, rows, Int[])) == (0, 0)
            end
        end
    end

    @testset "restricted sparse exact-oracle fixtures (adversarial patterns)" begin
        exact_fields = (CM.QQField(), CM.F2(), CM.F3(), CM.Fp(5))

        function _coerce_int_mat(field, Aint::AbstractMatrix{<:Integer})
            K = CM.coeff_type(field)
            m, n = size(Aint)
            A = Matrix{K}(undef, m, n)
            @inbounds for i in 1:m, j in 1:n
                A[i, j] = CM.coerce(field, Aint[i, j])
            end
            return A
        end

        # Rows are crafted as linear combinations with disjoint supports so rank/nullity
        # are known a priori over every exact field.
        Aint = [
            1 0 0 0 1 0 0 0 0 0 0;
            0 1 0 0 0 1 0 0 0 0 0;
            0 0 1 0 0 0 1 0 0 0 0;
            0 0 0 1 0 0 0 1 0 0 0;
            1 1 0 0 1 1 0 0 0 0 0;
            0 0 1 1 0 0 1 1 0 0 0;
            1 0 1 0 1 0 1 0 0 0 0;
            0 1 0 1 0 1 0 1 0 0 0;
            0 0 0 0 0 0 0 0 0 0 0
        ]
        rows = [8, 5, 2, 7, 1, 6]
        cols = [8, 6, 4, 2, 7, 5, 3, 1]
        expected_rank = 4
        expected_nullity = length(cols) - expected_rank

        for field in exact_fields
            K = CM.coeff_type(field)
            A = _coerce_int_mat(field, Aint)
            As = sparse(A)

            rr = FL.rank_restricted(field, As, rows, cols)
            @test rr == expected_rank

            N = FL.nullspace_restricted(field, As, rows, cols)
            @test size(N) == (length(cols), expected_nullity)
            @test As[rows, cols] * N == zeros(K, length(rows), expected_nullity)
        end

        Bint = [
            0 0 0 0 0 0;
            1 0 0 0 0 0;
            0 0 0 1 0 0;
            0 0 1 0 0 0;
            1 0 1 0 0 0;
            0 1 0 1 0 0;
            0 0 0 1 0 0;
            1 0 0 1 0 0
        ]
        srows = [2, 4, 5, 7, 8]
        scols = [1, 3, 4] # restricted block has known full column rank 3
        Xint = [
            1 0 1;
            0 1 1;
            1 1 0
        ]

        for field in exact_fields
            B = _coerce_int_mat(field, Bint)
            Bs = sparse(B)
            Xtrue = _coerce_int_mat(field, Xint)
            Y = B[:, scols] * Xtrue

            Xs = FL.solve_fullcolumn_restricted(field, Bs, srows, scols, Y)
            Xd = FL.solve_fullcolumn_restricted(field, B, srows, scols, Y)
            @test Xs == Xtrue
            @test Xd == Xtrue
        end
    end

    @testset "Characteristic-sensitive rank (small matrix)" begin
        A = [2 0;
             0 0]

        rqq = FL.rank(CM.QQField(), QQ.(A))
        rf2 = FL.rank(CM.F2(), CM.FpElem{2}.(A))
        rf3 = FL.rank(CM.F3(), CM.FpElem{3}.(A))

        @test rqq == 1
        @test rf2 == 0
        @test rf3 == 1
    end

    let field = CM.QQField()
    @testset "QQ linear algebra (FieldLinAlg QQ engine)" begin
        # Rank test
        A = QQ[QQ(1) QQ(2);
               QQ(2) QQ(4)]
        @test FL._rankQQ(A) == 1
        @test FL.rank(CM.QQField(), A) == 1

        # Nullspace test: A * v = 0
        N = FL._nullspaceQQ(A)
        @test size(N, 1) == 2
        @test size(N, 2) == 1
        v = N[:, 1]
        @test A * v == zeros(QQ, 2)

        # Solve full column rank system B*x = y
        B = QQ[QQ(1) QQ(0);
               QQ(0) QQ(1);
               QQ(1) QQ(1)]
        x_true = QQ[QQ(1), QQ(2)]
        y = B * x_true
        x = FL._solve_fullcolumnQQ(B, y)
        @test B * x == y
        @test FL.solve_fullcolumn(CM.QQField(), B, y) == x

        # Multiple right-hand sides
        Y = hcat(y, QQ(2) .* y)
        X = FL._solve_fullcolumnQQ(B, Y)
        @test B * X == Y

        @testset "_rrefQQ / _colspaceQQ / _solve_fullcolumnQQ edge cases" begin
            A = QQ[1 2 3;
                   2 4 6;
                   1 1 1]
            R, piv = FL._rrefQQ(A)

            @test piv == (1, 2)
            @test R == QQ[1 0 -1;
                          0 1  2;
                          0 0  0]

            C = FL._colspaceQQ(A)
            @test size(C) == (3, 2)
            @test C == A[:, collect(piv)]

            B = QQ[1 2;
                   2 4;
                   3 6]
            b = QQ[1, 2, 3]
            @test_throws ErrorException FL._solve_fullcolumnQQ(B, b)
        end
    end

    @testset "QQ sparse colspace agrees with dense" begin
        A = sparse(QQ[1 0 2;
                      0 1 3;
                      0 0 0])
        Cd = FL._colspaceQQ(Matrix{QQ}(A))
        Cs = FL._colspaceQQ(A)

        @test FL._rankQQ(Cd) == FL._rankQQ(Cs)
        @test FL._rankQQ(Cs) == FL._rankQQ(A)

        B = FL._colspaceQQ(A)
        for j in 1:size(Cs,2)
            x = FL._solve_fullcolumnQQ(B, Cs[:,j])
            @test B*x == Cs[:,j]
        end
    end

    @testset "QQ FullColumnFactor + cache correctness" begin
        B = QQ[1 0;
               0 1;
               1 1]
        y = B * QQ[2, 3]

        FL._clear_fullcolumn_cache!()
        x1 = FL._solve_fullcolumnQQ(B, y; cache=true)
        @test B*x1 == y
        @test haskey(FL._FULLCOLUMN_FACTOR_CACHE, B)
        x2 = FL._solve_fullcolumnQQ(B, y; cache=true)
        @test x2 == x1
        bady = QQ[1,0,0]
        @test_throws ErrorException FL._solve_fullcolumnQQ(B, bady; cache=true)

        # Nemo factor/cache path
        FL._clear_fullcolumn_cache!()
        xn = FL.solve_fullcolumn(CM.QQField(), B, y; backend=:nemo, cache=true)
        @test B * xn == y
        @test haskey(FL._NEMO_FULLCOLUMN_FACTOR_CACHE_QQ, B)
        xn2 = FL.solve_fullcolumn(CM.QQField(), B, y; backend=:nemo, cache=true)
        @test xn2 == xn
    end

    @testset "QQ factor_fullcolumn public wrapper + multi-RHS parity" begin
        F = CM.QQField()
        B = FL._qq_sparse_fullcolumn_rand(120, 40, 0.08; rng=MersenneTwister(0xA11CE))
        X = reshape(QQ[mod1(2i + 3j, 11) for i in 1:40, j in 1:12], 40, 12)
        Y = B * X

        fac = FL.factor_fullcolumn(F, B; backend=:auto, cache=false)
        @test fac isa FL.FullColumnSolveFactor
        @test FL.factor_backend(fac) == :julia_sparse
        Xhat = FL.solve_fullcolumn(F, B, Y; factor=fac, cache=false, check_rhs=true)
        @test Xhat == X

        jfac = FL._factor_fullcolumnQQ(B)
        Xraw = FL.solve_fullcolumn(F, B, Y; factor=jfac, cache=false, check_rhs=true)
        @test Xraw == Xhat

        @test_throws ErrorException FL.solve_fullcolumn(F, B, Y; backend=:nemo, factor=fac, cache=false)
    end

    @testset "QQ factor solve narrow/wide RHS gate parity" begin
        F = CM.QQField()
        B = FL._qq_sparse_fullcolumn_rand(90, 30, 0.09; rng=MersenneTwister(0xBEEF))
        fac = FL._factor_fullcolumnQQ(B)
        Xsmall = reshape(QQ[mod1(i + 2j, 7) for i in 1:30, j in 1:4], 30, 4)
        Ysmall = B * Xsmall
        Xwide = reshape(QQ[mod1(2i + 3j, 13) for i in 1:30, j in 1:12], 30, 12)
        Ywide = B * Xwide
        oldgate = FL._QQ_FACTOR_GATHER_MIN_RHS[]
        try
            FL._QQ_FACTOR_GATHER_MIN_RHS[] = typemax(Int)
            Xsmall_direct = FL.solve_fullcolumn(F, B, Ysmall; factor=fac, cache=false, check_rhs=true)
            Xwide_direct = FL.solve_fullcolumn(F, B, Ywide; factor=fac, cache=false, check_rhs=true)
            FL._QQ_FACTOR_GATHER_MIN_RHS[] = 1
            Xsmall_gather = FL.solve_fullcolumn(F, B, Ysmall; factor=fac, cache=false, check_rhs=true)
            Xwide_gather = FL.solve_fullcolumn(F, B, Ywide; factor=fac, cache=false, check_rhs=true)
            @test Xsmall_direct == Xsmall_gather == Xsmall
            @test Xwide_direct == Xwide_gather == Xwide
        finally
            FL._QQ_FACTOR_GATHER_MIN_RHS[] = oldgate
        end
    end

    @testset "QQ elimination_summary parity" begin
        F = CM.QQField()

        Ad = QQ[1 2 0;
                0 1 1;
                1 3 1;
                0 0 0]
        Sd = FL.elimination_summary(F, Ad; backend=:julia_exact)
        @test FL.rank(Sd) == FL.rank(F, Ad; backend=:julia_exact)
        @test FL.nullspace(Sd) == FL.nullspace(F, Ad; backend=:julia_exact)
        @test FL.colspace(Sd) == FL.colspace(F, Ad; backend=:julia_exact)
        @test FL._kernel_image_summary(Sd) == FL._kernel_image_summary(F, Ad; backend=:julia_exact)

        As = sparse(Ad)
        Ss = FL.elimination_summary(F, As)
        @test FL.rank(Ss) == FL.rank(F, As; backend=:julia_sparse)
        @test FL.nullspace(Ss) == FL.nullspace(F, As; backend=:julia_sparse)
        @test FL.colspace(Ss) == FL.colspace(F, As; backend=:julia_sparse)
        @test FL._kernel_image_summary(Ss) == FL._kernel_image_summary(F, As; backend=:julia_sparse)

        @test_throws ErrorException FL.elimination_summary(CM.F3(), Ad)
    end

    @testset "Nemo conversion counters" begin
        FL._reset_conversion_counters!()
        A = CM.BackendMatrix(QQ[1 2; 3 4]; backend=:nemo)
        r1 = FL.rank(CM.QQField(), A; backend=:nemo)
        @test r1 == 2
        c1 = FL._conversion_counters()
        @test c1.qq_to_nemo >= 1

        r2 = FL.rank(CM.QQField(), A; backend=:nemo)
        @test r2 == r1
        c2 = FL._conversion_counters()
        @test c2.qq_to_nemo_cache_hits >= c1.qq_to_nemo_cache_hits
    end

    @testset "QQ vs F2 cache parity (basic behavior)" begin
        @test isdefined(FL, :_clear_f2_fullcolumn_cache!)
        FL._clear_fullcolumn_cache!()
        FL._clear_f2_fullcolumn_cache!()
        @test isempty(FL._FULLCOLUMN_FACTOR_CACHE)
        @test isempty(FL._F2_FULLCOLUMN_FACTOR_CACHE)
    end

    @testset "QQ rank_dim matches exact rank on small matrices" begin
        A = QQ[1 2 3;
               2 4 6;
               1 0 1]
        @test FL._rankQQ(A) == 2
        @test FL._rankQQ_dim(A; backend=:auto) == 2
        @test FL._rankQQ_dim(A; backend=:modular) == 2
        @test FL.rank_dim(CM.QQField(), A; backend=:auto) == 2
    end

    @testset "QQ modular nullspace + solve" begin
        A = QQ[1 2 3 4;
               2 4 6 8;
               1 0 1 1]
        N = FL.nullspace(CM.QQField(), A; backend=:modular)
        @test size(N, 1) == size(A, 2)
        @test A * N == zeros(QQ, size(A, 1), size(N, 2))

        B = QQ[1 0;
               0 1;
               1 1;
               2 3]
        Xtrue = QQ[2 1;
                   3 4]
        Y = B * Xtrue
        X = FL.solve_fullcolumn(CM.QQField(), B, Y; backend=:modular)
        @test B * X == Y
    end

    @testset "QQ sparse nullspace + rank_restricted" begin
        A = sparse([1, 1, 2, 3],
                   [1, 3, 2, 4],
                   QQ[1, 2, -1, 3],
                   3, 4)

        Ns = FL._nullspaceQQ(A)
        Nd = FL._nullspaceQQ(Matrix(A))
        @test size(Ns, 1) == 4
        @test size(Ns, 2) == size(Nd, 2)
        @test A * Ns == zeros(QQ, size(A, 1), size(Ns, 2))
        @test FL._rankQQ(Ns) == size(Ns, 2)

        rng = MersenneTwister(123456)
        m, n = 30, 40
        nnz_target = 180
        I = rand(rng, 1:m, nnz_target)
        J = rand(rng, 1:n, nnz_target)
        V = [QQ(rand(rng, -3:3)) for _ in 1:nnz_target]
        A2 = sparse(I, J, V, m, n)
        dropzeros!(A2)

        for _ in 1:25
            rows = sort!(unique(rand(rng, 1:m, rand(rng, 1:15))))
            cols = sort!(unique(rand(rng, 1:n, rand(rng, 1:18))))
            r1 = FL._rankQQ_restricted(A2, rows, cols)
            r2 = FL._rankQQ(A2[rows, cols])
            @test r1 == r2
        end

        @test FL._rankQQ_restricted(A2, Int[], collect(1:n)) == 0
        @test FL._rankQQ_restricted(A2, collect(1:m), Int[]) == 0
        @test FL._rankQQ_restricted(A2, 1:m, 1:n) == FL._rankQQ(A2)
    end

    @testset "QQ rank_restricted dense" begin
        A = QQ[1 2 3 4;
               2 4 6 8;
               0 1 1 0;
               1 0 1 1]
        rows = [1, 3, 4]
        cols = [2, 3, 4]
        r1 = FL._rankQQ_restricted(sparse(A), rows, cols)
        r2 = FL._rankQQ(A[rows, cols])
        @test r1 == r2
    end

    @testset "QQ rref pivots on rectangular matrices" begin
        A = QQ[1 2 3 4;
               0 1 1 0]
        R, pivs = FL._rrefQQ(A)
        @test pivs == (1, 2)
        @test R == QQ[1 0 1 4;
                      0 1 1 0]

        B = QQ[1 0;
               0 1;
               1 1;
               0 0]
        Rb, pivs_b = FL._rrefQQ(B)
        @test pivs_b == (1, 2)
        @test Rb == QQ[1 0;
                       0 1;
                       0 0;
                       0 0]
    end

    @testset "QQ nullspace edge cases" begin
        A = Matrix{QQ}(I, 3, 3)
        N = FL._nullspaceQQ(A)
        @test size(N, 2) == 0

        Z = zeros(QQ, 2, 4)
        Nz = FL._nullspaceQQ(Z)
        @test size(Nz) == (4, 4)
        @test Z * Nz == zeros(QQ, 2, 4)
        @test FL._rankQQ(Nz) == 4
    end
    end

    @testset "Fp (p>3) engine parity" begin
        F5 = CM.Fp(5)
        F5Elem = CM.FpElem{5}
        fpmat(A) = F5Elem.(A .% 5)

        @testset "rank + rank_dim (dense)" begin
            Aint = [
                1 2 3 4;
                2 4 1 0;
                3 1 2 3;
                4 0 3 1
            ]
            A = fpmat(Aint)
            r1 = FL.rank(F5, A)
            r2 = FL.rank_dim(F5, A)
            @test r1 == r2
            @test 0 <= r1 <= min(size(A)...)
        end

        @testset "nullspace + rref" begin
            Aint = [
                1 2 3 4 0;
                0 1 1 2 3;
                2 1 4 0 1
            ]
            A = fpmat(Aint)
            N = FL.nullspace(F5, A)
            @test A * N == zeros(F5Elem, size(A, 1), size(N, 2))

            R, pivs = FL.rref(F5, A; pivots=true)
            @test length(pivs) == FL.rank(F5, A)
            @test all(p -> 1 <= p <= size(A, 2), pivs)
            @test size(R) == size(A)
        end

        @testset "solve_fullcolumn (dense + sparse)" begin
            n = 4
            m = 7
            B = vcat(fpmat(Matrix{Int}(I, n, n)), fpmat(rand(0:4, m - n, n)))
            X = fpmat(rand(0:4, n, 2))
            Y = B * X
            Xhat = FL.solve_fullcolumn(F5, B, Y)
            @test Xhat == X

            Bs = sparse(B)
            Y2 = Bs * X
            Xhat2 = FL.solve_fullcolumn(F5, Bs, Y2)
            @test Xhat2 == X
        end

        @testset "rank_restricted + colspace sparse" begin
            Aint = [
                1 2 0 1 2;
                2 1 1 0 1;
                1 1 2 2 0;
                0 2 1 1 2
            ]
            A = sparse(fpmat(Aint))
            rows = [1, 3, 4]
            cols = [2, 3, 5]
            r1 = FL.rank_restricted(F5, A, rows, cols)
            r2 = FL.rank(F5, Matrix(A)[rows, cols])
            @test r1 == r2

            C = FL.colspace(F5, A)
            @test size(C, 1) == size(A, 1)
            @test FL.rank(F5, C) == FL.rank(F5, A)
        end

        @testset "edge cases" begin
            A0 = fpmat(zeros(Int, 0, 5))
            @test FL.rank(F5, A0) == 0
            N0 = FL.nullspace(F5, A0)
            @test size(N0) == (5, 5)

            A1 = fpmat(zeros(Int, 4, 0))
            @test FL.rank(F5, A1) == 0
            N1 = FL.nullspace(F5, A1)
            @test size(N1) == (0, 0)
        end

        @testset "randomized properties + sparse nullspace" begin
            rng = MersenneTwister(20240204)
            for _ in 1:20
                m = rand(rng, 2:8)
                n = rand(rng, 2:8)
                A = fpmat(rand(rng, 0:4, m, n))
                r = FL.rank(F5, A)
                N = FL.nullspace(F5, A)
                @test size(N, 1) == n
                @test size(N, 2) == n - r
                @test A * N == zeros(F5Elem, m, size(N, 2))
            end

            for _ in 1:10
                m = rand(rng, 3:8)
                n = rand(rng, 3:8)
                nnz_target = rand(rng, 3:(m * n))
                I = rand(rng, 1:m, nnz_target)
                J = rand(rng, 1:n, nnz_target)
                V = [F5Elem(rand(rng, 0:4)) for _ in 1:nnz_target]
                A = sparse(I, J, V, m, n)
                dropzeros!(A)

                Ns = FL.nullspace(F5, A)
                Nd = FL.nullspace(F5, Matrix(A))
                @test size(Ns, 1) == n
                @test size(Ns, 2) == size(Nd, 2)
                @test A * Ns == zeros(F5Elem, m, size(Ns, 2))
                @test FL.rank(F5, Ns) == size(Ns, 2)
            end
        end

        @testset "sparse transpose/adjoint parity" begin
            A = sparse(fpmat([
                1 0 2 4;
                2 1 0 3;
                0 4 1 2
            ]))
            At = transpose(A)
            Aa = adjoint(A)
            @test FL.rank(F5, At) == FL.rank(F5, Matrix(At))
            @test FL.rank(F5, Aa) == FL.rank(F5, Matrix(Aa))

            Nt = FL.nullspace(F5, At)
            Na = FL.nullspace(F5, Aa)
            @test At * Nt == zeros(F5Elem, size(At, 1), size(Nt, 2))
            @test Aa * Na == zeros(F5Elem, size(Aa, 1), size(Na, 2))
        end

        @testset "backend selection prefers :fp_sparse for sparse" begin
            A = sparse(fpmat([1 0 2; 0 1 3; 2 3 1]))
            @test FL._choose_linalg_backend(F5, A; op=:rank) == :fp_sparse
            @test FL._choose_linalg_backend(F5, transpose(A); op=:nullspace) == :fp_sparse
            @test FL._choose_linalg_backend(F5, adjoint(A); op=:solve) == :fp_sparse
        end

        @testset "Nemo backend parity (p=5)" begin
            A = fpmat([1 2 3; 2 4 1; 3 1 2; 4 0 3])
            r = FL.rank(F5, A)
            rn = FL.rank(F5, A; backend=:nemo)
            @test r == rn
            Nn = FL.nullspace(F5, A; backend=:nemo)
            @test A * Nn == zeros(F5Elem, size(A, 1), size(Nn, 2))

            rd = FL.rank_dim(F5, A; backend=:nemo)
            @test rd == r

            B = fpmat([1 0;
                       0 1;
                       1 1;
                       2 3])
            Xtrue = fpmat([2 1;
                           3 4])
            Y = B * Xtrue
            Xn = FL.solve_fullcolumn(F5, B, Y; backend=:nemo, cache=true)
            @test Xn == Xtrue
            @test haskey(FL._NEMO_FULLCOLUMN_FACTOR_CACHE_FP, B)
        end

        @testset "backend algorithmic oracles on larger planted-rank fixtures" begin
            function _planted_rank_int(m::Int, n::Int, r::Int, rng::AbstractRNG)
                U = zeros(Int, m, r)
                V = zeros(Int, r, n)
                @inbounds for i in 1:r
                    U[i, i] = 1
                    V[i, i] = 1
                end
                @inbounds for i in r+1:m, j in 1:r
                    U[i, j] = rand(rng, 0:1)
                end
                @inbounds for i in 1:r, j in r+1:n
                    V[i, j] = rand(rng, 0:1)
                end
                return U * V
            end

            rng = MersenneTwister(20260214)
            m, n, r = 84, 116, 42
            Aint = _planted_rank_int(m, n, r, rng)
            A = fpmat(Aint)

            rj = FL.rank(F5, A; backend=:julia_exact)
            rn = FL.rank(F5, A; backend=:nemo)
            ra = FL.rank(F5, A; backend=:auto)
            @test rj == r
            @test rn == r
            @test ra == r

            Nn = FL.nullspace(F5, A; backend=:nemo)
            @test size(Nn) == (n, n - r)
            @test A * Nn == zeros(F5Elem, m, n - r)

            Bint = _planted_rank_int(88, 40, 40, rng)
            B = fpmat(Bint)
            Xtrue = fpmat(rand(rng, 0:4, 40, 3))
            Y = B * Xtrue
            Xj = FL.solve_fullcolumn(F5, B, Y; backend=:julia_exact)
            Xn = FL.solve_fullcolumn(F5, B, Y; backend=:nemo)
            @test Xj == Xtrue
            @test Xn == Xtrue
        end
    end

    @testset "QQ backend algorithmic oracles on larger planted-rank fixtures" begin
        function _planted_rank_int(m::Int, n::Int, r::Int, rng::AbstractRNG)
            U = zeros(Int, m, r)
            V = zeros(Int, r, n)
            @inbounds for i in 1:r
                U[i, i] = 1
                V[i, i] = 1
            end
            @inbounds for i in r+1:m, j in 1:r
                U[i, j] = rand(rng, 0:1)
            end
            @inbounds for i in 1:r, j in r+1:n
                V[i, j] = rand(rng, 0:1)
            end
            return U * V
        end

        rng = MersenneTwister(20260215)
        m, n, r = 72, 104, 36
        Aint = _planted_rank_int(m, n, r, rng)
        A = QQ.(Aint)
        qf = CM.QQField()

        @test FL.rank(qf, A; backend=:julia_exact) == r
        @test FL.rank(qf, A; backend=:nemo) == r
        @test FL.rank(qf, A; backend=:auto) == r
        @test FL.rank_dim(qf, A; backend=:modular) == r

        Nm = FL.nullspace(qf, A; backend=:modular)
        @test size(Nm) == (n, n - r)
        @test A * Nm == zeros(QQ, m, n - r)

        Bint = _planted_rank_int(80, 34, 34, rng)
        B = QQ.(Bint)
        Xtrue = QQ.(rand(rng, -2:2, 34, 2))
        Y = B * Xtrue

        Xm = FL.solve_fullcolumn(qf, B, Y; backend=:modular)
        Xn = FL.solve_fullcolumn(qf, B, Y; backend=:nemo)
        @test B * Xm == Y
        @test B * Xn == Y
    end

    @testset "Real engine parity" begin
        F = CM.RealField(Float64; rtol=1e-10, atol=1e-12)
        A = [1.0 2.0 3.0;
             2.0 4.0 6.0;
             0.0 1.0 1.0]
        r = FL.rank(F, A)
        @test r == 2
        @test FL.rank_dim(F, A) == 2

        N = FL.nullspace(F, A)
        @test size(N, 1) == size(A, 2)
        @test norm(A * N) <= 1e-8

        B = [1.0 0.0;
             0.0 1.0;
             1.0 1.0]
        x_true = [1.0, 2.0]
        y = B * x_true
        x = FL.solve_fullcolumn(F, B, y)
        @test norm(B * x - y) <= 1e-10

        R, pivs = FL.rref(F, A; pivots=true)
        @test length(pivs) == r
        @test size(R) == size(A)

        C = FL.colspace(F, A)
        @test size(C, 1) == size(A, 1)
        @test FL.rank(F, C) == r

        As = sparse(A)
        rs = FL.rank(F, As)
        @test rs == r

        Ns = FL.nullspace(F, As)
        @test size(Ns, 1) == size(A, 2)
        @test norm(As * Ns) <= 1e-8

        Bs = sparse(B)
        empty!(FL._FLOAT_SPARSE_FACTOR_CACHE)
        xs = FL.solve_fullcolumn(F, Bs, y; backend=:float_sparse_qr)
        @test norm(Bs * xs - y) <= 1e-10
        @test haskey(FL._FLOAT_SPARSE_FACTOR_CACHE, FL._float_sparse_cache_key(Bs))

        Rs, pivss = FL.rref(F, As; pivots=true)
        @test length(pivss) == rs
        @test size(Rs) == size(As)

        Cs = FL.colspace(F, As)
        @test size(Cs, 1) == size(As, 1)
        @test FL.rank(F, Cs) == rs

        @test FL._choose_linalg_backend(F, As; op=:rank) == :float_sparse_qr
    end

    @testset "Real engine algorithmic oracles" begin
        F = CM.RealField(Float64; rtol=1e-10, atol=1e-12)

        # Oracle 1: rank is known exactly from construction.
        Adep = [
            1.0 2.0 3.0 4.0;
            0.0 1.0 1.0 0.0;
            1.0 3.0 4.0 4.0; # row1 + row2
            2.0 4.0 6.0 8.0  # 2*row1
        ]
        @test FL.rank(F, Adep; backend=:float_dense_qr) == 2
        @test FL.rank(F, Adep; backend=:float_dense_svd) == 2

        # Oracle 2: near-diagonal matrix with unambiguous rank outcomes.
        Ahi = Matrix(Diagonal([1.0, 1e-6, 0.0]))
        Alo = Matrix(Diagonal([1.0, 1e-14, 0.0]))
        @test FL.rank(F, Ahi; backend=:float_dense_qr) == 2
        @test FL.rank(F, Ahi; backend=:float_dense_svd) == 2
        @test FL.rank(F, Alo; backend=:float_dense_qr) == 1
        @test FL.rank(F, Alo; backend=:float_dense_svd) == 1

        # Oracle 3: known nullspace direction span{[1,-2,1,0]}.
        Ak = [
            1.0 2.0 3.0 0.0;
            2.0 4.0 6.0 0.0;
            0.0 1.0 1.0 0.0
        ]
        vk = [-1.0, -1.0, 1.0, 0.0]
        Nq = FL.nullspace(F, Ak; backend=:float_dense_qr)
        Ns = FL.nullspace(F, Ak; backend=:float_dense_svd)
        @test size(Nq, 2) == 2
        @test size(Ns, 2) == 2
        @test norm(Ak * Nq) <= 1e-8
        @test norm(Ak * Ns) <= 1e-8
        # Oracle vector must lie in the span of each returned nullspace basis.
        cq = Nq \ vk
        cs = Ns \ vk
        @test norm(Nq * cq - vk) <= 1e-8
        @test norm(Ns * cs - vk) <= 1e-8

        # Oracle 4: solve with known exact RHS/solution across dense/sparse QR backends.
        B = [
            1.0 0.0 0.0;
            0.0 1.0 0.0;
            0.0 0.0 1.0;
            1.0 1.0 0.0;
            0.0 1.0 1.0;
            1.0 0.0 1.0
        ]
        Xtrue = [
            1.0  2.0;
            -1.0 3.0;
            0.5 -2.0
        ]
        Y = B * Xtrue
        Xd = FL.solve_fullcolumn(F, B, Y; backend=:float_dense_qr)
        Xs = FL.solve_fullcolumn(F, sparse(B), Y; backend=:float_sparse_qr)
        @test norm(Xd - Xtrue) <= 1e-10
        @test norm(Xs - Xtrue) <= 1e-10

        # Oracle 5: sparse restricted rank with a known submatrix rank.
        Abase = sparse([
            1.0 0.0 0.0 0.0 1.0 0.0;
            0.0 1.0 0.0 0.0 0.0 1.0;
            0.0 0.0 1.0 0.0 1.0 1.0;
            0.0 0.0 0.0 1.0 1.0 1.0;
            1.0 1.0 0.0 0.0 1.0 1.0;
            0.0 0.0 1.0 1.0 2.0 2.0
        ])
        rows = [1, 2, 5, 6]
        cols = [1, 2, 5, 6]
        # Submatrix:
        # [1 0 1 0
        #  0 1 0 1
        #  1 1 1 1
        #  0 0 2 2] has rank 3.
        @test FL.rank_restricted(F, Abase, rows, cols; backend=:float_sparse_qr) == 3

        # Optional oracle 6: sparse svds backend (when available).
        if FL._have_svds_backend()
            m = 30
            n = 42
            # [I_m 0] has rank m and nullity n-m.
            Asv = hcat(sparse(1.0I, m, m), spzeros(m, n - m))
            Nsv = FL.nullspace(F, Asv; backend=:float_sparse_svds)
            @test size(Nsv) == (n, n - m)
            @test norm(Asv * Nsv) <= 1e-7
            @test FL.rank(F, Asv; backend=:float_sparse_qr) == m
        end
    end

    @testset "Fp sparse adversarial oracle families (larger)" begin
        F5 = CM.Fp(5)
        F5Elem = CM.FpElem{5}
        rng = MersenneTwister(20260216)

        function _sparse_planted_rank_fp(rng::AbstractRNG, m::Int, n::Int, r::Int)
            U = zeros(Int, m, r)
            V = zeros(Int, r, n)
            @inbounds for i in 1:r
                U[i, i] = 1
                V[i, i] = 1
            end
            # Sparse low-density fill to keep structure adversarial but controlled.
            fill_u = max(1, div((m - r) * r, 14))
            fill_v = max(1, div(r * (n - r), 14))
            for _ in 1:fill_u
                i = rand(rng, r+1:m)
                j = rand(rng, 1:r)
                U[i, j] = rand(rng, 1:4)
            end
            for _ in 1:fill_v
                i = rand(rng, 1:r)
                j = rand(rng, r+1:n)
                V[i, j] = rand(rng, 1:4)
            end
            Aint = (U * V) .% 5
            return sparse(F5Elem.(Aint))
        end

        for (m, n, r) in ((80, 120, 45), (96, 144, 51))
            A = _sparse_planted_rank_fp(rng, m, n, r)
            @test FL.rank(F5, A; backend=:fp_sparse) == r
            @test FL.rank(F5, A; backend=:auto) == r

            N = FL.nullspace(F5, A; backend=:fp_sparse)
            @test size(N) == (n, n - r)
            @test A * N == zeros(F5Elem, m, n - r)

            Rt, pivs = FL.rref(F5, transpose(A); pivots=true, backend=:fp_sparse)
            @test length(pivs) == r
            @test size(Rt) == size(transpose(A))

            C = FL.colspace(F5, A; backend=:fp_sparse)
            @test size(C, 1) == m
            @test size(C, 2) == r
            @test FL.rank(F5, C; backend=:fp_sparse) == r
        end
    end

    @testset "Real numerically delicate oracle fixtures (backend-specific envelopes)" begin
        F = CM.RealField(Float64; rtol=1e-10, atol=1e-12)
        rng = MersenneTwister(20260217)

        function _orthonormal(rng::AbstractRNG, n::Int)
            Q = qr(randn(rng, n, n)).Q
            return Matrix(Q)
        end

        function _matrix_with_singulars(rng::AbstractRNG, sig::Vector{Float64})
            n = length(sig)
            U = _orthonormal(rng, n)
            V = _orthonormal(rng, n)
            return U * Diagonal(sig) * V'
        end

        # Tolerance envelope under F: tol ~= 1e-10 (scaled by opnorm).
        # So sigma=1e-8 should count, sigma=1e-12 should not.
        A4 = _matrix_with_singulars(rng, [1.0, 1e-2, 1e-5, 1e-8, 0.0])
        A3 = _matrix_with_singulars(rng, [1.0, 1e-2, 1e-5, 1e-12, 0.0])

        @test FL.rank(F, A4; backend=:float_dense_qr) == 4
        @test FL.rank(F, A4; backend=:float_dense_svd) == 4
        @test FL.rank(F, A3; backend=:float_dense_qr) == 3
        @test FL.rank(F, A3; backend=:float_dense_svd) == 3

        N4q = FL.nullspace(F, A4; backend=:float_dense_qr)
        N4s = FL.nullspace(F, A4; backend=:float_dense_svd)
        @test size(N4q, 2) == 1
        @test size(N4s, 2) == 1
        @test norm(A4 * N4q) <= 5e-7
        @test norm(A4 * N4s) <= 5e-7

        # Sparse delicate fixture with known nullity.
        As = sparse(A3)
        Ns_qr = FL.nullspace(F, As; backend=:float_sparse_qr)
        @test size(Ns_qr, 2) == 2
        @test norm(As * Ns_qr) <= 1e-6
        if FL._have_svds_backend()
            Ns_svds = FL.nullspace(F, As; backend=:float_sparse_svds)
            @test size(Ns_svds, 2) == 2
            @test norm(As * Ns_svds) <= 1e-5
        end
    end

    @testset "Real sparse extreme conditioning oracles (larger scale)" begin
        F = CM.RealField(Float64; rtol=1e-10, atol=1e-12)
        rng = MersenneTwister(20260218)

        function _sparse_rect_diag(diagvals::AbstractVector{<:Real}, m::Int, n::Int)
            d = min(length(diagvals), m, n)
            I = collect(1:d)
            J = collect(1:d)
            V = Float64.(diagvals[1:d])
            return sparse(I, J, V, m, n)
        end

        # Oracle family 1: tiny-pivot transition around tolerance envelope.
        # For F (rtol=1e-10, atol=1e-12), sigma=1e-6 counts, sigma=1e-14 drops.
        m = 220
        n = 320
        rbase = 170
        dhi = vcat(ones(rbase), [1e-6], zeros(m - rbase - 1))
        dlo = vcat(ones(rbase), [1e-14], zeros(m - rbase - 1))
        Ahi = _sparse_rect_diag(dhi, m, n)
        Alo = _sparse_rect_diag(dlo, m, n)

        @test FL.rank(F, Ahi; backend=:float_sparse_qr) == rbase + 1
        @test FL.rank(F, Alo; backend=:float_sparse_qr) == rbase

        Nhi_qr = FL.nullspace(F, Ahi; backend=:float_sparse_qr)
        Nlo_qr = FL.nullspace(F, Alo; backend=:float_sparse_qr)
        @test size(Nhi_qr, 2) == n - (rbase + 1)
        @test size(Nlo_qr, 2) == n - rbase
        @test norm(Ahi * Nhi_qr) <= 1e-7
        @test norm(Alo * Nlo_qr) <= 1e-7

        if FL._have_svds_backend()
            Nhi_svds = FL.nullspace(F, Ahi; backend=:float_sparse_svds)
            Nlo_svds = FL.nullspace(F, Alo; backend=:float_sparse_svds)
            @test size(Nhi_svds, 2) == n - (rbase + 1)
            @test size(Nlo_svds, 2) == n - rbase
            @test norm(Ahi * Nhi_svds) <= 1e-6
            @test norm(Alo * Nlo_svds) <= 1e-6
        end

        # Oracle family 2: large sparse ill-conditioned full-column solve.
        # We validate by residual envelope rather than exact X recovery.
        nb = 160
        mb = 260
        scales = 10.0 .^ range(0, -8, length=nb)
        Btop = _sparse_rect_diag(scales, nb, nb)
        extra_nnz = 8 * (mb - nb)
        I = rand(rng, 1:(mb - nb), extra_nnz)
        J = rand(rng, 1:nb, extra_nnz)
        V = randn(rng, extra_nnz) .* 1e-2
        Bbot = sparse(I, J, V, mb - nb, nb)
        B = vcat(Btop, Bbot)

        Xtrue = randn(rng, nb, 3)
        Y = B * Xtrue
        Xhat = FL.solve_fullcolumn(F, B, Y; backend=:float_sparse_qr)
        rel_res = norm(B * Xhat - Y) / max(1.0, norm(Y))
        @test rel_res <= 1e-7
    end

    @testset "backend-forced rref/colspace oracle checks (large sparse)" begin
        # QQ sparse oracle.
        qf = CM.QQField()
        mqq, nqq, rqq = 70, 110, 37
        Aqq = sparse(vcat(Matrix{Int}(I, rqq, rqq), zeros(Int, mqq - rqq, rqq)) *
                     hcat(Matrix{Int}(I, rqq, rqq), zeros(Int, rqq, nqq - rqq)))
        Aqq = sparse(QQ.(Matrix(Aqq)))
        Rq, pivq = FL.rref(qf, Aqq; pivots=true, backend=:julia_sparse)
        @test length(pivq) == rqq
        @test size(Rq) == size(Aqq)
        Cq = FL.colspace(qf, Aqq; backend=:julia_sparse)
        @test size(Cq, 2) == rqq
        @test FL.rank(qf, Cq; backend=:julia_exact) == rqq

        # Fp sparse oracle.
        F5 = CM.Fp(5)
        F5Elem = CM.FpElem{5}
        m5, n5, r5 = 76, 118, 44
        A5int = vcat(Matrix{Int}(I, r5, r5), zeros(Int, m5 - r5, r5)) *
                hcat(Matrix{Int}(I, r5, r5), zeros(Int, r5, n5 - r5))
        A5 = sparse(F5Elem.(A5int .% 5))
        R5, piv5 = FL.rref(F5, A5; pivots=true, backend=:fp_sparse)
        @test length(piv5) == r5
        C5 = FL.colspace(F5, A5; backend=:fp_sparse)
        @test size(C5, 2) == r5
        @test FL.rank(F5, C5; backend=:fp_sparse) == r5

        # Real sparse oracle.
        F = CM.RealField(Float64; rtol=1e-10, atol=1e-12)
        mr, nr, rr = 82, 124, 49
        Ar = sparse(vcat(Matrix{Float64}(I, rr, rr), zeros(Float64, mr - rr, rr)) *
                    hcat(Matrix{Float64}(I, rr, rr), zeros(Float64, rr, nr - rr)))
        Rr, pivr = FL.rref(F, Ar; pivots=true, backend=:float_sparse_qr)
        @test length(pivr) == rr
        Cr = FL.colspace(F, Ar; backend=:float_sparse_qr)
        @test size(Cr, 2) == rr
        @test FL.rank(F, Cr; backend=:float_sparse_qr) == rr
    end

    @testset "linalg threshold persistence + fingerprint gating" begin
        old = FL._current_linalg_thresholds()
        path = joinpath(mktempdir(), "linalg_thresholds.toml")

        FL._save_linalg_thresholds!(; path=path)
        FL.FP_NEMO_RANK_THRESHOLD[] = old["fp_nemo_rank_threshold"] + 111
        @test FL._load_linalg_thresholds!(; path=path, warn_on_mismatch=false)
        @test FL.FP_NEMO_RANK_THRESHOLD[] == old["fp_nemo_rank_threshold"]
        @test haskey(old, "modular_nullspace_threshold")
        @test haskey(old, "modular_solve_threshold")
        @test haskey(old, "modular_min_primes")
        @test haskey(old, "modular_max_primes")
        @test haskey(old, "rankqq_dim_small_threshold")
        @test haskey(old, "rankqq_restricted_words_nemo_threshold")
        @test haskey(old, "qq_nemo_rank_threshold_square")
        @test haskey(old, "qq_nemo_nullspace_threshold_tall")
        @test haskey(old, "qq_nemo_solve_threshold_wide")
        @test haskey(old, "qq_nemo_sparse_solve_threshold_square_low")
        @test haskey(old, "qq_nemo_sparse_solve_threshold_tall_mid")
        @test haskey(old, "qq_nemo_sparse_solve_threshold_wide_high")
        @test haskey(old, "qq_nemo_sparse_solve_policy_square_low")
        @test haskey(old, "qq_nemo_sparse_solve_policy_tall_mid")
        @test haskey(old, "qq_nemo_sparse_solve_policy_wide_high")
        @test haskey(old, "qq_modular_nullspace_threshold_square")
        @test haskey(old, "qq_modular_solve_threshold_tall")
        @test haskey(old, "zn_qq_dimat_submatrix_work_threshold")
        @test old["zn_qq_dimat_submatrix_work_threshold"] == FL.zn_qq_dimat_submatrix_work_threshold()
        @test old["zn_qq_dimat_submatrix_work_threshold"] >= 1

        doc = TOML.parsefile(path)
        doc["fingerprint"]["cpu_name"] = string(doc["fingerprint"]["cpu_name"], "_mismatch")
        open(path, "w") do io
            TOML.print(io, doc)
        end

        FL.FP_NEMO_RANK_THRESHOLD[] = old["fp_nemo_rank_threshold"] + 222
        @test !FL._load_linalg_thresholds!(; path=path, warn_on_mismatch=false)
        @test FL.FP_NEMO_RANK_THRESHOLD[] == old["fp_nemo_rank_threshold"] + 222

        @test FL._apply_linalg_thresholds!(old)
    end

    @testset "QQ backend routing uses op+shape thresholds" begin
        F = CM.QQField()
        old = FL._current_linalg_thresholds()
        try
            @test FL._apply_linalg_thresholds!(merge(
                old,
                Dict(
                    "qq_nemo_rank_threshold_square" => 10_000,
                    "qq_nemo_rank_threshold_tall" => 10_000,
                    "qq_nemo_rank_threshold_wide" => 10_000,
                    "qq_nemo_nullspace_threshold_square" => 10_000,
                    "qq_nemo_nullspace_threshold_tall" => 10_000,
                    "qq_nemo_nullspace_threshold_wide" => 10_000,
                    "qq_modular_nullspace_threshold_square" => 10_000,
                    "qq_modular_nullspace_threshold_tall" => 10_000,
                    "qq_modular_nullspace_threshold_wide" => 10_000,
                    "qq_nemo_solve_threshold_square" => 10_000,
                    "qq_nemo_solve_threshold_tall" => 10_000,
                    "qq_nemo_solve_threshold_wide" => 10_000,
                    "qq_nemo_sparse_solve_threshold_square_low" => 10_000,
                    "qq_nemo_sparse_solve_threshold_square_mid" => 10_000,
                    "qq_nemo_sparse_solve_threshold_square_high" => 10_000,
                    "qq_nemo_sparse_solve_threshold_tall_low" => 10_000,
                    "qq_nemo_sparse_solve_threshold_tall_mid" => 10_000,
                    "qq_nemo_sparse_solve_threshold_tall_high" => 10_000,
                    "qq_nemo_sparse_solve_threshold_wide_low" => 10_000,
                    "qq_nemo_sparse_solve_threshold_wide_mid" => 10_000,
                    "qq_nemo_sparse_solve_threshold_wide_high" => 10_000,
                    "qq_nemo_sparse_solve_policy_square_low" => 1,
                    "qq_nemo_sparse_solve_policy_square_mid" => 1,
                    "qq_nemo_sparse_solve_policy_square_high" => 1,
                    "qq_nemo_sparse_solve_policy_tall_low" => 1,
                    "qq_nemo_sparse_solve_policy_tall_mid" => 1,
                    "qq_nemo_sparse_solve_policy_tall_high" => 1,
                    "qq_nemo_sparse_solve_policy_wide_low" => 1,
                    "qq_nemo_sparse_solve_policy_wide_mid" => 1,
                    "qq_nemo_sparse_solve_policy_wide_high" => 1,
                    "qq_modular_solve_threshold_square" => 10_000,
                    "qq_modular_solve_threshold_tall" => 10_000,
                    "qq_modular_solve_threshold_wide" => 10_000,
                )
            ))
            A_small = fill(QQ(1), 20, 20)    # work=400 (square)
            A_tall = fill(QQ(1), 120, 40)    # work=4800 (tall)
            A_wide = fill(QQ(1), 40, 120)    # work=4800 (wide)
            As_tall = sparse(A_tall)
            As_wide = sparse(A_wide)
            @test FL._choose_linalg_backend(F, A_small; op=:rank) == :julia_exact
            @test FL._choose_linalg_backend(F, A_tall; op=:rank) == :julia_exact
            @test FL._choose_linalg_backend(F, A_wide; op=:rank) == :julia_exact
            @test FL._choose_linalg_backend(F, As_tall; op=:rank) == :julia_sparse
            @test FL._choose_linalg_backend(F, As_wide; op=:nullspace) == :julia_sparse
            @test FL._choose_linalg_backend(F, As_tall; op=:solve) == :julia_sparse

            @test FL._apply_linalg_thresholds!(merge(
                old,
                Dict(
                    "qq_nemo_rank_threshold_square" => 100,
                    "qq_nemo_rank_threshold_tall" => 100,
                    "qq_nemo_rank_threshold_wide" => 100,
                    "qq_nemo_nullspace_threshold_square" => 100,
                    "qq_nemo_nullspace_threshold_tall" => 100,
                    "qq_nemo_nullspace_threshold_wide" => 100,
                    "qq_nemo_solve_threshold_square" => 100,
                    "qq_nemo_solve_threshold_tall" => 100,
                    "qq_nemo_solve_threshold_wide" => 100,
                    "qq_nemo_sparse_solve_threshold_square_low" => 1,
                    "qq_nemo_sparse_solve_threshold_square_mid" => 1,
                    "qq_nemo_sparse_solve_threshold_square_high" => 1,
                    "qq_nemo_sparse_solve_threshold_tall_low" => 1,
                    "qq_nemo_sparse_solve_threshold_tall_mid" => 1,
                    "qq_nemo_sparse_solve_threshold_tall_high" => 1,
                    "qq_nemo_sparse_solve_threshold_wide_low" => 1,
                    "qq_nemo_sparse_solve_threshold_wide_mid" => 1,
                    "qq_nemo_sparse_solve_threshold_wide_high" => 1,
                    "qq_nemo_sparse_solve_policy_square_low" => 1,
                    "qq_nemo_sparse_solve_policy_square_mid" => 1,
                    "qq_nemo_sparse_solve_policy_square_high" => 1,
                    "qq_nemo_sparse_solve_policy_tall_low" => 1,
                    "qq_nemo_sparse_solve_policy_tall_mid" => 1,
                    "qq_nemo_sparse_solve_policy_tall_high" => 1,
                    "qq_nemo_sparse_solve_policy_wide_low" => 1,
                    "qq_nemo_sparse_solve_policy_wide_mid" => 1,
                    "qq_nemo_sparse_solve_policy_wide_high" => 1,
                )
            ))
            if FL._have_nemo()
                @test FL._choose_linalg_backend(F, A_small; op=:rank) == :nemo
                @test FL._choose_linalg_backend(F, A_tall; op=:rank) == :nemo
                @test FL._choose_linalg_backend(F, A_wide; op=:rank) == :nemo
                @test FL._choose_linalg_backend(F, As_tall; op=:rank) == :nemo
                @test FL._choose_linalg_backend(F, As_wide; op=:nullspace) == :nemo
                @test FL._choose_linalg_backend(F, As_tall; op=:solve) == :nemo
            end
        finally
            @test FL._apply_linalg_thresholds!(old)
        end
    end

    @testset "QQ restricted-words Nemo threshold persists + applies" begin
        old = FL._current_linalg_thresholds()
        path = joinpath(mktempdir(), "linalg_thresholds.toml")
        try
            @test FL._apply_linalg_thresholds!(merge(
                old,
                Dict("rankqq_restricted_words_nemo_threshold" => 37)
            ))
            @test FL.RANKQQ_RESTRICTED_WORDS_NEMO_THRESHOLD[] == 37
            FL._save_linalg_thresholds!(; path=path)
            @test FL._apply_linalg_thresholds!(merge(
                old,
                Dict("rankqq_restricted_words_nemo_threshold" => 11)
            ))
            @test FL.RANKQQ_RESTRICTED_WORDS_NEMO_THRESHOLD[] == 11
            @test FL._load_linalg_thresholds!(; path=path, warn_on_mismatch=false)
            @test FL.RANKQQ_RESTRICTED_WORDS_NEMO_THRESHOLD[] == 37
        finally
            @test FL._apply_linalg_thresholds!(old)
        end
    end

    @testset "QQ sparse solve routing uses density-bucket thresholds" begin
        F = CM.QQField()
        old = FL._current_linalg_thresholds()
        try
            B = sparse(fill(QQ(0), 90, 60))
            for i in 1:60
                B[i, i] = QQ(1)
            end
            # Moderate density pushes into :mid bucket on tall shape.
            for i in 61:90, j in 1:60
                if (i + 2j) % 5 == 0
                    B[i, j] = QQ(1)
                end
            end
            @test FL._apply_linalg_thresholds!(merge(
                old,
                Dict(
                    "qq_nemo_sparse_solve_threshold_tall_low" => 10_000,
                    "qq_nemo_sparse_solve_threshold_tall_mid" => 10_000,
                    "qq_nemo_sparse_solve_threshold_tall_high" => 10_000,
                    "qq_nemo_sparse_solve_policy_tall_low" => 1,
                    "qq_nemo_sparse_solve_policy_tall_mid" => 1,
                    "qq_nemo_sparse_solve_policy_tall_high" => 1,
                )
            ))
            @test FL._choose_linalg_backend(F, B; op=:solve) == :julia_sparse

            @test FL._apply_linalg_thresholds!(merge(
                old,
                Dict(
                    "qq_nemo_sparse_solve_threshold_tall_low" => 1,
                    "qq_nemo_sparse_solve_threshold_tall_mid" => 1,
                    "qq_nemo_sparse_solve_threshold_tall_high" => 1,
                    "qq_nemo_sparse_solve_policy_tall_low" => 1,
                    "qq_nemo_sparse_solve_policy_tall_mid" => 1,
                    "qq_nemo_sparse_solve_policy_tall_high" => 1,
                )
            ))
            if FL._have_nemo()
                @test FL._choose_linalg_backend(F, B; op=:solve) == :nemo
            else
                @test FL._choose_linalg_backend(F, B; op=:solve) == :julia_sparse
            end
        finally
            @test FL._apply_linalg_thresholds!(old)
        end
    end

    @testset "QQ sparse rank routing prefers julia on very sparse matrices" begin
        F = CM.QQField()
        old = FL._current_linalg_thresholds()
        try
            @test FL._apply_linalg_thresholds!(merge(
                old,
                Dict(
                    "qq_nemo_rank_threshold_square" => 100,
                    "qq_nemo_rank_threshold_tall" => 100,
                    "qq_nemo_rank_threshold_wide" => 100,
                )
            ))
            A = spzeros(QQ, 400, 300)
            for j in 1:300
                A[mod1(3j, 400), j] = QQ(1)
                j <= 100 && (A[mod1(5j + 7, 400), j] = QQ(2))
            end
            @test FL._choose_linalg_backend(F, A; op=:rank) == :julia_sparse
            @test FL.rank(F, A) == FL.rank(F, A; backend=:julia_sparse)
        finally
            @test FL._apply_linalg_thresholds!(old)
        end
    end

    @testset "QQ sparse colspace routing chooses the right backend by nnz regime" begin
        F = CM.QQField()
        A = spzeros(QQ, 1200, 220)
        for j in 1:220
            A[mod1(3j + 11, 1200), j] = QQ(1)
            A[mod1(7j + 19, 1200), j] = QQ(2)
        end
        @test FL._choose_linalg_backend(F, A; op=:colspace) == :julia_sparse
        Cauto = FL.colspace(F, A)
        Cj = FL.colspace(F, A; backend=:julia_sparse)
        @test FL.rank(F, Cauto) == FL.rank(F, Cj)

        Am = spzeros(QQ, 1200, 220)
        @inbounds for j in 1:220
            for t in 0:23
                Am[mod1(3j + 17t + 11, 1200), j] = QQ(1 + ((j + t) % 3))
            end
        end
        expected = FL._have_nemo() ? :nemo : :julia_sparse
        @test FL._choose_linalg_backend(F, Am; op=:colspace) == expected
        Cmod = FL.colspace(F, Am)
        @test FL.rank(F, Cmod) == FL.rank(F, FL.colspace(F, Am; backend=:julia_sparse))
        FL._have_nemo() && @test Cmod == FL.colspace(F, Am; backend=:nemo)
    end

    @testset "QQ sparse nullspace routing keeps Nemo on moderate-nnz sparse inputs" begin
        F = CM.QQField()
        old = FL._current_linalg_thresholds()
        try
            @test FL._apply_linalg_thresholds!(merge(
                old,
                Dict(
                    "qq_nemo_nullspace_threshold_square" => 100,
                    "qq_nemo_nullspace_threshold_tall" => 100,
                    "qq_nemo_nullspace_threshold_wide" => 100,
                )
            ))
            A = spzeros(QQ, 800, 120)
            @inbounds for j in 1:120
                for t in 0:71
                    A[mod1(j + 11t, 800), j] = QQ(1 + ((j + t) % 3))
                end
            end
            expected = FL._have_nemo() ? :nemo : :julia_sparse
            @test FL._choose_linalg_backend(F, A; op=:nullspace) == expected
        finally
            @test FL._apply_linalg_thresholds!(old)
        end
    end

    @testset "QQ sparse solve honors provided factor backend under auto routing" begin
        F = CM.QQField()
        Bsmall = spzeros(QQ, 800, 120)
        @inbounds for j in 1:120
            Bsmall[2j - 1, j] = QQ(1)
            Bsmall[mod1(5j + 7, 800), j] = QQ(2)
            j <= 80 && (Bsmall[mod1(7j + 13, 800), j] = QQ(3))
        end
        @test FL._choose_linalg_backend(F, Bsmall; op=:solve) == :julia_sparse

        B = FL._qq_sparse_fullcolumn_rand(500, 100, 0.05; rng=MersenneTwister(0x5EED))
        Xtrue = reshape(QQ[mod1(i + j, 5) for i in 1:100, j in 1:3], 100, 3)
        Y = B * Xtrue
        @test FL._choose_solve_backend(F, B; factor=nothing) ==
              FL._choose_linalg_backend(F, B; op=:solve)

        jfac = FL._factor_fullcolumnQQ(B)
        @test FL._choose_solve_backend(F, B; factor=jfac) == :julia_sparse
        Xj = FL.solve_fullcolumn(F, B, Y; backend=:auto, cache=false, factor=jfac, check_rhs=true)
        @test Xj == Xtrue

        if FL._have_nemo()
            nfac = FL._factor_fullcolumn_nemoQQ(B)
            @test FL._choose_solve_backend(F, B; factor=nfac) == :nemo
            Xn = FL.solve_fullcolumn(F, B, Y; backend=:auto, cache=false, factor=nfac, check_rhs=true)
            @test Xn == Xtrue
        end
    end

    @testset "QQ reusable analysis and factor summary" begin
        F = CM.QQField()
        A = FL._qq_sparse_rand(80, 50, 0.08; rng=MersenneTwister(0xA11CE))
        S = FL.elimination_summary(F, A)
        Ana = FL.analyze_matrix(F, A)
        @test FL.analysis_backend(Ana) in (:julia_exact, :julia_sparse, :nemo)
        @test FL.rank(Ana) == FL.rank(S)
        @test FL.nullspace(Ana) == FL.nullspace(S)
        @test FL.colspace(Ana) == FL.colspace(S)
        @test FL._kernel_image_summary(Ana) == FL._kernel_image_summary(S)

        B = FL._qq_sparse_fullcolumn_rand(120, 40, 0.06; rng=MersenneTwister(0xBEEF))
        Xtrue = reshape(QQ[CM.coerce(F, mod1(i + 2j, 7)) for i in 1:40, j in 1:6], 40, 6)
        Y = B * Xtrue
        Fac = FL.factor_fullcolumn(F, B; backend=:auto, cache=false)
        AnaF = FL.analyze_matrix(F, B; fullcolumn_factor=true, cache=false)
        @test FL.rank(Fac) == FL.rank(F, B)
        @test FL.colspace(Fac) == FL.colspace(F, B)
        @test FL.rank(AnaF) == FL.rank(F, B)
        @test FL.fullcolumn_factor(AnaF) !== nothing
        @test FL.solve_fullcolumn(F, B, Y; factor=Fac, check_rhs=true) == Xtrue
        @test FL.solve_fullcolumn(F, B, Y; analysis=AnaF, check_rhs=true) == Xtrue
        @test FL.elimination_summary(Fac) == FL.elimination_summary(AnaF)
        @test_throws ErrorException FL.solve_fullcolumn(F, B, Y; factor=Fac, analysis=AnaF)
    end

    @testset "Tiny <=4x4 fast-path parity" begin
        for field in FIELDS_FULL
            K = CM.coeff_type(field)
            tol = field isa CM.RealField ? (field.atol + 10 * field.rtol) : 0.0

            cmat(A::AbstractMatrix{<:Integer}) = begin
                m, n = size(A)
                M = Matrix{K}(undef, m, n)
                @inbounds for i in 1:m, j in 1:n
                    M[i, j] = CM.coerce(field, A[i, j])
                end
                M
            end

            A = cmat([
                1 2 3;
                2 4 6;
                0 1 1
            ])
            r = FL.rank(field, A)
            R, pivs = FL.rref(field, A; pivots=true)
            @test r == length(pivs)
            @test size(R) == size(A)

            B = cmat([
                1 2;
                0 1;
                1 1
            ])
            Xtrue = cmat([
                1 0;
                2 1
            ])
            Y = B * Xtrue
            Xhat = FL.solve_fullcolumn(field, B, Y)
            if field isa CM.RealField
                @test norm(B * Xhat - Y) <= max(tol, 1e-10)
            else
                @test Xhat == Xtrue
            end

            yv = Y[:, 1]
            xv = FL.solve_fullcolumn(field, B, yv)
            if field isa CM.RealField
                @test norm(B * xv - yv) <= max(tol, 1e-10)
            else
                @test xv == Xtrue[:, 1]
            end

            L = cmat([1 2 0; 0 1 1])
            Rm = cmat([1 0; 2 1; 0 1])
            Mtiny = FL._matmul(L, Rm)
            if field isa CM.RealField
                @test norm(Mtiny - (L * Rm)) <= max(tol, 1e-10)
            else
                @test Mtiny == L * Rm
            end

            As = sparse(A)
            rows = [1, 3]
            cols = [1, 2, 3]
            rr = FL.rank_restricted(field, As, rows, cols)
            @test rr == FL.rank(field, Matrix(A)[rows, cols])
        end
    end
end
