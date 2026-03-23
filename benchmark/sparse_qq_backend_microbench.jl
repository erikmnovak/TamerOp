#!/usr/bin/env julia
# sparse_qq_backend_microbench.jl
#
# Focused benchmark for exact sparse QQ backend routing on rank/nullspace.
# This isolates the sparse-Julia vs Nemo crossover on representative matrices
# so backend-heuristic changes can be checked without running the broad linalg
# scorecard.
#
# Kernel mapping:
# - exercises backend routing in `src/field_linalg/backend_routing.jl`
# - exercises sparse QQ backend crossover in `src/field_linalg/qq_engine.jl`
# - does not target the sparse streaming elimination kernel in
#   `src/field_linalg/sparse_rref.jl`

using Random
using SparseArrays

try
    using TamerOp
catch
    include(joinpath(@__DIR__, "..", "src", "TamerOp.jl"))
    using .TamerOp
end

const CM = TamerOp.CoreModules
const FL = TamerOp.FieldLinAlg

function _bench(name::AbstractString, f::Function; reps::Int=3)
    GC.gc()
    f()
    GC.gc()
    times_ms = Vector{Float64}(undef, reps)
    alloc_kib = Vector{Float64}(undef, reps)
    for i in 1:reps
        t = @timed f()
        times_ms[i] = 1000.0 * t.time
        alloc_kib[i] = t.bytes / 1024.0
    end
    sort!(times_ms)
    sort!(alloc_kib)
    mid = cld(reps, 2)
    println(rpad(name, 34),
            " median_time=", round(times_ms[mid], digits=3), " ms",
            " median_alloc=", round(alloc_kib[mid], digits=1), " KiB")
end

function _sparse_rank_case()
    A = spzeros(CM.QQ, 1309, 845)
    rng = Random.MersenneTwister(0xBEEF)
    @inbounds for j in 1:size(A, 2)
        for _ in 1:2
            A[rand(rng, 1:size(A, 1)), j] = CM.QQ(rand(rng, 1:3))
        end
    end
    return A
end

function _sparse_nullspace_case()
    A = spzeros(CM.QQ, 800, 120)
    @inbounds for j in 1:120
        for t in 0:71
            A[mod1(j + 11t, 800), j] = CM.QQ(1 + ((j + t) % 3))
        end
    end
    return A
end

function _sparse_colspace_case()
    A = spzeros(CM.QQ, 1200, 220)
    @inbounds for j in 1:220
        A[mod1(3j + 11, 1200), j] = CM.QQ(1)
        A[mod1(7j + 19, 1200), j] = CM.QQ(2)
    end
    return A
end

function _sparse_colspace_case_moderate()
    A = spzeros(CM.QQ, 1200, 220)
    @inbounds for j in 1:220
        for t in 0:23
            A[mod1(3j + 17t + 11, 1200), j] = CM.QQ(1 + ((j + t) % 3))
        end
    end
    return A
end

function _sparse_solve_case()
    B = FL._qq_sparse_fullcolumn_rand(800, 120, 0.02; rng=Random.MersenneTwister(0xFACE))
    X = [CM.QQ(i + j) for i in 1:size(B, 2), j in 1:8]
    Y = B * X
    return B, Y
end

function main()
    F = CM.QQField()

    Ar = _sparse_rank_case()
    println("== sparse QQ rank ==")
    println("size=", size(Ar), " nnz=", nnz(Ar),
            " auto=", FL._choose_linalg_backend(F, Ar; op=:rank))
    _bench("rank auto", () -> FL.rank(F, Ar))
    _bench("rank julia_sparse", () -> FL.rank(F, Ar; backend=:julia_sparse))
    FL._have_nemo() && _bench("rank nemo", () -> FL.rank(F, Ar; backend=:nemo))

    An = _sparse_nullspace_case()
    println()
    println("== sparse QQ nullspace ==")
    println("size=", size(An), " nnz=", nnz(An),
            " auto=", FL._choose_linalg_backend(F, An; op=:nullspace))
    _bench("nullspace auto", () -> FL.nullspace(F, An))
    _bench("nullspace julia_sparse", () -> FL.nullspace(F, An; backend=:julia_sparse))
    FL._have_nemo() && _bench("nullspace nemo", () -> FL.nullspace(F, An; backend=:nemo))
    _bench("elim summary build", () -> FL.elimination_summary(F, An))
    _bench("rank+null+col old", () -> begin
        FL.rank(F, An)
        FL.nullspace(F, An)
        FL.colspace(F, An)
    end)
    _bench("rank+null+col summary", () -> begin
        S = FL.elimination_summary(F, An)
        FL.rank(S)
        FL.nullspace(S)
        FL.colspace(S)
    end)
    _bench("kernel_image old", () -> FL._kernel_image_summary(F, An))
    _bench("kernel_image summary", () -> FL._kernel_image_summary(FL.elimination_summary(F, An)))

    Ac = _sparse_colspace_case()
    println()
    println("== sparse QQ colspace ==")
    println("size=", size(Ac), " nnz=", nnz(Ac),
            " auto=", FL._choose_linalg_backend(F, Ac; op=:colspace))
    _bench("colspace auto", () -> FL.colspace(F, Ac))
    _bench("colspace julia_sparse", () -> FL.colspace(F, Ac; backend=:julia_sparse))
    FL._have_nemo() && _bench("colspace nemo", () -> FL.colspace(F, Ac; backend=:nemo))

    Acm = _sparse_colspace_case_moderate()
    println("size=", size(Acm), " nnz=", nnz(Acm),
            " auto=", FL._choose_linalg_backend(F, Acm; op=:colspace))
    _bench("colspace mod auto", () -> FL.colspace(F, Acm))
    _bench("colspace mod julia", () -> FL.colspace(F, Acm; backend=:julia_sparse))
    FL._have_nemo() && _bench("colspace mod nemo", () -> FL.colspace(F, Acm; backend=:nemo))

    Bs, Ys = _sparse_solve_case()
    println()
    println("== sparse QQ solve_fullcolumn ==")
    println("size=", size(Bs), " nnz=", nnz(Bs),
            " auto=", FL._choose_linalg_backend(F, Bs; op=:solve))
    _bench("solve auto rhs=false", () -> FL.solve_fullcolumn(F, Bs, Ys; check_rhs=false, cache=true))
    _bench("solve julia rhs=false", () -> FL.solve_fullcolumn(F, Bs, Ys; backend=:julia_sparse, check_rhs=false, cache=true))
    FL._have_nemo() && _bench("solve nemo rhs=false", () -> FL.solve_fullcolumn(F, Bs, Ys; backend=:nemo, check_rhs=false, cache=true))
    _bench("solve auto rhs=true", () -> FL.solve_fullcolumn(F, Bs, Ys; check_rhs=true, cache=true))
    _bench("solve julia rhs=true", () -> FL.solve_fullcolumn(F, Bs, Ys; backend=:julia_sparse, check_rhs=true, cache=true))
    FL._have_nemo() && _bench("solve nemo rhs=true", () -> FL.solve_fullcolumn(F, Bs, Ys; backend=:nemo, check_rhs=true, cache=true))

    jfac = FL._factor_fullcolumnQQ(Bs)
    println("solve factor(julia) backend=", FL._choose_solve_backend(F, Bs; factor=jfac))
    _bench("solve auto factor", () -> FL.solve_fullcolumn(F, Bs, Ys; check_rhs=false, cache=false, factor=jfac))
    _bench("solve auto factor rhs=true", () -> FL.solve_fullcolumn(F, Bs, Ys; check_rhs=true, cache=false, factor=jfac))
    Xwide = [CM.QQ(i + 2j) for i in 1:size(Bs, 2), j in 1:32]
    Ywide = Bs * Xwide
    _bench("solve auto factor wide", () -> FL.solve_fullcolumn(F, Bs, Ywide; check_rhs=false, cache=false, factor=jfac))
    pfac = FL.factor_fullcolumn(F, Bs; backend=:auto, cache=false)
    println("solve factor(public) backend=", FL.factor_backend(pfac))
    _bench("factor public rank", () -> FL.rank(pfac))
    _bench("factor public nullspace", () -> FL.nullspace(pfac))
    _bench("factor public colspace", () -> FL.colspace(pfac))
    _bench("solve public factor", () -> FL.solve_fullcolumn(F, Bs, Ys; check_rhs=false, cache=false, factor=pfac))
    _bench("solve public factor rhs=true", () -> FL.solve_fullcolumn(F, Bs, Ys; check_rhs=true, cache=false, factor=pfac))
    _bench("solve public factor wide", () -> FL.solve_fullcolumn(F, Bs, Ywide; check_rhs=false, cache=false, factor=pfac))
    _bench("solve public factor via analysis", () -> begin
        ana = FL.analyze_matrix(F, Bs; fullcolumn_factor=true, cache=false)
        FL.solve_fullcolumn(F, Bs, Ys; check_rhs=false, cache=false, analysis=ana)
    end)
    _bench("solve public factor wide via analysis", () -> begin
        ana = FL.analyze_matrix(F, Bs; fullcolumn_factor=true, cache=false)
        FL.solve_fullcolumn(F, Bs, Ywide; check_rhs=false, cache=false, analysis=ana)
    end)
    if FL._have_nemo()
        nfac = FL._factor_fullcolumn_nemoQQ(Bs)
        println("solve factor(nemo) backend=", FL._choose_solve_backend(F, Bs; factor=nfac))
        _bench("solve auto factor nemo", () -> FL.solve_fullcolumn(F, Bs, Ys; check_rhs=false, cache=false, factor=nfac))
    end

    println()
    println("== sparse QQ reusable analysis ==")
    _bench("analysis build", () -> FL.analyze_matrix(F, An; cache=false))
    _bench("analysis rank+nullspace+colspace", () -> begin
        S = FL.analyze_matrix(F, An; cache=false)
        FL.rank(S) + size(FL.nullspace(S), 2) + size(FL.colspace(S), 2)
    end)
    _bench("analysis kernel_image", () -> begin
        S = FL.analyze_matrix(F, An; cache=false)
        KI = FL._kernel_image_summary(S)
        KI.rank + size(KI.ker, 2) + size(KI.img, 2)
    end)
end

main()
