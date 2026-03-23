#!/usr/bin/env julia
#
# linalg_backend_microbench.jl
#
# Purpose
# - Broad performance benchmark for the FieldLinAlg backend stack.
# - Compare operations across fields, matrix sizes, densities, and backend modes.
# - Report speedups against naive and materialized-dense baselines where meaningful.
#
# Operations covered
# - rank
# - nullspace
# - solve_fullcolumn
#
# Fields covered (default)
# - QQ
# - F2
# - F3
# - Fp(5)
# - Real(Float64)
#
# Baselines
# - Naive dense elimination rank baseline (small matrices only).
# - Naive dense full-column solve baseline (small matrices only).
# - Materialized-dense baseline for sparse cases:
#   run the same op on Matrix(A) / Matrix(B) to estimate old densify-style cost.
#
# Usage
#   julia --project=. benchmark/linalg_backend_microbench.jl
#   julia --project=. benchmark/linalg_backend_microbench.jl --profile=full --reps=7
#   julia --project=. benchmark/linalg_backend_microbench.jl --fields=qq,f2,f3,fp5,real --ops=rank,nullspace,solve
#

using Random
using SparseArrays
using LinearAlgebra

try
    using TamerOp
catch
    include(joinpath(@__DIR__, "..", "src", "TamerOp.jl"))
    using .TamerOp
end

const CM = TamerOp.CoreModules
const FL = TamerOp.FieldLinAlg

@inline function _parse_int_arg(args, key::String, default::Int)
    for a in args
        startswith(a, key * "=") || continue
        return max(1, parse(Int, split(a, "=", limit=2)[2]))
    end
    return default
end

@inline function _parse_str_arg(args, key::String, default::String)
    for a in args
        startswith(a, key * "=") || continue
        return split(a, "=", limit=2)[2]
    end
    return default
end

@inline function _parse_bool_arg(args, key::String, default::Bool)
    for a in args
        startswith(a, key * "=") || continue
        v = lowercase(split(a, "=", limit=2)[2])
        if v in ("1", "true", "yes", "y", "on")
            return true
        elseif v in ("0", "false", "no", "n", "off")
            return false
        else
            error("Invalid boolean for $key: $v")
        end
    end
    return default
end

function _bench(name::AbstractString, f::Function; reps::Int=5)
    GC.gc()
    f() # warmup
    GC.gc()
    times_ms = Vector{Float64}(undef, reps)
    bytes = Vector{Int}(undef, reps)
    for i in 1:reps
        t = @timed f()
        times_ms[i] = 1000.0 * t.time
        bytes[i] = t.bytes
    end
    sort!(times_ms)
    sort!(bytes)
    med_ms = times_ms[cld(reps, 2)]
    med_kib = bytes[cld(reps, 2)] / 1024.0
    println(rpad(name, 58), " median_time=", round(med_ms, digits=3),
            " ms  median_alloc=", round(med_kib, digits=1), " KiB")
    return (ms=med_ms, kib=med_kib)
end

@inline function _is_zero_val(field::CM.AbstractCoeffField, x)
    if field isa CM.RealField
        tol = field.atol + field.rtol * max(1.0, abs(Float64(x)))
        return abs(Float64(x)) <= tol
    end
    return x == zero(x)
end

@inline function _rand_int_nonzero(rng::AbstractRNG, lo::Int=-3, hi::Int=3)
    v = rand(rng, lo:hi)
    while v == 0
        v = rand(rng, lo:hi)
    end
    return v
end

@inline function _rand_scalar(field::CM.AbstractCoeffField, rng::AbstractRNG)
    if field isa CM.RealField
        return randn(rng)
    end
    return CM.coerce(field, rand(rng, -3:3))
end

@inline function _rand_nonzero_scalar(field::CM.AbstractCoeffField, rng::AbstractRNG)
    if field isa CM.RealField
        v = randn(rng)
        while abs(v) < 1e-9
            v = randn(rng)
        end
        return v
    end
    return CM.coerce(field, _rand_int_nonzero(rng))
end

function _rand_dense_matrix(field::CM.AbstractCoeffField, m::Int, n::Int, rng::AbstractRNG)
    K = CM.coeff_type(field)
    A = Matrix{K}(undef, m, n)
    @inbounds for i in 1:m, j in 1:n
        A[i, j] = _rand_scalar(field, rng)
    end
    return A
end

function _rand_sparse_matrix(field::CM.AbstractCoeffField, m::Int, n::Int, density::Float64, rng::AbstractRNG)
    K = CM.coeff_type(field)
    nnz_target = max(1, round(Int, m * n * density))
    I = Vector{Int}(undef, nnz_target)
    J = Vector{Int}(undef, nnz_target)
    V = Vector{K}(undef, nnz_target)
    @inbounds for t in 1:nnz_target
        I[t] = rand(rng, 1:m)
        J[t] = rand(rng, 1:n)
        V[t] = _rand_nonzero_scalar(field, rng)
    end
    return sparse(I, J, V, m, n)
end

function _rand_matrix(field::CM.AbstractCoeffField, m::Int, n::Int, density::Float64, rng::AbstractRNG)
    if density >= 0.999
        return _rand_dense_matrix(field, m, n, rng)
    end
    return _rand_sparse_matrix(field, m, n, density, rng)
end

function _full_column_matrix(field::CM.AbstractCoeffField, m::Int, n::Int, density::Float64, rng::AbstractRNG)
    m >= n || error("_full_column_matrix requires m >= n")
    K = CM.coeff_type(field)
    if density >= 0.999
        B = zeros(K, m, n)
        @inbounds for i in 1:n
            B[i, i] = one(K)
        end
        @inbounds for i in (n + 1):m, j in 1:n
            B[i, j] = _rand_scalar(field, rng)
        end
        return B
    else
        I = Int[]
        J = Int[]
        V = K[]
        sizehint!(I, n + round(Int, (m - n) * n * density))
        sizehint!(J, length(I))
        sizehint!(V, length(I))
        @inbounds for i in 1:n
            push!(I, i); push!(J, i); push!(V, one(K))
        end
        @inbounds for i in (n + 1):m, j in 1:n
            if rand(rng) < density
                push!(I, i); push!(J, j); push!(V, _rand_nonzero_scalar(field, rng))
            end
        end
        return sparse(I, J, V, m, n)
    end
end

function _naive_rank_dense(field::CM.AbstractCoeffField, A::Matrix)
    M = copy(A)
    m, n = size(M)
    r = 1
    rk = 0
    for c in 1:n
        r > m && break
        piv = 0
        @inbounds for i in r:m
            if !_is_zero_val(field, M[i, c])
                piv = i
                break
            end
        end
        piv == 0 && continue
        if piv != r
            M[r, :], M[piv, :] = M[piv, :], M[r, :]
        end
        pivv = M[r, c]
        @inbounds for j in c:n
            M[r, j] /= pivv
        end
        @inbounds for i in (r + 1):m
            fac = M[i, c]
            if !_is_zero_val(field, fac)
                for j in c:n
                    M[i, j] -= fac * M[r, j]
                end
            end
        end
        rk += 1
        r += 1
    end
    return rk
end

function _naive_solve_fullcolumn_dense(field::CM.AbstractCoeffField, B::Matrix, Y::Matrix)
    m, n = size(B)
    mY, rhs = size(Y)
    m == mY || error("B and Y row mismatch")
    A = hcat(copy(B), copy(Y))
    row = 1
    pivrow = fill(0, n)
    total_cols = n + rhs
    for c in 1:n
        row > m && error("rank-deficient B in naive solve")
        piv = 0
        @inbounds for i in row:m
            if !_is_zero_val(field, A[i, c])
                piv = i
                break
            end
        end
        piv == 0 && error("rank-deficient B in naive solve")
        if piv != row
            A[row, :], A[piv, :] = A[piv, :], A[row, :]
        end
        pv = A[row, c]
        @inbounds for j in c:total_cols
            A[row, j] /= pv
        end
        @inbounds for i in 1:m
            i == row && continue
            fac = A[i, c]
            if !_is_zero_val(field, fac)
                for j in c:total_cols
                    A[i, j] -= fac * A[row, j]
                end
            end
        end
        pivrow[c] = row
        row += 1
    end
    X = Matrix{eltype(B)}(undef, n, rhs)
    @inbounds for c in 1:n
        pr = pivrow[c]
        for j in 1:rhs
            X[c, j] = A[pr, n + j]
        end
    end
    return X
end

function _field_specs(tokens::AbstractVector{<:AbstractString})
    out = Tuple{String,CM.AbstractCoeffField}[]
    for t in tokens
        if t == "qq"
            push!(out, ("qq", CM.QQField()))
        elseif t == "f2"
            push!(out, ("f2", CM.F2()))
        elseif t == "f3"
            push!(out, ("f3", CM.F3()))
        elseif t == "fp5"
            push!(out, ("fp5", CM.Fp(5)))
        elseif t == "real"
            push!(out, ("real", CM.RealField(Float64; rtol=1e-10, atol=1e-12)))
        else
            error("unknown field token: $t")
        end
    end
    return out
end

function _backend_candidates(field::CM.AbstractCoeffField, op::Symbol, sparse_case::Bool)
    if field isa CM.QQField
        return [:auto, :julia_exact, :nemo]
    elseif field isa CM.PrimeField
        if field.p == 2
            return [:auto, :f2_bit, :julia_exact]
        elseif field.p == 3
            return [:auto, :f3_table, :julia_exact]
        else
            if sparse_case
                return [:auto, :fp_sparse, :julia_exact, :nemo]
            end
            return [:auto, :julia_exact, :nemo]
        end
    elseif field isa CM.RealField
        if sparse_case
            if op == :nullspace
                return [:auto, :float_sparse_qr, :float_sparse_svds]
            elseif op == :solve
                return [:auto, :float_sparse_qr]
            end
            return [:auto, :float_sparse_qr]
        end
        if op == :nullspace
            return [:auto, :float_dense_qr, :float_dense_svd]
        elseif op == :solve
            return [:auto, :float_dense_qr]
        end
        return [:auto, :float_dense_qr]
    end
    return [:auto]
end

function _size_grid(profile::Symbol, fname::String)
    if profile == :quick
        if fname == "qq"
            return [40, 72]
        elseif fname == "real"
            return [64, 128]
        else
            return [64, 160]
        end
    else
        if fname == "qq"
            return [40, 72, 112]
        elseif fname == "real"
            return [64, 128, 256, 384]
        else
            return [64, 160, 320, 512]
        end
    end
end

function _density_grid(profile::Symbol)
    return profile == :quick ? [1.0, 0.2, 0.05] : [1.0, 0.3, 0.1, 0.03, 0.01]
end

function _run_rank_case(field, A, backend::Symbol)
    return FL.rank(field, A; backend=backend)
end

function _run_nullspace_case(field, A, backend::Symbol)
    N = FL.nullspace(field, A; backend=backend)
    return size(N, 2)
end

function _run_solve_case(field, B, Y, backend::Symbol)
    X = FL.solve_fullcolumn(field, B, Y; backend=backend, cache=true)
    return size(X, 2)
end

function _try_bench(name::String, f::Function; reps::Int)
    try
        b = _bench(name, f; reps=reps)
        return (ok=true, ms=b.ms, kib=b.kib, err="")
    catch err
        println(rpad(name, 58), " skipped: ", sprint(showerror, err))
        return (ok=false, ms=NaN, kib=NaN, err=sprint(showerror, err))
    end
end

function main(args=ARGS)
    reps = _parse_int_arg(args, "--reps", 5)
    profile_str = lowercase(_parse_str_arg(args, "--profile", "quick"))
    profile = profile_str == "full" ? :full : :quick
    fields_str = lowercase(_parse_str_arg(args, "--fields", "qq,f2,f3,fp5,real"))
    ops_str = lowercase(_parse_str_arg(args, "--ops", "rank,nullspace,solve"))
    run_naive = _parse_bool_arg(args, "--naive", true)
    seed = _parse_int_arg(args, "--seed", 271828)

    field_specs = _field_specs(split(fields_str, ","))
    ops = Symbol[]
    for t in split(ops_str, ",")
        t == "rank" && push!(ops, :rank)
        t == "nullspace" && push!(ops, :nullspace)
        t == "solve" && push!(ops, :solve)
    end
    isempty(ops) && error("No valid ops selected.")

    println("FieldLinAlg backend microbench")
    println("profile=$(profile), reps=$(reps), fields=$(fields_str), ops=$(ops_str), naive=$(run_naive)")
    println("fingerprint=", FL._current_linalg_fingerprint())
    println("thresholds=", FL._current_linalg_thresholds())
    println()

    rng = Random.MersenneTwister(seed)
    for (fname, field) in field_specs
        println("=== field: ", fname, " ===")
        sizes = _size_grid(profile, fname)
        densities = _density_grid(profile)
        for n in sizes
            for dens in densities
                sparse_case = dens < 0.999
                m_rank = n
                n_rank = n
                m_null = max(8, div(n, 2))
                n_null = n
                m_solv = n + max(4, div(n, 3))
                n_solv = n
                rhs = min(8, max(2, div(n, 16)))
                println("  n=$(n) density=$(dens) sparse=$(sparse_case)")

                if :rank in ops
                    A = _rand_matrix(field, m_rank, n_rank, dens, rng)
                    naive_rank_ms = NaN
                    if run_naive && n <= 96
                        Ad = Matrix(A)
                        bna = _bench("    rank naive_dense", () -> _naive_rank_dense(field, Ad); reps=reps)
                        naive_rank_ms = bna.ms
                    end
                    if sparse_case
                        bmat = _try_bench("    rank materialized_dense", () -> FL.rank(field, Matrix(A); backend=:julia_exact); reps=reps)
                        if bmat.ok
                            println("      materialized dense baseline: ", round(bmat.ms, digits=3), " ms")
                        end
                    end

                    for be in _backend_candidates(field, :rank, sparse_case)
                        b = _try_bench("    rank backend=$(be)", () -> _run_rank_case(field, A, be); reps=reps)
                        if b.ok && isfinite(naive_rank_ms)
                            println("      speedup vs naive: ", round(naive_rank_ms / b.ms, digits=3), "x")
                        end
                    end
                end

                if :nullspace in ops
                    A = _rand_matrix(field, m_null, n_null, dens, rng)
                    if sparse_case
                        bmat = _try_bench("    nullspace materialized_dense", () -> FL.nullspace(field, Matrix(A); backend=:julia_exact); reps=reps)
                        bmat.ok || nothing
                    end
                    for be in _backend_candidates(field, :nullspace, sparse_case)
                        _try_bench("    nullspace backend=$(be)", () -> _run_nullspace_case(field, A, be); reps=reps)
                    end
                end

                if :solve in ops
                    B = _full_column_matrix(field, m_solv, n_solv, dens, rng)
                    Xtrue = _rand_dense_matrix(field, n_solv, rhs, rng)
                    Y = B * Xtrue
                    naive_solve_ms = NaN
                    if run_naive && n <= 80
                        Bd = Matrix(B)
                        Yd = Matrix(Y)
                        bna = _bench("    solve naive_dense", () -> _naive_solve_fullcolumn_dense(field, Bd, Yd); reps=reps)
                        naive_solve_ms = bna.ms
                    end
                    if sparse_case
                        _try_bench("    solve materialized_dense", () -> FL.solve_fullcolumn(field, Matrix(B), Matrix(Y); backend=:julia_exact, cache=true); reps=reps)
                    end
                    for be in _backend_candidates(field, :solve, sparse_case)
                        b = _try_bench("    solve backend=$(be)", () -> _run_solve_case(field, B, Y, be); reps=reps)
                        if b.ok && isfinite(naive_solve_ms)
                            println("      speedup vs naive: ", round(naive_solve_ms / b.ms, digits=3), "x")
                        end
                    end
                end
            end
        end
        println()
    end
end

main()
