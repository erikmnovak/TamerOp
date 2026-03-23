#!/usr/bin/env julia
#
# field_linalg_sparse_rref_microbench.jl
#
# Focused benchmark for sparse exact elimination primitives.
#
# Kernel mapping:
# - exercises `src/field_linalg/sparse_rref.jl` directly
# - intended for REF/RREF row arithmetic and streaming sparse elimination work
# - not intended to measure backend-routing crossover policy

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
const CC = TamerOp.ChainComplexes

function _parse_int_arg(args, key::String, default::Int)
    for a in args
        startswith(a, key * "=") || continue
        return parse(Int, split(a, "=", limit=2)[2])
    end
    return default
end

function _bench(name::AbstractString, f::Function; reps::Int=5)
    GC.gc()
    f()
    GC.gc()
    times_ms = Vector{Float64}(undef, reps)
    bytes = Vector{Int}(undef, reps)
    for i in 1:reps
        t = @timed f()
        times_ms[i] = 1000 * t.time
        bytes[i] = t.bytes
    end
    sort!(times_ms)
    sort!(bytes)
    med_ms = times_ms[cld(reps, 2)]
    med_kib = bytes[cld(reps, 2)] / 1024
    println(rpad(name, 44), " median_time=", round(med_ms, digits=3),
            " ms  median_alloc=", round(med_kib, digits=1), " KiB")
    return (ms=med_ms, kib=med_kib)
end

function _rand_sparse_qq_matrix(m::Int, n::Int, density::Float64, rng::AbstractRNG)
    nnz_target = max(1, round(Int, m * n * density))
    I = Vector{Int}(undef, nnz_target)
    J = Vector{Int}(undef, nnz_target)
    V = Vector{QQ}(undef, nnz_target)
    @inbounds for t in 1:nnz_target
        I[t] = rand(rng, 1:m)
        J[t] = rand(rng, 1:n)
        v = rand(rng, -3:3)
        while v == 0
            v = rand(rng, -3:3)
        end
        V[t] = QQ(v)
    end
    return sparse(I, J, V, m, n)
end

function _sparserows(A::SparseMatrixCSC{QQ,Int})
    m, n = size(A)
    rows = Vector{FL.SparseRow{QQ}}(undef, m)
    row_idx = [Int[] for _ in 1:m]
    row_val = [QQ[] for _ in 1:m]
    @inbounds for j in 1:n
        for ptr in nzrange(A, j)
            i = rowvals(A)[ptr]
            push!(row_idx[i], j)
            push!(row_val[i], nonzeros(A)[ptr])
        end
    end
    @inbounds for i in 1:m
        rows[i] = FL.SparseRow{QQ}(row_idx[i], row_val[i])
    end
    return rows
end

function _cochain_fixture(; ndeg::Int=4, cdim_min::Int=8, cdim_max::Int=12, density::Float64=0.30, seed::UInt64=UInt64(0x43434f48))
    rng = Random.MersenneTwister(seed)
    dims = [rand(rng, cdim_min:cdim_max) for _ in 1:(ndeg + 1)]
    d = Vector{SparseMatrixCSC{QQ,Int}}(undef, ndeg)
    for i in 1:ndeg
        d[i] = _rand_sparse_qq_matrix(dims[i + 1], dims[i], density, rng)
    end
    return CC.CochainComplex{QQ}(0, ndeg, dims, d)
end

function main(args=ARGS)
    reps = _parse_int_arg(args, "--reps", 5)
    m = _parse_int_arg(args, "--m", 64)
    n = _parse_int_arg(args, "--n", 64)
    density = parse(Float64, get(ENV, "SPARSE_RREF_DENSITY", "0.18"))
    rng = Random.MersenneTwister(UInt64(0x53505252))
    A = _rand_sparse_qq_matrix(m, n, density, rng)
    rows = _sparserows(A)
    C = _cochain_fixture(; density=density)

    println("FieldLinAlg sparse QQ RREF micro-benchmark")
    println("reps=$(reps), size=$(m)x$(n), density=$(density)\n")

    _bench("sparse_rref elimination", () -> begin
        R = FL._SparseRREF{QQ}(n)
        for row in rows
            FL._sparse_rref_push_homogeneous!(R, copy(row))
        end
        FL._rref_rank(R)
    end; reps=reps)

    _bench("kernel_image_summary sparse", () -> FL._kernel_image_summary(CM.QQField(), A); reps=reps)

    _bench("chain diff_summaries", () -> CC.cohomology_dims(C); reps=reps)
end

main()
