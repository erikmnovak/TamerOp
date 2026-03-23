using Random
using Printf
using SparseArrays

include(joinpath(@__DIR__, "..", "src", "DataTypes.jl"))
const DT = DataTypes

function _parse_flag(args::Vector{String}, key::String, default)
    prefix = key * "="
    for arg in args
        startswith(arg, prefix) || continue
        value = split(arg, "=", limit=2)[2]
        return default isa Int ? parse(Int, value) : value
    end
    return default
end

function _median_stats(f::Function; reps::Int)
    times = Vector{Float64}(undef, reps)
    allocs = Vector{Int}(undef, reps)
    for i in 1:reps
        stats = @timed f()
        times[i] = stats.time
        allocs[i] = stats.bytes
    end
    p = sortperm(times)
    mid = p[cld(reps, 2)]
    return (time=times[mid], bytes=allocs[mid])
end

function _complex_fixture(nv::Int, ne::Int; seed::Int=Int(0xD774))
    rng = MersenneTwister(seed)
    cells = [collect(1:nv), collect(1:ne)]
    I = Vector{Int}(undef, 2 * ne)
    J = Vector{Int}(undef, 2 * ne)
    V = Vector{Int}(undef, 2 * ne)
    t = 1
    for e in 1:ne
        u = rand(rng, 1:nv)
        v = rand(rng, 1:nv)
        u == v && (v = (v % nv) + 1)
        I[t] = u
        J[t] = e
        V[t] = -1
        t += 1
        I[t] = v
        J[t] = e
        V[t] = 1
        t += 1
    end
    boundaries = [sparse(I, J, V, nv, ne)]
    grades = [(rand(rng),) for _ in 1:(nv + ne)]
    multigrades = Vector{Vector{NTuple{1,Float64}}}(undef, length(grades))
    for i in eachindex(grades)
        g = grades[i]
        if isodd(i)
            multigrades[i] = [g]
        else
            multigrades[i] = [g, (g[1] + 0.25,)]
        end
    end
    return (; cells, boundaries, grades, multigrades)
end

function _old_cell_dims_from_cells(cells_by_dim)
    out = Int[]
    for (d, cells) in enumerate(cells_by_dim)
        for _ in cells
            push!(out, d - 1)
        end
    end
    return out
end

function _emulate_old_graded_complex_ctor(cells_by_dim, boundaries, grades)
    total = sum(length, cells_by_dim)
    cell_dims = length(grades) == total ? _old_cell_dims_from_cells(cells_by_dim) : fill(0, length(grades))
    N = length(grades[1])
    T = eltype(grades[1])
    ng = Vector{NTuple{N,T}}(undef, length(grades))
    for i in eachindex(grades)
        ng[i] = ntuple(j -> T(grades[i][j]), N)
    end
    return (cells_by_dim, boundaries, ng, cell_dims)
end

function _emulate_old_multicritical_ctor(cells_by_dim, boundaries, grades)
    total = sum(length, cells_by_dim)
    length(grades) == total || error("multicritical benchmark fixture mismatch")
    cell_dims = _old_cell_dims_from_cells(cells_by_dim)
    first_cell = findfirst(!isempty, grades)
    N = length(grades[first_cell][1])
    T = eltype(grades[first_cell][1])
    ng = Vector{Vector{NTuple{N,T}}}(undef, length(grades))
    for i in eachindex(grades)
        gi = grades[i]
        out = Vector{NTuple{N,T}}(undef, length(gi))
        for j in eachindex(gi)
            out[j] = ntuple(k -> T(gi[j][k]), N)
        end
        ng[i] = unique(out)
    end
    return (cells_by_dim, boundaries, ng, cell_dims)
end

function main(args)
    reps = _parse_flag(args, "--reps", 7)
    nv = _parse_flag(args, "--nv", 4096)
    ne = _parse_flag(args, "--ne", 4096)
    out = _parse_flag(args, "--out", joinpath(@__DIR__, "_tmp_data_types_complex_microbench.csv"))

    println("Timing policy: warm, same-process A/B baseline")
    println("reps=$reps, nv=$nv, ne=$ne")

    fixture = _complex_fixture(nv, ne)
    cases = [
        ("graded_complex_ctor_packed_metadata", () -> begin
            before = _median_stats(() -> _emulate_old_graded_complex_ctor(fixture.cells, fixture.boundaries, fixture.grades); reps=reps)
            after = _median_stats(() -> DT.GradedComplex(fixture.cells, fixture.boundaries, fixture.grades); reps=reps)
            return before, after
        end),
        ("multicritical_ctor_packed_metadata", () -> begin
            before = _median_stats(() -> _emulate_old_multicritical_ctor(fixture.cells, fixture.boundaries, fixture.multigrades); reps=reps)
            after = _median_stats(() -> DT.MultiCriticalGradedComplex(fixture.cells, fixture.boundaries, fixture.multigrades); reps=reps)
            return before, after
        end),
    ]

    rows = String[]
    push!(rows, "case,variant,time_ms,alloc_kib")
    for (label, run_case) in cases
        before, after = run_case()
        push!(rows, @sprintf("%s,before,%.6f,%.3f", label, 1.0e3 * before.time, before.bytes / 1024))
        push!(rows, @sprintf("%s,after,%.6f,%.3f", label, 1.0e3 * after.time, after.bytes / 1024))
        ratio = before.time / max(after.time, eps())
        println(@sprintf("%-32s  %.3f ms -> %.3f ms  (%.2fx),  %.1f KiB -> %.1f KiB",
            label, 1.0e3 * before.time, 1.0e3 * after.time, ratio,
            before.bytes / 1024, after.bytes / 1024))
    end

    open(out, "w") do io
        write(io, join(rows, "\n"))
        write(io, "\n")
    end
    println("wrote ", out)
end

main(ARGS)
