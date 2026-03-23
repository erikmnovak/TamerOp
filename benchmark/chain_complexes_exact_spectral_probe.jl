using Random
using Statistics

println("load core")
flush(stdout)
module _ProbeOwner
include("../src/CoreModules.jl")
println("load linalg")
flush(stdout)
include("../src/FieldLinAlg.jl")
println("load chain")
flush(stdout)
include("../src/ChainComplexes.jl")
end
println("load done")
flush(stdout)

using ._ProbeOwner.CoreModules
using ._ProbeOwner.FieldLinAlg
using ._ProbeOwner.ChainComplexes
using SparseArrays

const CM = _ProbeOwner.CoreModules
const FL = _ProbeOwner.FieldLinAlg
const CC = _ProbeOwner.ChainComplexes

function _random_coeff(rng::AbstractRNG, field::CM.AbstractCoeffField)
    field isa CM.QQField && return CM.coerce(field, rand(rng, -5:5))
    error("unsupported field")
end

_random_nonzero_coeff(rng::AbstractRNG, field::CM.AbstractCoeffField) =
    (x = _random_coeff(rng, field); iszero(x) ? one(CM.coeff_type(field)) : x)

function _random_matrix(rng::AbstractRNG,
                        field::CM.AbstractCoeffField,
                        m::Int,
                        n::Int;
                        density::Float64)
    K = CM.coeff_type(field)
    A = zeros(K, m, n)
    @inbounds for i in 1:m, j in 1:n
        rand(rng) < density || continue
        A[i, j] = _random_nonzero_coeff(rng, field)
    end
    return A
end

function _random_cochain_differentials(rng::AbstractRNG,
                                       field::CM.AbstractCoeffField,
                                       dims::Vector{Int};
                                       density::Float64)
    K = CM.coeff_type(field)
    nd = length(dims)
    out = Vector{SparseMatrixCSC{K,Int}}()
    nd <= 1 && return out

    d0 = sparse(_random_matrix(rng, field, dims[2], dims[1]; density=density))
    push!(out, d0)

    for i in 2:(nd - 1)
        prev = Matrix(out[i - 1])
        left = FL.nullspace(field, transpose(prev))
        r = size(left, 2)
        if dims[i + 1] == 0 || dims[i] == 0 || r == 0
            push!(out, spzeros(K, dims[i + 1], dims[i]))
            continue
        end
        coeff = _random_matrix(rng, field, dims[i + 1], r; density=density)
        d = coeff * transpose(left)
        push!(out, sparse(d))
    end

    return out
end

function _random_double_complex_vertical(rng::AbstractRNG,
                                         field::CM.AbstractCoeffField;
                                         amin::Int,
                                         amax::Int,
                                         bmin::Int,
                                         bmax::Int,
                                         dmin::Int,
                                         dmax::Int,
                                         density::Float64)
    K = CM.coeff_type(field)
    na = amax - amin + 1
    nb = bmax - bmin + 1
    dims = Matrix{Int}(undef, na, nb)
    for ai in 1:na, bi in 1:nb
        dims[ai, bi] = rand(rng, dmin:dmax)
    end

    dv = Array{SparseMatrixCSC{K,Int},2}(undef, na, nb)
    dh = Array{SparseMatrixCSC{K,Int},2}(undef, na, nb)

    for ai in 1:na
        col_dims = [dims[ai, bi] for bi in 1:nb]
        col_d = _random_cochain_differentials(rng, field, col_dims; density=density)
        for bi in 1:nb
            dv[ai, bi] = bi < nb ? col_d[bi] : spzeros(K, 0, col_dims[bi])
            dh[ai, bi] = ai < na ? spzeros(K, dims[ai + 1, bi], dims[ai, bi]) : spzeros(K, 0, dims[ai, bi])
        end
    end

    return CC.DoubleComplex{K}(amin, amax, bmin, bmax, dims, dv, dh)
end

function _digest_terms(spaces)
    acc = 0
    for sq in spaces
        acc += sq.dimZ + sq.dimB + sq.dimH + size(sq.Hrep, 1) + size(sq.Hrep, 2)
    end
    return acc
end

function _digest_diff(dr)
    acc = 0
    for M in dr
        acc += size(M, 1) + size(M, 2) + nnz(M)
    end
    return acc
end

function _bench(f; reps::Int)
    times = Float64[]
    allocs = Float64[]
    for _ in 1:reps
        GC.gc()
        GC.gc()
        local t = 0.0
        bytes = @allocated begin
            t = @elapsed f()
        end
        push!(times, t * 1000)
        push!(allocs, bytes / 1024)
    end
    return (median(times), median(allocs))
end

function _fixture(; seed::Int, density::Float64, dmin::Int, dmax::Int)
    rng = MersenneTwister(seed)
    field = CM.QQField()
    return _random_double_complex_vertical(
        rng,
        field;
        amin=0,
        amax=2,
        bmin=0,
        bmax=2,
        dmin=dmin,
        dmax=dmax,
        density=density,
    )
end

function _run_probe(probe::String, DC::CC.DoubleComplex{K}) where {K}
    if probe == "vertical_build_auto"
        return () -> begin
            ss = CC.spectral_sequence(DC; output=:full, first=:vertical)
            _digest_diff(CC.differential(ss, 1))
        end
    elseif probe == "horizontal_build_auto"
        return () -> begin
            ss = CC.spectral_sequence(DC; output=:full, first=:horizontal)
            _digest_diff(CC.differential(ss, 1))
        end
    elseif probe == "horizontal_build_optimized"
        return () -> begin
            old = CC._spectral_exact_filtimg_basis_mode[]
            try
                CC._spectral_exact_filtimg_basis_mode[] = :full
                ss = CC.spectral_sequence(DC; output=:full, first=:horizontal)
                _digest_diff(CC.differential(ss, 1))
            finally
                CC._spectral_exact_filtimg_basis_mode[] = old
            end
        end
    elseif probe == "page_terms_inf_vertical"
        return () -> begin
            ss = CC.spectral_sequence(DC; output=:full, first=:vertical)
            _digest_terms(CC.page_terms(ss, :inf))
        end
    elseif probe == "page_terms_inf_vertical_prebuilt"
        ss = CC.spectral_sequence(DC; output=:full, first=:vertical)
        return () -> begin
            ss.Einf_spaces.value = nothing
            _digest_terms(CC.page_terms(ss, :inf))
        end
    elseif probe == "page_terms_inf_horizontal_prebuilt"
        ss = CC.spectral_sequence(DC; output=:full, first=:horizontal)
        return () -> begin
            ss.Einf_spaces.value = nothing
            _digest_terms(CC.page_terms(ss, :inf))
        end
    elseif probe == "filtration_basis_mid_vertical"
        return () -> begin
            ss = CC.spectral_sequence(DC; output=:full, first=:vertical)
            tmid = ss.Tot.tmin + div(ss.Tot.tmax - ss.Tot.tmin, 2)
            B = CC.filtration_basis(ss, ss.pmin, tmid)
            size(B, 1) + size(B, 2)
        end
    else
        error("unknown probe '$probe'")
    end
end

function main(; probe::String,
              reps::Int=3,
              seed::Int=11,
              density::Float64=0.35,
              dmin::Int=6,
              dmax::Int=10)
    println("build fixture")
    flush(stdout)
    DC = _fixture(seed=seed, density=density, dmin=dmin, dmax=dmax)
    println("run probe=", probe)
    flush(stdout)
    ms, kib = _bench(_run_probe(probe, DC); reps=reps)
    println("RESULT probe=", probe, " median_ms=", ms, " median_kib=", kib)
    flush(stdout)
end

args = Dict{String,String}()
for a in ARGS
    if startswith(a, "--") && occursin("=", a)
        k, v = split(a[3:end], "=", limit=2)
        args[k] = v
    end
end

main(
    probe=get(args, "probe", "horizontal_build_auto"),
    reps=parse(Int, get(args, "reps", "3")),
    seed=parse(Int, get(args, "seed", "11")),
    density=parse(Float64, get(args, "density", "0.35")),
    dmin=parse(Int, get(args, "dmin", "6")),
    dmax=parse(Int, get(args, "dmax", "10")),
)
