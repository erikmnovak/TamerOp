#!/usr/bin/env julia
#
# chain_complexes_microbench.jl
#
# Purpose
# - Benchmark hot kernels in `src/ChainComplexes.jl` using deterministic synthetic fixtures.
# - Cover linear solves, cohomology/homology kernels, mapping-cone workflows,
#   and spectral-sequence construction/query paths.
#
# Usage
#   julia --project=. benchmark/chain_complexes_microbench.jl
#   julia --project=. benchmark/chain_complexes_microbench.jl --reps=7 --field=qq --verify=true
#

using LinearAlgebra
using Random
using SparseArrays

const TO = let pm = nothing
    try
        @eval using TamerOp
        pm = TamerOp
    catch
        try
            include(joinpath(@__DIR__, "..", "src", "TamerOp.jl"))
            pm = TamerOp
        catch
            @eval module _ChainComplexesBenchModules
                include(joinpath(@__DIR__, "..", "src", "CoreModules.jl"))
                include(joinpath(@__DIR__, "..", "src", "FieldLinAlg.jl"))
                include(joinpath(@__DIR__, "..", "src", "ChainComplexes.jl"))
            end
            pm = _ChainComplexesBenchModules
        end
    end
    pm
end

const CM = TO.CoreModules
const CC = TO.ChainComplexes
const FL = TO.FieldLinAlg

function _parse_int_arg(args, key::String, default::Int)
    for a in args
        startswith(a, key * "=") || continue
        return parse(Int, split(a, "=", limit=2)[2])
    end
    return default
end

function _parse_float_arg(args, key::String, default::Float64)
    for a in args
        startswith(a, key * "=") || continue
        return parse(Float64, split(a, "=", limit=2)[2])
    end
    return default
end

function _parse_string_arg(args, key::String, default::String)
    for a in args
        startswith(a, key * "=") || continue
        return lowercase(strip(split(a, "=", limit=2)[2]))
    end
    return default
end

function _parse_path_arg(args, key::String, default::String)
    for a in args
        startswith(a, key * "=") || continue
        return String(strip(split(a, "=", limit=2)[2]))
    end
    return default
end

function _parse_bool_arg(args, key::String, default::Bool)
    for a in args
        startswith(a, key * "=") || continue
        v = lowercase(strip(split(a, "=", limit=2)[2]))
        if v in ("1", "true", "yes", "y", "on")
            return true
        elseif v in ("0", "false", "no", "n", "off")
            return false
        else
            error("invalid boolean for $key: $v")
        end
    end
    return default
end

function _profile_defaults(profile::String)
    if profile == "default"
        return (
            density=0.30,
            ndeg=5,
            cdim_min=14,
            cdim_max=28,
            nrhs=6,
            solve_m=80,
            solve_n=64,
            solve_density=0.18,
            dc_na=4,
            dc_nb=4,
            dc_dim_min=8,
            dc_dim_max=16,
        )
    elseif profile == "spectral_large"
        return (
            density=0.35,
            ndeg=5,
            cdim_min=14,
            cdim_max=28,
            nrhs=6,
            solve_m=80,
            solve_n=64,
            solve_density=0.18,
            dc_na=6,
            dc_nb=6,
            dc_dim_min=14,
            dc_dim_max=26,
        )
    elseif profile == "qq_exact"
        return (
            density=0.30,
            ndeg=4,
            cdim_min=8,
            cdim_max=12,
            nrhs=6,
            solve_m=48,
            solve_n=36,
            solve_density=0.18,
            dc_na=4,
            dc_nb=3,
            dc_dim_min=6,
            dc_dim_max=10,
        )
    elseif profile == "qq_exact_les"
        return (
            density=0.30,
            ndeg=4,
            cdim_min=8,
            cdim_max=12,
            nrhs=6,
            solve_m=48,
            solve_n=36,
            solve_density=0.18,
            dc_na=4,
            dc_nb=3,
            dc_dim_min=6,
            dc_dim_max=10,
        )
    elseif profile == "qq_exact_spectral"
        return (
            density=0.30,
            ndeg=4,
            cdim_min=8,
            cdim_max=12,
            nrhs=4,
            solve_m=32,
            solve_n=24,
            solve_density=0.18,
            dc_na=3,
            dc_nb=2,
            dc_dim_min=2,
            dc_dim_max=4,
        )
    end
    error("unknown profile '$profile' (supported: default,spectral_large,qq_exact,qq_exact_les,qq_exact_spectral)")
end

@inline function _fmt_bench_value(x::Float64)
    if x >= 1.0
        return string(round(x, digits=3))
    elseif x >= 0.01
        return string(round(x, digits=4))
    else
        return string(round(x, digits=6))
    end
end

function _bench(name::AbstractString,
                f::Function;
                reps::Int=7,
                setup::Union{Nothing,Function}=nothing,
                inner_reps::Int=1)
    inner_reps >= 1 || error("_bench: inner_reps must be >= 1")

    run_once = inner_reps == 1 ? f : () -> begin
        acc = 0
        for _ in 1:inner_reps
            acc += f()
        end
        acc
    end

    GC.gc()
    run_once() # warmup
    GC.gc()
    times_ms = Vector{Float64}(undef, reps)
    bytes = Vector{Int}(undef, reps)
    for i in 1:reps
        setup === nothing || setup()
        m = @timed run_once()
        times_ms[i] = 1000.0 * m.time / inner_reps
        bytes[i] = m.bytes
    end
    sort!(times_ms)
    sort!(bytes)
    med_ms = times_ms[cld(reps, 2)]
    med_kib = bytes[cld(reps, 2)] / (1024.0 * inner_reps)
    println(rpad(name, 48), " median_time=", _fmt_bench_value(med_ms),
            " ms  median_alloc=", _fmt_bench_value(med_kib), " KiB")
    return (probe=String(name), median_ms=med_ms, median_kib=med_kib)
end

function _field_from_name(name::String)
    name == "qq" && return CM.QQField()
    name == "f2" && return CM.F2()
    name == "f3" && return CM.F3()
    name == "f5" && return CM.Fp(5)
    name == "real" && return CM.RealField()
    error("unknown field '$name' (supported: qq,f2,f3,f5,real)")
end

function _random_coeff(rng::AbstractRNG, field::CM.AbstractCoeffField)
    if field isa CM.RealField
        return randn(rng)
    end
    v = rand(rng, -3:3)
    return CM.coerce(field, v)
end

function _random_nonzero_coeff(rng::AbstractRNG, field::CM.AbstractCoeffField)
    if field isa CM.RealField
        x = randn(rng)
        abs(x) < 1e-12 && return 1.0
        return x
    end
    while true
        v = rand(rng, -3:3)
        v == 0 && continue
        return CM.coerce(field, v)
    end
end

function _random_matrix(rng::AbstractRNG, field::CM.AbstractCoeffField, m::Int, n::Int; density::Float64)
    K = CM.coeff_type(field)
    A = zeros(K, m, n)
    @inbounds for i in 1:m, j in 1:n
        rand(rng) < density || continue
        A[i, j] = _random_nonzero_coeff(rng, field)
    end
    return A
end

function _random_rhs(rng::AbstractRNG, field::CM.AbstractCoeffField, n::Int, nrhs::Int)
    K = CM.coeff_type(field)
    X = zeros(K, n, nrhs)
    @inbounds for i in 1:n, j in 1:nrhs
        X[i, j] = _random_coeff(rng, field)
    end
    return X
end

function _random_dims(rng::AbstractRNG, n::Int; dmin::Int, dmax::Int)
    n >= 1 || error("_random_dims: require n >= 1")
    (dmin >= 0 && dmax >= dmin) || error("_random_dims: invalid dim bounds")
    return [rand(rng, dmin:dmax) for _ in 1:n]
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

function _random_cochain_complex(rng::AbstractRNG,
                                 field::CM.AbstractCoeffField;
                                 tmin::Int,
                                 dims::Vector{Int},
                                 density::Float64)
    d = _random_cochain_differentials(rng, field, dims; density=density)
    tmax = tmin + length(dims) - 1
    return CC.CochainComplex{CM.coeff_type(field)}(tmin, tmax, dims, d)
end

function _sparse_eye(::Type{K}, n::Int) where {K}
    n == 0 && return spzeros(K, 0, 0)
    return sparse(1:n, 1:n, fill(one(K), n), n, n)
end

function _identity_cochain_map(C::CC.CochainComplex{K}) where {K}
    maps = Vector{SparseMatrixCSC{K,Int}}(undef, C.tmax - C.tmin + 1)
    for t in C.tmin:C.tmax
        idx = t - C.tmin + 1
        maps[idx] = _sparse_eye(K, C.dims[idx])
    end
    return CC.CochainMap(C, C, maps; tmin=C.tmin, tmax=C.tmax, check=true)
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
    (na >= 1 && nb >= 1) || error("_random_double_complex_vertical: invalid ranges")

    dims = Matrix{Int}(undef, na, nb)
    for ai in 1:na, bi in 1:nb
        dims[ai, bi] = rand(rng, dmin:dmax)
    end

    dv = Array{SparseMatrixCSC{K,Int},2}(undef, na, nb)
    dh = Array{SparseMatrixCSC{K,Int},2}(undef, na, nb)

    # Nontrivial vertical maps; zero horizontal maps.
    for ai in 1:na
        col_dims = [dims[ai, bi] for bi in 1:nb]
        col_d = _random_cochain_differentials(rng, field, col_dims; density=density)
        for bi in 1:nb
            if bi < nb
                dv[ai, bi] = col_d[bi]
            else
                dv[ai, bi] = spzeros(K, 0, col_dims[bi])
            end

            if ai < na
                dh[ai, bi] = spzeros(K, dims[ai + 1, bi], dims[ai, bi])
            else
                dh[ai, bi] = spzeros(K, 0, dims[ai, bi])
            end
        end
    end

    return CC.DoubleComplex{K}(amin, amax, bmin, bmax, dims, dv, dh)
end

function _check_d2_zero(C::CC.CochainComplex{K}) where {K}
    for i in 1:(length(C.d) - 1)
        D = C.d[i + 1] * C.d[i]
        dropzeros!(D)
        nnz(D) == 0 || error("fixture check failed: d^(i+1) * d^i != 0 at index $i")
    end
    return true
end

@inline _nnz_sum(ms::AbstractVector{<:SparseMatrixCSC}) = sum(nnz, ms)
@inline _digest_cc(C::CC.CochainComplex) = sum(C.dims) + _nnz_sum(C.d)
@inline _digest_H(H::CC.CohomologyData) = H.dimC + H.dimZ + H.dimB + H.dimH + size(H.Hrep, 1) + size(H.Hrep, 2)
@inline _digest_Hall(Hs) = sum(h -> h.dimH + h.dimZ + h.dimB, Hs)
@inline _digest_hom(H::CC.HomologyData) = H.dimC + H.dimZ + H.dimB + H.dimH
@inline _digest_tri(tri::CC.DistinguishedTriangle) = _digest_cc(tri.cone) + _digest_cc(tri.Cshift)

function _digest_les(les::CC.LongExactSequence)
    acc = 0
    for M in les.fH
        acc += size(M, 1) + size(M, 2)
    end
    for M in les.iH
        acc += size(M, 1) + size(M, 2)
    end
    for M in les.pH
        acc += size(M, 1) + size(M, 2)
    end
    for M in les.delta
        acc += size(M, 1) + size(M, 2)
    end
    return acc
end

function _digest_ss_page(P::CC.SpectralPage)
    s = 0
    @inbounds for i in axes(P.dims, 1), j in axes(P.dims, 2)
        s += P.dims[i, j]
    end
    return s
end

function _digest_ss_terms(Ts)
    s = 0
    @inbounds for i in axes(Ts, 1), j in axes(Ts, 2)
        sq = Ts[i, j]
        s += sq.dimH + sq.dimZ + sq.dimB
    end
    return s
end

function _digest_ss_diff(D)
    return sum(B -> nnz(B) + size(B, 1) + size(B, 2), D)
end

function _write_csv(path::AbstractString, rows)
    open(path, "w") do io
        println(io, "probe,median_ms,median_kib")
        for r in rows
            println(io, string(r.probe, ",", r.median_ms, ",", r.median_kib))
        end
    end
end

function main(; reps::Int=7,
              field_name::String="qq",
              suite::String="all",
              profile::String="default",
              seed::Int=0xC0DEC0DE,
              density::Float64=0.30,
              ndeg::Int=5,
              cdim_min::Int=14,
              cdim_max::Int=28,
              nrhs::Int=6,
              solve_m::Int=80,
              solve_n::Int=64,
              solve_density::Float64=0.18,
              dc_na::Int=4,
              dc_nb::Int=4,
              dc_dim_min::Int=8,
              dc_dim_max::Int=16,
              verify::Bool=true,
              out::String=joinpath(@__DIR__, "_tmp_chain_complexes_microbench.csv"))
    reps >= 1 || error("main: reps must be >= 1")
    ndeg >= 3 || error("main: ndeg must be >= 3")
    nrhs >= 1 || error("main: nrhs must be >= 1")
    suite in ("all", "spectral", "exact", "exact_les", "exact_spectral") ||
        error("main: suite must be 'all', 'spectral', 'exact', 'exact_les', or 'exact_spectral'")

    run_exact_core = suite in ("all", "exact", "exact_les")
    run_spectral = suite in ("all", "spectral", "exact", "exact_spectral")
    run_exact_spectral_only = suite == "exact_spectral"
    run_les_only = suite == "exact_les"

    field = _field_from_name(field_name)
    K = CM.coeff_type(field)
    rng = MersenneTwister(seed)

    dims = _random_dims(rng, ndeg; dmin=cdim_min, dmax=cdim_max)
    C = _random_cochain_complex(rng, field; tmin=0, dims=dims, density=density)
    fmap = _identity_cochain_map(C)

    tri = CC.mapping_cone_triangle(fmap)
    les = CC.long_exact_sequence(tri; output=:full)

    tmid = C.tmin + div(C.tmax - C.tmin, 2)
    Hmid = CC.cohomology_data(C, tmid)
    hvec = Hmid.dimH > 0 ? Hmid.Hrep[:, 1] : zeros(K, Hmid.dimC)

    # Build a small chain fixture from the first two cochain differentials.
    bd_curr = Matrix(transpose(C.d[1]))
    bd_next = Matrix(transpose(C.d[2]))
    Hhom = CC.homology_data(bd_next, bd_curr, 1)
    hcyc = Hhom.dimH > 0 ? Hhom.Hrep[:, 1] : zeros(K, Hhom.dimC)

    # Consistent linear system A * X = B for solve_particular probes.
    A = _random_matrix(rng, field, solve_m, solve_n; density=solve_density)
    Xtrue = _random_rhs(rng, field, solve_n, nrhs)
    B = A * Xtrue
    Asp = sparse(A)

    # Basis-extension fixture.
    Cbasis_input = _random_matrix(rng, field, max(24, cdim_max), max(8, div(cdim_max, 2)); density=0.45)

    DC = nothing
    ss_ref = nothing
    ssh_ref = nothing
    dref = nothing
    Tot_ref = nothing
    blocks_ref = nothing
    exact_term_ab = nothing
    if run_spectral
        DC = _random_double_complex_vertical(
            rng,
            field;
            amin=0,
            amax=max(0, dc_na - 1),
            bmin=0,
            bmax=max(0, dc_nb - 1),
            dmin=dc_dim_min,
            dmax=dc_dim_max,
            density=min(0.45, density + 0.10),
        )
        ss_ref = CC.spectral_sequence(DC; output=:full, first=:vertical)
        ssh_ref = CC.spectral_sequence(DC; output=:full, first=:horizontal)
        dref = CC.differential(ss_ref, 1)
        Tot_ref, blocks_ref = CC._total_complex_with_blocks(DC)
        if field isa CM.QQField
            idx = argmax(ss_ref.E2_dims)
            I = CartesianIndices(ss_ref.E2_dims)[idx]
            exact_term_ab = (DC.amin + I[1] - 1, DC.bmin + I[2] - 1)
        end
    end

    if verify
        _check_d2_zero(C)
        CC.is_cochain_map(fmap) || error("fixture check failed: identity cochain map did not validate")
        # The homology input should satisfy bd_curr * bd_next = 0.
        bad = bd_curr * bd_next
        if field isa CM.RealField
            norm_bad = opnorm(bad, Inf)
            norm_bad <= 1e-8 || error("fixture check failed: bd_curr * bd_next != 0 (residual=$norm_bad)")
        else
            all(iszero, bad) || error("fixture check failed: bd_curr * bd_next != 0")
        end
        if field isa CM.QQField && run_spectral
            ss_verify = CC.spectral_sequence(DC; output=:full, first=:vertical)
            ss_verify.filt_img.value === nothing ||
                error("fixture check failed: exact spectral sequence should keep filt_img lazy before basis access")
            tmid = ss_verify.Tot.tmin + div(ss_verify.Tot.tmax - ss_verify.Tot.tmin, 2)
            tidx_mid = tmid - ss_verify.Tot.tmin + 1
            _ = CC.filtration_dims(ss_verify, tmid)
            ss_verify.filt_img.value === nothing ||
                error("fixture check failed: filtration_dims should not materialize filt_img bases")
            _ = CC.filtration_basis(ss_verify, ss_verify.pmin, tmid)
            if CC._use_exact_columnwise_filtimg_basis(ss_verify, tidx_mid)
                ss_verify.filt_img.value !== nothing &&
                    error("fixture check failed: exact filtration_basis should stay column-lazy when the size gate selects columnwise mode")
                ss_verify.filt_img_cols[tidx_mid].value === nothing &&
                    error("fixture check failed: filtration_basis should populate the requested total-degree basis cache in columnwise mode")
            else
                ss_verify.filt_img.value === nothing &&
                    error("fixture check failed: exact filtration_basis should materialize the full basis cache when the size gate selects full mode")
            end
            ss_verify.Einf_spaces.value = nothing
            _ = CC.page_terms(ss_verify, :inf)
            if CC._use_exact_columnwise_einf_basis(ss_verify, tidx_mid)
                ss_verify.filt_img.value !== nothing &&
                    error("fixture check failed: exact E_inf materialization should stay column-lazy when the size gate selects columnwise mode")
            end
            pg_verify = CC._ss_compute_page_data(DC, Tot_ref, blocks_ref, :vertical, 2)
            pg_verify_ambient = CC._ss_compute_page_data_ambient(DC, Tot_ref, blocks_ref, :vertical, 2)
            pg_verify.dims == pg_verify_ambient.dims ||
                error("fixture check failed: exact page-data kernel changed page dimensions")
            delete!(ss_verify.page_cache, 2)
            delete!(ss_verify.diff_cache, 2)
            _ = CC.differential(ss_verify, 2)
            haskey(ss_verify.page_cache, 2) &&
                error("fixture check failed: exact differential build should not retain page_cache[r]")
        end
    end

    println("ChainComplexes micro-benchmark")
    println("suite=$(suite), profile=$(profile), reps=$(reps), field=$(field_name), density=$(density), ndeg=$(ndeg), dims=$(dims)")
    if run_spectral
        println("solve(m,n,nrhs)=($(solve_m),$(solve_n),$(nrhs)), dc=($(dc_na)x$(dc_nb))")
    else
        println("solve(m,n,nrhs)=($(solve_m),$(solve_n),$(nrhs))")
    end

    rows = NamedTuple[]
    tiny_inner_reps = profile == "spectral_large" ? 4096 : 1024
    small_inner_reps = profile == "spectral_large" ? 512 : 256

    if run_exact_core
        if run_les_only
            push!(rows, _bench("cc cohomology_data all", () -> begin
                _digest_Hall(CC.cohomology_data(C))
            end; reps=reps))

            push!(rows, _bench("cc induced_map_on_cohomology", () -> begin
                M = CC.induced_map_on_cohomology(les.HC[1], les.HD[1], CC._map_at(tri.f, les.tmin))
                size(M, 1) + size(M, 2)
            end; reps=reps, inner_reps=tiny_inner_reps))

            push!(rows, _bench("cc mapping_cone_triangle", () -> begin
                _digest_tri(CC.mapping_cone_triangle(fmap))
            end; reps=reps))

            push!(rows, _bench("cc long_exact_sequence", () -> begin
                _digest_les(CC.long_exact_sequence(tri; output=:full))
            end; reps=reps))

            push!(rows, _bench("cc exact long_exact_sequence focused", () -> begin
                _digest_les(CC.long_exact_sequence(tri; output=:full))
            end; reps=reps))
        else
        push!(rows, _bench("cc solve_particular dense", () -> begin
            X = CC.solve_particular(field, A, B)
            size(X, 1) + size(X, 2)
        end; reps=reps))

        push!(rows, _bench("cc solve_particular sparse", () -> begin
            X = CC.solve_particular(Asp, B)
            X === nothing && error("expected a consistent sparse system")
            size(X, 1) + size(X, 2)
        end; reps=reps))

        push!(rows, _bench("cc extend_to_basis", () -> begin
            Bext = CC.extend_to_basis(Matrix(Cbasis_input))
            FL.rank(field, Bext)
        end; reps=reps))

        push!(rows, _bench("cc diff_summaries", () -> begin
            Hs = CC._diff_summaries(C)
            sum(s.rank + size(s.ker, 2) + size(s.img, 2) for s in Hs)
        end; reps=reps))

        push!(rows, _bench("cc cohomology_data degree", () -> begin
            _digest_H(CC.cohomology_data(C, tmid))
        end; reps=reps))

        push!(rows, _bench("cc cohomology_data all", () -> begin
            _digest_Hall(CC.cohomology_data(C))
        end; reps=reps))

        push!(rows, _bench("cc cohomology_data all + basis", () -> begin
            Hall = CC.cohomology_data(C)
            sum(size(CC.basis(H), 2) for H in Hall)
        end; reps=reps))

        push!(rows, _bench("cc cohomology_dims", () -> begin
            sum(CC.cohomology_dims(C))
        end; reps=reps))

        push!(rows, _bench("cc cohomology_coordinates", () -> begin
            coords = CC.cohomology_coordinates(Hmid, hvec)
            size(coords, 1) + size(coords, 2)
        end; reps=reps, inner_reps=tiny_inner_reps))

        push!(rows, _bench("cc cohomology_representative", () -> begin
            coords = Hmid.dimH > 0 ? ones(K, Hmid.dimH) : zeros(K, 0)
            rep = CC.cohomology_representative(Hmid, coords)
            length(rep)
        end; reps=reps, inner_reps=tiny_inner_reps))

        push!(rows, _bench("cc induced_map_on_cohomology", () -> begin
            fdeg = _sparse_eye(K, Hmid.dimC)
            MH = CC.induced_map_on_cohomology(Hmid, Hmid, fdeg)
            size(MH, 1) + size(MH, 2)
        end; reps=reps, inner_reps=tiny_inner_reps))

        push!(rows, _bench("cc homology_data", () -> begin
            _digest_hom(CC.homology_data(bd_next, bd_curr, 1))
        end; reps=reps))

        push!(rows, _bench("cc homology_coordinates", () -> begin
            c = CC.homology_coordinates(Hhom, hcyc)
            size(c, 1) + size(c, 2)
        end; reps=reps, inner_reps=tiny_inner_reps))

        push!(rows, _bench("cc induced_map_on_homology", () -> begin
            fchain = _sparse_eye(K, Hhom.dimC)
            MH = CC.induced_map_on_homology(Hhom, Hhom, fchain)
            size(MH, 1) + size(MH, 2)
        end; reps=reps, inner_reps=tiny_inner_reps))

        push!(rows, _bench("cc cochain_map construct_check", () -> begin
            m = _identity_cochain_map(C)
            CC.is_cochain_map(m) ? 1 : 0
        end; reps=reps))

        push!(rows, _bench("cc mapping_cone", () -> begin
            _digest_cc(CC.mapping_cone(fmap))
        end; reps=reps))

        push!(rows, _bench("cc mapping_cone_triangle", () -> begin
            _digest_tri(CC.mapping_cone_triangle(fmap))
        end; reps=reps))

        push!(rows, _bench("cc long_exact_sequence", () -> begin
            _digest_les(CC.long_exact_sequence(tri; output=:full))
        end; reps=reps))

        push!(rows, _bench("cc total_complex", () -> begin
            Tot = CC.total_complex(DC)
            _digest_cc(Tot)
        end; reps=reps, inner_reps=small_inner_reps))

        if profile == "qq_exact" || suite == "exact"
            push!(rows, _bench("cc exact long_exact_sequence focused", () -> begin
                _digest_les(CC.long_exact_sequence(tri; output=:full))
            end; reps=reps))
        end
        end
    end

    if run_spectral
        DCv = DC
        ssref = ss_ref
        sshref = ssh_ref
        dref_local = dref
        Tot_ref_v = Tot_ref
        blocks_ref_v = blocks_ref
        exact_term_ab_v = exact_term_ab

        if run_exact_spectral_only
            push!(rows, _bench("cc exact horizontal build auto", () -> begin
                oldb = CC._spectral_exact_filtimg_basis_mode[]
                try
                    CC._spectral_exact_filtimg_basis_mode[] = :auto
                    ss = CC.spectral_sequence(DCv; output=:full, first=:horizontal)
                    _digest_ss_page(CC.page(ss, 1)) + _digest_ss_diff(CC.differential(ss, 1))
                finally
                    CC._spectral_exact_filtimg_basis_mode[] = oldb
                end
            end; reps=reps))

            push!(rows, _bench("cc exact horizontal build old_policy", () -> begin
                oldb = CC._spectral_exact_filtimg_basis_mode[]
                try
                    CC._spectral_exact_filtimg_basis_mode[] = :full
                    ss = CC.spectral_sequence(DCv; output=:full, first=:horizontal)
                    _digest_ss_page(CC.page(ss, 1)) + _digest_ss_diff(CC.differential(ss, 1))
                finally
                    CC._spectral_exact_filtimg_basis_mode[] = oldb
                end
            end; reps=reps))

            push!(rows, _bench("cc exact filtration_basis mid auto_materialization", () -> begin
                oldb = CC._spectral_exact_filtimg_basis_mode[]
                try
                    CC._spectral_exact_filtimg_basis_mode[] = :auto
                    ss = CC.spectral_sequence(DCv; output=:full, first=:vertical)
                    tmid = ss.Tot.tmin + div(ss.Tot.tmax - ss.Tot.tmin, 2)
                    B = CC.filtration_basis(ss, ss.pmin, tmid)
                    size(B, 1) + size(B, 2)
                finally
                    CC._spectral_exact_filtimg_basis_mode[] = oldb
                end
            end; reps=reps))

            push!(rows, _bench("cc exact filtration_basis mid old_materialization", () -> begin
                oldb = CC._spectral_exact_filtimg_basis_mode[]
                try
                    CC._spectral_exact_filtimg_basis_mode[] = :full
                    ss = CC.spectral_sequence(DCv; output=:full, first=:vertical)
                    tmid = ss.Tot.tmin + div(ss.Tot.tmax - ss.Tot.tmin, 2)
                    B = CC.filtration_basis(ss, ss.pmin, tmid)
                    size(B, 1) + size(B, 2)
                finally
                    CC._spectral_exact_filtimg_basis_mode[] = oldb
                end
            end; reps=reps))

            push!(rows, _bench("cc exact filtration_basis mid columnwise_forced", () -> begin
                oldb = CC._spectral_exact_filtimg_basis_mode[]
                try
                    CC._spectral_exact_filtimg_basis_mode[] = :columnwise
                    ss = CC.spectral_sequence(DCv; output=:full, first=:vertical)
                    tmid = ss.Tot.tmin + div(ss.Tot.tmax - ss.Tot.tmin, 2)
                    B = CC.filtration_basis(ss, ss.pmin, tmid)
                    size(B, 1) + size(B, 2)
                finally
                    CC._spectral_exact_filtimg_basis_mode[] = oldb
                end
            end; reps=reps))

            push!(rows, _bench("cc exact page_terms inf prebuilt auto_materialization", () -> begin
                oldb = CC._spectral_exact_filtimg_basis_mode[]
                olds = CC._spectral_exact_filtimg_summary_reuse[]
                try
                    CC._spectral_exact_filtimg_basis_mode[] = :auto
                    CC._spectral_exact_filtimg_summary_reuse[] = true
                    ssref.Einf_spaces.value = nothing
                    ssref.filt_img.value = nothing
                    for lv in ssref.filt_img_cols
                        lv.value = nothing
                    end
                    _digest_ss_terms(CC.page_terms(ssref, :inf))
                finally
                    CC._spectral_exact_filtimg_basis_mode[] = oldb
                    CC._spectral_exact_filtimg_summary_reuse[] = olds
                end
            end; reps=reps))

            push!(rows, _bench("cc exact page_terms inf prebuilt no_summary_reuse", () -> begin
                oldb = CC._spectral_exact_filtimg_basis_mode[]
                olds = CC._spectral_exact_filtimg_summary_reuse[]
                try
                    CC._spectral_exact_filtimg_basis_mode[] = :auto
                    CC._spectral_exact_filtimg_summary_reuse[] = false
                    ssref.Einf_spaces.value = nothing
                    ssref.filt_img.value = nothing
                    for lv in ssref.filt_img_cols
                        lv.value = nothing
                    end
                    CC._ss_clear_filtimg_summary_cache!(ssref)
                    _digest_ss_terms(CC.page_terms(ssref, :inf))
                finally
                    CC._spectral_exact_filtimg_basis_mode[] = oldb
                    CC._spectral_exact_filtimg_summary_reuse[] = olds
                end
            end; reps=reps))

            push!(rows, _bench("cc exact page_terms inf prebuilt old_materialization", () -> begin
                oldb = CC._spectral_exact_filtimg_basis_mode[]
                try
                    CC._spectral_exact_filtimg_basis_mode[] = :full
                    ssref.Einf_spaces.value = nothing
                    ssref.filt_img.value = nothing
                    for lv in ssref.filt_img_cols
                        lv.value = nothing
                    end
                    _digest_ss_terms(CC.page_terms(ssref, :inf))
                finally
                    CC._spectral_exact_filtimg_basis_mode[] = oldb
                end
            end; reps=reps))

            push!(rows, _bench("cc exact page_terms inf prebuilt columnwise_forced", () -> begin
                oldb = CC._spectral_exact_filtimg_basis_mode[]
                try
                    CC._spectral_exact_filtimg_basis_mode[] = :columnwise
                    ssref.Einf_spaces.value = nothing
                    ssref.filt_img.value = nothing
                    for lv in ssref.filt_img_cols
                        lv.value = nothing
                    end
                    _digest_ss_terms(CC.page_terms(ssref, :inf))
                finally
                    CC._spectral_exact_filtimg_basis_mode[] = oldb
                end
            end; reps=reps))

            push!(rows, _bench("cc exact horizontal page_terms inf auto_materialization", () -> begin
                oldb = CC._spectral_exact_filtimg_basis_mode[]
                try
                    CC._spectral_exact_filtimg_basis_mode[] = :auto
                    sshref.Einf_spaces.value = nothing
                    sshref.filt_img.value = nothing
                    for lv in sshref.filt_img_cols
                        lv.value = nothing
                    end
                    _digest_ss_terms(CC.page_terms(sshref, :inf))
                finally
                    CC._spectral_exact_filtimg_basis_mode[] = oldb
                end
            end; reps=reps))

            push!(rows, _bench("cc exact horizontal page_terms inf old_materialization", () -> begin
                oldb = CC._spectral_exact_filtimg_basis_mode[]
                try
                    CC._spectral_exact_filtimg_basis_mode[] = :full
                    sshref.Einf_spaces.value = nothing
                    sshref.filt_img.value = nothing
                    for lv in sshref.filt_img_cols
                        lv.value = nothing
                    end
                    _digest_ss_terms(CC.page_terms(sshref, :inf))
                finally
                    CC._spectral_exact_filtimg_basis_mode[] = oldb
                end
            end; reps=reps))

            push!(rows, _bench("cc exact _ss_compute_page_term r2 ambient", () -> begin
                aterm, bterm = exact_term_ab_v
                sq = CC._ss_compute_page_term_ambient(
                    DCv,
                    Tot_ref_v,
                    blocks_ref_v,
                    :vertical,
                    2,
                    aterm,
                    bterm,
                    CC._SSPageWorkspace(CM.coeff_type(field)),
                )
                sq.dimZ + sq.dimB + sq.dimH
            end; reps=reps))

            push!(rows, _bench("cc exact _ss_compute_page_data r2 ambient", () -> begin
                pg = CC._ss_compute_page_data_ambient(DCv, Tot_ref_v, blocks_ref_v, :vertical, 2)
                _digest_ss_terms(pg.spaces)
            end; reps=reps))
        else

        push!(rows, _bench("cc spectral_sequence build", () -> begin
            ss = CC.spectral_sequence(DCv; output=:full, first=:vertical)
            _digest_ss_page(CC.page(ss, 1)) + _digest_ss_diff(CC.differential(ss, 1))
        end; reps=reps))

        push!(rows, _bench("cc spectral_sequence build horizontal", () -> begin
            ss = CC.spectral_sequence(DCv; output=:full, first=:horizontal)
            _digest_ss_page(CC.page(ss, 1)) + _digest_ss_diff(CC.differential(ss, 1))
        end; reps=reps))

        if profile == "qq_exact" || suite == "exact"
            push!(rows, _bench("cc exact vertical build canonical", () -> begin
                ss = CC.spectral_sequence(DCv; output=:full, first=:vertical)
                _digest_ss_page(CC.page(ss, 1)) + _digest_ss_diff(CC.differential(ss, 1))
            end; reps=reps))

            push!(rows, _bench("cc exact horizontal build full_materialization", () -> begin
                old = CC._spectral_exact_filtimg_basis_mode[]
                try
                    CC._spectral_exact_filtimg_basis_mode[] = :full
                    ss = CC.spectral_sequence(DCv; output=:full, first=:horizontal)
                    _digest_ss_page(CC.page(ss, 1)) + _digest_ss_diff(CC.differential(ss, 1))
                finally
                    CC._spectral_exact_filtimg_basis_mode[] = old
                end
            end; reps=reps))
        end

        push!(rows, _bench("cc spectral page r2 warm", () -> begin
            _digest_ss_page(CC.page(ssref, 2))
        end; reps=reps, inner_reps=tiny_inner_reps))

        push!(rows, _bench("cc spectral page_terms r2 materialize", () -> begin
            ss = CC.spectral_sequence(DCv; output=:full, first=:vertical)
            _digest_ss_terms(CC.page_terms(ss, 2))
        end; reps=reps))

        if profile == "qq_exact" || suite == "exact"
            push!(rows, _bench("cc exact _ss_compute_page_term r2", () -> begin
                aterm, bterm = exact_term_ab_v
                sq = CC._ss_compute_page_term(
                    DCv,
                    Tot_ref_v,
                    blocks_ref_v,
                    :vertical,
                    2,
                    aterm,
                    bterm,
                    CC._SSPageWorkspace(CM.coeff_type(field)),
                )
                sq.dimZ + sq.dimB + sq.dimH
            end; reps=reps))

            push!(rows, _bench("cc exact page_terms r2 prebuilt", () -> begin
                delete!(ssref.page_cache, 2)
                _digest_ss_terms(CC.page_terms(ssref, 2))
            end; reps=reps))

            push!(rows, _bench("cc exact _ss_compute_page_data r2", () -> begin
                pg = CC._ss_compute_page_data(DCv, Tot_ref_v, blocks_ref_v, :vertical, 2)
                _digest_ss_terms(pg.spaces)
            end; reps=reps))

            push!(rows, _bench("cc exact _ss_compute_page_data r2 ambient", () -> begin
                pg = CC._ss_compute_page_data_ambient(DCv, Tot_ref_v, blocks_ref_v, :vertical, 2)
                _digest_ss_terms(pg.spaces)
            end; reps=reps))

            push!(rows, _bench("cc exact _ss_compute_differential r2", () -> begin
                pg = CC._ss_compute_page_data(DCv, Tot_ref_v, blocks_ref_v, :vertical, 2)
                _digest_ss_diff(CC._ss_compute_differential(DCv, Tot_ref_v, :vertical, pg, 2))
            end; reps=reps))

            push!(rows, _bench("cc exact differential r2 prebuilt", () -> begin
                delete!(ssref.page_cache, 2)
                delete!(ssref.diff_cache, 2)
                _digest_ss_diff(CC.differential(ssref, 2))
            end; reps=reps))

            push!(rows, _bench("cc exact filtration_dims mid warm", () -> begin
                tmid = ssref.Tot.tmin + div(ssref.Tot.tmax - ssref.Tot.tmin, 2)
                sum(values(CC.filtration_dims(ssref, tmid)))
            end; reps=reps, inner_reps=tiny_inner_reps))

            push!(rows, _bench("cc exact filtration_basis mid materialize", () -> begin
                ss = CC.spectral_sequence(DCv; output=:full, first=:vertical)
                tmid = ss.Tot.tmin + div(ss.Tot.tmax - ss.Tot.tmin, 2)
                B = CC.filtration_basis(ss, ss.pmin, tmid)
                size(B, 1) + size(B, 2)
            end; reps=reps))

            push!(rows, _bench("cc exact page_terms inf prebuilt", () -> begin
                olds = CC._spectral_exact_filtimg_summary_reuse[]
                try
                    CC._spectral_exact_filtimg_summary_reuse[] = true
                    ssref.Einf_spaces.value = nothing
                    _digest_ss_terms(CC.page_terms(ssref, :inf))
                finally
                    CC._spectral_exact_filtimg_summary_reuse[] = olds
                end
            end; reps=reps))

            push!(rows, _bench("cc exact page_terms inf prebuilt no_summary_reuse", () -> begin
                olds = CC._spectral_exact_filtimg_summary_reuse[]
                try
                    CC._spectral_exact_filtimg_summary_reuse[] = false
                    ssref.Einf_spaces.value = nothing
                    ssref.filt_img.value = nothing
                    for lv in ssref.filt_img_cols
                        lv.value = nothing
                    end
                    CC._ss_clear_filtimg_summary_cache!(ssref)
                    _digest_ss_terms(CC.page_terms(ssref, :inf))
                finally
                    CC._spectral_exact_filtimg_summary_reuse[] = olds
                end
            end; reps=reps))

            push!(rows, _bench("cc exact horizontal page_terms inf prebuilt", () -> begin
                sshref.Einf_spaces.value = nothing
                _digest_ss_terms(CC.page_terms(sshref, :inf))
            end; reps=reps))

            push!(rows, _bench("cc exact _ss_compute_differential r2 ambient", () -> begin
                old = CC._spectral_exact_diff_mode[]
                try
                    CC._spectral_exact_diff_mode[] = :ambient
                    pg = CC._ss_compute_page_data(DCv, Tot_ref_v, blocks_ref_v, :vertical, 2)
                    _digest_ss_diff(CC._ss_compute_differential(DCv, Tot_ref_v, :vertical, pg, 2))
                finally
                    CC._spectral_exact_diff_mode[] = old
                end
            end; reps=reps))

            push!(rows, _bench("cc exact vertical build cache_off", () -> begin
                oldcache = CC._spectral_exact_filtimg_cache_mode[]
                try
                    CC._spectral_exact_filtimg_cache_mode[] = :off
                    ss = CC.spectral_sequence(DCv; output=:full, first=:vertical)
                    _digest_ss_page(CC.page(ss, 1)) + _digest_ss_diff(CC.differential(ss, 1))
                finally
                    CC._spectral_exact_filtimg_cache_mode[] = oldcache
                end
            end; reps=reps))

            push!(rows, _bench("cc exact vertical build cache_on", () -> begin
                oldcache = CC._spectral_exact_filtimg_cache_mode[]
                try
                    CC._spectral_exact_filtimg_cache_mode[] = :on
                    ss = CC.spectral_sequence(DCv; output=:full, first=:vertical)
                    _digest_ss_page(CC.page(ss, 1)) + _digest_ss_diff(CC.differential(ss, 1))
                finally
                    CC._spectral_exact_filtimg_cache_mode[] = oldcache
                end
            end; reps=reps))
        end

        push!(rows, _bench("cc spectral differential r2 materialize", () -> begin
            ss = CC.spectral_sequence(DCv; output=:full, first=:vertical)
            _digest_ss_diff(CC.differential(ss, 2))
        end; reps=reps))

        push!(rows, _bench("cc spectral differential r1 warm", () -> begin
            _digest_ss_diff(CC.differential(ssref, 1))
        end; reps=reps, inner_reps=tiny_inner_reps))

        push!(rows, _bench("cc spectral differential r2 warm", () -> begin
            _digest_ss_diff(CC.differential(ssref, 2))
        end; reps=reps, inner_reps=tiny_inner_reps))

        push!(rows, _bench("cc spectral page_dims_dict inf warm", () -> begin
            d = CC.page_dims_dict(ssref, :inf; nonzero_only=true)
            length(d)
        end; reps=reps, inner_reps=tiny_inner_reps))

        push!(rows, _bench("cc spectral page_terms inf materialize", () -> begin
            ss = CC.spectral_sequence(DCv; output=:full, first=:vertical)
            _digest_ss_terms(CC.page_terms(ss, :inf))
        end; reps=reps))

        push!(rows, _bench("cc spectral split_total_cohomology", () -> begin
            t = ssref.Tot.tmin + div(ssref.Tot.tmax - ssref.Tot.tmin, 2)
            S = CC.split_total_cohomology(ssref, t)
            size(S.B, 1) + size(S.B, 2) + size(S.Binv, 1) + size(S.Binv, 2) + length(S.ranges)
        end; reps=reps, inner_reps=tiny_inner_reps))

        push!(rows, _bench("cc spectral convergence_report", () -> begin
            length(CC.convergence_report(ssref; verbose=false))
        end; reps=reps, inner_reps=tiny_inner_reps))

        push!(rows, _bench("cc spectral differential cache_hit", () -> begin
            # Ensure r=1 diff is precomputed so this measures lookup/read overhead.
            _ = dref_local
            _digest_ss_diff(CC.differential(ssref, 1))
        end; reps=reps, inner_reps=tiny_inner_reps))
        end
    end

    if suite == "all" || suite == "exact" || suite == "exact_les"
        # Include an end-to-end warm uncached path (algorithmic run; no startup/JIT).
        push!(rows, _bench("cc e2e warm_uncached", () -> begin
            local_rng = MersenneTwister(seed + 911)
            dims2 = _random_dims(local_rng, ndeg; dmin=cdim_min, dmax=cdim_max)
            C2 = _random_cochain_complex(local_rng, field; tmin=0, dims=dims2, density=density)
            fmap2 = _identity_cochain_map(C2)
            tri2 = CC.mapping_cone_triangle(fmap2)
            les2 = CC.long_exact_sequence(tri2; output=:full)
            _digest_cc(C2) + _digest_les(les2)
        end; reps=reps))
    end

    _write_csv(out, rows)
    println("Wrote ", out)
end

if abspath(PROGRAM_FILE) == @__FILE__
    profile = _parse_string_arg(ARGS, "--profile", "default")
    defaults = _profile_defaults(profile)
    suite = _parse_string_arg(ARGS, "--suite", "all")
    suite in ("all", "spectral", "exact", "exact_les", "exact_spectral") ||
        error("unknown suite '$suite' (supported: all,spectral,exact,exact_les,exact_spectral)")
    reps = max(1, _parse_int_arg(ARGS, "--reps", 7))
    field_name = _parse_string_arg(ARGS, "--field", "qq")
    if profile == "qq_exact_les"
        suite == "all" && (suite = "exact_les")
        suite == "exact_les" || error("profile=qq_exact_les requires suite=exact_les")
        field_name == "qq" || error("profile=qq_exact_les requires --field=qq")
        reps = max(3, reps)
    elseif profile == "qq_exact_spectral"
        suite == "all" && (suite = "exact_spectral")
        suite == "exact_spectral" || error("profile=qq_exact_spectral requires suite=exact_spectral")
        field_name == "qq" || error("profile=qq_exact_spectral requires --field=qq")
    end
    seed = _parse_int_arg(ARGS, "--seed", Int(0xC0DEC0DE))
    density = clamp(_parse_float_arg(ARGS, "--density", defaults.density), 0.0, 1.0)

    ndeg = max(3, _parse_int_arg(ARGS, "--ndeg", defaults.ndeg))
    cdim_min = max(0, _parse_int_arg(ARGS, "--cdim_min", defaults.cdim_min))
    cdim_max = max(cdim_min, _parse_int_arg(ARGS, "--cdim_max", defaults.cdim_max))

    nrhs = max(1, _parse_int_arg(ARGS, "--nrhs", defaults.nrhs))
    solve_m = max(1, _parse_int_arg(ARGS, "--solve_m", defaults.solve_m))
    solve_n = max(1, _parse_int_arg(ARGS, "--solve_n", defaults.solve_n))
    solve_density = clamp(_parse_float_arg(ARGS, "--solve_density", defaults.solve_density), 0.0, 1.0)

    dc_na = max(1, _parse_int_arg(ARGS, "--dc_na", defaults.dc_na))
    dc_nb = max(1, _parse_int_arg(ARGS, "--dc_nb", defaults.dc_nb))
    dc_dim_min = max(0, _parse_int_arg(ARGS, "--dc_dim_min", defaults.dc_dim_min))
    dc_dim_max = max(dc_dim_min, _parse_int_arg(ARGS, "--dc_dim_max", defaults.dc_dim_max))

    verify = _parse_bool_arg(ARGS, "--verify", true)
    out = _parse_path_arg(ARGS, "--out", joinpath(@__DIR__, "_tmp_chain_complexes_microbench.csv"))

    main(; reps=reps,
         field_name=field_name,
         suite=suite,
         profile=profile,
         seed=seed,
         density=density,
         ndeg=ndeg,
         cdim_min=cdim_min,
         cdim_max=cdim_max,
         nrhs=nrhs,
         solve_m=solve_m,
         solve_n=solve_n,
         solve_density=solve_density,
         dc_na=dc_na,
         dc_nb=dc_nb,
         dc_dim_min=dc_dim_min,
         dc_dim_max=dc_dim_max,
         verify=verify,
         out=out)
end
