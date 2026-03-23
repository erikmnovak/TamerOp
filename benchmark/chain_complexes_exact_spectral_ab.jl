#!/usr/bin/env julia

using Random
using SparseArrays
using Statistics

module _ChainComplexesExactSpectralAB
include(joinpath(@__DIR__, "..", "src", "CoreModules.jl"))
include(joinpath(@__DIR__, "..", "src", "FieldLinAlg.jl"))
include(joinpath(@__DIR__, "..", "src", "ChainComplexes.jl"))
end

const CM = _ChainComplexesExactSpectralAB.CoreModules
const FL = _ChainComplexesExactSpectralAB.FieldLinAlg
const CC = _ChainComplexesExactSpectralAB.ChainComplexes

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

function _parse_path_arg(args, key::String, default::String)
    for a in args
        startswith(a, key * "=") || continue
        return String(split(a, "=", limit=2)[2])
    end
    return default
end

function _parse_int_arg_min(args, key::String, default::Int, minv::Int)
    v = _parse_int_arg(args, key, default)
    v >= minv || error("$key must be >= $minv")
    return v
end

function _parse_bool_arg(args, key::String, default::Bool)
    for a in args
        startswith(a, key * "=") || continue
        v = lowercase(strip(split(a, "=", limit=2)[2]))
        v in ("1", "true", "yes", "on") && return true
        v in ("0", "false", "no", "off") && return false
        error("invalid boolean for $key: $v")
    end
    return default
end

function _write_csv(path::AbstractString, rows)
    open(path, "w") do io
        println(io, "probe,median_ms,median_kib")
        for row in rows
            println(io, row.probe, ",", row.median_ms, ",", row.median_kib)
        end
    end
end

@inline _coeff(field) = CM.coeff_type(field)

function _rc(rng, field)
    CM.coerce(field, rand(rng, -3:3))
end

_rnc(rng, field) = (x = _rc(rng, field); iszero(x) ? one(_coeff(field)) : x)

function _rmat(rng, field, m, n; density=0.30)
    K = _coeff(field)
    A = zeros(K, m, n)
    for i in 1:m, j in 1:n
        rand(rng) < density || continue
        A[i, j] = _rnc(rng, field)
    end
    return A
end

function _rdiffs(rng, field, dims; density=0.30)
    K = _coeff(field)
    out = Vector{SparseMatrixCSC{K,Int}}()
    length(dims) <= 1 && return out
    push!(out, sparse(_rmat(rng, field, dims[2], dims[1]; density=density)))
    for i in 2:(length(dims) - 1)
        prev = Matrix(out[i - 1])
        left = FL.nullspace(field, transpose(prev))
        r = size(left, 2)
        if dims[i + 1] == 0 || dims[i] == 0 || r == 0
            push!(out, spzeros(K, dims[i + 1], dims[i]))
        else
            coeff = _rmat(rng, field, dims[i + 1], r; density=density)
            push!(out, sparse(coeff * transpose(left)))
        end
    end
    return out
end

function _build_dc(; seed::Int=11, density::Float64=0.30, na::Int=3, nb::Int=2, dmin::Int=2, dmax::Int=4)
    rng = MersenneTwister(seed)
    field = CM.QQField()
    K = _coeff(field)
    dims = [rand(rng, dmin:dmax) for _ in 1:na, __ in 1:nb]
    dv = Array{SparseMatrixCSC{K,Int},2}(undef, na, nb)
    dh = Array{SparseMatrixCSC{K,Int},2}(undef, na, nb)
    for ai in 1:na
        col_dims = [dims[ai, bi] for bi in 1:nb]
        col_d = _rdiffs(rng, field, col_dims; density=density)
        for bi in 1:nb
            dv[ai, bi] = bi < nb ? col_d[bi] : spzeros(K, 0, col_dims[bi])
            dh[ai, bi] = ai < na ? spzeros(K, dims[ai + 1, bi], dims[ai, bi]) : spzeros(K, 0, dims[ai, bi])
        end
    end
    return CC.DoubleComplex{K}(0, na - 1, 0, nb - 1, dims, dv, dh)
end

function _digest_terms(A)
    s = 0
    for sq in A
        s += sq.dimZ + sq.dimB + sq.dimH + sq.ambient_dim
    end
    return s
end

function _digest_diff(A)
    s = 0
    for M in A
        s += size(M, 1) + size(M, 2) + nnz(M)
    end
    return s
end

function _digest_page(P)
    s = 0
    for x in P
        s += x
    end
    return s
end

function _bench(name::AbstractString, f::Function; reps::Int=3)
    times = Vector{Float64}(undef, reps)
    kib = Vector{Float64}(undef, reps)
    for i in 1:reps
        GC.gc()
        t = 0.0
        kib[i] = @allocated begin
            t = @elapsed f()
        end / 1024
        times[i] = t * 1000
    end
    return (probe=String(name), median_ms=Statistics.median(times), median_kib=Statistics.median(kib))
end

function _warmup()
    DC = _build_dc(seed=7, density=0.20, na=2, nb=2, dmin=1, dmax=1)
    ssv = CC.spectral_sequence(DC; output=:full, first=:vertical)
    ssh = CC.spectral_sequence(DC; output=:full, first=:horizontal)
    _ = CC.page(ssv, 1)
    _ = CC.differential(ssv, 1)
    _ = CC.filtration_dims(ssv, ssv.Tot.tmin)
    _ = CC.page_terms(ssv, :inf)
    _ = CC.page(ssh, 1)
    _ = CC.differential(ssh, 1)
    oldb = CC._spectral_exact_filtimg_basis_mode[]
    Tot, blocks = CC._total_complex_with_blocks(DC)
    try
        CC._spectral_exact_filtimg_basis_mode[] = :full
        _ = CC.page_terms(ssv, :inf)
        CC._spectral_exact_filtimg_basis_mode[] = :columnwise
        ssv.Einf_spaces.value = nothing
        ssv.filt_img.value = nothing
        for lv in ssv.filt_img_cols
            lv.value = nothing
        end
        _ = CC.page_terms(ssv, :inf)
        pg = CC._ss_compute_page_data(DC, Tot, blocks, :vertical, 2)
        _ = CC._ss_compute_page_data_ambient(DC, Tot, blocks, :vertical, 2)
        idx = argmax([sq.dimH for sq in pg.spaces])
        I = CartesianIndices(pg.spaces)[idx]
        ws = CC._SSPageWorkspace(_coeff(CM.QQField()))
        _ = CC._ss_compute_page_term(DC, Tot, blocks, :vertical, 2, DC.amin + I[1] - 1, DC.bmin + I[2] - 1, ws)
        _ = CC._ss_compute_page_term_ambient(DC, Tot, blocks, :vertical, 2, DC.amin + I[1] - 1, DC.bmin + I[2] - 1, ws)
        ssv.diff_cache[1] = CC.differential(ssv, 1)
        _ = CC.differential(ssv, 2)
    finally
        CC._spectral_exact_filtimg_basis_mode[] = oldb
    end
    return nothing
end

function main(; out::String=joinpath(@__DIR__, "_tmp_chain_complexes_exact_spectral_ab.csv"),
              seed::Int=11,
              density::Float64=0.30,
              na::Int=2,
              nb::Int=2,
              dmin::Int=1,
              dmax::Int=2,
              verify::Bool=true,
              reps::Int=3)
    DC = _build_dc(seed=seed, density=density, na=na, nb=nb, dmin=dmin, dmax=dmax)

    old_b = CC._spectral_exact_filtimg_basis_mode[]
    try
        println("warmup")
        flush(stdout)
        _warmup()

        ssv_auto = CC.spectral_sequence(DC; output=:full, first=:vertical)
        ssh_auto = CC.spectral_sequence(DC; output=:full, first=:horizontal)
        CC._spectral_exact_filtimg_basis_mode[] = :full
        ssv_old = CC.spectral_sequence(DC; output=:full, first=:vertical)
        ssh_old = CC.spectral_sequence(DC; output=:full, first=:horizontal)
        Tot_ref, blocks_ref = CC._total_complex_with_blocks(DC)
        idx_ref = argmax(ssv_auto.E2_dims)
        Iref = CartesianIndices(ssv_auto.E2_dims)[idx_ref]
        aterm_ref = DC.amin + Iref[1] - 1
        bterm_ref = DC.bmin + Iref[2] - 1

        if verify
            CC._spectral_exact_filtimg_basis_mode[] = :auto
            B_auto = CC.filtration_basis(ssv_auto, ssv_auto.pmin, ssv_auto.Tot.tmin)
            CC._spectral_exact_filtimg_basis_mode[] = :full
            B_old = CC.filtration_basis(ssv_old, ssv_old.pmin, ssv_old.Tot.tmin)
            Matrix(B_auto) == Matrix(B_old) || error("filtration_basis parity failed")
            ssv_auto.Einf_spaces.value = nothing
            ssv_old.Einf_spaces.value = nothing
            [sq.dimH for sq in CC.page_terms(ssv_auto, :inf)] == [sq.dimH for sq in CC.page_terms(ssv_old, :inf)] ||
                error("E_inf parity failed")
            Tot, blocks = CC._total_complex_with_blocks(DC)
            pg_exact = CC._ss_compute_page_data(DC, Tot, blocks, :vertical, 2)
            pg_ambient = CC._ss_compute_page_data_ambient(DC, Tot, blocks, :vertical, 2)
            pg_exact.dims == pg_ambient.dims || error("page-data ambient/exact parity failed")
            ssh_auto.E1_dims == ssh_old.E1_dims || error("horizontal E1 parity failed")
            ssh_auto.E2_dims == ssh_old.E2_dims || error("horizontal E2 parity failed")
            ssh_auto.Einf_dims == ssh_old.Einf_dims || error("horizontal E_inf parity failed")
        end

        rows = NamedTuple[]

        function pushrow(name, f)
            println("bench ", name)
            row = _bench(name, f; reps=reps)
            push!(rows, row)
            println(name, " ms=", row.median_ms, " kib=", row.median_kib)
            flush(stdout)
        end

        pushrow("exact_horizontal_build_old_policy", () -> begin
            oldb2 = CC._spectral_exact_filtimg_basis_mode[]
            try
                CC._spectral_exact_filtimg_basis_mode[] = :full
                ss = CC.spectral_sequence(DC; output=:full, first=:horizontal)
                _digest_page(CC.page(ss, 1)) + _digest_diff(CC.differential(ss, 1))
            finally
                CC._spectral_exact_filtimg_basis_mode[] = oldb2
            end
        end)

        pushrow("exact_horizontal_build_auto", () -> begin
            oldb2 = CC._spectral_exact_filtimg_basis_mode[]
            try
                CC._spectral_exact_filtimg_basis_mode[] = :auto
                ss = CC.spectral_sequence(DC; output=:full, first=:horizontal)
                _digest_page(CC.page(ss, 1)) + _digest_diff(CC.differential(ss, 1))
            finally
                CC._spectral_exact_filtimg_basis_mode[] = oldb2
            end
        end)

        pushrow("exact_page_terms_inf_old_materialization", () -> begin
            oldb2 = CC._spectral_exact_filtimg_basis_mode[]
            try
                CC._spectral_exact_filtimg_basis_mode[] = :full
                ssv_old.Einf_spaces.value = nothing
                ssv_old.filt_img.value = nothing
                for lv in ssv_old.filt_img_cols
                    lv.value = nothing
                end
                _digest_terms(CC.page_terms(ssv_old, :inf))
            finally
                CC._spectral_exact_filtimg_basis_mode[] = oldb2
            end
        end)

        pushrow("exact_page_terms_inf_auto_materialization", () -> begin
            oldb2 = CC._spectral_exact_filtimg_basis_mode[]
            try
                CC._spectral_exact_filtimg_basis_mode[] = :auto
                ssv_auto.Einf_spaces.value = nothing
                ssv_auto.filt_img.value = nothing
                for lv in ssv_auto.filt_img_cols
                    lv.value = nothing
                end
                _digest_terms(CC.page_terms(ssv_auto, :inf))
            finally
                CC._spectral_exact_filtimg_basis_mode[] = oldb2
            end
        end)

        pushrow("exact_horizontal_page_terms_inf_old_materialization", () -> begin
            oldb2 = CC._spectral_exact_filtimg_basis_mode[]
            try
                CC._spectral_exact_filtimg_basis_mode[] = :full
                ssh_old.Einf_spaces.value = nothing
                ssh_old.filt_img.value = nothing
                for lv in ssh_old.filt_img_cols
                    lv.value = nothing
                end
                _digest_terms(CC.page_terms(ssh_old, :inf))
            finally
                CC._spectral_exact_filtimg_basis_mode[] = oldb2
            end
        end)

        pushrow("exact_horizontal_page_terms_inf_auto_materialization", () -> begin
            oldb2 = CC._spectral_exact_filtimg_basis_mode[]
            try
                CC._spectral_exact_filtimg_basis_mode[] = :auto
                ssh_auto.Einf_spaces.value = nothing
                ssh_auto.filt_img.value = nothing
                for lv in ssh_auto.filt_img_cols
                    lv.value = nothing
                end
                _digest_terms(CC.page_terms(ssh_auto, :inf))
            finally
                CC._spectral_exact_filtimg_basis_mode[] = oldb2
            end
        end)

        pushrow("exact_filtration_basis_auto_materialization", () -> begin
            oldb2 = CC._spectral_exact_filtimg_basis_mode[]
            try
                CC._spectral_exact_filtimg_basis_mode[] = :auto
                ss = CC.spectral_sequence(DC; output=:full, first=:vertical)
                tmid = ss.Tot.tmin + div(ss.Tot.tmax - ss.Tot.tmin, 2)
                B = CC.filtration_basis(ss, ss.pmin, tmid)
                size(B, 1) + size(B, 2)
            finally
                CC._spectral_exact_filtimg_basis_mode[] = oldb2
            end
        end)

        pushrow("exact_filtration_basis_columnwise_forced", () -> begin
            oldb2 = CC._spectral_exact_filtimg_basis_mode[]
            try
                CC._spectral_exact_filtimg_basis_mode[] = :columnwise
                ss = CC.spectral_sequence(DC; output=:full, first=:vertical)
                tmid = ss.Tot.tmin + div(ss.Tot.tmax - ss.Tot.tmin, 2)
                B = CC.filtration_basis(ss, ss.pmin, tmid)
                size(B, 1) + size(B, 2)
            finally
                CC._spectral_exact_filtimg_basis_mode[] = oldb2
            end
        end)

        pushrow("exact_page_terms_inf_columnwise_forced", () -> begin
            oldb2 = CC._spectral_exact_filtimg_basis_mode[]
            try
                CC._spectral_exact_filtimg_basis_mode[] = :columnwise
                ssv_auto.Einf_spaces.value = nothing
                ssv_auto.filt_img.value = nothing
                for lv in ssv_auto.filt_img_cols
                    lv.value = nothing
                end
                _digest_terms(CC.page_terms(ssv_auto, :inf))
            finally
                CC._spectral_exact_filtimg_basis_mode[] = oldb2
            end
        end)

        pushrow("exact_page_term_r2_exact", () -> begin
            sq = CC._ss_compute_page_term(DC, Tot_ref, blocks_ref, :vertical, 2,
                                          aterm_ref, bterm_ref,
                                          CC._SSPageWorkspace(_coeff(CM.QQField())))
            sq.dimZ + sq.dimB + sq.dimH
        end)

        pushrow("exact_page_term_r2_ambient", () -> begin
            sq = CC._ss_compute_page_term_ambient(DC, Tot_ref, blocks_ref, :vertical, 2,
                                                  aterm_ref, bterm_ref,
                                                  CC._SSPageWorkspace(_coeff(CM.QQField())))
            sq.dimZ + sq.dimB + sq.dimH
        end)

        pushrow("exact_page_data_r2_exact", () -> begin
            pg = CC._ss_compute_page_data(DC, Tot_ref, blocks_ref, :vertical, 2)
            _digest_terms(pg.spaces)
        end)

        pushrow("exact_page_data_r2_ambient", () -> begin
            pg = CC._ss_compute_page_data_ambient(DC, Tot_ref, blocks_ref, :vertical, 2)
            _digest_terms(pg.spaces)
        end)

        pushrow("exact_differential_r2_exact", () -> begin
            pg = CC._ss_compute_page_data(DC, Tot_ref, blocks_ref, :vertical, 2)
            _digest_diff(CC._ss_compute_differential(DC, Tot_ref, :vertical, pg, 2))
        end)

        pushrow("exact_differential_r2_ambient", () -> begin
            old = CC._spectral_exact_diff_mode[]
            try
                CC._spectral_exact_diff_mode[] = :ambient
                pg = CC._ss_compute_page_data(DC, Tot_ref, blocks_ref, :vertical, 2)
                _digest_diff(CC._ss_compute_differential(DC, Tot_ref, :vertical, pg, 2))
            finally
                CC._spectral_exact_diff_mode[] = old
            end
        end)

        _write_csv(out, rows)
        println("Wrote ", out)
        flush(stdout)
    finally
        CC._spectral_exact_filtimg_basis_mode[] = old_b
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(out=_parse_path_arg(ARGS, "--out", joinpath(@__DIR__, "_tmp_chain_complexes_exact_spectral_ab.csv")),
         seed=_parse_int_arg(ARGS, "--seed", 11),
         density=_parse_float_arg(ARGS, "--density", 0.30),
         na=_parse_int_arg(ARGS, "--na", 2),
         nb=_parse_int_arg(ARGS, "--nb", 2),
         dmin=_parse_int_arg(ARGS, "--dmin", 1),
         dmax=_parse_int_arg(ARGS, "--dmax", 2),
         verify=_parse_bool_arg(ARGS, "--verify", true),
         reps=_parse_int_arg_min(ARGS, "--reps", 3, 1))
end
