# -----------------------------------------------------------------------------
# field_linalg/backend_routing.jl
#
# Scope:
#   Backend routing for individual operations together with Nemo conversion and
#   backend-wrapper helpers used across the FieldLinAlg kernels.
# Owns:
#   - `_choose_linalg_backend` and operation-specific routing policy,
#   - backend conversion/wrapper helpers,
#   - sparse/dense shape heuristics that decide which kernel family runs.
# Does not own:
#   - threshold state (`thresholds.jl`),
#   - actual elimination/solve kernels,
#   - public API wrappers.
# Depends on:
#   - `thresholds.jl` for all routing thresholds and backend traits,
#   - later engine files for the kernels selected here.
# -----------------------------------------------------------------------------

function _choose_linalg_backend(field::AbstractCoeffField, A; op::Symbol=:rank, backend::Symbol=:auto)
    backend != :auto && return backend
    if field isa QQField
        m, n = size(A)
        shape = _qq_shape_bucket(m, n)
        work = m * n
        if _is_sparse_like(A)
            nnzA = _sparse_nnz(A)
            dens = work == 0 ? 0.0 : (nnzA / work)
            dens_bucket = _qq_sparse_density_bucket(_qq_sparse_solve_fill(A))
            if op == :colspace
                if _qq_sparse_colspace_use_julia(shape, dens, nnzA, work)
                    return :julia_sparse
                end
                return _have_nemo() ? :nemo : :julia_sparse
            end
            if op == :nullspace
                if _qq_sparse_nullspace_use_julia(shape, dens, nnzA)
                    return :julia_sparse
                end
                if _have_nemo() && work >= _qq_nemo_threshold(:nullspace, shape)
                    return :nemo
                end
                return :julia_sparse
            end
            if op == :solve
                solve_thr = _qq_sparse_solve_threshold(shape, dens_bucket)
                use_ge = _qq_sparse_solve_use_ge(shape, dens_bucket)
                choose_nemo = use_ge ? (nnzA >= solve_thr) : (nnzA <= solve_thr)
                if !choose_nemo && _qq_sparse_solve_use_julia(shape, dens, nnzA)
                    return :julia_sparse
                end
                if _have_nemo() && choose_nemo
                    return :nemo
                end
                return :julia_sparse
            end
            if op == :rank && _qq_sparse_rank_use_julia(shape, dens, nnzA)
                return :julia_sparse
            end
            if dens < 0.10 && work <= 2_000
                return :julia_sparse
            end
            if _have_nemo() && work >= _qq_nemo_threshold(:rank, shape)
                return :nemo
            end
            return :julia_sparse
        end
        if op == :nullspace
            if _have_nemo() && work >= _qq_nemo_threshold(:nullspace, shape)
                return :nemo
            end
            if work >= _qq_modular_threshold(:nullspace, shape)
                return :modular
            end
        end
        if op == :solve
            if _have_nemo() && work >= _qq_nemo_threshold(:solve, shape)
                return :nemo
            end
            if work >= _qq_modular_threshold(:solve, shape)
                return :modular
            end
        end
        if _have_nemo() && work >= _qq_nemo_threshold(:rank, shape)
            return :nemo
        end
        return :julia_exact
    end
    if field isa PrimeField && field.p == 2
        return :f2_bit
    end
    if field isa PrimeField && field.p == 3
        return :f3_table
    end
    if field isa PrimeField && field.p > 3
        if _is_sparse_like(A)
            return :fp_sparse
        end
        if _have_nemo() && _nemo_dense_compatible(A)
            work = size(A, 1) * size(A, 2)
            if op == :rank && work >= FP_NEMO_RANK_THRESHOLD[]
                return :nemo
            elseif op == :nullspace && work >= FP_NEMO_NULLSPACE_THRESHOLD[]
                return :nemo
            elseif op == :solve && work >= FP_NEMO_SOLVE_THRESHOLD[]
                return :nemo
            end
        end
        return :julia_exact
    end
    if field isa RealField
        if _is_sparse_like(A)
            m, n = size(A)
            if op == :nullspace &&
               _have_svds_backend() &&
               min(m, n) >= FLOAT_SPARSE_SVDS_MIN_DIM[] &&
               _sparse_nnz(A) >= FLOAT_SPARSE_SVDS_MIN_NNZ[]
                return :float_sparse_svds
            end
            return :float_sparse_qr
        end
        if op == :nullspace && size(A, 1) * size(A, 2) >= FLOAT_NULLSPACE_SVD_THRESHOLD[]
            return :float_dense_svd
        end
        return :float_dense_qr
    end
    return :julia_exact
end

"""
    _explain_backend_choice(field, A; op=:rank, backend=:auto)

Internal debugging helper for FieldLinAlg backend selection.

Returns a compact named tuple describing:
- the chosen backend,
- the shape bucket used for QQ routing (when applicable),
- sparse-density / nnz summary,
- thresholds consulted,
- a human-readable reason for the choice.
"""
function _explain_backend_choice(field::AbstractCoeffField, A; op::Symbol=:rank, backend::Symbol=:auto)
    chosen = _choose_linalg_backend(field, A; op=op, backend=backend)
    m, n = size(A)
    work = m * n
    sparse_like = _is_sparse_like(A)
    nnzA = sparse_like ? _sparse_nnz(A) : nothing
    density = sparse_like && work > 0 ? nnzA / work : nothing
    fill = sparse_like && field isa QQField ? _qq_sparse_solve_fill(A) : nothing
    shape_bucket = field isa QQField ? _qq_shape_bucket(m, n) : nothing
    density_bucket = sparse_like && field isa QQField ? _qq_sparse_density_bucket(fill) : nothing
    thresholds = NamedTuple[]
    reason = "default exact backend"

    if backend != :auto
        return (
            chosen_backend = chosen,
            shape_bucket = shape_bucket,
            density_summary = (sparse = sparse_like, nnz = nnzA, work = work, density = density, fill = fill, density_bucket = density_bucket),
            thresholds_consulted = thresholds,
            reason = "backend keyword forced to $(backend)",
        )
    end

    if field isa QQField
        if sparse_like
            if op == :colspace
                push!(thresholds, (name=:qq_sparse_colspace_rule, value=:heuristic))
                reason = chosen == :julia_sparse ?
                    "QQ sparse colspace heuristic stayed on Julia sparse path" :
                    "QQ sparse colspace heuristic chose Nemo"
            elseif op == :nullspace
                push!(thresholds, (name=:qq_nemo_nullspace_threshold, value=_qq_nemo_threshold(:nullspace, shape_bucket)))
                reason = chosen == :julia_sparse ?
                    "QQ sparse nullspace heuristic stayed on Julia sparse path" :
                    "QQ sparse nullspace crossed the Nemo threshold"
            elseif op == :solve
                solve_thr = _qq_sparse_solve_threshold(shape_bucket, density_bucket)
                push!(thresholds, (name=:qq_sparse_solve_threshold, value=solve_thr))
                push!(thresholds, (name=:qq_sparse_solve_policy_ge, value=_qq_sparse_solve_use_ge(shape_bucket, density_bucket)))
                reason = chosen == :julia_sparse ?
                    "QQ sparse solve heuristic stayed on Julia sparse path" :
                    "QQ sparse solve heuristic selected Nemo"
            else
                push!(thresholds, (name=:qq_sparse_rank_julia_rule, value=:heuristic))
                push!(thresholds, (name=:qq_nemo_rank_threshold, value=_qq_nemo_threshold(:rank, shape_bucket)))
                reason = chosen == :julia_sparse ?
                    "QQ sparse rank stayed on Julia sparse path" :
                    "QQ sparse rank crossed the Nemo threshold"
            end
        else
            if op == :nullspace
                push!(thresholds, (name=:qq_nemo_nullspace_threshold, value=_qq_nemo_threshold(:nullspace, shape_bucket)))
                push!(thresholds, (name=:qq_modular_nullspace_threshold, value=_qq_modular_threshold(:nullspace, shape_bucket)))
                reason = chosen == :nemo ? "QQ dense nullspace crossed the Nemo threshold" :
                         chosen == :modular ? "QQ dense nullspace crossed the modular threshold" :
                         "QQ dense nullspace stayed on Julia exact path"
            elseif op == :solve
                push!(thresholds, (name=:qq_nemo_solve_threshold, value=_qq_nemo_threshold(:solve, shape_bucket)))
                push!(thresholds, (name=:qq_modular_solve_threshold, value=_qq_modular_threshold(:solve, shape_bucket)))
                reason = chosen == :nemo ? "QQ dense solve crossed the Nemo threshold" :
                         chosen == :modular ? "QQ dense solve crossed the modular threshold" :
                         "QQ dense solve stayed on Julia exact path"
            else
                push!(thresholds, (name=:qq_nemo_rank_threshold, value=_qq_nemo_threshold(:rank, shape_bucket)))
                reason = chosen == :nemo ?
                    "QQ dense rank crossed the Nemo threshold" :
                    "QQ dense rank stayed on Julia exact path"
            end
        end
    elseif field isa PrimeField && field.p == 2
        reason = "F2 uses the dedicated bit-packed engine"
    elseif field isa PrimeField && field.p == 3
        reason = "F3 uses the dedicated table-based engine"
    elseif field isa PrimeField
        if sparse_like
            reason = "generic prime-field sparse input stays on the Julia sparse exact path"
        else
            push!(thresholds, (name=:fp_nemo_rank_threshold, value=FP_NEMO_RANK_THRESHOLD[]))
            push!(thresholds, (name=:fp_nemo_nullspace_threshold, value=FP_NEMO_NULLSPACE_THRESHOLD[]))
            push!(thresholds, (name=:fp_nemo_solve_threshold, value=FP_NEMO_SOLVE_THRESHOLD[]))
            reason = chosen == :nemo ?
                "generic prime-field dense input crossed the Nemo threshold" :
                "generic prime-field dense input stayed on the Julia exact path"
        end
    elseif field isa RealField
        if sparse_like
            push!(thresholds, (name=:float_sparse_svds_min_dim, value=FLOAT_SPARSE_SVDS_MIN_DIM[]))
            push!(thresholds, (name=:float_sparse_svds_min_nnz, value=FLOAT_SPARSE_SVDS_MIN_NNZ[]))
            reason = chosen == :float_sparse_svds ?
                "real sparse nullspace crossed the SVDS gate" :
                "real sparse input stayed on sparse QR / sparse direct path"
        else
            push!(thresholds, (name=:float_nullspace_svd_threshold, value=FLOAT_NULLSPACE_SVD_THRESHOLD[]))
            reason = chosen == :float_dense_svd ?
                "real dense nullspace crossed the dense-SVD threshold" :
                "real dense input stayed on the dense QR path"
        end
    end

    return (
        chosen_backend = chosen,
        shape_bucket = shape_bucket,
        density_summary = (sparse = sparse_like, nnz = nnzA, work = work, density = density, fill = fill, density_bucket = density_bucket),
        thresholds_consulted = thresholds,
        reason = reason,
    )
end

# Conversions between Matrix{QQ} and Nemo matrices.
function _to_fmpq_mat(A::AbstractMatrix{QQ})
    _bump_counter!(_QQ_TO_NEMO_CONVERSIONS)
    return Nemo.matrix(Nemo.QQ, A)
end

function _to_fmpq_mat(A::BackendMatrix{QQ})
    if _backend_kind(A) != :nemo
        return _to_fmpq_mat(_unwrap_backend_matrix(A))
    end
    payload = _backend_payload(A)
    if payload !== nothing
        _bump_counter!(_QQ_TO_NEMO_CACHE_HITS)
        return payload
    end
    M = _to_fmpq_mat(_unwrap_backend_matrix(A))
    _set_backend_payload!(A, M)
    return M
end

function _to_fmpq_mat(A::Transpose{QQ,<:BackendMatrix{QQ}})
    return transpose(_to_fmpq_mat(parent(A)))
end

function _to_fmpq_mat(A::Adjoint{QQ,<:BackendMatrix{QQ}})
    return transpose(_to_fmpq_mat(parent(A)))
end

# Conversions between Matrix{FpElem{p}} and Nemo matrices over GF(p).
function _to_nemo_fp_mat(A::AbstractMatrix{FpElem{p}}) where {p}
    _bump_counter!(_FP_TO_NEMO_CONVERSIONS)
    Fp = Nemo.GF(p)
    m, n = size(A)
    M = Matrix{Int}(undef, m, n)
    @inbounds for i in 1:m, j in 1:n
        M[i, j] = A[i, j].val
    end
    return Nemo.matrix(Fp, M)
end

function _to_nemo_fp_mat(A::BackendMatrix{FpElem{p}}) where {p}
    if _backend_kind(A) != :nemo
        return _to_nemo_fp_mat(_unwrap_backend_matrix(A))
    end
    payload = _backend_payload(A)
    if payload isa Pair
        pp = first(payload)
        if pp == p
            _bump_counter!(_FP_TO_NEMO_CACHE_HITS)
            return last(payload)
        end
    end
    M = _to_nemo_fp_mat(_unwrap_backend_matrix(A))
    _set_backend_payload!(A, p => M)
    return M
end

function _to_nemo_fp_mat(A::Transpose{FpElem{p},<:BackendMatrix{FpElem{p}}}) where {p}
    return transpose(_to_nemo_fp_mat(parent(A)))
end

function _to_nemo_fp_mat(A::Adjoint{FpElem{p},<:BackendMatrix{FpElem{p}}}) where {p}
    return transpose(_to_nemo_fp_mat(parent(A)))
end

function _from_nemo_fp_mat(M, ::Val{p}) where {p}
    _bump_counter!(_FP_FROM_NEMO_CONVERSIONS)
    m, n = size(M)
    A = Matrix{FpElem{p}}(undef, m, n)
    @inbounds for i in 1:m, j in 1:n
        x = M[i, j]
        A[i, j] = FpElem{p}(Int(Nemo.lift(Nemo.ZZ, x)))
    end
    return A
end

function _from_fmpq_mat(M)
    _bump_counter!(_QQ_FROM_NEMO_CONVERSIONS)
    m, n = size(M)
    A = Matrix{QQ}(undef, m, n)
    @inbounds for i in 1:m, j in 1:n
        x = M[i, j]
        A[i, j] = QQ(BigInt(Nemo.numerator(x)), BigInt(Nemo.denominator(x)))
    end
    return A
end

# Nemo-backed implementations for QQ.
function _nemo_rref_qq_mat(M; pivots::Bool=true)
    _, R = Nemo.rref(M)
    Rq = _from_fmpq_mat(R)

    pivs = Int[]
    m, n = size(Rq)
    @inbounds for i in 1:m
        for j in 1:n
            if Rq[i, j] != 0
                push!(pivs, j)
                break
            end
        end
    end

    return pivots ? (Rq, Tuple(pivs)) : Rq
end

_nemo_rank_qq_mat(M) = Nemo.rank(M)

function _nemo_pivots_mat(M)
    _, R = Nemo.rref(M)
    pivs = Int[]
    m, n = size(R)
    @inbounds for i in 1:m
        for j in 1:n
            if !iszero(R[i, j])
                push!(pivs, j)
                break
            end
        end
    end
    return Tuple(pivs)
end

function _nemo_nullspace_qq_mat(M)
    _, N = Nemo.nullspace(M)
    return _from_fmpq_mat(N)
end

function _nemo_rref(::QQField, A::AbstractMatrix{QQ}; pivots::Bool=true)
    return _nemo_rref_qq_mat(_to_fmpq_mat(A); pivots=pivots)
end

_nemo_rank(::QQField, A::AbstractMatrix{QQ}) = _nemo_rank_qq_mat(_to_fmpq_mat(A))

function _nemo_nullspace(::QQField, A::AbstractMatrix{QQ})
    return _nemo_nullspace_qq_mat(_to_fmpq_mat(A))
end

# Nemo-backed implementations for Fp (p > 3).
function _nemo_rref_fp_mat(F::PrimeField, M; pivots::Bool=true)
    p = F.p
    p > 3 || error("nemo_rref: only for p > 3")
    _, R = Nemo.rref(M)
    Rf = _from_nemo_fp_mat(R, Val(p))

    pivs = Int[]
    m, n = size(Rf)
    @inbounds for i in 1:m
        for j in 1:n
            if Rf[i, j].val != 0
                push!(pivs, j)
                break
            end
        end
    end
    return pivots ? (Rf, Tuple(pivs)) : Rf
end

function _nemo_rref(F::PrimeField, A::AbstractMatrix{FpElem{p}}; pivots::Bool=true) where {p}
    p > 3 || error("nemo_rref: only for p > 3")
    F.p == p || error("nemo_rref: field mismatch")
    return _nemo_rref_fp_mat(F, _to_nemo_fp_mat(A); pivots=pivots)
end

function _nemo_rank_fp_mat(F::PrimeField, M)
    p = F.p
    p > 3 || error("nemo_rank: only for p > 3")
    return Nemo.rank(M)
end

_nemo_rank(F::PrimeField, A::AbstractMatrix{FpElem{p}}) where {p} =
    (p > 3 ? (F.p == p ? _nemo_rank_fp_mat(F, _to_nemo_fp_mat(A)) : error("nemo_rank: field mismatch")) :
     error("nemo_rank: only for p > 3"))

function _nemo_nullspace_fp_mat(F::PrimeField, M)
    p = F.p
    p > 3 || error("nemo_nullspace: only for p > 3")
    _, N = Nemo.nullspace(M)
    return _from_nemo_fp_mat(N, Val(p))
end

function _nemo_nullspace(F::PrimeField, A::AbstractMatrix{FpElem{p}}) where {p}
    p > 3 || error("nemo_nullspace: only for p > 3")
    F.p == p || error("nemo_nullspace: field mismatch")
    M = _to_nemo_fp_mat(A)
    return _nemo_nullspace_fp_mat(F, M)
end

@inline _nemo_matrix(::NemoMatrixBackend, ::QQField, A::AbstractMatrix{QQ}) = _to_fmpq_mat(A)
@inline function _nemo_matrix(::NemoMatrixBackend, F::PrimeField, A::AbstractMatrix{FpElem{p}}) where {p}
    p > 3 || error("Nemo backend matrix conversion is only defined for p > 3")
    F.p == p || error("Field mismatch for Nemo backend matrix conversion")
    return _to_nemo_fp_mat(A)
end
