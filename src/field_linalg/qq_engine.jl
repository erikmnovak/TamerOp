# -----------------------------------------------------------------------------
# field_linalg/qq_engine.jl
#
# Scope:
#   Exact rational linear algebra kernels, sparse/full-column solve helpers,
#   modular accelerators, and QQ-specific hot paths for TamerOp.FieldLinAlg.
# Owns:
#   - QQ rank/rref/nullspace/solve kernels,
#   - Nemo/modular QQ accelerators and crossover helpers,
#   - QQ-specific restricted-word and full-column factor paths.
# Does not own:
#   - public API wrappers,
#   - backend-choice policy,
#   - non-QQ kernels.
# Depends on:
#   - `thresholds.jl` and `backend_routing.jl` for backend choice,
#   - `sparse_rref.jl` for shared sparse exact elimination support.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# QQ engine (exact rational arithmetic)
# -----------------------------------------------------------------------------

# Exact linear algebra over QQ (pure Julia). Nemo acceleration is selected via
# _choose_linalg_backend and dispatched through _nemo_* methods in this module.
# Keep this hook for optional backend matrix wrappers.
@inline _to_backend_matrix(A) = A

function _rref_backend(A::AbstractMatrix{QQ}; pivots::Bool=true)
    M = Matrix{QQ}(A)
    m, n = size(M)
    pivs = Int[]
    row = 1

    for col in 1:n
        pivrow = 0
        @inbounds for r in row:m
            if M[r, col] != 0
                pivrow = r
                break
            end
        end
        pivrow == 0 && continue

        push!(pivs, col)

        if pivrow != row
            @inbounds for j in 1:n
                M[row, j], M[pivrow, j] = M[pivrow, j], M[row, j]
            end
        end

        pivval = M[row, col]
        @inbounds for j in 1:n
            M[row, j] /= pivval
        end

        @inbounds for r in 1:m
            r == row && continue
            fac = M[r, col]
            fac == 0 && continue
            for j in 1:n
                M[r, j] -= fac * M[row, j]
            end
        end

        row += 1
        row > m && break
    end

    return pivots ? (M, Tuple(pivs)) : M
end

"""
    _rrefQQ(A::AbstractMatrix{QQ}; pivots::Bool=true)

Return the row-reduced echelon form of `A`. If `pivots=true`, also return the
pivot column indices as a tuple.
"""
function _rrefQQ(A::AbstractMatrix{QQ}; pivots::Bool=true)
    return _rref_backend(_to_backend_matrix(A); pivots=pivots)
end

_rank_backend(A::AbstractMatrix{QQ}) = length(last(_rrefQQ(A; pivots=true)))

_rankQQ(A::AbstractMatrix{QQ}) = _rank_backend(_to_backend_matrix(A))
_rankQQ(A::AbstractMatrix{Float64}) = _rankQQ(Matrix{QQ}(A))

function _nullspace_backend(A::AbstractMatrix{QQ})
    R, pivs = _rrefQQ(A; pivots=true)
    m, n = size(R)

    # pivs is increasing; compute free cols without allocating a Set.
    free = Int[]
    piv_i = 1
    npiv = length(pivs)
    @inbounds for j in 1:n
        if piv_i <= npiv && pivs[piv_i] == j
            piv_i += 1
        else
            push!(free, j)
        end
    end

    isempty(free) && return zeros(QQ, n, 0)

    B = zeros(QQ, n, length(free))
    @inbounds for (k, jf) in enumerate(free)
        B[jf, k] = 1
        for (i, jp) in enumerate(pivs)
            B[jp, k] = -R[i, jf]
        end
    end
    return B
end

_nullspaceQQ(A::AbstractMatrix{QQ}) = _nullspace_backend(_to_backend_matrix(A))

function _colspaceQQ(A::AbstractMatrix{QQ})
    cols = collect(_pivot_columnsQQ(A))
    return A[:, cols]
end


# ---------------------------------------------------------------
# Reusable solvers for full-column-rank systems B * X = Y
# ---------------------------------------------------------------

"""
    FullColumnFactor{K}

Reusable data for solving `B * X = Y` when `B` has full column rank.

We select row indices `rows` such that `B[rows, :]` is an invertible n x n minor
(where n = size(B,2)), and precompute its inverse `invB`. Then for any RHS `Y`
in the column space of `B`, the unique solution is

    X = invB * Y[rows, :]

Notes
-----
* The factor does NOT store B, so it is safe to cache in a WeakKeyDict keyed by B.
* If B is mutated after factorization, the factor becomes invalid and must be rebuilt.
"""
struct FullColumnFactor{K}
    rows::Vector{Int}
    invB::Matrix{K}
end

struct NemoFullColumnFactorQQ
    rows::Vector{Int}
    invB::Any
end

struct NemoFullColumnFactorFp{p}
    rows::Vector{Int}
    invB::Any
end

const _QQ_FACTOR_GATHER_MIN_RHS = Ref(9)

@inline function _gather_rows!(dest::AbstractMatrix{QQ}, Y::AbstractMatrix{<:QQ}, rows::AbstractVector{Int})
    n = length(rows)
    rhs = size(Y, 2)
    size(dest, 1) == n || throw(DimensionMismatch("gather destination has wrong row count"))
    size(dest, 2) == rhs || throw(DimensionMismatch("gather destination has wrong column count"))
    @inbounds for j in 1:rhs
        for i in 1:n
            dest[i, j] = Y[rows[i], j]
        end
    end
    return dest
end

@inline function _gather_rows!(dest::AbstractVector{QQ}, Y::AbstractVector{<:QQ}, rows::AbstractVector{Int})
    n = length(rows)
    length(dest) == n || throw(DimensionMismatch("gather destination has wrong length"))
    @inbounds for i in 1:n
        dest[i] = Y[rows[i]]
    end
    return dest
end

# Cache of factors keyed by the left matrix B.
# IMPORTANT: values do not reference the key, so WeakKeyDict works correctly.
const _FULLCOLUMN_FACTOR_CACHE = WeakKeyDict{Any,Any}()
const _NEMO_FULLCOLUMN_FACTOR_CACHE_QQ = WeakKeyDict{Any,Any}()
const _NEMO_FULLCOLUMN_FACTOR_CACHE_FP = WeakKeyDict{Any,Any}()

@inline _can_weak_cache_key(x) = Base.ismutabletype(typeof(x))

# ---------------------------------------------------------------
# Pivot-column detection for QQ matrices (dense or sparse)
# ---------------------------------------------------------------

# Dense pivot columns via _rrefQQ
function _pivot_columnsQQ(A::AbstractMatrix{QQ})
    _, pivs = _rrefQQ(A)
    return pivs
end

# Sparse pivot columns via sparse RREF streaming
# Sparse pivot columns via sparse RREF streaming
function _pivot_columnsQQ(A::SparseMatrixCSC{QQ,Int})
    m, n = size(A)
    if m == 0 || n == 0
        return Int[]
    end
    R = _SparseRREF{QQ}(n)
    rows = _sparse_rows(A)
    maxrank = min(m, n)
    for i in 1:m
        _sparse_rref_push_homogeneous!(R, rows[i])
        if length(R.pivot_cols) == maxrank
            break
        end
    end
    return Tuple(sort!(copy(R.pivot_cols)))
end


# Avoid densification for transpose/adjoint sparse wrappers
_pivot_columnsQQ(A::Transpose{QQ,<:SparseMatrixCSC{QQ,Int}}) = _pivot_columnsQQ(sparse(A))
_pivot_columnsQQ(A::Adjoint{QQ,<:SparseMatrixCSC{QQ,Int}})  = _pivot_columnsQQ(sparse(A))

"""
    _factor_fullcolumnQQ(B::AbstractMatrix{<:QQ}) -> FullColumnFactor{QQ}

Build reusable factorization data for solving `B * X = Y` when B has full column rank.

Most users should call `_solve_fullcolumnQQ(B,Y)`; caching is automatic for mutable matrices.
"""
function _factor_fullcolumnQQ(B::AbstractMatrix{<:QQ})::FullColumnFactor{QQ}
    m, n = size(B)
    n == 0 && return FullColumnFactor{QQ}(Int[], Matrix{QQ}(I, 0, 0))

    rows = collect(_pivot_columnsQQ(transpose(B)))
    if length(rows) != n
        error("_solve_fullcolumnQQ: expected full column rank, got rank $(length(rows)) < $n")
    end

    Bsub = Matrix{QQ}(B[rows, :])
    invB = _solve_fullcolumn_rrefQQ(Bsub, Matrix{QQ}(I, n, n))
    return FullColumnFactor{QQ}(rows, invB)
end

# Ground-truth RREF solver, always exact, no caching
function _solve_fullcolumn_rrefQQ(B::AbstractMatrix{<:QQ}, Y::AbstractVecOrMat{<:QQ};
                                  check_rhs::Bool=true)
    want_vec = false
    Ymat = Y
    if Y isa AbstractVector
        want_vec = true
        Ymat = reshape(Y, :, 1)
    end

    m, n = size(B)
    size(Ymat, 1) == m || throw(DimensionMismatch("B and Y must have same row count"))

    if n == 0
        return want_vec ? QQ[] : zeros(QQ, 0, size(Ymat, 2))
    end

    By = hcat(Matrix{QQ}(B), Matrix{QQ}(Ymat))
    R, pivs = _rrefQQ(By)

    for j in 1:n
        (j in pivs) || error("expected full column rank; missing pivot in column $j")
    end
    if check_rhs
        for pj in pivs
            pj > n && error("right-hand side is not in column space of B")
        end
    end

    X = R[1:n, n+1:end]
    return want_vec ? vec(X) : X
end

# Fast solve using factor data
function _solve_fullcolumn_factorQQ(B::AbstractMatrix{<:QQ}, fac::FullColumnFactor{QQ},
                                    Y::AbstractVecOrMat{<:QQ};
                                    check_rhs::Bool=true)
    m, n = size(B)
    length(fac.rows) == n || error("stale FullColumnFactor: wrong row set length")
    if Y isa AbstractVector
        length(Y) == m || throw(DimensionMismatch("B and Y must have same row count"))
        ysub = Vector{QQ}(undef, n)
        _gather_rows!(ysub, Y, fac.rows)
        x = fac.invB * ysub
        if check_rhs && !_verify_solveQQ(B, x, Y)
            error("right-hand side is not in column space of B")
        end
        return x
    end

    Ymat = Y
    size(Ymat, 1) == m || throw(DimensionMismatch("B and Y must have same row count"))
    rhs = size(Ymat, 2)
    X = if rhs < _QQ_FACTOR_GATHER_MIN_RHS[]
        fac.invB * view(Ymat, fac.rows, :)
    else
        Ysub = Matrix{QQ}(undef, n, rhs)
        _gather_rows!(Ysub, Ymat, fac.rows)
        Xtmp = Matrix{QQ}(undef, n, rhs)
        mul!(Xtmp, fac.invB, Ysub)
        Xtmp
    end

    if check_rhs && !_verify_solveQQ(B, X, Ymat)
        error("right-hand side is not in column space of B")
    end

    return X
end

"""
    _solve_fullcolumnQQ(B, Y; cache=true, factor=nothing, check_rhs=true)

Solve `B*X = Y` over QQ under the assumption that B has full column rank.

If cache=true, a reusable factor is used; for mutable B (e.g. Matrix) it is cached
so repeated solves do not redo elimination.
"""
function _solve_fullcolumnQQ(B::AbstractMatrix{<:QQ}, Y::AbstractVecOrMat{<:QQ};
                            cache::Bool=true,
                            factor::Union{Nothing,FullColumnFactor{QQ}}=nothing,
                            check_rhs::Bool=true)
    if factor !== nothing
        return _solve_fullcolumn_factorQQ(B, factor, Y; check_rhs=check_rhs)
    end
    if cache
        fac = if _can_weak_cache_key(B)
            get!(_FULLCOLUMN_FACTOR_CACHE, B) do
                _factor_fullcolumnQQ(B)
            end
        else
            _factor_fullcolumnQQ(B)
        end
        return _solve_fullcolumn_factorQQ(B, fac, Y; check_rhs=check_rhs)
    end

    return _solve_fullcolumn_rrefQQ(B, Y; check_rhs=check_rhs)
end

@inline function _choose_solve_backend(field::QQField, B;
                                       backend::Symbol=:auto, factor=nothing)::Symbol
    backend != :auto && return _choose_linalg_backend(field, B; op=:solve, backend=backend)
    if factor isa NemoFullColumnFactorQQ
        return :nemo
    elseif factor isa FullColumnFactor{QQ}
        return _is_sparse_like(B) ? :julia_sparse : :julia_exact
    end
    return _choose_linalg_backend(field, B; op=:solve, backend=backend)
end

@inline function _choose_solve_backend(field::AbstractCoeffField, B;
                                       backend::Symbol=:auto, factor=nothing)::Symbol
    return _choose_linalg_backend(field, B; op=:solve, backend=backend)
end

"""
    _clear_fullcolumn_cache!()

Drop cached FullColumnFactor objects (useful in long sessions).
"""
function _clear_fullcolumn_cache!()
    empty!(_FULLCOLUMN_FACTOR_CACHE)
    empty!(_NEMO_FULLCOLUMN_FACTOR_CACHE_QQ)
    empty!(_NEMO_FULLCOLUMN_FACTOR_CACHE_FP)
    empty!(_FLOAT_SPARSE_FACTOR_CACHE)
    return nothing
end

function _factor_fullcolumn_nemoQQ(B::AbstractMatrix{<:QQ})::NemoFullColumnFactorQQ
    m, n = size(B)
    n == 0 && return NemoFullColumnFactorQQ(Int[], Nemo.matrix(Nemo.QQ, Matrix{QQ}(I, 0, 0)))

    _, rows_tup = _nemo_rref(QQField(), transpose(B); pivots=true)
    rows = collect(rows_tup)
    if length(rows) != n
        error("solve_fullcolumn_nemoQQ: expected full column rank, got rank $(length(rows)) < $n")
    end

    Bsub = Matrix{QQ}(B[rows, :])
    invB = inv(_to_fmpq_mat(Bsub))
    return NemoFullColumnFactorQQ(rows, invB)
end

function _solve_fullcolumn_nemoQQ(B::AbstractMatrix{<:QQ}, Y::AbstractVecOrMat{<:QQ};
                                  check_rhs::Bool=true, cache::Bool=true,
                                  factor::Union{Nothing,FullColumnFactor{QQ},NemoFullColumnFactorQQ}=nothing)
    if factor isa FullColumnFactor{QQ}
        return _solve_fullcolumn_factorQQ(B, factor, Y; check_rhs=check_rhs)
    end

    want_vec = false
    Ymat = Y
    if Y isa AbstractVector
        want_vec = true
        Ymat = reshape(Y, :, 1)
    end

    m, n = size(B)
    size(Ymat, 1) == m || throw(DimensionMismatch("B and Y must have same row count"))

    fac = factor
    if fac === nothing
        if cache && _can_weak_cache_key(B)
            fac = get!(_NEMO_FULLCOLUMN_FACTOR_CACHE_QQ, B) do
                _factor_fullcolumn_nemoQQ(B)
            end
        else
            fac = _factor_fullcolumn_nemoQQ(B)
        end
    end

    Ysub = if want_vec
        reshape(_gather_rows!(Vector{QQ}(undef, n), vec(Ymat), fac.rows), :, 1)
    elseif size(Ymat, 2) < _QQ_FACTOR_GATHER_MIN_RHS[]
        Matrix{QQ}(view(Ymat, fac.rows, :))
    else
        tmp = Matrix{QQ}(undef, n, size(Ymat, 2))
        _gather_rows!(tmp, Ymat, fac.rows)
    end
    Xn = fac.invB * _to_fmpq_mat(Ysub)
    X = _from_fmpq_mat(Xn)

    if check_rhs && !_verify_solveQQ(B, X, Ymat)
        error("solve_fullcolumn_nemoQQ: RHS check failed")
    end

    return want_vec ? vec(X) : X
end

function _factor_fullcolumn_nemo_fp(F::PrimeField, B::AbstractMatrix{FpElem{p}}) where {p}
    p > 3 || error("solve_fullcolumn_nemo_fp: only for p > 3")
    F.p == p || error("solve_fullcolumn_nemo_fp: field mismatch")

    m, n = size(B)
    n == 0 && return NemoFullColumnFactorFp{p}(Int[], Nemo.matrix(Nemo.GF(p), zeros(Int, 0, 0)))

    _, rows_tup = _nemo_rref(F, transpose(B); pivots=true)
    rows = collect(rows_tup)
    if length(rows) != n
        error("solve_fullcolumn_nemo_fp: expected full column rank, got rank $(length(rows)) < $n")
    end

    Bsub = Matrix{FpElem{p}}(B[rows, :])
    invB = inv(_to_nemo_fp_mat(Bsub))
    return NemoFullColumnFactorFp{p}(rows, invB)
end

function _solve_fullcolumn_nemo_fp(F::PrimeField, B::AbstractMatrix{FpElem{p}},
                                   Y::AbstractVecOrMat{FpElem{p}};
                                   check_rhs::Bool=true, cache::Bool=true,
                                   factor=nothing) where {p}
    p > 3 || error("solve_fullcolumn_nemo_fp: only for p > 3")
    F.p == p || error("solve_fullcolumn_nemo_fp: field mismatch")

    want_vec = false
    Ymat = Y
    if Y isa AbstractVector
        want_vec = true
        Ymat = reshape(Y, :, 1)
    end

    m, n = size(B)
    size(Ymat, 1) == m || throw(DimensionMismatch("B and Y must have same row count"))

    fac = factor
    if fac !== nothing && !(fac isa NemoFullColumnFactorFp{p})
        error("solve_fullcolumn_nemo_fp: incompatible factor type $(typeof(fac))")
    end
    if fac === nothing
        if cache && _can_weak_cache_key(B)
            fac = get!(_NEMO_FULLCOLUMN_FACTOR_CACHE_FP, B) do
                _factor_fullcolumn_nemo_fp(F, B)
            end
        else
            fac = _factor_fullcolumn_nemo_fp(F, B)
        end
    end

    Ysub = Matrix{FpElem{p}}(Ymat[fac.rows, :])
    Xn = fac.invB * _to_nemo_fp_mat(Ysub)
    X = _from_nemo_fp_mat(Xn, Val(p))

    if check_rhs && (B * X != Ymat)
        error("solve_fullcolumn_nemo_fp: RHS check failed")
    end

    return want_vec ? vec(X) : X
end


# ---------------------------------------------------------------------------
# Sparse elimination utilities for exact computations without densifying matrices.
#
# Motivation:
# Many "naturality systems" in this code base (e.g. Hom(M,N) and various lifting
# constraints) produce huge linear systems that are *extremely sparse* but were
# previously assembled into dense QQ matrices.  That is catastrophic for both
# memory and time.
#
# The routines below allow us to stream constraint rows one-at-a-time into an
# exact reduced-row-echelon (RREF) basis of the row space, represented as:
#
#   pivots[pivot_col] = Dict{Int,QQ}  (a pivot row with pivot entry 1 at pivot_col)
#
# Rows are sparse Dicts keyed by column index.  This makes memory proportional
# to the number of *nonzero* coefficients encountered, not to nvars*neqs.
#
# These helpers are intentionally internal (underscored) but documented because
# they are used across multiple modules (DerivedFunctors, ChainComplexes, etc.).
#
# The earlier implementation used nested Dict{Int, Dict{Int,QQ}} structures to
# represent sparse rows.  This is convenient but very allocation-heavy and
# cache-unfriendly.  The code below replaces that representation with:
#
#   * SparseRow{K}: sorted (col, val) vectors for each row.
#   * _SparseRREF{K}: a streaming RREF builder using merge-based sparse axpy.
#
# This layer is internal but sits on several hot paths (Hom computations,
# derived-functor lifting, and sparse linear solves).  Keeping it allocation-
# light has an outsized impact on overall library performance.
# ---------------------------------------------------------------------------



function _rankQQ(A::SparseMatrixCSC{QQ,Int})
    m, n = size(A)
    if m == 0 || n == 0
        return 0
    end

    R = _SparseRREF{QQ}(n)
    rows = _sparse_rows(A)

    maxrank = min(m, n)
    for i in 1:m
        _sparse_rref_push_homogeneous!(R, rows[i])
        if _rref_rank(R) == maxrank
            break
        end
    end
    return _rref_rank(R)
end

@inline function _modp_qQ(q::QQ, p::Int)::Int
    q == 0 && return 0
    den = mod(denominator(q), p)
    den == 0 && throw(DomainError(q, "denominator not invertible modulo $p"))
    num = mod(numerator(q), p)
    v = num * invmod(den, p)
    v %= p
    v < 0 && (v += p)
    return Int(v)
end

function _rref_modp_dense(A::AbstractMatrix{QQ}, p::Int)
    m, n = size(A)
    M = Matrix{Int}(undef, m, n)
    @inbounds for j in 1:n
        for i in 1:m
            M[i, j] = _modp_qQ(A[i, j], p)
        end
    end

    pivs = Int[]
    row = 1
    @inbounds for col in 1:n
        piv = row
        while piv <= m && M[piv, col] == 0
            piv += 1
        end
        piv > m && continue

        if piv != row
            for j in col:n
                M[row, j], M[piv, j] = M[piv, j], M[row, j]
            end
        end

        invp = invmod(M[row, col], p)
        for j in col:n
            v = M[row, j] * invp
            v %= p
            v < 0 && (v += p)
            M[row, j] = v
        end

        for i in 1:m
            i == row && continue
            f = M[i, col]
            f == 0 && continue
            for j in col:n
                v = M[i, j] - f * M[row, j]
                v %= p
                v < 0 && (v += p)
                M[i, j] = v
            end
        end

        push!(pivs, col)
        row += 1
        row > m && break
    end

    return M, pivs
end

function _nullspace_modp_from_rref(R::Matrix{Int}, pivs, n::Int, p::Int)
    npiv = length(pivs)
    free = Int[]
    piv_i = 1
    @inbounds for j in 1:n
        if piv_i <= npiv && pivs[piv_i] == j
            piv_i += 1
        else
            push!(free, j)
        end
    end
    isempty(free) && return Matrix{Int}(undef, n, 0)

    B = zeros(Int, n, length(free))
    @inbounds for (k, jf) in enumerate(free)
        B[jf, k] = 1
        for (i, jp) in enumerate(pivs)
            v = -R[i, jf]
            v %= p
            v < 0 && (v += p)
            B[jp, k] = v
        end
    end
    return B
end

function _solve_fullcolumn_modp(B::AbstractMatrix{QQ}, Y::AbstractVecOrMat{QQ}, p::Int)
    want_vec = false
    Ymat = Y
    if Y isa AbstractVector
        want_vec = true
        Ymat = reshape(Y, :, 1)
    end

    m, n = size(B)
    size(Ymat, 1) == m || throw(DimensionMismatch("B and Y must have same row count"))

    By = hcat(B, Ymat)
    R, pivs = _rref_modp_dense(By, p)
    for j in 1:n
        (j in pivs) || error("expected full column rank; missing pivot in column $j (mod $p)")
    end
    for pj in pivs
        pj > n && error("right-hand side is not in column space of B (mod $p)")
    end

    X = R[1:n, n+1:end]
    return want_vec ? vec(X) : X
end

function _nullspaceQQ(A::SparseMatrixCSC{QQ,Int})
    m, n = size(A)
    R = _SparseRREF{QQ}(n)
    rows = _sparse_rows(A)
    for i in 1:m
        _sparse_rref_push_homogeneous!(R, rows[i])
    end
    return _nullspace_from_pivots(R, n)
end

# Avoid densification for transpose/adjoint sparse wrappers
_nullspaceQQ(A::Transpose{QQ,<:SparseMatrixCSC{QQ,Int}}) = _nullspaceQQ(sparse(A))
_nullspaceQQ(A::Adjoint{QQ,<:SparseMatrixCSC{QQ,Int}})  = _nullspaceQQ(sparse(A))
_rankQQ(A::Transpose{QQ,<:SparseMatrixCSC{QQ,Int}}) = _rankQQ(sparse(A))
_rankQQ(A::Adjoint{QQ,<:SparseMatrixCSC{QQ,Int}})  = _rankQQ(sparse(A))

function _rankQQ_restricted(
    A::SparseMatrixCSC{QQ,Int},
    rows::AbstractVector{Int},
    cols::AbstractVector{Int};
    check::Bool=false,
)::Int
    nr = length(rows)
    nc = length(cols)
    if nr == 0 || nc == 0
        return 0
    end

    if check
        m, n = size(A)
        for r in rows
            @assert 1 <= r <= m
        end
        for c in cols
            @assert 1 <= c <= n
        end
    end

    m, _ = size(A)
    loc = _build_row_locator(rows, m)

    row_idx = [Int[] for _ in 1:nr]
    row_val = [QQ[] for _ in 1:nr]

    # Iterate over selected columns once. We remap columns to local indices
    # 1..nc so the RREF builder does not need to allocate pivot_pos of length
    # size(A,2) when only a small subset is used.
    @inbounds for (jloc, col) in enumerate(cols)
        for ptr in A.colptr[col]:(A.colptr[col + 1] - 1)
            r = A.rowval[ptr]
            i = _row_lookup(loc, r)
            if i != 0
                v = A.nzval[ptr]
                if !iszero(v)
                    push!(row_idx[i], jloc)
                    push!(row_val[i], v)
                end
            end
        end
    end

    R = _SparseRREF{QQ}(nc)
    maxrank = min(nr, nc)
    for i in 1:nr
        _sparse_rref_push_homogeneous!(R, SparseRow{QQ}(row_idx[i], row_val[i]))
        if _rref_rank(R) == maxrank
            break
        end
    end
    return _rref_rank(R)
end

function _rank_restricted_sparse_generic(
    A::SparseMatrixCSC{K,Int},
    rows::AbstractVector{Int},
    cols::AbstractVector{Int};
    check::Bool=false,
)::Int where {K}
    nr = length(rows)
    nc = length(cols)
    if nr == 0 || nc == 0
        return 0
    end

    if check
        m, n = size(A)
        for r in rows
            @assert 1 <= r <= m
        end
        for c in cols
            @assert 1 <= c <= n
        end
    end

    m, _ = size(A)
    loc = _build_row_locator(rows, m)

    row_idx = [Int[] for _ in 1:nr]
    row_val = [K[] for _ in 1:nr]

    @inbounds for (jloc, col) in enumerate(cols)
        for ptr in A.colptr[col]:(A.colptr[col + 1] - 1)
            r = A.rowval[ptr]
            i = _row_lookup(loc, r)
            if i != 0
                v = A.nzval[ptr]
                if !iszero(v)
                    push!(row_idx[i], jloc)
                    push!(row_val[i], v)
                end
            end
        end
    end

    R = _SparseRREF{K}(nc)
    maxrank = min(nr, nc)
    for i in 1:nr
        _sparse_rref_push_homogeneous!(R, SparseRow{K}(row_idx[i], row_val[i]))
        if _rref_rank(R) == maxrank
            break
        end
    end
    return _rref_rank(R)
end

function _sparse_extract_restricted(
    A::SparseMatrixCSC{T,Int},
    rows::AbstractVector{Int},
    cols::AbstractVector{Int};
    check::Bool=false,
) where {T}
    nr = length(rows)
    nc = length(cols)
    if nr == 0 || nc == 0
        return spzeros(T, nr, nc)
    end

    if check
        m, n = size(A)
        for r in rows
            @assert 1 <= r <= m
        end
        for c in cols
            @assert 1 <= c <= n
        end
    end

    m, _ = size(A)
    loc = _build_row_locator(rows, m)

    I = Int[]
    J = Int[]
    V = T[]
    sizehint!(I, nnz(A))
    sizehint!(J, nnz(A))
    sizehint!(V, nnz(A))

    @inbounds for (jloc, col) in enumerate(cols)
        for ptr in A.colptr[col]:(A.colptr[col + 1] - 1)
            r = A.rowval[ptr]
            i = _row_lookup(loc, r)
            if i != 0
                v = A.nzval[ptr]
                if !iszero(v)
                    push!(I, i)
                    push!(J, jloc)
                    push!(V, v)
                end
            end
        end
    end

    return sparse(I, J, V, nr, nc)
end

function _rank_restricted_float_sparse(
    F::RealField,
    A::SparseMatrixCSC,
    rows::AbstractVector{Int},
    cols::AbstractVector{Int};
    check::Bool=false,
)
    nr = length(rows)
    nc = length(cols)
    if nr == 0 || nc == 0
        return 0
    end
    S = _sparse_extract_restricted(A, rows, cols; check=check)
    R = qr(S).R
    tol = _float_tol(F, S)
    return count(x -> abs(x) > tol, diag(R))
end


"""
    _rank_modp_sparse(A::SparseMatrixCSC{QQ}, p::Int) -> Int

Compute the rank of a sparse QQ-matrix after reducing coefficients modulo `p`.

This is intended as a *fast certificate/diagnostic* tool:
- For matrices with integer coefficients, _rank_modp_sparse(A,p) <= _rankQQ(A).
- If _rank_modp_sparse(A,p) == ncols(A), then A has full column rank over QQ.

Notes
- Choose `p` as a reasonably large prime (e.g. around 1e6) to reduce the chance
  of accidental rank drops.
- If any denominator in A is divisible by p, reduction is undefined and this
  function throws DomainError.
"""
function _rank_modp_sparse(A::SparseMatrixCSC{QQ,Int}, p::Int)::Int
    m, n = size(A)
    if m == 0 || n == 0
        return 0
    end

    # Build sparse rows over Int mod p.
    row_nnz = zeros(Int, m)
    @inbounds for col in 1:n
        for ptr in A.colptr[col]:(A.colptr[col + 1] - 1)
            row_nnz[A.rowval[ptr]] += 1
        end
    end

    rows_idx = [Int[] for _ in 1:m]
    rows_val = [Int[] for _ in 1:m]
    @inbounds for i in 1:m
        sizehint!(rows_idx[i], row_nnz[i])
        sizehint!(rows_val[i], row_nnz[i])
    end

    @inbounds for col in 1:n
        for ptr in A.colptr[col]:(A.colptr[col + 1] - 1)
            r = A.rowval[ptr]
            v = A.nzval[ptr]
            if v != 0
                a = _modp_qQ(v, p)
                if a != 0
                    push!(rows_idx[r], col)
                    push!(rows_val[r], a)
                end
            end
        end
    end

    rows = Vector{SparseRow{Int}}(undef, m)
    @inbounds for i in 1:m
        rows[i] = SparseRow{Int}(rows_idx[i], rows_val[i])
    end

    pivot_pos = zeros(Int, n)
    pivot_cols = Int[]
    pivot_rows = SparseRow{Int}[]
    scratch_cols = Int[]
    scratch_coeffs = Int[]
    tmp_idx = Int[]
    tmp_val = Int[]
    maxrank = min(m, n)

    for i in 1:m
        row = rows[i]
        if isempty(row)
            continue
        end

        # Eliminate existing pivots (collect first to avoid mutating while iterating row).
        empty!(scratch_cols)
        empty!(scratch_coeffs)
        @inbounds for t in eachindex(row.idx)
            j = row.idx[t]
            pos = pivot_pos[j]
            if pos != 0
                push!(scratch_cols, j)
                push!(scratch_coeffs, row.val[t])
            end
        end
        @inbounds for k in eachindex(scratch_cols)
            j = scratch_cols[k]
            c = scratch_coeffs[k]
            if c != 0
                prow = pivot_rows[pivot_pos[j]]
                tmp_idx, tmp_val = _row_axpy_modp!(row, -c, prow, p, tmp_idx, tmp_val)
            end
        end
        if isempty(row)
            continue
        end

        piv = row.idx[1]
        a = row.val[1]
        if a == 0
            # Guard against malformed rows.
            continue
        end

        inv_a = invmod(a, p)
        @inbounds for t in eachindex(row.val)
            row.val[t] = mod(row.val[t] * inv_a, p)
        end

        # Eliminate new pivot from existing pivot rows.
        for pos in 1:length(pivot_rows)
            prow = pivot_rows[pos]
            c = _row_coeff(prow, piv)
            if c != 0
                tmp_idx, tmp_val = _row_axpy_modp!(prow, -c, row, p, tmp_idx, tmp_val)
            end
        end

        push!(pivot_cols, piv)
        push!(pivot_rows, copy(row))
        pivot_pos[piv] = length(pivot_rows)

        if length(pivot_rows) == maxrank
            break
        end
    end

    return length(pivot_rows)
end

# ---------------------------------------------------------------
# Fast rank computations for dimension-only queries
# ---------------------------------------------------------------

const DEFAULT_MODULAR_PRIMES = Int[
    998244353, 1000000007, 1000000009, 1004535809,
    1045430273, 1053818881, 1224736769, 1300234241,
]

"""
    _rank_modp_dense(A::AbstractMatrix{QQ}, p::Int) -> Int

Rank over F_p for dense matrices by Gaussian elimination.
Used by _rankQQ_dim for fast dimension computations.
"""
function _rank_modp_dense(A::AbstractMatrix{QQ}, p::Int)::Int
    m, n = size(A)
    (m == 0 || n == 0) && return 0

    M = Matrix{Int}(undef, m, n)

    @inbounds for j in 1:n
        for i in 1:m
            q = A[i, j]
            if q == 0
                M[i, j] = 0
            else
                num = Int(mod(numerator(q), p))
                den = Int(mod(denominator(q), p))
                v = num * invmod(den, p)
                v %= p
                v < 0 && (v += p)
                M[i, j] = v
            end
        end
    end

    r = 0
    row = 1
    @inbounds for col in 1:n
        piv = row
        while piv <= m && M[piv, col] == 0
            piv += 1
        end
        piv > m && continue

        if piv != row
            for j in col:n
                M[row, j], M[piv, j] = M[piv, j], M[row, j]
            end
        end

        invp = invmod(M[row, col], p)
        for j in col:n
            v = M[row, j] * invp
            v %= p
            v < 0 && (v += p)
            M[row, j] = v
        end

        for i in (row+1):m
            f = M[i, col]
            f == 0 && continue
            for j in col:n
                v = M[i, j] - f * M[row, j]
                v %= p
                v < 0 && (v += p)
                M[i, j] = v
            end
        end

        r += 1
        row += 1
        row > m && break
    end

    return r
end

_rank_modp(A::AbstractMatrix{QQ}, p::Int) = _rank_modp_dense(A, p)
_rank_modp(A::SparseMatrixCSC{QQ,Int}, p::Int) = _rank_modp_sparse(A, p)
_rank_modp(A::Transpose{QQ,<:SparseMatrixCSC{QQ,Int}}, p::Int) = _rank_modp_sparse(sparse(A), p)
_rank_modp(A::Adjoint{QQ,<:SparseMatrixCSC{QQ,Int}}, p::Int)  = _rank_modp_sparse(sparse(A), p)

# ---------------------------------------------------------------
# Modular nullspace / solve with rational reconstruction (QQ only)
# ---------------------------------------------------------------

function _crt_combine!(A::Matrix{BigInt}, modulus::BigInt, B::Matrix{Int}, p::Int)
    mp = Int(mod(modulus, p))
    invm = invmod(mp, p)
    m = modulus
    @inbounds for j in 1:size(A, 2)
        for i in 1:size(A, 1)
            a = A[i, j]
            b = B[i, j]
            r = Int(mod(a, p))
            t = (b - r) % p
            t < 0 && (t += p)
            t = (t * invm) % p
            A[i, j] = a + m * BigInt(t)
        end
    end
    return A
end

function _rat_reconstruct(r::BigInt, m::BigInt)
    r = mod(r, m)
    a0 = m
    a1 = r
    b0 = BigInt(0)
    b1 = BigInt(1)
    bound = isqrt(div(m, 2))

    while abs(a1) > bound
        q = div(a0, a1)
        a0, a1 = a1, a0 - q * a1
        b0, b1 = b1, b0 - q * b1
    end

    b1 == 0 && return nothing
    abs(b1) > bound && return nothing
    gcd(a1, b1) == 1 || return nothing
    b1 < 0 && (a1 = -a1; b1 = -b1)
    mod(b1 * r - a1, m) == 0 || return nothing
    return (a1, b1)
end

function _reconstruct_matrix_QQ(A::Matrix{BigInt}, mod::BigInt)
    m, n = size(A)
    out = Matrix{QQ}(undef, m, n)
    @inbounds for j in 1:n
        for i in 1:m
            rr = _rat_reconstruct(A[i, j], mod)
            rr === nothing && return nothing
            num, den = rr
            out[i, j] = QQ(num, den)
        end
    end
    return out
end

function _verify_nullspaceQQ(A::AbstractMatrix{QQ}, N::AbstractMatrix{QQ})::Bool
    size(N, 2) == 0 && return true
    Matrix{QQ}(A * N) == zeros(QQ, size(A, 1), size(N, 2))
end

function _verify_solveQQ(B::AbstractMatrix{QQ}, X::AbstractVecOrMat{QQ}, Y::AbstractVecOrMat{QQ})::Bool
    Matrix{QQ}(B * X) == Matrix{QQ}(Y)
end

function _verify_solveQQ(B::SparseMatrixCSC{QQ,Int}, X::AbstractVector{QQ}, Y::AbstractVector{QQ})::Bool
    m, n = size(B)
    length(X) == n || return false
    length(Y) == m || return false
    rows = rowvals(B)
    vals = nonzeros(B)
    acc = fill(zero(QQ), m)
    @inbounds for j in 1:n
        xj = X[j]
        iszero(xj) && continue
        for ptr in nzrange(B, j)
            acc[rows[ptr]] += vals[ptr] * xj
        end
    end
    @inbounds for i in 1:m
        acc[i] == Y[i] || return false
    end
    return true
end

function _verify_solveQQ(B::SparseMatrixCSC{QQ,Int}, X::AbstractMatrix{QQ}, Y::AbstractMatrix{QQ})::Bool
    m, n = size(B)
    size(X, 1) == n || return false
    size(Y, 1) == m || return false
    size(X, 2) == size(Y, 2) || return false
    rows = rowvals(B)
    vals = nonzeros(B)
    acc = fill(zero(QQ), m)
    @inbounds for rhs in 1:size(X, 2)
        fill!(acc, zero(QQ))
        for j in 1:n
            xj = X[j, rhs]
            iszero(xj) && continue
            for ptr in nzrange(B, j)
                acc[rows[ptr]] += vals[ptr] * xj
            end
        end
        for i in 1:m
            acc[i] == Y[i, rhs] || return false
        end
    end
    return true
end

function _nullspace_modularQQ(A::AbstractMatrix{QQ};
                              primes::Vector{Int}=DEFAULT_MODULAR_PRIMES,
                              min_primes::Int=MODULAR_MIN_PRIMES[],
                              max_primes::Int=MODULAR_MAX_PRIMES[])
    m, n = size(A)
    n == 0 && return zeros(QQ, 0, 0)

    pivs_ref = nothing
    basis_mod = Matrix{BigInt}(undef, 0, 0)
    mod = BigInt(1)
    used = 0

    for p in primes
        R, pivs = try
            _rref_modp_dense(A, p)
        catch
            continue
        end

        if pivs_ref === nothing
            pivs_ref = pivs
        elseif pivs != pivs_ref
            continue
        end
        used += 1

        basis_p = _nullspace_modp_from_rref(R, pivs, n, p)
        if size(basis_p, 2) == 0
            return zeros(QQ, n, 0)
        end

        if mod == 1
            basis_mod = Matrix{BigInt}(undef, size(basis_p, 1), size(basis_p, 2))
            @inbounds for j in 1:size(basis_p, 2)
                for i in 1:size(basis_p, 1)
                    basis_mod[i, j] = BigInt(basis_p[i, j])
                end
            end
            mod = BigInt(p)
        else
            _crt_combine!(basis_mod, mod, basis_p, p)
            mod *= p
        end

        if used >= min_primes
            N = _reconstruct_matrix_QQ(basis_mod, mod)
            if N !== nothing && _verify_nullspaceQQ(A, N)
                return N
            end
        end
        used >= max_primes && break
    end

    return nothing
end

function _solve_fullcolumn_modularQQ(B::AbstractMatrix{QQ}, Y::AbstractVecOrMat{QQ};
                                     primes::Vector{Int}=DEFAULT_MODULAR_PRIMES,
                                     min_primes::Int=MODULAR_MIN_PRIMES[],
                                     max_primes::Int=MODULAR_MAX_PRIMES[],
                                     check_rhs::Bool=true)
    want_vec = false
    Ymat = Y
    if Y isa AbstractVector
        want_vec = true
        Ymat = reshape(Y, :, 1)
    end

    n = size(B, 2)
    mod = BigInt(1)
    sol_mod = Matrix{BigInt}(undef, 0, 0)
    used = 0

    for p in primes
        Xp = try
            _solve_fullcolumn_modp(B, Ymat, p)
        catch
            continue
        end
        used += 1

        if mod == 1
            sol_mod = Matrix{BigInt}(undef, size(Xp, 1), size(Xp, 2))
            @inbounds for j in 1:size(Xp, 2)
                for i in 1:size(Xp, 1)
                    sol_mod[i, j] = BigInt(Xp[i, j])
                end
            end
            mod = BigInt(p)
        else
            _crt_combine!(sol_mod, mod, Xp, p)
            mod *= p
        end

        if used >= min_primes
            X = _reconstruct_matrix_QQ(sol_mod, mod)
            if X !== nothing
                if !check_rhs || _verify_solveQQ(B, X, Ymat)
                    return want_vec ? vec(X) : X
                end
            end
        end
        used >= max_primes && break
    end

    return nothing
end

"""
    _rankQQ_dim(A; backend=:auto, max_primes=4, primes=DEFAULT_MODULAR_PRIMES,
                 small_threshold=RANKQQ_DIM_SMALL_THRESHOLD[]) -> Int

Fast rank intended for dimension-only queries:
- backend=:exact  uses exact _rankQQ
- backend=:modular uses ranks mod several primes
- backend=:auto uses exact for small matrices, modular otherwise
"""
function _rankQQ_dim(A::AbstractMatrix{QQ};
                    backend::Symbol=:auto,
                    max_primes::Int=4,
                    primes::Vector{Int}=DEFAULT_MODULAR_PRIMES,
                    small_threshold::Int=RANKQQ_DIM_SMALL_THRESHOLD[])::Int
    m, n = size(A)
    (m == 0 || n == 0) && return 0

    backend == :exact && return _rankQQ(A)

    if backend == :auto
        (m*n <= small_threshold) && return _rankQQ(A)
        backend = :modular
    end

    backend != :modular && error("_rankQQ_dim: unsupported backend $(backend)")

    r = 0
    used = 0
    for p in primes
        used += 1
        try
            r = max(r, _rank_modp(A, p))
        catch
            continue
        end
        r == min(m,n) && break
        used >= max_primes && break
    end
    return r
end

function _rank_tiny_exact(A::AbstractMatrix{K}) where {K}
    m, n = size(A)
    if m == 0 || n == 0
        return 0
    end
    M = Matrix{K}(A)
    r = 0
    row = 1
    @inbounds for col in 1:n
        row > m && break
        piv = 0
        for rr in row:m
            if !iszero(M[rr, col])
                piv = rr
                break
            end
        end
        piv == 0 && continue
        if piv != row
            M[row, :], M[piv, :] = M[piv, :], M[row, :]
        end
        invp = inv(M[row, col])
        for j in col:n
            M[row, j] *= invp
        end
        for rr in (row + 1):m
            fac = M[rr, col]
            iszero(fac) && continue
            for j in col:n
                M[rr, j] -= fac * M[row, j]
            end
        end
        r += 1
        row += 1
    end
    return r
end

function _rank_tiny_float(F::RealField, A::AbstractMatrix)
    m, n = size(A)
    if m == 0 || n == 0
        return 0
    end
    M = Matrix{Float64}(A)
    tol = _float_tol(F, M)
    r = 0
    row = 1
    @inbounds for col in 1:n
        row > m && break
        piv = 0
        best = tol
        for rr in row:m
            v = abs(M[rr, col])
            if v > best
                best = v
                piv = rr
            end
        end
        piv == 0 && continue
        if piv != row
            M[row, :], M[piv, :] = M[piv, :], M[row, :]
        end
        pivval = M[row, col]
        for j in col:n
            M[row, j] /= pivval
        end
        for rr in (row + 1):m
            fac = M[rr, col]
            abs(fac) <= tol && continue
            for j in col:n
                M[rr, j] -= fac * M[row, j]
            end
        end
        r += 1
        row += 1
    end
    return r
end

@inline _rank_tiny(field::RealField, A::AbstractMatrix) = _rank_tiny_float(field, A)
@inline _rank_tiny(::AbstractCoeffField, A::AbstractMatrix) = _rank_tiny_exact(A)

@inline function _rhs_ok(field::RealField, BX::AbstractMatrix, Y::AbstractMatrix)
    return norm(BX - Y) <= _float_tol(field, BX)
end
@inline _rhs_ok(::AbstractCoeffField, BX::AbstractMatrix, Y::AbstractMatrix) = (BX == Y)

function _inverse_tiny_exact(B::Matrix{K}) where {K}
    n = size(B, 1)
    size(B, 2) == n || error("_inverse_tiny_exact: expected square matrix")
    Aug = zeros(K, n, 2 * n)
    @inbounds for i in 1:n
        for j in 1:n
            Aug[i, j] = B[i, j]
        end
        Aug[i, n + i] = one(K)
    end
    @inbounds for col in 1:n
        piv = 0
        for rr in col:n
            if !iszero(Aug[rr, col])
                piv = rr
                break
            end
        end
        piv == 0 && error("_inverse_tiny_exact: matrix not invertible")
        if piv != col
            Aug[col, :], Aug[piv, :] = Aug[piv, :], Aug[col, :]
        end
        invp = inv(Aug[col, col])
        for j in col:(2 * n)
            Aug[col, j] *= invp
        end
        for rr in 1:n
            rr == col && continue
            fac = Aug[rr, col]
            iszero(fac) && continue
            for j in col:(2 * n)
                Aug[rr, j] -= fac * Aug[col, j]
            end
        end
    end
    return Matrix(Aug[:, (n + 1):(2 * n)])
end

function _inverse_tiny_float(F::RealField, B::Matrix{Float64})
    n = size(B, 1)
    size(B, 2) == n || error("_inverse_tiny_float: expected square matrix")
    tol = _float_tol(F, B)
    Aug = zeros(Float64, n, 2 * n)
    @inbounds for i in 1:n
        for j in 1:n
            Aug[i, j] = B[i, j]
        end
        Aug[i, n + i] = 1.0
    end
    @inbounds for col in 1:n
        piv = 0
        best = tol
        for rr in col:n
            v = abs(Aug[rr, col])
            if v > best
                best = v
                piv = rr
            end
        end
        piv == 0 && error("_inverse_tiny_float: matrix not invertible")
        if piv != col
            Aug[col, :], Aug[piv, :] = Aug[piv, :], Aug[col, :]
        end
        invp = 1.0 / Aug[col, col]
        for j in col:(2 * n)
            Aug[col, j] *= invp
        end
        for rr in 1:n
            rr == col && continue
            fac = Aug[rr, col]
            abs(fac) <= tol && continue
            for j in col:(2 * n)
                Aug[rr, j] -= fac * Aug[col, j]
            end
        end
    end
    return Matrix(Aug[:, (n + 1):(2 * n)])
end

@inline _inverse_tiny(field::RealField, B::AbstractMatrix) = _inverse_tiny_float(field, Matrix{Float64}(B))
@inline _inverse_tiny(::AbstractCoeffField, B::AbstractMatrix{K}) where {K} = _inverse_tiny_exact(Matrix{K}(B))

function _tiny_row_combinations(m::Int, n::Int)
    if n == 0
        return [Int[]]
    elseif n == 1
        return [[i] for i in 1:m]
    elseif n == 2
        out = Vector{Vector{Int}}()
        for i in 1:(m - 1), j in (i + 1):m
            push!(out, [i, j])
        end
        return out
    elseif n == 3
        out = Vector{Vector{Int}}()
        for i in 1:(m - 2), j in (i + 1):(m - 1), k in (j + 1):m
            push!(out, [i, j, k])
        end
        return out
    elseif n == 4
        out = Vector{Vector{Int}}()
        for i in 1:(m - 3), j in (i + 1):(m - 2), k in (j + 1):(m - 1), l in (k + 1):m
            push!(out, [i, j, k, l])
        end
        return out
    end
    return Vector{Vector{Int}}()
end

function _select_fullrank_rows_tiny(field::AbstractCoeffField, B::AbstractMatrix)
    m, n = size(B)
    m < n && return Int[]
    for rows in _tiny_row_combinations(m, n)
        if _rank_tiny(field, view(B, rows, :)) == n
            return rows
        end
    end
    return Int[]
end

function _solve_fullcolumn_tiny(field::AbstractCoeffField, B, Y; check_rhs::Bool=true)
    want_vec = Y isa AbstractVector
    Ymat = want_vec ? reshape(Y, :, 1) : Matrix(Y)
    m, n = size(B)
    size(Ymat, 1) == m || throw(DimensionMismatch("B and Y must have same row count"))
    if n == 0
        if check_rhs
            z = zero(eltype(Ymat))
            @inbounds for y in Ymat
                y == z || error("solve_fullcolumn: right-hand side is not in column space of B")
            end
        end
        return want_vec ? eltype(Ymat)[] : zeros(eltype(Ymat), 0, size(Ymat, 2))
    end
    Bdense = Matrix(B)
    rows = _select_fullrank_rows_tiny(field, Bdense)
    isempty(rows) && error("solve_fullcolumn: expected full column rank in tiny solve path")
    invB = _inverse_tiny(field, view(Bdense, rows, :))
    X = _matmul(invB, Matrix(Ymat[rows, :]))
    if check_rhs
        BX = _matmul(Bdense, X)
        _rhs_ok(field, BX, Matrix(Ymat)) || error("solve_fullcolumn: RHS check failed in tiny solve path")
    end
    return want_vec ? vec(X) : X
end
