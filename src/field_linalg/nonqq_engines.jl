# -----------------------------------------------------------------------------
# field_linalg/nonqq_engines.jl
#
# Scope:
#   Non-QQ linear algebra kernels: F2, F3, floating-point, and generic prime-
#   field engines, including their exact solve/rref/nullspace helpers.
# Owns:
#   - F2/F3/Fp/RealField kernel implementations,
#   - non-QQ full-column factorization/solve helpers,
#   - field-specific dense/sparse elimination support outside QQ.
# Does not own:
#   - backend routing policy,
#   - QQ kernels,
#   - public dispatch wrappers.
# Depends on:
#   - `thresholds.jl` and `backend_routing.jl` for backend decisions,
#   - `sparse_rref.jl` for shared sparse elimination primitives where needed.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# F2 full-column solve cache
# -----------------------------------------------------------------------------

struct F2FullColumnFactor
    rows::Vector{Int}
    invB::Vector{Vector{UInt64}}  # packed n x n inverse
    n::Int
end

const _F2_FULLCOLUMN_FACTOR_CACHE = WeakKeyDict{Any,Any}()

function _clear_f2_fullcolumn_cache!()
    empty!(_F2_FULLCOLUMN_FACTOR_CACHE)
    return nothing
end

# -----------------------------------------------------------------------------
# F2 bitpacked engine
# -----------------------------------------------------------------------------

@inline _f2_blocks(ncols::Int) = (ncols + 63) >>> 6

@inline function _f2_getbit(row::Vector{UInt64}, col::Int)
    blk = (col - 1) >>> 6
    bit = (col - 1) & 63
    return (row[blk + 1] >>> bit) & UInt64(1)
end

# -----------------------------------------------------------------------------
# F3 table-based engine
# -----------------------------------------------------------------------------

const _F3_ADD = UInt8[
    0 1 2;
    1 2 0;
    2 0 1
]
const _F3_SUB = UInt8[
    0 2 1;
    1 0 2;
    2 1 0
]
const _F3_MUL = UInt8[
    0 0 0;
    0 1 2;
    0 2 1
]
const _F3_INV = UInt8[0, 1, 2]  # inv(0) unused; inv(1)=1, inv(2)=2

@inline _f3_sub(a::UInt8, b::UInt8) = _F3_SUB[a + 1, b + 1]
@inline _f3_mul(a::UInt8, b::UInt8) = _F3_MUL[a + 1, b + 1]
@inline _f3_inv(a::UInt8) = _F3_INV[a + 1]
@inline _f3_neg(a::UInt8) = _F3_SUB[1, a + 1]

struct F3FullColumnFactor
    rows::Vector{Int}
    invB::Matrix{FpElem{3}}
end

const _F3_FULLCOLUMN_FACTOR_CACHE = WeakKeyDict{Any,Any}()

function _clear_f3_fullcolumn_cache!()
    empty!(_F3_FULLCOLUMN_FACTOR_CACHE)
    return nothing
end

function _f3_uint(A::AbstractMatrix{FpElem{3}})
    m, n = size(A)
    M = Matrix{UInt8}(undef, m, n)
    @inbounds for i in 1:m, j in 1:n
        M[i, j] = UInt8(A[i, j].val)
    end
    return M
end

function _f3_uint(A::SparseMatrixCSC{FpElem{3},Int})
    m, n = size(A)
    M = fill(UInt8(0), m, n)
    @inbounds for col in 1:n
        for ptr in A.colptr[col]:(A.colptr[col + 1] - 1)
            r = A.rowval[ptr]
            v = A.nzval[ptr].val
            if v != 0
                M[r, col] = UInt8(v)
            end
        end
    end
    return M
end

function _f3_from_uint(M::Matrix{UInt8})
    m, n = size(M)
    A = Matrix{FpElem{3}}(undef, m, n)
    @inbounds for i in 1:m, j in 1:n
        A[i, j] = FpElem{3}(M[i, j])
    end
    return A
end

function _rref_f3(A::AbstractMatrix{FpElem{3}}; pivots::Bool=true)
    M = _f3_uint(A)
    m, n = size(M)
    pivs = Int[]
    row = 1

    for col in 1:n
        row > m && break
        pivrow = 0
        @inbounds for r in row:m
            if M[r, col] != 0x00
                pivrow = r
                break
            end
        end
        pivrow == 0 && continue

        if pivrow != row
            M[row, :], M[pivrow, :] = M[pivrow, :], M[row, :]
        end

        piv = M[row, col]
        invp = _f3_inv(piv)
        @inbounds for j in 1:n
            M[row, j] = _f3_mul(M[row, j], invp)
        end

        @inbounds for r in 1:m
            r == row && continue
            fac = M[r, col]
            fac == 0x00 && continue
            for j in 1:n
                M[r, j] = _f3_sub(M[r, j], _f3_mul(fac, M[row, j]))
            end
        end

        push!(pivs, col)
        row += 1
    end

    R = _f3_from_uint(M)
    return pivots ? (R, Tuple(pivs)) : R
end

function _rref_f3(A::SparseMatrixCSC{FpElem{3},Int}; pivots::Bool=true)
    m, n = size(A)
    R = _SparseRREF{FpElem{3}}(n)
    rows = _sparse_rows(A)
    for i in 1:m
        _sparse_rref_push_homogeneous!(R, rows[i])
    end

    # materialize dense RREF with pivot rows first, zero rows below
    rank = _rref_rank(R)
    Md = zeros(FpElem{3}, m, n)
    for i in 1:rank
        prow = R.pivot_rows[i]
        @inbounds for t in eachindex(prow.idx)
            Md[i, prow.idx[t]] = prow.val[t]
        end
    end
    pivs = copy(R.pivot_cols)
    return pivots ? (Md, Tuple(pivs)) : Md
end

function _rank_f3(A::AbstractMatrix{FpElem{3}})
    _, pivs = _rref_f3(A; pivots=true)
    return length(pivs)
end

function _rank_f3(A::SparseMatrixCSC{FpElem{3},Int})
    _, pivs = _rref_f3(A; pivots=true)
    return length(pivs)
end

_rank_f3(A::Transpose{FpElem{3},<:SparseMatrixCSC{FpElem{3},Int}}) = _rank_f3(sparse(A))
_rank_f3(A::Adjoint{FpElem{3},<:SparseMatrixCSC{FpElem{3},Int}})  = _rank_f3(sparse(A))

function _nullspace_f3(A::AbstractMatrix{FpElem{3}})
    R, pivs = _rref_f3(A; pivots=true)
    m, n = size(R)

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

    nfree = length(free)
    Z = zeros(FpElem{3}, n, nfree)
    nfree == 0 && return Z

    @inbounds for (k, jf) in enumerate(free)
        Z[jf, k] = FpElem{3}(1)
        for (i, jp) in enumerate(pivs)
            v = R[i, jf].val
            if v != 0
                Z[jp, k] = FpElem{3}(_f3_neg(UInt8(v)))
            end
        end
    end
    return Z
end

function _nullspace_f3(A::SparseMatrixCSC{FpElem{3},Int})
    m, n = size(A)
    R = _SparseRREF{FpElem{3}}(n)
    rows = _sparse_rows(A)
    for i in 1:m
        _sparse_rref_push_homogeneous!(R, rows[i])
    end
    return _nullspace_from_pivots(R, n)
end

_nullspace_f3(A::Transpose{FpElem{3},<:SparseMatrixCSC{FpElem{3},Int}}) = _nullspace_f3(sparse(A))
_nullspace_f3(A::Adjoint{FpElem{3},<:SparseMatrixCSC{FpElem{3},Int}})  = _nullspace_f3(sparse(A))

function _solve_fullcolumn_f3(B::AbstractMatrix{FpElem{3}},
                              Y::AbstractVecOrMat{FpElem{3}};
                              check_rhs::Bool=true,
                              cache::Bool=true,
                              factor::Union{Nothing,F3FullColumnFactor}=nothing)
    want_vec = false
    Ymat = Y
    if Y isa AbstractVector
        want_vec = true
        Ymat = reshape(Y, :, 1)
    end

    m, n = size(B)
    size(Ymat, 1) == m || throw(DimensionMismatch("B and Y must have same row count"))

    fact = factor
    if fact === nothing && cache && Base.ismutabletype(typeof(B))
        fact = get(_F3_FULLCOLUMN_FACTOR_CACHE, B, nothing)
    end

    if fact === nothing
        pivs = _pivot_cols_f3(transpose(B))
        if length(pivs) != n
            error("solve_fullcolumn_f3: expected full column rank, got rank $(length(pivs)) < $n")
        end
        Bsub = Matrix{FpElem{3}}(B[pivs, :])
        invB = _f3_inverse(Bsub)
        fact = F3FullColumnFactor(pivs, invB)
        if cache && Base.ismutabletype(typeof(B))
            _F3_FULLCOLUMN_FACTOR_CACHE[B] = fact
        end
    end

    rhs = size(Ymat, 2)
    X = fact.invB * Ymat[fact.rows, :]

    if check_rhs
        if B * X != Ymat
            error("solve_fullcolumn_f3: RHS check failed")
        end
    end

    return want_vec ? vec(X) : X
end

function _solve_fullcolumn_f3(B::SparseMatrixCSC{FpElem{3},Int},
                              Y::AbstractVecOrMat{FpElem{3}};
                              check_rhs::Bool=true,
                              cache::Bool=true,
                              factor::Union{Nothing,F3FullColumnFactor}=nothing)
    want_vec = false
    Ymat = Y
    if Y isa AbstractVector
        want_vec = true
        Ymat = reshape(Y, :, 1)
    end

    m, n = size(B)
    size(Ymat, 1) == m || throw(DimensionMismatch("B and Y must have same row count"))

    fact = factor
    if fact === nothing && cache && Base.ismutabletype(typeof(B))
        fact = get(_F3_FULLCOLUMN_FACTOR_CACHE, B, nothing)
    end

    if fact === nothing
        Bt = sparse(transpose(B))
        pivs = _pivot_cols_f3(Bt)
        if length(pivs) != n
            error("solve_fullcolumn_f3: expected full column rank, got rank $(length(pivs)) < $n")
        end
        Bsub = Matrix{FpElem{3}}(B[pivs, :])
        invB = _f3_inverse(Bsub)
        fact = F3FullColumnFactor(pivs, invB)
        if cache && Base.ismutabletype(typeof(B))
            _F3_FULLCOLUMN_FACTOR_CACHE[B] = fact
        end
    end

    X = fact.invB * Ymat[fact.rows, :]

    if check_rhs
        if B * X != Ymat
            error("solve_fullcolumn_f3: RHS check failed")
        end
    end

    return want_vec ? vec(X) : X
end

function _rank_restricted_f3(A::SparseMatrixCSC{FpElem{3},Int},
                             rows::AbstractVector{Int},
                             cols::AbstractVector{Int};
                             check::Bool=false)
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
    M = zeros(UInt8, nr, nc)
    loc = _build_row_locator(rows, m)
    @inbounds for (jloc, col) in enumerate(cols)
        for ptr in A.colptr[col]:(A.colptr[col + 1] - 1)
            r = A.rowval[ptr]
            i = _row_lookup(loc, r)
            if i != 0
                v = A.nzval[ptr].val
                if v != 0
                    M[i, jloc] = UInt8(v)
                end
            end
        end
    end
    return _rank_f3(_f3_from_uint(M))
end

function _pivot_cols_f3(A::AbstractMatrix{FpElem{3}})
    _, pivs = _rref_f3(A; pivots=true)
    return collect(pivs)
end

function _pivot_cols_f3(A::SparseMatrixCSC{FpElem{3},Int})
    m, n = size(A)
    R = _SparseRREF{FpElem{3}}(n)
    rows = _sparse_rows(A)
    for i in 1:m
        _sparse_rref_push_homogeneous!(R, rows[i])
    end
    return copy(R.pivot_cols)
end

function _f3_inverse(B::Matrix{FpElem{3}})
    n = size(B, 1)
    size(B, 2) == n || error("_f3_inverse: square matrix required")

    M = _f3_uint(B)
    Aug = Matrix{UInt8}(undef, n, 2 * n)
    @inbounds for i in 1:n
        for j in 1:n
            Aug[i, j] = M[i, j]
        end
        for j in 1:n
            Aug[i, n + j] = (i == j) ? UInt8(1) : UInt8(0)
        end
    end

    row = 1
    for col in 1:n
        row > n && break
        pivrow = 0
        @inbounds for r in row:n
            if Aug[r, col] != 0x00
                pivrow = r
                break
            end
        end
        pivrow == 0 && continue
        if pivrow != row
            Aug[row, :], Aug[pivrow, :] = Aug[pivrow, :], Aug[row, :]
        end

        piv = Aug[row, col]
        invp = _f3_inv(piv)
        @inbounds for j in 1:2*n
            Aug[row, j] = _f3_mul(Aug[row, j], invp)
        end

        @inbounds for r in 1:n
            r == row && continue
            fac = Aug[r, col]
            fac == 0x00 && continue
            for j in 1:2*n
                Aug[r, j] = _f3_sub(Aug[r, j], _f3_mul(fac, Aug[row, j]))
            end
        end

        row += 1
    end

    if row <= n
        error("_f3_inverse: matrix not invertible")
    end

    invB = Matrix{FpElem{3}}(undef, n, n)
    @inbounds for i in 1:n, j in 1:n
        invB[i, j] = FpElem{3}(Aug[i, n + j])
    end
    return invB
end

# -----------------------------------------------------------------------------
# Real / floating-point engine
# -----------------------------------------------------------------------------

function _float_tol(F::RealField, A)
    return F.atol + F.rtol * opnorm(A, 1)
end

function _rank_float(F::RealField, A::StridedMatrix{<:Real})
    R = qr(A, Val(true)).R
    d = diag(R)
    tol = _float_tol(F, A)
    return count(x -> abs(x) > tol, d)
end

function _rank_float(F::RealField, A::AbstractMatrix{<:Real})
    return _rank_float(F, Matrix{Float64}(A))
end

function _rank_float_svd(F::RealField, A)
    s = svdvals(Matrix(A))
    tol = _float_tol(F, A)
    return count(>(tol), s)
end

function _rank_float(F::RealField, A::SparseMatrixCSC)
    m, n = size(A)
    if m == 0 || n == 0
        return 0
    end
    R = qr(A).R
    d = diag(R)
    tol = _float_tol(F, A)
    return count(x -> abs(x) > tol, d)
end

function _rank_float(F::RealField, A::Transpose{<:Real,<:SparseMatrixCSC})
    return _rank_float(F, parent(A))
end

function _rank_float(F::RealField, A::Adjoint{<:Real,<:SparseMatrixCSC})
    return _rank_float(F, parent(A))
end

function _nullspace_from_qr_sparse_float(F::RealField, A)
    m, n = size(A)
    n == 0 && return zeros(Float64, 0, 0)
    m == 0 && return Matrix{Float64}(I, n, n)

    Fq = qr(A)
    R = Fq.R
    d = diag(R)
    tol = _float_tol(F, A)
    r = count(x -> abs(x) > tol, d)
    nfree = n - r
    nfree <= 0 && return zeros(Float64, n, 0)

    Zperm = zeros(Float64, n, nfree)
    if r > 0
        R11 = Matrix(view(R, 1:r, 1:r))
        R12 = Matrix(view(R, 1:r, (r + 1):n))
        Zperm[1:r, :] .= -(R11 \ R12)
    end
    @inbounds for j in 1:nfree
        Zperm[r + j, j] = 1.0
    end

    Z = zeros(Float64, n, nfree)
    pcol = Fq.pcol
    @inbounds for j in 1:n
        Z[pcol[j], :] .= Zperm[j, :]
    end
    return Z
end

function _nullspace_float_qr_dense(F::RealField, A::AbstractMatrix{<:Real})
    m, n = size(A)
    n == 0 && return zeros(Float64, 0, 0)
    m == 0 && return Matrix{Float64}(I, n, n)

    Fq = qr(A, Val(true))
    R = Fq.R
    d = diag(R)
    tol = _float_tol(F, A)
    r = count(x -> abs(x) > tol, d)
    nfree = n - r
    nfree <= 0 && return zeros(Float64, n, 0)

    Zperm = zeros(Float64, n, nfree)
    if r > 0
        R11 = Matrix(view(R, 1:r, 1:r))
        R12 = Matrix(view(R, 1:r, (r + 1):n))
        Zperm[1:r, :] .= -(R11 \ R12)
    end
    @inbounds for j in 1:nfree
        Zperm[r + j, j] = 1.0
    end

    Z = zeros(Float64, n, nfree)
    piv = Vector{Int}(Fq.p[1:n])
    @inbounds for j in 1:n
        Z[piv[j], :] .= Zperm[j, :]
    end
    return Z
end

function _nullspace_float_svd(F::RealField, A)
    S = svd(Matrix(A); full=true)
    r = _rank_float_svd(F, A)
    n = size(A, 2)
    r >= n && return zeros(eltype(S.Vt), n, 0)
    return Matrix(S.Vt[(r + 1):end, :])'
end

function _nullspace_float(F::RealField, A::AbstractMatrix{<:Real})
    return _nullspace_float_qr_dense(F, A)
end

function _nullspace_float(F::RealField, A::SparseMatrixCSC)
    return _nullspace_from_qr_sparse_float(F, A)
end

function _nullspace_float(F::RealField, A::Transpose{<:Real,<:SparseMatrixCSC})
    return _nullspace_from_qr_sparse_float(F, A)
end

function _nullspace_float(F::RealField, A::Adjoint{<:Real,<:SparseMatrixCSC})
    return _nullspace_from_qr_sparse_float(F, A)
end

function _rref_float(F::RealField, A; pivots::Bool=true)
    M = Matrix(A)
    Q, R, piv = qr(M, Val(true))
    r = _rank_float(F, A)
    pivs = Vector{Int}(piv[1:r])
    return pivots ? (M, Tuple(pivs)) : M
end

function _rref_float(F::RealField, A::SparseMatrixCSC; pivots::Bool=true)
    Fq = qr(A)
    r = _rank_float(F, A)
    pivs = Vector{Int}(Fq.pcol[1:r])
    return pivots ? (copy(A), Tuple(pivs)) : copy(A)
end

function _colspace_float(F::RealField, A)
    Q, R, piv = qr(Matrix(A), Val(true))
    r = _rank_float(F, A)
    cols = piv[1:r]
    return Matrix(A)[:, cols]
end

function _colspace_float(F::RealField, A::Transpose{<:Real,<:SparseMatrixCSC})
    Fq = qr(A)
    r = _rank_float(F, A)
    cols = Fq.pcol[1:r]
    return A[:, cols]
end

function _colspace_float(F::RealField, A::Adjoint{<:Real,<:SparseMatrixCSC})
    Fq = qr(A)
    r = _rank_float(F, A)
    cols = Fq.pcol[1:r]
    return A[:, cols]
end

function _colspace_float(F::RealField, A::SparseMatrixCSC)
    Fq = qr(A)
    r = _rank_float(F, A)
    cols = Fq.pcol[1:r]
    return A[:, cols]
end

function _solve_fullcolumn_float(F::RealField, B, Y; check_rhs::Bool=true)
    Ymat = Y isa AbstractVector ? reshape(Y, :, 1) : Matrix(Y)
    Fq = qr(Matrix(B), Val(true))
    X = Fq \ Ymat
    if check_rhs
        R = Matrix(B) * X - Ymat
        tol = _float_tol(F, B)
        norm(R) <= tol || error("solve_fullcolumn_float: RHS residual too large")
    end
    return (Y isa AbstractVector) ? vec(X) : X
end

const _FLOAT_SPARSE_FACTOR_CACHE = Dict{NTuple{5,UInt},Any}()
const FLOAT_SPARSE_FACTOR_CACHE_MAX = Ref(128)

@inline function _float_sparse_cache_key(B::SparseMatrixCSC)
    return (
        objectid(B.colptr),
        objectid(B.rowval),
        objectid(B.nzval),
        UInt(size(B, 1)),
        UInt(size(B, 2)),
    )
end

function _solve_fullcolumn_float(F::RealField, B::SparseMatrixCSC, Y;
                                 check_rhs::Bool=true, cache::Bool=true, factor=nothing)
    Ymat = Y isa AbstractVector ? reshape(Y, :, 1) : Matrix(Y)
    fac = factor
    if fac === nothing && cache
        fac = get(_FLOAT_SPARSE_FACTOR_CACHE, _float_sparse_cache_key(B), nothing)
    end
    if fac === nothing
        fac = qr(B)
        if cache
            if length(_FLOAT_SPARSE_FACTOR_CACHE) >= FLOAT_SPARSE_FACTOR_CACHE_MAX[]
                empty!(_FLOAT_SPARSE_FACTOR_CACHE)
            end
            _FLOAT_SPARSE_FACTOR_CACHE[_float_sparse_cache_key(B)] = fac
        end
    end
    X = fac \ Ymat
    if check_rhs
        R = B * X - Ymat
        tol = _float_tol(F, B)
        norm(R) <= tol || error("solve_fullcolumn_float: RHS residual too large")
    end
    return (Y isa AbstractVector) ? vec(X) : X
end
# -----------------------------------------------------------------------------
# Generic Fp (p > 3) exact engine (fallback when Nemo is unavailable)
# -----------------------------------------------------------------------------

function _rref_fp(A::AbstractMatrix{FpElem{p}}; pivots::Bool=true) where {p}
    M = Matrix{FpElem{p}}(A)
    m, n = size(M)
    pivs = Int[]
    row = 1

    for col in 1:n
        row > m && break
        pivrow = 0
        @inbounds for r in row:m
            if M[r, col].val != 0
                pivrow = r
                break
            end
        end
        pivrow == 0 && continue

        if pivrow != row
            M[row, :], M[pivrow, :] = M[pivrow, :], M[row, :]
        end

        piv = M[row, col]
        invp = inv(piv)
        @inbounds for j in 1:n
            M[row, j] *= invp
        end

        @inbounds for r in 1:m
            r == row && continue
            fac = M[r, col]
            fac.val == 0 && continue
            for j in 1:n
                M[r, j] -= fac * M[row, j]
            end
        end

        push!(pivs, col)
        row += 1
    end

    return pivots ? (M, Tuple(pivs)) : M
end

function _rank_fp(A::AbstractMatrix{FpElem{p}}) where {p}
    _, pivs = _rref_fp(A; pivots=true)
    return length(pivs)
end

function _rank_fp(A::SparseMatrixCSC{FpElem{p},Int}) where {p}
    m, n = size(A)
    if m == 0 || n == 0
        return 0
    end
    rows = _sparse_rows(A)
    R = _SparseRREF{FpElem{p}}(n)
    maxrank = min(m, n)
    @inbounds for i in 1:m
        _sparse_rref_push_homogeneous!(R, rows[i])
        if _rref_rank(R) == maxrank
            break
        end
    end
    return _rref_rank(R)
end

_rank_fp(A::Transpose{FpElem{p},<:SparseMatrixCSC{FpElem{p},Int}}) where {p} = _rank_fp(parent(A))
_rank_fp(A::Adjoint{FpElem{p},<:SparseMatrixCSC{FpElem{p},Int}}) where {p} = _rank_fp(parent(A))

function _nullspace_fp(A::AbstractMatrix{FpElem{p}}) where {p}
    R, pivs = _rref_fp(A; pivots=true)
    m, n = size(R)

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

    nfree = length(free)
    Z = zeros(FpElem{p}, n, nfree)
    nfree == 0 && return Z

    @inbounds for (k, jf) in enumerate(free)
        Z[jf, k] = FpElem{p}(1)
        for (i, jp) in enumerate(pivs)
            v = R[i, jf]
            if v.val != 0
                Z[jp, k] = -v
            end
        end
    end
    return Z
end

function _nullspace_fp(A::SparseMatrixCSC{FpElem{p},Int}) where {p}
    m, n = size(A)
    rows = _sparse_rows(A)
    R = _SparseRREF{FpElem{p}}(n)
    maxrank = min(m, n)
    @inbounds for i in 1:m
        _sparse_rref_push_homogeneous!(R, rows[i])
        if _rref_rank(R) == maxrank
            break
        end
    end
    return _nullspace_from_pivots(R, n)
end

function _nullspace_fp_from_transposed_parent(A::SparseMatrixCSC{FpElem{p},Int}) where {p}
    # For transpose(A), rows are exactly the original sparse columns of A.
    m, n = size(A)
    R = _SparseRREF{FpElem{p}}(m)
    maxrank = min(n, m)
    for col in 1:n
        rng = A.colptr[col]:(A.colptr[col + 1] - 1)
        idx = Int[]
        val = FpElem{p}[]
        sizehint!(idx, length(rng))
        sizehint!(val, length(rng))
        @inbounds for ptr in rng
            v = A.nzval[ptr]
            if !iszero(v)
                push!(idx, A.rowval[ptr])
                push!(val, v)
            end
        end
        _sparse_rref_push_homogeneous!(R, SparseRow{FpElem{p}}(idx, val))
        if _rref_rank(R) == maxrank
            break
        end
    end
    return _nullspace_from_pivots(R, m)
end

_nullspace_fp(A::Transpose{FpElem{p},<:SparseMatrixCSC{FpElem{p},Int}}) where {p} = _nullspace_fp_from_transposed_parent(parent(A))
_nullspace_fp(A::Adjoint{FpElem{p},<:SparseMatrixCSC{FpElem{p},Int}})  where {p} = _nullspace_fp_from_transposed_parent(parent(A))

function _solve_fullcolumn_fp(B::AbstractMatrix{FpElem{p}},
                              Y::AbstractVecOrMat{FpElem{p}};
                              check_rhs::Bool=true) where {p}
    want_vec = false
    Ymat = Y
    if Y isa AbstractVector
        want_vec = true
        Ymat = reshape(Y, :, 1)
    end

    m, n = size(B)
    size(Ymat, 1) == m || throw(DimensionMismatch("B and Y must have same row count"))

    Aug = hcat(B, Ymat)
    R, pivs = _rref_fp(Aug; pivots=true)
    if length(pivs) != n
        error("solve_fullcolumn_fp: expected full column rank, got rank $(length(pivs)) < $n")
    end

    rhs = size(Ymat, 2)
    X = zeros(FpElem{p}, n, rhs)
    @inbounds for (row, pcol) in enumerate(pivs)
        X[pcol, :] = R[row, n+1:n+rhs]
    end

    if check_rhs
        if B * X != Ymat
            error("solve_fullcolumn_fp: RHS check failed")
        end
    end

    return want_vec ? vec(X) : X
end

function _solve_fullcolumn_fp(B::SparseMatrixCSC{FpElem{p},Int},
                              Y::AbstractVecOrMat{FpElem{p}};
                              check_rhs::Bool=true) where {p}
    want_vec = false
    Ymat = Y
    if Y isa AbstractVector
        want_vec = true
        Ymat = reshape(Y, :, 1)
    end

    m, n = size(B)
    size(Ymat, 1) == m || throw(DimensionMismatch("B and Y must have same row count"))

    rhs = size(Ymat, 2)
    RA = _SparseRREFAugmented{FpElem{p}}(n, rhs)
    rows = _sparse_rows(B)

    @inbounds for i in 1:m
        row_rhs = Vector{FpElem{p}}(undef, rhs)
        for j in 1:rhs
            row_rhs[j] = Ymat[i, j]
        end
        status = _sparse_rref_push_augmented!(RA, rows[i], row_rhs)
        if status === :inconsistent
            error("solve_fullcolumn_fp: RHS is not in column space of B")
        end
    end

    pivs = RA.rref.pivot_cols
    if length(pivs) != n
        error("solve_fullcolumn_fp: expected full column rank, got rank $(length(pivs)) < $n")
    end
    @inbounds for pcol in pivs
        if pcol > n
            error("solve_fullcolumn_fp: RHS is not in column space of B")
        end
    end

    X = zeros(FpElem{p}, n, rhs)
    @inbounds for (row, pcol) in enumerate(pivs)
        X[pcol, :] .= RA.pivot_rhs[row]
    end

    if check_rhs
        if B * X != Ymat
            error("solve_fullcolumn_fp: RHS check failed")
        end
    end

    return want_vec ? vec(X) : X
end

@inline function _f2_setbit!(row::Vector{UInt64}, col::Int)
    blk = (col - 1) >>> 6
    bit = (col - 1) & 63
    row[blk + 1] |= (UInt64(1) << bit)
end

function _pack_f2(A::AbstractMatrix{FpElem{2}})
    m, n = size(A)
    nb = _f2_blocks(n)
    rows = [fill(UInt64(0), nb) for _ in 1:m]
    @inbounds for i in 1:m
        row = rows[i]
        for j in 1:n
            if A[i, j].val != 0
                _f2_setbit!(row, j)
            end
        end
    end
    return rows, n
end

function _pack_f2(A::SparseMatrixCSC{FpElem{2},Int})
    m, n = size(A)
    nb = _f2_blocks(n)
    rows = [fill(UInt64(0), nb) for _ in 1:m]
    @inbounds for col in 1:n
        for ptr in A.colptr[col]:(A.colptr[col + 1] - 1)
            r = A.rowval[ptr]
            if A.nzval[ptr].val != 0
                _f2_setbit!(rows[r], col)
            end
        end
    end
    return rows, n
end

function _unpack_f2(rows::Vector{Vector{UInt64}}, m::Int, n::Int)
    A = Matrix{FpElem{2}}(undef, m, n)
    z = FpElem{2}(0)
    o = FpElem{2}(1)
    @inbounds for i in 1:m
        row = rows[i]
        for j in 1:n
            A[i, j] = (_f2_getbit(row, j) == 1) ? o : z
        end
    end
    return A
end

function _f2_row_xor!(a::Vector{UInt64}, b::Vector{UInt64})
    @inbounds for k in eachindex(a)
        a[k] = xor(a[k], b[k])
    end
end

function _f2_inverse_packed(B::AbstractMatrix{FpElem{2}})
    n = size(B, 1)
    size(B, 2) == n || error("_f2_inverse_packed: square matrix required")

    nb = _f2_blocks(2 * n)
    rows = [fill(UInt64(0), nb) for _ in 1:n]
    @inbounds for i in 1:n
        row = rows[i]
        for j in 1:n
            if B[i, j].val != 0
                _f2_setbit!(row, j)
            end
        end
        _f2_setbit!(row, n + i)
    end

    pivs = _gauss_jordan_f2!(rows, 2 * n; pivot_cols=n)
    if length(pivs) != n
        error("_f2_inverse_packed: matrix not invertible")
    end

    inv_rows = [fill(UInt64(0), _f2_blocks(n)) for _ in 1:n]
    @inbounds for i in 1:n
        row = rows[i]
        for j in 1:n
            if _f2_getbit(row, n + j) == 1
                _f2_setbit!(inv_rows[i], j)
            end
        end
    end
    return inv_rows
end

function _gauss_jordan_f2!(rows::Vector{Vector{UInt64}}, ncols::Int; pivot_cols::Int=ncols)
    m = length(rows)
    pivs = Int[]
    r = 1
    nb = _f2_blocks(ncols)
    for col in 1:pivot_cols
        r > m && break
        blk = (col - 1) >>> 6
        bit = (col - 1) & 63
        mask = UInt64(1) << bit
        piv = 0
        @inbounds for i in r:m
            if (rows[i][blk + 1] & mask) != 0
                piv = i
                break
            end
        end
        piv == 0 && continue

        if piv != r
            rows[r], rows[piv] = rows[piv], rows[r]
        end

        # Eliminate from all other rows to get RREF.
        @inbounds for i in 1:m
            if i != r && (rows[i][blk + 1] & mask) != 0
                _f2_row_xor!(rows[i], rows[r])
            end
        end

        push!(pivs, col)
        r += 1
    end
    return pivs
end

function _rank_f2(A::AbstractMatrix{FpElem{2}})
    rows, n = _pack_f2(A)
    pivs = _gauss_jordan_f2!(rows, n)
    return length(pivs)
end

function _rank_f2(A::SparseMatrixCSC{FpElem{2},Int})
    rows, n = _pack_f2(A)
    pivs = _gauss_jordan_f2!(rows, n)
    return length(pivs)
end

_rank_f2(A::Transpose{FpElem{2},<:SparseMatrixCSC{FpElem{2},Int}}) = _rank_f2(sparse(A))
_rank_f2(A::Adjoint{FpElem{2},<:SparseMatrixCSC{FpElem{2},Int}})  = _rank_f2(sparse(A))

function _rref_f2(A::AbstractMatrix{FpElem{2}}; pivots::Bool=true)
    rows, n = _pack_f2(A)
    pivs = _gauss_jordan_f2!(rows, n)
    R = _unpack_f2(rows, size(A, 1), n)
    return pivots ? (R, Tuple(pivs)) : R
end

function _rref_f2(A::SparseMatrixCSC{FpElem{2},Int}; pivots::Bool=true)
    rows, n = _pack_f2(A)
    pivs = _gauss_jordan_f2!(rows, n)
    R = _unpack_f2(rows, size(A, 1), n)
    return pivots ? (R, Tuple(pivs)) : R
end

function _pivot_cols_f2(A::AbstractMatrix{FpElem{2}})
    rows, n = _pack_f2(A)
    return _gauss_jordan_f2!(rows, n)
end

function _pivot_cols_f2(A::SparseMatrixCSC{FpElem{2},Int})
    rows, n = _pack_f2(A)
    return _gauss_jordan_f2!(rows, n)
end

function _nullspace_f2(A::AbstractMatrix{FpElem{2}})
    m, n = size(A)
    rows, _ = _pack_f2(A)
    pivs = _gauss_jordan_f2!(rows, n)

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

    nfree = length(free)
    Z = Matrix{FpElem{2}}(undef, n, nfree)
    z = FpElem{2}(0)
    o = FpElem{2}(1)
    @inbounds for j in 1:n
        for k in 1:nfree
            Z[j, k] = z
        end
    end
    nfree == 0 && return Z

    # Free variable basis vectors.
    @inbounds for (k, jf) in enumerate(free)
        Z[jf, k] = o
    end

    # For each pivot row, set pivot coordinate from free coordinates.
    @inbounds for (row, pcol) in enumerate(pivs)
        prow = rows[row]
        for (k, jf) in enumerate(free)
            if _f2_getbit(prow, jf) == 1
                Z[pcol, k] = o
            end
        end
    end
    return Z
end

function _nullspace_f2(A::SparseMatrixCSC{FpElem{2},Int})
    m, n = size(A)
    rows, _ = _pack_f2(A)
    pivs = _gauss_jordan_f2!(rows, n)

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

    nfree = length(free)
    Z = Matrix{FpElem{2}}(undef, n, nfree)
    z = FpElem{2}(0)
    o = FpElem{2}(1)
    @inbounds for j in 1:n
        for k in 1:nfree
            Z[j, k] = z
        end
    end
    nfree == 0 && return Z

    @inbounds for (k, jf) in enumerate(free)
        Z[jf, k] = o
    end

    @inbounds for (row, pcol) in enumerate(pivs)
        prow = rows[row]
        for (k, jf) in enumerate(free)
            if _f2_getbit(prow, jf) == 1
                Z[pcol, k] = o
            end
        end
    end
    return Z
end

_nullspace_f2(A::Transpose{FpElem{2},<:SparseMatrixCSC{FpElem{2},Int}}) = _nullspace_f2(sparse(A))
_nullspace_f2(A::Adjoint{FpElem{2},<:SparseMatrixCSC{FpElem{2},Int}})  = _nullspace_f2(sparse(A))

function _rank_restricted_f2(A::SparseMatrixCSC{FpElem{2},Int},
                             rows::AbstractVector{Int},
                             cols::AbstractVector{Int};
                             check::Bool=false)
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

    nb = _f2_blocks(nc)
    subrows = [fill(UInt64(0), nb) for _ in 1:nr]
    @inbounds for (jloc, col) in enumerate(cols)
        for ptr in A.colptr[col]:(A.colptr[col + 1] - 1)
            r = A.rowval[ptr]
            i = _row_lookup(loc, r)
            if i != 0 && A.nzval[ptr].val != 0
                _f2_setbit!(subrows[i], jloc)
            end
        end
    end

    pivs = _gauss_jordan_f2!(subrows, nc)
    return length(pivs)
end

function _solve_fullcolumn_f2(B::AbstractMatrix{FpElem{2}},
                              Y::AbstractVecOrMat{FpElem{2}};
                              check_rhs::Bool=true,
                              cache::Bool=true,
                              factor::Union{Nothing,F2FullColumnFactor}=nothing)
    want_vec = false
    Ymat = Y
    if Y isa AbstractVector
        want_vec = true
        Ymat = reshape(Y, :, 1)
    end

    m, n = size(B)
    size(Ymat, 1) == m || throw(DimensionMismatch("B and Y must have same row count"))
    rhs = size(Ymat, 2)

    # Try cached factor first (if allowed).
    fact = factor
    if fact === nothing && cache && Base.ismutabletype(typeof(B))
        fact = get(_F2_FULLCOLUMN_FACTOR_CACHE, B, nothing)
    end

    if fact === nothing
        # Build factor: choose pivot rows from B^T and invert the minor.
        pivs = _pivot_cols_f2(transpose(B))
        if length(pivs) != n
            error("solve_fullcolumn_f2: expected full column rank, got rank $(length(pivs)) < $n")
        end
        Bsub = Matrix{FpElem{2}}(B[pivs, :])
        invB = _f2_inverse_packed(Bsub)
        fact = F2FullColumnFactor(pivs, invB, n)
        if cache && Base.ismutabletype(typeof(B))
            _F2_FULLCOLUMN_FACTOR_CACHE[B] = fact
        end
    end

    X = Matrix{FpElem{2}}(undef, n, rhs)
    z = FpElem{2}(0)
    o = FpElem{2}(1)
    @inbounds for i in 1:n, j in 1:rhs
        X[i, j] = z
    end

    # Compute X = invB * Y[rows, :]
    rows = fact.rows
    invB = fact.invB
    for j in 1:rhs
        # Build packed vector of Y[rows, j].
        v = fill(UInt64(0), _f2_blocks(n))
        @inbounds for i in 1:n
            if Ymat[rows[i], j].val != 0
                _f2_setbit!(v, i)
            end
        end

        @inbounds for i in 1:n
            acc = UInt64(0)
            for k in eachindex(invB[i])
                acc = xor(acc, invB[i][k] & v[k])
            end
            # parity of acc
            acc = xor(acc, acc >>> 32)
            acc = xor(acc, acc >>> 16)
            acc = xor(acc, acc >>> 8)
            acc = xor(acc, acc >>> 4)
            acc = xor(acc, acc >>> 2)
            acc = xor(acc, acc >>> 1)
            if acc & UInt64(1) != 0
                X[i, j] = o
            end
        end
    end

    if check_rhs
        # Verify B * X == Y (over F2).
        for i in 1:m
            for j in 1:rhs
                acc = 0
                @inbounds for k in 1:n
                    acc = xor(acc, (B[i, k].val & X[k, j].val))
                end
                acc == Ymat[i, j].val || error("solve_fullcolumn_f2: RHS check failed")
            end
        end
    end

    return want_vec ? vec(X) : X
end

function _solve_fullcolumn_f2(B::SparseMatrixCSC{FpElem{2},Int},
                              Y::AbstractVecOrMat{FpElem{2}};
                              check_rhs::Bool=true,
                              cache::Bool=true,
                              factor::Union{Nothing,F2FullColumnFactor}=nothing)
    want_vec = false
    Ymat = Y
    if Y isa AbstractVector
        want_vec = true
        Ymat = reshape(Y, :, 1)
    end

    m, n = size(B)
    size(Ymat, 1) == m || throw(DimensionMismatch("B and Y must have same row count"))
    rhs = size(Ymat, 2)

    fact = factor
    if fact === nothing && cache && Base.ismutabletype(typeof(B))
        fact = get(_F2_FULLCOLUMN_FACTOR_CACHE, B, nothing)
    end

    if fact === nothing
        Bt = sparse(transpose(B))
        pivs = _pivot_cols_f2(Bt)
        if length(pivs) != n
            error("solve_fullcolumn_f2: expected full column rank, got rank $(length(pivs)) < $n")
        end
        Bsub = Matrix{FpElem{2}}(B[pivs, :])
        invB = _f2_inverse_packed(Bsub)
        fact = F2FullColumnFactor(pivs, invB, n)
        if cache && Base.ismutabletype(typeof(B))
            _F2_FULLCOLUMN_FACTOR_CACHE[B] = fact
        end
    end

    X = Matrix{FpElem{2}}(undef, n, rhs)
    z = FpElem{2}(0)
    o = FpElem{2}(1)
    @inbounds for i in 1:n, j in 1:rhs
        X[i, j] = z
    end

    rows = fact.rows
    invB = fact.invB
    for j in 1:rhs
        v = fill(UInt64(0), _f2_blocks(n))
        @inbounds for i in 1:n
            if Ymat[rows[i], j].val != 0
                _f2_setbit!(v, i)
            end
        end

        @inbounds for i in 1:n
            acc = UInt64(0)
            for k in eachindex(invB[i])
                acc = xor(acc, invB[i][k] & v[k])
            end
            acc = xor(acc, acc >>> 32)
            acc = xor(acc, acc >>> 16)
            acc = xor(acc, acc >>> 8)
            acc = xor(acc, acc >>> 4)
            acc = xor(acc, acc >>> 2)
            acc = xor(acc, acc >>> 1)
            if acc & UInt64(1) != 0
                X[i, j] = o
            end
        end
    end

    if check_rhs
        for i in 1:m
            for j in 1:rhs
                acc = 0
                @inbounds for k in 1:n
                    acc = xor(acc, (B[i, k].val & X[k, j].val))
                end
                acc == Ymat[i, j].val || error("solve_fullcolumn_f2: RHS check failed")
            end
        end
    end

    return want_vec ? vec(X) : X
end
