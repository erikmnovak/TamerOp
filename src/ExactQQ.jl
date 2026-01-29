module ExactQQ
# Exact linear algebra over QQ. If Nemo.jl is present we use fmpq_mat; otherwise
# a pure-Julia exact fallback is provided here.  ASCII-only version.

using LinearAlgebra
using SparseArrays
using ..CoreModules: QQ

# Nemo integration is provided via a package extension.
# The base package always provides pure-Julia exact routines.
const _NEMO_ENABLED = Ref(false)

have_nemo() = _NEMO_ENABLED[]

# Hook: an extension may overload this to convert A to a backend-specific
# matrix representation (e.g. Nemo.fmpq_mat). The fallback is identity.
@inline _to_backend_matrix(A) = A

# Hooked backends:
# - fallback methods live on AbstractMatrix{QQ}
# - Nemo extension will add methods on Nemo.fmpq_mat

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

function rrefQQ(A::AbstractMatrix{QQ}; pivots::Bool=true)
    return _rref_backend(_to_backend_matrix(A); pivots=pivots)
end

_rank_backend(A::AbstractMatrix{QQ}) = length(last(rrefQQ(A; pivots=true)))

rankQQ(A::AbstractMatrix{QQ}) = _rank_backend(_to_backend_matrix(A))
rankQQ(A::AbstractMatrix{Float64}) = rankQQ(Matrix{QQ}(A))

function _nullspace_backend(A::AbstractMatrix{QQ})
    R, pivs = rrefQQ(A; pivots=true)
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

nullspaceQQ(A::AbstractMatrix{QQ}) = _nullspace_backend(_to_backend_matrix(A))

function colspaceQQ(A::AbstractMatrix{QQ})
    pivs = pivot_columnsQQ(A)
    return A[:, pivs]
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

# Cache of factors keyed by the left matrix B.
# IMPORTANT: values do not reference the key, so WeakKeyDict works correctly.
const _FULLCOLUMN_FACTOR_CACHE = WeakKeyDict{Any,Any}()

@inline _can_weak_cache_key(x) = Base.ismutabletype(typeof(x))

# ---------------------------------------------------------------
# Pivot-column detection for QQ matrices (dense or sparse)
# ---------------------------------------------------------------

# Dense pivot columns via rrefQQ
function pivot_columnsQQ(A::AbstractMatrix{QQ})::Vector{Int}
    _, pivs = rrefQQ(A)
    return collect(pivs)
end

# Sparse pivot columns via sparse RREF streaming
# Sparse pivot columns via sparse RREF streaming
function pivot_columnsQQ(A::SparseMatrixCSC{QQ,Int})::Vector{Int}
    m, n = size(A)
    if m == 0 || n == 0
        return Int[]
    end
    R = SparseRREF{QQ}(n)
    rows = _sparse_rows(A)
    maxrank = min(m, n)
    for i in 1:m
        _sparse_rref_push_homogeneous!(R, rows[i])
        if length(R.pivot_cols) == maxrank
            break
        end
    end
    return sort!(copy(R.pivot_cols))
end


# Avoid densification for transpose/adjoint sparse wrappers
pivot_columnsQQ(A::Transpose{QQ,<:SparseMatrixCSC{QQ,Int}}) = pivot_columnsQQ(sparse(A))
pivot_columnsQQ(A::Adjoint{QQ,<:SparseMatrixCSC{QQ,Int}})  = pivot_columnsQQ(sparse(A))

"""
    factor_fullcolumnQQ(B::AbstractMatrix{<:QQ}) -> FullColumnFactor{QQ}

Build reusable factorization data for solving `B * X = Y` when B has full column rank.

Most users should call `solve_fullcolumnQQ(B,Y)`; caching is automatic for mutable matrices.
"""
function factor_fullcolumnQQ(B::AbstractMatrix{<:QQ})::FullColumnFactor{QQ}
    m, n = size(B)
    n == 0 && return FullColumnFactor{QQ}(Int[], Matrix{QQ}(I, 0, 0))

    rows = pivot_columnsQQ(transpose(B))
    if length(rows) != n
        error("solve_fullcolumnQQ: expected full column rank, got rank $(length(rows)) < $n")
    end

    Bsub = Matrix{QQ}(B[rows, :])
    invB = _solve_fullcolumn_rrefQQ(Bsub, Matrix{QQ}(I, n, n))
    return FullColumnFactor{QQ}(rows, invB)
end

# Ground-truth RREF solver, always exact, no caching
function _solve_fullcolumn_rrefQQ(B::AbstractMatrix{<:QQ}, Y::AbstractVecOrMat{<:QQ})
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
    R, pivs = rrefQQ(By)

    for j in 1:n
        (j in pivs) || error("expected full column rank; missing pivot in column $j")
    end
    for pj in pivs
        pj > n && error("right-hand side is not in column space of B")
    end

    X = R[1:n, n+1:end]
    return want_vec ? vec(X) : X
end

# Fast solve using factor data
function _solve_fullcolumn_factorQQ(B::AbstractMatrix{<:QQ}, fac::FullColumnFactor{QQ},
                                    Y::AbstractVecOrMat{<:QQ};
                                    check_rhs::Bool=true)
    want_vec = false
    Ymat = Y
    if Y isa AbstractVector
        want_vec = true
        Ymat = reshape(Y, :, 1)
    end

    m, n = size(B)
    size(Ymat, 1) == m || throw(DimensionMismatch("B and Y must have same row count"))
    length(fac.rows) == n || error("stale FullColumnFactor: wrong row set length")

    Ysub = Matrix{QQ}(Ymat[fac.rows, :])
    X = fac.invB * Ysub

    if check_rhs
        # Verify B*X == Y
        if B isa Matrix{QQ}
            k = size(X, 2)
            @inbounds for j in 1:k
                for i in 1:m
                    s = zero(QQ)
                    for t in 1:n
                        s += B[i, t] * X[t, j]
                    end
                    s == Ymat[i, j] || error("right-hand side is not in column space of B")
                end
            end
        else
            Matrix{QQ}(B * X) == Matrix{QQ}(Ymat) || error("right-hand side is not in column space of B")
        end
    end

    return want_vec ? vec(X) : X
end

"""
    solve_fullcolumnQQ(B, Y; cache=true, factor=nothing, check_rhs=true)

Solve `B*X = Y` over QQ under the assumption that B has full column rank.

If cache=true, a reusable factor is used; for mutable B (e.g. Matrix) it is cached
so repeated solves do not redo elimination.
"""
function solve_fullcolumnQQ(B::AbstractMatrix{<:QQ}, Y::AbstractVecOrMat{<:QQ};
                            cache::Bool=true,
                            factor::Union{Nothing,FullColumnFactor{QQ}}=nothing,
                            check_rhs::Bool=true)
    if cache
        fac = factor
        if fac === nothing
            if _can_weak_cache_key(B)
                fac = get!(_FULLCOLUMN_FACTOR_CACHE, B) do
                    factor_fullcolumnQQ(B)
                end
            else
                fac = factor_fullcolumnQQ(B)
            end
        end
        return _solve_fullcolumn_factorQQ(B, fac, Y; check_rhs=check_rhs)
    end

    return _solve_fullcolumn_rrefQQ(B, Y)
end

"""
    clear_fullcolumn_cache!()

Drop cached FullColumnFactor objects (useful in long sessions).
"""
function clear_fullcolumn_cache!()
    empty!(_FULLCOLUMN_FACTOR_CACHE)
    return nothing
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
#   * SparseRREF{K}: a streaming RREF builder using merge-based sparse axpy.
#
# This layer is internal but sits on several hot paths (Hom computations,
# derived-functor lifting, and sparse linear solves).  Keeping it allocation-
# light has an outsized impact on overall library performance.
# ---------------------------------------------------------------------------


"""
    SparseRow{K}

A sparse row over an exact field `K` stored as parallel vectors:

  * `idx`: strictly increasing 1-based column indices
  * `val`: nonzero coefficients aligned with `idx`

The invariants are maintained by the elimination routines; callers that build
rows manually should either ensure the invariants, or call
`normalize_sparse_row!(idx, val)` before passing the row to the eliminator.
"""
mutable struct SparseRow{K}
    idx::Vector{Int}
    val::Vector{K}
end

SparseRow{K}() where {K} = SparseRow{K}(Int[], K[])

Base.isempty(r::SparseRow) = isempty(r.idx)
Base.length(r::SparseRow) = length(r.idx)
Base.copy(r::SparseRow{K}) where {K} = SparseRow{K}(copy(r.idx), copy(r.val))

"""
    normalize_sparse_row!(cols, vals)

In-place normalization for a sparse row represented by `(cols, vals)`:

  * sorts by `cols` using an insertion sort (fast for small rows),
  * combines duplicate columns by summation,
  * drops explicit zeros.

This helper is used by callers that assemble rows via repeated contributions.
"""
function normalize_sparse_row!(cols::Vector{Int}, vals::Vector{K}) where {K}
    @assert length(cols) == length(vals)
    n = length(cols)
    n <= 1 && return
    # Insertion sort by column index, carrying values along.
    @inbounds for i in 2:n
        c = cols[i]
        v = vals[i]
        j = i - 1
        while j >= 1 && cols[j] > c
            cols[j + 1] = cols[j]
            vals[j + 1] = vals[j]
            j -= 1
        end
        cols[j + 1] = c
        vals[j + 1] = v
    end

    # Combine duplicates and drop zeros.
    w = 0
    i = 1
    @inbounds while i <= n
        c = cols[i]
        s = vals[i]
        i += 1
        while i <= n && cols[i] == c
            s += vals[i]
            i += 1
        end
        if !iszero(s)
            w += 1
            cols[w] = c
            vals[w] = s
        end
    end
    resize!(cols, w)
    resize!(vals, w)
    return
end

@inline function _row_coeff(row::SparseRow{K}, j::Int) where {K}
    n = length(row.idx)
    n == 0 && return zero(K)
    if j < row.idx[1] || j > row.idx[end]
        return zero(K)
    end
    k = searchsortedfirst(row.idx, j)
    return (k <= n && row.idx[k] == j) ? row.val[k] : zero(K)
end

# Internal: row := row + a * other, with sparse merge and buffer swapping.
# Returns the scratch buffers that should be reused for the next call.
function _row_axpy!(
    row::SparseRow{K},
    a::K,
    other::SparseRow{K},
    tmp_idx::Vector{Int},
    tmp_val::Vector{K},
) where {K}
    iszero(a) && return tmp_idx, tmp_val

    empty!(tmp_idx)
    empty!(tmp_val)
    sizehint!(tmp_idx, length(row.idx) + length(other.idx))
    sizehint!(tmp_val, length(row.idx) + length(other.idx))

    i = 1
    j = 1
    m = length(row.idx)
    n = length(other.idx)

    @inbounds while i <= m || j <= n
        if j > n || (i <= m && row.idx[i] < other.idx[j])
            push!(tmp_idx, row.idx[i])
            push!(tmp_val, row.val[i])
            i += 1
        elseif i > m || other.idx[j] < row.idx[i]
            v = a * other.val[j]
            if !iszero(v)
                push!(tmp_idx, other.idx[j])
                push!(tmp_val, v)
            end
            j += 1
        else
            v = row.val[i] + a * other.val[j]
            if !iszero(v)
                push!(tmp_idx, row.idx[i])
                push!(tmp_val, v)
            end
            i += 1
            j += 1
        end
    end

    # Swap buffers to avoid copying: the caller owns tmp_idx/tmp_val and will
    # get back the old row buffers for reuse.
    old_idx = row.idx
    old_val = row.val
    row.idx = tmp_idx
    row.val = tmp_val
    tmp_idx = old_idx
    tmp_val = old_val
    empty!(tmp_idx)
    empty!(tmp_val)
    return tmp_idx, tmp_val
end

"""
    SparseRREF{K}(nvars)

Streaming sparse reduced row-echelon form builder over an exact field `K`.

The pivot bookkeeping uses a dense `Vector{Int}` of length `nvars`:
`pivot_pos[j] == 0` means column `j` is free, and otherwise stores the index
of the pivot row whose pivot is in column `j`.

This object is intentionally mutable and reuses scratch buffers to reduce
allocation in tight elimination loops.
"""
mutable struct SparseRREF{K}
    nvars::Int
    pivot_pos::Vector{Int}
    pivot_cols::Vector{Int}
    pivot_rows::Vector{SparseRow{K}}

    scratch_cols::Vector{Int}
    scratch_coeffs::Vector{K}

    tmp_idx::Vector{Int}
    tmp_val::Vector{K}
end

function SparseRREF{K}(nvars::Int) where {K}
    return SparseRREF{K}(
        nvars,
        zeros(Int, nvars),
        Int[],
        SparseRow{K}[],
        Int[],
        K[],
        Int[],
        K[],
    )
end

@inline _rref_rank(R::SparseRREF) = length(R.pivot_cols)

"""
    _sparse_rref_push_homogeneous!(R, row) -> Bool

Push one homogeneous equation `row * x = 0` into the sparse RREF builder `R`.

The input `row` is reduced in place. The function returns `true` iff this row
introduces a new pivot (i.e., increases the rank).
"""
function _sparse_rref_push_homogeneous!(R::SparseRREF{K}, row::SparseRow{K})::Bool where {K}
    isempty(row) && return false

    empty!(R.scratch_cols)
    empty!(R.scratch_coeffs)

    # Gather pivot columns present in the row (their coefficients are stable
    # during reduction because pivot rows are kept in strict RREF).
    @inbounds for t in eachindex(row.idx)
        j = row.idx[t]
        pos = R.pivot_pos[j]
        if pos != 0
            push!(R.scratch_cols, j)
            push!(R.scratch_coeffs, row.val[t])
        end
    end

    # Eliminate all previously known pivots from the new row.
    @inbounds for k in eachindex(R.scratch_cols)
        j = R.scratch_cols[k]
        c = R.scratch_coeffs[k]
        if !iszero(c)
            prow = R.pivot_rows[R.pivot_pos[j]]
            R.tmp_idx, R.tmp_val = _row_axpy!(row, -c, prow, R.tmp_idx, R.tmp_val)
        end
    end

    isempty(row) && return false

    # New pivot is the leftmost nonzero after reduction.
    p = row.idx[1]
    @assert R.pivot_pos[p] == 0
    a = row.val[1]
    inv_a = inv(a)
    @inbounds for t in eachindex(row.val)
        row.val[t] *= inv_a
    end

    # Eliminate this pivot column from all existing pivot rows to maintain RREF.
    for pos in 1:length(R.pivot_rows)
        prow = R.pivot_rows[pos]
        c = _row_coeff(prow, p)
        if !iszero(c)
            R.tmp_idx, R.tmp_val = _row_axpy!(prow, -c, row, R.tmp_idx, R.tmp_val)
        end
    end

    push!(R.pivot_cols, p)
    push!(R.pivot_rows, copy(row))
    R.pivot_pos[p] = length(R.pivot_rows)
    return true
end

"""
    SparseRREFAugmented{K}(nvars, nrhs)

Streaming sparse RREF builder for an augmented system `[A | B]` over an exact
field `K`, where `A` has `nvars` columns and `B` has `nrhs` columns.

Each pivot row stores its accompanying right-hand-side row in `pivot_rhs`.
"""
mutable struct SparseRREFAugmented{K}
    rref::SparseRREF{K}
    nrhs::Int
    pivot_rhs::Vector{Vector{K}}
end

function SparseRREFAugmented{K}(nvars::Int, nrhs::Int) where {K}
    return SparseRREFAugmented{K}(SparseRREF{K}(nvars), nrhs, Vector{Vector{K}}())
end

"""
    _sparse_rref_push_augmented!(R, row, rhs) -> Symbol

Push one augmented equation `row * x = rhs` into the sparse augmented RREF
builder `R`.

The input `row` and `rhs` are reduced in place. Return values:

  * `:pivot`        a new pivot was added (rank increased),
  * `:dependent`    the equation was redundant,
  * `:inconsistent` the system is inconsistent.
"""
function _sparse_rref_push_augmented!(
    R::SparseRREFAugmented{K},
    row::SparseRow{K},
    rhs::Vector{K},
)::Symbol where {K}
    RR = R.rref
    @assert length(rhs) == R.nrhs

    isempty(row) && return all(iszero, rhs) ? :dependent : :inconsistent

    empty!(RR.scratch_cols)
    empty!(RR.scratch_coeffs)

    @inbounds for t in eachindex(row.idx)
        j = row.idx[t]
        pos = RR.pivot_pos[j]
        if pos != 0
            push!(RR.scratch_cols, j)
            push!(RR.scratch_coeffs, row.val[t])
        end
    end

    # Reduce against existing pivots.
    @inbounds for k in eachindex(RR.scratch_cols)
        j = RR.scratch_cols[k]
        c = RR.scratch_coeffs[k]
        if !iszero(c)
            pos = RR.pivot_pos[j]
            prow = RR.pivot_rows[pos]
            prhs = R.pivot_rhs[pos]

            RR.tmp_idx, RR.tmp_val = _row_axpy!(row, -c, prow, RR.tmp_idx, RR.tmp_val)
            @inbounds for t in 1:R.nrhs
                rhs[t] -= c * prhs[t]
            end
        end
    end

    if isempty(row)
        return all(iszero, rhs) ? :dependent : :inconsistent
    end

    # New pivot and normalization.
    p = row.idx[1]
    @assert RR.pivot_pos[p] == 0
    a = row.val[1]
    inv_a = inv(a)
    @inbounds for t in eachindex(row.val)
        row.val[t] *= inv_a
    end
    @inbounds for t in 1:R.nrhs
        rhs[t] *= inv_a
    end

    # Eliminate this pivot from existing pivot rows.
    for pos in 1:length(RR.pivot_rows)
        prow = RR.pivot_rows[pos]
        c = _row_coeff(prow, p)
        if !iszero(c)
            RR.tmp_idx, RR.tmp_val = _row_axpy!(prow, -c, row, RR.tmp_idx, RR.tmp_val)
            prhs = R.pivot_rhs[pos]
            @inbounds for t in 1:R.nrhs
                prhs[t] -= c * rhs[t]
            end
        end
    end

    push!(RR.pivot_cols, p)
    push!(RR.pivot_rows, copy(row))
    push!(R.pivot_rhs, copy(rhs))
    RR.pivot_pos[p] = length(RR.pivot_rows)
    return :pivot
end

"""
    _nullspace_from_pivots(R, nvars) -> Matrix{QQ}

Return a basis matrix for the nullspace of the homogeneous system whose RREF
is stored in `R`.

The returned matrix has size `(nvars, nfree)`, where each column is a basis
vector for the nullspace.
"""
function _nullspace_from_pivots(R::SparseRREF{QQ}, nvars::Int)
    @assert nvars == R.nvars

    free_cols = Int[]
    sizehint!(free_cols, nvars - _rref_rank(R))
    for j in 1:nvars
        if R.pivot_pos[j] == 0
            push!(free_cols, j)
        end
    end

    nfree = length(free_cols)
    Z = zeros(QQ, nvars, nfree)
    if nfree == 0
        return Z
    end

    free_pos = zeros(Int, nvars)
    for (k, j) in enumerate(free_cols)
        free_pos[j] = k
        Z[j, k] = one(QQ)
    end

    # For each pivot equation x_p + sum_{free j} a_{p,j} x_j = 0, we set
    # x_p = -a_{p,j} on the basis vector corresponding to free variable j.
    for pos in 1:length(R.pivot_cols)
        p = R.pivot_cols[pos]
        prow = R.pivot_rows[pos]
        @inbounds for t in eachindex(prow.idx)
            j = prow.idx[t]
            k = free_pos[j]
            if k != 0
                Z[p, k] = -prow.val[t]
            end
        end
    end
    return Z
end

# Convert a SparseMatrixCSC into explicit SparseRow objects (row-wise storage).
# This is O(nnz) and avoids nested Dict allocations.
function _sparse_rows(A::SparseMatrixCSC{K,Int}) where {K}
    m, n = size(A)

    row_nnz = zeros(Int, m)
    @inbounds for col in 1:n
        for ptr in A.colptr[col]:(A.colptr[col + 1] - 1)
            row_nnz[A.rowval[ptr]] += 1
        end
    end

    rows_idx = [Int[] for _ in 1:m]
    rows_val = [K[] for _ in 1:m]
    for i in 1:m
        sizehint!(rows_idx[i], row_nnz[i])
        sizehint!(rows_val[i], row_nnz[i])
    end

    @inbounds for col in 1:n
        for ptr in A.colptr[col]:(A.colptr[col + 1] - 1)
            r = A.rowval[ptr]
            v = A.nzval[ptr]
            if !iszero(v)
                push!(rows_idx[r], col)
                push!(rows_val[r], v)
            end
        end
    end

    rows = Vector{SparseRow{K}}(undef, m)
    for i in 1:m
        rows[i] = SparseRow{K}(rows_idx[i], rows_val[i])
    end
    return rows
end

function rankQQ(A::SparseMatrixCSC{QQ,Int})
    m, n = size(A)
    if m == 0 || n == 0
        return 0
    end

    R = SparseRREF{QQ}(n)
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

function nullspaceQQ(A::SparseMatrixCSC{QQ,Int})
    m, n = size(A)
    R = SparseRREF{QQ}(n)
    rows = _sparse_rows(A)
    for i in 1:m
        _sparse_rref_push_homogeneous!(R, rows[i])
    end
    return _nullspace_from_pivots(R, n)
end

function rankQQ_restricted(
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

    # Map global row indices to local 1..nr indices (sparse dictionary: avoids O(m) memory).
    rowpos = Dict{Int,Int}()
    sizehint!(rowpos, nr)
    for (i, r) in enumerate(rows)
        rowpos[r] = i
    end

    row_idx = [Int[] for _ in 1:nr]
    row_val = [QQ[] for _ in 1:nr]

    # Iterate over selected columns once. We remap columns to local indices
    # 1..nc so the RREF builder does not need to allocate pivot_pos of length
    # size(A,2) when only a small subset is used.
    @inbounds for (jloc, col) in enumerate(cols)
        for ptr in A.colptr[col]:(A.colptr[col + 1] - 1)
            r = A.rowval[ptr]
            i = get(rowpos, r, 0)
            if i != 0
                v = A.nzval[ptr]
                if !iszero(v)
                    push!(row_idx[i], jloc)
                    push!(row_val[i], v)
                end
            end
        end
    end

    R = SparseRREF{QQ}(nc)
    maxrank = min(nr, nc)
    for i in 1:nr
        _sparse_rref_push_homogeneous!(R, SparseRow{QQ}(row_idx[i], row_val[i]))
        if _rref_rank(R) == maxrank
            break
        end
    end
    return _rref_rank(R)
end


"""
    rank_modp_sparse(A::SparseMatrixCSC{QQ}, p::Int) -> Int

Compute the rank of a sparse QQ-matrix after reducing coefficients modulo `p`.

This is intended as a *fast certificate/diagnostic* tool:
- For matrices with integer coefficients, rank_modp_sparse(A,p) <= rankQQ(A).
- If rank_modp_sparse(A,p) == ncols(A), then A has full column rank over QQ.

Notes
- Choose `p` as a reasonably large prime (e.g. around 1e6) to reduce the chance
  of accidental rank drops.
- If any denominator in A is divisible by p, reduction is undefined and this
  function throws DomainError.
"""
function rank_modp_sparse(A::SparseMatrixCSC{QQ,Int}, p::Int)::Int
    m, n = size(A)
    if m == 0 || n == 0
        return 0
    end

    # Build per-row dictionaries over Int mod p.
    row_nnz = zeros(Int, m)
    @inbounds for col in 1:n
        for ptr in A.colptr[col]:(A.colptr[col + 1] - 1)
            row_nnz[A.rowval[ptr]] += 1
        end
    end

    rows = [Dict{Int,Int}() for _ in 1:m]
    @inbounds for i in 1:m
        sizehint!(rows[i], row_nnz[i])
    end

    @inbounds for col in 1:n
        for ptr in A.colptr[col]:(A.colptr[col + 1] - 1)
            r = A.rowval[ptr]
            v = A.nzval[ptr]
            if v != 0
                a = _modp_qQ(v, p)
                if a != 0
                    rows[r][col] = a
                end
            end
        end
    end

    # Incremental elimination over GF(p) using the same "pivot dict" idea.
    pivots = Dict{Int,Dict{Int,Int}}()
    scratch = Int[]

    for i in 1:m
        row = rows[i]
        if isempty(row)
            continue
        end

        # eliminate existing pivots
        empty!(scratch)
        for (j, _) in row
            if haskey(pivots, j)
                push!(scratch, j)
            end
        end
        for j in scratch
            if haskey(row, j)
                c = row[j] % p
                if c != 0
                    prow = pivots[j]
                    for (k, vk) in prow
                        newv = mod(get(row, k, 0) - c * vk, p)
                        if newv == 0
                            if haskey(row, k)
                                delete!(row, k)
                            end
                        else
                            row[k] = newv
                        end
                    end
                end
            end
        end
        if isempty(row)
            continue
        end

        piv = minimum(keys(row))
        a = row[piv] % p
        if a == 0
            # Should not happen if we drop zeros properly, but guard anyway.
            delete!(row, piv)
            continue
        end

        inv_a = invmod(a, p)
        for (k, vk) in row
            row[k] = mod(vk * inv_a, p)
        end

        # eliminate new pivot from existing pivot rows
        for (q, prow) in pivots
            if q == piv
                continue
            end
            if haskey(prow, piv)
                c = prow[piv] % p
                if c != 0
                    for (k, vk) in row
                        newv = mod(get(prow, k, 0) - c * vk, p)
                        if newv == 0
                            if haskey(prow, k)
                                delete!(prow, k)
                            end
                        else
                            prow[k] = newv
                        end
                    end
                end
            end
        end

        pivots[piv] = copy(row)

        if length(pivots) == min(m, n)
            break
        end
    end

    return length(pivots)
end

# ---------------------------------------------------------------
# Fast rank computations for dimension-only queries
# ---------------------------------------------------------------

const DEFAULT_MODULAR_PRIMES = Int[
    998244353, 1000000007, 1000000009, 1004535809,
    1045430273, 1053818881, 1224736769, 1300234241,
]

"""
    rank_modp_dense(A::AbstractMatrix{QQ}, p::Int) -> Int

Rank over F_p for dense matrices by Gaussian elimination.
Used by rankQQ_dim for fast dimension computations.
"""
function rank_modp_dense(A::AbstractMatrix{QQ}, p::Int)::Int
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

rank_modp(A::AbstractMatrix{QQ}, p::Int) = rank_modp_dense(A, p)
rank_modp(A::SparseMatrixCSC{QQ,Int}, p::Int) = rank_modp_sparse(A, p)
rank_modp(A::Transpose{QQ,<:SparseMatrixCSC{QQ,Int}}, p::Int) = rank_modp_sparse(sparse(A), p)
rank_modp(A::Adjoint{QQ,<:SparseMatrixCSC{QQ,Int}}, p::Int)  = rank_modp_sparse(sparse(A), p)

"""
    rankQQ_dim(A; backend=:auto, max_primes=4, primes=DEFAULT_MODULAR_PRIMES,
                 small_threshold=20_000) -> Int

Fast rank intended for dimension-only queries:
- backend=:exact  uses exact rankQQ
- backend=:modular uses ranks mod several primes
- backend=:auto uses exact for small matrices, modular otherwise
"""
function rankQQ_dim(A::AbstractMatrix{QQ};
                    backend::Symbol=:auto,
                    max_primes::Int=4,
                    primes::Vector{Int}=DEFAULT_MODULAR_PRIMES,
                    small_threshold::Int=20_000)::Int
    m, n = size(A)
    (m == 0 || n == 0) && return 0

    backend == :exact && return rankQQ(A)

    if backend == :auto
        (m*n <= small_threshold) && return rankQQ(A)
        backend = :modular
    end

    backend != :modular && error("rankQQ_dim: unsupported backend $(backend)")

    r = 0
    used = 0
    for p in primes
        used += 1
        try
            r = max(r, rank_modp(A, p))
        catch
            continue
        end
        r == min(m,n) && break
        used >= max_primes && break
    end
    return r
end

nullityQQ_dim(A::AbstractMatrix{QQ}; kwargs...) = size(A,2) - rankQQ_dim(A; kwargs...)


export rankQQ, rankQQ_dim, nullityQQ_dim, nullspaceQQ, colspaceQQ, rrefQQ,
       solve_fullcolumnQQ, FullColumnFactor, factor_fullcolumnQQ,
       clear_fullcolumn_cache!, rank_modp_sparse, rank_modp_dense, have_nemo


end # module
