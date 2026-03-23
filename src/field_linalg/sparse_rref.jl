# -----------------------------------------------------------------------------
# field_linalg/sparse_rref.jl
#
# Scope:
#   Shared sparse exact elimination primitives used by multiple FieldLinAlg
#   engines and higher-level modules that rely on sparse row-space updates.
# Owns:
#   - sparse row data structures,
#   - REF/RREF streaming reducers,
#   - low-level exact sparse row arithmetic primitives.
# Does not own:
#   - backend routing,
#   - field-specific high-level solve/rank wrappers,
#   - public API entrypoints.
# Depends on:
#   - `thresholds.jl` for a small set of sparse-elimination gates,
#   - no later fragment; this file provides primitives consumed by QQ/non-QQ
#     engines and external subsystems like `FiniteFringe`.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Sparse streaming RREF (generic exact field)
# -----------------------------------------------------------------------------

mutable struct SparseRow{K}
    idx::Vector{Int}
    val::Vector{K}
end

SparseRow{K}() where {K} = SparseRow{K}(Int[], K[])

Base.isempty(r::SparseRow) = isempty(r.idx)
Base.length(r::SparseRow) = length(r.idx)
Base.copy(r::SparseRow{K}) where {K} = SparseRow{K}(copy(r.idx), copy(r.val))

mutable struct _SparseRowAccumulator{K}
    marks::Vector{UInt32}
    pos::Vector{Int}
    cols::Vector{Int}
    vals::Vector{K}
    tag::UInt32
end

function _SparseRowAccumulator{K}(nvars::Int) where {K}
    return _SparseRowAccumulator{K}(zeros(UInt32, nvars), zeros(Int, nvars), Int[], K[], UInt32(1))
end

function _reset_sparse_row_accumulator!(acc::_SparseRowAccumulator)
    empty!(acc.cols)
    empty!(acc.vals)
    tag = acc.tag + UInt32(1)
    if tag == 0
        fill!(acc.marks, 0)
        tag = UInt32(1)
    end
    acc.tag = tag
    return acc
end

@inline function _push_sparse_row_entry!(acc::_SparseRowAccumulator{K}, col::Int, v::K) where {K}
    iszero(v) && return acc
    @inbounds if acc.marks[col] != acc.tag
        acc.marks[col] = acc.tag
        pos = length(acc.cols) + 1
        acc.pos[col] = pos
        push!(acc.cols, col)
        push!(acc.vals, v)
    else
        pos = acc.pos[col]
        acc.vals[pos] += v
    end
    return acc
end

function _sort_parallel_insertion!(cols::Vector{Int}, vals)
    n = length(cols)
    n <= 1 && return
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
end

function _materialize_sparse_row!(row::SparseRow{K}, acc::_SparseRowAccumulator{K}) where {K}
    cols = acc.cols
    vals = acc.vals
    _sort_parallel_insertion!(cols, vals)

    n = length(cols)
    resize!(row.idx, n)
    resize!(row.val, n)

    w = 0
    @inbounds for i in 1:n
        v = vals[i]
        if !iszero(v)
            w += 1
            row.idx[w] = cols[i]
            row.val[w] = v
        end
    end

    resize!(row.idx, w)
    resize!(row.val, w)
    return row
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

function _row_axpy!(
    row::SparseRow{K},
    a::K,
    other::SparseRow{K},
    tmp_idx::Vector{Int},
    tmp_val::Vector{K},
) where {K}
    iszero(a) && return tmp_idx, tmp_val
    isempty(other) && return tmp_idx, tmp_val

    if isempty(row)
        empty!(tmp_idx)
        empty!(tmp_val)
        n = length(other.idx)
        resize!(tmp_idx, n)
        resize!(tmp_val, n)
        if a == one(K)
            @inbounds for t in 1:n
                tmp_idx[t] = other.idx[t]
                tmp_val[t] = other.val[t]
            end
        elseif a == -one(K)
            @inbounds for t in 1:n
                tmp_idx[t] = other.idx[t]
                tmp_val[t] = -other.val[t]
            end
        else
            @inbounds for t in 1:n
                tmp_idx[t] = other.idx[t]
                tmp_val[t] = a * other.val[t]
            end
        end
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

    empty!(tmp_idx)
    empty!(tmp_val)
    maxlen = length(row.idx) + length(other.idx)
    resize!(tmp_idx, maxlen)
    resize!(tmp_val, maxlen)

    i = 1
    j = 1
    m = length(row.idx)
    n = length(other.idx)
    w = 0

    if a == one(K)
        @inbounds while i <= m || j <= n
            if j > n || (i <= m && row.idx[i] < other.idx[j])
                w += 1
                tmp_idx[w] = row.idx[i]
                tmp_val[w] = row.val[i]
                i += 1
            elseif i > m || other.idx[j] < row.idx[i]
                v = other.val[j]
                if !iszero(v)
                    w += 1
                    tmp_idx[w] = other.idx[j]
                    tmp_val[w] = v
                end
                j += 1
            else
                v = row.val[i] + other.val[j]
                if !iszero(v)
                    w += 1
                    tmp_idx[w] = row.idx[i]
                    tmp_val[w] = v
                end
                i += 1
                j += 1
            end
        end
    elseif a == -one(K)
        @inbounds while i <= m || j <= n
            if j > n || (i <= m && row.idx[i] < other.idx[j])
                w += 1
                tmp_idx[w] = row.idx[i]
                tmp_val[w] = row.val[i]
                i += 1
            elseif i > m || other.idx[j] < row.idx[i]
                v = -other.val[j]
                if !iszero(v)
                    w += 1
                    tmp_idx[w] = other.idx[j]
                    tmp_val[w] = v
                end
                j += 1
            else
                v = row.val[i] - other.val[j]
                if !iszero(v)
                    w += 1
                    tmp_idx[w] = row.idx[i]
                    tmp_val[w] = v
                end
                i += 1
                j += 1
            end
        end
    else
        @inbounds while i <= m || j <= n
            if j > n || (i <= m && row.idx[i] < other.idx[j])
                w += 1
                tmp_idx[w] = row.idx[i]
                tmp_val[w] = row.val[i]
                i += 1
            elseif i > m || other.idx[j] < row.idx[i]
                v = a * other.val[j]
                if !iszero(v)
                    w += 1
                    tmp_idx[w] = other.idx[j]
                    tmp_val[w] = v
                end
                j += 1
            else
                v = row.val[i] + a * other.val[j]
                if !iszero(v)
                    w += 1
                    tmp_idx[w] = row.idx[i]
                    tmp_val[w] = v
                end
                i += 1
                j += 1
            end
        end
    end

    resize!(tmp_idx, w)
    resize!(tmp_val, w)
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

function _row_axpy_modp!(
    row::SparseRow{Int},
    a::Int,
    other::SparseRow{Int},
    p::Int,
    tmp_idx::Vector{Int},
    tmp_val::Vector{Int},
)
    a = mod(a, p)
    a == 0 && return tmp_idx, tmp_val

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
            v = row.val[i]
            if v != 0
                push!(tmp_idx, row.idx[i])
                push!(tmp_val, v)
            end
            i += 1
        elseif i > m || other.idx[j] < row.idx[i]
            v = mod(a * other.val[j], p)
            if v != 0
                push!(tmp_idx, other.idx[j])
                push!(tmp_val, v)
            end
            j += 1
        else
            v = mod(row.val[i] + a * other.val[j], p)
            if v != 0
                push!(tmp_idx, row.idx[i])
                push!(tmp_val, v)
            end
            i += 1
            j += 1
        end
    end

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

mutable struct _SparseREF{K}
    nvars::Int
    pivot_pos::Vector{Int}
    pivot_cols::Vector{Int}
    pivot_rows::Vector{SparseRow{K}}
    pivot_inv::Vector{K}
    scratch_cols::Vector{Int}
    scratch_coeffs::Vector{K}
    tmp_idx::Vector{Int}
    tmp_val::Vector{K}
end

function _SparseREF{K}(nvars::Int) where {K}
    return _SparseREF{K}(
        nvars,
        zeros(Int, nvars),
        Int[],
        SparseRow{K}[],
        K[],
        Int[],
        K[],
        Int[],
        K[],
    )
end

function _reset_sparse_ref!(R::_SparseREF)
    @inbounds for j in R.pivot_cols
        R.pivot_pos[j] = 0
    end
    empty!(R.pivot_cols)
    empty!(R.pivot_rows)
    empty!(R.pivot_inv)
    empty!(R.scratch_cols)
    empty!(R.scratch_coeffs)
    empty!(R.tmp_idx)
    empty!(R.tmp_val)
    return R
end

function _sparse_ref_push_homogeneous!(R::_SparseREF{K}, row::SparseRow{K})::Bool where {K}
    isempty(row) && return false

    while true
        empty!(R.scratch_cols)
        empty!(R.scratch_coeffs)

        @inbounds for t in eachindex(row.idx)
            j = row.idx[t]
            pos = R.pivot_pos[j]
            if pos != 0
                push!(R.scratch_cols, j)
                push!(R.scratch_coeffs, row.val[t])
            end
        end

        isempty(R.scratch_cols) && break

        @inbounds for k in eachindex(R.scratch_cols)
            j = R.scratch_cols[k]
            c = R.scratch_coeffs[k]
            if !iszero(c)
                pos = R.pivot_pos[j]
                prow = R.pivot_rows[pos]
                f = c * R.pivot_inv[pos]
                if !iszero(f)
                    R.tmp_idx, R.tmp_val = _row_axpy!(row, -f, prow, R.tmp_idx, R.tmp_val)
                end
            end
        end

        isempty(row) && return false
    end

    p = row.idx[1]
    @assert R.pivot_pos[p] == 0
    a = row.val[1]
    inv_a = inv(a)

    push!(R.pivot_cols, p)
    push!(R.pivot_rows, copy(row))
    push!(R.pivot_inv, inv_a)
    R.pivot_pos[p] = length(R.pivot_rows)
    return true
end

mutable struct _SparseRREF{K}
    nvars::Int
    pivot_pos::Vector{Int}
    pivot_cols::Vector{Int}
    pivot_rows::Vector{SparseRow{K}}
    col_rows::Vector{BitSet}
    touched_col_rows::Vector{Int}
    scratch_cols::Vector{Int}
    scratch_coeffs::Vector{K}
    scratch_rows::Vector{Int}
    tmp_idx::Vector{Int}
    tmp_val::Vector{K}
end

function _SparseRREF{K}(nvars::Int) where {K}
    return _SparseRREF{K}(
        nvars,
        zeros(Int, nvars),
        Int[],
        SparseRow{K}[],
        [BitSet() for _ in 1:nvars],
        Int[],
        Int[],
        K[],
        Int[],
        Int[],
        K[],
    )
end

@inline _rref_rank(R::_SparseRREF) = length(R.pivot_cols)

function _reset_sparse_rref!(R::_SparseRREF)
    @inbounds for j in R.pivot_cols
        R.pivot_pos[j] = 0
    end
    @inbounds for j in R.touched_col_rows
        empty!(R.col_rows[j])
    end
    empty!(R.pivot_cols)
    empty!(R.pivot_rows)
    empty!(R.touched_col_rows)
    empty!(R.scratch_cols)
    empty!(R.scratch_coeffs)
    empty!(R.scratch_rows)
    empty!(R.tmp_idx)
    empty!(R.tmp_val)
    return R
end

@inline function _incidence_rows!(R::_SparseRREF, j::Int)
    rows = R.col_rows[j]
    if isempty(rows)
        push!(R.touched_col_rows, j)
    end
    return rows
end

@inline function _incidence_add_nonpivot_cols!(R::_SparseRREF, rowpos::Int, row::SparseRow)
    @inbounds for t in 2:length(row.idx)
        j = row.idx[t]
        rows = _incidence_rows!(R, j)
        push!(rows, rowpos)
    end
    return R
end

@inline function _collect_incidence_rows!(R::_SparseRREF, p::Int)
    empty!(R.scratch_rows)
    for pos in R.col_rows[p]
        push!(R.scratch_rows, pos)
    end
    return R.scratch_rows
end

function _row_axpy_incidence!(
    R::_SparseRREF{K},
    rowpos::Int,
    row::SparseRow{K},
    a::K,
    other::SparseRow{K},
    tmp_idx::Vector{Int},
    tmp_val::Vector{K},
) where {K}
    iszero(a) && return tmp_idx, tmp_val
    isempty(other) && return tmp_idx, tmp_val
    isempty(row) && return _row_axpy!(row, a, other, tmp_idx, tmp_val)

    empty!(tmp_idx)
    empty!(tmp_val)
    maxlen = length(row.idx) + length(other.idx)
    resize!(tmp_idx, maxlen)
    resize!(tmp_val, maxlen)

    oldpivot = row.idx[1]
    i = 1
    j = 1
    m = length(row.idx)
    n = length(other.idx)
    w = 0

    if a == one(K)
        @inbounds while i <= m || j <= n
            if j > n || (i <= m && row.idx[i] < other.idx[j])
                w += 1
                tmp_idx[w] = row.idx[i]
                tmp_val[w] = row.val[i]
                i += 1
            elseif i > m || other.idx[j] < row.idx[i]
                col = other.idx[j]
                v = other.val[j]
                if !iszero(v)
                    w += 1
                    tmp_idx[w] = col
                    tmp_val[w] = v
                    if col != oldpivot
                        push!(_incidence_rows!(R, col), rowpos)
                    end
                end
                j += 1
            else
                col = row.idx[i]
                v = row.val[i] + other.val[j]
                if !iszero(v)
                    w += 1
                    tmp_idx[w] = col
                    tmp_val[w] = v
                elseif col != oldpivot
                    delete!(R.col_rows[col], rowpos)
                end
                i += 1
                j += 1
            end
        end
    elseif a == -one(K)
        @inbounds while i <= m || j <= n
            if j > n || (i <= m && row.idx[i] < other.idx[j])
                w += 1
                tmp_idx[w] = row.idx[i]
                tmp_val[w] = row.val[i]
                i += 1
            elseif i > m || other.idx[j] < row.idx[i]
                col = other.idx[j]
                v = -other.val[j]
                if !iszero(v)
                    w += 1
                    tmp_idx[w] = col
                    tmp_val[w] = v
                    if col != oldpivot
                        push!(_incidence_rows!(R, col), rowpos)
                    end
                end
                j += 1
            else
                col = row.idx[i]
                v = row.val[i] - other.val[j]
                if !iszero(v)
                    w += 1
                    tmp_idx[w] = col
                    tmp_val[w] = v
                elseif col != oldpivot
                    delete!(R.col_rows[col], rowpos)
                end
                i += 1
                j += 1
            end
        end
    else
        @inbounds while i <= m || j <= n
            if j > n || (i <= m && row.idx[i] < other.idx[j])
                w += 1
                tmp_idx[w] = row.idx[i]
                tmp_val[w] = row.val[i]
                i += 1
            elseif i > m || other.idx[j] < row.idx[i]
                col = other.idx[j]
                v = a * other.val[j]
                if !iszero(v)
                    w += 1
                    tmp_idx[w] = col
                    tmp_val[w] = v
                    if col != oldpivot
                        push!(_incidence_rows!(R, col), rowpos)
                    end
                end
                j += 1
            else
                col = row.idx[i]
                v = row.val[i] + a * other.val[j]
                if !iszero(v)
                    w += 1
                    tmp_idx[w] = col
                    tmp_val[w] = v
                elseif col != oldpivot
                    delete!(R.col_rows[col], rowpos)
                end
                i += 1
                j += 1
            end
        end
    end

    resize!(tmp_idx, w)
    resize!(tmp_val, w)
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

function _sparse_rref_push_homogeneous!(R::_SparseRREF{K}, row::SparseRow{K})::Bool where {K}
    isempty(row) && return false

    while true
        empty!(R.scratch_cols)
        empty!(R.scratch_coeffs)

        @inbounds for t in eachindex(row.idx)
            j = row.idx[t]
            pos = R.pivot_pos[j]
            if pos != 0
                push!(R.scratch_cols, j)
                push!(R.scratch_coeffs, row.val[t])
            end
        end

        isempty(R.scratch_cols) && break

        @inbounds for k in eachindex(R.scratch_cols)
            j = R.scratch_cols[k]
            c = R.scratch_coeffs[k]
            if !iszero(c)
                prow = R.pivot_rows[R.pivot_pos[j]]
                R.tmp_idx, R.tmp_val = _row_axpy!(row, -c, prow, R.tmp_idx, R.tmp_val)
            end
        end

        isempty(row) && return false
    end

    p = row.idx[1]
    @assert R.pivot_pos[p] == 0
    a = row.val[1]
    if a != one(K)
        inv_a = inv(a)
        row.val[1] = one(K)
        if inv_a == -one(K)
            @inbounds for t in 2:length(row.val)
                row.val[t] = -row.val[t]
            end
        else
            @inbounds for t in 2:length(row.val)
                row.val[t] *= inv_a
            end
        end
    end

    incident = _collect_incidence_rows!(R, p)
    for pos in incident
        prow = R.pivot_rows[pos]
        c = _row_coeff(prow, p)
        if !iszero(c)
            R.tmp_idx, R.tmp_val = _row_axpy_incidence!(R, pos, prow, -c, row, R.tmp_idx, R.tmp_val)
        end
    end

    push!(R.pivot_cols, p)
    push!(R.pivot_rows, copy(row))
    R.pivot_pos[p] = length(R.pivot_rows)
    _incidence_add_nonpivot_cols!(R, length(R.pivot_rows), row)
    return true
end

mutable struct _SparseRREFAugmented{K}
    rref::_SparseRREF{K}
    nrhs::Int
    pivot_rhs::Vector{Vector{K}}
end

function _SparseRREFAugmented{K}(nvars::Int, nrhs::Int) where {K}
    return _SparseRREFAugmented{K}(_SparseRREF{K}(nvars), nrhs, Vector{Vector{K}}())
end

function _sparse_rref_push_augmented!(
    R::_SparseRREFAugmented{K},
    row::SparseRow{K},
    rhs::Vector{K},
)::Symbol where {K}
    RR = R.rref
    @assert length(rhs) == R.nrhs

    isempty(row) && return all(iszero, rhs) ? :dependent : :inconsistent

    while true
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

        isempty(RR.scratch_cols) && break

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
    end

    p = row.idx[1]
    @assert RR.pivot_pos[p] == 0
    a = row.val[1]
    inv_a = inv(a)
    row.val[1] = one(K)
    @inbounds for t in 2:length(row.val)
        row.val[t] *= inv_a
    end
    @inbounds for t in 1:R.nrhs
        rhs[t] *= inv_a
    end

    incident = _collect_incidence_rows!(RR, p)
    for pos in incident
        prow = RR.pivot_rows[pos]
        c = _row_coeff(prow, p)
        if !iszero(c)
            RR.tmp_idx, RR.tmp_val = _row_axpy_incidence!(RR, pos, prow, -c, row, RR.tmp_idx, RR.tmp_val)
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
    _incidence_add_nonpivot_cols!(RR, length(RR.pivot_rows), row)
    return :pivot
end

function _nullspace_from_pivots(R::_SparseRREF{K}, nvars::Int) where {K}
    @assert nvars == R.nvars

    free_cols = Int[]
    sizehint!(free_cols, nvars - _rref_rank(R))
    for j in 1:nvars
        if R.pivot_pos[j] == 0
            push!(free_cols, j)
        end
    end

    nfree = length(free_cols)
    Z = zeros(K, nvars, nfree)
    if nfree == 0
        return Z
    end

    free_pos = zeros(Int, nvars)
    for (k, j) in enumerate(free_cols)
        free_pos[j] = k
        Z[j, k] = one(K)
    end

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
