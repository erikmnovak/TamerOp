module AbelianCategories

"""
    AbelianCategories

This module collects basic abelian-category constructions for the category of
finite-poset modules (PModule / PMorphism).

Design intent
- Keep these constructions independent of the indicator-resolution algorithms.
- Provide a single home for kernels/cokernels/images, pushouts/pullbacks,
  short exact sequences, and snake lemma output.
- Keep code ASCII-only.

Canonical public story
- Object only: `kernel`, `image`, `cokernel`, `coimage`.
- Object plus universal map: `kernel_with_inclusion`, `image_with_inclusion`,
  `cokernel_with_projection`, `coimage_with_projection`.
- Submodule bridge: `kernel_submodule`, `image_submodule`, then
  `quotient(S)` or `quotient_with_projection(S)`.
- For exact sequences, prefer constructing with `short_exact_sequence(...)`;
  `ShortExactSequence` is the concrete container type.

Cache usage
- Normal use: leave the documented default `cache=:auto`.
- If you will make many low-level calls on the same poset, call `build_cache!(Q)` once
  and then pass a reused `CoverCache` as `cache=cc`.
- The explicit `cache=cc` form is mainly for advanced/internal loops; most users
  should stay with `cache=:auto`.

Category-theoretic wrappers
- `limit` / `colimit` and the small diagram structs are kept as advanced
  category-theory surfaces.
- For everyday use, prefer the more concrete entrypoints such as `equalizer`,
  `coequalizer`, `pushout`, and `pullback`.

Most routines rely on FieldLinAlg and are field-generic. Exactness checks are
exact for exact fields and tolerance-based for RealField.

Downstream modules (DerivedFunctors, ModuleComplexes, ChangeOfPosets, ...)
should depend on this module rather than IndicatorResolutions when they only
need general categorical operations.
"""

using LinearAlgebra
using SparseArrays
import Base.Threads

import ..FiniteFringe
using ..FiniteFringe: cover_edges, nvertices, _succs, _preds,
                      _pred_slots_of_succ, _succ_slots_of_pred

using ..CoreModules: AbstractCoeffField, QQField, PrimeField, RealField, eye
using ..FieldLinAlg
import ..Modules

using ..Modules: CoverCache, _get_cover_cache,
                 CoverEdgeMapStore,
                 PosetCache,
                 PModule, PMorphism, id_morphism,
                 direct_sum, direct_sum_with_maps

# ------------------------------- helpers -------------------------------------

"""
    _is_zero_matrix(field, A) -> Bool

Internal field-aware zero test for a matrix.

Returns `true` when `A` is treated as zero over the given coefficient field:
- exact entrywise zero for exact fields,
- tolerance-based zero for `RealField`.

This is used throughout categorical parity/exactness checks so that the
`RealField` paths respect the field's configured tolerances.
"""

struct _CoverGraphLists
    preds::Vector{Vector{Int}}
    succs::Vector{Vector{Int}}
    pred_slot_of_succ::Vector{Vector{Int}}
end

@inline function _cover_graph_lists(cc::CoverCache)
    n = nvertices(cc.Q)
    preds = [collect(_preds(cc, v)) for v in 1:n]
    succs = [collect(_succs(cc, u)) for u in 1:n]
    pred_slot_of_succ = [collect(_pred_slots_of_succ(cc, u)) for u in 1:n]
    return _CoverGraphLists(preds, succs, pred_slot_of_succ)
end

function _is_zero_matrix(field::AbstractCoeffField, A::AbstractMatrix)
    if field isa RealField
        isempty(A) && return true
        maxabs = maximum(abs, A)
        tol = field.atol + field.rtol * maxabs
        return maxabs <= tol
    end
    return all(iszero, A)
end

@inline function _abelian_validation_error(fn::AbstractString, issues::Vector{String})
    error("$(fn): validation failed:\n - " * join(issues, "\n - "))
end

"""
    ValidationSummary

Display wrapper for validation reports returned by helpers such as
`check_submodule`, `check_short_exact_sequence`, and `check_snake_lemma`.

The canonical validation helpers still return structured `NamedTuple`s for easy
programmatic inspection. Wrap one of those reports with `validation_summary(...)`
when you want a compact REPL/notebook presentation of the same information.
"""
struct ValidationSummary{R}
    report::R
end

"""
    validation_summary(report) -> ValidationSummary

Wrap a validation report in a small display-oriented object.

Use this when you want the report returned by `check_*` helpers to print as a
compact mathematical summary with the issue list laid out line by line.

Best practices
- keep the original `NamedTuple` report for programmatic branching,
- call `validation_summary(report)` when you want a readable notebook or REPL
  display.
"""
@inline validation_summary(report::NamedTuple) = ValidationSummary(report)

@inline function _zero_composition(field::AbstractCoeffField,
                                   g::PMorphism,
                                   f::PMorphism)::Bool
    n = min(length(g.comps), length(f.comps))
    @inbounds for v in 1:n
        _is_zero_matrix(field, g.comps[v] * f.comps[v]) || return false
    end
    return true
end

const COKERNEL_BATCH_MIN_FANOUT = 3
const COKERNEL_BATCH_MIN_TOTAL_RHS_COLS = 128
const _FAST_SOLVE_NO_CHECK = Ref{Bool}(true)

"""
    _module_prefers_sparse_store(M) -> Bool

Internal storage heuristic for derived modules built from `M`.

Returns `true` when the first nonempty cover-edge map stored in `M` is sparse,
in which case downstream helper routines preserve sparse storage when possible.
"""
@inline function _module_prefers_sparse_store(M::PModule)
    @inbounds for maps_u in M.edge_maps.maps_to_succ
        isempty(maps_u) && continue
        return maps_u[1] isa SparseMatrixCSC
    end
    return false
end

@inline _store_out_type(::Type{K}, prefer_sparse::Bool) where {K} =
    prefer_sparse ? SparseMatrixCSC{K,Int} : Matrix{K}

@inline _zero_store_map(::Type{Matrix{K}}, ::Type{K}, m::Int, n::Int) where {K} =
    zeros(K, m, n)
@inline _zero_store_map(::Type{SparseMatrixCSC{K,Int}}, ::Type{K}, m::Int, n::Int) where {K} =
    spzeros(K, m, n)

@inline _to_store_map(::Type{Matrix{K}}, A::AbstractMatrix{K}) where {K} =
    (A isa Matrix{K} ? A : Matrix{K}(A))
@inline _to_store_map(::Type{SparseMatrixCSC{K,Int}}, A::AbstractMatrix{K}) where {K} =
    (A isa SparseMatrixCSC{K,Int} ? A : sparse(A))

"""
    _resolve_cover_cache(Q, cache) -> CoverCache

Normalize an internal cache argument to a concrete `CoverCache`.

Accepted values are:
- `cache=:auto`, which calls `_get_cover_cache(Q)`,
- `cache=cc::CoverCache`, which is returned unchanged.

Use this only from internal kernels. Public-facing wrappers should prefer
`_resolve_public_cover_cache`, which preserves the public `cache=:auto` policy.
"""
@inline _resolve_cover_cache(Q, cc::CoverCache) = cc
"""
    _resolve_cover_cache(Q, cache::Symbol) -> CoverCache

Symbol-dispatch branch of `_resolve_cover_cache`.

Currently the only accepted symbol is `:auto`, which builds or reuses the
standard cover cache for `Q`. Internal callers should pass a concrete
`CoverCache` whenever one is already available.
"""
@inline function _resolve_cover_cache(Q, cache::Symbol)
    cache === :auto || error("cache must be :auto or a CoverCache")
    return _get_cover_cache(Q)
end

const _PUBLIC_LAST_Q = Any[nothing for _ in 1:max(1, Threads.nthreads())]
const _PUBLIC_LAST_CC = Any[nothing for _ in 1:max(1, Threads.nthreads())]

@inline _cheap_wrapper_prefers_fast_auto_cache(field::AbstractCoeffField) =
    (field isa PrimeField) || (field isa RealField)

"""
    _peek_cover_cache(Q) -> Union{CoverCache,Nothing}

Return the live `CoverCache` already attached to the poset `Q`, if one is
present, or `nothing` otherwise.

This does not build a cache. It is used by the public `cache=:auto` fast path
to reuse an existing cache without forcing fresh construction.
"""
@inline function _peek_cover_cache(Q)
    if hasproperty(Q, :cache)
        pc = getproperty(Q, :cache)
        if pc isa PosetCache
            return pc.cover
        end
    end
    return nothing
end

"""
    _record_public_cover_cache!(Q, cc) -> CoverCache

Record a recently used `(Q, cc)` pair in the small thread-local AbelianCategories
public cache memo and return `cc`.

This is an internal latency optimization for repeated `cache=:auto` calls on the
same poset. It does not change ownership or lifetime of the cache.
"""
@inline function _record_public_cover_cache!(Q, cc::CoverCache)
    tid = Threads.threadid()
    @inbounds begin
        _PUBLIC_LAST_Q[tid] = Q
        _PUBLIC_LAST_CC[tid] = cc
    end
    return cc
end

"""
    _resolve_public_cover_cache(Q, field, cache) -> CoverCache

Resolve a public AbelianCategories cache argument to a concrete `CoverCache`.

Accepted values are:
- `cache=:auto`, which reuses a live cache when possible and otherwise builds one,
- `cache=cc::CoverCache`, which is returned unchanged.

Best practice:
- user-facing wrappers should call this helper,
- low-level cached kernels should accept `cc::CoverCache` directly.
"""
@inline _resolve_public_cover_cache(Q, field::AbstractCoeffField, cc::CoverCache) = cc

"""
    _resolve_public_cover_cache(Q, field, cache::Symbol) -> CoverCache

Public `cache=:auto` resolution branch for AbelianCategories entrypoints.

For cheap wrapper calls over `PrimeField` and `RealField`, this first checks for
an already-live cache before forcing construction. For exact-arithmetic hot
paths, it falls back to the standard `_get_cover_cache(Q)` behavior.
"""
@inline function _resolve_public_cover_cache(Q, field::AbstractCoeffField, cache::Symbol)
    cache === :auto || error("cache must be :auto or a CoverCache")
    if _cheap_wrapper_prefers_fast_auto_cache(field)
        tid = Threads.threadid()
        lastQ = @inbounds _PUBLIC_LAST_Q[tid]
        lastcc = @inbounds _PUBLIC_LAST_CC[tid]
        if lastQ === Q && lastcc isa CoverCache
            live = _peek_cover_cache(Q)
            if live === lastcc
                return lastcc
            end
        end
        live = _peek_cover_cache(Q)
        if live isa CoverCache
            return _record_public_cover_cache!(Q, live)
        end
    end
    return _record_public_cover_cache!(Q, _get_cover_cache(Q))
end

"""
    _selector_rows_from_basis(field, B) -> Union{Vector{Int},Nothing}

Detect whether the columns of `B` form a coordinate-selector basis.

Returns a vector of selected row indices when each column of `B` is a distinct
standard basis vector (up to the exact/tolerant field semantics), and `nothing`
otherwise.

This is an internal fast-path detector used to replace generic solves with
direct row/column extraction in kernel/image/cokernel transport.
"""
@inline function _selector_rows_from_basis(field::AbstractCoeffField, B::SparseMatrixCSC)
    k = size(B, 2)
    k == 0 && return Int[]
    rows = Vector{Int}(undef, k)
    seen = falses(size(B, 1))
    @inbounds for j in 1:k
        p0 = B.colptr[j]
        p1 = B.colptr[j + 1] - 1
        p0 > p1 && return nothing
        p0 == p1 || return nothing
        row = B.rowval[p0]
        seen[row] && return nothing
        _is_one_coeff(field, B.nzval[p0]) || return nothing
        rows[j] = row
        seen[row] = true
    end
    return rows
end

"""
    _selector_rows_from_basis(field, B::AbstractMatrix) -> Union{Vector{Int},Nothing}

Dense-matrix fallback for `_selector_rows_from_basis`.

The return contract is the same as for the sparse overload: selected row
indices when `B` is a coordinate-selector basis, and `nothing` otherwise.
"""
@inline function _selector_rows_from_basis(field::AbstractCoeffField, B::AbstractMatrix)
    k = size(B, 2)
    k == 0 && return Int[]
    rows = Vector{Int}(undef, k)
    seen = falses(size(B, 1))
    @inbounds for j in 1:k
        row = 0
        for i in 1:size(B, 1)
            Bij = B[i, j]
            _is_zero_coeff(field, Bij) && continue
            if row == 0 && _is_one_coeff(field, Bij)
                row = i
            else
                return nothing
            end
        end
        row == 0 && return nothing
        seen[row] && return nothing
        rows[j] = row
        seen[row] = true
    end
    return rows
end

"""
    _blockdiag_extract(K, Buv, Cuv, rows_v, rows_u, bu, bv) -> Matrix{K}

Extract the submatrix determined by selector row/column sets from the implicit
block-diagonal matrix `diag(Buv, Cuv)`.

Arguments `bu` and `bv` give the sizes of the `B`-summands at the source and
target vertices, so the helper can decide whether a requested row/column lies in
the `B` or `C` block.

This avoids materializing the full block-diagonal matrix in pushout code paths.
"""
@inline function _blockdiag_extract(
    ::Type{K},
    Buv::AbstractMatrix{K},
    Cuv::AbstractMatrix{K},
    rows_v::AbstractVector{Int},
    rows_u::AbstractVector{Int},
    bu::Int,
    bv::Int,
) where {K}
    X = Matrix{K}(undef, length(rows_v), length(rows_u))
    @inbounds for i in eachindex(rows_v)
        rv = rows_v[i]
        if rv <= bv
            for j in eachindex(rows_u)
                ru = rows_u[j]
                X[i, j] = ru <= bu ? Buv[rv, ru] : zero(K)
            end
        else
            rcv = rv - bv
            for j in eachindex(rows_u)
                ru = rows_u[j]
                X[i, j] = ru > bu ? Cuv[rcv, ru - bu] : zero(K)
            end
        end
    end
    return X
end

"""
    _blockdiag_selected_cols(K, Buv, Cuv, rows_u, bu) -> Matrix{K}

Build the selected-column submatrix of the implicit block-diagonal matrix
`diag(Buv, Cuv)` determined by the source selector rows `rows_u`.

This is the column-oriented companion to `_blockdiag_extract` and is used by
pushout transport code when only selected columns are needed.
"""
@inline function _blockdiag_selected_cols(
    ::Type{K},
    Buv::AbstractMatrix{K},
    Cuv::AbstractMatrix{K},
    rows_u::AbstractVector{Int},
    bu::Int,
) where {K}
    m = size(Buv, 1) + size(Cuv, 1)
    X = zeros(K, m, length(rows_u))
    bv = size(Buv, 1)
    @inbounds for j in eachindex(rows_u)
        ru = rows_u[j]
        if ru <= bu
            view(X, 1:bv, j) .= view(Buv, :, ru)
        else
            view(X, bv + 1:m, j) .= view(Cuv, :, ru - bu)
        end
    end
    return X
end

"""
    _is_zero_coeff(field, x) -> Bool

Internal field-aware scalar zero test.

Returns exact zero for exact fields and tolerance-based zero for `RealField`.
"""
@inline function _is_zero_coeff(field::AbstractCoeffField, x)
    if field isa RealField
        return abs(x) <= field.atol + field.rtol * abs(x)
    end
    return iszero(x)
end

"""
    _is_one_coeff(field, x) -> Bool

Internal field-aware scalar one test.

Returns exact equality to `one(x)` for exact fields and a tolerance-based test
for `RealField`.
"""
@inline function _is_one_coeff(field::AbstractCoeffField, x)
    if field isa RealField
        return abs(x - one(x)) <= field.atol + field.rtol * max(abs(x), one(abs(x)))
    end
    return x == one(x)
end

"""
    _is_partial_permutation(field, A) -> Bool

Return `true` when each column of `A` is either zero or a single `1`, with no
other nonzero entries.

This internal structural detector enables specialized exact kernel/cokernel
construction without generic nullspace computation.
"""
@inline function _is_partial_permutation(field::AbstractCoeffField, A::SparseMatrixCSC)
    field isa RealField && return false
    m, n = size(A)
    (m == 0 || n == 0) && return true
    @inbounds for j in 1:n
        s = A.colptr[j]
        e = A.colptr[j + 1] - 1
        if s > e
            continue
        elseif e != s
            return false
        end
        _is_one_coeff(field, A.nzval[s]) || return false
    end
    return true
end

"""
    _is_partial_permutation(field, A::AbstractMatrix) -> Bool

Dense-matrix fallback for `_is_partial_permutation`.

Returns `true` exactly when each column of `A` is either zero or a single unit
entry and no column contains more than one nonzero.
"""
@inline function _is_partial_permutation(field::AbstractCoeffField, A::AbstractMatrix)
    field isa RealField && return false
    m, n = size(A)
    @inbounds for j in 1:n
        nz = 0
        for i in 1:m
            v = A[i, j]
            _is_zero_coeff(field, v) && continue
            nz += 1
            nz == 1 || return false
            _is_one_coeff(field, v) || return false
        end
    end
    return true
end

"""
    _kernel_basis_partial_permutation(K, A) -> Matrix{K}

Construct a kernel basis for a matrix `A` known to satisfy the partial-
permutation pattern detected by `_is_partial_permutation`.

Returns a matrix whose columns form a basis of `ker(A)`. This avoids calling the
generic nullspace routine on a structure that admits a direct combinatorial
description.
"""
@inline function _kernel_basis_partial_permutation(::Type{K}, A::SparseMatrixCSC) where {K}
    n = size(A, 2)
    row_to_cols = Dict{Int,Vector{Int}}()
    zero_cols = Int[]
    sizehint!(zero_cols, n)
    @inbounds for j in 1:n
        s = A.colptr[j]
        e = A.colptr[j + 1] - 1
        if s > e
            push!(zero_cols, j)
        else
            r = A.rowval[s]
            cols = get!(row_to_cols, r, Int[])
            push!(cols, j)
        end
    end
    d = length(zero_cols)
    for cols in values(row_to_cols)
        d += max(0, length(cols) - 1)
    end
    B = zeros(K, n, d)
    c = 1
    @inbounds for j in zero_cols
        B[j, c] = one(K)
        c += 1
    end
    neg_one = -one(K)
    for cols in values(row_to_cols)
        length(cols) <= 1 && continue
        anchor = cols[1]
        for t in 2:length(cols)
            j = cols[t]
            B[j, c] = one(K)
            B[anchor, c] = neg_one
            c += 1
        end
    end
    return B
end

"""
    _kernel_basis_partial_permutation(K, A::AbstractMatrix) -> Matrix{K}

Dense-matrix fallback for `_kernel_basis_partial_permutation`.

Assumes the caller already established the partial-permutation structure and
returns the same coordinate-selector kernel basis as the sparse overload.
"""
@inline function _kernel_basis_partial_permutation(::Type{K}, A::AbstractMatrix) where {K}
    m, n = size(A)
    row_to_cols = Dict{Int,Vector{Int}}()
    zero_cols = Int[]
    sizehint!(zero_cols, n)
    @inbounds for j in 1:n
        col_zero = true
        row = 0
        for i in 1:m
            if !iszero(A[i, j])
                col_zero = false
                row = i
                break
            end
        end
        if col_zero
            push!(zero_cols, j)
        else
            cols = get!(row_to_cols, row, Int[])
            push!(cols, j)
        end
    end
    d = length(zero_cols)
    for cols in values(row_to_cols)
        d += max(0, length(cols) - 1)
    end
    B = zeros(K, n, d)
    c = 1
    @inbounds for j in zero_cols
        B[j, c] = one(K)
        c += 1
    end
    neg_one = -one(K)
    for cols in values(row_to_cols)
        length(cols) <= 1 && continue
        anchor = cols[1]
        for t in 2:length(cols)
            j = cols[t]
            B[j, c] = one(K)
            B[anchor, c] = neg_one
            c += 1
        end
    end
    return B
end

"""
    _cokernel_q_partial_permutation(K, A) -> Matrix{K}

Construct a quotient projection matrix for a partial-permutation matrix `A`.

Returns a matrix `q` whose rows form the canonical coordinate projection onto the
cokernel of `A`, again avoiding generic nullspace work on a structured input.
"""
@inline function _cokernel_q_partial_permutation(::Type{K}, A::SparseMatrixCSC) where {K}
    m, _ = size(A)
    row_hit = falses(m)
    @inbounds for j in 1:size(A, 2)
        s = A.colptr[j]
        e = A.colptr[j + 1] - 1
        if s <= e
            row_hit[A.rowval[s]] = true
        end
    end
    keep = Int[]
    sizehint!(keep, m)
    @inbounds for i in 1:m
        row_hit[i] || push!(keep, i)
    end
    c = length(keep)
    q = zeros(K, c, m)
    @inbounds for (r, i) in enumerate(keep)
        q[r, i] = one(K)
    end
    return q
end

"""
    _cokernel_q_partial_permutation(K, A::AbstractMatrix) -> Matrix{K}

Dense-matrix fallback for `_cokernel_q_partial_permutation`.

Assumes the partial-permutation structure has already been checked and returns
the quotient projection onto codomain coordinates not hit by `A`.
"""
@inline function _cokernel_q_partial_permutation(::Type{K}, A::AbstractMatrix) where {K}
    m, n = size(A)
    row_hit = falses(m)
    @inbounds for j in 1:n
        for i in 1:m
            if !iszero(A[i, j])
                row_hit[i] = true
                break
            end
        end
    end
    keep = Int[]
    sizehint!(keep, m)
    @inbounds for i in 1:m
        row_hit[i] || push!(keep, i)
    end
    c = length(keep)
    q = zeros(K, c, m)
    @inbounds for (r, i) in enumerate(keep)
        q[r, i] = one(K)
    end
    return q
end

"""
    _KernelIncrementalEntry{K}

Internal cache record for one vertex in an incremental kernel computation.

Fields:
- `map`: the last stalk matrix seen at that vertex,
- `basis`: the previously computed kernel basis for that stalk.
"""
struct _KernelIncrementalEntry{K}
    map::Matrix{K}
    basis::Matrix{K}
end

"""
    _CokernelIncrementalEntry{K}

Internal cache record for one vertex in an incremental cokernel computation.

Fields:
- `map`: the last stalk matrix seen at that vertex,
- `q`: the previously computed quotient projection for that stalk.
"""
struct _CokernelIncrementalEntry{K}
    map::Matrix{K}
    q::Matrix{K}
end

mutable struct _CokernelTransportCache{K}
    active_mask::BitVector
    frontier_mask::BitVector
    frontier_vertices::Vector{Int}
    qsig_rows::Vector{Int}
    qsig_cols::Vector{Int}
    qtrans::Vector{Union{Nothing,Matrix{K}}}
    selector_rows::Vector{Vector{Int}}
    selector_is_std::BitVector
    changed_vertices::Vector{Int}
end

@inline function _new_cokernel_transport_cache(::Type{K}, n::Int) where {K}
    return _CokernelTransportCache{K}(
        falses(n),
        falses(n),
        Int[],
        zeros(Int, n),
        zeros(Int, n),
        Vector{Union{Nothing,Matrix{K}}}(fill(nothing, n)),
        [Int[] for _ in 1:n],
        falses(n),
        Int[],
    )
end

const _VertexIncrementalCacheEntry{K} = Union{
    Nothing,
    _KernelIncrementalEntry{K},
    _CokernelIncrementalEntry{K},
}

"""
    _ensure_vertex_cache!(cache, n) -> cache

Resize and initialize a per-vertex incremental cache container to length `n`.

Any new entries are filled with `nothing`. Existing entries are preserved when
the length already matches.
"""
@inline function _ensure_vertex_cache!(cache::AbstractVector, n::Int)
    if length(cache) != n
        resize!(cache, n)
        fill!(cache, nothing)
    end
    return cache
end

"""
    _is_appended_zero_matrix(field, prev, curr) -> Bool

Return `true` when `curr` extends `prev` only by appending zero rows and/or zero
columns.

This is used by the incremental kernel/cokernel builders to cheaply extend a
previous basis/projection instead of recomputing it from scratch.
"""
@inline function _is_appended_zero_matrix(
    field::AbstractCoeffField,
    prev::AbstractMatrix,
    curr::AbstractMatrix,
)
    m1, n1 = size(prev)
    m2, n2 = size(curr)
    m2 >= m1 || return false
    n2 >= n1 || return false
    @inbounds for i in 1:m1
        for j in 1:n1
            prev[i, j] == curr[i, j] || return false
        end
    end
    @inbounds for i in (m1 + 1):m2
        for j in 1:n2
            _is_zero_coeff(field, curr[i, j]) || return false
        end
    end
    @inbounds for i in 1:m1
        for j in (n1 + 1):n2
            _is_zero_coeff(field, curr[i, j]) || return false
        end
    end
    return true
end

"""
    _extend_kernel_basis_appended(K, prev_basis, prev_n, curr_n) -> Matrix{K}

Extend a previously computed kernel basis when the underlying matrix has only
gained appended zero columns.

The returned basis contains the old basis and one new standard basis vector for
each appended zero column.
"""
@inline function _extend_kernel_basis_appended(
    ::Type{K},
    prev_basis::Matrix{K},
    prev_n::Int,
    curr_n::Int,
) where {K}
    curr_n >= prev_n || error("_extend_kernel_basis_appended: curr_n < prev_n")
    dprev = size(prev_basis, 2)
    extra = curr_n - prev_n
    B = zeros(K, curr_n, dprev + extra)
    if dprev > 0
        @views B[1:prev_n, 1:dprev] .= prev_basis
    end
    @inbounds for t in 1:extra
        B[prev_n + t, dprev + t] = one(K)
    end
    return B
end

"""
    _extend_cokernel_q_appended(K, prev_q, prev_m, curr_m) -> Matrix{K}

Extend a previously computed cokernel projection when the underlying matrix has
only gained appended zero rows.

The returned projection keeps the old quotient coordinates and adds one new
coordinate row for each appended zero row.
"""
@inline function _extend_cokernel_q_appended(
    ::Type{K},
    prev_q::Matrix{K},
    prev_m::Int,
    curr_m::Int,
) where {K}
    curr_m >= prev_m || error("_extend_cokernel_q_appended: curr_m < prev_m")
    cprev = size(prev_q, 1)
    extra = curr_m - prev_m
    q = zeros(K, cprev + extra, curr_m)
    if cprev > 0
        @views q[1:cprev, 1:prev_m] .= prev_q
    end
    @inbounds for t in 1:extra
        q[cprev + t, prev_m + t] = one(K)
    end
    return q
end

# --------------------------- kernel and upset presentation --------------------------

"""
    _kernel_with_inclusion_cached(f, cc; incremental_cache=nothing) -> (Kmod, iota)

Internal cached kernel constructor.

Returns:
- `Kmod`: the kernel module of `f`,
- `iota`: the inclusion `Kmod -> dom(f)`.

If `incremental_cache` is supplied, it may reuse vertexwise kernel bases across
closely related calls. Most external callers should use `kernel_with_inclusion`
instead of this low-level cached entrypoint.
"""
function _kernel_with_inclusion_cached(
    f::PMorphism{K},
    cc::CoverCache;
    incremental_cache::Union{Nothing,AbstractVector}=nothing,
) where {K}
    M = f.dom
    n = nvertices(M.Q)

    basisK = Vector{Matrix{K}}(undef, n)
    K_dims = zeros(Int, n)
    ic = incremental_cache
    ic === nothing || _ensure_vertex_cache!(ic, n)
    succs = [collect(_succs(cc, u)) for u in 1:n]
    pred_slot_of_succ = [collect(_pred_slots_of_succ(cc, u)) for u in 1:n]
    selector_rows = Vector{Vector{Int}}(undef, n)
    selector_is_std = falses(n)
    empty_selector = Int[]
    for i in 1:n
        fi = f.comps[i]
        B = nothing
        if ic !== nothing
            entry = ic[i]
            if entry isa _KernelIncrementalEntry{K}
                prev = entry.map
                prev_basis = entry.basis
                if prev === fi
                    B = prev_basis
                elseif size(prev, 1) <= size(fi, 1) &&
                       size(prev, 2) <= size(fi, 2) &&
                       _is_appended_zero_matrix(f.dom.field, prev, fi)
                    B = _extend_kernel_basis_appended(K, prev_basis, size(prev, 2), size(fi, 2))
                end
            end
        end
        if B === nothing
            B = if _is_partial_permutation(f.dom.field, fi)
                _kernel_basis_partial_permutation(K, fi)
            else
                FieldLinAlg.nullspace(f.dom.field, fi)
            end
        end
        if ic !== nothing
            ic[i] = fi isa Matrix{K} ? _KernelIncrementalEntry{K}(fi, B) :
                                       _KernelIncrementalEntry{K}(Matrix{K}(fi), B)
        end
        basisK[i] = B
        K_dims[i] = size(B, 2)
        rows = _selector_rows_from_basis(f.dom.field, B)
        if rows === nothing
            selector_rows[i] = empty_selector
        else
            selector_rows[i] = rows
            selector_is_std[i] = true
        end
    end

    # Build the kernel's structure maps directly in store-aligned form.
    preds = [collect(_preds(cc, v)) for v in 1:n]
    OutMatT = _store_out_type(K, _module_prefers_sparse_store(M))
    indegree = [length(preds[v]) for v in 1:n]
    field = f.dom.field
    is_qq = field isa QQField
    qq_basis_factors = Vector{Any}(undef, n)
    is_qq && fill!(qq_basis_factors, nothing)
    maps_from_pred = [Vector{OutMatT}(undef, length(preds[v])) for v in 1:n]
    maps_to_succ   = [Vector{OutMatT}(undef, length(succs[u])) for u in 1:n]

    @inbounds for u in 1:n
        su = succs[u]
        maps_u_M = M.edge_maps.maps_to_succ[u]   # aligned with su
        outu = maps_to_succ[u]
        is_selector_u = selector_is_std[u]
        rows_u = selector_rows[u]

        for j in eachindex(su)
            v = su[j]

            # If either stalk is 0, the induced map is the unique 0 map.
            if K_dims[u] == 0 || K_dims[v] == 0
                X = _zero_store_map(OutMatT, K, K_dims[v], K_dims[u])
                outu[j] = X
                ip = pred_slot_of_succ[u][j]
                maps_from_pred[v][ip] = X
                continue
            end

            # Induced map K(u) -> K(v): express M(u->v)*basisK[u] in basisK[v].
            T  = maps_u_M[j]
            is_selector_v = selector_is_std[v]
            rows_v = selector_rows[v]
            Xraw = if is_selector_u && is_selector_v
                T[rows_v, rows_u]
            else
                Im = is_selector_u ? T[:, rows_u] : T * basisK[u]
                if is_selector_v
                    Im[rows_v, :]
                elseif indegree[v] <= 1
                    # One-off edge solve: skip global solve-factor cache overhead.
                    FieldLinAlg.solve_fullcolumn(field, basisK[v], Im; check_rhs=false, cache=false)
                elseif is_qq
                    # High-indegree QQ vertices benefit from reusing a local factor.
                    fac = qq_basis_factors[v]
                    if fac === nothing
                        fac = FieldLinAlg._factor_fullcolumnQQ(basisK[v])
                        qq_basis_factors[v] = fac
                    end
                    FieldLinAlg.solve_fullcolumn(field, basisK[v], Im;
                                                 check_rhs=false, cache=false, factor=fac)
                else
                    FieldLinAlg.solve_fullcolumn(field, basisK[v], Im; check_rhs=false)
                end
            end
            X = _to_store_map(OutMatT, Xraw)

            outu[j] = X
            ip = pred_slot_of_succ[u][j]
            maps_from_pred[v][ip] = X
        end
    end

    storeK = CoverEdgeMapStore{K,OutMatT}(preds, succs, maps_from_pred, maps_to_succ, cc.nedges)
    Kmod = PModule{K}(M.Q, K_dims, storeK; field=M.field)

    iota = PMorphism{K}(Kmod, M, [basisK[i] for i in 1:n])
    return Kmod, iota
end

@inline _kernel_module_cached(f::PMorphism{K}, cc::CoverCache) where {K} =
    (_kernel_with_inclusion_cached(f, cc))[1]

"""
    kernel_with_inclusion(f; cache=:auto, incremental_cache=nothing) -> (K, iota)

Compute the kernel submodule of `f` together with its inclusion into `dom(f)`.

Inputs:
- `f : A -> B`, a morphism of `PModule`s on a common poset,
- `cache`, either `:auto` or a prebuilt `CoverCache`,
- `incremental_cache`, an optional internal reuse buffer for nearby repeated
  kernel calls.

Returns:
- `K`, the kernel module `ker(f)`,
- `iota : K -> A`, the universal inclusion.

Best practice:
- use `kernel(f)` when you only need the object,
- use `kernel_submodule(f)` when the next step is a quotient,
- keep `cache=:auto` unless you are running a low-level repeated-call loop on a
  fixed poset, in which case `build_cache!(Q)` once and reuse `cache=cc`.
"""
function kernel_with_inclusion(
    f::PMorphism{K};
    cache::Union{Symbol,CoverCache}=:auto,
    incremental_cache::Union{Nothing,AbstractVector}=nothing,
) where {K}
    cc = _resolve_cover_cache(f.dom.Q, cache)
    return _kernel_with_inclusion_cached(f, cc; incremental_cache=incremental_cache)
end


# ----------------------------
# Image with inclusion (dual to kernel_with_inclusion)
# ----------------------------

"""
    _image_with_inclusion_cached(f, cc) -> (Im, iota)

Internal cached image constructor.

Returns:
- `Im`: the image module of `f`,
- `iota`: the inclusion `Im -> cod(f)`.

This routine assumes a concrete `CoverCache` is already available. Use the
public `image_with_inclusion` wrapper unless you are already working inside a
cache-aware low-level loop.
"""
function _image_with_inclusion_cached(f::PMorphism{K}, cc::CoverCache) where {K}
    N = f.cod
    Q = N.Q
    n = nvertices(Q)

    bases = Vector{Matrix{K}}(undef, n)
    dims  = zeros(Int, n)

    preds  = N.edge_maps.preds
    succs  = N.edge_maps.succs
    succ_slot_of_pred = [_succ_slots_of_pred(cc, v) for v in 1:n]
    selector_rows = Vector{Vector{Int}}(undef, n)
    selector_is_std = falses(n)
    empty_selector = Int[]
    for i in 1:n
        B = FieldLinAlg.colspace(f.dom.field, f.comps[i])
        bases[i] = B
        dims[i]  = size(B, 2)
        rows = _selector_rows_from_basis(f.dom.field, B)
        if rows === nothing
            selector_rows[i] = empty_selector
        else
            selector_rows[i] = rows
            selector_is_std[i] = true
        end
    end

    # Build the image's structure maps directly in store-aligned form.
    storeN = N.edge_maps
    OutMatT = _store_out_type(K, _module_prefers_sparse_store(N))
    field = f.dom.field
    is_qq = field isa QQField
    qq_basis_factors = Vector{Any}(undef, n)
    is_qq && fill!(qq_basis_factors, nothing)

    maps_from_pred = [Vector{OutMatT}(undef, length(preds[v])) for v in 1:n]
    maps_to_succ   = [Vector{OutMatT}(undef, length(succs[u])) for u in 1:n]

    @inbounds for v in 1:n
        pv = preds[v]
        isempty(pv) && continue

        Bv = bases[v]
        dv = size(Bv, 2)
        if dv == 0
            for i in eachindex(pv)
                u = pv[i]
                su = succ_slot_of_pred[v][i]
                Auv = _zero_store_map(OutMatT, K, 0, dims[u])
                maps_from_pred[v][i] = Auv
                maps_to_succ[u][su] = Auv
            end
            continue
        end

        if selector_is_std[v]
            rows_v = selector_rows[v]
            for i in eachindex(pv)
                u = pv[i]
                du = dims[u]
                su = succ_slot_of_pred[v][i]
                Auv = if du == 0
                    _zero_store_map(OutMatT, K, dv, 0)
                elseif selector_is_std[u]
                    _to_store_map(OutMatT, storeN.maps_to_succ[u][su][rows_v, selector_rows[u]])
                else
                    _to_store_map(OutMatT, (storeN.maps_to_succ[u][su] * bases[u])[rows_v, :])
                end
                maps_from_pred[v][i] = Auv
                maps_to_succ[u][su] = Auv
            end
            continue
        end

        total_cols = 0
        for u in pv
            total_cols += dims[u]
        end
        if total_cols == 0
            for i in eachindex(pv)
                u = pv[i]
                su = succ_slot_of_pred[v][i]
                Auv = _zero_store_map(OutMatT, K, dv, 0)
                maps_from_pred[v][i] = Auv
                maps_to_succ[u][su] = Auv
            end
            continue
        end

        offsets = Vector{Int}(undef, length(pv) + 1)
        offsets[1] = 1
        for i in eachindex(pv)
            offsets[i + 1] = offsets[i] + dims[pv[i]]
        end

        rhs_all = Matrix{K}(undef, N.dims[v], total_cols)
        for i in eachindex(pv)
            u = pv[i]
            du = dims[u]
            du == 0 && continue
            su = succ_slot_of_pred[v][i]
            c0 = offsets[i]
            c1 = offsets[i + 1] - 1
            if selector_is_std[u]
                view(rhs_all, :, c0:c1) .= view(storeN.maps_to_succ[u][su], :, selector_rows[u])
            else
                mul!(view(rhs_all, :, c0:c1), storeN.maps_to_succ[u][su], bases[u])
            end
        end

        fac = nothing
        if is_qq && length(pv) > 1
            fac = qq_basis_factors[v]
            if fac === nothing
                fac = FieldLinAlg._factor_fullcolumnQQ(Bv)
                qq_basis_factors[v] = fac
            end
        end
        Xall = if fac === nothing
            FieldLinAlg.solve_fullcolumn(field, Bv, rhs_all;
                                         check_rhs=!_FAST_SOLVE_NO_CHECK[],
                                         cache=false)
        else
            FieldLinAlg.solve_fullcolumn(field, Bv, rhs_all;
                                         check_rhs=!_FAST_SOLVE_NO_CHECK[],
                                         cache=false, factor=fac)
        end

        for i in eachindex(pv)
            u = pv[i]
            du = dims[u]
            su = succ_slot_of_pred[v][i]
            Auv = if du == 0
                _zero_store_map(OutMatT, K, dv, 0)
            else
                c0 = offsets[i]
                c1 = offsets[i + 1] - 1
                _to_store_map(OutMatT, view(Xall, :, c0:c1))
            end
            maps_from_pred[v][i] = Auv
            maps_to_succ[u][su] = Auv
        end
    end

    storeIm = CoverEdgeMapStore{K,OutMatT}(preds, succs, maps_from_pred, maps_to_succ, storeN.nedges)
    Im = PModule{K}(Q, dims, storeIm; field=N.field)

    iota = PMorphism(Im, N, [bases[i] for i in 1:n])
    return Im, iota
end

@inline _image_module_cached(f::PMorphism{K}, cc::CoverCache) where {K} =
    (_image_with_inclusion_cached(f, cc))[1]

"""
    image_with_inclusion(f; cache=:auto) -> (Im, iota)

Compute the image of a morphism together with its universal inclusion.

Returns:
- `Im`: the image module, supported on the same poset as `cod(f)`,
- `iota`: the inclusion morphism `Im -> cod(f)`.

Best practice:
- use `image(f)` if you only need the object,
- use `image_submodule(f)` if your next step is `quotient(S)`,
- leave `cache=:auto` unless you are explicitly reusing a `CoverCache`.
"""
@inline function image_with_inclusion(f::PMorphism{K}; cache::Union{Symbol,CoverCache}=:auto) where {K}
    cc = _resolve_cover_cache(f.cod.Q, cache)
    return _image_with_inclusion_cached(f, cc)
end

"""
    _cokernel_module_cached(iota, cc; incremental_cache=nothing) -> (Cmod, q)

Internal cached cokernel constructor.

Returns:
- `Cmod`: the cokernel module of `iota`,
- `q`: the quotient morphism `cod(iota) -> Cmod`.

The quotient is built stalkwise and then transported along cover relations. When
`incremental_cache` is provided, the routine may reuse vertexwise quotient data
across nearby calls.
"""
function _cokernel_module_cached(
    iota::PMorphism{K},
    cc::CoverCache;
    incremental_cache::Union{Nothing,AbstractVector}=nothing,
    active_vertices::Union{Nothing,Vector{Int}}=nothing,
    active_mask::Union{Nothing,BitVector}=nothing,
    transport_cache::Union{Nothing,_CokernelTransportCache{K}}=nothing,
    graph_lists::Union{Nothing,_CoverGraphLists}=nothing,
) where {K}
    E = iota.cod; Q = E.Q; n = nvertices(Q)
    Cdims  = zeros(Int, n)
    qcomps = Vector{Matrix{K}}(undef, n)     # each is (dim C_i) x (dim E_i)
    field = E.field
    ic = incremental_cache
    ic === nothing || _ensure_vertex_cache!(ic, n)
    if active_mask === nothing && active_vertices !== nothing
        active_mask = falses(n)
        @inbounds for u in active_vertices
            active_mask[u] = true
        end
    end

    # Degreewise quotients. For a general map ii, q_i is a basis of the left
    # kernel of ii (as row vectors), i.e. transpose(nullspace(transpose(ii))).
    # This avoids a separate colspace pass.
    q_reused = falses(n)
    for i in 1:n
        ii = iota.comps[i]
        qi = nothing
        if active_mask !== nothing && !active_mask[i]
            qi = eye(field, size(ii, 1))
            q_reused[i] = true
        elseif ic !== nothing
            entry = ic[i]
            if entry isa _CokernelIncrementalEntry{K}
                prev = entry.map
                prev_q = entry.q
                if prev === ii
                    qi = prev_q
                    q_reused[i] = true
                elseif size(prev, 1) <= size(ii, 1) &&
                       size(prev, 2) <= size(ii, 2) &&
                       _is_appended_zero_matrix(field, prev, ii)
                    qi = _extend_cokernel_q_appended(K, prev_q, size(prev, 1), size(ii, 1))
                    q_reused[i] = true
                end
            end
        end
        if qi === nothing
            if _is_partial_permutation(field, ii)
                qi = _cokernel_q_partial_permutation(K, ii)
            else
                Ni = FieldLinAlg.nullspace(field, transpose(ii))     # dim E_i x (dim E_i - rank)
                qi = transpose(Ni)
            end
        end
        if ic !== nothing && (active_mask === nothing || active_mask[i])
            ic[i] = ii isa Matrix{K} ? _CokernelIncrementalEntry{K}(ii, qi) :
                                       _CokernelIncrementalEntry{K}(Matrix{K}(ii), qi)
        end
        Cdims[i] = size(qi, 1)
        qcomps[i] = qi
    end

    preds = graph_lists === nothing ? [collect(_preds(cc, v)) for v in 1:n] : graph_lists.preds
    succs = graph_lists === nothing ? [collect(_succs(cc, u)) for u in 1:n] : graph_lists.succs
    pred_slot_of_succ = graph_lists === nothing ?
        [collect(_pred_slots_of_succ(cc, u)) for u in 1:n] : graph_lists.pred_slot_of_succ
    qtrans = Vector{Matrix{K}}(undef, n)
    nactive = active_vertices === nothing ? (active_mask === nothing ? n : count(active_mask)) : length(active_vertices)
    use_active_frontier_fastpath = active_mask !== nothing && nactive < n
    active_succ_vertices = Int[]
    active_succ_mask = if use_active_frontier_fastpath
        mask = falses(n)
        if active_vertices !== nothing
            sizehint!(active_succ_vertices, length(active_vertices))
            @inbounds for u in active_vertices
                if !mask[u]
                    mask[u] = true
                    push!(active_succ_vertices, u)
                end
                for v in succs[u]
                    if !mask[v]
                        mask[v] = true
                        push!(active_succ_vertices, v)
                    end
                end
            end
        else
            mask = copy(active_mask)::BitVector
            @inbounds for u in 1:n
                if active_mask[u]
                    for v in succs[u]
                        mask[v] = true
                    end
                end
            end
            sizehint!(active_succ_vertices, count(mask))
            @inbounds for u in 1:n
                mask[u] || continue
                push!(active_succ_vertices, u)
            end
        end
        if count(mask) < n
            mask
        else
            use_active_frontier_fastpath = false
            empty!(active_succ_vertices)
            nothing
        end
    else
        nothing
    end
    selector_rows = Vector{Vector{Int}}(undef, n)
    selector_is_std = falses(n)
    empty_selector = Int[]
    changed_vertices = Int[]
    sizehint!(changed_vertices, max(1, nactive))
    if transport_cache !== nothing
        if length(transport_cache.active_mask) != n
            transport_cache.active_mask = falses(n)
        end
        if length(transport_cache.frontier_mask) != n
            transport_cache.frontier_mask = falses(n)
        end
        length(transport_cache.qsig_rows) == n || resize!(transport_cache.qsig_rows, n)
        length(transport_cache.qsig_cols) == n || resize!(transport_cache.qsig_cols, n)
        length(transport_cache.qtrans) == n || resize!(transport_cache.qtrans, n)
        length(transport_cache.selector_rows) == n || resize!(transport_cache.selector_rows, n)
        length(transport_cache.selector_is_std) == n || resize!(transport_cache.selector_is_std, n)
    end
    @inbounds for i in 1:n
        rows_i = size(qcomps[i], 1)
        cols_i = size(qcomps[i], 2)
        cache_reusable = transport_cache !== nothing &&
                         q_reused[i] &&
                         transport_cache.qsig_rows[i] == rows_i &&
                         transport_cache.qsig_cols[i] == cols_i &&
                         ((active_mask === nothing && !transport_cache.active_mask[i]) ||
                          (active_mask !== nothing && transport_cache.active_mask[i] == active_mask[i])) &&
                         ((active_succ_mask === nothing && !transport_cache.frontier_mask[i]) ||
                          (active_succ_mask !== nothing && transport_cache.frontier_mask[i] == active_succ_mask[i]))
        if cache_reusable
            qti = transport_cache.qtrans[i]
            qtrans[i] = qti === nothing ? Matrix(transpose(qcomps[i])) : qti
            selector_rows[i] = transport_cache.selector_rows[i]
            selector_is_std[i] = transport_cache.selector_is_std[i]
        else
            qti = transpose(qcomps[i])
            qtrans[i] = qti isa Matrix{K} ? qti : Matrix(qti)
            if !use_active_frontier_fastpath || active_succ_mask[i]
                rows = _selector_rows_from_basis(field, qtrans[i])
                if rows === nothing
                    selector_rows[i] = empty_selector
                else
                    selector_rows[i] = rows
                    selector_is_std[i] = true
                end
            else
                selector_rows[i] = empty_selector
            end
            push!(changed_vertices, i)
        end
        if transport_cache !== nothing
            transport_cache.qsig_rows[i] = rows_i
            transport_cache.qsig_cols[i] = cols_i
            transport_cache.active_mask[i] = active_mask === nothing ? false : active_mask[i]
            transport_cache.frontier_mask[i] = active_succ_mask === nothing ? false : active_succ_mask[i]
            transport_cache.qtrans[i] = qtrans[i]
            transport_cache.selector_rows[i] = selector_rows[i]
            transport_cache.selector_is_std[i] = selector_is_std[i]
        end
    end
    if transport_cache !== nothing
        empty!(transport_cache.changed_vertices)
        append!(transport_cache.changed_vertices, changed_vertices)
        empty!(transport_cache.frontier_vertices)
        if active_succ_mask !== nothing
            append!(transport_cache.frontier_vertices, active_succ_vertices)
        end
    end
    OutMatT = _store_out_type(K, _module_prefers_sparse_store(E))
    field_is_qq = field isa QQField

    maps_from_pred = [Vector{OutMatT}(undef, length(preds[v])) for v in 1:n]
    maps_to_succ   = [Vector{OutMatT}(undef, length(succs[u])) for u in 1:n]

    @inbounds for u in 1:n
        su     = succs[u]
        maps_u = E.edge_maps.maps_to_succ[u]   # aligned with su
        outu   = maps_to_succ[u]
        q_u_t = qtrans[u]
        q_u_t_mat = q_u_t isa Matrix{K} ? q_u_t : Matrix{K}(q_u_t)
        du = Cdims[u]
        is_selector_u = selector_is_std[u]
        rows_u = selector_rows[u]
        inactive_u = use_active_frontier_fastpath && active_mask !== nothing && !active_mask[u]

        if du == 0
            for j in eachindex(su)
                v = su[j]
                Auv = _zero_store_map(OutMatT, K, Cdims[v], 0)
                outu[j] = Auv
                ip = pred_slot_of_succ[u][j]
                maps_from_pred[v][ip] = Auv
            end
            continue
        end

        if inactive_u
            for j in eachindex(su)
                v = su[j]
                cv = Cdims[v]
                inactive_v = active_mask !== nothing && !active_mask[v]
                Auv = if cv == 0
                    _zero_store_map(OutMatT, K, 0, du)
                elseif inactive_v
                    _to_store_map(OutMatT, maps_u[j])
                elseif selector_is_std[v]
                    _to_store_map(OutMatT, view(maps_u[j], selector_rows[v], :))
                else
                    _to_store_map(OutMatT, qcomps[v] * maps_u[j])
                end
                outu[j] = Auv
                ip = pred_slot_of_succ[u][j]
                maps_from_pred[v][ip] = Auv
            end
            continue
        end

        # Batch all outgoing edge solves at fixed u into one solve_fullcolumn call.
        # For edge u->v we need A_uv with:
        #   q_v * T_uv = A_uv * q_u
        # equivalently (transpose):
        #   transpose(q_u) * transpose(A_uv) = transpose(T_uv) * transpose(q_v)
        # so each RHS block is transpose(T_uv) * qtrans[v].
        total_cols = 0
        for v in su
            total_cols += Cdims[v]
        end

        if total_cols == 0
            for j in eachindex(su)
                v = su[j]
                Auv = _zero_store_map(OutMatT, K, 0, du)
                outu[j] = Auv
                ip = pred_slot_of_succ[u][j]
                maps_from_pred[v][ip] = Auv
            end
            continue
        end

        if is_selector_u
            for j in eachindex(su)
                v = su[j]
                cv = Cdims[v]
                inactive_v = active_mask !== nothing && !active_mask[v]
                Auv = if cv == 0
                    _zero_store_map(OutMatT, K, 0, du)
                elseif inactive_u && inactive_v
                    _to_store_map(OutMatT, maps_u[j])
                elseif inactive_v
                    _to_store_map(OutMatT, view(maps_u[j], :, rows_u))
                elseif selector_is_std[v]
                    _to_store_map(OutMatT, maps_u[j][selector_rows[v], rows_u])
                else
                    _to_store_map(OutMatT, qcomps[v] * view(maps_u[j], :, rows_u))
                end
                outu[j] = Auv
                ip = pred_slot_of_succ[u][j]
                maps_from_pred[v][ip] = Auv
            end
            continue
        end

        use_batch = (length(su) >= COKERNEL_BATCH_MIN_FANOUT) &&
                    (total_cols >= COKERNEL_BATCH_MIN_TOTAL_RHS_COLS)
        qq_factor = (field_is_qq && length(su) > 1) ? FieldLinAlg._factor_fullcolumnQQ(q_u_t_mat) : nothing
        if !use_batch
            max_cv = 0
            for v in su
                max_cv = max(max_cv, Cdims[v])
            end
            rhs_buf = Matrix{K}(undef, size(q_u_t_mat, 1), max_cv)
            for j in eachindex(su)
                v = su[j]
                cv = Cdims[v]
                Auv = if cv == 0
                    _zero_store_map(OutMatT, K, 0, du)
                else
                    rhs = view(rhs_buf, :, 1:cv)
                    inactive_v = active_mask !== nothing && !active_mask[v]
                    if inactive_u && inactive_v
                        outu[j] = _to_store_map(OutMatT, maps_u[j])
                        ip = pred_slot_of_succ[u][j]
                        maps_from_pred[v][ip] = outu[j]
                        continue
                    elseif inactive_v
                        rhs .= transpose(maps_u[j])
                    elseif selector_is_std[v]
                        rhs .= transpose(view(maps_u[j], selector_rows[v], :))
                    else
                        mul!(rhs, transpose(maps_u[j]), qtrans[v])
                    end
                    X = if qq_factor === nothing
                        FieldLinAlg.solve_fullcolumn(field, q_u_t_mat, rhs;
                                                     check_rhs=!_FAST_SOLVE_NO_CHECK[],
                                                     cache=false)
                    else
                        FieldLinAlg.solve_fullcolumn(field, q_u_t_mat, rhs;
                                                     check_rhs=!_FAST_SOLVE_NO_CHECK[],
                                                     cache=false, factor=qq_factor)
                    end
                    _to_store_map(OutMatT, copy(transpose(X)))
                end
                outu[j] = Auv
                ip = pred_slot_of_succ[u][j]
                maps_from_pred[v][ip] = Auv
            end
            continue
        end

        offsets = Vector{Int}(undef, length(su) + 1)
        offsets[1] = 1
        for j in eachindex(su)
            offsets[j + 1] = offsets[j] + Cdims[su[j]]
        end

        rhs_all = Matrix{K}(undef, size(q_u_t_mat, 1), total_cols)
        for j in eachindex(su)
            v = su[j]
            cv = Cdims[v]
            cv == 0 && continue
            if inactive_u && active_mask !== nothing && !active_mask[v]
                continue
            end
            c0 = offsets[j]
            c1 = offsets[j + 1] - 1
            block = view(rhs_all, :, c0:c1)
            if active_mask !== nothing && !active_mask[v]
                block .= transpose(maps_u[j])
            elseif selector_is_std[v]
                block .= transpose(view(maps_u[j], selector_rows[v], :))
            else
                mul!(block, transpose(maps_u[j]), qtrans[v])
            end
        end

        Xall = if qq_factor === nothing
            FieldLinAlg.solve_fullcolumn(field, q_u_t_mat, rhs_all;
                                         check_rhs=!_FAST_SOLVE_NO_CHECK[],
                                         cache=false)
        else
            FieldLinAlg.solve_fullcolumn(field, q_u_t_mat, rhs_all;
                                         check_rhs=!_FAST_SOLVE_NO_CHECK[],
                                         cache=false, factor=qq_factor)
        end
        for j in eachindex(su)
            v = su[j]
            cv = Cdims[v]
            Auv = if cv == 0
                _zero_store_map(OutMatT, K, 0, du)
            elseif inactive_u && active_mask !== nothing && !active_mask[v]
                _to_store_map(OutMatT, maps_u[j])
            else
                c0 = offsets[j]
                c1 = offsets[j + 1] - 1
                Xblk = view(Xall, :, c0:c1)   # du x cv
                _to_store_map(OutMatT, copy(transpose(Xblk)))
            end
            outu[j] = Auv
            ip = pred_slot_of_succ[u][j]
            maps_from_pred[v][ip] = Auv
        end
    end

    storeC = CoverEdgeMapStore{K,OutMatT}(preds, succs, maps_from_pred, maps_to_succ, cc.nedges)
    Cmod = PModule{K}(Q, Cdims, storeC; field=field)
    q = PMorphism(E, Cmod, qcomps)
    return Cmod, q
end

@inline _cokernel_only_cached(iota::PMorphism{K}, cc::CoverCache) where {K} =
    (_cokernel_module_cached(iota, cc))[1]

"""
    _coimage_with_projection_cached(f, cc) -> (Coim, p)

Internal cached coimage constructor.

Returns the coimage `dom(f) / ker(f)` together with its quotient projection.
This is implemented as the cokernel of the cached kernel inclusion.
"""
@inline function _coimage_with_projection_cached(f::PMorphism{K}, cc::CoverCache) where {K}
    Kmod, iK = _kernel_with_inclusion_cached(f, cc)
    return _cokernel_module_cached(iK, cc)
end

@inline _coimage_only_cached(f::PMorphism{K}, cc::CoverCache) where {K} =
    (_coimage_with_projection_cached(f, cc))[1]

@inline _quotient_with_projection_cached(iota::PMorphism{K}, cc::CoverCache) where {K} =
    _cokernel_module_cached(iota, cc)

@inline _quotient_only_cached(iota::PMorphism{K}, cc::CoverCache) where {K} =
    (_quotient_with_projection_cached(iota, cc))[1]

"""
    _equalizer_cached(f, g, cc) -> (E, e)

Internal cached equalizer of two parallel morphisms.

Returns the equalizer object `E` and its inclusion `e : E -> dom(f)`. When
`f === g`, this returns the identity equalizer without extra algebra.
"""
@inline function _equalizer_cached(f::PMorphism{K}, g::PMorphism{K}, cc::CoverCache) where {K}
    if f === g
        return f.dom, id_morphism(f.dom)
    end
    return _kernel_with_inclusion_cached(_difference_morphism(f, g), cc)
end

"""
    _coequalizer_cached(f, g, cc) -> (Q, q)

Internal cached coequalizer of two parallel morphisms.

Returns the coequalizer object `Q` and its quotient map `q : cod(f) -> Q`. When
`f === g`, this returns the identity coequalizer without extra algebra.
"""
@inline function _coequalizer_cached(f::PMorphism{K}, g::PMorphism{K}, cc::CoverCache) where {K}
    if f === g
        return f.cod, id_morphism(f.cod)
    end
    return _cokernel_module_cached(_difference_morphism(f, g), cc)
end

"""
    _equalizer_public(f, g, cache) -> (E, e)

Public-wrapper backend for `equalizer` after resolving the public cache contract.

Prefer calling `equalizer` from user code.
"""
@inline function _equalizer_public(f::PMorphism{K}, g::PMorphism{K}, cache) where {K}
    if f === g
        return f.dom, id_morphism(f.dom)
    end
    cc = _resolve_public_cover_cache(f.dom.Q, f.dom.field, cache)
    return _kernel_with_inclusion_cached(_difference_morphism(f, g), cc)
end

"""
    _coequalizer_public(f, g, cache) -> (Q, q)

Public-wrapper backend for `coequalizer` after resolving the public cache
contract. Prefer calling `coequalizer` from user code.
"""
@inline function _coequalizer_public(f::PMorphism{K}, g::PMorphism{K}, cache) where {K}
    if f === g
        return f.cod, id_morphism(f.cod)
    end
    cc = _resolve_public_cover_cache(f.cod.Q, f.cod.field, cache)
    return _cokernel_module_cached(_difference_morphism(f, g), cc)
end

"""
    _coimage_with_projection_public(f, cache) -> (Coim, p)

Public-wrapper backend for `coimage_with_projection` after resolving the public
cache contract.
"""
@inline function _coimage_with_projection_public(f::PMorphism{K}, cache) where {K}
    cc = _resolve_public_cover_cache(f.dom.Q, f.dom.field, cache)
    _, iK = _kernel_with_inclusion_cached(f, cc)
    return _cokernel_module_cached(iK, cc)
end

"""
    _coimage_only_public(f, cache) -> Coim

Public-wrapper backend for `coimage` after resolving the public cache contract.
"""
@inline function _coimage_only_public(f::PMorphism{K}, cache) where {K}
    cc = _resolve_public_cover_cache(f.dom.Q, f.dom.field, cache)
    _, iK = _kernel_with_inclusion_cached(f, cc)
    return _cokernel_only_cached(iK, cc)
end

"""
    _quotient_with_projection_public(iota, cache) -> (Q, q)

Public-wrapper backend for `quotient_with_projection` after resolving the public
cache contract.
"""
@inline function _quotient_with_projection_public(iota::PMorphism{K}, cache) where {K}
    cc = _resolve_public_cover_cache(iota.cod.Q, iota.cod.field, cache)
    return _quotient_with_projection_cached(iota, cc)
end

"""
    _quotient_only_public(iota, cache) -> Q

Public-wrapper backend for `quotient` after resolving the public cache contract.
"""
@inline function _quotient_only_public(iota::PMorphism{K}, cache) where {K}
    cc = _resolve_public_cover_cache(iota.cod.Q, iota.cod.field, cache)
    return _quotient_only_cached(iota, cc)
end

"""
    _cokernel_module(iota; cache=:auto, incremental_cache=nothing) -> (Cmod, q)

Internal convenience wrapper around `_cokernel_module_cached`.

Returns the cokernel object and projection, resolving `cache` to a concrete
`CoverCache` first. This remains useful for internal callsites that want the
cached kernel but do not already hold a resolved cache.
"""
function _cokernel_module(
    iota::PMorphism{K};
    cache::Union{Symbol,CoverCache}=:auto,
    incremental_cache::Union{Nothing,AbstractVector}=nothing,
    active_vertices::Union{Nothing,Vector{Int}}=nothing,
    active_mask::Union{Nothing,BitVector}=nothing,
    transport_cache::Union{Nothing,_CokernelTransportCache{K}}=nothing,
    graph_lists::Union{Nothing,_CoverGraphLists}=nothing,
) where {K}
    cc = _resolve_cover_cache(iota.cod.Q, cache)
    return _cokernel_module_cached(iota, cc;
                                   incremental_cache=incremental_cache,
                                   active_vertices=active_vertices,
                                   active_mask=active_mask,
                                   transport_cache=transport_cache,
                                   graph_lists=graph_lists)
end

# =============================================================================
# Abelian category API (kernels/cokernels/images/coimages/quotients)
# =============================================================================

"""
    cokernel_with_projection(f; cache=:auto) -> (C, q)

Compute the cokernel of a morphism `f : A -> B`.

Mathematically, this returns the quotient module `C = B / im(f)` together with
the canonical projection `q : B -> C`.

Returns:
- `C`, the cokernel object,
- `q : B -> C`, the universal quotient map.

In this functor category, cokernels are computed stalkwise and the cover-edge
maps of the quotient are induced from those of `B`.

Caching:
- normal use: keep `cache=:auto`
- repeated low-level calls on one poset: call `build_cache!(Q)` once and reuse `cache=cc`

See also: `cokernel`, `image_with_inclusion`, `quotient_with_projection`.
"""
@inline function cokernel_with_projection(f::PMorphism{K}; cache::Union{Symbol,CoverCache}=:auto) where {K}
    cc = _resolve_cover_cache(f.cod.Q, cache)
    return _cokernel_module_cached(f, cc)
end

# ---------------------------------------------------------------------------
# Equalizers / coequalizers (abelian category = kernels / cokernels of f - g)
# ---------------------------------------------------------------------------

"""
    _difference_morphism(f, g) -> PMorphism

Return the stalkwise difference `f - g` of two parallel morphisms.

Both morphisms must have the same domain and codomain. This helper underlies
the equalizer/coequalizer implementations.
"""
function _difference_morphism(f::PMorphism{K}, g::PMorphism{K}) where {K}
    if f.dom !== g.dom || f.cod !== g.cod
        error("need parallel morphisms with the same domain and codomain")
    end
    comps = Matrix{K}[f.comps[i] - g.comps[i] for i in 1:nvertices(f.dom.Q)]
    return PMorphism{K}(f.dom, f.cod, comps)
end

"""
    equalizer(f, g; cache=:auto) -> (E, e)

Equalizer of parallel morphisms `f, g : A -> B`.

In an abelian category, the equalizer is `ker(f - g)` with inclusion map
`e : E -> A`.

Returns:
- `E` : the equalizer object
- `e` : the inclusion `E -> A`

Requirements:
- `f` and `g` must have the same domain and codomain.

Best practice:
- use `equalizer` directly for ordinary computations,
- use `limit(ParallelPairDiagram(f, g))` only when you are intentionally
  writing in the small-diagram categorical layer,
- leave `cache=:auto` unless you already hold a `CoverCache`.

See also: `coequalizer`, `kernel_with_inclusion`, `limit`.
"""
@inline function equalizer(f::PMorphism{K}, g::PMorphism{K}; cache::Union{Symbol,CoverCache}=:auto) where {K}
    return _equalizer_public(f, g, cache)
end

"""
    coequalizer(f, g; cache=:auto) -> (Q, q)

Coequalizer of parallel morphisms `f, g : A -> B`.

In an abelian category, the coequalizer is `coker(f - g)` with projection map
`q : B -> Q`.

Returns:
- `Q` : the coequalizer object
- `q` : the projection `B -> Q`

Requirements:
- `f` and `g` must be parallel morphisms.

Best practice:
- use `coequalizer` directly for ordinary computations,
- use `colimit(ParallelPairDiagram(f, g))` only in advanced diagram-level code,
- leave `cache=:auto` unless you already hold a `CoverCache`.

See also: `equalizer`, `cokernel_with_projection`, `colimit`.
"""
@inline function coequalizer(f::PMorphism{K}, g::PMorphism{K}; cache::Union{Symbol,CoverCache}=:auto) where {K}
    return _coequalizer_public(f, g, cache)
end


"""
    cokernel(f; cache=:auto) -> C

Return only the cokernel object of `f`.

This is the object-only companion of `cokernel_with_projection`.

Returns:
- `C`, the quotient module `cod(f) / im(f)`.

Best practice:
- use this when the quotient map is not needed,
- switch to `cokernel_with_projection` when you need the universal morphism,
- keep `cache=:auto` for normal use.
"""
@inline function cokernel(f::PMorphism{K}; cache::Union{Symbol,CoverCache}=:auto) where {K}
    cc = _resolve_cover_cache(f.cod.Q, cache)
    return _cokernel_only_cached(f, cc)
end

"""
    kernel(f; cache=:auto) -> K

Return only the kernel object of `f`.

This is the object-only companion of `kernel_with_inclusion`.

Returns:
- `K`, the submodule `ker(f)` as a `PModule`.

Best practice:
- use this when the inclusion map is not needed,
- switch to `kernel_with_inclusion` or `kernel_submodule` when you need the
  universal morphism or quotient bridge,
- keep `cache=:auto` for normal use.
"""
@inline function kernel(f::PMorphism{K}; cache::Union{Symbol,CoverCache}=:auto) where {K}
    cc = _resolve_cover_cache(f.dom.Q, cache)
    return _kernel_module_cached(f, cc)
end

"""
    image(f; cache=:auto) -> Im

Return only the image object of `f`.

Returns:
- `Im`, the image module viewed as a `PModule` on the codomain poset.

Best practice:
- use this when the inclusion `Im -> cod(f)` is not needed,
- use `image_with_inclusion` for the universal map,
- use `image_submodule` when the next step is `quotient(S)`.
"""
@inline function image(f::PMorphism{K}; cache::Union{Symbol,CoverCache}=:auto) where {K}
    cc = _resolve_cover_cache(f.cod.Q, cache)
    return _image_module_cached(f, cc)
end

"""
    coimage_with_projection(f; cache=:auto) -> (Coim, p)

Compute the coimage of f : A -> B, defined as Coim = A / ker(f),
together with the canonical projection p : A -> Coim.

In an abelian category, the canonical map Coim -> Im is an isomorphism.

Returns:
- `Coim`, the coimage module,
- `p : dom(f) -> Coim`, the canonical quotient projection.

Best practice:
- use `coimage(f)` when you only need the object,
- keep `cache=:auto` unless you are already reusing a `CoverCache`.
"""
@inline function coimage_with_projection(f::PMorphism{K}; cache::Union{Symbol,CoverCache}=:auto) where {K}
    return _coimage_with_projection_public(f, cache)
end

"""
    coimage(f; cache=:auto) -> Coim

Return only the coimage object `dom(f) / ker(f)`.

This is the object-only companion of `coimage_with_projection`.

Returns:
- `Coim`, the coimage module.

Best practice:
- use this when the quotient map is not needed,
- use `coimage_with_projection` when you need the universal morphism.
"""
@inline function coimage(f::PMorphism{K}; cache::Union{Symbol,CoverCache}=:auto) where {K}
    return _coimage_only_public(f, cache)
end

# Quotients of submodules: for now we represent a submodule by its inclusion morphism.
"""
    quotient_with_projection(iota; cache=:auto) -> (Q, q)

Given a monomorphism iota : N -> M (typically an inclusion), compute the quotient module
Q = M / N together with the projection q : M -> Q.

This is an advanced convenience when you already have an inclusion morphism.
The canonical public path is to build a `Submodule` via `kernel_submodule` or
`image_submodule`, then call `quotient_with_projection(S)`.

Returns:
- `Q`, the quotient module `cod(iota) / dom(iota)`,
- `q : cod(iota) -> Q`, the canonical projection.

Best practice:
- prefer `quotient_with_projection(S::Submodule)` in ordinary code,
- use this overload only when you already have an inclusion morphism in hand.
"""
@inline function quotient_with_projection(iota::PMorphism{K}; cache::Union{Symbol,CoverCache}=:auto) where {K}
    _is_monomorphism(iota) || error("quotient_with_projection: expected a monomorphism (inclusion-like morphism)")
    return _quotient_with_projection_public(iota, cache)
end

"""
    quotient(iota; cache=:auto) -> Q

Return only the quotient module M/N for an inclusion iota : N -> M.

This is an advanced convenience when you already have an inclusion morphism.
The canonical public path is `quotient(S)` for a `Submodule`.

Returns:
- `Q`, the quotient module `cod(iota) / dom(iota)`.

Best practice:
- prefer `quotient(S::Submodule)` in ordinary user-facing code,
- use this overload only when the inclusion morphism is already available.
"""
@inline function quotient(iota::PMorphism{K}; cache::Union{Symbol,CoverCache}=:auto) where {K}
    return _quotient_only_public(iota, cache)
end

# Small predicates

"""
    is_zero_morphism(f) -> Bool

Return `true` exactly when every stalk component of `f` is the zero matrix.

This check is field-aware: exact fields use exact comparison, while
`RealField` uses the project-wide tolerance semantics inside `_is_zero_matrix`.

Best practice:
- use this predicate for contract checks and algorithm guards,
- do not infer anything stronger than pointwise zero from it.
"""
function is_zero_morphism(f::PMorphism{K}) where {K}
    field = f.dom.field
    for A in f.comps
        _is_zero_matrix(field, A) || return false
    end
    return true
end

"""
    _is_monomorphism(f) -> Bool

Check whether f is a monomorphism in the functor category of P-modules.
For these representations, this is equivalent to f_i being injective for every vertex i.

This method uses field-aware ranks (exact for exact fields, tolerance-based for RealField).
"""
function _is_monomorphism(f::PMorphism{K}) where {K}
    Q = f.dom.Q
    @assert f.cod.Q === Q
    for i in 1:nvertices(Q)
        if FieldLinAlg.rank(f.dom.field, f.comps[i]) != f.dom.dims[i]
            return false
        end
    end
    return true
end

"""
    _is_epimorphism(f) -> Bool

Check whether f is an epimorphism (pointwise surjective).
Implemented via field-aware ranks (exact for exact fields, tolerance-based for RealField).
"""
function _is_epimorphism(f::PMorphism{K}) where {K}
    Q = f.dom.Q
    @assert f.cod.Q === Q
    for i in 1:nvertices(Q)
        if FieldLinAlg.rank(f.dom.field, f.comps[i]) != f.cod.dims[i]
            return false
        end
    end
    return true
end


# -----------------------------------------------------------------------------
# Submodules as first-class objects
# -----------------------------------------------------------------------------

"""
    Submodule(incl)

A lightweight categorical wrapper for a submodule `N <= M`.

If `incl : N -> M` is an inclusion morphism, `Submodule(incl)` packages that
subobject as a first-class value. The wrapper is intentionally minimal: it
stores only the inclusion morphism, and the represented module and ambient
module are recovered through accessors.

Best practices
- construct user-facing values with `submodule(incl; ...)` so monicity checks
  and clearer error messages are available,
- inspect with `describe(S)` and `dimensions(S)` before reaching into the stored
  morphism,
- use `submodule_object(S)`, `ambient_module(S)`, and `inclusion_map(S)` rather
  than raw field access.
"""
struct Submodule{K}
    incl::PMorphism{K}  # incl : sub -> ambient
end

"""
    submodule(incl; check_mono=true) -> Submodule

Build a `Submodule` from an inclusion map.

Inputs:
- `incl : N -> M`, intended to represent an inclusion,
- `check_mono`, whether to verify pointwise injectivity first.

Returns:
- `S`, a lightweight `Submodule` wrapper storing the inclusion map.

Best practice:
- leave `check_mono=true` unless the caller already proved monicity and needs to
  avoid duplicate work,
- prefer `kernel_submodule(f)` or `image_submodule(f)` when those canonical
  constructions already provide the submodule you need.
"""
function submodule(incl::PMorphism{K}; check_mono::Bool=true) where {K}
    check_mono && !_is_monomorphism(incl) && error("submodule: given inclusion is not a monomorphism")
    return Submodule{K}(incl)
end

"""
    _sub(S::Submodule) -> PModule

Return the underlying module object represented by the `Submodule`.

If `S` represents an inclusion `N <= M`, this returns `N`. This helper is
internal; user-facing code should usually work with the `Submodule` wrapper
itself unless raw module access is required.
"""
@inline _sub(S::Submodule) = S.incl.dom

"""
    _ambient(S::Submodule) -> PModule

Return the ambient module of a `Submodule`.

If `S` represents `N <= M`, this returns `M`. This helper is internal and is
used to keep owner-module logic explicit without exposing extra public surface.
"""
@inline _ambient(S::Submodule) = S.incl.cod

"""
    submodule_object(S::Submodule) -> PModule

Return the module represented by the subobject `S`.

If `S` was built from an inclusion `N <= M`, this returns `N`.

Best practice:
- use this accessor instead of reaching into internal fields or internal helpers
  when you want the stored submodule object explicitly.
"""
@inline submodule_object(S::Submodule) = _sub(S)

"""
    ambient_module(S::Submodule) -> PModule

Return the ambient module of the subobject `S`.

If `S` represents `N <= M`, this returns `M`.

Best practice:
- use this accessor when you want the ambient object for quotient or image
  constructions, instead of reaching into stored fields.
"""
@inline ambient_module(S::Submodule) = _ambient(S)

"""
    inclusion_map(S::Submodule) -> PMorphism

Return the inclusion morphism stored by `S`.

Best practice:
- use this accessor when passing a `Submodule` back into categorical routines,
  rather than reaching into `S.incl` directly.
"""
@inline inclusion_map(S::Submodule) = S.incl

"""
    dimensions(S::Submodule) -> NamedTuple

Return a compact dimension summary for a `Submodule`.

The result records:
- the number of vertices,
- the stalk dimensions of the represented submodule,
- the stalk dimensions of the ambient module,
- the corresponding total dimensions.
"""
@inline _submodule_dimensions(S::Submodule) = (
    vertices = nvertices(_ambient(S).Q),
    submodule = copy(_sub(S).dims),
    ambient = copy(_ambient(S).dims),
    submodule_total = sum(_sub(S).dims),
    ambient_total = sum(_ambient(S).dims),
)

"""
    describe(S::Submodule) -> NamedTuple

Return a compact mathematical summary of a `Submodule`.

This is the notebook/REPL companion to `show(S)`: it exposes the field, number
of vertices, and the dimension summary in a structured form.
"""
@inline _submodule_describe(S::Submodule{K}) where {K} = (
    kind = :submodule,
    field = _scalar_name(K),
    dimensions = _submodule_dimensions(S),
)

"""
    check_submodule(S; throw=false) -> NamedTuple

Validate a `Submodule` wrapper built from an inclusion map.

The returned report has fields:
- `kind = :submodule`
- `valid::Bool`
- `issues::Vector{String}`

What is checked
- structural validity of the inclusion morphism,
- structural validity of the represented and ambient modules,
- pointwise monomorphism of the inclusion.

Best practices
- use this after constructing a `Submodule` by hand or after wrapping a map you
  believe to be an inclusion,
- use `throw=true` in tests and scripts when invalid input should stop
  execution immediately.
"""
function check_submodule(S::Submodule{K}; throw::Bool=false) where {K}
    issues = String[]
    rep_sub = Modules.check_module(_sub(S))
    rep_amb = Modules.check_module(_ambient(S))
    rep_incl = Modules.check_morphism(S.incl)
    rep_sub.valid || append!(issues, "submodule object: " .* rep_sub.issues)
    rep_amb.valid || append!(issues, "ambient module: " .* rep_amb.issues)
    rep_incl.valid || append!(issues, "inclusion morphism: " .* rep_incl.issues)
    _is_monomorphism(S.incl) || push!(issues, "expected the stored inclusion map to be a monomorphism")
    report = (
        kind = :submodule,
        valid = isempty(issues),
        issues = issues,
        dimensions = _submodule_dimensions(S),
    )
    throw && !report.valid && _abelian_validation_error("check_submodule", issues)
    return report
end

"""
    _inclusion(S::Submodule) -> PMorphism

Return the stored inclusion morphism of a `Submodule`.

If `S` represents `N <= M`, this returns the monomorphism `N -> M`. This is an
internal helper used by quotient and exactness code paths.
"""
@inline _inclusion(S::Submodule) = S.incl

"""
    quotient_with_projection(S::Submodule; cache=:auto) -> (Q, q)

Compute the quotient of the ambient module by the represented submodule.

If `S` represents `N <= M`, this returns:
- `Q = M / N`,
- `q : M -> Q`, the canonical projection.

This is the canonical public quotient interface for ordinary user code.
Prefer this overload over the inclusion-morphism convenience overload unless
you specifically need to work with raw monomorphisms.
"""
@inline function quotient_with_projection(S::Submodule{K}; cache::Union{Symbol,CoverCache}=:auto) where {K}
    return _quotient_with_projection_public(S.incl, cache)
end
"""
    quotient(S::Submodule; cache=:auto) -> Q

Return only the quotient object associated to a `Submodule`.

If `S` represents `N <= M`, the returned module is `Q = M / N`.

Best practice:
- use this as the default quotient entrypoint in examples and user-facing code,
- use `quotient_with_projection(S)` when the canonical projection is also
  needed.
"""
@inline function quotient(S::Submodule{K}; cache::Union{Symbol,CoverCache}=:auto) where {K}
    return _quotient_only_public(S.incl, cache)
end

"""
    kernel_submodule(f; cache=:auto) -> Submodule

Return ker(f) <= dom(f) as a `Submodule`.

This is part of the canonical bridge from universal constructions to quotients:
`kernel_submodule(f)` / `image_submodule(f)` followed by `quotient(S)`.

Returns:
- `S`, a `Submodule` whose ambient object is `dom(f)` and whose underlying
  inclusion is the universal kernel inclusion.
"""
@inline function kernel_submodule(f::PMorphism{K}; cache::Union{Symbol,CoverCache}=:auto) where {K}
    cc = _resolve_cover_cache(f.dom.Q, cache)
    _, iota = _kernel_with_inclusion_cached(f, cc)
    return Submodule{K}(iota)
end

"""
    image_submodule(f; cache=:auto) -> Submodule

Return im(f) <= cod(f) as a `Submodule`.

This is part of the canonical bridge from universal constructions to quotients:
`kernel_submodule(f)` / `image_submodule(f)` followed by `quotient(S)`.

Returns:
- `S`, a `Submodule` whose ambient object is `cod(f)` and whose inclusion is the
  universal image inclusion.
"""
@inline function image_submodule(f::PMorphism{K}; cache::Union{Symbol,CoverCache}=:auto) where {K}
    cc = _resolve_cover_cache(f.cod.Q, cache)
    _, iota = _image_with_inclusion_cached(f, cc)
    return Submodule{K}(iota)
end

# -----------------------------------------------------------------------------
# Pushouts and pullbacks
# -----------------------------------------------------------------------------

"""
    _right_inverse_full_row(field, Q) -> Matrix

Compute a right inverse of a full-row-rank matrix `Q`.

If `Q` has size `r x m`, the returned matrix `R` has size `m x r` and satisfies
`Q * R = I_r`.

This is an internal helper used in cokernel transport and snake-lemma
construction. It assumes `Q` has full row rank and throws an error otherwise.
"""
function _right_inverse_full_row(field::AbstractCoeffField, Q::AbstractMatrix{K}) where {K}
    r, m = size(Q)
    if r == 0
        return zeros(K, m, 0)
    end
    Qm = Q isa Matrix{K} ? Q : Matrix(Q)
    _, pivs = FieldLinAlg.rref(field, Qm)
    length(pivs) == r || error("_right_inverse_full_row: expected full row rank, got rank $(length(pivs)) < $r")

    piv = collect(pivs[1:r])
    Qp = Qm[:, piv]
    invQp = FieldLinAlg.solve_fullcolumn(field, Qp, eye(field, r))

    R = zeros(K, m, r)
    @inbounds for i in 1:r
        R[piv[i], :] = invQp[i, :]
    end
    return R
end

"""
    _pushout_phi_component(K, fu, gu) -> Matrix{K}

Build the vertexwise block matrix `[fu; -gu]` used to define a pushout as a
cokernel.

This helper does not validate shape compatibility beyond what matrix assembly
already requires.
"""
@inline function _pushout_phi_component(::Type{K}, fu::AbstractMatrix{K}, gu::AbstractMatrix{K}) where {K}
    b = size(fu, 1)
    c = size(gu, 1)
    a = size(fu, 2)
    M = Matrix{K}(undef, b + c, a)
    if b > 0
        copyto!(view(M, 1:b, :), fu)
    end
    if c > 0
        @inbounds for i in 1:c, j in 1:a
            M[b + i, j] = -gu[i, j]
        end
    end
    return M
end

"""
    _pullback_psi_component(K, fu, gu) -> Matrix{K}

Build the vertexwise block matrix `[fu  -gu]` used to define a pullback as a
kernel.
"""
@inline function _pullback_psi_component(::Type{K}, fu::AbstractMatrix{K}, gu::AbstractMatrix{K}) where {K}
    d = size(fu, 1)
    b = size(fu, 2)
    c = size(gu, 2)
    M = Matrix{K}(undef, d, b + c)
    if b > 0
        copyto!(view(M, :, 1:b), fu)
    end
    if c > 0
        @inbounds for i in 1:d, j in 1:c
            M[i, b + j] = -gu[i, j]
        end
    end
    return M
end

"""
    _mul_pushout_rhs!(rhs, Buv, Cuv, qv_t, bu, bv) -> rhs

Fill `rhs` with the block action needed in the pushout transport solve for one
cover edge.

This is a mutating internal helper used to avoid repeated temporary block
matrices in the pushout builder.
"""
@inline function _mul_pushout_rhs!(
    rhs::AbstractMatrix{K},
    Buv::AbstractMatrix{K},
    Cuv::AbstractMatrix{K},
    qv_t::AbstractMatrix{K},
    bu::Int,
    bv::Int,
) where {K}
    if bu > 0
        mul!(view(rhs, 1:bu, :), transpose(Buv), view(qv_t, 1:bv, :))
    end
    cu = size(rhs, 1) - bu
    cv = size(qv_t, 1) - bv
    if cu > 0
        mul!(view(rhs, bu + 1:bu + cu, :), transpose(Cuv), view(qv_t, bv + 1:bv + cv, :))
    end
    return rhs
end

"""
    _mul_pullback_im!(im, Buv, Cuv, Ku, bu, bv) -> im

Fill `im` with the block image needed in the pullback transport solve for one
cover edge.

This is the pullback companion to `_mul_pushout_rhs!`.
"""
@inline function _mul_pullback_im!(
    im::AbstractMatrix{K},
    Buv::AbstractMatrix{K},
    Cuv::AbstractMatrix{K},
    Ku::AbstractMatrix{K},
    bu::Int,
    bv::Int,
) where {K}
    du = size(Ku, 2)
    if bv > 0
        mul!(view(im, 1:bv, 1:du), Buv, view(Ku, 1:bu, :))
    end
    cu = size(Ku, 1) - bu
    cv = size(im, 1) - bv
    if cv > 0
        mul!(view(im, bv + 1:bv + cv, 1:du), Cuv, view(Ku, bu + 1:bu + cu, :))
    end
    return im
end

"""
    _pushout_same_map(f, cc) -> (P, in1, in2, q, phi)

Internal specialized pushout constructor for the diagonal case `pushout(f, f)`.

Returns the pushout object, the two comparison maps into it, the quotient map
from `cod(f) oplus cod(f)`, and the defining map `phi`.
"""
function _pushout_same_map(
    f::PMorphism{K},
    cc::CoverCache,
) where {K}
    A = f.dom
    B = f.cod
    Cok, qf = _cokernel_module_cached(f, cc)
    P, _, _, _, _ = direct_sum_with_maps(B, Cok)
    S = direct_sum(B, B)
    n = nvertices(B.Q)

    q_comps = Vector{Matrix{K}}(undef, n)
    phi_comps = Vector{Matrix{K}}(undef, n)
    in1_comps = Vector{Matrix{K}}(undef, n)
    in2_comps = Vector{Matrix{K}}(undef, n)
    field = B.field
    @inbounds for u in 1:n
        b = B.dims[u]
        c = Cok.dims[u]
        qu = zeros(K, b + c, 2b)
        for t in 1:b
            qu[t, t] = one(K)
            qu[t, b + t] = one(K)
        end
        if c > 0 && b > 0
            view(qu, b + 1:b + c, 1:b) .= qf.comps[u]
        end
        q_comps[u] = qu
        phi_comps[u] = _pushout_phi_component(K, f.comps[u], f.comps[u])
        in1_comps[u] = copy(view(qu, :, 1:b))
        in2_comps[u] = copy(view(qu, :, b + 1:2b))
    end

    q = PMorphism{K}(S, P, q_comps)
    phi = PMorphism{K}(A, S, phi_comps)
    in1 = PMorphism{K}(B, P, in1_comps)
    in2 = PMorphism{K}(B, P, in2_comps)
    return P, in1, in2, q, phi
end

"""
    _pullback_same_map(f, cc) -> (P, pr1, pr2, iota, psi)

Internal specialized pullback constructor for the diagonal case `pullback(f, f)`.

Returns the pullback object, the two projections, the kernel inclusion into
`dom(f) oplus dom(f)`, and the defining map `psi`.
"""
function _pullback_same_map(
    f::PMorphism{K},
    cc::CoverCache,
) where {K}
    A = f.dom
    B = f.cod
    Kmod, iK = _kernel_with_inclusion_cached(f, cc)
    P, _, _, _, _ = direct_sum_with_maps(A, Kmod)
    S = direct_sum(A, A)
    n = nvertices(A.Q)

    iota_comps = Vector{Matrix{K}}(undef, n)
    psi_comps = Vector{Matrix{K}}(undef, n)
    pr1_comps = Vector{Matrix{K}}(undef, n)
    pr2_comps = Vector{Matrix{K}}(undef, n)
    @inbounds for u in 1:n
        a = A.dims[u]
        k = Kmod.dims[u]
        iu = zeros(K, 2a, a + k)
        for t in 1:a
            iu[t, t] = one(K)
            iu[a + t, t] = one(K)
        end
        if k > 0
            view(iu, a + 1:2a, a + 1:a + k) .= iK.comps[u]
        end
        iota_comps[u] = iu
        psi_comps[u] = _pullback_psi_component(K, f.comps[u], f.comps[u])
        pr1_comps[u] = copy(view(iu, 1:a, :))
        pr2_comps[u] = copy(view(iu, a + 1:2a, :))
    end

    iota = PMorphism{K}(P, S, iota_comps)
    psi = PMorphism{K}(S, B, psi_comps)
    pr1 = PMorphism{K}(P, A, pr1_comps)
    pr2 = PMorphism{K}(P, A, pr2_comps)
    return P, pr1, pr2, iota, psi
end

"""
    _pushout_public(f, g, cache) -> (P, inB, inC, q, phi)

Resolve the public cache contract and dispatch to the appropriate pushout
backend, including the `f === g` specialization.

User code should call `pushout`.
"""
@inline function _pushout_public(f::PMorphism{K}, g::PMorphism{K}, cache) where {K}
    @assert f.dom === g.dom
    cc = _resolve_public_cover_cache(f.cod.Q, f.cod.field, cache)
    if f === g
        return _pushout_same_map(f, cc)
    end
    B = f.cod
    C = g.cod
    S, P, q, phi = _pushout_cokernel_module(f, g, cc)
    Q = S.Q

    inB_comps = Vector{Matrix{K}}(undef, nvertices(Q))
    inC_comps = Vector{Matrix{K}}(undef, nvertices(Q))
    @inbounds for u in 1:nvertices(Q)
        qu = q.comps[u]
        b = B.dims[u]
        c = C.dims[u]
        inB_comps[u] = copy(view(qu, :, 1:b))
        inC_comps[u] = copy(view(qu, :, (b + 1):(b + c)))
    end
    inB = PMorphism{K}(B, P, inB_comps)
    inC = PMorphism{K}(C, P, inC_comps)
    return P, inB, inC, q, phi
end

"""
    _pullback_public(f, g, cache) -> (P, prB, prC, iota, psi)

Resolve the public cache contract and dispatch to the appropriate pullback
backend, including the `f === g` specialization.

User code should call `pullback`.
"""
@inline function _pullback_public(f::PMorphism{K}, g::PMorphism{K}, cache) where {K}
    @assert f.cod === g.cod
    cc = _resolve_public_cover_cache(f.dom.Q, f.dom.field, cache)
    if f === g
        return _pullback_same_map(f, cc)
    end
    B = f.dom
    C = g.dom
    S, P, iota, psi = _pullback_kernel_with_inclusion(f, g, cc)
    Q = S.Q

    prB_comps = Vector{Matrix{K}}(undef, nvertices(Q))
    prC_comps = Vector{Matrix{K}}(undef, nvertices(Q))
    @inbounds for u in 1:nvertices(Q)
        iu = iota.comps[u]
        b = B.dims[u]
        c = C.dims[u]
        prB_comps[u] = copy(view(iu, 1:b, :))
        prC_comps[u] = copy(view(iu, (b + 1):(b + c), :))
    end
    prB = PMorphism{K}(P, B, prB_comps)
    prC = PMorphism{K}(P, C, prC_comps)
    return P, prB, prC, iota, psi
end

"""
    _product_public(M, N) -> (P, pM, pN)

Internal backend for the binary categorical product wrapper.

Returns the biproduct object together with the two canonical projections.
"""
@inline function _product_public(M::PModule{K}, N::PModule{K}) where {K}
    P, _, _, pM, pN = direct_sum_with_maps(M, N)
    return P, pM, pN
end

"""
    _coproduct_public(M, N) -> (S, iM, iN)

Internal backend for the binary categorical coproduct wrapper.

Returns the biproduct object together with the two canonical injections.
"""
@inline function _coproduct_public(M::PModule{K}, N::PModule{K}) where {K}
    S, iM, iN, _, _ = direct_sum_with_maps(M, N)
    return S, iM, iN
end

"""
    _pushout_cokernel_module(f, g, cc) -> (S, P, q, phi)

Internal pushout builder via cokernels.

Returns:
- `S = cod(f) oplus cod(g)`,
- `P`: the pushout object,
- `q : S -> P`: the quotient map,
- `phi : dom(f) -> S`: the defining map whose cokernel is the pushout.

Higher-level callers typically use `_pushout_public` or `pushout`.
"""
function _pushout_cokernel_module(
    f::PMorphism{K},
    g::PMorphism{K},
    cc::CoverCache,
) where {K}
    A = f.dom
    B = f.cod
    C = g.cod
    Q = B.Q
    n = nvertices(Q)
    field = B.field

    phi_comps = Vector{Matrix{K}}(undef, n)
    Cdims = zeros(Int, n)
    qcomps = Vector{Matrix{K}}(undef, n)
    @inbounds for i in 1:n
        phi_i = _pushout_phi_component(K, f.comps[i], g.comps[i])
        phi_comps[i] = phi_i
        qi = if _is_partial_permutation(field, phi_i)
            _cokernel_q_partial_permutation(K, phi_i)
        else
            transpose(FieldLinAlg.nullspace(field, transpose(phi_i)))
        end
        qcomps[i] = qi
        Cdims[i] = size(qi, 1)
    end

    S = direct_sum(B, C)
    preds = [collect(_preds(cc, v)) for v in 1:n]
    succs = [collect(_succs(cc, u)) for u in 1:n]
    pred_slot_of_succ = [collect(_pred_slots_of_succ(cc, u)) for u in 1:n]
    qtrans = [transpose(qcomps[i]) for i in 1:n]
    selector_rows = Vector{Vector{Int}}(undef, n)
    selector_is_std = falses(n)
    empty_selector = Int[]
    for i in 1:n
        rows = _selector_rows_from_basis(field, qtrans[i])
        if rows === nothing
            selector_rows[i] = empty_selector
        else
            selector_rows[i] = rows
            selector_is_std[i] = true
        end
    end
    OutMatT = _store_out_type(K, _module_prefers_sparse_store(S))
    field_is_qq = field isa QQField

    maps_from_pred = [Vector{OutMatT}(undef, length(preds[v])) for v in 1:n]
    maps_to_succ = [Vector{OutMatT}(undef, length(succs[u])) for u in 1:n]

    @inbounds for u in 1:n
        su = succs[u]
        outu = maps_to_succ[u]
        du = Cdims[u]
        bu = B.dims[u]
        cu = C.dims[u]
        q_u_t = qtrans[u]
        q_u_t_mat = q_u_t isa Matrix{K} ? q_u_t : Matrix{K}(q_u_t)
        is_selector_u = selector_is_std[u]
        rows_u = selector_rows[u]

        if du == 0
            for j in eachindex(su)
                v = su[j]
                Auv = _zero_store_map(OutMatT, K, Cdims[v], 0)
                outu[j] = Auv
                maps_from_pred[v][pred_slot_of_succ[u][j]] = Auv
            end
            continue
        end

        total_cols = 0
        for v in su
            total_cols += Cdims[v]
        end
        if total_cols == 0
            for j in eachindex(su)
                v = su[j]
                Auv = _zero_store_map(OutMatT, K, 0, du)
                outu[j] = Auv
                maps_from_pred[v][pred_slot_of_succ[u][j]] = Auv
            end
            continue
        end

        if is_selector_u
            for j in eachindex(su)
                v = su[j]
                cv = Cdims[v]
                Auv = if cv == 0
                    _zero_store_map(OutMatT, K, 0, du)
                else
                    Buv = B.edge_maps.maps_to_succ[u][j]
                    Cuv = C.edge_maps.maps_to_succ[u][j]
                    raw = if selector_is_std[v]
                        _blockdiag_extract(K, Buv, Cuv, selector_rows[v], rows_u, bu, B.dims[v])
                    else
                        qcomps[v] * _blockdiag_selected_cols(K, Buv, Cuv, rows_u, bu)
                    end
                    _to_store_map(OutMatT, raw)
                end
                outu[j] = Auv
                maps_from_pred[v][pred_slot_of_succ[u][j]] = Auv
            end
            continue
        end

        use_batch = (length(su) >= COKERNEL_BATCH_MIN_FANOUT) &&
                    (total_cols >= COKERNEL_BATCH_MIN_TOTAL_RHS_COLS)
        qq_factor = (field_is_qq && length(su) > 1) ? FieldLinAlg._factor_fullcolumnQQ(q_u_t_mat) : nothing

        if !use_batch
            max_cv = 0
            for v in su
                max_cv = max(max_cv, Cdims[v])
            end
            rhs_buf = Matrix{K}(undef, bu + cu, max_cv)
            for j in eachindex(su)
                v = su[j]
                cv = Cdims[v]
                Auv = if cv == 0
                    _zero_store_map(OutMatT, K, 0, du)
                else
                    rhs = view(rhs_buf, :, 1:cv)
                    if selector_is_std[v]
                        rhs .= _blockdiag_selected_cols(K,
                                                        transpose(B.edge_maps.maps_to_succ[u][j]),
                                                        transpose(C.edge_maps.maps_to_succ[u][j]),
                                                        selector_rows[v],
                                                        B.dims[v])
                    else
                        _mul_pushout_rhs!(rhs,
                                          B.edge_maps.maps_to_succ[u][j],
                                          C.edge_maps.maps_to_succ[u][j],
                                          qtrans[v],
                                          bu,
                                          B.dims[v])
                    end
                    X = if qq_factor === nothing
                        FieldLinAlg.solve_fullcolumn(field, q_u_t_mat, rhs;
                                                     check_rhs=!_FAST_SOLVE_NO_CHECK[],
                                                     cache=false)
                    else
                        FieldLinAlg.solve_fullcolumn(field, q_u_t_mat, rhs;
                                                     check_rhs=!_FAST_SOLVE_NO_CHECK[],
                                                     cache=false, factor=qq_factor)
                    end
                    _to_store_map(OutMatT, copy(transpose(X)))
                end
                outu[j] = Auv
                maps_from_pred[v][pred_slot_of_succ[u][j]] = Auv
            end
            continue
        end

        offsets = Vector{Int}(undef, length(su) + 1)
        offsets[1] = 1
        for j in eachindex(su)
            offsets[j + 1] = offsets[j] + Cdims[su[j]]
        end
        rhs_all = Matrix{K}(undef, bu + cu, total_cols)
        for j in eachindex(su)
            v = su[j]
            cv = Cdims[v]
            cv == 0 && continue
            c0 = offsets[j]
            c1 = offsets[j + 1] - 1
            if selector_is_std[v]
                view(rhs_all, :, c0:c1) .= _blockdiag_selected_cols(K,
                                                                    transpose(B.edge_maps.maps_to_succ[u][j]),
                                                                    transpose(C.edge_maps.maps_to_succ[u][j]),
                                                                    selector_rows[v],
                                                                    B.dims[v])
            else
                _mul_pushout_rhs!(view(rhs_all, :, c0:c1),
                                  B.edge_maps.maps_to_succ[u][j],
                                  C.edge_maps.maps_to_succ[u][j],
                                  qtrans[v],
                                  bu,
                                  B.dims[v])
            end
        end
        Xall = if qq_factor === nothing
            FieldLinAlg.solve_fullcolumn(field, q_u_t_mat, rhs_all;
                                         check_rhs=!_FAST_SOLVE_NO_CHECK[],
                                         cache=false)
        else
            FieldLinAlg.solve_fullcolumn(field, q_u_t_mat, rhs_all;
                                         check_rhs=!_FAST_SOLVE_NO_CHECK[],
                                         cache=false, factor=qq_factor)
        end
        for j in eachindex(su)
            v = su[j]
            cv = Cdims[v]
            Auv = if cv == 0
                _zero_store_map(OutMatT, K, 0, du)
            else
                c0 = offsets[j]
                c1 = offsets[j + 1] - 1
                _to_store_map(OutMatT, copy(transpose(view(Xall, :, c0:c1))))
            end
            outu[j] = Auv
            maps_from_pred[v][pred_slot_of_succ[u][j]] = Auv
        end
    end

    storeC = CoverEdgeMapStore{K,OutMatT}(preds, succs, maps_from_pred, maps_to_succ, cc.nedges)
    Cmod = PModule{K}(Q, Cdims, storeC; field=field)
    q = PMorphism(S, Cmod, qcomps)
    phi = PMorphism{K}(A, S, phi_comps)
    return S, Cmod, q, phi
end

"""
    _pullback_kernel_with_inclusion(f, g, cc) -> (S, P, iota, psi)

Internal pullback builder via kernels.

Returns:
- `S = dom(f) oplus dom(g)`,
- `P`: the pullback object,
- `iota : P -> S`: the kernel inclusion,
- `psi : S -> cod(f)`: the defining map whose kernel is the pullback.

Higher-level callers typically use `_pullback_public` or `pullback`.
"""
function _pullback_kernel_with_inclusion(
    f::PMorphism{K},
    g::PMorphism{K},
    cc::CoverCache,
) where {K}
    B = f.dom
    C = g.dom
    D = f.cod
    Q = B.Q
    n = nvertices(Q)
    field = B.field

    psi_comps = Vector{Matrix{K}}(undef, n)
    basisK = Vector{Matrix{K}}(undef, n)
    K_dims = zeros(Int, n)
    @inbounds for i in 1:n
        psi_i = _pullback_psi_component(K, f.comps[i], g.comps[i])
        psi_comps[i] = psi_i
        Bi = if _is_partial_permutation(field, psi_i)
            _kernel_basis_partial_permutation(K, psi_i)
        else
            FieldLinAlg.nullspace(field, psi_i)
        end
        basisK[i] = Bi
        K_dims[i] = size(Bi, 2)
    end

    S = direct_sum(B, C)
    preds = [collect(_preds(cc, v)) for v in 1:n]
    succs = [collect(_succs(cc, u)) for u in 1:n]
    pred_slot_of_succ = [collect(_pred_slots_of_succ(cc, u)) for u in 1:n]
    selector_rows = Vector{Vector{Int}}(undef, n)
    selector_is_std = falses(n)
    empty_selector = Int[]
    for i in 1:n
        rows = _selector_rows_from_basis(field, basisK[i])
        if rows === nothing
            selector_rows[i] = empty_selector
        else
            selector_rows[i] = rows
            selector_is_std[i] = true
        end
    end
    OutMatT = _store_out_type(K, _module_prefers_sparse_store(S))
    indegree = [length(preds[v]) for v in 1:n]
    field_is_qq = field isa QQField
    qq_basis_factors = Vector{Any}(undef, n)
    field_is_qq && fill!(qq_basis_factors, nothing)

    maps_from_pred = [Vector{OutMatT}(undef, length(preds[v])) for v in 1:n]
    maps_to_succ = [Vector{OutMatT}(undef, length(succs[u])) for u in 1:n]

    @inbounds for u in 1:n
        su = succs[u]
        outu = maps_to_succ[u]
        du = K_dims[u]
        bu = B.dims[u]
        Ku = basisK[u]
        is_selector_u = selector_is_std[u]
        rows_u = selector_rows[u]

        if du == 0
            for j in eachindex(su)
                v = su[j]
                X = _zero_store_map(OutMatT, K, K_dims[v], 0)
                outu[j] = X
                maps_from_pred[v][pred_slot_of_succ[u][j]] = X
            end
            continue
        end

        max_rows = 0
        for v in su
            max_rows = max(max_rows, B.dims[v] + C.dims[v])
        end
        im_buf = Matrix{K}(undef, max_rows, du)
        for j in eachindex(su)
            v = su[j]
            Auv = if K_dims[v] == 0
                _zero_store_map(OutMatT, K, 0, du)
            else
                rhs = if is_selector_u
                    _blockdiag_selected_cols(K,
                                             B.edge_maps.maps_to_succ[u][j],
                                             C.edge_maps.maps_to_succ[u][j],
                                             rows_u,
                                             bu)
                else
                    buf = view(im_buf, 1:(B.dims[v] + C.dims[v]), 1:du)
                    _mul_pullback_im!(buf,
                                      B.edge_maps.maps_to_succ[u][j],
                                      C.edge_maps.maps_to_succ[u][j],
                                      Ku,
                                      bu,
                                      B.dims[v])
                end
                Xraw = if selector_is_std[v]
                    rhs[selector_rows[v], :]
                elseif indegree[v] <= 1
                    FieldLinAlg.solve_fullcolumn(field, basisK[v], rhs;
                                                 check_rhs=false,
                                                 cache=false)
                elseif field_is_qq
                    fac = qq_basis_factors[v]
                    if fac === nothing
                        fac = FieldLinAlg._factor_fullcolumnQQ(basisK[v])
                        qq_basis_factors[v] = fac
                    end
                    FieldLinAlg.solve_fullcolumn(field, basisK[v], rhs;
                                                 check_rhs=false,
                                                 cache=false, factor=fac)
                else
                    FieldLinAlg.solve_fullcolumn(field, basisK[v], rhs;
                                                 check_rhs=false)
                end
                _to_store_map(OutMatT, Xraw)
            end
            outu[j] = Auv
            maps_from_pred[v][pred_slot_of_succ[u][j]] = Auv
        end
    end

    storeK = CoverEdgeMapStore{K,OutMatT}(preds, succs, maps_from_pred, maps_to_succ, cc.nedges)
    Kmod = PModule{K}(Q, K_dims, storeK; field=field)
    iota = PMorphism{K}(Kmod, S, basisK)
    psi = PMorphism{K}(S, D, psi_comps)
    return S, Kmod, iota, psi
end

"""
    pushout(f, g; cache=:auto) -> (P, inB, inC, q, phi)

Compute the pushout of a span A --f--> B and A --g--> C.

Construction:
    P = (B oplus C) / im( (f, -g) : A -> B oplus C )

Returns:
- P : the pushout module
- inB : B -> P
- inC : C -> P
- q : (B oplus C) -> P (the quotient map)
- phi : A -> (B oplus C) (the map whose cokernel defines the pushout)

The maps satisfy inB o f == inC o g.

For most users, leave `cache=:auto`. If you are building many pushouts on the same
poset in a low-level loop, call `build_cache!(Q)` once and pass `cache=cc`.

Best practice:
- prefer `pushout` over `colimit(SpanDiagram(...))` unless you are explicitly
  coding at the diagram level,
- use the returned `inB` and `inC` as the canonical structure maps into the
  pushout object.
"""
function pushout(f::PMorphism{K}, g::PMorphism{K}; cache::Union{Symbol,CoverCache}=:auto) where {K}
    return _pushout_public(f, g, cache)
end

# ---------------------------------------------------------------------------
# Diagram objects + limit/colimit dispatch (small but useful categorical layer)
# ---------------------------------------------------------------------------

"""
Abstract supertype for small diagram objects valued in `PModule{K}`.

This is intentionally minimal: it supports the common finite shapes needed in
everyday categorical algebra:
- discrete pair (product/coproduct)
- parallel pair (equalizer/coequalizer)
- span (pushout)
- cospan (pullback)

This layer is intended for advanced/category-theoretic workflows. For ordinary use,
prefer the concrete constructions `product`, `coproduct`, `equalizer`, `coequalizer`,
`pushout`, and `pullback`.
"""
abstract type AbstractDiagram{K} end

"""
    DiscretePairDiagram(A, B)

The discrete diagram on two objects `A` and `B` (no arrows).

Advanced/category-theory surface for use with `limit` and `colimit`.
"""
struct DiscretePairDiagram{K} <: AbstractDiagram{K}
    A::PModule{K}
    B::PModule{K}
end

"""
    ParallelPairDiagram(f, g)

A parallel pair of morphisms `f, g : A -> B`.

Advanced/category-theory surface for use with `limit` and `colimit`.
"""
struct ParallelPairDiagram{K} <: AbstractDiagram{K}
    f::PMorphism{K}
    g::PMorphism{K}
end

"""
    SpanDiagram(f, g)

A span-shaped diagram `A --f--> B` and `A --g--> C`.

Its colimit is the pushout.

Advanced/category-theory surface for use with `colimit`.
"""
struct SpanDiagram{K} <: AbstractDiagram{K}
    f::PMorphism{K}  # A -> B
    g::PMorphism{K}  # A -> C
end

"""
    CospanDiagram(f, g)

A cospan-shaped diagram `B --f--> D` and `C --g--> D`.

Its limit is the pullback.

Advanced/category-theory surface for use with `limit`.
"""
struct CospanDiagram{K} <: AbstractDiagram{K}
    f::PMorphism{K}  # B -> D
    g::PMorphism{K}  # C -> D
end

"""
    limit(D::DiscretePairDiagram; cache=:auto) -> (P, pA, pB)

Compute the categorical product of the two objects in a discrete pair diagram.

Returns the product object and its two canonical projections. The `cache`
keyword is accepted for API uniformity but is not used in this shape.
"""
@inline function limit(D::DiscretePairDiagram{K}; cache::Union{Symbol,CoverCache}=:auto) where {K}
    return _product_public(D.A, D.B)
end

"""
    limit(D::ParallelPairDiagram; cache=:auto) -> (E, e)

Compute the limit of a parallel pair, i.e. the equalizer of the two maps.

Returns the equalizer object together with its inclusion into the common domain.
Prefer `equalizer(f, g)` in ordinary code; this overload is mainly for
advanced/category-theory workflows.
"""
@inline function limit(D::ParallelPairDiagram{K}; cache::Union{Symbol,CoverCache}=:auto) where {K}
    return _equalizer_public(D.f, D.g, cache)
end

"""
    limit(D::CospanDiagram; cache=:auto) -> (P, prB, prC, iota, psi)

Compute the limit of a cospan, i.e. the pullback of the two maps.

Returns the same tuple as `pullback(f, g)`. Prefer `pullback` directly unless
you are intentionally working in the diagram API.
"""
@inline function limit(D::CospanDiagram{K}; cache::Union{Symbol,CoverCache}=:auto) where {K}
    return _pullback_public(D.f, D.g, cache)
end

"""
    colimit(D::DiscretePairDiagram; cache=:auto) -> (S, iA, iB)

Compute the categorical coproduct of the two objects in a discrete pair
diagram.

Returns the coproduct object and its canonical injections. The `cache` keyword
is accepted for API uniformity but is not used in this shape.
"""
@inline function colimit(D::DiscretePairDiagram{K}; cache::Union{Symbol,CoverCache}=:auto) where {K}
    return _coproduct_public(D.A, D.B)
end

"""
    colimit(D::ParallelPairDiagram; cache=:auto) -> (Q, q)

Compute the colimit of a parallel pair, i.e. the coequalizer of the two maps.

Returns the coequalizer object together with its quotient map. Prefer
`coequalizer(f, g)` in ordinary code.
"""
@inline function colimit(D::ParallelPairDiagram{K}; cache::Union{Symbol,CoverCache}=:auto) where {K}
    return _coequalizer_public(D.f, D.g, cache)
end

"""
    colimit(D::SpanDiagram; cache=:auto) -> (P, inB, inC, q, phi)

Compute the colimit of a span, i.e. the pushout of the two maps.

Returns the same tuple as `pushout(f, g)`. Prefer `pushout` directly unless you
are intentionally working in the diagram API.
"""
@inline function colimit(D::SpanDiagram{K}; cache::Union{Symbol,CoverCache}=:auto) where {K}
    return _pushout_public(D.f, D.g, cache)
end

"""
    pullback(f, g; cache=:auto) -> (P, prB, prC, iota, psi)

Compute the pullback of a cospan B --f--> D and C --g--> D.

Construction:
    P = ker( (f, -g) : B oplus C -> D )

Returns:
- P : the pullback module
- prB : P -> B
- prC : P -> C
- iota : P -> (B oplus C) (the kernel inclusion)
- psi : (B oplus C) -> D (the map whose kernel defines the pullback)

The projections satisfy f o prB == g o prC.

For most users, leave `cache=:auto`. If you are building many pullbacks on the same
poset in a low-level loop, call `build_cache!(Q)` once and pass `cache=cc`.

Best practice:
- prefer `pullback` over `limit(CospanDiagram(...))` unless you are explicitly
  coding at the diagram level,
- use the returned `prB` and `prC` as the canonical projections.
"""
function pullback(f::PMorphism{K}, g::PMorphism{K}; cache::Union{Symbol,CoverCache}=:auto) where {K}
    return _pullback_public(f, g, cache)
end

# -----------------------------------------------------------------------------
# Exactness utilities: short exact sequences and snake lemma
# -----------------------------------------------------------------------------

"""
    ShortExactSequence

Concrete container type for a short exact sequence

    0 -> A --i--> B --p--> C -> 0.

Prefer constructing user-facing values with `short_exact_sequence(i, p; ...)`.
Use `ShortExactSequence(...)` directly when you explicitly want the concrete container type.

If `check=true` (default), we verify:
- p o i = 0
- i is a monomorphism (pointwise injective)
- p is an epimorphism (pointwise surjective)
- im(i) = ker(p) as submodules of B (pointwise equality)

The check is exact for exact fields and tolerance-based for RealField.

This object caches `ker(p)` and `im(i)` once computed, since many downstream
constructions (Ext/Tor LES, snake lemma, etc.) use them repeatedly.

Recommended cache usage:
- normal use: keep `cache=:auto`
- repeated low-level exactness checks on one poset: call `build_cache!(Q)` first and reuse `cache=cc`

For user-facing inspection, prefer:
- `describe(ses)` for a structured summary,
- `dimensions(ses)` for the size data,
- `exactness_summary(ses)` for the cheap cached exactness state,
- `check_short_exact_sequence(ses)` when you want an explicit validation report.
"""
mutable struct ShortExactSequence{K}
    A::PModule{K}
    B::PModule{K}
    C::PModule{K}
    i::PMorphism{K}
    p::PMorphism{K}
    checked::Bool
    exact::Bool
    ker_p::Union{Nothing,Tuple{PModule{K},PMorphism{K}}}
    img_i::Union{Nothing,Tuple{PModule{K},PMorphism{K}}}
end

"""
    ShortExactSequence(i, p; check=true, cache=:auto) -> ShortExactSequence

Construct the concrete `ShortExactSequence` container from two composable maps
`i : A -> B` and `p : B -> C`.

Returns a `ShortExactSequence` object storing the three modules, the two maps,
and lazy caches for `ker(p)` and `im(i)`.

Best practice:
- prefer `short_exact_sequence(i, p; ...)` in user-facing code,
- call this constructor directly only when you explicitly want the concrete type.
"""
function ShortExactSequence(i::PMorphism{K}, p::PMorphism{K};
                           check::Bool=true,
                           cache::Union{Symbol,CoverCache}=:auto) where {K}
    @assert i.cod === p.dom
    ses = ShortExactSequence{K}(i.dom, i.cod, p.cod, i, p, false, false, nothing, nothing)
    if check
        ok = is_exact(ses; cache=cache)
        ok || error("ShortExactSequence: maps do not form a short exact sequence")
    end
    return ses
end

"""
    short_exact_sequence(i, p; check=true, cache=:auto) -> ShortExactSequence

Canonical public constructor for a short exact sequence container.

Returns a validated `ShortExactSequence` when `check=true`, or an unchecked
container when `check=false`.

Best practice:
- use this in examples and user-facing code,
- set `check=false` only when the exactness has already been established
  elsewhere and you want to avoid duplicate work.
"""
@inline short_exact_sequence(i::PMorphism{K}, p::PMorphism{K};
                             check::Bool=true,
                             cache::Union{Symbol,CoverCache}=:auto) where {K} =
    ShortExactSequence(i, p; check=check, cache=cache)

"""
    inclusion_map(ses::ShortExactSequence) -> PMorphism

Return the left map `i : A -> B` stored in the short exact sequence

`0 -> A -i-> B -p-> C -> 0`.

Best practice:
- use this accessor rather than `ses.i` when passing the sequence back into
  categorical code.
"""
@inline inclusion_map(ses::ShortExactSequence) = ses.i

"""
    projection_map(ses::ShortExactSequence) -> PMorphism

Return the right map `p : B -> C` stored in the short exact sequence

`0 -> A -i-> B -p-> C -> 0`.

Best practice:
- use this accessor rather than `ses.p` when you want the canonical quotient map
  from the stored sequence.
"""
@inline projection_map(ses::ShortExactSequence) = ses.p

"""
    exactness_summary(ses::ShortExactSequence) -> NamedTuple

Return a cheap summary of the current exactness state of `ses`.

This helper never recomputes exactness. It only reports:
- whether exactness has already been checked,
- the cached exactness value (or `nothing` if unchecked),
- whether `ker(p)` and `im(i)` have been cached.

Use this when you want a quick status report before deciding whether to call
`is_exact(ses; ...)` or inspect the full sequence more closely.
"""
@inline exactness_summary(ses::ShortExactSequence) = (
    checked = ses.checked,
    exact = ses.checked ? ses.exact : nothing,
    ker_cached = ses.ker_p !== nothing,
    img_cached = ses.img_i !== nothing,
)

"""
    dimensions(ses::ShortExactSequence) -> NamedTuple

Return a compact dimension summary for a `ShortExactSequence`.

The summary records the number of vertices, the stalk dimensions of `A`, `B`,
and `C`, their total dimensions, and the current cached exactness state.
"""
@inline _short_exact_sequence_dimensions(ses::ShortExactSequence) = (
    vertices = nvertices(ses.B.Q),
    A = copy(ses.A.dims),
    B = copy(ses.B.dims),
    C = copy(ses.C.dims),
    A_total = sum(ses.A.dims),
    B_total = sum(ses.B.dims),
    C_total = sum(ses.C.dims),
    checked = ses.checked,
    exact = ses.checked ? ses.exact : nothing,
)

"""
    describe(ses::ShortExactSequence) -> NamedTuple

Return a compact mathematical summary of a `ShortExactSequence`.

This is the structured companion to the pretty-printer and is intended for
notebooks and programmatic inspection.
"""
@inline _short_exact_sequence_describe(ses::ShortExactSequence{K}) where {K} = (
    kind = :short_exact_sequence,
    field = _scalar_name(K),
    dimensions = _short_exact_sequence_dimensions(ses),
    ker_cached = ses.ker_p !== nothing,
    img_cached = ses.img_i !== nothing,
)

"""
    check_short_exact_sequence(ses; throw=false, cache=:auto) -> NamedTuple

Validate a `ShortExactSequence` container.

The returned report has fields:
- `kind = :short_exact_sequence`
- `valid::Bool`
- `issues::Vector{String}`

What is checked
- structural validity of `A`, `B`, `C`,
- structural validity of the maps `i` and `p`,
- composability `i.cod === p.dom`,
- vanishing of `p circ i`,
- exactness via `is_exact(ses; cache=...)`.

Best practices
- use this on hand-built short exact sequence containers,
- leave `cache=:auto` in ordinary code,
- use `throw=true` in tests when any failure should stop execution immediately.
"""
function check_short_exact_sequence(ses::ShortExactSequence{K};
                                    throw::Bool=false,
                                    cache::Union{Symbol,CoverCache}=:auto) where {K}
    issues = String[]
    repA = Modules.check_module(ses.A)
    repB = Modules.check_module(ses.B)
    repC = Modules.check_module(ses.C)
    repi = Modules.check_morphism(ses.i)
    repp = Modules.check_morphism(ses.p)
    repA.valid || append!(issues, "module A: " .* repA.issues)
    repB.valid || append!(issues, "module B: " .* repB.issues)
    repC.valid || append!(issues, "module C: " .* repC.issues)
    repi.valid || append!(issues, "map i: " .* repi.issues)
    repp.valid || append!(issues, "map p: " .* repp.issues)
    ses.i.cod === ses.p.dom || push!(issues, "expected i.cod === p.dom so that the maps are composable")
    if ses.i.cod === ses.p.dom
        _zero_composition(ses.A.field, ses.p, ses.i) ||
            push!(issues, "expected the composite p circ i to be the zero morphism")
    end
    is_exact(ses; cache=cache) || push!(issues, "expected the stored sequence to be exact")
    report = (
        kind = :short_exact_sequence,
        valid = isempty(issues),
        issues = issues,
        dimensions = _short_exact_sequence_dimensions(ses),
    )
    throw && !report.valid && _abelian_validation_error("check_short_exact_sequence", issues)
    return report
end

"""
    is_exact(ses; cache=:auto) -> Bool

Check whether the stored maps define a short exact sequence.

Returns `true` exactly when:
- `p circ i = 0`,
- `i` is pointwise injective,
- `p` is pointwise surjective,
- `im(i) = ker(p)` at every vertex.

The result is memoized inside `ses`, so repeated calls are cheap after the first
successful or failed check.
"""
function is_exact(ses::ShortExactSequence{K}; cache::Union{Symbol,CoverCache}=:auto) where {K}
    if ses.checked
        return ses.exact
    end

    i = ses.i
    p = ses.p
    A = ses.A
    B = ses.B
    C = ses.C
    @assert i.dom === A
    @assert i.cod === B
    @assert p.dom === B
    @assert p.cod === C
    cc = _resolve_public_cover_cache(B.Q, B.field, cache)

    # p o i = 0
    field = B.field
    for u in 1:nvertices(B.Q)
        comp = p.comps[u] * i.comps[u]
        _is_zero_matrix(field, comp) || (ses.checked = true; ses.exact = false; return false)
    end

    # i mono, p epi
    if !_is_monomorphism(i) || !_is_epimorphism(p)
        ses.checked = true
        ses.exact = false
        return false
    end

    # Compute and cache ker(p) and im(i) as submodules of B.
    if ses.ker_p === nothing
        ses.ker_p = _kernel_with_inclusion_cached(p, cc)
    end
    if ses.img_i === nothing
        ses.img_i = _image_with_inclusion_cached(i, cc)
    end
    (Kmod, incK) = ses.ker_p
    (Im, incIm) = ses.img_i

    # Compare subspaces at each vertex using ranks.
    Q = B.Q
    for u in 1:nvertices(Q)
        Au = incK.comps[u]
        Bu = incIm.comps[u]
        rA = FieldLinAlg.rank(B.field, Au)
        rB = FieldLinAlg.rank(B.field, Bu)
        if rA != rB
            ses.checked = true
            ses.exact = false
            return false
        end
        # span(Au,Bu) must have same dimension if they are equal.
        rAB = FieldLinAlg.rank(B.field, hcat(Au, Bu))
        if rAB != rA
            ses.checked = true
            ses.exact = false
            return false
        end
    end

    ses.checked = true
    ses.exact = true
    return true
end

"""
    _assert_exact(ses; cache=:auto) -> Nothing

Internal exactness guard.

Throws an error if `ses` is not exact and otherwise returns `nothing`. This is
used by internal routines such as `snake_lemma` when `check=true`.
"""
function _assert_exact(ses::ShortExactSequence{K}; cache::Union{Symbol,CoverCache}=:auto) where {K}
    is_exact(ses; cache=cache) || error("ShortExactSequence: sequence is not exact")
    return nothing
end

"""
    _induced_map_to_kernel(g, kA, kB) -> PMorphism

Build the morphism induced by `g` on kernels.

Inputs:
- `g : A -> B`,
- `kA : ker(fA) -> A`,
- `kB : ker(fB) -> B`.

Returns the unique morphism `ker(fA) -> ker(fB)` making the evident square
commute.
"""
function _induced_map_to_kernel(g::PMorphism{K},
                                 kA::PMorphism{K},
                                 kB::PMorphism{K}) where {K}
    Q = g.dom.Q
    @assert kA.cod === g.dom
    @assert g.cod === kB.cod
    Kdom = kA.dom
    Kcod = kB.dom
    comps = Vector{Matrix{K}}(undef, nvertices(Q))
    for u in 1:nvertices(Q)
        rhs = g.comps[u] * kA.comps[u]  # B_u x dim kerA_u
        comps[u] = FieldLinAlg.solve_fullcolumn(g.dom.field, kB.comps[u], rhs)  # dim kerB_u x dim kerA_u
    end
    return PMorphism{K}(Kdom, Kcod, comps)
end

"""
    _induced_map_from_cokernel(h, qA, qB) -> PMorphism

Build the morphism induced by `h` on cokernels.

Inputs:
- `h : A -> B`,
- `qA : A -> cokerA`,
- `qB : B -> cokerB`.

Returns the unique morphism `cokerA -> cokerB` compatible with the quotient
projections.
"""
function _induced_map_from_cokernel(h::PMorphism{K},
                                    qA::PMorphism{K},
                                    qB::PMorphism{K}) where {K}
    Q = h.dom.Q
    @assert qA.dom === h.dom
    @assert h.cod === qB.dom
    Cdom = qA.cod
    Ccod = qB.cod
    comps = Vector{Matrix{K}}(undef, nvertices(Q))
    for u in 1:nvertices(Q)
        Qsrc = qA.comps[u]
        rinv = _right_inverse_full_row(h.dom.field, Qsrc)
        rhs = qB.comps[u] * h.comps[u]  # cokerB_u x A_u
        comps[u] = rhs * rinv           # cokerB_u x cokerA_u
    end
    return PMorphism{K}(Cdom, Ccod, comps)
end

"""
    SnakeLemmaResult

Result of `snake_lemma`: it packages the six objects and five maps in the snake lemma
exact sequence, including the connecting morphism delta.

The long exact sequence has the form:

    ker(fA) -> ker(fB) -> ker(fC) --delta--> coker(fA) -> coker(fB) -> coker(fC)

All maps are returned as actual morphisms of P-modules (natural transformations).

For user-facing inspection, prefer:
- `describe(sn)` for a structured overview,
- `dimensions(sn)` for the size data,
- `kernel_objects(sn)` / `cokernel_objects(sn)` for the six stored objects,
- `kernel_inclusions(sn)` / `cokernel_projections(sn)` for the associated
  universal maps,
- `connecting_map(sn)` for the snake-lemma connecting morphism,
- `check_snake_lemma(sn)` for an explicit validation report.
"""
struct SnakeLemmaResult{K}
    kerA::Tuple{PModule{K},PMorphism{K}}   # (KerA, inclusion KerA -> A)
    kerB::Tuple{PModule{K},PMorphism{K}}
    kerC::Tuple{PModule{K},PMorphism{K}}
    cokA::Tuple{PModule{K},PMorphism{K}}   # (CokA, projection A' -> CokA) where A' is cod(fA)
    cokB::Tuple{PModule{K},PMorphism{K}}
    cokC::Tuple{PModule{K},PMorphism{K}}
    k1::PMorphism{K}                       # ker(fA) -> ker(fB)
    k2::PMorphism{K}                       # ker(fB) -> ker(fC)
    delta::PMorphism{K}                    # ker(fC) -> coker(fA)
    c1::PMorphism{K}                       # coker(fA) -> coker(fB)
    c2::PMorphism{K}                       # coker(fB) -> coker(fC)
end

"""
    connecting_map(sn::SnakeLemmaResult) -> PMorphism

Return the connecting morphism `delta : ker(fC) -> coker(fA)` in the snake
lemma exact sequence.
"""
@inline connecting_map(sn::SnakeLemmaResult) = sn.delta

"""
    kernel_objects(sn::SnakeLemmaResult) -> NamedTuple

Return the three kernel objects in the snake-lemma sequence as a named tuple
`(A=..., B=..., C=...)`.

This avoids unpacking the stored `(object, map)` tuples manually.
"""
@inline kernel_objects(sn::SnakeLemmaResult) = (A = sn.kerA[1], B = sn.kerB[1], C = sn.kerC[1])

"""
    kernel_inclusions(sn::SnakeLemmaResult) -> NamedTuple

Return the three kernel inclusions in the snake-lemma sequence as a named tuple
`(A=..., B=..., C=...)`.
"""
@inline kernel_inclusions(sn::SnakeLemmaResult) = (A = sn.kerA[2], B = sn.kerB[2], C = sn.kerC[2])

"""
    cokernel_objects(sn::SnakeLemmaResult) -> NamedTuple

Return the three cokernel objects in the snake-lemma sequence as a named tuple
`(A=..., B=..., C=...)`.
"""
@inline cokernel_objects(sn::SnakeLemmaResult) = (A = sn.cokA[1], B = sn.cokB[1], C = sn.cokC[1])

"""
    cokernel_projections(sn::SnakeLemmaResult) -> NamedTuple

Return the three cokernel projections in the snake-lemma sequence as a named
tuple `(A=..., B=..., C=...)`.
"""
@inline cokernel_projections(sn::SnakeLemmaResult) = (A = sn.cokA[2], B = sn.cokB[2], C = sn.cokC[2])

"""
    dimensions(sn::SnakeLemmaResult) -> NamedTuple

Return a compact dimension summary of the snake-lemma exact sequence.

The summary records the stalk dimensions of the three kernel objects and the
three cokernel objects that appear in the sequence.
"""
@inline _snake_lemma_dimensions(sn::SnakeLemmaResult) = (
    vertices = nvertices(sn.kerA[1].Q),
    kerA = copy(sn.kerA[1].dims),
    kerB = copy(sn.kerB[1].dims),
    kerC = copy(sn.kerC[1].dims),
    cokA = copy(sn.cokA[1].dims),
    cokB = copy(sn.cokB[1].dims),
    cokC = copy(sn.cokC[1].dims),
    kerA_total = sum(sn.kerA[1].dims),
    kerB_total = sum(sn.kerB[1].dims),
    kerC_total = sum(sn.kerC[1].dims),
    cokA_total = sum(sn.cokA[1].dims),
    cokB_total = sum(sn.cokB[1].dims),
    cokC_total = sum(sn.cokC[1].dims),
)

"""
    describe(sn::SnakeLemmaResult) -> NamedTuple

Return a compact mathematical summary of a `SnakeLemmaResult`.

The summary records the scalar type, the size of the underlying poset, and the
dimension summary of the kernel/cokernel sequence.
"""
@inline _snake_lemma_describe(sn::SnakeLemmaResult{K}) where {K} = (
    kind = :snake_lemma,
    field = _scalar_name(K),
    dimensions = _snake_lemma_dimensions(sn),
)

"""
    check_snake_lemma(sn; throw=false) -> NamedTuple

Validate a `SnakeLemmaResult`.

The returned report has fields:
- `kind = :snake_lemma`
- `valid::Bool`
- `issues::Vector{String}`

What is checked
- structural validity of the stored kernel and cokernel objects,
- structural validity of the five sequence maps,
- composability of consecutive maps,
- vanishing of consecutive compositions in the snake-lemma sequence.

Best practices
- use this on hand-built or experimentally modified snake-lemma output,
- use `throw=true` in tests when any contract failure should abort execution.
"""
function check_snake_lemma(sn::SnakeLemmaResult{K}; throw::Bool=false) where {K}
    issues = String[]
    kernel_data = (sn.kerA, sn.kerB, sn.kerC)
    cokernel_data = (sn.cokA, sn.cokB, sn.cokC)
    for (label, pair) in zip((:kerA, :kerB, :kerC), kernel_data)
        M, iota = pair
        repM = Modules.check_module(M)
        repf = Modules.check_morphism(iota)
        repM.valid || append!(issues, "$(label) object: " .* repM.issues)
        repf.valid || append!(issues, "$(label) inclusion: " .* repf.issues)
    end
    for (label, pair) in zip((:cokA, :cokB, :cokC), cokernel_data)
        M, proj = pair
        repM = Modules.check_module(M)
        repf = Modules.check_morphism(proj)
        repM.valid || append!(issues, "$(label) object: " .* repM.issues)
        repf.valid || append!(issues, "$(label) projection: " .* repf.issues)
    end
    for (label, f) in zip((:k1, :k2, :delta, :c1, :c2), (sn.k1, sn.k2, sn.delta, sn.c1, sn.c2))
        rep = Modules.check_morphism(f)
        rep.valid || append!(issues, "$(label): " .* rep.issues)
    end
    consecutive = ((:k1, sn.k1, :k2, sn.k2),
                   (:k2, sn.k2, :delta, sn.delta),
                   (:delta, sn.delta, :c1, sn.c1),
                   (:c1, sn.c1, :c2, sn.c2))
    field = sn.k1.dom.field
    for (f_label, f, g_label, g) in consecutive
        f.cod === g.dom || push!(issues, "expected $(g_label) to be composable after $(f_label)")
        if f.cod === g.dom
            _zero_composition(field, g, f) ||
                push!(issues, "expected the consecutive composite $(g_label) circ $(f_label) to be zero")
        end
    end
    report = (
        kind = :snake_lemma,
        valid = isempty(issues),
        issues = issues,
        dimensions = _snake_lemma_dimensions(sn),
    )
    throw && !report.valid && _abelian_validation_error("check_snake_lemma", issues)
    return report
end

"""
    _check_commutative_square(g1, f1, g2, f2) -> Bool

Check whether the square `g1 circ f1 = g2 circ f2` commutes at every stalk.

Returns a field-aware Boolean result, using exact equality for exact fields and
tolerance-aware equality for `RealField`.
"""
function _check_commutative_square(g1::PMorphism{K}, f1::PMorphism{K},
                                   g2::PMorphism{K}, f2::PMorphism{K}) where {K}
    Q = f1.dom.Q
    @assert f1.cod === g1.dom
    @assert f2.cod === g2.dom
    @assert f1.dom === f2.dom
    @assert g1.cod === g2.cod
    field = f1.dom.field
    for u in 1:nvertices(Q)
        left = g1.comps[u] * f1.comps[u]
        right = g2.comps[u] * f2.comps[u]
        _is_zero_matrix(field, left - right) || return false
    end
    return true
end

"""
    snake_lemma(top, bottom, fA, fB, fC; check=true, cache=:auto) -> SnakeLemmaResult

Compute the maps and objects in the snake lemma exact sequence for a commutative diagram
with exact rows:

    0 -> A  --i-->  B  --p-->  C  -> 0
          |        |        |
         fA       fB       fC
          |        |        |
    0 -> A' --i'->  B' --p'-> C' -> 0

Inputs:
- top    : ShortExactSequence for the top row (A,B,C,i,p)
- bottom : ShortExactSequence for the bottom row (A',B',C',i',p')
- fA, fB, fC : vertical morphisms A->A', B->B', C->C'

If `check=true`, we verify that both rows are exact and that the two squares commute.

The connecting morphism `delta : ker(fC) -> coker(fA)` is computed explicitly using
linear algebra in each stalk.

This is the canonical public snake-lemma entrypoint.

Recommended cache usage:
- normal use: keep `cache=:auto`
- repeated low-level snake-lemma calls on one poset: call `build_cache!(Q)` first and reuse `cache=cc`
"""
function snake_lemma(top::ShortExactSequence{K},
                     bottom::ShortExactSequence{K},
                     fA::PMorphism{K},
                     fB::PMorphism{K},
                     fC::PMorphism{K};
                     check::Bool=true,
                     cache::Union{Symbol,CoverCache}=:auto) where {K}
    cc_top = _resolve_public_cover_cache(top.B.Q, top.B.field, cache)
    cc_bottom = bottom.B.Q === top.B.Q ? cc_top : _resolve_public_cover_cache(bottom.B.Q, bottom.B.field, cache)
    ccA = fA.dom.Q === top.B.Q ? cc_top : _resolve_public_cover_cache(fA.dom.Q, fA.dom.field, cache)
    ccB = fB.dom.Q === top.B.Q ? cc_top : _resolve_public_cover_cache(fB.dom.Q, fB.dom.field, cache)
    ccC = fC.dom.Q === top.B.Q ? cc_top : _resolve_public_cover_cache(fC.dom.Q, fC.dom.field, cache)

    if check
        _assert_exact(top; cache=cc_top)
        _assert_exact(bottom; cache=cc_bottom)

        # Squares must commute:
        # fB o i = i' o fA
        ok1 = _check_commutative_square(fB, top.i, bottom.i, fA)
        ok1 || error("snake_lemma: left square does not commute")
        # p' o fB = fC o p
        ok2 = _check_commutative_square(bottom.p, fB, fC, top.p)
        ok2 || error("snake_lemma: right square does not commute")
    end

    # Kernels of vertical maps
    kerA = _kernel_with_inclusion_cached(fA, ccA)
    kerB = _kernel_with_inclusion_cached(fB, ccB)
    kerC = _kernel_with_inclusion_cached(fC, ccC)

    # Cokernels of vertical maps
    cokA = _cokernel_module_cached(fA, ccA)
    cokB = _cokernel_module_cached(fB, ccB)
    cokC = _cokernel_module_cached(fC, ccC)

    (KerA, incKerA) = kerA
    (KerB, incKerB) = kerB
    (KerC, incKerC) = kerC

    (CokA, qA) = cokA
    (CokB, qB) = cokB
    (CokC, qC) = cokC

    Q = top.A.Q

    # Induced maps on kernels: ker(fA) -> ker(fB) -> ker(fC)
    # k1 : KerA -> KerB induced by top.i : A -> B
    k1 = _induced_map_to_kernel(top.i, incKerA, incKerB)

    # k2 : KerB -> KerC induced by top.p : B -> C
    k2 = _induced_map_to_kernel(top.p, incKerB, incKerC)

    # Induced maps on cokernels: coker(fA) -> coker(fB) -> coker(fC)
    # c1 induced by bottom.i : A' -> B'
    c1 = _induced_map_from_cokernel(bottom.i, qA, qB)

    # c2 induced by bottom.p : B' -> C'
    c2 = _induced_map_from_cokernel(bottom.p, qB, qC)

    # Connecting morphism delta : ker(fC) -> coker(fA)
    delta_comps = Vector{Matrix{K}}(undef, nvertices(Q))
    for u in 1:nvertices(Q)
        kdim = KerC.dims[u]
        if kdim == 0 || CokA.dims[u] == 0
            delta_comps[u] = zeros(K, CokA.dims[u], kdim)
            continue
        end

        # Kc is C_u x kdim, columns are a basis of ker(fC_u) (embedded into C_u).
        Kc = incKerC.comps[u]

        # Lift basis elements in C_u to B_u via p : B -> C (top row).
        # Since p_u is surjective in a short exact sequence, we can use a right inverse.
        rinv_p = _right_inverse_full_row(top.p.dom.field, top.p.comps[u])  # B_u x C_u
        B_lift = rinv_p * Kc  # B_u x kdim

        # Apply fB to get elements in B'_u.
        Bp = fB.comps[u] * B_lift  # B'_u x kdim

        # Since Kc is in ker(fC), commutativity implies Bp is in ker(p') = im(i').
        Ap = FieldLinAlg.solve_fullcolumn(bottom.i.dom.field, bottom.i.comps[u], Bp)  # A'_u x kdim

        # Project to coker(fA): qA : A' -> CokA
        delta_comps[u] = qA.comps[u] * Ap  # CokA_u x kdim
    end
    delta = PMorphism{K}(KerC, CokA, delta_comps)

    return SnakeLemmaResult{K}(kerA, kerB, kerC, cokA, cokB, cokC, k1, k2, delta, c1, c2)
end

# ---------------------------------------------------------------------------
# Categorical biproduct/product/coproduct wrappers (mathematician-friendly API)
# ---------------------------------------------------------------------------

"""
    biproduct(M, N) -> (S, iM, iN, pM, pN)

Binary biproduct in the abelian category of `PModule`s.

Concretely, this is the direct sum `S = M \\oplus N` with its canonical
injections and projections:

- `iM : M -> S`, `iN : N -> S`
- `pM : S -> M`, `pN : S -> N`

This is a readability wrapper around `direct_sum_with_maps`.

Returns:
- `S`, the biproduct object,
- `iM`, `iN`, the canonical injections,
- `pM`, `pN`, the canonical projections.

See also: `direct_sum_with_maps`, `product`, `coproduct`.
"""
@inline biproduct(M::PModule{K}, N::PModule{K}) where {K} = direct_sum_with_maps(M, N)

"""
    coproduct(M, N) -> (S, iM, iN)

Categorical coproduct of two modules.

In an additive category (in particular for modules), the coproduct is the same
object as the product: the biproduct `M \\oplus N`.

Returns:
- `S`, the coproduct object,
- `iM : M -> S`,
- `iN : N -> S`.

Best practice:
- use `coproduct` for category-theoretic readability,
- use `biproduct` when you want both injections and projections at once.
"""
@inline function coproduct(M::PModule{K}, N::PModule{K}) where {K}
    return _coproduct_public(M, N)
end

"""
    product(M, N) -> (P, pM, pN)

Categorical product of two modules.

In an additive category (in particular for modules), the product is the same
object as the coproduct: the biproduct `M \\oplus N`.

Returns:
- `P`, the product object,
- `pM : P -> M`,
- `pN : P -> N`.

Best practice:
- use `product` for category-theoretic readability,
- use `biproduct` when you also need the injections.
"""
@inline function product(M::PModule{K}, N::PModule{K}) where {K}
    return _product_public(M, N)
end

# --- Finite products/coproducts (n-ary) ------------------------------------
#
# These are convenient in categorical workflows and are implemented using a
# single-pass block construction for speed (rather than iterating binary sums).

"""
    coproduct(mods::AbstractVector{<:PModule}) -> (S, injections)

Finite coproduct of a list of modules.

Returns:
- `S` : the direct sum object
- `injections[i] : mods[i] -> S` : the canonical injection maps

For an empty list, throws an error.

Best practice:
- use this n-ary form when you already have a collection of modules,
- use repeated binary `coproduct` only when the binary structure itself matters
  semantically.
"""
function coproduct(mods::AbstractVector{<:PModule{K}}) where {K}
    if isempty(mods)
        error("coproduct: need at least one module")
    elseif length(mods) == 1
        M = mods[1]
        return M, PMorphism{K}[id_morphism(M)]
    end
    S, injections, _ = _direct_sum_many_with_maps(mods)
    return S, injections
end

"""
    product(mods::AbstractVector{<:PModule}) -> (P, projections)

Finite product of a list of modules.

Returns:
- `P` : the direct sum object
- `projections[i] : P -> mods[i]` : the canonical projection maps

For an empty list, throws an error.

Best practice:
- use this n-ary form when you already have a collection of modules,
- use repeated binary `product` only when the binary structure itself matters
  semantically.
"""
function product(mods::AbstractVector{<:PModule{K}}) where {K}
    if isempty(mods)
        error("product: need at least one module")
    elseif length(mods) == 1
        M = mods[1]
        return M, PMorphism{K}[id_morphism(M)]
    end
    P, _, projections = _direct_sum_many_with_maps(mods)
    return P, projections
end

"""
    _direct_sum_many_with_maps(mods) -> (S, injections, projections)

Internal single-pass builder for finite biproducts of many modules.

Returns:
- `S`: the direct-sum module,
- `injections[i] : mods[i] -> S`,
- `projections[i] : S -> mods[i]`.

This helper exists for performance: it builds the whole biproduct in one pass
instead of iterating repeated binary direct sums.
"""
function _direct_sum_many_with_maps(mods::AbstractVector{<:PModule{K}}) where {K}
    m = length(mods)
    m == 0 && error("_direct_sum_many_with_maps: need at least one module")

    Q = mods[1].Q
    n = nvertices(Q)
    for M in mods
        if M.Q !== Q
            error("_direct_sum_many_with_maps: modules must live on the same poset")
        end
    end
    for M in mods
        M.field == mods[1].field || error("_direct_sum_many_with_maps: field mismatch")
    end

    # offsets[u][i] is the 0-based starting index of the i-th summand inside the
    # direct-sum fiber at vertex u. (So offsets[u][1] = 0.)
    offsets = [Vector{Int}(undef, m + 1) for _ in 1:n]
    for u in 1:n
        off = offsets[u]
        off[1] = 0
        for i in 1:m
            off[i+1] = off[i] + mods[i].dims[u]
        end
    end

    Sdims = [offsets[u][end] for u in 1:n]

    # Build cover-edge maps for the direct sum.
    cc = _get_cover_cache(Q)

    aligned = true
    for M in mods
        if length(M.edge_maps.preds) != n || length(M.edge_maps.succs) != n ||
           any(M.edge_maps.preds[v] != _preds(cc, v) for v in 1:n) ||
           any(M.edge_maps.succs[u] != _succs(cc, u) for u in 1:n)
            aligned = false
            break
        end
    end

    if aligned
        preds = [collect(_preds(cc, v)) for v in 1:n]
        succs = [collect(_succs(cc, u)) for u in 1:n]
        pred_slot_of_succ = [collect(_pred_slots_of_succ(cc, u)) for u in 1:n]

        maps_from_pred = [Vector{Matrix{K}}(undef, length(preds[v])) for v in 1:n]
        maps_to_succ   = [Vector{Matrix{K}}(undef, length(succs[u])) for u in 1:n]

        @inbounds for u in 1:n
            su = succs[u]
            outu = maps_to_succ[u]
            for j in eachindex(su)
                v = su[j]
                Auv = zeros(K, Sdims[v], Sdims[u])

                # Fill the block diagonal with summand maps.
                for i in 1:m
                    du = mods[i].dims[u]
                    dv = mods[i].dims[v]
                    if du == 0 || dv == 0
                        continue
                    end
                    r0 = offsets[v][i] + 1
                    r1 = offsets[v][i+1]
                    c0 = offsets[u][i] + 1
                    c1 = offsets[u][i+1]
                    block = mods[i].edge_maps.maps_to_succ[u][j]
                    copyto!(view(Auv, r0:r1, c0:c1), block)
                end

                outu[j] = Auv
                ip = pred_slot_of_succ[u][j]
                @inbounds maps_from_pred[v][ip] = Auv
            end
        end

        store = CoverEdgeMapStore{K,Matrix{K}}(preds, succs, maps_from_pred, maps_to_succ, cc.nedges)
        S = PModule{K}(Q, Sdims, store; field=mods[1].field)
    else
        # Fallback: keyed access (still O(|E|), but slower).
        edge_maps = Dict{Tuple{Int,Int}, Matrix{K}}()
        sizehint!(edge_maps, cc.nedges)
        for (u, v) in cover_edges(Q)
            Auv = zeros(K, Sdims[v], Sdims[u])
            for i in 1:m
                du = mods[i].dims[u]
                dv = mods[i].dims[v]
                if du == 0 || dv == 0
                    continue
                end
                r0 = offsets[v][i] + 1
                r1 = offsets[v][i+1]
                c0 = offsets[u][i] + 1
                c1 = offsets[u][i+1]
                block = mods[i].edge_maps[u, v]
                copyto!(view(Auv, r0:r1, c0:c1), block)
            end
            edge_maps[(u, v)] = Auv
        end
        S = PModule{K}(Q, Sdims, edge_maps; field=mods[1].field)
    end

    # Injections and projections (fill diagonal entries directly).
    injections = PMorphism{K}[]
    projections = PMorphism{K}[]
    for i in 1:m
        inj_comps = Vector{Matrix{K}}(undef, n)
        proj_comps = Vector{Matrix{K}}(undef, n)
        for u in 1:n
            du = mods[i].dims[u]
            Su = Sdims[u]
            Iu = zeros(K, Su, du)
            Pu = zeros(K, du, Su)
            if du != 0
                r0 = offsets[u][i] + 1
                c0 = offsets[u][i] + 1
                @inbounds for t in 1:du
                    Iu[r0 + t - 1, t] = one(K)
                    Pu[t, c0 + t - 1] = one(K)
                end
            end
            inj_comps[u] = Iu
            proj_comps[u] = Pu
        end
        push!(injections, PMorphism{K}(mods[i], S, inj_comps))
        push!(projections, PMorphism{K}(S, mods[i], proj_comps))
    end

    return S, injections, projections
end




# -----------------------------------------------------------------------------
# Pretty-printing / display (ASCII-only)
# -----------------------------------------------------------------------------
#
# Goals:
# - Mathematician-friendly summaries for 
# -  PModule / PMorphism (for REPL ergonomics when doing lots of algebra),
# Submodule, ShortExactSequence, and SnakeLemmaResult.
# - ASCII-only output (no Unicode arrows, symbols, etc).
# - Do NOT trigger heavy computations during printing:
#     * show(ShortExactSequence) must NOT call is_exact(ses)
#     * show(SnakeLemmaResult) must NOT recompute anything
# - Respect IOContext(:limit=>true) by truncating long dim vectors.

"""
    _scalar_name(K) -> String

Return a compact human-readable name for the scalar type `K`.

Used only by the display helpers in this file.
"""
_scalar_name(::Type{K}) where {K} = string(K)

"""
    _dims_stats(dims) -> (total, nnz, maxd)

Compute cheap summary statistics for a dimension vector.

Returns:
- `total`: the sum of all entries,
- `nnz`: the number of nonzero entries,
- `maxd`: the maximum entry.

This helper is allocation-free and is used by the lightweight pretty-printers.
"""
function _dims_stats(dims::AbstractVector{<:Integer})
    total = 0
    nnz = 0
    maxd = 0
    @inbounds for d0 in dims
        d = Int(d0)
        total += d
        if d != 0
            nnz += 1
            if d > maxd
                maxd = d
            end
        end
    end
    return total, nnz, maxd
end

"""
    _print_int_vec(io, v; max_elems=12, head=4, tail=3) -> Nothing

Print an integer vector to `io`, truncating when `IOContext(io, :limit=>true)`
requests compact output.

Returns `nothing` after writing to `io`. This helper is intentionally cheap and
is used by the ASCII-only pretty-printers in this module.
"""
function _print_int_vec(io::IO, v::AbstractVector{<:Integer};
                        max_elems::Int=12, head::Int=4, tail::Int=3)
    n = length(v)
    print(io, "[")
    if n == 0
        print(io, "]")
        return
    end

    limit = get(io, :limit, false)

    # Full print if not limiting or short enough.
    if !limit || n <= max_elems
        @inbounds for i in 1:n
            i > 1 && print(io, ", ")
            print(io, Int(v[i]))
        end
        print(io, "]")
        return
    end

    # Truncated print: head entries, "...", tail entries.
    h = min(head, n)
    t = min(tail, max(0, n - h))

    @inbounds for i in 1:h
        i > 1 && print(io, ", ")
        print(io, Int(v[i]))
    end

    if h < n - t
        print(io, ", ..., ")
    elseif h < n && t > 0
        print(io, ", ")
    end

    @inbounds for i in (n - t + 1):n
        i > (n - t + 1) && print(io, ", ")
        print(io, Int(v[i]))
    end

    print(io, "]")
    return
end

"""
    Base.show(io::IO, summary::ValidationSummary)

Compact one-line summary for a wrapped validation report.
"""
function Base.show(io::IO, summary::ValidationSummary)
    report = summary.report
    kind = get(report, :kind, :validation)
    valid = get(report, :valid, false)
    issues = get(report, :issues, String[])
    print(io, "ValidationSummary(kind=", kind,
          ", valid=", valid,
          ", issues=", length(issues), ")")
end

"""
    Base.show(io::IO, ::MIME"text/plain", summary::ValidationSummary)

Verbose multi-line summary for a wrapped validation report.
"""
function Base.show(io::IO, ::MIME"text/plain", summary::ValidationSummary)
    report = summary.report
    kind = get(report, :kind, :validation)
    valid = get(report, :valid, false)
    issues = get(report, :issues, String[])
    println(io, "ValidationSummary")
    println(io, "  kind = ", kind)
    println(io, "  valid = ", valid)
    if haskey(report, :dimensions)
        println(io, "  dimensions = ", report.dimensions)
    end
    if isempty(issues)
        println(io, "  issues = none")
    else
        println(io, "  issues:")
        for msg in issues
            println(io, "    - ", msg)
        end
    end
end

"""
    Base.show(io::IO, S::Submodule)

Compact one-line summary for `Submodule`.
"""
function Base.show(io::IO, S::Submodule{K}) where {K}
    N = _sub(S)
    M = _ambient(S)
    nverts = nvertices(M.Q)
    sub_sum, _, _ = _dims_stats(N.dims)
    amb_sum, _, _ = _dims_stats(M.dims)
    print(io,
          "Submodule(",
          "nverts=", nverts,
          ", sub_sum=", sub_sum,
          ", ambient_sum=", amb_sum,
          ")")
end

"""
    Base.show(io::IO, ::MIME"text/plain", S::Submodule)

Verbose multi-line summary for `Submodule`. ASCII-only.
"""
function Base.show(io::IO, ::MIME"text/plain", S::Submodule{K}) where {K}
    N = _sub(S)
    M = _ambient(S)
    nverts = nvertices(M.Q)

    sub_sum, sub_nnz, sub_max = _dims_stats(N.dims)
    amb_sum, amb_nnz, amb_max = _dims_stats(M.dims)

    println(io, "Submodule")
    println(io, "  scalars = ", _scalar_name(K))
    println(io, "  nverts = ", nverts)

    print(io, "  sub dims = ")
    _print_int_vec(io, N.dims)
    println(io)
    println(io, "    sum = ", sub_sum, ", nnz = ", sub_nnz, ", max = ", sub_max)

    print(io, "  ambient dims = ")
    _print_int_vec(io, M.dims)
    println(io)
    println(io, "    sum = ", amb_sum, ", nnz = ", amb_nnz, ", max = ", amb_max)

    println(io, "  inclusion : _sub(S) -> _ambient(S)  (stored in field incl)")
end


"""
    Base.show(io::IO, ses::ShortExactSequence)

Compact one-line summary for `ShortExactSequence`.

NOTE: This does NOT call `is_exact(ses)`; it only reports cached status.
"""
function Base.show(io::IO, ses::ShortExactSequence{K}) where {K}
    nverts = nvertices(ses.B.Q)
    Asum, _, _ = _dims_stats(ses.A.dims)
    Bsum, _, _ = _dims_stats(ses.B.dims)
    Csum, _, _ = _dims_stats(ses.C.dims)

    print(io,
          "ShortExactSequence(",
          "nverts=", nverts,
          ", A_sum=", Asum,
          ", B_sum=", Bsum,
          ", C_sum=", Csum,
          ", checked=", ses.checked,
          ", exact=")
    if ses.checked
        print(io, ses.exact)
    else
        print(io, "unknown")
    end
    print(io, ")")
end

"""
    Base.show(io::IO, ::MIME"text/plain", ses::ShortExactSequence)

Verbose multi-line summary for `ShortExactSequence`. ASCII-only.

NOTE: This does NOT call `is_exact(ses)`; it only reports cached status.
"""
function Base.show(io::IO, ::MIME"text/plain", ses::ShortExactSequence{K}) where {K}
    nverts = nvertices(ses.B.Q)

    Asum, Annz, Amax = _dims_stats(ses.A.dims)
    Bsum, Bnnz, Bmax = _dims_stats(ses.B.dims)
    Csum, Cnnz, Cmax = _dims_stats(ses.C.dims)

    println(io, "ShortExactSequence")
    println(io, "  0 -> A -(i)-> B -(p)-> C -> 0")
    println(io, "  scalars = ", _scalar_name(K))
    println(io, "  nverts = ", nverts)

    println(io, "  checked = ", ses.checked)
    if ses.checked
        println(io, "  exact = ", ses.exact)
    else
        println(io, "  exact = unknown (call is_exact(ses) to check and cache)")
    end
    println(io, "  caches: ker(p) = ", ses.ker_p !== nothing,
                ", im(i) = ", ses.img_i !== nothing)

    print(io, "  A dims = ")
    _print_int_vec(io, ses.A.dims)
    println(io)
    println(io, "    sum = ", Asum, ", nnz = ", Annz, ", max = ", Amax)

    print(io, "  B dims = ")
    _print_int_vec(io, ses.B.dims)
    println(io)
    println(io, "    sum = ", Bsum, ", nnz = ", Bnnz, ", max = ", Bmax)

    print(io, "  C dims = ")
    _print_int_vec(io, ses.C.dims)
    println(io)
    println(io, "    sum = ", Csum, ", nnz = ", Cnnz, ", max = ", Cmax)

    println(io, "  maps: use ses.i and ses.p (both are PMorphism objects)")
end


"""
    Base.show(io::IO, sn::SnakeLemmaResult)

Compact one-line summary for `SnakeLemmaResult`.
"""
function Base.show(io::IO, sn::SnakeLemmaResult{K}) where {K}
    nverts = nvertices(sn.delta.dom.Q)
    kerCsum, _, _ = _dims_stats(sn.kerC[1].dims)
    cokAsum, _, _ = _dims_stats(sn.cokA[1].dims)

    print(io,
          "SnakeLemmaResult(",
          "nverts=", nverts,
          ", delta: kerC_sum=", kerCsum,
          " -> cokerA_sum=", cokAsum,
          ")")
end

"""
    Base.show(io::IO, ::MIME"text/plain", sn::SnakeLemmaResult)

Verbose multi-line summary for `SnakeLemmaResult`. ASCII-only.
"""
function Base.show(io::IO, ::MIME"text/plain", sn::SnakeLemmaResult{K}) where {K}
    nverts = nvertices(sn.delta.dom.Q)

    println(io, "SnakeLemmaResult")
    println(io, "  kerA -> kerB -> kerC --delta--> cokerA -> cokerB -> cokerC")
    println(io, "  scalars = ", _scalar_name(K))
    println(io, "  nverts = ", nverts)

    """
        _print_obj(io2, name, tup) -> Nothing

    Local pretty-print helper for one kernel/cokernel entry of a
    `SnakeLemmaResult`.

    Writes a compact multi-line summary of the module in `tup[1]` and returns
    `nothing`.
    """
    function _print_obj(io2::IO, name::AbstractString, tup)
        M = tup[1]
        s, nnz, mx = _dims_stats(M.dims)
        print(io2, "  ", name, " dims = ")
        _print_int_vec(io2, M.dims)
        println(io2)
        println(io2, "    sum = ", s, ", nnz = ", nnz, ", max = ", mx)
    end

    _print_obj(io, "kerA", sn.kerA)
    _print_obj(io, "kerB", sn.kerB)
    _print_obj(io, "kerC", sn.kerC)
    _print_obj(io, "cokerA", sn.cokA)
    _print_obj(io, "cokerB", sn.cokB)
    _print_obj(io, "cokerC", sn.cokC)

    println(io, "  maps: k1, k2, delta, c1, c2  (access as fields on the result)")
end


end
