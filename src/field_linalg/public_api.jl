# -----------------------------------------------------------------------------
# field_linalg/public_api.jl
#
# Scope:
#   Public FieldLinAlg entrypoints plus restricted helpers and module
#   initialization for threshold loading/autotune.
# Owns:
#   - exported/public entrypoints (`rank`, `nullspace`, `colspace`,
#     `solve_fullcolumn`, restricted helpers),
#   - thin public dispatch over routing + kernel families,
#   - module initialization for threshold loading/autotune.
# Does not own:
#   - backend heuristics themselves,
#   - QQ/non-QQ kernel internals,
#   - sparse elimination primitives.
# Depends on:
#   - `backend_routing.jl` for runtime backend choice,
#   - `qq_engine.jl` / `nonqq_engines.jl` / `sparse_rref.jl` for the actual
#     kernels exposed here.
# -----------------------------------------------------------------------------
#
# Contributor note:
# - add a new threshold in `thresholds.jl`, then thread it through:
#   `_current_linalg_thresholds`, `_apply_linalg_thresholds!`, load/save, and
#   autotune if the threshold should be machine-fitted.
# - add a new backend rule in `backend_routing.jl`; keep the public API here as
#   a thin dispatcher over routing + kernels.
# - add parity tests in `test/test_field_linalg.jl`, preferably with explicit
#   backend forcing (`:auto` vs forced backend) when routing changes.
# - update benchmarks by change class:
#   routing/crossover changes -> `benchmark/sparse_qq_backend_microbench.jl`
#   sparse elimination changes -> `benchmark/field_linalg_sparse_rref_microbench.jl`
#   end-to-end user impact -> the owning subsystem benchmark that calls into
#   FieldLinAlg (e.g. FiniteFringe/Flange/ZnEncoding/ModuleComplexes).

# -----------------------------------------------------------------------------
# Public API (field-generic dispatch)
# -----------------------------------------------------------------------------

@inline function _free_columns_from_pivots(n::Int, pivs)
    free = Int[]
    sizehint!(free, max(0, n - length(pivs)))
    piv_i = 1
    npiv = length(pivs)
    @inbounds for j in 1:n
        if piv_i <= npiv && pivs[piv_i] == j
            piv_i += 1
        else
            push!(free, j)
        end
    end
    return free
end

function _kernel_from_rref(::Type{K}, R::AbstractMatrix{K}, pivs, n::Int) where {K}
    free = _free_columns_from_pivots(n, pivs)
    isempty(free) && return zeros(K, n, 0)

    Z = zeros(K, n, length(free))
    @inbounds for (k, jf) in enumerate(free)
        Z[jf, k] = one(K)
        for (i, jp) in enumerate(pivs)
            Z[jp, k] = -R[i, jf]
        end
    end
    return Z
end

@inline function _img_from_pivots(::Type{K}, A, pivs) where {K}
    isempty(pivs) && return zeros(K, size(A, 1), 0)
    return Matrix{K}(A[:, collect(pivs)])
end

function _img_from_pivots(::Type{K}, A::SparseMatrixCSC{K,Int}, pivs) where {K}
    npiv = length(pivs)
    npiv == 0 && return zeros(K, size(A, 1), 0)
    out = zeros(K, size(A, 1), npiv)
    @inbounds for (jout, jpiv) in enumerate(pivs)
        for ptr in nzrange(A, jpiv)
            out[rowvals(A)[ptr], jout] = nonzeros(A)[ptr]
        end
    end
    return out
end

function _kernel_image_summary_sparse(A::SparseMatrixCSC{K,Int}) where {K}
    m, n = size(A)
    if n == 0
        return (rank=0, ker=zeros(K, 0, 0), img=zeros(K, m, 0))
    end
    if m == 0
        return (rank=0, ker=Matrix{K}(I, n, n), img=zeros(K, 0, 0))
    end

    R = _SparseRREF{K}(n)
    rows = _sparse_rows(A)
    maxrank = min(m, n)
    @inbounds for i in 1:m
        _sparse_rref_push_homogeneous!(R, rows[i])
        if _rref_rank(R) == maxrank
            break
        end
    end
    pivs = R.pivot_cols
    return (rank=_rref_rank(R), ker=_nullspace_from_pivots(R, n), img=_img_from_pivots(K, A, pivs))
end

"""
    _kernel_image_summary(field, A; backend=:auto) -> NamedTuple

Internal fused summary for hot quotient/cohomology code paths.

Returns `(rank, ker, img)`, where:
- `rank` is the rank of `A`,
- `ker` is a basis matrix for `ker(A)`,
- `img` is a basis matrix for `im(A)`.

The QQ path fuses kernel/image extraction so callers do not pay for separate
`nullspace` and `colspace` passes.
"""
function _kernel_image_summary(field::AbstractCoeffField, A; backend::Symbol=:auto)
    if field isa QQField
        return _kernel_image_summary(elimination_summary(field, A; backend=backend))
    end
    ker = nullspace(field, A; backend=backend)
    img = colspace(field, A; backend=backend)
    return (rank=size(img, 2), ker=Matrix{eltype(ker)}(ker), img=Matrix{eltype(img)}(img))
end

"""
    FullColumnSolveFactor{F}

Reusable factor object for repeated `solve_fullcolumn(...)` calls with a fixed
left matrix `B`.

Create one with [`factor_fullcolumn`](@ref), then pass it back via
`solve_fullcolumn(field, B, Y; factor=fac)`.
"""
struct FullColumnSolveFactor{F,S}
    backend::Symbol
    payload::F
    summary::S
end

function Base.show(io::IO, fac::FullColumnSolveFactor)
    print(io, "FullColumnSolveFactor(backend=", fac.backend,
          ", payload=", nameof(typeof(fac.payload)),
          ", summary=", nameof(typeof(fac.summary)), ")")
end

factor_backend(fac::FullColumnSolveFactor) = fac.backend
elimination_summary(fac::FullColumnSolveFactor) = fac.summary
rank(fac::FullColumnSolveFactor) = rank(fac.summary)
nullspace(fac::FullColumnSolveFactor) = nullspace(fac.summary)
colspace(fac::FullColumnSolveFactor) = colspace(fac.summary)
_kernel_image_summary(fac::FullColumnSolveFactor) = _kernel_image_summary(fac.summary)

abstract type AbstractQQMatrixAnalysis end

"""
    QQMatrixAnalysis

Reusable exact analysis object for a `QQ` matrix. It stores an elimination
summary and, optionally, a full-column solve factor for repeated solves.
"""
struct QQMatrixAnalysis{S,F} <: AbstractQQMatrixAnalysis
    backend::Symbol
    summary::S
    factor::F
end

function Base.show(io::IO, A::QQMatrixAnalysis)
    print(io, "QQMatrixAnalysis(backend=", A.backend,
          ", summary=", nameof(typeof(A.summary)),
          ", factor=", nameof(typeof(A.factor)), ")")
end

analysis_backend(A::QQMatrixAnalysis) = A.backend
elimination_summary(A::QQMatrixAnalysis) = A.summary
fullcolumn_factor(A::QQMatrixAnalysis) = A.factor
factor_backend(A::QQMatrixAnalysis) = A.factor === nothing ? nothing : factor_backend(A.factor)

rank(A::QQMatrixAnalysis) = rank(A.summary)
nullspace(A::QQMatrixAnalysis) = nullspace(A.summary)
colspace(A::QQMatrixAnalysis) = colspace(A.summary)
_kernel_image_summary(A::QQMatrixAnalysis) = _kernel_image_summary(A.summary)

@inline _unwrap_factor(factor) = factor
@inline _unwrap_factor(factor::FullColumnSolveFactor) = factor.payload
@inline _unwrap_factor(analysis::QQMatrixAnalysis) = _unwrap_factor(analysis.factor)
@inline _wrapped_factor_backend(::Any) = nothing
@inline _wrapped_factor_backend(factor::FullColumnSolveFactor) = factor.backend
@inline _wrapped_factor_backend(analysis::QQMatrixAnalysis) = analysis.factor === nothing ? nothing : factor_backend(analysis.factor)

@inline function _resolve_factor_backend(field::AbstractCoeffField, B;
                                         backend::Symbol=:auto, factor=nothing)
    fac_backend = _wrapped_factor_backend(factor)
    if fac_backend !== nothing
        backend == :auto || backend == fac_backend ||
            error("solve_fullcolumn: explicit backend $backend does not match factor backend $fac_backend")
        return fac_backend
    end
    return field isa QQField ? _choose_solve_backend(field, B; backend=backend, factor=factor) :
                               _choose_linalg_backend(field, B; op=:solve, backend=backend)
end

@inline function _factor_fullcolumnQQ_cached(B::AbstractMatrix{<:QQ};
                                             cache::Bool=true,
                                             backend::Symbol=:julia_exact)
    if backend == :nemo
        if cache && _can_weak_cache_key(B)
            return get!(_NEMO_FULLCOLUMN_FACTOR_CACHE_QQ, B) do
                _factor_fullcolumn_nemoQQ(B)
            end
        end
        return _factor_fullcolumn_nemoQQ(B)
    end
    if cache && _can_weak_cache_key(B)
        return get!(_FULLCOLUMN_FACTOR_CACHE, B) do
            _factor_fullcolumnQQ(B)
        end
    end
    return _factor_fullcolumnQQ(B)
end

"""
    factor_fullcolumn(field, B; backend=:auto, cache=true)

Build a reusable exact factor for repeated solves of `B * X = Y` with
full-column-rank `B`.

Use the result with `solve_fullcolumn(field, B, Y; factor=fac)`. For `QQ`,
`backend=:auto` chooses the reusable exact backend and avoids modular solve
routing, since modular solves do not currently expose a reusable factor.
"""
function factor_fullcolumn(field::QQField, B; backend::Symbol=:auto, cache::Bool=true, summary=nothing)
    be = _choose_solve_backend(field, B; backend=backend, factor=nothing)
    if be == :modular
        be = _have_nemo() ? :nemo : (_is_sparse_like(B) ? :julia_sparse : :julia_exact)
    end
    fac = _factor_fullcolumnQQ_cached(B; cache=cache, backend=(be == :nemo ? :nemo : :julia_exact))
    sumobj = summary === nothing ? elimination_summary(field, B; backend=(be == :nemo ? :nemo : :julia_exact)) : summary
    return FullColumnSolveFactor{typeof(fac),typeof(sumobj)}(be, fac, sumobj)
end

function factor_fullcolumn(field::AbstractCoeffField, B; backend::Symbol=:auto, cache::Bool=true)
    error("FieldLinAlg.factor_fullcolumn: reusable factor surface is currently implemented only for QQ")
end

"""
    analyze_matrix(field, A; backend=:auto, cache=true, fullcolumn_factor=false)

Build a reusable exact `QQ` analysis object for `A`. The result exposes
`rank`, `nullspace`, `colspace`, and `_kernel_image_summary` without repeating
elimination. If `fullcolumn_factor=true`, it also stores a reusable
full-column solve factor for `solve_fullcolumn(...; analysis=...)`.
"""
function analyze_matrix(field::QQField, A; backend::Symbol=:auto, cache::Bool=true, fullcolumn_factor::Bool=false)
    be = _choose_linalg_backend(field, A; op=:rref, backend=backend)
    if be == :modular
        be = _have_nemo() ? :nemo : (_is_sparse_like(A) ? :julia_sparse : :julia_exact)
    end
    summary = elimination_summary(field, A; backend=be == :nemo ? :nemo : :julia_exact)
    fac = fullcolumn_factor ? factor_fullcolumn(field, A; backend=backend, cache=cache, summary=summary) : nothing
    return QQMatrixAnalysis{typeof(summary),typeof(fac)}(be, summary, fac)
end

function analyze_matrix(field::AbstractCoeffField, A; backend::Symbol=:auto, cache::Bool=true, fullcolumn_factor::Bool=false)
    error("FieldLinAlg.analyze_matrix: reusable exact analysis is currently implemented only for QQ")
end

abstract type AbstractQQEliminationSummary end

"""
    DenseQQEliminationSummary

Reusable exact elimination summary for a dense `QQ` matrix. Use
[`elimination_summary`](@ref) to build it, then query it with `rank`, `nullspace`,
`colspace`, or `_kernel_image_summary`.
"""
struct DenseQQEliminationSummary <: AbstractQQEliminationSummary
    rref::Matrix{QQ}
    pivots::Vector{Int}
    image_basis::Matrix{QQ}
end

"""
    SparseQQEliminationSummary

Reusable exact elimination summary for a sparse `QQ` matrix produced by the
streaming sparse RREF engine.
"""
struct SparseQQEliminationSummary <: AbstractQQEliminationSummary
    rref::_SparseRREF{QQ}
    ncols::Int
    image_basis::Matrix{QQ}
end

_summary_rank(S::DenseQQEliminationSummary) = length(S.pivots)
_summary_rank(S::SparseQQEliminationSummary) = _rref_rank(S.rref)
_summary_nullspace(S::DenseQQEliminationSummary) = _kernel_from_rref(QQ, S.rref, S.pivots, size(S.rref, 2))
_summary_nullspace(S::SparseQQEliminationSummary) = _nullspace_from_pivots(S.rref, S.ncols)
_summary_colspace(S::AbstractQQEliminationSummary) = S.image_basis

rank(S::AbstractQQEliminationSummary) = _summary_rank(S)
nullspace(S::AbstractQQEliminationSummary) = _summary_nullspace(S)
colspace(S::AbstractQQEliminationSummary) = _summary_colspace(S)
_kernel_image_summary(S::AbstractQQEliminationSummary) = (rank=rank(S), ker=nullspace(S), img=colspace(S))

function _build_dense_elimination_summaryQQ(A::AbstractMatrix{QQ}; backend::Symbol=:julia_exact)
    if backend == :nemo
        R, pivs = _nemo_rref(QQField(), A; pivots=true)
        cpivs = Int[collect(pivs)...]
        return DenseQQEliminationSummary(R, cpivs, _img_from_pivots(QQ, A, cpivs))
    end
    R, pivs = _rrefQQ(A; pivots=true)
    cpivs = Int[collect(pivs)...]
    return DenseQQEliminationSummary(R, cpivs, _img_from_pivots(QQ, A, cpivs))
end

function _build_sparse_elimination_summaryQQ(A::SparseMatrixCSC{QQ,Int})
    m, n = size(A)
    if n == 0
        R = _SparseRREF{QQ}(0)
        return SparseQQEliminationSummary(R, 0, zeros(QQ, m, 0))
    end
    if m == 0
        R = _SparseRREF{QQ}(n)
        return SparseQQEliminationSummary(R, n, zeros(QQ, 0, 0))
    end
    R = _SparseRREF{QQ}(n)
    rows = _sparse_rows(A)
    maxrank = min(m, n)
    @inbounds for i in 1:m
        _sparse_rref_push_homogeneous!(R, rows[i])
        if _rref_rank(R) == maxrank
            break
        end
    end
    return SparseQQEliminationSummary(R, n, _img_from_pivots(QQ, A, R.pivot_cols))
end

"""
    elimination_summary(field, A; backend=:auto)

Build a reusable exact elimination summary for `QQ` matrices. The returned
object can be queried with `rank(summary)`, `nullspace(summary)`,
`colspace(summary)`, and `_kernel_image_summary(summary)` without repeating
elimination.
"""
function elimination_summary(field::QQField, A; backend::Symbol=:auto)
    if A isa SparseMatrixCSC{QQ,Int}
        return _build_sparse_elimination_summaryQQ(A)
    end
    be = _choose_linalg_backend(field, A; op=:rref, backend=backend)
    if be == :modular
        be = _have_nemo() ? :nemo : :julia_exact
    end
    return _build_dense_elimination_summaryQQ(A; backend=(be == :nemo ? :nemo : :julia_exact))
end

function elimination_summary(field::AbstractCoeffField, A; backend::Symbol=:auto)
    error("FieldLinAlg.elimination_summary: reusable elimination summaries are currently implemented only for QQ")
end

function rref(field::AbstractCoeffField, A; pivots::Bool=true, backend::Symbol=:auto)
    if field isa QQField
        trait = _matrix_backend_trait(field, A; op=:rref, backend=backend)
        if trait isa NemoMatrixBackend
            return _nemo_rref_qq_mat(_nemo_matrix(trait, field, A); pivots=pivots)
        end
        return _rrefQQ(A; pivots=pivots)
    end
    if field isa PrimeField && field.p == 2
        return _rref_f2(A; pivots=pivots)
    end
    if field isa PrimeField && field.p == 3
        return _rref_f3(A; pivots=pivots)
    end
    if field isa PrimeField && field.p > 3
        trait = _matrix_backend_trait(field, A; op=:rref, backend=backend)
        if trait isa NemoMatrixBackend
            return _nemo_rref_fp_mat(field, _nemo_matrix(trait, field, A); pivots=pivots)
        end
        return _rref_fp(A; pivots=pivots)
    end
    if field isa RealField
        _ = _choose_linalg_backend(field, A; op=:rref, backend=backend)
        return _rref_float(field, A; pivots=pivots)
    end
    error("FieldLinAlg.rref: unsupported field $(typeof(field))")
end

function rank(field::AbstractCoeffField, A; backend::Symbol=:auto)
    # Call path:
    # `rank` -> backend routing (`backend_routing.jl`) -> QQ/non-QQ kernel
    # (`qq_engine.jl` or `nonqq_engines.jl`).
    if backend == :auto && _is_tiny_matrix(A)
        return _rank_tiny(field, A)
    end
    if field isa QQField
        trait = _matrix_backend_trait(field, A; op=:rank, backend=backend)
        if trait isa NemoMatrixBackend
            return _nemo_rank_qq_mat(_nemo_matrix(trait, field, A))
        end
        return _rankQQ(A)
    end
    if field isa PrimeField && field.p == 2
        return _rank_f2(A)
    end
    if field isa PrimeField && field.p == 3
        return _rank_f3(A)
    end
    if field isa PrimeField && field.p > 3
        trait = _matrix_backend_trait(field, A; op=:rank, backend=backend)
        if trait isa NemoMatrixBackend
            return _nemo_rank_fp_mat(field, _nemo_matrix(trait, field, A))
        end
        return _rank_fp(A)
    end
    if field isa RealField
        be = _choose_linalg_backend(field, A; op=:rank, backend=backend)
        if be == :float_dense_svd
            return _rank_float_svd(field, A)
        end
        return _rank_float(field, A)
    end
    error("FieldLinAlg.rank: unsupported field $(typeof(field))")
end

function nullspace(field::AbstractCoeffField, A; backend::Symbol=:auto)
    # Call path:
    # `nullspace` -> backend routing (`backend_routing.jl`) -> exact/float
    # kernel (`qq_engine.jl` or `nonqq_engines.jl`).
    if field isa QQField
        trait = _matrix_backend_trait(field, A; op=:nullspace, backend=backend)
        if trait isa NemoMatrixBackend
            return _nemo_nullspace_qq_mat(_nemo_matrix(trait, field, A))
        end
        be = _choose_linalg_backend(field, A; op=:nullspace, backend=backend)
        if be == :modular
            if !(A isa SparseMatrixCSC)
                N = _nullspace_modularQQ(A)
                N === nothing || return N
            end
        end
        return _nullspaceQQ(A)
    end
    if field isa PrimeField && field.p == 2
        return _nullspace_f2(A)
    end
    if field isa PrimeField && field.p == 3
        return _nullspace_f3(A)
    end
    if field isa PrimeField && field.p > 3
        trait = _matrix_backend_trait(field, A; op=:nullspace, backend=backend)
        if trait isa NemoMatrixBackend
            return _nemo_nullspace_fp_mat(field, _nemo_matrix(trait, field, A))
        end
        return _nullspace_fp(A)
    end
    if field isa RealField
        be = _choose_linalg_backend(field, A; op=:nullspace, backend=backend)
        if be == :float_dense_svd
            return _nullspace_float_svd(field, A)
        end
        if be == :float_sparse_svds
            Z = _nullspace_float_svds(field, A)
            Z === nothing || return Z
        end
        return _nullspace_float(field, A)
    end
    error("FieldLinAlg.nullspace: unsupported field $(typeof(field))")
end

function colspace(field::AbstractCoeffField, A; backend::Symbol=:auto)
    C, _ = _colspace_with_pivots(field, A; backend=backend)
    return C
end

function _colspace_with_pivots(field::AbstractCoeffField, A; backend::Symbol=:auto)
    if field isa QQField
        trait = _matrix_backend_trait(field, A; op=:colspace, backend=backend)
        if trait isa NemoMatrixBackend
            pivs = _nemo_pivots_mat(_nemo_matrix(trait, field, A))
            cpivs = Int[collect(pivs)...]
            return (A[:, cpivs], cpivs)
        end
        _, pivs = _rrefQQ(A; pivots=true)
        cpivs = Int[collect(pivs)...]
        return (_img_from_pivots(QQ, A, cpivs), cpivs)
    end
    if field isa PrimeField && field.p == 2
        pivs = _pivot_cols_f2(A)
        return (A[:, pivs], pivs)
    end
    if field isa PrimeField && field.p == 3
        _, pivs = _rref_f3(A; pivots=true)
        cpivs = Int[collect(pivs)...]
        return (A[:, cpivs], cpivs)
    end
    if field isa PrimeField && field.p > 3
        trait = _matrix_backend_trait(field, A; op=:colspace, backend=backend)
        if trait isa NemoMatrixBackend
            pivs = _nemo_pivots_mat(_nemo_matrix(trait, field, A))
            cpivs = Int[collect(pivs)...]
            return (A[:, cpivs], cpivs)
        end
        _, pivs = _rref_fp(A; pivots=true)
        cpivs = Int[collect(pivs)...]
        return (A[:, cpivs], cpivs)
    end
    if field isa RealField
        _, pivs = _rref_float(field, A; pivots=true)
        cpivs = Int[collect(pivs)...]
        return (_img_from_pivots(eltype(A), A, cpivs), cpivs)
    end
    error("FieldLinAlg._colspace_with_pivots: unsupported field $(typeof(field))")
end

function solve_fullcolumn(field::AbstractCoeffField, B, Y;
                          check_rhs::Bool=true, backend::Symbol=:auto,
                          cache::Bool=true, factor=nothing, analysis=nothing)
    # Call path:
    # `solve_fullcolumn` -> routing/factor selection (`backend_routing.jl`) ->
    # backend factorization/solve kernel (`qq_engine.jl` or
    # `nonqq_engines.jl`).
    if analysis !== nothing
        factor === nothing || error("solve_fullcolumn: pass either factor=... or analysis=..., not both")
        fullcolumn_factor(analysis) === nothing &&
            error("solve_fullcolumn: analysis object does not carry a full-column factor; build it with fullcolumn_factor=true")
        factor = analysis
    end
    if backend == :auto && _is_tiny_solve(B, Y) && !(field isa RealField)
        return _solve_fullcolumn_tiny(field, B, Y; check_rhs=check_rhs)
    end
    factor_backend = _resolve_factor_backend(field, B; backend=backend, factor=factor)
    factor_payload = _unwrap_factor(factor)
    if field isa QQField
        be = factor_backend
        if be == :nemo
            return _solve_fullcolumn_nemoQQ(B, Y; check_rhs=check_rhs, cache=cache, factor=factor_payload)
        end
        if be == :modular
            if !(B isa SparseMatrixCSC)
                X = _solve_fullcolumn_modularQQ(B, Y; check_rhs=check_rhs)
                X === nothing || return X
            end
        end
        jfactor = factor_payload isa FullColumnFactor{QQ} ? factor_payload : nothing
        return _solve_fullcolumnQQ(B, Y; check_rhs=check_rhs, cache=cache, factor=jfactor)
    end
    if field isa PrimeField && field.p == 2
        return _solve_fullcolumn_f2(B, Y; check_rhs=check_rhs, cache=cache, factor=factor_payload)
    end
    if field isa PrimeField && field.p == 3
        return _solve_fullcolumn_f3(B, Y; check_rhs=check_rhs, cache=cache, factor=factor_payload)
    end
    if field isa PrimeField && field.p > 3
        be = factor_backend
        if be == :nemo
            return _solve_fullcolumn_nemo_fp(field, B, Y; check_rhs=check_rhs, cache=cache, factor=factor_payload)
        end
        return _solve_fullcolumn_fp(B, Y; check_rhs=check_rhs)
    end
    if field isa RealField
        be = factor_backend
        if be == :float_sparse_qr
            Bs = B isa SparseMatrixCSC ? B : sparse(B)
            return _solve_fullcolumn_float(field, Bs, Y; check_rhs=check_rhs, cache=cache, factor=factor_payload)
        end
        return _solve_fullcolumn_float(field, B, Y; check_rhs=check_rhs)
    end
    error("FieldLinAlg.solve_fullcolumn: unsupported field $(typeof(field))")
end

function rank_dim(field::AbstractCoeffField, A; backend::Symbol=:auto, kwargs...)
    if field isa QQField
        return _rankQQ_dim(A; backend=backend, kwargs...)
    end
    if field isa PrimeField && field.p == 2
        return _rank_f2(A)
    end
    if field isa PrimeField && field.p == 3
        return _rank_f3(A)
    end
    if field isa PrimeField && field.p > 3
        be = _choose_linalg_backend(field, A; op=:rank, backend=backend)
        if be == :nemo
            return _nemo_rank(field, A)
        end
        return _rank_fp(A)
    end
    if field isa RealField
        be = _choose_linalg_backend(field, A; op=:rank, backend=backend)
        if be == :float_dense_svd
            return _rank_float_svd(field, A)
        end
        return _rank_float(field, A)
    end
    return rank(field, A; backend=backend)
end

function rank_restricted(field::AbstractCoeffField, A::SparseMatrixCSC,
                         rows::AbstractVector{Int}, cols::AbstractVector{Int};
                         backend::Symbol=:auto, kwargs...)
    if _is_tiny_matrix_dims(length(rows), length(cols)) &&
       !(field isa RealField) &&
       !(field isa PrimeField && field.p > 3)
        check = haskey(kwargs, :check) ? kwargs[:check] : false
        S = Matrix(_sparse_extract_restricted(A, rows, cols; check=check))
        return rank(field, S; backend=backend)
    end
    if field isa QQField
        return _rankQQ_restricted(A, rows, cols; kwargs...)
    end
    if field isa PrimeField && field.p == 2
        return _rank_restricted_f2(A, rows, cols; kwargs...)
    end
    if field isa PrimeField && field.p == 3
        return _rank_restricted_f3(A, rows, cols; kwargs...)
    end
    if field isa PrimeField && field.p > 3
        return _rank_restricted_sparse_generic(A, rows, cols; kwargs...)
    end
    if field isa RealField
        return _rank_restricted_float_sparse(field, A, rows, cols; kwargs...)
    end
    error("FieldLinAlg.rank_restricted: unsupported field $(typeof(field))")
end

function rank_restricted(field::AbstractCoeffField, A::AbstractMatrix,
                         rows::AbstractVector{Int}, cols::AbstractVector{Int};
                         backend::Symbol=:auto, kwargs...)
    A isa SparseMatrixCSC && return rank_restricted(field, A, rows, cols; backend=backend, kwargs...)
    nr = length(rows)
    nc = length(cols)
    if nr == 0 || nc == 0
        return 0
    end
    return rank(field, view(A, rows, cols); backend=backend, kwargs...)
end

@inline function _active_words_nonempty(words::AbstractVector{UInt64})::Bool
    @inbounds for w in words
        w == 0 || return true
    end
    return false
end

const RANKQQ_RESTRICTED_SPARSE_WORDS_NEMO_THRESHOLD = Ref(256)
const RANK_RESTRICTED_WORDS_SMALL_WORDS = Ref(1)
const RANK_RESTRICTED_WORDS_SMALL_WORK_THRESHOLD = Ref(256)

@inline function _small_words_mask_regime(row_words::AbstractVector{UInt64},
                                          col_words::AbstractVector{UInt64},
                                          nr::Int,
                                          nc::Int)
    return length(row_words) <= RANK_RESTRICTED_WORDS_SMALL_WORDS[] &&
           length(col_words) <= RANK_RESTRICTED_WORDS_SMALL_WORDS[] &&
           nr * nc <= RANK_RESTRICTED_WORDS_SMALL_WORK_THRESHOLD[]
end

function _decode_words_to_indices_small(words::AbstractVector{UInt64}, nmax::Int, nactive::Int)
    out = Vector{Int}(undef, nactive)
    k = 0
    @inbounds for wd in eachindex(words)
        w = words[wd]
        while w != 0
            tz = trailing_zeros(w)
            idx = ((wd - 1) << 6) + tz + 1
            if idx <= nmax
                k += 1
                out[k] = idx
            end
            w &= w - UInt64(1)
        end
    end
    return out
end

function _decode_selected_cols_from_words(col_words::AbstractVector{UInt64},
                                          ncols::Int,
                                          pivs::AbstractVector{Int})
    npiv = length(pivs)
    npiv == 0 && return Int[]
    active_cols = Int[]
    @inbounds for wd in eachindex(col_words)
        w = col_words[wd]
        while w != 0
            tz = trailing_zeros(w)
            col = ((wd - 1) << 6) + tz + 1
            col <= ncols && push!(active_cols, col)
            w &= w - UInt64(1)
        end
    end
    out = Vector{Int}(undef, npiv)
    @inbounds for (k, piv) in enumerate(pivs)
        1 <= piv <= length(active_cols) || throw(BoundsError(active_cols, piv))
        out[k] = active_cols[piv]
    end
    return out
end

function _rank_restricted_words_small_dense(field::AbstractCoeffField,
                                            A::AbstractMatrix,
                                            row_words::AbstractVector{UInt64},
                                            col_words::AbstractVector{UInt64},
                                            nr::Int,
                                            nc::Int,
                                            nrows::Int,
                                            ncols::Int;
                                            backend::Symbol=:auto,
                                            kwargs...)
    rows = _decode_words_to_indices_small(row_words, nrows, nr)
    cols = _decode_words_to_indices_small(col_words, ncols, nc)
    return rank_restricted(field, A, rows, cols; backend=backend, kwargs...)
end

function _materialize_restricted_dense_from_words(A::AbstractMatrix{T},
                                                  row_words::AbstractVector{UInt64},
                                                  col_words::AbstractVector{UInt64},
                                                  nr::Int,
                                                  nc::Int,
                                                  nrows::Int,
                                                  ncols::Int;
                                                  check::Bool=false) where {T}
    if nr == 0 || nc == 0
        return Matrix{T}(undef, nr, nc)
    end
    B = Matrix{T}(undef, nr, nc)
    jloc = 0
    @inbounds for wj in eachindex(col_words)
        w = col_words[wj]
        while w != 0
            tz = trailing_zeros(w)
            col = ((wj - 1) << 6) + tz + 1
            if col <= ncols
                check && @assert 1 <= col <= size(A, 2)
                jloc += 1
                iloc = 0
                for wi in eachindex(row_words)
                    rw = row_words[wi]
                    while rw != 0
                        rtz = trailing_zeros(rw)
                        row = ((wi - 1) << 6) + rtz + 1
                        if row <= nrows
                            check && @assert 1 <= row <= size(A, 1)
                            iloc += 1
                            B[iloc, jloc] = A[row, col]
                        end
                        rw &= rw - 1
                    end
                end
            end
            w &= w - 1
        end
    end
    return B
end

function _rank_restricted_words_nemo_qq(A::AbstractMatrix{QQ},
                                        row_words::AbstractVector{UInt64},
                                        col_words::AbstractVector{UInt64},
                                        nr::Int,
                                        nc::Int,
                                        nrows::Int,
                                        ncols::Int;
                                        check::Bool=false)
    if nr == 0 || nc == 0
        return 0
    end
    M = Nemo.QQMatrix(nr, nc)
    jloc = 0
    @inbounds for wj in eachindex(col_words)
        w = col_words[wj]
        while w != 0
            tz = trailing_zeros(w)
            col = ((wj - 1) << 6) + tz + 1
            if col <= ncols
                check && @assert 1 <= col <= size(A, 2)
                jloc += 1
                iloc = 0
                for wi in eachindex(row_words)
                    rw = row_words[wi]
                    while rw != 0
                        rtz = trailing_zeros(rw)
                        row = ((wi - 1) << 6) + rtz + 1
                        if row <= nrows
                            check && @assert 1 <= row <= size(A, 1)
                            iloc += 1
                            M[iloc, jloc] = A[row, col]
                        end
                        rw &= rw - 1
                    end
                end
            end
            w &= w - 1
        end
    end
    return _nemo_rank_qq_mat(M)
end

function _build_row_locator_from_words(row_words::AbstractVector{UInt64},
                                       nrows::Int,
                                       nr::Int)
    if _use_rowpos_stamp(nrows, nr)
        pos = zeros(Int, nrows)
        marks = zeros(UInt32, nrows)
        tag = UInt32(1)
        iloc = 0
        @inbounds for wi in eachindex(row_words)
            rw = row_words[wi]
            while rw != 0
                rtz = trailing_zeros(rw)
                row = ((wi - 1) << 6) + rtz + 1
                if row <= nrows
                    iloc += 1
                    pos[row] = iloc
                    marks[row] = tag
                end
                rw &= rw - UInt64(1)
            end
        end
        return _RowLocatorStamp(pos, marks, tag)
    end
    rowpos = Dict{Int,Int}()
    sizehint!(rowpos, nr)
    iloc = 0
    @inbounds for wi in eachindex(row_words)
        rw = row_words[wi]
        while rw != 0
            rtz = trailing_zeros(rw)
            row = ((wi - 1) << 6) + rtz + 1
            if row <= nrows
                iloc += 1
                rowpos[row] = iloc
            end
            rw &= rw - UInt64(1)
        end
    end
    return _RowLocatorDict(rowpos)
end

function _rank_restricted_sparse_words_nemo_qq(A::SparseMatrixCSC{QQ,Int},
                                               row_words::AbstractVector{UInt64},
                                               col_words::AbstractVector{UInt64},
                                               nr::Int,
                                               nc::Int,
                                               nrows::Int,
                                               ncols::Int;
                                               check::Bool=false)
    if nr == 0 || nc == 0
        return 0
    end
    loc = _build_row_locator_from_words(row_words, nrows, nr)
    M = Nemo.QQMatrix(nr, nc)
    jloc = 0
    @inbounds for wj in eachindex(col_words)
        w = col_words[wj]
        while w != 0
            tz = trailing_zeros(w)
            col = ((wj - 1) << 6) + tz + 1
            if col <= ncols
                check && @assert 1 <= col <= size(A, 2)
                jloc += 1
                for ptr in A.colptr[col]:(A.colptr[col + 1] - 1)
                    row = A.rowval[ptr]
                    iloc = _row_lookup(loc, row)
                    if iloc != 0
                        M[iloc, jloc] = A.nzval[ptr]
                    end
                end
            end
            w &= w - UInt64(1)
        end
    end
    return _nemo_rank_qq_mat(M)
end

function _decode_words_to_indices(words::AbstractVector{UInt64}, nmax::Int)
    out = Int[]
    sizehint!(out, nmax)
    @inbounds for wd in eachindex(words)
        w = words[wd]
        while w != 0
            tz = trailing_zeros(w)
            idx = ((wd - 1) << 6) + tz + 1
            idx <= nmax && push!(out, idx)
            w &= w - 1
        end
    end
    return out
end

function rank_restricted_words(field::AbstractCoeffField,
                               A::AbstractMatrix,
                               row_words::AbstractVector{UInt64},
                               col_words::AbstractVector{UInt64},
                               nr::Int,
                               nc::Int;
                               nrows::Int=size(A, 1),
                               ncols::Int=size(A, 2),
                               backend::Symbol=:auto,
                               kwargs...)
    if nr == 0 || nc == 0 || !_active_words_nonempty(row_words) || !_active_words_nonempty(col_words)
        return 0
    end
    if A isa SparseMatrixCSC
        rows = _decode_words_to_indices(row_words, nrows)
        cols = _decode_words_to_indices(col_words, ncols)
        return rank_restricted(field, A, rows, cols; backend=backend, kwargs...)
    end
    if _small_words_mask_regime(row_words, col_words, nr, nc)
        return _rank_restricted_words_small_dense(field, A, row_words, col_words,
                                                  nr, nc, nrows, ncols;
                                                  backend=backend, kwargs...)
    end
    check = haskey(kwargs, :check) ? kwargs[:check] : false
    B = _materialize_restricted_dense_from_words(A, row_words, col_words, nr, nc, nrows, ncols; check=check)
    dense_kwargs = _drop_check_kw(kwargs)
    return rank(field, B; backend=backend, dense_kwargs...)
end

function rank_restricted_words(field::QQField,
                               A::AbstractMatrix{QQ},
                               row_words::AbstractVector{UInt64},
                               col_words::AbstractVector{UInt64},
                               nr::Int,
                               nc::Int;
                               nrows::Int=size(A, 1),
                               ncols::Int=size(A, 2),
                               backend::Symbol=:auto,
                               kwargs...)
    if nr == 0 || nc == 0 || !_active_words_nonempty(row_words) || !_active_words_nonempty(col_words)
        return 0
    end
    check = haskey(kwargs, :check) ? kwargs[:check] : false
    shape = _qq_shape_bucket(nr, nc)
    work = nr * nc

    if !(A isa SparseMatrixCSC{QQ,Int}) && _small_words_mask_regime(row_words, col_words, nr, nc)
        return _rank_restricted_words_small_dense(field, A, row_words, col_words,
                                                  nr, nc, nrows, ncols;
                                                  backend=backend, kwargs...)
    end

    if A isa SparseMatrixCSC{QQ,Int}
        sparseA = A::SparseMatrixCSC{QQ,Int}
        if backend == :nemo ||
           (backend == :auto && _have_nemo() &&
            work >= RANKQQ_RESTRICTED_SPARSE_WORDS_NEMO_THRESHOLD[])
            return _rank_restricted_sparse_words_nemo_qq(sparseA, row_words, col_words, nr, nc, nrows, ncols; check=check)
        end
    end

    if backend == :nemo ||
       (backend == :auto && _have_nemo() &&
        work >= min(_qq_nemo_threshold(:rank, shape), RANKQQ_RESTRICTED_WORDS_NEMO_THRESHOLD[]))
        return _rank_restricted_words_nemo_qq(A, row_words, col_words, nr, nc, nrows, ncols; check=check)
    end

    B = _materialize_restricted_dense_from_words(A, row_words, col_words, nr, nc, nrows, ncols; check=check)
    dim_backend = backend == :julia_exact ? :exact : backend
    if dim_backend == :auto || dim_backend == :exact || dim_backend == :modular
        return _rankQQ_dim(B; backend=dim_backend)
    end
    dense_kwargs = _drop_check_kw(kwargs)
    return rank(field, B; backend=backend, dense_kwargs...)
end

function colspace_restricted_words(field::AbstractCoeffField,
                                   A::AbstractMatrix,
                                   row_words::AbstractVector{UInt64},
                                   col_words::AbstractVector{UInt64},
                                   nr::Int,
                                   nc::Int;
                                   nrows::Int=size(A, 1),
                                   ncols::Int=size(A, 2),
                                   backend::Symbol=:auto,
                                   pivots::Bool=false,
                                   kwargs...)
    K = coeff_type(field)
    if nr == 0 || nc == 0 || !_active_words_nonempty(row_words) || !_active_words_nonempty(col_words)
        Z = zeros(K, nr, 0)
        return pivots ? (Z, Int[]) : Z
    end
    check = haskey(kwargs, :check) ? kwargs[:check] : false
    localA = if A isa SparseMatrixCSC
        rows = _decode_words_to_indices_small(row_words, nrows, nr)
        cols = _decode_words_to_indices_small(col_words, ncols, nc)
        _sparse_extract_restricted(A, rows, cols; check=check)
    else
        _materialize_restricted_dense_from_words(A, row_words, col_words, nr, nc, nrows, ncols; check=check)
    end
    C, local_pivs = _colspace_with_pivots(field, localA; backend=backend)
    basis_cols = _decode_selected_cols_from_words(col_words, ncols, local_pivs)
    return pivots ? (C, basis_cols) : C
end

@inline function _restricted_rhs_view(Y::AbstractVector, rows::AbstractVector{Int})
    return view(Y, rows)
end

@inline function _restricted_rhs_view(Y::AbstractMatrix, rows::AbstractVector{Int})
    return view(Y, rows, :)
end

@inline function _drop_check_kw(kwargs::NamedTuple)
    return (; (k => v for (k, v) in pairs(kwargs) if k != :check)...)
end
@inline _drop_check_kw(kwargs::Base.Pairs) = _drop_check_kw(values(kwargs))

function nullspace_restricted(field::AbstractCoeffField, A::SparseMatrixCSC,
                              rows::AbstractVector{Int}, cols::AbstractVector{Int};
                              backend::Symbol=:auto, kwargs...)
    nr = length(rows)
    nc = length(cols)
    K = coeff_type(field)
    if nc == 0
        return zeros(K, 0, 0)
    end
    if nr == 0
        return eye(field, nc)
    end
    check = haskey(kwargs, :check) ? kwargs[:check] : false
    S = _sparse_extract_restricted(A, rows, cols; check=check)
    return nullspace(field, S; backend=backend)
end

function nullspace_restricted(field::AbstractCoeffField, A::AbstractMatrix,
                              rows::AbstractVector{Int}, cols::AbstractVector{Int};
                              backend::Symbol=:auto, kwargs...)
    A isa SparseMatrixCSC && return nullspace_restricted(field, A, rows, cols; backend=backend, kwargs...)
    nr = length(rows)
    nc = length(cols)
    K = coeff_type(field)
    if nc == 0
        return zeros(K, 0, 0)
    end
    if nr == 0
        return eye(field, nc)
    end
    return nullspace(field, view(A, rows, cols); backend=backend)
end

function solve_fullcolumn_restricted(field::AbstractCoeffField, B::SparseMatrixCSC,
                                     rows::AbstractVector{Int}, cols::AbstractVector{Int},
                                     Y::AbstractVecOrMat;
                                     check_rhs::Bool=true,
                                     backend::Symbol=:auto,
                                     kwargs...)
    nr = length(rows)
    nc = length(cols)
    if nc == 0
        rhs = Y isa AbstractVector ? 1 : size(Y, 2)
        K = coeff_type(field)
        Z = zeros(K, 0, rhs)
        return Y isa AbstractVector ? vec(Z) : Z
    end
    nr == 0 && error("solve_fullcolumn_restricted: cannot solve with zero selected rows and nonzero columns")
    check = haskey(kwargs, :check) ? kwargs[:check] : false
    kws = _drop_check_kw(kwargs)
    Bsub = _sparse_extract_restricted(B, rows, cols; check=check)
    Ysub = _restricted_rhs_view(Y, rows)
    return solve_fullcolumn(field, Bsub, Ysub; check_rhs=check_rhs, backend=backend, kws...)
end

function solve_fullcolumn_restricted(field::AbstractCoeffField, B::AbstractMatrix,
                                     rows::AbstractVector{Int}, cols::AbstractVector{Int},
                                     Y::AbstractVecOrMat;
                                     check_rhs::Bool=true,
                                     backend::Symbol=:auto,
                                     kwargs...)
    B isa SparseMatrixCSC && return solve_fullcolumn_restricted(field, B, rows, cols, Y;
                                                                check_rhs=check_rhs,
                                                                backend=backend, kwargs...)
    nr = length(rows)
    nc = length(cols)
    if nc == 0
        rhs = Y isa AbstractVector ? 1 : size(Y, 2)
        K = coeff_type(field)
        Z = zeros(K, 0, rhs)
        return Y isa AbstractVector ? vec(Z) : Z
    end
    nr == 0 && error("solve_fullcolumn_restricted: cannot solve with zero selected rows and nonzero columns")
    Bsub = view(B, rows, cols)
    Ysub = _restricted_rhs_view(Y, rows)
    kws = _drop_check_kw(kwargs)
    return solve_fullcolumn(field, Bsub, Ysub; check_rhs=check_rhs, backend=backend, kws...)
end

function solve_fullcolumn_restricted_words(field::AbstractCoeffField,
                                           B::AbstractMatrix,
                                           Y::AbstractMatrix,
                                           row_words::AbstractVector{UInt64},
                                           col_words::AbstractVector{UInt64},
                                           nr::Int,
                                           nc::Int;
                                           nrows::Int=size(Y, 1),
                                           ncols::Int=size(Y, 2),
                                           check_rhs::Bool=true,
                                           backend::Symbol=:auto,
                                           kwargs...)
    K = coeff_type(field)
    size(B, 1) == nr || error("solve_fullcolumn_restricted_words: expected size(B,1) == nr")
    if nc == 0 || !_active_words_nonempty(col_words)
        return zeros(K, size(B, 2), 0)
    end
    nr == 0 && error("solve_fullcolumn_restricted_words: cannot solve with zero selected rows and nonzero columns")
    check = haskey(kwargs, :check) ? kwargs[:check] : false
    Ysub = if Y isa SparseMatrixCSC
        rows = _decode_words_to_indices_small(row_words, nrows, nr)
        cols = _decode_words_to_indices_small(col_words, ncols, nc)
        _sparse_extract_restricted(Y, rows, cols; check=check)
    else
        _materialize_restricted_dense_from_words(Y, row_words, col_words, nr, nc, nrows, ncols; check=check)
    end
    kws = _drop_check_kw(kwargs)
    return solve_fullcolumn(field, B, Ysub; check_rhs=check_rhs, backend=backend, kws...)
end

function __init__()
    _LINALG_THRESHOLDS_INITIALIZED[] && return
    path = _linalg_thresholds_path()
    loaded = _load_linalg_thresholds!(; path=path, warn_on_mismatch=true)
    if !loaded && !isfile(path)
        try
            autotune_linalg_thresholds!(; path=path, save=true, quiet=true, profile=:startup)
        catch err
            @warn "FieldLinAlg: startup autotune failed; using defaults." path exception=(err, catch_backtrace())
        end
    end
    _LINALG_THRESHOLDS_INITIALIZED[] = true
    return nothing
end
