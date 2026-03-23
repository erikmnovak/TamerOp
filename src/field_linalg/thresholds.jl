# -----------------------------------------------------------------------------
# field_linalg/thresholds.jl
#
# Scope:
#   Threshold state, backend-trait plumbing, tiny-kernel heuristics, threshold
#   persistence helpers, and shared configuration for TamerOp.FieldLinAlg.
# Owns:
#   - global threshold refs and schema versioning,
#   - backend trait definitions shared by all kernels,
#   - threshold load/save/apply helpers.
# Does not own:
#   - backend choice logic (`backend_routing.jl`),
#   - QQ/non-QQ kernel implementations,
#   - public API wrappers.
# Depends on:
#   - no later private fragment; this file is loaded first and provides shared
#     state consumed by the rest of FieldLinAlg.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Backend selection + Nemo hooks
# -----------------------------------------------------------------------------

# Threshold semantics by operation:
# - `*_RANK_*` thresholds govern backend crossover for rank and rank-like
#   restricted helpers.
# - `*_NULLSPACE_*` thresholds govern nullspace backend crossover.
# - `*_COLSPACE_*` logic is expressed through routing helpers below rather than
#   a large standalone threshold family.
# - `*_SOLVE_*` thresholds govern full-column solve backend crossover, including
#   sparse-shape/density buckets.
# - `RANKQQ_RESTRICTED_WORDS_NEMO_THRESHOLD` and
#   `ZN_QQ_DIMAT_SUBMATRIX_WORK_THRESHOLD` are restricted/words-path controls
#   used by Flange/Zn/derived kernels that call into FieldLinAlg.

const _NEMO_ENABLED = Ref(true)
_have_nemo() = _NEMO_ENABLED[]

const NEMO_THRESHOLD = Ref(50_000)     # heuristic: use Nemo if m*n >= threshold
const QQ_NEMO_RANK_THRESHOLD_SQUARE = Ref(50_000)
const QQ_NEMO_RANK_THRESHOLD_TALL = Ref(50_000)
const QQ_NEMO_RANK_THRESHOLD_WIDE = Ref(50_000)
const QQ_NEMO_NULLSPACE_THRESHOLD_SQUARE = Ref(50_000)
const QQ_NEMO_NULLSPACE_THRESHOLD_TALL = Ref(50_000)
const QQ_NEMO_NULLSPACE_THRESHOLD_WIDE = Ref(50_000)
const QQ_NEMO_SOLVE_THRESHOLD_SQUARE = Ref(50_000)
const QQ_NEMO_SOLVE_THRESHOLD_TALL = Ref(50_000)
const QQ_NEMO_SOLVE_THRESHOLD_WIDE = Ref(50_000)
const QQ_NEMO_SPARSE_SOLVE_THRESHOLD_SQUARE_LOW = Ref(10_000)
const QQ_NEMO_SPARSE_SOLVE_THRESHOLD_SQUARE_MID = Ref(3_000)
const QQ_NEMO_SPARSE_SOLVE_THRESHOLD_SQUARE_HIGH = Ref(1_200)
const QQ_NEMO_SPARSE_SOLVE_THRESHOLD_TALL_LOW = Ref(10_000)
const QQ_NEMO_SPARSE_SOLVE_THRESHOLD_TALL_MID = Ref(3_000)
const QQ_NEMO_SPARSE_SOLVE_THRESHOLD_TALL_HIGH = Ref(1_200)
const QQ_NEMO_SPARSE_SOLVE_THRESHOLD_WIDE_LOW = Ref(10_000)
const QQ_NEMO_SPARSE_SOLVE_THRESHOLD_WIDE_MID = Ref(3_000)
const QQ_NEMO_SPARSE_SOLVE_THRESHOLD_WIDE_HIGH = Ref(1_200)
const QQ_NEMO_SPARSE_SOLVE_POLICY_SQUARE_LOW = Ref(1)   # 1 => use nnz >= threshold, -1 => use nnz <= threshold
const QQ_NEMO_SPARSE_SOLVE_POLICY_SQUARE_MID = Ref(1)
const QQ_NEMO_SPARSE_SOLVE_POLICY_SQUARE_HIGH = Ref(1)
const QQ_NEMO_SPARSE_SOLVE_POLICY_TALL_LOW = Ref(1)
const QQ_NEMO_SPARSE_SOLVE_POLICY_TALL_MID = Ref(1)
const QQ_NEMO_SPARSE_SOLVE_POLICY_TALL_HIGH = Ref(1)
const QQ_NEMO_SPARSE_SOLVE_POLICY_WIDE_LOW = Ref(1)
const QQ_NEMO_SPARSE_SOLVE_POLICY_WIDE_MID = Ref(1)
const QQ_NEMO_SPARSE_SOLVE_POLICY_WIDE_HIGH = Ref(1)
const QQ_MODULAR_NULLSPACE_THRESHOLD_SQUARE = Ref(120_000)
const QQ_MODULAR_NULLSPACE_THRESHOLD_TALL = Ref(120_000)
const QQ_MODULAR_NULLSPACE_THRESHOLD_WIDE = Ref(120_000)
const QQ_MODULAR_SOLVE_THRESHOLD_SQUARE = Ref(120_000)
const QQ_MODULAR_SOLVE_THRESHOLD_TALL = Ref(120_000)
const QQ_MODULAR_SOLVE_THRESHOLD_WIDE = Ref(120_000)
const FP_NEMO_RANK_THRESHOLD = Ref(40_000)
const FP_NEMO_NULLSPACE_THRESHOLD = Ref(60_000)
const FP_NEMO_SOLVE_THRESHOLD = Ref(60_000)
const FLOAT_NULLSPACE_SVD_THRESHOLD = Ref(180_000)
const FLOAT_SPARSE_SVDS_MIN_DIM = Ref(1_024)
const FLOAT_SPARSE_SVDS_MIN_NNZ = Ref(120_000)
const ZN_QQ_DIMAT_SUBMATRIX_WORK_THRESHOLD = Ref(48)
const MODULAR_NULLSPACE_THRESHOLD = Ref(120_000)
const MODULAR_SOLVE_THRESHOLD = Ref(120_000)
const MODULAR_MIN_PRIMES = Ref(2)
const MODULAR_MAX_PRIMES = Ref(6)
const RANKQQ_DIM_SMALL_THRESHOLD = Ref(20_000)
const RANKQQ_RESTRICTED_WORDS_NEMO_THRESHOLD = Ref(16)
const ROWPOS_STAMP_MIN_ROWS = Ref(128)
const ROWPOS_STAMP_DENSITY_NUM = Ref(1) # use stamp when nr/m >= 1/5
const ROWPOS_STAMP_DENSITY_DEN = Ref(5)
const TINY_LINALG_MAX_DIM = Ref(4)
const LINALG_THRESHOLD_SCHEMA_VERSION = 1
const _LINALG_THRESHOLDS_INITIALIZED = Ref(false)

@inline zn_qq_dimat_submatrix_work_threshold() = Int(ZN_QQ_DIMAT_SUBMATRIX_WORK_THRESHOLD[])
@inline function _set_zn_qq_dimat_submatrix_work_threshold!(x::Integer)
    ZN_QQ_DIMAT_SUBMATRIX_WORK_THRESHOLD[] = max(1, Int(x))
    return nothing
end

abstract type MatrixBackendTrait end
struct JuliaMatrixBackend <: MatrixBackendTrait end
struct NemoMatrixBackend <: MatrixBackendTrait end

const _JULIA_BACKEND_TRAIT = JuliaMatrixBackend()
const _NEMO_BACKEND_TRAIT = NemoMatrixBackend()
const _SVDS_IMPL = Ref{Any}(nothing)

@inline function _matrix_backend_trait(field::AbstractCoeffField, A;
                                      op::Symbol=:rank, backend::Symbol=:auto)
    be = _choose_linalg_backend(field, A; op=op, backend=backend)
    return be == :nemo ? _NEMO_BACKEND_TRAIT : _JULIA_BACKEND_TRAIT
end

const _QQ_TO_NEMO_CONVERSIONS = Base.Threads.Atomic{Int}(0)
const _QQ_TO_NEMO_CACHE_HITS = Base.Threads.Atomic{Int}(0)
const _QQ_FROM_NEMO_CONVERSIONS = Base.Threads.Atomic{Int}(0)
const _FP_TO_NEMO_CONVERSIONS = Base.Threads.Atomic{Int}(0)
const _FP_TO_NEMO_CACHE_HITS = Base.Threads.Atomic{Int}(0)
const _FP_FROM_NEMO_CONVERSIONS = Base.Threads.Atomic{Int}(0)

@inline _bump_counter!(x::Base.Threads.Atomic{Int}) = Base.Threads.atomic_add!(x, 1)

function _reset_conversion_counters!()
    _QQ_TO_NEMO_CONVERSIONS[] = 0
    _QQ_TO_NEMO_CACHE_HITS[] = 0
    _QQ_FROM_NEMO_CONVERSIONS[] = 0
    _FP_TO_NEMO_CONVERSIONS[] = 0
    _FP_TO_NEMO_CACHE_HITS[] = 0
    _FP_FROM_NEMO_CONVERSIONS[] = 0
    return nothing
end

function _conversion_counters()
    return (
        qq_to_nemo=_QQ_TO_NEMO_CONVERSIONS[],
        qq_to_nemo_cache_hits=_QQ_TO_NEMO_CACHE_HITS[],
        qq_from_nemo=_QQ_FROM_NEMO_CONVERSIONS[],
        fp_to_nemo=_FP_TO_NEMO_CONVERSIONS[],
        fp_to_nemo_cache_hits=_FP_TO_NEMO_CACHE_HITS[],
        fp_from_nemo=_FP_FROM_NEMO_CONVERSIONS[],
    )
end

@inline _have_svds_backend() = _SVDS_IMPL[] !== nothing

function _set_svds_impl!(impl)
    _SVDS_IMPL[] = impl
    return nothing
end

function _nullspace_float_svds(F::RealField, A)
    impl = _SVDS_IMPL[]
    impl === nothing && return nothing
    return impl(F, A)
end

@inline _nemo_dense_compatible(A) = A isa StridedMatrix
@inline _nemo_dense_compatible(::BackendMatrix) = true
@inline _is_sparse_like(A) =
    A isa SparseMatrixCSC ||
    A isa Transpose{<:Any,<:SparseMatrixCSC} ||
    A isa Adjoint{<:Any,<:SparseMatrixCSC}
@inline _sparse_parent(A::SparseMatrixCSC) = A
@inline _sparse_parent(A::Transpose{<:Any,<:SparseMatrixCSC}) = parent(A)
@inline _sparse_parent(A::Adjoint{<:Any,<:SparseMatrixCSC}) = parent(A)
@inline _sparse_nnz(A) = nnz(_sparse_parent(A))
@inline _use_rowpos_stamp(m::Int, nr::Int) =
    nr >= ROWPOS_STAMP_MIN_ROWS[] && nr * ROWPOS_STAMP_DENSITY_DEN[] >= m * ROWPOS_STAMP_DENSITY_NUM[]
@inline _is_tiny_dim(x::Int) = x <= TINY_LINALG_MAX_DIM[]
@inline _is_tiny_matrix_dims(m::Int, n::Int) = _is_tiny_dim(m) && _is_tiny_dim(n)
@inline _is_tiny_matrix(A::AbstractMatrix) = _is_tiny_matrix_dims(size(A, 1), size(A, 2))
@inline function _qq_shape_bucket(m::Int, n::Int)::Symbol
    # Bucket by aspect ratio to keep backend routing stable across square/tall/wide regimes.
    if m * 5 >= n * 4 && n * 5 >= m * 4
        return :square
    elseif m > n
        return :tall
    else
        return :wide
    end
end

@inline function _qq_nemo_threshold(op::Symbol, shape::Symbol)::Int
    if op == :rank
        return shape == :square ? QQ_NEMO_RANK_THRESHOLD_SQUARE[] :
               shape == :tall ? QQ_NEMO_RANK_THRESHOLD_TALL[] :
               QQ_NEMO_RANK_THRESHOLD_WIDE[]
    elseif op == :nullspace
        return shape == :square ? QQ_NEMO_NULLSPACE_THRESHOLD_SQUARE[] :
               shape == :tall ? QQ_NEMO_NULLSPACE_THRESHOLD_TALL[] :
               QQ_NEMO_NULLSPACE_THRESHOLD_WIDE[]
    elseif op == :solve
        return shape == :square ? QQ_NEMO_SOLVE_THRESHOLD_SQUARE[] :
               shape == :tall ? QQ_NEMO_SOLVE_THRESHOLD_TALL[] :
               QQ_NEMO_SOLVE_THRESHOLD_WIDE[]
    end
    return NEMO_THRESHOLD[]
end

@inline function _qq_modular_threshold(op::Symbol, shape::Symbol)::Int
    if op == :nullspace
        return shape == :square ? QQ_MODULAR_NULLSPACE_THRESHOLD_SQUARE[] :
               shape == :tall ? QQ_MODULAR_NULLSPACE_THRESHOLD_TALL[] :
               QQ_MODULAR_NULLSPACE_THRESHOLD_WIDE[]
    elseif op == :solve
        return shape == :square ? QQ_MODULAR_SOLVE_THRESHOLD_SQUARE[] :
               shape == :tall ? QQ_MODULAR_SOLVE_THRESHOLD_TALL[] :
               QQ_MODULAR_SOLVE_THRESHOLD_WIDE[]
    end
    return max(MODULAR_SOLVE_THRESHOLD[], MODULAR_NULLSPACE_THRESHOLD[])
end

@inline function _qq_sparse_density_bucket(dens::Float64)::Symbol
    if dens <= 0.08
        return :low
    elseif dens <= 0.28
        return :mid
    end
    return :high
end

@inline _qq_sparse_density_bucket(A) = _qq_sparse_density_bucket((size(A, 1) * size(A, 2)) == 0 ? 0.0 : (_sparse_nnz(A) / (size(A, 1) * size(A, 2))))

@inline function _qq_sparse_solve_fill(A)::Float64
    m, n = size(A)
    nnzA = _sparse_nnz(A)
    if m > n
        free_slots = (m - n) * n
        free_slots <= 0 && return 1.0
        return clamp((nnzA - n) / free_slots, 0.0, 1.0)
    end
    work = m * n
    work == 0 && return 0.0
    return clamp(nnzA / work, 0.0, 1.0)
end

@inline function _qq_sparse_solve_threshold(shape::Symbol, bucket::Symbol)::Int
    if shape == :square
        return bucket == :low ? QQ_NEMO_SPARSE_SOLVE_THRESHOLD_SQUARE_LOW[] :
               bucket == :mid ? QQ_NEMO_SPARSE_SOLVE_THRESHOLD_SQUARE_MID[] :
               QQ_NEMO_SPARSE_SOLVE_THRESHOLD_SQUARE_HIGH[]
    elseif shape == :tall
        return bucket == :low ? QQ_NEMO_SPARSE_SOLVE_THRESHOLD_TALL_LOW[] :
               bucket == :mid ? QQ_NEMO_SPARSE_SOLVE_THRESHOLD_TALL_MID[] :
               QQ_NEMO_SPARSE_SOLVE_THRESHOLD_TALL_HIGH[]
    end
    return bucket == :low ? QQ_NEMO_SPARSE_SOLVE_THRESHOLD_WIDE_LOW[] :
           bucket == :mid ? QQ_NEMO_SPARSE_SOLVE_THRESHOLD_WIDE_MID[] :
           QQ_NEMO_SPARSE_SOLVE_THRESHOLD_WIDE_HIGH[]
end

@inline function _qq_sparse_solve_use_ge(shape::Symbol, bucket::Symbol)::Bool
    if shape == :square
        return (bucket == :low ? QQ_NEMO_SPARSE_SOLVE_POLICY_SQUARE_LOW[] :
                bucket == :mid ? QQ_NEMO_SPARSE_SOLVE_POLICY_SQUARE_MID[] :
                QQ_NEMO_SPARSE_SOLVE_POLICY_SQUARE_HIGH[]) > 0
    elseif shape == :tall
        return (bucket == :low ? QQ_NEMO_SPARSE_SOLVE_POLICY_TALL_LOW[] :
                bucket == :mid ? QQ_NEMO_SPARSE_SOLVE_POLICY_TALL_MID[] :
                QQ_NEMO_SPARSE_SOLVE_POLICY_TALL_HIGH[]) > 0
    end
    return (bucket == :low ? QQ_NEMO_SPARSE_SOLVE_POLICY_WIDE_LOW[] :
            bucket == :mid ? QQ_NEMO_SPARSE_SOLVE_POLICY_WIDE_MID[] :
            QQ_NEMO_SPARSE_SOLVE_POLICY_WIDE_HIGH[]) > 0
end

@inline function _qq_sparse_solve_use_julia(_shape::Symbol, dens::Float64, nnzA::Int)::Bool
    dens >= 0.10 && return false
    # Sparse QQ solve_fullcolumn has a small-nnz regime where the Julia sparse
    # factor path beats Nemo even when the autotuned bucket thresholds would
    # otherwise promote to Nemo.
    return nnzA <= 2_048
end

@inline function _qq_sparse_rank_use_julia(shape::Symbol, dens::Float64, nnzA::Int)::Bool
    dens >= 0.10 && return false
    # Sparse QQ rank cost tracks nonzero structure more closely than ambient
    # matrix area; keep the Julia sparse engine on moderate-nnz inputs even
    # when the dense-shape Nemo thresholds would otherwise fire.
    return nnzA <= max(RANKQQ_DIM_SMALL_THRESHOLD[], _qq_nemo_threshold(:rank, shape))
end

@inline function _qq_sparse_nullspace_use_julia(shape::Symbol, dens::Float64, nnzA::Int)::Bool
    dens >= 0.10 && return false
    # Sparse QQ nullspace is much more sensitive to elimination growth than
    # rank; only keep the Julia sparse engine on genuinely tiny-nnz inputs.
    return nnzA <= min(_qq_nemo_threshold(:nullspace, shape), max(512, fld(RANKQQ_DIM_SMALL_THRESHOLD[], 4)))
end

@inline function _qq_sparse_colspace_use_julia(_shape::Symbol, dens::Float64, nnzA::Int, _work::Int)::Bool
    dens >= 0.10 && return false
    # Sparse QQ colspace relies on pivot-column extraction, which tends to
    # benefit from Nemo much earlier than pure rank, but extremely sparse
    # matrices still favor the Julia sparse engine.
    return nnzA <= 2_048
end
@inline function _is_tiny_solve(B, Y)
    rhs = Y isa AbstractVector ? 1 : size(Y, 2)
    return _is_tiny_matrix_dims(size(B, 1), size(B, 2)) && _is_tiny_dim(rhs)
end
@inline function _is_tiny_mul(A::AbstractMatrix, B::AbstractMatrix)
    m, k = size(A)
    k2, n = size(B)
    return k == k2 && _is_tiny_matrix_dims(m, k) && _is_tiny_matrix_dims(k2, n)
end

function _matmul_tiny(A::AbstractMatrix{TA}, B::AbstractMatrix{TB}) where {TA,TB}
    m, k = size(A)
    k2, n = size(B)
    k == k2 || throw(DimensionMismatch("A and B inner dimensions must match"))
    T = promote_type(TA, TB)
    C = zeros(T, m, n)
    @inbounds for j in 1:n
        for t in 1:k
            bt = B[t, j]
            iszero(bt) && continue
            for i in 1:m
                C[i, j] += A[i, t] * bt
            end
        end
    end
    return C
end

@inline function _matmul(A::AbstractMatrix, B::AbstractMatrix)
    if _is_tiny_mul(A, B)
        return _matmul_tiny(A, B)
    end
    return A * B
end

struct _RowLocatorDict
    rowpos::Dict{Int,Int}
end

struct _RowLocatorStamp
    pos::Vector{Int}
    marks::Vector{UInt32}
    tag::UInt32
end

function _build_row_locator(rows::AbstractVector{Int}, m::Int)
    nr = length(rows)
    if _use_rowpos_stamp(m, nr)
        pos = zeros(Int, m)
        marks = zeros(UInt32, m)
        tag = UInt32(1)
        @inbounds for (i, r) in enumerate(rows)
            pos[r] = i
            marks[r] = tag
        end
        return _RowLocatorStamp(pos, marks, tag)
    end
    rowpos = Dict{Int,Int}()
    sizehint!(rowpos, nr)
    @inbounds for (i, r) in enumerate(rows)
        rowpos[r] = i
    end
    return _RowLocatorDict(rowpos)
end

@inline _row_lookup(loc::_RowLocatorDict, r::Int) = get(loc.rowpos, r, 0)
@inline _row_lookup(loc::_RowLocatorStamp, r::Int) = @inbounds (loc.marks[r] == loc.tag ? loc.pos[r] : 0)

@inline _linalg_repo_root() = normpath(joinpath(@__DIR__, "..", ".."))

function _linalg_thresholds_path(; root::AbstractString=_linalg_repo_root())
    return joinpath(root, "linalg_thresholds.toml")
end

function _current_linalg_fingerprint()
    return Dict(
        "schema_version" => LINALG_THRESHOLD_SCHEMA_VERSION,
        "cpu_name" => String(Sys.CPU_NAME),
        "cpu_threads" => Int(Sys.CPU_THREADS),
        "julia_version" => string(VERSION),
        "word_size" => Int(Sys.WORD_SIZE),
        "os" => string(Sys.KERNEL),
        "blas_threads" => Int(LinearAlgebra.BLAS.get_num_threads()),
    )
end

function _current_linalg_thresholds()
    return Dict(
        "nemo_threshold" => Int(NEMO_THRESHOLD[]),
        "qq_nemo_rank_threshold_square" => Int(QQ_NEMO_RANK_THRESHOLD_SQUARE[]),
        "qq_nemo_rank_threshold_tall" => Int(QQ_NEMO_RANK_THRESHOLD_TALL[]),
        "qq_nemo_rank_threshold_wide" => Int(QQ_NEMO_RANK_THRESHOLD_WIDE[]),
        "qq_nemo_nullspace_threshold_square" => Int(QQ_NEMO_NULLSPACE_THRESHOLD_SQUARE[]),
        "qq_nemo_nullspace_threshold_tall" => Int(QQ_NEMO_NULLSPACE_THRESHOLD_TALL[]),
        "qq_nemo_nullspace_threshold_wide" => Int(QQ_NEMO_NULLSPACE_THRESHOLD_WIDE[]),
        "qq_nemo_solve_threshold_square" => Int(QQ_NEMO_SOLVE_THRESHOLD_SQUARE[]),
        "qq_nemo_solve_threshold_tall" => Int(QQ_NEMO_SOLVE_THRESHOLD_TALL[]),
        "qq_nemo_solve_threshold_wide" => Int(QQ_NEMO_SOLVE_THRESHOLD_WIDE[]),
        "qq_nemo_sparse_solve_threshold_square_low" => Int(QQ_NEMO_SPARSE_SOLVE_THRESHOLD_SQUARE_LOW[]),
        "qq_nemo_sparse_solve_threshold_square_mid" => Int(QQ_NEMO_SPARSE_SOLVE_THRESHOLD_SQUARE_MID[]),
        "qq_nemo_sparse_solve_threshold_square_high" => Int(QQ_NEMO_SPARSE_SOLVE_THRESHOLD_SQUARE_HIGH[]),
        "qq_nemo_sparse_solve_threshold_tall_low" => Int(QQ_NEMO_SPARSE_SOLVE_THRESHOLD_TALL_LOW[]),
        "qq_nemo_sparse_solve_threshold_tall_mid" => Int(QQ_NEMO_SPARSE_SOLVE_THRESHOLD_TALL_MID[]),
        "qq_nemo_sparse_solve_threshold_tall_high" => Int(QQ_NEMO_SPARSE_SOLVE_THRESHOLD_TALL_HIGH[]),
        "qq_nemo_sparse_solve_threshold_wide_low" => Int(QQ_NEMO_SPARSE_SOLVE_THRESHOLD_WIDE_LOW[]),
        "qq_nemo_sparse_solve_threshold_wide_mid" => Int(QQ_NEMO_SPARSE_SOLVE_THRESHOLD_WIDE_MID[]),
        "qq_nemo_sparse_solve_threshold_wide_high" => Int(QQ_NEMO_SPARSE_SOLVE_THRESHOLD_WIDE_HIGH[]),
        "qq_nemo_sparse_solve_policy_square_low" => Int(QQ_NEMO_SPARSE_SOLVE_POLICY_SQUARE_LOW[]),
        "qq_nemo_sparse_solve_policy_square_mid" => Int(QQ_NEMO_SPARSE_SOLVE_POLICY_SQUARE_MID[]),
        "qq_nemo_sparse_solve_policy_square_high" => Int(QQ_NEMO_SPARSE_SOLVE_POLICY_SQUARE_HIGH[]),
        "qq_nemo_sparse_solve_policy_tall_low" => Int(QQ_NEMO_SPARSE_SOLVE_POLICY_TALL_LOW[]),
        "qq_nemo_sparse_solve_policy_tall_mid" => Int(QQ_NEMO_SPARSE_SOLVE_POLICY_TALL_MID[]),
        "qq_nemo_sparse_solve_policy_tall_high" => Int(QQ_NEMO_SPARSE_SOLVE_POLICY_TALL_HIGH[]),
        "qq_nemo_sparse_solve_policy_wide_low" => Int(QQ_NEMO_SPARSE_SOLVE_POLICY_WIDE_LOW[]),
        "qq_nemo_sparse_solve_policy_wide_mid" => Int(QQ_NEMO_SPARSE_SOLVE_POLICY_WIDE_MID[]),
        "qq_nemo_sparse_solve_policy_wide_high" => Int(QQ_NEMO_SPARSE_SOLVE_POLICY_WIDE_HIGH[]),
        "qq_modular_nullspace_threshold_square" => Int(QQ_MODULAR_NULLSPACE_THRESHOLD_SQUARE[]),
        "qq_modular_nullspace_threshold_tall" => Int(QQ_MODULAR_NULLSPACE_THRESHOLD_TALL[]),
        "qq_modular_nullspace_threshold_wide" => Int(QQ_MODULAR_NULLSPACE_THRESHOLD_WIDE[]),
        "qq_modular_solve_threshold_square" => Int(QQ_MODULAR_SOLVE_THRESHOLD_SQUARE[]),
        "qq_modular_solve_threshold_tall" => Int(QQ_MODULAR_SOLVE_THRESHOLD_TALL[]),
        "qq_modular_solve_threshold_wide" => Int(QQ_MODULAR_SOLVE_THRESHOLD_WIDE[]),
        "fp_nemo_rank_threshold" => Int(FP_NEMO_RANK_THRESHOLD[]),
        "fp_nemo_nullspace_threshold" => Int(FP_NEMO_NULLSPACE_THRESHOLD[]),
        "fp_nemo_solve_threshold" => Int(FP_NEMO_SOLVE_THRESHOLD[]),
        "modular_nullspace_threshold" => Int(MODULAR_NULLSPACE_THRESHOLD[]),
        "modular_solve_threshold" => Int(MODULAR_SOLVE_THRESHOLD[]),
        "modular_min_primes" => Int(MODULAR_MIN_PRIMES[]),
        "modular_max_primes" => Int(MODULAR_MAX_PRIMES[]),
        "rankqq_dim_small_threshold" => Int(RANKQQ_DIM_SMALL_THRESHOLD[]),
        "rankqq_restricted_words_nemo_threshold" => Int(RANKQQ_RESTRICTED_WORDS_NEMO_THRESHOLD[]),
        "float_nullspace_svd_threshold" => Int(FLOAT_NULLSPACE_SVD_THRESHOLD[]),
        "float_sparse_svds_min_dim" => Int(FLOAT_SPARSE_SVDS_MIN_DIM[]),
        "float_sparse_svds_min_nnz" => Int(FLOAT_SPARSE_SVDS_MIN_NNZ[]),
        "zn_qq_dimat_submatrix_work_threshold" => Int(ZN_QQ_DIMAT_SUBMATRIX_WORK_THRESHOLD[]),
    )
end

function _apply_linalg_thresholds!(vals)::Bool
    try
        qq_nemo_fallback = Int(get(vals, "nemo_threshold", NEMO_THRESHOLD[]))
        NEMO_THRESHOLD[] = qq_nemo_fallback
        QQ_NEMO_RANK_THRESHOLD_SQUARE[] = Int(get(vals, "qq_nemo_rank_threshold_square", qq_nemo_fallback))
        QQ_NEMO_RANK_THRESHOLD_TALL[] = Int(get(vals, "qq_nemo_rank_threshold_tall", qq_nemo_fallback))
        QQ_NEMO_RANK_THRESHOLD_WIDE[] = Int(get(vals, "qq_nemo_rank_threshold_wide", qq_nemo_fallback))
        QQ_NEMO_NULLSPACE_THRESHOLD_SQUARE[] = Int(get(vals, "qq_nemo_nullspace_threshold_square", qq_nemo_fallback))
        QQ_NEMO_NULLSPACE_THRESHOLD_TALL[] = Int(get(vals, "qq_nemo_nullspace_threshold_tall", qq_nemo_fallback))
        QQ_NEMO_NULLSPACE_THRESHOLD_WIDE[] = Int(get(vals, "qq_nemo_nullspace_threshold_wide", qq_nemo_fallback))
        QQ_NEMO_SOLVE_THRESHOLD_SQUARE[] = Int(get(vals, "qq_nemo_solve_threshold_square", qq_nemo_fallback))
        QQ_NEMO_SOLVE_THRESHOLD_TALL[] = Int(get(vals, "qq_nemo_solve_threshold_tall", qq_nemo_fallback))
        QQ_NEMO_SOLVE_THRESHOLD_WIDE[] = Int(get(vals, "qq_nemo_solve_threshold_wide", qq_nemo_fallback))
        qq_sparse_solve_square_fallback = QQ_NEMO_SOLVE_THRESHOLD_SQUARE[]
        qq_sparse_solve_tall_fallback = QQ_NEMO_SOLVE_THRESHOLD_TALL[]
        qq_sparse_solve_wide_fallback = QQ_NEMO_SOLVE_THRESHOLD_WIDE[]
        QQ_NEMO_SPARSE_SOLVE_THRESHOLD_SQUARE_LOW[] = Int(get(vals, "qq_nemo_sparse_solve_threshold_square_low", qq_sparse_solve_square_fallback))
        QQ_NEMO_SPARSE_SOLVE_THRESHOLD_SQUARE_MID[] = Int(get(vals, "qq_nemo_sparse_solve_threshold_square_mid", qq_sparse_solve_square_fallback))
        QQ_NEMO_SPARSE_SOLVE_THRESHOLD_SQUARE_HIGH[] = Int(get(vals, "qq_nemo_sparse_solve_threshold_square_high", qq_sparse_solve_square_fallback))
        QQ_NEMO_SPARSE_SOLVE_THRESHOLD_TALL_LOW[] = Int(get(vals, "qq_nemo_sparse_solve_threshold_tall_low", qq_sparse_solve_tall_fallback))
        QQ_NEMO_SPARSE_SOLVE_THRESHOLD_TALL_MID[] = Int(get(vals, "qq_nemo_sparse_solve_threshold_tall_mid", qq_sparse_solve_tall_fallback))
        QQ_NEMO_SPARSE_SOLVE_THRESHOLD_TALL_HIGH[] = Int(get(vals, "qq_nemo_sparse_solve_threshold_tall_high", qq_sparse_solve_tall_fallback))
        QQ_NEMO_SPARSE_SOLVE_THRESHOLD_WIDE_LOW[] = Int(get(vals, "qq_nemo_sparse_solve_threshold_wide_low", qq_sparse_solve_wide_fallback))
        QQ_NEMO_SPARSE_SOLVE_THRESHOLD_WIDE_MID[] = Int(get(vals, "qq_nemo_sparse_solve_threshold_wide_mid", qq_sparse_solve_wide_fallback))
        QQ_NEMO_SPARSE_SOLVE_THRESHOLD_WIDE_HIGH[] = Int(get(vals, "qq_nemo_sparse_solve_threshold_wide_high", qq_sparse_solve_wide_fallback))
        QQ_NEMO_SPARSE_SOLVE_POLICY_SQUARE_LOW[] = Int(get(vals, "qq_nemo_sparse_solve_policy_square_low", QQ_NEMO_SPARSE_SOLVE_POLICY_SQUARE_LOW[]))
        QQ_NEMO_SPARSE_SOLVE_POLICY_SQUARE_MID[] = Int(get(vals, "qq_nemo_sparse_solve_policy_square_mid", QQ_NEMO_SPARSE_SOLVE_POLICY_SQUARE_MID[]))
        QQ_NEMO_SPARSE_SOLVE_POLICY_SQUARE_HIGH[] = Int(get(vals, "qq_nemo_sparse_solve_policy_square_high", QQ_NEMO_SPARSE_SOLVE_POLICY_SQUARE_HIGH[]))
        QQ_NEMO_SPARSE_SOLVE_POLICY_TALL_LOW[] = Int(get(vals, "qq_nemo_sparse_solve_policy_tall_low", QQ_NEMO_SPARSE_SOLVE_POLICY_TALL_LOW[]))
        QQ_NEMO_SPARSE_SOLVE_POLICY_TALL_MID[] = Int(get(vals, "qq_nemo_sparse_solve_policy_tall_mid", QQ_NEMO_SPARSE_SOLVE_POLICY_TALL_MID[]))
        QQ_NEMO_SPARSE_SOLVE_POLICY_TALL_HIGH[] = Int(get(vals, "qq_nemo_sparse_solve_policy_tall_high", QQ_NEMO_SPARSE_SOLVE_POLICY_TALL_HIGH[]))
        QQ_NEMO_SPARSE_SOLVE_POLICY_WIDE_LOW[] = Int(get(vals, "qq_nemo_sparse_solve_policy_wide_low", QQ_NEMO_SPARSE_SOLVE_POLICY_WIDE_LOW[]))
        QQ_NEMO_SPARSE_SOLVE_POLICY_WIDE_MID[] = Int(get(vals, "qq_nemo_sparse_solve_policy_wide_mid", QQ_NEMO_SPARSE_SOLVE_POLICY_WIDE_MID[]))
        QQ_NEMO_SPARSE_SOLVE_POLICY_WIDE_HIGH[] = Int(get(vals, "qq_nemo_sparse_solve_policy_wide_high", QQ_NEMO_SPARSE_SOLVE_POLICY_WIDE_HIGH[]))

        qq_mod_ns_fallback = Int(get(vals, "modular_nullspace_threshold", MODULAR_NULLSPACE_THRESHOLD[]))
        qq_mod_solve_fallback = Int(get(vals, "modular_solve_threshold", MODULAR_SOLVE_THRESHOLD[]))
        QQ_MODULAR_NULLSPACE_THRESHOLD_SQUARE[] = Int(get(vals, "qq_modular_nullspace_threshold_square", qq_mod_ns_fallback))
        QQ_MODULAR_NULLSPACE_THRESHOLD_TALL[] = Int(get(vals, "qq_modular_nullspace_threshold_tall", qq_mod_ns_fallback))
        QQ_MODULAR_NULLSPACE_THRESHOLD_WIDE[] = Int(get(vals, "qq_modular_nullspace_threshold_wide", qq_mod_ns_fallback))
        QQ_MODULAR_SOLVE_THRESHOLD_SQUARE[] = Int(get(vals, "qq_modular_solve_threshold_square", qq_mod_solve_fallback))
        QQ_MODULAR_SOLVE_THRESHOLD_TALL[] = Int(get(vals, "qq_modular_solve_threshold_tall", qq_mod_solve_fallback))
        QQ_MODULAR_SOLVE_THRESHOLD_WIDE[] = Int(get(vals, "qq_modular_solve_threshold_wide", qq_mod_solve_fallback))

        FP_NEMO_RANK_THRESHOLD[] = Int(get(vals, "fp_nemo_rank_threshold", FP_NEMO_RANK_THRESHOLD[]))
        FP_NEMO_NULLSPACE_THRESHOLD[] = Int(get(vals, "fp_nemo_nullspace_threshold", FP_NEMO_NULLSPACE_THRESHOLD[]))
        FP_NEMO_SOLVE_THRESHOLD[] = Int(get(vals, "fp_nemo_solve_threshold", FP_NEMO_SOLVE_THRESHOLD[]))
        MODULAR_NULLSPACE_THRESHOLD[] = Int(get(vals, "modular_nullspace_threshold", MODULAR_NULLSPACE_THRESHOLD[]))
        MODULAR_SOLVE_THRESHOLD[] = Int(get(vals, "modular_solve_threshold", MODULAR_SOLVE_THRESHOLD[]))
        MODULAR_MIN_PRIMES[] = Int(get(vals, "modular_min_primes", MODULAR_MIN_PRIMES[]))
        MODULAR_MAX_PRIMES[] = Int(get(vals, "modular_max_primes", MODULAR_MAX_PRIMES[]))
        RANKQQ_DIM_SMALL_THRESHOLD[] = Int(get(vals, "rankqq_dim_small_threshold", RANKQQ_DIM_SMALL_THRESHOLD[]))
        RANKQQ_RESTRICTED_WORDS_NEMO_THRESHOLD[] = Int(get(vals, "rankqq_restricted_words_nemo_threshold",
                                                           RANKQQ_RESTRICTED_WORDS_NEMO_THRESHOLD[]))
        FLOAT_NULLSPACE_SVD_THRESHOLD[] = Int(get(vals, "float_nullspace_svd_threshold", FLOAT_NULLSPACE_SVD_THRESHOLD[]))
        FLOAT_SPARSE_SVDS_MIN_DIM[] = Int(get(vals, "float_sparse_svds_min_dim", FLOAT_SPARSE_SVDS_MIN_DIM[]))
        FLOAT_SPARSE_SVDS_MIN_NNZ[] = Int(get(vals, "float_sparse_svds_min_nnz", FLOAT_SPARSE_SVDS_MIN_NNZ[]))
        _set_zn_qq_dimat_submatrix_work_threshold!(
            Int(get(vals, "zn_qq_dimat_submatrix_work_threshold",
                    ZN_QQ_DIMAT_SUBMATRIX_WORK_THRESHOLD[])))
        MODULAR_MIN_PRIMES[] = max(1, MODULAR_MIN_PRIMES[])
        MODULAR_MAX_PRIMES[] = max(MODULAR_MIN_PRIMES[], MODULAR_MAX_PRIMES[])
        RANKQQ_DIM_SMALL_THRESHOLD[] = max(1, RANKQQ_DIM_SMALL_THRESHOLD[])
        RANKQQ_RESTRICTED_WORDS_NEMO_THRESHOLD[] = max(1, RANKQQ_RESTRICTED_WORDS_NEMO_THRESHOLD[])
        QQ_NEMO_RANK_THRESHOLD_SQUARE[] = max(1, QQ_NEMO_RANK_THRESHOLD_SQUARE[])
        QQ_NEMO_RANK_THRESHOLD_TALL[] = max(1, QQ_NEMO_RANK_THRESHOLD_TALL[])
        QQ_NEMO_RANK_THRESHOLD_WIDE[] = max(1, QQ_NEMO_RANK_THRESHOLD_WIDE[])
        QQ_NEMO_NULLSPACE_THRESHOLD_SQUARE[] = max(1, QQ_NEMO_NULLSPACE_THRESHOLD_SQUARE[])
        QQ_NEMO_NULLSPACE_THRESHOLD_TALL[] = max(1, QQ_NEMO_NULLSPACE_THRESHOLD_TALL[])
        QQ_NEMO_NULLSPACE_THRESHOLD_WIDE[] = max(1, QQ_NEMO_NULLSPACE_THRESHOLD_WIDE[])
        QQ_NEMO_SOLVE_THRESHOLD_SQUARE[] = max(1, QQ_NEMO_SOLVE_THRESHOLD_SQUARE[])
        QQ_NEMO_SOLVE_THRESHOLD_TALL[] = max(1, QQ_NEMO_SOLVE_THRESHOLD_TALL[])
        QQ_NEMO_SOLVE_THRESHOLD_WIDE[] = max(1, QQ_NEMO_SOLVE_THRESHOLD_WIDE[])
        QQ_NEMO_SPARSE_SOLVE_THRESHOLD_SQUARE_LOW[] = max(1, QQ_NEMO_SPARSE_SOLVE_THRESHOLD_SQUARE_LOW[])
        QQ_NEMO_SPARSE_SOLVE_THRESHOLD_SQUARE_MID[] = max(1, QQ_NEMO_SPARSE_SOLVE_THRESHOLD_SQUARE_MID[])
        QQ_NEMO_SPARSE_SOLVE_THRESHOLD_SQUARE_HIGH[] = max(1, QQ_NEMO_SPARSE_SOLVE_THRESHOLD_SQUARE_HIGH[])
        QQ_NEMO_SPARSE_SOLVE_THRESHOLD_TALL_LOW[] = max(1, QQ_NEMO_SPARSE_SOLVE_THRESHOLD_TALL_LOW[])
        QQ_NEMO_SPARSE_SOLVE_THRESHOLD_TALL_MID[] = max(1, QQ_NEMO_SPARSE_SOLVE_THRESHOLD_TALL_MID[])
        QQ_NEMO_SPARSE_SOLVE_THRESHOLD_TALL_HIGH[] = max(1, QQ_NEMO_SPARSE_SOLVE_THRESHOLD_TALL_HIGH[])
        QQ_NEMO_SPARSE_SOLVE_THRESHOLD_WIDE_LOW[] = max(1, QQ_NEMO_SPARSE_SOLVE_THRESHOLD_WIDE_LOW[])
        QQ_NEMO_SPARSE_SOLVE_THRESHOLD_WIDE_MID[] = max(1, QQ_NEMO_SPARSE_SOLVE_THRESHOLD_WIDE_MID[])
        QQ_NEMO_SPARSE_SOLVE_THRESHOLD_WIDE_HIGH[] = max(1, QQ_NEMO_SPARSE_SOLVE_THRESHOLD_WIDE_HIGH[])
        QQ_NEMO_SPARSE_SOLVE_POLICY_SQUARE_LOW[] = QQ_NEMO_SPARSE_SOLVE_POLICY_SQUARE_LOW[] >= 0 ? 1 : -1
        QQ_NEMO_SPARSE_SOLVE_POLICY_SQUARE_MID[] = QQ_NEMO_SPARSE_SOLVE_POLICY_SQUARE_MID[] >= 0 ? 1 : -1
        QQ_NEMO_SPARSE_SOLVE_POLICY_SQUARE_HIGH[] = QQ_NEMO_SPARSE_SOLVE_POLICY_SQUARE_HIGH[] >= 0 ? 1 : -1
        QQ_NEMO_SPARSE_SOLVE_POLICY_TALL_LOW[] = QQ_NEMO_SPARSE_SOLVE_POLICY_TALL_LOW[] >= 0 ? 1 : -1
        QQ_NEMO_SPARSE_SOLVE_POLICY_TALL_MID[] = QQ_NEMO_SPARSE_SOLVE_POLICY_TALL_MID[] >= 0 ? 1 : -1
        QQ_NEMO_SPARSE_SOLVE_POLICY_TALL_HIGH[] = QQ_NEMO_SPARSE_SOLVE_POLICY_TALL_HIGH[] >= 0 ? 1 : -1
        QQ_NEMO_SPARSE_SOLVE_POLICY_WIDE_LOW[] = QQ_NEMO_SPARSE_SOLVE_POLICY_WIDE_LOW[] >= 0 ? 1 : -1
        QQ_NEMO_SPARSE_SOLVE_POLICY_WIDE_MID[] = QQ_NEMO_SPARSE_SOLVE_POLICY_WIDE_MID[] >= 0 ? 1 : -1
        QQ_NEMO_SPARSE_SOLVE_POLICY_WIDE_HIGH[] = QQ_NEMO_SPARSE_SOLVE_POLICY_WIDE_HIGH[] >= 0 ? 1 : -1
        QQ_MODULAR_NULLSPACE_THRESHOLD_SQUARE[] = max(1, QQ_MODULAR_NULLSPACE_THRESHOLD_SQUARE[])
        QQ_MODULAR_NULLSPACE_THRESHOLD_TALL[] = max(1, QQ_MODULAR_NULLSPACE_THRESHOLD_TALL[])
        QQ_MODULAR_NULLSPACE_THRESHOLD_WIDE[] = max(1, QQ_MODULAR_NULLSPACE_THRESHOLD_WIDE[])
        QQ_MODULAR_SOLVE_THRESHOLD_SQUARE[] = max(1, QQ_MODULAR_SOLVE_THRESHOLD_SQUARE[])
        QQ_MODULAR_SOLVE_THRESHOLD_TALL[] = max(1, QQ_MODULAR_SOLVE_THRESHOLD_TALL[])
        QQ_MODULAR_SOLVE_THRESHOLD_WIDE[] = max(1, QQ_MODULAR_SOLVE_THRESHOLD_WIDE[])
    catch
        return false
    end
    return true
end

function _save_linalg_thresholds!(; path::AbstractString=_linalg_thresholds_path())
    doc = Dict(
        "meta" => Dict(
            "created_utc" => Dates.format(Dates.now(Dates.UTC), dateformat"yyyy-mm-ddTHH:MM:SSZ"),
            "schema_version" => LINALG_THRESHOLD_SCHEMA_VERSION,
        ),
        "fingerprint" => _current_linalg_fingerprint(),
        "thresholds" => _current_linalg_thresholds(),
    )
    mkpath(dirname(path))
    open(path, "w") do io
        TOML.print(io, doc)
    end
    return path
end

function _fingerprints_match(a, b)::Bool
    keys = ("schema_version", "cpu_name", "cpu_threads", "julia_version", "word_size", "os", "blas_threads")
    for k in keys
        haskey(a, k) || return false
        haskey(b, k) || return false
        if string(a[k]) != string(b[k])
            return false
        end
    end
    return true
end

function _load_linalg_thresholds!(; path::AbstractString=_linalg_thresholds_path(), warn_on_mismatch::Bool=true)::Bool
    if !isfile(path)
        return false
    end
    doc = try
        TOML.parsefile(path)
    catch err
        @warn "FieldLinAlg: failed to parse thresholds file; using defaults." path exception=(err, catch_backtrace())
        return false
    end
    haskey(doc, "fingerprint") || return false
    haskey(doc, "thresholds") || return false

    current_fp = _current_linalg_fingerprint()
    stored_fp = doc["fingerprint"]
    if !_fingerprints_match(stored_fp, current_fp)
        if warn_on_mismatch
            @warn "FieldLinAlg: threshold fingerprint mismatch; using defaults. Run `FieldLinAlg.autotune_linalg_thresholds!()` to regenerate." path
        end
        return false
    end

    ok = _apply_linalg_thresholds!(doc["thresholds"])
    if !ok
        @warn "FieldLinAlg: malformed thresholds in file; using defaults." path
        return false
    end
    return true
end
