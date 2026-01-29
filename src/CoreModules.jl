module CoreModules
# -----------------------------------------------------------------------------
# Core prelude for this project
#  - QQ = Rational{BigInt} is the canonical exact scalar type
#  - Optional feature flags (e.g. for optional PL axis backend)
#  - Thin wrappers for exact linear algebra backends (optional Nemo)
#  - Exact rational <-> string helpers for serialization
# -----------------------------------------------------------------------------

using LinearAlgebra, SparseArrays

# ----- canonical field of scalars used everywhere --------------------------------
"Exact rationals used throughout (Rational{BigInt})."
const QQ = Rational{BigInt}
export QQ


"""
    _append_scaled_triplets!(I, J, V, A, row_off, col_off; scale=one(QQ))
    _append_scaled_triplets!(I, J, V, A, rows, cols; scale=one(QQ))

Internal helper for assembling sparse matrices via (I,J,V) triplets.

This eliminates the common anti-pattern:

    D = zeros(QQ, m, n)
    ... fill a few blocks/entries of D ...
    return sparse(D)

which allocates an m-by-n dense matrix even when the result is structurally sparse.

Two variants:

1. Offset-based (fast):
   Treat `A` as a contiguous block placed at global rows
       row_off .+ (1:size(A,1))
   and global cols
       col_off .+ (1:size(A,2))

2. Indexed (general):
   Place `A` at explicit global row indices `rows` and col indices `cols`.
   This supports non-contiguous placements (e.g. direct sums with custom order).

In both cases we append only nonzero entries of `A` (after scaling by `scale`).
Duplicates are allowed; `sparse(I,J,V,...)` will sum them.

All indexing is 1-based.
"""
function _append_scaled_triplets!(I::Vector{Int}, J::Vector{Int}, V::Vector{QQ},
                                 A::AbstractMatrix{QQ},
                                 row_off::Int, col_off::Int;
                                 scale::QQ = one(QQ))
    # Sparse fast-path
    if A isa SparseMatrixCSC{QQ,Int}
        Ii, Jj, Vv = findnz(A)
        @inbounds for k in eachindex(Ii)
            a = Vv[k]
            iszero(a) && continue
            push!(I, row_off + Ii[k])
            push!(J, col_off + Jj[k])
            push!(V, scale * a)
        end
        return nothing
    end

    # Transpose/adjoint of sparse: iterate parent nnz and swap indices
    if (A isa LinearAlgebra.Transpose{QQ,<:SparseMatrixCSC{QQ,Int}}) ||
       (A isa LinearAlgebra.Adjoint{QQ,<:SparseMatrixCSC{QQ,Int}})
        S = parent(A)
        Ii, Jj, Vv = findnz(S)
        @inbounds for k in eachindex(Ii)
            a = Vv[k]
            iszero(a) && continue
            push!(I, row_off + Jj[k])
            push!(J, col_off + Ii[k])
            push!(V, scale * a)   # QQ has no conjugation effect
        end
        return nothing
    end

    # Dense / generic: scan entries, skip zeros
    m, n = size(A)
    @inbounds for j in 1:n
        gj = col_off + j
        for i in 1:m
            a = A[i, j]
            iszero(a) && continue
            push!(I, row_off + i)
            push!(J, gj)
            push!(V, scale * a)
        end
    end
    return nothing
end

function _append_scaled_triplets!(I::Vector{Int}, J::Vector{Int}, V::Vector{QQ},
                                 A::AbstractMatrix{QQ},
                                 rows::AbstractVector{<:Integer},
                                 cols::AbstractVector{<:Integer};
                                 scale::QQ = one(QQ))
    @assert length(rows) == size(A, 1)
    @assert length(cols) == size(A, 2)

    if A isa SparseMatrixCSC{QQ,Int}
        Ii, Jj, Vv = findnz(A)
        @inbounds for k in eachindex(Ii)
            a = Vv[k]
            iszero(a) && continue
            push!(I, Int(rows[Ii[k]]))
            push!(J, Int(cols[Jj[k]]))
            push!(V, scale * a)
        end
        return nothing
    end

    if (A isa LinearAlgebra.Transpose{QQ,<:SparseMatrixCSC{QQ,Int}}) ||
       (A isa LinearAlgebra.Adjoint{QQ,<:SparseMatrixCSC{QQ,Int}})
        S = parent(A)
        Ii, Jj, Vv = findnz(S)
        @inbounds for k in eachindex(Ii)
            a = Vv[k]
            iszero(a) && continue
            push!(I, Int(rows[Jj[k]]))
            push!(J, Int(cols[Ii[k]]))
            push!(V, scale * a)
        end
        return nothing
    end

    m, n = size(A)
    @inbounds for j in 1:n
        gj = Int(cols[j])
        for i in 1:m
            a = A[i, j]
            iszero(a) && continue
            push!(I, Int(rows[i]))
            push!(J, gj)
            push!(V, scale * a)
        end
    end
    return nothing
end


# ----- feature flags --------------------------------------------------------------



# ----- (optional) thin wrappers so callers can hold 'a QQ matrix' abstractly -----
abstract type AbstractQQMatrix end

"Plain Julia dense QQ matrix wrapper."
struct QQDense <: AbstractQQMatrix
    data::Matrix{QQ}
end

"Extract a dense matrix of QQ from any AbstractQQMatrix."
to_dense(A::QQDense) = A.data

# Nemo integration (accelerated QQ linear algebra) is provided via a package
# extension, so CoreModules has no hard dependency on Nemo and does not
# dynamically load it here.

export AbstractQQMatrix, QQDense, to_dense

# ----- exact rational <-> string (for JSON round-trips) --------------------------
"Encode a rational as \"num/den\" so it survives JSON round-trips exactly."
rational_to_string(x::QQ) = string(numerator(x), "/", denominator(x))

"Inverse of `rational_to_string`."
function string_to_rational(s::AbstractString)::QQ
    t = split(strip(s), "/")
    length(t) == 2 || error("bad QQ string: $s")
    parse(BigInt, t[1]) // parse(BigInt, t[2])
end
export rational_to_string, string_to_rational

# ----- common abstract supertype for "encoding map" objects --------------------
"""
    AbstractPLikeEncodingMap

Abstract supertype for "encoding map" objects `pi` used throughout the finite-encoding
and PL backends.

This is a dispatch hook only: it imposes no required fields. Concrete encoding maps
should subtype this and implement at least:

  - `locate(pi, x)`

Optionally they may implement geometric hooks (see RegionGeometry) such as:

  - `region_weights(pi; box=...)`
  - `region_bbox(pi, r; box=...)`
  - `region_boundary_measure(pi, r; box=...)`, etc.

The name "PLike" is informal: it refers to encodings that behave like a
piecewise-constant classifier on a stratification of parameter space.
"""
abstract type AbstractPLikeEncodingMap end

"Convenience alias used in type signatures in higher-level code."
const PLikeEncodingMap = AbstractPLikeEncodingMap

export AbstractPLikeEncodingMap, PLikeEncodingMap


# ----- shared API hooks ----------------------------------------------------------

# ----- shared API hooks ----------------------------------------------------------

"""
    locate(pi, x::AbstractVector) -> Int
    locate(pi, x::NTuple{N,<:Real}) -> Int

Generic classifier hook for finite encodings.

Many parts of this codebase build a finite encoding poset `P` for some
parameter space `Q` (for example `Q = R^n` or `Q = Z^n`). The corresponding
classifier `pi : Q -> P` is represented in code by an "encoding map" object.

Required interface:
- Implement `locate(pi, x::AbstractVector)` for your encoding map type.
  The input vector `x` must have length `dimension(pi)`.

Optional (performance) interface:
- You may also implement `locate(pi, x::NTuple{N,<:Real})` for fast,
  allocation-free tuple dispatch. If you do not, a generic fallback method
  is provided which converts the tuple to a vector.

Return value convention:
- return an integer in 1:P.n identifying the region in the target finite poset, or
- return 0 when `x` does not lie in the encoded domain.

This convention matches the existing PL encoders.
"""
function locate end

# Default tuple fallback (for convenience). Backends that care about allocation-free
# tuple dispatch should implement a specialized tuple method.
function locate(pi::AbstractPLikeEncodingMap, x::NTuple{N,<:Real}) where {N}
    return locate(pi, collect(x))
end

"""
    dimension(pi) -> Int

Ambient dimension of the parameter space for an encoding map.

This should match the expected length of the coordinate vector accepted by
`locate(pi, x::AbstractVector)`.
"""
function dimension end

"""
    representatives(pi)

Return a finite collection of representative points for the regions of `pi`.

This is used for utilities that need a cheap summary of the encoding's extent in
parameter space (for example to infer a bounding box). A representative point
must be indexable and have length `dimension(pi)`.
"""
function representatives end

"""
    axes_from_encoding(pi)

Return a tuple of coordinate axes derived from the encoding itself.

Each axis is a sorted vector of coordinates (typically breakpoints / critical
values). Many invariant computations evaluate on an axis-aligned grid; this
function provides a reasonable default grid derived from the encoding.

Backends without a natural grid may omit this method and require callers to pass
explicit `axes`.
"""
function axes_from_encoding end

export locate, dimension, representatives, axes_from_encoding

# =============================================================================
# Option structs (public API)

"""
    EncodingOptions(; backend=:auto, max_regions=nothing, strict_eps=nothing)

Options controlling finite encodings. Used by:
* `ZnEncoding.encode_from_flange(s)` and related helpers (backend=:zn)
* `PLPolyhedra.encode_from_PL_fringe(s)` (backend=:pl)
* `PLBackend.encode_fringe_boxes` (backend=:pl_backend, optional)

Fields:
* `backend`: Symbol, one of `:auto`, `:zn`, `:pl`, `:pl_backend`.
* `max_regions`: Integer cap for region enumeration (backend-dependent defaults).
* `strict_eps`: QQ tolerance used by PL polyhedra backend (default backend constant).
"""
struct EncodingOptions
    backend::Symbol
    max_regions::Union{Nothing, Int}
    strict_eps::Any
end
EncodingOptions(; backend::Symbol=:auto, max_regions=nothing, strict_eps=nothing) =
    EncodingOptions(backend,
                    max_regions === nothing ? nothing : Int(max_regions),
                    strict_eps)

"""
    ResolutionOptions(; maxlen=3, minimal=false, check=true)

Options controlling (co)resolutions.

Fields:
* `maxlen`: length of resolution
* `minimal`: if true, request minimal resolution when available
* `check`: whether to verify minimality conditions
"""
struct ResolutionOptions
    maxlen::Int
    minimal::Bool
    check::Bool
end
ResolutionOptions(; maxlen::Int=3, minimal::Bool=false, check::Bool=true) =
    ResolutionOptions(maxlen, minimal, check)

"""
    InvariantOptions(; axes=nothing, axes_policy=:encoding, max_axis_len=256,
                     box=nothing, threads=nothing, strict=nothing)

Options controlling invariant computations (Euler surface, MMA decomposition,
signed barcode, distances, etc).
"""
struct InvariantOptions
    axes::Any
    axes_policy::Symbol
    max_axis_len::Int
    box::Any
    threads::Union{Nothing,Bool}
    strict::Union{Nothing,Bool}
end
InvariantOptions(; axes=nothing,
                 axes_policy::Symbol=:encoding,
                 max_axis_len::Int=256,
                 box=nothing,
                 threads=nothing,
                 strict=nothing) =
    InvariantOptions(axes, axes_policy, max_axis_len, box, threads, strict)

"""
    DerivedFunctorOptions(; maxdeg=3, model=:auto, canon=:auto)

Options controlling derived functor computations (Ext, Tor, etc).

Fields
------
- maxdeg: compute degrees 0..maxdeg (inclusive).
- model: selects the computational model. Its meaning depends on the derived functor:

  * Ext(M, N, df):
      :projective  - compute using a projective resolution of M.
      :injective   - compute using an injective resolution of N.
      :unified     - compute a model-independent ExtSpace containing both models and explicit
                     comparison isomorphisms.
      :auto        - alias for :projective.

  * Tor(Rop, L, df):
      :first   - resolve Rop (a P^op-module) and tensor with L.
      :second  - resolve L and tensor with Rop.
      :auto    - alias for :first.

- canon: only used when model == :unified for Ext. Chooses the canonical coordinate basis in the
  unified ExtSpace:
    :projective or :injective (or :auto as alias for :projective).
"""
struct DerivedFunctorOptions
    maxdeg::Int
    model::Symbol
    canon::Symbol
end
DerivedFunctorOptions(; maxdeg::Int=3, model::Symbol=:auto, canon::Symbol=:auto) =
    DerivedFunctorOptions(maxdeg, model, canon)


# -----------------------------------------------------------------------------
# Workflow / pipeline result objects

"""
    EncodingResult(P, M, pi; H=nothing, presentation=nothing,
                  opts=EncodingOptions(), backend=opts.backend, meta=NamedTuple())

Small workflow object representing the output of a user-facing `encode(...)`.

Fields
------
- P : finite encoding poset
- M : PModule on P
- pi: classifier map from the original domain to P
- H : optional FringeModule pushed down to P (kept to avoid recomputation)
- presentation : optional original presentation object (Flange / PLFringe / ...)
- opts : EncodingOptions used for the encoding
- backend : backend actually used (Symbol)
- meta : arbitrary metadata (NamedTuple recommended)
"""
struct EncodingResult{PType,MType,PiType,HType,PresType,MetaType}
    P::PType
    M::MType
    pi::PiType
    H::HType
    presentation::PresType
    opts::EncodingOptions
    backend::Symbol
    meta::MetaType
end

EncodingResult(P, M, pi;
               H=nothing,
               presentation=nothing,
               opts::EncodingOptions=EncodingOptions(),
               backend::Symbol=opts.backend,
               meta=NamedTuple()) =
    EncodingResult(P, M, pi, H, presentation, opts, backend, meta)

"""
    ResolutionResult(res; enc=nothing, betti=nothing, minimality=nothing,
                     opts=ResolutionOptions(), meta=NamedTuple())

Workflow object storing a resolution computation (projective or injective) plus provenance.
"""
struct ResolutionResult{ResType,EncType,BettiType,MinType,MetaType}
    res::ResType
    enc::EncType
    betti::BettiType
    minimality::MinType
    opts::ResolutionOptions
    meta::MetaType
end

ResolutionResult(res;
                 enc=nothing,
                 betti=nothing,
                 minimality=nothing,
                 opts::ResolutionOptions=ResolutionOptions(),
                 meta=NamedTuple()) =
    ResolutionResult(res, enc, betti, minimality, opts, meta)

"""
    InvariantResult(enc, which, value; opts=InvariantOptions(), meta=NamedTuple())

Workflow object storing a computed invariant plus provenance.
"""
struct InvariantResult{EncType,WhichType,ValType,MetaType}
    enc::EncType
    which::WhichType
    value::ValType
    opts::InvariantOptions
    meta::MetaType
end

InvariantResult(enc, which, value;
                opts::InvariantOptions=InvariantOptions(),
                meta=NamedTuple()) =
    InvariantResult(enc, which, value, opts, meta)


export EncodingOptions, ResolutionOptions, InvariantOptions, DerivedFunctorOptions
export EncodingResult, ResolutionResult, InvariantResult

end # module

# =============================================================================
# Stats
#
# Merged from the former src/Stats.jl to reduce file count.
#
# Important: this remains a sibling module `PosetModules.Stats` (not nested under
# CoreModules) so that internal code can continue to write:
#     using ..Stats: _wilson_interval
# without any refactor churn.
# =============================================================================
module Stats

# -----------------------------------------------------------------------------
# Small statistics helpers (no external dependencies)
# -----------------------------------------------------------------------------
#
# We avoid pulling in StatsFuns/Distributions to keep the dependency footprint
# minimal while still supporting confidence intervals and z-scores.

# Approximate inverse CDF for a standard normal distribution.
# Based on Peter John Acklam's rational approximation (public domain).
# Ref: http://home.online.no/~pjacklam/notes/invnorm/
function _normal_quantile(p::Real)
    if p <= 0
        return -Inf
    elseif p >= 1
        return Inf
    end

    # Coefficients in rational approximations.
    a = (-3.969683028665376e+01,
          2.209460984245205e+02,
         -2.759285104469687e+02,
          1.383577518672690e+02,
         -3.066479806614716e+01,
          2.506628277459239e+00)

    b = (-5.447609879822406e+01,
          1.615858368580409e+02,
         -1.556989798598866e+02,
          6.680131188771972e+01,
         -1.328068155288572e+01)

    c = (-7.784894002430293e-03,
         -3.223964580411365e-01,
         -2.400758277161838e+00,
         -2.549732539343734e+00,
          4.374664141464968e+00,
          2.938163982698783e+00)

    d = ( 7.784695709041462e-03,
          3.224671290700398e-01,
          2.445134137142996e+00,
          3.754408661907416e+00)

    # Define break-points.
    plow  = 0.02425
    phigh = 1 - plow

    if p < plow
        # Rational approximation for lower region.
        q = sqrt(-2*log(p))
        return (((((c[1]*q + c[2])*q + c[3])*q + c[4])*q + c[5])*q + c[6]) /
               ((((d[1]*q + d[2])*q + d[3])*q + d[4])*q + 1)
    elseif p > phigh
        # Rational approximation for upper region.
        q = sqrt(-2*log(1 - p))
        return -(((((c[1]*q + c[2])*q + c[3])*q + c[4])*q + c[5])*q + c[6]) /
                ((((d[1]*q + d[2])*q + d[3])*q + d[4])*q + 1)
    else
        # Rational approximation for central region.
        q = p - 0.5
        r = q*q
        return (((((a[1]*r + a[2])*r + a[3])*r + a[4])*r + a[5])*r + a[6])*q /
               (((((b[1]*r + b[2])*r + b[3])*r + b[4])*r + b[5])*r + 1)
    end
end

# Wilson score interval for a binomial proportion.
@inline function _wilson_interval(x::Integer, n::Integer; alpha::Real=0.05)
    if n <= 0
        return (0.0, 1.0)
    end
    if x < 0 || x > n
        throw(ArgumentError("x must satisfy 0 <= x <= n"))
    end

    z = _normal_quantile(1 - float(alpha)/2)
    phat = x / n
    denom = 1 + z^2 / n
    center = (phat + z^2/(2n)) / denom
    half = (z/denom) * sqrt((phat*(1 - phat) + z^2/(4n)) / n)

    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return (lo, hi)
end

end # module Stats
