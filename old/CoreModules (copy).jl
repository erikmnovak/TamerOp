module CoreModules
# -----------------------------------------------------------------------------
# Core prelude for this project
#  - QQ = Rational{BigInt} is the canonical exact scalar type
#  - Optional feature flags (e.g. for optional PL axis backend)
#  - Thin wrappers for exact linear algebra backends (optional Nemo)
#  - Exact rational <-> string helpers for serialization
# -----------------------------------------------------------------------------

using LinearAlgebra, SparseArrays, Random

# -----------------------------------------------------------------------------
# Small statistics helpers (no external dependencies)
# -----------------------------------------------------------------------------
#
# We avoid pulling in StatsFuns/Distributions to keep the dependency footprint
# minimal while still supporting confidence intervals and z-scores.

# Approximate inverse CDF for a standard normal distribution.
# Rational approximation due to Peter J. Acklam; excellent accuracy for CI use.
@inline function _normal_quantile(p::Float64)
    if p == 0.0
        return -Inf
    elseif p == 1.0
        return Inf
    end
    (0.0 < p < 1.0) || error("p must lie in (0,1)")

    # Coefficients for the central region.
    a1 = -3.969683028665376e+01
    a2 =  2.209460984245205e+02
    a3 = -2.759285104469687e+02
    a4 =  1.383577518672690e+02
    a5 = -3.066479806614716e+01
    a6 =  2.506628277459239e+00

    b1 = -5.447609879822406e+01
    b2 =  1.615858368580409e+02
    b3 = -1.556989798598866e+02
    b4 =  6.680131188771972e+01
    b5 = -1.328068155288572e+01

    # Coefficients for the tails.
    c1 = -7.784894002430293e-03
    c2 = -3.223964580411365e-01
    c3 = -2.400758277161838e+00
    c4 = -2.549732539343734e+00
    c5 =  4.374664141464968e+00
    c6 =  2.938163982698783e+00

    d1 =  7.784695709041462e-03
    d2 =  3.224671290700398e-01
    d3 =  2.445134137142996e+00
    d4 =  3.754408661907416e+00

    plow = 0.02425
    phigh = 1.0 - plow

    if p < plow
        q = sqrt(-2.0 * log(p))
        return (((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) /
               ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0)
    elseif p <= phigh
        q = p - 0.5
        r = q*q
        return (((((a1*r + a2)*r + a3)*r + a4)*r + a5)*r + a6) * q /
               (((((b1*r + b2)*r + b3)*r + b4)*r + b5)*r + 1.0)
    else
        q = sqrt(-2.0 * log(1.0 - p))
        return -(((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) /
                ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0)
    end
end

# Wilson score confidence interval for a binomial proportion.
# Returns (lo, hi) with 0 <= lo <= hi <= 1.
@inline function _wilson_interval(x::Integer, n::Integer; alpha::Real=0.05)
    n <= 0 && error("n must be positive")
    x < 0 && error("x must be nonnegative")
    x > n && error("x must be <= n")
    a = float(alpha)
    (0.0 < a < 1.0) || error("alpha must lie in (0,1)")

    z = _normal_quantile(1.0 - a/2.0)
    nf = float(n)
    phat = float(x) / nf
    z2 = z*z

    denom = 1.0 + z2/nf
    center = (phat + z2/(2.0*nf)) / denom
    half = (z * sqrt(phat*(1.0 - phat)/nf + z2/(4.0*nf*nf))) / denom

    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return (lo, hi)
end


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
"Enable optional axis-aligned PL backend if set to 1/true in ENV."
const ENABLE_PL_AXIS = get(ENV, "POSETMODULES_ENABLE_PL_AXIS", "false") in ("1","true","TRUE")
export ENABLE_PL_AXIS

# ----- (optional) thin wrappers so callers can hold 'a QQ matrix' abstractly -----
abstract type AbstractQQMatrix end

"Plain Julia dense QQ matrix wrapper."
struct QQDense <: AbstractQQMatrix
    data::Matrix{QQ}
end

"Extract a dense matrix of QQ from any AbstractQQMatrix."
to_dense(A::QQDense) = A.data

# Optional Nemo wrapper (no hard dependency)
const HAVE_NEMO = try
    @eval import Nemo
    true
catch
    false
end
if HAVE_NEMO
    "Nemo fmpq_mat wrapper; construct via QQNemo(A::Matrix{QQ})."
    struct QQNemo <: AbstractQQMatrix
        data::Nemo.fmpq_mat
    end
    QQNemo(A::Matrix{QQ}) = QQNemo(Nemo.fmpq_mat(A))
    to_dense(A::QQNemo) = Matrix{QQ}(A.data)
    export QQNemo
end
export AbstractQQMatrix, QQDense, to_dense, HAVE_NEMO

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

Optionally they may implement geometric hooks such as:

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

"""
    locate(pi, x) -> Int

Generic classifier hook for finite encodings.

Many parts of this codebase build a finite encoding poset `P` for some
parameter space `Q` (for example `Q = R^n` or `Q = Z^n`). The corresponding
classifier `pi : Q -> P` is represented in code by an "encoding map" object.

Concrete encoding map types should extend this function by implementing methods
`locate(pi, x)`.

Return value convention:
- return an integer in 1:P.n identifying the region in the target finite poset, or
- return 0 when `x` does not lie in the encoded domain (for example if a polyhedral
  representation is incomplete, or if an encoding was deliberately truncated).

This convention matches the existing PL encoders.
"""
function locate end

"""
    region_weights(pi; box=nothing, kwargs...) -> AbstractVector

Optional hook for weighting invariants computed on a finite encoding.

Many invariants in multiparameter persistence reduce, after choosing a finite
encoding map `pi : Q -> P`, to data indexed by regions/vertices in a finite
poset `P`. When comparing two modules via the restricted Hilbert function
(dimension surface), it is often useful to weight each region by its "size"
(e.g. volume in R^n, or lattice-point count in Z^n) inside a bounding box.

This function is an *extension point* (like `locate`). Concrete encoding map
types may provide methods.

Common conventions:
- `region_weights(pi; box=(a,b))` returns a vector `w` with `w[t] >= 0`.
- `length(w)` should equal the number of encoded regions (i.e. `P.n`).
- If no meaningful weighting is available, a method may return all ones.
"""
function region_weights end

"""
    region_bbox(pi, r; box=nothing, kwargs...)

Geometric hook: return a bounding box for region `r` of an encoding map `pi`.

Intended meaning:
- For encodings of R^n: return an axis-aligned bounding box for the region in R^n.
- Many regions are naturally unbounded; in that case, you should either:
  * require a user-supplied finite `box=(a,b)` and compute the bbox of (region intersect box), or
  * return `-Inf` / `Inf` in unbounded directions.

Return convention:
- Return `nothing` if the region has empty intersection with the supplied `box`
  (or is otherwise empty for the intended semantics).
- Otherwise return `(a, b)` where `a[i] <= b[i]` are the lower/upper corners.

This is an extension point: concrete encoding map types may provide methods.
"""
function region_bbox end

"""
    region_volume(pi, r; box=nothing, kwargs...) -> Real

Convenience wrapper: return the `r`-th entry of `region_weights(pi; box=box, kwargs...)`.

This is useful when you only need one region's weight/volume and want a
mathematician-friendly name.
"""
function region_volume(pi, r::Integer; box=nothing, kwargs...)
    w = region_weights(pi; box=box, kwargs...)
    (1 <= r <= length(w)) || error("region_volume: region index out of range")
    return w[Int(r)]
end

"""
    region_diameter(pi, r; metric=:L2, box=nothing, method=:bbox, kwargs...) -> Real

Estimate the diameter of region `r` under the chosen metric.

Default implementation (`method == :bbox`):
Compute the diameter of the bounding box returned by
`region_bbox(pi, r; box=box, kwargs...)`.

This default is conservative: it is an upper bound on the true diameter inside
the box (if the region is non-convex, disconnected, etc.).

Metrics supported by the default method:
- `:L2`   (Euclidean)
- `:Linf` (max norm)
- `:L1`   (taxicab)

Concrete encodings may provide specialized methods, e.g. a vertex-based diameter
for convex polyhedral regions.
"""
function region_diameter(
    pi,
    r::Integer;
    metric::Symbol=:L2,
    box=nothing,
    method::Symbol=:bbox,
    kwargs...
)
    method == :bbox || error("region_diameter: method must be :bbox (or implement a specialized method)")

    bb = region_bbox(pi, r; box=box, kwargs...)
    bb === nothing && return 0.0

    a, b = bb
    length(a) == length(b) || error("region_diameter: region_bbox returned mismatched endpoints")
    n = length(a)

    if metric == :Linf
        d = 0.0
        for i in 1:n
            li = float(b[i]) - float(a[i])
            isfinite(li) || return Inf
            d = max(d, abs(li))
        end
        return d
    elseif metric == :L2
        acc = 0.0
        for i in 1:n
            li = float(b[i]) - float(a[i])
            isfinite(li) || return Inf
            acc += li * li
        end
        return sqrt(acc)
    elseif metric == :L1
        s = 0.0
        for i in 1:n
            li = float(b[i]) - float(a[i])
            isfinite(li) || return Inf
            s += abs(li)
        end
        return s
    else
        error("region_diameter: metric must be :L1, :L2, or :Linf")
    end
end

"""
    region_widths(pi, r; box=nothing, kwargs...) -> Union{Nothing,Vector{Float64}}

Side lengths of the axis-aligned bounding box of region `r`.

This is computed from `region_bbox(pi, r; box=box, kwargs...)` as `b - a`,
where `(a,b)` is the returned bounding box.

Returns `nothing` if `region_bbox` returns `nothing` (empty intersection).

Note:
- If you do not supply a finite `box`, some coordinates may be unbounded and
  the corresponding widths may be `Inf`.
"""
function region_widths(pi, r::Integer; box=nothing, kwargs...)
    bb = region_bbox(pi, r; box=box, kwargs...)
    bb === nothing && return nothing
    a, b = bb
    length(a) == length(b) || error("region_widths: region_bbox returned mismatched endpoints")
    n = length(a)
    w = Vector{Float64}(undef, n)
    for i in 1:n
        w[i] = float(b[i]) - float(a[i])
    end
    return w
end

"""
    region_centroid(pi, r; box=nothing, method=:bbox, kwargs...) -> Union{Nothing,Vector{Float64}}

A convenient representative point ("centroid") for region `r`.

By default (`method == :bbox`), this returns the midpoint of the bounding box
returned by `region_bbox(pi, r; box=box, kwargs...)`.

Returns `nothing` if:
- the region does not intersect the box, or
- the bounding box is unbounded in some coordinate (supply a finite `box`).

Encodings may provide more refined notions (e.g. true centroid, Monte Carlo
centroid) by adding specialized methods.
"""
function region_centroid(pi, r::Integer; box=nothing, method::Symbol=:bbox, kwargs...)
    method == :bbox || error("region_centroid: unsupported method $method (default is :bbox)")
    bb = region_bbox(pi, r; box=box, kwargs...)
    bb === nothing && return nothing
    a, b = bb
    n = length(a)
    c = Vector{Float64}(undef, n)
    for i in 1:n
        ai = float(a[i])
        bi = float(b[i])
        (isfinite(ai) && isfinite(bi)) || return nothing
        c[i] = (ai + bi) / 2.0
    end
    return c
end

"""
    region_aspect_ratio(pi, r; box=nothing, kwargs...) -> Real

A simple anisotropy proxy for region `r` based on its axis-aligned bounding box.

Let `w = region_widths(pi, r; ...)`. This function returns:

    max(w) / min_positive(w)

where `min_positive(w)` is the smallest strictly positive width.

Conventions:
- Returns 0.0 if the region has empty intersection with the given box.
- Returns 1.0 if the region collapses to a point (all widths are 0).
- Returns Inf if the region is unbounded (some width is Inf) or degenerate
  (some widths are 0 while others are positive).
"""
function region_aspect_ratio(pi, r::Integer; box=nothing, kwargs...)
    w = region_widths(pi, r; box=box, kwargs...)
    w === nothing && return 0.0

    maxw = 0.0
    minw = Inf
    for wi in w
        aw = abs(float(wi))
        isfinite(aw) || return Inf
        maxw = max(maxw, aw)
        if aw > 0.0
            minw = min(minw, aw)
        end
    end

    if maxw == 0.0
        return 1.0
    elseif minw == Inf
        return Inf
    else
        return maxw / minw
    end
end

"""
    region_adjacency(pi; box, kwargs...) -> AbstractDict

Geometric hook: report how regions meet inside a finite bounding window.

Implementations should return a dictionary whose keys are unordered region pairs
`(r,s)` with `r < s`. The value is the estimated (n-1)-dimensional measure of
the interface between regions `r` and `s` inside `box=(a,b)`.

Notes:
- For n=2 this is (a notion of) boundary length.
- For n=3 this is boundary area.
- For n=1 this is a 0-dimensional measure (counting boundary points), so each
  interior boundary typically contributes 1.

This is currently implemented for axis-aligned box grids (`PLEncodingMapBoxes`).
"""
function region_adjacency end

"""
    region_facet_count(pi, r; kwargs...) -> Int

Geometric hook: return a facet/constraint count for region `r`.

For polyhedral backends, a reasonable default is the number of inequalities in
the stored H-representation. This is a complexity proxy (it may count redundant
inequalities and thus may exceed the true number of facets).
"""
function region_facet_count end

"""
    region_vertex_count(pi, r; box, kwargs...) -> Union{Nothing,Int}

Geometric hook: return the number of vertices of `region r` inside the window
`box=(a,b)`.

This may be expensive. Implementations are allowed to return `nothing` if vertex
enumeration is not attempted (e.g. too many constraints / combinations).
"""
function region_vertex_count end


"""
    region_boundary_measure(pi, r; box=nothing, strict=true)

Boundary measure of region `r` inside the window `box=(a,b)`.

This is the (n-1)-dimensional Hausdorff measure of the boundary of `R_r cap box`:
* n == 2: perimeter
* n == 3: surface area
* general n: hypersurface measure

Backends should implement this when feasible. The default method throws an error.
"""
function region_boundary_measure(pi, r; box=nothing, strict=true)
    error("region_boundary_measure not implemented for $(typeof(pi))")
end

"""
    region_perimeter(pi, r; box, kwargs...) -> Float64

Convenience wrapper for 2D regions. This calls
`region_boundary_measure(pi, r; box=box, kwargs...)`.
"""

"""
    region_boundary_measure_breakdown(pi, r; box=nothing, kwargs...) -> Vector{NamedTuple}

Diagnostic geometric hook: a per-facet decomposition of the boundary measure of
region `r` inside the window `box=(a,b)`.

Each entry is a NamedTuple and should contain at least:
- `measure::Float64`

Backends may optionally provide additional fields such as:
- `normal::Vector{Float64}`: an (outward) normal of the facet (not necessarily unit).
- `point::Vector{Float64}`: a representative point on the facet (e.g. barycenter).
- `neighbor::Union{Nothing,Int}`: the neighboring region across the facet, if detectable.
- `kind::Symbol`: e.g. `:internal` (between regions) or `:box` (window boundary).

The total boundary measure should satisfy, approximately:

    region_boundary_measure(pi, r; box=box) approx sum(e.measure for e in breakdown)

This is a diagnostic-level query and may not be implemented for all encodings.
"""
function region_boundary_measure_breakdown(pi, r; box=nothing, kwargs...)
    error("region_boundary_measure_breakdown not implemented for $(typeof(pi))")
end


# --- helper: infer box from cache kwarg when possible -----------------------
# Many geometry routines accept either:
#   - an explicit box=(a,b), or
#   - a cache object built for a specific box (for repeated queries).
#
# Some wrappers (region_perimeter/region_surface_area) are defined in CoreModules
# and therefore do not know the concrete cache type. To keep layering clean,
# we infer a box by looking for common cache field names.
function _box_from_cache_or_kwarg(box, kwargs)
    box !== nothing && return box

    # If the caller passed cache=..., try to read a stored box from it.
    if haskey(kwargs, :cache)
        cache = kwargs[:cache]
        cache === nothing && return nothing

        # Prefer exact boxes if available (common in Polyhedra-based caches).
        if hasproperty(cache, :box_q)
            return getproperty(cache, :box_q)
        end

        # Fall back to float boxes if exact is unavailable.
        if hasproperty(cache, :box_f)
            return getproperty(cache, :box_f)
        end

        # Generic fallback.
        if hasproperty(cache, :box)
            return getproperty(cache, :box)
        end
    end

    return nothing
end



function region_perimeter(pi, r; box=nothing, kwargs...)
    # Allow omission of box if cache supplies it.
    box = _box_from_cache_or_kwarg(box, kwargs)
    box === nothing && error("region_perimeter: please provide box=(a,b)")

    a, _ = box
    length(a) == 2 || error("region_perimeter: expected 2D box, got length(a)=$(length(a))")
    return float(region_boundary_measure(pi, r; box=box, kwargs...))
end

"""
    region_surface_area(pi, r; box, kwargs...) -> Float64

Convenience wrapper for 3D and higher. This calls
`region_boundary_measure(pi, r; box=box, kwargs...)`.
"""
function region_surface_area(pi, r; box=nothing, kwargs...)
    # Allow omission of box if cache supplies it.
    box = _box_from_cache_or_kwarg(box, kwargs)
    box === nothing && error("region_surface_area: please provide box=(a,b)")

    a, _ = box
    length(a) >= 3 || error("region_surface_area: expected 3D+ box, got length(a)=$(length(a))")
    return float(region_boundary_measure(pi, r; box=box, kwargs...))
end

"""
    region_principal_directions(pi, r; box, nsamples=20_000, rng=Random.default_rng(),
        strict=true, closure=true, max_proposals=10*nsamples,
        return_info=false, nbatches=0)

Estimate the mean, covariance matrix, and principal directions for the region `r`
inside a *finite* `box = (ell, u)` by rejection sampling.

Return value (always):
- `mean`  : estimated mean of a uniform point in the region (within the box)
- `cov`   : estimated covariance matrix
- `evals` : eigenvalues of `cov`, sorted in descending order
- `evecs` : corresponding eigenvectors (columns), matching `evals`
- `n_accepted`, `n_proposed` : acceptance diagnostics

If `return_info=true`, additional uncertainty estimates are returned:
- `mean_stderr` : coordinatewise standard error for the mean estimate
- `evals_stderr`: standard error for the eigenvalue estimates, based on batching
- `batch_evals` : per-batch eigenvalue vectors (each sorted)
- `batch_n_accepted` : accepted samples per batch
- `nbatches`    : number of batches actually used

Batching behavior:
- If `return_info=true` and `nbatches == 0`, a default of 10 batches is used.
- Batch statistics are computed on *accepted* samples.

Speed notes:
- Uses preallocated vectors and in-place Welford updates to reduce allocations.
"""
function region_principal_directions(pi, r::Integer;
    box=nothing,
    nsamples::Int=20_000,
    rng::AbstractRNG=Random.default_rng(),
    strict::Bool=true,
    closure::Bool=true,
    max_proposals::Int=10*nsamples,
    return_info::Bool=false,
    nbatches::Int=0
)
    box === nothing && error("A finite box is required for region_principal_directions.")
    (a, b) = box
    n = length(a)

    mu = zeros(Float64, n)
    C = zeros(Float64, n, n)

    x = Vector{Float64}(undef, n)
    delta = Vector{Float64}(undef, n)
    delta2 = Vector{Float64}(undef, n)

    nacc = 0
    nprop = 0

    # Optional batching for uncertainty estimates.
    want_batches = return_info ? (nbatches > 0 ? nbatches : 10) : 0
    batch_evals = Vector{Vector{Float64}}()
    batch_n = Int[]
    if want_batches > 0
        sizehint!(batch_evals, want_batches)
        sizehint!(batch_n, want_batches)
    end

    mu_b = want_batches > 0 ? zeros(Float64, n) : Float64[]
    C_b = want_batches > 0 ? zeros(Float64, n, n) : Matrix{Float64}(undef, 0, 0)
    delta_b = want_batches > 0 ? zeros(Float64, n) : Float64[]
    delta2_b = want_batches > 0 ? zeros(Float64, n) : Float64[]
    nacc_b = 0
    batch_target = want_batches > 0 ? max(2, Int(floor(nsamples / want_batches))) : 0

    while (nacc < nsamples) && (nprop < max_proposals)
        @inbounds for i in 1:n
            x[i] = rand(rng) * (b[i] - a[i]) + a[i]
        end
        nprop += 1

        if locate(pi, x; closure=closure, strict=strict) == r
            nacc += 1

            if nacc == 1
                copyto!(mu, x)
            else
                @inbounds for i in 1:n
                    delta[i] = x[i] - mu[i]
                end
                @inbounds for i in 1:n
                    mu[i] += delta[i] / nacc
                    delta2[i] = x[i] - mu[i]
                end
                @inbounds for i in 1:n, j in 1:n
                    C[i, j] += delta[i] * delta2[j]
                end
            end

            if want_batches > 0
                nacc_b += 1
                if nacc_b == 1
                    copyto!(mu_b, x)
                else
                    @inbounds for i in 1:n
                        delta_b[i] = x[i] - mu_b[i]
                    end
                    @inbounds for i in 1:n
                        mu_b[i] += delta_b[i] / nacc_b
                        delta2_b[i] = x[i] - mu_b[i]
                    end
                    @inbounds for i in 1:n, j in 1:n
                        C_b[i, j] += delta_b[i] * delta2_b[j]
                    end
                end

                if nacc_b >= batch_target
                    if nacc_b > 1
                        cov_b = C_b / (nacc_b - 1)
                        E_b = eigen(Symmetric(cov_b))
                        p_b = sortperm(E_b.values, rev=true)
                        push!(batch_evals, E_b.values[p_b])
                        push!(batch_n, nacc_b)
                    end
                    fill!(mu_b, 0.0)
                    fill!(C_b, 0.0)
                    nacc_b = 0
                end
            end
        end
    end

    # Final partial batch.
    if want_batches > 0 && nacc_b > 1
        cov_b = C_b / (nacc_b - 1)
        E_b = eigen(Symmetric(cov_b))
        p_b = sortperm(E_b.values, rev=true)
        push!(batch_evals, E_b.values[p_b])
        push!(batch_n, nacc_b)
    end

    if nacc <= 1
        cov = zeros(Float64, n, n)
        evals = zeros(Float64, n)
        evecs = Matrix{Float64}(I, n, n)
        mean_stderr = fill(NaN, n)
        evals_stderr = fill(NaN, n)
    else
        cov = C / (nacc - 1)
        E = eigen(Symmetric(cov))
        p = sortperm(E.values, rev=true)
        evals = E.values[p]
        evecs = E.vectors[:, p]

        mean_stderr = sqrt.(diag(cov) ./ nacc)

        if length(batch_evals) >= 2
            k = length(batch_evals)
            evals_stderr = Vector{Float64}(undef, n)
            for i in 1:n
                s = 0.0
                ss = 0.0
                for bvals in batch_evals
                    v = bvals[i]
                    s += v
                    ss += v*v
                end
                m = s / k
                var = (ss - k*m*m) / (k - 1)
                var = var < 0.0 ? 0.0 : var
                evals_stderr[i] = sqrt(var) / sqrt(k)
            end
        else
            evals_stderr = fill(NaN, n)
        end
    end

    if !return_info
        return (mean=mu, cov=cov, evals=evals, evecs=evecs,
            n_accepted=nacc, n_proposed=nprop)
    end

    return (mean=mu, cov=cov, evals=evals, evecs=evecs,
        mean_stderr=mean_stderr, evals_stderr=evals_stderr,
        batch_evals=batch_evals, batch_n_accepted=batch_n, nbatches=length(batch_evals),
        n_accepted=nacc, n_proposed=nprop)
end



# ------------------------------------------------------------------------------
# Additional geometric descriptors for regions
#
# These are intended as "mathematician-friendly" higher-level quantities built
# on top of the primitive geometry routines (volume, boundary measure, bbox,
# principal directions).
#
# Most of these routines only make sense for bounded regions, so they generally
# require a finite `box=(a,b)` to intersect with.
# ------------------------------------------------------------------------------

# Internal helper: volume of the unit n-ball in Euclidean space (Float64).
# We avoid SpecialFunctions by using the recursion:
#   omega_0 = 1, omega_1 = 2, omega_n = (2*pi/n) * omega_{n-2}.
@inline function _unit_ball_volume(n::Integer)
    n < 0 && error("unit ball volume: dimension must be >= 0, got $n")
    n == 0 && return 1.0
    n == 1 && return 2.0

    if iseven(n)
        omega = 1.0
        k = 2
        while k <= n
            omega *= (2.0 * pi) / k
            k += 2
        end
        return omega
    else
        omega = 2.0
        k = 3
        while k <= n
            omega *= (2.0 * pi) / k
            k += 2
        end
        return omega
    end
end

@inline function _isoperimetric_constant(n::Integer)
    # Sharp Euclidean isoperimetric inequality:
    #   S >= c_n * V^((n-1)/n),  where c_n = n * omega_n^(1/n).
    n < 1 && error("isoperimetric constant: dimension must be >= 1, got $n")
    omega = _unit_ball_volume(n)
    return float(n) * omega^(1.0 / float(n))
end

# Internal helper: circumradius of an axis-aligned bounding box around a center.
@inline function _bbox_circumradius(lo::AbstractVector, hi::AbstractVector,
    c::AbstractVector, metric::Symbol)
    n = length(lo)
    length(hi) == n || error("bbox_circumradius: lo/hi length mismatch")
    length(c) == n || error("bbox_circumradius: center length mismatch")

    if metric === :L2
        s2 = 0.0
        @inbounds for i in 1:n
            di = max(abs(lo[i] - c[i]), abs(hi[i] - c[i]))
            s2 += di * di
        end
        return sqrt(s2)
    elseif metric === :Linf
        dmax = 0.0
        @inbounds for i in 1:n
            di = max(abs(lo[i] - c[i]), abs(hi[i] - c[i]))
            dmax = max(dmax, di)
        end
        return dmax
    elseif metric === :L1
        s = 0.0
        @inbounds for i in 1:n
            di = max(abs(lo[i] - c[i]), abs(hi[i] - c[i]))
            s += di
        end
        return s
    else
        error("bbox_circumradius: unknown metric=$metric (use :L2, :L1, or :Linf)")
    end
end

# A small helper to call region_volume / region_boundary_measure with optional
# closure/cache/strict keywords, while gracefully degrading when the backend
# does not accept these keywords.
function _region_volume_maybe_cached(pi, r::Integer; box, closure::Bool=true, cache=nothing)
    if cache === nothing
        try
            return float(region_volume(pi, r; box=box, closure=closure))
        catch e
            if e isa MethodError
                return float(region_volume(pi, r; box=box))
            end
            rethrow()
        end
    end

    try
        return float(region_volume(pi, r; box=box, closure=closure, cache=cache))
    catch e
        if e isa MethodError
            # Maybe the backend doesn't accept `cache` and/or `closure`.
            try
                return float(region_volume(pi, r; box=box, closure=closure))
            catch e2
                if e2 isa MethodError
                    return float(region_volume(pi, r; box=box))
                end
                rethrow()
            end
        end
        rethrow()
    end
end

function _region_boundary_measure_maybe_cached(pi, r::Integer; box, strict::Bool=true,
    closure::Bool=true, cache=nothing)
    if cache !== nothing
        try
            return float(region_boundary_measure(pi, r; box=box, strict=strict, closure=closure, cache=cache))
        catch e
            if !(e isa MethodError)
                rethrow()
            end
        end
    end

    try
        return float(region_boundary_measure(pi, r; box=box, strict=strict, closure=closure))
    catch e
        if e isa MethodError
            try
                return float(region_boundary_measure(pi, r; box=box, strict=strict))
            catch e2
                if e2 isa MethodError
                    return float(region_boundary_measure(pi, r; box=box))
                end
                rethrow()
            end
        end
        rethrow()
    end
end

# Helper: call region_principal_directions with (optional) closure keyword.
#
# Some backends do not accept `closure` and/or additional diagnostics keywords.
# This helper is used by higher-level invariants that want a uniform API.
function _principal_directions_maybe_closure(pi, r::Integer; box, nsamples::Integer,
    rng, strict::Bool=true, closure::Bool=true, max_proposals::Integer=10*nsamples,
    return_info::Bool=false, nbatches::Int=0)
    try
        return region_principal_directions(pi, r; box=box, nsamples=nsamples, rng=rng,
            strict=strict, closure=closure, max_proposals=max_proposals,
            return_info=return_info, nbatches=nbatches)
    catch e
        if e isa MethodError
            # Retry without keywords that some backends do not implement.
            return region_principal_directions(pi, r; box=box, nsamples=nsamples, rng=rng,
                strict=strict, max_proposals=max_proposals,
                return_info=return_info, nbatches=nbatches)
        end
        rethrow()
    end
end


"""
    region_chebyshev_ball(pi, r; box, metric=:L2, method=:auto, kwargs...) -> NamedTuple

Return a `(center, radius)` pair describing a large inscribed ball of region `r`,
intersected with `box=(a,b)`.

This is backend-dependent:
- Convex polyhedral backends can compute a true Chebyshev (largest inscribed) ball.
- Non-convex backends may return a lower bound (still a valid inscribed ball).

The ball is interpreted in the norm specified by `metric`:
- `:L2`   Euclidean ball
- `:Linf` axis-aligned cube (L_infinity ball)
- `:L1`   cross-polytope (L1 ball)

Backends that do not implement this should throw an error.
"""
function region_chebyshev_ball(pi, r::Integer; box=nothing, metric::Symbol=:L2,
    method::Symbol=:auto, kwargs...)
    error("region_chebyshev_ball not implemented for $(typeof(pi)).")
end

"""
    region_chebyshev_center(pi, r; kwargs...) -> Vector{Float64}

Return the center of `region_chebyshev_ball(pi,r; ...)`.
"""
function region_chebyshev_center(pi, r::Integer; kwargs...)
    return region_chebyshev_ball(pi, r; kwargs...).center
end

"""
    region_inradius(pi, r; kwargs...) -> Float64

Return the radius of `region_chebyshev_ball(pi,r; ...)`.
"""
function region_inradius(pi, r::Integer; kwargs...)
    return region_chebyshev_ball(pi, r; kwargs...).radius
end

"""
    region_circumradius(pi, r; box, center=:bbox, metric=:L2, method=:bbox, kwargs...) -> Float64

Return an outer radius for region `r` intersected with `box=(a,b)`.

This is the radius of a ball (in the specified norm) centered at `center` that
contains the set.

Default behavior is robust across backends: it uses only the region bounding box.

Arguments:
- `center` can be:
  - `:bbox`       (default) center of the region bounding box
  - `:centroid`   region centroid (exact for some backends, otherwise approximate)
  - `:chebyshev`  Chebyshev center (requires `region_chebyshev_ball`)
  - an explicit vector
- `method=:bbox` uses only the axis-aligned bounding box.

Backends may define more accurate methods (e.g. using vertices or cell corners).
"""
function region_circumradius(pi, r::Integer; box=nothing, center=:bbox,
    metric::Symbol=:L2, method::Symbol=:bbox, kwargs...)
    box === nothing && error("region_circumradius: box=(a,b) is required")
    metric = Symbol(metric)
    method = Symbol(method)

    # Choose the center.
    c = nothing
    if center === :chebyshev
        c = region_chebyshev_center(pi, r; box=box, metric=metric, kwargs...)
    elseif center === :centroid
        c = region_centroid(pi, r; box=box, kwargs...)
    elseif center === :bbox
        lo, hi = region_bbox(pi, r; box=box, kwargs...)
        c = (lo .+ hi) ./ 2
    else
        c = center
    end

    if method === :bbox
        lo, hi = region_bbox(pi, r; box=box, kwargs...)
        return _bbox_circumradius(lo, hi, c, metric)
    else
        error("region_circumradius: method=$method not available for $(typeof(pi))")
    end
end

"""
    region_boundary_to_volume_ratio(pi, r; box, volume=nothing, boundary=nothing,
                                    strict=true, closure=true, cache=nothing) -> Float64

Return boundary_measure / volume for region `r` intersected with `box=(a,b)`.

This is not scale-invariant, but is often a useful "boundary-to-volume" statistic.
If volume is zero, returns `Inf` if boundary is positive, and `NaN` if both are zero.

If you already computed `volume` and/or `boundary`, pass them in to avoid recomputation.
"""
function region_boundary_to_volume_ratio(pi, r::Integer; box=nothing,
    volume=nothing, boundary=nothing,
    strict::Bool=true, closure::Bool=true, cache=nothing)
    box === nothing && error("region_boundary_to_volume_ratio: box=(a,b) is required")

    V = volume === nothing ? _region_volume_maybe_cached(pi, r; box=box, closure=closure, cache=cache) : float(volume)
    S = boundary === nothing ? _region_boundary_measure_maybe_cached(pi, r; box=box, strict=strict, closure=closure, cache=cache) : float(boundary)

    if V == 0.0
        return (S == 0.0) ? NaN : Inf
    end
    return S / V
end

"""
    region_isoperimetric_ratio(pi, r; box, volume=nothing, boundary=nothing,
                              kind=:quotient, strict=true, closure=true, cache=nothing) -> Float64

Compute an isoperimetric shape statistic for region `r` intersected with `box=(a,b)`.

Let `V` be the n-dimensional volume and `S` the (n-1)-dimensional boundary measure.
For `n >= 2`, the sharp Euclidean isoperimetric inequality states

    S >= c_n * V^((n-1)/n),   c_n = n * omega_n^(1/n),

where `omega_n` is the volume of the unit n-ball.

Supported `kind` values:
- `:quotient`           returns `c_n * V^((n-1)/n) / S` (1 for Euclidean balls; typically <= 1)
- `:ratio`              reciprocal of `:quotient`
- `:deficit`            `1 - :quotient`
- `:planar`             planar quotient `4*pi*V / S^2` (requires n==2)
- `:boundary_to_volume` returns `S / V` (not scale-invariant)

If `V <= 0` or `S <= 0`, returns `NaN`.
"""
function region_isoperimetric_ratio(pi, r::Integer; box=nothing,
    volume=nothing, boundary=nothing, kind::Symbol=:quotient,
    strict::Bool=true, closure::Bool=true, cache=nothing)
    box === nothing && error("region_isoperimetric_ratio: box=(a,b) is required")

    V = volume === nothing ? _region_volume_maybe_cached(pi, r; box=box, closure=closure, cache=cache) : float(volume)
    S = boundary === nothing ? _region_boundary_measure_maybe_cached(pi, r; box=box, strict=strict, closure=closure, cache=cache) : float(boundary)

    if kind === :boundary_to_volume
        return region_boundary_to_volume_ratio(pi, r; box=box, volume=V, boundary=S,
            strict=strict, closure=closure, cache=cache)
    end

    a, b = box
    n = length(a)
    n >= 2 || error("region_isoperimetric_ratio: requires ambient dimension n>=2, got n=$n")

    if V <= 0.0 || S <= 0.0
        return NaN
    end

    if kind === :planar
        n == 2 || error("region_isoperimetric_ratio(kind=:planar) requires n==2, got n=$n")
        return (4.0 * pi * V) / (S * S)
    end

    cn = _isoperimetric_constant(n)
    q = cn * V^((float(n) - 1.0) / float(n)) / S

    if kind === :quotient
        return q
    elseif kind === :ratio
        return 1.0 / q
    elseif kind === :deficit
        return 1.0 - q
    else
        error("region_isoperimetric_ratio: unknown kind=$kind")
    end
end

# Random directions on S^{n-1}. Columns are unit vectors.
function _random_unit_directions(n::Integer, ndirs::Integer; rng=Random.default_rng())
    n >= 1 || error("random_unit_directions: n must be >= 1")
    ndirs >= 1 || error("random_unit_directions: ndirs must be >= 1")

    U = Matrix{Float64}(undef, n, ndirs)
    @inbounds for j in 1:ndirs
        s2 = 0.0
        for i in 1:n
            u = randn(rng)
            U[i, j] = u
            s2 += u * u
        end
        if s2 == 0.0
            fill!(view(U, :, j), 0.0)
            U[1, j] = 1.0
        else
            invs = inv(sqrt(s2))
            for i in 1:n
                U[i, j] *= invs
            end
        end
    end
    return U
end

"""
    region_mean_width(pi, r; box, method=:auto, ndirs=256, nsamples=4000,
                      max_proposals=10*nsamples, rng=Random.default_rng(),
                      directions=nothing, strict=true, closure=true, cache=nothing) -> Float64

Estimate the mean width of region `r` intersected with `box=(a,b)`.

Mean width is the average of the width
`w(u) = sup_{x in K} <u,x> - inf_{x in K} <u,x>`
over unit directions `u` on the sphere.

Supported methods:
- `method=:cauchy` (only n==2, convex planar sets): mean_width = perimeter/pi
- `method=:mc`     Monte Carlo estimate using random directions and random points
- `method=:auto`   uses `:cauchy` when n==2, otherwise `:mc`

Backends may implement more accurate methods (e.g. using vertices or cell corners)
by defining a more specific method.
"""
function region_mean_width(pi, r::Integer; box=nothing, method::Symbol=:auto,
    ndirs::Integer=256, nsamples::Integer=4000, max_proposals::Integer=10*nsamples,
    rng=Random.default_rng(), directions=nothing,
    strict::Bool=true, closure::Bool=true, cache=nothing)
    box === nothing && error("region_mean_width: box=(a,b) is required")
    a, b = box
    n = length(a)

    method = Symbol(method)
    if method === :auto
        method = (n == 2) ? :cauchy : :mc
    end

    if method === :cauchy
        n == 2 || error("region_mean_width(method=:cauchy) requires n==2, got n=$n")
        return region_perimeter(pi, r; box=box, strict=strict) / pi
    elseif method !== :mc
        error("region_mean_width: unknown method=$method (use :auto, :cauchy, :mc)")
    end

    # Decide once whether `locate` supports `closure`.
    x0 = (a .+ b) ./ 2
    locate_fn = nothing
    try
        locate(pi, x0; strict=true, closure=true)
        locate_fn = (x)->locate(pi, x; strict=strict, closure=closure)
    catch e
        if e isa MethodError
            locate_fn = (x)->locate(pi, x; strict=strict)
        else
            rethrow()
        end
    end

    U = directions === nothing ? _random_unit_directions(n, ndirs; rng=rng) : directions
    size(U, 1) == n || error("region_mean_width: directions must have size (n,ndirs)")

    minproj = fill(Inf, size(U, 2))
    maxproj = fill(-Inf, size(U, 2))

    x = Vector{Float64}(undef, n)
    nacc = 0
    proposals = 0
    @inbounds while nacc < nsamples && proposals < max_proposals
        proposals += 1
        for i in 1:n
            x[i] = a[i] + rand(rng) * (b[i] - a[i])
        end
        q = locate_fn(x)
        if q == r
            nacc += 1
            for j in 1:size(U, 2)
                s = 0.0
                for i in 1:n
                    s += U[i, j] * x[i]
                end
                if s < minproj[j]
                    minproj[j] = s
                end
                if s > maxproj[j]
                    maxproj[j] = s
                end
            end
        elseif q == 0 && strict
            error("region_mean_width: encountered locate()==0; use strict=false or closure=true.")
        end
    end

    if nacc == 0
        return 0.0
    end

    wsum = 0.0
    @inbounds for j in 1:length(minproj)
        if isfinite(minproj[j])
            wsum += (maxproj[j] - minproj[j])
        end
    end
    return wsum / float(length(minproj))
end

"""
    region_minkowski_functionals(pi, r; box, volume=nothing, boundary=nothing,
                                 mean_width_method=:auto, mean_width_ndirs=256,
                                 mean_width_nsamples=4000, mean_width_max_proposals=10*mean_width_nsamples,
                                 mean_width_rng=Random.default_rng(), mean_width_directions=nothing,
                                 strict=true, closure=true, cache=nothing) -> NamedTuple

Compute a small bundle of Minkowski-type functionals for region `r` intersected with `box=(a,b)`.

Returned fields:
- `volume`           n-dimensional volume
- `boundary_measure` (n-1)-dimensional boundary measure
- `mean_width`       estimated mean width (see `region_mean_width`)

Pass `volume` and/or `boundary` if you already computed them.
"""
function region_minkowski_functionals(pi, r::Integer; box=nothing,
    volume=nothing, boundary=nothing,
    mean_width_method::Symbol=:auto,
    mean_width_ndirs::Integer=256,
    mean_width_nsamples::Integer=4000,
    mean_width_max_proposals::Integer=10*mean_width_nsamples,
    mean_width_rng=Random.default_rng(),
    mean_width_directions=nothing,
    strict::Bool=true, closure::Bool=true, cache=nothing)

    box === nothing && error("region_minkowski_functionals: box=(a,b) is required")

    V = volume === nothing ? _region_volume_maybe_cached(pi, r; box=box, closure=closure, cache=cache) : float(volume)
    S = boundary === nothing ? _region_boundary_measure_maybe_cached(pi, r; box=box, strict=strict, closure=closure, cache=cache) : float(boundary)

    mw = float(region_mean_width(pi, r; box=box, method=mean_width_method,
        ndirs=mean_width_ndirs, nsamples=mean_width_nsamples,
        max_proposals=mean_width_max_proposals, rng=mean_width_rng,
        directions=mean_width_directions, strict=strict, closure=closure, cache=cache))

    return (volume=V, boundary_measure=S, mean_width=mw)
end

"""
    region_covariance_anisotropy(pi, r; box, kind=:ratio, epsilon=0.0,
                                 nsamples=20000, max_proposals=10*nsamples,
                                 rng=Random.default_rng(), strict=true, closure=true) -> Float64

Compute an anisotropy score from the covariance eigenvalues returned by
`region_principal_directions(pi,r; ...)`.

Let `lambda_max >= ... >= lambda_min` be covariance eigenvalues.
Supported `kind`:
- `:ratio`      lambda_max / max(lambda_min, epsilon)
- `:log_ratio`  log(lambda_max / max(lambda_min, epsilon))
- `:normalized` (lambda_max - lambda_min) / (lambda_max + lambda_min + epsilon)

If lambda_max == 0 (degenerate region), returns 1 for `:ratio`, and 0 otherwise.
"""
function region_covariance_anisotropy(pi, r::Integer; box=nothing,
    kind::Symbol=:ratio, epsilon::Real=0.0,
    nsamples::Integer=20000, max_proposals::Integer=10*nsamples,
    rng=Random.default_rng(), strict::Bool=true, closure::Bool=true)

    box === nothing && error("region_covariance_anisotropy: box=(a,b) is required")

    pd = _principal_directions_maybe_closure(pi, r; box=box, nsamples=nsamples,
        rng=rng, strict=strict, closure=closure, max_proposals=max_proposals)

    evals = pd.evals
    isempty(evals) && return NaN
    lmax = float(evals[1])
    lmin = float(evals[end])
    eps = float(epsilon)

    if lmax == 0.0
        return (kind === :ratio) ? 1.0 : 0.0
    end

    if kind === :ratio
        return lmax / max(lmin, eps)
    elseif kind === :log_ratio
        return log(lmax / max(lmin, eps))
    elseif kind === :normalized
        return (lmax - lmin) / (lmax + lmin + eps)
    else
        error("region_covariance_anisotropy: unknown kind=$kind")
    end
end

"""
    region_covariance_eccentricity(pi, r; box, epsilon=0.0, kwargs...) -> Float64

An "eccentricity" score derived from covariance eigenvalues:

    ecc = sqrt(max(0, 1 - lambda_min / max(lambda_max, epsilon))).

Returns 0 for isotropic or point-like regions.
"""
function region_covariance_eccentricity(pi, r::Integer; box=nothing,
    epsilon::Real=0.0,
    nsamples::Integer=20000, max_proposals::Integer=10*nsamples,
    rng=Random.default_rng(), strict::Bool=true, closure::Bool=true)

    box === nothing && error("region_covariance_eccentricity: box=(a,b) is required")

    pd = _principal_directions_maybe_closure(pi, r; box=box, nsamples=nsamples,
        rng=rng, strict=strict, closure=closure, max_proposals=max_proposals)

    evals = pd.evals
    isempty(evals) && return NaN
    lmax = float(evals[1])
    lmin = float(evals[end])
    eps = float(epsilon)

    if lmax == 0.0
        return 0.0
    end

    t = 1.0 - (lmin / max(lmax, eps))
    return sqrt(max(t, 0.0))
end

"""
    region_anisotropy_scores(pi, r; box, epsilon=0.0, nsamples=20_000,
        max_proposals=10*nsamples, rng=Random.default_rng(),
        strict=true, closure=true, return_info=false, nbatches=0)

Compute several scale-invariant anisotropy scores derived from the covariance
matrix of a uniform point in region `r` (restricted to the finite `box`).

Output fields:
- `ratio`       : lambda_max / lambda_min (>= 1)
- `log_ratio`   : log(ratio)
- `normalized`  : (lambda_max - lambda_min) / (lambda_max + lambda_min) in [0,1)
- `eccentricity`: sqrt(1 - lambda_min/lambda_max) in [0,1)

If `epsilon > 0`, eigenvalues smaller than `epsilon` are clamped.

If `return_info=true`, requests batched diagnostics from
`region_principal_directions` and adds:
- `ratio_stderr`, `log_ratio_stderr`, `normalized_stderr`, `eccentricity_stderr`
computed from batch-to-batch variability (standard error of the estimate),
and includes `pca=pd` with sampling diagnostics.

This answers: "how stable is this feature under sampling?"
"""
function region_anisotropy_scores(pi, r::Integer;
    box=nothing,
    epsilon::Real=0.0,
    nsamples::Int=20_000,
    max_proposals::Int=10*nsamples,
    rng::AbstractRNG=Random.default_rng(),
    strict::Bool=true,
    closure::Bool=true,
    return_info::Bool=false,
    nbatches::Int=0
)
    box === nothing && error("A finite box is required for region_anisotropy_scores.")

    pd = _principal_directions_maybe_closure(pi, r;
        box=box,
        nsamples=nsamples,
        max_proposals=max_proposals,
        rng=rng,
        strict=strict,
        closure=closure,
        return_info=return_info,
        nbatches=nbatches)

    evals = pd.evals

    ratio = _covariance_anisotropy_from_evals(evals; kind=:ratio, epsilon=epsilon)
    log_ratio = _covariance_anisotropy_from_evals(evals; kind=:log_ratio, epsilon=epsilon)
    normalized = _covariance_anisotropy_from_evals(evals; kind=:normalized, epsilon=epsilon)
    ecc = _covariance_eccentricity_from_evals(evals; epsilon=epsilon)

    if !return_info
        return (ratio=ratio, log_ratio=log_ratio, normalized=normalized, eccentricity=ecc)
    end

    ratio_se = NaN
    log_ratio_se = NaN
    normalized_se = NaN
    ecc_se = NaN

    if haskey(pd, :batch_evals) && (pd.batch_evals !== nothing) && (length(pd.batch_evals) >= 2)
        k = length(pd.batch_evals)

        sr = 0.0; ssr = 0.0
        slr = 0.0; sslr = 0.0
        sn = 0.0; ssn = 0.0
        se = 0.0; sse = 0.0

        for bevals in pd.batch_evals
            rb = _covariance_anisotropy_from_evals(bevals; kind=:ratio, epsilon=epsilon)
            lrb = _covariance_anisotropy_from_evals(bevals; kind=:log_ratio, epsilon=epsilon)
            nb = _covariance_anisotropy_from_evals(bevals; kind=:normalized, epsilon=epsilon)
            eb = _covariance_eccentricity_from_evals(bevals; epsilon=epsilon)

            sr += rb; ssr += rb*rb
            slr += lrb; sslr += lrb*lrb
            sn += nb; ssn += nb*nb
            se += eb; sse += eb*eb
        end

        mr = sr / k
        var_r = (ssr - k*mr*mr) / (k - 1)
        var_r = var_r < 0.0 ? 0.0 : var_r
        ratio_se = sqrt(var_r) / sqrt(k)

        mlr = slr / k
        var_lr = (sslr - k*mlr*mlr) / (k - 1)
        var_lr = var_lr < 0.0 ? 0.0 : var_lr
        log_ratio_se = sqrt(var_lr) / sqrt(k)

        mn = sn / k
        var_n = (ssn - k*mn*mn) / (k - 1)
        var_n = var_n < 0.0 ? 0.0 : var_n
        normalized_se = sqrt(var_n) / sqrt(k)

        me = se / k
        var_e = (sse - k*me*me) / (k - 1)
        var_e = var_e < 0.0 ? 0.0 : var_e
        ecc_se = sqrt(var_e) / sqrt(k)
    end

    return (ratio=ratio, ratio_stderr=ratio_se,
        log_ratio=log_ratio, log_ratio_stderr=log_ratio_se,
        normalized=normalized, normalized_stderr=normalized_se,
        eccentricity=ecc, eccentricity_stderr=ecc_se,
        pca=pd)
end




export locate, region_weights, region_volume, region_bbox, region_widths, 
       region_centroid, region_aspect_ratio, region_diameter,
       region_adjacency, region_facet_count, region_vertex_count,
       region_boundary_measure, region_boundary_measure_breakdown, 
       region_perimeter, region_surface_area,
       region_principal_directions, 
       region_chebyshev_ball, region_chebyshev_center, region_inradius,
       region_circumradius,
       region_boundary_to_volume_ratio, region_isoperimetric_ratio,
       region_mean_width, region_minkowski_functionals,
       region_covariance_anisotropy, region_covariance_eccentricity, region_anisotropy_scores



end # module


