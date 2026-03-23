module SliceInvariants
# -----------------------------------------------------------------------------
# SliceInvariants.jl
#
# SliceInvariants owner module extracted from Invariants.jl.
# -----------------------------------------------------------------------------

"""
    SliceInvariants

Owner module for slice restrictions, 1-parameter barcode-based distances,
compiled slice plans, slice vectorizations, and related sliced invariant APIs.
"""

using LinearAlgebra
using JSON3
using ..CoreModules: EncodingCache, AbstractCoeffField, RegionPosetCachePayload,
                     AbstractSlicePlanCache
using ..Options: InvariantOptions
using ..EncodingCore: PLikeEncodingMap, CompiledEncoding, GridEncodingMap, locate, locate_many!, axes_from_encoding, dimension, representatives
using Statistics: mean
using ..Stats: _wilson_interval
using ..Encoding: EncodingMap
using ..PLPolyhedra
using ..RegionGeometry: region_weights, region_volume, region_bbox, region_widths,
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
using ..FieldLinAlg
using ..InvariantCore: SliceSpec, RankQueryCache,
                       chain, values, weight,
                       check_slice_spec,
                       _unwrap_compiled,
                       _default_strict, _default_threads, _drop_keys,
                       orthant_directions, _normalize_dir,
                       _selection_kwargs_from_opts, _axes_kwargs_from_opts,
                       _eye, rank_map,
                       FloatBarcode, IndexBarcode,
                       PackedBarcode, PackedBarcodeGrid, PackedIndexBarcode, PackedFloatBarcode,
                       _empty_index_barcode, _empty_float_barcode,
                       _empty_packed_index_barcode, _empty_packed_float_barcode,
                       _packed_grid_undef, _packed_grid_from_matrix,
                       _packed_total_multiplicity,
                       _to_float_barcode,
                       _packed_barcode_from_rank,
                       _barcode_from_packed, _pack_float_barcode,
                       _float_dict_matrix_from_packed_grid,
                       _points_from_packed!,
                       RANK_INVARIANT_MEMO_THRESHOLD, RECTANGLE_LOC_LINEAR_CACHE_THRESHOLD,
                       _use_array_memo, _new_array_memo, _grid_cache_index,
                       _memo_get, _memo_set!, _map_leq_cached,
                       _rank_cache_get!, _resolve_rank_query_cache,
                       _rank_query_point_tuple, _rank_query_locate!
import ..FiniteFringe: AbstractPoset, FinitePoset, FringeModule, Upset, Downset, fiber_dimension,
                       leq, leq_matrix, upset_indices, downset_indices, leq_col, nvertices, build_cache!,
                       _preds
import ..ZnEncoding
import ..IndicatorTypes: UpsetPresentation, DownsetCopresentation
import ..ModuleComplexes: ModuleCochainComplex
import ..ChangeOfPosets: pushforward_left, pushforward_right
import ..PLBackend
import ..ChainComplexes: describe
import ..DerivedFunctors: source_module
import Base.Threads
import ..Serialization: save_mpp_decomposition_json, load_mpp_decomposition_json,
                        save_mpp_image_json, load_mpp_image_json
import ..Modules: PModule, map_leq, CoverCache, _get_cover_cache
import ..IndicatorResolutions: pmodule_from_fringe
import ..ZnEncoding: ZnEncodingMap

# =============================================================================
# (4) Slice restrictions and 1-parameter barcodes
#     + (5) Approximate matching distance via bottleneck distances
# =============================================================================

# ----- Slice restrictions and 1-parameter barcodes ----------------------------

const _SLICE_CHAIN_USE_BATCHED_LOCATE = Ref(true)
const _SLICE_CHAIN_BATCHED_LOCATE_MIN_SAMPLES = Ref(128)
const _SLICE_USE_LANDSCAPE_FEATURE_CACHE = Ref(true)
const _SLICE_USE_PACKED_DISTANCE_FASTPATH = Ref(true)

@inline _slice_chain_supports_batched_locate(::GridEncodingMap) = true
@inline _slice_chain_supports_batched_locate(::PLPolyhedra.PLEncodingMap) = true
@inline _slice_chain_supports_batched_locate(::PLBackend.PLEncodingMapBoxes) = false
@inline _slice_chain_supports_batched_locate(pi::CompiledEncoding) = _slice_chain_supports_batched_locate(pi.pi)
@inline _slice_chain_supports_batched_locate(::Any) = false

@inline function _slice_chain_use_batched_locate(pi, nsamples::Int)
    _SLICE_CHAIN_USE_BATCHED_LOCATE[] || return false
    nsamples >= _SLICE_CHAIN_BATCHED_LOCATE_MIN_SAMPLES[] || return false
    return _slice_chain_supports_batched_locate(pi)
end

@inline function _tgrid_cache_hash(tgrid::AbstractVector{<:Real})
    h = hash(length(tgrid))
    @inbounds for t in tgrid
        h = hash(float(t), h)
    end
    return UInt(h)
end

# Internal: normalize an axis-aligned box argument.
# We accept boxes as (lo, hi), where lo and hi are vectors of the same length.
# Returns Float64 vectors for robust numeric clipping.
function _normalize_axis_aligned_box(box)
    (box isa Tuple && length(box) == 2) || error("expected box=(lo, hi)")
    lo, hi = box
    lo_v = Float64[float(x) for x in lo]
    hi_v = Float64[float(x) for x in hi]
    length(lo_v) == length(hi_v) || error("box endpoints must have the same length")
    @inbounds for i in 1:length(lo_v)
        lo_v[i] <= hi_v[i] || error("box must satisfy lo[i] <= hi[i] for all i")
    end
    return lo_v, hi_v
end

@inline _resolve_box(pi, box) = (box === :auto ? (pi isa PLikeEncodingMap ? window_box(pi) : nothing) : box)

@inline function _values_are_float_vector(v)
    v isa AbstractVector || return false
    @inbounds for x in v
        x isa AbstractFloat || return false
    end
    return true
end

@inline function _values_are_int_vector(v)
    v isa AbstractVector || return false
    @inbounds for x in v
        x isa Integer || return false
    end
    return true
end

mutable struct _SliceKernelScratch
    points_a::Vector{Tuple{Float64,Float64}}
    points_b::Vector{Tuple{Float64,Float64}}
end

_SliceKernelScratch() = _SliceKernelScratch(Tuple{Float64,Float64}[], Tuple{Float64,Float64}[])

@inline _scratch_arenas(threads::Bool) =
    [_SliceKernelScratch() for _ in 1:(threads ? Threads.maxthreadid() : 1)]

# Internal: intersect two axis-aligned boxes (either may be `nothing`).
# If both are provided, we use their intersection.
function _intersect_axis_aligned_boxes(box, box2)
    if box === nothing
        return _normalize_axis_aligned_box(box2)
    elseif box2 === nothing
        return _normalize_axis_aligned_box(box)
    end
    lo1, hi1 = _normalize_axis_aligned_box(box)
    lo2, hi2 = _normalize_axis_aligned_box(box2)
    length(lo1) == length(lo2) || error("box and box2 must have the same dimension")

    lo = similar(lo1)
    hi = similar(hi1)
    @inbounds for i in 1:length(lo)
        lo[i] = max(lo1[i], lo2[i])
        hi[i] = min(hi1[i], hi2[i])
    end
    return lo, hi
end

# Internal: parameter interval [tlo, thi] for which x(t)=x0+t*dir lies in an axis-aligned box.
# Returns (Inf, -Inf) if the line misses the box.
function _line_param_range_in_box_nd(
    x0::AbstractVector,
    dir::AbstractVector,
    box::Tuple{Vector{Float64},Vector{Float64}};
    atol::Float64=1e-12
)
    lo, hi = box
    d = length(lo)
    length(hi) == d || error("_line_param_range_in_box_nd: box endpoint length mismatch")
    length(x0) == d || error("_line_param_range_in_box_nd: x0 length mismatch")
    length(dir) == d || error("_line_param_range_in_box_nd: dir length mismatch")

    tlo = -Inf
    thi =  Inf

    @inbounds for i in 1:d
        a = lo[i]
        b = hi[i]
        a <= b || return (Inf, -Inf)

        xi = float(x0[i])
        di = float(dir[i])

        if di == 0.0
            # Coordinate is constant along the line; must already lie in [a,b].
            (xi < a - atol || xi > b + atol) && return (Inf, -Inf)
            continue
        end

        t1 = (a - xi) / di
        t2 = (b - xi) / di
        lo_i = min(t1, t2)
        hi_i = max(t1, t2)

        tlo = max(tlo, lo_i)
        thi = min(thi, hi_i)

        (tlo <= thi + atol) || return (Inf, -Inf)
    end

    return (tlo, thi)
end

"""
    slice_chain(pi, x0, dir, opts::InvariantOptions;
        ts=nothing,
        tmin=nothing,
        tmax=nothing,
        nsteps=1001,
        box2=nothing,
        drop_unknown=true,
        dedup=true,
        check_chain=false)
    ) -> (chain::Vector{Int}, tvals::Vector)

Sample the encoding map `pi` along the affine line x(t) = x0 + t*dir and return
the sequence of region indices visited by the samples, together with the
parameter values at which those regions were recorded.

The line is:
    x(t) = x0 + t * dir

Parameter grids:
- If `ts` is provided, it is used verbatim (after optional filtering to a window).
- Otherwise the routine samples `nsteps` equally spaced points in an interval
  determined by `tmin` and `tmax`.

Windowing / strictness:
- Clipping is controlled by `opts.box` (primary) and optional `box2` (secondary).
  If either is provided, we clip the sampling interval to the intersection of the
  line with the intersected axis-aligned boxes.
  * `opts.box === :auto` is supported and resolves via `window_box(pi)`.
- Strictness is controlled by `opts.strict` (defaults to `true`).

Sampling policy:
- If `ts` is provided, we sample exactly those parameter values.
- Otherwise, we sample `range(tmin, tmax, length=nsteps)`; if `tmin`/`tmax` are
  omitted and clipping is active, they are inferred from the box intersection.

Clipping rule:
- Whenever a window is present, any user-specified interval `[tmin, tmax]` is
  clipped to the line-window intersection interval.
- If the line does not intersect the window, the returned chain is empty.

Unknown regions:
- If `strict=true`, encountering `locate(pi, x) == 0` throws an error.
- If `strict=false`, unknown samples are either dropped (`drop_unknown=true`)
  or kept as region index 0.

If `dedup=true`, consecutive repeats of the same region index are removed.

If `check_chain=true`, the routine verifies that the resulting chain is monotone
in the region poset of `pi`.

Return value:
- `(chain, tvals)` where `chain::Vector{Int}` are region ids, and `tvals::Vector{Float64}`
  are the corresponding parameters. 
  Unknown points are either dropped or marked `0` depending on `drop_unknown`.
"""
function slice_chain(pi, x0::AbstractVector, dir::AbstractVector, opts::InvariantOptions;
    ts = nothing,
    tmin = nothing,
    tmax = nothing,
    nsteps::Int = 1001,
    box2 = nothing,
    drop_unknown::Bool = true,
    dedup::Bool = true,
    check_chain::Bool = false)
    return _slice_chain_collect(
        pi,
        Float64[float(x) for x in x0],
        Float64[float(x) for x in dir],
        opts;
        ts=ts,
        tmin=tmin,
        tmax=tmax,
        nsteps=nsteps,
        box2=box2,
        drop_unknown=drop_unknown,
        dedup=dedup,
        check_chain=check_chain,
    )
end

function _slice_chain_collect(
    pi,
    x0f::Vector{Float64},
    dirf::Vector{Float64},
    opts::InvariantOptions;
    ts = nothing,
    tmin = nothing,
    tmax = nothing,
    nsteps::Int = 1001,
    box2 = nothing,
    drop_unknown::Bool = true,
    dedup::Bool = true,
    check_chain::Bool = false,
    locate_ws = nothing,
)
    chain = Int[]
    tvals = Float64[]
    sizehint!(chain, min(nsteps, 256))
    sizehint!(tvals, min(nsteps, 256))
    _slice_chain_visit(
        pi,
        x0f,
        dirf,
        opts,
        (rid, t) -> begin
            push!(chain, rid)
            push!(tvals, t)
            return nothing
        end;
        ts=ts,
        tmin=tmin,
        tmax=tmax,
        nsteps=nsteps,
        box2=box2,
        drop_unknown=drop_unknown,
        dedup=dedup,
        locate_ws=locate_ws,
    )
    isempty(tvals) && return chain, tvals
    if check_chain
        _check_chain_monotone(pi, x0f, dirf, chain, tvals; strict=(opts.strict === nothing ? true : opts.strict))
    end
    return chain, tvals
end

@inline function _slice_chain_collect(
    pi,
    x0::AbstractVector,
    dir::NTuple{N,<:Real},
    opts::InvariantOptions;
    kwargs...,
) where {N}
    return _slice_chain_collect(
        pi,
        Float64[float(x) for x in x0],
        Float64[dir[i] for i in 1:N],
        opts;
        kwargs...,
    )
end

@inline function _slice_chain_collect(
    pi,
    x0::NTuple{N,<:Real},
    dir::AbstractVector,
    opts::InvariantOptions;
    kwargs...,
) where {N}
    return _slice_chain_collect(
        pi,
        Float64[x0[i] for i in 1:N],
        Float64[float(v) for v in dir],
        opts;
        kwargs...,
    )
end

@inline function _slice_chain_collect(
    pi,
    x0::NTuple{N,<:Real},
    dir::NTuple{N,<:Real},
    opts::InvariantOptions;
    kwargs...,
) where {N}
    return _slice_chain_collect(
        pi,
        Float64[x0[i] for i in 1:N],
        Float64[dir[i] for i in 1:N],
        opts;
        kwargs...,
    )
end

@inline function _slice_chain_emit!(
    pi,
    x::Vector{Float64},
    t::Float64,
    strict0::Bool,
    drop_unknown::Bool,
    dedup::Bool,
    last_rid::Int,
    visitor,
)
    rid = locate(pi, x)
    return _slice_chain_emit_rid!(rid, t, strict0, drop_unknown, dedup, last_rid, visitor)
end

@inline function _slice_chain_emit_rid!(
    rid::Int,
    t::Float64,
    strict0::Bool,
    drop_unknown::Bool,
    dedup::Bool,
    last_rid::Int,
    visitor,
)
    if rid == 0
        if drop_unknown
            return last_rid
        end
        if strict0
            error("slice_chain: locate(pi, x) returned 0 (unknown region). Set opts.strict=false to allow unknown samples.")
        end
    end
    if dedup && rid == last_rid
        return last_rid
    end
    visitor(rid, t)
    return rid
end

@inline function _slice_chain_locate_many!(
    dest::Vector{Int},
    pi,
    X::Matrix{Float64},
    opts::InvariantOptions,
)
    threaded = opts.threads === nothing ? false : opts.threads
    return locate_many!(dest, pi, X; threaded=threaded)
end

mutable struct _SliceLocateBatchWorkspace
    X::Matrix{Float64}
    rids::Vector{Int}
end

@inline _SliceLocateBatchWorkspace() = _SliceLocateBatchWorkspace(Matrix{Float64}(undef, 0, 0), Int[])

@inline function _slice_chain_locate_buffers!(
    ws::_SliceLocateBatchWorkspace,
    d::Int,
    nsteps::Int,
)
    if size(ws.X, 1) != d || size(ws.X, 2) != nsteps
        ws.X = Matrix{Float64}(undef, d, nsteps)
    end
    length(ws.rids) == nsteps || resize!(ws.rids, nsteps)
    return ws.X, ws.rids
end

function _slice_chain_visit_range_batched(
    pi,
    x0f::Vector{Float64},
    dirf::Vector{Float64},
    opts::InvariantOptions,
    visitor,
    tlo::Float64,
    thi::Float64,
    nsteps::Int,
    strict0::Bool,
    drop_unknown::Bool,
    dedup::Bool,
    locate_ws = nothing,
)
    d = length(x0f)
    X, rids = locate_ws === nothing ?
        (Matrix{Float64}(undef, d, nsteps), Vector{Int}(undef, nsteps)) :
        _slice_chain_locate_buffers!(locate_ws, d, nsteps)
    step_t = nsteps <= 1 ? 0.0 : (thi - tlo) / (nsteps - 1)
    @inbounds for k in 1:d
        xk = x0f[k] + tlo * dirf[k]
        dx = step_t * dirf[k]
        for j in 1:nsteps
            X[k, j] = xk
            xk += dx
        end
    end
    _slice_chain_locate_many!(rids, pi, X, opts)
    last_rid = typemin(Int)
    t = tlo
    @inbounds for j in 1:nsteps
        last_rid = _slice_chain_emit_rid!(rids[j], t, strict0, drop_unknown, dedup, last_rid, visitor)
        j < nsteps && (t += step_t)
    end
    return nothing
end

function _slice_chain_visit(
    pi,
    x0f::Vector{Float64},
    dirf::Vector{Float64},
    opts::InvariantOptions,
    visitor;
    ts = nothing,
    tmin = nothing,
    tmax = nothing,
    nsteps::Int = 1001,
    box2 = nothing,
    drop_unknown::Bool = true,
    dedup::Bool = true,
    locate_ws = nothing,
)
    strict0 = opts.strict === nothing ? true : opts.strict
    clip = (opts.box !== nothing) || (box2 !== nothing)
    atol = 1e-12

    b1 = opts.box === nothing ? nothing : _resolve_box(pi, opts.box)
    b2 = box2 === nothing ? nothing : _resolve_box(pi, box2)
    bx = if clip
        if b1 === nothing && b2 === nothing
            clip = false
            nothing
        else
            _intersect_axis_aligned_boxes(b1, b2)
        end
    else
        nothing
    end

    x = similar(x0f)
    last_rid = typemin(Int)
    if ts === nothing
        tmin_eff = tmin
        tmax_eff = tmax
        if clip
            tlo, thi = _line_param_range_in_box_nd(x0f, dirf, bx; atol=atol)
            tmin_eff = tmin_eff === nothing ? tlo : max(float(tmin_eff), tlo)
            tmax_eff = tmax_eff === nothing ? thi : min(float(tmax_eff), thi)
        end
        if tmin_eff === nothing || tmax_eff === nothing
            error("slice_chain: must provide tmin/tmax unless clipping is active (opts.box or box2)")
        end
        float(tmin_eff) <= float(tmax_eff) || return nothing

        tlo = float(tmin_eff)
        thi = float(tmax_eff)
        if nsteps <= 1
            @inbounds for k in eachindex(x)
                x[k] = x0f[k] + tlo * dirf[k]
            end
            _slice_chain_emit!(pi, x, tlo, strict0, drop_unknown, dedup, last_rid, visitor)
            return nothing
        end
        if _slice_chain_use_batched_locate(pi, nsteps)
            return _slice_chain_visit_range_batched(
                pi,
                x0f,
                dirf,
                opts,
                visitor,
                tlo,
                thi,
                nsteps,
                strict0,
                drop_unknown,
                dedup,
                locate_ws,
            )
        end

        step_t = (thi - tlo) / (nsteps - 1)
        @inbounds for k in eachindex(x)
            x[k] = x0f[k] + tlo * dirf[k]
        end
        t = tlo
        @inbounds for step_idx in 1:nsteps
            last_rid = _slice_chain_emit!(pi, x, t, strict0, drop_unknown, dedup, last_rid, visitor)
            if step_idx < nsteps
                t += step_t
                for k in eachindex(x)
                    x[k] += step_t * dirf[k]
                end
            end
        end
        return nothing
    end

    tclip = clip ? _line_param_range_in_box_nd(x0f, dirf, bx; atol=atol) : (0.0, 0.0)
    @inbounds for traw in ts
        t = float(traw)
        if clip && !((tclip[1] - atol) <= t <= (tclip[2] + atol))
            continue
        end
        for k in eachindex(x)
            x[k] = x0f[k] + t * dirf[k]
        end
        last_rid = _slice_chain_emit!(pi, x, t, strict0, drop_unknown, dedup, last_rid, visitor)
    end
    return nothing
end

# Tuple adapters keep the compute path vector-specialized while accepting tuple inputs.
@inline function slice_chain(pi, x0::AbstractVector, dir::NTuple{N,<:Real},
                             opts::InvariantOptions; kwargs...) where {N}
    return slice_chain(pi, x0, Float64[dir[i] for i in 1:N], opts; kwargs...)
end

@inline function slice_chain(pi, x0::NTuple{N,<:Real}, dir::AbstractVector,
                             opts::InvariantOptions; kwargs...) where {N}
    return slice_chain(pi, Float64[x0[i] for i in 1:N], dir, opts; kwargs...)
end

@inline function slice_chain(pi, x0::NTuple{N,<:Real}, dir::NTuple{N,<:Real},
                             opts::InvariantOptions; kwargs...) where {N}
    return slice_chain(pi,
        Float64[x0[i] for i in 1:N],
        Float64[dir[i] for i in 1:N],
        opts; kwargs...)
end

"""
    slice_chain(pi::ZnEncodingMap, x0, dir, opts::InvariantOptions; kmin=0, kmax=100, kwargs...)

Convenience wrapper for Zn encodings: samples integer parameters `kmin:kmax`.
"""
function slice_chain(pi::ZnEncodingMap, x0::AbstractVector{<:Integer}, dir::AbstractVector{<:Integer},
    opts::InvariantOptions;
    kmin::Int = 0,
    kmax::Int = 100,
    kwargs...)

    ts = kmin:kmax
    return invoke(slice_chain, Tuple{Any, AbstractVector, AbstractVector, InvariantOptions},
        pi, x0, dir, opts;
        ts=ts,
        kwargs...)
end

function slice_chain(pi::CompiledEncoding{<:ZnEncodingMap}, x0::AbstractVector{<:Integer}, dir::AbstractVector{<:Integer},
                     opts::InvariantOptions; kwargs...)
    return slice_chain(pi.pi, x0, dir, opts; kwargs...)
end



# Helper: assert that `chain` is a chain in the poset Q (consecutive comparability suffices).
function _assert_chain(Q, chain::AbstractVector{Int})
    m = length(chain)
    m >= 1 || error("expected a nonempty chain")
    for q in chain
        (1 <= q <= nvertices(Q)) || error("chain vertex out of range: $q")
    end
    for i in 1:m-1
        leq(Q, chain[i], chain[i+1]) || error("not a chain: chain[$i]=$(chain[i]) is not <= chain[$(i+1)]=$(chain[i+1])")
    end
    return nothing
end

# Helper: sanity-check monotonicity of a sampled chain along a line.
function _check_chain_monotone(pi, x0, dir, chain::AbstractVector{Int}, tvals::AbstractVector;
    strict::Bool=true)
    length(chain) == length(tvals) || error("slice_chain: chain/tvals length mismatch")
    # tvals should be nondecreasing.
    for i in 1:(length(tvals)-1)
        tvals[i] <= tvals[i+1] || error("slice_chain: tvals are not nondecreasing")
    end

    # If the encoding map exposes a poset, ensure the labels form a chain.
    Q = if hasproperty(pi, :P)
        getproperty(pi, :P)
    elseif hasproperty(pi, :Q)
        getproperty(pi, :Q)
    else
        nothing
    end
    if strict
        if Q !== nothing
            _assert_chain(Q, chain)
        else
            # In strict mode, require no unknown region labels.
            any(q -> q == 0, chain) && error("slice_chain: encountered unknown region label 0 in strict mode")
        end
    end
    return nothing
end

# Extend a length-m vector of parameter values to length m+1 by adding one extra endpoint.
# This lets us encode half-open intervals [birth, death) with death == m+1.
function _extend_values(values::AbstractVector)
    m = length(values)
    m >= 1 || error("values must be nonempty")
    if m == 1
        step = one(values[1])
    else
        step = values[end] - values[end-1]
        step == zero(step) && (step = one(step))
    end
    return vcat(values, values[end] + step)
end

struct _ExtendedValueView{T,V<:AbstractVector{T}} <: AbstractVector{T}
    values::V
    tail::T
end

@inline Base.size(v::_ExtendedValueView) = (length(v.values) + 1,)
@inline Base.length(v::_ExtendedValueView) = length(v.values) + 1
@inline Base.IndexStyle(::Type{<: _ExtendedValueView}) = IndexLinear()
@inline function Base.getindex(v::_ExtendedValueView{T}, i::Int) where {T}
    1 <= i <= length(v) || throw(BoundsError(v, i))
    return i <= length(v.values) ? v.values[i] : v.tail
end

@inline function _extended_values_view(values::AbstractVector{T}) where {T}
    m = length(values)
    m >= 1 || error("values must be nonempty")
    step = if m == 1
        one(values[1])
    else
        s = values[end] - values[end - 1]
        iszero(s) ? one(s) : s
    end
    return _ExtendedValueView{T,typeof(values)}(values, values[end] + step)
end

"""
    restrict_to_chain(M, chain) -> PModule{K}

Restrict a finite-poset module `M` to a chain of vertices.

The returned module lives on the chain poset {1,...,m} with arrows i -> i+1,
where m = length(chain). Its vector spaces are `M_{chain[i]}` and its structure
maps are the corresponding maps in `M`.

This is useful if you want to inspect the actual matrices along the slice.
For interval decomposition / barcodes, `slice_barcode` is usually the better entry point.
"""
function restrict_to_chain(M::PModule{K}, chain::AbstractVector{Int})::PModule{K} where {K}
    _assert_chain(M.Q, chain)
    m = length(chain)

    # Chain poset: i <= j iff i <= j.
    leq = falses(m, m)
    for i in 1:m
        for j in i:m
            leq[i, j] = true
        end
    end
    Qc = FinitePoset(leq; check=false)

    dims = [M.dims[chain[i]] for i in 1:m]

    # Only need cover edges (i,i+1).
    edge_maps = Dict{Tuple{Int,Int}, Matrix{K}}()
    cc = _get_cover_cache(M.Q)
    nQ = nvertices(M.Q)
    memo = _use_array_memo(nQ) ? _new_array_memo(K, nQ) : Dict{Tuple{Int,Int}, AbstractMatrix{K}}()
    for i in 1:m-1
        edge_maps[(i, i+1)] = Matrix(_map_leq_cached(M, chain[i], chain[i+1], cc, memo))
    end
    return PModule{K}(Qc, dims, edge_maps)
end

"""
    slice_barcode(M, chain; values=nothing, check_chain=true) -> Dict{Tuple{T,T},Int}

Compute the interval decomposition (1-parameter barcode) of the restriction of `M`
to a chain `chain = [q1, ..., qm]` with q1 <= ... <= qm.

The result is returned as a sparse dictionary encoding a multiset of half-open intervals:
    (birth, death) => multiplicity

Conventions:
- Births are indexed by chain position 1..m.
- Deaths are indexed by 2..m+1, where death == m+1 means "persists to the end".
- Intervals are half-open: [birth, death).

If `values` is provided, endpoints are taken from `values` instead of integer positions.
Acceptable lengths:
- length(values) == m: we extend by one extra endpoint to represent death == m+1.
- length(values) == m+1: used as-is.

Implementation note:
We compute the A_m-interval multiplicities by inclusion-exclusion on the 1D rank invariant
along the chain:
  mult(b,d) = r[b, d-1] - r[b-1, d-1] - r[b, d] + r[b-1, d]
where r[i,j] = rank(M(q_i -> q_j)) and out-of-range r is treated as 0.
"""
function slice_barcode(M::PModule{K}, chain::AbstractVector{Int};
    values=nothing,
    check_chain::Bool=true
) where {K}
    return _barcode_from_packed(
        _slice_barcode_packed(M, chain; values=values, check_chain=check_chain)
    )
end

function _slice_barcode_packed(
    M::PModule{K},
    chain::AbstractVector{Int};
    values=nothing,
    check_chain::Bool=true
) where {K}
    check_chain && _assert_chain(M.Q, chain)
    m = length(chain)

    endpoints = if values === nothing
        1:(m + 1)
    else
        length(values) == m || length(values) == m + 1 ||
            error("slice_barcode: values must have length m or m+1")
        length(values) == m ? _extended_values_view(values) : values
    end

    cc = _get_cover_cache(M.Q)
    nQ = nvertices(M.Q)
    memo = _use_array_memo(nQ) ? _new_array_memo(K, nQ) : Dict{Tuple{Int,Int}, AbstractMatrix{K}}()
    R = zeros(Int, m, m)
    @inbounds for i in 1:m
        for j in i:m
            R[i, j] = rank_map(M, chain[i], chain[j]; cache=cc, memo=memo)
        end
    end

    return _packed_barcode_from_rank(R, endpoints)
end

@inline function _slice_barcode_packed_with_workspace(
    M::PModule{K},
    chain::AbstractVector{Int},
    endpoints,
    cc,
    memo,
    Rwork::Matrix{Int},
) where {K}
    m = length(chain)
    m == 0 && return _empty_packed_float_barcode()
    @views R = Rwork[1:m, 1:m]
    @inbounds for i in 1:m
        for j in i:m
            R[i, j] = rank_map(M, chain[i], chain[j]; cache=cc, memo=memo)
        end
    end
    return _packed_barcode_from_rank(R, endpoints)
end

# Convenience wrapper: build the chain from a slice in the original domain first.
function slice_barcode(M::PModule{K}, pi, x0::AbstractVector, dir::AbstractVector,
                       opts::InvariantOptions=InvariantOptions(); kwargs...) where {K}
    chain, tvals = slice_chain(pi, x0, dir, opts; kwargs...)
    return slice_barcode(M, chain; values = tvals)
end


# ----- Bottleneck distance on 1D barcodes -------------------------------------

# Expand a sparse barcode dictionary to a multiset of diagram points.
@inline _barcode_points(bar::Vector{Tuple{Float64,Float64}})::Vector{Tuple{Float64,Float64}} = bar
@inline _barcode_points(bar::PackedFloatBarcode)::Vector{Tuple{Float64,Float64}} =
    _points_from_packed!(Tuple{Float64,Float64}[], bar)

@inline function _barcode_points!(out::Vector{Tuple{Float64,Float64}},
                                  bar::Vector{Tuple{Float64,Float64}})
    empty!(out)
    append!(out, bar)
    return out
end

@inline function _barcode_points!(out::Vector{Tuple{Float64,Float64}},
                                  bar::PackedFloatBarcode)
    return _points_from_packed!(out, bar)
end

@inline function _barcode_points(bar::PackedIndexBarcode)::Vector{Tuple{Float64,Float64}}
    out = Tuple{Float64,Float64}[]
    sizehint!(out, _packed_total_multiplicity(bar))
    @inbounds for i in eachindex(bar.pairs)
        p = bar.pairs[i]
        m = bar.mults[i]
        b = float(p.b)
        d = float(p.d)
        for _ in 1:m
            push!(out, (b, d))
        end
    end
    return out
end

@inline function _barcode_points!(out::Vector{Tuple{Float64,Float64}},
                                  bar::PackedIndexBarcode)
    empty!(out)
    sizehint!(out, _packed_total_multiplicity(bar))
    @inbounds for i in eachindex(bar.pairs)
        p = bar.pairs[i]
        m = bar.mults[i]
        b = float(p.b)
        d = float(p.d)
        for _ in 1:m
            push!(out, (b, d))
        end
    end
    return out
end

function _barcode_points(bar)::Vector{Tuple{Float64,Float64}}
    if bar isa AbstractVector
        pts = Tuple{Float64,Float64}[]
        for I in bar
            b, d = I
            push!(pts, (float(b), float(d)))
        end
        return pts
    elseif bar isa AbstractDict
        pts = Tuple{Float64,Float64}[]
        for (I, mult) in bar
            b, d = I
            for _ in 1:Int(mult)
                push!(pts, (float(b), float(d)))
            end
        end
        return pts
    else
        error("barcode_points: expected a vector of intervals or a dictionary")
    end
end

function _barcode_points!(out::Vector{Tuple{Float64,Float64}}, bar)
    empty!(out)
    if bar isa AbstractVector
        sizehint!(out, length(bar))
        for I in bar
            b, d = I
            push!(out, (float(b), float(d)))
        end
        return out
    elseif bar isa AbstractDict
        n = 0
        for mult in values(bar)
            n += Int(mult)
        end
        sizehint!(out, n)
        for (I, mult) in bar
            b, d = I
            bf = float(b)
            df = float(d)
            for _ in 1:Int(mult)
                push!(out, (bf, df))
            end
        end
        return out
    end
    error("barcode_points!: expected a vector of intervals or a dictionary")
end

#--------------------------------------------------------------------
# Wasserstein distance and kernels for 1-parameter barcodes.
#--------------------------------------------------------------------

function _point_distance(a::Tuple{<:Real,<:Real}, b::Tuple{<:Real,<:Real}, q::Real)
    dx = abs(float(a[1] - b[1]))
    dy = abs(float(a[2] - b[2]))
    if q == Inf
        return max(dx, dy)
    elseif q == 2
        return sqrt(dx * dx + dy * dy)
    elseif q == 1
        return dx + dy
    else
        error("_point_distance: supported q are 1, 2, Inf")
    end
end

function _diag_distance(a::Tuple{<:Real,<:Real}, q::Real)
    d = abs(float(a[2] - a[1]))
    if q == Inf
        return d / 2
    elseif q == 2
        return d / sqrt(2)
    elseif q == 1
        return d
    else
        error("_diag_distance: supported q are 1, 2, Inf")
    end
end

function _hungarian(cost::Matrix{Float64})
    n, m = size(cost)
    n == m || error("_hungarian: cost matrix must be square")

    u = zeros(Float64, n + 1)
    v = zeros(Float64, m + 1)
    p = zeros(Int, m + 1)
    way = zeros(Int, m + 1)

    for i in 1:n
        p[1] = i
        j0 = 1
        minv = fill(Inf, m + 1)
        used = fill(false, m + 1)
        way .= 0

        while true
            used[j0] = true
            i0 = p[j0]
            delta = Inf
            j1 = 1

            for j in 2:m+1
                if !used[j]
                    cur = cost[i0, j - 1] - u[i0 + 1] - v[j]
                    if cur < minv[j]
                        minv[j] = cur
                        way[j] = j0
                    end
                    if minv[j] < delta
                        delta = minv[j]
                        j1 = j
                    end
                end
            end

            for j in 1:m+1
                if used[j]
                    u[p[j] + 1] += delta
                    v[j] -= delta
                else
                    minv[j] -= delta
                end
            end

            j0 = j1
            if p[j0] == 0
                break
            end
        end

        while true
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 1
                break
            end
        end
    end

    assignment = zeros(Int, n)
    for j in 2:m+1
        i = p[j]
        if i != 0
            assignment[i] = j - 1
        end
    end

    total = 0.0
    for i in 1:n
        total += cost[i, assignment[i]]
    end
    return assignment, total
end

"""
    wasserstein_distance(bar1, bar2; p=2, q=Inf)

Compute the p-Wasserstein distance between two 1-parameter barcodes (persistence diagrams).

Barcodes use the same representation as `slice_barcode`, namely a dictionary
`Dict{Tuple{T,T},Int}` mapping (birth, death) to multiplicity.

Parameters:
- `p >= 1` is the Wasserstein exponent.
- `q` selects the ground metric on points in the plane: `q=Inf` (default) is L_infty,
  `q=2` is Euclidean, and `q=1` is L1.

Diagonal matching is included in the standard way. The return value is a Float64.
"""
@inline function _wasserstein_cost(i::Int, j::Int,
                                   P::Vector{Tuple{Float64,Float64}},
                                   Q::Vector{Tuple{Float64,Float64}},
                                   m::Int, n::Int,
                                   diagP::Vector{Float64},
                                   diagQ::Vector{Float64},
                                   q::Real, p::Real)
    if i <= m
        if j <= n
            return _point_distance(P[i], Q[j], q)^p
        else
            return diagP[i]^p
        end
    else
        if j <= n
            return diagQ[j]^p
        else
            return 0.0
        end
    end
end

function _auction_assignment(P::Vector{Tuple{Float64,Float64}},
                             Q::Vector{Tuple{Float64,Float64}};
                             p::Real=2, q::Real=Inf,
                             eps_factor::Real=5.0,
                             eps_min::Real=1e-6,
                             max_iters::Int=0)
    m = length(P)
    n = length(Q)
    N = m + n
    N == 0 && return Int[], 0.0

    diagP = Vector{Float64}(undef, m)
    for i in 1:m
        diagP[i] = _diag_distance(P[i], q)
    end
    diagQ = Vector{Float64}(undef, n)
    for j in 1:n
        diagQ[j] = _diag_distance(Q[j], q)
    end

    # Rough max cost for epsilon scaling.
    max_cost = 0.0
    for i in 1:m
        max_cost = max(max_cost, diagP[i]^p)
        for j in 1:n
            max_cost = max(max_cost, _point_distance(P[i], Q[j], q)^p)
        end
    end
    for j in 1:n
        max_cost = max(max_cost, diagQ[j]^p)
    end
    max_cost == 0.0 && return zeros(Int, N), 0.0

    epsilon = max_cost / 4
    eps_min = max(eps_min, max_cost * 1e-9)

    prices = zeros(Float64, N)
    owner = zeros(Int, N)   # item -> bidder
    assign = zeros(Int, N)  # bidder -> item

    max_iters == 0 && (max_iters = 10 * N * N)

    while epsilon > eps_min
        unassigned = Int[]
        for i in 1:N
            if assign[i] == 0
                push!(unassigned, i)
            end
        end

        iters = 0
        while !isempty(unassigned)
            iters += 1
            iters > max_iters && break

            i = pop!(unassigned)
            best = 0
            min1 = Inf
            min2 = Inf

            @inbounds for j in 1:N
                c = _wasserstein_cost(i, j, P, Q, m, n, diagP, diagQ, q, p) + prices[j]
                if c < min1
                    min2 = min1
                    min1 = c
                    best = j
                elseif c < min2
                    min2 = c
                end
            end

            min2 == Inf && (min2 = min1 + epsilon)
            bid = (min2 - min1) + epsilon
            prices[best] += bid

            prev = owner[best]
            owner[best] = i
            assign[i] = best
            if prev != 0
                assign[prev] = 0
                push!(unassigned, prev)
            end
        end

        epsilon /= eps_factor
    end

    total = 0.0
    for i in 1:N
        total += _wasserstein_cost(i, assign[i], P, Q, m, n, diagP, diagQ, q, p)
    end
    return assign, total
end

"""
    wasserstein_distance(bar1, bar2; p=2, q=Inf, backend=:auto)

Compute the p-Wasserstein distance between two 1-parameter barcodes (persistence diagrams).

`backend` options:
- `:auto`     (default): Hungarian for small diagrams, auction for larger ones
- `:hungarian`: always use Hungarian assignment
- `:auction`  : use auction algorithm with epsilon-scaling
"""
function _wasserstein_distance_points(
    P::Vector{Tuple{Float64,Float64}},
    Q::Vector{Tuple{Float64,Float64}};
    p::Real=2, q::Real=Inf, backend::Symbol=:auto,
)
    p >= 1 || error("wasserstein_distance: expected p >= 1")

    m = length(P)
    n = length(Q)
    N = m + n

    N == 0 && return 0.0

    use_hungarian = backend == :hungarian ||
                    (backend == :auto && N <= 30)
    if use_hungarian
        C = zeros(Float64, N, N)

        for i in 1:m
            for j in 1:n
                C[i, j] = _point_distance(P[i], Q[j], q)^p
            end
            for j in (n+1):N
                C[i, j] = _diag_distance(P[i], q)^p
            end
        end

        for i in (m+1):N
            for j in 1:n
                C[i, j] = _diag_distance(Q[j], q)^p
            end
        end

        _, cost = _hungarian(C)
        return cost^(1 / p)
    elseif backend == :auction || backend == :auto
        _, cost = _auction_assignment(P, Q; p=p, q=q)
        return cost^(1 / p)
    else
        error("wasserstein_distance: unknown backend=$(backend)")
    end
end

function wasserstein_distance(bar1, bar2; p::Real=2, q::Real=Inf, backend::Symbol=:auto)
    P = _barcode_points(bar1)
    Q = _barcode_points(bar2)
    return _wasserstein_distance_points(P, Q; p=p, q=q, backend=backend)
end

function wasserstein_distance(
    P::Vector{Tuple{Float64,Float64}},
    Q::Vector{Tuple{Float64,Float64}};
    p::Real=2, q::Real=Inf, backend::Symbol=:auto,
)
    return _wasserstein_distance_points(P, Q; p=p, q=q, backend=backend)
end

"""
    wasserstein_kernel(bar1, bar2; p=2, q=Inf, sigma=1.0, kind=:gaussian)

Kernels derived from the Wasserstein distance:

- `kind=:gaussian`  -> exp(-d^2/(2*sigma^2))
- `kind=:laplacian` -> exp(-d/sigma)
"""
function wasserstein_kernel(bar1, bar2; p::Real=2, q::Real=Inf, sigma::Real=1.0, kind::Symbol=:gaussian)
    sigma > 0 || error("wasserstein_kernel: sigma must be > 0")
    d = wasserstein_distance(bar1, bar2; p=p, q=q)
    if kind == :gaussian
        return exp(-(d * d) / (2 * float(sigma)^2))
    elseif kind == :laplacian
        return exp(-d / float(sigma))
    else
        error("wasserstein_kernel: unknown kind $(kind)")
    end
end

"""
    sliced_wasserstein_kernel(M, N, pi, opts::InvariantOptions; ...)

Compute a slice-averaged Wasserstein *kernel* between modules by:
1) building 1D barcodes on a family of slices, then
2) combining per-slice Wasserstein kernels (Gaussian by default).

This is an opts-primary API:
- `opts.box` is forwarded to slicing via the slice keywords (clipping the sampled line).
- `opts.strict` is forwarded to `locate` during slicing (defaults to `true`).

Notes:
- This wrapper uses `slice_kernel(...; kind=:wasserstein_gaussian, ...)`.
- `p` and `q` are forwarded to the underlying Wasserstein kernel on barcodes.
"""
function sliced_wasserstein_kernel(M::PModule{K}, N::PModule{K}, pi::PLikeEncodingMap, opts::InvariantOptions;
    directions = :auto,
    offsets = :auto,
    n_dirs::Integer = 100,
    n_offsets::Integer = 50,
    max_den::Integer = 8,
    include_axes::Bool = false,
    normalize_dirs::Symbol = :L1,

    # Weighting of directions/offsets in the final average.
    weight::Union{Symbol,Function,Real} = :lesnick_l1,
    offset_weights = nothing,
    normalize_weights::Bool = false,

    # Wasserstein kernel parameters.
    p::Real = 2,
    q = 1,
    sigma::Real = 1.0,
    lengthscale = nothing,

    # Optional second clipping box (intersected with opts.box).
    box2 = nothing,

    # Any extra slice-chain kwargs (tmin/tmax/nsteps, drop_unknown, dedup, etc.).
    slice_kwargs...) where {K}

    strict0 = opts.strict === nothing ? true : opts.strict

    # `slice_kernel` accepts `box`/`box2` via `slice_kwargs...` and forwards them to slicing.
    # We also pass `strict` explicitly (it is a real keyword on slice_kernel).
    return slice_kernel(M, N, pi;
        kind = :wasserstein_gaussian,
        directions = directions,
        offsets = offsets,
        n_dirs = n_dirs,
        n_offsets = n_offsets,
        max_den = max_den,
        include_axes = include_axes,
        normalize_dirs = normalize_dirs,

        # p/q are used by the Wasserstein kernels inside `_barcode_kernel`.
        p = p,
        q = q,

        sigma = sigma,
        lengthscale = lengthscale,

        direction_weight = weight,
        offset_weights = offset_weights,
        normalize_weights = normalize_weights,

        strict = strict0,

        # Forward clipping.
        box = opts.box,
        box2 = box2,

        slice_kwargs...)
end


# ---------------------------------------------------------------------------
# Internal helper: unify all slice-based distances in one implementation.
#
# There are two distinct ways "weights" are used in the literature/code:
#
#   (A) weight_mode = :integrate
#       W is treated as an integration weight matrix. Then:
#         agg = mean   -> sum(W .* d)   (a weighted mean if normalize_weights=true)
#         agg = :pmean -> (sum(W .* d.^p))^(1/p)   (Lp-type aggregate, if weights are normalized)
#
#   (B) weight_mode = :scale
#       W is treated as a multiplicative scaling matrix (matching-distance style).
#       In this mode the natural reduction is agg = maximum, giving max(W .* d).
#
# This helper is intentionally internal (leading underscore) to keep the public API
# small while avoiding code duplication across sliced and matching-style distances.
# ---------------------------------------------------------------------------
function _slice_based_barcode_distance(
    bcs1::AbstractMatrix,
    bcs2::AbstractMatrix;
    weights1 = uniform2d,
    weights2 = uniform2d,
    weight = (d, o) -> weights1(d) * weights2(o),
    dirs::AbstractVector{Tuple{Float64, Float64}},
    offs::AbstractVector{Tuple{Float64, Float64}},
    dist = :bottleneck,
    agg = :mean,
    agg_p = 1.0,
    agg_norm::Real = 1.0,
    threads::Bool = (Threads.nthreads() > 1),
)
    # Build separable weights and outer product.
    wdir = [direction_weight(d, weights1) for d in dirs]
    woff = [offset_weight(o, weights2) for o in offs]
    W = wdir .* transpose(woff)

    dist_fn = dist == :bottleneck ? bottleneck_distance :
              dist == :wasserstein ? wasserstein_distance :
              throw(ArgumentError("Unknown dist=$dist"))

    agg_mode = agg
    if !(agg_mode in (:mean, :pmean, :max)) && !(agg_mode isa Function)
        throw(ArgumentError("Unknown agg=$agg"))
    end

    nslices = length(bcs1)

    if threads && Threads.nthreads() > 1
        nT = Threads.nthreads()

        if agg_mode == :max
            best_by_slot = fill(0.0, nT)
            Threads.@threads for slot in 1:nT
                best = 0.0
                for idx in slot:nT:nslices
                    w = W[idx]
                    if w > 0
                        d = w * dist_fn(bcs1[idx], bcs2[idx])
                        if d > best
                            best = d
                        end
                    end
                end
                best_by_slot[slot] = best
            end
            return maximum(best_by_slot)
        end

        if agg_mode == :mean
            acc_by_slot = fill(0.0, nT)
            sumw_by_slot = fill(0.0, nT)
            Threads.@threads for slot in 1:nT
                acc = 0.0
                sumw = 0.0
                for idx in slot:nT:nslices
                    w = W[idx]
                    if w > 0
                        acc += w * dist_fn(bcs1[idx], bcs2[idx])
                        sumw += w
                    end
                end
                acc_by_slot[slot] = acc
                sumw_by_slot[slot] = sumw
            end
            acc = sum(acc_by_slot)
            sumw = sum(sumw_by_slot)
            return (sumw == 0.0) ? 0.0 : acc / sumw / float(agg_norm)
        end

        if agg_mode == :pmean
            p = float(agg_p)
            acc_by_slot = fill(0.0, nT)
            sumw_by_slot = fill(0.0, nT)
            Threads.@threads for slot in 1:nT
                acc = 0.0
                sumw = 0.0
                for idx in slot:nT:nslices
                    w = W[idx]
                    if w > 0
                        d = dist_fn(bcs1[idx], bcs2[idx])
                        acc += w * d^p
                        sumw += w
                    end
                end
                acc_by_slot[slot] = acc
                sumw_by_slot[slot] = sumw
            end
            acc = sum(acc_by_slot)
            sumw = sum(sumw_by_slot)
            if sumw == 0.0
                return 0.0
            end
            return (acc / sumw)^(1 / p) / float(agg_norm)
        end

        # Custom aggregator: compute all scaled distances in parallel, then call `agg`.
        vals = Vector{Float64}(undef, nslices)
        Threads.@threads for idx in 1:nslices
            vals[idx] = W[idx] * dist_fn(bcs1[idx], bcs2[idx])
        end
        return agg_mode(vals)
    end

    # Serial implementation (existing behavior).
    if agg_mode == :max
        best = 0.0
        for i in eachindex(bcs1)
            w = W[i]
            if w > 0
                best = max(best, w * dist_fn(bcs1[i], bcs2[i]))
            end
        end
        return best
    end

    if agg_mode == :mean
        acc = 0.0
        sumw = 0.0
        for i in eachindex(bcs1)
            w = W[i]
            if w > 0
                acc += w * dist_fn(bcs1[i], bcs2[i])
                sumw += w
            end
        end
        return (sumw == 0.0) ? 0.0 : acc / sumw / float(agg_norm)
    end

    if agg_mode == :pmean
        p = float(agg_p)
        acc = 0.0
        sumw = 0.0
        for i in eachindex(bcs1)
            w = W[i]
            if w > 0
                d = dist_fn(bcs1[i], bcs2[i])
                acc += w * d^p
                sumw += w
            end
        end
        return (sumw == 0.0) ? 0.0 : (acc / sumw)^(1 / p) / float(agg_norm)
    end

    vals = Float64[]
    for i in eachindex(bcs1)
        w = W[i]
        if w > 0
            push!(vals, w * dist_fn(bcs1[i], bcs2[i]))
        end
    end
    return agg_mode(vals)
end



# Internal helper: slice-based distance between two modules via an encoding map.
# This version builds slice barcodes (and weights) using `slice_barcodes`, then
# combines per-slice distances using either an integration-style aggregate
# (weight_mode = :integrate) or a matching-style scale-and-max (weight_mode = :scale).
function _slice_based_barcode_distance(
    M::PModule{K},
    N::PModule{K},
    pi::PLikeEncodingMap;
    dist_fn::Function = bottleneck_distance,
    dist_kwargs = NamedTuple(),
    weight_mode::Symbol = :integrate,
    dirs = :auto,
    offs = :auto,
    ndirs::Int = 16,
    noff::Int = 9,
    max_den::Integer = 8,
    include_axes::Bool = false,
    normalize_dirs::Symbol = :L1,
    weight::Union{Symbol,Function,Real} = :none,
    offset_weights = nothing,
    normalize_weights::Bool = (weight_mode == :integrate),
    agg = :mean,
    agg_p::Real = 2.0,
    agg_norm::Real = 1.0,
    threads::Bool = (Threads.nthreads() > 1),
    cache = nothing,
    slice_kwargs...
)::Float64 where {K}
    task = SliceDistanceTask(;
        dist_fn = dist_fn,
        dist_kwargs = dist_kwargs,
        weight_mode = weight_mode,
        agg = agg,
        agg_p = float(agg_p),
        agg_norm = float(agg_norm),
        threads = threads,
    )
    if cache === nothing
        box0 = get(slice_kwargs, :box, nothing)
        strict0 = get(slice_kwargs, :strict, nothing)
        slice_kwargs0 = (; (k => v for (k, v) in pairs(slice_kwargs) if k != :box && k != :strict)...)
        tmin0 = get(slice_kwargs0, :tmin, nothing)
        tmax0 = get(slice_kwargs0, :tmax, nothing)
        if box0 === nothing && tmin0 === nothing && tmax0 === nothing
            box0 = :auto
        end
        opts_direct = InvariantOptions(box = box0, strict = strict0, threads = threads)
        prep = _prepare_geometric_slice_query(
            pi,
            opts_direct;
            directions = dirs,
            offsets = offs,
            n_dirs = ndirs,
            n_offsets = noff,
            max_den = max_den,
            include_axes = include_axes,
            normalize_dirs = normalize_dirs,
            direction_weight = weight,
            offset_weights = offset_weights,
            normalize_weights = normalize_weights,
            slice_kwargs...,
        )
        isempty(prep.dirs_in) && return 0.0
        return _slice_pair_distance_geometric(
            M,
            N,
            prep.dirs_in,
            prep.offs,
            prep.weights,
            pi,
            prep.opts_chain,
            prep.filtered,
            task;
            drop_unknown = get(slice_kwargs, :drop_unknown, true),
        )
    end
    box0 = get(slice_kwargs, :box, nothing)
    strict0 = get(slice_kwargs, :strict, nothing)
    slice_kwargs0 = (; (k => v for (k, v) in pairs(slice_kwargs) if k != :box && k != :strict)...)
    tmin0 = get(slice_kwargs0, :tmin, nothing)
    tmax0 = get(slice_kwargs0, :tmax, nothing)
    if box0 === nothing && tmin0 === nothing && tmax0 === nothing
        box0 = :auto
    end
    opts_compile = InvariantOptions(box = box0, strict = strict0, threads = threads)
    plan = compile_slices(
        pi,
        opts_compile;
        directions = dirs,
        offsets = offs,
        n_dirs = ndirs,
        n_offsets = noff,
        max_den = max_den,
        include_axes = include_axes,
        normalize_dirs = normalize_dirs,
        direction_weight = weight,
        offset_weights = offset_weights,
        normalize_weights = normalize_weights,
        threads = threads,
        cache = cache,
        slice_kwargs0...
    )
    return run_invariants(plan, module_cache(M, N), task)
end

"""
        sliced_wasserstein_distance(M, N, pi, opts::InvariantOptions; ...)

A slice-based distance computed from 1D barcodes of line restrictions.

Approximate a sliced Wasserstein distance by:
- sampling directions/offsets,
- computing 1D Wasserstein distances on each slice barcode pair,
- combining via `agg` (mean/max/agg_p) and weighting.

Opts usage:
- `opts.strict` controls `locate` strictness (defaults to `true`).
- `opts.box` controls clipping of each slice chain.
- `opts.threads` controls parallel barcode computation (defaults to Threads.nthreads()>1).

Per-slice metric:
- Uses `wasserstein_distance` on the barcode for each slice.

Aggregation (key point for "Lp generalizations"):
- `agg = mean` or `agg = :mean` returns `sum(W .* d)` where `W` is the weight matrix from
  `slice_barcodes`. If `normalize_weights=true` (default), `sum(W) == 1`, so this is a
  genuine weighted mean.
- `agg = :pmean` returns `(sum(W .* d.^agg_p))^(1/agg_p)` (with `agg_p=Inf` giving a max).
  This is the clean "Lp-family" slice aggregate, matching the intention in your screenshot.
- `agg = maximum` or `agg = :maximum` returns `maximum(d)` across slices (unweighted).
- Any other callable `agg` is applied to the unweighted vector of per-slice distances.

Keywords `dirs`, `offsets`, `n_dirs`, `n_offsets`, `max_den`, `include_axes`,
`normalize_dirs`, `offset_weights`, `normalize_weights`, `offset_margin`, `weight`,
and `strict/box/box2` mirror the slicing machinery.

Wasserstein parameters:
- `p` is the Wasserstein exponent (p >= 1).
- `q` selects the ground metric on points (q=Inf is L_infty; q=2 Euclidean; q=1 L1).
"""
function sliced_wasserstein_distance(M::PModule{K}, N::PModule{K}, pi::PLikeEncodingMap, opts::InvariantOptions;
    directions = :auto,
    offsets = :auto,
    n_dirs::Integer = 16,
    n_offsets::Integer = 16,
    max_den::Integer = 8,
    include_axes::Bool = false,
    normalize_dirs::Symbol = :L1,
    offset_margin::Real = 0.05,

    # Wasserstein distance params.
    p = 2,
    q = 1,

    # Aggregation over slices.
    agg::Symbol = :mean,
    agg_p::Real = 2.0,
    agg_norm::Real = 1.0,

    # Weighting.
    weight::Union{Symbol,Function,Real} = :lesnick_l1,
    offset_weights = :cosine,
    normalize_weights::Bool = true,

    # Optional second clipping box.
    box2 = nothing,
    cache = nothing,

    slice_kwargs...)::Float64 where {K}

    strict0 = opts.strict === nothing ? true : opts.strict
    threads0 = opts.threads === nothing ? (Threads.nthreads() > 1) : opts.threads

    return _slice_based_barcode_distance(M, N, pi;
        dirs = directions,
        offs = offsets,
        ndirs = n_dirs,
        noff = n_offsets,
        max_den = max_den,
        include_axes = include_axes,
        normalize_dirs = normalize_dirs,
        offset_margin = offset_margin,

        # Slice distance definition.
        dist_fn = wasserstein_distance,
        dist_kwargs = (p=p, q=q),

        # Combining / weighting.
        agg = agg,
        agg_p = agg_p,
        agg_norm = agg_norm,
        weight_mode = :integrate,
        weight = weight,
        offset_weights = offset_weights,
        normalize_weights = normalize_weights,

        threads = threads0,
        cache = cache,

        # Windowing / strictness now come from opts.
        strict = strict0,
        box = opts.box,
        box2 = box2,

        slice_kwargs...)
end


"""
    sliced_bottleneck_distance(M, N, pi, opts::InvariantOptions; ...)

Exactly like `sliced_wasserstein_distance`, but the per-slice metric is
`bottleneck_distance` instead of `wasserstein_distance`.

This is the "sliced bottleneck" member of the same slice-based family.

Aggregation:
- `agg = mean` or `agg = :mean` returns `sum(W .* d)` (weighted mean if weights are normalized).
- `agg = :pmean` returns `(sum(W .* d.^agg_p))^(1/agg_p)` with `agg_p=Inf` giving a max.
- `agg = maximum` or `agg = :maximum` returns `maximum(d)` across slices (unweighted).
- Any other callable `agg` is applied to the unweighted vector of per-slice distances.

Approximate a sliced bottleneck distance by:
- sampling directions/offsets,
- computing 1D bottleneck distances on each slice barcode pair,
- combining via `agg` and weighting.

Opts usage:
- `opts.strict` controls `locate` strictness (defaults to `true`).
- `opts.box` controls clipping of each slice chain.
- `opts.threads` controls parallel barcode computation.
"""
function sliced_bottleneck_distance(M::PModule{K}, N::PModule{K}, pi::PLikeEncodingMap, opts::InvariantOptions;
    directions = :auto,
    offsets = :auto,
    n_dirs::Integer = 16,
    n_offsets::Integer = 16,
    max_den::Integer = 8,
    include_axes::Bool = false,
    normalize_dirs::Symbol = :L1,
    offset_margin::Real = 0.05,

    agg::Symbol = :mean,
    agg_p::Real = 2.0,
    agg_norm::Real = 1.0,

    weight::Union{Symbol,Function,Real} = :lesnick_l1,
    offset_weights = :cosine,
    normalize_weights::Bool = true,

    box2 = nothing,
    cache = nothing,

    slice_kwargs...)::Float64 where {K}

    strict0 = opts.strict === nothing ? true : opts.strict
    threads0 = opts.threads === nothing ? (Threads.nthreads() > 1) : opts.threads

    return _slice_based_barcode_distance(M, N, pi;
        dirs = directions,
        offs = offsets,
        ndirs = n_dirs,
        noff = n_offsets,
        max_den = max_den,
        include_axes = include_axes,
        normalize_dirs = normalize_dirs,
        offset_margin = offset_margin,

        dist_fn = bottleneck_distance,
        dist_kwargs = NamedTuple(),

        agg = agg,
        agg_p = agg_p,
        agg_norm = agg_norm,
        weight_mode = :integrate,
        weight = weight,
        offset_weights = offset_weights,
        normalize_weights = normalize_weights,

        threads = threads0,
        cache = cache,

        strict = strict0,
        box = opts.box,
        box2 = box2,

        slice_kwargs...)
end



_diag_cost(p::Tuple{Float64,Float64}) = 0.5 * abs(p[2] - p[1])
_linf_dist(p::Tuple{Float64,Float64}, q::Tuple{Float64,Float64}) = max(abs(p[1] - q[1]), abs(p[2] - q[2]))

# Hopcroft-Karp maximum matching for bipartite graphs given by adjacency lists.
function _hopcroft_karp(adj::Vector{Vector{Int}}, n_left::Int, n_right::Int)::Int
    pairU = fill(0, n_left)
    pairV = fill(0, n_right)
    dist = fill(0, n_left)
    INF = typemax(Int)

    function bfs()::Bool
        q = Int[]
        for u in 1:n_left
            if pairU[u] == 0
                dist[u] = 0
                push!(q, u)
            else
                dist[u] = INF
            end
        end

        found = false
        qi = 1
        while qi <= length(q)
            u = q[qi]
            qi += 1
            for v in adj[u]
                u2 = pairV[v]
                if u2 == 0
                    found = true
                elseif dist[u2] == INF
                    dist[u2] = dist[u] + 1
                    push!(q, u2)
                end
            end
        end
        return found
    end

    function dfs(u::Int)::Bool
        for v in adj[u]
            u2 = pairV[v]
            if u2 == 0 || (dist[u2] == dist[u] + 1 && dfs(u2))
                pairU[u] = v
                pairV[v] = u
                return true
            end
        end
        dist[u] = INF
        return false
    end

    matching = 0
    while bfs()
        for u in 1:n_left
            if pairU[u] == 0 && dfs(u)
                matching += 1
            end
        end
    end
    return matching
end

function _bottleneck_leq_eps(A::Vector{Tuple{Float64,Float64}}, B::Vector{Tuple{Float64,Float64}}, eps::Float64)::Bool
    m = length(A)
    n = length(B)

    # Left side: A points + n diagonal copies.
    # Right side: B points + m diagonal copies.
    n_left = m + n
    n_right = n + m

    diag_nodes = n+1:n+m  # right diagonal indices (may be empty)

    # Precompute which B points can be sent to the diagonal within eps.
    Bok = Int[]
    for j in 1:n
        if _diag_cost(B[j]) <= eps
            push!(Bok, j)
        end
    end

    adj = Vector{Vector{Int}}(undef, n_left)
    for u in 1:n_left
        adj[u] = Int[]
    end

    # A points.
    for i in 1:m
        p = A[i]
        neigh = adj[i]
        # to B points
        for j in 1:n
            if _linf_dist(p, B[j]) <= eps
                push!(neigh, j)
            end
        end
        # to diagonal (any right diagonal copy)
        if _diag_cost(p) <= eps
            for v in diag_nodes
                push!(neigh, v)
            end
        end
    end

    # Left diagonal copies (one per B point, but all identical as diagonal points).
    for k in 1:n
        neigh = adj[m + k]
        # match B points to diagonal when allowed
        for j in Bok
            push!(neigh, j)
        end
        # diagonal to diagonal always allowed (cost 0)
        for v in diag_nodes
            push!(neigh, v)
        end
    end

    match_size = _hopcroft_karp(adj, n_left, n_right)
    return match_size == n_left
end

"""
    bottleneck_distance(barA, barB) -> Float64

Compute the bottleneck distance between two 1D barcodes.

Inputs can be:
- a dictionary `(birth, death) => multiplicity` as returned by `slice_barcode`, or
- a vector of intervals `[(birth, death), ...]` (multiplicity by repetition).

We use the standard persistence-diagram L_infinity metric:
- cost((b,d),(b',d')) = max(|b-b'|, |d-d'|)
- cost((b,d), diagonal) = (d-b)/2

This implementation is exact for the given finite barcodes (up to floating-point).
"""
function _bottleneck_distance_points(
    A::Vector{Tuple{Float64,Float64}},
    B::Vector{Tuple{Float64,Float64}};
    backend::Symbol=:auto,
)::Float64
    if isempty(A) && isempty(B)
        return 0.0
    end

    backend == :auto && (backend = :hk)
    backend == :hk || error("bottleneck_distance: unknown backend=$(backend)")

    # Candidate eps values: all pairwise distances and diagonal costs.
    epss = Float64[0.0]
    for p in A
        push!(epss, _diag_cost(p))
    end
    for q in B
        push!(epss, _diag_cost(q))
    end
    for p in A
        for q in B
            push!(epss, _linf_dist(p, q))
        end
    end
    sort!(epss)
    epss = unique(epss)

    lo = 1
    hi = length(epss)
    while lo < hi
        mid = (lo + hi) >>> 1
        if _bottleneck_leq_eps(A, B, epss[mid])
            hi = mid
        else
            lo = mid + 1
        end
    end
    return epss[lo]
end

function bottleneck_distance(barA, barB; backend::Symbol=:auto)::Float64
    A = _barcode_points(barA)
    B = _barcode_points(barB)
    return _bottleneck_distance_points(A, B; backend=backend)
end

# Convenience: bottleneck distance between slice barcodes of two modules on the same chain.
function bottleneck_distance(M::PModule{K}, N::PModule{K}, chain::AbstractVector{Int}; kwargs...)::Float64 where {K}
    bM = slice_barcode(M, chain; kwargs...)
    bN = slice_barcode(N, chain; kwargs...)
    return bottleneck_distance(bM, bN)
end

# Aliases used by slice-based distance wrappers.
# These accept Wasserstein-style kwargs for API convenience.
matching_distance(barA, barB; kwargs...)::Float64 = bottleneck_distance(barA, barB)
matching_wasserstein_distance(barA, barB; p::Real=2, q::Real=1, kwargs...)::Float64 =
    wasserstein_distance(barA, barB; p=p, q=q, kwargs...)

# ----- Approximate matching distance ------------------------------------------

function _direction_weight(dir::Union{AbstractVector,NTuple{N,<:Real}}; scheme::Symbol=:lesnick_l1)::Float64 where {N}
    if scheme == :none
        return 1.0
    elseif scheme == :lesnick_l1
        # Normalize to L1 = 1 and take min coordinate.
        s = 0.0
        for x in dir
            x < 0 && error("direction_weight: expected nonnegative direction entries")
            s += float(x)
        end
        s == 0.0 && error("direction_weight: zero direction vector")
        mn = Inf
        for x in dir
            mn = min(mn, float(x) / s)
        end
        return float(mn)
    elseif scheme == :lesnick_linf
        # Normalize so max = 1 and take min coordinate.
        mx = 0.0
        for x in dir
            x < 0 && error("direction_weight: expected nonnegative direction entries")
            mx = max(mx, float(x))
        end
        mx == 0.0 && error("direction_weight: zero direction vector")
        mn = Inf
        for x in dir
            mn = min(mn, float(x) / mx)
        end
        return float(mn)
    else
        error("direction_weight: unknown scheme $scheme")
    end
end

# --------------------------------------------------------------------------
# Lightweight weight helpers used by slice-based invariants.
# These are tiny and allocation-free for inner loops.
# --------------------------------------------------------------------------

"""
    uniform2d(::Any) -> Float64

Trivial weight function returning 1.0. Used as the default for unweighted
direction/offset sampling.
"""
@inline uniform2d(::Any)::Float64 = 1.0

"""
    direction_weight(dir, spec) -> Float64

Evaluate a direction weight.

`spec` can be:
- a `Symbol` (e.g. `:lesnick_l1`, `:lesnick_linf`, `:none`)
- a function `w(dir)::Real`
- a real scalar (constant weight)
"""
@inline function direction_weight(dir::Union{AbstractVector{<:Real},NTuple{N,<:Real}}, spec)::Float64 where {N}
    if spec === :none || spec === :uniform
        return 1.0
    elseif spec isa Symbol
        return _direction_weight(dir; scheme=spec)
    elseif spec isa Function
        return Float64(spec(dir))
    elseif spec isa Real
        return Float64(spec)
    else
        throw(ArgumentError("direction_weight: unsupported spec type $(typeof(spec))"))
    end
end

"""
    offset_weight((c0,c1), spec) -> Float64

Weight for an *interval* of offsets (used for arrangement-cell weighting).

`spec` can be:
- `:uniform` / `:none`  -> 1.0
- `:length`             -> c1 - c0
- a function `w((c0,c1))::Real`
- a real scalar
"""
@inline function offset_weight(interval::Tuple{<:Real,<:Real}, spec)::Float64
    if spec === :none || spec === :uniform
        return 1.0
    elseif spec === :length
        return Float64(interval[2] - interval[1])
    elseif spec isa Function
        return Float64(spec(interval))
    elseif spec isa Real
        return Float64(spec)
    else
        throw(ArgumentError("offset_weight: unsupported spec $(spec)"))
    end
end

# For finite sampled offsets (slice_barcodes sampling). Distinct from interval weights.
function _offset_sample_weights(offs::AbstractVector, spec)
    n = length(offs)
    w = ones(Float64, n)
    if spec === nothing || spec === :none || spec === :uniform
        return w
    elseif spec === :cosine
        # Smooth window; normalize afterwards if requested.
        for i in 1:n
            t = (i - 0.5) / n
            w[i] = sin(pi * t)
        end
        return w
    elseif spec isa AbstractVector
        length(spec) == n || throw(ArgumentError("_offset_sample_weights: expected length(offset_weights)==length(offsets)"))
        for i in 1:n
            w[i] = Float64(spec[i])
        end
        return w
    elseif spec isa Function
        for i in 1:n
            w[i] = Float64(spec(offs[i]))
        end
        return w
    else
        throw(ArgumentError("_offset_sample_weights: unsupported offset_weights $(spec)"))
    end
end

"""
    sample_directions_2d(; max_den=8, include_axes=false, normalize=:L1) -> Vector{NTuple{2}}

Return a deterministic list of "standard" direction vectors in 2D.

This helper is intended for slice-based workflows such as:
* approximating the matching distance via `matching_distance_approx`,
* sampling fibered barcodes (RIVET-style) by sweeping a set of slopes.

Directions are generated from primitive integer pairs `(a,b)` with
`0 <= a,b <= max_den`, `gcd(a,b)=1`, and `(a,b) != (0,0)`.

By default we *exclude* the coordinate axes (require `a>0` and `b>0`), because
matching-distance slices typically assume strictly positive directions.
Set `include_axes=true` to also include axis directions.

The list is sorted by increasing slope `b/a` (with slope = +Inf when `a==0`).

Normalization:
* `normalize=:L1` (default): return Float64 directions with `d[1]+d[2]==1`.
* `normalize=:Linf`: return Float64 directions with `max(d)==1`.
* `normalize=:none`: return integer directions as 2-tuples (useful for Z^2).

Examples
--------

R^2 / PL encodings:

    dirs = sample_directions_2d(max_den=5)
    d = matching_distance_approx(M, N, pi; directions=dirs, offsets=[x0], tmin=0, tmax=1)

Z^2 encodings (integer directions and integer steps):

    dirsZ = sample_directions_2d(max_den=5; normalize=:none)
    d = matching_distance_approx(M, N, pi; directions=dirsZ, offsets=[g0], kmin=0, kmax=50)
"""
function sample_directions_2d(; max_den::Int=8, include_axes::Bool=false, normalize::Symbol=:L1)
    max_den >= 1 || error("sample_directions_2d: max_den must be >= 1")

    pairs = Tuple{Int,Int}[]
    for a in 0:max_den, b in 0:max_den
        (a == 0 && b == 0) && continue
        if !include_axes && (a == 0 || b == 0)
            continue
        end
        gcd(a, b) == 1 || continue
        push!(pairs, (a, b))
    end

    # Sort by slope b/a (a==0 treated as +Inf).
    slope(p::Tuple{Int,Int}) = p[1] == 0 ? Inf : (float(p[2]) / float(p[1]))
    sort!(pairs; by=slope)

    if normalize == :none
        return [(p[1], p[2]) for p in pairs]
    elseif normalize == :L1
        return [begin
            s = float(p[1] + p[2])
            (float(p[1]) / s, float(p[2]) / s)
        end for p in pairs]
    elseif normalize == :Linf
        return [begin
            s = float(max(p[1], p[2]))
            (float(p[1]) / s, float(p[2]) / s)
        end for p in pairs]
    else
        error("sample_directions_2d: normalize must be :L1, :Linf, or :none")
    end
end


# ----- Defaults for sliced invariants ------------------------------------------

"""
    encoding_box(pi::PLikeEncodingMap, opts::InvariantOptions; margin=0.05) -> (lo, hi)
    encoding_box(axes::Tuple{Vararg{<:AbstractVector}}, opts::InvariantOptions; margin=0.05) -> (lo, hi)
Return an axis-aligned bounding box for an encoding.

Opt override rules:

- If `opts.box` is a concrete box `(lo, hi)`, it is returned (normalized to Float64)
  and `margin` is ignored (exactly as the old `box=...` override behavior).
- If `opts.box === :auto`, we use `window_box(pi)` as the base box and then expand
  it by `margin`.
- If `opts.box === nothing`, we infer the box from representative points (and for
  an explicit axis tuple, from axis extents) and expand by `margin`.

The returned box is `(lo::Vector{Float64}, hi::Vector{Float64})`.
"""
function encoding_box(pi::PLikeEncodingMap, opts::InvariantOptions; margin::Real = 0.05)
    # Explicit concrete override: return it as-is (normalized), ignore margin.
    if opts.box !== nothing && opts.box !== :auto
        return _normalize_box(opts.box)
    end

    # Special override: :auto means "use window_box(pi)" as the base.
    if opts.box === :auto
        lo, hi = _normalize_box(window_box(pi))
        _apply_margin!(lo, hi, margin)
        return (lo, hi)
    end

    # Inference from representatives (default behavior).
    reps = representatives(pi)
    n = length(first(reps))

    lo = fill(Inf, n)
    hi = fill(-Inf, n)

    for r in reps
        for i in 1:n
            x = float(r[i])
            if x < lo[i]
                lo[i] = x
            end
            if x > hi[i]
                hi[i] = x
            end
        end
    end

    _apply_margin!(lo, hi, margin)
    return (lo, hi)
end

function encoding_box(axes::Tuple{Vararg{<:AbstractVector}}, opts::InvariantOptions; margin::Real = 0.05)
    # For an explicit axis tuple, :auto is treated the same as "infer" (default).
    if opts.box !== nothing && opts.box !== :auto
        return _normalize_box(opts.box)
    end

    lo = Float64[]
    hi = Float64[]
    for a in axes
        push!(lo, float(a[1]))
        push!(hi, float(a[end]))
    end

    _apply_margin!(lo, hi, margin)
    return (lo, hi)
end


function _normalize_box(box)
    lo, hi = box
    lo_v = Float64[float(x) for x in lo]
    hi_v = Float64[float(x) for x in hi]
    length(lo_v) == length(hi_v) || error("encoding_box: box endpoints must have same dimension")
    for i in 1:length(lo_v)
        lo_v[i] <= hi_v[i] || error("encoding_box: expected lo[i] <= hi[i] for all i")
    end
    return lo_v, hi_v
end

function _apply_margin!(lo::AbstractVector{<:Real}, hi::AbstractVector{<:Real}, margin::Real)
    @inbounds for i in 1:length(lo)
        m = (hi[i] - lo[i]) * float(margin)
        lo[i] -= m
        hi[i] += m
    end
    return lo, hi
end


"""
    window_box(pi::PLikeEncodingMap; padding=0.0, margin=0.05, integerize=:auto, method::Symbol=:reps)

Return a finite axis-aligned window `(ell, u)` in the parameter space of `pi`.

Backend-agnostic: relies only on:
- `representatives(pi)` (required)
- `dimension(pi)` (required)
- `axes_from_encoding(pi)` (optional; used to avoid degenerate axes)

Parameters:
- `padding`: absolute padding added to each side.
- `margin`: relative padding (fraction of side-length).
- `integerize`: `:auto`, `:always`, or `:never`.
- `method`: `:reps`, `:coords` (uses `axes_from_encoding`), or `:mix`.

If any axis is still degenerate after inference, we expand it by a minimal
width (1 for lattice encodings, 1.0 otherwise) before applying margin/padding.
"""
function window_box(pi::PLikeEncodingMap; padding=0.0, margin=0.05, integerize=:auto, method::Symbol=:reps)
    padding < 0 && error("window_box: padding must be nonnegative")
    margin < 0 && error("window_box: margin must be nonnegative")

    # window_box is a *pure inference* routine; do not respect user opts here.
    empty_opts = InvariantOptions()

    # bounding box from representatives, no padding
    reps_box = encoding_box(pi, empty_opts; margin=0.0)

    coords_box = nothing
    try
        ax = axes_from_encoding(pi)
        coords_box = encoding_box(ax, empty_opts; margin=0.0)
    catch e
        if !(e isa MethodError)
            rethrow()
        end
    end

    ell, u = if method === :reps
        reps_box
    elseif method === :coords
        coords_box === nothing && error("window_box(method=:coords) requires axes_from_encoding(pi) for $(typeof(pi)).")
        coords_box
    elseif method === :mix
        if coords_box === nothing
            reps_box
        else
            (min.(reps_box[1], coords_box[1]), max.(reps_box[2], coords_box[2]))
        end
    else
        error("window_box: unknown method=$method (expected :reps, :coords, or :mix)")
    end

    if method === :reps && coords_box !== nothing
        ell2, u2 = coords_box
        @inbounds for i in 1:length(ell)
            if abs(u[i] - ell[i]) < 1e-12
                ell[i] = ell2[i]
                u[i] = u2[i]
            end
        end
    end

    is_lattice = _is_lattice_encoding(pi)
    @inbounds for i in 1:length(ell)
        if abs(u[i] - ell[i]) < 1e-12
            # Avoid degenerate axes in inferred windows (e.g., free lattice directions).
            if is_lattice
                ell[i] -= 1
                u[i] += 1
            else
                ell[i] -= 1.0
                u[i] += 1.0
            end
        end
    end

    @inbounds for i in 1:length(ell)
        w = u[i] - ell[i]
        m = w * float(margin)
        ell[i] -= m + float(padding)
        u[i] += m + float(padding)
    end

    do_int = if integerize === :always
        true
    elseif integerize === :never
        false
    elseif integerize === :auto
        is_lattice
    else
        error("window_box: integerize must be :auto, :always, or :never")
    end

    return do_int ? (floor.(Int, ell), ceil.(Int, u)) : (ell, u)
end

# Decide whether to default to integer directions/offsets (lattice-style slicing).
_is_lattice_encoding(pi::PLikeEncodingMap) = begin
    pts = representatives(pi)
    isempty(pts) && return false
    p1 = first(pts)
    return eltype(p1) <: Integer
end



# Internal helper: choose a finite set of points that represent the parameter space.
function _encoding_points(pi)
    if hasproperty(pi, :reps)
        reps = getproperty(pi, :reps)
        if reps !== nothing && !isempty(reps)
            return reps
        end
    end
    if hasproperty(pi, :witnesses)
        w = getproperty(pi, :witnesses)
        if w !== nothing && !isempty(w)
            return w
        end
    end
    if hasproperty(pi, :coords)
        c = getproperty(pi, :coords)
        # Heuristic: treat `coords` as a point cloud only when it is "tall".
        # This avoids misclassifying PL grid coordinate arrays as a point cloud.
        if c !== nothing && !isempty(c) && c[1] isa AbstractVector{<:Real} &&
           length(c) > length(c[1])
            return c
        end
    end
    error("cannot infer defaults: encoding map has no non-empty `reps`, `witnesses`, or point-like `coords` field")
end

# Subsample a vector deterministically by taking approximately evenly spaced indices.
function _subsample_evenly(v::AbstractVector, n::Integer)
    n <= 0 && error("expected n >= 1")
    m = length(v)
    if m <= n
        return collect(v)
    end
    idxs = round.(Int, range(1, m, length=n))
    idxs = unique(idxs)
    return [v[i] for i in idxs]
end

"""
    default_directions(pi; n_dirs=16, max_den=8, include_axes=false, normalize=:L1)

Return a deterministic list of slice directions suitable for `pi`.

- For 2-parameter encodings, directions are generated from primitive integer
  pairs `(a,b)` with `1 <= a,b <= max_den` (and optionally axis directions).
- For higher dimension, a small collection of primitive integer vectors with
  entries in `1:max_den` is used.

If the encoding points of `pi` are integer-valued (lattice encodings), the
returned directions are integer tuples. Otherwise, directions are returned as
`Float64` tuples, normalized according to `normalize` (one of `:L1`, `:Linf`,
or `:none`).

This function is intended to provide sensible defaults; it is not a substitute
for problem-specific direction sampling.
"""
function default_directions(pi; n_dirs::Integer=16, max_den::Integer=8,
                           include_axes::Bool=false, normalize::Symbol=:L1)
    d = dimension(pi)
    integer_dirs = _is_lattice_encoding(pi)
    return default_directions(d; n_dirs=n_dirs, max_den=max_den,
                              include_axes=include_axes, normalize=normalize,
                              integer=integer_dirs)
end

"""
    default_directions(d::Integer; n_dirs=16, max_den=8, include_axes=false,
                       normalize=:L1, integer=false)

Low-level direction generator used by `default_directions(pi)`.
"""
function default_directions(d::Integer; n_dirs::Integer=16, max_den::Integer=8,
                           include_axes::Bool=false, normalize::Symbol=:L1,
                           integer::Bool=false)
    d <= 0 && error("default_directions: dimension must be positive")
    max_den <= 0 && error("default_directions: max_den must be positive")
    n_dirs <= 0 && error("default_directions: n_dirs must be positive")

    if d == 1
        return integer ? [(1,)] : [(1.0,)]
    elseif d == 2
        dirs_all = sample_directions_2d(max_den=max_den, include_axes=include_axes,
                                        normalize=(integer ? :none : normalize))
        return _subsample_evenly(dirs_all, n_dirs)
    end

    lo = include_axes ? 0 : 1
    dirs_int = Vector{NTuple{d,Int}}()
    ranges = ntuple(_ -> lo:max_den, d)
    for tup in Iterators.product(ranges...)
        all(x -> x == 0, tup) && continue
        g = 0
        for x in tup
            g = gcd(g, x)
        end
        g == 1 || continue
        push!(dirs_int, ntuple(i -> Int(tup[i]), d))
    end

    sort!(dirs_int; by=v -> begin
        s = sum(v)
        return Tuple(float(x) / float(s) for x in v)
    end)

    dirs_int = _subsample_evenly(dirs_int, n_dirs)

    if integer || normalize == :none
        return dirs_int
    elseif normalize == :L1
        return [begin
            s = float(sum(v))
            ntuple(i -> float(v[i]) / s, d)
        end for v in dirs_int]
    elseif normalize == :Linf
        return [begin
            s = float(maximum(v))
            ntuple(i -> float(v[i]) / s, d)
        end for v in dirs_int]
    else
        error("default_directions: normalize must be :L1, :Linf, or :none")
    end
end

"""
    default_offsets(pi::PLikeEncodingMap, opts::InvariantOptions;
        n_offsets=9,
        margin=0.05)

Pick a default set of "offset" points used when building many slices.

This is an opts-primary API:
- The working box is derived from `opts.box`:
  * `opts.box === nothing`  -> infer from representatives via `encoding_box(pi, opts)`
  * `opts.box === :auto`    -> start from `window_box(pi)` (handled by encoding_box)
  * concrete `(lo, hi)`     -> use it verbatim
- `margin` expands the inferred box (ignored for a concrete box override).

Returns a vector of offset points (each a tuple of `Float64`).
"""
function default_offsets(pi::PLikeEncodingMap, opts::InvariantOptions;
    n_offsets::Int = 9,
    margin::Real = 0.05)

    lo, hi = encoding_box(pi, opts; margin=margin)
    n = length(lo)
    offs = Vector{Tuple}(undef, n_offsets)
    ts = range(0.0, 1.0, length=n_offsets)
    for (i, t) in enumerate(ts)
        offs[i] = ntuple(k -> float(lo[k] + t * (hi[k] - lo[k])), n)
    end
    return offs
end

"""
    default_offsets(pi::PLikeEncodingMap, dir::AbstractVector{<:Real}, opts::InvariantOptions;
        n_offsets=9,
        margin=0.05)

Direction-aware default offsets: choose `n_offsets` points along a line orthogonal
to `dir`, spanning the projection of the working box along that normal.

The working box is derived from `opts.box` (see the 1-argument method).
"""
function _default_offsets_dir(pi::PLikeEncodingMap, dir::NTuple{N,<:Real}, opts::InvariantOptions;
    n_offsets::Int = 9,
    margin::Real = 0.05) where {N}

    lo, hi = encoding_box(pi, opts; margin=margin)

    n = ntuple(i -> float(dir[i]), N)
    nrm = _l2_norm(n)
    nrm == 0 && error("default_offsets: dir must be nonzero")
    n = ntuple(i -> n[i] / nrm, N)

    # Enumerate all corners of the axis-aligned box.
    corners = Vector{Tuple}()
    for bits in Iterators.product((0, 1) for _ in 1:length(lo))
        c = ntuple(i -> float(bits[i] == 0 ? lo[i] : hi[i]), N)
        push!(corners, c)
    end

    # Project corners onto the normal direction to get span.
    projs = [_dot(n, c) for c in corners]
    smin = minimum(projs)
    smax = maximum(projs)

    # Center point of the box, and its projection.
    ctr = ntuple(i -> float((lo[i] + hi[i]) / 2), N)
    cproj = _dot(n, ctr)

    # Offsets are "ctr shifted along n" so that dot(offset, n) spans [smin, smax].
    svals = range(smin, smax, length=n_offsets)
    return [ntuple(i -> ctr[i] + (s - cproj) * n[i], N) for s in svals]
end

function default_offsets(pi::PLikeEncodingMap, dir::AbstractVector{<:Real}, opts::InvariantOptions;
    n_offsets::Int = 9,
    margin::Real = 0.05)
    return _default_offsets_dir(pi, Tuple(dir), opts; n_offsets=n_offsets, margin=margin)
end

function default_offsets(pi::PLikeEncodingMap, dir::NTuple{N,<:Real}, opts::InvariantOptions;
    n_offsets::Int = 9,
    margin::Real = 0.05) where {N}
    return _default_offsets_dir(pi, dir, opts; n_offsets=n_offsets, margin=margin)
end


"""
    matching_distance_approx(M, N, slices; default_weight=1.0) -> Float64
    matching_distance_approx(M, N, pi; directions, offsets, weight=:lesnick_l1, ...) -> Float64

Approximate the (2D/ND) matching distance by taking a maximum of bottleneck
distances over a finite family of 1D slices.

Two ways to call this:

1) Provide slices explicitly:
    slices = [
        [q1,q2,...,qm],                                # a chain in M.Q
        (chain=[...], values=[...], weight=1.0),       # richer spec (NamedTuple)
        (chain, values, weight)                        # tuple form
    ]

2) Provide an encoding map and sample geometric slices:
    matching_distance_approx(M, N, pi;
        directions=[v1,v2,...],
        offsets=[x01,x02,...],
        tmin=..., tmax=..., nsteps=..., strict=true,
        weight=:lesnick_l1
    )

For each slice we:
- build a chain in the finite encoding (via `slice_chain` if using (2)),
- compute slice barcodes for M and N,
- compute bottleneck distance between those barcodes,
- multiply by a slice weight,
- take the maximum.

Notes:
- This is an approximation: increasing the number of sampled slices improves it.
- The default `weight=:lesnick_l1` matches the common Lesnick-Wright style scaling
  in 2D after L1 normalization; set `weight=:none` to disable weighting.
"""
function matching_distance_approx(
    M::PModule{K},
    N::PModule{K},
    slices::AbstractVector;
    default_weight=1.0
)::Float64 where {K}
    specs = collect_slices(slices; default_weight=default_weight)
    return _matching_distance_approx_specs(M, N, specs)
end

function matching_distance_approx(
    M::PModule{K},
    N::PModule{K},
    chain::AbstractVector{Int};
    default_weight=1.0
) where {K}
    return matching_distance_approx(M, N, [chain]; default_weight=default_weight)
end

"""
    matching_distance_approx(M, N, pi, opts::InvariantOptions; ...)

Approximate the matching distance by sampling slices and taking the max of
(per-slice) 1D matching distances.

Opts usage:
- `opts.strict` controls locate strictness (defaults to true).
- `opts.box` controls slice clipping.
- `opts.threads` controls parallel barcode computation.
"""
function matching_distance_approx(
    M::PModule{K},
    N::PModule{K},
    pi::PLikeEncodingMap,
    opts::InvariantOptions;
    directions = :auto,
    offsets = :auto,
    n_dirs::Integer = 100,
    n_offsets::Integer = 50,
    max_den::Integer = 8,
    include_axes::Bool = false,
    normalize_dirs::Symbol = :L1,
    weight::Union{Symbol,Function,Real} = :lesnick_l1,
    offset_weights = nothing,
    box2 = nothing,
    cache = nothing,
    slice_kwargs...
)::Float64 where {K}

    strict0 = opts.strict === nothing ? true : opts.strict
    threads0 = opts.threads === nothing ? (Threads.nthreads() > 1) : opts.threads

    return _slice_based_barcode_distance(
        M, N, pi;
        dirs = directions,
        offs = offsets,
        ndirs = n_dirs,
        noff = n_offsets,
        max_den = max_den,
        include_axes = include_axes,
        normalize_dirs = normalize_dirs,
        dist_fn = matching_distance,
        dist_kwargs = NamedTuple(),
        weight = weight,
        offset_weights = offset_weights,
        threads = threads0,
        cache = cache,
        strict = strict0,
        box = opts.box,
        box2 = box2,
        slice_kwargs...
    )
end

"""
    matching_wasserstein_distance_approx(M, N, pi, opts::InvariantOptions; ...)

Approximate the matching distance by taking the max of per-slice 1D Wasserstein
distances.

Opts usage:
- `opts.strict` controls locate strictness (defaults to true).
- `opts.box` controls slice clipping.
- `opts.threads` controls parallel barcode computation.

This is not the classical matching distance (which uses bottleneck), but it is exactly
the "matching_wasserstein_distance_approx" slice-based family member requested in your
screenshot: same structure, Wasserstein per slice, supremum reduction.
"""
function matching_wasserstein_distance_approx(
    M::PModule{K},
    N::PModule{K},
    pi,
    # NOTE: opts is positional (refactor pattern)
    opts::InvariantOptions;
    directions = :auto,
    offsets = :auto,
    n_dirs::Integer = 100,
    n_offsets::Integer = 50,
    max_den::Integer = 8,
    include_axes::Bool = false,
    normalize_dirs::Symbol = :L1,
    weight::Union{Symbol,Function,Real} = :lesnick_l1,
    offset_weights = nothing,
    box2 = nothing,
    p = 2,
    q = 1,
    cache = nothing,
    slice_kwargs...
)::Float64 where {K}

    strict0 = opts.strict === nothing ? true : opts.strict
    threads0 = opts.threads === nothing ? (Threads.nthreads() > 1) : opts.threads

    return _slice_based_barcode_distance(M, N, pi;
        dirs = directions,
        offs = offsets,
        ndirs = n_dirs,
        noff = n_offsets,
        max_den = max_den,
        include_axes = include_axes,
        normalize_dirs = normalize_dirs,
        dist_fn = wasserstein_distance,
        dist_kwargs = (p=p, q=q),
        weight_mode = :scale,
        weight = weight,
        offset_weights = offset_weights,
        threads = threads0,
        cache = cache,
        strict = strict0,
        box = opts.box,
        box2 = box2,
        slice_kwargs...)
end


"""
    matching_wasserstein_distance_approx(M, N, pi, slices; p=2, q=1)

Slice-explicit variant (mirrors `matching_distance_approx(M,N,pi,slices)`):
each element of `slices` can be either:

- a chain `Vector{Int}` (interpreted as (chain, weights=nothing, t=0, w=1))
- or a tuple `(chain, weights, t, w)` where `w` scales the per-slice distance

The output is:
    max_slice ( w * wasserstein_distance(barcode_slice_M, barcode_slice_N; p, q) )
"""
function matching_wasserstein_distance_approx(
    M::PModule{K},
    N::PModule{K},
    pi::PLikeEncodingMap,
    slices::Vector;
    p=2,
    q=1
)::Float64 where {K}
    d = 0.0
    for slice in slices
        chain = slice isa Vector ? slice : slice[1]
        weights = slice isa Vector ? nothing : slice[2]
        t = slice isa Vector ? 0 : slice[3]
        w = slice isa Vector ? 1 : slice[4]

        bcM = slice_barcode(M, pi, chain, weights, t)
        bcN = slice_barcode(N, pi, chain, weights, t)
        ds = w * wasserstein_distance(bcM, bcN; p=p, q=q)
        if ds > d
            d = ds
        end
    end
    return d
end



struct PersistenceLandscape1D
    tgrid::Vector{Float64}
    values::Matrix{Float64}
end

Base.size(pl::PersistenceLandscape1D) = size(pl.values)
Base.getindex(pl::PersistenceLandscape1D, k::Int, i::Int) = pl.values[k, i]
Base.getindex(pl::PersistenceLandscape1D, k::Int) = view(pl.values, k, :)

"""
    landscape_value(pl, k, t; extrapolate=false) -> Float64

Evaluate the k-th landscape layer at parameter value `t` by piecewise-linear
interpolation on the stored grid `pl.tgrid`.

If `t` lies outside the grid and `extrapolate=false` (default), this returns 0.0.
If `extrapolate=true`, it clamps to the nearest endpoint value.
"""
function landscape_value(pl::PersistenceLandscape1D, k::Int, t::Real; extrapolate::Bool=false)::Float64
    1 <= k <= size(pl.values, 1) || error("landscape_value: k out of range")
    tt = float(t)
    tg = pl.tgrid
    nt = length(tg)

    if nt == 0
        return 0.0
    elseif nt == 1
        return extrapolate ? pl.values[k, 1] : 0.0
    end

    # Outside the sampled window.
    if tt <= tg[1]
        return extrapolate ? pl.values[k, 1] : 0.0
    elseif tt >= tg[end]
        return extrapolate ? pl.values[k, end] : 0.0
    end

    # Find i with tg[i] <= tt < tg[i+1].
    i = searchsortedlast(tg, tt)
    i == nt && (i = nt - 1)

    t0 = tg[i]
    t1 = tg[i + 1]
    v0 = pl.values[k, i]
    v1 = pl.values[k, i + 1]

    # Linear interpolation (t0 < t1 is guaranteed by construction).
    alpha = (tt - t0) / (t1 - t0)
    return (1.0 - alpha) * v0 + alpha * v1
end

# Internal: clean and validate a grid.
function _clean_tgrid(tgrid)::Vector{Float64}
    tg = Float64[float(t) for t in collect(tgrid)]
    sort!(tg)
    unique!(tg)

    # We need at least two points for integration; fall back to a trivial grid.
    if length(tg) < 2
        tg = Float64[0.0, 1.0]
    end
    return tg
end

# Internal: the tent function associated to an interval (b,d).
# This is max(0, min(t-b, d-t)), i.e. a triangle with peak at (b+d)/2.
@inline function _tent_value(b::Float64, d::Float64, t::Float64)::Float64
    v = min(t - b, d - t)
    return v > 0 ? v : 0.0
end

"""
    persistence_landscape(bar; kmax=5, tgrid=nothing, nsteps=401) -> PersistenceLandscape1D

Compute the first `kmax` layers of the 1D persistence landscape associated to a
barcode `bar`.

Input formats for `bar`
-----------------------
- `Dict((birth, death) => multiplicity)`
- `Vector{Tuple{birth, death}}` (multiplicity 1)

Evaluation
----------
If `tgrid` is provided, the landscape is sampled on that grid (after sorting and
de-duplication). If `tgrid=nothing`, a default uniform grid with `nsteps` points
is chosen from `min(birth)` to `max(death)`.

The output stores a matrix `values` of size `(kmax, length(tgrid))` where
`values[k,i]` equals the k-th landscape layer evaluated at `tgrid[i]`.
"""
function persistence_landscape(
    bar;
    kmax::Int=5,
    tgrid=nothing,
    nsteps::Int=401
)::PersistenceLandscape1D
    kmax >= 1 || error("persistence_landscape: kmax must be >= 1")

    pts = _barcode_points(bar)  # Vector{Tuple{Float64,Float64}} (expanded multiplicities)

    # Choose a default grid if needed.
    if tgrid === nothing
        if isempty(pts)
            tg = Float64[0.0, 1.0]
        else
            bmin = minimum(p[1] for p in pts)
            dmax = maximum(p[2] for p in pts)
            (bmin < dmax) || (dmax = bmin + 1.0)
            tg = collect(range(bmin, dmax; length=nsteps))
        end
    else
        tg = collect(tgrid)
    end

    tg = _clean_tgrid(tg)
    nt = length(tg)
    vals = zeros(Float64, kmax, nt)
    _persistence_landscape_values!(vals, pts, tg; tent_scratch=Float64[])
    return PersistenceLandscape1D(tg, vals)
end

function _persistence_landscape_values!(
    dest::AbstractMatrix{Float64},
    pts::AbstractVector{<:Tuple{Float64,Float64}},
    tg::AbstractVector{<:Real};
    tent_scratch::Vector{Float64}=Float64[],
)
    kmax = size(dest, 1)
    nt = size(dest, 2)
    length(tg) == nt || error("_persistence_landscape_values!: destination width must match tgrid length")
    fill!(dest, 0.0)
    isempty(pts) && return dest

    empty!(tent_scratch)
    sizehint!(tent_scratch, length(pts))

    @inbounds for (b, d) in pts
        b < d || error("persistence_landscape: invalid interval with birth >= death: ($b, $d)")
    end

    @inbounds for j in 1:nt
        t = float(tg[j])
        empty!(tent_scratch)
        for idx in eachindex(pts)
            b, d = pts[idx]
            v = _tent_value(b, d, t)
            v > 0 && push!(tent_scratch, v)
        end
        if !isempty(tent_scratch)
            sort!(tent_scratch; rev=true)
            m = min(kmax, length(tent_scratch))
            for k in 1:m
                dest[k, j] = tent_scratch[k]
            end
        end
    end
    return dest
end

function _persistence_landscape_values!(
    dest::AbstractMatrix{Float64},
    bar,
    tg::AbstractVector{<:Real};
    points_scratch::Vector{Tuple{Float64,Float64}}=Tuple{Float64,Float64}[],
    tent_scratch::Vector{Float64}=Float64[],
)
    _barcode_points!(points_scratch, bar)
    return _persistence_landscape_values!(dest, points_scratch, tg; tent_scratch=tent_scratch)
end


"""
    MPLandscape

A finite-sampled approximation of a multiparameter persistence landscape.

This stores a family of 1D persistence landscapes indexed by a finite list of
slice directions and offsets.

Fields
------
- `kmax`: number of landscape layers stored per slice.
- `tgrid`: common evaluation grid for all slices.
- `values`: an array of size `(ndirs, noffsets, kmax, length(tgrid))`.
  `values[i,j,k,l]` is the k-th landscape layer at `tgrid[l]` for slice
  `(direction[i], offset[j])`.
- `weights`: an `(ndirs, noffsets)` matrix of nonnegative slice weights.
- `directions`, `offsets`: metadata as passed/used in construction.

See also `mp_landscape_distance` and `mp_landscape_kernel`.
"""

# ----- Slice vectorizations and sliced kernels -------------------------------

"""
    PersistenceImage1D

A persistence image for a 1D barcode (equivalently, a 1D persistence diagram),
represented on a rectangular grid.

We use the common (birth, persistence) coordinate system by default, where each
interval (b, d) is mapped to the point (b, d-b).

The `values` matrix is indexed as `values[iy, ix]` with:
- `ix` indexing `xgrid` (birth),
- `iy` indexing `ygrid` (persistence or death, depending on `coords`).
"""
struct PersistenceImage1D
    xgrid::Vector{Float64}
    ygrid::Vector{Float64}
    values::Matrix{Float64}
end

# Build a Float64 grid from a (lo, hi) range and a requested number of points.
function _grid_from_range(rng, n::Int)::Vector{Float64}
    n >= 1 || error("_grid_from_range: n must be >= 1")
    lo, hi = rng
    lo = float(lo)
    hi = float(hi)
    if !(isfinite(lo) && isfinite(hi))
        error("_grid_from_range: range endpoints must be finite")
    end
    if hi < lo
        error("_grid_from_range: need lo <= hi")
    end
    if lo == hi
        # Expand a degenerate range so range(...) makes sense.
        lo -= 0.5
        hi += 0.5
    end
    return collect(range(lo, hi; length=n))
end

# Extract global bounds from a single barcode:
# returns (bmin, dmax, pmax, n_intervals).
function _barcode_bounds_birth_death(bar)
    bmin = Inf
    dmax = -Inf
    pmax = 0.0
    n = 0
    for (bd, mult) in bar
        mult <= 0 && continue
        b0, d0 = bd
        b = float(b0)
        d = float(d0)
        d > b || continue
        bmin = min(bmin, b)
        dmax = max(dmax, d)
        pmax = max(pmax, d - b)
        n += mult
    end
    if bmin == Inf
        # Empty barcode: choose a harmless default.
        bmin = 0.0
        dmax = 1.0
        pmax = 1.0
        n = 0
    end
    return bmin, dmax, pmax, n
end

# Weight assigned to a single interval.
# This is used for persistence images, silhouettes, and entropy.
function _interval_weight(weighting, b::Real, d::Real; p::Real=1)::Float64
    if weighting isa Function
        return float(weighting(b, d))
    end
    if weighting == :none
        return 1.0
    elseif weighting == :persistence
        pers = float(d) - float(b)
        pers < 0 && return 0.0
        return pers^float(p)
    else
        error("_interval_weight: unknown weighting $(weighting); supported: :none, :persistence, or a function")
    end
end

# Normalize a Float64 matrix in-place.
function _normalize_matrix!(A::AbstractMatrix{Float64}, mode::Symbol)
    mode == :none && return A
    if mode == :l1
        s = sum(abs.(A))
        s > 0 && (A ./= s)
    elseif mode == :l2
        s = norm(A)
        s > 0 && (A ./= s)
    elseif mode == :max
        m = maximum(A)
        m > 0 && (A ./= m)
    else
        error("_normalize_matrix!: normalize must be :none, :l1, :l2, or :max")
    end
    return A
end

"""
    persistence_image(bar; ... ) -> PersistenceImage1D

Compute a persistence image for a 1D barcode `bar` (a dictionary mapping
interval endpoints `(b, d)` to multiplicity).

This is a standard, ML-friendly vectorization of a persistence diagram: each
interval contributes a Gaussian "blob" centered at (birth, persistence) by
default.

Keywords:
- `xgrid`, `ygrid`: explicit grids (centers) for birth and persistence/death.
- `birth_range`, `pers_range`: ranges used if grids are not provided.
- `nbirth`, `npers`: grid sizes used if grids are not provided.
- `sigma`: Gaussian bandwidth (> 0).
- `coords`: `:birth_persistence` (default) or `:birth_death`.
- `weighting`: `:persistence` (default), `:none`, or a function `(b,d)->w`.
- `p`: exponent used when `weighting=:persistence`.
- `normalize`: `:none` (default), `:l1`, `:l2`, or `:max`.

Notes:
- The discretization is by evaluation at grid centers (not exact pixel integrals).
- Empty barcodes yield an all-zero image.
"""
function persistence_image(bar;
                           xgrid=0:0.1:1,
                           ygrid=0:0.1:1,
                           sigma=0.1,
                           coords=:birth_persistence,
                           weighting=:persistence,
                           p=1,
                           normalize=:none,
                           differentiable::Bool=false,
                           threads::Bool = (Threads.nthreads() > 1))

    xg = collect(xgrid)
    yg = collect(ygrid)
    inv2sig2 = 1.0 / (2*sigma^2)

    if !differentiable
        # Fast in-place version (existing behavior)
        img = zeros(Float64, length(yg), length(xg))
        if threads && Threads.nthreads() > 1
            Threads.@threads for ix in 1:length(xg)
                xgix = xg[ix]
                for iy in 1:length(yg)
                    ygiy = yg[iy]
                    acc = 0.0
                    for (bd, mult) in bar
                        b = bd[1]
                        d = bd[2]

                        x, y = coords == :birth_persistence ? (b, d-b) :
                               coords == :birth_death ? (b, d) :
                               coords == :midlife_persistence ? (0.5*(b+d), d-b) :
                               error("unknown coords=$coords")

                        w = mult * _interval_weight(weighting, b, d; p=p)
                        dx2 = (x - xgix)^2
                        dy2 = (y - ygiy)^2
                        acc += w * exp(-(dx2+dy2)*inv2sig2)
                    end
                    img[iy, ix] = acc
                end
            end
        else
            for (bd, mult) in bar
                b = bd[1]
                d = bd[2]

                x, y = coords == :birth_persistence ? (b, d-b) :
                       coords == :birth_death ? (b, d) :
                       coords == :midlife_persistence ? (0.5*(b+d), d-b) :
                       error("unknown coords=$coords")

                w = mult * _interval_weight(weighting, b, d; p=p)

                for ix in eachindex(xg)
                    dx2 = (x - xg[ix])^2
                    for iy in eachindex(yg)
                        dy2 = (y - yg[iy])^2
                        img[iy,ix] += w * exp(-(dx2+dy2)*inv2sig2)
                    end
                end
            end
        end

        _normalize_matrix!(img, normalize)
        return PersistenceImage1D(xg, yg, img)
    end

    # Pure/non-mutating version for AD friendliness.
    terms = [begin
                b = bd[1]
                d = bd[2]
                x, y = coords == :birth_persistence ? (b, d-b) :
                       coords == :birth_death ? (b, d) :
                       coords == :midlife_persistence ? (0.5*(b+d), d-b) :
                       error("unknown coords=$coords")
                w = mult * _interval_weight(weighting, b, d; p=p)
                (x, y, w)
            end for (bd, mult) in bar]

    img = [sum(t[3] * exp(-(((t[1]-xg[ix])^2 + (t[2]-yg[iy])^2))*inv2sig2) for t in terms)
           for iy in eachindex(yg), ix in eachindex(xg)]

    # Non-mutating normalization
    img = normalize == :none ? img :
          normalize == :l1  ? (sum(abs, img) > 0 ? img ./ sum(abs, img) : img) :
          normalize == :l2  ? (sqrt(sum(abs2, img)) > 0 ? img ./ sqrt(sum(abs2, img)) : img) :
          normalize == :max ? (maximum(abs, img) > 0 ? img ./ maximum(abs, img) : img) :
          error("unknown normalize=$normalize")

    return PersistenceImage1D(xg, yg, img)
end

"""
    feature_map(x; kind=:persistence_image, flatten=true, differentiable=false, kwargs...)

Compute a fixed-size numeric representation suitable for ML.

Currently supported:
- `x` is a barcode dictionary and `kind=:persistence_image`
- `x` is a `PersistenceImage1D`

If `flatten=true`, returns a vector in column-major order.
If `differentiable=true` and kind supports it, avoids mutation (Zygote-friendly).
"""
function feature_map(x;
                     kind::Symbol = :persistence_image,
                     flatten::Bool = true,
                     differentiable::Bool = false,
                     kwargs...)
    mat = if x isa PersistenceImage1D
        x.values
    elseif kind == :persistence_image
        persistence_image(x; differentiable=differentiable, kwargs...).values
    else
        error("unsupported feature_map kind=$kind for input $(typeof(x))")
    end
    return flatten ? vec(mat) : mat
end

"""
    feature_vector(x; kwargs...)

Alias for `feature_map(x; flatten=true, ...)`.
"""
feature_vector(x; kwargs...) = feature_map(x; flatten=true, kwargs...)


"""
    persistence_silhouette(bar; tgrid, weighting=:persistence, p=1, normalize=true)

Compute a persistence silhouette for a 1D barcode.

A silhouette is the weighted average (by default, persistence weights) of the
tent functions associated to each interval. If `normalize=true`, the output is
divided by the total weight, so the result is scale-stable.

This is a standard 1D functional summary, distinct from landscapes:
- landscapes take successive maxima of tents (layered max),
- silhouettes average tents (weighted sum / average).
"""
function persistence_silhouette(
    bar;
    tgrid,
    weighting=:persistence,
    p::Real=1,
    normalize::Bool=true
)::Vector{Float64}
    tg = _clean_tgrid(tgrid)
    out = zeros(Float64, length(tg))
    denom = 0.0

    for (bd, mult) in bar
        mult <= 0 && continue
        b0, d0 = bd
        b = float(b0)
        d = float(d0)
        d > b || continue

        w = float(mult) * _interval_weight(weighting, b, d; p=p)
        w == 0.0 && continue

        denom += w
        for it in eachindex(tg)
            out[it] += w * _tent_value(b, d, tg[it])
        end
    end

    if normalize && denom > 0.0
        out ./= denom
    end
    return out
end

"""
    barcode_entropy(bar; normalize=true, base=exp(1), weighting=:persistence, p=1) -> Float64

Compute "persistent entropy" for a barcode.

We interpret the barcode as a multiset of intervals. Each interval gets a
nonnegative weight `w_i` (by default, persistence^p). We normalize to a
probability distribution and return Shannon entropy.

- `normalize=true` divides by `log(n_intervals)` (common in the literature),
  yielding values in [0, 1] when there are at least 2 intervals.
- `base` controls the logarithm base (default: natural).
"""
function barcode_entropy(
    bar;
    normalize::Bool=true,
    base::Real=exp(1),
    weighting=:persistence,
    p::Real=1
)::Float64
    base = float(base)
    base > 0 || error("barcode_entropy: base must be > 0")

    tot = 0.0
    n = 0
    ws = Vector{Tuple{Float64, Int}}()

    for (bd, mult) in bar
        mult <= 0 && continue
        b0, d0 = bd
        b = float(b0)
        d = float(d0)
        d > b || continue

        w = _interval_weight(weighting, b, d; p=p)
        w = max(w, 0.0)
        w == 0.0 && continue

        push!(ws, (w, mult))
        tot += w * float(mult)
        n += mult
    end

    (tot > 0.0 && n > 0) || return 0.0

    invlogbase = 1.0 / log(base)
    H = 0.0
    for (w, mult) in ws
        p_i = w / tot
        H -= float(mult) * p_i * (log(p_i) * invlogbase)
    end

    if normalize && n > 1
        H /= (log(float(n)) * invlogbase)
    end
    return H
end

"""
    barcode_summary(bar; normalize_entropy=true) -> NamedTuple

Summary statistics for a 1D barcode. Intended for quick diagnostics and
lightweight ML features.

Returns a NamedTuple with fields:
- `n_intervals`
- `total_persistence`
- `max_persistence`
- `mean_persistence`
- `l2_persistence`
- `entropy`
"""
function barcode_summary(bar; normalize_entropy::Bool=true)
    n = 0
    total = 0.0
    maxp = 0.0
    sumsq = 0.0

    for (bd, mult) in bar
        mult <= 0 && continue
        b0, d0 = bd
        b = float(b0)
        d = float(d0)
        d > b || continue

        pers = d - b
        n += mult
        total += float(mult) * pers
        maxp = max(maxp, pers)
        sumsq += float(mult) * pers * pers
    end

    meanp = (n > 0) ? (total / float(n)) : 0.0
    l2p = sqrt(sumsq)
    ent = barcode_entropy(bar; normalize=normalize_entropy)

    return (n_intervals=n,
            total_persistence=total,
            max_persistence=maxp,
            mean_persistence=meanp,
            l2_persistence=l2p,
            entropy=ent)
end

const _DEFAULT_BARCODE_SUMMARY_FIELDS =
    (:n_intervals, :total_persistence, :max_persistence, :mean_persistence, :l2_persistence, :entropy)

function _barcode_summary_vector(bar; fields=_DEFAULT_BARCODE_SUMMARY_FIELDS, normalize_entropy::Bool=true)
    nt = barcode_summary(bar; normalize_entropy=normalize_entropy)
    return Float64[float(getproperty(nt, f)) for f in fields]
end

# Flatten a PersistenceLandscape1D to a single feature vector.
# Convention: k-major ordering, i.e.
#   [L_1(t1), ..., L_1(tN), L_2(t1), ..., L_2(tN), ...]
function _landscape_feature_vector(pl::PersistenceLandscape1D)::Vector{Float64}
    kmax, nt = size(pl.values)
    out = Vector{Float64}(undef, kmax * nt)
    idx = 1
    for k in 1:kmax
        for it in 1:nt
            out[idx] = pl.values[k, it]
            idx += 1
        end
    end
    return out
end

# Flatten a PersistenceImage1D to a single feature vector.
# Convention: y-major ordering, with x varying fastest:
#   [row1(x1..xN), row2(x1..xN), ...]
function _image_feature_vector(PI::PersistenceImage1D)::Vector{Float64}
    ny, nx = size(PI.values)
    out = Vector{Float64}(undef, nx * ny)
    idx = 1
    for iy in 1:ny
        for ix in 1:nx
            out[idx] = PI.values[iy, ix]
            idx += 1
        end
    end
    return out
end

# Typed explicit slice specs for stable statistics pipelines.
#
# `values === nothing` means endpoint indices are used (index barcode mode).
# For numeric endpoint values, use `Vector{Int}` or `Vector{Float64}`.
@inline _to_int_vec(v::Vector{Int}) = v
@inline _to_int_vec(v::AbstractVector{<:Integer}) = Int.(v)

# Parse slice specs at API boundaries.
# Supported inputs:
# - chain::Vector{Int}
# - NamedTuple/struct with fields (chain, [values], [weight])
# - Tuple (chain, [values], [weight])
function _parse_slice_spec(spec; default_weight::Real = 1.0, weight_fn = nothing)
    chain = nothing
    values = nothing
    w = default_weight

    if spec isa AbstractVector{<:Integer}
        chain = spec
    elseif spec isa SliceSpec
        chain = spec.chain
        values = spec.values
        w = spec.weight
    elseif spec isa Tuple
        length(spec) >= 1 || error("_parse_slice_spec: tuple slice spec must have at least a chain")
        chain = spec[1]
        length(spec) >= 2 && (values = spec[2])
        length(spec) >= 3 && (w = spec[3])
    else
        hasproperty(spec, :chain) || error("_parse_slice_spec: unrecognized slice spec (missing :chain field)")
        chain = getproperty(spec, :chain)
        hasproperty(spec, :values) && (values = getproperty(spec, :values))
        if hasproperty(spec, :weight)
            w = getproperty(spec, :weight)
        elseif weight_fn !== nothing && hasproperty(spec, :dir)
            w = weight_fn(spec)
        end
    end

    chain isa AbstractVector{<:Integer} || error("_parse_slice_spec: slice chain must be an integer vector")
    chain_vec = chain isa Vector{Int} ? chain : collect(Int, chain)
    return (chain = chain_vec, values = values, weight = float(w))
end

const _SliceValues = Union{Nothing,Vector{Int},Vector{Float64}}

@inline function _normalize_slice_values(v)::_SliceValues
    if v === nothing
        return nothing
    elseif _values_are_int_vector(v)
        return (v isa Vector{Int}) ? copy(v) : collect(Int, v)
    elseif v isa AbstractVector{<:Real}
        return (v isa Vector{Float64}) ? copy(v) : Float64[float(x) for x in v]
    elseif v isa AbstractVector && all(x -> x isa Real, v)
        return Float64[float(x) for x in v]
    end
    throw(ArgumentError("collect_slices: unsupported values payload type $(typeof(v))"))
end

@inline _slice_mode_name(::Type{Nothing}) = :none
@inline _slice_mode_name(::Type{<:AbstractVector{<:Integer}}) = :int
@inline _slice_mode_name(::Type{<:AbstractVector{<:AbstractFloat}}) = :float
@inline _slice_mode_name(::Type{<:AbstractVector{<:Real}}) = :float

function _slice_mode_from_vals(vals::Vector{_SliceValues})::Symbol
    n_none = 0
    n_int = 0
    n_float = 0
    for v in vals
        if v === nothing
            n_none += 1
        elseif v isa Vector{Int}
            n_int += 1
        elseif v isa Vector{Float64}
            n_float += 1
        else
            throw(ArgumentError("collect_slices: unsupported values payload type $(typeof(v))"))
        end
    end
    n = length(vals)
    if n_none == n
        return :none
    elseif n_int == n
        return :int
    elseif (n_int + n_float) == n
        return :float
    end
    throw(ArgumentError("collect_slices: mixed values modes (some with values, some without) are not supported in typed mode"))
end

function _collect_slices_none(chains::Vector{Vector{Int}}, weights::Vector{Float64})
    out = Vector{SliceSpec{Float64,Nothing}}(undef, length(chains))
    @inbounds for i in eachindex(chains)
        out[i] = SliceSpec{Float64,Nothing}(chains[i], nothing, weights[i])
    end
    return out
end

function _collect_slices_int(chains::Vector{Vector{Int}}, vals::Vector{_SliceValues}, weights::Vector{Float64})
    out = Vector{SliceSpec{Float64,Vector{Int}}}(undef, length(chains))
    @inbounds for i in eachindex(chains)
        vi = vals[i]
        vi === nothing && throw(ArgumentError("collect_slices(values_mode=:int): values[$i] is missing"))
        if vi isa Vector{Int}
            out[i] = SliceSpec{Float64,Vector{Int}}(chains[i], copy(vi), weights[i])
        elseif vi isa Vector{Float64} && _values_are_int_vector(vi)
            out[i] = SliceSpec{Float64,Vector{Int}}(chains[i], round.(Int, vi), weights[i])
        else
            throw(ArgumentError("collect_slices(values_mode=:int): values[$i] is not integer-valued"))
        end
    end
    return out
end

function _collect_slices_float(chains::Vector{Vector{Int}}, vals::Vector{_SliceValues}, weights::Vector{Float64})
    out = Vector{SliceSpec{Float64,Vector{Float64}}}(undef, length(chains))
    @inbounds for i in eachindex(chains)
        vi = vals[i]
        vi === nothing && throw(ArgumentError("collect_slices(values_mode=:float): values[$i] is missing"))
        if vi isa Vector{Float64}
            out[i] = SliceSpec{Float64,Vector{Float64}}(chains[i], copy(vi), weights[i])
        elseif vi isa Vector{Int}
            out[i] = SliceSpec{Float64,Vector{Float64}}(chains[i], Float64[float(v) for v in vi], weights[i])
        else
            throw(ArgumentError("collect_slices(values_mode=:float): values[$i] is not numeric"))
        end
    end
    return out
end

"""
    collect_slices(slices; default_weight=1.0, values=nothing, values_mode=:auto, weight_fn=nothing)

Normalize boundary slice specs into a concrete, type-stable `Vector{SliceSpec}`.

`values_mode` options:
- `:auto`  infer from input (all `nothing`, all integer vectors, or all numeric vectors),
- `:none`  force index-mode slices (`values = nothing`),
- `:int`   force integer-valued endpoints,
- `:float` force floating-point endpoints.
"""
function collect_slices(slices::AbstractVector;
                        default_weight::Real=1.0,
                        values=nothing,
                        values_mode::Symbol=:auto,
                        weight_fn=nothing)
    n = length(slices)
    chains = Vector{Vector{Int}}(undef, n)
    vals = Vector{_SliceValues}(undef, n)
    weights = Vector{Float64}(undef, n)

    has_values_override = values !== nothing
    if has_values_override
        (values isa AbstractVector && length(values) == n) ||
            throw(ArgumentError("collect_slices: values must be a vector of length length(slices) or nothing"))
    end

    @inbounds for i in 1:n
        chain, spec_vals, w = _parse_slice_spec(slices[i]; default_weight=default_weight, weight_fn=weight_fn)
        chains[i] = chain
        vals[i] = _normalize_slice_values(has_values_override ? values[i] : spec_vals)
        weights[i] = w
    end

    mode = values_mode
    if mode == :auto
        mode = _slice_mode_from_vals(vals)
    elseif mode == :index
        mode = :none
    elseif !(mode in (:none, :int, :float))
        throw(ArgumentError("collect_slices: values_mode must be :auto, :none, :int, :float, or :index"))
    end

    if mode == :none
        return _collect_slices_none(chains, weights)
    elseif mode == :int
        return _collect_slices_int(chains, vals, weights)
    else
        return _collect_slices_float(chains, vals, weights)
    end
end

# Generic iterable boundary: force materialization once at the boundary.
collect_slices(slices_iter; kwargs...) = collect_slices(collect(slices_iter); kwargs...)

# Already typed fast-path.
collect_slices(slices::AbstractVector{<:SliceSpec}; kwargs...) = slices

# Convenience for exact-slice family outputs returning NamedTuple with a :slices field.
collect_slices(x::NamedTuple; kwargs...) =
    hasproperty(x, :slices) ? collect_slices(getproperty(x, :slices); kwargs...) :
    throw(ArgumentError("collect_slices: expected NamedTuple with :slices"))

const SLICE_SPEC_SCHEMA_VERSION = 1

"""
    save_slices_json(path, slices; kwargs...) -> path

Serialize typed slice specs as JSON for reproducible statistics workflows.
"""
function save_slices_json(path::AbstractString, slices; kwargs...)
    specs = collect_slices(slices; kwargs...)
    mode = isempty(specs) ? "none" : String(_slice_mode_name(typeof(first(specs).values)))
    rows = Vector{Dict{String,Any}}(undef, length(specs))
    @inbounds for i in eachindex(specs)
        s = specs[i]
        rows[i] = Dict(
            "chain" => s.chain,
            "values" => s.values === nothing ? nothing : s.values,
            "weight" => float(s.weight),
        )
    end
    obj = Dict(
        "kind" => "slice_specs",
        "schema_version" => SLICE_SPEC_SCHEMA_VERSION,
        "values_mode" => mode,
        "slices" => rows,
    )
    open(path, "w") do io
        JSON3.write(io, obj; allow_inf=true, indent=2)
    end
    return path
end

"""
    load_slices_json(path; values_mode=:auto) -> Vector{SliceSpec}

Load typed slice specs from JSON created by `save_slices_json`.
"""
function load_slices_json(path::AbstractString; values_mode::Symbol=:auto)
    obj = open(JSON3.read, path)
    kind = haskey(obj, "kind") ? String(obj["kind"]) : ""
    kind == "slice_specs" || throw(ArgumentError("load_slices_json: unsupported kind $(kind)"))
    ver = haskey(obj, "schema_version") ? Int(obj["schema_version"]) : 0
    ver <= SLICE_SPEC_SCHEMA_VERSION || throw(ArgumentError("load_slices_json: unsupported schema_version $(ver)"))
    rows = haskey(obj, "slices") ? obj["slices"] : Any[]
    spec_rows = Vector{NamedTuple{(:chain,:values,:weight),Tuple{Vector{Int},_SliceValues,Float64}}}(undef, length(rows))
    @inbounds for i in eachindex(rows)
        r = rows[i]
        chain = haskey(r, "chain") ? Vector{Int}(Int.(collect(r["chain"]))) : Int[]
        vals = haskey(r, "values") ? r["values"] : nothing
        vals2 = vals === nothing ? nothing : _normalize_slice_values(collect(vals))
        w = haskey(r, "weight") ? float(r["weight"]) : 1.0
        spec_rows[i] = (chain=chain, values=vals2, weight=w)
    end
    mode = if values_mode == :auto
        haskey(obj, "values_mode") ? Symbol(String(obj["values_mode"])) : :auto
    else
        values_mode
    end
    mode == :index && (mode = :none)
    return collect_slices(spec_rows; values_mode=mode)
end

function _matching_distance_approx_specs(
    M::PModule{K},
    N::PModule{K},
    slices::AbstractVector{<:SliceSpec{<:Real,Nothing}},
)::Float64 where {K}
    best = 0.0
    @inbounds for spec in slices
        isempty(spec.chain) && continue
        bM = slice_barcode(M, spec.chain; values=nothing)
        bN = slice_barcode(N, spec.chain; values=nothing)
        best = max(best, float(spec.weight) * bottleneck_distance(bM, bN))
    end
    return best
end

function _matching_distance_approx_specs(
    M::PModule{K},
    N::PModule{K},
    slices::AbstractVector{<:SliceSpec{<:Real,<:AbstractVector{<:Integer}}},
)::Float64 where {K}
    best = 0.0
    @inbounds for spec in slices
        isempty(spec.chain) && continue
        bM = slice_barcode(M, spec.chain; values=spec.values)
        bN = slice_barcode(N, spec.chain; values=spec.values)
        best = max(best, float(spec.weight) * bottleneck_distance(bM, bN))
    end
    return best
end

function _matching_distance_approx_specs(
    M::PModule{K},
    N::PModule{K},
    slices::AbstractVector{<:SliceSpec{<:Real,<:AbstractVector{<:Real}}},
)::Float64 where {K}
    best = 0.0
    @inbounds for spec in slices
        isempty(spec.chain) && continue
        bM = slice_barcode(M, spec.chain; values=spec.values)
        bN = slice_barcode(N, spec.chain; values=spec.values)
        best = max(best, float(spec.weight) * bottleneck_distance(bM, bN))
    end
    return best
end

const _EMPTY_CHAIN_POOL = Int[]
const _EMPTY_VALS_POOL = Float64[]
const _PooledChainSlice = SubArray{Int,1,Vector{Int},Tuple{UnitRange{Int}},true}

struct _PooledChainRows <: AbstractVector{_PooledChainSlice}
    pool::Vector{Int}
    start::Vector{Int}
    len::Vector{Int}
end

"""
    CompiledSlicePlan

Precompiled slice geometry for repeated `slice_barcodes`/distance queries on a fixed
encoding map and sampling configuration.
"""
struct CompiledSlicePlan
    dirs::Vector{Vector{Float64}}
    offs::Vector{Vector{Float64}}
    weights::Matrix{Float64}
    chain_pool::Vector{Int}
    chain_start::Vector{Int}
    chain_len::Vector{Int}
    chains::_PooledChainRows
    vals_pool::Vector{Float64}
    vals_start::Vector{Int}
    vals_len::Vector{Int}
    nd::Int
    no::Int
end

@inline Base.size(rows::_PooledChainRows) = (length(rows.start),)
@inline Base.length(rows::_PooledChainRows) = length(rows.start)
@inline Base.IndexStyle(::Type{<: _PooledChainRows}) = IndexLinear()
@inline function Base.getindex(rows::_PooledChainRows, idx::Int)
    s = rows.start[idx]
    l = rows.len[idx]
    return l == 0 ? @view(_EMPTY_CHAIN_POOL[1:0]) : @view(rows.pool[s:(s + l - 1)])
end

@inline _plan_chain(plan::CompiledSlicePlan, idx::Int) = plan.chains[idx]
@inline function _plan_vals(plan::CompiledSlicePlan, idx::Int)
    s = plan.vals_start[idx]
    l = plan.vals_len[idx]
    return l == 0 ? @view(_EMPTY_VALS_POOL[1:0]) : @view(plan.vals_pool[s:(s + l - 1)])
end

@inline function _compiled_slice_plan(
    dirs::Vector{Vector{Float64}},
    offs::Vector{Vector{Float64}},
    W::Matrix{Float64},
    chain_pool::Vector{Int},
    chain_start::Vector{Int},
    chain_len::Vector{Int},
    vals_pool::Vector{Float64},
    vals_start::Vector{Int},
    vals_len::Vector{Int},
    nd::Int,
    no::Int,
)
    return CompiledSlicePlan(
        dirs,
        offs,
        W,
        chain_pool,
        chain_start,
        chain_len,
        _PooledChainRows(chain_pool, chain_start, chain_len),
        vals_pool,
        vals_start,
        vals_len,
        nd,
        no,
    )
end

function _compiled_slice_plan_from_vectors(
    dirs::Vector{Vector{Float64}},
    offs::Vector{Vector{Float64}},
    W::Matrix{Float64},
    chains::Vector{Vector{Int}},
    vals_tmp::Vector{Vector{Float64}},
    nd::Int,
    no::Int,
)
    ns = length(chains)
    chain_start = zeros(Int, ns)
    chain_len = zeros(Int, ns)
    vals_start = zeros(Int, ns)
    vals_len = zeros(Int, ns)

    total_chain = 0
    total_vals = 0
    @inbounds for idx in 1:ns
        lc = length(chains[idx])
        lv = length(vals_tmp[idx])
        chain_len[idx] = lc
        vals_len[idx] = lv
        if lc > 0
            chain_start[idx] = total_chain + 1
            total_chain += lc
        end
        if lv > 0
            vals_start[idx] = total_vals + 1
            total_vals += lv
        end
    end

    chain_pool = Vector{Int}(undef, total_chain)
    vals_pool = Vector{Float64}(undef, total_vals)
    @inbounds for idx in 1:ns
        lc = chain_len[idx]
        lc == 0 || copyto!(chain_pool, chain_start[idx], chains[idx], 1, lc)
        lv = vals_len[idx]
        lv == 0 || copyto!(vals_pool, vals_start[idx], vals_tmp[idx], 1, lv)
    end

    return _compiled_slice_plan(dirs, offs, W, chain_pool, chain_start, chain_len,
                                vals_pool, vals_start, vals_len, nd, no)
end

mutable struct _SliceChainCountState
    n::Int
end

@inline function (state::_SliceChainCountState)(::Int, ::Float64)
    state.n += 1
    return nothing
end

mutable struct _SliceChainFillState
    chain_pool::Vector{Int}
    vals_pool::Vector{Float64}
    chain_pos::Int
    vals_pos::Int
end

@inline function (state::_SliceChainFillState)(rid::Int, t::Float64)
    state.chain_pool[state.chain_pos] = rid
    state.vals_pool[state.vals_pos] = t
    state.chain_pos += 1
    state.vals_pos += 1
    return nothing
end

@inline function _slice_chain_count(
    pi,
    x0::Vector{Float64},
    dir::Vector{Float64},
    opts::InvariantOptions;
    locate_ws = nothing,
    kwargs...
)
    state = _SliceChainCountState(0)
    _slice_chain_visit(pi, x0, dir, opts, state; locate_ws=locate_ws, kwargs...)
    return state.n
end

@inline function _slice_chain_fill!(
    chain_pool::Vector{Int},
    chain_start::Int,
    vals_pool::Vector{Float64},
    vals_start::Int,
    pi,
    x0::Vector{Float64},
    dir::Vector{Float64},
    opts::InvariantOptions;
    locate_ws = nothing,
    kwargs...
)
    state = _SliceChainFillState(chain_pool, vals_pool, chain_start, vals_start)
    _slice_chain_visit(pi, x0, dir, opts, state; locate_ws=locate_ws, kwargs...)
    return nothing
end


struct SlicePlanCacheKey
    pi_id::UInt
    normalize_dirs::Symbol
    n_dirs::Int
    n_offsets::Int
    max_den::Int
    include_axes::Bool
    offset_margin::Float64
    drop_unknown::Bool
    strict_code::Int8
    box_hash::UInt
    directions_hash::UInt
    offsets_hash::UInt
    weight_hash::UInt
    kwargs_hash::UInt
end

mutable struct SlicePlanCache <: AbstractSlicePlanCache
    lock::ReentrantLock
    plans::Dict{SlicePlanCacheKey,CompiledSlicePlan}
end

SlicePlanCache() = SlicePlanCache(ReentrantLock(), Dict{SlicePlanCacheKey,CompiledSlicePlan}())

const _GLOBAL_SLICE_PLAN_CACHE = SlicePlanCache()

function clear_slice_plan_cache!(cache::SlicePlanCache = _GLOBAL_SLICE_PLAN_CACHE)
    Base.lock(cache.lock)
    try
        empty!(cache.plans)
    finally
        Base.unlock(cache.lock)
    end
    return nothing
end

"""
    SliceModuleCache(M)

Module-specific cache wrapper used by `run_invariants` and compiled-plan
surfaces. The cache stores packed per-plan barcode grids so repeated distance,
kernel, and feature queries on the same `(module, plan)` reuse slice barcodes.
"""
struct SliceBarcodeCacheKey
    plan_id::UInt
    value_mode::Symbol
end

struct SliceLandscapeCacheKey
    plan_id::UInt
    kmax::Int
    tgrid_len::Int
    tgrid_hash::UInt
    tgrid_first::Float64
    tgrid_last::Float64
end

mutable struct SliceModuleCache{K,F<:AbstractCoeffField,MatT<:AbstractMatrix{K}}
    M::PModule{K,F,MatT}
    lock::ReentrantLock
    packed_plan_barcodes::Dict{SliceBarcodeCacheKey,PackedBarcodeGrid{PackedFloatBarcode}}
    landscape_plan_features::Dict{SliceLandscapeCacheKey,Vector{Vector{Float64}}}
end

"""
    SliceModulePairCache(A, B)

Pair cache for two modules sharing one compiled slice plan.
"""
struct SliceModulePairCache{CA<:SliceModuleCache,CB<:SliceModuleCache}
    A::CA
    B::CB
end

const _GLOBAL_SLICE_MODULE_CACHE_LOCK = ReentrantLock()
const _GLOBAL_SLICE_MODULE_CACHE = IdDict{Any,SliceModuleCache}()

function clear_slice_module_cache!()
    Base.lock(_GLOBAL_SLICE_MODULE_CACHE_LOCK)
    try
        empty!(_GLOBAL_SLICE_MODULE_CACHE)
    finally
        Base.unlock(_GLOBAL_SLICE_MODULE_CACHE_LOCK)
    end
    return nothing
end

function module_cache(M::PModule{K,F,MatT}) where {K,F<:AbstractCoeffField,MatT<:AbstractMatrix{K}}
    Base.lock(_GLOBAL_SLICE_MODULE_CACHE_LOCK)
    try
        cache = get(_GLOBAL_SLICE_MODULE_CACHE, M, nothing)
        if cache === nothing
            cache = SliceModuleCache{K,F,MatT}(
                M,
                ReentrantLock(),
                Dict{SliceBarcodeCacheKey,PackedBarcodeGrid{PackedFloatBarcode}}(),
                Dict{SliceLandscapeCacheKey,Vector{Vector{Float64}}}(),
            )
            _GLOBAL_SLICE_MODULE_CACHE[M] = cache
        end
        return cache::SliceModuleCache{K,F,MatT}
    finally
        Base.unlock(_GLOBAL_SLICE_MODULE_CACHE_LOCK)
    end
end

@inline module_cache(A::PModule, B::PModule) = SliceModulePairCache(module_cache(A), module_cache(B))

@inline _slice_plan_barcode_key(plan::CompiledSlicePlan) =
    SliceBarcodeCacheKey(UInt(objectid(plan)), :t)

@inline function _slice_plan_landscape_key(
    plan::CompiledSlicePlan,
    tgrid::AbstractVector{Float64},
    kmax::Int,
)
    n = length(tgrid)
    first_t = n == 0 ? NaN : tgrid[1]
    last_t = n == 0 ? NaN : tgrid[end]
    return SliceLandscapeCacheKey(
        UInt(objectid(plan)),
        kmax,
        n,
        _tgrid_cache_hash(tgrid),
        first_t,
        last_t,
    )
end

function _slice_barcodes_plan_packed_uncached(
    M::PModule{K},
    plan::CompiledSlicePlan;
    threads::Bool = (Threads.nthreads() > 1),
) where {K}
    ns = plan.nd * plan.no
    bars = _packed_grid_undef(PackedFloatBarcode, plan.nd, plan.no)
    max_chain = isempty(plan.chain_len) ? 0 : maximum(plan.chain_len)
    if max_chain == 0
        @inbounds for idx in 1:ns
            bars[idx] = _empty_packed_float_barcode()
        end
        return bars
    end

    build_cache!(M.Q; cover=true, updown=true)
    cc = _get_cover_cache(M.Q)
    nQ = nvertices(M.Q)
    use_array_memo = _use_array_memo(nQ)

    if threads && Threads.nthreads() > 1
        nT = Threads.nthreads()
        memo_by_thread = use_array_memo ?
            [_new_array_memo(K, nQ) for _ in 1:nT] :
            [Dict{Tuple{Int,Int}, AbstractMatrix{K}}() for _ in 1:nT]
        rank_by_thread = [Matrix{Int}(undef, max_chain, max_chain) for _ in 1:nT]
        Threads.@threads for idx in 1:ns
            chain = _plan_chain(plan, idx)
            if isempty(chain)
                bars[idx] = _empty_packed_float_barcode()
                continue
            end
            vals = _plan_vals(plan, idx)
            tid = Threads.threadid()
            bars[idx] = _slice_barcode_packed_with_workspace(
                M,
                chain,
                _extended_values_view(vals),
                cc,
                memo_by_thread[tid],
                rank_by_thread[tid],
            )
        end
    else
        memo = use_array_memo ? _new_array_memo(K, nQ) : Dict{Tuple{Int,Int}, AbstractMatrix{K}}()
        rank_work = Matrix{Int}(undef, max_chain, max_chain)
        @inbounds for idx in 1:ns
            chain = _plan_chain(plan, idx)
            if isempty(chain)
                bars[idx] = _empty_packed_float_barcode()
                continue
            end
            vals = _plan_vals(plan, idx)
            bars[idx] = _slice_barcode_packed_with_workspace(
                M,
                chain,
                _extended_values_view(vals),
                cc,
                memo,
                rank_work,
            )
        end
    end
    return bars
end

function _slice_barcodes_plan_packed_cached(
    cache::SliceModuleCache,
    plan::CompiledSlicePlan;
    threads::Bool = (Threads.nthreads() > 1),
)
    key = _slice_plan_barcode_key(plan)
    Base.lock(cache.lock)
    try
        bars = get(cache.packed_plan_barcodes, key, nothing)
        bars === nothing || return bars
    finally
        Base.unlock(cache.lock)
    end

    bars = _slice_barcodes_plan_packed_uncached(cache.M, plan; threads=threads)

    Base.lock(cache.lock)
    try
        return get!(cache.packed_plan_barcodes, key, bars)
    finally
        Base.unlock(cache.lock)
    end
end

@inline function _slice_barcodes_plan_result_from_packed(
    bars::PackedBarcodeGrid{PackedFloatBarcode},
    plan::CompiledSlicePlan;
    packed::Bool,
)
    return _slice_barcodes_result_from_packed(bars, plan.weights, plan.dirs, plan.offs; packed=packed)
end

"""
    SliceBarcodesResult

Typed owner-level result wrapper for sliced barcode families.

This stores:
- `barcodes`: the computed barcode collection, either dense or packed,
- `weights`: slice weights aligned with the barcode collection,
- `dirs`: the sampled slice directions,
- `offs`: the sampled slice offsets/basepoints.

Use semantic accessors such as [`slice_barcodes`](@ref), [`slice_weights`](@ref),
[`slice_directions`](@ref), [`slice_offsets`](@ref), and
[`packed_barcodes`](@ref) instead of field archaeology. Prefer
`describe(result)` for cheap-first inspection before unpacking the full barcode
collection.
"""
struct SliceBarcodesResult{B,W,D,O}
    barcodes::B
    weights::W
    dirs::D
    offs::O
end

@inline _slice_barcodes_result(barcodes, weights, dirs, offs) =
    SliceBarcodesResult(barcodes, weights, dirs, offs)

@inline _slice_barcodes_result_without_geometry(barcodes, weights) =
    _slice_barcodes_result(barcodes, weights, Any[], Any[])

"""
    SliceFeaturesResult

Typed owner-level result wrapper for sliced feature workflows.

This stores:
- `features`: the aggregated feature payload,
- `weights`: the slice weights used for aggregation,
- `featurizer`: the chosen per-slice feature family,
- `aggregate`: the cross-slice aggregation mode.

Use [`slice_features`](@ref), [`slice_weights`](@ref), [`feature_kind`](@ref),
and [`feature_aggregate`](@ref) instead of reading fields directly. Prefer
`describe(result)` for cheap-first inspection before unpacking the feature
payload.
"""
struct SliceFeaturesResult{F,W,FT,AT}
    features::F
    weights::W
    featurizer::FT
    aggregate::AT
end

@inline _slice_features_result(features, weights; featurizer, aggregate) =
    SliceFeaturesResult(features, weights, featurizer, aggregate)

@inline function _slice_barcodes_result_from_packed(
    bars,
    weights::AbstractMatrix{Float64},
    dirs,
    offs;
    packed::Bool,
)
    if packed
        return _slice_barcodes_result(bars, weights, dirs, offs)
    end
    return _slice_barcodes_result(_float_dict_matrix_from_packed_grid(bars), weights, dirs, offs)
end

@inline function _slice_barcodes_plan_result_uncached(
    M::PModule{K},
    plan::CompiledSlicePlan;
    packed::Bool = false,
    threads::Bool = (Threads.nthreads() > 1),
) where {K}
    bars = _slice_barcodes_plan_packed_uncached(M, plan; threads=threads)
    return _slice_barcodes_plan_result_from_packed(bars, plan; packed=packed)
end

"""
    SliceBarcodesTask(; packed=false, threads=Threads.nthreads() > 1)

Task descriptor for `run_invariants(plan, module_cache, task)` that computes
slice barcodes on a compiled plan.
"""
Base.@kwdef struct SliceBarcodesTask
    packed::Bool = false
    threads::Bool = (Threads.nthreads() > 1)
end

"""
    SliceDistanceTask(; ...)

Task descriptor for per-slice distance aggregation over a compiled plan.
"""
Base.@kwdef struct SliceDistanceTask{F,NT,AT}
    dist_fn::F = bottleneck_distance
    dist_kwargs::NT = NamedTuple()
    weight_mode::Symbol = :integrate
    agg::AT = :mean
    agg_p::Float64 = 2.0
    agg_norm::Float64 = 1.0
    threads::Bool = (Threads.nthreads() > 1)
end

"""
    SliceKernelTask(; ...)

Task descriptor for sliced kernel aggregation over a compiled plan.
"""
Base.@kwdef struct SliceKernelTask{KT,GT}
    kind::KT = :bottleneck_gaussian
    sigma::Float64 = 1.0
    gamma::GT = nothing
    p::Float64 = 2.0
    q::Float64 = Inf
    tgrid = nothing
    tgrid_nsteps::Int = 401
    kmax::Int = 5
    threads::Bool = (Threads.nthreads() > 1)
end

# ----- UX layer: inspection, accessors, and validation -----------------------

"""
    SliceInvariantValidationSummary

Compact wrapper around a validation report produced by the `SliceInvariants`
`check_*` helpers.

Every wrapped report contains at least:
- `kind`: symbolic object kind
- `valid`: overall validation result
- `issues`: a tuple of human-readable validation issues
"""
struct SliceInvariantValidationSummary{R}
    report::R
end

"""
    slice_invariant_validation_summary(report) -> SliceInvariantValidationSummary

Wrap a raw validation report returned by the `SliceInvariants` UX-layer
`check_*` helpers in a compact display-oriented object.
"""
@inline slice_invariant_validation_summary(report::NamedTuple) = SliceInvariantValidationSummary(report)

@inline function _sliceinv_issue_report(
    kind::Symbol,
    valid::Bool;
    issues::AbstractVector{<:AbstractString}=String[],
    kwargs...,
)
    return (; kind, valid, issues=Tuple(String.(issues)), kwargs...)
end

@inline function _throw_invalid_sliceinvariants(fname::Symbol, issues::AbstractVector{<:AbstractString})
    msg = isempty(issues) ? "invalid object" : " - " * join(issues, "\n - ")
    Base.throw(ArgumentError(string(fname, ": validation failed\n", msg)))
end

function Base.show(io::IO, summary::SliceInvariantValidationSummary)
    r = summary.report
    print(io, "SliceInvariantValidationSummary(kind=", r.kind,
          ", valid=", r.valid,
          ", issues=", length(r.issues), ")")
end

function Base.show(io::IO, ::MIME"text/plain", summary::SliceInvariantValidationSummary)
    r = summary.report
    println(io, "SliceInvariantValidationSummary")
    println(io, "  kind: ", r.kind)
    println(io, "  valid: ", r.valid)
    for key in propertynames(r)
        key in (:kind, :valid, :issues) && continue
        println(io, "  ", key, ": ", repr(getproperty(r, key)))
    end
    if isempty(r.issues)
        print(io, "  issues: []")
    else
        println(io, "  issues:")
        for issue in r.issues
            println(io, "    - ", issue)
        end
    end
end

@inline _sliceinv_callable_label(f::Function) = string(f)
@inline _sliceinv_callable_label(f) = string(typeof(f))

@inline _sliceinv_point_kind(x::AbstractVector{<:Integer}) = :integer_vector
@inline _sliceinv_point_kind(x::AbstractVector{<:AbstractFloat}) = :float_vector
@inline _sliceinv_point_kind(x::AbstractVector{<:Real}) = :real_vector
@inline _sliceinv_point_kind(x::NTuple{N,<:Integer}) where {N} = :integer_tuple
@inline _sliceinv_point_kind(x::NTuple{N,<:AbstractFloat}) where {N} = :float_tuple
@inline _sliceinv_point_kind(x::NTuple{N,<:Real}) where {N} = :real_tuple
@inline _sliceinv_point_kind(x::Tuple) = all(v -> v isa Real, x) ? :real_tuple : :invalid
@inline _sliceinv_point_kind(::Any) = :invalid

@inline _sliceinv_point_length(x::AbstractVector) = length(x)
@inline _sliceinv_point_length(x::Tuple) = length(x)
@inline _sliceinv_point_length(::Any) = nothing

@inline _sliceinv_plan_ambient_dim(plan::CompiledSlicePlan) =
    !isempty(plan.dirs) ? length(first(plan.dirs)) :
    !isempty(plan.offs) ? length(first(plan.offs)) : 0

"""
    landscape_grid(pl)
    landscape_values(pl)
    landscape_layers(pl)

Cheap semantic accessors for a [`PersistenceLandscape1D`](@ref).

Use these before reading raw storage fields directly.
"""
@inline landscape_grid(pl::PersistenceLandscape1D) = pl.tgrid
@inline landscape_values(pl::PersistenceLandscape1D) = pl.values
@inline landscape_layers(pl::PersistenceLandscape1D)::Int = size(pl.values, 1)
@inline feature_dimension(pl::PersistenceLandscape1D)::Int = length(pl.values)

"""
    image_xgrid(PI)
    image_ygrid(PI)
    image_values(PI)

Cheap semantic accessors for a [`PersistenceImage1D`](@ref).

Use [`image_shape`](@ref) for the cheap shape summary and `image_values(PI)` when
you explicitly need the image matrix.
"""
@inline image_xgrid(PI::PersistenceImage1D) = PI.xgrid
@inline image_ygrid(PI::PersistenceImage1D) = PI.ygrid
@inline image_values(PI::PersistenceImage1D) = PI.values
@inline image_shape(PI::PersistenceImage1D) = size(PI.values)

"""
    plan_directions(plan)
    plan_offsets(plan)
    plan_weights(plan)

Cheap semantic accessors for a [`CompiledSlicePlan`](@ref).

Use these before reaching into the plan's internal fields directly.
"""
@inline plan_directions(plan::CompiledSlicePlan) = plan.dirs
@inline plan_offsets(plan::CompiledSlicePlan) = plan.offs
@inline plan_weights(plan::CompiledSlicePlan) = plan.weights
@inline nslices(plan::CompiledSlicePlan)::Int = plan.nd * plan.no
@inline total_weight(plan::CompiledSlicePlan)::Float64 = sum(plan.weights)
@inline plan_has_values(plan::CompiledSlicePlan)::Bool = any(>(0), plan.vals_len)
@inline plan_value_mode(::CompiledSlicePlan) = :t

"""
    slice_spec(plan, idx) -> SliceSpec{Float64,Vector{Float64}}

Return the `idx`-th explicit slice encoded by a compiled plan.

This is the canonical semantic accessor for recovering one weighted slice from a
[`CompiledSlicePlan`](@ref) without materializing the entire collection via
[`collect_slices`](@ref).
"""
function slice_spec(plan::CompiledSlicePlan, idx::Int)
    1 <= idx <= nslices(plan) || throw(BoundsError(plan, idx))
    i = div(idx - 1, plan.no) + 1
    j = (idx - 1) % plan.no + 1
    return SliceSpec(
        Vector{Int}(_plan_chain(plan, idx));
        values=Vector{Float64}(_plan_vals(plan, idx)),
        weight=plan.weights[i, j],
    )
end

"""
    slice_barcodes(result)
    slice_weights(result)
    slice_directions(result)
    slice_offsets(result)
    packed_barcodes(result)

Semantic accessors for a [`SliceBarcodesResult`](@ref).

`packed_barcodes(result)` returns the stored packed barcode grid when the result
was computed in packed mode and `nothing` otherwise.
"""
@inline slice_barcodes(result::SliceBarcodesResult) = result.barcodes
@inline slice_weights(result::SliceBarcodesResult) = result.weights
@inline slice_directions(result::SliceBarcodesResult) = result.dirs
@inline slice_offsets(result::SliceBarcodesResult) = result.offs
@inline packed_barcodes(result::SliceBarcodesResult) =
    result.barcodes isa PackedBarcodeGrid ? result.barcodes : nothing

"""
    slice_features(result)
    slice_weights(result)
    feature_kind(result)
    feature_aggregate(result)

Semantic accessors for a [`SliceFeaturesResult`](@ref).
"""
@inline slice_features(result::SliceFeaturesResult) = result.features
@inline slice_weights(result::SliceFeaturesResult) = result.weights
@inline feature_kind(result::SliceFeaturesResult) = result.featurizer
@inline feature_aggregate(result::SliceFeaturesResult) = result.aggregate

"""
    cached_barcode_plan_count(cache)
    cached_landscape_plan_count(cache)

Cheap scalar cache inspectors for a [`SliceModuleCache`](@ref).
"""
@inline source_module(cache::SliceModuleCache) = cache.M
@inline left_cache(pair::SliceModulePairCache) = pair.A
@inline right_cache(pair::SliceModulePairCache) = pair.B
@inline cached_barcode_plan_count(cache::SliceModuleCache)::Int = length(cache.packed_plan_barcodes)
@inline cached_landscape_plan_count(cache::SliceModuleCache)::Int = length(cache.landscape_plan_features)

"""
    task_kind(task)
    task_threads(task)
    distance_function(task)
    kernel_kind(task)

Cheap semantic accessors for slice task descriptors.
"""
@inline task_kind(::SliceBarcodesTask) = :barcodes
@inline task_kind(::SliceDistanceTask) = :distance
@inline task_kind(::SliceKernelTask) = :kernel
@inline task_threads(task::Union{SliceBarcodesTask,SliceDistanceTask,SliceKernelTask}) = task.threads
@inline distance_function(task::SliceDistanceTask) = task.dist_fn
@inline kernel_kind(task::SliceKernelTask) = task.kind
@inline kernel_sigma(task::SliceKernelTask) = task.sigma

@inline _slice_barcodes_result_shape(result::SliceBarcodesResult) = size(slice_barcodes(result))
@inline _slice_barcodes_result_slice_count(result::SliceBarcodesResult)::Int = length(slice_weights(result))

@inline function _slice_features_result_shape(result::SliceFeaturesResult)
    feats = slice_features(result)
    return feats isa Number ? nothing : size(feats)
end

@inline _slice_features_result_slice_count(result::SliceFeaturesResult)::Int = length(slice_weights(result))

@inline function _sliceinv_describe(result::SliceBarcodesResult)
    return (;
        kind=:slice_barcodes_result,
        packed=!isnothing(packed_barcodes(result)),
        barcode_shape=_slice_barcodes_result_shape(result),
        weight_shape=size(slice_weights(result)),
        slice_count=_slice_barcodes_result_slice_count(result),
        ndirections=length(slice_directions(result)),
        noffsets=length(slice_offsets(result)),
        total_weight=sum(slice_weights(result)),
    )
end

@inline function _sliceinv_describe(result::SliceFeaturesResult)
    return (;
        kind=:slice_features_result,
        feature_kind=feature_kind(result),
        aggregate=feature_aggregate(result),
        scalar_output=slice_features(result) isa Number,
        feature_shape=_slice_features_result_shape(result),
        weight_shape=size(slice_weights(result)),
        slice_count=_slice_features_result_slice_count(result),
        total_weight=sum(slice_weights(result)),
    )
end

@inline function _sliceinv_describe(pl::PersistenceLandscape1D)
    return (;
        kind=:persistence_landscape_1d,
        grid_length=length(landscape_grid(pl)),
        landscape_layers=landscape_layers(pl),
        feature_dimension=feature_dimension(pl),
        t_range=isempty(pl.tgrid) ? nothing : (pl.tgrid[1], pl.tgrid[end]),
    )
end

@inline function _sliceinv_describe(PI::PersistenceImage1D)
    return (;
        kind=:persistence_image_1d,
        image_shape=image_shape(PI),
        feature_dimension=length(PI.values),
        x_range=isempty(PI.xgrid) ? nothing : (PI.xgrid[1], PI.xgrid[end]),
        y_range=isempty(PI.ygrid) ? nothing : (PI.ygrid[1], PI.ygrid[end]),
    )
end

@inline function _sliceinv_describe(plan::CompiledSlicePlan)
    return (;
        kind=:compiled_slice_plan,
        ambient_dim=_sliceinv_plan_ambient_dim(plan),
        ndirections=plan.nd,
        noffsets=plan.no,
        nslices=nslices(plan),
        total_weight=total_weight(plan),
        has_values=plan_has_values(plan),
        value_mode=plan_value_mode(plan),
    )
end

@inline function _sliceinv_describe(cache::SlicePlanCache)
    Base.lock(cache.lock)
    try
        plans = collect(values(cache.plans))
        return (;
            kind=:slice_plan_cache,
            cached_plans=length(plans),
            ambient_dims=Tuple(sort!(collect(Set(_sliceinv_plan_ambient_dim(plan) for plan in plans)))),
        )
    finally
        Base.unlock(cache.lock)
    end
end

@inline function _sliceinv_describe(cache::SliceModuleCache)
    return (;
        kind=:slice_module_cache,
        source_module_type=typeof(source_module(cache)),
        poset_size=nvertices(source_module(cache).Q),
        cached_barcode_plans=cached_barcode_plan_count(cache),
        cached_landscape_plans=cached_landscape_plan_count(cache),
    )
end

@inline function _sliceinv_describe(pair::SliceModulePairCache)
    left = left_cache(pair)
    right = right_cache(pair)
    return (;
        kind=:slice_module_pair_cache,
        left_poset_size=nvertices(source_module(left).Q),
        right_poset_size=nvertices(source_module(right).Q),
        left_cached_barcode_plans=cached_barcode_plan_count(left),
        right_cached_barcode_plans=cached_barcode_plan_count(right),
        left_cached_landscape_plans=cached_landscape_plan_count(left),
        right_cached_landscape_plans=cached_landscape_plan_count(right),
    )
end

@inline function _sliceinv_describe(task::SliceBarcodesTask)
    return (;
        kind=:slice_barcodes_task,
        task_kind=task_kind(task),
        packed=task.packed,
        threads=task_threads(task),
    )
end

@inline function _sliceinv_describe(task::SliceDistanceTask)
    return (;
        kind=:slice_distance_task,
        task_kind=task_kind(task),
        distance_function=_sliceinv_callable_label(distance_function(task)),
        weight_mode=task.weight_mode,
        agg=task.agg === mean ? :mean : task.agg === maximum ? :max : task.agg,
        agg_p=task.agg_p,
        agg_norm=task.agg_norm,
        threads=task_threads(task),
    )
end

@inline function _sliceinv_describe(task::SliceKernelTask)
    return (;
        kind=:slice_kernel_task,
        task_kind=task_kind(task),
        kernel_kind=kernel_kind(task),
        sigma=kernel_sigma(task),
        gamma=task.gamma,
        p=task.p,
        q=task.q,
        kmax=task.kmax,
        tgrid_nsteps=task.tgrid_nsteps,
        threads=task_threads(task),
    )
end

describe(pl::PersistenceLandscape1D) = _sliceinv_describe(pl)
describe(PI::PersistenceImage1D) = _sliceinv_describe(PI)
describe(result::SliceBarcodesResult) = _sliceinv_describe(result)
describe(result::SliceFeaturesResult) = _sliceinv_describe(result)
describe(plan::CompiledSlicePlan) = _sliceinv_describe(plan)
describe(cache::SlicePlanCache) = _sliceinv_describe(cache)
describe(cache::SliceModuleCache) = _sliceinv_describe(cache)
describe(pair::SliceModulePairCache) = _sliceinv_describe(pair)
describe(task::SliceBarcodesTask) = _sliceinv_describe(task)
describe(task::SliceDistanceTask) = _sliceinv_describe(task)
describe(task::SliceKernelTask) = _sliceinv_describe(task)

function Base.show(io::IO, pl::PersistenceLandscape1D)
    d = _sliceinv_describe(pl)
    print(io, "PersistenceLandscape1D(k=", d.landscape_layers,
          ", ngrid=", d.grid_length, ")")
end

function Base.show(io::IO, ::MIME"text/plain", pl::PersistenceLandscape1D)
    d = _sliceinv_describe(pl)
    print(io, "PersistenceLandscape1D",
          "\n  landscape_layers = ", d.landscape_layers,
          "\n  grid_length = ", d.grid_length,
          "\n  feature_dimension = ", d.feature_dimension,
          "\n  t_range = ", d.t_range)
end

function Base.show(io::IO, PI::PersistenceImage1D)
    d = _sliceinv_describe(PI)
    print(io, "PersistenceImage1D(shape=", d.image_shape, ")")
end

function Base.show(io::IO, ::MIME"text/plain", PI::PersistenceImage1D)
    d = _sliceinv_describe(PI)
    print(io, "PersistenceImage1D",
          "\n  image_shape = ", d.image_shape,
          "\n  feature_dimension = ", d.feature_dimension,
          "\n  x_range = ", d.x_range,
          "\n  y_range = ", d.y_range)
end

function Base.show(io::IO, result::SliceBarcodesResult)
    d = _sliceinv_describe(result)
    print(io, "SliceBarcodesResult(shape=", d.barcode_shape,
          ", packed=", d.packed, ")")
end

function Base.show(io::IO, ::MIME"text/plain", result::SliceBarcodesResult)
    d = _sliceinv_describe(result)
    print(io, "SliceBarcodesResult",
          "\n  barcode_shape = ", d.barcode_shape,
          "\n  weight_shape = ", d.weight_shape,
          "\n  slice_count = ", d.slice_count,
          "\n  packed = ", d.packed,
          "\n  ndirections = ", d.ndirections,
          "\n  noffsets = ", d.noffsets,
          "\n  total_weight = ", d.total_weight)
end

function Base.show(io::IO, result::SliceFeaturesResult)
    d = _sliceinv_describe(result)
    print(io, "SliceFeaturesResult(kind=", d.feature_kind,
          ", aggregate=", d.aggregate, ")")
end

function Base.show(io::IO, ::MIME"text/plain", result::SliceFeaturesResult)
    d = _sliceinv_describe(result)
    print(io, "SliceFeaturesResult",
          "\n  feature_kind = ", d.feature_kind,
          "\n  aggregate = ", d.aggregate,
          "\n  scalar_output = ", d.scalar_output,
          "\n  feature_shape = ", d.feature_shape,
          "\n  weight_shape = ", d.weight_shape,
          "\n  slice_count = ", d.slice_count,
          "\n  total_weight = ", d.total_weight)
end

function Base.show(io::IO, plan::CompiledSlicePlan)
    d = _sliceinv_describe(plan)
    print(io, "CompiledSlicePlan(n=", d.ambient_dim,
          ", slices=", d.nslices,
          ", dirs=", d.ndirections,
          ", offsets=", d.noffsets, ")")
end

function Base.show(io::IO, ::MIME"text/plain", plan::CompiledSlicePlan)
    d = _sliceinv_describe(plan)
    print(io, "CompiledSlicePlan",
          "\n  ambient_dim = ", d.ambient_dim,
          "\n  ndirections = ", d.ndirections,
          "\n  noffsets = ", d.noffsets,
          "\n  nslices = ", d.nslices,
          "\n  total_weight = ", d.total_weight,
          "\n  has_values = ", d.has_values,
          "\n  value_mode = ", d.value_mode)
end

function Base.show(io::IO, cache::SlicePlanCache)
    d = _sliceinv_describe(cache)
    print(io, "SlicePlanCache(cached_plans=", d.cached_plans, ")")
end

function Base.show(io::IO, ::MIME"text/plain", cache::SlicePlanCache)
    d = _sliceinv_describe(cache)
    print(io, "SlicePlanCache",
          "\n  cached_plans = ", d.cached_plans,
          "\n  ambient_dims = ", d.ambient_dims)
end

function Base.show(io::IO, cache::SliceModuleCache)
    d = _sliceinv_describe(cache)
    print(io, "SliceModuleCache(poset_size=", d.poset_size,
          ", barcodes=", d.cached_barcode_plans,
          ", landscapes=", d.cached_landscape_plans, ")")
end

function Base.show(io::IO, ::MIME"text/plain", cache::SliceModuleCache)
    d = _sliceinv_describe(cache)
    print(io, "SliceModuleCache",
          "\n  source_module_type = ", d.source_module_type,
          "\n  poset_size = ", d.poset_size,
          "\n  cached_barcode_plans = ", d.cached_barcode_plans,
          "\n  cached_landscape_plans = ", d.cached_landscape_plans)
end

function Base.show(io::IO, pair::SliceModulePairCache)
    d = _sliceinv_describe(pair)
    print(io, "SliceModulePairCache(left=", d.left_poset_size,
          ", right=", d.right_poset_size, ")")
end

function Base.show(io::IO, ::MIME"text/plain", pair::SliceModulePairCache)
    d = _sliceinv_describe(pair)
    print(io, "SliceModulePairCache",
          "\n  left_poset_size = ", d.left_poset_size,
          "\n  right_poset_size = ", d.right_poset_size,
          "\n  left_cached_barcode_plans = ", d.left_cached_barcode_plans,
          "\n  right_cached_barcode_plans = ", d.right_cached_barcode_plans,
          "\n  left_cached_landscape_plans = ", d.left_cached_landscape_plans,
          "\n  right_cached_landscape_plans = ", d.right_cached_landscape_plans)
end

function Base.show(io::IO, task::SliceBarcodesTask)
    d = _sliceinv_describe(task)
    print(io, "SliceBarcodesTask(packed=", d.packed,
          ", threads=", d.threads, ")")
end

function Base.show(io::IO, ::MIME"text/plain", task::SliceBarcodesTask)
    d = _sliceinv_describe(task)
    print(io, "SliceBarcodesTask",
          "\n  packed = ", d.packed,
          "\n  threads = ", d.threads)
end

function Base.show(io::IO, task::SliceDistanceTask)
    d = _sliceinv_describe(task)
    print(io, "SliceDistanceTask(dist=", d.distance_function,
          ", agg=", d.agg,
          ", threads=", d.threads, ")")
end

function Base.show(io::IO, ::MIME"text/plain", task::SliceDistanceTask)
    d = _sliceinv_describe(task)
    print(io, "SliceDistanceTask",
          "\n  distance_function = ", d.distance_function,
          "\n  weight_mode = ", d.weight_mode,
          "\n  agg = ", d.agg,
          "\n  agg_p = ", d.agg_p,
          "\n  agg_norm = ", d.agg_norm,
          "\n  threads = ", d.threads)
end

function Base.show(io::IO, task::SliceKernelTask)
    d = _sliceinv_describe(task)
    print(io, "SliceKernelTask(kind=", d.kernel_kind,
          ", sigma=", d.sigma,
          ", threads=", d.threads, ")")
end

function Base.show(io::IO, ::MIME"text/plain", task::SliceKernelTask)
    d = _sliceinv_describe(task)
    print(io, "SliceKernelTask",
          "\n  kernel_kind = ", d.kernel_kind,
          "\n  sigma = ", d.sigma,
          "\n  gamma = ", d.gamma,
          "\n  p = ", d.p,
          "\n  q = ", d.q,
          "\n  kmax = ", d.kmax,
          "\n  tgrid_nsteps = ", d.tgrid_nsteps,
          "\n  threads = ", d.threads)
end

"""
    landscape_summary(pl) -> NamedTuple
    persistence_image_summary(PI) -> NamedTuple
    slice_plan_summary(plan) -> NamedTuple
    slice_cache_summary(cache) -> NamedTuple
    slice_pair_cache_summary(pair) -> NamedTuple
    slice_task_summary(task) -> NamedTuple
    slice_collection_summary(slices) -> NamedTuple

Owner-local cheap-first summaries for the `SliceInvariants` subsystem.

Use these before asking for raw slice collections, packed barcode grids, or
feature arrays.

# Example

```julia
using TamerOp

# Assume `pi` is an encoding map and `M`, `N` are modules on its region poset.
opts = TamerOp.Advanced.InvariantOptions(box=([0.0], [3.0]))

plan = TamerOp.Advanced.compile_slice_plan(
    pi, opts; directions=[[1.0]], offsets=[[0.0], [1.0]], tmin=0.0, tmax=3.0, nsteps=33
)
cache = TamerOp.Advanced.module_cache(M)
paircache = TamerOp.Advanced.module_cache(M, N)

TamerOp.describe(plan)
TamerOp.Advanced.slice_plan_summary(plan)
TamerOp.Advanced.run_invariants(plan, cache, TamerOp.Advanced.SliceBarcodesTask())
TamerOp.Advanced.run_invariants(plan, paircache, TamerOp.Advanced.SliceDistanceTask())
TamerOp.Advanced.slice_features(M, plan; featurizer=:landscape)
```
"""
@inline landscape_summary(pl::PersistenceLandscape1D) = describe(pl)
@inline persistence_image_summary(PI::PersistenceImage1D) = describe(PI)
@inline slice_plan_summary(plan::CompiledSlicePlan) = describe(plan)
@inline slice_cache_summary(cache::SlicePlanCache) = describe(cache)
@inline slice_cache_summary(cache::SliceModuleCache) = describe(cache)
@inline slice_pair_cache_summary(pair::SliceModulePairCache) = describe(pair)
@inline slice_task_summary(task::Union{SliceBarcodesTask,SliceDistanceTask,SliceKernelTask}) = describe(task)

function slice_collection_summary(slices)
    specs = collect_slices(slices)
    isempty(specs) && return (;
        kind=:slice_collection,
        nslices=0,
        values_mode=nothing,
        total_weight=0.0,
        max_chain_length=0,
    )
    values_modes = unique(
        values(spec) === nothing ? :index :
        (eltype(values(spec)) <: Integer ? :integer : :real)
        for spec in specs
    )
    return (;
        kind=:slice_collection,
        nslices=length(specs),
        values_mode=length(values_modes) == 1 ? only(values_modes) : Tuple(values_modes),
        total_weight=sum(float(weight(spec)) for spec in specs),
        max_chain_length=maximum(length(chain(spec)) for spec in specs),
    )
end

"""
    check_persistence_landscape(pl; throw=false) -> NamedTuple

Validate the structural contract of a [`PersistenceLandscape1D`](@ref).

This is the preferred explicit validator for hand-built landscape objects.
"""
function check_persistence_landscape(pl::PersistenceLandscape1D; throw::Bool=false)
    issues = String[]
    tg = landscape_grid(pl)
    vals = landscape_values(pl)

    size(vals, 2) == length(tg) ||
        push!(issues, "landscape values has $(size(vals, 2)) columns, expected $(length(tg)) from tgrid.")
    all(isfinite, tg) || push!(issues, "landscape grid must be finite.")
    all(isfinite, vals) || push!(issues, "landscape values must be finite.")
    any(<(0.0), vals) && push!(issues, "landscape values must be nonnegative.")
    length(tg) >= 2 || push!(issues, "landscape grid must contain at least two points.")
    all(diff(tg) .> 0.0) || push!(issues, "landscape grid must be strictly increasing.")
    if size(vals, 1) >= 2
        @inbounds for k in 1:(size(vals, 1) - 1), j in axes(vals, 2)
            vals[k + 1, j] <= vals[k, j] + 1e-12 || begin
                push!(issues, "landscape layers must be pointwise nonincreasing in k.")
                break
            end
        end
    end

    valid = isempty(issues)
    throw && !valid && _throw_invalid_sliceinvariants(:check_persistence_landscape, issues)
    return _sliceinv_issue_report(:persistence_landscape_1d, valid;
                                  grid_length=length(tg),
                                  landscape_layers=landscape_layers(pl),
                                  feature_dimension=feature_dimension(pl),
                                  issues=issues)
end

"""
    check_persistence_image(PI; throw=false) -> NamedTuple

Validate the structural contract of a [`PersistenceImage1D`](@ref).

This is the preferred explicit validator for hand-built persistence-image
objects.
"""
function check_persistence_image(PI::PersistenceImage1D; throw::Bool=false)
    issues = String[]
    xg = image_xgrid(PI)
    yg = image_ygrid(PI)
    vals = image_values(PI)

    size(vals) == (length(yg), length(xg)) ||
        push!(issues, "image values has size $(size(vals)), expected ($(length(yg)), $(length(xg))) from the grids.")
    all(isfinite, xg) || push!(issues, "xgrid must be finite.")
    all(isfinite, yg) || push!(issues, "ygrid must be finite.")
    all(isfinite, vals) || push!(issues, "image values must be finite.")
    any(<(0.0), vals) && push!(issues, "image values must be nonnegative.")
    length(xg) >= 1 || push!(issues, "xgrid must contain at least one point.")
    length(yg) >= 1 || push!(issues, "ygrid must contain at least one point.")
    length(xg) >= 2 && !all(diff(xg) .> 0.0) && push!(issues, "xgrid must be strictly increasing.")
    length(yg) >= 2 && !all(diff(yg) .> 0.0) && push!(issues, "ygrid must be strictly increasing.")

    valid = isempty(issues)
    throw && !valid && _throw_invalid_sliceinvariants(:check_persistence_image, issues)
    return _sliceinv_issue_report(:persistence_image_1d, valid;
                                  image_shape=image_shape(PI),
                                  feature_dimension=length(vals),
                                  issues=issues)
end

"""
    check_compiled_slice_plan(plan; throw=false) -> NamedTuple

Validate the structural contract of a compiled slice plan.

This is the preferred explicit validator before debugging malformed hand-built
plans or caches in notebooks.
"""
function check_compiled_slice_plan(plan::CompiledSlicePlan; throw::Bool=false)
    issues = String[]
    ns = nslices(plan)
    ambient = _sliceinv_plan_ambient_dim(plan)

    size(plan.weights) == (plan.nd, plan.no) ||
        push!(issues, "weights has size $(size(plan.weights)), expected ($(plan.nd), $(plan.no)).")
    length(plan.dirs) == plan.nd ||
        push!(issues, "expected $(plan.nd) directions, found $(length(plan.dirs)).")
    length(plan.offs) == plan.no ||
        push!(issues, "expected $(plan.no) offsets, found $(length(plan.offs)).")
    length(plan.chain_start) == ns ||
        push!(issues, "chain_start has length $(length(plan.chain_start)), expected $ns.")
    length(plan.chain_len) == ns ||
        push!(issues, "chain_len has length $(length(plan.chain_len)), expected $ns.")
    length(plan.vals_start) == ns ||
        push!(issues, "vals_start has length $(length(plan.vals_start)), expected $ns.")
    length(plan.vals_len) == ns ||
        push!(issues, "vals_len has length $(length(plan.vals_len)), expected $ns.")
    length(plan.chains) == ns ||
        push!(issues, "pooled chain rows has length $(length(plan.chains)), expected $ns.")

    @inbounds for (i, dir) in pairs(plan.dirs)
        length(dir) == ambient || push!(issues, "direction $i has length $(length(dir)), expected $ambient.")
        all(isfinite, dir) || push!(issues, "direction $i contains non-finite entries.")
    end
    @inbounds for (j, off) in pairs(plan.offs)
        length(off) == ambient || push!(issues, "offset $j has length $(length(off)), expected $ambient.")
        all(isfinite, off) || push!(issues, "offset $j contains non-finite entries.")
    end

    all(isfinite, plan.weights) || push!(issues, "weights must be finite.")
    any(<(0), plan.weights) && push!(issues, "weights must be nonnegative.")

    chain_pool_len = length(plan.chain_pool)
    vals_pool_len = length(plan.vals_pool)
    @inbounds for idx in eachindex(plan.chain_len)
        lc = plan.chain_len[idx]
        sc = plan.chain_start[idx]
        lc >= 0 || push!(issues, "chain_len[$idx] is negative.")
        if lc == 0
            sc == 0 || push!(issues, "chain_start[$idx] must be 0 when chain_len[$idx] == 0.")
        else
            1 <= sc <= chain_pool_len || push!(issues, "chain_start[$idx] is out of bounds.")
            sc + lc - 1 <= chain_pool_len || push!(issues, "chain row $idx exceeds chain_pool bounds.")
        end

        lv = plan.vals_len[idx]
        sv = plan.vals_start[idx]
        lv >= 0 || push!(issues, "vals_len[$idx] is negative.")
        if lv == 0
            sv == 0 || push!(issues, "vals_start[$idx] must be 0 when vals_len[$idx] == 0.")
        else
            1 <= sv <= vals_pool_len || push!(issues, "vals_start[$idx] is out of bounds.")
            sv + lv - 1 <= vals_pool_len || push!(issues, "value row $idx exceeds vals_pool bounds.")
            lv == lc || push!(issues, "value row $idx has length $lv, expected chain length $lc.")
        end
    end

    valid = isempty(issues)
    throw && !valid && _throw_invalid_sliceinvariants(:check_compiled_slice_plan, issues)
    return _sliceinv_issue_report(:compiled_slice_plan, valid;
                                  ambient_dim=ambient,
                                  ndirections=plan.nd,
                                  noffsets=plan.no,
                                  nslices=ns,
                                  total_weight=total_weight(plan),
                                  has_values=plan_has_values(plan),
                                  value_mode=plan_value_mode(plan),
                                  issues=issues)
end

"""
    check_slice_plan_cache(cache; throw=false) -> NamedTuple

Validate a [`SlicePlanCache`](@ref) by checking each cached compiled plan.
"""
function check_slice_plan_cache(cache::SlicePlanCache; throw::Bool=false)
    Base.lock(cache.lock)
    try
        reports = [check_compiled_slice_plan(plan; throw=false) for plan in values(cache.plans)]
        issues = String[]
        for (idx, report) in pairs(reports)
            report.valid || push!(issues, "cached plan $idx is invalid.")
        end
        valid = isempty(issues)
        throw && !valid && _throw_invalid_sliceinvariants(:check_slice_plan_cache, issues)
        return _sliceinv_issue_report(:slice_plan_cache, valid;
                                      cached_plans=length(reports),
                                      invalid_plans=count(report -> !report.valid, reports),
                                      issues=issues)
    finally
        Base.unlock(cache.lock)
    end
end

"""
    check_slice_module_cache(cache; throw=false) -> NamedTuple

Validate a module-side slice cache.
"""
function check_slice_module_cache(cache::SliceModuleCache; throw::Bool=false)
    issues = String[]
    M = source_module(cache)
    M isa PModule || push!(issues, "cache source_module is not a PModule.")
    valid = isempty(issues)
    throw && !valid && _throw_invalid_sliceinvariants(:check_slice_module_cache, issues)
    return _sliceinv_issue_report(:slice_module_cache, valid;
                                  poset_size=nvertices(M.Q),
                                  cached_barcode_plans=cached_barcode_plan_count(cache),
                                  cached_landscape_plans=cached_landscape_plan_count(cache),
                                  issues=issues)
end

"""
    check_slice_module_pair_cache(pair; throw=false) -> NamedTuple

Validate a paired module cache used by sliced distances and kernels.
"""
function check_slice_module_pair_cache(pair::SliceModulePairCache; throw::Bool=false)
    left_report = check_slice_module_cache(left_cache(pair); throw=false)
    right_report = check_slice_module_cache(right_cache(pair); throw=false)
    issues = String[]
    left_report.valid || push!(issues, "left cache is invalid.")
    right_report.valid || push!(issues, "right cache is invalid.")
    valid = isempty(issues)
    throw && !valid && _throw_invalid_sliceinvariants(:check_slice_module_pair_cache, issues)
    return _sliceinv_issue_report(:slice_module_pair_cache, valid;
                                  left_valid=left_report.valid,
                                  right_valid=right_report.valid,
                                  left_cached_barcode_plans=cached_barcode_plan_count(left_cache(pair)),
                                  right_cached_barcode_plans=cached_barcode_plan_count(right_cache(pair)),
                                  issues=issues)
end

"""
    check_slice_barcodes_task(task; throw=false) -> NamedTuple

Validate a [`SliceBarcodesTask`](@ref).
"""
function check_slice_barcodes_task(task::SliceBarcodesTask; throw::Bool=false)
    issues = String[]
    valid = isempty(issues)
    throw && !valid && _throw_invalid_sliceinvariants(:check_slice_barcodes_task, issues)
    return _sliceinv_issue_report(:slice_barcodes_task, valid;
                                  packed=task.packed,
                                  threads=task.threads,
                                  issues=issues)
end

"""
    check_slice_distance_task(task; throw=false) -> NamedTuple

Validate a [`SliceDistanceTask`](@ref).
"""
function check_slice_distance_task(task::SliceDistanceTask; throw::Bool=false)
    issues = String[]
    agg_mode = task.agg === mean ? :mean : task.agg === maximum ? :max : task.agg
    task.weight_mode in (:integrate, :scale) ||
        push!(issues, "weight_mode must be :integrate or :scale.")
    (agg_mode in (:mean, :pmean, :max) || agg_mode isa Function) ||
        push!(issues, "agg must be :mean, :pmean, :max, mean, maximum, or a callable.")
    (task.agg_p > 0 || isinf(task.agg_p)) ||
        push!(issues, "agg_p must be positive or Inf.")
    (isfinite(task.agg_norm) && task.agg_norm > 0) ||
        push!(issues, "agg_norm must be finite and positive.")
    valid = isempty(issues)
    throw && !valid && _throw_invalid_sliceinvariants(:check_slice_distance_task, issues)
    return _sliceinv_issue_report(:slice_distance_task, valid;
                                  distance_function=_sliceinv_callable_label(task.dist_fn),
                                  weight_mode=task.weight_mode,
                                  agg=agg_mode,
                                  agg_p=task.agg_p,
                                  agg_norm=task.agg_norm,
                                  issues=issues)
end

"""
    check_slice_kernel_task(task; throw=false) -> NamedTuple

Validate a [`SliceKernelTask`](@ref).
"""
function check_slice_kernel_task(task::SliceKernelTask; throw::Bool=false)
    issues = String[]
    allowed_kind = task.kind isa Function ||
        task.kind in (:bottleneck_gaussian, :bottleneck_laplacian,
                      :wasserstein_gaussian, :wasserstein_laplacian,
                      :landscape_linear, :landscape_gaussian, :landscape_laplacian)
    allowed_kind || push!(issues, "kernel kind must be one of the documented symbols or a callable.")
    (isfinite(task.sigma) && task.sigma > 0) || push!(issues, "sigma must be finite and positive.")
    (task.gamma === nothing || isfinite(float(task.gamma))) || push!(issues, "gamma must be finite when provided.")
    (isfinite(task.p) && task.p > 0) || push!(issues, "p must be finite and positive.")
    ((isfinite(task.q) && task.q > 0) || isinf(task.q)) || push!(issues, "q must be positive or Inf.")
    task.kmax >= 1 || push!(issues, "kmax must be >= 1.")
    task.tgrid_nsteps >= 2 || push!(issues, "tgrid_nsteps must be >= 2.")
    if task.tgrid !== nothing
        try
            _clean_tgrid(task.tgrid)
        catch err
            push!(issues, "tgrid is invalid: $(err)")
        end
    end
    valid = isempty(issues)
    throw && !valid && _throw_invalid_sliceinvariants(:check_slice_kernel_task, issues)
    return _sliceinv_issue_report(:slice_kernel_task, valid;
                                  kernel_kind=task.kind,
                                  sigma=task.sigma,
                                  p=task.p,
                                  q=task.q,
                                  kmax=task.kmax,
                                  tgrid_nsteps=task.tgrid_nsteps,
                                  issues=issues)
end

"""
    check_slice_direction(pi, dir; throw=false) -> NamedTuple

Validate one slice direction against the ambient dimension of an encoding map.
"""
function check_slice_direction(pi_or_enc, dir; throw::Bool=false)
    pi = _unwrap_compiled(pi_or_enc)
    issues = String[]
    ambient = pi isa PLikeEncodingMap ? dimension(pi) : nothing
    dir_kind = _sliceinv_point_kind(dir)
    len = _sliceinv_point_length(dir)
    pi isa PLikeEncodingMap || push!(issues, "expected PLikeEncodingMap or CompiledEncoding{<:PLikeEncodingMap}.")
    dir_kind === :invalid && push!(issues, "direction must be a real tuple or real vector.")
    ambient !== nothing && len !== nothing && len != ambient &&
        push!(issues, "direction has length $len, expected ambient dimension $ambient.")
    if dir_kind !== :invalid && (dir isa AbstractVector{<:Real} || dir isa Tuple)
        all(isfinite, dir) || push!(issues, "direction entries must be finite.")
        try
            _normalize_dir(dir, :none)
        catch err
            push!(issues, sprint(showerror, err))
        end
    end
    valid = isempty(issues)
    throw && !valid && _throw_invalid_sliceinvariants(:check_slice_direction, issues)
    return _sliceinv_issue_report(:slice_direction, valid;
                                  ambient_dim=ambient,
                                  query_kind=dir_kind,
                                  direction_length=len,
                                  issues=issues)
end

"""
    check_slice_basepoint(pi, x0; throw=false) -> NamedTuple

Validate one slice basepoint against the ambient dimension of an encoding map.
"""
function check_slice_basepoint(pi_or_enc, x0; throw::Bool=false)
    pi = _unwrap_compiled(pi_or_enc)
    issues = String[]
    ambient = pi isa PLikeEncodingMap ? dimension(pi) : nothing
    x_kind = _sliceinv_point_kind(x0)
    len = _sliceinv_point_length(x0)
    pi isa PLikeEncodingMap || push!(issues, "expected PLikeEncodingMap or CompiledEncoding{<:PLikeEncodingMap}.")
    x_kind === :invalid && push!(issues, "basepoint must be a real tuple or real vector.")
    ambient !== nothing && len !== nothing && len != ambient &&
        push!(issues, "basepoint has length $len, expected ambient dimension $ambient.")
    if x_kind !== :invalid && (x0 isa AbstractVector{<:Real} || x0 isa Tuple)
        all(isfinite, x0) || push!(issues, "basepoint entries must be finite.")
    end
    valid = isempty(issues)
    throw && !valid && _throw_invalid_sliceinvariants(:check_slice_basepoint, issues)
    return _sliceinv_issue_report(:slice_basepoint, valid;
                                  ambient_dim=ambient,
                                  query_kind=x_kind,
                                  basepoint_length=len,
                                  issues=issues)
end

"""
    check_slice_request(pi, x0, dir, opts; throw=false) -> NamedTuple

Validate the cheap request contract for direct slice queries.
"""
function check_slice_request(pi_or_enc, x0, dir, opts=InvariantOptions(); throw::Bool=false)
    pi = _unwrap_compiled(pi_or_enc)
    issues = String[]
    base_report = check_slice_basepoint(pi_or_enc, x0; throw=false)
    dir_report = check_slice_direction(pi_or_enc, dir; throw=false)
    !base_report.valid && append!(issues, String.(base_report.issues))
    !dir_report.valid && append!(issues, String.(dir_report.issues))

    box_kind = nothing
    box_report = nothing
    if !(opts isa InvariantOptions)
        push!(issues, "opts must be an InvariantOptions value.")
    else
        box_kind = opts.box === nothing ? :none : opts.box === :auto ? :auto : :explicit
        if opts.box === :auto
            if pi isa PLikeEncodingMap
                try
                    box_report = _normalize_box(window_box(pi))
                catch err
                    push!(issues, "automatic window_box inference failed: $(sprint(showerror, err))")
                end
            end
        elseif opts.box !== nothing
            try
                lo, hi = _normalize_box(opts.box)
                box_report = (lo, hi)
                ambient = pi isa PLikeEncodingMap ? dimension(pi) : nothing
                ambient !== nothing && length(lo) != ambient &&
                    push!(issues, "box has dimension $(length(lo)), expected ambient dimension $ambient.")
            catch err
                push!(issues, "box is invalid: $(sprint(showerror, err))")
            end
        end
    end

    valid = isempty(issues)
    throw && !valid && _throw_invalid_sliceinvariants(:check_slice_request, issues)
    return _sliceinv_issue_report(:slice_request, valid;
                                  ambient_dim=(pi isa PLikeEncodingMap ? dimension(pi) : nothing),
                                  basepoint_valid=base_report.valid,
                                  direction_valid=dir_report.valid,
                                  box_kind=box_kind,
                                  box=box_report,
                                  issues=issues)
end

"""
    check_slice_specs(slices; throw=false) -> NamedTuple

Validate an explicit slice collection or anything accepted by
[`collect_slices`](@ref).
"""
function check_slice_specs(slices; throw::Bool=false)
    issues = String[]
    specs = SliceSpec[]
    try
        specs = collect_slices(slices)
    catch err
        push!(issues, sprint(showerror, err))
    end
    reports = NamedTuple[]
    if isempty(issues)
        reports = [check_slice_spec(spec; throw=false) for spec in specs]
        for (idx, report) in pairs(reports)
            report.valid || append!(issues, ["slice $idx: " * issue for issue in report.issues])
            ch = chain(specs[idx])
            issorted(ch) || push!(issues, "slice $idx chain is not nondecreasing.")
            allunique(ch) || push!(issues, "slice $idx chain contains repeated region ids.")
        end
    end
    value_modes = isempty(specs) ? () : Tuple(unique(values(spec) === nothing ? :index :
        (eltype(values(spec)) <: Integer ? :integer : :real) for spec in specs))
    valid = isempty(issues)
    throw && !valid && _throw_invalid_sliceinvariants(:check_slice_specs, issues)
    return _sliceinv_issue_report(:slice_specs, valid;
                                  nslices=length(specs),
                                  values_mode=length(value_modes) == 1 ? only(value_modes) : value_modes,
                                  total_weight=sum(float(weight(spec)) for spec in specs),
                                  max_chain_length=isempty(specs) ? 0 : maximum(length(chain(spec)) for spec in specs),
                                  issues=issues)
end

@inline _plan_idx(no::Int, i::Int, j::Int) = (i - 1) * no + j

"""
    collect_slices(plan::CompiledSlicePlan; values=:t) -> Vector{SliceSpec}

Convert a compiled slice plan into an explicit typed slice list, useful for
serialization or passing precompiled slices through other APIs.
"""
function collect_slices(plan::CompiledSlicePlan; values::Symbol=:t)
    ns = plan.nd * plan.no
    if values == :index
        specs = Vector{SliceSpec{Float64,Nothing}}(undef, ns)
        @inbounds for i in 1:plan.nd, j in 1:plan.no
            idx = _plan_idx(plan.no, i, j)
            specs[idx] = SliceSpec{Float64,Nothing}(Vector{Int}(_plan_chain(plan, idx)), nothing, plan.weights[i, j])
        end
        return specs
    elseif values == :t
        specs = Vector{SliceSpec{Float64,Vector{Float64}}}(undef, ns)
        @inbounds for i in 1:plan.nd, j in 1:plan.no
            idx = _plan_idx(plan.no, i, j)
            specs[idx] = SliceSpec{Float64,Vector{Float64}}(
                Vector{Int}(_plan_chain(plan, idx)),
                Vector{Float64}(_plan_vals(plan, idx)),
                plan.weights[i, j],
            )
        end
        return specs
    end
    throw(ArgumentError("collect_slices(plan): values must be :t or :index"))
end


function _plan_cache_key(
    pi,
    directions,
    offsets,
    normalize_dirs::Symbol,
    n_dirs::Integer,
    n_offsets::Integer,
    max_den::Integer,
    include_axes::Bool,
    offset_margin::Real,
    drop_unknown::Bool,
    strict_kw,
    box_kw,
    direction_weight,
    offset_weights,
    normalize_weights::Bool,
    filtered::NamedTuple,
)
    strict_code = strict_kw === nothing ? Int8(-1) : (Bool(strict_kw) ? Int8(1) : Int8(0))
    return SlicePlanCacheKey(
        UInt(objectid(pi)),
        normalize_dirs,
        Int(n_dirs),
        Int(n_offsets),
        Int(max_den),
        include_axes,
        Float64(offset_margin),
        drop_unknown,
        strict_code,
        UInt(hash(box_kw)),
        UInt(hash(directions)),
        UInt(hash(offsets)),
        UInt(hash((direction_weight, offset_weights, normalize_weights))),
        UInt(hash(filtered)),
    )
end

"""
    compile_slice_plan(pi::PLikeEncodingMap; ...) -> CompiledSlicePlan

Precompute `(chain, values)` for each sampled `(direction, offset)` slice once, then
reuse with `slice_barcodes(M, plan; packed=true)` across many modules.
"""
function compile_slice_plan(
    pi::PLikeEncodingMap,
    opts::InvariantOptions=InvariantOptions();
    directions = :auto,
    offsets = :auto,
    n_dirs::Integer = 16,
    n_offsets::Integer = 9,
    max_den::Integer = 8,
    include_axes::Bool = false,
    normalize_dirs::Symbol = :none,
    direction_weight::Union{Symbol,Function,Real} = :none,
    offset_weights = nothing,
    normalize_weights::Bool = true,
    offset_margin::Real = 0.05,
    drop_unknown::Bool = true,
    threads::Bool = (Threads.nthreads() > 1),
    cache::Union{Nothing,SlicePlanCache} = nothing,
    slice_kwargs...
)
    dirs0 = directions
    offs0 = offsets

    if dirs0 === :auto || dirs0 === nothing
        dirs0 = default_directions(pi;
                                   n_dirs = n_dirs,
                                   max_den = max_den,
                                   include_axes = include_axes,
                                   normalize = (normalize_dirs == :none ? :none : normalize_dirs))
    end

    haskey(slice_kwargs, :box) &&
        throw(ArgumentError("compile_slice_plan: pass box via opts::InvariantOptions, not keyword :box"))
    haskey(slice_kwargs, :strict) &&
        throw(ArgumentError("compile_slice_plan: pass strict via opts::InvariantOptions, not keyword :strict"))

    box_kw = opts.box
    strict_kw = opts.strict

    opts_offsets = InvariantOptions(box = box_kw, pl_mode = opts.pl_mode)
    opts_chain   = InvariantOptions(box = box_kw, strict = strict_kw, pl_mode = opts.pl_mode)
    if opts_chain.box === nothing && !haskey(slice_kwargs, :tmin) && !haskey(slice_kwargs, :tmax)
        opts_offsets = InvariantOptions(box = :auto, pl_mode = opts.pl_mode)
        opts_chain   = InvariantOptions(box = :auto, strict = strict_kw, pl_mode = opts.pl_mode)
    end

    filtered = (;
        (k => v for (k, v) in pairs(slice_kwargs)
            if k != :default_weight && k != :kmin && k != :kmax_param && k != :lengthscale)...)

    key = cache === nothing ? nothing : _plan_cache_key(
        pi, directions, offsets, normalize_dirs, n_dirs, n_offsets, max_den,
        include_axes, offset_margin, drop_unknown, strict_kw, box_kw,
        direction_weight, offset_weights, normalize_weights, filtered,
    )

    if cache !== nothing
        Base.lock(cache.lock)
        try
            cached = get(cache.plans, key, nothing)
            cached === nothing || return cached
        finally
            Base.unlock(cache.lock)
        end
    end

    if offs0 === :auto || offs0 === nothing
        offs0 = default_offsets(pi, opts_offsets; n_offsets = n_offsets, margin = offset_margin)
    end

    if !isempty(offs0) && length(offs0[1]) == 0
        empty_plan = _compiled_slice_plan(
            Vector{Vector{Float64}}(),
            Vector{Vector{Float64}}(),
            zeros(Float64, 0, 0),
            Int[],
            Int[],
            Int[],
            Float64[],
            Int[],
            Int[],
            0,
            0,
        )
        if cache !== nothing
            Base.lock(cache.lock)
            try
                cache.plans[key] = empty_plan
            finally
                Base.unlock(cache.lock)
            end
        end
        return empty_plan
    end

    offs_vec = Vector{Vector{Float64}}(undef, length(offs0))
    @inbounds for j in eachindex(offs0)
        x0 = offs0[j]
        if x0 isa AbstractVector
            offs_vec[j] = Float64[float(v) for v in x0]
        elseif x0 isa Tuple
            offs_vec[j] = Float64[float(v) for v in x0]
        else
            throw(ArgumentError("compile_slice_plan: expected offset basepoints as vectors/tuples, got $(typeof(x0))."))
        end
    end

    if !isempty(dirs0) && !isempty(offs_vec) && length(dirs0[1]) != length(offs_vec[1])
        dirs0 = default_directions(length(offs_vec[1]);
                                   n_dirs = n_dirs,
                                   max_den = max_den,
                                   include_axes = include_axes,
                                   normalize = (normalize_dirs == :none ? :none : normalize_dirs))
    end

    dirs_in = [_normalize_dir(dir, normalize_dirs) for dir in dirs0]
    dirs_vec = Vector{Vector{Float64}}(undef, length(dirs_in))
    @inbounds for i in eachindex(dirs_in)
        dirs_vec[i] = Float64[dirs_in[i][k] for k in eachindex(dirs_in[i])]
    end

    nd = length(dirs_vec)
    no = length(offs_vec)
    nd > 0 || error("compile_slice_plan: directions is empty")
    no > 0 || error("compile_slice_plan: offsets is empty")

    wdir = Vector{Float64}(undef, nd)
    @inbounds for i in 1:nd
        wdir[i] = SliceInvariants.direction_weight(dirs_vec[i], direction_weight)
    end
    woff = _offset_sample_weights(offs_vec, offset_weights)

    W = wdir * woff'
    if normalize_weights
        s = sum(W)
        s > 0 || error("compile_slice_plan: total slice weight is zero")
        W ./= s
    end

    ns = nd * no
    chain_len = zeros(Int, ns)
    vals_len = zeros(Int, ns)
    if threads && Threads.nthreads() > 1
        locate_ws_by_thread = [_SliceLocateBatchWorkspace() for _ in 1:Threads.nthreads()]
        Threads.@threads for idx in 1:ns
            i = div(idx - 1, no) + 1
            j = (idx - 1) % no + 1
            n = _slice_chain_count(
                pi,
                offs_vec[j],
                dirs_vec[i],
                opts_chain;
                drop_unknown = drop_unknown,
                locate_ws = locate_ws_by_thread[Threads.threadid()],
                filtered...,
            )
            chain_len[idx] = n
            vals_len[idx] = n
        end
    else
        locate_ws = _SliceLocateBatchWorkspace()
        @inbounds for idx in 1:ns
            i = div(idx - 1, no) + 1
            j = (idx - 1) % no + 1
            n = _slice_chain_count(
                pi,
                offs_vec[j],
                dirs_vec[i],
                opts_chain;
                drop_unknown = drop_unknown,
                locate_ws = locate_ws,
                filtered...,
            )
            chain_len[idx] = n
            vals_len[idx] = n
        end
    end

    chain_start = zeros(Int, ns)
    vals_start = zeros(Int, ns)
    total_chain = 0
    total_vals = 0
    @inbounds for idx in 1:ns
        lc = chain_len[idx]
        lv = vals_len[idx]
        if lc > 0
            chain_start[idx] = total_chain + 1
            total_chain += lc
        end
        if lv > 0
            vals_start[idx] = total_vals + 1
            total_vals += lv
        end
    end

    chain_pool = Vector{Int}(undef, total_chain)
    vals_pool = Vector{Float64}(undef, total_vals)

    if threads && Threads.nthreads() > 1
        locate_ws_by_thread = [_SliceLocateBatchWorkspace() for _ in 1:Threads.nthreads()]
        Threads.@threads for idx in 1:ns
            chain_len[idx] == 0 && continue
            i = div(idx - 1, no) + 1
            j = (idx - 1) % no + 1
            _slice_chain_fill!(
                chain_pool,
                chain_start[idx],
                vals_pool,
                vals_start[idx],
                pi,
                offs_vec[j],
                dirs_vec[i],
                opts_chain;
                drop_unknown = drop_unknown,
                locate_ws = locate_ws_by_thread[Threads.threadid()],
                filtered...,
            )
        end
    else
        locate_ws = _SliceLocateBatchWorkspace()
        @inbounds for i in 1:nd, j in 1:no
            idx = _plan_idx(no, i, j)
            chain_len[idx] == 0 && continue
            _slice_chain_fill!(
                chain_pool,
                chain_start[idx],
                vals_pool,
                vals_start[idx],
                pi,
                offs_vec[j],
                dirs_vec[i],
                opts_chain;
                drop_unknown = drop_unknown,
                locate_ws = locate_ws,
                filtered...,
            )
        end
    end

    plan = _compiled_slice_plan(
        dirs_vec,
        offs_vec,
        W,
        chain_pool,
        chain_start,
        chain_len,
        vals_pool,
        vals_start,
        vals_len,
        nd,
        no,
    )

    if cache !== nothing
        Base.lock(cache.lock)
        try
            existing = get(cache.plans, key, nothing)
            if existing === nothing
                cache.plans[key] = plan
            else
                plan = existing
            end
        finally
            Base.unlock(cache.lock)
        end
    end

    return plan
end

"""
    compile_slices(pi, opts::InvariantOptions=InvariantOptions(); kwargs...) -> CompiledSlicePlan

Public phase-1 compile entrypoint. This is a thin adapter over `compile_slice_plan`
that resolves `box`/`strict`/`threads` through `InvariantOptions`.
"""
function compile_slices(
    pi::PLikeEncodingMap,
    opts::InvariantOptions=InvariantOptions();
    kwargs...
)
    kwargs_nt = NamedTuple(kwargs)
    haskey(kwargs_nt, :box) &&
        throw(ArgumentError("compile_slices: pass box via opts::InvariantOptions, not keyword :box"))
    haskey(kwargs_nt, :strict) &&
        throw(ArgumentError("compile_slices: pass strict via opts::InvariantOptions, not keyword :strict"))
    kwargs2 = (; (k => v for (k, v) in pairs(kwargs_nt) if k != :threads)...)
    return compile_slice_plan(pi, opts;
                              threads = _default_threads(get(kwargs_nt, :threads, opts.threads)),
                              kwargs2...)
end

compile_slices(pi::CompiledEncoding{<:PLikeEncodingMap}, opts::InvariantOptions=InvariantOptions(); kwargs...) =
    compile_slices(pi.pi, opts; kwargs...)

function slice_barcodes(
    M::PModule{K},
    plan::CompiledSlicePlan;
    packed::Bool = false,
    threads::Bool = (Threads.nthreads() > 1),
) where {K}
    bars = _slice_barcodes_plan_packed_cached(module_cache(M), plan; threads=threads)
    return _slice_barcodes_plan_result_from_packed(bars, plan; packed=packed)
end

@inline function run_invariants(plan::CompiledSlicePlan, cache::SliceModuleCache, task::SliceBarcodesTask)
    bars = _slice_barcodes_plan_packed_cached(cache, plan; threads=task.threads)
    return _slice_barcodes_plan_result_from_packed(bars, plan; packed=task.packed)
end

@inline run_invariants(plan::CompiledSlicePlan, M::PModule, task::SliceBarcodesTask) =
    run_invariants(plan, module_cache(M), task)

@inline function _slice_distance_fast_kind(task::SliceDistanceTask)
    dist_fn = task.dist_fn
    if dist_fn === bottleneck_distance || dist_fn === matching_distance
        return :bottleneck
    elseif dist_fn === wasserstein_distance || dist_fn === matching_wasserstein_distance
        return :wasserstein
    end
    return :none
end

@inline function _packed_distance_value!(
    scratch::_SliceKernelScratch,
    bcM::PackedFloatBarcode,
    bcN::PackedFloatBarcode,
    kind::Symbol,
    task::SliceDistanceTask,
)::Float64
    _points_from_packed!(scratch.points_a, bcM)
    _points_from_packed!(scratch.points_b, bcN)
    if kind === :bottleneck
        return _bottleneck_distance_points(
            scratch.points_a,
            scratch.points_b;
            backend = get(task.dist_kwargs, :backend, :auto),
        )
    elseif kind === :wasserstein
        return _wasserstein_distance_points(
            scratch.points_a,
            scratch.points_b;
            p = get(task.dist_kwargs, :p, 2),
            q = get(task.dist_kwargs, :q, Inf),
            backend = get(task.dist_kwargs, :backend, :auto),
        )
    end
    error("_packed_distance_value!: unsupported distance kind=$kind")
end

function _run_slice_distance_from_packed_barcodes(
    bcsM::PackedBarcodeGrid{PackedFloatBarcode},
    bcsN::PackedBarcodeGrid{PackedFloatBarcode},
    W::AbstractMatrix{Float64},
    task::SliceDistanceTask,
)::Float64
    agg_mode = (task.agg === mean) ? :mean : (task.agg === maximum) ? :max : task.agg
    agg_mode in (:mean, :pmean, :max) || return _run_slice_distance_from_barcodes_generic(bcsM, bcsN, W, task)

    dist_kind = _slice_distance_fast_kind(task)
    dist_kind === :none && return _run_slice_distance_from_barcodes_generic(bcsM, bcsN, W, task)

    threads = task.threads && Threads.nthreads() > 1
    scratch_by_thread = _scratch_arenas(threads)
    sumw = sum(W)
    sumw == 0.0 && return 0.0

    if task.weight_mode == :scale
        if threads
            nT = Threads.nthreads()
            best_by_slot = fill(0.0, nT)
            Threads.@threads for slot in 1:nT
                scratch = scratch_by_thread[Threads.threadid()]
                best = 0.0
                for idx in slot:nT:length(bcsM)
                    w = W[idx]
                    w == 0.0 && continue
                    d = _packed_distance_value!(scratch, bcsM[idx], bcsN[idx], dist_kind, task)
                    best = max(best, w * d)
                end
                best_by_slot[slot] = best
            end
            return maximum(best_by_slot) / float(task.agg_norm)
        end

        scratch = scratch_by_thread[1]
        best = 0.0
        @inbounds for idx in eachindex(bcsM)
            w = W[idx]
            w == 0.0 && continue
            d = _packed_distance_value!(scratch, bcsM[idx], bcsN[idx], dist_kind, task)
            best = max(best, w * d)
        end
        return best / float(task.agg_norm)
    elseif task.weight_mode == :integrate
        if agg_mode == :mean
            if threads
                nT = Threads.nthreads()
                acc_by_slot = fill(0.0, nT)
                Threads.@threads for slot in 1:nT
                    scratch = scratch_by_thread[Threads.threadid()]
                    acc = 0.0
                    for idx in slot:nT:length(bcsM)
                        w = W[idx]
                        w == 0.0 && continue
                        acc += w * _packed_distance_value!(scratch, bcsM[idx], bcsN[idx], dist_kind, task)
                    end
                    acc_by_slot[slot] = acc
                end
                return (sum(acc_by_slot) / sumw) / float(task.agg_norm)
            end

            scratch = scratch_by_thread[1]
            acc = 0.0
            @inbounds for idx in eachindex(bcsM)
                w = W[idx]
                w == 0.0 && continue
                acc += w * _packed_distance_value!(scratch, bcsM[idx], bcsN[idx], dist_kind, task)
            end
            return (acc / sumw) / float(task.agg_norm)
        elseif agg_mode == :pmean
            p = float(task.agg_p)
            if threads
                nT = Threads.nthreads()
                acc_by_slot = fill(0.0, nT)
                Threads.@threads for slot in 1:nT
                    scratch = scratch_by_thread[Threads.threadid()]
                    acc = 0.0
                    for idx in slot:nT:length(bcsM)
                        w = W[idx]
                        w == 0.0 && continue
                        d = _packed_distance_value!(scratch, bcsM[idx], bcsN[idx], dist_kind, task)
                        acc += w * d^p
                    end
                    acc_by_slot[slot] = acc
                end
                return ((sum(acc_by_slot) / sumw)^(1 / p)) / float(task.agg_norm)
            end

            scratch = scratch_by_thread[1]
            acc = 0.0
            @inbounds for idx in eachindex(bcsM)
                w = W[idx]
                w == 0.0 && continue
                d = _packed_distance_value!(scratch, bcsM[idx], bcsN[idx], dist_kind, task)
                acc += w * d^p
            end
            return ((acc / sumw)^(1 / p)) / float(task.agg_norm)
        elseif agg_mode == :max
            if threads
                nT = Threads.nthreads()
                best_by_slot = fill(0.0, nT)
                Threads.@threads for slot in 1:nT
                    scratch = scratch_by_thread[Threads.threadid()]
                    best = 0.0
                    for idx in slot:nT:length(bcsM)
                        w = W[idx]
                        w == 0.0 && continue
                        d = _packed_distance_value!(scratch, bcsM[idx], bcsN[idx], dist_kind, task)
                        best = max(best, w * d)
                    end
                    best_by_slot[slot] = best
                end
                return maximum(best_by_slot) / float(task.agg_norm)
            end

            scratch = scratch_by_thread[1]
            best = 0.0
            @inbounds for idx in eachindex(bcsM)
                w = W[idx]
                w == 0.0 && continue
                d = _packed_distance_value!(scratch, bcsM[idx], bcsN[idx], dist_kind, task)
                best = max(best, w * d)
            end
            return best / float(task.agg_norm)
        end
    end

    return _run_slice_distance_from_barcodes_generic(bcsM, bcsN, W, task)
end

function _run_slice_distance_from_barcodes_generic(
    bcsM,
    bcsN,
    W::AbstractMatrix{Float64},
    task::SliceDistanceTask,
)::Float64
    sumw = sum(W)
    if sumw == 0.0
        return 0.0
    end

    agg_mode = (task.agg === mean) ? :mean : (task.agg === maximum) ? :max : task.agg
    dist_fn = task.dist_fn
    dist_kwargs = task.dist_kwargs
    threads = task.threads

    if task.weight_mode == :scale
        if threads && Threads.nthreads() > 1
            nT = Threads.nthreads()
            best_by_slot = fill(0.0, nT)
            Threads.@threads for slot in 1:nT
                best = 0.0
                for idx in slot:nT:length(bcsM)
                    w = W[idx]
                    w == 0.0 && continue
                    d = dist_fn(bcsM[idx], bcsN[idx]; dist_kwargs...)
                    best = max(best, w * d)
                end
                best_by_slot[slot] = best
            end
            return maximum(best_by_slot) / float(task.agg_norm)
        end
        best = 0.0
        for idx in eachindex(bcsM)
            w = W[idx]
            w == 0.0 && continue
            d = dist_fn(bcsM[idx], bcsN[idx]; dist_kwargs...)
            best = max(best, w * d)
        end
        return best / float(task.agg_norm)
    elseif task.weight_mode == :integrate
        if agg_mode == :mean
            if threads && Threads.nthreads() > 1
                nT = Threads.nthreads()
                acc_by_slot = fill(0.0, nT)
                Threads.@threads for slot in 1:nT
                    acc = 0.0
                    for idx in slot:nT:length(bcsM)
                        w = W[idx]
                        w == 0.0 && continue
                        acc += w * dist_fn(bcsM[idx], bcsN[idx]; dist_kwargs...)
                    end
                    acc_by_slot[slot] = acc
                end
                acc = sum(acc_by_slot)
                return (acc / sumw) / float(task.agg_norm)
            end
            acc = 0.0
            for idx in eachindex(bcsM)
                w = W[idx]
                w == 0.0 && continue
                acc += w * dist_fn(bcsM[idx], bcsN[idx]; dist_kwargs...)
            end
            return (acc / sumw) / float(task.agg_norm)
        elseif agg_mode == :pmean
            p = float(task.agg_p)
            if threads && Threads.nthreads() > 1
                nT = Threads.nthreads()
                acc_by_slot = fill(0.0, nT)
                Threads.@threads for slot in 1:nT
                    acc = 0.0
                    for idx in slot:nT:length(bcsM)
                        w = W[idx]
                        w == 0.0 && continue
                        d = dist_fn(bcsM[idx], bcsN[idx]; dist_kwargs...)
                        acc += w * d^p
                    end
                    acc_by_slot[slot] = acc
                end
                acc = sum(acc_by_slot)
                return ((acc / sumw)^(1 / p)) / float(task.agg_norm)
            end
            acc = 0.0
            for idx in eachindex(bcsM)
                w = W[idx]
                w == 0.0 && continue
                d = dist_fn(bcsM[idx], bcsN[idx]; dist_kwargs...)
                acc += w * d^p
            end
            return ((acc / sumw)^(1 / p)) / float(task.agg_norm)
        elseif agg_mode == :max
            if threads && Threads.nthreads() > 1
                nT = Threads.nthreads()
                best_by_slot = fill(0.0, nT)
                Threads.@threads for slot in 1:nT
                    best = 0.0
                    for idx in slot:nT:length(bcsM)
                        w = W[idx]
                        w == 0.0 && continue
                        d = dist_fn(bcsM[idx], bcsN[idx]; dist_kwargs...)
                        best = max(best, w * d)
                    end
                    best_by_slot[slot] = best
                end
                return maximum(best_by_slot) / float(task.agg_norm)
            end
            best = 0.0
            for idx in eachindex(bcsM)
                w = W[idx]
                w == 0.0 && continue
                d = dist_fn(bcsM[idx], bcsN[idx]; dist_kwargs...)
                best = max(best, w * d)
            end
            return best / float(task.agg_norm)
        elseif agg_mode isa Function
            if threads && Threads.nthreads() > 1
                vals = Vector{Float64}(undef, length(bcsM))
                Threads.@threads for idx in 1:length(bcsM)
                    vals[idx] = dist_fn(bcsM[idx], bcsN[idx]; dist_kwargs...)
                end
                return float(agg_mode(vals)) / float(task.agg_norm)
            end
            vals = Float64[]
            for idx in eachindex(bcsM)
                push!(vals, dist_fn(bcsM[idx], bcsN[idx]; dist_kwargs...))
            end
            return float(agg_mode(vals)) / float(task.agg_norm)
        else
            throw(ArgumentError("run_invariants: unknown agg=$(task.agg)"))
        end
    end
    throw(ArgumentError("run_invariants: unknown weight_mode=$(task.weight_mode)"))
end

function _run_slice_distance_from_barcodes(
    bcsM,
    bcsN,
    W::AbstractMatrix{Float64},
    task::SliceDistanceTask,
)::Float64
    if _SLICE_USE_PACKED_DISTANCE_FASTPATH[] &&
       bcsM isa PackedBarcodeGrid{PackedFloatBarcode} &&
       bcsN isa PackedBarcodeGrid{PackedFloatBarcode}
        return _run_slice_distance_from_packed_barcodes(bcsM, bcsN, W, task)
    end
    return _run_slice_distance_from_barcodes_generic(bcsM, bcsN, W, task)
end

function run_invariants(plan::CompiledSlicePlan, cache::SliceModulePairCache, task::SliceDistanceTask)::Float64
    dataM = run_invariants(plan, cache.A, SliceBarcodesTask(; packed = true, threads = task.threads))
    dataN = run_invariants(plan, cache.B, SliceBarcodesTask(; packed = true, threads = task.threads))
    return _run_slice_distance_from_barcodes(dataM.barcodes, dataN.barcodes, plan.weights, task)
end

function run_invariants(plan::CompiledSlicePlan, modules::Tuple{<:PModule,<:PModule}, task::SliceDistanceTask)::Float64
    return run_invariants(plan, module_cache(modules[1], modules[2]), task)
end

@inline _kernel_uses_points_fast(kind::Symbol) =
    kind in (:bottleneck_gaussian, :bottleneck_laplacian, :wasserstein_gaussian, :wasserstein_laplacian)
@inline _kernel_uses_points_fast(::Any) = false

@inline _kernel_uses_landscape_features(kind::Symbol) =
    kind in (:landscape_gaussian, :landscape_laplacian, :landscape_linear)
@inline _kernel_uses_landscape_features(::Any) = false

@inline _all_packed_float_grid(::PackedBarcodeGrid{PackedFloatBarcode}) = true
function _all_packed_float_grid(bcs)
    @inbounds for idx in eachindex(bcs)
        bcs[idx] isa PackedFloatBarcode || return false
    end
    return true
end

@inline function _kernel_from_points(
    ptsA::Vector{Tuple{Float64,Float64}},
    ptsB::Vector{Tuple{Float64,Float64}},
    kind::Symbol,
    sigma::Float64,
    gamma,
    p::Float64,
    q::Float64,
)::Float64
    if kind === :bottleneck_gaussian
        sigma > 0 || error("_run_slice_kernel_from_barcodes: sigma must be > 0")
        d = bottleneck_distance(ptsA, ptsB)
        g = (gamma === nothing) ? (1.0 / (2.0 * sigma^2)) : float(gamma)
        return exp(-g * d * d)
    elseif kind === :bottleneck_laplacian
        sigma > 0 || error("_run_slice_kernel_from_barcodes: sigma must be > 0")
        d = bottleneck_distance(ptsA, ptsB)
        return exp(-d / sigma)
    elseif kind === :wasserstein_gaussian
        sigma > 0 || error("_run_slice_kernel_from_barcodes: sigma must be > 0")
        d = wasserstein_distance(ptsA, ptsB; p=p, q=q)
        return exp(-(d * d) / (2.0 * sigma^2))
    elseif kind === :wasserstein_laplacian
        sigma > 0 || error("_run_slice_kernel_from_barcodes: sigma must be > 0")
        d = wasserstein_distance(ptsA, ptsB; p=p, q=q)
        return exp(-d / sigma)
    else
        error("_run_slice_kernel_from_barcodes: unsupported fast point kernel kind=$kind")
    end
end

function _landscape_feature_cache(
    bcs,
    tgrid::Vector{Float64},
    kmax::Int;
    threads::Bool,
)
    out = Vector{Vector{Float64}}(undef, length(bcs))
    if threads && Threads.nthreads() > 1
        Threads.@threads for idx in eachindex(out)
            pl = persistence_landscape(bcs[idx]; kmax=kmax, tgrid=tgrid)
            out[idx] = _landscape_feature_vector(pl)
        end
    else
        @inbounds for idx in eachindex(out)
            pl = persistence_landscape(bcs[idx]; kmax=kmax, tgrid=tgrid)
            out[idx] = _landscape_feature_vector(pl)
        end
    end
    return out
end

function _landscape_feature_cache(
    cache::SliceModuleCache,
    plan::CompiledSlicePlan,
    bars::PackedBarcodeGrid{PackedFloatBarcode},
    tgrid::Vector{Float64},
    kmax::Int;
    threads::Bool,
)
    key = _slice_plan_landscape_key(plan, tgrid, kmax)
    Base.lock(cache.lock)
    try
        feats = get(cache.landscape_plan_features, key, nothing)
        feats === nothing || return feats
    finally
        Base.unlock(cache.lock)
    end

    feats = _landscape_feature_cache(bars, tgrid, kmax; threads=threads)

    Base.lock(cache.lock)
    try
        return get!(cache.landscape_plan_features, key, feats)
    finally
        Base.unlock(cache.lock)
    end
end

@inline function _kernel_from_features(
    vA::Vector{Float64},
    vB::Vector{Float64},
    kind::Symbol,
    sigma::Float64,
    gamma,
)::Float64
    if kind === :landscape_linear
        return dot(vA, vB)
    elseif kind === :landscape_gaussian
        sigma > 0 || error("_run_slice_kernel_from_barcodes: sigma must be > 0")
        d2 = 0.0
        @inbounds for i in eachindex(vA, vB)
            d = vA[i] - vB[i]
            d2 += d * d
        end
        g = (gamma === nothing) ? (1.0 / (2.0 * sigma^2)) : float(gamma)
        return exp(-g * d2)
    elseif kind === :landscape_laplacian
        sigma > 0 || error("_run_slice_kernel_from_barcodes: sigma must be > 0")
        d2 = 0.0
        @inbounds for i in eachindex(vA, vB)
            d = vA[i] - vB[i]
            d2 += d * d
        end
        return exp(-sqrt(d2) / sigma)
    else
        error("_run_slice_kernel_from_barcodes: unsupported feature kernel kind=$kind")
    end
end

function _run_slice_kernel_from_features(
    featM::AbstractVector{<:AbstractVector{Float64}},
    featN::AbstractVector{<:AbstractVector{Float64}},
    W::AbstractMatrix{Float64},
    task::SliceKernelTask,
)::Float64
    length(featM) == length(featN) || error("run_invariants: feature grid shape mismatch")

    sumw = sum(W)
    sumw > 0.0 || error("run_invariants: total weight is zero")
    threads = task.threads && Threads.nthreads() > 1

    if threads
        nT = Threads.nthreads()
        acc_by_slot = fill(0.0, nT)
        Threads.@threads for slot in 1:nT
            acc = 0.0
            for idx in slot:nT:length(featM)
                w = W[idx]
                w == 0.0 && continue
                acc += w * _kernel_from_features(featM[idx], featN[idx], task.kind, task.sigma, task.gamma)
            end
            acc_by_slot[slot] = acc
        end
        return sum(acc_by_slot) / sumw
    end

    acc = 0.0
    @inbounds for idx in eachindex(featM)
        w = W[idx]
        w == 0.0 && continue
        acc += w * _kernel_from_features(featM[idx], featN[idx], task.kind, task.sigma, task.gamma)
    end
    return acc / sumw
end

function _run_slice_kernel_from_barcodes(
    bM,
    bN,
    W::AbstractMatrix{Float64},
    task::SliceKernelTask,
)::Float64
    size(bM) == size(bN) || error("run_invariants: barcode grid shape mismatch")

    sumw = sum(W)
    sumw > 0.0 || error("run_invariants: total weight is zero")
    kind = task.kind
    threads = task.threads && Threads.nthreads() > 1

    # Landscape kernels: compile per-slice feature vectors once per module+plan pair,
    # then run one typed weighted aggregation pass.
    if _kernel_uses_landscape_features(kind)
        tg = task.tgrid
        if tg === nothing
            tg = _default_tgrid_from_barcodes(vcat(vec(bM), vec(bN)); nsteps=task.tgrid_nsteps)
        end
        tg = _clean_tgrid(tg)
        featM = _landscape_feature_cache(bM, tg, task.kmax; threads=threads)
        featN = _landscape_feature_cache(bN, tg, task.kmax; threads=threads)
        return _run_slice_kernel_from_features(featM, featN, W, task)
    end

    # Fast point-kernel path on packed barcodes.
    if _kernel_uses_points_fast(kind) && _all_packed_float_grid(bM) && _all_packed_float_grid(bN)
        scratch_by_thread = _scratch_arenas(threads)
        if threads
            nT = Threads.nthreads()
            acc_by_slot = fill(0.0, nT)
            Threads.@threads for slot in 1:nT
                scratch = scratch_by_thread[Threads.threadid()]
                acc = 0.0
                for idx in slot:nT:length(bM)
                    w = W[idx]
                    w == 0.0 && continue
                    _points_from_packed!(scratch.points_a, bM[idx]::PackedFloatBarcode)
                    _points_from_packed!(scratch.points_b, bN[idx]::PackedFloatBarcode)
                    acc += w * _kernel_from_points(
                        scratch.points_a, scratch.points_b, kind, task.sigma, task.gamma, task.p, task.q
                    )
                end
                acc_by_slot[slot] = acc
            end
            return sum(acc_by_slot) / sumw
        end

        scratch = scratch_by_thread[1]
        acc = 0.0
        @inbounds for idx in eachindex(bM)
            w = W[idx]
            w == 0.0 && continue
            _points_from_packed!(scratch.points_a, bM[idx]::PackedFloatBarcode)
            _points_from_packed!(scratch.points_b, bN[idx]::PackedFloatBarcode)
            acc += w * _kernel_from_points(
                scratch.points_a, scratch.points_b, kind, task.sigma, task.gamma, task.p, task.q
            )
        end
        return acc / sumw
    end

    tg = task.tgrid
    if tg !== nothing
        tg = _clean_tgrid(tg)
    end

    nd, no = size(bM)
    if threads
        nT = Threads.nthreads()
        acc_by_slot = fill(0.0, nT)
        Threads.@threads for slot in 1:nT
            acc = 0.0
            for k in slot:nT:(nd * no)
                i = div(k - 1, no) + 1
                j = (k - 1) % no + 1
                w = W[i, j]
                w == 0.0 && continue
                acc += w * _barcode_kernel(bM[i, j], bN[i, j];
                                           kind=kind, sigma=task.sigma, gamma=task.gamma,
                                           p=task.p, q=task.q, tgrid=tg, kmax=task.kmax)
            end
            acc_by_slot[slot] = acc
        end
        return sum(acc_by_slot) / sumw
    end

    acc = 0.0
    for i in 1:nd, j in 1:no
        w = W[i, j]
        w == 0.0 && continue
        acc += w * _barcode_kernel(bM[i, j], bN[i, j];
                                   kind=kind, sigma=task.sigma, gamma=task.gamma,
                                   p=task.p, q=task.q, tgrid=tg, kmax=task.kmax)
    end
    return acc / sumw
end

function run_invariants(plan::CompiledSlicePlan, cache::SliceModulePairCache, task::SliceKernelTask)::Float64
    dataM = run_invariants(plan, cache.A, SliceBarcodesTask(; packed = true, threads = task.threads))
    dataN = run_invariants(plan, cache.B, SliceBarcodesTask(; packed = true, threads = task.threads))
    if _SLICE_USE_LANDSCAPE_FEATURE_CACHE[] &&
       _kernel_uses_landscape_features(task.kind) &&
       dataM.barcodes isa PackedBarcodeGrid{PackedFloatBarcode} &&
       dataN.barcodes isa PackedBarcodeGrid{PackedFloatBarcode}
        tg = task.tgrid
        if tg === nothing
            tg = _default_tgrid_from_barcodes(vcat(vec(dataM.barcodes), vec(dataN.barcodes)); nsteps=task.tgrid_nsteps)
        end
        tg = _clean_tgrid(tg)
        featM = _landscape_feature_cache(cache.A, plan, dataM.barcodes, tg, task.kmax; threads=task.threads)
        featN = _landscape_feature_cache(cache.B, plan, dataN.barcodes, tg, task.kmax; threads=task.threads)
        return _run_slice_kernel_from_features(featM, featN, plan.weights, task)
    end
    return _run_slice_kernel_from_barcodes(dataM.barcodes, dataN.barcodes, plan.weights, task)
end

function run_invariants(plan::CompiledSlicePlan, modules::Tuple{<:PModule,<:PModule}, task::SliceKernelTask)::Float64
    return run_invariants(plan, module_cache(modules[1], modules[2]), task)
end

"""
    slice_barcodes(M, slices; default_weight=1.0, normalize_weights=true) -> SliceBarcodesResult

Compute the 1D slice barcodes of a (multi-parameter) module `M` for an explicit
collection of slice specs.

Each slice spec can be:
- a chain `Vector{Int}`,
- a NamedTuple `(chain=..., values=..., weight=...)`,
- a Tuple `(chain, values, weight)`.

Returns a [`SliceBarcodesResult`](@ref). Use [`slice_barcodes`](@ref) and
[`slice_weights`](@ref) on that result for cheap-first inspection.
"""
function slice_barcodes(M::PModule{K}, slices::AbstractVector;
                        default_weight::Real=1.0,
                        normalize_weights::Bool=true,
                        values=nothing,
                        threads::Bool=Threads.nthreads() > 1,
                        packed::Bool=false) where {K}
    specs = collect_slices(slices; default_weight=default_weight, values=values)
    return slice_barcodes(M, specs; normalize_weights=normalize_weights, threads=threads, packed=packed)
end

function slice_barcodes(M::PModule{K}, slices::AbstractVector{<:SliceSpec{<:Real,Nothing}};
                        normalize_weights::Bool=true,
                        threads::Bool=Threads.nthreads() > 1,
                        packed::Bool=false) where {K}
    n = length(slices)
    weights = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        weights[i] = isempty(slices[i].chain) ? 0.0 : float(slices[i].weight)
    end
    if normalize_weights
        s = sum(weights)
        if s > 0
            weights ./= s
        else
            weights .= 0.0
        end
    end

    if packed
        bcs = Vector{PackedIndexBarcode}(undef, n)
        if threads && Threads.nthreads() > 1
            Threads.@threads for i in 1:n
                ch = slices[i].chain
                bcs[i] = isempty(ch) ? _empty_packed_index_barcode() :
                    (_slice_barcode_packed(M, ch; values=nothing)::PackedIndexBarcode)
            end
        else
            @inbounds for i in 1:n
                ch = slices[i].chain
                bcs[i] = isempty(ch) ? _empty_packed_index_barcode() :
                    (_slice_barcode_packed(M, ch; values=nothing)::PackedIndexBarcode)
            end
        end
        return _slice_barcodes_result_without_geometry(bcs, weights)
    end

    bcs = Vector{IndexBarcode}(undef, n)
    if threads && Threads.nthreads() > 1
        Threads.@threads for i in 1:n
            ch = slices[i].chain
            bcs[i] = isempty(ch) ? _empty_index_barcode() : slice_barcode(M, ch; values=nothing)
        end
    else
        @inbounds for i in 1:n
            ch = slices[i].chain
            bcs[i] = isempty(ch) ? _empty_index_barcode() : slice_barcode(M, ch; values=nothing)
        end
    end
    return _slice_barcodes_result_without_geometry(bcs, weights)
end

function slice_barcodes(M::PModule{K}, slices::AbstractVector{<:SliceSpec{<:Real,<:AbstractVector{<:Integer}}};
                        normalize_weights::Bool=true,
                        threads::Bool=Threads.nthreads() > 1,
                        packed::Bool=false) where {K}
    n = length(slices)
    weights = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        weights[i] = isempty(slices[i].chain) ? 0.0 : float(slices[i].weight)
    end
    if normalize_weights
        s = sum(weights)
        if s > 0
            weights ./= s
        else
            weights .= 0.0
        end
    end

    if packed
        bcs = Vector{PackedIndexBarcode}(undef, n)
        if threads && Threads.nthreads() > 1
            Threads.@threads for i in 1:n
                spec = slices[i]
                bcs[i] = isempty(spec.chain) ? _empty_packed_index_barcode() :
                    (_slice_barcode_packed(M, spec.chain; values=spec.values)::PackedIndexBarcode)
            end
        else
            @inbounds for i in 1:n
                spec = slices[i]
                bcs[i] = isempty(spec.chain) ? _empty_packed_index_barcode() :
                    (_slice_barcode_packed(M, spec.chain; values=spec.values)::PackedIndexBarcode)
            end
        end
        return _slice_barcodes_result_without_geometry(bcs, weights)
    end

    bcs = Vector{IndexBarcode}(undef, n)
    if threads && Threads.nthreads() > 1
        Threads.@threads for i in 1:n
            spec = slices[i]
            bcs[i] = isempty(spec.chain) ? _empty_index_barcode() : slice_barcode(M, spec.chain; values=spec.values)
        end
    else
        @inbounds for i in 1:n
            spec = slices[i]
            bcs[i] = isempty(spec.chain) ? _empty_index_barcode() : slice_barcode(M, spec.chain; values=spec.values)
        end
    end
    return _slice_barcodes_result_without_geometry(bcs, weights)
end

function slice_barcodes(M::PModule{K}, slices::AbstractVector{<:SliceSpec{<:Real,<:AbstractVector{<:Real}}};
                        normalize_weights::Bool=true,
                        threads::Bool=Threads.nthreads() > 1,
                        packed::Bool=false) where {K}
    n = length(slices)
    weights = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        weights[i] = isempty(slices[i].chain) ? 0.0 : float(slices[i].weight)
    end
    if normalize_weights
        s = sum(weights)
        if s > 0
            weights ./= s
        else
            weights .= 0.0
        end
    end

    if packed
        bcs = Vector{PackedFloatBarcode}(undef, n)
        if threads && Threads.nthreads() > 1
            Threads.@threads for i in 1:n
                spec = slices[i]
                bcs[i] = isempty(spec.chain) ? _empty_packed_float_barcode() :
                    (_slice_barcode_packed(M, spec.chain; values=spec.values)::PackedFloatBarcode)
            end
        else
            @inbounds for i in 1:n
                spec = slices[i]
                bcs[i] = isempty(spec.chain) ? _empty_packed_float_barcode() :
                    (_slice_barcode_packed(M, spec.chain; values=spec.values)::PackedFloatBarcode)
            end
        end
        return _slice_barcodes_result_without_geometry(bcs, weights)
    end

    bcs = Vector{FloatBarcode}(undef, n)
    if threads && Threads.nthreads() > 1
        Threads.@threads for i in 1:n
            spec = slices[i]
            bcs[i] = isempty(spec.chain) ? _empty_float_barcode() : slice_barcode(M, spec.chain; values=spec.values)
        end
    else
        @inbounds for i in 1:n
            spec = slices[i]
            bcs[i] = isempty(spec.chain) ? _empty_float_barcode() : slice_barcode(M, spec.chain; values=spec.values)
        end
    end
    return _slice_barcodes_result_without_geometry(bcs, weights)
end


slice_barcodes(M::PModule{K}, chain::AbstractVector{Int}; kwargs...) where {K} =
    slice_barcodes(M, [chain]; kwargs...)

"""
    slice_barcodes(M, pi; directions, offsets, ...) -> SliceBarcodesResult

Compute slice barcodes via a finite encoding map `pi` (Rn or Zn style).

This mirrors the "geometric slicing" API of `mp_landscape`:

- `directions`: list of direction vectors.
- `offsets`: list of basepoints (one per offset).
- For Rn/PL encodings: provide `tmin, tmax, nsteps`.
- For Zn encodings: provide `kmin, kmax_param`.

Weights:
- `direction_weight` applies `_direction_weight` (e.g. :lesnick_l1).
- `offset_weights` can be:
  * `nothing` (uniform), or
  * a vector of length `length(offsets)`, or
  * a function `x0 -> weight`.

Returns a [`SliceBarcodesResult`](@ref) carrying the barcode grid, weight
matrix, and the direction/offset metadata used to generate it. Use
`packed=true` when downstream code can remain on the packed-barcode path.
"""
function slice_barcodes(
    M::PModule{K},
    pi;
    opts::InvariantOptions = InvariantOptions(),
    directions = :auto,
    offsets = :auto,
    normalize_dirs::Symbol = :none,
    direction_weight::Union{Symbol,Function,Real} = :none,
    offset_weights = nothing,
    normalize_weights::Bool = true,
    drop_unknown::Bool = true,
    values = nothing,
    threads::Bool = (Threads.nthreads() > 1),
    packed::Bool = false,
    slice_kwargs...
) where {K}
    if directions === :auto || directions === nothing
        error("slice_barcodes: provide directions explicitly for non-PLike encodings")
    end
    if offsets === :auto || offsets === nothing
        error("slice_barcodes: provide offsets explicitly for non-PLike encodings")
    end

    box0 = get(slice_kwargs, :box, opts.box)
    strict0 = get(slice_kwargs, :strict, opts.strict)
    slice_kwargs0 = (;
        (k => v for (k, v) in pairs(slice_kwargs)
            if k != :box && k != :strict)...)

    opts_chain = InvariantOptions(
        axes = opts.axes,
        axes_policy = opts.axes_policy,
        max_axis_len = opts.max_axis_len,
        box = box0,
        threads = opts.threads,
        strict = strict0,
        pl_mode = opts.pl_mode,
    )
    tmin0 = get(slice_kwargs0, :tmin, nothing)
    tmax0 = get(slice_kwargs0, :tmax, nothing)
    if opts_chain.box === nothing && tmin0 === nothing && tmax0 === nothing
        opts_chain = InvariantOptions(box = :auto, strict = opts_chain.strict,
                                      threads = opts.threads, axes = opts.axes,
                                      axes_policy = opts.axes_policy, max_axis_len = opts.max_axis_len,
                                      pl_mode = opts.pl_mode)
    end

    filtered = (;
        (k => v for (k, v) in pairs(slice_kwargs0)
            if k != :kmin && k != :kmax_param && k != :default_weight && k != :lengthscale)...)

    dirs_in = [_normalize_dir(dir, normalize_dirs) for dir in directions]
    offs0 = offsets

    nd = length(dirs_in)
    no = length(offs0)
    nd > 0 || error("slice_barcodes: directions is empty")
    no > 0 || error("slice_barcodes: offsets is empty")

    wdir = Vector{Float64}(undef, nd)
    for i in 1:nd
        wdir[i] = SliceInvariants.direction_weight(dirs_in[i], direction_weight)
    end
    woff = _offset_sample_weights(offs0, offset_weights)

    W = wdir * woff'
    if normalize_weights
        s = sum(W)
        s > 0 || error("slice_barcodes: total slice weight is zero")
        W ./= s
    end

    if packed
        if values === nothing || _values_are_float_vector(values)
            bcs = Matrix{PackedFloatBarcode}(undef, nd, no)
            if threads && Threads.nthreads() > 1
                Threads.@threads for k in 1:(nd * no)
                    i = div((k - 1), no) + 1
                    j = (k - 1) % no + 1
                    chain, tvals = slice_chain(pi, offs0[j], dirs_in[i], opts_chain; drop_unknown=drop_unknown, filtered...)
                    if isempty(chain) || isempty(tvals)
                        bcs[i, j] = _empty_packed_float_barcode()
                        continue
                    end
                    vals_use = values === nothing ? tvals : values
                    bcs[i, j] = _slice_barcode_packed(M, chain; values = vals_use, check_chain = false)::PackedFloatBarcode
                end
            else
                for i in 1:nd, j in 1:no
                    chain, tvals = slice_chain(pi, offs0[j], dirs_in[i], opts_chain; drop_unknown=drop_unknown, filtered...)
                    if isempty(chain) || isempty(tvals)
                        bcs[i, j] = _empty_packed_float_barcode()
                        continue
                    end
                    vals_use = values === nothing ? tvals : values
                    bcs[i, j] = _slice_barcode_packed(M, chain; values = vals_use, check_chain = false)::PackedFloatBarcode
                end
            end
            return _slice_barcodes_result(_packed_grid_from_matrix(bcs), W, dirs_in, offs0)
        elseif _values_are_int_vector(values)
            bcs = Matrix{PackedIndexBarcode}(undef, nd, no)
            if threads && Threads.nthreads() > 1
                Threads.@threads for k in 1:(nd * no)
                    i = div((k - 1), no) + 1
                    j = (k - 1) % no + 1
                    chain, tvals = slice_chain(pi, offs0[j], dirs_in[i], opts_chain; drop_unknown=drop_unknown, filtered...)
                    if isempty(chain) || isempty(tvals)
                        bcs[i, j] = _empty_packed_index_barcode()
                        continue
                    end
                    bcs[i, j] = _slice_barcode_packed(M, chain; values = values, check_chain = false)::PackedIndexBarcode
                end
            else
                for i in 1:nd, j in 1:no
                    chain, tvals = slice_chain(pi, offs0[j], dirs_in[i], opts_chain; drop_unknown=drop_unknown, filtered...)
                    if isempty(chain) || isempty(tvals)
                        bcs[i, j] = _empty_packed_index_barcode()
                        continue
                    end
                    bcs[i, j] = _slice_barcode_packed(M, chain; values = values, check_chain = false)::PackedIndexBarcode
                end
            end
            return _slice_barcodes_result(_packed_grid_from_matrix(bcs), W, dirs_in, offs0)
        else
            bcs = Matrix{PackedFloatBarcode}(undef, nd, no)
            if threads && Threads.nthreads() > 1
                Threads.@threads for k in 1:(nd * no)
                    i = div((k - 1), no) + 1
                    j = (k - 1) % no + 1
                    chain, tvals = slice_chain(pi, offs0[j], dirs_in[i], opts_chain; drop_unknown=drop_unknown, filtered...)
                    if isempty(chain) || isempty(tvals)
                        bcs[i, j] = _empty_packed_float_barcode()
                        continue
                    end
                    vals_use = values === nothing ? tvals : values
                    pb = _slice_barcode_packed(M, chain; values = vals_use, check_chain = false)
                    bcs[i, j] = pb isa PackedFloatBarcode ? pb : _pack_float_barcode(_barcode_from_packed(pb))
                end
            else
                for i in 1:nd, j in 1:no
                    chain, tvals = slice_chain(pi, offs0[j], dirs_in[i], opts_chain; drop_unknown=drop_unknown, filtered...)
                    if isempty(chain) || isempty(tvals)
                        bcs[i, j] = _empty_packed_float_barcode()
                        continue
                    end
                    vals_use = values === nothing ? tvals : values
                    pb = _slice_barcode_packed(M, chain; values = vals_use, check_chain = false)
                    bcs[i, j] = pb isa PackedFloatBarcode ? pb : _pack_float_barcode(_barcode_from_packed(pb))
                end
            end
            return _slice_barcodes_result(_packed_grid_from_matrix(bcs), W, dirs_in, offs0)
        end
    end

    if values === nothing || _values_are_float_vector(values)
        bcs = Matrix{FloatBarcode}(undef, nd, no)
        if threads && Threads.nthreads() > 1
            Threads.@threads for k in 1:(nd * no)
                i = div((k - 1), no) + 1
                j = (k - 1) % no + 1
                chain, tvals = slice_chain(pi, offs0[j], dirs_in[i], opts_chain; drop_unknown=drop_unknown, filtered...)
                if isempty(chain) || isempty(tvals)
                    bcs[i, j] = _empty_float_barcode()
                    continue
                end
                vals_use = values === nothing ? tvals : values
                bcs[i, j] = slice_barcode(M, chain; values = vals_use, check_chain = false)
            end
        else
            for i in 1:nd, j in 1:no
                chain, tvals = slice_chain(pi, offs0[j], dirs_in[i], opts_chain; drop_unknown=drop_unknown, filtered...)
                if isempty(chain) || isempty(tvals)
                    bcs[i, j] = _empty_float_barcode()
                    continue
                end
                vals_use = values === nothing ? tvals : values
                bcs[i, j] = slice_barcode(M, chain; values = vals_use, check_chain = false)
            end
        end
        return _slice_barcodes_result(bcs, W, dirs_in, offs0)
    elseif _values_are_int_vector(values)
        bcs = Matrix{IndexBarcode}(undef, nd, no)
        if threads && Threads.nthreads() > 1
            Threads.@threads for k in 1:(nd * no)
                i = div((k - 1), no) + 1
                j = (k - 1) % no + 1
                chain, tvals = slice_chain(pi, offs0[j], dirs_in[i], opts_chain; drop_unknown=drop_unknown, filtered...)
                if isempty(chain) || isempty(tvals)
                    bcs[i, j] = _empty_index_barcode()
                    continue
                end
                bcs[i, j] = slice_barcode(M, chain; values = values, check_chain = false)
            end
        else
            for i in 1:nd, j in 1:no
                chain, tvals = slice_chain(pi, offs0[j], dirs_in[i], opts_chain; drop_unknown=drop_unknown, filtered...)
                if isempty(chain) || isempty(tvals)
                    bcs[i, j] = _empty_index_barcode()
                    continue
                end
                bcs[i, j] = slice_barcode(M, chain; values = values, check_chain = false)
            end
        end
        return _slice_barcodes_result(bcs, W, dirs_in, offs0)
    else
        bcs = Matrix{FloatBarcode}(undef, nd, no)
        if threads && Threads.nthreads() > 1
            Threads.@threads for k in 1:(nd * no)
                i = div((k - 1), no) + 1
                j = (k - 1) % no + 1
                chain, tvals = slice_chain(pi, offs0[j], dirs_in[i], opts_chain; drop_unknown=drop_unknown, filtered...)
                if isempty(chain) || isempty(tvals)
                    bcs[i, j] = _empty_float_barcode()
                    continue
                end
                vals_use = values === nothing ? tvals : values
                bcs[i, j] = _to_float_barcode(slice_barcode(M, chain; values = vals_use, check_chain = false))
            end
        else
            for i in 1:nd, j in 1:no
                chain, tvals = slice_chain(pi, offs0[j], dirs_in[i], opts_chain; drop_unknown=drop_unknown, filtered...)
                if isempty(chain) || isempty(tvals)
                    bcs[i, j] = _empty_float_barcode()
                    continue
                end
                vals_use = values === nothing ? tvals : values
                bcs[i, j] = _to_float_barcode(slice_barcode(M, chain; values = vals_use, check_chain = false))
            end
        end
        return _slice_barcodes_result(bcs, W, dirs_in, offs0)
    end
end

function _prepare_geometric_slice_query(
    pi::PLikeEncodingMap,
    opts::InvariantOptions;
    directions = :auto,
    offsets = :auto,
    n_dirs::Integer = 16,
    n_offsets::Integer = 9,
    max_den::Integer = 8,
    include_axes::Bool = false,
    normalize_dirs::Symbol = :none,
    direction_weight::Union{Symbol,Function,Real} = :none,
    offset_weights = nothing,
    normalize_weights::Bool = true,
    offset_margin::Real = 0.05,
    slice_kwargs...
)
    box0 = get(slice_kwargs, :box, opts.box)
    strict0 = get(slice_kwargs, :strict, opts.strict)
    slice_kwargs0 = (; (k => v for (k, v) in pairs(slice_kwargs) if k != :box && k != :strict)...)

    opts_compile = InvariantOptions(
        axes = opts.axes,
        axes_policy = opts.axes_policy,
        max_axis_len = opts.max_axis_len,
        box = box0,
        threads = opts.threads,
        strict = strict0,
        pl_mode = opts.pl_mode,
    )

    dirs0 = directions
    offs0 = offsets
    if dirs0 === :auto || dirs0 === nothing
        dirs0 = default_directions(
            pi;
            n_dirs = n_dirs,
            max_den = max_den,
            include_axes = include_axes,
            normalize = (normalize_dirs == :none ? :none : normalize_dirs),
        )
    end

    opts_offsets = InvariantOptions(box = opts_compile.box, pl_mode = opts_compile.pl_mode)
    opts_chain = opts_compile
    tmin0 = get(slice_kwargs0, :tmin, nothing)
    tmax0 = get(slice_kwargs0, :tmax, nothing)
    if opts_chain.box === nothing && tmin0 === nothing && tmax0 === nothing
        opts_offsets = InvariantOptions(box = :auto, pl_mode = opts_compile.pl_mode)
        opts_chain = InvariantOptions(
            box = :auto,
            strict = opts_compile.strict,
            threads = opts.threads,
            axes = opts.axes,
            axes_policy = opts.axes_policy,
            max_axis_len = opts.max_axis_len,
            pl_mode = opts_compile.pl_mode,
        )
    end

    filtered = (;
        (k => v for (k, v) in pairs(slice_kwargs0)
            if k != :default_weight && k != :kmin && k != :kmax_param && k != :lengthscale)...)

    if offs0 === :auto || offs0 === nothing
        offs0 = default_offsets(pi, opts_offsets; n_offsets = n_offsets, margin = offset_margin)
    end

    if !isempty(offs0) && length(offs0[1]) == 0
        return (; dirs_in = Vector{Vector{Float64}}(), offs = offs0, weights = zeros(Float64, 0, 0), opts_chain, filtered)
    end

    if !isempty(dirs0) && !isempty(offs0) && length(dirs0[1]) != length(offs0[1])
        dirs0 = default_directions(
            length(offs0[1]);
            n_dirs = n_dirs,
            max_den = max_den,
            include_axes = include_axes,
            normalize = (normalize_dirs == :none ? :none : normalize_dirs),
        )
    end

    dirs_in = [_normalize_dir(dir, normalize_dirs) for dir in dirs0]
    nd = length(dirs_in)
    no = length(offs0)
    nd > 0 || error("slice_barcodes: directions is empty")
    no > 0 || error("slice_barcodes: offsets is empty")

    wdir = Vector{Float64}(undef, nd)
    @inbounds for i in 1:nd
        wdir[i] = SliceInvariants.direction_weight(dirs_in[i], direction_weight)
    end
    woff = _offset_sample_weights(offs0, offset_weights)
    W = wdir * woff'
    if normalize_weights
        s = sum(W)
        s > 0 || error("slice_barcodes: total slice weight is zero")
        W ./= s
    end

    return (; dirs_in, offs = offs0, weights = W, opts_chain, filtered)
end

function _slice_barcodes_geometric_packed(
    M::PModule{K},
    dirs_in,
    offs0,
    pi::PLikeEncodingMap,
    opts_chain::InvariantOptions,
    filtered::NamedTuple;
    drop_unknown::Bool = true,
    threads::Bool = (Threads.nthreads() > 1),
) where {K}
    nd = length(dirs_in)
    no = length(offs0)
    ns = nd * no
    bars = _packed_grid_undef(PackedFloatBarcode, nd, no)

    build_cache!(M.Q; cover=true, updown=true)
    cc = _get_cover_cache(M.Q)
    nQ = nvertices(M.Q)
    use_array_memo = _use_array_memo(nQ)

    if threads && Threads.nthreads() > 1
        nT = Threads.nthreads()
        memo_by_thread = use_array_memo ?
            [_new_array_memo(K, nQ) for _ in 1:nT] :
            [Dict{Tuple{Int,Int}, AbstractMatrix{K}}() for _ in 1:nT]
        rank_by_thread = [Matrix{Int}(undef, 0, 0) for _ in 1:nT]
        locate_ws_by_thread = [_SliceLocateBatchWorkspace() for _ in 1:nT]
        Threads.@threads for idx in 1:ns
            i = div(idx - 1, no) + 1
            j = (idx - 1) % no + 1
            tid = Threads.threadid()
            chain, tvals = _slice_chain_collect(
                pi,
                offs0[j],
                dirs_in[i],
                opts_chain;
                drop_unknown = drop_unknown,
                locate_ws = locate_ws_by_thread[tid],
                filtered...,
            )
            if isempty(chain)
                bars[idx] = _empty_packed_float_barcode()
                continue
            end
            m = length(chain)
            if size(rank_by_thread[tid], 1) < m
                rank_by_thread[tid] = Matrix{Int}(undef, m, m)
            end
            bars[idx] = _slice_barcode_packed_with_workspace(
                M,
                chain,
                _extended_values_view(tvals),
                cc,
                memo_by_thread[tid],
                rank_by_thread[tid],
            )
        end
    else
        memo = use_array_memo ? _new_array_memo(K, nQ) : Dict{Tuple{Int,Int}, AbstractMatrix{K}}()
        rank_work = Matrix{Int}(undef, 0, 0)
        locate_ws = _SliceLocateBatchWorkspace()
        @inbounds for idx in 1:ns
            i = div(idx - 1, no) + 1
            j = (idx - 1) % no + 1
            chain, tvals = _slice_chain_collect(
                pi,
                offs0[j],
                dirs_in[i],
                opts_chain;
                drop_unknown = drop_unknown,
                locate_ws = locate_ws,
                filtered...,
            )
            if isempty(chain)
                bars[idx] = _empty_packed_float_barcode()
                continue
            end
            m = length(chain)
            if size(rank_work, 1) < m
                rank_work = Matrix{Int}(undef, m, m)
            end
            bars[idx] = _slice_barcode_packed_with_workspace(
                M,
                chain,
                _extended_values_view(tvals),
                cc,
                memo,
                rank_work,
            )
        end
    end

    return bars
end

function _slice_pair_distance_geometric_packed_fast(
    M::PModule{K},
    N::PModule{K},
    dirs_in,
    offs0,
    W::AbstractMatrix{Float64},
    pi::PLikeEncodingMap,
    opts_chain::InvariantOptions,
    filtered::NamedTuple,
    task::SliceDistanceTask;
    drop_unknown::Bool = true,
)::Float64 where {K}
    nd = length(dirs_in)
    no = length(offs0)
    ns = nd * no

    build_cache!(M.Q; cover=true, updown=true)
    build_cache!(N.Q; cover=true, updown=true)
    ccM = _get_cover_cache(M.Q)
    ccN = _get_cover_cache(N.Q)
    nQM = nvertices(M.Q)
    nQN = nvertices(N.Q)
    use_array_memo_M = _use_array_memo(nQM)
    use_array_memo_N = _use_array_memo(nQN)
    threads = task.threads && Threads.nthreads() > 1
    agg_mode = (task.agg === mean) ? :mean : (task.agg === maximum) ? :max : task.agg
    sumw = sum(W)
    sumw == 0.0 && return 0.0
    dist_kind = _slice_distance_fast_kind(task)
    scratch_by_thread = _scratch_arenas(threads)

    if threads
        nT = Threads.nthreads()
        memoM_by_thread = use_array_memo_M ?
            [_new_array_memo(K, nQM) for _ in 1:nT] :
            [Dict{Tuple{Int,Int}, AbstractMatrix{K}}() for _ in 1:nT]
        memoN_by_thread = use_array_memo_N ?
            [_new_array_memo(K, nQN) for _ in 1:nT] :
            [Dict{Tuple{Int,Int}, AbstractMatrix{K}}() for _ in 1:nT]
        rankM_by_thread = [Matrix{Int}(undef, 0, 0) for _ in 1:nT]
        rankN_by_thread = [Matrix{Int}(undef, 0, 0) for _ in 1:nT]
        locate_ws_by_thread = [_SliceLocateBatchWorkspace() for _ in 1:nT]

        if task.weight_mode == :scale || agg_mode == :max
            best_by_slot = fill(0.0, nT)
            Threads.@threads for slot in 1:nT
                tid = Threads.threadid()
                scratch = scratch_by_thread[tid]
                best = 0.0
                for idx in slot:nT:ns
                    w = W[idx]
                    w == 0.0 && continue
                    i = div(idx - 1, no) + 1
                    j = (idx - 1) % no + 1
                    chain, tvals = _slice_chain_collect(
                        pi,
                        offs0[j],
                        dirs_in[i],
                        opts_chain;
                        drop_unknown = drop_unknown,
                        locate_ws = locate_ws_by_thread[tid],
                        filtered...,
                    )
                    isempty(chain) && continue
                    m = length(chain)
                    if size(rankM_by_thread[tid], 1) < m
                        rankM_by_thread[tid] = Matrix{Int}(undef, m, m)
                    end
                    if size(rankN_by_thread[tid], 1) < m
                        rankN_by_thread[tid] = Matrix{Int}(undef, m, m)
                    end
                    endpoints = _extended_values_view(tvals)
                    pbM = _slice_barcode_packed_with_workspace(M, chain, endpoints, ccM, memoM_by_thread[tid], rankM_by_thread[tid])
                    pbN = _slice_barcode_packed_with_workspace(N, chain, endpoints, ccN, memoN_by_thread[tid], rankN_by_thread[tid])
                    d = _packed_distance_value!(scratch, pbM, pbN, dist_kind, task)
                    best = max(best, w * d)
                end
                best_by_slot[slot] = best
            end
            return maximum(best_by_slot) / float(task.agg_norm)
        elseif task.weight_mode == :integrate && agg_mode == :mean
            acc_by_slot = fill(0.0, nT)
            Threads.@threads for slot in 1:nT
                tid = Threads.threadid()
                scratch = scratch_by_thread[tid]
                acc = 0.0
                for idx in slot:nT:ns
                    w = W[idx]
                    w == 0.0 && continue
                    i = div(idx - 1, no) + 1
                    j = (idx - 1) % no + 1
                    chain, tvals = _slice_chain_collect(
                        pi,
                        offs0[j],
                        dirs_in[i],
                        opts_chain;
                        drop_unknown = drop_unknown,
                        locate_ws = locate_ws_by_thread[tid],
                        filtered...,
                    )
                    isempty(chain) && continue
                    m = length(chain)
                    if size(rankM_by_thread[tid], 1) < m
                        rankM_by_thread[tid] = Matrix{Int}(undef, m, m)
                    end
                    if size(rankN_by_thread[tid], 1) < m
                        rankN_by_thread[tid] = Matrix{Int}(undef, m, m)
                    end
                    endpoints = _extended_values_view(tvals)
                    pbM = _slice_barcode_packed_with_workspace(M, chain, endpoints, ccM, memoM_by_thread[tid], rankM_by_thread[tid])
                    pbN = _slice_barcode_packed_with_workspace(N, chain, endpoints, ccN, memoN_by_thread[tid], rankN_by_thread[tid])
                    acc += w * _packed_distance_value!(scratch, pbM, pbN, dist_kind, task)
                end
                acc_by_slot[slot] = acc
            end
            return (sum(acc_by_slot) / sumw) / float(task.agg_norm)
        elseif task.weight_mode == :integrate && agg_mode == :pmean
            p = float(task.agg_p)
            acc_by_slot = fill(0.0, nT)
            Threads.@threads for slot in 1:nT
                tid = Threads.threadid()
                scratch = scratch_by_thread[tid]
                acc = 0.0
                for idx in slot:nT:ns
                    w = W[idx]
                    w == 0.0 && continue
                    i = div(idx - 1, no) + 1
                    j = (idx - 1) % no + 1
                    chain, tvals = _slice_chain_collect(
                        pi,
                        offs0[j],
                        dirs_in[i],
                        opts_chain;
                        drop_unknown = drop_unknown,
                        locate_ws = locate_ws_by_thread[tid],
                        filtered...,
                    )
                    isempty(chain) && continue
                    m = length(chain)
                    if size(rankM_by_thread[tid], 1) < m
                        rankM_by_thread[tid] = Matrix{Int}(undef, m, m)
                    end
                    if size(rankN_by_thread[tid], 1) < m
                        rankN_by_thread[tid] = Matrix{Int}(undef, m, m)
                    end
                    endpoints = _extended_values_view(tvals)
                    pbM = _slice_barcode_packed_with_workspace(M, chain, endpoints, ccM, memoM_by_thread[tid], rankM_by_thread[tid])
                    pbN = _slice_barcode_packed_with_workspace(N, chain, endpoints, ccN, memoN_by_thread[tid], rankN_by_thread[tid])
                    d = _packed_distance_value!(scratch, pbM, pbN, dist_kind, task)
                    acc += w * d^p
                end
                acc_by_slot[slot] = acc
            end
            return ((sum(acc_by_slot) / sumw)^(1 / p)) / float(task.agg_norm)
        end
    else
        memoM = use_array_memo_M ? _new_array_memo(K, nQM) : Dict{Tuple{Int,Int}, AbstractMatrix{K}}()
        memoN = use_array_memo_N ? _new_array_memo(K, nQN) : Dict{Tuple{Int,Int}, AbstractMatrix{K}}()
        rankM = Matrix{Int}(undef, 0, 0)
        rankN = Matrix{Int}(undef, 0, 0)
        locate_ws = _SliceLocateBatchWorkspace()
        scratch = scratch_by_thread[1]

        if task.weight_mode == :scale || agg_mode == :max
            best = 0.0
            @inbounds for idx in 1:ns
                w = W[idx]
                w == 0.0 && continue
                i = div(idx - 1, no) + 1
                j = (idx - 1) % no + 1
                chain, tvals = _slice_chain_collect(
                    pi,
                    offs0[j],
                    dirs_in[i],
                    opts_chain;
                    drop_unknown = drop_unknown,
                    locate_ws = locate_ws,
                    filtered...,
                )
                isempty(chain) && continue
                m = length(chain)
                if size(rankM, 1) < m
                    rankM = Matrix{Int}(undef, m, m)
                end
                if size(rankN, 1) < m
                    rankN = Matrix{Int}(undef, m, m)
                end
                endpoints = _extended_values_view(tvals)
                pbM = _slice_barcode_packed_with_workspace(M, chain, endpoints, ccM, memoM, rankM)
                pbN = _slice_barcode_packed_with_workspace(N, chain, endpoints, ccN, memoN, rankN)
                d = _packed_distance_value!(scratch, pbM, pbN, dist_kind, task)
                best = max(best, w * d)
            end
            return best / float(task.agg_norm)
        elseif task.weight_mode == :integrate && agg_mode == :mean
            acc = 0.0
            @inbounds for idx in 1:ns
                w = W[idx]
                w == 0.0 && continue
                i = div(idx - 1, no) + 1
                j = (idx - 1) % no + 1
                chain, tvals = _slice_chain_collect(
                    pi,
                    offs0[j],
                    dirs_in[i],
                    opts_chain;
                    drop_unknown = drop_unknown,
                    locate_ws = locate_ws,
                    filtered...,
                )
                isempty(chain) && continue
                m = length(chain)
                if size(rankM, 1) < m
                    rankM = Matrix{Int}(undef, m, m)
                end
                if size(rankN, 1) < m
                    rankN = Matrix{Int}(undef, m, m)
                end
                endpoints = _extended_values_view(tvals)
                pbM = _slice_barcode_packed_with_workspace(M, chain, endpoints, ccM, memoM, rankM)
                pbN = _slice_barcode_packed_with_workspace(N, chain, endpoints, ccN, memoN, rankN)
                acc += w * _packed_distance_value!(scratch, pbM, pbN, dist_kind, task)
            end
            return (acc / sumw) / float(task.agg_norm)
        elseif task.weight_mode == :integrate && agg_mode == :pmean
            p = float(task.agg_p)
            acc = 0.0
            @inbounds for idx in 1:ns
                w = W[idx]
                w == 0.0 && continue
                i = div(idx - 1, no) + 1
                j = (idx - 1) % no + 1
                chain, tvals = _slice_chain_collect(
                    pi,
                    offs0[j],
                    dirs_in[i],
                    opts_chain;
                    drop_unknown = drop_unknown,
                    locate_ws = locate_ws,
                    filtered...,
                )
                isempty(chain) && continue
                m = length(chain)
                if size(rankM, 1) < m
                    rankM = Matrix{Int}(undef, m, m)
                end
                if size(rankN, 1) < m
                    rankN = Matrix{Int}(undef, m, m)
                end
                endpoints = _extended_values_view(tvals)
                pbM = _slice_barcode_packed_with_workspace(M, chain, endpoints, ccM, memoM, rankM)
                pbN = _slice_barcode_packed_with_workspace(N, chain, endpoints, ccN, memoN, rankN)
                d = _packed_distance_value!(scratch, pbM, pbN, dist_kind, task)
                acc += w * d^p
            end
            return ((acc / sumw)^(1 / p)) / float(task.agg_norm)
        end
    end

    error("_slice_pair_distance_geometric_packed_fast: unsupported reduction")
end

function _slice_pair_distance_geometric(
    M::PModule{K},
    N::PModule{K},
    dirs_in,
    offs0,
    W::AbstractMatrix{Float64},
    pi::PLikeEncodingMap,
    opts_chain::InvariantOptions,
    filtered::NamedTuple,
    task::SliceDistanceTask;
    drop_unknown::Bool = true,
) where {K}
    nd = length(dirs_in)
    no = length(offs0)
    ns = nd * no
    agg_mode = (task.agg === mean) ? :mean : (task.agg === maximum) ? :max : task.agg
    if _SLICE_USE_PACKED_DISTANCE_FASTPATH[] &&
       _slice_distance_fast_kind(task) != :none &&
       agg_mode in (:mean, :pmean, :max)
        return _slice_pair_distance_geometric_packed_fast(
            M, N, dirs_in, offs0, W, pi, opts_chain, filtered, task;
            drop_unknown = drop_unknown,
        )
    end
    barsM = _packed_grid_undef(PackedFloatBarcode, nd, no)
    barsN = _packed_grid_undef(PackedFloatBarcode, nd, no)

    build_cache!(M.Q; cover=true, updown=true)
    build_cache!(N.Q; cover=true, updown=true)
    ccM = _get_cover_cache(M.Q)
    ccN = _get_cover_cache(N.Q)
    nQM = nvertices(M.Q)
    nQN = nvertices(N.Q)
    use_array_memo_M = _use_array_memo(nQM)
    use_array_memo_N = _use_array_memo(nQN)
    threads = task.threads && Threads.nthreads() > 1

    if threads
        nT = Threads.nthreads()
        memoM_by_thread = use_array_memo_M ?
            [_new_array_memo(K, nQM) for _ in 1:nT] :
            [Dict{Tuple{Int,Int}, AbstractMatrix{K}}() for _ in 1:nT]
        memoN_by_thread = use_array_memo_N ?
            [_new_array_memo(K, nQN) for _ in 1:nT] :
            [Dict{Tuple{Int,Int}, AbstractMatrix{K}}() for _ in 1:nT]
        rankM_by_thread = [Matrix{Int}(undef, 0, 0) for _ in 1:nT]
        rankN_by_thread = [Matrix{Int}(undef, 0, 0) for _ in 1:nT]
        locate_ws_by_thread = [_SliceLocateBatchWorkspace() for _ in 1:nT]
        Threads.@threads for idx in 1:ns
            i = div(idx - 1, no) + 1
            j = (idx - 1) % no + 1
            tid = Threads.threadid()
            if W[idx] == 0.0
                barsM[idx] = _empty_packed_float_barcode()
                barsN[idx] = _empty_packed_float_barcode()
                continue
            end
            chain, tvals = _slice_chain_collect(
                pi,
                offs0[j],
                dirs_in[i],
                opts_chain;
                drop_unknown = drop_unknown,
                locate_ws = locate_ws_by_thread[tid],
                filtered...,
            )
            if isempty(chain)
                barsM[idx] = _empty_packed_float_barcode()
                barsN[idx] = _empty_packed_float_barcode()
                continue
            end
            m = length(chain)
            if size(rankM_by_thread[tid], 1) < m
                rankM_by_thread[tid] = Matrix{Int}(undef, m, m)
            end
            if size(rankN_by_thread[tid], 1) < m
                rankN_by_thread[tid] = Matrix{Int}(undef, m, m)
            end
            endpoints = _extended_values_view(tvals)
            barsM[idx] = _slice_barcode_packed_with_workspace(M, chain, endpoints, ccM, memoM_by_thread[tid], rankM_by_thread[tid])
            barsN[idx] = _slice_barcode_packed_with_workspace(N, chain, endpoints, ccN, memoN_by_thread[tid], rankN_by_thread[tid])
        end
    else
        memoM = use_array_memo_M ? _new_array_memo(K, nQM) : Dict{Tuple{Int,Int}, AbstractMatrix{K}}()
        memoN = use_array_memo_N ? _new_array_memo(K, nQN) : Dict{Tuple{Int,Int}, AbstractMatrix{K}}()
        rankM = Matrix{Int}(undef, 0, 0)
        rankN = Matrix{Int}(undef, 0, 0)
        locate_ws = _SliceLocateBatchWorkspace()
        @inbounds for idx in 1:ns
            i = div(idx - 1, no) + 1
            j = (idx - 1) % no + 1
            if W[idx] == 0.0
                barsM[idx] = _empty_packed_float_barcode()
                barsN[idx] = _empty_packed_float_barcode()
                continue
            end
            chain, tvals = _slice_chain_collect(
                pi,
                offs0[j],
                dirs_in[i],
                opts_chain;
                drop_unknown = drop_unknown,
                locate_ws = locate_ws,
                filtered...,
            )
            if isempty(chain)
                barsM[idx] = _empty_packed_float_barcode()
                barsN[idx] = _empty_packed_float_barcode()
                continue
            end
            m = length(chain)
            if size(rankM, 1) < m
                rankM = Matrix{Int}(undef, m, m)
            end
            if size(rankN, 1) < m
                rankN = Matrix{Int}(undef, m, m)
            end
            endpoints = _extended_values_view(tvals)
            barsM[idx] = _slice_barcode_packed_with_workspace(M, chain, endpoints, ccM, memoM, rankM)
            barsN[idx] = _slice_barcode_packed_with_workspace(N, chain, endpoints, ccN, memoN, rankN)
        end
    end

    return _run_slice_distance_from_barcodes(barsM, barsN, W, task)
end

function slice_barcodes(
    M::PModule{K},
    pi::PLikeEncodingMap;
    opts::InvariantOptions = InvariantOptions(),
    directions = :auto,
    offsets = :auto,
    n_dirs::Integer = 16,
    n_offsets::Integer = 9,
    max_den::Integer = 8,
    include_axes::Bool = false,
    normalize_dirs::Symbol = :none,
    direction_weight::Union{Symbol,Function,Real} = :none,
    offset_weights = nothing,
    normalize_weights::Bool = true,
    offset_margin::Real = 0.05,
    drop_unknown::Bool = true,
    values = nothing,
    threads::Bool = (Threads.nthreads() > 1),
    packed::Bool = false,
    cache::Union{Nothing,SlicePlanCache} = nothing,
    slice_kwargs...
) where {K}
    prep = _prepare_geometric_slice_query(
        pi,
        opts;
        directions = directions,
        offsets = offsets,
        n_dirs = n_dirs,
        n_offsets = n_offsets,
        max_den = max_den,
        include_axes = include_axes,
        normalize_dirs = normalize_dirs,
        direction_weight = direction_weight,
        offset_weights = offset_weights,
        normalize_weights = normalize_weights,
        offset_margin = offset_margin,
        slice_kwargs...,
    )

    if isempty(prep.dirs_in)
        bars0 = packed ? _packed_grid_undef(PackedFloatBarcode, 0, 0) : Matrix{FloatBarcode}(undef, 0, 0)
        return _slice_barcodes_result(bars0, prep.weights, prep.dirs_in, prep.offs)
    end

    if values === nothing
        if cache === nothing
            bars = _slice_barcodes_geometric_packed(
                M,
                prep.dirs_in,
                prep.offs,
                pi,
                prep.opts_chain,
                prep.filtered;
                drop_unknown = drop_unknown,
                threads = threads,
            )
            return _slice_barcodes_result_from_packed(bars, prep.weights, prep.dirs_in, prep.offs; packed=packed)
        end
        plan = compile_slices(
            pi,
            opts;
            directions = directions,
            offsets = offsets,
            n_dirs = n_dirs,
            n_offsets = n_offsets,
            max_den = max_den,
            include_axes = include_axes,
            normalize_dirs = normalize_dirs,
            direction_weight = direction_weight,
            offset_weights = offset_weights,
            normalize_weights = normalize_weights,
            offset_margin = offset_margin,
            drop_unknown = drop_unknown,
            threads = threads,
            cache = cache,
            slice_kwargs...,
        )
        return run_invariants(plan, module_cache(M), SliceBarcodesTask(; packed = packed, threads = threads))
    end

    # Determine default directions/offsets if requested.
    dirs0 = directions
    offs0 = offsets

    if dirs0 === :auto || dirs0 === nothing
        dirs0 = default_directions(pi;
                                  n_dirs = n_dirs,
                                  max_den = max_den,
                                  include_axes = include_axes,
                                  normalize = (normalize_dirs == :none ? :none : normalize_dirs))
    end

    opts_offsets = InvariantOptions(box = opts_compile.box, pl_mode = opts_compile.pl_mode)
    opts_chain   = opts_compile
    tmin0 = get(slice_kwargs0, :tmin, nothing)
    tmax0 = get(slice_kwargs0, :tmax, nothing)
    if opts_chain.box === nothing && tmin0 === nothing && tmax0 === nothing
        opts_offsets = InvariantOptions(box = :auto, pl_mode = opts_compile.pl_mode)
        opts_chain   = InvariantOptions(box = :auto, strict = opts_compile.strict,
                                        threads = opts.threads, axes = opts.axes,
                                        axes_policy = opts.axes_policy, max_axis_len = opts.max_axis_len,
                                        pl_mode = opts_compile.pl_mode)
    end

    filtered = (;
        (k => v for (k, v) in pairs(slice_kwargs0)
            if k != :default_weight && k != :kmin && k != :kmax_param && k != :lengthscale)...)

    if offs0 === :auto || offs0 === nothing
        offs0 = default_offsets(pi, opts_offsets; n_offsets = n_offsets, margin = offset_margin)
    end

    # Degenerate case: empty-dimensional offsets (e.g., missing witnesses).
    if !isempty(offs0) && length(offs0[1]) == 0
        dirT = eltype(dirs0)
        bars0 = packed ? _packed_grid_undef(PackedFloatBarcode, 0, 0) : Matrix{FloatBarcode}(undef, 0, 0)
        return _slice_barcodes_result(bars0, zeros(Float64, 0, 0), Vector{dirT}(undef, 0), offs0)
    end

    if !isempty(dirs0) && !isempty(offs0) && length(dirs0[1]) != length(offs0[1])
        dirs0 = default_directions(length(offs0[1]);
                                  n_dirs = n_dirs,
                                  max_den = max_den,
                                  include_axes = include_axes,
                                  normalize = (normalize_dirs == :none ? :none : normalize_dirs))
    end

    # Normalize directions (e.g. L1) if requested.
    dirs_in = [_normalize_dir(dir, normalize_dirs) for dir in dirs0]

    nd = length(dirs_in)
    no = length(offs0)
    nd > 0 || error("slice_barcodes: directions is empty")
    no > 0 || error("slice_barcodes: offsets is empty")

    # Slice weights (outer product of per-direction and per-offset weights).
    wdir = Vector{Float64}(undef, nd)
    for i in 1:nd
        wdir[i] = SliceInvariants.direction_weight(dirs_in[i], direction_weight)
    end
    woff = _offset_sample_weights(offs0, offset_weights)

    W = wdir * woff'
    if normalize_weights
        s = sum(W)
        if s > 0
            W ./= s
        end
    end

    if packed
        if values === nothing || _values_are_float_vector(values)
            bcs = Matrix{PackedFloatBarcode}(undef, nd, no)
            if threads && Threads.nthreads() > 1
                Threads.@threads for k in 1:(nd * no)
                    i = div((k - 1), no) + 1
                    j = (k - 1) % no + 1
                    chain, tvals = slice_chain(pi, offs0[j], dirs_in[i], opts_chain; drop_unknown=drop_unknown, filtered...)
                    if isempty(chain) || isempty(tvals)
                        bcs[i, j] = _empty_packed_float_barcode()
                        continue
                    end
                    vals_use = values === nothing ? tvals : values
                    bcs[i, j] = _slice_barcode_packed(M, chain; values = vals_use, check_chain = false)::PackedFloatBarcode
                end
            else
                for i in 1:nd, j in 1:no
                    chain, tvals = slice_chain(pi, offs0[j], dirs_in[i], opts_chain; drop_unknown=drop_unknown, filtered...)
                    if isempty(chain) || isempty(tvals)
                        bcs[i, j] = _empty_packed_float_barcode()
                        continue
                    end
                    vals_use = values === nothing ? tvals : values
                    bcs[i, j] = _slice_barcode_packed(M, chain; values = vals_use, check_chain = false)::PackedFloatBarcode
                end
            end
            return _slice_barcodes_result(_packed_grid_from_matrix(bcs), W, dirs_in, offs0)
        elseif _values_are_int_vector(values)
            bcs = Matrix{PackedIndexBarcode}(undef, nd, no)
            if threads && Threads.nthreads() > 1
                Threads.@threads for k in 1:(nd * no)
                    i = div((k - 1), no) + 1
                    j = (k - 1) % no + 1
                    chain, tvals = slice_chain(pi, offs0[j], dirs_in[i], opts_chain; drop_unknown=drop_unknown, filtered...)
                    if isempty(chain) || isempty(tvals)
                        bcs[i, j] = _empty_packed_index_barcode()
                        continue
                    end
                    bcs[i, j] = _slice_barcode_packed(M, chain; values = values, check_chain = false)::PackedIndexBarcode
                end
            else
                for i in 1:nd, j in 1:no
                    chain, tvals = slice_chain(pi, offs0[j], dirs_in[i], opts_chain; drop_unknown=drop_unknown, filtered...)
                    if isempty(chain) || isempty(tvals)
                        bcs[i, j] = _empty_packed_index_barcode()
                        continue
                    end
                    bcs[i, j] = _slice_barcode_packed(M, chain; values = values, check_chain = false)::PackedIndexBarcode
                end
            end
            return _slice_barcodes_result(_packed_grid_from_matrix(bcs), W, dirs_in, offs0)
        else
            bcs = Matrix{PackedFloatBarcode}(undef, nd, no)
            if threads && Threads.nthreads() > 1
                Threads.@threads for k in 1:(nd * no)
                    i = div((k - 1), no) + 1
                    j = (k - 1) % no + 1
                    chain, tvals = slice_chain(pi, offs0[j], dirs_in[i], opts_chain; drop_unknown=drop_unknown, filtered...)
                    if isempty(chain) || isempty(tvals)
                        bcs[i, j] = _empty_packed_float_barcode()
                        continue
                    end
                    vals_use = values === nothing ? tvals : values
                    pb = _slice_barcode_packed(M, chain; values = vals_use, check_chain = false)
                    bcs[i, j] = pb isa PackedFloatBarcode ? pb : _pack_float_barcode(_barcode_from_packed(pb))
                end
            else
                for i in 1:nd, j in 1:no
                    chain, tvals = slice_chain(pi, offs0[j], dirs_in[i], opts_chain; drop_unknown=drop_unknown, filtered...)
                    if isempty(chain) || isempty(tvals)
                        bcs[i, j] = _empty_packed_float_barcode()
                        continue
                    end
                    vals_use = values === nothing ? tvals : values
                    pb = _slice_barcode_packed(M, chain; values = vals_use, check_chain = false)
                    bcs[i, j] = pb isa PackedFloatBarcode ? pb : _pack_float_barcode(_barcode_from_packed(pb))
                end
            end
            return _slice_barcodes_result(_packed_grid_from_matrix(bcs), W, dirs_in, offs0)
        end
    end

    if values === nothing || _values_are_float_vector(values)
        bcs = Matrix{FloatBarcode}(undef, nd, no)
        if threads && Threads.nthreads() > 1
            Threads.@threads for k in 1:(nd * no)
                i = div((k - 1), no) + 1
                j = (k - 1) % no + 1
                chain, tvals = slice_chain(pi, offs0[j], dirs_in[i], opts_chain; drop_unknown=drop_unknown, filtered...)
                if isempty(chain) || isempty(tvals)
                    bcs[i, j] = _empty_float_barcode()
                    continue
                end
                vals_use = values === nothing ? tvals : values
                bcs[i, j] = slice_barcode(M, chain; values = vals_use, check_chain = false)
            end
        else
            for i in 1:nd, j in 1:no
                chain, tvals = slice_chain(pi, offs0[j], dirs_in[i], opts_chain; drop_unknown=drop_unknown, filtered...)
                if isempty(chain) || isempty(tvals)
                    bcs[i, j] = _empty_float_barcode()
                    continue
                end
                vals_use = values === nothing ? tvals : values
                bcs[i, j] = slice_barcode(M, chain; values = vals_use, check_chain = false)
            end
        end
        return _slice_barcodes_result(bcs, W, dirs_in, offs0)
    elseif _values_are_int_vector(values)
        bcs = Matrix{IndexBarcode}(undef, nd, no)
        if threads && Threads.nthreads() > 1
            Threads.@threads for k in 1:(nd * no)
                i = div((k - 1), no) + 1
                j = (k - 1) % no + 1
                chain, tvals = slice_chain(pi, offs0[j], dirs_in[i], opts_chain; drop_unknown=drop_unknown, filtered...)
                if isempty(chain) || isempty(tvals)
                    bcs[i, j] = _empty_index_barcode()
                    continue
                end
                bcs[i, j] = slice_barcode(M, chain; values = values, check_chain = false)
            end
        else
            for i in 1:nd, j in 1:no
                chain, tvals = slice_chain(pi, offs0[j], dirs_in[i], opts_chain; drop_unknown=drop_unknown, filtered...)
                if isempty(chain) || isempty(tvals)
                    bcs[i, j] = _empty_index_barcode()
                    continue
                end
                bcs[i, j] = slice_barcode(M, chain; values = values, check_chain = false)
            end
        end
        return _slice_barcodes_result(bcs, W, dirs_in, offs0)
    else
        bcs = Matrix{FloatBarcode}(undef, nd, no)
        if threads && Threads.nthreads() > 1
            Threads.@threads for k in 1:(nd * no)
                i = div((k - 1), no) + 1
                j = (k - 1) % no + 1
                chain, tvals = slice_chain(pi, offs0[j], dirs_in[i], opts_chain; drop_unknown=drop_unknown, filtered...)
                if isempty(chain) || isempty(tvals)
                    bcs[i, j] = _empty_float_barcode()
                    continue
                end
                vals_use = values === nothing ? tvals : values
                bcs[i, j] = _to_float_barcode(slice_barcode(M, chain; values = vals_use, check_chain = false))
            end
        else
            for i in 1:nd, j in 1:no
                chain, tvals = slice_chain(pi, offs0[j], dirs_in[i], opts_chain; drop_unknown=drop_unknown, filtered...)
                if isempty(chain) || isempty(tvals)
                    bcs[i, j] = _empty_float_barcode()
                    continue
                end
                vals_use = values === nothing ? tvals : values
                bcs[i, j] = _to_float_barcode(slice_barcode(M, chain; values = vals_use, check_chain = false))
            end
        end
        return _slice_barcodes_result(bcs, W, dirs_in, offs0)
    end
end

# -----------------------------------------------------------------------------
# Opts-aware slice_barcodes adapter
# -----------------------------------------------------------------------------

function slice_barcodes(M, pi::ZnEncodingMap, opts::InvariantOptions; kwargs...)
    # Explicit kwargs override opts.
    box     = get(kwargs, :box, opts.box)
    strict  = get(kwargs, :strict, opts.strict)
    threads = get(kwargs, :threads, opts.threads)

    strict0  = _default_strict(strict)
    threads0 = _default_threads(threads)

    # Avoid passing duplicates downstream.
    kwargs2 = _drop_keys(kwargs, (:box, :strict, :threads))

    return slice_barcodes(M, pi;
        box = box,
        strict = strict0,
        threads = threads0,
        kwargs2...
    )
end

slice_barcodes(M, pi::CompiledEncoding{<:ZnEncodingMap}, opts::InvariantOptions; kwargs...) =
    slice_barcodes(M, pi.pi, opts; kwargs...)



# Bounds from a collection of barcodes.
function _barcode_bounds_collection(bcs)
    bmin = Inf
    dmax = -Inf
    pmax = 0.0
    for bc in bcs
        for (bd, mult) in bc
            mult <= 0 && continue
            b0, d0 = bd
            b = float(b0)
            d = float(d0)
            d > b || continue
            bmin = min(bmin, b)
            dmax = max(dmax, d)
            pmax = max(pmax, d - b)
        end
    end
    if bmin == Inf
        bmin = 0.0
        dmax = 1.0
        pmax = 1.0
    end
    return bmin, dmax, pmax
end

# Convert a user featurizer output to a numeric vector.
function _as_feature_vector(x)::Vector{Float64}
    if x isa Real
        return Float64[float(x)]
    elseif x isa AbstractVector
        return Float64[float(v) for v in x]
    else
        error("featurizer must return a real or a vector of reals; got $(typeof(x))")
    end
end

# Default t-grid (global, so features align across slices).
function _default_tgrid_from_barcodes(bcs; nsteps::Int=401)::Vector{Float64}
    bmin, dmax, _ = _barcode_bounds_collection(bcs)
    nsteps < 2 && (nsteps = 2)
    if bmin == dmax
        bmin -= 0.5
        dmax += 0.5
    end
    return collect(range(bmin, dmax; length=nsteps))
end

# Default image grids (global, so features align across slices).
function _default_image_grids_from_barcodes(
    bcs;
    img_xgrid=nothing,
    img_ygrid=nothing,
    img_birth_range=nothing,
    img_pers_range=nothing,
    img_nbirth::Int=20,
    img_npers::Int=20
)
    bmin, dmax, pmax = _barcode_bounds_collection(bcs)

    xg = if img_xgrid === nothing
        br = (img_birth_range === nothing) ? (bmin, dmax) : img_birth_range
        _grid_from_range(br, img_nbirth)
    else
        Float64[float(x) for x in collect(img_xgrid)]
    end

    yg = if img_ygrid === nothing
        pr = (img_pers_range === nothing) ? (0.0, pmax) : img_pers_range
        _grid_from_range(pr, img_npers)
    else
        Float64[float(y) for y in collect(img_ygrid)]
    end

    return xg, yg
end

# Aggregate per-slice feature vectors (explicit slices).
function _aggregate_feature_vectors(
    feats::AbstractVector{<:AbstractVector{Float64}},
    W::AbstractVector{Float64};
    aggregate::Symbol=:mean,
    unwrap_scalar::Bool=true
)
    ns = length(feats)
    ns > 0 || error("_aggregate_feature_vectors: empty feature list")
    d = length(feats[1])

    # Shape check
    for i in 2:ns
        length(feats[i]) == d || error("_aggregate_feature_vectors: feature length mismatch")
    end

    if aggregate == :stack
        A = zeros(Float64, ns, d)
        for i in 1:ns
            for j in 1:d
                A[i, j] = feats[i][j]
            end
        end
        if unwrap_scalar && d == 1
            return vec(A)
        end
        return A
    elseif aggregate == :sum || aggregate == :mean
        acc = zeros(Float64, d)
        sumw = sum(W)
        for i in 1:ns
            w = W[i]
            w == 0.0 && continue
            for j in 1:d
                acc[j] += w * feats[i][j]
            end
        end
        if aggregate == :mean
            sumw > 0.0 || error("slice_features: total weight is zero")
            acc ./= sumw
        end
        if unwrap_scalar && d == 1
            return acc[1]
        end
        return acc
    else
        error("slice_features: aggregate must be :mean, :sum, or :stack")
    end
end

# Aggregate per-slice feature vectors (geometric slices, 2D array of slices).
function _aggregate_feature_vectors(
    feats::AbstractMatrix{<:AbstractVector{Float64}},
    W::AbstractMatrix{Float64};
    aggregate::Symbol=:mean,
    unwrap_scalar::Bool=true
)
    nd, no = size(feats)
    d = length(feats[1, 1])

    # Shape check
    for i in 1:nd, j in 1:no
        length(feats[i, j]) == d || error("_aggregate_feature_vectors: feature length mismatch")
    end

    if aggregate == :stack
        A = zeros(Float64, nd, no, d)
        for i in 1:nd, j in 1:no
            for k in 1:d
                A[i, j, k] = feats[i, j][k]
            end
        end
        if unwrap_scalar && d == 1
            B = zeros(Float64, nd, no)
            for i in 1:nd, j in 1:no
                B[i, j] = A[i, j, 1]
            end
            return B
        end
        return A
    elseif aggregate == :sum || aggregate == :mean
        acc = zeros(Float64, d)
        sumw = sum(W)
        for i in 1:nd, j in 1:no
            w = W[i, j]
            w == 0.0 && continue
            for k in 1:d
                acc[k] += w * feats[i, j][k]
            end
        end
        if aggregate == :mean
            sumw > 0.0 || error("slice_features: total weight is zero")
            acc ./= sumw
        end
        if unwrap_scalar && d == 1
            return acc[1]
        end
        return acc
    else
        error("slice_features: aggregate must be :mean, :sum, or :stack")
    end
end

"""
    slice_features(M, slices; featurizer=:landscape, aggregate=:mean, ...) -> SliceFeaturesResult
    slice_features(M, pi; directions, offsets, featurizer=:landscape, aggregate=:mean, ...) -> SliceFeaturesResult

Compute multiparameter "slice features" by:
1) slicing a multi-parameter module into many 1D modules (fibers/lines),
2) computing a 1D featurization per slice,
3) aggregating across slices.

This supports the standard ML pipeline described in the multiparameter
literature (slice barcodes -> aggregate 1D vectorizations).

Supported `featurizer` values:
- `:landscape`  -> flatten persistence landscapes on a common `tgrid`
- `:image`      -> flatten persistence images on common grids
- `:silhouette` -> persistence silhouette on a common `tgrid`
- `:entropy`    -> persistent entropy (scalar)
- `:summary`    -> a small vector of summary stats
- or any function `bc -> real_or_vector`

Aggregation modes (`aggregate`):
- `:mean`  -> weighted average
- `:sum`   -> weighted sum
- `:stack` -> return per-slice features (matrix for explicit slices, 3D array for geometric slices)

Key options:
- `normalize_weights`: normalize slice weights (default true)
- `unwrap_scalar`: return scalar when the feature dimension is 1 (default true)

Returns a [`SliceFeaturesResult`](@ref). Use [`slice_features`](@ref) on the
result for the feature payload and [`slice_weights`](@ref) for the aligned
slice weights.
"""
function slice_features(
    M::PModule{K},
    slices::AbstractVector;
    featurizer=:landscape,
    aggregate::Symbol=:mean,
    default_weight::Real=1.0,
    normalize_weights::Bool=true,
    unwrap_scalar::Bool=true,
    threads::Bool = (Threads.nthreads() > 1),
    # Landscape / silhouette defaults:
    kmax::Int=5,
    tgrid=nothing,
    tgrid_nsteps::Int=401,
    # Silhouette options:
    sil_weighting=:persistence,
    sil_p::Real=1,
    sil_normalize::Bool=true,
    # Image options:
    img_xgrid=nothing,
    img_ygrid=nothing,
    img_birth_range=nothing,
    img_pers_range=nothing,
    img_nbirth::Int=20,
    img_npers::Int=20,
    img_sigma::Real=1.0,
    img_coords::Symbol=:birth_persistence,
    img_weighting=:persistence,
    img_p::Real=1,
    img_normalize::Symbol=:none,
    # Entropy options:
    entropy_normalize::Bool=true,
    entropy_weighting=:persistence,
    entropy_p::Real=1,
    # Summary options:
    summary_fields=_DEFAULT_BARCODE_SUMMARY_FIELDS,
    summary_normalize_entropy::Bool=true
) where {K}
    specs = collect_slices(slices; default_weight=default_weight)
    data = slice_barcodes(M, specs;
        normalize_weights=normalize_weights,
        threads=threads,
        packed=true)
    bcs = data.barcodes
    W = data.weights

    # Global grids if needed
    tg = tgrid
    if (featurizer == :landscape || featurizer == :silhouette) && tg === nothing
        tg = _default_tgrid_from_barcodes(bcs; nsteps=tgrid_nsteps)
    end
    if tg !== nothing
        tg = _clean_tgrid(tg)
    end

    xg = nothing
    yg = nothing
    if featurizer == :image
        xg, yg = _default_image_grids_from_barcodes(
            bcs;
            img_xgrid=img_xgrid,
            img_ygrid=img_ygrid,
            img_birth_range=img_birth_range,
            img_pers_range=img_pers_range,
            img_nbirth=img_nbirth,
            img_npers=img_npers
        )
    end

    feats = Vector{Vector{Float64}}(undef, length(bcs))
    if threads && Threads.nthreads() > 1
        Threads.@threads for i in 1:length(bcs)
            bc = bcs[i]
            if featurizer == :landscape
                pl = persistence_landscape(bc; kmax=kmax, tgrid=tg)
                feats[i] = _landscape_feature_vector(pl)
            elseif featurizer == :silhouette
                s = persistence_silhouette(bc; tgrid=tg, weighting=sil_weighting, p=sil_p, normalize=sil_normalize)
                feats[i] = Float64[s...]
            elseif featurizer == :image
                PI = persistence_image(bc;
                                       xgrid=xg,
                                       ygrid=yg,
                                       sigma=img_sigma,
                                       coords=img_coords,
                                       weighting=img_weighting,
                                       p=img_p,
                                       normalize=img_normalize)
                feats[i] = _image_feature_vector(PI)
            elseif featurizer == :entropy
                e = barcode_entropy(bc; normalize=entropy_normalize, weighting=entropy_weighting, p=entropy_p)
                feats[i] = Float64[float(e)]
            elseif featurizer == :summary
                feats[i] = _barcode_summary_vector(bc; fields=summary_fields, normalize_entropy=summary_normalize_entropy)
            elseif featurizer isa Function
                feats[i] = _as_feature_vector(featurizer(bc))
            else
                error("slice_features: unknown featurizer $(featurizer)")
            end
        end
    else
        for i in 1:length(bcs)
            bc = bcs[i]
            if featurizer == :landscape
                pl = persistence_landscape(bc; kmax=kmax, tgrid=tg)
                feats[i] = _landscape_feature_vector(pl)
            elseif featurizer == :silhouette
                s = persistence_silhouette(bc; tgrid=tg, weighting=sil_weighting, p=sil_p, normalize=sil_normalize)
                feats[i] = Float64[s...]
            elseif featurizer == :image
                PI = persistence_image(bc;
                                       xgrid=xg,
                                       ygrid=yg,
                                       sigma=img_sigma,
                                       coords=img_coords,
                                       weighting=img_weighting,
                                       p=img_p,
                                       normalize=img_normalize)
                feats[i] = _image_feature_vector(PI)
            elseif featurizer == :entropy
                e = barcode_entropy(bc; normalize=entropy_normalize, weighting=entropy_weighting, p=entropy_p)
                feats[i] = Float64[float(e)]
            elseif featurizer == :summary
                feats[i] = _barcode_summary_vector(bc; fields=summary_fields, normalize_entropy=summary_normalize_entropy)
            elseif featurizer isa Function
                feats[i] = _as_feature_vector(featurizer(bc))
            else
                error("slice_features: unknown featurizer $(featurizer)")
            end
        end
    end

    out = _aggregate_feature_vectors(feats, W; aggregate=aggregate, unwrap_scalar=unwrap_scalar)
    return _slice_features_result(out, W; featurizer=featurizer, aggregate=aggregate)
end

slice_features(M::PModule{K}, chain::AbstractVector{Int}; kwargs...) where {K} =
    slice_features(M, [chain]; kwargs...)

function slice_features(
    M::PModule{K},
    plan::CompiledSlicePlan;
    featurizer=:landscape,
    aggregate::Symbol=:mean,
    normalize_weights::Bool=true,
    unwrap_scalar::Bool=true,
    threads::Bool = (Threads.nthreads() > 1),
    # Landscape / silhouette defaults:
    kmax::Int=5,
    tgrid=nothing,
    tgrid_nsteps::Int=401,
    # Silhouette options:
    sil_weighting=:persistence,
    sil_p::Real=1,
    sil_normalize::Bool=true,
    # Image options:
    img_xgrid=nothing,
    img_ygrid=nothing,
    img_birth_range=nothing,
    img_pers_range=nothing,
    img_nbirth::Int=20,
    img_npers::Int=20,
    img_sigma::Real=1.0,
    img_coords::Symbol=:birth_persistence,
    img_weighting=:persistence,
    img_p::Real=1,
    img_normalize::Symbol=:none,
    # Entropy options:
    entropy_normalize::Bool=true,
    entropy_weighting=:persistence,
    entropy_p::Real=1,
    # Summary options:
    summary_fields=_DEFAULT_BARCODE_SUMMARY_FIELDS,
    summary_normalize_entropy::Bool=true
) where {K}
    cacheM = module_cache(M)
    data = slice_barcodes(M, plan; packed=true, threads=threads)
    bcs = data.barcodes
    W = normalize_weights ? data.weights : copy(data.weights)

    # Global grids if needed
    tg = tgrid
    if (featurizer == :landscape || featurizer == :silhouette) && tg === nothing
        tg = _default_tgrid_from_barcodes(bcs; nsteps=tgrid_nsteps)
    end
    if tg !== nothing
        tg = _clean_tgrid(tg)
    end

    if featurizer == :landscape &&
       _SLICE_USE_LANDSCAPE_FEATURE_CACHE[] &&
       bcs isa PackedBarcodeGrid{PackedFloatBarcode}
        feats = _landscape_feature_cache(cacheM, plan, bcs, tg, kmax; threads=threads)
        out = _aggregate_feature_vectors(reshape(feats, size(bcs)...), W; aggregate=aggregate, unwrap_scalar=unwrap_scalar)
        return _slice_features_result(out, W; featurizer=featurizer, aggregate=aggregate)
    end

    xg = nothing
    yg = nothing
    if featurizer == :image
        xg, yg = _default_image_grids_from_barcodes(
            bcs;
            img_xgrid=img_xgrid,
            img_ygrid=img_ygrid,
            img_birth_range=img_birth_range,
            img_pers_range=img_pers_range,
            img_nbirth=img_nbirth,
            img_npers=img_npers
        )
    end

    nd, no = size(bcs)
    feats = Array{Vector{Float64}}(undef, nd, no)
    if threads && Threads.nthreads() > 1
        Threads.@threads for k in 1:(nd * no)
            i = div(k - 1, no) + 1
            j = (k - 1) % no + 1
            bc = bcs[i, j]
            if featurizer == :landscape
                pl = persistence_landscape(bc; kmax=kmax, tgrid=tg)
                feats[i, j] = _landscape_feature_vector(pl)
            elseif featurizer == :silhouette
                s = persistence_silhouette(bc; tgrid=tg, weighting=sil_weighting, p=sil_p, normalize=sil_normalize)
                feats[i, j] = Float64[s...]
            elseif featurizer == :image
                PI = persistence_image(bc;
                                       xgrid=xg,
                                       ygrid=yg,
                                       sigma=img_sigma,
                                       coords=img_coords,
                                       weighting=img_weighting,
                                       p=img_p,
                                       normalize=img_normalize)
                feats[i, j] = _image_feature_vector(PI)
            elseif featurizer == :entropy
                e = barcode_entropy(bc; normalize=entropy_normalize, weighting=entropy_weighting, p=entropy_p)
                feats[i, j] = Float64[float(e)]
            elseif featurizer == :summary
                feats[i, j] = _barcode_summary_vector(bc; fields=summary_fields, normalize_entropy=summary_normalize_entropy)
            elseif featurizer isa Function
                feats[i, j] = _as_feature_vector(featurizer(bc))
            else
                error("slice_features: unknown featurizer $(featurizer)")
            end
        end
    else
        for i in 1:nd, j in 1:no
            bc = bcs[i, j]
            if featurizer == :landscape
                pl = persistence_landscape(bc; kmax=kmax, tgrid=tg)
                feats[i, j] = _landscape_feature_vector(pl)
            elseif featurizer == :silhouette
                s = persistence_silhouette(bc; tgrid=tg, weighting=sil_weighting, p=sil_p, normalize=sil_normalize)
                feats[i, j] = Float64[s...]
            elseif featurizer == :image
                PI = persistence_image(bc;
                                       xgrid=xg,
                                       ygrid=yg,
                                       sigma=img_sigma,
                                       coords=img_coords,
                                       weighting=img_weighting,
                                       p=img_p,
                                       normalize=img_normalize)
                feats[i, j] = _image_feature_vector(PI)
            elseif featurizer == :entropy
                e = barcode_entropy(bc; normalize=entropy_normalize, weighting=entropy_weighting, p=entropy_p)
                feats[i, j] = Float64[float(e)]
            elseif featurizer == :summary
                feats[i, j] = _barcode_summary_vector(bc; fields=summary_fields, normalize_entropy=summary_normalize_entropy)
            elseif featurizer isa Function
                feats[i, j] = _as_feature_vector(featurizer(bc))
            else
                error("slice_features: unknown featurizer $(featurizer)")
            end
        end
    end

    out = _aggregate_feature_vectors(feats, W; aggregate=aggregate, unwrap_scalar=unwrap_scalar)
    return _slice_features_result(out, W; featurizer=featurizer, aggregate=aggregate)
end

function slice_features(
    M::PModule{K},
    pi;
    opts::InvariantOptions=InvariantOptions(),
    directions=nothing,
    offsets=nothing,
    featurizer=:landscape,
    aggregate::Symbol=:mean,
    normalize_weights::Bool=true,
    unwrap_scalar::Bool=true,
    threads::Bool = (Threads.nthreads() > 1),
    # Geometric slice parameters:
    tmin::Union{Real,Nothing}=nothing,
    tmax::Union{Real,Nothing}=nothing,
    nsteps::Int=401,
    kmin=nothing,
    kmax_param=nothing,
    strict::Bool=true,
    drop_unknown::Bool=true,
    dedup::Bool=true,
    normalize_dirs::Symbol=:none,
    direction_weight::Symbol=:none,
    offset_weights=nothing,
    # Landscape / silhouette defaults:
    kmax::Int=5,
    tgrid=nothing,
    tgrid_nsteps::Int=401,
    # Silhouette options:
    sil_weighting=:persistence,
    sil_p::Real=1,
    sil_normalize::Bool=true,
    # Image options:
    img_xgrid=nothing,
    img_ygrid=nothing,
    img_birth_range=nothing,
    img_pers_range=nothing,
    img_nbirth::Int=20,
    img_npers::Int=20,
    img_sigma::Real=1.0,
    img_coords::Symbol=:birth_persistence,
    img_weighting=:persistence,
    img_p::Real=1,
    img_normalize::Symbol=:none,
    # Entropy options:
    entropy_normalize::Bool=true,
    entropy_weighting=:persistence,
    entropy_p::Real=1,
    # Summary options:
    summary_fields=_DEFAULT_BARCODE_SUMMARY_FIELDS,
    summary_normalize_entropy::Bool=true,
    cache::Union{Nothing,SlicePlanCache}=nothing,
    slice_kwargs...
) where {K}
    strict0 = opts.strict === nothing ? strict : opts.strict
    opts0 = InvariantOptions(
        axes=opts.axes,
        axes_policy=opts.axes_policy,
        max_axis_len=opts.max_axis_len,
        box=opts.box,
        threads=opts.threads,
        strict=strict0,
        pl_mode=opts.pl_mode,
    )
    data = slice_barcodes(M, pi;
                          opts=opts0,
                          directions=directions,
                          offsets=offsets,
                          tmin=tmin,
                          tmax=tmax,
                          nsteps=nsteps,
                          kmin=kmin,
                          kmax_param=kmax_param,
                          drop_unknown=drop_unknown,
                          dedup=dedup,
                          normalize_dirs=normalize_dirs,
                          direction_weight=direction_weight,
                          offset_weights=offset_weights,
                          normalize_weights=normalize_weights,
                          cache=cache,
                          packed=true,
                          slice_kwargs...)

    bcs = data.barcodes
    W = data.weights

    # Global grids if needed
    tg = tgrid
    if (featurizer == :landscape || featurizer == :silhouette) && tg === nothing
        tg = _default_tgrid_from_barcodes(bcs; nsteps=tgrid_nsteps)
    end
    if tg !== nothing
        tg = _clean_tgrid(tg)
    end

    xg = nothing
    yg = nothing
    if featurizer == :image
        xg, yg = _default_image_grids_from_barcodes(
            bcs;
            img_xgrid=img_xgrid,
            img_ygrid=img_ygrid,
            img_birth_range=img_birth_range,
            img_pers_range=img_pers_range,
            img_nbirth=img_nbirth,
            img_npers=img_npers
        )
    end

    nd, no = size(bcs)
    feats = Array{Vector{Float64}}(undef, nd, no)
    if threads && Threads.nthreads() > 1
        Threads.@threads for k in 1:(nd * no)
            i = div(k - 1, no) + 1
            j = (k - 1) % no + 1
            bc = bcs[i, j]
            if featurizer == :landscape
                pl = persistence_landscape(bc; kmax=kmax, tgrid=tg)
                feats[i, j] = _landscape_feature_vector(pl)
            elseif featurizer == :silhouette
                s = persistence_silhouette(bc; tgrid=tg, weighting=sil_weighting, p=sil_p, normalize=sil_normalize)
                feats[i, j] = Float64[s...]
            elseif featurizer == :image
                PI = persistence_image(bc;
                                       xgrid=xg,
                                       ygrid=yg,
                                       sigma=img_sigma,
                                       coords=img_coords,
                                       weighting=img_weighting,
                                       p=img_p,
                                       normalize=img_normalize)
                feats[i, j] = _image_feature_vector(PI)
            elseif featurizer == :entropy
                e = barcode_entropy(bc; normalize=entropy_normalize, weighting=entropy_weighting, p=entropy_p)
                feats[i, j] = Float64[float(e)]
            elseif featurizer == :summary
                feats[i, j] = _barcode_summary_vector(bc; fields=summary_fields, normalize_entropy=summary_normalize_entropy)
            elseif featurizer isa Function
                feats[i, j] = _as_feature_vector(featurizer(bc))
            else
                error("slice_features: unknown featurizer $(featurizer)")
            end
        end
    else
        for i in 1:nd, j in 1:no
            bc = bcs[i, j]
            if featurizer == :landscape
                pl = persistence_landscape(bc; kmax=kmax, tgrid=tg)
                feats[i, j] = _landscape_feature_vector(pl)
            elseif featurizer == :silhouette
                s = persistence_silhouette(bc; tgrid=tg, weighting=sil_weighting, p=sil_p, normalize=sil_normalize)
                feats[i, j] = Float64[s...]
            elseif featurizer == :image
                PI = persistence_image(bc;
                                       xgrid=xg,
                                       ygrid=yg,
                                       sigma=img_sigma,
                                       coords=img_coords,
                                       weighting=img_weighting,
                                       p=img_p,
                                       normalize=img_normalize)
                feats[i, j] = _image_feature_vector(PI)
            elseif featurizer == :entropy
                e = barcode_entropy(bc; normalize=entropy_normalize, weighting=entropy_weighting, p=entropy_p)
                feats[i, j] = Float64[float(e)]
            elseif featurizer == :summary
                feats[i, j] = _barcode_summary_vector(bc; fields=summary_fields, normalize_entropy=summary_normalize_entropy)
            elseif featurizer isa Function
                feats[i, j] = _as_feature_vector(featurizer(bc))
            else
                error("slice_features: unknown featurizer $(featurizer)")
            end
        end
    end

    out = _aggregate_feature_vectors(feats, W; aggregate=aggregate, unwrap_scalar=unwrap_scalar)
    return _slice_features_result(out, W; featurizer=featurizer, aggregate=aggregate)
end

# Kernel between 1D barcodes (used by slice_kernel).
function _barcode_kernel(
    bA,
    bB;
    kind=:bottleneck_gaussian,
    sigma::Real=1.0,
    gamma=nothing,

    # Wasserstein kernel parameters:
    p::Real=2,
    q::Real=Inf,

    # Landscape kernel parameters:
    tgrid=nothing,
    kmax::Int=5
)::Float64
    if kind isa Function
        return float(kind(bA, bB))
    end

    if kind == :bottleneck_gaussian
        d = bottleneck_distance(bA, bB)
        g = (gamma === nothing) ? (1.0 / (2.0 * float(sigma)^2)) : float(gamma)
        float(sigma) > 0 || error("_barcode_kernel: sigma must be > 0 for gaussian kernels")
        return exp(-g * d * d)
    elseif kind == :bottleneck_laplacian
        d = bottleneck_distance(bA, bB)
        float(sigma) > 0 || error("_barcode_kernel: sigma must be > 0 for laplacian kernels")
        return exp(-d / float(sigma))
    elseif kind == :landscape_gaussian
        tgrid === nothing && error("_barcode_kernel: tgrid required for landscape kernels")
        plA = persistence_landscape(bA; kmax=kmax, tgrid=tgrid)
        plB = persistence_landscape(bB; kmax=kmax, tgrid=tgrid)
        vA = _landscape_feature_vector(plA)
        vB = _landscape_feature_vector(plB)
        d = norm(vA - vB)
        g = (gamma === nothing) ? (1.0 / (2.0 * float(sigma)^2)) : float(gamma)
        float(sigma) > 0 || error("_barcode_kernel: sigma must be > 0 for gaussian kernels")
        return exp(-g * d * d)
    elseif kind == :landscape_laplacian
        tgrid === nothing && error("_barcode_kernel: tgrid required for landscape kernels")
        plA = persistence_landscape(bA; kmax=kmax, tgrid=tgrid)
        plB = persistence_landscape(bB; kmax=kmax, tgrid=tgrid)
        vA = _landscape_feature_vector(plA)
        vB = _landscape_feature_vector(plB)
        d = norm(vA - vB)
        float(sigma) > 0 || error("_barcode_kernel: sigma must be > 0 for laplacian kernels")
        return exp(-d / float(sigma))
    elseif kind == :landscape_linear
        tgrid === nothing && error("_barcode_kernel: tgrid required for landscape kernels")
        plA = persistence_landscape(bA; kmax=kmax, tgrid=tgrid)
        plB = persistence_landscape(bB; kmax=kmax, tgrid=tgrid)
        vA = _landscape_feature_vector(plA)
        vB = _landscape_feature_vector(plB)
        return dot(vA, vB)
    elseif kind == :wasserstein_gaussian
        sigma > 0 || error("_barcode_kernel: sigma must be > 0")
        d = wasserstein_distance(bA, bB; p=p, q=q)
        return exp(-(d * d) / (2 * float(sigma)^2))
    elseif kind == :wasserstein_laplacian
        sigma > 0 || error("_barcode_kernel: sigma must be > 0")
        d = wasserstein_distance(bA, bB; p=p, q=q)
        return exp(-d / float(sigma))
    else
        error("_barcode_kernel: unknown kind $(kind)")
    end
end

"""
    slice_kernel(M, N, slices; kind=:bottleneck_gaussian, ...) -> Float64
    slice_kernel(M, N, pi; directions, offsets, kind=:bottleneck_gaussian, ...) -> Float64

Compute a sliced kernel by:
- restricting `M` and `N` to each slice,
- computing a 1D kernel per slice,
- averaging (weighted) across slices.

This matches the "sliced-kernel" pattern common in the multiparameter
literature (integrate 1-parameter kernels over lines).

Supported `kind` values:
- `:bottleneck_gaussian`  (Gaussian kernel on bottleneck distance)
- `:bottleneck_laplacian` (Laplacian kernel on bottleneck distance)
- `:landscape_gaussian`   (Gaussian kernel on L2 distance between landscape vectors)
- `:landscape_laplacian`
- `:landscape_linear`     (linear kernel on landscape vectors)
- `:wasserstein_gaussian` (Gaussian kernel on p-Wasserstein distance)
- `:wasserstein_laplacian`
- or any function `(bcM, bcN) -> Float64`
"""
function slice_kernel(
    M::PModule{K},
    N::PModule{K},
    slices::AbstractVector;
    kind=:bottleneck_gaussian,
    sigma::Real=1.0,
    gamma=nothing,
    p::Real = 2,
    q::Real = Inf,
    default_weight::Real=1.0,
    normalize_weights::Bool=true,
    # Landscape kernel parameters:
    tgrid=nothing,
    tgrid_nsteps::Int=401,
    kmax::Int=5,
    threads::Bool = (Threads.nthreads() > 1)
)::Float64 where {K}
    dataM = slice_barcodes(M, slices;
                           default_weight=default_weight,
                           normalize_weights=normalize_weights,
                           packed=true,
                           threads=threads)
    dataN = slice_barcodes(N, slices;
                           default_weight=default_weight,
                           normalize_weights=normalize_weights,
                           packed=true,
                           threads=threads)

    bM = dataM.barcodes
    bN = dataN.barcodes
    W = dataM.weights

    length(bM) == length(bN) || error("slice_kernel: slice list length mismatch")

    tg = tgrid
    if (kind == :landscape_gaussian || kind == :landscape_laplacian || kind == :landscape_linear) && tg === nothing
        # Choose a common global grid based on both families of barcodes.
        tg = _default_tgrid_from_barcodes(vcat(bM, bN); nsteps=tgrid_nsteps)
    end
    if tg !== nothing
        tg = _clean_tgrid(tg)
    end

    sumw = sum(W)
    sumw > 0.0 || error("slice_kernel: total weight is zero")

    if threads && Threads.nthreads() > 1
        nT = Threads.nthreads()
        acc_by_slot = fill(0.0, nT)
        Threads.@threads for slot in 1:nT
            acc = 0.0
            for i in slot:nT:length(bM)
                w = W[i]
                w == 0.0 && continue
                acc += w * _barcode_kernel(bM[i], bN[i];
                                           kind=kind, sigma=sigma, gamma=gamma,
                                           p=p, q=q, tgrid=tg, kmax=kmax)
            end
            acc_by_slot[slot] = acc
        end
        acc = sum(acc_by_slot)
    else
        acc = 0.0
        for i in 1:length(bM)
            w = W[i]
            w == 0.0 && continue
            acc += w * _barcode_kernel(bM[i], bN[i];
                                       kind=kind, sigma=sigma, gamma=gamma,
                                       p=p, q=q, tgrid=tg, kmax=kmax)
        end
    end

    # Always return the weighted average (not the raw weighted sum).
    return acc / sumw
end

slice_kernel(M::PModule{K}, N::PModule{K}, chain::AbstractVector{Int}; kwargs...) where {K} =
    slice_kernel(M, N, [chain]; kwargs...)

function slice_kernel(
    M::PModule{K},
    N::PModule{K},
    plan::CompiledSlicePlan;
    kind=:bottleneck_gaussian,
    sigma::Real=1.0,
    gamma=nothing,
    p::Real = 2,
    q::Real = Inf,
    tgrid=nothing,
    tgrid_nsteps::Int=401,
    kmax::Int=5,
    threads::Bool = (Threads.nthreads() > 1),
)::Float64 where {K}
    task = SliceKernelTask(;
        kind = kind,
        sigma = float(sigma),
        gamma = gamma,
        p = float(p),
        q = float(q),
        tgrid = tgrid,
        tgrid_nsteps = tgrid_nsteps,
        kmax = kmax,
        threads = threads,
    )
    return run_invariants(plan, module_cache(M, N), task)
end

function slice_kernel(
    M::PModule{K},
    N::PModule{K},
    pi;
    directions=nothing,
    offsets=nothing,
    kind=:bottleneck_gaussian,
    sigma::Real=1.0,
    gamma=nothing,
    p::Real = 2,
    q::Real = Inf,
    normalize_weights::Bool=true,
    # Geometric slice parameters:
    tmin::Union{Real,Nothing}=nothing,
    tmax::Union{Real,Nothing}=nothing,
    nsteps::Int=401,
    kmin=nothing,
    kmax_param=nothing,
    strict::Bool=true,
    drop_unknown::Bool=true,
    dedup::Bool=true,
    normalize_dirs::Symbol=:none,
    direction_weight::Symbol=:none,
    offset_weights=nothing,
    # Landscape kernel parameters:
    tgrid=nothing,
    tgrid_nsteps::Int=401,
    kmax::Int=5,
    threads::Bool = (Threads.nthreads() > 1),
    slice_kwargs...
)::Float64 where {K}
    box0 = get(slice_kwargs, :box, nothing)
    strict0 = get(slice_kwargs, :strict, strict)
    tmin0 = tmin
    tmax0 = tmax
    if box0 === nothing && tmin0 === nothing && tmax0 === nothing && pi isa PLikeEncodingMap
        box0 = :auto
    end
    opts_compile = InvariantOptions(box = box0, strict = strict0, threads = threads)
    slice_kwargs0 = (; (k => v for (k, v) in pairs(slice_kwargs) if k != :box && k != :strict)...)
    plan = compile_slices(
        pi,
        opts_compile;
        directions = directions,
        offsets = offsets,
        normalize_dirs = normalize_dirs,
        direction_weight = direction_weight,
        offset_weights = offset_weights,
        normalize_weights = normalize_weights,
        drop_unknown = drop_unknown,
        threads = threads,
        tmin = tmin,
        tmax = tmax,
        nsteps = nsteps,
        kmin = kmin,
        kmax_param = kmax_param,
        dedup = dedup,
        slice_kwargs0...
    )
    task = SliceKernelTask(;
        kind = kind,
        sigma = float(sigma),
        gamma = gamma,
        p = float(p),
        q = float(q),
        tgrid = tgrid,
        tgrid_nsteps = tgrid_nsteps,
        kmax = kmax,
        threads = threads,
    )
    return run_invariants(plan, module_cache(M, N), task)
end

# Convenience wrappers: allow working directly with presentations.
slice_barcodes(H::FringeModule{K}, args...; kwargs...) where {K} =
    slice_barcodes(pmodule_from_fringe(H), args...; kwargs...)
slice_features(H::FringeModule{K}, args...; kwargs...) where {K} =
    slice_features(pmodule_from_fringe(H), args...; kwargs...)
slice_kernel(H::FringeModule{K}, H2::FringeModule{K}, args...; kwargs...) where {K} =
    slice_kernel(pmodule_from_fringe(H), pmodule_from_fringe(H2), args...; kwargs...)

end # module SliceInvariants
