# =============================================================================
# Workflow.jl
#
# User-facing workflow helpers and small pipeline objects.
#
# PosetModules.jl is meant to be an "API map" file (includes + re-exports).
# This file holds the small amount of glue code that makes the public workflow
# smooth and predictable:
#
#   presentation  ->  encode(...)  ->  EncodingResult
#                                ->  resolve(...) -> ResolutionResult
#                                ->  invariants(...) -> InvariantResult
#                                ->  ext(...) / tor(...) etc.
#
# Implementation details remain in the submodules:
#   - ZnEncoding.jl (Z^n encoding)
#   - PLBackend.jl / PLPolyhedra.jl (R^n encoding backends)
#   - DerivedFunctors.jl (Ext/Tor + resolutions)
#   - Invariants.jl
# =============================================================================

using LinearAlgebra  # UniformScaling I / transpose
using SparseArrays

# Keep this file self-contained: import the handful of names it mentions
# in type annotations or uses unqualified.
using .CoreModules: QQ,
                    EncodingOptions, ResolutionOptions, DerivedFunctorOptions, InvariantOptions,
                    EncodingResult, ResolutionResult, InvariantResult

using .IndicatorResolutions: pmodule_from_fringe
using .FlangeZn: Flange
using .PLBackend: BoxUpset, BoxDownset, encode_fringe_boxes
using .Encoding: build_uptight_encoding_from_fringe,
                 pushforward_fringe_along_encoding,
                 PostcomposedEncodingMap
using .DataPipeline: PointCloud, ImageNd, GraphData, EmbeddedPlanarGraph2D, GradedComplex,
                     FiltrationSpec, GridEncodingMap, poset_from_axes, grid_index
using .Modules: PModule, PMorphism, cover_edges, dim_at
using .ModuleComplexes: ModuleCochainComplex, cohomology_module

@inline _resolve_encoding_opts(opts::Union{EncodingOptions,Nothing}) =
    opts === nothing ? EncodingOptions() : opts
@inline function _resolve_encoding_alias(enc::Union{EncodingOptions,Nothing},
                                        opts::Union{EncodingOptions,Nothing})
    if enc !== nothing && opts !== nothing
        error("encode: pass either enc or opts, not both.")
    end
    return _resolve_encoding_opts(opts === nothing ? enc : opts)
end
@inline _resolve_resolution_opts(opts::Union{ResolutionOptions,Nothing}) =
    opts === nothing ? ResolutionOptions() : opts
@inline _resolve_df_opts(opts::Union{DerivedFunctorOptions,Nothing}) =
    opts === nothing ? DerivedFunctorOptions() : opts
@inline _resolve_invariant_opts(opts::Union{InvariantOptions,Nothing}) =
    opts === nothing ? InvariantOptions() : opts
using .IndicatorResolutions: projective_cover
using .FiniteFringe: AbstractPoset, FringeModule, Upset, Downset, principal_upset, principal_downset,
                     leq, nvertices, poset_equal_opposite
using .FlangeZn: Face, IndFlat, IndInj, Flange

# Cache for cubical structures and posets to reuse across repeated grids.
const _cubical_cache = Dict{Any,Any}()
const _poset_cache = Dict{Any,AbstractPoset}()

# -----------------------------------------------------------------------------
# Backend selection for R^n encoding (PLBackend vs PLPolyhedra)

# Workflow entrypoints designed to:
# - accept input presentations,
# - call lower-level encoding / resolution / derived-functor / invariant routines,
# - return small results objects with provenance:
#
#   presentation -> encode(...)    -> EncodingResult
#               -> resolve(...)   -> ResolutionResult
#               -> invariant(...) -> InvariantResult
#
# Derived-functor entrypoints (`hom`, `ext`, `tor`) return graded objects
# queried via the GradedSpaces interface (degree_range, dim, basis, coordinates, representative).
#
# A future goal is that these are essentially the only names exported by default.

"""
    has_polyhedra_backend()::Bool

Return true if the optional Polyhedra-based backend is available at runtime.
"""
has_polyhedra_backend()::Bool = PLPolyhedra.HAVE_POLY

has_polyhedra_backend()::Bool = PLPolyhedra.HAVE_POLY

"""
    available_pl_backends()::Vector{Symbol}

Report which R^n encoding backends are available in this session.
Always includes :pl_backend (because PLBackend is always loaded).
Includes :pl if Polyhedra/CDDLib are available.
"""
function available_pl_backends()::Vector{Symbol}
    out = Symbol[:pl_backend]
    if has_polyhedra_backend()
        push!(out, :pl)
    end
    return out
end

# Internal: normalize user backend selectors for R^n
# Canonical symbols:
#   :pl_backend  (PLBackend, axis-aligned box partition)
#   :pl          (PLPolyhedra / Polyhedra backend)
function _normalize_pl_backend(b::Symbol)::Symbol
    b == :auto && return :auto
    b == :pl && return :pl
    (b == :pl_backend || b == :pl_backend_boxes || b == :boxes || b == :axis) && return :pl_backend
    error("Unknown R^n encoding backend symbol: $(b). Try :auto, :pl_backend, or :pl.")
end

# Internal: infer ambient dimension from box generators.
function _infer_box_dim(Ups::Vector{BoxUpset}, Downs::Vector{BoxDownset})
    if !isempty(Ups)
        return length(Ups[1].ell)
    elseif !isempty(Downs)
        return length(Downs[1].u)
    else
        error("Cannot infer dimension: both Ups and Downs are empty.")
    end
end

"""
    supports_pl_backend(Ups::Vector{BoxUpset}, Downs::Vector{BoxDownset}; opts=EncodingOptions())::Bool

Return true iff PLBackend can encode this box presentation under the given options.
This checks:
- axis-aligned boxes with finite coordinates,
- region count does not exceed opts.max_regions if set.
"""
function supports_pl_backend(Ups::Vector{BoxUpset}, Downs::Vector{BoxDownset};
                             opts::Union{EncodingOptions,Nothing}=nothing)::Bool
    opts = _resolve_encoding_opts(opts)
    max_regions = opts.max_regions
    # PLBackend can only handle finite, axis-aligned boxes (encoded as BoxUpset/BoxDownset).
    # If any generator has +/-Inf bounds, or other invalid bounds, reject here.
    for U in Ups
        for a in U.ell
            isfinite(a) || return false
        end
    end
    for D in Downs
        for b in D.u
            isfinite(b) || return false
        end
    end

    # If max_regions is set, do a cheap upper bound check on the grid size.
    # PLBackend partitions via coordinate grid from generator endpoints.
    if max_regions !== nothing
        n = _infer_box_dim(Ups, Downs)
        coords = [QQ[] for _ in 1:n]
        for U in Ups
            for i in 1:n
                push!(coords[i], U.ell[i])
            end
        end
        for D in Downs
            for i in 1:n
                push!(coords[i], D.u[i])
            end
        end
        # unique, sorted
        for i in 1:n
            coords[i] = sort!(unique!(coords[i]))
        end
        # number of cells in the induced grid:
        # product over dimensions of (k_i - 1).
        cells = 1
        for i in 1:n
            ki = length(coords[i])
            if ki < 2
                return false
            end
            cells *= (ki - 1)
            if cells > max_regions
                return false
            end
        end
    end

    return true
end

"""
    supports_pl_backend(F::PLPolyhedra.PLFringe; opts=EncodingOptions())::Bool

Return true iff the PLFringe can be converted to a box presentation and encoded by PLBackend
under the given options.
"""
function supports_pl_backend(F::PLPolyhedra.PLFringe; opts::Union{EncodingOptions,Nothing}=nothing)::Bool
    opts = _resolve_encoding_opts(opts)
    try
        Ups, Downs = boxes_from_pl_fringe(F)
        return supports_pl_backend(Ups, Downs; opts=opts)
    catch
        return false
    end
end

# -----------------------------------------------------------------------------
# Conversions between PLFringe and box generators for PLBackend

"""
    boxes_from_pl_fringe(F::PLPolyhedra.PLFringe) -> (Ups::Vector{BoxUpset}, Downs::Vector{BoxDownset})

Convert an axis-aligned PLFringe (as used by PLPolyhedra) into box generators for PLBackend.

Throws if the fringe contains non-axis-aligned generators or is otherwise incompatible.
"""
function boxes_from_pl_fringe(F::PLPolyhedra.PLFringe)
    Ups = Vector{BoxUpset}(undef, length(F.Ups))
    for i in eachindex(F.Ups)
        U = F.Ups[i]
        if length(U.A) != 2 * F.n
            error("PLFringe upset is not axis-aligned (A has wrong size).")
        end
        lo = Vector{QQ}(undef, F.n)
        hi = Vector{QQ}(undef, F.n)
        for j in 1:F.n
            lo[j] = U.b[j]
            hi[j] = U.b[F.n + j]
        end
        Ups[i] = BoxUpset(lo, hi)
    end

    Downs = Vector{BoxDownset}(undef, length(F.Downs))
    for i in eachindex(F.Downs)
        D = F.Downs[i]
        if length(D.A) != 2 * F.n
            error("PLFringe downset is not axis-aligned (A has wrong size).")
        end
        lo = Vector{QQ}(undef, F.n)
        hi = Vector{QQ}(undef, F.n)
        for j in 1:F.n
            lo[j] = D.b[j]
            hi[j] = D.b[F.n + j]
        end
        Downs[i] = BoxDownset(lo, hi)
    end

    return Ups, Downs
end

"""
    pl_fringe_from_boxes(Ups::Vector{BoxUpset}, Downs::Vector{BoxDownset}) -> PLPolyhedra.PLFringe

Convert a box presentation into a PLFringe (axis-aligned) for use with PLPolyhedra.
"""
function pl_fringe_from_boxes(Ups::Vector{BoxUpset}, Downs::Vector{BoxDownset})
    n = _infer_box_dim(Ups, Downs)

    PUps = Vector{PLPolyhedra.PLUpset}(undef, length(Ups))
    for i in eachindex(Ups)
        ell = Ups[i].ell
        A = -Matrix{QQ}(I, n, n)
        b = -ell
        PUps[i] = PLPolyhedra.PLUpset(A, b)
    end

    PDowns = Vector{PLPolyhedra.PLDownset}(undef, length(Downs))
    for i in eachindex(Downs)
        u = Downs[i].u
        A = Matrix{QQ}(I, n, n)
        b = u
        PDowns[i] = PLPolyhedra.PLDownset(A, b)
    end

    return PLPolyhedra.PLFringe(n, PUps, PDowns)
end

# -----------------------------------------------------------------------------
# Backend chooser and encoding wrappers for R^n

"""
    choose_pl_backend(...; opts=EncodingOptions())::Symbol

Choose which R^n encoding backend to use.
Returns :pl_backend or :pl.

Rules:
- If opts.backend is explicitly set (e.g. :pl_backend or :pl), try to honor it.
- If opts.backend=:auto, prefer :pl_backend when supported; otherwise fall back to :pl.
- If :pl is requested but Polyhedra backend is unavailable, throw.
"""
function choose_pl_backend(Ups::Vector{BoxUpset}, Downs::Vector{BoxDownset};
                           opts::Union{EncodingOptions,Nothing}=nothing)::Symbol
    opts = _resolve_encoding_opts(opts)
    b = _normalize_pl_backend(opts.backend)
    if b == :pl_backend
        supports_pl_backend(Ups, Downs; opts=opts) || error("PLBackend requested, but this input is not supported by PLBackend.")
        return :pl_backend
    elseif b == :pl
        has_polyhedra_backend() || error("PLPolyhedra backend requested, but Polyhedra/CDDLib is not available.")
        return :pl
    elseif b == :auto
        if supports_pl_backend(Ups, Downs; opts=opts)
            return :pl_backend
        else
            has_polyhedra_backend() || error("PLBackend not applicable and Polyhedra backend is unavailable.")
            return :pl
        end
    else
        error("Internal error: unexpected normalized backend $(b).")
    end
end

function choose_pl_backend(F::PLPolyhedra.PLFringe; opts::Union{EncodingOptions,Nothing}=nothing)::Symbol
    opts = _resolve_encoding_opts(opts)
    b = _normalize_pl_backend(opts.backend)
    if b == :pl_backend
        supports_pl_backend(F; opts=opts) || error("PLBackend requested, but this PLFringe is not supported by PLBackend.")
        return :pl_backend
    elseif b == :pl
        has_polyhedra_backend() || error("PLPolyhedra backend requested, but Polyhedra/CDDLib is not available.")
        return :pl
    elseif b == :auto
        if supports_pl_backend(F; opts=opts)
            return :pl_backend
        else
            has_polyhedra_backend() || error("PLBackend not applicable and Polyhedra backend is unavailable.")
            return :pl
        end
    else
        error("Internal error: unexpected normalized backend $(b).")
    end
end

"""
    encode_from_fringe(...)

Unified R^n encoder:
- If :pl_backend is chosen, uses PLBackend.encode_fringe_boxes(...)
- If :pl is chosen, uses PLPolyhedra.encode_from_PL_fringe(...)
"""
function encode_from_fringe(Ups::Vector{BoxUpset}, Downs::Vector{BoxDownset}, Phi::AbstractMatrix{QQ},
                            opts::Union{EncodingOptions,Nothing}=nothing)
    opts = _resolve_encoding_opts(opts)
    b = choose_pl_backend(Ups, Downs; opts=opts)
    if b == :pl_backend
        return encode_fringe_boxes(Ups, Downs, Phi, opts; poset_kind=opts.poset_kind)
    else
        F = pl_fringe_from_boxes(Ups, Downs)
        return PLPolyhedra.encode_from_PL_fringe(F, opts; poset_kind=opts.poset_kind)
    end
end

function encode_from_fringe(F::PLPolyhedra.PLFringe, opts::Union{EncodingOptions,Nothing}=nothing)
    opts = _resolve_encoding_opts(opts)
    b = choose_pl_backend(F; opts=opts)
    if b == :pl_backend
        Ups, Downs = boxes_from_pl_fringe(F)
        Phi = reshape(ones(QQ, length(Downs) * length(Ups)), length(Downs), length(Ups))
        return encode_fringe_boxes(Ups, Downs, Phi, opts; poset_kind=opts.poset_kind)
    else
        return PLPolyhedra.encode_from_PL_fringe(F, opts; poset_kind=opts.poset_kind)
    end
end

"""
    encode_pmodule_from_fringe(...)

Return (M, pi) where M is the finite-poset PModule obtained from the encoded fringe module.
"""
function encode_pmodule_from_fringe(Ups::Vector{BoxUpset}, Downs::Vector{BoxDownset}, Phi::AbstractMatrix{QQ},
                                    opts::Union{EncodingOptions,Nothing}=nothing)
    opts = _resolve_encoding_opts(opts)
    P, H, pi = encode_from_fringe(Ups, Downs, Phi, opts)
    return pmodule_from_fringe(H), pi
end

function encode_pmodule_from_fringe(F::PLPolyhedra.PLFringe, opts::Union{EncodingOptions,Nothing}=nothing)
    opts = _resolve_encoding_opts(opts)
    P, H, pi = encode_from_fringe(F, opts)
    return pmodule_from_fringe(H), pi
end

encode_from_fringe(Ups::Vector{BoxUpset}, Downs::Vector{BoxDownset}, Phi::AbstractMatrix{QQ};
                   opts::Union{EncodingOptions,Nothing}=nothing) =
    encode_from_fringe(Ups, Downs, Phi, _resolve_encoding_opts(opts))

encode_from_fringe(F::PLPolyhedra.PLFringe;
                   opts::Union{EncodingOptions,Nothing}=nothing) =
    encode_from_fringe(F, _resolve_encoding_opts(opts))

encode_pmodule_from_fringe(Ups::Vector{BoxUpset}, Downs::Vector{BoxDownset}, Phi::AbstractMatrix{QQ};
                           opts::Union{EncodingOptions,Nothing}=nothing) =
    encode_pmodule_from_fringe(Ups, Downs, Phi, _resolve_encoding_opts(opts))

encode_pmodule_from_fringe(F::PLPolyhedra.PLFringe;
                           opts::Union{EncodingOptions,Nothing}=nothing) =
    encode_pmodule_from_fringe(F, _resolve_encoding_opts(opts))

# -----------------------------------------------------------------------------
# Workflow entrypoints (narrative API)

"""
    encode(x; backend=:auto, max_regions=nothing, strict_eps=nothing, poset_kind=:signature) -> EncodingResult
    encode(x1, x2, ...; backend=:auto, ...) -> Vector{EncodingResult}

High-level entrypoint that turns a presentation object into a finite encoding poset model.

- Z^n inputs (Flange{QQ}) use the Zn encoder (backend=:auto or :zn).
- R^n inputs (PLFringe, or box generators) use:
    * PLBackend when possible (axis-aligned, finite bounds, and not too many regions),
    * otherwise PLPolyhedra (if Polyhedra/CDDLib are available).

The returned EncodingResult stores:
- P : the finite encoding poset
- H : the pushed-down fringe module on P (optional but populated here)
- M : the finite-poset module pmodule_from_fringe(H)
- pi: the classifier map from the original domain to P
- presentation / opts / backend / meta : provenance

Notes
-----
* For multiple PL fringes, only the PLPolyhedra backend currently supports common-encoding.
  If you need common-encoding and you asked for PLBackend explicitly, we throw an error.
"""
function encode(x; backend::Symbol=:auto, max_regions=nothing, strict_eps=nothing, poset_kind::Symbol=:signature)
    enc = EncodingOptions(; backend=backend, max_regions=max_regions, strict_eps=strict_eps, poset_kind=poset_kind)
    return encode(x, enc)
end

# -----------------------
# Z^n presentations

function encode(FG::Flange{QQ}, enc::EncodingOptions)
    if enc.backend != :auto && enc.backend != :zn
        error("encode(Flange): EncodingOptions.backend must be :auto or :zn (got $(enc.backend))")
    end
    P, H, pi = ZnEncoding.encode_from_flange(FG, enc)
    M = pmodule_from_fringe(H)
    return EncodingResult(P, M, pi; H=H, presentation=FG, opts=enc, backend=:zn, meta=(;))
end

function encode(FG::Flange{QQ};
                enc=nothing,
                opts=nothing,
                backend::Symbol=:auto,
                max_regions=nothing,
                strict_eps=nothing,
                poset_kind::Symbol=:signature)
    if enc !== nothing || opts !== nothing
        if backend != :auto || max_regions !== nothing || strict_eps !== nothing || poset_kind != :signature
            error("encode(Flange): pass either enc/opts or backend/max_regions/strict_eps/poset_kind, not both.")
        end
        return encode(FG, _resolve_encoding_alias(enc, opts))
    end
    enc2 = EncodingOptions(; backend=backend, max_regions=max_regions, strict_eps=strict_eps, poset_kind=poset_kind)
    return encode(FG, enc2)
end

function encode(FGs::AbstractVector{<:Flange{QQ}}, enc::EncodingOptions)
    if enc.backend != :auto && enc.backend != :zn
        error("encode(Vector{Flange}): EncodingOptions.backend must be :auto or :zn (got $(enc.backend))")
    end
    P, Hs, pi = ZnEncoding.encode_from_flanges(FGs, enc)
    out = Vector{EncodingResult}(undef, length(FGs))
    for i in eachindex(FGs)
        H = Hs[i]
        out[i] = EncodingResult(P, pmodule_from_fringe(H), pi;
                                H=H, presentation=FGs[i], opts=enc, backend=:zn, meta=(;))
    end
    return out
end

function encode(FGs::AbstractVector{<:Flange{QQ}};
                enc=nothing,
                opts=nothing,
                backend::Symbol=:auto,
                max_regions=nothing,
                strict_eps=nothing,
                poset_kind::Symbol=:signature)
    if enc !== nothing || opts !== nothing
        if backend != :auto || max_regions !== nothing || strict_eps !== nothing || poset_kind != :signature
            error("encode(Vector{Flange}): pass either enc/opts or backend/max_regions/strict_eps/poset_kind, not both.")
        end
        return encode(FGs, _resolve_encoding_alias(enc, opts))
    end
    enc2 = EncodingOptions(; backend=backend, max_regions=max_regions, strict_eps=strict_eps, poset_kind=poset_kind)
    return encode(FGs, enc2)
end

encode(FG1::Flange{QQ}, FG2::Flange{QQ}; kwargs...) =
    encode(Flange{QQ}[FG1, FG2]; kwargs...)

encode(FG1::Flange{QQ}, FG2::Flange{QQ}, FG3::Flange{QQ}; kwargs...) =
    encode(Flange{QQ}[FG1, FG2, FG3]; kwargs...)

# -----------------------
# R^n presentations

function encode(F::PLPolyhedra.PLFringe, enc::Union{EncodingOptions,Nothing}=nothing)
    enc = _resolve_encoding_opts(enc)
    P, H, pi = encode_from_fringe(F, enc)
    M = pmodule_from_fringe(H)
    b = choose_pl_backend(F; opts=enc)
    return EncodingResult(P, M, pi; H=H, presentation=F, opts=enc, backend=b, meta=(;))
end

function encode(Ups::Vector{BoxUpset}, Downs::Vector{BoxDownset}, Phi::AbstractMatrix{QQ},
                enc::Union{EncodingOptions,Nothing}=nothing)
    enc = _resolve_encoding_opts(enc)
    P, H, pi = encode_from_fringe(Ups, Downs, Phi, enc)
    M = pmodule_from_fringe(H)
    b = choose_pl_backend(Ups, Downs; opts=enc)
    return EncodingResult(P, M, pi;
                          H=H, presentation=(Ups=Ups, Downs=Downs, Phi=Phi),
                          opts=enc, backend=b, meta=(;))
end

# Convenience overloads for common BoxFringe encodings
encode(Ups::Vector{BoxUpset}, Downs::Vector{BoxDownset}, enc::Union{EncodingOptions,Nothing}=nothing) =
    encode(Ups, Downs,
           reshape(ones(QQ, length(Downs) * length(Ups)), length(Downs), length(Ups)),
           _resolve_encoding_opts(enc))

function encode(Ups::Vector{BoxUpset}, Downs::Vector{BoxDownset};
                backend::Symbol=:auto, max_regions=nothing, strict_eps=nothing, poset_kind::Symbol=:signature)
    enc = EncodingOptions(; backend=backend, max_regions=max_regions, strict_eps=strict_eps, poset_kind=poset_kind)
    return encode(Ups, Downs, enc)
end

function encode(Ups::Vector{BoxUpset}, Downs::Vector{BoxDownset}, Phi_vec::AbstractVector{QQ},
                enc::Union{EncodingOptions,Nothing}=nothing)
    enc = _resolve_encoding_opts(enc)
    Phi = reshape(Phi_vec, length(Downs), length(Ups))
    return encode(Ups, Downs, Phi, enc)
end

function encode(Fs::AbstractVector{<:PLPolyhedra.PLFringe}, enc::EncodingOptions)
    # Today, only the PLPolyhedra backend supports common-encoding multiple PL fringes.
    if _normalize_pl_backend(enc.backend) == :pl_backend
        error("encode(Vector{PLFringe}): common encoding for PLBackend is not implemented; use backend=:pl or backend=:auto.")
    end
    if enc.backend != :auto && enc.backend != :pl
        error("encode(Vector{PLFringe}): EncodingOptions.backend must be :auto or :pl (got $(enc.backend))")
    end
    P, Hs, pi = PLPolyhedra.encode_from_PL_fringes(Fs, enc; poset_kind=enc.poset_kind)
    out = Vector{EncodingResult}(undef, length(Fs))
    for i in eachindex(Fs)
        H = Hs[i]
        out[i] = EncodingResult(P, pmodule_from_fringe(H), pi;
                                H=H, presentation=Fs[i], opts=enc, backend=:pl, meta=(;))
    end
    return out
end

encode(Fs::AbstractVector{<:PLPolyhedra.PLFringe}; enc=nothing, opts=nothing) =
    encode(Fs, _resolve_encoding_alias(enc, opts))

encode(F1::PLPolyhedra.PLFringe, F2::PLPolyhedra.PLFringe; kwargs...) =
    encode(PLPolyhedra.PLFringe[F1, F2]; kwargs...)

encode(F1::PLPolyhedra.PLFringe, F2::PLPolyhedra.PLFringe, F3::PLPolyhedra.PLFringe; kwargs...) =
    encode(PLPolyhedra.PLFringe[F1, F2, F3]; kwargs...)

# -----------------------------------------------------------------------------
# Data ingestion pipeline (datasets -> graded complex -> cohomology -> fringe)

@inline function _tuple_leq(a::NTuple{N,Int}, b::NTuple{N,Int},
                            orientation::NTuple{N,Int}) where {N}
    @inbounds for i in 1:N
        if orientation[i] == 1
            if a[i] > b[i]
                return false
            end
        else
            if a[i] < b[i]
                return false
            end
        end
    end
    return true
end

function _grid_tuples(sizes::NTuple{N,Int}) where {N}
    total = 1
    for i in 1:N
        total *= sizes[i]
    end
    out = Vector{NTuple{N,Int}}(undef, total)
    cur = ones(Int, N)
    for lin in 1:total
        out[lin] = ntuple(i -> cur[i], N)
        for i in 1:N
            cur[i] += 1
            if cur[i] <= sizes[i]
                break
            else
                cur[i] = 1
            end
        end
    end
    return out
end

function _axes_from_grades(grades::Vector{<:NTuple{N,T}}, n::Int) where {N,T}
    axes = [T[] for _ in 1:n]
    for g in grades
        length(g) == n || error("grade has length $(length(g)) but expected $n")
        for i in 1:n
            push!(axes[i], g[i])
        end
    end
    for i in 1:n
        axes[i] = sort!(unique!(axes[i]))
    end
    return ntuple(i -> axes[i], n)
end

function _validate_axes_sorted(axes)
    for (i, ax) in enumerate(axes)
        if !issorted(ax)
            error("encode_from_data: axis $i must be sorted ascending.")
        end
    end
    return nothing
end

function _validate_axes_kind(axes; axis_kind=nothing)
    # axis_kind may be :zn / :rn to force integer vs real axes.
    is_int_typed_axis(ax) = all(x -> x isa Integer, ax)
    is_int_valued_axis(ax) = all(x -> (x isa Integer) || (x isa Real && isinteger(x)), ax)

    if axis_kind === nothing
        return nothing
    end

    axis_kind in (:zn, :rn) || error("encode_from_data: axis_kind must be :zn or :rn.")
    if axis_kind == :zn
        for (i, ax) in enumerate(axes)
            if !is_int_typed_axis(ax)
                error("encode_from_data: axis $i must be integer-typed for axis_kind=:zn.")
            end
        end
    else
        for (i, ax) in enumerate(axes)
            if is_int_typed_axis(ax)
                error("encode_from_data: axis $i is integer-typed but axis_kind=:rn.")
            end
        end
    end
    return nothing
end

function _axes_key(axes)
    return tuple((Tuple(ax) for ax in axes)...)
end

function _coarsen_axis(ax::Vector{T}, max_len::Int) where {T}
    n = length(ax)
    n <= max_len && return ax
    idxs = round.(Int, range(1, n; length=max_len))
    return ax[idxs]
end

function _coarsen_axes(axes::NTuple{N,Vector{T}}, max_len::Int) where {N,T}
    return ntuple(i -> _coarsen_axis(axes[i], max_len), N)
end

function _quantize_grades(G::GradedComplex, eps)
    N = length(G.grades[1])
    eps_vec = if eps isa Number
        ntuple(_ -> Float64(eps), N)
    elseif eps isa Tuple
        length(eps) == N || error("encode_from_data: eps length mismatch.")
        ntuple(i -> Float64(eps[i]), N)
    else
        error("encode_from_data: eps must be a number or tuple.")
    end
    grades = Vector{NTuple{N,Float64}}(undef, length(G.grades))
    for i in eachindex(G.grades)
        g = G.grades[i]
        grades[i] = ntuple(j -> round(Float64(g[j]) / eps_vec[j]) * eps_vec[j], N)
    end
    return GradedComplex(G.cells_by_dim, G.boundaries, [collect(grades[i]) for i in eachindex(grades)]; cell_dims=G.cell_dims)
end

function _poset_from_axes_cached(axes, orientation)
    key = (_axes_key(axes), orientation)
    P = get(_poset_cache, key, nothing)
    if P === nothing
        P = poset_from_axes(axes; orientation=orientation, kind=:grid)
        _poset_cache[key] = P
    end
    return P
end

function _poset_vertex_coords(axes::NTuple{N,Vector{T}}) where {N,T}
    sizes = ntuple(i -> length(axes[i]), N)
    idxs = _grid_tuples(sizes)
    coords = Vector{Vector{Int}}(undef, length(idxs))
    for i in eachindex(idxs)
        coords[i] = [Int(axes[j][idxs[i][j]]) for j in 1:N]
    end
    return coords
end

function _grades_by_dim(G::GradedComplex)
    counts = [length(c) for c in G.cells_by_dim]
    total = sum(counts)
    length(G.grades) == total ||
        error("GradedComplex.grades length $(length(G.grades)) does not match total cells $(total).")
    out = Vector{Vector{typeof(G.grades[1])}}(undef, length(counts))
    idx = 1
    for d in 1:length(counts)
        out[d] = Vector{typeof(G.grades[1])}(undef, counts[d])
        for j in 1:counts[d]
            out[d][j] = G.grades[idx]
            idx += 1
        end
    end
    return out
end

function _birth_indices(grades::Vector{<:NTuple{N,<:Real}},
                        axes::NTuple{N,Vector{<:Real}},
                        orientation::NTuple{N,Int}) where {N}
    births = Vector{NTuple{N,Int}}(undef, length(grades))
    for i in eachindex(grades)
        g = grades[i]
        length(g) == N || error("grade length $(length(g)) does not match N=$(N)")
        births[i] = ntuple(k -> begin
            val = orientation[k] == 1 ? g[k] : -g[k]
            idx = searchsortedlast(axes[k], val)
            idx >= 1 || error("grade component $(g[k]) falls below axis minimum for axis $k")
            idx
        end, N)
    end
    return births
end

function _active_lists(births::Vector{NTuple{N,Int}},
                       vertex_idxs::Vector{NTuple{N,Int}},
                       orientation::NTuple{N,Int}) where {N}
    active = Vector{Vector{Int}}(undef, length(vertex_idxs))
    for i in eachindex(vertex_idxs)
        lst = Int[]
        vi = vertex_idxs[i]
        for c in eachindex(births)
            if _tuple_leq(births[c], vi, orientation)
                push!(lst, c)
            end
        end
        active[i] = lst
    end
    return active
end

function _pos_maps(active::Vector{Vector{Int}})
    maps = Vector{Dict{Int,Int}}(undef, length(active))
    for i in eachindex(active)
        d = Dict{Int,Int}()
        for (j, c) in enumerate(active[i])
            d[c] = j
        end
        maps[i] = d
    end
    return maps
end

function _pmodule_from_active_lists(P::AbstractPoset,
                                    active::Vector{Vector{Int}})
    dims = [length(lst) for lst in active]
    edge_maps = Dict{Tuple{Int,Int}, Matrix{QQ}}()
    for (u, v) in cover_edges(P)
        Mu = active[u]
        Mv = active[v]
        pos_v = Dict{Int,Int}()
        for (i, c) in enumerate(Mv)
            pos_v[c] = i
        end
        A = zeros(QQ, length(Mv), length(Mu))
        for (j, c) in enumerate(Mu)
            i = get(pos_v, c, 0)
            if i != 0
                A[i, j] = 1//1
            end
        end
        edge_maps[(u, v)] = A
    end
    return PModule{QQ}(P, dims, edge_maps)
end

function cochain_complex_from_graded_complex(G::GradedComplex,
                                             P::AbstractPoset,
                                             axes::NTuple{N,Vector{T}};
                                             orientation::NTuple{N,Int}=ntuple(_ -> 1, N)) where {N,T}
    sizes = ntuple(i -> length(axes[i]), N)
    vertex_idxs = _grid_tuples(sizes)

    grades_by_dim = _grades_by_dim(G)
    births_by_dim = Vector{Vector{NTuple{N,Int}}}(undef, length(grades_by_dim))
    active_by_dim = Vector{Vector{Vector{Int}}}(undef, length(grades_by_dim))
    pos_by_dim = Vector{Vector{Dict{Int,Int}}}(undef, length(grades_by_dim))
    for d in 1:length(grades_by_dim)
        births_by_dim[d] = _birth_indices(grades_by_dim[d], axes, orientation)
        active_by_dim[d] = _active_lists(births_by_dim[d], vertex_idxs, orientation)
        pos_by_dim[d] = _pos_maps(active_by_dim[d])
    end

    terms = Vector{PModule{QQ}}(undef, length(grades_by_dim))
    for d in 1:length(grades_by_dim)
        terms[d] = _pmodule_from_active_lists(P, active_by_dim[d])
    end

    diffs = PMorphism{QQ}[]
    expected = length(grades_by_dim) - 1
    boundaries = G.boundaries
    if length(boundaries) < expected
        for k in (length(boundaries) + 1):expected
            rows = length(grades_by_dim[k])
            cols = length(grades_by_dim[k + 1])
            push!(boundaries, spzeros(Int, rows, cols))
        end
    elseif length(boundaries) > expected
        error("GradedComplex.boundaries length mismatch (expected $(expected)).")
    end

    for k in 1:length(boundaries)
        B = boundaries[k]  # boundary C_{k+1} -> C_k
        comps = Vector{Matrix{QQ}}(undef, nvertices(P))
        for i in 1:nvertices(P)
            Lk = active_by_dim[k][i]
            Lk1 = active_by_dim[k+1][i]
            rowmap = pos_by_dim[k][i]
            colmap = pos_by_dim[k+1][i]
            M = spzeros(QQ, length(Lk1), length(Lk))
            Ii, Jj, Vv = findnz(B)
            @inbounds for t in eachindex(Ii)
                r = get(rowmap, Ii[t], 0)
                c = get(colmap, Jj[t], 0)
                if r != 0 && c != 0
                    M[c, r] = QQ(Vv[t])
                end
            end
            comps[i] = Matrix{QQ}(M)
        end
        push!(diffs, PMorphism{QQ}(terms[k], terms[k+1], comps))
    end

    return ModuleCochainComplex(terms, diffs; tmin=0, check=true)
end

"""
    fringe_presentation(M::PModule{QQ}) -> FringeModule{QQ}

Construct a canonical fringe presentation whose image recovers `M`.
"""
function fringe_presentation(M::PModule{QQ})
    P = M.Q
    F0, pi0, gens_at_F0 = projective_cover(M)
    E0, iota0, gens_at_E0 = IndicatorResolutions._injective_hull(M)

    # Map F0 -> E0 whose image is M.
    comps = Vector{Matrix{QQ}}(undef, nvertices(P))
    for i in 1:nvertices(P)
        comps[i] = iota0.comps[i] * pi0.comps[i]
    end
    f = PMorphism{QQ}(F0, E0, comps)

    U = IndicatorResolutions._principal_upsets_from_gens(P, gens_at_F0)
    D = IndicatorResolutions._principal_downsets_from_gens(P, gens_at_E0)

    L0 = IndicatorResolutions._local_index_list_up(P, gens_at_F0)
    L1 = IndicatorResolutions._local_index_list_down(P, gens_at_E0)

    globalU = Tuple{Int,Int}[]
    for p in 1:nvertices(P)
        append!(globalU, gens_at_F0[p])
    end
    globalD = Tuple{Int,Int}[]
    for u in 1:nvertices(P)
        append!(globalD, gens_at_E0[u])
    end

    function local_pos(L, g::Tuple{Int,Int}, i::Int)
        for (j, gg) in enumerate(L[i])
            if gg == g
                return j
            end
        end
        return 0
    end

    phi = spzeros(QQ, length(D), length(U))
    for (lambda, (plambda, jlambda)) in enumerate(globalU)
        for (theta, (ptheta, jtheta)) in enumerate(globalD)
            if leq(P, plambda, ptheta)
                i = plambda
                col = local_pos(L0, (plambda, jlambda), i)
                row = local_pos(L1, (ptheta, jtheta), i)
                if col > 0 && row > 0
                    val = f.comps[i][row, col]
                    if val != 0
                        phi[theta, lambda] = val
                    end
                end
            end
        end
    end

    return FringeModule{QQ}(P, U, D, phi)
end

"""
    flange_presentation(M::PModule{QQ}, pi::GridEncodingMap) -> Flange{QQ}

Construct a Z^n flange presentation for an integer-valued grid encoding.
"""
function flange_presentation(M::PModule{QQ}, pi::GridEncodingMap)
    P = M.Q
    n = length(pi.coords)
    for ax in pi.coords
        all(x -> x isa Integer || (x isa Real && isinteger(x)), ax) ||
            error("flange_presentation: axes must be integer-valued.")
    end
    F0, pi0, gens_at_F0 = projective_cover(M)
    E0, iota0, gens_at_E0 = IndicatorResolutions._injective_hull(M)

    comps = Vector{Matrix{QQ}}(undef, nvertices(P))
    for i in 1:nvertices(P)
        comps[i] = iota0.comps[i] * pi0.comps[i]
    end
    f = PMorphism{QQ}(F0, E0, comps)

    tau = Face(n, falses(n))
    coords = _poset_vertex_coords(pi.coords)

    flats = IndFlat[]
    for p in 1:nvertices(P)
        for _ in gens_at_F0[p]
            push!(flats, IndFlat(tau, coords[p]; id=:F))
        end
    end

    injectives = IndInj[]
    for u in 1:nvertices(P)
        for _ in gens_at_E0[u]
            push!(injectives, IndInj(tau, coords[u]; id=:E))
        end
    end

    L0 = IndicatorResolutions._local_index_list_up(P, gens_at_F0)
    L1 = IndicatorResolutions._local_index_list_down(P, gens_at_E0)

    globalU = Tuple{Int,Int}[]
    for p in 1:nvertices(P)
        append!(globalU, gens_at_F0[p])
    end
    globalD = Tuple{Int,Int}[]
    for u in 1:nvertices(P)
        append!(globalD, gens_at_E0[u])
    end

    function local_pos(L, g::Tuple{Int,Int}, i::Int)
        for (j, gg) in enumerate(L[i])
            if gg == g
                return j
            end
        end
        return 0
    end

    Phi = spzeros(QQ, length(injectives), length(flats))
    for (lambda, (plambda, jlambda)) in enumerate(globalU)
        for (theta, (ptheta, jtheta)) in enumerate(globalD)
            if leq(P, plambda, ptheta)
                i = plambda
                col = local_pos(L0, (plambda, jlambda), i)
                row = local_pos(L1, (ptheta, jtheta), i)
                if col > 0 && row > 0
                    val = f.comps[i][row, col]
                    if val != 0
                        Phi[theta, lambda] = val
                    end
                end
            end
        end
    end

    return Flange{QQ}(n, flats, injectives, Matrix{QQ}(Phi))
end

function _simplicial_boundary(simplices::Vector{Vector{Int}}, faces::Vector{Vector{Int}})
    K = isempty(simplices) ? length(faces[1]) + 1 : length(simplices[1])
    face_index = Dict{Tuple{Vararg{Int}},Int}()
    for (i, f) in enumerate(faces)
        face_index[Tuple(f)] = i
    end
    I = Int[]
    J = Int[]
    V = Int[]
    for (j, s) in enumerate(simplices)
        for i in 1:K
            f = [s[t] for t in 1:K if t != i]
            row = face_index[Tuple(f)]
            push!(I, row)
            push!(J, j)
            push!(V, isodd(i) ? 1 : -1)
        end
    end
    return sparse(I, J, V, length(faces), length(simplices))
end

function _combinations(n::Int, k::Int)
    if k == 0
        return [Int[]]
    end
    out = Vector{Vector{Int}}()
    function rec(start::Int, acc::Vector{Int})
        if length(acc) == k
            push!(out, copy(acc))
            return
        end
        for i in start:(n - (k - length(acc)) + 1)
            push!(acc, i)
            rec(i + 1, acc)
            pop!(acc)
        end
    end
    rec(1, Int[])
    return out
end

function _filter_params(params::NamedTuple, drop::Vector{Symbol})
    return (; (k => v for (k, v) in pairs(params) if !(k in drop))...)
end

function _knn_distances(points::Vector{Vector{Float64}}, k::Int)
    n = length(points)
    dists = Vector{Float64}(undef, n)
    for i in 1:n
        ds = Float64[]
        for j in 1:n
            if i == j
                continue
            end
            push!(ds, norm(points[i] .- points[j]))
        end
        sort!(ds)
        k <= length(ds) || error("kNN k=$k exceeds number of neighbors.")
        dists[i] = ds[k]
    end
    return dists
end

function _graded_complex_from_point_cloud(data::PointCloud, spec::FiltrationSpec)
    max_dim = get(spec.params, :max_dim, 1)
    kind = spec.kind
    points = data.points
    n = length(points)
    if n == 0
        error("PointCloud has no points.")
    end
    # pairwise distances
    dist = zeros(Float64, n, n)
    for i in 1:n
        for j in (i+1):n
            d = norm(points[i] .- points[j])
            dist[i, j] = d
            dist[j, i] = d
        end
    end
    # simplices by dimension
    simplices = Vector{Vector{Vector{Int}}}(undef, max_dim + 1)
    simplices[1] = [ [i] for i in 1:n ]
    total = length(simplices[1])
    max_simplices = get(spec.params, :max_simplices, nothing)

    if get(spec.params, :sparse_rips, false)
        radius = get(spec.params, :radius, nothing)
        radius === nothing && error("sparse_rips requires radius.")
        # 1-skeleton only in sparse mode for speed.
        sims = Vector{Vector{Int}}()
        for i in 1:n, j in i+1:n
            if dist[i, j] <= radius
                push!(sims, [i, j])
            end
        end
        simplices = [simplices[1], sims]
        max_dim = 1
        total += length(sims)
        if max_simplices !== nothing && total > max_simplices
            error("PointCloud Rips: exceeded max_simplices=$(max_simplices).")
        end
    elseif get(spec.params, :approx_rips, false)
        radius = get(spec.params, :radius, nothing)
        radius === nothing && error("approx_rips requires radius.")
        max_edges = get(spec.params, :max_edges, nothing)
        max_degree = get(spec.params, :max_degree, nothing)
        sample_frac = get(spec.params, :sample_frac, nothing)
        sims = Vector{Vector{Int}}()
        deg = zeros(Int, n)
        for i in 1:n
            for j in i+1:n
                if dist[i, j] <= radius
                    if max_degree !== nothing && (deg[i] >= max_degree || deg[j] >= max_degree)
                        continue
                    end
                    if sample_frac !== nothing && rand() > sample_frac
                        continue
                    end
                    push!(sims, [i, j])
                    deg[i] += 1
                    deg[j] += 1
                    if max_edges !== nothing && length(sims) >= max_edges
                        break
                    end
                end
            end
            if max_edges !== nothing && length(sims) >= max_edges
                break
            end
        end
        simplices = [simplices[1], sims]
        max_dim = 1
        total += length(sims)
        if max_simplices !== nothing && total > max_simplices
            error("PointCloud Rips: exceeded max_simplices=$(max_simplices).")
        end
    else
        for k in 2:max_dim+1
            sims = Vector{Vector{Int}}()
            for comb in _combinations(n, k)
                push!(sims, comb)
            end
            simplices[k] = sims
            total += length(sims)
            if max_simplices !== nothing && total > max_simplices
                error("PointCloud Rips: exceeded max_simplices=$(max_simplices).")
            end
        end
    end

    if kind == :rips
        grades = Vector{NTuple{1,Float64}}()
        for s in simplices[1]
            push!(grades, (0.0,))
        end
        for k in 2:max_dim+1
            for s in simplices[k]
                maxd = 0.0
                for i in 1:length(s)
                    for j in (i+1):length(s)
                        d = dist[s[i], s[j]]
                        if d > maxd
                            maxd = d
                        end
                    end
                end
                push!(grades, (maxd,))
            end
        end
    elseif kind == :rips_density || kind == :rips_knn
        knn_k = get(spec.params, :density_k, 2)
        dens = _knn_distances([Vector{Float64}(p) for p in points], knn_k)
        grades = Vector{NTuple{2,Float64}}()
        for s in simplices[1]
            v = s[1]
            push!(grades, (0.0, dens[v]))
        end
        for k in 2:max_dim+1
            for s in simplices[k]
                maxd = 0.0
                maxden = 0.0
                for i in 1:length(s)
                    vi = s[i]
                    if dens[vi] > maxden
                        maxden = dens[vi]
                    end
                    for j in (i+1):length(s)
                        d = dist[s[i], s[j]]
                        if d > maxd
                            maxd = d
                        end
                    end
                end
                push!(grades, (maxd, maxden))
            end
        end
    elseif kind == :witness
        landmarks = get(spec.params, :landmarks, nothing)
        landmarks === nothing && error("witness requires landmarks.")
        length(landmarks) > 0 || error("witness: landmarks cannot be empty.")
        pts = [points[i] for i in landmarks]
        params2 = _filter_params(spec.params, [:landmarks, :sparse_rips, :radius])
        params2 = merge(params2, (max_dim = max_dim,))
        return _graded_complex_from_point_cloud(PointCloud(pts), FiltrationSpec(; kind=:rips, params2...))
    else
        error("Unsupported point cloud filtration kind: $(kind).")
    end

    boundaries = SparseMatrixCSC{Int,Int}[]
    for k in 2:max_dim+1
        Bk = _simplicial_boundary(simplices[k], simplices[k-1])
        push!(boundaries, Bk)
    end

    cells = [collect(1:length(simplices[k])) for k in 1:length(simplices)]
    G = GradedComplex(cells, boundaries, grades)
    axes = get(spec.params, :axes, _axes_from_grades(grades, length(grades[1])))
    orientation = get(spec.params, :orientation, ntuple(_ -> 1, length(axes)))
    return G, axes, orientation
end

function _graded_complex_from_graph(data::GraphData, spec::FiltrationSpec)
    kind = spec.kind
    n = data.n
    edges = data.edges
    if kind == :graph_lower_star || kind == :clique_lower_star
        vertex_grades = get(spec.params, :vertex_grades, nothing)
        vertex_grades === nothing && error("graph filtration requires vertex_grades.")
        length(vertex_grades) == n || error("vertex_grades length mismatch.")
        N = length(vertex_grades[1])
        grades = Vector{NTuple{N,eltype(vertex_grades[1])}}()
        for g in vertex_grades
            length(g) == N || error("vertex_grades entries must have length N=$N")
            push!(grades, ntuple(i -> g[i], N))
        end

        if kind == :graph_lower_star
            simplices0 = [ [i] for i in 1:n ]
            simplices1 = [ [u, v] for (u, v) in edges ]
            for (u, v) in edges
                gu = vertex_grades[u]
                gv = vertex_grades[v]
                push!(grades, ntuple(i -> max(gu[i], gv[i]), N))
            end
            B1 = _simplicial_boundary(simplices1, simplices0)
            G = GradedComplex([collect(1:n), collect(1:length(edges))], [B1], grades)
            axes = get(spec.params, :axes, _axes_from_grades(grades, N))
            orientation = get(spec.params, :orientation, ntuple(_ -> 1, N))
            return G, axes, orientation
        else
            max_dim = get(spec.params, :max_dim, 2)
            adj = falses(n, n)
            for (u, v) in edges
                adj[u, v] = true
                adj[v, u] = true
            end
            simplices = Vector{Vector{Vector{Int}}}(undef, max_dim + 1)
            simplices[1] = [ [i] for i in 1:n ]
            for k in 2:max_dim+1
                sims = Vector{Vector{Int}}()
                for comb in _combinations(n, k)
                    ok = true
                    for i in 1:k
                        for j in (i+1):k
                            if !adj[comb[i], comb[j]]
                                ok = false
                                break
                            end
                        end
                        if !ok
                            break
                        end
                    end
                    if ok
                        push!(sims, comb)
                    end
                end
                simplices[k] = sims
            end
            for k in 2:max_dim+1
                for s in simplices[k]
                    push!(grades, ntuple(i -> maximum(vertex_grades[v][i] for v in s), N))
                end
            end
            boundaries = SparseMatrixCSC{Int,Int}[]
            for k in 2:max_dim+1
                push!(boundaries, _simplicial_boundary(simplices[k], simplices[k-1]))
            end
            cells = [collect(1:length(simplices[k])) for k in 1:length(simplices)]
            G = GradedComplex(cells, boundaries, grades)
            axes = get(spec.params, :axes, _axes_from_grades(grades, N))
            orientation = get(spec.params, :orientation, ntuple(_ -> 1, N))
            return G, axes, orientation
        end
    elseif kind == :edge_weighted
        weights = get(spec.params, :edge_weights, nothing)
        weights === nothing && error("edge_weighted requires edge_weights.")
        length(weights) == length(edges) || error("edge_weights length mismatch.")
        grades = Vector{NTuple{1,Float64}}()
        for _ in 1:n
            push!(grades, (0.0,))
        end
        for w in weights
            push!(grades, (Float64(w),))
        end
        simplices0 = [ [i] for i in 1:n ]
        simplices1 = [ [u, v] for (u, v) in edges ]
        B1 = _simplicial_boundary(simplices1, simplices0)
        G = GradedComplex([collect(1:n), collect(1:length(edges))], [B1], grades)
        axes = get(spec.params, :axes, _axes_from_grades(grades, 1))
        orientation = get(spec.params, :orientation, (1,))
        return G, axes, orientation
    else
        error("Unsupported graph filtration kind: $(kind).")
    end
end

function _distance_transform(mask::AbstractArray{Bool})
    dims = size(mask)
    N = length(dims)
    coords = collect(CartesianIndices(mask))
    true_pts = [c for c in coords if mask[c]]
    out = zeros(Float64, dims)
    for c in coords
        if mask[c]
            out[c] = 0.0
            continue
        end
        best = Inf
        for t in true_pts
            s = 0.0
            for i in 1:N
                d = (c[i] - t[i])
                s += d * d
            end
            if s < best
                best = s
            end
        end
        out[c] = sqrt(best)
    end
    return out
end

function _cubical_structure(dims::NTuple{N,Int}) where {N}
    key = Tuple(dims)
    cached = get(_cubical_cache, key, nothing)
    if cached !== nothing
        return cached
    end
    cells_by_dim = Vector{Vector{Tuple{NTuple{N,Int},NTuple{N,Int}}}}(undef, N + 1)
    cell_index = Vector{Dict{Tuple{NTuple{N,Int},NTuple{N,Int}},Int}}(undef, N + 1)
    for k in 0:N
        cells_by_dim[k+1] = Tuple{NTuple{N,Int},NTuple{N,Int}}[]
        cell_index[k+1] = Dict{Tuple{NTuple{N,Int},NTuple{N,Int}},Int}()
    end

    for mask_vec in _combinations(N, 0)
        mask = ntuple(i -> 0, N)
        for coords in CartesianIndices(Tuple(dims))
            origin = ntuple(i -> coords[i], N)
            key2 = (origin, mask)
            push!(cells_by_dim[1], key2)
        end
    end
    for k in 1:N
        for mask_idxs in _combinations(N, k)
            mask = ntuple(i -> (i in mask_idxs ? 1 : 0), N)
            ranges = Vector{UnitRange{Int}}(undef, N)
            for i in 1:N
                ranges[i] = 1:(dims[i] - mask[i])
            end
            for coords in CartesianIndices(Tuple(ranges))
                origin = ntuple(i -> coords[i], N)
                key2 = (origin, mask)
                push!(cells_by_dim[k+1], key2)
            end
        end
    end

    for k in 0:N
        for (i, key2) in enumerate(cells_by_dim[k+1])
            cell_index[k+1][key2] = i
        end
    end

    boundaries = SparseMatrixCSC{Int,Int}[]
    for k in 1:N
        I = Int[]
        J = Int[]
        V = Int[]
        for (j, (origin, mask)) in enumerate(cells_by_dim[k+1])
            axes = [i for i in 1:N if mask[i] == 1]
            for (pos, axis) in enumerate(axes)
                mask_face = ntuple(i -> (i == axis ? 0 : mask[i]), N)
                origin_low = origin
                origin_high = ntuple(i -> (i == axis ? origin[i] + 1 : origin[i]), N)
                low_key = (origin_low, mask_face)
                high_key = (origin_high, mask_face)
                row_low = cell_index[k][low_key]
                row_high = cell_index[k][high_key]
                sign_low = isodd(pos) ? 1 : -1
                sign_high = -sign_low
                push!(I, row_low); push!(J, j); push!(V, sign_low)
                push!(I, row_high); push!(J, j); push!(V, sign_high)
            end
        end
        push!(boundaries, sparse(I, J, V, length(cells_by_dim[k]), length(cells_by_dim[k+1])))
    end

    cached = (cells_by_dim=cells_by_dim, cell_index=cell_index, boundaries=boundaries)
    _cubical_cache[key] = cached
    return cached
end

function _graded_complex_from_image_channels(channels::Vector{<:AbstractArray}, spec::FiltrationSpec)
    img = channels[1]
    dims = size(img)
    for c in channels
        size(c) == dims || error("All channels must have the same size.")
    end
    N = length(dims)
    C = length(channels)

    cached = _cubical_structure(dims)
    cells_by_dim = cached.cells_by_dim
    cell_index = cached.cell_index

    grades = Vector{NTuple{C,Float64}}()
    for k in 0:N
        for (origin, mask) in cells_by_dim[k+1]
            maxv = fill(-Inf, C)
            ranges = Vector{UnitRange{Int}}(undef, N)
            for i in 1:N
                ranges[i] = origin[i]:(origin[i] + mask[i])
            end
            for coords in CartesianIndices(Tuple(ranges))
                for ch in 1:C
                    v = Float64(channels[ch][coords])
                    if v > maxv[ch]
                        maxv[ch] = v
                    end
                end
            end
            push!(grades, ntuple(i -> maxv[i], C))
        end
    end

    cells = [collect(1:length(cells_by_dim[k+1])) for k in 0:N]
    G = GradedComplex(cells, cached.boundaries, grades)
    axes = get(spec.params, :axes, _axes_from_grades(grades, C))
    orientation = get(spec.params, :orientation, ntuple(_ -> 1, C))
    return G, axes, orientation
end

function _graded_complex_from_image(data::ImageNd, spec::FiltrationSpec)
    if haskey(spec.params, :channels)
        chans = spec.params[:channels]
        return _graded_complex_from_image_channels(chans, spec)
    elseif spec.kind == :image_distance_bifiltration
        mask = get(spec.params, :mask, nothing)
        mask === nothing && error("image_distance_bifiltration requires mask.")
        dist = _distance_transform(mask)
        chans = [data.data, dist]
        return _graded_complex_from_image_channels(chans, spec)
    else
        return _graded_complex_from_image_channels([data.data], spec)
    end
end

function _graded_complex_from_data(data, spec::FiltrationSpec)
    if data isa GradedComplex
        N = length(data.grades[1])
        axes = get(spec.params, :axes, _axes_from_grades(data.grades, N))
        orientation = get(spec.params, :orientation, ntuple(_ -> 1, N))
        return data, axes, orientation
    elseif data isa PointCloud
        return _graded_complex_from_point_cloud(data, spec)
    elseif data isa ImageNd
        return _graded_complex_from_image(data, spec)
    elseif data isa GraphData
        return _graded_complex_from_graph(data, spec)
    elseif data isa EmbeddedPlanarGraph2D
        if spec.kind == :wing_vein_bifiltration
            grid = get(spec.params, :grid, (32, 32))
            bbox = get(spec.params, :bbox, data.bbox)
            bbox === nothing && error("wing_vein_bifiltration requires bbox.")
            xmin, xmax, ymin, ymax = bbox
            nx, ny = grid
            xs = range(xmin, xmax; length=nx)
            ys = range(ymin, ymax; length=ny)
            distV = zeros(Float64, nx, ny)
            distE = zeros(Float64, nx, ny)
            verts = data.vertices
            segments = Vector{Tuple{Vector{Float64},Vector{Float64}}}()
            if data.polylines === nothing
                for (u, v) in data.edges
                    push!(segments, (verts[u], verts[v]))
                end
            else
                for poly in data.polylines
                    for i in 1:(length(poly)-1)
                        push!(segments, (poly[i], poly[i+1]))
                    end
                end
            end
            for i in 1:nx, j in 1:ny
                x = xs[i]; y = ys[j]
                dv = Inf
                for v in verts
                    d = hypot(x - v[1], y - v[2])
                    if d < dv
                        dv = d
                    end
                end
                distV[i, j] = dv
                de = Inf
                for (a, b) in segments
                    ax, ay = a[1], a[2]
                    bx, by = b[1], b[2]
                    vx = bx - ax
                    vy = by - ay
                    wx = x - ax
                    wy = y - ay
                    t = (vx*wx + vy*wy) / (vx*vx + vy*vy)
                    if t < 0
                        px, py = ax, ay
                    elseif t > 1
                        px, py = bx, by
                    else
                        px, py = ax + t*vx, ay + t*vy
                    end
                    d = hypot(x - px, y - py)
                    if d < de
                        de = d
                    end
                end
                distE[i, j] = de
            end
            chans = [-distV, distE]
            img = ImageNd(chans[1])
            spec2 = FiltrationSpec(; kind=:image_distance_bifiltration, channels=chans,
                                    orientation=get(spec.params, :orientation, (-1, 1)))
            return _graded_complex_from_image(img, spec2)
        else
            # Treat as a graph with externally supplied vertex grades
            gspec = FiltrationSpec(; kind=spec.kind, spec.params...)
            gdata = GraphData(length(data.vertices), data.edges;
                              coords=data.vertices, weights=nothing, T=eltype(data.vertices[1]))
            return _graded_complex_from_graph(gdata, gspec)
        end
    else
        error("Unsupported dataset type for encode_from_data.")
    end
end

"""
    encode_from_data(data, spec; degree=0) -> EncodingResult

Ingest a dataset + filtration spec into a persistence module and return
an EncodingResult with a fringe presentation.
"""
function encode_from_data(data, spec::FiltrationSpec;
                          degree::Int=0,
                          return_tuple::Bool=false,
                          emit::Symbol=:fringe)
    G, axes, orientation = _graded_complex_from_data(data, spec)
    if haskey(spec.params, :eps)
        G = _quantize_grades(G, spec.params[:eps])
        axes = _axes_from_grades(G.grades, length(G.grades[1]))
    end
    axes_policy = get(spec.params, :axes_policy, :encoding)
    if axes_policy == :as_given
        haskey(spec.params, :axes) || error("encode_from_data: axes_policy=:as_given requires axes.")
        axes = spec.params[:axes]
    elseif axes_policy == :coarsen
        max_len = get(spec.params, :max_axis_len, nothing)
        max_len === nothing && error("encode_from_data: axes_policy=:coarsen requires max_axis_len.")
        axes = _coarsen_axes(axes, Int(max_len))
    elseif axes_policy != :encoding
        error("encode_from_data: unknown axes_policy $(axes_policy).")
    end
    _validate_axes_sorted(axes)
    _validate_axes_kind(axes; axis_kind=get(spec.params, :axis_kind, nothing))
    if emit == :flange && get(spec.params, :axis_kind, nothing) != :zn
        error("encode_from_data: emit=:flange requires axis_kind=:zn (integer axes).")
    end
    P = _poset_from_axes_cached(axes, orientation)
    pi = GridEncodingMap(P, axes; orientation=orientation)
    C = cochain_complex_from_graded_complex(G, P, axes; orientation=orientation)
    M = cohomology_module(C, degree)
    H = fringe_presentation(M)
    if emit == :flange
        FG = flange_presentation(M, pi)
        return EncodingResult(P, M, pi; H=H, presentation=(data=data, spec=spec),
                              opts=EncodingOptions(; backend=:data), backend=:data,
                              meta=(; flange=FG))
    elseif emit != :fringe
        error("encode_from_data: emit must be :fringe or :flange.")
    end
    if return_tuple
        return (P, H, pi)
    end
    return EncodingResult(P, M, pi; H=H, presentation=(data=data, spec=spec),
                          opts=EncodingOptions(; backend=:data), backend=:data, meta=(;))
end

ingest(data, spec::FiltrationSpec; degree::Int=0) =
    encode_from_data(data, spec; degree=degree, return_tuple=true)

# -----------------------------------------------------------------------------
# Coarsening / compression of finite encodings

"""
    coarsen(enc::EncodingResult; method=:uptight) -> EncodingResult

Coarsen/compress the finite encoding poset of an existing `EncodingResult`.

Currently supported:
- `method=:uptight`: build an uptight encoding from `enc.H`, push the fringe forward
  along the resulting finite encoding map, rebuild the pmodule, and postcompose the
  *ambient* classifier so `locate` still works on the new region poset.

This leaves `enc.backend` unchanged (the ambient encoding backend is still the same),
but replaces `(P, M, pi, H)` with their coarsened versions.
"""
function coarsen(enc::EncodingResult; method::Symbol = :uptight)
    method === :uptight || error("coarsen: unsupported method=$method. Currently only :uptight is supported.")

    # 1) Compute coarsening map pi : (old P) -> (new P2)
    upt = build_uptight_encoding_from_fringe(enc.H)
    pi = upt.pi

    # 2) Push fringe module forward along pi
    H2 = pushforward_fringe_along_encoding(enc.H, pi)

    # 3) Rebuild pmodule on the coarsened region poset
    M2 = pmodule_from_fringe(H2)

    # 4) Postcompose old ambient classifier with the coarsening map
    pi2 = PostcomposedEncodingMap(enc.pi, pi)

    # 5) Preserve meta (Dict or NamedTuple), but record what happened
    meta2 = if enc.meta isa AbstractDict
        d = copy(enc.meta)
        d[:coarsen_method] = method
        d[:coarsen_n_before] = nvertices(enc.P)
        d[:coarsen_n_after] = nvertices(pi.P)
        d
    elseif enc.meta isa NamedTuple
        merge(enc.meta, (coarsen_method = method, coarsen_n_before = nvertices(enc.P), coarsen_n_after = nvertices(pi.P)))
    else
        Dict{Symbol,Any}(
            :orig_meta => enc.meta,
            :coarsen_method => method,
            :coarsen_n_before => nvertices(enc.P),
            :coarsen_n_after => nvertices(pi.P),
        )
    end

    return EncodingResult(pi.P, M2, pi2;
        H = H2,
        presentation = enc.presentation,
        opts = enc.opts,
        backend = enc.backend,
        meta = meta2,
    )
end

# -----------------------------------------------------------------------------
# Accessors (avoid field spelunking in user code / docs)
# -----------------------------------------------------------------------------

"""
    poset(enc::EncodingResult)

Return the finite encoding poset used by `enc`.
"""
poset(enc::EncodingResult) = enc.P

"""
    pmodule(enc::EncodingResult)

Return the encoded module stored inside `enc`.
"""
pmodule(enc::EncodingResult) = enc.M

"""
    classifier(enc::EncodingResult)

Return the classifier / encoding map `pi` stored inside `enc`.
"""
classifier(enc::EncodingResult) = enc.pi

"""
    backend(enc::EncodingResult)

Return the backend symbol used during encoding.
"""
backend(enc::EncodingResult) = enc.backend

"""
    presentation(enc::EncodingResult)

Return the original presentation object used to create `enc`.
"""
presentation(enc::EncodingResult) = enc.presentation


# -----------------------------------------------------------------------------
# Homological algebra helpers for ResolutionResult
# -----------------------------------------------------------------------------

"""
    resolution(res::ResolutionResult)

Return the underlying resolution object stored inside `res`.
"""
resolution(res::ResolutionResult) = res.res

"""
    betti(res::ResolutionResult)

Return the Betti table stored inside `res` (may be `nothing`).
"""
betti(res::ResolutionResult) = res.betti

"""
    minimality_report(res::ResolutionResult)

Return the minimality report stored inside `res` (may be `nothing`).
"""
minimality_report(res::ResolutionResult) = res.minimality

# Forward minimality helpers for explicit resolution objects.
minimality_report(res::DerivedFunctors.Resolutions.ProjectiveResolution{QQ};
                  check_cover::Bool=true) =
    DerivedFunctors.minimality_report(res; check_cover=check_cover)
minimality_report(res::DerivedFunctors.Resolutions.InjectiveResolution{QQ};
                  check_hull::Bool=true) =
    DerivedFunctors.minimality_report(res; check_hull=check_hull)

"""
    is_minimal(res::ResolutionResult) -> Bool

Return whether the resolution stored in `res` was proven minimal by the chosen backend.
"""
is_minimal(res::ResolutionResult) = res.minimality !== nothing

is_minimal(res::DerivedFunctors.Resolutions.ProjectiveResolution{QQ};
           check_cover::Bool=true) =
    DerivedFunctors.is_minimal(res; check_cover=check_cover)
is_minimal(res::DerivedFunctors.Resolutions.InjectiveResolution{QQ};
           check_hull::Bool=true) =
    DerivedFunctors.is_minimal(res; check_hull=check_hull)

assert_minimal(res::DerivedFunctors.Resolutions.ProjectiveResolution{QQ};
               check_cover::Bool=true) =
    DerivedFunctors.assert_minimal(res; check_cover=check_cover)
assert_minimal(res::DerivedFunctors.Resolutions.InjectiveResolution{QQ};
               check_hull::Bool=true) =
    DerivedFunctors.assert_minimal(res; check_hull=check_hull)



# -----------------------------------------------------------------------------
# Derived functors and invariants from workflow objects

"""
    hom(A, B)

Compute Hom(A, B) for finite-poset modules.

Convenience overloads accept EncodingResult and unwrap the underlying PModule.
"""
function hom(A::EncodingResult, B::EncodingResult)
    (A.P === B.P) || error("hom: encodings are on different posets; use encode(x, y; ...) to common-encode first.")
    return DerivedFunctors.Hom(A.M, B.M)
end

function hom(A::EncodingResult, B::Modules.PModule{QQ})
    (A.M.Q === B.Q) || error("hom: posets mismatch.")
    return DerivedFunctors.Hom(A.M, B)
end

function hom(A::Modules.PModule{QQ}, B::EncodingResult)
    (A.Q === B.M.Q) || error("hom: posets mismatch.")
    return DerivedFunctors.Hom(A, B.M)
end

function hom(A::Modules.PModule{QQ}, B::Modules.PModule{QQ})
    (A.Q === B.Q) || error("hom: posets mismatch.")
    return DerivedFunctors.Hom(A, B)
end

"""
    tor(Rop, L; maxdeg=3, model=:auto)

Compute Tor_t(Rop, L), where `Rop` is a right-module represented as a module on
the opposite poset P^op, and `L` is a left-module on P.

For EncodingResult inputs, the underlying posets must be opposite:
    poset_equal_opposite(L.P, Rop.P)
"""
function tor(Rop::EncodingResult, L::EncodingResult;
             maxdeg::Int=3, model::Symbol=:auto)
    poset_equal_opposite(L.P, Rop.P) || error("tor: expected first argument on opposite poset of the second.")
    df = DerivedFunctorOptions(; maxdeg=maxdeg, model=model, canon=:none)
    return DerivedFunctors.Tor(Rop.M, L.M, df)
end

function tor(Rop::Modules.PModule{QQ}, L::Modules.PModule{QQ};
             maxdeg::Int=3, model::Symbol=:auto)
    poset_equal_opposite(L.Q, Rop.Q) || error("tor: expected first argument on opposite poset of the second.")
    df = DerivedFunctorOptions(; maxdeg=maxdeg, model=model, canon=:none)
    return DerivedFunctors.Tor(Rop, L, df)
end

tor(Rop::EncodingResult, L::Modules.PModule{QQ}; kwargs...) = tor(Rop.M, L; kwargs...)
tor(Rop::Modules.PModule{QQ}, L::EncodingResult; kwargs...) = tor(Rop, L.M; kwargs...)


"""
    ext_algebra(enc; maxdeg=3, model=:auto)

Compute the Ext-algebra Ext^*(M, M) with Yoneda product, where M is the module stored in `enc`.
"""
function ext_algebra(enc::EncodingResult; maxdeg::Int=3, model::Symbol=:auto, canon::Symbol=:auto)
    df = DerivedFunctorOptions(; maxdeg=maxdeg, model=model, canon=canon)
    return DerivedFunctors.ExtAlgebra(enc.M, df)
end

"""
    ext(A::EncodingResult, B::EncodingResult; maxdeg=3, model=:auto, canon=:auto)

Compute Ext^t(A, B) using the finite-poset modules stored in EncodingResult.

If `A` and `B` are not encoded on the same poset object, you must common-encode first:
    encs = encode(x, y; backend=...)
    E = ext(encs[1], encs[2])
"""
function ext(A::EncodingResult, B::EncodingResult;
             maxdeg::Int=3, model::Symbol=:auto, canon::Symbol=:auto)
    (A.P === B.P) || error("ext: encodings are on different posets; use encode(x, y; ...) to common-encode first.")
    df = DerivedFunctorOptions(; maxdeg=maxdeg, model=model, canon=canon)
    return DerivedFunctors.Ext(A.M, B.M, df)
end

function ext(A::EncodingResult, B::Modules.PModule{QQ};
             maxdeg::Int=3, model::Symbol=:auto, canon::Symbol=:auto)
    (A.M.Q === B.Q) || error("ext: module B lives on a different poset; common-encode first.")
    df = DerivedFunctorOptions(; maxdeg=maxdeg, model=model, canon=canon)
    return DerivedFunctors.Ext(A.M, B, df)
end

"""
    resolve(enc::EncodingResult; kind=:projective, opts=ResolutionOptions(), minimality=false)

Compute a projective or injective resolution starting from an EncodingResult.
Returns a ResolutionResult that stores the resolution plus provenance.

- kind=:projective (default) computes a ProjectiveResolution and stores its Betti table.
- kind=:injective computes an InjectiveResolution and stores its Bass table in `meta`.

Minimality checks can be expensive; enable with `minimality=true`.
"""
function resolve(enc::EncodingResult;
                 kind::Symbol=:projective,
                 opts::Union{ResolutionOptions,Nothing}=nothing,
                 minimality::Bool=false,
                 check_hull::Bool=true)
    opts = _resolve_resolution_opts(opts)
    kind_norm = kind in (:proj, :projective) ? :projective :
                kind in (:inj, :injective)   ? :injective  :
                error("resolve: kind must be :projective or :injective (got $(kind))")

    if kind_norm == :projective
        res = DerivedFunctors.projective_resolution(enc.M, opts)
        b = DerivedFunctors.betti(res)
        mrep = minimality ? DerivedFunctors.minimality_report(res) : nothing
        return ResolutionResult(res; enc=enc, betti=b, minimality=mrep, opts=opts, meta=(kind=:projective,))
    else
        res = DerivedFunctors.injective_resolution(enc.M, opts)
        bass = DerivedFunctors.bass(res)
        mrep = minimality ? DerivedFunctors.minimality_report(res; check_hull=check_hull) : nothing
        return ResolutionResult(res; enc=enc, betti=nothing, minimality=mrep, opts=opts,
                                meta=(kind=:injective, bass=bass))
    end
end

# -----------------------------------------------------------------------------
# Complex-level homological algebra
# -----------------------------------------------------------------------------

"""
    rhom(C, N; kwargs...)

Compute RHom(C, N) where C is a module cochain complex and N is a module.
"""
rhom(C::ModuleComplexes.ModuleCochainComplex{QQ}, N::Modules.PModule{QQ}; kwargs...) =
    ModuleComplexes.RHom(C, N; kwargs...)

rhom(C::ModuleComplexes.ModuleCochainComplex{QQ}, N::EncodingResult; kwargs...) =
    rhom(C, N.M; kwargs...)


"""
    derived_tensor(Rop, C; kwargs...)

Compute derived tensor Rop \\otimes^L C, where Rop is on P^op and C is on P.
"""
derived_tensor(Rop::Modules.PModule{QQ}, C::ModuleComplexes.ModuleCochainComplex{QQ}; kwargs...) =
    ModuleComplexes.DerivedTensor(Rop, C; kwargs...)

derived_tensor(Rop::EncodingResult, C::ModuleComplexes.ModuleCochainComplex{QQ}; kwargs...) =
    derived_tensor(Rop.M, C; kwargs...)


"""
    hyperext(C, N; maxdeg=3, kwargs...)

Compute HyperExt^t(C, N) for a module cochain complex C and a module N.
"""
function hyperext(C::ModuleComplexes.ModuleCochainComplex{QQ}, N::Modules.PModule{QQ};
                  maxdeg::Int=3, kwargs...)
    df = DerivedFunctorOptions(; maxdeg=maxdeg, model=:auto, canon=:none)
    return ModuleComplexes.hyperExt(C, N, df; kwargs...)
end

hyperext(C::ModuleComplexes.ModuleCochainComplex{QQ}, N::EncodingResult; kwargs...) =
    hyperext(C, N.M; kwargs...)


"""
    hypertor(Rop, C; maxdeg=3, kwargs...)

Compute HyperTor_t(Rop, C) for a right-module Rop (on P^op) and a module complex C (on P).
"""
function hypertor(Rop::Modules.PModule{QQ}, C::ModuleComplexes.ModuleCochainComplex{QQ};
                  maxdeg::Int=3, kwargs...)
    df = DerivedFunctorOptions(; maxdeg=maxdeg, model=:auto, canon=:none)
    return ModuleComplexes.hyperTor(Rop, C, df; kwargs...)
end

hypertor(Rop::EncodingResult, C::ModuleComplexes.ModuleCochainComplex{QQ}; kwargs...) =
    hypertor(Rop.M, C; kwargs...)



# Internal helper: call an invariant function with a few common signatures.
function _call_invariant(f, enc::EncodingResult, opts::InvariantOptions; kwargs...)
    M = enc.M
    pi = enc.pi

    if hasmethod(f, Tuple{typeof(M), typeof(pi), typeof(opts)})
        return f(M, pi, opts; kwargs...)
    elseif hasmethod(f, Tuple{typeof(M), typeof(pi)})
        return f(M, pi; kwargs...)
    elseif hasmethod(f, Tuple{typeof(M), typeof(opts)})
        return f(M, opts; kwargs...)
    elseif hasmethod(f, Tuple{typeof(M)})
        return f(M; kwargs...)
    else
        error("invariants: no method found for $(f) with signatures (M), (M,pi), (M,opts), or (M,pi,opts)")
    end
end

# Internal helper: call an invariant function with a few common signatures.
function _call_invariant(f, enc::EncodingResult, opts::InvariantOptions; kwargs...)
    if hasmethod(f, Tuple{typeof(enc.M), typeof(enc.pi), typeof(opts)})
        return f(enc.M, enc.pi, opts; kwargs...)
    elseif hasmethod(f, Tuple{typeof(enc.M), typeof(enc.pi)})
        return f(enc.M, enc.pi; kwargs...)
    elseif hasmethod(f, Tuple{typeof(enc.M), typeof(opts)})
        return f(enc.M, opts; kwargs...)
    elseif hasmethod(f, Tuple{typeof(enc.M)})
        return f(enc.M; kwargs...)
    else
        error("invariant: no method found for invariant function $(f). Expected f(M), f(M,pi), f(M,opts), or f(M,pi,opts).")
    end
end


"""
    invariant(enc::EncodingResult; which=:rank_invariant, opts=InvariantOptions(), kwargs...) -> InvariantResult

Compute a single invariant from the `Invariants` submodule and wrap it in an `InvariantResult`.

`which` may be:
- a Symbol naming a function in `Invariants` (e.g. `:rank_invariant`)
- a callable itself
"""
function invariant(enc::EncodingResult;
                   which=:rank_invariant,
                   opts::Union{InvariantOptions,Nothing}=nothing,
                   kwargs...)
    opts = _resolve_invariant_opts(opts)
    f = which isa Symbol ? getfield(Invariants, which) : which
    val = _call_invariant(f, enc, opts; kwargs...)
    return InvariantResult(enc, which, val; opts=opts, meta=NamedTuple())
end


"""
    invariants(enc::EncodingResult; which=..., opts=InvariantOptions(), kwargs...) -> Vector{InvariantResult}

Batch convenience wrapper around `invariant`.

`which` may be:
- a Symbol or callable (compute one)
- a vector of Symbols / callables (compute many)
"""
function invariants(enc::EncodingResult;
                    which=[:rank_invariant],
                    opts::Union{InvariantOptions,Nothing}=nothing,
                    kwargs...)
    opts = _resolve_invariant_opts(opts)
    if which isa AbstractVector
        return [invariant(enc; which=w, opts=opts, kwargs...) for w in which]
    else
        return [invariant(enc; which=which, opts=opts, kwargs...)]
    end
end


# -----------------------------------------------------------------------------
# Curated invariant entrypoints (stable value-returning wrappers)
# -----------------------------------------------------------------------------

rank_invariant(M::PModule{QQ}, opts::Union{InvariantOptions,Nothing}=nothing; kwargs...) =
    Invariants.rank_invariant(M, _resolve_invariant_opts(opts); kwargs...)

rank_map(M::PModule{QQ}, pi, x, y, opts::Union{InvariantOptions,Nothing}=nothing; kwargs...) =
    Invariants.rank_map(M, pi, x, y, _resolve_invariant_opts(opts); kwargs...)
rank_map(M::PModule{QQ}, a::Int, b::Int; kwargs...) = Invariants.rank_map(M, a, b; kwargs...)

restricted_hilbert(M::PModule{QQ}) = Invariants.restricted_hilbert(M)

restricted_hilbert(M::PModule{QQ}, pi, x, opts::Union{InvariantOptions,Nothing}=nothing; kwargs...) =
    Invariants.restricted_hilbert(M, pi, x, _resolve_invariant_opts(opts); kwargs...)

function euler_surface(M::PModule{QQ}, pi, opts::Union{InvariantOptions,Nothing}=nothing; kwargs...)
    opts = _resolve_invariant_opts(opts)
    if any(haskey(kwargs, k) for k in (:axes, :axes_policy, :max_axis_len, :box, :threads, :strict))
        opts = InvariantOptions(
            axes = get(kwargs, :axes, opts.axes),
            axes_policy = get(kwargs, :axes_policy, opts.axes_policy),
            max_axis_len = get(kwargs, :max_axis_len, opts.max_axis_len),
            box = get(kwargs, :box, opts.box),
            threads = get(kwargs, :threads, opts.threads),
            strict = get(kwargs, :strict, opts.strict),
        )
        kw_nt = NamedTuple(kwargs)
        kwargs = Base.structdiff(kw_nt, (; axes=nothing, axes_policy=nothing, max_axis_len=nothing, box=nothing, threads=nothing, strict=nothing))
    end
    return Invariants.euler_surface(M, pi, opts; kwargs...)
end

euler_surface(C::ModuleCochainComplex{QQ}, pi, opts::Union{InvariantOptions,Nothing}=nothing; kwargs...) =
    Invariants.euler_surface(C, pi, _resolve_invariant_opts(opts); kwargs...)

slice_barcode(M::PModule{QQ}, chain::AbstractVector{Int}, opts::Union{InvariantOptions,Nothing}=nothing; kwargs...) =
    Invariants.slice_barcode(M, chain; kwargs...)

slice_barcodes(M::PModule{QQ}, chains::AbstractVector, opts::Union{InvariantOptions,Nothing}=nothing; kwargs...) =
    Invariants.slice_barcodes(M, chains; kwargs...)

rank_invariant(enc::EncodingResult; opts::Union{InvariantOptions,Nothing}=nothing, kwargs...) =
    invariant(enc; which=:rank_invariant, opts=opts, kwargs...).value

restricted_hilbert(enc::EncodingResult; opts::Union{InvariantOptions,Nothing}=nothing, kwargs...) =
    invariant(enc; which=:restricted_hilbert, opts=opts, kwargs...).value

euler_surface(enc::EncodingResult; opts::Union{InvariantOptions,Nothing}=nothing, kwargs...) =
    invariant(enc; which=:euler_surface, opts=opts, kwargs...).value

ecc(enc::EncodingResult; opts::Union{InvariantOptions,Nothing}=nothing, kwargs...) =
    invariant(enc; which=:ecc, opts=opts, kwargs...).value

slice_barcode(enc::EncodingResult; opts::Union{InvariantOptions,Nothing}=nothing, kwargs...) =
    invariant(enc; which=:slice_barcode, opts=opts, kwargs...).value

slice_barcodes(enc::EncodingResult; opts::Union{InvariantOptions,Nothing}=nothing, kwargs...) =
    invariant(enc; which=:slice_barcodes, opts=opts, kwargs...).value

mp_landscape(enc::EncodingResult; opts::Union{InvariantOptions,Nothing}=nothing, kwargs...) =
    invariant(enc; which=:mp_landscape, opts=opts, kwargs...).value
mp_landscape(M::PModule{QQ}, slices::AbstractVector; kwargs...) =
    Invariants.mp_landscape(M, slices; kwargs...)
function mp_landscape(M::PModule{QQ}, pi; kwargs...)
    opt_keys = (:box, :strict, :threads, :axes, :axes_policy, :max_axis_len)
    if any(k -> haskey(kwargs, k), opt_keys)
        base = InvariantOptions()
        opts = InvariantOptions(
            box = get(kwargs, :box, base.box),
            strict = get(kwargs, :strict, base.strict),
            threads = get(kwargs, :threads, base.threads),
            axes = get(kwargs, :axes, base.axes),
            axes_policy = get(kwargs, :axes_policy, base.axes_policy),
            max_axis_len = get(kwargs, :max_axis_len, base.max_axis_len),
        )
        kwargs_nt = NamedTuple(kwargs)
        kwargs2 = (; (k => v for (k, v) in pairs(kwargs_nt) if !(k in opt_keys))...)
        return Invariants.mp_landscape(M, pi, opts; kwargs2...)
    end
    return Invariants.mp_landscape(M, pi; kwargs...)
end

mpp_decomposition(enc::EncodingResult; opts::Union{InvariantOptions,Nothing}=nothing, kwargs...) =
    invariant(enc; which=:mpp_decomposition, opts=opts, kwargs...).value

mpp_image(enc::EncodingResult; opts::Union{InvariantOptions,Nothing}=nothing, kwargs...) =
    invariant(enc; which=:mpp_image, opts=opts, kwargs...).value


"""
    matching_distance(encA, encB; method=:auto, opts=InvariantOptions(), kwargs...)

Distance between two encoded modules, assuming a common encoding map `pi` (as produced by
`encode([A,B]; ...)` style common-encoding).

`method`:
- `:auto` or `:approx`    uses `Invariants.matching_distance_approx`
- `:exact_2d`             uses `Invariants.matching_distance_exact_2d`
"""
function matching_distance(encA::EncodingResult, encB::EncodingResult;
                           method::Symbol=:auto,
                           opts::Union{InvariantOptions,Nothing}=nothing,
                           kwargs...)
    opts = _resolve_invariant_opts(opts)
    (encA.P === encB.P) || error("matching_distance: encodings are on different posets; common-encode first.")
    (encA.pi === encB.pi) || error("matching_distance: encodings do not share a common classifier map pi; common-encode first.")

    pi = encA.pi
    if method == :auto || method == :approx
        return Invariants.matching_distance_approx(encA.M, encB.M, pi, opts; kwargs...)
    elseif method == :exact_2d
        return Invariants.matching_distance_exact_2d(encA.M, encB.M, pi, opts; kwargs...)
    else
        error("matching_distance: unknown method=$(method). Supported: :auto, :approx, :exact_2d")
    end
end
