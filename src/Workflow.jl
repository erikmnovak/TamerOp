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
        return length(Ups[1].lo)
    elseif !isempty(Downs)
        return length(Downs[1].lo)
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
function supports_pl_backend(Ups::Vector{BoxUpset}, Downs::Vector{BoxDownset}; opts::EncodingOptions=EncodingOptions())::Bool
    max_regions = opts.max_regions
    # PLBackend can only handle finite, axis-aligned boxes (encoded as BoxUpset/BoxDownset).
    # If any generator has +/-Inf bounds, or other invalid bounds, reject here.
    for U in Ups
        for (a,b) in zip(U.lo, U.hi)
            if !(isfinite(a) && isfinite(b))
                return false
            end
        end
    end
    for D in Downs
        for (a,b) in zip(D.lo, D.hi)
            if !(isfinite(a) && isfinite(b))
                return false
            end
        end
    end

    # If max_regions is set, do a cheap upper bound check on the grid size.
    # PLBackend partitions via coordinate grid from generator endpoints.
    if max_regions !== nothing
        n = _infer_box_dim(Ups, Downs)
        coords = [QQ[] for _ in 1:n]
        for U in Ups
            for i in 1:n
                push!(coords[i], U.lo[i])
                push!(coords[i], U.hi[i])
            end
        end
        for D in Downs
            for i in 1:n
                push!(coords[i], D.lo[i])
                push!(coords[i], D.hi[i])
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
function supports_pl_backend(F::PLPolyhedra.PLFringe; opts::EncodingOptions=EncodingOptions())::Bool
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
        lo = Ups[i].lo
        hi = Ups[i].hi
        A = vcat(Matrix{QQ}(I, n, n), -Matrix{QQ}(I, n, n))
        b = vcat(lo, -hi)
        PUps[i] = PLPolyhedra.PLUpset(A, b)
    end

    PDowns = Vector{PLPolyhedra.PLDownset}(undef, length(Downs))
    for i in eachindex(Downs)
        lo = Downs[i].lo
        hi = Downs[i].hi
        A = vcat(Matrix{QQ}(I, n, n), -Matrix{QQ}(I, n, n))
        b = vcat(lo, -hi)
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
function choose_pl_backend(Ups::Vector{BoxUpset}, Downs::Vector{BoxDownset}; opts::EncodingOptions=EncodingOptions())::Symbol
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

function choose_pl_backend(F::PLPolyhedra.PLFringe; opts::EncodingOptions=EncodingOptions())::Symbol
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
                            opts::EncodingOptions=EncodingOptions())
    b = choose_pl_backend(Ups, Downs; opts=opts)
    if b == :pl_backend
        return encode_fringe_boxes(Ups, Downs, Phi; opts=opts)
    else
        F = pl_fringe_from_boxes(Ups, Downs)
        return PLPolyhedra.encode_from_PL_fringe(F, opts)
    end
end

function encode_from_fringe(F::PLPolyhedra.PLFringe, opts::EncodingOptions=EncodingOptions())
    b = choose_pl_backend(F; opts=opts)
    if b == :pl_backend
        Ups, Downs = boxes_from_pl_fringe(F)
        Phi = reshape(ones(QQ, length(Downs) * length(Ups)), length(Downs), length(Ups))
        return encode_fringe_boxes(Ups, Downs, Phi; opts=opts)
    else
        return PLPolyhedra.encode_from_PL_fringe(F, opts)
    end
end

"""
    encode_pmodule_from_fringe(...)

Return (M, pi) where M is the finite-poset PModule obtained from the encoded fringe module.
"""
function encode_pmodule_from_fringe(Ups::Vector{BoxUpset}, Downs::Vector{BoxDownset}, Phi::AbstractMatrix{QQ},
                                    opts::EncodingOptions=EncodingOptions())
    P, H, pi = encode_from_fringe(Ups, Downs, Phi, opts)
    return pmodule_from_fringe(H), pi
end

function encode_pmodule_from_fringe(F::PLPolyhedra.PLFringe, opts::EncodingOptions=EncodingOptions())
    P, H, pi = encode_from_fringe(F, opts)
    return pmodule_from_fringe(H), pi
end

# -----------------------------------------------------------------------------
# Workflow entrypoints (narrative API)

"""
    encode(x; backend=:auto, max_regions=nothing, strict_eps=nothing) -> EncodingResult
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
function encode(x; backend::Symbol=:auto, max_regions=nothing, strict_eps=nothing)
    enc = EncodingOptions(; backend=backend, max_regions=max_regions, strict_eps=strict_eps)
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

encode(FG1::Flange{QQ}, FG2::Flange{QQ}; kwargs...) =
    encode(Flange{QQ}[FG1, FG2]; kwargs...)

encode(FG1::Flange{QQ}, FG2::Flange{QQ}, FG3::Flange{QQ}; kwargs...) =
    encode(Flange{QQ}[FG1, FG2, FG3]; kwargs...)

# -----------------------
# R^n presentations

function encode(F::PLPolyhedra.PLFringe, enc::EncodingOptions=EncodingOptions())
    P, H, pi = encode_from_fringe(F, enc)
    M = pmodule_from_fringe(H)
    b = choose_pl_backend(F; opts=enc)
    return EncodingResult(P, M, pi; H=H, presentation=F, opts=enc, backend=b, meta=(;))
end

function encode(Ups::Vector{BoxUpset}, Downs::Vector{BoxDownset}, Phi::AbstractMatrix{QQ},
                enc::EncodingOptions=EncodingOptions())
    P, H, pi = encode_from_fringe(Ups, Downs, Phi, enc)
    M = pmodule_from_fringe(H)
    b = choose_pl_backend(Ups, Downs; opts=enc)
    return EncodingResult(P, M, pi;
                          H=H, presentation=(Ups=Ups, Downs=Downs, Phi=Phi),
                          opts=enc, backend=b, meta=(;))
end

# Convenience overloads for common BoxFringe encodings
encode(Ups::Vector{BoxUpset}, Downs::Vector{BoxDownset}, enc::EncodingOptions=EncodingOptions()) =
    encode(Ups, Downs, reshape(ones(QQ, length(Downs) * length(Ups)), length(Downs), length(Ups)), enc)

function encode(Ups::Vector{BoxUpset}, Downs::Vector{BoxDownset}, Phi_vec::AbstractVector{QQ},
                enc::EncodingOptions=EncodingOptions())
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
    P, Hs, pi = PLPolyhedra.encode_from_PL_fringes(Fs, enc)
    out = Vector{EncodingResult}(undef, length(Fs))
    for i in eachindex(Fs)
        H = Hs[i]
        out[i] = EncodingResult(P, pmodule_from_fringe(H), pi;
                                H=H, presentation=Fs[i], opts=enc, backend=:pl, meta=(;))
    end
    return out
end

encode(F1::PLPolyhedra.PLFringe, F2::PLPolyhedra.PLFringe; kwargs...) =
    encode(PLPolyhedra.PLFringe[F1, F2]; kwargs...)

encode(F1::PLPolyhedra.PLFringe, F2::PLPolyhedra.PLFringe, F3::PLPolyhedra.PLFringe; kwargs...) =
    encode(PLPolyhedra.PLFringe[F1, F2, F3]; kwargs...)

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
        d[:coarsen_n_before] = enc.P.n
        d[:coarsen_n_after] = pi.P.n
        d
    elseif enc.meta isa NamedTuple
        merge(enc.meta, (coarsen_method = method, coarsen_n_before = enc.P.n, coarsen_n_after = pi.P.n))
    else
        Dict{Symbol,Any}(
            :orig_meta => enc.meta,
            :coarsen_method => method,
            :coarsen_n_before => enc.P.n,
            :coarsen_n_after => pi.P.n,
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

"""
    is_minimal(res::ResolutionResult) -> Bool

Return whether the resolution stored in `res` was proven minimal by the chosen backend.
"""
is_minimal(res::ResolutionResult) = res.minimal



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
    L.P.leq == transpose(Rop.P.leq)
"""
function tor(Rop::EncodingResult, L::EncodingResult;
             maxdeg::Int=3, model::Symbol=:auto)
    (L.P.leq == transpose(Rop.P.leq)) || error("tor: expected first argument on opposite poset of the second.")
    df = DerivedFunctorOptions(; maxdeg=maxdeg, model=model, canon=:none)
    return DerivedFunctors.Tor(Rop.M, L.M, df)
end

function tor(Rop::Modules.PModule{QQ}, L::Modules.PModule{QQ};
             maxdeg::Int=3, model::Symbol=:auto)
    (L.Q.leq == transpose(Rop.Q.leq)) || error("tor: expected first argument on opposite poset of the second.")
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
                 opts::ResolutionOptions=ResolutionOptions(),
                 minimality::Bool=false,
                 check_hull::Bool=true)
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
                   opts::InvariantOptions=InvariantOptions(),
                   kwargs...)
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
                    opts::InvariantOptions=InvariantOptions(),
                    kwargs...)
    if which isa AbstractVector
        return [invariant(enc; which=w, opts=opts, kwargs...) for w in which]
    else
        return [invariant(enc; which=which, opts=opts, kwargs...)]
    end
end


# -----------------------------------------------------------------------------
# Curated invariant entrypoints (stable value-returning wrappers)
# -----------------------------------------------------------------------------

rank_invariant(enc::EncodingResult; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    invariant(enc; which=:rank_invariant, opts=opts, kwargs...).value

restricted_hilbert(enc::EncodingResult; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    invariant(enc; which=:restricted_hilbert, opts=opts, kwargs...).value

euler_surface(enc::EncodingResult; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    invariant(enc; which=:euler_surface, opts=opts, kwargs...).value

ecc(enc::EncodingResult; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    invariant(enc; which=:ecc, opts=opts, kwargs...).value

slice_barcode(enc::EncodingResult; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    invariant(enc; which=:slice_barcode, opts=opts, kwargs...).value

slice_barcodes(enc::EncodingResult; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    invariant(enc; which=:slice_barcodes, opts=opts, kwargs...).value

mp_landscape(enc::EncodingResult; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    invariant(enc; which=:mp_landscape, opts=opts, kwargs...).value

mpp_decomposition(enc::EncodingResult; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    invariant(enc; which=:mpp_decomposition, opts=opts, kwargs...).value

mpp_image(enc::EncodingResult; opts::InvariantOptions=InvariantOptions(), kwargs...) =
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
                           opts::InvariantOptions=InvariantOptions(),
                           kwargs...)
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
