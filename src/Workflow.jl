"""
    Workflow

User-facing orchestration layer for the canonical mathematical tasks in
TamerOp.

Canonical workflow policy
- prefer task-oriented entrypoints:
  `encode`, `common_refinement`, `restriction`, `pushforward_left`,
  `pushforward_right`, `resolve`, `betti_table`, `bass_table`,
  `invariant`, `invariants`, `hom_dimension`, `hom`, `ext`, `tor`, and
  `matching_distance_exact_2d`,
- prefer typed result wrappers such as [`EncodingResult`](@ref),
  [`EncodedComplexResult`](@ref), [`ResolutionResult`](@ref), and
  [`InvariantResult`](@ref),
- treat raw tuples as explicit opt-in only, e.g. `encode(...; output=:raw)`,
- use the top-level cache contract `cache=:auto` for one-shot calls or
  `cache=sc::SessionCache` for explicit cross-call reuse.

Workflow output policy
- `encode`, `resolve`, and `invariant` return typed result wrappers by
  default,
- `restriction`, `pushforward_left`, and `pushforward_right` return
  [`ModuleTranslationResult`](@ref) wrappers that preserve workflow
  provenance,
- `hom`, `ext`, `tor`, `rhom`, `hyperext`, and `hypertor` return algebraic
  objects directly,
- convenience wrappers such as `rank_invariant(enc)`, `betti_table(enc)`,
  `bass_table(enc)`, `matching_distance_exact_2d(encA, encB)`, or
  `mpp_image(enc)` return bare values,
- raw tuples and intermediate ingestion stages remain explicit opt-in only.

Ownership map
- `encode`, `coarsen`, `common_refinement`, `restriction`,
  `pushforward_left`, and `pushforward_right` own workflow-facing
  encoding/change-of-poset orchestration,
- `resolve`, `hom_dimension`, `hom`, `ext`, `tor`, and `ext_algebra` own the
  high-level algebraic workflow wrappers,
- `invariant`, `invariants`, `slice_barcodes`, `mp_landscape`, and the
  value-returning invariant shims own the high-level invariant workflow
  surface,
- the private cache helpers in this file translate the single workflow cache
  contract into owner-specific caches.

What this module does not own
- encoding backends and compiled encoding internals; those belong to
  `EncodingCore`, `Encoding`, `ZnEncoding`, `PLBackend`, and `PLPolyhedra`,
- module algebra, resolutions, and derived functors; those belong to
  `Modules`, `IndicatorResolutions`, `DerivedFunctors`, and
  `ModuleComplexes`,
- invariant kernels; those belong to `Invariants`, `SignedMeasures`,
  `SliceInvariants`, `Fibered2D`, and `MultiparameterImages`.

Contributor note
- keep this file thin and task-oriented,
- add logic here only when it standardizes the public workflow contract or
  unifies cache/output behavior across owners,
- push owner-specific algorithms, planners, and heavy kernels back into the
  subsystem that actually owns them.

Canonical workflows

```jldoctest
julia> using TamerOp

julia> const PLB = TamerOp.PLBackend;

julia> enc = TamerOp.encode([PLB.BoxUpset([0.0])], [PLB.BoxDownset([2.0])];
                                 backend=:pl_backend);

julia> TamerOp.result_summary(enc).kind
:encoding_result

julia> res = TamerOp.resolve(enc);

julia> TamerOp.describe(res).kind
:resolution_result
```

```jldoctest
julia> using TamerOp

julia> const FZ = TamerOp.FlangeZn;

julia> tau = FZ.face(2, []);

julia> FM = FZ.Flange(2,
                      [FZ.IndFlat(tau, [0, 0]; id=:F_M)],
                      [FZ.IndInj(tau, [2, 2]; id=:E_M)],
                      reshape([TamerOp.QQ(1)], 1, 1));

julia> FN = FZ.Flange(2,
                      [FZ.IndFlat(tau, [1, 1]; id=:F_N)],
                      [FZ.IndInj(tau, [3, 3]; id=:E_N)],
                      reshape([TamerOp.QQ(1)], 1, 1));

julia> encM, encN = TamerOp.encode((FM, FN); backend=:zn);

julia> TamerOp.hom_dimension(encM, encN) >= 0
true

julia> TamerOp.dim(TamerOp.ext(encM, encN; maxdeg=0), 0) >= 0
true
```

```jldoctest
julia> using TamerOp

julia> const PLB = TamerOp.PLBackend;

julia> enc = TamerOp.encode([PLB.BoxUpset([0.0])], [PLB.BoxDownset([2.0])];
                                 backend=:pl_backend);

julia> inv = TamerOp.invariant(enc; which=:restricted_hilbert);

julia> length(TamerOp.invariants(enc; which=[:restricted_hilbert, :rank_invariant]))
2
```

```jldoctest
julia> using TamerOp

julia> const PLB = TamerOp.PLBackend;

julia> enc = TamerOp.encode([PLB.BoxUpset([0.0])], [PLB.BoxDownset([2.0])];
                                 backend=:pl_backend);

julia> bars = TamerOp.slice_barcodes(enc; opts=TamerOp.InvariantOptions(threads=false));

julia> length(bars) >= 1
true

julia> TamerOp.describe(TamerOp.mp_landscape(enc; opts=TamerOp.InvariantOptions(threads=false))).kind
:mp_landscape
```

```jldoctest
julia> using TamerOp

julia> const PLB = TamerOp.PLBackend;

julia> enc_auto = TamerOp.encode([PLB.BoxUpset([0.0])], [PLB.BoxDownset([2.0])];
                                      backend=:pl_backend);

julia> sc = TamerOp.SessionCache();

julia> enc_cached = TamerOp.encode([PLB.BoxUpset([0.0])], [PLB.BoxDownset([2.0])];
                                        backend=:pl_backend, cache=sc);

julia> TamerOp.describe(TamerOp.invariant(enc_cached; which=:restricted_hilbert, cache=sc)).kind
:invariant_result

julia> typeof(enc_auto) == typeof(enc_cached)
true
```
"""

module Workflow

using LinearAlgebra  # UniformScaling I / transpose
using SparseArrays
using JSON3
using Dates

# Keep this file self-contained: import the handful of names it mentions
# in type annotations or uses unqualified.
using ..CoreModules: QQ, QQField, AbstractCoeffField, coeff_type, coerce,
                    ResolutionCache, SessionCache, EncodingCache,
                    _encoding_cache!, _session_resolution_cache, _session_hom_cache, _set_session_hom_cache!,
                    _session_slice_plan_cache, _set_session_slice_plan_cache!,
                    _field_cache_key,
                    _resolve_workflow_session_cache, _resolve_workflow_specialized_cache,
                    _workflow_encoding_cache,
                    _session_get_zn_pushforward_fringe, _session_set_zn_pushforward_fringe!,
                    _session_get_zn_pushforward_module, _session_set_zn_pushforward_module!,
                    _resolution_cache_from_session, _slot_cache_from_session,
                    GeometryCachePayload
using ..Options: EncodingOptions, ResolutionOptions, DerivedFunctorOptions, InvariantOptions,
                 ConstructionBudget, ConstructionOptions, PipelineOptions
using ..DataTypes: PointCloud, ImageNd, GraphData, EmbeddedPlanarGraph2D,
                   GradedComplex, MultiCriticalGradedComplex,
                   SimplexTreeMulti, simplex_count, max_simplex_dim, simplex_vertices, simplex_grades
using ..EncodingCore: AbstractPLikeEncodingMap, CompiledEncoding, GridEncodingMap,
                      _compile_encoding_cached
using ..Results: EncodingResult, EncodedComplexResult, CohomologyDimsResult, ModuleTranslationResult, ResolutionResult, InvariantResult,
                 _encoding_with_session_cache, materialize_module, module_dims, encoding_complex, _materialize_complex
import ..EncodingCore: locate, dimension, representatives, axes_from_encoding, _grid_strides,
                      GridEncodingMap
import ..Serialization
import ..Serialization: TAMER_FEATURE_SCHEMA_VERSION, feature_schema_header, validate_feature_metadata_schema

import ..IndicatorResolutions
import ..ZnEncoding
import ..FiniteFringe
using ..IndicatorResolutions: pmodule_from_fringe
import ..IndicatorResolutions: fringe_presentation
using ..PLPolyhedra
using ..PLBackend: BoxUpset, BoxDownset, encode_fringe_boxes
using ..Encoding: build_uptight_encoding_from_fringe,
                 pushforward_fringe_along_encoding,
                 PostcomposedEncodingMap
using ..Modules: PModule, PMorphism, cover_edges, dim_at
using ..FiniteFringe: AbstractPoset, FinitePoset, GridPoset, ProductOfChainsPoset, FringeModule,
                     Upset, Downset, principal_upset, principal_downset, leq, nvertices,
                     poset_equal_opposite
using ..DerivedFunctors
using ..Invariants
using ..SignedMeasures
using ..SliceInvariants: CompiledSlicePlan, SlicePlanCache
import ..Fibered2D
import ..ChangeOfPosets
import ..Modules
import ..ModuleComplexes
using ..ModuleComplexes: ModuleCochainComplex


import ..FlangeZn
using ..FlangeZn: Face, IndFlat, IndInj, Flange, change_field

@inline function _hom_cache_from_session(cache::Union{Nothing,DerivedFunctors.HomSystemCache},
                                         session_cache::Union{Nothing,SessionCache},
                                         ::Type{K}) where {K}
    return _slot_cache_from_session(
        cache,
        session_cache,
        _session_hom_cache,
        _set_session_hom_cache!,
        DerivedFunctors.HomSystemCache{DerivedFunctors.HomSpace{K},SparseMatrixCSC{K,Int},SparseMatrixCSC{K,Int}},
        () -> DerivedFunctors.HomSystemCache{K}(),
    )
end

@inline function _slice_plan_cache_from_session(cache::Union{Nothing,SlicePlanCache},
                                                session_cache::Union{Nothing,SessionCache})
    return _slot_cache_from_session(
        cache,
        session_cache,
        _session_slice_plan_cache,
        _set_session_slice_plan_cache!,
        SlicePlanCache,
        SlicePlanCache,
    )
end

@inline function _workflow_zn_pushforward_module(P,
                                                 pi,
                                                 FG::Flange,
                                                 session_cache::Union{Nothing,SessionCache},
                                                 poset_kind::Symbol)
    flange_fp = ZnEncoding._flange_presentation_fingerprint(FG)
    field_key = _field_cache_key(FG.field)
    if session_cache !== nothing
        cached = _session_get_zn_pushforward_module(session_cache, pi.encoding_fingerprint, poset_kind, flange_fp, field_key)
        if cached !== nothing
            item = cached::NamedTuple
            return item.H, item.M
        end
    end

    plan, flat_masks, inj_masks = ZnEncoding._strict_pushforward_plan_and_masks(pi, FG; session_cache=session_cache)
    M = ZnEncoding._pmodule_from_pushforward_plan(P, FG, plan, flat_masks, inj_masks)
    H = if session_cache === nothing
        ZnEncoding._strict_pushforward_fringe_from_plan(P, FG, plan, flat_masks, inj_masks)
    else
        cached_H = _session_get_zn_pushforward_fringe(session_cache, pi.encoding_fingerprint, poset_kind, flange_fp, field_key)
        if cached_H !== nothing
            cached_H
        else
            Hnew = ZnEncoding._strict_pushforward_fringe_from_plan(P, FG, plan, flat_masks, inj_masks)
            _session_set_zn_pushforward_fringe!(session_cache, pi.encoding_fingerprint, poset_kind, flange_fp, field_key, Hnew)
        end
    end

    session_cache === nothing || _session_set_zn_pushforward_module!(session_cache, pi.encoding_fingerprint, poset_kind, flange_fp, field_key, (H=H, M=M))
    return H, M
end

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
                             opts::EncodingOptions=EncodingOptions())::Bool
    opts = opts
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
function supports_pl_backend(F::PLPolyhedra.PLFringe; opts::EncodingOptions=EncodingOptions())::Bool
    opts = opts
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
                           opts::EncodingOptions=EncodingOptions())::Symbol
    opts = opts
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
    opts = opts
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
function encode_from_fringe(Ups::Vector{BoxUpset}, Downs::Vector{BoxDownset}, Phi::AbstractMatrix,
                            opts::EncodingOptions=EncodingOptions())
    opts = opts
    b = choose_pl_backend(Ups, Downs; opts=opts)
    Phi_QQ = Matrix{QQ}(Phi)
    if b == :pl_backend
        P, H, pi = encode_fringe_boxes(Ups, Downs, Phi_QQ, opts; poset_kind=opts.poset_kind)
    else
        F = pl_fringe_from_boxes(Ups, Downs)
        P, H, pi = PLPolyhedra.encode_from_PL_fringe(F, opts; poset_kind=opts.poset_kind)
    end
    H = (opts.field == QQField()) ? H : change_field(H, opts.field)
    return P, H, pi
end

function encode_from_fringe(F::PLPolyhedra.PLFringe, opts::EncodingOptions=EncodingOptions())
    opts = opts
    b = choose_pl_backend(F; opts=opts)
    if b == :pl_backend
        Ups, Downs = boxes_from_pl_fringe(F)
        Phi = reshape(ones(QQ, length(Downs) * length(Ups)), length(Downs), length(Ups))
        P, H, pi = encode_fringe_boxes(Ups, Downs, Phi, opts; poset_kind=opts.poset_kind)
    else
        P, H, pi = PLPolyhedra.encode_from_PL_fringe(F, opts; poset_kind=opts.poset_kind)
    end
    H = (opts.field == QQField()) ? H : change_field(H, opts.field)
    return P, H, pi
end

# -----------------------------------------------------------------------------
# Presentation conversions (module <-> fringe/flange)
# -----------------------------------------------------------------------------

"""
    flange_presentation(M::PModule{K}, pi::GridEncodingMap) -> Flange{K}

Construct a Z^n flange presentation for an integer-valued grid encoding.
"""
function _workflow_grid_tuples(sizes::NTuple{N,Int}) where {N}
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

function _workflow_poset_vertex_coords(axes::NTuple{N,Vector{T}}) where {N,T}
    sizes = ntuple(i -> length(axes[i]), N)
    idxs = _workflow_grid_tuples(sizes)
    coords = Vector{Vector{Int}}(undef, length(idxs))
    for i in eachindex(idxs)
        coords[i] = [Int(axes[j][idxs[i][j]]) for j in 1:N]
    end
    return coords
end

function flange_presentation(M::PModule{K}, pi::GridEncodingMap) where {K}
    P = M.Q
    field = M.field
    n = length(pi.coords)
    for ax in pi.coords
        all(x -> x isa Integer || (x isa Real && isinteger(x)), ax) ||
            error("flange_presentation: axes must be integer-valued.")
    end
    F0, pi0, gens_at_F0 = IndicatorResolutions.projective_cover(M)
    E0, iota0, gens_at_E0 = IndicatorResolutions._injective_hull(M)

    comps = Vector{Matrix{K}}(undef, nvertices(P))
    for i in 1:nvertices(P)
        comps[i] = iota0.comps[i] * pi0.comps[i]
    end
    f = PMorphism{K}(F0, E0, comps)

    tau = Face(n, falses(n))
    coords = _workflow_poset_vertex_coords(pi.coords)

    flats = IndFlat{n}[]
    for p in 1:nvertices(P)
        for _ in gens_at_F0[p]
            push!(flats, IndFlat(tau, coords[p]; id=:F))
        end
    end

    injectives = IndInj{n}[]
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

    Phi = spzeros(K, length(injectives), length(flats))
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

    return Flange{K}(n, flats, injectives, Matrix{K}(Phi); field=field)
end

@inline function _normalize_encoding_output(output::Symbol)
    if output === :result || output === :encoding_result
        return :result
    elseif output === :raw
        return :raw
    end
    error("output must be :result, :encoding_result, or :raw (got $output)")
end

@inline function _return_encoding(res::EncodingResult, output::Symbol)
    normalized = _normalize_encoding_output(output)
    return normalized === :result ? res : (res.P, res.H, res.pi)
end

# -----------------------------------------------------------------------------
# Workflow entrypoints (narrative API)

@doc raw"""
    encode(...)

Canonical workflow entrypoint with two distinct families that share one name.

Presentation encoding
- `encode(x; backend=:auto, output=:result, cache=:auto)` takes a mathematical
  presentation object such as a `Flange`, `PLFringe`, box-generator
  presentation, or `SyntheticBoxFringe`.
- The canonical simple-user result is `EncodingResult`.
- `output=:raw` is advanced opt-in and returns the storage-shaped tuple
  `(P, H, pi)` for the same encoding task.

Data-ingestion execution
- `encode(data, filtration; stage=:auto, cache=:auto, ...)` runs the graded
  ingestion pipeline from typed data plus a filtration/specification.
- `encode(plan::IngestionPlan; degree=0)` is the canonical final-stage entry
  when you already prepared a plan with `plan_ingestion(...)`.
- `stage` selects where the ingestion pipeline stops; `stage=:encoding_result`
  is the canonical simple-user final stage.

`output` vs `stage`
- `output` belongs to the presentation-encoding family and chooses between a
  typed wrapper and an explicit raw payload for one encoding task.
- `stage` belongs to the ingestion family and chooses which intermediate or
  final pipeline object to return.

Workflow output policy
- `encode`, `resolve`, and `invariant` return typed wrappers by default,
- `hom`, `ext`, `tor`, `rhom`, `hyperext`, and `hypertor` return algebraic
  objects directly,
- convenience invariant wrappers such as `rank_invariant(enc)` return bare
  values,
- raw tuples and intermediate pipeline stages are explicit opt-in only.

Use `describe(...)` or owner-local helpers such as `result_summary(...)` before
materializing heavier downstream objects.
""" encode

"""
    encode(x; backend=:auto, max_regions=nothing, strict_eps=nothing, poset_kind=:signature,
           output=:result, cache=:auto) -> EncodingResult
    encode(xs::AbstractVector/tuple; backend=:auto, ...) -> Vector{EncodingResult}

Canonical high-level entrypoint that turns a presentation object into a finite
encoding-poset model.

- Z^n inputs (Flange{K}) use the Zn encoder (backend=:auto or :zn).
- R^n inputs (`PLFringe`, box generators, or `SyntheticBoxFringe`) use:
    * PLBackend when possible (axis-aligned, finite bounds, and not too many regions),
    * otherwise PLPolyhedra (if Polyhedra/CDDLib are available).

The returned EncodingResult stores:
- P : the finite encoding poset
- H : the pushed-down fringe module on P (optional but populated here)
- M : the finite-poset module pmodule_from_fringe(H)
- pi: the classifier map from the original domain to P
- presentation / opts / backend / meta : provenance

Cache contract
- `cache=:auto` allocates a one-shot workflow cache for the call,
- `cache=sc::SessionCache` reuses encoding, translation, and downstream algebra
  caches across calls,
- do not pass `session_cache=...` here; the canonical workflow surface is
  `cache=...`.

Output policy
- `output=:result` is the canonical default and returns an
  [`EncodingResult`](@ref),
- `output=:encoding_result` is also accepted when you want terminology aligned
  with ingestion stage names,
- `output=:raw` is explicit opt-in and returns `(P, H, pi)` for advanced code
  that truly needs the storage-shaped payload.

Cheap-first workflow
- start with `describe(enc)` or `result_summary(enc)`,
- inspect `dimensions(enc)` / `encoding_poset(enc)` / `encoding_map(enc)`,
- only then ask for `resolve`, `invariant`, `hom`, `ext`, or `tor`.

Notes
-----
* For multiple PL fringes, only the PLPolyhedra backend currently supports common-encoding.
  If you need common-encoding and you asked for PLBackend explicitly, we throw an error.

# Examples

```jldoctest
julia> using TamerOp

julia> const PLB = TamerOp.PLBackend;

julia> enc = TamerOp.encode([PLB.BoxUpset([0.0])], [PLB.BoxDownset([2.0])];
                                 backend=:pl_backend);

julia> TamerOp.describe(enc).kind
:encoding_result
```
"""
# -----------------------------------------------------------------------------
# Section 1: Workflow orchestration + public entrypoints
# -----------------------------------------------------------------------------

function encode(x; backend::Symbol=:auto, max_regions=nothing, strict_eps=nothing,
                poset_kind::Symbol=:signature, field::AbstractCoeffField=QQField(),
                output::Symbol=:result,
                cache=:auto)
    session_cache = _resolve_workflow_session_cache(cache)
    enc = EncodingOptions(; backend=backend, max_regions=max_regions,
                          strict_eps=strict_eps, poset_kind=poset_kind, field=field)
    return encode(x, enc; output=output, cache=session_cache)
end

# -----------------------
# Z^n presentations

function encode(FG::Flange{K}, enc::EncodingOptions;
                output::Symbol=:result,
                cache=:auto) where {K}
    session_cache = _resolve_workflow_session_cache(cache)
    if enc.backend != :auto && enc.backend != :zn
        error("encode(Flange): EncodingOptions.backend must be :auto or :zn (got $(enc.backend))")
    end
    FG2 = (FG.field == enc.field) ? FG : change_field(FG, enc.field)
    P, pi = ZnEncoding.encode_poset_from_flanges((FG2,), enc;
                                                 poset_kind=enc.poset_kind,
                                                 session_cache=session_cache)
    H, M = _workflow_zn_pushforward_module(P, pi, FG2, session_cache, enc.poset_kind)
    pi2 = _compile_encoding_cached(P, pi, session_cache)
    res = EncodingResult(P, M, pi2; H=H, presentation=FG2, opts=enc, backend=:zn, meta=(;))
    return _return_encoding(res, output)
end

function encode(FG::Flange{K};
                backend::Symbol=:auto,
                max_regions=nothing,
                strict_eps=nothing,
                poset_kind::Symbol=:signature,
                field::Union{AbstractCoeffField,Nothing}=nothing,
                output::Symbol=:result,
                cache=:auto) where {K}
    session_cache = _resolve_workflow_session_cache(cache)
    field === nothing && (field = FG.field)
    enc2 = EncodingOptions(; backend=backend, max_regions=max_regions,
                           strict_eps=strict_eps, poset_kind=poset_kind, field=field)
    return encode(FG, enc2; output=output, cache=session_cache)
end

function encode(FGs::Union{AbstractVector{<:Flange}, Tuple{Vararg{Flange}}},
                enc::EncodingOptions;
                output::Symbol=:result,
                cache=:auto)
    session_cache = _resolve_workflow_session_cache(cache)
    if enc.backend != :auto && enc.backend != :zn
        error("encode(Vector{Flange}): EncodingOptions.backend must be :auto or :zn (got $(enc.backend))")
    end
    K2 = coeff_type(enc.field)
    FGs2 = Vector{Flange{K2}}(undef, length(FGs))
    @inbounds for i in eachindex(FGs)
        FGi = FGs[i]
        FGs2[i] = (FGi.field == enc.field) ? FGi : change_field(FGi, enc.field)
    end
    P, pi = ZnEncoding.encode_poset_from_flanges(FGs2, enc;
                                                 poset_kind=enc.poset_kind,
                                                 session_cache=session_cache)
    Hs = Vector{FiniteFringe.FringeModule{K2}}(undef, length(FGs2))
    Ms = Vector{PModule{K2}}(undef, length(FGs2))
    @inbounds for i in eachindex(FGs2)
        Hs[i], Ms[i] = _workflow_zn_pushforward_module(P, pi, FGs2[i], session_cache, enc.poset_kind)
    end
    out = Vector{EncodingResult}(undef, length(FGs))
    pi2 = _compile_encoding_cached(P, pi, session_cache)
    for i in eachindex(FGs2)
        H = Hs[i]
        out[i] = EncodingResult(P, Ms[i], pi2;
                                H=H, presentation=FGs2[i], opts=enc, backend=:zn, meta=(;))
    end
    return output === :result ? out : [_return_encoding(enc_i, output) for enc_i in out]
end

function encode(FGs::Union{AbstractVector{<:Flange}, Tuple{Vararg{Flange}}};
                backend::Symbol=:auto,
                max_regions=nothing,
                strict_eps=nothing,
                poset_kind::Symbol=:signature,
                field::Union{AbstractCoeffField,Nothing}=nothing,
                output::Symbol=:result,
                cache=:auto)
    session_cache = _resolve_workflow_session_cache(cache)
    field === nothing && (field = FGs[1].field)
    enc2 = EncodingOptions(; backend=backend, max_regions=max_regions,
                           strict_eps=strict_eps, poset_kind=poset_kind, field=field)
    return encode(FGs, enc2; output=output, cache=session_cache)
end

# -----------------------
# R^n presentations

function encode(F::PLPolyhedra.PLFringe, enc::EncodingOptions=EncodingOptions();
                output::Symbol=:result,
                cache=:auto)
    session_cache = _resolve_workflow_session_cache(cache)
    P, H, pi = encode_from_fringe(F, enc)
    M = pmodule_from_fringe(H)
    b = choose_pl_backend(F; opts=enc)
    pi2 = _compile_encoding_cached(P, pi, session_cache)
    res = EncodingResult(P, M, pi2; H=H, presentation=F, opts=enc, backend=b, meta=(;))
    return _return_encoding(res, output)
end

function encode(Ups::Vector{BoxUpset}, Downs::Vector{BoxDownset}, Phi::AbstractMatrix,
                enc::EncodingOptions=EncodingOptions();
                output::Symbol=:result,
                cache=:auto)
    session_cache = _resolve_workflow_session_cache(cache)
    P, H, pi = encode_from_fringe(Ups, Downs, Phi, enc)
    M = pmodule_from_fringe(H)
    b = choose_pl_backend(Ups, Downs; opts=enc)
    pi2 = _compile_encoding_cached(P, pi, session_cache)
    res = EncodingResult(P, M, pi2;
                         H=H, presentation=(Ups=Ups, Downs=Downs, Phi=Phi),
                         opts=enc, backend=b, meta=(;))
    return _return_encoding(res, output)
end

# Convenience overloads for common BoxFringe encodings
encode(Ups::Vector{BoxUpset}, Downs::Vector{BoxDownset}, enc::EncodingOptions;
       output::Symbol=:result,
       cache=:auto) =
    encode(Ups, Downs,
           reshape(ones(QQ, length(Downs) * length(Ups)), length(Downs), length(Ups)),
           enc; output=output, cache=cache)

function encode(Ups::Vector{BoxUpset}, Downs::Vector{BoxDownset};
                backend::Symbol=:auto, max_regions=nothing, strict_eps=nothing,
                poset_kind::Symbol=:signature, field::AbstractCoeffField=QQField(),
                output::Symbol=:result,
                cache=:auto)
    enc = EncodingOptions(; backend=backend, max_regions=max_regions,
                          strict_eps=strict_eps, poset_kind=poset_kind, field=field)
    return encode(Ups, Downs, enc; output=output, cache=cache)
end

function encode(Ups::Vector{BoxUpset}, Downs::Vector{BoxDownset}, Phi_vec::AbstractVector,
                enc::EncodingOptions=EncodingOptions();
                output::Symbol=:result,
                cache=:auto)
    Phi = reshape(Phi_vec, length(Downs), length(Ups))
    return encode(Ups, Downs, Phi, enc; output=output, cache=cache)
end

function _encode_synthetic_box_fringe(B,
                                      enc::EncodingOptions=EncodingOptions();
                                      output::Symbol=:encoding_result,
                                      cache=:auto)
    session_cache = _resolve_workflow_session_cache(cache)
    Ups = FiniteFringe.birth_upsets(B)
    Downs = FiniteFringe.death_downsets(B)
    Phi = FlangeZn.coefficient_matrix(B)
    P, H, pi = encode_from_fringe(Ups, Downs, Phi, enc)
    M = pmodule_from_fringe(H)
    b = choose_pl_backend(Ups, Downs; opts=enc)
    pi2 = _compile_encoding_cached(P, pi, session_cache)
    res = EncodingResult(P, M, pi2;
                         H=H, presentation=B,
                         opts=enc, backend=b, meta=(;))
    return _return_encoding(res, output)
end

function encode(Fs::AbstractVector{<:PLPolyhedra.PLFringe}, enc::EncodingOptions;
                output::Symbol=:result,
                cache=:auto)
    session_cache = _resolve_workflow_session_cache(cache)
    # Today, only the PLPolyhedra backend supports common-encoding multiple PL fringes.
    if _normalize_pl_backend(enc.backend) == :pl_backend
        error("encode(Vector{PLFringe}): common encoding for PLBackend is not implemented; use backend=:pl or backend=:auto.")
    end
    if enc.backend != :auto && enc.backend != :pl
        error("encode(Vector{PLFringe}): EncodingOptions.backend must be :auto or :pl (got $(enc.backend))")
    end
    P, Hs, pi = PLPolyhedra.encode_from_PL_fringes(Fs, enc; poset_kind=enc.poset_kind)
    out = Vector{EncodingResult}(undef, length(Fs))
    pi2 = _compile_encoding_cached(P, pi, session_cache)
    for i in eachindex(Fs)
        H = Hs[i]
        out[i] = EncodingResult(P, pmodule_from_fringe(H), pi2;
                                H=H, presentation=Fs[i], opts=enc, backend=:pl, meta=(;))
    end
    return output === :result ? out : [_return_encoding(enc_i, output) for enc_i in out]
end

function encode(Fs::Tuple{Vararg{PLPolyhedra.PLFringe}}, enc::EncodingOptions;
                output::Symbol=:result,
                cache=:auto)
    return encode(PLPolyhedra.PLFringe[Fs...], enc; output=output, cache=cache)
end

encode(Fs::AbstractVector{<:PLPolyhedra.PLFringe};
       enc::EncodingOptions=EncodingOptions(),
       output::Symbol=:result,
       cache=:auto) =
    encode(Fs, enc;
           output=output,
           cache=cache)

function encode(Fs::Tuple{Vararg{PLPolyhedra.PLFringe}};
                enc::EncodingOptions=EncodingOptions(),
                output::Symbol=:result,
                cache=:auto)
    return encode(PLPolyhedra.PLFringe[Fs...];
                  enc=enc,
                  output=output,
                  cache=cache)
end

# -----------------------------------------------------------------------------
# Coarsening / compression of finite encodings

"""
    coarsen(enc::EncodingResult; method=:uptight, cache=:auto) -> EncodingResult

Coarsen/compress the finite encoding poset of an existing `EncodingResult`.

Currently supported:
- `method=:uptight`: build an uptight encoding from `enc.H`, push the fringe forward
  along the resulting finite encoding map, rebuild the pmodule, and postcompose the
  *ambient* classifier so `locate` still works on the new region poset.
  If `enc.H === nothing`, coarsen materializes a fringe from `enc.M` on demand.

This leaves `enc.backend` unchanged (the ambient encoding backend is still the same),
but replaces `(P, M, pi, H)` with their coarsened versions.

Cache contract
- `cache=:auto` uses a one-shot workflow cache,
- `cache=sc::SessionCache` reuses the postcomposition/compiled-encoding
  wrapper across repeated coarsening workflows.

Cheap-first workflow
- compare `describe(enc)` and `describe(coarsen(enc))`,
- inspect the resulting `EncodingResult` before unwrapping raw pieces.

```jldoctest
julia> using TamerOp

julia> const PLB = TamerOp.PLBackend;

julia> enc = TamerOp.encode([PLB.BoxUpset([0.0])], [PLB.BoxDownset([2.0])];
                                 backend=:pl_backend);

julia> TamerOp.describe(TamerOp.coarsen(enc)).kind
:encoding_result
```
"""
function coarsen(enc::EncodingResult;
                 method::Symbol = :uptight,
                 cache=:auto)
    session_cache = _resolve_workflow_session_cache(cache)
    method === :uptight || error("coarsen: unsupported method=$method. Currently only :uptight is supported.")
    M0 = materialize_module(enc.M)
    H = enc.H === nothing ? fringe_presentation(M0) : enc.H

    # 1) Compute coarsening map pi : (old P) -> (new P2)
    upt = build_uptight_encoding_from_fringe(H)
    pi = upt.pi

    # 2) Push fringe module forward along pi
    H2 = pushforward_fringe_along_encoding(H, pi)

    # 3) Rebuild pmodule on the coarsened region poset
    M2 = pmodule_from_fringe(H2)

    # 4) Postcompose old ambient classifier with the coarsening map
    pi2 = PostcomposedEncodingMap(enc.pi, pi)
    pi2c = _compile_encoding_cached(pi.P, pi2, session_cache)

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

    return EncodingResult(pi.P, M2, pi2c;
        H = H2,
        presentation = enc.presentation,
        opts = enc.opts,
        backend = enc.backend,
        meta = meta2,
    )
end

@inline function _workflow_postcomposed_classifier(enc::EncodingResult, pi, session_cache)
    raw_pi = enc.pi
    raw_pi === nothing && return nothing
    raw_pi isa AbstractPLikeEncodingMap || return nothing
    pi0 = raw_pi
    pi2 = PostcomposedEncodingMap(pi0, pi)
    return _compile_encoding_cached(pi.P, pi2, session_cache)
end

@inline function _workflow_translation_meta(enc::EncodingResult, kind::Symbol; kwargs...)
    extra = NamedTuple(kwargs)
    if enc.meta isa NamedTuple
        return merge(enc.meta, (translation_kind=kind,), extra)
    elseif enc.meta isa AbstractDict
        d = copy(enc.meta)
        d[:translation_kind] = kind
        for (k, v) in pairs(extra)
            d[k] = v
        end
        return d
    end
    return merge((translation_kind=kind, source_meta=enc.meta), extra)
end

@inline function _workflow_translation_result(kind::Symbol,
                                              M::PModule,
                                              pi,
                                              enc::EncodingResult;
                                              classifier=nothing,
                                              meta=NamedTuple())
    return ModuleTranslationResult(kind, M, pi;
                                   classifier=classifier,
                                   source=enc,
                                   meta=meta)
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
pmodule(enc::EncodingResult) = materialize_module(enc.M)

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
# Change-of-poset orchestration on workflow results
# -----------------------------------------------------------------------------

"""
    common_refinement(A::EncodingResult, B::EncodingResult; cache=:auto, kwargs...) -> CommonRefinementTranslationResult

Translate two encoded modules to a common refinement poset and return the typed
change-of-poset result from `ChangeOfPosets`.

This is the workflow entrypoint for "put these two encoded objects on one
finite poset so I can inspect the translated pair". The returned
`CommonRefinementTranslationResult` is already a semantic owner-level result
object, so the workflow layer forwards it directly instead of repackaging it.

Cheap-first workflow
- inspect `describe(out)` first,
- then use `common_poset(out)` and `translated_modules(out)`,
- only then feed the translated modules into heavier Hom or derived-functor
  calculations.

Cache contract
- `cache=:auto` is the canonical one-shot path,
- `cache=sc::SessionCache` reuses common-refinement translation plans across
  repeated calls.
"""
function common_refinement(A::EncodingResult, B::EncodingResult; cache=:auto, kwargs...)
    session_cache = _resolve_workflow_session_cache(cache)
    return ChangeOfPosets.encode_pmodules_to_common_poset(
        pmodule(A),
        pmodule(B);
        session_cache=session_cache,
        kwargs...,
    )
end

"""
    restriction(pi, enc::EncodingResult; cache=:auto) -> ModuleTranslationResult

Canonical workflow wrapper for change-of-poset restriction on an encoded module.

This delegates to `ChangeOfPosets.restriction(pi, pmodule(enc); ...)`, wraps the
translated module in a typed workflow result, and preserves provenance through
[`source_result`](@ref).

Because restriction along `pi : Q -> P` does not generally determine an ambient
classifier into `Q`, the returned [`ModuleTranslationResult`](@ref) typically
has no ambient encoding map.

Cache contract
- `cache=:auto` is the canonical one-shot path,
- `cache=sc::SessionCache` reuses translation plans across repeated calls.
"""
function restriction(pi, enc::EncodingResult; cache=:auto, kwargs...)
    session_cache = _resolve_workflow_session_cache(cache)
    M = ChangeOfPosets.restriction(pi, pmodule(enc); session_cache=session_cache, kwargs...)
    meta = _workflow_translation_meta(enc, :restriction)
    return _workflow_translation_result(:restriction, M, pi, enc; classifier=nothing, meta=meta)
end

"""
    pushforward_left(pi, enc::EncodingResult; cache=:auto) -> ModuleTranslationResult

Canonical workflow wrapper for left pushforward along a finite monotone map.

This delegates to `ChangeOfPosets.pushforward_left(pi, pmodule(enc); ...)`,
postcomposes the ambient classifier when available, and returns a typed
[`ModuleTranslationResult`](@ref) that keeps provenance from `enc`.
"""
function pushforward_left(pi, enc::EncodingResult; cache=:auto, kwargs...)
    session_cache = _resolve_workflow_session_cache(cache)
    M = ChangeOfPosets.pushforward_left(pi, pmodule(enc); session_cache=session_cache, kwargs...)
    cls = _workflow_postcomposed_classifier(enc, pi, session_cache)
    meta = _workflow_translation_meta(enc, :pushforward_left)
    return _workflow_translation_result(:pushforward_left, M, pi, enc; classifier=cls, meta=meta)
end

"""
    pushforward_right(pi, enc::EncodingResult; cache=:auto) -> ModuleTranslationResult

Canonical workflow wrapper for right pushforward along a finite monotone map.

This delegates to `ChangeOfPosets.pushforward_right(pi, pmodule(enc); ...)`,
postcomposes the ambient classifier when available, and returns a typed
[`ModuleTranslationResult`](@ref) with preserved provenance.
"""
function pushforward_right(pi, enc::EncodingResult; cache=:auto, kwargs...)
    session_cache = _resolve_workflow_session_cache(cache)
    M = ChangeOfPosets.pushforward_right(pi, pmodule(enc); session_cache=session_cache, kwargs...)
    cls = _workflow_postcomposed_classifier(enc, pi, session_cache)
    meta = _workflow_translation_meta(enc, :pushforward_right)
    return _workflow_translation_result(:pushforward_right, M, pi, enc; classifier=cls, meta=meta)
end

"""
    derived_pushforward_left(pi, enc::EncodingResult; opts=DerivedFunctorOptions(), cache=:auto) -> Vector{ModuleTranslationResult}

Canonical workflow wrapper for the left-derived pushforward of an encoded module.

The returned vector stores one [`ModuleTranslationResult`](@ref) per derived
degree, with degree `i` recorded in `result.meta.derived_degree`.
"""
function derived_pushforward_left(pi, enc::EncodingResult;
                                  opts::DerivedFunctorOptions=DerivedFunctorOptions(),
                                  cache=:auto,
                                  kwargs...)
    session_cache = _resolve_workflow_session_cache(cache)
    Ms = ChangeOfPosets.derived_pushforward_left(pi, pmodule(enc), opts; session_cache=session_cache, kwargs...)
    cls = _workflow_postcomposed_classifier(enc, pi, session_cache)
    out = Vector{ModuleTranslationResult}(undef, length(Ms))
    @inbounds for i in eachindex(Ms)
        meta = _workflow_translation_meta(enc, :derived_pushforward_left;
                                          derived_degree=i - 1,
                                          derived_opts=opts)
        out[i] = _workflow_translation_result(:derived_pushforward_left, Ms[i], pi, enc;
                                              classifier=cls, meta=meta)
    end
    return out
end

"""
    derived_pushforward_right(pi, enc::EncodingResult; opts=DerivedFunctorOptions(), cache=:auto) -> Vector{ModuleTranslationResult}

Canonical workflow wrapper for the right-derived pushforward of an encoded module.

The returned vector stores one [`ModuleTranslationResult`](@ref) per derived
degree, with degree `i` recorded in `result.meta.derived_degree`.
"""
function derived_pushforward_right(pi, enc::EncodingResult;
                                   opts::DerivedFunctorOptions=DerivedFunctorOptions(),
                                   cache=:auto,
                                   kwargs...)
    session_cache = _resolve_workflow_session_cache(cache)
    Ms = ChangeOfPosets.derived_pushforward_right(pi, pmodule(enc), opts; session_cache=session_cache, kwargs...)
    cls = _workflow_postcomposed_classifier(enc, pi, session_cache)
    out = Vector{ModuleTranslationResult}(undef, length(Ms))
    @inbounds for i in eachindex(Ms)
        meta = _workflow_translation_meta(enc, :derived_pushforward_right;
                                          derived_degree=i - 1,
                                          derived_opts=opts)
        out[i] = _workflow_translation_result(:derived_pushforward_right, Ms[i], pi, enc;
                                              classifier=cls, meta=meta)
    end
    return out
end

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
minimality_report(res::DerivedFunctors.Resolutions.ProjectiveResolution{K};
                  check_cover::Bool=true) where {K} =
    DerivedFunctors.minimality_report(res; check_cover=check_cover)
minimality_report(res::DerivedFunctors.Resolutions.InjectiveResolution{K};
                  check_hull::Bool=true) where {K} =
    DerivedFunctors.minimality_report(res; check_hull=check_hull)

"""
    is_minimal(res::ResolutionResult) -> Bool

Return whether the resolution stored in `res` was proven minimal by the chosen backend.
"""
is_minimal(res::ResolutionResult) = res.minimality !== nothing

is_minimal(res::DerivedFunctors.Resolutions.ProjectiveResolution{K};
           check_cover::Bool=true) where {K} =
    DerivedFunctors.is_minimal(res; check_cover=check_cover)
is_minimal(res::DerivedFunctors.Resolutions.InjectiveResolution{K};
           check_hull::Bool=true) where {K} =
    DerivedFunctors.is_minimal(res; check_hull=check_hull)

assert_minimal(res::DerivedFunctors.Resolutions.ProjectiveResolution{K};
               check_cover::Bool=true) where {K} =
    DerivedFunctors.assert_minimal(res; check_cover=check_cover)
assert_minimal(res::DerivedFunctors.Resolutions.InjectiveResolution{K};
               check_hull::Bool=true) where {K} =
    DerivedFunctors.assert_minimal(res; check_hull=check_hull)

"""
    betti_table(enc::EncodingResult; opts=ResolutionOptions(), cache=:auto, pad_to=nothing) -> Matrix{Int}
    betti_table(res::ResolutionResult; pad_to=nothing) -> Matrix{Int}

Cheap workflow wrapper returning a dense Betti table.

This is the canonical notebook-facing path when you want multiplicity data from
a projective resolution without inspecting resolution internals directly.

Workflow policy
- on `EncodingResult`, this builds or reuses a projective resolution through
  [`resolve`](@ref) and returns the bare dense matrix,
- on `ResolutionResult`, this formats the stored projective resolution directly,
- use [`resolve`](@ref) when you want the full typed wrapper and provenance.

Cache contract
- `cache=:auto` is the canonical one-shot path,
- `cache=sc::SessionCache` reuses the same workflow resolution cache across
  repeated Betti, Ext, and resolution calls.

```jldoctest
julia> using TamerOp

julia> const PLB = TamerOp.PLBackend;

julia> enc = TamerOp.encode([PLB.BoxUpset([0.0])], [PLB.BoxDownset([2.0])];
                                 backend=:pl_backend);

julia> TamerOp.betti_table(enc) isa Matrix{Int}
true
```
"""
betti_table(res::DerivedFunctors.Resolutions.ProjectiveResolution{K};
            pad_to::Union{Nothing,Int}=nothing) where {K} =
    DerivedFunctors.betti_table(res; pad_to=pad_to)

function betti_table(res::ResolutionResult; pad_to::Union{Nothing,Int}=nothing)
    resobj = resolution(res)
    resobj isa DerivedFunctors.Resolutions.ProjectiveResolution ||
        error("betti_table: expected a projective ResolutionResult; use bass_table for injective results.")
    return DerivedFunctors.betti_table(resobj; pad_to=pad_to)
end

function betti_table(enc::EncodingResult;
                     opts::ResolutionOptions=ResolutionOptions(),
                     cache=:auto,
                     pad_to::Union{Nothing,Int}=nothing)
    res = resolve(enc; kind=:projective, opts=opts, cache=cache)
    return betti_table(res; pad_to=pad_to)
end

"""
    bass_table(enc::EncodingResult; opts=ResolutionOptions(), cache=:auto, pad_to=nothing) -> Matrix{Int}
    bass_table(res::ResolutionResult; pad_to=nothing) -> Matrix{Int}

Cheap workflow wrapper returning a dense Bass table.

This is the canonical notebook-facing path when you want multiplicity data from
an injective resolution without inspecting resolution internals directly.

Workflow policy
- on `EncodingResult`, this builds or reuses an injective resolution through
  [`resolve`](@ref) and returns the bare dense matrix,
- on `ResolutionResult`, this formats the stored injective resolution directly,
- use [`resolve`](@ref) when you want the full typed wrapper and provenance.

Cache contract
- `cache=:auto` is the canonical one-shot path,
- `cache=sc::SessionCache` reuses the same workflow resolution cache across
  repeated Bass, Ext, and resolution calls.

```jldoctest
julia> using TamerOp

julia> const PLB = TamerOp.PLBackend;

julia> enc = TamerOp.encode([PLB.BoxUpset([0.0])], [PLB.BoxDownset([2.0])];
                                 backend=:pl_backend);

julia> TamerOp.bass_table(enc) isa Matrix{Int}
true
```
"""
bass_table(res::DerivedFunctors.Resolutions.InjectiveResolution{K};
           pad_to::Union{Nothing,Int}=nothing) where {K} =
    DerivedFunctors.bass_table(res; pad_to=pad_to)

function bass_table(res::ResolutionResult; pad_to::Union{Nothing,Int}=nothing)
    resobj = resolution(res)
    resobj isa DerivedFunctors.Resolutions.InjectiveResolution ||
        error("bass_table: expected an injective ResolutionResult; use betti_table for projective results.")
    return DerivedFunctors.bass_table(resobj; pad_to=pad_to)
end

function bass_table(enc::EncodingResult;
                    opts::ResolutionOptions=ResolutionOptions(),
                    cache=:auto,
                    pad_to::Union{Nothing,Int}=nothing)
    res = resolve(enc; kind=:injective, opts=opts, cache=cache)
    return bass_table(res; pad_to=pad_to)
end

@inline function _workflow_geometry_get(cache::Union{Nothing,EncodingCache}, key)
    cache === nothing && return nothing
    Base.lock(cache.lock)
    try
        entry = get(cache.geometry, key, nothing)
        return entry === nothing ? nothing : entry.value
    finally
        Base.unlock(cache.lock)
    end
end

@inline function _workflow_geometry_set!(cache::Union{Nothing,EncodingCache}, key, value)
    cache === nothing && return value
    Base.lock(cache.lock)
    try
        cache.geometry[key] = GeometryCachePayload(value)
    finally
        Base.unlock(cache.lock)
    end
    return value
end

@inline function _workflow_exact2d_box_key(box)
    box === nothing && return nothing
    return (Tuple(box[1]), Tuple(box[2]))
end

@inline function _workflow_exact2d_arrangement_key(pi,
                                                   opts::InvariantOptions;
                                                   normalize_dirs=:L1,
                                                   include_axes::Bool=false,
                                                   atol::Real=1e-12,
                                                   max_combinations::Int=200_000,
                                                   max_vertices::Int=20_000,
                                                   max_cells::Int=5_000_000,
                                                   precompute::Symbol=:cells_barcodes)
    arr_precompute = precompute in (:cells, :cells_barcodes) ? :cells : :none
    return (
        :workflow_exact2d_arrangement,
        UInt(objectid(pi)),
        _workflow_exact2d_box_key(opts.box),
        opts.strict,
        normalize_dirs,
        include_axes,
        Float64(atol),
        max_combinations,
        max_vertices,
        max_cells,
        arr_precompute,
    )
end

@inline function _workflow_exact2d_family_key(arr;
                                              weight::Symbol=:lesnick_l1,
                                              store_values::Bool=true)
    return (
        :workflow_exact2d_family,
        UInt(objectid(arr)),
        weight,
        store_values,
    )
end

function _workflow_exact2d_arrangement(pi,
                                       opts::InvariantOptions,
                                       cache::Union{Nothing,EncodingCache};
                                       normalize_dirs=:L1,
                                       include_axes::Bool=false,
                                       atol::Real=1e-12,
                                       max_combinations::Int=200_000,
                                       max_vertices::Int=20_000,
                                       max_cells::Int=5_000_000,
                                       precompute::Symbol=:cells_barcodes)
    key = _workflow_exact2d_arrangement_key(
        pi,
        opts;
        normalize_dirs=normalize_dirs,
        include_axes=include_axes,
        atol=atol,
        max_combinations=max_combinations,
        max_vertices=max_vertices,
        max_cells=max_cells,
        precompute=precompute,
    )
    cached = _workflow_geometry_get(cache, key)
    cached === nothing || return cached
    arr_precompute = precompute in (:cells, :cells_barcodes) ? :cells : :none
    arr = Fibered2D.fibered_arrangement_2d(
        pi,
        opts;
        normalize_dirs=normalize_dirs,
        include_axes=include_axes,
        atol=atol,
        max_combinations=max_combinations,
        max_vertices=max_vertices,
        max_cells=max_cells,
        precompute=arr_precompute,
    )
    return _workflow_geometry_set!(cache, key, arr)
end

function _workflow_exact2d_family(arr,
                                  cache::Union{Nothing,EncodingCache};
                                  weight::Symbol=:lesnick_l1,
                                  store_values::Bool=true)
    key = _workflow_exact2d_family_key(arr; weight=weight, store_values=store_values)
    cached = _workflow_geometry_get(cache, key)
    cached === nothing || return cached
    fam = Fibered2D.fibered_slice_family_2d(
        arr;
        direction_weight=weight,
        store_values=store_values,
    )
    return _workflow_geometry_set!(cache, key, fam)
end

@inline function _workflow_fringe_cache_key(enc::EncodingResult)
    return (
        :workflow_fringe_from_encoding,
        UInt(objectid(enc.P)),
        UInt(objectid(enc.M)),
    )
end

@inline function _workflow_fringe_for_hom_dimension(enc::EncodingResult,
                                                    cache::Union{Nothing,EncodingCache})
    H = enc.H
    H === nothing || return H

    key = _workflow_fringe_cache_key(enc)
    cached = _workflow_geometry_get(cache, key)
    cached === nothing || return cached

    Hnew = fringe_presentation(pmodule(enc))
    return _workflow_geometry_set!(cache, key, Hnew)
end

@inline function _workflow_hom_transport(transport::Symbol)
    transport in (:shared, :common_refinement) ||
        throw(ArgumentError("transport must be :shared or :common_refinement, got $(repr(transport))."))
    return transport
end


# -----------------------------------------------------------------------------
# Derived functors and invariants from workflow objects

"""
    hom_dimension(A::EncodingResult, B::EncodingResult; cache=:auto,
                  transport=:shared) -> Int

Return the dimension of `Hom(A,B)` using the cheapest workflow path.

This is a dimension-only workflow fast path:
- If `A.H` / `B.H` are present, they are used directly.
- Otherwise fringe presentations are materialized once and reused via workflow cache.

Contract
- with `transport=:shared` (the default), `A` and `B` must be encoded on the
  same finite poset object,
- with `transport=:common_refinement`, `A` and `B` may live on different
  finite posets and are compared through
  `ChangeOfPosets.hom_dim_common_refinement(...)`,
- use `encode(x, y; ...)` first when you want a shared encoding, or choose
  `transport=:common_refinement` when separate encodings should be translated
  automatically.

Cache contract
- `cache=:auto` is the canonical one-shot path,
- `cache=sc::SessionCache` reuses fringe materializations and workflow caches,
- with `transport=:common_refinement`, the same session cache is threaded into
  `ChangeOfPosets` translation plans.

Prefer this over [`hom`](@ref) when you only need the dimension.

```jldoctest
julia> using TamerOp

julia> const PLB = TamerOp.PLBackend;

julia> enc = TamerOp.encode([PLB.BoxUpset([0.0])], [PLB.BoxDownset([2.0])];
                                 backend=:pl_backend);

julia> TamerOp.hom_dimension(enc, enc) >= 1
true
```
"""
function hom_dimension(A::EncodingResult, B::EncodingResult;
                       cache=:auto,
                       transport::Symbol=:shared)
    transport = _workflow_hom_transport(transport)
    session_cache = _resolve_workflow_session_cache(cache)
    if transport === :common_refinement
        return ChangeOfPosets.hom_dim_common_refinement(pmodule(A), pmodule(B);
                                                        session_cache=session_cache)
    end
    (A.P === B.P) || error("hom_dimension: encodings are on different posets; use encode(x, y; ...) to common-encode first, or pass transport=:common_refinement.")
    enc_cache = _workflow_encoding_cache(session_cache)
    HA = _workflow_fringe_for_hom_dimension(A, enc_cache)
    HB = _workflow_fringe_for_hom_dimension(B, enc_cache)
    return FiniteFringe.hom_dimension(HA, HB)
end

"""
    hom(A::EncodingResult, B::EncodingResult; cache=:auto,
        transport=:shared)

Compute the full `Hom(A, B)` object for two encoded modules on the same finite
poset, or transport them through a common refinement when requested.

Use [`hom_dimension`](@ref) first when you only need the dimension. Choose
`hom(...)` when you explicitly want the heavier graded/basis-bearing Hom
object.

Transport policy
- `transport=:shared` is the canonical default and requires `A.P === B.P`,
- `transport=:common_refinement` delegates to
  `ChangeOfPosets.hom_common_refinement(...)` and may return a lazy
  common-refinement Hom object.

Cache contract
- `cache=:auto` creates a one-shot Hom-system cache,
- `cache=sc::SessionCache` reuses the same workflow cache across repeated Hom
  queries,
- with `transport=:common_refinement`, the session cache is threaded into
  change-of-poset translation planning instead of the Hom-system cache.
"""
function hom(A::EncodingResult, B::EncodingResult;
             cache=:auto,
             transport::Symbol=:shared)
    transport = _workflow_hom_transport(transport)
    if transport === :common_refinement
        session_cache = _resolve_workflow_session_cache(cache)
        return ChangeOfPosets.hom_common_refinement(pmodule(A), pmodule(B);
                                                    session_cache=session_cache)
    end
    (A.P === B.P) || error("hom: encodings are on different posets; use encode(x, y; ...) to common-encode first, or pass transport=:common_refinement.")
    MA = pmodule(A)
    MB = pmodule(B)
    K = coeff_type(MA.field)
    cache_hom, session_cache = _resolve_workflow_specialized_cache(cache, DerivedFunctors.HomSystemCache)
    cache2 = _hom_cache_from_session(cache_hom, session_cache, K)
    return DerivedFunctors.Hom(MA, MB; cache=cache2)
end

function hom(A::Modules.PModule{K}, B::Modules.PModule{K};
             cache=:auto) where {K}
    (A.Q === B.Q) || error("hom: posets mismatch.")
    cache_hom, session_cache = _resolve_workflow_specialized_cache(cache, DerivedFunctors.HomSystemCache)
    cache2 = _hom_cache_from_session(cache_hom, session_cache, K)
    return DerivedFunctors.Hom(A, B; cache=cache2)
end

"""
    matching_distance_exact_2d(encA::EncodingResult, encB::EncodingResult;
                               opts=InvariantOptions(), cache=:auto, kwargs...) -> Float64

Exact 2D workflow wrapper for the full arrangement-cell matching distance.

This is the canonical workflow entrypoint when both encoded modules share one
2D classifier map and you want deterministic exactness rather than the sampled
slice approximation from [`matching_distance`](@ref).

Contract
- `encA.P === encB.P` and `encA.pi === encB.pi`; common-encode first when they
  do not already share the same classifier map,
- the shared classifier map must be 2-dimensional,
- `cache=:auto` is the canonical one-shot path,
- `cache=sc::SessionCache` reuses the exact arrangement and slice family across
  repeated exact-2D calls with the same classifier/options.

Cheap-first workflow
- use [`matching_distance`](@ref) for quick approximate exploration,
- choose `matching_distance_exact_2d(...)` when you want exactness on the same
  shared 2D encoding,
- pass `cache=sc` when exact 2D queries will be repeated.
"""
matching_distance_exact_2d(M::PModule{K},
                           N::PModule{K},
                           pi,
                           opts::InvariantOptions;
                           kwargs...) where {K} =
    Fibered2D.matching_distance_exact_2d(M, N, pi, opts; kwargs...)

matching_distance_exact_2d(M::PModule{K},
                           N::PModule{K},
                           pi;
                           opts::InvariantOptions=InvariantOptions(),
                           kwargs...) where {K} =
    matching_distance_exact_2d(M, N, pi, opts; kwargs...)

matching_distance_exact_2d(cacheM::Fibered2D.FiberedBarcodeCache2D,
                           cacheN::Fibered2D.FiberedBarcodeCache2D;
                           kwargs...) =
    Fibered2D.matching_distance_exact_2d(cacheM, cacheN; kwargs...)

function matching_distance_exact_2d(encA::EncodingResult,
                                    encB::EncodingResult;
                                    opts::InvariantOptions=InvariantOptions(),
                                    cache=:auto,
                                    weight::Symbol=:lesnick_l1,
                                    normalize_dirs=:L1,
                                    include_axes::Bool=false,
                                    atol::Real=1e-12,
                                    max_combinations::Int=200_000,
                                    max_vertices::Int=20_000,
                                    max_cells::Int=5_000_000,
                                    precompute::Symbol=:cells_barcodes,
                                    arrangement=nothing,
                                    store_values::Bool=true,
                                    family=nothing)
    (encA.P === encB.P) ||
        error("matching_distance_exact_2d: encodings are on different posets; common-encode first.")
    (encA.pi === encB.pi) ||
        error("matching_distance_exact_2d: encodings do not share a common classifier map pi; common-encode first.")
    dimension(encA.pi) == 2 ||
        error("matching_distance_exact_2d: exact 2D distance requires a 2-dimensional classifier map.")

    session_cache = _resolve_workflow_session_cache(cache)
    encA2 = _encoding_with_session_cache(encA, session_cache)
    encB2 = _encoding_with_session_cache(encB, session_cache)
    pi = encA2.pi
    enc_cache = _workflow_encoding_cache(session_cache)

    arr = arrangement === nothing ? _workflow_exact2d_arrangement(
        pi,
        opts,
        enc_cache;
        normalize_dirs=normalize_dirs,
        include_axes=include_axes,
        atol=atol,
        max_combinations=max_combinations,
        max_vertices=max_vertices,
        max_cells=max_cells,
        precompute=precompute,
    ) : arrangement

    fam = family === nothing ? _workflow_exact2d_family(
        arr,
        enc_cache;
        weight=weight,
        store_values=store_values,
    ) : family

    return Fibered2D.matching_distance_exact_2d(
        pmodule(encA2),
        pmodule(encB2),
        pi,
        opts;
        weight=weight,
        normalize_dirs=normalize_dirs,
        include_axes=include_axes,
        atol=atol,
        max_combinations=max_combinations,
        max_vertices=max_vertices,
        max_cells=max_cells,
        precompute=precompute,
        arrangement=arr,
        store_values=store_values,
        family=fam,
    )
end

"""
    tor(Rop, L; maxdeg=3, model=:auto, cache=:auto)

Compute Tor_t(Rop, L), where `Rop` is a right-module represented as a module on
the opposite poset P^op, and `L` is a left-module on P.

For EncodingResult inputs, the underlying posets must be opposite:
    poset_equal_opposite(L.P, Rop.P)

Workflow policy
- prefer `cache=:auto` for one-shot calls,
- pass `cache=sc::SessionCache` to reuse resolution work across repeated Tor
  calls,
- keep `model=:auto` unless you have a reason to force a specific derived
  resolution model.
"""
function tor(Rop::EncodingResult, L::EncodingResult;
             maxdeg::Int=3, model::Symbol=:auto,
             cache=:auto)
    poset_equal_opposite(L.P, Rop.P) || error("tor: expected first argument on opposite poset of the second.")
    Rm = pmodule(Rop)
    Lm = pmodule(L)
    df = DerivedFunctorOptions(; maxdeg=maxdeg, model=model, canon=:none)
    model_eff = model == :auto ? :first : model
    cache_target = model_eff == :second ? Lm : Rm
    cache_res, session_cache = _resolve_workflow_specialized_cache(cache, ResolutionCache)
    cache2 = _resolution_cache_from_session(cache_res, session_cache, cache_target)
    return DerivedFunctors.Tor(Rm, Lm, df; cache=cache2)
end

function tor(Rop::Modules.PModule{K}, L::Modules.PModule{K};
             maxdeg::Int=3, model::Symbol=:auto,
             cache=:auto) where {K}
    poset_equal_opposite(L.Q, Rop.Q) || error("tor: expected first argument on opposite poset of the second.")
    df = DerivedFunctorOptions(; maxdeg=maxdeg, model=model, canon=:none)
    model_eff = model == :auto ? :first : model
    cache_target = model_eff == :second ? L : Rop
    cache_res, session_cache = _resolve_workflow_specialized_cache(cache, ResolutionCache)
    cache2 = _resolution_cache_from_session(cache_res, session_cache, cache_target)
    return DerivedFunctors.Tor(Rop, L, df; cache=cache2)
end

"""
    ext_algebra(enc; maxdeg=3, model=:auto, canon=:auto)

Compute the Ext-algebra `Ext^*(M, M)` with Yoneda product for the module stored
in `enc`.

This is the canonical workflow entrypoint when you want the algebra object
rather than a single Ext space.

Cheap-first workflow
- inspect `nonzero_degrees(ext_algebra(enc; maxdeg=...))` or
  `algebra_summary(...)` before materializing many element-level objects.
"""
function ext_algebra(enc::EncodingResult; maxdeg::Int=3, model::Symbol=:auto, canon::Symbol=:auto)
    df = DerivedFunctorOptions(; maxdeg=maxdeg, model=model, canon=canon)
    return DerivedFunctors.ExtAlgebra(pmodule(enc), df)
end

"""
    ext(A::EncodingResult, B::EncodingResult; maxdeg=3, model=:auto, canon=:auto, cache=:auto)

Compute Ext^t(A, B) using the finite-poset modules stored in EncodingResult.

If `A` and `B` are not encoded on the same poset object, you must common-encode first:
    encs = encode(x, y; backend=...)
    E = ext(encs[1], encs[2])

Workflow policy
- prefer `cache=:auto` for one-shot calls,
- pass `cache=sc::SessionCache` to reuse resolution work across repeated Ext
  calls,
- keep `model=:auto` unless you intentionally want the injective or projective
  route.
"""
function ext(A::EncodingResult, B::EncodingResult;
             maxdeg::Int=3, model::Symbol=:auto, canon::Symbol=:auto,
             cache=:auto)
    (A.P === B.P) || error("ext: encodings are on different posets; use encode(x, y; ...) to common-encode first.")
    MA = pmodule(A)
    MB = pmodule(B)
    df = DerivedFunctorOptions(; maxdeg=maxdeg, model=model, canon=canon)
    model_eff = model == :auto ? :projective : model
    cache_target = model_eff == :injective ? MB :
                   model_eff == :projective ? MA : nothing
    cache_res, session_cache = _resolve_workflow_specialized_cache(cache, ResolutionCache)
    cache2 = _resolution_cache_from_session(cache_res, session_cache, cache_target)
    return DerivedFunctors.Ext(MA, MB, df; cache=cache2)
end

function ext(A::Modules.PModule{K}, B::Modules.PModule{K};
             maxdeg::Int=3, model::Symbol=:auto, canon::Symbol=:auto,
             cache=:auto) where {K}
    (A.Q === B.Q) || error("ext: modules live on different posets.")
    df = DerivedFunctorOptions(; maxdeg=maxdeg, model=model, canon=canon)
    model_eff = model == :auto ? :projective : model
    cache_target = model_eff == :injective ? B :
                   model_eff == :projective ? A : nothing
    cache_res, session_cache = _resolve_workflow_specialized_cache(cache, ResolutionCache)
    cache2 = _resolution_cache_from_session(cache_res, session_cache, cache_target)
    return DerivedFunctors.Ext(A, B, df; cache=cache2)
end

"""
    resolve(enc::EncodingResult; kind=:projective, opts=ResolutionOptions(),
            minimality=false, cache=:auto)

Compute a projective or injective resolution starting from an EncodingResult.
Returns a ResolutionResult that stores the resolution plus provenance.

- kind=:projective (default) computes a ProjectiveResolution and stores its Betti table.
- kind=:injective computes an InjectiveResolution and stores its Bass table in `meta`.

Minimality checks can be expensive; enable with `minimality=true`.

If `cache` is `:auto` (default), a temporary session cache is used for the call.
Pass a shared `SessionCache()` via `cache=...` to reuse resolutions across calls.

Cheap-first workflow
- call `describe(res)` or `result_summary(res)` first,
- inspect `resolution_object(res)` and any stored Betti/Bass summary before
  moving on to `ext`, `tor`, or derived maps.

```jldoctest
julia> using TamerOp

julia> const PLB = TamerOp.PLBackend;

julia> enc = TamerOp.encode([PLB.BoxUpset([0.0])], [PLB.BoxDownset([2.0])];
                                 backend=:pl_backend);

julia> res = TamerOp.resolve(enc);

julia> TamerOp.describe(res).kind
:resolution_result
```
"""
function resolve(enc::EncodingResult;
                 kind::Symbol=:projective,
                 opts::ResolutionOptions=ResolutionOptions(),
                 minimality::Bool=false,
                 check_hull::Bool=true,
                 cache=:auto)
    opts = opts
    M = pmodule(enc)
    cache_res, session_cache = _resolve_workflow_specialized_cache(cache, ResolutionCache)
    cache2 = _resolution_cache_from_session(cache_res, session_cache, M)
    kind in (:projective, :injective) ||
        error("resolve: kind must be :projective or :injective (got $(kind))")

    if kind == :projective
        res = DerivedFunctors.projective_resolution(M, opts; cache=cache2)
        b = DerivedFunctors.betti(res)
        mrep = minimality ? DerivedFunctors.minimality_report(res) : nothing
        return ResolutionResult(res; enc=enc, betti=b, minimality=mrep, opts=opts, meta=(kind=:projective,))
    else
        res = DerivedFunctors.injective_resolution(M, opts; cache=cache2)
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
    rhom(enc::EncodedComplexResult, N; kwargs...)

Compute `RHom(C, N)` where `C` is a module cochain complex and `N` is a module.

Source/target convention
- `C` is the first argument in the derived Hom bifunctor,
- `N` is the coefficient module in the second argument.

Workflow policy
- `cache=:auto` is the canonical one-shot path,
- `cache=sc::SessionCache` reuses the Hom-system cache across repeated RHom and
  HyperExt queries,
- prefer `hom_dimension(...)`, `hom(...)`, or `ext(...)` first when you only
  need a cheaper summary or a single Ext space rather than the full RHom
  complex.

Return contract
- returns the algebraic RHom object directly, not a workflow result wrapper.
"""
function rhom(C::ModuleComplexes.ModuleCochainComplex{K}, N::Modules.PModule{K};
              cache=:auto,
              kwargs...) where {K}
    cache_hom, session_cache = _resolve_workflow_specialized_cache(cache, DerivedFunctors.HomSystemCache)
    cache2 = _hom_cache_from_session(cache_hom, session_cache, K)
    return ModuleComplexes.RHom(C, N; cache=cache2, kwargs...)
end

rhom(enc::EncodedComplexResult, N::Modules.PModule{K}; cache=:auto, kwargs...) where {K} =
    rhom(_materialize_complex(encoding_complex(enc)), N; cache=cache, kwargs...)

"""
    derived_tensor(Rop, C; kwargs...)
    derived_tensor(Rop, enc::EncodedComplexResult; kwargs...)

Compute derived tensor Rop \\otimes^L C, where Rop is on P^op and C is on P.
"""
derived_tensor(Rop::Modules.PModule{K}, C::ModuleComplexes.ModuleCochainComplex{K}; kwargs...) where {K} =
    ModuleComplexes.DerivedTensor(Rop, C; kwargs...)

derived_tensor(Rop::EncodingResult, C::ModuleComplexes.ModuleCochainComplex; kwargs...) =
    derived_tensor(pmodule(Rop), C; kwargs...)

derived_tensor(Rop::Modules.PModule{K}, enc::EncodedComplexResult; kwargs...) where {K} =
    derived_tensor(Rop, _materialize_complex(encoding_complex(enc)); kwargs...)

derived_tensor(Rop::EncodingResult, enc::EncodedComplexResult; kwargs...) =
    derived_tensor(pmodule(Rop), _materialize_complex(encoding_complex(enc)); kwargs...)


"""
    hyperext(C, N; maxdeg=3, kwargs...)
    hyperext(enc::EncodedComplexResult, N; maxdeg=3, kwargs...)

Compute `HyperExt^t(C, N)` for a module cochain complex `C` and a module `N`.

Source/target convention
- `C` is the first derived argument,
- `N` is the coefficient module in the second argument.

Workflow policy
- `cache=:auto` is the canonical one-shot path,
- `cache=sc::SessionCache` reuses the same Hom-system cache as `rhom(...)`,
- keep `maxdeg` small when you only need low degrees.

Cheap-first workflow
- prefer `ext(...)` when you want ordinary Ext for encoded modules,
- use `hyperext(...)` when the first argument is already a cochain complex.

Return contract
- returns the algebraic HyperExt object directly, not an `InvariantResult` or
  `ResolutionResult`.
"""
function hyperext(C::ModuleComplexes.ModuleCochainComplex{K}, N::Modules.PModule{K};
                  maxdeg::Int=3,
                  cache=:auto,
                  kwargs...) where {K}
    cache_hom, session_cache = _resolve_workflow_specialized_cache(cache, DerivedFunctors.HomSystemCache)
    cache2 = _hom_cache_from_session(cache_hom, session_cache, K)
    return ModuleComplexes.hyperExt(C, N; maxlen=maxdeg, cache=cache2, kwargs...)
end

hyperext(enc::EncodedComplexResult, N::Modules.PModule{K};
         maxdeg::Int=3,
         cache=:auto,
         kwargs...) where {K} =
    hyperext(_materialize_complex(encoding_complex(enc)), N; maxdeg=maxdeg, cache=cache, kwargs...)

"""
    hypertor(Rop, C; maxdeg=3, kwargs...)
    hypertor(Rop, enc::EncodedComplexResult; maxdeg=3, kwargs...)

Compute `HyperTor_t(Rop, C)` for a right-module `Rop` on `P^op` and a module
complex `C` on `P`.

Source/target convention
- `Rop` is the right-module argument on the opposite poset,
- `C` is the left-module complex argument on `P`.

Workflow policy
- `maxdeg` truncates the computed degree range,
- keep the default model unless you are deliberately matching a derived-functor
  backend experiment,
- prefer `tor(...)` first when the second argument is an ordinary module rather
  than a module complex.

Return contract
- returns the algebraic HyperTor object directly.
"""
function hypertor(Rop::Modules.PModule{K}, C::ModuleComplexes.ModuleCochainComplex{K};
                  maxdeg::Int=3, kwargs...) where {K}
    return ModuleComplexes.hyperTor(Rop, C; maxlen=maxdeg, kwargs...)
end

hypertor(Rop::EncodingResult, C::ModuleComplexes.ModuleCochainComplex; kwargs...) =
    hypertor(pmodule(Rop), C; kwargs...)

hypertor(Rop::Modules.PModule{K}, enc::EncodedComplexResult; maxdeg::Int=3, kwargs...) where {K} =
    hypertor(Rop, _materialize_complex(encoding_complex(enc)); maxdeg=maxdeg, kwargs...)

hypertor(Rop::EncodingResult, enc::EncodedComplexResult; maxdeg::Int=3, kwargs...) =
    hypertor(pmodule(Rop), _materialize_complex(encoding_complex(enc)); maxdeg=maxdeg, kwargs...)



@inline _is_invariant_call_method_miss(err, f) =
    err isa MethodError && (err.f === f || err.f === Core.kwcall)

@inline function _try_invariant_call(f, args...; kwargs...)
    try
        return true, f(args...; kwargs...)
    catch err
        _is_invariant_call_method_miss(err, f) || rethrow()
        return false, nothing
    end
end

function _call_invariant(f, enc::EncodingResult, opts::InvariantOptions; kwargs...)
    dims = module_dims(enc.M)
    ok, v = _try_invariant_call(f, dims, enc.pi; opts=opts, kwargs...)
    ok && return v
    ok, v = _try_invariant_call(f, dims, enc.pi; kwargs...)
    ok && return v
    ok, v = _try_invariant_call(f, dims, opts; kwargs...)
    ok && return v
    ok, v = _try_invariant_call(f, dims; kwargs...)
    ok && return v

    M = pmodule(enc)
    ok, v = _try_invariant_call(f, M, enc.pi; opts=opts, kwargs...)
    ok && return v
    ok, v = _try_invariant_call(f, M, enc.pi; kwargs...)
    ok && return v
    ok, v = _try_invariant_call(f, M, opts; kwargs...)
    ok && return v
    ok, v = _try_invariant_call(f, M; kwargs...)
    ok && return v
    ok, v = _try_invariant_call(f, enc, opts; kwargs...)
    ok && return v
    ok, v = _try_invariant_call(f, enc; kwargs...)
    ok && return v
    error("invariant: no method found for invariant function $(f). Expected f(M), f(M,pi), f(M,opts), or f(M,pi; opts=...).")
end

# Internal helper for dims-only cohomology outputs.
function _call_invariant(f, enc::CohomologyDimsResult, opts::InvariantOptions; kwargs...)
    ok, v = _try_invariant_call(f, enc.dims, enc.pi; opts=opts, kwargs...)
    ok && return v
    ok, v = _try_invariant_call(f, enc.dims, enc.pi; kwargs...)
    ok && return v
    ok, v = _try_invariant_call(f, enc.dims, opts; kwargs...)
    ok && return v
    ok, v = _try_invariant_call(f, enc.dims; kwargs...)
    ok && return v
    ok, v = _try_invariant_call(f, enc, opts; kwargs...)
    ok && return v
    ok, v = _try_invariant_call(f, enc; kwargs...)
    ok && return v
    error("invariant: no method found for invariant function $(f). Expected dims-compatible signatures.")
end

function _call_invariant(f, enc::EncodedComplexResult, opts::InvariantOptions; kwargs...)
    C = encoding_complex(enc)
    ok, v = _try_invariant_call(f, C, enc.pi; opts=opts, kwargs...)
    ok && return v
    ok, v = _try_invariant_call(f, C, enc.pi; kwargs...)
    ok && return v
    ok, v = _try_invariant_call(f, C, opts; kwargs...)
    ok && return v
    ok, v = _try_invariant_call(f, C; kwargs...)
    ok && return v
    Cmat = _materialize_complex(C)
    ok, v = _try_invariant_call(f, Cmat, enc.pi; opts=opts, kwargs...)
    ok && return v
    ok, v = _try_invariant_call(f, Cmat, enc.pi; kwargs...)
    ok && return v
    ok, v = _try_invariant_call(f, Cmat, opts; kwargs...)
    ok && return v
    ok, v = _try_invariant_call(f, Cmat; kwargs...)
    ok && return v
    ok, v = _try_invariant_call(f, enc, opts; kwargs...)
    ok && return v
    ok, v = _try_invariant_call(f, enc; kwargs...)
    ok && return v
    error("invariant: no method found for invariant function $(f). Expected whole-complex-compatible signatures.")
end

@inline function _resolve_invariant_function(which::Symbol)
    if isdefined(Invariants, which)
        return getfield(Invariants, which)
    elseif isdefined(SignedMeasures, which)
        return getfield(SignedMeasures, which)
    end
    error("invariant: unknown invariant symbol $(which). Expected a function in Invariants or SignedMeasures.")
end


"""
    invariant(enc::EncodingResult; which=:rank_invariant, opts=InvariantOptions(), kwargs...) -> InvariantResult
    invariant(enc::EncodedComplexResult; which=:euler_signed_measure, opts=InvariantOptions(), kwargs...) -> InvariantResult

Compute one invariant from the invariant-family owners and wrap it in an
[`InvariantResult`](@ref).

`which` may be:
- a Symbol naming a function in `Invariants` (e.g. `:rank_invariant`)
- a callable itself

Cache contract
- `cache=:auto` is the canonical one-shot path,
- `cache=sc::SessionCache` reuses the compiled encoding/result wrapper across
  repeated invariant calls.

Cheap-first workflow
- start with one invariant via `invariant(...)`,
- inspect `describe(inv)` / `result_summary(inv)`,
- only then batch with [`invariants`](@ref) if needed.

```jldoctest
julia> using TamerOp

julia> const PLB = TamerOp.PLBackend;

julia> enc = TamerOp.encode([PLB.BoxUpset([0.0])], [PLB.BoxDownset([2.0])];
                                 backend=:pl_backend);

julia> inv = TamerOp.invariant(enc; which=:restricted_hilbert);

julia> TamerOp.describe(inv).kind
:invariant_result
```
"""
function invariant(enc::EncodingResult;
                   which=:rank_invariant,
                   opts::InvariantOptions=InvariantOptions(),
                   cache=:auto,
                   kwargs...)
    opts = opts
    session_cache = _resolve_workflow_session_cache(cache)
    enc2 = _encoding_with_session_cache(enc, session_cache)
    f = which isa Symbol ? _resolve_invariant_function(which) : which
    val = _call_invariant(f, enc2, opts; kwargs...)
    return InvariantResult(enc2, which, val; opts=opts, meta=NamedTuple())
end

"""
    invariant(enc::CohomologyDimsResult; which=:restricted_hilbert, opts=InvariantOptions(), kwargs...) -> InvariantResult

Compute a single dims-compatible invariant and wrap it in an `InvariantResult`.

`which` may be:
- a Symbol naming a function in `Invariants` (e.g. `:restricted_hilbert`)
- a callable itself

This is the canonical cheap-first workflow surface once you already have a
[`CohomologyDimsResult`](@ref).
"""
function invariant(enc::CohomologyDimsResult;
                   which=:restricted_hilbert,
                   opts::InvariantOptions=InvariantOptions(),
                   cache=:auto,
                   kwargs...)
    opts = opts
    session_cache = _resolve_workflow_session_cache(cache)
    enc2 = _encoding_with_session_cache(enc, session_cache)
    f = which isa Symbol ? _resolve_invariant_function(which) : which
    val = _call_invariant(f, enc2, opts; kwargs...)
    return InvariantResult(enc2, which, val; opts=opts, meta=NamedTuple())
end

"""
    invariant(enc::EncodedComplexResult; which=:euler_signed_measure,
              opts=InvariantOptions(), kwargs...) -> InvariantResult

Compute one whole-complex invariant from an encoded cochain-complex workflow
result and wrap it in an [`InvariantResult`](@ref).

This is the canonical workflow surface for exact invariants whose mathematical
source is the full encoded cochain complex rather than one chosen cohomology
module.
"""
function invariant(enc::EncodedComplexResult;
                   which=:euler_signed_measure,
                   opts::InvariantOptions=InvariantOptions(),
                   cache=:auto,
                   kwargs...)
    opts = opts
    session_cache = _resolve_workflow_session_cache(cache)
    enc2 = _encoding_with_session_cache(enc, session_cache)
    f = which isa Symbol ? _resolve_invariant_function(which) : which
    val = _call_invariant(f, enc2, opts; kwargs...)
    return InvariantResult(enc2, which, val; opts=opts, meta=NamedTuple())
end


"""
    invariants(enc::EncodingResult; which=..., opts=InvariantOptions(), kwargs...) -> Vector{InvariantResult}

Batch convenience wrapper around `invariant`.

`which` may be:
- a Symbol or callable (compute one)
- a vector of Symbols / callables (compute many)

Prefer [`invariant`](@ref) first when you only need one invariant; use this
wrapper when batching is the actual goal.
"""
function invariants(enc::EncodingResult;
                    which=[:rank_invariant],
                    opts::InvariantOptions=InvariantOptions(),
                    cache=:auto,
                    kwargs...)
    opts = opts
    if which isa AbstractVector
        return [invariant(enc; which=w, opts=opts, cache=cache, kwargs...) for w in which]
    else
        return [invariant(enc; which=which, opts=opts, cache=cache, kwargs...)]
    end
end

function invariants(enc::CohomologyDimsResult;
                    which=[:restricted_hilbert],
                    opts::InvariantOptions=InvariantOptions(),
                    cache=:auto,
                    kwargs...)
    opts = opts
    if which isa AbstractVector
        return [invariant(enc; which=w, opts=opts, cache=cache, kwargs...) for w in which]
    else
        return [invariant(enc; which=which, opts=opts, cache=cache, kwargs...)]
    end
end

function invariants(enc::EncodedComplexResult;
                    which=[:euler_signed_measure],
                    opts::InvariantOptions=InvariantOptions(),
                    cache=:auto,
                    kwargs...)
    opts = opts
    if which isa AbstractVector
        return [invariant(enc; which=w, opts=opts, cache=cache, kwargs...) for w in which]
    else
        return [invariant(enc; which=which, opts=opts, cache=cache, kwargs...)]
    end
end


# -----------------------------------------------------------------------------
# Curated invariant entrypoints (stable value-returning wrappers)
# -----------------------------------------------------------------------------

@doc raw"""
    rank_invariant(M::PModule; opts=InvariantOptions(), kwargs...)
    rank_invariant(enc::EncodingResult; opts=InvariantOptions(), kwargs...)

Workflow convenience wrapper returning the bare rank-invariant value.

- On `PModule`, this forwards to the owner-level invariant kernel directly.
- On `EncodingResult`, this routes through `invariant(enc; which=:rank_invariant)`
  and then unwraps `.value`.

Prefer `invariant(...)` when you want provenance, `describe(...)`, or
`result_summary(...)` on the result wrapper.
""" rank_invariant
rank_invariant(M::PModule{K}, opts::InvariantOptions=InvariantOptions(); kwargs...) where {K} =
    Invariants.rank_invariant(M, opts; kwargs...)

rank_map(M::PModule{K}, pi, x, y, opts::InvariantOptions=InvariantOptions(); kwargs...) where {K} =
    Invariants.rank_map(M, pi, x, y, opts; kwargs...)
rank_map(M::PModule{K}, a::Int, b::Int; kwargs...) where {K} = Invariants.rank_map(M, a, b; kwargs...)

restricted_hilbert(M::PModule{K}) where {K} = Invariants.restricted_hilbert(M)

@doc raw"""
    restricted_hilbert(M::PModule; opts=InvariantOptions(), kwargs...)
    restricted_hilbert(enc::EncodingResult; opts=InvariantOptions(), kwargs...)
    restricted_hilbert(enc::CohomologyDimsResult; opts=InvariantOptions(), kwargs...)

Workflow convenience wrapper returning the bare restricted-Hilbert value.

- `EncodingResult` and `CohomologyDimsResult` inputs use the workflow cache
  contract and then unwrap the value from `InvariantResult`.
- Use `invariant(...; which=:restricted_hilbert)` when you want a typed wrapper
  with provenance instead of just the value.
""" restricted_hilbert
restricted_hilbert(M::PModule{K}, pi, x, opts::InvariantOptions=InvariantOptions(); kwargs...) where {K} =
    Invariants.restricted_hilbert(M, pi, x, opts; kwargs...)

@doc raw"""
    point_signed_measure(enc::EncodingResult; opts=InvariantOptions(), cache=:auto, kwargs...)
    point_signed_measure(enc::EncodedComplexResult; opts=InvariantOptions(), cache=:auto, kwargs...)

Workflow convenience wrapper returning a point-signed measure from an encoded
workflow object.

For `EncodingResult` inputs this means: compute the Euler surface on the chosen
grid and return its Mobius inversion as a `PointSignedMeasure`. This is the
same mathematical object returned by [`euler_signed_measure`](@ref); prefer the
more specific name when you want to emphasize the Euler interpretation.
""" point_signed_measure
function point_signed_measure(enc::EncodingResult;
                              opts::InvariantOptions=InvariantOptions(),
                              cache=:auto,
                              kwargs...)
    return euler_signed_measure(enc; opts=opts, cache=cache, kwargs...)
end

function point_signed_measure(enc::EncodedComplexResult;
                              opts::InvariantOptions=InvariantOptions(),
                              cache=:auto,
                              kwargs...)
    return euler_signed_measure(enc; opts=opts, cache=cache, kwargs...)
end

@doc raw"""
    euler_signed_measure(enc::EncodingResult; opts=InvariantOptions(), cache=:auto, kwargs...)
    euler_signed_measure(enc::EncodedComplexResult; opts=InvariantOptions(), cache=:auto, kwargs...)

Workflow convenience wrapper returning the bare Euler signed measure.

- `cache=:auto` is the canonical one-shot path,
- `cache=sc::SessionCache` reuses the same workflow/session cache across
  repeated Euler and signed-measure calls.

Prefer `invariant(...; which=:euler_signed_measure)` when you want provenance
packaged in an `InvariantResult` instead of the bare point measure.
""" euler_signed_measure
function euler_signed_measure(enc::EncodingResult;
                              opts::InvariantOptions=InvariantOptions(),
                              cache=:auto,
                              kwargs...)
    session_cache = _resolve_workflow_session_cache(cache)
    enc2 = _encoding_with_session_cache(enc, session_cache)
    return SignedMeasures.euler_signed_measure(pmodule(enc2), enc2.pi, opts;
                                               cache=session_cache,
                                               kwargs...)
end

function euler_signed_measure(enc::EncodedComplexResult;
                              opts::InvariantOptions=InvariantOptions(),
                              cache=:auto,
                              kwargs...)
    session_cache = _resolve_workflow_session_cache(cache)
    enc2 = _encoding_with_session_cache(enc, session_cache)
    return SignedMeasures.euler_signed_measure(encoding_complex(enc2), enc2.pi, opts;
                                               cache=session_cache,
                                               kwargs...)
end

@doc raw"""
    rectangle_signed_barcode(enc::EncodingResult; opts=InvariantOptions(), cache=:auto, kwargs...)

Workflow convenience wrapper returning the bare rectangle signed barcode.

This supports Zn-style finite encodings and finite axis-grid encodings such as
`GridEncodingMap`. For grid encodings with noninteger semantic coordinates, the
returned barcode lives on the selected encoding-index grid; use
`encoding_axes(enc)` to map rectangle endpoints back to semantic coordinates.
""" rectangle_signed_barcode
function rectangle_signed_barcode(enc::EncodingResult;
                                  opts::InvariantOptions=InvariantOptions(),
                                  cache=:auto,
                                  kwargs...)
    session_cache = _resolve_workflow_session_cache(cache)
    enc2 = _encoding_with_session_cache(enc, session_cache)
    return SignedMeasures.rectangle_signed_barcode(pmodule(enc2), enc2.pi, opts;
                                                   cache=session_cache,
                                                   kwargs...)
end

@doc raw"""
    rectangle_signed_barcode_image(enc::EncodingResult;
                                   opts=InvariantOptions(),
                                   cache=:auto,
                                   rect_kwargs=NamedTuple(),
                                   kwargs...)

Workflow convenience wrapper returning a lightweight image of the rectangle
signed barcode of `enc`.

- barcode-construction keywords belong in `rect_kwargs`,
- image keywords such as `xs`, `ys`, `sigma`, `mode`, and `cutoff_tol` are
  passed directly as `kwargs...`,
- `cache` only affects the barcode-construction phase.
""" rectangle_signed_barcode_image
function rectangle_signed_barcode_image(enc::EncodingResult;
                                        opts::InvariantOptions=InvariantOptions(),
                                        cache=:auto,
                                        rect_kwargs=NamedTuple(),
                                        kwargs...)
    rect_kwargs isa NamedTuple ||
        throw(ArgumentError("rectangle_signed_barcode_image: rect_kwargs must be a NamedTuple."))
    sb = rectangle_signed_barcode(enc; opts=opts, cache=cache, rect_kwargs...)
    return SignedMeasures.rectangle_signed_barcode_image(sb; kwargs...)
end

@doc raw"""
    euler_surface(M, pi; opts=InvariantOptions(), kwargs...)
    euler_surface(enc::EncodingResult; opts=InvariantOptions(), kwargs...)
    euler_surface(enc::EncodedComplexResult; opts=InvariantOptions(), kwargs...)
    euler_surface(enc::CohomologyDimsResult; opts=InvariantOptions(), kwargs...)

Workflow convenience wrapper returning the bare Euler-surface value.

- For encoded workflow objects, this is the cheap owner-local shorthand for
  `invariant(...; which=:euler_surface).value`.
- Prefer the typed `InvariantResult` route when you want provenance or notebook
  inspection before using the surface data.
""" euler_surface
function euler_surface(M::PModule{K}, pi, opts::InvariantOptions=InvariantOptions(); kwargs...) where {K}
    opts = opts
    if any(haskey(kwargs, k) for k in (:axes, :axes_policy, :max_axis_len, :box, :threads, :strict))
        opts = InvariantOptions(
            axes = get(kwargs, :axes, opts.axes),
            axes_policy = get(kwargs, :axes_policy, opts.axes_policy),
            max_axis_len = get(kwargs, :max_axis_len, opts.max_axis_len),
            box = get(kwargs, :box, opts.box),
            threads = get(kwargs, :threads, opts.threads),
            strict = get(kwargs, :strict, opts.strict),
            pl_mode = get(kwargs, :pl_mode, opts.pl_mode),
        )
        kw_nt = NamedTuple(kwargs)
        kwargs = Base.structdiff(kw_nt, (; axes=nothing, axes_policy=nothing, max_axis_len=nothing, box=nothing, threads=nothing, strict=nothing, pl_mode=nothing))
    end
    return SignedMeasures.euler_surface(M, pi, opts; kwargs...)
end

euler_surface(C::ModuleCochainComplex{K}, pi, opts::InvariantOptions=InvariantOptions(); kwargs...) where {K} =
    SignedMeasures.euler_surface(C, pi, opts; kwargs...)

@doc raw"""
    slice_barcode(M::PModule, chain; opts=InvariantOptions(), kwargs...)
    slice_barcode(enc::EncodingResult; opts=InvariantOptions(), kwargs...)

Workflow convenience wrapper returning one slice-barcode value directly.

- On `EncodingResult`, this unwraps `invariant(...; which=:slice_barcode).value`.
- Use `slice_barcodes(...)` when you want a family of slices or cached plan
  reuse across repeated calls.
""" slice_barcode
slice_barcode(M::PModule{K}, chain::AbstractVector{Int}, opts::InvariantOptions=InvariantOptions(); kwargs...) where {K} =
    Invariants.slice_barcode(M, chain; kwargs...)

slice_barcodes(M::PModule{K}, chains::AbstractVector, opts::InvariantOptions=InvariantOptions(); kwargs...) where {K} =
    Invariants.slice_barcodes(M, chains; kwargs...)

function slice_barcodes(M::PModule{K}, plan::CompiledSlicePlan, opts::InvariantOptions=InvariantOptions(); kwargs...) where {K}
    opts0 = opts
    kwargs_nt = NamedTuple(kwargs)
    for k in keys(kwargs_nt)
        (k == :threads || k == :packed) && continue
        throw(ArgumentError("slice_barcodes(plan): unsupported keyword :$k"))
    end
    return Invariants.slice_barcodes(M, plan;
        packed = get(kwargs_nt, :packed, false),
        threads = get(kwargs_nt, :threads, opts0.threads))
end

function slice_barcodes(M::PModule{K}, pi, opts::InvariantOptions=InvariantOptions();
                        cache=:auto,
                        kwargs...) where {K}
    opts0 = opts
    kwargs_nt = NamedTuple(kwargs)
    haskey(kwargs_nt, :box) &&
        throw(ArgumentError("slice_barcodes: pass box via opts::InvariantOptions, not keyword :box"))
    haskey(kwargs_nt, :strict) &&
        throw(ArgumentError("slice_barcodes: pass strict via opts::InvariantOptions, not keyword :strict"))
    kwargs2 = (; (k => v for (k, v) in pairs(kwargs_nt) if k != :threads)...)
    cache_slice, session_cache = _resolve_workflow_specialized_cache(cache, SlicePlanCache)
    cache2 = _slice_plan_cache_from_session(cache_slice, session_cache)
    return Invariants.slice_barcodes(M, pi;
        opts = opts0,
        threads = get(kwargs_nt, :threads, opts0.threads),
        cache = cache2,
        kwargs2...)
end

rank_invariant(enc::EncodingResult; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    invariant(enc; which=:rank_invariant, opts=opts, kwargs...).value

restricted_hilbert(enc::EncodingResult; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    invariant(enc; which=:restricted_hilbert, opts=opts, kwargs...).value

restricted_hilbert(enc::CohomologyDimsResult; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    invariant(enc; which=:restricted_hilbert, opts=opts, kwargs...).value

euler_surface(enc::EncodingResult; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    invariant(enc; which=:euler_surface, opts=opts, kwargs...).value

euler_surface(enc::EncodedComplexResult; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    invariant(enc; which=:euler_surface, opts=opts, kwargs...).value

euler_surface(enc::CohomologyDimsResult; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    invariant(enc; which=:euler_surface, opts=opts, kwargs...).value

slice_barcode(enc::EncodingResult; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    invariant(enc; which=:slice_barcode, opts=opts, kwargs...).value

"""
    slice_barcodes(enc::EncodingResult; opts=InvariantOptions(), cache=:auto, kwargs...)

Compute slice barcodes from an [`EncodingResult`](@ref) using the workflow
cache contract.

Cache contract
- `cache=:auto` uses a one-shot slice-plan cache,
- `cache=sc::SessionCache` reuses compiled slice plans across calls.

Query/options contract
- pass finite-box and strictness choices via `opts::InvariantOptions`,
- keep ad hoc thread overrides in `kwargs...`,
- do not pass `session_cache=...` here.
"""
function slice_barcodes(enc::EncodingResult;
                        opts::InvariantOptions=InvariantOptions(),
                        cache=:auto,
                        kwargs...)
    session_cache = _resolve_workflow_session_cache(cache)
    return slice_barcodes(pmodule(enc), enc.pi, opts;
                          cache=session_cache,
                          kwargs...)
end

"""
    mp_landscape(enc::EncodingResult; opts=InvariantOptions(), cache=:auto, kwargs...)

Compute the multiparameter landscape of an [`EncodingResult`](@ref).

Cache contract
- `cache=:auto` uses a one-shot slice-plan cache,
- `cache=sc::SessionCache` reuses compiled slice plans across calls.

Cheap-first workflow
- inspect `describe(enc)` first,
- then call `mp_landscape(...)` only when you actually want the heavier
  image/landscape object rather than a cheaper invariant summary.
"""
function mp_landscape(enc::EncodingResult;
                      opts::InvariantOptions=InvariantOptions(),
                      cache=:auto,
                      kwargs...)
    session_cache = _resolve_workflow_session_cache(cache)
    return mp_landscape(pmodule(enc), enc.pi;
                        opts=opts,
                        cache=session_cache,
                        kwargs...)
end
mp_landscape(M::PModule{K}, slices::AbstractVector; kwargs...) where {K} =
    Invariants.mp_landscape(M, slices; kwargs...)
function mp_landscape(M::PModule{K}, pi;
                      opts::InvariantOptions=InvariantOptions(),
                      cache=:auto,
                      kwargs...) where {K}
    opt_keys = (:box, :strict, :threads, :axes, :axes_policy, :max_axis_len, :pl_mode)
    cache_slice, session_cache = _resolve_workflow_specialized_cache(cache, SlicePlanCache)
    cache2 = _slice_plan_cache_from_session(cache_slice, session_cache)
    opts0 = opts
    if any(k -> haskey(kwargs, k), opt_keys)
        base = opts0
        opts = InvariantOptions(
            box = get(kwargs, :box, base.box),
            strict = get(kwargs, :strict, base.strict),
            threads = get(kwargs, :threads, base.threads),
            axes = get(kwargs, :axes, base.axes),
            axes_policy = get(kwargs, :axes_policy, base.axes_policy),
            max_axis_len = get(kwargs, :max_axis_len, base.max_axis_len),
            pl_mode = get(kwargs, :pl_mode, base.pl_mode),
        )
        kwargs_nt = NamedTuple(kwargs)
        kwargs2 = (; (k => v for (k, v) in pairs(kwargs_nt) if !(k in opt_keys))...)
        return Invariants.mp_landscape(M, pi, opts; cache=cache2, kwargs2...)
    end
    return Invariants.mp_landscape(M, pi, opts0; cache=cache2, kwargs...)
end

@doc raw"""
    mpp_decomposition(enc::EncodingResult; opts=InvariantOptions(), kwargs...)
    mpp_decomposition(M, pi; opts=InvariantOptions(), kwargs...)

Workflow convenience wrapper returning the bare multiparameter decomposition.

This is the lightest owner-local way to obtain the decomposition data from an
encoded workflow object. Use `invariant(...; which=:mpp_decomposition)` when you
want a typed `InvariantResult` wrapper instead of the raw value.
""" mpp_decomposition
mpp_decomposition(enc::EncodingResult; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    invariant(enc; which=:mpp_decomposition, opts=opts, kwargs...).value
function mpp_decomposition(M::PModule{K}, pi;
                           opts::InvariantOptions=InvariantOptions(),
                           kwargs...) where {K}
    return Invariants.mpp_decomposition(M, pi, opts; kwargs...)
end

@doc raw"""
    mpp_image(enc::EncodingResult; opts=InvariantOptions(), kwargs...)
    mpp_image(M, pi; opts=InvariantOptions(), kwargs...)

Workflow convenience wrapper returning the bare multiparameter image.

This is the value-returning shorthand for
`invariant(...; which=:mpp_image).value`. Prefer the typed wrapper when you want
provenance, validation, or `describe(...)` before materializing the image.
""" mpp_image
mpp_image(enc::EncodingResult; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    invariant(enc; which=:mpp_image, opts=opts, kwargs...).value
function mpp_image(M::PModule{K}, pi;
                   opts::InvariantOptions=InvariantOptions(),
                   kwargs...) where {K}
    return Invariants.mpp_image(M, pi, opts; kwargs...)
end

end # module Workflow
