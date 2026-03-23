module ModuleComplexes
using ..DerivedFunctors

function _resolution_offsets(res)
    return res.offsets
end

function _resolution_offsets(res::DerivedFunctors.Resolutions.InjectiveResolution)
    return _gens_offsets(res.gens)
end

function _gens_offsets(gens)
    offsets = Vector{Vector{Int}}(undef, length(gens))
    for i in eachindex(gens)
        gi = gens[i]
        if isempty(gi)
            offsets[i] = Int[]
            continue
        end
        if eltype(gi) <: AbstractVector
            lens = length.(gi)
            offs = Vector{Int}(undef, length(lens) + 1)
            offs[1] = 0
            for j in 1:length(lens)
                offs[j + 1] = offs[j] + lens[j]
            end
            offsets[i] = offs
        else
            offsets[i] = collect(0:length(gi))
        end
    end
    return offsets
end

using LinearAlgebra
using SparseArrays
import Base.Threads

using ..CoreModules: _append_scaled_triplets!
using ..Options: ResolutionOptions
using ..FieldLinAlg
using ..FiniteFringe
using ..FiniteFringe: AbstractPoset, FinitePoset, cover_edges, nvertices, upset_indices
using ..Modules: PModule, PMorphism, id_morphism,
                 zero_pmodule, zero_morphism,
                 direct_sum, direct_sum_with_maps,
                 map_leq, map_leq_many

import ..IndicatorResolutions
import ..AbelianCategories
import ..ChainComplexes
using ..AbelianCategories: kernel_with_inclusion, image_with_inclusion, _cokernel_module


import ..ChainComplexes:
    shift, extend_range, mapping_cone, mapping_cone_triangle,
    induced_map_on_cohomology, _map_at, describe,
    source, target, component, differential
using ..ChainComplexes:
    CochainComplex, DoubleComplex, CochainMap,
    total_complex, cohomology_data, spectral_sequence,
    cohomology_coordinates, cohomology_representative,
    solve_particular

using ..DerivedFunctors:
    HomSpace, Hom,
    HomSystemCache, hom_with_cache, precompose_matrix_cached, postcompose_matrix_cached,
    ProjectiveResolution, InjectiveResolution,
    injective_resolution, projective_resolution,
    lift_injective_chainmap,
    compose
import ..DerivedFunctors:
    source_module, target_module, nonzero_degrees, degree_dimensions, total_dimension

# Internal Functoriality helpers live in DerivedFunctors.Functoriality.
using ..DerivedFunctors.Functoriality:
    _tensor_map_on_tor_chains_from_projective_coeff,
    _lift_pmodule_map_to_projective_resolution_chainmap_coeff

import ..AbelianCategories: connecting_map

# Extend the DerivedFunctors.GradedSpaces interface for the "hyper" derived objects
# computed in this file (HyperExtSpace and HyperTorSpace).
import ..DerivedFunctors.GradedSpaces:
    degree_range, dim, basis, representative, coordinates, cycles, boundaries


import Base.Threads

const IR = IndicatorResolutions
const FF = FiniteFringe

const _FRINGE_PMODULE_CACHE_LOCK = ReentrantLock()
const _FRINGE_PMODULE_CACHES = WeakKeyDict{Any, IdDict{Any, Any}}()
const _INJECTIVE_RESOLUTION_CACHE_LOCK = ReentrantLock()
const _INJECTIVE_RESOLUTION_CACHES = WeakKeyDict{Any, IdDict{Any, Dict{Int, Any}}}()
const _RHOM_COMPLEX_CACHE_LOCK = ReentrantLock()
const _RHOM_COMPLEX_CACHES = WeakKeyDict{Any, Dict{Tuple{UInt, UInt, UInt, Int, Bool}, Any}}()
const _HYPEREXT_SPACE_CACHE_LOCK = ReentrantLock()
const _HYPEREXT_SPACE_CACHES = WeakKeyDict{Any, IdDict{Any, Any}}()
const _INJECTIVE_LIFT_CACHE_LOCK = ReentrantLock()
const _INJECTIVE_LIFT_CACHES = WeakKeyDict{Any, Dict{NTuple{3, UInt}, Any}}()
const _REBASED_PMORPHISM_CACHE_LOCK = ReentrantLock()
const _REBASED_PMORPHISM_CACHES = WeakKeyDict{Any, Dict{NTuple{3, UInt}, Any}}()
const _PROJECTIVE_LIFT_CACHE_LOCK = ReentrantLock()
const _PROJECTIVE_LIFT_CACHES = WeakKeyDict{Any, Dict{NTuple{4, UInt}, Any}}()
const _RHOM_MAP_FIRST_PLAN_LOCK = ReentrantLock()
const _RHOM_MAP_FIRST_PLAN_CACHES = WeakKeyDict{Any, Dict{Tuple{UInt, UInt}, Any}}()
const _RHOM_MAP_SECOND_PLAN_LOCK = ReentrantLock()
const _RHOM_MAP_SECOND_PLAN_CACHES = WeakKeyDict{Any, Dict{Tuple{UInt, UInt}, Any}}()
const _DTENSOR_MAP_FIRST_PLAN_LOCK = ReentrantLock()
const _DTENSOR_MAP_FIRST_PLAN_CACHES = WeakKeyDict{Any, Dict{Tuple{UInt, UInt}, Any}}()
const _DTENSOR_MAP_FIRST_RESULT_LOCK = ReentrantLock()
const _DTENSOR_MAP_FIRST_RESULT_CACHES = WeakKeyDict{Any, Dict{NTuple{3, UInt}, Any}}()
const _DTENSOR_MAP_SECOND_PLAN_LOCK = ReentrantLock()
const _DTENSOR_MAP_SECOND_PLAN_CACHES = WeakKeyDict{Any, Dict{Tuple{UInt, UInt}, Any}}()
const _DTENSOR_MAP_SECOND_RESULT_LOCK = ReentrantLock()
const _DTENSOR_MAP_SECOND_RESULT_CACHES = WeakKeyDict{Any, Dict{NTuple{3, UInt}, Any}}()
const _DTENSOR_MAP_SECOND_CACHE_MIN_WORK = Ref(64)

"""
    source_map(H::ModuleCochainHomotopy)

Return the first cochain map in a module-valued cochain homotopy.

Use this accessor instead of reaching into raw homotopy fields when you want to
inspect or validate the source side of `H : f => g`.
"""
function source_map end

"""
    target_map(H::ModuleCochainHomotopy)

Return the second cochain map in a module-valued cochain homotopy.

This is the semantic companion to [`source_map`](@ref) for cheap-first
inspection of homotopy data.
"""
function target_map end

"""
    triangle_objects(T::ModuleDistinguishedTriangle)

Return the three module complexes appearing in a distinguished triangle as the
named tuple `(; source, target, cone)`.

Prefer this accessor over raw field inspection when exploring a triangle at the
REPL or in notebooks.
"""
function triangle_objects end

"""
    triangle_maps(T::ModuleDistinguishedTriangle)

Return the structure maps of a distinguished triangle as the named tuple
`(; morphism, inclusion, projection)`.

This is the cheap inspection surface for triangle data before looking at the
individual component maps degree-by-degree.
"""
function triangle_maps end

"""
    underlying_complex(X)

Return the ordinary cochain complex underlying a module-derived construction.

Current methods are provided for [`RHomComplex`](@ref) and
[`DerivedTensorComplex`](@ref). Use this when you need the chain-level object
that supports cohomology, spectral-sequence, or map-level inspection.
"""
function underlying_complex end

struct ModuleComplexValidationSummary{R}
    report::R
end

"""
    module_complex_validation_summary(report) -> ModuleComplexValidationSummary

Wrap a raw validation report returned by `check_module_*` in a compact
display-oriented object.
"""
@inline module_complex_validation_summary(report::NamedTuple) = ModuleComplexValidationSummary(report)

@inline function _module_complex_report(
    kind::Symbol,
    valid::Bool;
    issues::AbstractVector{<:AbstractString}=String[],
    kwargs...,
)
    return (; kind, valid, issues=Tuple(String.(issues)), kwargs...)
end

@inline function _throw_invalid_module_complex(fname::Symbol, issues::AbstractVector{<:AbstractString})
    msg = isempty(issues) ? "invalid object" : " - " * join(issues, "\n - ")
    Base.throw(ArgumentError(string(fname, ": validation failed\n", msg)))
end

function Base.show(io::IO, summary::ModuleComplexValidationSummary)
    r = summary.report
    print(io, "ModuleComplexValidationSummary(kind=", r.kind,
          ", valid=", r.valid,
          ", issues=", length(r.issues), ")")
end

function Base.show(io::IO, ::MIME"text/plain", summary::ModuleComplexValidationSummary)
    r = summary.report
    println(io, "ModuleComplexValidationSummary")
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

struct _RHomMapFirstDegreePlan{H}
    row_off::Vector{Int}
    col_off::Vector{Int}
    pdeg::Vector{Int}
    Hdom::Vector{H}
    Hcod::Vector{H}
end

@inline function _tensor_map_second_plan_work(Tsrc, Ttgt)
    return (length(Tsrc.C.terms) + length(Ttgt.C.terms)) * sum(length, Tsrc.resR.gens)
end

struct _RHomMapSecondDegreePlan{H}
    row_off::Vector{Int}
    col_off::Vector{Int}
    ideg::Vector{Int}
    Hsrc::Vector{H}
    Htgt::Vector{H}
end

struct _RHomMapPlan{P}
    tmin::Int
    tmax::Int
    dims_src::Vector{Int}
    dims_tgt::Vector{Int}
    degrees::Vector{P}
end

struct _TensorMapFirstDegreePlan{K}
    row_off::Vector{Int}
    col_off::Vector{Int}
    adeg::Vector{Int}
    terms::Vector{PModule{K}}
    dom_gens::Vector{Vector{Int}}
    cod_gens::Vector{Vector{Int}}
    dom_offsets::Vector{Vector{Int}}
    cod_offsets::Vector{Vector{Int}}
end

struct _TensorMapSecondDegreePlan{K}
    block_pdeg::Vector{Int}
    block_u::Vector{Int}
    block_row0::Vector{Int}
    block_col0::Vector{Int}
    col_owner::Vector{Int}
    col_local::Vector{Int}
end

struct _TensorMapPlan{P}
    tmin::Int
    tmax::Int
    dims_src::Vector{Int}
    dims_tgt::Vector{Int}
    degrees::Vector{P}
end

@inline function _tensor_map_second_col_nnz(block::AbstractMatrix, j::Int)
    if issparse(block)
        return length(nzrange(block, j))
    end
    c = 0
    @inbounds for i in axes(block, 1)
        iszero(block[i, j]) || (c += 1)
    end
    return c
end

@inline function _fill_tensor_map_second_col!(
    rowval::Vector{Int},
    nzval::Vector{K},
    pos::Int,
    block::AbstractMatrix{K},
    j::Int,
    row0::Int,
) where {K}
    if issparse(block)
        @inbounds for ptr in nzrange(block, j)
            rowval[pos] = row0 + rowvals(block)[ptr]
            nzval[pos] = nonzeros(block)[ptr]
            pos += 1
        end
        return pos
    end
    @inbounds for i in axes(block, 1)
        v = block[i, j]
        iszero(v) && continue
        rowval[pos] = row0 + i
        nzval[pos] = v
        pos += 1
    end
    return pos
end

function _assemble_tensor_map_second_degree(
    g,
    dplan::_TensorMapSecondDegreePlan{K},
    dim_tgt::Int,
    dim_src::Int,
) where {K}
    blocks = Vector{AbstractMatrix{K}}(undef, length(dplan.block_u))
    isempty(blocks) && return spzeros(K, dim_tgt, dim_src)
    @inbounds for b in eachindex(blocks)
        blocks[b] = _map(g, dplan.block_pdeg[b]).comps[dplan.block_u[b]]
    end
    colptr = Vector{Int}(undef, dim_src + 1)
    colptr[1] = 1
    nnz_total = 0
    @inbounds for j in 1:dim_src
        owner = dplan.col_owner[j]
        if owner != 0
            nnz_total += _tensor_map_second_col_nnz(blocks[owner], dplan.col_local[j])
        end
        colptr[j + 1] = nnz_total + 1
    end

    rowval = Vector{Int}(undef, nnz_total)
    nzval = Vector{K}(undef, nnz_total)
    pos = 1
    @inbounds for j in 1:dim_src
        owner = dplan.col_owner[j]
        owner == 0 && continue
        pos = _fill_tensor_map_second_col!(
            rowval,
            nzval,
            pos,
            blocks[owner],
            dplan.col_local[j],
            dplan.block_row0[owner],
        )
    end
    return SparseMatrixCSC{K,Int}(dim_tgt, dim_src, colptr, rowval, nzval)
end


@inline function _pair_plan_cached!(builder::Function,
                                    store::WeakKeyDict{Any,Dict{Tuple{UInt, UInt},Any}},
                                    lock::ReentrantLock,
                                    owner,
                                    src,
                                    tgt)
    Base.lock(lock) do
        shard = get!(store, owner) do
            Dict{Tuple{UInt, UInt}, Any}()
        end
        key = (UInt(objectid(src)), UInt(objectid(tgt)))
        cached = get(shard, key, nothing)
        cached === nothing || return cached
        plan = builder()
        shard[key] = plan
        return plan
    end
end

@inline function _pmodule_from_fringe_cached(H::FF.FringeModule{K}, ::Nothing) where {K}
    return IR.pmodule_from_fringe(H)
end

function _pmodule_from_fringe_cached(H::FF.FringeModule{K}, cache::HomSystemCache) where {K}
    lock(_FRINGE_PMODULE_CACHE_LOCK) do
        shard = get!(_FRINGE_PMODULE_CACHES, cache) do
            IdDict{Any, Any}()
        end
        cached = get(shard, H, nothing)
        cached === nothing || return cached::PModule{K}
        M = IR.pmodule_from_fringe(H)
        shard[H] = M
        return M
    end
end

@inline function _injective_resolution_cached(
    N::PModule{K},
    maxlen::Int,
    ::Nothing,
) where {K}
    return injective_resolution(N, ResolutionOptions(maxlen=maxlen))
end

function _injective_resolution_cached(
    N::PModule{K},
    maxlen::Int,
    cache::HomSystemCache,
) where {K}
    lock(_INJECTIVE_RESOLUTION_CACHE_LOCK) do
        shard = get!(_INJECTIVE_RESOLUTION_CACHES, cache) do
            IdDict{Any, Dict{Int, Any}}()
        end
        per_module = get!(shard, N) do
            Dict{Int, Any}()
        end
        cached = get(per_module, maxlen, nothing)
        cached === nothing || return cached::InjectiveResolution{K}
        res = injective_resolution(N, ResolutionOptions(maxlen=maxlen))
        per_module[maxlen] = res
        return res
    end
end

@inline _rhom_complex_cache_key(C, N, resN, maxlen::Int, threads::Bool) =
    (UInt(objectid(C)), UInt(objectid(N)), UInt(objectid(resN)), maxlen, threads)

function _rhom_complex_cached(
    C,
    N::PModule{K};
    maxlen::Int = 3,
    resN = nothing,
    cache::Union{Nothing, HomSystemCache} = nothing,
    threads::Bool = (Threads.nthreads() > 1),
) where {K}
    resN_use = resN === nothing ? _injective_resolution_cached(N, maxlen, cache) : resN
    cache === nothing && return RHomComplex(C, N; maxlen=maxlen, resN=resN_use, cache=cache, threads=threads)
    key = _rhom_complex_cache_key(C, N, resN_use, maxlen, threads)
    lock(_RHOM_COMPLEX_CACHE_LOCK) do
        shard = get!(_RHOM_COMPLEX_CACHES, cache) do
            Dict{Tuple{UInt, UInt, UInt, Int, Bool}, Any}()
        end
        cached = get(shard, key, nothing)
        cached === nothing || return cached::RHomComplex{K}
        R = RHomComplex(C, N; maxlen=maxlen, resN=resN_use, cache=cache, threads=threads)
        shard[key] = R
        return R
    end
end

function _rhom_complex_cached(
    C,
    H::FF.FringeModule{K};
    maxlen::Int = 3,
    resN = nothing,
    cache::Union{Nothing, HomSystemCache} = nothing,
    threads::Bool = (Threads.nthreads() > 1),
) where {K}
    N = _pmodule_from_fringe_cached(H, cache)
    return _rhom_complex_cached(C, N; maxlen=maxlen, resN=resN, cache=cache, threads=threads)
end

@inline function _hyperext_space_cached(R, ::Nothing)
    return HyperExtSpace(R, cohomology_data(R.tot))
end

function _hyperext_space_cached(R, cache::HomSystemCache)
    lock(_HYPEREXT_SPACE_CACHE_LOCK) do
        shard = get!(_HYPEREXT_SPACE_CACHES, cache) do
            IdDict{Any, Any}()
        end
        cached = get(shard, R, nothing)
        cached === nothing || return cached
        H = HyperExtSpace(R, cohomology_data(R.tot))
        shard[R] = H
        return H
    end
end

@inline function _hyperext_cached(
    C,
    N::PModule{K};
    maxlen::Int = 3,
    resN = nothing,
    cache::Union{Nothing, HomSystemCache} = nothing,
    threads::Bool = (Threads.nthreads() > 1),
) where {K}
    R = _rhom_complex_cached(C, N; maxlen=maxlen, resN=resN, cache=cache, threads=threads)
    return _hyperext_space_cached(R, cache)
end

@inline function _hyperext_cached(
    C,
    H::FF.FringeModule{K};
    maxlen::Int = 3,
    resN = nothing,
    cache::Union{Nothing, HomSystemCache} = nothing,
    threads::Bool = (Threads.nthreads() > 1),
) where {K}
    R = _rhom_complex_cached(C, H; maxlen=maxlen, resN=resN, cache=cache, threads=threads)
    return _hyperext_space_cached(R, cache)
end

@inline _injective_lift_cache_key(g, res_dom, res_cod) =
    (UInt(objectid(g)), UInt(objectid(res_dom)), UInt(objectid(res_cod)))

@inline function _lift_injective_chainmap_cached(
    g::PMorphism{K},
    res_dom::InjectiveResolution{K},
    res_cod::InjectiveResolution{K};
    check::Bool = true,
    cache::Union{Nothing,HomSystemCache}=nothing,
) where {K}
    cache === nothing && return lift_injective_chainmap(g, res_dom, res_cod; check=check)
    key = _injective_lift_cache_key(g, res_dom, res_cod)
    lock(_INJECTIVE_LIFT_CACHE_LOCK) do
        shard = get!(_INJECTIVE_LIFT_CACHES, cache) do
            Dict{NTuple{3, UInt}, Any}()
        end
        cached = get(shard, key, nothing)
        cached === nothing || return cached::Vector{PMorphism{K}}
        phis = lift_injective_chainmap(g, res_dom, res_cod; check=check)
        shard[key] = phis
        return phis
    end
end

@inline _rebased_pmodule_morphism_cache_key(g, dom, cod) =
    (UInt(objectid(g)), UInt(objectid(dom)), UInt(objectid(cod)))

@inline function _rebase_pmodule_morphism_cached(
    g::PMorphism{K},
    dom::PModule{K},
    cod::PModule{K},
    ::Nothing,
) where {K}
    return (g.dom === dom && g.cod === cod) ? g : PMorphism{K}(dom, cod, g.comps)
end

function _rebase_pmodule_morphism_cached(
    g::PMorphism{K},
    dom::PModule{K},
    cod::PModule{K},
    cache::HomSystemCache,
) where {K}
    g.dom === dom && g.cod === cod && return g
    key = _rebased_pmodule_morphism_cache_key(g, dom, cod)
    lock(_REBASED_PMORPHISM_CACHE_LOCK) do
        shard = get!(_REBASED_PMORPHISM_CACHES, cache) do
            Dict{NTuple{3, UInt}, Any}()
        end
        cached = get(shard, key, nothing)
        cached === nothing || return cached::PMorphism{K}
        rebased = PMorphism{K}(dom, cod, g.comps)
        shard[key] = rebased
        return rebased
    end
end

@inline _projective_lift_cache_key(f, res_dom, res_cod, upto::Int) =
    (UInt(objectid(f)), UInt(objectid(res_dom)), UInt(objectid(res_cod)), UInt(upto))

@inline function _lift_projective_chainmap_coeff_uncached(
    f::PMorphism{K},
    res_dom::ProjectiveResolution{K},
    res_cod::ProjectiveResolution{K};
    upto::Int,
) where {K}
    return _lift_pmodule_map_to_projective_resolution_chainmap_coeff(res_dom, res_cod, f; upto=upto)
end

function _lift_projective_chainmap_coeff_cached(
    f::PMorphism{K},
    res_dom::ProjectiveResolution{K},
    res_cod::ProjectiveResolution{K};
    upto::Int,
    cache::HomSystemCache,
) where {K}
    key = _projective_lift_cache_key(f, res_dom, res_cod, upto)
    lock(_PROJECTIVE_LIFT_CACHE_LOCK) do
        shard = get!(_PROJECTIVE_LIFT_CACHES, cache) do
            Dict{NTuple{4, UInt}, Any}()
        end
        cached = get(shard, key, nothing)
        cached === nothing || return cached::Vector{SparseMatrixCSC{K, Int}}
        coeffs = _lift_projective_chainmap_coeff_uncached(f, res_dom, res_cod; upto=upto)
        shard[key] = coeffs
        return coeffs
    end
end


"""
    DirectSumModuleMap{K}

Internal helper type used by certain lifting routines.

Some internal algorithms represent morphisms between direct sums of principal
downset modules by storing the induced linear maps on selected vertex bases.
This struct packages those vertexwise matrices in a concrete, documented way.

Fields
------
  - `comps::Vector{Matrix{K}}`:
      A list of matrices. The intended semantics of `comps[i]` depend on the
      caller; typically it is the matrix describing the map at the i-th selected
      base vertex.

Notes
-----
This type is *not* part of the public API and is not exported. It exists so that
method signatures that refer to `DirectSumModuleMap` are well-typed, and so that
internal callers can pass a small, allocation-free container.
"""
struct DirectSumModuleMap{K}
    comps::Vector{Matrix{K}}
end

# ============================================================
# Module cochain complexes + maps
# ============================================================

struct ModuleCochainComplex{K}
    tmin::Int
    tmax::Int
    terms::Vector{PModule{K}}      # length = tmax-tmin+1
    diffs::Vector{PMorphism{K}}    # length = tmax-tmin
end

"""
    ModuleCochainComplex(tmin, tmax, terms, diffs)

Internal positional constructor that normalizes abstract vector inputs into the
concrete storage used by `ModuleCochainComplex{K}`.
"""
function ModuleCochainComplex(
    tmin::Int,
    tmax::Int,
    terms::AbstractVector{<:PModule{K}},
    diffs::AbstractVector{<:PMorphism{K}},
) where {K}
    terms_vec = PModule{K}[]
    append!(terms_vec, terms)
    diffs_vec = PMorphism{K}[]
    append!(diffs_vec, diffs)
    return ModuleCochainComplex{K}(tmin, tmax, terms_vec, diffs_vec)
end

poset(C::ModuleCochainComplex) = C.terms[1].Q

# Max cohomological degree stored in a module cochain complex.
maxdeg_of_complex(C::ModuleCochainComplex) = C.tmax

@inline function _matrix_is_zero(M)
    # Fast exact check for "all entries are zero".
    for x in M
        if x != 0
            return false
        end
    end
    return true
end

@inline _field_of_complex(C::ModuleCochainComplex) = C.terms[1].field

@inline function _eye(::Type{K}, n::Int) where {K}
    M = zeros(K, n, n)
    for i in 1:n
        M[i, i] = one(K)
    end
    return M
end

# Internal structural equality for PModules used in validation.
# We avoid defining `==` globally. This is needed because zero modules are
# often constructed on the fly, so pointer-identity is too strict.
@inline function _pmodule_equal(M::PModule{K}, N::PModule{K}) where {K}
    M === N && return true
    M.Q === N.Q || return false
    M.dims == N.dims || return false
    # Compare cover-edge maps using store-aligned traversal (succs + maps_to_succ)
    # to avoid keyed lookups in the common validation path.
    storeM = M.edge_maps
    storeN = N.edge_maps
    succs = storeM.succs
    mapsM = storeM.maps_to_succ
    mapsN = storeN.maps_to_succ

    @inbounds for u in 1:nvertices(M.Q)
        Mu = mapsM[u]
        Nu = mapsN[u]
        for j in eachindex(succs[u])
            if Mu[j] != Nu[j]
                return false
            end
        end
    end
    return true
end

"""
    ModuleCochainComplex(terms, diffs; tmin=0, check=true)

Construct a bounded cochain complex of PModules

    C^tmin --d--> C^(tmin+1) --d--> ... --d--> C^tmax

with `terms[i] = C^(tmin + i - 1)` and `diffs[i] = d^(tmin + i - 1)`.

If `check=true`, we validate:
  * all terms live over the same poset;
  * each differential has the correct domain/codomain;
  * d^(t+1) circ d^t = 0 in every degree (fiberwise, vertex-by-vertex).
"""
function ModuleCochainComplex(
    terms::AbstractVector{<:PModule{K}},
    diffs::AbstractVector{<:PMorphism{K}};
    tmin::Int=0,
    check::Bool=true,
) where {K}
    @assert length(diffs) == length(terms) - 1
    tmax = tmin + length(terms) - 1

    if check
        Q = terms[1].Q
        for (i, T) in enumerate(terms)
            if T.Q !== Q
                error("ModuleCochainComplex: term $i lives over a different poset")
            end
        end

        # Check each differential's endpoints.
        for i in 1:length(diffs)
            d = diffs[i]
            if !_pmodule_equal(d.dom, terms[i]) || !_pmodule_equal(d.cod, terms[i+1])
                t = tmin + (i - 1)
                error("ModuleCochainComplex: differential at degree $t has wrong domain/codomain")
            end
        end

        # Check d^(t+1) * d^t = 0 fiberwise.
        Qn = nvertices(Q)
        for i in 1:(length(diffs) - 1)
            t = tmin + (i - 1)
            d1 = diffs[i]
            d2 = diffs[i+1]
            for u in 1:Qn
                prod = d2.comps[u] * d1.comps[u]
                if !_matrix_is_zero(prod)
                    error("ModuleCochainComplex: d^(t+1)*d^t != 0 at degree $t, vertex $u")
                end
            end
        end
    end

    return ModuleCochainComplex{K}(tmin, tmax, terms, diffs)
end

function ModuleCochainComplex(
    terms::AbstractVector{<:PModule},
    diffs::AbstractVector{<:PMorphism};
    tmin::Int=0,
    check::Bool=true,
)
    isempty(terms) && throw(ArgumentError("ModuleCochainComplex: terms must be nonempty"))
    K = typeof(terms[1]).parameters[1]
    return ModuleCochainComplex(
        Vector{PModule{K}}(terms),
        Vector{PMorphism{K}}(diffs);
        tmin = tmin,
        check = check,
    )
end




_term(C::ModuleCochainComplex{K}, t::Int) where {K} =
    (t < C.tmin || t > C.tmax) ? zero_pmodule(poset(C); field=_field_of_complex(C)) : C.terms[t - C.tmin + 1]

_diff(C::ModuleCochainComplex{K}, t::Int) where {K} =
    (t < C.tmin || t >= C.tmax) ? zero_morphism(_term(C,t), _term(C,t+1)) : C.diffs[t - C.tmin + 1]

@inline degree_range(C::ModuleCochainComplex) = C.tmin:C.tmax
@inline component(C::ModuleCochainComplex{K}, t::Int) where {K} = _term(C, t)
@inline differential(C::ModuleCochainComplex{K}, t::Int) where {K} = _diff(C, t)

@inline function describe(C::ModuleCochainComplex)
    return (
        kind=:module_cochain_complex,
        field=_field_of_complex(C),
        nvertices=nvertices(poset(C)),
        degree_range=degree_range(C),
        term_dimensions=Tuple(sum(M.dims) for M in C.terms),
        ndifferentials=length(C.diffs),
    )
end

function Base.show(io::IO, C::ModuleCochainComplex)
    d = describe(C)
    print(io, "ModuleCochainComplex(degrees=", repr(d.degree_range),
          ", term_dimensions=", repr(d.term_dimensions), ")")
end

function Base.show(io::IO, ::MIME"text/plain", C::ModuleCochainComplex)
    d = describe(C)
    print(io, "ModuleCochainComplex",
          "\n  field: ", d.field,
          "\n  nvertices: ", d.nvertices,
          "\n  degree_range: ", repr(d.degree_range),
          "\n  term_dimensions: ", repr(d.term_dimensions),
          "\n  ndifferentials: ", d.ndifferentials)
end

"""
    ModuleCochainComplex(
        Hs::AbstractVector{<:FF.FringeModule},
        ds::AbstractVector{<:IR.PMorphism{K}};
        tmin::Integer = 0,
        check::Bool = true,
    )

Convenience constructor for building a cochain complex from finite fringe modules.

This is a mathematician-facing API feature: if you naturally specify terms using
"upset/downset + matrix" (finite fringe data), you should not have to manually
convert each term to an `IndicatorResolutions.PModule`.

Implementation details:
- Each `FringeModule` term is converted to a `PModule{K}` using
  `IndicatorResolutions.pmodule_from_fringe`.
- We then call the standard `ModuleCochainComplex(::Vector{PModule}, ::Vector{PMorphism}; ...)`
  constructor.
- The coefficient field is taken from each fringe module (and preserved by
  `pmodule_from_fringe`).

Parameters
- `Hs`: terms of the cochain complex as fringe modules.
- `ds`: differentials as `PMorphism{K}`. (Same convention as the PModule constructor.)
- `tmin`: cohomological starting degree (term `Hs[1]` is placed in degree `tmin`).
- `check`: if true, run structural checks (same meaning as in the PModule constructor).
"""
function ModuleCochainComplex(
    Hs::AbstractVector{<:FF.FringeModule},
    ds::AbstractVector{<:IR.PMorphism{K}};
    tmin::Integer = 0,
    check::Bool = true,
 ) where {K}
    # Convert each fringe module into a PModule{K}.
    Ms = Vector{IR.PModule{K}}(undef, length(Hs))
    for i in eachindex(Hs)
        Ms[i] = IR.pmodule_from_fringe(Hs[i])
    end

    # Delegate to the existing, fully-checked constructor.
    return ModuleCochainComplex(
        Ms,
        Vector{IR.PMorphism{K}}(ds);
        tmin = Int(tmin),
        check = check,
    )
end

function ModuleCochainComplex(
    Hs::AbstractVector{<:FF.FringeModule},
    ds::AbstractVector{<:IR.PMorphism};
    tmin::Integer = 0,
    check::Bool = true,
)
    isempty(Hs) && throw(ArgumentError("ModuleCochainComplex: Hs must be nonempty"))
    K = eltype(Hs[1].phi)
    Ms = Vector{IR.PModule{K}}(undef, length(Hs))
    for i in eachindex(Hs)
        Ms[i] = IR.pmodule_from_fringe(Hs[i])
    end
    return ModuleCochainComplex(
        Ms,
        Vector{IR.PMorphism{K}}(ds);
        tmin = Int(tmin),
        check = check,
    )
end

"""
    ModuleCochainMap(C, D, comps; tmin, tmax, check=true)

A cochain map f : C -> D, i.e. degreewise morphisms

    f^t : C^t -> D^t

such that for every degree t:

    d_D^t circ f^t = f^(t+1) circ d_C^t

Outside the provided range `[tmin, tmax]`, the map is taken to be zero.

If `check=true`, we verify the chain-map equation in *all relevant degrees*,
including the boundary degrees where one side uses the implicit zero map.
"""
struct ModuleCochainMap{K}
    C::ModuleCochainComplex{K}
    D::ModuleCochainComplex{K}
    tmin::Int
    tmax::Int
    comps::Vector{PMorphism{K}}
end

_map(f::ModuleCochainMap{K}, t::Int) where {K} =
    (t < f.tmin || t > f.tmax) ?
        zero_morphism(_term(f.C, t), _term(f.D, t)) :
        f.comps[t - f.tmin + 1]

@inline degree_range(f::ModuleCochainMap) = f.tmin:f.tmax
@inline source(f::ModuleCochainMap) = f.C
@inline target(f::ModuleCochainMap) = f.D
@inline component(f::ModuleCochainMap{K}, t::Int) where {K} = _map(f, t)

@inline function describe(f::ModuleCochainMap)
    return (
        kind=:module_cochain_map,
        field=_field_of_complex(f.C),
        degree_range=degree_range(f),
        source_degree_range=degree_range(f.C),
        target_degree_range=degree_range(f.D),
        ncomponents=length(f.comps),
    )
end

function Base.show(io::IO, f::ModuleCochainMap)
    d = describe(f)
    print(io, "ModuleCochainMap(degrees=", repr(d.degree_range),
          ", ncomponents=", d.ncomponents, ")")
end

function Base.show(io::IO, ::MIME"text/plain", f::ModuleCochainMap)
    d = describe(f)
    print(io, "ModuleCochainMap",
          "\n  field: ", d.field,
          "\n  degree_range: ", repr(d.degree_range),
          "\n  source_degree_range: ", repr(d.source_degree_range),
          "\n  target_degree_range: ", repr(d.target_degree_range),
          "\n  ncomponents: ", d.ncomponents)
end

function ModuleCochainMap(
    C::ModuleCochainComplex{K},
    D::ModuleCochainComplex{K},
    comps::Vector{PMorphism{K}};
    tmin=nothing,
    tmax=nothing,
    check::Bool=true,
) where {K}
    tmin = isnothing(tmin) ? C.tmin : tmin
    tmax = isnothing(tmax) ? C.tmax : tmax
    @assert length(comps) == tmax - tmin + 1

    if check
        if poset(C) !== poset(D)
            error("ModuleCochainMap: domain and codomain complexes live over different posets")
        end

        # Degreewise domain/codomain checks (structural, not pointer-only).
        for t in tmin:tmax
            ft = comps[t - tmin + 1]
            if !_pmodule_equal(ft.dom, _term(C, t)) || !_pmodule_equal(ft.cod, _term(D, t))
                error("ModuleCochainMap: component f^$t has wrong domain/codomain")
            end
        end

        # Chain-map equation must also hold at boundary degrees:
        # we check t in [tmin-1, tmax].
        Q = poset(C)
        for t in (tmin - 1):tmax
            dD = _diff(D, t)
            dC = _diff(C, t)

            ft  = (t < tmin)  ? zero_morphism(_term(C, t), _term(D, t)) :
                                comps[t - tmin + 1]
            ftp = (t + 1 > tmax) ? zero_morphism(_term(C, t+1), _term(D, t+1)) :
                                  comps[t + 1 - tmin + 1]

            for u in 1:nvertices(Q)
                lhs = dD.comps[u] * ft.comps[u]
                rhs = ftp.comps[u] * dC.comps[u]
                if lhs != rhs
                    error("ModuleCochainMap: chain map equation fails at degree $t, vertex $u")
                end
            end
        end
    end

    return ModuleCochainMap{K}(C, D, tmin, tmax, comps)
end

function ModuleCochainMap(
    C::ModuleCochainComplex{K},
    D::ModuleCochainComplex{K},
    comps::AbstractVector{<:PMorphism};
    tmin=nothing,
    tmax=nothing,
    check::Bool=true,
) where {K}
    return ModuleCochainMap(
        C,
        D,
        Vector{PMorphism{K}}(comps);
        tmin = tmin,
        tmax = tmax,
        check = check,
    )
end

# Identity cochain map on a module complex.
function idmap(C::ModuleCochainComplex{K}) where {K}
    comps = [id_morphism(_term(C, t)) for t in C.tmin:C.tmax]
    return ModuleCochainMap(C, C, comps; tmin=C.tmin, tmax=C.tmax, check=true)
end


# ============================================================
# Cochain homotopies
# ============================================================

"""
    ModuleCochainHomotopy(f, g, hcomps; tmin, tmax, check=true)

A cochain homotopy between cochain maps f,g : C -> D.

The data is morphisms
    h^t : C^t -> D^(t-1)
such that for all t:

    f^t - g^t = d_D^(t-1) circ h^t + h^(t+1) circ d_C^t

Outside `[tmin,tmax]`, h^t is interpreted as 0.

If `check=true`, validation is performed fiberwise.
"""
struct ModuleCochainHomotopy{K}
    f::ModuleCochainMap{K}
    g::ModuleCochainMap{K}
    tmin::Int
    tmax::Int
    comps::Vector{PMorphism{K}}
end

_hcomp(H::ModuleCochainHomotopy{K}, t::Int) where {K} =
    (t < H.tmin || t > H.tmax) ?
        zero_morphism(_term(H.f.C, t), _term(H.f.D, t-1)) :
        H.comps[t - H.tmin + 1]

@inline degree_range(H::ModuleCochainHomotopy) = H.tmin:H.tmax
@inline source_map(H::ModuleCochainHomotopy) = H.f
@inline target_map(H::ModuleCochainHomotopy) = H.g
@inline component(H::ModuleCochainHomotopy{K}, t::Int) where {K} = _hcomp(H, t)

@inline function describe(H::ModuleCochainHomotopy)
    return (
        kind=:module_cochain_homotopy,
        field=_field_of_complex(H.f.C),
        degree_range=degree_range(H),
        source_map_range=degree_range(H.f),
        target_map_range=degree_range(H.g),
        ncomponents=length(H.comps),
    )
end

function Base.show(io::IO, H::ModuleCochainHomotopy)
    d = describe(H)
    print(io, "ModuleCochainHomotopy(degrees=", repr(d.degree_range),
          ", ncomponents=", d.ncomponents, ")")
end

function Base.show(io::IO, ::MIME"text/plain", H::ModuleCochainHomotopy)
    d = describe(H)
    print(io, "ModuleCochainHomotopy",
          "\n  field: ", d.field,
          "\n  degree_range: ", repr(d.degree_range),
          "\n  source_map_range: ", repr(d.source_map_range),
          "\n  target_map_range: ", repr(d.target_map_range),
          "\n  ncomponents: ", d.ncomponents)
end

"""
    is_cochain_homotopy(H)

Return true iff H satisfies the cochain homotopy identity.
"""
function is_cochain_homotopy(H::ModuleCochainHomotopy{K}) where {K}
    f = H.f
    g = H.g
    C = f.C
    D = f.D

    Q = poset(C)
    tcheck_min = min(f.tmin, g.tmin, H.tmin) - 1
    tcheck_max = max(f.tmax, g.tmax, H.tmax)

    for t in tcheck_min:tcheck_max
        ft = _map(f, t)
        gt = _map(g, t)

        dD_prev = _diff(D, t-1)
        dC_t    = _diff(C, t)

        ht   = _hcomp(H, t)
        htp1 = _hcomp(H, t+1)

        for u in 1:nvertices(Q)
            left  = ft.comps[u] - gt.comps[u]
            right = dD_prev.comps[u] * ht.comps[u] + htp1.comps[u] * dC_t.comps[u]
            if left != right
                return false
            end
        end
    end

    return true
end

function ModuleCochainHomotopy(
    f::ModuleCochainMap{K},
    g::ModuleCochainMap{K},
    comps::Vector{PMorphism{K}};
    tmin=nothing,
    tmax=nothing,
    check::Bool=true,
) where {K}
    if f.C !== g.C || f.D !== g.D
        error("ModuleCochainHomotopy: maps must have the same domain and codomain complexes")
    end

    tmin = isnothing(tmin) ? f.C.tmin : tmin
    tmax = isnothing(tmax) ? f.C.tmax : tmax
    @assert length(comps) == tmax - tmin + 1

    if check
        for t in tmin:tmax
            ht = comps[t - tmin + 1]
            if !_pmodule_equal(ht.dom, _term(f.C, t)) || !_pmodule_equal(ht.cod, _term(f.D, t-1))
                error("ModuleCochainHomotopy: component h^$t has wrong domain/codomain")
            end
        end
    end

    H = ModuleCochainHomotopy{K}(f, g, tmin, tmax, comps)

    if check && !is_cochain_homotopy(H)
        error("ModuleCochainHomotopy: homotopy identity does not hold")
    end

    return H
end

function ModuleCochainHomotopy(
    f::ModuleCochainMap{K},
    g::ModuleCochainMap{K},
    comps::AbstractVector{<:PMorphism};
    tmin=nothing,
    tmax=nothing,
    check::Bool=true,
) where {K}
    return ModuleCochainHomotopy(
        f,
        g,
        Vector{PMorphism{K}}(comps);
        tmin = tmin,
        tmax = tmax,
        check = check,
    )
end



# ============================================================
# shift / extend_range for module complexes
# ============================================================

function shift(C::ModuleCochainComplex{K}, k::Int) where {K}
    if k == 0
        return C
    end
    diffs = C.diffs
    if isodd(k)
        diffs = [PMorphism{K}(d.dom, d.cod, [-M for M in d.comps]) for d in diffs]
    end
    return ModuleCochainComplex{K}(C.tmin + k, C.tmax + k, C.terms, diffs)
end

function extend_range(C::ModuleCochainComplex{K}, tmin::Int, tmax::Int) where {K}
    Q = poset(C)
    terms = PModule{K}[]
    diffs = PMorphism{K}[]
    for t in tmin:tmax
        push!(terms, _term(C,t))
    end
    for t in tmin:(tmax-1)
        push!(diffs, _diff(C,t))
    end
    return ModuleCochainComplex{K}(tmin,tmax,terms,diffs)
end

# ============================================================
# mapping cone for module cochain maps
# ============================================================

"""
    mapping_cone(f::ModuleCochainMap) -> ModuleCochainComplex

Construct the module-valued mapping cone of a cochain map `f : C -> D`.

The returned complex represents the standard cone object `Cone(f)` in the
derived category. It is the canonical module-level path when you want the cone
itself, not just its cohomology. For cheap-first inspection, call
[`module_complex_summary`](@ref) on the result before exploring individual
terms or differentials.
"""
function mapping_cone(f::ModuleCochainMap{K}) where {K}
    C = f.C
    D = f.D
    tmin = min(D.tmin, C.tmin - 1)
    tmax = max(D.tmax, C.tmax - 1)

    terms = PModule{K}[]
    diffs = PMorphism{K}[]

    for t in tmin:tmax
        push!(terms, direct_sum(_term(D,t), _term(C,t+1)))
    end

    for t in tmin:(tmax-1)
        dom = terms[t - tmin + 1]
        cod = terms[t - tmin + 2]
        Dt  = _term(D,t); Dt1 = _term(D,t+1)
        Ct1 = _term(C,t+1); Ct2 = _term(C,t+2)

        dD  = _diff(D,t)
        dC  = _diff(C,t+1)
        ft1 = _map(f,t+1)

        comps = Vector{Matrix{K}}(undef, poset(C).n)
        for u in 1:poset(C).n
            a = Dt.dims[u]; b = Ct1.dims[u]
            c = Dt1.dims[u]; d = Ct2.dims[u]
            M = zeros(K, c+d, a+b)
            if c>0 && a>0; M[1:c, 1:a] .= dD.comps[u]; end
            if c>0 && b>0; M[1:c, a+1:a+b] .= ft1.comps[u]; end
            if d>0 && b>0; M[c+1:c+d, a+1:a+b] .= -dC.comps[u]; end
            comps[u] = M
        end
        push!(diffs, PMorphism{K}(dom,cod,comps))
    end

    return ModuleCochainComplex{K}(tmin,tmax,terms,diffs)
end

# Triangle object (optional but included)
struct ModuleDistinguishedTriangle{K}
    C::ModuleCochainComplex{K}
    D::ModuleCochainComplex{K}
    Cone::ModuleCochainComplex{K}
    f::ModuleCochainMap{K}
    i::ModuleCochainMap{K}
    p::ModuleCochainMap{K}
end

@inline triangle_objects(T::ModuleDistinguishedTriangle) = (; source=T.C, target=T.D, cone=T.Cone)
@inline triangle_maps(T::ModuleDistinguishedTriangle) = (; morphism=T.f, inclusion=T.i, projection=T.p)
@inline connecting_map(T::ModuleDistinguishedTriangle) = T.p

@inline function describe(T::ModuleDistinguishedTriangle)
    return (
        kind=:module_distinguished_triangle,
        field=_field_of_complex(T.C),
        source_degree_range=degree_range(T.C),
        target_degree_range=degree_range(T.D),
        cone_degree_range=degree_range(T.Cone),
    )
end

function Base.show(io::IO, T::ModuleDistinguishedTriangle)
    d = describe(T)
    print(io, "ModuleDistinguishedTriangle(cone_degrees=", repr(d.cone_degree_range), ")")
end

function Base.show(io::IO, ::MIME"text/plain", T::ModuleDistinguishedTriangle)
    d = describe(T)
    print(io, "ModuleDistinguishedTriangle",
          "\n  field: ", d.field,
          "\n  source_degree_range: ", repr(d.source_degree_range),
          "\n  target_degree_range: ", repr(d.target_degree_range),
          "\n  cone_degree_range: ", repr(d.cone_degree_range))
end

"""
    mapping_cone_triangle(f::ModuleCochainMap) -> ModuleDistinguishedTriangle

Return the canonical distinguished triangle

`C --f--> D -> Cone(f) -> C[1]`

attached to a module-valued cochain map.

Use this when the categorical triangle is the object of interest. For quick
inspection, prefer [`triangle_summary`](@ref) or [`describe`](@ref) before
descending into the component maps.
"""
function mapping_cone_triangle(f::ModuleCochainMap{K}) where {K}
    C, D = f.C, f.D
    Cone = mapping_cone(f)
    # maps D -> Cone and Cone -> C[1]
    tmin = Cone.tmin
    tmax = Cone.tmax
    i_comps = PMorphism{K}[]
    p_comps = PMorphism{K}[]
    for t in tmin:tmax
        Dt = _term(D,t)
        Ct1 = _term(C,t+1)
        S = _term(Cone,t)
        # injection D^t -> D^t oplus C^{t+1}
        comps_i = Vector{Matrix{K}}(undef, poset(C).n)
        comps_p = Vector{Matrix{K}}(undef, poset(C).n)
        for u in 1:poset(C).n
            a = Dt.dims[u]; b = Ct1.dims[u]
            inj = zeros(K, a+b, a)
            proj = zeros(K, b, a+b)
            if a > 0
                inj[1:a, 1:a] .= _eye(K, a)
            end
            if b > 0
                proj[1:b, a+1:a+b] .= _eye(K, b)
            end
            comps_i[u] = inj
            comps_p[u] = proj
        end
        push!(i_comps, PMorphism{K}(Dt,S,comps_i))
        push!(p_comps, PMorphism{K}(S,Ct1,comps_p))
    end
    i = ModuleCochainMap(D,Cone,i_comps; tmin=tmin, tmax=tmax)
    p = ModuleCochainMap(Cone,shift(C,1),p_comps; tmin=tmin, tmax=tmax)
    return ModuleDistinguishedTriangle{K}(C,D,Cone,f,i,p)
end

# ============================================================
# Cohomology as PModules
# ============================================================

"""
    cohomology_module_data(C, t)

Return the module-valued cohomology data in degree `t` for a
[`ModuleCochainComplex`](@ref).

The result is the named tuple `(Z, iZ, B, iB, j, H, q)` where:
- `Z = ker d^t`,
- `iZ : Z -> C^t` is the kernel inclusion,
- `B = im d^{t-1}`,
- `iB : B -> C^t` is the image inclusion,
- `j : B -> Z` is the induced inclusion into cycles,
- `H = Z / B` is the cohomology module,
- `q : Z -> H` is the quotient map.

This is the heavy path when you need the full submodule/quotient structure.
If you only need the cohomology module object, use [`cohomology_module`](@ref)
instead.
"""
function cohomology_module_data(C::ModuleCochainComplex{K}, t::Int) where {K}
    M  = _term(C,t)
    d0 = _diff(C,t-1)
    d1 = _diff(C,t)

    Z, iZ = kernel_with_inclusion(d1)
    B, iB = image_with_inclusion(d0)

    # j: B -> Z such that iZ circ j = iB
    Q = poset(C)
    jcomps = Vector{Matrix{K}}(undef, nvertices(Q))
    field = M.field
    for u in 1:nvertices(Q)
        if B.dims[u] == 0
            jcomps[u] = zeros(K, Z.dims[u], 0)
        elseif Z.dims[u] == 0
            jcomps[u] = zeros(K, 0, B.dims[u])
        else
            jcomps[u] = FieldLinAlg.solve_fullcolumn(field, iZ.comps[u], iB.comps[u]; check_rhs=false)
        end
    end
    j = PMorphism{K}(B,Z,jcomps)

    H, q = _cokernel_module(j)

    return (Z=Z, iZ=iZ, B=B, iB=iB, j=j, H=H, q=q)
end

"""
    cohomology_module(C, t) -> PModule

Return the cohomology module `H^t(C)` in degree `t`.

This is the cheap canonical path for module-valued cohomology when you do not
need cycle, boundary, or quotient witness maps. Use
[`cohomology_module_data`](@ref) only when the heavier quotient data is
mathematically necessary.
"""
cohomology_module(C::ModuleCochainComplex{K}, t::Int) where {K} = cohomology_module_data(C,t).H

# induced map on cohomology modules
function induced_map_on_cohomology_modules(f::ModuleCochainMap{K}, t::Int) where {K}
    field = _field_of_complex(f.C)
    Cd = cohomology_module_data(f.C,t)
    Dd = cohomology_module_data(f.D,t)

    ft = _map(f,t)
    Q = poset(f.C)
    compsH = Vector{Matrix{K}}(undef, nvertices(Q))
    @inbounds for u in 1:nvertices(Q)
        rhsZ = ft.comps[u] * Cd.iZ.comps[u]
        if Dd.Z.dims[u] == 0
            compsH[u] = zeros(K, Dd.H.dims[u], Cd.H.dims[u])
        else
            z_to_z = FieldLinAlg.solve_fullcolumn(field, Dd.iZ.comps[u], rhsZ)
            rhsH = Dd.q.comps[u] * z_to_z
            qC = Cd.q.comps[u]
            if size(qC, 1) == 0
                compsH[u] = zeros(K, size(rhsH, 1), 0)
            else
                rinv = AbelianCategories._right_inverse_full_row(field, qC)
                compsH[u] = rhsH * rinv
            end
        end
    end
    return PMorphism{K}(Cd.H, Dd.H, compsH)
end

# quasi-isomorphism check
function is_isomorphism(f::PMorphism{K}) where {K}
    Q = f.dom.Q
    field = f.dom.field
    for u in 1:nvertices(Q)
        if f.dom.dims[u] != f.cod.dims[u]
            return false
        end
        if FieldLinAlg.rank(field, f.comps[u]) != f.dom.dims[u]
            return false
        end
    end
    return true
end

"""
    is_quasi_isomorphism(f::ModuleCochainMap) -> Bool

Return `true` exactly when `f` induces isomorphisms on all module-valued
cohomology groups in its supported degree range.

This is the cheap-first predicate for testing whether a module cochain map is a
quasi-isomorphism. When it returns `false` and you need more detail, inspect
the induced maps from [`induced_map_on_cohomology_modules`](@ref) degree by
degree.
"""
function is_quasi_isomorphism(f::ModuleCochainMap{K}) where {K}
    tmin = min(f.C.tmin, f.D.tmin)
    tmax = max(f.C.tmax, f.D.tmax)
    for t in tmin:tmax
        hf = induced_map_on_cohomology_modules(f,t)
        if !is_isomorphism(hf)
            return false
        end
    end
    return true
end

# ============================================================
# RHom and derived tensor (bicomplex + total complex)
# ============================================================

struct RHomComplex{K}
    C::ModuleCochainComplex{K}
    N::PModule{K}
    resN
    homs::Array{HomSpace{K},2}
    DC::DoubleComplex{K}
    tot::CochainComplex{K}
end

"""
    source_module(R::RHomComplex)
    target_module(R::RHomComplex)
    underlying_complex(R::RHomComplex)

Semantic accessors for an [`RHomComplex`](@ref).

- `source_module(R)` returns the module complex `C`
- `target_module(R)` returns the coefficient module `N`
- `underlying_complex(R)` returns the total cochain complex computing
  `RHom(C, N)`

Use these accessors instead of raw field inspection when exploring derived Hom
data.
"""
@inline source_module(R::RHomComplex) = R.C
@inline target_module(R::RHomComplex) = R.N
@inline underlying_complex(R::RHomComplex) = R.tot

@inline function describe(R::RHomComplex)
    return (
        kind=:rhom_complex,
        field=R.N.field,
        source_degree_range=degree_range(R.C),
        target_total_dim=sum(R.N.dims),
        block_shape=size(R.homs),
        total_degree_range=R.tot.tmin:R.tot.tmax,
    )
end

function Base.show(io::IO, R::RHomComplex)
    d = describe(R)
    print(io, "RHomComplex(total_degrees=", repr(d.total_degree_range),
          ", block_shape=", repr(d.block_shape), ")")
end

function Base.show(io::IO, ::MIME"text/plain", R::RHomComplex)
    d = describe(R)
    print(io, "RHomComplex",
          "\n  field: ", d.field,
          "\n  source_degree_range: ", repr(d.source_degree_range),
          "\n  target_total_dim: ", d.target_total_dim,
          "\n  block_shape: ", repr(d.block_shape),
          "\n  total_degree_range: ", repr(d.total_degree_range))
end


function RHomComplex(
    C::ModuleCochainComplex{K},
    N::PModule{K};
    maxlen::Int = 3,
    resN = nothing,
    cache::Union{Nothing,HomSystemCache}=nothing,
    threads::Bool = (Threads.nthreads() > 1),
) where {K}
    Q = N.Q
    maxlen = maxlen
    maxdeg = maxdeg_of_complex(C)
    resN = (resN === nothing) ? injective_resolution(N, ResolutionOptions(maxlen=maxlen)) : resN

    na, nb = maxdeg + 1, maxlen + 1
    homs = Array{HomSpace{K}}(undef, na, nb)
    dims = zeros(Int, na, nb)

    # Build Hom blocks (expensive) in parallel if requested.
    if threads && Threads.nthreads() > 1
        nT = Threads.nthreads()
        Threads.@threads for slot in 1:nT
            for idx in slot:nT:(na * nb)
                ia = div((idx - 1), nb) + 1
                ib = (idx - 1) % nb + 1
                p = ia - 1
                q = ib - 1

                Cp = _term(C, p)
                Eb = resN.Emods[q + 1]

                h = hom_with_cache(Cp, Eb; cache=cache)
                homs[ia, ib] = h
                dims[ia, ib] = dim(h)
            end
        end
    else
        for ia in 1:na, ib in 1:nb
            p = ia - 1
            q = ib - 1
            Cp = _term(C, p)
            Eb = resN.Emods[q + 1]
            homs[ia, ib] = hom_with_cache(Cp, Eb; cache=cache)
            dims[ia, ib] = dim(homs[ia, ib])
        end
    end

    # Vertical (C direction) differentials.
    dv = Array{SparseMatrixCSC{K, Int}}(undef, na, nb)
    if threads && Threads.nthreads() > 1
        nT = Threads.nthreads()
        Threads.@threads for slot in 1:nT
            for idx in slot:nT:(na * nb)
                ia = div((idx - 1), nb) + 1
                ib = (idx - 1) % nb + 1
                if ia == na
                    dv[ia, ib] = spzeros(K, dims[ia, ib], 0)
                else
                    p = ia - 1
                    dv[ia, ib] = precompose_matrix_cached(homs[ia, ib], homs[ia + 1, ib], _diff(C, p); cache=cache)
                end
            end
        end
    else
        for ia in 1:na, ib in 1:nb
            if ia == na
                dv[ia, ib] = spzeros(K, dims[ia, ib], 0)
            else
                p = ia - 1
                dv[ia, ib] = precompose_matrix_cached(homs[ia, ib], homs[ia + 1, ib], _diff(C, p); cache=cache)
            end
        end
    end

    # Horizontal (resolution direction) differentials.
    dh = Array{SparseMatrixCSC{K, Int}}(undef, na, nb)
    if threads && Threads.nthreads() > 1
        nT = Threads.nthreads()
        Threads.@threads for slot in 1:nT
            for idx in slot:nT:(na * nb)
                ia = div((idx - 1), nb) + 1
                ib = (idx - 1) % nb + 1
                if ib == nb
                    dh[ia, ib] = spzeros(K, dims[ia, ib], 0)
                else
                    q = ib - 1
                    dh[ia, ib] = postcompose_matrix_cached(homs[ia, ib + 1], homs[ia, ib], resN.d_mor[q + 1]; cache=cache)
                end
            end
        end
    else
        for ia in 1:na, ib in 1:nb
            if ib == nb
                dh[ia, ib] = spzeros(K, dims[ia, ib], 0)
            else
                q = ib - 1
                dh[ia, ib] = postcompose_matrix_cached(homs[ia, ib + 1], homs[ia, ib], resN.d_mor[q + 1]; cache=cache)
            end
        end
    end

    # Index convention: a = cochain degree in C (0..maxdeg), b = injective degree in resN (0..maxlen).
    DC = DoubleComplex{K}(0, maxdeg, 0, maxlen, dims, dv, dh)
    tot = total_complex(DC)
    return RHomComplex{K}(C, N, resN, homs, DC, tot)
end

function RHomComplex(
    C::ModuleCochainComplex{K},
    H::FF.FringeModule{K};
    maxlen::Int = 3,
    resN = nothing,
    cache::Union{Nothing,HomSystemCache}=nothing,
    threads::Bool = (Threads.nthreads() > 1),
) where {K}
    return _rhom_complex_cached(C, H; maxlen=maxlen, resN=resN, cache=cache, threads=threads)
end

"""
    RHom(C, N; maxlen=3, ...) -> CochainComplex

Return the total cochain complex computing the derived Hom object `RHom(C, N)`.

This is the canonical compute path when you want the chain-level object that
feeds cohomology, spectral sequences, or induced maps. If you want the richer
container with the bicomplex blocks and cached Hom pieces, use
[`RHomComplex`](@ref) instead. For cheap inspection before touching chain data,
prefer [`rhom_summary`](@ref) on the container form.
"""
RHom(C::ModuleCochainComplex{K}, N::PModule{K}; kwargs...) where {K} =
    _rhom_complex_cached(C, N; kwargs...).tot
RHom(C::ModuleCochainComplex{K}, H::FF.FringeModule{K}; kwargs...) where {K} =
    _rhom_complex_cached(C, H; kwargs...).tot


# ------------------------------------------------------------
# Functoriality: induced maps on RHom(-, N)
# ------------------------------------------------------------

@inline function _cc_dim_at(C::CochainComplex, t::Int)
    if t < C.tmin || t > C.tmax
        return 0
    end
    return C.dims[t - C.tmin + 1]
end

function _tot_block_offsets(DC::DoubleComplex, t::Int)
    d = Dict{Tuple{Int,Int},Int}()
    off = 1
    for a in DC.amin:DC.amax
        b = t - a
        if DC.bmin <= b <= DC.bmax
            ai = a - DC.amin + 1
            bi = b - DC.bmin + 1
            d[(a,b)] = off
            off += DC.dims[ai,bi]
        end
    end
    return d
end

function _build_rhom_map_first_plan(Rdom::RHomComplex{K}, Rcod::RHomComplex{K}) where {K}
    tmin = min(Rdom.tot.tmin, Rcod.tot.tmin)
    tmax = max(Rdom.tot.tmax, Rcod.tot.tmax)
    degrees = Vector{_RHomMapFirstDegreePlan{HomSpace{K}}}(undef, tmax - tmin + 1)
    dims_src = Vector{Int}(undef, length(degrees))
    dims_tgt = Vector{Int}(undef, length(degrees))
    for idx in eachindex(degrees)
        t = tmin + idx - 1
        dims_src[idx] = _cc_dim_at(Rdom.tot, t)
        dims_tgt[idx] = _cc_dim_at(Rcod.tot, t)
        off_src = _tot_block_offsets(Rdom.DC, t)
        off_tgt = _tot_block_offsets(Rcod.DC, t)
        row_off = Int[]
        col_off = Int[]
        pdeg = Int[]
        hdom = HomSpace{K}[]
        hcod = HomSpace{K}[]
        for ((a, b), src_off) in off_src
            row = get(off_tgt, (a, b), 0)
            row == 0 && continue
            ai_src = a - Rdom.DC.amin + 1
            bi_src = b - Rdom.DC.bmin + 1
            ai_tgt = a - Rcod.DC.amin + 1
            bi_tgt = b - Rcod.DC.bmin + 1
            dim_block_src = Rdom.DC.dims[ai_src, bi_src]
            dim_block_tgt = Rcod.DC.dims[ai_tgt, bi_tgt]
            (dim_block_src == 0 || dim_block_tgt == 0) && continue
            push!(row_off, row - 1)
            push!(col_off, src_off - 1)
            push!(pdeg, a)
            push!(hdom, Rdom.homs[ai_src, bi_src])
            push!(hcod, Rcod.homs[ai_tgt, bi_tgt])
        end
        degrees[idx] = _RHomMapFirstDegreePlan{HomSpace{K}}(row_off, col_off, pdeg, hdom, hcod)
    end
    return _RHomMapPlan{_RHomMapFirstDegreePlan{HomSpace{K}}}(tmin, tmax, dims_src, dims_tgt, degrees)
end

@inline function _rhom_map_first_plan(Rdom::RHomComplex{K},
                                      Rcod::RHomComplex{K},
                                      ::Nothing) where {K}
    return _build_rhom_map_first_plan(Rdom, Rcod)
end

@inline function _rhom_map_first_plan(Rdom::RHomComplex{K},
                                      Rcod::RHomComplex{K},
                                      cache::HomSystemCache) where {K}
    return _pair_plan_cached!(_RHOM_MAP_FIRST_PLAN_CACHES, _RHOM_MAP_FIRST_PLAN_LOCK, cache, Rdom, Rcod) do
        _build_rhom_map_first_plan(Rdom, Rcod)
    end::_RHomMapPlan{_RHomMapFirstDegreePlan{HomSpace{K}}}
end

function _build_rhom_map_second_plan(Rsrc::RHomComplex{K}, Rtgt::RHomComplex{K}) where {K}
    tmin = min(Rsrc.tot.tmin, Rtgt.tot.tmin)
    tmax = max(Rsrc.tot.tmax, Rtgt.tot.tmax)
    degrees = Vector{_RHomMapSecondDegreePlan{HomSpace{K}}}(undef, tmax - tmin + 1)
    dims_src = Vector{Int}(undef, length(degrees))
    dims_tgt = Vector{Int}(undef, length(degrees))
    for idx in eachindex(degrees)
        t = tmin + idx - 1
        dims_src[idx] = _cc_dim_at(Rsrc.tot, t)
        dims_tgt[idx] = _cc_dim_at(Rtgt.tot, t)
        off_src = _tot_block_offsets(Rsrc.DC, t)
        off_tgt = _tot_block_offsets(Rtgt.DC, t)
        row_off = Int[]
        col_off = Int[]
        ideg = Int[]
        hsrc = HomSpace{K}[]
        htgt = HomSpace{K}[]
        for ((A, B), tgt_off) in off_tgt
            src_off = get(off_src, (A, B), 0)
            src_off == 0 && continue
            ia_src = A - Rsrc.DC.amin + 1
            ib_src = B - Rsrc.DC.bmin + 1
            ia_tgt = A - Rtgt.DC.amin + 1
            ib_tgt = B - Rtgt.DC.bmin + 1
            dim_block_src = Rsrc.DC.dims[ia_src, ib_src]
            dim_block_tgt = Rtgt.DC.dims[ia_tgt, ib_tgt]
            (dim_block_src == 0 || dim_block_tgt == 0) && continue
            push!(row_off, tgt_off - 1)
            push!(col_off, src_off - 1)
            push!(ideg, B)
            push!(hsrc, Rsrc.homs[ia_src, ib_src])
            push!(htgt, Rtgt.homs[ia_tgt, ib_tgt])
        end
        degrees[idx] = _RHomMapSecondDegreePlan{HomSpace{K}}(row_off, col_off, ideg, hsrc, htgt)
    end
    return _RHomMapPlan{_RHomMapSecondDegreePlan{HomSpace{K}}}(tmin, tmax, dims_src, dims_tgt, degrees)
end

@inline function _rhom_map_second_plan(Rsrc::RHomComplex{K},
                                       Rtgt::RHomComplex{K},
                                       ::Nothing) where {K}
    return _build_rhom_map_second_plan(Rsrc, Rtgt)
end

@inline function _rhom_map_second_plan(Rsrc::RHomComplex{K},
                                       Rtgt::RHomComplex{K},
                                       cache::HomSystemCache) where {K}
    return _pair_plan_cached!(_RHOM_MAP_SECOND_PLAN_CACHES, _RHOM_MAP_SECOND_PLAN_LOCK, cache, Rsrc, Rtgt) do
        _build_rhom_map_second_plan(Rsrc, Rtgt)
    end::_RHomMapPlan{_RHomMapSecondDegreePlan{HomSpace{K}}}
end

"""
    rhom_map_first(f, Rdom, Rcod; check=true)

Induced map on derived Hom complexes in the first variable.

If f : C -> D is a cochain map, then RHom(-,N) is contravariant, so we obtain:

    RHom(D,N) -> RHom(C,N)

`Rdom` must be `RHomComplex(D,N)` and `Rcod` must be `RHomComplex(C,N)`.

Strict functoriality (exact equality of matrices under composition) requires
that both RHom complexes were built using the same injective resolution
object (`Rdom.resN === Rcod.resN`).
"""
function rhom_map_first(
    f::ModuleCochainMap{K},
    Rdom::RHomComplex{K},
    Rcod::RHomComplex{K};
    check::Bool=true,
    cache::Union{Nothing,HomSystemCache}=nothing,
    threads::Bool = (Threads.nthreads() > 1),
) where {K}
    if f.C !== Rcod.C || f.D !== Rdom.C
        error("rhom_map_first: RHom complexes do not match the given map f : C -> D")
    end
    if Rdom.N !== Rcod.N
        error("rhom_map_first: codomain module N differs")
    end
    if Rdom.resN !== Rcod.resN
        error("rhom_map_first: strict functoriality requires the same injective resolution object")
    end

    tot_src = Rdom.tot
    tot_tgt = Rcod.tot

    plan = _rhom_map_first_plan(Rdom, Rcod, cache)
    tmin = plan.tmin
    tmax = plan.tmax
    maps = Vector{SparseMatrixCSC{K,Int}}(undef, length(plan.degrees))

    if threads && Threads.nthreads() > 1 && (tmax >= tmin)
        Threads.@threads for idx in eachindex(plan.degrees)
            dim_src = plan.dims_src[idx]
            dim_tgt = plan.dims_tgt[idx]

            if dim_src == 0 || dim_tgt == 0
                maps[idx] = spzeros(K, dim_tgt, dim_src)
                continue
            end
            dplan = plan.degrees[idx]
            I = Int[]; J = Int[]; V = K[]
            for k in eachindex(dplan.pdeg)
                F = precompose_matrix_cached(dplan.Hdom[k], dplan.Hcod[k], _map(f, dplan.pdeg[k]); cache=cache)
                _append_scaled_triplets!(I, J, V, F, dplan.row_off[k], dplan.col_off[k])
            end

            maps[idx] = sparse(I, J, V, dim_tgt, dim_src)
        end
    else
        for idx in eachindex(plan.degrees)
            dim_src = plan.dims_src[idx]
            dim_tgt = plan.dims_tgt[idx]

            if dim_src == 0 || dim_tgt == 0
                maps[idx] = spzeros(K, dim_tgt, dim_src)
                continue
            end
            dplan = plan.degrees[idx]
            I = Int[]; J = Int[]; V = K[]
            for k in eachindex(dplan.pdeg)
                F = precompose_matrix_cached(dplan.Hdom[k], dplan.Hcod[k], _map(f, dplan.pdeg[k]); cache=cache)
                _append_scaled_triplets!(I, J, V, F, dplan.row_off[k], dplan.col_off[k])
            end

            maps[idx] = sparse(I, J, V, dim_tgt, dim_src)
        end
    end

    return CochainMap(tot_src, tot_tgt, maps; tmin=tmin, tmax=tmax, check=check)
end

"""
    rhom_map_first(f, N; maxlen=3, resN=nothing, check=true)

Convenience wrapper: build RHom complexes using a shared injective resolution,
then return the induced map on totals.
"""
function rhom_map_first(
    f::ModuleCochainMap{K},
    N::PModule{K};
    maxlen::Int=3,
    resN=nothing,
    check::Bool=true,
    cache::Union{Nothing,HomSystemCache}=nothing,
    threads::Bool = (Threads.nthreads() > 1),
) where {K}
    resN = isnothing(resN) ? _injective_resolution_cached(N, maxlen, cache) : resN
    Rdom = _rhom_complex_cached(f.D, N; maxlen=maxlen, resN=resN, cache=cache, threads=threads)
    Rcod = _rhom_complex_cached(f.C, N; maxlen=maxlen, resN=resN, cache=cache, threads=threads)
    return rhom_map_first(f, Rdom, Rcod; check=check, cache=cache, threads=threads)
end

function rhom_map_first(
    f::ModuleCochainMap{K},
    H::FF.FringeModule{K};
    maxlen::Int=3,
    resN=nothing,
    check::Bool=true,
    cache::Union{Nothing,HomSystemCache}=nothing,
    threads::Bool = (Threads.nthreads() > 1),
) where {K}
    return rhom_map_first(f, _pmodule_from_fringe_cached(H, cache);
        maxlen=maxlen, resN=resN, check=check, cache=cache, threads=threads)
end


################################################################################
# Internal utilities for a canonical (deterministic) chain map between injective
# resolutions. This is the key ingredient for rhom_map_second (covariant in N).
################################################################################

"""
    _bases_from_injective_gens(gens) -> Vector{Int}

Internal helper used by injective chain-map lifting.

In the current codebase, `DerivedFunctors.InjectiveResolution.gens[b+1]` is already a
flat `Vector{Int}` containing the *base vertices* (one per principal downset summand,
with repetition).

Older versions stored injective generators in a "gens_at" format:
a vector indexed by vertices, where `gens_at[u]` was a list of generators based at `u`.

This function accepts both layouts and returns the canonical flat list of base vertices.
"""
function _bases_from_injective_gens(gens)::Vector{Int}
    isempty(gens) && return Int[]

    # New format: already a flat list of base vertices.
    if gens isa Vector{Int}
        return gens
    end
    if eltype(gens) <: Integer
        return [Int(x) for x in gens]
    end

    # Older format: gens[u] is iterable, with one entry per generator based at u.
    bases = Int[]
    for u in 1:length(gens)
        for _ in gens[u]
            push!(bases, u)
        end
    end
    return bases
end

# Active indices at vertex i for a downset direct sum determined by bases.
# A generator based at u is active at i iff i <= u in the poset.
# Returned lists are in increasing global generator index order.
function _active_indices_from_bases(Q::AbstractPoset, bases::Vector{Int})
    n = nvertices(Q)
    by_base = [Int[] for _ in 1:n]
    for (j, b) in enumerate(bases)
        push!(by_base[b], j)
    end
    active = Vector{Vector{Int}}(undef, n)
    for i in 1:n
        idx = Int[]
        for b in upset_indices(Q, i)
            isempty(by_base[b]) && continue
            append!(idx, by_base[b])
        end
        active[i] = idx
    end
    return active
end

# Given coefficient matrix C (ncod x ndom) describing a morphism between direct
# sums of principal downsets, build the corresponding PMorphism.
function _pmorphism_from_downset_coeff(
    E::PModule{K},
    Ep::PModule{K},
    act_dom::Vector{Vector{Int}},
    act_cod::Vector{Vector{Int}},
    C::Matrix{K}
) where {K}
    comps = Vector{Matrix{K}}(undef, nvertices(E.Q))
    for i in 1:nvertices(E.Q)
        comps[i] = C[act_cod[i], act_dom[i]]
    end
    return PMorphism{K}(E, Ep, comps)
end

# NOTE: The authoritative implementation is DerivedFunctors.lift_injective_chainmap.

"""
    rhom_map_second(g, Rsrc, Rtgt; check=true)

Covariant functoriality of RHom in the second argument.

Given:
  * `g : N -> Np` a module morphism
  * `Rsrc = RHomComplex(C, N)`
  * `Rtgt = RHomComplex(C, Np)`

returns a cochain map of total complexes:

    RHom(C, N)  ->  RHom(C, Np)

This is implemented by canonically lifting `g` to a chain map between the
injective resolutions used inside `Rsrc` and `Rtgt`, then postcomposing on
each `Hom(C^p, E^b)` block. The result is assembled as a sparse block matrix.

The lift is deterministic (via solve_particular), so repeated calls are stable.
"""
function rhom_map_second(
    g::PMorphism{K},
    Rsrc::RHomComplex{K},
    Rtgt::RHomComplex{K};
    check::Bool = true,
    cache::Union{Nothing,HomSystemCache}=nothing,
    threads::Bool = (Threads.nthreads() > 1),
) where {K}
    @assert Rsrc.C === Rtgt.C
    @assert g.dom === Rsrc.N
    @assert g.cod === Rtgt.N
    @assert Rsrc.tot.tmin == Rtgt.tot.tmin
    @assert Rsrc.tot.tmax == Rtgt.tot.tmax
    @assert length(Rsrc.resN.d_mor) == length(Rtgt.resN.d_mor)

    # Canonical lift between injective resolutions.
    phis = _lift_injective_chainmap_cached(g, Rsrc.resN, Rtgt.resN; check=check, cache=cache)
    plan = _rhom_map_second_plan(Rsrc, Rtgt, cache)
    maps = Vector{SparseMatrixCSC{K, Int}}(undef, length(plan.degrees))

    if threads && Threads.nthreads() > 1 && (plan.tmax >= plan.tmin)
        Threads.@threads for idx in eachindex(plan.degrees)
            dim_src = plan.dims_src[idx]
            dim_tgt = plan.dims_tgt[idx]
            dplan = plan.degrees[idx]
            I = Int[]; J = Int[]; V = K[]
            for k in eachindex(dplan.ideg)
                Mb = postcompose_matrix_cached(dplan.Htgt[k], dplan.Hsrc[k], phis[dplan.ideg[k] + 1]; cache=cache)
                _append_scaled_triplets!(I, J, V, Mb, dplan.row_off[k], dplan.col_off[k])
            end
            maps[idx] = sparse(I, J, V, dim_tgt, dim_src)
        end
    else
        for idx in eachindex(plan.degrees)
            dim_src = plan.dims_src[idx]
            dim_tgt = plan.dims_tgt[idx]
            dplan = plan.degrees[idx]
            I = Int[]; J = Int[]; V = K[]
            for k in eachindex(dplan.ideg)
                Mb = postcompose_matrix_cached(dplan.Htgt[k], dplan.Hsrc[k], phis[dplan.ideg[k] + 1]; cache=cache)
                _append_scaled_triplets!(I, J, V, Mb, dplan.row_off[k], dplan.col_off[k])
            end
            maps[idx] = sparse(I, J, V, dim_tgt, dim_src)
        end
    end

    return CochainMap(Rsrc.tot, Rtgt.tot, maps; check=check)
end

function rhom_map_second(
    g::PMorphism{K},
    C::ModuleCochainComplex{K},
    Hsrc::FF.FringeModule{K},
    Htgt::FF.FringeModule{K};
    maxlen::Int=3,
    resN=nothing,
    check::Bool=true,
    cache::Union{Nothing,HomSystemCache}=nothing,
    threads::Bool = (Threads.nthreads() > 1),
) where {K}
    Nsrc = _pmodule_from_fringe_cached(Hsrc, cache)
    Ntgt = (Hsrc === Htgt) ? Nsrc : _pmodule_from_fringe_cached(Htgt, cache)
    g = _rebase_pmodule_morphism_cached(g, Nsrc, Ntgt, cache)
    resNsrc = isnothing(resN) ? _injective_resolution_cached(Nsrc, maxlen, cache) : resN
    resNtgt = if Ntgt === Nsrc
        resNsrc
    elseif isnothing(resN)
        _injective_resolution_cached(Ntgt, maxlen, cache)
    else
        resN
    end
    Rsrc = _rhom_complex_cached(C, Nsrc; maxlen=maxlen, resN=resNsrc, cache=cache, threads=threads)
    Rtgt = Ntgt === Nsrc ? Rsrc : _rhom_complex_cached(C, Ntgt; maxlen=maxlen, resN=resNtgt, cache=cache, threads=threads)
    return rhom_map_second(g, Rsrc, Rtgt; check=check, cache=cache, threads=threads)
end

function rhom_map_second(
    gH::FF.FringeModule{K},
    C::ModuleCochainComplex{K},
    Hsrc::FF.FringeModule{K},
    Htgt::FF.FringeModule{K};
    maxlen::Int=3,
    resN=nothing,
    check::Bool=true,
    cache::Union{Nothing,HomSystemCache}=nothing,
    threads::Bool = (Threads.nthreads() > 1),
) where {K}
    g = IndicatorResolutions.pmodule_from_fringe(gH)
    return rhom_map_second(g, C, Hsrc, Htgt; maxlen=maxlen, resN=resN, check=check, cache=cache, threads=threads)
end




struct HyperExtSpace{K}
    R::RHomComplex{K}
    cohom
end

"""
    source_module(H::HyperExtSpace)
    target_module(H::HyperExtSpace)

Return the source module complex and coefficient module underlying a
[`HyperExtSpace`](@ref).

These are the preferred semantic accessors when relating a computed hyper-Ext
space back to its mathematical inputs.
"""
@inline source_module(H::HyperExtSpace) = H.R.C
@inline target_module(H::HyperExtSpace) = H.R.N

"""
    hyperExt(C, N; maxlen=3, ...) -> HyperExtSpace

Compute the graded hyper-Ext object `Ext^*(C, N)` attached to a module complex
`C` and a coefficient module `N`.

The returned [`HyperExtSpace`](@ref) is already the cheap-first surface:
inspect it with [`hyperext_summary`](@ref), [`nonzero_degrees`](@ref), or
[`degree_dimensions`](@ref) before asking for heavier basis or representative
data degree-by-degree.
"""
function hyperExt(C::ModuleCochainComplex{K}, N::PModule{K}; kwargs...) where {K}
    return _hyperext_cached(C, N; kwargs...)
end

function hyperExt(C::ModuleCochainComplex{K}, H::FF.FringeModule{K}; kwargs...) where {K}
    return _hyperext_cached(C, H; kwargs...)
end

dim(H::HyperExtSpace, t::Int) = (t < H.R.tot.tmin || t > H.R.tot.tmax) ? 0 : H.cohom[t - H.R.tot.tmin + 1].dimH

"""
    nonzero_degrees(H::HyperExtSpace)
    degree_dimensions(H::HyperExtSpace)
    total_dimension(H::HyperExtSpace)

Cheap scalar accessors for a [`HyperExtSpace`](@ref).

- `nonzero_degrees` returns the cohomological degrees supporting nonzero
  hyper-Ext.
- `degree_dimensions` returns the degree-to-dimension table on the same support.
- `total_dimension` returns the sum of those dimensions.

These are the preferred notebook/REPL helpers before asking for bases or
representatives in individual degrees.
"""
@inline function nonzero_degrees(H::HyperExtSpace)
    return [t for t in degree_range(H) if dim(H, t) != 0]
end

@inline function degree_dimensions(H::HyperExtSpace)
    return Dict(t => dim(H, t) for t in degree_range(H) if dim(H, t) != 0)
end

@inline total_dimension(H::HyperExtSpace) = sum(values(degree_dimensions(H)))

@inline function describe(H::HyperExtSpace)
    return (
        kind=:hyperext_space,
        field=target_module(H).field,
        source_degree_range=degree_range(source_module(H)),
        degree_range=degree_range(H),
        nonzero_degrees=Tuple(nonzero_degrees(H)),
        degree_dimensions=degree_dimensions(H),
        total_dimension=total_dimension(H),
    )
end

function Base.show(io::IO, H::HyperExtSpace)
    d = describe(H)
    print(io, "HyperExtSpace(nonzero_degrees=", repr(d.nonzero_degrees),
          ", total_dimension=", d.total_dimension, ")")
end

function Base.show(io::IO, ::MIME"text/plain", H::HyperExtSpace)
    d = describe(H)
    print(io, "HyperExtSpace",
          "\n  field: ", d.field,
          "\n  source_degree_range: ", repr(d.source_degree_range),
          "\n  degree_range: ", repr(d.degree_range),
          "\n  nonzero_degrees: ", repr(d.nonzero_degrees),
          "\n  degree_dimensions: ", repr(d.degree_dimensions),
          "\n  total_dimension: ", d.total_dimension)
end

"""
    induced_map_on_cohomology(f, HCdom, HCcod, t)

Convenience wrapper to compute the induced map on cohomology in degree t
from a full cochain map `f::CochainMap`, using cached cohomology data.

- `HCdom` and `HCcod` should be the outputs of `cohomology_data(f.C)` and
  `cohomology_data(f.D)` respectively.
- Returns a dense matrix representing H^t(f) with respect to the stored bases.

This method is used by hyperExt_map_* and hyperTor_map_* to provide a
mathematician-friendly API.
"""
function induced_map_on_cohomology(
    f::CochainMap{K},
    HCdom,
    HCcod,
    t::Int
) where {K}
    # If degree is outside either range, return a correctly-sized zero map.
    if t < f.C.tmin || t > f.C.tmax || t < f.D.tmin || t > f.D.tmax
        dim_dom = (t < f.C.tmin || t > f.C.tmax) ? 0 : HCdom[t - f.C.tmin + 1].dimH
        dim_cod = (t < f.D.tmin || t > f.D.tmax) ? 0 : HCcod[t - f.D.tmin + 1].dimH
        return zeros(K, dim_cod, dim_dom)
    end

    Ht_dom = HCdom[t - f.C.tmin + 1]
    Ht_cod = HCcod[t - f.D.tmin + 1]
    Ft = _map_at(f, t)
    return induced_map_on_cohomology(Ht_dom, Ht_cod, Ft)
end


"""
    hyperExt_map_first(f, Hcod, Hdom; t, check=true)

Given a cochain map `f : C -> D` of module complexes, this returns the induced
map on hyperExt in degree `t`:

    Ext^t(D, N) -> Ext^t(C, N)

Here `Hcod = hyperExt(C,N)` and `Hdom = hyperExt(D,N)`.

This is the mathematically expected contravariance in the first argument.
"""
function hyperExt_map_first(
    f::ModuleCochainMap{K},
    Hcod::HyperExtSpace{K},
    Hdom::HyperExtSpace{K};
    t::Int,
    check::Bool = true,
    cache::Union{Nothing,HomSystemCache}=nothing
) where {K}
    Rmap = rhom_map_first(f, Hcod.R, Hdom.R; check=check, cache=cache)
    return induced_map_on_cohomology(Rmap, Hdom.cohom, Hcod.cohom, t)
end

"""
    hyperExt_map_second(g, Hsrc, Htgt; t, check=true)

Given a module morphism `g : N -> Np`, this returns the induced map on hyperExt
in degree `t`:

    Ext^t(C, N) -> Ext^t(C, Np)

Here `Hsrc = hyperExt(C,N)` and `Htgt = hyperExt(C,Np)`.

This is the mathematically expected covariance in the second argument.
"""
function hyperExt_map_second(
    g::PMorphism{K},
    Hsrc::HyperExtSpace{K},
    Htgt::HyperExtSpace{K};
    t::Int,
    check::Bool = true,
    cache::Union{Nothing,HomSystemCache}=nothing
) where {K}
    Rmap = rhom_map_second(g, Hsrc.R, Htgt.R; check=check, cache=cache)
    return induced_map_on_cohomology(Rmap, Hsrc.cohom, Htgt.cohom, t)
end



# ------------------------------------------------------------

struct DerivedTensorComplex{K}
    Rop::PModule{K}
    C::ModuleCochainComplex{K}
    resR
    DC::DoubleComplex{K}
    tot::CochainComplex{K}
end

"""
    source_module(T::DerivedTensorComplex)
    target_module(T::DerivedTensorComplex)
    underlying_complex(T::DerivedTensorComplex)

Semantic accessors for a [`DerivedTensorComplex`](@ref).

- `source_module(T)` returns the right module supplying the projective
  resolution
- `target_module(T)` returns the module cochain complex
- `underlying_complex(T)` returns the total cochain complex computing the
  derived tensor product
"""
@inline source_module(T::DerivedTensorComplex) = T.Rop
@inline target_module(T::DerivedTensorComplex) = T.C
@inline underlying_complex(T::DerivedTensorComplex) = T.tot

@inline function describe(T::DerivedTensorComplex)
    return (
        kind=:derived_tensor_complex,
        field=T.Rop.field,
        source_total_dim=sum(T.Rop.dims),
        target_degree_range=degree_range(T.C),
        total_degree_range=T.tot.tmin:T.tot.tmax,
        bicomplex_a_range=T.DC.amin:T.DC.amax,
        bicomplex_b_range=T.DC.bmin:T.DC.bmax,
    )
end

function Base.show(io::IO, T::DerivedTensorComplex)
    d = describe(T)
    print(io, "DerivedTensorComplex(total_degrees=", repr(d.total_degree_range),
          ", source_total_dim=", d.source_total_dim, ")")
end

function Base.show(io::IO, ::MIME"text/plain", T::DerivedTensorComplex)
    d = describe(T)
    print(io, "DerivedTensorComplex",
          "\n  field: ", d.field,
          "\n  source_total_dim: ", d.source_total_dim,
          "\n  target_degree_range: ", repr(d.target_degree_range),
          "\n  total_degree_range: ", repr(d.total_degree_range),
          "\n  bicomplex_a_range: ", repr(d.bicomplex_a_range),
          "\n  bicomplex_b_range: ", repr(d.bicomplex_b_range))
end

function DerivedTensorComplex(
    Rop::PModule{K},
    C::ModuleCochainComplex{K};
    maxlen::Int = 3,
    maxdeg::Int = C.tmax,
    threads::Bool = (Threads.nthreads() > 1),
    check::Bool = false,
) where {K}
    resR = projective_resolution(Rop, ResolutionOptions(maxlen=maxlen, minimal=true, check=check))

    # Double-complex bidegrees:
    #   A = -a  where a = 0..maxlen is the projective-resolution (homological) degree
    #   B = p   where p runs over cochain degrees of C
    # Total degree is t = A + B = p - a, so hyperTor_n corresponds to H^{-n}.
    amin, amax = -maxlen, 0
    bmin, bmax = C.tmin, maxdeg

    na = amax - amin + 1
    nb = bmax - bmin + 1

    dims = zeros(Int, na, nb)
    dv = Array{SparseMatrixCSC{K, Int}}(undef, na, nb)
    dh = Array{SparseMatrixCSC{K, Int}}(undef, na, nb)

    total_jobs = na * nb

    build_cell = function (ai::Int, bi::Int)
        A = amin + (ai - 1)
        B = bmin + (bi - 1)

        a = -A  # projective-resolution degree (>= 0)
        p = B   # cochain degree in C

        Mp = _term(C, p)
        gens_a = resR.gens[a + 1]

        offs_dom = _offs_for_gens(Mp, gens_a)
        dims[ai, bi] = offs_dom[end]

        # Vertical differential: dv(A,B): (A,B) -> (A,B+1)
        # Use sign (-1)^a so that total differential is dv + dh.
        if B < bmax
            Mp1 = _term(C, p + 1)
            dC = _diff(C, p)  # PMorphism Mp -> Mp1
            offs_cod = _offs_for_gens(Mp1, gens_a)

            sgn = isodd(a) ? -one(K) : one(K)

            Itrip = Int[]
            Jtrip = Int[]
            Vtrip = K[]
            for (i, u) in enumerate(gens_a)
                # Block: dC.comps[u] : Mp[u] -> Mp1[u]
                _append_scaled_triplets!(
                    Itrip, Jtrip, Vtrip,
                    dC.comps[u],
                    offs_cod[i],
                    offs_dom[i];
                    scale = sgn,
                )
            end
            dv[ai, bi] = sparse(Itrip, Jtrip, Vtrip, offs_cod[end], offs_dom[end])
        else
            # Unused by total_complex at the top boundary (B == bmax), but keep typed.
            dv[ai, bi] = spzeros(K, 0, offs_dom[end])
        end

        # Horizontal differential: dh(A,B): (A,B) -> (A+1,B)
        # This corresponds to a -> a-1 in the projective resolution.
        if A < amax
            gens_am1 = resR.gens[a]        # degree (a-1) generators
            dP = resR.d_mat[a]             # matrix from gens_a -> gens_am1 (rows = gens_am1, cols = gens_a)
            offs_cod = _offs_for_gens(Mp, gens_am1)
            cacheMp = FiniteFringe._get_cover_cache(Mp.Q)

            Itrip = Int[]
            Jtrip = Int[]
            Vtrip = K[]
            pairs = Tuple{Int,Int}[]
            pair_i = Int[]
            pair_j = Int[]
            pair_c = K[]
            for (i, u) in enumerate(gens_a)
                for (j, v) in enumerate(gens_am1)
                    c = dP[j, i]
                    c == 0 && continue
                    push!(pairs, (u, v))
                    push!(pair_i, i)
                    push!(pair_j, j)
                    push!(pair_c, c)
                end
            end

            if !isempty(pairs)
                maps = map_leq_many(Mp, pairs; cache=cacheMp)
                @inbounds for idx in eachindex(pairs)
                    Muv = maps[idx]
                    _append_scaled_triplets!(
                        Itrip, Jtrip, Vtrip,
                        Muv,
                        offs_cod[pair_j[idx]],
                        offs_dom[pair_i[idx]];
                        scale = pair_c[idx],
                    )
                end
            end
            dh[ai, bi] = sparse(Itrip, Jtrip, Vtrip, offs_cod[end], offs_dom[end])
        else
            # Unused by total_complex at the right boundary (A == amax), but keep typed.
            dh[ai, bi] = spzeros(K, 0, offs_dom[end])
        end

        return nothing
    end

    if threads && total_jobs > 1
        Threads.@threads for idx in 1:total_jobs
            bi = Int(div(idx - 1, na)) + 1
            ai = (idx - 1) % na + 1
            build_cell(ai, bi)
        end
    else
        for bi in 1:nb
            for ai in 1:na
                build_cell(ai, bi)
            end
        end
    end

    DC = DoubleComplex(amin, amax, bmin, bmax, dims, dv, dh)
    tot = total_complex(DC)

    return DerivedTensorComplex{K}(Rop, C, resR, DC, tot)
end

function _build_tensor_map_first_plan(Tdom::DerivedTensorComplex{K},
                                      Tcod::DerivedTensorComplex{K}) where {K}
    tmin = min(Tdom.tot.tmin, Tcod.tot.tmin)
    tmax = max(Tdom.tot.tmax, Tcod.tot.tmax)
    degrees = Vector{_TensorMapFirstDegreePlan{K}}(undef, tmax - tmin + 1)
    dims_src = Vector{Int}(undef, length(degrees))
    dims_tgt = Vector{Int}(undef, length(degrees))
    for idx in eachindex(degrees)
        t = tmin + idx - 1
        dims_src[idx] = _cc_dim_at(Tdom.tot, t)
        dims_tgt[idx] = _cc_dim_at(Tcod.tot, t)
        off_dom = _tot_block_offsets(Tdom.DC, t)
        off_cod = _tot_block_offsets(Tcod.DC, t)
        row_off = Int[]; col_off = Int[]; adeg = Int[]
        terms = PModule{K}[]
        dom_gens = Vector{Vector{Int}}()
        cod_gens = Vector{Vector{Int}}()
        dom_offsets = Vector{Vector{Int}}()
        cod_offsets = Vector{Vector{Int}}()
        for (key, src_off) in off_dom
            haskey(off_cod, key) || continue
            tgt_off = off_cod[key]
            A, p = key
            a = -A
            (a < 0 || a > (length(Tdom.resR.Pmods) - 1) || a > (length(Tcod.resR.Pmods) - 1)) && continue
            Mp = _term(Tdom.C, p)
            gens_dom = Tdom.resR.gens[a + 1]
            gens_cod = Tcod.resR.gens[a + 1]
            push!(row_off, tgt_off - 1)
            push!(col_off, src_off - 1)
            push!(adeg, a)
            push!(terms, Mp)
            push!(dom_gens, gens_dom)
            push!(cod_gens, gens_cod)
            push!(dom_offsets, _offs_for_gens(Mp, gens_dom))
            push!(cod_offsets, _offs_for_gens(Mp, gens_cod))
        end
        degrees[idx] = _TensorMapFirstDegreePlan{K}(row_off, col_off, adeg, terms, dom_gens, cod_gens, dom_offsets, cod_offsets)
    end
    return _TensorMapPlan{_TensorMapFirstDegreePlan{K}}(tmin, tmax, dims_src, dims_tgt, degrees)
end

@inline function _tensor_map_first_plan(Tdom::DerivedTensorComplex{K},
                                        Tcod::DerivedTensorComplex{K},
                                        ::Nothing) where {K}
    return _build_tensor_map_first_plan(Tdom, Tcod)
end

@inline function _tensor_map_first_plan(Tdom::DerivedTensorComplex{K},
                                        Tcod::DerivedTensorComplex{K},
                                        cache::HomSystemCache) where {K}
    return _pair_plan_cached!(_DTENSOR_MAP_FIRST_PLAN_CACHES, _DTENSOR_MAP_FIRST_PLAN_LOCK, cache, Tdom, Tcod) do
        _build_tensor_map_first_plan(Tdom, Tcod)
    end::_TensorMapPlan{_TensorMapFirstDegreePlan{K}}
end

@inline _dtensor_map_first_cache_key(f, Tdom, Tcod) =
    (UInt(objectid(f)), UInt(objectid(Tdom)), UInt(objectid(Tcod)))

@inline function _derived_tensor_map_first_cached(
    builder::Function,
    f,
    Tdom::DerivedTensorComplex{K},
    Tcod::DerivedTensorComplex{K},
    ::Nothing,
) where {K}
    return builder()
end

function _derived_tensor_map_first_cached(
    builder::Function,
    f,
    Tdom::DerivedTensorComplex{K},
    Tcod::DerivedTensorComplex{K},
    cache::HomSystemCache,
) where {K}
    return lock(_DTENSOR_MAP_FIRST_RESULT_LOCK) do
        shard = get!(_DTENSOR_MAP_FIRST_RESULT_CACHES, cache) do
            Dict{NTuple{3, UInt}, Any}()
        end
        key = _dtensor_map_first_cache_key(f, Tdom, Tcod)
        cached = get(shard, key, nothing)
        cached === nothing || return cached::CochainMap{K}
        F = builder()
        shard[key] = F
        return F
    end
end

function _build_tensor_map_second_plan(Tsrc::DerivedTensorComplex{K},
                                       Ttgt::DerivedTensorComplex{K}) where {K}
    tmin = min(Tsrc.tot.tmin, Ttgt.tot.tmin)
    tmax = max(Tsrc.tot.tmax, Ttgt.tot.tmax)
    degrees = Vector{_TensorMapSecondDegreePlan{K}}(undef, tmax - tmin + 1)
    dims_src = Vector{Int}(undef, length(degrees))
    dims_tgt = Vector{Int}(undef, length(degrees))
    for idx in eachindex(degrees)
        t = tmin + idx - 1
        dims_src[idx] = _cc_dim_at(Tsrc.tot, t)
        dims_tgt[idx] = _cc_dim_at(Ttgt.tot, t)
        off_src = _tot_block_offsets(Tsrc.DC, t)
        off_tgt = _tot_block_offsets(Ttgt.DC, t)
        block_pdeg = Int[]
        block_u = Int[]
        block_row0 = Int[]
        block_col0 = Int[]
        col_owner = zeros(Int, dims_src[idx])
        col_local = zeros(Int, dims_src[idx])
        block_id = 0
        for (key, src_off) in off_src
            haskey(off_tgt, key) || continue
            tgt_off = off_tgt[key]
            A, p = key
            a = -A
            (a < 0 || a > (length(Tsrc.resR.Pmods) - 1)) && continue
            Mp = _term(Tsrc.C, p)
            Mp1 = _term(Ttgt.C, p)
            gs = Tsrc.resR.gens[a + 1]
            offs_src = _offs_for_gens(Mp, gs)
            offs_tgt = _offs_for_gens(Mp1, gs)
            @inbounds for (i, u) in enumerate(gs)
                block_id += 1
                src_dim = Mp.dims[u]
                col0 = src_off - 1 + offs_src[i]
                push!(block_pdeg, p)
                push!(block_u, u)
                push!(block_row0, tgt_off - 1 + offs_tgt[i])
                push!(block_col0, col0)
                for loc in 1:src_dim
                    col_owner[col0 + loc] = block_id
                    col_local[col0 + loc] = loc
                end
            end
        end
        degrees[idx] = _TensorMapSecondDegreePlan{K}(block_pdeg, block_u, block_row0, block_col0, col_owner, col_local)
    end
    return _TensorMapPlan{_TensorMapSecondDegreePlan{K}}(tmin, tmax, dims_src, dims_tgt, degrees)
end

@inline function _tensor_map_second_plan(Tsrc::DerivedTensorComplex{K},
                                         Ttgt::DerivedTensorComplex{K},
                                         ::Nothing) where {K}
    return _build_tensor_map_second_plan(Tsrc, Ttgt)
end

@inline function _tensor_map_second_plan(Tsrc::DerivedTensorComplex{K},
                                         Ttgt::DerivedTensorComplex{K},
                                         cache::HomSystemCache) where {K}
    if _tensor_map_second_plan_work(Tsrc, Ttgt) < _DTENSOR_MAP_SECOND_CACHE_MIN_WORK[]
        return _build_tensor_map_second_plan(Tsrc, Ttgt)
    end
    return _pair_plan_cached!(_DTENSOR_MAP_SECOND_PLAN_CACHES, _DTENSOR_MAP_SECOND_PLAN_LOCK, cache, Tsrc, Ttgt) do
        _build_tensor_map_second_plan(Tsrc, Ttgt)
    end::_TensorMapPlan{_TensorMapSecondDegreePlan{K}}
end

@inline _dtensor_map_second_cache_key(g, Tsrc, Ttgt) =
    (UInt(objectid(g)), UInt(objectid(Tsrc)), UInt(objectid(Ttgt)))

@inline function _derived_tensor_map_second_cached(
    builder::Function,
    g,
    Tsrc::DerivedTensorComplex{K},
    Ttgt::DerivedTensorComplex{K},
    ::Nothing,
) where {K}
    return builder()
end

function _derived_tensor_map_second_cached(
    builder::Function,
    g,
    Tsrc::DerivedTensorComplex{K},
    Ttgt::DerivedTensorComplex{K},
    cache::HomSystemCache,
) where {K}
    return lock(_DTENSOR_MAP_SECOND_RESULT_LOCK) do
        shard = get!(_DTENSOR_MAP_SECOND_RESULT_CACHES, cache) do
            Dict{NTuple{3, UInt}, Any}()
        end
        key = _dtensor_map_second_cache_key(g, Tsrc, Ttgt)
        cached = get(shard, key, nothing)
        cached === nothing || return cached::CochainMap{K}
        G = builder()
        shard[key] = G
        return G
    end
end



"""
    DerivedTensor(Rop, C; maxlen=3, ...) -> CochainComplex

Return the total cochain complex computing the derived tensor product
`Rop tensor^L C`.

Use this when the chain-level total complex is the real target. If you also
want the underlying bicomplex bookkeeping and projective-resolution data, use
[`DerivedTensorComplex`](@ref). For cheap inspection before touching cochains,
prefer [`derived_tensor_summary`](@ref) on the container form.
"""
DerivedTensor(Rop::PModule{K}, C::ModuleCochainComplex{K}; kwargs...) where {K} =
    DerivedTensorComplex(Rop, C; kwargs...).tot

struct HyperTorSpace{K}
    T::DerivedTensorComplex{K}
    cohom
end

"""
    source_module(H::HyperTorSpace)
    target_module(H::HyperTorSpace)

Return the right module and module complex underlying a
[`HyperTorSpace`](@ref).

These accessors provide the semantic inputs to the hyper-Tor computation
without forcing users into raw field inspection.
"""
@inline source_module(H::HyperTorSpace) = H.T.Rop
@inline target_module(H::HyperTorSpace) = H.T.C

"""
    hyperTor(Rop, C; maxlen=3, ...) -> HyperTorSpace

Compute the graded hyper-Tor object `Tor_*(Rop, C)` attached to a right module
and a module cochain complex.

The returned [`HyperTorSpace`](@ref) is already the cheap-first exploration
surface. Start with [`hypertor_summary`](@ref), [`nonzero_degrees`](@ref), and
[`degree_dimensions`](@ref) before requesting bases or representatives in
specific Tor degrees.
"""
function hyperTor(Rop::PModule{K}, C::ModuleCochainComplex{K}; kwargs...) where {K}
    T = DerivedTensorComplex(Rop,C; kwargs...)
    return HyperTorSpace{K}(T, cohomology_data(T.tot))
end

function hyperTor(H::FF.FringeModule{K}, C::ModuleCochainComplex{K}; kwargs...) where {K}
    return hyperTor(IR.pmodule_from_fringe(H), C; kwargs...)
end

"""
    dim(H::HyperTorSpace, n::Int) -> Int

Dimension of `hyperTor_n`, computed as `H^{-n}` of the total cochain complex.

By convention:
- returns 0 for `n < 0`,
- returns 0 when the required total degree `t = -n` lies outside the stored range.
"""
function dim(H::HyperTorSpace, n::Int)
    if n < 0
        return 0
    end
    # hyperTor_n = H^{-n}(Tot)
    t = -n
    tmin = H.T.tot.tmin
    tmax = H.T.tot.tmax
    if t < tmin || t > tmax
        return 0
    end
    return H.cohom[t - tmin + 1].dimH
end

# ---------------------------------------------------------------------------
# GradedSpaces interface for HyperTorSpace
# ---------------------------------------------------------------------------

"""
    degree_range(H::HyperTorSpace) -> UnitRange{Int}

Tor degrees `n` for which this `HyperTorSpace` stores data.

Internally, the total complex is a cochain complex in degrees `t`, and
`hyperTor_n` corresponds to cohomology degree `t = -n`. This function returns
the induced nonnegative range of `n` values.
"""
function degree_range(H::HyperTorSpace)
    tmin = H.T.tot.tmin
    tmax = H.T.tot.tmax

    # We need t = -n in [tmin, tmax], so n in [-tmax, -tmin].
    n_lo = max(0, -tmax)
    n_hi = -tmin
    if n_lo > n_hi
        return 0:-1  # empty range
    end
    return n_lo:n_hi
end

"""
    nonzero_degrees(H::HyperTorSpace)
    degree_dimensions(H::HyperTorSpace)
    total_dimension(H::HyperTorSpace)

Cheap scalar accessors for a [`HyperTorSpace`](@ref).

- `nonzero_degrees` returns the homological degrees supporting nonzero Tor.
- `degree_dimensions` returns the Tor-dimension table on that support.
- `total_dimension` returns the sum of those dimensions.

These helpers are the preferred first stop in notebooks and REPL sessions
before materializing basis or representative data.
"""
@inline function nonzero_degrees(H::HyperTorSpace)
    return [n for n in degree_range(H) if dim(H, n) != 0]
end

@inline function degree_dimensions(H::HyperTorSpace)
    return Dict(n => dim(H, n) for n in degree_range(H) if dim(H, n) != 0)
end

@inline total_dimension(H::HyperTorSpace) = sum(values(degree_dimensions(H)))

@inline function describe(H::HyperTorSpace)
    return (
        kind=:hypertor_space,
        field=source_module(H).field,
        source_total_dim=sum(source_module(H).dims),
        target_degree_range=degree_range(target_module(H)),
        degree_range=degree_range(H),
        nonzero_degrees=Tuple(nonzero_degrees(H)),
        degree_dimensions=degree_dimensions(H),
        total_dimension=total_dimension(H),
    )
end

function Base.show(io::IO, H::HyperTorSpace)
    d = describe(H)
    print(io, "HyperTorSpace(nonzero_degrees=", repr(d.nonzero_degrees),
          ", total_dimension=", d.total_dimension, ")")
end

function Base.show(io::IO, ::MIME"text/plain", H::HyperTorSpace)
    d = describe(H)
    print(io, "HyperTorSpace",
          "\n  field: ", d.field,
          "\n  source_total_dim: ", d.source_total_dim,
          "\n  target_degree_range: ", repr(d.target_degree_range),
          "\n  degree_range: ", repr(d.degree_range),
          "\n  nonzero_degrees: ", repr(d.nonzero_degrees),
          "\n  degree_dimensions: ", repr(d.degree_dimensions),
          "\n  total_dimension: ", d.total_dimension)
end

"""
    module_complex_summary(C)
    module_map_summary(f)
    module_homotopy_summary(h)
    triangle_summary(T)
    rhom_summary(R)
    hyperext_summary(H)
    derived_tensor_summary(T)
    hypertor_summary(H)

Owner-local summary aliases for the main user-visible objects in
`ModuleComplexes`.

Each helper returns the same structured summary as `describe(...)`, but under a
subsystem-specific name that is easier to discover from notebook and REPL
workflows.

Typical cheap-first workflow
```julia
# After building modules M and N on the same finite poset:
C = ModuleCochainComplex([M, M], [zero_morphism(M, M)]; tmin=0)
module_complex_summary(C)

H0 = cohomology_module(C, 0)
R = RHomComplex(C, N; maxlen=1)
rhom_summary(R)

HX = hyperExt(C, N; maxlen=1)
hyperext_summary(HX)
```
"""
@inline module_complex_summary(C::ModuleCochainComplex) = describe(C)
@inline module_map_summary(f::ModuleCochainMap) = describe(f)
@inline module_homotopy_summary(H::ModuleCochainHomotopy) = describe(H)
@inline triangle_summary(T::ModuleDistinguishedTriangle) = describe(T)
@inline rhom_summary(R::RHomComplex) = describe(R)
@inline hyperext_summary(H::HyperExtSpace) = describe(H)
@inline derived_tensor_summary(T::DerivedTensorComplex) = describe(T)
@inline hypertor_summary(H::HyperTorSpace) = describe(H)

@inline function _module_complex_structurally_equal(C::ModuleCochainComplex{K},
                                                    D::ModuleCochainComplex{K}) where {K}
    degree_range(C) == degree_range(D) || return false
    length(C.terms) == length(D.terms) || return false
    length(C.diffs) == length(D.diffs) || return false
    for i in eachindex(C.terms)
        _pmodule_equal(C.terms[i], D.terms[i]) || return false
    end
    for i in eachindex(C.diffs)
        fi = C.diffs[i]
        gi = D.diffs[i]
        _pmodule_equal(fi.dom, gi.dom) || return false
        _pmodule_equal(fi.cod, gi.cod) || return false
        fi.comps == gi.comps || return false
    end
    return true
end

@inline function _double_complex_total_dims(DC::DoubleComplex)
    tmin = DC.amin + DC.bmin
    tmax = DC.amax + DC.bmax
    dims = zeros(Int, tmax - tmin + 1)
    @inbounds for a in DC.amin:DC.amax
        ai = a - DC.amin + 1
        for b in DC.bmin:DC.bmax
            bi = b - DC.bmin + 1
            dims[a + b - tmin + 1] += DC.dims[ai, bi]
        end
    end
    return tmin, tmax, dims
end

"""
    check_module_complex(C; throw=false) -> NamedTuple

Validate a hand-built [`ModuleCochainComplex`](@ref).

The report checks poset/field consistency across terms, differential
domain/codomain compatibility, and the identity `d^(t+1) * d^t = 0`. Use
`throw=true` when invalid complexes should raise immediately instead of
returning a report.
"""
function check_module_complex(C::ModuleCochainComplex{K}; throw::Bool=false) where {K}
    issues = String[]
    Q = poset(C)
    field = _field_of_complex(C)
    for (i, M) in enumerate(C.terms)
        M.Q === Q || push!(issues, "term $(i) lives over a different poset.")
        M.field == field || push!(issues, "term $(i) uses a different coefficient field.")
    end
    for (i, d) in enumerate(C.diffs)
        _pmodule_equal(d.dom, C.terms[i]) || push!(issues, "differential at degree $(C.tmin + i - 1) has wrong domain.")
        _pmodule_equal(d.cod, C.terms[i + 1]) || push!(issues, "differential at degree $(C.tmin + i - 1) has wrong codomain.")
    end
    d_squared_zero = true
    for i in 1:max(length(C.diffs) - 1, 0)
        left = C.diffs[i]
        right = C.diffs[i + 1]
        for u in 1:nvertices(Q)
            if right.comps[u] * left.comps[u] != zeros(K, size(right.comps[u], 1), size(left.comps[u], 2))
                d_squared_zero = false
                push!(issues, "d^(t+1) * d^t is nonzero at degree $(C.tmin + i - 1), vertex $u.")
                break
            end
        end
        d_squared_zero || break
    end
    valid = isempty(issues)
    throw && !valid && _throw_invalid_module_complex(:check_module_complex, issues)
    return _module_complex_report(:module_complex, valid;
                                  degree_range=degree_range(C),
                                  nterms=length(C.terms),
                                  ndifferentials=length(C.diffs),
                                  d_squared_zero=d_squared_zero,
                                  issues=issues)
end

"""
    check_module_complex_map(f; throw=false) -> NamedTuple

Validate a hand-built [`ModuleCochainMap`](@ref).

The report checks domain/codomain compatibility of the degreewise components and
the chain-map identity `d_D * f = f * d_C`.
"""
function check_module_complex_map(f::ModuleCochainMap{K}; throw::Bool=false) where {K}
    issues = String[]
    Q = poset(f.C)
    poset(f.D) === Q || push!(issues, "domain and codomain complexes live over different posets.")
    for t in f.tmin:f.tmax
        ft = _map(f, t)
        _pmodule_equal(ft.dom, _term(f.C, t)) || push!(issues, "component f^$t has wrong domain.")
        _pmodule_equal(ft.cod, _term(f.D, t)) || push!(issues, "component f^$t has wrong codomain.")
    end
    chain_map = true
    for t in (f.tmin - 1):f.tmax
        dD = _diff(f.D, t)
        dC = _diff(f.C, t)
        ft = _map(f, t)
        ftp = _map(f, t + 1)
        for u in 1:nvertices(Q)
            if dD.comps[u] * ft.comps[u] != ftp.comps[u] * dC.comps[u]
                chain_map = false
                push!(issues, "chain-map identity fails at degree $t, vertex $u.")
                break
            end
        end
        chain_map || break
    end
    valid = isempty(issues)
    throw && !valid && _throw_invalid_module_complex(:check_module_complex_map, issues)
    return _module_complex_report(:module_cochain_map, valid;
                                  degree_range=degree_range(f),
                                  ncomponents=length(f.comps),
                                  chain_map=chain_map,
                                  issues=issues)
end

"""
    check_module_homotopy(H; throw=false) -> NamedTuple

Validate a hand-built [`ModuleCochainHomotopy`](@ref).

The report checks source/target map compatibility, component endpoints, and the
cochain-homotopy identity.
"""
function check_module_homotopy(H::ModuleCochainHomotopy{K}; throw::Bool=false) where {K}
    issues = String[]
    (H.f.C === H.g.C && H.f.D === H.g.D) || push!(issues, "source and target maps must share the same domain and codomain complexes.")
    for t in H.tmin:H.tmax
        ht = _hcomp(H, t)
        _pmodule_equal(ht.dom, _term(H.f.C, t)) || push!(issues, "component h^$t has wrong domain.")
        _pmodule_equal(ht.cod, _term(H.f.D, t - 1)) || push!(issues, "component h^$t has wrong codomain.")
    end
    homotopy_identity = is_cochain_homotopy(H)
    homotopy_identity || push!(issues, "cochain-homotopy identity does not hold.")
    valid = isempty(issues)
    throw && !valid && _throw_invalid_module_complex(:check_module_homotopy, issues)
    return _module_complex_report(:module_cochain_homotopy, valid;
                                  degree_range=degree_range(H),
                                  ncomponents=length(H.comps),
                                  homotopy_identity=homotopy_identity,
                                  issues=issues)
end

"""
    check_module_triangle(T; throw=false) -> NamedTuple

Validate a hand-built [`ModuleDistinguishedTriangle`](@ref).

The report checks that the three structural maps have compatible sources and
targets and that the stored cone/projection data matches the canonical mapping
cone of the triangle morphism.
"""
function check_module_triangle(T::ModuleDistinguishedTriangle{K}; throw::Bool=false) where {K}
    issues = String[]
    objs = triangle_objects(T)
    maps = triangle_maps(T)
    source_target_ok = true
    source(maps.morphism) === objs.source || (push!(issues, "triangle morphism has the wrong source complex."); source_target_ok = false)
    target(maps.morphism) === objs.target || (push!(issues, "triangle morphism has the wrong target complex."); source_target_ok = false)
    source(maps.inclusion) === objs.target || (push!(issues, "triangle inclusion has the wrong source complex."); source_target_ok = false)
    target(maps.inclusion) === objs.cone || (push!(issues, "triangle inclusion has the wrong target complex."); source_target_ok = false)
    source(maps.projection) === objs.cone || (push!(issues, "triangle projection has the wrong source complex."); source_target_ok = false)
    expected_shift = shift(objs.source, 1)
    projection_targets_shift = _module_complex_structurally_equal(target(maps.projection), expected_shift)
    projection_targets_shift || push!(issues, "triangle projection must target the shift C[1].")
    expected_cone = mapping_cone(maps.morphism)
    cone_matches = _module_complex_structurally_equal(objs.cone, expected_cone)
    cone_matches || push!(issues, "stored cone does not match mapping_cone(f).")
    valid = isempty(issues)
    throw && !valid && _throw_invalid_module_complex(:check_module_triangle, issues)
    return _module_complex_report(:module_distinguished_triangle, valid;
                                  source_degree_range=degree_range(objs.source),
                                  target_degree_range=degree_range(objs.target),
                                  cone_degree_range=degree_range(objs.cone),
                                  source_target_ok=source_target_ok,
                                  projection_targets_shift=projection_targets_shift,
                                  cone_matches=cone_matches,
                                  issues=issues)
end

"""
    check_rhom_complex(R; throw=false) -> NamedTuple

Validate a hand-built [`RHomComplex`](@ref).

The report checks block-array shape compatibility with the source complex and
injective resolution, plus total-complex degree and dimension consistency.
"""
function check_rhom_complex(R::RHomComplex{K}; throw::Bool=false) where {K}
    issues = String[]
    expected_shape = (length(R.C.terms), length(R.resN.Emods))
    size(R.homs) == expected_shape || push!(issues, "homs has shape $(size(R.homs)), expected $expected_shape.")
    size(R.DC.dims) == expected_shape || push!(issues, "bicomplex dims have shape $(size(R.DC.dims)), expected $expected_shape.")
    R.DC.amin == 0 || push!(issues, "bicomplex amin must equal 0.")
    R.DC.amax == maxdeg_of_complex(R.C) || push!(issues, "bicomplex amax must equal maxdeg_of_complex(C).")
    R.DC.bmin == 0 || push!(issues, "bicomplex bmin must equal 0.")
    R.DC.bmax == length(R.resN.Emods) - 1 || push!(issues, "bicomplex bmax must match the injective-resolution length.")
    tmin, tmax, dims = _double_complex_total_dims(R.DC)
    R.tot.tmin == tmin || push!(issues, "total complex tmin $(R.tot.tmin) does not match bicomplex total degree minimum $tmin.")
    R.tot.tmax == tmax || push!(issues, "total complex tmax $(R.tot.tmax) does not match bicomplex total degree maximum $tmax.")
    R.tot.dims == dims || push!(issues, "total complex dimensions do not match the bicomplex totals.")
    block_dims_match = true
    for ia in axes(R.homs, 1), ib in axes(R.homs, 2)
        if dim(R.homs[ia, ib]) != R.DC.dims[ia, ib]
            block_dims_match = false
            push!(issues, "Hom block ($(ia), $(ib)) has dimension $(dim(R.homs[ia, ib])) but bicomplex stores $(R.DC.dims[ia, ib]).")
            break
        end
    end
    valid = isempty(issues)
    throw && !valid && _throw_invalid_module_complex(:check_rhom_complex, issues)
    return _module_complex_report(:rhom_complex, valid;
                                  block_shape=size(R.homs),
                                  total_degree_range=R.tot.tmin:R.tot.tmax,
                                  block_dims_match=block_dims_match,
                                  issues=issues)
end

"""
    check_derived_tensor_complex(T; throw=false) -> NamedTuple

Validate a hand-built [`DerivedTensorComplex`](@ref).

The report checks bicomplex shape compatibility with the projective resolution
and source complex, plus total-complex degree and dimension consistency.
"""
function check_derived_tensor_complex(T::DerivedTensorComplex{K}; throw::Bool=false) where {K}
    issues = String[]
    expected_a = -(length(T.resR.gens) - 1):0
    T.DC.amin:T.DC.amax == expected_a || push!(issues, "bicomplex a-range $(T.DC.amin:T.DC.amax) does not match projective-resolution length $(expected_a).")
    T.DC.bmin == T.C.tmin || push!(issues, "bicomplex bmin $(T.DC.bmin) must equal the source complex tmin $(T.C.tmin).")
    T.DC.bmax <= T.C.tmax || push!(issues, "bicomplex bmax $(T.DC.bmax) exceeds the source complex tmax $(T.C.tmax).")
    size(T.DC.dims, 1) == length(T.resR.gens) || push!(issues, "bicomplex first axis has size $(size(T.DC.dims, 1)), expected $(length(T.resR.gens)).")
    size(T.DC.dims, 2) == (T.DC.bmax - T.DC.bmin + 1) || push!(issues, "bicomplex second axis has inconsistent size $(size(T.DC.dims, 2)).")
    tmin, tmax, dims = _double_complex_total_dims(T.DC)
    T.tot.tmin == tmin || push!(issues, "total complex tmin $(T.tot.tmin) does not match bicomplex total degree minimum $tmin.")
    T.tot.tmax == tmax || push!(issues, "total complex tmax $(T.tot.tmax) does not match bicomplex total degree maximum $tmax.")
    T.tot.dims == dims || push!(issues, "total complex dimensions do not match the bicomplex totals.")
    valid = isempty(issues)
    throw && !valid && _throw_invalid_module_complex(:check_derived_tensor_complex, issues)
    return _module_complex_report(:derived_tensor_complex, valid;
                                  bicomplex_a_range=T.DC.amin:T.DC.amax,
                                  bicomplex_b_range=T.DC.bmin:T.DC.bmax,
                                  total_degree_range=T.tot.tmin:T.tot.tmax,
                                  issues=issues)
end

"""
    cycles(H::HyperTorSpace, n::Int) -> Matrix{K}

Columns form a basis of the cycle space in the relevant total cochain degree
`t = -n`. Returns a 0 times 0 matrix if `n` is outside `degree_range(H)`.
"""
function cycles(H::HyperTorSpace, n::Int)
    if n < 0
        return zeros(K, 0, 0)
    end
    t = -n
    tmin = H.T.tot.tmin
    tmax = H.T.tot.tmax
    if t < tmin || t > tmax
        return zeros(K, 0, 0)
    end
    return H.cohom[t - tmin + 1].K
end

"""
    boundaries(H::HyperTorSpace, n::Int) -> Matrix{K}

Columns form a basis of the boundary space in the relevant total cochain degree
`t = -n`. Returns a 0 times 0 matrix if `n` is outside `degree_range(H)`.
"""
function boundaries(H::HyperTorSpace, n::Int)
    if n < 0
        return zeros(K, 0, 0)
    end
    t = -n
    tmin = H.T.tot.tmin
    tmax = H.T.tot.tmax
    if t < tmin || t > tmax
        return zeros(K, 0, 0)
    end
    return H.cohom[t - tmin + 1].B
end

"""
    representative(H::HyperTorSpace, n::Int, coords::AbstractVector{K}) -> Vector{K}

Given coordinates of a class in `hyperTor_n` with respect to the fixed basis,
return a cocycle representative in total cochain degree `t = -n`.

Requirements:
- `n` must lie in `degree_range(H)`.
- `length(coords)` must equal `dim(H,n)`.
"""
function representative(H::HyperTorSpace, n::Int, coords::AbstractVector{K}) where {K}
    if n < 0
        throw(DomainError(n, "Tor degree n must be nonnegative."))
    end
    t = -n
    tmin = H.T.tot.tmin
    tmax = H.T.tot.tmax
    if t < tmin || t > tmax
        throw(DomainError(n, "n must lie in degree_range(H) = $(degree_range(H))."))
    end
    data = H.cohom[t - tmin + 1]
    d = size(data.Hrep, 2)
    if length(coords) != d
        throw(DimensionMismatch("Expected coordinates of length $d, got $(length(coords))."))
    end
    return cohomology_representative(data, coords)
end

"""
    coordinates(H::HyperTorSpace, n::Int, cocycle::AbstractVector{K}) -> Vector{K}

Compute coordinates of a cocycle representative in `hyperTor_n` relative to the
fixed basis, via total cochain degree `t = -n`.
"""
function coordinates(H::HyperTorSpace, n::Int, cocycle::AbstractVector{K}) where {K}
    if n < 0
        throw(DomainError(n, "Tor degree n must be nonnegative."))
    end
    t = -n
    tmin = H.T.tot.tmin
    tmax = H.T.tot.tmax
    if t < tmin || t > tmax
        throw(DomainError(n, "n must lie in degree_range(H) = $(degree_range(H))."))
    end
    data = H.cohom[t - tmin + 1]
    x = cohomology_coordinates(data, cocycle)
    return vec(x)
end

"""
    basis(H::HyperTorSpace, n::Int) -> Vector{Vector{K}}

Return a list of cocycle representatives forming a basis of `hyperTor_n`.
If `n` is outside `degree_range(H)` or `dim(H,n) == 0`, returns an empty vector.
"""
function basis(H::HyperTorSpace{K}, n::Int) where {K}
    if n < 0
        return Vector{Vector{K}}()
    end
    t = -n
    tmin = H.T.tot.tmin
    tmax = H.T.tot.tmax
    if t < tmin || t > tmax
        return Vector{Vector{K}}()
    end
    Hrep = H.cohom[t - tmin + 1].Hrep
    d = size(Hrep, 2)
    B = Vector{Vector{K}}(undef, d)
    for i in 1:d
        B[i] = Hrep[:, i]
    end
    return B
end


# ---------------------------------------------------------------------------
# GradedSpaces interface for HyperExtSpace
# ---------------------------------------------------------------------------

"""
    degree_range(H::HyperExtSpace) -> UnitRange{Int}

Inclusive range of total degrees `t` for which this `HyperExtSpace` stores
cohomology data of the total cochain complex `H.R.tot`.

This is the canonical iterator for graded-space queries:
`dim(H,t)`, `basis(H,t)`, `representative(H,t,coords)`, `coordinates(H,t,z)`,
`cycles(H,t)`, and `boundaries(H,t)`.
"""
degree_range(H::HyperExtSpace) = H.R.tot.tmin:H.R.tot.tmax

"""
    cycles(H::HyperExtSpace, t::Int) -> Matrix{K}

Columns form a basis of the cycle space `ker(d^t)` inside the total cochain group
in degree `t`. Returns a 0 times 0 matrix if `t` is outside `degree_range(H)`.
"""
function cycles(H::HyperExtSpace, t::Int)
    tmin = H.R.tot.tmin
    tmax = H.R.tot.tmax
    if t < tmin || t > tmax
        return zeros(K, 0, 0)
    end
    return H.cohom[t - tmin + 1].K
end

"""
    boundaries(H::HyperExtSpace, t::Int) -> Matrix{K}

Columns form a basis of the boundary space `im(d^(t-1))` inside the total cochain
group in degree `t`. Returns a 0 times 0 matrix if `t` is outside `degree_range(H)`.
"""
function boundaries(H::HyperExtSpace, t::Int)
    tmin = H.R.tot.tmin
    tmax = H.R.tot.tmax
    if t < tmin || t > tmax
        return zeros(K, 0, 0)
    end
    return H.cohom[t - tmin + 1].B
end

"""
    representative(H::HyperExtSpace, t::Int, coords::AbstractVector{K}) -> Vector{K}

Given coordinates of a class in `HyperExt^t` with respect to the fixed basis
chosen by `cohomology_data`, return a cocycle representative in the ambient
total cochain group in degree `t`.

Requirements:
- `t` must lie in `degree_range(H)`.
- `length(coords)` must equal `dim(H,t)`.
"""
function representative(H::HyperExtSpace, t::Int, coords::AbstractVector{K}) where {K}
    tmin = H.R.tot.tmin
    tmax = H.R.tot.tmax
    if t < tmin || t > tmax
        throw(DomainError(t, "t must lie in degree_range(H) = $(tmin):$(tmax)."))
    end
    data = H.cohom[t - tmin + 1]
    d = size(data.Hrep, 2)
    if length(coords) != d
        throw(DimensionMismatch("Expected coordinates of length $d, got $(length(coords))."))
    end
    return cohomology_representative(data, coords)
end

"""
    coordinates(H::HyperExtSpace, t::Int, cocycle::AbstractVector{K}) -> Vector{K}

Compute the coordinate vector of the cohomology class of `cocycle` in `HyperExt^t`,
relative to the fixed basis chosen by `cohomology_data`.

Notes:
- This assumes `cocycle` is in the correct ambient cochain group and is a cocycle.
- If `cocycle` is not a cocycle, the underlying solver may error or return
  coordinates for an implicitly projected class depending on consistency.
"""
function coordinates(H::HyperExtSpace, t::Int, cocycle::AbstractVector{K}) where {K}
    tmin = H.R.tot.tmin
    tmax = H.R.tot.tmax
    if t < tmin || t > tmax
        throw(DomainError(t, "t must lie in degree_range(H) = $(tmin):$(tmax)."))
    end
    data = H.cohom[t - tmin + 1]
    x = cohomology_coordinates(data, cocycle)
    return vec(x)
end

"""
    basis(H::HyperExtSpace, t::Int) -> Vector{Vector{K}}

Return a list of cocycle representatives forming a basis of `HyperExt^t`.
If `t` is outside `degree_range(H)` or `dim(H,t) == 0`, returns an empty vector.

Each basis element is a cochain vector in the ambient total cochain group.
"""
function basis(H::HyperExtSpace{K}, t::Int) where {K}
    tmin = H.R.tot.tmin
    tmax = H.R.tot.tmax
    if t < tmin || t > tmax
        return Vector{Vector{K}}()
    end
    Hrep = H.cohom[t - tmin + 1].Hrep
    d = size(Hrep, 2)
    B = Vector{Vector{K}}(undef, d)
    for i in 1:d
        B[i] = Hrep[:, i]
    end
    return B
end

# Mirror the graded-space interface onto the shared ChainComplexes inspection
# surface so callers using the canonical public `basis` / `representative` /
# `coordinates` entrypoints do not need to know that HyperExt/HyperTor live in
# ModuleComplexes.
ChainComplexes.basis(H::HyperExtSpace{K}, t::Int) where {K} = basis(H, t)
ChainComplexes.coordinates(H::HyperExtSpace{K}, t::Int, cocycle::AbstractVector{K}) where {K} =
    coordinates(H, t, cocycle)
ChainComplexes.basis(H::HyperTorSpace{K}, n::Int) where {K} = basis(H, n)
ChainComplexes.coordinates(H::HyperTorSpace{K}, n::Int, cycle::AbstractVector{K}) where {K} =
    coordinates(H, n, cycle)


# ============================================================
# Chain-level maps for derived tensor (covariant in both vars)
# ============================================================

@inline function _offs_for_gens(M::PModule{K}, gens::Vector{Int}) where {K}
    o = zeros(Int, length(gens) + 1)
    for i in 1:length(gens)
        u = gens[i]
        o[i+1] = o[i] + M.dims[u]
    end
    return o
end

"""
    derived_tensor_map_first(f, Tdom, Tcod; check=true)

Chain-level map on total derived tensor complexes induced by a morphism
in the *first* argument (right module variable):

    f : Rop -> Rop'

This produces a cochain map:

    Tot( Rop  otimes^L C ) -> Tot( Rop' otimes^L C )

Strict functoriality at the chain level uses the deterministic lift
provided by `_lift_pmodule_map_to_projective_resolution_chainmap_coeff`.
"""
function derived_tensor_map_first(
    f::PMorphism{K},
    Tdom::DerivedTensorComplex{K},
    Tcod::DerivedTensorComplex{K};
    check::Bool = true,
    cache=nothing,
    threads::Bool = (Threads.nthreads() > 1),
) where {K}
    f.dom === Tdom.Rop || error("derived_tensor_map_first: f.dom must equal Tdom.Rop")
    f.cod === Tcod.Rop || error("derived_tensor_map_first: f.cod must equal Tcod.Rop")
    Tdom.C === Tcod.C || error("derived_tensor_map_first: complexes must be identical objects for strict functoriality")

    # Lift module map to a chain map between projective resolutions (coeff matrices per degree).
    upto = min(length(Tdom.resR.Pmods), length(Tcod.resR.Pmods)) - 1
    coeffs = cache === nothing ?
        _lift_projective_chainmap_coeff_uncached(
            f,
            Tdom.resR,
            Tcod.resR;
            upto=upto,
        ) :
        _lift_projective_chainmap_coeff_cached(
            f,
            Tdom.resR,
            Tcod.resR;
            upto=upto,
            cache=cache,
        )

    # Optional validation of chain map relation: d_cod[a]*F_a == F_{a-1}*d_dom[a]
    if check
        upto = length(coeffs) - 1
        for a in 1:upto
            lhs = Tcod.resR.d_mat[a] * coeffs[a+1]
            rhs = coeffs[a] * Tdom.resR.d_mat[a]
            lhs == rhs || error("derived_tensor_map_first: lifted coefficients fail chain map check at degree $a")
        end
    end

    return _derived_tensor_map_first_cached(f, Tdom, Tcod, cache) do
        tmin = min(Tdom.tot.tmin, Tcod.tot.tmin)
        tmax = max(Tdom.tot.tmax, Tcod.tot.tmax)
        plan = _tensor_map_first_plan(Tdom, Tcod, cache)
        maps = _assemble_derived_tensor_map_first_maps(coeffs, plan; cache=cache, threads=threads)
        # The lifted coefficient relation already certifies this is a chain map.
        return CochainMap(Tdom.tot, Tcod.tot, maps; tmin=tmin, tmax=tmax, check=false)
    end
end

@inline function _derived_tensor_map_first_upto(
    Tdom::DerivedTensorComplex,
    t::Int,
)
    off_dom = _tot_block_offsets(Tdom.DC, t)
    isempty(off_dom) && return -1
    upto = -1
    for (key, _) in off_dom
        A, _ = key
        a = -A
        a > upto && (upto = a)
    end
    return upto
end

function _assemble_derived_tensor_map_first_maps(
    coeffs::Vector{<:AbstractMatrix{K}},
    plan::_TensorMapPlan{_TensorMapFirstDegreePlan{K}};
    cache=nothing,
    threads::Bool=(Threads.nthreads() > 1),
) where {K}
    maps = Vector{SparseMatrixCSC{K,Int}}(undef, length(plan.degrees))
    if threads && Threads.nthreads() > 1 && !isempty(plan.degrees)
        Threads.@threads for idx in eachindex(plan.degrees)
            dim_dom = plan.dims_src[idx]
            dim_cod = plan.dims_tgt[idx]
            dplan = plan.degrees[idx]
            I = Int[]; J = Int[]; V = K[]
            for k in eachindex(dplan.adeg)
                block = _tensor_map_on_tor_chains_from_projective_coeff(
                    dplan.terms[k],
                    dplan.dom_gens[k],
                    dplan.cod_gens[k],
                    dplan.dom_offsets[k],
                    dplan.cod_offsets[k],
                    coeffs[dplan.adeg[k] + 1];
                    cache=cache,
                )
                _append_scaled_triplets!(I, J, V, block, dplan.row_off[k], dplan.col_off[k])
            end
            maps[idx] = sparse(I, J, V, dim_cod, dim_dom)
        end
    else
        for idx in eachindex(plan.degrees)
            dim_dom = plan.dims_src[idx]
            dim_cod = plan.dims_tgt[idx]
            dplan = plan.degrees[idx]
            I = Int[]; J = Int[]; V = K[]
            for k in eachindex(dplan.adeg)
                block = _tensor_map_on_tor_chains_from_projective_coeff(
                    dplan.terms[k],
                    dplan.dom_gens[k],
                    dplan.cod_gens[k],
                    dplan.dom_offsets[k],
                    dplan.cod_offsets[k],
                    coeffs[dplan.adeg[k] + 1];
                    cache=cache,
                )
                _append_scaled_triplets!(I, J, V, block, dplan.row_off[k], dplan.col_off[k])
            end
            maps[idx] = sparse(I, J, V, dim_cod, dim_dom)
        end
    end
    return maps
end

function _derived_tensor_map_first_degree(
    f::PMorphism{K},
    Tdom::DerivedTensorComplex{K},
    Tcod::DerivedTensorComplex{K},
    t::Int;
    check::Bool = true,
    cache=nothing,
    coeffs = nothing,
) where {K}
    f.dom === Tdom.Rop || error("_derived_tensor_map_first_degree: f.dom must equal Tdom.Rop")
    f.cod === Tcod.Rop || error("_derived_tensor_map_first_degree: f.cod must equal Tcod.Rop")
    Tdom.C === Tcod.C || error("_derived_tensor_map_first_degree: complexes must be identical objects for strict functoriality")

    plan = _tensor_map_first_plan(Tdom, Tcod, cache)
    if t < plan.tmin || t > plan.tmax
        return spzeros(K, 0, 0)
    end
    idx = t - plan.tmin + 1
    dim_dom = plan.dims_src[idx]
    dim_cod = plan.dims_tgt[idx]
    if dim_dom == 0 || dim_cod == 0
        return spzeros(K, dim_cod, dim_dom)
    end

    dplan = plan.degrees[idx]
    isempty(dplan.adeg) && return spzeros(K, dim_cod, dim_dom)

    upto = min(
        maximum(dplan.adeg; init=-1),
        length(Tdom.resR.Pmods) - 1,
        length(Tcod.resR.Pmods) - 1,
    )
    coeffs_use = coeffs === nothing ?
        (cache === nothing ?
            _lift_projective_chainmap_coeff_uncached(
                f,
                Tdom.resR,
                Tcod.resR;
                upto=upto,
            ) :
            _lift_projective_chainmap_coeff_cached(
                f,
                Tdom.resR,
                Tcod.resR;
                upto=upto,
                cache=cache,
            )) :
        coeffs

    if check
        for a in 1:upto
            lhs = Tcod.resR.d_mat[a] * coeffs_use[a + 1]
            rhs = coeffs_use[a] * Tdom.resR.d_mat[a]
            lhs == rhs || error("_derived_tensor_map_first_degree: lifted coefficients fail chain map check at degree $a")
        end
    end

    I = Int[]
    J = Int[]
    V = K[]
    for k in eachindex(dplan.adeg)
        dplan.adeg[k] > upto && continue
        block = _tensor_map_on_tor_chains_from_projective_coeff(
            dplan.terms[k],
            dplan.dom_gens[k],
            dplan.cod_gens[k],
            dplan.dom_offsets[k],
            dplan.cod_offsets[k],
            coeffs_use[dplan.adeg[k] + 1];
            cache=cache,
        )
        _append_scaled_triplets!(I, J, V, block, dplan.row_off[k], dplan.col_off[k])
    end
    return sparse(I, J, V, dim_cod, dim_dom)
end


"""
    derived_tensor_map_second(g, Tsrc, Ttgt; check=true)

Chain-level map on total derived tensor complexes induced by a cochain map
in the *second* argument (complex variable):

    g : C -> C'

with fixed right module Rop.

Produces:

    Tot(Rop otimes^L C) -> Tot(Rop otimes^L C')

Strict chain-level functoriality requires that Tsrc and Ttgt use the same
projective resolution object (or at least same gens ordering). Here we
require identical gens lists for safety.
"""
function derived_tensor_map_second(
    g::ModuleCochainMap{K},
    Tsrc::DerivedTensorComplex{K},
    Ttgt::DerivedTensorComplex{K};
    check::Bool = true,
    cache=nothing,
    threads::Bool = (Threads.nthreads() > 1),
) where {K}
    g.C === Tsrc.C || error("derived_tensor_map_second: g.C must equal Tsrc.C")
    g.D === Ttgt.C || error("derived_tensor_map_second: g.D must equal Ttgt.C")
    Tsrc.Rop === Ttgt.Rop || error("derived_tensor_map_second: right modules must match")
    Tsrc.resR.gens == Ttgt.resR.gens || error("derived_tensor_map_second: resolutions must have identical gens ordering")
    return _derived_tensor_map_second_cached(g, Tsrc, Ttgt, cache) do
        tmin = min(Tsrc.tot.tmin, Ttgt.tot.tmin)
        tmax = max(Tsrc.tot.tmax, Ttgt.tot.tmax)
        maps = Vector{SparseMatrixCSC{K,Int}}(undef, tmax - tmin + 1)
        plan = _tensor_map_second_plan(Tsrc, Ttgt, cache)

        if threads && Threads.nthreads() > 1 && (tmax >= tmin)
            Threads.@threads for idx in 1:(tmax - tmin + 1)
                maps[idx] = _assemble_tensor_map_second_degree(g, plan.degrees[idx], plan.dims_tgt[idx], plan.dims_src[idx])
            end
        else
            for idx in 1:(tmax - tmin + 1)
                maps[idx] = _assemble_tensor_map_second_degree(g, plan.degrees[idx], plan.dims_tgt[idx], plan.dims_src[idx])
            end
        end

        return CochainMap(Tsrc.tot, Ttgt.tot, maps; tmin=tmin, tmax=tmax, check=check)
    end
end


# ============================================================
# Induced maps on hyperTor_n (mathematician-friendly API)
# ============================================================

"""
    hyperTor_map_first(f, Hdom, Hcod; n, check=true)

Given a morphism of right modules f : Rop -> Rop', return the induced map:

    Tor_n(Rop, C) -> Tor_n(Rop', C)

Here `Hdom = hyperTor(Rop, C)` and `Hcod = hyperTor(Rop', C)`.

Convention: Tor_n = H^{-n}(Tot).
"""
function hyperTor_map_first(
    f::PMorphism{K},
    Hdom::HyperTorSpace{K},
    Hcod::HyperTorSpace{K};
    n::Int,
    check::Bool = true,
    cache=nothing,
) where {K}
    t = -n
    if t < Hdom.T.tot.tmin || t > Hdom.T.tot.tmax || t < Hcod.T.tot.tmin || t > Hcod.T.tot.tmax
        dim_dom = (t < Hdom.T.tot.tmin || t > Hdom.T.tot.tmax) ? 0 : Hdom.cohom[t - Hdom.T.tot.tmin + 1].dimH
        dim_cod = (t < Hcod.T.tot.tmin || t > Hcod.T.tot.tmax) ? 0 : Hcod.cohom[t - Hcod.T.tot.tmin + 1].dimH
        return zeros(K, dim_cod, dim_dom)
    end
    Ft = _derived_tensor_map_first_degree(f, Hdom.T, Hcod.T, t; check=check, cache=cache)
    Ht_dom = Hdom.cohom[t - Hdom.T.tot.tmin + 1]
    Ht_cod = Hcod.cohom[t - Hcod.T.tot.tmin + 1]
    return induced_map_on_cohomology(Ht_dom, Ht_cod, Ft)
end


"""
    hyperTor_map_second(g, Hsrc, Htgt; n, check=true)

Given a cochain map g : C -> C', return the induced map:

    Tor_n(Rop, C) -> Tor_n(Rop, C')

Here `Hsrc = hyperTor(Rop, C)` and `Htgt = hyperTor(Rop, C')`.

Convention: Tor_n = H^{-n}(Tot).
"""
function hyperTor_map_second(
    g::ModuleCochainMap{K},
    Hsrc::HyperTorSpace{K},
    Htgt::HyperTorSpace{K};
    n::Int,
    check::Bool = true
) where {K}
    Tmap = derived_tensor_map_second(g, Hsrc.T, Htgt.T; check=check)
    return induced_map_on_cohomology(Tmap, Hsrc.cohom, Htgt.cohom, -n)
end


end # module
