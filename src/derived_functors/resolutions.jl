# resolutions.jl -- projective/injective resolution builders

"""
Resolutions: projective/injective resolution builders and diagnostics.

This submodule should own:
- resolution constructors (projective, injective, minimal variants)
- minimality diagnostics and reports
- any caching structures tied to resolution construction

It is expected to be the main consumer of IndicatorResolutions machinery.
"""
module Resolutions

    using LinearAlgebra
    using SparseArrays
    import Base.Threads

    using ...CoreModules: AbstractCoeffField, RealField, ResolutionCache, ResolutionKey2, _resolution_key2,
                          field_from_eltype, coeff_type,
                          ProjectiveResolutionPayload, InjectiveResolutionPayload
    using ...Options: ResolutionOptions
    using ...Modules: PModule, PMorphism, _get_cover_cache
    using ...FiniteFringe: AbstractPoset, FinitePoset, FringeModule, Upset, cover_edges, is_subset,
                           leq, nvertices, poset_equal, upset_indices, downset_indices
    using ...AbelianCategories: kernel_with_inclusion, _KernelIncrementalEntry, _CokernelIncrementalEntry,
                                 is_zero_morphism
    using ...IndicatorResolutions: projective_cover, pmodule_from_fringe
    using  ...IndicatorResolutions: _injective_hull, _cokernel_module, _nonzero_dim_vertices,
                                    _indicator_new_array_memo, _new_resolution_workspace
    import ...IndicatorResolutions: resolution_length
    using ...FiniteFringe: AbstractPoset
    using ...FieldLinAlg: _SparseRREFAugmented, SparseRow, _sparse_rref_push_augmented!, rref


    import ..Utils
    import ..Utils
    import ..Utils
    import ..Utils: compose
    import ..DerivedFunctors: resolution_terms, resolution_differentials, augmentation_map,
                              coaugmentation_map, source_module, resolution_summary,
                              check_projective_resolution, check_injective_resolution,
                              _derived_validation_report, _throw_invalid_derived_functor

    # ----------------------------
    # Poset comparison utility
    # ----------------------------

    function _same_poset(Q1::AbstractPoset, Q2::AbstractPoset)::Bool
        return poset_equal(Q1, Q2)
    end

    @inline function _reset_indicator_memo!(memo::AbstractVector)
        fill!(memo, nothing)
        return memo
    end

    # Internal A/B knobs for cache-only microbenchmarks. Keep enabled in normal runs.
    const PROJECTIVE_PRIMARY_CACHE_ENABLED = Ref(true)
    const INJECTIVE_PRIMARY_CACHE_ENABLED = Ref(true)

    @inline _resolution_cache_shard_index(dicts) =
        min(length(dicts), max(1, Threads.threadid()))

    @inline function _projective_primary_dict(cache::ResolutionCache, ::Type{R}) where {R}
        PROJECTIVE_PRIMARY_CACHE_ENABLED[] || return nothing
        cache.projective_primary_type === R || return nothing
        primary = cache.projective_primary
        primary === nothing && return nothing
        isempty(primary) && return nothing
        return primary::Dict{ResolutionKey2,R}
    end

    @inline function _projective_primary_shard(cache::ResolutionCache, ::Type{R}) where {R}
        PROJECTIVE_PRIMARY_CACHE_ENABLED[] || return nothing
        cache.projective_primary_type === R || return nothing
        shard = cache.projective_primary_shards[_resolution_cache_shard_index(cache.projective_primary_shards)]
        shard === nothing && return nothing
        return shard::Dict{ResolutionKey2,R}
    end

    @inline function _reset_projective_promotion_state!(cache::ResolutionCache)
        cache.projective_promotion_type = nothing
        cache.projective_promotion_hits = 0
        return nothing
    end

    @inline function _note_projective_fallback_hit_locked!(cache::ResolutionCache, ::Type{R}) where {R}
        PROJECTIVE_PRIMARY_CACHE_ENABLED[] || return false
        if cache.projective_primary_type !== nothing &&
           cache.projective_primary !== nothing &&
           !isempty(cache.projective_primary) &&
           cache.projective_primary_type !== R
            return false
        end
        if cache.projective_promotion_type === R
            cache.projective_promotion_hits += 1
        else
            cache.projective_promotion_type = R
            cache.projective_promotion_hits = 1
        end
        return cache.projective_promotion_hits >= 2
    end

    @inline function _promote_projective_primary_locked!(cache::ResolutionCache, ::Type{R}) where {R}
        PROJECTIVE_PRIMARY_CACHE_ENABLED[] || return nothing
        if cache.projective_primary_type === R
            primary = cache.projective_primary
            primary === nothing && return nothing
            _reset_projective_promotion_state!(cache)
            return primary::Dict{ResolutionKey2,R}
        end
        isempty(cache.projective) && return nothing
        if cache.projective_primary_type !== nothing && !isempty(cache.projective_primary)
            return nothing
        end
        primary = Dict{ResolutionKey2,R}()
        for (k, payload) in cache.projective
            v = payload.value
            if !(v isa R)
                _reset_projective_promotion_state!(cache)
                return nothing
            end
            primary[k] = v
        end
        cache.projective_primary_type = R
        cache.projective_primary = primary
        fill!(cache.projective_primary_shards, nothing)
        empty!(cache.projective)
        for shard in cache.projective_shards
            empty!(shard)
        end
        _reset_projective_promotion_state!(cache)
        return primary
    end

    @inline function _ensure_projective_primary_shard_locked!(cache::ResolutionCache, ::Type{R}) where {R}
        length(cache.projective_primary_shards) == 1 && return nothing
        idx = _resolution_cache_shard_index(cache.projective_primary_shards)
        shard = cache.projective_primary_shards[idx]
        if shard === nothing
            shard = Dict{ResolutionKey2,R}()
            cache.projective_primary_shards[idx] = shard
        end
        return shard::Dict{ResolutionKey2,R}
    end

    @inline function _cache_projective_get(cache::ResolutionCache, key::ResolutionKey2, ::Type{R}) where {R}
        primary = _projective_primary_dict(cache, R)
        if primary !== nothing
            if length(cache.projective_primary_shards) == 1
                return get(primary, key, nothing)
            end
            shard = _projective_primary_shard(cache, R)
            shard === nothing || begin
                v = get(shard, key, nothing)
                v === nothing || return v
            end
            Base.lock(cache.lock)
            try
                v = get(primary, key, nothing)
                if v !== nothing
                    shard === nothing || (shard[key] = v)
                end
                return v
            finally
                Base.unlock(cache.lock)
            end
        end

        if length(cache.projective_shards) == 1
            Base.lock(cache.lock)
            try
                v = get(cache.projective, key, nothing)
                v === nothing && return nothing
                if _note_projective_fallback_hit_locked!(cache, R)
                    primary = _promote_projective_primary_locked!(cache, R)
                    if primary !== nothing
                        return get(primary, key, nothing)
                    end
                end
                return v.value::R
            finally
                Base.unlock(cache.lock)
            end
        end
        shard = cache.projective_shards[_resolution_cache_shard_index(cache.projective_shards)]
        v = get(shard, key, nothing)
        if v !== nothing && !PROJECTIVE_PRIMARY_CACHE_ENABLED[]
            return (v.value::R)
        end
        Base.lock(cache.lock)
        try
            v === nothing && (v = get(cache.projective, key, nothing))
            if v !== nothing && _note_projective_fallback_hit_locked!(cache, R)
                primary = _promote_projective_primary_locked!(cache, R)
                if primary !== nothing
                    vv = get(primary, key, nothing)
                    if vv !== nothing
                        shard_primary = _projective_primary_shard(cache, R)
                        shard_primary === nothing || (shard_primary[key] = vv)
                    end
                    return vv
                end
            end
        finally
            Base.unlock(cache.lock)
        end
        v === nothing || begin
            vv = v.value::R
            shard[key] = v
            return vv
        end
        return nothing
    end

    @inline function _cache_projective_store!(cache::ResolutionCache, key::ResolutionKey2, val::R) where {R}
        primary = _projective_primary_dict(cache, R)
        if primary !== nothing
            if length(cache.projective_primary_shards) == 1
                extant = get(primary, key, nothing)
                extant === nothing || return extant
                primary[key] = val
                return val
            end
            shard = _projective_primary_shard(cache, R)
            shard === nothing || begin
                extant = get(shard, key, nothing)
                extant === nothing || return extant
            end
            Base.lock(cache.lock)
            try
                extant = get(primary, key, nothing)
                extant === nothing || return extant
                primary[key] = val
                if shard === nothing
                    shard = _ensure_projective_primary_shard_locked!(cache, R)
                end
                shard === nothing || (shard[key] = val)
                return val
            finally
                Base.unlock(cache.lock)
            end
        end

        if length(cache.projective_shards) == 1
            extant = get(cache.projective, key, nothing)
            extant === nothing || return (extant.value::R)
            payload = ProjectiveResolutionPayload(val)
            _reset_projective_promotion_state!(cache)
            cache.projective[key] = payload
            return val
        end
        shard = cache.projective_shards[_resolution_cache_shard_index(cache.projective_shards)]
        existing = get(shard, key, nothing)
        existing === nothing || return (existing.value::R)
        payload = ProjectiveResolutionPayload(val)
        _reset_projective_promotion_state!(cache)
        shard[key] = payload
        Base.lock(cache.lock)
        out = get(cache.projective, key, nothing)
        if out === nothing
            cache.projective[key] = payload
            out = payload
        end
        Base.unlock(cache.lock)
        outR = out.value::R
        shard[key] = out
        return outR
    end

    @inline function _injective_primary_dict(cache::ResolutionCache, ::Type{R}) where {R}
        INJECTIVE_PRIMARY_CACHE_ENABLED[] || return nothing
        cache.injective_primary_type === R || return nothing
        primary = cache.injective_primary
        primary === nothing && return nothing
        isempty(primary) && return nothing
        return primary::Dict{ResolutionKey2,R}
    end

    @inline function _injective_primary_shard(cache::ResolutionCache, ::Type{R}) where {R}
        INJECTIVE_PRIMARY_CACHE_ENABLED[] || return nothing
        cache.injective_primary_type === R || return nothing
        shard = cache.injective_primary_shards[_resolution_cache_shard_index(cache.injective_primary_shards)]
        shard === nothing && return nothing
        return shard::Dict{ResolutionKey2,R}
    end

    @inline function _reset_injective_promotion_state!(cache::ResolutionCache)
        cache.injective_promotion_type = nothing
        cache.injective_promotion_hits = 0
        return nothing
    end

    @inline function _note_injective_fallback_hit_locked!(cache::ResolutionCache, ::Type{R}) where {R}
        INJECTIVE_PRIMARY_CACHE_ENABLED[] || return false
        if cache.injective_primary_type !== nothing &&
           cache.injective_primary !== nothing &&
           !isempty(cache.injective_primary) &&
           cache.injective_primary_type !== R
            return false
        end
        if cache.injective_promotion_type === R
            cache.injective_promotion_hits += 1
        else
            cache.injective_promotion_type = R
            cache.injective_promotion_hits = 1
        end
        return cache.injective_promotion_hits >= 2
    end

    @inline function _promote_injective_primary_locked!(cache::ResolutionCache, ::Type{R}) where {R}
        INJECTIVE_PRIMARY_CACHE_ENABLED[] || return nothing
        if cache.injective_primary_type === R
            primary = cache.injective_primary
            primary === nothing && return nothing
            _reset_injective_promotion_state!(cache)
            return primary::Dict{ResolutionKey2,R}
        end
        isempty(cache.injective) && return nothing
        if cache.injective_primary_type !== nothing && !isempty(cache.injective_primary)
            return nothing
        end
        primary = Dict{ResolutionKey2,R}()
        for (k, payload) in cache.injective
            v = payload.value
            if !(v isa R)
                _reset_injective_promotion_state!(cache)
                return nothing
            end
            primary[k] = v
        end
        cache.injective_primary_type = R
        cache.injective_primary = primary
        fill!(cache.injective_primary_shards, nothing)
        empty!(cache.injective)
        for shard in cache.injective_shards
            empty!(shard)
        end
        _reset_injective_promotion_state!(cache)
        return primary
    end

    @inline function _ensure_injective_primary_shard_locked!(cache::ResolutionCache, ::Type{R}) where {R}
        length(cache.injective_primary_shards) == 1 && return nothing
        idx = _resolution_cache_shard_index(cache.injective_primary_shards)
        shard = cache.injective_primary_shards[idx]
        if shard === nothing
            shard = Dict{ResolutionKey2,R}()
            cache.injective_primary_shards[idx] = shard
        end
        return shard::Dict{ResolutionKey2,R}
    end

    @inline function _cache_injective_get(cache::ResolutionCache, key::ResolutionKey2, ::Type{R}) where {R}
        primary = _injective_primary_dict(cache, R)
        if primary !== nothing
            if length(cache.injective_primary_shards) == 1
                return get(primary, key, nothing)
            end
            shard = _injective_primary_shard(cache, R)
            shard === nothing || begin
                v = get(shard, key, nothing)
                v === nothing || return v
            end
            Base.lock(cache.lock)
            try
                v = get(primary, key, nothing)
                if v !== nothing
                    shard === nothing || (shard[key] = v)
                end
                return v
            finally
                Base.unlock(cache.lock)
            end
        end

        if length(cache.injective_shards) == 1
            Base.lock(cache.lock)
            try
                v = get(cache.injective, key, nothing)
                v === nothing && return nothing
                if _note_injective_fallback_hit_locked!(cache, R)
                    primary = _promote_injective_primary_locked!(cache, R)
                    if primary !== nothing
                        return get(primary, key, nothing)
                    end
                end
                return v.value::R
            finally
                Base.unlock(cache.lock)
            end
        end
        shard = cache.injective_shards[_resolution_cache_shard_index(cache.injective_shards)]
        v = get(shard, key, nothing)
        if v !== nothing && !INJECTIVE_PRIMARY_CACHE_ENABLED[]
            return (v.value::R)
        end
        Base.lock(cache.lock)
        try
            v === nothing && (v = get(cache.injective, key, nothing))
            if v !== nothing && _note_injective_fallback_hit_locked!(cache, R)
                primary = _promote_injective_primary_locked!(cache, R)
                if primary !== nothing
                    vv = get(primary, key, nothing)
                    if vv !== nothing
                        shard_primary = _injective_primary_shard(cache, R)
                        shard_primary === nothing || (shard_primary[key] = vv)
                    end
                    return vv
                end
            end
        finally
            Base.unlock(cache.lock)
        end
        v === nothing || begin
            vv = v.value::R
            shard[key] = v
            return vv
        end
        return nothing
    end

    @inline function _cache_injective_store!(cache::ResolutionCache, key::ResolutionKey2, val::R) where {R}
        primary = _injective_primary_dict(cache, R)
        if primary !== nothing
            if length(cache.injective_primary_shards) == 1
                extant = get(primary, key, nothing)
                extant === nothing || return extant
                primary[key] = val
                return val
            end
            shard = _injective_primary_shard(cache, R)
            shard === nothing || begin
                extant = get(shard, key, nothing)
                extant === nothing || return extant
            end
            Base.lock(cache.lock)
            try
                extant = get(primary, key, nothing)
                extant === nothing || return extant
                primary[key] = val
                if shard === nothing
                    shard = _ensure_injective_primary_shard_locked!(cache, R)
                end
                shard === nothing || (shard[key] = val)
                return val
            finally
                Base.unlock(cache.lock)
            end
        end

        if length(cache.injective_shards) == 1
            extant = get(cache.injective, key, nothing)
            extant === nothing || return (extant.value::R)
            payload = InjectiveResolutionPayload(val)
            _reset_injective_promotion_state!(cache)
            cache.injective[key] = payload
            return val
        end
        shard = cache.injective_shards[_resolution_cache_shard_index(cache.injective_shards)]
        existing = get(shard, key, nothing)
        existing === nothing || return (existing.value::R)
        payload = InjectiveResolutionPayload(val)
        _reset_injective_promotion_state!(cache)
        shard[key] = payload
        Base.lock(cache.lock)
        out = get(cache.injective, key, nothing)
        if out === nothing
            cache.injective[key] = payload
            out = payload
        end
        Base.unlock(cache.lock)
        outR = out.value::R
        shard[key] = out
        return outR
    end

    # ----------------------------
    # Projective resolution (explicit summands + coefficient matrices)
    # ----------------------------

    struct ProjectiveResolution{K}
        M::PModule{K}
        # NOTE: PModule{K} is a UnionAll type (PModule{K,F,MatT} where
        # F<:AbstractCoeffField and MatT<:AbstractMatrix{K}).
        # Vector{PModule{K}} would reject natural vectors like Vector{PModule{K,field,Matrix{K}}}.
        # Using Vector{<:PModule{K}} makes those natural vectors the canonical API.
        Pmods::Vector{<:PModule{K}}                  # P_0 .. P_L
        gens::Vector{Vector{Int}}                    # base vertex per summand (same order as summands)
        d_mor::Vector{<:PMorphism{K}}                # d_a : P_a -> P_{a-1}, a=1..L
        d_mat::Vector{SparseMatrixCSC{K, Int}}       # coefficient matrices (rows cod summands, cols dom summands)
        aug::PMorphism{K}                            # P_0 -> M
    end

    @inline resolution_terms(res::ProjectiveResolution) = copy(res.Pmods)
    @inline resolution_differentials(res::ProjectiveResolution) = copy(res.d_mor)
    @inline augmentation_map(res::ProjectiveResolution) = res.aug
    @inline source_module(res::ProjectiveResolution) = res.M
    @inline resolution_length(res::ProjectiveResolution) = length(res.d_mor)

    """
        resolution_summary(res::ProjectiveResolution) -> NamedTuple

    Cheap-first summary of a `ProjectiveResolution`.

    This reports the source field, ambient poset size, degree range, total size
    of each projective term, and the number of stored generators in each
    degree, without materializing any new lifts or comparison data.

    Start with this or `resolution_length(res)` in notebook and REPL work. Ask
    for `resolution_terms(res)`, `resolution_differentials(res)`, or explicit
    comparison maps only when you need the full chain-level data.
    """
    @inline function resolution_summary(res::ProjectiveResolution)
        return (
            kind=:projective_resolution,
            side=:projective,
            field=res.M.field,
            nvertices=nvertices(res.M.Q),
            degree_range=0:resolution_length(res),
            resolution_length=resolution_length(res),
            term_dimensions=Tuple(sum(P.dims) for P in res.Pmods),
            generator_counts=Tuple(length(g) for g in res.gens),
        )
    end


    function _flatten_gens_at(gens_at)
        out = Int[]
        for u in 1:length(gens_at)
            for tup in gens_at[u]
                push!(out, tup[1])
            end
        end
        return out
    end

    struct _PackedActiveIndexPlan
        data::Vector{Int}
        ptr::Vector{Int}
        base_pos::Vector{Int}
        base_data::Vector{Int}
        base_ptr::Vector{Int}
    end

    const _ACTIVE_INDEX_PLAN_LOCK = ReentrantLock()
    const _ACTIVE_INDEX_PLAN_CACHE = Dict{Tuple{UInt,UInt64,Bool},_PackedActiveIndexPlan}()
    const _RESOLUTIONS_USE_INJECTIVE_DOWNSET_SYSTEM_CACHE = Ref(true)
    const _RESOLUTIONS_USE_INJECTIVE_DOWNSET_SOLVE_PLAN_CACHE = Ref(true)
    const _RESOLUTIONS_USE_PROJECTIVE_LIFT_STRUCTURE_CACHE = Ref(true)

    @inline function _downset_dense_identity(::Type{K}, n::Int) where {K}
        I = zeros(K, n, n)
        @inbounds for i in 1:n
            I[i, i] = one(K)
        end
        return I
    end

    abstract type _AbstractDownsetSolvePlan{K} end

    struct _ExactDownsetSolvePlan{K,F<:AbstractCoeffField} <: _AbstractDownsetSolvePlan{K}
        field::F
        transform::Matrix{K}
        pivcols::Vector{Int}
        rank::Int
        nvars::Int
    end

    struct _RealDownsetSolvePlan{K,F<:AbstractCoeffField} <: _AbstractDownsetSolvePlan{K}
        field::F
        A::Matrix{K}
    end

    function _downset_particular_solve_plan(field::RealField, A::AbstractMatrix{K}) where {K}
        return _RealDownsetSolvePlan{K,typeof(field)}(field, Matrix(A))
    end

    function _downset_particular_solve_plan(field::AbstractCoeffField, A::AbstractMatrix{K}) where {K}
        m, n = size(A)
        Aug = hcat(Matrix(A), _downset_dense_identity(K, m))
        R, pivs_all = rref(field, Aug)
        pivs = Int[]
        for p in pivs_all
            p <= n && push!(pivs, p)
        end
        return _ExactDownsetSolvePlan{K,typeof(field)}(
            field,
            Matrix(@view R[:, n+1:n+m]),
            pivs,
            length(pivs),
            n,
        )
    end

    function _downset_solve_particular(plan::_RealDownsetSolvePlan{K}, B::AbstractMatrix{K}) where {K}
        X = try
            plan.A \ Matrix(B)
        catch err
            if err isa LinearAlgebra.SingularException
                error("_downset_solve_particular: inconsistent system")
            end
            rethrow()
        end
        R = plan.A * X - B
        maxabs = isempty(R) ? zero(K) : maximum(abs, R)
        tol = plan.field.atol + plan.field.rtol * (isempty(plan.A) ? zero(K) : opnorm(plan.A, 1))
        maxabs <= tol || error("_downset_solve_particular: inconsistent system (residual=$maxabs, tol=$tol)")
        return X
    end

    function _downset_solve_particular(plan::_ExactDownsetSolvePlan{K}, B::AbstractMatrix{K}) where {K}
        size(B, 1) == size(plan.transform, 2) || error("_downset_solve_particular: row mismatch")
        Y = plan.transform * Matrix(B)
        if plan.rank < size(Y, 1)
            @inbounds for j in 1:size(Y, 2), i in (plan.rank + 1):size(Y, 1)
                iszero(Y[i, j]) || error("_downset_solve_particular: inconsistent system")
            end
        end
        X = zeros(K, plan.nvars, size(Y, 2))
        @inbounds for (row, pcol) in enumerate(plan.pivcols)
            X[pcol, :] = Y[row, :]
        end
        return X
    end

    @inline function _active_index_plan_key(P::AbstractPoset, base_vertices::Vector{Int}, upset_side::Bool)
        return (hash(P, zero(UInt64)), hash(base_vertices, zero(UInt64)), upset_side)
    end

    function _build_packed_active_index_plan(P::AbstractPoset, base_vertices::Vector{Int}, upset_side::Bool)
        n = nvertices(P)
        ns = length(base_vertices)
        counts = zeros(Int, n)
        base_counts = zeros(Int, n)

        @inbounds for i in 1:ns
            v = base_vertices[i]
            base_counts[v] += 1
            verts = upset_side ? upset_indices(P, v) : downset_indices(P, v)
            for u in verts
                counts[u] += 1
            end
        end

        ptr = Vector{Int}(undef, n + 1)
        ptr[1] = 1
        @inbounds for u in 1:n
            ptr[u + 1] = ptr[u] + counts[u]
        end

        base_ptr = Vector{Int}(undef, n + 1)
        base_ptr[1] = 1
        @inbounds for u in 1:n
            base_ptr[u + 1] = base_ptr[u] + base_counts[u]
        end

        data = Vector{Int}(undef, ptr[end] - 1)
        base_data = Vector{Int}(undef, base_ptr[end] - 1)
        next_ptr = copy(ptr)
        next_base_ptr = copy(base_ptr)
        base_pos = Vector{Int}(undef, ns)

        @inbounds for i in 1:ns
            v = base_vertices[i]
            p = next_base_ptr[v]
            base_data[p] = i
            next_base_ptr[v] = p + 1

            verts = upset_side ? upset_indices(P, v) : downset_indices(P, v)
            for u in verts
                p = next_ptr[u]
                data[p] = i
                next_ptr[u] = p + 1
                if u == v
                    base_pos[i] = p - ptr[u] + 1
                end
            end
        end

        return _PackedActiveIndexPlan(data, ptr, base_pos, base_data, base_ptr)
    end

    function _cached_packed_active_index_plan(P::AbstractPoset, base_vertices::Vector{Int}, upset_side::Bool)
        key = _active_index_plan_key(P, base_vertices, upset_side)
        lock(_ACTIVE_INDEX_PLAN_LOCK)
        try
            get!(_ACTIVE_INDEX_PLAN_CACHE, key) do
                _build_packed_active_index_plan(P, base_vertices, upset_side)
            end
        finally
            unlock(_ACTIVE_INDEX_PLAN_LOCK)
        end
    end

    @inline _packed_active_upset_plan(P::AbstractPoset, base_vertices::Vector{Int}) =
        _cached_packed_active_index_plan(P, base_vertices, true)

    @inline _packed_active_downset_plan(P::AbstractPoset, base_vertices::Vector{Int}) =
        _cached_packed_active_index_plan(P, base_vertices, false)

    function _active_upset_indices(P::AbstractPoset, base_vertices::Vector{Int})
        plan = _packed_active_upset_plan(P, base_vertices)
        active = [Int[] for _ in 1:nvertices(P)]
        @inbounds for u in 1:nvertices(P)
            start = plan.ptr[u]
            stop = plan.ptr[u + 1] - 1
            if start <= stop
                active[u] = plan.data[start:stop]
            end
        end
        return active
    end



    """
        _active_downset_indices(P::AbstractPoset, base_vertices::Vector{Int}) -> Vector{Vector{Int}}

    For a direct sum of principal downsets

        oplus_i k[Dn(base_vertices[i])],

    return `active[u]` = the list of summand indices that are active at vertex `u`.

    Convention:
    - A principal downset Dn(v) contains u iff u <= v.
    - Summand i is active at u iff leq(P, u, base_vertices[i]).

    The returned lists are in increasing summand index order. This matches the fiber
    basis ordering used in `_injective_hull` and makes coefficient extraction stable.
    """
    function _active_downset_indices(P::AbstractPoset, base_vertices::Vector{Int})
        plan = _packed_active_downset_plan(P, base_vertices)
        active = [Int[] for _ in 1:nvertices(P)]
        @inbounds for u in 1:nvertices(P)
            start = plan.ptr[u]
            stop = plan.ptr[u + 1] - 1
            if start <= stop
                active[u] = plan.data[start:stop]
            end
        end
        return active
    end

    struct _BaseVertexGroupsKey
        qid::UInt
        bases_id::UInt
    end

    const _BASE_VERTEX_GROUPS_CACHE = Dict{_BaseVertexGroupsKey,Any}()
    const _BASE_VERTEX_GROUPS_LOCK = ReentrantLock()
    const _ACTIVE_UPSET_VECTOR_CACHE = Dict{_BaseVertexGroupsKey,Any}()
    const _ACTIVE_UPSET_VECTOR_LOCK = ReentrantLock()

    @inline function _base_vertex_groups(P::AbstractPoset, base_vertices::Vector{Int})
        if !_RESOLUTIONS_USE_PROJECTIVE_LIFT_STRUCTURE_CACHE[]
            groups = [Int[] for _ in 1:nvertices(P)]
            @inbounds for i in eachindex(base_vertices)
                push!(groups[base_vertices[i]], i)
            end
            return groups
        end
        key = _BaseVertexGroupsKey(UInt(objectid(P)), UInt(objectid(base_vertices)))
        lock(_BASE_VERTEX_GROUPS_LOCK)
        try
            return get!(_BASE_VERTEX_GROUPS_CACHE, key) do
                groups = [Int[] for _ in 1:nvertices(P)]
                @inbounds for i in eachindex(base_vertices)
                    push!(groups[base_vertices[i]], i)
                end
                groups
            end
        finally
            unlock(_BASE_VERTEX_GROUPS_LOCK)
        end
    end

    @inline function _active_upset_indices_cached(P::AbstractPoset, base_vertices::Vector{Int})
        if !_RESOLUTIONS_USE_PROJECTIVE_LIFT_STRUCTURE_CACHE[]
            return _active_upset_indices(P, base_vertices)
        end
        key = _BaseVertexGroupsKey(UInt(objectid(P)), UInt(objectid(base_vertices)))
        lock(_ACTIVE_UPSET_VECTOR_LOCK)
        try
            return get!(_ACTIVE_UPSET_VECTOR_CACHE, key) do
                _active_upset_indices(P, base_vertices)
            end
        finally
            unlock(_ACTIVE_UPSET_VECTOR_LOCK)
        end
    end


    """
        _coeff_matrix_downsets(P, dom_bases, cod_bases, f) -> SparseMatrixCSC{K,Int}

    Extract the scalar coefficient matrix of a morphism between direct sums of
    principal downsets.

    Inputs:
    - `dom_bases`: base vertices of the domain summands (each summand is k[Dn(v)]).
    - `cod_bases`: base vertices of the codomain summands (each summand is k[Dn(w)]).
    - `f`: a `PMorphism{K}` whose domain/codomain are those direct sums.

    Output:
    - A sparse matrix C of size (length(cod_bases) x length(dom_bases)) such that
    C[row, col] is the scalar multiplying the unique (up to scalar) map

        k[Dn(dom_bases[col])] -> k[Dn(cod_bases[row])].

    Implementation detail:
    - For downsets, the distinguished generator of k[Dn(w)] lives at vertex w.
    To read the scalar for a map into k[Dn(w)], evaluate at u = w (the codomain
    base vertex), where the codomain generator is visible.
    """
    function _coeff_matrix_downsets(P::AbstractPoset,
                                    dom_bases::Vector{Int},
                                    cod_bases::Vector{Int},
                                    f::PMorphism{K}) where {K}
        n_dom = length(dom_bases)
        n_cod = length(cod_bases)

        dom_plan = _packed_active_downset_plan(P, dom_bases)
        cod_plan = _packed_active_downset_plan(P, cod_bases)

        I = Int[]
        J = Int[]
        V = K[]
        sizehint!(I, n_cod)
        sizehint!(J, n_cod)
        sizehint!(V, n_cod)

        @inbounds for row in 1:n_cod
            u = cod_bases[row]
            pos_row = cod_plan.base_pos[row]
            Fu = f.comps[u]
            start = dom_plan.ptr[u]
            stop = dom_plan.ptr[u + 1] - 1
            local_col = 1
            for idx in start:stop
                col = dom_plan.data[idx]
                val = Fu[pos_row, local_col]
                if !iszero(val)
                    push!(I, row)
                    push!(J, col)
                    push!(V, val)
                end
                local_col += 1
            end
        end

        return sparse(I, J, V, n_cod, n_dom)
    end

    """
        _coeff_matrix_upsets(P, dom_bases, cod_bases, f) -> SparseMatrixCSC{K,Int}

    Extract the coefficient matrix of a morphism `f` between direct sums of principal
    upsets, where those direct sums are indexed by lists of base vertices.

    Interpretation:

    - `dom_bases[i]` is the base vertex of the i-th domain summand (a principal upset).
    - `cod_bases[j]` is the base vertex of the j-th codomain summand.

    The coefficient C[j,i] is read at the *domain base vertex* u = dom_bases[i].
    At that vertex, the domain summand i is guaranteed active, and any codomain
    summand j that can receive a nonzero map is also active there.

    This is the "upset-side" analog of `_coeff_matrix_downsets`, but it is
    column-driven (read at domain base vertices) rather than row-driven.

    Performance notes:
    - Uses precomputed active-index lists at each vertex.
    - Uses `searchsortedfirst` instead of a closure-based `findfirst` to avoid
    allocations and to exploit sorted active lists.
    - Assembles as sparse triplets, then calls `sparse(...)` once.
    """
    function _coeff_matrix_upsets(P::AbstractPoset,
                                dom_bases::Vector{Int},
                                cod_bases::Vector{Int},
                                f::PMorphism{K}) where {K}
        n_dom = length(dom_bases)
        n_cod = length(cod_bases)

        dom_plan = _packed_active_upset_plan(P, dom_bases)
        cod_plan = _packed_active_upset_plan(P, cod_bases)

        I = Int[]
        J = Int[]
        V = K[]

        # A mild sizehint: in many resolutions, differentials are relatively sparse.
        sizehint!(I, n_dom)
        sizehint!(J, n_dom)
        sizehint!(V, n_dom)

        @inbounds for col in 1:n_dom
            u = dom_bases[col]
            pos_col = dom_plan.base_pos[col]
            Fu = f.comps[u]  # (#active cod at u) x (#active dom at u)

            # Read the image of the domain generator at u as a column of Fu.
            start = cod_plan.ptr[u]
            stop = cod_plan.ptr[u + 1] - 1
            pos_row = 1
            for idx in start:stop
                row = cod_plan.data[idx]
                val = Fu[pos_row, pos_col]
                if !iszero(val)
                    push!(I, row)
                    push!(J, col)
                    push!(V, val)
                end
                pos_row += 1
            end
        end

        return sparse(I, J, V, n_cod, n_dom)
    end


    # Extract coefficient matrix for a morphism between sums of principal upsets.
    # Domain and codomain are direct sums of principal upsets indexed by base vertex lists.
    function _coeff_matrix_upsets(dom_bases::AbstractVector{<:Upset},
                                  cod_bases::AbstractVector{<:Upset},
                                  ::Type{K}) where {K}
        n_dom = length(dom_bases)
        n_cod = length(cod_bases)

        I = Int[]
        J = Int[]
        V = K[]

        @inbounds for i in 1:n_dom
            Ui = dom_bases[i]
            for j in 1:n_cod
                Uj = cod_bases[j]
                # Nonzero iff cod upset is contained in dom upset.
                if is_subset(Uj, Ui)
                    push!(I, j)
                    push!(J, i)
                    push!(V, one(K))
                end
            end
        end

        return sparse(I, J, V, n_cod, n_dom)
    end


    # Internal implementation: build a projective resolution truncated/padded to `maxlen`.
    function _projective_resolution_impl(M::PModule{K}, maxlen::Int;
                                         threads::Bool = (Threads.nthreads() > 1)) where {K}
        maxlen >= 0 || error("_projective_resolution_impl: maxlen must be >= 0")
        n = nvertices(M.Q)
        cc = _get_cover_cache(M.Q)
        map_memo = _indicator_new_array_memo(K, n)
        ws = _new_resolution_workspace(K, n)
        kernel_cache = Vector{Union{Nothing,_KernelIncrementalEntry{K}}}(undef, n)
        fill!(kernel_cache, nothing)
        # Step 0
        P0, pi0, gens0 = projective_cover(
            M;
            cache=cc,
            map_memo=map_memo,
            workspace=ws,
            threads=threads,
        )
        bases0 = _flatten_gens_at(gens0)

        Pmods = PModule{K}[]
        push!(Pmods, P0)
        gens = Vector{Int}[bases0]
        d_mor = PMorphism{K}[]
        d_mat = SparseMatrixCSC{K, Int}[]

        # kernel K1 -> P0
        Kmod, iota = kernel_with_inclusion(
            pi0;
            cache=cc,
            incremental_cache=kernel_cache,
        )

        prevBases = bases0
        prevK = Kmod
        prevIota = iota

        for step in 1:maxlen
            # stop if kernel is zero
            if sum(prevK.dims) == 0
                break
            end

            _reset_indicator_memo!(map_memo)
            Pn, pin, gensn = projective_cover(
                prevK;
                cache=cc,
                map_memo=map_memo,
                workspace=ws,
                threads=threads,
            )
            basesn = _flatten_gens_at(gensn)

            # differential d_step = prevIota circ pin : Pn -> previous P
            d = compose(prevIota, pin)
            push!(Pmods, Pn)
            push!(gens, basesn)
            push!(d_mor, d)
            push!(d_mat, _coeff_matrix_upsets(M.Q, basesn, prevBases, d))

            # next kernel
            Kn, iotan = kernel_with_inclusion(
                pin;
                cache=cc,
                incremental_cache=kernel_cache,
            )

            prevBases = basesn
            prevK = Kn
            prevIota = iotan
        end

        res = ProjectiveResolution{K}(M, Pmods, gens, d_mor, d_mat, pi0)
        _pad_projective_resolution!(res, maxlen)   # pads with zero P-modules/differentials
        return res
    end

    """
        projective_resolution(M, res::ResolutionOptions) -> ProjectiveResolution
        projective_resolution(F::FringeModule, res::ResolutionOptions) -> ProjectiveResolution

    Build a projective resolution of `M`

        ... -> P_2 -> P_1 -> P_0 -> M -> 0

    up to degree `res.maxlen`.

    Cheap-first workflow
    - Start with `resolution_summary(resolution)` or `resolution_length(resolution)`.
    - Use `betti_table(resolution)` for multiplicity data before inspecting
      explicit terms or differentials.

    Heavier data
    - Use `resolution_terms(resolution)` and
      `resolution_differentials(resolution)` when you need the actual modules and
      morphisms.
    - Use `check_projective_resolution(resolution)` when validating a hand-built
      or externally cross-checked resolution.
    """
    # Public entrypoint: require ResolutionOptions.
    function projective_resolution(M::PModule{K}, res::ResolutionOptions;
                                   threads::Bool = (Threads.nthreads() > 1),
                                   cache::Union{Nothing,ResolutionCache}=nothing) where {K}
        key = _resolution_key2(M, res.maxlen)
        R = cache === nothing ? nothing : _cache_projective_get(cache, key, ProjectiveResolution{K})
        if R === nothing
            R = _projective_resolution_impl(M, res.maxlen; threads=threads)
            cache === nothing || (R = _cache_projective_store!(cache, key, R))
        end
        if res.minimal && res.check
            assert_minimal(R; check_cover=true)
        end
        return R
    end

    """
        projective_resolution(F::FringeModule, res::ResolutionOptions) -> ProjectiveResolution

    Convenience overload that first converts the fringe presentation to its
    associated `PModule`.
    """
    # Convenience overload for fringe modules.
    function projective_resolution(M::FringeModule{K}, res::ResolutionOptions;
                                   threads::Bool = (Threads.nthreads() > 1),
                                   cache::Union{Nothing,ResolutionCache}=nothing) where {K}
        key = _resolution_key2(M, res.maxlen)
        R = cache === nothing ? nothing : _cache_projective_get(cache, key, ProjectiveResolution{K})
        if R !== nothing
            if res.minimal && res.check
                assert_minimal(R; check_cover=true)
            end
            return R
        end
        R = projective_resolution(pmodule_from_fringe(M), res; threads=threads, cache=cache)
        cache === nothing || _cache_projective_store!(cache, key, R)
        return R
    end

    # =============================================================================
    # Betti and Bass numbers (multiplicities of indecomposable summands)
    # =============================================================================

    # Trim trailing all-zero rows (but keep at least 1 row if the table is nonempty).
    function _trim_trailing_zero_rows(B::AbstractMatrix{<:Integer})
        r, c = size(B)
        if r == 0 || c == 0
            return B
        end
        last = r
        while last > 1
            allzero = true
            for j in 1:c
                if B[last, j] != 0
                    allzero = false
                    break
                end
            end
            if allzero
                last -= 1
            else
                break
            end
        end
        return B[1:last, :]
    end

    function _pad_or_truncate_rows(B::AbstractMatrix{T}, nrows::Int) where {T<:Integer}
        r, c = size(B)
        if nrows == r
            return Matrix{T}(B)
        elseif nrows < r
            return Matrix{T}(B[1:nrows, :])
        else
            out = zeros(T, nrows, c)
            out[1:r, :] .= B
            return out
        end
    end

    """
        betti(res::ProjectiveResolution{K}) -> Dict{Tuple{Int,Int},Int}

    Return the Betti numbers of a projective resolution.

    Interpretation:
    - `res.Pmods[a+1]` is the projective in homological degree `a`.
    - Each term splits as a direct sum of indecomposable projectives k[Up(v)].
    - `res.gens[a+1]` stores the base vertex `v` for each summand, with repetition.

    Output convention:
    - The dictionary key `(a, v)` means "homological degree a, vertex v".
    - The value is the multiplicity of k[Up(v)] in P_a.
    - Only positive multiplicities appear as keys.

    This is the poset-module analogue of multigraded Betti numbers in commutative algebra.
    It is *not* a polynomial-ring Betti table unless you have explicitly modeled a
    polynomial-ring module as a poset module and computed its resolution in that category.
    """
    function betti(res::ProjectiveResolution{K}) where {K}
        out = Dict{Tuple{Int,Int},Int}()
        L = length(res.Pmods) - 1
        for a in 0:L
            for v in res.gens[a+1]
                key = (a, v)
                out[key] = get(out, key, 0) + 1
            end
        end
        return out
    end


    """
        betti_table(res::ProjectiveResolution{K}) -> Matrix{Int}

    Return a dense Betti table B.

    - Rows are homological degrees a = 0,1,2,...
    - Columns are vertices v = 1,...,Q.n
    - Entry B[a+1, v] is the multiplicity of k[Up(v)] in P_a.

    This is purely a formatting/convenience layer over `betti(res)`.
    """
    function betti_table(res::ProjectiveResolution{K}; pad_to::Union{Nothing,Int}=nothing) where {K}
        Q = res.M.Q
        L = length(res.Pmods) - 1
        B = zeros(Int, L + 1, Q.n)
        for a in 0:L
            for v in res.gens[a + 1]
                B[a + 1, v] += 1
            end
        end

        if pad_to === nothing
            return _trim_trailing_zero_rows(B)
        else
            pad_to >= 0 || error("betti_table: pad_to must be >= 0")
            return _pad_or_truncate_rows(B, pad_to + 1)
        end
    end


    """
        betti(M::PModule{K}, res::ResolutionOptions) -> Dict{Tuple{Int,Int},Int}

    Convenience wrapper:
    - build `projective_resolution(M, res)`,
    - return its Betti numbers.

    If you need full control over the chosen resolution object, call
    `projective_resolution` yourself and then call `betti(resolution)`.
    """
    function betti(M::PModule{K}, res::ResolutionOptions) where {K}
        return betti(projective_resolution(M, res))
    end

    """
        injective_resolution(F::FringeModule, res::ResolutionOptions) -> InjectiveResolution

    Convenience overload that first converts the fringe presentation to its
    associated `PModule`.
    """
    # Convenience overload for fringe modules.
    function betti(M::FringeModule{K}, res::ResolutionOptions) where {K}
        return betti(pmodule_from_fringe(M), res)
    end

    # ----------------------------
    # Minimality diagnostics for projective resolutions
    # ----------------------------

    """
        _vertex_counts(bases::Vector{Int}, nverts::Int) -> Vector{Int}

    Return the multiplicity vector of vertices in `bases`.

    If `bases` is the list of base vertices for a direct sum of principal upsets,
    then the output c satisfies:

        c[v] = number of copies of k[Up(v)].

    We use this for minimality certification, since multiplicities are the canonical
    data of a minimal resolution (up to isomorphism).
    """
    function _vertex_counts(bases::Vector{Int}, nverts::Int)
        c = zeros(Int, nverts)
        for v in bases
            c[v] += 1
        end
        return c
    end


    """
        ProjectiveMinimalityReport

    Returned by `minimality_report(res::ProjectiveResolution{K})`.

    Fields:
    - `minimal`:
        True iff all requested checks passed.

    - `cover_ok`:
        True iff the augmentation P0 -> M is a projective cover, checked by comparing
        vertex multiplicities with a freshly computed cover of M.

    - `cover_expected`, `cover_actual`:
        Multiplicity vectors of principal upsets in a projective cover of M and in
        the resolution's P0.

    - `diagonal_violations`:
        A list of tuples (a, v, row, col, val) witnessing non-minimality in higher
        degrees. Interpretation:
        - a is the homological degree of the differential d_a : P_a -> P_{a-1},
        - v is the vertex,
        - (row, col) is an entry in the coefficient matrix of d_a,
        - val is the nonzero scalar coefficient,
        - and the entry corresponds to a map k[Up(v)] -> k[Up(v)].
        Any such nonzero scalar is an isomorphism on that indecomposable summand,
        hence it splits off a contractible subcomplex. Minimal projective resolutions
        forbid this.
    """
    struct ProjectiveMinimalityReport{K}
        minimal::Bool
        cover_ok::Bool
        cover_expected::Vector{Int}
        cover_actual::Vector{Int}
        diagonal_violations::Vector{Tuple{Int,Int,Int,Int,K}}
    end


    """
        minimality_report(res::ProjectiveResolution{K}; check_cover=true) -> ProjectiveMinimalityReport

    Certify minimality of a projective resolution in the standard finite-dimensional
    algebra sense (incidence algebra / poset representation sense).

    Checks performed:

    1. (Optional) `check_cover`:
    Verify that P0 -> M is a projective cover by comparing multiplicities of
    principal upsets against a freshly computed `projective_cover(M)`.

    2. Verify the "no units on diagonal" condition:
    For every differential d_a : P_a -> P_{a-1}, there is no nonzero coefficient
    from a k[Up(v)] summand in P_a to a k[Up(v)] summand in P_{a-1}.
    """
    function minimality_report(res::ProjectiveResolution{K}; check_cover::Bool=true) where {K}
        Q = res.M.Q
        n = Q.n

        cover_actual = _vertex_counts(res.gens[1], n)
        cover_expected = copy(cover_actual)
        cover_ok = true

        if check_cover
            _, _, gens_at = projective_cover(res.M)
            cover_expected = _vertex_counts(_flatten_gens_at(gens_at), n)
            cover_ok = (cover_expected == cover_actual)
        end

        # Detect forbidden k[Up(v)] -> k[Up(v)] coefficients in the differentials.
        violations = Tuple{Int,Int,Int,Int,K}[]
        for a in 1:length(res.d_mat)
            D = res.d_mat[a]
            dom = res.gens[a+1]   # bases in P_a
            cod = res.gens[a]     # bases in P_{a-1}

            for col in 1:size(D, 2)
                for ptr in D.colptr[col]:(D.colptr[col+1] - 1)
                    row = D.rowval[ptr]
                    val = D.nzval[ptr]
                    if !iszero(val) && (cod[row] == dom[col])
                        v = dom[col]
                        push!(violations, (a, v, row, col, val))
                    end
                end
            end
        end

        minimal = cover_ok && isempty(violations)
        return ProjectiveMinimalityReport{K}(minimal, cover_ok, cover_expected, cover_actual, violations)
    end


    """
        is_minimal(res::ProjectiveResolution{K}; check_cover=true) -> Bool

    Return `true` iff `minimality_report(res; check_cover=check_cover).minimal` is true.
    """
    function is_minimal(res::ProjectiveResolution{K}; check_cover::Bool=true) where {K}
        return minimality_report(res; check_cover=check_cover).minimal
    end


    """
        assert_minimal(res::ProjectiveResolution{K}; check_cover=true) -> true

    Throw a descriptive error if the resolution fails minimality checks.
    Return `true` otherwise.

    This is intended for:
    - test suites,
    - defensively checking assumptions before extracting "minimal Betti invariants".
    """
    function assert_minimal(res::ProjectiveResolution{K}; check_cover::Bool=true) where {K}
        R = minimality_report(res; check_cover=check_cover)
        if R.minimal
            return true
        end

        if !R.cover_ok
            error("Projective resolution is not minimal: P0 -> M is not a projective cover. " *
                "Expected cover multiplicities = $(R.cover_expected), got = $(R.cover_actual).")
        end

        if !isempty(R.diagonal_violations)
            (a, v, row, col, val) = R.diagonal_violations[1]
            error("Projective resolution is not minimal: differential d_$a has a nonzero coefficient " *
                "val = $val from k[Up($v)] in P_$a (column $col) to k[Up($v)] in P_$(a-1) (row $row).")
        end

        error("Projective resolution failed minimality checks for an unknown reason.")
    end

    # =============================================================================
    # Injective resolutions
    # =============================================================================

    """
    Injective resolution of a module N:
        0 -> N -> E^0 -> E^1 -> ...

    The field `gens[b+1]` stores the base vertices of the indecomposable injective
    summands k[Dn(v)] appearing in E^b, with repetition.

    This makes it possible to extract Bass-type multiplicity data (injective summands
    by vertex and cohomological degree) in a canonical, user-facing way.
    """
    struct InjectiveResolution{K}
        N::PModule{K}
        # Same UnionAll issue as ProjectiveResolution.Pmods.
        Emods::Vector{<:PModule{K}}       # E^0, E^1, ...
        gens::Vector{Vector{Int}}         # base vertices per injective summand in each E^b
        d_mor::Vector{<:PMorphism{K}}     # E^b -> E^{b+1}
        iota0::PMorphism{K}               # N -> E^0
    end

    @inline resolution_terms(res::InjectiveResolution) = copy(res.Emods)
    @inline resolution_differentials(res::InjectiveResolution) = copy(res.d_mor)
    @inline coaugmentation_map(res::InjectiveResolution) = res.iota0
    @inline source_module(res::InjectiveResolution) = res.N
    @inline resolution_length(res::InjectiveResolution) = length(res.d_mor)

    """
        resolution_summary(res::InjectiveResolution) -> NamedTuple

    Cheap-first summary of an `InjectiveResolution`.

    This reports the source field, ambient poset size, degree range, total size
    of each injective term, and the number of stored generators in each
    degree, without materializing any new lifts or comparison data.

    Start with this or `resolution_length(res)` in notebook and REPL work. Ask
    for `resolution_terms(res)`, `resolution_differentials(res)`, or explicit
    coaugmentation data only when you need the full chain-level object.
    """
    @inline function resolution_summary(res::InjectiveResolution)
        return (
            kind=:injective_resolution,
            side=:injective,
            field=res.N.field,
            nvertices=nvertices(res.N.Q),
            degree_range=0:resolution_length(res),
            resolution_length=resolution_length(res),
            term_dimensions=Tuple(sum(E.dims) for E in res.Emods),
            generator_counts=Tuple(length(g) for g in res.gens),
        )
    end

    function Base.show(io::IO, res::ProjectiveResolution)
        d = resolution_summary(res)
        print(io, "ProjectiveResolution(length=", d.resolution_length,
              ", field=", d.field,
              ", generator_counts=", repr(d.generator_counts), ")")
    end

    function Base.show(io::IO, ::MIME"text/plain", res::ProjectiveResolution)
        d = resolution_summary(res)
        print(io, "ProjectiveResolution",
              "\n  field: ", d.field,
              "\n  nvertices: ", d.nvertices,
              "\n  degree_range: ", repr(d.degree_range),
              "\n  resolution_length: ", d.resolution_length,
              "\n  term_dimensions: ", repr(d.term_dimensions),
              "\n  generator_counts: ", repr(d.generator_counts))
    end

    function Base.show(io::IO, res::InjectiveResolution)
        d = resolution_summary(res)
        print(io, "InjectiveResolution(length=", d.resolution_length,
              ", field=", d.field,
              ", generator_counts=", repr(d.generator_counts), ")")
    end

    function Base.show(io::IO, ::MIME"text/plain", res::InjectiveResolution)
        d = resolution_summary(res)
        print(io, "InjectiveResolution",
              "\n  field: ", d.field,
              "\n  nvertices: ", d.nvertices,
              "\n  degree_range: ", repr(d.degree_range),
              "\n  resolution_length: ", d.resolution_length,
              "\n  term_dimensions: ", repr(d.term_dimensions),
              "\n  generator_counts: ", repr(d.generator_counts))
    end


    """
        injective_resolution(N::PModule{K}, res::ResolutionOptions) -> InjectiveResolution{K}

    Build an injective resolution
        0 -> N -> E^0 -> E^1 -> ... -> E^maxlen

    Implementation notes:
    - `injective_hull` is computed degreewise via `_injective_hull`.
    - The differentials are obtained by extending the map N^b -> E^b to E^b -> E^{b+1}.

    The resulting resolution is intended to be suitable for Ext computations and
    Bass-number extraction (multiplicity of injective indecomposables).
    """
    function _injective_resolution_impl(N::PModule{K}, maxlen::Int;
                                        threads::Bool = (Threads.nthreads() > 1)) where {K}
        maxlen >= 0 || error("_injective_resolution_impl: maxlen must be >= 0")
        n = nvertices(N.Q)
        cc = _get_cover_cache(N.Q)
        map_memo = _indicator_new_array_memo(K, n)
        ws = _new_resolution_workspace(K, n)
        cokernel_cache = Vector{Union{Nothing,_CokernelIncrementalEntry{K}}}(undef, n)
        fill!(cokernel_cache, nothing)
        support_vertices = _nonzero_dim_vertices(N.dims)

        E0, iota0, gens0 = _injective_hull(
            N;
            cache=cc,
            map_memo=map_memo,
            workspace=ws,
            support_vertices=support_vertices,
            threads=threads,
        )
        Emods = PModule{K}[]
        push!(Emods, E0)
        gens  = [_flatten_gens_at(gens0)]
        d_mor = PMorphism{K}[]

        C0, pi0 = _cokernel_module(
            iota0;
            cache=cc,
            incremental_cache=cokernel_cache,
            active_vertices=support_vertices,
        )
        prevC  = C0
        prevPi = pi0
        support_vertices = _nonzero_dim_vertices(C0.dims)

        for step in 1:maxlen
            _reset_indicator_memo!(map_memo)
            En, iotan, gensn = _injective_hull(
                prevC;
                cache=cc,
                map_memo=map_memo,
                workspace=ws,
                support_vertices=support_vertices,
                threads=threads,
            )
            push!(Emods, En)
            push!(gens, _flatten_gens_at(gensn))

            dn = compose(iotan, prevPi)   # E^{step-1} -> E^{step}
            push!(d_mor, dn)

            Cn, pin = _cokernel_module(
                iotan;
                cache=cc,
                incremental_cache=cokernel_cache,
                active_vertices=support_vertices,
            )
            prevC  = Cn
            prevPi = pin
            support_vertices = _nonzero_dim_vertices(Cn.dims)
        end

        return InjectiveResolution{K}(N, Emods, gens, d_mor, iota0)
    end

    # Convenience overload for fringe modules.
    # This mirrors the existing projective_resolution(::FringeModule) overload and
    # matters because encoding layers (Zn and PL/Rn) naturally produce FringeModule data.
    function injective_resolution(N::FringeModule{K}, res::ResolutionOptions;
                                  threads::Bool = (Threads.nthreads() > 1),
                                  cache::Union{Nothing,ResolutionCache}=nothing) where {K}
        key = _resolution_key2(N, res.maxlen)
        R = cache === nothing ? nothing : _cache_injective_get(cache, key, InjectiveResolution{K})
        if R !== nothing
            if res.minimal && res.check
                assert_minimal(R; check_hull=true)
            end
            return R
        end
        R = injective_resolution(pmodule_from_fringe(N), res; threads=threads, cache=cache)
        cache === nothing || _cache_injective_store!(cache, key, R)
        return R
    end

    """
        injective_resolution(N, res::ResolutionOptions) -> InjectiveResolution
        injective_resolution(F::FringeModule, res::ResolutionOptions) -> InjectiveResolution

    Build an injective resolution of `N`

        0 -> N -> E^0 -> E^1 -> E^2 -> ...

    up to degree `res.maxlen`.

    Cheap-first workflow
    - Start with `resolution_summary(resolution)` or `resolution_length(resolution)`.
    - Use `bass_table(resolution)` for multiplicity data before inspecting
      explicit terms or differentials.

    Heavier data
    - Use `resolution_terms(resolution)` and
      `resolution_differentials(resolution)` when you need the actual modules and
      morphisms.
    - Use `check_injective_resolution(resolution)` when validating a hand-built
      or externally cross-checked resolution.
    """
    #Pubic entrypoint API:
    function injective_resolution(N::PModule{K}, res::ResolutionOptions;
                                  threads::Bool = (Threads.nthreads() > 1),
                                  cache::Union{Nothing,ResolutionCache}=nothing) where {K}
        key = _resolution_key2(N, res.maxlen)
        R = cache === nothing ? nothing : _cache_injective_get(cache, key, InjectiveResolution{K})
        if R === nothing
            R = _injective_resolution_impl(N, res.maxlen; threads=threads)
            cache === nothing || (R = _cache_injective_store!(cache, key, R))
        end
        if res.minimal && res.check
            assert_minimal(R; check_hull=true)
        end
        return R
    end




    """
        bass(res::InjectiveResolution{K}) -> Dict{Tuple{Int,Int},Int}

    Bass numbers for an injective resolution.

    Interpretation:
    - `res.Emods[b+1]` is the injective in cohomological degree b (i.e. E^b).
    - Each E^b splits as a direct sum of indecomposable injectives k[Dn(v)].
    - `res.gens[b+1]` stores the base vertex v for each summand, with repetition.

    Output convention:
    - Key `(b, v)` means "cohomological degree b, vertex v".
    - Value is multiplicity of k[Dn(v)] in E^b.
    """
    function bass(res::InjectiveResolution{K}) where {K}
        out = Dict{Tuple{Int,Int},Int}()
        L = length(res.Emods) - 1
        for b in 0:L
            for v in res.gens[b+1]
                key = (b, v)
                out[key] = get(out, key, 0) + 1
            end
        end
        return out
    end


    """
        bass_table(res::InjectiveResolution{K}) -> Matrix{Int}

    Dense Bass table, analogous to `betti_table`:

    - Rows are cohomological degrees b = 0,1,2,...
    - Columns are vertices v = 1,...,Q.n
    - Entry B[b+1, v] is the multiplicity of k[Dn(v)] in E^b.
    """
    function bass_table(res::InjectiveResolution{K}; pad_to::Union{Nothing,Int}=nothing) where {K}
        Q = res.N.Q
        L = length(res.Emods) - 1
        B = zeros(Int, L + 1, Q.n)
        for b in 0:L
            for v in res.gens[b + 1]
                B[b + 1, v] += 1
            end
        end

        if pad_to === nothing
            return _trim_trailing_zero_rows(B)
        else
            pad_to >= 0 || error("bass_table: pad_to must be >= 0")
            return _pad_or_truncate_rows(B, pad_to + 1)
        end
    end


    """
        bass(N::PModule{K}, res::ResolutionOptions) -> Dict{Tuple{Int,Int},Int}

    Convenience wrapper:
    - build `injective_resolution(N, res)`,
    - return its Bass numbers.
    """
    function bass(N::PModule{K}, res::ResolutionOptions) where {K}
        return bass(injective_resolution(N, res))
    end

    function bass(N::FringeModule{K}, res::ResolutionOptions) where {K}
        return bass(pmodule_from_fringe(N), res)
    end


    # ----------------------------
    # Minimality diagnostics for injective resolutions
    # ----------------------------

    """
        InjectiveMinimalityReport

    Returned by `minimality_report(res::InjectiveResolution{K})`.

    Fields mirror `ProjectiveMinimalityReport`, but for injective resolutions:

    - `hull_ok` compares multiplicities in E^0 against a freshly computed injective hull.
    - `diagonal_violations` records nonzero coefficients k[Dn(v)] -> k[Dn(v)] in the
    differentials E^{b-1} -> E^b. Such coefficients split off contractible summands
    and are forbidden in a minimal injective resolution.
    """
    struct InjectiveMinimalityReport{K}
        minimal::Bool
        hull_ok::Bool
        hull_expected::Vector{Int}
        hull_actual::Vector{Int}
        diagonal_violations::Vector{Tuple{Int,Int,Int,Int,K}}
    end


    """
        minimality_report(res::InjectiveResolution{K}; check_hull=true) -> InjectiveMinimalityReport

    Certify minimality of an injective resolution in the standard finite-dimensional
    algebra sense.

    Checks performed:

    1. (Optional) `check_hull`:
    Verify that N -> E^0 is an injective hull by comparing multiplicities of principal
    downsets against a freshly computed `_injective_hull(N)`.

    2. Verify the "no units on diagonal" condition:
    For every differential d^b : E^{b-1} -> E^b, there is no nonzero coefficient
    k[Dn(v)] -> k[Dn(v)].
    """
    function minimality_report(res::InjectiveResolution{K}; check_hull::Bool=true) where {K}
        Q = res.N.Q
        n = Q.n

        hull_actual = _vertex_counts(res.gens[1], n)
        hull_expected = copy(hull_actual)
        hull_ok = true

        if check_hull
            _, _, gens_at = _injective_hull(res.N)
            hull_expected = _vertex_counts(_flatten_gens_at(gens_at), n)
            hull_ok = (hull_expected == hull_actual)
        end

        violations = Tuple{Int,Int,Int,Int,K}[]
        for b in 1:length(res.d_mor)
            dom = res.gens[b]     # bases in E^{b-1}
            cod = res.gens[b+1]   # bases in E^b

            D = _coeff_matrix_downsets(Q, dom, cod, res.d_mor[b])

            for col in 1:size(D, 2)
                for ptr in D.colptr[col]:(D.colptr[col+1] - 1)
                    row = D.rowval[ptr]
                    val = D.nzval[ptr]
                    if !iszero(val) && (cod[row] == dom[col])
                        v = dom[col]
                        push!(violations, (b, v, row, col, val))
                    end
                end
            end
        end

        minimal = hull_ok && isempty(violations)
        return InjectiveMinimalityReport{K}(minimal, hull_ok, hull_expected, hull_actual, violations)
    end


    """
        is_minimal(res::InjectiveResolution{K}; check_hull=true) -> Bool
    """
    function is_minimal(res::InjectiveResolution{K}; check_hull::Bool=true) where {K}
        return minimality_report(res; check_hull=check_hull).minimal
    end


    """
        assert_minimal(res::InjectiveResolution{K}; check_hull=true) -> true

    Throw a descriptive error if the injective resolution fails minimality checks.
    """
    function assert_minimal(res::InjectiveResolution{K}; check_hull::Bool=true) where {K}
        R = minimality_report(res; check_hull=check_hull)
        if R.minimal
            return true
        end

        if !R.hull_ok
            error("Injective resolution is not minimal: N -> E0 is not an injective hull. " *
                "Expected hull multiplicities = $(R.hull_expected), got = $(R.hull_actual).")
        end

        if !isempty(R.diagonal_violations)
            (b, v, row, col, val) = R.diagonal_violations[1]
            error("Injective resolution is not minimal: differential d^$b has a nonzero coefficient " *
                "val = $val from k[Dn($v)] in E_$(b-1) (column $col) to k[Dn($v)] in E_$b (row $row).")
        end

        error("Injective resolution failed minimality checks for an unknown reason.")
    end

    """
        check_projective_resolution(res; check_cover=false, throw=false) -> NamedTuple

    Validate the structural resolution data of a `ProjectiveResolution`.

    This check is intended for hand-built or externally cross-checked resolution
    objects. It verifies degree alignment, generator bookkeeping, differential
    shapes, and the chain relation `d_a * d_(a+1) = 0`. When `check_cover=true`,
    the report also records whether the built-in minimality diagnostics certify
    the cover stage, but a nonminimal resolution is not treated as invalid.
    """
    function check_projective_resolution(
        res::ProjectiveResolution{K};
        check_cover::Bool=false,
        throw::Bool=false,
    ) where {K}
        issues = String[]
        nverts = nvertices(res.M.Q)

        length(res.Pmods) == length(res.gens) ||
            push!(issues, "number of term modules must match number of generator blocks.")
        length(res.d_mor) == length(res.d_mat) ||
            push!(issues, "number of differentials must match number of coefficient matrices.")
        length(res.Pmods) == length(res.d_mor) + 1 ||
            push!(issues, "projective resolution must have one more term than differential.")
        isempty(res.Pmods) || res.aug.dom == first(res.Pmods) ||
            push!(issues, "augmentation domain must equal P_0.")
        res.aug.cod == res.M || push!(issues, "augmentation codomain must equal the source module.")

        for (a, gens_a) in enumerate(res.gens)
            all(1 <= u <= nverts for u in gens_a) ||
                push!(issues, "generator block $a contains a vertex outside 1:$nverts.")
        end

        for a in eachindex(res.d_mor)
            d = res.d_mor[a]
            d.dom == res.Pmods[a + 1] || push!(issues, "d_$a domain must equal P_$a.")
            d.cod == res.Pmods[a] || push!(issues, "d_$a codomain must equal P_$(a - 1).")
            size(res.d_mat[a]) == (length(res.gens[a]), length(res.gens[a + 1])) ||
                push!(issues, "coefficient matrix $a has size $(size(res.d_mat[a])) but expected ($(length(res.gens[a])), $(length(res.gens[a + 1]))).")
        end

        for a in 1:max(0, length(res.d_mor) - 1)
            is_zero_morphism(compose(res.d_mor[a], res.d_mor[a + 1])) ||
                push!(issues, "d_$a * d_$(a + 1) must vanish.")
        end

        isempty(res.d_mor) || is_zero_morphism(compose(res.aug, res.d_mor[1])) ||
            push!(issues, "augmentation must kill the first differential.")

        minrep = minimality_report(res; check_cover=check_cover)
        report = _derived_validation_report(
            :projective_resolution,
            isempty(issues);
            side=:projective,
            field=res.M.field,
            nvertices=nverts,
            resolution_length=resolution_length(res),
            generator_counts=Tuple(length(g) for g in res.gens),
            cover_checked=check_cover,
            cover_ok=minrep.cover_ok,
            minimal=minrep.minimal,
            issues=issues,
        )
        throw && !report.valid && _throw_invalid_derived_functor(:check_projective_resolution, issues)
        return report
    end

    """
        check_injective_resolution(res; check_hull=false, throw=false) -> NamedTuple

    Validate the structural resolution data of an `InjectiveResolution`.

    This check is intended for hand-built or externally cross-checked resolution
    objects. It verifies degree alignment, generator bookkeeping, differential
    shapes, and the chain relation `d^(b+1) * d^b = 0`. When `check_hull=true`,
    the report also records whether the built-in minimality diagnostics certify
    the hull stage, but a nonminimal resolution is not treated as invalid.
    """
    function check_injective_resolution(
        res::InjectiveResolution{K};
        check_hull::Bool=false,
        throw::Bool=false,
    ) where {K}
        issues = String[]
        nverts = nvertices(res.N.Q)

        length(res.Emods) == length(res.gens) ||
            push!(issues, "number of term modules must match number of generator blocks.")
        length(res.Emods) == length(res.d_mor) + 1 ||
            push!(issues, "injective resolution must have one more term than differential.")
        isempty(res.Emods) || res.iota0.cod == first(res.Emods) ||
            push!(issues, "coaugmentation codomain must equal E^0.")
        res.iota0.dom == res.N || push!(issues, "coaugmentation domain must equal the source module.")

        for (b, gens_b) in enumerate(res.gens)
            all(1 <= u <= nverts for u in gens_b) ||
                push!(issues, "generator block $b contains a vertex outside 1:$nverts.")
        end

        for b in eachindex(res.d_mor)
            d = res.d_mor[b]
            d.dom == res.Emods[b] || push!(issues, "d^$((b - 1)) domain must equal E^$((b - 1)).")
            d.cod == res.Emods[b + 1] || push!(issues, "d^$((b - 1)) codomain must equal E^$b.")
        end

        for b in 1:max(0, length(res.d_mor) - 1)
            is_zero_morphism(compose(res.d_mor[b + 1], res.d_mor[b])) ||
                push!(issues, "d^$b * d^$((b - 1)) must vanish.")
        end

        isempty(res.d_mor) || is_zero_morphism(compose(res.d_mor[1], res.iota0)) ||
            push!(issues, "first differential must kill the coaugmentation.")

        minrep = minimality_report(res; check_hull=check_hull)
        report = _derived_validation_report(
            :injective_resolution,
            isempty(issues);
            side=:injective,
            field=res.N.field,
            nvertices=nverts,
            resolution_length=resolution_length(res),
            generator_counts=Tuple(length(g) for g in res.gens),
            hull_checked=check_hull,
            hull_ok=minrep.hull_ok,
            minimal=minrep.minimal,
            issues=issues,
        )
        throw && !report.valid && _throw_invalid_derived_functor(:check_injective_resolution, issues)
        return report
    end


    # ---------------------------------------------------------------------------
    # Injective chain-map lifting
    # ---------------------------------------------------------------------------

    # Injective resolutions constructed by `injective_resolution` are explicit direct sums
    # of principal downset modules. Each summand is determined by a base vertex b, and the
    # summand contributes a 1-dimensional fiber at u iff u <= b.
    #
    # A morphism between such sums is determined by coefficients between summands. A map
    # Dn(v) -> Dn(w) can be nonzero only when w <= v. This restriction is essential for the
    # resulting fiberwise matrices to commute with structure maps (edges).
    #
    # The helpers below solve Phi o f = g for Phi between downset sums, then package Phi as a
    # genuine PMorphism. The public API is `lift_injective_chainmap`.

    # Build an index map for unknown coefficients, restricting to valid downset maps:
    # coefficient (r,c) corresponds to Dn(dom_bases[c]) -> Dn(cod_bases[r]) and is allowed
    # only if cod_bases[r] <= dom_bases[c].
    function _downset_hom_varidx(Q, dom_bases::Vector{Int}, cod_bases::Vector{Int})
        n_dom = length(dom_bases)
        n_cod = length(cod_bases)
        var_idx = zeros(Int, n_cod, n_dom)
        idx = 0
        for r in 1:n_cod
            br = cod_bases[r]
            for c in 1:n_dom
                bc = dom_bases[c]
                if leq(Q, br, bc)   # br <= bc
                    idx += 1
                    var_idx[r, c] = idx
                end
            end
        end
        return var_idx, idx
    end

    struct _DownsetStructureKey
        qid::UInt
        dom_id::UInt
        cod_id::UInt
    end

    struct _DownsetHomStructure
        act_dom::Vector{Vector{Int}}
        act_cod::Vector{Vector{Int}}
        var_idx::Matrix{Int}
        nunk::Int
    end

    struct _DownsetSystemKey
        fmap_id::UInt
        dom_id::UInt
        cod_id::UInt
    end

    struct _DownsetSystemRowRef
        u::Int
        rpos::Int
        j::Int
    end

    @inline function _dense_from_sparse_rows(::Type{K}, rows::Vector{SparseRow{K}}, ncols::Int) where {K}
        A = zeros(K, length(rows), ncols)
        @inbounds for i in eachindex(rows)
            row = rows[i]
            for t in eachindex(row.idx)
                A[i, row.idx[t]] = row.val[t]
            end
        end
        return A
    end

    struct _DownsetPostcomposeSystem{K,PlanT<:_AbstractDownsetSolvePlan{K}}
        structure::_DownsetHomStructure
        rows::Vector{SparseRow{K}}
        rhs_refs::Vector{_DownsetSystemRowRef}
        solve_plan::PlanT
    end

    const _DOWNSET_HOM_STRUCTURE_CACHE = Dict{_DownsetStructureKey,Any}()
    const _DOWNSET_HOM_STRUCTURE_LOCK = ReentrantLock()
    const _DOWNSET_POSTCOMPOSE_SYSTEM_CACHE = Dict{_DownsetSystemKey,Any}()
    const _DOWNSET_POSTCOMPOSE_SYSTEM_LOCK = ReentrantLock()

    @inline function _cached_downset_hom_structure(Q::AbstractPoset,
                                                   dom_bases::Vector{Int},
                                                   cod_bases::Vector{Int})
        if !_RESOLUTIONS_USE_INJECTIVE_DOWNSET_SYSTEM_CACHE[]
            act_dom = _active_downset_indices(Q, dom_bases)
            act_cod = _active_downset_indices(Q, cod_bases)
            var_idx, nunk = _downset_hom_varidx(Q, dom_bases, cod_bases)
            return _DownsetHomStructure(act_dom, act_cod, var_idx, nunk)
        end
        key = _DownsetStructureKey(UInt(objectid(Q)), UInt(objectid(dom_bases)), UInt(objectid(cod_bases)))
        lock(_DOWNSET_HOM_STRUCTURE_LOCK)
        try
            return get!(_DOWNSET_HOM_STRUCTURE_CACHE, key) do
                act_dom = _active_downset_indices(Q, dom_bases)
                act_cod = _active_downset_indices(Q, cod_bases)
                var_idx, nunk = _downset_hom_varidx(Q, dom_bases, cod_bases)
                _DownsetHomStructure(act_dom, act_cod, var_idx, nunk)
            end
        finally
            unlock(_DOWNSET_HOM_STRUCTURE_LOCK)
        end
    end

    function _build_downset_postcompose_system(f::PMorphism{K},
                                               structure::_DownsetHomStructure) where {K}
        Q = f.dom.Q
        rows = Vector{SparseRow{K}}()
        rhs_refs = Vector{_DownsetSystemRowRef}()
        sizehint!(rows, sum(length(structure.act_cod[u]) * size(f.comps[u], 2) for u in 1:Q.n))
        sizehint!(rhs_refs, length(rows))
        @inbounds for u in 1:Q.n
            rows_u = structure.act_cod[u]
            cols_u = structure.act_dom[u]
            Fu = f.comps[u]
            dX = size(Fu, 2)
            for rpos in 1:length(rows_u)
                r = rows_u[rpos]
                for j in 1:dX
                    idx = Int[]
                    vals = K[]
                    sizehint!(idx, length(cols_u))
                    sizehint!(vals, length(cols_u))
                    for cpos in 1:length(cols_u)
                        c = cols_u[cpos]
                        vidx = structure.var_idx[r, c]
                        vidx == 0 && continue
                        a = Fu[cpos, j]
                        iszero(a) && continue
                        push!(idx, vidx)
                        push!(vals, a)
                    end
                    push!(rows, SparseRow{K}(idx, vals))
                    push!(rhs_refs, _DownsetSystemRowRef(u, rpos, j))
                end
            end
        end
        plan = _downset_particular_solve_plan(f.dom.field, _dense_from_sparse_rows(K, rows, structure.nunk))
        return _DownsetPostcomposeSystem{K,typeof(plan)}(structure, rows, rhs_refs, plan)
    end

    @inline function _cached_downset_postcompose_system(f::PMorphism{K},
                                                        dom_bases::Vector{Int},
                                                        cod_bases::Vector{Int}) where {K}
        if !_RESOLUTIONS_USE_INJECTIVE_DOWNSET_SYSTEM_CACHE[]
            return _build_downset_postcompose_system(f, _cached_downset_hom_structure(f.dom.Q, dom_bases, cod_bases))
        end
        key = _DownsetSystemKey(UInt(objectid(f)), UInt(objectid(dom_bases)), UInt(objectid(cod_bases)))
        lock(_DOWNSET_POSTCOMPOSE_SYSTEM_LOCK)
        try
            return get!(_DOWNSET_POSTCOMPOSE_SYSTEM_CACHE, key) do
                _build_downset_postcompose_system(f, _cached_downset_hom_structure(f.dom.Q, dom_bases, cod_bases))
            end
        finally
            unlock(_DOWNSET_POSTCOMPOSE_SYSTEM_LOCK)
        end
    end

    # Turn a global coefficient matrix into an actual PMorphism by restricting to active
    # summands at each vertex (in the standard injective-hull basis order).
    @inline function _gather_component_matrix(C::AbstractMatrix{K},
                                              rows::Vector{Int},
                                              cols::Vector{Int}) where {K}
        nr = length(rows)
        nc = length(cols)
        M = Matrix{K}(undef, nr, nc)
        @inbounds for j in 1:nc
            cj = cols[j]
            for i in 1:nr
                M[i, j] = C[rows[i], cj]
            end
        end
        return M
    end

    function _pmorphism_from_downset_coeff(E::PModule{K}, Ep::PModule{K},
                                        act_dom::Vector{Vector{Int}},
                                        act_cod::Vector{Vector{Int}},
                                        C::Matrix{K}) where {K}
        Q = E.Q
        @assert _same_poset(Q, Ep.Q)
        comps = Vector{Matrix{K}}(undef, Q.n)
        for u in 1:Q.n
            rows = act_cod[u]
            cols = act_dom[u]
            comps[u] = _gather_component_matrix(C, rows, cols)
        end
        return PMorphism{K}(E, Ep, comps)
    end

    # Solve for C (coefficients) such that Phi(C) o f = g, where Phi(C) is a morphism between
    # direct sums of principal downsets described by dom_bases/cod_bases.
    #
    # This is used internally by lift_injective_chainmap, which threads the downset-basis
    # metadata explicitly (dom_bases, cod_bases, act_dom, act_cod) to avoid recomputation and
    # to keep the solver honest.
    """
        _solve_downset_postcompose_coeff(f, g, dom_bases, cod_bases, act_dom, act_cod; check=true) -> Matrix{K}

    Solve for a coefficient matrix `C` such that, fiberwise for every vertex `u`,
    `C[act_cod[u], act_dom[u]] * f(u) = g(u)`.

    This is implemented as a streaming sparse augmented elimination:
    each scalar equation is assembled as a sparse row in the unknown coefficients and pushed into
    `FieldLinAlg._SparseRREFAugmented`. A deterministic particular solution is returned (free variables set to 0).

    If the system is inconsistent, this throws an error (as that indicates a bug or incompatible truncation).
    """
    function _solve_downset_postcompose_coeff(
        f::PMorphism{K},
        g::PMorphism{K},
        dom_bases::Vector{Int},
        cod_bases::Vector{Int},
        act_dom::Vector{Vector{Int}},
        act_cod::Vector{Vector{Int}};
        check::Bool = true,
    ) where {K}
        Q = f.dom.Q

        if check
            @assert g.dom === f.dom
            @assert f.cod.Q === Q
            @assert g.cod.Q === Q
            @assert length(act_dom) == Q.n
            @assert length(act_cod) == Q.n
            for u in 1:Q.n
                @assert f.cod.dims[u] == length(act_dom[u])
                @assert g.cod.dims[u] == length(act_cod[u])
                @assert size(f.comps[u], 1) == f.cod.dims[u]
                @assert size(g.comps[u], 1) == g.cod.dims[u]
                @assert size(f.comps[u], 2) == f.dom.dims[u]
                @assert size(g.comps[u], 2) == g.dom.dims[u]
                @assert size(f.comps[u], 2) == size(g.comps[u], 2)
            end
        end

        system = _cached_downset_postcompose_system(f, dom_bases, cod_bases)
        structure = system.structure
        var_idx = structure.var_idx
        nunk = structure.nunk

        x = if _RESOLUTIONS_USE_INJECTIVE_DOWNSET_SOLVE_PLAN_CACHE[]
            rhs_mat = zeros(K, length(system.rhs_refs), 1)
            @inbounds for eq in eachindex(system.rhs_refs)
                ref = system.rhs_refs[eq]
                rhs_mat[eq, 1] = g.comps[ref.u][ref.rpos, ref.j]
            end
            _downset_solve_particular(system.solve_plan, rhs_mat)[:, 1]
        else
            R = _SparseRREFAugmented{K}(nunk, 1)
            rhs = Vector{K}(undef, 1)

            @inbounds for eq in eachindex(system.rows)
                ref = system.rhs_refs[eq]
                rhs[1] = g.comps[ref.u][ref.rpos, ref.j]
                row = SparseRow{K}(copy(system.rows[eq].idx), copy(system.rows[eq].val))
                status = _sparse_rref_push_augmented!(R, row, rhs)
                if status === :inconsistent
                    error("Inconsistent system in injective lift. This indicates a bug or incompatible resolution truncation.")
                end
            end

            out = zeros(K, nunk)
            @inbounds for pos in 1:length(R.rref.pivot_cols)
                pcol = R.rref.pivot_cols[pos]
                out[pcol] = R.pivot_rhs[pos][1]
            end
            out
        end

        # Assemble the full coefficient matrix C (zeros where no downset-hom variable exists).
        n_dom = length(dom_bases)
        n_cod = length(cod_bases)
        C = zeros(K, n_cod, n_dom)
        for r in 1:n_cod, c in 1:n_dom
            vidx = var_idx[r, c]
            vidx == 0 && continue
            C[r, c] = x[vidx]
        end
        return C
    end



    """
        lift_injective_chainmap(g, res_dom, res_cod; upto=nothing, check=true)

    Lift a module morphism `g : N -> Np` to a cochain map between chosen injective resolutions.

    If

        res_dom : 0 -> N  -> E0 -> E1 -> E2 -> ...
        res_cod : 0 -> Np -> F0 -> F1 -> F2 -> ...

    then the result is a vector `phis` with

        phis[k+1] : E_k -> F_k    (k = 0..upto)

    satisfying:

        phis[1] o res_dom.iota0 = res_cod.iota0 o g
        phis[k+1] o res_dom.d_mor[k] = res_cod.d_mor[k] o phis[k]   for k = 1..upto

    The lift exists because the `F_k` are injective. The choice is deterministic: linear
    systems are solved via `Utils.solve_particular` with free variables set to 0.

    Speed tip: reuse `res_dom` and `res_cod` when lifting many maps between the same modules.
    """
    function lift_injective_chainmap(g::PMorphism{K},
                                    res_dom::InjectiveResolution{K},
                                    res_cod::InjectiveResolution{K};
                                    upto::Union{Nothing, Int} = nothing,
                                    check::Bool = true) where {K}
        if check
            @assert g.dom === res_dom.N
            @assert g.cod === res_cod.N
        end
        Q = g.dom.Q

        Ldom = length(res_dom.d_mor)
        Lcod = length(res_cod.d_mor)
        L = (upto === nothing) ? min(Ldom, Lcod) : upto

        if check
            @assert 0 <= L
            @assert L <= Ldom
            @assert L <= Lcod
        end

        phis = Vector{PMorphism{K}}(undef, L + 1)

        # Degree 0: solve phi0 o iota0_dom = iota0_cod o g.
        dom_bases0 = res_dom.gens[1]
        cod_bases0 = res_cod.gens[1]
        structure0 = _cached_downset_hom_structure(Q, dom_bases0, cod_bases0)

        rhs0_comps = [res_cod.iota0.comps[u] * g.comps[u] for u in 1:Q.n]
        rhs0 = PMorphism{K}(g.dom, res_cod.Emods[1], rhs0_comps)

        C0 = _solve_downset_postcompose_coeff(res_dom.iota0, rhs0,
                                            dom_bases0, cod_bases0,
                                            structure0.act_dom, structure0.act_cod;
                                            check = check)
        phis[1] = _pmorphism_from_downset_coeff(res_dom.Emods[1], res_cod.Emods[1],
                                            structure0.act_dom, structure0.act_cod, C0)

        # Higher degrees: solve phi^k o d_dom^{k-1} = d_cod^{k-1} o phi^{k-1}.
        for k in 1:L
            dom_bases = res_dom.gens[k+1]
            cod_bases = res_cod.gens[k+1]
            structure = _cached_downset_hom_structure(Q, dom_bases, cod_bases)

            rhs_comps = [res_cod.d_mor[k].comps[u] * phis[k].comps[u] for u in 1:Q.n]
            rhs = PMorphism{K}(res_dom.Emods[k], res_cod.Emods[k+1], rhs_comps)

            Ck = _solve_downset_postcompose_coeff(res_dom.d_mor[k], rhs,
                                                dom_bases, cod_bases,
                                                structure.act_dom, structure.act_cod;
                                                check = check)
            phis[k+1] = _pmorphism_from_downset_coeff(res_dom.Emods[k+1], res_cod.Emods[k+1],
                                                    structure.act_dom, structure.act_cod, Ck)
        end

        return phis
    end

    """
        lift_injective_chainmap(g; maxlen=3, check=true)

    Convenience wrapper:
    - builds injective resolutions of `g.dom` and `g.cod` up to `maxlen`
    - lifts `g` up to degree `maxlen`

    Returns a named tuple `(res_dom, res_cod, phis)`.
    """
    function lift_injective_chainmap(g::PMorphism{K}; maxlen::Int = 3, check::Bool = true) where {K}
        res_dom = injective_resolution(g.dom, ResolutionOptions(maxlen=maxlen))
        res_cod = injective_resolution(g.cod, ResolutionOptions(maxlen=maxlen))
        phis = lift_injective_chainmap(g, res_dom, res_cod; upto=maxlen, check=check)
        return (res_dom=res_dom, res_cod=res_cod, phis=phis)
    end

    # Pad a projective resolution with zeros so Ext(M,N; maxdeg=d) always has tmax=d.
    function _zero_pmodule(Q::AbstractPoset, field::AbstractCoeffField)
        K = coeff_type(field)
        edge = Dict{Tuple{Int,Int},Matrix{K}}()
        for (u,v) in cover_edges(Q)
            edge[(u,v)] = zeros(K, 0, 0)
        end
        return PModule{K}(Q, zeros(Int, nvertices(Q)), edge; field=field)
    end

    function _pad_projective_resolution!(res::ProjectiveResolution{K}, maxdeg::Int) where {K}
        L = length(res.Pmods) - 1
        if L >= maxdeg
            return
        end

        Q = res.M.Q

        for a in (L+1):maxdeg
            # Add P_a = 0 and an empty generator list.
            push!(res.Pmods, _zero_pmodule(Q, res.M.field))
            push!(res.gens, Int[])

            # Add the zero differential d_a : P_a -> P_{a-1} as a PMorphism.
            dom = res.Pmods[a+1]   # P_a
            cod = res.Pmods[a]     # P_{a-1}
            comps = [zeros(K, cod.dims[v], dom.dims[v]) for v in 1:nvertices(Q)]
            push!(res.d_mor, PMorphism{K}(dom, cod, comps))

            # Also pad the matrix-on-generators representation.
            cod_summands = length(res.gens[a])     # summands of P_{a-1}
            dom_summands = length(res.gens[a+1])   # summands of P_a (0)
            push!(res.d_mat, spzeros(K, cod_summands, dom_summands))
        end

        @assert length(res.d_mor) == length(res.Pmods) - 1
        @assert length(res.d_mat) == length(res.Pmods) - 1
        return
    end

end
