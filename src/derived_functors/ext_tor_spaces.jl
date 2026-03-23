# ext_tor_spaces.jl -- typed Hom/Ext/Tor spaces and constructors

"""
ExtTorSpaces: typed containers for Hom/Ext/Tor spaces and their constructors.

This submodule should define (move here incrementally):
- HomSpace, ExtSpaceProjective, ExtSpaceInjective, TorSpace, etc
- constructors like Hom, Ext, ExtInjective, Tor
- graded-space methods that expose dims/bases/representatives
"""
module ExtTorSpaces
    using LinearAlgebra: rank, I, mul!
    using SparseArrays

    using ...CoreModules: AbstractCoeffField, RealField, ResolutionCache, ResolutionKey3, ResolutionKey4,
                          ExtProjectivePayload, ExtInjectivePayload, ExtUnifiedPayload,
                          TorFirstPayload, TorSecondPayload,
                          _resolution_key3, _resolution_key4, field_from_eltype
    using ...Options: ResolutionOptions, DerivedFunctorOptions
    import ...CoreModules: _append_scaled_triplets!
    import ...FieldLinAlg
    import ...FieldLinAlg: _SparseRREF, SparseRow,
              _SparseRowAccumulator, _reset_sparse_row_accumulator!,
              _push_sparse_row_entry!, _materialize_sparse_row!,
              _sparse_rref_push_homogeneous!,
              _nullspace_from_pivots

    using ...IndicatorTypes: UpsetPresentation, DownsetCopresentation
    using ...Modules: PModule, PMorphism, MapLeqQueryBatch, map_leq, map_leq_many, map_leq_many!,
                       _prepare_map_leq_batch_owned, _append_map_leq_many_scaled_triplets!
    using ...FiniteFringe: AbstractPoset, FinitePoset, FringeModule, fiber_dimension, Upset, Downset,
                           leq, leq_matrix, poset_equal, poset_equal_opposite, nvertices
    using ...AbelianCategories: kernel_with_inclusion
    using ...IndicatorResolutions: pmodule_from_fringe, indicator_resolutions,
        minimal_upset_presentation_one_step, minimal_downset_copresentation_one_step,
        upset_resolution, downset_resolution, projective_cover,
        verify_upset_resolution, verify_downset_resolution

    using ...ChainComplexes

    import ..DerivedFunctors: _build_total_offsets_grid, _total_offset_get
    import ..Utils: compose
    import ..HomExtEngine: ext_dims_via_resolutions, build_hom_tot_complex,
        CompCache, size_block, _block_offset, _component_inclusion_matrix_cached, _accum!
    import ..Resolutions: ProjectiveResolution, InjectiveResolution, _PackedActiveIndexPlan,
        _active_upset_indices, _packed_active_upset_plan,
        projective_resolution, injective_resolution, _pad_projective_resolution!
    import ..DerivedFunctors: source_module, target_module, nonzero_degrees, degree_dimensions,
                              total_dimension, hom_summary, ext_summary, tor_summary,
                              underlying_ext_space, underlying_tor_space, cached_product_degrees

    # Graded-space interface shared across derived objects.
    # Importing these names ensures that methods defined here extend the single shared function
    # objects (DerivedFunctors.GradedSpaces.*) rather than creating unrelated functions.
    import ..GradedSpaces: degree_range, dim, basis, representative, coordinates, cycles, boundaries

    @inline function _map_blocks_buffer(M::PModule{K,F,MatT}, n::Int) where {K,F,MatT<:AbstractMatrix{K}}
        return Vector{MatT}(undef, n)
    end

    @inline function _cache_ext_projective_get(cache::ResolutionCache, key::ResolutionKey3, ::Type{R}) where {R}
        if Threads.maxthreadid() == 1
            cached = get(cache.ext_projective, key, nothing)
            return cached === nothing ? nothing : (cached.value::R)
        end
        Base.lock(cache.lock)
        try
            cached = get(cache.ext_projective, key, nothing)
            return cached === nothing ? nothing : (cached.value::R)
        finally
            Base.unlock(cache.lock)
        end
    end

    @inline function _cache_ext_projective_store!(cache::ResolutionCache, key::ResolutionKey3, value::R) where {R}
        if Threads.maxthreadid() == 1
            cached = get(cache.ext_projective, key, nothing)
            cached === nothing || return (cached.value::R)
            cache.ext_projective[key] = ExtProjectivePayload(value)
            return value
        end
        Base.lock(cache.lock)
        try
            cached = get(cache.ext_projective, key, nothing)
            cached === nothing || return (cached.value::R)
            cache.ext_projective[key] = ExtProjectivePayload(value)
            return value
        finally
            Base.unlock(cache.lock)
        end
    end

    @inline function _cache_ext_injective_get(cache::ResolutionCache, key::ResolutionKey3, ::Type{R}) where {R}
        if Threads.maxthreadid() == 1
            cached = get(cache.ext_injective, key, nothing)
            return cached === nothing ? nothing : (cached.value::R)
        end
        Base.lock(cache.lock)
        try
            cached = get(cache.ext_injective, key, nothing)
            return cached === nothing ? nothing : (cached.value::R)
        finally
            Base.unlock(cache.lock)
        end
    end

    @inline function _cache_ext_injective_store!(cache::ResolutionCache, key::ResolutionKey3, value::R) where {R}
        if Threads.maxthreadid() == 1
            cached = get(cache.ext_injective, key, nothing)
            cached === nothing || return (cached.value::R)
            cache.ext_injective[key] = ExtInjectivePayload(value)
            return value
        end
        Base.lock(cache.lock)
        try
            cached = get(cache.ext_injective, key, nothing)
            cached === nothing || return (cached.value::R)
            cache.ext_injective[key] = ExtInjectivePayload(value)
            return value
        finally
            Base.unlock(cache.lock)
        end
    end

    @inline function _cache_ext_unified_get(cache::ResolutionCache, key::ResolutionKey4, ::Type{R}) where {R}
        if Threads.maxthreadid() == 1
            cached = get(cache.ext_unified, key, nothing)
            return cached === nothing ? nothing : (cached.value::R)
        end
        Base.lock(cache.lock)
        try
            cached = get(cache.ext_unified, key, nothing)
            return cached === nothing ? nothing : (cached.value::R)
        finally
            Base.unlock(cache.lock)
        end
    end

    @inline function _cache_ext_unified_store!(cache::ResolutionCache, key::ResolutionKey4, value::R) where {R}
        if Threads.maxthreadid() == 1
            cached = get(cache.ext_unified, key, nothing)
            cached === nothing || return (cached.value::R)
            cache.ext_unified[key] = ExtUnifiedPayload(value)
            return value
        end
        Base.lock(cache.lock)
        try
            cached = get(cache.ext_unified, key, nothing)
            cached === nothing || return (cached.value::R)
            cache.ext_unified[key] = ExtUnifiedPayload(value)
            return value
        finally
            Base.unlock(cache.lock)
        end
    end

    @inline function _cache_tor_first_get(cache::ResolutionCache, key::ResolutionKey3, ::Type{R}) where {R}
        if Threads.maxthreadid() == 1
            cached = get(cache.tor_first, key, nothing)
            return cached === nothing ? nothing : (cached.value::R)
        end
        Base.lock(cache.lock)
        try
            cached = get(cache.tor_first, key, nothing)
            return cached === nothing ? nothing : (cached.value::R)
        finally
            Base.unlock(cache.lock)
        end
    end

    @inline function _cache_tor_first_store!(cache::ResolutionCache, key::ResolutionKey3, value::R) where {R}
        if Threads.maxthreadid() == 1
            cached = get(cache.tor_first, key, nothing)
            cached === nothing || return (cached.value::R)
            cache.tor_first[key] = TorFirstPayload(value)
            return value
        end
        Base.lock(cache.lock)
        try
            cached = get(cache.tor_first, key, nothing)
            cached === nothing || return (cached.value::R)
            cache.tor_first[key] = TorFirstPayload(value)
            return value
        finally
            Base.unlock(cache.lock)
        end
    end

    @inline function _cache_tor_second_get(cache::ResolutionCache, key::ResolutionKey3, ::Type{R}) where {R}
        if Threads.maxthreadid() == 1
            cached = get(cache.tor_second, key, nothing)
            return cached === nothing ? nothing : (cached.value::R)
        end
        Base.lock(cache.lock)
        try
            cached = get(cache.tor_second, key, nothing)
            return cached === nothing ? nothing : (cached.value::R)
        finally
            Base.unlock(cache.lock)
        end
    end

    @inline function _cache_tor_second_store!(cache::ResolutionCache, key::ResolutionKey3, value::R) where {R}
        if Threads.maxthreadid() == 1
            cached = get(cache.tor_second, key, nothing)
            cached === nothing || return (cached.value::R)
            cache.tor_second[key] = TorSecondPayload(value)
            return value
        end
        Base.lock(cache.lock)
        try
            cached = get(cache.tor_second, key, nothing)
            cached === nothing || return (cached.value::R)
            cache.tor_second[key] = TorSecondPayload(value)
            return value
        finally
            Base.unlock(cache.lock)
        end
    end

    @inline _canon_tag(canon::Symbol) =
        canon === :projective ? UInt8(0x01) :
        canon === :injective ? UInt8(0x02) :
        error("_canon_tag: unsupported canon $(canon)")

    @inline _ext_unified_key(M, N, maxdeg::Int, canon::Symbol, check::Bool) =
        _resolution_key4(M, N, maxdeg, Int(_canon_tag(canon) | (check ? 0x10 : 0x00)))


    # ----------------------------
    # Hom space with explicit basis
    # ----------------------------

    mutable struct HomSpace{K}
        dom::PModule{K}
        cod::PModule{K}
        basis::Union{Nothing,Vector{PMorphism{K}}}
        basis_matrix::Matrix{K}  # columns are vectorizations of basis morphisms
        offsets::Vector{Int}     # per vertex block offsets in the vectorization
    end

    function Base.getproperty(H::HomSpace{K}, s::Symbol) where {K}
        if s === :basis
            return _ensure_hom_basis!(H)
        end
        return getfield(H, s)
    end

    function _hom_offsets(M::PModule{K}, N::PModule{K}) where {K}
        Q = M.Q
        @assert N.Q === Q
        nq = nvertices(Q)
        offs = zeros(Int, nq + 1)
        for i in 1:nq
            offs[i+1] = offs[i] + N.dims[i] * M.dims[i]
        end
        return offs
    end

    function _morphism_to_vector(f::PMorphism{K}, offs::Vector{Int}) where {K}
        Q = f.dom.Q
        v = zeros(K, offs[end], 1)
        for i in 1:nvertices(Q)
            di = f.cod.dims[i]
            ei = f.dom.dims[i]
            if di == 0 || ei == 0
                continue
            end
            block = vec(f.comps[i]) # column-major
            s = offs[i] + 1
            t = offs[i+1]
            v[s:t, 1] = block
        end
        return v
    end

    function _vector_to_morphism(dom::PModule{K}, cod::PModule{K}, offs::Vector{Int}, x::AbstractVector{K}) where {K}
        Q = dom.Q
        nq = nvertices(Q)
        comps = Vector{Matrix{K}}(undef, nq)
        for i in 1:nq
            di = cod.dims[i]
            ei = dom.dims[i]
            if di == 0 || ei == 0
                comps[i] = zeros(K, di, ei)
                continue
            end
            s = offs[i] + 1
            t = offs[i+1]
            comps[i] = reshape(x[s:t], di, ei)
        end
        return PMorphism{K}(dom, cod, comps)
    end

    """
        Hom(M::PModule{K}, N::PModule{K}) -> HomSpace{K}

    Compute Hom_Q(M,N) together with an explicit basis of module morphisms.

    Cheap-first workflow
    - Start with `hom_summary(H)`, `dim(H)`, or `degree_dimensions(H)` after the
      call returns.
    - Ask for `basis(H)` only when you actually need explicit morphisms.

    Performance notes
    - This function used to assemble a dense constraint matrix A=zeros(K,neqs,nvars)
    and then call FieldLinAlg.nullspace(field, A). That is prohibitively expensive when nvars is
    large, even though each constraint row is very sparse.
    - We now stream each naturality equation into an exact sparse RREF reducer
    (dictionary-of-rows) without ever materializing A.

    Mathematical content
    - Unknowns are the entries of the vertex maps F_u : M_u -> N_u for all u.
    - Constraints are the naturality equations along cover edges (u <. v):
        N_{uv} * F_u - F_v * M_{uv} = 0.
    """
    function Hom(M::PModule{K}, N::PModule{K}) where {K}
        Q = M.Q
        @assert N.Q === Q

        offs = _hom_offsets(M, N)
        nvars = offs[end]

        # Degenerate case: all vertex dimensions are zero.
        if nvars == 0
            return HomSpace{K}(M, N, PMorphism{K}[], zeros(K, 0, 0), offs)
        end

        dM = M.dims
        dN = N.dims

        # RREF basis of the row space of the constraint system, streamed row-by-row.
        R = _SparseRREF{K}(nvars)
        row = SparseRow{K}()
        acc = _SparseRowAccumulator{K}(nvars)
        fullrank = false

        storeM = M.edge_maps
        storeN = N.edge_maps

        # Performance: iterate cover edges using the store-aligned adjacency
        # (succs + maps_to_succ) to avoid keyed lookups like store[(u,v)] in hot loops.
        succs = storeM.succs
        mapsM = storeM.maps_to_succ
        mapsN = storeN.maps_to_succ

        @inbounds for u in 1:nvertices(Q)
            su = succs[u]
            Mu = mapsM[u]
            Nu = mapsN[u]

            du = dM[u]
            dNu = dN[u]
            (du == 0) && continue

            for j in eachindex(su)
                v = su[j]

                dv = dM[v]
                dNv = dN[v]
                (dNv == 0) && continue

                Nuv = Nu[j]
                Muv = Mu[j]

                for ii in 1:dNv, jj in 1:du
                    _reset_sparse_row_accumulator!(acc)

                    # Nuv * F_u
                    for k in 1:dNu
                        c = Nuv[ii, k]
                        if !iszero(c)
                            col = offs[u] + k + (jj - 1) * dNu
                            _push_sparse_row_entry!(acc, col, c)
                        end
                    end

                    # -F_v * Muv
                    for l in 1:dv
                        c = Muv[l, jj]
                        if !iszero(c)
                            col = offs[v] + ii + (l - 1) * dNv
                            _push_sparse_row_entry!(acc, col, -c)
                        end
                    end

                    _materialize_sparse_row!(row, acc)
                    isempty(row.idx) && continue
                    _sparse_rref_push_homogeneous!(R, row)

                    if length(R.pivot_cols) == nvars
                        fullrank = true
                        break
                    end
                end

                fullrank && break
            end
            fullrank && break
        end

        basis_matrix = fullrank ? zeros(K, nvars, 0) : _nullspace_from_pivots(R, nvars)

        return HomSpace{K}(M, N, nothing, basis_matrix, offs)
    end

    Hom(H::FringeModule{K}, N::PModule{K}) where {K} =
        Hom(pmodule_from_fringe(H), N)
    Hom(M::PModule{K}, H::FringeModule{K}) where {K} =
        Hom(M, pmodule_from_fringe(H))
    Hom(H1::FringeModule{K}, H2::FringeModule{K}) where {K} =
        Hom(pmodule_from_fringe(H1), pmodule_from_fringe(H2))


    function _ensure_hom_basis!(H::HomSpace{K}) where {K}
        basis_cache = getfield(H, :basis)
        basis_cache !== nothing && return basis_cache
        B = getfield(H, :basis_matrix)
        offs = getfield(H, :offsets)
        out = Vector{PMorphism{K}}(undef, size(B, 2))
        @inbounds for j in 1:size(B, 2)
            out[j] = _vector_to_morphism(getfield(H, :dom), getfield(H, :cod), offs, @view(B[:, j]))
        end
        setfield!(H, :basis, out)
        return out
    end

    dimension(H::HomSpace) = size(getfield(H, :basis_matrix), 2)
    basis(H::HomSpace) = _ensure_hom_basis!(H)
    dim(H::HomSpace) = size(getfield(H, :basis_matrix), 2)

    """
        hom_ext_first_page(F, E) -> (dimHom, dimExt1)

    Compute degree-0 and degree-1 dimensions (Hom and Ext^1) from full indicator
    resolutions associated to a pair of one-step (co)presentation objects.

    This uses the underlying fringe modules stored on the presentations and
    runs the full indicator-resolution Ext computation. It is intentionally
    correctness-first and does not use the indicator-only 2x2 approximation.

    This is a derived-functor driver and intentionally does not live in `IndicatorResolutions`.
    """
    function hom_ext_first_page(F::UpsetPresentation{K}, E::DownsetCopresentation{K}) where {K}
        if F.H === nothing || E.H === nothing
            error("hom_ext_first_page requires presentations built from FringeModule data.")
        end

        ext = ext_dimensions_via_indicator_resolutions(F.H, E.H; maxlen=2, verify=false)
        return get(ext, 0, 0), get(ext, 1, 0)
    end

    """
        ext_dimensions_via_indicator_resolutions(HM, HN; maxlen=10, verify=true, vertices=:all)

    High-level Ext-dimension computation via indicator resolutions.

    Builds indicator resolutions for `HM` and `HN`, optionally verifies them on selected vertices,
    then calls `ext_dims_via_resolutions` from `HomExtEngine`.
    """
    function ext_dimensions_via_indicator_resolutions(HM::FringeModule{K},
                                                    HN::FringeModule{K};
                                                    maxlen::Int=10,
                                                    verify::Bool=true,
                                                    vertices::Symbol=:all,
                                                    cache::Union{Nothing,ResolutionCache}=nothing) where {K}
        F, dF, E, dE = indicator_resolutions(HM, HN; maxlen=maxlen, cache=cache)

        if verify
            verify_upset_resolution(F, dF; vertices=vertices)
            verify_downset_resolution(E, dE; vertices=vertices)
        end

        return ext_dims_via_resolutions(F, dF, E, dE)
    end

    # ----------------------------
    # Ext via projective resolution: explicit cochains, cycles, boundaries, basis
    # ----------------------------

    struct ExtSpaceProjective{K,QT<:AbstractPoset}
        # Underlying poset/module data.
        #
        # NOTE: We store these as explicit fields (rather than via Base.getproperty shims)
        # so that field access stays type-stable and fast, and so the API has a single
        # canonical set of names.
        Q::QT
        M::PModule{K}

        # Projective resolution data and target module.
        res::ProjectiveResolution{K}
        N::PModule{K}

        # Cohomology computation data.
        complex::ChainComplexes.CochainComplex{K}
        offsets::Vector{Vector{Int}}
        cohom::Vector{ChainComplexes.CohomologyData{K}}

        # Degree range (stored explicitly to avoid getproperty runtime aliasing).
        tmin::Int
        tmax::Int
    end

    function ExtSpaceProjective(
        res::ProjectiveResolution{K},
        N::PModule{K},
        complex::ChainComplexes.CochainComplex{K},
        offsets::Vector{Vector{Int}},
        cohom::Vector{ChainComplexes.CohomologyData{K}}
    ) where {K}
        M = getfield(res, :M)
        Q = getfield(M, :Q)
        tmin = getfield(complex, :tmin)
        tmax = getfield(complex, :tmax)
        return ExtSpaceProjective{K,typeof(Q)}(Q, M, res, N, complex, offsets, cohom, tmin, tmax)
    end

    """
        _block_offsets_for_gens(M, gens) -> Vector{Int}

    Internal helper for Ext/Tor constructions.

    In the projective-resolution model of Ext, each term P_a is represented as a
    direct sum of indecomposable projectives k[Up(v)], recorded by the list

        res.gens[a+1] = [v1, v2, ..., vk].

    For any target module M, there is a canonical identification

        Hom(k[Up(v)], M) ~= M_v

    (a morphism out of a principal upset is determined by its value on the
    generator at v). Consequently,

        Hom(P_a, M) ~= direct_sum_{i=1..k} M_{v_i}.

    We store cochains in a single coordinate vector by concatenating these fibers.
    This function returns the cumulative offsets that locate each block M_{v_i}.

    Return value
    ------------
    The returned vector offs has length k+1 with offs[1] = 0 and

        offs[i+1] = offs[i] + dim(M_{v_i}).

    So the block corresponding to v_i occupies indices

        (offs[i] + 1):(offs[i+1])

    in a coordinate vector.
    """
    function _block_offsets_for_gens(M::PModule{K}, gens::Vector{Int}) where {K}
        offs = zeros(Int, length(gens) + 1)
        for i in 1:length(gens)
            v = gens[i]
            offs[i+1] = offs[i] + M.dims[v]
        end
        return offs
    end

    mutable struct _HomDiffWorkspace{K,MatT<:AbstractMatrix{K}}
        pairs::Vector{Tuple{Int,Int}}
        map_blocks::Vector{MatT}
        itrip::Vector{Int}
        jtrip::Vector{Int}
        vtrip::Vector{K}
    end

    @inline function _HomDiffWorkspace(::Type{K}, ::Type{MatT}) where {K,MatT<:AbstractMatrix{K}}
        return _HomDiffWorkspace{K,MatT}(Tuple{Int,Int}[], MatT[], Int[], Int[], K[])
    end

    function _build_hom_differential(
        res::ProjectiveResolution{K},
        N::PModule{K,F,MatT,QT},
        a::Int,
        offs_cod::Vector{Int},
        offs_dom::Vector{Int};
        workspace::Union{Nothing,_HomDiffWorkspace{K,MatT}}=nothing,
    ) where {K,F,MatT<:AbstractMatrix{K},QT}
        # a is the chain degree of the projective differential d_a: P_a -> P_{a-1}
        # On cochains: d^{a-1} : Hom(P_{a-1}, N) -> Hom(P_a, N)
        dom_gens = res.gens[a+1]      # summands of P_a
        cod_gens = res.gens[a]        # summands of P_{a-1}
        delta = res.d_mat[a]          # rows cod, cols dom

        out_dim = offs_dom[end]
        in_dim = offs_cod[end]

        Ii, Jj, Vv = findnz(delta)

        nnz_delta = length(Vv)
        if nnz_delta == 0
            return spzeros(K, out_dim, in_dim)
        end

        ws = workspace === nothing ? _HomDiffWorkspace(K, MatT) : workspace
        resize!(ws.pairs, nnz_delta)
        @inbounds for k in 1:nnz_delta
            ws.pairs[k] = (cod_gens[Ii[k]], dom_gens[Jj[k]])
        end
        pair_batch = _prepare_map_leq_batch_owned(ws.pairs)
        resize!(ws.map_blocks, nnz_delta)
        map_leq_many!(ws.map_blocks, N, pair_batch)
        do_threads = Threads.nthreads() > 1 && nnz_delta >= 64

        Itrip = ws.itrip
        Jtrip = ws.jtrip
        Vtrip = ws.vtrip
        empty!(Itrip)
        empty!(Jtrip)
        empty!(Vtrip)

        if do_threads
            nth = Threads.nthreads()
            local_I = [Int[] for _ in 1:nth]
            local_J = [Int[] for _ in 1:nth]
            local_V = [K[] for _ in 1:nth]

            Threads.@threads :static for tid in 1:nth
                kstart = fld((tid - 1) * nnz_delta, nth) + 1
                kend = fld(tid * nnz_delta, nth)
                Ii_loc = local_I[tid]
                Jj_loc = local_J[tid]
                Vv_loc = local_V[tid]

                for k in kstart:kend
                    j = Ii[k]   # cod summand index
                    i = Jj[k]   # dom summand index
                    c = Vv[k]
                    iszero(c) && continue

                    A = ws.map_blocks[k]  # N_vj -> N_ui

                    # Insert into block (rows for ui) x (cols for vj)
                    _append_scaled_triplets!(Ii_loc, Jj_loc, Vv_loc, A,
                                            offs_dom[i], offs_cod[j]; scale=c)
                end
            end

            total_nnz = 0
            for tid in 1:nth
                total_nnz += length(local_V[tid])
            end
            sizehint!(Itrip, total_nnz)
            sizehint!(Jtrip, total_nnz)
            sizehint!(Vtrip, total_nnz)
            for tid in 1:nth
                append!(Itrip, local_I[tid])
                append!(Jtrip, local_J[tid])
                append!(Vtrip, local_V[tid])
            end
        else
            for k in 1:nnz_delta
                j = Ii[k]   # cod summand index
                i = Jj[k]   # dom summand index
                c = Vv[k]
                iszero(c) && continue

                A = ws.map_blocks[k]  # N_vj -> N_ui

                # Insert into block (rows for ui) x (cols for vj)
                _append_scaled_triplets!(Itrip, Jtrip, Vtrip, A,
                                        offs_dom[i], offs_cod[j]; scale=c)
            end
        end

        return sparse(Itrip, Jtrip, Vtrip, out_dim, in_dim)
    end

    function _projective_ext_cochain_complex(
        res::ProjectiveResolution{K},
        N::PModule{K,F,MatT,QT};
        threads::Bool=(Threads.nthreads() > 1),
    ) where {K,F,MatT<:AbstractMatrix{K},QT}
        L = length(res.Pmods) - 1
        dimsC = Vector{Int}(undef, L + 1)
        offs = Vector{Vector{Int}}(undef, L + 1)
        for a in 0:L
            oa = _block_offsets_for_gens(N, res.gens[a + 1])
            offs[a + 1] = oa
            dimsC[a + 1] = oa[end]
        end

        dC = Vector{SparseMatrixCSC{K, Int}}(undef, L)
        if threads && Threads.nthreads() > 1 && L >= 2
            Threads.@threads for a in 1:L
                dC[a] = _build_hom_differential(res, N, a, offs[a], offs[a + 1])
            end
        else
            ws = _HomDiffWorkspace(K, MatT)
            for a in 1:L
                dC[a] = _build_hom_differential(res, N, a, offs[a], offs[a + 1]; workspace=ws)
            end
        end

        C = ChainComplexes.CochainComplex{K}(0, L, dimsC, dC)
        return C, offs
    end

    """
        Ext(M, N, df::DerivedFunctorOptions)

    Compute Ext^t(M,N) for 0 <= t <= df.maxdeg.

    The return type depends on df.model (interpreted for Ext):

    - :projective (or :auto): ExtSpaceProjective, computed from a projective resolution of M.
    - :injective: ExtSpaceInjective, computed from an injective resolution of N.
    - :unified: ExtSpace, containing both the projective and injective models with explicit comparison
    isomorphisms. The field df.canon chooses which coordinate basis is treated as canonical in the
    unified object (:projective or :injective; :auto is an alias for :projective).

    Cheap-first workflow
    - Start with `ext_summary(E)`, `nonzero_degrees(E)`, and
      `degree_dimensions(E)`.
    - Ask for `basis(E, t)` or `representative(E, t, ...)` only when you need
      explicit cocycles or comparison-level coordinates.

    Example
    -------
    A typical exploration pattern is:
    1. build `H = Hom(M, N)` and inspect `hom_summary(H)`,
    2. build `E = Ext(M, N, DerivedFunctorOptions(maxdeg=2))` and inspect
       `ext_summary(E)`,
    3. build `T = Tor(Rop, N, DerivedFunctorOptions(maxdeg=2))` and inspect
       `tor_summary(T)`,
    4. only then ask for `basis(E, t)`, `representative(E, t, ...)`, or
       `basis(T, s)` in the specific degrees you need.

    The options object is required; no keyword-based signature is provided.
    """
    function Ext(M::PModule{K}, N::PModule{K}, df::DerivedFunctorOptions;
                 cache::Union{Nothing,ResolutionCache}=nothing) where {K}
        maxdeg = df.maxdeg
        model = df.model === :auto ? :projective : df.model
        canon = df.canon === :auto ? :projective : df.canon

        if model === :projective
            return _Ext_projective(M, N; maxdeg=maxdeg, cache=cache)
        elseif model === :injective
            df_inj = DerivedFunctorOptions(maxdeg=maxdeg, model=:injective, canon=canon)
            return ExtInjective(M, N, df_inj; cache=cache)
        elseif model === :unified
            df_uni = DerivedFunctorOptions(maxdeg=maxdeg, model=:unified, canon=canon)
            return ExtSpace(M, N, df_uni; cache=cache)
        else
            error("Ext: unknown df.model=$(df.model). Supported for Ext: :projective, :injective, :unified, :auto.")
        end
    end

    function Ext(M::FringeModule{K}, N::FringeModule{K}, df::DerivedFunctorOptions;
                 cache::Union{Nothing,ResolutionCache}=nothing) where {K}
        Mp = pmodule_from_fringe(M)
        Np = pmodule_from_fringe(N)
        return Ext(Mp, Np, df; cache=cache)
    end

    function Ext(res::ProjectiveResolution{K}, N::PModule{K,F,MatT,QT};
                 threads::Bool=(Threads.nthreads() > 1)) where {K,F,MatT<:AbstractMatrix{K},QT}
        C, offs = _projective_ext_cochain_complex(res, N; threads=threads)
        cohom = ChainComplexes.cohomology_data(C)
        return ExtSpaceProjective(res, N, C, offs, cohom)
    end

    @inline function _Ext_projective_one_shot(
        M::PModule{K},
        N::PModule{K};
        maxdeg::Int=3,
        threads::Bool=(Threads.nthreads() > 1),
    ) where {K}
        local_cache = ResolutionCache()
        res = projective_resolution(M, ResolutionOptions(maxlen=maxdeg); threads=threads, cache=local_cache)
        _pad_projective_resolution!(res, maxdeg)
        return Ext(res, N; threads=threads)
    end

    # Internal helper: the traditional projective-resolution model of Ext.
    # This is the behavior that `Ext(M,N, DerivedFunctorOptions(...; model=:projective))` uses.
    function _Ext_projective(M::PModule{K}, N::PModule{K};
                             maxdeg::Int=3,
                             threads::Bool=(Threads.nthreads() > 1),
                             cache::Union{Nothing,ResolutionCache}=nothing) where {K}
        cache === nothing && return _Ext_projective_one_shot(M, N; maxdeg=maxdeg, threads=threads)
        if cache !== nothing
            key = _resolution_key3(M, N, maxdeg)
            cached = _cache_ext_projective_get(cache, key, ExtSpaceProjective{K})
            cached === nothing || return cached
        end
        res = projective_resolution(M, ResolutionOptions(maxlen=maxdeg); threads=threads, cache=cache)
        _pad_projective_resolution!(res, maxdeg)
        E = Ext(res, N; threads=threads)
        return cache === nothing ? E : _cache_ext_projective_store!(cache, _resolution_key3(M, N, maxdeg), E)
    end

    function dim(E::ExtSpaceProjective, t::Int)
        if t < E.tmin || t > E.tmax
            return 0
        end
        return E.cohom[t+1].dimH
    end

    function cycles(E::ExtSpaceProjective, t::Int)
        return E.cohom[t+1].K
    end

    function boundaries(E::ExtSpaceProjective, t::Int)
        return E.cohom[t+1].B
    end

    function representative(E::ExtSpaceProjective, t::Int, i::Int)
        Hrep = E.cohom[t+1].Hrep
        return Hrep[:, i]
    end

    """
        representative(E::ExtSpaceProjective, t::Int, coords::AbstractVector{K}) -> Vector{K} where {K}

    Return an explicit cocycle representative in the cochain space C^t of the Ext class
    whose coordinates (in the basis chosen internally by `E`) are given by `coords`.

    Mathematically:
    - `E` stores a basis of H^t(C) by choosing cocycle representatives (columns of `Hrep`).
    - This function returns the linear combination of those cocycles.

    This is useful when you want:
    - explicit chain-level representatives of arbitrary Ext elements,
    - Yoneda products on representatives,
    - custom linear combinations without manually forming them.
    """
    function representative(E::ExtSpaceProjective{K}, t::Int, coords::AbstractVector{K}) where {K}
        if t < 0 || t > E.tmax
            error("representative: degree t must satisfy 0 <= t <= tmax.")
        end
        Hrep = E.cohom[t+1].Hrep
        if length(coords) != size(Hrep, 2)
            error("representative: coordinate vector has length $(length(coords)), expected $(size(Hrep,2)).")
        end
        v = Hrep * reshape(coords, :, 1)
        return vec(v)
    end


    function basis(E::ExtSpaceProjective{K}, t::Int) where {K}
        Hrep = E.cohom[t+1].Hrep
        out = Vector{Vector{K}}(undef, size(Hrep, 2))
        for i in 1:size(Hrep, 2)
            out[i] = Vector{K}(Hrep[:, i])
        end
        return out
    end

    # Split a cochain vector into a list of fiber-vectors, one per projective summand generator.
    function split_cochain(E::ExtSpaceProjective, t::Int, v::AbstractVector{K}) where {K}
        offs = E.offsets[t+1]
        gens = E.res.gens[t+1]
        parts = Vector{Vector{K}}(undef, length(gens))
        for i in 1:length(gens)
            parts[i] = Vector{K}(v[(offs[i]+1):offs[i+1]])
        end
        return gens, parts
    end

    # Reduce a cocycle in C^t to Ext-coordinates in the chosen basis.
    function coordinates(E::ExtSpaceProjective, t::Int, cocycle::AbstractVector{K}) where {K}
        data = E.cohom[t+1]
        return ChainComplexes.cohomology_coordinates(data, cocycle)[:, 1]
    end

    function _blockdiag_on_hom_cochains(g::PMorphism{K}, gens::Vector{Int}, offs_src::Vector{Int}, offs_tgt::Vector{Int}) where {K}
        F = zeros(K, offs_tgt[end], offs_src[end])
        for i in 1:length(gens)
            u = gens[i]
            rows = (offs_tgt[i] + 1):(offs_tgt[i+1])
            cols = (offs_src[i] + 1):(offs_src[i+1])
            F[rows, cols] = g.comps[u]
        end
        return F
    end

    function _blockdiag_on_hom_cochains_sparse(
        g::PMorphism{K},
        gens::Vector{Int},
        offs_src::Vector{Int},
        offs_tgt::Vector{Int}
    ) where {K}
        out_dim = offs_tgt[end]
        in_dim  = offs_src[end]

        I = Int[]
        J = Int[]
        V = K[]

        @inbounds for i in 1:length(gens)
            u = gens[i]
            _append_scaled_triplets!(I, J, V, g.comps[u], offs_tgt[i], offs_src[i])
        end

        return sparse(I, J, V, out_dim, in_dim)
    end

    # Internal: encode a morphism P_t -> N as a cochain vector in C^t = Hom(P_t, N)
    # using the generator ordering stored in the projective resolution.
    function _cochain_vector_from_morphism(E::ExtSpaceProjective{K}, t::Int, f::PMorphism{K}) where {K}
        if t < 0 || t > E.tmax
            error("_cochain_vector_from_morphism: degree out of range.")
        end
        if f.dom !== E.res.Pmods[t+1] || f.cod !== E.N
            error("_cochain_vector_from_morphism: expected a morphism P_t -> N for the given Ext space.")
        end

        bases = E.res.gens[t+1]
        offs  = E.offsets[t+1]
        out = zeros(K, offs[end])
        plan = _packed_active_upset_plan(E.res.M.Q, bases)

        # For each summand i with base vertex u = bases[i],
        # locate the column position of i inside the fiber (P_t)_u,
        # then read off the image of that generator under f.
        for i in 1:length(bases)
            u = bases[i]
            du = E.N.dims[u]
            if du == 0
                continue
            end

            out[(offs[i]+1):offs[i+1]] = f.comps[u][:, plan.base_pos[i]]
        end

        return out
    end

    struct ExtSpaceInjective{K}
        # Underlying modules.
        M::PModule{K}
        N::PModule{K}

        # Injective resolution data (of N) and the Hom(M, I^t) cochain complex.
        res::InjectiveResolution{K}
        homs::Vector{HomSpace{K}}
        complex::ChainComplexes.CochainComplex{K}
        cohom::Vector{ChainComplexes.CohomologyData{K}}

        # Degree range stored explicitly to avoid getproperty runtime aliasing.
        tmin::Int
        tmax::Int
    end

    function ExtSpaceInjective(
        M::PModule{K},
        res::InjectiveResolution{K},
        homs::Vector{HomSpace{K}},
        complex::ChainComplexes.CochainComplex{K},
        cohom::Vector{ChainComplexes.CohomologyData{K}}
    ) where {K}
        N = getfield(res, :N)
        tmin = getfield(complex, :tmin)
        tmax = getfield(complex, :tmax)
        return ExtSpaceInjective{K}(M, N, res, homs, complex, cohom, tmin, tmax)
    end


    """
        representative(E::ExtSpaceInjective, t::Int, coords::AbstractVector{K}) -> Vector{K}

    Same as the projective-model method, but for an Ext space computed via an injective
    resolution of the second argument.

    Returns a cocycle in the cochain space Hom(M, E^t) (assembled over all degrees).
    """
    function representative(E::ExtSpaceInjective{K}, t::Int, coords::AbstractVector{K}) where {K}
        if t < 0 || t > E.tmax
            error("representative: degree t must satisfy 0 <= t <= tmax.")
        end
        Hrep = E.cohom[t+1].Hrep
        if length(coords) != size(Hrep, 2)
            error("representative: coordinate vector has length $(length(coords)), expected $(size(Hrep,2)).")
        end
        v = Hrep * reshape(coords, :, 1)
        return vec(v)
    end

    # -----------------------------------------------------------------------------
    # Basic queries for ExtSpaceInjective (parity with ExtSpaceProjective)
    # -----------------------------------------------------------------------------

    function dim(E::ExtSpaceInjective, t::Int)
        if t < E.tmin || t > E.tmax
            return 0
        end
        return E.cohom[t + 1].dimH
    end

    function cycles(E::ExtSpaceInjective, t::Int)
        return E.cohom[t + 1].K
    end

    function boundaries(E::ExtSpaceInjective, t::Int)
        return E.cohom[t + 1].B
    end

    function representative(E::ExtSpaceInjective, t::Int, i::Int)
        Hrep = E.cohom[t + 1].Hrep
        return Hrep[:, i]
    end

    function basis(E::ExtSpaceInjective{K}, t::Int) where {K}
        Hrep = E.cohom[t + 1].Hrep
        out = Vector{Vector{K}}(undef, size(Hrep, 2))
        for i in 1:size(Hrep, 2)
            out[i] = Vector{K}(Hrep[:, i])
        end
        return out
    end

    # Reduce a cocycle in C^t = Hom(M, E^t) to Ext-coordinates in the chosen basis.
    function coordinates(E::ExtSpaceInjective, t::Int, cocycle::AbstractVector{K}) where {K}
        data = E.cohom[t + 1]
        return ChainComplexes.cohomology_coordinates(data, cocycle)[:, 1]
    end

    mutable struct _PostcomposeWorkspace{K}
        rhs::Matrix{K}
    end

    @inline _PostcomposeWorkspace(::Type{K}) where {K} = _PostcomposeWorkspace{K}(zeros(K, 0, 0))

    @inline function _empty_sparse_matrix!(cache::Dict{Tuple{Int,Int},SparseMatrixCSC{K,Int}},
                                           nrows::Int,
                                           ncols::Int) where {K}
        return get!(cache, (nrows, ncols)) do
            spzeros(K, nrows, ncols)
        end
    end

    @inline function _empty_dense_row_cache(::Type{K}, dims::AbstractVector{<:Integer}) where {K}
        return [zeros(K, 0, Int(d)) for d in dims]
    end

    @inline function _empty_dense_col_cache(::Type{K}, dims::AbstractVector{<:Integer}) where {K}
        return [zeros(K, Int(d), 0) for d in dims]
    end

    @inline function _postcompose_rhs_buffer!(
        ws::Union{Nothing,_PostcomposeWorkspace{K}},
        nrows::Int,
        ncols::Int,
    ) where {K}
        if ws === nothing
            return zeros(K, nrows, ncols)
        end
        rhs = ws.rhs
        if size(rhs, 1) != nrows || size(rhs, 2) != ncols
            rhs = zeros(K, nrows, ncols)
            ws.rhs = rhs
        else
            fill!(rhs, zero(K))
        end
        return rhs
    end

    function _postcompose_matrix(
        Hdom::HomSpace{K},
        Hcod::HomSpace{K},
        g::PMorphism{K};
        workspace::Union{Nothing,_PostcomposeWorkspace{K}}=nothing,
    ) where {K}
        @assert Hdom.dom === Hcod.dom
        @assert Hcod.cod === g.dom
        @assert Hdom.cod === g.cod

        ncols = dim(Hcod)
        nrows = dim(Hdom)
        if ncols == 0 || nrows == 0
            return zeros(K, nrows, ncols)
        end

        rhs = _postcompose_rhs_buffer!(workspace, size(Hdom.basis_matrix, 1), ncols)
        M = Hdom.dom
        @inbounds for i in 1:nvertices(M.Q)
            mi = M.dims[i]
            src_dim = Hcod.cod.dims[i]
            dst_dim = Hdom.cod.dims[i]
            if mi == 0 || src_dim == 0 || dst_dim == 0
                continue
            end
            src_rng = (Hcod.offsets[i] + 1):Hcod.offsets[i + 1]
            dst_rng = (Hdom.offsets[i] + 1):Hdom.offsets[i + 1]
            src_view = reshape(@view(Hcod.basis_matrix[src_rng, :]), src_dim, mi * ncols)
            dst_view = reshape(@view(rhs[dst_rng, :]), dst_dim, mi * ncols)
            mul!(dst_view, g.comps[i], src_view)
        end
        return FieldLinAlg.solve_fullcolumn(Hdom.dom.field, Hdom.basis_matrix, rhs)
    end

    """
        ExtInjective(M, N, df::DerivedFunctorOptions)

    Compute Ext^t(M,N) for 0 <= t <= df.maxdeg using an injective resolution of N.
    Returns an ExtSpaceInjective.
    """
    function ExtInjective(M::PModule{K}, N::PModule{K}, df::DerivedFunctorOptions;
                          cache::Union{Nothing,ResolutionCache}=nothing) where {K}
        if !(df.model === :auto || df.model === :injective)
            error("ExtInjective: df.model must be :injective or :auto, got $(df.model)")
        end
        if cache === nothing
            threads = Threads.nthreads() > 1
            local_cache = ResolutionCache()
            resN = injective_resolution(N, ResolutionOptions(maxlen=df.maxdeg); threads=threads, cache=local_cache)
            return ExtInjective(M, resN; threads=threads)
        end
        if cache !== nothing
            key = _resolution_key3(M, N, df.maxdeg)
            cached = _cache_ext_injective_get(cache, key, ExtSpaceInjective{K})
            cached === nothing || return cached
        end
        threads = Threads.nthreads() > 1
        resN = injective_resolution(N, ResolutionOptions(maxlen=df.maxdeg); threads=threads, cache=cache)
        E = ExtInjective(M, resN; threads=threads)
        return cache === nothing ? E : _cache_ext_injective_store!(cache, _resolution_key3(M, N, df.maxdeg), E)
    end

    function ExtInjective(M::PModule{K}, resN::InjectiveResolution{K};
                          threads::Bool=(Threads.nthreads() > 1)) where {K}
        # Build the cochain complex C^b = Hom(M, E^b), where
        #   0 -> N -> E^0 -> E^1 -> ... -> E^L
        # is the chosen injective resolution of N.
        #
        # IMPORTANT:
        # - We represent each Hom space using an explicit basis (HomSpace), and we
        #   represent cochains as coordinate vectors in that basis.
        # - The cochain differential is induced by postcomposition with the
        #   resolution differential d^b : E^b -> E^{b+1}.

        L = length(resN.Emods) - 1

        homs = Vector{HomSpace{K}}(undef, L + 1)
        dims = Vector{Int}(undef, L + 1)

        for b in 0:L
            Hb = Hom(M, resN.Emods[b + 1])
            homs[b + 1] = Hb
            dims[b + 1] = dim(Hb)
        end

        # Build differentials dC[b+1] : C^b -> C^{b+1} for b = 0..L-1.
        dC = Vector{SparseMatrixCSC{K, Int}}(undef, L)
        empty_dC = [Dict{Tuple{Int,Int},SparseMatrixCSC{K,Int}}() for _ in 1:Threads.maxthreadid()]

        if threads && Threads.nthreads() > 1 && L >= 2
            Threads.@threads for b in 0:(L - 1)
                Hb  = homs[b + 1]
                Hb1 = homs[b + 2]
                db  = resN.d_mor[b + 1]   # E^b -> E^{b+1}

                dimHb  = dims[b + 1]
                dimHb1 = dims[b + 2]

                # Fast-path: empty source or target.
                if dimHb == 0 || dimHb1 == 0
                    dC[b + 1] = _empty_sparse_matrix!(empty_dC[Threads.threadid()], dimHb1, dimHb)
                    continue
                end
                dC[b + 1] = sparse(_postcompose_matrix(Hb1, Hb, db))
            end
        else
            ws = _PostcomposeWorkspace(K)
            empty_cache = empty_dC[1]
            for b in 0:(L - 1)
                Hb  = homs[b + 1]
                Hb1 = homs[b + 2]
                db  = resN.d_mor[b + 1]   # E^b -> E^{b+1}

                dimHb  = dims[b + 1]
                dimHb1 = dims[b + 2]

                # Fast-path: empty source or target.
                if dimHb == 0 || dimHb1 == 0
                    dC[b + 1] = _empty_sparse_matrix!(empty_cache, dimHb1, dimHb)
                    continue
                end
                dC[b + 1] = sparse(_postcompose_matrix(Hb1, Hb, db; workspace=ws))
            end
        end

        C = ChainComplexes.CochainComplex{K}(0, L, dims, dC)
        cohom = ChainComplexes.cohomology_data(C)

        return ExtSpaceInjective(M, resN, homs, C, cohom)
    end

    # =============================================================================
    # Model-independent Ext layer: explicit comparison isomorphisms + coherent basis transport
    # =============================================================================

    # We build an explicit comparison between:
    #  - the projective model: Ext computed from a projective resolution of M, i.e. Hom(P_*, N)
    #  - the injective model:  Ext computed from an injective resolution of N, i.e. Hom(M, E^*)
    #
    # The bridge is the standard total complex Tot(Hom(P_*, E^*)), with explicit cochain maps:
    #   Hom(P_*, N)  -> Tot(Hom(P_*, E^*))  <- Hom(M, E^*)
    #
    # We then solve for basis transport on cohomology degree-by-degree.

    # Compute the same block offsets used by `ChainComplexes.total_complex`:
    # For each total degree t, the blocks are ordered by increasing a, where b = t-a.
    #
    # NOTE:
    # We qualify DoubleComplex as ChainComplexes.DoubleComplex because ChainComplexes does not
    # export it, and DerivedFunctors should not rely on export lists for internal types.
    function _total_offsets(DC::ChainComplexes.DoubleComplex{K}) where {K}
        offsets, _, tmin, tmax = _build_total_offsets_grid(DC.amin, DC.amax, DC.bmin, DC.bmax, DC.dims)
        return offsets, tmin, tmax
    end


    function _precompose_to_projective_cochains_matrix(
        Hcod::HomSpace{K},
        aug::PMorphism{K},
        bases::Vector{Int},
        active::_PackedActiveIndexPlan,
        offs::Vector{Int},
        target::PModule{K},
    ) where {K}
        ncols = dim(Hcod)
        out = zeros(K, offs[end], ncols)
        basis_matrix = getfield(Hcod, :basis_matrix)
        offsets = getfield(Hcod, :offsets)

        @inbounds for u in 1:(length(active.base_ptr) - 1)
            gstart = active.base_ptr[u]
            gstop = active.base_ptr[u + 1] - 1
            if gstart > gstop
                continue
            end
            eu = target.dims[u]
            mu = Hcod.dom.dims[u]
            if eu == 0 || mu == 0
                continue
            end

            src_rng = (offsets[u] + 1):offsets[u + 1]
            src_view = reshape(@view(basis_matrix[src_rng, :]), eu, mu * ncols)
            aug_u = aug.comps[u]

            for gidx in gstart:gstop
                i = active.base_data[gidx]
                pos = active.base_pos[i]
                dst = @view(out[(offs[i] + 1):offs[i + 1], :])
                for k in 1:mu
                    coeff = aug_u[k, pos]
                    iszero(coeff) && continue
                    src_col = k
                    for j in 1:ncols
                        @simd for r in 1:eu
                            dst[r, j] += coeff * src_view[r, src_col]
                        end
                        src_col += mu
                    end
                end
            end
        end

        return out
    end

    """
        comparison_isomorphisms(Eproj, Einj; maxdeg=min(Eproj.tmax, Einj.tmax), check=true)

    Return explicit basis-transport matrices between the projective and injective models.

    The result is a pair `(P2I, I2P)` where, for each degree t:

    - `P2I[t+1]` maps coordinates in Ext^t(M,N) from the projective basis to the injective basis.
    - `I2P[t+1]` maps coordinates in Ext^t(M,N) from the injective basis to the projective basis.

    These matrices are computed by an explicit total-complex comparison
    Tot(Hom(P_*, E^*)) and are exact over exact fields.
    """
    function comparison_isomorphisms(
        Eproj::ExtSpaceProjective{K},
        Einj::ExtSpaceInjective{K};
        maxdeg::Int=min(Eproj.tmax, Einj.tmax),
        check::Bool=true
    ) where {K}
        return _comparison_projective_injective(Eproj, Einj; maxdeg=maxdeg, check=check)
    end

    # Internal engine: compute the comparison using Tot(Hom(P_*,E^*)).
    function _build_ext_comparison_context(
        Eproj::ExtSpaceProjective{K},
        Einj::ExtSpaceInjective{K},
        maxdeg::Int,
    ) where {K}
        @assert Eproj.M === Einj.M
        @assert Eproj.N === Einj.res.N

        maxdeg = min(maxdeg, Eproj.tmax, Einj.tmax)
        resP = Eproj.res
        resE = Einj.res
        Q = Eproj.M.Q

        amax = maxdeg
        bmax = maxdeg

        offs_blocks = Array{Vector{Int}}(undef, amax + 1, bmax + 1)
        dims_blocks = zeros(Int, amax + 1, bmax + 1)

        for a in 0:amax
            gens_a = resP.gens[a + 1]
            for b in 0:bmax
                Eb = resE.Emods[b + 1]
                off = _block_offsets_for_gens(Eb, gens_a)
                offs_blocks[a + 1, b + 1] = off
                dims_blocks[a + 1, b + 1] = off[end]
            end
        end

        dv = Array{SparseMatrixCSC{K,Int}}(undef, amax + 1, bmax + 1)
        dh = Array{SparseMatrixCSC{K,Int}}(undef, amax + 1, bmax + 1)
        empty_boundary_cache = Dict{Tuple{Int,Int},SparseMatrixCSC{K,Int}}()

        for a in 0:amax
            gens_a = resP.gens[a + 1]
            for b in 0:bmax
                dom_dim = dims_blocks[a + 1, b + 1]
                if b < bmax
                    g = resE.d_mor[b + 1]
                    offs_src = offs_blocks[a + 1, b + 1]
                    offs_tgt = offs_blocks[a + 1, b + 2]
                    dv[a + 1, b + 1] = _blockdiag_on_hom_cochains_sparse(g, gens_a, offs_src, offs_tgt)
                else
                    dv[a + 1, b + 1] = _empty_sparse_matrix!(empty_boundary_cache, 0, dom_dim)
                end

                if a < amax
                    Eb = resE.Emods[b + 1]
                    H = _build_hom_differential(resP, Eb, a + 1, offs_blocks[a + 1, b + 1], offs_blocks[a + 2, b + 1])
                    if isodd(b)
                        H = -H
                    end
                    dh[a + 1, b + 1] = H
                else
                    dh[a + 1, b + 1] = _empty_sparse_matrix!(empty_boundary_cache, 0, dom_dim)
                end
            end
        end

        DC = ChainComplexes.DoubleComplex{K}(0, amax, 0, bmax, dims_blocks, dv, dh)
        Tot = ChainComplexes.total_complex(DC)
        tot_offsets, tot_tmin, _ = _total_offsets(DC)
        basesP0 = resP.gens[1]
        activeP0 = _packed_active_upset_plan(Q, basesP0)
        return (maxdeg, resP, resE, DC, Tot, dims_blocks, offs_blocks, tot_offsets, tot_tmin, basesP0, activeP0)
    end

    function _comparison_projective_injective_degree(
        Eproj::ExtSpaceProjective{K},
        Einj::ExtSpaceInjective{K},
        t::Int;
        check::Bool=true,
    ) where {K}
        maxdeg, resP, resE, DC, Tot, dims_blocks, offs_blocks, tot_offsets, tot_tmin, basesP0, activeP0 =
            _build_ext_comparison_context(Eproj, Einj, t)

        t <= maxdeg || error("_comparison_projective_injective_degree: degree t=$(t) exceeds maxdeg=$(maxdeg)")
        cohomT = ChainComplexes.cohomology_data(Tot, t)

        tot_dim = Tot.dims[t - Tot.tmin + 1]
        dom_dim_proj = Eproj.complex.dims[t - Eproj.complex.tmin + 1]
        Fproj = zeros(K, tot_dim, dom_dim_proj)
        off_t0 = _total_offset_get(tot_offsets, t, tot_tmin, DC.amin, t)
        block_dim = dims_blocks[t + 1, 1]
        block_map = _blockdiag_on_hom_cochains(resE.iota0, resP.gens[t + 1], Eproj.offsets[t + 1], offs_blocks[t + 1, 1])
        Fproj[(off_t0 + 1):(off_t0 + block_dim), :] = block_map

        dom_dim_inj = Einj.complex.dims[t - Einj.complex.tmin + 1]
        Finj = zeros(K, tot_dim, dom_dim_inj)
        off_0t = _total_offset_get(tot_offsets, t, tot_tmin, DC.amin, 0)
        Eb = resE.Emods[t + 1]
        offs0t = offs_blocks[1, t + 1]
        block_dim_0t = dims_blocks[1, t + 1]
        block_inj = _precompose_to_projective_cochains_matrix(
            Einj.homs[t + 1],
            resP.aug,
            basesP0,
            activeP0,
            offs0t,
            Eb,
        )
        Finj[(off_0t + 1):(off_0t + block_dim_0t), :] = block_inj

        mp = ChainComplexes.induced_map_on_cohomology(Eproj.cohom[t + 1], cohomT, Fproj)
        mi = ChainComplexes.induced_map_on_cohomology(Einj.cohom[t + 1], cohomT, Finj)

        field = Eproj.M.field
        P2I = FieldLinAlg.solve_fullcolumn(field, mi, mp)
        I2P = FieldLinAlg.solve_fullcolumn(field, mp, mi)

        if check
            dp = dim(Eproj, t)
            di = dim(Einj, t)
            if dp != di
                error("comparison: dim mismatch at t=$(t): projective=$(dp), injective=$(di)")
            end
            I_d = Matrix{K}(I, dp, dp)
            if I2P * P2I != I_d
                error("comparison: I2P*P2I != identity at t=$(t)")
            end
            if P2I * I2P != I_d
                error("comparison: P2I*I2P != identity at t=$(t)")
            end
        end

        return P2I, I2P
    end

    function _comparison_projective_injective(
        Eproj::ExtSpaceProjective{K},
        Einj::ExtSpaceInjective{K};
        maxdeg::Int=min(Eproj.tmax, Einj.tmax),
        check::Bool=true
    ) where {K}
        maxdeg, resP, resE, DC, Tot, dims_blocks, offs_blocks, tot_offsets, tot_tmin, basesP0, activeP0 =
            _build_ext_comparison_context(Eproj, Einj, maxdeg)

        cohomT = Vector{ChainComplexes.CohomologyData{K}}(undef, maxdeg + 1)
        if Threads.nthreads() > 1 && maxdeg >= 1
            Threads.@threads for i in 1:(maxdeg + 1)
                cohomT[i] = ChainComplexes.cohomology_data(Tot, i - 1)
            end
        else
            for i in 1:(maxdeg + 1)
                cohomT[i] = ChainComplexes.cohomology_data(Tot, i - 1)
            end
        end

        P2I = Union{Nothing,Matrix{K}}[nothing for _ in 0:maxdeg]
        I2P = Union{Nothing,Matrix{K}}[nothing for _ in 0:maxdeg]

        for t in 0:maxdeg
            # ---- cochain map: Hom(P_t, N) -> Tot in degree t via N -> E^0
            tot_dim = Tot.dims[t - Tot.tmin + 1]
            dom_dim_proj = Eproj.complex.dims[t - Eproj.complex.tmin + 1]

            Fproj = zeros(K, tot_dim, dom_dim_proj)
            off_t0 = _total_offset_get(tot_offsets, t, tot_tmin, DC.amin, t)
            block_dim = dims_blocks[t+1, 1]  # (a=t, b=0)
            # Postcompose with iota0: N -> E^0, block-diagonal over generators of P_t.
            block_map = _blockdiag_on_hom_cochains(resE.iota0, resP.gens[t+1], Eproj.offsets[t+1], offs_blocks[t+1, 1])
            Fproj[(off_t0 + 1):(off_t0 + block_dim), :] = block_map

            # ---- cochain map: Hom(M, E^t) -> Tot in degree t via precompose with aug: P0 -> M
            dom_dim_inj = Einj.complex.dims[t - Einj.complex.tmin + 1]
            Finj = zeros(K, tot_dim, dom_dim_inj)
            off_0t = _total_offset_get(tot_offsets, t, tot_tmin, DC.amin, 0)
            Eb = resE.Emods[t+1]
            offs0t = offs_blocks[1, t+1]     # (a=0, b=t)
            block_dim_0t = dims_blocks[1, t+1]
            block_inj = _precompose_to_projective_cochains_matrix(
                Einj.homs[t+1],
                resP.aug,
                basesP0,
                activeP0,
                offs0t,
                Eb,
            )
            Finj[(off_0t + 1):(off_0t + block_dim_0t), :] = block_inj

            # induced maps on cohomology
            mp = ChainComplexes.induced_map_on_cohomology(Eproj.cohom[t+1], cohomT[t+1], Fproj)
            mi = ChainComplexes.induced_map_on_cohomology(Einj.cohom[t+1],  cohomT[t+1], Finj)

            # Solve for explicit basis transport:
            #   mi * (proj->inj) = mp
            # so proj->inj = FieldLinAlg.solve_fullcolumn(mi, mp)
            field = Eproj.M.field
            P2I[t+1] = FieldLinAlg.solve_fullcolumn(field, mi, mp)
            I2P[t+1] = FieldLinAlg.solve_fullcolumn(field, mp, mi)

            if check
                dp = dim(Eproj, t)
                di = dim(Einj, t)
                if dp != di
                    error("comparison: dim mismatch at t=$(t): projective=$(dp), injective=$(di)")
                end
                I_d = Matrix{K}(I, dp, dp)
                if I2P[t+1] * P2I[t+1] != I_d
                    error("comparison: I2P*P2I != identity at t=$(t)")
                end
                if P2I[t+1] * I2P[t+1] != I_d
                    error("comparison: P2I*I2P != identity at t=$(t)")
                end
            end
        end

        return P2I, I2P
    end

    """
        ExtSpace(M, N, df::DerivedFunctorOptions; check=true)

    A model-independent Ext object that contains BOTH:

    - the projective-resolution model of Ext(M,N), and
    - the injective-resolution model of Ext(M,N),

    together with explicit comparison isomorphisms between them.

    The field df.maxdeg sets the truncation degree (0 <= t <= df.maxdeg).
    The field df.canon chooses which coordinate basis is treated as canonical in the unified object:
    :projective or :injective (or :auto as alias for :projective).
    """
    mutable struct _ExtComparisonCache{K}
        P2I::Vector{Union{Nothing,Matrix{K}}}
        I2P::Vector{Union{Nothing,Matrix{K}}}
        check::Bool
        complete::Bool
    end

    mutable struct ExtSpace{K,QT<:AbstractPoset}
        M::PModule{K}
        N::PModule{K}
        Eproj::Union{Nothing,ExtSpaceProjective{K,QT}}
        Einj::Union{Nothing,ExtSpaceInjective{K}}
        comparison::_ExtComparisonCache{K}
        canon::Symbol
        tmin::Int
        tmax::Int
        threads::Bool
        cache::Union{Nothing,ResolutionCache}
    end

    struct _ExtComparisonFamily{K,QT<:AbstractPoset}
        E::ExtSpace{K,QT}
        projective_to_injective::Bool
    end

    Base.IndexStyle(::Type{<:_ExtComparisonFamily}) = IndexLinear()
    Base.eltype(::Type{_ExtComparisonFamily{K,QT}}) where {K,QT} = Matrix{K}
    Base.length(F::_ExtComparisonFamily) = F.E.tmax - F.E.tmin + 1
    Base.size(F::_ExtComparisonFamily) = (length(F),)

    @inline function _new_ext_comparison_cache(::Type{K}, maxdeg::Int, check::Bool) where {K}
        slots = Union{Nothing,Matrix{K}}[nothing for _ in 0:maxdeg]
        return _ExtComparisonCache{K}(copy(slots), slots, check, false)
    end

    @inline function _ensure_ext_comparison_degree!(E::ExtSpace{K,QT}, t::Int) where {K,QT}
        cmp = E.comparison
        p2i = cmp.P2I[t + 1]
        i2p = cmp.I2P[t + 1]
        if p2i === nothing || i2p === nothing
            p2i_t, i2p_t = _comparison_projective_injective_degree(
                _ensure_ext_projective!(E),
                _ensure_ext_injective!(E),
                t;
                check=cmp.check,
            )
            cmp.P2I[t + 1] = p2i_t
            cmp.I2P[t + 1] = i2p_t
            p2i = p2i_t
            i2p = i2p_t
            cmp.complete = all(!isnothing, cmp.P2I) && all(!isnothing, cmp.I2P)
        end
        return (p2i::Matrix{K}, i2p::Matrix{K})
    end

    @inline function _ensure_ext_projective!(E::ExtSpace{K,QT}) where {K,QT}
        proj = E.Eproj
        if proj === nothing
            proj = _Ext_projective(E.M, E.N; maxdeg=E.tmax, threads=E.threads, cache=E.cache)
            E.Eproj = proj
        end
        return proj::ExtSpaceProjective{K,QT}
    end

    @inline function _ensure_ext_injective!(E::ExtSpace{K}) where {K}
        inj = E.Einj
        if inj === nothing
            df = DerivedFunctorOptions(maxdeg=E.tmax, model=:injective, canon=E.canon)
            inj = ExtInjective(E.M, E.N, df; cache=E.cache)
            E.Einj = inj
        end
        return inj::ExtSpaceInjective{K}
    end

    @inline function _ensure_ext_comparison!(E::ExtSpace{K}) where {K}
        cmp = E.comparison
        if !cmp.complete
            P2I, I2P = _comparison_projective_injective(
                _ensure_ext_projective!(E),
                _ensure_ext_injective!(E);
                maxdeg=E.tmax,
                check=cmp.check,
            )
            cmp.P2I = P2I
            cmp.I2P = I2P
            cmp.complete = true
        end
        return (
            cmp.P2I::Vector{Union{Nothing,Matrix{K}}},
            cmp.I2P::Vector{Union{Nothing,Matrix{K}}},
        )
    end

    @inline _comparison_P2I(E::ExtSpace{K,QT}, t::Int) where {K,QT} = _ensure_ext_comparison_degree!(E, t)[1]
    @inline _comparison_I2P(E::ExtSpace{K,QT}, t::Int) where {K,QT} = _ensure_ext_comparison_degree!(E, t)[2]

    function Base.getindex(F::_ExtComparisonFamily{K,QT}, i::Int) where {K,QT}
        @boundscheck 1 <= i <= length(F) || throw(BoundsError(F, i))
        t = F.E.tmin + i - 1
        return F.projective_to_injective ? _comparison_P2I(F.E, t) : _comparison_I2P(F.E, t)
    end

    function Base.iterate(F::_ExtComparisonFamily{K,QT}, state::Int=1) where {K,QT}
        state > length(F) && return nothing
        _ensure_ext_comparison!(F.E)
        mats = F.projective_to_injective ? F.E.comparison.P2I : F.E.comparison.I2P
        return ((mats[state]::Matrix{K}), state + 1)
    end


    function Base.show(io::IO, E::ExtSpace{K}) where {K}
        d = ext_summary(E)
        print(io, "ExtSpace(unified; canon=$(E.canon), nonzero_degrees=", repr(d.nonzero_degrees), ")")
    end

    function Base.show(io::IO, ::MIME"text/plain", E::ExtSpace)
        d = ext_summary(E)
        print(io, "ExtSpace",
              "\n  model: ", d.model,
              "\n  canon: ", E.canon,
              "\n  field: ", d.field,
              "\n  nvertices: ", d.nvertices,
              "\n  degree_range: ", repr(d.degree_range),
              "\n  nonzero_degrees: ", repr(d.nonzero_degrees),
              "\n  degree_dimensions: ", repr(d.degree_dimensions),
              "\n  total_dimension: ", d.total_dimension,
              "\n  projective_model_ready: ", E.Eproj !== nothing,
              "\n  injective_model_ready: ", E.Einj !== nothing)
    end

    """
        ExtSpace(M, N, df::DerivedFunctorOptions; check=true)

    Return the model-independent ExtSpace containing both the projective and injective models,
    together with explicit comparison isomorphisms.

    - df.maxdeg controls the truncation degree.
    - df.canon selects the canonical coordinate basis in the unified object (:projective or :injective;
    :auto is an alias for :projective).
    - df.model must be :unified or :auto.
    """
    function ExtSpace(M::PModule{K}, N::PModule{K}, df::DerivedFunctorOptions;
                      check::Bool=true,
                      cache::Union{Nothing,ResolutionCache}=nothing) where {K}
        if !(df.model === :auto || df.model === :unified)
            error("ExtSpace: df.model must be :unified or :auto, got $(df.model)")
        end
        maxdeg = df.maxdeg
        canon = df.canon === :auto ? :projective : df.canon
        if !(canon === :projective || canon === :injective)
            error("ExtSpace: df.canon must be :projective or :injective (or :auto), got $(df.canon)")
        end

        threads = Threads.nthreads() > 1
        cache_key = cache === nothing ? nothing : _ext_unified_key(M, N, maxdeg, canon, check)
        if cache_key !== nothing
            cached = _cache_ext_unified_get(cache, cache_key, ExtSpace{K,typeof(M.Q)})
            cached === nothing || return cached
        end
        Eproj = canon === :projective ? _Ext_projective(M, N; maxdeg=maxdeg, threads=threads, cache=cache) : nothing
        Einj = canon === :injective ? ExtInjective(M, N, DerivedFunctorOptions(maxdeg=maxdeg, model=:injective, canon=canon); cache=cache) : nothing
        cmp = _new_ext_comparison_cache(K, maxdeg, check)
        E = ExtSpace{K,typeof(M.Q)}(M, N, Eproj, Einj, cmp, canon, 0, maxdeg, threads, cache)
        return cache_key === nothing ? E : _cache_ext_unified_store!(cache, cache_key, E)
    end


    # Accessors for the two realizations
    projective_model(E::ExtSpace{K}) where {K} = _ensure_ext_projective!(E)
    injective_model(E::ExtSpace{K}) where {K}  = _ensure_ext_injective!(E)
    comparison_isomorphisms(E::ExtSpace{K,QT}) where {K,QT} =
        (_ExtComparisonFamily{K,QT}(E, true), _ExtComparisonFamily{K,QT}(E, false))

    """
        comparison_isomorphism(E, t; from=:projective, to=:injective)

    Return the explicit basis-transport matrix in degree t.

    Allowed model symbols: :projective, :injective, :canonical.
    """
    function comparison_isomorphism(
        E::ExtSpace{K},
        t::Int;
        from::Symbol=:projective,
        to::Symbol=:injective
    ) where {K}
        @assert 0 <= t <= E.tmax

        if from === :canonical
            from = E.canon
        end
        if to === :canonical
            to = E.canon
        end

        if from === to
            d = dim(E, t)
            return Matrix{K}(I, d, d)
        end

        if from === :projective && to === :injective
            return _comparison_P2I(E, t)
        elseif from === :injective && to === :projective
            return _comparison_I2P(E, t)
        else
            error("comparison_isomorphism: from=$(from), to=$(to) not supported")
        end
    end

    function dim(E::ExtSpace{K}, t::Int) where {K}
        if E.Eproj !== nothing
            return dim(E.Eproj, t)
        elseif E.Einj !== nothing
            return dim(E.Einj, t)
        end
        return dim(_ensure_ext_projective!(E), t)
    end

    """
        representative(E, t, coords; model=:canonical)

    Return a cocycle representative of the Ext class given by `coords` (in the CANONICAL basis).

    The keyword `model` chooses the computational realization in which the cocycle is returned:
    - model = :canonical (default) returns a cocycle in the canonical model chosen by E.canon
    - model = :projective returns a cocycle in Hom(P_t, N)
    - model = :injective  returns a cocycle in Hom(M, E^t)

    The coordinates are always interpreted in the canonical basis.
    """
    function representative(
        E::ExtSpace{K},
        t::Int,
        coords::AbstractVector{K};
        model::Symbol=:canonical
    ) where {K}
        @assert 0 <= t <= E.tmax
        if model === :canonical
            model = E.canon
        end

        if model === :projective
            if E.canon === :projective
                return representative(_ensure_ext_projective!(E), t, coords)
            else
                # coords are canonical injective; convert to projective
                coordsP = _comparison_I2P(E, t) * coords
                return representative(_ensure_ext_projective!(E), t, coordsP)
            end
        elseif model === :injective
            if E.canon === :injective
                return representative(_ensure_ext_injective!(E), t, coords)
            else
                # coords are canonical projective; convert to injective
                coordsI = _comparison_P2I(E, t) * coords
                return representative(_ensure_ext_injective!(E), t, coordsI)
            end
        else
            error("representative(::ExtSpace): model must be :projective, :injective, or :canonical")
        end
    end

    """
        coordinates(E, t, cocycle; model=:canonical)

    Compute the coordinates (in the CANONICAL basis) of a cocycle representative.

    The keyword `model` specifies which cochain complex the cocycle lives in:
    - model = :canonical means the cocycle is in E.canon's complex
    - model = :projective means cocycle is in Hom(P_t, N)
    - model = :injective means cocycle is in Hom(M, E^t)
    """
    function coordinates(
        E::ExtSpace{K},
        t::Int,
        cocycle::AbstractVector{K};
        model::Symbol=:canonical
    ) where {K}
        @assert 0 <= t <= E.tmax
        if model === :canonical
            model = E.canon
        end

        if model === :projective
            coordsP = coordinates(_ensure_ext_projective!(E), t, cocycle)
            if E.canon === :projective
                return coordsP
            else
                return _comparison_P2I(E, t) * coordsP
            end
        elseif model === :injective
            coordsI = coordinates(_ensure_ext_injective!(E), t, cocycle)
            if E.canon === :injective
                return coordsI
            else
                return _comparison_I2P(E, t) * coordsI
            end
        else
            error("coordinates(::ExtSpace): model must be :projective, :injective, or :canonical")
        end
    end

    """
        basis(E, t; model=:canonical)

    Return a basis of Ext^t(M,N) as cocycle representatives in the requested model,
    ordered by the canonical coordinate basis.

    This is the "coherent basis transport" API: the basis vectors correspond to the
    same Ext classes regardless of which model you ask for.
    """
    function basis(E::ExtSpace{K}, t::Int; model::Symbol=:canonical) where {K}
        d = dim(E, t)
        out = Vector{Vector{K}}(undef, d)
        for i in 1:d
            e = zeros(K, d)
            e[i] = one(K)
            out[i] = representative(E, t, e; model=model)
        end
        return out
    end

    # ----------------------------
    # Tor for right module (as left module over opposite poset) vs left module
    # ----------------------------

    struct TorSpace{K}
        resRop::ProjectiveResolution{K}    # projective resolution computed on P^op
        L::PModule{K}                      # left module on P
        bd::Vector{SparseMatrixCSC{K, Int}}  # boundaries bd_s : C_s -> C_{s-1}, s=1..S
        dims::Vector{Int}                    # dim C_s for s=0..S
        offsets::Vector{Vector{Int}}         # offsets per degree
        homol::Vector{ChainComplexes.HomologyData{K}}  # homology data per degree
    end

    # NOTE: The maximum computed Tor degree is (length(T.dims) - 1).

    function _op_poset(P::AbstractPoset)
        leq = transpose(leq_matrix(P))
        return FinitePoset(leq; check=false)
    end

    # ----------------------------------------------------------------------
    # Alternative Tor model: resolve the second argument.
    # ----------------------------------------------------------------------

    """
        TorSpaceSecond

    A Tor computation obtained by resolving the *second* argument L (a P-module) and
    tensoring that projective resolution with the fixed right module Rop (a P^op-module).

    Chain groups:
        C_s = oplus_{u in gens_s(resL)} Rop_u

    The boundary matrices are built from the resolution differentials and the structure
    maps of Rop.

    This model is especially convenient for:
    - functoriality and long exact sequences in the *first* argument (Rop),
    - cap/actions of Ext(L,L) on Tor(Rop,L), where chain-level maps live on the L-resolution.
    """
    struct TorSpaceSecond{K}
        resL::ProjectiveResolution{K}
        Rop::PModule{K}
        bd::Vector{SparseMatrixCSC{K, Int64}}
        dims::Vector{Int64}
        offsets::Vector{Vector{Int64}}
        homol::Vector{ChainComplexes.HomologyData{K}}
    end

    # NOTE: The maximum computed Tor degree is (length(T.dims) - 1).

    # Small helper used for defensive checks when the user supplies a precomputed resolution.
    # (We avoid requiring object identity `===` and instead check structural equality.)
    function _same_pmodule(M::PModule{K}, N::PModule{K}) where {K}
        return poset_equal(M.Q, N.Q) && (M.dims == N.dims) && (M.edge_maps == N.edge_maps)
    end

    # Internal A/B knob for Tor boundary assembly. The triplet path avoids
    # sparse slice assignment and reuses per-call buffers across degrees.
    const _TOR_USE_TRIPLET_BOUNDARY_ASSEMBLY = Ref(true)
    # Internal A/B knob for direct Tor triplet assembly. When enabled, Tor
    # appends sparse triplets directly from a cached `map_leq_many` plan
    # instead of materializing a `map_blocks` vector first.
    const _TOR_USE_DIRECT_MAP_TRIPLET_ASSEMBLY = Ref(true)
    # Internal heuristic gates for the direct triplet route. The direct path is
    # best on small Tor assembly plans, but medium bicomplex blocks regress once
    # the coefficient plan and block footprint get larger.
    const _TOR_DIRECT_MAP_TRIPLET_MAX_PLAN_NNZ = Ref(64)
    const _TOR_DIRECT_MAP_TRIPLET_MAX_BLOCK_WORK = Ref(12_000)
    const _TOR_DIRECT_MAP_TRIPLET_MAX_BICOMPLEX_PLAN_NNZ = Ref(32)
    const _TOR_DIRECT_MAP_TRIPLET_MAX_BICOMPLEX_BLOCK_WORK = Ref(4_000)

    mutable struct _TorBoundaryWorkspace{K,MatT<:AbstractMatrix{K}}
        pairs::Vector{Tuple{Int,Int}}
        map_blocks::Vector{MatT}
        I::Vector{Int}
        J::Vector{Int}
        V::Vector{K}
    end

    struct _TorCoeffPlan{K}
        rows::Vector{Int}
        cols::Vector{Int}
        vals::Vector{K}
        batch::MapLeqQueryBatch
    end

    @inline function _tor_boundary_workspace(M::PModule{K,F,MatT}) where {K,F,MatT<:AbstractMatrix{K}}
        return _TorBoundaryWorkspace{K,MatT}(Tuple{Int,Int}[], Vector{MatT}(), Int[], Int[], K[])
    end

    @inline function _tor_direct_triplet_work(plan::_TorCoeffPlan,
                                              cod_offs::AbstractVector{<:Integer},
                                              dom_offs::AbstractVector{<:Integer})
        work = 0
        @inbounds for k in eachindex(plan.rows)
            row = plan.rows[k]
            col = plan.cols[k]
            work += (Int(cod_offs[row + 1]) - Int(cod_offs[row])) *
                    (Int(dom_offs[col + 1]) - Int(dom_offs[col]))
        end
        return work
    end

    @inline function _tor_use_direct_triplets(plan::_TorCoeffPlan,
                                              cod_offs::AbstractVector{<:Integer},
                                              dom_offs::AbstractVector{<:Integer};
                                              bicomplex::Bool=false)
        _TOR_USE_DIRECT_MAP_TRIPLET_ASSEMBLY[] || return false
        nnz_plan = length(plan.rows)
        max_nnz = bicomplex ? _TOR_DIRECT_MAP_TRIPLET_MAX_BICOMPLEX_PLAN_NNZ[] :
                              _TOR_DIRECT_MAP_TRIPLET_MAX_PLAN_NNZ[]
        nnz_plan <= max_nnz || return false
        max_work = bicomplex ? _TOR_DIRECT_MAP_TRIPLET_MAX_BICOMPLEX_BLOCK_WORK[] :
                               _TOR_DIRECT_MAP_TRIPLET_MAX_BLOCK_WORK[]
        return _tor_direct_triplet_work(plan, cod_offs, dom_offs) <= max_work
    end

    function _tor_coeff_plan(delta::SparseMatrixCSC{K,Int},
                             dom_bases::AbstractVector{<:Integer},
                             cod_bases::AbstractVector{<:Integer}) where {K}
        nnz_delta = nnz(delta)
        rows = Vector{Int}(undef, nnz_delta)
        cols = Vector{Int}(undef, nnz_delta)
        vals = Vector{K}(undef, nnz_delta)
        pairs = Vector{Tuple{Int,Int}}(undef, nnz_delta)
        rows_delta = SparseArrays.rowvals(delta)
        vals_delta = nonzeros(delta)

        idx = 1
        @inbounds for col in 1:size(delta, 2)
            u = Int(dom_bases[col])
            for ptr in nzrange(delta, col)
                row = rows_delta[ptr]
                rows[idx] = row
                cols[idx] = col
                vals[idx] = vals_delta[ptr]
                pairs[idx] = (u, Int(cod_bases[row]))
                idx += 1
            end
        end
        return _TorCoeffPlan{K}(rows, cols, vals, _prepare_map_leq_batch_owned(pairs))
    end

    function _tor_boundary_denseassign(Mmod::PModule{K}, plan::_TorCoeffPlan{K},
                                       cod_offs::AbstractVector{<:Integer},
                                       dom_offs::AbstractVector{<:Integer},
                                       out_dim::Int, in_dim::Int) where {K}
        B = spzeros(K, out_dim, in_dim)
        map_blocks = _map_blocks_buffer(Mmod, length(plan.rows))
        map_leq_many!(map_blocks, Mmod, plan.batch)
        @inbounds for k in eachindex(plan.rows)
            j = plan.rows[k]
            i = plan.cols[k]
            c = plan.vals[k]
            iszero(c) && continue
            rows = (Int(cod_offs[j]) + 1):Int(cod_offs[j + 1])
            cols = (Int(dom_offs[i]) + 1):Int(dom_offs[i + 1])
            B[rows, cols] = c * map_blocks[k]
        end
        return B
    end

    function _tor_boundary_triplets!(ws::_TorBoundaryWorkspace{K,MatT},
                                     Mmod::PModule{K,F,MatT},
                                     plan::_TorCoeffPlan{K},
                                     cod_offs::AbstractVector{<:Integer},
                                     dom_offs::AbstractVector{<:Integer},
                                     out_dim::Int, in_dim::Int) where {K,F,MatT<:AbstractMatrix{K}}
        nnz_delta = length(plan.rows)
        nnz_delta == 0 && return spzeros(K, out_dim, in_dim)

        empty!(ws.I)
        empty!(ws.J)
        empty!(ws.V)

        if _tor_use_direct_triplets(plan, cod_offs, dom_offs)
            _append_map_leq_many_scaled_triplets!(ws.I, ws.J, ws.V, Mmod, plan.batch,
                                                  plan.rows, plan.cols, cod_offs, dom_offs, plan.vals)
            return sparse(ws.I, ws.J, ws.V, out_dim, in_dim)
        end

        resize!(ws.map_blocks, nnz_delta)
        map_leq_many!(ws.map_blocks, Mmod, plan.batch)

        @inbounds for k in eachindex(plan.rows)
            row = plan.rows[k]
            col = plan.cols[k]
            c = plan.vals[k]
            iszero(c) && continue
            _append_scaled_triplets!(ws.I, ws.J, ws.V, ws.map_blocks[k],
                                     Int(cod_offs[row]), Int(dom_offs[col]); scale=c)
        end
        return sparse(ws.I, ws.J, ws.V, out_dim, in_dim)
    end

    function _tor_boundary_matrix(Mmod::PModule{K,F,MatT},
                                  plan::_TorCoeffPlan{K},
                                  cod_offs::AbstractVector{<:Integer},
                                  dom_offs::AbstractVector{<:Integer},
                                  out_dim::Int, in_dim::Int,
                                  ws::_TorBoundaryWorkspace{K,MatT}) where {K,F,MatT<:AbstractMatrix{K}}
        if _TOR_USE_TRIPLET_BOUNDARY_ASSEMBLY[]
            return _tor_boundary_triplets!(ws, Mmod, plan, cod_offs, dom_offs, out_dim, in_dim)
        end
        return _tor_boundary_denseassign(Mmod, plan, cod_offs, dom_offs, out_dim, in_dim)
    end

    # Internal implementation: resolve the first argument (existing Tor behavior).
    function _Tor_resolve_first(Rop::PModule{K}, L::PModule{K};
                                maxdeg::Int=3,
                                threads::Bool=(Threads.nthreads() > 1),
                                res::Union{Nothing, ProjectiveResolution{K}}=nothing) where {K}
        Pop = Rop.Q
        P = _op_poset(Pop)
        @assert poset_equal(L.Q, P)

        # Projective resolution of Rop as a Pop module.
        if res === nothing
            res = projective_resolution(Rop, ResolutionOptions(maxlen=maxdeg); threads=threads)
        end
        S = length(res.Pmods) - 1

        # Chain group dims and block offsets.
        dims = Int64[]
        offs = Vector{Vector{Int64}}()
        for s in 0:S
            gens_s = res.gens[s + 1]
            os = zeros(Int64, length(gens_s) + 1)
            for i in 1:length(gens_s)
                u = gens_s[i]
                os[i + 1] = os[i] + L.dims[u]
            end
            push!(offs, os)
            push!(dims, os[end])
        end

        # Boundary matrices C_s -> C_{s-1}.
        bd = Vector{SparseMatrixCSC{K, Int64}}(undef, S)
        workspaces = [_tor_boundary_workspace(L) for _ in 1:Threads.maxthreadid()]
        if threads && Threads.nthreads() > 1 && S >= 2
            Threads.@threads for s in 1:S
                dom_bases = res.gens[s + 1]
                cod_bases = res.gens[s]
                delta = res.d_mat[s]   # rows=cod, cols=dom
                ws = workspaces[Threads.threadid()]
                plan = _tor_coeff_plan(delta, dom_bases, cod_bases)
                bd[s] = _tor_boundary_matrix(L, plan,
                                             offs[s], offs[s + 1], dims[s], dims[s + 1], ws)
            end
        else
            ws = workspaces[1]
            for s in 1:S
                dom_bases = res.gens[s + 1]
                cod_bases = res.gens[s]
                delta = res.d_mat[s]   # rows=cod, cols=dom
                plan = _tor_coeff_plan(delta, dom_bases, cod_bases)
                bd[s] = _tor_boundary_matrix(L, plan,
                                             offs[s], offs[s + 1], dims[s], dims[s + 1], ws)
            end
        end

        # Homology data per degree.
        homol = Vector{ChainComplexes.HomologyData{K}}(undef, S + 1)
        empty_left = _empty_dense_row_cache(K, dims)
        empty_right = _empty_dense_col_cache(K, dims)
        if threads && Threads.nthreads() > 1 && S >= 1
            Threads.@threads for s in 0:S
                bd_curr = (s == 0) ? empty_left[s + 1] : bd[s]
                bd_next = (s == S) ? empty_right[s + 1] : bd[s + 1]
                homol[s + 1] = ChainComplexes.homology_data(bd_next, bd_curr, s)
            end
        else
            for s in 0:S
                bd_curr = (s == 0) ? empty_left[s + 1] : bd[s]
                bd_next = (s == S) ? empty_right[s + 1] : bd[s + 1]
                homol[s + 1] = ChainComplexes.homology_data(bd_next, bd_curr, s)
            end
        end

        return TorSpace{K}(res, L, bd, dims, offs, homol)
    end

    # Internal implementation: resolve the second argument.
    function _Tor_resolve_second(Rop::PModule{K}, L::PModule{K};
                                maxdeg::Int=3,
                                threads::Bool=(Threads.nthreads() > 1),
                                res::Union{Nothing, ProjectiveResolution{K}}=nothing) where {K}
        Pop = Rop.Q
        P = _op_poset(Pop)
        @assert poset_equal(L.Q, P)

        # Projective resolution of L as a P module.
        resL = (res === nothing) ? projective_resolution(L, ResolutionOptions(maxlen=maxdeg); threads=threads) : res
        S = length(resL.Pmods) - 1

        # Chain group dims and block offsets: C_s = oplus Rop_u.
        dims = Int64[]
        offs = Vector{Vector{Int64}}()
        for s in 0:S
            gens_s = resL.gens[s + 1]
            os = zeros(Int64, length(gens_s) + 1)
            for i in 1:length(gens_s)
                u = gens_s[i]
                os[i + 1] = os[i] + Rop.dims[u]
            end
            push!(offs, os)
            push!(dims, os[end])
        end

        # Boundary matrices C_s -> C_{s-1}.
        bd = Vector{SparseMatrixCSC{K, Int64}}(undef, S)
        workspaces = [_tor_boundary_workspace(Rop) for _ in 1:Threads.maxthreadid()]
        if threads && Threads.nthreads() > 1 && S >= 2
            Threads.@threads for s in 1:S
                dom_bases = resL.gens[s + 1]
                cod_bases = resL.gens[s]
                delta = resL.d_mat[s]   # rows=cod, cols=dom
                ws = workspaces[Threads.threadid()]
                plan = _tor_coeff_plan(delta, dom_bases, cod_bases)
                bd[s] = _tor_boundary_matrix(Rop, plan,
                                             offs[s], offs[s + 1], dims[s], dims[s + 1], ws)
            end
        else
            ws = workspaces[1]
            for s in 1:S
                dom_bases = resL.gens[s + 1]
                cod_bases = resL.gens[s]
                delta = resL.d_mat[s]   # rows=cod, cols=dom
                plan = _tor_coeff_plan(delta, dom_bases, cod_bases)
                bd[s] = _tor_boundary_matrix(Rop, plan,
                                             offs[s], offs[s + 1], dims[s], dims[s + 1], ws)
            end
        end

        # Homology data per degree.
        homol = Vector{ChainComplexes.HomologyData{K}}(undef, S + 1)
        empty_left = _empty_dense_row_cache(K, dims)
        empty_right = _empty_dense_col_cache(K, dims)
        if threads && Threads.nthreads() > 1 && S >= 1
            Threads.@threads for s in 0:S
                bd_curr = (s == 0) ? empty_left[s + 1] : bd[s]
                bd_next = (s == S) ? empty_right[s + 1] : bd[s + 1]
                homol[s + 1] = ChainComplexes.homology_data(bd_next, bd_curr, s)
            end
        else
            for s in 0:S
                bd_curr = (s == 0) ? empty_left[s + 1] : bd[s]
                bd_next = (s == S) ? empty_right[s + 1] : bd[s + 1]
                homol[s + 1] = ChainComplexes.homology_data(bd_next, bd_curr, s)
            end
        end

        return TorSpaceSecond{K}(resL, Rop, bd, dims, offs, homol)
    end

    """
        Tor(Rop, L, df::DerivedFunctorOptions; res=nothing)

    Compute Tor_s(Rop, L) for 0 <= s <= df.maxdeg.

    - Rop is a P^op-module (right module over P).
    - L is a P-module.

    df.model selects the computational model for Tor:
    - :first (or :auto): resolve Rop and tensor with L.
    - :second: resolve L and tensor with Rop.

    Cheap-first workflow
    - Start with `tor_summary(T)`, `nonzero_degrees(T)`, and
      `degree_dimensions(T)`.
    - Ask for `basis(T, s)` or `representative(T, s, ...)` only when you need
      explicit chain-level representatives.

    You may optionally supply a projective resolution via keyword `res`.
    If supplied, df.maxdeg is ignored and the maximum computed degree is determined by the length of `res`.
    """
    function Tor(Rop::PModule{K}, L::PModule{K}, df::DerivedFunctorOptions;
                 res=nothing,
                 cache::Union{Nothing,ResolutionCache}=nothing) where {K}
        model = df.model === :auto ? :first : df.model
        cache_key = cache === nothing || res !== nothing ? nothing : _resolution_key3(Rop, L, df.maxdeg)

        if res !== nothing && !(res isa ProjectiveResolution{K})
            error("Tor: res must be ProjectiveResolution{K} or nothing, got $(typeof(res))")
        end

        if model === :first
            if cache_key !== nothing
                cached = _cache_tor_first_get(cache, cache_key, TorSpace{K})
                cached === nothing || return cached
            end
            if res !== nothing
                @assert _same_pmodule(res.M, Rop)
            end
            if res === nothing && cache !== nothing
                res = projective_resolution(Rop, ResolutionOptions(maxlen=df.maxdeg); cache=cache)
            end
            T = _Tor_resolve_first(Rop, L; maxdeg=df.maxdeg, threads=(Threads.nthreads() > 1), res=res)
            return cache_key === nothing ? T : _cache_tor_first_store!(cache, cache_key, T)
        elseif model === :second
            if cache_key !== nothing
                cached = _cache_tor_second_get(cache, cache_key, TorSpaceSecond{K})
                cached === nothing || return cached
            end
            if res !== nothing
                @assert _same_pmodule(res.M, L)
            end
            if res === nothing && cache !== nothing
                res = projective_resolution(L, ResolutionOptions(maxlen=df.maxdeg); cache=cache)
            end
            T = _Tor_resolve_second(Rop, L; maxdeg=df.maxdeg, threads=(Threads.nthreads() > 1), res=res)
            return cache_key === nothing ? T : _cache_tor_second_store!(cache, cache_key, T)
        else
            error("Tor: unknown df.model=$(df.model). Supported for Tor: :first, :second, :auto.")
        end
    end

    # -------------------------------------------------------------------------
    # Graded-space API for Tor objects
    # -------------------------------------------------------------------------

    """
        dim(T::TorSpace, s::Int) -> Int
    """
    dim(T::TorSpace, s::Int) = T.homol[s + 1].dimH

    """
        cycles(T::TorSpace, s::Int) -> Matrix
    """
    cycles(T::TorSpace, s::Int) = T.homol[s + 1].Z

    """
        boundaries(T::TorSpace, s::Int) -> Matrix
    """
    boundaries(T::TorSpace, s::Int) = T.homol[s + 1].B

    """
        basis(T::TorSpace, s::Int) -> Vector{Vector}

    List of cycle representatives for a basis of Tor_s.
    """
    function basis(T::TorSpace{K}, s::Int) where {K}
        Hrep = T.homol[s + 1].Hrep
        out = Vector{Vector{K}}(undef, size(Hrep, 2))
        for i in 1:size(Hrep, 2)
            out[i] = Vector{K}(Hrep[:, i])
        end
        return out
    end

    """
        representative(T::TorSpace, s::Int, coords::AbstractVector) -> Vector

    Chain-level representative of the Tor class with coordinates `coords`.
    """
    function representative(T::TorSpace{K}, s::Int, coords::AbstractVector) where {K}
        data = T.homol[s + 1]
        KT = eltype(data.Hrep)
        c = Vector{KT}(coords)
        size(data.Hrep, 2) == length(c) || throw(DimensionMismatch(
            "representative(TorSpace): expected coords of length $(size(data.Hrep,2)), got $(length(c))"
        ))
        return data.Hrep * c
    end

    """
        coordinates(T::TorSpace, s::Int, z::AbstractVector) -> Vector

    Homology coordinates of a cycle `z`. Uses ChainComplexes.homology_coordinates.
    """
    function coordinates(T::TorSpace{K}, s::Int, z::AbstractVector) where {K}
        data = T.homol[s + 1]
        KT = eltype(data.Hrep)
        zvec = Vector{KT}(z)
        c = ChainComplexes.homology_coordinates(data, zvec)
        return vec(c)
    end

    # -------------------------------------------------------------------------
    # Resolve-second Tor object (TorSpaceSecond)
    # -------------------------------------------------------------------------

    dim(T::TorSpaceSecond, s::Int) = T.homol[s + 1].dimH

    # BUGFIX: homology_data returns HomologyData, which has fields Z and B, not Zrep/Brep.
    cycles(T::TorSpaceSecond, s::Int) = T.homol[s + 1].Z
    boundaries(T::TorSpaceSecond, s::Int) = T.homol[s + 1].B

    """
        basis(T::TorSpaceSecond, s::Int) -> Vector{Vector}

    List of cycle representatives for a basis of Tor_s (resolve-second model).
    """
    function basis(T::TorSpaceSecond{K}, s::Int) where {K}
        Hrep = T.homol[s + 1].Hrep
        out = Vector{Vector{K}}(undef, size(Hrep, 2))
        for i in 1:size(Hrep, 2)
            out[i] = Vector{K}(Hrep[:, i])
        end
        return out
    end

    function representative(T::TorSpaceSecond{K}, s::Int, coords::AbstractVector) where {K}
        data = T.homol[s + 1]
        KT = eltype(data.Hrep)
        c = Vector{KT}(coords)
        size(data.Hrep, 2) == length(c) || throw(DimensionMismatch(
            "representative(TorSpaceSecond): expected coords of length $(size(data.Hrep,2)), got $(length(c))"
        ))
        return data.Hrep * c
    end

    function coordinates(T::TorSpaceSecond{K}, s::Int, z::AbstractVector) where {K}
        data = T.homol[s + 1]
        KT = eltype(data.Hrep)
        zvec = Vector{KT}(z)
        c = ChainComplexes.homology_coordinates(data, zvec)
        return vec(c)
    end


        """
        degree_range(E::ExtSpaceProjective) -> UnitRange{Int}
    """
    degree_range(E::ExtSpaceProjective) = E.tmin:E.tmax

    """
        degree_range(E::ExtSpaceInjective) -> UnitRange{Int}
    """
    degree_range(E::ExtSpaceInjective) = E.tmin:E.tmax

    """
        degree_range(E::ExtSpace) -> UnitRange{Int}
    """
    degree_range(E::ExtSpace) = E.tmin:E.tmax

    """
        degree_range(T::TorSpace) -> UnitRange{Int}
    """
    degree_range(T::TorSpace) = 0:(length(T.dims) - 1)

    """
        degree_range(T::TorSpaceSecond) -> UnitRange{Int}
    """
    degree_range(T::TorSpaceSecond) = 0:(length(T.dims) - 1)


        """
        degree_range(H::HomSpace) -> UnitRange{Int}

    HomSpace is ungraded, viewed as concentrated in degree 0.
    """
    degree_range(H::HomSpace) = 0:0

    """
        dim(H::HomSpace, t::Int) -> Int

    Only `t == 0` is valid.
    """
    function dim(H::HomSpace, t::Int)
        t == 0 || throw(DomainError(t, "HomSpace is concentrated in degree 0"))
        return dim(H)
    end

    """
        basis(H::HomSpace, t::Int)

    Only `t == 0` is valid.
    """
    function basis(H::HomSpace, t::Int)
        t == 0 || throw(DomainError(t, "HomSpace is concentrated in degree 0"))
        return basis(H)
    end

    @inline source_module(H::HomSpace) = H.dom
    @inline target_module(H::HomSpace) = H.cod

    @inline source_module(E::ExtSpaceProjective) = E.M
    @inline target_module(E::ExtSpaceProjective) = E.N

    @inline source_module(E::ExtSpaceInjective) = E.M
    @inline target_module(E::ExtSpaceInjective) = E.N

    @inline source_module(E::ExtSpace) = E.M
    @inline target_module(E::ExtSpace) = E.N

    @inline source_module(T::TorSpace) = T.resRop.M
    @inline target_module(T::TorSpace) = T.L

    @inline source_module(T::TorSpaceSecond) = T.Rop
    @inline target_module(T::TorSpaceSecond) = T.resL.M

    @inline function _nonzero_degrees_impl(space)
        return [t for t in degree_range(space) if dim(space, t) != 0]
    end

    @inline function _degree_dimensions_impl(space)
        out = Dict{Int,Int}()
        for t in degree_range(space)
            d = dim(space, t)
            d == 0 || (out[t] = d)
        end
        return out
    end

    @inline nonzero_degrees(H::HomSpace) = dim(H) == 0 ? Int[] : [0]
    @inline degree_dimensions(H::HomSpace) = dim(H) == 0 ? Dict{Int,Int}() : Dict(0 => dim(H))

    @inline nonzero_degrees(E::ExtSpaceProjective) = _nonzero_degrees_impl(E)
    @inline degree_dimensions(E::ExtSpaceProjective) = _degree_dimensions_impl(E)

    @inline nonzero_degrees(E::ExtSpaceInjective) = _nonzero_degrees_impl(E)
    @inline degree_dimensions(E::ExtSpaceInjective) = _degree_dimensions_impl(E)

    @inline nonzero_degrees(E::ExtSpace) = _nonzero_degrees_impl(E)
    @inline degree_dimensions(E::ExtSpace) = _degree_dimensions_impl(E)

    @inline nonzero_degrees(T::TorSpace) = _nonzero_degrees_impl(T)
    @inline degree_dimensions(T::TorSpace) = _degree_dimensions_impl(T)

    @inline nonzero_degrees(T::TorSpaceSecond) = _nonzero_degrees_impl(T)
    @inline degree_dimensions(T::TorSpaceSecond) = _degree_dimensions_impl(T)

    @inline total_dimension(H::HomSpace) = dim(H)
    @inline total_dimension(E::ExtSpaceProjective) = sum(values(degree_dimensions(E)))
    @inline total_dimension(E::ExtSpaceInjective) = sum(values(degree_dimensions(E)))
    @inline total_dimension(E::ExtSpace) = sum(values(degree_dimensions(E)))
    @inline total_dimension(T::TorSpace) = sum(values(degree_dimensions(T)))
    @inline total_dimension(T::TorSpaceSecond) = sum(values(degree_dimensions(T)))

    """
        hom_summary(H) -> NamedTuple

    Cheap-first summary of a `HomSpace`.

    This reports the field, ambient poset size, total source/target fiber
    dimensions, and whether a basis cache is already populated.

    Start with this, `dim(H)`, or `degree_dimensions(H)` before materializing
    the full basis of morphisms.
    """
    @inline function hom_summary(H::HomSpace)
        return (
            kind=:hom_space,
            field=H.dom.field,
            nvertices=nvertices(H.dom.Q),
            dimension=dim(H),
            degree_range=degree_range(H),
            basis_cached=getfield(H, :basis) !== nothing,
            source_total_dim=sum(H.dom.dims),
            target_total_dim=sum(H.cod.dims),
        )
    end

    @inline function _ext_summary_impl(kind::Symbol, model::Symbol, E)
        return (
            kind=kind,
            model=model,
            field=source_module(E).field,
            nvertices=nvertices(source_module(E).Q),
            degree_range=degree_range(E),
            nonzero_degrees=Tuple(nonzero_degrees(E)),
            degree_dimensions=degree_dimensions(E),
            total_dimension=total_dimension(E),
        )
    end

    """
        ext_summary(E) -> NamedTuple

    Cheap-first summary of an Ext container.

    This reports the chosen model, graded support, and graded dimensions
    without forcing bases, representatives, or comparison isomorphisms.

    Start with this, `nonzero_degrees(E)`, and `degree_dimensions(E)` before
    asking for explicit classes or comparison data.
    """
    @inline ext_summary(E::ExtSpaceProjective) = _ext_summary_impl(:ext_space, :projective, E)
    @inline ext_summary(E::ExtSpaceInjective) = _ext_summary_impl(:ext_space, :injective, E)
    @inline ext_summary(E::ExtSpace) = _ext_summary_impl(:ext_space, :unified, E)

    @inline function _tor_summary_impl(kind::Symbol, model::Symbol, T)
        return (
            kind=kind,
            model=model,
            field=source_module(T).field,
            nvertices=nvertices(source_module(T).Q),
            degree_range=degree_range(T),
            nonzero_degrees=Tuple(nonzero_degrees(T)),
            degree_dimensions=degree_dimensions(T),
            total_dimension=total_dimension(T),
        )
    end

    """
        tor_summary(T) -> NamedTuple

    Cheap-first summary of a Tor container.

    This reports the chosen Tor model, graded support, and graded dimensions
    without forcing representatives or chain-level coordinates.

    Start with this, `nonzero_degrees(T)`, and `degree_dimensions(T)` before
    asking for explicit Tor classes or chain-level data.
    """
    @inline tor_summary(T::TorSpace) = _tor_summary_impl(:tor_space, :first, T)
    @inline tor_summary(T::TorSpaceSecond) = _tor_summary_impl(:tor_space, :second, T)

    """
        underlying_ext_space(E)

    Return the additive Ext-space object underlying `E`.

    For projective, injective, and unified Ext models this returns the object
    itself; the accessor exists so user code can treat thin wrapper objects and
    direct Ext containers uniformly.
    """
    @inline underlying_ext_space(E::ExtSpaceProjective) = E
    @inline underlying_ext_space(E::ExtSpaceInjective) = E
    @inline underlying_ext_space(E::ExtSpace) = E

    function Base.show(io::IO, H::HomSpace)
        d = hom_summary(H)
        print(io, "HomSpace(field=", d.field, ", dimension=", d.dimension, ")")
    end

    function Base.show(io::IO, ::MIME"text/plain", H::HomSpace)
        d = hom_summary(H)
        print(io, "HomSpace",
              "\n  field: ", d.field,
              "\n  nvertices: ", d.nvertices,
              "\n  degree_range: ", repr(d.degree_range),
              "\n  dimension: ", d.dimension,
              "\n  basis_cached: ", d.basis_cached,
              "\n  source_total_dim: ", d.source_total_dim,
              "\n  target_total_dim: ", d.target_total_dim)
    end

    function Base.show(io::IO, E::ExtSpaceProjective)
        d = ext_summary(E)
        print(io, "ExtSpaceProjective(field=", d.field,
              ", nonzero_degrees=", repr(d.nonzero_degrees), ")")
    end

    function Base.show(io::IO, ::MIME"text/plain", E::ExtSpaceProjective)
        d = ext_summary(E)
        print(io, "ExtSpaceProjective",
              "\n  field: ", d.field,
              "\n  nvertices: ", d.nvertices,
              "\n  degree_range: ", repr(d.degree_range),
              "\n  nonzero_degrees: ", repr(d.nonzero_degrees),
              "\n  degree_dimensions: ", repr(d.degree_dimensions),
              "\n  total_dimension: ", d.total_dimension)
    end

    function Base.show(io::IO, E::ExtSpaceInjective)
        d = ext_summary(E)
        print(io, "ExtSpaceInjective(field=", d.field,
              ", nonzero_degrees=", repr(d.nonzero_degrees), ")")
    end

    function Base.show(io::IO, ::MIME"text/plain", E::ExtSpaceInjective)
        d = ext_summary(E)
        print(io, "ExtSpaceInjective",
              "\n  field: ", d.field,
              "\n  nvertices: ", d.nvertices,
              "\n  degree_range: ", repr(d.degree_range),
              "\n  nonzero_degrees: ", repr(d.nonzero_degrees),
              "\n  degree_dimensions: ", repr(d.degree_dimensions),
              "\n  total_dimension: ", d.total_dimension)
    end

    function Base.show(io::IO, T::TorSpace)
        d = tor_summary(T)
        print(io, "TorSpace(model=:first, nonzero_degrees=", repr(d.nonzero_degrees), ")")
    end

    function Base.show(io::IO, ::MIME"text/plain", T::TorSpace)
        d = tor_summary(T)
        print(io, "TorSpace",
              "\n  model: ", d.model,
              "\n  field: ", d.field,
              "\n  nvertices: ", d.nvertices,
              "\n  degree_range: ", repr(d.degree_range),
              "\n  nonzero_degrees: ", repr(d.nonzero_degrees),
              "\n  degree_dimensions: ", repr(d.degree_dimensions),
              "\n  total_dimension: ", d.total_dimension)
    end

    """
        underlying_tor_space(T)

    Return the additive Tor-space object underlying `T`.

    For both first- and second-argument Tor models this returns the object
    itself; the accessor exists so user code can treat Tor containers and Tor
    algebras uniformly.
    """
    @inline underlying_tor_space(T::TorSpace) = T

    function Base.show(io::IO, T::TorSpaceSecond)
        d = tor_summary(T)
        print(io, "TorSpaceSecond(model=:second, nonzero_degrees=", repr(d.nonzero_degrees), ")")
    end

    function Base.show(io::IO, ::MIME"text/plain", T::TorSpaceSecond)
        d = tor_summary(T)
        print(io, "TorSpaceSecond",
              "\n  model: ", d.model,
              "\n  field: ", d.field,
              "\n  nvertices: ", d.nvertices,
              "\n  degree_range: ", repr(d.degree_range),
              "\n  nonzero_degrees: ", repr(d.nonzero_degrees),
              "\n  degree_dimensions: ", repr(d.degree_dimensions),
              "\n  total_dimension: ", d.total_dimension)
    end

    @inline underlying_tor_space(T::TorSpaceSecond) = T

end
