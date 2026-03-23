# spectral_sequences.jl -- double complexes and spectral sequence computations

"""
SpectralSequences: double-complex and page computations.

This submodule is intended to hold:
- Tor bicomplex constructions
- spectral sequence pages and differentials
- convergence and comparison utilities
"""
module SpectralSequences
    using LinearAlgebra
    using SparseArrays

    using ...CoreModules: AbstractCoeffField, RealField, ResolutionCache,
                          ResolutionKey3, ResolutionKey5,
                          ExtDoubleComplexPayload, TorDoubleComplexPlanPayload, TorDoubleComplexPayload,
                          _resolution_key3, _resolution_key5, field_from_eltype
    using ...Options: ResolutionOptions
    import ...CoreModules: _append_scaled_triplets!

    using ...Modules: PModule, MapLeqQueryBatch, map_leq, map_leq_many, map_leq_many!,
                       _prepare_map_leq_batch_owned, _append_map_leq_many_scaled_triplets!
    using ...ChainComplexes
    import ...ChainComplexes: spectral_sequence_summary, convergence_page
    import ...IndicatorResolutions
    using ...IndicatorResolutions: upset_resolution, downset_resolution
    using ...IndicatorTypes: UpsetPresentation, DownsetCopresentation

    import ..HomExtEngine: build_hom_bicomplex_data
    import ..Resolutions: projective_resolution, _pad_projective_resolution!
    import ..ExtTorSpaces: ExtSpaceProjective, ExtSpaceInjective, Ext, ExtInjective, Tor, TorSpace, TorSpaceSecond,
                           _TorCoeffPlan, _tor_coeff_plan, _tor_use_direct_triplets,
                           _TOR_USE_DIRECT_MAP_TRIPLET_ASSEMBLY
    import ..DerivedFunctors: page_dimensions, double_complex_summary, wrapped_spectral_sequence,
                               check_ext_spectral_sequence, check_tor_spectral_sequence,
                               _derived_validation_report, _throw_invalid_derived_functor

    @inline function _map_blocks_buffer(M::PModule{K,F,MatT}, n::Int) where {K,F,MatT<:AbstractMatrix{K}}
        return Vector{MatT}(undef, n)
    end

    const _TOR_DOUBLE_COMPLEX_CACHE_TENSOR_OFFSETS = Ref(true)
    const _TOR_DOUBLE_COMPLEX_CACHE_FASTPATH = Ref(true)

    struct _TorDoubleComplexPlan{K}
        amin::Int
        amax::Int
        bmin::Int
        bmax::Int
        dims::Matrix{Int}
        gens_by_a::Vector{Vector{Int}}
        tensor_offsets::Union{Nothing,Matrix{Vector{Int64}}}
        hplans::Vector{_TorCoeffPlan{K}}
    end

    @inline function _ext_doublecomplex_cache_key(M, N, maxlen::Int)
        return _resolution_key3(M, N, maxlen)
    end

    @inline function _tor_doublecomplex_cache_key(Rop, L, lenR::Int, lenL::Int)
        return _resolution_key5(Rop, L, lenR, lenL, 0)
    end

    @inline function _cache_ext_doublecomplex_get(cache::ResolutionCache, key::ResolutionKey3)
        Base.lock(cache.lock)
        payload = get(cache.ext_doublecomplex, key, nothing)
        Base.unlock(cache.lock)
        return payload === nothing ? nothing : payload.value
    end

    @inline function _cache_ext_doublecomplex_store!(cache::ResolutionCache, key::ResolutionKey3, value)
        Base.lock(cache.lock)
        cache.ext_doublecomplex[key] = ExtDoubleComplexPayload(value)
        Base.unlock(cache.lock)
        return value
    end

    @inline function _cache_tor_doublecomplex_plan_get(cache::ResolutionCache, key::ResolutionKey5)
        Base.lock(cache.lock)
        payload = get(cache.tor_doublecomplex_plan, key, nothing)
        Base.unlock(cache.lock)
        return payload === nothing ? nothing : payload.value
    end

    @inline function _cache_tor_doublecomplex_plan_store!(cache::ResolutionCache, key::ResolutionKey5, value)
        Base.lock(cache.lock)
        cache.tor_doublecomplex_plan[key] = TorDoubleComplexPlanPayload(value)
        Base.unlock(cache.lock)
        return value
    end

    @inline function _cache_tor_doublecomplex_get(cache::ResolutionCache, key::ResolutionKey5)
        if _TOR_DOUBLE_COMPLEX_CACHE_FASTPATH[] && Threads.nthreads() == 1
            payload = get(cache.tor_doublecomplex, key, nothing)
            return payload === nothing ? nothing : payload.value
        end
        Base.lock(cache.lock)
        payload = get(cache.tor_doublecomplex, key, nothing)
        Base.unlock(cache.lock)
        return payload === nothing ? nothing : payload.value
    end

    @inline function _cache_tor_doublecomplex_store!(cache::ResolutionCache, key::ResolutionKey5, value)
        payload = TorDoubleComplexPayload(value)
        if _TOR_DOUBLE_COMPLEX_CACHE_FASTPATH[] && Threads.nthreads() == 1
            cache.tor_doublecomplex[key] = payload
            return value
        end
        Base.lock(cache.lock)
        cache.tor_doublecomplex[key] = payload
        Base.unlock(cache.lock)
        return value
    end

    mutable struct _TorBicomplexTripletWorkspace{K}
        I::Vector{Int}
        J::Vector{Int}
        V::Vector{K}
    end

    @inline _tor_bicomplex_workspace(::Type{K}) where {K} =
        _TorBicomplexTripletWorkspace{K}(Int[], Int[], K[])

    @inline function _tensor_offsets_for(gens_a::Vector{Int}, Qb::PModule{K}) where {K}
        offs = zeros(Int64, length(gens_a) + 1)
        @inbounds for i in 1:length(gens_a)
            offs[i + 1] = offs[i] + Qb.dims[gens_a[i]]
        end
        return offs
    end

    function _build_tor_doublecomplex_plan(Rop::PModule{K},
                                           L::PModule{K},
                                           resR,
                                           resL,
                                           lenR::Int,
                                           lenL::Int;
                                           threads::Bool=false) where {K}
        amin, amax = -lenR, 0
        bmin, bmax = -lenL, 0
        na = lenR + 1
        nb = lenL + 1

        gens_by_a = Vector{Vector{Int}}(undef, na)
        @inbounds for ia in 1:na
            a = -(amin + (ia - 1))
            gens_by_a[ia] = resR.gens[a + 1]
        end

        dims = zeros(Int, na, nb)
        if threads && Threads.nthreads() > 1
            Threads.@threads for idx in 1:(na * nb)
                ia = (idx - 1) % na + 1
                ib = Int(div(idx - 1, na)) + 1
                gens_a = gens_by_a[ia]
                Qb = resL.Pmods[-(bmin + (ib - 1)) + 1]
                d = 0
                @inbounds for u in gens_a
                    d += Qb.dims[u]
                end
                dims[ia, ib] = d
            end
        else
            @inbounds for ia in 1:na
                gens_a = gens_by_a[ia]
                for ib in 1:nb
                    Qb = resL.Pmods[-(bmin + (ib - 1)) + 1]
                    d = 0
                    for u in gens_a
                        d += Qb.dims[u]
                    end
                    dims[ia, ib] = d
                end
            end
        end

        tensor_offsets = if _TOR_DOUBLE_COMPLEX_CACHE_TENSOR_OFFSETS[]
            offs = Matrix{Vector{Int64}}(undef, na, nb)
            @inbounds for ia in 1:na
                gens_a = gens_by_a[ia]
                for ib in 1:nb
                    offs[ia, ib] = _tensor_offsets_for(gens_a, resL.Pmods[-(bmin + (ib - 1)) + 1])
                end
            end
            offs
        else
            nothing
        end

        hplans = Vector{_TorCoeffPlan{K}}(undef, max(0, na - 1))
        @inbounds for ia in 1:max(0, na - 1)
            a = -(amin + (ia - 1))
            hplans[ia] = _tor_coeff_plan(resR.d_mat[a], gens_by_a[ia], gens_by_a[ia + 1])
        end

        return _TorDoubleComplexPlan{K}(amin, amax, bmin, bmax, dims, gens_by_a, tensor_offsets, hplans)
    end

    function _tor_vertical_tensor_matrix!(ws::_TorBicomplexTripletWorkspace{K},
                                          gens_a::AbstractVector{<:Integer},
                                          dQ,
                                          offs_cod::AbstractVector{<:Integer},
                                          offs_dom::AbstractVector{<:Integer},
                                          sgn::K,
                                          out_dim::Int, in_dim::Int) where {K}
        empty!(ws.I)
        empty!(ws.J)
        empty!(ws.V)
        @inbounds for (i, u) in enumerate(gens_a)
            _append_scaled_triplets!(ws.I, ws.J, ws.V, dQ.comps[Int(u)],
                                     Int(offs_cod[i]), Int(offs_dom[i]); scale=sgn)
        end
        return sparse(ws.I, ws.J, ws.V, out_dim, in_dim)
    end

    function _tor_horizontal_tensor_matrix!(ws::_TorBicomplexTripletWorkspace{K},
                                            Qb::PModule{K,F,MatT},
                                            plan::_TorCoeffPlan{K},
                                            offs_cod::AbstractVector{<:Integer},
                                            offs_dom::AbstractVector{<:Integer},
                                            out_dim::Int, in_dim::Int) where {K,F,MatT<:AbstractMatrix{K}}
        empty!(ws.I)
        empty!(ws.J)
        empty!(ws.V)

        if _tor_use_direct_triplets(plan, offs_cod, offs_dom; bicomplex=true)
            _append_map_leq_many_scaled_triplets!(ws.I, ws.J, ws.V, Qb, plan.batch,
                                                  plan.rows, plan.cols, offs_cod, offs_dom, plan.vals)
            return sparse(ws.I, ws.J, ws.V, out_dim, in_dim)
        end

        map_blocks = _map_blocks_buffer(Qb, length(plan.rows))
        map_leq_many!(map_blocks, Qb, plan.batch)
        @inbounds for k in eachindex(plan.rows)
            c = plan.vals[k]
            iszero(c) && continue
            _append_scaled_triplets!(ws.I, ws.J, ws.V, map_blocks[k],
                                     Int(offs_cod[plan.rows[k]]), Int(offs_dom[plan.cols[k]]); scale=c)
        end
        return sparse(ws.I, ws.J, ws.V, out_dim, in_dim)
    end



    """
        ExtDoubleComplex(M, N; maxlen=nothing) -> ChainComplexes.DoubleComplex{K}

    Build the bounded double complex C^{a,b} = Hom(F_a, E^b) where:
    - F is an upset resolution of M,
    - E is a downset resolution of N.

    Tot(ExtDoubleComplex(M,N)) computes Ext^*(M,N).

    If maxlen is provided, both resolutions are truncated at that length.
    If maxlen is nothing, each resolution is computed until it terminates.

    Cheap-first workflow
    - Start with `double_complex_summary(DC)` or `ChainComplexes.bicomplex_summary(DC)`.
    - Ask for the actual differentials only when you need chain-level bicomplex
      data.
    """
    function ExtDoubleComplex(M::PModule{K}, N::PModule{K};
                              maxlen::Union{Nothing,Int}=nothing,
                              threads::Bool = (Threads.nthreads() > 1),
                              cache::Union{Nothing,ResolutionCache}=nothing) where {K}
        cache_key = (cache === nothing || maxlen === nothing) ? nothing : _ext_doublecomplex_cache_key(M, N, maxlen)
        if cache_key !== nothing
            cached = _cache_ext_doublecomplex_get(cache, cache_key)
            cached === nothing || return cached
        end
        F, dF = IndicatorResolutions.upset_resolution(M; maxlen=maxlen, threads=threads)
        E, dE = IndicatorResolutions.downset_resolution(N; maxlen=maxlen, threads=threads)
        DC = ExtDoubleComplex(F, dF, E, dE; threads=threads, cache=cache)
        return cache_key === nothing ? DC : _cache_ext_doublecomplex_store!(cache, cache_key, DC)
    end

    """
        ExtDoubleComplex(M::PModule{K}, N::PModule{K}, res::ResolutionOptions)
    
    Options-based overload.
    
    This is equivalent to calling `ExtDoubleComplex(M, N; maxlen=res.maxlen)`.
    Only `res.maxlen` is used; the other fields of `ResolutionOptions` do not
    affect indicator resolutions.
    """
    function ExtDoubleComplex(M::PModule{K}, N::PModule{K}, res::ResolutionOptions;
                              threads::Bool = (Threads.nthreads() > 1),
                              cache::Union{Nothing,ResolutionCache}=nothing) where {K}
        return ExtDoubleComplex(M, N; maxlen=res.maxlen, threads=threads, cache=cache)
    end

    """
        ExtDoubleComplex(F, dF, E, dE) -> ChainComplexes.DoubleComplex{K}

    Low-level constructor from precomputed resolutions.
    """
    function ExtDoubleComplex(F::AbstractVector{<:UpsetPresentation{K}},
                              dF::Vector{SparseMatrixCSC{K,Int}},
                              E::AbstractVector{<:DownsetCopresentation{K}},
                              dE::Vector{SparseMatrixCSC{K,Int}};
                              threads::Bool = (Threads.nthreads() > 1),
                              cache::Union{Nothing,ResolutionCache}=nothing) where {K}
        dims, dv, dh = build_hom_bicomplex_data(F, dF, E, dE; threads=threads, cache=cache)
        A = length(F) - 1
        B = length(E) - 1
        return ChainComplexes.DoubleComplex{K}(0, A, 0, B, dims, dv, dh)
    end

    """
        ExtSpectralSequence(M, N; first=:vertical, maxlen=nothing) -> ChainComplexes.SpectralSequence{K}

    Compute the spectral sequence associated to the Ext bicomplex Hom(F_a, E^b).

    - first=:vertical uses vertical cohomology first (E1^{a,b} = H^b of columns).
    - first=:horizontal uses horizontal cohomology first.

    The returned object includes E1, d1, E2, Einf (graded pieces), and dim H^*(Tot).

    Cheap-first workflow
    - Start with `spectral_sequence_summary(ss)`, `page_dimensions(ss, 2)`, and
      `ChainComplexes.convergence_page(ss)`.
    - Ask for `ChainComplexes.page_terms(ss, r)` only when you need explicit
      `SubquotientData` terms.

    Example
    -------
    A typical exploration pattern is:
    1. build `ss = ExtSpectralSequence(M, N; maxlen=2)`,
    2. inspect `spectral_sequence_summary(ss)`,
    3. inspect `page_dimensions(ss, 2)`,
    4. inspect `ChainComplexes.term_dims(ss, t)` or `ChainComplexes.convergence_page(ss)`,
    5. only then ask for `ChainComplexes.page_terms(ss, 2)` if explicit page
       terms are needed.
    """
    function ExtSpectralSequence(M::PModule{K}, N::PModule{K};
                                first::Symbol=:vertical,
                                maxlen::Union{Nothing,Int}=nothing,
                                threads::Bool = (Threads.nthreads() > 1),
                                cache::Union{Nothing,ResolutionCache}=nothing) where {K}
        DC = ExtDoubleComplex(M, N; maxlen=maxlen, threads=threads, cache=cache)
        return ChainComplexes.spectral_sequence(DC; output=:full, first=first)
    end

    """
        ExtSpectralSequence(M::PModule{K}, N::PModule{K}, res::ResolutionOptions;
                            first=:vertical)
    
    Options-based overload.
    
    This is equivalent to calling `ExtSpectralSequence(M, N; first=first, maxlen=res.maxlen)`.
    Only `res.maxlen` is used.
    """
    function ExtSpectralSequence(M::PModule{K}, N::PModule{K}, res::ResolutionOptions;
                                 first::Symbol = :vertical,
                                 threads::Bool = (Threads.nthreads() > 1),
                                 cache::Union{Nothing,ResolutionCache}=nothing) where {K}
        DC = ExtDoubleComplex(M, N, res; threads=threads, cache=cache)
        return ChainComplexes.spectral_sequence(DC; output=:full, first=first)
    end

    # ----------------------------------------------------------------------
    # Tor bicomplex / spectral sequence helpers
    # ----------------------------------------------------------------------

    """
        TorDoubleComplex(Rop, L; maxlen=nothing, maxlenR=nothing, maxlenL=nothing)

    Build a double complex computing Tor_*(Rop,L) from:

    - a projective resolution of Rop as a P^op-module (right module),
    - a projective resolution of L as a P-module (left module).

    Conventions and indexing:

    The underlying (homological) bicomplex is
        C_{a,b} = P_a(Rop) otimes Q_b(L),  a,b >= 0,

    with horizontal differential induced by d_P and vertical differential induced by d_Q.

    We return it as a `ChainComplexes.DoubleComplex` in *cochain* bidegrees (A,B) = (-a,-b),
    so that the total cohomology satisfies
        H^t(Tot(C)) = Tor_{-t}(Rop, L).

    Sign convention:

    `ChainComplexes.total_complex` forms the total differential as dv + dh, so we must build dv and dh
    so that dv*dh + dh*dv = 0. We use the standard Koszul sign on the vertical differential:
        dv = (-1)^a * (id otimes d_Q).

    Keyword arguments:
    - maxlen: if set, both resolutions are built/padded to this length (override with maxlenR/maxlenL).
    - maxlenR: length for the Rop resolution.
    - maxlenL: length for the L resolution.
    """
    function TorDoubleComplex(Rop::PModule{K}, L::PModule{K};
        maxlen=nothing,
        maxlenR=nothing,
        maxlenL=nothing,
        threads::Bool = (Threads.nthreads() > 1),
        cache::Union{Nothing,ResolutionCache}=nothing,
    ) where {K}
        # Choose resolution lengths.
        lenR = (maxlenR === nothing) ? (maxlen === nothing ? 3 : maxlen) : maxlenR
        lenL = (maxlenL === nothing) ? (maxlen === nothing ? 3 : maxlen) : maxlenL
        @assert lenR >= 0 && lenL >= 0

        cache_key = cache === nothing ? nothing : _tor_doublecomplex_cache_key(Rop, L, lenR, lenL)
        if cache_key !== nothing
            cached = _cache_tor_doublecomplex_get(cache, cache_key)
            cached === nothing || return cached
        end

        # Build and pad both resolutions so the double complex is a full rectangle.
        resR = projective_resolution(Rop, ResolutionOptions(maxlen=lenR); cache=cache)
        resL = projective_resolution(L, ResolutionOptions(maxlen=lenL); cache=cache)
        _pad_projective_resolution!(resR, lenR)
        _pad_projective_resolution!(resL, lenL)
        plan_key = cache === nothing ? nothing : _tor_doublecomplex_cache_key(resR, resL, lenR, lenL)
        plan = if plan_key === nothing
            _build_tor_doublecomplex_plan(Rop, L, resR, resL, lenR, lenL; threads=threads)
        else
            cached_plan = _cache_tor_doublecomplex_plan_get(cache, plan_key)
            cached_plan === nothing ?
                _cache_tor_doublecomplex_plan_store!(cache, plan_key,
                                                     _build_tor_doublecomplex_plan(Rop, L, resR, resL, lenR, lenL; threads=threads)) :
                cached_plan
        end

        amin = plan.amin
        amax = plan.amax
        bmin = plan.bmin
        bmax = plan.bmax
        na = amax - amin + 1
        nb = bmax - bmin + 1
        dims = plan.dims
        tensor_offsets = plan.tensor_offsets

        # Vertical differentials: dv_{A,B} : C^{A,B} -> C^{A,B+1}
        # Corresponds to (-1)^a * (id otimes d_Q_b) in the original chain bicomplex.
        dv = Array{SparseMatrixCSC{K, Int64}, 2}(undef, na, nb - 1)
        dv_workspaces = [_tor_bicomplex_workspace(K) for _ in 1:Threads.maxthreadid()]
        if threads && Threads.nthreads() > 1
            Threads.@threads for ia in 1:na
                ws = dv_workspaces[Threads.threadid()]
                A = amin + (ia - 1)
                a = -A
                gens_a = plan.gens_by_a[ia]
                sgn = isodd(a) ? -one(K) : one(K)
                for ib in 1:(nb - 1)
                    B = bmin + (ib - 1)
                    b = -B
                    dQ = resL.d_mor[b]
                    Qb = resL.Pmods[b + 1]
                    Qbm1 = resL.Pmods[b]

                    offs_dom = tensor_offsets === nothing ? _tensor_offsets_for(gens_a, Qb) : tensor_offsets[ia, ib]
                    offs_cod = tensor_offsets === nothing ? _tensor_offsets_for(gens_a, Qbm1) : tensor_offsets[ia, ib + 1]
                    dv[ia, ib] = _tor_vertical_tensor_matrix!(ws, gens_a, dQ, offs_cod, offs_dom,
                                                              sgn, offs_cod[end], offs_dom[end])
                end
            end
        else
            ws = dv_workspaces[1]
            for ia in 1:na
                A = amin + (ia - 1)
                a = -A
                gens_a = plan.gens_by_a[ia]
                sgn = isodd(a) ? -one(K) : one(K)
                for ib in 1:(nb - 1)
                    B = bmin + (ib - 1)
                    b = -B
                    # d_Q_b : Q_b -> Q_{b-1}, stored at index b (since b>=1 here)
                    dQ = resL.d_mor[b]
                    Qb = resL.Pmods[b + 1]
                    Qbm1 = resL.Pmods[b]  # b-1

                    offs_dom = tensor_offsets === nothing ? _tensor_offsets_for(gens_a, Qb) : tensor_offsets[ia, ib]
                    offs_cod = tensor_offsets === nothing ? _tensor_offsets_for(gens_a, Qbm1) : tensor_offsets[ia, ib + 1]
                    dv[ia, ib] = _tor_vertical_tensor_matrix!(ws, gens_a, dQ, offs_cod, offs_dom,
                                                              sgn, offs_cod[end], offs_dom[end])
                end
            end
        end

        dh = Array{SparseMatrixCSC{K, Int64}, 2}(undef, na - 1, nb)
        # --- Horizontal differentials dh: (P_a otimes Q_b) -> (P_{a-1} otimes Q_b) ---
        # dP_a is stored as a sparse coefficient matrix between generators:
        # for each nonzero (j,i,c): gen u=gens_dom[i] maps to gen v=gens_cod[j] with scalar c.
        # In the tensor with Q_b (a P-module), this yields Q_b[u] -> Q_b[v] via map_leq(Q_b, u, v).

        dh_workspaces = [_tor_bicomplex_workspace(K) for _ in 1:Threads.maxthreadid()]
        if threads && Threads.nthreads() > 1
            Threads.@threads for ia in 1:(na - 1)
                ws = dh_workspaces[Threads.threadid()]
                gens_dom = plan.gens_by_a[ia]
                gens_cod = plan.gens_by_a[ia + 1]
                hplan = plan.hplans[ia]
                for ib in 1:nb
                    B = bmin + (ib - 1)
                    b = -B
                    Qb = resL.Pmods[b + 1]
                    offs_dom = tensor_offsets === nothing ? _tensor_offsets_for(gens_dom, Qb) : tensor_offsets[ia, ib]
                    offs_cod = tensor_offsets === nothing ? _tensor_offsets_for(gens_cod, Qb) : tensor_offsets[ia + 1, ib]
                    dh[ia, ib] = _tor_horizontal_tensor_matrix!(ws, Qb, hplan, offs_cod, offs_dom,
                                                                offs_cod[end], offs_dom[end])
                end
            end
        else
            ws = dh_workspaces[1]
            for ia in 1:(na - 1)
                gens_dom = plan.gens_by_a[ia]
                gens_cod = plan.gens_by_a[ia + 1]
                hplan = plan.hplans[ia]

                for ib in 1:nb
                    B = bmin + (ib - 1)
                    b = -B
                    Qb = resL.Pmods[b + 1]  # Q_b (a P-module)
                    offs_dom = tensor_offsets === nothing ? _tensor_offsets_for(gens_dom, Qb) : tensor_offsets[ia, ib]
                    offs_cod = tensor_offsets === nothing ? _tensor_offsets_for(gens_cod, Qb) : tensor_offsets[ia + 1, ib]
                    dh[ia, ib] = _tor_horizontal_tensor_matrix!(ws, Qb, hplan, offs_cod, offs_dom,
                                                                offs_cod[end], offs_dom[end])
                end
            end
        end

        DC = ChainComplexes.DoubleComplex(amin, amax, bmin, bmax, dims, dv, dh)
        return cache_key === nothing ? DC : _cache_tor_doublecomplex_store!(cache, cache_key, DC)
    end

    """
        TorSpectralSequence(Rop, L; maxlen=nothing, maxlenR=nothing, maxlenL=nothing, first=:vertical)

    Return a spectral sequence associated to `TorDoubleComplex(Rop,L)`.

    This is a small wrapper around `ChainComplexes.SpectralSequence` that reindexes bidegrees
    to a homological convention (a,b) >= 0.

    Internals:
    - Underlying SS is computed on cochain bidegrees (A,B) = (-a,-b).
    - Public indexing uses (a,b) >= 0.

    The wrapper extends the `ChainComplexes` SS API so you can call:
    - `E_r(ss, r)` / `page(ss,r)` and index the returned page as `pg[a,b]`.
    - `term(ss,r,(a,b))`, `differential(ss,r,(a,b))`, `dr_target`, `dr_source`.
    - `edge_inclusion(ss,(a,b))`, `edge_projection(ss,(a,b))`.
    - `convergence_report(ss)`.

    Note: the abutment is total cohomology of the cochain total complex; interpret it as Tor via
        H^t(Tot) = Tor_{-t}.

    Cheap-first workflow
    - Start with `spectral_sequence_summary(ss)` and `page_dimensions(ss, 2)`.
    - Ask for `ChainComplexes.page_terms(ss, r)` only when you need explicit
      `SubquotientData` terms.
    """
    struct TorSpectralSequence{K}
        ss::ChainComplexes.SpectralSequence{K}
    end

    struct TorSpectralPage{K}
        pg::ChainComplexes.SpectralPage{K}
    end

    # Reindexing: term at (a,b) corresponds to underlying (-a,-b).
    Base.getindex(P::TorSpectralPage, a::Int, b::Int) = P.pg[-a, -b]


    # ---------------------------------------------------------------------------
    # Term-level access for Tor spectral sequences (SubquotientData pages)
    # ---------------------------------------------------------------------------

    """
        TorSpectralTermsPage

    A thin wrapper around `ChainComplexes.SpectralTermsPage` that reindexes the
    bidegrees so that `P[(a,b)]` corresponds to the underlying cohomological
    bidegree `(-a,-b)`.

    This mirrors `TorSpectralPage` (dimensions) but for actual `SubquotientData`
    terms (object-level access).
    """
    struct TorSpectralTermsPage{K} <: AbstractMatrix{ChainComplexes.SubquotientData{K}}
        pg::ChainComplexes.SpectralTermsPage
    end

    Base.size(P::TorSpectralTermsPage) = size(P.pg)
    Base.getindex(P::TorSpectralTermsPage, i::Int, j::Int) = P.pg[i, j]
    Base.getindex(P::TorSpectralTermsPage, ab::Tuple{Int,Int}) = P.pg[(-ab[1], -ab[2])]

    function ChainComplexes.E_r_terms(TSS::TorSpectralSequence{K}, r::Union{Int,Symbol}) where {K}
        return TorSpectralTermsPage{K}(ChainComplexes.E_r_terms(TSS.ss, r))
    end

    ChainComplexes.E2_terms(TSS::TorSpectralSequence) = ChainComplexes.E_r_terms(TSS, 2)

    # Convenience: allow `page_terms(TSS, r)` in Tor indexing convention.
    ChainComplexes.page_terms(TSS::TorSpectralSequence, r::Union{Int,Symbol}) = ChainComplexes.E_r_terms(TSS, r)

    function ChainComplexes.page_terms_dict(TSS::TorSpectralSequence{K}, r::Union{Int,Symbol}; nonzero_only::Bool=true) where {K}
        base = ChainComplexes.page_terms_dict(TSS.ss, r; nonzero_only=nonzero_only)
        out = Dict{Tuple{Int,Int}, ChainComplexes.SubquotientData{K}}()
        for ((A, B), sq) in base
            out[(-A, -B)] = sq
        end
        return out
    end

    function ChainComplexes.page_dims_dict(TSS::TorSpectralSequence, r::Union{Int,Symbol}; nonzero_only::Bool=true)
        base = ChainComplexes.page_dims_dict(TSS.ss, r; nonzero_only=nonzero_only)
        out = Dict{Tuple{Int,Int}, Int}()
        for ((A, B), d) in base
            out[(-A, -B)] = d
        end
        return out
    end



    # Extend the ChainComplexes spectral sequence API for the Tor wrapper.
    function ChainComplexes.E_r(TSS::TorSpectralSequence{K}, r::Int) where {K}
        return TorSpectralPage{K}(ChainComplexes.E_r(TSS.ss, r))
    end

    ChainComplexes.page(TSS::TorSpectralSequence{K}, r::Int) where {K} = ChainComplexes.E_r(TSS, r)

    function ChainComplexes.term(TSS::TorSpectralSequence, r::Int, ab::Tuple{Int, Int})
        a, b = ab
        return ChainComplexes.term(TSS.ss, r, (-a, -b))
    end

    function ChainComplexes.differential(TSS::TorSpectralSequence, r::Int, ab::Tuple{Int, Int})
        a, b = ab
        return ChainComplexes.differential(TSS.ss, r, (-a, -b))
    end

    function ChainComplexes.dr_target(TSS::TorSpectralSequence, r::Int, ab::Tuple{Int, Int})
        a, b = ab
        A, B = ChainComplexes.dr_target(TSS.ss, r, (-a, -b))
        return (-A, -B)
    end

    function ChainComplexes.dr_source(TSS::TorSpectralSequence, r::Int, ab::Tuple{Int, Int})
        a, b = ab
        A, B = ChainComplexes.dr_source(TSS.ss, r, (-a, -b))
        return (-A, -B)
    end

    ChainComplexes.convergence_report(TSS::TorSpectralSequence) = ChainComplexes.convergence_report(TSS.ss)
    ChainComplexes.convergence_page(TSS::TorSpectralSequence) = ChainComplexes.convergence_page(TSS.ss)

    function ChainComplexes.edge_inclusion(TSS::TorSpectralSequence, ab::Tuple{Int, Int})
        a, b = ab
        return ChainComplexes.edge_inclusion(TSS.ss, (-a, -b))
    end

    function ChainComplexes.edge_projection(TSS::TorSpectralSequence, ab::Tuple{Int, Int})
        a, b = ab
        return ChainComplexes.edge_projection(TSS.ss, (-a, -b))
    end

    function TorSpectralSequence(Rop::PModule{K}, L::PModule{K};
                                 maxlen=nothing, maxlenR=nothing, maxlenL=nothing, first=:vertical,
                                 threads::Bool = (Threads.nthreads() > 1),
                                 cache::Union{Nothing,ResolutionCache}=nothing) where {K}
        DC = TorDoubleComplex(Rop, L; maxlen=maxlen, maxlenR=maxlenR, maxlenL=maxlenL,
                              threads=threads, cache=cache)
        ss = ChainComplexes.spectral_sequence(DC; output=:full, first=first)
        return TorSpectralSequence{K}(ss)
    end

    @inline function page_dimensions(ss::ChainComplexes.SpectralSequence, r::Union{Int,Symbol}; nonzero_only::Bool=true)
        return ChainComplexes.page_dims_dict(ss, r; nonzero_only=nonzero_only)
    end

    @inline function page_dimensions(ss::ChainComplexes.SpectralSequence; page::Union{Int,Symbol}=2, nonzero_only::Bool=true)
        return page_dimensions(ss, page; nonzero_only=nonzero_only)
    end

    @inline function page_dimensions(TSS::TorSpectralSequence, r::Union{Int,Symbol}; nonzero_only::Bool=true)
        return ChainComplexes.page_dims_dict(TSS, r; nonzero_only=nonzero_only)
    end

    @inline function page_dimensions(TSS::TorSpectralSequence; page::Union{Int,Symbol}=2, nonzero_only::Bool=true)
        return page_dimensions(TSS, page; nonzero_only=nonzero_only)
    end

    @inline function spectral_sequence_summary(TSS::TorSpectralSequence)
        base = ChainComplexes.spectral_sequence_summary(TSS.ss)
        return merge(base, (
            kind=:tor_spectral_sequence,
            indexing=:tor_homological,
            wrapped_kind=base.kind,
        ))
    end

    """
        wrapped_spectral_sequence(TSS::TorSpectralSequence)

    Return the underlying `ChainComplexes.SpectralSequence` stored in the Tor
    wrapper.

    Use this only when you need the full chain-complex spectral-sequence
    interface. For routine inspection start with `spectral_sequence_summary(TSS)`
    or `page_dimensions(TSS, page)`.
    """
    @inline wrapped_spectral_sequence(TSS::TorSpectralSequence) = TSS.ss

    function Base.show(io::IO, TSS::TorSpectralSequence)
        d = spectral_sequence_summary(TSS)
        print(io, "TorSpectralSequence(convergence_page=", d.convergence_page,
              ", page2_terms=", length(page_dimensions(TSS, 2)), ")")
    end

    function Base.show(io::IO, ::MIME"text/plain", TSS::TorSpectralSequence)
        d = spectral_sequence_summary(TSS)
        print(io, "TorSpectralSequence",
              "\n  field: ", d.field,
              "\n  index_kind: ", d.indexing,
              "\n  convergence_page: ", d.convergence_page,
              "\n  e2_nonzero_terms: ", length(page_dimensions(TSS, 2)),
              "\n  total_degree_range: ", repr(d.total_degree_range))
    end

    """
        double_complex_summary(DC) -> NamedTuple

    Cheap summary alias for double complexes arising in the derived-functor
    layer. This delegates to `ChainComplexes.bicomplex_summary`.

    Use this before asking for specific bidegree blocks or differentials when
    you only need an overview of the derived double complex.
    """
    @inline function double_complex_summary(DC::ChainComplexes.DoubleComplex)
        base = ChainComplexes.bicomplex_summary(DC)
        return merge(base, (kind=:derived_double_complex,))
    end

    """
        check_ext_spectral_sequence(ss; throw=false) -> NamedTuple

    Validate that an Ext spectral sequence object supports the canonical
    cheap-first inspection workflow.
    """
    function check_ext_spectral_sequence(ss::ChainComplexes.SpectralSequence; throw::Bool=false)
        issues = String[]
        summary = try
            ChainComplexes.spectral_sequence_summary(ss)
        catch err
            push!(issues, sprint(showerror, err))
            nothing
        end
        page2 = try
            page_dimensions(ss, 2)
        catch err
            push!(issues, sprint(showerror, err))
            Dict{Tuple{Int,Int},Int}()
        end
        try
            ChainComplexes.convergence_page(ss)
        catch err
            push!(issues, sprint(showerror, err))
        end
        report = _derived_validation_report(
            :ext_spectral_sequence,
            isempty(issues);
            wrapped_kind=summary === nothing ? nothing : summary.kind,
            page2_nonzero=length(page2),
            issues=issues,
        )
        throw && !report.valid && _throw_invalid_derived_functor(:check_ext_spectral_sequence, issues)
        return report
    end

    """
        check_tor_spectral_sequence(ss; throw=false) -> NamedTuple

    Validate that a Tor spectral sequence wrapper supports the canonical
    cheap-first inspection workflow.
    """
    function check_tor_spectral_sequence(TSS::TorSpectralSequence; throw::Bool=false)
        issues = String[]
        summary = try
            spectral_sequence_summary(TSS)
        catch err
            push!(issues, sprint(showerror, err))
            nothing
        end
        page2 = try
            page_dimensions(TSS, 2)
        catch err
            push!(issues, sprint(showerror, err))
            Dict{Tuple{Int,Int},Int}()
        end
        try
            ChainComplexes.convergence_page(TSS)
        catch err
            push!(issues, sprint(showerror, err))
        end
        report = _derived_validation_report(
            :tor_spectral_sequence,
            isempty(issues);
            wrapped_kind=summary === nothing ? nothing : summary.kind,
            page2_nonzero=length(page2),
            issues=issues,
        )
        throw && !report.valid && _throw_invalid_derived_functor(:check_tor_spectral_sequence, issues)
        return report
    end

end
