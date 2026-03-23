# hom_ext_engine.jl -- low-level Hom/Ext engine kernels

"""
    module HomExtEngine

Low-level routines for Hom/Ext computations from indicator resolutions.

Public API:
- build_hom_tot_complex
- build_hom_bicomplex_data
- ext_dims_via_resolutions
- pi0_count
"""
module HomExtEngine
    # -----------------------------------------------------------------------------
    # Hom/Ext via indicator resolutions
    # -----------------------------------------------------------------------------

    using SparseArrays
    using ...CoreModules: AbstractCoeffField, RealField, ResolutionCache, ResolutionKey5,
                          HomBicomplexPayload, _resolution_key5, field_from_eltype
    using ...FieldLinAlg
    using ...FiniteFringe: AbstractPoset, FinitePoset, Upset, Downset, cover_edges, nvertices
    using ...IndicatorTypes: UpsetPresentation, DownsetCopresentation
    import ..DerivedFunctors: _build_total_offsets_grid, _total_offset_get

    """
        _hasse_undirected(P) -> adj

    Return the undirected adjacency list of the Hasse (cover) graph of the finite poset `P`.

    Vertices are numbered `1:P.n`. The result is symmetric: if `j` is in `adj[i]` then `i`
    is in `adj[j]`.

    Used internally to compute connected components of intersections `U cap D`.
    """
    function _hasse_undirected(P::AbstractPoset)
        CE = cover_edges(P)
        n = nvertices(P)
        adj = [Int[] for _ in 1:n]
        for (a, b) in CE
            push!(adj[a], b)
            push!(adj[b], a)
        end
        return adj
    end

    """
        CompCache(P)

    Cache for connected-component computations on intersections `U cap D`.

    The Hom/Ext assembly repeatedly queries connected components of `U.mask .& D.mask`
    across many pairs (U, D). This cache memoizes results keyed by the pair of masks.
    """
    mutable struct CompCache{K}
        P::AbstractPoset
        adj::Vector{Vector{Int}}
        components_of_intersection_shards::Vector{Dict{Tuple{UInt,UInt}, Tuple{Vector{Int}, Int}}}
        component_inclusion_matrix_shards::Vector{Dict{NTuple{4,UInt},SparseMatrixCSC{K,Int}}}

        function CompCache{K}(P::AbstractPoset) where {K}
            adj = _hasse_undirected(P)
            nshards = max(1, Threads.maxthreadid())
            return new(P, adj,
                       [Dict{Tuple{UInt,UInt}, Tuple{Vector{Int}, Int}}() for _ in 1:nshards],
                       [Dict{NTuple{4,UInt},SparseMatrixCSC{K,Int}}() for _ in 1:nshards])
        end
    end

    const _HOM_ASSEMBLY_USE_TRIPLETS = Ref(true)
    const _HOM_ASSEMBLY_CACHE_OFFSETS = Ref(true)

    @inline function _hom_bicomplex_cache_tag()::UInt8
        tag = (_HOM_ASSEMBLY_USE_TRIPLETS[] ? 0x01 : 0x00) |
              (_HOM_ASSEMBLY_CACHE_OFFSETS[] ? 0x02 : 0x00)
        return UInt8(tag)
    end

    @inline function _hom_bicomplex_cache_key(F, E)
        return _resolution_key5(F, E, length(F) - 1, length(E) - 1, Int(_hom_bicomplex_cache_tag()))
    end

    @inline function _cache_hom_bicomplex_get(cache::ResolutionCache, key::ResolutionKey5)
        Base.lock(cache.lock)
        payload = get(cache.hom_bicomplex, key, nothing)
        Base.unlock(cache.lock)
        return payload === nothing ? nothing : payload.value
    end

    @inline function _cache_hom_bicomplex_store!(cache::ResolutionCache, key::ResolutionKey5, value)
        Base.lock(cache.lock)
        cache.hom_bicomplex[key] = HomBicomplexPayload(value)
        Base.unlock(cache.lock)
        return value
    end

    mutable struct _HomTripletWorkspace{K}
        I::Vector{Int}
        J::Vector{Int}
        V::Vector{K}
        empty_sparse::Dict{Tuple{Int,Int},SparseMatrixCSC{K,Int}}
    end

    @inline _hom_triplet_workspace(::Type{K}) where {K} =
        _HomTripletWorkspace{K}(Int[], Int[], K[], Dict{Tuple{Int,Int},SparseMatrixCSC{K,Int}}())

    @inline _mask_signature(mask::BitVector)::UInt = UInt(objectid(mask))

    @inline _compcache_shard_index(shards) = min(length(shards), max(1, Threads.threadid()))
    @inline _compcache_shard(shards) = shards[_compcache_shard_index(shards)]

    """
        _components_of_mask(adj, mask) -> (comp_id, ncomp)

    Compute connected components of the induced subgraph on the vertex subset `mask`.

    Inputs:
    - `adj`: undirected adjacency list (1-indexed).
    - `mask`: BitVector selecting active vertices.

    Returns:
    - `comp_id`: component label per vertex (0 for inactive vertices).
    - `ncomp`: number of connected components among active vertices.
    """
    function _components_of_mask(adj::Vector{Vector{Int}}, mask::BitVector)
        n = length(adj)
        comp_id = fill(0, n)
        cid = 0
        for v in 1:n
            if !mask[v] || comp_id[v] != 0
                continue
            end
            cid += 1
            stack = [v]
            comp_id[v] = cid
            while !isempty(stack)
                x = pop!(stack)
                for y in adj[x]
                    if mask[y] && comp_id[y] == 0
                        comp_id[y] = cid
                        push!(stack, y)
                    end
                end
            end
        end
        return comp_id, cid
    end

    """
        _components_of_intersection(P, U, D) -> (comp_id, ncomp)

    Compute connected components of the intersection mask `U.mask .& D.mask` in the undirected
    Hasse (cover) graph of the finite poset `P`.
    """
    function _components_of_intersection(P::AbstractPoset, U::Upset, D::Downset)
        adj = _hasse_undirected(P)
        return _components_of_mask(adj, U.mask .& D.mask)
    end

    """
        _components_cached!(cache, U, uid, D, did) -> (comp_id, ncomp)

    Return the connected components of `U cap D`, using and updating `cache`.

    The cache key is a compact integer signature of the bitmasks.
    """
    function _components_cached!(C::CompCache, U::Upset, uid::Int, D::Downset, did::Int)
        k = (_mask_signature(U.mask), _mask_signature(D.mask))
        shard = _compcache_shard(C.components_of_intersection_shards)
        if haskey(shard, k)
            return shard[k]
        end
        val = _components_of_intersection(C.P, U, D)
        shard[k] = val
        return val
    end

    """
        _component_inclusion_matrix_cached(C, Ubig, Dbig, uid_big, did_big,
                                            Usmall, Dsmall, uid_small, did_small, K)

    Return the component-level **restriction** map induced by containment
        (Usmall cap Dsmall) subseteq (Ubig cap Dbig).

    The result is a sparse matrix in component bases:
        H0(Ubig cap Dbig)  to  H0(Usmall cap Dsmall)

    So the matrix has size (n_small, n_big) with rows = small components,
    cols = big components.
    """
    function _component_inclusion_matrix_cached(
        C::CompCache{K},
        Ubig::Upset, Dbig::Downset, uid_big::Int, did_big::Int,
        Usmall::Upset, Dsmall::Downset, uid_small::Int, did_small::Int,
        ::Type{K},
    ) where {K}

        # IMPORTANT:
        # The (uid_big, did_big, ...) indices are only *local* enumerate indices inside each (a,b) block,
        # so they collide across different blocks and can return the wrong cached matrix.
        # Key instead by content signatures of the masks.
        key = (_mask_signature(Ubig.mask), _mask_signature(Dbig.mask),
               _mask_signature(Usmall.mask), _mask_signature(Dsmall.mask))

        shard = _compcache_shard(C.component_inclusion_matrix_shards)
        if haskey(shard, key)
            return shard[key]
        end

        comps_big, nb = _components_cached!(C, Ubig, uid_big, Dbig, did_big)
        comps_small, ns = _components_cached!(C, Usmall, uid_small, Dsmall, did_small)

        # pick a representative vertex from each small component
        rep_small = fill(0, ns)
        @inbounds for v in 1:length(comps_small)
            cs = comps_small[v]
            if cs != 0 && rep_small[cs] == 0
                rep_small[cs] = v
            end
        end

        rows = Int[]
        cols = Int[]
        vals = K[]
        sizehint!(rows, ns); sizehint!(cols, ns); sizehint!(vals, ns)

        @inbounds for cs in 1:ns
            v = rep_small[cs]
            cb = comps_big[v]   # since small subseteq big, v lies in big intersection too
            cb != 0 || error("internal: small component representative not in big intersection")
            push!(rows, cs)
            push!(cols, cb)
            push!(vals, one(K))
        end

        M = sparse(rows, cols, vals, ns, nb)
        shard[key] = M
        return M
    end

    function size_block(C::CompCache, U_by_a, D_by_b, a::Int, b::Int)
        Ulist = U_by_a[a + 1]
        Dlist = D_by_b[b + 1]
        tot = 0
        for (i, U) in enumerate(Ulist), (j, D) in enumerate(Dlist)
            _, ncomp = _components_cached!(C, U, i, D, j)
            tot += ncomp
        end
        return tot
    end

    """
        _block_offset(cache, U_by_a, D_by_b, a, b, j, i) -> Int

    Return the 0-based offset of the (j, i) sub-block inside the component basis of

        Hom(F_a, E^b).

    The basis is ordered by pairs `(D_j, U_i)`, and within each pair by connected components
    of `U_i cap D_j`.
    """
    function _block_offset(C::CompCache, U_by_a, D_by_b, a::Int, b::Int, j::Int, i::Int)
        Ulist = U_by_a[a + 1]
        Dlist = D_by_b[b + 1]

        off = 0
        # all blocks with downset index < j
        for jj in 1:(j - 1)
            for ii in 1:length(Ulist)
                _, ncomp = _components_cached!(C, Ulist[ii], ii, Dlist[jj], jj)
                off += ncomp
            end
        end
        # within downset j, all upsets with index < i
        for ii in 1:(i - 1)
            _, ncomp = _components_cached!(C, Ulist[ii], ii, Dlist[j], j)
            off += ncomp
        end
        return off
    end

    """
        _accum!(S, r0, c0, B) -> S

    Add the sparse block matrix `B` into the sparse matrix `S` with the top-left corner of `B`
    placed at `(r0, c0)` (1-based indexing). Returns `S`.
    """
    function _accum!(S::SparseMatrixCSC{K,Int}, r0::Int, c0::Int, B::SparseMatrixCSC{K,Int}) where {K}
        rows, cols, vals = findnz(B)
        @inbounds for k in eachindex(vals)
            S[r0 + rows[k] - 1, c0 + cols[k] - 1] += vals[k]
        end
        return S
    end

    @inline function _empty_sparse_matrix!(cache::Dict{Tuple{Int,Int},SparseMatrixCSC{K,Int}},
                                           nrows::Int, ncols::Int) where {K}
        return get!(cache, (nrows, ncols)) do
            spzeros(K, nrows, ncols)
        end
    end

    @inline function _append_shifted_scaled_triplets!(I::Vector{Int}, J::Vector{Int}, V::Vector{K},
                                                      B::SparseMatrixCSC{K,Int},
                                                      r0::Int, c0::Int;
                                                      scale::K=one(K)) where {K}
        rows, cols, vals = findnz(B)
        @inbounds for k in eachindex(vals)
            v = scale * vals[k]
            iszero(v) && continue
            push!(I, r0 + rows[k] - 1)
            push!(J, c0 + cols[k] - 1)
            push!(V, v)
        end
        return nothing
    end

    function _block_offset_table(C::CompCache, U_by_a, D_by_b, a::Int, b::Int)
        Ulist = U_by_a[a + 1]
        Dlist = D_by_b[b + 1]
        offs = Matrix{Int}(undef, length(Dlist), length(Ulist))
        off = 0
        for j in 1:length(Dlist), i in 1:length(Ulist)
            offs[j, i] = off
            _, ncomp = _components_cached!(C, Ulist[i], i, Dlist[j], j)
            off += ncomp
        end
        return offs
    end

    function _build_block_offset_tables(C::CompCache, U_by_a, D_by_b, A::Int, B::Int)
        tabs = Array{Matrix{Int},2}(undef, A + 1, B + 1)
        for a in 0:A, b in 0:B
            tabs[a + 1, b + 1] = _block_offset_table(C, U_by_a, D_by_b, a, b)
        end
        return tabs
    end

    @inline function _block_offset_lookup(offset_tables, C::CompCache, U_by_a, D_by_b,
                                          a::Int, b::Int, j::Int, i::Int)
        offset_tables === nothing && return _block_offset(C, U_by_a, D_by_b, a, b, j, i)
        return offset_tables[a + 1, b + 1][j, i]
    end

    @inline function _reset!(ws::_HomTripletWorkspace)
        empty!(ws.I)
        empty!(ws.J)
        empty!(ws.V)
        return ws
    end

    @inline function _finalize_sparse!(ws::_HomTripletWorkspace{K}, nrows::Int, ncols::Int) where {K}
        return isempty(ws.V) ? _empty_sparse_matrix!(ws.empty_sparse, nrows, ncols) :
               sparse(ws.I, ws.J, ws.V, nrows, ncols)
    end

    """
        size_block(P, U, D) -> Int

    Size of a block corresponding to `(U cap D)` in the component basis.
    """
    function size_block(P::AbstractPoset, U::Upset, D::Downset)
        _, ncomp = _components_of_intersection(P, U, D)
        return ncomp
    end

    """
        build_hom_tot_complex(F, dF, E, dE; maxlen=10) -> (C, dC)

    Assemble the total Hom complex associated to an upset resolution `(F, dF)` and a downset
    resolution `(E, dE)`.

    The output is a graded list of vector spaces `C[k]` with differentials `dC[k]`.
    """
    function build_hom_tot_complex(F::AbstractVector{<:UpsetPresentation{K}},
                                dF::Vector{SparseMatrixCSC{K,Int}},
                                E::AbstractVector{<:DownsetCopresentation{K}},
                                dE::Vector{SparseMatrixCSC{K,Int}};
                                threads::Bool=false) where {K}
        A = length(F) - 1              # top degree on the F-side
        B = length(E) - 1              # top degree on the E-side
        P = F[1].P
        compcache = CompCache{K}(P)
        caches = threads && Threads.nthreads() > 1 ?
            [CompCache{K}(P) for _ in 1:max(1, Threads.maxthreadid())] :
            Vector{CompCache{K}}()

        U_by_a = [f.U0 for f in F]
        D_by_b = [e.D0 for e in E]

        tmin, tmax = 0, A + B
        T = tmax - tmin + 1
        block_sizes = zeros(Int, A+1, B+1)

        # size of Hom(F_a, E^b) in the component basis.
        # Use the module-level helper (also used by build_hom_bicomplex_data).

        # compute block offsets per total degree
        if threads && Threads.nthreads() > 1
            Threads.@threads for idx in 1:((A + 1) * (B + 1))
                a = (idx - 1) % (A + 1)
                b = Int(div(idx - 1, (A + 1)))
                c = caches[_compcache_shard_index(caches)]
                block_sizes[a+1, b+1] = size_block(c, U_by_a, D_by_b, a, b)
            end
        else
            for a in 0:A, b in 0:B
                block_sizes[a+1, b+1] = size_block(compcache, U_by_a, D_by_b, a, b)
            end
        end

        offs_by_ta, dimsCt, _, _ = _build_total_offsets_grid(0, A, 0, B, block_sizes)
        offset_tables = _HOM_ASSEMBLY_CACHE_OFFSETS[] ?
            _build_block_offset_tables(compcache, U_by_a, D_by_b, A, B) : nothing

        # prepare differentials d^t : C^t to C^{t+1}
        dts = Vector{SparseMatrixCSC{K,Int}}(undef, T-1)
        trip_ws = _HOM_ASSEMBLY_USE_TRIPLETS[] ?
            [_hom_triplet_workspace(K) for _ in 1:max(1, Threads.maxthreadid())] :
            _HomTripletWorkspace{K}[]

        # fill post- and pre-composition contributions
        if threads && Threads.nthreads() > 1
            Threads.@threads for t in tmin:(tmax - 1)
                c = caches[_compcache_shard_index(caches)]
                idx = t - tmin + 1
                use_triplets = _HOM_ASSEMBLY_USE_TRIPLETS[]
                ws = use_triplets ? _reset!(trip_ws[Threads.threadid()]) : nothing
                M = use_triplets ? nothing : spzeros(K, dimsCt[idx+1], dimsCt[idx])

                alo = max(0, t - B)
                ahi = min(A, t)
                for a in alo:ahi
                    b = t - a
                    U = U_by_a[a+1]
                    D = D_by_b[b+1]
                    src0 = _total_offset_get(offs_by_ta, t, tmin, 0, a)

                    # post: Hom(F_a,E^b) -> Hom(F_a,E^{b+1}) via rho (if b < B)
                    if b < B
                        dst0 = _total_offset_get(offs_by_ta, t + 1, tmin, 0, a)
                        for (rowD1, D1j) in enumerate(D_by_b[b+2]), (colD0, D0j) in enumerate(D)
                            coeff = dE[b+1][rowD1, colD0]
                            if coeff != zero(K)
                                for (i, Uai) in enumerate(U)
                                    Bmat = _component_inclusion_matrix_cached(c,
                                        Uai, D0j, i, colD0,
                                        Uai, D1j, i, rowD1, K)

                                    if nnz(Bmat) > 0
                                        r0 = dst0 + _block_offset_lookup(offset_tables, c, U_by_a, D_by_b, a, b+1, rowD1, i) + 1
                                        c0 = src0 + _block_offset_lookup(offset_tables, c, U_by_a, D_by_b, a, b,   colD0, i) + 1
                                        if use_triplets
                                            _append_shifted_scaled_triplets!(ws.I, ws.J, ws.V, Bmat, r0, c0; scale=coeff)
                                        else
                                            _accum!(M, r0, c0, coeff * Bmat)
                                        end
                                    end
                                end
                            end
                        end
                    end

                    # pre: Hom(F_a,E^b) -> Hom(F_{a-1},E^b) via delta (if a < A) with sign (-1)^b
                    if a < A
                        sign = isodd(b) ? -one(K) : one(K)     # (-1)^b
                        dst0 = _total_offset_get(offs_by_ta, t + 1, tmin, 0, a + 1)

                        Unexts = U_by_a[a+2]  # U_{a+1}
                        for (rowUnext, Unext) in enumerate(Unexts), (colUcur, Ucur) in enumerate(U)
                            coeff = dF[a+1][rowUnext, colUcur]  # delta_a : U_{a+1} -> U_a
                            if coeff != zero(K)
                                for (j, Dbj) in enumerate(D)
                                    # restriction: (Ucur cap Dbj) -> (Unext cap Dbj)
                                    Bmat = _component_inclusion_matrix_cached(c,
                                        Ucur,  Dbj, colUcur,  j,
                                        Unext, Dbj, rowUnext, j, K)

                                    if nnz(Bmat) > 0
                                        r0 = dst0 + _block_offset_lookup(offset_tables, c, U_by_a, D_by_b, a+1, b, j, rowUnext) + 1
                                        c0 = src0 + _block_offset_lookup(offset_tables, c, U_by_a, D_by_b, a,   b, j, colUcur)  + 1
                                        if use_triplets
                                            _append_shifted_scaled_triplets!(ws.I, ws.J, ws.V, Bmat, r0, c0; scale=(sign * coeff))
                                        else
                                            _accum!(M, r0, c0, (sign * coeff) * Bmat)
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
                dts[idx] = use_triplets ? _finalize_sparse!(ws, dimsCt[idx+1], dimsCt[idx]) : M
            end
        else
            ws = _HOM_ASSEMBLY_USE_TRIPLETS[] ? trip_ws[1] : nothing
            for t in tmin:(tmax - 1)
                idx = t - tmin + 1
                use_triplets = _HOM_ASSEMBLY_USE_TRIPLETS[]
                use_triplets && _reset!(ws)
                M = use_triplets ? nothing : spzeros(K, dimsCt[idx+1], dimsCt[idx])
                alo = max(0, t - B)
                ahi = min(A, t)
                for a in alo:ahi
                    b = t - a
                    U = U_by_a[a+1]; D = D_by_b[b+1]
                    src0 = _total_offset_get(offs_by_ta, t, tmin, 0, a)

                    # post: Hom(F_a,E^b) \to Hom(F_a,E^{b+1}) via rho (if b < B)
                    if b < B
                        dst0 = _total_offset_get(offs_by_ta, t + 1, tmin, 0, a)
                        for (rowD1, D1j) in enumerate(D_by_b[b+2]), (colD0, D0j) in enumerate(D)
                            coeff = dE[b+1][rowD1, colD0]
                            if coeff != zero(K)
                                for (i, Uai) in enumerate(U)
                                    Bmat = _component_inclusion_matrix_cached(compcache,
                                        Uai, D0j, i, colD0,
                                        Uai, D1j, i, rowD1, K)

                                    if nnz(Bmat) > 0
                                        r0 = dst0 + _block_offset_lookup(offset_tables, compcache, U_by_a, D_by_b, a, b+1, rowD1, i) + 1
                                        c0 = src0 + _block_offset_lookup(offset_tables, compcache, U_by_a, D_by_b, a, b,   colD0, i) + 1
                                        if use_triplets
                                            _append_shifted_scaled_triplets!(ws.I, ws.J, ws.V, Bmat, r0, c0; scale=coeff)
                                        else
                                            _accum!(M, r0, c0, coeff * Bmat)
                                        end
                                    end
                                end
                            end
                        end
                    end

                    # pre: Hom(F_a,E^b) \to Hom(F_{a-1},E^b) via delta (if a >= 1) with sign -(-1)^a
                    if a < A
                        sign = isodd(b) ? -one(K) : one(K)     # (-1)^b
                        dst0 = _total_offset_get(offs_by_ta, t + 1, tmin, 0, a + 1)

                        Unexts = U_by_a[a+2]  # U_{a+1}
                        for (rowUnext, Unext) in enumerate(Unexts), (colUcur, Ucur) in enumerate(U)
                            coeff = dF[a+1][rowUnext, colUcur]  # delta_a : U_{a+1} -> U_a
                            if coeff != zero(K)
                                for (j, Dbj) in enumerate(D)
                                    # restriction: (Ucur cap Dbj) -> (Unext cap Dbj)
                                    Bmat = _component_inclusion_matrix_cached(compcache,
                                        Ucur,  Dbj, colUcur,  j,
                                        Unext, Dbj, rowUnext, j, K)

                                    if nnz(Bmat) > 0
                                        r0 = dst0 + _block_offset_lookup(offset_tables, compcache, U_by_a, D_by_b, a+1, b, j, rowUnext) + 1
                                        c0 = src0 + _block_offset_lookup(offset_tables, compcache, U_by_a, D_by_b, a,   b, j, colUcur)  + 1
                                        if use_triplets
                                            _append_shifted_scaled_triplets!(ws.I, ws.J, ws.V, Bmat, r0, c0; scale=(sign * coeff))
                                        else
                                            _accum!(M, r0, c0, (sign * coeff) * Bmat)
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
                dts[idx] = use_triplets ? _finalize_sparse!(ws, dimsCt[idx+1], dimsCt[idx]) : M
            end
        end

        return dimsCt, dts
    end

    """
        build_hom_bicomplex_data(F, dF, E, dE) -> (dims, dv, dh)

        Return the bicomplex data used internally by `build_hom_tot_complex`.

    Build the Hom bicomplex C^{a,b} = Hom(F_a, E^b) from an upset resolution (F, dF)
    and a downset resolution (E, dE).

    Conventions match `build_hom_tot_complex`:
    - Vertical differential dv is postcomposition with dE.
    - Horizontal differential dh is signed precomposition with dF, using the sign (-1)^b,
    so that the total differential on Tot is dv + dh and squares to zero.

    Returns:
    - dims[a+1, b+1] = dim Hom(F_a, E^b)
    - dv[a+1, b+1] : C^{a,b} -> C^{a,b+1}
    - dh[a+1, b+1] : C^{a,b} -> C^{a+1,b}
    """
    function build_hom_bicomplex_data(F::AbstractVector{<:UpsetPresentation{K}},
                                    dF::Vector{SparseMatrixCSC{K,Int}},
                                    E::AbstractVector{<:DownsetCopresentation{K}},
                                    dE::Vector{SparseMatrixCSC{K,Int}};
                                    threads::Bool=false,
                                    cache::Union{Nothing,ResolutionCache}=nothing) where {K}
        cache_key = cache === nothing ? nothing : _hom_bicomplex_cache_key(F, E)
        if cache_key !== nothing
            cached = _cache_hom_bicomplex_get(cache, cache_key)
            cached === nothing || return cached
        end
        A = length(F) - 1
        B = length(E) - 1
        if length(dF) != A || length(dE) != B
            error("build_hom_bicomplex_data: inconsistent differential lengths.")
        end

        P = F[1].P
        compcache = CompCache{K}(P)
        caches = threads && Threads.nthreads() > 1 ?
            [CompCache{K}(P) for _ in 1:max(1, Threads.maxthreadid())] :
            Vector{CompCache{K}}()
        U_by_a = [f.U0 for f in F]
        D_by_b = [e.D0 for e in E]

        # Block dimensions.
        dims = zeros(Int, A+1, B+1)
        if threads && Threads.nthreads() > 1
            Threads.@threads for idx in 1:((A + 1) * (B + 1))
                a = (idx - 1) % (A + 1)
                b = Int(div(idx - 1, (A + 1)))
                c = caches[_compcache_shard_index(caches)]
                dims[a+1, b+1] = size_block(c, U_by_a, D_by_b, a, b)
            end
        else
            for a in 0:A, b in 0:B
                dims[a+1, b+1] = size_block(compcache, U_by_a, D_by_b, a, b)
            end
        end

        offset_tables = _HOM_ASSEMBLY_CACHE_OFFSETS[] ?
            _build_block_offset_tables(compcache, U_by_a, D_by_b, A, B) : nothing

        dv = Array{SparseMatrixCSC{K,Int},2}(undef, A+1, B+1)
        dh = Array{SparseMatrixCSC{K,Int},2}(undef, A+1, B+1)
        empty_cache = Dict{Tuple{Int,Int},SparseMatrixCSC{K,Int}}()
        for a in 0:A, b in 0:B
            srcdim = dims[a+1, b+1]
            dv[a+1, b+1] = _empty_sparse_matrix!(empty_cache, (b < B) ? dims[a+1, b+2] : 0, srcdim)
            dh[a+1, b+1] = _empty_sparse_matrix!(empty_cache, (a < A) ? dims[a+2, b+1] : 0, srcdim)
        end

        trip_ws = _HOM_ASSEMBLY_USE_TRIPLETS[] ?
            [_hom_triplet_workspace(K) for _ in 1:max(1, Threads.maxthreadid())] :
            _HomTripletWorkspace{K}[]

        # Vertical differential: postcomposition with dE[b+1] : E^b -> E^{b+1}.
        if threads && Threads.nthreads() > 1
            Threads.@threads for idx in 1:((A + 1) * B)
                a = (idx - 1) % (A + 1)
                b = Int(div(idx - 1, (A + 1)))
                c = caches[_compcache_shard_index(caches)]
                U = U_by_a[a+1]
                D0 = D_by_b[b+1]
                D1 = D_by_b[b+2]
                use_triplets = _HOM_ASSEMBLY_USE_TRIPLETS[]
                ws = use_triplets ? _reset!(trip_ws[Threads.threadid()]) : nothing
                M = use_triplets ? nothing : spzeros(K, dims[a+1, b+2], dims[a+1, b+1])
                for rowD1 in 1:length(D1), colD0 in 1:length(D0)
                    coeff = dE[b+1][rowD1, colD0]
                    if iszero(coeff)
                        continue
                    end
                    for i in 1:length(U)
                        Bmat = _component_inclusion_matrix_cached(
                            c,
                            U[i], D0[colD0], i, colD0,
                            U[i], D1[rowD1], i, rowD1,
                            K
                        )
                        r0 = _block_offset_lookup(offset_tables, c, U_by_a, D_by_b, a, b+1, rowD1, i) + 1
                        c0 = _block_offset_lookup(offset_tables, c, U_by_a, D_by_b, a, b,   colD0, i) + 1
                        if use_triplets
                            _append_shifted_scaled_triplets!(ws.I, ws.J, ws.V, Bmat, r0, c0; scale=coeff)
                        else
                            _accum!(M, r0, c0, coeff * Bmat)
                        end
                    end
                end
                dv[a+1, b+1] = use_triplets ? _finalize_sparse!(ws, dims[a+1, b+2], dims[a+1, b+1]) : M
            end
        else
            ws = _HOM_ASSEMBLY_USE_TRIPLETS[] ? trip_ws[1] : nothing
            for a in 0:A
                U = U_by_a[a+1]
                for b in 0:(B-1)
                    D0 = D_by_b[b+1]
                    D1 = D_by_b[b+2]
                    use_triplets = _HOM_ASSEMBLY_USE_TRIPLETS[]
                    use_triplets && _reset!(ws)
                    M = use_triplets ? nothing : spzeros(K, dims[a+1, b+2], dims[a+1, b+1])
                    for rowD1 in 1:length(D1), colD0 in 1:length(D0)
                        coeff = dE[b+1][rowD1, colD0]
                        if iszero(coeff)
                            continue
                        end
                        for i in 1:length(U)
                            Bmat = _component_inclusion_matrix_cached(
                                compcache,
                                U[i], D0[colD0], i, colD0,
                                U[i], D1[rowD1], i, rowD1,
                                K
                            )
                            r0 = _block_offset_lookup(offset_tables, compcache, U_by_a, D_by_b, a, b+1, rowD1, i) + 1
                            c0 = _block_offset_lookup(offset_tables, compcache, U_by_a, D_by_b, a, b,   colD0, i) + 1
                            if use_triplets
                                _append_shifted_scaled_triplets!(ws.I, ws.J, ws.V, Bmat, r0, c0; scale=coeff)
                            else
                                _accum!(M, r0, c0, coeff * Bmat)
                            end
                        end
                    end
                    dv[a+1, b+1] = use_triplets ? _finalize_sparse!(ws, dims[a+1, b+2], dims[a+1, b+1]) : M
                end
            end
        end

        # Horizontal differential: signed precomposition with dF[a+1] : F_{a+1} -> F_a.
        if threads && Threads.nthreads() > 1
            Threads.@threads for idx in 1:((A) * (B + 1))
                a = (idx - 1) % A
                b = Int(div(idx - 1, A))
                c = caches[_compcache_shard_index(caches)]
                sign = isodd(b) ? -one(K) : one(K)
                D0 = D_by_b[b+1]
                Ucur = U_by_a[a+1]
                Unext = U_by_a[a+2]
                use_triplets = _HOM_ASSEMBLY_USE_TRIPLETS[]
                ws = use_triplets ? _reset!(trip_ws[Threads.threadid()]) : nothing
                M = use_triplets ? nothing : spzeros(K, dims[a+2, b+1], dims[a+1, b+1])
                for rowUnext in 1:length(Unext), colUcur in 1:length(Ucur)
                    coeff = dF[a+1][rowUnext, colUcur]
                    if iszero(coeff)
                        continue
                    end
                    for j in 1:length(D0)
                        Bmat = _component_inclusion_matrix_cached(
                            c,
                            Ucur[colUcur], D0[j], colUcur, j,
                            Unext[rowUnext], D0[j], rowUnext, j,
                            K
                        )
                        r0 = _block_offset_lookup(offset_tables, c, U_by_a, D_by_b, a+1, b, j, rowUnext) + 1
                        c0 = _block_offset_lookup(offset_tables, c, U_by_a, D_by_b, a,   b, j, colUcur) + 1
                        if use_triplets
                            _append_shifted_scaled_triplets!(ws.I, ws.J, ws.V, Bmat, r0, c0; scale=(sign * coeff))
                        else
                            _accum!(M, r0, c0, (sign * coeff) * Bmat)
                        end
                    end
                end
                dh[a+1, b+1] = use_triplets ? _finalize_sparse!(ws, dims[a+2, b+1], dims[a+1, b+1]) : M
            end
        else
            ws = _HOM_ASSEMBLY_USE_TRIPLETS[] ? trip_ws[1] : nothing
            for b in 0:B
                sign = isodd(b) ? -one(K) : one(K)
                D0 = D_by_b[b+1]
                for a in 0:(A-1)
                    Ucur = U_by_a[a+1]
                    Unext = U_by_a[a+2]
                    use_triplets = _HOM_ASSEMBLY_USE_TRIPLETS[]
                    use_triplets && _reset!(ws)
                    M = use_triplets ? nothing : spzeros(K, dims[a+2, b+1], dims[a+1, b+1])
                    for rowUnext in 1:length(Unext), colUcur in 1:length(Ucur)
                        coeff = dF[a+1][rowUnext, colUcur]
                        if iszero(coeff)
                            continue
                        end
                        for j in 1:length(D0)
                            Bmat = _component_inclusion_matrix_cached(
                                compcache,
                                Ucur[colUcur], D0[j], colUcur, j,
                                Unext[rowUnext], D0[j], rowUnext, j,
                                K
                            )
                            r0 = _block_offset_lookup(offset_tables, compcache, U_by_a, D_by_b, a+1, b, j, rowUnext) + 1
                            c0 = _block_offset_lookup(offset_tables, compcache, U_by_a, D_by_b, a,   b, j, colUcur) + 1
                            if use_triplets
                                _append_shifted_scaled_triplets!(ws.I, ws.J, ws.V, Bmat, r0, c0; scale=(sign * coeff))
                            else
                                _accum!(M, r0, c0, (sign * coeff) * Bmat)
                            end
                        end
                    end
                    dh[a+1, b+1] = use_triplets ? _finalize_sparse!(ws, dims[a+2, b+1], dims[a+1, b+1]) : M
                end
            end
        end

        data = (dims, dv, dh)
        return cache_key === nothing ? data : _cache_hom_bicomplex_store!(cache, cache_key, data)
    end

    """
        ext_dims_via_resolutions(F, dF, E, dE) -> Dict{Int,Int}

    Given an upset resolution F* with differentials dF and a downset resolution E* with
    differentials dE, assemble the total cochain complex C^t = oplus_{a+b=t} Hom(F_a, E^b)
    and return a dictionary mapping total degree t to dim H^t.

    This densifies each sparse block for rank computations.
    """
    function ext_dims_via_resolutions(F::AbstractVector{<:UpsetPresentation{K}},
                                    dF::Vector{SparseMatrixCSC{K,Int}},
                                    E::AbstractVector{<:DownsetCopresentation{K}},
                                    dE::Vector{SparseMatrixCSC{K,Int}};
                                    threads::Bool=false) where {K}
        dimsCt, dts = build_hom_tot_complex(F, dF, E, dE; threads=threads)
        field = (F[1].H === nothing) ? field_from_eltype(K) : F[1].H.field
        
        A = length(F) - 1
        B = length(E) - 1
        tmin, tmax = 0, A + B

        dimsH_vals = Vector{Int}(undef, tmax - tmin + 1)
        if threads && Threads.nthreads() > 1
            Threads.@threads for t in tmin:tmax
                i = t - tmin + 1
                dimC = dimsCt[i]
                r_next = (t < tmax) ? FieldLinAlg.rank_dim(field, dts[i]) : 0
                r_prev = (t > tmin) ? FieldLinAlg.rank_dim(field, dts[i-1]) : 0
                dimsH_vals[i] = dimC - r_next - r_prev
            end
        else
            for t in tmin:tmax
                i = t - tmin + 1
                dimC = dimsCt[i]
                r_next = (t < tmax) ? FieldLinAlg.rank_dim(field, dts[i]) : 0
                r_prev = (t > tmin) ? FieldLinAlg.rank_dim(field, dts[i-1]) : 0
                dimsH_vals[i] = dimC - r_next - r_prev
            end
        end
        dimsH = Dict{Int,Int}()
        sizehint!(dimsH, length(dimsH_vals))
        @inbounds for t in tmin:tmax
            dimsH[t] = dimsH_vals[t - tmin + 1]
        end
        return dimsH
    end

    """
        pi0_count(P, U, D) -> Int

    Return the number of connected components of `U cap D` in the undirected Hasse (cover) graph
    of the finite poset `P`.
    """
    function pi0_count(P::AbstractPoset, U::Upset, D::Downset)
        C = CompCache{Int}(P)
        _, ncomp = _components_cached!(C, U, 0, D, 0)
        return ncomp
    end

end # module HomExtEngine
