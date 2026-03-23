# functoriality.jl -- derived maps and long exact sequence machinery

"""
Functoriality: maps induced on Hom/Ext/Tor by morphisms in either argument.

This submodule should define (move here incrementally):
- ext_map_first, ext_map_second
- tor_map_first, tor_map_second
- connecting morphisms in long exact sequences
- any internal caches used to compute these maps efficiently
"""
module Functoriality

    using LinearAlgebra
    using SparseArrays
    using Base.Threads

    using ...CoreModules: AbstractCoeffField, RealField, ResolutionCache,
                          ResolutionKey3, ResolutionKey5,
                          ExtDoubleComplexPayload, TorDoubleComplexPlanPayload, TorDoubleComplexPayload,
                          AbstractHomSystemCache,
                          _resolution_key3, _resolution_key5, field_from_eltype
    using ...Options: ResolutionOptions, DerivedFunctorOptions
    import ...CoreModules: _append_scaled_triplets!
    using ...FieldLinAlg
    import ...FieldLinAlg: SparseRow

    using ...Modules: PModule, PMorphism, MapLeqQueryBatch, map_leq, map_leq_many, map_leq_many!,
                       _prepare_map_leq_batch_owned, _append_map_leq_many_scaled_triplets!,
                       _accum_map_leq_many_scaled_matvecs!
    import ...FiniteFringe: AbstractPoset, nvertices, leq, poset_equal
    using ...ChainComplexes
    import ...ChainComplexes: sequence_dimensions, sequence_maps, sequence_entry
    using ...AbelianCategories: ShortExactSequence

    import ..Utils
    import ..Utils: compose
    import ..ExtTorSpaces: ExtSpaceProjective, ExtSpaceInjective, ExtSpace,
        TorSpace, TorSpaceSecond, Ext, Tor, ExtInjective, HomSpace, Hom,
        _comparison_P2I, _comparison_I2P
    import ..Resolutions: ProjectiveResolution, InjectiveResolution,
        projective_resolution, injective_resolution, _pad_projective_resolution!,
        lift_injective_chainmap, _base_vertex_groups, _active_upset_indices_cached
    import ..DerivedFunctors: derived_les_summary

    import ..GradedSpaces: degree_range, dim

    import ..ExtTorSpaces: _cochain_vector_from_morphism, split_cochain, _blockdiag_on_hom_cochains, _morphism_to_vector

    @inline function _map_blocks_buffer(M::PModule{K,F,MatT}, n::Int) where {K,F,MatT<:AbstractMatrix{K}}
        return Vector{MatT}(undef, n)
    end

    const _FUNCTORIALITY_USE_DIRECT_COEFF_TRIPLETS = Ref(true)
    const _FUNCTORIALITY_USE_DIRECT_COEFF_MATVECS = Ref(true)
    const _FUNCTORIALITY_USE_COEFF_PLAN_CACHE = Ref(true)
    # Support-pattern batched solves and Hom workspaces are kept behind explicit
    # internal gates. They are enabled by default only for problem sizes large
    # enough to amortize plan construction / scratch setup.
    const _FUNCTORIALITY_USE_SUPPORT_SOLVE_PLANS = Ref(true)
    const _FUNCTORIALITY_USE_HOM_WORKSPACES = Ref(true)
    const _FUNCTORIALITY_DIRECT_COEFF_MIN_NNZ = Ref(8)
    const _FUNCTORIALITY_DIRECT_COCYCLE_MIN_NNZ = Ref(8)
    const _FUNCTORIALITY_SUPPORT_PLAN_MIN_EST_WORK = Ref(256)
    const _FUNCTORIALITY_HOM_WORKSPACE_MIN_SIZE = Ref(512)
    const _FUNCTORIALITY_USE_HOM_SOLVE_WORKSPACE_CACHE = Ref(true)
    const _FUNCTORIALITY_USE_HOM_BASIS_SOLVE_PLAN_CACHE = Ref(true)
    const _FUNCTORIALITY_USE_PROJECTIVE_Q0_JOINT_FALLBACK = Ref(true)
    const _FUNCTORIALITY_USE_TENSOR_COEFF_RESULT_CACHE = Ref(true)

    @inline _coeff_nnz(coeff::SparseMatrixCSC) = nnz(coeff)
    @inline function _coeff_nnz(coeff::AbstractMatrix)
        n = 0
        @inbounds for x in coeff
            iszero(x) || (n += 1)
        end
        return n
    end

    mutable struct _PrecomposeWorkspace{K}
        rhs::Matrix{K}
    end

    @inline _PrecomposeWorkspace(::Type{K}) where {K} = _PrecomposeWorkspace{K}(zeros(K, 0, 0))

    mutable struct _PostcomposeWorkspace{K}
        rhs::Matrix{K}
    end

    @inline _PostcomposeWorkspace(::Type{K}) where {K} = _PostcomposeWorkspace{K}(zeros(K, 0, 0))

    mutable struct _HomSolveWorkspaceEntry{K}
        pre::_PrecomposeWorkspace{K}
        post::_PostcomposeWorkspace{K}
    end

    abstract type _AbstractParticularSolvePlan{K} end

    mutable struct _HomSolvePlanEntry{K}
        plan::_AbstractParticularSolvePlan{K}
        pre::_PrecomposeWorkspace{K}
        post::_PostcomposeWorkspace{K}
    end

    const _HOM_SOLVE_WORKSPACE_CACHE = Dict{UInt,Any}()
    const _HOM_SOLVE_WORKSPACE_LOCK = ReentrantLock()
    const _HOM_BASIS_SOLVE_PLAN_CACHE = Dict{UInt,Any}()
    const _HOM_BASIS_SOLVE_PLAN_LOCK = ReentrantLock()

    @inline function _rhs_buffer!(
        ::Nothing,
        ::Type{K},
        nrows::Int,
        ncols::Int,
    ) where {K}
        return zeros(K, nrows, ncols)
    end

    @inline function _rhs_buffer!(
        ws::Union{_PrecomposeWorkspace{K},_PostcomposeWorkspace{K}},
        ::Type{K},
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

    @inline function _hom_solve_workspace_entry(H::HomSpace{K}) where {K}
        if !_FUNCTORIALITY_USE_HOM_SOLVE_WORKSPACE_CACHE[]
            return _HomSolveWorkspaceEntry{K}(_PrecomposeWorkspace(K), _PostcomposeWorkspace(K))
        end
        key = UInt(objectid(H.basis_matrix))
        lock(_HOM_SOLVE_WORKSPACE_LOCK)
        try
            return get!(_HOM_SOLVE_WORKSPACE_CACHE, key) do
                _HomSolveWorkspaceEntry{K}(_PrecomposeWorkspace(K), _PostcomposeWorkspace(K))
            end
        finally
            unlock(_HOM_SOLVE_WORKSPACE_LOCK)
        end
    end

    @inline function _hom_solve_plan_entry(H::HomSpace{K}) where {K}
        key = UInt(objectid(H.basis_matrix))
        if !_FUNCTORIALITY_USE_HOM_BASIS_SOLVE_PLAN_CACHE[]
            return _HomSolvePlanEntry{K}(
                _particular_solve_plan(H.dom.field, H.basis_matrix),
                _PrecomposeWorkspace(K),
                _PostcomposeWorkspace(K),
            )
        end
        lock(_HOM_BASIS_SOLVE_PLAN_LOCK)
        try
            return get!(_HOM_BASIS_SOLVE_PLAN_CACHE, key) do
                _HomSolvePlanEntry{K}(
                    _particular_solve_plan(H.dom.field, H.basis_matrix),
                    _PrecomposeWorkspace(K),
                    _PostcomposeWorkspace(K),
                )
            end
        finally
            unlock(_HOM_BASIS_SOLVE_PLAN_LOCK)
        end
    end

    struct _ExactParticularSolvePlan{K,F<:AbstractCoeffField} <: _AbstractParticularSolvePlan{K}
        field::F
        transform::Matrix{K}
        pivcols::Vector{Int}
        rank::Int
        nvars::Int
    end

    struct _RealParticularSolvePlan{K,F<:AbstractCoeffField} <: _AbstractParticularSolvePlan{K}
        field::F
        A::Matrix{K}
    end

    struct _SupportSolvePlanKey
        matrix_id::UInt
        support_hash::UInt
        nrows::Int
        ncols::Int
    end

    const _SUPPORT_SOLVE_PLAN_CACHE = [Dict{_SupportSolvePlanKey,Any}() for _ in 1:max(1, Threads.maxthreadid())]
    @inline _support_plan_shard() = _SUPPORT_SOLVE_PLAN_CACHE[min(length(_SUPPORT_SOLVE_PLAN_CACHE), max(1, Threads.threadid()))]

    @inline function _support_hash(cols::AbstractVector{<:Integer})
        h = hash(length(cols), UInt(0x9172b7a9))
        @inbounds for c in cols
            h = hash(Int(c), h)
        end
        return h
    end

    @inline _support_solve_est_work(nrows::Int, ncols::Int, nrhs::Int) =
        max(1, nrows) * max(1, ncols) * max(1, nrhs)

    @inline function _use_support_solve_plan(nrows::Int, ncols::Int, nrhs::Int)
        _FUNCTORIALITY_USE_SUPPORT_SOLVE_PLANS[] || return false
        return _support_solve_est_work(nrows, ncols, nrhs) >= _FUNCTORIALITY_SUPPORT_PLAN_MIN_EST_WORK[]
    end

    @inline function _use_hom_workspace(nrows::Int, ncols::Int)
        _FUNCTORIALITY_USE_HOM_WORKSPACES[] || return false
        return max(1, nrows) * max(1, ncols) >= _FUNCTORIALITY_HOM_WORKSPACE_MIN_SIZE[]
    end

    @inline function _cached_particular_solve_plan(field::AbstractCoeffField,
                                                   A::AbstractMatrix{K},
                                                   matrix_id::UInt,
                                                   support_hash::UInt) where {K}
        shard = _support_plan_shard()
        key = _SupportSolvePlanKey(matrix_id, support_hash, size(A, 1), size(A, 2))
        return get!(shard, key) do
            _particular_solve_plan(field, A)
        end
    end

    @inline function _dense_identity(::Type{K}, n::Int) where {K}
        I = zeros(K, n, n)
        @inbounds for i in 1:n
            I[i, i] = one(K)
        end
        return I
    end

    function _particular_solve_plan(field::RealField, A::AbstractMatrix{K}) where {K}
        return _RealParticularSolvePlan{K,typeof(field)}(field, Matrix(A))
    end

    function _particular_solve_plan(field::AbstractCoeffField, A::AbstractMatrix{K}) where {K}
        m, n = size(A)
        Aug = hcat(Matrix(A), _dense_identity(K, m))
        R, pivs_all = FieldLinAlg.rref(field, Aug)
        pivs = Int[]
        for p in pivs_all
            p <= n && push!(pivs, p)
        end
        return _ExactParticularSolvePlan{K,typeof(field)}(
            field,
            Matrix(@view R[:, n+1:n+m]),
            pivs,
            length(pivs),
            n,
        )
    end

    function _solve_particular(plan::_RealParticularSolvePlan{K}, B::AbstractMatrix{K}) where {K}
        X = plan.A \ Matrix(B)
        R = plan.A * X - B
        maxabs = isempty(R) ? zero(K) : maximum(abs, R)
        tol = plan.field.atol + plan.field.rtol * (isempty(plan.A) ? zero(K) : opnorm(plan.A, 1))
        maxabs <= tol || error("solve_particular: inconsistent system (residual=$maxabs, tol=$tol)")
        return X
    end

    @inline function _projective_q0_step_inconsistent(err)
        err isa ErrorException || return false
        msg = sprint(showerror, err)
        return occursin("solve_particular: inconsistent system", msg) ||
               occursin("_lift_q0_parts_to_chainmap_coeff: inconsistent constraints", msg)
    end

    @inline function _dense_from_sparse_rows_local(::Type{K},
                                                   rows::Vector{SparseRow{K}},
                                                   ncols::Int) where {K}
        A = zeros(K, length(rows), ncols)
        @inbounds for i in eachindex(rows)
            row = rows[i]
            for t in eachindex(row.idx)
                A[i, row.idx[t]] = row.val[t]
            end
        end
        return A
    end

    @inline function _projective_var_index(active::Vector{Vector{Int}},
                                           base_vertices::Vector{Int},
                                           nrows::Int;
                                           start::Int = 0)
        var_idx = zeros(Int, nrows, length(base_vertices))
        nunk = start
        @inbounds for col in eachindex(base_vertices)
            act = active[base_vertices[col]]
            for row in act
                nunk += 1
                var_idx[row, col] = nunk
            end
        end
        return var_idx, nunk
    end

    @inline function _active_local_positions(active::Vector{Vector{Int}},
                                             base_vertices::Vector{Int})
        pos = zeros(Int, length(base_vertices))
        @inbounds for i in eachindex(base_vertices)
            act = active[base_vertices[i]]
            for p in eachindex(act)
                if act[p] == i
                    pos[i] = p
                    break
                end
            end
            pos[i] == 0 && error("_active_local_positions: missing generator $i in active list")
        end
        return pos
    end

    function _solve_projective_q0_q1_joint_coeff(resL::ProjectiveResolution{K},
                                                 resM::ProjectiveResolution{K},
                                                 alpha_parts::Vector{Vector{K}},
                                                 active_M::Vector{Vector{Vector{Int}}}) where {K}
        length(active_M) >= 2 || error("_solve_projective_q0_q1_joint_coeff: expected active data for degrees 0 and 1")

        if resL === resM && f.dom === f.cod
            scalar = _pmorphism_scalar_multiple_identity(f)
            scalar === nothing || return _scalar_chainmap_coeffs(resL, scalar, upto)
        end

        dom_gens_0 = resL.gens[1]
        dom_gens_1 = resL.gens[2]
        cod_bases_0 = resM.gens[1]
        cod_bases_1 = resM.gens[2]

        m0 = length(cod_bases_0)
        n0 = length(dom_gens_0)
        m1 = length(cod_bases_1)
        n1 = length(dom_gens_1)

        var0, nunk0 = _projective_var_index(active_M[1], dom_gens_0, m0)
        var1, nunk = _projective_var_index(active_M[2], dom_gens_1, m1; start=nunk0)

        rows = Vector{SparseRow{K}}()
        rhs = K[]
        sizehint!(rows, sum(length(alpha_parts[i]) for i in eachindex(alpha_parts)) + max(1, m0 * n1))
        sizehint!(rhs, length(rows))

        @inbounds for col in 1:n0
            u = dom_gens_0[col]
            act = active_M[1][u]
            A_u = Matrix(resM.aug.comps[u])
            rhs_col = alpha_parts[col]
            for rpos in 1:length(rhs_col)
                idx = Int[]
                vals = K[]
                sizehint!(idx, length(act))
                sizehint!(vals, length(act))
                for (apos, row) in enumerate(act)
                    a = A_u[rpos, apos]
                    iszero(a) && continue
                    push!(idx, var0[row, col])
                    push!(vals, a)
                end
                push!(rows, SparseRow{K}(idx, vals))
                push!(rhs, rhs_col[rpos])
            end
        end

        D1M = Matrix(resM.d_mat[1])
        D1L = Matrix(resL.d_mat[1])
        @inbounds for col in 1:n1
            u = dom_gens_1[col]
            allowed = active_M[2][u]
            src_idx = Int[]
            src_val = K[]
            for p in 1:n0
                d = D1L[p, col]
                iszero(d) && continue
                push!(src_idx, p)
                push!(src_val, d)
            end
            for r in 1:m0
                idx = Int[]
                vals = K[]
                sizehint!(idx, length(allowed) + length(src_idx))
                sizehint!(vals, length(allowed) + length(src_idx))
                for row in allowed
                    a = D1M[r, row]
                    iszero(a) && continue
                    push!(idx, var1[row, col])
                    push!(vals, a)
                end
                for t in eachindex(src_idx)
                    vidx = var0[r, src_idx[t]]
                    vidx == 0 && continue
                    push!(idx, vidx)
                    push!(vals, -src_val[t])
                end
                isempty(idx) && continue
                push!(rows, SparseRow{K}(idx, vals))
                push!(rhs, zero(K))
            end
        end

        A = _dense_from_sparse_rows_local(K, rows, nunk)
        plan = _particular_solve_plan(resM.M.field, A)
        X = _solve_particular(plan, reshape(rhs, :, 1))[:, 1]

        I0 = Int[]
        J0 = Int[]
        V0 = K[]
        @inbounds for col in 1:n0, row in active_M[1][dom_gens_0[col]]
            c = X[var0[row, col]]
            iszero(c) || (push!(I0, row); push!(J0, col); push!(V0, c))
        end

        I1 = Int[]
        J1 = Int[]
        V1 = K[]
        @inbounds for col in 1:n1, row in active_M[2][dom_gens_1[col]]
            c = X[var1[row, col]]
            iszero(c) || (push!(I1, row); push!(J1, col); push!(V1, c))
        end

        F0 = sparse(I0, J0, V0, m0, n0)
        F1 = sparse(I1, J1, V1, m1, n1)
        return F0, F1
    end

    function _projective_chain_step_coeff(resL::ProjectiveResolution{K},
                                          resM::ProjectiveResolution{K},
                                          Fprev::SparseMatrixCSC{K, Int},
                                          active_M::Vector{Vector{Vector{Int}}},
                                          k::Int) where {K}
        DkM = Matrix(resM.d_mat[k])
        DkL = Matrix(resL.d_mat[k])
        RHS = Fprev * DkL

        cod_bases_k = resM.gens[k + 1]
        dom_bases_k = resL.gens[k + 1]

        mk = length(cod_bases_k)
        nk = length(dom_bases_k)

        Ik = Int[]
        Jk = Int[]
        Vk = K[]

        groupsk = _base_vertex_groups(resM.M.Q, dom_bases_k)
        @inbounds for u in 1:length(groupsk)
            cols = groupsk[u]
            isempty(cols) && continue
            allowed = active_M[k + 1][u]

            if isempty(allowed)
                @inbounds for col in cols
                    any(!iszero, @view RHS[:, col]) &&
                        error("_lift_q0_parts_to_chainmap_coeff: inconsistent constraints at (k=$k, dom_summand=$col).")
                end
                continue
            end

            A = @view DkM[:, allowed]
            nrhs = length(cols)
            if _use_support_solve_plan(size(A, 1), size(A, 2), nrhs)
                plan = _cached_particular_solve_plan(
                    resM.M.field,
                    A,
                    UInt(objectid(resM.d_mat[k])),
                    _support_hash(allowed),
                )
                X = _solve_particular(plan, Matrix(@view RHS[:, cols]))
                @inbounds for (jj, col) in enumerate(cols), (pos, j) in enumerate(allowed)
                    c = X[pos, jj]
                    iszero(c) || (push!(Ik, j); push!(Jk, col); push!(Vk, c))
                end
            else
                @inbounds for col in cols
                    x = Utils.solve_particular(resM.M.field, A, reshape(@view(RHS[:, col]), :, 1))
                    for (pos, j) in enumerate(allowed)
                        c = x[pos, 1]
                        iszero(c) || (push!(Ik, j); push!(Jk, col); push!(Vk, c))
                    end
                end
            end
        end

        return sparse(Ik, Jk, Vk, mk, nk)
    end

    function _solve_particular(plan::_ExactParticularSolvePlan{K}, B::AbstractMatrix{K}) where {K}
        size(B, 1) == size(plan.transform, 2) || error("_solve_particular: row mismatch")
        Y = plan.transform * Matrix(B)
        if plan.rank < size(Y, 1)
            @inbounds for j in 1:size(Y, 2), i in (plan.rank + 1):size(Y, 1)
                iszero(Y[i, j]) || error("solve_particular: inconsistent system")
            end
        end
        X = zeros(K, plan.nvars, size(Y, 2))
        @inbounds for (row, pcol) in enumerate(plan.pivcols)
            X[pcol, :] = Y[row, :]
        end
        return X
    end

    struct _CoeffMapPlanKey
        module_id::UInt
        dom_id::UInt
        cod_id::UInt
        pattern_hash::UInt
        mode::UInt8
        nrows::Int
        ncols::Int
    end

    struct _CoeffMapPlan
        coeff_rows::Vector{Int}
        coeff_cols::Vector{Int}
        row_block_ids::Vector{Int}
        col_block_ids::Vector{Int}
        coeff_slots::Vector{Int}
        batch::MapLeqQueryBatch
    end

    struct _TensorCoeffCacheKey
        module_id::UInt
        dom_id::UInt
        cod_id::UInt
        dom_off_id::UInt
        cod_off_id::UInt
        coeff_id::UInt
    end

    const _COEFF_MAP_PLAN_CACHE = [Dict{_CoeffMapPlanKey,_CoeffMapPlan}() for _ in 1:max(1, Threads.maxthreadid())]
    const _TENSOR_COEFF_PLAN_CACHE_LOCK = ReentrantLock()
    const _TENSOR_COEFF_PLAN_CACHES = WeakKeyDict{Any, Dict{_TensorCoeffCacheKey, Any}}()
    const _TENSOR_COEFF_RESULT_CACHE_LOCK = ReentrantLock()
    const _TENSOR_COEFF_RESULT_CACHES = WeakKeyDict{Any, Dict{_TensorCoeffCacheKey, Any}}()

    @inline _coeff_plan_shard() = _COEFF_MAP_PLAN_CACHE[min(length(_COEFF_MAP_PLAN_CACHE), max(1, Threads.threadid()))]

    @inline function _coeff_sparse_pattern_hash(coeff::SparseMatrixCSC)
        h = hash(size(coeff, 1), UInt(0x6f7b9a1d))
        h = hash(size(coeff, 2), h)
        h = hash(coeff.colptr, h)
        return hash(coeff.rowval, h)
    end

    @inline function _coeff_plan_key(M::PModule,
                                     dom_gens::Vector{Int},
                                     cod_gens::Vector{Int},
                                     coeff::SparseMatrixCSC,
                                     mode::UInt8)
        return _CoeffMapPlanKey(UInt(objectid(M)),
                                UInt(objectid(dom_gens)),
                                UInt(objectid(cod_gens)),
                                _coeff_sparse_pattern_hash(coeff),
                                mode,
                                size(coeff, 1),
                                size(coeff, 2))
    end

    @inline function _tensor_coeff_cache_key(M::PModule,
                                             dom_bases::Vector{Int},
                                             cod_bases::Vector{Int},
                                             dom_offsets::Vector{Int},
                                             cod_offsets::Vector{Int},
                                             coeff::AbstractMatrix)
        return _TensorCoeffCacheKey(UInt(objectid(M)),
                                    UInt(objectid(dom_bases)),
                                    UInt(objectid(cod_bases)),
                                    UInt(objectid(dom_offsets)),
                                    UInt(objectid(cod_offsets)),
                                    UInt(objectid(coeff)))
    end

    @inline function _coeff_plan_scales(coeff::SparseMatrixCSC{K,Int},
                                        plan::_CoeffMapPlan) where {K}
        nzv = nonzeros(coeff)
        scales = Vector{K}(undef, length(plan.coeff_slots))
        @inbounds for k in eachindex(scales)
            scales[k] = nzv[plan.coeff_slots[k]]
        end
        return scales
    end

    @inline function _coeff_plan_scales(coeff::AbstractMatrix{K},
                                        plan::_CoeffMapPlan) where {K}
        scales = Vector{K}(undef, length(plan.coeff_rows))
        @inbounds for k in eachindex(scales)
            scales[k] = coeff[plan.coeff_rows[k], plan.coeff_cols[k]]
        end
        return scales
    end

    function _build_precompose_coeff_plan(coeff::SparseMatrixCSC{K,Int},
                                          dom_gens::Vector{Int},
                                          cod_gens::Vector{Int}) where {K}
        coeff_rows = Int[]
        coeff_cols = Int[]
        row_blocks = Int[]
        col_blocks = Int[]
        coeff_slots = Int[]
        pairs = Tuple{Int,Int}[]
        @inbounds for col in 1:min(length(dom_gens), size(coeff, 2))
            for ptr in coeff.colptr[col]:(coeff.colptr[col + 1] - 1)
                row = coeff.rowval[ptr]
                iszero(coeff.nzval[ptr]) && continue
                push!(coeff_rows, row)
                push!(coeff_cols, col)
                push!(row_blocks, col)
                push!(col_blocks, row)
                push!(coeff_slots, ptr)
                push!(pairs, (cod_gens[row], dom_gens[col]))
            end
        end
        return _CoeffMapPlan(coeff_rows, coeff_cols, row_blocks, col_blocks, coeff_slots,
                             _prepare_map_leq_batch_owned(pairs))
    end

    function _build_precompose_coeff_plan(coeff::AbstractMatrix{K},
                                          dom_gens::Vector{Int},
                                          cod_gens::Vector{Int}) where {K}
        coeff_rows = Int[]
        coeff_cols = Int[]
        row_blocks = Int[]
        col_blocks = Int[]
        pairs = Tuple{Int,Int}[]
        @inbounds for col in 1:length(dom_gens), row in 1:length(cod_gens)
            c = coeff[row, col]
            iszero(c) && continue
            push!(coeff_rows, row)
            push!(coeff_cols, col)
            push!(row_blocks, col)
            push!(col_blocks, row)
            push!(pairs, (cod_gens[row], dom_gens[col]))
        end
        return _CoeffMapPlan(coeff_rows, coeff_cols, row_blocks, col_blocks, Int[],
                             _prepare_map_leq_batch_owned(pairs))
    end

    function _build_tensor_coeff_plan(coeff::SparseMatrixCSC{K,Int},
                                      dom_bases::Vector{Int},
                                      cod_bases::Vector{Int}) where {K}
        coeff_rows = Int[]
        coeff_cols = Int[]
        row_blocks = Int[]
        col_blocks = Int[]
        coeff_slots = Int[]
        pairs = Tuple{Int,Int}[]
        @inbounds for col in 1:min(length(dom_bases), size(coeff, 2))
            for ptr in coeff.colptr[col]:(coeff.colptr[col + 1] - 1)
                row = coeff.rowval[ptr]
                iszero(coeff.nzval[ptr]) && continue
                push!(coeff_rows, row)
                push!(coeff_cols, col)
                push!(row_blocks, row)
                push!(col_blocks, col)
                push!(coeff_slots, ptr)
                push!(pairs, (dom_bases[col], cod_bases[row]))
            end
        end
        return _CoeffMapPlan(coeff_rows, coeff_cols, row_blocks, col_blocks, coeff_slots,
                             _prepare_map_leq_batch_owned(pairs))
    end

    function _build_tensor_coeff_plan(coeff::AbstractMatrix{K},
                                      dom_bases::Vector{Int},
                                      cod_bases::Vector{Int}) where {K}
        coeff_rows = Int[]
        coeff_cols = Int[]
        row_blocks = Int[]
        col_blocks = Int[]
        pairs = Tuple{Int,Int}[]
        @inbounds for col in 1:length(dom_bases), row in 1:length(cod_bases)
            c = coeff[row, col]
            iszero(c) && continue
            push!(coeff_rows, row)
            push!(coeff_cols, col)
            push!(row_blocks, row)
            push!(col_blocks, col)
            push!(pairs, (dom_bases[col], cod_bases[row]))
        end
        return _CoeffMapPlan(coeff_rows, coeff_cols, row_blocks, col_blocks, Int[],
                             _prepare_map_leq_batch_owned(pairs))
    end

    function _build_cocycle_coeff_plan(coeff::SparseMatrixCSC{K,Int},
                                       dom_bases::Vector{Int},
                                       mid_bases::Vector{Int},
                                       Q::AbstractPoset) where {K}
        coeff_rows = Int[]
        coeff_cols = Int[]
        row_blocks = Int[]
        col_blocks = Int[]
        coeff_slots = Int[]
        pairs = Tuple{Int,Int}[]
        @inbounds for col in 1:min(length(dom_bases), size(coeff, 2))
            u = dom_bases[col]
            for ptr in coeff.colptr[col]:(coeff.colptr[col + 1] - 1)
                row = coeff.rowval[ptr]
                c = coeff.nzval[ptr]
                iszero(c) && continue
                v = mid_bases[row]
                leq(Q, v, u) || continue
                push!(coeff_rows, row)
                push!(coeff_cols, col)
                push!(row_blocks, col)
                push!(col_blocks, row)
                push!(coeff_slots, ptr)
                push!(pairs, (v, u))
            end
        end
        return _CoeffMapPlan(coeff_rows, coeff_cols, row_blocks, col_blocks, coeff_slots,
                             _prepare_map_leq_batch_owned(pairs))
    end

    function _build_cocycle_coeff_plan(coeff::AbstractMatrix{K},
                                       dom_bases::Vector{Int},
                                       mid_bases::Vector{Int},
                                       Q::AbstractPoset) where {K}
        coeff_rows = Int[]
        coeff_cols = Int[]
        row_blocks = Int[]
        col_blocks = Int[]
        pairs = Tuple{Int,Int}[]
        @inbounds for col in 1:length(dom_bases)
            u = dom_bases[col]
            for row in 1:length(mid_bases)
                c = coeff[row, col]
                iszero(c) && continue
                v = mid_bases[row]
                leq(Q, v, u) || continue
                push!(coeff_rows, row)
                push!(coeff_cols, col)
                push!(row_blocks, col)
                push!(col_blocks, row)
                push!(pairs, (v, u))
            end
        end
        return _CoeffMapPlan(coeff_rows, coeff_cols, row_blocks, col_blocks, Int[],
                             _prepare_map_leq_batch_owned(pairs))
    end

    function _cached_sparse_coeff_plan(M::PModule,
                                       dom_gens::Vector{Int},
                                       cod_gens::Vector{Int},
                                       coeff::SparseMatrixCSC,
                                       mode::UInt8,
                                       builder::Function)
        if !_FUNCTORIALITY_USE_COEFF_PLAN_CACHE[]
            return builder(coeff, dom_gens, cod_gens)
        end
        key = _coeff_plan_key(M, dom_gens, cod_gens, coeff, mode)
        shard = _coeff_plan_shard()
        return get!(shard, key) do
            builder(coeff, dom_gens, cod_gens)
        end
    end

    @inline function _precompose_coeff_plan(N::PModule,
                                            dom_gens::Vector{Int},
                                            cod_gens::Vector{Int},
                                            coeff::SparseMatrixCSC)
        return _cached_sparse_coeff_plan(N, dom_gens, cod_gens, coeff, 0x01, _build_precompose_coeff_plan)
    end

    @inline _precompose_coeff_plan(N::PModule,
                                   dom_gens::Vector{Int},
                                   cod_gens::Vector{Int},
                                   coeff::AbstractMatrix) =
        _build_precompose_coeff_plan(coeff, dom_gens, cod_gens)

    @inline function _tensor_coeff_plan(M::PModule,
                                        dom_bases::Vector{Int},
                                        cod_bases::Vector{Int},
                                        coeff::SparseMatrixCSC)
        return _cached_sparse_coeff_plan(M, dom_bases, cod_bases, coeff, 0x02, _build_tensor_coeff_plan)
    end

    @inline _tensor_coeff_plan(M::PModule,
                               dom_bases::Vector{Int},
                               cod_bases::Vector{Int},
                               coeff::AbstractMatrix) =
        _build_tensor_coeff_plan(coeff, dom_bases, cod_bases)

    @inline function _tensor_coeff_plan_cached(M::PModule,
                                               dom_bases::Vector{Int},
                                               cod_bases::Vector{Int},
                                               dom_offsets::Vector{Int},
                                               cod_offsets::Vector{Int},
                                               coeff::AbstractMatrix,
                                               ::Nothing)
        return _tensor_coeff_plan(M, dom_bases, cod_bases, coeff)
    end

    function _tensor_coeff_plan_cached(M::PModule,
                                       dom_bases::Vector{Int},
                                       cod_bases::Vector{Int},
                                       dom_offsets::Vector{Int},
                                       cod_offsets::Vector{Int},
                                       coeff::AbstractMatrix,
                                       cache::AbstractHomSystemCache)
        return lock(_TENSOR_COEFF_PLAN_CACHE_LOCK) do
            shard = get!(_TENSOR_COEFF_PLAN_CACHES, cache) do
                Dict{_TensorCoeffCacheKey, Any}()
            end
            key = _tensor_coeff_cache_key(M, dom_bases, cod_bases, dom_offsets, cod_offsets, coeff)
            cached = get(shard, key, nothing)
            cached === nothing || return cached::_CoeffMapPlan
            plan = _tensor_coeff_plan(M, dom_bases, cod_bases, coeff)
            shard[key] = plan
            return plan
        end
    end

    @inline function _tensor_coeff_result_cached(builder::Function,
                                                 M::PModule,
                                                 dom_bases::Vector{Int},
                                                 cod_bases::Vector{Int},
                                                 dom_offsets::Vector{Int},
                                                 cod_offsets::Vector{Int},
                                                 coeff::AbstractMatrix,
                                                 ::Nothing)
        return builder()
    end

    function _tensor_coeff_result_cached(builder::Function,
                                         M::PModule{K},
                                         dom_bases::Vector{Int},
                                         cod_bases::Vector{Int},
                                         dom_offsets::Vector{Int},
                                         cod_offsets::Vector{Int},
                                         coeff::AbstractMatrix{K},
                                         cache::AbstractHomSystemCache) where {K}
        _FUNCTORIALITY_USE_TENSOR_COEFF_RESULT_CACHE[] || return builder()
        return lock(_TENSOR_COEFF_RESULT_CACHE_LOCK) do
            shard = get!(_TENSOR_COEFF_RESULT_CACHES, cache) do
                Dict{_TensorCoeffCacheKey, Any}()
            end
            key = _tensor_coeff_cache_key(M, dom_bases, cod_bases, dom_offsets, cod_offsets, coeff)
            cached = get(shard, key, nothing)
            cached === nothing || return cached::SparseMatrixCSC{K,Int}
            block = builder()
            shard[key] = block
            return block
        end
    end

    @inline function _cocycle_coeff_plan(M::PModule,
                                         dom_bases::Vector{Int},
                                         mid_bases::Vector{Int},
                                         coeff::SparseMatrixCSC)
        return _cached_sparse_coeff_plan(M, dom_bases, mid_bases, coeff, 0x03,
                                         (A, d, c) -> _build_cocycle_coeff_plan(A, d, c, M.Q))
    end

    @inline _cocycle_coeff_plan(M::PModule,
                                dom_bases::Vector{Int},
                                mid_bases::Vector{Int},
                                coeff::AbstractMatrix) =
        _build_cocycle_coeff_plan(coeff, dom_bases, mid_bases, M.Q)




    # ----------------------------
    # Functoriality in second argument: g : N -> N2 induces Ext(M,N) -> Ext(M,N2)
    # ----------------------------

    function ext_map_second(E1::ExtSpaceProjective{K}, E2::ExtSpaceProjective{K}, g::PMorphism{K}; t::Int) where {K}
        # Both E1 and E2 must be built from the same projective resolution (same M).
        @assert length(E1.res.gens) == length(E2.res.gens)
        gens_t = E1.res.gens[t+1]
        F = _blockdiag_on_hom_cochains(g, gens_t, E1.offsets[t+1], E2.offsets[t+1])
        return ChainComplexes.induced_map_on_cohomology(E1.cohom[t+1], E2.cohom[t+1], F)
    end

    # -------------------------------------------------------------------------------------
    # Functoriality: Ext via projective resolutions
    # -------------------------------------------------------------------------------------

    # Internal: given a map of projectives P_dom -> P_cod expressed in the indicator-summand
    # bases by the coefficient matrix `coeff`, build the induced map on Hom cochains
    #
    #   Hom(P_cod, N) -> Hom(P_dom, N)
    #
    # by precomposition. This is the same pattern as _build_hom_differential, but with an
    # arbitrary coefficient matrix instead of a differential from a resolution.
    function _precompose_on_hom_cochains_from_projective_coeff(
        N::PModule{K},
        dom_gens::Vector{Int},
        cod_gens::Vector{Int},
        dom_offsets::Vector{Int},
        cod_offsets::Vector{Int},
        coeff::AbstractMatrix{K}
    ) where {K}
        out_dim = dom_offsets[end]
        in_dim  = cod_offsets[end]

        I = Int[]
        J = Int[]
        V = K[]

        if _FUNCTORIALITY_USE_DIRECT_COEFF_TRIPLETS[] && _coeff_nnz(coeff) >= _FUNCTORIALITY_DIRECT_COEFF_MIN_NNZ[]
            plan = _precompose_coeff_plan(N, dom_gens, cod_gens, coeff)
            scales = _coeff_plan_scales(coeff, plan)
            _append_map_leq_many_scaled_triplets!(I, J, V, N, plan.batch,
                                                  plan.row_block_ids, plan.col_block_ids,
                                                  dom_offsets, cod_offsets, scales)
            return sparse(I, J, V, out_dim, in_dim)
        end

        # Fallback path for A/B parity: same mathematics, but it still materializes
        # the map blocks eagerly.
        if issparse(coeff)
            Icoeff, Jcoeff, Vcoeff = findnz(coeff)  # rows, cols, vals
            pairs = Vector{Tuple{Int,Int}}(undef, length(Vcoeff))
            @inbounds for k in eachindex(Vcoeff)
                pairs[k] = (cod_gens[Icoeff[k]], dom_gens[Jcoeff[k]])
            end
            pair_batch = _prepare_map_leq_batch_owned(pairs)
            map_blocks = _map_blocks_buffer(N, length(pairs))
            map_leq_many!(map_blocks, N, pair_batch)
            @inbounds for k in eachindex(Vcoeff)
                j = Icoeff[k]   # cod index
                i = Jcoeff[k]   # dom index
                c = Vcoeff[k]
                iszero(c) && continue

                A = map_blocks[k]
                _append_scaled_triplets!(I, J, V, A, dom_offsets[i], cod_offsets[j]; scale=c)
            end
        else
            # Dense coeff -> scan directly (avoid sparse(coeff) + findnz intermediates).
            dom_idx = Int[]
            cod_idx = Int[]
            vals = K[]
            pairs = Tuple{Int,Int}[]
            @inbounds for i in 1:length(dom_gens)
                for j in 1:length(cod_gens)
                    c = coeff[j, i]
                    iszero(c) && continue

                    push!(dom_idx, i)
                    push!(cod_idx, j)
                    push!(vals, c)
                    push!(pairs, (cod_gens[j], dom_gens[i]))
                end
            end
            pair_batch = _prepare_map_leq_batch_owned(pairs)
            map_blocks = _map_blocks_buffer(N, length(pairs))
            map_leq_many!(map_blocks, N, pair_batch)
            @inbounds for k in eachindex(vals)
                i = dom_idx[k]
                j = cod_idx[k]
                _append_scaled_triplets!(I, J, V, map_blocks[k], dom_offsets[i], cod_offsets[j]; scale=vals[k])
            end
        end

        return sparse(I, J, V, out_dim, in_dim)
    end



    # Internal: lift a module morphism f : M -> Mp to a chain map between the chosen
    # projective resolutions resM -> resMp. The output is a vector F where
    #
    #   F[k+1] is the coefficient matrix of the chain map component P_k(M) -> P_k(Mp)
    #
    # written in the indicator-summand bases of resM and resMp.
    #
    # This uses the same deterministic lifting routine used for Yoneda products (q = 0).
    function _lift_pmodule_map_to_projective_resolution_chainmap_coeff(
        resM::ProjectiveResolution{K},
        resMp::ProjectiveResolution{K},
        f::PMorphism{K};
        upto::Int
    ) where {K}
        @assert f.dom === resM.M
        @assert f.cod === resMp.M
        @assert upto >= 0
        @assert upto <= (length(resM.Pmods) - 1)
        @assert upto <= (length(resMp.Pmods) - 1)

        # View f as a degree-0 cocycle in Hom(P_0(M), Mp) by composing with the
        # augmentation P_0(M) -> M. For q = 0 we can read the cochain blocks
        # directly from that morphism without materializing the full Ext model.
        alpha_mor = compose(f, resM.aug)
        dom_gens_0 = resM.gens[1]
        active_dom_0 = _active_upset_indices_cached(resM.M.Q, dom_gens_0)
        local_pos = _active_local_positions(active_dom_0, dom_gens_0)
        alpha_parts = Vector{Vector{K}}(undef, length(dom_gens_0))
        @inbounds for i in eachindex(dom_gens_0)
            u = dom_gens_0[i]
            if resMp.M.dims[u] == 0
                alpha_parts[i] = K[]
            else
                alpha_parts[i] = copy(@view alpha_mor.comps[u][:, local_pos[i]])
            end
        end

        return _lift_q0_parts_to_chainmap_coeff(resM, resMp, alpha_parts; upto=upto)
    end

    function _lift_q0_parts_to_chainmap_coeff(resL::ProjectiveResolution{K},
                                              resM::ProjectiveResolution{K},
                                              alpha_parts::Vector{Vector{K}};
                                              upto::Int = 0) where {K}
        if upto < 0
            error("_lift_q0_parts_to_chainmap_coeff: keyword `upto` must be >= 0.")
        end

        active_M = Vector{Vector{Vector{Int}}}(undef, upto + 1)
        for k in 0:upto
            cod_bases_k = resM.gens[k + 1]
            active_M[k + 1] = _active_upset_indices_cached(resM.M.Q, cod_bases_k)
        end

        F = Vector{SparseMatrixCSC{K, Int}}(undef, upto + 1)

        dom_gens_0 = resL.gens[1]
        cod_bases_0 = resM.gens[1]
        m0 = length(cod_bases_0)
        n0 = length(dom_gens_0)

        I0 = Int[]
        J0 = Int[]
        V0 = K[]

        groups0 = _base_vertex_groups(resM.M.Q, dom_gens_0)
        @inbounds for u in 1:length(groups0)
            cols = groups0[u]
            isempty(cols) && continue
            act = active_M[1][u]
            isempty(act) && continue

            A_u = Matrix(resM.aug.comps[u])
            nrhs = length(cols)
            if _use_support_solve_plan(size(A_u, 1), size(A_u, 2), nrhs)
                rhslen = isempty(cols) ? 0 : length(alpha_parts[cols[1]])
                B = zeros(K, rhslen, nrhs)
                @inbounds for (jj, col) in enumerate(cols)
                    rhs = alpha_parts[col]
                    isempty(rhs) || copyto!(@view(B[:, jj]), rhs)
                end
                plan = _cached_particular_solve_plan(
                    resM.M.field,
                    A_u,
                    UInt(objectid(resM.aug.comps[u])),
                    UInt(length(act)),
                )
                X = _solve_particular(plan, B)
                @inbounds for (jj, col) in enumerate(cols), (pos, j) in enumerate(act)
                    c = X[pos, jj]
                    iszero(c) || (push!(I0, j); push!(J0, col); push!(V0, c))
                end
            else
                @inbounds for col in cols
                    rhs = alpha_parts[col]
                    isempty(rhs) && continue
                    x = Utils.solve_particular(resM.M.field, A_u, reshape(rhs, :, 1))
                    for (pos, j) in enumerate(act)
                        c = x[pos, 1]
                        iszero(c) || (push!(I0, j); push!(J0, col); push!(V0, c))
                    end
                end
            end
        end

        F[1] = sparse(I0, J0, V0, m0, n0)

        kstart = 1
        if upto >= 1
            if _FUNCTORIALITY_USE_PROJECTIVE_Q0_JOINT_FALLBACK[]
                try
                    F[2] = _projective_chain_step_coeff(resL, resM, F[1], active_M, 1)
                catch err
                    if _projective_q0_step_inconsistent(err)
                        F[1], F[2] = _solve_projective_q0_q1_joint_coeff(resL, resM, alpha_parts, active_M)
                    else
                        rethrow()
                    end
                end
            else
                F[2] = _projective_chain_step_coeff(resL, resM, F[1], active_M, 1)
            end
            kstart = 2
        end

        for k in kstart:upto
            F[k + 1] = _projective_chain_step_coeff(resL, resM, F[k], active_M, k)
        end

        return F
    end

    @inline function _pmorphism_is_identity(f::PMorphism{K}) where {K}
        f.dom === f.cod || return false
        @inbounds for u in eachindex(f.comps)
            d = f.dom.dims[u]
            A = f.comps[u]
            size(A, 1) == d && size(A, 2) == d || return false
            for j in 1:d
                for i in 1:d
                    v = A[i, j]
                    if i == j
                        v == one(K) || return false
                    else
                        iszero(v) || return false
                    end
                end
            end
        end
        return true
    end

    @inline function _pmorphism_scalar_multiple_identity(f::PMorphism{K}) where {K}
        f.dom === f.cod || return nothing
        scalar = nothing
        @inbounds for u in eachindex(f.comps)
            d = f.dom.dims[u]
            A = f.comps[u]
            size(A, 1) == d && size(A, 2) == d || return nothing
            for j in 1:d
                for i in 1:d
                    v = A[i, j]
                    if i == j
                        if scalar === nothing
                            scalar = v
                        elseif v != scalar
                            return nothing
                        end
                    else
                        iszero(v) || return nothing
                    end
                end
            end
        end
        return scalar === nothing ? one(K) : scalar
    end

    @inline function _identity_chainmap_coeffs(res::ProjectiveResolution{K}, maxlen::Int) where {K}
        H = Vector{SparseMatrixCSC{K, Int}}(undef, maxlen + 1)
        @inbounds for k in 0:maxlen
            n = length(res.gens[k + 1])
            if n == 0
                H[k + 1] = spzeros(K, 0, 0)
            else
                idx = collect(1:n)
                vals = fill(one(K), n)
                H[k + 1] = sparse(idx, idx, vals, n, n)
            end
        end
        return H
    end

    @inline function _scalar_chainmap_coeffs(res::ProjectiveResolution{K}, scalar::K, maxlen::Int) where {K}
        isone(scalar) && return _identity_chainmap_coeffs(res, maxlen)
        H = Vector{SparseMatrixCSC{K, Int}}(undef, maxlen + 1)
        @inbounds for k in 0:maxlen
            n = length(res.gens[k + 1])
            if n == 0
                H[k + 1] = spzeros(K, 0, 0)
            else
                idx = collect(1:n)
                vals = fill(scalar, n)
                H[k + 1] = sparse(idx, idx, vals, n, n)
            end
        end
        return H
    end

    """
        lift_chainmap(res_dom, res_cod, f; maxlen, check=true)

    Lift a P-module morphism `f : res_dom.M -> res_cod.M` to a chain map between the
    chosen projective resolutions `res_dom` and `res_cod`.

    This returns a vector `H` of length `maxlen+1` where `H[k+1]` is the coefficient
    matrix of the lifted map on degree `k`:

        H[k+1] : res_dom.Pmods[k+1] -> res_cod.Pmods[k+1]

    in the indicator-summand bases used by the projective resolutions.

    Notes
    -----
    - This is deliberately "low-level": it returns coefficient matrices (sparse).
    `ChangeOfPosets.pushforward_left_complex` converts these into `PMorphism`s
    using `_pmorphism_from_upset_coeff`.
    - If you want the lift only up to the resolution length, you should pass
    `maxlen <= min(length(res_dom.Pmods), length(res_cod.Pmods)) - 1`.
    """
    function lift_chainmap(res_dom::ProjectiveResolution{K},
                        res_cod::ProjectiveResolution{K},
                        f::PMorphism{K};
                        maxlen::Int,
                        check::Bool=true) where {K}

        if check
            @assert f.dom === res_dom.M "lift_chainmap: f.dom must equal res_dom.M"
            @assert f.cod === res_cod.M "lift_chainmap: f.cod must equal res_cod.M"
        end

        # Identity-on-self is a special case that must always admit the obvious
        # identity chain map on the chosen resolution. The generic q=0 lifting
        # routine can pick a degree-0 particular solution that fails to extend on
        # some direct-sum fixtures, so short-circuit this canonical case.
        if res_dom === res_cod && f.dom === f.cod && _pmorphism_is_identity(f)
            return _identity_chainmap_coeffs(res_dom, maxlen)
        end

        # Internal routine expects `upto` = highest homological degree to lift.
        return _lift_pmodule_map_to_projective_resolution_chainmap_coeff(
            res_dom, res_cod, f; upto=maxlen
        )
    end


    """
        ext_map_first(EMN::ExtSpaceProjective{K}, EMPN::ExtSpaceProjective{K}, f::PMorphism{K}; t::Int)

    Contravariant map in the first argument for the projective-resolution Ext model.

    Given a morphism `f : M -> Mp` and a fixed module `N`, this returns the induced map

        Ext^t(Mp, N) -> Ext^t(M, N)

    with respect to the bases stored in `EMPN` (source) and `EMN` (target).

    This method:
    1. Lifts `f` to a chain map between the stored projective resolutions
    `P(M) -> P(Mp)` (deterministically, using the same lifting routine as the Yoneda product).
    2. Applies `Hom(-,N)` to obtain a cochain map
    `Hom(P(Mp),N) -> Hom(P(M),N)`.
    3. Passes to cohomology in degree `t`.

    See also: `ext_map_second`.
    """
    function ext_map_first(
        EMN::ExtSpaceProjective{K},
        EMPN::ExtSpaceProjective{K},
        f::PMorphism{K};
        t::Int
    ) where {K}
        @assert EMN.N === EMPN.N
        @assert EMN.res.M === f.dom
        @assert EMPN.res.M === f.cod
        @assert t >= 0
        @assert t <= EMN.tmax
        @assert t <= EMPN.tmax

        # Lift f to a chain map between the chosen projective resolutions up to degree t.
        coeffs = _lift_pmodule_map_to_projective_resolution_chainmap_coeff(
            EMN.res, EMPN.res, f; upto=t
        )
        coeff_t = coeffs[t+1]  # P_t(M) -> P_t(Mp)

        # The induced cochain map at degree t is precomposition:
        # Hom(P_t(Mp), N) -> Hom(P_t(M), N).
        dom_gens = EMN.res.gens[t+1]    # generators of P_t(M)
        cod_gens = EMPN.res.gens[t+1]   # generators of P_t(Mp)

        F = _precompose_on_hom_cochains_from_projective_coeff(
            EMN.N,
            dom_gens,
            cod_gens,
            EMN.offsets[t+1],
            EMPN.offsets[t+1],
            coeff_t
        )

        return ChainComplexes.induced_map_on_cohomology(EMPN.cohom[t+1], EMN.cohom[t+1], F)
    end

    # NOTE:
    # The Ext functoriality helper ext_map_second for the injective model is defined
    # later in this file, after ExtSpaceInjective is defined.

    # ----------------------------
    # Connecting homomorphism for 0 -> A --i--> B --p--> C -> 0 in the second argument
    # Uses the projective model, so Hom(P_a,-) is exact.
    # ----------------------------

    function _connecting_hom_solve_reduce(field::AbstractCoeffField,
                                          left_map::AbstractMatrix{K},
                                          boundary_map::AbstractMatrix{K},
                                          right_map::AbstractMatrix{K},
                                          source_hrep::AbstractMatrix{K},
                                          target_cohom) where {K}
        out = Matrix{K}(undef, target_cohom.dimH, size(source_hrep, 2))
        @inline function _assign_coords!(dest::AbstractVector{K}, v)
            if v isa AbstractMatrix
                dest .= @view(v[:, 1])
            else
                dest .= v
            end
            return dest
        end
        if _use_support_solve_plan(size(left_map, 1), size(left_map, 2), size(source_hrep, 2)) ||
           _use_support_solve_plan(size(right_map, 1), size(right_map, 2), size(source_hrep, 2))
            left_plan = _particular_solve_plan(field, left_map)
            right_plan = _particular_solve_plan(field, right_map)
            Y = _solve_particular(left_plan, source_hrep)
            DY = boundary_map * Y
            X = _solve_particular(right_plan, DY)
            @inbounds for j in 1:size(X, 2)
                _assign_coords!(@view(out[:, j]), ChainComplexes.cohomology_coordinates(target_cohom, @view(X[:, j])))
            end
            return out
        end

        @inbounds for j in 1:size(source_hrep, 2)
            y = Utils.solve_particular(field, left_map, reshape(@view(source_hrep[:, j]), :, 1))[:, 1]
            dy = boundary_map * y
            x = Utils.solve_particular(field, right_map, reshape(dy, :, 1))[:, 1]
            _assign_coords!(@view(out[:, j]), ChainComplexes.cohomology_coordinates(target_cohom, x))
        end
        return out
    end

    function connecting_hom(EA::ExtSpaceProjective{K}, EB::ExtSpaceProjective{K}, EC::ExtSpaceProjective{K},
                            i::PMorphism{K}, p::PMorphism{K}; t::Int) where {K}
        
        # Sanity checks (avoid BoundsError when the resolution is too short).
        if EA.res !== EB.res || EA.res !== EC.res
            error("connecting_hom: EA, EB, EC must share the same ProjectiveResolution.")
        end
        if t < 0
            error("connecting_hom: t must be >= 0.")
        end
        # To compute delta^t we need resolution data through degree t+1.
        if t + 1 > EA.tmax || t + 1 > EB.tmax || t > EC.tmax
            error("connecting_hom: need EA and EB computed through degree t+1=$(t+1) and EC through degree t=$t. Recompute with larger maxdeg/maxlen.")
        end
        
        
        # delta : Ext^t(M,C) -> Ext^{t+1}(M,A)

        # cochain degree t maps:
        gens_t = EA.res.gens[t+1]
        It = _blockdiag_on_hom_cochains(i, gens_t, EA.offsets[t+1], EB.offsets[t+1])   # CA^t -> CB^t
        Pt = _blockdiag_on_hom_cochains(p, gens_t, EB.offsets[t+1], EC.offsets[t+1])   # CB^t -> CC^t

        gens_tp1 = EA.res.gens[t+2]
        Itp1 = _blockdiag_on_hom_cochains(i, gens_tp1, EA.offsets[t+2], EB.offsets[t+2])
        Ptp1 = _blockdiag_on_hom_cochains(p, gens_tp1, EB.offsets[t+2], EC.offsets[t+2])

        dBt = EB.complex.d[t+1]      # CB^t -> CB^{t+1}
        # Basis of Ext^t(M,C) as cocycles (columns)
        HrepC = EC.cohom[t+1].Hrep
        return _connecting_hom_solve_reduce(EA.M.field, Pt, dBt, Itp1, HrepC, EA.cohom[t+2])
    end

    struct ExtLongExactSequenceSecond{K}
        tmin::Int
        tmax::Int
        EA::ExtSpaceProjective{K}
        EB::ExtSpaceProjective{K}
        EC::ExtSpaceProjective{K}
        iH::Vector{Matrix{K}}      # Ext^t(M,A) -> Ext^t(M,B), index t+1
        pH::Vector{Matrix{K}}      # Ext^t(M,B) -> Ext^t(M,C), index t+1
        delta::Vector{Matrix{K}}  # Ext^t(M,C) -> Ext^{t+1}(M,A), index t+1
    end

    """
        ExtLongExactSequenceSecond(M, A, B, C, i, p, df::DerivedFunctorOptions)

    Package the long exact sequence in the second argument of Ext coming from
    a short exact sequence 0 -> A --i--> B --p--> C -> 0:

    ... -> Ext^t(M,A) -> Ext^t(M,B) -> Ext^t(M,C) -> Ext^{t+1}(M,A) -> ...

    The object stores maps for t = 0..df.maxdeg, including the connecting maps delta^t.
    Internally, a single projective resolution of M is built and shared across EA, EB, EC.

    This function uses the projective-resolution model of Ext, so df.model must be :projective or :auto.
    """
    function ExtLongExactSequenceSecond(M::PModule{K},
                                    A::PModule{K},
                                    B::PModule{K},
                                    C::PModule{K},
                                    i::PMorphism{K},
                                    p::PMorphism{K},
                                    df::DerivedFunctorOptions) where {K}
        if !(df.model === :auto || df.model === :projective)
            error("ExtLongExactSequenceSecond: df.model must be :projective or :auto, got $(df.model)")
        end
        maxdeg = df.maxdeg
        # Need Ext up to degree maxdeg+1 to define delta^maxdeg.
        res = projective_resolution(M, ResolutionOptions(maxlen=maxdeg + 1))
        _pad_projective_resolution!(res, maxdeg+1)

        EA = Ext(res, A)
        EB = Ext(res, B)
        EC = Ext(res, C)

        iH = Matrix{K}[]
        pH = Matrix{K}[]
        delta = Matrix{K}[]
        for t in 0:maxdeg
            push!(iH, ext_map_second(EA, EB, i; t=t))
            push!(pH, ext_map_second(EB, EC, p; t=t))
            push!(delta, connecting_hom(EA, EB, EC, i, p; t=t))
        end

        return ExtLongExactSequenceSecond{K}(0, maxdeg, EA, EB, EC, iH, pH, delta)
    end

    """
        ExtLongExactSequenceSecond(M, ses, df::DerivedFunctorOptions)

    Convenience wrapper: build the Ext long exact sequence in the second argument from
    a checked short exact sequence object.
    """
    function ExtLongExactSequenceSecond(M::PModule{K}, ses::ShortExactSequence{K}, df::DerivedFunctorOptions) where {K}
        return ExtLongExactSequenceSecond(M, ses.A, ses.B, ses.C, ses.i, ses.p, df)
    end

    # -----------------------------------------------------------------------------
    # Internal: lift a cocycle in Hom(P_q(L), M) to a degree-q chain map P(L) -> P(M)
    # (enough components to compose with a degree-p cocycle from Ext^p(M,N)).
    # -----------------------------------------------------------------------------

    # ---------------------------------------------------------------------
    # Convenience overload: allow a single cocycle to be passed as a vector.
    #
    # Internally, the lifting routine is implemented for matrices because it
    # can lift multiple cocycles simultaneously (each column is one RHS).
    # Many user-facing helpers (and `representative`) naturally produce a
    # single cocycle as a vector, so we interpret that as a 1-column matrix.
    # ---------------------------------------------------------------------
    function _lift_cocycle_to_chainmap_coeff(resL::ProjectiveResolution{K},
                                            resM::ProjectiveResolution{K},
                                            E_LM::ExtSpaceProjective{K},
                                            q::Int,
                                            alpha_cocycle::AbstractVector{K};
                                            upto::Int) where {K}
        # Treat the cochain vector as a single column.
        alpha_mat = reshape(alpha_cocycle, :, 1)
        return _lift_cocycle_to_chainmap_coeff(resL, resM, E_LM, q, alpha_mat; upto=upto)
    end

    """
        _lift_cocycle_to_chainmap_coeff(resL, resM, E_LM, q, alpha_cocycle; upto)

    Given:
    - `resL`: a projective resolution of L,
    - `resM`: a projective resolution of M,
    - a cocycle `alpha_cocycle` in C^q = Hom(P_q(L), M),

    construct (non-canonically, but deterministically) coefficient matrices describing
    a degree-q chain map
        F : P(L) -> P(M),
    i.e. maps
        F_k : P_{q+k}(L) -> P_k(M)
    for k = 0,1,...,upto

    Return value:
    - a vector `F` where `F[k+1]` is the coefficient matrix of F_k.

    This is the standard "comparison map" construction used to implement the Yoneda
    product via projective resolutions.
    """
    function _lift_cocycle_to_chainmap_coeff(resL::ProjectiveResolution{K},
                                            resM::ProjectiveResolution{K},
                                            E_LM::ExtSpaceProjective{K},
                                            q::Int,
                                            alpha_cocycle::AbstractMatrix{K};
                                            upto::Int = 0) where {K}

        # Return a chain map F lifting the cocycle alpha:
        #
        #   alpha : P_q(L) -> M
        #
        # The lift is a chain map between projective resolutions:
        #
        #   F_k : P_{q+k}(L) -> P_k(M)
        #
        # represented by coefficient matrices in the chosen direct-sum bases.
        #
        # Performance note:
        # Historically this routine built dense coefficient matrices via zeros(K, ...)
        # and then filled only the entries allowed by the poset constraints. In most
        # poset-resolution situations, these chain maps are sparse, so dense zeros are
        # a large avoidable cost. We now build SparseMatrixCSC directly from (row,col,val)
        # triples. Downstream code uses AbstractMatrix{K}, so sparsity is compatible.

        if upto < 0
            error("_lift_cocycle_to_chainmap_coeff: keyword `upto` must be >= 0.")
        end

        # Precompute which summands in each P_k(M) are eligible to receive maps from a base u.
        active_M = Vector{Vector{Vector{Int}}}(undef, upto + 1)
        for k in 0:upto
            cod_bases_k = resM.gens[k+1]  # indices in the direct sum decomposition of P_k(M)
            active_M[k+1] = _active_upset_indices_cached(resM.M.Q, cod_bases_k)
        end

        # Coefficients of F_k, stored as SparseMatrixCSC{K}.
        # Convention: F[k+1] is the coefficient matrix for F_k.
        F = Vector{SparseMatrixCSC{K, Int}}(undef, upto + 1)

        # Step k = 0: solve for F_0 column-by-column via augmented constraints from alpha.
        dom_gens_q  = resL.gens[q+1]      # summands in P_q(L)
        cod_bases_0 = resM.gens[1]        # summands in P_0(M)

        m0 = length(cod_bases_0)
        n0 = length(dom_gens_q)

        I0 = Int[]   # row indices
        J0 = Int[]   # col indices
        V0 = K[]    # values

        # alpha_cocycle is stored in the cochain basis of E_LM at degree q:
        # it is a concatenation of blocks in M(u), one block for each generator u in P_q(L).
        # Split it into one RHS vector per generator of P_q(L).
        if size(alpha_cocycle, 2) != 1
            throw(ArgumentError("_lift_cocycle_to_chainmap_coeff currently expects a single cocycle column (got $(size(alpha_cocycle,2)))."))
        end
        _, alpha_parts = split_cochain(E_LM, q, Vector{K}(alpha_cocycle[:, 1]))

        groups0 = _base_vertex_groups(resM.M.Q, dom_gens_q)
        @inbounds for u in 1:length(groups0)
            cols = groups0[u]
            isempty(cols) && continue
            act = active_M[1][u]
            isempty(act) && continue

            A_u = Matrix(resM.aug.comps[u])
            nrhs = length(cols)
            if _use_support_solve_plan(size(A_u, 1), size(A_u, 2), nrhs)
                rhslen = length(alpha_parts[cols[1]])
                B = zeros(K, rhslen, nrhs)
                @inbounds for (jj, col) in enumerate(cols)
                    rhs = alpha_parts[col]
                    isempty(rhs) || copyto!(@view(B[:, jj]), rhs)
                end
                plan = _cached_particular_solve_plan(
                    resM.M.field,
                    A_u,
                    UInt(objectid(resM.aug.comps[u])),
                    UInt(length(act)),
                )
                X = _solve_particular(plan, B)
                @inbounds for (jj, col) in enumerate(cols), (pos, j) in enumerate(act)
                    c = X[pos, jj]
                    if !iszero(c)
                        push!(I0, j)
                        push!(J0, col)
                        push!(V0, c)
                    end
                end
            else
                @inbounds for col in cols
                    rhs = alpha_parts[col]
                    isempty(rhs) && continue
                    x = Utils.solve_particular(resM.M.field, A_u, reshape(rhs, :, 1))
                    for (pos, j) in enumerate(act)
                        c = x[pos, 1]
                        if !iszero(c)
                            push!(I0, j)
                            push!(J0, col)
                            push!(V0, c)
                        end
                    end
                end
            end
        end

        F[1] = sparse(I0, J0, V0, m0, n0)

        # Steps k >= 1: solve the chain-map equations
        #   d_k^M * F_k = F_{k-1} * d_{q+k}^L
        for k in 1:upto
            DkM  = Matrix(resM.d_mat[k])          # P_k(M) -> P_{k-1}(M)
            DqkL = Matrix(resL.d_mat[q + k])      # P_{q+k}(L) -> P_{q+k-1}(L)

            RHS = F[k] * DqkL                     # matrix in Hom(P_{q+k}(L), P_{k-1}(M))

            cod_bases_k  = resM.gens[k+1]         # summands in P_k(M) (these are the columns of DkM)
            dom_bases_qk = resL.gens[q+k+1]       # summands in P_{q+k}(L)

            mk = length(cod_bases_k)
            nk = length(dom_bases_qk)

            Ik = Int[]
            Jk = Int[]
            Vk = K[]

            groupsk = _base_vertex_groups(resM.M.Q, dom_bases_qk)
            @inbounds for u in 1:length(groupsk)
                cols = groupsk[u]
                isempty(cols) && continue
                allowed = active_M[k+1][u]

                if isempty(allowed)
                    @inbounds for col in cols
                        any(!iszero, @view RHS[:, col]) &&
                            error("_lift_cocycle_to_chainmap_coeff: inconsistent constraints at (k=$k, dom_summand=$col).")
                    end
                    continue
                end

                A = @view DkM[:, allowed]
                nrhs = length(cols)
                if _use_support_solve_plan(size(A, 1), size(A, 2), nrhs)
                    plan = _cached_particular_solve_plan(
                        resM.M.field,
                        A,
                        UInt(objectid(resM.d_mat[k])),
                        _support_hash(allowed),
                    )
                    X = _solve_particular(plan, Matrix(@view RHS[:, cols]))
                    @inbounds for (jj, col) in enumerate(cols), (pos, j) in enumerate(allowed)
                        c = X[pos, jj]
                        if !iszero(c)
                            push!(Ik, j)
                            push!(Jk, col)
                            push!(Vk, c)
                        end
                    end
                else
                    @inbounds for col in cols
                        x = Utils.solve_particular(resM.M.field, A, reshape(@view(RHS[:, col]), :, 1))
                        for (pos, j) in enumerate(allowed)
                            c = x[pos, 1]
                            if !iszero(c)
                                push!(Ik, j)
                                push!(Jk, col)
                                push!(Vk, c)
                            end
                        end
                    end
                end
            end

            F[k+1] = sparse(Ik, Jk, Vk, mk, nk)
        end

        return F
    end

    # Contravariant map in first argument: f : M -> Mp induces Ext^t(Mp,N) -> Ext^t(M,N)
    function ext_map_first(EMN::ExtSpaceInjective{K}, EMPN::ExtSpaceInjective{K}, f::PMorphism{K}; t::Int) where {K}
        # map on cochains at degree t: Hom(Mp, E^t) -> Hom(M, E^t), g |-> g circ f
        Hsrc = EMPN.homs[t+1]
        Htgt = EMN.homs[t+1]
        F = _precompose_matrix(Htgt, Hsrc, f)
        return ChainComplexes.induced_map_on_cohomology(EMPN.cohom[t+1], EMN.cohom[t+1], F)
    end

    """
        ext_map_second(EMN, EMNp, g; t, check=true)

    Induced map Ext^t(M,N) -> Ext^t(M,Np) computed in the injective model.

    Implementation:
    1. Lift `g : N -> Np` to a cochain map between the chosen injective resolutions using
    `lift_injective_chainmap` (only up to degree t).
    2. Postcompose on Hom(M,-) in degree t and take the induced map on cohomology.

    This avoids the older basis-transport through the projective model and is usually faster.
    """
    function ext_map_second(EMN::ExtSpaceInjective{K},
                            EMNp::ExtSpaceInjective{K},
                            g::PMorphism{K};
                            t::Int,
                            check::Bool = true) where {K}
        @assert g.dom === EMN.N
        @assert g.cod === EMNp.N
        @assert 0 <= t <= EMN.tmax
        @assert 0 <= t <= EMNp.tmax

        phis = lift_injective_chainmap(g, EMN.res, EMNp.res; upto=t, check=check)

        # postcompose by phi^t : E^t -> E'^t
        F = _postcompose_matrix(EMNp.homs[t+1], EMN.homs[t+1], phis[t+1])

        return ChainComplexes.induced_map_on_cohomology(EMN.cohom[t+1], EMNp.cohom[t+1], F)
    end


    # -----------------------------------------------------------------------------
    # Internal helper: precomposition matrices on Hom spaces
    # -----------------------------------------------------------------------------

    """
        _precompose_matrix(Hdom, Hcod, f) -> Matrix{K}

    Given a morphism f: A -> B and Hom spaces

    - Hcod = Hom(B, E)
    - Hdom = Hom(A, E)

    return the matrix of the linear map
        f^* : Hom(B, E) -> Hom(A, E),   g |-> g circ f
    in the bases stored inside the HomSpace objects.

    This is the cochain-level map used by Ext functoriality in the first argument
    when Ext is computed via an injective resolution.
    """
    function _precompose_matrix(Hdom::HomSpace{K},
                                Hcod::HomSpace{K},
                                f::PMorphism{K};
                                workspace::Union{Nothing,_PrecomposeWorkspace{K}}=nothing) where {K}
        ncols = dim(Hcod)
        nrows = dim(Hdom)
        if ncols == 0 || nrows == 0
            return zeros(K, nrows, ncols)
        end

        plan_entry = if _FUNCTORIALITY_USE_HOM_BASIS_SOLVE_PLAN_CACHE[]
            _hom_solve_plan_entry(Hdom)
        else
            nothing
        end
        if workspace === nothing
            if plan_entry !== nothing
                workspace = plan_entry.pre
            elseif _FUNCTORIALITY_USE_HOM_SOLVE_WORKSPACE_CACHE[]
                workspace = _hom_solve_workspace_entry(Hdom).pre
            end
        end

        rhs = _rhs_buffer!(workspace, K, size(Hdom.basis_matrix, 1), ncols)
        @inbounds for i in 1:nvertices(Hdom.dom.Q)
            ai = Hdom.dom.dims[i]
            bi = Hcod.dom.dims[i]
            ei = Hdom.cod.dims[i]
            if ai == 0 || bi == 0 || ei == 0
                continue
            end

            src_rng = (Hcod.offsets[i] + 1):Hcod.offsets[i + 1]
            dst_rng = (Hdom.offsets[i] + 1):Hdom.offsets[i + 1]
            src_view = reshape(@view(Hcod.basis_matrix[src_rng, :]), ei, bi * ncols)
            dst_view = reshape(@view(rhs[dst_rng, :]), ei, ai * ncols)
            fi = f.comps[i]

            for a in 1:ai
                for b in 1:bi
                    coeff = fi[b, a]
                    iszero(coeff) && continue
                    src_col = b
                    dst_col = a
                    for j in 1:ncols
                        @simd for r in 1:ei
                            dst_view[r, dst_col] += coeff * src_view[r, src_col]
                        end
                        src_col += bi
                        dst_col += ai
                    end
                end
            end
        end
        if plan_entry !== nothing
            return _solve_particular(plan_entry.plan, rhs)
        end
        return FieldLinAlg.solve_fullcolumn(Hdom.dom.field, Hdom.basis_matrix, rhs)
    end

    """
        _postcompose_matrix(Hdom, Hcod, g)

    Return the matrix representing postcomposition with `g` on Hom spaces.

    If
    * `Hcod == Hom(M, E)` and
    * `Hdom == Hom(M, Eprime)` and
    * `g : E -> Eprime`,
    then this returns the matrix of the linear map

        g_* : Hom(M, E) -> Hom(M, Eprime),   phi |-> g circ phi.

    The basis used is the one stored in `Hcod` and `Hdom`.
    """
    function _postcompose_matrix(Hdom::HomSpace{K},
                                 Hcod::HomSpace{K},
                                 g::PMorphism{K};
                                 workspace::Union{Nothing,_PostcomposeWorkspace{K}}=nothing) where {K}
        # Sanity checks: all Hom spaces must share the same domain module.
        @assert Hdom.dom === Hcod.dom
        @assert Hcod.cod === g.dom
        @assert Hdom.cod === g.cod

        ncols = dim(Hcod)
        nrows = dim(Hdom)
        if ncols == 0 || nrows == 0
            return zeros(K, nrows, ncols)
        end

        plan_entry = if _FUNCTORIALITY_USE_HOM_BASIS_SOLVE_PLAN_CACHE[]
            _hom_solve_plan_entry(Hdom)
        else
            nothing
        end
        if workspace === nothing
            if plan_entry !== nothing
                workspace = plan_entry.post
            elseif _FUNCTORIALITY_USE_HOM_SOLVE_WORKSPACE_CACHE[]
                workspace = _hom_solve_workspace_entry(Hdom).post
            end
        end

        rhs = _rhs_buffer!(workspace, K, size(Hdom.basis_matrix, 1), ncols)
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

        if plan_entry !== nothing
            return _solve_particular(plan_entry.plan, rhs)
        end
        return FieldLinAlg.solve_fullcolumn(Hdom.dom.field, Hdom.basis_matrix, rhs)
    end



    """
        connecting_hom_first(EA, EB, EC, i, p; t) -> Matrix{K}

    Connecting homomorphism for a short exact sequence in the first (contravariant) argument:

        0 -> A --i--> B --p--> C -> 0.

    Fix an injective resolution of N and compute Ext via the cochain complexes Hom(-, E^*).
    The associated long exact sequence contains the connecting map

        delta^t : Ext^t(A, N) -> Ext^{t+1}(C, N).

    This function returns the matrix of delta^t in the chosen Ext bases.

    Requirements:
    - EA, EB, EC must be `ExtSpaceInjective` objects built from the *same*
    `InjectiveResolution` of N.
    - i: A -> B and p: B -> C should define a short exact sequence in the first argument.
    """
    function connecting_hom_first(EA::ExtSpaceInjective{K},
                                EB::ExtSpaceInjective{K},
                                EC::ExtSpaceInjective{K},
                                i::PMorphism{K},
                                p::PMorphism{K}; t::Int) where {K}

        if EA.res !== EB.res || EA.res !== EC.res
            error("connecting_hom_first: EA, EB, EC must share the same InjectiveResolution.")
        end
        if t < 0 || t >= EA.tmax
            error("connecting_hom_first: t must satisfy 0 <= t <= tmax-1.")
        end

        # Cochain-level maps at the relevant degrees:
        #   i^* : Hom(B, E^t) -> Hom(A, E^t)
        #   p^* : Hom(C, E^{t+1}) -> Hom(B, E^{t+1})
        max_rhs = max(dim(EB.homs[t+1]), dim(EC.homs[t+2]))
        max_rows = max(size(EA.homs[t+1].basis_matrix, 1), size(EB.homs[t+2].basis_matrix, 1))
        ws = _use_hom_workspace(max_rows, max_rhs) ? _PrecomposeWorkspace(K) : nothing
        It   = _precompose_matrix(EA.homs[t+1], EB.homs[t+1], i; workspace=ws)
        Ptp1 = _precompose_matrix(EB.homs[t+2], EC.homs[t+2], p; workspace=ws)

        return _connecting_hom_solve_reduce(EA.M.field, It, EB.complex.d[t+1], Ptp1, EA.cohom[t+1].Hrep, EC.cohom[t+2])
    end

    # =============================================================================
    # Packaged long exact sequence in the FIRST argument (symmetric to Second-argument LES)
    # =============================================================================

    """
        ExtLongExactSequenceFirst(A, B, C, N, i, p, df::DerivedFunctorOptions)
        ExtLongExactSequenceFirst(ses, N, df::DerivedFunctorOptions)

    Package the long exact sequence in the first argument of Ext induced by a short exact sequence

    0 -> A --i--> B --p--> C -> 0.

    This long exact sequence is computed using the injective model of Ext (a shared injective
    resolution of N). The bases of all terms are those coming from that shared resolution.

    The stored maps cover degrees t = 0..df.maxdeg, including the connecting maps.
    """
    struct ExtLongExactSequenceFirst{K}
        tmin::Int
        tmax::Int
        EA::ExtSpaceInjective{K}  # Ext^*(A, N)
        EB::ExtSpaceInjective{K}  # Ext^*(B, N)
        EC::ExtSpaceInjective{K}  # Ext^*(C, N)

        # Maps on Ext in degree t are stored at index t+1.
        # pH[t+1] : Ext^t(C,N) -> Ext^t(B,N)
        # iH[t+1] : Ext^t(B,N) -> Ext^t(A,N)
        # delta[t+1] : Ext^t(A,N) -> Ext^(t+1)(C,N)
        pH::Vector{Matrix{K}}
        iH::Vector{Matrix{K}}
        delta::Vector{Matrix{K}}
    end

    """
        ExtLongExactSequenceFirst(A, B, C, N, i, p, df::DerivedFunctorOptions)

    Package the long exact sequence in the first argument of Ext coming from
    a short exact sequence 0 -> A --i--> B --p--> C -> 0:

    ... -> Ext^t(C,N) -> Ext^t(B,N) -> Ext^t(A,N) -> Ext^{t+1}(C,N) -> ...

    The object stores maps for t = 0..df.maxdeg, including the connecting maps delta^t.
    Internally, a single injective resolution of N is built and shared across EA, EB, EC.

    This function uses the injective-resolution model of Ext, so df.model must be :injective or :auto.
    """
    function ExtLongExactSequenceFirst(A::PModule{K},
                                    B::PModule{K},
                                    C::PModule{K},
                                    N::PModule{K},
                                    i::PMorphism{K},
                                    p::PMorphism{K},
                                    df::DerivedFunctorOptions) where {K}
        @assert i.dom === A && i.cod === B
        @assert p.dom === B && p.cod === C

        maxdeg = df.maxdeg

        # Need Ext^{t+1} for t leq maxdeg, so resolve N one step further.
        resN = injective_resolution(N, ResolutionOptions(maxlen=maxdeg + 1))
        EA = ExtInjective(A, resN)
        EB = ExtInjective(B, resN)
        EC = ExtInjective(C, resN)
        @assert EA.res === EB.res && EB.res === EC.res

        pH    = Vector{Matrix{K}}(undef, maxdeg + 1)
        iH    = Vector{Matrix{K}}(undef, maxdeg + 1)
        delta = Vector{Matrix{K}}(undef, maxdeg + 1)

        for t in 0:maxdeg
            pH[t+1]    = ext_map_first(EB, EC, p; t=t)
            iH[t+1]    = ext_map_first(EA, EB, i; t=t)
            delta[t+1] = connecting_hom_first(EA, EB, EC, i, p; t=t)
        end

        return ExtLongExactSequenceFirst{K}(0, maxdeg, EA, EB, EC, pH, iH, delta)
    end

    function ExtLongExactSequenceFirst(A::PModule{K},
                                    B::PModule{K},
                                    C::PModule{K},
                                    N::PModule{K},
                                    i::PMorphism{K},
                                    p::PMorphism{K};
                                    maxdeg::Int=4,
                                    model::Symbol=:auto,
                                    canon::Symbol=:projective) where {K}
        df = DerivedFunctorOptions(maxdeg=maxdeg, model=model, canon=canon)
        return ExtLongExactSequenceFirst(A, B, C, N, i, p, df)
    end

        # -----------------------------------------------------------------------------
    # Convenience wrappers accepting a ShortExactSequence
    # -----------------------------------------------------------------------------

    """
        ExtLongExactSequenceFirst(ses, N, df::DerivedFunctorOptions)

    Convenience wrapper: build the Ext long exact sequence in the first argument from
    a checked short exact sequence object.
    """
    function ExtLongExactSequenceFirst(ses::ShortExactSequence{K}, N::PModule{K}, df::DerivedFunctorOptions) where {K}
        return ExtLongExactSequenceFirst(ses.A, ses.B, ses.C, N, ses.i, ses.p, df)
    end

    """
        connecting_hom_first(EA, EB, EC, ses; t)

    Convenience wrapper for the Ext connecting morphism in the first argument when the
    short exact sequence is passed as a `ShortExactSequence` object.
    """
    function connecting_hom_first(EA::ExtSpaceInjective{K},
                                EB::ExtSpaceInjective{K},
                                EC::ExtSpaceInjective{K},
                                ses::ShortExactSequence{K}; t::Int) where {K}
        return connecting_hom_first(EA, EB, EC, ses.i, ses.p; t=t)
    end

    """
        ext_map_first(EMN, EMPN, f; t, backend=:projective)

    Induced map on Ext in degree t, contravariant in the first argument:
    given f: Mp -> M, returns f^*: Ext^t(M,N) -> Ext^t(Mp,N).

    The matrix is always returned in the CANONICAL bases of EMN and EMPN.

    `backend` chooses which realization is used for computation. `:projective` is always
    available and is the default.
    """
    function ext_map_first(
        EMN::ExtSpace{K},
        EMPN::ExtSpace{K},
        f::PMorphism{K};
        t::Int,
        backend::Symbol=:projective
    ) where {K}
        @assert EMN.N === EMPN.N
        @assert EMN.canon === EMPN.canon

        if backend === :projective || backend === :auto
            Aproj = ext_map_first(EMN.Eproj, EMPN.Eproj, f; t=t)
            if EMN.canon === :projective
                return Aproj
            else
                return _comparison_P2I(EMN, t) * Aproj * _comparison_I2P(EMPN, t)
            end
        elseif backend === :injective
            # Only safe if they literally share the same injective resolution object.
            if EMN.Einj.res !== EMPN.Einj.res
                error("ext_map_first(::ExtSpace, backend=:injective) requires a shared injective resolution.")
            end
            Ainj = ext_map_first(EMN.Einj, EMPN.Einj, f; t=t)
            if EMN.canon === :injective
                return Ainj
            else
                return _comparison_I2P(EMN, t) * Ainj * _comparison_P2I(EMPN, t)
            end
        else
            error("ext_map_first(::ExtSpace): unknown backend=$(backend)")
        end
    end

    """
        ext_map_second(EMN, EMNp, g; t, backend=:projective)

    Induced map on Ext in degree t, covariant in the second argument:
    given g: N -> Np, returns g_*: Ext^t(M,N) -> Ext^t(M,Np).

    The matrix is always returned in the CANONICAL bases of EMN and EMNp.

    `backend=:projective` is fastest and always available.
    `backend=:injective` uses the symmetric injective functoriality layer.
    """
    function ext_map_second(
        EMN::ExtSpace{K},
        EMNp::ExtSpace{K},
        g::PMorphism{K};
        t::Int,
        backend::Symbol=:projective
    ) where {K}
        @assert EMN.M === EMNp.M
        @assert EMN.canon === EMNp.canon

        if backend === :projective || backend === :auto
            Aproj = ext_map_second(EMN.Eproj, EMNp.Eproj, g; t=t)
            if EMN.canon === :projective
                return Aproj
            else
                return _comparison_P2I(EMNp, t) * Aproj * _comparison_I2P(EMN, t)
            end
        elseif backend === :injective
            Ainj = ext_map_second(EMN.Einj, EMNp.Einj, g; t=t)
            if EMN.canon === :injective
                return Ainj
            else
                return _comparison_I2P(EMNp, t) * Ainj * _comparison_P2I(EMN, t)
            end
        else
            error("ext_map_second(::ExtSpace): unknown backend=$(backend)")
        end
    end

    # ----------------------------------------------------------------------
    # Functoriality helpers (shared by TorSpace and TorSpaceSecond)
    # ----------------------------------------------------------------------

    # This helper takes a coefficient matrix describing a map between direct sums of upsets
    # (principal projectives) and tensors it with an arbitrary PModule `M` via structure maps.
    #
    # The coefficient matrix `coeff` has:
    #   - columns indexed by dom_bases
    #   - rows indexed by cod_bases
    #
    # For each nonzero entry (row j, col i) with coefficient c, we place the block:
    #   c * map_leq(M, dom_bases[i], cod_bases[j])
    #
    # This works for:
    # - TorSpace (coeff lives in P^op, M is a P-module),
    # - TorSpaceSecond (coeff lives in P,   M is a P^op-module),
    # because in both cases the nonzero entries correspond to comparable pairs in the relevant poset.
    function _tensor_map_on_tor_chains_from_projective_coeff(
        M::PModule{K},
        dom_bases::Vector{Int},
        cod_bases::Vector{Int},
        dom_offsets::Vector{Int},
        cod_offsets::Vector{Int},
        coeff::AbstractMatrix{K};
        cache::Union{Nothing,AbstractHomSystemCache}=nothing,
    ) where {K}
        return _tensor_coeff_result_cached(M, dom_bases, cod_bases, dom_offsets, cod_offsets, coeff, cache) do
            out_dim = cod_offsets[end]
            in_dim  = dom_offsets[end]

            I = Int[]
            J = Int[]
            V = K[]

            if _FUNCTORIALITY_USE_DIRECT_COEFF_TRIPLETS[] && _coeff_nnz(coeff) >= _FUNCTORIALITY_DIRECT_COEFF_MIN_NNZ[]
                plan = _tensor_coeff_plan_cached(M, dom_bases, cod_bases, dom_offsets, cod_offsets, coeff, cache)
                scales = _coeff_plan_scales(coeff, plan)
                _append_map_leq_many_scaled_triplets!(I, J, V, M, plan.batch,
                                                      plan.row_block_ids, plan.col_block_ids,
                                                      cod_offsets, dom_offsets, scales)
                return sparse(I, J, V, out_dim, in_dim)
            end

            # Fallback path for A/B parity: same mathematics, but it still materializes
            # the map blocks eagerly.
            if issparse(coeff)
                Icoeff, Jcoeff, Vcoeff = findnz(coeff)  # rows, cols, vals
                pairs = Vector{Tuple{Int,Int}}(undef, length(Vcoeff))
                @inbounds for k in eachindex(Vcoeff)
                    pairs[k] = (dom_bases[Jcoeff[k]], cod_bases[Icoeff[k]])
                end
                pair_batch = _prepare_map_leq_batch_owned(pairs)
                map_blocks = _map_blocks_buffer(M, length(pairs))
                map_leq_many!(map_blocks, M, pair_batch)
                @inbounds for k in eachindex(Vcoeff)
                    j = Icoeff[k]   # cod index
                    i = Jcoeff[k]   # dom index
                    c = Vcoeff[k]
                    iszero(c) && continue

                    Muv = map_blocks[k]
                    _append_scaled_triplets!(I, J, V, Muv, cod_offsets[j], dom_offsets[i]; scale=c)
                end
            else
                # Dense coeff: scan directly (no sparse(coeff) + findnz).
                dom_idx = Int[]
                cod_idx = Int[]
                vals = K[]
                pairs = Tuple{Int,Int}[]
                @inbounds for i in 1:length(dom_bases)
                    for j in 1:length(cod_bases)
                        c = coeff[j, i]
                        iszero(c) && continue

                        push!(dom_idx, i)
                        push!(cod_idx, j)
                        push!(vals, c)
                        push!(pairs, (dom_bases[i], cod_bases[j]))
                    end
                end
                pair_batch = _prepare_map_leq_batch_owned(pairs)
                map_blocks = _map_blocks_buffer(M, length(pairs))
                map_leq_many!(map_blocks, M, pair_batch)
                @inbounds for k in eachindex(vals)
                    i = dom_idx[k]
                    j = cod_idx[k]
                    _append_scaled_triplets!(I, J, V, map_blocks[k], cod_offsets[j], dom_offsets[i]; scale=vals[k])
                end
            end

            return sparse(I, J, V, out_dim, in_dim)
        end
    end

    # Block-diagonal chain map induced by a module morphism on each summand of a direct sum.
    # Used in tor_map_first for TorSpaceSecond and in connecting morphisms for Tor LES.
    function _tor_blockdiag_map_on_chains(
        f::PMorphism{K},
        gens::Vector{Int},
        dom_offsets::Vector{Int},
        cod_offsets::Vector{Int}
    ) where {K}
        out_dim = cod_offsets[end]
        in_dim  = dom_offsets[end]

        I = Int[]
        J = Int[]
        V = K[]

        @inbounds for i in 1:length(gens)
            u = gens[i]
            _append_scaled_triplets!(I, J, V, f.comps[u], cod_offsets[i], dom_offsets[i])
        end

        return sparse(I, J, V, out_dim, in_dim)
    end


    # ----------------------------------------------------------------------
    # Functoriality: resolve-second model
    # ----------------------------------------------------------------------

    """
        tor_map_first(T1, T2, f; s)

    For `TorSpaceSecond` objects, Tor is *strictly functorial* in the first argument by a block-diagonal map.

    Here `f : Rop -> Rop'` is a P^op-module map (right module map).
    The output is the induced linear map:
        Tor_s(Rop, L) -> Tor_s(Rop', L)
    in the chosen homology bases.
    """
    function tor_map_first(T1::TorSpaceSecond{K}, T2::TorSpaceSecond{K}, f::PMorphism{K};
        s::Union{Nothing,Int}=nothing,
        n::Union{Nothing,Int}=nothing
    ) where {K}
        s === nothing && (s = n)
        s === nothing && error("tor_map_first: provide s or n")
        @assert poset_equal(T1.resL.M.Q, T2.resL.M.Q)
        @assert T1.resL.gens == T2.resL.gens
        @assert poset_equal(f.dom.Q, T1.Rop.Q)
        @assert poset_equal(f.cod.Q, T2.Rop.Q)

        gens_s = T1.resL.gens[s + 1]
        F = _tor_blockdiag_map_on_chains(f, gens_s, T1.offsets[s + 1], T2.offsets[s + 1])
        return ChainComplexes.induced_map_on_homology(T1.homol[s + 1], T2.homol[s + 1], F)
    end

    tor_map_first(f::PMorphism{K}, T1::TorSpaceSecond{K}, T2::TorSpaceSecond{K};
        s::Union{Nothing,Int}=nothing,
        n::Union{Nothing,Int}=nothing
    ) where {K} = tor_map_first(T1, T2, f; s=s, n=n)

    """
        tor_map_second(T1, T2, g; s)

    For `TorSpaceSecond` objects, functoriality in the second argument requires lifting the module map
    to a chain map between projective resolutions, and then tensoring with the fixed right module.

    Here `g : L -> L'` is a P-module map.
    The output is the induced linear map:
        Tor_s(Rop, L) -> Tor_s(Rop, L')
    in the chosen homology bases.
    """
    function tor_map_second(T1::TorSpaceSecond{K}, T2::TorSpaceSecond{K}, g::PMorphism{K};
        s::Union{Nothing,Int}=nothing,
        n::Union{Nothing,Int}=nothing,
        cache::Union{Nothing,AbstractHomSystemCache}=nothing,
    ) where {K}
        s === nothing && (s = n)
        s === nothing && error("tor_map_second: provide s or n")
        @assert poset_equal(T1.Rop.Q, T2.Rop.Q)
        @assert poset_equal(g.dom.Q, T1.resL.M.Q)
        @assert poset_equal(g.cod.Q, T2.resL.M.Q)

        # Lift g to a chain map between the chosen projective resolutions.
        coeffs = _lift_pmodule_map_to_projective_resolution_chainmap_coeff(T1.resL, T2.resL, g; upto=s)
        coeff = coeffs[s + 1]  # rows=cod gens, cols=dom gens

        dom_bases = T1.resL.gens[s + 1]
        cod_bases = T2.resL.gens[s + 1]
        F = _tensor_map_on_tor_chains_from_projective_coeff(
            T1.Rop, dom_bases, cod_bases, T1.offsets[s + 1], T2.offsets[s + 1], coeff; cache=cache
        )
        return ChainComplexes.induced_map_on_homology(T1.homol[s + 1], T2.homol[s + 1], F)
    end

    tor_map_second(g::PMorphism{K}, T1::TorSpaceSecond{K}, T2::TorSpaceSecond{K};
        s::Union{Nothing,Int}=nothing,
        n::Union{Nothing,Int}=nothing,
        cache::Union{Nothing,AbstractHomSystemCache}=nothing
    ) where {K} = tor_map_second(T1, T2, g; s=s, n=n, cache=cache)

    # ============================================================
    # Functoriality for TorSpace (resolve FIRST variable)
    # ============================================================

    """
        tor_map_second(T1::TorSpace{K}, T2::TorSpace{K}, g::PMorphism{K}; s::Int) -> Matrix{K}

    Induced map on Tor_s in the second variable, when Tor was computed by resolving
    the first variable (i.e. using `model=:first`).

    Mathematically: if g : L -> L' is a morphism of P-modules, then there is an induced map
    Tor_s(Rop, L) -> Tor_s(Rop, L').

    Implementation notes:
    - Requires that T1 and T2 were computed using the SAME projective resolution of Rop,
    so that the chain-level direct sum decomposition matches degreewise.
    - The chain-level map is block-diagonal over the resolution summands: each summand
    is a copy of L_u, and we apply g_u on that block.
    """
    function tor_map_second(T1::TorSpace{K}, T2::TorSpace{K}, g::PMorphism{K};
        s::Union{Nothing,Int}=nothing,
        n::Union{Nothing,Int}=nothing,
        cache::Union{Nothing,AbstractHomSystemCache}=nothing,
    ) where {K}
        s === nothing && (s = n)
        s === nothing && error("tor_map_second: provide s or n")
        @assert T1.resRop.gens == T2.resRop.gens
        gens_s = T1.resRop.gens[s + 1]
        F = _tor_blockdiag_map_on_chains(g, gens_s, T1.offsets[s + 1], T2.offsets[s + 1])
        return ChainComplexes.induced_map_on_homology(T1.homol[s+1], T2.homol[s+1], F)
    end

    tor_map_second(g::PMorphism{K}, T1::TorSpace{K}, T2::TorSpace{K};
        s::Union{Nothing,Int}=nothing,
        n::Union{Nothing,Int}=nothing,
        cache::Union{Nothing,AbstractHomSystemCache}=nothing
    ) where {K} = tor_map_second(T1, T2, g; s=s, n=n, cache=cache)

    """
        tor_map_first(T1::TorSpace{K}, T2::TorSpace{K}, f::PMorphism{K}; s::Int) -> Matrix{K}

    Induced map on Tor_s in the first variable, when Tor was computed by resolving
    the first variable (i.e. using `model=:first`).

    Mathematically: if f : Rop -> Rop' is a morphism of P^op-modules, then there is an induced map
    Tor_s(Rop, L) -> Tor_s(Rop', L).

    Implementation strategy:
    - Lift f to a chain map between projective resolutions (coefficient matrices)
    - Tensor that chain map with L (using the same helper as the Tor boundary construction)
    - Pass to induced map on homology.
    """
    function tor_map_first(T1::TorSpace{K}, T2::TorSpace{K}, f::PMorphism{K};
        s::Union{Nothing,Int}=nothing,
        n::Union{Nothing,Int}=nothing,
        cache::Union{Nothing,AbstractHomSystemCache}=nothing,
    ) where {K}
        s === nothing && (s = n)
        s === nothing && error("tor_map_first: provide s or n")
        # Lift f to a chain map between the two resolutions (must be compatible).
        coeffs = _lift_pmodule_map_to_projective_resolution_chainmap_coeff(T1.resRop, T2.resRop, f; upto=s)
        coeff = coeffs[s+1]

        dom_bases = T1.resRop.gens[s + 1]
        cod_bases = T2.resRop.gens[s + 1]

        F = _tensor_map_on_tor_chains_from_projective_coeff(
            T1.L, dom_bases, cod_bases, T1.offsets[s + 1], T2.offsets[s + 1], coeff; cache=cache
        )

        return ChainComplexes.induced_map_on_homology(T1.homol[s+1], T2.homol[s+1], F)
    end

    tor_map_first(f::PMorphism{K}, T1::TorSpace{K}, T2::TorSpace{K};
        s::Union{Nothing,Int}=nothing,
        n::Union{Nothing,Int}=nothing,
        cache::Union{Nothing,AbstractHomSystemCache}=nothing
    ) where {K} = tor_map_first(T1, T2, f; s=s, n=n, cache=cache)

    # -----------------------------------------------------------------------------
    # Connecting morphisms for Tor long exact sequences
    # -----------------------------------------------------------------------------

    """
        _connecting_tor(TA, TB, TC, i, p; s)

    Internal helper for Tor long exact sequences.

    Given a short exact sequence of modules
        0 -> A --i--> B --p--> C -> 0
    and Tor spaces
        TA = Tor(..., A),  TB = Tor(..., B),  TC = Tor(..., C)
    computed with a *shared* resolution (same resolved variable and same generators),
    compute the connecting morphism

        delta_s : Tor_s(..., C) -> Tor_{s-1}(..., A)

    in the chosen homology bases.

    Implementation detail:
    We use the standard chain-level construction, choosing deterministic lifts via
    `Utils.solve_particular`:
    - lift a homology representative z in C_s to y in B_s,
    - apply the boundary d_B to get d_B(y) in B_{s-1},
    - lift d_B(y) through i_{s-1} to x in A_{s-1},
    - project x to homology coordinates in H_{s-1}(A).

    Requirements:
    - s >= 1
    - TA, TB, TC share the same resolved variable:
    - for `TorSpace`:      TA.resRop.gens == TB.resRop.gens == TC.resRop.gens
    - for `TorSpaceSecond`: TA.resL.gens  == TB.resL.gens  == TC.resL.gens
    """
    function _connecting_tor(TA::TorSpace{K}, TB::TorSpace{K}, TC::TorSpace{K},
                            i::PMorphism{K}, p::PMorphism{K}; s::Int) where {K}
        if s < 1
            error("_connecting_tor: s must be >= 1 (got $s)")
        end

        @assert TA.resRop.gens == TB.resRop.gens == TC.resRop.gens

        # Degree s and s-1 generator lists for the shared resolution of Rop.
        gens_s   = TA.resRop.gens[s+1]  # chains in degree s
        gens_sm1 = TA.resRop.gens[s]    # chains in degree s-1

        # Chain-level maps induced by i and p.
        # p_s   : B_s     -> C_s
        # i_{s-1}: A_{s-1} -> B_{s-1}
        p_s   = _tor_blockdiag_map_on_chains(p, gens_s,   TB.offsets[s+1], TC.offsets[s+1])
        i_sm1 = _tor_blockdiag_map_on_chains(i, gens_sm1, TA.offsets[s],   TB.offsets[s])

        # Boundary in B: d_s : B_s -> B_{s-1}
        dB = TB.bd[s]

        src = TC.homol[s+1]  # Tor_s(C)
        tgt = TA.homol[s]    # Tor_{s-1}(A)

        delta = zeros(K, tgt.dimH, src.dimH)

        for j in 1:src.dimH
            z = src.Hrep[:, j]  # cycle representative in C_s

            liftB = Utils.solve_particular(TA.resRop.M.field, p_s, reshape(z, :, 1))
            if liftB === nothing
                error("_connecting_tor: failed to lift a Tor cycle from C_s to B_s at s=$s")
            end
            b = liftB[:, 1]

            db = dB * b

            liftA = Utils.solve_particular(TA.resRop.M.field, i_sm1, reshape(db, :, 1))
            if liftA === nothing
                error("_connecting_tor: failed to lift a boundary from B_{s-1} to A_{s-1} at s=$s")
            end
            a = liftA[:, 1]

            coords = ChainComplexes.homology_coordinates(tgt, a)
            delta[:, j] = coords[:, 1]
        end

        return delta
    end

    function _connecting_tor(TA::TorSpaceSecond{K}, TB::TorSpaceSecond{K}, TC::TorSpaceSecond{K},
                            i::PMorphism{K}, p::PMorphism{K}; s::Int) where {K}
        if s < 1
            error("_connecting_tor: s must be >= 1 (got $s)")
        end

        @assert TA.resL.gens == TB.resL.gens == TC.resL.gens

        # Degree s and s-1 generator lists for the shared resolution of L.
        gens_s   = TA.resL.gens[s+1]  # chains in degree s
        gens_sm1 = TA.resL.gens[s]    # chains in degree s-1

        # Chain-level maps induced by i and p.
        p_s   = _tor_blockdiag_map_on_chains(p, gens_s,   TB.offsets[s+1], TC.offsets[s+1])
        i_sm1 = _tor_blockdiag_map_on_chains(i, gens_sm1, TA.offsets[s],   TB.offsets[s])

        dB = TB.bd[s]

        src = TC.homol[s+1]  # Tor_s(C)
        tgt = TA.homol[s]    # Tor_{s-1}(A)

        delta = zeros(K, tgt.dimH, src.dimH)

        for j in 1:src.dimH
            z = src.Hrep[:, j]

            liftB = Utils.solve_particular(TA.resL.M.field, p_s, reshape(z, :, 1))
            if liftB === nothing
                error("_connecting_tor: failed to lift a Tor cycle from C_s to B_s at s=$s")
            end
            b = liftB[:, 1]

            db = dB * b

            liftA = Utils.solve_particular(TA.resL.M.field, i_sm1, reshape(db, :, 1))
            if liftA === nothing
                error("_connecting_tor: failed to lift a boundary from B_{s-1} to A_{s-1} at s=$s")
            end
            a = liftA[:, 1]

            coords = ChainComplexes.homology_coordinates(tgt, a)
            delta[:, j] = coords[:, 1]
        end

        return delta
    end


    # ============================
    # Packaged Tor long exact sequences (LES) for short exact sequences
    # ============================

    """
        TorLongExactSequenceSecond(Rop, i, p, df::DerivedFunctorOptions)

    Package the long exact sequence in the *second* argument of Tor induced by a short exact sequence
    of P-modules

        0 -> A --i--> B --p--> C -> 0

    The resulting long exact sequence has the form

        ... -> Tor_s(Rop, A) -> Tor_s(Rop, B) -> Tor_s(Rop, C) -> Tor_{s-1}(Rop, A) -> ...

    We set `maxdeg = df.maxdeg` and force the Tor computation model to `:first` (resolve `Rop`) so that
    a single projective resolution of `Rop` can be reused for A, B, and C.

    Indexing convention:
    - `iH[s+1]` and `pH[s+1]` store the induced maps in degrees `s = 0..maxdeg`.
    - `delta[s]` stores the connecting map in degree `s = 1..maxdeg` (so `length(delta) == maxdeg`).

    The Tor spaces are stored as `TorA`, `TorB`, and `TorC`.
    """

    struct TorLongExactSequenceSecond{K}
        Rop::PModule{K}
        A::PModule{K}
        B::PModule{K}
        C::PModule{K}
        i::PMorphism{K}
        p::PMorphism{K}

        # Tor spaces (resolve the first variable).
        TorA::TorSpace{K}
        TorB::TorSpace{K}
        TorC::TorSpace{K}

        # Induced maps on Tor, stored by degree s=1..maxdeg.
        iH::Vector{Matrix{K}}
        pH::Vector{Matrix{K}}
        delta::Vector{Matrix{K}}

        maxdeg::Int
    end

    """
        TorLongExactSequenceSecond(Rop, i, p, df::DerivedFunctorOptions)
        TorLongExactSequenceSecond(Rop, ses, df::DerivedFunctorOptions)

    Package the long exact sequence in the second argument of Tor coming from a short exact sequence
    0 -> A --i--> B --p--> C -> 0:

    ... -> Tor_s(Rop,A) -> Tor_s(Rop,B) -> Tor_s(Rop,C) -> Tor_{s-1}(Rop,A) -> ...

    Maps are stored for s = 0..df.maxdeg. A single projective resolution of Rop is built and reused.
    This is computed using the :first model of Tor (resolve Rop).
    """
    function TorLongExactSequenceSecond(Rop::PModule{K}, i::PMorphism{K}, p::PMorphism{K},
                                    df::DerivedFunctorOptions) where {K}
        # For stability of the LES, we must resolve Rop (model = :first) so that all Tor spaces
        # share the same projective resolution of Rop.
        if !(df.model === :auto || df.model === :first)
            error("TorLongExactSequenceSecond requires model :first or :auto.")
        end
        maxdeg = df.maxdeg

        # Short exact sequence 0 -> A --i--> B --p--> C -> 0 in the second variable.
        A = i.dom
        B = i.cod
        C = p.cod
        @assert poset_equal(B.Q, A.Q) && poset_equal(B.Q, C.Q)
        @assert i.cod == p.dom

        # Shared resolution of Rop, padded out to maxdeg.
        resR = projective_resolution(Rop, ResolutionOptions(maxlen=maxdeg))
        _pad_projective_resolution!(resR, maxdeg)

        df_first = DerivedFunctorOptions(maxdeg=maxdeg, model=:first)
        TA = Tor(Rop, A, df_first; res=resR)
        TB = Tor(Rop, B, df_first; res=resR)
        TC = Tor(Rop, C, df_first; res=resR)

        # Induced maps on Tor in degrees 0..maxdeg.
        iH = [tor_map_second(TA, TB, i; s=s) for s in 0:maxdeg]
        pH = [tor_map_second(TB, TC, p; s=s) for s in 0:maxdeg]

        # Connecting maps delta_s: Tor_s(Rop,C) -> Tor_{s-1}(Rop,A) for s = 1..maxdeg.
        # Store a dummy zero map at index 1 (s=0) for consistent 1-based indexing.
        delta = Vector{Matrix{K}}(undef, maxdeg + 1)
        delta[1] = zeros(K, 0, dim(TC, 0))
        for s in 1:maxdeg
            delta[s + 1] = _connecting_tor(TA, TB, TC, i, p; s=s)
        end

        return TorLongExactSequenceSecond{K}(Rop, A, B, C, i, p, TA, TB, TC, iH, pH, delta, maxdeg)
    end


    function TorLongExactSequenceSecond(Rop::PModule{K}, ses::ShortExactSequence{K}, df::DerivedFunctorOptions) where {K}
        return TorLongExactSequenceSecond(Rop, ses.i, ses.p, df)
    end


    """
        TorLongExactSequenceFirst(L, i, p, df::DerivedFunctorOptions)

    Package the long exact sequence in the *first* argument of Tor induced by a short exact sequence
    of P^op-modules

        0 -> A --i--> B --p--> C -> 0

    The resulting long exact sequence has the form

        ... -> Tor_s(A, L) -> Tor_s(B, L) -> Tor_s(C, L) -> Tor_{s-1}(A, L) -> ...

    We set `maxdeg = df.maxdeg` and force the Tor computation model to `:second` (resolve `L`) so that
    a single projective resolution of `L` can be reused for A, B, and C.

    Indexing convention:
    - `iH[s+1]` and `pH[s+1]` store the induced maps in degrees `s = 0..maxdeg`.
    - `delta[s]` stores the connecting map in degree `s = 1..maxdeg` (so `length(delta) == maxdeg`).

    The Tor spaces are stored as `TorA`, `TorB`, and `TorC`.
    """

    struct TorLongExactSequenceFirst{K}
        L::PModule{K}
        A::PModule{K}
        B::PModule{K}
        C::PModule{K}
        i::PMorphism{K}
        p::PMorphism{K}

        # Tor spaces (resolve the second variable).
        TorA::TorSpaceSecond{K}
        TorB::TorSpaceSecond{K}
        TorC::TorSpaceSecond{K}

        # Induced maps on Tor, stored by degree s=1..maxdeg.
        iH::Vector{Matrix{K}}
        pH::Vector{Matrix{K}}
        delta::Vector{Matrix{K}}

        maxdeg::Int
    end

    """
        TorLongExactSequenceFirst(L, i, p, df::DerivedFunctorOptions)

    Compute and package the Tor long exact sequence for a short exact sequence in the first argument:

        0 -> A --i--> B --p--> C -> 0

    into maps

        ... -> Tor_s(A,L) -> Tor_s(B,L) -> Tor_s(C,L) -> Tor_{s-1}(A,L) -> ...

    and store `iH`, `pH` for degrees 0..maxdeg, and `delta` for degrees 1..maxdeg.
    """
    function TorLongExactSequenceFirst(L::PModule{K}, i::PMorphism{K}, p::PMorphism{K},
                                    df::DerivedFunctorOptions) where {K}
        # For stability of the LES, we must resolve L (model = :second) so that all Tor spaces
        # share the same projective resolution of L.
        if !(df.model === :auto || df.model === :second)
            error("TorLongExactSequenceFirst requires model :second or :auto.")
        end
        maxdeg = df.maxdeg

        # Short exact sequence 0 -> A --i--> B --p--> C -> 0 in the first variable.
        A = i.dom
        B = i.cod
        C = p.cod
        @assert poset_equal(B.Q, A.Q) && poset_equal(B.Q, C.Q)
        @assert i.cod == p.dom

        # Shared resolution of L, padded out to maxdeg.
        resL = projective_resolution(L, ResolutionOptions(maxlen=maxdeg))
        _pad_projective_resolution!(resL, maxdeg)

        df_second = DerivedFunctorOptions(maxdeg=maxdeg, model=:second)
        TA = Tor(A, L, df_second; res=resL)
        TB = Tor(B, L, df_second; res=resL)
        TC = Tor(C, L, df_second; res=resL)

        # Induced maps on Tor in degrees 0..maxdeg.
        iH = [tor_map_first(TA, TB, i; s=s) for s in 0:maxdeg]
        pH = [tor_map_first(TB, TC, p; s=s) for s in 0:maxdeg]

        # Connecting maps delta_s: Tor_s(C,L) -> Tor_{s-1}(A,L) for s = 1..maxdeg.
        # Store a dummy zero map at index 1 (s=0) for consistent 1-based indexing.
        delta = Vector{Matrix{K}}(undef, maxdeg + 1)
        delta[1] = zeros(K, 0, dim(TC, 0))
        for s in 1:maxdeg
            delta[s + 1] = _connecting_tor(TA, TB, TC, i, p; s=s)
        end

        return TorLongExactSequenceFirst{K}(L, A, B, C, i, p, TA, TB, TC, iH, pH, delta, maxdeg)
    end

    function TorLongExactSequenceFirst(L::PModule{K}, ses::ShortExactSequence{K}, df::DerivedFunctorOptions) where {K}
        return TorLongExactSequenceFirst(L, ses.i, ses.p, df)
    end

    @inline degree_range(les::ExtLongExactSequenceSecond) = les.tmin:les.tmax
    @inline degree_range(les::ExtLongExactSequenceFirst) = les.tmin:les.tmax
    @inline degree_range(les::TorLongExactSequenceSecond) = 0:les.maxdeg
    @inline degree_range(les::TorLongExactSequenceFirst) = 0:les.maxdeg

    @inline function _ext_les_index(les::Union{ExtLongExactSequenceSecond,ExtLongExactSequenceFirst}, degree::Int)
        degree in degree_range(les) || error("sequence access: degree $degree is outside $(first(degree_range(les))):$(last(degree_range(les))).")
        return degree - les.tmin + 1
    end

    @inline function _tor_les_index(les::Union{TorLongExactSequenceSecond,TorLongExactSequenceFirst}, degree::Int)
        degree in degree_range(les) || error("sequence access: degree $degree is outside 0:$(les.maxdeg).")
        return degree + 1
    end

    @inline function sequence_dimensions(les::ExtLongExactSequenceSecond, degree::Int)
        return (
            A=dim(les.EA, degree),
            B=dim(les.EB, degree),
            C=dim(les.EC, degree),
            Ashift=dim(les.EA, degree + 1),
        )
    end
    @inline sequence_dimensions(les::ExtLongExactSequenceSecond; degree::Int) = sequence_dimensions(les, degree)

    @inline function sequence_maps(les::ExtLongExactSequenceSecond{K}, degree::Int) where {K}
        idx = _ext_les_index(les, degree)
        return (i=les.iH[idx], p=les.pH[idx], delta=les.delta[idx])
    end
    @inline sequence_maps(les::ExtLongExactSequenceSecond{K}; degree::Int) where {K} = sequence_maps(les, degree)

    @inline function sequence_entry(les::ExtLongExactSequenceSecond{K}, degree::Int) where {K}
        return (; degree=degree, sequence_dimensions(les, degree)..., sequence_maps(les, degree)...)
    end
    @inline sequence_entry(les::ExtLongExactSequenceSecond{K}; degree::Int) where {K} = sequence_entry(les, degree)

    @inline function sequence_dimensions(les::ExtLongExactSequenceFirst, degree::Int)
        return (
            C=dim(les.EC, degree),
            B=dim(les.EB, degree),
            A=dim(les.EA, degree),
            Cshift=dim(les.EC, degree + 1),
        )
    end
    @inline sequence_dimensions(les::ExtLongExactSequenceFirst; degree::Int) = sequence_dimensions(les, degree)

    @inline function sequence_maps(les::ExtLongExactSequenceFirst{K}, degree::Int) where {K}
        idx = _ext_les_index(les, degree)
        return (p=les.pH[idx], i=les.iH[idx], delta=les.delta[idx])
    end
    @inline sequence_maps(les::ExtLongExactSequenceFirst{K}; degree::Int) where {K} = sequence_maps(les, degree)

    @inline function sequence_entry(les::ExtLongExactSequenceFirst{K}, degree::Int) where {K}
        return (; degree=degree, sequence_dimensions(les, degree)..., sequence_maps(les, degree)...)
    end
    @inline sequence_entry(les::ExtLongExactSequenceFirst{K}; degree::Int) where {K} = sequence_entry(les, degree)

    @inline function sequence_dimensions(les::TorLongExactSequenceSecond, degree::Int)
        return (
            A=dim(les.TorA, degree),
            B=dim(les.TorB, degree),
            C=dim(les.TorC, degree),
            Aprev=degree == 0 ? 0 : dim(les.TorA, degree - 1),
        )
    end
    @inline sequence_dimensions(les::TorLongExactSequenceSecond; degree::Int) = sequence_dimensions(les, degree)

    @inline function sequence_maps(les::TorLongExactSequenceSecond{K}, degree::Int) where {K}
        idx = _tor_les_index(les, degree)
        return (i=les.iH[idx], p=les.pH[idx], delta=les.delta[idx])
    end
    @inline sequence_maps(les::TorLongExactSequenceSecond{K}; degree::Int) where {K} = sequence_maps(les, degree)

    @inline function sequence_entry(les::TorLongExactSequenceSecond{K}, degree::Int) where {K}
        return (; degree=degree, sequence_dimensions(les, degree)..., sequence_maps(les, degree)...)
    end
    @inline sequence_entry(les::TorLongExactSequenceSecond{K}; degree::Int) where {K} = sequence_entry(les, degree)

    @inline function sequence_dimensions(les::TorLongExactSequenceFirst, degree::Int)
        return (
            A=dim(les.TorA, degree),
            B=dim(les.TorB, degree),
            C=dim(les.TorC, degree),
            Aprev=degree == 0 ? 0 : dim(les.TorA, degree - 1),
        )
    end
    @inline sequence_dimensions(les::TorLongExactSequenceFirst; degree::Int) = sequence_dimensions(les, degree)

    @inline function sequence_maps(les::TorLongExactSequenceFirst{K}, degree::Int) where {K}
        idx = _tor_les_index(les, degree)
        return (i=les.iH[idx], p=les.pH[idx], delta=les.delta[idx])
    end
    @inline sequence_maps(les::TorLongExactSequenceFirst{K}; degree::Int) where {K} = sequence_maps(les, degree)

    @inline function sequence_entry(les::TorLongExactSequenceFirst{K}, degree::Int) where {K}
        return (; degree=degree, sequence_dimensions(les, degree)..., sequence_maps(les, degree)...)
    end
    @inline sequence_entry(les::TorLongExactSequenceFirst{K}; degree::Int) where {K} = sequence_entry(les, degree)

    """
        derived_les_summary(seq) -> NamedTuple

    Cheap-first summary of a derived long exact sequence package.

    This reports the field, degree range, and endpoint dimensions without
    materializing any additional connecting-map data beyond what the sequence
    object already stores.

    Start with this, `sequence_dimensions(seq, t)`, and `sequence_maps(seq, t)`
    before inspecting specific connecting morphisms or individual entries.
    """
    @inline function _les_field(les::Union{ExtLongExactSequenceSecond,ExtLongExactSequenceFirst})
        return getfield(les, :EA).M.field
    end

    @inline function _les_field(les::Union{TorLongExactSequenceSecond,TorLongExactSequenceFirst})
        T = getfield(les, :TorA)
        return T isa TorSpaceSecond ? T.Rop.field : T.resRop.M.field
    end

    @inline function _les_summary(kind::Symbol, les)
        dr = degree_range(les)
        firstdeg = first(dr)
        lastdeg = last(dr)
        return (
            kind=kind,
            field=_les_field(les),
            degree_range=dr,
            first_entry=sequence_entry(les, firstdeg),
            last_entry=sequence_entry(les, lastdeg),
        )
    end

    @inline derived_les_summary(les::ExtLongExactSequenceSecond) = _les_summary(:ext_long_exact_sequence_second, les)
    @inline derived_les_summary(les::ExtLongExactSequenceFirst) = _les_summary(:ext_long_exact_sequence_first, les)
    @inline derived_les_summary(les::TorLongExactSequenceSecond) = _les_summary(:tor_long_exact_sequence_second, les)
    @inline derived_les_summary(les::TorLongExactSequenceFirst) = _les_summary(:tor_long_exact_sequence_first, les)

    function Base.show(io::IO, les::ExtLongExactSequenceSecond)
        d = derived_les_summary(les)
        print(io, "ExtLongExactSequenceSecond(degrees=", repr(d.degree_range), ")")
    end

    function Base.show(io::IO, ::MIME"text/plain", les::ExtLongExactSequenceSecond)
        d = derived_les_summary(les)
        print(io, "ExtLongExactSequenceSecond",
              "\n  field: ", d.field,
              "\n  degree_range: ", repr(d.degree_range),
              "\n  first_entry: ", repr(d.first_entry),
              "\n  last_entry: ", repr(d.last_entry))
    end

    function Base.show(io::IO, les::ExtLongExactSequenceFirst)
        d = derived_les_summary(les)
        print(io, "ExtLongExactSequenceFirst(degrees=", repr(d.degree_range), ")")
    end

    function Base.show(io::IO, ::MIME"text/plain", les::ExtLongExactSequenceFirst)
        d = derived_les_summary(les)
        print(io, "ExtLongExactSequenceFirst",
              "\n  field: ", d.field,
              "\n  degree_range: ", repr(d.degree_range),
              "\n  first_entry: ", repr(d.first_entry),
              "\n  last_entry: ", repr(d.last_entry))
    end

    function Base.show(io::IO, les::TorLongExactSequenceSecond)
        d = derived_les_summary(les)
        print(io, "TorLongExactSequenceSecond(degrees=", repr(d.degree_range), ")")
    end

    function Base.show(io::IO, ::MIME"text/plain", les::TorLongExactSequenceSecond)
        d = derived_les_summary(les)
        print(io, "TorLongExactSequenceSecond",
              "\n  field: ", d.field,
              "\n  degree_range: ", repr(d.degree_range),
              "\n  first_entry: ", repr(d.first_entry),
              "\n  last_entry: ", repr(d.last_entry))
    end

    function Base.show(io::IO, les::TorLongExactSequenceFirst)
        d = derived_les_summary(les)
        print(io, "TorLongExactSequenceFirst(degrees=", repr(d.degree_range), ")")
    end

    function Base.show(io::IO, ::MIME"text/plain", les::TorLongExactSequenceFirst)
        d = derived_les_summary(les)
        print(io, "TorLongExactSequenceFirst",
              "\n  field: ", d.field,
              "\n  degree_range: ", repr(d.degree_range),
              "\n  first_entry: ", repr(d.first_entry),
              "\n  last_entry: ", repr(d.last_entry))
    end

end
