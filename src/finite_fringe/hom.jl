# finite_fringe/hom.jl
# Scope: Hom-dimension planning, routing, dense/sparse kernels, and public hom_dimension entrypoints.

# ------------------ Hom for fringe modules via commuting squares ------------------
#
# We build the linear system for commuting squares in the indicator-module category
# using the full Hom descriptions from Prop. 3.10 (componentwise bases).
#
# V1 = Hom(F_M, F_N)  basis = components of U_M[i] contained in U_N[j]
# V2 = Hom(E_M, E_N)  basis = components of D_N[t] contained in D_M[s]
# W  = Hom(F_M, E_N)  basis = components of U_M[i] cap D_N[t]
#
# Then Hom(M,N) = ker(d0) / (ker(T) + ker(S)) with d0 = [T  -S].

const HOM_DIM_SPARSE_BUILD_MIN_ENTRIES = Ref(20_000)
const HOM_DIM_SPARSE_DENSITY_THRESHOLD = Ref(0.30)
const HOM_DIM_INTERNAL_DENSEIDX_TOTAL_THRESHOLD = Ref(24)
const HOM_DIM_INTERNAL_DENSEIDX_W_THRESHOLD = Ref(24)
const HOM_DIM_INTERNAL_DENSEIDX_WORK_THRESHOLD = Ref(384)
const HOM_DIM_INTERNAL_DENSEIDX_COMPONENT_THRESHOLD = Ref(12)
const HOM_DIM_INTERNAL_DENSEIDX_DENSITY_THRESHOLD = Ref(0.45)
const HOM_DIM_INTERNAL_DENSEIDX_TOTAL_AMBIGUITY_BAND = Ref(16)
const HOM_DIM_INTERNAL_DENSEIDX_W_AMBIGUITY_BAND = Ref(16)
const HOM_DIM_INTERNAL_DENSEIDX_WORK_AMBIGUITY_BAND = Ref(384)
const HOM_DIM_INTERNAL_DENSEIDX_COMPONENT_AMBIGUITY_BAND = Ref(8)
const HOM_DIM_INTERNAL_DENSEIDX_DENSITY_AMBIGUITY_BAND = Ref(0.08)
const HOM_DIM_DENSE_DENSITY_FULL_SCAN_ENTRIES = Ref(32_768)
const HOM_DIM_DENSE_DENSITY_SAMPLE_SIZE = Ref(4_096)

@inline function _matrix_density(A::SparseMatrixCSC)
    m, n = size(A)
    den = m * n
    den == 0 && return 0.0
    return nnz(A) / den
end

@inline function _estimate_dense_matrix_density(A::AbstractMatrix)
    den = length(A)
    den == 0 && return 0.0
    if den <= HOM_DIM_DENSE_DENSITY_FULL_SCAN_ENTRIES[]
        nz = 0
        @inbounds for v in A
            !iszero(v) && (nz += 1)
        end
        return nz / den
    end

    sample_n = min(den, HOM_DIM_DENSE_DENSITY_SAMPLE_SIZE[])
    step = max(1, fld(den, sample_n))
    nz = 0
    seen = 0
    idx = 0
    @inbounds for v in A
        idx += 1
        ((idx - 1) % step == 0) || continue
        !iszero(v) && (nz += 1)
        seen += 1
        seen >= sample_n && break
    end
    seen == 0 && return 0.0
    return nz / seen
end

_matrix_density(A::AbstractMatrix) = _estimate_dense_matrix_density(A)

@inline _hom_route_size_bin(w::Int) = w <= 0 ? 0 : (floor(Int, log2(float(w))) + 1)

@inline function _hom_route_features(M::FringeModule,
                                     N::FringeModule,
                                     layout::_FringeLayoutPlan)
    sketch = layout.sketch
    return (
        W_dim = layout.W_dim,
        total_dim = sketch.V1_dim + sketch.V2_dim,
        work_est = sketch.t_work_est + sketch.s_work_est,
        comp_total = sum(sketch.Ucomp_n_M) + sum(sketch.Dcomp_n_N),
        dmin = min(M.phi_density, N.phi_density),
        dmax = max(M.phi_density, N.phi_density),
    )
end

@inline function _prefer_dense_idx_internal(feat)
    return feat.W_dim <= HOM_DIM_INTERNAL_DENSEIDX_W_THRESHOLD[] &&
           feat.total_dim <= HOM_DIM_INTERNAL_DENSEIDX_TOTAL_THRESHOLD[] &&
           feat.work_est <= HOM_DIM_INTERNAL_DENSEIDX_WORK_THRESHOLD[] &&
           feat.comp_total <= HOM_DIM_INTERNAL_DENSEIDX_COMPONENT_THRESHOLD[] &&
           feat.dmin >= HOM_DIM_INTERNAL_DENSEIDX_DENSITY_THRESHOLD[]
end

@inline function _dense_idx_sparse_ambiguous(feat)
    return feat.W_dim <= HOM_DIM_INTERNAL_DENSEIDX_W_THRESHOLD[] +
                         HOM_DIM_INTERNAL_DENSEIDX_W_AMBIGUITY_BAND[] &&
           feat.total_dim <= HOM_DIM_INTERNAL_DENSEIDX_TOTAL_THRESHOLD[] +
                             HOM_DIM_INTERNAL_DENSEIDX_TOTAL_AMBIGUITY_BAND[] &&
           feat.work_est <= HOM_DIM_INTERNAL_DENSEIDX_WORK_THRESHOLD[] +
                            HOM_DIM_INTERNAL_DENSEIDX_WORK_AMBIGUITY_BAND[] &&
           feat.comp_total <= HOM_DIM_INTERNAL_DENSEIDX_COMPONENT_THRESHOLD[] +
                              HOM_DIM_INTERNAL_DENSEIDX_COMPONENT_AMBIGUITY_BAND[] &&
           feat.dmin >= HOM_DIM_INTERNAL_DENSEIDX_DENSITY_THRESHOLD[] -
                        HOM_DIM_INTERNAL_DENSEIDX_DENSITY_AMBIGUITY_BAND[]
end

@inline function _heuristic_hom_internal_choice(M::FringeModule,
                                                N::FringeModule)
    layout = _ensure_hom_layout_plan!(M, N)
    feat = _hom_route_features(M, N, layout)
    (feat.W_dim == 0 || feat.total_dim == 0) && return :sparse_path
    _prefer_dense_idx_internal(feat) && return :dense_idx_internal
    _dense_idx_sparse_ambiguous(feat) && return nothing
    return :sparse_path
end

@inline function _resolve_hom_dimension_path(M::FringeModule,
                                             N::FringeModule)
    choice = _heuristic_hom_internal_choice(M, N)
    return choice === nothing ? :auto_internal : choice
end

@inline _hom_density_bin(d::Float64) = clamp(Int(floor(d * 16.0)), 0, 16)
@inline _hom_work_bin(w::Int) = w <= 0 ? 0 : (floor(Int, log2(float(w))) + 1)

@inline function _hom_route_fingerprint(M::FringeModule,
                                        N::FringeModule,
                                        route::Symbol)
    layout = _ensure_hom_layout_plan!(M, N)
    feat = _hom_route_features(M, N, layout)
    return UInt64(hash((route,
                        length(M.U), length(M.D), length(N.U), length(N.D),
                        _hom_density_bin(M.phi_density), _hom_density_bin(N.phi_density),
                        _hom_work_bin(feat.work_est),
                        _hom_route_size_bin(feat.W_dim),
                        _hom_route_size_bin(feat.total_dim),
                        _hom_route_size_bin(feat.comp_total),
                        M.phi isa SparseMatrixCSC, N.phi isa SparseMatrixCSC,
                        typeof(M.field), typeof(M.phi), typeof(N.phi)), UInt(0)))
end

function _hcat_signed_sparse(T::SparseMatrixCSC{K,Int},
                             S::SparseMatrixCSC{K,Int}) where {K}
    mT, nT = size(T)
    mS, nS = size(S)
    mT == mS || error("_hcat_signed_sparse: row mismatch $mT vs $mS")

    nnzT = nnz(T)
    nnzS = nnz(S)
    colptr = Vector{Int}(undef, nT + nS + 1)
    rowval = Vector{Int}(undef, nnzT + nnzS)
    nzval = Vector{K}(undef, nnzT + nnzS)

    p = 1
    colptr[1] = 1
    @inbounds for j in 1:nT
        firstptr = T.colptr[j]
        lastptr = T.colptr[j + 1] - 1
        if firstptr <= lastptr
            len = lastptr - firstptr + 1
            copyto!(rowval, p, T.rowval, firstptr, len)
            copyto!(nzval, p, T.nzval, firstptr, len)
            p += len
        end
        colptr[j + 1] = p
    end
    @inbounds for j in 1:nS
        firstptr = S.colptr[j]
        lastptr = S.colptr[j + 1] - 1
        if firstptr <= lastptr
            len = lastptr - firstptr + 1
            copyto!(rowval, p, S.rowval, firstptr, len)
            for t in 0:(len - 1)
                nzval[p + t] = -S.nzval[firstptr + t]
            end
            p += len
        end
        colptr[nT + j + 1] = p
    end
    return SparseMatrixCSC(mT, nT + nS, colptr, rowval, nzval)
end

@inline function _rank_hcat_signed(field::AbstractCoeffField,
                                   T::SparseMatrixCSC,
                                   S::SparseMatrixCSC)
    return FieldLinAlg.rank(field, _hcat_signed_sparse(T, S))
end

@inline function _rank_hcat_signed(field::AbstractCoeffField,
                                   T::AbstractMatrix,
                                   S::AbstractMatrix)
    return FieldLinAlg.rank(field, hcat(T, -S))
end

@inline function _rank_hcat_signed_workspace!(field::AbstractCoeffField,
                                              out::AbstractMatrix{K},
                                              T::AbstractMatrix{K},
                                              S::AbstractMatrix{K}) where {K}
    m, nT = size(T)
    mS, nS = size(S)
    m == mS || error("_rank_hcat_signed_workspace!: row mismatch $m vs $mS")
    size(out, 1) == m && size(out, 2) == nT + nS ||
        error("_rank_hcat_signed_workspace!: workspace has wrong size")

    @inbounds for j in 1:nT
        for i in 1:m
            out[i, j] = T[i, j]
        end
    end
    @inbounds for j in 1:nS
        jj = nT + j
        for i in 1:m
            out[i, jj] = -S[i, j]
        end
    end
    return FieldLinAlg.rank(field, out)
end

function _build_sparse_hcat_workspace(T::SparseMatrixCSC{K,Int},
                                      S::SparseMatrixCSC{K,Int}) where {K}
    mT, nT = size(T)
    mS, nS = size(S)
    mT == mS || error("_build_sparse_hcat_workspace: row mismatch $mT vs $mS")

    nnzT = nnz(T)
    nnzS = nnz(S)
    colptr = Vector{Int}(undef, nT + nS + 1)
    rowval = Vector{Int}(undef, nnzT + nnzS)
    nzval = Vector{K}(undef, nnzT + nnzS)

    p = 1
    colptr[1] = 1
    @inbounds for j in 1:nT
        firstptr = T.colptr[j]
        lastptr = T.colptr[j + 1] - 1
        len = lastptr - firstptr + 1
        if len > 0
            copyto!(rowval, p, T.rowval, firstptr, len)
            p += len
        end
        colptr[j + 1] = p
    end
    @inbounds for j in 1:nS
        firstptr = S.colptr[j]
        lastptr = S.colptr[j + 1] - 1
        len = lastptr - firstptr + 1
        if len > 0
            copyto!(rowval, p, S.rowval, firstptr, len)
            p += len
        end
        colptr[nT + j + 1] = p
    end
    return SparseMatrixCSC(mT, nT + nS, colptr, rowval, nzval), nnzT
end

function _build_sparse_row_plan(A::SparseMatrixCSC{K,Int}) where {K}
    m, n = size(A)
    counts = zeros(Int, m)
    @inbounds for p in eachindex(A.rowval)
        counts[A.rowval[p]] += 1
    end

    ptr = Vector{Int}(undef, m + 1)
    ptr[1] = 1
    @inbounds for i in 1:m
        ptr[i + 1] = ptr[i] + counts[i]
    end

    cols = Vector{Int}(undef, length(A.rowval))
    nzptr = Vector{Int}(undef, length(A.rowval))
    nextptr = copy(ptr[1:m])
    @inbounds for col in 1:n
        for p in A.colptr[col]:(A.colptr[col + 1] - 1)
            row = A.rowval[p]
            q = nextptr[row]
            cols[q] = col
            nzptr[q] = p
            nextptr[row] = q + 1
        end
    end
    max_row_nnz = isempty(counts) ? 0 : maximum(counts)
    return _SparseRowPlan(ptr, cols, nzptr, counts, max_row_nnz)
end

@inline function _fill_sparse_row_from_plan!(
    row::FieldLinAlg.SparseRow{K},
    A::SparseMatrixCSC{K,Int},
    plan::_SparseRowPlan,
    ridx::Int,
) where {K}
    lo = plan.ptr[ridx]
    hi = plan.ptr[ridx + 1] - 1
    if hi < lo
        resize!(row.idx, 0)
        resize!(row.val, 0)
        return row
    end

    resize!(row.idx, hi - lo + 1)
    resize!(row.val, hi - lo + 1)
    w = 0
    @inbounds for p in lo:hi
        v = A.nzval[plan.nzptr[p]]
        iszero(v) && continue
        w += 1
        row.idx[w] = plan.cols[p]
        row.val[w] = v
    end
    resize!(row.idx, w)
    resize!(row.val, w)
    return row
end

@inline function _fill_sparse_union_row_from_right!(
    row::FieldLinAlg.SparseRow{K},
    L::SparseMatrixCSC{K,Int},
    Lplan::_SparseRowPlan,
    ridx::Int,
    right_row::FieldLinAlg.SparseRow{K},
    left_cols::Int,
) where {K}
    llo = Lplan.ptr[ridx]
    lhi = Lplan.ptr[ridx + 1] - 1
    maxlen = max(0, lhi - llo + 1) + length(right_row)
    if maxlen == 0
        resize!(row.idx, 0)
        resize!(row.val, 0)
        return row
    end

    resize!(row.idx, maxlen)
    resize!(row.val, maxlen)
    w = 0
    @inbounds begin
        for p in llo:lhi
            v = L.nzval[Lplan.nzptr[p]]
            iszero(v) && continue
            w += 1
            row.idx[w] = Lplan.cols[p]
            row.val[w] = v
        end
        for p in 1:length(right_row)
            w += 1
            row.idx[w] = left_cols + right_row.idx[p]
            row.val[w] = -right_row.val[p]
        end
    end
    resize!(row.idx, w)
    resize!(row.val, w)
    return row
end

@inline function _rank_hcat_signed_sparse_workspace!(field::AbstractCoeffField,
                                                     out::SparseMatrixCSC{K,Int},
                                                     T::SparseMatrixCSC{K,Int},
                                                     S::SparseMatrixCSC{K,Int},
                                                     nnzT::Int) where {K}
    nnzT == nnz(T) || error("_rank_hcat_signed_sparse_workspace!: nnzT mismatch")
    nnzS = nnz(S)
    nz = out.nzval
    @inbounds begin
        copyto!(nz, 1, T.nzval, 1, nnzT)
        for i in 1:nnzS
            nz[nnzT + i] = -S.nzval[i]
        end
    end
    return FieldLinAlg.rank(field, out)
end

@inline function _rank_hcat_signed_sparse_workspace_with_prefix_rank!(
    field::AbstractCoeffField,
    out::SparseMatrixCSC{K,Int},
    L::SparseMatrixCSC{K,Int},
    R::SparseMatrixCSC{K,Int},
    nnzL::Int,
    left_cols::Int,
) where {K}
    nnzL == nnz(L) || error("_rank_hcat_signed_sparse_workspace_with_prefix_rank!: nnzL mismatch")
    nnzR = nnz(R)
    nz = out.nzval
    @inbounds begin
        copyto!(nz, 1, L.nzval, 1, nnzL)
        for i in 1:nnzR
            nz[nnzL + i] = -R.nzval[i]
        end
    end

    m, n = size(out)
    if m == 0 || n == 0
        return 0, 0
    end
    red = FieldLinAlg._SparseRREF{K}(n)
    rows = FieldLinAlg._sparse_rows(out)
    maxrank = min(m, n)
    rleft = 0
    @inbounds for i in 1:m
        if FieldLinAlg._sparse_rref_push_homogeneous!(red, rows[i])
            red.pivot_cols[end] <= left_cols && (rleft += 1)
            length(red.pivot_cols) == maxrank && break
        end
    end
    return length(red.pivot_cols), rleft
end

@inline function _rank_hcat_signed_sparse_rowplans_with_side_rank!(
    union_red::FieldLinAlg._SparseREF{K},
    side_red::FieldLinAlg._SparseREF{K},
    union_row::FieldLinAlg.SparseRow{K},
    side_row::FieldLinAlg.SparseRow{K},
    L::SparseMatrixCSC{K,Int},
    Lplan::_SparseRowPlan,
    R::SparseMatrixCSC{K,Int},
    Rplan::_SparseRowPlan,
    left_cols::Int,
    cached_side_rank::Int = -1,
) where {K}
    m, nL = size(L)
    mR, nR = size(R)
    m == mR || error("_rank_hcat_signed_sparse_rowplans_with_side_rank!: row mismatch")
    left_cols == nL || error("_rank_hcat_signed_sparse_rowplans_with_side_rank!: left_cols mismatch")

    FieldLinAlg._reset_sparse_ref!(union_red)
    need_side_rank = cached_side_rank < 0
    need_side_rank && FieldLinAlg._reset_sparse_ref!(side_red)

    max_union = min(m, nL + nR)
    max_side = need_side_rank ? min(m, nR) : 0
    runion = 0
    rside = need_side_rank ? 0 : cached_side_rank
    rleft = 0
    @inbounds for i in 1:m
        if runion < max_union || (need_side_rank && rside < max_side)
            _fill_sparse_row_from_plan!(side_row, R, Rplan, i)
        else
            resize!(side_row.idx, 0)
            resize!(side_row.val, 0)
        end
        if runion < max_union
            _fill_sparse_union_row_from_right!(union_row, L, Lplan, i, side_row, left_cols)
            if !isempty(union_row) && FieldLinAlg._sparse_ref_push_homogeneous!(union_red, union_row)
                runion += 1
                union_red.pivot_cols[end] <= left_cols && (rleft += 1)
            end
        end
        if need_side_rank && rside < max_side &&
           !isempty(side_row) &&
           FieldLinAlg._sparse_ref_push_homogeneous!(side_red, side_row)
            rside += 1
        end
        if runion == max_union && (!need_side_rank || rside == max_side)
            break
        end
    end
    return runion, rleft, rside
end

@inline function _sparse_col_row_ptr(A::SparseMatrixCSC{K,Int},
                                     row::Int,
                                     col::Int) where {K}
    lo = A.colptr[col]
    hi = A.colptr[col + 1] - 1
    while lo <= hi
        mid = (lo + hi) >>> 1
        rv = A.rowval[mid]
        if rv < row
            lo = mid + 1
        elseif rv > row
            hi = mid - 1
        else
            return mid
        end
    end
    return 0
end

@inline function _hom_intersection_dim(field::AbstractCoeffField,
                                       T::AbstractMatrix,
                                       S::AbstractMatrix,
                                       rT::Int,
                                       rS::Int)
    (rT == 0 || rS == 0) && return 0
    BT = FieldLinAlg.colspace(field, T)
    BS = FieldLinAlg.colspace(field, S)
    (size(BT, 2) == 0 || size(BS, 2) == 0) && return 0
    rUnion = FieldLinAlg.rank(field, hcat(BT, BS))
    return rT + rS - rUnion
end

"Undirected adjacency of the Hasse cover graph."
function _cover_undirected_adjacency(P::AbstractPoset)
    return _get_cover_cache(P).undir::_PackedAdjacency
end

"Connected components of a subset mask in the undirected Hasse cover graph."
function _component_data(adj::_PackedAdjacency, mask::BitVector)
    n = length(mask)
    comp = fill(0, n)
    reps = Int[]
    cid = 0
    queue = Int[]
    for v in 1:n
        if mask[v] && comp[v] == 0
            cid += 1
            push!(reps, v)
            empty!(queue)
            push!(queue, v)
            comp[v] = cid
            head = 1
            while head <= length(queue)
                x = queue[head]
                head += 1
                lo, hi = _adj_bounds(adj, x)
                @inbounds for p in lo:hi
                    y = adj.idx[p]
                    if mask[y] && comp[y] == 0
                        comp[y] = cid
                        push!(queue, y)
                    end
                end
            end
        end
    end

    comp_masks = [falses(n) for _ in 1:cid]
    for v in 1:n
        c = comp[v]
        if c != 0
            comp_masks[c][v] = true
        end
    end
    return comp, cid, comp_masks, reps
end

@inline function _component_words(comp::Vector{Int}, ncomp::Int, n::Int)
    nchunks = cld(n, 64)
    words = [zeros(UInt64, nchunks) for _ in 1:ncomp]
    @inbounds for v in 1:n
        c = comp[v]
        c == 0 && continue
        word = ((v - 1) >>> 6) + 1
        bit = UInt64(1) << ((v - 1) & 63)
        words[c][word] |= bit
    end
    if ncomp > 0
        lastmask = _tailmask(n)
        @inbounds for c in 1:ncomp
            words[c][end] &= lastmask
        end
    end
    return words
end

@inline function _ensure_hom_cache!(M::FringeModule{K}) where {K}
    hc = M.hom_cache[]
    if hc.adj === nothing
        hc.adj = _cover_undirected_adjacency(M.P)
    end
    return hc::_FringeHomCache{K}
end

@inline function _pair_cache_ids(N::FringeModule)
    return UInt64(objectid(N)), UInt64(objectid(N.phi))
end

function _lookup_pair_cache(hc::_FringeHomCache{K},
                            N::FringeModule) where {K}
    pid, pphi = _pair_cache_ids(N)
    entries = hc.pair_cache
    @inbounds for i in eachindex(entries)
        entry = entries[i]
        if entry.partner_id == pid && entry.partner_phi_id == pphi
            if i != 1
                entries[i], entries[1] = entries[1], entries[i]
            end
            return entries[1]
        end
    end
    return nothing
end

function _ensure_pair_cache!(hc::_FringeHomCache{K},
                             N::FringeModule) where {K}
    entry = _lookup_pair_cache(hc, N)
    entry !== nothing && return entry::_FringePairCache{K}

    pid, pphi = _pair_cache_ids(N)
    entry = _FringePairCache{K}(pid, pphi, nothing, nothing, nothing, nothing, nothing)
    pushfirst!(hc.pair_cache, entry)
    max_entries = HOM_PAIR_CACHE_MAX_ENTRIES[]
    if length(hc.pair_cache) > max_entries
        resize!(hc.pair_cache, max_entries)
    end
    return entry
end

@inline function _route_fingerprint_choice_get(hc::_FringeHomCache, fkey::UInt64)
    entries = hc.route_fingerprint_choice
    @inbounds for i in eachindex(entries)
        entry = entries[i]
        entry.fingerprint == fkey && return entry.choice
    end
    return nothing
end

function _route_fingerprint_choice_set!(hc::_FringeHomCache,
                                        fkey::UInt64,
                                        choice::Symbol)
    entries = hc.route_fingerprint_choice
    @inbounds for i in eachindex(entries)
        if entries[i].fingerprint == fkey
            entries[i] = _FringeRouteChoiceEntry(fkey, choice)
            return choice
        end
    end
    push!(entries, _FringeRouteChoiceEntry(fkey, choice))
    return choice
end

@inline function _poset_cache_or_nothing(P)
    if hasproperty(P, :cache)
        pc = getproperty(P, :cache)
        return pc isa PosetCache ? pc : nothing
    end
    return nothing
end

@inline function _hom_route_choice_get(P, fkey::UInt64)
    pc = _poset_cache_or_nothing(P)
    pc === nothing && return nothing
    Base.lock(pc.lock)
    try
        return get(pc.hom_route_choice, fkey, nothing)
    finally
        Base.unlock(pc.lock)
    end
end

@inline function _hom_route_choice_set!(P, fkey::UInt64, choice::Symbol)
    pc = _poset_cache_or_nothing(P)
    pc === nothing && return nothing
    Base.lock(pc.lock)
    try
        pc.hom_route_choice[fkey] = choice
    finally
        Base.unlock(pc.lock)
    end
    return nothing
end

function _build_component_decomp(adj::_PackedAdjacency,
                                 sets::AbstractVector)
    nsets = length(sets)
    comp_id = Vector{Vector{Int}}(undef, nsets)
    comp_words = Vector{Vector{Vector{UInt64}}}(undef, nsets)
    comp_n = Vector{Int}(undef, nsets)
    @inbounds for i in 1:nsets
        cid, ncomp, _, _ = _component_data(adj, sets[i].mask)
        comp_id[i] = cid
        comp_words[i] = _component_words(cid, ncomp, length(sets[i].mask))
        comp_n[i] = ncomp
    end
    return _FringeComponentDecomp(comp_id, comp_words, comp_n)
end

@inline function _ensure_upset_component_decomp!(M::FringeModule)
    hc = _ensure_hom_cache!(M)
    if hc.upset === nothing
        hc.upset = _build_component_decomp(hc.adj::_PackedAdjacency, M.U)
    end
    return hc.upset::_FringeComponentDecomp
end

@inline function _ensure_downset_component_decomp!(M::FringeModule)
    hc = _ensure_hom_cache!(M)
    if hc.downset === nothing
        hc.downset = _build_component_decomp(hc.adj::_PackedAdjacency, M.D)
    end
    return hc.downset::_FringeComponentDecomp
end

@inline function _words_subset(words::Vector{UInt64},
                               target_chunks::Vector{UInt64},
                               lastmask::UInt64)
    nchunks = length(words)
    nchunks == 0 && return true
    @inbounds for w in 1:(nchunks - 1)
        (words[w] & ~target_chunks[w]) == 0 || return false
    end
    @inbounds return (words[nchunks] & ~target_chunks[nchunks] & lastmask) == 0
end

function _component_subset_targets(comp_words::Vector{Vector{Vector{UInt64}}},
                                   target_masks::Vector{BitVector})
    ntargets = length(target_masks)
    if ntargets == 0
        return [BitVector[] for _ in eachindex(comp_words)]
    end

    nverts = length(target_masks[1])
    lastmask = _tailmask(nverts)
    target_chunks = [mask.chunks for mask in target_masks]
    out = Vector{Vector{BitVector}}(undef, length(comp_words))
    @inbounds for i in eachindex(comp_words)
        cwords = comp_words[i]
        targets_i = Vector{BitVector}(undef, length(cwords))
        for c in eachindex(cwords)
            bits = falses(ntargets)
            words = cwords[c]
            for j in 1:ntargets
                _words_subset(words, target_chunks[j], lastmask) && (bits[j] = true)
            end
            targets_i[c] = bits
        end
        out[i] = targets_i
    end
    return out
end

@inline function _count_target_links(targets::Vector{Vector{BitVector}})
    total = 0
    @inbounds for targets_i in targets
        for bits in targets_i
            total += count(identity, bits)
        end
    end
    return total
end

@inline function _matrix_nnz_est(A::SparseMatrixCSC)
    return nnz(A)
end

@inline function _matrix_nnz_est(A::AbstractMatrix)
    return round(Int, _matrix_density(A) * length(A))
end

@inline function _component_target_masks(sets)
    return getfield.(sets, :mask)
end

function _build_hom_layout_sketch(M::FringeModule, N::FringeModule)
    Udec_M = _ensure_upset_component_decomp!(M)
    Ddec_N = _ensure_downset_component_decomp!(N)

    U_targets = _component_subset_targets(Udec_M.comp_words, _component_target_masks(N.U))
    D_targets = _component_subset_targets(Ddec_N.comp_words, _component_target_masks(M.D))

    V1_dim = _count_target_links(U_targets)
    V2_dim = _count_target_links(D_targets)
    nUN = length(N.U)
    nUM = length(M.U)
    t_work_est = fld(_matrix_nnz_est(N.phi) * max(V1_dim, 1), max(nUN, 1))
    s_work_est = fld(_matrix_nnz_est(M.phi) * max(V2_dim, 1), max(nUM, 1))
    return _FringeLayoutSketch(Udec_M.comp_id, Udec_M.comp_n,
                               Ddec_N.comp_id, Ddec_N.comp_n,
                               U_targets, D_targets,
                               V1_dim, V2_dim, t_work_est, s_work_est)
end

@inline function _ensure_hom_layout_sketch!(M::FringeModule{K},
                                            N::FringeModule) where {K}
    hc = _ensure_hom_cache!(M)
    entry = _ensure_pair_cache!(hc, N)
    sketch = entry.layout_sketch
    if sketch === nothing
        sketch = _build_hom_layout_sketch(M, N)
        entry.layout_sketch = sketch
    end
    return sketch::_FringeLayoutSketch
end

function _build_hom_layout_plan(M::FringeModule{K},
                                N::FringeModule{K}) where {K}
    @assert M.P === N.P "Posets must match"
    adj = (_ensure_hom_cache!(M).adj)::_PackedAdjacency
    sketch = _ensure_hom_layout_sketch!(M, N)
    nUM = length(M.U)
    nDN = length(N.D)
    w_index, w_data, W_dim = _build_wpair_layout(
        adj, M.U, N.D,
        sketch.Ucomp_id_M, sketch.Ucomp_n_M,
        sketch.Dcomp_id_N, sketch.Dcomp_n_N,
        nvertices(M.P)
    )
    return _FringeLayoutPlan(sketch, w_index, w_data, W_dim, nUM, nDN)
end

@inline function _ensure_hom_layout_plan!(M::FringeModule{K},
                                          N::FringeModule{K}) where {K}
    hc = _ensure_hom_cache!(M)
    entry = _ensure_pair_cache!(hc, N)
    plan = entry.layout_plan
    if plan === nothing
        plan = _build_hom_layout_plan(M, N)
        entry.layout_plan = plan
    end
    return plan::_FringeLayoutPlan
end

# Count connected components in mask_a intersect mask_b without materializing the intersection mask.
function _component_reps_intersection!(
    reps::Vector{Int},
    adj::_PackedAdjacency,
    mask_a::BitVector,
    mask_b::BitVector,
    marks::Vector{Int},
    mark::Int,
    queue::Vector{Int},
)
    empty!(reps)
    ncomp = 0
    n = length(mask_a)
    @inbounds for v in 1:n
        if marks[v] != mark && mask_a[v] && mask_b[v]
            ncomp += 1
            push!(reps, v)
            marks[v] = mark
            empty!(queue)
            push!(queue, v)
            head = 1
            while head <= length(queue)
                x = queue[head]
                head += 1
                lo, hi = _adj_bounds(adj, x)
                @inbounds for p in lo:hi
                    y = adj.idx[p]
                    if marks[y] != mark && mask_a[y] && mask_b[y]
                        marks[y] = mark
                        push!(queue, y)
                    end
                end
            end
        end
    end
    return ncomp
end

@inline function _pack_row_buckets(rows_by_comp::Vector{Vector{Int}})
    ncomp = length(rows_by_comp)
    ptr = Vector{Int}(undef, ncomp + 1)
    ptr[1] = 1
    total = 0
    @inbounds for c in 1:ncomp
        total += length(rows_by_comp[c])
        ptr[c + 1] = total + 1
    end
    rows = Vector{Int}(undef, total)
    p = 1
    @inbounds for c in 1:ncomp
        rc = rows_by_comp[c]
        len = length(rc)
        if len > 0
            copyto!(rows, p, rc, 1, len)
            p += len
        end
    end
    return ptr, rows
end

@inline function _wpair_u_bounds(w::_WPairData, cU::Int)
    return w.u_ptr[cU], w.u_ptr[cU + 1] - 1
end

@inline function _wpair_d_bounds(w::_WPairData, cD::Int)
    return w.d_ptr[cD], w.d_ptr[cD + 1] - 1
end

function _build_wpair_layout(adj::_PackedAdjacency,
                             Usets::AbstractVector,
                             Dsets::AbstractVector,
                             Ucomp_id::Vector{Vector{Int}},
                             Ucomp_n::Vector{Int},
                             Dcomp_id::Vector{Vector{Int}},
                             Dcomp_n::Vector{Int},
                             nverts::Int)
    nU = length(Usets)
    nD = length(Dsets)
    w_index = zeros(Int, nU, nD)
    w_data = _WPairData[]
    W_dim = 0
    marks = zeros(Int, nverts)
    queue = Int[]
    reps_int = Int[]
    mark = 1

    for iU in 1:nU
        for jD in 1:nD
            mark += 1
            if mark == typemax(Int)
                fill!(marks, 0)
                mark = 1
            end
            ncomp_int = _component_reps_intersection!(
                reps_int, adj, Usets[iU].mask, Dsets[jD].mask, marks, mark, queue
            )
            if ncomp_int > 0
                base = W_dim
                W_dim += ncomp_int

                rows_by_u = [Int[] for _ in 1:Ucomp_n[iU]]
                rows_by_d = [Int[] for _ in 1:Dcomp_n[jD]]
                @inbounds for c in 1:ncomp_int
                    row = base + c
                    v = reps_int[c]
                    cu = Ucomp_id[iU][v]
                    cd = Dcomp_id[jD][v]
                    push!(rows_by_u[cu], row)
                    push!(rows_by_d[cd], row)
                end
                u_ptr, u_rows = _pack_row_buckets(rows_by_u)
                d_ptr, d_rows = _pack_row_buckets(rows_by_d)
                push!(w_data, _WPairData(u_ptr, u_rows, d_ptr, d_rows))
                w_index[iU, jD] = length(w_data)
            end
        end
    end

    return w_index, w_data, W_dim
end

function _build_dense_idx_hom_plan(M::FringeModule{K},
                                   N::FringeModule{K}) where {K}
    @assert M.P === N.P "Posets must match"
    layout = _ensure_hom_layout_plan!(M, N)
    sketch = layout.sketch
    nUM = length(M.U)
    nDN = length(N.D)
    Ucomp_n_M = sketch.Ucomp_n_M
    Dcomp_n_N = sketch.Dcomp_n_N
    U_targets = sketch.U_targets
    D_targets = sketch.D_targets
    w_index = layout.w_index
    w_data = layout.w_data
    W_dim = layout.W_dim

    V1 = Tuple{Int,Int,Int}[]
    for iM in 1:nUM
        targets_i = U_targets[iM]
        for cU in 1:Ucomp_n_M[iM]
            jN = findnext(targets_i[cU], 1)
            while jN !== nothing
                push!(V1, (iM, jN, cU))
                jN = findnext(targets_i[cU], jN + 1)
            end
        end
    end
    V1_dim = length(V1)

    V2 = Tuple{Int,Int,Int}[]
    for tN in 1:nDN
        targets_t = D_targets[tN]
        for cD in 1:Dcomp_n_N[tN]
            sM = findnext(targets_t[cD], 1)
            while sM !== nothing
                push!(V2, (sM, tN, cD))
                sM = findnext(targets_t[cD], sM + 1)
            end
        end
    end
    V2_dim = length(V2)

    t_rows = Int[]
    t_cols = Int[]
    t_tN = Int[]
    t_jN = Int[]
    sizehint!(t_rows, max(16, W_dim * 4))
    sizehint!(t_cols, max(16, W_dim * 4))
    sizehint!(t_tN, max(16, W_dim * 4))
    sizehint!(t_jN, max(16, W_dim * 4))

    @inbounds for (col, (iM, jN, cU)) in enumerate(V1)
        for tN in 1:nDN
            pid = w_index[iM, tN]
            pid == 0 && continue
            w = w_data[pid]
            lo, hi = _wpair_u_bounds(w, cU)
            lo > hi && continue
            for k in lo:hi
                push!(t_rows, w.u_rows[k])
                push!(t_cols, col)
                push!(t_tN, tN)
                push!(t_jN, jN)
            end
        end
    end

    s_rows = Int[]
    s_cols = Int[]
    s_sM = Int[]
    s_iM = Int[]
    sizehint!(s_rows, max(16, W_dim * 4))
    sizehint!(s_cols, max(16, W_dim * 4))
    sizehint!(s_sM, max(16, W_dim * 4))
    sizehint!(s_iM, max(16, W_dim * 4))

    @inbounds for (col, (sM, tN, cD)) in enumerate(V2)
        for iM in 1:nUM
            pid = w_index[iM, tN]
            pid == 0 && continue
            w = w_data[pid]
            lo, hi = _wpair_d_bounds(w, cD)
            lo > hi && continue
            for k in lo:hi
                push!(s_rows, w.d_rows[k])
                push!(s_cols, col)
                push!(s_sM, sM)
                push!(s_iM, iM)
            end
        end
    end

    Tbuf = zeros(K, W_dim, V1_dim)
    Sbuf = zeros(K, W_dim, V2_dim)
    bigbuf = Matrix{K}(undef, W_dim, V1_dim + V2_dim)
    return _FringeDenseIdxPlan{K}(W_dim, V1_dim, V2_dim,
                                  t_rows, t_cols, t_tN, t_jN,
                                  s_rows, s_cols, s_sM, s_iM,
                                  Tbuf, Sbuf, bigbuf)
end

@inline function _ensure_dense_idx_hom_plan!(M::FringeModule{K},
                                             N::FringeModule{K}) where {K}
    hc = _ensure_hom_cache!(M)
    entry = _ensure_pair_cache!(hc, N)
    plan = entry.dense_idx_plan
    if plan === nothing
        plan = _build_dense_idx_hom_plan(M, N)
        entry.dense_idx_plan = plan
    end
    return plan::_FringeDenseIdxPlan{K}
end

function _hom_dimension_dense_idx_path(M::FringeModule{K},
                                         N::FringeModule{K}) where {K}
    plan = _ensure_dense_idx_hom_plan!(M, N)
    T = plan.Tbuf
    S = plan.Sbuf
    z = zero(K)
    fill!(T, z)
    fill!(S, z)

    Nphi = N.phi
    Mphi = M.phi
    @inbounds for i in eachindex(plan.t_rows)
        v = Nphi[plan.t_tN[i], plan.t_jN[i]]
        v == z && continue
        T[plan.t_rows[i], plan.t_cols[i]] += v
    end
    @inbounds for i in eachindex(plan.s_rows)
        v = Mphi[plan.s_sM[i], plan.s_iM[i]]
        v == z && continue
        S[plan.s_rows[i], plan.s_cols[i]] += v
    end

    rT = FieldLinAlg.rank(M.field, T)
    rS = FieldLinAlg.rank(M.field, S)
    rBig = _rank_hcat_signed_workspace!(M.field, plan.bigbuf, T, S)
    dimKer_big = (plan.V1_dim + plan.V2_dim) - rBig
    dimKer_T = plan.V1_dim - rT
    dimKer_S = plan.V2_dim - rS
    return dimKer_big - (dimKer_T + dimKer_S)
end

function _build_sparse_hom_plan(M::FringeModule{K},
                                N::FringeModule{K}) where {K}
    @assert M.P === N.P "Posets must match"
    layout = _ensure_hom_layout_plan!(M, N)
    sketch = layout.sketch
    nUM = length(M.U)
    nDN = length(N.D)
    Ucomp_n_M = sketch.Ucomp_n_M
    Dcomp_n_N = sketch.Dcomp_n_N
    U_targets = sketch.U_targets
    D_targets = sketch.D_targets
    w_index = layout.w_index
    w_data = layout.w_data
    W_dim = layout.W_dim

    V1 = NTuple{3,Int}[]
    for iM in 1:nUM
        targets_i = U_targets[iM]
        for cU in 1:Ucomp_n_M[iM]
            jN = findnext(targets_i[cU], 1)
            while jN !== nothing
                push!(V1, (iM, jN, cU))
                jN = findnext(targets_i[cU], jN + 1)
            end
        end
    end

    V2 = NTuple{3,Int}[]
    for tN in 1:nDN
        targets_t = D_targets[tN]
        for cD in 1:Dcomp_n_N[tN]
            sM = findnext(targets_t[cD], 1)
            while sM !== nothing
                push!(V2, (sM, tN, cD))
                sM = findnext(targets_t[cD], sM + 1)
            end
        end
    end

    z = zero(K)
    Nphi = N.phi
    t_colptr = Vector{Int}(undef, length(V1) + 1)
    t_rowval = Int[]
    t_tN = Int[]
    t_jN = Int[]
    t_colptr[1] = 1
    @inbounds for col in eachindex(V1)
        (iM, jN, cU) = V1[col]
        for tN in 1:nDN
            Nphi[tN, jN] == z && continue
            pid = w_index[iM, tN]
            pid == 0 && continue
            w = w_data[pid]
            lo, hi = _wpair_u_bounds(w, cU)
            for k in lo:hi
                push!(t_rowval, w.u_rows[k])
                push!(t_tN, tN)
                push!(t_jN, jN)
            end
        end
        t_colptr[col + 1] = length(t_rowval) + 1
    end
    t_nzval = Vector{K}(undef, length(t_rowval))
    @inbounds for i in eachindex(t_nzval)
        t_nzval[i] = Nphi[t_tN[i], t_jN[i]]
    end
    t_nzptr = if Nphi isa SparseMatrixCSC{K,Int}
        ptrs = Vector{Int}(undef, length(t_tN))
        @inbounds for i in eachindex(t_tN)
            p = _sparse_col_row_ptr(Nphi, t_tN[i], t_jN[i])
            p == 0 && error("_build_sparse_hom_plan: internal sparse pointer miss for T entry.")
            ptrs[i] = p
        end
        ptrs
    else
        nothing
    end
    t_nzptr_max = (t_nzptr === nothing || isempty(t_nzptr)) ? 0 : maximum(t_nzptr)
    T = SparseMatrixCSC(W_dim, length(V1), t_colptr, t_rowval, t_nzval)
    t_rows = _build_sparse_row_plan(T)

    Mphi = M.phi
    s_colptr = Vector{Int}(undef, length(V2) + 1)
    s_rowval = Int[]
    s_sM = Int[]
    s_iM = Int[]
    s_colptr[1] = 1
    @inbounds for col in eachindex(V2)
        (sM, tN, cD) = V2[col]
        for iM in 1:nUM
            Mphi[sM, iM] == z && continue
            pid = w_index[iM, tN]
            pid == 0 && continue
            w = w_data[pid]
            lo, hi = _wpair_d_bounds(w, cD)
            for k in lo:hi
                push!(s_rowval, w.d_rows[k])
                push!(s_sM, sM)
                push!(s_iM, iM)
            end
        end
        s_colptr[col + 1] = length(s_rowval) + 1
    end
    s_nzval = Vector{K}(undef, length(s_rowval))
    @inbounds for i in eachindex(s_nzval)
        s_nzval[i] = Mphi[s_sM[i], s_iM[i]]
    end
    s_nzptr = if Mphi isa SparseMatrixCSC{K,Int}
        ptrs = Vector{Int}(undef, length(s_sM))
        @inbounds for i in eachindex(s_sM)
            p = _sparse_col_row_ptr(Mphi, s_sM[i], s_iM[i])
            p == 0 && error("_build_sparse_hom_plan: internal sparse pointer miss for S entry.")
            ptrs[i] = p
        end
        ptrs
    else
        nothing
    end
    s_nzptr_max = (s_nzptr === nothing || isempty(s_nzptr)) ? 0 : maximum(s_nzptr)
    S = SparseMatrixCSC(W_dim, length(V2), s_colptr, s_rowval, s_nzval)
    s_rows = _build_sparse_row_plan(S)

    hcat_buf, nnzT = _build_sparse_hcat_workspace(T, S)
    hcat_buf_rev, nnzS = _build_sparse_hcat_workspace(S, T)
    union_row = FieldLinAlg.SparseRow{K}()
    side_row = FieldLinAlg.SparseRow{K}()
    return _FringeSparsePlan{K}(W_dim, V1, V2, w_index, w_data, nUM, nDN,
                                T, S, t_tN, t_jN, s_sM, s_iM,
                                t_rows, s_rows,
                                t_nzptr, s_nzptr,
                                t_nzptr_max, s_nzptr_max,
                                hcat_buf, hcat_buf_rev,
                                nnzT, nnzS,
                                FieldLinAlg._SparseREF{K}(size(T, 2) + size(S, 2)),
                                FieldLinAlg._SparseREF{K}(size(T, 2)),
                                FieldLinAlg._SparseREF{K}(size(S, 2)),
                                union_row,
                                side_row,
                                0, 0, false, false)
end

@inline function _ensure_sparse_hom_plan!(M::FringeModule{K},
                                          N::FringeModule{K}) where {K}
    hc = _ensure_hom_cache!(M)
    entry = _ensure_pair_cache!(hc, N)
    plan = entry.sparse_plan
    if plan === nothing
        plan = _build_sparse_hom_plan(M, N)
        entry.sparse_plan = plan
    end
    return plan::_FringeSparsePlan{K}
end

function _hom_dimension_sparse_path(M::FringeModule{K}, N::FringeModule{K}) where {K}
    plan = _ensure_sparse_hom_plan!(M, N)
    T = plan.T
    S = plan.S
    z = zero(K)
    Nphi = N.phi
    Mphi = M.phi

    t_nzptr = plan.t_nzptr
    sameT = true
    if t_nzptr !== nothing && Nphi isa SparseMatrixCSC{K,Int} &&
       plan.t_nzptr_max <= length(Nphi.nzval)
        @inbounds for i in eachindex(T.nzval)
            v = Nphi.nzval[t_nzptr[i]]
            sameT && T.nzval[i] != v && (sameT = false)
            T.nzval[i] = v == z ? z : v
        end
    else
        @inbounds for i in eachindex(T.nzval)
            v = Nphi[plan.t_tN[i], plan.t_jN[i]]
            sameT && T.nzval[i] != v && (sameT = false)
            T.nzval[i] = v == z ? z : v
        end
    end

    s_nzptr = plan.s_nzptr
    sameS = true
    if s_nzptr !== nothing && Mphi isa SparseMatrixCSC{K,Int} &&
       plan.s_nzptr_max <= length(Mphi.nzval)
        @inbounds for i in eachindex(S.nzval)
            v = Mphi.nzval[s_nzptr[i]]
            sameS && S.nzval[i] != v && (sameS = false)
            S.nzval[i] = v == z ? z : v
        end
    else
        @inbounds for i in eachindex(S.nzval)
            v = Mphi[plan.s_sM[i], plan.s_iM[i]]
            sameS && S.nzval[i] != v && (sameS = false)
            S.nzval[i] = v == z ? z : v
        end
    end

    nT = size(T, 2)
    nS = size(S, 2)
    if nS <= nT
        cached_rS = (sameS && plan.cached_S_valid) ? plan.cached_rS : -1
        rUnion, rT, rS = _rank_hcat_signed_sparse_rowplans_with_side_rank!(
            plan.union_red, plan.s_red, plan.union_row, plan.side_row,
            T, plan.t_rows, S, plan.s_rows, nT, cached_rS,
        )
        plan.cached_rS = rS
        plan.cached_S_valid = true
        return rT + rS - rUnion
    else
        cached_rT = (sameT && plan.cached_T_valid) ? plan.cached_rT : -1
        rUnion, rS, rT = _rank_hcat_signed_sparse_rowplans_with_side_rank!(
            plan.union_red, plan.t_red, plan.union_row, plan.side_row,
            S, plan.s_rows, T, plan.t_rows, nS, cached_rT,
        )
        plan.cached_rT = rT
        plan.cached_T_valid = true
        return rT + rS - rUnion
    end
end

@inline function _hom_dimension_with_path(M::FringeModule{K},
                                          N::FringeModule{K},
                                          path::Symbol) where {K}
    path === :sparse_path && return _hom_dimension_sparse_path(M, N)
    path === :dense_idx_internal && return _hom_dimension_dense_idx_path(M, N)
    error("_hom_dimension_with_path: unknown path $(repr(path)).")
end

@inline function _store_hom_route_choice!(hc::_FringeHomCache,
                                          P,
                                          entry,
                                          fkey::UInt64,
                                          choice::Symbol)
    entry.route_choice = choice
    _route_fingerprint_choice_set!(hc, fkey, choice)
    _hom_route_choice_set!(P, fkey, choice)
    return choice
end

function _select_hom_internal_path_timed!(M::FringeModule{K},
                                          N::FringeModule{K},
                                          hc::_FringeHomCache{K},
                                          entry::_FringePairCache{K},
                                          fkey::UInt64) where {K}
    hc.route_timing_fallbacks += 1
    ref_val = nothing
    best = typemax(UInt64)
    choice = :sparse_path
    for path in (:dense_idx_internal, :sparse_path)
        t0 = time_ns()
        v = _hom_dimension_with_path(M, N, path)
        dt = time_ns() - t0
        if ref_val === nothing
            ref_val = v
        else
            v == ref_val || error("hom_dimension path mismatch during one-shot internal selection.")
        end
        if dt < best
            best = dt
            choice = path
        end
    end
    return _store_hom_route_choice!(hc, M.P, entry, fkey, choice)
end

function _select_hom_internal_path!(M::FringeModule{K},
                                    N::FringeModule{K}) where {K}
    hc = _ensure_hom_cache!(M)
    entry = _ensure_pair_cache!(hc, N)
    choice = entry.route_choice
    choice !== nothing && return choice::Symbol

    _ = _ensure_hom_layout_plan!(M, N)
    fkey = _hom_route_fingerprint(M, N, :internal_choice)
    choice = _hom_route_choice_get(M.P, fkey)
    if choice !== nothing
        entry.route_choice = choice
        _route_fingerprint_choice_set!(hc, fkey, choice)
        return choice::Symbol
    end
    choice = _route_fingerprint_choice_get(hc, fkey)
    if choice !== nothing
        entry.route_choice = choice
        return choice::Symbol
    end

    choice = _heuristic_hom_internal_choice(M, N)
    if choice !== nothing
        return _store_hom_route_choice!(hc, M.P, entry, fkey, choice)
    end

    return _select_hom_internal_path_timed!(M, N, hc, entry, fkey)
end

function _clear_hom_route_choice!(M::FringeModule{K}) where {K}
    hc = _ensure_hom_cache!(M)
    for entry in hc.pair_cache
        entry.route_choice = nothing
    end
    empty!(hc.route_fingerprint_choice)
    hc.route_timing_fallbacks = 0
    pc = _poset_cache_or_nothing(M.P)
    if pc !== nothing
        Base.lock(pc.lock)
        try
            empty!(pc.hom_route_choice)
        finally
            Base.unlock(pc.lock)
        end
    end
    return nothing
end

"""
    hom_dimension(M, N) -> Int

Return the dimension of the morphism space `Hom(M, N)` in the finite-fringe
category over the common coefficient field.

# Mathematical meaning

For fringe modules `M` and `N` over the same ambient finite poset and the same
ground field `k`, this computes

```math
\\dim_k \\operatorname{Hom}(M, N).
```

The implementation uses the production route selector and automatically chooses
between dense-index and sparse internal kernels.

# Inputs

- `M::FringeModule{K}`: source fringe module.
- `N::FringeModule{K}`: target fringe module.

# Output

- An `Int` equal to the dimension of the `k`-vector space `Hom(M, N)`.

# Domain/codomain conventions

- `M` is the domain object.
- `N` is the codomain object.
- Both modules must live over the same ambient poset object.

# Failure / contract behavior

- Throws if `M.P !== N.P`.
- Assumes both modules satisfy the usual finite-fringe presentation invariants.

# Performance notes

- Repeated calls on the same source/target pair automatically benefit from
  internal `Hom` planning, route selection, and sparse/dense kernel caches.
- Users normally should not force internal routes; the canonical call is
  `hom_dimension(M, N)`.

# Best practices

- Use this as the primary user-facing `Hom`-dimension query.
- If you are comparing many targets against the same source (or vice versa),
  repeated calls are expected and are a supported fast path.
- This function returns only the dimension of the `Hom` space. It does not
  compute a basis of morphisms or a matrix model of the morphism space; use it
  when the dimension is the mathematical quantity you actually want.

# Examples

```jldoctest
julia> using TamerOp

julia> const FF = TamerOp.FiniteFringe
Main.TamerOp.FiniteFringe

julia> P = FF.FinitePoset(Bool[1 1 1;
                              0 1 1;
                              0 0 1]);

julia> M = FF.one_by_one_fringe(P,
                                FF.principal_upset(P, 2),
                                FF.principal_downset(P, 2),
                                1; field=QQField());

julia> FF.hom_dimension(M, M)
1
```
"""
function hom_dimension(M::FringeModule{K}, N::FringeModule{K}) where {K}
    @assert M.P === N.P "Posets must match"
    path = _select_hom_internal_path!(M, N)
    return _hom_dimension_with_path(M, N, path)
end


# ---------- utility: dense\tosparse over K ----------

"""
    _dense_to_sparse_K(A)

Convert a dense matrix `A` to a sparse matrix with the same element type.

For an explicit target coefficient type `K`, call `_dense_to_sparse_K(A, K)`.
"""
function _dense_to_sparse_K(A::AbstractMatrix{T}, ::Type{K}) where {T,K}
    m,n = size(A)
    S = spzeros(K, m, n)
    for j in 1:n, i in 1:m
        v = K(A[i,j])
        if v != zero(K); S[i,j] = v; end
    end
    S
end

_dense_to_sparse_K(A::AbstractMatrix{T}) where {T} = _dense_to_sparse_K(A, T)
