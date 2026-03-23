using Random
using Printf
using SparseArrays

include(joinpath(@__DIR__, "..", "src", "DataTypes.jl"))
const DT = DataTypes

function _parse_flag(args::Vector{String}, key::String, default)
    prefix = key * "="
    for arg in args
        startswith(arg, prefix) || continue
        value = split(arg, "=", limit=2)[2]
        return default isa Int ? parse(Int, value) : value
    end
    return default
end

function _median_stats(f::Function; reps::Int)
    times = Vector{Float64}(undef, reps)
    allocs = Vector{Int}(undef, reps)
    for i in 1:reps
        stats = @timed f()
        times[i] = stats.time
        allocs[i] = stats.bytes
    end
    p = sortperm(times)
    mid = p[cld(reps, 2)]
    return (time=times[mid], bytes=allocs[mid])
end

function _triangulated_fixture(ntri::Int)
    nv = 3 * ntri
    ne = 3 * ntri
    nf = ntri
    cells = [collect(1:nv), collect(1:ne), collect(1:nf)]

    I1 = Vector{Int}(undef, 2 * ne)
    J1 = Vector{Int}(undef, 2 * ne)
    V1 = Vector{Int}(undef, 2 * ne)
    I2 = Vector{Int}(undef, 3 * nf)
    J2 = Vector{Int}(undef, 3 * nf)
    V2 = Vector{Int}(undef, 3 * nf)

    p1 = 1
    p2 = 1
    for t in 1:ntri
        v0 = 3 * (t - 1)
        e0 = 3 * (t - 1)
        a = v0 + 1
        b = v0 + 2
        c = v0 + 3
        eab = e0 + 1
        ebc = e0 + 2
        eac = e0 + 3

        I1[p1] = a; J1[p1] = eab; V1[p1] = 1; p1 += 1
        I1[p1] = b; J1[p1] = eab; V1[p1] = -1; p1 += 1
        I1[p1] = b; J1[p1] = ebc; V1[p1] = 1; p1 += 1
        I1[p1] = c; J1[p1] = ebc; V1[p1] = -1; p1 += 1
        I1[p1] = a; J1[p1] = eac; V1[p1] = 1; p1 += 1
        I1[p1] = c; J1[p1] = eac; V1[p1] = -1; p1 += 1

        I2[p2] = eab; J2[p2] = t; V2[p2] = 1; p2 += 1
        I2[p2] = ebc; J2[p2] = t; V2[p2] = -1; p2 += 1
        I2[p2] = eac; J2[p2] = t; V2[p2] = 1; p2 += 1
    end

    boundaries = [
        sparse(I1, J1, V1, nv, ne),
        sparse(I2, J2, V2, ne, nf),
    ]

    total = nv + ne + nf
    grades = Vector{NTuple{2,Float64}}(undef, total)
    multigrades = Vector{Vector{NTuple{2,Float64}}}(undef, total)
    for i in 1:total
        g = (Float64(mod(i, 97)), Float64(mod(3 * i, 89)))
        grades[i] = g
        multigrades[i] = isodd(i) ? [g] : [g, (g[1] + 1.0, g[2] + 1.0)]
    end

    Gg = DT.GradedComplex(cells, boundaries, grades)
    Gm = DT.MultiCriticalGradedComplex(cells, boundaries, multigrades)
    return (; Gg, Gm)
end

@inline function _tuple_dom_leq(a::NTuple{N,T}, b::NTuple{N,T}) where {N,T}
    @inbounds for i in 1:N
        a[i] <= b[i] || return false
    end
    return true
end

function _minimal_tuple_set(vals::Vector{NTuple{N,T}}) where {N,T}
    isempty(vals) && return vals
    keep = trues(length(vals))
    @inbounds for i in eachindex(vals)
        keep[i] || continue
        vi = vals[i]
        for j in eachindex(vals)
            i == j && continue
            _tuple_dom_leq(vals[j], vi) || continue
            if vals[j] != vi
                keep[i] = false
                break
            end
        end
    end
    return vals[keep]
end

function _maximal_tuple_set(vals::Vector{NTuple{N,T}}) where {N,T}
    isempty(vals) && return vals
    keep = trues(length(vals))
    @inbounds for i in eachindex(vals)
        keep[i] || continue
        vi = vals[i]
        for j in eachindex(vals)
            i == j && continue
            _tuple_dom_leq(vi, vals[j]) || continue
            if vals[j] != vi
                keep[i] = false
                break
            end
        end
    end
    return vals[keep]
end

@inline function _tuple_lex_lt(a::NTuple{N,T}, b::NTuple{N,T}) where {N,T}
    @inbounds for i in 1:N
        ai = a[i]
        bi = b[i]
        ai < bi && return true
        ai > bi && return false
    end
    return false
end

function _select_one_critical_grade(grades::AbstractVector{<:NTuple{N,T}},
                                    selector::Symbol) where {N,T}
    best = grades[1]
    if selector === :first
        return best
    elseif selector === :lexmin
        @inbounds for i in 2:length(grades)
            gi = grades[i]
            if _tuple_lex_lt(gi, best)
                best = gi
            end
        end
        return best
    elseif selector === :lexmax
        @inbounds for i in 2:length(grades)
            gi = grades[i]
            if _tuple_lex_lt(best, gi)
                best = gi
            end
        end
        return best
    end
    error("unsupported selector")
end

@inline function _tuple_componentwise_max(a::NTuple{N,T}, b::NTuple{N,T}) where {N,T}
    return ntuple(i -> (a[i] >= b[i] ? a[i] : b[i]), N)
end

_old_estimate_cell_counts(G::Union{DT.GradedComplex,DT.MultiCriticalGradedComplex}) =
    BigInt[big(length(cells)) for cells in G.cells_by_dim]

function _new_estimate_cell_counts(G::Union{DT.GradedComplex,DT.MultiCriticalGradedComplex})
    _, dim_offsets = DT._packed_cell_storage(G)
    nd = length(dim_offsets) - 1
    out = Vector{BigInt}(undef, nd)
    @inbounds for slot in 1:nd
        out[slot] = big(DT._packed_dim_count(dim_offsets, slot))
    end
    return out
end

_old_criticality(G::DT.MultiCriticalGradedComplex) = maximum(length, G.grades)

function _new_criticality(G::DT.MultiCriticalGradedComplex)
    grade_offsets, _ = DT._packed_multigrade_storage(G)
    maxcrit = 0
    @inbounds for i in 1:(length(grade_offsets) - 1)
        maxcrit = max(maxcrit, grade_offsets[i + 1] - grade_offsets[i])
    end
    return maxcrit
end

function _old_normalize_multicritical(G::DT.MultiCriticalGradedComplex; keep::Symbol=:minimal)
    out = Vector{Vector{typeof(G.grades[1][1])}}(undef, length(G.grades))
    for i in eachindex(G.grades)
        gi = unique(G.grades[i])
        if keep === :minimal
            gi = _minimal_tuple_set(gi)
        elseif keep === :maximal
            gi = _maximal_tuple_set(gi)
        end
        out[i] = gi
    end
    return DT.MultiCriticalGradedComplex(G.cells_by_dim, G.boundaries, out; cell_dims=G.cell_dims)
end

function _new_normalize_multicritical(G::DT.MultiCriticalGradedComplex; keep::Symbol=:minimal)
    keep === :unique && return G
    grade_offsets, grade_data = DT._packed_multigrade_storage(G)
    N = length(grade_data[1])
    T = eltype(grade_data[1])
    out_offsets = Vector{Int}(undef, length(grade_offsets))
    out_offsets[1] = 1
    out_data = NTuple{N,T}[]
    sizehint!(out_data, length(grade_data))
    @inbounds for i in 1:(length(grade_offsets) - 1)
        lo = grade_offsets[i]
        hi = grade_offsets[i + 1] - 1
        gi = Vector{NTuple{N,T}}(undef, hi - lo + 1)
        copyto!(gi, 1, grade_data, lo, length(gi))
        if keep === :minimal
            gi = _minimal_tuple_set(gi)
        else
            gi = _maximal_tuple_set(gi)
        end
        append!(out_data, gi)
        out_offsets[i + 1] = length(out_data) + 1
    end
    return DT._rebuild_multicritical_complex(G, out_offsets, out_data)
end

function _old_grades_by_dim(G::DT.MultiCriticalGradedComplex)
    counts = [length(c) for c in G.cells_by_dim]
    Tgrade = typeof(G.grades[1])
    out = Vector{Vector{Tgrade}}(undef, length(counts))
    idx = 1
    for d in 1:length(counts)
        out[d] = Vector{Tgrade}(undef, counts[d])
        for j in 1:counts[d]
            out[d][j] = G.grades[idx]
            idx += 1
        end
    end
    return out
end

function _new_grades_by_dim(G::DT.MultiCriticalGradedComplex)
    _, dim_offsets = DT._packed_cell_storage(G)
    grade_offsets, grade_data = DT._packed_multigrade_storage(G)
    counts = DT._packed_dim_counts(dim_offsets)
    CellT = SubArray{eltype(grade_data),1,typeof(grade_data),Tuple{UnitRange{Int}},true}
    out = Vector{Vector{CellT}}(undef, length(counts))
    idx = 1
    @inbounds for d in 1:length(counts)
        out[d] = Vector{CellT}(undef, counts[d])
        for j in 1:counts[d]
            lo = grade_offsets[idx]
            hi = grade_offsets[idx + 1] - 1
            out[d][j] = @view grade_data[lo:hi]
            idx += 1
        end
    end
    return out
end

function _old_one_criticalify(G::DT.MultiCriticalGradedComplex;
                              selector::Symbol=:lexmin,
                              enforce_boundary::Bool=true)
    total = length(G.grades)
    grades = Vector{typeof(G.grades[1][1])}(undef, total)
    @inbounds for i in 1:total
        grades[i] = _select_one_critical_grade(G.grades[i], selector)
    end
    if enforce_boundary
        counts = [length(c) for c in G.cells_by_dim]
        offsets = Vector{Int}(undef, length(counts) + 1)
        offsets[1] = 0
        @inbounds for d in 1:length(counts)
            offsets[d + 1] = offsets[d] + counts[d]
        end
        @inbounds for d in 2:length(counts)
            B = G.boundaries[d - 1]
            prev_off = offsets[d - 1]
            cur_off = offsets[d]
            for j in 1:counts[d]
                g = grades[cur_off + j]
                for p in nzrange(B, j)
                    i = rowvals(B)[p]
                    g = _tuple_componentwise_max(g, grades[prev_off + i])
                end
                grades[cur_off + j] = g
            end
        end
    end
    return DT.GradedComplex(G.cells_by_dim, G.boundaries, grades; cell_dims=G.cell_dims)
end

function _new_one_criticalify(G::DT.MultiCriticalGradedComplex;
                              selector::Symbol=:lexmin,
                              enforce_boundary::Bool=true)
    _, dim_offsets = DT._packed_cell_storage(G)
    grade_offsets, grade_data = DT._packed_multigrade_storage(G)
    total = length(grade_offsets) - 1
    grades = Vector{eltype(grade_data)}(undef, total)
    @inbounds for i in 1:total
        lo = grade_offsets[i]
        hi = grade_offsets[i + 1] - 1
        grades[i] = _select_one_critical_grade((@view grade_data[lo:hi]), selector)
    end
    if enforce_boundary
        counts = DT._packed_dim_counts(dim_offsets)
        nd = length(counts)
        @inbounds for slot in 2:nd
            B = G.boundaries[slot - 1]
            prev_off = dim_offsets[slot - 1] - 1
            cur_off = dim_offsets[slot] - 1
            for j in 1:counts[slot]
                g = grades[cur_off + j]
                for p in nzrange(B, j)
                    i = rowvals(B)[p]
                    g = _tuple_componentwise_max(g, grades[prev_off + i])
                end
                grades[cur_off + j] = g
            end
        end
    end
    return DT._rebuild_graded_complex(G, grades)
end

function _old_simplices_from_complex(G::Union{DT.GradedComplex,DT.MultiCriticalGradedComplex})
    nd = length(G.cells_by_dim)
    counts = [length(c) for c in G.cells_by_dim]
    simplices = Vector{Vector{Vector{Int}}}(undef, nd)
    simplices[1] = [[i] for i in 1:counts[1]]
    for d in 2:nd
        B = G.boundaries[d - 1]
        curr = Vector{Vector{Int}}(undef, counts[d])
        prev = simplices[d - 1]
        @inbounds for j in 1:counts[d]
            lo = B.colptr[j]
            hi = B.colptr[j + 1] - 1
            verts = Int[]
            for p in lo:hi
                append!(verts, prev[B.rowval[p]])
            end
            sort!(verts)
            unique!(verts)
            curr[j] = verts
        end
        simplices[d] = curr
    end
    return simplices
end

function _new_simplices_from_complex(G::Union{DT.GradedComplex,DT.MultiCriticalGradedComplex})
    _, dim_offsets = DT._packed_cell_storage(G)
    nd = length(dim_offsets) - 1
    counts = DT._packed_dim_counts(dim_offsets)
    simplices = Vector{Vector{Vector{Int}}}(undef, nd)
    simplices[1] = [[i] for i in 1:counts[1]]
    for d in 2:nd
        B = G.boundaries[d - 1]
        curr = Vector{Vector{Int}}(undef, counts[d])
        prev = simplices[d - 1]
        @inbounds for j in 1:counts[d]
            lo = B.colptr[j]
            hi = B.colptr[j + 1] - 1
            verts = Int[]
            for p in lo:hi
                append!(verts, prev[B.rowval[p]])
            end
            sort!(verts)
            unique!(verts)
            curr[j] = verts
        end
        simplices[d] = curr
    end
    return simplices
end

function _old_simplex_tree_grade_sets(simplices::Vector{Vector{Vector{Int}}},
                                      grades::AbstractVector{<:NTuple{N,T}}) where {N,T}
    total = sum(length, simplices)
    out = Vector{Vector{NTuple{N,T}}}(undef, total)
    idx = 1
    for d in eachindex(simplices)
        for _ in eachindex(simplices[d])
            out[idx] = NTuple{N,T}[grades[idx]]
            idx += 1
        end
    end
    return out
end

function _old_simplex_tree_grade_sets(simplices::Vector{Vector{Vector{Int}}},
                                      grades::AbstractVector{<:AbstractVector{<:NTuple{N,T}}}) where {N,T}
    total = sum(length, simplices)
    out = Vector{Vector{NTuple{N,T}}}(undef, total)
    @inbounds for i in 1:total
        out[i] = unique(Vector{NTuple{N,T}}(grades[i]))
    end
    return out
end

function _old_simplex_tree_multi_from_simplices(simplices::Vector{Vector{Vector{Int}}},
                                                grades::AbstractVector)
    total = sum(length, simplices)
    grade_sets = _old_simplex_tree_grade_sets(simplices, grades)
    N = length(grade_sets[1][1])
    T = eltype(grade_sets[1][1])
    simplex_offsets = Int[1]
    simplex_vertices_flat = Int[]
    simplex_dims = Int[]
    dim_offsets = Int[1]
    grade_offsets = Int[1]
    grade_data = NTuple{N,T}[]
    gidx = 1
    for d in eachindex(simplices)
        dim = d - 1
        for s in simplices[d]
            ss = sort!(unique(Int[x for x in s]))
            append!(simplex_vertices_flat, ss)
            push!(simplex_offsets, length(simplex_vertices_flat) + 1)
            push!(simplex_dims, dim)
            for g in grade_sets[gidx]
                push!(grade_data, ntuple(i -> T(g[i]), N))
            end
            push!(grade_offsets, length(grade_data) + 1)
            gidx += 1
        end
        push!(dim_offsets, length(simplex_dims) + 1)
    end
    return DT.SimplexTreeMulti(simplex_offsets, simplex_vertices_flat, simplex_dims,
                               dim_offsets, grade_offsets, grade_data)
end

function _new_simplex_tree_multi_from_simplices_packed(simplices::Vector{Vector{Vector{Int}}},
                                                       grade_offsets::AbstractVector{Int},
                                                       grade_data::AbstractVector{<:NTuple{N,T}}) where {N,T}
    total = sum(length, simplices)
    simplex_offsets = Int[1]
    simplex_vertices_flat = Int[]
    simplex_dims = Int[]
    dim_offsets = Int[1]
    sizehint!(simplex_dims, total)
    sizehint!(simplex_offsets, total + 1)
    sizehint!(simplex_vertices_flat, sum(length, Iterators.flatten(simplices)))
    gidx = 1
    for d in eachindex(simplices)
        dim = d - 1
        for s in simplices[d]
            ss = sort!(unique(Int[x for x in s]))
            append!(simplex_vertices_flat, ss)
            push!(simplex_offsets, length(simplex_vertices_flat) + 1)
            push!(simplex_dims, dim)
            gidx += 1
        end
        push!(dim_offsets, length(simplex_dims) + 1)
    end
    return DT.SimplexTreeMulti(simplex_offsets, simplex_vertices_flat, simplex_dims,
                               dim_offsets, copy(grade_offsets), Vector{NTuple{N,T}}(grade_data))
end

_old_simplex_tree_multi_from_complex(G::DT.GradedComplex) =
    _old_simplex_tree_multi_from_simplices(_old_simplices_from_complex(G), G.grades)

_old_simplex_tree_multi_from_complex(G::DT.MultiCriticalGradedComplex) =
    _old_simplex_tree_multi_from_simplices(_old_simplices_from_complex(G), G.grades)

function _new_simplex_tree_multi_from_complex(G::DT.GradedComplex)
    total = length(G.grades)
    return _new_simplex_tree_multi_from_simplices_packed(
        _new_simplices_from_complex(G),
        collect(1:(total + 1)),
        G.grades,
    )
end

function _new_simplex_tree_multi_from_complex(G::DT.MultiCriticalGradedComplex)
    grade_offsets, grade_data = DT._packed_multigrade_storage(G)
    return _new_simplex_tree_multi_from_simplices_packed(_new_simplices_from_complex(G), grade_offsets, grade_data)
end

function _old_simplex_tree_dim_simplices(ST::DT.SimplexTreeMulti, dim_slot::Int)
    lo = ST.dim_offsets[dim_slot]
    hi = ST.dim_offsets[dim_slot + 1] - 1
    out = Vector{Vector{Int}}(undef, max(0, hi - lo + 1))
    p = 1
    for sid in lo:hi
        out[p] = Vector{Int}(DT.simplex_vertices(ST, sid))
        p += 1
    end
    return out
end

function _old_simplicial_boundary_hash(simplices::Vector{Vector{Int}}, faces::Vector{Vector{Int}})
    K = isempty(simplices) ? (isempty(faces) ? 0 : length(faces[1]) + 1) : length(simplices[1])
    K == 0 && return spzeros(Int, length(faces), length(simplices))
    face_index = Dict{Tuple{Vararg{Int}},Int}()
    for (i, f) in enumerate(faces)
        face_index[Tuple(f)] = i
    end
    I = Int[]; J = Int[]; V = Int[]
    sizehint!(I, K * length(simplices))
    sizehint!(J, K * length(simplices))
    sizehint!(V, K * length(simplices))
    for (j, s) in enumerate(simplices)
        for i in 1:K
            f = [s[t] for t in 1:K if t != i]
            row = face_index[Tuple(f)]
            push!(I, row); push!(J, j); push!(V, isodd(i) ? 1 : -1)
        end
    end
    return sparse(I, J, V, length(faces), length(simplices))
end

_old_simplex_tree_is_onecritical(ST::DT.SimplexTreeMulti) =
    all(i -> (ST.grade_offsets[i + 1] - ST.grade_offsets[i]) == 1, 1:DT.simplex_count(ST))

function _old_graded_complex_from_simplex_tree(ST::DT.SimplexTreeMulti{N,T}) where {N,T}
    nd = length(ST.dim_offsets) - 1
    simplices = Vector{Vector{Vector{Int}}}(undef, nd)
    for d in 1:nd
        simplices[d] = _old_simplex_tree_dim_simplices(ST, d)
    end
    boundaries = SparseMatrixCSC{Int,Int}[]
    for d in 2:nd
        push!(boundaries, _old_simplicial_boundary_hash(simplices[d], simplices[d - 1]))
    end
    cells = [collect(1:length(simplices[d])) for d in 1:nd]
    if _old_simplex_tree_is_onecritical(ST)
        grades = Vector{NTuple{N,T}}(undef, DT.simplex_count(ST))
        @inbounds for i in 1:DT.simplex_count(ST)
            grades[i] = ST.grade_data[ST.grade_offsets[i]]
        end
        return DT.GradedComplex(cells, boundaries, grades)
    end
    grades = Vector{Vector{NTuple{N,T}}}(undef, DT.simplex_count(ST))
    @inbounds for i in 1:DT.simplex_count(ST)
        lo = ST.grade_offsets[i]
        hi = ST.grade_offsets[i + 1] - 1
        grades[i] = unique(Vector{NTuple{N,T}}(ST.grade_data[lo:hi]))
    end
    return DT.MultiCriticalGradedComplex(cells, boundaries, grades)
end

function _new_simplex_tree_unique_grade_storage(ST::DT.SimplexTreeMulti{N,T}) where {N,T}
    ns = DT.simplex_count(ST)
    out_offsets = Vector{Int}(undef, ns + 1)
    out_offsets[1] = 1
    out_data = Vector{NTuple{N,T}}()
    sizehint!(out_data, length(ST.grade_data))
    @inbounds for sid in 1:ns
        lo = ST.grade_offsets[sid]
        hi = ST.grade_offsets[sid + 1] - 1
        for k in lo:hi
            g = ST.grade_data[k]
            duplicate = false
            for prev in out_offsets[sid]:length(out_data)
                if out_data[prev] == g
                    duplicate = true
                    break
                end
            end
            duplicate && continue
            push!(out_data, g)
        end
        out_offsets[sid + 1] = length(out_data) + 1
    end
    return out_offsets, out_data
end

function _new_graded_complex_from_simplex_tree(ST::DT.SimplexTreeMulti{N,T}) where {N,T}
    nd = length(ST.dim_offsets) - 1
    simplices = Vector{Vector{Vector{Int}}}(undef, nd)
    for d in 1:nd
        simplices[d] = _old_simplex_tree_dim_simplices(ST, d)
    end
    boundaries = SparseMatrixCSC{Int,Int}[]
    for d in 2:nd
        push!(boundaries, _old_simplicial_boundary_hash(simplices[d], simplices[d - 1]))
    end
    cell_ids = Vector{Int}(undef, DT.simplex_count(ST))
    p = 1
    @inbounds for slot in 1:(length(ST.dim_offsets) - 1)
        count = ST.dim_offsets[slot + 1] - ST.dim_offsets[slot]
        for local_id in 1:count
            cell_ids[p] = local_id
            p += 1
        end
    end
    dim_offsets = copy(ST.dim_offsets)
    if _old_simplex_tree_is_onecritical(ST)
        grades = Vector{NTuple{N,T}}(undef, DT.simplex_count(ST))
        @inbounds for i in 1:DT.simplex_count(ST)
            grades[i] = ST.grade_data[ST.grade_offsets[i]]
        end
        return DT.GradedComplex{N,T}(cell_ids, dim_offsets, boundaries, grades)
    end
    grade_offsets, grade_data = _new_simplex_tree_unique_grade_storage(ST)
    return DT.MultiCriticalGradedComplex{N,T}(cell_ids, dim_offsets, boundaries, grade_offsets, grade_data)
end

function _old_axes_from_grades(grades::Vector{<:NTuple{N,T}}, n::Int) where {N,T}
    seen = [Set{Float64}() for _ in 1:n]
    for g in grades
        for i in 1:n
            push!(seen[i], Float64(g[i]))
        end
    end
    return ntuple(i -> sort!(collect(seen[i])), n)
end

function _old_axes_from_multigrades(grades::Vector{<:AbstractVector{<:NTuple{N,T}}}, n::Int) where {N,T}
    flat = NTuple{N,T}[]
    sizehint!(flat, sum(length, grades))
    for gs in grades
        append!(flat, gs)
    end
    return _old_axes_from_grades(flat, n)
end

function _old_axes_from_simplex_tree(ST::DT.SimplexTreeMulti{N,T}) where {N,T}
    if _old_simplex_tree_is_onecritical(ST)
        grades = Vector{NTuple{N,T}}(undef, DT.simplex_count(ST))
        @inbounds for i in 1:DT.simplex_count(ST)
            grades[i] = ST.grade_data[ST.grade_offsets[i]]
        end
        return _old_axes_from_grades(grades, N)
    end
    grades = Vector{Vector{NTuple{N,T}}}(undef, DT.simplex_count(ST))
    @inbounds for i in 1:DT.simplex_count(ST)
        lo = ST.grade_offsets[i]
        hi = ST.grade_offsets[i + 1] - 1
        grades[i] = Vector{NTuple{N,T}}(ST.grade_data[lo:hi])
    end
    return _old_axes_from_multigrades(grades, N)
end

_new_axes_from_simplex_tree(ST::DT.SimplexTreeMulti{N,T}) where {N,T} =
    _old_axes_from_grades(ST.grade_data, N)

_bench_equal(a, b) = a == b

function _bench_equal_boundaries(A::Vector{SparseMatrixCSC{Int,Int}},
                                 B::Vector{SparseMatrixCSC{Int,Int}})
    length(A) == length(B) || return false
    for i in eachindex(A)
        size(A[i]) == size(B[i]) || return false
        A[i] == B[i] || return false
    end
    return true
end

function _bench_equal(a::DT.SimplexTreeMulti, b::DT.SimplexTreeMulti)
    return a.simplex_offsets == b.simplex_offsets &&
           a.simplex_vertices == b.simplex_vertices &&
           a.simplex_dims == b.simplex_dims &&
           a.dim_offsets == b.dim_offsets &&
           a.grade_offsets == b.grade_offsets &&
           a.grade_data == b.grade_data
end

function _bench_equal(a::DT.GradedComplex, b::DT.GradedComplex)
    return getfield(a, :cell_ids) == getfield(b, :cell_ids) &&
           getfield(a, :dim_offsets) == getfield(b, :dim_offsets) &&
           _bench_equal_boundaries(a.boundaries, b.boundaries) &&
           a.grades == b.grades
end

function _bench_equal(a::DT.MultiCriticalGradedComplex, b::DT.MultiCriticalGradedComplex)
    return getfield(a, :cell_ids) == getfield(b, :cell_ids) &&
           getfield(a, :dim_offsets) == getfield(b, :dim_offsets) &&
           _bench_equal_boundaries(a.boundaries, b.boundaries) &&
           getfield(a, :grade_offsets) == getfield(b, :grade_offsets) &&
           getfield(a, :grade_data) == getfield(b, :grade_data)
end

function _assert_parity(f_before::Function, f_after::Function)
    a = f_before()
    b = f_after()
    _bench_equal(a, b) || error("benchmark parity check failed")
end

function main(args)
    reps = _parse_flag(args, "--reps", 7)
    ntri = _parse_flag(args, "--ntri", 2048)
    out = _parse_flag(args, "--out", joinpath(@__DIR__, "_tmp_data_ingestion_complex_layout_microbench.csv"))

    println("Timing policy: warm, same-process A/B owner-kernel emulation")
    println("reps=$reps, ntri=$ntri")

    fixture = _triangulated_fixture(ntri)
    Gg = fixture.Gg
    Gm = fixture.Gm

    _assert_parity(() -> _old_estimate_cell_counts(Gg), () -> _new_estimate_cell_counts(Gg))
    _assert_parity(() -> _old_estimate_cell_counts(Gm), () -> _new_estimate_cell_counts(Gm))
    _assert_parity(() -> _old_criticality(Gm), () -> _new_criticality(Gm))
    _assert_parity(() -> _old_normalize_multicritical(Gm; keep=:minimal).grades,
                   () -> _new_normalize_multicritical(Gm; keep=:minimal).grades)
    _assert_parity(() -> _old_grades_by_dim(Gm), () -> _new_grades_by_dim(Gm))
    _assert_parity(() -> _old_one_criticalify(Gm).grades, () -> _new_one_criticalify(Gm).grades)
    _assert_parity(() -> _old_simplices_from_complex(Gg), () -> _new_simplices_from_complex(Gg))
    _assert_parity(() -> _old_simplices_from_complex(Gm), () -> _new_simplices_from_complex(Gm))
    _assert_parity(() -> _old_simplex_tree_multi_from_complex(Gg), () -> _new_simplex_tree_multi_from_complex(Gg))
    _assert_parity(() -> _old_simplex_tree_multi_from_complex(Gm), () -> _new_simplex_tree_multi_from_complex(Gm))
    stg = _new_simplex_tree_multi_from_complex(Gg)
    stm = _new_simplex_tree_multi_from_complex(Gm)
    _assert_parity(() -> _old_graded_complex_from_simplex_tree(stg), () -> _new_graded_complex_from_simplex_tree(stg))
    _assert_parity(() -> _old_graded_complex_from_simplex_tree(stm), () -> _new_graded_complex_from_simplex_tree(stm))
    _assert_parity(() -> _old_axes_from_simplex_tree(stg), () -> _new_axes_from_simplex_tree(stg))
    _assert_parity(() -> _old_axes_from_simplex_tree(stm), () -> _new_axes_from_simplex_tree(stm))

    cases = [
        ("estimate_cell_counts_graded", () -> begin
            before = _median_stats(() -> _old_estimate_cell_counts(Gg); reps=reps)
            after = _median_stats(() -> _new_estimate_cell_counts(Gg); reps=reps)
            return before, after
        end),
        ("estimate_cell_counts_multicritical", () -> begin
            before = _median_stats(() -> _old_estimate_cell_counts(Gm); reps=reps)
            after = _median_stats(() -> _new_estimate_cell_counts(Gm); reps=reps)
            return before, after
        end),
        ("criticality_multicritical", () -> begin
            before = _median_stats(() -> _old_criticality(Gm); reps=reps)
            after = _median_stats(() -> _new_criticality(Gm); reps=reps)
            return before, after
        end),
        ("normalize_multicritical", () -> begin
            before = _median_stats(() -> _old_normalize_multicritical(Gm; keep=:minimal); reps=reps)
            after = _median_stats(() -> _new_normalize_multicritical(Gm; keep=:minimal); reps=reps)
            return before, after
        end),
        ("grades_by_dim_multicritical", () -> begin
            before = _median_stats(() -> _old_grades_by_dim(Gm); reps=reps)
            after = _median_stats(() -> _new_grades_by_dim(Gm); reps=reps)
            return before, after
        end),
        ("one_criticalify_multicritical", () -> begin
            before = _median_stats(() -> _old_one_criticalify(Gm); reps=reps)
            after = _median_stats(() -> _new_one_criticalify(Gm); reps=reps)
            return before, after
        end),
        ("simplices_from_complex_graded", () -> begin
            before = _median_stats(() -> _old_simplices_from_complex(Gg); reps=reps)
            after = _median_stats(() -> _new_simplices_from_complex(Gg); reps=reps)
            return before, after
        end),
        ("simplices_from_complex_multicritical", () -> begin
            before = _median_stats(() -> _old_simplices_from_complex(Gm); reps=reps)
            after = _median_stats(() -> _new_simplices_from_complex(Gm); reps=reps)
            return before, after
        end),
        ("simplex_tree_from_complex_graded", () -> begin
            before = _median_stats(() -> _old_simplex_tree_multi_from_complex(Gg); reps=reps)
            after = _median_stats(() -> _new_simplex_tree_multi_from_complex(Gg); reps=reps)
            return before, after
        end),
        ("simplex_tree_from_complex_multicritical", () -> begin
            before = _median_stats(() -> _old_simplex_tree_multi_from_complex(Gm); reps=reps)
            after = _median_stats(() -> _new_simplex_tree_multi_from_complex(Gm); reps=reps)
            return before, after
        end),
        ("graded_complex_from_simplex_tree_multicritical", () -> begin
            before = _median_stats(() -> _old_graded_complex_from_simplex_tree(stm); reps=reps)
            after = _median_stats(() -> _new_graded_complex_from_simplex_tree(stm); reps=reps)
            return before, after
        end),
        ("axes_from_simplex_tree_multicritical", () -> begin
            before = _median_stats(() -> _old_axes_from_simplex_tree(stm); reps=reps)
            after = _median_stats(() -> _new_axes_from_simplex_tree(stm); reps=reps)
            return before, after
        end),
    ]

    rows = String[]
    push!(rows, "case,variant,time_ms,alloc_kib")
    for (label, run_case) in cases
        before, after = run_case()
        push!(rows, @sprintf("%s,before,%.6f,%.3f", label, 1.0e3 * before.time, before.bytes / 1024))
        push!(rows, @sprintf("%s,after,%.6f,%.3f", label, 1.0e3 * after.time, after.bytes / 1024))
        ratio = before.time / max(after.time, eps())
        println(@sprintf("%-36s  %.3f ms -> %.3f ms  (%.2fx),  %.1f KiB -> %.1f KiB",
            label, 1.0e3 * before.time, 1.0e3 * after.time, ratio,
            before.bytes / 1024, after.bytes / 1024))
    end

    open(out, "w") do io
        write(io, join(rows, "\n"))
        write(io, "\n")
    end
    println("wrote ", out)
end

main(ARGS)
