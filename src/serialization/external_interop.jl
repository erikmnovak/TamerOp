# External interop loaders for Gudhi/Ripser/DIPHA-style fixtures and
# boundary/reduced-complex/PModule formats.

# =============================================================================
# B0) Interop adapters: GUDHI / Ripserer / Eirene (JSON)
# =============================================================================

function _simplicial_boundary_from_lists(simplices::Vector{Vector{Int}},
                                         faces::Vector{Vector{Int}})
    face_index = Dict{Tuple{Vararg{Int}},Int}()
    for (i, f) in enumerate(faces)
        face_index[Tuple(f)] = i
    end
    I = Int[]
    J = Int[]
    V = Int[]
    for (j, s) in enumerate(simplices)
        k = length(s)
        for i in 1:k
            f = [s[t] for t in 1:k if t != i]
            row = face_index[Tuple(f)]
            push!(I, row)
            push!(J, j)
            push!(V, isodd(i) ? 1 : -1)
        end
    end
    return sparse(I, J, V, length(faces), length(simplices))
end

function _graded_complex_from_simplex_list(simplices::Vector{Vector{Int}}, grades_any::AbstractVector)
    length(simplices) == length(grades_any) ||
        error("simplices and grades length mismatch.")
    max_dim = maximum(length.(simplices)) - 1
    by_dim = [Vector{Vector{Int}}() for _ in 0:max_dim]
    grades = Vector{Vector{Float64}}()
    for (s, g) in zip(simplices, grades_any)
        d = length(s) - 1
        push!(by_dim[d+1], s)
        if g isa AbstractVector
            push!(grades, Vector{Float64}(g))
        else
            push!(grades, [Float64(g)])
        end
    end
    boundaries = SparseMatrixCSC{Int,Int}[]
    for d in 2:length(by_dim)
        push!(boundaries, _simplicial_boundary_from_lists(by_dim[d], by_dim[d-1]))
    end
    cells = [collect(1:length(by_dim[d])) for d in 1:length(by_dim)]
    return GradedComplex(cells, boundaries, grades)
end

"""
    load_gudhi_json(path) -> GradedComplex

Expected schema:
{
  "simplices": [[0],[1],[0,1],...],
  "filtration": [0.0, 0.0, 1.0, ...]   # or list-of-lists for multiparameter
}
"""
function load_gudhi_json(path::AbstractString)
    obj = _json_read(path)
    simplices = [Vector{Int}(s) for s in obj["simplices"]]
    grades = obj["filtration"]
    return _graded_complex_from_simplex_list(simplices, grades)
end

"""
    load_ripserer_json(path) -> GradedComplex

Expected schema:
{
  "simplices": [[0],[1],[0,1],...],
  "filtration": [0.0, 0.0, 1.0, ...]
}
"""
function load_ripserer_json(path::AbstractString)
    obj = _json_read(path)
    simplices = [Vector{Int}(s) for s in obj["simplices"]]
    grades = obj["filtration"]
    return _graded_complex_from_simplex_list(simplices, grades)
end

"""
    load_eirene_json(path) -> GradedComplex

Expected schema:
{
  "simplices": [[0],[1],[0,1],...],
  "filtration": [0.0, 0.0, 1.0, ...]
}
"""
function load_eirene_json(path::AbstractString)
    obj = _json_read(path)
    simplices = [Vector{Int}(s) for s in obj["simplices"]]
    grades = obj["filtration"]
    return _graded_complex_from_simplex_list(simplices, grades)
end

function _read_structured_lines(path::AbstractString)
    out = String[]
    open(path, "r") do io
        for raw in eachline(io)
            line = strip(raw)
            isempty(line) && continue
            startswith(line, "#") && continue
            startswith(line, "//") && continue
            push!(out, line)
        end
    end
    return out
end

function _take_rivet_flags(lines::Vector{String})
    flags = Dict{String,String}()
    i = 1
    while i <= length(lines) && startswith(lines[i], "--")
        parts = split(lines[i], r"\s+"; limit=2)
        key = lowercase(replace(parts[1], "--" => ""))
        val = length(parts) == 2 ? strip(parts[2]) : "true"
        flags[key] = val
        i += 1
    end
    return flags, lines[i:end]
end

function _minimal_bigrades(grades::Vector{NTuple{2,Float64}})
    u = unique(grades)
    keep = trues(length(u))
    for i in eachindex(u)
        ai = u[i]
        for j in eachindex(u)
            i == j && continue
            aj = u[j]
            if aj[1] <= ai[1] && aj[2] <= ai[2] && (aj != ai)
                keep[i] = false
                break
            end
        end
    end
    out = u[keep]
    sort!(out)
    return out
end

function _parse_rivet_simplex_grade_line(line::AbstractString)
    parts = split(line, ';'; limit=2)
    length(parts) == 2 || error("RIVET bifiltration line must contain ';': $(line)")
    simplex = Int[parse(Int, t) for t in split(strip(parts[1]))]
    isempty(simplex) && error("RIVET bifiltration simplex cannot be empty.")
    sort!(unique!(simplex))

    gtoks = split(replace(strip(parts[2]), "," => " "))
    isempty(gtoks) && error("RIVET bifiltration line has no grades: $(line)")
    iseven(length(gtoks)) || error("RIVET bifiltration grade list must have even length: $(line)")
    grades = NTuple{2,Float64}[]
    for i in 1:2:length(gtoks)
        push!(grades, (parse(Float64, gtoks[i]), parse(Float64, gtoks[i+1])))
    end
    return simplex, _minimal_bigrades(grades)
end

function _normalize_simplex_indices!(simplices::Vector{Vector{Int}})
    minv = minimum(minimum(s) for s in simplices)
    if minv == 0
        for s in simplices
            for i in eachindex(s)
                s[i] += 1
            end
        end
    elseif minv < 1
        error("RIVET simplices must be 0-based or 1-based integer indices.")
    end
    return simplices
end

function _graded_complex_from_simplex_list_multicritical(simplices::Vector{Vector{Int}},
                                                         gradesets::Vector{Vector{NTuple{2,Float64}}})
    length(simplices) == length(gradesets) || error("simplices and grade sets length mismatch.")
    max_dim = maximum(length.(simplices)) - 1
    by_dim = [Vector{Vector{Int}}() for _ in 0:max_dim]
    g_by_dim = [Vector{Vector{NTuple{2,Float64}}}() for _ in 0:max_dim]
    for (s, gs) in zip(simplices, gradesets)
        d = length(s) - 1
        push!(by_dim[d+1], s)
        push!(g_by_dim[d+1], gs)
    end
    boundaries = SparseMatrixCSC{Int,Int}[]
    for d in 2:length(by_dim)
        push!(boundaries, _simplicial_boundary_from_lists(by_dim[d], by_dim[d-1]))
    end
    cells = [collect(1:length(by_dim[d])) for d in 1:length(by_dim)]
    flat_multi = Vector{Vector{NTuple{2,Float64}}}()
    for d in 1:length(g_by_dim)
        append!(flat_multi, g_by_dim[d])
    end
    if all(length(gs) == 1 for gs in flat_multi)
        flat = [gs[1] for gs in flat_multi]
        return GradedComplex(cells, boundaries, flat)
    end
    return MultiCriticalGradedComplex(cells, boundaries, flat_multi)
end

"""
    load_rivet_bifiltration(path) -> Union{GradedComplex, MultiCriticalGradedComplex}

Parse a canonical RIVET bifiltration text file with
`--datatype bifiltration` and payload lines `simplex ; x y [x y ...]`.
"""
function load_rivet_bifiltration(path::AbstractString)
    raw = _read_structured_lines(path)
    isempty(raw) && error("RIVET bifiltration: empty file.")
    flags, lines = _take_rivet_flags(raw)
    haskey(flags, "datatype") || error("RIVET bifiltration: missing canonical `--datatype bifiltration` header.")
    lowercase(flags["datatype"]) == "bifiltration" ||
        error("RIVET loader expected --datatype bifiltration, got $(flags["datatype"]).")
    isempty(lines) && error("RIVET bifiltration: no payload lines.")
    simplices = Vector{Vector{Int}}(undef, length(lines))
    gradesets = Vector{Vector{NTuple{2,Float64}}}(undef, length(lines))
    for i in eachindex(lines)
        s, gs = _parse_rivet_simplex_grade_line(lines[i])
        simplices[i] = s
        gradesets[i] = gs
    end
    _normalize_simplex_indices!(simplices)
    return _graded_complex_from_simplex_list_multicritical(simplices, gradesets)
end

function _parse_rivet_firep_column(line::AbstractString)
    parts = split(line, ';'; limit=2)
    length(parts) == 2 || error("RIVET FIRep column line must contain ';': $(line)")
    gtok = split(strip(parts[1]))
    length(gtok) == 2 || error("RIVET FIRep column grade must have exactly two coordinates: $(line)")
    grade = (parse(Float64, gtok[1]), parse(Float64, gtok[2]))
    rhs = strip(parts[2])
    idxs = isempty(rhs) ? Int[] : Int[parse(Int, t) for t in split(rhs)]
    return grade, idxs
end

"""
    load_rivet_firep(path) -> GradedComplex

Parse a RIVET FIRep text file (`--datatype firep` or raw FIRep payload).
Builds a graded complex with dimensions 0,1,2 from the FIRep matrices.
"""
function load_rivet_firep(path::AbstractString)
    raw = _read_structured_lines(path)
    isempty(raw) && error("RIVET FIRep: empty file.")
    flags, lines = _take_rivet_flags(raw)
    if haskey(flags, "datatype")
        lowercase(flags["datatype"]) == "firep" ||
            error("RIVET loader expected --datatype firep, got $(flags["datatype"]).")
    end
    isempty(lines) && error("RIVET FIRep: missing payload.")

    hdr = split(lines[1])
    length(hdr) == 3 || error("RIVET FIRep header must be: t s r")
    t = parse(Int, hdr[1])  # C2 generators
    s = parse(Int, hdr[2])  # C1 generators
    r = parse(Int, hdr[3])  # C0 generators
    t >= 0 && s >= 0 && r >= 0 || error("RIVET FIRep counts must be nonnegative.")
    length(lines) == 1 + t + s || error("RIVET FIRep: expected $(1+t+s) payload lines, found $(length(lines)).")

    c2_grades = Vector{NTuple{2,Float64}}(undef, t)
    I2 = Int[]; J2 = Int[]
    for j in 1:t
        g, rows = _parse_rivet_firep_column(lines[1 + j])
        c2_grades[j] = g
        for i in rows
            push!(I2, i)
            push!(J2, j)
        end
    end
    if !isempty(I2)
        minimum(I2) == 0 && (I2 .= I2 .+ 1)
        minimum(I2) >= 1 || error("RIVET FIRep: invalid C2->C1 row index.")
        maximum(I2) <= s || error("RIVET FIRep: C2->C1 row index out of range.")
    end

    c1_grades = Vector{NTuple{2,Float64}}(undef, s)
    I1 = Int[]; J1 = Int[]
    for j in 1:s
        g, rows = _parse_rivet_firep_column(lines[1 + t + j])
        c1_grades[j] = g
        for i in rows
            push!(I1, i)
            push!(J1, j)
        end
    end
    if !isempty(I1)
        minimum(I1) == 0 && (I1 .= I1 .+ 1)
        minimum(I1) >= 1 || error("RIVET FIRep: invalid C1->C0 row index.")
        maximum(I1) <= r || error("RIVET FIRep: C1->C0 row index out of range.")
    end

    B1 = sparse(I1, J1, ones(Int, length(I1)), r, s)  # C1 -> C0
    B2 = sparse(I2, J2, ones(Int, length(I2)), s, t)  # C2 -> C1

    c0_grades = Vector{NTuple{2,Float64}}(undef, r)
    incident = [Int[] for _ in 1:r]
    Ii, Jj, _ = findnz(B1)
    @inbounds for k in eachindex(Ii)
        push!(incident[Ii[k]], Jj[k])
    end
    for i in 1:r
        if isempty(incident[i])
            c0_grades[i] = (0.0, 0.0)
        else
            xs = Float64[c1_grades[j][1] for j in incident[i]]
            ys = Float64[c1_grades[j][2] for j in incident[i]]
            c0_grades[i] = (minimum(xs), minimum(ys))
        end
    end

    cells = [collect(1:r), collect(1:s), collect(1:t)]
    grades = vcat(c0_grades, c1_grades, c2_grades)
    return GradedComplex(cells, [B1, B2], grades)
end

"""
    load_gudhi_txt(path) -> GradedComplex
    load_ripserer_txt(path) -> GradedComplex
    load_eirene_txt(path) -> GradedComplex

Parse a simplex filtration from a text file with one simplex per line.

Supported line formats (whitespace-separated):
1) "dim v1 v2 ... vk filtration"
2) "v1 v2 ... vk filtration"   (dimension inferred from count)

Blank lines and lines starting with '#' are ignored.
"""
function _load_simplex_filtration_txt(path::AbstractString)
    lines = String[]
    open(path, "r") do io
        for raw in eachline(io)
            line = strip(raw)
            isempty(line) && continue
            startswith(line, "#") && continue
            push!(lines, line)
        end
    end

    # Heuristic: if any line has exactly two tokens, treat the file as
    # "vertices + filtration" (no leading dimension token). Otherwise, use
    # the "dim v1 v2 ... filtration" format.
    has_dim_prefix = true
    for line in lines
        parts = split(line)
        length(parts) >= 2 || error("Invalid simplex line: '$line'")
        if length(parts) == 2
            has_dim_prefix = false
            break
        end
    end

    simplices = Vector{Vector{Int}}()
    grades = Vector{Float64}()
    for line in lines
        parts = split(line)
        if has_dim_prefix
            dim = parse(Int, parts[1])
            verts = [parse(Int, parts[i]) for i in 2:(dim+2)]
            filt = parse(Float64, parts[dim+3])
            push!(simplices, verts)
            push!(grades, filt)
        else
            verts = [parse(Int, parts[i]) for i in 1:(length(parts)-1)]
            filt = parse(Float64, parts[end])
            push!(simplices, verts)
            push!(grades, filt)
        end
    end

    return _graded_complex_from_simplex_list(simplices, grades)
end

load_gudhi_txt(path::AbstractString) = _load_simplex_filtration_txt(path)
load_ripserer_txt(path::AbstractString) = _load_simplex_filtration_txt(path)
load_eirene_txt(path::AbstractString) = _load_simplex_filtration_txt(path)

# -----------------------------------------------------------------------------
# B2) Interop adapters: boundary/reduced complexes and direct PModules
# -----------------------------------------------------------------------------

@inline _packed_signature_rows_type(nw::Int) = Core.apply_type(PackedSignatureRows, nw)

@inline function _pack_words_obj_from_matrix(words::Matrix{UInt64}, bitlen::Int)
    nw, nrows = size(words)
    ncols = bitlen
    flat = Vector{UInt64}(undef, nrows * nw)
    @inbounds for i in 1:nrows
        base = (i - 1) * nw
        for w in 1:nw
            flat[base + w] = words[w, i]
        end
        if ncols > 0
            flat[base + nw] &= _mask_lastword(ncols)
        end
    end
    return _MaskPackedWordsJSON(kind="packed_words_v1",
                                nrows=nrows,
                                ncols=ncols,
                                words_per_row=nw,
                                words=flat)
end

@inline function _pack_signature_rows_obj(rows::PackedSignatureRows)
    return _pack_words_obj_from_matrix(rows.words, rows.bitlen)
end

@inline function _pack_bitmatrix_obj(L::BitMatrix)
    return _pack_words_obj_from_matrix(_pack_bitmatrix_words(L), size(L, 2))
end

function _pack_bitmatrix_words(L::BitMatrix)
    nrows, ncols = size(L)
    nw = max(1, cld(max(ncols, 1), 64))
    words = zeros(UInt64, nw, nrows)
    @inbounds for i in 1:nrows
        for j in 1:ncols
            if L[i, j]
                w = ((j - 1) >>> 6) + 1
                words[w, i] |= (UInt64(1) << ((j - 1) & 63))
            end
        end
        if ncols > 0
            words[nw, i] &= _mask_lastword(ncols)
        end
    end
    return words
end

function _unpack_words_matrix(mask_obj::_MaskPackedWordsJSON, name::String;
                              nrows_expected::Union{Nothing,Int}=nothing,
                              ncols_expected::Union{Nothing,Int}=nothing)
    nrows = mask_obj.nrows
    ncols = mask_obj.ncols
    nw = mask_obj.words_per_row
    nrows_expected === nothing || (nrows == nrows_expected || error("$(name).nrows must equal $(nrows_expected)."))
    ncols_expected === nothing || (ncols == ncols_expected || error("$(name).ncols must equal $(ncols_expected)."))
    nw == max(1, cld(max(ncols, 1), 64)) || error("$(name).words_per_row is inconsistent with ncols.")
    flat = mask_obj.words
    length(flat) == nrows * nw || error("$(name).words length mismatch.")
    words = Matrix{UInt64}(undef, nw, nrows)
    @inbounds for i in 1:nrows
        base = (i - 1) * nw
        for w in 1:nw
            words[w, i] = flat[base + w]
        end
        if ncols > 0
            words[nw, i] &= _mask_lastword(ncols)
        end
    end
    return words, nrows, ncols
end

function _parse_signature_rows_packed(mask_obj::_MaskPackedWordsJSON, row_name::String)
    words, _, bitlen = _unpack_words_matrix(mask_obj, row_name)
    return _packed_signature_rows_type(size(words, 1))(words, bitlen)
end

function _parse_signature_rows_packed(rows_any, row_name::String)
    rows_any isa AbstractVector || error("$(row_name) must be a list-of-lists.")
    nrows = length(rows_any)
    bitlen = if nrows == 0
        0
    else
        first_row = rows_any[1]
        first_row isa AbstractVector || error("$(row_name) row 1 must be a list.")
        length(first_row)
    end
    nw = max(1, cld(max(bitlen, 1), 64))
    words = zeros(UInt64, nw, nrows)
    @inbounds for i in 1:nrows
        row = rows_any[i]
        row isa AbstractVector || error("$(row_name) row $(i) must be a list.")
        length(row) == bitlen || error("$(row_name) row $(i) length mismatch; expected $(bitlen).")
        for j in 1:bitlen
            x = row[j]
            x isa Bool || error("$(row_name) entries must be Bool (row $(i), col $(j)).")
            if x
                w = ((j - 1) >>> 6) + 1
                words[w, i] |= (UInt64(1) << ((j - 1) & 63))
            end
        end
    end
    return _packed_signature_rows_type(nw)(words, bitlen)
end

function _decode_bitmatrix(mask_obj::_MaskPackedWordsJSON, name::String,
                           nrows_expected::Int, ncols_expected::Int)
    words, nrows, ncols = _unpack_words_matrix(mask_obj, name;
                                               nrows_expected=nrows_expected,
                                               ncols_expected=ncols_expected)
    L = falses(nrows, ncols)
    @inbounds for i in 1:nrows
        for j in 1:ncols
            w = ((j - 1) >>> 6) + 1
            bit = UInt64(1) << ((j - 1) & 63)
            L[i, j] = (words[w, i] & bit) != 0
        end
    end
    return L
end

@inline function _is_packed_words_obj(obj)::Bool
    return !(obj isa AbstractVector) && haskey(obj, "kind") && String(obj["kind"]) == "packed_words_v1"
end

function _parse_poset_from_typed(poset_obj::_FinitePosetJSON)
    n = poset_obj.n
    leq = _decode_bitmatrix(poset_obj.leq, "FinitePoset.leq", n, n)
    P = FinitePoset(leq)
    _clear_cover_cache!(P)
    return P
end

function _parse_poset_from_typed(poset_obj::_ProductOfChainsPosetJSON)
    P = ProductOfChainsPoset(poset_obj.sizes)
    poset_obj.n == nvertices(P) || error("ProductOfChainsPoset.n mismatch.")
    _clear_cover_cache!(P)
    return P
end

function _parse_poset_from_typed(poset_obj::_GridPosetJSON)
    coords_any = poset_obj.coords
    coords = ntuple(i -> Vector{Float64}(coords_any[i]), length(coords_any))
    P = GridPoset(coords)
    poset_obj.n == nvertices(P) || error("GridPoset.n mismatch.")
    _clear_cover_cache!(P)
    return P
end

function _parse_poset_from_typed(poset_obj::_ProductPosetJSON)
    poset_obj.left === nothing && error("ProductPoset missing required key 'left'.")
    poset_obj.right === nothing && error("ProductPoset missing required key 'right'.")
    P1 = _parse_poset_from_typed(poset_obj.left)
    P2 = _parse_poset_from_typed(poset_obj.right)
    P = ProductPoset(P1, P2)
    poset_obj.n == nvertices(P) || error("ProductPoset.n mismatch.")
    _clear_cover_cache!(P)
    return P
end

function _parse_poset_from_typed(poset_obj::_SignaturePosetJSON)
    sig_y = _parse_signature_rows_packed(poset_obj.sig_y, "SignaturePoset.sig_y")
    sig_z = _parse_signature_rows_packed(poset_obj.sig_z, "SignaturePoset.sig_z")
    P = SignaturePoset(sig_y, sig_z)
    poset_obj.n == nvertices(P) || error("SignaturePoset.n mismatch.")
    _clear_cover_cache!(P)
    return P
end

function _parse_poset_from_typed(poset_obj::_PosetJSON)
    error("Unsupported typed poset payload: $(typeof(poset_obj))")
end

@inline function _pack_masks_obj(masks::AbstractVector{<:BitVector})
    nrows = length(masks)
    ncols = nrows == 0 ? 0 : length(masks[1])
    nw = max(1, cld(max(ncols, 1), 64))
    words = zeros(UInt64, nrows * nw)
    @inbounds for i in 1:nrows
        mask = masks[i]
        length(mask) == ncols || error("mask row length mismatch at row $(i).")
        base = (i - 1) * nw
        chunks = mask.chunks
        nchunks = length(chunks)
        for w in 1:min(nchunks, nw)
            words[base + w] = chunks[w]
        end
        if ncols > 0
            words[base + nw] &= _mask_lastword(ncols)
        end
    end
    return _MaskPackedWordsJSON(kind="packed_words_v1",
                                nrows=nrows,
                                ncols=ncols,
                                words_per_row=nw,
                                words=words)
end

@inline function _decode_masks_from_words(words::Matrix{UInt64}, nrows::Int, ncols::Int)
    nw = size(words, 1)
    masks = Vector{BitVector}(undef, nrows)
    if ncols == 0
        @inbounds for i in 1:nrows
            masks[i] = BitVector(undef, 0)
        end
        return masks
    end
    lastmask = _mask_lastword(ncols)
    nchunks = cld(ncols, 64)
    @inbounds for i in 1:nrows
        mask = falses(ncols)
        for w in 1:nchunks
            mask.chunks[w] = words[w, i]
        end
        mask.chunks[nchunks] &= lastmask
        masks[i] = mask
    end
    return masks
end

function _decode_masks(mask_obj::_MaskPackedWordsJSON, name::String, ncols_expected::Int)
    words, nrows, ncols = _unpack_words_matrix(mask_obj, name; ncols_expected=ncols_expected)
    return _decode_masks_from_words(words, nrows, ncols)
end

@inline function _packed_mask_meta(mask_obj::_MaskPackedWordsJSON,
                                   name::String;
                                   nrows_expected::Union{Nothing,Int}=nothing,
                                   ncols_expected::Union{Nothing,Int}=nothing)
    nrows = mask_obj.nrows
    ncols = mask_obj.ncols
    nw = mask_obj.words_per_row
    nrows_expected === nothing || (nrows == nrows_expected || error("$(name).nrows must equal $(nrows_expected)."))
    ncols_expected === nothing || (ncols == ncols_expected || error("$(name).ncols must equal $(ncols_expected)."))
    nw == max(1, cld(max(ncols, 1), 64)) || error("$(name).words_per_row is inconsistent with ncols.")
    flat = mask_obj.words
    length(flat) == nrows * nw || error("$(name).words length mismatch.")
    lastmask = ncols > 0 ? _mask_lastword(ncols) : UInt64(0)
    return flat, nrows, ncols, nw, lastmask
end

struct _PackedCoverValidationPlan
    src_word::Vector{Int}
    dst_word::Vector{Int}
    src_bit::Vector{UInt64}
    dst_bit::Vector{UInt64}
end

function _packed_cover_validation_plan(P::AbstractPoset)
    covers = collect(FiniteFringe.cover_edges(P))
    ncover = length(covers)
    src_word = Vector{Int}(undef, ncover)
    dst_word = Vector{Int}(undef, ncover)
    src_bit = Vector{UInt64}(undef, ncover)
    dst_bit = Vector{UInt64}(undef, ncover)
    @inbounds for i in eachindex(covers)
        src, dst = covers[i]
        src_word[i] = ((src - 1) >>> 6) + 1
        dst_word[i] = ((dst - 1) >>> 6) + 1
        src_bit[i] = UInt64(1) << ((src - 1) & 63)
        dst_bit[i] = UInt64(1) << ((dst - 1) & 63)
    end
    return _PackedCoverValidationPlan(src_word, dst_word, src_bit, dst_bit)
end

function _validate_upset_words(plan::_PackedCoverValidationPlan,
                               flat::Vector{UInt64},
                               nrows::Int,
                               nw::Int,
                               name::String)
    @inbounds for row in 1:nrows
        base = (row - 1) * nw
        for i in eachindex(plan.src_word)
            (flat[base + plan.src_word[i]] & plan.src_bit[i]) == 0 && continue
            (flat[base + plan.dst_word[i]] & plan.dst_bit[i]) != 0 && continue
            error("$(name)[$(row)] is not an upset under strict validation. If this file was produced by TamerOp and you trust it, load with validation=:trusted.")
        end
    end
    return nothing
end

function _validate_downset_words(plan::_PackedCoverValidationPlan,
                                 flat::Vector{UInt64},
                                 nrows::Int,
                                 nw::Int,
                                 name::String)
    @inbounds for row in 1:nrows
        base = (row - 1) * nw
        for i in eachindex(plan.src_word)
            (flat[base + plan.dst_word[i]] & plan.dst_bit[i]) == 0 && continue
            (flat[base + plan.src_word[i]] & plan.src_bit[i]) != 0 && continue
            error("$(name)[$(row)] is not a downset under strict validation. If this file was produced by TamerOp and you trust it, load with validation=:trusted.")
        end
    end
    return nothing
end

@inline function _mask_from_word_row(flat::Vector{UInt64},
                                     row::Int,
                                     ncols::Int,
                                     nw::Int,
                                     lastmask::UInt64)
    mask = falses(ncols)
    nchunks = cld(ncols, 64)
    base = (row - 1) * nw
    @inbounds for w in 1:nchunks
        mask.chunks[w] = flat[base + w]
    end
    if nchunks > 0
        mask.chunks[nchunks] &= lastmask
    end
    return mask
end

function _decode_upsets_from_words(P::AbstractPoset,
                                   flat::Vector{UInt64},
                                   nrows::Int,
                                   ncols::Int,
                                   nw::Int,
                                   lastmask::UInt64)
    U = Vector{FiniteFringe.Upset}(undef, nrows)
    @inbounds for row in 1:nrows
        U[row] = FiniteFringe.Upset(P, _mask_from_word_row(flat, row, ncols, nw, lastmask))
    end
    return U
end

function _decode_downsets_from_words(P::AbstractPoset,
                                     flat::Vector{UInt64},
                                     nrows::Int,
                                     ncols::Int,
                                     nw::Int,
                                     lastmask::UInt64)
    D = Vector{FiniteFringe.Downset}(undef, nrows)
    @inbounds for row in 1:nrows
        D[row] = FiniteFringe.Downset(P, _mask_from_word_row(flat, row, ncols, nw, lastmask))
    end
    return D
end

function _decode_upsets(mask_obj::_MaskPackedWordsJSON, P::AbstractPoset, name::String, ncols_expected::Int)
    flat, nrows, ncols, nw, lastmask = _packed_mask_meta(mask_obj, name; ncols_expected=ncols_expected)
    return _decode_upsets_from_words(P, flat, nrows, ncols, nw, lastmask)
end

function _decode_downsets(mask_obj::_MaskPackedWordsJSON, P::AbstractPoset, name::String, ncols_expected::Int)
    flat, nrows, ncols, nw, lastmask = _packed_mask_meta(mask_obj, name; ncols_expected=ncols_expected)
    return _decode_downsets_from_words(P, flat, nrows, ncols, nw, lastmask)
end

function _validated_upsets(mask_obj::_MaskPackedWordsJSON,
                           P::AbstractPoset,
                           plan::_PackedCoverValidationPlan,
                           name::String,
                           ncols_expected::Int)
    flat, nrows, ncols, nw, lastmask = _packed_mask_meta(mask_obj, name; ncols_expected=ncols_expected)
    _validate_upset_words(plan, flat, nrows, nw, name)
    return _decode_upsets_from_words(P, flat, nrows, ncols, nw, lastmask)
end

function _validated_downsets(mask_obj::_MaskPackedWordsJSON,
                             P::AbstractPoset,
                             plan::_PackedCoverValidationPlan,
                             name::String,
                             ncols_expected::Int)
    flat, nrows, ncols, nw, lastmask = _packed_mask_meta(mask_obj, name; ncols_expected=ncols_expected)
    _validate_downset_words(plan, flat, nrows, nw, name)
    return _decode_downsets_from_words(P, flat, nrows, ncols, nw, lastmask)
end

@inline function _build_upsets(P::AbstractPoset,
                               masks::Vector{BitVector},
                               validate_masks::Bool)
    U = Vector{FiniteFringe.Upset}(undef, length(masks))
    @inbounds for t in eachindex(masks)
        mask = masks[t]
        if validate_masks
            Uc = FiniteFringe.upset_closure(P, mask)
            Uc.mask == mask || error("U[$(t)] is not an upset under strict validation. If this file was produced by TamerOp and you trust it, load with validation=:trusted.")
            U[t] = Uc
        else
            U[t] = FiniteFringe.Upset(P, mask)
        end
    end
    return U
end

@inline function _build_downsets(P::AbstractPoset,
                                 masks::Vector{BitVector},
                                 validate_masks::Bool)
    D = Vector{FiniteFringe.Downset}(undef, length(masks))
    @inbounds for t in eachindex(masks)
        mask = masks[t]
        if validate_masks
            Dc = FiniteFringe.downset_closure(P, mask)
            Dc.mask == mask || error("D[$(t)] is not a downset under strict validation. If this file was produced by TamerOp and you trust it, load with validation=:trusted.")
            D[t] = Dc
        else
            D[t] = FiniteFringe.Downset(P, mask)
        end
    end
    return D
end

@inline function _phi_dims(phi::_PhiJSON)
    return phi.m, phi.k
end

const _QQ_CHUNK_BASE = 1_000_000_000
const _QQ_CHUNK_BASE_BIG = BigInt(_QQ_CHUNK_BASE)

@inline function _bigint_to_chunks!(chunks::Vector{UInt32}, ptr::Vector{Int}, x::BigInt)
    y = abs(x)
    if y == 0
        push!(chunks, UInt32(0))
        push!(ptr, length(chunks) + 1)
        return
    end
    while y != 0
        y, r = divrem(y, _QQ_CHUNK_BASE)
        push!(chunks, UInt32(r))
    end
    push!(ptr, length(chunks) + 1)
end

@inline function _chunks_to_bigint(chunks::Vector{UInt32}, ptr::Vector{Int}, idx::Int, name::String)
    a = ptr[idx]
    b = ptr[idx + 1] - 1
    (a >= 1 && b >= a && b <= length(chunks)) || error("$(name) chunk pointer out of range at index $(idx).")
    x = BigInt(0)
    @inbounds for t in b:-1:a
        x *= _QQ_CHUNK_BASE_BIG
        x += Int(chunks[t])
    end
    return x
end

function _phi_obj(H::FringeModule)
    m, k = size(H.phi)
    if H.field isa QQField
        len = m * k
        num_sign = Vector{Int8}(undef, len)
        num_ptr = Int[1]
        den_ptr = Int[1]
        num_chunks = UInt32[]
        den_chunks = UInt32[]
        sizehint!(num_ptr, len + 1)
        sizehint!(den_ptr, len + 1)
        @inbounds for idx in 1:len
            q = QQ(H.phi[idx])
            num = numerator(q)
            den = denominator(q)
            num_sign[idx] = num > 0 ? Int8(1) : (num < 0 ? Int8(-1) : Int8(0))
            _bigint_to_chunks!(num_chunks, num_ptr, num)
            _bigint_to_chunks!(den_chunks, den_ptr, den)
        end
        return _PhiQQChunksJSON(kind="qq_chunks_v1",
                                m=m,
                                k=k,
                                base=_QQ_CHUNK_BASE,
                                num_sign=num_sign,
                                num_ptr=num_ptr,
                                num_chunks=num_chunks,
                                den_ptr=den_ptr,
                                den_chunks=den_chunks)
    elseif H.field isa PrimeField
        data = Vector{Int}(undef, m * k)
        @inbounds for idx in 1:(m * k)
            data[idx] = Int(coerce(H.field, H.phi[idx]).val)
        end
        return _PhiFpFlatJSON(kind="fp_flat_v1", m=m, k=k, data=data)
    elseif H.field isa RealField
        data = Vector{Float64}(undef, m * k)
        @inbounds for idx in 1:(m * k)
            data[idx] = Float64(H.phi[idx])
        end
        return _PhiRealFlatJSON(kind="real_flat_v1", m=m, k=k, data=data)
    end
    error("Unsupported coefficient field for phi serialization: $(typeof(H.field))")
end

function _decode_phi(phi_obj::_PhiQQChunksJSON,
                     saved_field::QQField,
                     target_field::QQField,
                     m_expected::Int,
                     k_expected::Int)
    m, k = _phi_dims(phi_obj)
    (m == m_expected && k == k_expected) || error("phi dimensions mismatch (expected $(m_expected)x$(k_expected), got $(m)x$(k)).")
    phi_obj.base == _QQ_CHUNK_BASE || error("qq_chunks_v1.base must be $(_QQ_CHUNK_BASE).")
    len = m * k
    length(phi_obj.num_sign) == len || error("qq_chunks_v1.num_sign length mismatch.")
    length(phi_obj.num_ptr) == len + 1 || error("qq_chunks_v1.num_ptr length mismatch.")
    length(phi_obj.den_ptr) == len + 1 || error("qq_chunks_v1.den_ptr length mismatch.")
    phi_obj.num_ptr[1] == 1 || error("qq_chunks_v1.num_ptr must start at 1.")
    phi_obj.den_ptr[1] == 1 || error("qq_chunks_v1.den_ptr must start at 1.")
    phi_obj.num_ptr[end] == length(phi_obj.num_chunks) + 1 || error("qq_chunks_v1.num_ptr terminator mismatch.")
    phi_obj.den_ptr[end] == length(phi_obj.den_chunks) + 1 || error("qq_chunks_v1.den_ptr terminator mismatch.")
    K = coeff_type(target_field)
    Phi = Matrix{K}(undef, m, k)
    zero_q = zero(QQ)
    @inbounds for idx in 1:len
        s = phi_obj.num_sign[idx]
        (s == -1 || s == 0 || s == 1) || error("qq_chunks_v1.num_sign must be in {-1,0,1}.")
        den = _chunks_to_bigint(phi_obj.den_chunks, phi_obj.den_ptr, idx, "qq_chunks_v1.den")
        den == 0 && error("qq_chunks_v1.den must be nonzero.")
        if s == 0
            Phi[idx] = zero_q
            continue
        end
        num = _chunks_to_bigint(phi_obj.num_chunks, phi_obj.num_ptr, idx, "qq_chunks_v1.num")
        Phi[idx] = (s < 0 ? -num : num) // den
    end
    return Phi
end

function _decode_phi(phi_obj::_PhiQQChunksJSON,
                     saved_field::QQField,
                     target_field::AbstractCoeffField,
                     m_expected::Int,
                     k_expected::Int)
    m, k = _phi_dims(phi_obj)
    (m == m_expected && k == k_expected) || error("phi dimensions mismatch (expected $(m_expected)x$(k_expected), got $(m)x$(k)).")
    phi_obj.base == _QQ_CHUNK_BASE || error("qq_chunks_v1.base must be $(_QQ_CHUNK_BASE).")
    len = m * k
    length(phi_obj.num_sign) == len || error("qq_chunks_v1.num_sign length mismatch.")
    length(phi_obj.num_ptr) == len + 1 || error("qq_chunks_v1.num_ptr length mismatch.")
    length(phi_obj.den_ptr) == len + 1 || error("qq_chunks_v1.den_ptr length mismatch.")
    phi_obj.num_ptr[1] == 1 || error("qq_chunks_v1.num_ptr must start at 1.")
    phi_obj.den_ptr[1] == 1 || error("qq_chunks_v1.den_ptr must start at 1.")
    phi_obj.num_ptr[end] == length(phi_obj.num_chunks) + 1 || error("qq_chunks_v1.num_ptr terminator mismatch.")
    phi_obj.den_ptr[end] == length(phi_obj.den_chunks) + 1 || error("qq_chunks_v1.den_ptr terminator mismatch.")
    K = coeff_type(target_field)
    Phi = Matrix{K}(undef, m, k)
    zero_q = zero(QQ)
    @inbounds for idx in 1:len
        s = phi_obj.num_sign[idx]
        (s == -1 || s == 0 || s == 1) || error("qq_chunks_v1.num_sign must be in {-1,0,1}.")
        den = _chunks_to_bigint(phi_obj.den_chunks, phi_obj.den_ptr, idx, "qq_chunks_v1.den")
        den == 0 && error("qq_chunks_v1.den must be nonzero.")
        if s == 0
            Phi[idx] = coerce(target_field, zero_q)
            continue
        end
        num = _chunks_to_bigint(phi_obj.num_chunks, phi_obj.num_ptr, idx, "qq_chunks_v1.num")
        Phi[idx] = coerce(target_field, (s < 0 ? -num : num) // den)
    end
    return Phi
end

function _decode_phi(phi_obj::_PhiFpFlatJSON,
                     saved_field::PrimeField,
                     target_field::PrimeField,
                     m_expected::Int,
                     k_expected::Int)
    m, k = _phi_dims(phi_obj)
    (m == m_expected && k == k_expected) || error("phi dimensions mismatch (expected $(m_expected)x$(k_expected), got $(m)x$(k)).")
    len = m * k
    length(phi_obj.data) == len || error("fp_flat_v1.data length mismatch.")
    K = coeff_type(target_field)
    Phi = Matrix{K}(undef, m, k)
    @inbounds for idx in 1:len
        Phi[idx] = coerce(target_field, phi_obj.data[idx])
    end
    return Phi
end

function _decode_phi(phi_obj::_PhiFpFlatJSON,
                     saved_field::PrimeField,
                     target_field::AbstractCoeffField,
                     m_expected::Int,
                     k_expected::Int)
    m, k = _phi_dims(phi_obj)
    (m == m_expected && k == k_expected) || error("phi dimensions mismatch (expected $(m_expected)x$(k_expected), got $(m)x$(k)).")
    len = m * k
    length(phi_obj.data) == len || error("fp_flat_v1.data length mismatch.")
    K = coeff_type(target_field)
    Phi = Matrix{K}(undef, m, k)
    @inbounds for idx in 1:len
        v = coerce(saved_field, phi_obj.data[idx])
        Phi[idx] = coerce(target_field, v)
    end
    return Phi
end

function _decode_phi(phi_obj::_PhiRealFlatJSON,
                     saved_field::RealField,
                     target_field::RealField,
                     m_expected::Int,
                     k_expected::Int)
    m, k = _phi_dims(phi_obj)
    (m == m_expected && k == k_expected) || error("phi dimensions mismatch (expected $(m_expected)x$(k_expected), got $(m)x$(k)).")
    len = m * k
    length(phi_obj.data) == len || error("real_flat_v1.data length mismatch.")
    K = coeff_type(target_field)
    Phi = Matrix{K}(undef, m, k)
    @inbounds for idx in 1:len
        Phi[idx] = K(phi_obj.data[idx])
    end
    return Phi
end

function _decode_phi(phi_obj::_PhiRealFlatJSON,
                     saved_field::RealField,
                     target_field::AbstractCoeffField,
                     m_expected::Int,
                     k_expected::Int)
    m, k = _phi_dims(phi_obj)
    (m == m_expected && k == k_expected) || error("phi dimensions mismatch (expected $(m_expected)x$(k_expected), got $(m)x$(k)).")
    len = m * k
    length(phi_obj.data) == len || error("real_flat_v1.data length mismatch.")
    K = coeff_type(target_field)
    Phi = Matrix{K}(undef, m, k)
    @inbounds for idx in 1:len
        Phi[idx] = coerce(target_field, phi_obj.data[idx])
    end
    return Phi
end

function _decode_phi(phi_obj::_PhiJSON,
                     saved_field::AbstractCoeffField,
                     target_field::AbstractCoeffField,
                     m_expected::Int,
                     k_expected::Int)
    error("Unsupported phi payload/field combination: $(typeof(phi_obj)) with saved_field=$(typeof(saved_field))")
end

function _parse_poset_from_obj(poset_obj)
    kind = haskey(poset_obj, "kind") ? String(poset_obj["kind"]) : "FinitePoset"
    if kind == "FinitePoset"
        haskey(poset_obj, "n") || error("poset missing required key 'n'.")
        haskey(poset_obj, "leq") || error("poset missing required key 'leq'.")
        n = Int(poset_obj["n"])
        leq_any = poset_obj["leq"]
        leq = if _is_packed_words_obj(leq_any)
            packed = _MaskPackedWordsJSON(kind=String(leq_any["kind"]),
                                          nrows=Int(leq_any["nrows"]),
                                          ncols=Int(leq_any["ncols"]),
                                          words_per_row=Int(leq_any["words_per_row"]),
                                          words=Vector{UInt64}(leq_any["words"]))
            _decode_bitmatrix(packed, "FinitePoset.leq", n, n)
        else
            leq_any isa AbstractVector || error("poset.leq must be a list-of-lists.")
            length(leq_any) == n || error("poset.leq must have n=$(n) rows")
            L = falses(n, n)
            for i in 1:n
                row = leq_any[i]
                row isa AbstractVector || error("poset.leq row $(i) must be a list.")
                length(row) == n || error("poset.leq row length mismatch (row $(i); expected n=$(n)).")
                for j in 1:n
                    x = row[j]
                    x isa Bool || error("poset.leq entries must be Bool (row $(i), col $(j)).")
                    L[i, j] = x
                end
            end
            L
        end
        P = FinitePoset(leq)
        _clear_cover_cache!(P)
        return P
    elseif kind == "ProductOfChainsPoset"
        haskey(poset_obj, "sizes") || error("ProductOfChainsPoset missing required key 'sizes'.")
        sizes = Vector{Int}(poset_obj["sizes"])
        P = ProductOfChainsPoset(sizes)
        _clear_cover_cache!(P)
        return P
    elseif kind == "GridPoset"
        haskey(poset_obj, "coords") || error("GridPoset missing required key 'coords'.")
        coords_any = poset_obj["coords"]
        coords_any isa AbstractVector || error("GridPoset.coords must be a list-of-lists.")
        coords = ntuple(i -> Vector{Float64}(coords_any[i]), length(coords_any))
        P = GridPoset(coords)
        _clear_cover_cache!(P)
        return P
    elseif kind == "ProductPoset"
        haskey(poset_obj, "left") || error("ProductPoset missing required key 'left'.")
        haskey(poset_obj, "right") || error("ProductPoset missing required key 'right'.")
        P1 = _parse_poset_from_obj(poset_obj["left"])
        P2 = _parse_poset_from_obj(poset_obj["right"])
        P = ProductPoset(P1, P2)
        _clear_cover_cache!(P)
        return P
    elseif kind == "SignaturePoset"
        haskey(poset_obj, "sig_y") || error("SignaturePoset missing required key 'sig_y'.")
        haskey(poset_obj, "sig_z") || error("SignaturePoset missing required key 'sig_z'.")
        sig_y_any = poset_obj["sig_y"]
        sig_z_any = poset_obj["sig_z"]
        sig_y = if _is_packed_words_obj(sig_y_any)
            _parse_signature_rows_packed(
                _MaskPackedWordsJSON(kind=String(sig_y_any["kind"]),
                                     nrows=Int(sig_y_any["nrows"]),
                                     ncols=Int(sig_y_any["ncols"]),
                                     words_per_row=Int(sig_y_any["words_per_row"]),
                                     words=Vector{UInt64}(sig_y_any["words"])),
                "SignaturePoset.sig_y")
        else
            _parse_signature_rows_packed(sig_y_any, "SignaturePoset.sig_y")
        end
        sig_z = if _is_packed_words_obj(sig_z_any)
            _parse_signature_rows_packed(
                _MaskPackedWordsJSON(kind=String(sig_z_any["kind"]),
                                     nrows=Int(sig_z_any["nrows"]),
                                     ncols=Int(sig_z_any["ncols"]),
                                     words_per_row=Int(sig_z_any["words_per_row"]),
                                     words=Vector{UInt64}(sig_z_any["words"])),
                "SignaturePoset.sig_z")
        else
            _parse_signature_rows_packed(sig_z_any, "SignaturePoset.sig_z")
        end
        P = SignaturePoset(sig_y, sig_z)
        _clear_cover_cache!(P)
        return P
    else
        error("Unsupported poset kind: $(kind)")
    end
end

function _poset_obj(P::AbstractPoset; include_leq::Union{Bool,Symbol}=:auto)
    include_leq_resolved = _resolve_include_leq(P, include_leq)
    if P isa FinitePoset
        include_leq_resolved || error("Cannot omit leq for FinitePoset serialization.")
        L = leq_matrix(P)
        return Dict("kind" => "FinitePoset",
                    "n" => nvertices(P),
                    "leq" => _pack_bitmatrix_obj(L))
    elseif P isa ProductOfChainsPoset
        obj = Dict("kind" => "ProductOfChainsPoset",
                   "n" => nvertices(P),
                   "sizes" => collect(P.sizes))
        if include_leq_resolved
            L = leq_matrix(P)
            obj["leq"] = _pack_bitmatrix_obj(L)
        end
        return obj
    elseif P isa GridPoset
        obj = Dict("kind" => "GridPoset",
                   "n" => nvertices(P),
                   "coords" => [collect(c) for c in P.coords])
        if include_leq_resolved
            L = leq_matrix(P)
            obj["leq"] = _pack_bitmatrix_obj(L)
        end
        return obj
    elseif P isa ProductPoset
        obj = Dict("kind" => "ProductPoset",
                   "n" => nvertices(P),
                   "left" => _poset_obj(P.P1; include_leq=(include_leq === :auto ? :auto : include_leq_resolved)),
                   "right" => _poset_obj(P.P2; include_leq=(include_leq === :auto ? :auto : include_leq_resolved)))
        if include_leq_resolved
            L = leq_matrix(P)
            obj["leq"] = _pack_bitmatrix_obj(L)
        end
        return obj
    elseif P isa SignaturePoset
        obj = Dict("kind" => "SignaturePoset",
                   "n" => nvertices(P),
                   "sig_y" => _pack_signature_rows_obj(P.sig_y),
                   "sig_z" => _pack_signature_rows_obj(P.sig_z))
        if include_leq_resolved
            L = leq_matrix(P)
            obj["leq"] = _pack_bitmatrix_obj(L)
        end
        return obj
    else
        L = leq_matrix(P)
        return Dict("kind" => "FinitePoset",
                    "n" => nvertices(P),
                    "leq" => _pack_bitmatrix_obj(L))
    end
end

"""
    load_boundary_complex_json(path) -> GradedComplex

Expected schema (external adapter):
{
  "cells_by_dim": [[1,2,...], [1,2,...], ...]  // or "counts_by_dim": [n0, n1, ...]
  "boundaries": [ {"m":..,"n":..,"I":[..],"J":[..],"V":[..]}, ... ],
  "grades": [ [..], [..], ... ],
  "cell_dims": [..]   // optional
}
"""
function load_boundary_complex_json(path::AbstractString)
    obj = _json_read(path)
    cells = if haskey(obj, "cells_by_dim")
        [Vector{Int}(c) for c in obj["cells_by_dim"]]
    elseif haskey(obj, "counts_by_dim")
        counts = Vector{Int}(obj["counts_by_dim"])
        out = Vector{Vector{Int}}(undef, length(counts))
        for d in 1:length(counts)
            out[d] = collect(1:counts[d])
        end
        out
    else
        error("boundary complex JSON missing 'cells_by_dim' or 'counts_by_dim'.")
    end
    boundaries = SparseMatrixCSC{Int,Int}[]
    for b in obj["boundaries"]
        m = Int(b["m"]); n = Int(b["n"])
        I = Vector{Int}(b["I"])
        J = Vector{Int}(b["J"])
        V = Vector{Int}(b["V"])
        push!(boundaries, sparse(I, J, V, m, n))
    end
    grades = [Vector{Float64}(g) for g in obj["grades"]]
    cell_dims = haskey(obj, "cell_dims") ? Vector{Int}(obj["cell_dims"]) : nothing
    return GradedComplex(cells, boundaries, grades; cell_dims=cell_dims)
end

"""
    load_reduced_complex_json(path) -> GradedComplex

Alias for `load_boundary_complex_json`, intended for reduced boundary matrices.
"""
load_reduced_complex_json(path::AbstractString) = load_boundary_complex_json(path)

"""
    load_pmodule_json(path; field=nothing) -> PModule

Expected schema:
{
  "poset": { "kind": "FinitePoset", "n": n, "leq": [[...]] },
  "dims": [d1, d2, ...],
  "edges": [ {"src": i, "dst": j, "mat": [[...]]}, ... ],
  "coeff_field": { ... }   // optional; defaults to QQ
}
"""
function load_pmodule_json(path::AbstractString; field::Union{Nothing,AbstractCoeffField}=nothing)
    obj = _json_read(path)
    haskey(obj, "poset") || error("pmodule JSON missing 'poset'.")
    P = _parse_poset_from_obj(obj["poset"])
    saved_field = haskey(obj, "coeff_field") ? _field_from_obj(obj["coeff_field"]) : QQField()
    target_field = field === nothing ? saved_field : field
    K = coeff_type(target_field)
    haskey(obj, "dims") || error("pmodule JSON missing 'dims'.")
    dims = Vector{Int}(obj["dims"])
    length(dims) == nvertices(P) || error("pmodule dims length mismatch with poset size.")
    haskey(obj, "edges") || error("pmodule JSON missing 'edges'.")
    edge_maps = Dict{Tuple{Int,Int},Matrix{K}}()
    for e in obj["edges"]
        src = Int(e["src"])
        dst = Int(e["dst"])
        mat_any = e["mat"]
        mat_any isa AbstractVector || error("pmodule edge mat must be a list-of-lists.")
        m = length(mat_any)
        n = m == 0 ? 0 : length(mat_any[1])
        M = zeros(K, m, n)
        for i in 1:m
            row = mat_any[i]
            length(row) == n || error("pmodule edge mat row length mismatch.")
            for j in 1:n
                val = _scalar_from_json(saved_field, row[j])
                M[i, j] = target_field === saved_field ? val : coerce(target_field, val)
            end
        end
        edge_maps[(src, dst)] = M
    end
    return PModule{K}(P, dims, edge_maps; field=target_field)
end

# -----------------------------------------------------------------------------
# B1) Interop adapters: Ripser/DIPHA distance matrix formats
# -----------------------------------------------------------------------------

function _read_numeric_rows(path::AbstractString)
    rows = Vector{Vector{Float64}}()
    open(path, "r") do io
        for raw in eachline(io)
            line = strip(raw)
            isempty(line) && continue
            startswith(line, "#") && continue
            line = replace(line, ',' => ' ')
            line = replace(line, ';' => ' ')
            parts = split(line)
            isempty(parts) && continue
            row = Float64[parse(Float64, p) for p in parts]
            push!(rows, row)
        end
    end
    return rows
end

function _matrix_from_rows(rows::Vector{Vector{Float64}})
    isempty(rows) && error("distance matrix: empty file.")

    if length(rows) >= 2 && length(rows[1]) == 1
        n_header = Int(round(rows[1][1]))
        if n_header >= 1 && length(rows) - 1 == n_header &&
           all(length(rows[i]) == n_header for i in 2:length(rows))
            dist = zeros(Float64, n_header, n_header)
            for i in 1:n_header
                dist[i, :] = rows[i + 1]
            end
            return dist
        end
    end

    same_len = all(length(r) == length(rows[1]) for r in rows)
    if same_len
        m = length(rows)
        k = length(rows[1])
        if m == k
            dist = zeros(Float64, m, m)
            for i in 1:m
                dist[i, :] = rows[i]
            end
            return dist
        elseif k == 1
            vals = [r[1] for r in rows]
            n = round(Int, sqrt(length(vals)))
            n * n == length(vals) || error("distance matrix: flat list length is not a perfect square.")
            dist = zeros(Float64, n, n)
            for i in 1:n, j in 1:n
                dist[i, j] = vals[(i - 1) * n + j]
            end
            return dist
        elseif m == 1
            vals = rows[1]
            n = round(Int, sqrt(length(vals)))
            n * n == length(vals) || error("distance matrix: flat list length is not a perfect square.")
            dist = zeros(Float64, n, n)
            for i in 1:n, j in 1:n
                dist[i, j] = vals[(i - 1) * n + j]
            end
            return dist
        end
    end

    vals = reduce(vcat, rows)
    n = round(Int, sqrt(length(vals)))
    n * n == length(vals) || error("distance matrix: could not infer square size.")
    dist = zeros(Float64, n, n)
    for i in 1:n, j in 1:n
        dist[i, j] = vals[(i - 1) * n + j]
    end
    return dist
end

function _infer_n_from_triangular_len(len::Int)
    n1 = Int(floor((sqrt(1 + 8 * len) - 1) / 2))
    if div(n1 * (n1 + 1), 2) == len
        return n1, true
    end
    n2 = Int(floor((1 + sqrt(1 + 8 * len)) / 2))
    if div(n2 * (n2 - 1), 2) == len
        return n2, false
    end
    return 0, false
end

function _triangular_from_rows(rows::Vector{Vector{Float64}}; upper::Bool)
    n = length(rows)
    if upper
        include_diag = if all(length(rows[i]) == n - i + 1 for i in 1:n)
            true
        elseif all(length(rows[i]) == n - i for i in 1:n)
            false
        else
            return nothing
        end
        dist = fill(Inf, n, n)
        for i in 1:n
            row = rows[i]
            if include_diag
                for k in 1:(n - i + 1)
                    j = i + k - 1
                    dist[i, j] = row[k]
                    dist[j, i] = row[k]
                end
            else
                for k in 1:(n - i)
                    j = i + k
                    dist[i, j] = row[k]
                    dist[j, i] = row[k]
                end
            end
            dist[i, i] = 0.0
        end
        return dist
    else
        include_diag = if all(length(rows[i]) == i for i in 1:n)
            true
        elseif all(length(rows[i]) == i - 1 for i in 1:n)
            false
        else
            return nothing
        end
        dist = fill(Inf, n, n)
        for i in 1:n
            row = rows[i]
            if include_diag
                for j in 1:i
                    dist[i, j] = row[j]
                    dist[j, i] = row[j]
                end
            else
                for j in 1:(i - 1)
                    dist[i, j] = row[j]
                    dist[j, i] = row[j]
                end
            end
            dist[i, i] = 0.0
        end
        return dist
    end
end

function _triangular_from_vals(vals::Vector{Float64}; upper::Bool)
    n, include_diag = _infer_n_from_triangular_len(length(vals))
    n > 0 || error("triangular distance list length is not valid.")
    dist = fill(Inf, n, n)
    idx = 1
    if upper
        for i in 1:n
            if include_diag
                for j in i:n
                    val = vals[idx]; idx += 1
                    dist[i, j] = val
                    dist[j, i] = val
                end
            else
                for j in (i + 1):n
                    val = vals[idx]; idx += 1
                    dist[i, j] = val
                    dist[j, i] = val
                end
            end
            dist[i, i] = 0.0
        end
    else
        for i in 1:n
            if include_diag
                for j in 1:i
                    val = vals[idx]; idx += 1
                    dist[i, j] = val
                    dist[j, i] = val
                end
            else
                for j in 1:(i - 1)
                    val = vals[idx]; idx += 1
                    dist[i, j] = val
                    dist[j, i] = val
                end
            end
            dist[i, i] = 0.0
        end
    end
    return dist
end

function _combinations(n::Int, k::Int)
    if k == 0
        return [Int[]]
    end
    out = Vector{Vector{Int}}()
    function rec(start::Int, acc::Vector{Int})
        if length(acc) == k
            push!(out, copy(acc))
            return
        end
        for i in start:(n - (k - length(acc)) + 1)
            push!(acc, i)
            rec(i + 1, acc)
            pop!(acc)
        end
    end
    rec(1, Int[])
    return out
end

@inline function _dm_budget_check_max_simplices!(total::Integer, budget::ConstructionBudget)
    ms = budget.max_simplices
    if ms !== nothing && total > ms
        error("distance matrix Rips: exceeded max_simplices=$(ms).")
    end
    return nothing
end

@inline function _dm_budget_check_max_edges!(edge_count, budget::ConstructionBudget)
    cap = budget.max_edges
    if cap !== nothing && big(edge_count) > big(cap)
        error("distance matrix Rips: exceeded max_edges=$(cap).")
    end
    return nothing
end

function _dm_edges_radius(dist::AbstractMatrix{<:Real}, radius::Float64)
    n = size(dist, 1)
    edges = Vector{Vector{Int}}()
    for i in 1:n, j in i+1:n
        if Float64(dist[i, j]) <= radius
            push!(edges, [i, j])
        end
    end
    return edges
end

function _dm_edges_knn(dist::AbstractMatrix{<:Real}, k::Int)
    n = size(dist, 1)
    0 < k < n || error("construction.sparsify=:knn requires 0 < knn < n.")
    e = Set{Tuple{Int,Int}}()
    for i in 1:n
        neigh = [(Float64(dist[i, j]), j) for j in 1:n if j != i]
        sort!(neigh, by=x -> x[1])
        tmax = min(k, length(neigh))
        for t in 1:tmax
            j = neigh[t][2]
            a, b = min(i, j), max(i, j)
            push!(e, (a, b))
        end
    end
    edges = [[ab[1], ab[2]] for ab in e]
    sort!(edges; by=s -> (s[1], s[2]))
    return edges
end

function _dm_edges_collapse_dominated(edges::Vector{Vector{Int}},
                                      dist::AbstractMatrix{<:Real};
                                      tol::Float64=1e-12)
    n = size(dist, 1)
    out = Vector{Vector{Int}}()
    for e in edges
        u, v = e[1], e[2]
        duv = Float64(dist[u, v])
        dominated = false
        for w in 1:n
            (w == u || w == v) && continue
            if max(Float64(dist[u, w]), Float64(dist[w, v])) <= duv + tol
                dominated = true
                break
            end
        end
        dominated || push!(out, e)
    end
    return out
end

function _dm_edges_collapse_acyclic(edges::Vector{Vector{Int}},
                                    dist::AbstractMatrix{<:Real})
    n = size(dist, 1)
    parent = collect(1:n)
    rank = zeros(Int, n)
    function findp(x)
        while parent[x] != x
            parent[x] = parent[parent[x]]
            x = parent[x]
        end
        return x
    end
    function unite(x, y)
        rx, ry = findp(x), findp(y)
        rx == ry && return false
        if rank[rx] < rank[ry]
            parent[rx] = ry
        elseif rank[rx] > rank[ry]
            parent[ry] = rx
        else
            parent[ry] = rx
            rank[rx] += 1
        end
        return true
    end
    idx = sortperm(1:length(edges); by=i -> Float64(dist[edges[i][1], edges[i][2]]))
    out = Vector{Vector{Int}}()
    for i in idx
        e = edges[i]
        unite(e[1], e[2]) && push!(out, e)
    end
    return out
end

function _dm_apply_collapse(edges::Vector{Vector{Int}},
                            dist::AbstractMatrix{<:Real},
                            collapse::Symbol)
    if collapse == :none
        return edges
    elseif collapse == :dominated_edges
        return _dm_edges_collapse_dominated(edges, dist)
    elseif collapse == :acyclic
        return _dm_edges_collapse_acyclic(edges, dist)
    end
    error("construction.collapse must be :none, :dominated_edges, or :acyclic.")
end

function _graded_complex_from_distance_matrix(dist::AbstractMatrix{<:Real};
                                              max_dim::Int=1,
                                              radius::Union{Nothing,Real}=nothing,
                                              knn::Union{Nothing,Int}=nothing,
                                              construction::ConstructionOptions=ConstructionOptions())
    size(dist, 1) == size(dist, 2) || error("distance matrix must be square.")
    max_dim >= 0 || error("max_dim must be >= 0.")
    n = size(dist, 1)
    n > 0 || error("distance matrix has size 0.")

    sparsify = construction.sparsify
    collapse = construction.collapse
    budget = construction.budget

    if sparsify == :greedy_perm
        error("construction.sparsify=:greedy_perm is not supported for distance-matrix ingestion.")
    end
    if sparsify != :none && max_dim > 1
        error("construction.sparsify=$(sparsify) currently supports max_dim <= 1 for distance-matrix ingestion.")
    end
    if collapse != :none && sparsify == :none
        error("construction.collapse requires construction.sparsify != :none for distance-matrix ingestion.")
    end
    if radius !== nothing && sparsify != :radius
        error("radius is only valid when construction.sparsify=:radius.")
    end
    if knn !== nothing && sparsify != :knn
        error("knn is only valid when construction.sparsify=:knn.")
    end

    simplices = Vector{Vector{Vector{Int}}}(undef, max_dim + 1)
    simplices[1] = [ [i] for i in 1:n ]
    total = length(simplices[1])

    if sparsify == :none
        if max_dim >= 1
            _dm_budget_check_max_edges!(binomial(big(n), big(2)), budget)
        end
        for k in 2:max_dim+1
            sims = Vector{Vector{Int}}()
            for comb in _combinations(n, k)
                push!(sims, comb)
            end
            simplices[k] = sims
            total += length(sims)
            _dm_budget_check_max_simplices!(total, budget)
        end
    else
        edges = if sparsify == :radius
            radius === nothing && error("construction.sparsify=:radius requires radius.")
            _dm_edges_radius(dist, Float64(radius))
        elseif sparsify == :knn
            knn === nothing && error("construction.sparsify=:knn requires knn.")
            _dm_edges_knn(dist, Int(knn))
        else
            error("construction.sparsify must be :none, :radius, or :knn for distance-matrix ingestion.")
        end
        edges = _dm_apply_collapse(edges, dist, collapse)
        _dm_budget_check_max_edges!(length(edges), budget)
        simplices = [simplices[1], edges]
        max_dim = 1
        total += length(edges)
        _dm_budget_check_max_simplices!(total, budget)
    end

    grades = Vector{Vector{Float64}}()
    for _ in simplices[1]
        push!(grades, [0.0])
    end
    for k in 2:max_dim+1
        for s in simplices[k]
            maxd = 0.0
            for i in 1:length(s)
                for j in (i+1):length(s)
                    d = Float64(dist[s[i], s[j]])
                    if d > maxd
                        maxd = d
                    end
                end
            end
            push!(grades, [maxd])
        end
    end

    boundaries = SparseMatrixCSC{Int,Int}[]
    for k in 2:max_dim+1
        Bk = _simplicial_boundary_from_lists(simplices[k], simplices[k-1])
        push!(boundaries, Bk)
    end
    cells = [collect(1:length(simplices[k])) for k in 1:length(simplices)]
    return GradedComplex(cells, boundaries, grades)
end

"""
    load_ripser_point_cloud(path) -> PointCloud

Parse a Ripser-style point cloud (whitespace-separated coordinates per line).
"""
function load_ripser_point_cloud(path::AbstractString)
    rows = _read_numeric_rows(path)
    isempty(rows) && error("point cloud file has no points.")
    return PointCloud(_coords_from_flat_rowmajor(length(rows), length(rows[1]), vcat(rows...)))
end

"""
    load_ripser_distance(path; max_dim=1, radius=nothing, knn=nothing,
                         construction=ConstructionOptions()) -> GradedComplex

Parse a full distance matrix (square) and build a 1-parameter Rips graded complex.
"""
function load_ripser_distance(path::AbstractString;
                              max_dim::Int=1,
                              radius::Union{Nothing,Real}=nothing,
                              knn::Union{Nothing,Int}=nothing,
                              construction::ConstructionOptions=ConstructionOptions())
    rows = _read_numeric_rows(path)
    dist = _matrix_from_rows(rows)
    return _graded_complex_from_distance_matrix(dist;
                                                max_dim=max_dim,
                                                radius=radius,
                                                knn=knn,
                                                construction=construction)
end

"""
    load_ripser_lower_distance(path; max_dim=1, radius=nothing, knn=nothing,
                               construction=ConstructionOptions()) -> GradedComplex

Parse a lower-triangular distance matrix (row-wise or flat list) and build a Rips complex.
"""
function load_ripser_lower_distance(path::AbstractString;
                                    max_dim::Int=1,
                                    radius::Union{Nothing,Real}=nothing,
                                    knn::Union{Nothing,Int}=nothing,
                                    construction::ConstructionOptions=ConstructionOptions())
    rows = _read_numeric_rows(path)
    dist = _triangular_from_rows(rows; upper=false)
    if dist === nothing
        dist = _triangular_from_vals(reduce(vcat, rows); upper=false)
    end
    return _graded_complex_from_distance_matrix(dist;
                                                max_dim=max_dim,
                                                radius=radius,
                                                knn=knn,
                                                construction=construction)
end

"""
    load_ripser_upper_distance(path; max_dim=1, radius=nothing, knn=nothing,
                               construction=ConstructionOptions()) -> GradedComplex

Parse an upper-triangular distance matrix (row-wise or flat list) and build a Rips complex.
"""
function load_ripser_upper_distance(path::AbstractString;
                                    max_dim::Int=1,
                                    radius::Union{Nothing,Real}=nothing,
                                    knn::Union{Nothing,Int}=nothing,
                                    construction::ConstructionOptions=ConstructionOptions())
    rows = _read_numeric_rows(path)
    dist = _triangular_from_rows(rows; upper=true)
    if dist === nothing
        dist = _triangular_from_vals(reduce(vcat, rows); upper=true)
    end
    return _graded_complex_from_distance_matrix(dist;
                                                max_dim=max_dim,
                                                radius=radius,
                                                knn=knn,
                                                construction=construction)
end

"""
    load_ripser_sparse_triplet(path; max_dim=1, radius=nothing, knn=nothing,
                               construction=ConstructionOptions()) -> GradedComplex

Parse a sparse triplet format: each nonzero entry as "i j d".
Indices can be 0-based or 1-based.
"""
function load_ripser_sparse_triplet(path::AbstractString;
                                    max_dim::Int=1,
                                    radius::Union{Nothing,Real}=nothing,
                                    knn::Union{Nothing,Int}=nothing,
                                    construction::ConstructionOptions=ConstructionOptions())
    rows = _read_numeric_rows(path)
    isempty(rows) && error("sparse triplet file is empty.")
    for row in rows
        length(row) == 3 || error("sparse triplet rows must have 3 entries.")
    end
    idxs = [Int(round(r[1])) for r in rows]
    jdxs = [Int(round(r[2])) for r in rows]
    base0 = any(i == 0 for i in idxs) || any(j == 0 for j in jdxs)
    if base0
        idxs .= idxs .+ 1
        jdxs .= jdxs .+ 1
    end
    n = max(maximum(idxs), maximum(jdxs))
    n > 0 || error("sparse triplet: could not infer matrix size.")
    dist = fill(Inf, n, n)
    for i in 1:n
        dist[i, i] = 0.0
    end
    for t in 1:length(rows)
        i = idxs[t]
        j = jdxs[t]
        d = Float64(rows[t][3])
        if d < dist[i, j]
            dist[i, j] = d
            dist[j, i] = d
        end
    end
    return _graded_complex_from_distance_matrix(dist;
                                                max_dim=max_dim,
                                                radius=radius,
                                                knn=knn,
                                                construction=construction)
end

"""
    load_ripser_binary_lower_distance(path; max_dim=1, radius=nothing, knn=nothing,
                                      construction=ConstructionOptions()) -> GradedComplex

Parse Ripser's binary lower-triangular distance matrix (Float64 values).
"""
function load_ripser_binary_lower_distance(path::AbstractString;
                                           max_dim::Int=1,
                                           radius::Union{Nothing,Real}=nothing,
                                           knn::Union{Nothing,Int}=nothing,
                                           construction::ConstructionOptions=ConstructionOptions())
    vals = Float64[]
    open(path, "r") do io
        while !eof(io)
            push!(vals, read(io, Float64))
        end
    end
    dist = _triangular_from_vals(vals; upper=false)
    return _graded_complex_from_distance_matrix(dist;
                                                max_dim=max_dim,
                                                radius=radius,
                                                knn=knn,
                                                construction=construction)
end

"""
    load_dipha_distance_matrix(path; max_dim=1, radius=nothing, knn=nothing,
                               construction=ConstructionOptions()) -> GradedComplex

Parse a DIPHA binary distance matrix and build a Rips graded complex.
"""
function load_dipha_distance_matrix(path::AbstractString;
                                    max_dim::Int=1,
                                    radius::Union{Nothing,Real}=nothing,
                                    knn::Union{Nothing,Int}=nothing,
                                    construction::ConstructionOptions=ConstructionOptions())
    open(path, "r") do io
        eof(io) && error("DIPHA distance matrix: empty file.")
        magic = read(io, Int64)
        magic == 8067171840 || error("DIPHA: invalid magic value.")
        _ = read(io, Int64) # file type id
        n = Int(read(io, Int64))
        n > 0 || error("DIPHA: invalid matrix size.")
        vals = Vector{Float64}(undef, n * n)
        read!(io, vals)
        dist = zeros(Float64, n, n)
        idx = 1
        for i in 1:n, j in 1:n
            dist[i, j] = vals[idx]
            idx += 1
        end
        return _graded_complex_from_distance_matrix(dist;
                                                    max_dim=max_dim,
                                                    radius=radius,
                                                    knn=knn,
                                                    construction=construction)
    end
end

"""
    load_ripser_lower_distance_streaming(path; radius, max_dim=1) -> GradedComplex

Streaming reader for lower-triangular distance matrices (text). Builds a 1-skeleton
Rips complex without loading the full matrix.
"""
function load_ripser_lower_distance_streaming(path::AbstractString; radius, max_dim::Int=1)
    max_dim == 1 || error("streaming lower distance currently supports max_dim=1 only.")
    radius === nothing && error("streaming lower distance requires radius.")
    edges = Vector{Vector{Int}}()
    grades = Vector{Vector{Float64}}()
    n = 0
    include_diag = nothing
    open(path, "r") do io
        for raw in eachline(io)
            line = strip(raw)
            isempty(line) && continue
            startswith(line, "#") && continue
            line = replace(line, ',' => ' ')
            parts = split(line)
            isempty(parts) && continue
            row = Float64[parse(Float64, p) for p in parts]
            n += 1
            if include_diag === nothing
                if length(row) == n
                    include_diag = true
                elseif length(row) == n - 1
                    include_diag = false
                else
                    error("streaming lower distance: row length mismatch at row $(n).")
                end
            else
                expected = include_diag ? n : n - 1
                length(row) == expected || error("streaming lower distance: row length mismatch at row $(n).")
            end
            for j in 1:length(row)
                if include_diag && j == n
                    continue
                end
                d = row[j]
                if d <= radius
                    push!(edges, [j, n])
                    push!(grades, [d])
                end
            end
        end
    end
    n > 0 || error("streaming lower distance: no rows found.")
    vertices = [ [i] for i in 1:n ]
    cells = [collect(1:length(vertices)), collect(1:length(edges))]
    all_grades = Vector{Vector{Float64}}(undef, length(vertices) + length(edges))
    for i in 1:length(vertices)
        all_grades[i] = [0.0]
    end
    for i in 1:length(edges)
        all_grades[length(vertices) + i] = grades[i]
    end
    B1 = _simplicial_boundary_from_lists(edges, vertices)
    return GradedComplex(cells, [B1], all_grades)
end

@inline function _load_flange_obj(obj;
                                  field::Union{Nothing,AbstractCoeffField}=nothing)
    haskey(obj, "kind") && String(obj["kind"]) == "FlangeZn" ||
        error("Flange JSON must set kind=\"FlangeZn\".")
    n = Int(obj["n"])

    mkface(idxs) = Face(n, begin
        m = falses(n)
        for t in idxs
            m[Int(t)] = true
        end
        m
    end)

    flats = [IndFlat(mkface(Vector{Int}(f["tau"])), Vector{Int}(f["b"]); id=:F)
             for f in obj["flats"]]
    injectives = [IndInj(mkface(Vector{Int}(e["tau"])), Vector{Int}(e["b"]); id=:E)
                  for e in obj["injectives"]]

    saved_field = haskey(obj, "coeff_field") ? _field_from_obj(obj["coeff_field"]) : QQField()
    target_field = field === nothing ? saved_field : field
    K = coeff_type(target_field)
    m = length(injectives)
    k = length(flats)
    Phi = Matrix{K}(undef, m, k)
    for i in 1:m, j in 1:k
        s = _scalar_from_json(saved_field, obj["phi"][i][j])
        if target_field !== saved_field
            s = coerce(target_field, s)
        end
        Phi[i, j] = s
    end
    return Flange{K}(n, flats, injectives, Phi; field=target_field)
end

"""
    load_flange_json(path; field=nothing, validation=:strict) -> FlangeZn.Flange

Load a flange artifact written by [`save_flange_json`](@ref).

This is a strict owned-schema loader. When `field` is supplied, coefficients
are coerced into that target field. Prefer [`flange_json_summary`](@ref) for a
cheap-first family check and [`check_flange_json`](@ref) when you need explicit
schema validation before loading.
Use `validation=:trusted` only for TamerOp-produced flange files. The
trusted path keeps the same owned-family contract but skips no extra work yet;
the keyword is present to keep the owned loader surface uniform.
"""
function load_flange_json(path::AbstractString;
                          field::Union{Nothing,AbstractCoeffField}=nothing,
                          validation::Symbol=:strict)
    _resolve_validation_mode(validation)
    return _load_flange_obj(_json_read(path); field=field)
end

