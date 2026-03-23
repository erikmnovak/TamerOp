module ZnEncoding

using LinearAlgebra
using SparseArrays
using Random
using Base.Threads

using ..CoreModules: AbstractCoeffField, coeff_type, eye, SessionCache, _field_cache_key,
                     _session_get_zn_encoding_artifact, _session_set_zn_encoding_artifact!,
                     _session_get_zn_pushforward_plan, _session_set_zn_pushforward_plan!,
                     _session_get_zn_pushforward_fringe, _session_set_zn_pushforward_fringe!
using ..CoreModules.CoeffFields: QQField, PrimeField, RealField
using ..Options: EncodingOptions
using ..EncodingCore: AbstractPLikeEncodingMap, CompiledEncoding
using ..Stats: _wilson_interval

import ..EncodingCore: locate, locate_many!, locate_many, dimension, representatives, axes_from_encoding
import ..RegionGeometry: region_weights, region_adjacency
import ..FieldLinAlg
import ..FiniteFringe
using ..FiniteFringe: AbstractPoset, FinitePoset, ProductOfChainsPoset, Upset, Downset,
                       upset_closure, downset_closure, intersects, FringeModule, CoverEdges,
                       poset_equal
import ..FiniteFringe: nvertices, leq, upset_indices, downset_indices, leq_row, leq_col, cover_edges
using ..Modules: PModule, PosetCache
using ..FlangeZn: Flange, IndFlat, IndInj, in_flat, in_inj
import ..FlangeZn: flats, injectives, generator_counts

# Build the finite grid poset on [a,b] subset Z^n, ordered coordinatewise.
# Returns (Q, coords) where coords[i] is an NTuple{n,Int} in mixed-radix order.
# Uses a structured ProductOfChainsPoset to avoid materializing the transitive closure.
function grid_poset(a::NTuple{N,Int}, b::NTuple{N,Int}) where {N}
    lens = ntuple(i -> b[i] - a[i] + 1, N)
    if any(l -> l <= 0, lens)
        error("grid_poset: invalid box")
    end
    Q = ProductOfChainsPoset(lens)

    total = prod(lens)
    coords = Vector{NTuple{N, Int}}(undef, total)
    strides = Vector{Int}(undef, N)
    strides[1] = 1
    @inbounds for i in 2:N
        strides[i] = strides[i - 1] * lens[i - 1]
    end
    @inbounds for idx in 1:total
        coords[idx] = ntuple(k -> a[k] + (div(idx - 1, strides[k]) % lens[k]), N)
    end
    return Q, coords
end

##############################
# Packed signature storage/keys
##############################

struct PackedSignatureRow{NW} <: AbstractVector{Bool}
    words::Matrix{UInt64}
    col::Int
    bitlen::Int
end

"""
    PackedSignatureRows{NW}

Packed signature table with `NW` UInt64 words per region-signature row.
`bitlen` tracks the logical number of bits (generators) in each row.
"""
struct PackedSignatureRows{NW} <: AbstractVector{PackedSignatureRow{NW}}
    words::Matrix{UInt64}
    bitlen::Int
    function PackedSignatureRows{NW}(words::Matrix{UInt64}, bitlen::Int) where {NW}
        size(words, 1) == NW || error("PackedSignatureRows: expected $(NW) words per row, got $(size(words, 1))")
        bitlen >= 0 || error("PackedSignatureRows: bitlen must be nonnegative")
        bitlen <= 64 * NW || error("PackedSignatureRows: bitlen=$(bitlen) exceeds word capacity $(64 * NW)")
        new{NW}(words, bitlen)
    end
end

Base.IndexStyle(::Type{<:PackedSignatureRows}) = IndexLinear()
Base.IndexStyle(::Type{<:PackedSignatureRow}) = IndexLinear()
Base.size(S::PackedSignatureRows) = (size(S.words, 2),)
Base.length(S::PackedSignatureRows) = size(S.words, 2)
Base.eltype(::Type{PackedSignatureRows{NW}}) where {NW} = PackedSignatureRow{NW}
Base.eltype(::Type{PackedSignatureRow{NW}}) where {NW} = Bool

@inline function Base.getindex(S::PackedSignatureRows{NW}, i::Int) where {NW}
    @boundscheck checkbounds(S, i)
    return PackedSignatureRow{NW}(S.words, i, S.bitlen)
end

Base.size(r::PackedSignatureRow) = (r.bitlen,)
Base.length(r::PackedSignatureRow) = r.bitlen

@inline function Base.getindex(r::PackedSignatureRow, i::Int)::Bool
    @boundscheck checkbounds(r, i)
    w = ((i - 1) >>> 6) + 1
    bit = UInt64(1) << ((i - 1) & 63)
    return (r.words[w, r.col] & bit) != 0
end

@inline function Base.iterate(r::PackedSignatureRow, state::Int=1)
    state > r.bitlen && return nothing
    return (r[state], state + 1)
end

@inline _word_lastmask(bitlen::Int) = begin
    rem = bitlen & 63
    rem == 0 ? typemax(UInt64) : (UInt64(1) << rem) - 1
end

@inline function _sig_subset_words(words_a::Matrix{UInt64},
                                   idx_a::Int,
                                   words_b::Matrix{UInt64},
                                   idx_b::Int,
                                   lastmask::UInt64)::Bool
    nw = size(words_a, 1)
    @inbounds for w in 1:nw
        diff = words_a[w, idx_a] & ~words_b[w, idx_b]
        if w == nw
            diff &= lastmask
        end
        diff == 0 || return false
    end
    return true
end

@inline function _sig_popcount_words(words::Matrix{UInt64}, idx::Int, lastmask::UInt64)::Int
    nw = size(words, 1)
    c = 0
    @inbounds for w in 1:nw
        word = words[w, idx]
        if w == nw
            word &= lastmask
        end
        c += count_ones(word)
    end
    return c
end

function _word_matrix_from_cols(cols::Vector{NTuple{NW,UInt64}}, ::Val{NW}) where {NW}
    n = length(cols)
    out = Matrix{UInt64}(undef, NW, n)
    @inbounds for j in 1:n
        col = cols[j]
        for w in 1:NW
            out[w, j] = col[w]
        end
    end
    return out
end

@inline function _words_from_signature_row(row::AbstractVector{Bool}, ::Val{MW}) where {MW}
    len = length(row)
    return ntuple(w -> begin
        word = UInt64(0)
        base = (w - 1) * 64
        @inbounds for j in 1:64
            i = base + j
            i > len && break
            if row[i]
                word |= (UInt64(1) << (j - 1))
            end
        end
        word
    end, MW)
end

function _pack_signature_rows(rows::PackedSignatureRows{MW},
                              bitlen::Int,
                              ::Val{MW}) where {MW}
    rows.bitlen == bitlen || error("_pack_signature_rows: row bit length mismatch")
    return rows
end

function _pack_signature_rows(rows::AbstractVector{<:AbstractVector{Bool}},
                              bitlen::Int,
                              ::Val{MW}) where {MW}
    packed = Vector{NTuple{MW,UInt64}}(undef, length(rows))
    @inbounds for i in eachindex(rows)
        row = rows[i]
        length(row) == bitlen || error("_pack_signature_rows: signature row $(i) has wrong length")
        packed[i] = _words_from_signature_row(row, Val(MW))
    end
    return PackedSignatureRows{MW}(_word_matrix_from_cols(packed, Val(MW)), bitlen)
end

"""
    SigKey{MY,MZ}

Packed key for (y,z)-signature lookup in `ZnEncodingMap`.
"""
struct SigKey{MY,MZ}
    y::NTuple{MY,UInt64}
    z::NTuple{MZ,UInt64}
end

@inline function _pack_signature_words(flats::Vector{IndFlat{N}},
                                       g::Union{AbstractVector{<:Integer},NTuple{N,<:Integer}},
                                       ::Val{MY}) where {N,MY}
    m = length(flats)
    return ntuple(w -> begin
        word = UInt64(0)
        base = (w - 1) * 64
        @inbounds for j in 1:64
            i = base + j
            i > m && break
            if in_flat(flats[i], g)
                word |= (UInt64(1) << (j - 1))
            end
        end
        word
    end, MY)
end

@inline function _pack_signature_words(injectives::Vector{IndInj{N}},
                                       g::Union{AbstractVector{<:Integer},NTuple{N,<:Integer}},
                                       ::Val{MZ}) where {N,MZ}
    r = length(injectives)
    return ntuple(w -> begin
        word = UInt64(0)
        base = (w - 1) * 64
        @inbounds for j in 1:64
            i = base + j
            i > r && break
            if !in_inj(injectives[i], g)
                word |= (UInt64(1) << (j - 1))
            end
        end
        word
    end, MZ)
end

@inline function _pack_bitvector_words(sig::AbstractVector{Bool}, ::Val{MW}) where {MW}
    len = length(sig)
    return ntuple(w -> begin
        word = UInt64(0)
        base = (w - 1) * 64
        @inbounds for j in 1:64
            i = base + j
            i > len && break
            if sig[i]
                word |= (UInt64(1) << (j - 1))
            end
        end
        word
    end, MW)
end

@inline function _sigkey_from_bitvectors(y::AbstractVector{Bool},
                                         z::AbstractVector{Bool},
                                         ::Val{MY},
                                         ::Val{MZ}) where {MY,MZ}
    SigKey{MY,MZ}(_pack_bitvector_words(y, Val(MY)),
                  _pack_bitvector_words(z, Val(MZ)))
end

@inline function _sigkey_at(g::Union{AbstractVector{<:Integer},NTuple{N,<:Integer}},
                            flats::Vector{IndFlat{N}},
                            injectives::Vector{IndInj{N}},
                            ::Val{MY},
                            ::Val{MZ}) where {N,MY,MZ}
    return SigKey{MY,MZ}(_pack_signature_words(flats, g, Val(MY)),
                         _pack_signature_words(injectives, g, Val(MZ)))
end

@inline function _set_sigword_bit!(words::Vector{UInt64}, idx::Int, value::Bool)
    w = (idx - 1) >>> 6
    b = UInt64(1) << ((idx - 1) & 63)
    pos = w + 1
    if value
        words[pos] |= b
    else
        words[pos] &= ~b
    end
    return nothing
end

@inline function _sigkey_from_words(y_words::Vector{UInt64},
                                    z_words::Vector{UInt64},
                                    ::Val{MY},
                                    ::Val{MZ}) where {MY,MZ}
    return SigKey{MY,MZ}(ntuple(i -> y_words[i], Val(MY)),
                         ntuple(i -> z_words[i], Val(MZ)))
end

function _build_signature_events(coords::NTuple{N,Vector{Int}},
                                 flats::Vector{IndFlat{N}},
                                 injectives::Vector{IndInj{N}}) where {N}
    flat_events = [ [Int[] for _ in 1:length(coords[i])] for i in 1:N ]
    inj_events  = [ [Int[] for _ in 1:length(coords[i])] for i in 1:N ]

    @inbounds for (fidx, F) in enumerate(flats)
        for i in 1:N
            F.tau.coords[i] && continue
            ci = coords[i]
            t = searchsortedfirst(ci, F.b[i])
            if t <= length(ci) && ci[t] == F.b[i]
                push!(flat_events[i][t], fidx)
            end
        end
    end

    @inbounds for (jidx, E) in enumerate(injectives)
        for i in 1:N
            E.tau.coords[i] && continue
            ci = coords[i]
            thr = E.b[i] + 1
            t = searchsortedfirst(ci, thr)
            if t <= length(ci) && ci[t] == thr
                push!(inj_events[i][t], jidx)
            end
        end
    end

    return flat_events, inj_events
end

@inline function _apply_signature_step!(axis::Int, old_s::Int, new_s::Int,
                                        flat_events,
                                        inj_events,
                                        flat_unsat::Vector{Int},
                                        inj_viol::Vector{Int},
                                        y_words::Vector{UInt64},
                                        z_words::Vector{UInt64})
    @boundscheck abs(new_s - old_s) == 1 || error("_apply_signature_step!: expected adjacent slab indices")
    if new_s > old_s
        # Crossing threshold index `new_s` upward.
        t = new_s
        @inbounds for f in flat_events[axis][t]
            flat_unsat[f] -= 1
            if flat_unsat[f] == 0
                _set_sigword_bit!(y_words, f, true)
            end
        end
        @inbounds for j in inj_events[axis][t]
            inj_viol[j] += 1
            if inj_viol[j] == 1
                _set_sigword_bit!(z_words, j, true)
            end
        end
    else
        # Crossing threshold index `old_s` downward.
        t = old_s
        @inbounds for f in flat_events[axis][t]
            was_active = flat_unsat[f] == 0
            flat_unsat[f] += 1
            if was_active
                _set_sigword_bit!(y_words, f, false)
            end
        end
        @inbounds for j in inj_events[axis][t]
            inj_viol[j] -= 1
            if inj_viol[j] == 0
                _set_sigword_bit!(z_words, j, false)
            end
        end
    end
    return nothing
end

function _collect_signatures_incremental(coords::NTuple{N,Vector{Int}},
                                         flats::Vector{IndFlat{N}},
                                         injectives::Vector{IndInj{N}},
                                         max_regions::Int,
                                         ::Val{MY},
                                         ::Val{MZ}) where {N,MY,MZ}
    m = length(flats)
    r = length(injectives)

    flat_events, inj_events = _build_signature_events(coords, flats, injectives)
    shape = ntuple(i -> length(coords[i]) + 1, N)
    strides_vec = Vector{Int}(undef, N)
    strides_vec[1] = 1
    @inbounds for i in 2:N
        strides_vec[i] = strides_vec[i - 1] * shape[i - 1]
    end
    strides = ntuple(i -> strides_vec[i], N)
    total_cells = strides_vec[N] * shape[N]
    # Avoid pathological memory blow-ups on very large grids; dict fallback remains available.
    max_lookup_cells = max(1_000_000, 16 * max_regions)
    cell_to_region = total_cells <= max_lookup_cells ? fill(0, total_cells) : nothing

    digits = zeros(Int, N)
    dirs = ones(Int, N)
    gvals = Vector{Int}(undef, N)
    lin_idx = 1
    @inbounds for i in 1:N
        gvals[i] = _slab_rep(coords[i], 0)
    end

    flat_unsat = zeros(Int, m)
    @inbounds for fidx in 1:m
        F = flats[fidx]
        c = 0
        for i in 1:N
            F.tau.coords[i] && continue
            c += (gvals[i] < F.b[i]) ? 1 : 0
        end
        flat_unsat[fidx] = c
    end

    inj_viol = zeros(Int, r)
    @inbounds for jidx in 1:r
        E = injectives[jidx]
        c = 0
        for i in 1:N
            E.tau.coords[i] && continue
            c += (gvals[i] >= E.b[i] + 1) ? 1 : 0
        end
        inj_viol[jidx] = c
    end

    y_words = zeros(UInt64, MY)
    z_words = zeros(UInt64, MZ)
    @inbounds for fidx in 1:m
        if flat_unsat[fidx] == 0
            _set_sigword_bit!(y_words, fidx, true)
        end
    end
    @inbounds for jidx in 1:r
        if inj_viol[jidx] > 0
            _set_sigword_bit!(z_words, jidx, true)
        end
    end

    seen = Dict{SigKey{MY,MZ},Int}()
    sig_y_cols = Vector{NTuple{MY,UInt64}}()
    sig_z_cols = Vector{NTuple{MZ,UInt64}}()
    reps  = Vector{NTuple{N,Int}}()

    function _record_state!()
        key = _sigkey_from_words(y_words, z_words, Val(MY), Val(MZ))
        rid = get(seen, key, 0)
        if rid == 0
            push!(sig_y_cols, ntuple(i -> y_words[i], Val(MY)))
            push!(sig_z_cols, ntuple(i -> z_words[i], Val(MZ)))
            push!(reps, ntuple(i -> gvals[i], N))
            rid = length(sig_y_cols)
            seen[key] = rid
            if length(sig_y_cols) > max_regions
                error("encode_poset_from_flanges: exceeded max_regions=$max_regions")
            end
        end
        if cell_to_region !== nothing
            @inbounds cell_to_region[lin_idx] = rid
        end
    end

    _record_state!()
    while true
        changed_axis = 0
        @inbounds for axis in 1:N
            nxt = digits[axis] + dirs[axis]
            if 0 <= nxt < shape[axis]
                old = digits[axis]
                digits[axis] = nxt
                _apply_signature_step!(axis, old, nxt, flat_events, inj_events,
                                       flat_unsat, inj_viol, y_words, z_words)
                gvals[axis] = _slab_rep(coords[axis], nxt)
                lin_idx += (nxt - old) * strides[axis]
                changed_axis = axis
                if axis > 1
                    for j in 1:(axis - 1)
                        dirs[j] = -dirs[j]
                    end
                end
                break
            end
        end
        changed_axis == 0 && break
        _record_state!()
    end

    sig_y = PackedSignatureRows{MY}(_word_matrix_from_cols(sig_y_cols, Val(MY)), m)
    sig_z = PackedSignatureRows{MZ}(_word_matrix_from_cols(sig_z_cols, Val(MZ)), r)
    return sig_y, sig_z, reps, seen, shape, strides, cell_to_region
end

@inline function _insert_sorted_unique!(v::Vector{Int}, x::Int)
    p = searchsortedfirst(v, x)
    if p > length(v) || v[p] != x
        insert!(v, p, x)
    end
    return nothing
end

@inline function _word_contains(words::AbstractVector{UInt64}, idx::Int)::Bool
    wd = ((idx - 1) >>> 6) + 1
    bit = UInt64(1) << ((idx - 1) & 63)
    return @inbounds (words[wd] & bit) != 0
end

function _decode_words_to_sorted!(dst::Vector{Int},
                                  words::AbstractVector{UInt64},
                                  nmax::Int,
                                  nactive::Int=0)
    empty!(dst)
    nactive > 0 && sizehint!(dst, nactive)
    @inbounds for wd in eachindex(words)
        w = words[wd]
        while w != 0
            tz = trailing_zeros(w)
            idx = ((wd - 1) << 6) + tz + 1
            idx <= nmax && push!(dst, idx)
            w &= w - UInt64(1)
        end
    end
    return dst
end

function _decode_word_additions!(dst::Vector{Int},
                                 old_words::AbstractVector{UInt64},
                                 new_words::AbstractVector{UInt64},
                                 nmax::Int)
    empty!(dst)
    sizehint!(dst, 8)
    @inbounds for wd in eachindex(new_words)
        w = new_words[wd] & ~old_words[wd]
        while w != 0
            tz = trailing_zeros(w)
            idx = ((wd - 1) << 6) + tz + 1
            idx <= nmax && push!(dst, idx)
            w &= w - UInt64(1)
        end
    end
    return dst
end

@inline function _active_basis_cols!(dst::Vector{Int},
                                     basis_cols::Vector{Int},
                                     col_words::AbstractVector{UInt64})
    empty!(dst)
    sizehint!(dst, length(basis_cols))
    @inbounds for col in basis_cols
        _word_contains(col_words, col) && push!(dst, col)
    end
    return dst
end

@inline function _basis_cols_removed(prev_basis_cols::Vector{Int},
                                     col_words::AbstractVector{UInt64})::Bool
    @inbounds for col in prev_basis_cols
        _word_contains(col_words, col) || return true
    end
    return false
end

@inline function _basis_symdiff_leq(a::Vector{Int}, b::Vector{Int}, limit::Int)::Bool
    i = 1
    j = 1
    na = length(a)
    nb = length(b)
    miss = 0
    @inbounds while i <= na && j <= nb
        ai = a[i]
        bj = b[j]
        if ai == bj
            i += 1
            j += 1
        elseif ai < bj
            miss += 1
            miss <= limit || return false
            i += 1
        else
            miss += 1
            miss <= limit || return false
            j += 1
        end
    end
    miss += (na - i + 1) + (nb - j + 1)
    return miss <= limit
end

@inline function _pack_transition_key(u::Int, v::Int)::UInt64
    return (UInt64(u) << 32) | UInt64(v)
end

function _build_box_membership_events(flats::Vector{IndFlat{N}},
                                      injectives::Vector{IndInj{N}},
                                      a::NTuple{N,Int},
                                      lens::NTuple{N,Int}) where {N}
    flat_fwd = [ [Int[] for _ in 1:lens[i]] for i in 1:N ]
    flat_bwd = [ [Int[] for _ in 1:lens[i]] for i in 1:N ]
    inj_fwd  = [ [Int[] for _ in 1:lens[i]] for i in 1:N ]
    inj_bwd  = [ [Int[] for _ in 1:lens[i]] for i in 1:N ]

    @inbounds for (fidx, F) in enumerate(flats)
        for i in 1:N
            F.tau.coords[i] && continue
            # forward: x -> x+1 activates at x+1 == b  => old digit = b-a-1
            df = F.b[i] - a[i] - 1
            if 0 <= df < lens[i] - 1
                push!(flat_fwd[i][df + 1], fidx)
            end
            # backward: x -> x-1 deactivates at x == b => old digit = b-a
            db = F.b[i] - a[i]
            if 1 <= db <= lens[i] - 1
                push!(flat_bwd[i][db + 1], fidx)
            end
        end
    end

    @inbounds for (jidx, E) in enumerate(injectives)
        for i in 1:N
            E.tau.coords[i] && continue
            # forward: x -> x+1 violates at x == b      => old digit = b-a
            df = E.b[i] - a[i]
            if 0 <= df < lens[i] - 1
                push!(inj_fwd[i][df + 1], jidx)
            end
            # backward: x -> x-1 unviolates at x == b+1 => old digit = b-a+1
            db = E.b[i] - a[i] + 1
            if 1 <= db <= lens[i] - 1
                push!(inj_bwd[i][db + 1], jidx)
            end
        end
    end

    return flat_fwd, flat_bwd, inj_fwd, inj_bwd
end

mutable struct _BoxBasisEntry{K}
    id::Int
    key::UInt64
    nrows::Int
    ncols::Int
    row_words::Vector{UInt64}
    col_words::Vector{UInt64}
    rows::Vector{Int}
    cols::Vector{Int}
    basis_cols::Vector{Int}
    coeffs::Union{Nothing,Matrix{K}}
    B::Matrix{K}
    dim::Int
end

"""
    ZnBoxBasisCache{K}

Reusable basis cache for repeated `pmodule_on_box` calls on the same flange.
This is a power-user hook; pass via `pmodule_on_box(...; cache=...)`.
"""
mutable struct ZnBoxBasisCache{K}
    flange_key::UInt64
    basis_cache::Dict{UInt64, Vector{_BoxBasisEntry{K}}}
    transition_cache::Dict{UInt64, Matrix{K}}
    next_entry_id::Int
    basis_cache_hits::Int
    basis_cache_misses::Int
    incremental_refinements::Int
    full_basis_recomputes::Int
    transition_cache_hits::Int
    transition_cache_misses::Int
    sparse_transition_solves::Int
    dense_transition_solves::Int
    identity_transition_fastpaths::Int
    coeff_transition_fastpaths::Int
    basis_change_transition_fastpaths::Int
end

# Internal benchmark/diagnostic override for transition solve routing in
# `pmodule_on_box`. `:auto` uses the field-sensitive heuristic, `:always`
# forces sparse solves, and `:never` forces dense solves.
const _ZN_TRANSITION_SPARSE_OVERRIDE = Ref{Symbol}(:auto)
const _ZN_BASIS_CHANGE_FASTPATH = Ref(true)

@inline function _transition_matrix_density(A::AbstractMatrix)
    total = length(A)
    total == 0 && return 0.0
    nz = 0
    @inbounds for x in A
        !iszero(x) && (nz += 1)
    end
    return nz / total
end

@inline function _should_sparse_transition_solve(::QQField,
                                                 Bv::AbstractMatrix,
                                                 Im::AbstractMatrix)
    workB = size(Bv, 1) * size(Bv, 2)
    workI = size(Im, 1) * size(Im, 2)
    if workB < 8192 || workI < 2048
        return false
    end
    if min(size(Bv, 1), size(Bv, 2)) < 32
        return false
    end
    return (_transition_matrix_density(Bv) <= 0.10) &&
           (_transition_matrix_density(Im) <= 0.10)
end

@inline function _should_sparse_transition_solve(F::PrimeField,
                                                 Bv::AbstractMatrix,
                                                 Im::AbstractMatrix)
    if F.p <= 3
        return false
    end
    workB = size(Bv, 1) * size(Bv, 2)
    workI = size(Im, 1) * size(Im, 2)
    densB = _transition_matrix_density(Bv)
    densI = _transition_matrix_density(Im)
    if workB < 256 || workI < 128
        return false
    end
    if min(size(Bv, 1), size(Bv, 2)) < 8
        return false
    end
    return densB <= 0.20 && densI <= 0.20
end

@inline _should_sparse_transition_solve(::RealField, Bv::AbstractMatrix, Im::AbstractMatrix) = false

@inline function _hash_word_state(words::AbstractVector{UInt64}, seed::UInt)
    h = seed
    @inbounds for w in words
        h = hash(w, h)
    end
    return UInt64(h)
end

@inline function _basis_pattern_fingerprint(row_words::AbstractVector{UInt64},
                                            nrows::Int,
                                            col_words::AbstractVector{UInt64},
                                            ncols::Int)
    h = hash(nrows)
    h = hash(length(row_words), h)
    h = _hash_word_state(row_words, h)
    h = hash(ncols, h)
    h = hash(length(col_words), h)
    h = _hash_word_state(col_words, h)
    return UInt64(h)
end

ZnBoxBasisCache{K}() where {K} = ZnBoxBasisCache{K}(0x0,
                                                    Dict{UInt64, Vector{_BoxBasisEntry{K}}}(),
                                                    Dict{UInt64, Matrix{K}}(),
                                                    1,
                                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

@inline function _flange_fingerprint(FG::Flange)
    h = hash(FG.n)
    @inbounds for F in FG.flats
        h = hash(_flat_key(F), h)
    end
    @inbounds for E in FG.injectives
        h = hash(_inj_key(E), h)
    end
    h = hash(size(FG.phi), h)
    return UInt64(h)
end

@inline function _flange_presentation_fingerprint(FG::Flange)
    h = hash(_flange_fingerprint(FG))
    # Include coefficient data so cached pushforwards remain correct when
    # generator sets are equal but presentation matrices differ.
    @inbounds for x in FG.phi
        h = hash(x, h)
    end
    return UInt64(h)
end

@inline function _hash_sorted_u64(vals::Vector{UInt64}, seed::UInt)
    sort!(vals)
    h = seed
    @inbounds for v in vals
        h = hash(v, h)
    end
    return UInt64(h)
end

const _GENERATOR_KEY = Tuple{Tuple,Tuple}

@inline function _generator_keyset_fingerprint(n::Int,
                                               flat_keys::AbstractVector{<:_GENERATOR_KEY},
                                               inj_keys::AbstractVector{<:_GENERATOR_KEY})
    flat_hashes = Vector{UInt64}(undef, length(flat_keys))
    inj_hashes = Vector{UInt64}(undef, length(inj_keys))
    @inbounds for i in eachindex(flat_keys)
        flat_hashes[i] = UInt64(hash(flat_keys[i]))
    end
    @inbounds for i in eachindex(inj_keys)
        inj_hashes[i] = UInt64(hash(inj_keys[i]))
    end
    h = hash(n)
    h = hash(length(flat_hashes), h)
    h = _hash_sorted_u64(flat_hashes, h)
    h = hash(length(inj_hashes), h)
    h = _hash_sorted_u64(inj_hashes, h)
    return UInt64(h)
end

@inline function _encoding_fingerprint(n::Int,
                                       flats::AbstractVector{<:IndFlat},
                                       injectives::AbstractVector{<:IndInj})
    flat_keys = Vector{_GENERATOR_KEY}(undef, length(flats))
    inj_keys = Vector{_GENERATOR_KEY}(undef, length(injectives))
    @inbounds for i in eachindex(flats)
        flat_keys[i] = _flat_key(flats[i])
    end
    @inbounds for i in eachindex(injectives)
        inj_keys[i] = _inj_key(injectives[i])
    end
    return _generator_keyset_fingerprint(n, flat_keys, inj_keys)
end

function compile_zn_box_cache(FG::Flange{K}) where {K}
    return ZnBoxBasisCache{K}(_flange_fingerprint(FG),
                              Dict{UInt64, Vector{_BoxBasisEntry{K}}}(),
                              Dict{UInt64, Matrix{K}}(),
                              1,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
end

"""
    box_cache_stats(cache::ZnBoxBasisCache)

Return a compact summary of `ZnBoxBasisCache` utilization counters and sizes.
Useful for profiling scripts that call `pmodule_on_box(...; cache=...)` repeatedly.
"""
function box_cache_stats(cache::ZnBoxBasisCache)
    basis_entries = 0
    @inbounds for bucket in values(cache.basis_cache)
        basis_entries += length(bucket)
    end
    btot = cache.basis_cache_hits + cache.basis_cache_misses
    ttot = cache.transition_cache_hits + cache.transition_cache_misses
    return (
        flange_key = cache.flange_key,
        basis_buckets = length(cache.basis_cache),
        basis_entries = basis_entries,
        transition_entries = length(cache.transition_cache),
        next_entry_id = cache.next_entry_id,
        basis_cache_hits = cache.basis_cache_hits,
        basis_cache_misses = cache.basis_cache_misses,
        basis_hit_rate = btot == 0 ? 0.0 : cache.basis_cache_hits / btot,
        incremental_refinements = cache.incremental_refinements,
        full_basis_recomputes = cache.full_basis_recomputes,
        transition_cache_hits = cache.transition_cache_hits,
        transition_cache_misses = cache.transition_cache_misses,
        transition_hit_rate = ttot == 0 ? 0.0 : cache.transition_cache_hits / ttot,
        sparse_transition_solves = cache.sparse_transition_solves,
        dense_transition_solves = cache.dense_transition_solves,
        identity_transition_fastpaths = cache.identity_transition_fastpaths,
        coeff_transition_fastpaths = cache.coeff_transition_fastpaths,
        basis_change_transition_fastpaths = cache.basis_change_transition_fastpaths,
    )
end

# Construct the PModule on the grid box induced by a flange presentation:
# M_g = im(Phi_g : F_g -> E_g), and maps are induced from the E-structure maps (projections).
#
# This returns a PModule over the finite grid poset. It is the object you want for Ext/Tor on that layer.
function pmodule_on_box(FG::Flange{K};
                        a::NTuple{N,Int},
                        b::NTuple{N,Int},
                        cache::Union{Nothing,ZnBoxBasisCache{K}}=nothing) where {K,N}
    Q, coords = grid_poset(a, b)
    ncoords = length(coords)

    r = length(FG.injectives)
    c = length(FG.flats)
    Phi = FG.phi
    field = FG.field
    lens = ntuple(i -> b[i] - a[i] + 1, N)

    # For each vertex, store active injective rows and image basis B_g.
    active_rows = Vector{Vector{Int}}(undef, ncoords)
    B = Vector{Matrix{K}}(undef, ncoords)
    dims = zeros(Int, ncoords)

    row_words_len = max(1, cld(r, 64))
    col_words_len = max(1, cld(c, 64))

    flat_fwd, flat_bwd, inj_fwd, inj_bwd = _build_box_membership_events(
        FG.flats, FG.injectives, a, lens
    )

    # Initialize counters at the lower corner g=a.
    flat_unsat = zeros(Int, c)
    @inbounds for fidx in 1:c
        F = FG.flats[fidx]
        cnt = 0
        for i in 1:N
            F.tau.coords[i] && continue
            cnt += (a[i] < F.b[i]) ? 1 : 0
        end
        flat_unsat[fidx] = cnt
    end

    inj_viol = zeros(Int, r)
    @inbounds for jidx in 1:r
        E = FG.injectives[jidx]
        cnt = 0
        for i in 1:N
            E.tau.coords[i] && continue
            cnt += (a[i] > E.b[i]) ? 1 : 0
        end
        inj_viol[jidx] = cnt
    end

    row_words = zeros(UInt64, row_words_len)
    col_words = zeros(UInt64, col_words_len)
    nrows_now = 0
    ncols_now = 0

    @inbounds for jidx in 1:r
        if inj_viol[jidx] == 0
            _set_sigword_bit!(row_words, jidx, true)
            nrows_now += 1
        end
    end
    @inbounds for fidx in 1:c
        if flat_unsat[fidx] == 0
            _set_sigword_bit!(col_words, fidx, true)
            ncols_now += 1
        end
    end

    # Cache B_g for repeated active-row/active-col patterns and transition solves.
    basis_cache = Dict{UInt64, Vector{_BoxBasisEntry{K}}}()
    transition_cache = Dict{UInt64, Matrix{K}}()
    next_entry_id_ref = Ref(1)
    basis_hits_ref = Ref(0)
    basis_misses_ref = Ref(0)
    incr_refines_ref = Ref(0)
    full_recompute_ref = Ref(0)
    transition_hits_ref = Ref(0)
    transition_misses_ref = Ref(0)
    sparse_solves_ref = Ref(0)
    dense_solves_ref = Ref(0)
    identity_fastpaths_ref = Ref(0)
    coeff_fastpaths_ref = Ref(0)
    basis_change_fastpaths_ref = Ref(0)
    if cache === nothing
        basis_cache = Dict{UInt64, Vector{_BoxBasisEntry{K}}}()
        transition_cache = Dict{UInt64, Matrix{K}}()
        next_entry_id_ref[] = 1
    else
        fkey = _flange_fingerprint(FG)
        if cache.flange_key != fkey
            cache.flange_key = fkey
            empty!(cache.basis_cache)
            empty!(cache.transition_cache)
            cache.next_entry_id = 1
            cache.basis_cache_hits = 0
            cache.basis_cache_misses = 0
            cache.incremental_refinements = 0
            cache.full_basis_recomputes = 0
            cache.transition_cache_hits = 0
            cache.transition_cache_misses = 0
            cache.sparse_transition_solves = 0
            cache.dense_transition_solves = 0
            cache.identity_transition_fastpaths = 0
            cache.coeff_transition_fastpaths = 0
            cache.basis_change_transition_fastpaths = 0
        end
        basis_cache = cache.basis_cache
        transition_cache = cache.transition_cache
        next_entry_id_ref[] = cache.next_entry_id
    end

    @inline function _find_basis_entry(hashkey::UInt64)
        bucket = get(basis_cache, hashkey, nothing)
        bucket === nothing && return nothing
        @inbounds for entry in bucket
            if entry.key == hashkey &&
               entry.nrows == nrows_now &&
               entry.ncols == ncols_now &&
               entry.row_words == row_words &&
               entry.col_words == col_words
                return entry
            end
        end
        return nothing
    end

    @inline function _store_basis_entry!(hashkey::UInt64,
                                         rows_state::Vector{Int},
                                         cols_state::Vector{Int},
                                         Bg::Matrix{K},
                                         basis_cols::Vector{Int},
                                         coeffs::Union{Nothing,Matrix{K}},
                                         dim::Int)
        eid = next_entry_id_ref[]
        next_entry_id_ref[] = eid + 1
        entry = _BoxBasisEntry{K}(eid, hashkey, length(rows_state), length(cols_state),
                                  copy(row_words), copy(col_words),
                                  copy(rows_state), copy(cols_state),
                                  copy(basis_cols), coeffs, Bg, dim)
        push!(get!(basis_cache, hashkey, Vector{_BoxBasisEntry{K}}()), entry)
        return entry
    end

    @inline function _compute_basis_from_words(curr_rows::Vector{Int},
                                               row_words_state::AbstractVector{UInt64},
                                               col_words_state::AbstractVector{UInt64},
                                               nrows_state::Int,
                                               ncols_state::Int)
        if nrows_state == 0 || ncols_state == 0
            return zeros(K, nrows_state, 0), Int[]
        end
        Bg, basis_cols = FieldLinAlg.colspace_restricted_words(field, Phi, row_words_state, col_words_state,
                                                               nrows_state, ncols_state;
                                                               nrows=r, ncols=c, backend=:auto,
                                                               pivots=true)
        return Bg, basis_cols
    end

    @inline function _compute_coeffs(row_words_state::AbstractVector{UInt64},
                                     col_words_state::AbstractVector{UInt64},
                                     nrows_state::Int,
                                     ncols_state::Int,
                                     Bg::Matrix{K})
        d = size(Bg, 2)
        if d == 0 || ncols_state == 0
            return zeros(K, d, ncols_state)
        end
        return FieldLinAlg.solve_fullcolumn_restricted_words(field, Bg, Phi,
                                                             row_words_state, col_words_state,
                                                             nrows_state, ncols_state;
                                                             nrows=r, ncols=c,
                                                             check_rhs=false, backend=:auto)
    end

    had_prev = false
    prev_entry::Union{Nothing,_BoxBasisEntry{K}} = nothing
    curr_rows = Int[]
    curr_cols = Int[]
    row_added = Int[]
    col_added = Int[]
    candidate_cols = Int[]

    strides = Vector{Int}(undef, N)
    strides[1] = 1
    @inbounds for i in 2:N
        strides[i] = strides[i - 1] * lens[i - 1]
    end
    digits = zeros(Int, N)
    dirs = ones(Int, N)
    lin_idx = 1
    basis_entry_ids = zeros(Int, ncoords)
    basis_entries = Vector{_BoxBasisEntry{K}}(undef, ncoords)

    @inline function _basis_cols_active_in_entry(entry::_BoxBasisEntry{K},
                                                 basis_cols::Vector{Int}) where {K}
        @inbounds for colid in basis_cols
            _word_contains(entry.col_words, colid) || return false
        end
        return true
    end

    @inline function _transition_from_direct_coeffs(entry_src::_BoxBasisEntry{K},
                                                    entry_tgt::_BoxBasisEntry{K};
                                                    require_cached::Bool=false) where {K}
        _basis_cols_active_in_entry(entry_tgt, entry_src.basis_cols) || return nothing
        if require_cached && entry_tgt.coeffs === nothing
            return nothing
        end
        if entry_tgt.coeffs === nothing
            entry_tgt.coeffs = _compute_coeffs(entry_tgt.row_words, entry_tgt.col_words,
                                               entry_tgt.nrows, entry_tgt.ncols,
                                               entry_tgt.B)
        end
        coeffs = entry_tgt.coeffs::Matrix{K}
        dv = size(coeffs, 1)
        du = length(entry_src.basis_cols)
        X = Matrix{K}(undef, dv, du)
        @inbounds for j in 1:du
            colid = entry_src.basis_cols[j]
            t = searchsortedfirst(entry_tgt.cols, colid)
            if t > length(entry_tgt.cols) || entry_tgt.cols[t] != colid
                return nothing
            end
            @views X[:, j] .= coeffs[:, t]
        end
        return X
    end

    @inline function _transition_from_small_basis_change(entry_src::_BoxBasisEntry{K},
                                                         entry_tgt::_BoxBasisEntry{K}) where {K}
        du = length(entry_src.basis_cols)
        dv = length(entry_tgt.basis_cols)
        du == dv || return nothing
        _basis_symdiff_leq(entry_src.basis_cols, entry_tgt.basis_cols, 2) || return nothing

        C = zeros(K, du, du)
        added_tgt = Int[]
        added_pos = Int[]
        sizehint!(added_tgt, 2)
        sizehint!(added_pos, 2)
        @inbounds for j in 1:du
            colid = entry_tgt.basis_cols[j]
            p = searchsortedfirst(entry_src.basis_cols, colid)
            if p <= du && entry_src.basis_cols[p] == colid
                C[p, j] = one(K)
            else
                push!(added_tgt, colid)
                push!(added_pos, j)
            end
        end
        isempty(added_tgt) && return eye(field, du)

        rhs = Matrix{K}(undef, entry_src.nrows, length(added_tgt))
        @inbounds for k in eachindex(added_tgt)
            j = added_pos[k]
            @views rhs[:, k] .= entry_tgt.B[:, j]
        end
        Y = FieldLinAlg.solve_fullcolumn(field, entry_src.B, rhs; check_rhs=false)
        @inbounds for k in eachindex(added_pos)
            @views C[:, added_pos[k]] .= Y[:, k]
        end
        return FieldLinAlg.solve_fullcolumn(field, C, eye(field, du); check_rhs=false)
    end

    @inbounds for _ in 1:ncoords
        h = _basis_pattern_fingerprint(row_words, nrows_now, col_words, ncols_now)
        entry = _find_basis_entry(h)
        if entry === nothing
            basis_misses_ref[] += 1
            if had_prev && prev_entry !== nothing
                rows_unchanged = prev_entry.row_words == row_words
                _decode_word_additions!(col_added, prev_entry.col_words, col_words, c)
                basis_removed = _basis_cols_removed(prev_entry.basis_cols, col_words)

                if !basis_removed
                    _decode_words_to_sorted!(curr_cols, col_words, c, ncols_now)

                    # True small-change fast path: rows unchanged and only nonbasis columns disappeared.
                    if rows_unchanged && isempty(col_added)
                        incr_refines_ref[] += 1
                        entry = _store_basis_entry!(h, prev_entry.rows, curr_cols,
                                                    prev_entry.B, prev_entry.basis_cols,
                                                    nothing, prev_entry.dim)
                    else
                        rows_state = prev_entry.rows
                        if !rows_unchanged
                            _decode_words_to_sorted!(curr_rows, row_words, r, nrows_now)
                            rows_state = curr_rows
                            _decode_word_additions!(row_added, prev_entry.row_words, row_words, r)
                        end

                        _active_basis_cols!(candidate_cols, prev_entry.basis_cols, col_words)
                        @inbounds for colid in col_added
                            _insert_sorted_unique!(candidate_cols, colid)
                        end

                        # If rows are added, previously dependent nonbasis columns may
                        # become independent. Only revisit those columns in that case.
                        if !rows_unchanged && !isempty(row_added)
                            if prev_entry.coeffs === nothing
                                prev_entry.coeffs = _compute_coeffs(prev_entry.row_words, prev_entry.col_words,
                                                                    prev_entry.nrows, prev_entry.ncols,
                                                                    prev_entry.B)
                            end
                            coeffs_prev = prev_entry.coeffs::Matrix{K}
                            prev_cols_state = prev_entry.cols
                            dprev = size(coeffs_prev, 1)
                            nbasis = length(prev_entry.basis_cols)
                            @inbounds for t in eachindex(prev_cols_state)
                                colid = prev_cols_state[t]
                                _word_contains(col_words, colid) || continue
                                q = searchsortedfirst(candidate_cols, colid)
                                if q <= length(candidate_cols) && candidate_cols[q] == colid
                                    continue
                                end

                                violated = false
                                for rr in row_added
                                    lhs = Phi[rr, colid]
                                    rhs = zero(K)
                                    kmax = min(dprev, nbasis)
                                    for k in 1:kmax
                                        rhs += coeffs_prev[k, t] * Phi[rr, prev_entry.basis_cols[k]]
                                    end
                                    if lhs != rhs
                                        violated = true
                                        break
                                    end
                                end
                                violated && _insert_sorted_unique!(candidate_cols, colid)
                            end
                        end

                        Bg, basis_cols = _compute_basis_from_words(rows_state, row_words, col_words,
                                                                   length(rows_state), ncols_now)
                        incr_refines_ref[] += 1
                        entry = _store_basis_entry!(h, rows_state, curr_cols,
                                                    Bg, basis_cols, nothing, size(Bg, 2))
                    end
                end
            end

            if entry === nothing
                _decode_words_to_sorted!(curr_rows, row_words, r, nrows_now)
                _decode_words_to_sorted!(curr_cols, col_words, c, ncols_now)
                Bg, basis_cols = _compute_basis_from_words(curr_rows, row_words, col_words,
                                                           nrows_now, ncols_now)
                full_recompute_ref[] += 1
                entry = _store_basis_entry!(h, curr_rows, curr_cols,
                                            Bg, basis_cols, nothing, size(Bg, 2))
            end
        else
            basis_hits_ref[] += 1
        end

        active_rows[lin_idx] = entry.rows
        B[lin_idx] = entry.B
        dims[lin_idx] = entry.dim
        basis_entry_ids[lin_idx] = entry.id
        basis_entries[lin_idx] = entry

        if !had_prev
            had_prev = true
        end
        prev_entry = entry

        moved = false
        for axis in 1:N
            nxt = digits[axis] + dirs[axis]
            if 0 <= nxt < lens[axis]
                old = digits[axis]
                digits[axis] = nxt

                if dirs[axis] == 1
                    for f in flat_fwd[axis][old + 1]
                        flat_unsat[f] -= 1
                        if flat_unsat[f] == 0
                            _set_sigword_bit!(col_words, f, true)
                            ncols_now += 1
                        end
                    end
                    for j in inj_fwd[axis][old + 1]
                        inj_viol[j] += 1
                        if inj_viol[j] == 1
                            _set_sigword_bit!(row_words, j, false)
                            nrows_now -= 1
                        end
                    end
                else
                    for f in flat_bwd[axis][old + 1]
                        was_active = flat_unsat[f] == 0
                        flat_unsat[f] += 1
                        if was_active
                            _set_sigword_bit!(col_words, f, false)
                            ncols_now -= 1
                        end
                    end
                    for j in inj_bwd[axis][old + 1]
                        inj_viol[j] -= 1
                        if inj_viol[j] == 0
                            _set_sigword_bit!(row_words, j, true)
                            nrows_now += 1
                        end
                    end
                end

                lin_idx += dirs[axis] * strides[axis]
                moved = true
                if axis > 1
                    for j in 1:(axis - 1)
                        dirs[j] = -dirs[j]
                    end
                end
                break
            end
        end
        moved || break
    end

    # Build edge maps along cover edges in the grid poset using induced maps from E (projection).
    edge_maps = Dict{Tuple{Int, Int}, Matrix{K}}()

    # Gather rows_h from rows_g directly (rows_h subset rows_g), avoiding Pu * B[u].
    function gather_projection_rows(Bu::Matrix{K}, rows_g::Vector{Int}, rows_h::Vector{Int})
        Ph = length(rows_h)
        du = size(Bu, 2)
        out = Matrix{K}(undef, Ph, du)
        Pg = length(rows_g)
        j = 1
        @inbounds for i in 1:Ph
            target = rows_h[i]
            while j <= Pg && rows_g[j] < target
                j += 1
            end
            if j > Pg || rows_g[j] != target
                error("pmodule_on_box: projection mismatch; expected rows_h subset rows_g")
            end
            for c in 1:du
                out[i, c] = Bu[j, c]
            end
        end
        return out
    end

    @inline function should_sparse_solve(Bv::Matrix{K}, Im::Matrix{K}) where {K}
        mode = _ZN_TRANSITION_SPARSE_OVERRIDE[]
        mode === :always && return true
        mode === :never && return false
        return _should_sparse_transition_solve(field, Bv, Im)
    end

    edges = collect(cover_edges(Q))
    ne = length(edges)
    slot_of_edge = Vector{Int}(undef, ne)
    slot_keys = UInt64[]
    slot_rep_edge = Int[]
    slot_index = Dict{UInt64,Int}()

    @inbounds for ei in 1:ne
        u, v = edges[ei]
        key = _pack_transition_key(basis_entry_ids[u], basis_entry_ids[v])
        s = get(slot_index, key, 0)
        if s == 0
            s = length(slot_keys) + 1
            slot_index[key] = s
            push!(slot_keys, key)
            push!(slot_rep_edge, ei)
        end
        slot_of_edge[ei] = s
    end

    nslots = length(slot_keys)
    transitions = Vector{Matrix{K}}(undef, nslots)
    missing_slots = Int[]
    sizehint!(missing_slots, nslots)

    @inbounds for s in 1:nslots
        key = slot_keys[s]
        X = get(transition_cache, key, nothing)
        if X === nothing
            push!(missing_slots, s)
            transition_misses_ref[] += 1
        else
            transitions[s] = X
            transition_hits_ref[] += 1
        end
    end

    function solve_transition_slot(s::Int)
        ei = slot_rep_edge[s]
        u, v = edges[ei]
        du = dims[u]
        dv = dims[v]
        if dv == 0 || du == 0
            return zeros(K, dv, du)
        end

        entry_u = basis_entries[u]
        entry_v = basis_entries[v]
        if entry_u.id == entry_v.id
            identity_fastpaths_ref[] += 1
            return eye(field, du)
        end
        same_rows = entry_u.nrows == entry_v.nrows &&
                    entry_u.row_words == entry_v.row_words
        if same_rows
            X = _transition_from_direct_coeffs(entry_u, entry_v; require_cached=true)
            if X !== nothing
                coeff_fastpaths_ref[] += 1
                return X
            end
            if _ZN_BASIS_CHANGE_FASTPATH[]
                X = _transition_from_small_basis_change(entry_u, entry_v)
                if X !== nothing
                    basis_change_fastpaths_ref[] += 1
                    return X
                end
            end
            X = _transition_from_direct_coeffs(entry_u, entry_v)
            if X !== nothing
                coeff_fastpaths_ref[] += 1
                return X
            end
        end

        rows_u = active_rows[u]
        rows_v = active_rows[v]
        Im = gather_projection_rows(B[u], rows_u, rows_v)  # E_v x du
        Bv = B[v]

        if should_sparse_solve(Bv, Im)
            sparse_solves_ref[] += 1
            return FieldLinAlg.solve_fullcolumn(field, sparse(Bv), sparse(Im))
        end
        dense_solves_ref[] += 1
        return FieldLinAlg.solve_fullcolumn(field, Bv, Im)
    end

    if !isempty(missing_slots)
        if Threads.nthreads() > 1 && length(missing_slots) >= 32
            Threads.@threads :static for mi in eachindex(missing_slots)
                s = missing_slots[mi]
                transitions[s] = solve_transition_slot(s)
            end
        else
            @inbounds for s in missing_slots
                transitions[s] = solve_transition_slot(s)
            end
        end
        @inbounds for s in missing_slots
            transition_cache[slot_keys[s]] = transitions[s]
        end
    end

    edge_values = Vector{Matrix{K}}(undef, ne)
    if Threads.nthreads() > 1 && ne >= 256
        Threads.@threads :static for ei in 1:ne
            edge_values[ei] = transitions[slot_of_edge[ei]]
        end
    else
        @inbounds for ei in 1:ne
            edge_values[ei] = transitions[slot_of_edge[ei]]
        end
    end

    @inbounds for ei in 1:ne
        edge_maps[edges[ei]] = edge_values[ei]
    end

    if cache !== nothing
        cache.next_entry_id = next_entry_id_ref[]
        cache.basis_cache_hits += basis_hits_ref[]
        cache.basis_cache_misses += basis_misses_ref[]
        cache.incremental_refinements += incr_refines_ref[]
        cache.full_basis_recomputes += full_recompute_ref[]
        cache.transition_cache_hits += transition_hits_ref[]
        cache.transition_cache_misses += transition_misses_ref[]
        cache.sparse_transition_solves += sparse_solves_ref[]
        cache.dense_transition_solves += dense_solves_ref[]
        cache.identity_transition_fastpaths += identity_fastpaths_ref[]
        cache.coeff_transition_fastpaths += coeff_fastpaths_ref[]
        cache.basis_change_transition_fastpaths += basis_change_fastpaths_ref[]
    end

    return PModule{K}(Q, Vector{Int}(dims), edge_maps; field=field)
end

# =============================================================================
# Miller-style finite encoding for Z^n (without enumerating lattice points)
# ...
# =============================================================================

struct ZnPushforwardPlan
    flat_idxs::Vector{Int}
    inj_idxs::Vector{Int}
    zero_pairs::Vector{Tuple{Int,Int}}
end

mutable struct ZnPushforwardCache
    lock::Base.ReentrantLock
    flat_index::Union{Nothing,Dict{_GENERATOR_KEY,Int}}
    inj_index::Union{Nothing,Dict{_GENERATOR_KEY,Int}}
    flat_masks::Union{Nothing,Vector{BitVector}}
    inj_masks::Union{Nothing,Vector{BitVector}}
    plan_by_flange::Dict{UInt64,ZnPushforwardPlan}
    ZnPushforwardCache() = new(Base.ReentrantLock(), nothing, nothing, nothing, nothing,
                               Dict{UInt64,ZnPushforwardPlan}())
end

"""
    ZnEncodingMap

A classifier `pi : Z^n -> P` produced by `encode_from_flange` or
`encode_from_flanges`.

The target poset `P` is the uptight poset on (y,z)-signatures, where

* `y_i(g) = 1` means the point `g` lies in the `i`-th flat (an upset).
* `z_j(g) = 1` means the point `g` lies in the complement of the `j`-th
  injective (also an upset, since `Z^n` is discrete).

Fields
* `n`              : ambient dimension
* `coords[i]`      : sorted unique critical integers along axis i
* `sig_y[t]`       : y-signature view for region t (Bool vector, one per flat)
* `sig_z[t]`       : z-signature view for region t (Bool vector, one per injective)
* `reps[t]`        : representative lattice point for region t (as an `NTuple`)
* `flats`          : the global flat list used to build signatures
* `injectives`     : the global injective list used to build signatures
* `sig_to_region`  : dictionary mapping a signature key to its region index
* `cell_shape`     : slab-cell shape `(length(coords[i]) + 1)_i`
* `cell_strides`   : mixed-radix strides for slab-cell linear indexing
* `cell_to_region` : optional direct slab-cell->region lookup table

The method `locate(pi, g)` returns the region index in `1:P.n` for the point
`g`, or `0` if the signature is not present in the dictionary.
"""
struct ZnEncodingMap{N,MY,MZ} <: AbstractPLikeEncodingMap
    n::Int
    coords::NTuple{N,Vector{Int}}
    sig_y::PackedSignatureRows{MY}
    sig_z::PackedSignatureRows{MZ}
    reps::Vector{NTuple{N,Int}}
    flats::Vector{IndFlat{N}}
    injectives::Vector{IndInj{N}}
    encoding_fingerprint::UInt64
    sig_to_region::Dict{SigKey{MY,MZ},Int}
    cell_shape::NTuple{N,Int}
    cell_strides::NTuple{N,Int}
    cell_to_region::Union{Nothing,Vector{Int}}
    pushforward_cache::ZnPushforwardCache
end

function ZnEncodingMap(n::Int,
                       coords::NTuple{N,Vector{Int}},
                       sig_y::Union{PackedSignatureRows{MY},AbstractVector{<:AbstractVector{Bool}}},
                       sig_z::Union{PackedSignatureRows{MZ},AbstractVector{<:AbstractVector{Bool}}},
                       reps::Vector{NTuple{N,Int}},
                       flats::Vector{IndFlat{N}},
                       injectives::Vector{IndInj{N}},
                       sig_to_region::Dict{SigKey{MY,MZ},Int}) where {N,MY,MZ}
    n == N || error("ZnEncodingMap: n=$n does not match reps tuple length $N")
    nregions = length(reps)
    packed_y = _pack_signature_rows(sig_y, length(flats), Val(MY))
    packed_z = _pack_signature_rows(sig_z, length(injectives), Val(MZ))
    length(packed_y) == nregions || error("ZnEncodingMap: sig_y region count mismatch")
    length(packed_z) == nregions || error("ZnEncodingMap: sig_z region count mismatch")
    enc_key = _encoding_fingerprint(n, flats, injectives)
    shape = ntuple(i -> length(coords[i]) + 1, N)
    strides_vec = Vector{Int}(undef, N)
    strides_vec[1] = 1
    @inbounds for i in 2:N
        strides_vec[i] = strides_vec[i - 1] * shape[i - 1]
    end
    strides = ntuple(i -> strides_vec[i], N)
    return ZnEncodingMap{N,MY,MZ}(n, coords, packed_y, packed_z, reps, flats, injectives, enc_key, sig_to_region,
                                  shape, strides, nothing, ZnPushforwardCache())
end

function ZnEncodingMap(n::Int,
                       coords::NTuple{N,Vector{Int}},
                       sig_y::Union{PackedSignatureRows{MY},AbstractVector{<:AbstractVector{Bool}}},
                       sig_z::Union{PackedSignatureRows{MZ},AbstractVector{<:AbstractVector{Bool}}},
                       reps::Vector{NTuple{N,Int}},
                       flats::Vector{IndFlat{N}},
                       injectives::Vector{IndInj{N}},
                       sig_to_region::Dict{SigKey{MY,MZ},Int},
                       cell_shape::NTuple{N,Int},
                       cell_strides::NTuple{N,Int},
                       cell_to_region::Union{Nothing,Vector{Int}}) where {N,MY,MZ}
    n == N || error("ZnEncodingMap: n=$n does not match reps tuple length $N")
    nregions = length(reps)
    packed_y = _pack_signature_rows(sig_y, length(flats), Val(MY))
    packed_z = _pack_signature_rows(sig_z, length(injectives), Val(MZ))
    length(packed_y) == nregions || error("ZnEncodingMap: sig_y region count mismatch")
    length(packed_z) == nregions || error("ZnEncodingMap: sig_z region count mismatch")
    enc_key = _encoding_fingerprint(n, flats, injectives)
    return ZnEncodingMap{N,MY,MZ}(n, coords, packed_y, packed_z, reps, flats, injectives, enc_key, sig_to_region,
                                  cell_shape, cell_strides, cell_to_region, ZnPushforwardCache())
end

"""
    ZnEncodingCache

Advanced cache handle for repeated Zn encoding queries. This is an opt-in
power-user API; regular workflow users should keep using `cache=:auto` or
`SessionCache` at the workflow layer.
"""
struct ZnEncodingCache{N,MY,MZ,PType}
    P::PType
    pi::ZnEncodingMap{N,MY,MZ}
end

@inline _unwrap_zn_pi(pi::ZnEncodingMap) = pi
@inline _unwrap_zn_pi(cache::ZnEncodingCache) = cache.pi
@inline _unwrap_zn_pi(enc::CompiledEncoding{<:ZnEncodingMap}) = enc.pi

"""
    compile_zn_cache(P, pi)
    compile_zn_cache(pi)
    compile_zn_cache(FGs, opts=EncodingOptions(); poset_kind=:signature)

Construct a reusable owner-local cache for repeated Zn encoding queries.

The returned [`ZnEncodingCache`](@ref) stores a finite region poset together
with the underlying [`ZnEncodingMap`](@ref). This is the owner-level cache to
use when you plan to call [`locate`](@ref), [`locate_many!`](@ref),
[`region_weights`](@ref), or [`region_adjacency`](@ref) repeatedly on the same
encoding.

Best practice:
- use `compile_zn_cache(pi)` for a bare map when the structured
  `:signature` region poset is sufficient;
- use `compile_zn_cache(P, pi)` when you already have a specific finite region
  poset, including dense `FinitePoset` realizations;
- use `poset_kind=:signature` for the cheapest structured cache and
  `poset_kind=:dense` only when downstream code truly needs a dense poset.
"""
compile_zn_cache(P, pi::ZnEncodingMap{N,MY,MZ}) where {N,MY,MZ} = ZnEncodingCache{N,MY,MZ,typeof(P)}(P, pi)
compile_zn_cache(pi::ZnEncodingMap) = ZnEncodingCache(SignaturePoset(pi.sig_y, pi.sig_z), pi)
compile_zn_cache(enc::CompiledEncoding{<:ZnEncodingMap}) = ZnEncodingCache(enc.P, enc.pi)

function compile_zn_cache(FGs::Union{AbstractVector{<:Flange}, Tuple{Vararg{Flange}}},
                          opts::EncodingOptions=EncodingOptions();
                          poset_kind::Symbol=:signature)
    P, pi = encode_poset_from_flanges(FGs, opts; poset_kind=poset_kind)
    return ZnEncodingCache(P, pi)
end

compile_zn_cache(FG::Flange, opts::EncodingOptions=EncodingOptions(); poset_kind::Symbol=:signature) =
    compile_zn_cache((FG,), opts; poset_kind=poset_kind)

compile_zn_cache(FG1::Flange, FG2::Flange, opts::EncodingOptions=EncodingOptions(); poset_kind::Symbol=:signature) =
    compile_zn_cache((FG1, FG2), opts; poset_kind=poset_kind)

compile_zn_cache(FG1::Flange, FG2::Flange, FG3::Flange, opts::EncodingOptions=EncodingOptions(); poset_kind::Symbol=:signature) =
    compile_zn_cache((FG1, FG2, FG3), opts; poset_kind=poset_kind)

"""
    SignaturePoset(sig_y, sig_z)

Structured poset on region signatures with order defined by componentwise inclusion:
`i <= j` iff `sig_y[i] <= sig_y[j]` and `sig_z[i] <= sig_z[j]`.
"""
struct SignaturePoset{MY,MZ} <: AbstractPoset
    sig_y::PackedSignatureRows{MY}
    sig_z::PackedSignatureRows{MZ}
    n::Int
    y_lastmask::UInt64
    z_lastmask::UInt64
    cache::PosetCache
end

function SignaturePoset(sig_y::PackedSignatureRows{MY},
                        sig_z::PackedSignatureRows{MZ}) where {MY,MZ}
    length(sig_y) == length(sig_z) || error("SignaturePoset: sig_y and sig_z length mismatch")
    return SignaturePoset{MY,MZ}(sig_y, sig_z, length(sig_y),
                                 _word_lastmask(sig_y.bitlen),
                                 _word_lastmask(sig_z.bitlen),
                                 PosetCache())
end

function SignaturePoset(sig_y::AbstractVector{<:AbstractVector{Bool}},
                        sig_z::AbstractVector{<:AbstractVector{Bool}})
    length(sig_y) == length(sig_z) || error("SignaturePoset: sig_y and sig_z length mismatch")
    n = length(sig_y)
    m = n == 0 ? 0 : length(sig_y[1])
    r = n == 0 ? 0 : length(sig_z[1])
    my = cld(max(m, 1), 64)
    mz = cld(max(r, 1), 64)
    py = _pack_signature_rows(sig_y, m, Val(my))
    pz = _pack_signature_rows(sig_z, r, Val(mz))
    return SignaturePoset(py, pz)
end

@inline function _sig_subset(a::BitVector, b::BitVector)::Bool
    length(a) == length(b) || error("SignaturePoset: signature length mismatch")
    ac = a.chunks
    bc = b.chunks
    nchunks = length(ac)
    r = length(a) & 63
    lastmask = (r == 0) ? typemax(UInt64) : (UInt64(1) << r) - 1
    @inbounds for w in 1:nchunks
        diff = ac[w] & ~bc[w]
        if w == nchunks
            diff &= lastmask
        end
        if diff != 0
            return false
        end
    end
    return true
end

@inline function _sig_popcount(sig::BitVector)::Int
    chunks = sig.chunks
    nchunks = length(chunks)
    nchunks == 0 && return 0
    r = length(sig) & 63
    lastmask = (r == 0) ? typemax(UInt64) : (UInt64(1) << r) - 1
    c = 0
    @inbounds for w in 1:nchunks
        word = chunks[w]
        if w == nchunks
            word &= lastmask
        end
        c += count_ones(word)
    end
    return c
end

nvertices(P::SignaturePoset) = P.n
leq(P::SignaturePoset, i::Int, j::Int) =
    _sig_subset_words(P.sig_y.words, i, P.sig_y.words, j, P.y_lastmask) &&
    _sig_subset_words(P.sig_z.words, i, P.sig_z.words, j, P.z_lastmask)

# SignaturePoset should avoid generic n^2 up/down cache materialization.
FiniteFringe._updown_cache_skip_auto(::SignaturePoset) = true

@inline function _sig_row_words(words::Matrix{UInt64}, idx::Int, ::Val{NW}) where {NW}
    return ntuple(w -> @inbounds(words[w, idx]), NW)
end

@inline function _sig_row_subset_col_words(row::NTuple{NW,UInt64},
                                           words::Matrix{UInt64},
                                           idx::Int,
                                           lastmask::UInt64)::Bool where {NW}
    @inbounds for w in 1:NW
        diff = row[w] & ~words[w, idx]
        if w == NW
            diff &= lastmask
        end
        diff == 0 || return false
    end
    return true
end

@inline function _sig_col_subset_row_words(words::Matrix{UInt64},
                                           idx::Int,
                                           row::NTuple{NW,UInt64},
                                           lastmask::UInt64)::Bool where {NW}
    @inbounds for w in 1:NW
        diff = words[w, idx] & ~row[w]
        if w == NW
            diff &= lastmask
        end
        diff == 0 || return false
    end
    return true
end

@inline function _sig_tuple_subset(a::NTuple{NW,UInt64},
                                   b::NTuple{NW,UInt64},
                                   lastmask::UInt64)::Bool where {NW}
    @inbounds for w in 1:NW
        diff = a[w] & ~b[w]
        if w == NW
            diff &= lastmask
        end
        diff == 0 || return false
    end
    return true
end

function upset_indices(P::SignaturePoset{MY,MZ}, i::Int) where {MY,MZ}
    cached = FiniteFringe._cached_upset_indices(P, i)
    cached === nothing || return FiniteFringe.IndicesView(cached)
    yrow = _sig_row_words(P.sig_y.words, i, Val(MY))
    zrow = _sig_row_words(P.sig_z.words, i, Val(MZ))
    return _SignatureLeqIter{MY,MZ}(P.sig_y.words, P.sig_z.words, yrow, zrow,
                                    P.y_lastmask, P.z_lastmask, P.n, true)
end

function downset_indices(P::SignaturePoset{MY,MZ}, i::Int) where {MY,MZ}
    cached = FiniteFringe._cached_downset_indices(P, i)
    cached === nothing || return FiniteFringe.IndicesView(cached)
    yrow = _sig_row_words(P.sig_y.words, i, Val(MY))
    zrow = _sig_row_words(P.sig_z.words, i, Val(MZ))
    return _SignatureLeqIter{MY,MZ}(P.sig_y.words, P.sig_z.words, yrow, zrow,
                                    P.y_lastmask, P.z_lastmask, P.n, false)
end

leq_row(P::SignaturePoset, i::Int) = upset_indices(P, i)
leq_col(P::SignaturePoset, j::Int) = downset_indices(P, j)

@inline function _sig_leq_pair(P::SignaturePoset, i::Int, j::Int)::Bool
    return _sig_subset_words(P.sig_y.words, i, P.sig_y.words, j, P.y_lastmask) &&
           _sig_subset_words(P.sig_z.words, i, P.sig_z.words, j, P.z_lastmask)
end

struct _SignatureLeqIter{MY,MZ}
    y_words::Matrix{UInt64}
    z_words::Matrix{UInt64}
    y_row::NTuple{MY,UInt64}
    z_row::NTuple{MZ,UInt64}
    y_lastmask::UInt64
    z_lastmask::UInt64
    n::Int
    upset::Bool
end

Base.IteratorSize(::Type{<:_SignatureLeqIter}) = Base.HasLength()
Base.eltype(::Type{<:_SignatureLeqIter}) = Int

function Base.length(it::_SignatureLeqIter{MY,MZ}) where {MY,MZ}
    n = it.n
    cnt = 0
    if it.upset
        @inbounds for j in 1:n
            cnt += (_sig_row_subset_col_words(it.y_row, it.y_words, j, it.y_lastmask) &&
                    _sig_row_subset_col_words(it.z_row, it.z_words, j, it.z_lastmask)) ? 1 : 0
        end
    else
        @inbounds for j in 1:n
            cnt += (_sig_col_subset_row_words(it.y_words, j, it.y_row, it.y_lastmask) &&
                    _sig_col_subset_row_words(it.z_words, j, it.z_row, it.z_lastmask)) ? 1 : 0
        end
    end
    return cnt
end

function Base.iterate(it::_SignatureLeqIter{MY,MZ}, state::Int=1) where {MY,MZ}
    n = it.n
    if it.upset
        @inbounds for j in state:n
            if _sig_row_subset_col_words(it.y_row, it.y_words, j, it.y_lastmask) &&
               _sig_row_subset_col_words(it.z_row, it.z_words, j, it.z_lastmask)
                return j, j + 1
            end
        end
    else
        @inbounds for j in state:n
            if _sig_col_subset_row_words(it.y_words, j, it.y_row, it.y_lastmask) &&
               _sig_col_subset_row_words(it.z_words, j, it.z_row, it.z_lastmask)
                return j, j + 1
            end
        end
    end
    return nothing
end

function _signature_cover_edges_uncached(P::SignaturePoset{MY,MZ}) where {MY,MZ}
    n = nvertices(P)
    mat = falses(n, n)
    edges = Tuple{Int,Int}[]
    n == 0 && return CoverEdges(mat, edges)

    yrows = Vector{NTuple{MY,UInt64}}(undef, n)
    zrows = Vector{NTuple{MZ,UInt64}}(undef, n)
    ycounts = Vector{Int}(undef, n)
    zcounts = Vector{Int}(undef, n)
    maxy = 0
    maxz = 0
    @inbounds for i in 1:n
        yrow = _sig_row_words(P.sig_y.words, i, Val(MY))
        zrow = _sig_row_words(P.sig_z.words, i, Val(MZ))
        yrows[i] = yrow
        zrows[i] = zrow
        yc = _sig_popcount_words(P.sig_y.words, i, P.y_lastmask)
        zc = _sig_popcount_words(P.sig_z.words, i, P.z_lastmask)
        ycounts[i] = yc
        zcounts[i] = zc
        yc > maxy && (maxy = yc)
        zc > maxz && (maxz = zc)
    end

    buckets = [Int[] for _ in 0:maxy, _ in 0:maxz]
    @inbounds for i in 1:n
        push!(buckets[ycounts[i] + 1, zcounts[i] + 1], i)
    end

    mins = Int[]
    mins_y = NTuple{MY,UInt64}[]
    mins_z = NTuple{MZ,UInt64}[]
    @inbounds for i in 1:n
        empty!(mins)
        empty!(mins_y)
        empty!(mins_z)
        yi = ycounts[i]
        zi = zcounts[i]
        yrow_i = yrows[i]
        zrow_i = zrows[i]
        ri = yi + zi
        for yc in yi:maxy
            for zc in zi:maxz
                yc + zc <= ri && continue
                for j in buckets[yc + 1, zc + 1]
                    _sig_tuple_subset(yrow_i, yrows[j], P.y_lastmask) || continue
                    _sig_tuple_subset(zrow_i, zrows[j], P.z_lastmask) || continue
                    dominated = false
                    @inbounds for k in eachindex(mins)
                        if _sig_tuple_subset(mins_y[k], yrows[j], P.y_lastmask) &&
                           _sig_tuple_subset(mins_z[k], zrows[j], P.z_lastmask)
                            dominated = true
                            break
                        end
                    end
                    dominated && continue
                    push!(mins, j)
                    push!(mins_y, yrows[j])
                    push!(mins_z, zrows[j])
                    push!(edges, (i, j))
                    mat[i, j] = true
                end
            end
        end
    end

    return CoverEdges(mat, edges)
end

function cover_edges(P::SignaturePoset; cached::Bool=true)
    if !cached
        return _signature_cover_edges_uncached(P)
    end
    C = P.cache.cover_edges
    C === nothing || return C

    Base.lock(P.cache.lock)
    try
        C = P.cache.cover_edges
        if C === nothing
            C = _signature_cover_edges_uncached(P)
            P.cache.cover_edges = C
        end
        return C
    finally
        Base.unlock(P.cache.lock)
    end
end

# --- Core encoding-map interface ------------------------------------------------

dimension(pi::ZnEncodingMap) = pi.n
representatives(pi::ZnEncodingMap) = pi.reps

"""
    axes_from_encoding(pi::ZnEncodingMap)

Infer a coordinate grid along each axis from the critical coordinates stored
in `pi.coords`.

For each axis i with breakpoints c1 < c2 < ... < ck, integer slab representatives
are: c1-1, c1, c2, ..., ck.

If an axis has no breakpoints, we return [0].
"""
function axes_from_encoding(pi::ZnEncodingMap)
    n = pi.n
    length(pi.coords) == n || error("axes_from_encoding: expected length(pi.coords) == pi.n")
    return ntuple(i -> begin
        ci = pi.coords[i]
        if isempty(ci)
            [0]
        else
            ax = Vector{Int}(undef, length(ci) + 1)
            ax[1] = ci[1] - 1
            @inbounds for j in 1:length(ci)
                ax[j + 1] = ci[j]
            end
            sort!(ax)
            unique!(ax)
            ax
        end
    end, n)
end

function region_poset end
function nregions end
function critical_coordinates end
function critical_coordinate_counts end
function region_representatives end
function region_representative end
function region_signature end
function zn_region_summary end
function cell_shape end
function has_direct_lookup end
function poset_kind end
function zn_encoding_summary end
function zn_query_summary end
function check_zn_box end
function check_zn_encoding_map end
function check_signature_poset end
function check_zn_cache end
function check_zn_query_point end
function check_zn_query_matrix end
function zn_encoding_validation_summary end

"""
    ZnEncodingValidationSummary

Display-oriented wrapper for reports returned by the owner-local `ZnEncoding`
validation helpers.

Use [`zn_encoding_validation_summary`](@ref) to turn a raw report from
[`check_zn_encoding_map`](@ref), [`check_signature_poset`](@ref),
[`check_zn_cache`](@ref), [`check_zn_query_point`](@ref), or
[`check_zn_query_matrix`](@ref) into a notebook/REPL-friendly summary object.
"""
struct ZnEncodingValidationSummary{R}
    report::R
end

"""
    zn_encoding_validation_summary(report) -> ZnEncodingValidationSummary

Wrap a raw `ZnEncoding` validation report in a display-oriented summary object.
"""
@inline zn_encoding_validation_summary(report::NamedTuple) = ZnEncodingValidationSummary(report)

@inline _zn_issue_report(kind::Symbol, valid::Bool; kwargs...) = (; kind, valid, kwargs...)

@inline function _throw_invalid_zn(fn::Symbol, issues::Vector{String})
    throw(ArgumentError(string(fn) * ": " * join(issues, " ")))
end

@inline _zn_poset_kind(::SignaturePoset) = :signature
@inline _zn_poset_kind(::Any) = :dense
@inline _zn_generator_counts(pi::ZnEncodingMap) = (; flats=length(pi.flats), injectives=length(pi.injectives))
@inline _zn_generator_counts(P::SignaturePoset) = (; flats=P.sig_y.bitlen, injectives=P.sig_z.bitlen)
@inline _critical_coordinate_counts(coords::Tuple) = ntuple(i -> length(coords[i]), length(coords))
@inline _critical_coordinate_counts(pi::ZnEncodingMap) = _critical_coordinate_counts(pi.coords)

"""
    region_poset(pi::ZnEncodingMap)
    region_poset(cache::ZnEncodingCache)

Return the finite region poset attached to a Zn encoding object.

For bare [`ZnEncodingMap`](@ref) values, this reconstructs the native
signature poset from the stored packed `(y, z)` signatures. For
[`ZnEncodingCache`](@ref) and compiled encodings, it returns the already
attached finite poset.

Best practice:
- call `region_poset(cache)` when you already have a compiled/cache-bearing
  object and want repeated region-poset work to reuse that attachment;
- call `region_poset(pi)` for one-off inspection of a bare
  [`ZnEncodingMap`](@ref);
- use [`poset_kind`](@ref) to check whether you are looking at the structured
  `:signature` poset or a dense finite-poset realization.
"""
@inline region_poset(pi::ZnEncodingMap) = SignaturePoset(pi.sig_y, pi.sig_z)
@inline region_poset(cache::ZnEncodingCache) = cache.P
@inline region_poset(enc::CompiledEncoding{<:ZnEncodingMap}) = enc.P

"""
    nregions(pi)

Return the number of finite regions represented by a Zn encoding object.
"""
@inline nregions(pi::ZnEncodingMap) = length(pi.reps)
@inline nregions(cache::ZnEncodingCache) = nregions(cache.pi)
@inline nregions(enc::CompiledEncoding{<:ZnEncodingMap}) = nregions(enc.pi)
@inline nregions(P::SignaturePoset) = nvertices(P)

"""
    critical_coordinates(pi)

Return the stored critical lattice coordinates along each axis.

This is the owner-local alias for the coordinate-grid information carried by a
[`ZnEncodingMap`](@ref). Use [`axes_from_encoding`](@ref) when you want slab
representatives rather than the raw critical breakpoints.
"""
@inline critical_coordinates(pi::ZnEncodingMap) = pi.coords
@inline critical_coordinates(cache::ZnEncodingCache) = critical_coordinates(cache.pi)
@inline critical_coordinates(enc::CompiledEncoding{<:ZnEncodingMap}) = critical_coordinates(enc.pi)

"""
    critical_coordinate_counts(pi) -> Tuple

Return the number of stored critical coordinates along each axis.

This is the preferred cheap scalar accessor when you want to know the slab
complexity of a Zn encoding object without materializing or inspecting the full
critical-coordinate tuple.
"""
@inline critical_coordinate_counts(pi::ZnEncodingMap) = _critical_coordinate_counts(pi)
@inline critical_coordinate_counts(cache::ZnEncodingCache) = critical_coordinate_counts(cache.pi)
@inline critical_coordinate_counts(enc::CompiledEncoding{<:ZnEncodingMap}) = critical_coordinate_counts(enc.pi)

"""
    region_representatives(pi)
    region_representative(pi, r)

Return all stored lattice representatives, or the representative for region
`r`, of a Zn encoding object.
"""
@inline region_representatives(pi::ZnEncodingMap) = pi.reps
@inline region_representatives(cache::ZnEncodingCache) = region_representatives(cache.pi)
@inline region_representatives(enc::CompiledEncoding{<:ZnEncodingMap}) = region_representatives(enc.pi)
@inline region_representative(pi::ZnEncodingMap, r::Integer) = (@boundscheck checkbounds(pi.reps, r); pi.reps[r])
@inline region_representative(cache::ZnEncodingCache, r::Integer) = region_representative(cache.pi, r)
@inline region_representative(enc::CompiledEncoding{<:ZnEncodingMap}, r::Integer) = region_representative(enc.pi, r)

"""
    region_signature(pi, r) -> NamedTuple

Return the `(y, z)` signature of region `r` as
`(; y=Vector{Bool}, z=Vector{Bool})`.

Use this for notebook or REPL inspection when you want the actual signature
bits rather than the packed internal storage.

The returned vectors are intentionally materialized and user-readable. Keep
`describe(...)`, [`zn_encoding_summary`](@ref), or [`zn_query_summary`](@ref)
as the cheap/default inspection path, and only call `region_signature(...)`
when you actually want the signature bits of a specific region.
"""
@inline function region_signature(pi::ZnEncodingMap, r::Integer)
    @boundscheck begin
        checkbounds(pi.sig_y, r)
        checkbounds(pi.sig_z, r)
    end
    return (; y=collect(Bool, pi.sig_y[r]), z=collect(Bool, pi.sig_z[r]))
end
@inline region_signature(cache::ZnEncodingCache, r::Integer) = region_signature(cache.pi, r)
@inline region_signature(enc::CompiledEncoding{<:ZnEncodingMap}, r::Integer) = region_signature(enc.pi, r)
@inline function region_signature(P::SignaturePoset, r::Integer)
    @boundscheck begin
        checkbounds(P.sig_y, r)
        checkbounds(P.sig_z, r)
    end
    return (; y=collect(Bool, P.sig_y[r]), z=collect(Bool, P.sig_z[r]))
end

"""
    cell_shape(pi)

Return the slab-cell shape used by the Zn encoding direct lookup table.
"""
@inline cell_shape(pi::ZnEncodingMap) = pi.cell_shape
@inline cell_shape(cache::ZnEncodingCache) = cell_shape(cache.pi)
@inline cell_shape(enc::CompiledEncoding{<:ZnEncodingMap}) = cell_shape(enc.pi)

"""
    has_direct_lookup(pi) -> Bool

Return whether the Zn encoding object stores a direct slab-cell-to-region
lookup table for cheap integer and floating-point location queries.
"""
@inline has_direct_lookup(pi::ZnEncodingMap) = pi.cell_to_region !== nothing
@inline has_direct_lookup(cache::ZnEncodingCache) = has_direct_lookup(cache.pi)
@inline has_direct_lookup(enc::CompiledEncoding{<:ZnEncodingMap}) = has_direct_lookup(enc.pi)

@inline flats(pi::ZnEncodingMap) = pi.flats
@inline flats(cache::ZnEncodingCache) = flats(cache.pi)
@inline flats(enc::CompiledEncoding{<:ZnEncodingMap}) = flats(enc.pi)
@inline injectives(pi::ZnEncodingMap) = pi.injectives
@inline injectives(cache::ZnEncodingCache) = injectives(cache.pi)
@inline injectives(enc::CompiledEncoding{<:ZnEncodingMap}) = injectives(enc.pi)

"""
    generator_counts(x) -> NamedTuple

Return the number of flat and injective generators recorded by a Zn encoding
object as `(; flats=..., injectives=...)`.

This is the preferred cheap scalar accessor when you only need presentation
sizes rather than the full `flats(...)` and `injectives(...)` payloads.
"""
@inline generator_counts(pi::ZnEncodingMap) = _zn_generator_counts(pi)
@inline generator_counts(cache::ZnEncodingCache) = generator_counts(cache.pi)
@inline generator_counts(enc::CompiledEncoding{<:ZnEncodingMap}) = generator_counts(enc.pi)
@inline generator_counts(P::SignaturePoset) = _zn_generator_counts(P)

"""
    poset_kind(pi_or_cache) -> Symbol

Return `:signature` for structured signature-poset views and `:dense` for
materialized finite-poset realizations carried by cache-bearing objects.
"""
@inline poset_kind(pi::ZnEncodingMap) = :signature
@inline poset_kind(P::SignaturePoset) = :signature
@inline poset_kind(cache::ZnEncodingCache) = _zn_poset_kind(cache.P)
@inline poset_kind(enc::CompiledEncoding{<:ZnEncodingMap}) = _zn_poset_kind(enc.P)

"""
    zn_encoding_summary(x) -> NamedTuple

Owner-local inspection surface for `ZnEncoding`.

This mirrors the shared `describe(...)` entrypoint, but keeps the Zn owner
module self-discoverable in notebooks and the REPL.

Accepted inputs include bare [`ZnEncodingMap`](@ref) values, structured
[`SignaturePoset`](@ref) objects, explicit [`ZnEncodingCache`](@ref) handles,
and `CompiledEncoding{<:ZnEncodingMap}` wrappers.

# Examples

```julia
using TamerOp

field = TamerOp.CoreModules.QQField()
FZ = TamerOp.FlangeZn
ZE = TamerOp.ZnEncoding

tau = FZ.Face(1, [false])
F = FZ.IndFlat(tau, [0]; id=:F)
E = FZ.IndInj(tau, [2]; id=:E)
FG = FZ.Flange{TamerOp.CoreModules.QQ}(1, [F], [E], reshape([1//1], 1, 1); field=field)

P, H, pi = ZE.encode_from_flange(FG)
ZE.zn_encoding_summary(pi)
ZE.zn_query_summary(pi, (0,))
```
"""
@inline zn_encoding_summary(pi::ZnEncodingMap) = _znencoding_describe(pi)
@inline zn_encoding_summary(P::SignaturePoset) = _znencoding_describe(P)
@inline zn_encoding_summary(cache::ZnEncodingCache) = _znencoding_describe(cache)
@inline zn_encoding_summary(enc::CompiledEncoding{<:ZnEncodingMap}) = _znencoding_describe(ZnEncodingCache(enc.P, enc.pi))

@inline _zn_query_point_key(g::Tuple) = g
@inline _zn_query_point_key(g::AbstractVector) = Tuple(g)
@inline _zn_query_point_key(g) = g

"""
    zn_query_summary(pi, g) -> NamedTuple

Return a cheap semantic summary of a single Zn encoding query.

This helper is intended for notebook and REPL exploration. It reports:
- the query point,
- the integer/float query kind,
- the located region id,
- the stored region representative when a region is found,
- the support sizes of the `y`/`z` signatures,
- whether direct slab lookup is available,
- whether the query landed outside the represented region set (`0` from
  [`locate`](@ref)).

This accepts the same owner objects as [`locate`](@ref), including
[`ZnEncodingCache`](@ref) and compiled encodings.
"""
function zn_query_summary(pi_or_cache, g)
    pi = _unwrap_zn_pi(pi_or_cache)
    report = check_zn_query_point(pi, g; throw=true)
    rid = locate(pi_or_cache, g)
    sig_counts = if rid == 0
        nothing
    else
        sig = region_signature(pi, rid)
        (; y=count(identity, sig.y), z=count(identity, sig.z))
    end
    return (;
        kind=:zn_query,
        point=_zn_query_point_key(g),
        query_kind=report.query_kind,
        region=rid,
        representative=rid == 0 ? nothing : region_representative(pi, rid),
        signature_support_sizes=sig_counts,
        direct_lookup_enabled=has_direct_lookup(pi),
        outside=(rid == 0),
    )
end

"""
    zn_region_summary(pi, r) -> NamedTuple

Return a compact semantic summary of region `r` in a Zn encoding object.

This helper is intended for notebook and REPL exploration when you already know
the region id and want the canonical region-level payload in one place instead
of separately calling [`region_representative`](@ref),
[`region_signature`](@ref), and [`generator_counts`](@ref).
"""
function zn_region_summary(pi_or_cache, r::Integer)
    pi = _unwrap_zn_pi(pi_or_cache)
    sig = region_signature(pi, r)
    return (;
        kind=:zn_region,
        region=Int(r),
        representative=region_representative(pi, r),
        signature=sig,
        signature_support_sizes=(; y=count(identity, sig.y), z=count(identity, sig.z)),
        generator_counts=generator_counts(pi),
        direct_lookup_enabled=has_direct_lookup(pi),
    )
end

@inline _zn_box_endpoint_kind(x::AbstractVector{<:Integer}) = :integer_vector
@inline _zn_box_endpoint_kind(x::Tuple) = all(v -> v isa Integer, x) ? :integer_tuple : :invalid
@inline _zn_box_endpoint_kind(::Any) = :invalid

@inline _zn_box_endpoint_length(x::AbstractVector) = length(x)
@inline _zn_box_endpoint_length(x::Tuple) = length(x)
@inline _zn_box_endpoint_length(::Any) = nothing

@inline function _zn_box_total_points(a::AbstractVector{<:Integer}, b::AbstractVector{<:Integer})
    total = BigInt(1)
    @inbounds for i in eachindex(a, b)
        total *= BigInt(b[i]) - BigInt(a[i]) + 1
    end
    return total
end

"""
    check_zn_box(pi, box; throw=false) -> NamedTuple

Validate a finite integer query box for [`region_weights`](@ref) or
[`region_adjacency`](@ref).

The accepted contract is a pair `(a, b)` where both endpoints are integer
vectors or integer tuples of ambient dimension `dimension(pi)`, interpreted as
the inclusive lattice box `{ g in Z^n : a_i <= g_i <= b_i }`.
"""
function check_zn_box(pi_or_cache, box; throw::Bool=false)
    pi = _unwrap_zn_pi(pi_or_cache)
    issues = String[]
    endpoint_types = nothing
    endpoint_lengths = nothing
    total_points = nothing
    if !(box isa Tuple && length(box) == 2)
        push!(issues, "box must be a pair (a, b) of integer endpoints.")
    else
        a, b = box
        kind_a = _zn_box_endpoint_kind(a)
        kind_b = _zn_box_endpoint_kind(b)
        endpoint_types = (kind_a, kind_b)
        kind_a === :invalid && push!(issues, "box lower endpoint must be an integer vector or integer tuple.")
        kind_b === :invalid && push!(issues, "box upper endpoint must be an integer vector or integer tuple.")
        len_a = _zn_box_endpoint_length(a)
        len_b = _zn_box_endpoint_length(b)
        endpoint_lengths = (len_a, len_b)
        if len_a !== nothing && len_a != dimension(pi)
            push!(issues, "box lower endpoint has length $len_a, expected ambient dimension $(dimension(pi)).")
        end
        if len_b !== nothing && len_b != dimension(pi)
            push!(issues, "box upper endpoint has length $len_b, expected ambient dimension $(dimension(pi)).")
        end
        if isempty(issues)
            av = collect(Int, a)
            bv = collect(Int, b)
            @inbounds for i in eachindex(av, bv)
                av[i] <= bv[i] || push!(issues, "box endpoint mismatch on axis $i: expected a[$i] <= b[$i].")
            end
            isempty(issues) && (total_points = _zn_box_total_points(av, bv))
        end
    end
    valid = isempty(issues)
    throw && !valid && _throw_invalid_zn(:check_zn_box, issues)
    return _zn_issue_report(:zn_box, valid;
                            ambient_dim=dimension(pi),
                            endpoint_types=endpoint_types,
                            endpoint_lengths=endpoint_lengths,
                            total_points=total_points,
                            issues=issues)
end

@inline function _znencoding_describe(pi::ZnEncodingMap)
    return (;
        kind=:zn_encoding_map,
        ambient_dim=dimension(pi),
        nregions=nregions(pi),
        generator_counts=generator_counts(pi),
        poset_kind=poset_kind(pi),
        direct_lookup_enabled=has_direct_lookup(pi),
        critical_coordinate_counts=critical_coordinate_counts(pi),
        cell_shape=cell_shape(pi),
    )
end

@inline function _znencoding_describe(P::SignaturePoset)
    return (;
        kind=:signature_poset,
        ambient_dim=nothing,
        nregions=nregions(P),
        generator_counts=generator_counts(P),
        poset_kind=poset_kind(P),
        direct_lookup_enabled=nothing,
        critical_coordinate_counts=nothing,
        cell_shape=nothing,
    )
end

@inline function _znencoding_describe(cache::ZnEncodingCache)
    return (;
        kind=:zn_encoding_cache,
        ambient_dim=dimension(cache.pi),
        nregions=nregions(cache.pi),
        generator_counts=generator_counts(cache.pi),
        poset_kind=poset_kind(cache),
        direct_lookup_enabled=has_direct_lookup(cache.pi),
        critical_coordinate_counts=critical_coordinate_counts(cache.pi),
        cell_shape=cell_shape(cache.pi),
    )
end

function Base.show(io::IO, pi::ZnEncodingMap)
    d = _znencoding_describe(pi)
    print(io, "ZnEncodingMap(n=", d.ambient_dim,
          ", regions=", d.nregions,
          ", flats=", d.generator_counts.flats,
          ", injectives=", d.generator_counts.injectives,
          ", direct_lookup=", d.direct_lookup_enabled, ")")
end

function Base.show(io::IO, ::MIME"text/plain", pi::ZnEncodingMap)
    d = _znencoding_describe(pi)
    print(io, "ZnEncodingMap",
          "\n  ambient_dim = ", d.ambient_dim,
          "\n  nregions = ", d.nregions,
          "\n  generator_counts = ", d.generator_counts,
          "\n  poset_kind = ", repr(d.poset_kind),
          "\n  direct_lookup_enabled = ", d.direct_lookup_enabled,
          "\n  critical_coordinate_counts = ", d.critical_coordinate_counts,
          "\n  cell_shape = ", d.cell_shape)
end

function Base.show(io::IO, P::SignaturePoset)
    d = _znencoding_describe(P)
    print(io, "SignaturePoset(regions=", d.nregions,
          ", flats=", d.generator_counts.flats,
          ", injectives=", d.generator_counts.injectives, ")")
end

function Base.show(io::IO, ::MIME"text/plain", P::SignaturePoset)
    d = _znencoding_describe(P)
    print(io, "SignaturePoset",
          "\n  nregions = ", d.nregions,
          "\n  generator_counts = ", d.generator_counts,
          "\n  poset_kind = ", repr(d.poset_kind))
end

function Base.show(io::IO, cache::ZnEncodingCache)
    d = _znencoding_describe(cache)
    print(io, "ZnEncodingCache(n=", d.ambient_dim,
          ", regions=", d.nregions,
          ", poset_kind=", repr(d.poset_kind),
          ", direct_lookup=", d.direct_lookup_enabled, ")")
end

function Base.show(io::IO, ::MIME"text/plain", cache::ZnEncodingCache)
    d = _znencoding_describe(cache)
    print(io, "ZnEncodingCache",
          "\n  ambient_dim = ", d.ambient_dim,
          "\n  nregions = ", d.nregions,
          "\n  generator_counts = ", d.generator_counts,
          "\n  poset_kind = ", repr(d.poset_kind),
          "\n  direct_lookup_enabled = ", d.direct_lookup_enabled,
          "\n  critical_coordinate_counts = ", d.critical_coordinate_counts,
          "\n  cell_shape = ", d.cell_shape)
end

@inline _zn_point_length(g::AbstractVector) = length(g)
@inline _zn_point_length(g::Tuple) = length(g)
@inline _zn_point_length(::Any) = nothing

@inline _zn_point_kind(g::AbstractVector{<:Integer}) = :integer
@inline _zn_point_kind(g::AbstractVector{<:AbstractFloat}) = :float
@inline _zn_point_kind(g::Tuple) = all(x -> x isa Integer, g) ? :integer :
                                   all(x -> x isa AbstractFloat, g) ? :float : :invalid
@inline _zn_point_kind(::Any) = :invalid

@inline _zn_matrix_kind(X::AbstractMatrix{<:Integer}) = :integer
@inline _zn_matrix_kind(X::AbstractMatrix{<:AbstractFloat}) = :float
@inline _zn_matrix_kind(::Any) = :invalid

function _check_critical_coordinates!(issues::Vector{String}, coords, n::Int)
    length(coords) == n || push!(issues, "critical coordinate tuple has length $(length(coords)), expected $n.")
    @inbounds for i in 1:min(length(coords), n)
        axis = coords[i]
        axis isa AbstractVector || begin
            push!(issues, "critical coordinate axis $i must be an abstract vector of integers.")
            continue
        end
        all(x -> x isa Integer, axis) || push!(issues, "critical coordinate axis $i must contain only integers.")
        issorted(axis) || push!(issues, "critical coordinate axis $i must be sorted.")
        for j in 2:length(axis)
            axis[j - 1] < axis[j] || begin
                push!(issues, "critical coordinate axis $i must be strictly increasing.")
                break
            end
        end
    end
    return nothing
end

"""
    check_signature_poset(P; throw=false) -> NamedTuple

Validate a hand-built [`SignaturePoset`](@ref).

This helper checks:
- the packed y/z signature tables have the same number of regions,
- the cached region count matches those tables,
- the cached last-word masks agree with the packed signature widths.

Use this on hand-built signature posets or after low-level transformations that
operate below the normal `encode_*` entrypoints. Wrap the returned report with
[`zn_encoding_validation_summary`](@ref) when you want a display-oriented
summary.
"""
function check_signature_poset(P::SignaturePoset; throw::Bool=false)
    issues = String[]
    length(P.sig_y) == length(P.sig_z) || push!(issues, "sig_y and sig_z have different region counts.")
    nvertices(P) == length(P.sig_y) || push!(issues, "cached region count $(nvertices(P)) does not match sig_y length $(length(P.sig_y)).")
    nvertices(P) == length(P.sig_z) || push!(issues, "cached region count $(nvertices(P)) does not match sig_z length $(length(P.sig_z)).")
    P.y_lastmask == _word_lastmask(P.sig_y.bitlen) || push!(issues, "y_lastmask does not match sig_y bitlen.")
    P.z_lastmask == _word_lastmask(P.sig_z.bitlen) || push!(issues, "z_lastmask does not match sig_z bitlen.")
    valid = isempty(issues)
    throw && !valid && _throw_invalid_zn(:check_signature_poset, issues)
    return _zn_issue_report(:signature_poset, valid;
                            nregions=nregions(P),
                            generator_counts=_zn_generator_counts(P),
                            poset_kind=:signature,
                            issues=issues)
end

"""
    check_zn_encoding_map(pi; throw=false) -> NamedTuple

Validate a hand-built [`ZnEncodingMap`](@ref).

This helper checks:
- the ambient dimension agrees with stored coordinates and representatives,
- region counts agree across signatures, representatives, and signature lookup,
- signature widths agree with the flat/injective families,
- the direct lookup metadata is shape-consistent when present.

This is the owner-specific validator to use when a map is being assembled or
modified by hand. Prefer [`zn_encoding_summary`](@ref) or `describe(...)` for
cheap inspection, and keep `check_zn_encoding_map(...)` for actual contract
validation.
"""
function check_zn_encoding_map(pi::ZnEncodingMap; throw::Bool=false)
    issues = String[]
    pi.n >= 0 || push!(issues, "ambient dimension must be nonnegative.")
    _check_critical_coordinates!(issues, pi.coords, pi.n)
    nr = nregions(pi)
    length(pi.sig_y) == nr || push!(issues, "sig_y has $(length(pi.sig_y)) regions, expected $nr.")
    length(pi.sig_z) == nr || push!(issues, "sig_z has $(length(pi.sig_z)) regions, expected $nr.")
    pi.sig_y.bitlen == length(pi.flats) || push!(issues,
        "sig_y bitlen $(pi.sig_y.bitlen) does not match flat count $(length(pi.flats)).")
    pi.sig_z.bitlen == length(pi.injectives) || push!(issues,
        "sig_z bitlen $(pi.sig_z.bitlen) does not match injective count $(length(pi.injectives)).")
    for (i, rep) in pairs(pi.reps)
        length(rep) == pi.n || push!(issues, "representative $i has length $(length(rep)), expected $(pi.n).")
    end
    length(pi.sig_to_region) == nr || push!(issues,
        "signature lookup stores $(length(pi.sig_to_region)) regions, expected $nr.")
    vals = collect(values(pi.sig_to_region))
    all(v -> 1 <= v <= nr, vals) || push!(issues, "signature lookup contains a region index outside 1:$nr.")
    length(unique(vals)) == nr || push!(issues, "signature lookup values do not cover each region exactly once.")
    expected_shape = ntuple(i -> length(pi.coords[i]) + 1, pi.n)
    pi.cell_shape == expected_shape || push!(issues,
        "cell_shape $(pi.cell_shape) does not match critical coordinates $(expected_shape).")
    expected_strides = ntuple(i -> i == 1 ? 1 : prod(expected_shape[1:(i - 1)]), pi.n)
    pi.cell_strides == expected_strides || push!(issues,
        "cell_strides $(pi.cell_strides) do not match cell_shape $(expected_shape).")
    expected_fingerprint = _encoding_fingerprint(pi.n, pi.flats, pi.injectives)
    pi.encoding_fingerprint == expected_fingerprint || push!(issues,
        "encoding fingerprint does not match the stored flat/injective families.")
    if pi.cell_to_region !== nothing
        total_cells = isempty(pi.cell_shape) ? 1 : prod(pi.cell_shape)
        length(pi.cell_to_region) == total_cells || push!(issues,
            "direct lookup table has length $(length(pi.cell_to_region)), expected $total_cells.")
        all(v -> 0 <= v <= nr, pi.cell_to_region) || push!(issues,
            "direct lookup table contains a region index outside 0:$nr.")
    end
    valid = isempty(issues)
    throw && !valid && _throw_invalid_zn(:check_zn_encoding_map, issues)
    return _zn_issue_report(:zn_encoding_map, valid;
                            ambient_dim=dimension(pi),
                            nregions=nr,
                            generator_counts=_zn_generator_counts(pi),
                            poset_kind=:signature,
                            direct_lookup_enabled=has_direct_lookup(pi),
                            critical_coordinate_counts=_critical_coordinate_counts(pi),
                            cell_shape=cell_shape(pi),
                            issues=issues)
end

"""
    check_zn_cache(cache; throw=false) -> NamedTuple

Validate a [`ZnEncodingCache`](@ref).

This helper checks:
- the wrapped [`ZnEncodingMap`](@ref) passes [`check_zn_encoding_map`](@ref),
- the cached poset has the same number of regions as the map,
- cached signature posets also pass [`check_signature_poset`](@ref).

Use this before repeated advanced-owner workflows when you are carrying an
explicit cache object and want to confirm that the cached poset/map pairing is
still coherent.
"""
function check_zn_cache(cache::ZnEncodingCache; throw::Bool=false)
    issues = String[]
    map_report = check_zn_encoding_map(cache.pi)
    append!(issues, ["map: $msg" for msg in map_report.issues])
    nvertices(cache.P) == nregions(cache.pi) || push!(issues,
        "cached poset has $(nvertices(cache.P)) vertices, expected $(nregions(cache.pi)).")
    if cache.P isa SignaturePoset
        poset_report = check_signature_poset(cache.P)
        append!(issues, ["poset: $msg" for msg in poset_report.issues])
    end
    valid = isempty(issues)
    throw && !valid && _throw_invalid_zn(:check_zn_cache, issues)
    return _zn_issue_report(:zn_encoding_cache, valid;
                            ambient_dim=dimension(cache.pi),
                            nregions=nregions(cache.pi),
                            generator_counts=_zn_generator_counts(cache.pi),
                            poset_kind=_zn_poset_kind(cache.P),
                            direct_lookup_enabled=has_direct_lookup(cache.pi),
                            issues=issues)
end

"""
    check_zn_query_point(pi, g; throw=false) -> NamedTuple

Validate a single Zn encoding query point before calling [`locate`](@ref).

Accepted query shapes are:
- integer vectors and integer tuples in `Z^n`,
- floating-point vectors and floating-point tuples in `R^n`.

Use this helper when query points come from user input rather than trusted
internal code. Integer queries are the native contract. Floating-point queries
are accepted because `locate(...)` rounds componentwise to the nearest lattice
point before classification.
"""
function check_zn_query_point(pi_or_cache, g; throw::Bool=false)
    pi = _unwrap_zn_pi(pi_or_cache)
    issues = String[]
    if !(g isa AbstractVector || g isa Tuple)
        push!(issues, "point must be an integer or floating-point vector/tuple.")
    end
    kind = _zn_point_kind(g)
    kind === :invalid && push!(issues,
        "point must be either all-integer or all-floating, matching the supported locate contracts.")
    glen = _zn_point_length(g)
    if glen !== nothing && glen != dimension(pi)
        push!(issues, "point has length $glen, expected ambient dimension $(dimension(pi)).")
    end
    valid = isempty(issues)
    throw && !valid && _throw_invalid_zn(:check_zn_query_point, issues)
    return _zn_issue_report(:zn_query_point, valid;
                            ambient_dim=dimension(pi),
                            query_kind=kind,
                            point_type=typeof(g),
                            point_length=glen,
                            issues=issues)
end

"""
    check_zn_query_matrix(pi, X; throw=false) -> NamedTuple

Validate a batched Zn encoding query matrix before calling
[`locate_many!`](@ref).

Accepted query shapes are matrices with one query per column and either:
- integer entries, or
- floating-point entries.

Use this before [`locate_many!`](@ref) when batched query data comes from an
external source. The accepted matrix contracts are intentionally strict so they
match the actual high-performance owner methods.
"""
function check_zn_query_matrix(pi_or_cache, X; throw::Bool=false)
    pi = _unwrap_zn_pi(pi_or_cache)
    issues = String[]
    if !(X isa AbstractMatrix)
        push!(issues, "X must be an integer or floating-point matrix with one query per column.")
    end
    kind = _zn_matrix_kind(X)
    kind === :invalid && push!(issues,
        "X must have eltype <: Integer or <: AbstractFloat to match the supported locate_many! contracts.")
    matrix_rows = X isa AbstractMatrix ? size(X, 1) : nothing
    if matrix_rows !== nothing && matrix_rows != dimension(pi)
        push!(issues, "X has $matrix_rows rows, expected ambient dimension $(dimension(pi)).")
    end
    valid = isempty(issues)
    throw && !valid && _throw_invalid_zn(:check_zn_query_matrix, issues)
    return _zn_issue_report(:zn_query_matrix, valid;
                            ambient_dim=dimension(pi),
                            query_kind=kind,
                            matrix_type=typeof(X),
                            matrix_size=X isa AbstractMatrix ? size(X) : nothing,
                            issues=issues)
end

function Base.show(io::IO, summary::ZnEncodingValidationSummary)
    report = summary.report
    print(io, "ZnEncodingValidationSummary(kind=", report.kind,
          ", valid=", report.valid,
          ", issues=", length(report.issues), ")")
end

function Base.show(io::IO, ::MIME"text/plain", summary::ZnEncodingValidationSummary)
    report = summary.report
    println(io, "ZnEncodingValidationSummary")
    println(io, "  kind: ", report.kind)
    println(io, "  valid: ", report.valid)
    for (key, value) in pairs(report)
        key in (:kind, :valid, :issues) && continue
        println(io, "  ", key, ": ", value)
    end
    println(io, "  issues:")
    if isempty(report.issues)
        println(io, "    (none)")
    else
        for issue in report.issues
            println(io, "    - ", issue)
        end
    end
end


# ---------------------------- Internal helpers ---------------------------------

"Key for identifying an indecomposable flat up to equality of the underlying upset (ignores `id`)."
_flat_key(F::IndFlat) = (F.b, Tuple(F.tau.coords))

"Key for identifying an indecomposable injective up to equality of the underlying downset (ignores `id`)."
_inj_key(E::IndInj) = (E.b, Tuple(E.tau.coords))

"""
Build lookup dictionaries for the generator lists used by an encoding.

Returns:
- flat_index[key] = i, where i is an index into `flats`
- inj_index[key]  = j, where j is an index into `injectives`

The keys ignore label `id` on purpose: the encoding depends only on the underlying
(up)set/(down)set, not the symbol used to name it.
"""
function _generator_index_dicts(flats::Vector{IndFlat{N}},
                                injectives::Vector{IndInj{N}}) where {N}
    flat_index = Dict{_GENERATOR_KEY, Int}()
    for (i, F) in enumerate(flats)
        key = _flat_key(F)
        if !haskey(flat_index, key)
            flat_index[key] = i
        end
    end

    inj_index = Dict{_GENERATOR_KEY, Int}()
    for (j, E) in enumerate(injectives)
        key = _inj_key(E)
        if !haskey(inj_index, key)
            inj_index[key] = j
        end
    end

    return flat_index, inj_index
end

function _build_pushforward_masks(sig_y::PackedSignatureRows,
                                  sig_z::PackedSignatureRows,
                                  nflat::Int,
                                  ninj::Int)
    n = length(sig_y)
    flat_masks = Vector{BitVector}(undef, nflat)
    @inbounds for i in 1:nflat
        mask = falses(n)
        wy = ((i - 1) >>> 6) + 1
        by = UInt64(1) << ((i - 1) & 63)
        for t in 1:n
            mask[t] = (sig_y.words[wy, t] & by) != 0
        end
        flat_masks[i] = mask
    end

    inj_masks = Vector{BitVector}(undef, ninj)
    @inbounds for j in 1:ninj
        mask = falses(n)
        wz = ((j - 1) >>> 6) + 1
        bz = UInt64(1) << ((j - 1) & 63)
        for t in 1:n
            mask[t] = (sig_z.words[wz, t] & bz) == 0
        end
        inj_masks[j] = mask
    end
    return flat_masks, inj_masks
end

function _ensure_pushforward_cache!(pi::ZnEncodingMap)
    cache = pi.pushforward_cache
    flat_index = cache.flat_index
    inj_index = cache.inj_index
    flat_masks = cache.flat_masks
    inj_masks = cache.inj_masks
    if flat_index !== nothing && inj_index !== nothing && flat_masks !== nothing && inj_masks !== nothing
        return cache
    end

    Base.lock(cache.lock)
    try
        if cache.flat_index === nothing || cache.inj_index === nothing
            fi, ji = _generator_index_dicts(pi.flats, pi.injectives)
            cache.flat_index = fi
            cache.inj_index = ji
        end
        if cache.flat_masks === nothing || cache.inj_masks === nothing
            fm, im = _build_pushforward_masks(pi.sig_y, pi.sig_z, pi.sig_y.bitlen, pi.sig_z.bitlen)
            cache.flat_masks = fm
            cache.inj_masks = im
        end
        return cache
    finally
        Base.unlock(cache.lock)
    end
end

@inline function _zero_pairs_from_indices(flat_masks::Vector{BitVector},
                                          inj_masks::Vector{BitVector},
                                          flat_idxs::Vector{Int},
                                          inj_idxs::Vector{Int})
    zero_pairs = Tuple{Int,Int}[]
    @inbounds for jloc in 1:length(inj_idxs)
        dj = inj_masks[inj_idxs[jloc]]
        for iloc in 1:length(flat_idxs)
            if !intersects(flat_masks[flat_idxs[iloc]], dj)
                push!(zero_pairs, (jloc, iloc))
            end
        end
    end
    return zero_pairs
end

function _get_or_build_pushforward_plan!(pi::ZnEncodingMap, FG::Flange;
                                         session_cache::Union{Nothing,SessionCache}=nothing)
    cache = _ensure_pushforward_cache!(pi)
    fkey = _flange_fingerprint(FG)
    plan = get(cache.plan_by_flange, fkey, nothing)
    plan === nothing || return plan

    if session_cache !== nothing
        shared = _session_get_zn_pushforward_plan(session_cache, pi.encoding_fingerprint, fkey)
        if shared !== nothing
            shared_plan = shared isa ZnPushforwardPlan ? shared::ZnPushforwardPlan :
                ZnPushforwardPlan(shared.flat_idxs, shared.inj_idxs, shared.zero_pairs)
            Base.lock(cache.lock)
            try
                get!(cache.plan_by_flange, fkey, shared_plan)
            finally
                Base.unlock(cache.lock)
            end
            return shared_plan
        end
    end

    built_plan = nothing
    Base.lock(cache.lock)
    try
        plan = get(cache.plan_by_flange, fkey, nothing)
        if plan !== nothing
            built_plan = plan
        else
            flat_index = cache.flat_index::Dict{_GENERATOR_KEY,Int}
            inj_index = cache.inj_index::Dict{_GENERATOR_KEY,Int}
            flat_masks = cache.flat_masks::Vector{BitVector}
            inj_masks = cache.inj_masks::Vector{BitVector}

            flat_idxs = Vector{Int}(undef, length(FG.flats))
            @inbounds for i in 1:length(FG.flats)
                key = _flat_key(FG.flats[i])
                idx = get(flat_index, key, 0)
                idx == 0 && error("_pushforward_flange_to_fringe(strict=true): flat label $(FG.flats[i]) not present in encoding generators")
                flat_idxs[i] = idx
            end

            inj_idxs = Vector{Int}(undef, length(FG.injectives))
            @inbounds for j in 1:length(FG.injectives)
                key = _inj_key(FG.injectives[j])
                idx = get(inj_index, key, 0)
                idx == 0 && error("_pushforward_flange_to_fringe(strict=true): injective label $(FG.injectives[j]) not present in encoding generators")
                inj_idxs[j] = idx
            end

            zero_pairs = _zero_pairs_from_indices(flat_masks, inj_masks, flat_idxs, inj_idxs)
            built_plan = ZnPushforwardPlan(flat_idxs, inj_idxs, zero_pairs)
            cache.plan_by_flange[fkey] = built_plan
        end
    finally
        Base.unlock(cache.lock)
    end

    plan = built_plan::ZnPushforwardPlan
    if session_cache !== nothing
        _session_set_zn_pushforward_plan!(session_cache, pi.encoding_fingerprint, fkey, plan)
    end
    return plan
end

@inline function _images_on_P_from_masks(P::AbstractPoset,
                                         flat_masks::Vector{BitVector},
                                         inj_masks::Vector{BitVector},
                                         flat_idxs::AbstractVector{<:Integer},
                                         inj_idxs::AbstractVector{<:Integer})
    m = length(flat_idxs)
    r = length(inj_idxs)
    Uhat = Vector{Upset}(undef, m)
    Dhat = Vector{Downset}(undef, r)
    @inbounds for loc in 1:m
        Uhat[loc] = Upset(P, flat_masks[Int(flat_idxs[loc])])
    end
    @inbounds for loc in 1:r
        Dhat[loc] = Downset(P, inj_masks[Int(inj_idxs[loc])])
    end
    return Uhat, Dhat
end

@inline function _strict_pushforward_plan_and_masks(pi::ZnEncodingMap,
                                                    FG::Flange;
                                                    session_cache::Union{Nothing,SessionCache}=nothing)
    plan = _get_or_build_pushforward_plan!(pi, FG; session_cache=session_cache)
    cache = _ensure_pushforward_cache!(pi)
    flat_masks = cache.flat_masks::Vector{BitVector}
    inj_masks = cache.inj_masks::Vector{BitVector}
    return plan, flat_masks, inj_masks
end

@inline function _strict_pushforward_fringe_from_plan(P::AbstractPoset,
                                                      FG::Flange{K},
                                                      plan::ZnPushforwardPlan,
                                                      flat_masks::Vector{BitVector},
                                                      inj_masks::Vector{BitVector}) where {K}
    Uhat, Dhat = _images_on_P_from_masks(P, flat_masks, inj_masks, plan.flat_idxs, plan.inj_idxs)
    Phi = _monomialize_phi_with_zero_pairs(FG.phi, plan.zero_pairs)
    return FringeModule{K}(P, Uhat, Dhat, Phi; field=FG.field)
end

@inline function _gather_rows_subset(rows_u::Vector{Int},
                                     rows_v::Vector{Int},
                                     Bu::AbstractMatrix{K},
                                     field::AbstractCoeffField) where {K}
    du = size(Bu, 2)
    out = Matrix{K}(undef, length(rows_v), du)
    fill!(out, zero(K))
    iu = 1
    iv = 1
    nu = length(rows_u)
    nv = length(rows_v)
    @inbounds while iu <= nu && iv <= nv
        ru = rows_u[iu]
        rv = rows_v[iv]
        if ru == rv
            for c in 1:du
                out[iv, c] = Bu[iu, c]
            end
            iu += 1
            iv += 1
        elseif ru < rv
            iu += 1
        else
            iv += 1
        end
    end
    return out
end

function _rowcol_activity_from_plan(flat_masks::Vector{BitVector},
                                    inj_masks::Vector{BitVector},
                                    plan::ZnPushforwardPlan,
                                    n::Int)
    cols_at = [Int[] for _ in 1:n]
    rows_at = [Int[] for _ in 1:n]

    @inbounds for (iloc, fidx) in enumerate(plan.flat_idxs)
        mask = flat_masks[fidx]
        t = findnext(mask, 1)
        while t !== nothing
            push!(cols_at[t], iloc)
            t = findnext(mask, t + 1)
        end
    end

    @inbounds for (jloc, didx) in enumerate(plan.inj_idxs)
        mask = inj_masks[didx]
        t = findnext(mask, 1)
        while t !== nothing
            push!(rows_at[t], jloc)
            t = findnext(mask, t + 1)
        end
    end
    return cols_at, rows_at
end

"""
    _pmodule_from_pushforward_plan(P, FG, plan, flat_masks, inj_masks)

Construct the strict pushed-forward `PModule` directly from cached pushforward plan
and generator membership masks, without materializing an intermediate FringeModule.
"""
function _pmodule_from_pushforward_plan(P::AbstractPoset,
                                        FG::Flange{K},
                                        plan::ZnPushforwardPlan,
                                        flat_masks::Vector{BitVector},
                                        inj_masks::Vector{BitVector}) where {K}
    field = FG.field
    n = nvertices(P)
    cols_at, rows_at = _rowcol_activity_from_plan(flat_masks, inj_masks, plan, n)

    B = Vector{Matrix{K}}(undef, n)
    dims = zeros(Int, n)
    @inbounds for q in 1:n
        cols = cols_at[q]
        rows = rows_at[q]
        if isempty(cols) || isempty(rows)
            B[q] = Matrix{K}(undef, length(rows), 0)
            dims[q] = 0
            continue
        end
        phi_q = @view FG.phi[rows, cols]
        Bq = FieldLinAlg.colspace(field, phi_q)
        B[q] = Bq
        dims[q] = size(Bq, 2)
    end

    edge_maps = Dict{Tuple{Int,Int},Matrix{K}}()
    C = cover_edges(P)
    @inbounds for (u, v) in C
        du = dims[u]
        dv = dims[v]
        if du == 0 || dv == 0
            Z = Matrix{K}(undef, dv, du)
            fill!(Z, zero(K))
            edge_maps[(u, v)] = Z
            continue
        end
        Im = _gather_rows_subset(rows_at[u], rows_at[v], B[u], field)
        X = FieldLinAlg.solve_fullcolumn(field, B[v], Im)
        edge_maps[(u, v)] = X
    end
    return PModule{K}(P, dims, edge_maps; field=field)
end

@inline function _monomialize_phi_with_zero_pairs(phi::AbstractMatrix{K},
                                                   zero_pairs::Vector{Tuple{Int,Int}}) where {K}
    Phi = Matrix{K}(phi)
    @inbounds for (j, i) in zero_pairs
        Phi[j, i] = zero(K)
    end
    return Phi
end


"Collect the per-axis critical coordinates needed to make all signatures constant."
function _critical_coords(flats::Vector{IndFlat{N}}, injectives::Vector{IndInj{N}}) where {N}
    n = isempty(flats) ? (isempty(injectives) ? 0 : length(injectives[1].b)) : length(flats[1].b)
    coords = [Int[] for _ in 1:n]

    # Flats contribute thresholds for predicates g[i] >= b[i].
    for F in flats
        @inbounds for i in 1:n
            F.tau.coords[i] && continue
            push!(coords[i], F.b[i])
        end
    end

    # Injectives appear in signatures via their complements:
    #   g in complement(E)  <=>  g[i] >= (b[i] + 1) for all constrained coordinates.
    for E in injectives
        @inbounds for i in 1:n
            E.tau.coords[i] && continue
            push!(coords[i], E.b[i] + 1)
        end
    end

    for i in 1:n
        sort!(coords[i])
        unique!(coords[i])
    end
    return ntuple(i -> coords[i], n)
end

"Uptight poset on signatures by componentwise inclusion."
function _uptight_from_signatures(sig_y::PackedSignatureRows, sig_z::PackedSignatureRows)
    rN = length(sig_y)
    leq = falses(rN, rN)
    ymask = _word_lastmask(sig_y.bitlen)
    zmask = _word_lastmask(sig_z.bitlen)
    @inbounds for i in 1:rN
        leq[i, i] = true
    end
    @inbounds for i in 1:rN, j in 1:rN
        leq[i, j] = _sig_subset_words(sig_y.words, i, sig_y.words, j, ymask) &&
                    _sig_subset_words(sig_z.words, i, sig_z.words, j, zmask)
    end
    return FinitePoset(leq; check=false)
end

function _uptight_from_signatures(sig_y::AbstractVector{<:AbstractVector{Bool}},
                                  sig_z::AbstractVector{<:AbstractVector{Bool}})
    rN = length(sig_y)
    rN == length(sig_z) || error("_uptight_from_signatures: length mismatch")
    m = rN == 0 ? 0 : length(sig_y[1])
    r = rN == 0 ? 0 : length(sig_z[1])
    my = cld(max(m, 1), 64)
    mz = cld(max(r, 1), 64)
    py = _pack_signature_rows(sig_y, m, Val(my))
    pz = _pack_signature_rows(sig_z, r, Val(mz))
    return _uptight_from_signatures(py, pz)
end

"Images of the chosen generator upsets/downsets on the encoded poset P."
function _images_on_P(P::AbstractPoset,
                      sig_y::PackedSignatureRows,
                      sig_z::PackedSignatureRows,
                      flat_idxs::AbstractVector{<:Integer},
                      inj_idxs::AbstractVector{<:Integer})
    nflat = sig_y.bitlen
    ninj = sig_z.bitlen
    flat_masks, inj_masks = _build_pushforward_masks(sig_y, sig_z, nflat, ninj)
    return _images_on_P_from_masks(P, flat_masks, inj_masks, flat_idxs, inj_idxs)
end

function _images_on_P(P::AbstractPoset,
                      sig_y::AbstractVector{<:AbstractVector{Bool}},
                      sig_z::AbstractVector{<:AbstractVector{Bool}},
                      flat_idxs::AbstractVector{<:Integer},
                      inj_idxs::AbstractVector{<:Integer})
    n = length(sig_y)
    n == length(sig_z) || error("_images_on_P: signature length mismatch")
    nflat = n == 0 ? 0 : length(sig_y[1])
    ninj = n == 0 ? 0 : length(sig_z[1])
    py = _pack_signature_rows(sig_y, nflat, Val(cld(max(nflat, 1), 64)))
    pz = _pack_signature_rows(sig_z, ninj, Val(cld(max(ninj, 1), 64)))
    return _images_on_P(P, py, pz, flat_idxs, inj_idxs)
end

"Zero out entries that are forced to be 0 by disjointness of labels (monomiality)."
function _monomialize_phi(phi::AbstractMatrix{K}, Uhat::AbstractVector{<:Upset}, Dhat::AbstractVector{<:Downset}) where {K}
    Phi = Matrix{K}(phi)
    for j in 1:length(Dhat), i in 1:length(Uhat)
        if !intersects(Uhat[i], Dhat[j])
            Phi[j,i] = zero(K)
        end
    end
    return Phi
end

function _collect_encoding_generators(FGs::Union{AbstractVector{<:Flange}, Tuple{Vararg{Flange}}})
    length(FGs) > 0 || error("_collect_encoding_generators: need at least one flange")
    n = FGs[1].n
    @inbounds for FG in FGs
        FG.n == n || error("_collect_encoding_generators: dimension mismatch")
    end

    flats_all = IndFlat{n}[]
    injectives_all = IndInj{n}[]
    flat_seen = Dict{_GENERATOR_KEY, Int}()
    inj_seen  = Dict{_GENERATOR_KEY, Int}()

    @inbounds for FG in FGs
        for F in FG.flats
            key = _flat_key(F)
            if !haskey(flat_seen, key)
                push!(flats_all, F)
                flat_seen[key] = length(flats_all)
            end
        end
        for E in FG.injectives
            key = _inj_key(E)
            if !haskey(inj_seen, key)
                push!(injectives_all, E)
                inj_seen[key] = length(injectives_all)
            end
        end
    end

    flat_keys = collect(keys(flat_seen))
    inj_keys = collect(keys(inj_seen))
    return n, flats_all, injectives_all, flat_keys, inj_keys
end

# ----------------------------- Public API --------------------------------------

"""
    encode_poset_from_flanges(FGs, opts::EncodingOptions; poset_kind=:signature) -> (P, pi)

Construct only the finite encoding poset `P` and classifier `pi : Z^n -> P`
from the union of all flat and injective labels appearing in the given Z^n
flange presentations.

Arguments
- `FGs`: a vector (or tuple) of `Flange` objects. All inputs must have the same
  ambient dimension `n`.
- `opts`: an `EncodingOptions` (required).
  - `opts.backend` must be `:auto` or `:zn`.
  - `opts.max_regions` caps the number of distinct regions/signatures (default: 200_000).
- `poset_kind`: `:signature` (structured, default) or `:dense` (materialized `FinitePoset`).

This is the "finite encoding poset" step: extract critical coordinates, form the
product decomposition into finitely many slabs, sample one representative per cell,
and quotient by equal (y,z)-signatures.

Best practice:
- keep `poset_kind=:signature` as the cheap/default path for inspection and
  downstream kernels that only need the region order abstractly;
- request `poset_kind=:dense` only when later code truly needs a materialized
  `FinitePoset`;
- pair the result with [`compile_zn_cache`](@ref) when you expect repeated
  query or region-geometry work.

Use `_pushforward_flange_to_fringe(P, pi, FG)` to push a flange presentation
down to a finite fringe presentation on `P` without rebuilding the encoding.
"""
function encode_poset_from_flanges(FGs::Union{AbstractVector{<:Flange}, Tuple{Vararg{Flange}}},
                                   opts::EncodingOptions;
                                   poset_kind::Symbol = :signature,
                                   session_cache::Union{Nothing,SessionCache}=nothing)
    if opts.backend != :auto && opts.backend != :zn
        error("encode_poset_from_flanges: EncodingOptions.backend must be :auto or :zn")
    end
    max_regions = opts.max_regions === nothing ? 200_000 : Int(opts.max_regions)
    n, flats_all, injectives_all, flat_keys, inj_keys = _collect_encoding_generators(FGs)
    encoding_fp = _generator_keyset_fingerprint(n, flat_keys, inj_keys)

    if session_cache !== nothing
        cached = _session_get_zn_encoding_artifact(session_cache, encoding_fp, poset_kind, max_regions)
        if cached !== nothing
            art = cached::NamedTuple
            return art.P, art.pi
        end
    end

    coords = _critical_coords(flats_all, injectives_all)

    MY = max(1, cld(length(flats_all), 64))
    MZ = max(1, cld(length(injectives_all), 64))
    sig_y, sig_z, reps, sig_to_region, cell_shape, cell_strides, cell_to_region = _collect_signatures_incremental(
        coords, flats_all, injectives_all, max_regions, Val(MY), Val(MZ)
    )

    if poset_kind == :signature
        P = SignaturePoset(sig_y, sig_z)
    elseif poset_kind == :dense
        P = _uptight_from_signatures(sig_y, sig_z)
    else
        error("encode_poset_from_flanges: poset_kind must be :signature or :dense")
    end

    pi = ZnEncodingMap(n, coords, sig_y, sig_z, reps, flats_all, injectives_all, sig_to_region,
                       cell_shape, cell_strides, cell_to_region)
    if session_cache !== nothing
        _session_set_zn_encoding_artifact!(session_cache, encoding_fp, poset_kind, max_regions, (P=P, pi=pi))
    end
    return P, pi
end

# Keyword-friendly overloads (opts may be nothing).
encode_poset_from_flanges(FGs::Union{AbstractVector{<:Flange}, Tuple{Vararg{Flange}}};
                          opts::EncodingOptions=EncodingOptions(),
                          poset_kind::Symbol = :signature,
                          session_cache::Union{Nothing,SessionCache}=nothing) =
    encode_poset_from_flanges(FGs, opts; poset_kind = poset_kind, session_cache=session_cache)

# Small-arity overloads (avoid "varargs then opts" signatures).
function encode_poset_from_flanges(FG::Flange, opts::EncodingOptions;
                                   poset_kind::Symbol = :signature)
    return encode_poset_from_flanges((FG,), opts; poset_kind = poset_kind)
end

encode_poset_from_flanges(FG::Flange;
                          opts::EncodingOptions=EncodingOptions(),
                          poset_kind::Symbol = :signature) =
    encode_poset_from_flanges((FG,), opts; poset_kind = poset_kind)

function encode_poset_from_flanges(FG1::Flange, FG2::Flange, opts::EncodingOptions;
                                   poset_kind::Symbol = :signature)
    return encode_poset_from_flanges((FG1, FG2), opts; poset_kind = poset_kind)
end

encode_poset_from_flanges(FG1::Flange, FG2::Flange;
                          opts::EncodingOptions=EncodingOptions(),
                          poset_kind::Symbol = :signature) =
    encode_poset_from_flanges((FG1, FG2), opts; poset_kind = poset_kind)

function encode_poset_from_flanges(FG1::Flange, FG2::Flange, FG3::Flange, opts::EncodingOptions;
                                   poset_kind::Symbol = :signature)
    return encode_poset_from_flanges((FG1, FG2, FG3), opts; poset_kind = poset_kind)
end

encode_poset_from_flanges(FG1::Flange, FG2::Flange, FG3::Flange;
                          opts::EncodingOptions=EncodingOptions(),
                          poset_kind::Symbol = :signature) =
    encode_poset_from_flanges((FG1, FG2, FG3), opts; poset_kind = poset_kind)

"""
    _pushforward_flange_to_fringe(P, pi, FG; strict=true) -> FringeModule{K}

Convert a Z^n flange presentation `FG` into a fringe presentation on the finite
encoding poset `P` determined by `pi`.

Interpretation (paper-level):
This is the direct "flange -> fringe" bridge (cf. Miller, Remark 6.14): once an
encoding `pi : Z^n -> P` is fixed, the fringe presentation on `P` is obtained by

1. pushing forward each flat label to an upset in `P`,
2. pushing forward each injective label to a downset in `P`, and
3. reusing the scalar coefficient matrix `Phi` from the flange presentation,
   with entries forced to zero when the pushed labels are disjoint on `P`.

Safety contract:
- If `strict=true` (default), every generator label in `FG` must occur among the
  generators stored in `pi` up to equality of the underlying set (same `b` and same
  `tau`). This guarantees that membership is constant on `pi`-regions and the image
  upset/downset is computed purely by reading signature bits.

- If `strict=false`, membership is tested only on region representatives `pi.reps[t]`.
  This is only correct if each label of `FG` is constant on each region of `pi`.
"""
function _pushforward_flange_to_fringe(P::AbstractPoset, pi::ZnEncodingMap, FG::Flange{K};
                                       strict::Bool=true,
                                       session_cache::Union{Nothing,SessionCache}=nothing,
                                       poset_kind::Symbol=:signature) where {K}
    FG.n == pi.n || error("_pushforward_flange_to_fringe: dimension mismatch (FG.n != pi.n)")
    nvertices(P) == length(pi.sig_y) || error("_pushforward_flange_to_fringe: P incompatible with pi (nvertices(P) != length(pi.sig_y))")
    length(pi.sig_y) == length(pi.sig_z) || error("_pushforward_flange_to_fringe: malformed pi (sig_y and sig_z lengths differ)")

    if strict
        flange_fp = UInt64(0)
        field_key = UInt(0)
        if session_cache !== nothing
            flange_fp = _flange_presentation_fingerprint(FG)
            field_key = _field_cache_key(FG.field)
            cached = _session_get_zn_pushforward_fringe(session_cache, pi.encoding_fingerprint, poset_kind, flange_fp, field_key)
            if cached !== nothing
                return cached::FringeModule{K}
            end
        end

        plan, flat_masks, inj_masks = _strict_pushforward_plan_and_masks(pi, FG; session_cache=session_cache)
        H = _strict_pushforward_fringe_from_plan(P, FG, plan, flat_masks, inj_masks)
        if session_cache !== nothing
            _session_set_zn_pushforward_fringe!(session_cache, pi.encoding_fingerprint, poset_kind, flange_fp, field_key, H)
        end
        return H
    else
        # Fallback: decide membership by evaluating on region representatives.
        m = length(FG.flats)
        r = length(FG.injectives)
        Uhat = Vector{Upset}(undef, m)
        Dhat = Vector{Downset}(undef, r)

        for i in 1:m
            mask = BitVector([in_flat(FG.flats[i], pi.reps[t]) for t in 1:nvertices(P)])
            Uhat[i] = upset_closure(P, mask)
        end
        for j in 1:r
            mask = BitVector([in_inj(FG.injectives[j], pi.reps[t]) for t in 1:nvertices(P)])
            Dhat[j] = downset_closure(P, mask)
        end

        Phi = _monomialize_phi(FG.phi, Uhat, Dhat)
        return FringeModule{K}(P, Uhat, Dhat, Phi; field=FG.field)
    end
end

"""
    locate(pi::ZnEncodingMap, g) -> Int

Return the region index for a point query against a [`ZnEncodingMap`](@ref).

Accepted inputs:
- `g::AbstractVector{<:Integer}`
- `g::NTuple{N,<:Integer}`

Convenience for slice and geometry code:
- `x::AbstractVector{<:AbstractFloat}` and floating-point tuples are rounded
  componentwise to the nearest integer lattice point and then located.

Return value:
- an integer in `1:length(pi.reps)` when the point lands in a represented
  region;
- `0` when the rounded lattice point has no stored signature. This should be
  read as "outside / not represented" by downstream code.

Performance notes:
- when [`has_direct_lookup`](@ref) is `true`, `locate(...)` uses the prebuilt
  slab-cell lookup table;
- otherwise it hashes the point's `(y, z)` signature on demand.

Use [`check_zn_query_point`](@ref) when query shape comes from user input and
use [`zn_query_summary`](@ref) when you want a cheap exploratory report instead
of just the region id.

Implementation note:
These methods must be base cases. They must not call `locate` again on an integer vector/tuple,
otherwise it is easy to introduce infinite mutual recursion and a StackOverflowError.
"""
@inline function _cell_linear_index(pi::ZnEncodingMap{N}, g::NTuple{N,Int}) where {N}
    idx = 1
    @inbounds for i in 1:N
        s = searchsortedlast(pi.coords[i], g[i]) # 0-based slab index
        idx += s * pi.cell_strides[i]
    end
    return idx
end

@inline function _cell_linear_index(pi::ZnEncodingMap, g::AbstractVector{<:Integer})
    idx = 1
    @inbounds for i in 1:pi.n
        s = searchsortedlast(pi.coords[i], Int(g[i])) # 0-based slab index
        idx += s * pi.cell_strides[i]
    end
    return idx
end

function locate(pi::ZnEncodingMap{N,MY,MZ}, g::AbstractVector{<:Integer}) where {N,MY,MZ}
    length(g) == pi.n || error("locate: expected a vector of length $(pi.n), got $(length(g))")
    if pi.cell_to_region !== nothing
        idx = _cell_linear_index(pi, g)
        return @inbounds pi.cell_to_region[idx]
    end
    return get(pi.sig_to_region, _sigkey_at(g, pi.flats, pi.injectives, Val(MY), Val(MZ)), 0)
end

function locate(pi::ZnEncodingMap{N,MY,MZ}, g::NTuple{N,<:Integer}) where {N,MY,MZ}
    N == pi.n || error("locate: expected a tuple of length $(pi.n), got $(N)")
    if pi.cell_to_region !== nothing
        gi = ntuple(i -> Int(g[i]), N)
        idx = _cell_linear_index(pi, gi)
        return @inbounds pi.cell_to_region[idx]
    end
    return get(pi.sig_to_region, _sigkey_at(g, pi.flats, pi.injectives, Val(MY), Val(MZ)), 0)
end

# Convenience only: lets slice-based code pass Float64 points.
# This method must NOT accept integer vectors, otherwise it can recurse forever.
function locate(pi::ZnEncodingMap, x::AbstractVector{<:AbstractFloat})
    length(x) == pi.n || error("locate: expected a vector of length $(pi.n), got $(length(x))")
    g = ntuple(i -> round(Int, x[i]), pi.n)
    return locate(pi, g)
end

@inline function locate(pi::ZnEncodingMap{N}, x::NTuple{N,<:AbstractFloat}) where {N}
    N == pi.n || error("locate: expected a tuple of length $(pi.n), got $(N)")
    g = ntuple(i -> round(Int, x[i]), N)
    return locate(pi, g)
end

locate(cache::ZnEncodingCache, g) = locate(cache.pi, g)

@inline function _locate_col(pi::ZnEncodingMap{N,MY,MZ}, X::AbstractMatrix{<:Integer}, col::Int) where {N,MY,MZ}
    if pi.cell_to_region !== nothing
        idx = 1
        @inbounds for i in 1:N
            s = searchsortedlast(pi.coords[i], Int(X[i, col]))
            idx += s * pi.cell_strides[i]
        end
        return @inbounds pi.cell_to_region[idx]
    end
    g = ntuple(i -> Int(@inbounds X[i, col]), N)
    return get(pi.sig_to_region, _sigkey_at(g, pi.flats, pi.injectives, Val(MY), Val(MZ)), 0)
end

@inline function _locate_col(pi::ZnEncodingMap{N,MY,MZ}, X::AbstractMatrix{<:AbstractFloat}, col::Int) where {N,MY,MZ}
    if pi.cell_to_region !== nothing
        idx = 1
        @inbounds for i in 1:N
            s = searchsortedlast(pi.coords[i], round(Int, X[i, col]))
            idx += s * pi.cell_strides[i]
        end
        return @inbounds pi.cell_to_region[idx]
    end
    g = ntuple(i -> round(Int, @inbounds X[i, col]), N)
    return get(pi.sig_to_region, _sigkey_at(g, pi.flats, pi.injectives, Val(MY), Val(MZ)), 0)
end

"""
    locate_many!(dest, pi_or_cache, X; threaded=true) -> dest

Batch Zn classifier evaluation with one query per column of `X`.

Accepted owners:
- `pi::ZnEncodingMap`
- `cache::ZnEncodingCache`
- `enc::CompiledEncoding{<:ZnEncodingMap}`

Accepted matrix contracts:
- `AbstractMatrix{<:Integer}` for native lattice queries,
- `AbstractMatrix{<:AbstractFloat}` for rounded floating-point queries.

Return value:
- `dest[j]` is the region id for the `j`-th query column,
- `0` means the rounded query point is outside the represented region set.

Performance notes:
- when [`has_direct_lookup`](@ref) is `true`, each query column uses the cheap
  slab-cell lookup path;
- caches help when you want repeated classification together with an attached
  region poset;
- keep `poset_kind=:signature` on the cache-building side unless you truly need
  a dense finite-poset realization elsewhere.
"""
function locate_many!(dest::AbstractVector{<:Integer},
                      pi_or_cache::Union{ZnEncodingMap, ZnEncodingCache, CompiledEncoding{<:ZnEncodingMap}},
                      X::AbstractMatrix{<:Integer};
                      threaded::Bool=true)
    pi = _unwrap_zn_pi(pi_or_cache)
    size(X, 1) == pi.n || error("locate_many!: expected X with $(pi.n) rows, got $(size(X, 1))")
    length(dest) == size(X, 2) || error("locate_many!: destination length mismatch")
    np = size(X, 2)
    if threaded && nthreads() > 1 && np >= 1024
        Threads.@threads :static for j in 1:np
            @inbounds dest[j] = _locate_col(pi, X, j)
        end
    else
        @inbounds for j in 1:np
            dest[j] = _locate_col(pi, X, j)
        end
    end
    return dest
end

function locate_many!(dest::AbstractVector{<:Integer},
                      pi_or_cache::Union{ZnEncodingMap, ZnEncodingCache, CompiledEncoding{<:ZnEncodingMap}},
                      X::AbstractMatrix{<:AbstractFloat};
                      threaded::Bool=true)
    pi = _unwrap_zn_pi(pi_or_cache)
    size(X, 1) == pi.n || error("locate_many!: expected X with $(pi.n) rows, got $(size(X, 1))")
    length(dest) == size(X, 2) || error("locate_many!: destination length mismatch")
    np = size(X, 2)
    if threaded && nthreads() > 1 && np >= 1024
        Threads.@threads :static for j in 1:np
            @inbounds dest[j] = _locate_col(pi, X, j)
        end
    else
        @inbounds for j in 1:np
            dest[j] = _locate_col(pi, X, j)
        end
    end
    return dest
end

function locate_many(pi_or_cache::Union{ZnEncodingMap, ZnEncodingCache, CompiledEncoding{<:ZnEncodingMap}},
                     X::AbstractMatrix{<:Integer}; threaded::Bool=true)
    out = Vector{Int}(undef, size(X, 2))
    return locate_many!(out, pi_or_cache, X; threaded=threaded)
end

function locate_many(pi_or_cache::Union{ZnEncodingMap, ZnEncodingCache, CompiledEncoding{<:ZnEncodingMap}},
                     X::AbstractMatrix{<:AbstractFloat}; threaded::Bool=true)
    out = Vector{Int}(undef, size(X, 2))
    return locate_many!(out, pi_or_cache, X; threaded=threaded)
end

# ---------------------------------------------------------------------------
# Lattice counting helpers for Z^n encoders (ZnEncodingMap)
#
# ZnEncodingMap stores, for each coordinate axis i, a sorted list pi.coords[i]
# of "critical coordinates". These induce slabs (integer intervals) on each axis:
#
#   s = 0: (-inf, coords[1]-1]
#   s = 1: [coords[1], coords[2]-1]
#   ...
#   s = k: [coords[k], +inf)
#
# A product of slabs gives a cell of Z^n on which the (y,z)-signature is constant.
# This makes exact counting in a box feasible by iterating over slab-cells rather
# than over every lattice point (when the number of relevant cells is moderate).
# ---------------------------------------------------------------------------

@inline function _slab_index(coords_i::Vector{Int}, x::Int)
    # Return s in 0:length(coords_i) such that x lies in slab s.
    # For empty coords_i (a completely free axis), there is only one slab, indexed by 0.
    isempty(coords_i) && return 0
    return searchsortedlast(coords_i, x)
end

@inline function _slab_rep(coords_i::Vector{Int}, s::Int)
    # Choose an integer representative inside slab s.
    # This is used only for signature lookup (locate); any point in the slab works.
    isempty(coords_i) && return 0
    if s <= 0
        return coords_i[1] - 1
    elseif s >= length(coords_i)
        return coords_i[end]
    else
        return coords_i[s]
    end
end

@inline function _slab_interval(s::Int, coords_i::Vector{Int})
    # Return inclusive integer bounds (lo, hi) of slab s induced by coords_i.
    # Slabs are indexed by s in 0:length(coords_i), with the convention:
    #   s = 0: (-Inf, coords[1]-1]
    #   1 <= s <= m-1: [coords[s], coords[s+1]-1]
    #   s = m: [coords[m], +Inf)
    #
    # The return values are clipped to the Int range; this is sufficient because
    # callers only intersect these bounds with an Int box [a,b].
    isempty(coords_i) && return (typemin(Int), typemax(Int))

    m = length(coords_i)
    if s <= 0
        c1 = coords_i[1]
        # Avoid wrap-around on typemin(Int)-1.
        c1 == typemin(Int) && return (1, 0)  # empty slab
        return (typemin(Int), c1 - 1)
    elseif s >= m
        return (coords_i[m], typemax(Int))
    else
        cnext = coords_i[s + 1]
        cnext == typemin(Int) && return (1, 0)  # empty slab (degenerate)
        return (coords_i[s], cnext - 1)
    end
end


@inline function _slab_count_in_interval(coords_i::Vector{Int}, s::Int, a::Int, b::Int, ::Type{Int})
    # Exact count of integers in (slab s) intersect [a,b], returned as Int.
    if isempty(coords_i)
        # Only one slab; everything lies in it.
        return b - a + 1
    end
    lo, hi = _slab_interval(s, coords_i)
    L = max(a, lo)
    U = min(b, hi)
    return (L <= U) ? (U - L + 1) : 0
end

@inline function _slab_count_in_interval(coords_i::Vector{Int}, s::Int, a::Int, b::Int, ::Type{BigInt})
    # Same as above, but safe for huge intervals (avoids Int overflow in U-L+1).
    if isempty(coords_i)
        return BigInt(b) - BigInt(a) + 1
    end
    lo, hi = _slab_interval(s, coords_i)
    L = max(BigInt(a), BigInt(lo))
    U = min(BigInt(b), BigInt(hi))
    return (L <= U) ? (U - L + 1) : BigInt(0)
end

@inline function _as_int_tuple(::Val{N}, x) where {N}
    if x isa NTuple{N,Int}
        return x
    elseif x isa NTuple{N,<:Integer}
        return ntuple(i -> Int(x[i]), N)
    elseif x isa AbstractVector
        length(x) == N || error("region_weights(Zn): box dimension mismatch")
        return ntuple(i -> Int(x[i]), N)
    else
        v = collect(x)
        length(v) == N || error("region_weights(Zn): box dimension mismatch")
        return ntuple(i -> Int(v[i]), N)
    end
end

function _box_lattice_size_big(a::Vector{Int}, b::Vector{Int})
    # Total lattice points in the box [a,b] in Z^n, as BigInt (overflow-safe).
    n = length(a)
    tot = BigInt(1)
    @inbounds for i in 1:n
        li = BigInt(b[i]) - BigInt(a[i]) + 1
        if li <= 0
            return BigInt(0)
        end
        tot *= li
    end
    return tot
end

function _box_lattice_size_big(a::NTuple{N,Int}, b::NTuple{N,Int}) where {N}
    tot = BigInt(1)
    @inbounds for i in 1:N
        li = BigInt(b[i]) - BigInt(a[i]) + 1
        if li <= 0
            return BigInt(0)
        end
        tot *= li
    end
    return tot
end

function _choose_count_type(count_type, total_points_big::BigInt)
    # Decide whether to use Int or BigInt for exact counting.
    if count_type === :auto
        return (total_points_big <= BigInt(typemax(Int))) ? Int : BigInt
    elseif count_type === Int || count_type === BigInt
        return count_type
    else
        error("region_weights: count_type must be Int, BigInt, or :auto")
    end
end

function _region_weights_cells(pi::ZnEncodingMap,
                              a::AbstractVector{Int},
                              b::AbstractVector{Int},
                              lo::Vector{Int},
                              hi::Vector{Int};
                              strict::Bool=true,
                              T::Type=Int)
    # Exact counting by iterating over slab-cells intersecting the box.
    #
    # Speed notes:
    # - we only iterate over slabs that actually meet [a[i], b[i]] (lo..hi)
    # - we precompute (rep, count) per slab per axis
    # - we special-case n<=3 to use tuple-based locate(pi, (..)) for less overhead

    n = pi.n
    nregions = length(pi.sig_y)
    w = zeros(T, nregions)

    # Precompute per-axis slab reps and slab intersection counts.
    reps = Vector{Vector{Int}}(undef, n)
    cnts = Vector{Vector{T}}(undef, n)
    @inbounds for i in 1:n
        ci = pi.coords[i]
        nslabs = hi[i] - lo[i] + 1
        reps_i = Vector{Int}(undef, nslabs)
        cnts_i = Vector{T}(undef, nslabs)
        k = 1
        for s in lo[i]:hi[i]
            reps_i[k] = _slab_rep(ci, s)
            cnts_i[k] = _slab_count_in_interval(ci, s, a[i], b[i], T)
            k += 1
        end
        reps[i] = reps_i
        cnts[i] = cnts_i
    end

    # Fast paths for n <= 3 (type-stable tuples).
    if n == 1
        reps1 = reps[1]; cnt1 = cnts[1]
        zT = zero(T)
        @inbounds for i1 in eachindex(cnt1)
            c1 = cnt1[i1]
            c1 == zT && continue
            t = locate(pi, (reps1[i1],))
            if t == 0
                strict && error("region_weights: cell representative left encoding domain (locate==0)")
                continue
            end
            w[t] += c1
        end
        return w
    elseif n == 2
        reps1 = reps[1]; cnt1 = cnts[1]
        reps2 = reps[2]; cnt2 = cnts[2]
        zT = zero(T)
        @inbounds for i1 in eachindex(cnt1)
            c1 = cnt1[i1]
            c1 == zT && continue
            g1 = reps1[i1]
            for i2 in eachindex(cnt2)
                c2 = cnt2[i2]
                c2 == zT && continue
                t = locate(pi, (g1, reps2[i2]))
                if t == 0
                    strict && error("region_weights: cell representative left encoding domain (locate==0)")
                    continue
                end
                w[t] += c1 * c2
            end
        end
        return w
    elseif n == 3
        reps1 = reps[1]; cnt1 = cnts[1]
        reps2 = reps[2]; cnt2 = cnts[2]
        reps3 = reps[3]; cnt3 = cnts[3]
        zT = zero(T)
        @inbounds for i1 in eachindex(cnt1)
            c1 = cnt1[i1]
            c1 == zT && continue
            g1 = reps1[i1]
            for i2 in eachindex(cnt2)
                c2 = cnt2[i2]
                c2 == zT && continue
                g2 = reps2[i2]
                c12 = c1 * c2
                for i3 in eachindex(cnt3)
                    c3 = cnt3[i3]
                    c3 == zT && continue
                    t = locate(pi, (g1, g2, reps3[i3]))
                    if t == 0
                        strict && error("region_weights: cell representative left encoding domain (locate==0)")
                        continue
                    end
                    w[t] += c12 * c3
                end
            end
        end
        return w
    end

    # Generic n: odometer over per-axis slab indices (no allocations per cell).
    idx = ones(Int, n)
    maxidx = Int[length(cnts[i]) for i in 1:n]
    g = Vector{Int}(undef, n)
    zT = zero(T)
    oneT = one(T)

    while true
        vol = oneT
        @inbounds for i in 1:n
            c = cnts[i][idx[i]]
            if c == zT
                vol = zT
                break
            end
            vol *= c
            g[i] = reps[i][idx[i]]
        end

        if vol != zT
            t = locate(pi, g)
            if t == 0
                strict && error("region_weights: cell representative left encoding domain (locate==0)")
            else
                w[t] += vol
            end
        end

        # increment idx (odometer)
        k = n
        @inbounds while k >= 1 && idx[k] == maxidx[k]
            k -= 1
        end
        k == 0 && break
        idx[k] += 1
        @inbounds for j in (k+1):n
            idx[j] = 1
        end
    end

    return w
end

function _region_weights_cells(pi::ZnEncodingMap,
                              a::NTuple{N,Int},
                              b::NTuple{N,Int},
                              lo::Vector{Int},
                              hi::Vector{Int};
                              strict::Bool=true,
                              T::Type=Int) where {N}
    n = pi.n
    nregions = length(pi.sig_y)
    w = zeros(T, nregions)

    reps = Vector{Vector{Int}}(undef, n)
    cnts = Vector{Vector{T}}(undef, n)
    @inbounds for i in 1:n
        ci = pi.coords[i]
        nslabs = hi[i] - lo[i] + 1
        reps_i = Vector{Int}(undef, nslabs)
        cnts_i = Vector{T}(undef, nslabs)
        k = 1
        for s in lo[i]:hi[i]
            reps_i[k] = _slab_rep(ci, s)
            cnts_i[k] = _slab_count_in_interval(ci, s, a[i], b[i], T)
            k += 1
        end
        reps[i] = reps_i
        cnts[i] = cnts_i
    end

    if n == 1
        reps1 = reps[1]; cnt1 = cnts[1]
        zT = zero(T)
        @inbounds for i1 in eachindex(cnt1)
            c1 = cnt1[i1]
            c1 == zT && continue
            t = locate(pi, (reps1[i1],))
            if t == 0
                strict && error("region_weights: cell representative left encoding domain (locate==0)")
            else
                w[t] += c1
            end
        end
        return w
    elseif n == 2
        reps1, reps2 = reps
        cnt1, cnt2 = cnts
        zT = zero(T)
        @inbounds for i1 in eachindex(cnt1)
            c1 = cnt1[i1]
            c1 == zT && continue
            for i2 in eachindex(cnt2)
                c2 = cnt2[i2]
                c2 == zT && continue
                t = locate(pi, (reps1[i1], reps2[i2]))
                if t == 0
                    strict && error("region_weights: cell representative left encoding domain (locate==0)")
                else
                    w[t] += c1 * c2
                end
            end
        end
        return w
    elseif n == 3
        reps1, reps2, reps3 = reps
        cnt1, cnt2, cnt3 = cnts
        zT = zero(T)
        @inbounds for i1 in eachindex(cnt1)
            c1 = cnt1[i1]
            c1 == zT && continue
            for i2 in eachindex(cnt2)
                c2 = cnt2[i2]
                c2 == zT && continue
                for i3 in eachindex(cnt3)
                    c3 = cnt3[i3]
                    c3 == zT && continue
                    t = locate(pi, (reps1[i1], reps2[i2], reps3[i3]))
                    if t == 0
                        strict && error("region_weights: cell representative left encoding domain (locate==0)")
                    else
                        w[t] += c1 * c2 * c3
                    end
                end
            end
        end
        return w
    end

    idx = ones(Int, n)
    maxidx = [length(cnts[i]) for i in 1:n]
    g = Vector{Int}(undef, n)
    zT = zero(T)
    while true
        vol = one(T)
        @inbounds for i in 1:n
            c = cnts[i][idx[i]]
            if c == zT
                vol = zT
                break
            end
            vol *= c
            g[i] = reps[i][idx[i]]
        end

        if vol != zT
            t = locate(pi, g)
            if t == 0
                strict && error("region_weights: cell representative left encoding domain (locate==0)")
            else
                w[t] += vol
            end
        end

        k = n
        @inbounds while k >= 1 && idx[k] == maxidx[k]
            k -= 1
        end
        k == 0 && break
        idx[k] += 1
        @inbounds for j in (k+1):n
            idx[j] = 1
        end
    end

    return w
end

function _region_weights_points(pi::ZnEncodingMap,
                               a::AbstractVector{Int},
                               b::AbstractVector{Int};
                               strict::Bool=true,
                               T::Type=Int)
    # Exact enumeration of lattice points in [a,b] (only sensible for small boxes).
    n = pi.n
    nregions = length(pi.sig_y)
    w = zeros(T, nregions)
    oneT = one(T)

    if n == 1
        @inbounds for x1 in a[1]:b[1]
            t = locate(pi, (x1,))
            if t == 0
                strict && error("region_weights: point left encoding domain (locate==0)")
                continue
            end
            w[t] += oneT
        end
        return w
    elseif n == 2
        @inbounds for x1 in a[1]:b[1]
            for x2 in a[2]:b[2]
                t = locate(pi, (x1, x2))
                if t == 0
                    strict && error("region_weights: point left encoding domain (locate==0)")
                    continue
                end
                w[t] += oneT
            end
        end
        return w
    elseif n == 3
        @inbounds for x1 in a[1]:b[1]
            for x2 in a[2]:b[2]
                for x3 in a[3]:b[3]
                    t = locate(pi, (x1, x2, x3))
                    if t == 0
                        strict && error("region_weights: point left encoding domain (locate==0)")
                        continue
                    end
                    w[t] += oneT
                end
            end
        end
        return w
    end

    # Generic n: odometer on points.
    g = copy(a)
    while true
        t = locate(pi, g)
        if t == 0
            strict && error("region_weights: point left encoding domain (locate==0)")
        else
            w[t] += oneT
        end

        # increment g in the box [a,b]
        k = n
        @inbounds while k >= 1
            if g[k] < b[k]
                g[k] += 1
                for j in (k+1):n
                    g[j] = a[j]
                end
                break
            end
            k -= 1
        end
        k == 0 && break
    end

    return w
end

function _region_weights_points(pi::ZnEncodingMap,
                               a::NTuple{N,Int},
                               b::NTuple{N,Int};
                               strict::Bool=true,
                               T::Type=Int) where {N}
    n = pi.n
    nregions = length(pi.sig_y)
    w = zeros(T, nregions)
    oneT = one(T)

    if n == 1
        @inbounds for x1 in a[1]:b[1]
            t = locate(pi, (x1,))
            if t == 0
                strict && error("region_weights: point left encoding domain (locate==0)")
                continue
            end
            w[t] += oneT
        end
        return w
    elseif n == 2
        @inbounds for x1 in a[1]:b[1]
            for x2 in a[2]:b[2]
                t = locate(pi, (x1, x2))
                if t == 0
                    strict && error("region_weights: point left encoding domain (locate==0)")
                    continue
                end
                w[t] += oneT
            end
        end
        return w
    elseif n == 3
        @inbounds for x1 in a[1]:b[1]
            for x2 in a[2]:b[2]
                for x3 in a[3]:b[3]
                    t = locate(pi, (x1, x2, x3))
                    if t == 0
                        strict && error("region_weights: point left encoding domain (locate==0)")
                        continue
                    end
                    w[t] += oneT
                end
            end
        end
        return w
    end

    g = Vector{Int}(undef, n)
    @inbounds for i in 1:n
        g[i] = a[i]
    end
    while true
        t = locate(pi, g)
        if t == 0
            strict && error("region_weights: point left encoding domain (locate==0)")
        else
            w[t] += oneT
        end

        k = n
        @inbounds while k >= 1
            if g[k] < b[k]
                g[k] += 1
                for j in (k+1):n
                    g[j] = a[j]
                end
                break
            end
            k -= 1
        end
        k == 0 && break
    end

    return w
end

function _region_weights_sample(pi::ZnEncodingMap,
                               a::AbstractVector{Int},
                               b::AbstractVector{Int};
                               strict::Bool=true,
                               nsamples::Integer=50_000,
                               rng::Random.AbstractRNG=Random.default_rng())
    # Monte Carlo estimator for region weights in a box.
    #
    # Returns: (weights::Vector{Float64}, stderr::Vector{Float64}, total_points_big::BigInt)
    n = pi.n
    nregions = length(pi.sig_y)
    total_points_big = _box_lattice_size_big(a, b)
    total_points = Float64(total_points_big)  # may overflow to Inf if astronomically large

    ns = Int(nsamples)
    ns > 0 || error("region_weights: nsamples must be positive")

    counts = zeros(Int, nregions)
    ranges = [a[i]:b[i] for i in 1:n]
    g = Vector{Int}(undef, n)

    @inbounds for s in 1:ns
        for i in 1:n
            g[i] = rand(rng, ranges[i])
        end
        t = locate(pi, g)
        if t == 0
            strict && error("region_weights: sampled point left encoding domain (locate==0)")
            continue
        end
        counts[t] += 1
    end

    p = counts ./ ns
    weights = total_points .* p

    # Per-bin standard error of a Bernoulli proportion (good scale estimate).
    stderr = total_points .* sqrt.(p .* (1 .- p) ./ ns)

    return weights, stderr, counts, total_points_big
end

function _region_weights_sample(pi::ZnEncodingMap,
                               a::NTuple{N,Int},
                               b::NTuple{N,Int};
                               strict::Bool=true,
                               nsamples::Integer=50_000,
                               rng::Random.AbstractRNG=Random.default_rng()) where {N}
    n = pi.n
    nregions = length(pi.sig_y)
    total_points_big = _box_lattice_size_big(a, b)
    total_points = Float64(total_points_big)

    ns = Int(nsamples)
    ns > 0 || error("region_weights: nsamples must be positive")

    counts = zeros(Int, nregions)
    ranges = ntuple(i -> a[i]:b[i], n)
    g = Vector{Int}(undef, n)

    @inbounds for s in 1:ns
        for i in 1:n
            g[i] = rand(rng, ranges[i])
        end
        t = locate(pi, g)
        if t == 0
            strict && error("region_weights: sampled point left encoding domain (locate==0)")
            continue
        end
        counts[t] += 1
    end

    p = counts ./ ns
    weights = total_points .* p
    stderr = total_points .* sqrt.(p .* (1 .- p) ./ ns)

    return weights, stderr, counts, total_points_big
end

# Internal sampling helper returning (weights, stderr, counts, total_points).
"""
    region_weights(pi::ZnEncodingMap; box=nothing, method=:auto, kwargs...)

Compute a nonnegative weight for each region in a `ZnEncodingMap`.

The primary use of these weights is to form weighted statistics over regions
(e.g. cross validation or empirical risk) without iterating over every lattice
point in a box.

Keyword arguments

* `box` : either `nothing` (return a vector of ones) or a tuple `(a, b)` of
  integer vectors describing a half-open box `{g in Z^n : a <= g < b}`.

* `method` : one of `:cells`, `:points`, `:sample`, `:auto`.

  * `:cells`  : exact counting by iterating over unit cells in the box.
  * `:points` : exact counting by iterating over lattice points in the box.
  * `:sample` : Monte Carlo sampling of lattice points in the box.
  * `:auto`   : choose an exact method based on `count_type` and size limits.

* `count_type` : used only when `method == :auto`.

  * `:cells`  : request exact cell counts (default).
  * `:points` : request exact point counts.

* `max_cells`, `max_points` : safety limits used by `:auto` (and also enforced
  by the corresponding exact method). If the requested box size exceeds the
  relevant limit, an error containing the phrase "box too large" is thrown.

* `nsamples`, `rng` : sampling parameters for `method == :sample`.

* `strict` : if `true`, any sampled / enumerated point not belonging to a known
  region triggers an error.

* `return_info` : if `true`, return a `NamedTuple` with additional diagnostic
  fields. For exact methods this includes `method_used`; for sampling this
  includes `stderr`.

Best practice:
- keep `box=nothing` when you only need uniform region weights;
- keep `method=:auto` as the cheap/default path and override only when you know
  the box regime;
- use a cache-bearing object when weights will be part of repeated downstream
  workflows;
- `locate(...) == 0` inside the requested box means the encoding does not
  represent that lattice point; with `strict=true` this is treated as an error.
"""
function region_weights(
    pi::ZnEncodingMap{N};
    box=nothing,
    method::Symbol=:auto,
    count_type=:auto,
    max_cells::Int=typemax(Int),
    max_points::Int=typemax(Int),
    nsamples::Int=50_000,
    rng::Union{Nothing,Random.AbstractRNG}=nothing,
    strict::Bool=true,
    return_info::Bool=false,
    alpha::Real=0.05,
) where {N}
    nregions = length(pi.sig_y)

    # No box: return uniform weights (useful when downstream code does not
    # care about geometric weighting).
    if box === nothing
        w = ones(Float64, nregions)
        if !return_info
            return w
        end
        ci = Vector{Tuple{Float64, Float64}}(undef, nregions)
        @inbounds for i in 1:nregions
            ci[i] = (w[i], w[i])
        end
        return (weights=w,
            stderr=zeros(Float64, nregions),
            ci=ci,
            alpha=float(alpha),
            method=:unscaled,
            total_points=NaN,
            nsamples=0,
            counts=nothing)
    end

    a_in, b_in = box

    a = _as_int_tuple(Val(N), a_in)
    b = _as_int_tuple(Val(N), b_in)

    @inbounds for i in 1:pi.n
        a[i] <= b[i] || error("region_weights(Zn): invalid box with b < a")
    end

    # Total lattice points in the inclusive integer box [a, b].
    total_points_big = _box_lattice_size_big(a, b)

    # Slab index bounds per axis: slabs that intersect [a[i], b[i]].
    lo = Vector{Int}(undef, pi.n)
    hi = Vector{Int}(undef, pi.n)
    total_cells_big = BigInt(1)
    @inbounds for i in 1:pi.n
        ci = pi.coords[i]
        lo_i = _slab_index(ci, a[i])
        hi_i = _slab_index(ci, b[i])
        lo[i] = lo_i
        hi[i] = hi_i
        nslabs = hi_i - lo_i + 1
        nslabs >= 1 || error("region_weights(Zn): internal error (empty slab range)")
        total_cells_big *= BigInt(nslabs)
    end

    _too_large(kind::Symbol, total::BigInt, max_allowed::Int) =
        error("region_weights(Zn): box too large for $(kind) (total=$(total) > max=$(max_allowed))")

    # Decide which method to use.
    method_used = method
    if method == :auto
        if total_cells_big <= BigInt(max_cells)
            method_used = :cells
        elseif total_points_big <= BigInt(max_points)
            method_used = :points
        else
            method_used = :sample
        end
    end

    if method_used == :cells
        total_cells_big > BigInt(max_cells) && _too_large(:cells, total_cells_big, max_cells)
        T = _choose_count_type(count_type, total_points_big)
        w = _region_weights_cells(pi, a, b, lo, hi; strict=strict, T=T)
        if !return_info
            return w
        end
        ci = Vector{Tuple{Float64, Float64}}(undef, nregions)
        @inbounds for i in 1:nregions
            x = float(w[i])
            ci[i] = (x, x)
        end
        return (weights=w,
            stderr=zeros(Float64, nregions),
            ci=ci,
            alpha=float(alpha),
            method=:cells,
            total_points=total_points_big,
            total_cells=total_cells_big,
            nsamples=0,
            counts=nothing)

    elseif method_used == :points
        total_points_big > BigInt(max_points) && _too_large(:points, total_points_big, max_points)
        T = _choose_count_type(count_type, total_points_big)
        w = _region_weights_points(pi, a, b; strict=strict, T=T)
        if !return_info
            return w
        end
        ci = Vector{Tuple{Float64, Float64}}(undef, nregions)
        @inbounds for i in 1:nregions
            x = float(w[i])
            ci[i] = (x, x)
        end
        return (weights=w,
            stderr=zeros(Float64, nregions),
            ci=ci,
            alpha=float(alpha),
            method=:points,
            total_points=total_points_big,
            nsamples=0,
            counts=nothing)

    elseif method_used == :sample
        rng2 = rng === nothing ? Random.default_rng() : rng
        w, stderr, counts, total_points_big2 =
            _region_weights_sample(pi, a, b; nsamples=nsamples, rng=rng2, strict=strict)

        if !return_info
            return w
        end

        total_points_f = Float64(total_points_big2)  # may overflow to Inf
        aalpha = float(alpha)
        ci = Vector{Tuple{Float64, Float64}}(undef, nregions)
        @inbounds for i in 1:nregions
            (plo, phi) = _wilson_interval(counts[i], nsamples; alpha=aalpha)
            ci[i] = (total_points_f * plo, total_points_f * phi)
        end

        return (weights=w,
            stderr=stderr,
            ci=ci,
            alpha=aalpha,
            method=:sample,
            total_points=total_points_big2,
            nsamples=Int(nsamples),
            counts=counts)

    else
        error("region_weights(Zn): unknown method=$(method); use :cells, :points, :sample, or :auto")
    end
end


"""
    region_adjacency(pi::ZnEncodingMap; box, strict=true) -> Dict{Tuple{Int,Int},Int}

Compute region adjacency in the lattice case by counting unit (n-1)-faces across
region boundaries inside the integer box `(a,b)`.

The returned dictionary is keyed by unordered region pairs `(r,s)` with `r < s`,
and the value is an integer count of boundary faces.

Implementation notes (speed):
  * uses the slab decomposition induced by `pi.coords`
  * computes a region id per slab-cell (not per lattice point)
  * counts interface faces by scanning neighboring slab-cells
  * specialized fast loops for n=1 and n=2

Best practice:
- this is a box-dependent query, so pass an explicit finite `box`;
- keep `strict=true` unless `0` from [`locate`](@ref) should be tolerated as
  "outside / not represented";
- the returned adjacency counts are most useful on the structured
  `:signature` region poset unless a later dense-poset algorithm truly needs a
  materialized finite-poset realization.
"""
function region_adjacency(pi::ZnEncodingMap; box, strict::Bool=true)
    box === nothing && error("region_adjacency(Zn): provide box")
    a, b = box
    length(a) == pi.n || error("region_adjacency(Zn): box dimension mismatch")
    length(b) == pi.n || error("region_adjacency(Zn): box dimension mismatch")
    nregions = length(pi.sig_y)

    # Use a packed upper-triangular accumulator when region count is moderate.
    # This avoids tuple-hash churn in hot adjacency loops.
    upper_len = div(nregions * (nregions - 1), 2)
    use_packed = (upper_len > 0) && (upper_len * sizeof(Int) <= 32 * 1024 * 1024)
    adj_packed = use_packed ? zeros(Int, upper_len) : Int[]
    adj_dict = use_packed ? Dict{Tuple{Int,Int},Int}() : Dict{Tuple{Int,Int},Int}()

    @inline function _packed_idx(i::Int, j::Int)
        # i < j, 1-based, packed by columns j=2..n
        return div((j - 1) * (j - 2), 2) + i
    end

    @inline function _add_adj!(r::Int, s::Int, w::Int)
        (w <= 0 || r == 0 || s == 0 || r == s) && return
        i, j = (r < s ? (r, s) : (s, r))
        if use_packed
            adj_packed[_packed_idx(i, j)] += w
        else
            adj_dict[(i, j)] = get(adj_dict, (i, j), 0) + w
        end
        return
    end

    function _finalize_adj()
        if !use_packed
            return adj_dict
        end
        out = Dict{Tuple{Int,Int},Int}()
        idx = 1
        @inbounds for j in 2:nregions
            for i in 1:(j - 1)
                v = adj_packed[idx]
                if v != 0
                    out[(i, j)] = v
                end
                idx += 1
            end
        end
        return out
    end

    # helper: count points of slab index s in coords ci within [ai,bi]
    slab_count(ci::AbstractVector{Int}, s::Int, ai::Int, bi::Int) = begin
        lo, hi = if isempty(ci)
            (-typemax(Int), typemax(Int))
        elseif s == 0
            (-typemax(Int), ci[1]-1)
        elseif s == length(ci)
            (ci[end], typemax(Int))
        else
            (ci[s], ci[s+1]-1)
        end
        lo2 = max(lo, ai)
        hi2 = min(hi, bi)
        (hi2 < lo2) ? 0 : (hi2 - lo2 + 1)
    end

    # helper: pick a representative integer in slab s
    slab_rep(ci::AbstractVector{Int}, s::Int) = begin
        if isempty(ci)
            0
        elseif s == 0
            ci[1] - 1
        elseif s == length(ci)
            ci[end]
        else
            ci[s]
        end
    end

    # number of slabs per axis
    lens = Vector{Int}(undef, pi.n)
    for i in 1:pi.n
        lens[i] = length(pi.coords[i]) + 1
    end

    # precompute slab counts per axis
    counts = Vector{Vector{Int}}(undef, pi.n)
    for i in 1:pi.n
        ci = pi.coords[i]
        li = lens[i]
        cnt = Vector{Int}(undef, li)
        for s in 0:(li-1)
            cnt[s+1] = slab_count(ci, s, a[i], b[i])
        end
        counts[i] = cnt
    end

    # region id per slab-cell
    if pi.n == 1
        L1 = lens[1]
        reg = Vector{Int}(undef, L1)
        ci1 = pi.coords[1]
        for s1 in 0:(L1-1)
            g1 = slab_rep(ci1, s1)
            r = locate(pi, (g1,))
            if r == 0 && strict
                error("region_adjacency(Zn): unknown signature at representative point")
            end
            reg[s1+1] = r
        end

        for i1 in 1:(L1-1)
            r = reg[i1]
            s = reg[i1+1]
            if r != 0 && s != 0 && r != s
                if counts[1][i1] > 0 && counts[1][i1+1] > 0
                    _add_adj!(r, s, 1)
                end
            end
        end
        return _finalize_adj()
    end

    if pi.n == 2
        L1, L2 = lens[1], lens[2]
        ci1, ci2 = pi.coords[1], pi.coords[2]
        reg = Array{Int}(undef, L1, L2)

        for s1 in 0:(L1-1), s2 in 0:(L2-1)
            g1 = slab_rep(ci1, s1)
            g2 = slab_rep(ci2, s2)
            r = locate(pi, (g1,g2))
            if r == 0 && strict
                error("region_adjacency(Zn): unknown signature at representative point")
            end
            reg[s1+1, s2+1] = r
        end

        # scan neighbors along axis 1
        for i1 in 1:(L1-1), i2 in 1:L2
            r = reg[i1, i2]
            s = reg[i1+1, i2]
            if r != 0 && s != 0 && r != s
                if counts[1][i1] > 0 && counts[1][i1+1] > 0
                    cross = counts[2][i2]
                    if cross > 0
                        _add_adj!(r, s, cross)
                    end
                end
            end
        end

        # scan neighbors along axis 2
        for i1 in 1:L1, i2 in 1:(L2-1)
            r = reg[i1, i2]
            s = reg[i1, i2+1]
            if r != 0 && s != 0 && r != s
                if counts[2][i2] > 0 && counts[2][i2+1] > 0
                    cross = counts[1][i1]
                    if cross > 0
                        _add_adj!(r, s, cross)
                    end
                end
            end
        end

        return _finalize_adj()
    end

    # generic n >= 3 (still slab-based; not per lattice point)
    shape = ntuple(i -> lens[i], pi.n)
    reg = Array{Int}(undef, shape)

    for I in CartesianIndices(reg)
        g = ntuple(k -> slab_rep(pi.coords[k], I[k]-1), pi.n)
        r = locate(pi, g)
        if r == 0 && strict
            error("region_adjacency(Zn): unknown signature at representative point")
        end
        reg[I] = r
    end

    steps = [CartesianIndex(ntuple(i -> (i==k ? 1 : 0), pi.n)) for k in 1:pi.n]

    for k in 1:pi.n
        step = steps[k]
        for I in CartesianIndices(reg)
            if I[k] == shape[k]
                continue
            end
            J = I + step
            r = reg[I]
            s = reg[J]
            if r == 0 || s == 0 || r == s
                continue
            end
            if counts[k][I[k]] == 0 || counts[k][I[k]+1] == 0
                continue
            end
            cross = 1
            for t in 1:pi.n
                if t == k
                    continue
                end
                ct = counts[t][I[t]]
                if ct == 0
                    cross = 0
                    break
                end
                cross *= ct
            end
            if cross > 0
                _add_adj!(r, s, cross)
            end
        end
    end

    return _finalize_adj()
end


"""
    encode_from_flange(FG::Flange{K}, opts::EncodingOptions; poset_kind=:signature) -> (P, H, pi)

Encode a single Z^n flange presentation `FG` to a finite encoding poset `P` and a
finite-poset fringe module `H` on `P`, together with the classifier `pi : Z^n -> P`
(as a `ZnEncodingMap`).

`opts` is required.
- `opts.backend` must be `:auto` or `:zn`.
- `opts.max_regions` caps the number of distinct regions/signatures (default: 200_000).

Best practice:
- use `poset_kind=:signature` as the cheap/default path;
- move to `poset_kind=:dense` only when a downstream algorithm really requires a
  dense finite poset;
- inspect the returned classifier with [`zn_encoding_summary`](@ref) or
  `describe(pi)` before touching heavier downstream objects.
"""
function encode_from_flange(FG::Flange{K}, opts::EncodingOptions;
                            poset_kind::Symbol = :signature) where {K}
    if opts.backend != :auto && opts.backend != :zn
        error("encode_from_flange: EncodingOptions.backend must be :auto or :zn")
    end
    P, Hs, pi = encode_from_flanges((FG,), opts; poset_kind = poset_kind)
    return P, Hs[1], pi
end

encode_from_flange(FG::Flange{K};
                   opts::EncodingOptions=EncodingOptions(),
                   poset_kind::Symbol = :signature) where {K} =
    encode_from_flange(FG, opts; poset_kind = poset_kind)

function encode_from_flange(
    P::AbstractPoset,
    FG::Flange{K},
    opts::EncodingOptions;
    check_poset::Bool = true,
    poset_kind::Symbol = :signature,
) where {K}
    if opts.backend != :auto && opts.backend != :zn
        error("encode_from_flange: EncodingOptions.backend must be :auto or :zn")
    end
    P2, Hs, pi = encode_from_flanges(P, (FG,), opts;
                                     check_poset = check_poset, poset_kind = poset_kind)
    return P2, Hs[1], pi
end

function encode_from_flange(
    P::AbstractPoset,
    FG::Flange{K};
    opts::EncodingOptions=EncodingOptions(),
    check_poset::Bool = true,
    poset_kind::Symbol = :signature,
) where {K}
    return encode_from_flange(P, FG, opts;
                              check_poset = check_poset, poset_kind = poset_kind)
end

"""
    encode_from_flanges(FGs, opts::EncodingOptions; poset_kind=:signature) -> (P, Hs, pi)

Common-encode several Z^n flange presentations to a single finite encoding poset `P`,
and return the pushed-down fringe modules `Hs` on `P`.

Arguments
- `FGs`: a vector (or tuple) of `Flange{K}`.
- `opts`: an `EncodingOptions` (required).
  `opts.backend` must be `:auto` or `:zn`.
  `opts.max_regions` caps the number of distinct regions/signatures (default: 200_000).

Returns
- `P`  : the common finite encoding poset
- `Hs` : a vector of `FiniteFringe.FringeModule{K}`, one per input flange
- `pi` : classifier `pi : Z^n -> P` (as `ZnEncodingMap`)

Best practice:
- keep `poset_kind=:signature` unless a downstream algorithm explicitly needs a
  dense finite poset;
- use [`compile_zn_cache`](@ref) when the returned classifier will serve many
  repeated query or region-geometry calls;
- inspect with [`zn_encoding_summary`](@ref) or `describe(pi)` before
  materializing other owner data.
"""
function encode_from_flanges(FGs::Union{AbstractVector{<:Flange{K}}, Tuple{Vararg{Flange{K}}}},
                             opts::EncodingOptions;
                             poset_kind::Symbol = :signature) where {K}
    if opts.backend != :auto && opts.backend != :zn
        error("encode_from_flanges: EncodingOptions.backend must be :auto or :zn")
    end
    P, pi = encode_poset_from_flanges(FGs, opts; poset_kind = poset_kind)

    Hs = Vector{FringeModule{K}}(undef, length(FGs))
    for k in 1:length(FGs)
        Hs[k] = _pushforward_flange_to_fringe(P, pi, FGs[k]; strict=true)
    end
    return P, Hs, pi
end

encode_from_flanges(FGs::Union{AbstractVector{<:Flange{K}}, Tuple{Vararg{Flange{K}}}};
                    opts::EncodingOptions=EncodingOptions(),
                    poset_kind::Symbol = :signature) where {K} =
    encode_from_flanges(FGs, opts; poset_kind = poset_kind)

"""
    encode_from_flanges(P, FGs, opts::EncodingOptions; check_poset=true, poset_kind=:signature) -> (P, Hs, pi)

Use a user-provided poset `P` (possibly structured) as the encoding poset.
We still build the encoding map `pi` from the flanges; `check_poset=true`
verifies that `P` has the same order as the internally constructed poset.

Use this variant when you already own the target finite poset and want to keep
that object stable across multiple pushes. Keep `check_poset=true` unless `P`
is trusted internal data and you are benchmarking the no-check path.
"""
function encode_from_flanges(
    P::AbstractPoset,
    FGs::Union{AbstractVector{<:Flange{K}}, Tuple{Vararg{Flange{K}}}},
    opts::EncodingOptions;
    check_poset::Bool = true,
    poset_kind::Symbol = :signature,
) where {K}
    if opts.backend != :auto && opts.backend != :zn
        error("encode_from_flanges: EncodingOptions.backend must be :auto or :zn")
    end
    P0, pi = encode_poset_from_flanges(FGs, opts; poset_kind = poset_kind)
    if check_poset
        nvertices(P) == nvertices(P0) || error("encode_from_flanges: provided P has wrong size")
        poset_equal(P, P0) || error("encode_from_flanges: provided P is not equal to the encoding poset")
    end

    Hs = Vector{FringeModule{K}}(undef, length(FGs))
    for k in 1:length(FGs)
        Hs[k] = _pushforward_flange_to_fringe(P, pi, FGs[k]; strict=true)
    end
    return P, Hs, pi
end

function encode_from_flanges(
    P::AbstractPoset,
    FGs::Union{AbstractVector{<:Flange{K}}, Tuple{Vararg{Flange{K}}}};
    opts::EncodingOptions=EncodingOptions(),
    check_poset::Bool = true,
    poset_kind::Symbol = :signature,
) where {K}
    return encode_from_flanges(P, FGs, opts;
                               check_poset = check_poset, poset_kind = poset_kind)
end

# Small-arity overloads (avoid "varargs then opts" signatures).
function encode_from_flanges(FG1::Flange{K}, FG2::Flange{K}, opts::EncodingOptions;
                             poset_kind::Symbol = :signature) where {K}
    return encode_from_flanges((FG1, FG2), opts; poset_kind = poset_kind)
end

encode_from_flanges(FG1::Flange{K}, FG2::Flange{K};
                    opts::EncodingOptions=EncodingOptions(),
                    poset_kind::Symbol = :signature) where {K} =
    encode_from_flanges((FG1, FG2), opts; poset_kind = poset_kind)

function encode_from_flanges(FG1::Flange{K}, FG2::Flange{K}, FG3::Flange{K}, opts::EncodingOptions;
                             poset_kind::Symbol = :signature) where {K}
    return encode_from_flanges((FG1, FG2, FG3), opts; poset_kind = poset_kind)
end

encode_from_flanges(FG1::Flange{K}, FG2::Flange{K}, FG3::Flange{K};
                    opts::EncodingOptions=EncodingOptions(),
                    poset_kind::Symbol = :signature) where {K} =
    encode_from_flanges((FG1, FG2, FG3), opts; poset_kind = poset_kind)

function encode_from_flanges(
    P::AbstractPoset,
    FG1::Flange{K},
    FG2::Flange{K},
    opts::EncodingOptions;
    check_poset::Bool = true,
    poset_kind::Symbol = :signature,
) where {K}
    return encode_from_flanges(P, (FG1, FG2), opts;
                               check_poset = check_poset, poset_kind = poset_kind)
end

function encode_from_flanges(
    P::AbstractPoset,
    FG1::Flange{K},
    FG2::Flange{K};
    opts::EncodingOptions=EncodingOptions(),
    check_poset::Bool = true,
    poset_kind::Symbol = :signature,
) where {K}
    return encode_from_flanges(P, (FG1, FG2), opts;
                               check_poset = check_poset, poset_kind = poset_kind)
end

function encode_from_flanges(
    P::AbstractPoset,
    FG1::Flange{K},
    FG2::Flange{K},
    FG3::Flange{K},
    opts::EncodingOptions;
    check_poset::Bool = true,
    poset_kind::Symbol = :signature,
) where {K}
    return encode_from_flanges(P, (FG1, FG2, FG3), opts;
                               check_poset = check_poset, poset_kind = poset_kind)
end

function encode_from_flanges(
    P::AbstractPoset,
    FG1::Flange{K},
    FG2::Flange{K},
    FG3::Flange{K};
    opts::EncodingOptions=EncodingOptions(),
    check_poset::Bool = true,
    poset_kind::Symbol = :signature,
) where {K}
    return encode_from_flanges(P, (FG1, FG2, FG3), opts;
                               check_poset = check_poset, poset_kind = poset_kind)
end


# -----------------------------------------------------------------------------
# CompiledEncoding forwarding (treat compiled encodings as primary)
# -----------------------------------------------------------------------------

@inline _unwrap_encoding(pi::CompiledEncoding) = pi.pi

region_weights(pi::CompiledEncoding{<:ZnEncodingMap}; kwargs...) =
    region_weights(_unwrap_encoding(pi); kwargs...)
region_adjacency(pi::CompiledEncoding{<:ZnEncodingMap}; kwargs...) =
    region_adjacency(_unwrap_encoding(pi); kwargs...)
locate_many!(dest::AbstractVector{<:Integer}, pi::CompiledEncoding{<:ZnEncodingMap}, X::AbstractMatrix{<:Integer}; kwargs...) =
    locate_many!(dest, _unwrap_encoding(pi), X; kwargs...)
locate_many!(dest::AbstractVector{<:Integer}, pi::CompiledEncoding{<:ZnEncodingMap}, X::AbstractMatrix{<:AbstractFloat}; kwargs...) =
    locate_many!(dest, _unwrap_encoding(pi), X; kwargs...)
locate_many(pi::CompiledEncoding{<:ZnEncodingMap}, X::AbstractMatrix{<:Integer}; kwargs...) =
    locate_many(_unwrap_encoding(pi), X; kwargs...)
locate_many(pi::CompiledEncoding{<:ZnEncodingMap}, X::AbstractMatrix{<:AbstractFloat}; kwargs...) =
    locate_many(_unwrap_encoding(pi), X; kwargs...)


end
