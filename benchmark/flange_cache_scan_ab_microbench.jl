#!/usr/bin/env julia
# benchmark/flange_cache_scan_ab_microbench.jl
#
# A/B microbenchmark for the two FlangeZn hot-path changes implemented in this
# branch:
#   1. typed open-addressed cache table vs the historical Dict-of-buckets cache
#   2. constrained-dimension active-set scans vs full-dimension/free-mask scans
#
# This benchmark is intentionally self-contained so it can compare before/after
# kernels directly without depending on a checked-out historical source tree.

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)

using Random

struct BenchKernel{N}
    ncoord_words::Int
    nflat::Int
    ninj::Int
    flat_b::Matrix{Int}
    inj_b::Matrix{Int}
    flat_free_words::Matrix{UInt64}
    inj_free_words::Matrix{UInt64}
    flat_offsets::Vector{Int}
    flat_dims::Vector{Int}
    flat_bounds::Vector{Int}
    inj_offsets::Vector{Int}
    inj_dims::Vector{Int}
    inj_bounds::Vector{Int}
    coord_word::Vector{Int}
    coord_mask::Vector{UInt64}
end

mutable struct OldEntry
    row_words::Vector{UInt64}
    col_words::Vector{UInt64}
    value::Int
end

mutable struct NewTable
    slots::Vector{Int}
    hashes::Vector{UInt64}
    values::Vector{Int}
    row_words::Matrix{UInt64}
    col_words::Matrix{UInt64}
    nentries::Int
end

@inline _bit_word(i::Int) = ((i - 1) >>> 6) + 1
@inline _bit_mask(i::Int) = UInt64(1) << ((i - 1) & 63)
@inline function _next_pow2_at_least(n::Int)
    cap = 1
    while cap < n
        cap <<= 1
    end
    return cap
end
@inline _cache_slot_capacity(entry_cap::Int) = max(8, _next_pow2_at_least(max(2 * entry_cap, 8)))

function NewTable(nrow_words::Int, ncol_words::Int, entry_cap::Int)
    slot_cap = _cache_slot_capacity(entry_cap)
    return NewTable(zeros(Int, slot_cap),
                    Vector{UInt64}(undef, entry_cap),
                    Vector{Int}(undef, entry_cap),
                    Matrix{UInt64}(undef, nrow_words, entry_cap),
                    Matrix{UInt64}(undef, ncol_words, entry_cap),
                    0)
end

@inline _cache_slot_index(key::UInt64, nslots::Int) = Int(key & UInt64(nslots - 1)) + 1
@inline _next_slot(slot::Int, nslots::Int) = slot == nslots ? 1 : slot + 1

@inline function _same_words(a::Vector{UInt64}, b::Vector{UInt64})::Bool
    length(a) == length(b) || return false
    @inbounds for i in eachindex(a, b)
        a[i] == b[i] || return false
    end
    return true
end

@inline function _same_words_stored(store::Matrix{UInt64}, idx::Int, words::Vector{UInt64})::Bool
    @inbounds for i in eachindex(words)
        store[i, idx] == words[i] || return false
    end
    return true
end

@inline function _copy_words_to_store!(store::Matrix{UInt64}, idx::Int, words::Vector{UInt64})
    @inbounds for i in eachindex(words)
        store[i, idx] = words[i]
    end
    return nothing
end

function _rehash_cache_slots!(table::NewTable, nslots::Int)
    slots = zeros(Int, nslots)
    @inbounds for idx in 1:table.nentries
        key = table.hashes[idx]
        slot = _cache_slot_index(key, nslots)
        while slots[slot] != 0
            slot = _next_slot(slot, nslots)
        end
        slots[slot] = idx
    end
    table.slots = slots
    return table
end

function _grow_cache_table!(table::NewTable, new_entry_cap::Int)
    old_cap = length(table.values)
    new_entry_cap <= old_cap && return table
    nrow_words = size(table.row_words, 1)
    ncol_words = size(table.col_words, 1)
    new_hashes = Vector{UInt64}(undef, new_entry_cap)
    new_values = Vector{Int}(undef, new_entry_cap)
    new_row_words = Matrix{UInt64}(undef, nrow_words, new_entry_cap)
    new_col_words = Matrix{UInt64}(undef, ncol_words, new_entry_cap)
    @inbounds for idx in 1:table.nentries
        new_hashes[idx] = table.hashes[idx]
        new_values[idx] = table.values[idx]
    end
    if nrow_words != 0 && table.nentries != 0
        @views new_row_words[:, 1:table.nentries] .= table.row_words[:, 1:table.nentries]
    end
    if ncol_words != 0 && table.nentries != 0
        @views new_col_words[:, 1:table.nentries] .= table.col_words[:, 1:table.nentries]
    end
    table.hashes = new_hashes
    table.values = new_values
    table.row_words = new_row_words
    table.col_words = new_col_words
    _rehash_cache_slots!(table, _cache_slot_capacity(new_entry_cap))
    return table
end

@inline function _mark_active!(words::Vector{UInt64}, idx::Int)
    words[_bit_word(idx)] |= _bit_mask(idx)
    return nothing
end

function make_kernel(; N::Int=3, nflats::Int=48, ninj::Int=48, seed::UInt64=UInt64(0xF1A94E))
    rng = Random.MersenneTwister(seed)
    ncoord_words = cld(N, 64)
    flat_b = Matrix{Int}(undef, N, nflats)
    inj_b = Matrix{Int}(undef, N, ninj)
    flat_free_words = Matrix{UInt64}(undef, ncoord_words, nflats)
    inj_free_words = Matrix{UInt64}(undef, ncoord_words, ninj)
    fill!(flat_free_words, 0)
    fill!(inj_free_words, 0)
    flat_offsets = Vector{Int}(undef, nflats + 1)
    inj_offsets = Vector{Int}(undef, ninj + 1)
    flat_dims = Int[]
    flat_bounds = Int[]
    inj_dims = Int[]
    inj_bounds = Int[]
    coord_word = [_bit_word(d) for d in 1:N]
    coord_mask = [_bit_mask(d) for d in 1:N]

    for j in 1:nflats
        flat_offsets[j] = length(flat_dims) + 1
        for d in 1:N
            free = rand(rng) < 0.45
            b = rand(rng, -8:12)
            flat_b[d, j] = b
            if free
                flat_free_words[coord_word[d], j] |= coord_mask[d]
            else
                push!(flat_dims, d)
                push!(flat_bounds, b)
            end
        end
    end
    flat_offsets[nflats + 1] = length(flat_dims) + 1

    for i in 1:ninj
        inj_offsets[i] = length(inj_dims) + 1
        for d in 1:N
            free = rand(rng) < 0.45
            b = rand(rng, -8:12)
            inj_b[d, i] = b
            if free
                inj_free_words[coord_word[d], i] |= coord_mask[d]
            else
                push!(inj_dims, d)
                push!(inj_bounds, b)
            end
        end
    end
    inj_offsets[ninj + 1] = length(inj_dims) + 1

    return BenchKernel{N}(ncoord_words, nflats, ninj, flat_b, inj_b, flat_free_words, inj_free_words,
                          flat_offsets, flat_dims, flat_bounds, inj_offsets, inj_dims, inj_bounds,
                          coord_word, coord_mask)
end

@inline function contains_flat_old(k::BenchKernel{N}, j::Int, g::NTuple{N,Int}) where {N}
    @inbounds for d in 1:N
        wd = k.coord_word[d]
        bm = k.coord_mask[d]
        ((k.flat_free_words[wd, j] & bm) != 0) && continue
        g[d] >= k.flat_b[d, j] || return false
    end
    return true
end

@inline function contains_inj_old(k::BenchKernel{N}, i::Int, g::NTuple{N,Int}) where {N}
    @inbounds for d in 1:N
        wd = k.coord_word[d]
        bm = k.coord_mask[d]
        ((k.inj_free_words[wd, i] & bm) != 0) && continue
        g[d] <= k.inj_b[d, i] || return false
    end
    return true
end

@inline function contains_flat_new(k::BenchKernel{N}, j::Int, g::NTuple{N,Int}) where {N}
    @inbounds for ptr in k.flat_offsets[j]:(k.flat_offsets[j + 1] - 1)
        d = k.flat_dims[ptr]
        g[d] >= k.flat_bounds[ptr] || return false
    end
    return true
end

@inline function contains_inj_new(k::BenchKernel{N}, i::Int, g::NTuple{N,Int}) where {N}
    @inbounds for ptr in k.inj_offsets[i]:(k.inj_offsets[i + 1] - 1)
        d = k.inj_dims[ptr]
        g[d] <= k.inj_bounds[ptr] || return false
    end
    return true
end

function fill_active_old!(rows::Vector{Int}, cols::Vector{Int}, row_words::Vector{UInt64}, col_words::Vector{UInt64}, k::BenchKernel{N}, g::NTuple{N,Int}) where {N}
    empty!(rows); empty!(cols)
    fill!(row_words, 0); fill!(col_words, 0)
    @inbounds for i in 1:k.ninj
        if contains_inj_old(k, i, g)
            push!(rows, i)
            _mark_active!(row_words, i)
        end
    end
    @inbounds for j in 1:k.nflat
        if contains_flat_old(k, j, g)
            push!(cols, j)
            _mark_active!(col_words, j)
        end
    end
    return length(rows) + length(cols)
end

function fill_active_new_words!(row_words::Vector{UInt64}, col_words::Vector{UInt64}, k::BenchKernel{N}, g::NTuple{N,Int}) where {N}
    fill!(row_words, 0); fill!(col_words, 0)
    nr = 0
    nc = 0
    @inbounds for i in 1:k.ninj
        if contains_inj_new(k, i, g)
            _mark_active!(row_words, i)
            nr += 1
        end
    end
    @inbounds for j in 1:k.nflat
        if contains_flat_new(k, j, g)
            _mark_active!(col_words, j)
            nc += 1
        end
    end
    return nr + nc
end

@inline function active_words_hash(row_words::Vector{UInt64}, col_words::Vector{UInt64})
    h = UInt64(0x243f6a8885a308d3)
    @inbounds for w in row_words
        h = xor(h, w + UInt64(0x9e3779b97f4a7c15) + (h << 6) + (h >>> 2))
    end
    @inbounds for w in col_words
        h = xor(h, w + UInt64(0x9e3779b97f4a7c15) + (h << 6) + (h >>> 2))
    end
    return h
end

function old_lookup!(table::Dict{UInt64, Vector{OldEntry}}, key::UInt64, row_words::Vector{UInt64}, col_words::Vector{UInt64})
    bucket = get(table, key, nothing)
    bucket === nothing && return -1
    @inbounds for ent in bucket
        if _same_words(ent.row_words, row_words) && _same_words(ent.col_words, col_words)
            return ent.value
        end
    end
    return -1
end

function old_store!(table::Dict{UInt64, Vector{OldEntry}}, key::UInt64, row_words::Vector{UInt64}, col_words::Vector{UInt64}, value::Int)
    bucket = get!(table, key, OldEntry[])
    push!(bucket, OldEntry(copy(row_words), copy(col_words), value))
    return value
end

function new_lookup!(table::NewTable, key::UInt64, row_words::Vector{UInt64}, col_words::Vector{UInt64})
    table.nentries == 0 && return -1
    nslots = length(table.slots)
    slot = _cache_slot_index(key, nslots)
    while true
        idx = @inbounds table.slots[slot]
        idx == 0 && return -1
        if @inbounds(table.hashes[idx] == key) &&
           _same_words_stored(table.row_words, idx, row_words) &&
           _same_words_stored(table.col_words, idx, col_words)
            return @inbounds table.values[idx]
        end
        slot = _next_slot(slot, nslots)
    end
end

function new_store!(table::NewTable, key::UInt64, row_words::Vector{UInt64}, col_words::Vector{UInt64}, value::Int)
    entry_cap = length(table.values)
    if table.nentries >= entry_cap || (table.nentries + 1) * 10 >= length(table.slots) * 7
        _grow_cache_table!(table, max(entry_cap << 1, table.nentries + 1))
    end
    nslots = length(table.slots)
    slot = _cache_slot_index(key, nslots)
    while @inbounds(table.slots[slot]) != 0
        slot = _next_slot(slot, nslots)
    end
    idx = table.nentries + 1
    @inbounds table.hashes[idx] = key
    @inbounds table.values[idx] = value
    _copy_words_to_store!(table.row_words, idx, row_words)
    _copy_words_to_store!(table.col_words, idx, col_words)
    @inbounds table.slots[slot] = idx
    table.nentries = idx
    return value
end

function bench(name, f; reps=7)
    GC.gc(); f(); GC.gc()
    times = Float64[]
    allocs = Float64[]
    for _ in 1:reps
        m = @timed f()
        push!(times, 1000m.time)
        push!(allocs, m.bytes / 1024)
    end
    sort!(times); sort!(allocs)
    t = times[cld(reps,2)]
    a = allocs[cld(reps,2)]
    println(rpad(name, 34), " median_time=", round(t, digits=3), " ms  median_alloc=", round(a, digits=1), " KiB")
    return (ms=t, kib=a)
end

function main(; reps::Int=7, nqueries::Int=6000, N::Int=3, nflats::Int=48, ninj::Int=48)
    k = make_kernel(; N=N, nflats=nflats, ninj=ninj)
    rng = Random.MersenneTwister(0xBEE5)
    points = [ntuple(_ -> rand(rng, -10:14), N) for _ in 1:nqueries]

    rows = Int[]
    cols = Int[]
    sizehint!(rows, ninj)
    sizehint!(cols, nflats)
    row_words = zeros(UInt64, cld(ninj, 64))
    col_words = zeros(UInt64, cld(nflats, 64))

    # Build a realistic lookup workload from the active-set generator itself.
    keys = Vector{UInt64}(undef, nqueries)
    row_store = [zeros(UInt64, length(row_words)) for _ in 1:nqueries]
    col_store = [zeros(UInt64, length(col_words)) for _ in 1:nqueries]
    vals = Vector{Int}(undef, nqueries)
    for (idx, p) in enumerate(points)
        vals[idx] = fill_active_new_words!(row_words, col_words, k, p)
        row_store[idx] .= row_words
        col_store[idx] .= col_words
        keys[idx] = active_words_hash(row_words, col_words)
    end

    old_table = Dict{UInt64, Vector{OldEntry}}()
    new_table = NewTable(length(row_words), length(col_words), min(nqueries, 64))
    for i in 1:nqueries
        if old_lookup!(old_table, keys[i], row_store[i], col_store[i]) < 0
            old_store!(old_table, keys[i], row_store[i], col_store[i], vals[i])
        end
        if new_lookup!(new_table, keys[i], row_store[i], col_store[i]) < 0
            new_store!(new_table, keys[i], row_store[i], col_store[i], vals[i])
        end
    end

    println("Flange cache/scan A-B microbenchmark")
    println("reps=$(reps), nqueries=$(nqueries), N=$(N), nflats=$(nflats), ninj=$(ninj)\n")

    b_scan_old = bench("scan fill (old)", () -> begin
        s = 0
        for p in points
            s += fill_active_old!(rows, cols, row_words, col_words, k, p)
        end
        s
    end; reps=reps)

    b_scan_new = bench("scan fill (new words)", () -> begin
        s = 0
        for p in points
            s += fill_active_new_words!(row_words, col_words, k, p)
        end
        s
    end; reps=reps)

    b_cache_old = bench("cache warm lookup (old)", () -> begin
        s = 0
        for i in 1:nqueries
            s += old_lookup!(old_table, keys[i], row_store[i], col_store[i])
        end
        s
    end; reps=reps)

    b_cache_new = bench("cache warm lookup (new)", () -> begin
        s = 0
        for i in 1:nqueries
            s += new_lookup!(new_table, keys[i], row_store[i], col_store[i])
        end
        s
    end; reps=reps)

    b_insert_old = bench("cache build+lookup (old)", () -> begin
        tbl = Dict{UInt64, Vector{OldEntry}}()
        s = 0
        for i in 1:nqueries
            v = old_lookup!(tbl, keys[i], row_store[i], col_store[i])
            if v < 0
                v = old_store!(tbl, keys[i], row_store[i], col_store[i], vals[i])
            end
            s += v
        end
        s
    end; reps=reps)

    b_insert_new = bench("cache build+lookup (new)", () -> begin
        tbl = NewTable(length(row_words), length(col_words), 64)
        s = 0
        for i in 1:nqueries
            v = new_lookup!(tbl, keys[i], row_store[i], col_store[i])
            if v < 0
                v = new_store!(tbl, keys[i], row_store[i], col_store[i], vals[i])
            end
            s += v
        end
        s
    end; reps=reps)

    println("\nscan speedup (new/old): ", round(b_scan_old.ms / max(b_scan_new.ms, 1e-9), digits=3), "x")
    println("warm-cache speedup (new/old): ", round(b_cache_old.ms / max(b_cache_new.ms, 1e-9), digits=3), "x")
    println("build+lookup speedup (new/old): ", round(b_insert_old.ms / max(b_insert_new.ms, 1e-9), digits=3), "x")
end

if abspath(PROGRAM_FILE) == @__FILE__
    local reps = 7
    local nqueries = 6000
    for a in ARGS
        startswith(a, "--reps=") && (reps = parse(Int, split(a, "=", limit=2)[2]))
        startswith(a, "--nqueries=") && (nqueries = parse(Int, split(a, "=", limit=2)[2]))
    end
    main(; reps=reps, nqueries=nqueries)
end
