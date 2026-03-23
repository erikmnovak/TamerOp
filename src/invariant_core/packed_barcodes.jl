# -----------------------------------------------------------------------------
# packed_barcodes.jl
#
# Shared packed barcode representations and conversion helpers used by
# `SliceInvariants` and `Fibered2D`.
# -----------------------------------------------------------------------------

const IndexBarcode = Dict{Tuple{Int,Int},Int}
const FloatBarcode = Dict{Tuple{Float64,Float64},Int}

@inline _empty_index_barcode() = IndexBarcode()
@inline _empty_float_barcode() = FloatBarcode()

struct EndpointPair{T<:Real}
    b::T
    d::T
end

"""
    PackedBarcode{T<:Real}

Packed internal barcode representation used in hot loops.

- `pairs[i]` stores one endpoint pair,
- `mults[i]` stores its multiplicity.

This type is intentionally backend-facing. Prefer owner-level slice or fibered
summary helpers for ordinary workflows, and use `describe(...)` here only when
you are debugging packed-barcode kernels directly.
"""
struct PackedBarcode{T<:Real}
    pairs::Vector{EndpointPair{T}}
    mults::Vector{Int}
end

const PackedIndexBarcode = PackedBarcode{Int}
const PackedFloatBarcode = PackedBarcode{Float64}

@inline _empty_packed_index_barcode() = PackedIndexBarcode(EndpointPair{Int}[], Int[])
@inline _empty_packed_float_barcode() = PackedFloatBarcode(EndpointPair{Float64}[], Int[])

"""
    PackedBarcodeGrid{B<:PackedBarcode}

Packed barcode grid used internally for slice and fibered pipelines.

The grid stores a flat barcode buffer with deterministic matrix indexing, so
high-throughput kernels can work with a matrix-like object without paying for a
matrix-of-dictionaries representation.
"""
struct PackedBarcodeGrid{B<:PackedBarcode} <: AbstractMatrix{B}
    flat::Vector{B}
    nd::Int
    no::Int
    function PackedBarcodeGrid{B}(flat::Vector{B}, nd::Int, no::Int) where {B<:PackedBarcode}
        length(flat) == nd * no || error("PackedBarcodeGrid: flat length mismatch")
        return new{B}(flat, nd, no)
    end
end

@inline PackedBarcodeGrid{B}(::UndefInitializer, nd::Int, no::Int) where {B<:PackedBarcode} =
    PackedBarcodeGrid{B}(Vector{B}(undef, nd * no), nd, no)

@inline Base.size(g::PackedBarcodeGrid) = (g.nd, g.no)
@inline Base.length(g::PackedBarcodeGrid) = length(g.flat)
@inline Base.axes(g::PackedBarcodeGrid) = (Base.OneTo(g.nd), Base.OneTo(g.no))
@inline Base.IndexStyle(::Type{<:PackedBarcodeGrid}) = IndexLinear()
@inline Base.getindex(g::PackedBarcodeGrid{B}, i::Int, j::Int) where {B<:PackedBarcode} = g.flat[(j - 1) * g.nd + i]
@inline Base.getindex(g::PackedBarcodeGrid{B}, k::Int) where {B<:PackedBarcode} = g.flat[k]
@inline function Base.setindex!(g::PackedBarcodeGrid{B}, v::B, i::Int, j::Int) where {B<:PackedBarcode}
    g.flat[(j - 1) * g.nd + i] = v
    return g
end
@inline function Base.setindex!(g::PackedBarcodeGrid{B}, v::B, k::Int) where {B<:PackedBarcode}
    g.flat[k] = v
    return g
end
@inline Base.vec(g::PackedBarcodeGrid) = g.flat

@inline _packed_grid_undef(::Type{B}, nd::Int, no::Int) where {B<:PackedBarcode} =
    PackedBarcodeGrid{B}(undef, nd, no)

@inline _packed_grid_from_matrix(M::Matrix{B}) where {B<:PackedBarcode} =
    PackedBarcodeGrid{B}(vec(M), size(M, 1), size(M, 2))

Base.length(pb::PackedBarcode) = length(pb.pairs)
Base.isempty(pb::PackedBarcode) = isempty(pb.pairs)

@inline npairs(pb::PackedBarcode) = length(pb.pairs)
@inline total_multiplicity(pb::PackedBarcode) = _packed_total_multiplicity(pb)
@inline grid_size(g::PackedBarcodeGrid) = size(g)

@inline function _packed_barcode_describe(pb::PackedBarcode{T}) where {T<:Real}
    return (;
        kind=:packed_barcode,
        endpoint_type=T,
        npairs=npairs(pb),
        total_multiplicity=total_multiplicity(pb),
        empty=isempty(pb),
    )
end

@inline function _packed_grid_describe(g::PackedBarcodeGrid{B}) where {B<:PackedBarcode}
    return (;
        kind=:packed_barcode_grid,
        barcode_type=B,
        grid_size=grid_size(g),
        ncells=length(g),
    )
end

"""
    describe(pb::PackedBarcode) -> NamedTuple
    describe(g::PackedBarcodeGrid) -> NamedTuple

Return a compact debugging-oriented summary of a packed barcode object.
"""
describe(pb::PackedBarcode) = _packed_barcode_describe(pb)
describe(g::PackedBarcodeGrid) = _packed_grid_describe(g)

function show(io::IO, pb::PackedBarcode)
    d = describe(pb)
    print(io, "PackedBarcode(npairs=", d.npairs,
          ", multiplicity=", d.total_multiplicity,
          ", endpoint_type=", d.endpoint_type, ")")
end

function show(io::IO, ::MIME"text/plain", pb::PackedBarcode)
    d = describe(pb)
    println(io, "PackedBarcode")
    println(io, "  endpoint_type: ", d.endpoint_type)
    println(io, "  npairs: ", d.npairs)
    println(io, "  total_multiplicity: ", d.total_multiplicity)
    println(io, "  empty: ", d.empty)
end

function show(io::IO, g::PackedBarcodeGrid)
    d = describe(g)
    print(io, "PackedBarcodeGrid(size=", d.grid_size,
          ", barcode_type=", d.barcode_type, ")")
end

function show(io::IO, ::MIME"text/plain", g::PackedBarcodeGrid)
    d = describe(g)
    println(io, "PackedBarcodeGrid")
    println(io, "  size: ", d.grid_size)
    println(io, "  barcode_type: ", d.barcode_type)
    println(io, "  ncells: ", d.ncells)
end

@inline function Base.iterate(pb::PackedBarcode{T}, state::Int=1) where {T}
    state > length(pb.pairs) && return nothing
    p = pb.pairs[state]
    return (((p.b, p.d), pb.mults[state]), state + 1)
end

@inline function _packed_total_multiplicity(pb::PackedBarcode)
    s = 0
    @inbounds for m in pb.mults
        s += m
    end
    return s
end

@inline function _to_float_barcode(bc)
    bc isa FloatBarcode && return bc
    bc isa PackedFloatBarcode && return _barcode_from_packed(bc)
    if bc isa PackedBarcode
        out = FloatBarcode()
        sizehint!(out, length(bc.pairs))
        @inbounds for i in eachindex(bc.pairs)
            p = bc.pairs[i]
            out[(float(p.b), float(p.d))] = bc.mults[i]
        end
        return out
    end
    out = FloatBarcode()
    for ((b, d), mult) in bc
        out[(float(b), float(d))] = get(out, (float(b), float(d)), 0) + Int(mult)
    end
    return out
end

function _pack_index_barcode(bc::IndexBarcode)::PackedIndexBarcode
    n = length(bc)
    pairs = Vector{EndpointPair{Int}}(undef, n)
    mults = Vector{Int}(undef, n)
    i = 0
    for ((b, d), m) in bc
        i += 1
        pairs[i] = EndpointPair{Int}(b, d)
        mults[i] = Int(m)
    end
    if n > 1
        ord = sortperm(pairs; by = p -> (p.b, p.d))
        pairs = pairs[ord]
        mults = mults[ord]
    end
    return PackedIndexBarcode(pairs, mults)
end

function _pack_float_barcode(bc)::PackedFloatBarcode
    n = length(bc)
    pairs = Vector{EndpointPair{Float64}}(undef, n)
    mults = Vector{Int}(undef, n)
    i = 0
    for ((b, d), m) in bc
        i += 1
        pairs[i] = EndpointPair{Float64}(float(b), float(d))
        mults[i] = Int(m)
    end
    if n > 1
        ord = sortperm(pairs; by = p -> (p.b, p.d))
        pairs = pairs[ord]
        mults = mults[ord]
    end
    return PackedFloatBarcode(pairs, mults)
end

@inline function _points_from_packed!(out::Vector{Tuple{Float64,Float64}}, pb::PackedFloatBarcode)
    empty!(out)
    sizehint!(out, _packed_total_multiplicity(pb))
    @inbounds for i in eachindex(pb.pairs)
        p = pb.pairs[i]
        m = pb.mults[i]
        for _ in 1:m
            push!(out, (p.b, p.d))
        end
    end
    return out
end

@inline function _points_from_index_packed_and_values!(
    out::Vector{Tuple{Float64,Float64}},
    pb::PackedIndexBarcode,
    vals::AbstractVector{<:Real},
)
    empty!(out)
    sizehint!(out, _packed_total_multiplicity(pb))
    @inbounds for i in eachindex(pb.pairs)
        p = pb.pairs[i]
        b = float(vals[p.b])
        d = float(vals[p.d])
        m = pb.mults[i]
        for _ in 1:m
            push!(out, (b, d))
        end
    end
    return out
end

@inline function _points_from_index_packed_and_values!(
    out::Vector{Tuple{Float64,Float64}},
    pb::PackedIndexBarcode,
    vals_pool::AbstractVector{Float64},
    start::Int,
)
    empty!(out)
    sizehint!(out, _packed_total_multiplicity(pb))
    @inbounds for i in eachindex(pb.pairs)
        p = pb.pairs[i]
        b = vals_pool[start + p.b - 1]
        d = vals_pool[start + p.d - 1]
        m = pb.mults[i]
        for _ in 1:m
            push!(out, (b, d))
        end
    end
    return out
end

@inline function _packed_barcode_from_rank(R::AbstractMatrix{Int}, endpoints::AbstractVector{T}) where {T<:Real}
    m = size(R, 1)
    getR(i, j) = (i < 1 || j < 1 || i > m || j > m || i > j) ? 0 : R[i, j]

    pairs = EndpointPair{T}[]
    mults = Int[]
    sizehint!(pairs, (m * (m + 1)) >>> 1)
    sizehint!(mults, (m * (m + 1)) >>> 1)

    @inbounds for b in 1:m
        for d in (b+1):(m+1)
            mult = getR(b, d-1) - getR(b-1, d-1) - getR(b, d) + getR(b-1, d)
            mult < 0 && error("slice_barcode: negative multiplicity detected at (b,d)=($b,$d)")
            if mult > 0
                push!(pairs, EndpointPair{T}(endpoints[b], endpoints[d]))
                push!(mults, mult)
            end
        end
    end

    return PackedBarcode{T}(pairs, mults)
end

@inline function _barcode_from_packed(pb::PackedBarcode{T}) where {T<:Real}
    out = Dict{Tuple{T,T}, Int}()
    sizehint!(out, length(pb.pairs))
    @inbounds for i in eachindex(pb.pairs)
        p = pb.pairs[i]
        out[(p.b, p.d)] = pb.mults[i]
    end
    return out
end

@inline function _float_barcode_from_index_packed_values(
    pb::PackedIndexBarcode,
    vals::AbstractVector{<:Real},
)
    out = FloatBarcode()
    sizehint!(out, length(pb.pairs))
    @inbounds for i in eachindex(pb.pairs)
        p = pb.pairs[i]
        out[(float(vals[p.b]), float(vals[p.d]))] = pb.mults[i]
    end
    return out
end

@inline function _float_packed_from_index_packed_values(
    pb::PackedIndexBarcode,
    vals::AbstractVector{<:Real},
)::PackedFloatBarcode
    pairs = Vector{EndpointPair{Float64}}(undef, length(pb.pairs))
    mults = copy(pb.mults)
    @inbounds for i in eachindex(pb.pairs)
        p = pb.pairs[i]
        pairs[i] = EndpointPair{Float64}(float(vals[p.b]), float(vals[p.d]))
    end
    return PackedFloatBarcode(pairs, mults)
end

@inline function _float_dict_matrix_from_packed_grid(grid::PackedBarcodeGrid{<:PackedBarcode{Float64}})
    nd, no = size(grid)
    out = Matrix{FloatBarcode}(undef, nd, no)
    @inbounds for j in 1:no, i in 1:nd
        out[i, j] = _barcode_from_packed(grid[i, j])
    end
    return out
end

@inline function _index_dict_matrix_from_packed_grid(grid::PackedBarcodeGrid{<:PackedBarcode{Int}})
    nd, no = size(grid)
    out = Matrix{IndexBarcode}(undef, nd, no)
    @inbounds for j in 1:no, i in 1:nd
        out[i, j] = _barcode_from_packed(grid[i, j])
    end
    return out
end
