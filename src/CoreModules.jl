module CoreModules
# -----------------------------------------------------------------------------
# Core prelude for this project
#  - QQ = Rational{BigInt} is the canonical exact scalar type
#  - Optional feature flags (e.g. for optional PL axis backend)
#  - Thin wrappers for exact linear algebra backends (optional Nemo)
#  - Exact rational <-> string helpers for serialization
# -----------------------------------------------------------------------------

using LinearAlgebra, SparseArrays

# ----- canonical field of scalars used everywhere --------------------------------
"Exact rationals used throughout (Rational{BigInt})."
const QQ = Rational{BigInt}


# ----- coefficient field layer ---------------------------------------------------
module CoeffFields
using LinearAlgebra
using Random
using ..CoreModules: QQ


"Abstract supertype for coefficient fields."
abstract type AbstractCoeffField end

"Exact rationals (QQ)."
struct QQField <: AbstractCoeffField end

"Real (floating) field with tolerances."
struct RealField{T<:AbstractFloat} <: AbstractCoeffField
    rtol::T
    atol::T
end

function RealField(::Type{T}; rtol::T = sqrt(eps(T)), atol::T = zero(T)) where {T<:AbstractFloat}
    RealField{T}(rtol, atol)
end

"Prime field of characteristic p."
struct PrimeField <: AbstractCoeffField
    p::Int
    function PrimeField(p::Integer)
        p > 1 || throw(ArgumentError("prime field requires p > 1"))
        new(Int(p))
    end
end

Base.:(==)(::QQField, ::QQField) = true
Base.:(==)(a::RealField{T}, b::RealField{T}) where {T<:AbstractFloat} =
    (a.rtol == b.rtol) && (a.atol == b.atol)
Base.:(==)(a::PrimeField, b::PrimeField) = a.p == b.p

F2() = PrimeField(2)
F3() = PrimeField(3)
Fp(p::Integer) = PrimeField(p)

"Element type for prime fields, parameterized by modulus."
struct FpElem{p} <: Integer
    val::Int
    function FpElem{p}(x::Integer) where {p}
        x isa FpElem{p} && return x
        new{p}(mod(x, p))
    end
end

Base.show(io::IO, x::FpElem{p}) where {p} = print(io, x.val)
Base.zero(::Type{FpElem{p}}) where {p} = FpElem{p}(0)
Base.one(::Type{FpElem{p}}) where {p} = FpElem{p}(1)
Base.iszero(x::FpElem{p}) where {p} = x.val == 0
Base.conj(x::FpElem{p}) where {p} = x
Base.hash(x::FpElem{p}, h::UInt) where {p} = hash(x.val, h)

Base.convert(::Type{FpElem{p}}, x::Integer) where {p} = FpElem{p}(x)
Base.convert(::Type{FpElem{p}}, x::FpElem{p}) where {p} = x
Base.promote_rule(::Type{FpElem{p}}, ::Type{<:Integer}) where {p} = FpElem{p}

Base.:+(a::FpElem{p}, b::FpElem{p}) where {p} = FpElem{p}(a.val + b.val)
Base.:-(a::FpElem{p}, b::FpElem{p}) where {p} = FpElem{p}(a.val - b.val)
Base.:-(a::FpElem{p}) where {p} = FpElem{p}(-a.val)
Base.:*(a::FpElem{p}, b::FpElem{p}) where {p} = FpElem{p}(a.val * b.val)
Base.:(==)(a::FpElem{p}, b::FpElem{p}) where {p} = a.val == b.val

function Base.inv(a::FpElem{p}) where {p}
    a.val == 0 && throw(DomainError(a, "division by zero in Fp"))
    return FpElem{p}(invmod(a.val, p))
end

Base.:/(a::FpElem{p}, b::FpElem{p}) where {p} = a * inv(b)

"Return the scalar element type used for a given field."
coeff_type(::QQField) = QQ
coeff_type(::RealField{T}) where {T<:AbstractFloat} = T
coeff_type(F::PrimeField) = FpElem{F.p}

"Infer a coefficient field object from an element type."
field_from_eltype(::Type{QQ}) = QQField()
field_from_eltype(::Type{<:Rational}) = QQField()
field_from_eltype(::Type{T}) where {T<:AbstractFloat} = RealField(T)
field_from_eltype(::Type{FpElem{p}}) where {p} = PrimeField(p)
field_from_eltype(::Type{K}) where {K} =
    throw(ArgumentError("no field mapping for element type $(K)"))

"Field-aware zero/one."
Base.zero(F::AbstractCoeffField) = zero(coeff_type(F))
Base.one(F::AbstractCoeffField) = one(coeff_type(F))

"Coerce scalars into the given coefficient field."
coerce(::QQField, x::Integer) = QQ(x)
coerce(::QQField, x::Rational) = QQ(x)
coerce(::QQField, x::AbstractFloat) = rationalize(BigInt, x)
coerce(::QQField, x::QQ) = x
coerce(::QQField, x::FpElem{p}) where {p} = QQ(x.val)

coerce(::RealField{T}, x::Integer) where {T<:AbstractFloat} = T(x)
coerce(::RealField{T}, x::Rational) where {T<:AbstractFloat} =
    T(numerator(x)) / T(denominator(x))
coerce(::RealField{T}, x::AbstractFloat) where {T<:AbstractFloat} = T(x)

function coerce(F::PrimeField, x::Integer)
    K = coeff_type(F)
    return K(x)
end

function coerce(F::PrimeField, x::Rational)
    p = F.p
    den = denominator(x)
    g = gcd(den, p)
    g == 1 || g == one(g) || throw(ArgumentError("denominator not invertible mod $p"))
    num = numerator(x)
    num_mod = Int(mod(num, p))
    den_mod = Int(mod(den, p))
    inv_den = invmod(den_mod, p)
    return FpElem{p}(num_mod * inv_den)
end

function coerce(F::PrimeField, x::FpElem{p}) where {p}
    F.p == p || throw(ArgumentError("cannot coerce FpElem{$p} into Fp($(F.p))"))
    return x
end

coerce(F::PrimeField, x::AbstractFloat) =
    throw(ArgumentError("cannot coerce float into Fp($(F.p)) without an explicit rule"))

"Allocate a dense zeros matrix over the field."
function zeros(F::AbstractCoeffField, m::Integer, n::Integer)
    K = coeff_type(F)
    A = Matrix{K}(undef, m, n)
    fill!(A, zero(K))
    return A
end

"Allocate a dense ones matrix over the field."
function ones(F::AbstractCoeffField, m::Integer, n::Integer)
    K = coeff_type(F)
    A = Matrix{K}(undef, m, n)
    fill!(A, one(K))
    return A
end

"Allocate a dense identity matrix over the field."
function eye(F::AbstractCoeffField, n::Integer)
    K = coeff_type(F)
    A = Matrix{K}(undef, n, n)
    z = zero(K)
    o = one(K)
    @inbounds for j in 1:n
        for i in 1:n
            A[i, j] = (i == j) ? o : z
        end
    end
    return A
end

"Allocate a dense random matrix over the field."
function rand(F::AbstractCoeffField, m::Integer, n::Integer; density::Real=1.0)
    (0.0 <= density <= 1.0) || throw(ArgumentError("density must be in [0,1]"))
    K = coeff_type(F)
    A = Matrix{K}(undef, m, n)

    if density == 1.0
        @inbounds for j in 1:n, i in 1:m
            A[i, j] = _rand_scalar(F)
        end
        return A
    end

    z = zero(K)
    @inbounds for j in 1:n, i in 1:m
        A[i, j] = (Base.rand() <= density) ? _rand_scalar(F) : z
    end
    return A
end

_rand_scalar(::QQField) = QQ(Base.rand(-5:5))
_rand_scalar(::RealField{T}) where {T<:AbstractFloat} = Base.rand(T)
_rand_scalar(F::PrimeField) = FpElem{F.p}(Base.rand(0:F.p-1))

end # module CoeffFields

using .CoeffFields: AbstractCoeffField, QQField, RealField, PrimeField,
    F2, F3, Fp, coeff_type, coerce, FpElem, field_from_eltype,
    eye, zeros, ones, rand

"""
    BackendMatrix{K}

Dense matrix wrapper that can carry an optional backend-native payload
(e.g. a Nemo matrix) to avoid repeated conversion in hot paths.
"""
mutable struct BackendMatrix{K} <: AbstractMatrix{K}
    data::Matrix{K}
    backend::Symbol
    payload::Any
    function BackendMatrix{K}(data::Matrix{K};
                              backend::Symbol=:nemo,
                              payload::Any=nothing) where {K}
        new{K}(data, backend, payload)
    end
end

BackendMatrix(A::AbstractMatrix{K}; backend::Symbol=:nemo, payload::Any=nothing) where {K} =
    BackendMatrix{K}(Matrix{K}(A); backend=backend, payload=payload)

Base.size(A::BackendMatrix) = size(A.data)
Base.axes(A::BackendMatrix) = axes(A.data)
Base.IndexStyle(::Type{<:BackendMatrix}) = IndexCartesian()
Base.getindex(A::BackendMatrix, i::Int, j::Int) = @inbounds A.data[i, j]
function Base.setindex!(A::BackendMatrix, v, i::Int, j::Int)
    @inbounds A.data[i, j] = v
    # Keep cached backend payload coherent with dense storage.
    A.payload = nothing
    return v
end
Base.parent(A::BackendMatrix) = A.data
Base.Matrix(A::BackendMatrix{K}) where {K} = copy(A.data)
Base.copy(A::BackendMatrix{K}) where {K} =
    BackendMatrix{K}(copy(A.data); backend=A.backend, payload=nothing)
Base.convert(::Type{Matrix{K}}, A::BackendMatrix{K}) where {K} = copy(A.data)
Base.convert(::Type{BackendMatrix{K}}, A::AbstractMatrix{K}) where {K} =
    BackendMatrix{K}(Matrix{K}(A); backend=:nemo, payload=nothing)

Base.:*(A::BackendMatrix{K}, B::AbstractMatrix{K}) where {K} = A.data * B
Base.:*(A::AbstractMatrix{K}, B::BackendMatrix{K}) where {K} = A * B.data
Base.:*(A::BackendMatrix{K}, B::BackendMatrix{K}) where {K} = A.data * B.data

@inline _unwrap_backend_matrix(A::BackendMatrix) = A.data
@inline _backend_kind(A::BackendMatrix) = A.backend
@inline _backend_payload(A::BackendMatrix) = A.payload
@inline function _set_backend_payload!(A::BackendMatrix, payload)
    A.payload = payload
    return A
end


"""
    change_field(x, field)

Coerce a structure into the specified coefficient field.
"""
function change_field end


"""
    _append_scaled_triplets!(I, J, V, A, row_off, col_off; scale=one(eltype(A)))
    _append_scaled_triplets!(I, J, V, A, rows, cols; scale=one(eltype(A)))

Internal helper for assembling sparse matrices via (I,J,V) triplets.

This eliminates the common anti-pattern:

    D = zeros(QQ, m, n)
    ... fill a few blocks/entries of D ...
    return sparse(D)

which allocates an m-by-n dense matrix even when the result is structurally sparse.

Two variants:

1. Offset-based (fast):
   Treat `A` as a contiguous block placed at global rows
       row_off .+ (1:size(A,1))
   and global cols
       col_off .+ (1:size(A,2))

2. Indexed (general):
   Place `A` at explicit global row indices `rows` and col indices `cols`.
   This supports non-contiguous placements (e.g. direct sums with custom order).

In both cases we append only nonzero entries of `A` (after scaling by `scale`).
Duplicates are allowed; `sparse(I,J,V,...)` will sum them.

All indexing is 1-based.
"""
function _append_scaled_triplets!(I::Vector{Int}, J::Vector{Int}, V::Vector{K},
                                 A::AbstractMatrix{K},
                                 row_off::Int, col_off::Int;
                                 scale::K = one(K)) where {K}
    # Sparse fast-path
    if A isa SparseMatrixCSC{K,Int}
        Ii, Jj, Vv = findnz(A)
        @inbounds for k in eachindex(Ii)
            a = Vv[k]
            iszero(a) && continue
            push!(I, row_off + Ii[k])
            push!(J, col_off + Jj[k])
            push!(V, scale * a)
        end
        return nothing
    end

    # Transpose/adjoint of sparse: iterate parent nnz and swap indices
    if (A isa LinearAlgebra.Transpose{K,<:SparseMatrixCSC{K,Int}}) ||
       (A isa LinearAlgebra.Adjoint{K,<:SparseMatrixCSC{K,Int}})
        S = parent(A)
        Ii, Jj, Vv = findnz(S)
        @inbounds for k in eachindex(Ii)
            a = Vv[k]
            iszero(a) && continue
            push!(I, row_off + Jj[k])
            push!(J, col_off + Ii[k])
            push!(V, scale * a)
        end
        return nothing
    end

    # Dense / generic: scan entries, skip zeros
    m, n = size(A)
    @inbounds for j in 1:n
        gj = col_off + j
        for i in 1:m
            a = A[i, j]
            iszero(a) && continue
            push!(I, row_off + i)
            push!(J, gj)
            push!(V, scale * a)
        end
    end
    return nothing
end

function _append_scaled_triplets!(I::Vector{Int}, J::Vector{Int}, V::Vector{K},
                                 A::AbstractMatrix{K},
                                 rows::AbstractVector{<:Integer},
                                 cols::AbstractVector{<:Integer};
                                 scale::K = one(K)) where {K}
    @assert length(rows) == size(A, 1)
    @assert length(cols) == size(A, 2)

    if A isa SparseMatrixCSC{K,Int}
        Ii, Jj, Vv = findnz(A)
        @inbounds for k in eachindex(Ii)
            a = Vv[k]
            iszero(a) && continue
            push!(I, Int(rows[Ii[k]]))
            push!(J, Int(cols[Jj[k]]))
            push!(V, scale * a)
        end
        return nothing
    end

    if (A isa LinearAlgebra.Transpose{K,<:SparseMatrixCSC{K,Int}}) ||
       (A isa LinearAlgebra.Adjoint{K,<:SparseMatrixCSC{K,Int}})
        S = parent(A)
        Ii, Jj, Vv = findnz(S)
        @inbounds for k in eachindex(Ii)
            a = Vv[k]
            iszero(a) && continue
            push!(I, Int(rows[Jj[k]]))
            push!(J, Int(cols[Ii[k]]))
            push!(V, scale * a)
        end
        return nothing
    end

    m, n = size(A)
    @inbounds for j in 1:n
        gj = Int(cols[j])
        for i in 1:m
            a = A[i, j]
            iszero(a) && continue
            push!(I, Int(rows[i]))
            push!(J, gj)
            push!(V, scale * a)
        end
    end
    return nothing
end


# ----- feature flags --------------------------------------------------------------



# ----- exact rational <-> string (for JSON round-trips) --------------------------
"Encode a rational as \"num/den\" so it survives JSON round-trips exactly."
rational_to_string(x::QQ) = string(numerator(x), "/", denominator(x))

"Inverse of `rational_to_string`."
function string_to_rational(s::AbstractString)::QQ
    t = split(strip(s), "/")
    length(t) == 2 || error("bad QQ string: $s")
    parse(BigInt, t[1]) // parse(BigInt, t[2])
end

"""
    ResolutionCache()

Thread-safe cache object for repeated resolution-oriented computations.

Stored entries:
- projective resolutions, keyed by `(objectid(module), maxlen)`
- injective resolutions, keyed by `(objectid(module), maxlen)`
- indicator resolution tuples, keyed by `(objectid(HM), objectid(HN), maxlen_or_neg1)`
"""
struct ResolutionKey2
    a::UInt
    maxlen::Int
end

struct ResolutionKey3
    a::UInt
    b::UInt
    maxlen::Int
end

struct ResolutionKey4
    a::UInt
    b::UInt
    maxlen::Int
    tag::UInt8
end

struct ResolutionKey5
    a::UInt
    b::UInt
    len1::Int
    len2::Int
    tag::UInt8
end

@inline _resolution_key2(a, maxlen::Integer) = ResolutionKey2(UInt(objectid(a)), Int(maxlen))
@inline _resolution_key3(a, b, maxlen::Integer) = ResolutionKey3(UInt(objectid(a)), UInt(objectid(b)), Int(maxlen))
@inline _resolution_key4(a, b, maxlen::Integer, tag::Integer) =
    ResolutionKey4(UInt(objectid(a)), UInt(objectid(b)), Int(maxlen), UInt8(tag))
@inline _resolution_key5(a, b, len1::Integer, len2::Integer, tag::Integer=0) =
    ResolutionKey5(UInt(objectid(a)), UInt(objectid(b)), Int(len1), Int(len2), UInt8(tag))

abstract type AbstractCachePayload end

struct ProjectiveResolutionPayload{R} <: AbstractCachePayload
    value::R
end

struct InjectiveResolutionPayload{R} <: AbstractCachePayload
    value::R
end

struct IndicatorResolutionPayload{R} <: AbstractCachePayload
    value::R
end

struct ExtProjectivePayload{R} <: AbstractCachePayload
    value::R
end

struct ExtInjectivePayload{R} <: AbstractCachePayload
    value::R
end

struct ExtUnifiedPayload{R} <: AbstractCachePayload
    value::R
end

struct TorFirstPayload{R} <: AbstractCachePayload
    value::R
end

struct TorSecondPayload{R} <: AbstractCachePayload
    value::R
end

struct HomBicomplexPayload{R} <: AbstractCachePayload
    value::R
end

struct ExtDoubleComplexPayload{R} <: AbstractCachePayload
    value::R
end

struct TorDoubleComplexPlanPayload{R} <: AbstractCachePayload
    value::R
end

struct TorDoubleComplexPayload{R} <: AbstractCachePayload
    value::R
end

struct PosetCachePayload{P} <: AbstractCachePayload
    value::P
end

struct CubicalCachePayload{C} <: AbstractCachePayload
    value::C
end

struct RegionPosetCachePayload{P} <: AbstractCachePayload
    value::P
end

struct GeometryCachePayload{G} <: AbstractCachePayload
    value::G
end

struct ModulePayload{P} <: AbstractCachePayload
    value::P
end

struct ZnEncodingArtifact{P,Pi} <: AbstractCachePayload
    P::P
    pi::Pi
end

struct ZnPushforwardFringeArtifact{H} <: AbstractCachePayload
    H::H
end

struct ZnPushforwardModuleArtifact{H,M} <: AbstractCachePayload
    H::H
    M::M
end

struct ProductPosetCacheEntry{K1,K2,P,Pi1,Pi2} <: AbstractCachePayload
    key1::K1
    key2::K2
    P::P
    pi1::Pi1
    pi2::Pi2
end

mutable struct ResolutionCache
    lock::Base.ReentrantLock
    projective::Dict{ResolutionKey2,ProjectiveResolutionPayload}
    injective::Dict{ResolutionKey2,InjectiveResolutionPayload}
    indicator::Dict{ResolutionKey3,IndicatorResolutionPayload}
    ext_projective::Dict{ResolutionKey3,ExtProjectivePayload}
    ext_injective::Dict{ResolutionKey3,ExtInjectivePayload}
    ext_unified::Dict{ResolutionKey4,ExtUnifiedPayload}
    tor_first::Dict{ResolutionKey3,TorFirstPayload}
    tor_second::Dict{ResolutionKey3,TorSecondPayload}
    hom_bicomplex::Dict{ResolutionKey5,HomBicomplexPayload}
    ext_doublecomplex::Dict{ResolutionKey3,ExtDoubleComplexPayload}
    tor_doublecomplex_plan::Dict{ResolutionKey5,TorDoubleComplexPlanPayload}
    tor_doublecomplex::Dict{ResolutionKey5,TorDoubleComplexPayload}
    projective_promotion_type::Union{Nothing,DataType}
    projective_promotion_hits::Int
    injective_promotion_type::Union{Nothing,DataType}
    injective_promotion_hits::Int
    projective_primary_type::Union{Nothing,DataType}
    projective_primary::Any
    injective_primary_type::Union{Nothing,DataType}
    injective_primary::Any
    indicator_primary_type::Union{Nothing,DataType}
    indicator_primary::Any
    # Thread-sharded memo stores for lock-free fast-path lookups/inserts.
    projective_shards::Vector{Dict{ResolutionKey2,ProjectiveResolutionPayload}}
    injective_shards::Vector{Dict{ResolutionKey2,InjectiveResolutionPayload}}
    indicator_shards::Vector{Dict{ResolutionKey3,IndicatorResolutionPayload}}
    projective_primary_shards::Vector{Any}
    injective_primary_shards::Vector{Any}
    indicator_primary_shards::Vector{Any}
end

function ResolutionCache()
    nshards = max(1, Base.Threads.maxthreadid())
    return ResolutionCache(
        Base.ReentrantLock(),
        Dict{ResolutionKey2,ProjectiveResolutionPayload}(),
        Dict{ResolutionKey2,InjectiveResolutionPayload}(),
        Dict{ResolutionKey3,IndicatorResolutionPayload}(),
        Dict{ResolutionKey3,ExtProjectivePayload}(),
        Dict{ResolutionKey3,ExtInjectivePayload}(),
        Dict{ResolutionKey4,ExtUnifiedPayload}(),
        Dict{ResolutionKey3,TorFirstPayload}(),
        Dict{ResolutionKey3,TorSecondPayload}(),
        Dict{ResolutionKey5,HomBicomplexPayload}(),
        Dict{ResolutionKey3,ExtDoubleComplexPayload}(),
        Dict{ResolutionKey5,TorDoubleComplexPlanPayload}(),
        Dict{ResolutionKey5,TorDoubleComplexPayload}(),
        nothing,
        0,
        nothing,
        0,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        [Dict{ResolutionKey2,ProjectiveResolutionPayload}() for _ in 1:nshards],
        [Dict{ResolutionKey2,InjectiveResolutionPayload}() for _ in 1:nshards],
        [Dict{ResolutionKey3,IndicatorResolutionPayload}() for _ in 1:nshards],
        fill(nothing, nshards),
        fill(nothing, nshards),
        fill(nothing, nshards),
    )
end

function _clear_resolution_cache!(cache::ResolutionCache)
    Base.lock(cache.lock)
    empty!(cache.projective)
    empty!(cache.injective)
    empty!(cache.indicator)
    empty!(cache.ext_projective)
    empty!(cache.ext_injective)
    empty!(cache.ext_unified)
    empty!(cache.tor_first)
    empty!(cache.tor_second)
    empty!(cache.hom_bicomplex)
    empty!(cache.ext_doublecomplex)
    empty!(cache.tor_doublecomplex_plan)
    empty!(cache.tor_doublecomplex)
    cache.projective_promotion_type = nothing
    cache.projective_promotion_hits = 0
    cache.injective_promotion_type = nothing
    cache.injective_promotion_hits = 0
    if cache.projective_primary_type === nothing
        cache.projective_primary = nothing
    else
        empty!(cache.projective_primary)
    end
    if cache.injective_primary_type === nothing
        cache.injective_primary = nothing
    else
        empty!(cache.injective_primary)
    end
    cache.indicator_primary_type = nothing
    cache.indicator_primary = nothing
    for d in cache.projective_shards
        empty!(d)
    end
    for d in cache.injective_shards
        empty!(d)
    end
    for d in cache.indicator_shards
        empty!(d)
    end
    for i in eachindex(cache.projective_primary_shards)
        shard = cache.projective_primary_shards[i]
        shard === nothing || empty!(shard)
    end
    for i in eachindex(cache.injective_primary_shards)
        shard = cache.injective_primary_shards[i]
        shard === nothing || empty!(shard)
    end
    fill!(cache.indicator_primary_shards, nothing)
    Base.unlock(cache.lock)
    return nothing
end

const _ENCODING_POSET_KEY = Tuple{Tuple,Tuple{Vararg{Int}}}
const _ENCODING_CUBICAL_KEY = Tuple{Vararg{Int}}
const _ENCODING_GEOMETRY_KEY = Tuple

struct _SessionProductKey
    a::UInt
    b::UInt
    @inline _SessionProductKey(a::UInt, b::UInt) = new(a, b)
    @inline _SessionProductKey(a, b) = new(UInt(objectid(a)), UInt(objectid(b)))
end

const _SESSION_PRODUCT_KEY = _SessionProductKey
const _SESSION_ZN_ENCODING_KEY = Tuple{UInt64,Symbol,Int}
const _SESSION_ZN_PLAN_KEY = Tuple{UInt64,UInt64}
const _SESSION_ZN_PUSH_KEY = Tuple{UInt64,Symbol,UInt64,UInt}
const _SESSION_ZN_PLAN_VALUE = NamedTuple{
    (:flat_idxs,:inj_idxs,:zero_pairs),
    Tuple{Vector{Int},Vector{Int},Vector{Tuple{Int,Int}}},
}

"""
    EncodingCache()

Per-encoding cache bucket for geometry/poset-derived artifacts.

Intended contents:
- `posets`: derived encoding posets (e.g. axes/orientation -> poset)
- `cubical`: cubical cell structures keyed by grid size
- `region_posets`: reconstructed region posets keyed by signature identities
"""
mutable struct EncodingCache
    lock::Base.ReentrantLock
    posets::Dict{_ENCODING_POSET_KEY,PosetCachePayload}
    cubical::Dict{_ENCODING_CUBICAL_KEY,CubicalCachePayload}
    region_posets::Dict{Tuple{UInt,UInt,Symbol},RegionPosetCachePayload}
    geometry::Dict{_ENCODING_GEOMETRY_KEY,GeometryCachePayload}
end

EncodingCache() = EncodingCache(Base.ReentrantLock(),
                                Dict{_ENCODING_POSET_KEY,PosetCachePayload}(),
                                Dict{_ENCODING_CUBICAL_KEY,CubicalCachePayload}(),
                                Dict{Tuple{UInt,UInt,Symbol},RegionPosetCachePayload}(),
                                Dict{_ENCODING_GEOMETRY_KEY,GeometryCachePayload}())

function _clear_encoding_cache!(cache::EncodingCache)
    Base.lock(cache.lock)
    try
        empty!(cache.posets)
        empty!(cache.cubical)
        empty!(cache.region_posets)
        empty!(cache.geometry)
    finally
        Base.unlock(cache.lock)
    end
    return nothing
end

"""
    ModuleCache(module_id, field_key)

Module-scoped cache bucket keyed by module identity + field identity.
"""
mutable struct ModuleCache
    module_id::UInt
    field_key::UInt
    resolution::ResolutionCache
    payload::Dict{Symbol,ModulePayload}
end

ModuleCache(module_id::UInt, field_key::UInt) =
    ModuleCache(module_id, field_key, ResolutionCache(), Dict{Symbol,ModulePayload}())

function _clear_module_cache!(cache::ModuleCache)
    _clear_resolution_cache!(cache.resolution)
    empty!(cache.payload)
    return nothing
end

abstract type AbstractHomSystemCache end
abstract type AbstractSlicePlanCache end

"""
    SessionCache()

Cross-query cache root with explicit lifetime controlled by the caller.

`SessionCache` is the canonical reusable cache object for high-level workflow
entrypoints that accept `cache=...`.

Best practices:
- simple users should usually pass `cache=:auto` and let the workflow create a
  short-lived cache automatically,
- advanced users should construct one `SessionCache()` and reuse it across
  related calls when they want warm-path speedups,
- pass `cache=nothing` to disable cross-call reuse explicitly.

`SessionCache` is a workflow-level cache root, not a mathematical object. Users
should treat it as an opaque performance tool rather than inspecting internal
storage fields directly.

Hierarchy:
- SessionCache (long-lived, workflow-level)
  - EncodingCache buckets keyed by poset identity
  - ModuleCache buckets keyed by `(objectid(module), _field_cache_key(module.field))`
  - Zn encoding artifacts `(P, pi)` keyed by `(encoding_fingerprint, poset_kind, max_regions)`
  - Zn pushforward plans keyed by `(encoding_fingerprint, flange_fingerprint)`
  - Zn pushed fringes keyed by `(encoding_fingerprint, poset_kind, flange_fingerprint, field_key)`
  - Zn pushed modules keyed by `(encoding_fingerprint, poset_kind, flange_fingerprint, field_key)`
"""
mutable struct SessionCache
    encoding_locks::Vector{Base.ReentrantLock}
    encoding::Vector{Dict{UInt,EncodingCache}}
    module_locks::Vector{Base.ReentrantLock}
    modules::Vector{Dict{Tuple{UInt,UInt},ModuleCache}}
    module_field_keys::Vector{IdDict{Any,UInt}}
    resolution::ResolutionCache
    hom_system::Union{Nothing,AbstractHomSystemCache}
    slice_plan::Union{Nothing,AbstractSlicePlanCache}
    zn_encoding_locks::Vector{Base.ReentrantLock}
    zn_plan_locks::Vector{Base.ReentrantLock}
    zn_fringe_locks::Vector{Base.ReentrantLock}
    zn_module_locks::Vector{Base.ReentrantLock}
    zn_encoding_artifacts::Vector{Dict{_SESSION_ZN_ENCODING_KEY,ZnEncodingArtifact{Any,Any}}}
    zn_pushforward_plan::Vector{Dict{_SESSION_ZN_PLAN_KEY,_SESSION_ZN_PLAN_VALUE}}
    zn_pushforward_fringe::Vector{Dict{_SESSION_ZN_PUSH_KEY,ZnPushforwardFringeArtifact{Any}}}
    zn_pushforward_module::Vector{Dict{_SESSION_ZN_PUSH_KEY,ZnPushforwardModuleArtifact{Any,Any}}}
    product_dense::Dict{_SESSION_PRODUCT_KEY,ProductPosetCacheEntry{Any,Any,Any,Any,Any}}
    product_obj::Dict{_SESSION_PRODUCT_KEY,ProductPosetCacheEntry{Any,Any,Any,Any,Any}}
end

const _SESSION_CACHE_SHARDS = 16
const _SESSION_ZN_CACHE_SHARDS = 16

@inline _session_cache_nshards() = Threads.nthreads() == 1 ? 1 : _SESSION_CACHE_SHARDS
@inline _session_zn_cache_nshards() = Threads.nthreads() == 1 ? 1 : _SESSION_ZN_CACHE_SHARDS

function SessionCache()
    ncore_shards = _session_cache_nshards()
    nzn_shards = _session_zn_cache_nshards()
    return SessionCache([Base.ReentrantLock() for _ in 1:ncore_shards],
                              [Dict{UInt,EncodingCache}() for _ in 1:ncore_shards],
                              [Base.ReentrantLock() for _ in 1:ncore_shards],
                              [Dict{Tuple{UInt,UInt},ModuleCache}() for _ in 1:ncore_shards],
                              [IdDict{Any,UInt}() for _ in 1:ncore_shards],
                              ResolutionCache(),
                              nothing,
                              nothing,
                              [Base.ReentrantLock() for _ in 1:nzn_shards],
                              [Base.ReentrantLock() for _ in 1:nzn_shards],
                              [Base.ReentrantLock() for _ in 1:nzn_shards],
                              [Base.ReentrantLock() for _ in 1:nzn_shards],
                              [Dict{_SESSION_ZN_ENCODING_KEY,ZnEncodingArtifact{Any,Any}}() for _ in 1:nzn_shards],
                              [Dict{_SESSION_ZN_PLAN_KEY,_SESSION_ZN_PLAN_VALUE}() for _ in 1:nzn_shards],
                              [Dict{_SESSION_ZN_PUSH_KEY,ZnPushforwardFringeArtifact{Any}}() for _ in 1:nzn_shards],
                              [Dict{_SESSION_ZN_PUSH_KEY,ZnPushforwardModuleArtifact{Any,Any}}() for _ in 1:nzn_shards],
                              Dict{_SESSION_PRODUCT_KEY,ProductPosetCacheEntry{Any,Any,Any,Any,Any}}(),
                              Dict{_SESSION_PRODUCT_KEY,ProductPosetCacheEntry{Any,Any,Any,Any,Any}}())
end

@inline function _session_cache_summary_counts(session::SessionCache)
    return (
        encoding_buckets=_session_encoding_bucket_count(session),
        module_buckets=_session_module_bucket_count(session),
        zn_encoding_artifacts=_session_zn_encoding_artifact_count(session),
        zn_pushforward_plans=_session_zn_pushforward_plan_count(session),
        zn_pushforward_fringes=_session_zn_pushforward_fringe_count(session),
        zn_pushforward_modules=_session_zn_pushforward_module_count(session),
        product_dense=length(session.product_dense),
        product_object=length(session.product_obj),
        has_hom_system=session.hom_system !== nothing,
        has_slice_plan=session.slice_plan !== nothing,
    )
end

function Base.show(io::IO, session::SessionCache)
    counts = _session_cache_summary_counts(session)
    print(io,
          "SessionCache(",
          "encoding=", counts.encoding_buckets,
          ", modules=", counts.module_buckets,
          ", zn=", counts.zn_encoding_artifacts + counts.zn_pushforward_plans +
                   counts.zn_pushforward_fringes + counts.zn_pushforward_modules,
          ")")
end

function Base.show(io::IO, ::MIME"text/plain", session::SessionCache)
    counts = _session_cache_summary_counts(session)
    print(io,
          "SessionCache\n",
          "  encoding buckets: ", counts.encoding_buckets, "\n",
          "  module buckets: ", counts.module_buckets, "\n",
          "  zn encoding artifacts: ", counts.zn_encoding_artifacts, "\n",
          "  zn pushforward plans: ", counts.zn_pushforward_plans, "\n",
          "  zn pushforward fringes: ", counts.zn_pushforward_fringes, "\n",
          "  zn pushforward modules: ", counts.zn_pushforward_modules, "\n",
          "  product caches: dense=", counts.product_dense,
          ", object=", counts.product_object, "\n",
          "  cached heavy helpers: hom_system=", counts.has_hom_system,
          ", slice_plan=", counts.has_slice_plan, "\n",
          "  shard layout: core=", length(session.encoding),
          ", zn=", length(session.zn_encoding_artifacts))
end

const _FIELD_CACHE_SEED = UInt(0x9E37_79B9_7F4A_7C15)

@inline _field_cache_key(::QQField)::UInt = UInt(0x514F_514F_514F_514F)
@inline _field_cache_key(F::PrimeField)::UInt =
    xor(UInt(0x4650_5F00_0000_0001), UInt(F.p))
@inline _field_cache_key(F::RealField{T}) where {T<:AbstractFloat} =
    UInt(hash((T, F.rtol, F.atol), _FIELD_CACHE_SEED))
@inline _field_cache_key(field::AbstractCoeffField)::UInt =
    UInt(hash(field, _FIELD_CACHE_SEED))

@inline _poset_cache_key(P)::UInt = UInt(objectid(P))
@inline function _module_cache_key(M)
    fid = hasproperty(M, :field) ? _field_cache_key(getproperty(M, :field)) : UInt(0)
    return (UInt(objectid(M)), fid)
end

@inline _session_shard_index(key::UInt, nshards::Int) = Int((key % UInt(nshards)) + 1)
@inline _session_shard_index(key::Tuple{UInt,UInt}, nshards::Int) = Int((key[1] % UInt(nshards)) + 1)
@inline _session_shard_index(key::Tuple{UInt64,Symbol,Int}, nshards::Int) = Int((key[1] % UInt64(nshards)) + 1)
@inline _session_shard_index(key::Tuple{UInt64,Symbol,UInt64,UInt}, nshards::Int) = Int((key[1] % UInt64(nshards)) + 1)

function _module_cache_key(session::SessionCache, M)
    mid = UInt(objectid(M))
    hasproperty(M, :field) || return (mid, UInt(0))
    nshards = length(session.module_field_keys)
    idx = _session_shard_index(mid, nshards)
    keys = session.module_field_keys[idx]
    if Threads.nthreads() == 1
        if haskey(keys, M)
            return (mid, keys[M])
        end
        fid = _field_cache_key(getproperty(M, :field))
        keys[M] = fid
        return (mid, fid)
    end
    lock = session.module_locks[idx]
    Base.lock(lock)
    try
        if haskey(keys, M)
            return (mid, keys[M])
        end
        fid = _field_cache_key(getproperty(M, :field))
        keys[M] = fid
        return (mid, fid)
    finally
        Base.unlock(lock)
    end
end

function _encoding_cache!(session::SessionCache, key::UInt)
    nshards = length(session.encoding)
    idx = _session_shard_index(key, nshards)
    shard = session.encoding[idx]
    if Threads.nthreads() == 1
        return get!(shard, key) do
            EncodingCache()
        end
    end
    lock = session.encoding_locks[idx]
    Base.lock(lock)
    try
        return get!(shard, key) do
            EncodingCache()
        end
    finally
        Base.unlock(lock)
    end
end

_encoding_cache!(session::SessionCache, P) = _encoding_cache!(session, _poset_cache_key(P))

function _module_cache!(session::SessionCache, key::Tuple{UInt,UInt})
    nshards = length(session.modules)
    idx = _session_shard_index(key, nshards)
    shard = session.modules[idx]
    if Threads.nthreads() == 1
        return get!(shard, key) do
            ModuleCache(key[1], key[2])
        end
    end
    lock = session.module_locks[idx]
    Base.lock(lock)
    try
        return get!(shard, key) do
            ModuleCache(key[1], key[2])
        end
    finally
        Base.unlock(lock)
    end
end

_module_cache!(session::SessionCache, M) = _module_cache!(session, _module_cache_key(session, M))

function _invalidate_encoding_cache!(session::SessionCache, key::UInt)
    nshards = length(session.encoding)
    idx = _session_shard_index(key, nshards)
    shard = session.encoding[idx]
    if Threads.nthreads() == 1
        cache = pop!(shard, key, nothing)
        cache === nothing || _clear_encoding_cache!(cache)
        return nothing
    end
    lock = session.encoding_locks[idx]
    Base.lock(lock)
    try
        cache = pop!(shard, key, nothing)
        cache === nothing || _clear_encoding_cache!(cache)
    finally
        Base.unlock(lock)
    end
    return nothing
end

_invalidate_encoding_cache!(session::SessionCache, P) =
    _invalidate_encoding_cache!(session, _poset_cache_key(P))

function _invalidate_module_cache!(session::SessionCache, key::Tuple{UInt,UInt})
    nshards = length(session.modules)
    idx = _session_shard_index(key, nshards)
    shard = session.modules[idx]
    if Threads.nthreads() == 1
        cache = pop!(shard, key, nothing)
        cache === nothing || _clear_module_cache!(cache)
        return nothing
    end
    lock = session.module_locks[idx]
    Base.lock(lock)
    try
        cache = pop!(shard, key, nothing)
        cache === nothing || _clear_module_cache!(cache)
    finally
        Base.unlock(lock)
    end
    return nothing
end

_invalidate_module_cache!(session::SessionCache, M) =
    _invalidate_module_cache!(session, _module_cache_key(session, M))

function _session_encoding_values(session::SessionCache)
    out = EncodingCache[]
    for i in eachindex(session.encoding)
        if Threads.nthreads() == 1
            append!(out, values(session.encoding[i]))
        else
            Base.lock(session.encoding_locks[i])
            try
                append!(out, values(session.encoding[i]))
            finally
                Base.unlock(session.encoding_locks[i])
            end
        end
    end
    return out
end

function _session_module_values(session::SessionCache)
    out = ModuleCache[]
    for i in eachindex(session.modules)
        if Threads.nthreads() == 1
            append!(out, values(session.modules[i]))
        else
            Base.lock(session.module_locks[i])
            try
                append!(out, values(session.modules[i]))
            finally
                Base.unlock(session.module_locks[i])
            end
        end
    end
    return out
end

@inline _session_encoding_bucket_count(session::SessionCache) = sum(length, session.encoding)
@inline _session_module_bucket_count(session::SessionCache) = sum(length, session.modules)

@inline _session_resolution_cache(session::SessionCache) = session.resolution
@inline _session_resolution_cache(session::SessionCache, M) = _module_cache!(session, M).resolution

@inline _session_hom_cache(session::SessionCache) = session.hom_system
@inline function _set_session_hom_cache!(session::SessionCache, cache::AbstractHomSystemCache)
    session.hom_system = cache
    return cache
end

@inline _session_slice_plan_cache(session::SessionCache) = session.slice_plan
@inline function _set_session_slice_plan_cache!(session::SessionCache, cache::AbstractSlicePlanCache)
    session.slice_plan = cache
    return cache
end

@inline _session_zn_encoding_artifact_count(session::SessionCache) = sum(length, session.zn_encoding_artifacts)
@inline _session_zn_pushforward_plan_count(session::SessionCache) = sum(length, session.zn_pushforward_plan)
@inline _session_zn_pushforward_fringe_count(session::SessionCache) = sum(length, session.zn_pushforward_fringe)
@inline _session_zn_pushforward_module_count(session::SessionCache) = sum(length, session.zn_pushforward_module)

@inline function _session_get_zn_pushforward_plan(session::SessionCache,
                                                  encoding_fp::UInt64,
                                                  flange_fp::UInt64)
    key = (encoding_fp, flange_fp)
    nshards = length(session.zn_pushforward_plan)
    idx = _session_shard_index(key, nshards)
    shard = session.zn_pushforward_plan[idx]
    if Threads.nthreads() == 1
        return get(shard, key, nothing)
    end
    lock = session.zn_plan_locks[idx]
    Base.lock(lock)
    try
        return get(shard, key, nothing)
    finally
        Base.unlock(lock)
    end
end

@inline function _session_get_zn_encoding_artifact(session::SessionCache,
                                                   encoding_fp::UInt64,
                                                   poset_kind::Symbol,
                                                   max_regions::Int)
    key = (encoding_fp, poset_kind, max_regions)
    nshards = length(session.zn_encoding_artifacts)
    idx = _session_shard_index(key, nshards)
    shard = session.zn_encoding_artifacts[idx]
    if Threads.nthreads() == 1
        entry = get(shard, key, nothing)
        entry === nothing && return nothing
        return (P=entry.P, pi=entry.pi)
    end
    lock = session.zn_encoding_locks[idx]
    Base.lock(lock)
    try
        entry = get(shard, key, nothing)
        entry === nothing && return nothing
        return (P=entry.P, pi=entry.pi)
    finally
        Base.unlock(lock)
    end
end

@inline function _session_set_zn_encoding_artifact!(session::SessionCache,
                                                    encoding_fp::UInt64,
                                                    poset_kind::Symbol,
                                                    max_regions::Int,
                                                    artifact)
    key = (encoding_fp, poset_kind, max_regions)
    nshards = length(session.zn_encoding_artifacts)
    idx = _session_shard_index(key, nshards)
    shard = session.zn_encoding_artifacts[idx]
    payload = artifact isa NamedTuple{(:P,:pi)} ?
        ZnEncodingArtifact{Any,Any}(artifact.P, artifact.pi) :
        ZnEncodingArtifact{Any,Any}(getproperty(artifact, :P), getproperty(artifact, :pi))
    if Threads.nthreads() == 1
        shard[key] = payload
    else
        lock = session.zn_encoding_locks[idx]
        Base.lock(lock)
        try
            shard[key] = payload
        finally
            Base.unlock(lock)
        end
    end
    return (P=payload.P, pi=payload.pi)
end

@inline function _session_set_zn_pushforward_plan!(session::SessionCache,
                                                   encoding_fp::UInt64,
                                                   flange_fp::UInt64,
                                                   plan)
    key = (encoding_fp, flange_fp)
    nshards = length(session.zn_pushforward_plan)
    idx = _session_shard_index(key, nshards)
    shard = session.zn_pushforward_plan[idx]
    flat_idxs = getproperty(plan, :flat_idxs)
    inj_idxs = getproperty(plan, :inj_idxs)
    zero_pairs = getproperty(plan, :zero_pairs)
    payload = (
        flat_idxs = flat_idxs isa Vector{Int} ? flat_idxs : Vector{Int}(flat_idxs),
        inj_idxs = inj_idxs isa Vector{Int} ? inj_idxs : Vector{Int}(inj_idxs),
        zero_pairs = zero_pairs isa Vector{Tuple{Int,Int}} ? zero_pairs : Vector{Tuple{Int,Int}}(zero_pairs),
    )
    if Threads.nthreads() == 1
        shard[key] = payload
    else
        lock = session.zn_plan_locks[idx]
        Base.lock(lock)
        try
            shard[key] = payload
        finally
            Base.unlock(lock)
        end
    end
    return payload
end

@inline function _session_get_zn_pushforward_fringe(session::SessionCache,
                                                    encoding_fp::UInt64,
                                                    poset_kind::Symbol,
                                                    flange_fp::UInt64,
                                                    field_key::UInt)
    key = (encoding_fp, poset_kind, flange_fp, field_key)
    nshards = length(session.zn_pushforward_fringe)
    idx = _session_shard_index(key, nshards)
    shard = session.zn_pushforward_fringe[idx]
    if Threads.nthreads() == 1
        entry = get(shard, key, nothing)
        return entry === nothing ? nothing : entry.H
    end
    lock = session.zn_fringe_locks[idx]
    Base.lock(lock)
    try
        entry = get(shard, key, nothing)
        return entry === nothing ? nothing : entry.H
    finally
        Base.unlock(lock)
    end
end

@inline function _session_set_zn_pushforward_fringe!(session::SessionCache,
                                                     encoding_fp::UInt64,
                                                     poset_kind::Symbol,
                                                     flange_fp::UInt64,
                                                     field_key::UInt,
                                                     fringe)
    key = (encoding_fp, poset_kind, flange_fp, field_key)
    nshards = length(session.zn_pushforward_fringe)
    idx = _session_shard_index(key, nshards)
    shard = session.zn_pushforward_fringe[idx]
    payload = ZnPushforwardFringeArtifact{Any}(fringe)
    if Threads.nthreads() == 1
        shard[key] = payload
    else
        lock = session.zn_fringe_locks[idx]
        Base.lock(lock)
        try
            shard[key] = payload
        finally
            Base.unlock(lock)
        end
    end
    return fringe
end

@inline function _session_get_zn_pushforward_module(session::SessionCache,
                                                    encoding_fp::UInt64,
                                                    poset_kind::Symbol,
                                                    flange_fp::UInt64,
                                                    field_key::UInt)
    key = (encoding_fp, poset_kind, flange_fp, field_key)
    nshards = length(session.zn_pushforward_module)
    idx = _session_shard_index(key, nshards)
    shard = session.zn_pushforward_module[idx]
    if Threads.nthreads() == 1
        entry = get(shard, key, nothing)
        return entry === nothing ? nothing : (H=entry.H, M=entry.M)
    end
    lock = session.zn_module_locks[idx]
    Base.lock(lock)
    try
        entry = get(shard, key, nothing)
        return entry === nothing ? nothing : (H=entry.H, M=entry.M)
    finally
        Base.unlock(lock)
    end
end

@inline function _session_set_zn_pushforward_module!(session::SessionCache,
                                                     encoding_fp::UInt64,
                                                     poset_kind::Symbol,
                                                     flange_fp::UInt64,
                                                     field_key::UInt,
                                                     mod)
    key = (encoding_fp, poset_kind, flange_fp, field_key)
    nshards = length(session.zn_pushforward_module)
    idx = _session_shard_index(key, nshards)
    shard = session.zn_pushforward_module[idx]
    payload = mod isa NamedTuple{(:H,:M)} ?
        ZnPushforwardModuleArtifact{Any,Any}(mod.H, mod.M) :
        ZnPushforwardModuleArtifact{Any,Any}(getproperty(mod, :H), getproperty(mod, :M))
    if Threads.nthreads() == 1
        shard[key] = payload
    else
        lock = session.zn_module_locks[idx]
        Base.lock(lock)
        try
            shard[key] = payload
        finally
            Base.unlock(lock)
        end
    end
    return (H=payload.H, M=payload.M)
end

const _WORKFLOW_ENCODING_CACHE_KEY = typemax(UInt)

@inline function _workflow_cache_argument_error(cache)
    got = cache === nothing ? "nothing" : repr(cache)
    return ArgumentError(
        "cache must be one of :auto, nothing, or SessionCache(). " *
        "Use cache=:auto for per-call automatic reuse, cache=SessionCache() " *
        "to reuse work across calls, or cache=nothing to disable caching. " *
        "Got $(got)."
    )
end

@inline function _resolve_workflow_session_cache(cache)
    if cache === :auto
        return SessionCache()
    elseif cache === nothing
        return nothing
    elseif cache isa SessionCache
        return cache
    end
    throw(_workflow_cache_argument_error(cache))
end

@inline function _resolve_workflow_specialized_cache(cache, ::Type{T}) where {T}
    # Public workflow contract stays simple: only :auto|nothing|SessionCache.
    # Specialized caches are derived from the resolved SessionCache internally.
    return nothing, _resolve_workflow_session_cache(cache)
end

@inline function _workflow_encoding_cache(session_cache::Union{Nothing,SessionCache})
    session_cache === nothing && return nothing
    return _encoding_cache!(session_cache, _WORKFLOW_ENCODING_CACHE_KEY)
end

@inline function _resolution_cache_from_session(cache::Union{Nothing,ResolutionCache},
                                                session_cache::Union{Nothing,SessionCache},
                                                M=nothing)
    cache !== nothing && return cache
    session_cache === nothing && return nothing
    return M === nothing ? _session_resolution_cache(session_cache) :
                           _session_resolution_cache(session_cache, M)
end

@inline function _slot_cache_from_session(cache,
                                          session_cache::Union{Nothing,SessionCache},
                                          getter::Function,
                                          setter::Function,
                                          expected::Type,
                                          ctor)
    cache !== nothing && return cache
    session_cache === nothing && return nothing
    slot = getter(session_cache)
    if !(slot isa expected)
        slot = ctor()
        setter(session_cache, slot)
    end
    return slot
end

function _clear_session_cache!(session::SessionCache)
    for i in eachindex(session.encoding)
        Base.lock(session.encoding_locks[i])
        try
            for c in values(session.encoding[i])
                _clear_encoding_cache!(c)
            end
            empty!(session.encoding[i])
        finally
            Base.unlock(session.encoding_locks[i])
        end
    end
    for i in eachindex(session.modules)
        Base.lock(session.module_locks[i])
        try
            for c in values(session.modules[i])
                _clear_module_cache!(c)
            end
            empty!(session.modules[i])
            empty!(session.module_field_keys[i])
        finally
            Base.unlock(session.module_locks[i])
        end
    end
    _clear_resolution_cache!(session.resolution)
    session.hom_system = nothing
    session.slice_plan = nothing
    empty!(session.product_dense)
    empty!(session.product_obj)
    for i in eachindex(session.zn_encoding_artifacts)
        Base.lock(session.zn_encoding_locks[i])
        try
            empty!(session.zn_encoding_artifacts[i])
        finally
            Base.unlock(session.zn_encoding_locks[i])
        end
    end
    for i in eachindex(session.zn_pushforward_plan)
        Base.lock(session.zn_plan_locks[i])
        try
            empty!(session.zn_pushforward_plan[i])
        finally
            Base.unlock(session.zn_plan_locks[i])
        end
    end
    for i in eachindex(session.zn_pushforward_fringe)
        Base.lock(session.zn_fringe_locks[i])
        try
            empty!(session.zn_pushforward_fringe[i])
        finally
            Base.unlock(session.zn_fringe_locks[i])
        end
    end
    for i in eachindex(session.zn_pushforward_module)
        Base.lock(session.zn_module_locks[i])
        try
            empty!(session.zn_pushforward_module[i])
        finally
            Base.unlock(session.zn_module_locks[i])
        end
    end
    return nothing
end

end # module CoreModules
