# finite_fringe/sets_and_types.jl
# Scope: upset/downset types, layout/cache helper structs, set operations, closures, and equality/hash support.

"""
    Upset(P, mask) -> Upset

An upset (upward-closed subset) of a finite ambient poset.

# Mathematical meaning

An `Upset` represents a subset `U subseteq P` such that whenever `p in U` and `p <= q`
in the ambient poset, then `q in U`.

# Inputs

- `P::AbstractPoset`: the ambient finite poset.
- `mask::BitVector`: membership mask of length `nvertices(P)`.

# Output

- An `Upset{typeof(P)}` attached to the ambient poset `P`.

# Stored representation

- `P`: the ambient poset object.
- `mask[i] == true` if and only if vertex `i` belongs to the upset.

# Invariants

- `length(mask) == nvertices(P)`.
- Users are expected to construct semantically valid upsets. The inner
  constructor checks only the mask length, while `check_upset` verifies closure.

# Failure / contract behavior

- Throws if the mask length does not match the ambient poset size.

# Best practices

- Prefer `principal_upset`, `upset_closure`, or `upset_from_generators` over
  hand-building masks unless you already know the mask is closed.
- Use `support(U)` and `contains(U, q)` in user code instead of reaching into
  `U.mask` directly.
"""
struct Upset{P<:AbstractPoset}
    P::P
    mask::BitVector
    function Upset(Pobj::PT, mask::BitVector) where {PT<:AbstractPoset}
        nvertices(Pobj) == length(mask) ||
            error("Upset: mask length $(length(mask)) must equal nvertices(P)=$(nvertices(Pobj)).")
        new{PT}(Pobj, mask)
    end
end

"""
    Downset(P, mask) -> Downset

A downset (downward-closed subset) of a finite ambient poset.

# Mathematical meaning

A `Downset` represents a subset `D subseteq P` such that whenever `q in D` and `p <= q`
in the ambient poset, then `p in D`.

# Inputs

- `P::AbstractPoset`: the ambient finite poset.
- `mask::BitVector`: membership mask of length `nvertices(P)`.

# Output

- A `Downset{typeof(P)}` attached to the ambient poset `P`.

# Stored representation

- `P`: the ambient poset object.
- `mask[i] == true` if and only if vertex `i` belongs to the downset.

# Invariants

- `length(mask) == nvertices(P)`.
- Users are expected to construct semantically valid downsets. The inner
  constructor checks only the mask length, while `check_downset` verifies
  closure.

# Failure / contract behavior

- Throws if the mask length does not match the ambient poset size.

# Best practices

- Prefer `principal_downset`, `downset_closure`, or `downset_from_generators`
  over hand-building masks unless you already know the mask is closed.
- Use `support(D)` and `contains(D, q)` in user code instead of reaching into
  `D.mask` directly.
"""
struct Downset{P<:AbstractPoset}
    P::P
    mask::BitVector
    function Downset(Pobj::PT, mask::BitVector) where {PT<:AbstractPoset}
        nvertices(Pobj) == length(mask) ||
            error("Downset: mask length $(length(mask)) must equal nvertices(P)=$(nvertices(Pobj)).")
        new{PT}(Pobj, mask)
    end
end

@inline _ff_support(mask::BitVector) = findall(identity, mask)

"""
    support(U::Upset) -> Vector{Int}
    support(D::Downset) -> Vector{Int}

Return the list of vertices contained in an upset or downset.

This is the semantic accessor for set membership in `FiniteFringe`: prefer it
when you want the actual support as vertex labels, and reserve direct `mask`
inspection for low-level or allocation-sensitive code.
"""
@inline support(U::Upset) = _ff_support(U.mask)
@inline support(D::Downset) = _ff_support(D.mask)

"""
    contains(U::Upset, q::Integer) -> Bool
    contains(D::Downset, q::Integer) -> Bool

Return whether vertex `q` belongs to the given upset or downset.

This is the notebook-friendly membership query for finite-fringe set objects.
It uses the stored membership mask directly and throws `BoundsError` when `q`
is outside the base poset.
"""
@inline function contains(U::Upset, q::Integer)
    qi = Int(q)
    1 <= qi <= length(U.mask) || throw(BoundsError(U.mask, qi))
    return U.mask[qi]
end

@inline function contains(D::Downset, q::Integer)
    qi = Int(q)
    1 <= qi <= length(D.mask) || throw(BoundsError(D.mask, qi))
    return D.mask[qi]
end

@inline function _fringe_describe(U::Upset)
    return (kind=:upset,
            nvertices=nvertices(U.P),
            cardinality=count(U.mask),
            support=support(U))
end

@inline function _fringe_describe(D::Downset)
    return (kind=:downset,
            nvertices=nvertices(D.P),
            cardinality=count(D.mask),
            support=support(D))
end

@inline function _support_preview(supp::AbstractVector{<:Integer}; max_items::Int=8)
    if length(supp) <= max_items
        return "(support=$(repr(collect(supp))))"
    end
    head = collect(supp[1:max_items])
    return "(support=$(repr(head))..., cardinality=$(length(supp)))"
end

struct _WPairData
    # Packed row lists by source-upset component.
    u_ptr::Vector{Int}
    u_rows::Vector{Int}
    # Packed row lists by target-downset component.
    d_ptr::Vector{Int}
    d_rows::Vector{Int}
end

struct _FringeComponentDecomp
    comp_id::Vector{Vector{Int}}
    comp_words::Vector{Vector{Vector{UInt64}}}
    comp_n::Vector{Int}
end

struct _FringeLayoutSketch
    Ucomp_id_M::Vector{Vector{Int}}
    Ucomp_n_M::Vector{Int}
    Dcomp_id_N::Vector{Vector{Int}}
    Dcomp_n_N::Vector{Int}
    U_targets::Vector{Vector{BitVector}}
    D_targets::Vector{Vector{BitVector}}
    V1_dim::Int
    V2_dim::Int
    t_work_est::Int
    s_work_est::Int
end

struct _FringeLayoutPlan
    sketch::_FringeLayoutSketch
    w_index::Matrix{Int}
    w_data::Vector{_WPairData}
    W_dim::Int
    nUM::Int
    nDN::Int
end

mutable struct _FringeDenseIdxPlan{K}
    W_dim::Int
    V1_dim::Int
    V2_dim::Int
    t_rows::Vector{Int}
    t_cols::Vector{Int}
    t_tN::Vector{Int}
    t_jN::Vector{Int}
    s_rows::Vector{Int}
    s_cols::Vector{Int}
    s_sM::Vector{Int}
    s_iM::Vector{Int}
    Tbuf::Matrix{K}
    Sbuf::Matrix{K}
    bigbuf::Matrix{K}
end

struct _SparseRowPlan
    ptr::Vector{Int}
    cols::Vector{Int}
    nzptr::Vector{Int}
    row_nnz::Vector{Int}
    max_row_nnz::Int
end

mutable struct _FringeSparsePlan{K}
    W_dim::Int
    V1::Vector{NTuple{3,Int}}
    V2::Vector{NTuple{3,Int}}
    w_index::Matrix{Int}
    w_data::Vector{_WPairData}
    nUM::Int
    nDN::Int
    T::SparseMatrixCSC{K,Int}
    S::SparseMatrixCSC{K,Int}
    t_tN::Vector{Int}
    t_jN::Vector{Int}
    s_sM::Vector{Int}
    s_iM::Vector{Int}
    t_rows::_SparseRowPlan
    s_rows::_SparseRowPlan
    t_nzptr::Union{Nothing,Vector{Int}}
    s_nzptr::Union{Nothing,Vector{Int}}
    t_nzptr_max::Int
    s_nzptr_max::Int
    hcat_buf::SparseMatrixCSC{K,Int}
    hcat_buf_rev::SparseMatrixCSC{K,Int}
    nnzT::Int
    nnzS::Int
    union_red::FieldLinAlg._SparseREF{K}
    t_red::FieldLinAlg._SparseREF{K}
    s_red::FieldLinAlg._SparseREF{K}
    union_row::FieldLinAlg.SparseRow{K}
    side_row::FieldLinAlg.SparseRow{K}
    cached_rT::Int
    cached_rS::Int
    cached_T_valid::Bool
    cached_S_valid::Bool
end

mutable struct _FringePairCache{K}
    partner_id::UInt64
    partner_phi_id::UInt64
    layout_sketch::Union{Nothing,_FringeLayoutSketch}
    layout_plan::Union{Nothing,_FringeLayoutPlan}
    dense_idx_plan::Union{Nothing,_FringeDenseIdxPlan{K}}
    sparse_plan::Union{Nothing,_FringeSparsePlan{K}}
    route_choice::Union{Nothing,Symbol}
end

struct _FringeRouteChoiceEntry
    fingerprint::UInt64
    choice::Symbol
end

mutable struct _FringeHomCache{K}
    adj::Union{Nothing,_PackedAdjacency}
    upset::Union{Nothing,_FringeComponentDecomp}
    downset::Union{Nothing,_FringeComponentDecomp}
    pair_cache::Vector{_FringePairCache{K}}
    route_fingerprint_choice::Vector{_FringeRouteChoiceEntry}
    route_timing_fallbacks::Int
    _FringeHomCache{K}() where {K} = new(nothing, nothing, nothing,
                                         _FringePairCache{K}[],
                                         _FringeRouteChoiceEntry[],
                                         0)
end

struct _FiberQueryIndex
    col_ptr::Vector{Int}
    col_idx::Vector{Int}
    row_ptr::Vector{Int}
    row_idx::Vector{Int}
end

const FIBER_DIM_EAGER_INDEX_MAX_CELLS = Ref(65_536)
const FIBER_DIM_LAZY_FULL_INDEX_MIN_QUERIES = Ref(8)
const FIBER_DIM_LAZY_FULL_INDEX_MAX_QUERIES = Ref(64)
const HOM_PAIR_CACHE_MAX_ENTRIES = Ref(8)

@inline function _clear_sparse_side_rank_cache!(plan::_FringeSparsePlan)
    plan.cached_T_valid = false
    plan.cached_S_valid = false
    return plan
end


Base.length(U::Upset) = length(U.mask)
Base.length(D::Downset) = length(D.mask)
Base.eltype(::Type{<:Upset}) = Int
Base.eltype(::Type{<:Downset}) = Int
Base.IteratorSize(::Type{<:Upset}) = Base.HasLength()
Base.IteratorSize(::Type{<:Downset}) = Base.HasLength()

# Iterate over vertices contained in an upset/downset.
function Base.iterate(U::Upset, state::Int=1)
    n = length(U.mask)
    i = state
    @inbounds while i <= n
        if U.mask[i]
            return i, i + 1
        end
        i += 1
    end
    return nothing
end

function Base.iterate(D::Downset, state::Int=1)
    n = length(D.mask)
    i = state
    @inbounds while i <= n
        if D.mask[i]
            return i, i + 1
        end
        i += 1
    end
    return nothing
end

# Allocation-free bitset predicates (used heavily in matching / slicing code).
@inline function is_subset(a::BitVector, b::BitVector)
    @assert length(a) == length(b)
    ac = a.chunks
    bc = b.chunks
    nchunks = length(ac)
    lastmask = _tailmask(length(a))
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

@inline function intersects(a::BitVector, b::BitVector)
    @assert length(a) == length(b)
    ac = a.chunks
    bc = b.chunks
    nchunks = length(ac)
    lastmask = _tailmask(length(a))
    @inbounds for w in 1:nchunks
        v = ac[w] & bc[w]
        if w == nchunks
            v &= lastmask
        end
        if v != 0
            return true
        end
    end
    return false
end

is_subset(U1::Upset, U2::Upset) = is_subset(U1.mask, U2.mask)
is_subset(D1::Downset, D2::Downset) = is_subset(D1.mask, D2.mask)
intersects(U::Upset, D::Downset) = intersects(U.mask, D.mask)

"""
    upset_closure(P, S) -> Upset

Return the smallest upset of `P` containing the specified subset.

# Inputs

- `P::AbstractPoset`: ambient poset.
- `S::BitVector`: membership mask for the generating subset.

# Output

- The upward closure of `S` as an `Upset`.

# Best practices

- Use this when you already have a Boolean mask and want the mathematically
  canonical upset it generates.
- For generator lists, prefer `upset_from_generators`.
"""
function upset_closure(P::AbstractPoset, S::BitVector)
    U = copy(S)
    n = nvertices(P)
    for i in 1:n
        U[i] || continue
        for j in upset_indices(P, i)
            U[j] = true
        end
    end
    Upset(P, U)
end

"""
    downset_closure(P, S) -> Downset

Return the smallest downset of `P` containing the specified subset.

# Inputs

- `P::AbstractPoset`: ambient poset.
- `S::BitVector`: membership mask for the generating subset.

# Output

- The downward closure of `S` as a `Downset`.

# Best practices

- Use this when you already have a Boolean mask and want the mathematically
  canonical downset it generates.
- For generator lists, prefer `downset_from_generators`.
"""
function downset_closure(P::AbstractPoset, S::BitVector)
    D = copy(S)
    n = nvertices(P)
    for j in 1:n
        D[j] || continue
        for i in downset_indices(P, j)
            D[i] = true
        end
    end
    Downset(P, D)
end

"""
    upset_from_generators(P, gens) -> Upset
    upset_from_generators(P, gens_mask) -> Upset

Construct the upset generated by a list of vertices or by a Boolean generator
mask.

# Inputs

- `P::AbstractPoset`: ambient poset.
- `gens::AbstractVector{<:Integer}`: generator vertices.
- `gens_mask::AbstractVector{Bool}`: Boolean mask marking generators.

# Output

- The upward closure of the specified generators as an `Upset`.

# Failure / contract behavior

- Throws if a generator index is out of bounds.
- Throws if a generator mask length does not match `nvertices(P)`.

# Best practices

- Prefer this over manually constructing an upset when the data naturally comes
  as generators rather than a preclosed mask.
"""
function upset_from_generators(P::AbstractPoset, gens::AbstractVector{<:Integer})
    n = nvertices(P)
    mask = falses(n)
    @inbounds for g in gens
        gi = Int(g)
        (1 <= gi <= n) || error("upset_from_generators: generator index $gi out of bounds for poset with $n vertices.")
        mask[gi] = true
    end
    return upset_closure(P, mask)
end

"""
    downset_from_generators(P, gens) -> Downset
    downset_from_generators(P, gens_mask) -> Downset

Construct the downset generated by a list of vertices or by a Boolean generator
mask.

# Inputs

- `P::AbstractPoset`: ambient poset.
- `gens::AbstractVector{<:Integer}`: generator vertices.
- `gens_mask::AbstractVector{Bool}`: Boolean mask marking generators.

# Output

- The downward closure of the specified generators as a `Downset`.

# Failure / contract behavior

- Throws if a generator index is out of bounds.
- Throws if a generator mask length does not match `nvertices(P)`.

# Best practices

- Prefer this over manually constructing a downset when the data naturally comes
  as generators rather than a preclosed mask.
"""
function downset_from_generators(P::AbstractPoset, gens::AbstractVector{<:Integer})
    n = nvertices(P)
    mask = falses(n)
    @inbounds for g in gens
        gi = Int(g)
        (1 <= gi <= n) || error("downset_from_generators: generator index $gi out of bounds for poset with $n vertices.")
        mask[gi] = true
    end
    return downset_closure(P, mask)
end

function upset_from_generators(P::AbstractPoset, gens_mask::AbstractVector{Bool})
    n = nvertices(P)
    length(gens_mask) == n ||
        error("upset_from_generators: mask length $(length(gens_mask)) must equal nvertices(P)=$n.")
    return upset_closure(P, BitVector(gens_mask))
end

function downset_from_generators(P::AbstractPoset, gens_mask::AbstractVector{Bool})
    n = nvertices(P)
    length(gens_mask) == n ||
        error("downset_from_generators: mask length $(length(gens_mask)) must equal nvertices(P)=$n.")
    return downset_closure(P, BitVector(gens_mask))
end

"""
    principal_upset(P, p) -> Upset

Return the principal upset `upset_closurep = {q in P : p <= q}`.

# Inputs

- `P::AbstractPoset`: ambient poset.
- `p::Int`: distinguished vertex.

# Output

- The principal upset generated by `p`.

# Failure / contract behavior

- Relies on the underlying poset accessors; out-of-bounds vertex indices will
  raise a bounds error through the ambient iteration path.

# Best practices

- Use this as the canonical representable upset attached to a vertex.
"""
function principal_upset(P::AbstractPoset, p::Int)
    n = nvertices(P)
    mask = falses(n)
    for q in upset_indices(P, p)
        mask[q] = true
    end
    return Upset(P, BitVector(mask))
end

"""
    principal_downset(P, p) -> Downset

Return the principal downset `downset_closurep = {q in P : q <= p}`.

# Inputs

- `P::AbstractPoset`: ambient poset.
- `p::Int`: distinguished vertex.

# Output

- The principal downset generated by `p`.

# Failure / contract behavior

- Relies on the underlying poset accessors; out-of-bounds vertex indices will
  raise a bounds error through the ambient iteration path.

# Best practices

- Use this as the canonical corepresentable downset attached to a vertex.
"""
function principal_downset(P::AbstractPoset, p::Int)
    n = nvertices(P)
    mask = falses(n)
    for q in downset_indices(P, p)
        mask[q] = true
    end
    return Downset(P, BitVector(mask))
end

#############################
# Structural equality + hashing
#############################

import Base: ==, isequal, hash

# ---- FinitePoset ----
# Adjust field names (:n, :leq) to match your struct definition.
# From your printout it looks like FinitePoset(3, Bool[...]) so probably:
#   n::Int
#   leq::AbstractMatrix{Bool}   (or BitMatrix / Matrix{Bool})
==(P::FinitePoset, Q::FinitePoset) =
    (P.n == Q.n) && (P._leq == Q._leq)

isequal(P::FinitePoset, Q::FinitePoset) = (P == Q)

hash(P::FinitePoset, h::UInt) = hash(P.n, hash(P._leq, h))


# ---- Upset ----
# Adjust field names to match your Upset struct.
# From your printout: Upset(FinitePoset(...), Bool[...]) so likely:
#   P::FinitePoset
#   mem::AbstractVector{Bool}   (or BitVector)
==(U::Upset, V::Upset) =
    (U.P == V.P) && (U.mask == V.mask)

isequal(U::Upset, V::Upset) = (U == V)

hash(U::Upset, h::UInt) = hash(U.P, hash(U.mask, h))


# ---- Downset ----
# Same idea as Upset. Adjust field names.
==(D::Downset, E::Downset) =
    (D.P == E.P) && (D.mask == E.mask)

isequal(D::Downset, E::Downset) = (D == E)

hash(D::Downset, h::UInt) = hash(D.P, hash(D.mask, h))

"""
    check_upset(U; throw=false) -> NamedTuple

Validate a hand-built [`Upset`](@ref).

The returned report checks:
- the membership mask length matches `nvertices(U.P)`,
- the stored support is upward closed in the base poset.

Set `throw=true` to raise an `ArgumentError` instead of returning an invalid
report. For notebook and REPL inspection, prefer pairing this with
`fringe_summary(U)` or `describe(U)` on the shared summary surface.
"""
function check_upset(U::Upset; throw::Bool=false)
    issues = String[]
    n = nvertices(U.P)
    length(U.mask) == n || push!(issues, "mask length $(length(U.mask)) must equal nvertices(base poset)=$n")
    if isempty(issues)
        @inbounds for i in 1:n
            U.mask[i] || continue
            for j in upset_indices(U.P, i)
                if !U.mask[j]
                    push!(issues, "mask is not upward closed: $i is present but $j is missing")
                    break
                end
            end
            isempty(issues) || break
        end
    end
    report = (kind=:upset,
              valid=isempty(issues),
              nvertices=n,
              cardinality=count(U.mask),
              issues=issues)
    if throw && !report.valid
        Base.throw(ArgumentError("check_upset: invalid upset: " * join(report.issues, "; ")))
    end
    return report
end

"""
    check_downset(D; throw=false) -> NamedTuple

Validate a hand-built [`Downset`](@ref).

The returned report checks:
- the membership mask length matches `nvertices(D.P)`,
- the stored support is downward closed in the base poset.

Set `throw=true` to raise an `ArgumentError` instead of returning an invalid
report. For notebook and REPL inspection, prefer pairing this with
`fringe_summary(D)` or `describe(D)`.
"""
function check_downset(D::Downset; throw::Bool=false)
    issues = String[]
    n = nvertices(D.P)
    length(D.mask) == n || push!(issues, "mask length $(length(D.mask)) must equal nvertices(base poset)=$n")
    if isempty(issues)
        @inbounds for j in 1:n
            D.mask[j] || continue
            for i in downset_indices(D.P, j)
                if !D.mask[i]
                    push!(issues, "mask is not downward closed: $j is present but $i is missing")
                    break
                end
            end
            isempty(issues) || break
        end
    end
    report = (kind=:downset,
              valid=isempty(issues),
              nvertices=n,
              cardinality=count(D.mask),
              issues=issues)
    if throw && !report.valid
        Base.throw(ArgumentError("check_downset: invalid downset: " * join(report.issues, "; ")))
    end
    return report
end

function Base.show(io::IO, U::Upset)
    d = _fringe_describe(U)
    print(io, "Upset(|U|=", d.cardinality,
          ", nvertices=", d.nvertices, ")")
end

function Base.show(io::IO, ::MIME"text/plain", U::Upset)
    d = _fringe_describe(U)
    print(io, "Upset\n  cardinality: ", d.cardinality,
          "\n  nvertices: ", d.nvertices,
          "\n  ", _support_preview(d.support))
end

function Base.show(io::IO, D::Downset)
    d = _fringe_describe(D)
    print(io, "Downset(|D|=", d.cardinality,
          ", nvertices=", d.nvertices, ")")
end

function Base.show(io::IO, ::MIME"text/plain", D::Downset)
    d = _fringe_describe(D)
    print(io, "Downset\n  cardinality: ", d.cardinality,
          "\n  nvertices: ", d.nvertices,
          "\n  ", _support_preview(d.support))
end


# =========================================
# Fringe presentations (Defs. 3.16 - 3.17)
# =========================================
