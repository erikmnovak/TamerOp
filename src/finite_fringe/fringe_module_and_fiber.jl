# finite_fringe/fringe_module_and_fiber.jl
# Scope: FringeModule construction, field coercion/change_field, monomial checks, and fiber-dimension queries.

"""
    FringeModule{K,P,F,MAT}

Finite-fringe presentation of a module over a finite poset.

# Mathematical meaning

A `FringeModule` encodes a module by a finite presentation consisting of:

- birth upsets `U_1, ..., U_n`,
- death downsets `D_1, ..., D_m`,
- a coefficient matrix `phi` of size `m x n` over a ground field `k`.

The intended interpretation is that columns represent generators born on the
upsets, rows represent relations or cogenerators supported on the downsets, and
the matrix `phi` records the presentation coefficients.

# Ambient data

- `P`: the ambient finite poset.
- `U`: the list of birth upsets, one per generator (matrix columns).
- `D`: the list of death downsets, one per relation (matrix rows).

# Stored representation

- `field`: the coefficient field object.
- `phi`: the `m x n` presentation matrix.
- `phi_density`: cached density summary for routing heuristics.
- `fiber_index`, `fiber_queries`, `fiber_dims`: lazy/cached fiber-dimension data.
- `hom_cache`: lazy/cached `Hom` planning and route data.

# Invariants

- `size(phi, 1) == length(D)` and `size(phi, 2) == length(U)`.
- Every upset/downset lives over the same ambient poset `P`.
- `coeff_type(field) == K`.
- Nonzero entries satisfy the support condition `U_i  intersect  D_j != empty`.

# Best practices

- Use the public constructor `FringeModule{K}(P, U, D, phi; field=...)` rather
  than constructing this parametric type directly.
- Use `one_by_one_fringe(...)` for simple interval-like examples and tests.
- Use semantic accessors such as `birth_upsets`, `death_downsets`,
  `fringe_coefficients`, `fiber_dimension`, and `hom_dimension` rather than
  inspecting cache fields directly.
- `Upset` and `Downset` inputs are expected to be already closed. If you start
  from generators, use `upset_from_generators` and `downset_from_generators`
  first.
"""
struct FringeModule{K, P<:AbstractPoset, F<:AbstractCoeffField, MAT<:AbstractMatrix{K}}
    field::F
    P::P
    U::Vector{Upset{P}}               # birth upsets (columns)
    D::Vector{Downset{P}}             # death downsets (rows)
    phi::MAT                          # size |D| x |U|
    phi_density::Float64
    fiber_index::Base.RefValue{Union{Nothing,_FiberQueryIndex}}
    fiber_queries::Base.RefValue{Int}
    fiber_dims::Base.RefValue{Union{Nothing,Vector{Int}}}
    hom_cache::Base.RefValue{_FringeHomCache{K}}

    function FringeModule{K,P,F,MAT}(field::F,
                                     Pobj::P,
                                     U::Vector{Upset{P}},
                                     D::Vector{Downset{P}},
                                     phi::MAT) where {K, P<:AbstractPoset, F<:AbstractCoeffField, MAT<:AbstractMatrix{K}}
        @assert size(phi,1) == length(D) && size(phi,2) == length(U)
        coeff_type(field) == K || error("FringeModule: coeff_type(field) != K")

        idx = _should_build_fiber_query_index(Pobj, U, D) ?
            _build_fiber_query_index(Pobj, U, D) : nothing

        M = new{K,P,F,MAT}(field, Pobj, U, D, phi, _matrix_density(phi),
                         Ref{Union{Nothing,_FiberQueryIndex}}(idx),
                         Ref{Int}(0),
                         Ref{Union{Nothing,Vector{Int}}}(nothing),
                         Ref(_FringeHomCache{K}()))
        _check_monomial_condition(M)
        return M
    end
end

"""
    base_poset(x) -> AbstractPoset

Return the base poset underlying an upset, downset, or fringe module.

Use this accessor instead of field inspection (`x.P`) when writing examples,
notebooks, and validation code.
"""
@inline base_poset(U::Upset) = U.P
@inline base_poset(D::Downset) = D.P
@inline base_poset(M::FringeModule) = M.P

"""
    ambient_poset(x) -> AbstractPoset

Return the ambient poset of a finite-fringe object.

This is the canonical semantic accessor for the ambient finite poset of:

- a finite poset itself,
- an upset,
- a downset,
- a fringe module.

`ambient_poset` is intended for user-facing code and notebooks; internally,
`base_poset` remains the owner-local synonym.
"""
@inline ambient_poset(P::AbstractPoset) = P
@inline ambient_poset(U::Upset) = U.P
@inline ambient_poset(D::Downset) = D.P
@inline ambient_poset(M::FringeModule) = M.P

"""
    field(M::FringeModule) -> AbstractCoeffField

Return the coefficient field of a fringe module.

This is the canonical semantic accessor for the scalar field of a
`FringeModule`; prefer it over direct field access in user-facing code.
"""
@inline field(M::FringeModule) = M.field

"""
    birth_upsets(M::FringeModule) -> Vector{Upset}

Return the ordered list of birth upsets (generator supports) of `M`.

These correspond to the columns of `fringe_coefficients(M)`.
"""
@inline birth_upsets(M::FringeModule) = M.U

"""
    death_downsets(M::FringeModule) -> Vector{Downset}

Return the ordered list of death downsets (relation supports) of `M`.

These correspond to the rows of `fringe_coefficients(M)`.
"""
@inline death_downsets(M::FringeModule) = M.D

"""
    fringe_coefficients(M::FringeModule) -> AbstractMatrix

Return the coefficient matrix of the finite-fringe presentation.

This is the semantic accessor for the presentation matrix; prefer it over
direct `M.phi` field access in user-facing code.
"""
@inline fringe_coefficients(M::FringeModule) = M.phi

"""
    ngenerators(M::FringeModule) -> Int
    nrelations(M::FringeModule) -> Int

Return the number of birth generators and death relations in the presentation
of `M`.
"""
@inline ngenerators(M::FringeModule) = size(M.phi, 2)
@inline nrelations(M::FringeModule) = size(M.phi, 1)

@inline function _fringe_describe(M::FringeModule)
    return (kind=:fringe_module,
            field=M.field,
            poset_kind=_fringe_describe(M.P).kind,
            nvertices=nvertices(M.P),
            ngenerators=ngenerators(M),
            nrelations=nrelations(M),
            matrix_size=size(M.phi),
            phi_density=M.phi_density)
end

@inline function _fringe_field_label(field::AbstractCoeffField)
    field isa QQField && return "QQ"
    return string(nameof(typeof(field)))
end

"""
    dimensions(M::FringeModule) -> NamedTuple
    dimensions(M::FringeModule, q::Int) -> NamedTuple

Return fiber-dimension summaries for a fringe module.

`dimensions(M)` returns a compact summary of all pointwise fiber dimensions,
while `dimensions(M, q)` returns the dimension of the fiber at the specified
vertex.

The summary fields are:

- `vertices`: number of ambient vertices,
- `fibers`: vector of pointwise fiber dimensions,
- `total`: sum of the fiber dimensions,
- `maximum_fiber`: maximum pointwise fiber dimension,
- `ngenerators`, `nrelations`: presentation sizes.

For the single-vertex form, the fields are:

- `vertex`
- `fiber`

Best practices
- Use `dimensions(M)` for a semantic overview in notebooks and the REPL.
- Use `fiber_dimension(M, q)` when you only need one vertex and want the most
  direct query surface.
"""
@inline function _fringe_dimensions(M::FringeModule)
    dims = _ensure_fiber_dim_cache!(M)
    @inbounds for q in 1:nvertices(M.P)
        dims[q] == typemin(Int) || continue
        dims[q] = fiber_dimension(M, q)
    end
    return (
        vertices = nvertices(M.P),
        fibers = copy(dims),
        total = sum(dims),
        maximum_fiber = isempty(dims) ? 0 : maximum(dims),
        ngenerators = ngenerators(M),
        nrelations = nrelations(M),
    )
end

@inline function _fringe_dimensions(M::FringeModule, q::Int)
    return (
        vertex = q,
        fiber = fiber_dimension(M, q),
    )
end

const _FRINGE_SHOW_FIBER_SUMMARY_MAX_VERTICES = 12

@inline function _fringe_show_fiber_summary(M::FringeModule)
    n = nvertices(M.P)
    dims = M.fiber_dims[]
    if dims !== nothing
        known = count(!=(typemin(Int)), dims)
        if known == n
            vals = copy(dims)
            return "fibers=$(repr(vals)), total=$(sum(vals)), max=$(isempty(vals) ? 0 : maximum(vals))"
        elseif known > 0
            return "fiber_cache=$known/$n known (use dimensions(M) for full summary)"
        end
    end
    if n <= _FRINGE_SHOW_FIBER_SUMMARY_MAX_VERTICES
        d = _fringe_dimensions(M)
        return "fibers=$(repr(d.fibers)), total=$(d.total), max=$(d.maximum_fiber)"
    end
    return "fiber_summary=lazy (use dimensions(M))"
end

# ----------------------------
# Field coercion helpers
# ----------------------------

function _coerce_matrix(field::AbstractCoeffField, A::AbstractMatrix{K}) where {K}
    K2 = coeff_type(field)
    if A isa SparseMatrixCSC{K,Int}
        S = spzeros(K2, size(A, 1), size(A, 2))
        @inbounds for j in 1:size(A, 2)
            for idx in A.colptr[j]:(A.colptr[j + 1] - 1)
                i = A.rowval[idx]
                S[i, j] = coerce(field, A.nzval[idx])
            end
        end
        return S
    end

    M = Matrix{K2}(undef, size(A, 1), size(A, 2))
    @inbounds for j in 1:size(A, 2), i in 1:size(A, 1)
        M[i, j] = coerce(field, A[i, j])
    end
    return M
end

"""
    change_field(H, field)

Return a FringeModule obtained by coercing `phi` into `field`.
"""
function change_field(H::FringeModule{K}, field::AbstractCoeffField) where {K}
    K2 = coeff_type(field)
    Phi = _coerce_matrix(field, H.phi)
    return FringeModule{K2, typeof(H.P), typeof(field), typeof(Phi)}(field, H.P, H.U, H.D, Phi)
end

"""
    FiniteFringeValidationSummary

Pretty printable wrapper for reports returned by the `FiniteFringe` validation
helpers.

Use [`finite_fringe_validation_summary`](@ref) to turn a raw report from
`check_poset`, `check_upset`, `check_downset`, `check_fringe_data`, or
`check_fringe_module` into a notebook/REPL-friendly summary object.
"""
struct FiniteFringeValidationSummary{R}
    report::R
end

"""
    finite_fringe_validation_summary(report) -> FiniteFringeValidationSummary

Wrap a raw `FiniteFringe` validation report in a display-oriented summary
object.

Best practices
- call the relevant `check_*` helper first to obtain the structural report;
- wrap it with `finite_fringe_validation_summary(...)` for notebook or REPL
  inspection;
- keep the raw report when you need programmatic access to the `issues` vector.
"""
@inline finite_fringe_validation_summary(report::NamedTuple) = FiniteFringeValidationSummary(report)

"""
    fringe_summary(x) -> NamedTuple

Owner-local summary entrypoint for finite-fringe objects.

This mirrors the shared `describe(...)` surface without requiring users to know
that the generic is owned by another subsystem. Supported inputs currently
include finite-fringe posets, upsets, downsets, and fringe modules.
"""
fringe_summary(P::AbstractPoset) = _fringe_describe(P)
fringe_summary(U::Upset) = _fringe_describe(U)
fringe_summary(D::Downset) = _fringe_describe(D)
fringe_summary(M::FringeModule) = _fringe_describe(M)

"""
    check_fringe_data(P, U, D, phi; field=nothing, throw=false) -> NamedTuple

Validate hand-built data intended for [`FringeModule`](@ref) construction.

The returned report checks:
- every upset/downset belongs to the same base poset `P`,
- the matrix shape matches `length(D) x length(U)`,
- the individual upsets/downsets are structurally valid,
- `coeff_type(field)` matches `eltype(phi)` when `field` is supplied,
- every nonzero entry of `phi` satisfies the monomial support condition
  `U[i] intersect D[j] != emptyset`.

Use this before constructing a fringe presentation by hand when you want a
diagnostic report instead of the constructor failing on the first violation.
Set `throw=true` to raise an `ArgumentError` on invalid input.
Wrap the returned report with [`finite_fringe_validation_summary`](@ref) when
you want a compact notebook/REPL view.
"""
function check_fringe_data(P::AbstractPoset,
                           Uin::AbstractVector{<:Upset},
                           Din::AbstractVector{<:Downset},
                           phi::AbstractMatrix;
                           field::Union{Nothing,AbstractCoeffField}=nothing,
                           throw::Bool=false)
    issues = String[]
    poset_report = check_poset(P)
    poset_report.valid || append!(issues, "base poset: " .* poset_report.issues)

    m, n = size(phi)
    m == length(Din) || push!(issues, "matrix row count $m must equal number of downsets $(length(Din))")
    n == length(Uin) || push!(issues, "matrix column count $n must equal number of upsets $(length(Uin))")

    field === nothing || coeff_type(field) == eltype(phi) ||
        push!(issues, "coeff_type(field) must match eltype(phi)")

    @inbounds for (i, U) in enumerate(Uin)
        U.P === P || push!(issues, "upset $i does not belong to the supplied base poset")
        rep = check_upset(U)
        rep.valid || append!(issues, "upset $i: " .* rep.issues)
    end
    @inbounds for (j, D) in enumerate(Din)
        D.P === P || push!(issues, "downset $j does not belong to the supplied base poset")
        rep = check_downset(D)
        rep.valid || append!(issues, "downset $j: " .* rep.issues)
    end

    if m == length(Din) && n == length(Uin)
        if phi isa SparseMatrixCSC
            @inbounds for col in 1:size(phi, 2)
                for ptr in phi.colptr[col]:(phi.colptr[col + 1] - 1)
                    row = phi.rowval[ptr]
                    iszero(phi.nzval[ptr]) && continue
                    intersects(Uin[col], Din[row]) || push!(issues, "nonzero phi[$row,$col] requires U[$col] intersect D[$row] != emptyset")
                end
            end
        else
            @inbounds for row in 1:m, col in 1:n
                iszero(phi[row, col]) && continue
                intersects(Uin[col], Din[row]) || push!(issues, "nonzero phi[$row,$col] requires U[$col] intersect D[$row] != emptyset")
            end
        end
    end

    report = (kind=:fringe_data,
              valid=isempty(issues),
              nvertices=nvertices(P),
              ngenerators=length(Uin),
              nrelations=length(Din),
              matrix_size=size(phi),
              issues=issues)
    if throw && !report.valid
        Base.throw(ArgumentError("check_fringe_data: invalid fringe data: " * join(report.issues, "; ")))
    end
    return report
end

"""
    check_fringe_module(M; throw=false) -> NamedTuple

Validate an existing [`FringeModule`](@ref).

The returned report checks the base-poset contract, all upset/downset supports,
matrix shape, coefficient-field compatibility, and the monomial support
condition on the stored matrix entries.

Set `throw=true` to raise an `ArgumentError` instead of returning an invalid
report. Wrap the returned report with [`finite_fringe_validation_summary`](@ref)
for notebook/REPL inspection and pair it with [`fringe_summary`](@ref) when you
also want the compact presentation summary of `M`.
"""
function check_fringe_module(M::FringeModule; throw::Bool=false)
    report0 = check_fringe_data(M.P, M.U, M.D, M.phi; field=M.field)
    issues = copy(report0.issues)
    report = (kind=:fringe_module,
              valid=isempty(issues),
              nvertices=report0.nvertices,
              ngenerators=report0.ngenerators,
              nrelations=report0.nrelations,
              matrix_size=report0.matrix_size,
              phi_density=M.phi_density,
              issues=issues)
    if throw && !report.valid
        Base.throw(ArgumentError("check_fringe_module: invalid FringeModule: " * join(report.issues, "; ")))
    end
    return report
end

# Canonical public constructor: infer matrix storage from `phi`, require explicit `field`.
@inline function _coerce_upsets(P::PT, Uin::AbstractVector{<:Upset}) where {PT<:AbstractPoset}
    U = Vector{Upset{PT}}(undef, length(Uin))
    @inbounds for i in eachindex(Uin)
        Ui = Uin[i]
        Ui.P === P || error("FringeModule: all upsets must belong to the same poset object P.")
        U[i] = Ui::Upset{PT}
    end
    return U
end

@inline function _coerce_downsets(P::PT, Din::AbstractVector{<:Downset}) where {PT<:AbstractPoset}
    D = Vector{Downset{PT}}(undef, length(Din))
    @inbounds for i in eachindex(Din)
        Di = Din[i]
        Di.P === P || error("FringeModule: all downsets must belong to the same poset object P.")
        D[i] = Di::Downset{PT}
    end
    return D
end

"""
    FringeModule{K}(P, U, D, phi; field) -> FringeModule{K}

Construct a finite-fringe module over the coefficient field `field`.

# Inputs

- `P::AbstractPoset`: ambient finite poset.
- `U`: vector of birth upsets.
- `D`: vector of death downsets.
- `phi::AbstractMatrix{K}`: coefficient matrix with rows indexed by `D` and
  columns indexed by `U`.
- `field::AbstractCoeffField`: explicit coefficient field. This keyword is
  required on the canonical constructor so the field contract stays explicit.

# Output

- A `FringeModule{K}` over the ambient poset `P`.

# Domain/codomain conventions

- Columns of `phi` correspond to generators supported on the upsets in `U`.
- Rows of `phi` correspond to relations supported on the downsets in `D`.

# Failure / contract behavior

- Throws if the upset/downset objects do not live over the same ambient poset.
- Throws if `size(phi)` does not match `length(D) x length(U)`.
- Throws if the coefficient type of `field` does not equal `K`.
- Throws if the support condition for nonzero entries fails.

# Best practices

- Pass an explicit `field=` keyword even when `K` already determines the scalar
  type; this keeps examples mathematically explicit and avoids hidden defaults.
- This constructor expects already-closed `Upset` and `Downset` objects. It
  does not close generator sets for you.
- There are no legacy aliases and no hidden encode-vs-pmodule bridges on this
  constructor path; it directly builds a fringe module from the supplied
  presentation data.
- For hand-built data, run `check_fringe_module(M)` after construction if you
  want a structured validation report.

# Examples

```jldoctest
julia> using TamerOp

julia> const FF = TamerOp.FiniteFringe
Main.TamerOp.FiniteFringe

julia> P = FF.FinitePoset(Bool[1 1 1;
                              0 1 1;
                              0 0 1]);

julia> U = [FF.principal_upset(P, 2)];

julia> D = [FF.principal_downset(P, 2)];

julia> phi = reshape([QQ(1)], 1, 1);

julia> M = FF.FringeModule{QQ}(P, U, D, phi; field=QQField())
FringeModule(field=QQ, nvertices=3, ngenerators=1, nrelations=1)

julia> dimensions(M).fibers
[0, 1, 0]
```
"""
function FringeModule{K}(P::PT,
                         Uin::AbstractVector{<:Upset},
                         Din::AbstractVector{<:Downset},
                         phi::AbstractMatrix{K};
                         field::AbstractCoeffField) where {K,PT<:AbstractPoset}
    F = typeof(field)
    U = _coerce_upsets(P, Uin)
    D = _coerce_downsets(P, Din)
    return FringeModule{K, PT, F, typeof(phi)}(field, P, U, D, phi)
end

"""
    one_by_one_fringe(P, U, D, scalar; field=QQField()) -> FringeModule
    one_by_one_fringe(P, U_mask, D_mask, scalar; field=QQField()) -> FringeModule

Construct the simplest nontrivial finite-fringe module: a `1 x 1` presentation.

# Mathematical meaning

This is the fringe module with:

- one generator supported on the upset `U`,
- one relation supported on the downset `D`,
- one presentation coefficient `scalar`.

Equivalently, the presentation matrix is the `1 x 1` matrix `[scalar]`.

# Inputs

- `P::AbstractPoset`: ambient poset.
- `U::Upset` or `U_mask::AbstractVector{Bool}`: generator support.
- `D::Downset` or `D_mask::AbstractVector{Bool}`: relation support.
- `scalar`: the unique matrix entry.
- `field::AbstractCoeffField=QQField()`: coefficient field used to coerce the
  scalar and construct the matrix.

# Output

- A `FringeModule` with one birth upset, one death downset, and a `1 x 1`
  structure matrix.

# Failure / contract behavior

- Throws if mask lengths do not match the ambient finite poset size.
- Throws if the upset/downset data does not match the ambient poset.
- Throws if the scalar cannot be coerced into the requested field.

# Best practices

- Use this for tests, examples, notebook exploration, and interval-like toy
  modules.
- Pass the scalar explicitly; if you want the identity coefficient, use
  `one(coeff_type(field))` or an explicit field element.
- Prefer this constructor over hand-building a `1 x 1` sparse matrix.
- Both the `Upset`/`Downset` form and the mask form expect already-closed
  supports. If you start from generators, use `principal_upset`,
  `principal_downset`, `upset_from_generators`, or `downset_from_generators`
  first.
- There are no legacy aliases and no hidden encode-vs-pmodule bridges here;
  this constructor directly produces a fringe module.

# Examples

```jldoctest
julia> using TamerOp

julia> const FF = TamerOp.FiniteFringe
Main.TamerOp.FiniteFringe

julia> P = FF.FinitePoset(Bool[1 1 1;
                              0 1 1;
                              0 0 1]);

julia> U = FF.principal_upset(P, 2);

julia> D = FF.principal_downset(P, 2);

julia> M = FF.one_by_one_fringe(P, U, D, 1; field=QQField())
FringeModule(field=QQ, nvertices=3, ngenerators=1, nrelations=1)

julia> dimensions(M).fibers
[0, 1, 0]
```
"""
function one_by_one_fringe(P::AbstractPoset, U::Upset, D::Downset, scalar::K;
                           field::AbstractCoeffField=QQField()) where {K}
    s = coerce(field, scalar)
    phi = spzeros(typeof(s), 1, 1)
    phi[1, 1] = s
    return FringeModule{typeof(s)}(P, [U], [D], phi; field=field)
end

# ------------------ mask-based convenience overloads ------------------

function _coerce_bool_mask(P::FinitePoset, mask::AbstractVector{Bool}, name::AbstractString)::BitVector
    if length(mask) != P.n
        error(name * " mask must have length P.n=" * string(P.n) *
              "; got length " * string(length(mask)) * ".")
    end
    return (mask isa BitVector) ? mask : BitVector(mask)
end

function one_by_one_fringe(P::FinitePoset,
                           U_mask::AbstractVector{Bool},
                           D_mask::AbstractVector{Bool},
                           scalar::K;
                           field::AbstractCoeffField=QQField()) where {K}
    Um = _coerce_bool_mask(P, U_mask, "Upset")
    Dm = _coerce_bool_mask(P, D_mask, "Downset")
    return one_by_one_fringe(P, Upset(P, Um), Downset(P, Dm), coerce(field, scalar); field=field)
end




# Prop. 3.18: Nonzero entry only if U_i \cap D_j \neq \emptyset.
function _check_monomial_condition(M::FringeModule{K}) where {K}
    m, n = size(M.phi)
    @assert m == length(M.D) && n == length(M.U) "Dimension mismatch"

    phi = M.phi
    if phi isa SparseMatrixCSC
        I, J, V = findnz(phi)
        for t in eachindex(V)
            v = V[t]
            if v != zero(K)
                j = I[t]; i = J[t]
                @assert intersects(M.U[i], M.D[j]) "Nonzero phi[j,i] requires U[i] cap D[j] neq emptyset (Prop. 3.18)"
            end
        end
    else
        for j in 1:m, i in 1:n
            v = phi[j,i]
            if v != zero(K)
                @assert intersects(M.U[i], M.D[j]) "Nonzero phi[j,i] requires U[i] cap D[j] neq emptyset (Prop. 3.18)"
            end
        end
    end
end

function Base.show(io::IO, M::FringeModule)
    d = _fringe_describe(M)
    print(io, "FringeModule(field=", _fringe_field_label(d.field),
          ", nvertices=", d.nvertices,
          ", ngenerators=", d.ngenerators,
          ", nrelations=", d.nrelations, ")")
end

function Base.show(io::IO, ::MIME"text/plain", M::FringeModule)
    d = _fringe_describe(M)
    print(io, "FringeModule\n  field: ", _fringe_field_label(d.field),
          "\n  poset_kind: ", d.poset_kind,
          "\n  nvertices: ", d.nvertices,
          "\n  ngenerators: ", d.ngenerators,
          "\n  nrelations: ", d.nrelations,
          "\n  matrix_size: ", repr(d.matrix_size),
          "\n  phi_density: ", d.phi_density,
          "\n  ", _fringe_show_fiber_summary(M))
end

function Base.show(io::IO, summary::FiniteFringeValidationSummary)
    r = summary.report
    print(io, "FiniteFringeValidationSummary(kind=", r.kind,
          ", valid=", r.valid,
          ", issues=", length(r.issues), ")")
end

function Base.show(io::IO, ::MIME"text/plain", summary::FiniteFringeValidationSummary)
    r = summary.report
    println(io, "FiniteFringeValidationSummary")
    println(io, "  kind: ", r.kind)
    println(io, "  valid: ", r.valid)
    for key in propertynames(r)
        key in (:kind, :valid, :issues) && continue
        println(io, "  ", key, ": ", repr(getproperty(r, key)))
    end
    if isempty(r.issues)
        print(io, "  issues: []")
    else
        println(io, "  issues:")
        for issue in r.issues
            println(io, "    - ", issue)
        end
    end
end


# ------------------ evaluation (degreewise image; after Def. 3.17) ------------------


"""
    fiber_dimension(M, q) -> Int

Return the fiber dimension of a fringe module at a vertex of the ambient poset.

# Mathematical meaning

For a fringe module `M` over a field `k` and a vertex `q` of the ambient poset,
this computes

```math
\\dim_k M_q,
```

equivalently the rank of the degreewise presentation matrix obtained by
restricting to generators and relations active at `q`.

# Inputs

- `M::FringeModule`: fringe module over a finite poset.
- `q::Int`: ambient vertex index.

# Output

- An `Int` equal to the vector-space dimension of the fiber at `q`.

# Failure / contract behavior

- Throws if `q` is out of bounds for the ambient poset.
- Relies on the module satisfying the usual finite-fringe presentation
  invariants.

# Performance notes

- The implementation uses lazy fiber-query indexing and cached restricted-rank
  computations internally.
- Repeated calls on the same module automatically benefit from those caches; no
  special user action is required.

# Best practices

- Use this as the canonical cheap summary of a fringe module at a vertex.
- If you need values at many vertices, it is still fine to call
  `fiber_dimension(M, q)` repeatedly; the subsystem is optimized for that usage.
- `fiber_dimension(M, q)` returns the vector-space dimension of the fiber at the
  single vertex `q`; use `dimensions(M)` when you want the full pointwise
  dimension profile across the ambient poset.

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

julia> FF.fiber_dimension(M, 2)
1

julia> FF.fiber_dimension(M, 1)
0
```
"""
@inline function _should_build_fiber_query_index(P::AbstractPoset,
                                                 U::AbstractVector,
                                                 D::AbstractVector)
    return nvertices(P) * (length(U) + length(D)) <= FIBER_DIM_EAGER_INDEX_MAX_CELLS[]
end

function _build_fiber_query_index(P::AbstractPoset,
                                  Usets::AbstractVector,
                                  Dsets::AbstractVector)
    n = nvertices(P)
    col_counts = zeros(Int, n)
    row_counts = zeros(Int, n)

    @inbounds for U in Usets
        _foreach_setbit(U.mask) do q
            col_counts[q] += 1
        end
    end

    @inbounds for D in Dsets
        _foreach_setbit(D.mask) do q
            row_counts[q] += 1
        end
    end

    col_ptr = Vector{Int}(undef, n + 1)
    row_ptr = Vector{Int}(undef, n + 1)
    col_ptr[1] = 1
    row_ptr[1] = 1
    @inbounds for q in 1:n
        col_ptr[q + 1] = col_ptr[q] + col_counts[q]
        row_ptr[q + 1] = row_ptr[q] + row_counts[q]
    end

    col_idx = Vector{Int}(undef, col_ptr[end] - 1)
    row_idx = Vector{Int}(undef, row_ptr[end] - 1)
    col_next = copy(col_ptr)
    row_next = copy(row_ptr)

    @inbounds for (i, U) in enumerate(Usets)
        _foreach_setbit(U.mask) do q
            col_idx[col_next[q]] = i
            col_next[q] += 1
        end
    end

    @inbounds for (j, D) in enumerate(Dsets)
        _foreach_setbit(D.mask) do q
            row_idx[row_next[q]] = j
            row_next[q] += 1
        end
    end

    return _FiberQueryIndex(col_ptr, col_idx, row_ptr, row_idx)
end

function _build_fiber_query_index(M::FringeModule)
    return _build_fiber_query_index(M.P, M.U, M.D)
end

function _build_fiber_query_slice(Usets::AbstractVector,
                                  Dsets::AbstractVector,
                                  q::Int)
    ncols = 0
    @inbounds for U in Usets
        ncols += U.mask[q] ? 1 : 0
    end
    cols = Vector{Int}(undef, ncols)
    next = 1
    @inbounds for i in eachindex(Usets)
        Usets[i].mask[q] || continue
        cols[next] = i
        next += 1
    end

    nrows = 0
    @inbounds for D in Dsets
        nrows += D.mask[q] ? 1 : 0
    end
    rows = Vector{Int}(undef, nrows)
    next = 1
    @inbounds for j in eachindex(Dsets)
        Dsets[j].mask[q] || continue
        rows[next] = j
        next += 1
    end

    return rows, cols
end

function _build_fiber_query_slice_words(Usets::AbstractVector,
                                        Dsets::AbstractVector,
                                        q::Int)
    col_words = zeros(UInt64, cld(length(Usets), 64))
    row_words = zeros(UInt64, cld(length(Dsets), 64))
    nc = 0
    nr = 0
    @inbounds for i in eachindex(Usets)
        Usets[i].mask[q] || continue
        col_words[((i - 1) >>> 6) + 1] |= UInt64(1) << ((i - 1) & 63)
        nc += 1
    end
    @inbounds for j in eachindex(Dsets)
        Dsets[j].mask[q] || continue
        row_words[((j - 1) >>> 6) + 1] |= UInt64(1) << ((j - 1) & 63)
        nr += 1
    end
    return row_words, col_words, nr, nc
end

@inline function _fiber_lazy_full_index_after(M::FringeModule)
    n = nvertices(M.P)
    return max(FIBER_DIM_LAZY_FULL_INDEX_MIN_QUERIES[],
               min(FIBER_DIM_LAZY_FULL_INDEX_MAX_QUERIES[], cld(n, 4)))
end

@inline function _should_materialize_full_fiber_index(M::FringeModule)
    return M.fiber_queries[] >= _fiber_lazy_full_index_after(M)
end

@inline function _ensure_fiber_query_index!(M::FringeModule)
    idx = M.fiber_index[]
    idx !== nothing && return idx
    idx = _build_fiber_query_index(M)
    M.fiber_index[] = idx
    return idx
end

@inline function _ensure_fiber_dim_cache!(M::FringeModule)
    dims = M.fiber_dims[]
    if dims === nothing
        dims = fill(typemin(Int), nvertices(M.P))
        M.fiber_dims[] = dims
    end
    return dims
end

function fiber_dimension(M::FringeModule{K}, q::Int) where {K}
    dims = _ensure_fiber_dim_cache!(M)
    cached = dims[q]
    cached != typemin(Int) && return cached

    idx = M.fiber_index[]
    rows = cols = nothing
    row_words = col_words = nothing
    nr = nc = 0
    if idx === nothing
        M.fiber_queries[] += 1
        if _should_materialize_full_fiber_index(M)
            idx = _ensure_fiber_query_index!(M)
        else
            row_words, col_words, nr, nc = _build_fiber_query_slice_words(M.U, M.D, q)
        end
    end

    if idx === nothing
        if nr == 0 || nc == 0
            dims[q] = 0
            return 0
        end
        d = FieldLinAlg.rank_restricted_words(M.field, M.phi,
                                              row_words, col_words, nr, nc;
                                              nrows=size(M.phi, 1),
                                              ncols=size(M.phi, 2))
        dims[q] = d
        return d
    end

    clo, chi = idx.col_ptr[q], idx.col_ptr[q + 1] - 1
    rlo, rhi = idx.row_ptr[q], idx.row_ptr[q + 1] - 1
    if clo > chi || rlo > rhi
        dims[q] = 0
        return 0
    end
    cols = @view idx.col_idx[clo:chi]
    rows = @view idx.row_idx[rlo:rhi]
    d = FieldLinAlg.rank_restricted(M.field, M.phi, rows, cols)
    dims[q] = d
    return d
end
