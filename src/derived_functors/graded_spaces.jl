# graded_spaces.jl -- graded-space interface and cross-subsystem bridges


"""
GradedSpaces: a small internal interface for "graded vector space-like" derived objects.

This module defines a shared set of generic functions:

- degree_range(space)
- dim(space, t)
- basis(space, t)
- representative(space, t, coords)
- coordinates(space, t, cocycle)

Optional (implemented when meaningful):

- cycles(space, t)
- boundaries(space, t)

Design notes:

- The interface is intentionally minimal.
- The coordinate convention is that `coords` is expressed in the basis returned by `basis(space, t)`.
- Degrees are integers. Most objects live in nonnegative degrees, but the interface permits negative
  degrees (useful for HyperTor style indexing).

Concrete derived objects must `import ..GradedSpaces: ...` and then define methods on these
shared function objects.
"""
module GradedSpaces

    """
        degree_range(space) -> UnitRange{Int}

    Return the range of degrees for which the graded object `space` has been computed and stored.
    """
    function degree_range end

    """
        dim(space, t::Integer) -> Int

    Return the dimension of the graded component of `space` in degree `t`.
    Must be consistent with `basis(space, t)`.
    """
    function dim end

    """
        basis(space, t::Integer)

    Return a basis for the graded component of `space` in degree `t`.

    The container type is not fixed, but must be consistent with the coordinate convention used by
    `representative` and `coordinates`.
    """
    function basis end

    """
        representative(space, t::Integer, coords::AbstractVector) -> Any

    Return a chain-level representative of the class with coordinate vector `coords` in degree `t`.
    """
    function representative end

    """
        coordinates(space, t::Integer, cocycle) -> AbstractVector

    Return the coordinate vector of the class represented by `cocycle` in degree `t`.
    Concrete implementations may require that `cocycle` is a cycle/cocycle.
    """
    function coordinates end

    """
        cycles(space, t::Integer)

    Optional. Return a representation of the cycle space in degree `t`.
    """
    function cycles end

    """
        boundaries(space, t::Integer)

    Optional. Return a representation of the boundary space in degree `t`.
    """
    function boundaries end

end # module GradedSpaces

using ..ChainComplexes

GradedSpaces.degree_range(C::ChainComplexes.CochainComplex) = ChainComplexes.degree_range(C)
GradedSpaces.degree_range(f::ChainComplexes.CochainMap) = ChainComplexes.degree_range(f)
GradedSpaces.degree_range(les::ChainComplexes.LongExactSequence) = ChainComplexes.degree_range(les)
GradedSpaces.basis(H::ChainComplexes.CohomologyData{K}) where {K} = ChainComplexes.basis(H)
GradedSpaces.basis(H::ChainComplexes.HomologyData{K}) where {K} = ChainComplexes.basis(H)
GradedSpaces.basis(SQ::ChainComplexes.SubquotientData{K}) where {K} = ChainComplexes.basis(SQ)
"""
    basis(M::PModule, q::Integer) -> Matrix

Return the canonical basis of the stalk `M(q)`.

For a `PModule`, the default basis is the standard coordinate basis of the
vector space at vertex `q`. The returned matrix has one basis vector per column.

Best practices
- Use this when you want an explicit coordinate convention at a single vertex.
- Pair it with `coordinates(M, q, x)` to move between vectors and basis
  coordinates.
- Prefer this accessor over reading raw dimensions and constructing identity
  matrices by hand.
"""
GradedSpaces.basis(M::PModule, q::Integer) = Modules._pmodule_basis(M, Int(q))

GradedSpaces.representative(H::ChainComplexes.CohomologyData{K}, coords::AbstractVector{K}) where {K} =
    vec(ChainComplexes.cohomology_representative(H, coords))
GradedSpaces.representative(H::ChainComplexes.HomologyData{K}, coords::AbstractVector{K}) where {K} =
    vec(ChainComplexes.homology_representative(H, coords))
GradedSpaces.representative(SQ::ChainComplexes.SubquotientData{K}, coords::AbstractVector{K}) where {K} =
    vec(ChainComplexes.subquotient_representative(SQ, coords))

GradedSpaces.coordinates(H::ChainComplexes.CohomologyData{K}, cocycle::AbstractVector{K}) where {K} =
    vec(ChainComplexes.coordinates(H, cocycle))
GradedSpaces.coordinates(H::ChainComplexes.HomologyData{K}, cycle::AbstractVector{K}) where {K} =
    vec(ChainComplexes.coordinates(H, cycle))
GradedSpaces.coordinates(SQ::ChainComplexes.SubquotientData{K}, z::AbstractVector{K}) where {K} =
    vec(ChainComplexes.coordinates(SQ, z))
"""
    coordinates(M::PModule, q::Integer, x) -> AbstractVector or Matrix

Express a stalk vector or a matrix of stalk vectors in the canonical basis of
`M(q)`.

For `PModule`s this is the identity coordinate convention on the stalk at
vertex `q`, so the main value of this helper is contract checking and API
uniformity with other graded objects.

Accepted inputs
- `x::AbstractVector`: a single vector in `M(q)`,
- `x::AbstractMatrix`: one vector per column in `M(q)`.

Best practices
- Use this accessor rather than manually trusting dimensions when writing
  examples, notebooks, or tests.
- When coordinates fail, the error message gives a direct size mismatch at the
  requested vertex.
"""
GradedSpaces.coordinates(M::PModule{K}, q::Integer, x::AbstractVector{K}) where {K} =
    Modules._pmodule_coordinates(M, Int(q), x)
GradedSpaces.coordinates(M::PModule{K}, q::Integer, X::AbstractMatrix{K}) where {K} =
    Modules._pmodule_coordinates(M, Int(q), X)

"""
    dimensions(M::PModule) -> NamedTuple
    dimensions(M::PModule, q::Integer) -> NamedTuple
    dimensions(f::PMorphism) -> NamedTuple

Return compact dimension summaries for module objects.

For a `PModule`, the whole-object summary reports:
- number of vertices,
- stalk dimensions,
- total dimension,
- maximum stalk dimension,
- number of cover-edge maps.

For `dimensions(M, q)`, the summary reports the dimension of the stalk `M(q)`.

For a `PMorphism`, the summary reports:
- number of vertices,
- domain stalk dimensions,
- codomain stalk dimensions,
- number of nonzero component maps.

Best practices
- Use `dimensions(...)` first when orienting yourself in a new example.
- Use `describe(...)` when you want a higher-level semantic summary.
- Reserve raw field inspection for debugging storage-level issues.
"""
ChainComplexes.dimensions(M::PModule) = Modules._pmodule_dimensions(M)
ChainComplexes.dimensions(M::PModule, q::Integer) = Modules._pmodule_dimensions(M, Int(q))
ChainComplexes.dimensions(f::PMorphism) = Modules._pmorphism_dimensions(f)

"""
    describe(M::PModule) -> NamedTuple
    describe(f::PMorphism) -> NamedTuple

Return a compact mathematical summary of a module object.

For `PModule`, the summary includes the field, number of vertices, total
dimension, maximum stalk dimension, number of cover-edge maps, and the poset
type.

For `PMorphism`, the summary includes the field, number of vertices, total
domain and codomain dimensions, and the number of nonzero component maps.

Best practices
- Use `describe(...)` in the REPL or notebooks when you want a compact overview
  without reading raw fields.
- Combine `describe(...)` with `check_module(...)` / `check_morphism(...)` for
  hand-built examples.
"""
ChainComplexes.describe(M::PModule) = Modules._pmodule_describe(M)
ChainComplexes.describe(f::PMorphism) = Modules._pmorphism_describe(f)

"""
    dimensions(S::Submodule) -> NamedTuple
    dimensions(ses::ShortExactSequence) -> NamedTuple
    dimensions(sn::SnakeLemmaResult) -> NamedTuple

Return compact dimension summaries for categorical result objects from
`AbelianCategories`.

These summaries are intended as the first inspection tool in the REPL or in
notebooks, before reading raw fields such as `incl`, `i`, `p`, or `delta`.
"""
ChainComplexes.dimensions(S::AbelianCategories.Submodule) = AbelianCategories._submodule_dimensions(S)
ChainComplexes.dimensions(ses::AbelianCategories.ShortExactSequence) = AbelianCategories._short_exact_sequence_dimensions(ses)
ChainComplexes.dimensions(sn::AbelianCategories.SnakeLemmaResult) = AbelianCategories._snake_lemma_dimensions(sn)

"""
    describe(S::Submodule) -> NamedTuple
    describe(ses::ShortExactSequence) -> NamedTuple
    describe(sn::SnakeLemmaResult) -> NamedTuple

Return structured summaries for the main wrapper/result objects in
`AbelianCategories`.

Best practices
- use `describe(...)` for a semantic overview,
- use `dimensions(...)` for the compact size data,
- use the dedicated accessors and validation helpers when you need the stored
  maps or want to verify a hand-built object.
"""
ChainComplexes.describe(S::AbelianCategories.Submodule) = AbelianCategories._submodule_describe(S)
ChainComplexes.describe(ses::AbelianCategories.ShortExactSequence) = AbelianCategories._short_exact_sequence_describe(ses)
ChainComplexes.describe(sn::AbelianCategories.SnakeLemmaResult) = AbelianCategories._snake_lemma_describe(sn)
