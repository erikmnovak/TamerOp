"""
    SyntheticData

Simple synthetic-data generators for raw data containers and algebraic test
objects used throughout `TamerOp`.

Design notes
- keep generators direct and parameterized, in the spirit of
  `multipers.data.synthetic`,
- return typed `TamerOp` objects instead of raw arrays/tuples whenever the
  project already has a canonical container,
- provide lightweight family helpers so parameter sweeps are easy to build for
  visualization and benchmarking.
"""
module SyntheticData

using Random
using LinearAlgebra

using ..CoreModules: QQField, QQ, AbstractCoeffField, coeff_type, coerce
using ..Options: EncodingOptions
using ..DataTypes: PointCloud, ImageNd, EmbeddedPlanarGraph2D
using ..FiniteFringe: FinitePoset, principal_upset, principal_downset, FringeModule
using ..PLBackend: BoxUpset, BoxDownset
using ..PLPolyhedra: PolyUnion, PLUpset, PLDownset, PLFringe, make_hpoly
using ..FlangeZn: Face, IndFlat, IndInj, Flange, face
import ..DataTypes: ambient_dim, npoints, nvertices, nedges
import ..FiniteFringe: birth_upsets, death_downsets, field
import ..FlangeZn: coefficient_matrix
import ..ChainComplexes: describe
import ..Workflow: encode, _encode_synthetic_box_fringe

"""
    SyntheticFamily

A typed container for a family of synthetic objects produced by sweeping a
single generator over a list or grid of parameter choices.

Use [`synthetic_family`](@ref) when you already have explicit parameter
configurations, and [`sweep_family`](@ref) when you want the Cartesian product
of a parameter grid.
"""
struct SyntheticFamily{T,P,L}
    generator::Symbol
    items::Vector{T}
    parameters::Vector{P}
    labels::Vector{L}
end

"""
    SyntheticBoxFringe

Typed wrapper for axis-aligned synthetic box-fringe data.

This object stores the box upsets, box downsets, and coefficient matrix needed
by `PLBackend.encode_fringe_boxes(...)`, while avoiding raw tuple archaeology.
"""
struct SyntheticBoxFringe{K,F<:AbstractCoeffField}
    field::F
    upsets::Vector{BoxUpset}
    downsets::Vector{BoxDownset}
    phi::Matrix{K}
end

@inline synthetic_items(fam::SyntheticFamily) = fam.items
@inline synthetic_parameters(fam::SyntheticFamily) = fam.parameters
@inline synthetic_labels(fam::SyntheticFamily) = fam.labels
@inline synthetic_generator(fam::SyntheticFamily) = fam.generator
@inline synthetic_family_summary(fam::SyntheticFamily) = describe(fam)
@inline box_fringe_summary(B::SyntheticBoxFringe) = describe(B)

@inline ambient_dim(B::SyntheticBoxFringe) = isempty(B.upsets) ? (isempty(B.downsets) ? 0 : length(B.downsets[1].u)) : length(B.upsets[1].ell)
@inline birth_upsets(B::SyntheticBoxFringe) = B.upsets
@inline death_downsets(B::SyntheticBoxFringe) = B.downsets
@inline coefficient_matrix(B::SyntheticBoxFringe) = B.phi
@inline field(B::SyntheticBoxFringe) = B.field

Base.length(fam::SyntheticFamily) = length(fam.items)
Base.size(fam::SyntheticFamily) = (length(fam),)
Base.firstindex(::SyntheticFamily) = 1
Base.lastindex(fam::SyntheticFamily) = length(fam)
Base.getindex(fam::SyntheticFamily, i::Int) = fam.items[i]
Base.iterate(fam::SyntheticFamily, state::Int=1) = state > length(fam) ? nothing : (fam.items[state], state + 1)
Base.eltype(::Type{SyntheticFamily{T,P,L}}) where {T,P,L} = T

function _describe_item_kind(x)
    if hasmethod(describe, Tuple{typeof(x)})
        desc = describe(x)
        if desc isa NamedTuple && haskey(desc, :kind)
            return desc.kind
        end
    end
    return Symbol(nameof(typeof(x)))
end

function describe(fam::SyntheticFamily)
    n = length(fam)
    param_keys = n == 0 ? Symbol[] : collect(keys(fam.parameters[1]))
    label_preview = n <= 4 ? collect(fam.labels) : vcat(collect(fam.labels[1:3]), ["..."])
    item_kind = n == 0 ? :empty : _describe_item_kind(fam.items[1])
    return (
        kind = :synthetic_family,
        generator = fam.generator,
        nitems = n,
        item_kind = item_kind,
        parameter_keys = param_keys,
        labels = label_preview,
    )
end

function describe(B::SyntheticBoxFringe)
    return (
        kind = :synthetic_box_fringe,
        ambient_dim = ambient_dim(B),
        nupsets = length(B.upsets),
        ndownsets = length(B.downsets),
        matrix_size = size(B.phi),
        field = field(B),
    )
end

function Base.show(io::IO, fam::SyntheticFamily)
    d = describe(fam)
    print(io, "SyntheticFamily(generator=", d.generator, ", nitems=", d.nitems, ", item_kind=", d.item_kind, ")")
end

function Base.show(io::IO, ::MIME"text/plain", fam::SyntheticFamily)
    d = describe(fam)
    print(io,
        "SyntheticFamily",
        "\n  generator: ", d.generator,
        "\n  nitems: ", d.nitems,
        "\n  item_kind: ", d.item_kind,
        "\n  parameter_keys: ", repr(d.parameter_keys),
        "\n  labels: ", repr(d.labels))
end

function Base.show(io::IO, B::SyntheticBoxFringe)
    d = describe(B)
    print(io, "SyntheticBoxFringe(n=", d.ambient_dim, ", nupsets=", d.nupsets, ", ndownsets=", d.ndownsets, ")")
end

function Base.show(io::IO, ::MIME"text/plain", B::SyntheticBoxFringe)
    d = describe(B)
    print(io,
        "SyntheticBoxFringe",
        "\n  ambient_dim: ", d.ambient_dim,
        "\n  nupsets: ", d.nupsets,
        "\n  ndownsets: ", d.ndownsets,
        "\n  matrix_size: ", repr(d.matrix_size),
        "\n  field: ", repr(d.field))
end

function _default_family_label(cfg::NamedTuple)
    parts = String[]
    for k in keys(cfg)
        push!(parts, string(k, "=", getproperty(cfg, k)))
    end
    return join(parts, ", ")
end

@inline function _generator_symbol(f::Function)
    sym = nameof(f)
    return sym isa Symbol ? sym : :synthetic_generator
end

function check_synthetic_family(fam::SyntheticFamily; throw::Bool=false)
    issues = String[]
    length(fam.items) == length(fam.parameters) || push!(issues, "items and parameters must have the same length")
    length(fam.items) == length(fam.labels) || push!(issues, "items and labels must have the same length")
    valid = isempty(issues)
    throw && !valid && Base.throw(ArgumentError("check_synthetic_family: " * join(issues, "; ")))
    return (; kind=:synthetic_family, valid, nitems=length(fam), issues)
end

function check_synthetic_box_fringe(B::SyntheticBoxFringe; throw::Bool=false)
    issues = String[]
    dim = ambient_dim(B)
    size(B.phi, 1) == length(B.downsets) || push!(issues, "coefficient matrix row count must match number of downsets")
    size(B.phi, 2) == length(B.upsets) || push!(issues, "coefficient matrix column count must match number of upsets")
    coeff_type(field(B)) == eltype(B.phi) || push!(issues, "coeff_type(field) must match coefficient matrix element type")
    for (i, U) in enumerate(B.upsets)
        length(U.ell) == dim || push!(issues, "upsets[$i] has ambient dimension $(length(U.ell)), expected $dim")
    end
    for (i, D) in enumerate(B.downsets)
        length(D.u) == dim || push!(issues, "downsets[$i] has ambient dimension $(length(D.u)), expected $dim")
    end
    for i in 1:length(B.downsets), j in 1:length(B.upsets)
        if !iszero(B.phi[i, j])
            lo = B.upsets[j].ell
            hi = B.downsets[i].u
            length(lo) == length(hi) || continue
            all(lo[k] <= hi[k] for k in eachindex(lo, hi)) ||
                push!(issues, "phi[$i,$j] is nonzero but the corresponding upset/downset box is empty")
        end
    end
    valid = isempty(issues)
    throw && !valid && Base.throw(ArgumentError("check_synthetic_box_fringe: " * join(issues, "; ")))
    return (; kind=:synthetic_box_fringe, valid, ambient_dim=dim, issues)
end

function encode(B::SyntheticBoxFringe, enc::EncodingOptions;
                output::Symbol=:encoding_result,
                cache=:auto)
    return _encode_synthetic_box_fringe(B, enc; output=output, cache=cache)
end

function encode(B::SyntheticBoxFringe;
                backend::Symbol=:auto,
                max_regions=nothing,
                strict_eps=nothing,
                poset_kind::Symbol=:signature,
                field::Union{AbstractCoeffField,Nothing}=nothing,
                output::Symbol=:encoding_result,
                cache=:auto)
    field === nothing && (field = B.field)
    enc = EncodingOptions(; backend=backend, max_regions=max_regions,
                          strict_eps=strict_eps, poset_kind=poset_kind, field=field)
    return encode(B, enc; output=output, cache=cache)
end

function _normalize_center(center, dim::Int)
    center === nothing && return zeros(Float64, dim)
    length(center) == dim || error("center length $(length(center)) does not match dim=$dim")
    return Float64[center...]
end

function _normalize_scalars(scalars, n::Int, field::AbstractCoeffField)
    K = coeff_type(field)
    if scalars === nothing
        return [coerce(field, one(Int)) for _ in 1:n]
    elseif scalars isa Number
        s = coerce(field, scalars)
        return [s for _ in 1:n]
    else
        length(scalars) == n || error("expected $n scalars, got $(length(scalars))")
        return K[coerce(field, scalars[i]) for i in 1:n]
    end
end

function _diag_matrix(vals::AbstractVector{K}) where {K}
    n = length(vals)
    Phi = zeros(K, n, n)
    @inbounds for i in 1:n
        Phi[i, i] = vals[i]
    end
    return Phi
end

function _coerce_matrix(field::AbstractCoeffField,
                        phi_in,
                        nrows::Int,
                        ncols::Int;
                        context::AbstractString)
    size(phi_in) == (nrows, ncols) ||
        error("$context: phi must have size ($nrows, $ncols), got $(size(phi_in))")
    K = coeff_type(field)
    Phi = Matrix{K}(undef, nrows, ncols)
    @inbounds for i in 1:nrows, j in 1:ncols
        Phi[i, j] = coerce(field, phi_in[i, j])
    end
    return Phi
end

function _upper_bidiagonal_matrix(field::AbstractCoeffField,
                                  n::Int;
                                  diagonal::Real=1,
                                  superdiag::Real=-1)
    K = coeff_type(field)
    Phi = zeros(K, n, n)
    d = coerce(field, diagonal)
    s = coerce(field, superdiag)
    @inbounds for i in 1:n
        Phi[i, i] = d
        i < n && (Phi[i, i + 1] = s)
    end
    return Phi
end

function _chain_poset(n::Int)
    n >= 0 || error("chain_bar_fringe: n must be nonnegative")
    leq = falses(n, n)
    @inbounds for i in 1:n
        for j in i:n
            leq[i, j] = true
        end
    end
    return FinitePoset(leq; check=false)
end

function _diamond_poset()
    leq = falses(4, 4)
    @inbounds for i in 1:4
        leq[i, i] = true
    end
    leq[1, 2] = true
    leq[1, 3] = true
    leq[1, 4] = true
    leq[2, 4] = true
    leq[3, 4] = true
    return FinitePoset(leq; check=false)
end

function _normalize_chain_bars(bars)
    isempty(bars) && error("bars must be nonempty")
    out = Tuple{Int,Int}[]
    max_death = 0
    for (k, bar) in enumerate(bars)
        length(bar) == 2 || error("bars[$k] must be a pair (birth, death)")
        b = Int(bar[1])
        d = Int(bar[2])
        1 <= b <= d || error("bars[$k] must satisfy 1 <= birth <= death")
        push!(out, (b, d))
        max_death = max(max_death, d)
    end
    return out, max_death
end

function _normalize_box_bars(bars; integer::Bool=false)
    isempty(bars) && error("bars must be nonempty")
    lower = Vector{Vector{Float64}}()
    upper = Vector{Vector{Float64}}()
    dim = nothing
    for (k, bar) in enumerate(bars)
        length(bar) == 2 || error("bars[$k] must be a pair (lower, upper)")
        lo = integer ? Float64[Int[x for x in bar[1]]...] : Float64[bar[1]...]
        hi = integer ? Float64[Int[x for x in bar[2]]...] : Float64[bar[2]...]
        length(lo) == length(hi) || error("bars[$k] lower/upper lengths differ")
        all(lo[i] <= hi[i] for i in eachindex(lo, hi)) || error("bars[$k] must satisfy lower <= upper coordinatewise")
        if dim === nothing
            dim = length(lo)
        else
            length(lo) == dim || error("bars[$k] ambient dimension mismatch")
        end
        push!(lower, lo)
        push!(upper, hi)
    end
    return lower, upper, Int(dim)
end

function _orthant_upset_poly(lo::AbstractVector{<:Real})
    n = length(lo)
    A = zeros(Float64, n, n)
    b = zeros(Float64, n)
    @inbounds for i in 1:n
        A[i, i] = -1.0
        b[i] = -float(lo[i])
    end
    return PLUpset(PolyUnion(n, [make_hpoly(A, b)]))
end

function _orthant_downset_poly(hi::AbstractVector{<:Real})
    n = length(hi)
    A = zeros(Float64, n, n)
    b = zeros(Float64, n)
    @inbounds for i in 1:n
        A[i, i] = 1.0
        b[i] = float(hi[i])
    end
    return PLDownset(PolyUnion(n, [make_hpoly(A, b)]))
end

"""
    noisy_annulus(; n_annulus=1000, n_noise=200, r_inner=1.0, r_outer=2.0,
                    dim=2, center=nothing, rng=Random.default_rng()) -> PointCloud

Generate a point cloud consisting of a uniform annulus in `R^dim` together with
uniform diffuse noise in the surrounding box.

This follows the same simple spirit as `multipers.data.synthetic.noisy_annulus`,
but returns a typed `PointCloud` directly.
"""
function noisy_annulus(; n_annulus::Integer=1000,
                         n_noise::Integer=200,
                         r_inner::Real=1.0,
                         r_outer::Real=2.0,
                         dim::Integer=2,
                         center=nothing,
                         rng::AbstractRNG=Random.default_rng())
    nn = Int(n_annulus)
    nnoise = Int(n_noise)
    d = Int(dim)
    1 <= d || error("dim must be positive")
    0 <= r_inner <= r_outer || error("expected 0 <= r_inner <= r_outer")
    c = _normalize_center(center, d)

    theta = randn(rng, nn, d)
    norms = sqrt.(sum(abs2, theta; dims=2))
    theta ./= norms
    rs = sqrt.(rand(rng, nn) .* (float(r_outer)^2 - float(r_inner)^2) .+ float(r_inner)^2)
    annulus = theta .* rs
    annulus .+= reshape(c, 1, d)

    noise = rand(rng, nnoise, d) .* (2.2 * float(r_outer)) .- (1.1 * float(r_outer))
    noise .+= reshape(c, 1, d)
    return PointCloud(vcat(annulus, noise))
end

"""
    three_annuli(; npoints=500, noutliers=500, rng=Random.default_rng()) -> PointCloud

Generate a two-dimensional point cloud with three disjoint annuli and a cloud
of diffuse outliers.
"""
function three_annuli(; npoints::Integer=500,
                        noutliers::Integer=500,
                        rng::AbstractRNG=Random.default_rng())
    q, r = divrem(Int(npoints), 3)
    counts = (q, q + (r > 0 ? 1 : 0), q + (r > 1 ? 1 : 0))
    outliers = rand(rng, Int(noutliers), 2) .* 4 .- 2
    a1 = point_matrix(noisy_annulus(; n_annulus=counts[1], n_noise=0, r_inner=0.6, r_outer=0.9, center=[1.0, -0.2], rng=rng))
    a2 = point_matrix(noisy_annulus(; n_annulus=counts[2], n_noise=0, r_inner=0.4, r_outer=0.55, center=[-1.2, -1.0], rng=rng))
    a3 = point_matrix(noisy_annulus(; n_annulus=counts[3], n_noise=0, r_inner=0.3, r_outer=0.4, center=[-0.7, 1.1], rng=rng))
    return PointCloud(vcat(outliers, a1, a2, a3))
end

"""
    coupled_orbit(; npoints=1000, r=1.0, x0=nothing, rng=Random.default_rng()) -> PointCloud

Generate a point cloud from the two-dimensional coupled logistic-like orbit used
in synthetic dynamical examples.
"""
function coupled_orbit(; npoints::Integer=1000,
                         r::Real=1.0,
                         x0=nothing,
                         rng::AbstractRNG=Random.default_rng())
    n = Int(npoints)
    n >= 1 || error("npoints must be positive")
    x, y = if x0 === nothing
        vals = rand(rng, 2)
        (vals[1], vals[2])
    else
        length(x0) == 2 || error("x0 must have length 2")
        (float(x0[1]), float(x0[2]))
    end
    pts = Matrix{Float64}(undef, n, 2)
    pts[1, 1] = x
    pts[1, 2] = y
    @inbounds for i in 2:n
        x = mod(x + float(r) * y * (1 - y), 1.0)
        y = mod(y + float(r) * x * (1 - x), 1.0)
        pts[i, 1] = x
        pts[i, 2] = y
    end
    return PointCloud(pts)
end

"""
    gaussian_clusters(; counts=[100,100], centers=[[-1,0],[1,0]], std=0.1,
                        rng=Random.default_rng()) -> PointCloud

Generate a point cloud made of isotropic Gaussian clusters.
"""
function gaussian_clusters(; counts::AbstractVector{<:Integer}=[100, 100],
                              centers::AbstractVector=[[-1.0, 0.0], [1.0, 0.0]],
                              std=0.1,
                              rng::AbstractRNG=Random.default_rng())
    length(counts) == length(centers) || error("counts and centers must have the same length")
    length(centers) > 0 || error("need at least one center")
    dim = length(centers[1])
    sigmas = std isa Number ? fill(float(std), length(counts)) : Float64[std...]
    length(sigmas) == length(counts) || error("std must be scalar or have one entry per cluster")
    total = sum(Int, counts)
    pts = Matrix{Float64}(undef, total, dim)
    row = 1
    for i in eachindex(counts, centers, sigmas)
        c = Float64[centers[i]...]
        length(c) == dim || error("all centers must have the same ambient dimension")
        ni = Int(counts[i])
        sigma = sigmas[i]
        sigma >= 0 || error("standard deviations must be nonnegative")
        block = randn(rng, ni, dim) .* sigma
        block .+= reshape(c, 1, dim)
        pts[row:(row + ni - 1), :] = block
        row += ni
    end
    return PointCloud(pts)
end

"""
    checkerboard_image(; size=(64,64), blocks=(8,8), low=0.0, high=1.0) -> ImageNd

Generate a scalar image whose values alternate on a checkerboard grid.
"""
function checkerboard_image(; size::Tuple{Int,Int}=(64, 64),
                              blocks::Tuple{Int,Int}=(8, 8),
                              low::Real=0.0,
                              high::Real=1.0)
    m, n = size
    bm, bn = blocks
    m >= 1 && n >= 1 || error("image size must be positive")
    bm >= 1 && bn >= 1 || error("block counts must be positive")
    data = Matrix{Float64}(undef, m, n)
    @inbounds for i in 1:m, j in 1:n
        bi = Int(fld((i - 1) * bm, m))
        bj = Int(fld((j - 1) * bn, n))
        data[i, j] = iseven(bi + bj) ? float(low) : float(high)
    end
    return ImageNd(data)
end

"""
    planar_grid_graph(; nrows=4, ncols=4, spacing=1.0, diagonals=false) -> EmbeddedPlanarGraph2D

Generate a rectangular embedded planar graph with grid coordinates.
"""
function planar_grid_graph(; nrows::Integer=4,
                             ncols::Integer=4,
                             spacing::Real=1.0,
                             diagonals::Bool=false)
    r = Int(nrows)
    c = Int(ncols)
    r >= 1 && c >= 1 || error("nrows and ncols must be positive")
    verts = Matrix{Float64}(undef, r * c, 2)
    idx(i, j) = (i - 1) * c + j
    @inbounds for i in 1:r, j in 1:c
        k = idx(i, j)
        verts[k, 1] = float(spacing) * (j - 1)
        verts[k, 2] = float(spacing) * (i - 1)
    end
    edges = Tuple{Int,Int}[]
    @inbounds for i in 1:r, j in 1:c
        j < c && push!(edges, (idx(i, j), idx(i, j + 1)))
        i < r && push!(edges, (idx(i, j), idx(i + 1, j)))
        if diagonals && i < r && j < c
            push!(edges, (idx(i, j), idx(i + 1, j + 1)))
            push!(edges, (idx(i + 1, j), idx(i, j + 1)))
        end
    end
    bbox = (0.0, 0.0, float(spacing) * (c - 1), float(spacing) * (r - 1))
    return EmbeddedPlanarGraph2D(verts, edges; bbox=bbox)
end

"""
    chain_bar_fringe(; bars=[(2,4)], n=nothing, scalars=nothing, field=QQField()) -> FringeModule

Generate a finite-fringe module on a chain poset by placing one interval-like
bar on each pair `(birth, death)`.

Each bar contributes one birth upset, one death downset, and one diagonal entry
in the presentation matrix.
"""
function chain_bar_fringe(; bars=[(2, 4)],
                            n::Union{Nothing,Integer}=nothing,
                            scalars=nothing,
                            field::AbstractCoeffField=QQField())
    barvec, max_death = _normalize_chain_bars(bars)
    nn = n === nothing ? max_death : Int(n)
    nn >= max_death || error("n must be at least the largest death index")
    P = _chain_poset(nn)
    U = [principal_upset(P, b) for (b, _) in barvec]
    D = [principal_downset(P, d) for (_, d) in barvec]
    vals = _normalize_scalars(scalars, length(barvec), field)
    Phi = _diag_matrix(vals)
    return FringeModule{coeff_type(field)}(P, U, D, Phi; field=field)
end

"""
    coupled_chain_fringe(; bars=[(1,4), (2,5), (3,6)], phi=nothing, n=nothing,
                           field=QQField()) -> FringeModule

Generate a finite-fringe module on a chain whose presentation matrix couples
neighboring bars.

The default `phi` is upper-bidiagonal with diagonal entries `1` and first
superdiagonal entries `-1`, so adjacent bars share relations instead of living
as a pure direct sum.
"""
function coupled_chain_fringe(; bars=[(1, 4), (2, 5), (3, 6)],
                                phi=nothing,
                                n::Union{Nothing,Integer}=nothing,
                                field::AbstractCoeffField=QQField())
    barvec, max_death = _normalize_chain_bars(bars)
    nn = n === nothing ? max_death : Int(n)
    nn >= max_death || error("n must be at least the largest death index")
    P = _chain_poset(nn)
    U = [principal_upset(P, b) for (b, _) in barvec]
    D = [principal_downset(P, d) for (_, d) in barvec]
    Phi = phi === nothing ?
        _upper_bidiagonal_matrix(field, length(barvec)) :
        _coerce_matrix(field, phi, length(D), length(U); context="coupled_chain_fringe")
    return FringeModule{coeff_type(field)}(P, U, D, Phi; field=field)
end

"""
    diamond_fringe(; phi=nothing, field=QQField()) -> FringeModule

Generate a finite-fringe module on the four-vertex diamond poset

```text
    4
   / \\
  2   3
   \\ /
    1
```

with two branch generators meeting in a shared top relation.

The default `phi` is the `1 x 2` row matrix `[1 -1]`, so the two branch bars
interact at the top vertex instead of remaining independent.
"""
function diamond_fringe(; phi=nothing,
                          field::AbstractCoeffField=QQField())
    P = _diamond_poset()
    U = [principal_upset(P, 2), principal_upset(P, 3)]
    D = [principal_downset(P, 4)]
    Phi = phi === nothing ?
        _coerce_matrix(field, [1 -1], 1, 2; context="diamond_fringe") :
        _coerce_matrix(field, phi, length(D), length(U); context="diamond_fringe")
    return FringeModule{coeff_type(field)}(P, U, D, Phi; field=field)
end

"""
    box_bar_fringe(; bars=[([0,0],[1,1])], scalars=nothing, field=QQField()) -> SyntheticBoxFringe

Generate axis-aligned box-fringe data suitable for `PLBackend.encode_fringe_boxes`.
"""
function box_bar_fringe(; bars=[([0.0, 0.0], [1.0, 1.0])],
                          scalars=nothing,
                          field::AbstractCoeffField=QQField())
    lower, upper, _ = _normalize_box_bars(bars)
    U = [BoxUpset(lo) for lo in lower]
    D = [BoxDownset(hi) for hi in upper]
    vals = _normalize_scalars(scalars, length(lower), field)
    Phi = _diag_matrix(vals)
    B = SyntheticBoxFringe(field, U, D, Phi)
    check_synthetic_box_fringe(B; throw=true)
    return B
end

"""
    staircase_box_fringe(; bars=[...], phi=nothing, field=QQField()) -> SyntheticBoxFringe

Generate overlapping axis-aligned box-fringe data with a default upper-bidiagonal
coupling matrix.

The default boxes form a staircase in `R^2`, so adjacent generators overlap and
the non-diagonal entries of `phi` survive the box-intersection check.
"""
function staircase_box_fringe(; bars=[([0.0, 0.0], [1.5, 1.0]),
                                      ([0.75, 0.5], [2.5, 2.0]),
                                      ([1.5, 1.0], [3.5, 3.0])],
                                phi=nothing,
                                field::AbstractCoeffField=QQField())
    lower, upper, _ = _normalize_box_bars(bars)
    U = [BoxUpset(lo) for lo in lower]
    D = [BoxDownset(hi) for hi in upper]
    Phi = phi === nothing ?
        _upper_bidiagonal_matrix(field, length(lower)) :
        _coerce_matrix(field, phi, length(D), length(U); context="staircase_box_fringe")
    B = SyntheticBoxFringe(field, U, D, Phi)
    check_synthetic_box_fringe(B; throw=true)
    return B
end

"""
    pl_box_fringe(; bars=[([0,0],[1,1])], scalars=nothing, field=QQField()) -> PLFringe

Generate a PL fringe whose birth and death sets are single-part axis-aligned
orthants, matching the same bar semantics as [`box_bar_fringe`](@ref).
"""
function pl_box_fringe(; bars=[([0.0, 0.0], [1.0, 1.0])],
                         scalars=nothing,
                         field::AbstractCoeffField=QQField())
    lower, upper, _ = _normalize_box_bars(bars)
    U = [_orthant_upset_poly(lo) for lo in lower]
    D = [_orthant_downset_poly(hi) for hi in upper]
    vals = _normalize_scalars(scalars, length(lower), field)
    Phi = _diag_matrix(vals)
    return PLFringe(U, D, Phi)
end

"""
    coupled_pl_fringe(; bars=[...], phi=nothing, field=QQField()) -> PLFringe

Generate a PL fringe whose birth/death pieces are orthant polyhedra with a
non-diagonal coupling matrix.

This is the PL analogue of [`staircase_box_fringe`](@ref): the default birth
and death sets form a staircase of overlapping boxes, while the default `phi`
couples neighboring generators through an upper-bidiagonal pattern.
"""
function coupled_pl_fringe(; bars=[([0.0, 0.0], [1.5, 1.0]),
                                   ([0.75, 0.5], [2.5, 2.0]),
                                   ([1.5, 1.0], [3.5, 3.0])],
                             phi=nothing,
                             field::AbstractCoeffField=QQField())
    lower, upper, _ = _normalize_box_bars(bars)
    U = [_orthant_upset_poly(lo) for lo in lower]
    D = [_orthant_downset_poly(hi) for hi in upper]
    Phi = phi === nothing ?
        _upper_bidiagonal_matrix(field, length(lower)) :
        _coerce_matrix(field, phi, length(D), length(U); context="coupled_pl_fringe")
    return PLFringe(U, D, Phi)
end

"""
    orthant_bar_flange(; bars=[([0,0],[1,1])], scalars=nothing, field=QQField()) -> Flange

Generate a flange in `Z^n` from fully constrained orthant generators.

Each bar `(lower, upper)` defines one indexed flat and one indexed injective,
with diagonal coefficient matrix.
"""
function orthant_bar_flange(; bars=[([0, 0], [1, 1])],
                              scalars=nothing,
                              field::AbstractCoeffField=QQField())
    lower, upper, dim = _normalize_box_bars(bars; integer=true)
    tau = face(dim, Int[])
    flats = [IndFlat(tau, Int[round(Int, x) for x in lo]; id=Symbol(:F, i)) for (i, lo) in enumerate(lower)]
    injectives = [IndInj(tau, Int[round(Int, x) for x in hi]; id=Symbol(:E, i)) for (i, hi) in enumerate(upper)]
    vals = _normalize_scalars(scalars, length(lower), field)
    Phi = _diag_matrix(vals)
    return Flange{coeff_type(field)}(dim, flats, injectives, Phi; field=field)
end

"""
    mixed_face_flange(; phi=nothing, field=QQField()) -> Flange

Generate a flange in `Z^2` whose flats and injectives use different free-face
patterns, together with a non-diagonal coupling matrix.

The default object mixes strip-like generators and a fully constrained corner
generator, which makes it useful for testing flange code beyond the fully
orthant/diagonal regime of [`orthant_bar_flange`](@ref).
"""
function mixed_face_flange(; phi=nothing,
                             field::AbstractCoeffField=QQField())
    n = 2
    flats = [
        IndFlat(face(n, [1]), [0, 0]; id=:Fx),
        IndFlat(face(n, [2]), [0, 0]; id=:Fy),
        IndFlat(face(n, Int[]), [1, 1]; id=:Fq),
    ]
    injectives = [
        IndInj(face(n, [1]), [3, 2]; id=:Ex),
        IndInj(face(n, [2]), [2, 3]; id=:Ey),
        IndInj(face(n, Int[]), [3, 3]; id=:Eq),
    ]
    Phi = phi === nothing ?
        _coerce_matrix(field,
                       [1 1 0;
                        1 0 1;
                        0 1 1],
                       length(injectives),
                       length(flats);
                       context="mixed_face_flange") :
        _coerce_matrix(field, phi, length(injectives), length(flats); context="mixed_face_flange")
    return Flange{coeff_type(field)}(n, flats, injectives, Phi; field=field)
end

"""
    synthetic_family(f, configs; generator=nothing, label=nothing) -> SyntheticFamily

Build a typed family of synthetic objects by applying `f(; cfg...)` to each
NamedTuple configuration in `configs`.

If `label` is omitted, labels are generated from the configuration entries.
"""
function synthetic_family(f::Function,
                          configs::AbstractVector{<:NamedTuple};
                          generator::Union{Nothing,Symbol}=nothing,
                          label::Union{Nothing,Function}=nothing)
    isempty(configs) && error("synthetic_family: configs must be nonempty")
    cfgs = collect(configs)
    items = [f(; cfg...) for cfg in cfgs]
    T = typeof(items[1])
    all(x -> x isa T, items) || error("synthetic_family: generator must return a single concrete item type across the family")
    labels = String[]
    labelf = label === nothing ? _default_family_label : label
    for cfg in cfgs
        push!(labels, String(labelf(cfg)))
    end
    fam = SyntheticFamily{T, typeof(cfgs[1]), String}(generator === nothing ? _generator_symbol(f) : generator,
                                                      T[items...],
                                                      typeof(cfgs[1])[cfgs...],
                                                      labels)
    check_synthetic_family(fam; throw=true)
    return fam
end

"""
    sweep_family(f; sweep, generator=nothing, label=nothing, base_kwargs...) -> SyntheticFamily

Generate a synthetic family from the Cartesian product of the parameter grid in
`sweep`.

Example
-------
```julia
fam = sweep_family(noisy_annulus;
    sweep=(r_outer=[1.5, 2.0], n_noise=[0, 100]),
    n_annulus=200,
    dim=2,
)
```
"""
function sweep_family(f::Function;
                      sweep::NamedTuple,
                      generator::Union{Nothing,Symbol}=nothing,
                      label::Union{Nothing,Function}=nothing,
                      base_kwargs...)
    keys_sweep = collect(keys(sweep))
    isempty(keys_sweep) && error("sweep_family: sweep must contain at least one parameter")
    value_lists = [collect(getproperty(sweep, k)) for k in keys_sweep]
    all(!isempty(v) for v in value_lists) || error("sweep_family: each sweep parameter must have at least one value")
    base_cfg = NamedTuple(base_kwargs)
    cfgs = NamedTuple[]
    for prod in Iterators.product(value_lists...)
        delta = NamedTuple{Tuple(keys_sweep)}(prod)
        push!(cfgs, merge(base_cfg, delta))
    end
    return synthetic_family(f, cfgs; generator=generator, label=label)
end

@inline point_matrix(pc::PointCloud) = getfield(pc, :coords)

end # module
