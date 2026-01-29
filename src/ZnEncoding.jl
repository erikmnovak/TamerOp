module ZnEncoding

using LinearAlgebra
using SparseArrays
using Random

using ..CoreModules: QQ, AbstractPLikeEncodingMap, EncodingOptions
using ..Stats: _wilson_interval

import ..CoreModules: locate, dimension, representatives, axes_from_encoding
import ..RegionGeometry: region_weights, region_adjacency
using ..ExactQQ: colspaceQQ, solve_fullcolumnQQ
using ..FiniteFringe: FinitePoset, cover_edges, Upset, Downset,
                       upset_closure, downset_closure, intersects, FringeModule
using ..Modules: PModule
using ..FlangeZn: Flange, IndFlat, IndInj, in_flat, in_inj

# Build the finite grid poset on [a,b] subset Z^n, ordered coordinatewise.
# Returns (Q, coords) where coords[i] is an NTuple{n,Int}.
function grid_poset(a::Vector{Int}, b::Vector{Int})
    n = length(a)
    @assert length(b) == n
    lens = [b[i] - a[i] + 1 for i in 1:n]
    if any(lens .<= 0)
        error("grid_poset: invalid box")
    end

    # enumerate all lattice points in the box
    ranges = [a[i]:b[i] for i in 1:n]

    # Preallocate and fill all lattice points in the box.
    N = prod(lens)
    coords = Vector{NTuple{n, Int}}(undef, N)
    idx = 1
    for tup in Iterators.product(ranges...)
        coords[idx] = tup
        idx += 1
    end
    @assert idx == N + 1

    leq = BitMatrix(false, N, N)
    @inbounds for i in 1:N
        gi = coords[i]
        @inbounds for j in 1:N
            gj = coords[j]
            ok = true
            @inbounds for k in 1:n
                if gi[k] > gj[k]
                    ok = false
                    break
                end
            end
            leq[i, j] = ok
        end
    end
    Q = FinitePoset(leq; check=false)
    return Q, coords
end

# Construct the PModule on the grid box induced by a flange presentation:
# M_g = im(Phi_g : F_g -> E_g), and maps are induced from the E-structure maps (projections).
#
# This returns a PModule over the finite grid poset. It is the object you want for Ext/Tor on that layer.
function pmodule_on_box(FG::Flange{QQ}; a::Vector{Int}, b::Vector{Int})
    Q, coords = grid_poset(a, b)
    N = length(coords)

    r = length(FG.injectives)
    c = length(FG.flats)
    Phi = FG.phi

    # For each vertex, compute:
    # - active injectives (rows) in E_g
    # - active flats (cols) in F_g
    # - B_g = basis matrix for im(Phi_g) inside E_g coordinates
    active_rows = Vector{Vector{Int}}(undef, N)
    B = Vector{Matrix{QQ}}(undef, N)
    dims = zeros(Int, N)

    @inbounds for i in 1:N
        g = collect(coords[i])

        rows = Int[]
        for rr in 1:r
            if in_inj(FG.injectives[rr], g)
                push!(rows, rr)
            end
        end
        cols = Int[]
        for cc in 1:c
            if in_flat(FG.flats[cc], g)
                push!(cols, cc)
            end
        end

        active_rows[i] = rows

        if isempty(rows) || isempty(cols)
            B[i] = zeros(QQ, length(rows), 0)
            dims[i] = 0
        else
            Phi_g = Phi[rows, cols]
            Bg = colspaceQQ(Phi_g)   # rows x dim(im)
            B[i] = Bg
            dims[i] = size(Bg, 2)
        end
    end

    # Build edge maps along cover edges in the grid poset using induced maps from E (projection).
    C = cover_edges(Q)
    edge_maps = Dict{Tuple{Int, Int}, Matrix{QQ}}()

    # Helper: build projection E_g -> E_h by selecting the common injective summands (rows_h subset rows_g).
    function projection_matrix(rows_g::Vector{Int}, rows_h::Vector{Int})
        Pg = length(rows_g)
        Ph = length(rows_h)
        P = zeros(QQ, Ph, Pg)
        # rows_* are sorted by construction
        j = 1
        for i in 1:Ph
            target = rows_h[i]
            while j <= Pg && rows_g[j] < target
                j += 1
            end
            if j > Pg || rows_g[j] != target
                error("pmodule_on_box: projection mismatch; expected rows_h subset rows_g")
            end
            P[i, j] = one(QQ)
        end
        return P
    end

    for u in 1:N
        for v in 1:N
            if C[u, v]
                # u < v is a cover edge in grid poset, so coordwise u <= v.
                # Injectives are downsets: active_rows[v] subset active_rows[u].
                rows_u = active_rows[u]
                rows_v = active_rows[v]

                du = dims[u]
                dv = dims[v]

                if dv == 0 || du == 0
                    edge_maps[(u, v)] = zeros(QQ, dv, du)
                    continue
                end

                Pu = projection_matrix(rows_u, rows_v)     # E_u -> E_v
                Im = Pu * B[u]                             # E_v x du
                X = solve_fullcolumnQQ(B[v], Im)          # dv x du
                edge_maps[(u, v)] = X
            end
        end
    end

    return PModule{QQ}(Q, Vector{Int}(dims), edge_maps)
end

# =============================================================================
# Miller-style finite encoding for Z^n (without enumerating lattice points)
# ...
# =============================================================================

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
* `sig_y[t]`       : y-signature of region t (BitVector, one per flat)
* `sig_z[t]`       : z-signature of region t (BitVector, one per injective)
* `reps[t]`        : representative lattice point for region t
* `flats`          : the global flat list used to build signatures
* `injectives`     : the global injective list used to build signatures
* `sig_to_region`  : dictionary mapping a signature key to its region index

The method `locate(pi, g)` returns the region index in `1:P.n` for the point
`g`, or `0` if the signature is not present in the dictionary.
"""
struct ZnEncodingMap <: AbstractPLikeEncodingMap
    n::Int
    coords::Vector{Vector{Int}}
    sig_y::Vector{BitVector}
    sig_z::Vector{BitVector}
    reps::Vector{Vector{Int}}
    flats::Vector{IndFlat}
    injectives::Vector{IndInj}
    sig_to_region::Dict{Tuple{Tuple,Tuple},Int}
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


# ---------------------------- Internal helpers ---------------------------------

"Key for identifying an indecomposable flat up to equality of the underlying upset (ignores `id`)."
_flat_key(F::IndFlat) = (Tuple(F.b), Tuple(F.tau.coords))

"Key for identifying an indecomposable injective up to equality of the underlying downset (ignores `id`)."
_inj_key(E::IndInj) = (Tuple(E.b), Tuple(E.tau.coords))

"""
Build lookup dictionaries for the generator lists used by an encoding.

Returns:
- flat_index[key] = i, where i is an index into `flats`
- inj_index[key]  = j, where j is an index into `injectives`

The keys ignore label `id` on purpose: the encoding depends only on the underlying
(up)set/(down)set, not the symbol used to name it.
"""
function _generator_index_dicts(flats::Vector{IndFlat},
                                injectives::Vector{IndInj})
    flat_index = Dict{Tuple{Tuple,Tuple}, Int}()
    for (i, F) in enumerate(flats)
        key = _flat_key(F)
        if !haskey(flat_index, key)
            flat_index[key] = i
        end
    end

    inj_index = Dict{Tuple{Tuple,Tuple}, Int}()
    for (j, E) in enumerate(injectives)
        key = _inj_key(E)
        if !haskey(inj_index, key)
            inj_index[key] = j
        end
    end

    return flat_index, inj_index
end


"Collect the per-axis critical coordinates needed to make all signatures constant."
function _critical_coords(flats::Vector{IndFlat}, injectives::Vector{IndInj})
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
    return coords
end

"Representative lattice point for the product cell indexed by `idx` (0-based slab indices)."
function _cell_rep(coords::Vector{Vector{Int}}, idx)
    n = length(coords)
    g = Vector{Int}(undef, n)
    @inbounds for i in 1:n
        ci = coords[i]
        if isempty(ci)
            # This axis never appears in any generator inequality, so any value works.
            g[i] = 0
            continue
        end
        if idx[i] == 0
            g[i] = ci[1] - 1
        elseif idx[i] == length(ci)
            g[i] = ci[end]
        else
            g[i] = ci[idx[i]]
        end
    end
    return g
end

"Compute the (y,z) signature at a lattice point g."
function _signature_at(g::AbstractVector{<:Integer},
                       flats::Vector{IndFlat}, injectives::Vector{IndInj})
    y = falses(length(flats))
    z = falses(length(injectives))
    @inbounds for i in 1:length(flats)
        y[i] = in_flat(flats[i], g)
    end
    @inbounds for j in 1:length(injectives)
        z[j] = !in_inj(injectives[j], g)
    end
    return y, z
end

# Tuple overload avoids allocating `collect(g)` when the lattice point is already an NTuple.
function _signature_at(g::NTuple{N,<:Integer}, flats::Vector{IndFlat}, injectives::Vector{IndInj}) where {N}
    y = falses(length(flats))
    z = falses(length(injectives))
    @inbounds for i in eachindex(flats)
        y[i] = in_flat(flats[i], g)
    end
    @inbounds for j in eachindex(injectives)
        z[j] = !in_inj(injectives[j], g)
    end
    return y, z
end

"Uptight poset on signatures by componentwise inclusion (then transitively closed)."
function _uptight_from_signatures(sig_y::Vector{BitVector}, sig_z::Vector{BitVector})
    rN = length(sig_y)
    leq = falses(rN, rN)
    for i in 1:rN
        leq[i,i] = true
    end
    for i in 1:rN, j in 1:rN
        leq[i,j] = all(sig_y[i] .<= sig_y[j]) && all(sig_z[i] .<= sig_z[j])
    end
    # Transitive closure (harmless even though inclusion is already transitive).
    for k in 1:rN, i in 1:rN, j in 1:rN
        leq[i,j] = leq[i,j] || (leq[i,k] && leq[k,j])
    end
    return FinitePoset(leq)
end

"Images of the chosen generator upsets/downsets on the encoded poset P."
function _images_on_P(P::FinitePoset,
                      sig_y::Vector{BitVector}, sig_z::Vector{BitVector},
                      flat_idxs::AbstractVector{<:Integer},
                      inj_idxs::AbstractVector{<:Integer})
    m = length(flat_idxs)
    r = length(inj_idxs)
    Uhat = Vector{Upset}(undef, m)
    Dhat = Vector{Downset}(undef, r)

    for (loc, i0) in enumerate(flat_idxs)
        i = Int(i0)
        mask = BitVector([sig_y[t][i] == 1 for t in 1:P.n])
        Uhat[loc] = upset_closure(P, mask)
    end
    for (loc, j0) in enumerate(inj_idxs)
        j = Int(j0)
        mask = BitVector([sig_z[t][j] == 0 for t in 1:P.n])
        Dhat[loc] = downset_closure(P, mask)
    end
    return Uhat, Dhat
end

"Zero out entries that are forced to be 0 by disjointness of labels (monomiality)."
function _monomialize_phi(phi::AbstractMatrix{K}, Uhat::Vector{Upset}, Dhat::Vector{Downset}) where {K}
    Phi = Matrix{K}(phi)
    for j in 1:length(Dhat), i in 1:length(Uhat)
        if !intersects(Uhat[i], Dhat[j])
            Phi[j,i] = zero(K)
        end
    end
    return Phi
end

# ----------------------------- Public API --------------------------------------

"""
    encode_poset_from_flanges(FGs, opts::EncodingOptions) -> (P, pi)

Construct only the finite encoding poset `P` and classifier `pi : Z^n -> P`
from the union of all flat and injective labels appearing in the given Z^n
flange presentations.

Arguments
- `FGs`: a vector (or tuple) of `Flange` objects. All inputs must have the same
  ambient dimension `n`.
- `opts`: an `EncodingOptions` (required).
  - `opts.backend` must be `:auto` or `:zn`.
  - `opts.max_regions` caps the number of distinct regions/signatures (default: 200_000).

This is the "finite encoding poset" step: extract critical coordinates, form the
product decomposition into finitely many slabs, sample one representative per cell,
and quotient by equal (y,z)-signatures.

Use `fringe_from_flange(P, pi, FG)` to push a flange presentation down to a finite
fringe presentation on `P` without rebuilding the encoding.
"""
function encode_poset_from_flanges(FGs::AbstractVector{<:Flange}, opts::EncodingOptions)
    if opts.backend != :auto && opts.backend != :zn
        error("encode_poset_from_flanges: EncodingOptions.backend must be :auto or :zn")
    end
    max_regions = opts.max_regions === nothing ? 200_000 : Int(opts.max_regions)

    length(FGs) > 0 || error("encode_poset_from_flanges: need at least one flange")

    n = FGs[1].n
    for FG in FGs
        FG.n == n || error("encode_poset_from_flanges: dimension mismatch")
    end

    # Deduplicate generators up to underlying set equality.
    flats_all = IndFlat[]
    injectives_all = IndInj[]
    flat_seen = Dict{Tuple{Tuple,Tuple}, Int}()
    inj_seen  = Dict{Tuple{Tuple,Tuple}, Int}()

    for FG in FGs
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

    coords = _critical_coords(flats_all, injectives_all)

    axes = [0:length(coords[i]) for i in 1:n]
    seen = Dict{Tuple{Tuple,Tuple},Int}()
    sig_y = BitVector[]
    sig_z = BitVector[]
    reps  = Vector{Vector{Int}}()

    for idx in Iterators.product(axes...)
        g = _cell_rep(coords, idx)
        y, z = _signature_at(g, flats_all, injectives_all)
        key = (Tuple(y), Tuple(z))
        if !haskey(seen, key)
            push!(sig_y, y)
            push!(sig_z, z)
            push!(reps, g)
            seen[key] = length(sig_y)
            if length(sig_y) > max_regions
                error("encode_poset_from_flanges: exceeded max_regions=$max_regions")
            end
        end
    end

    P = _uptight_from_signatures(sig_y, sig_z)

    sig_to_region = Dict{Tuple{Tuple,Tuple},Int}()
    for t in 1:length(sig_y)
        sig_to_region[(Tuple(sig_y[t]), Tuple(sig_z[t]))] = t
    end

    pi = ZnEncodingMap(n, coords, sig_y, sig_z, reps, flats_all, injectives_all, sig_to_region)
    return P, pi
end

# Tuple-friendly overload.
function encode_poset_from_flanges(FGs::Tuple{Vararg{Flange}}, opts::EncodingOptions)
    return encode_poset_from_flanges(collect(FGs), opts)
end

# Small-arity overloads (avoid "varargs then opts" signatures).
function encode_poset_from_flanges(FG::Flange, opts::EncodingOptions)
    return encode_poset_from_flanges(Flange[FG], opts)
end

function encode_poset_from_flanges(FG1::Flange, FG2::Flange, opts::EncodingOptions)
    return encode_poset_from_flanges(Flange[FG1, FG2], opts)
end

function encode_poset_from_flanges(FG1::Flange, FG2::Flange, FG3::Flange, opts::EncodingOptions)
    return encode_poset_from_flanges(Flange[FG1, FG2, FG3], opts)
end

"""
    fringe_from_flange(P, pi, FG; strict=true) -> FringeModule{K}

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
function fringe_from_flange(P::FinitePoset, pi::ZnEncodingMap, FG::Flange{K};
                            strict::Bool=true) where {K}
    FG.n == pi.n || error("fringe_from_flange: dimension mismatch (FG.n != pi.n)")
    P.n == length(pi.sig_y) || error("fringe_from_flange: P incompatible with pi (P.n != length(pi.sig_y))")
    length(pi.sig_y) == length(pi.sig_z) || error("fringe_from_flange: malformed pi (sig_y and sig_z lengths differ)")

    if strict
        flat_index, inj_index = _generator_index_dicts(pi.flats, pi.injectives)

        flat_idxs = Vector{Int}(undef, length(FG.flats))
        for i in 1:length(FG.flats)
            key = _flat_key(FG.flats[i])
            idx = get(flat_index, key, 0)
            idx == 0 && error("fringe_from_flange(strict=true): flat label $(FG.flats[i]) not present in encoding generators")
            flat_idxs[i] = idx
        end

        inj_idxs = Vector{Int}(undef, length(FG.injectives))
        for j in 1:length(FG.injectives)
            key = _inj_key(FG.injectives[j])
            idx = get(inj_index, key, 0)
            idx == 0 && error("fringe_from_flange(strict=true): injective label $(FG.injectives[j]) not present in encoding generators")
            inj_idxs[j] = idx
        end

        Uhat, Dhat = _images_on_P(P, pi.sig_y, pi.sig_z, flat_idxs, inj_idxs)
        Phi = _monomialize_phi(FG.phi, Uhat, Dhat)
        return FringeModule{K}(P, Uhat, Dhat, Phi)
    else
        # Fallback: decide membership by evaluating on region representatives.
        m = length(FG.flats)
        r = length(FG.injectives)
        Uhat = Vector{Upset}(undef, m)
        Dhat = Vector{Downset}(undef, r)

        for i in 1:m
            mask = BitVector([in_flat(FG.flats[i], pi.reps[t]) for t in 1:P.n])
            Uhat[i] = upset_closure(P, mask)
        end
        for j in 1:r
            mask = BitVector([in_inj(FG.injectives[j], pi.reps[t]) for t in 1:P.n])
            Dhat[j] = downset_closure(P, mask)
        end

        Phi = _monomialize_phi(FG.phi, Uhat, Dhat)
        return FringeModule{K}(P, Uhat, Dhat, Phi)
    end
end

"""
    locate(pi::ZnEncodingMap, g) -> Int

Return the region index for a lattice point `g` in Z^n.

Accepted inputs:
- `g::AbstractVector{<:Integer}`
- `g::NTuple{N,<:Integer}`

Convenience (for slice code that forms real-valued points):
- `x::AbstractVector{<:AbstractFloat}` is rounded componentwise to the nearest integer
  lattice point and then located.

Return value:
- An integer in `1:length(pi.reps)` if the signature is present in `pi.sig_to_region`.
- `0` if the signature is not present (interpreted as "unknown/outside" by downstream code).

Implementation note:
These methods must be base cases. They must not call `locate` again on an integer vector/tuple,
otherwise it is easy to introduce infinite mutual recursion and a StackOverflowError.
"""
function locate(pi::ZnEncodingMap, g::AbstractVector{<:Integer})
    length(g) == pi.n || error("locate: expected a vector of length $(pi.n), got $(length(g))")
    y, z = _signature_at(g, pi.flats, pi.injectives)
    return get(pi.sig_to_region, (Tuple(y), Tuple(z)), 0)
end

function locate(pi::ZnEncodingMap, g::NTuple{N,<:Integer}) where {N}
    N == pi.n || error("locate: expected a tuple of length $(pi.n), got $(N)")
    y, z = _signature_at(g, pi.flats, pi.injectives)
    return get(pi.sig_to_region, (Tuple(y), Tuple(z)), 0)
end

# Convenience only: lets slice-based code pass Float64 points.
# This method must NOT accept integer vectors, otherwise it can recurse forever.
function locate(pi::ZnEncodingMap, x::AbstractVector{<:AbstractFloat})
    length(x) == pi.n || error("locate: expected a vector of length $(pi.n), got $(length(x))")
    g = round.(Int, x)
    return locate(pi, g)
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
                              a::Vector{Int},
                              b::Vector{Int},
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

function _region_weights_points(pi::ZnEncodingMap,
                               a::Vector{Int},
                               b::Vector{Int};
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

function _region_weights_sample(pi::ZnEncodingMap,
                               a::Vector{Int},
                               b::Vector{Int};
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
"""
function region_weights(
    pi::ZnEncodingMap;
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
)
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

    # Coerce endpoints to Vector{Int} for downstream helpers.
    a = a_in isa Vector{Int} ? a_in : collect(Int, a_in)
    b = b_in isa Vector{Int} ? b_in : collect(Int, b_in)

    length(a) == pi.n || error("region_weights(Zn): box dimension mismatch")
    length(b) == pi.n || error("region_weights(Zn): box dimension mismatch")

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
"""
function region_adjacency(pi::ZnEncodingMap; box, strict::Bool=true)
    box === nothing && error("region_adjacency(Zn): provide box")
    a, b = box
    length(a) == pi.n || error("region_adjacency(Zn): box dimension mismatch")
    length(b) == pi.n || error("region_adjacency(Zn): box dimension mismatch")

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

        adj = Dict{Tuple{Int,Int},Int}()
        for i1 in 1:(L1-1)
            r = reg[i1]
            s = reg[i1+1]
            if r != 0 && s != 0 && r != s
                if counts[1][i1] > 0 && counts[1][i1+1] > 0
                    i, j = (r < s ? (r,s) : (s,r))
                    adj[(i,j)] = get(adj, (i,j), 0) + 1
                end
            end
        end
        return adj
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

        adj = Dict{Tuple{Int,Int},Int}()

        # scan neighbors along axis 1
        for i1 in 1:(L1-1), i2 in 1:L2
            r = reg[i1, i2]
            s = reg[i1+1, i2]
            if r != 0 && s != 0 && r != s
                if counts[1][i1] > 0 && counts[1][i1+1] > 0
                    cross = counts[2][i2]
                    if cross > 0
                        i, j = (r < s ? (r,s) : (s,r))
                        adj[(i,j)] = get(adj, (i,j), 0) + cross
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
                        i, j = (r < s ? (r,s) : (s,r))
                        adj[(i,j)] = get(adj, (i,j), 0) + cross
                    end
                end
            end
        end

        return adj
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
    adj = Dict{Tuple{Int,Int},Int}()

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
                i, j = (r < s ? (r,s) : (s,r))
                adj[(i,j)] = get(adj, (i,j), 0) + cross
            end
        end
    end

    return adj
end


"""
    encode_from_flange(FG::Flange{K}, opts::EncodingOptions) -> (P, H, pi)

Encode a single Z^n flange presentation `FG` to a finite encoding poset `P` and a
finite-poset fringe module `H` on `P`, together with the classifier `pi : Z^n -> P`
(as a `ZnEncodingMap`).

`opts` is required.
- `opts.backend` must be `:auto` or `:zn`.
- `opts.max_regions` caps the number of distinct regions/signatures (default: 200_000).
"""
function encode_from_flange(FG::Flange{K}, opts::EncodingOptions) where {K}
    if opts.backend != :auto && opts.backend != :zn
        error("encode_from_flange: EncodingOptions.backend must be :auto or :zn")
    end
    P, Hs, pi = encode_from_flanges(Flange{K}[FG], opts)
    return P, Hs[1], pi
end

"""
    encode_from_flanges(FGs, opts::EncodingOptions) -> (P, Hs, pi)

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
"""
function encode_from_flanges(FGs::AbstractVector{<:Flange{K}}, opts::EncodingOptions) where {K}
    if opts.backend != :auto && opts.backend != :zn
        error("encode_from_flanges: EncodingOptions.backend must be :auto or :zn")
    end
    P, pi = encode_poset_from_flanges(FGs, opts)

    Hs = Vector{FringeModule{K}}(undef, length(FGs))
    for k in 1:length(FGs)
        Hs[k] = fringe_from_flange(P, pi, FGs[k]; strict=true)
    end
    return P, Hs, pi
end

# Tuple-friendly overload.
function encode_from_flanges(FGs::Tuple{Vararg{Flange{K}}}, opts::EncodingOptions) where {K}
    return encode_from_flanges(collect(FGs), opts)
end

# Small-arity overloads (avoid "varargs then opts" signatures).
function encode_from_flanges(FG1::Flange{K}, FG2::Flange{K}, opts::EncodingOptions) where {K}
    return encode_from_flanges(Flange{K}[FG1, FG2], opts)
end

function encode_from_flanges(FG1::Flange{K}, FG2::Flange{K}, FG3::Flange{K}, opts::EncodingOptions) where {K}
    return encode_from_flanges(Flange{K}[FG1, FG2, FG3], opts)
end




export ZnEncodingMap,
       encode_poset_from_flanges,
       fringe_from_flange,
       encode_from_flange,
       encode_from_flanges

end
