module ZnEncoding

using LinearAlgebra
using SparseArrays
using Random

using ..CoreModules: QQ, AbstractPLikeEncodingMap, _wilson_interval
import ..CoreModules: locate, region_weights, region_adjacency
using ..ExactQQ: colspaceQQ, solve_fullcolumnQQ
using ..FiniteFringe: FinitePoset, cover_edges, Upset, Downset,
                       upset_closure, downset_closure, intersects, FringeModule
using ..IndicatorResolutions: PModule
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
    ZnEncodingMap{K}

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
struct ZnEncodingMap{K} <: AbstractPLikeEncodingMap
    n::Int
    coords::Vector{Vector{Int}}
    sig_y::Vector{BitVector}
    sig_z::Vector{BitVector}
    reps::Vector{Vector{Int}}
    flats::Vector{IndFlat}
    injectives::Vector{IndInj}
    sig_to_region::Dict{Tuple{Tuple,Tuple},Int}
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
    encode_poset_from_flanges(FG1, FG2, ...; max_regions=200_000) -> (P, pi)

Construct only the finite encoding poset `P` and classifier `pi : Z^n -> P`
from the union of all flat and injective labels appearing in the given Z^n
flange presentations.

This is the "finite encoding poset" step: extract critical coordinates, form the
product decomposition into finitely many slabs, sample one representative per cell,
and quotient by equal (y,z)-signatures.

Use `fringe_from_flange(P, pi, FG)` to push a flange presentation down to a finite
fringe presentation on `P` without rebuilding the encoding.
"""
function encode_poset_from_flanges(FGs::Vararg{Flange{K}}; max_regions::Int=200_000) where {K}
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

    pi = ZnEncodingMap{K}(n, coords, sig_y, sig_z, reps, flats_all, injectives_all, sig_to_region)
    return P, pi
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
function fringe_from_flange(P::FinitePoset, pi::ZnEncodingMap{K}, FG::Flange{K};
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

"""
    region_weights(pi::ZnEncodingMap; box=nothing, method=:auto, count_type=:auto,
                   max_cells=1_000_000, max_points=5_000_000,
                   strict=true, return_info=false,
                   nsamples=10_000, rng=Random.default_rng(), alpha=0.05)

Compute nonnegative weights for the regions of a Z^n-encoding.

The weights are intended for downstream summaries where each region contributes in
proportion to how many lattice points (within a bounding box) map into that region.

Arguments
- `box`: either `nothing` (uniform weights) or a tuple `(a,b)` of integer vectors.
  The box represents the integer lattice points `g` with `a[i] <= g[i] <= b[i]`.
- `method`:
    * `:cells`  - exact, counts slab-cells induced by `pi.coords` (fast when few cells)
    * `:points` - exact, enumerates all lattice points (fast only for small boxes)
    * `:sample` - approximate, Monte Carlo sampling
    * `:auto`   - choose among `:cells`, `:points`, `:sample` using `max_cells/max_points`
- `count_type` controls the integer type used for exact counts:
    * `:int`    - use `Int`
    * `:bigint` - use `BigInt`
    * `:auto`   - use `Int` when the total number of lattice points fits, else `BigInt`

Keywords for sampling
- `nsamples`, `rng`, `alpha`: forwarded to the sampling backend.

Return value
- If `return_info=false`: returns the weight vector.
- If `return_info=true`: returns a named tuple with fields:
    `method`, `weights`, `stderr`, `ci`, `alpha`, `total_points`, `nsamples`, `counts`, `count_type`.

Notes
- When `box === nothing`, the result is the uniform weight vector `ones(Int, length(pi.sig_y))`.
- When `strict=false`, points that fall outside the encoding (if any) are ignored.
  When `strict=true`, such points trigger an error.
"""
function region_weights(
    pi::ZnEncodingMap;
    box=nothing,
    method::Symbol=:auto,
    count_type::Symbol=:auto,
    max_cells::Int=1_000_000,
    max_points::Int=5_000_000,
    strict::Bool=true,
    return_info::Bool=false,
    nsamples::Int=10_000,
    rng=Random.default_rng(),
    alpha::Real=0.05,
)
    nreg = length(pi.sig_y)

    # If no box is supplied, return uniform weights.
    if box === nothing
        w = ones(Int, nreg)
        if !return_info
            return w
        end
        stderr = zeros(Float64, nreg)
        ci = [(1.0, 1.0) for _ in 1:nreg]
        return (
            method=:uniform,
            weights=w,
            stderr=stderr,
            ci=ci,
            alpha=float(alpha),
            total_points=nothing,
            nsamples=0,
            counts=nothing,
            count_type=:int,
        )
    end

    a, b = box
    length(a) == pi.n || error("region_weights(Zn): box dimension mismatch")
    length(b) == pi.n || error("region_weights(Zn): box dimension mismatch")

    # Convert to concrete Int vectors for the internal hot loops.
    aI = [Int(x) for x in a]
    bI = [Int(x) for x in b]

    # Slab index interval induced by the coordinate decomposition.
    lo, hi = _slab_index_interval(pi, aI, bI)

    # Total number of lattice points in the box (as BigInt for safety).
    total_points_big = _box_lattice_size_big(aI, bI)

    # Choose an integer type for exact counting.
    T = Int
    if count_type === :int
        T = Int
    elseif count_type === :bigint
        T = BigInt
    elseif count_type === :auto
        T = (total_points_big <= BigInt(typemax(Int))) ? Int : BigInt
    else
        error("region_weights(Zn): count_type must be :auto, :int, or :bigint")
    end
    count_type_used = (T === Int) ? :int : :bigint

    # Automatic method selection based on cheap size estimates.
    chosen = method
    if method === :auto
        # Estimate the number of slab-cells intersecting the box.
        ncells_big = BigInt(1)
        @inbounds for i in 1:pi.n
            len = hi[i] - lo[i] + 1
            # Allow for empty intervals.
            if len <= 0
                ncells_big = BigInt(0)
                break
            end
            ncells_big *= BigInt(len)
            if ncells_big > BigInt(max_cells) && total_points_big > BigInt(max_points)
                # Early stop: already too large for both exact methods.
                break
            end
        end

        if ncells_big <= BigInt(max_cells)
            chosen = :cells
        elseif total_points_big <= BigInt(max_points)
            chosen = :points
        else
            chosen = :sample
        end
    end

    if chosen === :cells
        w, total_big = _region_weights_cells(pi, aI, bI, lo, hi; strict=strict, T=T)
        if !return_info
            return w
        end
        stderr = zeros(Float64, nreg)
        wf = [float(x) for x in w]
        ci = [(wf[i], wf[i]) for i in 1:nreg]
        return (
            method=:cells,
            weights=w,
            stderr=stderr,
            ci=ci,
            alpha=float(alpha),
            total_points=total_big,
            nsamples=0,
            counts=nothing,
            count_type=count_type_used,
        )

    elseif chosen === :points
        w, total_big = _region_weights_points(pi, aI, bI; strict=strict, T=T)
        if !return_info
            return w
        end
        stderr = zeros(Float64, nreg)
        wf = [float(x) for x in w]
        ci = [(wf[i], wf[i]) for i in 1:nreg]
        return (
            method=:points,
            weights=w,
            stderr=stderr,
            ci=ci,
            alpha=float(alpha),
            total_points=total_big,
            nsamples=0,
            counts=nothing,
            count_type=count_type_used,
        )

    elseif chosen === :sample
        w, stderr, ci, total_big, counts, _raw = _region_weights_sample(
            pi,
            aI,
            bI;
            nsamples=nsamples,
            rng=rng,
            alpha=alpha,
            strict=strict,
            return_counts=true,
        )
        if !return_info
            return w
        end
        return (
            method=:sample,
            weights=w,
            stderr=stderr,
            ci=ci,
            alpha=float(alpha),
            total_points=total_big,
            nsamples=nsamples,
            counts=counts,
            count_type=:int,
        )

    else
        error("region_weights(Zn): unknown method=$method; expected :auto, :cells, :points, or :sample")
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
    encode_from_flange(FG::Flange{K}; max_regions=200_000) -> (P, H, pi)

Build a finite encoding poset `P` for a single Z^n flange presentation `FG`
without enumerating lattice points in a bounding box.

Keyword arguments
- max_regions::Int = 200_000:
    Hard cap on the number of distinct regions/signatures allowed during encoding.
"""
function encode_from_flange(FG::Flange{K};
                            max_regions::Int = 200_000) where {K}
    P, Hs, pi = encode_from_flanges(FG; max_regions = max_regions)
    return P, Hs[1], pi
end


"""
    encode_from_flanges(FG1, FG2, ...; max_regions=200_000) -> (P, Hs, pi)

Build a single finite encoding poset `P` that simultaneously encodes all flats
and injectives appearing in the provided Z^n flange presentations, and return
their pushed-down fringe presentations `Hs` on `P`.

Keyword arguments
- max_regions::Int = 200_000:
    Hard cap on the number of distinct regions/signatures allowed during encoding.
"""
function encode_from_flanges(FGs::Vararg{Flange{K}};
                             max_regions::Int = 200_000) where {K}
    P, pi = encode_poset_from_flanges(FGs...; max_regions = max_regions)

    Hs = Vector{FringeModule{K}}(undef, length(FGs))
    for k in 1:length(FGs)
        Hs[k] = fringe_from_flange(P, pi, FGs[k]; strict = true)
    end
    return P, Hs, pi
end


export ZnEncodingMap,
       encode_poset_from_flanges,
       fringe_from_flange,
       encode_from_flange,
       encode_from_flanges

end
