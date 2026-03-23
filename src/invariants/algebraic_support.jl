# This fragment owns algebraic/support summaries for TamerOp.Invariants,
# including Betti/Bass tables, region-poset access, value-stratified measures,
# and support measures tied to algebraic labels. It depends on earlier rank and
# module-summary helpers and does not own the final support-geometry query
# family.

# ----- Betti/Bass tables for indicator resolutions ------------------------------

# Helper for principal upset summands: unique minimal element.
function _unique_minimal_vertex(U::Upset)::Int
    P = U.P
    mins = Int[]
    for q in 1:nvertices(P)
        U.mask[q] || continue
        # q is minimal in U if there is no p < q also in U.
        ismin = true
        for p in downset_indices(P, q)
            (p == q) && continue
            if U.mask[p]
                ismin = false
                break
            end
        end
        ismin && push!(mins, q)
    end
    length(mins) == 1 || error("expected a principal upset; found $(length(mins)) minimal vertices")
    return mins[1]
end

# Helper for principal downset summands: unique maximal element.
function _unique_maximal_vertex(D::Downset)::Int
    P = D.P
    maxs = Int[]
    for q in 1:nvertices(P)
        D.mask[q] || continue
        # q is maximal in D if there is no r > q also in D.
        ismax = true
        for r in upset_indices(P, q)
            (r == q) && continue
            if D.mask[r]
                ismax = false
                break
            end
        end
        ismax && push!(maxs, q)
    end
    length(maxs) == 1 || error("expected a principal downset; found $(length(maxs)) maximal vertices")
    return maxs[1]
end


# -----------------------------------------------------------------------------
# IMPORTANT:
# TamerOp exports betti/betti_table/bass/bass_table from DerivedFunctors.
# So we must EXTEND those functions here (not define new Invariants.betti, etc).
# -----------------------------------------------------------------------------
import ..DerivedFunctors: betti, betti_table, bass, bass_table

"""
    betti(F) -> Dict{Tuple{Int,Int},Int}

Compute multigraded Betti numbers from an upset indicator resolution.

The input `F` is the `F` returned by `upset_resolution`, i.e. a vector of
`UpsetPresentation{K}` objects. Each term is a direct sum of principal upsets.

The returned dictionary is keyed by `(a, p)` where:
* `a` is the homological degree (starting at 0)
* `p` is a vertex of the underlying poset

The value is the multiplicity of the principal upset based at `p` in term `a`.

This is the same data returned by `betti(projective_resolution(...))`, but is
available directly from the indicator-resolution output.
"""
function betti(F::AbstractVector{<:UpsetPresentation{K}}) where {K}
    length(F) > 0 || return Dict{Tuple{Int,Int},Int}()
    out = Dict{Tuple{Int,Int}, Int}()
    for i in 1:length(F)
        a = i - 1
        for U in F[i].U0
            p = _unique_minimal_vertex(U)
            out[(a, p)] = get(out, (a, p), 0) + 1
        end
    end
    return out
end

"""
    betti_table(F) -> Matrix{Int}

Dense Betti table for an upset indicator resolution.

Row `a+1` and column `p` stores the multiplicity of the principal upset based at
vertex `p` in homological degree `a`.
"""
function betti_table(F::AbstractVector{<:UpsetPresentation{K}}; pad_to::Union{Nothing,Int}=nothing) where {K}
    length(F) > 0 || return zeros(Int, 0, 0)
    P = F[1].P
    B = zeros(Int, length(F), nvertices(P))
    for ((a, p), m) in betti(F)
        B[a + 1, p] = m
    end

    if pad_to === nothing
        return B
    else
        pad_to >= 0 || error("betti_table: pad_to must be >= 0")
        r = pad_to + 1
        if size(B, 1) == r
            return B
        elseif size(B, 1) > r
            return B[1:r, :]
        else
            C = zeros(Int, r, nvertices(P))
            C[1:size(B,1), :] .= B
            return C
        end
    end
end

"""
    bass(E) -> Dict{Tuple{Int,Int},Int}

Compute multigraded Bass numbers from a downset indicator resolution.

The input `E` is the `E` returned by `downset_resolution`, i.e. a vector of
`DownsetCopresentation{K}` objects. Each term is a direct sum of principal
downsets.

The returned dictionary is keyed by `(b, p)` where:
* `b` is the cohomological degree (starting at 0)
* `p` is a vertex of the underlying poset

The value is the multiplicity of the principal downset with top element `p` in
term `b`.
"""
function bass(E::AbstractVector{<:DownsetCopresentation{K}}) where {K}
    length(E) > 0 || return Dict{Tuple{Int,Int},Int}()
    out = Dict{Tuple{Int,Int}, Int}()
    for i in 1:length(E)
        b = i - 1
        for D in E[i].D0
            p = _unique_maximal_vertex(D)
            out[(b, p)] = get(out, (b, p), 0) + 1
        end
    end
    return out
end

"""
    bass_table(E) -> Matrix{Int}

Dense Bass table for a downset indicator resolution.

Row `b+1` and column `p` stores the multiplicity of the principal downset with
top vertex `p` in cohomological degree `b`.
"""
function bass_table(E::AbstractVector{<:DownsetCopresentation{K}}; pad_to::Union{Nothing,Int}=nothing) where {K}
    length(E) > 0 || return zeros(Int, 0, 0)
    P = E[1].P
    B = zeros(Int, length(E), nvertices(P))
    for ((b, p), m) in bass(E)
        B[b + 1, p] = m
    end

    if pad_to === nothing
        return B
    else
        pad_to >= 0 || error("bass_table: pad_to must be >= 0")
        r = pad_to + 1
        if size(B, 1) == r
            return B
        elseif size(B, 1) > r
            return B[1:r, :]
        else
            C = zeros(Int, r, nvertices(P))
            C[1:size(B,1), :] .= B
            return C
        end
    end
end


# =============================================================================
# (5) Linking region size measures back to algebraic / derived invariants
# =============================================================================
#
# The main idea is simple: many algebraic quantities (or quantities derived from
# algebraic computations) are constant on encoding regions. Once you can label
# each region by such an invariant, you can measure the "locus" where it takes
# a given value by summing region weights.
#
# This section provides:
#   - `region_values(pi, f; ...)` evaluate a region-constant function to get a vector.
#   - `measure_by_value(values, pi, opts::InvariantOptions; ...)` stratify a window by an arbitrary label.
#   - `support_measure(mask, pi, opts::InvariantOptions; ...)` measure a subset of regions given as a mask.
#   - `vertex_set_measure(vertices, pi, opts::InvariantOptions; ...)` measure a subset given as indices.
#   - `betti_support_measures(B, pi, opts::InvariantOptions; ...)` /
#     `bass_support_measures(B, pi, opts::InvariantOptions; ...)` compute size summaries
#     of where Betti/Bass numbers are supported.


"""
    _nregions_encoding(pi) -> Int

Internal helper: determine the number of regions for common encoding map types.

This is used by `region_values(pi, f; arg=:index)` to allow value evaluation by
region index even when representatives are not available.

We try (in this order):
- `length(pi.sig_y)` if `pi.sig_y` exists,
- `length(pi.regions)` if `pi.regions` exists,
- `length(pi.reps)` if `pi.reps` exists.
"""
function _nregions_encoding(pi)
    if hasproperty(pi, :sig_y)
        sy = getproperty(pi, :sig_y)
        sy !== nothing && return length(sy)
    end
    if hasproperty(pi, :regions)
        rg = getproperty(pi, :regions)
        rg !== nothing && return length(rg)
    end
    if hasproperty(pi, :reps)
        reps = getproperty(pi, :reps)
        reps !== nothing && return length(reps)
    end
    error("_nregions_encoding: cannot determine number of regions; expected pi.sig_y, pi.regions, or pi.reps")
end


"""
    region_values(pi, f; arg=:rep) -> AbstractVector

Evaluate a region-constant function on each region of an encoding.

This is a lightweight helper for building "stratifications" of parameter space
by any quantity that is constant on regions.

Arguments
---------
- `pi`: an encoding map. If `arg` is `:rep` or `:both`, then `pi` must provide
  representative points via `representatives(pi)`. If `arg` is `:index`,
  representatives are not needed, but `pi` must expose the number of regions via
  one of the common fields `pi.sig_y`, `pi.regions`, or `pi.reps`.
- `f`: function used to label regions.

Keyword arguments
-----------------
- `arg` controls how `f` is called:
  - `:rep`   calls `f(rep)` where `rep = representatives(pi)[r]` is a representative point.
  - `:index` calls `f(r)` where `r` is the region index (1-based).
  - `:both`  calls `f(r, rep)`.

Returns
-------
A vector `vals` with `vals[r] = f(...)` for each region.

Notes
-----
This is intended for derived or external invariants that you can compute on a
single representative point per region.
"""
function region_values(pi, f; arg::Symbol=:rep)
    if arg === :index
        n = _nregions_encoding(pi)
        n == 0 && return Any[]
        v1 = f(1)
        T = typeof(v1)
        out = Vector{T}(undef, n)
        out[1] = v1
        @inbounds for r in 2:n
            out[r] = f(r)
        end
        return out
    end

    # arg == :rep or :both require representatives.
    reps = representatives(pi)
    reps === nothing && error("region_values: representatives(pi) returned nothing")
    n = length(reps)
    n == 0 && return Any[]

    # Compute the first value to infer element type.
    v1 = if arg === :rep
        f(reps[1])
    elseif arg === :both
        f(1, reps[1])
    else
        error("region_values: arg must be :rep, :index, or :both")
    end

    T = typeof(v1)
    out = Vector{T}(undef, n)
    out[1] = v1

    if n == 1
        return out
    end

    if arg === :rep
        @inbounds for r in 2:n
            out[r] = f(reps[r])
        end
    else
        @inbounds for r in 2:n
            out[r] = f(r, reps[r])
        end
    end
    return out
end



# ------------------------------ Region poset access ------------------------------
#
# For "PL-like" encodings (ZnEncoding, PLPolyhedra, PLBackend), the encoder returns
# a region poset P together with an encoding map pi. The encoding map typically
# stores only per-region signatures (sig_y, sig_z) and representative points.
#
# The projected-invariant machinery (and slice_chain sanity checks) needs access
# to this region poset. We therefore reconstruct it from signatures when it is
# not stored explicitly, and cache the result keyed by the encoding map object.

@inline function _encoding_cache_from_pi(pi)::Union{Nothing,EncodingCache}
    pi isa CompiledEncoding || return nothing
    meta = pi.meta
    meta isa EncodingCache && return meta
    if meta isa NamedTuple && hasproperty(meta, :encoding_cache)
        ec = getproperty(meta, :encoding_cache)
        return ec isa EncodingCache ? ec : nothing
    end
    if meta isa AbstractDict && haskey(meta, :encoding_cache)
        ec = meta[:encoding_cache]
        return ec isa EncodingCache ? ec : nothing
    end
    return nothing
end

@inline function _sig_leq(a::AbstractVector{Bool}, b::AbstractVector{Bool})::Bool
    # Componentwise order on {0,1}^m: a <= b iff whenever a[k] is true, b[k] is true.
    length(a) == length(b) || error("_sig_leq: signature length mismatch")
    @inbounds for k in eachindex(a, b)
        if a[k] && !b[k]
            return false
        end
    end
    return true
end

function _uptight_poset_from_signatures(
    sig_y::AbstractVector{<:AbstractVector{Bool}},
    sig_z::AbstractVector{<:AbstractVector{Bool}},
)::FinitePoset
    rN = length(sig_y)
    length(sig_z) == rN || error("_uptight_poset_from_signatures: sig_y and sig_z length mismatch")

    leq = falses(rN, rN)
    @inbounds for i in 1:rN
        yi = sig_y[i]
        zi = sig_z[i]
        for j in 1:rN
            if _sig_leq(yi, sig_y[j]) && _sig_leq(zi, sig_z[j])
                leq[i, j] = true
            end
        end
    end

    # Signature inclusion is reflexive and transitive, so no transitive-closure pass is needed.
    # Signatures come from encoder regions (one signature per region), so we skip FinitePoset
    # validation in this hot reconstruction path.
    return FinitePoset(leq; check=false)
end

"""
    region_poset(pi; poset_kind=:signature, cache=nothing) -> AbstractPoset

Return the finite "region poset" underlying a `PLikeEncodingMap` `pi`.

In this codebase, the region poset is the *uptight poset on signatures*:
for regions r and s we declare r <= s iff both signature bitvectors satisfy

    sig_y[r] <= sig_y[s]   and   sig_z[r] <= sig_z[s]

componentwise (i.e. every upset/downset membership bit that is true in r is also
true in s). This is exactly the poset P constructed by the encoders.

Implementation notes
--------------------
* If the encoding object stores a region poset directly as a field/property `P`,
  we return it.
* Otherwise we reconstruct P from the stored signatures `pi.sig_y` and `pi.sig_z`.
  If `cache` is an `EncodingCache` (or if `pi` is a `CompiledEncoding` carrying one),
  we cache the reconstructed region poset there.
"""
function region_poset(pi::PLikeEncodingMap;
                      poset_kind::Symbol = :signature,
                      cache::Union{Nothing,EncodingCache}=nothing)
    if pi isa CompiledEncoding
        return pi.P
    end
    # Fast path: the encoding itself stores P.
    if hasproperty(pi, :P)
        P = getproperty(pi, :P)
        if P isa AbstractPoset
            return P
        end
    end

    # Signature-based reconstruction (the standard situation for encoders in this repo).
    if !(hasproperty(pi, :sig_y) && hasproperty(pi, :sig_z))
        error("region_poset: pi has no property P and no (sig_y, sig_z); cannot reconstruct for type $(typeof(pi))")
    end
    sig_y = getproperty(pi, :sig_y)
    sig_z = getproperty(pi, :sig_z)

    cache_eff = cache === nothing ? _encoding_cache_from_pi(pi) : cache
    if cache_eff !== nothing
        key = (UInt(objectid(sig_y)), UInt(objectid(sig_z)), poset_kind)
        Base.lock(cache_eff.lock)
        try
            entry = get(cache_eff.region_posets, key, nothing)
            entry === nothing || return entry.value
        finally
            Base.unlock(cache_eff.lock)
        end
    end

    if poset_kind == :signature
        sig_y_bits = [BitVector(collect(row)) for row in sig_y]
        sig_z_bits = [BitVector(collect(row)) for row in sig_z]
        Pnew = ZnEncoding.SignaturePoset(sig_y_bits, sig_z_bits)
    elseif poset_kind == :dense
        Pnew = _uptight_poset_from_signatures(sig_y, sig_z)
    else
        error("region_poset: poset_kind must be :signature or :dense")
    end

    if cache_eff !== nothing
        key = (UInt(objectid(sig_y)), UInt(objectid(sig_z)), poset_kind)
        Base.lock(cache_eff.lock)
        try
            cache_eff.region_posets[key] = RegionPosetCachePayload(Pnew)
        finally
            Base.unlock(cache_eff.lock)
        end
    end
    return Pnew
end



"""
    measure_by_value(values, pi, opts::InvariantOptions; weights=nothing, kwargs...) -> Dict

Given a vector `values` indexed by regions, return the total region-weight (measure)
for each distinct value.

This is a generalization of `measure_by_dimension`, and supports "locus size" questions:
- "how much parameter space has invariant == v?"
- "how much space lies in each stratum of a region-constant derived invariant?"

Uses fields of `opts`
---------------------
- `opts.box`: required if `weights` not provided (or set `opts.box=:auto`).
- `opts.strict`: if not `nothing`, forwarded to `region_weights` as `strict=...`.

Keywords
--------
- `weights`: optionally provide precomputed region weights.
- `kwargs...`: forwarded to `region_weights(pi; ...)` when weights are not provided.
  Do not pass `box` here. Avoid passing `strict` here; use `opts.strict` instead.

Returns
-------
A dictionary `Dict{V,T}` where:
- `V = eltype(values)`
- `T = promote_type(eltype(weights), Int)`
"""
function measure_by_value(values::AbstractVector, pi, opts::InvariantOptions; weights=nothing, kwargs...)
    w = if weights === nothing
        opts.box === nothing && error("measure_by_value: provide opts.box (or opts.box=:auto) or pass weights=...")
        bb = _resolve_box(pi, opts.box)
        if opts.strict === nothing
            region_weights(pi; box=bb, kwargs...)
        else
            region_weights(pi; box=bb, strict=opts.strict, kwargs...)
        end
    else
        weights
    end
    length(w) == length(values) || error("measure_by_value: weights length does not match values length")

    T = promote_type(eltype(w), Int)
    out = Dict{eltype(values), T}()
    zT = zero(T)
    @inbounds for i in eachindex(values)
        v = values[i]
        out[v] = get(out, v, zT) + w[i]
    end
    return out
end

function measure_by_value(values::AbstractVector, v, pi, opts::InvariantOptions; weights=nothing, kwargs...)
    out = measure_by_value(values, pi, opts; weights=weights, kwargs...)
    T = eltype(Base.values(out))
    return get(out, v, zero(T))
end


"""
    measure_by_value(f, pi, opts::InvariantOptions; arg=:rep, weights=nothing, kwargs...) -> Dict

Convenience wrapper:
- compute `values = region_values(pi, f; arg=arg)`
- then call `measure_by_value(values, pi, opts; ...)`

Uses fields of `opts`
---------------------
- `opts.box`, `opts.strict` (passed to the underlying `measure_by_value(values, ...)`).
"""
function measure_by_value(f::Function, pi, opts::InvariantOptions; arg::Symbol=:rep, weights=nothing, kwargs...)
    vals = region_values(pi, f; arg=arg)
    return measure_by_value(vals, pi, opts; weights=weights, kwargs...)
end



"""
    support_measure(mask, pi, opts::InvariantOptions; weights=nothing, kwargs...) -> Real

Measure the union of a subset of regions, given as a boolean mask.

Uses opts.box and opts.strict.

This complements `support_measure(M, pi)` which measures the parameter-space
subset {dim M(x) >= 1}. Here, you provide the subset directly.
"""
function support_measure(mask::AbstractVector{Bool}, pi, opts::InvariantOptions; weights=nothing, kwargs...)
    w = if weights === nothing
        bb = _resolve_box(pi, opts.box)
        if opts.strict === nothing
            region_weights(pi; box=bb, kwargs...)
        else
            region_weights(pi; box=bb, strict=opts.strict, kwargs...)
        end
    else
        weights
    end

    length(mask) == length(w) || error("support_measure(mask,...): mask length does not match weights length")

    T = eltype(w)
    s = zero(T)
    @inbounds for i in eachindex(mask)
        if mask[i]
            s += w[i]
        end
    end
    return s
end


"""
    vertex_set_measure(vertices, pi, opts::InvariantOptions; weights=nothing, kwargs...) -> Real

Measure the union of a subset of regions, given as a list of region indices.

This is intentionally not called `support_measure`, because `support_measure`
already supports passing a vector of Hilbert values (dimensions) in place of a
module, and those are also integer vectors. We avoid dispatch ambiguity by using
a distinct name.

Duplicate indices are ignored.

Uses fields of `opts`
---------------------
- `opts.box`: required if `weights` not provided (or set `opts.box=:auto`).
- `opts.strict`: if not `nothing`, forwarded to `region_weights` as `strict=...`.

Keywords
--------
- `weights`: optionally provide precomputed region weights.
- `kwargs...`: forwarded to `region_weights(pi; ...)` when weights are not provided.
"""
function vertex_set_measure(vertices::AbstractVector{<:Integer}, pi, opts::InvariantOptions; weights=nothing, kwargs...)
    w = if weights === nothing
        opts.box === nothing && error("vertex_set_measure: provide opts.box (or opts.box=:auto) or pass weights=...")
        bb = _resolve_box(pi, opts.box)
        if opts.strict === nothing
            region_weights(pi; box=bb, kwargs...)
        else
            region_weights(pi; box=bb, strict=opts.strict, kwargs...)
        end
    else
        weights
    end

    n = length(w)
    T = promote_type(eltype(w), Int)
    s = zero(T)

    # Track duplicates so each region is counted at most once.
    seen = falses(n)
    @inbounds for p in vertices
        1 <= p <= n || error("vertex_set_measure: vertex index $p out of range 1:$n")
        if !seen[p]
            s += w[p]
            seen[p] = true
        end
    end
    return s
end



# Internal helper: support and mass measures for multigraded tables indexed by (deg, vertex).
function _multigraded_support_measures(tbl::Dict{Tuple{Int,Int},<:Integer}, w)
    n = length(w)
    T = promote_type(eltype(w), Int)

    # Determine degrees present.
    dmax = -1
    for (k, _) in tbl
        d = k[1]
        d > dmax && (dmax = d)
    end
    if dmax < 0
        return (support_by_degree=zeros(T, 0), mass_by_degree=zeros(T, 0), support_union=zero(T), support_total=zero(T), mass_total=zero(T))
    end

    support_by = zeros(T, dmax + 1)
    mass_by = zeros(T, dmax + 1)
    seen_union = falses(n)
    su = zero(T)

    @inbounds for ((d, p), m) in tbl
        0 <= d <= dmax || continue
        1 <= p <= n || error("multigraded_support: vertex index $p out of range 1:$n")
        wp = w[p]
        if !seen_union[p]
            seen_union[p] = true
            su += wp
        end
        if m != 0
            support_by[d + 1] += wp
            mass_by[d + 1] += wp * T(m)
        end
    end

    mt = sum(mass_by)
    return (support_by_degree=support_by, mass_by_degree=mass_by, support_union=su, support_total=mt, mass_total=mt)
end

function _multigraded_support_measures(B::AbstractMatrix{<:Integer}, w)
    n = length(w)
    size(B, 2) == n || error("multigraded_support: matrix columns must match length(weights)")
    r = size(B, 1)
    T = promote_type(eltype(w), Int)
    support_by = zeros(T, r)
    mass_by = zeros(T, r)
    su = zero(T)

    # For each vertex p, scan all degrees and accumulate.
    @inbounds for p in 1:n
        wp = w[p]
        anynz = false
        for a in 1:r
            m = B[a, p]
            if m != 0
                anynz = true
                support_by[a] += wp
                mass_by[a] += wp * T(m)
            end
        end
        anynz && (su += wp)
    end

    mt = sum(mass_by)
    return (support_by_degree=support_by, mass_by_degree=mass_by, support_union=su, support_total=mt, mass_total=mt)
end


"""
    betti_support_measures(B, pi, opts::InvariantOptions; weights=nothing, kwargs...) -> BettiSupportMeasuresSummary

Compute region-size measures of where Betti numbers are supported.

Input `B` may be:
- a `Dict{Tuple{Int,Int},Int}` as returned by `betti(res)`,
- a dense matrix as returned by `betti_table(res)`,
- any object accepted by `betti` (e.g. a `ProjectiveResolution`).

The returned `BettiSupportMeasuresSummary` contains:
- `support_by_degree`: entry a+1 is the measure of regions p with a nonzero Betti
  number in homological degree a.
- `mass_by_degree`: entry a+1 is sum_p beta_{a,p} * w[p].
- `support_union`: measure of the union of all Betti-support vertices.
- `mass_total`: total multiplicity-weighted measure across all degrees.

Uses fields of `opts`
---------------------
- `opts.box`: required if `weights` not provided (or set `opts.box=:auto`).
- `opts.strict`: if not `nothing`, forwarded to `region_weights` as `strict=...`.

Keywords
--------
- `weights`: optionally provide precomputed region weights.
- `kwargs...`: forwarded to `region_weights(pi; ...)` when weights are not provided.
"""
function betti_support_measures(B, pi, opts::InvariantOptions; weights=nothing, kwargs...)
    w = if weights === nothing
        opts.box === nothing && error("betti_support_measures: provide opts.box (or opts.box=:auto) or pass weights=...")
        bb = _resolve_box(pi, opts.box)
        if opts.strict === nothing
            region_weights(pi; box=bb, kwargs...)
        else
            region_weights(pi; box=bb, strict=opts.strict, kwargs...)
        end
    else
        weights
    end

    if B isa Dict{Tuple{Int,Int},<:Integer}
        return BettiSupportMeasuresSummary(_multigraded_support_measures(B, w))
    elseif B isa AbstractMatrix{<:Integer}
        return BettiSupportMeasuresSummary(_multigraded_support_measures(B, w))
    else
        # Fall back: interpret B as a resolution-like object.
        return betti_support_measures(betti(B), pi, opts; weights=weights, kwargs...)
    end
end


"""
    bass_support_measures(B, pi, opts::InvariantOptions; weights=nothing, kwargs...) -> BassSupportMeasuresSummary

Analog of `betti_support_measures` for multigraded Bass numbers.

Input `B` may be:
- a `Dict{Tuple{Int,Int},Int}` as returned by `bass(res)`,
- a dense matrix as returned by `bass_table(res)`,
- any object accepted by `bass` (e.g. an `InjectiveResolution`).

Uses fields of `opts`
---------------------
- `opts.box`, `opts.strict` (as in `betti_support_measures`).
"""
function bass_support_measures(B, pi, opts::InvariantOptions; weights=nothing, kwargs...)
    w = if weights === nothing
        opts.box === nothing && error("bass_support_measures: provide opts.box (or opts.box=:auto) or pass weights=...")
        bb = _resolve_box(pi, opts.box)
        if opts.strict === nothing
            region_weights(pi; box=bb, kwargs...)
        else
            region_weights(pi; box=bb, strict=opts.strict, kwargs...)
        end
    else
        weights
    end

    if B isa Dict{Tuple{Int,Int},<:Integer}
        return BassSupportMeasuresSummary(_multigraded_support_measures(B, w))
    elseif B isa AbstractMatrix{<:Integer}
        return BassSupportMeasuresSummary(_multigraded_support_measures(B, w))
    else
        return bass_support_measures(bass(B), pi, opts; weights=weights, kwargs...)
    end
end
