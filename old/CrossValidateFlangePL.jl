module CrossValidateFlangePL
# ------------------------------------------------------------------------------
# Cross-validate a Z^n flange against an axis-aligned PL evaluation on the same
# lattice points in a convex-projection box [a,b].  No external Axis* module.
# ------------------------------------------------------------------------------

using LinearAlgebra
import ..FlangeZn: Flange, bounding_box, dim_at
using ..CoreModules: QQ            # canonical exact rational scalar
import ..ExactQQ: rankQQ           # exact rank over QQ

# Minimal axis-aligned shapes for cross-check (not exported elsewhere).
# IMPORTANT: we do not emulate +/- infinity with sentinels.  Instead we
# carry an explicit "free" bitmask that marks coordinates whose bound is
# ignored.  This keeps membership tests exact and avoids magic numbers.
struct AxisUpset{T}
    # a[i] is only used when free[i] == false; otherwise it is a dummy.
    a::Vector{T}      # lower thresholds
    free::BitVector   # coordinates that are free in Z^tau
end
struct AxisDownset{T}
    # b[i] is only used when free[i] == false; otherwise it is a dummy.
    b::Vector{T}      # upper thresholds
    free::BitVector   # coordinates that are free in Z^tau
end


struct AxisFringe{TU,TD,TF}
     n::Int
     births::Vector{AxisUpset{TU}}
     deaths::Vector{AxisDownset{TD}}
     Phi::Matrix{TF}  # same scalar "monomial" matrix
 end

"Translate Zn indecomposables to axis-aligned PL up/down sets for cross-checking."
function flange_to_axis(fr::Flange{K}) where {K}
    # Keep everything over QQ to avoid floating-point artifacts.
    n = fr.n
    births = AxisUpset{QQ}[]
    for F in fr.flats
        a = Vector{QQ}(undef, n)
        for i in 1:n
            # When F.tau.coords[i] is true (free coordinate), a[i] is never read.
            # We still store a QQ value to keep the vector concrete.
            a[i] = QQ(F.b[i])
        end
        push!(births, AxisUpset{QQ}(a, BitVector(F.tau.coords)))
    end
    deaths = AxisDownset{QQ}[]
    for E in fr.injectives
        b = Vector{QQ}(undef, n)
        for i in 1:n
            # Same comment as above: b[i] is ignored when the coordinate is free.
            b[i] = QQ(E.b[i])
        end
        push!(deaths, AxisDownset{QQ}(b, BitVector(E.tau.coords)))
    end
    # DO NOT coerce Phi to Float64; keep it exact over QQ.
    Phi = Matrix{QQ}(fr.Phi)
    return AxisFringe{QQ,QQ,QQ}(n, births, deaths, Phi)
end

"Test membership of a lattice point x in an axis-aligned upset or downset (generic and exact)."
_contains(U::AxisUpset{T}, x::Vector{T}) where {T} =
    all(U.free[i] || (x[i] >= U.a[i]) for i in 1:length(x))
_contains(D::AxisDownset{T}, x::Vector{T}) where {T} =
    all(D.free[i] || (x[i] <= D.b[i]) for i in 1:length(x))

"""
    cross_validate(fr::Flange; margin=1, rankfun=rankQQ)

1. Build the convex-projection box [a,b] (heuristic 'bounding_box').
2. Evaluate 'dim M_g' for all integer 'g in [a,b]' via the flange.
3. Build an axis-aligned proxy and evaluate the same lattice points.
4. Compare; return '(all_equal?, report::Dict)'.
"""
function cross_validate(fr::Flange; margin=1, rankfun=rankQQ)
    a, b = bounding_box(fr; margin)
    ranges = (a[i]:b[i] for i in 1:fr.n)
    pts = collect(Iterators.product(ranges...))

    # Flange evaluation (exact over QQ if rankfun = rankQQ).
    dims_Z = Dict{Tuple{Vararg{Int}}, Int}()
    for t in pts
        g = collect(Int.(t))
        dims_Z[Tuple(g...)] = dim_at(fr, g; rankfun=rankfun)
    end

    # Axis-aligned proxy (stay exact; compare apples-to-apples with the flange)
    afr = flange_to_axis(fr)
    dims_PL = Dict{Tuple{Vararg{Int}}, Int}()
    for t in pts
        # Use rationals for coordinates to make membership tests exact.
        x = QQ.(collect(Int.(t)))
        rows = [ _contains(d, x) for d in afr.deaths ]
        cols = [ _contains(u, x) for u in afr.births ]
        idxr = findall(identity, rows)
        idxc = findall(identity, cols)
        if isempty(idxr) || isempty(idxc)
            dims_PL[Tuple(Int.(t)...)] = 0
        else
            # Rank exactly with the same provided rank function.
            Phi_x = afr.Phi[idxr, idxc]
            dims_PL[Tuple(Int.(t)...)] = rankfun(Phi_x)
        end
    end

    # Compare
    mism = Dict{Tuple{Vararg{Int}}, Tuple{Int,Int}}()
    for (k,v) in dims_Z
        if v != dims_PL[k]; mism[k] = (v, dims_PL[k]); end
    end
    ok = isempty(mism)
    report = Dict("box" => (a,b), "mismatches" => mism,
                  "tested" => length(pts), "agree" => length(pts) - length(mism))
    return ok, report
end

export cross_validate, flange_to_axis
end # module
