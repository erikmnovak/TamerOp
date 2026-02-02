module AbelianCategories

"""
    AbelianCategories

This module collects basic abelian-category constructions for the category of
finite-poset modules (PModule / PMorphism).

Design intent
- Keep these constructions independent of the indicator-resolution algorithms.
- Provide a single home for kernels/cokernels/images, pushouts/pullbacks,
  short exact sequences, and snake lemma output.
- Keep code ASCII-only.

Most routines are currently implemented over QQ (ExactQQ.QQ) because they rely
on exact linear algebra (rank / nullspace / column space).

Downstream modules (DerivedFunctors, ModuleComplexes, ChangeOfPosets, ...)
should depend on this module rather than IndicatorResolutions when they only
need general categorical operations.
"""

using LinearAlgebra
using SparseArrays

import ..FiniteFringe
using ..FiniteFringe: cover_edges, nvertices

using ..CoreModules: QQ
using ..ExactQQ: rrefQQ, rankQQ, nullspaceQQ, solve_fullcolumnQQ, colspaceQQ

using ..Modules: CoverCache, cover_cache,
                 CoverEdgeMapStore, _find_sorted_index,
                 PModule, PMorphism, id_morphism,
                 direct_sum_with_maps

# Public API (exported to PosetModules via `using .AbelianCategories`).
export kernel_with_inclusion, kernel,
       image_with_inclusion, image,
       cokernel_with_projection, cokernel,
       coimage_with_projection, coimage,
       quotient_with_projection, quotient,
       is_zero_morphism, is_monomorphism, is_epimorphism,
       Submodule, submodule, sub, ambient, inclusion,
       kernel_submodule, image_submodule,
       pushout, pullback,
       ShortExactSequence, short_exact_sequence, is_exact, assert_exact,
       snake_lemma, SnakeLemmaResult,
       biproduct, product, coproduct, equalizer, coequalizer,
       AbstractDiagram, DiscretePairDiagram, ParallelPairDiagram, SpanDiagram, CospanDiagram,
       limit, colimit

# --------------------------- kernel and upset presentation --------------------------

"Kernel of f with inclusion iota : ker(f) to dom(f), degreewise."
function kernel_with_inclusion(f::PMorphism{QQ}; cache::Union{Nothing,CoverCache}=nothing)
    M = f.dom
    n = nvertices(M.Q)

    basisK = Vector{Matrix{QQ}}(undef, n)
    K_dims = zeros(Int, n)
    for i in 1:n
        B = nullspaceQQ(f.comps[i])
        basisK[i] = B
        K_dims[i] = size(B, 2)
    end

    # Build the kernel's structure maps directly in store-aligned form.
    cc = (cache === nothing ? cover_cache(M.Q) : cache)
    preds = cc.preds
    succs = cc.succs

    maps_from_pred = [Vector{Matrix{QQ}}(undef, length(preds[v])) for v in 1:n]
    maps_to_succ   = [Vector{Matrix{QQ}}(undef, length(succs[u])) for u in 1:n]

    @inbounds for u in 1:n
        su = succs[u]
        maps_u_M = M.edge_maps.maps_to_succ[u]   # aligned with su
        outu = maps_to_succ[u]

        for j in eachindex(su)
            v = su[j]

            # If either stalk is 0, the induced map is the unique 0 map.
            if K_dims[u] == 0 || K_dims[v] == 0
                X = zeros(QQ, K_dims[v], K_dims[u])
                outu[j] = X
                ip = _find_sorted_index(preds[v], u)
                maps_from_pred[v][ip] = X
                continue
            end

            # Induced map K(u) -> K(v): express M(u->v)*basisK[u] in basisK[v].
            T  = maps_u_M[j]
            Im = T * basisK[u]
            X  = solve_fullcolumnQQ(basisK[v], Im; check_rhs=false)

            outu[j] = X
            ip = _find_sorted_index(preds[v], u)
            maps_from_pred[v][ip] = X
        end
    end

    storeK = CoverEdgeMapStore{QQ,Matrix{QQ}}(preds, succs, maps_from_pred, maps_to_succ, cc.nedges)
    K = PModule{QQ,Matrix{QQ}}(M.Q, K_dims, storeK)

    iota = PMorphism{QQ}(K, M, [basisK[i] for i in 1:n])
    return K, iota
end


# ----------------------------
# Image with inclusion (dual to kernel_with_inclusion)
# ----------------------------

"""
    image_with_inclusion(f::PMorphism{QQ}) -> (Im, iota)

Compute the image submodule Im subseteq cod(f) with the inclusion morphism iota: Im -> cod(f).
"""
function image_with_inclusion(f::PMorphism{QQ}; cache::Union{Nothing,CoverCache}=nothing)
    N = f.cod
    Q = N.Q
    n = nvertices(Q)

    bases = Vector{Matrix{QQ}}(undef, n)
    dims  = zeros(Int, n)

    for i in 1:n
        B = colspaceQQ(f.comps[i])
        bases[i] = B
        dims[i]  = size(B, 2)
    end

    # Build the image's structure maps directly in store-aligned form.
    storeN = N.edge_maps
    preds  = storeN.preds
    succs  = storeN.succs

    maps_from_pred = [Vector{Matrix{QQ}}(undef, length(preds[v])) for v in 1:n]
    maps_to_succ   = [Vector{Matrix{QQ}}(undef, length(succs[u])) for u in 1:n]

    @inbounds for u in 1:n
        su = succs[u]
        Nu = storeN.maps_to_succ[u]
        outu = maps_to_succ[u]

        Bu = bases[u]
        du = size(Bu, 2)

        for j in eachindex(su)
            v = su[j]
            Bv = bases[v]
            dv = size(Bv, 2)

            # Induced map Im(u) -> Im(v): express N(u->v)*Bu in Bv.
            Auv = if du == 0
                zeros(QQ, dv, 0)
            elseif dv == 0
                zeros(QQ, 0, du)
            else
                T = Nu[j] * Bu
                solve_fullcolumnQQ(Bv, T)
            end

            outu[j] = Auv
            ip = _find_sorted_index(preds[v], u)
            maps_from_pred[v][ip] = Auv
        end
    end

    storeIm = CoverEdgeMapStore{QQ,Matrix{QQ}}(preds, succs, maps_from_pred, maps_to_succ, storeN.nedges)
    Im = PModule{QQ,Matrix{QQ}}(Q, dims, storeIm)

    iota = PMorphism(Im, N, [bases[i] for i in 1:n])
    return Im, iota
end

# Degreewise cokernel of iota : E0 <- M, produced as a P-module C together with
# the quotient q : E0 -> C.  The quotient is represented by surjections q_i whose
# kernels are colspace(iota_i).
function _cokernel_module(iota::PMorphism{QQ}; cache::Union{Nothing,CoverCache}=nothing)
    E = iota.cod; Q = E.Q; n = nvertices(Q)
    Cdims  = zeros(Int, n)
    qcomps = Vector{Matrix{QQ}}(undef, n)     # each is (dim C_i) x (dim E_i)

    # degreewise quotients
    for i in 1:n
        Bi = colspaceQQ(iota.comps[i])        # dim E_i x rank
        Ni = nullspaceQQ(transpose(Bi))       # dim E_i x (dim E_i - rank)
        Cdims[i]  = size(Ni, 2)
        qcomps[i] = transpose(Ni)
    end

    # structure maps of C
    Cedges = Dict{Tuple{Int,Int}, Matrix{QQ}}()
    cc = (cache === nothing ? cover_cache(Q) : cache)

    @inbounds for u in 1:n
        su     = cc.succs[u]
        maps_u = E.edge_maps.maps_to_succ[u]   # aligned with su

        for j in eachindex(su)
            v = su[j]

            if Cdims[u] > 0 && Cdims[v] > 0
                T = maps_u[j]  # E_u -> E_v along this cover edge

                # Induced quotient map: enforce q_v * T = A * q_u.
                X = solve_fullcolumnQQ(transpose(qcomps[u]), transpose(qcomps[v] * T))
                Cedges[(u, v)] = transpose(X)  # dim C_v x dim C_u
            else
                Cedges[(u, v)] = zeros(QQ, Cdims[v], Cdims[u])
            end
        end
    end

    Cmod = PModule{QQ}(Q, Cdims, Cedges)
    q = PMorphism{QQ}(E, Cmod, qcomps)
    return Cmod, q
end

# =============================================================================
# Abelian category API (kernels/cokernels/images/coimages/quotients)
# =============================================================================

"""
    cokernel_with_projection(f; cache=nothing) -> (C, q)

Compute the cokernel of a morphism f : A -> B as the quotient module C = B / im(f),
together with the quotient map q : B -> C.

In this functor category (finite poset modules over a field), cokernels are computed
pointwise (vertexwise) and the structure maps of the quotient are induced.

This is the dual companion of `kernel_with_inclusion`.
"""
function cokernel_with_projection(f::PMorphism{QQ}; cache::Union{Nothing,CoverCache}=nothing)
    return _cokernel_module(f; cache=cache)
end

# ---------------------------------------------------------------------------
# Equalizers / coequalizers (abelian category = kernels / cokernels of f - g)
# ---------------------------------------------------------------------------

# Internal: difference of parallel maps (same dom/cod).
function _difference_morphism(f::PMorphism{QQ}, g::PMorphism{QQ})
    if f.dom !== g.dom || f.cod !== g.cod
        error("need parallel morphisms with the same domain and codomain")
    end
    comps = Matrix{QQ}[f.comps[i] - g.comps[i] for i in 1:nvertices(f.dom.Q)]
    return PMorphism{QQ}(f.dom, f.cod, comps)
end

"""
    equalizer(f, g; cache=nothing) -> (E, e)

Equalizer of parallel morphisms `f, g : A -> B`.

In an abelian category, the equalizer is `ker(f - g)` with inclusion map
`e : E -> A`.

Returns:
- `E` : the equalizer object
- `e` : the inclusion `E -> A`

See also: `coequalizer`, `kernel_with_inclusion`.
"""
function equalizer(f::PMorphism{QQ}, g::PMorphism{QQ}; cache=nothing)
    h = _difference_morphism(f, g)
    return kernel_with_inclusion(h; cache=cache)
end

"""
    coequalizer(f, g; cache=nothing) -> (Q, q)

Coequalizer of parallel morphisms `f, g : A -> B`.

In an abelian category, the coequalizer is `coker(f - g)` with projection map
`q : B -> Q`.

Returns:
- `Q` : the coequalizer object
- `q` : the projection `B -> Q`

See also: `equalizer`, `cokernel_with_projection`.
"""
function coequalizer(f::PMorphism{QQ}, g::PMorphism{QQ}; cache=nothing)
    h = _difference_morphism(f, g)
    return cokernel_with_projection(h; cache=cache)
end


"""
    cokernel(f; cache=nothing) -> C

Return only the cokernel module.
"""
cokernel(f::PMorphism{QQ}; cache::Union{Nothing,CoverCache}=nothing) =
    (cokernel_with_projection(f; cache=cache))[1]

"""
    kernel(f; cache=nothing) -> K

Return only the kernel module.
"""
kernel(f::PMorphism{QQ}; cache::Union{Nothing,CoverCache}=nothing) =
    (kernel_with_inclusion(f; cache=cache))[1]

"""
    image(f) -> Im

Return only the image module (a submodule of `codomain(f)`).
Use `image_with_inclusion` if you also want the inclusion map `Im -> codomain(f)`.
"""
image(f::PMorphism{QQ}) = (image_with_inclusion(f))[1]

"""
    coimage_with_projection(f; cache=nothing) -> (Coim, p)

Compute the coimage of f : A -> B, defined as Coim = A / ker(f),
together with the canonical projection p : A -> Coim.

In an abelian category, the canonical map Coim -> Im is an isomorphism.
"""
function coimage_with_projection(f::PMorphism{QQ}; cache::Union{Nothing,CoverCache}=nothing)
    K, iK = kernel_with_inclusion(f; cache=cache)
    Coim, p = _cokernel_module(iK; cache=cache)
    return Coim, p
end

coimage(f::PMorphism{QQ}; cache::Union{Nothing,CoverCache}=nothing) =
    (coimage_with_projection(f; cache=cache))[1]

# Quotients of submodules: for now we represent a submodule by its inclusion morphism.
"""
    quotient_with_projection(iota; cache=nothing) -> (Q, q)

Given a monomorphism iota : N -> M (typically an inclusion), compute the quotient module
Q = M / N together with the projection q : M -> Q.

This is an alias for `cokernel_with_projection(iota)`.
"""
function quotient_with_projection(iota::PMorphism{QQ}; cache::Union{Nothing,CoverCache}=nothing)
    return cokernel_with_projection(iota; cache=cache)
end

"""
    quotient(iota; cache=nothing) -> Q

Return only the quotient module M/N for an inclusion iota : N -> M.
"""
quotient(iota::PMorphism{QQ}; cache::Union{Nothing,CoverCache}=nothing) =
    (quotient_with_projection(iota; cache=cache))[1]

# Small predicates

"""
    is_zero_morphism(f) -> Bool

Return true if all components of f are the zero matrix.
"""
function is_zero_morphism(f::PMorphism{K}) where {K}
    for A in f.comps
        all(iszero, A) || return false
    end
    return true
end

"""
    is_monomorphism(f) -> Bool

Check whether f is a monomorphism in the functor category of P-modules.
For these representations, this is equivalent to f_i being injective for every vertex i.

This method is implemented for QQ using exact rank computations.
"""
function is_monomorphism(f::PMorphism{QQ})
    Q = f.dom.Q
    @assert f.cod.Q === Q
    for i in 1:nvertices(Q)
        if rankQQ(f.comps[i]) != f.dom.dims[i]
            return false
        end
    end
    return true
end

"""
    is_epimorphism(f) -> Bool

Check whether f is an epimorphism (pointwise surjective).
Implemented for QQ via exact ranks.
"""
function is_epimorphism(f::PMorphism{QQ})
    Q = f.dom.Q
    @assert f.cod.Q === Q
    for i in 1:nvertices(Q)
        if rankQQ(f.comps[i]) != f.cod.dims[i]
            return false
        end
    end
    return true
end


# -----------------------------------------------------------------------------
# Submodules as first-class objects
# -----------------------------------------------------------------------------

"""
    Submodule(incl)

A lightweight wrapper around an inclusion morphism `incl : N -> M` representing the submodule
N <= M. The ambient module is `ambient(S)` and the underlying module is `sub(S)`.

This wrapper is intentionally minimal: it stores only the inclusion map.
"""
struct Submodule{K}
    incl::PMorphism{K}  # incl : sub -> ambient
end

"""
    submodule(incl; check_mono=true) -> Submodule

Build a `Submodule` from an inclusion map.

If `check_mono=true`, verify that `incl` is a monomorphism (QQ only).
"""
function submodule(incl::PMorphism{QQ}; check_mono::Bool=true)
    check_mono && !is_monomorphism(incl) && error("submodule: given inclusion is not a monomorphism")
    return Submodule{QQ}(incl)
end
function submodule(incl::PMorphism{K}; check_mono::Bool=false) where {K}
    check_mono && error("submodule: check_mono is only implemented for QQ")
    return Submodule{K}(incl)
end

"Underlying submodule N (as a P-module) for a `Submodule` N <= M."
@inline sub(S::Submodule) = S.incl.dom

"Ambient module M for a `Submodule` N <= M."
@inline ambient(S::Submodule) = S.incl.cod

"Inclusion map N -> M for a `Submodule`."
@inline inclusion(S::Submodule) = S.incl

"""
    quotient_with_projection(S::Submodule; cache=nothing) -> (Q, q)
    quotient(S::Submodule; cache=nothing) -> Q

Compute the quotient M/N given a submodule S representing N <= M.
"""
quotient_with_projection(S::Submodule{QQ}; cache::Union{Nothing,CoverCache}=nothing) =
    quotient_with_projection(S.incl; cache=cache)
quotient(S::Submodule{QQ}; cache::Union{Nothing,CoverCache}=nothing) =
    quotient(S.incl; cache=cache)

"""
    quotient_with_projection(M, S::Submodule; cache=nothing) -> (Q, q)
    quotient(M, S::Submodule; cache=nothing) -> Q

Convenience methods matching the common mathematical notation `M/N`.
These verify that `ambient(S) === M`.
"""
function quotient_with_projection(M::PModule{QQ}, S::Submodule{QQ}; cache::Union{Nothing,CoverCache}=nothing)
    ambient(S) === M || error("quotient_with_projection: submodule is not a submodule of the given ambient module")
    return quotient_with_projection(S; cache=cache)
end
quotient(M::PModule{QQ}, S::Submodule{QQ}; cache::Union{Nothing,CoverCache}=nothing) =
    (quotient_with_projection(M, S; cache=cache))[1]

"""
    quotient_with_projection(M, iota; cache=nothing) -> (Q, q)
    quotient(M, iota; cache=nothing) -> Q

Convenience overloads where the submodule is given as an inclusion morphism iota : N -> M.
"""
function quotient_with_projection(M::PModule{QQ}, iota::PMorphism{QQ}; cache::Union{Nothing,CoverCache}=nothing)
    iota.cod === M || error("quotient_with_projection: inclusion morphism does not target the given ambient module")
    return quotient_with_projection(iota; cache=cache)
end
quotient(M::PModule{QQ}, iota::PMorphism{QQ}; cache::Union{Nothing,CoverCache}=nothing) =
    (quotient_with_projection(M, iota; cache=cache))[1]

"""
    kernel_submodule(f; cache=nothing) -> Submodule

Return ker(f) <= dom(f) as a `Submodule`. (The inclusion is provided by `kernel_with_inclusion`.)
"""
function kernel_submodule(f::PMorphism{QQ}; cache::Union{Nothing,CoverCache}=nothing)
    K, iota = kernel_with_inclusion(f; cache=cache)
    return Submodule{QQ}(iota)
end

"""
    image_submodule(f) -> Submodule

Return im(f) <= cod(f) as a `Submodule`.
"""
function image_submodule(f::PMorphism{QQ})
    Im, iota = image_with_inclusion(f)
    return Submodule{QQ}(iota)
end

# -----------------------------------------------------------------------------
# Pushouts and pullbacks
# -----------------------------------------------------------------------------

# Internal: compute a right inverse for a full-row-rank matrix Q (r x m), so Q * rinv = I_r.
function _right_inverse_full_rowQQ(Q::Matrix{QQ})
    r, m = size(Q)
    if r == 0
        return zeros(QQ, m, 0)
    end
    G = Q * transpose(Q)  # r x r, invertible if Q has full row rank
    invG = solve_fullcolumnQQ(G, Matrix{QQ}(I, r, r))
    return transpose(Q) * invG  # m x r
end

"""
    pushout(f, g; cache=nothing) -> (P, inB, inC, q, phi)

Compute the pushout of a span A --f--> B and A --g--> C.

Construction:
    P = (B oplus C) / im( (f, -g) : A -> B oplus C )

Returns:
- P : the pushout module
- inB : B -> P
- inC : C -> P
- q : (B oplus C) -> P (the quotient map)
- phi : A -> (B oplus C) (the map whose cokernel defines the pushout)

The maps satisfy inB o f == inC o g.
"""
function pushout(f::PMorphism{QQ}, g::PMorphism{QQ}; cache::Union{Nothing,CoverCache}=nothing)
    @assert f.dom === g.dom
    A = f.dom
    B = f.cod
    C = g.cod
    S, iB, iC, pB, pC = direct_sum_with_maps(B, C)
    Q = S.Q
    # phi = iB o f - iC o g : A -> B oplus C
    phi_comps = Vector{Matrix{QQ}}(undef, nvertices(Q))
    for u in 1:nvertices(Q)
        phi_comps[u] = iB.comps[u] * f.comps[u] - iC.comps[u] * g.comps[u]
    end
    phi = PMorphism{QQ}(A, S, phi_comps)

    P, q = _cokernel_module(phi; cache=cache)

    # inB = q o iB, inC = q o iC
    inB_comps = Vector{Matrix{QQ}}(undef, nvertices(Q))
    inC_comps = Vector{Matrix{QQ}}(undef, nvertices(Q))
    for u in 1:nvertices(Q)
        inB_comps[u] = q.comps[u] * iB.comps[u]
        inC_comps[u] = q.comps[u] * iC.comps[u]
    end
    inB = PMorphism{QQ}(B, P, inB_comps)
    inC = PMorphism{QQ}(C, P, inC_comps)

    return P, inB, inC, q, phi
end

# ---------------------------------------------------------------------------
# Diagram objects + limit/colimit dispatch (small but useful categorical layer)
# ---------------------------------------------------------------------------

"""
Abstract supertype for small diagram objects valued in `PModule{K}`.

This is intentionally minimal: it supports the common finite shapes needed in
everyday categorical algebra:
- discrete pair (product/coproduct)
- parallel pair (equalizer/coequalizer)
- span (pushout)
- cospan (pullback)
"""
abstract type AbstractDiagram{K} end

"""
    DiscretePairDiagram(A, B)

The discrete diagram on two objects `A` and `B` (no arrows).
"""
struct DiscretePairDiagram{K} <: AbstractDiagram{K}
    A::PModule{K}
    B::PModule{K}
end

"""
    ParallelPairDiagram(f, g)

A parallel pair of morphisms `f, g : A -> B`.
"""
struct ParallelPairDiagram{K} <: AbstractDiagram{K}
    f::PMorphism{K}
    g::PMorphism{K}
end

"""
    SpanDiagram(f, g)

A span-shaped diagram `A --f--> B` and `A --g--> C`.

Its colimit is the pushout.
"""
struct SpanDiagram{K} <: AbstractDiagram{K}
    f::PMorphism{K}  # A -> B
    g::PMorphism{K}  # A -> C
end

"""
    CospanDiagram(f, g)

A cospan-shaped diagram `B --f--> D` and `C --g--> D`.

Its limit is the pullback.
"""
struct CospanDiagram{K} <: AbstractDiagram{K}
    f::PMorphism{K}  # B -> D
    g::PMorphism{K}  # C -> D
end

"""
    limit(D; cache=nothing)

Compute a limit of a supported small diagram object.

Supported shapes:
- `DiscretePairDiagram`: product
- `ParallelPairDiagram` (over QQ): equalizer
- `CospanDiagram` (over QQ): pullback
"""
function limit(D::DiscretePairDiagram{K}; cache=nothing) where {K}
    return product(D.A, D.B)
end

function limit(D::ParallelPairDiagram{QQ}; cache=nothing)
    return equalizer(D.f, D.g; cache=cache)
end

function limit(D::CospanDiagram{QQ}; cache=nothing)
    return pullback(D.f, D.g; cache=cache)
end

"""
    colimit(D; cache=nothing)

Compute a colimit of a supported small diagram object.

Supported shapes:
- `DiscretePairDiagram`: coproduct
- `ParallelPairDiagram` (over QQ): coequalizer
- `SpanDiagram` (over QQ): pushout
"""
function colimit(D::DiscretePairDiagram{K}; cache=nothing) where {K}
    return coproduct(D.A, D.B)
end

function colimit(D::ParallelPairDiagram{QQ}; cache=nothing)
    return coequalizer(D.f, D.g; cache=cache)
end

function colimit(D::SpanDiagram{QQ}; cache=nothing)
    return pushout(D.f, D.g; cache=cache)
end


# Internal: solve A*X = B over QQ, returning one particular solution with free variables set to 0.
# This duplicates ChainComplexes.solve_particularQQ but avoids introducing a dependency.
function _solve_particularQQ(A::AbstractMatrix{QQ}, B::AbstractMatrix{QQ})
    A0 = Matrix{QQ}(A)
    B0 = Matrix{QQ}(B)
    m, n = size(A0)
    @assert size(B0, 1) == m

    Aug = hcat(A0, B0)
    R, pivs_all = rrefQQ(Aug)
    rhs = size(B0, 2)

    # consistency check: zero row in A-part with nonzero in RHS-part
    for i in 1:m
        if all(R[i, 1:n] .== 0)
            if any(R[i, n+1:n+rhs] .!= 0)
                error("_solve_particularQQ: inconsistent system")
            end
        end
    end

    pivs = Int[]
    for p in pivs_all
        p <= n && push!(pivs, p)
    end

    X = zeros(QQ, n, rhs)
    for (row, pcol) in enumerate(pivs)
        X[pcol, :] = R[row, n+1:n+rhs]
    end
    return X
end

"""
    pullback(f, g; cache=nothing) -> (P, prB, prC, iota, psi)

Compute the pullback of a cospan B --f--> D and C --g--> D.

Construction:
    P = ker( (f, -g) : B oplus C -> D )

Returns:
- P : the pullback module
- prB : P -> B
- prC : P -> C
- iota : P -> (B oplus C) (the kernel inclusion)
- psi : (B oplus C) -> D (the map whose kernel defines the pullback)

The projections satisfy f o prB == g o prC.
"""
function pullback(f::PMorphism{QQ}, g::PMorphism{QQ}; cache::Union{Nothing,CoverCache}=nothing)
    @assert f.cod === g.cod
    B = f.dom
    C = g.dom
    D = f.cod
    S, iB, iC, pB, pC = direct_sum_with_maps(B, C)
    Q = S.Q

    # psi = f o pB - g o pC : B oplus C -> D
    psi_comps = Vector{Matrix{QQ}}(undef, nvertices(Q))
    for u in 1:nvertices(Q)
        psi_comps[u] = f.comps[u] * pB.comps[u] - g.comps[u] * pC.comps[u]
    end
    psi = PMorphism{QQ}(S, D, psi_comps)

    P, iota = kernel_with_inclusion(psi; cache=cache)  # iota : P -> S

    prB_comps = Vector{Matrix{QQ}}(undef, nvertices(Q))
    prC_comps = Vector{Matrix{QQ}}(undef, nvertices(Q))
    for u in 1:nvertices(Q)
        prB_comps[u] = pB.comps[u] * iota.comps[u]
        prC_comps[u] = pC.comps[u] * iota.comps[u]
    end
    prB = PMorphism{QQ}(P, B, prB_comps)
    prC = PMorphism{QQ}(P, C, prC_comps)

    return P, prB, prC, iota, psi
end

# -----------------------------------------------------------------------------
# Exactness utilities: short exact sequences and snake lemma
# -----------------------------------------------------------------------------

"""
    ShortExactSequence(i, p; check=true, cache=nothing)

Record a short exact sequence 0 -> A --i--> B --p--> C -> 0.

If `check=true` (default), we verify:
- p o i = 0
- i is a monomorphism (pointwise injective)
- p is an epimorphism (pointwise surjective)
- im(i) = ker(p) as submodules of B (pointwise equality)

The check is exact over QQ.

This object caches `ker(p)` and `im(i)` once computed, since many downstream
constructions (Ext/Tor LES, snake lemma, etc.) use them repeatedly.
"""
mutable struct ShortExactSequence{K}
    A::PModule{K}
    B::PModule{K}
    C::PModule{K}
    i::PMorphism{K}
    p::PMorphism{K}
    checked::Bool
    exact::Bool
    ker_p::Union{Nothing,Tuple{PModule{K},PMorphism{K}}}
    img_i::Union{Nothing,Tuple{PModule{K},PMorphism{K}}}
end

function ShortExactSequence(i::PMorphism{QQ}, p::PMorphism{QQ};
                           check::Bool=true,
                           cache::Union{Nothing,CoverCache}=nothing)
    @assert i.cod === p.dom
    ses = ShortExactSequence{QQ}(i.dom, i.cod, p.cod, i, p, false, false, nothing, nothing)
    if check
        ok = is_exact(ses; cache=cache)
        ok || error("ShortExactSequence: maps do not form a short exact sequence")
    end
    return ses
end

"""
    short_exact_sequence(i, p; check=true, cache=nothing) -> ShortExactSequence

Alias for `ShortExactSequence(i, p; check=..., cache=...)`.
"""
short_exact_sequence(i::PMorphism{QQ}, p::PMorphism{QQ};
                     check::Bool=true,
                     cache::Union{Nothing,CoverCache}=nothing) =
    ShortExactSequence(i, p; check=check, cache=cache)

""" 
    is_exact(ses; cache=nothing) -> Bool

Check whether the stored maps define a short exact sequence.
Results are cached inside the object.
"""
function is_exact(ses::ShortExactSequence{QQ}; cache::Union{Nothing,CoverCache}=nothing)
    if ses.checked
        return ses.exact
    end

    i = ses.i
    p = ses.p
    A = ses.A
    B = ses.B
    C = ses.C
    @assert i.dom === A
    @assert i.cod === B
    @assert p.dom === B
    @assert p.cod === C

    # p o i = 0
    for u in 1:nvertices(B.Q)
        comp = p.comps[u] * i.comps[u]
        all(iszero, comp) || (ses.checked = true; ses.exact = false; return false)
    end

    # i mono, p epi
    if !is_monomorphism(i) || !is_epimorphism(p)
        ses.checked = true
        ses.exact = false
        return false
    end

    # Compute and cache ker(p) and im(i) as submodules of B.
    if ses.ker_p === nothing
        ses.ker_p = kernel_with_inclusion(p; cache=cache)
    end
    if ses.img_i === nothing
        ses.img_i = image_with_inclusion(i)
    end
    (K, incK) = ses.ker_p
    (Im, incIm) = ses.img_i

    # Compare subspaces at each vertex using ranks.
    Q = B.Q
    for u in 1:nvertices(Q)
        Au = incK.comps[u]
        Bu = incIm.comps[u]
        rA = rankQQ(Au)
        rB = rankQQ(Bu)
        if rA != rB
            ses.checked = true
            ses.exact = false
            return false
        end
        # span(Au,Bu) must have same dimension if they are equal.
        rAB = rankQQ(hcat(Au, Bu))
        if rAB != rA
            ses.checked = true
            ses.exact = false
            return false
        end
    end

    ses.checked = true
    ses.exact = true
    return true
end

""" 
    assert_exact(ses; cache=nothing)

Throw an error if `ses` is not exact.
"""
function assert_exact(ses::ShortExactSequence{QQ}; cache::Union{Nothing,CoverCache}=nothing)
    is_exact(ses; cache=cache) || error("ShortExactSequence: sequence is not exact")
    return nothing
end

# Internal: induced map between kernels, given g : A -> B and kernel inclusions kA : ker(fA) -> A, kB : ker(fB) -> B.
function _induced_map_to_kernelQQ(g::PMorphism{QQ},
                                 kA::PMorphism{QQ},
                                 kB::PMorphism{QQ})
    Q = g.dom.Q
    @assert kA.cod === g.dom
    @assert g.cod === kB.cod
    Kdom = kA.dom
    Kcod = kB.dom
    comps = Vector{Matrix{QQ}}(undef, nvertices(Q))
    for u in 1:nvertices(Q)
        rhs = g.comps[u] * kA.comps[u]  # B_u x dim kerA_u
        comps[u] = solve_fullcolumnQQ(kB.comps[u], rhs)  # dim kerB_u x dim kerA_u
    end
    return PMorphism{QQ}(Kdom, Kcod, comps)
end

# Internal: induced map between cokernels, given h : A -> B and cokernel projections qA : A -> cokerA, qB : B -> cokerB.
function _induced_map_from_cokernelQQ(h::PMorphism{QQ},
                                     qA::PMorphism{QQ},
                                     qB::PMorphism{QQ})
    Q = h.dom.Q
    @assert qA.dom === h.dom
    @assert h.cod === qB.dom
    Cdom = qA.cod
    Ccod = qB.cod
    comps = Vector{Matrix{QQ}}(undef, nvertices(Q))
    for u in 1:nvertices(Q)
        Qsrc = qA.comps[u]
        rinv = _right_inverse_full_rowQQ(Qsrc)
        rhs = qB.comps[u] * h.comps[u]  # cokerB_u x A_u
        comps[u] = rhs * rinv           # cokerB_u x cokerA_u
    end
    return PMorphism{QQ}(Cdom, Ccod, comps)
end

"""
    SnakeLemmaResult

Result of `snake_lemma`: it packages the six objects and five maps in the snake lemma
exact sequence, including the connecting morphism delta.

The long exact sequence has the form:

    ker(fA) -> ker(fB) -> ker(fC) --delta--> coker(fA) -> coker(fB) -> coker(fC)

All maps are returned as actual morphisms of P-modules (natural transformations).
"""
struct SnakeLemmaResult{K}
    kerA::Tuple{PModule{K},PMorphism{K}}   # (KerA, inclusion KerA -> A)
    kerB::Tuple{PModule{K},PMorphism{K}}
    kerC::Tuple{PModule{K},PMorphism{K}}
    cokA::Tuple{PModule{K},PMorphism{K}}   # (CokA, projection A' -> CokA) where A' is cod(fA)
    cokB::Tuple{PModule{K},PMorphism{K}}
    cokC::Tuple{PModule{K},PMorphism{K}}
    k1::PMorphism{K}                       # ker(fA) -> ker(fB)
    k2::PMorphism{K}                       # ker(fB) -> ker(fC)
    delta::PMorphism{K}                    # ker(fC) -> coker(fA)
    c1::PMorphism{K}                       # coker(fA) -> coker(fB)
    c2::PMorphism{K}                       # coker(fB) -> coker(fC)
end

# Internal: check commutativity of a square g1 o f1 == g2 o f2 at all vertices.
function _check_commutative_squareQQ(g1::PMorphism{QQ}, f1::PMorphism{QQ},
                                     g2::PMorphism{QQ}, f2::PMorphism{QQ})
    Q = f1.dom.Q
    @assert f1.cod === g1.dom
    @assert f2.cod === g2.dom
    @assert f1.dom === f2.dom
    @assert g1.cod === g2.cod
    for u in 1:nvertices(Q)
        left = g1.comps[u] * f1.comps[u]
        right = g2.comps[u] * f2.comps[u]
        left == right || return false
    end
    return true
end

"""
    snake_lemma(top, bottom, fA, fB, fC; check=true, cache=nothing) -> SnakeLemmaResult

Compute the maps and objects in the snake lemma exact sequence for a commutative diagram
with exact rows:

    0 -> A  --i-->  B  --p-->  C  -> 0
          |        |        |
         fA       fB       fC
          |        |        |
    0 -> A' --i'->  B' --p'-> C' -> 0

Inputs:
- top    : ShortExactSequence for the top row (A,B,C,i,p)
- bottom : ShortExactSequence for the bottom row (A',B',C',i',p')
- fA, fB, fC : vertical morphisms A->A', B->B', C->C'

If `check=true`, we verify that both rows are exact and that the two squares commute.

The connecting morphism `delta : ker(fC) -> coker(fA)` is computed explicitly using
linear algebra in each stalk.
"""
function snake_lemma(top::ShortExactSequence{QQ},
                     bottom::ShortExactSequence{QQ},
                     fA::PMorphism{QQ},
                     fB::PMorphism{QQ},
                     fC::PMorphism{QQ};
                     check::Bool=true,
                     cache::Union{Nothing,CoverCache}=nothing)

    if check
        assert_exact(top; cache=cache)
        assert_exact(bottom; cache=cache)

        # Squares must commute:
        # fB o i = i' o fA
        ok1 = _check_commutative_squareQQ(fB, top.i, bottom.i, fA)
        ok1 || error("snake_lemma: left square does not commute")
        # p' o fB = fC o p
        ok2 = _check_commutative_squareQQ(bottom.p, fB, fC, top.p)
        ok2 || error("snake_lemma: right square does not commute")
    end

    # Kernels of vertical maps
    kerA = kernel_with_inclusion(fA; cache=cache)
    kerB = kernel_with_inclusion(fB; cache=cache)
    kerC = kernel_with_inclusion(fC; cache=cache)

    # Cokernels of vertical maps
    cokA = cokernel_with_projection(fA; cache=cache)
    cokB = cokernel_with_projection(fB; cache=cache)
    cokC = cokernel_with_projection(fC; cache=cache)

    (KerA, incKerA) = kerA
    (KerB, incKerB) = kerB
    (KerC, incKerC) = kerC

    (CokA, qA) = cokA
    (CokB, qB) = cokB
    (CokC, qC) = cokC

    Q = top.A.Q

    # Induced maps on kernels: ker(fA) -> ker(fB) -> ker(fC)
    # k1 : KerA -> KerB induced by top.i : A -> B
    k1 = _induced_map_to_kernelQQ(top.i, incKerA, incKerB)

    # k2 : KerB -> KerC induced by top.p : B -> C
    k2 = _induced_map_to_kernelQQ(top.p, incKerB, incKerC)

    # Induced maps on cokernels: coker(fA) -> coker(fB) -> coker(fC)
    # c1 induced by bottom.i : A' -> B'
    c1 = _induced_map_from_cokernelQQ(bottom.i, qA, qB)

    # c2 induced by bottom.p : B' -> C'
    c2 = _induced_map_from_cokernelQQ(bottom.p, qB, qC)

    # Connecting morphism delta : ker(fC) -> coker(fA)
    delta_comps = Vector{Matrix{QQ}}(undef, nvertices(Q))
    for u in 1:nvertices(Q)
        kdim = KerC.dims[u]
        if kdim == 0 || CokA.dims[u] == 0
            delta_comps[u] = zeros(QQ, CokA.dims[u], kdim)
            continue
        end

        # Kc is C_u x kdim, columns are a basis of ker(fC_u) (embedded into C_u).
        Kc = incKerC.comps[u]

        # Lift basis elements in C_u to B_u via p : B -> C (top row).
        # Since p_u is surjective in a short exact sequence, we can use a right inverse.
        rinv_p = _right_inverse_full_rowQQ(top.p.comps[u])  # B_u x C_u
        B_lift = rinv_p * Kc  # B_u x kdim

        # Apply fB to get elements in B'_u.
        Bp = fB.comps[u] * B_lift  # B'_u x kdim

        # Since Kc is in ker(fC), commutativity implies Bp is in ker(p') = im(i').
        Ap = solve_fullcolumnQQ(bottom.i.comps[u], Bp)  # A'_u x kdim

        # Project to coker(fA): qA : A' -> CokA
        delta_comps[u] = qA.comps[u] * Ap  # CokA_u x kdim
    end
    delta = PMorphism{QQ}(KerC, CokA, delta_comps)

    return SnakeLemmaResult{QQ}(kerA, kerB, kerC, cokA, cokB, cokC, k1, k2, delta, c1, c2)
end

"""
    snake_lemma(i, p, i2, p2, fA, fB, fC; check=true, cache=nothing) -> SnakeLemmaResult

Convenience overload: provide the four row maps directly instead of pre-constructing
`ShortExactSequence` objects for the top and bottom rows.
"""
function snake_lemma(i::PMorphism{QQ}, p::PMorphism{QQ},
                     i2::PMorphism{QQ}, p2::PMorphism{QQ},
                     fA::PMorphism{QQ}, fB::PMorphism{QQ}, fC::PMorphism{QQ};
                     check::Bool=true,
                     cache::Union{Nothing,CoverCache}=nothing)
    top = ShortExactSequence(i, p; check=check, cache=cache)
    bottom = ShortExactSequence(i2, p2; check=check, cache=cache)
    return snake_lemma(top, bottom, fA, fB, fC; check=check, cache=cache)
end

# ---------------------------------------------------------------------------
# Categorical biproduct/product/coproduct wrappers (mathematician-friendly API)
# ---------------------------------------------------------------------------

"""
    biproduct(M, N) -> (S, iM, iN, pM, pN)

Binary biproduct in the abelian category of `PModule`s.

Concretely, this is the direct sum `S = M \\oplus N` with its canonical
injections and projections:

- `iM : M -> S`, `iN : N -> S`
- `pM : S -> M`, `pN : S -> N`

This is a readability wrapper around `direct_sum_with_maps`.

See also: `direct_sum_with_maps`, `product`, `coproduct`.
"""
biproduct(M::PModule{K}, N::PModule{K}) where {K} = direct_sum_with_maps(M, N)

"""
    coproduct(M, N) -> (S, iM, iN)

Categorical coproduct of two modules.

In an additive category (in particular for modules), the coproduct is the same
object as the product: the biproduct `M \\oplus N`.

Returns the object `S` and the canonical injections.
"""
function coproduct(M::PModule{K}, N::PModule{K}) where {K}
    S, iM, iN, _, _ = direct_sum_with_maps(M, N)
    return S, iM, iN
end

"""
    product(M, N) -> (P, pM, pN)

Categorical product of two modules.

In an additive category (in particular for modules), the product is the same
object as the coproduct: the biproduct `M \\oplus N`.

Returns the object `P` and the canonical projections.
"""
function product(M::PModule{K}, N::PModule{K}) where {K}
    P, _, _, pM, pN = direct_sum_with_maps(M, N)
    return P, pM, pN
end

# --- Finite products/coproducts (n-ary) ------------------------------------
#
# These are convenient in categorical workflows and are implemented using a
# single-pass block construction for speed (rather than iterating binary sums).

"""
    coproduct(mods::AbstractVector{<:PModule}) -> (S, injections)

Finite coproduct of a list of modules.

Returns:
- `S` : the direct sum object
- `injections[i] : mods[i] -> S` : the canonical injection maps

For an empty list, throws an error.
"""
function coproduct(mods::AbstractVector{<:PModule{K}}) where {K}
    if isempty(mods)
        error("coproduct: need at least one module")
    elseif length(mods) == 1
        M = mods[1]
        return M, PMorphism{K}[id_morphism(M)]
    end
    S, injections, _ = _direct_sum_many_with_maps(mods)
    return S, injections
end

"""
    product(mods::AbstractVector{<:PModule}) -> (P, projections)

Finite product of a list of modules.

Returns:
- `P` : the direct sum object
- `projections[i] : P -> mods[i]` : the canonical projection maps

For an empty list, throws an error.
"""
function product(mods::AbstractVector{<:PModule{K}}) where {K}
    if isempty(mods)
        error("product: need at least one module")
    elseif length(mods) == 1
        M = mods[1]
        return M, PMorphism{K}[id_morphism(M)]
    end
    P, _, projections = _direct_sum_many_with_maps(mods)
    return P, projections
end

# Vararg convenience:
coproduct(M::PModule{K}, N::PModule{K}, rest::PModule{K}...) where {K} =
    coproduct(PModule{K}[M, N, rest...])

product(M::PModule{K}, N::PModule{K}, rest::PModule{K}...) where {K} =
    product(PModule{K}[M, N, rest...])

# --- Internal helper: build one big direct sum in a single pass -------------
#
# Returns (S, injections, projections), where:
# - injections[i] : mods[i] -> S
# - projections[i] : S -> mods[i]
#
# This keeps the common "poset + cover edge maps" structure and uses:
# - block-diagonal edge maps
# - block-identity injections/projections at each vertex
function _direct_sum_many_with_maps(mods::AbstractVector{<:PModule{K}}) where {K}
    m = length(mods)
    m == 0 && error("_direct_sum_many_with_maps: need at least one module")

    Q = mods[1].Q
    n = nvertices(Q)
    for M in mods
        if M.Q !== Q
            error("_direct_sum_many_with_maps: modules must live on the same poset")
        end
    end

    # offsets[u][i] is the 0-based starting index of the i-th summand inside the
    # direct-sum fiber at vertex u. (So offsets[u][1] = 0.)
    offsets = [Vector{Int}(undef, m + 1) for _ in 1:n]
    for u in 1:n
        off = offsets[u]
        off[1] = 0
        for i in 1:m
            off[i+1] = off[i] + mods[i].dims[u]
        end
    end

    Sdims = [offsets[u][end] for u in 1:n]

    # Build cover-edge maps for the direct sum.
    cc = cover_cache(Q)

    aligned = true
    for M in mods
        if M.edge_maps.preds != cc.preds || M.edge_maps.succs != cc.succs
            aligned = false
            break
        end
    end

    if aligned
        preds = cc.preds
        succs = cc.succs

        maps_from_pred = [Vector{Matrix{K}}(undef, length(preds[v])) for v in 1:n]
        maps_to_succ   = [Vector{Matrix{K}}(undef, length(succs[u])) for u in 1:n]

        @inbounds for u in 1:n
            su = succs[u]
            outu = maps_to_succ[u]
            for j in eachindex(su)
                v = su[j]
                Auv = zeros(K, Sdims[v], Sdims[u])

                # Fill the block diagonal with summand maps.
                for i in 1:m
                    du = mods[i].dims[u]
                    dv = mods[i].dims[v]
                    if du == 0 || dv == 0
                        continue
                    end
                    r0 = offsets[v][i] + 1
                    r1 = offsets[v][i+1]
                    c0 = offsets[u][i] + 1
                    c1 = offsets[u][i+1]
                    block = mods[i].edge_maps.maps_to_succ[u][j]
                    copyto!(view(Auv, r0:r1, c0:c1), block)
                end

                outu[j] = Auv
                ip = _find_sorted_index(preds[v], u)
                @inbounds maps_from_pred[v][ip] = Auv
            end
        end

        store = CoverEdgeMapStore{K,Matrix{K}}(preds, succs, maps_from_pred, maps_to_succ, cc.nedges)
        S = PModule{K,Matrix{K}}(Q, Sdims, store)
    else
        # Fallback: keyed access (still O(|E|), but slower).
        edge_maps = Dict{Tuple{Int,Int}, Matrix{K}}()
        sizehint!(edge_maps, cc.nedges)
        for (u, v) in cover_edges(Q)
            Auv = zeros(K, Sdims[v], Sdims[u])
            for i in 1:m
                du = mods[i].dims[u]
                dv = mods[i].dims[v]
                if du == 0 || dv == 0
                    continue
                end
                r0 = offsets[v][i] + 1
                r1 = offsets[v][i+1]
                c0 = offsets[u][i] + 1
                c1 = offsets[u][i+1]
                block = mods[i].edge_maps[u, v]
                copyto!(view(Auv, r0:r1, c0:c1), block)
            end
            edge_maps[(u, v)] = Auv
        end
        S = PModule{K}(Q, Sdims, edge_maps)
    end

    # Injections and projections (fill diagonal entries directly).
    injections = PMorphism{K}[]
    projections = PMorphism{K}[]
    for i in 1:m
        inj_comps = Vector{Matrix{K}}(undef, n)
        proj_comps = Vector{Matrix{K}}(undef, n)
        for u in 1:n
            du = mods[i].dims[u]
            Su = Sdims[u]
            Iu = zeros(K, Su, du)
            Pu = zeros(K, du, Su)
            if du != 0
                r0 = offsets[u][i] + 1
                c0 = offsets[u][i] + 1
                @inbounds for t in 1:du
                    Iu[r0 + t - 1, t] = one(K)
                    Pu[t, c0 + t - 1] = one(K)
                end
            end
            inj_comps[u] = Iu
            proj_comps[u] = Pu
        end
        push!(injections, PMorphism{K}(mods[i], S, inj_comps))
        push!(projections, PMorphism{K}(S, mods[i], proj_comps))
    end

    return S, injections, projections
end




# -----------------------------------------------------------------------------
# Pretty-printing / display (ASCII-only)
# -----------------------------------------------------------------------------
#
# Goals:
# - Mathematician-friendly summaries for 
# -  PModule / PMorphism (for REPL ergonomics when doing lots of algebra),
# Submodule, ShortExactSequence, and SnakeLemmaResult.
# - ASCII-only output (no Unicode arrows, symbols, etc).
# - Do NOT trigger heavy computations during printing:
#     * show(ShortExactSequence) must NOT call is_exact(ses)
#     * show(SnakeLemmaResult) must NOT recompute anything
# - Respect IOContext(:limit=>true) by truncating long dim vectors.

# Internal: human-readable scalar name (keeps QQ readable).
_scalar_name(::Type{QQ}) = "QQ"
_scalar_name(::Type{K}) where {K} = string(K)

# Internal: cheap stats for a dims vector (no allocations).
# Returns (sum, nnz, max).
function _dims_stats(dims::AbstractVector{<:Integer})
    total = 0
    nnz = 0
    maxd = 0
    @inbounds for d0 in dims
        d = Int(d0)
        total += d
        if d != 0
            nnz += 1
            if d > maxd
                maxd = d
            end
        end
    end
    return total, nnz, maxd
end

# Internal: print integer vector, truncating if the IOContext requests it.
function _print_int_vec(io::IO, v::AbstractVector{<:Integer};
                        max_elems::Int=12, head::Int=4, tail::Int=3)
    n = length(v)
    print(io, "[")
    if n == 0
        print(io, "]")
        return
    end

    limit = get(io, :limit, false)

    # Full print if not limiting or short enough.
    if !limit || n <= max_elems
        @inbounds for i in 1:n
            i > 1 && print(io, ", ")
            print(io, Int(v[i]))
        end
        print(io, "]")
        return
    end

    # Truncated print: head entries, "...", tail entries.
    h = min(head, n)
    t = min(tail, max(0, n - h))

    @inbounds for i in 1:h
        i > 1 && print(io, ", ")
        print(io, Int(v[i]))
    end

    if h < n - t
        print(io, ", ..., ")
    elseif h < n && t > 0
        print(io, ", ")
    end

    @inbounds for i in (n - t + 1):n
        i > (n - t + 1) && print(io, ", ")
        print(io, Int(v[i]))
    end

    print(io, "]")
    return
end

"""
    Base.show(io::IO, M::PModule{K}) where {K}

Compact one-line summary for a `PModule`.

This is intended for quick REPL inspection. It prints:
- nverts: number of vertices in the underlying finite poset,
- sum/nnz/max: cheap statistics of the stalk dimension vector,
- dims: the stalk dimensions (truncated if `IOContext(io, :limit=>true)`),
- cover_maps: number of stored structure maps along cover edges.

ASCII-only by design.
"""
function Base.show(io::IO, M::PModule{K}) where {K}
    nverts = nvertices(M.Q)
    s, nnz, mx = _dims_stats(M.dims)

    print(io, "PModule(")
    print(io, "nverts=", nverts)
    print(io, ", sum=", s)
    print(io, ", nnz=", nnz)
    print(io, ", max=", mx)
    print(io, ", dims=")
    _print_int_vec(io, M.dims)
    print(io, ", cover_maps=", length(M.edge_maps))
    print(io, ")")
end

"""
    Base.show(io::IO, ::MIME"text/plain", M::PModule{K}) where {K}

Verbose multi-line summary for a `PModule` (what the REPL typically shows).

This is still cheap: it only scans `dims` once and does not compute ranks,
images, kernels, etc.
"""
function Base.show(io::IO, ::MIME"text/plain", M::PModule{K}) where {K}
    nverts = nvertices(M.Q)
    s, nnz, mx = _dims_stats(M.dims)

    println(io, "PModule")
    println(io, "  scalars = ", _scalar_name(K))
    println(io, "  nverts = ", nverts)

    print(io, "  dims = ")
    _print_int_vec(io, M.dims)
    println(io)

    println(io, "    sum = ", s, ", nnz = ", nnz, ", max = ", mx)
    println(io, "  cover_maps = ", length(M.edge_maps), "  (maps stored along cover edges)")
end

"""
    Base.show(io::IO, f::PMorphism{K}) where {K}

Compact one-line summary for a vertexwise morphism of P-modules.

We intentionally avoid any expensive linear algebra (ranks, images, etc.) here.
"""
function Base.show(io::IO, f::PMorphism{K}) where {K}
    n_dom = nvertices(f.dom.Q)
    n_cod = nvertices(f.cod.Q)

    dom_sum, _, _ = _dims_stats(f.dom.dims)
    cod_sum, _, _ = _dims_stats(f.cod.dims)

    print(io, "PMorphism(")
    if f.dom === f.cod
        print(io, "endo, ")
    end

    if n_dom == n_cod
        print(io, "nverts=", n_dom)
    else
        # Should not happen in well-formed inputs, but keep printing robust.
        print(io, "nverts_dom=", n_dom, ", nverts_cod=", n_cod)
    end

    print(io, ", dom_sum=", dom_sum)
    print(io, ", cod_sum=", cod_sum)
    print(io, ", comps=", length(f.comps))
    print(io, ")")
end

"""
    Base.show(io::IO, ::MIME"text/plain", f::PMorphism{K}) where {K}

Verbose multi-line summary for `PMorphism`.

We report basic size information about the domain/codomain and indicate the
intended per-vertex matrix sizes, without verifying them (verification can be
added as a separate validator; printing should stay cheap and noninvasive).
"""
function Base.show(io::IO, ::MIME"text/plain", f::PMorphism{K}) where {K}
    n_dom = nvertices(f.dom.Q)
    n_cod = nvertices(f.cod.Q)

    dom_sum, dom_nnz, dom_max = _dims_stats(f.dom.dims)
    cod_sum, cod_nnz, cod_max = _dims_stats(f.cod.dims)

    println(io, "PMorphism")
    println(io, "  scalars = ", _scalar_name(K))

    if n_dom == n_cod
        println(io, "  nverts = ", n_dom)
    else
        println(io, "  nverts_dom = ", n_dom)
        println(io, "  nverts_cod = ", n_cod)
    end

    println(io, "  endomorphism = ", (f.dom === f.cod))

    print(io, "  dom dims = ")
    _print_int_vec(io, f.dom.dims)
    println(io)
    println(io, "    sum = ", dom_sum, ", nnz = ", dom_nnz, ", max = ", dom_max)

    print(io, "  cod dims = ")
    _print_int_vec(io, f.cod.dims)
    println(io)
    println(io, "    sum = ", cod_sum, ", nnz = ", cod_nnz, ", max = ", cod_max)

    println(io, "  comps: ", length(f.comps), " vertexwise linear maps")
    println(io, "    comps[i] has size cod.dims[i] x dom.dims[i] (for each vertex i)")
end

"""
    Base.show(io::IO, S::Submodule)

Compact one-line summary for `Submodule`.
"""
function Base.show(io::IO, S::Submodule{K}) where {K}
    N = sub(S)
    M = ambient(S)
    nverts = nvertices(M.Q)
    sub_sum, _, _ = _dims_stats(N.dims)
    amb_sum, _, _ = _dims_stats(M.dims)
    print(io,
          "Submodule(",
          "nverts=", nverts,
          ", sub_sum=", sub_sum,
          ", ambient_sum=", amb_sum,
          ")")
end

"""
    Base.show(io::IO, ::MIME"text/plain", S::Submodule)

Verbose multi-line summary for `Submodule`. ASCII-only.
"""
function Base.show(io::IO, ::MIME"text/plain", S::Submodule{K}) where {K}
    N = sub(S)
    M = ambient(S)
    nverts = nvertices(M.Q)

    sub_sum, sub_nnz, sub_max = _dims_stats(N.dims)
    amb_sum, amb_nnz, amb_max = _dims_stats(M.dims)

    println(io, "Submodule")
    println(io, "  scalars = ", _scalar_name(K))
    println(io, "  nverts = ", nverts)

    print(io, "  sub dims = ")
    _print_int_vec(io, N.dims)
    println(io)
    println(io, "    sum = ", sub_sum, ", nnz = ", sub_nnz, ", max = ", sub_max)

    print(io, "  ambient dims = ")
    _print_int_vec(io, M.dims)
    println(io)
    println(io, "    sum = ", amb_sum, ", nnz = ", amb_nnz, ", max = ", amb_max)

    println(io, "  inclusion : sub(S) -> ambient(S)  (use inclusion(S) to access)")
end


"""
    Base.show(io::IO, ses::ShortExactSequence)

Compact one-line summary for `ShortExactSequence`.

NOTE: This does NOT call `is_exact(ses)`; it only reports cached status.
"""
function Base.show(io::IO, ses::ShortExactSequence{K}) where {K}
    nverts = nvertices(ses.B.Q)
    Asum, _, _ = _dims_stats(ses.A.dims)
    Bsum, _, _ = _dims_stats(ses.B.dims)
    Csum, _, _ = _dims_stats(ses.C.dims)

    print(io,
          "ShortExactSequence(",
          "nverts=", nverts,
          ", A_sum=", Asum,
          ", B_sum=", Bsum,
          ", C_sum=", Csum,
          ", checked=", ses.checked,
          ", exact=")
    if ses.checked
        print(io, ses.exact)
    else
        print(io, "unknown")
    end
    print(io, ")")
end

"""
    Base.show(io::IO, ::MIME"text/plain", ses::ShortExactSequence)

Verbose multi-line summary for `ShortExactSequence`. ASCII-only.

NOTE: This does NOT call `is_exact(ses)`; it only reports cached status.
"""
function Base.show(io::IO, ::MIME"text/plain", ses::ShortExactSequence{K}) where {K}
    nverts = nvertices(ses.B.Q)

    Asum, Annz, Amax = _dims_stats(ses.A.dims)
    Bsum, Bnnz, Bmax = _dims_stats(ses.B.dims)
    Csum, Cnnz, Cmax = _dims_stats(ses.C.dims)

    println(io, "ShortExactSequence")
    println(io, "  0 -> A -(i)-> B -(p)-> C -> 0")
    println(io, "  scalars = ", _scalar_name(K))
    println(io, "  nverts = ", nverts)

    println(io, "  checked = ", ses.checked)
    if ses.checked
        println(io, "  exact = ", ses.exact)
    else
        println(io, "  exact = unknown (call is_exact(ses) to check and cache)")
    end
    println(io, "  caches: ker(p) = ", ses.ker_p !== nothing,
                ", im(i) = ", ses.img_i !== nothing)

    print(io, "  A dims = ")
    _print_int_vec(io, ses.A.dims)
    println(io)
    println(io, "    sum = ", Asum, ", nnz = ", Annz, ", max = ", Amax)

    print(io, "  B dims = ")
    _print_int_vec(io, ses.B.dims)
    println(io)
    println(io, "    sum = ", Bsum, ", nnz = ", Bnnz, ", max = ", Bmax)

    print(io, "  C dims = ")
    _print_int_vec(io, ses.C.dims)
    println(io)
    println(io, "    sum = ", Csum, ", nnz = ", Cnnz, ", max = ", Cmax)

    println(io, "  maps: use ses.i and ses.p (both are PMorphism objects)")
end


"""
    Base.show(io::IO, sn::SnakeLemmaResult)

Compact one-line summary for `SnakeLemmaResult`.
"""
function Base.show(io::IO, sn::SnakeLemmaResult{K}) where {K}
    nverts = nvertices(sn.delta.dom.Q)
    kerCsum, _, _ = _dims_stats(sn.kerC[1].dims)
    cokAsum, _, _ = _dims_stats(sn.cokA[1].dims)

    print(io,
          "SnakeLemmaResult(",
          "nverts=", nverts,
          ", delta: kerC_sum=", kerCsum,
          " -> cokerA_sum=", cokAsum,
          ")")
end

"""
    Base.show(io::IO, ::MIME"text/plain", sn::SnakeLemmaResult)

Verbose multi-line summary for `SnakeLemmaResult`. ASCII-only.
"""
function Base.show(io::IO, ::MIME"text/plain", sn::SnakeLemmaResult{K}) where {K}
    nverts = nvertices(sn.delta.dom.Q)

    println(io, "SnakeLemmaResult")
    println(io, "  kerA -> kerB -> kerC --delta--> cokerA -> cokerB -> cokerC")
    println(io, "  scalars = ", _scalar_name(K))
    println(io, "  nverts = ", nverts)

    # Helper to print one object entry (kerX/cokerX).
    function _print_obj(io2::IO, name::AbstractString, tup)
        M = tup[1]
        s, nnz, mx = _dims_stats(M.dims)
        print(io2, "  ", name, " dims = ")
        _print_int_vec(io2, M.dims)
        println(io2)
        println(io2, "    sum = ", s, ", nnz = ", nnz, ", max = ", mx)
    end

    _print_obj(io, "kerA", sn.kerA)
    _print_obj(io, "kerB", sn.kerB)
    _print_obj(io, "kerC", sn.kerC)
    _print_obj(io, "cokerA", sn.cokA)
    _print_obj(io, "cokerB", sn.cokB)
    _print_obj(io, "cokerC", sn.cokC)

    println(io, "  maps: k1, k2, delta, c1, c2  (access as fields on the result)")
end


end
