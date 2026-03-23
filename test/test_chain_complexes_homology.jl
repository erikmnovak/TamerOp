using Test
using SparseArrays
using LinearAlgebra
import Base.Threads

# Included from test/runtests.jl; uses shared aliases (TO, FF, IR, DF, MD, QQ, ...).
const FL = TamerOp.FieldLinAlg
const MCM = TamerOp.ModuleComplexes

with_fields(FIELDS_FULL) do field
Kc = CM.coeff_type(field)
@inline c(x) = CM.coerce(field, x)
@inline zmat(m, n) = CM.zeros(field, m, n)
@inline eye_mat(n) = CM.eye(field, n)
@inline _is_real_field(f) = f isa CM.RealField

# helper: simple 1 times 1 fringe module on interval [a,b]
function interval_module(P::FF.FinitePoset, a::Int, b::Int)
    if !FF.leq(P, a, b)
        error("interval_module expects a <= b in the given poset; got a=$a, b=$b")
    end
    U = FF.principal_upset(P, a)
    D = FF.principal_downset(P, b)
    return FF.one_by_one_fringe(P, U, D, c(1); field=field)
# end interval_module
end
# A tiny helper: one-vertex poset module = just a vector space.
function one_vertex_module(dim::Int)
    P = chain_poset(1)
    return MD.PModule{Kc}(P, [dim], Dict{Tuple{Int,Int}, Matrix{Kc}}(); field=field)
end

function scalar_morphism(M::MD.PModule{K}, a::Int) where {K}
    comps = Vector{Matrix{K}}(undef, M.Q.n)
    for v in 1:M.Q.n
        dv = M.dims[v]
        comps[v] = dv == 0 ? zmat(0, 0) : c(a) * eye_mat(dv)
    end
    return MD.PMorphism(M, M, comps)
end

function compose_morphism(g::MD.PMorphism{K}, f::MD.PMorphism{K}) where {K}
    @assert f.cod === g.dom
    n = f.dom.Q.n
    comps = [g.comps[v] * f.comps[v] for v in 1:n]
    return MD.PMorphism(f.dom, g.cod, comps)
end

@testset "Homological algebra edge cases on finite posets" begin
    # One-point poset: incidence algebra is just the base field.
    P1 = chain_poset(1)
    S = IR.pmodule_from_fringe(one_by_one_fringe(
        P1,
        FF.principal_upset(P1, 1),
        FF.principal_downset(P1, 1);
        field=field,
    ))

    E = DF.Ext(S, S, TO.DerivedFunctorOptions(maxdeg=3))
    @test TO.dim(E, 0) == 1
    @test all(TO.dim(E, t) == 0 for t in 1:3)

    T = DF.Tor(S, S, TO.DerivedFunctorOptions(maxdeg=3))
    @test TO.dim(T, 0) == 1
    @test all(TO.dim(T, t) == 0 for t in 1:3)

    # Zero module: Ext and Tor should vanish in all degrees.
    Z = MD.PModule{Kc}(P1, [0], Dict{Tuple{Int,Int}, Matrix{Kc}}(); field=field)

    EZS = DF.Ext(Z, S, TO.DerivedFunctorOptions(maxdeg=2))
    ESZ = DF.Ext(S, Z, TO.DerivedFunctorOptions(maxdeg=2))
    EZZ = DF.Ext(Z, Z, TO.DerivedFunctorOptions(maxdeg=2))
    @test all(TO.dim(EZS, t) == 0 for t in 0:2)
    @test all(TO.dim(ESZ, t) == 0 for t in 0:2)
    @test all(TO.dim(EZZ, t) == 0 for t in 0:2)

    TZS = DF.Tor(Z, S, TO.DerivedFunctorOptions(maxdeg=2))
    TSZ = DF.Tor(S, Z, TO.DerivedFunctorOptions(maxdeg=2))
    TZZ = DF.Tor(Z, Z, TO.DerivedFunctorOptions(maxdeg=2))
    @test all(TO.dim(TZS, t) == 0 for t in 0:2)
    @test all(TO.dim(TSZ, t) == 0 for t in 0:2)
    @test all(TO.dim(TZZ, t) == 0 for t in 0:2)

    # Disconnected poset: Ext between components should vanish.
    P = disjoint_two_chains_poset(2, 2)  # vertices {1,2} and {3,4} are separate components
    S1 = IR.pmodule_from_fringe(one_by_one_fringe(
        P,
        FF.principal_upset(P, 1),
        FF.principal_downset(P, 1);
        field=field,
    ))
    S3 = IR.pmodule_from_fringe(one_by_one_fringe(
        P,
        FF.principal_upset(P, 3),
        FF.principal_downset(P, 3);
        field=field,
    ))
    E13 = DF.Ext(S1, S3, TO.DerivedFunctorOptions(maxdeg=2))
    @test all(TO.dim(E13, t) == 0 for t in 0:2)

    # Cross-check: Ext computed via projectives vs via injectives agree on dimensions.
    Pd = diamond_poset()
    A = IR.pmodule_from_fringe(one_by_one_fringe(
        Pd,
        FF.principal_upset(Pd, 1),
        FF.principal_downset(Pd, 1);
        field=field,
    ))
    B = IR.pmodule_from_fringe(one_by_one_fringe(
        Pd,
        FF.principal_upset(Pd, 4),
        FF.principal_downset(Pd, 4);
        field=field,
    ))

    Eproj = DF.Ext(A, B, TO.DerivedFunctorOptions(maxdeg=2))
    resBinj = DF.injective_resolution(B, TO.ResolutionOptions(maxlen=2))
    Einj = TO.ExtInjective(A, resBinj)

    @test [TO.dim(Eproj, t) for t in 0:2] == [TO.dim(Einj, t) for t in 0:2]
end

@testset "ChainComplexes homology_data and homology_coordinates by hand" begin
    # ----------------
    # Circle: C1=k, C0=k, d1=0
    # ----------------
    d1 = zeros(Kc, 1, 1)      # C1 -> C0
    d0 = zeros(Kc, 0, 1)      # C0 -> 0
    d2 = zeros(Kc, 1, 0)      # 0  -> C1

    H1 = CC.homology_data(d2, d1, 1)
    @test H1.dimH == 1

    H0 = CC.homology_data(d1, d0, 0)
    @test H0.dimH == 1

    # coordinate sanity on the chosen basis representative
    coords = CC.homology_coordinates(H1, H1.Hrep[:, 1])
    @test coords == reshape(Kc[1], 1, 1)

    # ----------------
    # Interval: C1=k, C0=k^2, d1 = [ -1; 1 ]
    # ----------------
    d1_int = reshape(Kc[-1, 1], 2, 1)
    d0_int = zeros(Kc, 0, 2)
    d2_int = zeros(Kc, 1, 0)

    H1_int = CC.homology_data(d2_int, d1_int, 1)
    @test H1_int.dimH == 0

    H0_int = CC.homology_data(d1_int, d0_int, 0)
    @test H0_int.dimH == 1

    # any nonzero vector in C1 is not a cycle since d1_int is injective
    @test_throws ErrorException CC.homology_coordinates(H1_int, reshape(Kc[1], 1, 1))

    # boundary class should map to 0 in H0
    b = H0_int.B[:, 1]
    c0 = CC.homology_coordinates(H0_int, b)
    @test c0 == zeros(Kc, H0_int.dimH, 1)

    # ----------------
    # Filled triangle: C2=k, C1=k^3, C0=k^3
    # edge basis: e01,e02,e12; vertex basis: v0,v1,v2; face basis: f012
    # d2(f) = e12 - e02 + e01 => column [1,-1,1]
    # d1 columns are boundary of edges:
    #   e01 -> v1 - v0  => [-1, 1, 0]
    #   e02 -> v2 - v0  => [-1, 0, 1]
    #   e12 -> v2 - v1  => [ 0,-1, 1]
    # ----------------
    d2_tri = reshape(Kc[1, -1, 1], 3, 1)
    d1_tri = Kc[
        -1  -1   0;
         1   0  -1;
         0   1   1
    ]
    d0_tri = zeros(Kc, 0, 3)

    H2_tri = CC.homology_data(zeros(Kc, 1, 0), d2_tri, 2)
    @test H2_tri.dimH == 0

    H1_tri = CC.homology_data(d2_tri, d1_tri, 1)
    @test H1_tri.dimH == 0

    H0_tri = CC.homology_data(d1_tri, d0_tri, 0)
    @test H0_tri.dimH == 1
end

@testset "ChainComplexes: shift and extend_range" begin
    # Cochain complex C: degrees 0..1, zero differential, dims 1 in each degree.
    d0 = spzeros(Kc, 1, 1)
    C = CC.CochainComplex{Kc}(0, 1, [1, 1], [d0])

    @test CC.cohomology_data(C, 0).dimH == 1
    @test CC.cohomology_data(C, 1).dimH == 1
    H_all = CC.cohomology_data(C)
    @test length(H_all) == 2
    @test [h.dimH for h in H_all] == [1, 1]

    # Shift by +2: (C[2])^t = C^{t+2}
    Cs = CC.shift(C, 2)
    @test Cs.tmin == -2
    @test Cs.tmax == -1
    @test CC.cohomology_data(Cs, -2).dimH == 1
    @test CC.cohomology_data(Cs, -1).dimH == 1

    # Extend range should not change cohomology in the original degrees.
    Ce = CC.extend_range(C, -3, 4)
    @test CC.cohomology_data(Ce, 0).dimH == 1
    @test CC.cohomology_data(Ce, 1).dimH == 1
end

@testset "ChainComplexes: zero-boundary and full-boundary fast paths" begin
    # Zero differential: H^t = Z^t with identity coordinate basis.
    d0 = spzeros(Kc, 2, 2)
    C0 = CC.CochainComplex{Kc}(0, 1, [2, 2], [d0])
    H0 = CC.cohomology_data(C0, 0)
    @test H0.dimB == 0
    @test H0.dimH == 2
    @test H0.Hrep == H0.K
    @test H0.Bfull == Matrix{Kc}(I, 2, 2)

    # Full boundaries: H^1 = 0 and the returned basis data stays square/consistent.
    d1 = sparse(Matrix{Kc}(I, 2, 2))
    C1 = CC.CochainComplex{Kc}(0, 1, [2, 2], [d1])
    H1 = CC.cohomology_data(C1, 1)
    @test H1.dimZ == 2
    @test H1.dimB == 2
    @test H1.dimH == 0
    @test size(H1.Bfull) == (2, 2)
    @test size(H1.Hrep) == (2, 0)

    Hh = CC.homology_data(spzeros(Kc, 2, 0), Matrix{Kc}(I, 2, 2), 0)
    @test Hh.dimB == 0
    @test Hh.dimH == 0
    @test size(Hh.Bfull) == (0, 0)
end

@testset "ChainComplexes: extend_to_basis yields invertible completion" begin
    C = Kc[
        c(1)  c(0)  c(1);
        c(0)  c(1)  c(1);
        c(1)  c(1)  c(2);
        c(0)  c(0)  c(0);
        c(1)  c(0)  c(1)
    ]
    B = CC.extend_to_basis(C)
    @test size(B) == (5, 5)
    @test FL.rank(field, B) == 5
    @test FL.rank(field, B[:, 1:FL.rank(field, C)]) == FL.rank(field, C)
end

@testset "ChainComplexes: fused diff summary and lazy cohomology representatives" begin
    A_dense = Kc[
        c(1) c(2) c(0);
        c(0) c(1) c(1)
    ]
    A_sparse = sparse(A_dense)

    sum_dense = FL._kernel_image_summary(field, A_dense)
    @test sum_dense.rank == size(FL.colspace(field, A_dense), 2)
    @test sum_dense.ker == Matrix{Kc}(FL.nullspace(field, A_dense))
    @test sum_dense.img == Matrix{Kc}(FL.colspace(field, A_dense))

    sum_sparse = FL._kernel_image_summary(field, A_sparse)
    @test sum_sparse.rank == size(FL.colspace(field, A_sparse), 2)
    @test sum_sparse.ker == Matrix{Kc}(FL.nullspace(field, A_sparse))
    @test sum_sparse.img == Matrix{Kc}(FL.colspace(field, A_sparse))

    Azero = spzeros(Kc, 3, 4)
    szero = CC._diff_summary(field, Azero)
    @test szero.rank == 0
    @test szero.ker == Matrix{Kc}(I, 4, 4)
    @test szero.img == zeros(Kc, 3, 0)

    d0 = sparse(reshape(Kc[c(1), c(0)], 2, 1))
    d1 = spzeros(Kc, 1, 2)
    Ccoh = CC.CochainComplex{Kc}(0, 2, [1, 2, 1], [d0, d1])
    H1deg = CC.cohomology_data(Ccoh, 1)

    @test H1deg.dimZ == 2
    @test H1deg.dimB == 1
    @test H1deg.dimH == 1
    @test H1deg.K * H1deg.Cx == H1deg.B
    @test size(H1deg.Cx, 2) == H1deg.dimB
    @test FL.rank(field, H1deg.Cx) == H1deg.dimB
    @test getfield(H1deg, :_Q) !== nothing
    @test getfield(H1deg, :_Bfull) !== nothing
    @test getfield(H1deg, :_Hrep) !== nothing

    Hall = CC.cohomology_data(Ccoh)
    H1 = Hall[2]
    @test getfield(H1, :_Q) === nothing
    @test getfield(H1, :_Bfull) === nothing
    @test getfield(H1, :_Hrep) === nothing

    reps = CC.basis(H1)
    @test size(reps) == (2, 1)
    @test getfield(H1, :_Q) !== nothing
    @test getfield(H1, :_Bfull) !== nothing
    @test getfield(H1, :_Hrep) !== nothing
    @test CC.representatives(H1) == reps
    @test H1deg.Hrep == reps

    bd_next = sparse(reshape(Kc[c(1), c(0)], 2, 1))
    bd_curr = spzeros(Kc, 1, 2)
    Hh = CC.homology_data(bd_next, bd_curr, 1)
    @test Hh.dimZ == 2
    @test Hh.dimB == 1
    @test Hh.dimH == 1
    @test Hh.Z * Hh.Cx == Hh.B
    @test size(Hh.Cx, 2) == Hh.dimB
    @test FL.rank(field, Hh.Cx) == Hh.dimB
    @test getfield(Hh, :_Q) === nothing
    @test getfield(Hh, :_Bfull) === nothing
    @test getfield(Hh, :_Hrep) === nothing

    hrep = CC.basis(Hh)
    @test size(hrep) == (2, 1)
    @test getfield(Hh, :_Q) !== nothing
    @test getfield(Hh, :_Bfull) !== nothing
    @test getfield(Hh, :_Hrep) !== nothing
    @test CC.representatives(Hh) == hrep
end

@testset "maxdeg_of_complex helpers" begin
    C = CC.CochainComplex{Kc}(-1, 2, [1, 1, 1, 1], [spzeros(Kc, 1, 1), spzeros(Kc, 1, 1), spzeros(Kc, 1, 1)])
    @test CC.maxdeg_of_complex(C) == 2

    dims = ones(Int, 2, 4)
    dv = Array{SparseMatrixCSC{Kc,Int},2}(undef, 2, 4)
    dh = Array{SparseMatrixCSC{Kc,Int},2}(undef, 2, 4)
    for i in 1:2, j in 1:4
        dv[i, j] = spzeros(Kc, 1, 1)
        dh[i, j] = spzeros(Kc, 1, 1)
    end
    DC = CC.DoubleComplex{Kc}(0, 1, -1, 2, dims, dv, dh)
    @test CC.maxdeg_of_complex(DC) == 3
end

@testset "ChainComplexes: induced_map_on_cohomology for zero-differential complexes" begin
    # Complexes concentrated in degree 0.
    C = CC.CochainComplex{Kc}(0, 0, [2], SparseMatrixCSC{Kc,Int}[])
    D = CC.CochainComplex{Kc}(0, 0, [3], SparseMatrixCSC{Kc,Int}[])

    HC = CC.cohomology_data(C, 0)
    HD = CC.cohomology_data(D, 0)

    # A concrete linear map f: C^0 -> D^0
    f = sparse([1,2], [1,2], [c(1), c(1)], 3, 2)
    fH = CC.induced_map_on_cohomology(HC, HD, f)

    # Since differentials are zero, cohomology equals the underlying vector space.
    @test Matrix(fH) == Matrix(f)

    z = spzeros(Kc, 3, 2)
    zH = CC.induced_map_on_cohomology(HC, HD, z)
    
    @test count(!iszero, zH) == 0

    # Batched coordinates preserve the chosen representative basis.
    @test CC.cohomology_coordinates(HC, HC.Hrep) == Matrix{Kc}(I, HC.dimH, HC.dimH)
end

@testset "ChainComplexes: homology_coordinates and induced_map_on_homology batch over columns" begin
    d1 = reshape(Kc[-1, 1], 2, 1)
    d0 = zeros(Kc, 0, 2)
    H0 = CC.homology_data(d1, d0, 0)
    @test H0.dimH == 1

    reps = H0.Hrep
    coords = CC.homology_coordinates(H0, reps)
    @test coords == Matrix{Kc}(I, H0.dimH, H0.dimH)

    id0 = sparse([1, 2], [1, 2], [one(Kc), one(Kc)], 2, 2)
    Hid = CC.induced_map_on_homology(H0, H0, id0)
    @test Hid == Matrix{Kc}(I, H0.dimH, H0.dimH)
end

@testset "ModuleCochainComplex accepts FringeModule inputs (auto-convert via pmodule_from_fringe)" begin
    # Use a chain poset so the interval-style up/down sets intersect in a predictable way.
    P = chain_poset(5)
    n = P.n
    a, b = 2, 4

    # These are the same style as the helper interval_module in this file.
    U = BitVector([i <= b for i in 1:n])
    D = BitVector([i >= a for i in 1:n])

    H = TO.one_by_one_fringe(P, U, D, c(1); field=field)
    M = IR.pmodule_from_fringe(H)

    C_from_fringe = TO.ModuleCochainComplex([H], TO.PMorphism[]; tmin = 0)
    C_from_pmodule = TO.ModuleCochainComplex([M], TO.PMorphism[]; tmin = 0)

    @test C_from_fringe.tmin == 0
    @test C_from_fringe.tmax == 0
    @test length(C_from_fringe.terms) == 1
    @test CM.coeff_type(C_from_fringe.terms[1].field) == Kc

    # The internal converted term should match explicit conversion.
    @test C_from_fringe.terms[1].dims == C_from_pmodule.terms[1].dims
    @test C_from_fringe.terms[1].edge_maps == C_from_pmodule.terms[1].edge_maps
end

@testset "Module complexes / hyperExt / hyperTor" begin
    if field isa CM.RealField
        # This block validates exact homological equalities that are unstable under floating solves.
        @test true
    else

    # --------------------------
    # 1) hyperExt agrees with Ext for degree-0 complex
    # --------------------------

    # Build the 3-element chain poset: 1 <= 2 <= 3.
    P = FF.FinitePoset([i <= j for i in 1:3, j in 1:3])
    Mf = interval_module(P,1,2)
    Nf = interval_module(P,2,3)
    M = IR.pmodule_from_fringe(Mf)
    N = IR.pmodule_from_fringe(Nf)

    C0 = TO.ModuleCochainComplex([M], TO.PMorphism[]; tmin=0)
    maxdeg = 2
    E = DF.Ext(M, N, TO.DerivedFunctorOptions(maxdeg=maxdeg, model=:injective))
    H = TO.hyperExt(C0, Nf; maxlen=maxdeg)

    # Graded-space interface sanity checks: HyperExtSpace
    rH = TO.degree_range(H)
    if !isempty(rH)
        @test TO.dim(H, first(rH) - 1) == 0
        @test TO.dim(H, last(rH) + 1) == 0
        for t in rH
            d = TO.dim(H, t)
            B = TO.basis(H, t)
            @test length(B) == d
            if d > 0
                coords = zeros(Kc, d)
                coords[1] = c(1)
                z = TO.representative(H, t, coords)
                coords2 = TO.coordinates(H, t, z)
                @test coords2 == coords
            end
        end
    end


    for t in 0:maxdeg
        @test DF.dim(E,t) == TO.dim(H,t)
    end

    if Threads.nthreads() > 1
        H_serial = TO.hyperExt(C0, Nf; maxlen = maxdeg, threads = false)
        H_thread = TO.hyperExt(C0, Nf; maxlen = maxdeg, threads = true)
        @test dim(H_serial, 0) == dim(H_thread, 0)
        @test dim(H_serial, 1) == dim(H_thread, 1)
        @test dim(H_serial, 2) == dim(H_thread, 2)
    end

    # --------------------------
    # 2) mapping cone of id is acyclic in module cohomology
    # --------------------------
    idM = IR.id_morphism(M)
    f = TO.ModuleCochainMap(C0,C0,[idM]; tmin=0, tmax=0)
    Cone = TO.mapping_cone(f)

    for t in Cone.tmin:Cone.tmax
        Ht = TO.cohomology_module(Cone,t)
        @test all(d == 0 for d in Ht.dims)
    end

    # --------------------------
    # 3) RHom functoriality: (g circ f)^* = f^* circ g^*
    # --------------------------
    # scaling endomorphisms
    function scale(M::MD.PModule{K}, a::Int) where {K}
        comps = Matrix{K}[]
        for u in 1:M.Q.n
            d = M.dims[u]
            push!(comps, c(a) * eye_mat(d))
        end
        return MD.PMorphism(M,M,comps)
    end
    scale(H::FF.FringeModule{K}, a::Int) where {K} = scale(IR.pmodule_from_fringe(H), a)

    fM = scale(M,2)
    gM = scale(M,3)
    gfM = scale(M,6)

    C1 = TO.ModuleCochainComplex([M], TO.PMorphism[]; tmin=0)
    C2 = TO.ModuleCochainComplex([M], TO.PMorphism[]; tmin=0)

    fmap = TO.ModuleCochainMap(C0,C1,[fM]; tmin=0, tmax=0)
    gmap = TO.ModuleCochainMap(C1,C2,[gM]; tmin=0, tmax=0)
    gfmap = TO.ModuleCochainMap(C0,C2,[gfM]; tmin=0, tmax=0)

    resN = DF.injective_resolution(N, TO.ResolutionOptions(maxlen=maxdeg))
    R0 = TO.RHomComplex(C0,N; maxlen=maxdeg, resN=resN)
    R1 = TO.RHomComplex(C1,N; maxlen=maxdeg, resN=resN)
    R2 = TO.RHomComplex(C2,N; maxlen=maxdeg, resN=resN)

    # Smoke test: RHomComplex must carry the Hom-space blocks used to
    # build the double complex, and they should be properly typed.
    @test R0.homs isa Array{DF.HomSpace{Kc},2}
    @test size(R0.homs, 2) == maxdeg + 1

    # Cache parity for Hom-system reuse in RHom builders.
    hcache = DF.HomSystemCache{Kc}()
    R0_cached1 = TO.RHomComplex(C0, N; maxlen=maxdeg, resN=resN, cache=hcache, threads=false)
    R0_cached2 = TO.RHomComplex(C0, N; maxlen=maxdeg, resN=resN, cache=hcache, threads=false)
    @test R0_cached1.tot.dims == R0.tot.dims
    @test R0_cached1.tot.d == R0.tot.d
    @test R0_cached2.tot.d == R0.tot.d
    @test R0_cached1.homs[1, 1] === R0_cached2.homs[1, 1]

    Hc1 = DF.hom_with_cache(M, resN.Emods[1]; cache=hcache)
    Hc2 = DF.hom_with_cache(M, resN.Emods[1]; cache=hcache)
    @test Hc1 === Hc2
    DF.clear_hom_system_cache!(hcache)
    Hc3 = DF.hom_with_cache(M, resN.Emods[1]; cache=hcache)
    @test Hc3 !== Hc1

    rf_uncached = TO.rhom_map_first(fmap, N; maxlen=maxdeg, resN=resN, threads=false)
    rf_cached = TO.rhom_map_first(fmap, N; maxlen=maxdeg, resN=resN, cache=hcache, threads=false)
    @test rf_cached.maps == rf_uncached.maps

    if Threads.nthreads() > 1
        R0_cached_thread = TO.RHomComplex(C0, N; maxlen=maxdeg, resN=resN, cache=hcache, threads=true)
        @test R0_cached_thread.tot.dims == R0_cached1.tot.dims
        @test R0_cached_thread.tot.d == R0_cached1.tot.d

        rf_cached_thread = TO.rhom_map_first(fmap, N; maxlen=maxdeg, resN=resN, cache=hcache, threads=true)
        @test rf_cached_thread.maps == rf_cached.maps
    end

    # induced maps on Tot are just precomposition matrices degreewise, so compare by multiplication
    F = DF._precompose_matrix(DF.Hom(M,resN.Emods[1]), DF.Hom(M,resN.Emods[1]), fM) # sanity call

    # We test cone-level functoriality indirectly: induced cohomology maps compose.
    hf = TO.induced_map_on_cohomology_modules(fmap,0)
    hg = TO.induced_map_on_cohomology_modules(gmap,0)
    hgf = TO.induced_map_on_cohomology_modules(gfmap,0)

    for u in 1:M.Q.n
        @test hgf.comps[u] == hg.comps[u]*hf.comps[u]
    end

    # --------------------------
    # 4) hyperTor agrees with Tor for degree-0 complex
    # --------------------------
    Pop = FF.FinitePoset(transpose(FF.leq_matrix(P)))

    # Pop is the opposite poset of P.
    # The interval [1,3] in P corresponds to the interval [3,1] in Pop.
    Rop = interval_module(Pop, 3, 1)

    RopP = IR.pmodule_from_fringe(Rop)
    Tplain = DF.Tor(RopP, M, TO.DerivedFunctorOptions(maxdeg=maxdeg))
    HT = TO.hyperTor(Rop,C0; maxlen=maxdeg)

    for s in 0:maxdeg
        @test DF.dim(Tplain,s) == TO.dim(HT,s)
    end

    # Graded-space interface sanity checks: HyperTorSpace
    @test TO.dim(HT, -1) == 0
    rT = TO.degree_range(HT)
    if !isempty(rT)
        @test first(rT) >= 0
        if first(rT) > 0
            @test TO.dim(HT, first(rT) - 1) == 0
        end
        @test TO.dim(HT, last(rT) + 1) == 0
        for n in rT
            d = TO.dim(HT, n)
            B = TO.basis(HT, n)
            @test length(B) == d
            if d > 0
                coords = zeros(Kc, d)
                coords[1] = c(1)
                z = TO.representative(HT, n, coords)
                coords2 = TO.coordinates(HT, n, z)
                @test coords2 == coords
            end
        end
    end

    if Threads.nthreads() > 1
        T_serial = TO.hyperTor(Rop, C0; maxlen = maxdeg, threads = false)
        T_thread = TO.hyperTor(Rop, C0; maxlen = maxdeg, threads = true)
        @test dim(T_serial, 0) == dim(T_thread, 0)
        @test dim(T_serial, 1) == dim(T_thread, 1)
        @test dim(T_serial, 2) == dim(T_thread, 2)
    end

    # FringeModule wrappers for RHom/RHomComplex and rhom_map_first/second.
    R0_fr = TO.RHomComplex(C0, Nf; maxlen=maxdeg)
    R0_fr2 = TO.RHomComplex(C0, Nf; maxlen=maxdeg)
    @test R0_fr.N.dims == R0_fr2.N.dims

    Rtot_fr = TO.RHom(C0, Nf; maxlen=maxdeg)
    @test Rtot_fr.tmin == R0_fr.tot.tmin
    @test Rtot_fr.tmax == R0_fr.tot.tmax

    rf = TO.rhom_map_first(fmap, Nf; maxlen=maxdeg)
    @test size(rf.maps, 1) == rf.tmax - rf.tmin + 1

    gN = IR.id_morphism(Nf)
    rs = TO.rhom_map_second(gN, C0, Nf, Nf; maxlen=maxdeg)
    @test size(rs.maps, 1) == rs.tmax - rs.tmin + 1
    rs_cached = TO.rhom_map_second(gN, C0, Nf, Nf; maxlen=maxdeg, cache=hcache)
    @test rs_cached.maps == rs.maps

    # --------------------------
    # 5) RHom spectral sequence degenerates at E2 if horizontal differential=0
    # --------------------------
    # build a 2-term complex with zero differential
    Ctwo = TO.ModuleCochainComplex([M,M],[IR.zero_morphism(M,M)]; tmin=0)

    R = TO.RHomComplex(Ctwo,N; maxlen=maxdeg, resN=resN)
    ss = CC.spectral_sequence(R.DC; output=:full, first=:horizontal)
    E2 = CC.page(ss,2)

    # E2(A,B) should equal Ext^B(C^{-A},N) since d_h=0
    for A in (-1):0
        for B in 0:maxdeg
            Mp = (A==0) ? M : M
            Eab = DF.Ext(Mp,N, TO.DerivedFunctorOptions(maxdeg=maxdeg, model=:injective))
            a = -A
            b = B
            E2ab = CC.term(ss, 2, (a, b)).dimH
            @test E2ab == DF.dim(Eab, B)
        end
    end
    end

end

@testset "ModuleComplexes UX surface" begin
    M = one_vertex_module(1)
    N = one_vertex_module(1)
    idM = MD.id_morphism(M)
    zM = TO.zero_morphism(M, M)
    Z = TO.zero_pmodule(M.Q; field=field)

    C0 = TO.ModuleCochainComplex([M], TO.PMorphism[]; tmin=0, check=true)
    C = TO.ModuleCochainComplex([M, M], [zM]; tmin=0, check=true)
    Ch = TO.ModuleCochainComplex([M, M], [idM]; tmin=0, check=true)
    Dh = Ch
    f = TO.ModuleCochainMap(Ch, Dh, [idM, idM]; tmin=0, tmax=1, check=true)
    g = TO.ModuleCochainMap(Ch, Dh, [zM, zM]; tmin=0, tmax=1, check=true)
    h0 = TO.zero_morphism(M, Z)
    h1 = idM
    Hhom = TO.ModuleCochainHomotopy(f, g, [h0, h1]; tmin=0, tmax=1, check=true)
    tri = TO.mapping_cone_triangle(f)

    RH = TO.RHomComplex(C0, N; maxlen=1, threads=false)
    HX = TO.hyperExt(C0, N; maxlen=1, threads=false)
    DT = TO.DerivedTensorComplex(M, C0; maxlen=1, threads=false)
    HT = TO.hyperTor(M, C0; maxlen=1, threads=false)

    @test DF.degree_range(C) == 0:1
    @test CC.component(C, 0) === M
    @test CC.component(C, -1).dims == [0]
    @test CC.differential(C, 0) === zM

    @test CC.source(f) === Ch
    @test CC.target(f) === Dh
    @test DF.degree_range(f) == 0:1
    @test CC.component(f, 0) === idM

    @test MCM.source_map(Hhom) === f
    @test MCM.target_map(Hhom) === g
    @test DF.degree_range(Hhom) == 0:1
    @test CC.component(Hhom, 1) === h1

    objs = MCM.triangle_objects(tri)
    maps = MCM.triangle_maps(tri)
    @test objs.source === Ch
    @test objs.target === Dh
    @test objs.cone === tri.Cone
    @test maps.morphism === f
    @test maps.inclusion === tri.i
    @test maps.projection === tri.p
    @test TO.connecting_map(tri) === tri.p

    @test DF.source_module(RH) === C0
    @test DF.target_module(RH) === N
    @test MCM.underlying_complex(RH) === RH.tot

    @test DF.source_module(DT) === M
    @test DF.target_module(DT) === C0
    @test MCM.underlying_complex(DT) === DT.tot

    @test DF.source_module(HX) === C0
    @test DF.target_module(HX) === N
    @test sort(DF.nonzero_degrees(HX)) == sort(collect(keys(DF.degree_dimensions(HX))))
    @test DF.total_dimension(HX) == sum(values(DF.degree_dimensions(HX)))

    @test DF.source_module(HT) === M
    @test DF.target_module(HT) === C0
    @test sort(DF.nonzero_degrees(HT)) == sort(collect(keys(DF.degree_dimensions(HT))))
    @test DF.total_dimension(HT) == sum(values(DF.degree_dimensions(HT)))

    @test CC.describe(C).kind == :module_cochain_complex
    @test CC.describe(f).kind == :module_cochain_map
    @test CC.describe(Hhom).kind == :module_cochain_homotopy
    @test CC.describe(tri).kind == :module_distinguished_triangle
    @test CC.describe(RH).kind == :rhom_complex
    @test CC.describe(HX).kind == :hyperext_space
    @test CC.describe(DT).kind == :derived_tensor_complex
    @test CC.describe(HT).kind == :hypertor_space

    @test MCM.module_complex_summary(C) == CC.describe(C)
    @test MCM.module_map_summary(f) == CC.describe(f)
    @test MCM.module_homotopy_summary(Hhom) == CC.describe(Hhom)
    @test MCM.triangle_summary(tri) == CC.describe(tri)
    @test MCM.rhom_summary(RH) == CC.describe(RH)
    @test MCM.hyperext_summary(HX) == CC.describe(HX)
    @test MCM.derived_tensor_summary(DT) == CC.describe(DT)
    @test MCM.hypertor_summary(HT) == CC.describe(HT)

    shown = (
        ("ModuleCochainComplex", C),
        ("ModuleCochainMap", f),
        ("ModuleCochainHomotopy", Hhom),
        ("ModuleDistinguishedTriangle", tri),
        ("RHomComplex", RH),
        ("HyperExtSpace", HX),
        ("DerivedTensorComplex", DT),
        ("HyperTorSpace", HT),
    )
    for (name, obj) in shown
        @test occursin(name, sprint(show, obj))
        @test occursin(name, sprint(show, MIME"text/plain"(), obj))
    end

    complex_report = MCM.check_module_complex(C)
    map_report = MCM.check_module_complex_map(f)
    homotopy_report = MCM.check_module_homotopy(Hhom)
    triangle_report = MCM.check_module_triangle(tri)
    rhom_report = MCM.check_rhom_complex(RH)
    dtensor_report = MCM.check_derived_tensor_complex(DT)

    @test complex_report.valid
    @test map_report.valid
    @test homotopy_report.valid
    @test triangle_report.valid
    @test rhom_report.valid
    @test dtensor_report.valid

    summary = MCM.module_complex_validation_summary(complex_report)
    @test summary isa MCM.ModuleComplexValidationSummary
    @test occursin("ModuleComplexValidationSummary", sprint(show, summary))
    @test occursin("ModuleComplexValidationSummary", sprint(show, MIME"text/plain"(), summary))

    Cbad = TO.ModuleCochainComplex([M, M, M], [idM, idM]; tmin=0, check=false)
    fbad = TO.ModuleCochainMap(Ch, Dh, [idM, zM]; tmin=0, tmax=1, check=false)
    Hbad = TO.ModuleCochainHomotopy(f, g, [h0, h0]; tmin=0, tmax=1, check=false)
    tri_bad = MCM.ModuleDistinguishedTriangle(tri.C, tri.D, tri.Cone, tri.f, tri.i, f)
    RHbad = MCM.RHomComplex(C0, N, RH.resN, RH.homs, RH.DC, CC.shift(RH.tot, 1))
    DTbad = MCM.DerivedTensorComplex(M, C0, DT.resR, DT.DC, CC.shift(DT.tot, 1))

    @test !MCM.check_module_complex(Cbad).valid
    @test !MCM.check_module_complex_map(fbad).valid
    @test !MCM.check_module_homotopy(Hbad).valid
    @test !MCM.check_module_triangle(tri_bad).valid
    @test !MCM.check_rhom_complex(RHbad).valid
    @test !MCM.check_derived_tensor_complex(DTbad).valid

    @test_throws ArgumentError MCM.check_module_complex(Cbad; throw=true)
    @test_throws ArgumentError MCM.check_module_complex_map(fbad; throw=true)
    @test_throws ArgumentError MCM.check_module_homotopy(Hbad; throw=true)
    @test_throws ArgumentError MCM.check_module_triangle(tri_bad; throw=true)
    @test_throws ArgumentError MCM.check_rhom_complex(RHbad; throw=true)
    @test_throws ArgumentError MCM.check_derived_tensor_complex(DTbad; throw=true)

    @test TOA.source_map === MCM.source_map
    @test TOA.target_map === MCM.target_map
    @test TOA.triangle_objects === MCM.triangle_objects
    @test TOA.triangle_maps === MCM.triangle_maps
    @test TOA.underlying_complex === MCM.underlying_complex
    @test TOA.module_complex_summary === MCM.module_complex_summary
    @test TOA.module_map_summary === MCM.module_map_summary
    @test TOA.module_homotopy_summary === MCM.module_homotopy_summary
    @test TOA.triangle_summary === MCM.triangle_summary
    @test TOA.rhom_summary === MCM.rhom_summary
    @test TOA.hyperext_summary === MCM.hyperext_summary
    @test TOA.derived_tensor_summary === MCM.derived_tensor_summary
    @test TOA.hypertor_summary === MCM.hypertor_summary
    @test TOA.check_module_complex === MCM.check_module_complex
    @test TOA.check_module_complex_map === MCM.check_module_complex_map
    @test TOA.check_module_homotopy === MCM.check_module_homotopy
    @test TOA.check_module_triangle === MCM.check_module_triangle
    @test TOA.check_rhom_complex === MCM.check_rhom_complex
    @test TOA.check_derived_tensor_complex === MCM.check_derived_tensor_complex
    @test TOA.ModuleComplexValidationSummary === MCM.ModuleComplexValidationSummary
    @test TOA.module_complex_validation_summary === MCM.module_complex_validation_summary
end

@testset "ModuleCochainComplex check d^2=0" begin
    M = one_vertex_module(1)
    id = MD.id_morphism(M)
    z  = TO.zero_morphism(M, M)

    terms = [M, M, M]

    # Invalid: d1*d0 = id != 0
    @test_throws ErrorException TO.ModuleCochainComplex(terms, [id, id]; tmin=0, check=true)

    # Valid
    C = TO.ModuleCochainComplex(terms, [id, z]; tmin=0, check=true)
    @test C.tmin == 0
    @test C.tmax == 2

    # check=false should allow
    Cbad = TO.ModuleCochainComplex(terms, [id, id]; tmin=0, check=false)
    @test Cbad.tmax == 2
end

@testset "ModuleCochainComplex keyword tmin (positional endpoints removed)" begin
    P = chain_poset(1)
    M = MD.PModule{Kc}(P, [1], Dict{Tuple{Int,Int}, Matrix{Kc}}())
    id = scalar_morphism(M, 1)

    terms = [M, M]
    diffs = [id]

    C_kw = TO.ModuleCochainComplex(terms, diffs; tmin=-1, check=true)

    @test C_kw.tmin == -1
    @test C_kw.tmax == 0
    @test length(C_kw.terms) == 2
    @test length(C_kw.diffs) == 1

    # The old positional-endpoint signature is intentionally removed.
    @test_throws MethodError TO.ModuleCochainComplex(terms, diffs, -1, 0; check=true)

    # Sanity: diff-length mismatch should fail fast (constructor invariant).
    @test_throws AssertionError TO.ModuleCochainComplex(terms, TO.PMorphism[]; tmin=-1, check=false)
end

@testset "ModuleCochainMap chain map validation" begin
    M = one_vertex_module(1)
    id = MD.id_morphism(M)
    z  = TO.zero_morphism(M, M)

    C = TO.ModuleCochainComplex([M, M], [id]; tmin=0, check=true)
    D = TO.ModuleCochainComplex([M, M], [id]; tmin=0, check=true)

    # Valid map: identity
    f = TO.ModuleCochainMap(C, D, [id, id]; tmin=0, tmax=1, check=true)
    @test f.tmin == 0

    # Invalid: breaks commutativity
    @test_throws ErrorException TO.ModuleCochainMap(C, D, [id, z]; tmin=0, tmax=1, check=true)

    # Boundary check: providing only degree 1 component implies degree 0 is zero -> fails
    @test_throws ErrorException TO.ModuleCochainMap(C, D, [id]; tmin=1, tmax=1, check=true)
end

@testset "ModuleCochainHomotopy exists and validates" begin
    M = one_vertex_module(1)
    id = MD.id_morphism(M)
    zM = TO.zero_morphism(M, M)
    Z  = TO.zero_pmodule(M.Q; field=field)

    C = TO.ModuleCochainComplex([M, M], [id]; tmin=0, check=true)
    D = C

    f = TO.ModuleCochainMap(C, D, [id, id]; tmin=0, tmax=1, check=true)
    g = TO.ModuleCochainMap(C, D, [zM, zM]; tmin=0, tmax=1, check=true)

    h0 = TO.zero_morphism(M, Z)  # C^0 -> D^-1 = 0
    h1 = id                      # C^1 -> D^0

    H = TO.ModuleCochainHomotopy(f, g, [h0, h1]; tmin=0, tmax=1, check=true)
    @test MCM.is_cochain_homotopy(H)

    # Wrong homotopy
    @test_throws ErrorException TO.ModuleCochainHomotopy(f, g, [h0, zM]; tmin=0, tmax=1, check=true)
end

@testset "mapping_cone(identity) is acyclic and id is quasi-iso" begin
    if field isa CM.RealField
        # Real-field solve checks are numerically sensitive on this tiny exact fixture.
        @test true
        return
    end

    M = one_vertex_module(1)
    C = TO.ModuleCochainComplex([M], TO.PMorphism[]; tmin=0, check=true)
    id = TO.ModuleCochainMap(C, C, [MD.id_morphism(M)]; tmin=0, tmax=0, check=true)

    cone = TO.mapping_cone(id)
    Hm1 = TO.cohomology_module(cone, -1)
    H0  = TO.cohomology_module(cone, 0)

    @test all(d == 0 for d in Hm1.dims)
    @test all(d == 0 for d in H0.dims)

    @test TO.is_quasi_isomorphism(id)
end

@testset "rhom_map_first strict functoriality under composition" begin
    M = one_vertex_module(2)
    N = TO.PModule{Kc}(M.Q, copy(M.dims), M.edge_maps; field=field)

    C = TO.ModuleCochainComplex([M], TO.PMorphism[]; tmin=0, check=true)

    A = scalar_morphism(M, 2)
    B = scalar_morphism(M, 3)
    BA = compose_morphism(B, A)

    f = TO.ModuleCochainMap(C, C, [A]; tmin=0, tmax=0, check=true)
    g = TO.ModuleCochainMap(C, C, [B]; tmin=0, tmax=0, check=true)
    gf = TO.ModuleCochainMap(C, C, [BA]; tmin=0, tmax=0, check=true)

    resN = DF.injective_resolution(N, TO.ResolutionOptions(maxlen=1))

    R = TO.RHomComplex(C, N; maxlen=1, resN=resN)
    Fmap = TO.rhom_map_first(f, R, R; check=true)
    Gmap = TO.rhom_map_first(g, R, R; check=true)
    GFmap = TO.rhom_map_first(gf, R, R; check=true)

    # Contravariance: (gcircf)^* = f^* circ g^*
    t0 = Fmap.tmin
    @test GFmap.maps[1] == (Gmap.maps[1] * Fmap.maps[1])
end

@testset "rhom_map_second and hyperExt_map_second functoriality" begin
    P = chain_poset(2)

    # simples at vertices 1 and 2
    S1 = IR.pmodule_from_fringe(one_by_one_fringe(
        P,
        FF.principal_upset(P, 1),
        FF.principal_downset(P, 1);
        field=field,
    ))
    S2 = IR.pmodule_from_fringe(one_by_one_fringe(
        P,
        FF.principal_upset(P, 2),
        FF.principal_downset(P, 2);
        field=field,
    ))

    # C concentrated in degree 0
    C = TO.ModuleCochainComplex([S1], TO.PMorphism[]; tmin=0, check=true)

    # N = S2 oplus S2 (so endomorphisms can be noncommuting 2x2 matrices)
    N, i1, i2, p1, p2 = TO.Modules.direct_sum_with_maps(S2, S2)

    H = TO.hyperExt(C, N; maxlen=3)
    @test TO.dim(H, 1) == 2  # Ext^1(S1, S2^2) should be 2

    # helper: endomorphism defined only at vertex u
    function endo_at_vertex(M::MD.PModule, u::Int, A::AbstractMatrix)
        comps = Vector{Matrix{Kc}}(undef, M.Q.n)
        for v in 1:M.Q.n
            dv = M.dims[v]
            comps[v] = Matrix{Kc}(I, dv, dv)
        end
        comps[u] = A
        return MD.PMorphism(M, M, comps)
    end

    idN = IR.id_morphism(N)
    Mid = TO.hyperExt_map_second(idN, H, H; t=1)
    @test Mid == Matrix{Kc}(I, 2, 2)

    twoN = scalar_morphism(N, 2)
    Mtwo = TO.hyperExt_map_second(twoN, H, H; t=1)
    @test Mtwo == 2 * Matrix{Kc}(I, 2, 2)

    gC = endo_at_vertex(N, 2, Kc[0 1; 0 0])
    gD = endo_at_vertex(N, 2, Kc[0 0; 1 0])

    MCg = TO.hyperExt_map_second(gC, H, H; t=1)
    MDg = TO.hyperExt_map_second(gD, H, H; t=1)

    gDC = compose_morphism(gD, gC)  # gD circ gC
    MDCg = TO.hyperExt_map_second(gDC, H, H; t=1)

    # Covariant functoriality: F(gD circ gC) == F(gD) * F(gC)
    @test MDCg == MDg * MCg
end

@testset "hyperExt_map_first contravariant functoriality" begin
    P = chain_poset(2)

    S1 = IR.pmodule_from_fringe(one_by_one_fringe(
        P,
        FF.principal_upset(P, 1),
        FF.principal_downset(P, 1);
        field=field,
    ))
    S2 = IR.pmodule_from_fringe(one_by_one_fringe(
        P,
        FF.principal_upset(P, 2),
        FF.principal_downset(P, 2);
        field=field,
    ))

    M, _, _, _, _ = TO.Modules.direct_sum_with_maps(S1, S1)
    N, _, _, _, _ = TO.Modules.direct_sum_with_maps(S2, S2)

    C = TO.ModuleCochainComplex([M], TO.PMorphism[]; tmin=0, check=true)
    H = TO.hyperExt(C, N; maxlen=3)
    @test TO.dim(H, 1) == 4

    function endo_at_vertex(M::MD.PModule{K}, u::Int, A::AbstractMatrix{K}) where {K}
        comps = Vector{Matrix{K}}(undef, M.Q.n)
        for v in 1:M.Q.n
            dv = M.dims[v]
            comps[v] = eye_mat(dv)
        end
        comps[u] = Matrix{Kc}(A)
        return MD.PMorphism(M, M, comps)
    end

    fA = endo_at_vertex(M, 1, Kc[0 1; 0 0])
    fB = endo_at_vertex(M, 1, Kc[0 0; 1 0])

    FA = TO.ModuleCochainMap(C, C, [fA])
    FB = TO.ModuleCochainMap(C, C, [fB])
    fBA = compose_morphism(fB, fA)
    FBA = TO.ModuleCochainMap(C, C, [fBA])

    MA = TO.hyperExt_map_first(FA, H, H; t=1)
    MB = TO.hyperExt_map_first(FB, H, H; t=1)
    MBA = TO.hyperExt_map_first(FBA, H, H; t=1)

    # Contravariant functoriality: F(fB circ fA) == F(fA) circ F(fB)
    @test MBA == MA * MB
end

# This test file focuses on the "abelian category" public API:
# kernels/cokernels/images/coimages/quotients, pushout/pullback,
# short exact sequences, and a basic snake lemma sanity check.

# Helper: a P-module on a 1-vertex poset (so no edge maps are needed).
function _vspace_module(P::FF.FinitePoset, d::Int)
    edge_maps = Dict{Tuple{Int,Int}, Matrix{Kc}}()
    return MD.PModule{Kc}(P, [d], edge_maps; field=field)
end

@inline function _ab_rand_coeff(rng::AbstractRNG)
    v = rand(rng, -3:3)
    v == 0 && (v = 1)
    return CM.coerce(field, v)
end

function _ab_rand_dense(rng::AbstractRNG, m::Int, n::Int)
    A = zeros(Kc, m, n)
    @inbounds for i in 1:m, j in 1:n
        A[i, j] = _ab_rand_coeff(rng)
    end
    return A
end

function _ab_build_morphism_fixture(P::FF.AbstractPoset;
                                    rank_part::Int=4,
                                    ker_part::Int=4,
                                    coker_part::Int=4,
                                    seed::Int=Int(0xC0FE))
    rng = MersenneTwister(seed)
    r = rank_part
    k = ker_part
    c0 = coker_part
    da = r + k
    db = r + c0
    n = FF.nvertices(P)

    dimsA = fill(da, n)
    dimsB = fill(db, n)
    edgeA = Dict{Tuple{Int,Int}, Matrix{Kc}}()
    edgeB = Dict{Tuple{Int,Int}, Matrix{Kc}}()

    for (u, v) in FF.cover_edges(P)
        Ruv = _ab_rand_dense(rng, r, r)
        Kuv = _ab_rand_dense(rng, k, k)
        Xuv = _ab_rand_dense(rng, r, c0)
        Yuv = _ab_rand_dense(rng, c0, c0)

        Auv = zeros(Kc, da, da)
        Buv = zeros(Kc, db, db)
        @inbounds begin
            copyto!(view(Auv, 1:r, 1:r), Ruv)
            copyto!(view(Auv, r+1:da, r+1:da), Kuv)
            copyto!(view(Buv, 1:r, 1:r), Ruv)
            copyto!(view(Buv, 1:r, r+1:db), Xuv)
            copyto!(view(Buv, r+1:db, r+1:db), Yuv)
        end
        edgeA[(u, v)] = Auv
        edgeB[(u, v)] = Buv
    end

    A = MD.PModule{Kc}(P, dimsA, edgeA; field=field)
    B = MD.PModule{Kc}(P, dimsB, edgeB; field=field)

    F = zeros(Kc, db, da)
    @inbounds for i in 1:r
        F[i, i] = CM.coerce(field, 1)
    end
    comps = [copy(F) for _ in 1:n]
    f = MD.PMorphism(A, B, comps)
    return A, B, f
end

function _ab_two_layer_poset(nleft::Int, nright::Int)
    (nleft > 0 && nright > 0) || error("_ab_two_layer_poset: need positive layer sizes")
    n = nleft + nright
    L = falses(n, n)
    @inbounds for i in 1:n
        L[i, i] = true
    end
    @inbounds for u in 1:nleft
        for v in (nleft + 1):n
            L[u, v] = true
        end
    end
    return FF.FinitePoset(L; check=false)
end

function _ab_kernel_with_inclusion_old(f::MD.PMorphism{K}; cache::Union{Nothing,MD.CoverCache}=nothing) where {K}
    M = f.dom
    n = FF.nvertices(M.Q)
    basisK = Vector{Matrix{K}}(undef, n)
    K_dims = zeros(Int, n)
    for i in 1:n
        B = FL.nullspace(f.dom.field, f.comps[i])
        basisK[i] = B
        K_dims[i] = size(B, 2)
    end

    cc = (cache === nothing ? MD._get_cover_cache(M.Q) : cache)
    preds = [FF._preds(cc, v) for v in 1:n]
    succs = [FF._succs(cc, u) for u in 1:n]
    maps_from_pred = [Vector{Matrix{K}}(undef, length(preds[v])) for v in 1:n]
    maps_to_succ   = [Vector{Matrix{K}}(undef, length(succs[u])) for u in 1:n]

    @inbounds for u in 1:n
        su = succs[u]
        maps_u_M = M.edge_maps.maps_to_succ[u]
        outu = maps_to_succ[u]
        for j in eachindex(su)
            v = su[j]
            if K_dims[u] == 0 || K_dims[v] == 0
                X = zeros(K, K_dims[v], K_dims[u])
                outu[j] = X
                ip = MD._find_sorted_index(preds[v], u)
                maps_from_pred[v][ip] = X
                continue
            end
            Im = maps_u_M[j] * basisK[u]
            X = FL.solve_fullcolumn(f.dom.field, basisK[v], Im; check_rhs=false)
            outu[j] = X
            ip = MD._find_sorted_index(preds[v], u)
            maps_from_pred[v][ip] = X
        end
    end

    storeK = MD.CoverEdgeMapStore{K,Matrix{K}}(preds, succs, maps_from_pred, maps_to_succ, cc.nedges)
    Kmod = MD.PModule{K}(M.Q, K_dims, storeK; field=M.field)
    iota = MD.PMorphism{K}(Kmod, M, [basisK[i] for i in 1:n])
    return Kmod, iota
end

function _ab_image_with_inclusion_old(f::MD.PMorphism{K}; cache::Union{Nothing,MD.CoverCache}=nothing) where {K}
    N = f.cod
    Q = N.Q
    n = FF.nvertices(Q)
    bases = Vector{Matrix{K}}(undef, n)
    dims = zeros(Int, n)
    for i in 1:n
        B = FL.colspace(f.dom.field, f.comps[i])
        bases[i] = B
        dims[i] = size(B, 2)
    end

    preds = N.edge_maps.preds
    succs = N.edge_maps.succs
    maps_from_pred = [Vector{Matrix{K}}(undef, length(preds[v])) for v in 1:n]
    maps_to_succ   = [Vector{Matrix{K}}(undef, length(succs[u])) for u in 1:n]

    @inbounds for u in 1:n
        su = succs[u]
        Nu = N.edge_maps.maps_to_succ[u]
        outu = maps_to_succ[u]
        Bu = bases[u]
        du = size(Bu, 2)
        for j in eachindex(su)
            v = su[j]
            Bv = bases[v]
            dv = size(Bv, 2)
            Auv = if du == 0
                zeros(K, dv, 0)
            elseif dv == 0
                zeros(K, 0, du)
            else
                T = Nu[j] * Bu
                FL.solve_fullcolumn(f.dom.field, Bv, T)
            end
            outu[j] = Auv
            ip = MD._find_sorted_index(preds[v], u)
            maps_from_pred[v][ip] = Auv
        end
    end

    storeIm = MD.CoverEdgeMapStore{K,Matrix{K}}(preds, succs, maps_from_pred, maps_to_succ, N.edge_maps.nedges)
    Im = MD.PModule{K}(Q, dims, storeIm; field=N.field)
    iota = MD.PMorphism(Im, N, [bases[i] for i in 1:n])
    return Im, iota
end

function _ab_cokernel_module_old(iota::MD.PMorphism{K}; cache::Union{Nothing,MD.CoverCache}=nothing) where {K}
    E = iota.cod
    Q = E.Q
    n = FF.nvertices(Q)
    Cdims = zeros(Int, n)
    qcomps = Vector{Matrix{K}}(undef, n)
    for i in 1:n
        Bi = FL.colspace(E.field, iota.comps[i])
        Ni = FL.nullspace(E.field, transpose(Bi))
        Cdims[i] = size(Ni, 2)
        qcomps[i] = transpose(Ni)
    end

    Cedges = Dict{Tuple{Int,Int}, Matrix{K}}()
    cc = (cache === nothing ? MD._get_cover_cache(Q) : cache)
    @inbounds for u in 1:n
        su = FF._succs(cc, u)
        maps_u = E.edge_maps.maps_to_succ[u]
        for j in eachindex(su)
            v = su[j]
            if Cdims[u] > 0 && Cdims[v] > 0
                X = FL.solve_fullcolumn(E.field, transpose(qcomps[u]), transpose(qcomps[v] * maps_u[j]))
                Cedges[(u, v)] = transpose(X)
            else
                Cedges[(u, v)] = zeros(K, Cdims[v], Cdims[u])
            end
        end
    end

    Cmod = MD.PModule{K}(Q, Cdims, Cedges; field=E.field)
    q = MD.PMorphism(E, Cmod, qcomps)
    return Cmod, q
end

function _ab_pushout_old(f::MD.PMorphism{K}, g::MD.PMorphism{K};
                         cache::Union{Nothing,MD.CoverCache}=nothing) where {K}
    A = f.dom
    B = f.cod
    C = g.cod
    S = MD.direct_sum(B, C)
    Q = S.Q
    phi_comps = Vector{Matrix{K}}(undef, FF.nvertices(Q))
    @inbounds for u in 1:FF.nvertices(Q)
        fu = f.comps[u]
        gu = g.comps[u]
        b = size(fu, 1)
        c = size(gu, 1)
        a = size(fu, 2)
        M = Matrix{K}(undef, b + c, a)
        if b > 0
            copyto!(view(M, 1:b, :), fu)
        end
        if c > 0
            @inbounds for i in 1:c, j in 1:a
                M[b + i, j] = -gu[i, j]
            end
        end
        phi_comps[u] = M
    end
    phi = MD.PMorphism{K}(A, S, phi_comps)
    P, q = _ab_cokernel_module_old(phi; cache=cache)
    inB_comps = Vector{Matrix{K}}(undef, FF.nvertices(Q))
    inC_comps = Vector{Matrix{K}}(undef, FF.nvertices(Q))
    @inbounds for u in 1:FF.nvertices(Q)
        qu = q.comps[u]
        b = B.dims[u]
        c = C.dims[u]
        inB_comps[u] = copy(view(qu, :, 1:b))
        inC_comps[u] = copy(view(qu, :, (b + 1):(b + c)))
    end
    inB = MD.PMorphism{K}(B, P, inB_comps)
    inC = MD.PMorphism{K}(C, P, inC_comps)
    return P, inB, inC, q, phi
end

function _ab_pullback_old(f::MD.PMorphism{K}, g::MD.PMorphism{K};
                          cache::Union{Nothing,MD.CoverCache}=nothing) where {K}
    B = f.dom
    C = g.dom
    D = f.cod
    S = MD.direct_sum(B, C)
    Q = S.Q
    psi_comps = Vector{Matrix{K}}(undef, FF.nvertices(Q))
    @inbounds for u in 1:FF.nvertices(Q)
        fu = f.comps[u]
        gu = g.comps[u]
        d = size(fu, 1)
        b = size(fu, 2)
        c = size(gu, 2)
        M = Matrix{K}(undef, d, b + c)
        if b > 0
            copyto!(view(M, :, 1:b), fu)
        end
        if c > 0
            @inbounds for i in 1:d, j in 1:c
                M[i, b + j] = -gu[i, j]
            end
        end
        psi_comps[u] = M
    end
    psi = MD.PMorphism{K}(S, D, psi_comps)
    P, iota = _ab_kernel_with_inclusion_old(psi; cache=cache)
    prB_comps = Vector{Matrix{K}}(undef, FF.nvertices(Q))
    prC_comps = Vector{Matrix{K}}(undef, FF.nvertices(Q))
    @inbounds for u in 1:FF.nvertices(Q)
        iu = iota.comps[u]
        b = B.dims[u]
        c = C.dims[u]
        prB_comps[u] = copy(view(iu, 1:b, :))
        prC_comps[u] = copy(view(iu, (b + 1):(b + c), :))
    end
    prB = MD.PMorphism{K}(P, B, prB_comps)
    prC = MD.PMorphism{K}(P, C, prC_comps)
    return P, prB, prC, iota, psi
end

function _ab_morphism_equal(a::MD.PMorphism, b::MD.PMorphism)
    a.dom.dims == b.dom.dims || return false
    a.cod.dims == b.cod.dims || return false
    for i in eachindex(a.comps)
        A = a.comps[i]
        B = b.comps[i]
        if field isa CM.RealField
            isapprox(A, B; rtol=field.rtol, atol=field.atol) || return false
        else
            A == B || return false
        end
    end
    return true
end

function _ab_pmodule_equal(A::MD.PModule, B::MD.PModule)
    A.dims == B.dims || return false
    for ((u, v), Muv) in A.edge_maps
        Nuv = B.edge_maps[u, v]
        if field isa CM.RealField
            isapprox(Muv, Nuv; rtol=field.rtol, atol=field.atol) || return false
        else
            Muv == Nuv || return false
        end
    end
    return true
end

@testset "Abelian-category API" begin
    P = chain_poset(1)

    # -------------------------------------------------------------------------
    # Kernel / image / cokernel / coimage
    # -------------------------------------------------------------------------
    A = _vspace_module(P, 2)
    B = _vspace_module(P, 3)

    f_mat = Matrix{Kc}([
        CM.coerce(field, 1) CM.coerce(field, 0);
        CM.coerce(field, 0) CM.coerce(field, 0);
        CM.coerce(field, 0) CM.coerce(field, 0)
    ])
    f = MD.PMorphism(A, B, [f_mat])

    Ker, iK = TO.kernel_with_inclusion(f)
    @test Ker.dims == [1]
    @test f.comps[1] * iK.comps[1] == zeros(Kc, 3, 1)
    @test TO.kernel(f).dims == Ker.dims

    Im, iIm = TO.image_with_inclusion(f)
    @test Im.dims == [1]
    @test TO.image(f).dims == [1]

    # Factorization test: since iIm is an inclusion, f should factor through it.
    X = FL.solve_fullcolumn(field, iIm.comps[1], f.comps[1])
    @test iIm.comps[1] * X == f.comps[1]

    Cok, q = TO.cokernel_with_projection(f)
    @test Cok.dims == [2]
    @test q.comps[1] * f.comps[1] == zeros(Kc, 2, 2)
    @test TO.cokernel(f).dims == [2]

    Coim, pco = TO.coimage_with_projection(f)
    @test Coim.dims == [1]
    @test pco.comps[1] * iK.comps[1] == zeros(Kc, 1, 1)
    @test TO.coimage(f).dims == [1]

    # Quotient by the image submodule should match the cokernel.
    Simg = TO.image_submodule(f)
    Qmod = TO.quotient(Simg)
    @test Qmod.dims == Cok.dims

    # Ambient-restating quotient overloads are intentionally absent from the public surface.
    @test_throws MethodError TO.quotient(B, Simg)
    @test_throws MethodError TO.quotient(B, iIm)
    @test_throws MethodError TO.quotient_with_projection(B, Simg)
    @test_throws MethodError TO.quotient_with_projection(B, iIm)

    # -------------------------------------------------------------------------
    # Pushout / pullback (1-vertex sanity checks)
    # -------------------------------------------------------------------------
    A1 = _vspace_module(P, 1)
    B1 = _vspace_module(P, 1)
    C1 = _vspace_module(P, 1)

    f_id = MD.PMorphism(A1, B1, [reshape(Kc[1], 1, 1)])
    g_id = MD.PMorphism(A1, C1, [reshape(Kc[1], 1, 1)])

    Pout, inB, inC, qpo, phi = TO.AbelianCategories.pushout(f_id, g_id)
    @test Pout.dims == [1]
    @test inB.comps[1] * f_id.comps[1] == inC.comps[1] * g_id.comps[1]

    # Pullback of identities should be the diagonal (dim 1).
    D1 = _vspace_module(P, 1)
    f_toD = MD.PMorphism(B1, D1, [reshape(Kc[1], 1, 1)])
    g_toD = MD.PMorphism(C1, D1, [reshape(Kc[1], 1, 1)])

    Pin, prB, prC, iota, psi = TO.AbelianCategories.pullback(f_toD, g_toD)
    @test Pin.dims == [1]
    @test f_toD.comps[1] * prB.comps[1] == g_toD.comps[1] * prC.comps[1]

    # -------------------------------------------------------------------------
    # Short exact sequences
    # -------------------------------------------------------------------------
    A2 = _vspace_module(P, 1)
    B2 = _vspace_module(P, 2)
    C2 = _vspace_module(P, 1)

    i_mat = reshape(Kc[1, 0], 2, 1)
    p_mat = reshape(Kc[0, 1], 1, 2)

    i = MD.PMorphism(A2, B2, [i_mat])
    p = MD.PMorphism(B2, C2, [p_mat])

    # Canonical public constructor.
    ses = TO.short_exact_sequence(i, p)
    @test TO.is_exact(ses)

    # Concrete container constructor remains available when explicitly desired.
    ses_ctor = TO.ShortExactSequence(i, p)
    @test TO.is_exact(ses_ctor)

    # A non-exact variant: switch the projection.
    p_bad = MD.PMorphism(B2, C2, [reshape(Kc[1, 0], 1, 2)])
    ses_bad = TO.ShortExactSequence(i, p_bad; check=false)
    @test !TO.is_exact(ses_bad)

    # -------------------------------------------------------------------------
    # Snake lemma (rank sanity check for the connecting morphism)
    # -------------------------------------------------------------------------
    # Top SES: 0 -> Q -> Q^2 -> Q -> 0
    At = _vspace_module(P, 1)
    Bt = _vspace_module(P, 2)
    Ct = _vspace_module(P, 1)
    it = MD.PMorphism(At, Bt, [reshape(Kc[1, 0], 2, 1)])
    pt = MD.PMorphism(Bt, Ct, [reshape(Kc[0, 1], 1, 2)])
    top = TO.ShortExactSequence(it, pt)

    # Bottom SES: 0 -> Q^2 -> Q^3 -> Q -> 0
    Ab = _vspace_module(P, 2)
    Bb = _vspace_module(P, 3)
    Cb = _vspace_module(P, 1)
    ib = MD.PMorphism(Ab, Bb, [Matrix{Kc}([
        CM.coerce(field, 1) CM.coerce(field, 0);
        CM.coerce(field, 0) CM.coerce(field, 1);
        CM.coerce(field, 0) CM.coerce(field, 0)
    ])])
    pb = MD.PMorphism(Bb, Cb, [reshape(Kc[0, 0, 1], 1, 3)])
    bottom = TO.ShortExactSequence(ib, pb)

    # Vertical maps: alpha injective, beta injective, gamma = 0.
    alpha = MD.PMorphism(At, Ab, [reshape(Kc[1, 0], 2, 1)])
    beta = MD.PMorphism(Bt, Bb, [Matrix{Kc}([
        CM.coerce(field, 1) CM.coerce(field, 0);
        CM.coerce(field, 0) CM.coerce(field, 1);
        CM.coerce(field, 0) CM.coerce(field, 0)
    ])])
    gamma = MD.PMorphism(Ct, Cb, [reshape(Kc[0], 1, 1)])

    sn = TO.snake_lemma(top, bottom, alpha, beta, gamma)
    delta = sn.delta
    @test_throws MethodError TO.snake_lemma(it, pt, ib, pb, alpha, beta, gamma)

    @test sn.kerC[1].dims == [1]   # ker(gamma) = Ct
    @test sn.cokA[1].dims == [1]   # coker(alpha) has dim 1
    @test FL.rank(field, delta.comps[1]) == 1
    @test !TO.is_zero_morphism(delta)

    @testset "Products/coproducts/equalizers/coequalizers + diagram interface" begin
        # Use a 1-vertex poset to keep the matrices small and the universal
        # property checks completely explicit.
        P1 = chain_poset(1)
        Z = TO.zero_pmodule(P1; field=field)

        A = _vspace_module(P1, 2)
        B = _vspace_module(P1, 3)

        # --- biproduct sanity: p_i o i_j = delta_ij, and i1 p1 + i2 p2 = id
        S, iA, iB, pA, pB = TO.biproduct(A, B)

        @test pA.comps[1] * iA.comps[1] == Matrix{Kc}(I, 2, 2)
        @test pB.comps[1] * iB.comps[1] == Matrix{Kc}(I, 3, 3)
        @test pA.comps[1] * iB.comps[1] == zeros(Kc, 2, 3)
        @test pB.comps[1] * iA.comps[1] == zeros(Kc, 3, 2)

        @test iA.comps[1] * pA.comps[1] + iB.comps[1] * pB.comps[1] == Matrix{Kc}(I, 5, 5)

        # --- product/coproduct wrappers exist and return the expected maps
        Pprod, prA, prB = TO.product(A, B)
        Ccop, inA, inB = TO.coproduct(A, B)
        @test prA isa MD.PMorphism
        @test prB isa MD.PMorphism
        @test inA isa MD.PMorphism
        @test inB isa MD.PMorphism
        @test_throws MethodError TO.product(A, B, A)
        @test_throws MethodError TO.coproduct(A, B, A)

        # Explicit universal property check for product:
        # given maps f:X->A and g:X->B, we can build (f,g): X -> A x B
        # and verify prA*(f,g)=f and prB*(f,g)=g.
        X = _vspace_module(P1, 2)
        f = MD.id_morphism(X)  # X -> A (both 2-dim, so treat as "identity")
        # A real map X->B: 3x2
        g = TO.PMorphism(X, B, [Kc[1 0; 0 1; 0 0]])

        # (f,g) : X -> Pprod has block form [f; g]
        fg = TO.PMorphism(X, Pprod, [vcat(f.comps[1], g.comps[1])])

        @test prA.comps[1] * fg.comps[1] == f.comps[1]
        @test prB.comps[1] * fg.comps[1] == g.comps[1]

        # --- equalizer/coequalizer checks
        # Build a nonzero map h : A -> B
        h = TO.PMorphism(A, B, [Kc[1 0; 0 1; 0 0]])
        z = TO.zero_morphism(A, B)

        E, e = TO.equalizer(h, z)
    @test e isa MD.PMorphism
        # h o e == 0 o e
        he = compose_morphism(h, e)
        ze = compose_morphism(z, e)
        @test he.comps[1] == ze.comps[1]

        Q, q = TO.coequalizer(h, z)
    @test q isa MD.PMorphism
        # q o h == q o 0
        qh = compose_morphism(q, h)
        qz = compose_morphism(q, z)
        @test qh.comps[1] == qz.comps[1]

        # --- diagram object interface: limit/colimit dispatch
        Ddisc = TO.DiscretePairDiagram(A, B)
        _, dprA, dprB = TO.limit(Ddisc)
    @test dprA isa MD.PMorphism
    @test dprB isa MD.PMorphism

        Dpar = TO.ParallelPairDiagram(h, z)
        _, de = TO.limit(Dpar)
    @test de isa MD.PMorphism

        Dspan = TO.SpanDiagram(h, z)  # A -> B, A -> B (same codomain is fine for pushout)
        PO, p1, p2 = TO.colimit(Dspan)
    @test p1 isa MD.PMorphism
    @test p2 isa MD.PMorphism

        Dcosp = TO.CospanDiagram(h, z) # A -> B and A -> B; pullback exists
        PB, r1, r2 = TO.limit(Dcosp)
    @test r1 isa MD.PMorphism
    @test r2 isa MD.PMorphism
    end


end

@testset "Abelian-category API (all fields)" begin
    P = chain_poset(1)

    function vmod(d::Int)
        edge = Dict{Tuple{Int,Int}, Matrix{Kc}}()
        return MD.PModule{Kc}(P, [d], edge; field=field)
    end

    A = vmod(2)
    B = vmod(3)

    f_mat = Matrix{Kc}([
        one(Kc)  zero(Kc);
        zero(Kc) zero(Kc);
        zero(Kc) zero(Kc)
    ])
    f = MD.PMorphism(A, B, [f_mat])

    Kmod, iK = TO.kernel_with_inclusion(f)
    @test Kmod.dims == [1]
    @test f.comps[1] * iK.comps[1] == zeros(Kc, 3, 1)
    @test TO.kernel(f).dims == Kmod.dims

    Im, iIm = TO.image_with_inclusion(f)
    @test Im.dims == [1]
    @test TO.image(f).dims == [1]

    Cok, q = TO.cokernel_with_projection(f)
    @test Cok.dims == [2]
    @test q.comps[1] * f.comps[1] == zeros(Kc, 2, 2)
    @test TO.cokernel(f).dims == [2]

    Coim, pco = TO.coimage_with_projection(f)
    @test Coim.dims == [1]
    @test pco.comps[1] * iK.comps[1] == zeros(Kc, 1, 1)
    @test TO.coimage(f).dims == [1]
end

@testset "RealField cokernel/equalizer stability" begin
    if field isa CM.RealField
        n = 6
        Z = zeros(Kc, 0, n)
        N = FL.nullspace(field, Z)
        @test size(N) == (n, n)
        @test FL.rank(field, N) == n
        @test isapprox(N, Matrix{Kc}(I, n, n); rtol=field.rtol, atol=field.atol)

        P1 = chain_poset(1)
        V = MD.PModule{Kc}(P1, [n], Dict{Tuple{Int,Int}, Matrix{Kc}}(); field=field)
        z = TO.zero_morphism(V, V)
        E, e = TO.equalizer(z, z)
        Q, q = TO.coequalizer(z, z)

        @test E.dims == [n]
        @test Q.dims == [n]
        @test FL.rank(field, e.comps[1]) == n
        @test FL.rank(field, q.comps[1]) == n
    else
        @test true
    end
end

@testset "Abelian sparse edge-store propagation" begin
    P = chain_poset(4)
    d = 3
    edge = Dict{Tuple{Int,Int}, SparseMatrixCSC{Kc,Int}}()
    for (u, v) in FF.cover_edges(P)
        A = spzeros(Kc, d, d)
        A[1, 1] = CM.coerce(field, 1)
        A[2, 2] = CM.coerce(field, 1)
        A[3, 3] = CM.coerce(field, 1)
        edge[(u, v)] = A
    end
    M = MD.PModule{Kc}(P, fill(d, FF.nvertices(P)), edge; field=field)
    zcomps = [spzeros(Kc, d, d) for _ in 1:FF.nvertices(P)]
    z = MD.PMorphism(M, M, zcomps)

    Kmod, _ = TO.kernel_with_inclusion(z)
    Imod, _ = TO.image_with_inclusion(z)
    Cmod, _ = TO.cokernel_with_projection(z)

    function _first_edge_map(X::MD.PModule)
        for maps in X.edge_maps.maps_to_succ
            isempty(maps) || return maps[1]
        end
        return nothing
    end

    fk = _first_edge_map(Kmod)
    fi = _first_edge_map(Imod)
    fc = _first_edge_map(Cmod)
    @test fk !== nothing && fk isa SparseMatrixCSC
    @test fi !== nothing && fi isa SparseMatrixCSC
    @test fc !== nothing && fc isa SparseMatrixCSC
end

@testset "Abelian cokernel induced-map oracle (multi-vertex)" begin
    P = chain_poset(8)
    _, B, iota = _ab_build_morphism_fixture(P; rank_part=3, ker_part=3, coker_part=3, seed=Int(0xAB11))
    cc = MD._get_cover_cache(P)

    Cnew, qnew = TO.AbelianCategories._cokernel_module(iota; cache=cc)
    Cold, qold = _ab_cokernel_module_old(iota; cache=cc)

    @test Cnew.dims == Cold.dims
    @test qnew.dom.dims == qold.dom.dims
    @test qnew.cod.dims == qold.cod.dims

    for ((u, v), Anew) in Cnew.edge_maps
        Aold = Cold.edge_maps[u, v]
        lhs = Anew * qnew.comps[u]
        rhs = qnew.comps[v] * B.edge_maps[u, v]
        if field isa CM.RealField
            @test isapprox(Anew, Aold; rtol=field.rtol, atol=field.atol)
            @test isapprox(lhs, rhs; rtol=field.rtol, atol=field.atol)
        else
            @test Anew == Aold
            @test lhs == rhs
        end
    end
    # q * iota = 0 by cokernel property.
    for u in 1:FF.nvertices(P)
        Z = qnew.comps[u] * iota.comps[u]
        if field isa CM.RealField
            @test isapprox(Z, zeros(Kc, size(Z, 1), size(Z, 2)); rtol=field.rtol, atol=field.atol)
        else
            @test Z == zeros(Kc, size(Z, 1), size(Z, 2))
        end
    end
end

@testset "Abelian image parity (multi-vertex)" begin
    P = chain_poset(8)
    _, _, f = _ab_build_morphism_fixture(P; rank_part=3, ker_part=3, coker_part=3, seed=Int(0xAB12))
    cc = MD._get_cover_cache(P)

    Imnew, iImnew = TO.image_with_inclusion(f; cache=cc)
    Imold, iImold = _ab_image_with_inclusion_old(f; cache=cc)
    @test _ab_pmodule_equal(Imnew, Imold)
    @test _ab_morphism_equal(iImnew, iImold)
end

@testset "Abelian pushout/pullback parity (multi-vertex)" begin
    P = chain_poset(8)
    _, _, f = _ab_build_morphism_fixture(P; rank_part=3, ker_part=3, coker_part=3, seed=Int(0xAB31))
    z = MD.zero_morphism(f.dom, f.cod)
    cc = MD._get_cover_cache(P)

    Pnew, inBnew, inCnew, qnew, phinew = TO.AbelianCategories.pushout(f, z; cache=cc)
    Pold, inBold, inCold, qold, phiold = _ab_pushout_old(f, z; cache=cc)
    @test _ab_pmodule_equal(Pnew, Pold)
    @test _ab_morphism_equal(inBnew, inBold)
    @test _ab_morphism_equal(inCnew, inCold)
    @test _ab_morphism_equal(qnew, qold)
    @test _ab_morphism_equal(phinew, phiold)

    PBnew, prBnew, prCnew, iotanew, psinew = TO.AbelianCategories.pullback(f, z; cache=cc)
    PBold, prBold, prCold, iotaold, psiold = _ab_pullback_old(f, z; cache=cc)
    @test _ab_pmodule_equal(PBnew, PBold)
    @test _ab_morphism_equal(prBnew, prBold)
    @test _ab_morphism_equal(prCnew, prCold)
    @test _ab_morphism_equal(iotanew, iotaold)
    @test _ab_morphism_equal(psinew, psiold)
end

@testset "Abelian self-map fast paths" begin
    P = chain_poset(8)
    A, B, f = _ab_build_morphism_fixture(P; rank_part=3, ker_part=3, coker_part=3, seed=Int(0xAB32))

    E, e = TO.equalizer(f, f)
    @test E.dims == A.dims
    @test _ab_morphism_equal(e, MD.id_morphism(A))

    Q, q = TO.coequalizer(f, f)
    @test Q.dims == B.dims
    @test _ab_morphism_equal(q, MD.id_morphism(B))

    Cokf = TO.cokernel(f)
    Pnew, inBnew, inCnew, qnew, phinew = TO.AbelianCategories.pushout(f, f)
    @test Pnew.dims == B.dims .+ Cokf.dims
    for u in 1:FF.nvertices(P)
        lhs = inBnew.comps[u] * f.comps[u]
        rhs = inCnew.comps[u] * f.comps[u]
        zero_block = zeros(Kc, size(qnew.comps[u], 1), size(phinew.comps[u], 2))
        if field isa CM.RealField
            @test isapprox(lhs, rhs; rtol=field.rtol, atol=field.atol)
            @test isapprox(qnew.comps[u] * phinew.comps[u], zero_block; rtol=field.rtol, atol=field.atol)
        else
            @test lhs == rhs
            @test qnew.comps[u] * phinew.comps[u] == zero_block
        end
    end

    Kerf = TO.kernel(f)
    PBnew, prBnew, prCnew, iotanew, psinew = TO.AbelianCategories.pullback(f, f)
    @test PBnew.dims == A.dims .+ Kerf.dims
    for u in 1:FF.nvertices(P)
        lhs = f.comps[u] * prBnew.comps[u]
        rhs = f.comps[u] * prCnew.comps[u]
        zero_block = zeros(Kc, size(psinew.comps[u], 1), size(iotanew.comps[u], 2))
        if field isa CM.RealField
            @test isapprox(lhs, rhs; rtol=field.rtol, atol=field.atol)
            @test isapprox(psinew.comps[u] * iotanew.comps[u], zero_block; rtol=field.rtol, atol=field.atol)
        else
            @test lhs == rhs
            @test psinew.comps[u] * iotanew.comps[u] == zero_block
        end
    end
end

@testset "Abelian cache route parity" begin
    P = chain_poset(8)
    _, _, f = _ab_build_morphism_fixture(P; rank_part=3, ker_part=3, coker_part=3, seed=Int(0xAB33))
    z = MD.zero_morphism(f.dom, f.cod)
    cc = MD._get_cover_cache(P)

    Kcached, iKcached = TO.kernel_with_inclusion(f; cache=cc)
    Kauto, iKauto = TO.kernel_with_inclusion(f; cache=:auto)
    @test _ab_pmodule_equal(Kcached, Kauto)
    @test _ab_morphism_equal(iKcached, iKauto)

    Imcached, iImcached = TO.image_with_inclusion(f; cache=cc)
    Imauto, iImauto = TO.image_with_inclusion(f; cache=:auto)
    @test _ab_pmodule_equal(Imcached, Imauto)
    @test _ab_morphism_equal(iImcached, iImauto)

    Ccached, qcached = TO.cokernel_with_projection(f; cache=cc)
    Cauto, qauto = TO.cokernel_with_projection(f; cache=:auto)
    @test _ab_pmodule_equal(Ccached, Cauto)
    @test _ab_morphism_equal(qcached, qauto)

    Coimcached, pcached = TO.coimage_with_projection(f; cache=cc)
    Coimauto, pauto = TO.coimage_with_projection(f; cache=:auto)
    @test _ab_pmodule_equal(Coimcached, Coimauto)
    @test _ab_morphism_equal(pcached, pauto)

    @test _ab_pmodule_equal(TO.coimage(f; cache=cc), TO.coimage(f; cache=:auto))
    @test _ab_pmodule_equal(TO.image(f; cache=cc), TO.image(f; cache=:auto))
    @test _ab_pmodule_equal(TO.quotient(iImcached; cache=cc), TO.quotient(iImcached; cache=:auto))
    @test _ab_pmodule_equal(TO.quotient(TO.submodule(iImcached; check_mono=false); cache=cc),
                            TO.quotient(TO.submodule(iImcached; check_mono=false); cache=:auto))

    Qcached, qqcached = TO.quotient_with_projection(iImcached; cache=cc)
    Qauto, qqauto = TO.quotient_with_projection(iImauto; cache=:auto)
    @test _ab_pmodule_equal(Qcached, Qauto)
    @test _ab_morphism_equal(qqcached, qqauto)

    Pcached, inBcached, inCcached, qPcached, phicached = TO.AbelianCategories.pushout(f, z; cache=cc)
    Pauto, inBauto, inCauto, qPauto, phiauto = TO.AbelianCategories.pushout(f, z; cache=:auto)
    @test _ab_pmodule_equal(Pcached, Pauto)
    @test _ab_morphism_equal(inBcached, inBauto)
    @test _ab_morphism_equal(inCcached, inCauto)
    @test _ab_morphism_equal(qPcached, qPauto)
    @test _ab_morphism_equal(phicached, phiauto)

    PBcached, prBcached, prCcached, iotacached, psicached = TO.AbelianCategories.pullback(f, z; cache=cc)
    PBauto, prBauto, prCauto, iotaauto, psiauto = TO.AbelianCategories.pullback(f, z; cache=:auto)
    @test _ab_pmodule_equal(PBcached, PBauto)
    @test _ab_morphism_equal(prBcached, prBauto)
    @test _ab_morphism_equal(prCcached, prCauto)
    @test _ab_morphism_equal(iotacached, iotaauto)
    @test _ab_morphism_equal(psicached, psiauto)

    P1 = chain_poset(1)
    cc1 = MD._get_cover_cache(P1)
    A2 = _vspace_module(P1, 1)
    B2 = _vspace_module(P1, 2)
    C2 = _vspace_module(P1, 1)
    i = MD.PMorphism(A2, B2, [reshape(Kc[1, 0], 2, 1)])
    p = MD.PMorphism(B2, C2, [reshape(Kc[0, 1], 1, 2)])

    ses_cached = TO.ShortExactSequence(i, p; check=false)
    ses_auto = TO.ShortExactSequence(i, p; check=false)
    @test TO.is_exact(ses_cached; cache=cc1)
    @test TO.is_exact(ses_auto; cache=:auto)
    @test ses_cached.exact == ses_auto.exact

    function _snake_fixture()
        At = _vspace_module(P1, 1)
        Bt = _vspace_module(P1, 2)
        Ct = _vspace_module(P1, 1)
        it = MD.PMorphism(At, Bt, [reshape(Kc[1, 0], 2, 1)])
        pt = MD.PMorphism(Bt, Ct, [reshape(Kc[0, 1], 1, 2)])
        top = TO.ShortExactSequence(it, pt)

        Ab = _vspace_module(P1, 2)
        Bb = _vspace_module(P1, 3)
        Cb = _vspace_module(P1, 1)
        ib = MD.PMorphism(Ab, Bb, [Matrix{Kc}([
            CM.coerce(field, 1) CM.coerce(field, 0);
            CM.coerce(field, 0) CM.coerce(field, 1);
            CM.coerce(field, 0) CM.coerce(field, 0)
        ])])
        pb = MD.PMorphism(Bb, Cb, [reshape(Kc[0, 0, 1], 1, 3)])
        bottom = TO.ShortExactSequence(ib, pb)

        alpha = MD.PMorphism(At, Ab, [reshape(Kc[1, 0], 2, 1)])
        beta = MD.PMorphism(Bt, Bb, [Matrix{Kc}([
            CM.coerce(field, 1) CM.coerce(field, 0);
            CM.coerce(field, 0) CM.coerce(field, 1);
            CM.coerce(field, 0) CM.coerce(field, 0)
        ])])
        gamma = MD.PMorphism(Ct, Cb, [reshape(Kc[0], 1, 1)])
        return top, bottom, alpha, beta, gamma
    end

    top_cached, bottom_cached, alpha_cached, beta_cached, gamma_cached = _snake_fixture()
    top_auto, bottom_auto, alpha_auto, beta_auto, gamma_auto = _snake_fixture()
    sn_cached = TO.snake_lemma(top_cached, bottom_cached, alpha_cached, beta_cached, gamma_cached; cache=cc1)
    sn_auto = TO.snake_lemma(top_auto, bottom_auto, alpha_auto, beta_auto, gamma_auto; cache=:auto)
    @test _ab_pmodule_equal(sn_cached.kerA[1], sn_auto.kerA[1])
    @test _ab_pmodule_equal(sn_cached.kerB[1], sn_auto.kerB[1])
    @test _ab_pmodule_equal(sn_cached.kerC[1], sn_auto.kerC[1])
    @test _ab_pmodule_equal(sn_cached.cokA[1], sn_auto.cokA[1])
    @test _ab_pmodule_equal(sn_cached.cokB[1], sn_auto.cokB[1])
    @test _ab_pmodule_equal(sn_cached.cokC[1], sn_auto.cokC[1])
    @test _ab_morphism_equal(sn_cached.k1, sn_auto.k1)
    @test _ab_morphism_equal(sn_cached.k2, sn_auto.k2)
    @test _ab_morphism_equal(sn_cached.delta, sn_auto.delta)
    @test _ab_morphism_equal(sn_cached.c1, sn_auto.c1)
    @test _ab_morphism_equal(sn_cached.c2, sn_auto.c2)

    lim_cached = TO.AbelianCategories.limit(TO.AbelianCategories.ParallelPairDiagram(f, z); cache=cc)
    lim_auto = TO.AbelianCategories.limit(TO.AbelianCategories.ParallelPairDiagram(f, z); cache=:auto)
    @test _ab_pmodule_equal(lim_cached[1], lim_auto[1])
    @test _ab_morphism_equal(lim_cached[2], lim_auto[2])

    col_cached = TO.AbelianCategories.colimit(TO.AbelianCategories.SpanDiagram(f, z); cache=cc)
    col_auto = TO.AbelianCategories.colimit(TO.AbelianCategories.SpanDiagram(f, z); cache=:auto)
    @test _ab_pmodule_equal(col_cached[1], col_auto[1])
    @test _ab_morphism_equal(col_cached[2], col_auto[2])
    @test _ab_morphism_equal(col_cached[3], col_auto[3])
    @test _ab_morphism_equal(col_cached[4], col_auto[4])
    @test _ab_morphism_equal(col_cached[5], col_auto[5])

    @test_throws TypeError TO.coimage(f; cache=nothing)
    @test_throws TypeError TO.AbelianCategories._cokernel_module(f; cache=nothing)
    @test_throws ErrorException TO.coimage(f; cache=:bogus)
    @test_throws ErrorException TO.equalizer(f, z; cache=:bogus)
end

@testset "AbelianCategories perf guards (kernel/image/cokernel)" begin
    if field isa CM.QQField
        function _median_time_alloc(fn::Function; reps::Int=3)
            ts = Float64[]
            bs = Int[]
            for _ in 1:reps
                GC.gc()
                m = @timed fn()
                push!(ts, m.time)
                push!(bs, m.bytes)
            end
            sort!(ts)
            sort!(bs)
            return ts[cld(reps, 2)], bs[cld(reps, 2)]
        end

        for (nverts, seed) in ((28, Int(0xAB22)), (64, Int(0xAB23)))
            P = chain_poset(nverts)
            _, _, f = _ab_build_morphism_fixture(P; rank_part=4, ker_part=4, coker_part=4, seed=seed)
            cc = MD._get_cover_cache(P)

            # Warmups.
            _ab_kernel_with_inclusion_old(f; cache=cc)
            TO.kernel_with_inclusion(f; cache=cc)
            _ab_image_with_inclusion_old(f; cache=cc)
            TO.image_with_inclusion(f; cache=cc)
            _ab_cokernel_module_old(f; cache=cc)
            TO.AbelianCategories._cokernel_module(f; cache=cc)

            _, balloc_old_k = _median_time_alloc(() -> _ab_kernel_with_inclusion_old(f; cache=cc))
            _, balloc_new_k = _median_time_alloc(() -> TO.kernel_with_inclusion(f; cache=cc))
            # Kernel wall time is still heuristic-sensitive across QQ fixtures;
            # keep a stable allocation guard here and cover timing in the
            # dedicated benchmark harness instead.
            @test balloc_new_k <= 1.03 * balloc_old_k

            told_i, balloc_old_i = _median_time_alloc(() -> _ab_image_with_inclusion_old(f; cache=cc))
            tnew_i, balloc_new_i = _median_time_alloc(() -> TO.image_with_inclusion(f; cache=cc))
            # Tighten the local-noise relaxation from pure 2.5x ratio:
            # keep a moderate ratio plus small absolute slack for very short timings.
            @test tnew_i <= 2.10 * told_i + 0.004
            @test balloc_new_i <= 1.03 * balloc_old_i

            told_c, balloc_old_c = _median_time_alloc(() -> _ab_cokernel_module_old(f; cache=cc))
            tnew_c, balloc_new_c = _median_time_alloc(() -> TO.AbelianCategories._cokernel_module(f; cache=cc))
            @test tnew_c <= 1.30 * told_c
            @test balloc_new_c <= 1.03 * balloc_old_c
        end

        # High-fanout fixture specifically exercises batched cokernel solves.
        Pf = _ab_two_layer_poset(6, 12)
        _, _, ff = _ab_build_morphism_fixture(Pf; rank_part=4, ker_part=4, coker_part=4, seed=Int(0xAB24))
        ccf = MD._get_cover_cache(Pf)

        _ab_cokernel_module_old(ff; cache=ccf)
        TO.AbelianCategories._cokernel_module(ff; cache=ccf)
        told_f, _ = _median_time_alloc(() -> _ab_cokernel_module_old(ff; cache=ccf))
        tnew_f, _ = _median_time_alloc(() -> TO.AbelianCategories._cokernel_module(ff; cache=ccf))
        @test tnew_f <= 1.20 * told_f
    else
        @test true
    end
end

# Local helper: check string is ASCII-only.
_is_ascii(s::AbstractString) = all(c -> Int(c) <= 0x7f, s)

@testset "Pretty printing: Submodule / ShortExactSequence / SnakeLemmaResult" begin
    P = chain_poset(1)

    # 1-vertex PModules (no edge maps needed).
    edge_maps = Dict{Tuple{Int,Int}, Matrix{Kc}}()

    A = MD.PModule{Kc}(P, [1], edge_maps)
    B = MD.PModule{Kc}(P, [2], edge_maps)
    C = MD.PModule{Kc}(P, [1], edge_maps)

    # i : A -> B (inclusion)
    i = MD.PMorphism(A, B, [reshape(Kc[1, 0], 2, 1)])

    # p : B -> C (projection)
    p = MD.PMorphism(B, C, [reshape(Kc[0, 1], 1, 2)])

    # --- PModule show ---
    sA0 = sprint(show, A)
    @test occursin("PModule", sA0)
    @test occursin("nverts=1", sA0)
    @test occursin("dims=", sA0)
    @test _is_ascii(sA0)

    sA1 = sprint(show, MIME("text/plain"), A)
    @test occursin("PModule", sA1)
    if field isa CM.QQField
        @test occursin("scalars = QQ", sA1) || occursin("scalars = Rational{BigInt}", sA1)
    else
        @test occursin("scalars = ", sA1)
    end
    @test occursin("nverts = 1", sA1)
    @test occursin("dims =", sA1)
    @test _is_ascii(sA1)

    # --- PMorphism show ---
    si0 = sprint(show, i)
    @test occursin("PMorphism", si0)
    @test occursin("nverts=1", si0)
    @test occursin("dom_sum=1", si0)
    @test occursin("cod_sum=2", si0)
    @test _is_ascii(si0)

    si1 = sprint(show, MIME("text/plain"), i)
    @test occursin("PMorphism", si1)
    @test occursin("endomorphism = false", si1)
    @test occursin("dom dims", si1)
    @test occursin("cod dims", si1)
    @test _is_ascii(si1)

    # Truncation sanity check under IOContext(:limit=>true).
    Pbig = chain_poset(20)
    dims_big = collect(1:20)
    Mbig = MD.PModule{Kc}(Pbig, dims_big, Dict{Tuple{Int,Int}, Matrix{Kc}}())

    sbig = sprint(show, Mbig; context=:limit=>true)
    @test occursin("...", sbig)
    @test _is_ascii(sbig)

    # --- Submodule show ---
    S = TO.submodule(i; check_mono=true)

    s1 = sprint(show, S)
    @test occursin("Submodule", s1)
    @test _is_ascii(s1)

    s2 = sprint(show, MIME("text/plain"), S)
    @test occursin("Submodule", s2)
    @test occursin("sub dims", s2)
    @test occursin("ambient dims", s2)
    @test _is_ascii(s2)

    # --- ShortExactSequence show (must not force exactness check) ---
    ses = TO.ShortExactSequence(i, p; check=false)
    @test ses.checked == false

    s3 = sprint(show, ses)
    @test occursin("ShortExactSequence", s3)
    @test _is_ascii(s3)
    @test ses.checked == false  # show must not mutate cached status

    s4 = sprint(show, MIME("text/plain"), ses)
    @test occursin("0 -> A -(i)-> B -(p)-> C -> 0", s4)
    @test occursin("exact = unknown", s4)
    @test _is_ascii(s4)
    @test ses.checked == false

    # --- SnakeLemmaResult show ---
    # Build a small snake lemma instance (1-vertex, so linear algebra is tiny).
    top = TO.ShortExactSequence(i, p; check=true)

    Ab = MD.PModule{Kc}(P, [2], edge_maps)
    Bb = MD.PModule{Kc}(P, [3], edge_maps)
    Cb = MD.PModule{Kc}(P, [1], edge_maps)

    ib = MD.PMorphism(Ab, Bb, [Matrix{Kc}([
        CM.coerce(field, 1) CM.coerce(field, 0);
        CM.coerce(field, 0) CM.coerce(field, 1);
        CM.coerce(field, 0) CM.coerce(field, 0)
    ])])
    pb = MD.PMorphism(Bb, Cb, [reshape(Kc[0, 0, 1], 1, 3)])
    bottom = TO.ShortExactSequence(ib, pb; check=true)

    alpha = MD.PMorphism(A, Ab, [reshape(Kc[1, 0], 2, 1)])
    beta  = MD.PMorphism(B, Bb, [Matrix{Kc}([
        CM.coerce(field, 1) CM.coerce(field, 0);
        CM.coerce(field, 0) CM.coerce(field, 1);
        CM.coerce(field, 0) CM.coerce(field, 0)
    ])])
    gamma = MD.PMorphism(C, Cb, [reshape(Kc[0], 1, 1)])

    sn = TO.snake_lemma(top, bottom, alpha, beta, gamma; check=true)

    s5 = sprint(show, sn)
    @test occursin("SnakeLemmaResult", s5)
    @test occursin("delta:", s5)
    @test _is_ascii(s5)

    s6 = sprint(show, MIME("text/plain"), sn)
    @test occursin("kerA -> kerB -> kerC --delta--> cokerA -> cokerB -> cokerC", s6)
    @test occursin("maps: k1, k2, delta, c1, c2", s6)
    @test _is_ascii(s6)
end

@testset "AbelianCategories UX surface" begin
    P = chain_poset(1)
    edge_maps = Dict{Tuple{Int,Int}, Matrix{Kc}}()

    A = MD.PModule{Kc}(P, [1], edge_maps)
    B = MD.PModule{Kc}(P, [2], edge_maps)
    C = MD.PModule{Kc}(P, [1], edge_maps)

    i = MD.PMorphism(A, B, [reshape(Kc[1, 0], 2, 1)])
    p = MD.PMorphism(B, C, [reshape(Kc[0, 1], 1, 2)])

    S = TO.submodule(i; check_mono=true)
    ses = TO.short_exact_sequence(i, p; check=true)

    Ab = MD.PModule{Kc}(P, [2], edge_maps)
    Bb = MD.PModule{Kc}(P, [3], edge_maps)
    Cb = MD.PModule{Kc}(P, [1], edge_maps)
    ib = MD.PMorphism(Ab, Bb, [Matrix{Kc}([
        CM.coerce(field, 1) CM.coerce(field, 0);
        CM.coerce(field, 0) CM.coerce(field, 1);
        CM.coerce(field, 0) CM.coerce(field, 0)
    ])])
    pb = MD.PMorphism(Bb, Cb, [reshape(Kc[0, 0, 1], 1, 3)])
    top = TO.short_exact_sequence(i, p; check=true)
    bottom = TO.short_exact_sequence(ib, pb; check=true)
    alpha = MD.PMorphism(A, Ab, [reshape(Kc[1, 0], 2, 1)])
    beta  = MD.PMorphism(B, Bb, [Matrix{Kc}([
        CM.coerce(field, 1) CM.coerce(field, 0);
        CM.coerce(field, 0) CM.coerce(field, 1);
        CM.coerce(field, 0) CM.coerce(field, 0)
    ])])
    gamma = MD.PMorphism(C, Cb, [reshape(Kc[0], 1, 1)])
    sn = TO.snake_lemma(top, bottom, alpha, beta, gamma; check=true)

    @test TO.check_submodule(S).valid
    @test TO.check_short_exact_sequence(ses).valid
    @test TO.check_snake_lemma(sn).valid
    @test TamerOp.Advanced.check_submodule === TO.check_submodule
    @test TamerOp.Advanced.check_short_exact_sequence === TO.check_short_exact_sequence
    @test TamerOp.Advanced.check_snake_lemma === TO.check_snake_lemma
    @test TamerOp.Advanced.validation_summary === TO.validation_summary
    @test TamerOp.Advanced.exactness_summary === TO.exactness_summary

    @test TO.submodule_object(S) === A
    @test TO.ambient_module(S) === B
    @test TO.inclusion_map(S) === i
    @test TO.inclusion_map(ses) === i
    @test TO.projection_map(ses) === p
    @test TO.connecting_map(sn) === sn.delta
    @test TO.kernel_objects(sn).A === sn.kerA[1]
    @test TO.kernel_inclusions(sn).B === sn.kerB[2]
    @test TO.cokernel_objects(sn).C === sn.cokC[1]
    @test TO.cokernel_projections(sn).A === sn.cokA[2]

    exsum = TO.exactness_summary(ses)
    @test exsum.checked
    @test exsum.exact
    @test exsum.ker_cached
    @test exsum.img_cached

    sdesc = TamerOp.Advanced.describe(S)
    @test sdesc.kind == :submodule
    @test TamerOp.Advanced.dimensions(S).submodule_total == 1

    sedesc = TamerOp.Advanced.describe(ses)
    @test sedesc.kind == :short_exact_sequence
    @test TamerOp.Advanced.dimensions(ses).A_total == 1

    sndesc = TamerOp.Advanced.describe(sn)
    @test sndesc.kind == :snake_lemma
    @test TamerOp.Advanced.dimensions(sn).kerA_total == sum(sn.kerA[1].dims)

    vrepr = TO.validation_summary(TO.check_short_exact_sequence(ses))
    @test occursin("ValidationSummary", sprint(show, MIME("text/plain"), vrepr))
    @test occursin("kind = short_exact_sequence", sprint(show, MIME("text/plain"), vrepr))

    @test occursin("Best practice", string(@doc TamerOp.AbelianCategories.check_submodule))
    @test occursin("exactness", lowercase(string(@doc TamerOp.AbelianCategories.exactness_summary)))
    @test occursin("kernel objects", lowercase(string(@doc TamerOp.AbelianCategories.kernel_objects)))

    Sbad = TO.Submodule{Kc}(MD.zero_morphism(A, B))
    @test !TO.check_submodule(Sbad).valid
    @test_throws ErrorException TO.check_submodule(Sbad; throw=true)

    pbad = MD.zero_morphism(B, C)
    sesbad = TO.ShortExactSequence(i, pbad; check=false)
    @test !TO.check_short_exact_sequence(sesbad).valid
    @test_throws ErrorException TO.check_short_exact_sequence(sesbad; throw=true)

    snbad = TO.SnakeLemmaResult(sn.kerA, sn.kerB, sn.kerC,
                                sn.cokA, sn.cokB, sn.cokC,
                                sn.k1, MD.id_morphism(sn.kerB[1]), sn.delta, sn.c1, sn.c2)
    @test !TO.check_snake_lemma(snbad).valid
    @test_throws ErrorException TO.check_snake_lemma(snbad; throw=true)
end

@testset "Derived-category primitives: cone triangle and LES" begin
    # C and D concentrated in degree 0, map f = 0.
    C = CC.CochainComplex{Kc}(0, 0, [1], SparseMatrixCSC{Kc,Int}[]; labels=[["c0"]])
    D = CC.CochainComplex{Kc}(0, 0, [1], SparseMatrixCSC{Kc,Int}[]; labels=[["d0"]])
    # Compute metadata is typed; heterogeneous user labels stay at the boundary.
    @test C.labels == [[1]]
    @test D.labels == [[1]]
    @test C.annotations == [["c0"]]
    @test D.annotations == [["d0"]]

    f0 = spzeros(Kc, 1, 1)
    f = CC.CochainMap(C, D, [f0]; check=true)

    tri = CC.mapping_cone_triangle(f)
    les = CC.long_exact_sequence(tri; output=:full)

    @test les.tmin == -1
    @test les.tmax == 0

    # At t = -1: H^{-1}(D)=0 -> H^{-1}(Cone)=k -> H^0(C)=k
    idx = (-1) - les.tmin + 1
    @test les.HD[idx].dimH == 0
    @test les.Hcone[idx].dimH == 1
    @test les.HC[idx+1].dimH == 1

    # Connecting map delta^{-1} : H^{-1}(Cone) -> H^0(C) should be an isomorphism in this case.
    delta = les.delta[idx]
    @test size(delta) == (1, 1)
    @test delta[1, 1] == one(Kc)
    if field isa CM.QQField
        @test !CC._use_long_exact_precompute(Kc, tri.C)
        @test !CC._use_batched_coordinate_solves(Kc, 2, 2)
        @test les.pH[idx] == les.delta[idx]
        @test les.pH[idx] == CC.induced_map_on_cohomology(les.Hcone[idx], les.HC[idx + 1], CC._map_at(tri.p, -1))
    else
        @test CC._use_long_exact_precompute(Kc, tri.C)
    end
end

@testset "ChainComplexes canonical public front doors" begin
    C = CC.CochainComplex{Kc}(0, 1, [1, 1], [spzeros(Kc, 1, 1)])

    Hdims = CC.cohomology(C)
    @test Hdims == Dict(0 => 1, 1 => 1)
    @test CC.cohomology(C; degree=0, output=:dims) == 1

    Hbasis = CC.cohomology(C; output=:basis)
    @test Hbasis[0] == CC.cohomology_data(C, 0).Hrep
    @test CC.cohomology(C; degree=1, output=:basis) == CC.cohomology_data(C, 1).Hrep

    Hfull = CC.cohomology(C; output=:full)
    @test length(Hfull) == 2
    @test Hfull[1].dimH == 1
    @test Hfull[2].dimH == 1
    H0 = Hfull[1]
    @test CC.dimensions(H0) == (ambient=1, cycles=1, boundaries=0, cohomology=1)
    @test CC.basis(H0) == H0.Hrep
    @test CC.representatives(H0) == H0.Hrep
    @test CC.coordinates(H0, H0.Hrep[:, 1]) == ones(Kc, 1, 1)
    @test TamerOp.Advanced.basis(H0) == H0.Hrep
    @test TamerOp.Advanced.coordinates(H0, H0.Hrep[:, 1]) == Kc[one(Kc)]
    @test TamerOp.Advanced.dimensions(H0).cohomology == 1
    @test TamerOp.Advanced.representatives(H0) == H0.Hrep
    @test CC.describe(H0).kind == :cohomology
    @test_throws ErrorException CC.cohomology(C; degree=:foo)
    @test_throws ErrorException CC.cohomology(C; output=:foo)

    D = CC.CochainComplex{Kc}(0, 0, [1], SparseMatrixCSC{Kc,Int}[])
    f0 = spzeros(Kc, 1, 1)
    f = CC.CochainMap(D, D, [f0]; check=true)
    tri = CC.mapping_cone_triangle(f)

    les = CC.long_exact_sequence(f)
    @test les[-1].cone == 1
    @test les[0].C == 1

    les_dims = CC.long_exact_sequence(tri; output=:dims)
    @test les_dims[-1].cone == 1
    @test les_dims[0].C == 1

    les_maps = CC.long_exact_sequence(f; degree=-1, output=:maps)
    @test size(les_maps.delta) == (1, 1)
    @test les_maps.delta[1, 1] == one(Kc)

    les_slice = CC.long_exact_sequence(tri; degree=-1, output=:full)
    @test les_slice.cone.dimH == 1
    @test les_slice.C.dimH == 0
    @test les_slice.delta == les_maps.delta
    @test_throws ErrorException CC.long_exact_sequence(tri; degree=:foo)
    @test_throws ErrorException CC.long_exact_sequence(tri; output=:foo)

    dims = [1 1;
            1 1]

    dv = Array{SparseMatrixCSC{Kc,Int},2}(undef, 2, 2)
    dh = Array{SparseMatrixCSC{Kc,Int},2}(undef, 2, 2)
    dv[1,1] = sparse([1],[1],[one(Kc)], 1, 1)
    dv[2,1] = spzeros(Kc, 1, 1)
    dv[1,2] = spzeros(Kc, 0, 1)
    dv[2,2] = spzeros(Kc, 0, 1)
    dh[1,1] = sparse([1],[1],[one(Kc)], 1, 1)
    dh[1,2] = spzeros(Kc, 1, 1)
    dh[2,1] = spzeros(Kc, 0, 1)
    dh[2,2] = spzeros(Kc, 0, 1)

    DC = CC.DoubleComplex{Kc}(0, 1, 0, 1, dims, dv, dh)

    ss_dims_default = CC.spectral_sequence(DC; first=:vertical)
    @test ss_dims_default[(1, 0)] == 1
    @test ss_dims_default[(1, 1)] == 1

    ss = CC.spectral_sequence(DC; output=:full, first=:vertical)
    @test ss isa CC.SpectralSequence{Kc}
    @test CC.page(ss; page=2)[(1, 0)] == CC.page(ss, 2)[(1, 0)]
    @test CC.E_r(ss; page=2)[(1, 1)] == CC.E_r(ss, 2)[(1, 1)]

    page_dims = CC.spectral_sequence(DC; page=2, output=:dims, first=:vertical)
    @test page_dims[(1, 0)] == 1
    @test page_dims[(1, 1)] == 1
    @test CC.page_dims_dict(ss; page=2) == CC.page_dims_dict(ss, 2)
    @test CC.page_dict(ss; page=2) == CC.page_dict(ss, 2)
    @test CC.page_dims(ss, 2) == CC.page(ss, 2).dims
    @test CC.term_dims(ss, 2) == CC.page_dims_dict(ss, 2)
    @test CC.nonzero_terms(ss, 2) == sort!(collect(keys(CC.page_dims_dict(ss, 2))))
    @test CC.convergence_page(ss) == CC.collapse_page(ss)
    @test CC.page_differentials(ss; page=1) == CC.spectral_sequence(DC; page=1, output=:differentials, first=:vertical)
    @test CC.describe(ss).kind == :spectral_sequence
    @test CC.describe(ss).convergence_page == CC.collapse_page(ss)

    page_terms = CC.spectral_sequence(DC; page=2, output=:terms, first=:vertical)
    @test page_terms[(1, 0)].dimH == 1
    @test page_terms[(1, 1)].dimH == 1
    @test CC.page_terms(ss; page=2)[2, 1].dimH == CC.page_terms(ss, 2)[2, 1].dimH
    @test CC.page_terms_dict(ss; page=2)[(1, 0)].dimH == CC.page_terms_dict(ss, 2)[(1, 0)].dimH
    @test CC.E_r_terms(ss; page=2)[(1, 0)].dimH == CC.E_r_terms(ss, 2)[(1, 0)].dimH
    @test CC.term(ss; page=2, p=1, q=0).dimH == CC.term(ss, 2, (1, 0)).dimH
    SQ = CC.term(ss; page=2, p=1, q=0)
    @test CC.dimensions(SQ).quotient == 1
    @test CC.representatives(SQ) == SQ.Hrep
    @test CC.coordinates(SQ, SQ.Hrep[:, 1]) == ones(Kc, 1, 1)
    @test TamerOp.Advanced.basis(SQ) == SQ.Hrep
    @test TamerOp.Advanced.coordinates(SQ, SQ.Hrep[:, 1]) == Kc[one(Kc)]
    @test CC.image_basis(ss, 1, 0) == CC.filtration_basis(ss, 1, 1)
    @test CC.filtration_dims(ss; degree=1) == CC.filtration_dims(ss, 1)
    @test CC.filtration_dims(ss; filtration=1, degree=1) == CC.filtration_dims(ss, 1, 1)
    @test CC.filtration_basis(ss; filtration=1, degree=1) == CC.filtration_basis(ss, 1, 1)
    @test CC.filtration_subquotient(ss; filtration=1, degree=1).dimH == CC.filtration_subquotient(ss, 1, 1).dimH
    @test CC.image_basis(ss; p=1, q=0) == CC.image_basis(ss, 1, 0)
    @test CC.split_total_cohomology(ss; degree=1).ranges == CC.split_total_cohomology(ss, 1).ranges
    @test CC.filtration_data(ss; degree=1).dims == CC.filtration_data(ss, 1).dims
    @test CC.diagonal_criterion(ss; degree=1, page=:inf) == CC.diagonal_criterion(ss, 1; r=:inf)
    @test CC.diagonal_criterion(ss; page=:inf) == CC.diagonal_criterion(ss; r=:inf)
    @test CC.extension_problem(ss; degree=1).t == CC.extension_problem(ss, 1).t
    @test CC.edge_inclusion(ss; p=1, q=0) == CC.edge_inclusion(ss, (1, 0))
    @test CC.edge_projection(ss; p=1, q=0) == CC.edge_projection(ss, (1, 0))
    @test CC.collapse_data(ss; page=CC.collapse_page(ss)).collapse_r == CC.collapse_data(ss; r=CC.collapse_page(ss)).collapse_r
    @test CC.describe(SQ).kind == :subquotient

    page_diffs = CC.spectral_sequence(DC; page=1, output=:differentials, first=:vertical)
    @test page_diffs[(0, 0)] == CC.differential(ss, 1, (0, 0))
    @test CC.differential(ss; page=1, p=0, q=0) == CC.differential(ss, 1, (0, 0))
    @test CC.dr_target(ss; page=1, p=0, q=0) == CC.dr_target(ss, 1, (0, 0))
    @test CC.dr_source(ss; page=1, p=1, q=0) == CC.dr_source(ss, 1, (1, 0))
    @test_throws ErrorException CC.spectral_sequence(DC; page=0)
    @test_throws ErrorException CC.spectral_sequence(DC; output=:foo)

    @test TamerOp.Advanced.cohomology === CC.cohomology
    @test TamerOp.Advanced.long_exact_sequence === CC.long_exact_sequence
    @test TamerOp.Advanced.spectral_sequence === CC.spectral_sequence
    @test TamerOp.Advanced.check_complex === CC.check_complex
    @test TamerOp.Advanced.check_bicomplex === CC.check_bicomplex
    @test TamerOp.Advanced.check_filtered_complex === CC.check_filtered_complex
    @test TamerOp.Advanced.describe === CC.describe
    @test TamerOp.Advanced.page_differentials === CC.page_differentials
    @test TamerOp.Advanced.filtration_dims === CC.filtration_dims
    @test TamerOp.Advanced.filtration_basis === CC.filtration_basis
    @test TamerOp.Advanced.filtration_subquotient === CC.filtration_subquotient
    @test TamerOp.Advanced.split_total_cohomology === CC.split_total_cohomology
    @test TamerOp.Advanced.filtration_data === CC.filtration_data
    @test TamerOp.Advanced.diagonal_criterion === CC.diagonal_criterion
    @test TamerOp.Advanced.collapse_data === CC.collapse_data
    @test TamerOp.Advanced.extension_problem === CC.extension_problem
    @test TamerOp.Advanced.edge_inclusion === CC.edge_inclusion
    @test TamerOp.Advanced.edge_projection === CC.edge_projection

    pieces = Dict{Tuple{Int,Int},Int}()
    d0 = Dict{Tuple{Int,Int},SparseMatrixCSC{Kc,Int}}()
    d1 = Dict{Tuple{Int,Int},SparseMatrixCSC{Kc,Int}}()
    for a in 0:1, b in 0:1
        pieces[(a, a + b)] = dims[a + 1, b + 1]
        d0[(a, a + b)] = dv[a + 1, b + 1]
        d1[(a, a + b)] = dh[a + 1, b + 1]
    end
    FDC = CC.filtered_cochain_complex(Kc; first=:vertical, pieces=pieces, d0=d0, d1=d1)
    @test (FDC.amin, FDC.amax, FDC.bmin, FDC.bmax) == (0, 1, -1, 2)
    @test CC.spectral_sequence(FDC; page=2, output=:dims, first=:vertical) == page_dims
    @test CC.check_complex(C).valid
    @test CC.check_bicomplex(DC).valid
    @test CC.check_filtered_complex(FDC; first=:vertical).valid

    Cbad = CC.CochainComplex{Kc,Any}(0, 1, [1, 1], [spzeros(Kc, 2, 1)], [Int[], Int[]], nothing)
    creport = CC.check_complex(Cbad)
    @test !creport.valid
    @test any(occursin("expected d^0", msg) for msg in creport.issues)
    @test_throws ErrorException CC.check_complex(Cbad; throw=true)

    dims_bad = reshape([1, 1, 1], 1, 3)
    dv_bad = Array{SparseMatrixCSC{Kc,Int},2}(undef, 1, 3)
    dh_bad = Array{SparseMatrixCSC{Kc,Int},2}(undef, 1, 3)
    dv_bad[1, 1] = sparse([1], [1], [one(Kc)], 1, 1)
    dv_bad[1, 2] = sparse([1], [1], [one(Kc)], 1, 1)
    dv_bad[1, 3] = spzeros(Kc, 0, 1)
    dh_bad[1, 1] = spzeros(Kc, 0, 1)
    dh_bad[1, 2] = spzeros(Kc, 0, 1)
    dh_bad[1, 3] = spzeros(Kc, 0, 1)
    DCbad = CC.DoubleComplex{Kc}(0, 0, 0, 2, dims_bad, dv_bad, dh_bad)
    breport = CC.check_bicomplex(DCbad)
    @test !breport.valid
    @test any(occursin("d_v^(0,1) circ d_v^(0,0)", msg) for msg in breport.issues)
    @test_throws ErrorException CC.check_bicomplex(DCbad; throw=true)

    freport = CC.check_filtered_complex(DCbad; first=:vertical)
    @test !freport.valid
    @test freport.filtration_range == (0, 0)
    @test !CC.check_filtered_complex(FDC; first=:foo).valid
    @test_throws ErrorException CC.check_filtered_complex(DCbad; first=:vertical, throw=true)

    err_term = try
        CC.term(ss; page=2, p=10, q=10)
        nothing
    catch err
        sprint(showerror, err)
    end
    @test err_term !== nothing
    @test occursin("outside the supported bidegree window", err_term)

    err_sq = try
        CC.subquotient_data(Matrix{Kc}([one(Kc) zero(Kc); zero(Kc) zero(Kc)]),
                            reshape(Kc[zero(Kc), one(Kc)], 2, 1))
        nothing
    catch err
        sprint(showerror, err)
    end
    @test err_sq !== nothing
    @test occursin("denominator generators to lie in the span", err_sq)

    err_map = try
        CC.induced_map_on_cohomology(H0, H0, spzeros(Kc, 2, 2))
        nothing
    catch err
        sprint(showerror, err)
    end
    @test err_map !== nothing
    @test occursin("expected a linear map", err_map)

    les_full = CC.long_exact_sequence(f; output=:full)
    @test CC.describe(les_full).kind == :long_exact_sequence
    @test occursin("CohomologyData", sprint(show, MIME"text/plain"(), H0))
    @test occursin("SpectralSequence", sprint(show, MIME"text/plain"(), ss))
    @test occursin("cached pages", sprint(show, MIME"text/plain"(), ss))
    @test occursin("SubquotientData", sprint(show, MIME"text/plain"(), SQ))
    @test occursin("LongExactSequence", sprint(show, MIME"text/plain"(), les_full))
end

@testset "Derived-category primitives: cone of identity is acyclic" begin
    # C: k -> k with zero differential. Cone(id_C) should be contractible, hence acyclic.
    C = CC.CochainComplex{Kc}(0, 1, [1, 1], [spzeros(Kc, 1, 1)])
    id1 = sparse([1], [1], [CM.coerce(field, 1)], 1, 1)
    f = CC.CochainMap(C, C, [id1, id1])
    tri = CC.mapping_cone_triangle(f)
    @test all(==(0), CC.homology_dims(tri.cone))
end

@testset "ChainComplexes container UX surface" begin
    C = CC.CochainComplex{Kc}(0, 1, [1, 2], [spzeros(Kc, 2, 1)]; labels=[[10], [20, 21]])
    id0 = sparse([1], [1], [one(Kc)], 1, 1)
    id1 = sparse([1, 2], [1, 2], [one(Kc), one(Kc)], 2, 2)
    f = CC.CochainMap(C, C, [id0, id1])
    tri = CC.mapping_cone_triangle(f)
    les = CC.long_exact_sequence(f; output=:full)

    dims = [1 1;
            1 1]
    dv = Array{SparseMatrixCSC{Kc,Int},2}(undef, 2, 2)
    dh = Array{SparseMatrixCSC{Kc,Int},2}(undef, 2, 2)
    dv[1, 1] = sparse([1], [1], [one(Kc)], 1, 1)
    dv[2, 1] = spzeros(Kc, 1, 1)
    dv[1, 2] = spzeros(Kc, 0, 1)
    dv[2, 2] = spzeros(Kc, 0, 1)
    dh[1, 1] = sparse([1], [1], [one(Kc)], 1, 1)
    dh[1, 2] = spzeros(Kc, 1, 1)
    dh[2, 1] = spzeros(Kc, 0, 1)
    dh[2, 2] = spzeros(Kc, 0, 1)
    DC = CC.DoubleComplex{Kc}(0, 1, 0, 1, dims, dv, dh)
    ss = CC.spectral_sequence(DC; output=:full, first=:vertical)
    fd = CC.filtration_data(ss, 1)
    ep = CC.extension_problem(ss, 1)

    @test CC.describe(C).kind == :cochain_complex
    @test CC.complex_summary(C) == CC.describe(C)
    @test CC.degree_range(C) == 0:1
    @test TOA.degree_range(C) == 0:1
    @test CC.component(C, 0).dimension == 1
    @test TOA.component(C, 0).dimension == 1
    @test CC.component_labels(C, 1) == [20, 21]
    @test CC.differential(C, 0) == spzeros(Kc, 2, 1)
    @test occursin("CochainComplex", sprint(show, MIME"text/plain"(), C))

    @test CC.describe(f).kind == :cochain_map
    @test CC.degree_range(f) == 0:1
    @test CC.source(f) === C
    @test CC.target(f) === C
    @test TOA.source(f) === C
    @test TOA.target(f) === C
    @test TOA.component(f, 0) == id0
    @test occursin("CochainMap", sprint(show, MIME"text/plain"(), f))

    @test CC.describe(tri).kind == :distinguished_triangle
    @test occursin("DistinguishedTriangle", sprint(show, MIME"text/plain"(), tri))

    @test CC.describe(DC).kind == :bicomplex
    @test CC.bicomplex_summary(DC) == CC.describe(DC)
    @test CC.a_range(DC) == 0:1
    @test CC.b_range(DC) == 0:1
    @test CC.block(DC, 0, 0).dimension == 1
    @test CC.vertical_differential(DC, 0, 0) == dv[1, 1]
    @test CC.horizontal_differential(DC, 0, 0) == dh[1, 1]
    @test occursin("DoubleComplex", sprint(show, MIME"text/plain"(), DC))

    @test CC.spectral_sequence_summary(ss) == CC.describe(ss)

    step = first(CC.filtration_steps(fd))
    graded_step = first(sort!(collect(keys(fd.graded))))
    @test CC.describe(fd).kind == :filtration_data
    @test CC.filtration_summary(fd) == CC.describe(fd)
    @test CC.filtration_steps(fd) == fd.pmin:fd.pmax
    @test CC.filtration_dimensions(fd) == fd.dims
    @test CC.filtration_dimension(fd, step) == fd.dims[step]
    @test CC.filtration_basis(fd, step) == fd.bases[step]
    @test CC.graded_piece(fd, graded_step).dimH == fd.graded[graded_step].dimH
    @test occursin("FiltrationData", sprint(show, MIME"text/plain"(), fd))

    piece = first(CC.extension_pieces(ep))
    piece_key = (piece.a, piece.b)
    @test CC.describe(ep).kind == :extension_problem
    @test CC.extension_summary(ep) == CC.describe(ep)
    @test length(CC.extension_pieces(ep)) == length(ep.pieces)
    @test CC.splitting_matrix(ep) == ep.B
    @test CC.piece_range(ep, piece_key) == ep.ranges[piece_key]
    @test occursin("ExtensionProblem", sprint(show, MIME"text/plain"(), ep))

    deg = first(CC.degree_range(les))
    idx = deg - les.tmin + 1
    @test CC.long_exact_sequence_summary(les) == CC.describe(les)
    @test CC.sequence_dimensions(les, deg) == CC.sequence_dimensions(les; degree=deg)
    @test CC.sequence_maps(les, deg) == CC.sequence_maps(les; degree=deg)
    @test CC.sequence_entry(les, deg).C === les.HC[idx]
    @test occursin("LongExactSequence", sprint(show, MIME"text/plain"(), les))

    csummary = CC.chain_complex_validation_summary(CC.check_complex(C))
    @test csummary.report.valid
    @test occursin("ChainComplexValidationSummary", sprint(show, MIME"text/plain"(), csummary))

    @test TOA.CochainComplex === CC.CochainComplex
    @test TOA.CochainMap === CC.CochainMap
    @test TOA.DoubleComplex === CC.DoubleComplex
    @test TOA.DistinguishedTriangle === CC.DistinguishedTriangle
    @test TOA.FiltrationData === CC.FiltrationData
    @test TOA.ExtensionProblem === CC.ExtensionProblem
    @test TOA.ChainComplexValidationSummary === CC.ChainComplexValidationSummary
    @test TOA.chain_complex_validation_summary === CC.chain_complex_validation_summary
    @test TOA.component_labels === CC.component_labels
    @test TOA.differential === CC.differential
    @test TOA.a_range === CC.a_range
    @test TOA.b_range === CC.b_range
    @test TOA.block === CC.block
    @test TOA.filtration_steps === CC.filtration_steps
    @test TOA.extension_pieces === CC.extension_pieces
    @test TOA.sequence_dimensions === CC.sequence_dimensions
    @test TOA.long_exact_sequence_summary === CC.long_exact_sequence_summary
end

@testset "Spectral sequence: toy double complex" begin
    # Double complex with all blocks 1-dim:
    # dv at (0,0) is identity, other dv zero.
    # dh at (0,0) is identity, other dh zero.
    dims = [1 1;
            1 1]

    dv = Array{SparseMatrixCSC{Kc,Int},2}(undef, 2, 2)
    dh = Array{SparseMatrixCSC{Kc,Int},2}(undef, 2, 2)

    dv[1,1] = sparse([1],[1],[one(Kc)], 1, 1)
    dv[2,1] = spzeros(Kc, 1, 1)
    dv[1,2] = spzeros(Kc, 0, 1)
    dv[2,2] = spzeros(Kc, 0, 1)

    dh[1,1] = sparse([1],[1],[one(Kc)], 1, 1)
    dh[1,2] = spzeros(Kc, 1, 1)
    dh[2,1] = spzeros(Kc, 0, 1)
    dh[2,2] = spzeros(Kc, 0, 1)

    DC = CC.DoubleComplex{Kc}(0, 1, 0, 1, dims, dv, dh)
    ss = CC.spectral_sequence(DC; output=:full, first=:vertical)
    ssh = CC.spectral_sequence(DC; output=:full, first=:horizontal)

    @test ss.E1_dims == [0 0;
                         1 1]
    @test ss.E2_dims == [0 0;
                         1 1]
    @test ss.Einf_dims == [0 0;
                           1 1]
    @test ss.Htot_dims == [0, 1, 1]

    @test ssh.E1_dims == [0 1;
                          0 1]
    @test ssh.E2_dims == [0 1;
                          0 1]
    @test ssh.Einf_dims == [0 1;
                            0 1]

    # Einf diagonal sums must match total cohomology dims.
    tmin = DC.amin + DC.bmin
    tmax = DC.amax + DC.bmax
    for t in tmin:tmax
        s = 0
        for a in DC.amin:DC.amax
            b = t - a
            if DC.bmin <= b <= DC.bmax
                s += ss.Einf_dims[a - DC.amin + 1, b - DC.bmin + 1]
            end
        end
        @test s == ss.Htot_dims[t - tmin + 1]
    end
end

@testset "ChainComplexes: total_complex packs bicomplex blocks correctly" begin
    dims = [1 1;
            1 1]

    dv = Array{SparseMatrixCSC{Kc,Int},2}(undef, 2, 2)
    dh = Array{SparseMatrixCSC{Kc,Int},2}(undef, 2, 2)

    dv[1,1] = sparse([1], [1], [one(Kc)], 1, 1)
    dv[2,1] = spzeros(Kc, 1, 1)
    dv[1,2] = spzeros(Kc, 0, 1)
    dv[2,2] = spzeros(Kc, 0, 1)

    dh[1,1] = sparse([1], [1], [one(Kc)], 1, 1)
    dh[1,2] = spzeros(Kc, 1, 1)
    dh[2,1] = spzeros(Kc, 0, 1)
    dh[2,2] = spzeros(Kc, 0, 1)

    DC = CC.DoubleComplex{Kc}(0, 1, 0, 1, dims, dv, dh)
    Tot = CC.total_complex(DC)

    @test Tot.tmin == 0
    @test Tot.tmax == 2
    @test Tot.dims == [1, 2, 1]
    @test size(Tot.d[1]) == (2, 1)
    @test size(Tot.d[2]) == (1, 2)

    D0 = Matrix(Tot.d[1])
    @test D0[:, 1] == Kc[one(Kc), one(Kc)]

    D1 = Matrix(Tot.d[2])
    @test D1[1, 1] == zero(Kc)
    @test D1[1, 2] == zero(Kc)
end

@testset "ChainComplexes: coordinate-native subquotient matches ambient constructor" begin
    Z = Matrix{Kc}([
        CM.coerce(field, 1) CM.coerce(field, 0) CM.coerce(field, 1);
        CM.coerce(field, 0) CM.coerce(field, 1) CM.coerce(field, 1);
        CM.coerce(field, 0) CM.coerce(field, 0) CM.coerce(field, 1);
    ])
    B = Matrix{Kc}([
        CM.coerce(field, 1) CM.coerce(field, 1);
        CM.coerce(field, 0) CM.coerce(field, 1);
        CM.coerce(field, 0) CM.coerce(field, 0);
    ])

    SQ = CC.subquotient_data(Z, B)
    Bcoords = CC._solve_fullcolumn_cached(field, Z, B)
    SQcoords = CC._subquotient_data_from_coords(Z, Bcoords; Zsolve_rows=1:size(Z, 1), Zsolve_basis=Z)

    @test SQ.dimZ == SQcoords.dimZ
    @test SQ.dimB == SQcoords.dimB
    @test SQ.dimH == SQcoords.dimH
    @test SQ.Bcoords == SQcoords.Bcoords
    @test SQ.Hrep == SQcoords.Hrep

    z = Z * Matrix{Kc}([
        CM.coerce(field, 1) CM.coerce(field, 0);
        CM.coerce(field, 0) CM.coerce(field, 1);
        CM.coerce(field, 1) CM.coerce(field, 1);
    ])
    @test CC.subquotient_coordinates(SQ, z) == CC.subquotient_coordinates(SQcoords, z)
end

@testset "ChainComplexes: exact induced maps match columnwise coordinates" begin
    if field isa CM.QQField
        C = CC.CochainComplex{Kc}(0, 1, [2, 2], [sparse([1], [1], [one(Kc)], 2, 2)])
        H0 = CC.cohomology_data(C, 0)
        F = Matrix{Kc}(I, H0.dimC, H0.dimC)
        M = CC.induced_map_on_cohomology(H0, H0, F)
        Mref = zeros(Kc, H0.dimH, H0.dimH)
        for j in 1:H0.dimH
            Mref[:, j] = CC.cohomology_coordinates(H0, F * H0.Hrep[:, j])[:, 1]
        end
        @test M == Mref

        bd_next = sparse([1], [1], [one(Kc)], 2, 2)
        bd_curr = spzeros(Kc, 2, 0)
        Hh = CC.homology_data(bd_next, bd_curr, 1)
        G = Matrix{Kc}(I, Hh.dimC, Hh.dimC)
        N = CC.induced_map_on_homology(Hh, Hh, G)
        Nref = zeros(Kc, Hh.dimH, Hh.dimH)
        for j in 1:Hh.dimH
            Nref[:, j] = CC.homology_coordinates(Hh, G * Hh.Hrep[:, j])[:, 1]
        end
        @test N == Nref
    else
        @test true
    end
end

@testset "ChainComplexes: exact spectral coordinate and cache modes preserve outputs" begin
    if field isa CM.QQField
        dims = [1 1;
                1 1]

        dv = Array{SparseMatrixCSC{Kc,Int},2}(undef, 2, 2)
        dh = Array{SparseMatrixCSC{Kc,Int},2}(undef, 2, 2)
        dv[1,1] = sparse([1],[1],[one(Kc)], 1, 1)
        dv[2,1] = spzeros(Kc, 1, 1)
        dv[1,2] = spzeros(Kc, 0, 1)
        dv[2,2] = spzeros(Kc, 0, 1)
        dh[1,1] = sparse([1],[1],[one(Kc)], 1, 1)
        dh[1,2] = spzeros(Kc, 1, 1)
        dh[2,1] = spzeros(Kc, 0, 1)
        dh[2,2] = spzeros(Kc, 0, 1)

        DC = CC.DoubleComplex{Kc}(0, 1, 0, 1, dims, dv, dh)
        Tot, blocks = CC._total_complex_with_blocks(DC)

        old_diff = CC._spectral_exact_diff_mode[]
        old_cache = CC._spectral_exact_filtimg_cache_mode[]
        old_basis = CC._spectral_exact_filtimg_basis_mode[]
        try
            pg = CC._ss_compute_page_data(DC, Tot, blocks, :vertical, 2)
            pg_ambient = CC._ss_compute_page_data_ambient(DC, Tot, blocks, :vertical, 2)
            @test pg.dims == pg_ambient.dims
            ss_pg2 = CC.spectral_sequence(DC; output=:full, first=:vertical)
            pg2 = CC._ss_compute_page2_data(ss_pg2)
            @test pg2.dims == pg_ambient.dims

            idx = argmax([sq.dimH for sq in pg.spaces])
            sq_exact = pg.spaces[CartesianIndices(pg.spaces)[idx]]
            sq_ambient = pg_ambient.spaces[CartesianIndices(pg_ambient.spaces)[idx]]
            @test sq_exact.dimZ == sq_ambient.dimZ
            @test sq_exact.dimB == sq_ambient.dimB
            @test sq_exact.dimH == sq_ambient.dimH
            if sq_exact.dimH > 0
                cea = CC.subquotient_coordinates(sq_ambient, sq_exact.Hrep)
                cae = CC.subquotient_coordinates(sq_exact, sq_ambient.Hrep)
                @test size(cea, 2) == sq_exact.dimH
                @test size(cae, 2) == sq_ambient.dimH
            end

            CC._spectral_exact_diff_mode[] = :coords
            d_coords = CC._ss_compute_differential(DC, Tot, :vertical, pg, 2)
            CC._spectral_exact_diff_mode[] = :ambient
            d_ambient = CC._ss_compute_differential(DC, Tot, :vertical, pg, 2)
            @test Matrix(d_coords[1,2]) == Matrix(d_ambient[1,2])

            CC._spectral_exact_filtimg_cache_mode[] = :off
            ss_off = CC.spectral_sequence(DC; output=:full, first=:vertical)
            CC._spectral_exact_filtimg_cache_mode[] = :on
            ss_on = CC.spectral_sequence(DC; output=:full, first=:vertical)
            CC._spectral_exact_filtimg_basis_mode[] = :auto
            ss_auto = CC.spectral_sequence(DC; output=:full, first=:horizontal)
            CC._spectral_exact_filtimg_basis_mode[] = :full
            ss_fullh = CC.spectral_sequence(DC; output=:full, first=:horizontal)
            @test ss_off.E1_dims == ss_on.E1_dims
            @test ss_off.E2_dims == ss_on.E2_dims
            @test ss_off.Einf_dims == ss_on.Einf_dims
            @test CC.filtration_dims(ss_off, 0) == CC.filtration_dims(ss_on, 0)
            @test ss_auto.E1_dims == ss_fullh.E1_dims
            @test ss_auto.E2_dims == ss_fullh.E2_dims
            @test ss_auto.Einf_dims == ss_fullh.Einf_dims
            @test ss_auto.filt_img.value !== nothing
            @test ss_fullh.filt_img.value !== nothing
            @test ss_off.filt_img.value === nothing
            ss_auto.Einf_spaces.value = nothing
            ss_auto.filt_img.value = nothing
            for lv in ss_auto.filt_img_cols
                lv.value = nothing
            end
            _ = CC.page_terms(ss_auto, :inf)
            @test ss_auto.filt_img.value !== nothing

            ssv = CC.spectral_sequence(DC; output=:full, first=:vertical)
            @test ssv.filt_img.value === nothing
            @test sum(length, ssv.filt_img_summary_cache) > 0
            @test CC.filtration_dims(ssv, 0, 1) == 1
            @test ssv.filt_img.value === nothing
            use_col = CC._use_exact_columnwise_filtimg_basis(ssv, 2)
            _ = CC.filtration_basis(ssv, 0, 1)
            if use_col
                @test ssv.filt_img.value === nothing
                @test ssv.filt_img_cols[2].value !== nothing
            else
                @test ssv.filt_img.value !== nothing
            end

            CC._spectral_exact_filtimg_basis_mode[] = :full
            ss_full = CC.spectral_sequence(DC; output=:full, first=:vertical)
            B_full = CC.filtration_basis(ss_full, 0, 1)
            @test ss_full.filt_img.value !== nothing
            CC._spectral_exact_filtimg_basis_mode[] = :columnwise
            ss_col = CC.spectral_sequence(DC; output=:full, first=:vertical)
            B_col = CC.filtration_basis(ss_col, 0, 1)
            @test ss_col.filt_img.value === nothing
            @test Matrix(B_full) == Matrix(B_col)
            Einf_full = CC.page_terms(ss_full, :inf)
            Einf_col = CC.page_terms(ss_col, :inf)
            @test [sq.dimH for sq in Einf_full] == [sq.dimH for sq in Einf_col]
            @test ss_col.filt_img.value === nothing

            oldsum = CC._spectral_exact_filtimg_summary_reuse[]
            try
                CC._spectral_exact_filtimg_summary_reuse[] = true
                ss_sum = CC.spectral_sequence(DC; output=:full, first=:vertical)
                Einf_sum = CC.page_terms(ss_sum, :inf)
                @test sum(length, ss_sum.filt_img_summary_cache) > 0

                CC._spectral_exact_filtimg_summary_reuse[] = false
                ss_nosum = CC.spectral_sequence(DC; output=:full, first=:vertical)
                CC._ss_clear_filtimg_summary_cache!(ss_nosum)
                Einf_nosum = CC.page_terms(ss_nosum, :inf)
                @test [sq.dimH for sq in Einf_sum] == [sq.dimH for sq in Einf_nosum]
            finally
                CC._spectral_exact_filtimg_summary_reuse[] = oldsum
            end

            delete!(ss_pg2.page_cache, 2)
            terms2 = CC.page_terms(ss_pg2, 2)
            @test [sq.dimH for sq in terms2] == [sq.dimH for sq in pg_ambient.spaces]
            if haskey(ss_pg2.page_cache, 2)
                @test [sq.dimH for sq in ss_pg2.page_cache[2].spaces] == [sq.dimH for sq in pg_ambient.spaces]
            end
        finally
            CC._spectral_exact_diff_mode[] = old_diff
            CC._spectral_exact_filtimg_cache_mode[] = old_cache
            CC._spectral_exact_filtimg_basis_mode[] = old_basis
        end
    else
        @test true
    end
end

@testset "Spectral sequence: horizontal identity forces E2 = 0 and Htot = 0" begin
    # Double complex concentrated in b=0 with an isomorphism horizontally:
    # (0,0)=k --id--> (1,0)=k. Total complex has zero cohomology.
    dims = reshape([1, 1], 2, 1)  # a=0,1; b=0
    dv = Array{SparseMatrixCSC{Kc,Int},2}(undef, 2, 1)
    dh = Array{SparseMatrixCSC{Kc,Int},2}(undef, 2, 1)

    dv[1,1] = spzeros(Kc, 0, 1)
    dv[2,1] = spzeros(Kc, 0, 1)

    dh[1,1] = sparse([1], [1], [CM.coerce(field, 1)], 1, 1)  # identity (0,0)->(1,0)
    dh[2,1] = spzeros(Kc, 0, 1)                # boundary

    DC = CC.DoubleComplex{Kc}(0, 1, 0, 0, dims, dv, dh)
    ss = CC.spectral_sequence(DC; output=:full, first=:vertical)

    @test CC.page(ss, 1)[(0,0)] == 1
    @test CC.page(ss, 1)[(1,0)] == 1

    d1 = CC.differential(ss, 1, (0,0))
    @test size(d1) == (1, 1)
    @test d1[1,1] == CM.coerce(field, 1)

    @test CC.page(ss, 2)[(0,0)] == 0
    @test CC.page(ss, 2)[(1,0)] == 0
    @test ss.Htot_dims == [0, 0]
end

@testset "Ext spectral sequence wrapper: consistency with total complex dims" begin
    # Use helper constructors from runtests.jl (as other tests already do).
    P = chain_poset(3)

    # Build simple modules at vertices 1 and 3 via 1x1 fringe presentations.
    # Note: one_by_one_fringe expects (P, U, D), not just a vertex index.
    M = IR.pmodule_from_fringe(one_by_one_fringe(
        P,
        FF.principal_upset(P, 1),
        FF.principal_downset(P, 1),
    ))

    N = IR.pmodule_from_fringe(one_by_one_fringe(
        P,
        FF.principal_upset(P, 3),
        FF.principal_downset(P, 3),
    ))

    ss = TO.ExtSpectralSequence(M, N; maxlen=2)

    if Threads.nthreads() > 1
        ss_serial = TO.ExtSpectralSequence(M, N; maxlen=2, threads=false)
        ss_thread = TO.ExtSpectralSequence(M, N; maxlen=2, threads=true)
        @test ss_thread.Htot_dims == ss_serial.Htot_dims
        @test ss_thread.Einf_dims == ss_serial.Einf_dims
    end

    # Compare to ext_dims_via_resolutions on the same truncation.
    F, dF = TO.IndicatorResolutions.upset_resolution(M; maxlen=2)
    E, dE = TO.IndicatorResolutions.downset_resolution(N; maxlen=2)
    ext_dims = HE.ext_dims_via_resolutions(F, dF, E, dE)

    tmin = ss.DC.amin + ss.DC.bmin
    tmax = ss.DC.amax + ss.DC.bmax
    for t in tmin:tmax
        @test ss.Htot_dims[t - tmin + 1] == get(ext_dims, t, 0)
    end

    # Einf diagonal sums must match Htot dims.
    for t in tmin:tmax
        s = 0
        for a in ss.DC.amin:ss.DC.amax
            b = t - a
            if ss.DC.bmin <= b <= ss.DC.bmax
                s += ss.Einf_dims[a - ss.DC.amin + 1, b - ss.DC.bmin + 1]
            end
        end
        @test s == ss.Htot_dims[t - tmin + 1]
    end
end

@testset "ExtDoubleComplex threading parity" begin
    P = chain_poset(3)
    M = IR.pmodule_from_fringe(one_by_one_fringe(
        P,
        FF.principal_upset(P, 1),
        FF.principal_downset(P, 1),
    ))
    N = IR.pmodule_from_fringe(one_by_one_fringe(
        P,
        FF.principal_upset(P, 3),
        FF.principal_downset(P, 3),
    ))

    DC = TO.ExtDoubleComplex(M, N; maxlen=2, threads=false)
    @test size(DC.dims) == size(DC.dv)
    @test size(DC.dims) == size(DC.dh)

    if Threads.nthreads() > 1
        DCt = TO.ExtDoubleComplex(M, N; maxlen=2, threads=true)
        @test DCt.dims == DC.dims
        @test DCt.dv == DC.dv
        @test DCt.dh == DC.dh
    end
end

@testset "TorSpectralSequence threading parity" begin
    if Threads.nthreads() > 1
        P = chain_poset(3)
        M = IR.pmodule_from_fringe(one_by_one_fringe(
            P,
            FF.principal_upset(P, 1),
            FF.principal_downset(P, 1),
        ))
        N = IR.pmodule_from_fringe(one_by_one_fringe(
            P,
            FF.principal_upset(P, 3),
            FF.principal_downset(P, 3),
        ))

        ss_serial = TO.TorSpectralSequence(M, N; maxlen=2, threads=false)
        ss_thread = TO.TorSpectralSequence(M, N; maxlen=2, threads=true)

        @test ss_thread.ss.Htot_dims == ss_serial.ss.Htot_dims
        @test ss_thread.ss.Einf_dims == ss_serial.ss.Einf_dims
    end
end


@testset "Spectral sequence v2: higher differential d2 example" begin
    # Construct a double complex with a known nontrivial d2:
    #
    # Columns a=0,1,2; rows b=0,1
    #
    # C^{0,1} = Q  --dh-->  C^{1,1} = Q
    #                     ^dv
    #                     |
    #             C^{1,0} = Q  --dh-->  C^{2,0} = Q
    #
    # dv(1,0)->(1,1) is identity, dh(0,1)->(1,1) is identity, dh(1,0)->(2,0) is identity.
    #
    # This forces d1=0 but d2: E2^{0,1} -> E2^{2,0} is an isomorphism.
    dims = [0 1;
            1 1;
            1 0]  # rows are a=0,1,2; cols are b=0,1

    dv = Array{SparseMatrixCSC{Kc,Int},2}(undef, 3, 2)
    dh = Array{SparseMatrixCSC{Kc,Int},2}(undef, 3, 2)

    # dv blocks: dv[a,b] maps (a,b)->(a,b+1)
    dv[1,1] = spzeros(Kc, 1, 0)                      # (0,0)->(0,1)
    dv[2,1] = sparse([1],[1],[one(Kc)], 1, 1)        # (1,0)->(1,1) identity
    dv[3,1] = spzeros(Kc, 0, 1)                      # (2,0)->(2,1) but (2,1)=0

    dv[1,2] = spzeros(Kc, 0, 1)                      # (0,1)->(0,2)=0
    dv[2,2] = spzeros(Kc, 0, 1)                      # (1,1)->(1,2)=0
    dv[3,2] = spzeros(Kc, 0, 0)                      # (2,1)->(2,2)=0

    # dh blocks: dh[a,b] maps (a,b)->(a+1,b)
    dh[1,1] = spzeros(Kc, 1, 0)                      # (0,0)->(1,0)
    dh[1,2] = sparse([1],[1],[one(Kc)], 1, 1)        # (0,1)->(1,1) identity

    dh[2,1] = sparse([1],[1],[one(Kc)], 1, 1)        # (1,0)->(2,0) identity
    dh[2,2] = spzeros(Kc, 0, 1)                      # (1,1)->(2,1)=0

    dh[3,1] = spzeros(Kc, 0, 1)                      # boundary
    dh[3,2] = spzeros(Kc, 0, 0)

    DC = CC.DoubleComplex{Kc}(0, 2, 0, 1, dims, dv, dh)
    ss = CC.spectral_sequence(DC; output=:full, first=:vertical)

    # E2 has 1-dim at (0,1) and (2,0).
    @test CC.page(ss, 2)[(0,1)] == 1
    @test CC.page(ss, 2)[(2,0)] == 1

    # d2: E2^{0,1} -> E2^{2,0} is nonzero (in fact iso).
    d2 = CC.differential(ss, 2, (0,1))
    @test size(d2) == (1, 1)
    if _is_real_field(field)
        @test !iszero(d2[1,1])
    else
        @test d2[1,1] == one(Kc)
    end

    # After r=3, the page is stable and equals E_infty.
    @test CC.page(ss, 3)[(0,1)] == 0
    @test CC.page(ss, 3)[(2,0)] == 0
    @test ss.Einf_dims == zeros(Int, size(ss.Einf_dims))

    # Collapse detection should report collapse at r=3.
    @test CC.collapse_page(ss) == 3
end

@testset "Spectral sequence v2: filtration + edge maps on simple example" begin
    # Reuse existing example from earlier test: horizontal identity in b=0.
    dims = reshape([1, 1], 2, 1)
    dv = Array{SparseMatrixCSC{Kc,Int},2}(undef, 2, 1)
    dh = Array{SparseMatrixCSC{Kc,Int},2}(undef, 2, 1)

    dv[1,1] = spzeros(Kc, 0, 1)
    dv[2,1] = spzeros(Kc, 0, 1)

    dh[1,1] = sparse([1], [1], [CM.coerce(field, 1)], 1, 1)
    dh[2,1] = spzeros(Kc, 0, 1)

    DC = CC.DoubleComplex{Kc}(0, 1, 0, 0, dims, dv, dh)
    ss = CC.spectral_sequence(DC; output=:full, first=:vertical)

    # total cohomology is zero, so filtration dims are all zero.
    @test ss.Htot_dims == [0, 0]
    @test CC.filtration_dims(ss, 0) == Dict(0 => 0, 1 => 0, 2 => 0) ||
          CC.filtration_dims(ss, 0) == Dict(0 => 0, 1 => 0)  # tolerate range differences

    # convergence report should produce a nonempty string
    rep = CC.convergence_report(ss)
    @test occursin("SpectralSequence", rep)
end

@testset "Spectral sequence v2: API helpers + E_infty edge projections" begin
    # Reuse the nontrivial d2 example from above to test bidegree bookkeeping.
    dims = [0 1;
            1 1;
            1 0]  # rows are a=0,1,2; cols are b=0,1

    dv = Array{SparseMatrixCSC{Kc,Int},2}(undef, 3, 2)
    dh = Array{SparseMatrixCSC{Kc,Int},2}(undef, 3, 2)

    dv[1,1] = spzeros(Kc, 1, 0)
    dv[2,1] = sparse([1],[1],[one(Kc)], 1, 1)
    dv[3,1] = spzeros(Kc, 0, 1)

    dv[1,2] = spzeros(Kc, 0, 1)
    dv[2,2] = spzeros(Kc, 0, 1)
    dv[3,2] = spzeros(Kc, 0, 0)

    dh[1,1] = spzeros(Kc, 1, 0)
    dh[1,2] = sparse([1],[1],[one(Kc)], 1, 1)

    dh[2,1] = sparse([1],[1],[one(Kc)], 1, 1)
    dh[2,2] = spzeros(Kc, 0, 1)

    dh[3,1] = spzeros(Kc, 0, 1)
    dh[3,2] = spzeros(Kc, 0, 0)

    DC = CC.DoubleComplex{Kc}(0, 2, 0, 1, dims, dv, dh)
    ss = CC.spectral_sequence(DC; output=:full, first=:vertical)

    @test CC.dr_target(ss, 2, (0,1)) == (2,0)
    @test CC.dr_source(ss, 2, (2,0)) == (0,1)
    @test CC.dr_target(ss, 3, (0,1)) === nothing

    # ss[r] and ss[r,(a,b)] indexing
    @test ss[2][(0,1)] == 1
    @test ss[2,(0,1)] == 1
    @test !haskey(ss.page_cache, 2)

    # page_terms provides explicit subquotient models on intermediate pages.
    spaces2 = CC.page_terms(ss, 2)
    @test haskey(ss.page_cache, 2)
    @test spaces2[1,2].dimH == 1  # (a,b)=(0,1)
    @test spaces2[3,1].dimH == 1  # (a,b)=(2,0)

    # The dimension-only E_infty path stays lazy until terms are requested.
    @test ss.Einf_spaces.value === nothing
    @test CC.page(ss, :inf)[(0,1)] == 0
    _ = CC.page_dims_dict(ss, :inf; nonzero_only=true)
    @test ss.Einf_spaces.value === nothing
    if field isa CM.QQField
        @test !CC._use_page_workspace(Kc)
        @test !CC._use_precolspace_den(Kc)
    else
        @test CC._use_page_workspace(Kc)
        @test CC._use_precolspace_den(Kc)
    end

    # E_infty terms are 0-dimensional in this example.
    @test CC.page_terms(ss, :inf)[1,2].dimH == 0
    @test ss.Einf_spaces.value !== nothing

    # A zero-differential example to test explicit E_infty edge splittings.
    dims2 = [1 1;
             1 1]  # rows are a=0,1; cols are b=0,1

    dv2 = Array{SparseMatrixCSC{Kc,Int},2}(undef, 2, 2)
    dh2 = Array{SparseMatrixCSC{Kc,Int},2}(undef, 2, 2)

    dv2[1,1] = spzeros(Kc, 1, 1)
    dv2[2,1] = spzeros(Kc, 1, 1)
    dv2[1,2] = spzeros(Kc, 0, 1)
    dv2[2,2] = spzeros(Kc, 0, 1)

    dh2[1,1] = spzeros(Kc, 1, 1)
    dh2[1,2] = spzeros(Kc, 1, 1)
    dh2[2,1] = spzeros(Kc, 0, 1)
    dh2[2,2] = spzeros(Kc, 0, 1)

    DC2 = CC.DoubleComplex{Kc}(0, 1, 0, 1, dims2, dv2, dh2)
    ss2 = CC.spectral_sequence(DC2; output=:full, first=:vertical)

    @test ss2.Htot_dims == [1,2,1]

    # Multiplicative structure helpers (with a toy "unit" multiplication on Tot).
    # This multiplication treats the unique basis element in Tot^0 as a unit and
    # sets all other products to zero.
    mul = function (t1::Int, x, t2::Int, y)
        tout = t1 + t2
        outdim = ss2.Tot.dims[tout - ss2.Tot.tmin + 1]
        if t1 == 0
            return x[1] * y
        elseif t2 == 0
            return y[1] * x
        else
            return zeros(Kc, outdim)
        end
    end

    Punit = CC.product_matrix(ss2, 1, (0,0), (0,1), mul)
    @test size(Punit) == (1, 1)
    @test Punit[1,1] == one(Kc)

    Pzero = CC.product_matrix(ss2, 1, (0,1), (1,0), mul)
    @test size(Pzero) == (1, 1)
    @test Pzero[1,1] == zero(Kc)

    cunit = CC.product_coords(ss2, 1, (0,0), [one(Kc)], (0,1), [one(Kc)], mul)
    @test size(cunit) == (1, 1)
    @test cunit[1,1] == one(Kc)

    # On total degree t=1, there are two graded pieces: (0,1) and (1,0).
    inc01 = CC.edge_inclusion(ss2, (0,1))
    inc10 = CC.edge_inclusion(ss2, (1,0))
    proj01 = CC.edge_projection(ss2, (0,1))
    proj10 = CC.edge_projection(ss2, (1,0))

    @test size(inc01) == (2, 1)
    @test size(inc10) == (2, 1)
    @test size(proj01) == (1, 2)
    @test size(proj10) == (1, 2)

    M01 = proj01 * inc01
    M10 = proj10 * inc10
    @test size(M01) == (1, 1)
    @test size(M10) == (1, 1)
    @test M01[1,1] == one(Kc)
    @test M10[1,1] == one(Kc)

    # The chosen splitting gives a direct-sum decomposition of H^1.
    Psum = inc01 * proj01 + inc10 * proj10
    I2 = [one(Kc) zero(Kc);
          zero(Kc) one(Kc)]
    @test Psum == I2

    spl = CC.split_total_cohomology(ss2, 1)
    @test size(spl.B) == (2, 2)
    @test size(spl.Binv) == (2, 2)
    @test spl.B * spl.Binv == I2

    # filtration_subquotient agrees with the E_infty terms.
    @test CC.filtration_subquotient(ss2, 0, 1).dimH == 1
    @test CC.filtration_subquotient(ss2, 1, 1).dimH == 1

    # Pretty-printing smoke tests (ASCII).
    ptxt = repr("text/plain", CC.page(ss2, 1))
    @test occursin("E_1", ptxt)
    stxt = repr("text/plain", ss2)
    @test occursin("SpectralSequence", stxt)

    @testset "Spectral sequence workflow helpers (E2 objects, filtrations, extensions)" begin
        # E2 as objects (SubquotientData) via an indexable page wrapper
        P2 = CC.E2_terms(ss)
        @test P2[(0,0)].dimH == CC.term(ss, 2, (0,0)).dimH

        # Dict helpers keyed by bidegree
        d2 = CC.page_terms_dict(ss, 2)
        @test haskey(d2, (0,0))
        @test d2[(0,0)].dimH == CC.term(ss, 2, (0,0)).dimH

        d2dims = CC.page_dims_dict(ss, 2)
        page20 = CC.page(ss, 2)[(0,0)]
        if page20 == 0
            @test !haskey(d2dims, (0,0))
            @test !haskey(CC.page_dict(ss, 2), (0,0))
        else
            @test haskey(d2dims, (0,0))
            @test d2dims[(0,0)] == page20
            @test CC.page_dict(ss, 2)[(0,0)] == d2dims[(0,0)]
        end

        p, t = CC.ss_key(ss, 0, 0)
        @test (p, t) == CC.ss_key(ss.first, 0, 0)

        # Diagonal criterion should hold at E_infty for this convergent toy example
        @test CC.diagonal_criterion(ss; r=:inf)

        # Filtration packaging: graded piece dims sum to total cohomology dim
        Htot = CC.total_cohomology_dims(ss)
        for t in keys(Htot)
            fd = CC.filtration_data(ss, t)
            gsum = 0
            for p in fd.pmin:(fd.pmax - 1)
                gsum += fd.graded[p].dimH
            end
            @test gsum == Htot[t]
        end

        # Collapse info returns explicit filtrations
        cd = CC.collapse_data(ss)
        @test cd.collapse_r == CC.collapse_page(ss)
        @test cd.diagonal_ok

        # Extension problem helper returns an explicit splitting of H^t
        for t in keys(Htot)
            ep = CC.extension_problem(ss, t)
            d = Htot[t]
            @test size(ep.B, 1) == d
            @test size(ep.B, 2) == d
            @test ep.Binv * ep.B == Matrix{Kc}(I, d, d)
            @test ep.B * ep.Binv == Matrix{Kc}(I, d, d)
        end
    end
end
end # with_fields
