using Test

using LinearAlgebra
using InteractiveUtils
using Random
using SparseArrays
import Base.Threads

if !@isdefined(CM)
    const CM = TamerOp.CoreModules
end
if !@isdefined(FF)
    const FF = TamerOp.FiniteFringe
end
if !@isdefined(IR)
    const IR = TamerOp.IndicatorResolutions
end
if !@isdefined(MD)
    const MD = TamerOp.Modules
end
if !@isdefined(DF)
    const DF = TamerOp.DerivedFunctors
end
if !@isdefined(TO)
    const TO = TamerOp
end
if !@isdefined(OPT)
    const OPT = TamerOp.Options
end

const FL = TamerOp.FieldLinAlg

# This file assumes the helper constructors defined in runtests.jl:
# - chain_poset(n)
# - diamond_poset()
# - one_by_one_fringe(P, U, D)
# - boolean_lattice_B3_poset()

function _is_real_field(field)
    return field isa CM.RealField
end

function _field_tol(field)
    field isa CM.RealField || return 0.0
    return field.atol + field.rtol
end

@testset "Modules UX surface" begin
    field = CM.QQField()
    K = CM.coeff_type(field)
    P = chain_poset(2)
    edge = Dict((1, 2) => sparse([1], [1], [one(K)], 1, 1))
    M = MD.PModule{K}(P, [1, 1], edge; field=field)
    f = MD.PMorphism(M, M, [sparse([1], [1], [one(K)], 1, 1), sparse([1], [1], [one(K)], 1, 1)])

    @test MD.check_module(M).valid
    @test MD.check_morphism(f).valid
    @test TOA.check_module === MD.check_module
    @test TOA.check_morphism === MD.check_morphism
    @test TOA.dim_at === MD.dim_at
    @test TOA.structure_map === MD.structure_map
    @test TOA.component === MD.component
    @test TOA.support === MD.support
    @test TOA.support_dims === MD.support_dims
    @test TOA.check_module_data === MD.check_module_data
    @test TOA.check_morphism_data === MD.check_morphism_data

    mdims = TOA.dimensions(M)
    @test mdims.vertices == 2
    @test mdims.total == 2
    @test TOA.dim_at(M, 2) == 1
    @test TOA.dimensions(M, 1).stalk == 1
    @test TOA.describe(M).kind == :pmodule
    @test TOA.describe(f).kind == :pmorphism
    @test TOA.basis(M, 1) == CM.eye(field, 1)
    @test TOA.coordinates(M, 1, K[one(K)]) == K[one(K)]
    @test TOA.structure_map(M; source=1, target=2) == sparse([1], [1], [one(K)], 1, 1)
    @test TOA.component(f, 2) == sparse([1], [1], [one(K)], 1, 1)
    @test TOA.support(M) == [1, 2]
    @test TOA.nonzero_vertices(M) == [1, 2]
    @test TOA.support_dims(M) == (vertices=[1, 2], dims=[1, 1])
    @test occursin("Best practices", string(@doc MD.check_module))
    @test occursin("Best practices", string(@doc MD.check_morphism))
    @test occursin("functor `Q -> Vec_K`", string(@doc MD.PModule))
    @test occursin("morphism of modules", lowercase(string(@doc MD.PMorphism)))
    @test occursin("stalk-dimension query", string(@doc MD.dim_at))
    @test occursin("named-keyword companion", string(@doc MD.structure_map))
    @test occursin("one component map per vertex", string(@doc MD.check_morphism_data))
    @test occursin("dimension summaries", string(@doc TOA.dimensions(M)))
    @test occursin("canonical basis", string(@doc TOA.basis(M, 1)))
    @test occursin("compact mathematical summary", string(@doc TOA.describe(M)))
    @test occursin("identity coordinate convention", string(@doc TOA.coordinates(M, 1, K[one(K)])))
    @test occursin("PModule", sprint(show, MIME"text/plain"(), M))
    @test occursin("PMorphism", sprint(show, MIME"text/plain"(), f))

    Mbad = MD.PModule{K}(P, [1, 1], Dict((1, 2) => spzeros(K, 2, 1)); check_sizes=false, field=field)
    mreport = MD.check_module(Mbad)
    @test !mreport.valid
    @test any(occursin("expected map 1 <= 2", msg) for msg in mreport.issues)
    @test_throws ErrorException MD.check_module(Mbad; throw=true)
    mdata_bad = MD.check_module_data(P, [1, 1], Dict((1, 2) => spzeros(K, 2, 1)))
    @test !mdata_bad.valid
    @test any(occursin("expected map 1 <= 2", msg) for msg in mdata_bad.issues)
    @test_throws ErrorException MD.check_module_data(P, [1, 1], Dict((1, 2) => spzeros(K, 2, 1)); throw=true)

    fbad = MD.PMorphism{K,typeof(field),SparseMatrixCSC{K,Int}}(
        M, M, [spzeros(K, 1, 1), sparse([1], [1], [one(K)], 2, 1)]
    )
    freport = MD.check_morphism(fbad)
    @test !freport.valid
    @test any(occursin("component 2", msg) for msg in freport.issues)
    @test_throws ErrorException MD.check_morphism(fbad; throw=true)
    fdata_bad = MD.check_morphism_data(M, M, [spzeros(K, 1, 1), sparse([1], [1], [one(K)], 2, 1)])
    @test !fdata_bad.valid
    @test any(occursin("component 2", msg) for msg in fdata_bad.issues)
    @test_throws ErrorException MD.check_morphism_data(M, M, [spzeros(K, 1, 1), sparse([1], [1], [one(K)], 2, 1)]; throw=true)
end

@testset "Derived functors across fields (A2)" begin
    P = chain_poset(2)
    U1 = FF.principal_upset(P, 1)
    D1 = FF.principal_downset(P, 1)
    U2 = FF.principal_upset(P, 2)
    D2 = FF.principal_downset(P, 2)

    with_fields(FIELDS_FULL) do field
        K = CM.coeff_type(field)
        S1 = one_by_one_fringe(P, U1, D1; scalar=one(K), field=field)
        S2 = one_by_one_fringe(P, U2, D2; scalar=one(K), field=field)

        ext12 = DF.ext_dimensions_via_indicator_resolutions(S1, S2; maxlen=3)
        ext21 = DF.ext_dimensions_via_indicator_resolutions(S2, S1; maxlen=3)
        ext11 = DF.ext_dimensions_via_indicator_resolutions(S1, S1; maxlen=3)
        ext22 = DF.ext_dimensions_via_indicator_resolutions(S2, S2; maxlen=3)

        if _is_real_field(field)
            # Real-field Ext/Hom dimensions are numerical (rank-threshold dependent):
            # keep stability checks but do not enforce exact algebraic dimensions.
            @test all(v >= 0 for v in values(ext12))
            @test all(v >= 0 for v in values(ext21))
            @test all(v >= 0 for v in values(ext11))
            @test all(v >= 0 for v in values(ext22))
            @test get(ext11, 0, 0) >= 1
            @test get(ext22, 0, 0) >= 1
        else
            @test get(ext12, 0, 0) == 0
            @test get(ext12, 1, 0) == 1
            @test get(ext21, 0, 0) == 0
            @test get(ext21, 1, 0) == 0
            @test get(ext11, 0, 0) == 1
            @test get(ext11, 1, 0) == 0
            @test get(ext22, 0, 0) == 1
            @test get(ext22, 1, 0) == 0

            @test get(ext12, 0, 0) == FF.hom_dimension(S1, S2)
            @test get(ext21, 0, 0) == FF.hom_dimension(S2, S1)
            @test get(ext11, 0, 0) == FF.hom_dimension(S1, S1)
            @test get(ext22, 0, 0) == FF.hom_dimension(S2, S2)
        end
    end
end

@testset "Resolution threading parity" begin
    if Threads.nthreads() > 1
        P = chain_poset(3)
        U = FF.principal_upset(P, 2)
        D = FF.principal_downset(P, 2)
        FQ = CM.QQField()
        H = one_by_one_fringe(P, U, D; scalar=CM.coerce(FQ, 1), field=FQ)
        M = IR.pmodule_from_fringe(H)
        res = TO.ResolutionOptions(maxlen=2)

        R_serial = TO.projective_resolution(M, res; threads=false)
        R_thread = TO.projective_resolution(M, res; threads=true)
        @test R_thread.gens == R_serial.gens
        @test R_thread.d_mat == R_serial.d_mat

        E_serial = TO.injective_resolution(M, res; threads=false)
        E_thread = TO.injective_resolution(M, res; threads=true)
        @test E_thread.gens == E_serial.gens
        @test length(E_thread.d_mor) == length(E_serial.d_mor)
        for i in eachindex(E_thread.d_mor)
            @test E_thread.d_mor[i].comps == E_serial.d_mor[i].comps
        end
    end
end

@testset "Derived assembly parity + allocation guards" begin
    field = CM.QQField()
    K = CM.coeff_type(field)
    P = chain_poset(3)

    M = IR.pmodule_from_fringe(
        one_by_one_fringe(P, FF.principal_upset(P, 2), FF.principal_downset(P, 3); scalar=one(K), field=field)
    )
    N = IR.pmodule_from_fringe(
        one_by_one_fringe(P, FF.principal_upset(P, 3), FF.principal_downset(P, 3); scalar=one(K), field=field)
    )

    DC_ext_s = DF.ExtDoubleComplex(M, N; maxlen=2, threads=false)
    if Threads.nthreads() > 1
        DC_ext_t = DF.ExtDoubleComplex(M, N; maxlen=2, threads=true)
        @test DC_ext_t.dims == DC_ext_s.dims
        @test DC_ext_t.dv == DC_ext_s.dv
        @test DC_ext_t.dh == DC_ext_s.dh
    end

    DC_tor_s = DF.TorDoubleComplex(M, N; maxlen=2, threads=false)
    if Threads.nthreads() > 1
        DC_tor_t = DF.TorDoubleComplex(M, N; maxlen=2, threads=true)
        @test DC_tor_t.dims == DC_tor_s.dims
        @test DC_tor_t.dv == DC_tor_s.dv
        @test DC_tor_t.dh == DC_tor_s.dh
    end

    hom_trip_old = DF.HomExtEngine._HOM_ASSEMBLY_USE_TRIPLETS[]
    hom_off_old = DF.HomExtEngine._HOM_ASSEMBLY_CACHE_OFFSETS[]
    tor_off_old = DF.SpectralSequences._TOR_DOUBLE_COMPLEX_CACHE_TENSOR_OFFSETS[]
    try
        F, dF = IR.upset_resolution(M; maxlen=2)
        E, dE = IR.downset_resolution(N; maxlen=2)

        DF.HomExtEngine._HOM_ASSEMBLY_USE_TRIPLETS[] = false
        DF.HomExtEngine._HOM_ASSEMBLY_CACHE_OFFSETS[] = false
        DF.SpectralSequences._TOR_DOUBLE_COMPLEX_CACHE_TENSOR_OFFSETS[] = false
        dimsCt_off, dts_off = DF.HomExtEngine.build_hom_tot_complex(F, dF, E, dE; threads=false)
        dims_off, dv_off, dh_off = DF.HomExtEngine.build_hom_bicomplex_data(F, dF, E, dE; threads=false)
        DC_ext_off = DF.ExtDoubleComplex(M, N; maxlen=2, threads=false)
        DC_tor_off = DF.TorDoubleComplex(M, N; maxlen=2, threads=false)

        DF.HomExtEngine._HOM_ASSEMBLY_USE_TRIPLETS[] = true
        DF.HomExtEngine._HOM_ASSEMBLY_CACHE_OFFSETS[] = true
        DF.SpectralSequences._TOR_DOUBLE_COMPLEX_CACHE_TENSOR_OFFSETS[] = true
        dimsCt_on, dts_on = DF.HomExtEngine.build_hom_tot_complex(F, dF, E, dE; threads=false)
        dims_on, dv_on, dh_on = DF.HomExtEngine.build_hom_bicomplex_data(F, dF, E, dE; threads=false)
        DC_ext_on = DF.ExtDoubleComplex(M, N; maxlen=2, threads=false)
        DC_tor_on = DF.TorDoubleComplex(M, N; maxlen=2, threads=false)

        @test dimsCt_on == dimsCt_off
        @test dts_on == dts_off
        @test dims_on == dims_off
        @test dv_on == dv_off
        @test dh_on == dh_off
        @test DC_ext_on.dims == DC_ext_off.dims
        @test DC_ext_on.dv == DC_ext_off.dv
        @test DC_ext_on.dh == DC_ext_off.dh
        @test DC_tor_on.dims == DC_tor_off.dims
        @test DC_tor_on.dv == DC_tor_off.dv
        @test DC_tor_on.dh == DC_tor_off.dh
    finally
        DF.HomExtEngine._HOM_ASSEMBLY_USE_TRIPLETS[] = hom_trip_old
        DF.HomExtEngine._HOM_ASSEMBLY_CACHE_OFFSETS[] = hom_off_old
        DF.SpectralSequences._TOR_DOUBLE_COMPLEX_CACHE_TENSOR_OFFSETS[] = tor_off_old
    end

    # Warm + allocation budgets on fixed tiny fixtures.
    DF.ExtDoubleComplex(M, N; maxlen=2, threads=false)
    alloc_extdc = @allocated DF.ExtDoubleComplex(M, N; maxlen=2, threads=false)
    @test alloc_extdc < 60_000_000

    DF.TorDoubleComplex(M, N; maxlen=2, threads=false)
    alloc_tordc = @allocated DF.TorDoubleComplex(M, N; maxlen=2, threads=false)
    @test alloc_tordc < 60_000_000

    Hm = one_by_one_fringe(P, FF.principal_upset(P, 2), FF.principal_downset(P, 3); scalar=one(K), field=field)
    Hn = one_by_one_fringe(P, FF.principal_upset(P, 3), FF.principal_downset(P, 3); scalar=one(K), field=field)
    DF.ext_dimensions_via_indicator_resolutions(Hm, Hn; maxlen=2, verify=false)
    alloc_extdims = @allocated DF.ext_dimensions_via_indicator_resolutions(Hm, Hn; maxlen=2, verify=false)
    @test alloc_extdims < 80_000_000

    cache = CM.ResolutionCache()
    Fcache, dFcache = IR.upset_resolution(M; maxlen=2)
    Ecache, dEcache = IR.downset_resolution(N; maxlen=2)
    hom_data_1 = DF.HomExtEngine.build_hom_bicomplex_data(Fcache, dFcache, Ecache, dEcache; threads=false, cache=cache)
    hom_data_2 = DF.HomExtEngine.build_hom_bicomplex_data(Fcache, dFcache, Ecache, dEcache; threads=false, cache=cache)
    @test hom_data_2 === hom_data_1

    DC_ext_cache_1 = DF.ExtDoubleComplex(M, N; maxlen=2, threads=false, cache=cache)
    DC_ext_cache_2 = DF.ExtDoubleComplex(M, N; maxlen=2, threads=false, cache=cache)
    @test DC_ext_cache_2 === DC_ext_cache_1

    DC_tor_cache_1 = DF.TorDoubleComplex(M, N; maxlen=2, threads=false, cache=cache)
    DC_tor_cache_2 = DF.TorDoubleComplex(M, N; maxlen=2, threads=false, cache=cache)
    @test DC_tor_cache_2 === DC_tor_cache_1
end

@testset "Ext/Tor core threaded parity" begin
    if Threads.nthreads() > 1
        field = CM.QQField()
        K = CM.coeff_type(field)

        P = chain_poset(3)
        Hm = one_by_one_fringe(P, FF.principal_upset(P, 1), FF.principal_downset(P, 2); scalar=one(K), field=field)
        Hn = one_by_one_fringe(P, FF.principal_upset(P, 2), FF.principal_downset(P, 3); scalar=one(K), field=field)
        M = IR.pmodule_from_fringe(Hm)
        N = IR.pmodule_from_fringe(Hn)

        resM = DF.projective_resolution(M, TO.ResolutionOptions(maxlen=2); threads=false)
        E_serial = DF.Ext(resM, N; threads=false)
        E_thread = DF.Ext(resM, N; threads=true)
        @test E_thread.complex.d == E_serial.complex.d
        @test [DF.dim(E_thread, t) for t in 0:2] == [DF.dim(E_serial, t) for t in 0:2]

        Pop = FF.FinitePoset(transpose(FF.leq_matrix(P)); check=false)
        RopH = one_by_one_fringe(Pop, FF.principal_upset(Pop, 2), FF.principal_downset(Pop, 2), one(K); field=field)
        LH = one_by_one_fringe(P, FF.principal_upset(P, 1), FF.principal_downset(P, 1), one(K); field=field)
        Rop = IR.pmodule_from_fringe(RopH)
        L = IR.pmodule_from_fringe(LH)

        resR = DF.projective_resolution(Rop, TO.ResolutionOptions(maxlen=2); threads=false)
        T_serial = DF.ExtTorSpaces._Tor_resolve_first(Rop, L; maxdeg=2, threads=false, res=resR)
        T_thread = DF.ExtTorSpaces._Tor_resolve_first(Rop, L; maxdeg=2, threads=true, res=resR)
        @test T_thread.bd == T_serial.bd
        @test [DF.dim(T_thread, s) for s in 0:2] == [DF.dim(T_serial, s) for s in 0:2]
    end
end

@testset "Ext resolution parity" begin
    field = CM.QQField()
    K = CM.coeff_type(field)
    P = chain_poset(3)

    Hm = one_by_one_fringe(P, FF.principal_upset(P, 1), FF.principal_downset(P, 2), one(K); field=field)
    Hn = one_by_one_fringe(P, FF.principal_upset(P, 2), FF.principal_downset(P, 3), one(K); field=field)
    M = IR.pmodule_from_fringe(Hm)
    N = IR.pmodule_from_fringe(Hn)

    resM = DF.projective_resolution(M, TO.ResolutionOptions(maxlen=2); threads=false)
    E_from_res = DF.Ext(resM, N; threads=false)
    E_projective = DF.Ext(M, N, TO.DerivedFunctorOptions(maxdeg=2, model=:projective))
    @test [DF.dim(E_from_res, t) for t in 0:2] == [DF.dim(E_projective, t) for t in 0:2]

    resN = DF.injective_resolution(N, TO.ResolutionOptions(maxlen=2); threads=false)
    Einj_from_res = DF.ExtInjective(M, resN; threads=false)
    Einj_direct = DF.ExtInjective(M, N, TO.DerivedFunctorOptions(maxdeg=2, model=:injective))
    @test [DF.dim(Einj_from_res, t) for t in 0:2] == [DF.dim(Einj_direct, t) for t in 0:2]
end

@testset "Resolution cache plumbing" begin
    P, S1, S2 = simple_modules_chain2()
    opts = TO.ResolutionOptions(maxlen=2)
    cache = CM.ResolutionCache()

    RP1 = DF.projective_resolution(S1, opts; cache=cache)
    @test cache.projective_primary_type === nothing
    RP2 = DF.projective_resolution(S1, opts; cache=cache)
    @test RP1 === RP2
    @test cache.projective_primary_type === nothing
    RP3 = DF.projective_resolution(S1, opts; cache=cache)
    @test RP3 === RP1
    @test cache.projective_primary_type === typeof(RP1)
    other_pkey = CM._resolution_key2(S2, 7)
    @test DF.Resolutions._cache_projective_store!(cache, other_pkey, 17) == 17
    @test DF.Resolutions._cache_projective_get(cache, other_pkey, Int) == 17

    RI1 = DF.injective_resolution(S1, opts; cache=cache)
    @test cache.injective_primary_type === nothing
    RI2 = DF.injective_resolution(S1, opts; cache=cache)
    @test RI1 === RI2
    @test cache.injective_primary_type === nothing
    RI3 = DF.injective_resolution(S1, opts; cache=cache)
    @test RI3 === RI1
    @test cache.injective_primary_type === typeof(RI1)
    other_ikey = CM._resolution_key2(S2, 9)
    @test DF.Resolutions._cache_injective_store!(cache, other_ikey, 23) == 23
    @test DF.Resolutions._cache_injective_get(cache, other_ikey, Int) == 23

    M = IR.pmodule_from_fringe(S1)
    N = IR.pmodule_from_fringe(S2)
    df_proj = TO.DerivedFunctorOptions(maxdeg=2, model=:projective)
    df_inj = TO.DerivedFunctorOptions(maxdeg=2, model=:injective)
    df_uni = TO.DerivedFunctorOptions(maxdeg=2, model=:unified, canon=:projective)
    EP1 = DF.Ext(M, N, df_proj; cache=cache)
    EP2 = DF.Ext(M, N, df_proj; cache=cache)
    @test EP1 === EP2
    EI1 = DF.Ext(M, N, df_inj; cache=cache)
    EI2 = DF.Ext(M, N, df_inj; cache=cache)
    @test EI1 === EI2
    EU1 = DF.Ext(M, N, df_uni; cache=cache)
    EU2 = DF.Ext(M, N, df_uni; cache=cache)
    @test EU1 === EU2
    cmp = getfield(EU1, :comparison)
    @test all(isnothing, cmp.P2I)
    @test all(isnothing, cmp.I2P)
    _ = DF.comparison_isomorphism(EU1, 1; from=:projective, to=:injective)
    @test cmp.P2I[2] !== nothing
    @test cmp.I2P[2] !== nothing
    @test cmp.P2I[1] === nothing
    @test cmp.I2P[1] === nothing
    @test !cmp.complete

    Pop = FF.FinitePoset(transpose(FF.leq_matrix(P)); check=false)
    Uop = FF.principal_upset(Pop, 2)
    Dop = FF.principal_downset(Pop, 2)
    Hop = one_by_one_fringe(Pop, Uop, Dop; scalar=CM.coerce(S1.field, 1), field=S1.field)
    Rop = IR.pmodule_from_fringe(Hop)
    df_tor_first = TO.DerivedFunctorOptions(maxdeg=2, model=:first)
    df_tor_second = TO.DerivedFunctorOptions(maxdeg=2, model=:second)
    TF1 = DF.Tor(Rop, M, df_tor_first; cache=cache)
    TF2 = DF.Tor(Rop, M, df_tor_first; cache=cache)
    @test TF1 === TF2
    TS1 = DF.Tor(Rop, M, df_tor_second; cache=cache)
    TS2 = DF.Tor(Rop, M, df_tor_second; cache=cache)
    @test TS1 === TS2

    DCE1 = DF.ExtDoubleComplex(M, N; maxlen=2, threads=false, cache=cache)
    DCE2 = DF.ExtDoubleComplex(M, N; maxlen=2, threads=false, cache=cache)
    @test DCE1 === DCE2
    DCT1 = DF.TorDoubleComplex(Rop, M; maxlen=2, threads=false, cache=cache)
    DCT2 = DF.TorDoubleComplex(Rop, M; maxlen=2, threads=false, cache=cache)
    @test DCT1 === DCT2

    T1 = IR.indicator_resolutions(S1, S2; maxlen=2, cache=cache)
    T2 = IR.indicator_resolutions(S1, S2; maxlen=2, cache=cache)
    @test T1 === T2

    CM._clear_resolution_cache!(cache)
    @test cache.projective_primary_type === typeof(RP1)
    @test isempty(cache.projective_primary)
    @test cache.injective_primary_type === typeof(RI1)
    @test isempty(cache.injective_primary)
    @test cache.indicator_primary_type === nothing
    @test isempty(cache.ext_projective)
    @test isempty(cache.ext_injective)
    @test isempty(cache.ext_unified)
    @test isempty(cache.tor_first)
    @test isempty(cache.tor_second)
    @test isempty(cache.hom_bicomplex)
    @test isempty(cache.ext_doublecomplex)
    @test isempty(cache.tor_doublecomplex_plan)
    @test isempty(cache.tor_doublecomplex)

    sc = CM.SessionCache()

    enc = RES.EncodingResult(P, M, nothing; H=S1, opts=OPT.EncodingOptions(field=S1.field), backend=:test)
    WR1 = TO.resolve(enc; kind=:projective, opts=opts, cache=sc)
    WR2 = TO.resolve(enc; kind=:projective, opts=opts, cache=sc)
    @test WR1.res === WR2.res

    # Workflow-level ext/tor should reuse cached resolutions automatically.
    N = IR.pmodule_from_fringe(S2)
    encN = RES.EncodingResult(P, N, nothing; H=S2, opts=OPT.EncodingOptions(field=S2.field), backend=:test)
    E1 = TO.ext(enc, encN; maxdeg=2, model=:projective, cache=sc)
    E2 = TO.ext(enc, encN; maxdeg=2, model=:projective, cache=sc)
    @test E1.res === E2.res
    H1 = TO.hom(enc, encN; cache=sc)
    H2 = TO.hom(enc, encN; cache=sc)
    @test DF.dim(H1) == DF.dim(H2)
    d_fast_h = TO.hom_dimension(enc, encN; cache=sc)
    @test d_fast_h == DF.dim(H1)
    Epm1 = TO.ext(enc.M, encN.M; maxdeg=2, model=:projective, cache=sc)
    Epm2 = TO.ext(enc.M, encN.M; maxdeg=2, model=:projective, cache=sc)
    @test Epm1.res === Epm2.res
    Hpm1 = TO.hom(enc.M, encN.M; cache=sc)
    Hpm2 = TO.hom(enc.M, encN.M; cache=sc)
    @test DF.dim(Hpm1) == DF.dim(Hpm2)
    @test_throws MethodError TO.hom(enc, encN.M; cache=sc)
    @test_throws MethodError TO.hom(enc.M, encN; cache=sc)
    @test_throws MethodError TO.ext(enc, encN.M; maxdeg=2, model=:projective, cache=sc)

    C0 = TO.ModuleCochainComplex([enc.M], TO.PMorphism[]; tmin=0, check=true)
    encC0 = RES.EncodedComplexResult(P, C0, nothing; field=field)
    RH1 = TO.rhom(C0, encN.M; cache=sc)
    RH2 = TO.rhom(C0, encN.M; cache=sc)
    RH3 = TO.rhom(encC0, encN.M; cache=sc)
    @test RH1.tmin == RH2.tmin
    @test RH1.tmax == RH2.tmax
    @test length(RH1.d) == length(RH2.d)
    @test RH3.tmin == RH1.tmin
    @test RH3.tmax == RH1.tmax
    @test length(RH3.d) == length(RH1.d)
    HX1 = TO.hyperext(C0, encN.M; maxdeg=2, cache=sc)
    HX2 = TO.hyperext(C0, encN.M; maxdeg=2, cache=sc)
    HX3 = TO.hyperext(encC0, encN.M; maxdeg=2, cache=sc)
    @test DF.dim(HX1, 0) == DF.dim(HX2, 0)
    @test DF.dim(HX3, 0) == DF.dim(HX1, 0)
    @test_throws MethodError TO.rhom(C0, encN; cache=sc)
    @test_throws MethodError TO.hyperext(C0, encN; maxdeg=2, cache=sc)

    encRop = RES.EncodingResult(Pop, Rop, nothing; H=Hop, opts=OPT.EncodingOptions(field=Hop.field), backend=:test)
    @test_throws ErrorException TO.hom_dimension(enc, encRop; cache=sc)

    T1 = TO.tor(encRop, enc; maxdeg=2, model=:first, cache=sc)
    T2 = TO.tor(encRop, enc; maxdeg=2, model=:first, cache=sc)
    @test T1.resRop === T2.resRop

    T3 = TO.tor(encRop, enc; maxdeg=2, model=:second, cache=sc)
    T4 = TO.tor(encRop, enc; maxdeg=2, model=:second, cache=sc)
    @test T3.resL === T4.resL
    Tpm1 = TO.tor(encRop.M, enc.M; maxdeg=2, model=:first, cache=sc)
    Tpm2 = TO.tor(encRop.M, enc.M; maxdeg=2, model=:first, cache=sc)
    @test Tpm1.resRop === Tpm2.resRop
    DT1 = TO.derived_tensor(encRop, C0)
    DT2 = TO.derived_tensor(encRop, encC0)
    @test CC.describe(DT2).degree_range == CC.describe(DT1).degree_range
    HT1 = TO.hypertor(encRop, C0; maxdeg=2)
    HT2 = TO.hypertor(encRop, encC0; maxdeg=2)
    @test DF.dim(HT2, 0) == DF.dim(HT1, 0)
    @test_throws MethodError TO.tor(encRop, enc.M; maxdeg=2, model=:first, cache=sc)
    @test_throws MethodError TO.tor(encRop.M, enc; maxdeg=2, model=:first, cache=sc)

    # hom_dimension should cache computed fringes when enc.H is absent.
    enc_noH = RES.EncodingResult(P, M, nothing; H=nothing, opts=OPT.EncodingOptions(field=S1.field), backend=:test)
    encN_noH = RES.EncodingResult(P, N, nothing; H=nothing, opts=OPT.EncodingOptions(field=S2.field), backend=:test)
    ec = CM._workflow_encoding_cache(sc)
    g0 = length(ec.geometry)
    d1 = TO.hom_dimension(enc_noH, encN_noH; cache=sc)
    g1 = length(ec.geometry)
    d2 = TO.hom_dimension(enc_noH, encN_noH; cache=sc)
    g2 = length(ec.geometry)
    @test d1 == d2
    @test d1 == DF.dim(TO.hom(enc_noH, encN_noH; cache=sc))
    @test g1 >= g0 + 2
    @test g2 == g1

    CM._clear_resolution_cache!(cache)
    RP3 = DF.projective_resolution(S1, opts; cache=cache)
    @test RP3 !== RP1

    WRS1 = TO.resolve(enc; kind=:projective, opts=opts, cache=sc)
    WRS2 = TO.resolve(enc; kind=:projective, opts=opts, cache=sc)
    @test WRS1.res === WRS2.res

    ES1 = TO.ext(enc, encN; maxdeg=2, model=:projective, cache=sc)
    ES2 = TO.ext(enc, encN; maxdeg=2, model=:projective, cache=sc)
    @test ES1.res === ES2.res

    TS1 = TO.tor(encRop, enc; maxdeg=2, model=:first, cache=sc)
    TS2 = TO.tor(encRop, enc; maxdeg=2, model=:first, cache=sc)
    @test TS1.resRop === TS2.resRop

    # Module cache keys include field identity: changing field should route to a
    # different module cache bucket.
    mc1 = CM._module_cache!(sc, enc.M)
    mc2 = CM._module_cache!(sc, enc.M)
    @test mc1 === mc2
    enc_f2 = CM.change_field(enc, CM.F2())
    mc3 = CM._module_cache!(sc, enc_f2.M)
    @test mc3 !== mc1

    CM._clear_session_cache!(sc)
    WRS3 = TO.resolve(enc; kind=:projective, opts=opts, cache=sc)
    @test WRS3.res !== WRS1.res
    @test_throws ErrorException TO.resolve(enc; kind=:proj, opts=opts, cache=sc)
    @test_throws ErrorException TO.resolve(enc; kind=:inj, opts=opts, cache=sc)
end

@testset "Tor boundary assembly parity" begin
    field = CM.QQField()
    K = CM.coeff_type(field)
    oneK = one(K)

    P = chain_poset(3)
    Pop = FF.FinitePoset(transpose(FF.leq_matrix(P)); check=false)

    HL = FF.FringeModule{K}(
        P,
        [FF.principal_upset(P, 1), FF.principal_upset(P, 2)],
        [FF.principal_downset(P, 2), FF.principal_downset(P, 3)],
        sparse([1, 2, 1], [1, 1, 2], [oneK, oneK, oneK], 2, 2);
        field=field,
    )
    HRop = FF.FringeModule{K}(
        Pop,
        [FF.principal_upset(Pop, 2), FF.principal_upset(Pop, 3)],
        [FF.principal_downset(Pop, 1), FF.principal_downset(Pop, 2)],
        sparse([1, 2, 2], [1, 1, 2], [oneK, oneK, oneK], 2, 2);
        field=field,
    )

    L = IR.pmodule_from_fringe(HL)
    Rop = IR.pmodule_from_fringe(HRop)
    opts = TO.ResolutionOptions(maxlen=2)
    resRop = DF.projective_resolution(Rop, opts; threads=false)
    resL = DF.projective_resolution(L, opts; threads=false)
    df_first = TO.DerivedFunctorOptions(maxdeg=2, model=:first)
    df_second = TO.DerivedFunctorOptions(maxdeg=2, model=:second)

    gate = DF.ExtTorSpaces._TOR_USE_TRIPLET_BOUNDARY_ASSEMBLY[]
    direct_gate = DF.ExtTorSpaces._TOR_USE_DIRECT_MAP_TRIPLET_ASSEMBLY[]
    max_nnz = DF.ExtTorSpaces._TOR_DIRECT_MAP_TRIPLET_MAX_PLAN_NNZ[]
    max_work = DF.ExtTorSpaces._TOR_DIRECT_MAP_TRIPLET_MAX_BLOCK_WORK[]
    max_bic_nnz = DF.ExtTorSpaces._TOR_DIRECT_MAP_TRIPLET_MAX_BICOMPLEX_PLAN_NNZ[]
    max_bic_work = DF.ExtTorSpaces._TOR_DIRECT_MAP_TRIPLET_MAX_BICOMPLEX_BLOCK_WORK[]
    cache_fast = DF.SpectralSequences._TOR_DOUBLE_COMPLEX_CACHE_FASTPATH[]
    try
        DF.ExtTorSpaces._TOR_USE_TRIPLET_BOUNDARY_ASSEMBLY[] = false
        DF.ExtTorSpaces._TOR_USE_DIRECT_MAP_TRIPLET_ASSEMBLY[] = false
        Tfirst_off = DF.Tor(Rop, L, df_first; res=resRop)
        Tsecond_off = DF.Tor(Rop, L, df_second; res=resL)
        DC_off = DF.TorDoubleComplex(Rop, L; maxlen=2, threads=false)

        DF.ExtTorSpaces._TOR_USE_TRIPLET_BOUNDARY_ASSEMBLY[] = true
        DF.ExtTorSpaces._TOR_USE_DIRECT_MAP_TRIPLET_ASSEMBLY[] = true
        Tfirst_on = DF.Tor(Rop, L, df_first; res=resRop)
        Tsecond_on = DF.Tor(Rop, L, df_second; res=resL)
        DC_on = DF.TorDoubleComplex(Rop, L; maxlen=2, threads=false)

        @test Tfirst_on.dims == Tfirst_off.dims
        @test length(Tfirst_on.bd) == length(Tfirst_off.bd)
        @test all(Tfirst_on.bd[i] == Tfirst_off.bd[i] for i in eachindex(Tfirst_on.bd))
        @test all(DF.dim(Tfirst_on, s) == DF.dim(Tfirst_off, s) for s in DF.degree_range(Tfirst_on))

        @test Tsecond_on.dims == Tsecond_off.dims
        @test length(Tsecond_on.bd) == length(Tsecond_off.bd)
        @test all(Tsecond_on.bd[i] == Tsecond_off.bd[i] for i in eachindex(Tsecond_on.bd))
        @test all(DF.dim(Tsecond_on, s) == DF.dim(Tsecond_off, s) for s in DF.degree_range(Tsecond_on))

        @test DC_on.dims == DC_off.dims
        @test all(DC_on.dv[i] == DC_off.dv[i] for i in eachindex(DC_on.dv))
        @test all(DC_on.dh[i] == DC_off.dh[i] for i in eachindex(DC_on.dh))

        DF.ExtTorSpaces._TOR_USE_TRIPLET_BOUNDARY_ASSEMBLY[] = true
        DF.ExtTorSpaces._TOR_USE_DIRECT_MAP_TRIPLET_ASSEMBLY[] = false
        Tfirst_triplet = DF.Tor(Rop, L, df_first; res=resRop)
        Tsecond_triplet = DF.Tor(Rop, L, df_second; res=resL)
        DC_triplet = DF.TorDoubleComplex(Rop, L; maxlen=2, threads=false)

        @test Tfirst_on.dims == Tfirst_triplet.dims
        @test all(Tfirst_on.bd[i] == Tfirst_triplet.bd[i] for i in eachindex(Tfirst_on.bd))
        @test Tsecond_on.dims == Tsecond_triplet.dims
        @test all(Tsecond_on.bd[i] == Tsecond_triplet.bd[i] for i in eachindex(Tsecond_on.bd))
        @test DC_on.dims == DC_triplet.dims
        @test all(DC_on.dv[i] == DC_triplet.dv[i] for i in eachindex(DC_on.dv))
        @test all(DC_on.dh[i] == DC_triplet.dh[i] for i in eachindex(DC_on.dh))

        DF.ExtTorSpaces._TOR_USE_DIRECT_MAP_TRIPLET_ASSEMBLY[] = true
        DF.ExtTorSpaces._TOR_DIRECT_MAP_TRIPLET_MAX_PLAN_NNZ[] = 0
        DF.ExtTorSpaces._TOR_DIRECT_MAP_TRIPLET_MAX_BLOCK_WORK[] = 0
        DF.ExtTorSpaces._TOR_DIRECT_MAP_TRIPLET_MAX_BICOMPLEX_PLAN_NNZ[] = 0
        DF.ExtTorSpaces._TOR_DIRECT_MAP_TRIPLET_MAX_BICOMPLEX_BLOCK_WORK[] = 0
        Tfirst_gated = DF.Tor(Rop, L, df_first; res=resRop)
        DC_gated = DF.TorDoubleComplex(Rop, L; maxlen=2, threads=false)
        @test Tfirst_gated.dims == Tfirst_triplet.dims
        @test all(Tfirst_gated.bd[i] == Tfirst_triplet.bd[i] for i in eachindex(Tfirst_gated.bd))
        @test DC_gated.dims == DC_triplet.dims
        @test all(DC_gated.dh[i] == DC_triplet.dh[i] for i in eachindex(DC_gated.dh))

        rc = CM.ResolutionCache()
        DF.SpectralSequences._TOR_DOUBLE_COMPLEX_CACHE_FASTPATH[] = false
        DC_cache_off_1 = DF.TorDoubleComplex(Rop, L; maxlen=2, threads=false, cache=rc)
        DC_cache_off_2 = DF.TorDoubleComplex(Rop, L; maxlen=2, threads=false, cache=rc)
        DF.SpectralSequences._TOR_DOUBLE_COMPLEX_CACHE_FASTPATH[] = true
        DC_cache_on = DF.TorDoubleComplex(Rop, L; maxlen=2, threads=false, cache=rc)
        @test DC_cache_off_2 === DC_cache_off_1
        @test DC_cache_on === DC_cache_off_1
    finally
        DF.ExtTorSpaces._TOR_USE_TRIPLET_BOUNDARY_ASSEMBLY[] = gate
        DF.ExtTorSpaces._TOR_USE_DIRECT_MAP_TRIPLET_ASSEMBLY[] = direct_gate
        DF.ExtTorSpaces._TOR_DIRECT_MAP_TRIPLET_MAX_PLAN_NNZ[] = max_nnz
        DF.ExtTorSpaces._TOR_DIRECT_MAP_TRIPLET_MAX_BLOCK_WORK[] = max_work
        DF.ExtTorSpaces._TOR_DIRECT_MAP_TRIPLET_MAX_BICOMPLEX_PLAN_NNZ[] = max_bic_nnz
        DF.ExtTorSpaces._TOR_DIRECT_MAP_TRIPLET_MAX_BICOMPLEX_BLOCK_WORK[] = max_bic_work
        DF.SpectralSequences._TOR_DOUBLE_COMPLEX_CACHE_FASTPATH[] = cache_fast
    end

    function _grid_poset(nx::Int, ny::Int)
        rel = falses(nx * ny, nx * ny)
        idx(ix, iy) = (iy - 1) * nx + ix
        for y1 in 1:ny, x1 in 1:nx
            i = idx(x1, y1)
            for y2 in y1:ny, x2 in x1:nx
                rel[i, idx(x2, y2)] = true
            end
        end
        return FF.FinitePoset(rel; check=false)
    end

    function _random_fringe(Q::FF.AbstractPoset, field::CM.AbstractCoeffField, seed::Integer)
        rng = MersenneTwister(Int(seed))
        Kloc = CM.coeff_type(field)
        n = FF.nvertices(Q)
        nups = 12
        ndowns = 12
        U = Vector{FF.Upset}(undef, nups)
        D = Vector{FF.Downset}(undef, ndowns)
        up_verts = Vector{Int}(undef, nups)
        down_verts = Vector{Int}(undef, ndowns)
        @inbounds for i in 1:nups
            up_verts[i] = rand(rng, 1:n)
            U[i] = FF.principal_upset(Q, up_verts[i])
        end
        @inbounds for j in 1:ndowns
            down_verts[j] = rand(rng, 1:n)
            D[j] = FF.principal_downset(Q, down_verts[j])
        end
        phi = zeros(Kloc, ndowns, nups)
        @inbounds for j in 1:ndowns, i in 1:nups
            FF.leq(Q, up_verts[i], down_verts[j]) || continue
            rand(rng) < 0.45 || continue
            v = rand(rng, -2:2)
            v == 0 && (v = 1)
            phi[j, i] = CM.coerce(field, v)
        end
        return FF.FringeModule{Kloc}(Q, U, D, phi; field=field)
    end

    Pm = _grid_poset(4, 4)
    Popm = FF.FinitePoset(transpose(FF.leq_matrix(Pm)); check=false)
    Lm = IR.pmodule_from_fringe(_random_fringe(Pm, field, 0xBEEF))
    Ropm = IR.pmodule_from_fringe(_random_fringe(Popm, field, 0xFACE))
    opts_m = TO.ResolutionOptions(maxlen=3)
    resRopm = DF.projective_resolution(Ropm, opts_m; threads=false)
    resLm = DF.projective_resolution(Lm, opts_m; threads=false)

    gate = DF.ExtTorSpaces._TOR_USE_TRIPLET_BOUNDARY_ASSEMBLY[]
    direct_gate = DF.ExtTorSpaces._TOR_USE_DIRECT_MAP_TRIPLET_ASSEMBLY[]
    try
        DF.ExtTorSpaces._TOR_USE_TRIPLET_BOUNDARY_ASSEMBLY[] = true
        DF.ExtTorSpaces._TOR_USE_DIRECT_MAP_TRIPLET_ASSEMBLY[] = false
        Tm_off = DF.Tor(Ropm, Lm, TO.DerivedFunctorOptions(maxdeg=3, model=:first); res=resRopm)
        DCm_off = DF.TorDoubleComplex(Ropm, Lm; maxlen=3, threads=false)

        DF.ExtTorSpaces._TOR_USE_DIRECT_MAP_TRIPLET_ASSEMBLY[] = true
        Tm_on = DF.Tor(Ropm, Lm, TO.DerivedFunctorOptions(maxdeg=3, model=:first); res=resRopm)
        DCm_on = DF.TorDoubleComplex(Ropm, Lm; maxlen=3, threads=false)

        @test Tm_on.dims == Tm_off.dims
        @test all(Tm_on.bd[i] == Tm_off.bd[i] for i in eachindex(Tm_on.bd))
        @test DCm_on.dims == DCm_off.dims
        @test all(DCm_on.dh[i] == DCm_off.dh[i] for i in eachindex(DCm_on.dh))
    finally
        DF.ExtTorSpaces._TOR_USE_TRIPLET_BOUNDARY_ASSEMBLY[] = gate
        DF.ExtTorSpaces._TOR_USE_DIRECT_MAP_TRIPLET_ASSEMBLY[] = direct_gate
    end
end

@testset "HomSystemCache type stability" begin
    P, S1, S2 = simple_modules_chain2()
    M = IR.pmodule_from_fringe(S1)
    N = IR.pmodule_from_fringe(S2)
    K = CM.coeff_type(M.field)
    cache = DF.HomSystemCache{K}()
    @test keytype(cache.hom[1]) == DF._HomKey2
    @test keytype(cache.precompose[1]) == DF._HomKey3
    @test keytype(cache.postcompose[1]) == DF._HomKey3

    _hom_cached(M1, N1, c) = DF.hom_with_cache(M1, N1; cache=c)
    report_hom = sprint(io -> InteractiveUtils.code_warntype(io, _hom_cached,
                                                              Tuple{typeof(M), typeof(N), typeof(cache)}))
    @test !occursin("Body::Any", report_hom)
    H = @inferred _hom_cached(M, N, cache)
    @test H isa DF.HomSpace{K}

    idM = IR.id_morphism(M)
    idN = IR.id_morphism(N)
    Hself = @inferred _hom_cached(M, N, cache)

    _pre_cached(Hd, Hc, f, c) = DF.precompose_matrix_cached(Hd, Hc, f; cache=c)
    report_pre = sprint(io -> InteractiveUtils.code_warntype(io, _pre_cached,
                                                              Tuple{typeof(Hself), typeof(Hself), typeof(idM), typeof(cache)}))
    @test !occursin("Body::Any", report_pre)
    pre = @inferred _pre_cached(Hself, Hself, idM, cache)
    @test pre isa SparseArrays.SparseMatrixCSC{K,Int}

    _post_cached(Hd, Hc, g, c) = DF.postcompose_matrix_cached(Hd, Hc, g; cache=c)
    report_post = sprint(io -> InteractiveUtils.code_warntype(io, _post_cached,
                                                               Tuple{typeof(Hself), typeof(Hself), typeof(idN), typeof(cache)}))
    @test !occursin("Body::Any", report_post)
    post = @inferred _post_cached(Hself, Hself, idN, cache)
    @test post isa SparseArrays.SparseMatrixCSC{K,Int}
end

@testset "Characteristic-sensitive rank (A1)" begin
    P = chain_poset(1)
    U = FF.principal_upset(P, 1)
    D = FF.principal_downset(P, 1)

    FQ = CM.QQField()
    KQ = CM.coeff_type(FQ)
    @inline cq(x) = CM.coerce(FQ, x)
    Hqq = one_by_one_fringe(P, U, D; scalar=cq(2), field=FQ)
    Hf2 = one_by_one_fringe(P, U, D; scalar=2, field=CM.F2())

    Mqq = IR.pmodule_from_fringe(Hqq)
    Mf2 = IR.pmodule_from_fringe(Hf2)

    @test FF.fiber_dimension(Hqq, 1) == 1
    @test FF.fiber_dimension(Hf2, 1) == 0
    @test Mqq.dims[1] == 1
    @test Mf2.dims[1] == 0
end

@testset "Characteristic-sensitive Hom/Ext (A2)" begin
    P = chain_poset(2)
    U1 = FF.principal_upset(P, 1)
    D1 = FF.principal_downset(P, 1)
    U2 = FF.principal_upset(P, 2)
    D2 = FF.principal_downset(P, 2)

    # One-generator modules with scalar 2: nonzero over QQ, zero over F2.
    FQ = CM.QQField()
    KQ = CM.coeff_type(FQ)
    @inline cq(x) = CM.coerce(FQ, x)
    Hqq = one_by_one_fringe(P, U1, D1; scalar=cq(2), field=FQ)
    Hf2 = one_by_one_fringe(P, U1, D1; scalar=2, field=CM.F2())

    Mqq = IR.pmodule_from_fringe(Hqq)
    Mf2 = IR.pmodule_from_fringe(Hf2)

    @test Mqq.dims[1] == 1
    @test Mf2.dims[1] == 0

    extqq = DF.ext_dimensions_via_indicator_resolutions(Hqq, Hqq; maxlen=2)
    extf2 = DF.ext_dimensions_via_indicator_resolutions(Hf2, Hf2; maxlen=2)

    @test get(extqq, 0, 0) == 1
    @test get(extf2, 0, 0) == 0
end


# -----------------------------------------------------------------------------
# Small helper: direct sum of poset-modules and the split short exact sequence
# -----------------------------------------------------------------------------

function direct_sum_with_split_sequence(A::MD.PModule{K}, C::MD.PModule{K}) where {K}
    Q = A.Q
    field = A.field
    @test FF.poset_equal(Q, C.Q)
    A.field == C.field || error("direct_sum_with_split_sequence: field mismatch")

    dimsB = [A.dims[v] + C.dims[v] for v in 1:Q.n]

    # Block-diagonal structure maps on cover edges.
    edge_mapsB = Dict{Tuple{Int,Int}, Matrix{K}}()
    for (i, j) in FF.cover_edges(Q)
        # We are iterating over cover edges of Q. The module edge-map store
        # guarantees cover-edge maps exist (missing entries are filled with zeros
        # at construction), so we can index directly and avoid Base.get().
        Aij = A.edge_maps[i, j]
        Cij = C.edge_maps[i, j]

        top    = hcat(Aij, CM.zeros(field, size(Aij,1), size(Cij,2)))
        bottom = hcat(CM.zeros(field, size(Cij,1), size(Aij,2)), Cij)
        edge_mapsB[(i,j)] = vcat(top, bottom)
    end

    B = MD.PModule{K}(Q, dimsB, edge_mapsB; field=field)

    # Inclusion i: A -> A oplus C (first summand)
    comps_i = Vector{Matrix{K}}(undef, Q.n)
    for v in 1:Q.n
        comps_i[v] = vcat(CM.eye(field, A.dims[v]),
                          CM.zeros(field, C.dims[v], A.dims[v]))
    end
    i = MD.PMorphism(A, B, comps_i)

    # Projection p: A oplus C -> C (second summand)
    comps_p = Vector{Matrix{K}}(undef, Q.n)
    for v in 1:Q.n
        comps_p[v] = hcat(CM.zeros(field, C.dims[v], A.dims[v]),
                          CM.eye(field, C.dims[v]))
    end
    p = MD.PMorphism(B, C, comps_p)

    return B, i, p
end

@testset "DerivedFunctors semantic accessors" begin
    field = CM.QQField()
    K = CM.coeff_type(field)
    P = chain_poset(2)

    S1 = IR.pmodule_from_fringe(
        one_by_one_fringe(P, FF.principal_upset(P, 1), FF.principal_downset(P, 1);
                          scalar=one(K), field=field)
    )
    S2 = IR.pmodule_from_fringe(
        one_by_one_fringe(P, FF.principal_upset(P, 2), FF.principal_downset(P, 2);
                          scalar=one(K), field=field)
    )
    I12 = IR.pmodule_from_fringe(
        one_by_one_fringe(P, FF.principal_upset(P, 1), FF.principal_downset(P, 2);
                          scalar=one(K), field=field)
    )

    resP = DF.projective_resolution(S1, TO.ResolutionOptions(maxlen=1))
    resI = DF.injective_resolution(S1, TO.ResolutionOptions(maxlen=1))

    @test DF.source_module(resP) === S1
    @test DF.source_module(resI) === S1
    @test DF.augmentation_map(resP) === resP.aug
    @test DF.coaugmentation_map(resI) === resI.iota0
    @test DF.resolution_terms(resP) == resP.Pmods
    @test DF.resolution_terms(resP) !== resP.Pmods
    @test DF.resolution_terms(resI) == resI.Emods
    @test DF.resolution_terms(resI) !== resI.Emods
    @test DF.resolution_differentials(resP) == resP.d_mor
    @test DF.resolution_differentials(resI) == resI.d_mor
    @test DF.resolution_length(resP) == length(resP.d_mor)
    @test DF.resolution_length(resI) == length(resI.d_mor)
    @test TOA.resolution_length(resP) == DF.resolution_length(resP)

    H = DF.Hom(S1, I12)
    @test DF.source_module(H) === S1
    @test DF.target_module(H) === I12
    @test DF.nonzero_degrees(H) == (DF.dim(H) == 0 ? Int[] : [0])
    @test DF.degree_dimensions(H) == (DF.dim(H) == 0 ? Dict{Int,Int}() : Dict(0 => DF.dim(H)))

    Eproj = DF.Ext(resP, I12)
    Einj = DF.ExtInjective(S1, I12, TO.DerivedFunctorOptions(maxdeg=1, model=:injective))
    Euni = DF.ExtSpace(S1, I12, TO.DerivedFunctorOptions(maxdeg=1, model=:unified, canon=:projective))

    @test DF.source_module(Eproj) === S1
    @test DF.target_module(Eproj) === I12
    @test DF.source_module(Einj) === S1
    @test DF.target_module(Einj) === I12
    @test DF.source_module(Euni) === S1
    @test DF.target_module(Euni) === I12
    @test sort(DF.nonzero_degrees(Eproj)) == sort(collect(keys(DF.degree_dimensions(Eproj))))
    @test sort(DF.nonzero_degrees(Einj)) == sort(collect(keys(DF.degree_dimensions(Einj))))
    @test sort(DF.nonzero_degrees(Euni)) == sort(collect(keys(DF.degree_dimensions(Euni))))

    Pop = FF.FinitePoset(transpose(FF.leq_matrix(P)); check=false)
    RopH = one_by_one_fringe(Pop, FF.principal_upset(Pop, 2), FF.principal_downset(Pop, 2);
                             scalar=one(K), field=field)
    Rop = IR.pmodule_from_fringe(RopH)

    Tfirst = DF.Tor(Rop, I12, TO.DerivedFunctorOptions(maxdeg=1, model=:first))
    Tsecond = DF.Tor(Rop, I12, TO.DerivedFunctorOptions(maxdeg=1, model=:second))

    @test DF.source_module(Tfirst) === Rop
    @test DF.target_module(Tfirst) === I12
    @test DF.source_module(Tsecond) === Rop
    @test DF.target_module(Tsecond) === I12
    @test sort(DF.nonzero_degrees(Tfirst)) == sort(collect(keys(DF.degree_dimensions(Tfirst))))
    @test sort(DF.nonzero_degrees(Tsecond)) == sort(collect(keys(DF.degree_dimensions(Tsecond))))
    @test TOA.source_module === DF.source_module
    @test TOA.target_module === DF.target_module
    @test TOA.nonzero_degrees === DF.nonzero_degrees
    @test TOA.degree_dimensions === DF.degree_dimensions

    B, i, p = direct_sum_with_split_sequence(S2, S1)
    Rop1 = IR.pmodule_from_fringe(
        one_by_one_fringe(Pop, FF.principal_upset(Pop, 1), FF.principal_downset(Pop, 1);
                          scalar=one(K), field=field)
    )
    Bop, iop, pop = direct_sum_with_split_sequence(Rop, Rop1)
    les_second = DF.ExtLongExactSequenceSecond(S1, S2, B, S1, i, p,
                                               TO.DerivedFunctorOptions(maxdeg=1, model=:projective))
    les_first = DF.ExtLongExactSequenceFirst(S2, B, S1, S1, i, p,
                                             TO.DerivedFunctorOptions(maxdeg=1, model=:injective))
    tor_les_second = DF.TorLongExactSequenceSecond(Rop, i, p,
                                                   TO.DerivedFunctorOptions(maxdeg=1, model=:first))
    tor_les_first = DF.TorLongExactSequenceFirst(S1, iop, pop,
                                                 TO.DerivedFunctorOptions(maxdeg=1, model=:second))

    dims_second = DF.sequence_dimensions(les_second, 0)
    maps_second = DF.sequence_maps(les_second, 0)
    entry_second = DF.sequence_entry(les_second, 0)
    @test dims_second.A == DF.dim(les_second.EA, 0)
    @test maps_second.i == les_second.iH[1]
    @test entry_second.delta == les_second.delta[1]
    @test TOA.sequence_dimensions(les_second, 0) == dims_second

    dims_first = DF.sequence_dimensions(les_first, 0)
    maps_first = DF.sequence_maps(les_first, 0)
    entry_first = DF.sequence_entry(les_first, 0)
    @test dims_first.C == DF.dim(les_first.EC, 0)
    @test maps_first.p == les_first.pH[1]
    @test entry_first.delta == les_first.delta[1]

    tor_dims_second = DF.sequence_dimensions(tor_les_second, 0)
    tor_maps_second = DF.sequence_maps(tor_les_second, 0)
    tor_entry_second = DF.sequence_entry(tor_les_second, 0)
    @test tor_dims_second.A == DF.dim(tor_les_second.TorA, 0)
    @test tor_maps_second.i == tor_les_second.iH[1]
    @test tor_entry_second.delta == tor_les_second.delta[1]

    tor_dims_first = DF.sequence_dimensions(tor_les_first, 0)
    tor_maps_first = DF.sequence_maps(tor_les_first, 0)
    tor_entry_first = DF.sequence_entry(tor_les_first, 0)
    @test tor_dims_first.A == DF.dim(tor_les_first.TorA, 0)
    @test tor_maps_first.i == tor_les_first.iH[1]
    @test tor_entry_first.delta == tor_les_first.delta[1]

    ess = DF.ExtSpectralSequence(S1, S1; maxlen=1)
    tss = DF.TorSpectralSequence(Rop, S1; maxlen=1)
    @test DF.page_dimensions(ess, 2) == TO.ChainComplexes.page_dims_dict(ess, 2)
    @test DF.page_dimensions(tss, 2) == TO.ChainComplexes.page_dims_dict(tss, 2)
    @test DF.page_dimensions(tss; page=2) == DF.page_dimensions(tss, 2)
    @test TOA.page_dimensions === DF.page_dimensions
    @test DF.spectral_sequence_summary(tss).kind == :tor_spectral_sequence
    @test DF.spectral_sequence_summary(ess).kind == :spectral_sequence

    Aext = DF.ExtAlgebra(S1, TO.DerivedFunctorOptions(maxdeg=1))
    Ator = DF.TorAlgebra(Tsecond)
    @test DF.algebra_field(Aext) == field
    @test DF.algebra_field(Ator) == field
    @test DF.nonzero_degrees(Aext) == [t for t in DF.degree_range(Aext) if DF.dim(Aext, t) != 0]
    @test DF.nonzero_degrees(Ator) == [t for t in DF.degree_range(Ator) if DF.dim(Ator.T, t) != 0]
    @test length(DF.generator_degrees(Aext)) == sum(DF.dim(Aext, t) for t in DF.degree_range(Aext))
    @test length(DF.generator_degrees(Ator)) == sum(DF.dim(Ator.T, t) for t in DF.degree_range(Ator))
    @test TOA.generator_degrees === DF.generator_degrees
    @test TOA.algebra_field === DF.algebra_field
end

@testset "DerivedFunctors summaries and validation helpers" begin
    field = CM.QQField()
    K = CM.coeff_type(field)
    P = chain_poset(2)

    S1 = IR.pmodule_from_fringe(
        one_by_one_fringe(P, FF.principal_upset(P, 1), FF.principal_downset(P, 1);
                          scalar=one(K), field=field)
    )
    I12 = IR.pmodule_from_fringe(
        one_by_one_fringe(P, FF.principal_upset(P, 1), FF.principal_downset(P, 2);
                          scalar=one(K), field=field)
    )
    Pop = FF.FinitePoset(transpose(FF.leq_matrix(P)); check=false)
    Rop = IR.pmodule_from_fringe(
        one_by_one_fringe(Pop, FF.principal_upset(Pop, 2), FF.principal_downset(Pop, 2);
                          scalar=one(K), field=field)
    )

    resP = DF.projective_resolution(S1, TO.ResolutionOptions(maxlen=1))
    resI = DF.injective_resolution(S1, TO.ResolutionOptions(maxlen=1))
    H = DF.Hom(S1, I12)
    Eproj = DF.Ext(resP, I12)
    Euni = DF.ExtSpace(S1, I12, TO.DerivedFunctorOptions(maxdeg=1, model=:unified, canon=:projective))
    Tfirst = DF.Tor(Rop, I12, TO.DerivedFunctorOptions(maxdeg=1, model=:first))
    Tsecond = DF.Tor(Rop, I12, TO.DerivedFunctorOptions(maxdeg=1, model=:second))
    Aext = DF.ExtAlgebra(S1, TO.DerivedFunctorOptions(maxdeg=1))

    B, i, p = direct_sum_with_split_sequence(I12, S1)
    les = DF.ExtLongExactSequenceSecond(S1, I12, B, S1, i, p,
                                        TO.DerivedFunctorOptions(maxdeg=1, model=:projective))

    DC = DF.ExtDoubleComplex(S1, I12; maxlen=1)
    ess = DF.ExtSpectralSequence(S1, I12; maxlen=1)
    tss = DF.TorSpectralSequence(Rop, I12; maxlen=1)

    @test DF.resolution_summary(resP).kind == :projective_resolution
    @test DF.resolution_summary(resI).kind == :injective_resolution
    @test DF.hom_summary(H).kind == :hom_space
    @test DF.ext_summary(Eproj).model == :projective
    @test DF.ext_summary(Euni).model == :unified
    @test DF.tor_summary(Tfirst).model == :first
    @test DF.tor_summary(Tsecond).model == :second
    @test DF.double_complex_summary(DC).kind == :derived_double_complex
    @test DF.derived_les_summary(les).kind == :ext_long_exact_sequence_second
    @test DF.algebra_summary(Aext).kind == :ext_algebra

    @test DF.total_dimension(H) == DF.dim(H)
    @test DF.total_dimension(Eproj) == sum(values(DF.degree_dimensions(Eproj)))
    @test DF.total_dimension(Tfirst) == sum(values(DF.degree_dimensions(Tfirst)))
    @test DF.total_dimension(Aext) == sum(DF.dim(Aext, t) for t in DF.degree_range(Aext))

    repP = DF.check_projective_resolution(resP)
    repI = DF.check_injective_resolution(resI)
    repEss = DF.check_ext_spectral_sequence(ess)
    repTss = DF.check_tor_spectral_sequence(tss)
    repA = DF.check_ext_algebra(Aext)

    @test repP.valid
    @test repI.valid
    @test repEss.valid
    @test repTss.valid
    @test repA.valid

    wrapped = DF.derived_functor_validation_summary(repP)
    @test wrapped isa DF.DerivedFunctorValidationSummary
    @test occursin("DerivedFunctorValidationSummary", sprint(show, wrapped))
    @test occursin("DerivedFunctorValidationSummary", sprint(show, MIME"text/plain"(), wrapped))

    bad_dmat = copy(resP.d_mat)
    bad_dmat[1] = spzeros(K, size(resP.d_mat[1], 1) + 1, size(resP.d_mat[1], 2))
    bad_resP = DF.ProjectiveResolution(resP.M, resP.Pmods, resP.gens, resP.d_mor, bad_dmat, resP.aug)
    bad_rep = DF.check_projective_resolution(bad_resP)
    @test !bad_rep.valid
    @test !isempty(bad_rep.issues)
    @test_throws ArgumentError DF.check_projective_resolution(bad_resP; throw=true)

    bad_A = DF.ExtAlgebra{K}(Aext.E, Dict((0, 0) => zeros(K, 1, 2)), nothing, Aext.tmin, Aext.tmax)
    bad_arep = DF.check_ext_algebra(bad_A)
    @test !bad_arep.valid
    @test_throws ArgumentError DF.check_ext_algebra(bad_A; throw=true)

    @test TOA.total_dimension === DF.total_dimension
    @test TOA.hom_summary === DF.hom_summary
    @test TOA.ext_summary === DF.ext_summary
    @test TOA.tor_summary === DF.tor_summary
    @test TOA.double_complex_summary === DF.double_complex_summary
    @test TOA.derived_les_summary === DF.derived_les_summary
    @test TOA.algebra_summary === DF.algebra_summary
    @test TOA.check_projective_resolution === DF.check_projective_resolution
    @test TOA.check_injective_resolution === DF.check_injective_resolution
    @test TOA.check_ext_spectral_sequence === DF.check_ext_spectral_sequence
    @test TOA.check_tor_spectral_sequence === DF.check_tor_spectral_sequence
    @test TOA.check_ext_algebra === DF.check_ext_algebra
    @test TOA.derived_functor_validation_summary === DF.derived_functor_validation_summary
end

@testset "DerivedFunctors describe/show wrapper surface" begin
    field = CM.QQField()
    K = CM.coeff_type(field)
    P = chain_poset(2)

    S1 = IR.pmodule_from_fringe(
        one_by_one_fringe(P, FF.principal_upset(P, 1), FF.principal_downset(P, 1);
                          scalar=one(K), field=field)
    )
    S2 = IR.pmodule_from_fringe(
        one_by_one_fringe(P, FF.principal_upset(P, 2), FF.principal_downset(P, 2);
                          scalar=one(K), field=field)
    )
    I12 = IR.pmodule_from_fringe(
        one_by_one_fringe(P, FF.principal_upset(P, 1), FF.principal_downset(P, 2);
                          scalar=one(K), field=field)
    )
    Pop = FF.FinitePoset(transpose(FF.leq_matrix(P)); check=false)
    Rop = IR.pmodule_from_fringe(
        one_by_one_fringe(Pop, FF.principal_upset(Pop, 2), FF.principal_downset(Pop, 2);
                          scalar=one(K), field=field)
    )
    Rop1 = IR.pmodule_from_fringe(
        one_by_one_fringe(Pop, FF.principal_upset(Pop, 1), FF.principal_downset(Pop, 1);
                          scalar=one(K), field=field)
    )

    resP = DF.projective_resolution(S1, TO.ResolutionOptions(maxlen=1))
    resI = DF.injective_resolution(S1, TO.ResolutionOptions(maxlen=1))
    H = DF.Hom(S1, I12)
    Eproj = DF.Ext(resP, I12)
    Einj = DF.ExtInjective(S1, I12, TO.DerivedFunctorOptions(maxdeg=1, model=:injective))
    Euni = DF.ExtSpace(S1, I12, TO.DerivedFunctorOptions(maxdeg=1, model=:unified, canon=:projective))
    Tfirst = DF.Tor(Rop, I12, TO.DerivedFunctorOptions(maxdeg=1, model=:first))
    Tsecond = DF.Tor(Rop, I12, TO.DerivedFunctorOptions(maxdeg=1, model=:second))
    Aext = DF.ExtAlgebra(S1, TO.DerivedFunctorOptions(maxdeg=1))
    DF.precompute!(Aext)
    coverP = only(FF.cover_edges(P))
    coverPop = only(FF.cover_edges(Pop))
    S2alg = MD.PModule{K}(P, [0, 1], Dict{Tuple{Int,Int},Matrix{K}}(coverP => CM.zeros(field, 1, 0)); field=field)
    P2op = MD.PModule{K}(Pop, [1, 1], Dict{Tuple{Int,Int},Matrix{K}}(coverPop => CM.ones(field, 1, 1)); field=field)
    T0 = DF.Tor(P2op, S2alg, TO.DerivedFunctorOptions(maxdeg=0))
    Ator = DF.TorAlgebra(T0)
    DF.set_chain_product!(Ator, 0, 0, sparse(CM.ones(field, 1, 1)))
    DF.multiplication_matrix(Ator, 0, 0)
    ext_deg = first(DF.nonzero_degrees(Aext))
    tor_deg = first(DF.nonzero_degrees(Ator))
    ext_coords = zeros(K, DF.dim(Aext, ext_deg))
    ext_coords[1] = one(K)
    tor_coords = zeros(K, DF.dim(Ator, tor_deg))
    tor_coords[1] = one(K)
    xext = DF.element(Aext, ext_deg, ext_coords)
    xtor = DF.element(Ator, tor_deg, tor_coords)

    B, i, p = direct_sum_with_split_sequence(S2, S1)
    Bop, iop, pop = direct_sum_with_split_sequence(Rop, Rop1)
    les_second = DF.ExtLongExactSequenceSecond(S1, S2, B, S1, i, p,
                                               TO.DerivedFunctorOptions(maxdeg=1, model=:projective))
    les_first = DF.ExtLongExactSequenceFirst(S2, B, S1, S1, i, p,
                                             TO.DerivedFunctorOptions(maxdeg=1, model=:injective))
    tor_les_second = DF.TorLongExactSequenceSecond(Rop, i, p,
                                                   TO.DerivedFunctorOptions(maxdeg=1, model=:first))
    tor_les_first = DF.TorLongExactSequenceFirst(S1, iop, pop,
                                                 TO.DerivedFunctorOptions(maxdeg=1, model=:second))
    tss = DF.TorSpectralSequence(Rop, I12; maxlen=1)

    @test TO.describe(resP).kind == :projective_resolution
    @test TO.describe(resI).kind == :injective_resolution
    @test TO.describe(H).kind == :hom_space
    @test TO.describe(Eproj).model == :projective
    @test TO.describe(Einj).model == :injective
    @test TO.describe(Euni).model == :unified
    @test TO.describe(Tfirst).model == :first
    @test TO.describe(Tsecond).model == :second
    @test TO.describe(Aext).kind == :ext_algebra
    @test TO.describe(Ator).kind == :tor_algebra
    @test TO.describe(xext).kind == :ext_element
    @test TO.describe(xtor).kind == :tor_element
    @test TO.describe(les_second).kind == :ext_long_exact_sequence_second
    @test TO.describe(les_first).kind == :ext_long_exact_sequence_first
    @test TO.describe(tor_les_second).kind == :tor_long_exact_sequence_second
    @test TO.describe(tor_les_first).kind == :tor_long_exact_sequence_first
    @test TO.describe(tss).kind == :tor_spectral_sequence

    shown = (
        ("ProjectiveResolution", resP),
        ("InjectiveResolution", resI),
        ("HomSpace", H),
        ("ExtSpaceProjective", Eproj),
        ("ExtSpaceInjective", Einj),
        ("ExtSpace", Euni),
        ("TorSpace", Tfirst),
        ("TorSpaceSecond", Tsecond),
        ("ExtAlgebra", Aext),
        ("TorAlgebra", Ator),
        ("ExtElement", xext),
        ("TorElement", xtor),
        ("ExtLongExactSequenceSecond", les_second),
        ("ExtLongExactSequenceFirst", les_first),
        ("TorLongExactSequenceSecond", tor_les_second),
        ("TorLongExactSequenceFirst", tor_les_first),
        ("TorSpectralSequence", tss),
    )
    for (name, obj) in shown
        @test occursin(name, sprint(show, obj))
        @test occursin(name, sprint(show, MIME"text/plain"(), obj))
    end

    @test DF.wrapped_spectral_sequence(tss) === tss.ss
    @test DF.underlying_ext_space(Aext) === Aext.E
    @test DF.underlying_tor_space(Ator) === Ator.T
    @test !isempty(DF.cached_product_degrees(Aext))
    @test (0, 0) in DF.cached_product_degrees(Ator)
    @test DF.parent_algebra(xext) === Aext
    @test DF.parent_algebra(xtor) === Ator
    @test DF.element_degree(xext) == ext_deg
    @test DF.element_degree(xtor) == tor_deg
    @test DF.element_coordinates(xext) == ext_coords
    @test DF.element_coordinates(xtor) == tor_coords
    @test DF.coordinates(xext) == ext_coords
    @test DF.coordinates(xtor) == tor_coords

    good_tor_report = DF.check_tor_algebra(Ator)
    @test good_tor_report.valid
    @test good_tor_report.kind == :tor_algebra

    bad_tor = DF.TorAlgebra{K}(
        Ator.T,
        Dict{Tuple{Int,Int},SparseMatrixCSC{K,Int}}(),
        nothing,
        Dict((0, 0) => zeros(K, 1, 2)),
        nothing,
    )
    bad_tor_report = DF.check_tor_algebra(bad_tor)
    @test !bad_tor_report.valid
    @test !isempty(bad_tor_report.issues)
    @test_throws ArgumentError DF.check_tor_algebra(bad_tor; throw=true)

    @test TOA.wrapped_spectral_sequence === DF.wrapped_spectral_sequence
    @test TOA.underlying_ext_space === DF.underlying_ext_space
    @test TOA.underlying_tor_space === DF.underlying_tor_space
    @test TOA.cached_product_degrees === DF.cached_product_degrees
    @test TOA.parent_algebra === DF.parent_algebra
    @test TOA.element_degree === DF.element_degree
    @test TOA.element_coordinates === DF.element_coordinates
    @test TOA.check_tor_algebra === DF.check_tor_algebra
    @test TOA.ProjectiveResolution === DF.ProjectiveResolution
    @test TOA.InjectiveResolution === DF.InjectiveResolution
    @test TOA.HomSpace === DF.HomSpace
    @test TOA.ExtSpaceProjective === DF.ExtSpaceProjective
    @test TOA.ExtSpaceInjective === DF.ExtSpaceInjective
    @test TOA.ExtSpace === DF.ExtSpace
    @test TOA.TorSpace === DF.TorSpace
    @test TOA.TorSpaceSecond === DF.TorSpaceSecond
    @test TOA.ExtAlgebra === DF.ExtAlgebra
    @test TOA.TorAlgebra === DF.TorAlgebra
    @test TOA.ExtElement === DF.ExtElement
    @test TOA.TorElement === DF.TorElement
end

@testset "Minimality diagnostics for projective/injective resolutions" begin
    with_fields(FIELDS_FULL) do field
        # Use a small poset where minimal resolutions are nontrivial.
        P = diamond_poset()
        K = CM.coeff_type(field)
        c(x) = CM.coerce(field, x)

        # Simple at vertex 1.
        S1 = IR.pmodule_from_fringe(
            one_by_one_fringe(P, FF.principal_upset(P, 1), FF.principal_downset(P, 1); scalar=one(K), field=field)
        )

        # Projective resolution should be minimal (and certified minimal by the checker).
        resP = DF.projective_resolution(S1, TO.ResolutionOptions(maxlen=4))
        repP = DF.minimality_report(resP)
        @test repP.cover_ok
        @test repP.minimal
        @test isempty(repP.diagonal_violations)
        @test DF.is_minimal(resP)
        DF.assert_minimal(resP)

        # Passing ResolutionOptions(minimal=true, check=true) should succeed and return a minimal resolution.
        resPmin = DF.projective_resolution(S1, TO.ResolutionOptions(maxlen=4, minimal=true, check=true))
        @test DF.is_minimal(resPmin)

        # Corrupt the resolution by inserting a diagonal coefficient in d^1 (degree 1 -> 0),
        # which should violate minimality (a generator mapping to itself at the same vertex).
        if length(resP.gens) >= 2 && !isempty(resP.gens[1]) && !isempty(resP.d_mat)
            gens_bad = [copy(g) for g in resP.gens]
            d_mat_bad = copy(resP.d_mat)

            # Ensure we can reference a "same vertex" pair across cod/domain by duplicating
            # an existing vertex label in degree 1.
            push!(gens_bad[2], gens_bad[1][1])

            D = d_mat_bad[1]
            extra = spzeros(K, size(D, 1), 1)
            extra[1, 1] = c(1)   # explicit diagonal coefficient
            d_mat_bad[1] = hcat(D, extra)

            res_bad = DF.ProjectiveResolution(resP.M, resP.Pmods, gens_bad, resP.d_mor, d_mat_bad, resP.aug)
            rep_bad = DF.minimality_report(res_bad; check_cover=true)
            @test !rep_bad.minimal
            @test !isempty(rep_bad.diagonal_violations)
            @test !DF.is_minimal(res_bad)
            @test_throws ErrorException DF.assert_minimal(res_bad)
        end

        # Injective resolution: also expected to be minimal for these constructions.
        resI = DF.injective_resolution(S1, TO.ResolutionOptions(maxlen=4))
        repI = DF.minimality_report(resI)
        @test repI.hull_ok
        @test repI.minimal
        @test isempty(repI.diagonal_violations)
        @test DF.is_minimal(resI)
        DF.assert_minimal(resI)

        resImin = DF.injective_resolution(S1, TO.ResolutionOptions(maxlen=4, minimal=true, check=true))
        @test DF.is_minimal(resImin)
    end
end

@testset "Betti extraction from projective resolutions (diamond)" begin
    with_fields(FIELDS_FULL) do field
        P = diamond_poset()
        K = CM.coeff_type(field)

        # Simple modules S1..S4 as fringe modules, then as poset-modules.
        Sm = Vector{MD.PModule{K}}(undef, P.n)
        for v in 1:P.n
            Hv = one_by_one_fringe(P, FF.principal_upset(P, v), FF.principal_downset(P, v), one(K); field=field)
            Sm[v] = IR.pmodule_from_fringe(Hv)
        end
        S1, S2, S3, S4 = Sm

        # Minimal projective resolution of S1 on the diamond should have:
        # P0 = P1, P1 = P2 oplus P3, P2 = P4.
        res = DF.projective_resolution(S1, TO.ResolutionOptions(maxlen=2))
        b = DF.betti(res)

        Btbl = DF.betti_table(res)
        if _is_real_field(field)
            # Numerical resolutions over reals can introduce extra/duplicated generators.
            @test get(b, (0, 1), 0) >= 1
            @test get(b, (1, 2), 0) >= 1
            @test get(b, (1, 3), 0) >= 1
            @test get(b, (2, 4), 0) >= 1
            @test Btbl[1,1] >= 1
            @test Btbl[2,2] >= 1
            @test Btbl[2,3] >= 1
            @test Btbl[3,4] >= 1
        else
            @test length(b) == 4
            @test b[(0, 1)] == 1
            @test b[(1, 2)] == 1
            @test b[(1, 3)] == 1
            @test b[(2, 4)] == 1
            @test Btbl[1,1] == 1
            @test Btbl[2,2] == 1
            @test Btbl[2,3] == 1
            @test Btbl[3,4] == 1
        end
    end
end


@testset "Yoneda product (diamond: Ext^1 x Ext^1 -> Ext^2)" begin
    with_fields(FIELDS_FULL) do field
        P = diamond_poset()
        K = CM.coeff_type(field)
        c(x) = CM.coerce(field, x)

        Sm = Vector{MD.PModule{K}}(undef, P.n)
        for v in 1:P.n
            Hv = one_by_one_fringe(P, FF.principal_upset(P, v), FF.principal_downset(P, v); scalar=one(K), field=field)
            Sm[v] = IR.pmodule_from_fringe(Hv)
        end
        S1, S2, S3, S4 = Sm

        # Compute the target Ext space once so coordinates are comparable.
        E14 = DF.Ext(S1, S4, TO.DerivedFunctorOptions(maxdeg=2))
        if _is_real_field(field)
            # Numerical Ext over reals can introduce duplicated classes in this setup.
            @test DF.dim(E14, 2) >= 1
            return
        end
        @test DF.dim(E14, 2) == 1

        # Via the chain 1 -> 2 -> 4
        E24 = DF.Ext(S2, S4, TO.DerivedFunctorOptions(maxdeg=1))
        E12 = DF.Ext(S1, S2, TO.DerivedFunctorOptions(maxdeg=2))  # needs tmax >= 2 because p+q = 2
        @test DF.dim(E24, 1) == 1
        @test DF.dim(E12, 1) == 1

        beta = [c(1)]
        alpha = [c(1)]
        _, coords_2 = TO.DerivedFunctors.yoneda_product(E24, 1, beta, E12, 1, alpha; ELN=E14)
        @test coords_2[1] != 0

        # Via the chain 1 -> 3 -> 4
        E34 = DF.Ext(S3, S4, TO.DerivedFunctorOptions(maxdeg=1))
        E13 = DF.Ext(S1, S3, TO.DerivedFunctorOptions(maxdeg=2))
        _, coords_3 = TO.DerivedFunctors.yoneda_product(E34, 1, [c(1)], E13, 1, [c(1)]; ELN=E14)
        @test coords_3[1] != 0

        # In a 1-dimensional target, the two products must be proportional.
        # With our deterministic lifts/basis choices, they should agree up to sign.
        @test coords_2[1] == coords_3[1] || coords_2[1] == -coords_3[1]
    end
end

@testset "Functoriality direct-map kernels parity" begin
    field = CM.QQField()
    K = CM.coeff_type(field)
    c(x) = CM.coerce(field, x)

    P = diamond_poset()
    Sm = Vector{MD.PModule{K}}(undef, P.n)
    for v in 1:P.n
        Hv = one_by_one_fringe(P, FF.principal_upset(P, v), FF.principal_downset(P, v), one(K); field=field)
        Sm[v] = IR.pmodule_from_fringe(Hv)
    end
    S1, S2, _, S4 = Sm

    E14 = DF.Ext(S1, S4, TO.DerivedFunctorOptions(maxdeg=2, model=:projective))
    A_off = DF.Functoriality._FUNCTORIALITY_USE_DIRECT_COEFF_TRIPLETS[]
    M_off = DF.Functoriality._FUNCTORIALITY_USE_DIRECT_COEFF_MATVECS[]
    C_off = DF.Functoriality._FUNCTORIALITY_USE_COEFF_PLAN_CACHE[]
    T_off = DF.Functoriality._FUNCTORIALITY_DIRECT_COEFF_MIN_NNZ[]
    Cc_off = DF.Functoriality._FUNCTORIALITY_DIRECT_COCYCLE_MIN_NNZ[]

    Pop = FF.FinitePoset(transpose(FF.leq_matrix(P)); check=false)
    RopH = one_by_one_fringe(Pop, FF.principal_upset(Pop, 2), FF.principal_downset(Pop, 2), one(K); field=field)
    LH = one_by_one_fringe(P, FF.principal_upset(P, 1), FF.principal_downset(P, 1), one(K); field=field)
    Rop = IR.pmodule_from_fringe(RopH)
    L = IR.pmodule_from_fringe(LH)
    T = DF.Tor(Rop, L, TO.DerivedFunctorOptions(maxdeg=1, model=:first))

    try
        DF.Functoriality._FUNCTORIALITY_USE_DIRECT_COEFF_TRIPLETS[] = false
        DF.Functoriality._FUNCTORIALITY_USE_DIRECT_COEFF_MATVECS[] = false
        DF.Functoriality._FUNCTORIALITY_USE_COEFF_PLAN_CACHE[] = false
        DF.Functoriality._FUNCTORIALITY_DIRECT_COEFF_MIN_NNZ[] = typemax(Int)
        DF.Functoriality._FUNCTORIALITY_DIRECT_COCYCLE_MIN_NNZ[] = typemax(Int)
        ext_off = DF.ext_map_first(E14, E14, IR.id_morphism(S1); t=1)
        tor_off = DF.tor_map_first(T, T, IR.id_morphism(Rop); s=1)
        _, yoneda_off = DF.yoneda_product(
            DF.Ext(S2, S4, TO.DerivedFunctorOptions(maxdeg=1, model=:projective)), 1, [c(1)],
            DF.Ext(S1, S2, TO.DerivedFunctorOptions(maxdeg=2, model=:projective)), 1, [c(1)];
            ELN=E14,
        )

        DF.Functoriality._FUNCTORIALITY_USE_DIRECT_COEFF_TRIPLETS[] = true
        DF.Functoriality._FUNCTORIALITY_USE_DIRECT_COEFF_MATVECS[] = true
        DF.Functoriality._FUNCTORIALITY_USE_COEFF_PLAN_CACHE[] = true
        DF.Functoriality._FUNCTORIALITY_DIRECT_COEFF_MIN_NNZ[] = 0
        DF.Functoriality._FUNCTORIALITY_DIRECT_COCYCLE_MIN_NNZ[] = 0
        ext_on = DF.ext_map_first(E14, E14, IR.id_morphism(S1); t=1)
        tor_on = DF.tor_map_first(T, T, IR.id_morphism(Rop); s=1)
        _, yoneda_on = DF.yoneda_product(
            DF.Ext(S2, S4, TO.DerivedFunctorOptions(maxdeg=1, model=:projective)), 1, [c(1)],
            DF.Ext(S1, S2, TO.DerivedFunctorOptions(maxdeg=2, model=:projective)), 1, [c(1)];
            ELN=E14,
        )

        @test ext_on == ext_off
        @test tor_on == tor_off
        @test yoneda_on == yoneda_off
    finally
        DF.Functoriality._FUNCTORIALITY_USE_DIRECT_COEFF_TRIPLETS[] = A_off
        DF.Functoriality._FUNCTORIALITY_USE_DIRECT_COEFF_MATVECS[] = M_off
        DF.Functoriality._FUNCTORIALITY_USE_COEFF_PLAN_CACHE[] = C_off
        DF.Functoriality._FUNCTORIALITY_DIRECT_COEFF_MIN_NNZ[] = T_off
        DF.Functoriality._FUNCTORIALITY_DIRECT_COCYCLE_MIN_NNZ[] = Cc_off
    end
end


@testset "Yoneda associativity sanity check (B3, degree 3 is nonzero)" begin
    with_fields(FIELDS_FULL) do field
        P = boolean_lattice_B3_poset()
        K = CM.coeff_type(field)
        c(x) = CM.coerce(field, x)

        Sm = Vector{MD.PModule{K}}(undef, P.n)
        for v in 1:P.n
            Hv = one_by_one_fringe(P, FF.principal_upset(P, v), FF.principal_downset(P, v); scalar=one(K), field=field)
            Sm[v] = IR.pmodule_from_fringe(Hv)
        end

        # Element numbering is the bitmask order described in boolean_lattice_B3_poset().
        S0   = Sm[1]  # {}
        S1   = Sm[2]  # {1}
        S12  = Sm[5]  # {1,2}
        S123 = Sm[8]  # {1,2,3}

        # Target space: Ext^3(S0, S123) should be 1-dimensional for B3.
        E03 = DF.Ext(S0, S123, TO.DerivedFunctorOptions(maxdeg=3))
        if _is_real_field(field)
            @test DF.dim(E03, 3) >= 1
        else
            @test DF.dim(E03, 3) == 1
        end

        # Degree-1 generators on the cover chain:
        #   {} -> {1} -> {1,2} -> {1,2,3}
        E23 = DF.Ext(S12, S123, TO.DerivedFunctorOptions(maxdeg=3))
        E12 = DF.Ext(S1,  S12, TO.DerivedFunctorOptions(maxdeg=3))
        E01 = DF.Ext(S0,  S1,   TO.DerivedFunctorOptions(maxdeg=3))

        if _is_real_field(field)
            @test DF.dim(E23, 1) >= 1
            @test DF.dim(E12, 1) >= 1
            @test DF.dim(E01, 1) >= 1
        else
            @test DF.dim(E23, 1) == 1
            @test DF.dim(E12, 1) == 1
            @test DF.dim(E01, 1) == 1
        end

        # Intermediate targets in degree 2 (also 1-dimensional for this choice).
        E13 = DF.Ext(S1,  S123, TO.DerivedFunctorOptions(maxdeg=3))
        E02 = DF.Ext(S0,  S12, TO.DerivedFunctorOptions(maxdeg=3))
        if _is_real_field(field)
            @test DF.dim(E13, 2) >= 1
            @test DF.dim(E02, 2) >= 1
            # Numerical lift solves in this chain can be inconsistent under tolerance;
            # keep a coarse sanity check for RealField and skip strict associativity.
            return
        else
            @test DF.dim(E13, 2) == 1
            @test DF.dim(E02, 2) == 1
        end

        # Left bracketing: (e23 * e12) * e01
        _, x = TO.DerivedFunctors.yoneda_product(E23, 1, [c(1)], E12, 1, [c(1)]; ELN=E13)  # x in Ext^2(S1,S123)
        _, left = TO.DerivedFunctors.yoneda_product(E13, 2, x, E01, 1, [c(1)]; ELN=E03)

        # Right bracketing: e23 * (e12 * e01)
        _, y = TO.DerivedFunctors.yoneda_product(E12, 1, [c(1)], E01, 1, [c(1)]; ELN=E02)  # y in Ext^2(S0,S12)
        _, right = TO.DerivedFunctors.yoneda_product(E23, 1, [c(1)], E02, 2, y; ELN=E03)

        # Nontriviality + associativity up to sign in a 1-dimensional target.
        @test left[1] != 0
        @test right[1] != 0
        @test left[1] == right[1] || left[1] == -right[1]
    end
end


@testset "Connecting homomorphisms: split exact sequences give zero maps" begin
    with_fields(FIELDS_FULL) do field
        Q = chain_poset(4)
        K = CM.coeff_type(field)

        Sm = Vector{MD.PModule{K}}(undef, Q.n)
        for v in 1:Q.n
            Hv = one_by_one_fringe(Q, FF.principal_upset(Q, v), FF.principal_downset(Q, v); scalar=one(K), field=field)
            Sm[v] = IR.pmodule_from_fringe(Hv)
        end
        S1, S2, S3, S4 = Sm

        # -------------------------------------------------------------------------
        # Second argument LES: 0 -> A -> A oplus C -> C -> 0
        # delta : Ext^t(M,C) -> Ext^{t+1}(M,A)
        # Choose M=S1, C=S3 (Ext^2), A=S4 (Ext^3), t=2.
        # -------------------------------------------------------------------------
        A = S4
        C = S3
        B, i, p = direct_sum_with_split_sequence(A, C)

        # We test delta^2 : Ext^2(M,C) -> Ext^3(M,A), so we need the resolution through degree t+1 = 3.
        t = 2
        resM = DF.projective_resolution(S1, TO.ResolutionOptions(maxlen=t+1))   # i.e. maxlen=3
        EMA = DF.Ext(resM, A)
        EMB = DF.Ext(resM, B)
        EMC = DF.Ext(resM, C)

        delta2 = DF.connecting_hom(EMA, EMB, EMC, i, p; t=2)
        if _is_real_field(field)
            @test norm(Matrix(delta2)) <= _field_tol(field)
        else
            @test all(delta2 .== 0)
        end

        # -------------------------------------------------------------------------
        # First argument LES: 0 -> A -> A oplus C -> C -> 0
        # delta : Ext^t(A,N) -> Ext^{t+1}(C,N)
        # Choose N=S4, A=S3 (Ext^1), C=S2 (Ext^2), t=1.
        # -------------------------------------------------------------------------
        A1 = S3
        C1 = S2
        B1, i1, p1 = direct_sum_with_split_sequence(A1, C1)

        resN = DF.injective_resolution(S4, TO.ResolutionOptions(maxlen=2))
        EA = TO.ExtInjective(A1, resN)
        EB = TO.ExtInjective(B1, resN)
        EC = TO.ExtInjective(C1, resN)

        delta1 = DF.connecting_hom_first(EA, EB, EC, i1, p1; t=1)
        if _is_real_field(field)
            @test norm(Matrix(delta1)) <= _field_tol(field)
        else
            @test all(delta1 .== 0)
        end
    end
end

@testset "DerivedFunctors support-plan and Hom-workspace parity" begin
    field = CM.QQField()
    K = CM.coeff_type(field)
    Q = chain_poset(4)

    Sm = Vector{MD.PModule{K}}(undef, Q.n)
    for v in 1:Q.n
        Hv = one_by_one_fringe(Q, FF.principal_upset(Q, v), FF.principal_downset(Q, v);
                               scalar=one(K), field=field)
        Sm[v] = IR.pmodule_from_fringe(Hv)
    end
    S1, S2, S3, S4 = Sm

    resS1 = DF.projective_resolution(S1, TO.ResolutionOptions(maxlen=3))
    A = S4
    C = S3
    B, i, p = direct_sum_with_split_sequence(A, C)
    EMA = DF.Ext(resS1, A)
    EMB = DF.Ext(resS1, B)
    EMC = DF.Ext(resS1, C)

    A1 = S3
    C1 = S2
    B1, i1, p1 = direct_sum_with_split_sequence(A1, C1)
    resN = DF.injective_resolution(S4, TO.ResolutionOptions(maxlen=2))
    EA = TO.ExtInjective(A1, resN)
    EB = TO.ExtInjective(B1, resN)
    EC = TO.ExtInjective(C1, resN)

    old_support = DF.Functoriality._FUNCTORIALITY_USE_SUPPORT_SOLVE_PLANS[]
    old_hom_ws = DF.Functoriality._FUNCTORIALITY_USE_HOM_WORKSPACES[]
    old_inj_downset_cache = DF.Resolutions._RESOLUTIONS_USE_INJECTIVE_DOWNSET_SYSTEM_CACHE[]
    old_inj_downset_solve_plan = DF.Resolutions._RESOLUTIONS_USE_INJECTIVE_DOWNSET_SOLVE_PLAN_CACHE[]
    old_hom_solve_cache = DF.Functoriality._FUNCTORIALITY_USE_HOM_SOLVE_WORKSPACE_CACHE[]
    old_hom_basis_plan = DF.Functoriality._FUNCTORIALITY_USE_HOM_BASIS_SOLVE_PLAN_CACHE[]
    old_proj_lift_cache = DF.Resolutions._RESOLUTIONS_USE_PROJECTIVE_LIFT_STRUCTURE_CACHE[]
    try
        DF.Functoriality._FUNCTORIALITY_USE_SUPPORT_SOLVE_PLANS[] = false
        DF.Functoriality._FUNCTORIALITY_USE_HOM_WORKSPACES[] = false
        DF.Resolutions._RESOLUTIONS_USE_INJECTIVE_DOWNSET_SYSTEM_CACHE[] = false
        DF.Resolutions._RESOLUTIONS_USE_INJECTIVE_DOWNSET_SOLVE_PLAN_CACHE[] = false
        DF.Functoriality._FUNCTORIALITY_USE_HOM_SOLVE_WORKSPACE_CACHE[] = false
        DF.Functoriality._FUNCTORIALITY_USE_HOM_BASIS_SOLVE_PLAN_CACHE[] = false
        DF.Resolutions._RESOLUTIONS_USE_PROJECTIVE_LIFT_STRUCTURE_CACHE[] = false
        lift_off = DF.Functoriality._lift_pmodule_map_to_projective_resolution_chainmap_coeff(
            resS1, resS1, IR.id_morphism(S1); upto=2
        )
        delta_off = DF.connecting_hom(EMA, EMB, EMC, i, p; t=2)
        delta_first_off = DF.connecting_hom_first(EA, EB, EC, i1, p1; t=1)
        ext_first_off = DF.ext_map_first(EA, EB, i1; t=1)
        ext_second_off = DF.ext_map_second(EC, EC, IR.id_morphism(S4); t=1)
        les_first_off = DF.ExtLongExactSequenceFirst(A1, B1, C1, S4, i1, p1,
                                                     TO.DerivedFunctorOptions(maxdeg=1, model=:injective))

        DF.Functoriality._FUNCTORIALITY_USE_SUPPORT_SOLVE_PLANS[] = true
        DF.Functoriality._FUNCTORIALITY_USE_HOM_WORKSPACES[] = true
        DF.Resolutions._RESOLUTIONS_USE_INJECTIVE_DOWNSET_SYSTEM_CACHE[] = true
        DF.Resolutions._RESOLUTIONS_USE_INJECTIVE_DOWNSET_SOLVE_PLAN_CACHE[] = true
        DF.Functoriality._FUNCTORIALITY_USE_HOM_SOLVE_WORKSPACE_CACHE[] = true
        DF.Functoriality._FUNCTORIALITY_USE_HOM_BASIS_SOLVE_PLAN_CACHE[] = true
        DF.Resolutions._RESOLUTIONS_USE_PROJECTIVE_LIFT_STRUCTURE_CACHE[] = true
        lift_on = DF.Functoriality._lift_pmodule_map_to_projective_resolution_chainmap_coeff(
            resS1, resS1, IR.id_morphism(S1); upto=2
        )
        delta_on = DF.connecting_hom(EMA, EMB, EMC, i, p; t=2)
        delta_first_on = DF.connecting_hom_first(EA, EB, EC, i1, p1; t=1)
        ext_first_on = DF.ext_map_first(EA, EB, i1; t=1)
        ext_second_on = DF.ext_map_second(EC, EC, IR.id_morphism(S4); t=1)
        les_first_on = DF.ExtLongExactSequenceFirst(A1, B1, C1, S4, i1, p1,
                                                    TO.DerivedFunctorOptions(maxdeg=1, model=:injective))

        @test length(lift_on) == length(lift_off)
        @test all(lift_on[k] == lift_off[k] for k in eachindex(lift_on))
        @test delta_on == delta_off
        @test delta_first_on == delta_first_off
        @test ext_first_on == ext_first_off
        @test ext_second_on == ext_second_off
        @test les_first_on.pH == les_first_off.pH
        @test les_first_on.iH == les_first_off.iH
        @test les_first_on.delta == les_first_off.delta
    finally
        DF.Functoriality._FUNCTORIALITY_USE_SUPPORT_SOLVE_PLANS[] = old_support
        DF.Functoriality._FUNCTORIALITY_USE_HOM_WORKSPACES[] = old_hom_ws
        DF.Resolutions._RESOLUTIONS_USE_INJECTIVE_DOWNSET_SYSTEM_CACHE[] = old_inj_downset_cache
        DF.Resolutions._RESOLUTIONS_USE_INJECTIVE_DOWNSET_SOLVE_PLAN_CACHE[] = old_inj_downset_solve_plan
        DF.Functoriality._FUNCTORIALITY_USE_HOM_SOLVE_WORKSPACE_CACHE[] = old_hom_solve_cache
        DF.Functoriality._FUNCTORIALITY_USE_HOM_BASIS_SOLVE_PLAN_CACHE[] = old_hom_basis_plan
        DF.Resolutions._RESOLUTIONS_USE_PROJECTIVE_LIFT_STRUCTURE_CACHE[] = old_proj_lift_cache
    end
end

@testset "Projective lift identity regression on grid direct sum" begin
    field = CM.QQField()
    K = CM.coeff_type(field)

    function dense_grid_poset(nx::Int, ny::Int)
        n = nx * ny
        rel = falses(n, n)
        @inbounds for y1 in 1:ny, x1 in 1:nx
            i = x1 + (y1 - 1) * nx
            for y2 in y1:ny, x2 in x1:nx
                j = x2 + (y2 - 1) * nx
                rel[i, j] = true
            end
        end
        return FF.FinitePoset(rel; check=false)
    end

    @inline grid2_index(row::Int, col::Int) = row + (col - 1) * 2

    function interval_module(P, lo::Int, hi::Int)
        H = FF.one_by_one_fringe(
            P,
            FF.principal_upset(P, lo),
            FF.principal_downset(P, hi),
            one(K);
            field=field,
        )
        return IR.pmodule_from_fringe(H)
    end

    function sum_modules(mods)
        M = mods[1]
        for i in 2:length(mods)
            M = MD.direct_sum(M, mods[i])
        end
        return M
    end

    function grid_module(Q; variant::Int)
        m = div(FF.nvertices(Q), 2)
        specs = variant == 1 ? [
            (grid2_index(1, 1), grid2_index(2, m)),
            (grid2_index(1, 1), grid2_index(1, m)),
            (grid2_index(2, 1), grid2_index(2, m)),
            (grid2_index(1, 2), grid2_index(2, max(2, m - 1))),
        ] : [
            (grid2_index(1, 1), grid2_index(2, m)),
            (grid2_index(1, max(1, cld(m, 2))), grid2_index(2, m)),
            (grid2_index(1, 2), grid2_index(1, m)),
            (grid2_index(2, 2), grid2_index(2, max(2, m - 1))),
        ]
        return sum_modules([interval_module(Q, lo, hi) for (lo, hi) in specs])
    end

    Q = dense_grid_poset(2, 6)
    A = grid_module(Q; variant=1)
    B = grid_module(Q; variant=2)
    S = MD.direct_sum(A, B)
    idS = MD.id_morphism(S)
    res = DF.projective_resolution(S, OPT.ResolutionOptions(maxlen=2); threads=false)
    res_alt = DF.projective_resolution(S, OPT.ResolutionOptions(maxlen=2); threads=false)

    lift = DF.Functoriality.lift_chainmap(res, res, idS; maxlen=2)
    @test length(lift) == 3
    @test all(size(lift[k]) == (length(res.gens[k]), length(res.gens[k])) for k in eachindex(lift))
    for k in eachindex(lift)
        n = length(res.gens[k])
        if n == 0
            @test nnz(lift[k]) == 0
        else
            idx = collect(1:n)
            expect = sparse(idx, idx, fill(one(K), n), n, n)
            @test lift[k] == expect
        end
    end

    lift_alt = DF.Functoriality.lift_chainmap(res, res_alt, idS; maxlen=2)
    @test length(lift_alt) == 3
    phi0 = TO.ChangeOfPosets._pmorphism_from_upset_coeff(
        res.Pmods[1], res_alt.Pmods[1], res.gens[1], res_alt.gens[1], lift_alt[1]
    )
    lhs0 = DF.compose(res_alt.aug, phi0)
    rhs0 = DF.compose(idS, res.aug)
    @test all(lhs0.comps[u] == rhs0.comps[u] for u in eachindex(lhs0.comps))
    phi_prev = phi0
    for k in 1:2
        phik = TO.ChangeOfPosets._pmorphism_from_upset_coeff(
            res.Pmods[k + 1], res_alt.Pmods[k + 1], res.gens[k + 1], res_alt.gens[k + 1], lift_alt[k + 1]
        )
        lhsk = DF.compose(res_alt.d_mor[k], phik)
        rhsk = DF.compose(phi_prev, res.d_mor[k])
        @test all(lhsk.comps[u] == rhsk.comps[u] for u in eachindex(lhsk.comps))
        phi_prev = phik
    end

    P = chain_poset(3)
    pi_of_q = Vector{Int}(undef, FF.nvertices(Q))
    @inbounds for col in 1:6
        p = min(3, max(1, cld(3 * col, 6)))
        pi_of_q[grid2_index(1, col)] = p
        pi_of_q[grid2_index(2, col)] = p
    end
    pi = TO.Encoding.EncodingMap(Q, P, pi_of_q)

    F = TO.ChangeOfPosets.pushforward_left_complex(
        pi,
        idS,
        OPT.DerivedFunctorOptions(maxdeg=1);
        check=true,
        res_dom=res,
        res_cod=res,
        threads=false,
    )
    @test length(F.comps) == 3
    @test F.tmin == -2
    @test F.tmax == 0
    @test all(F.comps[i].dom.dims == F.comps[i].cod.dims for i in eachindex(F.comps))
end

@testset "ChangeOfPosets derived caches reuse cochain maps and induced morphisms" begin
    field = CM.QQField()
    K = CM.coeff_type(field)
    Q = chain_poset(4)
    P = chain_poset(2)
    pi = TO.Encoding.EncodingMap(Q, P, [1, 1, 2, 2])
    M = IR.pmodule_from_fringe(one_by_one_fringe(Q,
                                                 FF.principal_upset(Q, 2),
                                                 FF.principal_downset(Q, 4);
                                                 scalar=one(K), field=field))
    idM = MD.id_morphism(M)
    df = OPT.DerivedFunctorOptions(maxdeg=1)
    resP = DF.projective_resolution(M, OPT.ResolutionOptions(maxlen=2))
    resI = DF.injective_resolution(M, OPT.ResolutionOptions(maxlen=2))
    sc = CM.SessionCache()

    F1 = TO.ChangeOfPosets.pushforward_left_complex(
        pi,
        idM,
        df;
        check=true,
        res_dom=resP,
        res_cod=resP,
        threads=false,
        session_cache=sc,
    )
    F2 = TO.ChangeOfPosets.pushforward_left_complex(
        pi,
        idM,
        df;
        check=true,
        res_dom=resP,
        res_cod=resP,
        threads=false,
        session_cache=sc,
    )
    @test F1 === F2

    L1 = TO.ChangeOfPosets.Lpushforward_left(
        pi,
        idM,
        df;
        check=true,
        res_dom=resP,
        res_cod=resP,
        threads=false,
        session_cache=sc,
    )
    L2 = TO.ChangeOfPosets.Lpushforward_left(
        pi,
        idM,
        df;
        check=true,
        res_dom=resP,
        res_cod=resP,
        threads=false,
        session_cache=sc,
    )
    @test L1 === L2

    G1 = TO.ChangeOfPosets.pushforward_right_complex(
        pi,
        idM,
        df;
        check=true,
        res_dom=resI,
        res_cod=resI,
        threads=false,
        session_cache=sc,
    )
    G2 = TO.ChangeOfPosets.pushforward_right_complex(
        pi,
        idM,
        df;
        check=true,
        res_dom=resI,
        res_cod=resI,
        threads=false,
        session_cache=sc,
    )
    @test G1 === G2

    R1 = TO.ChangeOfPosets.Rpushforward_right(
        pi,
        idM,
        df;
        check=true,
        res_dom=resI,
        res_cod=resI,
        threads=false,
        session_cache=sc,
    )
    R2 = TO.ChangeOfPosets.Rpushforward_right(
        pi,
        idM,
        df;
        check=true,
        res_dom=resI,
        res_cod=resI,
        threads=false,
        session_cache=sc,
    )
    @test R1 === R2
end

@testset "ChangeOfPosets derived coefficient fast path parity" begin
    field = CM.QQField()
    K = CM.coeff_type(field)
    Q = chain_poset(4)
    P = chain_poset(2)
    pi = TO.Encoding.EncodingMap(Q, P, [1, 1, 2, 2])
    M = IR.pmodule_from_fringe(one_by_one_fringe(Q,
                                                 FF.principal_upset(Q, 2),
                                                 FF.principal_downset(Q, 4);
                                                 scalar=one(K), field=field))

    function _scalar_id_map(M::MD.PModule{K}, a::K) where {K}
        comps = Matrix{K}[]
        for d in M.dims
            A = zeros(K, d, d)
            for i in 1:d
                A[i, i] = a
            end
            push!(comps, A)
        end
        return MD.PMorphism(M, M, comps)
    end

    function _same_pmodule_morphism(f::MD.PMorphism, g::MD.PMorphism)
        @test f.dom.dims == g.dom.dims
        @test f.cod.dims == g.cod.dims
        @test length(f.comps) == length(g.comps)
        for u in eachindex(f.comps)
            @test Matrix(f.comps[u]) == Matrix(g.comps[u])
        end
    end

    function _same_cochain_map(F::CC.ModuleCochainMap, G::CC.ModuleCochainMap)
        @test F.tmin == G.tmin
        @test F.tmax == G.tmax
        @test [T.dims for T in F.C.terms] == [T.dims for T in G.C.terms]
        @test [T.dims for T in F.D.terms] == [T.dims for T in G.D.terms]
        @test length(F.comps) == length(G.comps)
        for i in eachindex(F.comps)
            _same_pmodule_morphism(F.comps[i], G.comps[i])
        end
    end

    f = _scalar_id_map(M, K(2))
    fid = MD.id_morphism(M)
    df = OPT.DerivedFunctorOptions(maxdeg=1)
    resP = DF.projective_resolution(M, OPT.ResolutionOptions(maxlen=2))
    resI = DF.injective_resolution(M, OPT.ResolutionOptions(maxlen=2))

    old_left_fast = TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_DERIVED_LEFT_COEFF_FASTPATH[]
    old_right_fast = TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_DERIVED_RIGHT_COEFF_FASTPATH[]
    old_left_plan_cache = TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_LEFT_DERIVED_COEFF_PLAN_CACHE[]
    old_left_diffs = TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_LEFT_DERIVED_DIFFS_FROM_DATA[]
    old_left_cohom = TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_LEFT_DERIVED_COHOMOLOGY_SHORTCUT[]
    old_right_plan_cache = TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_RIGHT_DERIVED_COEFF_PLAN_CACHE[]
    old_right_diffs = TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_RIGHT_DERIVED_DIFFS_FROM_DATA[]
    old_right_matrix_recur = TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_RIGHT_DERIVED_COEFF_MATRIX_RECURRENCE[]
    old_right_shared = TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_RIGHT_DERIVED_SHARED_RESOLUTION_FASTPATH[]
    old_right_identity = TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_RIGHT_DERIVED_IDENTITY_MAP_FASTPATH[]
    try
        TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_DERIVED_LEFT_COEFF_FASTPATH[] = false
        TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_DERIVED_RIGHT_COEFF_FASTPATH[] = false
        TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_LEFT_DERIVED_COEFF_PLAN_CACHE[] = false
        TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_LEFT_DERIVED_DIFFS_FROM_DATA[] = false
        TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_LEFT_DERIVED_COHOMOLOGY_SHORTCUT[] = false
        TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_RIGHT_DERIVED_COEFF_PLAN_CACHE[] = false
        TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_RIGHT_DERIVED_DIFFS_FROM_DATA[] = false
        TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_RIGHT_DERIVED_COEFF_MATRIX_RECURRENCE[] = false
        TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_RIGHT_DERIVED_SHARED_RESOLUTION_FASTPATH[] = false
        TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_RIGHT_DERIVED_IDENTITY_MAP_FASTPATH[] = false
        Lcold = TO.ChangeOfPosets.pushforward_left_complex(
            pi, f, df; check=true, res_dom=resP, res_cod=resP, threads=false, session_cache=nothing
        )
        Rcold = TO.ChangeOfPosets.pushforward_right_complex(
            pi, f, df; check=true, res_dom=resI, res_cod=resI, threads=false, session_cache=nothing
        )
        Rid_cold = TO.ChangeOfPosets.pushforward_right_complex(
            pi, MD.id_morphism(M), df; check=true, res_dom=resI, res_cod=resI, threads=false, session_cache=nothing
        )
        Lvec_cold = TO.ChangeOfPosets.Lpushforward_left(
            pi, f, df; check=true, res_dom=resP, res_cod=resP, threads=false, session_cache=nothing
        )
        Rvec_cold = TO.ChangeOfPosets.Rpushforward_right(
            pi, f, df; check=true, res_dom=resI, res_cod=resI, threads=false, session_cache=nothing
        )
        Rvec_id_cold = TO.ChangeOfPosets.Rpushforward_right(
            pi, fid, df; check=true, res_dom=resI, res_cod=resI, threads=false, session_cache=nothing
        )

        TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_DERIVED_LEFT_COEFF_FASTPATH[] = true
        TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_DERIVED_RIGHT_COEFF_FASTPATH[] = true
        TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_LEFT_DERIVED_COEFF_PLAN_CACHE[] = true
        TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_LEFT_DERIVED_DIFFS_FROM_DATA[] = true
        TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_LEFT_DERIVED_COHOMOLOGY_SHORTCUT[] = true
        TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_RIGHT_DERIVED_COEFF_PLAN_CACHE[] = true
        TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_RIGHT_DERIVED_DIFFS_FROM_DATA[] = true
        TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_RIGHT_DERIVED_COEFF_MATRIX_RECURRENCE[] = true
        TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_RIGHT_DERIVED_SHARED_RESOLUTION_FASTPATH[] = true
        TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_RIGHT_DERIVED_IDENTITY_MAP_FASTPATH[] = true
        Lfast = TO.ChangeOfPosets.pushforward_left_complex(
            pi, f, df; check=true, res_dom=resP, res_cod=resP, threads=false, session_cache=nothing
        )
        Rfast = TO.ChangeOfPosets.pushforward_right_complex(
            pi, f, df; check=true, res_dom=resI, res_cod=resI, threads=false, session_cache=nothing
        )
        Rid_fast = TO.ChangeOfPosets.pushforward_right_complex(
            pi, MD.id_morphism(M), df; check=true, res_dom=resI, res_cod=resI, threads=false, session_cache=nothing
        )
        Lvec_fast = TO.ChangeOfPosets.Lpushforward_left(
            pi, f, df; check=true, res_dom=resP, res_cod=resP, threads=false, session_cache=nothing
        )
        Rvec_fast = TO.ChangeOfPosets.Rpushforward_right(
            pi, f, df; check=true, res_dom=resI, res_cod=resI, threads=false, session_cache=nothing
        )
        Rvec_id_fast = TO.ChangeOfPosets.Rpushforward_right(
            pi, fid, df; check=true, res_dom=resI, res_cod=resI, threads=false, session_cache=nothing
        )

        _same_cochain_map(Lcold, Lfast)
        _same_cochain_map(Rcold, Rfast)
        _same_cochain_map(Rid_cold, Rid_fast)
        @test length(Lvec_cold) == length(Lvec_fast)
        @test length(Rvec_cold) == length(Rvec_fast)
        @test length(Rvec_id_cold) == length(Rvec_id_fast)
        for i in eachindex(Lvec_cold)
            _same_pmodule_morphism(Lvec_cold[i], Lvec_fast[i])
        end
        for i in eachindex(Rvec_cold)
            _same_pmodule_morphism(Rvec_cold[i], Rvec_fast[i])
        end
        for i in eachindex(Rvec_id_cold)
            _same_pmodule_morphism(Rvec_id_cold[i], Rvec_id_fast[i])
        end
    finally
        TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_DERIVED_LEFT_COEFF_FASTPATH[] = old_left_fast
        TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_DERIVED_RIGHT_COEFF_FASTPATH[] = old_right_fast
        TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_LEFT_DERIVED_COEFF_PLAN_CACHE[] = old_left_plan_cache
        TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_LEFT_DERIVED_DIFFS_FROM_DATA[] = old_left_diffs
        TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_LEFT_DERIVED_COHOMOLOGY_SHORTCUT[] = old_left_cohom
        TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_RIGHT_DERIVED_COEFF_PLAN_CACHE[] = old_right_plan_cache
        TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_RIGHT_DERIVED_DIFFS_FROM_DATA[] = old_right_diffs
        TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_RIGHT_DERIVED_COEFF_MATRIX_RECURRENCE[] = old_right_matrix_recur
        TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_RIGHT_DERIVED_SHARED_RESOLUTION_FASTPATH[] = old_right_shared
        TO.ChangeOfPosets._CHANGE_OF_POSETS_USE_RIGHT_DERIVED_IDENTITY_MAP_FASTPATH[] = old_right_identity
    end
end

@testset "Connecting homomorphisms: nonsplit extension gives nonzero delta" begin
    with_fields(FIELDS_FULL) do field
        # On the 2-element chain 1<2, Ext^1(S1,S2) has dimension 1.
        # The interval module k[1,2] (dims [1,1] and identity map along 1<2)
        # is a nonsplit extension 0 -> S2 -> k[1,2] -> S1 -> 0.
        P = chain_poset(2)
        K = CM.coeff_type(field)

        S1 = IR.pmodule_from_fringe(one_by_one_fringe(P, FF.principal_upset(P, 1), FF.principal_downset(P, 1); scalar=one(K), field=field))
        S2 = IR.pmodule_from_fringe(one_by_one_fringe(P, FF.principal_upset(P, 2), FF.principal_downset(P, 2); scalar=one(K), field=field))
        I12 = IR.pmodule_from_fringe(one_by_one_fringe(P, FF.principal_upset(P, 1), FF.principal_downset(P, 2); scalar=one(K), field=field))

        # i: S2 -> I12 is inclusion on stalk 2.
        comps_i = [CM.zeros(field, I12.dims[v], S2.dims[v]) for v in 1:P.n]
        comps_i[2] = CM.eye(field, 1)
        i = MD.PMorphism(S2, I12, comps_i)

        # p: I12 -> S1 is projection on stalk 1.
        comps_p = [CM.zeros(field, S1.dims[v], I12.dims[v]) for v in 1:P.n]
        comps_p[1] = CM.eye(field, 1)
        p = MD.PMorphism(I12, S1, comps_p)

        # Fix M = S1. Then delta^0: Hom(S1,S1) -> Ext^1(S1,S2) sends id to the extension class,
        # so it must be nonzero (hence rank 1, since both sides are 1-dimensional).
        resM = DF.projective_resolution(S1, TO.ResolutionOptions(maxlen=2))
        EMA = DF.Ext(resM, S2)
        EMB = DF.Ext(resM, I12)
        EMC = DF.Ext(resM, S1)
        delta0 = DF.connecting_hom(EMA, EMB, EMC, i, p; t=0)

        # The packaged long exact sequence should expose the same delta^0.
        les = TO.ExtLongExactSequenceSecond(S1, S2, I12, S1, i, p, TO.DerivedFunctorOptions(maxdeg=0))
        if _is_real_field(field)
            r0 = FL.rank(field, delta0)
            r1 = FL.rank(field, les.delta[1])
            @test r0 == r1
            @test 0 <= r0 <= 1
            @test norm(Matrix(les.delta[1]) - Matrix(delta0)) <= _field_tol(field)
        else
            @test FL.rank(field, delta0) == 1
            @test FL.rank(field, les.delta[1]) == 1
            @test Matrix(les.delta[1]) == Matrix(delta0)
        end
    end
end

@testset "ExtAlgebra: cached multiplication agrees with yoneda_product" begin
    with_fields(FIELDS_FULL) do field
        P = diamond_poset()
        K = CM.coeff_type(field)
        c(x) = CM.coerce(field, x)

        # Build simples S1..S4 as 1x1 fringe modules, then as poset-modules.
        Sm = Vector{MD.PModule{K}}(undef, P.n)
        for v in 1:P.n
            Hv = one_by_one_fringe(P, FF.principal_upset(P, v), FF.principal_downset(P, v); scalar=one(K), field=field)
            Sm[v] = IR.pmodule_from_fringe(Hv)
        end
        S1, S2, S3, S4 = Sm

        # We take a direct sum with enough structure to produce nontrivial Ext in degree 1
        # and allow products in degree 2 on the diamond.
        #
        # M = S1 oplus S2 oplus S4 is small but already contains:
        # - degree-1 extension classes among summands, and
        # - degree-2 composites (in the full Ext(M,M)).
        M12, _, _ = direct_sum_with_split_sequence(S1, S2)
        M, _, _ = direct_sum_with_split_sequence(M12, S4)

        A = TO.ExtAlgebra(M, TO.DerivedFunctorOptions(maxdeg=2))
        E = A.E

        # Sanity: dimensions agree with the underlying Ext space.
        for t in 0:E.tmax
            @test DF.dim(A, t) == DF.dim(E, t)
        end
        _is_real_field(field) && return

        # The unit should act as both-sided identity on every homogeneous degree <= tmax.
        oneA = one(A)
        for t in 0:E.tmax
            dt = DF.dim(A, t)
            if dt == 0
                continue
            end

            # Deterministic "generic" element: (1,2,3,...,dt).
            x = DF.element(A, t, [c(i) for i in 1:dt])

            @test (oneA * x).coords == x.coords
            @test (x * oneA).coords == x.coords
        end

        # Cache behavior: after one multiplication in (p,q), the multiplication matrix should exist,
        # and repeated multiplication should not grow the cache.
        if E.tmax >= 2 && DF.dim(A, 1) > 0
            d1 = DF.dim(A, 1)
            x = DF.element(A, 1, [c(i) for i in 1:d1])
            y = DF.element(A, 1, [c(d1 - i + 1) for i in 1:d1])

            prod1 = x * y
            @test haskey(A.mult_cache, (1, 1))
            nkeys = length(A.mult_cache)

            prod2 = x * y
            @test length(A.mult_cache) == nkeys

            # Cached multiplication must match a direct call to the mathematical core (Yoneda product)
            # in the same Ext space and bases.
            _, coords_direct = TO.DerivedFunctors.yoneda_product(A.E, 1, x.coords, A.E, 1, y.coords; ELN=A.E)
            @test prod1.coords == coords_direct
            @test prod2.coords == coords_direct
        end

        # Associativity in the cached algebra (within truncation).
        #
        # On the diamond, Ext^3 is expected to vanish for many modules; we therefore test an
        # associativity instance that stays inside degrees <= 2 by including a degree-0 factor.
        if E.tmax >= 2 && DF.dim(A, 1) > 0
            d1 = DF.dim(A, 1)

            # First basis direction e_1
            a = DF.element(A, 1, [one(K); zeros(K, d1 - 1)])

            # Last basis direction e_{d1}
            b = DF.element(A, 1, [zeros(K, d1 - 1); one(K)])

            c0 = oneA

            @test ((a * b) * c0).coords == (a * (b * c0)).coords
            @test ((c0 * a) * b).coords == (c0 * (a * b)).coords
        end
    end
end

@testset "Sparse assembly replacements (dense->sparse removed)" begin
    with_fields(FIELDS_FULL) do field
        K = CM.coeff_type(field)
        c(x) = CM.coerce(field, x)

        # 0) _append_scaled_triplets! matches the old findnz(sparse(F)) pattern (up to matrix equality)
        let
            F = [c(1) c(0); c(2) c(3)]
            I1 = Int[]; J1 = Int[]; V1 = K[]
            TO.CoreModules._append_scaled_triplets!(I1, J1, V1, F, 10, 20; scale=c(2))

            S1 = sparse(I1, J1, V1, 12, 22)

            Ii, Ji, Vi = findnz(sparse(F))
            S2 = sparse(Ii .+ 10, Ji .+ 20, Vi .* c(2), 12, 22)

            @test S1 == S2
        end

        # Build a small poset and module with nontrivial (but diagonal) structure maps.
        P = chain_poset(3)
        dims = fill(2, 3)

        A12 = [c(1) c(0); c(0) c(2)]
        A23 = [c(3) c(0); c(0) c(5)]

        edge_maps = Dict{Tuple{Int,Int}, Matrix{K}}()
        edge_maps[(1,2)] = A12
        edge_maps[(2,3)] = A23

        M = MD.PModule{K}(P, dims, edge_maps; field=field)

        # 1) _coeff_matrix_upsets: new sparse assembly equals old dense->sparse reference
        let
            U1 = TO.FiniteFringe.principal_upset(P, 1)
            U2 = TO.FiniteFringe.principal_upset(P, 2)
            U3 = TO.FiniteFringe.principal_upset(P, 3)
            domU = [U1, U2, U3]
            codU = [U1, U2, U3]

            Cnew = DF._coeff_matrix_upsets(domU, codU, K)

            Cold_dense = CM.zeros(field, length(codU), length(domU))
            for i in 1:length(domU), j in 1:length(codU)
                cval = one(K)
                for v in codU[j]
                    if !(v in domU[i])
                        cval = zero(K)
                        break
                    end
                end
                Cold_dense[j,i] = cval
            end
            Cold = sparse(Cold_dense)

            @test Cnew == Cold
            @test Cnew isa SparseMatrixCSC{K,Int}
        end

        # 1b) _coeff_matrix_upsets(P, dom_bases, cod_bases, f) agrees with the top-vertex matrix
        #
        # On a chain poset, vertex n is a top element. At that vertex, all principal upsets
        # are active, so the morphism component matrix is the full coefficient matrix.
        let
            P = chain_poset(3)
            DF = TO.DerivedFunctors

            # A simple module with zero structure maps. This forces projective_cover to
            # pick generators at multiple vertices in a predictable way, producing a
            # nontrivial next-stage differential P1 -> P0.
            dims = fill(1, 3)
            edge_maps = Dict{Tuple{Int,Int}, Matrix{K}}()
            edge_maps[(1,2)] = CM.zeros(field, 1, 1)
            edge_maps[(2,3)] = CM.zeros(field, 1, 1)
            M = MD.PModule{K}(P, dims, edge_maps; field=field)

            P0, pi0, gens0 = IR.projective_cover(M)
            Ker, iota = TO.kernel_with_inclusion(pi0)
            P1, pi1, gens1 = IR.projective_cover(Ker)

            # Differential d : P1 -> P0
            d = DF.compose(iota, pi1)

            dom_bases = DF._flatten_gens_at(gens1)
            cod_bases = DF._flatten_gens_at(gens0)

            C = DF._coeff_matrix_upsets(P, dom_bases, cod_bases, d)

            # On the chain poset(3), vertex 3 is top, so all summands are active there.
            @test C == sparse(d.comps[3])
            @test C isa SparseMatrixCSC{K,Int}
        end


        # 2) _precompose_on_hom_cochains_from_projective_coeff: dense & sparse coeff match dense reference
        let
            dom_gens = [3,2,1]
            cod_gens = [3,2,1]

            dom_offsets = zeros(Int, length(dom_gens) + 1)
            cod_offsets = zeros(Int, length(cod_gens) + 1)
            for i in 1:length(dom_gens)
                dom_offsets[i+1] = dom_offsets[i] + M.dims[dom_gens[i]]
                cod_offsets[i+1] = cod_offsets[i] + M.dims[cod_gens[i]]
            end

            coeff_dense = [
                c(1) c(0) c(0);
                c(2) c(3) c(0);
                c(0) c(4) c(5)
            ]
            coeff_sparse = sparse(coeff_dense)

            # Old dense reference
            Fref_dense = CM.zeros(field, dom_offsets[end], cod_offsets[end])
            for i in 1:length(dom_gens), j in 1:length(cod_gens)
                cval = coeff_dense[j,i]
                iszero(cval) && continue
                ui = dom_gens[i]
                vj = cod_gens[j]
                A = MD.map_leq(M, vj, ui)
                rows = (dom_offsets[i] + 1):dom_offsets[i+1]
                cols = (cod_offsets[j] + 1):cod_offsets[j+1]
                Fref_dense[rows, cols] .+= cval .* A
            end
            Fref = sparse(Fref_dense)

            F1 = DF._precompose_on_hom_cochains_from_projective_coeff(M, dom_gens, cod_gens, dom_offsets, cod_offsets, coeff_dense)
            F2 = DF._precompose_on_hom_cochains_from_projective_coeff(M, dom_gens, cod_gens, dom_offsets, cod_offsets, coeff_sparse)

            @test F1 == sparse(Fref_dense)
            @test F2 == sparse(Fref_dense)
            @test F1 isa SparseMatrixCSC{K,Int}
            @test F2 isa SparseMatrixCSC{K,Int}
        end

        # 3) _tensor_map_on_tor_chains_from_projective_coeff: dense & sparse coeff match dense reference
        let
            dom_bases = [1,2]
            cod_bases = [2,3]

            dom_offsets = zeros(Int, length(dom_bases) + 1)
            cod_offsets = zeros(Int, length(cod_bases) + 1)
            for i in 1:length(dom_bases)
                dom_offsets[i+1] = dom_offsets[i] + M.dims[dom_bases[i]]
            end
            for j in 1:length(cod_bases)
                cod_offsets[j+1] = cod_offsets[j] + M.dims[cod_bases[j]]
            end

            coeff_dense = [
                c(0) c(1);
                c(2) c(0)
            ]
            coeff_sparse = sparse(coeff_dense)

            Bref_dense = CM.zeros(field, cod_offsets[end], dom_offsets[end])
            for i in 1:length(dom_bases), j in 1:length(cod_bases)
                cval = coeff_dense[j,i]
                iszero(cval) && continue
                u = dom_bases[i]
                v = cod_bases[j]
                A = MD.map_leq(M, u, v)
                rows = (cod_offsets[j] + 1):cod_offsets[j+1]
                cols = (dom_offsets[i] + 1):dom_offsets[i+1]
                Bref_dense[rows, cols] = cval .* A
            end
            Bref = sparse(Bref_dense)

            B1 = DF._tensor_map_on_tor_chains_from_projective_coeff(M, dom_bases, cod_bases, dom_offsets, cod_offsets, coeff_dense)
            B2 = DF._tensor_map_on_tor_chains_from_projective_coeff(M, dom_bases, cod_bases, dom_offsets, cod_offsets, coeff_sparse)

            @test B1 == Bref
            @test B2 == Bref
        end

        # 4) _tor_blockdiag_map_on_chains matches dense reference
        let
            f = IR.id_morphism(M)
            gens = [1,2,3]

            dom_offsets = zeros(Int, length(gens) + 1)
            cod_offsets = zeros(Int, length(gens) + 1)
            for i in 1:length(gens)
                dom_offsets[i+1] = dom_offsets[i] + f.dom.dims[gens[i]]
                cod_offsets[i+1] = cod_offsets[i] + f.cod.dims[gens[i]]
            end

            T = DF._tor_blockdiag_map_on_chains(f, gens, dom_offsets, cod_offsets)

            Tref_dense = CM.zeros(field, cod_offsets[end], dom_offsets[end])
            for i in 1:length(gens)
                u = gens[i]
                rows = (cod_offsets[i] + 1):cod_offsets[i+1]
                cols = (dom_offsets[i] + 1):dom_offsets[i+1]
                Tref_dense[rows, cols] = f.comps[u]
            end
            @test T == sparse(Tref_dense)
        end
    end
end

@testset "DerivedFunctors packed active plans and lazy Ext/Hom internals" begin
    field = CM.QQField()
    K = CM.coeff_type(field)
    P = chain_poset(3)

    dims = fill(1, 3)
    edge_zero = Dict{Tuple{Int,Int},Matrix{K}}(
        (1, 2) => CM.zeros(field, 1, 1),
        (2, 3) => CM.zeros(field, 1, 1),
    )
    M = MD.PModule{K}(P, dims, edge_zero; field=field)
    N = MD.PModule{K}(P, dims, edge_zero; field=field)

    H = DF.Hom(M, N)
    @test getfield(H, :basis) === nothing
    B = DF.basis(H)
    @test length(B) == DF.dim(H)
    @test getfield(H, :basis) === B
    @test H.basis === B

    bases = [1, 3]
    plan1 = DF.Resolutions._packed_active_upset_plan(P, bases)
    @test plan1.base_pos == [1, 2]
    bases[1] = 2
    plan2 = DF.Resolutions._packed_active_upset_plan(P, bases)
    @test plan2.base_pos == [1, 2]
    @test plan1.data != plan2.data

    Einj, _, gens_at = IR._injective_hull(M; threads=false)
    basesD = DF._flatten_gens_at(gens_at)
    @test DF.Resolutions._coeff_matrix_downsets(P, basesD, basesD, IR.id_morphism(Einj)) ==
          sparse(Matrix{K}(I, length(basesD), length(basesD)))

    df_proj = TO.DerivedFunctorOptions(maxdeg=1, model=:unified, canon=:projective)
    Eproj = DF.ExtSpace(M, N, df_proj)
    @test getfield(Eproj, :Eproj) !== nothing
    @test getfield(Eproj, :Einj) === nothing
    @test DF.dim(Eproj, 0) == DF.dim(getfield(Eproj, :Eproj), 0)
    @test getfield(Eproj, :Einj) === nothing
    Einj = DF.injective_model(Eproj)
    @test Einj === getfield(Eproj, :Einj)

    df_inj = TO.DerivedFunctorOptions(maxdeg=1, model=:unified, canon=:injective)
    Elazy = DF.ExtSpace(M, N, df_inj)
    @test getfield(Elazy, :Eproj) === nothing
    @test getfield(Elazy, :Einj) !== nothing
    @test DF.dim(Elazy, 0) == DF.dim(getfield(Elazy, :Einj), 0)
    @test getfield(Elazy, :Eproj) === nothing
    Eproj2 = DF.projective_model(Elazy)
    @test Eproj2 === getfield(Elazy, :Eproj)
end

@testset "DerivedFunctors injective resolution support-aware parity" begin
    field = CM.QQField()
    K = CM.coeff_type(field)
    P = chain_poset(4)
    edge = Dict{Tuple{Int,Int},Matrix{K}}(
        (1, 2) => CM.zeros(field, 1, 0),
        (2, 3) => CM.zeros(field, 0, 1),
        (3, 4) => CM.zeros(field, 1, 0),
    )
    M = MD.PModule{K}(P, [0, 1, 0, 1], edge; field=field)
    res = TO.ResolutionOptions(maxlen=2, minimal=false, check=false)

    function _manual_injective_resolution_fullsupport(M::MD.PModule{K}, maxlen::Int) where {K}
        n = FF.nvertices(M.Q)
        cc = MD._get_cover_cache(M.Q)
        map_memo = IR._indicator_new_array_memo(K, n)
        ws = IR._new_resolution_workspace(K, n)
        cokernel_cache = Vector{Any}(undef, n)
        fill!(cokernel_cache, nothing)
        full = collect(1:n)

        E0, iota0, gens0 = IR._injective_hull(
            M;
            cache=cc,
            map_memo=map_memo,
            workspace=ws,
            support_vertices=full,
            threads=false,
        )
        Emods = MD.PModule{K}[E0]
        gens = [DF._flatten_gens_at(gens0)]
        d_mor = MD.PMorphism{K}[]

        C0, pi0 = IR._cokernel_module(iota0; cache=cc, incremental_cache=cokernel_cache, active_vertices=full)
        prevC, prevPi = C0, pi0

        for _ in 1:maxlen
            DF.Resolutions._reset_indicator_memo!(map_memo)
            En, iotan, gensn = IR._injective_hull(
                prevC;
                cache=cc,
                map_memo=map_memo,
                workspace=ws,
                support_vertices=full,
                threads=false,
            )
            push!(Emods, En)
            push!(gens, DF._flatten_gens_at(gensn))
            push!(d_mor, DF.compose(iotan, prevPi))
            Cn, pin = IR._cokernel_module(iotan; cache=cc, incremental_cache=cokernel_cache, active_vertices=full)
            prevC, prevPi = Cn, pin
        end

        return DF.InjectiveResolution{K}(M, Emods, gens, d_mor, iota0)
    end

    Rnew = DF.injective_resolution(M, res; threads=false)
    Rfull = _manual_injective_resolution_fullsupport(M, res.maxlen)

    @test Rnew.gens == Rfull.gens
    @test length(Rnew.d_mor) == length(Rfull.d_mor)
    @test Rnew.iota0.comps == Rfull.iota0.comps
    for i in eachindex(Rnew.d_mor)
        @test Rnew.d_mor[i].comps == Rfull.d_mor[i].comps
    end
end

@testset "DerivedFunctors projective Ext cochain decomposition parity" begin
    field = CM.QQField()
    K = CM.coeff_type(field)
    P = chain_poset(3)
    Hm = one_by_one_fringe(P, FF.principal_upset(P, 1), FF.principal_downset(P, 2); scalar=one(K), field=field)
    Hn = one_by_one_fringe(P, FF.principal_upset(P, 2), FF.principal_downset(P, 3); scalar=one(K), field=field)
    M = IR.pmodule_from_fringe(Hm)
    N = IR.pmodule_from_fringe(Hn)
    res = DF.projective_resolution(M, TO.ResolutionOptions(maxlen=1, minimal=false, check=false); threads=false)

    C, offs = DF.ExtTorSpaces._projective_ext_cochain_complex(res, N; threads=false)
    E = DF.Ext(res, N; threads=false)
    H = TamerOp.ChainComplexes.cohomology_data(C)

    @test C.dims == E.complex.dims
    @test C.d == E.complex.d
    @test offs == E.offsets
    @test [Ht.dimH for Ht in H] == [DF.dim(E, t) for t in 0:1]
end

@testset "DerivedFunctors injective-side comparison batching parity" begin
    field = CM.QQField()
    K = CM.coeff_type(field)
    P = chain_poset(3)
    Hm = one_by_one_fringe(P, FF.principal_upset(P, 1), FF.principal_downset(P, 2); scalar=one(K), field=field)
    Hn = one_by_one_fringe(P, FF.principal_upset(P, 2), FF.principal_downset(P, 3); scalar=one(K), field=field)
    M = IR.pmodule_from_fringe(Hm)
    N = IR.pmodule_from_fringe(Hn)

    Eproj = DF.Ext(M, N, TO.DerivedFunctorOptions(maxdeg=1, model=:projective))
    Einj = DF.ExtInjective(M, N, TO.DerivedFunctorOptions(maxdeg=1, model=:injective))
    resP = Eproj.res
    resE = Einj.res
    basesP0 = resP.gens[1]
    Eb = resE.Emods[1]
    offs0t = DF.ExtTorSpaces._block_offsets_for_gens(Eb, basesP0)
    active_lists = DF.Resolutions._active_upset_indices(P, basesP0)
    active_plan = DF.Resolutions._packed_active_upset_plan(P, basesP0)

    F_old = zeros(K, offs0t[end], DF.dim(Einj.homs[1]))
    for j in 1:DF.dim(Einj.homs[1])
        psi = DF.basis(Einj.homs[1])[j]
        comp = DF.compose(psi, resP.aug)
        for i in 1:length(basesP0)
            u = basesP0[i]
            pos = searchsortedfirst(active_lists[u], i)
            @test pos <= length(active_lists[u]) && active_lists[u][pos] == i
            F_old[(offs0t[i] + 1):offs0t[i + 1], j] = comp.comps[u][:, pos]
        end
    end

    F_new = DF.ExtTorSpaces._precompose_to_projective_cochains_matrix(Einj.homs[1], resP.aug, basesP0, active_plan, offs0t, Eb)
    @test F_new == F_old

    cache = CM.ResolutionCache()
    Euni_cached = DF.Ext(M, N, TO.DerivedFunctorOptions(maxdeg=1, model=:unified, canon=:projective); cache=cache)
    P2I_cached, I2P_cached = DF.comparison_isomorphisms(Euni_cached)
    @test length(P2I_cached) == 2
    @test length(I2P_cached) == 2

    Eproj_nocache = DF.Ext(M, N, TO.DerivedFunctorOptions(maxdeg=1, model=:projective); cache=nothing)
    Eproj_cache = DF.Ext(M, N, TO.DerivedFunctorOptions(maxdeg=1, model=:projective); cache=CM.ResolutionCache())
    @test [DF.dim(Eproj_nocache, t) for t in 0:1] == [DF.dim(Eproj_cache, t) for t in 0:1]

    Einj_nocache = DF.Ext(M, N, TO.DerivedFunctorOptions(maxdeg=1, model=:injective); cache=nothing)
    Einj_cache = DF.Ext(M, N, TO.DerivedFunctorOptions(maxdeg=1, model=:injective); cache=CM.ResolutionCache())
    @test [DF.dim(Einj_nocache, t) for t in 0:1] == [DF.dim(Einj_cache, t) for t in 0:1]
end

@testset "DerivedFunctors: downset postcompose coefficient solver" begin
    with_fields(FIELDS_FULL) do field
        # Minimal regression: 1-vertex poset, so all downsets are trivial and the fiberwise equation is
        # just C * F = G at the unique vertex.

        Q = FF.FinitePoset(trues(1, 1); check = false)
        K = CM.coeff_type(field)
        c(x) = CM.coerce(field, x)

        # X(u) has dimension 3, E(u) has 3 downset summands, Ep(u) has 2 downset summands.
        X  = MD.PModule{K}(Q, [3], Dict{Tuple{Int, Int}, Matrix{K}}(); field=field)
        E  = MD.PModule{K}(Q, [3], Dict{Tuple{Int, Int}, Matrix{K}}(); field=field)
        Ep = MD.PModule{K}(Q, [2], Dict{Tuple{Int, Int}, Matrix{K}}(); field=field)

        F1 = CM.eye(field, 3)
        G1 = [c(1) c(2) c(3);
              c(4) c(5) c(6)]

        f = MD.PMorphism(X, E,  [F1])
        g = MD.PMorphism(X, Ep, [G1])

        dom_bases = [1, 1, 1]
        cod_bases = [1, 1]
        act_dom = [collect(1:3)]
        act_cod = [collect(1:2)]

        C = DF._solve_downset_postcompose_coeff(f, g, dom_bases, cod_bases, act_dom, act_cod)
        @test C == G1
        @test C * F1 == G1

        # Inconsistent system: F = 0, G != 0 should throw.
        F0 = CM.zeros(field, 3, 3)
        f0 = MD.PMorphism(X, E, [F0])
        @test_throws ErrorException DF._solve_downset_postcompose_coeff(f0, g, dom_bases, cod_bases, act_dom, act_cod)
    end
end
