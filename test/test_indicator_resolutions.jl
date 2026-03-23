using Test

using Test
using LinearAlgebra

const FL = TamerOp.FieldLinAlg

@testset "IndicatorResolutions internal invariants + Ext on A2" begin
    # Internal PModule should match fiber_dimension from the fringe.
    P = chain_poset(3)
    field = CM.QQField()
    K = CM.coeff_type(field)
    M = one_by_one_fringe(P, FF.principal_upset(P, 2), FF.principal_downset(P, 2); scalar=one(K), field=field)
    PMM = IR.pmodule_from_fringe(M)
    @test PMM.edge_maps isa MD.CoverEdgeMapStore{K,Matrix{K}}
    for q in 1:P.n
        @test PMM.dims[q] == FF.fiber_dimension(M, q)
    end

    # Projective cover should be surjective on each vertex (full row rank).
    F0, pi0, _ = IR.projective_cover(PMM)
    for q in 1:P.n
        @test FL.rank(field, pi0.comps[q]) == PMM.dims[q]
    end

    # Kernel inclusion iota: K -> F0 should be injective and satisfy pi0 * iota = 0.
    K, iota = TO.kernel_with_inclusion(pi0)
    for q in 1:P.n
        @test FL.rank(field, iota.comps[q]) == K.dims[q]
        Z = pi0.comps[q] * iota.comps[q]
        @test Z == CM.zeros(field, size(Z,1), size(Z,2))
    end

    # Now test Ext dimensions on the A2 chain: 1 < 2
    _, S1, S2 = simple_modules_chain2()

    ext12 = DF.ext_dimensions_via_indicator_resolutions(S1, S2; maxlen=3)
    ext21 = DF.ext_dimensions_via_indicator_resolutions(S2, S1; maxlen=3)
    ext11 = DF.ext_dimensions_via_indicator_resolutions(S1, S1; maxlen=3)
    ext22 = DF.ext_dimensions_via_indicator_resolutions(S2, S2; maxlen=3)

    # Known quiver A2 facts:
    # Hom(S1,S2)=0, Ext^1(S1,S2)=1
    # Hom(S2,S1)=0, Ext^1(S2,S1)=0
    # Endomorphisms: Hom(Si,Si)=1, Ext^1(Si,Si)=0
    @test get(ext12, 0, 0) == 0
    @test get(ext12, 1, 0) == 1

    @test get(ext21, 0, 0) == 0
    @test get(ext21, 1, 0) == 0

    @test get(ext11, 0, 0) == 1
    @test get(ext11, 1, 0) == 0

    @test get(ext22, 0, 0) == 1
    @test get(ext22, 1, 0) == 0

    # Ext^0 should agree with FiniteFringe.hom_dimension on these simple cases.
    @test get(ext12, 0, 0) == FF.hom_dimension(S1, S2)
    @test get(ext21, 0, 0) == FF.hom_dimension(S2, S1)
    @test get(ext11, 0, 0) == FF.hom_dimension(S1, S1)
    @test get(ext22, 0, 0) == FF.hom_dimension(S2, S2)
end

@testset "IndicatorResolutions cached birth plans + typed workspace" begin
    P = diamond_poset()
    field = CM.QQField()
    K = CM.coeff_type(field)

    IR._clear_indicator_prefix_caches!()
    up1 = IR._upset_birth_block_plan(P)
    down1 = IR._downset_birth_block_plan(P)
    @test up1 === IR._upset_birth_block_plan(P)
    @test down1 === IR._downset_birth_block_plan(P)

    ws = IR._new_resolution_workspace(K, FF.nvertices(P))
    EntryT = TamerOp.AbelianCategories._VertexIncrementalCacheEntry{K}
    @test eltype(ws.kernel_vertex_cache) === EntryT

    IR._clear_indicator_prefix_caches!()
    @test IR._upset_birth_block_plan(P) !== up1
    @test IR._downset_birth_block_plan(P) !== down1
end

@testset "IndicatorResolutions F3 left-inverse regression" begin
    field = CM.F3()
    K = CM.coeff_type(field)
    c(x) = CM.coerce(field, x)

    S = reshape(K[c(1), c(1), c(1)], 3, 1)
    @test transpose(S) * S == reshape(K[c(0)], 1, 1)

    L = IR._left_inverse_full_column(field, S)
    @test L * S == CM.eye(field, 1)
end

@testset "IndicatorResolutions F3 downset regression completes" begin
    P = diamond_poset()
    field = CM.F3()
    K = CM.coeff_type(field)
    c(x) = CM.coerce(field, x)

    dims = [3, 1, 1, 0]
    edge = Dict{Tuple{Int,Int}, Matrix{K}}(
        (1, 2) => reshape(K[c(1), c(2), c(0)], 1, 3),
        (1, 3) => reshape(K[c(1), c(0), c(2)], 1, 3),
    )
    M = MD.PModule{K}(P, dims, edge; field=field)

    E, dE = IR.downset_resolution(M; maxlen=2, threads=false)
    @test IR.verify_downset_resolution(E, dE)
end

@testset "IndicatorResolutions basis helper parity to dense oracle" begin
    P = diamond_poset()
    field = CM.QQField()
    K = CM.coeff_type(field)

    U = [FF.principal_upset(P, 2), FF.principal_upset(P, 3)]
    D = [FF.principal_downset(P, 4)]
    Phi = spzeros(K, 1, 2)
    Phi[1, 1] = CM.coerce(field, 1)
    Phi[1, 2] = CM.coerce(field, 1)
    H = FF.FringeModule{K}(P, U, D, Phi; field=field)
    M = IR.pmodule_from_fringe(H)

    FF.build_cache!(P; cover=true, updown=true)
    cc = MD._get_cover_cache(P)

    same_colspace(A, B) = (size(A, 2) == size(B, 2) &&
                           FL.rank(field, hcat(A, B)) == size(A, 2))
    same_rowspace(A, B) = (size(A, 1) == size(B, 1) &&
                           FL.rank(field, hcat(transpose(A), transpose(B))) == size(A, 1))

    for v in 1:FF.nvertices(P)
        Bv = IR._incoming_image_basis(M, v; cache=cc)
        pv = FF._preds(cc, v)
        Bref = if isempty(pv) || M.dims[v] == 0
            zeros(K, M.dims[v], 0)
        else
            FL.colspace(field, hcat([M.edge_maps[u, v] for u in pv]...))
        end
        @test same_colspace(Bv, Bref)
    end

    for u in 1:FF.nvertices(P)
        Su = IR._outgoing_span_basis(M, u; cache=cc)
        su = FF._succs(cc, u)
        Sref = if isempty(su) || M.dims[u] == 0
            zeros(K, 0, M.dims[u])
        else
            stacked = vcat([M.edge_maps[u, v] for v in su]...)
            transpose(FL.colspace(field, transpose(stacked)))
        end
        @test same_rowspace(Su, Sref)

        Zu = IR._socle_basis(M, u; cache=cc)
        Zref = if isempty(su) || M.dims[u] == 0
            CM.eye(field, M.dims[u])
        else
            stacked = vcat([M.edge_maps[u, v] for v in su]...)
            FL.nullspace(field, stacked)
        end
        @test same_colspace(Zu, Zref)
    end
end

@testset "IndicatorResolutions gate parity" begin
    P = diamond_poset()
    field = CM.QQField()
    K = CM.coeff_type(field)

    U = [FF.principal_upset(P, 2), FF.principal_upset(P, 3)]
    D = [FF.principal_downset(P, 4)]
    Phi = spzeros(K, 1, 2)
    Phi[1, 1] = CM.coerce(field, 1)
    Phi[1, 2] = CM.coerce(field, 1)
    H = FF.FringeModule{K}(P, U, D, Phi; field=field)

    old_store_edges = IR.INDICATOR_PMODULE_DIRECT_STORE_MIN_EDGES[]
    old_store_work = IR.INDICATOR_PMODULE_DIRECT_STORE_MIN_WORK[]
    old_up_inc = IR.INDICATOR_INCREMENTAL_LINALG[]
    old_up_inc_maps = IR.INDICATOR_INCREMENTAL_LINALG_MIN_MAPS[]
    old_up_inc_entries = IR.INDICATOR_INCREMENTAL_LINALG_MIN_ENTRIES[]
    old_down_inc_maps = IR.INDICATOR_DOWNSET_INCREMENTAL_LINALG_MIN_MAPS[]
    old_down_inc_entries = IR.INDICATOR_DOWNSET_INCREMENTAL_LINALG_MIN_ENTRIES[]

    try
        M_base = IR.pmodule_from_fringe(H)

        IR.INDICATOR_PMODULE_DIRECT_STORE_MIN_EDGES[] = typemax(Int)
        IR.INDICATOR_PMODULE_DIRECT_STORE_MIN_WORK[] = typemax(Int)
        M_dict = IR.pmodule_from_fringe(H)

        IR.INDICATOR_PMODULE_DIRECT_STORE_MIN_EDGES[] = 0
        IR.INDICATOR_PMODULE_DIRECT_STORE_MIN_WORK[] = 0
        M_store = IR.pmodule_from_fringe(H)

        @test M_base.dims == M_store.dims
        @test M_base.edge_maps == M_store.edge_maps
        @test M_dict.dims == M_store.dims
        @test M_dict.edge_maps == M_store.edge_maps

        FF.build_cache!(P; cover=true, updown=true)
        cc = MD._get_cover_cache(P)

        IR.INDICATOR_INCREMENTAL_LINALG[] = false
        IR.INDICATOR_DOWNSET_INCREMENTAL_LINALG_MIN_MAPS[] = typemax(Int)
        IR.INDICATOR_DOWNSET_INCREMENTAL_LINALG_MIN_ENTRIES[] = typemax(Int)
        dense_in = [IR._incoming_image_basis(M_store, v; cache=cc) for v in 1:FF.nvertices(P)]
        dense_out = [IR._outgoing_span_basis(M_store, u; cache=cc) for u in 1:FF.nvertices(P)]
        dense_soc = [IR._socle_basis(M_store, u; cache=cc) for u in 1:FF.nvertices(P)]

        IR.INDICATOR_INCREMENTAL_LINALG[] = true
        IR.INDICATOR_INCREMENTAL_LINALG_MIN_MAPS[] = 1
        IR.INDICATOR_INCREMENTAL_LINALG_MIN_ENTRIES[] = 0
        IR.INDICATOR_DOWNSET_INCREMENTAL_LINALG_MIN_MAPS[] = 1
        IR.INDICATOR_DOWNSET_INCREMENTAL_LINALG_MIN_ENTRIES[] = 0
        inc_in = [IR._incoming_image_basis(M_store, v; cache=cc) for v in 1:FF.nvertices(P)]
        inc_out = [IR._outgoing_span_basis(M_store, u; cache=cc) for u in 1:FF.nvertices(P)]
        inc_soc = [IR._socle_basis(M_store, u; cache=cc) for u in 1:FF.nvertices(P)]

        same_colspace(A, B) = (size(A, 2) == size(B, 2) &&
                               FL.rank(field, hcat(A, B)) == size(A, 2))
        same_rowspace(A, B) = (size(A, 1) == size(B, 1) &&
                               FL.rank(field, hcat(transpose(A), transpose(B))) == size(A, 1))

        for i in eachindex(dense_in)
            @test same_colspace(dense_in[i], inc_in[i])
            @test same_rowspace(dense_out[i], inc_out[i])
            @test same_colspace(dense_soc[i], inc_soc[i])
        end
    finally
        IR.INDICATOR_PMODULE_DIRECT_STORE_MIN_EDGES[] = old_store_edges
        IR.INDICATOR_PMODULE_DIRECT_STORE_MIN_WORK[] = old_store_work
        IR.INDICATOR_INCREMENTAL_LINALG[] = old_up_inc
        IR.INDICATOR_INCREMENTAL_LINALG_MIN_MAPS[] = old_up_inc_maps
        IR.INDICATOR_INCREMENTAL_LINALG_MIN_ENTRIES[] = old_up_inc_entries
        IR.INDICATOR_DOWNSET_INCREMENTAL_LINALG_MIN_MAPS[] = old_down_inc_maps
        IR.INDICATOR_DOWNSET_INCREMENTAL_LINALG_MIN_ENTRIES[] = old_down_inc_entries
    end
end

@testset "IndicatorResolutions upset auto profile is inspectable" begin
    P = diamond_poset()
    field = CM.QQField()
    K = CM.coeff_type(field)
    edge = Dict{Tuple{Int,Int}, Matrix{K}}(
        (1, 2) => reshape(K[CM.coerce(field, 1)], 1, 1),
        (1, 3) => reshape(K[CM.coerce(field, 1)], 1, 1),
        (2, 4) => reshape(K[CM.coerce(field, 1)], 1, 1),
        (3, 4) => reshape(K[CM.coerce(field, 1)], 1, 1),
    )
    M = MD.PModule{K}(P, ones(Int, 4), edge; field=field)
    prof = IR._indicator_upset_auto_profile(M; maxlen=2)
    @test prof.n == FF.nvertices(P)
    @test prof.total_dims == sum(M.dims)
    @test prof.vertex_cache isa Bool
    @test prof.prefix_cache isa Bool
    @test prof.incremental_linalg_thresholds isa Tuple{Int,Int}
end

@testset "IndicatorResolutions vertex incremental cache parity" begin
    P = diamond_poset()
    field = CM.QQField()
    K = CM.coeff_type(field)

    U = [FF.principal_upset(P, 2), FF.principal_upset(P, 3)]
    D = [FF.principal_downset(P, 4)]
    Phi = spzeros(K, 1, 2)
    Phi[1, 1] = CM.coerce(field, 1)
    Phi[1, 2] = CM.coerce(field, 1)
    H = FF.FringeModule{K}(P, U, D, Phi; field=field)
    M = IR.pmodule_from_fringe(H)
    cc = MD._get_cover_cache(P)

    old_up_enabled = IR.INDICATOR_INCREMENTAL_VERTEX_CACHE[]
    old_up_min_vertices = IR.INDICATOR_INCREMENTAL_VERTEX_CACHE_MIN_VERTICES[]
    old_up_min_total_dims = IR.INDICATOR_INCREMENTAL_VERTEX_CACHE_MIN_TOTAL_DIMS[]

    try
        IR.INDICATOR_INCREMENTAL_VERTEX_CACHE[] = false
        F_off, dF_off = IR.upset_resolution(M; maxlen=2, cache=cc, threads=false)

        IR.INDICATOR_INCREMENTAL_VERTEX_CACHE[] = true
        IR.INDICATOR_INCREMENTAL_VERTEX_CACHE_MIN_VERTICES[] = 0
        IR.INDICATOR_INCREMENTAL_VERTEX_CACHE_MIN_TOTAL_DIMS[] = 0
        F_on, dF_on = IR.upset_resolution(M; maxlen=2, cache=cc, threads=false)

        @test length(F_off) == length(F_on)
        @test dF_off == dF_on
        for i in eachindex(F_off)
            @test F_off[i].U0 == F_on[i].U0
            @test F_off[i].U1 == F_on[i].U1
            @test F_off[i].delta == F_on[i].delta
        end
    finally
        IR.INDICATOR_INCREMENTAL_VERTEX_CACHE[] = old_up_enabled
        IR.INDICATOR_INCREMENTAL_VERTEX_CACHE_MIN_VERTICES[] = old_up_min_vertices
        IR.INDICATOR_INCREMENTAL_VERTEX_CACHE_MIN_TOTAL_DIMS[] = old_up_min_total_dims
    end
end

@testset "IndicatorResolutions field-specific incremental thresholds respect explicit overrides" begin
    @test IR._indicator_incremental_linalg_enabled(Val(:downset))

    qq_maps, qq_entries = IR._indicator_incremental_union_thresholds(CM.QQField())
    f3_maps, f3_entries = IR._indicator_incremental_union_thresholds(CM.F3())
    @test (qq_maps, qq_entries) == (
        IR._INDICATOR_INCREMENTAL_LINALG_MIN_MAPS_QQ,
        IR._INDICATOR_INCREMENTAL_LINALG_MIN_ENTRIES_QQ,
    )
    @test (f3_maps, f3_entries) == (
        IR._INDICATOR_INCREMENTAL_LINALG_MIN_MAPS_PRIME,
        IR._INDICATOR_INCREMENTAL_LINALG_MIN_ENTRIES_PRIME,
    )

    qq_v, qq_dims = IR._indicator_vertex_cache_thresholds(CM.QQField())
    f3_v, f3_dims = IR._indicator_vertex_cache_thresholds(CM.F3())
    @test f3_v >= qq_v
    @test f3_dims >= qq_dims

    qq_down_maps, qq_down_entries = IR._indicator_incremental_union_thresholds(CM.QQField(), Val(:downset))
    f3_down_maps, f3_down_entries = IR._indicator_incremental_union_thresholds(CM.F3(), Val(:downset))
    @test qq_down_maps > qq_maps
    @test qq_down_entries > qq_entries
    @test (qq_down_maps, qq_down_entries) == (
        IR._INDICATOR_DOWNSET_INCREMENTAL_LINALG_MIN_MAPS_QQ,
        IR._INDICATOR_DOWNSET_INCREMENTAL_LINALG_MIN_ENTRIES_QQ,
    )
    @test (f3_down_maps, f3_down_entries) == (
        IR._INDICATOR_DOWNSET_INCREMENTAL_LINALG_MIN_MAPS_PRIME,
        IR._INDICATOR_DOWNSET_INCREMENTAL_LINALG_MIN_ENTRIES_PRIME,
    )

    old_maps = IR.INDICATOR_INCREMENTAL_LINALG_MIN_MAPS[]
    old_entries = IR.INDICATOR_INCREMENTAL_LINALG_MIN_ENTRIES[]
    old_v = IR.INDICATOR_INCREMENTAL_VERTEX_CACHE_MIN_VERTICES[]
    old_dims = IR.INDICATOR_INCREMENTAL_VERTEX_CACHE_MIN_TOTAL_DIMS[]
    old_down_maps = IR.INDICATOR_DOWNSET_INCREMENTAL_LINALG_MIN_MAPS[]
    old_down_entries = IR.INDICATOR_DOWNSET_INCREMENTAL_LINALG_MIN_ENTRIES[]
    try
        IR.INDICATOR_INCREMENTAL_LINALG_MIN_MAPS[] = 1
        IR.INDICATOR_INCREMENTAL_LINALG_MIN_ENTRIES[] = 0
        IR.INDICATOR_INCREMENTAL_VERTEX_CACHE_MIN_VERTICES[] = 0
        IR.INDICATOR_INCREMENTAL_VERTEX_CACHE_MIN_TOTAL_DIMS[] = 0
        IR.INDICATOR_DOWNSET_INCREMENTAL_LINALG_MIN_MAPS[] = 2
        IR.INDICATOR_DOWNSET_INCREMENTAL_LINALG_MIN_ENTRIES[] = 3
        @test IR._indicator_incremental_union_thresholds(CM.F3()) == (1, 0)
        @test IR._indicator_vertex_cache_thresholds(CM.F3()) == (0, 0)
        @test IR._indicator_incremental_union_thresholds(CM.F3(), Val(:downset)) == (2, 3)
    finally
        IR.INDICATOR_INCREMENTAL_LINALG_MIN_MAPS[] = old_maps
        IR.INDICATOR_INCREMENTAL_LINALG_MIN_ENTRIES[] = old_entries
        IR.INDICATOR_INCREMENTAL_VERTEX_CACHE_MIN_VERTICES[] = old_v
        IR.INDICATOR_INCREMENTAL_VERTEX_CACHE_MIN_TOTAL_DIMS[] = old_dims
        IR.INDICATOR_DOWNSET_INCREMENTAL_LINALG_MIN_MAPS[] = old_down_maps
        IR.INDICATOR_DOWNSET_INCREMENTAL_LINALG_MIN_ENTRIES[] = old_down_entries
    end
end

@testset "IndicatorResolutions dense-id assembly + budgets" begin
    # Non-chain shape exercises the active-generator edge assembly logic.
    P = diamond_poset()
    field = CM.QQField()
    K = CM.coeff_type(field)

    U = [FF.principal_upset(P, 2), FF.principal_upset(P, 3)]
    D = [FF.principal_downset(P, 4)]
    Phi = spzeros(K, 1, 2)
    Phi[1, 1] = CM.coerce(field, 1)
    Phi[1, 2] = CM.coerce(field, 1)
    H = FF.FringeModule{K}(P, U, D, Phi; field=field)
    M = IR.pmodule_from_fringe(H)

    F0s, pi0s, _ = IR.projective_cover(M; threads=false)
    E0s, iotas, _ = IR._injective_hull(M; threads=false)

    for (u, v) in FF.cover_edges(P)
        @test F0s.edge_maps[u, v] isa SparseMatrixCSC
        @test E0s.edge_maps[u, v] isa SparseMatrixCSC
    end

    if Threads.nthreads() > 1
        F0t, pi0t, _ = IR.projective_cover(M; threads=true)
        @test F0t.dims == F0s.dims
        @test pi0t.comps == pi0s.comps
        for (u, v) in FF.cover_edges(P)
            @test F0t.edge_maps[u, v] == F0s.edge_maps[u, v]
        end

        E0t, iotat, _ = IR._injective_hull(M; threads=true)
        @test E0t.dims == E0s.dims
        @test iotat.comps == iotas.comps
        for (u, v) in FF.cover_edges(P)
            @test E0t.edge_maps[u, v] == E0s.edge_maps[u, v]
        end
    end

    # Allocation guards: warm then measure on fixed fixture.
    IR.projective_cover(M; threads=false)
    alloc_proj_cover = @allocated IR.projective_cover(M; threads=false)
    @test alloc_proj_cover < 25_000_000

    IR._injective_hull(M; threads=false)
    alloc_inj_hull = @allocated IR._injective_hull(M; threads=false)
    @test alloc_inj_hull < 25_000_000
end

@testset "IndicatorResolutions support-aware injective plans preserve parity" begin
    P = diamond_poset()
    mult = [2, 0, 3, 0]
    active = [1, 3]
    Edims_full, sources_full = IR._injective_active_plan(P, mult)
    Edims_sparse, sources_sparse = IR._injective_active_plan(P, mult, active)
    @test Edims_sparse == Edims_full
    @test IR._packed_lists_to_vectors(sources_sparse, FF.nvertices(P)) ==
          IR._packed_lists_to_vectors(sources_full, FF.nvertices(P))

    field = CM.QQField()
    K = CM.coeff_type(field)
    c(x) = CM.coerce(field, x)
    edge = Dict{Tuple{Int,Int}, Matrix{K}}(
        (1, 2) => reshape(K[c(1), c(0)], 1, 2),
        (1, 3) => reshape(K[c(0), c(1)], 1, 2),
        (2, 4) => zeros(K, 0, 1),
        (3, 4) => zeros(K, 0, 1),
    )
    M = MD.PModule{K}(P, [2, 1, 1, 0], edge; field=field)
    support = IR._nonzero_dim_vertices(M.dims)
    E_full, iota_full, gens_full = IR._injective_hull(M; threads=false)
    E_sparse, iota_sparse, gens_sparse = IR._injective_hull(M; support_vertices=support, threads=false)
    @test E_sparse.dims == E_full.dims
    @test E_sparse.edge_maps == E_full.edge_maps
    @test iota_sparse.comps == iota_full.comps
    @test gens_sparse == gens_full

    AB = TamerOp.AbelianCategories
    C_full, q_full = AB._cokernel_module(iota_full; cache=:auto)
    C_sparse, q_sparse = AB._cokernel_module(iota_sparse; cache=:auto, active_vertices=support)
    @test C_sparse.dims == C_full.dims
    @test C_sparse.edge_maps == C_full.edge_maps
    @test q_sparse.comps == q_full.comps

    support_mask = IR._vertex_mask(FF.nvertices(P), support)
    C_mask, q_mask = AB._cokernel_module(iota_sparse; cache=:auto, active_vertices=support, active_mask=support_mask)
    @test C_mask.dims == C_full.dims
    @test C_mask.edge_maps == C_full.edge_maps
    @test q_mask.comps == q_full.comps
end

@testset "IndicatorResolutions packed injective plan parity" begin
    P = diamond_poset()
    mult = [2, 0, 3, 0]
    active = [1, 3]
    KQQ = CM.coeff_type(CM.QQField())
    Edims, active_sources = IR._injective_active_plan(P, mult, active)
    gid_starts = Vector{Int}(undef, FF.nvertices(P) + 1)
    IR._fill_generator_starts!(gid_starts, mult)
    gids = IR._injective_active_gid_plan(active_sources, gid_starts, Edims)
    ids = IR._injective_active_ids(active_sources, gid_starts, Edims)
    for u in 1:FF.nvertices(P)
        @test IR._projection_identity_sparse(KQQ, gids, u, u) == IR._projection_identity_sparse(KQQ, ids[u], ids[u])
    end

    field = CM.QQField()
    K = CM.coeff_type(field)
    c(x) = CM.coerce(field, x)
    edge = Dict{Tuple{Int,Int}, Matrix{K}}(
        (1, 2) => reshape(K[c(1), c(0)], 1, 2),
        (1, 3) => reshape(K[c(0), c(1)], 1, 2),
        (2, 4) => zeros(K, 0, 1),
        (3, 4) => zeros(K, 0, 1),
    )
    M = MD.PModule{K}(P, [2, 1, 1, 0], edge; field=field)
    old_min_socle = IR._INDICATOR_INJECTIVE_PACKED_PLAN_MIN_SOCLE_VERTICES[]
    old_min_gens = IR._INDICATOR_INJECTIVE_PACKED_PLAN_MIN_TOTAL_GENS[]
    old_min_hull = IR._INDICATOR_INJECTIVE_PACKED_PLAN_MIN_TOTAL_HULL_DIMS[]
    try
        IR._INDICATOR_INJECTIVE_PACKED_PLAN_MIN_SOCLE_VERTICES[] = typemax(Int)
        IR._INDICATOR_INJECTIVE_PACKED_PLAN_MIN_TOTAL_GENS[] = typemax(Int)
        IR._INDICATOR_INJECTIVE_PACKED_PLAN_MIN_TOTAL_HULL_DIMS[] = typemax(Int)
        E_ids, iota_ids, gens_ids = IR._injective_hull(M; support_vertices=IR._nonzero_dim_vertices(M.dims), threads=false)
        IR._INDICATOR_INJECTIVE_PACKED_PLAN_MIN_SOCLE_VERTICES[] = 0
        IR._INDICATOR_INJECTIVE_PACKED_PLAN_MIN_TOTAL_GENS[] = 0
        IR._INDICATOR_INJECTIVE_PACKED_PLAN_MIN_TOTAL_HULL_DIMS[] = 0
        E_packed, iota_packed, gens_packed = IR._injective_hull(M; support_vertices=IR._nonzero_dim_vertices(M.dims), threads=false)
        @test E_packed.dims == E_ids.dims
        @test E_packed.edge_maps == E_ids.edge_maps
        @test iota_packed.comps == iota_ids.comps
        @test gens_packed == gens_ids
    finally
        IR._INDICATOR_INJECTIVE_PACKED_PLAN_MIN_SOCLE_VERTICES[] = old_min_socle
        IR._INDICATOR_INJECTIVE_PACKED_PLAN_MIN_TOTAL_GENS[] = old_min_gens
        IR._INDICATOR_INJECTIVE_PACKED_PLAN_MIN_TOTAL_HULL_DIMS[] = old_min_hull
    end
end

@testset "IndicatorResolutions downset rho value-only parity" begin
    P = diamond_poset()
    field = CM.QQField()
    K = CM.coeff_type(field)
    c(x) = CM.coerce(field, x)
    edge = Dict{Tuple{Int,Int}, Matrix{K}}(
        (1, 2) => reshape(K[c(1), c(0)], 1, 2),
        (1, 3) => reshape(K[c(0), c(1)], 1, 2),
        (2, 4) => zeros(K, 0, 1),
        (3, 4) => zeros(K, 0, 1),
    )
    M = MD.PModule{K}(P, [2, 1, 1, 0], edge; field=field)
    FF.build_cache!(P; cover=true, updown=true)
    cc = MD._get_cover_cache(P)
    old = IR._INDICATOR_DOWNSET_RHO_VALUE_ONLY_REUSE[]
    try
        IR._INDICATOR_DOWNSET_RHO_VALUE_ONLY_REUSE[] = false
        E_old, dE_old = IR.downset_resolution(M; maxlen=2, cache=cc, threads=false)
        IR._INDICATOR_DOWNSET_RHO_VALUE_ONLY_REUSE[] = true
        E_new, dE_new = IR.downset_resolution(M; maxlen=2, cache=cc, threads=false)
        @test length(E_new) == length(E_old)
        @test length(dE_new) == length(dE_old)
        @test [Ei.D0 for Ei in E_new] == [Ei.D0 for Ei in E_old]
        @test [Ei.D1 for Ei in E_new] == [Ei.D1 for Ei in E_old]
        @test dE_new == dE_old
    finally
        IR._INDICATOR_DOWNSET_RHO_VALUE_ONLY_REUSE[] = old
    end
end

@testset "IndicatorResolutions downset rho sparse-workspace parity" begin
    P = diamond_poset()
    field = CM.QQField()
    K = CM.coeff_type(field)

    U = [FF.principal_upset(P, 2), FF.principal_upset(P, 3)]
    D = [FF.principal_downset(P, 4)]
    Phi = spzeros(K, 1, 2)
    Phi[1, 1] = CM.coerce(field, 1)
    Phi[1, 2] = CM.coerce(field, 1)
    H = FF.FringeModule{K}(P, U, D, Phi; field=field)
    M = IR.pmodule_from_fringe(H)
    cc = MD._get_cover_cache(P)

    old = IR._INDICATOR_DOWNSET_RHO_SPARSE_WORKSPACE[]
    try
        IR._INDICATOR_DOWNSET_RHO_SPARSE_WORKSPACE[] = false
        E_off, dE_off = IR.downset_resolution(M; maxlen=2, cache=cc, threads=false)

        IR._INDICATOR_DOWNSET_RHO_SPARSE_WORKSPACE[] = true
        E_on, dE_on = IR.downset_resolution(M; maxlen=2, cache=cc, threads=false)

        @test length(E_on) == length(E_off)
        @test dE_on == dE_off
    finally
        IR._INDICATOR_DOWNSET_RHO_SPARSE_WORKSPACE[] = old
    end
end

@testset "IndicatorResolutions downset cokernel transport reuse parity" begin
    P = diamond_poset()
    field = CM.QQField()
    K = CM.coeff_type(field)
    c(x) = CM.coerce(field, x)
    edge = Dict{Tuple{Int,Int}, Matrix{K}}(
        (1, 2) => reshape(K[c(1), c(0)], 1, 2),
        (1, 3) => reshape(K[c(0), c(1)], 1, 2),
        (2, 4) => zeros(K, 0, 1),
        (3, 4) => zeros(K, 0, 1),
    )
    M = MD.PModule{K}(P, [2, 1, 1, 0], edge; field=field)
    FF.build_cache!(P; cover=true, updown=true)
    cc = MD._get_cover_cache(P)
    old = IR._INDICATOR_DOWNSET_COKERNEL_TRANSPORT_REUSE[]
    old_min_vertices = IR._INDICATOR_DOWNSET_TRANSPORT_REUSE_MIN_VERTICES[]
    old_min_total_dims = IR._INDICATOR_DOWNSET_TRANSPORT_REUSE_MIN_TOTAL_DIMS[]
    try
        IR._INDICATOR_DOWNSET_TRANSPORT_REUSE_MIN_VERTICES[] = 0
        IR._INDICATOR_DOWNSET_TRANSPORT_REUSE_MIN_TOTAL_DIMS[] = 0
        IR._INDICATOR_DOWNSET_COKERNEL_TRANSPORT_REUSE[] = false
        E_old, dE_old = IR.downset_resolution(M; maxlen=2, cache=cc, threads=false)
        IR._INDICATOR_DOWNSET_COKERNEL_TRANSPORT_REUSE[] = true
        E_new, dE_new = IR.downset_resolution(M; maxlen=2, cache=cc, threads=false)
        @test length(E_new) == length(E_old)
        @test length(dE_new) == length(dE_old)
        @test [Ei.D0 for Ei in E_new] == [Ei.D0 for Ei in E_old]
        @test [Ei.D1 for Ei in E_new] == [Ei.D1 for Ei in E_old]
        @test dE_new == dE_old
    finally
        IR._INDICATOR_DOWNSET_COKERNEL_TRANSPORT_REUSE[] = old
        IR._INDICATOR_DOWNSET_TRANSPORT_REUSE_MIN_VERTICES[] = old_min_vertices
        IR._INDICATOR_DOWNSET_TRANSPORT_REUSE_MIN_TOTAL_DIMS[] = old_min_total_dims
    end
end

@testset "IndicatorResolutions cached cover-graph parity" begin
    P = diamond_poset()
    AB = TamerOp.AbelianCategories
    field = CM.QQField()
    K = CM.coeff_type(field)
    c(x) = CM.coerce(field, x)
    edge = Dict{Tuple{Int,Int}, Matrix{K}}(
        (1, 2) => reshape(K[c(1), c(0)], 1, 2),
        (1, 3) => reshape(K[c(0), c(1)], 1, 2),
        (2, 4) => zeros(K, 0, 1),
        (3, 4) => zeros(K, 0, 1),
    )
    M = MD.PModule{K}(P, [2, 1, 1, 0], edge; field=field)
    FF.build_cache!(P; cover=true, updown=true)
    cc = MD._get_cover_cache(P)
    graph_lists = AB._cover_graph_lists(cc)
    support0 = IR._nonzero_dim_vertices(M.dims)

    E0_ref, iota0_ref, gens0_ref = IR._injective_hull(M; cache=cc, support_vertices=support0, threads=false)
    E0_new, iota0_new, gens0_new = IR._injective_hull(M; cache=cc, support_vertices=support0, graph_lists=graph_lists, threads=false)
    @test E0_new.dims == E0_ref.dims
    @test E0_new.edge_maps == E0_ref.edge_maps
    @test iota0_new.comps == iota0_ref.comps
    @test gens0_new == gens0_ref

    C0_ref, q0_ref = AB._cokernel_module(iota0_ref; cache=cc, active_vertices=support0)
    C0_new, q0_new = AB._cokernel_module(iota0_new; cache=cc, active_vertices=support0, graph_lists=graph_lists)
    @test C0_new.dims == C0_ref.dims
    @test C0_new.edge_maps == C0_ref.edge_maps
    @test q0_new.comps == q0_ref.comps

    support1 = IR._nonzero_dim_vertices(C0_ref.dims)
    E1_ref, j_ref, gens1_ref = IR._injective_hull(C0_ref; cache=cc, support_vertices=support1, threads=false)
    E1_new, j_new, gens1_new = IR._injective_hull(C0_new; cache=cc, support_vertices=support1, graph_lists=graph_lists, threads=false)
    @test E1_new.dims == E1_ref.dims
    @test E1_new.edge_maps == E1_ref.edge_maps
    @test j_new.comps == j_ref.comps
    @test gens1_new == gens1_ref

    C1_ref, q1_ref = AB._cokernel_module(j_ref; cache=cc, active_vertices=support1)
    C1_new, q1_new = AB._cokernel_module(j_new; cache=cc, active_vertices=support1, graph_lists=graph_lists)
    @test C1_new.dims == C1_ref.dims
    @test C1_new.edge_maps == C1_ref.edge_maps
    @test q1_new.comps == q1_ref.comps
end

@testset "IndicatorResolutions injective generator plan parity" begin
    P = diamond_poset()
    field = CM.QQField()
    K = CM.coeff_type(field)
    c(x) = CM.coerce(field, x)
    edge = Dict{Tuple{Int,Int}, Matrix{K}}(
        (1, 2) => reshape(K[c(1), c(0)], 1, 2),
        (1, 3) => reshape(K[c(0), c(1)], 1, 2),
        (2, 4) => zeros(K, 0, 1),
        (3, 4) => zeros(K, 0, 1),
    )
    M = MD.PModule{K}(P, [2, 1, 1, 0], edge; field=field)
    FF.build_cache!(P; cover=true, updown=true)
    cc = MD._get_cover_cache(P)
    support = IR._nonzero_dim_vertices(M.dims)
    E_ref, iota_ref, gens_ref = IR._injective_hull(M; cache=cc, support_vertices=support, threads=false)
    E_plan, iota_plan, plan = IR._injective_hull(M; cache=cc, support_vertices=support, materialize_gens=false, threads=false)
    @test E_plan.dims == E_ref.dims
    @test E_plan.edge_maps == E_ref.edge_maps
    @test iota_plan.comps == iota_ref.comps
    @test gens_ref isa IR.InjectiveGenerators
    @test [collect(g) for g in gens_ref] == IR._materialize_injective_gens(plan)
    @test IR._principal_downsets_from_plan(P, plan) == IR._principal_downsets_from_gens(P, gens_ref)

    old_min_socle = IR._INDICATOR_INJECTIVE_PACKED_PLAN_MIN_SOCLE_VERTICES[]
    old_min_gens = IR._INDICATOR_INJECTIVE_PACKED_PLAN_MIN_TOTAL_GENS[]
    old_min_hull = IR._INDICATOR_INJECTIVE_PACKED_PLAN_MIN_TOTAL_HULL_DIMS[]
    try
        IR._INDICATOR_INJECTIVE_PACKED_PLAN_MIN_SOCLE_VERTICES[] = 0
        IR._INDICATOR_INJECTIVE_PACKED_PLAN_MIN_TOTAL_GENS[] = 0
        IR._INDICATOR_INJECTIVE_PACKED_PLAN_MIN_TOTAL_HULL_DIMS[] = 0
        E_packed, iota_packed, plan_packed = IR._injective_hull(M; cache=cc, support_vertices=support, materialize_gens=false, threads=false)
        IR._INDICATOR_INJECTIVE_PACKED_PLAN_MIN_SOCLE_VERTICES[] = typemax(Int)
        IR._INDICATOR_INJECTIVE_PACKED_PLAN_MIN_TOTAL_GENS[] = typemax(Int)
        IR._INDICATOR_INJECTIVE_PACKED_PLAN_MIN_TOTAL_HULL_DIMS[] = typemax(Int)
        E_simple, iota_simple, plan_simple = IR._injective_hull(M; cache=cc, support_vertices=support, materialize_gens=false, threads=false)
        @test E_packed.dims == E_simple.dims == E_ref.dims
        @test E_packed.edge_maps == E_simple.edge_maps == E_ref.edge_maps
        @test iota_packed.comps == iota_simple.comps == iota_ref.comps
        @test [collect(g) for g in gens_ref] == IR._materialize_injective_gens(plan_packed) == IR._materialize_injective_gens(plan_simple)
    finally
        IR._INDICATOR_INJECTIVE_PACKED_PLAN_MIN_SOCLE_VERTICES[] = old_min_socle
        IR._INDICATOR_INJECTIVE_PACKED_PLAN_MIN_TOTAL_GENS[] = old_min_gens
        IR._INDICATOR_INJECTIVE_PACKED_PLAN_MIN_TOTAL_HULL_DIMS[] = old_min_hull
    end
end

@testset "IndicatorResolutions downset auto profile" begin
    field = CM.QQField()
    K = CM.coeff_type(field)
    grid_poset(nx, ny) = begin
        rel = falses(nx * ny, nx * ny)
        idx(ix, iy) = (iy - 1) * nx + ix
        @inbounds for y1 in 1:ny, x1 in 1:nx
            i = idx(x1, y1)
            for y2 in y1:ny, x2 in x1:nx
                rel[i, idx(x2, y2)] = true
            end
        end
        FF.FinitePoset(rel; check=false)
    end
    zero_edge_module(P, dim) = begin
        edge = Dict{Tuple{Int,Int},Matrix{K}}()
        for (u, v) in FF.cover_edges(P)
            edge[(u, v)] = zeros(K, dim, dim)
        end
        MD.PModule{K}(P, fill(dim, FF.nvertices(P)), edge; field=field)
    end
    P_small = grid_poset(4, 4)
    FF.build_cache!(P_small; cover=true, updown=true)
    small = zero_edge_module(P_small, 1)
    prof_small = IR._indicator_downset_auto_profile(small; maxlen=2)
    @test prof_small.transport_reuse == false
    @test prof_small.injective_reuse == false
    @test prof_small.prefix_cache == false

    P_mid = grid_poset(8, 8)
    FF.build_cache!(P_mid; cover=true, updown=true)
    mid = zero_edge_module(P_mid, 2)
    prof_mid = IR._indicator_downset_auto_profile(mid; maxlen=2)
    @test prof_mid.transport_reuse == true
    @test prof_mid.injective_reuse == false
    @test prof_mid.prefix_cache == false

    P_big = grid_poset(12, 12)
    FF.build_cache!(P_big; cover=true, updown=true)
    big = zero_edge_module(P_big, 2)
    prof_big = IR._indicator_downset_auto_profile(big; maxlen=3)
    @test prof_big.transport_reuse == true
    @test prof_big.injective_reuse == false
    @test prof_big.prefix_cache == false
end

@testset "IndicatorResolutions projective generator plan parity" begin
    P = diamond_poset()
    field = CM.QQField()
    K = CM.coeff_type(field)
    c(x) = CM.coerce(field, x)
    edge = Dict{Tuple{Int,Int}, Matrix{K}}(
        (1, 2) => reshape(K[c(1), c(0)], 1, 2),
        (1, 3) => reshape(K[c(0), c(1)], 1, 2),
        (2, 4) => zeros(K, 0, 1),
        (3, 4) => zeros(K, 0, 1),
    )
    M = MD.PModule{K}(P, [2, 1, 1, 0], edge; field=field)
    FF.build_cache!(P; cover=true, updown=true)
    cc = MD._get_cover_cache(P)
    F_ref, pi_ref, gens_ref = IR.projective_cover(M; cache=cc, threads=false)
    F_plan, pi_plan, plan = IR.projective_cover(M; cache=cc, materialize_gens=false, threads=false)
    @test F_plan.dims == F_ref.dims
    @test F_plan.edge_maps == F_ref.edge_maps
    @test pi_plan.comps == pi_ref.comps
    @test gens_ref isa IR.ProjectiveGenerators
    @test [collect(g) for g in gens_ref] == IR._materialize_projective_gens(plan)
    @test IR._principal_upsets_from_plan(P, plan) == IR._principal_upsets_from_gens(P, gens_ref)
end

@testset "IndicatorResolutions explicit workspace/cache parity" begin
    P = diamond_poset()
    field = CM.QQField()
    K = CM.coeff_type(field)

    U = [FF.principal_upset(P, 2), FF.principal_upset(P, 3)]
    D = [FF.principal_downset(P, 4)]
    Phi = spzeros(K, 1, 2)
    Phi[1, 1] = CM.coerce(field, 1)
    Phi[1, 2] = CM.coerce(field, 1)
    H = FF.FringeModule{K}(P, U, D, Phi; field=field)
    M = IR.pmodule_from_fringe(H)

    FF.build_cache!(P; cover=true, updown=true)
    cc = MD._get_cover_cache(P)
    memo = IR._indicator_new_array_memo(K, FF.nvertices(P))
    ws = IR._new_resolution_workspace(K, FF.nvertices(P))

    F_ref, dF_ref = IR.upset_resolution(M; maxlen=3, threads=false)
    E_ref, dE_ref = IR.downset_resolution(M; maxlen=3, threads=false)

    F_opt, dF_opt = IR.upset_resolution(
        M;
        maxlen=3,
        cache=cc,
        map_memo=memo,
        workspace=ws,
        threads=false,
    )
    E_opt, dE_opt = IR.downset_resolution(
        M;
        maxlen=3,
        cache=cc,
        map_memo=memo,
        workspace=ws,
        threads=false,
    )

    @test length(F_opt) == length(F_ref)
    @test dF_opt == dF_ref
    @test length(E_opt) == length(E_ref)
    @test dE_opt == dE_ref
end

@testset "IndicatorResolutions downset support narrowing parity" begin
    P = diamond_poset()
    field = CM.QQField()
    K = CM.coeff_type(field)

    U = [FF.principal_upset(P, 2), FF.principal_upset(P, 3)]
    D = [FF.principal_downset(P, 4)]
    Phi = spzeros(K, 1, 2)
    Phi[1, 1] = CM.coerce(field, 1)
    Phi[1, 2] = CM.coerce(field, 1)
    H = FF.FringeModule{K}(P, U, D, Phi; field=field)
    M = IR.pmodule_from_fringe(H)

    old = IR._INDICATOR_DOWNSET_SUPPORT_NARROWING[]
    try
        IR._INDICATOR_DOWNSET_SUPPORT_NARROWING[] = false
        E_off, dE_off = IR.downset_resolution(M; maxlen=3, threads=false)
        IR._INDICATOR_DOWNSET_SUPPORT_NARROWING[] = true
        E_on, dE_on = IR.downset_resolution(M; maxlen=3, threads=false)
        @test length(E_on) == length(E_off)
        @test dE_on == dE_off
        @test [length(E.D0) for E in E_on] == [length(E.D0) for E in E_off]
    finally
        IR._INDICATOR_DOWNSET_SUPPORT_NARROWING[] = old
    end
end

@testset "IndicatorResolutions downset frontier-vertices parity" begin
    P = diamond_poset()
    field = CM.QQField()
    K = CM.coeff_type(field)

    U = [FF.principal_upset(P, 2), FF.principal_upset(P, 3)]
    D = [FF.principal_downset(P, 4)]
    Phi = spzeros(K, 1, 2)
    Phi[1, 1] = CM.coerce(field, 1)
    Phi[1, 2] = CM.coerce(field, 1)
    H = FF.FringeModule{K}(P, U, D, Phi; field=field)
    M = IR.pmodule_from_fringe(H)

    old = IR._INDICATOR_DOWNSET_FRONTIER_VERTICES[]
    try
        IR._INDICATOR_DOWNSET_FRONTIER_VERTICES[] = false
        E_off, dE_off = IR.downset_resolution(M; maxlen=3, threads=false)
        IR._INDICATOR_DOWNSET_FRONTIER_VERTICES[] = true
        E_on, dE_on = IR.downset_resolution(M; maxlen=3, threads=false)
        @test length(E_on) == length(E_off)
        @test dE_on == dE_off
        @test [length(E.D0) for E in E_on] == [length(E.D0) for E in E_off]
    finally
        IR._INDICATOR_DOWNSET_FRONTIER_VERTICES[] = old
    end
end

@testset "IndicatorResolutions upset active-source delta parity" begin
    P = diamond_poset()
    field = CM.QQField()
    K = CM.coeff_type(field)

    U = [FF.principal_upset(P, 2), FF.principal_upset(P, 3)]
    D = [FF.principal_downset(P, 4)]
    Phi = spzeros(K, 1, 2)
    Phi[1, 1] = CM.coerce(field, 1)
    Phi[1, 2] = CM.coerce(field, 1)
    H = FF.FringeModule{K}(P, U, D, Phi; field=field)
    M = IR.pmodule_from_fringe(H)

    old = IR._INDICATOR_UPSET_ACTIVE_SOURCE_DELTA[]
    try
        IR._INDICATOR_UPSET_ACTIVE_SOURCE_DELTA[] = false
        F_ref, dF_ref = IR.upset_resolution(M; maxlen=3, threads=false)

        IR._INDICATOR_UPSET_ACTIVE_SOURCE_DELTA[] = true
        F_opt, dF_opt = IR.upset_resolution(M; maxlen=3, threads=false)

        @test length(F_opt) == length(F_ref)
        @test dF_opt == dF_ref
    finally
        IR._INDICATOR_UPSET_ACTIVE_SOURCE_DELTA[] = old
    end
end

@testset "IndicatorResolutions fringe wrapper parity" begin
    P = diamond_poset()
    field = CM.QQField()
    K = CM.coeff_type(field)

    U = [FF.principal_upset(P, 2), FF.principal_upset(P, 3)]
    D = [FF.principal_downset(P, 4)]
    Phi = spzeros(K, 1, 2)
    Phi[1, 1] = CM.coerce(field, 1)
    Phi[1, 2] = CM.coerce(field, 1)
    H = FF.FringeModule{K}(P, U, D, Phi; field=field)
    M = IR.pmodule_from_fringe(H)

    F_ref, dF_ref = IR.upset_resolution(M; maxlen=3, threads=false)
    E_ref, dE_ref = IR.downset_resolution(M; maxlen=3, threads=false)
    F, dF, E, dE = IR.indicator_resolutions(H, H; maxlen=3, threads=false)

    @test length(F) == length(F_ref)
    @test dF == dF_ref
    @test length(E) == length(E_ref)
    @test dE == dE_ref
end

@testset "IndicatorResolutions fringe_presentation roundtrip" begin
    P = diamond_poset()
    field = CM.QQField()
    K = CM.coeff_type(field)

    U = [FF.principal_upset(P, 1), FF.principal_upset(P, 2), FF.principal_upset(P, 3)]
    D = [FF.principal_downset(P, 2), FF.principal_downset(P, 3), FF.principal_downset(P, 4)]
    Phi = spzeros(K, 3, 3)
    Phi[1, 1] = CM.coerce(field, 1)
    Phi[2, 1] = CM.coerce(field, 1)
    Phi[2, 3] = CM.coerce(field, 1)
    Phi[3, 2] = CM.coerce(field, 1)
    Phi[3, 3] = CM.coerce(field, 1)
    H = FF.FringeModule{K}(P, U, D, Phi; field=field)

    M = IR.pmodule_from_fringe(H)
    H2 = IR.fringe_presentation(M)
    M2 = IR.pmodule_from_fringe(H2)

    @test H2.P === P
    @test M2.dims == M.dims
    @test M2.edge_maps == M.edge_maps
end

@testset "IndicatorResolutions UX surface" begin
    P = diamond_poset()
    field = CM.QQField()
    K = CM.coeff_type(field)

    U = [FF.principal_upset(P, 1), FF.principal_upset(P, 2), FF.principal_upset(P, 3)]
    D = [FF.principal_downset(P, 2), FF.principal_downset(P, 3), FF.principal_downset(P, 4)]
    Phi = spzeros(K, 3, 3)
    Phi[1, 1] = CM.coerce(field, 1)
    Phi[2, 1] = CM.coerce(field, 1)
    Phi[2, 3] = CM.coerce(field, 1)
    Phi[3, 2] = CM.coerce(field, 1)
    Phi[3, 3] = CM.coerce(field, 1)
    H = FF.FringeModule{K}(P, U, D, Phi; field=field)
    M = IR.pmodule_from_fringe(H)

    cover = IR.projective_cover(M; threads=false)
    @test cover isa IR.ProjectiveCoverResult
    @test TO.describe(cover).kind == :projective_cover
    @test IR.cover_module(cover) === first(cover)
    @test IR.cover_map(cover) === collect(cover)[2]
    @test IR.augmentation(cover) === IR.cover_map(cover)
    @test IR.resolution_generators(cover) === collect(cover)[3]
    @test IR.generator_vertices(IR.resolution_generators(cover)) == IR.generator_vertices(cover.generators)
    @test IR.generator_blocks(IR.resolution_generators(cover)) == IR.generator_blocks(cover.generators)
    @test IR.materialize_generators(cover) == IR.materialize_generators(IR.resolution_generators(cover))
    @test TO.describe(IR.resolution_generators(cover)).side == :projective
    @test IR.resolution_length(cover) == 0
    @test IR.generator_count(cover) == sum(IR.generator_count_by_degree(cover))
    rep_cover = IR.check_projective_cover(cover)
    @test rep_cover.valid
    @test occursin("IndicatorResolutionValidationSummary", sprint(show, IR.indicator_resolution_validation_summary(rep_cover)))
    @test IR.cover_summary(cover).side == :projective
    @test IR.projective_cover(M; output=:summary, threads=false).side == :projective
    @test_throws ArgumentError IR.projective_cover(M; output=:bogus, threads=false)

    hull = IR.injective_hull(M; threads=false)
    @test hull isa IR.InjectiveHullResult
    @test TO.describe(hull).kind == :injective_hull
    @test IR.hull_module(hull) === first(hull)
    @test IR.hull_map(hull) === collect(hull)[2]
    @test IR.coaugmentation(hull) === IR.hull_map(hull)
    @test IR.resolution_generators(hull) === collect(hull)[3]
    @test IR.materialize_generators(hull) == IR.materialize_generators(IR.resolution_generators(hull))
    @test TO.describe(IR.resolution_generators(hull)).side == :injective
    @test IR.resolution_length(hull) == 0
    @test IR.generator_count(hull) == sum(IR.generator_count_by_degree(hull))
    rep_hull = IR.check_injective_hull(hull)
    @test rep_hull.valid
    @test IR.hull_summary(hull).side == :injective
    @test IR.injective_hull(M; output=:summary, threads=false).side == :injective

    up = IR.upset_resolution(M; maxlen=2, threads=false)
    down = IR.downset_resolution(M; maxlen=2, threads=false)
    both = IR.indicator_resolutions(H, H; maxlen=2, threads=false)

    @test up isa IR.UpsetResolutionResult
    @test down isa IR.DownsetResolutionResult
    @test both isa IR.IndicatorResolutionsResult
    @test TO.describe(up).side == :upset
    @test TO.describe(down).side == :downset
    @test TO.describe(both).kind == :indicator_resolutions
    @test TO.describe(IR.projective_resolution(both)) == TO.describe(up)
    @test TO.describe(IR.injective_resolution(both)) == TO.describe(down)
    @test IR.resolution_modules(up) === first(up)
    @test IR.resolution_maps(up) === collect(up)[2]
    @test length(IR.resolution_modules(both).projective) == length(IR.resolution_modules(up))
    @test length(IR.resolution_modules(both).injective) == length(IR.resolution_modules(down))
    @test all(A == B for (A, B) in zip(IR.resolution_maps(both).projective, IR.resolution_maps(up)))
    @test all(A == B for (A, B) in zip(IR.resolution_maps(both).injective, IR.resolution_maps(down)))
    @test all(
        A.U0 == B.U0 && A.U1 == B.U1 && A.delta == B.delta
        for (A, B) in zip(IR.resolution_modules(both).projective, IR.resolution_modules(up))
    )
    @test all(
        A.D0 == B.D0 && A.D1 == B.D1 && A.rho == B.rho
        for (A, B) in zip(IR.resolution_modules(both).injective, IR.resolution_modules(down))
    )
    @test IR.augmentation(up) isa MD.PMorphism
    @test IR.coaugmentation(down) isa MD.PMorphism
    @test eltype(IR.resolution_generators(up)) <: IR.ProjectiveGenerators
    @test eltype(IR.resolution_generators(down)) <: IR.InjectiveGenerators
    @test eltype(IR.resolution_generators(both).projective) <: IR.ProjectiveGenerators
    @test eltype(IR.resolution_generators(both).injective) <: IR.InjectiveGenerators
    @test TO.describe(first(IR.resolution_generators(up))).side == :projective
    @test TO.describe(first(IR.resolution_generators(down))).side == :injective
    @test IR.materialize_generators(up) == [IR.materialize_generators(g) for g in IR.resolution_generators(up)]
    @test IR.materialize_generators(down) == [IR.materialize_generators(g) for g in IR.resolution_generators(down)]
    @test keys(IR.materialize_generators(both)) == (:projective, :injective)
    @test IR.check_resolution(up).valid
    @test IR.check_resolution(down).valid
    @test IR.check_resolution(both).valid
    @test IR.resolution_summary(up).resolution_length == length(IR.resolution_maps(up))
    @test IR.resolution_summary(both).projective.side == :upset
    @test IR.resolution_length(up) == length(IR.resolution_maps(up))
    @test IR.resolution_length(down) == length(IR.resolution_maps(down))
    @test IR.resolution_length(both).projective == IR.resolution_length(up)
    @test IR.generator_count(up) == sum(IR.generator_count_by_degree(up))
    @test IR.generator_count(down) == sum(IR.generator_count_by_degree(down))
    @test IR.generator_count(both).injective == IR.generator_count(down)
    @test IR.upset_resolution(M; maxlen=2, output=:summary, threads=false).side == :upset
    @test IR.downset_resolution(M; maxlen=2, output=:summary, threads=false).side == :downset
    @test IR.indicator_resolutions(H, H; maxlen=2, output=:summary, threads=false).kind == :indicator_resolutions
    @test_throws ArgumentError IR.downset_resolution(M; maxlen=2, output=:bad, threads=false)

    H2 = IR.fringe_presentation(M)
    @test IR.presentation_module(H2) === H2
    @test IR.presentation_map(H2) == FF.fringe_coefficients(H2)
    rep_present = IR.check_fringe_presentation(H2; source=M)
    @test rep_present.valid
    @test IR.presentation_summary(H2).kind == :fringe_presentation

    @test TamerOp.Advanced.cover_module === IR.cover_module
    @test TamerOp.Advanced.hull_module === IR.hull_module
    @test TamerOp.Advanced.resolution_modules === IR.resolution_modules
    @test TamerOp.Advanced.hull_summary === IR.hull_summary
    @test TamerOp.Advanced.presentation_summary === IR.presentation_summary
    @test TamerOp.Advanced.materialize_generators === IR.materialize_generators
    @test TamerOp.Advanced.resolution_length === IR.resolution_length
    @test TamerOp.Advanced.generator_count === IR.generator_count
    @test TamerOp.Advanced.check_projective_cover === IR.check_projective_cover
    @test TamerOp.Advanced.indicator_resolution_validation_summary === IR.indicator_resolution_validation_summary
end

@testset "IndicatorResolutions cache miss lifecycle parity" begin
    P = diamond_poset()
    field = CM.QQField()
    K = CM.coeff_type(field)

    U = [FF.principal_upset(P, 2), FF.principal_upset(P, 3)]
    D = [FF.principal_downset(P, 4)]
    Phi = spzeros(K, 1, 2)
    Phi[1, 1] = CM.coerce(field, 1)
    Phi[1, 2] = CM.coerce(field, 1)
    H = FF.FringeModule{K}(P, U, D, Phi; field=field)

    rc_fresh = CM.ResolutionCache()
    F_fresh, dF_fresh, E_fresh, dE_fresh = IR.indicator_resolutions(H, H; maxlen=3, threads=false, cache=rc_fresh)

    rc_clear = CM.ResolutionCache()
    CM._clear_resolution_cache!(rc_clear)
    F_clear, dF_clear, E_clear, dE_clear = IR.indicator_resolutions(H, H; maxlen=3, threads=false, cache=rc_clear)

    @test length(F_fresh) == length(F_clear)
    @test dF_fresh == dF_clear
    @test length(E_fresh) == length(E_clear)
    @test dE_fresh == dE_clear
end

@testset "IndicatorResolutions adaptive cache admission policy" begin
    P = diamond_poset()
    field = CM.QQField()
    K = CM.coeff_type(field)

    U = [FF.principal_upset(P, 2), FF.principal_upset(P, 3)]
    D = [FF.principal_downset(P, 4)]
    Phi = spzeros(K, 1, 2)
    Phi[1, 1] = CM.coerce(field, 1)
    Phi[1, 2] = CM.coerce(field, 1)
    H = FF.FringeModule{K}(P, U, D, Phi; field=field)

    payload = IR.indicator_resolutions(H, H; maxlen=3, threads=false)
    cache_payload = IR._indicator_resolutions_cache_payload(payload)
    key = CM._resolution_key3(H, H, 3)
    PT = typeof(P)
    UP = TamerOp.IndicatorTypes.UpsetPresentation{K,PT,Nothing,SparseMatrixCSC{K,Int}}
    DP = TamerOp.IndicatorTypes.DownsetCopresentation{K,PT,Nothing,SparseMatrixCSC{K,Int}}
    cache_val_type = Tuple{
        Vector{UP},
        Vector{SparseMatrixCSC{K,Int}},
        Vector{DP},
        Vector{SparseMatrixCSC{K,Int}},
    }

    rc = CM.ResolutionCache()
    stored = IR._resolution_cache_indicator_store!(rc, key, cache_payload)
    @test stored === cache_payload
    @test IR._resolution_cache_indicator_get(rc, key, cache_val_type) === cache_payload
    @test rc.indicator_primary_type === typeof(cache_payload)
    @test eltype(cache_payload[1]) === UP
    @test eltype(cache_payload[3]) === DP

    other_key = CM._resolution_key3(H, H, 4)
    @test IR._resolution_cache_indicator_store!(rc, other_key, 17) == 17
    @test IR._resolution_cache_indicator_get(rc, other_key, Int) == 17
end

@testset "IndicatorResolutions threaded resolution parity" begin
    Threads.nthreads() > 1 || begin
        @test true
        return
    end

    P = diamond_poset()
    field = CM.QQField()
    K = CM.coeff_type(field)

    U = [FF.principal_upset(P, 2), FF.principal_upset(P, 3)]
    D = [FF.principal_downset(P, 4)]
    Phi = spzeros(K, 1, 2)
    Phi[1, 1] = CM.coerce(field, 1)
    Phi[1, 2] = CM.coerce(field, 1)
    H = FF.FringeModule{K}(P, U, D, Phi; field=field)
    M = IR.pmodule_from_fringe(H)

    FF.build_cache!(P; cover=true, updown=true)
    cc = MD._get_cover_cache(P)
    memo = IR._indicator_new_array_memo(K, FF.nvertices(P))
    ws = IR._new_resolution_workspace(K, FF.nvertices(P))

    F_ref, dF_ref = IR.upset_resolution(M; maxlen=3, threads=false)
    E_ref, dE_ref = IR.downset_resolution(M; maxlen=3, threads=false)

    F_thr, dF_thr = IR.upset_resolution(
        M;
        maxlen=3,
        cache=cc,
        map_memo=memo,
        workspace=ws,
        threads=true,
    )
    E_thr, dE_thr = IR.downset_resolution(
        M;
        maxlen=3,
        cache=cc,
        map_memo=memo,
        workspace=ws,
        threads=true,
    )

    @test length(F_thr) == length(F_ref)
    @test dF_thr == dF_ref
    @test length(E_thr) == length(E_ref)
    @test dE_thr == dE_ref
end

@testset "IndicatorResolutions batched map_leq cache helper parity" begin
    P = chain_poset(4)
    field = CM.QQField()
    K = CM.coeff_type(field)

    U = [FF.principal_upset(P, 1), FF.principal_upset(P, 2)]
    D = [FF.principal_downset(P, 3), FF.principal_downset(P, 4)]
    Phi = spzeros(K, 2, 2)
    Phi[1, 1] = CM.coerce(field, 1)
    Phi[2, 2] = CM.coerce(field, 1)
    H = FF.FringeModule{K}(P, U, D, Phi; field=field)
    M = IR.pmodule_from_fringe(H)

    FF.build_cache!(P; cover=true, updown=true)
    cc = MD._get_cover_cache(P)
    memo = IR._indicator_new_array_memo(K, FF.nvertices(P))

    pairs = Tuple{Int,Int}[(1, 2), (1, 4), (2, 4), (1, 4)]
    mats = IR._map_leq_cached_many_indicator(M, pairs, cc, memo)
    for i in eachindex(pairs)
        u, v = pairs[i]
        @test mats[i] == MD.map_leq(M, u, v; cache=cc)
        @test IR._indicator_memo_get(memo, FF.nvertices(P), u, v) !== nothing
    end
end

@testset "IndicatorResolutions map_leq_fill_memo threshold paths parity" begin
    P = chain_poset(4)
    field = CM.QQField()
    K = CM.coeff_type(field)

    U = [FF.principal_upset(P, 1), FF.principal_upset(P, 2)]
    D = [FF.principal_downset(P, 3), FF.principal_downset(P, 4)]
    Phi = spzeros(K, 2, 2)
    Phi[1, 1] = CM.coerce(field, 1)
    Phi[2, 2] = CM.coerce(field, 1)
    H = FF.FringeModule{K}(P, U, D, Phi; field=field)
    M = IR.pmodule_from_fringe(H)

    FF.build_cache!(P; cover=true, updown=true)
    cc = MD._get_cover_cache(P)
    n = FF.nvertices(P)
    memo = IR._indicator_new_array_memo(K, n)
    ws = IR._new_resolution_workspace(K, n)

    small_pairs = Tuple{Int,Int}[(1, 2), (1, 4)]
    IR._map_leq_fill_memo_indicator!(M, small_pairs, cc, memo, ws)
    for (u, v) in small_pairs
        @test IR._indicator_memo_get(memo, n, u, v) == MD.map_leq(M, u, v; cache=cc)
    end

    fill!(memo, nothing)
    large_pairs = Tuple{Int,Int}[(1, 2), (1, 3), (1, 4), (2, 3), (2, 4)]
    IR._map_leq_fill_memo_indicator!(M, large_pairs, cc, memo, ws)
    for (u, v) in large_pairs
        @test IR._indicator_memo_get(memo, n, u, v) == MD.map_leq(M, u, v; cache=cc)
    end
end

@testset "AbelianCategories structural selector kernel/cokernel path" begin
    P = chain_poset(2)
    field = CM.QQField()
    K = CM.coeff_type(field)
    I2 = Matrix{K}(I, 2, 2)

    edge_dom = Dict{Tuple{Int,Int}, Matrix{K}}((1, 2) => I2)
    edge_cod = Dict{Tuple{Int,Int}, Matrix{K}}((1, 2) => I2)
    Dom = MD.PModule{K}(P, [2, 2], edge_dom; field=field)
    Cod = MD.PModule{K}(P, [2, 2], edge_cod; field=field)

    S = zeros(K, 2, 2)
    S[1, 1] = one(K)
    f = MD.PMorphism{K}(Dom, Cod, [S, S])

    AB = TamerOp.AbelianCategories
    @test AB._is_partial_permutation(field, S)

    Kmod, iota = AB.kernel_with_inclusion(f)
    Cmod, q = AB._cokernel_module(f)

    @test Kmod.dims == [1, 1]
    @test Cmod.dims == [1, 1]
    expected_i = zeros(K, 2, 1)
    expected_i[2, 1] = one(K)
    expected_q = zeros(K, 1, 2)
    expected_q[1, 2] = one(K)
    @test iota.comps[1] == expected_i
    @test iota.comps[2] == expected_i
    @test q.comps[1] == expected_q
    @test q.comps[2] == expected_q

    # Projection+selector composite case: repeated row usage.
    T = zeros(K, 2, 2)
    T[1, 1] = one(K)
    T[1, 2] = one(K)
    g = MD.PMorphism{K}(Dom, Cod, [T, T])
    K2, i2 = AB.kernel_with_inclusion(g)
    C2, q2 = AB._cokernel_module(g)
    @test K2.dims == [1, 1]
    @test C2.dims == [1, 1]
    @test size(i2.comps[1]) == (2, 1)
    @test size(q2.comps[1]) == (1, 2)
    @test g.comps[1] * i2.comps[1] == zeros(K, 2, 1)
end

@testset "AbelianCategories incremental kernel/cokernel cache parity" begin
    P = chain_poset(2)
    field = CM.QQField()
    K = CM.coeff_type(field)

    I2 = Matrix{K}(I, 2, 2)
    I3 = Matrix{K}(I, 3, 3)

    dom1 = MD.PModule{K}(P, [2, 2], Dict{Tuple{Int,Int}, Matrix{K}}((1, 2) => I2); field=field)
    cod1 = MD.PModule{K}(P, [2, 2], Dict{Tuple{Int,Int}, Matrix{K}}((1, 2) => I2); field=field)
    S = zeros(K, 2, 2)
    S[1, 1] = one(K)
    f1 = MD.PMorphism{K}(dom1, cod1, [S, S])

    dom2 = MD.PModule{K}(P, [3, 3], Dict{Tuple{Int,Int}, Matrix{K}}((1, 2) => I3); field=field)
    cod2 = MD.PModule{K}(P, [3, 3], Dict{Tuple{Int,Int}, Matrix{K}}((1, 2) => I3); field=field)
    S3 = zeros(K, 3, 3)
    S3[1:2, 1:2] .= S
    f2 = MD.PMorphism{K}(dom2, cod2, [S3, S3])

    AB = TamerOp.AbelianCategories
    kcache = Any[]
    AB.kernel_with_inclusion(f1; incremental_cache=kcache)
    K_inc, i_inc = AB.kernel_with_inclusion(f2; incremental_cache=kcache)
    K_ref, i_ref = AB.kernel_with_inclusion(f2)
    @test K_inc.dims == K_ref.dims
    @test i_inc.comps == i_ref.comps

    m1 = MD.PModule{K}(P, [2, 2], Dict{Tuple{Int,Int}, Matrix{K}}((1, 2) => I2); field=field)
    e1 = MD.PModule{K}(P, [2, 2], Dict{Tuple{Int,Int}, Matrix{K}}((1, 2) => I2); field=field)
    iota1 = MD.PMorphism{K}(m1, e1, [S, S])

    m2 = MD.PModule{K}(P, [3, 3], Dict{Tuple{Int,Int}, Matrix{K}}((1, 2) => I3); field=field)
    e2 = MD.PModule{K}(P, [3, 3], Dict{Tuple{Int,Int}, Matrix{K}}((1, 2) => I3); field=field)
    iota2 = MD.PMorphism{K}(m2, e2, [S3, S3])

    ccache = Any[]
    AB._cokernel_module(iota1; incremental_cache=ccache)
    C_inc, q_inc = AB._cokernel_module(iota2; incremental_cache=ccache)
    C_ref, q_ref = AB._cokernel_module(iota2)
    @test C_inc.dims == C_ref.dims
    @test q_inc.comps == q_ref.comps
end

@testset "IndicatorResolutions map batch plan reuse in workspace" begin
    P = diamond_poset()
    field = CM.QQField()
    K = CM.coeff_type(field)

    U = [FF.principal_upset(P, 2), FF.principal_upset(P, 3)]
    D = [FF.principal_downset(P, 4)]
    Phi = spzeros(K, 1, 2)
    Phi[1, 1] = CM.coerce(field, 1)
    Phi[1, 2] = CM.coerce(field, 1)
    H = FF.FringeModule{K}(P, U, D, Phi; field=field)
    M = IR.pmodule_from_fringe(H)

    ws = IR._new_resolution_workspace(K, FF.nvertices(P))
    memo = IR._indicator_new_array_memo(K, FF.nvertices(P))
    old_thresh = IR.INDICATOR_MAP_BATCH_THRESHOLD[]
    try
        IR.INDICATOR_MAP_BATCH_THRESHOLD[] = 1
        IR.projective_cover(M; map_memo=memo, workspace=ws, threads=false)
        n1 = length(ws.map_batch_cache)
        IR.projective_cover(M; map_memo=memo, workspace=ws, threads=false)
        n2 = length(ws.map_batch_cache)
        IR.injective_hull(M; map_memo=memo, workspace=ws, threads=false)
        n3 = length(ws.map_batch_cache)
        @test n1 > 0
        @test n2 == n1
        @test n3 >= n2
    finally
        IR.INDICATOR_MAP_BATCH_THRESHOLD[] = old_thresh
    end
end

@testset "IndicatorResolutions incremental linalg gate parity" begin
    P = diamond_poset()
    field = CM.QQField()
    K = CM.coeff_type(field)

    U = [FF.principal_upset(P, 2), FF.principal_upset(P, 3)]
    D = [FF.principal_downset(P, 4)]
    Phi = spzeros(K, 1, 2)
    Phi[1, 1] = CM.coerce(field, 1)
    Phi[1, 2] = CM.coerce(field, 1)
    H = FF.FringeModule{K}(P, U, D, Phi; field=field)
    M = IR.pmodule_from_fringe(H)

    old_up = IR.INDICATOR_INCREMENTAL_LINALG[]
    old_down_maps = IR.INDICATOR_DOWNSET_INCREMENTAL_LINALG_MIN_MAPS[]
    old_down_entries = IR.INDICATOR_DOWNSET_INCREMENTAL_LINALG_MIN_ENTRIES[]
    try
        IR.INDICATOR_INCREMENTAL_LINALG[] = false
        IR.INDICATOR_DOWNSET_INCREMENTAL_LINALG_MIN_MAPS[] = typemax(Int)
        IR.INDICATOR_DOWNSET_INCREMENTAL_LINALG_MIN_ENTRIES[] = typemax(Int)
        F_off, dF_off = IR.upset_resolution(M; maxlen=2, threads=false)
        E_off, dE_off = IR.downset_resolution(M; maxlen=2, threads=false)

        IR.INDICATOR_INCREMENTAL_LINALG[] = true
        IR.INDICATOR_DOWNSET_INCREMENTAL_LINALG_MIN_MAPS[] = 1
        IR.INDICATOR_DOWNSET_INCREMENTAL_LINALG_MIN_ENTRIES[] = 0
        F_on, dF_on = IR.upset_resolution(M; maxlen=2, threads=false)
        E_on, dE_on = IR.downset_resolution(M; maxlen=2, threads=false)

        @test length(F_off) == length(F_on)
        @test dF_off == dF_on
        @test length(E_off) == length(E_on)
        @test dE_off == dE_on
    finally
        IR.INDICATOR_INCREMENTAL_LINALG[] = old_up
        IR.INDICATOR_DOWNSET_INCREMENTAL_LINALG_MIN_MAPS[] = old_down_maps
        IR.INDICATOR_DOWNSET_INCREMENTAL_LINALG_MIN_ENTRIES[] = old_down_entries
    end
end

@testset "IndicatorResolutions prefix cache extends maxlen incrementally" begin
    P = diamond_poset()
    field = CM.QQField()
    K = CM.coeff_type(field)

    U = [FF.principal_upset(P, 2), FF.principal_upset(P, 3)]
    D = [FF.principal_downset(P, 4)]
    Phi = spzeros(K, 1, 2)
    Phi[1, 1] = CM.coerce(field, 1)
    Phi[1, 2] = CM.coerce(field, 1)
    H = FF.FringeModule{K}(P, U, D, Phi; field=field)
    M0 = IR.pmodule_from_fringe(H)
    M = M0
    for _ in 1:5
        M = MD.direct_sum(M, M0)
    end

    old_up_prefix = IR.INDICATOR_PREFIX_CACHE_ENABLED[]
    old_up_steps = IR.INDICATOR_UPSET_PREFIX_CACHE_MIN_STEPS[]
    old_up_vertices = IR.INDICATOR_UPSET_PREFIX_CACHE_MIN_VERTICES[]
    old_up_dims = IR.INDICATOR_UPSET_PREFIX_CACHE_MIN_TOTAL_DIMS[]
    try
        IR.INDICATOR_PREFIX_CACHE_ENABLED[] = true
        IR.INDICATOR_UPSET_PREFIX_CACHE_MIN_STEPS[] = 1
        IR.INDICATOR_UPSET_PREFIX_CACHE_MIN_VERTICES[] = 1
        IR.INDICATOR_UPSET_PREFIX_CACHE_MIN_TOTAL_DIMS[] = 0
        IR._clear_indicator_prefix_caches!()
        F1, dF1 = IR.upset_resolution(M; maxlen=1, threads=false)
        E1, dE1 = IR.downset_resolution(M; maxlen=1, threads=false)
        @test length(dF1) <= 1
        @test length(dE1) <= 1
        @test IR._upset_prefix_steps(M) == length(dF1)

        F2, dF2 = IR.upset_resolution(M; maxlen=2, threads=false)
        E2, dE2 = IR.downset_resolution(M; maxlen=2, threads=false)
        @test length(F2) >= length(F1)
        @test length(E2) >= length(E1)
        @test length(dF2) >= length(dF1)
        @test length(dE2) >= length(dE1)
        @test IR._upset_prefix_steps(M) == length(dF2)
    finally
        IR.INDICATOR_PREFIX_CACHE_ENABLED[] = old_up_prefix
        IR.INDICATOR_UPSET_PREFIX_CACHE_MIN_STEPS[] = old_up_steps
        IR.INDICATOR_UPSET_PREFIX_CACHE_MIN_VERTICES[] = old_up_vertices
        IR.INDICATOR_UPSET_PREFIX_CACHE_MIN_TOTAL_DIMS[] = old_up_dims
    end
end

@testset "IndicatorResolutions thread gate tiny-case policy" begin
    @test IR._indicator_use_threads(true, 8, 32, 128) == false
    @test IR._indicator_use_threads(false, 10_000, 10_000, 10^8) == false
end

@testset "Cover-edge maps are label-consistent on non-chain posets" begin
    # Poset with relations: 1<3<4 and 2<4 (2 incomparable with 3)
    leq = falses(4,4)
    for i in 1:4
        leq[i,i] = true
    end
    leq[1,3] = true
    leq[3,4] = true
    leq[1,4] = true
    leq[2,4] = true
    P = FF.FinitePoset(leq)

    # A tiny fringe module that typically forces generators at vertices 2 and 3.
    U = [FF.principal_upset(P, 2), FF.principal_upset(P, 3)]
    D = [FF.principal_downset(P, 4)]
    field = CM.QQField()
    K = CM.coeff_type(field)
    Phi = spzeros(K, 1, 2)
    Phi[1,1] = CM.coerce(field, 1)
    Phi[1,2] = CM.coerce(field, 1)
    H = FF.FringeModule{K}(P, U, D, Phi; field=field)

    M = IR.pmodule_from_fringe(H)

    # Projective cover must be a P-module morphism.
    F0, pi0, _ = IR.projective_cover(M)
    C = FF.cover_edges(P)
    for (u, v) in C
        lhs = M.edge_maps[u, v] * pi0.comps[u]
        rhs = pi0.comps[v] * F0.edge_maps[u, v]
        @test lhs == rhs
    end

    # Injective hull inclusion must be a P-module morphism.
    E, iota, _ = IR._injective_hull(M)
    for (u, v) in C
        lhs = E.edge_maps[u, v] * iota.comps[u]
        rhs = iota.comps[v] * M.edge_maps[u, v]
        @test lhs == rhs
    end
end

@testset "verify_upset_resolution / verify_downset_resolution catch illegal monomial support" begin
    P = diamond_poset()
    field = CM.QQField()
    K = CM.coeff_type(field)

    # Deliberately invalid "resolution step": for upset resolutions, nonzero in delta (row i, col j)
    # requires U_row subset U_col. We violate that on the diamond poset.
    U2 = FF.principal_upset(P, 2)
    U3 = FF.principal_upset(P, 3)

    F0 = IR.UpsetPresentation{K}(P, [U2, U3], FF.Upset[], spzeros(K, 0, 2), nothing)
    F1 = IR.UpsetPresentation{K}(P, [U2],     FF.Upset[], spzeros(K, 0, 1), nothing)
    F_infer = IR.UpsetPresentation(P, [U2], FF.Upset[], spzeros(K, 0, 1), nothing)
    F_coerced = IR.UpsetPresentation(P, [U2], FF.Upset[], spzeros(Float64, 0, 1), nothing; field=field)
    @test eltype(F_infer.delta) == K
    @test eltype(F_coerced.delta) == K
    @test F_infer isa IR.UpsetPresentation{K,typeof(P),Nothing,SparseMatrixCSC{K,Int}}
    @test F_coerced isa IR.UpsetPresentation{K,typeof(P),Nothing,SparseMatrixCSC{K,Int}}
    @test_throws MethodError IR.UpsetPresentation{K}(P, [U2], FF.Upset[], spzeros(K, 0, 1), nothing; field=field)

    # delta is |U1| x |U0| = 1 x 2. Put a nonzero at (U2 row, U3 col).
    # Since U2 is not a subset of U3, this must be rejected.
    delta_bad = spzeros(K, 1, 2)
    delta_bad[1, 2] = CM.coerce(field, 1)

    @test_throws ErrorException IR.verify_upset_resolution([F0, F1], [delta_bad];
        check_d2=false, check_exactness=false)

    # Dual check for downset copresentations: nonzero in rho (row i, col j) requires
    # D_row subset D_col, where rows come from the later stage.
    D2 = FF.principal_downset(P, 2)
    D3 = FF.principal_downset(P, 3)

    E0 = IR.DownsetCopresentation{K}(P, [D3], FF.Downset[], spzeros(K, 0, 1), nothing)
    E1 = IR.DownsetCopresentation{K}(P, [D2], FF.Downset[], spzeros(K, 0, 1), nothing)
    E_infer = IR.DownsetCopresentation(P, [D2], FF.Downset[], spzeros(K, 0, 1), nothing)
    E_coerced = IR.DownsetCopresentation(P, [D2], FF.Downset[], spzeros(Float64, 0, 1), nothing; field=field)
    @test eltype(E_infer.rho) == K
    @test eltype(E_coerced.rho) == K
    @test E_infer isa IR.DownsetCopresentation{K,typeof(P),Nothing,SparseMatrixCSC{K,Int}}
    @test E_coerced isa IR.DownsetCopresentation{K,typeof(P),Nothing,SparseMatrixCSC{K,Int}}
    @test_throws MethodError IR.DownsetCopresentation{K}(P, [D2], FF.Downset[], spzeros(K, 0, 1), nothing; field=field)

    rho_bad = spzeros(K, 1, 1)
    rho_bad[1, 1] = CM.coerce(field, 1)  # D2 is not a subset of D3

    @test_throws ErrorException IR.verify_downset_resolution([E0, E1], [rho_bad];
        check_d2=false, check_exactness=false)
end

@testset "IndicatorTypes inspection and validation UX" begin
    IT = TamerOp.IndicatorTypes
    P = chain_poset(3)
    P2 = chain_poset(3)
    U1 = FF.principal_upset(P, 1)
    U2 = FF.principal_upset(P, 2)
    D2 = FF.principal_downset(P, 2)
    D3 = FF.principal_downset(P, 3)

    delta = reshape(QQ[1], 1, 1)
    rho = reshape(QQ[1], 1, 1)

    F = IT.UpsetPresentation{QQ}(P, [U1], [U2], delta, nothing)
    E = IT.DownsetCopresentation{QQ}(P, [D2], [D3], rho, nothing)

    @test TO.ambient_poset(F) === P
    @test TO.base_poset(F) === P
    @test TO.field(F) == CM.QQField()
    @test TO.describe(F).kind == :upset_presentation
    @test TO.describe(F).ngenerators == 1
    @test IT.generator_labels(F) == [U1]
    @test IT.relation_labels(F) == [U2]
    @test IT.presentation_matrix(F) == delta
    @test IT.attached_fringe(F) === nothing
    @test sprint(show, F) == "UpsetPresentation(field=QQ, nvertices=3, ngenerators=1, nrelations=1)"
    @test occursin("matrix_size: (1, 1)", repr("text/plain", F))

    @test TO.ambient_poset(E) === P
    @test TO.base_poset(E) === P
    @test TO.field(E) == CM.QQField()
    @test TO.describe(E).kind == :downset_copresentation
    @test TO.describe(E).ncogenerators == 1
    @test IT.cogenerator_labels(E) == [D2]
    @test IT.corelation_labels(E) == [D3]
    @test IT.copresentation_matrix(E) == rho
    @test IT.attached_fringe(E) === nothing
    @test sprint(show, E) == "DownsetCopresentation(field=QQ, nvertices=3, ncogenerators=1, ncorelations=1)"
    @test occursin("matrix_size: (1, 1)", repr("text/plain", E))

    repF = IT.check_upset_presentation(F)
    repE = IT.check_downset_copresentation(E)
    @test repF.valid
    @test repE.valid
    @test IT.check_upset_presentation(F; throw=true).valid
    @test IT.check_downset_copresentation(E; throw=true).valid

    Ubad = FF.principal_upset(P2, 1)
    Dbad = FF.principal_downset(P2, 2)
    F_bad = IT.UpsetPresentation{QQ,typeof(P),Nothing,Matrix{QQ}}(P, [Ubad], [U2], delta, nothing)
    E_bad = IT.DownsetCopresentation{QQ,typeof(P),Nothing,Matrix{QQ}}(P, [Dbad], [D3], rho, nothing)
    repF_bad = IT.check_upset_presentation(F_bad)
    repE_bad = IT.check_downset_copresentation(E_bad)
    @test !repF_bad.valid
    @test !repE_bad.valid
    @test any(occursin("ambient poset", s) || occursin("does not belong to the ambient poset", s) for s in repF_bad.issues)
    @test any(occursin("ambient poset", s) || occursin("does not belong to the ambient poset", s) for s in repE_bad.issues)
    @test_throws ArgumentError IT.check_upset_presentation(F_bad; throw=true)
    @test_throws ArgumentError IT.check_downset_copresentation(E_bad; throw=true)

    @test TOA.UpsetPresentation === IT.UpsetPresentation
    @test TOA.DownsetCopresentation === IT.DownsetCopresentation
    @test TOA.check_upset_presentation === IT.check_upset_presentation
    @test TOA.check_downset_copresentation === IT.check_downset_copresentation
    @test TOA.generator_labels === IT.generator_labels
    @test TOA.copresentation_matrix === IT.copresentation_matrix
end


@testset "Indicator resolutions on the diamond poset (by-hand checks)" begin
    # -------------------------------------------------------------------------
    # The diamond poset (a.k.a. the rank-2 Boolean lattice) is the first place
    # where projective/injective resolutions can have length 2 (not just 1),
    # because there are two different length-2 chains from bottom to top:
    #   1 -> 2 -> 4
    #   1 -> 3 -> 4
    #
    # In the category algebra kP (representations of the poset), this produces
    # a genuine relation between the two composites, and it manifests as:
    #   Ext^2(S1, S4) = 1
    # while Ext^1(S1, S4) = 0 (since 1<4 is not a cover).
    #
    # These tests check:
    #  * the shape of the computed resolutions for S1 and S4 (length 2),
    #  * the "by hand" pattern of the differentials at the coefficient-matrix
    #    level (equal-entry and sum-to-zero patterns),
    #  * Ext^0/1/2 between ALL simples on the diamond against an interval-based
    #    computation (reduced cohomology in degrees -1 and 0).
    # -------------------------------------------------------------------------

    P = diamond_poset()
    field = CM.QQField()
    K = CM.coeff_type(field)

    # Sanity: cover edges should be exactly 1->2, 1->3, 2->4, 3->4.
    C = FF.cover_edges(P)
    @test C[1, 2] == true
    @test C[1, 3] == true
    @test C[2, 4] == true
    @test C[3, 4] == true
    @test C[1, 4] == false
    @test C[2, 3] == false
    @test C[3, 2] == false

    # Sanity: principal upsets/downsets on a non-chain poset.
    @test FF.principal_upset(P, 1).mask == BitVector([true,  true,  true,  true])
    @test FF.principal_upset(P, 2).mask == BitVector([false, true,  false, true])
    @test FF.principal_upset(P, 3).mask == BitVector([false, false, true,  true])
    @test FF.principal_upset(P, 4).mask == BitVector([false, false, false, true])

    @test FF.principal_downset(P, 1).mask == BitVector([true,  false, false, false])
    @test FF.principal_downset(P, 2).mask == BitVector([true,  true,  false, false])
    @test FF.principal_downset(P, 3).mask == BitVector([true,  false, true,  false])
    @test FF.principal_downset(P, 4).mask == BitVector([true,  true,  true,  true])

    # Simple module S_p at vertex p: support only at p, with all structure maps zero.
    simple_at(p::Int) = one_by_one_fringe(P, FF.principal_upset(P, p), FF.principal_downset(P, p); scalar=one(K), field=field)
    S = [simple_at(p) for p in 1:P.n]
    S1, S2, S3, S4 = S

    # -------------------------------------------------------------------------
    # (A) Resolution shape checks for the "interesting" case S1 and S4.
    # -------------------------------------------------------------------------

    # Projective (upset) resolution of S1 should have length 2:
    #   F2 -> F1 -> F0 -> S1 -> 0
    # with:
    #   U0 = [Up(1)]
    #   U1 = [Up(2), Up(3)]
    #   U2 = [Up(4)]
    F, dF = IR.upset_resolution(S1; maxlen=10)

    @test length(F) == 3
    @test length(dF) == 2

    # Verify structural correctness of the upset resolution (d^2=0 + exactness).
    @test IR.verify_upset_resolution(F, dF)

    U_by_a = [f.U0 for f in F]
    @test length(U_by_a[1]) == 1
    @test U_by_a[1][1].mask == FF.principal_upset(P, 1).mask

    @test length(U_by_a[2]) == 2
    @test U_by_a[2][1].mask == FF.principal_upset(P, 2).mask
    @test U_by_a[2][2].mask == FF.principal_upset(P, 3).mask

    @test length(U_by_a[3]) == 1
    @test U_by_a[3][1].mask == FF.principal_upset(P, 4).mask

    # Differential patterns, robust to a global nonzero scalar choice:
    #
    # delta0 : U1 -> U0 should send both branch generators (at 2 and 3)
    # to the unique generator at 1 with the SAME coefficient.
    #
    # delta1 : U2 -> U1 encodes the relation between the two branches at the top;
    # the two coefficients must sum to zero (so they are negatives of each other).
    delta0 = dF[1]
    delta1 = dF[2]

    @test size(delta0) == (2, 1)
    @test size(delta1) == (1, 2)

    a = delta0[1, 1]
    b = delta0[2, 1]
    @test a != 0 && b != 0
    @test a == b

    c = delta1[1, 1]
    d = delta1[1, 2]
    @test c != 0 && d != 0
    @test c + d == 0

    # Composition must be 0 (also checked by verify_upset_resolution, but this is "by hand").
    @test Matrix(delta1 * delta0) == CM.zeros(field, 1, 1)

    # Injective (downset) resolution of S4 should have length 2:
    #   0 -> S4 -> E0 -> E1 -> E2
    # with:
    #   D0 = [Down(4)]
    #   D1 = [Down(2), Down(3)]
    #   D2 = [Down(1)]
    E, dE = IR.downset_resolution(S4; maxlen=10)

    @test length(E) == 3
    @test length(dE) == 2

    # Verify structural correctness of the downset resolution.
    @test IR.verify_downset_resolution(E, dE)

    D_by_b = [e.D0 for e in E]
    @test length(D_by_b[1]) == 1
    @test D_by_b[1][1].mask == FF.principal_downset(P, 4).mask

    @test length(D_by_b[2]) == 2
    @test D_by_b[2][1].mask == FF.principal_downset(P, 2).mask
    @test D_by_b[2][2].mask == FF.principal_downset(P, 3).mask

    @test length(D_by_b[3]) == 1
    @test D_by_b[3][1].mask == FF.principal_downset(P, 1).mask

    rho0 = dE[1]
    rho1 = dE[2]

    @test size(rho0) == (2, 1)
    @test size(rho1) == (1, 2)

    r1 = rho0[1, 1]
    r2 = rho0[2, 1]
    @test r1 != 0 && r2 != 0
    @test r1 == r2

    s1 = rho1[1, 1]
    s2 = rho1[1, 2]
    @test s1 != 0 && s2 != 0
    @test s1 + s2 == 0

    @test Matrix(rho1 * rho0) == CM.zeros(field, 1, 1)

    # -------------------------------------------------------------------------
    # (B) Ext^0/1/2 between ALL simple modules on the diamond, computed "by hand"
    #     from the open interval (x,y).
    #
    # For degree 1 and 2 on simples, we only need:
    #   - Ext^1 corresponds to reduced H^{-1} of the order complex of (x,y),
    #     which is 1 iff (x,y) is empty (i.e. x<y is a cover), else 0.
    #   - Ext^2 corresponds to reduced H^0 of that order complex, i.e.
    #       (#connected components of the open interval) - 1,
    #     and is 0 for empty intervals.
    #
    # On the diamond: (1,4) = {2,3} has 2 components, so Ext^2(S1,S4)=1.
    # -------------------------------------------------------------------------

    function strict_interval(P::FF.FinitePoset, x::Int, y::Int)
        if x == y || !FF.leq(P, x, y)
            return Int[]
        end
        return [z for z in 1:P.n if z != x && z != y && FF.leq(P, x, z) && FF.leq(P, z, y)]
    end

    # Count connected components in the induced Hasse graph on 'verts' (undirected).
    function interval_components(P::FF.FinitePoset, verts::Vector{Int})
        isempty(verts) && return 0

        inV = falses(P.n)
        for v in verts
            inV[v] = true
        end

        C = FF.cover_edges(P)
        adj = [Int[] for _ in 1:P.n]
        for u in 1:P.n, v in 1:P.n
            if C[u, v] && inV[u] && inV[v]
                push!(adj[u], v)
                push!(adj[v], u)
            end
        end

        seen = falses(P.n)
        comps = 0
        for v in verts
            if seen[v]
                continue
            end
            comps += 1
            stack = [v]
            seen[v] = true
            while !isempty(stack)
                a = pop!(stack)
                for b in adj[a]
                    if !seen[b]
                        seen[b] = true
                        push!(stack, b)
                    end
                end
            end
        end
        return comps
    end

    function expected_ext_dims_simple(P::FF.FinitePoset, x::Int, y::Int)
        # Expected (Ext^0, Ext^1, Ext^2) for simples at x and y.
        if x == y
            return (ext0 = 1, ext1 = 0, ext2 = 0)
        end
        if !FF.leq(P, x, y)
            return (ext0 = 0, ext1 = 0, ext2 = 0)
        end

        verts = strict_interval(P, x, y)
        if isempty(verts)
            # Cover relation => reduced H^{-1}(empty) = k => Ext^1 = 1.
            return (ext0 = 0, ext1 = 1, ext2 = 0)
        end

        # Nonempty interval => Ext^1 = 0 and Ext^2 = reduced H^0 = components-1.
        c = interval_components(P, verts)
        return (ext0 = 0, ext1 = 0, ext2 = max(c - 1, 0))
    end

    for x in 1:P.n, y in 1:P.n
        ext_xy = DF.ext_dimensions_via_indicator_resolutions(S[x], S[y]; maxlen=6)
        exp = expected_ext_dims_simple(P, x, y)

        @test get(ext_xy, 0, 0) == exp.ext0
        @test get(ext_xy, 1, 0) == exp.ext1
        @test get(ext_xy, 2, 0) == exp.ext2

        # On this poset, the only possible nonzero higher group for simples would
        # come from higher reduced cohomology of intervals, which does not occur here.
        @test get(ext_xy, 3, 0) == 0
        @test get(ext_xy, 4, 0) == 0
    end

    # -------------------------------------------------------------------------
    # (C) First-page vs full Ext: S1 -> S4 is the motivating case.
    # The "one-step" data cannot see Ext^2, but the full resolution must.
    # -------------------------------------------------------------------------
    F1 = IR.upset_presentation_one_step(S1)
    E1 = IR.downset_copresentation_one_step(S4)
    hom0, ext1 = DF.hom_ext_first_page(F1, E1)

    @test hom0 == 0
    @test ext1 == 0

    ext_full = DF.ext_dimensions_via_indicator_resolutions(S1, S4; maxlen=6)
    @test get(ext_full, 2, 0) == 1
end


@testset "CoverCache memoization" begin
    P = chain_poset(5)

    # Clearing and recomputing should produce a fresh cache object.
    MD._clear_cover_cache!(P)
    ccA = MD._get_cover_cache(P)
    MD._clear_cover_cache!(P)
    ccB = MD._get_cover_cache(P)
    @test ccA !== ccB

    cc1 = MD._get_cover_cache(P)
    cc2 = MD._get_cover_cache(P)
    @test cc1 === cc2

    @test cc1.Q === P
    @test length(cc1.pred_ptr) == P.n + 1
    @test length(cc1.succ_ptr) == P.n + 1

    # On a chain, each vertex i>1 has exactly one cover predecessor i-1.
    @test isempty(FF._preds(cc1, 1))
    for i in 2:P.n
        @test collect(FF._preds(cc1, i)) == [i - 1]
    end
end

@testset "Poset cache lifecycle" begin
    P = chain_poset(4)
    @test isdefined(P, :cache)
    @test P.cache.cover_edges === nothing
    @test P.cache.cover === nothing
    @test P.cache.upsets === nothing
    @test P.cache.downsets === nothing

    Ce = FF.cover_edges(P)
    @test P.cache.cover_edges === Ce

    cc = MD._get_cover_cache(P)
    @test P.cache.cover === cc
    @test P.cache.cover !== nothing

    u1 = FF.upset_indices(P, 1)
    d1 = FF.downset_indices(P, 1)
    @test P.cache.upsets !== nothing
    @test P.cache.downsets !== nothing
    @test collect(u1) == P.cache.upsets[1]
    @test collect(d1) == P.cache.downsets[1]

    MD._clear_cover_cache!(P)
    @test P.cache.cover_edges === nothing
    @test P.cache.cover === nothing
    @test P.cache.upsets === nothing
    @test P.cache.downsets === nothing

    cc2 = MD._get_cover_cache(P)
    @test cc2 !== cc
end


@testset "map_leq uses cached chosen-chain pointers" begin
    P = chain_poset(4)
    field = CM.QQField()
    K = CM.coeff_type(field)
    c(x) = CM.coerce(field, x)
    cc = MD._get_cover_cache(P)
    FF._clear_chain_parent_cache!(cc)
    n_before = sum(length.(cc.chain_parent))

    # Build a tiny 2-dimensional module on the chain 1<2<3<4.
    dims = [2, 2, 2, 2]
    edge = Dict{Tuple{Int,Int}, Matrix{K}}()
    for (u, v) in FF.cover_edges(P)
        edge[(u, v)] = CM.zeros(field, dims[v], dims[u])
    end
    edge[(1, 2)] = K[c(2) c(0); c(0) c(3)]
    edge[(2, 3)] = K[c(5) c(0); c(0) c(7)]
    edge[(3, 4)] = K[c(11) c(0); c(0) c(13)]
    M = MD.PModule{K}(P, dims, edge; field=field)
    for d in M.map_compose
        empty!(d)
    end
    m_before = sum(length.(M.map_compose))

    A14 = MD.map_leq(M, 1, 4; cache=cc)
    @test A14 == K[c(110) c(0); c(0) c(273)]

    # The chosen-chain cache should now contain at least the (1,4) entry.
    @test length(cc.chain_parent) >= 1
    n_after_first = sum(length.(cc.chain_parent))
    m_after_first = sum(length.(M.map_compose))
    @test m_after_first == m_before

    A14b = MD.map_leq(M, 1, 4; cache=cc)
    @test A14b == A14
    @test sum(length.(cc.chain_parent)) == n_after_first
    @test sum(length.(M.map_compose)) > m_after_first
end

@testset "map_leq long-chain cold path preserves multiplication order" begin
    P = chain_poset(5)
    field = CM.QQField()
    K = CM.coeff_type(field)
    c(x) = CM.coerce(field, x)
    cc = MD._get_cover_cache(P)
    FF._clear_chain_parent_cache!(cc)

    dims = fill(2, 5)
    edge = Dict{Tuple{Int,Int}, Matrix{K}}(
        (1, 2) => K[c(1) c(1); c(0) c(1)],
        (2, 3) => K[c(2) c(0); c(1) c(1)],
        (3, 4) => K[c(1) c(0); c(3) c(1)],
        (4, 5) => K[c(1) c(2); c(0) c(1)],
    )
    M = MD.PModule{K}(P, dims, edge; field=field)
    MD._clear_map_leq_memo!(M)

    A15 = MD.map_leq(M, 1, 5; cache=cc)
    @test A15 == K[c(16) c(18); c(7) c(8)]
    @test sum(length.(cc.chain_parent)) > 0
    @test sum(length.(M.map_compose)) == 0
    slot_hits = M.map_pred_slot_dense === nothing ? sum(length, M.map_pred_slot) :
        sum(d -> count(identity, d.seen), M.map_pred_slot_dense)
    @test slot_hits > 0

    A15b = MD.map_leq(M, 1, 5; cache=cc)
    @test A15b == A15
    @test sum(length.(M.map_compose)) > 0
end

@testset "map_leq tiny fast paths bypass compose memo" begin
    field = CM.QQField()
    K = CM.coeff_type(field)
    c(x) = CM.coerce(field, x)

    # 1x1 path: scalar composition path should not populate compose memo.
    P1 = chain_poset(4)
    dims1 = [1, 1, 1, 1]
    edge1 = Dict{Tuple{Int,Int}, Matrix{K}}(
        (1, 2) => reshape([c(2)], 1, 1),
        (2, 3) => reshape([c(3)], 1, 1),
        (3, 4) => reshape([c(5)], 1, 1),
    )
    M1 = MD.PModule{K}(P1, dims1, edge1; field=field)
    cc1 = MD._get_cover_cache(P1)
    FF._clear_chain_parent_cache!(cc1)
    for d in M1.map_compose
        empty!(d)
    end
    @test MD.map_leq(M1, 1, 4; cache=cc1) == reshape([c(30)], 1, 1)
    @test sum(length.(M1.map_compose)) == 0

    # Two-edge path with nontrivial dimensions should also skip compose memo.
    P2 = chain_poset(3)
    dims2 = [3, 2, 4]
    edge2 = Dict{Tuple{Int,Int}, Matrix{K}}(
        (1, 2) => K[
            c(1) c(2) c(0);
            c(0) c(3) c(4)
        ],
        (2, 3) => K[
            c(1) c(0);
            c(0) c(1);
            c(2) c(0);
            c(0) c(2)
        ],
    )
    M2 = MD.PModule{K}(P2, dims2, edge2; field=field)
    cc2 = MD._get_cover_cache(P2)
    FF._clear_chain_parent_cache!(cc2)
    for d in M2.map_compose
        empty!(d)
    end
    A13 = MD.map_leq(M2, 1, 3; cache=cc2)
    @test A13 == edge2[(2, 3)] * edge2[(1, 2)]
    @test sum(length.(M2.map_compose)) == 0
end


@testset "map_leq is path-independent on a functorial diamond" begin
    P = diamond_poset()
    field = CM.QQField()
    K = CM.coeff_type(field)
    c(x) = CM.coerce(field, x)
    cc = MD._get_cover_cache(P)
    FF._clear_chain_parent_cache!(cc)

    # A functorial module where the two length-2 paths 1->2->4 and 1->3->4 agree.
    dims = [1, 1, 1, 1]
    edge = Dict{Tuple{Int,Int}, Matrix{K}}()
    for (u, v) in FF.cover_edges(P)
        edge[(u, v)] = reshape([c(0)], 1, 1)
    end
    edge[(1, 2)] = reshape([c(2)], 1, 1)
    edge[(2, 4)] = reshape([c(5)], 1, 1)   # composite 10
    edge[(1, 3)] = reshape([c(1)], 1, 1)
    edge[(3, 4)] = reshape([c(10)], 1, 1)  # composite also 10
    M = MD.PModule{K}(P, dims, edge; field=field)
    for d in M.map_compose
        empty!(d)
    end

    A14 = MD.map_leq(M, 1, 4; cache=cc)
    @test A14 == reshape([c(10)], 1, 1)
end

@testset "map_leq compose memo is per-module" begin
    P = chain_poset(4)
    field = CM.QQField()
    K = CM.coeff_type(field)
    c(x) = CM.coerce(field, x)

    dims = [2, 2, 2, 2]
    edgeA = Dict{Tuple{Int,Int}, Matrix{K}}(
        (1, 2) => K[c(2) c(0); c(0) c(3)],
        (2, 3) => K[c(3) c(0); c(0) c(5)],
        (3, 4) => K[c(11) c(0); c(0) c(13)],
    )
    edgeB = Dict{Tuple{Int,Int}, Matrix{K}}(
        (1, 2) => K[c(5) c(0); c(0) c(7)],
        (2, 3) => K[c(7) c(0); c(0) c(11)],
        (3, 4) => K[c(13) c(0); c(0) c(17)],
    )
    M1 = MD.PModule{K}(P, dims, edgeA; field=field)
    M2 = MD.PModule{K}(P, dims, edgeB; field=field)

    cc = MD._get_cover_cache(P)
    FF._clear_chain_parent_cache!(cc)
    for d in M1.map_compose
        empty!(d)
    end
    for d in M2.map_compose
        empty!(d)
    end

    A1 = MD.map_leq(M1, 1, 4; cache=cc)
    A2 = MD.map_leq(M2, 1, 4; cache=cc)
    @test A1 == K[c(66) c(0); c(0) c(195)]
    @test A2 == K[c(455) c(0); c(0) c(1309)]
    @test A1 != A2
    @test sum(length.(M1.map_compose)) == 0
    @test sum(length.(M2.map_compose)) == 0
    @test MD.map_leq(M1, 1, 4; cache=cc) == A1
    @test MD.map_leq(M2, 1, 4; cache=cc) == A2
    @test sum(length.(M1.map_compose)) >= 1
    @test sum(length.(M2.map_compose)) >= 1
end

@testset "map_leq_many parity and preallocated output" begin
    P = chain_poset(4)
    field = CM.QQField()
    K = CM.coeff_type(field)
    c(x) = CM.coerce(field, x)

    dims = [1, 1, 1, 1]
    edge = Dict{Tuple{Int,Int}, Matrix{K}}(
        (1, 2) => reshape([c(2)], 1, 1),
        (2, 3) => reshape([c(3)], 1, 1),
        (3, 4) => reshape([c(5)], 1, 1),
    )
    M = MD.PModule{K}(P, dims, edge; field=field)
    cc = MD._get_cover_cache(P)
    FF._clear_chain_parent_cache!(cc)
    for d in M.map_compose
        empty!(d)
    end

    pairs = Tuple{Int,Int}[(1, 1), (1, 2), (1, 3), (1, 4), (2, 4), (4, 4)]
    batch = MD.map_leq_many(M, pairs; cache=cc)
    @test length(batch) == length(pairs)
    @test batch[1] == reshape([c(1)], 1, 1)
    @test batch[4] == reshape([c(30)], 1, 1)

    @inbounds for i in eachindex(pairs)
        u, v = pairs[i]
        @test batch[i] == MD.map_leq(M, u, v; cache=cc)
    end

    out = Vector{Matrix{K}}(undef, length(pairs))
    MD.map_leq_many!(out, M, pairs; cache=cc)
    @test out == batch
    @test sum(length.(M.map_compose)) == 0
end

@testset "map_leq_many cached plan reuse + signature invalidation" begin
    P = chain_poset(7)
    field = CM.F3()
    K = CM.coeff_type(field)
    c(x) = CM.coerce(field, x)

    dims = fill(2, 7)
    edge = Dict{Tuple{Int,Int}, Matrix{K}}()
    @inbounds for u in 1:6
        edge[(u, u + 1)] = K[c(1) c(2); c(0) c(1)]
    end
    M = MD.PModule{K}(P, dims, edge; field=field)
    cc = MD._get_cover_cache(P)
    old_min = MD.MAP_LEQ_MANY_PLAN_MIN_LEN[]
    try
        MD.MAP_LEQ_MANY_PLAN_MIN_LEN[] = 1

        pairs = Tuple{Int,Int}[(1, 1), (1, 3), (2, 5), (1, 7), (3, 7), (4, 7)]
        b1 = MD.map_leq_many(M, pairs; cache=cc)
        @test sum(length.(M.map_compose)) == 0
        b2 = MD.map_leq_many(M, pairs; cache=cc)
        @test b1 == b2
        @test any(!isempty(d) for d in M.map_many_plan)
        @test sum(length.(M.map_compose)) > 0

        # Mutate an interior pair in place (first/last unchanged); signature check must rebuild.
        pairs[3] = (2, 4)
        b3 = MD.map_leq_many(M, pairs; cache=cc)
        @test b3[3] == MD.map_leq(M, 2, 4; cache=cc)
        @test all(b3[i] == MD.map_leq(M, pairs[i][1], pairs[i][2]; cache=cc) for i in eachindex(pairs))
    finally
        MD.MAP_LEQ_MANY_PLAN_MIN_LEN[] = old_min
    end
end

@testset "map_leq_many one-off long batch avoids plan cache and preserves parity" begin
    P = FF.ProductOfChainsPoset((16, 16))
    field = CM.F3()
    K = CM.coeff_type(field)
    c(x) = CM.coerce(field, x)

    dims = fill(2, FF.nvertices(P))
    edge = Dict{Tuple{Int,Int}, Matrix{K}}()
    for (u, v) in FF.cover_edges(P)
        edge[(u, v)] = K[c(1) c(1); c(0) c(1)]
    end
    M = MD.PModule{K}(P, dims, edge; field=field)
    cc = MD._get_cover_cache(P)
    pairs = Tuple{Int,Int}[
        (1, 256), (2, 256), (17, 256), (18, 256),
        (1, 255), (2, 255), (16, 256), (1, 240),
    ]

    old_plan_min = MD.MAP_LEQ_MANY_PLAN_MIN_LEN[]
    old_oneoff_min = MD.MAP_LEQ_MANY_ONEOFF_LONG_MIN_LEN[]
    old_oneoff_long = MD.MAP_LEQ_MANY_ONEOFF_LONG_MIN_LONG[]
    try
        MD.MAP_LEQ_MANY_PLAN_MIN_LEN[] = 1
        MD.MAP_LEQ_MANY_ONEOFF_LONG_MIN_LEN[] = 1
        MD.MAP_LEQ_MANY_ONEOFF_LONG_MIN_LONG[] = 1

        MD._clear_map_leq_memo!(M)
        MD._clear_map_leq_many_plan_cache!(M)
        FF._clear_chain_parent_cache!(cc)

        @test MD._map_leq_many_raw_route_kind(M, pairs, cc) == :oneoff_long
        batch = MD.map_leq_many(M, pairs; cache=cc)
        @test all(isempty(d) for d in M.map_many_plan)
        @test sum(length.(M.map_compose)) == 0
        @test M.map_compose_dense === nothing || all(!any(m.seen) for m in M.map_compose_dense)
        @test all(batch[i] == MD.map_leq(M, pairs[i][1], pairs[i][2]; cache=cc) for i in eachindex(pairs))
    finally
        MD.MAP_LEQ_MANY_PLAN_MIN_LEN[] = old_plan_min
        MD.MAP_LEQ_MANY_ONEOFF_LONG_MIN_LEN[] = old_oneoff_min
        MD.MAP_LEQ_MANY_ONEOFF_LONG_MIN_LONG[] = old_oneoff_long
    end
end

@testset "map_leq_many raw route falls back on low-overlap long sweep" begin
    P = FF.ProductOfChainsPoset((12, 12))
    field = CM.QQField()
    K = CM.coeff_type(field)
    c(x) = CM.coerce(field, x)

    dims = fill(2, FF.nvertices(P))
    edge = Dict{Tuple{Int,Int}, Matrix{K}}()
    for (u, v) in FF.cover_edges(P)
        edge[(u, v)] = K[c(1) c(1); c(0) c(1)]
    end
    M = MD.PModule{K}(P, dims, edge; field=field)
    cc = MD._get_cover_cache(P)

    function long_hops(u, v)
        d = v
        hops = 0
        while d != u
            d = FF._chosen_predecessor(cc, u, d)
            hops += 1
        end
        return hops
    end

    pairs = Tuple{Int,Int}[]
    rng = MersenneTwister(0xB2B2)
    while length(pairs) < 64
        u = rand(rng, 1:FF.nvertices(P))
        v = rand(rng, 1:FF.nvertices(P))
        u == v && continue
        FF.leq(P, u, v) || continue
        long_hops(u, v) >= 3 || continue
        push!(pairs, (u, v))
    end

    old_plan_min = MD.MAP_LEQ_MANY_PLAN_MIN_LEN[]
    old_oneoff_min = MD.MAP_LEQ_MANY_ONEOFF_LONG_MIN_LEN[]
    old_oneoff_long = MD.MAP_LEQ_MANY_ONEOFF_LONG_MIN_LONG[]
    old_oneoff_overlap = MD.MAP_LEQ_MANY_ONEOFF_MIN_OVERLAP_QQ[]
    old_oneoff_target = MD.MAP_LEQ_MANY_ONEOFF_MIN_TARGET_REPEAT_QQ[]
    old_scalar_overlap = MD.MAP_LEQ_MANY_LONG_SCALAR_MAX_OVERLAP_QQ[]
    old_scalar_target = MD.MAP_LEQ_MANY_LONG_SCALAR_MAX_TARGET_REPEAT_QQ[]
    old_scalar_hops = MD.MAP_LEQ_MANY_LONG_MIN_AVG_HOPS_QQ[]
    old_nemo_hops = MD.MAP_LEQ_QQ_NEMO_LONG_MIN_HOPS[]
    old_nemo_work = MD.MAP_LEQ_QQ_NEMO_LONG_MIN_WORK[]
    try
        MD.MAP_LEQ_MANY_PLAN_MIN_LEN[] = typemax(Int)
        MD.MAP_LEQ_MANY_ONEOFF_LONG_MIN_LEN[] = 1
        MD.MAP_LEQ_MANY_ONEOFF_LONG_MIN_LONG[] = 1
        MD.MAP_LEQ_MANY_ONEOFF_MIN_OVERLAP_QQ[] = 1.0
        MD.MAP_LEQ_MANY_ONEOFF_MIN_TARGET_REPEAT_QQ[] = 1.0
        MD.MAP_LEQ_MANY_LONG_SCALAR_MAX_OVERLAP_QQ[] = 1.0
        MD.MAP_LEQ_MANY_LONG_SCALAR_MAX_TARGET_REPEAT_QQ[] = 1.0
        MD.MAP_LEQ_MANY_LONG_MIN_AVG_HOPS_QQ[] = 0.0
        FF._clear_chain_parent_cache!(cc)
        MD._clear_map_leq_memo!(M)
        MD._clear_map_leq_many_plan_cache!(M)
        @test MD._map_leq_many_raw_route_kind(M, pairs, cc) == :scalar_fallback
        batch = MD.map_leq_many(M, pairs; cache=cc)
        @test all(isempty(d) for d in M.map_many_plan)
        @test sum(length.(M.map_compose)) == 0
        @test all(batch[i] == MD.map_leq(M, pairs[i][1], pairs[i][2]; cache=cc) for i in eachindex(pairs))
    finally
        MD.MAP_LEQ_MANY_PLAN_MIN_LEN[] = old_plan_min
        MD.MAP_LEQ_MANY_ONEOFF_LONG_MIN_LEN[] = old_oneoff_min
        MD.MAP_LEQ_MANY_ONEOFF_LONG_MIN_LONG[] = old_oneoff_long
        MD.MAP_LEQ_MANY_ONEOFF_MIN_OVERLAP_QQ[] = old_oneoff_overlap
        MD.MAP_LEQ_MANY_ONEOFF_MIN_TARGET_REPEAT_QQ[] = old_oneoff_target
        MD.MAP_LEQ_MANY_LONG_SCALAR_MAX_OVERLAP_QQ[] = old_scalar_overlap
        MD.MAP_LEQ_MANY_LONG_SCALAR_MAX_TARGET_REPEAT_QQ[] = old_scalar_target
        MD.MAP_LEQ_MANY_LONG_MIN_AVG_HOPS_QQ[] = old_scalar_hops
        MD.MAP_LEQ_QQ_NEMO_LONG_MIN_HOPS[] = old_nemo_hops
        MD.MAP_LEQ_QQ_NEMO_LONG_MIN_WORK[] = old_nemo_work
    end
end

@testset "map_leq_many raw route uses plan build on overlap-heavy long sweep" begin
    P = FF.ProductOfChainsPoset((12, 12))
    field = CM.QQField()
    K = CM.coeff_type(field)
    c(x) = CM.coerce(field, x)

    dims = fill(2, FF.nvertices(P))
    edge = Dict{Tuple{Int,Int}, Matrix{K}}()
    for (u, v) in FF.cover_edges(P)
        edge[(u, v)] = K[c(1) c(1); c(0) c(1)]
    end
    M = MD.PModule{K}(P, dims, edge; field=field)
    cc = MD._get_cover_cache(P)

    pairs = Tuple{Int,Int}[]
    rng = MersenneTwister(0xB3B3)
    target = FF.nvertices(P)
    while length(pairs) < 128
        u = rand(rng, 1:(target - 1))
        FF.leq(P, u, target) || continue
        push!(pairs, (u, target))
    end

    old_plan_min = MD.MAP_LEQ_MANY_PLAN_MIN_LEN[]
    old_oneoff_min = MD.MAP_LEQ_MANY_ONEOFF_LONG_MIN_LEN[]
    old_oneoff_long = MD.MAP_LEQ_MANY_ONEOFF_LONG_MIN_LONG[]
    old_oneoff_overlap = MD.MAP_LEQ_MANY_ONEOFF_MIN_OVERLAP_QQ[]
    old_oneoff_target = MD.MAP_LEQ_MANY_ONEOFF_MIN_TARGET_REPEAT_QQ[]
    old_nemo_hops = MD.MAP_LEQ_QQ_NEMO_LONG_MIN_HOPS[]
    old_nemo_work = MD.MAP_LEQ_QQ_NEMO_LONG_MIN_WORK[]
    try
        MD.MAP_LEQ_MANY_PLAN_MIN_LEN[] = 1
        MD.MAP_LEQ_MANY_ONEOFF_LONG_MIN_LEN[] = 1
        MD.MAP_LEQ_MANY_ONEOFF_LONG_MIN_LONG[] = 1
        MD.MAP_LEQ_MANY_ONEOFF_MIN_OVERLAP_QQ[] = 1.0
        MD.MAP_LEQ_MANY_ONEOFF_MIN_TARGET_REPEAT_QQ[] = 1.0
        FF._clear_chain_parent_cache!(cc)
        MD._clear_map_leq_memo!(M)
        MD._clear_map_leq_many_plan_cache!(M)
        @test MD._map_leq_many_raw_route_kind(M, pairs, cc) == :plan_build
        batch = MD.map_leq_many(M, pairs; cache=cc)
        @test any(!isempty(d) for d in M.map_many_plan)
        @test all(batch[i] == MD.map_leq(M, pairs[i][1], pairs[i][2]; cache=cc) for i in eachindex(pairs))

        if MD.FieldLinAlg._have_nemo()
            MD.MAP_LEQ_QQ_NEMO_LONG_MIN_HOPS[] = 1
            MD.MAP_LEQ_QQ_NEMO_LONG_MIN_WORK[] = 0
            MD.FieldLinAlg._reset_conversion_counters!()
            FF._clear_chain_parent_cache!(cc)
            MD._clear_map_leq_memo!(M)
            MD._clear_map_leq_many_plan_cache!(M)
            batch = MD.map_leq_many(M, pairs; cache=cc)
            counters = MD.FieldLinAlg._conversion_counters()
            @test counters.qq_to_nemo > 0
            @test all(batch[i] == MD.map_leq(M, pairs[i][1], pairs[i][2]; cache=cc) for i in eachindex(pairs))
        end
    finally
        MD.MAP_LEQ_MANY_PLAN_MIN_LEN[] = old_plan_min
        MD.MAP_LEQ_MANY_ONEOFF_LONG_MIN_LEN[] = old_oneoff_min
        MD.MAP_LEQ_MANY_ONEOFF_LONG_MIN_LONG[] = old_oneoff_long
        MD.MAP_LEQ_MANY_ONEOFF_MIN_OVERLAP_QQ[] = old_oneoff_overlap
        MD.MAP_LEQ_MANY_ONEOFF_MIN_TARGET_REPEAT_QQ[] = old_oneoff_target
        MD.MAP_LEQ_QQ_NEMO_LONG_MIN_HOPS[] = old_nemo_hops
        MD.MAP_LEQ_QQ_NEMO_LONG_MIN_WORK[] = old_nemo_work
    end
end

@testset "map_leq_many scalar long batch reuses preallocated outputs" begin
    P = FF.ProductOfChainsPoset((12, 12))
    field = CM.QQField()
    K = CM.coeff_type(field)
    c(x) = CM.coerce(field, x)

    dims = fill(2, FF.nvertices(P))
    edge = Dict{Tuple{Int,Int}, Matrix{K}}()
    for (u, v) in FF.cover_edges(P)
        edge[(u, v)] = K[c(1) c(1); c(0) c(1)]
    end
    M = MD.PModule{K}(P, dims, edge; field=field)
    cc = MD._get_cover_cache(P)

    function long_hops(u, v)
        d = v
        hops = 0
        while d != u
            d = FF._chosen_predecessor(cc, u, d)
            hops += 1
        end
        return hops
    end

    pairs = Tuple{Int,Int}[]
    seen = Set{Tuple{Int,Int}}()
    rng = MersenneTwister(0xB4B4)
    while length(pairs) < 64
        u = rand(rng, 1:FF.nvertices(P))
        v = rand(rng, 1:FF.nvertices(P))
        u == v && continue
        FF.leq(P, u, v) || continue
        long_hops(u, v) >= 3 || continue
        pair = (u, v)
        pair in seen && continue
        push!(seen, pair)
        push!(pairs, pair)
    end

    old_plan_min = MD.MAP_LEQ_MANY_PLAN_MIN_LEN[]
    old_oneoff_min = MD.MAP_LEQ_MANY_ONEOFF_LONG_MIN_LEN[]
    old_oneoff_long = MD.MAP_LEQ_MANY_ONEOFF_LONG_MIN_LONG[]
    old_oneoff_overlap = MD.MAP_LEQ_MANY_ONEOFF_MIN_OVERLAP_QQ[]
    old_oneoff_target = MD.MAP_LEQ_MANY_ONEOFF_MIN_TARGET_REPEAT_QQ[]
    old_scalar_overlap = MD.MAP_LEQ_MANY_LONG_SCALAR_MAX_OVERLAP_QQ[]
    old_scalar_target = MD.MAP_LEQ_MANY_LONG_SCALAR_MAX_TARGET_REPEAT_QQ[]
    old_scalar_hops = MD.MAP_LEQ_MANY_LONG_MIN_AVG_HOPS_QQ[]
    old_nemo_hops = MD.MAP_LEQ_QQ_NEMO_LONG_MIN_HOPS[]
    old_nemo_work = MD.MAP_LEQ_QQ_NEMO_LONG_MIN_WORK[]
    try
        MD.MAP_LEQ_MANY_PLAN_MIN_LEN[] = typemax(Int)
        MD.MAP_LEQ_MANY_ONEOFF_LONG_MIN_LEN[] = 1
        MD.MAP_LEQ_MANY_ONEOFF_LONG_MIN_LONG[] = 1
        MD.MAP_LEQ_MANY_ONEOFF_MIN_OVERLAP_QQ[] = 1.0
        MD.MAP_LEQ_MANY_ONEOFF_MIN_TARGET_REPEAT_QQ[] = 1.0
        MD.MAP_LEQ_MANY_LONG_SCALAR_MAX_OVERLAP_QQ[] = 1.0
        MD.MAP_LEQ_MANY_LONG_SCALAR_MAX_TARGET_REPEAT_QQ[] = 1.0
        MD.MAP_LEQ_MANY_LONG_MIN_AVG_HOPS_QQ[] = 0.0
        FF._clear_chain_parent_cache!(cc)
        MD._clear_map_leq_memo!(M)
        MD._clear_map_leq_many_plan_cache!(M)
        arena = MD._map_leq_many_plan_arena(M)
        stats = MD._fill_map_leq_many_plan_arena!(arena, M, pairs, P, cc)
        @test MD._prefer_map_leq_many_scalar_long(stats, M.field)

        out = Vector{Matrix{K}}(undef, length(pairs))
        @test MD._map_leq_many_scalar_long_batch!(out, M, arena, stats)
        ids = map(objectid, out)
        @test sum(length.(M.map_compose)) == 0

        FF._clear_chain_parent_cache!(cc)
        MD._clear_map_leq_memo!(M)
        @test MD._map_leq_many_scalar_long_batch!(out, M, arena, stats)
        @test map(objectid, out) == ids
        @test sum(length.(M.map_compose)) == 0
        @test all(out[i] == MD.map_leq(M, pairs[i][1], pairs[i][2]; cache=cc) for i in eachindex(pairs))

        if MD.FieldLinAlg._have_nemo()
            MD.MAP_LEQ_QQ_NEMO_LONG_MIN_HOPS[] = 1
            MD.MAP_LEQ_QQ_NEMO_LONG_MIN_WORK[] = 0
            MD.FieldLinAlg._reset_conversion_counters!()
            FF._clear_chain_parent_cache!(cc)
            MD._clear_map_leq_memo!(M)
            @test MD._map_leq_many_scalar_long_batch!(out, M, arena, stats)
            counters = MD.FieldLinAlg._conversion_counters()
            @test counters.qq_to_nemo > 0
            @test all(out[i] == MD.map_leq(M, pairs[i][1], pairs[i][2]; cache=cc) for i in eachindex(pairs))
        end
    finally
        MD.MAP_LEQ_MANY_PLAN_MIN_LEN[] = old_plan_min
        MD.MAP_LEQ_MANY_ONEOFF_LONG_MIN_LEN[] = old_oneoff_min
        MD.MAP_LEQ_MANY_ONEOFF_LONG_MIN_LONG[] = old_oneoff_long
        MD.MAP_LEQ_MANY_ONEOFF_MIN_OVERLAP_QQ[] = old_oneoff_overlap
        MD.MAP_LEQ_MANY_ONEOFF_MIN_TARGET_REPEAT_QQ[] = old_oneoff_target
        MD.MAP_LEQ_MANY_LONG_SCALAR_MAX_OVERLAP_QQ[] = old_scalar_overlap
        MD.MAP_LEQ_MANY_LONG_SCALAR_MAX_TARGET_REPEAT_QQ[] = old_scalar_target
        MD.MAP_LEQ_MANY_LONG_MIN_AVG_HOPS_QQ[] = old_scalar_hops
        MD.MAP_LEQ_QQ_NEMO_LONG_MIN_HOPS[] = old_nemo_hops
        MD.MAP_LEQ_QQ_NEMO_LONG_MIN_WORK[] = old_nemo_work
    end
end

@testset "map_leq_many prepared batch parity + source-mutation isolation" begin
    P = chain_poset(7)
    field = CM.F3()
    K = CM.coeff_type(field)
    c(x) = CM.coerce(field, x)

    dims = fill(2, 7)
    edge = Dict{Tuple{Int,Int}, Matrix{K}}()
    @inbounds for u in 1:6
        edge[(u, u + 1)] = K[c(1) c(2); c(0) c(1)]
    end
    M = MD.PModule{K}(P, dims, edge; field=field)
    cc = MD._get_cover_cache(P)
    old_min = MD.MAP_LEQ_MANY_PLAN_MIN_LEN[]
    try
        MD.MAP_LEQ_MANY_PLAN_MIN_LEN[] = 1
        pairs = Tuple{Int,Int}[(1, 1), (1, 3), (2, 5), (1, 7), (3, 7), (4, 7)]
        batch = MD.prepare_map_leq_batch(pairs)

        b_vec = MD.map_leq_many(M, pairs; cache=cc)
        b_batch = MD.map_leq_many(M, batch; cache=cc)
        @test b_batch == b_vec

        # Source vector can mutate; prepared batch stays stable because it owns a copy.
        pairs[2] = (1, 4)
        b_batch_after = MD.map_leq_many(M, batch; cache=cc)
        b_vec_after = MD.map_leq_many(M, pairs; cache=cc)
        @test b_batch_after == b_batch
        @test b_vec_after[2] == MD.map_leq(M, 1, 4; cache=cc)
        @test b_batch_after[2] == MD.map_leq(M, 1, 3; cache=cc)
        @test any(!isempty(d) for d in M.map_many_batch_plan)
        @test any(x -> x !== nothing, M.map_many_batch_last)

        entry = only(filter(x -> x !== nothing, M.map_many_batch_last))
        plan = (entry::MD._MapLeqManyBatchPlanEntry).plan
        @test !isempty(plan.suffix_u)
        @test length(plan.query_suffix) == length(batch.pairs)
        dense_before = cc.chain_parent_dense === nothing ? 0 :
            sum(d -> count(identity, d.seen), cc.chain_parent_dense)
        dict_before = sum(length, cc.chain_parent)
        FF._clear_chain_parent_cache!(cc)
        dense_cleared = cc.chain_parent_dense === nothing ? 0 :
            sum(d -> count(identity, d.seen), cc.chain_parent_dense)
        @test dense_cleared == 0
        @test sum(length, cc.chain_parent) == 0
        b_batch_again = MD.map_leq_many(M, batch; cache=cc)
        dense_after = cc.chain_parent_dense === nothing ? 0 :
            sum(d -> count(identity, d.seen), cc.chain_parent_dense)
        dict_after = sum(length, cc.chain_parent)
        @test b_batch_again == b_batch
        @test dense_before >= 0
        @test dict_before >= 0
        @test dense_after == 0
        @test dict_after == 0

        MD._clear_map_leq_many_plan_cache!(M)
        @test all(x -> x === nothing, M.map_many_batch_last)

        out = Vector{Matrix{K}}(undef, length(b_batch))
        MD.map_leq_many!(out, M, batch; cache=cc)
        @test out == b_batch
    finally
        MD.MAP_LEQ_MANY_PLAN_MIN_LEN[] = old_min
    end
end

@testset "Modules negative API contracts (map_leq / map_leq_many!)" begin
    P = diamond_poset()  # 2 and 3 are incomparable
    field = CM.QQField()
    K = CM.coeff_type(field)
    c(x) = CM.coerce(field, x)
    cc = MD._get_cover_cache(P)

    dims = [1, 1, 1, 1]
    edge = Dict{Tuple{Int,Int}, Matrix{K}}(
        (1, 2) => reshape([c(2)], 1, 1),
        (1, 3) => reshape([c(3)], 1, 1),
        (2, 4) => reshape([c(5)], 1, 1),
        (3, 4) => reshape([c(10)], 1, 1),
    )
    M = MD.PModule{K}(P, dims, edge; field=field)

    # map_leq contracts
    @test_throws ErrorException MD.map_leq(M, 0, 1; cache=cc)
    @test_throws ErrorException MD.map_leq(M, 1, 5; cache=cc)
    @test_throws ErrorException MD.map_leq(M, 2, 3; cache=cc)  # incomparable
    @test_throws ErrorException MD.map_leq(
        M, 1, 4;
        cache=cc,
        opts=OPT.ModuleOptions(cache=cc),
    )

    # map_leq_many! contracts
    pairs = Tuple{Int,Int}[(1, 1), (1, 2), (1, 4)]
    @test_throws ErrorException MD.map_leq_many!(Vector{Matrix{K}}(undef, 2), M, pairs; cache=cc)
    @test_throws ErrorException MD.map_leq_many!(Vector{Matrix{K}}(undef, 1), M, Tuple{Int,Int}[(0, 1)]; cache=cc)
    @test_throws ErrorException MD.map_leq_many!(Vector{Matrix{K}}(undef, 1), M, Tuple{Int,Int}[(2, 3)]; cache=cc)
    @test_throws ErrorException MD.map_leq_many!(
        Vector{Matrix{K}}(undef, 1),
        M,
        Tuple{Int,Int}[(1, 4)];
        cache=cc,
        opts=OPT.ModuleOptions(cache=cc),
    )

    batch_ok = MD.prepare_map_leq_batch(Tuple{Int,Int}[(1, 1), (1, 2)])
    @test_throws ErrorException MD.map_leq_many!(Vector{Matrix{K}}(undef, 1), M, batch_ok; cache=cc)

    batch_bad = MD.prepare_map_leq_batch(Tuple{Int,Int}[(2, 3)])
    @test_throws ErrorException MD.map_leq_many(M, batch_bad; cache=cc)
end

@testset "ModuleOptions contracts (cache + check_sizes)" begin
    P = chain_poset(3)
    field = CM.QQField()
    K = CM.coeff_type(field)
    c(x) = CM.coerce(field, x)
    cc = MD._get_cover_cache(P)

    dims = [1, 1, 1]
    edge = Dict{Tuple{Int,Int}, Matrix{K}}(
        (1, 2) => reshape([c(2)], 1, 1),
        (2, 3) => reshape([c(3)], 1, 1),
    )
    M = MD.PModule{K}(P, dims, edge; field=field)

    # opts.cache should be equivalent to cache keyword.
    A_kw = MD.map_leq(M, 1, 3; cache=cc)
    A_opt = MD.map_leq(M, 1, 3; opts=OPT.ModuleOptions(cache=cc))
    @test A_kw == A_opt

    pairs = Tuple{Int,Int}[(1, 2), (1, 3), (2, 3)]
    B_kw = MD.map_leq_many(M, pairs; cache=cc)
    B_opt = MD.map_leq_many(M, pairs; opts=OPT.ModuleOptions(cache=cc))
    @test B_kw == B_opt

    # check_sizes contract in constructor.
    bad_edge = Dict{Tuple{Int,Int}, Matrix{K}}(
        (1, 2) => reshape([c(1), c(2)], 2, 1),  # wrong row count for dims[2] == 1
        (2, 3) => reshape([c(1)], 1, 1),
    )
    @test_throws ErrorException MD.PModule{K}(P, dims, bad_edge; field=field)
    M_bad = MD.PModule{K}(P, dims, bad_edge; field=field, opts=OPT.ModuleOptions(check_sizes=false))
    @test M_bad.dims == dims

    # Passing both explicit keyword and non-default opts is rejected.
    @test_throws ErrorException MD.PModule{K}(
        P, dims, edge;
        field=field,
        check_sizes=false,
        opts=OPT.ModuleOptions(check_sizes=false),
    )
end

@testset "map_leq uses dense chain-parent cache on large finite posets" begin
    old_min = FF.CHAIN_PARENT_DENSE_MIN_ENTRIES[]
    old_per = FF.CHAIN_PARENT_DENSE_MAX_ENTRIES_PER_THREAD[]
    old_total = FF.CHAIN_PARENT_DENSE_MAX_TOTAL_ENTRIES[]
    try
        FF.CHAIN_PARENT_DENSE_MIN_ENTRIES[] = 1
        FF.CHAIN_PARENT_DENSE_MAX_ENTRIES_PER_THREAD[] = 1_000_000
        FF.CHAIN_PARENT_DENSE_MAX_TOTAL_ENTRIES[] = max(1, Threads.maxthreadid()) * 1_000_000

        P = chain_poset(40)
        cc = MD._get_cover_cache(P)
        @test cc.chain_parent_dense !== nothing
        FF._clear_chain_parent_cache!(cc)

        field = CM.QQField()
        K = CM.coeff_type(field)
        oneK = CM.coerce(field, 1)
        dims = ones(Int, 40)
        edge = Dict{Tuple{Int,Int}, Matrix{K}}()
        for (u, v) in FF.cover_edges(P)
            edge[(u, v)] = reshape([oneK], 1, 1)
        end
        M = MD.PModule{K}(P, dims, edge; field=field)
        for d in M.map_compose
            empty!(d)
        end

        pairs = Tuple{Int,Int}[(1, 40), (5, 40), (10, 35), (20, 39), (1, 30)]
        mats = MD.map_leq_many(M, pairs; cache=cc)
        @test all(A -> A == reshape([oneK], 1, 1), mats)
        @test sum(length.(cc.chain_parent)) == 0
        @test any(m -> any(m.seen), cc.chain_parent_dense)
    finally
        FF.CHAIN_PARENT_DENSE_MIN_ENTRIES[] = old_min
        FF.CHAIN_PARENT_DENSE_MAX_ENTRIES_PER_THREAD[] = old_per
        FF.CHAIN_PARENT_DENSE_MAX_TOTAL_ENTRIES[] = old_total
    end
end

@testset "direct_sum preserves sparse edge-map storage" begin
    P = chain_poset(2)
    field = CM.QQField()
    K = CM.coeff_type(field)
    c(x) = CM.coerce(field, x)

    dimsA = [70, 70]
    dimsB = [60, 60]
    edgeA = Dict{Tuple{Int,Int}, SparseMatrixCSC{K,Int}}(
        (1, 2) => sparse([1, 2], [3, 15], K[c(2), c(5)], dimsA[2], dimsA[1]),
    )
    edgeB = Dict{Tuple{Int,Int}, SparseMatrixCSC{K,Int}}(
        (1, 2) => sparse([4, 9], [5, 11], K[c(7), c(11)], dimsB[2], dimsB[1]),
    )

    A = MD.PModule{K}(P, dimsA, edgeA; field=field)
    B = MD.PModule{K}(P, dimsB, edgeB; field=field)
    S = MD.direct_sum(A, B)
    @test S.edge_maps.maps_from_pred[2][1] isa SparseMatrixCSC{K,Int}
    M12 = S.edge_maps[1, 2]
    @test nnz(M12) == 4
    @test M12[1, 3] == c(2)
    @test M12[2, 15] == c(5)
    @test M12[dimsA[2] + 4, dimsA[1] + 5] == c(7)
    @test M12[dimsA[2] + 9, dimsA[1] + 11] == c(11)
end

@testset "PModule stores concrete poset type + map_leq identity cache reuse" begin
    P = chain_poset(3)
    field = CM.QQField()
    K = CM.coeff_type(field)
    oneK = CM.coerce(field, 1)
    dims = [2, 2, 2]
    edge = Dict{Tuple{Int,Int}, Matrix{K}}(
        (1, 2) => reshape(K[oneK, oneK, oneK, oneK], 2, 2),
        (2, 3) => reshape(K[oneK, oneK, oneK, oneK], 2, 2),
    )
    M = MD.PModule{K}(P, dims, edge; field=field)
    cc = MD._get_cover_cache(P)

    @test M isa MD.PModule{K,typeof(field),Matrix{K},typeof(P)}

    A22 = MD.map_leq(M, 2, 2; cache=cc)
    B22 = MD.map_leq(M, 2, 2; cache=cc)
    @test A22 === B22

    batch = MD.map_leq_many(M, Tuple{Int,Int}[(2, 2), (2, 2), (1, 1)]; cache=cc)
    @test batch[1] === batch[2]
    @test batch[1] === A22
end

@testset "map_leq dense compose memo table (finite-poset path)" begin
    P = chain_poset(6)
    field = CM.QQField()
    K = CM.coeff_type(field)
    c(x) = CM.coerce(field, x)
    dims = fill(2, 6)
    edge = Dict{Tuple{Int,Int}, Matrix{K}}()
    @inbounds for u in 1:5
        edge[(u, u + 1)] = K[c(u + 1) c(0); c(0) c(u + 1)]
    end

    old_min = MD.MAP_LEQ_DENSE_MEMO_MIN_ENTRIES[]
    old_max = MD.MAP_LEQ_DENSE_MEMO_MAX_ENTRIES_PER_THREAD[]
    try
        MD.MAP_LEQ_DENSE_MEMO_MIN_ENTRIES[] = 1
        MD.MAP_LEQ_DENSE_MEMO_MAX_ENTRIES_PER_THREAD[] = 10_000
        M = MD.PModule{K}(P, dims, edge; field=field)
        cc = MD._get_cover_cache(P)

        @test M.map_compose_dense !== nothing
        A = MD.map_leq(M, 1, 6; cache=cc)
        @test A == K[c(720) c(0); c(0) c(720)]
        @test sum(length.(M.map_compose)) == 0
        if M.map_pred_slot_dense === nothing
            @test any(!isempty(d) for d in M.map_pred_slot)
        else
            @test any(any(d.seen) for d in M.map_pred_slot_dense)
        end
        @test all(!any(d.seen) for d in M.map_compose_dense)

        MD._clear_map_leq_memo!(M)
        @test all(!any(d.seen) for d in M.map_compose_dense)
    finally
        MD.MAP_LEQ_DENSE_MEMO_MIN_ENTRIES[] = old_min
        MD.MAP_LEQ_DENSE_MEMO_MAX_ENTRIES_PER_THREAD[] = old_max
    end
end

@testset "direct_sum route policy picks dense for high-density sparse inputs" begin
    P = chain_poset(2)
    field = CM.RealField(Float64; rtol=1e-10, atol=1e-12)
    K = CM.coeff_type(field)
    dims = [64, 64]

    low_vals = fill(one(K), 8)
    low_edgeA = Dict{Tuple{Int,Int}, SparseMatrixCSC{K,Int}}(
        (1, 2) => sparse(collect(1:8), collect(1:8), low_vals, dims[2], dims[1]),
    )
    low_edgeB = Dict{Tuple{Int,Int}, SparseMatrixCSC{K,Int}}(
        (1, 2) => sparse(collect(9:16), collect(9:16), low_vals, dims[2], dims[1]),
    )

    # Build a fully-populated sparse matrix (density ~= 1) to force dense output.
    Ii = Int[]
    Jj = Int[]
    Vv = K[]
    sizehint!(Ii, dims[1] * dims[2])
    sizehint!(Jj, dims[1] * dims[2])
    sizehint!(Vv, dims[1] * dims[2])
    @inbounds for j in 1:dims[1], i in 1:dims[2]
        push!(Ii, i)
        push!(Jj, j)
        push!(Vv, one(K))
    end
    high_dense_sparse = sparse(Ii, Jj, Vv, dims[2], dims[1])
    high_edgeA = Dict{Tuple{Int,Int}, SparseMatrixCSC{K,Int}}((1, 2) => high_dense_sparse)
    high_edgeB = Dict{Tuple{Int,Int}, SparseMatrixCSC{K,Int}}((1, 2) => high_dense_sparse)

    old_min = MD.DIRECT_SUM_SPARSE_MIN_TOTAL_ENTRIES[]
    try
        MD.DIRECT_SUM_SPARSE_MIN_TOTAL_ENTRIES[] = 1

        Slow = MD.direct_sum(
            MD.PModule{K}(P, dims, low_edgeA; field=field),
            MD.PModule{K}(P, dims, low_edgeB; field=field),
        )
        @test Slow.edge_maps[1, 2] isa SparseMatrixCSC{K,Int}

        Shigh = MD.direct_sum(
            MD.PModule{K}(P, dims, high_edgeA; field=field),
            MD.PModule{K}(P, dims, high_edgeB; field=field),
        )
        @test Shigh.edge_maps[1, 2] isa Matrix{K}
    finally
        MD.DIRECT_SUM_SPARSE_MIN_TOTAL_ENTRIES[] = old_min
    end
end

@testset "direct_sum field-aware route + cached edge stats" begin
    P = chain_poset(2)

    function denseish_sparse(field::CM.AbstractCoeffField, shift::Int)
        K = CM.coeff_type(field)
        ii = Int[]
        jj = Int[]
        vv = K[]
        @inbounds for j in 1:40, i in 1:40
            ((i + 2j + shift) % 5 == 0) && continue
            push!(ii, i)
            push!(jj, j)
            push!(vv, CM.coerce(field, ((i + j + shift) % 7) + 1))
        end
        return sparse(ii, jj, vv, 40, 40)
    end

    cases = (
        (CM.QQField(), true),
        (CM.F3(), false),
        (CM.RealField(Float64; rtol=1e-10, atol=1e-12), false),
    )

    old_min = MD.DIRECT_SUM_SPARSE_MIN_TOTAL_ENTRIES[]
    try
        MD.DIRECT_SUM_SPARSE_MIN_TOTAL_ENTRIES[] = 1
        for (field, expect_sparse) in cases
            K = CM.coeff_type(field)
            A = MD.PModule{K}(P, [40, 40], Dict((1, 2) => denseish_sparse(field, 0)); field=field)
            B = MD.PModule{K}(P, [40, 40], Dict((1, 2) => denseish_sparse(field, 2)); field=field)
            S = MD.direct_sum(A, B)

            @test A.direct_sum_stats.n_edges == 1
            @test A.direct_sum_stats.total_entries == 1600
            @test A.direct_sum_stats.total_nnz == nnz(A.edge_maps[1, 2])

            if expect_sparse
                @test S.edge_maps[1, 2] isa SparseMatrixCSC{K,Int}
            else
                @test S.edge_maps[1, 2] isa Matrix{K}
            end
        end
    finally
        MD.DIRECT_SUM_SPARSE_MIN_TOTAL_ENTRIES[] = old_min
    end
end

@testset "direct_sum route uses small-edge field gate" begin
    P = chain_poset(2)

    function low_sparse_edge(field::CM.AbstractCoeffField, shift::Int)
        K = CM.coeff_type(field)
        vals = K[CM.coerce(field, 1), CM.coerce(field, 2), CM.coerce(field, 3)]
        rows = Int[1, 3, 6]
        cols = Int[1 + (shift % 2), 4, 6]
        return sparse(rows, cols, vals, 6, 6)
    end

    cases = (
        (CM.QQField(), true),
        (CM.F3(), false),
        (CM.RealField(Float64; rtol=1e-10, atol=1e-12), false),
    )

    old_min = MD.DIRECT_SUM_SPARSE_MIN_TOTAL_ENTRIES[]
    try
        MD.DIRECT_SUM_SPARSE_MIN_TOTAL_ENTRIES[] = 1
        for (field, expect_sparse) in cases
            K = CM.coeff_type(field)
            A = MD.PModule{K}(P, [6, 6], Dict((1, 2) => low_sparse_edge(field, 0)); field=field)
            B = MD.PModule{K}(P, [6, 6], Dict((1, 2) => low_sparse_edge(field, 1)); field=field)

            @test MD._direct_sum_sparse_preferred(A, B) == expect_sparse

            S = MD.direct_sum(A, B)
            if expect_sparse
                @test S.edge_maps[1, 2] isa SparseMatrixCSC{K,Int}
            else
                @test S.edge_maps[1, 2] isa Matrix{K}
            end
        end
    finally
        MD.DIRECT_SUM_SPARSE_MIN_TOTAL_ENTRIES[] = old_min
    end
end

@testset "map_leq short-chain last-pair micro-cache promotes repeats" begin
    P = chain_poset(3)
    field = CM.QQField()
    K = CM.coeff_type(field)
    c(x) = CM.coerce(field, x)

    dims = [2, 2, 2]
    edge = Dict{Tuple{Int,Int}, Matrix{K}}(
        (1, 2) => K[c(2) c(1); c(0) c(3)],
        (2, 3) => K[c(5) c(0); c(1) c(4)],
    )
    M = MD.PModule{K}(P, dims, edge; field=field)
    cc = MD._get_cover_cache(P)

    MD._clear_map_leq_memo!(M)
    A13_1 = MD.map_leq(M, 1, 3; cache=cc)
    @test sum(length.(M.map_compose)) == 0

    A13_2 = MD.map_leq(M, 1, 3; cache=cc)
    @test A13_2 == A13_1
    @test sum(length.(M.map_compose)) >= 1

    slot = min(length(M.map_last_pair.us), max(1, Base.Threads.threadid()))
    @test M.map_last_pair.seen[slot]
    @test M.map_last_pair.us[slot] == 1
    @test M.map_last_pair.vs[slot] == 3
    @test M.map_last_pair.promoted[slot]
end

@testset "map_leq cold short-chain fast path avoids chain-parent cache churn" begin
    P = chain_poset(3)
    field = CM.F3()
    K = CM.coeff_type(field)
    c(x) = CM.coerce(field, x)

    dims = [2, 2, 2]
    edge = Dict{Tuple{Int,Int}, Matrix{K}}(
        (1, 2) => K[c(1) c(2); c(0) c(1)],
        (2, 3) => K[c(2) c(1); c(1) c(0)],
    )
    M = MD.PModule{K}(P, dims, edge; field=field)
    cc = MD._get_cover_cache(P)

    FF._clear_chain_parent_cache!(cc)
    MD._clear_map_leq_memo!(M)

    dense_before = cc.chain_parent_dense === nothing ? 0 :
        sum(d -> count(identity, d.seen), cc.chain_parent_dense)
    dict_before = sum(length, cc.chain_parent)
    @test dense_before == 0
    @test dict_before == 0

    A13 = MD.map_leq(M, 1, 3; cache=cc)
    @test A13 == edge[(2, 3)] * edge[(1, 2)]

    dense_after = cc.chain_parent_dense === nothing ? 0 :
        sum(d -> count(identity, d.seen), cc.chain_parent_dense)
    dict_after = sum(length, cc.chain_parent)
    @test dense_after == 0
    @test dict_after == 0
end

@testset "map_leq_many plan arena does not alias cached plan payload" begin
    P = chain_poset(6)
    field = CM.QQField()
    K = CM.coeff_type(field)
    oneK = CM.coerce(field, 1)

    dims = ones(Int, 6)
    edge = Dict{Tuple{Int,Int}, Matrix{K}}()
    @inbounds for u in 1:5
        edge[(u, u + 1)] = reshape(K[oneK], 1, 1)
    end
    M = MD.PModule{K}(P, dims, edge; field=field)
    cc = MD._get_cover_cache(P)

    pairs1 = Tuple{Int,Int}[(1, 6), (1, 5), (2, 6), (3, 6)]
    pairs2 = Tuple{Int,Int}[(1, 4), (2, 5), (3, 5), (4, 6)]

    old_min = MD.MAP_LEQ_MANY_PLAN_MIN_LEN[]
    old_max = MD.MAP_LEQ_MANY_PLAN_MAX_PER_THREAD[]
    try
        MD.MAP_LEQ_MANY_PLAN_MIN_LEN[] = 1
        MD.MAP_LEQ_MANY_PLAN_MAX_PER_THREAD[] = 1024
        MD._clear_map_leq_many_plan_cache!(M)

        MD.map_leq_many(M, pairs1; cache=cc)
        slot = min(length(M.map_many_plan), max(1, Base.Threads.threadid()))
        dict = M.map_many_plan[slot]
        key1 = MD._map_leq_many_plan_key(pairs1)
        @test haskey(dict, key1)
        plan1 = dict[key1]
        us_copy = copy(plan1.us)
        chain_copy = copy(plan1.chain_data)

        MD.map_leq_many(M, pairs2; cache=cc)
        @test plan1.us == us_copy
        @test plan1.chain_data == chain_copy

        arena = M.map_many_plan_arena[slot]
        @test !(arena.us === plan1.us)
        @test !(arena.chain_data === plan1.chain_data)
    finally
        MD.MAP_LEQ_MANY_PLAN_MIN_LEN[] = old_min
        MD.MAP_LEQ_MANY_PLAN_MAX_PER_THREAD[] = old_max
    end
end

@testset "Modules direct oracle checks across non-QQ fields" begin
    nonqq_fields = (CM.F2(), CM.F3(), CM.Fp(5), CM.RealField(Float64; rtol=1e-10, atol=1e-12))

    for field in nonqq_fields
        K = CM.coeff_type(field)
        c(x) = CM.coerce(field, x)

        # map_leq and map_leq_many: exact oracle on a chain (unique path).
        P = chain_poset(4)
        dims = [2, 2, 2, 2]
        E12 = K[c(1) c(2); c(0) c(3)]
        E23 = K[c(2) c(0); c(4) c(1)]
        E34 = K[c(1) c(1); c(0) c(2)]
        edge = Dict{Tuple{Int,Int}, Matrix{K}}(
            (1, 2) => E12,
            (2, 3) => E23,
            (3, 4) => E34,
        )
        M = MD.PModule{K}(P, dims, edge; field=field)
        cc = MD._get_cover_cache(P)
        FF._clear_chain_parent_cache!(cc)
        for d in M.map_compose
            empty!(d)
        end

        oracle13 = E23 * E12
        oracle14 = E34 * oracle13
        @test MD.map_leq(M, 1, 3; cache=cc) == oracle13
        @test MD.map_leq(M, 1, 4; cache=cc) == oracle14

        pairs = Tuple{Int,Int}[(1, 1), (1, 2), (1, 3), (1, 4), (2, 4)]
        batch = MD.map_leq_many(M, pairs; cache=cc)
        @test batch[1] == CM.eye(field, dims[1])
        @test batch[3] == oracle13
        @test batch[4] == oracle14

        out = Vector{Matrix{K}}(undef, length(pairs))
        MD.map_leq_many!(out, M, pairs; cache=cc)
        @test out == batch

        # direct_sum: hand-assembled block oracle.
        Q2 = chain_poset(2)
        A = MD.PModule{K}(Q2, [2, 1], Dict((1, 2) => reshape(K[c(2), c(3)], 1, 2)); field=field)
        B = MD.PModule{K}(Q2, [1, 2], Dict((1, 2) => reshape(K[c(5), c(7)], 2, 1)); field=field)
        S = MD.direct_sum(A, B)
        M12 = S.edge_maps[1, 2]
        oracle12 = K[
            c(2) c(3) c(0);
            c(0) c(0) c(5);
            c(0) c(0) c(7)
        ]
        @test M12 == oracle12

        # canonical injections/projections are algebraically correct.
        _, iA, iB, pA, pB = MD.direct_sum_with_maps(A, B)
        @test pA.comps[1] * iA.comps[1] == CM.eye(field, A.dims[1])
        @test pB.comps[2] * iB.comps[2] == CM.eye(field, B.dims[2])
        @test pA.comps[1] * iB.comps[1] == CM.zeros(field, A.dims[1], B.dims[1])
    end
end

@testset "Modules randomized differential suite vs naive chain oracle" begin
    function rand_coeff(rng::Random.AbstractRNG, field)
        if field isa CM.RealField
            return CM.coerce(field, randn(rng))
        end
        v = rand(rng, -3:3)
        v == 0 && (v = 1)
        return CM.coerce(field, v)
    end

    function chain_oracle(M::MD.PModule{K}, u::Int, v::Int) where {K}
        u == v && return CM.eye(M.field, M.dims[v])
        A = M.edge_maps[u, u + 1]
        for k in (u + 1):(v - 1)
            A = FL._matmul(M.edge_maps[k, k + 1], A)
        end
        return A
    end

    function mat_eq_field(field, A, B)
        if field isa CM.RealField
            return isapprox(Matrix{Float64}(A), Matrix{Float64}(B); rtol=1e-8, atol=1e-9)
        end
        return A == B
    end

    rng = Random.MersenneTwister(0x5A17)
    densities = (0.2, 0.5, 0.8)
    sizes = (4, 6, 8)
    trials = 2

    with_fields(FIELDS_FULL) do field
        K = CM.coeff_type(field)
        for n in sizes
            P = chain_poset(n)
            cc = MD._get_cover_cache(P)
            for dens in densities
                for t in 1:trials
                    dims = [rand(rng, 1:4) for _ in 1:n]
                    edge = Dict{Tuple{Int,Int}, Matrix{K}}()
                    for u in 1:(n - 1)
                        A = CM.zeros(field, dims[u + 1], dims[u])
                        @inbounds for i in 1:dims[u + 1], j in 1:dims[u]
                            rand(rng) < dens || continue
                            A[i, j] = rand_coeff(rng, field)
                        end
                        edge[(u, u + 1)] = A
                    end

                    M = MD.PModule{K}(P, dims, edge; field=field)
                    FF._clear_chain_parent_cache!(cc)
                    for d in M.map_compose
                        empty!(d)
                    end

                    pairs = Tuple{Int,Int}[]
                    sizehint!(pairs, 32)
                    for _ in 1:32
                        u = rand(rng, 1:n)
                        v = rand(rng, u:n)
                        push!(pairs, (u, v))
                    end

                    batch = MD.map_leq_many(M, pairs; cache=cc)
                    out = Vector{Matrix{K}}(undef, length(pairs))
                    MD.map_leq_many!(out, M, pairs; cache=cc)
                    @test length(batch) == length(pairs)
                    @test length(out) == length(pairs)

                    @inbounds for i in eachindex(pairs)
                        u, v = pairs[i]
                        oracle = chain_oracle(M, u, v)
                        got = MD.map_leq(M, u, v; cache=cc)
                        @test mat_eq_field(field, got, oracle)
                        @test mat_eq_field(field, batch[i], oracle)
                        @test mat_eq_field(field, out[i], oracle)
                    end
                end
            end
        end
    end
end

@testset "Modules randomized branching-poset oracles across fields" begin
    function rand_nonzero_coeff(rng::Random.AbstractRNG, field)
        if field isa CM.RealField
            v = 0.0
            while v == 0.0
                v = randn(rng)
            end
            return CM.coerce(field, v)
        end
        v = zero(CM.coeff_type(field))
        while iszero(v)
            v = CM.coerce(field, rand(rng, -5:5))
        end
        return v
    end

    function random_branching_poset(rng::Random.AbstractRNG, n::Int)
        L = falses(n, n)
        @inbounds for i in 1:n
            L[i, i] = true
        end
        # Force a branching core 1<2,1<3,2<4,3<4.
        if n >= 4
            L[1, 2] = true
            L[1, 3] = true
            L[2, 4] = true
            L[3, 4] = true
        end
        for i in 1:n
            for j in (i + 1):n
                rand(rng) < 0.28 || continue
                L[i, j] = true
            end
        end
        # Transitive closure.
        for k in 1:n
            for i in 1:n
                L[i, k] || continue
                @inbounds for j in 1:n
                    L[k, j] || continue
                    L[i, j] = true
                end
            end
        end
        return FF.FinitePoset(L; check=false)
    end

    rng = Random.MersenneTwister(0xBADA55)
    trials = 4
    d = 3

    mat_eq_field(field, A, B) = field isa CM.RealField ?
        isapprox(Matrix{Float64}(A), Matrix{Float64}(B); rtol=1e-8, atol=1e-9) :
        (A == B)

    with_fields(FIELDS_FULL) do field
        K = CM.coeff_type(field)
        for _ in 1:trials
            P = random_branching_poset(rng, 7)
            n = FF.nvertices(P)
            scales = [[rand_nonzero_coeff(rng, field) for _ in 1:d] for _ in 1:n]
            dims = fill(d, n)
            edge = Dict{Tuple{Int,Int}, Matrix{K}}()
            for (u, v) in FF.cover_edges(P)
                A = CM.zeros(field, d, d)
                @inbounds for k in 1:d
                    A[k, k] = scales[v][k] / scales[u][k]
                end
                edge[(u, v)] = A
            end
            M = MD.PModule{K}(P, dims, edge; field=field)
            cc = MD._get_cover_cache(P)
            FF._clear_chain_parent_cache!(cc)
            for dct in M.map_compose
                empty!(dct)
            end

            pairs = Tuple{Int,Int}[]
            for _ in 1:24
                u = rand(rng, 1:n)
                v = rand(rng, 1:n)
                FF.leq(P, u, v) || continue
                push!(pairs, (u, v))
            end
            isempty(pairs) && continue

            batch = MD.map_leq_many(M, pairs; cache=cc)
            @inbounds for i in eachindex(pairs)
                u, v = pairs[i]
                oracle = CM.zeros(field, d, d)
                for k in 1:d
                    oracle[k, k] = scales[v][k] / scales[u][k]
                end
                @test mat_eq_field(field, batch[i], oracle)
                @test mat_eq_field(field, MD.map_leq(M, u, v; cache=cc), oracle)
            end
        end
    end
end

@testset "Modules threaded batch parity under contention" begin
    Threads.nthreads() > 1 || return

    field = CM.QQField()
    K = CM.coeff_type(field)
    c(x) = CM.coerce(field, x)
    rng = Random.MersenneTwister(0xC011EC7)

    P = chain_poset(18)
    dims = ones(Int, 18)
    edge = Dict{Tuple{Int,Int}, Matrix{K}}()
    for (u, v) in FF.cover_edges(P)
        edge[(u, v)] = reshape([c(rand(rng, 1:9))], 1, 1)
    end
    M = MD.PModule{K}(P, dims, edge; field=field)
    cc = MD._get_cover_cache(P)
    FF._clear_chain_parent_cache!(cc)
    for dct in M.map_compose
        empty!(dct)
    end

    base_pairs = Tuple{Int,Int}[]
    for _ in 1:64
        u = rand(rng, 1:18)
        v = rand(rng, u:18)
        push!(base_pairs, (u, v))
    end
    pairs = vcat(base_pairs, base_pairs, base_pairs, base_pairs)
    serial = MD.map_leq_many(M, pairs; cache=cc)

    nruns = max(8, 4 * Threads.nthreads())
    threaded = Vector{Vector{Matrix{K}}}(undef, nruns)
    Threads.@threads :static for r in 1:nruns
        threaded[r] = MD.map_leq_many(M, pairs; cache=cc)
    end
    for r in 1:nruns
        @test threaded[r] == serial
    end

    threaded_inplace = Vector{Vector{Matrix{K}}}(undef, nruns)
    Threads.@threads :static for r in 1:nruns
        out = Vector{Matrix{K}}(undef, length(pairs))
        MD.map_leq_many!(out, M, pairs; cache=cc)
        threaded_inplace[r] = out
    end
    for r in 1:nruns
        @test threaded_inplace[r] == serial
    end
end

@testset "Modules backendized map storage oracle (Nemo path)" begin
    FL._have_nemo() || return

    old_thr = FL.NEMO_THRESHOLD[]
    try
        FL.NEMO_THRESHOLD[] = 1
        field = CM.QQField()
        K = CM.coeff_type(field)
        c(x) = CM.coerce(field, x)

        P = chain_poset(3)
        dims = [4, 4, 4]
        E12 = K[c(1) c(2) c(0) c(0);
                c(0) c(1) c(3) c(0);
                c(0) c(0) c(1) c(4);
                c(0) c(0) c(0) c(1)]
        E23 = K[c(2) c(0) c(0) c(0);
                c(0) c(3) c(0) c(0);
                c(0) c(0) c(5) c(0);
                c(0) c(0) c(0) c(7)]
        M = MD.PModule{K}(P, dims, Dict((1, 2) => E12, (2, 3) => E23); field=field)
        @test M.edge_maps[1, 2] isa CM.BackendMatrix{K}
        @test M.edge_maps[2, 3] isa CM.BackendMatrix{K}

        cc = MD._get_cover_cache(P)
        for dct in M.map_compose
            empty!(dct)
        end
        oracle13 = E23 * E12
        @test Matrix{K}(MD.map_leq(M, 1, 3; cache=cc)) == oracle13
        out = MD.map_leq_many(M, Tuple{Int,Int}[(1, 2), (1, 3), (2, 3)]; cache=cc)
        @test Matrix{K}(out[2]) == oracle13
    finally
        FL.NEMO_THRESHOLD[] = old_thr
    end
end

@testset "Modules perf regression guard (map_leq_many memo path)" begin
    function old_map_leq_chain(M::MD.PModule{K}, preds::Vector{Vector{Int}}, u::Int, v::Int) where {K}
        u == v && return CM.eye(M.field, M.dims[v])
        uv = (u, v)
        if haskey(M.edge_maps, u, v)
            return M.edge_maps[u, v]
        end
        idx = findfirst(x -> x != u && FF.leq(M.Q, u, x), preds[v])
        w = (idx === nothing) ? u : preds[v][idx]
        return FL._matmul(M.edge_maps[w, v], old_map_leq_chain(M, preds, u, w))
    end

    field = CM.QQField()
    K = CM.coeff_type(field)
    c(x) = CM.coerce(field, x)
    P = chain_poset(12)
    dims = fill(2, 12)
    edge = Dict{Tuple{Int,Int}, Matrix{K}}()
    for u in 1:11
        edge[(u, u + 1)] = K[c(1) c(u % 3 + 1); c(0) c(1)]
    end
    M = MD.PModule{K}(P, dims, edge; field=field)
    cc = MD._get_cover_cache(P)
    FF._clear_chain_parent_cache!(cc)
    for dct in M.map_compose
        empty!(dct)
    end

    rng = Random.MersenneTwister(0xFEED)
    seed_pairs = Tuple{Int,Int}[]
    for _ in 1:32
        u = rand(rng, 1:12)
        v = rand(rng, u:12)
        push!(seed_pairs, (u, v))
    end
    pairs = vcat(seed_pairs, seed_pairs, seed_pairs, seed_pairs, seed_pairs)
    preds = [collect(FF._preds(cc, v)) for v in 1:12]

    # warmup
    MD.map_leq_many(M, pairs; cache=cc)
    for (u, v) in pairs
        old_map_leq_chain(M, preds, u, v)
    end

    old_times = Float64[]
    new_times = Float64[]
    old_allocs = Int[]
    new_allocs = Int[]
    for _ in 1:4
        GC.gc()
        t_old = @timed begin
            for (u, v) in pairs
                old_map_leq_chain(M, preds, u, v)
            end
        end
        push!(old_times, t_old.time)
        push!(old_allocs, t_old.bytes)

        GC.gc()
        t_new = @timed MD.map_leq_many(M, pairs; cache=cc)
        push!(new_times, t_new.time)
        push!(new_allocs, t_new.bytes)
    end

    sort!(old_times); sort!(new_times)
    sort!(old_allocs); sort!(new_allocs)
    med_old_t = old_times[2]
    med_new_t = new_times[2]
    med_old_a = old_allocs[2]
    med_new_a = new_allocs[2]

    @test med_new_t <= 1.20 * med_old_t
    @test med_new_a <= med_old_a
end

@testset "CoverEdgeMapStore correctness" begin
    # simple chain 1 < 2 < 3
    leq = Bool[
        1 1 1;
        0 1 1;
        0 0 1
    ]
    Q = FF.FinitePoset(leq)
    field = CM.QQField()
    K = CM.coeff_type(field)
    c(x) = CM.coerce(field, x)

    dims = [1, 1, 1]

    # cover edges: (1,2), (2,3)
    edge_maps = Dict{Tuple{Int,Int}, Matrix{K}}()
    edge_maps[(1,2)] = reshape([c(1)], 1, 1)
    edge_maps[(2,3)] = reshape([c(2)], 1, 1)

    M = MD.PModule{K}(Q, dims, edge_maps; field=field)

    @test M.edge_maps[1, 2] == reshape([c(1)], 1, 1)
    @test M.edge_maps[2, 3] == reshape([c(2)], 1, 1)
    @test_throws KeyError M.edge_maps[1, 3]  # not a cover edge

    # lock in the stricter API (no tuple indexing)
    @test_throws MethodError M.edge_maps[(1,2)]

    # check predecessor alignment
    @test M.edge_maps.preds[2] == [1]
    @test M.edge_maps.preds[3] == [2]
    @test M.edge_maps.maps_from_pred[2][1] == reshape([c(1)], 1, 1)
    @test M.edge_maps.maps_from_pred[3][1] == reshape([c(2)], 1, 1)

    # iteration yields cover edges
    seen = Set{Tuple{Int,Int}}()
    for (e, A) in M.edge_maps
        push!(seen, e)
        u, v = e
        @test A == M.edge_maps[u, v]
    end
    @test seen == Set([(1,2), (2,3)])
end
