#!/usr/bin/env julia
# FiniteFringe hot-path microbenchmarks.
# This script benchmarks the production FiniteFringe kernels most affected by
# cache/layout changes: repeated fiber-dimension queries and repeated
# hom-dimension queries on fixed fringe-module pairs.

using Random
using SparseArrays

try
    using TamerOp
catch
    include(joinpath(@__DIR__, "..", "src", "TamerOp.jl"))
    using .TamerOp
end

const FF = TamerOp.FiniteFringe
const CM = TamerOp.CoreModules

function _parse_int_arg(args, key::String, default::Int)
    for arg in args
        startswith(arg, key * "=") || continue
        return parse(Int, split(arg, "=", limit=2)[2])
    end
    return default
end

function _bench(name::AbstractString, f::Function; reps::Int)
    GC.gc()
    f()
    GC.gc()
    times_ms = Vector{Float64}(undef, reps)
    alloc_kib = Vector{Float64}(undef, reps)
    for i in 1:reps
        m = @timed f()
        times_ms[i] = 1000.0 * m.time
        alloc_kib[i] = m.bytes / 1024.0
    end
    sort!(times_ms)
    sort!(alloc_kib)
    mid = cld(reps, 2)
    println(rpad(name, 34),
            " median_time=", round(times_ms[mid], digits=3), " ms",
            " median_alloc=", round(alloc_kib[mid], digits=1), " KiB")
    return (ms=times_ms[mid], kib=alloc_kib[mid])
end

function _random_poset(n::Int; p::Float64=0.035, seed::Int=0xFF01)
    rng = Random.MersenneTwister(seed)
    leq = falses(n, n)
    @inbounds for i in 1:n
        leq[i, i] = true
        for j in (i + 1):n
            leq[i, j] = rand(rng) < p
        end
    end
    @inbounds for k in 1:n, i in 1:n, j in 1:n
        leq[i, j] = leq[i, j] || (leq[i, k] && leq[k, j])
    end
    return FF.FinitePoset(leq; check=false)
end

function _random_fringe_module(P::FF.AbstractPoset, field::CM.AbstractCoeffField;
                               nu::Int, nd::Int, density::Float64, seed::Int)
    rng = Random.MersenneTwister(seed)
    K = CM.coeff_type(field)
    n = FF.nvertices(P)
    U = Vector{FF.Upset}(undef, nu)
    D = Vector{FF.Downset}(undef, nd)
    @inbounds for i in 1:nu
        U[i] = FF.upset_closure(P, BitVector(rand(rng, Bool, n)))
    end
    @inbounds for j in 1:nd
        D[j] = FF.downset_closure(P, BitVector(rand(rng, Bool, n)))
    end
    phi = spzeros(K, nd, nu)
    @inbounds for j in 1:nd, i in 1:nu
        FF.intersects(U[i], D[j]) || continue
        rand(rng) < density || continue
        v = rand(rng, -3:3)
        v == 0 && continue
        phi[j, i] = CM.coerce(field, v)
    end
    return FF.FringeModule{K}(P, U, D, phi; field=field)
end

function _fiber_scan_reference(M::FF.FringeModule, queries::Vector{Int})
    s = 0
    @inbounds for q in queries
        cols = findall(U -> U.mask[q], M.U)
        rows = findall(D -> D.mask[q], M.D)
        isempty(cols) || isempty(rows) && continue
        s += TamerOp.FieldLinAlg.rank_restricted(M.field, M.phi, rows, cols)
    end
    return s
end

function _fiber_current(M::FF.FringeModule, queries::Vector{Int})
    s = 0
    @inbounds for q in queries
        s += FF.fiber_dimension(M, q)
    end
    return s
end

function _fiber_first_query_fullindex(M::FF.FringeModule, q::Int)
    idx = FF._build_fiber_query_index(M)
    M.fiber_index[] = idx
    return FF.fiber_dimension(M, q)
end

function _fiber_first_queries_current(M::FF.FringeModule, queries::Vector{Int})
    s = 0
    @inbounds for q in queries
        s += FF.fiber_dimension(M, q)
    end
    return s
end

function _fiber_first_queries_fullindex(M::FF.FringeModule, queries::Vector{Int})
    idx = FF._build_fiber_query_index(M)
    M.fiber_index[] = idx
    return _fiber_first_queries_current(M, queries)
end

function _fiber_first_query_vecslice(M::FF.FringeModule, q::Int)
    rows, cols = FF._build_fiber_query_slice(M.U, M.D, q)
    if isempty(rows) || isempty(cols)
        return 0
    end
    return TamerOp.FieldLinAlg.rank_restricted(M.field, M.phi, rows, cols)
end

function _fiber_first_query_words(M::FF.FringeModule, q::Int)
    row_words, col_words, nr, nc = FF._build_fiber_query_slice_words(M.U, M.D, q)
    if nr == 0 || nc == 0
        return 0
    end
    return TamerOp.FieldLinAlg.rank_restricted_words(M.field, M.phi, row_words, col_words, nr, nc;
                                                          nrows=size(M.phi, 1), ncols=size(M.phi, 2))
end

function _fiber_query_slice(M::FF.FringeModule, q::Int)
    rows, cols = FF._build_fiber_query_slice(M.U, M.D, q)
    return length(rows) + length(cols)
end

function _old_component_target_reference(M::FF.FringeModule, N::FF.FringeModule)
    adj = FF._cover_undirected_adjacency(M.P)
    Ucomp_masks = Vector{Vector{BitVector}}(undef, length(M.U))
    Dcomp_masks = Vector{Vector{BitVector}}(undef, length(N.D))
    @inbounds for i in eachindex(M.U)
        _, _, masks, _ = FF._component_data(adj, M.U[i].mask)
        Ucomp_masks[i] = masks
    end
    @inbounds for t in eachindex(N.D)
        _, _, masks, _ = FF._component_data(adj, N.D[t].mask)
        Dcomp_masks[t] = masks
    end

    total = 0
    targetU = getfield.(N.U, :mask)
    targetD = getfield.(M.D, :mask)
    @inbounds for cmasks in Ucomp_masks
        for mask in cmasks
            for tmask in targetU
                FF.is_subset(mask, tmask) && (total += 1)
            end
        end
    end
    @inbounds for cmasks in Dcomp_masks
        for mask in cmasks
            for tmask in targetD
                FF.is_subset(mask, tmask) && (total += 1)
            end
        end
    end
    return total
end

function _new_component_target_current(M::FF.FringeModule, N::FF.FringeModule)
    hc = FF._ensure_hom_cache!(M)
    entry = FF._ensure_pair_cache!(hc, N)
    entry.layout_sketch = nothing
    entry.layout_plan = nothing
    sketch = FF._ensure_hom_layout_sketch!(M, N)
    return sketch.V1_dim + sketch.V2_dim
end

function _old_sparse_hom_kernel(field, plan)
    T = plan.T
    S = plan.S
    rank_backend = field isa CM.QQField ? :julia_sparse : :auto
    if size(S, 2) <= size(T, 2)
        rS = TamerOp.FieldLinAlg.rank(field, S; backend=rank_backend)
        rUnion, rT = FF._rank_hcat_signed_sparse_workspace_with_prefix_rank!(
            field, plan.hcat_buf, T, S, plan.nnzT, size(T, 2)
        )
        return rT + rS - rUnion
    end
    rT = TamerOp.FieldLinAlg.rank(field, T; backend=rank_backend)
    rUnion, rS = FF._rank_hcat_signed_sparse_workspace_with_prefix_rank!(
        field, plan.hcat_buf_rev, S, T, plan.nnzS, size(S, 2)
    )
    return rT + rS - rUnion
end

function _old_sparse_onepass_rref_reducer(plan)
    T = plan.T
    S = plan.S
    if size(S, 2) > size(T, 2)
        error("_old_sparse_onepass_rref_reducer currently expects |S| <= |T|")
    end
    union_red = TamerOp.FieldLinAlg._SparseRREF{eltype(T.nzval)}(size(T, 2) + size(S, 2))
    side_red = TamerOp.FieldLinAlg._SparseRREF{eltype(T.nzval)}(size(S, 2))
    union_row = TamerOp.FieldLinAlg.SparseRow{eltype(T.nzval)}(Int[], eltype(T.nzval)[])
    side_row = TamerOp.FieldLinAlg.SparseRow{eltype(T.nzval)}(Int[], eltype(T.nzval)[])
    m = size(T, 1)
    nL = size(T, 2)
    nR = size(S, 2)
    max_union = min(m, nL + nR)
    max_side = min(m, nR)
    runion = 0
    rside = 0
    rleft = 0
    @inbounds for i in 1:m
        if runion < max_union || rside < max_side
            FF._fill_sparse_row_from_plan!(side_row, S, plan.s_rows, i)
        else
            resize!(side_row.idx, 0)
            resize!(side_row.val, 0)
        end
        if runion < max_union
            FF._fill_sparse_union_row_from_right!(union_row, T, plan.t_rows, i, side_row, nL)
            if !isempty(union_row) && TamerOp.FieldLinAlg._sparse_rref_push_homogeneous!(union_red, union_row)
                runion += 1
                union_red.pivot_cols[end] <= nL && (rleft += 1)
            end
        end
        if rside < max_side && !isempty(side_row) &&
           TamerOp.FieldLinAlg._sparse_rref_push_homogeneous!(side_red, side_row)
            rside += 1
        end
        if runion == max_union && rside == max_side
            break
        end
    end
    return runion, rleft, rside
end

function main(args=ARGS)
    reps = _parse_int_arg(args, "--reps", 5)
    n = _parse_int_arg(args, "--n", 80)
    nu = _parse_int_arg(args, "--nu", 18)
    nd = _parse_int_arg(args, "--nd", 18)
    fiber_queries = _parse_int_arg(args, "--fiber_queries", 800)

    println("FiniteFringe hot-path microbench")
    println("reps=$(reps), n=$(n), nu=$(nu), nd=$(nd), fiber_queries=$(fiber_queries)\n")

    field = CM.QQField()
    P = _random_poset(n; seed=Int(0xFF01))
    Mf = _random_fringe_module(P, field; nu=max(nu, 20), nd=max(nd, 20), density=0.18, seed=Int(0xFF11))
    rng = Random.MersenneTwister(Int(0xFF12))
    queries = rand(rng, 1:FF.nvertices(P), fiber_queries)

    expected_f = _fiber_scan_reference(Mf, queries)
    got_f = _fiber_current(Mf, queries)
    expected_f == got_f || error("fiber parity failed: expected $(expected_f), got $(got_f)")

    println("== fiber_dimension batch ==")
    b_f_scan = _bench("fiber scan baseline", () -> _fiber_scan_reference(Mf, queries); reps=reps)
    b_f_cold = _bench("fiber current fresh", () -> begin
        Mc = _random_fringe_module(P, field; nu=max(nu, 20), nd=max(nd, 20), density=0.18, seed=Int(0xFF11))
        _fiber_current(Mc, queries)
    end; reps=reps)
    _fiber_current(Mf, queries)
    b_f_warm = _bench("fiber current warm", () -> _fiber_current(Mf, queries); reps=reps)
    println("fiber speedup fresh/scan = ", round(b_f_scan.ms / b_f_cold.ms, digits=2), "x")
    println("fiber speedup warm/scan  = ", round(b_f_scan.ms / b_f_warm.ms, digits=2), "x")
    println()

    P_big = _random_poset(max(128, 2n); seed=Int(0xFF13))
    M_big = _random_fringe_module(P_big, field; nu=max(260, nu * 12), nd=max(260, nd * 12), density=0.08, seed=Int(0xFF14))
    q_big = rand(rng, 1:FF.nvertices(P_big))
    nfirst = FF._fiber_lazy_full_index_after(M_big)
    qs_big = rand(rng, 1:FF.nvertices(P_big), nfirst)
    println("== fiber_dimension first query (no eager index) ==")
    b_f_slice = _bench("fiber large slice build", () -> begin
        Mc = FF.FringeModule{CM.coeff_type(field)}(P_big, M_big.U, M_big.D, M_big.phi; field=field)
        _fiber_query_slice(Mc, q_big)
    end; reps=reps)
    b_f_rank_vec = _bench("fiber large slice+rank vec", () -> begin
        Mc = FF.FringeModule{CM.coeff_type(field)}(P_big, M_big.U, M_big.D, M_big.phi; field=field)
        _fiber_first_query_vecslice(Mc, q_big)
    end; reps=reps)
    b_f_rank_words = _bench("fiber large slice+rank words", () -> begin
        Mc = FF.FringeModule{CM.coeff_type(field)}(P_big, M_big.U, M_big.D, M_big.phi; field=field)
        _fiber_first_query_words(Mc, q_big)
    end; reps=reps)
    b_f_fullbuild = _bench("fiber large full build", () -> begin
        Mc = FF.FringeModule{CM.coeff_type(field)}(P_big, M_big.U, M_big.D, M_big.phi; field=field)
        FF._build_fiber_query_index(Mc)
    end; reps=reps)
    b_f_large_old = _bench("fiber large first query old", () -> begin
        Mc = FF.FringeModule{CM.coeff_type(field)}(P_big, M_big.U, M_big.D, M_big.phi; field=field)
        _fiber_first_query_fullindex(Mc, q_big)
    end; reps=reps)
    b_f_large = _bench("fiber large first query", () -> begin
        Mc = FF.FringeModule{CM.coeff_type(field)}(P_big, M_big.U, M_big.D, M_big.phi; field=field)
        FF.fiber_dimension(Mc, q_big)
    end; reps=reps)
    seq_old_name = "fiber large first$(nfirst) old"
    seq_cur_name = "fiber large first$(nfirst) current"
    b_f_large_seq_old = _bench(seq_old_name, () -> begin
        Mc = FF.FringeModule{CM.coeff_type(field)}(P_big, M_big.U, M_big.D, M_big.phi; field=field)
        _fiber_first_queries_fullindex(Mc, qs_big)
    end; reps=reps)
    b_f_large_seq = _bench(seq_cur_name, () -> begin
        Mc = FF.FringeModule{CM.coeff_type(field)}(P_big, M_big.U, M_big.D, M_big.phi; field=field)
        _fiber_first_queries_current(Mc, qs_big)
    end; reps=reps)
    println("fiber slice/full build    = ", round(b_f_fullbuild.ms / b_f_slice.ms, digits=2), "x")
    println("fiber words/vec rank      = ", round(b_f_rank_vec.ms / b_f_rank_words.ms, digits=2), "x")
    println("fiber first query speedup = ", round(b_f_large_old.ms / b_f_large.ms, digits=2), "x")
    println("fiber first$(nfirst) speedup     = ", round(b_f_large_seq_old.ms / b_f_large_seq.ms, digits=2), "x")
    println("fiber large first query = ", round(b_f_large.ms, digits=3), " ms")
    println()

    M = _random_fringe_module(P, field; nu=nu, nd=nd, density=0.16, seed=Int(0xFF21))
    N = _random_fringe_module(P, field; nu=nu, nd=nd, density=0.16, seed=Int(0xFF22))
    h_ref = FF._hom_dimension_with_path(M, N, :sparse_path)
    h_auto = FF.hom_dimension(M, N)
    h_ref == h_auto || error("hom parity failed: sparse=$(h_ref), auto=$(h_auto)")

    println("== hom_dimension repeated pair ==")
    b_h_fresh = _bench("hom auto fresh pair", () -> begin
        Mc = _random_fringe_module(P, field; nu=nu, nd=nd, density=0.16, seed=Int(0xFF21))
        Nc = _random_fringe_module(P, field; nu=nu, nd=nd, density=0.16, seed=Int(0xFF22))
        FF.hom_dimension(Mc, Nc)
    end; reps=reps)
    FF.hom_dimension(M, N)
    b_h_warm = _bench("hom auto warm pair", () -> FF.hom_dimension(M, N); reps=reps)
    b_h_sparse = _bench("hom sparse path warm", () -> FF._hom_dimension_with_path(M, N, :sparse_path); reps=reps)
    println("hom warmup gain = ", round(b_h_fresh.ms / b_h_warm.ms, digits=2), "x")
    println("hom auto/sparse = ", round(b_h_sparse.ms / b_h_warm.ms, digits=2), "x")
    println()
    println("== component-target build / route selection ==")
    ref_targets = _old_component_target_reference(M, N)
    cur_targets = _new_component_target_current(M, N)
    ref_targets == cur_targets || error("component-target parity failed: old=$(ref_targets), new=$(cur_targets)")
    _bench("component targets old", () -> _old_component_target_reference(M, N); reps=reps)
    _bench("component targets new", () -> _new_component_target_current(M, N); reps=reps)
    b_route = _bench("hom route fresh", () -> begin
        Mc = _random_fringe_module(P, field; nu=nu, nd=nd, density=0.16, seed=Int(0xFF31))
        Nc = _random_fringe_module(P, field; nu=nu, nd=nd, density=0.16, seed=Int(0xFF32))
        FF._select_hom_internal_path!(Mc, Nc)
    end; reps=reps)
    println("hom route fresh = ", round(b_route.ms, digits=3), " ms")

    sparse_plan = FF._ensure_sparse_hom_plan!(M, N)
    old_sparse = _old_sparse_hom_kernel(field, sparse_plan)
    old_sparse == h_ref || error("old sparse kernel parity failed: old=$(old_sparse), new=$(h_ref)")
    small = size(sparse_plan.S, 2) <= size(sparse_plan.T, 2) ? sparse_plan.S : sparse_plan.T
    small_cols = size(small, 2)
    union_call = if size(sparse_plan.S, 2) <= size(sparse_plan.T, 2)
        () -> FF._rank_hcat_signed_sparse_workspace_with_prefix_rank!(
            field, sparse_plan.hcat_buf,
            sparse_plan.T, sparse_plan.S, sparse_plan.nnzT, size(sparse_plan.T, 2)
        )
    else
        () -> FF._rank_hcat_signed_sparse_workspace_with_prefix_rank!(
            field, sparse_plan.hcat_buf_rev,
            sparse_plan.S, sparse_plan.T, sparse_plan.nnzS, size(sparse_plan.S, 2)
        )
    end
    onepass_call = if size(sparse_plan.S, 2) <= size(sparse_plan.T, 2)
        () -> FF._rank_hcat_signed_sparse_rowplans_with_side_rank!(
            sparse_plan.union_red, sparse_plan.s_red, sparse_plan.union_row, sparse_plan.side_row,
            sparse_plan.T, sparse_plan.t_rows, sparse_plan.S, sparse_plan.s_rows, size(sparse_plan.T, 2)
        )
    else
        () -> FF._rank_hcat_signed_sparse_rowplans_with_side_rank!(
            sparse_plan.union_red, sparse_plan.t_red, sparse_plan.union_row, sparse_plan.side_row,
            sparse_plan.S, sparse_plan.s_rows, sparse_plan.T, sparse_plan.t_rows, size(sparse_plan.S, 2)
        )
    end
    sparse_uncached_call = () -> begin
        FF._clear_sparse_side_rank_cache!(sparse_plan)
        FF._hom_dimension_with_path(M, N, :sparse_path)
    end
    sparse_cached_call = () -> FF._hom_dimension_with_path(M, N, :sparse_path)
    println()
    println("== sparse-rank subcomponents ==")
    println("small rank backend = ",
            TamerOp.FieldLinAlg._choose_linalg_backend(field, small; op=:rank, backend=:auto),
            " (cols=", small_cols, ", nnz=", nnz(small), ")")
    _bench("hom sparse small rank", () -> TamerOp.FieldLinAlg.rank(field, small); reps=reps)
    _bench("hom sparse union prefix", union_call; reps=reps)
    b_old_kernel = _bench("hom sparse old kernel", () -> _old_sparse_hom_kernel(field, sparse_plan); reps=reps)
    b_onepass = _bench("hom sparse one-pass", onepass_call; reps=reps)
    if size(sparse_plan.S, 2) <= size(sparse_plan.T, 2)
        b_ref = _bench("hom sparse ref reducer", onepass_call; reps=reps)
        b_rref = _bench("hom sparse rref reducer", () -> _old_sparse_onepass_rref_reducer(sparse_plan); reps=reps)
        println("ref/rref reducer      = ", round(b_rref.ms / b_ref.ms, digits=2), "x")
    end
    sparse_cached_call()
    b_sparse_uncached = _bench("hom sparse no sidecache", sparse_uncached_call; reps=reps)
    b_sparse_cached = _bench("hom sparse cached side", sparse_cached_call; reps=reps)
    println("one-pass/old kernel = ", round(b_old_kernel.ms / b_onepass.ms, digits=2), "x")
    println("sidecache speedup    = ", round(b_sparse_uncached.ms / b_sparse_cached.ms, digits=2), "x")
end

main()
