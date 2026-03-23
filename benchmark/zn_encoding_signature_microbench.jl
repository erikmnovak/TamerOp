#!/usr/bin/env julia
#
# zn_encoding_signature_microbench.jl
#
# Purpose
# - Micro-benchmark two ZnEncoding optimizations:
#   1) packed signature keys (`SigKey`) for region lookup, compared to tuple-key baseline.
#   2) cover-edge iteration (`for (u,v) in cover_edges(Q)`) vs dense scan baseline (`if C[u,v]`).
#   3) `pmodule_on_box` incremental membership + basis cache, compared to a naive
#      per-cell full-scan baseline.
#   4) Flange `dim_at` default auto path (size-aware), compared to the old
#      always-submatrix materialization path.
#   5) Flange `degree_matrix!` buffer-reuse paths, compared to allocating
#      `degree_matrix` calls.
#
# Scope
# - Uses a deterministic synthetic Z^n flange fixture.
# - Reports median wall-time and allocation over repeated runs.
# - This is a focused kernel benchmark, not an end-to-end library benchmark.
#
# Usage
#   julia --project=. benchmark/zn_encoding_signature_microbench.jl
#   julia --project=. benchmark/zn_encoding_signature_microbench.jl --reps=12 --nqueries=30000 --grid=18
#   julia --project=. benchmark/zn_encoding_signature_microbench.jl --pmodule=1 --pmodule_reps=4
#   julia --project=. benchmark/zn_encoding_signature_microbench.jl --pmodule_all_fields=1
#   julia --project=. benchmark/zn_encoding_signature_microbench.jl --flange_degree_matrix=1 --flange_degree_matrix_reps=6

using Random

try
    using TamerOp
catch
    include(joinpath(@__DIR__, "..", "src", "TamerOp.jl"))
    using .TamerOp
end

const TO = TamerOp.Advanced
const ZE = TO.ZnEncoding
const FZ = TO.FlangeZn
const CM = TO.CoreModules
const FF = TO.FiniteFringe
const FL = TamerOp.FieldLinAlg

function _parse_arg(args, key::String, default::Int)
    for a in args
        startswith(a, key * "=") || continue
        return max(1, parse(Int, split(a, "=", limit=2)[2]))
    end
    return default
end

function _parse_bool(args, key::String, default::Bool)
    for a in args
        startswith(a, key * "=") || continue
        v = lowercase(split(a, "=", limit=2)[2])
        if v in ("1", "true", "yes", "y", "on")
            return true
        elseif v in ("0", "false", "no", "n", "off")
            return false
        else
            error("Invalid boolean for $key: $v")
        end
    end
    return default
end

function _bench(name::AbstractString, f::Function; reps::Int=10)
    GC.gc()
    f() # warmup
    GC.gc()
    times_ms = Vector{Float64}(undef, reps)
    bytes = Vector{Int}(undef, reps)
    for i in 1:reps
        m = @timed f()
        times_ms[i] = 1000.0 * m.time
        bytes[i] = m.bytes
    end
    sort!(times_ms)
    sort!(bytes)
    med_ms = times_ms[cld(reps, 2)]
    med_kib = bytes[cld(reps, 2)] / 1024.0
    println(rpad(name, 44), " median_time=", round(med_ms, digits=3),
            " ms  median_alloc=", round(med_kib, digits=1), " KiB")
    return (ms=med_ms, kib=med_kib)
end

function _rand_phi(field::CM.AbstractCoeffField, nrows::Int, ncols::Int, rng::AbstractRNG)
    K = CM.coeff_type(field)
    phi = Matrix{K}(undef, nrows, ncols)
    @inbounds for i in 1:nrows, j in 1:ncols
        phi[i, j] = CM.coerce(field, rand(rng, -2:2))
    end
    return phi
end

function _flange_fixture_cache_friendly(field::CM.AbstractCoeffField; ncuts::Int=8)
    n = 2
    tau = FZ.face(n, [false, true]) # dependence only on x1 -> many repeated states across x2
    thresholds = collect(-ncuts:ncuts)
    flats = Vector{FZ.IndFlat{n}}(undef, length(thresholds))
    injectives = Vector{FZ.IndInj{n}}(undef, length(thresholds))
    @inbounds for (i, t) in enumerate(thresholds)
        flats[i] = FZ.IndFlat(tau, (t, 0); id=Symbol(:F, i))
        injectives[i] = FZ.IndInj(tau, (t + 1, 0); id=Symbol(:E, i))
    end
    phi = _rand_phi(field, length(injectives), length(flats), Random.MersenneTwister(0xA11CF001))
    FG = FZ.Flange{CM.coeff_type(field)}(n, flats, injectives, phi; field=field)
    return FG, (-2 * ncuts, -2 * ncuts), (2 * ncuts, 2 * ncuts)
end

function _flange_fixture_adversarial(field::CM.AbstractCoeffField; ncuts::Int=8)
    n = 2
    rng = Random.MersenneTwister(0xA11CF002)
    m = 2 * ncuts + 4
    tau = FZ.face(n, [false, false]) # full 2D constraints -> more distinct active sets
    flats = Vector{FZ.IndFlat{n}}(undef, m)
    injectives = Vector{FZ.IndInj{n}}(undef, m)
    @inbounds for i in 1:m
        b1 = rand(rng, -2 * ncuts:2 * ncuts)
        b2 = rand(rng, -2 * ncuts:2 * ncuts)
        flats[i] = FZ.IndFlat(tau, (b1, b2); id=Symbol(:F, i))
        e1 = rand(rng, -2 * ncuts:2 * ncuts)
        e2 = rand(rng, -2 * ncuts:2 * ncuts)
        injectives[i] = FZ.IndInj(tau, (e1, e2); id=Symbol(:E, i))
    end
    phi = _rand_phi(field, m, m, rng)
    FG = FZ.Flange{CM.coeff_type(field)}(n, flats, injectives, phi; field=field)
    return FG, (-2 * ncuts, -2 * ncuts), (2 * ncuts, 2 * ncuts)
end

function _flange_fixture_basis_change(field::CM.AbstractCoeffField)
    n = 1
    flats = [
        FZ.IndFlat(FZ.face(n, [false]), (2,); id=:BF1),
        FZ.IndFlat(FZ.face(n, [true]),  (2,); id=:BF2),
        FZ.IndFlat(FZ.face(n, [false]), (3,); id=:BF3),
    ]
    injectives = [
        FZ.IndInj(FZ.face(n, [false]), (10_000,); id=:BE1),
        FZ.IndInj(FZ.face(n, [false]), (10_000,); id=:BE2),
    ]
    K = CM.coeff_type(field)
    phi = K[1 0 1;
            0 1 1]
    FG = FZ.Flange{K}(n, flats, injectives, phi; field=field)
    return FG, (2,), (3,)
end

function _naive_pmodule_on_box(FG::FZ.Flange{K}; a::NTuple{N,Int}, b::NTuple{N,Int}) where {K,N}
    Q, coords = ZE.grid_poset(a, b)
    r = length(FG.injectives)
    c = length(FG.flats)
    Phi = FG.phi
    field = FG.field

    active_rows = Vector{Vector{Int}}(undef, length(coords))
    B = Vector{Matrix{K}}(undef, length(coords))
    dims = zeros(Int, length(coords))

    @inbounds for i in eachindex(coords)
        g = coords[i]
        rows = Int[]
        cols = Int[]
        for rr in 1:r
            FZ.in_inj(FG.injectives[rr], g) && push!(rows, rr)
        end
        for cc in 1:c
            FZ.in_flat(FG.flats[cc], g) && push!(cols, cc)
        end
        active_rows[i] = rows

        if isempty(rows) || isempty(cols)
            B[i] = zeros(K, length(rows), 0)
            dims[i] = 0
        else
            Phi_g = Phi[rows, cols]
            Bg = FL.colspace(field, Phi_g)
            B[i] = Bg
            dims[i] = size(Bg, 2)
        end
    end

    edge_maps = Dict{Tuple{Int,Int}, Matrix{K}}()
    function _gather_rows(Bu::Matrix{K}, rows_u::Vector{Int}, rows_v::Vector{Int})
        out = Matrix{K}(undef, length(rows_v), size(Bu, 2))
        j = 1
        @inbounds for i in 1:length(rows_v)
            t = rows_v[i]
            while rows_u[j] < t
                j += 1
            end
            for col in 1:size(Bu, 2)
                out[i, col] = Bu[j, col]
            end
        end
        return out
    end

    for (u, v) in FF.cover_edges(Q)
        du = dims[u]
        dv = dims[v]
        if dv == 0 || du == 0
            edge_maps[(u, v)] = zeros(K, dv, du)
        else
            Im = _gather_rows(B[u], active_rows[u], active_rows[v])
            edge_maps[(u, v)] = FL.solve_fullcolumn(field, B[v], Im)
        end
    end

    return TO.Modules.PModule{K}(Q, Vector{Int}(dims), edge_maps; field=field)
end

function _pmodule_equal(M1, M2)
    M1.dims == M2.dims || return false
    for (u, v) in FF.cover_edges(M1.Q)
        A = M1.edge_maps[u, v]
        B = M2.edge_maps[u, v]
        size(A) == size(B) || return false
        all(A .== B) || return false
    end
    return true
end

function _pmodule_fields(all_fields::Bool)
    if all_fields
        return [("qq", CM.QQField()), ("f2", CM.F2()), ("f3", CM.F3()), ("fp5", CM.Fp(5))]
    end
    return [("qq", CM.QQField()), ("f2", CM.F2())]
end

function _neighbor_boxes(a::NTuple{N,Int}, b::NTuple{N,Int}) where {N}
    width1 = b[1] - a[1]
    last_start = max(a[1], b[1] - 4)
    return [(Base.setindex(a, x, 1), Base.setindex(b, x + width1, 1))
            for x in a[1]:last_start]
end

function _run_pmodule_bench(; reps::Int=4, ncuts::Int=8, all_fields::Bool=false)
    println("\npmodule_on_box benchmark (incremental+cache vs naive full scan)")
    println("reps=$(reps), ncuts=$(ncuts), fields=$(all_fields ? "qq,f2,f3,fp5" : "qq,f2")\n")
    cases = [("friendly", _flange_fixture_cache_friendly), ("adversarial", _flange_fixture_adversarial)]
    for (fname, field) in _pmodule_fields(all_fields)
        println("field=$(fname)")
        for (cname, mk) in cases
            FG, a, b = mk(field; ncuts=ncuts)
            Mnew = ZE.pmodule_on_box(FG; a=a, b=b)
            Mold = _naive_pmodule_on_box(FG; a=a, b=b)
            _pmodule_equal(Mnew, Mold) || error("pmodule parity failed for field=$(fname), case=$(cname)")

            bcache = ZE.compile_zn_box_cache(FG)
            bnew = _bench("pmodule new ($cname)", () -> ZE.pmodule_on_box(FG; a=a, b=b); reps=reps)
            old_transition_mode = ZE._ZN_TRANSITION_SPARSE_OVERRIDE[]
            bdenseforced = try
                ZE._ZN_TRANSITION_SPARSE_OVERRIDE[] = :never
                _bench("pmodule dense-forced ($cname)", () -> ZE.pmodule_on_box(FG; a=a, b=b); reps=reps)
            finally
                ZE._ZN_TRANSITION_SPARSE_OVERRIDE[] = old_transition_mode
            end
            bcached = _bench("pmodule cached ($cname)", () -> ZE.pmodule_on_box(FG; a=a, b=b, cache=bcache); reps=reps)
            neighbors = _neighbor_boxes(a, b)
            bneighbors_uncached = _bench("pmodule neighbor uncached ($cname)", () -> begin
                for (aa, bb) in neighbors
                    ZE.pmodule_on_box(FG; a=aa, b=bb)
                end
            end; reps=reps)
            bneighbors = _bench("pmodule neighbor cached ($cname)", () -> begin
                for (aa, bb) in neighbors
                    ZE.pmodule_on_box(FG; a=aa, b=bb, cache=bcache)
                end
            end; reps=reps)
            bold = _bench("pmodule naive ($cname)", () -> _naive_pmodule_on_box(FG; a=a, b=b); reps=reps)
            println("  speedup naive/new: ", round(bold.ms / bnew.ms, digits=3), "x")
            println("  speedup dense-forced/new: ", round(bdenseforced.ms / max(1e-9, bnew.ms), digits=3), "x")
            println("  speedup naive/cached: ", round(bold.ms / bcached.ms, digits=3), "x")
            println("  neighbor uncached/cached speedup: ", round(bneighbors_uncached.ms / max(1e-9, bneighbors.ms), digits=3), "x")
            println("  neighbor uncached/cached alloc ratio: ", round(bneighbors_uncached.kib / max(1e-9, bneighbors.kib), digits=3), "x")
            println("  alloc ratio naive/new: ", round(bold.kib / max(1e-9, bnew.kib), digits=3), "x")
            println("  alloc ratio naive/cached: ", round(bold.kib / max(1e-9, bcached.kib), digits=3), "x")
        end
        FGbc, abc, bbc_box = _flange_fixture_basis_change(field)
        bcache_bc = ZE.compile_zn_box_cache(FGbc)
        bbcur = _bench("pmodule basis-change", () -> ZE.pmodule_on_box(FGbc; a=abc, b=bbc_box); reps=reps)
        old_basis_change_mode = ZE._ZN_BASIS_CHANGE_FASTPATH[]
        bbslow = try
            ZE._ZN_BASIS_CHANGE_FASTPATH[] = false
            _bench("pmodule basis-change off", () -> ZE.pmodule_on_box(FGbc; a=abc, b=bbc_box); reps=reps)
        finally
            ZE._ZN_BASIS_CHANGE_FASTPATH[] = old_basis_change_mode
        end
        bbcached = _bench("pmodule basis-change cached", () -> ZE.pmodule_on_box(FGbc; a=abc, b=bbc_box, cache=bcache_bc); reps=reps)
        println("  basis-change off/current speedup: ", round(bbslow.ms / max(1e-9, bbcur.ms), digits=3), "x")
        println("  basis-change cache speedup: ", round(bbcur.ms / max(1e-9, bbcached.ms), digits=3), "x")
        println()
    end
end

@inline function _dim_at_submatrix_old(FG::FZ.Flange{K}, g::NTuple{N,Int}) where {K,N}
    rows = FZ.active_injectives(FG.injectives, g)
    cols = FZ.active_flats(FG.flats, g)
    if isempty(rows) || isempty(cols)
        return 0
    end
    return FL.rank(FG.field, FG.phi[rows, cols])
end

function _sample_box_points(a::NTuple{N,Int}, b::NTuple{N,Int}, nqueries::Int, rng::AbstractRNG) where {N}
    out = Vector{NTuple{N,Int}}(undef, nqueries)
    @inbounds for i in 1:nqueries
        out[i] = ntuple(k -> rand(rng, a[k]:b[k]), N)
    end
    return out
end

function _run_flange_dimat_bench(; reps::Int=8, nqueries::Int=20_000, ncuts::Int=10)
    println("\nFlange dim_at benchmark (auto path vs always-submatrix baseline)")
    println("reps=$(reps), nqueries=$(nqueries), ncuts=$(ncuts)\n")
    cases = [("friendly", _flange_fixture_cache_friendly), ("adversarial", _flange_fixture_adversarial)]
    for (fname, field) in [("qq", CM.QQField()), ("f2", CM.F2())]
        println("field=$(fname)")
        for (cname, mk) in cases
            FG, a, b = mk(field; ncuts=ncuts)
            rng = Random.MersenneTwister(hash((fname, cname, nqueries)))
            queries = _sample_box_points(a, b, nqueries, rng)

            # parity check before timing
            @inbounds for g in queries
                FZ.dim_at(FG, g) == _dim_at_submatrix_old(FG, g) ||
                    error("dim_at parity failed for field=$(fname), case=$(cname)")
            end

            bnew = _bench("dim_at auto ($cname)", () -> begin
                s = 0
                @inbounds for g in queries
                    s += FZ.dim_at(FG, g)
                end
                s
            end; reps=reps)

            bold = _bench("dim_at submatrix old ($cname)", () -> begin
                s = 0
                @inbounds for g in queries
                    s += _dim_at_submatrix_old(FG, g)
                end
                s
            end; reps=reps)

            println("  speedup old/auto: ", round(bold.ms / bnew.ms, digits=3), "x")
            println("  alloc ratio old/auto: ", round(bold.kib / max(1e-9, bnew.kib), digits=3), "x")
        end
        println()
    end
end

function _run_flange_degree_matrix_bench(; reps::Int=8, nqueries::Int=20_000, ncuts::Int=10)
    println("\nFlange degree_matrix benchmark (allocating vs reuse)")
    println("reps=$(reps), nqueries=$(nqueries), ncuts=$(ncuts)\n")
    cases = [("friendly", _flange_fixture_cache_friendly), ("adversarial", _flange_fixture_adversarial)]
    for (fname, field) in [("qq", CM.QQField()), ("f2", CM.F2())]
        println("field=$(fname)")
        for (cname, mk) in cases
            FG, a, b = mk(field; ncuts=ncuts)
            rng = Random.MersenneTwister(hash((fname, cname, nqueries, :degree_matrix)))
            queries = _sample_box_points(a, b, nqueries, rng)

            @inbounds for g in queries
                A_alloc, rows_alloc, cols_alloc = FZ.degree_matrix(FG, g)
                A_reuse, rows_reuse, cols_reuse = FZ.degree_matrix!(Int[], Int[], FG, g)
                Matrix(A_alloc) == Matrix(A_reuse) ||
                    error("degree_matrix parity failed for field=$(fname), case=$(cname)")
                rows_alloc == rows_reuse || error("rows parity failed for field=$(fname), case=$(cname)")
                cols_alloc == cols_reuse || error("cols parity failed for field=$(fname), case=$(cname)")
            end

            b_alloc = _bench("degree_matrix alloc ($cname)", () -> begin
                s = 0
                @inbounds for g in queries
                    A, rows, cols = FZ.degree_matrix(FG, g)
                    s += size(A, 1) + size(A, 2) + length(rows) + length(cols)
                end
                s
            end; reps=reps)

            rows_buf = Int[]
            cols_buf = Int[]
            b_reuse = _bench("degree_matrix! caller buffers ($cname)", () -> begin
                s = 0
                @inbounds for g in queries
                    A, rows, cols = FZ.degree_matrix!(rows_buf, cols_buf, FG, g)
                    s += size(A, 1) + size(A, 2) + length(rows) + length(cols)
                end
                s
            end; reps=reps)

            b_scratch = _bench("degree_matrix! thread scratch ($cname)", () -> begin
                s = 0
                @inbounds for g in queries
                    A, rows, cols = FZ.degree_matrix!(FG, g)
                    s += size(A, 1) + size(A, 2) + length(rows) + length(cols)
                end
                s
            end; reps=reps)

            println("  speedup alloc/reuse: ", round(b_alloc.ms / b_reuse.ms, digits=3), "x")
            println("  speedup alloc/scratch: ", round(b_alloc.ms / b_scratch.ms, digits=3), "x")
            println("  alloc ratio alloc/reuse: ", round(b_alloc.kib / max(1e-9, b_reuse.kib), digits=3), "x")
            println("  alloc ratio alloc/scratch: ", round(b_alloc.kib / max(1e-9, b_scratch.kib), digits=3), "x")
        end
        println()
    end
end

function _random_face(n::Int, rng::AbstractRNG)
    mask = falses(n)
    @inbounds for i in 1:n
        mask[i] = rand(rng) < 0.35
    end
    return FZ.face(n, Vector{Bool}(mask))
end

function _fixture(; n::Int=3, nflats::Int=14, ninj::Int=14, reps_box::Int=4)
    rng = Random.MersenneTwister(0x5a4e454e434f4445) # "ZNENCODE"
    field = CM.QQField()
    K = CM.coeff_type(field)

    flats = Vector{FZ.IndFlat{n}}(undef, nflats)
    injectives = Vector{FZ.IndInj{n}}(undef, ninj)
    for i in 1:nflats
        b = [rand(rng, -2:2) for _ in 1:n]
        flats[i] = FZ.IndFlat(_random_face(n, rng), b; id=Symbol(:F, i))
    end
    for j in 1:ninj
        b = [rand(rng, -1:4) for _ in 1:n]
        injectives[j] = FZ.IndInj(_random_face(n, rng), b; id=Symbol(:E, j))
    end

    phi = Matrix{K}(undef, ninj, nflats)
    @inbounds for i in 1:ninj, j in 1:nflats
        phi[i, j] = CM.coerce(field, rand(rng, -2:2))
    end
    FG = FZ.Flange{K}(n, flats, injectives, phi)

    enc = TO.encode(FG, TO.EncodingOptions(backend=:zn, max_regions=200_000, poset_kind=:signature, field=field))
    pi = enc.pi.pi

    # Tuple-key baseline map (old style).
    tuple_map = Dict{Tuple{Tuple,Tuple},Int}()
    for t in 1:length(pi.sig_y)
        tuple_map[(Tuple(pi.sig_y[t]), Tuple(pi.sig_z[t]))] = t
    end

    # Query set for locate benchmarks.
    q = Vector{NTuple{n,Int}}(undef, 0)
    # fill later in main according to requested nqueries

    # Product grid for edge-iteration benchmark.
    gsz = ntuple(_ -> reps_box, n)
    Q = FF.ProductOfChainsPoset(gsz)

    return (pi=pi, tuple_map=tuple_map, queries=q, Q=Q, rng=rng, n=n)
end

function _make_queries!(queries::Vector{NTuple{N,Int}}, nqueries::Int, rng::AbstractRNG) where {N}
    resize!(queries, nqueries)
    @inbounds for i in 1:nqueries
        queries[i] = ntuple(_ -> rand(rng, -4:6), N)
    end
    return queries
end

@inline function _locate_tuple(pi::ZE.ZnEncodingMap, tuple_map::Dict{Tuple{Tuple,Tuple},Int}, g)
    y = falses(length(pi.flats))
    z = falses(length(pi.injectives))
    @inbounds for i in eachindex(pi.flats)
        y[i] = FZ.in_flat(pi.flats[i], g)
    end
    @inbounds for j in eachindex(pi.injectives)
        z[j] = !FZ.in_inj(pi.injectives[j], g)
    end
    return get(tuple_map, (Tuple(y), Tuple(z)), 0)
end

function _dense_cover_scan_count(Q)
    C = FF.cover_edges(Q)
    n = FF.nvertices(Q)
    s = 0
    @inbounds for u in 1:n
        for v in 1:n
            if C[u, v]
                s += 1
            end
        end
    end
    return s
end

function _edge_iterator_count(Q)
    s = 0
    for _ in FF.cover_edges(Q)
        s += 1
    end
    return s
end

function _signature_cover_edges_uncached_old_count(P)
    n = FF.nvertices(P)
    n == 0 && return 0
    ranks = Vector{Int}(undef, n)
    maxrank = 0
    @inbounds for i in 1:n
        r = ZE._sig_popcount_words(P.sig_y.words, i, P.y_lastmask) +
            ZE._sig_popcount_words(P.sig_z.words, i, P.z_lastmask)
        ranks[i] = r
        r > maxrank && (maxrank = r)
    end
    buckets = [Int[] for _ in 0:maxrank]
    @inbounds for i in 1:n
        push!(buckets[ranks[i] + 1], i)
    end
    mins = Int[]
    nedges = 0
    @inbounds for i in 1:n
        empty!(mins)
        ri = ranks[i]
        for r in (ri + 1):maxrank
            for j in buckets[r + 1]
                ZE._sig_leq_pair(P, i, j) || continue
                dominated = false
                for k in mins
                    if ZE._sig_leq_pair(P, k, j)
                        dominated = true
                        break
                    end
                end
                dominated && continue
                push!(mins, j)
                nedges += 1
            end
        end
    end
    return nedges
end

function _signature_cover_edges_uncached_count(P)
    s = 0
    for _ in FF.cover_edges(P; cached=false)
        s += 1
    end
    return s
end

function main(; reps::Int=10, nqueries::Int=20_000, grid::Int=18,
              flange_dimat::Bool=true, flange_dimat_reps::Int=8, flange_dimat_ncuts::Int=10,
              flange_degree_matrix::Bool=true, flange_degree_matrix_reps::Int=8,
              flange_degree_matrix_ncuts::Int=10,
              pmodule::Bool=true, pmodule_reps::Int=4, pmodule_ncuts::Int=8,
              pmodule_all_fields::Bool=false)
    fx = _fixture(; reps_box=grid)
    pi = fx.pi
    tuple_map = fx.tuple_map
    queries = _make_queries!(fx.queries, nqueries, fx.rng)
    Q = fx.Q

    packed_lookup = () -> begin
        s = 0
        @inbounds for g in queries
            s += TO.locate(pi, g)
        end
        s
    end

    tuple_lookup = () -> begin
        s = 0
        @inbounds for g in queries
            s += _locate_tuple(pi, tuple_map, g)
        end
        s
    end

    packed_build = () -> begin
        MY = max(1, cld(length(pi.flats), 64))
        MZ = max(1, cld(length(pi.injectives), 64))
        d = Dict{ZE.SigKey{MY,MZ},Int}()
        @inbounds for t in 1:length(pi.sig_y)
            d[ZE._sigkey_from_bitvectors(pi.sig_y[t], pi.sig_z[t], Val(MY), Val(MZ))] = t
        end
        d
    end

    tuple_build = () -> begin
        d = Dict{Tuple{Tuple,Tuple},Int}()
        @inbounds for t in 1:length(pi.sig_y)
            d[(Tuple(pi.sig_y[t]), Tuple(pi.sig_z[t]))] = t
        end
        d
    end

    println("ZnEncoding signature/edge micro-benchmark")
    println("reps=$(reps), nqueries=$(nqueries), nregions=$(length(pi.sig_y)), grid=$(grid)^$(fx.n)\n")

    b1 = _bench("lookup packed-key locate", packed_lookup; reps=reps)
    b2 = _bench("lookup tuple-key baseline", tuple_lookup; reps=reps)
    b3 = _bench("build packed-key map", packed_build; reps=reps)
    b4 = _bench("build tuple-key map", tuple_build; reps=reps)
    b5 = _bench("cover-edge iterator count", () -> _edge_iterator_count(Q); reps=reps)
    b6 = _bench("dense scan C[u,v] baseline", () -> _dense_cover_scan_count(Q); reps=reps)
    Qsig = ZE.SignaturePoset(pi.sig_y, pi.sig_z)
    @assert _signature_cover_edges_uncached_count(Qsig) == _signature_cover_edges_uncached_old_count(Qsig)
    b7 = _bench("signature uncached cover current", () -> _signature_cover_edges_uncached_count(Qsig); reps=reps)
    b8 = _bench("signature uncached cover old", () -> _signature_cover_edges_uncached_old_count(Qsig); reps=reps)

    println("\nRelative speedups (baseline/new):")
    println("lookup tuple / packed: ", round(b2.ms / b1.ms, digits=3), "x")
    println("build tuple / packed:  ", round(b4.ms / b3.ms, digits=3), "x")
    println("dense scan / iterator: ", round(b6.ms / b5.ms, digits=3), "x")
    println("signature cover old/current: ", round(b8.ms / b7.ms, digits=3), "x")

    if flange_dimat
        _run_flange_dimat_bench(; reps=flange_dimat_reps, nqueries=nqueries, ncuts=flange_dimat_ncuts)
    end

    if flange_degree_matrix
        _run_flange_degree_matrix_bench(; reps=flange_degree_matrix_reps, nqueries=nqueries,
                                        ncuts=flange_degree_matrix_ncuts)
    end

    if pmodule
        _run_pmodule_bench(; reps=pmodule_reps, ncuts=pmodule_ncuts, all_fields=pmodule_all_fields)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    reps = _parse_arg(ARGS, "--reps", 10)
    nqueries = _parse_arg(ARGS, "--nqueries", 20_000)
    grid = _parse_arg(ARGS, "--grid", 18)
    flange_dimat = _parse_bool(ARGS, "--flange_dimat", true)
    flange_dimat_reps = _parse_arg(ARGS, "--flange_dimat_reps", 8)
    flange_dimat_ncuts = _parse_arg(ARGS, "--flange_dimat_ncuts", 10)
    flange_degree_matrix = _parse_bool(ARGS, "--flange_degree_matrix", true)
    flange_degree_matrix_reps = _parse_arg(ARGS, "--flange_degree_matrix_reps", 8)
    flange_degree_matrix_ncuts = _parse_arg(ARGS, "--flange_degree_matrix_ncuts", 10)
    pmodule = _parse_bool(ARGS, "--pmodule", true)
    pmodule_reps = _parse_arg(ARGS, "--pmodule_reps", 4)
    pmodule_ncuts = _parse_arg(ARGS, "--pmodule_ncuts", 8)
    pmodule_all_fields = _parse_bool(ARGS, "--pmodule_all_fields", false)
    main(; reps=reps, nqueries=nqueries, grid=grid,
         flange_dimat=flange_dimat, flange_dimat_reps=flange_dimat_reps,
         flange_dimat_ncuts=flange_dimat_ncuts,
         flange_degree_matrix=flange_degree_matrix,
         flange_degree_matrix_reps=flange_degree_matrix_reps,
         flange_degree_matrix_ncuts=flange_degree_matrix_ncuts,
         pmodule=pmodule, pmodule_reps=pmodule_reps,
         pmodule_ncuts=pmodule_ncuts, pmodule_all_fields=pmodule_all_fields)
end
