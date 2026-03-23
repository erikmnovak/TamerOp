# -----------------------------------------------------------------------------
# field_linalg/autotune.jl
#
# Scope:
#   Internal benchmarking and autotuning routines for FieldLinAlg backend and
#   threshold selection.
# Owns:
#   - microbenchmark probes used to fit threshold refs,
#   - startup/full-profile autotune routines,
#   - persistence-facing threshold selection workflow.
# Does not own:
#   - the thresholds themselves (`thresholds.jl`),
#   - runtime backend routing decisions (`backend_routing.jl`),
#   - kernel implementations.
# Depends on:
#   - `thresholds.jl` for mutable threshold refs and persistence schema,
#   - `backend_routing.jl` / kernel files for the code paths being benchmarked.
# -----------------------------------------------------------------------------
#
# Benchmark selection strategy:
# - Routing benchmarks:
#   use `benchmark/sparse_qq_backend_microbench.jl` when backend crossover or
#   threshold policy changes. These scripts answer "which backend should win?"
#   rather than "is the kernel itself fast?".
# - Kernel benchmarks:
#   use `benchmark/field_linalg_sparse_rref_microbench.jl` for sparse exact
#   elimination changes in `sparse_rref.jl`. These isolate row arithmetic and
#   elimination behavior without routing noise.
# - End-to-end benchmarks:
#   use the owner-module benchmark that calls into FieldLinAlg when a change is
#   meant to improve subsystem behavior (e.g. FiniteFringe, FlangeZn,
#   ZnEncoding, ModuleComplexes). These answer "did the user-facing call get
#   faster?".
#
# Autotune parameter grouping:
# - `_autotune_*_shapes` define the probe matrix families.
# - `_autotune_*_threshold_candidates` define the candidate values fitted for a
#   threshold family.
# - `_autotune_*_reps` / query counts define measurement stability.
# - `_autotune_*_steps` feed progress accounting only; they do not affect the
#   fitted thresholds.

function _bench_elapsed(f; reps::Int=2)
    f() # warmup
    tbest = Inf
    for _ in 1:reps
        GC.gc()
        t = @elapsed f()
        if t < tbest
            tbest = t
        end
    end
    return tbest
end

function _bench_elapsed_median(f; reps::Int=3)
    reps <= 1 && return _bench_elapsed(f; reps=1)
    f() # warmup
    ts = Vector{Float64}(undef, reps)
    for i in 1:reps
        GC.gc()
        ts[i] = @elapsed f()
    end
    sort!(ts)
    return ts[cld(reps, 2)]
end

@inline _autotune_nemo_shapes(profile::Symbol=:full) =
    profile == :startup ?
    ((96, 96), (128, 128), (96, 192), (192, 96), (128, 320), (320, 128)) :
    ((96, 96), (128, 128), (160, 160), (192, 192),
     (96, 192), (192, 96), (128, 320), (320, 128), (128, 400), (400, 128))

@inline _autotune_fp_rank_shapes(profile::Symbol=:full) =
    profile == :startup ?
    ((64, 64), (96, 96), (64, 128), (128, 64)) :
    ((64, 64), (96, 96), (128, 128), (64, 128), (128, 64), (96, 192), (192, 96))

@inline _autotune_fp_nullspace_shapes(profile::Symbol=:full) =
    profile == :startup ?
    ((32, 32), (48, 48), (32, 64), (64, 32)) :
    ((32, 32), (48, 48), (64, 64), (32, 64), (64, 32), (48, 96), (96, 48))

@inline _autotune_fp_solve_shapes(profile::Symbol=:full) =
    profile == :startup ?
    ((72, 48), (96, 64), (120, 80), (96, 48), (144, 72)) :
    ((72, 48), (96, 64), (120, 80), (144, 96), (96, 48), (144, 72), (192, 96))

@inline _autotune_float_dense_shapes(profile::Symbol=:full) =
    profile == :startup ?
    ((96, 96), (128, 128), (96, 192), (192, 96)) :
    ((96, 96), (128, 128), (160, 160), (96, 192), (192, 96), (128, 256), (256, 128))

@inline _autotune_modular_nullspace_shapes(profile::Symbol=:full) =
    profile == :startup ?
    ((56, 56), (64, 96), (96, 64), (72, 72)) :
    ((56, 56), (72, 72), (88, 88), (104, 104), (64, 96), (96, 64), (80, 120), (120, 80))

@inline _autotune_modular_solve_shapes(profile::Symbol=:full) =
    profile == :startup ?
    ((60, 40), (84, 56), (108, 72), (128, 88)) :
    ((60, 40), (84, 56), (108, 72), (132, 88), (96, 48), (128, 64), (160, 80))

@inline _autotune_rankqq_dim_shapes(profile::Symbol=:full) =
    profile == :startup ?
    ((64, 64), (96, 96), (64, 128), (128, 64)) :
    ((64, 64), (96, 96), (128, 128), (160, 160), (64, 128), (128, 64), (96, 192), (192, 96))

@inline _autotune_float_sparse_dims() = (512, 1024)
@inline _autotune_float_sparse_densities() = (0.004, 0.01)
@inline _autotune_zn_dimat_probe_ncuts(profile::Symbol=:full) =
    profile == :startup ? (2, 4, 6) : (2, 4, 6, 8)
@inline _autotune_zn_dimat_threshold_candidates(profile::Symbol=:full) =
    profile == :startup ? (24, 32, 48, 64, 96) : (16, 24, 32, 48, 64, 96, 128, 192)
@inline _autotune_zn_dimat_queries_per_probe(profile::Symbol=:full) =
    profile == :startup ? 800 : 1_500
@inline _autotune_zn_dimat_reps(profile::Symbol=:full) = profile == :startup ? 1 : 2
@inline _autotune_rank_words_probe_shapes(profile::Symbol=:full) =
    profile == :startup ? ((16, 16), (24, 24)) : ((16, 16), (24, 24), (32, 32), (48, 32), (32, 48))
@inline _autotune_rank_words_threshold_candidates(profile::Symbol=:full) =
    profile == :startup ? (4, 8, 12, 16, 24, 32, 48) : (4, 8, 12, 16, 24, 32, 48, 64, 96, 128)
@inline _autotune_rank_words_grid_width(profile::Symbol=:full) =
    profile == :startup ? 32 : 64
@inline _autotune_rank_words_reps(profile::Symbol=:full) = profile == :startup ? 1 : 2
@inline function _autotune_zn_dimat_steps(profile::Symbol)
    return 2 * length(_autotune_zn_dimat_probe_ncuts(profile)) *
           length(_autotune_zn_dimat_threshold_candidates(profile))
end
@inline function _autotune_rank_words_steps(profile::Symbol)
    return length(_autotune_rank_words_probe_shapes(profile)) *
           length(_autotune_rank_words_threshold_candidates(profile))
end

@inline _noop_progress_step(::AbstractString) = nothing

mutable struct _AutotuneProgress
    total::Int
    done::Int
    enabled::Bool
    last_pct::Int
end

@inline function _progress_bar(done::Int, total::Int; width::Int=24)
    denom = max(total, 1)
    frac = clamp(done / denom, 0.0, 1.0)
    filled = clamp(round(Int, width * frac), 0, width)
    return "[" * repeat("=", filled) * repeat(".", width - filled) * "]"
end

function _autotune_progress_init(total::Int; enabled::Bool)
    return _AutotuneProgress(max(total, 1), 0, enabled, -1)
end

function _autotune_progress_step!(st::_AutotuneProgress, label::AbstractString)
    st.done = min(st.total, st.done + 1)
    st.enabled || return nothing
    pct = clamp(floor(Int, 100 * st.done / st.total), 0, 99)
    if pct != st.last_pct || !isempty(label)
        st.last_pct = pct
        @info "FieldLinAlg autotune progress" percent=pct bar=_progress_bar(st.done, st.total) step=label
    end
    return nothing
end

function _autotune_progress_finish!(st::_AutotuneProgress)
    st.done = st.total
    st.enabled || return nothing
    @info "FieldLinAlg autotune progress" percent=100 bar=_progress_bar(st.done, st.total) step="complete"
    return nothing
end

function _autotune_modular_steps(profile::Symbol)
    nmax = max(0, min(length(DEFAULT_MODULAR_PRIMES), 8) - 1) # 2:min(...)
    nmin = min(4, max(1, MODULAR_MAX_PRIMES[]))
    n_ns = length(_autotune_modular_nullspace_shapes(profile))
    n_solve = length(_autotune_modular_solve_shapes(profile))
    n_rank = length(_autotune_rankqq_dim_shapes(profile))
    # rank-prime sweep + min-prime sweep + nullspace crossover + solve crossover + rank crossover
    return 3 * nmax + 2 * nmin + n_ns + n_solve + n_rank
end

function _autotune_total_steps(profile::Symbol)
    steps = 0
    if _have_nemo()
        steps += length(_autotune_fp_rank_shapes(profile)) +
                 length(_autotune_fp_nullspace_shapes(profile)) +
                 length(_autotune_fp_solve_shapes(profile))
    end
    steps += length(_autotune_float_dense_shapes(profile))
    if _have_svds_backend()
        steps += length(_autotune_float_sparse_dims()) * length(_autotune_float_sparse_densities())
    end
    if profile == :full
        steps += _autotune_modular_steps(profile)
        steps += _autotune_zn_dimat_steps(profile)
    end
    steps += _autotune_qq_routing_steps(profile)
    steps += _autotune_rank_words_steps(profile)
    return max(steps, 1)
end

function _fp_dense_rand(field::PrimeField, m::Int, n::Int; rng=Random.default_rng())
    p = field.p
    p > 3 || error("_fp_dense_rand only for p > 3")
    A = Matrix{FpElem{p}}(undef, m, n)
    @inbounds for i in 1:m, j in 1:n
        A[i, j] = FpElem{p}(rand(rng, 0:(p - 1)))
    end
    return A
end

function _qq_dense_rand(m::Int, n::Int; rng=Random.default_rng())
    A = Matrix{QQ}(undef, m, n)
    @inbounds for i in 1:m, j in 1:n
        A[i, j] = QQ(rand(rng, -3:3))
    end
    return A
end

function _qq_dense_singular_rand(n::Int; rng=Random.default_rng())
    A = _qq_dense_rand(n, n; rng=rng)
    n >= 2 || return A
    @inbounds A[:, n] = A[:, 1]
    return A
end

function _qq_dense_fullcolumn_rand(m::Int, n::Int; rng=Random.default_rng())
    B = _qq_dense_rand(m, n; rng=rng)
    d = min(m, n)
    @inbounds for i in 1:d
        B[i, i] = QQ(1)
    end
    return B
end

@inline function _qq_nonzero_rand(rng::Random.AbstractRNG)
    v = rand(rng, -3:3)
    while v == 0
        v = rand(rng, -3:3)
    end
    return QQ(v)
end

function _qq_sparse_rand(m::Int, n::Int, density::Float64; rng=Random.default_rng())
    dens = clamp(density, 1e-6, 1.0)
    nnz_target = max(1, round(Int, m * n * dens))
    I = Vector{Int}(undef, nnz_target)
    J = Vector{Int}(undef, nnz_target)
    V = Vector{QQ}(undef, nnz_target)
    @inbounds for t in 1:nnz_target
        I[t] = rand(rng, 1:m)
        J[t] = rand(rng, 1:n)
        V[t] = _qq_nonzero_rand(rng)
    end
    return sparse(I, J, V, m, n)
end

function _qq_sparse_fullcolumn_rand(m::Int, n::Int, density::Float64; rng=Random.default_rng())
    m >= n || error("_qq_sparse_fullcolumn_rand: requires m >= n")
    dens = clamp(density, 1e-6, 1.0)
    I = Int[]
    J = Int[]
    V = QQ[]
    sizehint!(I, n + round(Int, (m - n) * n * dens))
    sizehint!(J, n + round(Int, (m - n) * n * dens))
    sizehint!(V, n + round(Int, (m - n) * n * dens))
    @inbounds for i in 1:n
        push!(I, i)
        push!(J, i)
        push!(V, QQ(1))
    end
    @inbounds for i in (n + 1):m, j in 1:n
        rand(rng) < dens || continue
        push!(I, i)
        push!(J, j)
        push!(V, _qq_nonzero_rand(rng))
    end
    return sparse(I, J, V, m, n)
end

function _float_sparse_singular_rand(n::Int, density::Float64; rng=Random.default_rng())
    A = sprand(rng, Float64, n, n, density)
    if n >= 2
        A[:, n] = A[:, 1]
    end
    if nnz(A) == 0
        A[1, 1] = 1.0
        if n >= 2
            A[1, n] = 1.0
        end
    end
    return A
end

function _pick_crossover_threshold(works::Vector{Int}, tj::Vector{Float64}, tf::Vector{Float64}, default::Int;
                                   speedup_ratio::Float64=0.9)
    ord = sortperm(works)
    @inbounds for oi in ord
        if tf[oi] < speedup_ratio * tj[oi]
            return works[oi]
        end
    end
    return default
end

@inline function _find_crossover_pos(works::Vector{Int}, tj::Vector{Float64}, tf::Vector{Float64};
                                     speedup_ratio::Float64=0.9)
    ord = sortperm(works)
    @inbounds for (pos, oi) in enumerate(ord)
        if tf[oi] < speedup_ratio * tj[oi]
            return pos, ord
        end
    end
    return nothing, ord
end

function _refine_crossover_boundary!(works::Vector{Int},
                                     tj::Vector{Float64},
                                     tf::Vector{Float64},
                                     bench_pair::Function;
                                     speedup_ratio::Float64=0.9,
                                     extra_reps::Int=4,
                                     window::Int=1)
    pos, ord = _find_crossover_pos(works, tj, tf; speedup_ratio=speedup_ratio)
    pos === nothing && return nothing
    lo = max(1, pos - window)
    hi = min(length(ord), pos + window)
    @inbounds for q in lo:hi
        i = ord[q]
        tj_i, tf_i = bench_pair(i, extra_reps)
        tj[i] = tj_i
        tf[i] = tf_i
    end
    return nothing
end

@inline _threshold_regret(pred::Float64, best::Float64, margin::Float64) =
    max(0.0, pred / max(best, eps(Float64)) - (1.0 + margin))

function _smooth_by_work(works::Vector{Int}, vals::Vector{Float64}; radius::Int=1)
    n = length(works)
    ord = sortperm(works)
    out = similar(vals)
    @inbounds for p in 1:n
        lo = max(1, p - radius)
        hi = min(n, p + radius)
        localv = Vector{Float64}(undef, hi - lo + 1)
        t = 1
        for q in lo:hi
            localv[t] = vals[ord[q]]
            t += 1
        end
        out[ord[p]] = median(localv)
    end
    return out
end

function _nearest_indices(works::Vector{Int}, target::Int, k::Int)
    n = length(works)
    n == 0 && return Int[]
    ord = sortperm(abs.(works .- target))
    return ord[1:min(k, n)]
end

function _fit_single_threshold(works::Vector{Int}, t_exact::Vector{Float64}, t_other::Vector{Float64};
                               margin::Float64=0.15)
    cands = sort(unique(vcat(works, maximum(works) + 1)))
    best_thr = cands[end]
    best_loss = Inf
    best_mismatch = typemax(Int)
    @inline label(i) = (t_other[i] <= (1.0 - margin) * t_exact[i]) ? :other : :exact
    @inbounds for thr in cands
        loss = 0.0
        mismatch = 0
        for i in eachindex(works)
            pred_label = works[i] >= thr ? :other : :exact
            mismatch += pred_label == label(i) ? 0 : 1
            pred = pred_label == :other ? t_other[i] : t_exact[i]
            best = min(t_exact[i], t_other[i])
            loss += _threshold_regret(pred, best, margin)
        end
        if mismatch < best_mismatch || (mismatch == best_mismatch && loss < best_loss - 1e-12)
            best_loss = loss
            best_mismatch = mismatch
            best_thr = thr
        end
    end
    return best_thr, best_loss, best_mismatch
end

function _fit_double_threshold(works::Vector{Int},
                               t_exact::Vector{Float64},
                               t_mod::Vector{Float64},
                               t_nemo::Vector{Float64};
                               margin::Float64=0.15)
    cands = sort(unique(vcat(works, maximum(works) + 1)))
    best_mod = cands[end]
    best_nemo = cands[end]
    best_loss = Inf
    best_mismatch = typemax(Int)
    @inline function label(i)
        if t_nemo[i] <= (1.0 - margin) * min(t_exact[i], t_mod[i])
            return :nemo
        elseif t_mod[i] <= (1.0 - margin) * t_exact[i]
            return :modular
        else
            return :exact
        end
    end
    @inbounds for thr_mod in cands
        for thr_nemo in cands
            thr_nemo < thr_mod && continue
            loss = 0.0
            mismatch = 0
            for i in eachindex(works)
                pred_label = works[i] >= thr_nemo ? :nemo : (works[i] >= thr_mod ? :modular : :exact)
                mismatch += pred_label == label(i) ? 0 : 1
                pred = pred_label == :nemo ? t_nemo[i] : (pred_label == :modular ? t_mod[i] : t_exact[i])
                best = min(t_exact[i], min(t_mod[i], t_nemo[i]))
                loss += _threshold_regret(pred, best, margin)
            end
            if mismatch < best_mismatch || (mismatch == best_mismatch && loss < best_loss - 1e-12)
                best_loss = loss
                best_mismatch = mismatch
                best_mod = thr_mod
                best_nemo = thr_nemo
            end
        end
    end
    return best_mod, best_nemo, best_loss, best_mismatch
end

function _eval_single_threshold(works::Vector{Int}, t_exact::Vector{Float64}, t_other::Vector{Float64},
                                thr::Int; margin::Float64=0.15)
    loss = 0.0
    mismatch = 0
    @inline label(i) = (t_other[i] <= (1.0 - margin) * t_exact[i]) ? :other : :exact
    @inbounds for i in eachindex(works)
        pred_label = works[i] >= thr ? :other : :exact
        mismatch += pred_label == label(i) ? 0 : 1
        pred = pred_label == :other ? t_other[i] : t_exact[i]
        best = min(t_exact[i], t_other[i])
        loss += _threshold_regret(pred, best, margin)
    end
    return loss, mismatch
end

function _fit_single_threshold_le(works::Vector{Int}, t_exact::Vector{Float64}, t_other::Vector{Float64};
                                  margin::Float64=0.15)
    cands = sort(unique(vcat(works, minimum(works) - 1)))
    best_thr = cands[1]
    best_loss = Inf
    best_mismatch = typemax(Int)
    @inline label(i) = (t_other[i] <= (1.0 - margin) * t_exact[i]) ? :other : :exact
    @inbounds for thr in cands
        loss = 0.0
        mismatch = 0
        for i in eachindex(works)
            pred_label = works[i] <= thr ? :other : :exact
            mismatch += pred_label == label(i) ? 0 : 1
            pred = pred_label == :other ? t_other[i] : t_exact[i]
            best = min(t_exact[i], t_other[i])
            loss += _threshold_regret(pred, best, margin)
        end
        if mismatch < best_mismatch || (mismatch == best_mismatch && loss < best_loss - 1e-12)
            best_loss = loss
            best_mismatch = mismatch
            best_thr = thr
        end
    end
    return best_thr, best_loss, best_mismatch
end

function _eval_single_threshold_le(works::Vector{Int}, t_exact::Vector{Float64}, t_other::Vector{Float64},
                                   thr::Int; margin::Float64=0.15)
    loss = 0.0
    mismatch = 0
    @inline label(i) = (t_other[i] <= (1.0 - margin) * t_exact[i]) ? :other : :exact
    @inbounds for i in eachindex(works)
        pred_label = works[i] <= thr ? :other : :exact
        mismatch += pred_label == label(i) ? 0 : 1
        pred = pred_label == :other ? t_other[i] : t_exact[i]
        best = min(t_exact[i], t_other[i])
        loss += _threshold_regret(pred, best, margin)
    end
    return loss, mismatch
end

function _eval_double_threshold(works::Vector{Int}, t_exact::Vector{Float64}, t_mod::Vector{Float64}, t_nemo::Vector{Float64},
                                thr_mod::Int, thr_nemo::Int; margin::Float64=0.15)
    loss = 0.0
    mismatch = 0
    @inline function label(i)
        if t_nemo[i] <= (1.0 - margin) * min(t_exact[i], t_mod[i])
            return :nemo
        elseif t_mod[i] <= (1.0 - margin) * t_exact[i]
            return :modular
        else
            return :exact
        end
    end
    @inbounds for i in eachindex(works)
        pred_label = works[i] >= thr_nemo ? :nemo : (works[i] >= thr_mod ? :modular : :exact)
        mismatch += pred_label == label(i) ? 0 : 1
        pred = pred_label == :nemo ? t_nemo[i] : (pred_label == :modular ? t_mod[i] : t_exact[i])
        best = min(t_exact[i], min(t_mod[i], t_nemo[i]))
        loss += _threshold_regret(pred, best, margin)
    end
    return loss, mismatch
end

@inline function _accept_threshold_update(new_loss::Float64, new_mismatch::Int,
                                          old_loss::Float64, old_mismatch::Int)
    if !isfinite(old_loss)
        return isfinite(new_loss)
    end
    if new_loss <= old_loss * 0.99
        return true
    end
    if new_loss <= old_loss * 1.01 && new_mismatch < old_mismatch
        return true
    end
    return false
end

function _qq_probe_shapes(op::Symbol, shape::Symbol, profile::Symbol, holdout::Bool)
    if op == :rank
        if shape == :square
            return holdout ?
                (profile == :startup ? ((28, 28), (40, 40), (72, 72), (104, 104)) :
                                       ((24, 24), (36, 36), (56, 56), (72, 72), (88, 88), (104, 104), (120, 120))) :
                (profile == :startup ? ((24, 24), (40, 40), (64, 64), (96, 96), (128, 128)) :
                                       ((20, 20), (28, 28), (48, 48), (64, 64), (80, 80), (96, 96), (112, 112), (128, 128)))
        elseif shape == :tall
            return holdout ?
                (profile == :startup ? ((40, 20), (72, 36), (112, 56), (152, 76)) :
                                       ((36, 18), (52, 26), (104, 52), (120, 60), (136, 68), (152, 76), (176, 88))) :
                (profile == :startup ? ((32, 16), (64, 32), (96, 48), (128, 64), (160, 80)) :
                                       ((28, 14), (44, 22), (88, 44), (112, 56), (128, 64), (144, 72), (160, 80), (192, 96)))
        else
            return holdout ?
                (profile == :startup ? ((20, 40), (36, 72), (56, 112), (76, 152)) :
                                       ((18, 36), (26, 52), (52, 104), (60, 120), (68, 136), (76, 152), (88, 176))) :
                (profile == :startup ? ((16, 32), (32, 64), (48, 96), (64, 128), (80, 160)) :
                                       ((14, 28), (22, 44), (44, 88), (56, 112), (64, 128), (72, 144), (80, 160), (96, 192)))
        end
    elseif op == :nullspace
        if shape == :square
            return holdout ?
                (profile == :startup ? ((24, 24), (40, 40), (56, 56), (88, 88)) :
                                       ((20, 20), (32, 32), (48, 48), (64, 64), (80, 80), (96, 96), (112, 112))) :
                (profile == :startup ? ((20, 20), (32, 32), (48, 48), (72, 72), (96, 96)) :
                                       ((16, 16), (24, 24), (40, 40), (56, 56), (72, 72), (88, 88), (104, 104)))
        elseif shape == :tall
            return holdout ?
                (profile == :startup ? ((48, 24), (72, 36), (112, 56), (144, 72)) :
                                       ((40, 20), (56, 28), (96, 48), (112, 56), (128, 64), (144, 72), (160, 80))) :
                (profile == :startup ? ((40, 20), (64, 32), (96, 48), (128, 64), (160, 80)) :
                                       ((32, 16), (48, 24), (80, 40), (96, 48), (112, 56), (128, 64), (144, 72)))
        else
            return holdout ?
                (profile == :startup ? ((24, 48), (36, 72), (56, 112), (72, 144)) :
                                       ((20, 40), (28, 56), (48, 96), (56, 112), (64, 128), (72, 144), (80, 160))) :
                (profile == :startup ? ((20, 40), (32, 64), (48, 96), (64, 128), (80, 160)) :
                                       ((16, 32), (24, 48), (40, 80), (48, 96), (56, 112), (64, 128), (72, 144)))
        end
    elseif op == :solve
        if shape == :square
            return holdout ?
                (profile == :startup ? ((32, 32), (48, 48), (64, 64), (88, 88)) :
                                       ((28, 28), (40, 40), (56, 56), (72, 72), (88, 88), (104, 104), (120, 120))) :
                (profile == :startup ? ((24, 24), (40, 40), (56, 56), (80, 80), (104, 104)) :
                                       ((20, 20), (32, 32), (48, 48), (64, 64), (80, 80), (96, 96), (112, 112)))
        elseif shape == :tall
            return holdout ?
                (profile == :startup ? ((42, 28), (72, 48), (96, 64), (132, 88)) :
                                       ((36, 24), (54, 36), (84, 56), (108, 72), (132, 88), (156, 104), (180, 120))) :
                (profile == :startup ? ((36, 24), (60, 40), (84, 56), (108, 72), (132, 88)) :
                                       ((30, 20), (48, 32), (72, 48), (96, 64), (120, 80), (144, 96), (168, 112)))
        else
            # Full-column solve naturally uses m>=n; use square probes for wide bucket fallback.
            return _qq_probe_shapes(op, :square, profile, holdout)
        end
    end
    return Tuple{Int,Int}[]
end

@inline _qq_sparse_probe_densities(profile::Symbol) =
    profile == :startup ? (0.2,) : (0.2, 0.05)

function _qq_case(op::Symbol, m::Int, n::Int, rng::Random.AbstractRNG;
                  sparse::Bool=false, density::Float64=0.2)
    if op == :rank
        return sparse ? _qq_sparse_rand(m, n, density; rng=rng) : _qq_dense_rand(m, n; rng=rng)
    elseif op == :nullspace
        A = sparse ? _qq_sparse_rand(m, n, density; rng=rng) : _qq_dense_rand(m, n; rng=rng)
        min(m, n) >= 2 && (A[:, end] = A[:, 1])
        return A
    elseif op == :solve
        m >= n || error("solve probe requires m >= n")
        B = sparse ? _qq_sparse_fullcolumn_rand(m, n, density; rng=rng) : _qq_dense_fullcolumn_rand(m, n; rng=rng)
        X = _qq_dense_rand(n, 2; rng=rng)
        return (B, B * X)
    end
    error("unsupported QQ probe op: $(op)")
end

function _bench_qq_probe(op::Symbol, probe, backend::Symbol; reps::Int=1)
    F = QQField()
    try
        if op == :rank
            A = probe::AbstractMatrix{QQ}
            exact_backend = _is_sparse_like(A) ? :julia_sparse : :julia_exact
            if backend == :julia_exact
                return _bench_elapsed(() -> rank(F, A; backend=exact_backend); reps=reps)
            elseif backend == :nemo
                _have_nemo() || return Inf
                return _bench_elapsed(() -> rank(F, A; backend=:nemo); reps=reps)
            elseif backend == :modular
                return Inf
            end
        elseif op == :nullspace
            A = probe::AbstractMatrix{QQ}
            exact_backend = _is_sparse_like(A) ? :julia_sparse : :julia_exact
            if backend == :julia_exact
                return _bench_elapsed(() -> nullspace(F, A; backend=exact_backend); reps=reps)
            elseif backend == :nemo
                _have_nemo() || return Inf
                return _bench_elapsed(() -> nullspace(F, A; backend=:nemo); reps=reps)
            elseif backend == :modular
                _is_sparse_like(A) && return Inf
                return _bench_elapsed(() -> nullspace(F, A; backend=:modular); reps=reps)
            end
        elseif op == :solve
            B, Y = probe::Tuple{AbstractMatrix{QQ},Matrix{QQ}}
            exact_backend = _is_sparse_like(B) ? :julia_sparse : :julia_exact
            if backend == :julia_exact
                return _bench_elapsed(() -> solve_fullcolumn(F, B, Y; backend=exact_backend, check_rhs=false, cache=false); reps=reps)
            elseif backend == :nemo
                _have_nemo() || return Inf
                return _bench_elapsed(() -> solve_fullcolumn(F, B, Y; backend=:nemo, check_rhs=false, cache=false); reps=reps)
            elseif backend == :modular
                _is_sparse_like(B) && return Inf
                return _bench_elapsed(() -> solve_fullcolumn(F, B, Y; backend=:modular, check_rhs=false, cache=false); reps=reps)
            end
        end
    catch
        return Inf
    end
    return Inf
end

function _set_qq_threshold!(op::Symbol, shape::Symbol, backend::Symbol, value::Int)
    v = max(1, Int(value))
    if backend == :nemo && op == :rank
        if shape == :square
            QQ_NEMO_RANK_THRESHOLD_SQUARE[] = v
        elseif shape == :tall
            QQ_NEMO_RANK_THRESHOLD_TALL[] = v
        else
            QQ_NEMO_RANK_THRESHOLD_WIDE[] = v
        end
    elseif backend == :nemo && op == :nullspace
        if shape == :square
            QQ_NEMO_NULLSPACE_THRESHOLD_SQUARE[] = v
        elseif shape == :tall
            QQ_NEMO_NULLSPACE_THRESHOLD_TALL[] = v
        else
            QQ_NEMO_NULLSPACE_THRESHOLD_WIDE[] = v
        end
    elseif backend == :nemo && op == :solve
        if shape == :square
            QQ_NEMO_SOLVE_THRESHOLD_SQUARE[] = v
        elseif shape == :tall
            QQ_NEMO_SOLVE_THRESHOLD_TALL[] = v
        else
            QQ_NEMO_SOLVE_THRESHOLD_WIDE[] = v
        end
    elseif backend == :modular && op == :nullspace
        if shape == :square
            QQ_MODULAR_NULLSPACE_THRESHOLD_SQUARE[] = v
        elseif shape == :tall
            QQ_MODULAR_NULLSPACE_THRESHOLD_TALL[] = v
        else
            QQ_MODULAR_NULLSPACE_THRESHOLD_WIDE[] = v
        end
    elseif backend == :modular && op == :solve
        if shape == :square
            QQ_MODULAR_SOLVE_THRESHOLD_SQUARE[] = v
        elseif shape == :tall
            QQ_MODULAR_SOLVE_THRESHOLD_TALL[] = v
        else
            QQ_MODULAR_SOLVE_THRESHOLD_WIDE[] = v
        end
    end
    return nothing
end

@inline function _get_qq_threshold(op::Symbol, shape::Symbol, backend::Symbol)
    return backend == :nemo ? _qq_nemo_threshold(op, shape) : _qq_modular_threshold(op, shape)
end

function _qq_probe_cases(op::Symbol, shape::Symbol, profile::Symbol, holdout::Bool)
    shapes = collect(_qq_probe_shapes(op, shape, profile, holdout))
    if holdout && profile == :startup && length(shapes) > 2
        shapes = [shapes[1], shapes[end]]
    end
    cases = NamedTuple{(:m, :n, :sparse, :density),Tuple{Int, Int, Bool, Float64}}[]
    dens_train = _qq_sparse_probe_densities(profile)
    dens_hold = (dens_train[1],)
    sparse_dens = holdout ? dens_hold : dens_train
    sparse_work_min = profile == :startup ? 900 : 600
    for (m, n) in shapes
        push!(cases, (m=m, n=n, sparse=false, density=1.0))
        if op == :rank && m * n >= sparse_work_min
            for dens in sparse_dens
                push!(cases, (m=m, n=n, sparse=true, density=dens))
            end
        end
    end
    return cases
end

function _autotune_qq_shape_thresholds!(op::Symbol, shape::Symbol, profile::Symbol, rng::Random.AbstractRNG,
                                        progress_step::Function=_noop_progress_step)
    train_cases = _qq_probe_cases(op, shape, profile, false)
    hold_cases = _qq_probe_cases(op, shape, profile, true)
    isempty(train_cases) && return

    ntrain = length(train_cases)
    train_work = Vector{Int}(undef, ntrain)
    train_probe = Vector{Any}(undef, ntrain)
    t_exact = Vector{Float64}(undef, ntrain)
    t_mod = fill(Inf, ntrain)
    t_nemo = fill(Inf, ntrain)
    for (i, case) in enumerate(train_cases)
        m = case.m
        n = case.n
        probe = if op == :solve
            seed = rand(rng, UInt64)
            train_probe[i] = seed
            _qq_case(op, m, n, Random.MersenneTwister(seed); sparse=case.sparse, density=case.density)
        else
            p = _qq_case(op, m, n, rng; sparse=case.sparse, density=case.density)
            train_probe[i] = p
            p
        end
        train_work[i] = m * n
        t_exact[i] = _bench_qq_probe(op, probe, :julia_exact; reps=1)
        if op != :rank
            t_mod[i] = _bench_qq_probe(op, probe, :modular; reps=1)
        end
        t_nemo[i] = _bench_qq_probe(op, probe, :nemo; reps=1)
        op == :solve && _clear_solve_autotune_state!()
        progress_step("qq-$(op)-$(shape) train $(i)/$(ntrain)")
    end

    s_exact = _smooth_by_work(train_work, t_exact)
    s_mod = _smooth_by_work(train_work, t_mod)
    s_nemo = _smooth_by_work(train_work, t_nemo)
    margin = 0.15
    nnear = profile == :full ? 4 : 2
    reps_ref = profile == :full ? 4 : 2
    if op == :rank
        thr_nemo, _, _ = _fit_single_threshold(train_work, s_exact, s_nemo; margin=margin)
        for idx in _nearest_indices(train_work, thr_nemo, nnear)
            p = train_probe[idx]
            t_exact[idx] = _bench_qq_probe(op, p, :julia_exact; reps=reps_ref)
            t_nemo[idx] = _bench_qq_probe(op, p, :nemo; reps=reps_ref)
        end
        s_exact = _smooth_by_work(train_work, t_exact)
        s_nemo = _smooth_by_work(train_work, t_nemo)
        thr_nemo, _, _ = _fit_single_threshold(train_work, s_exact, s_nemo; margin=margin)

        # Holdout guard: keep incumbent if new threshold is worse.
        nh = length(hold_cases)
        h_work = Vector{Int}(undef, nh)
        h_e = Vector{Float64}(undef, nh)
        h_n = Vector{Float64}(undef, nh)
        for (i, case) in enumerate(hold_cases)
            m = case.m
            n = case.n
            p = _qq_case(op, m, n, rng; sparse=case.sparse, density=case.density)
            h_work[i] = m * n
            h_e[i] = _bench_qq_probe(op, p, :julia_exact; reps=1)
            h_n[i] = _bench_qq_probe(op, p, :nemo; reps=1)
            progress_step("qq-$(op)-$(shape) holdout $(i)/$(nh)")
        end
        old_thr = _get_qq_threshold(op, shape, :nemo)
        old_loss, old_mismatch = _eval_single_threshold(h_work, h_e, h_n, old_thr; margin=margin)
        new_loss, new_mismatch = _eval_single_threshold(h_work, h_e, h_n, thr_nemo; margin=margin)
        accept = _accept_threshold_update(new_loss, new_mismatch, old_loss, old_mismatch) ||
                 (new_mismatch == old_mismatch && new_loss <= old_loss * 1.01 && thr_nemo < old_thr)
        _set_qq_threshold!(op, shape, :nemo, accept ? thr_nemo : old_thr)
    else
        thr_mod, thr_nemo, _, _ = _fit_double_threshold(train_work, s_exact, s_mod, s_nemo; margin=margin)
        for idx in union(_nearest_indices(train_work, thr_mod, nnear), _nearest_indices(train_work, thr_nemo, nnear))
            p = if op == :solve
                case = train_cases[idx]
                _qq_case(op, case.m, case.n, Random.MersenneTwister(train_probe[idx]);
                         sparse=case.sparse, density=case.density)
            else
                train_probe[idx]
            end
            t_exact[idx] = _bench_qq_probe(op, p, :julia_exact; reps=reps_ref)
            t_mod[idx] = _bench_qq_probe(op, p, :modular; reps=reps_ref)
            t_nemo[idx] = _bench_qq_probe(op, p, :nemo; reps=reps_ref)
            op == :solve && _clear_solve_autotune_state!()
        end
        s_exact = _smooth_by_work(train_work, t_exact)
        s_mod = _smooth_by_work(train_work, t_mod)
        s_nemo = _smooth_by_work(train_work, t_nemo)
        thr_mod, thr_nemo, _, _ = _fit_double_threshold(train_work, s_exact, s_mod, s_nemo; margin=margin)

        nh = length(hold_cases)
        h_work = Vector{Int}(undef, nh)
        h_e = Vector{Float64}(undef, nh)
        h_m = Vector{Float64}(undef, nh)
        h_n = Vector{Float64}(undef, nh)
        for (i, case) in enumerate(hold_cases)
            m = case.m
            n = case.n
            p = if op == :solve
                _qq_case(op, m, n, Random.MersenneTwister(rand(rng, UInt64));
                         sparse=case.sparse, density=case.density)
            else
                _qq_case(op, m, n, rng; sparse=case.sparse, density=case.density)
            end
            h_work[i] = m * n
            h_e[i] = _bench_qq_probe(op, p, :julia_exact; reps=1)
            h_m[i] = _bench_qq_probe(op, p, :modular; reps=1)
            h_n[i] = _bench_qq_probe(op, p, :nemo; reps=1)
            op == :solve && _clear_solve_autotune_state!()
            progress_step("qq-$(op)-$(shape) holdout $(i)/$(nh)")
        end
        old_mod = _get_qq_threshold(op, shape, :modular)
        old_nemo = _get_qq_threshold(op, shape, :nemo)
        old_loss, old_mismatch = _eval_double_threshold(h_work, h_e, h_m, h_n, old_mod, old_nemo; margin=margin)
        new_loss, new_mismatch = _eval_double_threshold(h_work, h_e, h_m, h_n, thr_mod, thr_nemo; margin=margin)
        accept = _accept_threshold_update(new_loss, new_mismatch, old_loss, old_mismatch) ||
                 (new_mismatch == old_mismatch && new_loss <= old_loss * 1.01 &&
                  thr_mod <= old_mod && thr_nemo <= old_nemo &&
                  (thr_mod < old_mod || thr_nemo < old_nemo))
        if accept
            _set_qq_threshold!(op, shape, :modular, thr_mod)
            _set_qq_threshold!(op, shape, :nemo, thr_nemo)
        else
            _set_qq_threshold!(op, shape, :modular, old_mod)
            _set_qq_threshold!(op, shape, :nemo, old_nemo)
        end
    end
    return nothing
end

@inline function _set_qq_sparse_solve_threshold!(shape::Symbol, bucket::Symbol, value::Int)
    v = max(1, Int(value))
    if shape == :square
        if bucket == :low
            QQ_NEMO_SPARSE_SOLVE_THRESHOLD_SQUARE_LOW[] = v
        elseif bucket == :mid
            QQ_NEMO_SPARSE_SOLVE_THRESHOLD_SQUARE_MID[] = v
        else
            QQ_NEMO_SPARSE_SOLVE_THRESHOLD_SQUARE_HIGH[] = v
        end
    elseif shape == :tall
        if bucket == :low
            QQ_NEMO_SPARSE_SOLVE_THRESHOLD_TALL_LOW[] = v
        elseif bucket == :mid
            QQ_NEMO_SPARSE_SOLVE_THRESHOLD_TALL_MID[] = v
        else
            QQ_NEMO_SPARSE_SOLVE_THRESHOLD_TALL_HIGH[] = v
        end
    else
        if bucket == :low
            QQ_NEMO_SPARSE_SOLVE_THRESHOLD_WIDE_LOW[] = v
        elseif bucket == :mid
            QQ_NEMO_SPARSE_SOLVE_THRESHOLD_WIDE_MID[] = v
        else
            QQ_NEMO_SPARSE_SOLVE_THRESHOLD_WIDE_HIGH[] = v
        end
    end
    return nothing
end

@inline _get_qq_sparse_solve_threshold(shape::Symbol, bucket::Symbol) = _qq_sparse_solve_threshold(shape, bucket)

@inline function _set_qq_sparse_solve_policy!(shape::Symbol, bucket::Symbol, use_ge::Bool)
    v = use_ge ? 1 : -1
    if shape == :square
        if bucket == :low
            QQ_NEMO_SPARSE_SOLVE_POLICY_SQUARE_LOW[] = v
        elseif bucket == :mid
            QQ_NEMO_SPARSE_SOLVE_POLICY_SQUARE_MID[] = v
        else
            QQ_NEMO_SPARSE_SOLVE_POLICY_SQUARE_HIGH[] = v
        end
    elseif shape == :tall
        if bucket == :low
            QQ_NEMO_SPARSE_SOLVE_POLICY_TALL_LOW[] = v
        elseif bucket == :mid
            QQ_NEMO_SPARSE_SOLVE_POLICY_TALL_MID[] = v
        else
            QQ_NEMO_SPARSE_SOLVE_POLICY_TALL_HIGH[] = v
        end
    else
        if bucket == :low
            QQ_NEMO_SPARSE_SOLVE_POLICY_WIDE_LOW[] = v
        elseif bucket == :mid
            QQ_NEMO_SPARSE_SOLVE_POLICY_WIDE_MID[] = v
        else
            QQ_NEMO_SPARSE_SOLVE_POLICY_WIDE_HIGH[] = v
        end
    end
    return nothing
end

@inline _get_qq_sparse_solve_policy(shape::Symbol, bucket::Symbol) = _qq_sparse_solve_use_ge(shape, bucket)

@inline function _qq_sparse_solve_shapes(shape::Symbol, profile::Symbol, holdout::Bool)
    if shape == :square
        return holdout ?
            (profile == :startup ? ((48, 48), (72, 72), (88, 88)) :
                                   ((40, 40), (56, 56), (72, 72), (88, 88), (104, 104))) :
            (profile == :startup ? ((40, 40), (56, 56), (64, 64), (80, 80)) :
                                   ((32, 32), (48, 48), (64, 64), (80, 80), (96, 96)))
    elseif shape == :tall
        return holdout ?
            (profile == :startup ? ((72, 48), (90, 60), (96, 72), (132, 88)) :
                                   ((60, 40), (84, 56), (96, 72), (108, 72), (132, 88), (156, 104))) :
            (profile == :startup ? ((60, 40), (84, 56), (96, 72), (108, 72), (120, 80)) :
                                   ((48, 32), (72, 48), (96, 64), (108, 72), (120, 80), (144, 96)))
    end
    # Full-column solve is naturally square/tall; wide fallback mirrors square.
    return _qq_sparse_solve_shapes(:square, profile, holdout)
end

@inline function _qq_sparse_solve_probe_cases(shape::Symbol, bucket::Symbol, profile::Symbol, holdout::Bool)
    dens = bucket == :low ? 0.05 : (bucket == :mid ? 0.2 : 0.45)
    shapes = collect(_qq_sparse_solve_shapes(shape, profile, holdout))
    return [(m=s[1], n=s[2], density=dens) for s in shapes]
end

@inline function _qq_sparse_solve_probe(case, seed::UInt64)
    rng = Random.MersenneTwister(seed)
    B = _qq_sparse_fullcolumn_rand(case.m, case.n, case.density; rng=rng)
    X = _qq_dense_rand(case.n, 4; rng=rng)
    return (B, B * X)
end

@inline function _clear_solve_autotune_state!()
    _clear_fullcolumn_cache!()
    GC.gc()
    return nothing
end

function _autotune_qq_sparse_solve_thresholds!(shape::Symbol, bucket::Symbol, profile::Symbol,
                                               rng::Random.AbstractRNG, progress_step::Function=_noop_progress_step)
    train_cases = _qq_sparse_solve_probe_cases(shape, bucket, profile, false)
    hold_cases = _qq_sparse_solve_probe_cases(shape, bucket, profile, true)
    isempty(train_cases) && return

    ntrain = length(train_cases)
    train_nnz = Vector{Int}(undef, ntrain)
    train_seed = Vector{UInt64}(undef, ntrain)
    t_exact = Vector{Float64}(undef, ntrain)
    t_nemo = Vector{Float64}(undef, ntrain)
    for (i, case) in enumerate(train_cases)
        seed = rand(rng, UInt64)
        train_seed[i] = seed
        probe = _qq_sparse_solve_probe(case, seed)
        train_nnz[i] = _sparse_nnz(probe[1])
        t_exact[i] = _bench_qq_probe(:solve, probe, :julia_exact; reps=1)
        t_nemo[i] = _bench_qq_probe(:solve, probe, :nemo; reps=1)
        _clear_solve_autotune_state!()
        progress_step("qq-solve-sparse-$(shape)-$(bucket) train $(i)/$(ntrain)")
    end

    s_exact = _smooth_by_work(train_nnz, t_exact)
    s_nemo = _smooth_by_work(train_nnz, t_nemo)
    margin = 0.05
    nnear = profile == :full ? 4 : 2
    reps_ref = profile == :full ? 4 : 2
    thr_ge, _, _ = _fit_single_threshold(train_nnz, s_exact, s_nemo; margin=margin)
    thr_le, _, _ = _fit_single_threshold_le(train_nnz, s_exact, s_nemo; margin=margin)
    for idx in union(_nearest_indices(train_nnz, thr_ge, nnear), _nearest_indices(train_nnz, thr_le, nnear))
        p = _qq_sparse_solve_probe(train_cases[idx], train_seed[idx])
        t_exact[idx] = _bench_qq_probe(:solve, p, :julia_exact; reps=reps_ref)
        t_nemo[idx] = _bench_qq_probe(:solve, p, :nemo; reps=reps_ref)
        _clear_solve_autotune_state!()
    end
    s_exact = _smooth_by_work(train_nnz, t_exact)
    s_nemo = _smooth_by_work(train_nnz, t_nemo)
    thr_ge, _, _ = _fit_single_threshold(train_nnz, s_exact, s_nemo; margin=margin)
    thr_le, _, _ = _fit_single_threshold_le(train_nnz, s_exact, s_nemo; margin=margin)

    nh = length(hold_cases)
    h_nnz = Vector{Int}(undef, nh)
    h_e = Vector{Float64}(undef, nh)
    h_n = Vector{Float64}(undef, nh)
    for (i, case) in enumerate(hold_cases)
        probe = _qq_sparse_solve_probe(case, rand(rng, UInt64))
        h_nnz[i] = _sparse_nnz(probe[1])
        h_e[i] = _bench_qq_probe(:solve, probe, :julia_exact; reps=1)
        h_n[i] = _bench_qq_probe(:solve, probe, :nemo; reps=1)
        _clear_solve_autotune_state!()
        progress_step("qq-solve-sparse-$(shape)-$(bucket) holdout $(i)/$(nh)")
    end

    ge_loss, ge_mismatch = _eval_single_threshold(h_nnz, h_e, h_n, thr_ge; margin=margin)
    le_loss, le_mismatch = _eval_single_threshold_le(h_nnz, h_e, h_n, thr_le; margin=margin)
    cand_use_ge = ge_loss < le_loss - 1e-9 || (abs(ge_loss - le_loss) <= 1e-9 && ge_mismatch <= le_mismatch)
    cand_thr = cand_use_ge ? thr_ge : thr_le
    cand_loss = cand_use_ge ? ge_loss : le_loss
    cand_mismatch = cand_use_ge ? ge_mismatch : le_mismatch

    old_thr = _get_qq_sparse_solve_threshold(shape, bucket)
    old_use_ge = _get_qq_sparse_solve_policy(shape, bucket)
    if old_use_ge
        old_loss, old_mismatch = _eval_single_threshold(h_nnz, h_e, h_n, old_thr; margin=margin)
        accept = _accept_threshold_update(cand_loss, cand_mismatch, old_loss, old_mismatch) ||
                 (cand_mismatch == old_mismatch && cand_loss <= old_loss * 1.01 &&
                  ((cand_use_ge && cand_thr < old_thr) || (!cand_use_ge)))
        if accept
            _set_qq_sparse_solve_threshold!(shape, bucket, cand_thr)
            _set_qq_sparse_solve_policy!(shape, bucket, cand_use_ge)
        end
    else
        old_loss, old_mismatch = _eval_single_threshold_le(h_nnz, h_e, h_n, old_thr; margin=margin)
        accept = _accept_threshold_update(cand_loss, cand_mismatch, old_loss, old_mismatch) ||
                 (cand_mismatch == old_mismatch && cand_loss <= old_loss * 1.01 &&
                  ((!cand_use_ge && cand_thr > old_thr) || cand_use_ge))
        if accept
            _set_qq_sparse_solve_threshold!(shape, bucket, cand_thr)
            _set_qq_sparse_solve_policy!(shape, bucket, cand_use_ge)
        end
    end
    return nothing
end

function _autotune_qq_sparse_solve_steps(profile::Symbol)
    steps = 0
    for shape in (:square, :tall), bucket in (:low, :mid, :high)
        steps += length(_qq_sparse_solve_probe_cases(shape, bucket, profile, false))
        steps += length(_qq_sparse_solve_probe_cases(shape, bucket, profile, true))
    end
    return steps
end

function _autotune_qq_routing_thresholds!(profile::Symbol=:full, progress_step::Function=_noop_progress_step)
    rng = Random.MersenneTwister(0x51515151)
    for shape in (:square, :tall, :wide)
        _autotune_qq_shape_thresholds!(:rank, shape, profile, rng, progress_step)
    end
    for shape in (:square, :tall, :wide)
        _autotune_qq_shape_thresholds!(:nullspace, shape, profile, rng, progress_step)
    end
    for shape in (:square, :tall, :wide)
        _autotune_qq_shape_thresholds!(:solve, shape, profile, rng, progress_step)
    end
    for shape in (:square, :tall), bucket in (:low, :mid, :high)
        _autotune_qq_sparse_solve_thresholds!(shape, bucket, profile, rng, progress_step)
    end
    # Mirror wide thresholds to square defaults for compatibility in rare wide-solve calls.
    for bucket in (:low, :mid, :high)
        _set_qq_sparse_solve_threshold!(:wide, bucket, _get_qq_sparse_solve_threshold(:square, bucket))
        _set_qq_sparse_solve_policy!(:wide, bucket, _get_qq_sparse_solve_policy(:square, bucket))
    end

    # Keep legacy scalar thresholds synchronized for compatibility/reporting.
    NEMO_THRESHOLD[] = Int(round(median(Float64[
        QQ_NEMO_RANK_THRESHOLD_SQUARE[],
        QQ_NEMO_RANK_THRESHOLD_TALL[],
        QQ_NEMO_RANK_THRESHOLD_WIDE[],
    ])))
    MODULAR_NULLSPACE_THRESHOLD[] = Int(round(median(Float64[
        QQ_MODULAR_NULLSPACE_THRESHOLD_SQUARE[],
        QQ_MODULAR_NULLSPACE_THRESHOLD_TALL[],
        QQ_MODULAR_NULLSPACE_THRESHOLD_WIDE[],
    ])))
    MODULAR_SOLVE_THRESHOLD[] = Int(round(median(Float64[
        QQ_MODULAR_SOLVE_THRESHOLD_SQUARE[],
        QQ_MODULAR_SOLVE_THRESHOLD_TALL[],
        QQ_MODULAR_SOLVE_THRESHOLD_WIDE[],
    ])))
    return nothing
end

function _autotune_qq_routing_steps(profile::Symbol)
    steps = 0
    for op in (:rank, :nullspace, :solve), shape in (:square, :tall, :wide)
        steps += length(_qq_probe_cases(op, shape, profile, false))
        steps += length(_qq_probe_cases(op, shape, profile, true))
    end
    steps += _autotune_qq_sparse_solve_steps(profile)
    return steps
end

function _autotune_fp_thresholds!(profile::Symbol=:full, progress_step::Function=_noop_progress_step)
    if !_have_nemo()
        return
    end
    F = PrimeField(5)
    rng = Random.MersenneTwister(0x54414d4552)
    rank_shapes = _autotune_fp_rank_shapes(profile)
    ns_shapes = _autotune_fp_nullspace_shapes(profile)
    solve_shapes = _autotune_fp_solve_shapes(profile)

    wr = Int[]
    trj = Float64[]
    trn = Float64[]
    rank_cases = Matrix{FpElem{5}}[]
    for (k, (m, n)) in enumerate(rank_shapes)
        A = _fp_dense_rand(F, m, n; rng=rng)
        push!(rank_cases, A)
        push!(wr, m * n)
        push!(trj, _bench_elapsed(() -> _rank_fp(A); reps=2))
        push!(trn, _bench_elapsed(() -> _nemo_rank(F, A); reps=2))
        progress_step("fp-rank threshold probe $(k)/$(length(rank_shapes))")
    end
    _refine_crossover_boundary!(wr, trj, trn,
        (i, reps) -> (_bench_elapsed(() -> _rank_fp(rank_cases[i]); reps=reps),
                      _bench_elapsed(() -> _nemo_rank(F, rank_cases[i]); reps=reps));
        extra_reps=(profile == :full ? 4 : 3),
        window=(profile == :full ? 2 : 1))
    FP_NEMO_RANK_THRESHOLD[] = _pick_crossover_threshold(wr, trj, trn, FP_NEMO_RANK_THRESHOLD[])

    wn = Int[]
    tnj = Float64[]
    tnn = Float64[]
    ns_cases = Matrix{FpElem{5}}[]
    for (k, (m, n)) in enumerate(ns_shapes)
        A = _fp_dense_rand(F, m, n; rng=rng)
        push!(ns_cases, A)
        push!(wn, m * n)
        push!(tnj, _bench_elapsed(() -> _nullspace_fp(A); reps=1))
        push!(tnn, _bench_elapsed(() -> _nemo_nullspace(F, A); reps=1))
        progress_step("fp-nullspace threshold probe $(k)/$(length(ns_shapes))")
    end
    _refine_crossover_boundary!(wn, tnj, tnn,
        (i, reps) -> (_bench_elapsed(() -> _nullspace_fp(ns_cases[i]); reps=reps),
                      _bench_elapsed(() -> _nemo_nullspace(F, ns_cases[i]); reps=reps));
        extra_reps=(profile == :full ? 3 : 2),
        window=(profile == :full ? 2 : 1))
    FP_NEMO_NULLSPACE_THRESHOLD[] = _pick_crossover_threshold(wn, tnj, tnn, FP_NEMO_NULLSPACE_THRESHOLD[])

    ws = Int[]
    tsj = Float64[]
    tsn = Float64[]
    solve_cases = Tuple{Matrix{FpElem{5}},Matrix{FpElem{5}}}[]
    for (k, (m, n)) in enumerate(solve_shapes)
        B = _fp_dense_rand(F, m, n; rng=rng)
        for i in 1:n
            B[i, i] = FpElem{5}(1)
        end
        X = _fp_dense_rand(F, n, 2; rng=rng)
        Y = B * X
        push!(solve_cases, (B, Y))
        push!(ws, m * n)
        push!(tsj, _bench_elapsed(() -> _solve_fullcolumn_fp(B, Y; check_rhs=false); reps=1))
        push!(tsn, _bench_elapsed(() -> _solve_fullcolumn_nemo_fp(F, B, Y; check_rhs=false, cache=false); reps=1))
        progress_step("fp-solve threshold probe $(k)/$(length(solve_shapes))")
    end
    _refine_crossover_boundary!(ws, tsj, tsn,
        (i, reps) -> begin
            B, Y = solve_cases[i]
            return (_bench_elapsed(() -> _solve_fullcolumn_fp(B, Y; check_rhs=false); reps=reps),
                    _bench_elapsed(() -> _solve_fullcolumn_nemo_fp(F, B, Y; check_rhs=false, cache=false); reps=reps))
        end;
        extra_reps=(profile == :full ? 3 : 2),
        window=(profile == :full ? 2 : 1))
    FP_NEMO_SOLVE_THRESHOLD[] = _pick_crossover_threshold(ws, tsj, tsn, FP_NEMO_SOLVE_THRESHOLD[])
end

function _autotune_float_thresholds!(profile::Symbol=:full, progress_step::Function=_noop_progress_step)
    F = RealField(Float64; rtol=1e-10, atol=1e-12)
    rng = Random.MersenneTwister(0x464c4f4154)
    shapes = _autotune_float_dense_shapes(profile)
    works = Int[]
    tqr = Float64[]
    tsvd = Float64[]
    cases = Matrix{Float64}[]
    for (k, (m, n)) in enumerate(shapes)
        A = rand(rng, Float64, m, n)
        push!(cases, A)
        push!(works, m * n)
        push!(tqr, _bench_elapsed(() -> _nullspace_float_qr_dense(F, A); reps=1))
        push!(tsvd, _bench_elapsed(() -> _nullspace_float_svd(F, A); reps=1))
        progress_step("float-dense nullspace threshold probe $(k)/$(length(shapes))")
    end
    _refine_crossover_boundary!(works, tqr, tsvd,
        (i, reps) -> (_bench_elapsed(() -> _nullspace_float_qr_dense(F, cases[i]); reps=reps),
                      _bench_elapsed(() -> _nullspace_float_svd(F, cases[i]); reps=reps));
        extra_reps=(profile == :full ? 4 : 3),
        window=(profile == :full ? 2 : 1))
    FLOAT_NULLSPACE_SVD_THRESHOLD[] = _pick_crossover_threshold(works, tqr, tsvd, FLOAT_NULLSPACE_SVD_THRESHOLD[])
end

function _autotune_float_sparse_svds_thresholds!(progress_step::Function=_noop_progress_step)
    _have_svds_backend() || return
    F = RealField(Float64; rtol=1e-10, atol=1e-12)
    rng = Random.MersenneTwister(0x53564453)
    dims = _autotune_float_sparse_dims()
    densities = _autotune_float_sparse_densities()

    winning_dims = Int[]
    winning_nnz = Int[]
    k = 0

    for n in dims
        for d in densities
            A = _float_sparse_singular_rand(n, d; rng=rng)
            tq = _bench_elapsed(() -> _nullspace_from_qr_sparse_float(F, A); reps=1)
            ts = _bench_elapsed(() -> begin
                Z = _nullspace_float_svds(F, A)
                Z === nothing || return Z
                return _nullspace_from_qr_sparse_float(F, A)
            end; reps=1)
            if ts < 0.9 * tq
                push!(winning_dims, n)
                push!(winning_nnz, nnz(A))
            end
            k += 1
            progress_step("float-sparse svds gate probe $(k)/$(length(dims) * length(densities))")
        end
    end

    isempty(winning_dims) && return
    FLOAT_SPARSE_SVDS_MIN_DIM[] = minimum(winning_dims)
    FLOAT_SPARSE_SVDS_MIN_NNZ[] = minimum(winning_nnz)
end

function _autotune_modular_thresholds!(profile::Symbol=:full, progress_step::Function=_noop_progress_step)
    rng = Random.MersenneTwister(0x4d4f44554c) # "MODUL"

    # Tune modular prime budget for _rankQQ_dim using exact-rank parity.
    rank_mats = [_qq_dense_rand(n, n; rng=rng) for n in (80, 112, 144)]
    rank_exact = [_rankQQ(A) for A in rank_mats]
    max_candidates = 2:min(length(DEFAULT_MODULAR_PRIMES), 8)
    best_max = MODULAR_MAX_PRIMES[]
    best_t = Inf
    for maxp in max_candidates
        ok = true
        ttot = 0.0
        for (k, (A, rex)) in enumerate(zip(rank_mats, rank_exact))
            rr = _rankQQ_dim(A; backend=:modular,
                            max_primes=maxp,
                            primes=DEFAULT_MODULAR_PRIMES,
                            small_threshold=1)
            ttot += _bench_elapsed(() -> _rankQQ_dim(A; backend=:modular,
                                                    max_primes=maxp,
                                                    primes=DEFAULT_MODULAR_PRIMES,
                                                    small_threshold=1); reps=1)
            progress_step("modular-rank max_primes=$(maxp) probe $(k)/$(length(rank_mats))")
            if rr != rex
                ok = false
                break
            end
        end
        if ok && ttot < best_t
            best_t = ttot
            best_max = maxp
        end
    end
    MODULAR_MAX_PRIMES[] = max(1, best_max)

    # Tune min_primes for reconstruction-based modular paths.
    ns_probe = _qq_dense_singular_rand(72; rng=rng)
    solve_B = _qq_dense_fullcolumn_rand(96, 64; rng=rng)
    solve_X = _qq_dense_rand(64, 2; rng=rng)
    solve_Y = solve_B * solve_X

    min_candidates = 1:min(4, MODULAR_MAX_PRIMES[])
    best_min = MODULAR_MIN_PRIMES[]
    best_min_t = Inf
    for minp in min_candidates
        ttot = 0.0
        N = _nullspace_modularQQ(ns_probe;
                                 primes=DEFAULT_MODULAR_PRIMES,
                                 min_primes=minp,
                                 max_primes=MODULAR_MAX_PRIMES[])
        Xh = _solve_fullcolumn_modularQQ(solve_B, solve_Y;
                                         primes=DEFAULT_MODULAR_PRIMES,
                                         min_primes=minp,
                                         max_primes=MODULAR_MAX_PRIMES[],
                                         check_rhs=false)
        ttot += _bench_elapsed(() -> _nullspace_modularQQ(ns_probe;
                                                           primes=DEFAULT_MODULAR_PRIMES,
                                                           min_primes=minp,
                                                           max_primes=MODULAR_MAX_PRIMES[]); reps=1)
        progress_step("modular min_primes=$(minp) nullspace probe")
        ttot += _bench_elapsed(() -> _solve_fullcolumn_modularQQ(solve_B, solve_Y;
                                                                  primes=DEFAULT_MODULAR_PRIMES,
                                                                  min_primes=minp,
                                                                  max_primes=MODULAR_MAX_PRIMES[],
                                                                  check_rhs=false); reps=1)
        progress_step("modular min_primes=$(minp) solve probe")
        ok = (N !== nothing && _verify_nullspaceQQ(ns_probe, N) &&
              Xh !== nothing && _verify_solveQQ(solve_B, Xh, solve_Y))
        if ok && ttot < best_min_t
            best_min_t = ttot
            best_min = minp
        end
    end
    MODULAR_MIN_PRIMES[] = min(best_min, MODULAR_MAX_PRIMES[])

    # Tune crossover threshold for QQ modular nullspace and solve.
    wn = Int[]
    tne = Float64[]
    tnm = Float64[]
    ns_shapes = _autotune_modular_nullspace_shapes(profile)
    ns_cases = Matrix{QQ}[]
    for (k, (m, n)) in enumerate(ns_shapes)
        A = _qq_dense_rand(m, n; rng=rng)
        min(m, n) >= 2 && (A[:, end] = A[:, 1])
        push!(ns_cases, A)
        push!(wn, m * n)
        push!(tne, _bench_elapsed(() -> _nullspaceQQ(A); reps=1))
        push!(tnm, _bench_elapsed(() -> begin
            N = _nullspace_modularQQ(A;
                                     primes=DEFAULT_MODULAR_PRIMES,
                                     min_primes=MODULAR_MIN_PRIMES[],
                max_primes=MODULAR_MAX_PRIMES[])
            N === nothing ? _nullspaceQQ(A) : N
        end; reps=1))
        progress_step("modular-nullspace crossover probe $(k)/$(length(ns_shapes))")
    end
    _refine_crossover_boundary!(wn, tne, tnm,
        (i, reps) -> begin
            A = ns_cases[i]
            return (_bench_elapsed(() -> _nullspaceQQ(A); reps=reps),
                    _bench_elapsed(() -> begin
                        N = _nullspace_modularQQ(A;
                                                 primes=DEFAULT_MODULAR_PRIMES,
                                                 min_primes=MODULAR_MIN_PRIMES[],
                                                 max_primes=MODULAR_MAX_PRIMES[])
                        N === nothing ? _nullspaceQQ(A) : N
                    end; reps=reps))
        end;
        extra_reps=4,
        window=2)
    MODULAR_NULLSPACE_THRESHOLD[] = _pick_crossover_threshold(wn, tne, tnm, MODULAR_NULLSPACE_THRESHOLD[])

    ws = Int[]
    tse = Float64[]
    tsm = Float64[]
    solve_shapes = _autotune_modular_solve_shapes(profile)
    solve_cases = Tuple{Matrix{QQ},Matrix{QQ}}[]
    for (k, (m, n)) in enumerate(solve_shapes)
        B = _qq_dense_fullcolumn_rand(m, n; rng=rng)
        X = _qq_dense_rand(size(B, 2), 2; rng=rng)
        Y = B * X
        push!(solve_cases, (B, Y))
        push!(ws, m * n)
        push!(tse, _bench_elapsed(() -> _solve_fullcolumnQQ(B, Y; check_rhs=false); reps=1))
        push!(tsm, _bench_elapsed(() -> begin
            Xh = _solve_fullcolumn_modularQQ(B, Y;
                                             primes=DEFAULT_MODULAR_PRIMES,
                                             min_primes=MODULAR_MIN_PRIMES[],
                                             max_primes=MODULAR_MAX_PRIMES[],
                                             check_rhs=false)
            Xh === nothing ? _solve_fullcolumnQQ(B, Y; check_rhs=false) : Xh
        end; reps=1))
        progress_step("modular-solve crossover probe $(k)/$(length(solve_shapes))")
    end
    _refine_crossover_boundary!(ws, tse, tsm,
        (i, reps) -> begin
            B, Y = solve_cases[i]
            return (_bench_elapsed(() -> _solve_fullcolumnQQ(B, Y; check_rhs=false); reps=reps),
                    _bench_elapsed(() -> begin
                        Xh = _solve_fullcolumn_modularQQ(B, Y;
                                                         primes=DEFAULT_MODULAR_PRIMES,
                                                         min_primes=MODULAR_MIN_PRIMES[],
                                                         max_primes=MODULAR_MAX_PRIMES[],
                                                         check_rhs=false)
                        Xh === nothing ? _solve_fullcolumnQQ(B, Y; check_rhs=false) : Xh
                    end; reps=reps))
        end;
        extra_reps=4,
        window=2)
    MODULAR_SOLVE_THRESHOLD[] = _pick_crossover_threshold(ws, tse, tsm, MODULAR_SOLVE_THRESHOLD[])

    # Tune _rankQQ_dim exact/modular crossover.
    wr = Int[]
    tre = Float64[]
    trm = Float64[]
    rank_shapes = _autotune_rankqq_dim_shapes(profile)
    rank_cases = Matrix{QQ}[]
    for (k, (m, n)) in enumerate(rank_shapes)
        A = _qq_dense_rand(m, n; rng=rng)
        push!(rank_cases, A)
        push!(wr, m * n)
        push!(tre, _bench_elapsed(() -> _rankQQ(A); reps=1))
        push!(trm, _bench_elapsed(() -> _rankQQ_dim(A; backend=:modular,
                                                   max_primes=MODULAR_MAX_PRIMES[],
                                                   primes=DEFAULT_MODULAR_PRIMES,
                                                   small_threshold=1); reps=1))
        progress_step("_rankQQ_dim crossover probe $(k)/$(length(rank_shapes))")
    end
    _refine_crossover_boundary!(wr, tre, trm,
        (i, reps) -> begin
            A = rank_cases[i]
            return (_bench_elapsed(() -> _rankQQ(A); reps=reps),
                    _bench_elapsed(() -> _rankQQ_dim(A; backend=:modular,
                                                    max_primes=MODULAR_MAX_PRIMES[],
                                                    primes=DEFAULT_MODULAR_PRIMES,
                                                    small_threshold=1); reps=reps))
        end;
        extra_reps=4,
        window=2)
    RANKQQ_DIM_SMALL_THRESHOLD[] = _pick_crossover_threshold(wr, tre, trm, RANKQQ_DIM_SMALL_THRESHOLD[])
end

function _zn_rand_phi(cm, field, nrows::Int, ncols::Int, rng::Random.AbstractRNG)
    K = cm.coeff_type(field)
    Phi = Matrix{K}(undef, nrows, ncols)
    @inbounds for i in 1:nrows, j in 1:ncols
        Phi[i, j] = cm.coerce(field, rand(rng, -2:2))
    end
    return Phi
end

function _zn_dimat_fixture_friendly(fz, cm, field; ncuts::Int, rng::Random.AbstractRNG)
    n = 2
    tau = fz.face(n, [false, true])
    thresholds = collect(-ncuts:ncuts)
    FlatT = typeof(fz.IndFlat(tau, (0, 0); id=:F0))
    InjT = typeof(fz.IndInj(tau, (1, 0); id=:E0))
    flats = Vector{FlatT}(undef, length(thresholds))
    injectives = Vector{InjT}(undef, length(thresholds))
    @inbounds for (i, t) in enumerate(thresholds)
        flats[i] = fz.IndFlat(tau, (t, 0); id=Symbol(:F, i))
        injectives[i] = fz.IndInj(tau, (t + 1, 0); id=Symbol(:E, i))
    end
    Phi = _zn_rand_phi(cm, field, length(injectives), length(flats), rng)
    FG = fz.Flange{cm.coeff_type(field)}(n, flats, injectives, Phi; field=field)
    return FG, (-2 * ncuts, -2 * ncuts), (2 * ncuts, 2 * ncuts)
end

function _zn_dimat_fixture_adversarial(fz, cm, field; ncuts::Int, rng::Random.AbstractRNG)
    n = 2
    m = 2 * ncuts + 4
    tau = fz.face(n, [false, false])
    FlatT = typeof(fz.IndFlat(tau, (0, 0); id=:F0))
    InjT = typeof(fz.IndInj(tau, (0, 0); id=:E0))
    flats = Vector{FlatT}(undef, m)
    injectives = Vector{InjT}(undef, m)
    lo = -2 * ncuts
    hi = 2 * ncuts
    @inbounds for i in 1:m
        flats[i] = fz.IndFlat(tau, (rand(rng, lo:hi), rand(rng, lo:hi)); id=Symbol(:F, i))
        injectives[i] = fz.IndInj(tau, (rand(rng, lo:hi), rand(rng, lo:hi)); id=Symbol(:E, i))
    end
    Phi = _zn_rand_phi(cm, field, m, m, rng)
    FG = fz.Flange{cm.coeff_type(field)}(n, flats, injectives, Phi; field=field)
    return FG, (lo, lo), (hi, hi)
end

function _zn_sample_box_points(a::NTuple{N,Int}, b::NTuple{N,Int}, nqueries::Int,
                               rng::Random.AbstractRNG) where {N}
    out = Vector{NTuple{N,Int}}(undef, nqueries)
    @inbounds for i in 1:nqueries
        out[i] = ntuple(k -> rand(rng, a[k]:b[k]), N)
    end
    return out
end

function _rank_words_fixture(fz, cm, field; nflats::Int, ninj::Int, seed::UInt)
    rng = Random.MersenneTwister(seed)
    K = cm.coeff_type(field)
    flats = Vector{fz.IndFlat{2}}(undef, nflats)
    injectives = Vector{fz.IndInj{2}}(undef, ninj)
    @inbounds for i in 1:nflats
        b = [rand(rng, -8:12) for _ in 1:2]
        flats[i] = fz.IndFlat(fz.face(2, [rand(rng) < 0.4 for _ in 1:2]), b; id=Symbol(:F, i))
    end
    @inbounds for j in 1:ninj
        b = [rand(rng, -6:14) for _ in 1:2]
        injectives[j] = fz.IndInj(fz.face(2, [rand(rng) < 0.4 for _ in 1:2]), b; id=Symbol(:E, j))
    end
    phi = Matrix{K}(undef, ninj, nflats)
    @inbounds for i in 1:ninj, j in 1:nflats
        phi[i, j] = cm.coerce(field, rand(rng, -2:2))
    end
    return fz.Flange{K}(2, flats, injectives, phi; field=field)
end

function _rank_words_probe_states(fz, FG, nx::Int)
    pts = Vector{NTuple{2,Int}}(undef, 8 * nx)
    k = 1
    @inbounds for j in 0:7, i in 0:(nx - 1)
        pts[k] = (-16 + i, -6 + 3j)
        k += 1
    end
    probe = fz.FlangeDimCache(FG)
    seen = Set{UInt64}()
    states = Tuple{Vector{UInt64},Vector{UInt64},Int,Int}[]
    @inbounds for p in pts
        nr = fz._fill_active_injectives_words_kernel!(probe.row_words, probe.kernel, p)
        nc = fz._fill_active_flats_words_kernel!(probe.col_words, probe.kernel, p)
        (nr == 0 || nc == 0) && continue
        key = fz._active_words_hash(probe.row_words, probe.col_words)
        key in seen && continue
        push!(seen, key)
        push!(states, (copy(probe.row_words), copy(probe.col_words), nr, nc))
    end
    return states
end

function _autotune_rank_words_threshold!(profile::Symbol=:full, progress_step::Function=_noop_progress_step)
    _have_nemo() || return nothing
    pm = parentmodule(@__MODULE__)
    if !isdefined(pm, :FlangeZn) || !isdefined(pm, :CoreModules)
        return nothing
    end
    fz = getfield(pm, :FlangeZn)
    cm = getfield(pm, :CoreModules)
    field = cm.QQField()
    nx = _autotune_rank_words_grid_width(profile)
    reps = _autotune_rank_words_reps(profile)
    probes = NamedTuple{(:FG, :states, :label),Tuple{Any,Vector{Tuple{Vector{UInt64},Vector{UInt64},Int,Int}},String}}[]
    for (idx, (nflats, ninj)) in enumerate(_autotune_rank_words_probe_shapes(profile))
        FG = _rank_words_fixture(fz, cm, field; nflats=nflats, ninj=ninj,
                                 seed=0x5241575752440000 + UInt(idx))
        states = _rank_words_probe_states(fz, FG, nx)
        isempty(states) || push!(probes, (FG=FG, states=states, label="nflat=$(nflats) ninj=$(ninj)"))
    end
    isempty(probes) && return nothing

    FG0 = probes[1].FG
    for (row_words, col_words, nr, nc) in Iterators.take(probes[1].states, min(length(probes[1].states), 8))
        rank_restricted_words(FG0.field, FG0.phi, row_words, col_words, nr, nc;
                              nrows=size(FG0.phi, 1), ncols=size(FG0.phi, 2), backend=:auto)
    end

    old_thr = Int(RANKQQ_RESTRICTED_WORDS_NEMO_THRESHOLD[])
    best_thr = old_thr
    best_time = Inf
    cand_times = Dict{Int,Float64}()
    for thr in _autotune_rank_words_threshold_candidates(profile)
        RANKQQ_RESTRICTED_WORDS_NEMO_THRESHOLD[] = thr
        ttot = 0.0
        for probe in probes
            FG = probe.FG
            ttot += _bench_elapsed_median(() -> begin
                s = 0
                @inbounds for (row_words, col_words, nr, nc) in probe.states
                    s += rank_restricted_words(FG.field, FG.phi, row_words, col_words, nr, nc;
                                               nrows=size(FG.phi, 1), ncols=size(FG.phi, 2),
                                               backend=:auto)
                end
                s
            end; reps=reps)
            progress_step("qq-rank-words threshold=$(thr) $(probe.label)")
        end
        cand_times[thr] = ttot
        if ttot < best_time
            best_time = ttot
            best_thr = thr
        end
    end

    old_time = get(cand_times, old_thr, Inf)
    if best_thr != old_thr && !(best_time <= 0.95 * old_time)
        best_thr = old_thr
    end

    RANKQQ_RESTRICTED_WORDS_NEMO_THRESHOLD[] = best_thr
    return nothing
end

function _autotune_zn_dimat_threshold!(profile::Symbol=:full, progress_step::Function=_noop_progress_step)
    pm = parentmodule(@__MODULE__)
    if !isdefined(pm, :FlangeZn) || !isdefined(pm, :CoreModules)
        return nothing
    end
    fz = getfield(pm, :FlangeZn)
    cm = getfield(pm, :CoreModules)
    field = cm.QQField()
    nqueries = _autotune_zn_dimat_queries_per_probe(profile)
    reps = _autotune_zn_dimat_reps(profile)
    ncutses = _autotune_zn_dimat_probe_ncuts(profile)
    candidates = _autotune_zn_dimat_threshold_candidates(profile)
    rng = Random.MersenneTwister(0x5a4e44494d4154) # "ZNDIMAT"

    probes = Tuple{Any,Vector{NTuple{2,Int}},String}[]
    for ncuts in ncutses
        FGf, af, bf = _zn_dimat_fixture_friendly(fz, cm, field; ncuts=ncuts, rng=rng)
        Qf = _zn_sample_box_points(af, bf, nqueries, rng)
        push!(probes, (FGf, Qf, "friendly ncuts=$(ncuts)"))

        FGa, aa, ba = _zn_dimat_fixture_adversarial(fz, cm, field; ncuts=ncuts, rng=rng)
        Qa = _zn_sample_box_points(aa, ba, nqueries, rng)
        push!(probes, (FGa, Qa, "adversarial ncuts=$(ncuts)"))
    end

    # Warm up compilation for both dim_at branches on one representative probe.
    if !isempty(probes)
        FG0, Q0, _ = probes[1]
        for i in 1:min(length(Q0), 64)
            fz.dim_at(FG0, Q0[i])
        end
    end

    best_thr = Int(ZN_QQ_DIMAT_SUBMATRIX_WORK_THRESHOLD[])
    best_time = Inf
    old_thr = best_thr
    cand_times = Dict{Int,Float64}()
    for thr in candidates
        _set_zn_qq_dimat_submatrix_work_threshold!(thr)
        ttot = 0.0
        for (FG, Q, label) in probes
            ttot += _bench_elapsed_median(() -> begin
                s = 0
                @inbounds for g in Q
                    s += fz.dim_at(FG, g)
                end
                s
            end; reps=reps)
            progress_step("zn-dim_at threshold=$(thr) $(label)")
        end
        cand_times[thr] = ttot
        if ttot < best_time
            best_time = ttot
            best_thr = thr
        end
    end

    # Keep incumbent unless the candidate win is material to avoid noise-driven jumps.
    old_time = get(cand_times, old_thr, Inf)
    if best_thr != old_thr && !(best_time <= 0.95 * old_time)
        best_thr = old_thr
    end

    _set_zn_qq_dimat_submatrix_work_threshold!(best_thr)
    return nothing
end

function autotune_linalg_thresholds!(; path::AbstractString=_linalg_thresholds_path(),
                                     save::Bool=true,
                                     quiet::Bool=false,
                                     profile::Symbol=:full)
    profile in (:full, :startup) || error("autotune_linalg_thresholds!: profile must be :full or :startup.")
    progress = _autotune_progress_init(_autotune_total_steps(profile); enabled=!quiet)
    step = label -> _autotune_progress_step!(progress, label)
    old = _current_linalg_thresholds()
    try
        profile == :full && _autotune_modular_thresholds!(profile, step)
        _autotune_fp_thresholds!(profile, step)
        _autotune_float_thresholds!(profile, step)
        _autotune_float_sparse_svds_thresholds!(step)
        _autotune_qq_routing_thresholds!(profile, step)
        _autotune_rank_words_threshold!(profile, step)
        profile == :full && _autotune_zn_dimat_threshold!(profile, step)
    catch err
        _apply_linalg_thresholds!(old)
        rethrow(err)
    end
    _autotune_progress_finish!(progress)
    if save
        _save_linalg_thresholds!(; path=path)
    end
    quiet || @info "FieldLinAlg: autotuned thresholds." path profile thresholds=_current_linalg_thresholds()
    return _current_linalg_thresholds()
end
