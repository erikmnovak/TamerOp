# =============================================================================
# options_helpers.jl
#
# Shared invariant-option normalization helpers.
#
# Owns:
# - strict/thread default normalization
# - lightweight keyword filtering helpers
# - orthant direction normalization
# - direction normalization shared across slice/image/fibered invariants
# - invariant selection/axes keyword projection
#
# Does not own:
# - box normalization itself
# - invariant kernels
# =============================================================================

# Developer note
# - `_default_*` normalizes optional user-supplied option fields into the
#   canonical invariant defaults consumed by owner kernels.
# - `orthant_directions` and `_normalize_dir` canonicalize direction data as
#   positive `Float64` tuples/vectors, with explicit `:none`, `:L1`, or `:Linf`
#   normalization semantics.
# - `_selection_kwargs_from_opts` and `_axes_kwargs_from_opts` project just the
#   `InvariantOptions` fields consumed downstream by invariant-family owners.
#
# Call path
# - User options -> shared normalization here -> owner-specific kernels in
#   `Invariants`, `SliceInvariants`, `Fibered2D`, or `MultiparameterImages`.

@inline _default_strict(x) = (x === nothing) ? true : x
@inline _default_threads(x) = (x === nothing) ? (Threads.nthreads() > 1) : x

@inline function _drop_keys(nt::NamedTuple, bad::Tuple)
    return (; (k => v for (k, v) in pairs(nt) if !(k in bad))...)
end
@inline _drop_keys(pairs::Base.Pairs, bad::Tuple) = _drop_keys(NamedTuple(pairs), bad)

@inline function orthant_directions(d::Integer)
    d > 0 || error("orthant_directions: dimension must be positive")
    v = ntuple(_ -> 1.0 / d, d)
    return [v]
end

@inline function orthant_directions(d::Integer, directions)
    d > 0 || error("orthant_directions: dimension must be positive")
    dirs = Vector{NTuple{d,Float64}}()
    for dir in directions
        length(dir) == d || error("orthant_directions: expected direction of length $d, got $(length(dir))")
        s = 0.0
        @inbounds for i in 1:d
            s += abs(float(dir[i]))
        end
        s > 0 || error("orthant_directions: zero direction not allowed")
        push!(dirs, ntuple(i -> abs(float(dir[i])) / s, d))
    end
    return dirs
end

@inline function _normalize_dir(dir::NTuple{N,<:Real}, normalize::Symbol) where {N}
    if normalize === :none
        return ntuple(i -> Float64(float(dir[i])), N)
    elseif normalize === :L1
        s = 0.0
        @inbounds for i in 1:N
            s += abs(Float64(float(dir[i])))
        end
        s > 0 || error("_normalize_dir: zero direction vector")
        return ntuple(i -> Float64(float(dir[i])) / s, N)
    elseif normalize === :Linf
        m = 0.0
        @inbounds for i in 1:N
            m = max(m, abs(Float64(float(dir[i]))))
        end
        m > 0 || error("_normalize_dir: zero direction vector")
        return ntuple(i -> Float64(float(dir[i])) / m, N)
    else
        error("_normalize_dir: normalize must be :none, :L1, or :Linf")
    end
end

function _normalize_dir(dir::AbstractVector, normalize::Symbol)
    if normalize === :none
        return Float64[float(x) for x in dir]
    elseif normalize === :L1
        s = 0.0
        @inbounds for i in eachindex(dir)
            s += abs(Float64(float(dir[i])))
        end
        s > 0 || error("_normalize_dir: zero direction vector")
        return Float64[Float64(float(x)) / s for x in dir]
    elseif normalize === :Linf
        m = 0.0
        @inbounds for i in eachindex(dir)
            m = max(m, abs(Float64(float(dir[i]))))
        end
        m > 0 || error("_normalize_dir: zero direction vector")
        return Float64[Float64(float(x)) / m for x in dir]
    else
        error("_normalize_dir: normalize must be :none, :L1, or :Linf")
    end
end

@inline function _selection_kwargs_from_opts(opts::InvariantOptions)
    return (;
        box = opts.box,
        strict = _default_strict(opts.strict),
        threads = _default_threads(opts.threads),
    )
end

@inline function _axes_kwargs_from_opts(opts::InvariantOptions)
    return (;
        axes = opts.axes,
        axes_policy = opts.axes_policy,
        max_axis_len = opts.max_axis_len,
    )
end

@inline function _eye(::Type{K}, n::Int) where {K}
    M = zeros(K, n, n)
    for i in 1:n
        M[i, i] = one(K)
    end
    return M
end
