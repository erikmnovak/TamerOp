module TamerOpDistancesExt

using Distances

const TO = let pm = nothing
    if isdefined(Main, :TamerOp)
        pm = getfield(Main, :TamerOp)
    else
        @eval import TamerOp
        pm = TamerOp
    end
    pm
end

const FEA = TO.Featurizers
const Inv = TO.Invariants

abstract type AbstractTamerDistanceMetric <: Distances.PreMetric end

struct MatchingDistanceMetric{O,NT<:NamedTuple} <: AbstractTamerDistanceMetric
    method::Symbol
    opts::O
    kwargs::NT
end

function MatchingDistanceMetric(; method::Symbol=:auto,
                                 opts=nothing,
                                 kwargs...)
    return MatchingDistanceMetric{typeof(opts),typeof((; kwargs...))}(method, opts, (; kwargs...))
end

@inline function Distances.evaluate(m::MatchingDistanceMetric, x, y)
    return TO.matching_distance(x, y; method=m.method, opts=m.opts, m.kwargs...)
end

struct MPLandscapeDistanceMetric <: AbstractTamerDistanceMetric
    p::Float64
    weight_mode::Symbol
end

MPLandscapeDistanceMetric(; p::Real=2, weight_mode::Symbol=:check) =
    MPLandscapeDistanceMetric(float(p), weight_mode)

@inline function Distances.evaluate(m::MPLandscapeDistanceMetric, x, y)
    return Inv.mp_landscape_distance(x, y; p=m.p, weight_mode=m.weight_mode)
end

struct ProjectedDistanceMetric{W} <: AbstractTamerDistanceMetric
    dist::Symbol
    agg::Symbol
    p::Float64
    q::Float64
    dir_weights::W
    threads::Bool
end

function ProjectedDistanceMetric(; dist::Symbol=:bottleneck,
                                  agg::Symbol=:mean,
                                  p::Real=1,
                                  q::Real=1,
                                  dir_weights=nothing,
                                  threads::Bool=(Base.Threads.nthreads() > 1))
    w = dir_weights === nothing ? nothing : Float64[float(v) for v in dir_weights]
    return ProjectedDistanceMetric{typeof(w)}(dist, agg, float(p), float(q), w, threads)
end

@inline function Distances.evaluate(m::ProjectedDistanceMetric, x, y)
    return Inv.projected_distance(x, y;
                                  dist=m.dist,
                                  agg=m.agg,
                                  p=m.p,
                                  q=m.q,
                                  dir_weights=m.dir_weights,
                                  threads=m.threads)
end

struct BottleneckDistanceMetric{NT<:NamedTuple} <: AbstractTamerDistanceMetric
    kwargs::NT
end

BottleneckDistanceMetric(; kwargs...) =
    BottleneckDistanceMetric{typeof((; kwargs...))}((; kwargs...))

@inline function Distances.evaluate(m::BottleneckDistanceMetric, x, y)
    return Inv.bottleneck_distance(x, y; m.kwargs...)
end

struct WassersteinDistanceMetric{NT<:NamedTuple} <: AbstractTamerDistanceMetric
    kwargs::NT
end

WassersteinDistanceMetric(; kwargs...) =
    WassersteinDistanceMetric{typeof((; kwargs...))}((; kwargs...))

@inline function Distances.evaluate(m::WassersteinDistanceMetric, x, y)
    return Inv.wasserstein_distance(x, y; m.kwargs...)
end

struct MPPImageDistanceMetric <: AbstractTamerDistanceMetric end

@inline Distances.evaluate(::MPPImageDistanceMetric, x, y) = Inv.mpp_image_distance(x, y)

function Distances.pairwise(m::AbstractTamerDistanceMetric, xs::AbstractVector)
    n = length(xs)
    D = Matrix{Float64}(undef, n, n)
    @inbounds for i in 1:n
        xi = xs[i]
        for j in i:n
            v = float(Distances.evaluate(m, xi, xs[j]))
            D[i, j] = v
            D[j, i] = v
        end
    end
    return D
end

function Distances.pairwise(m::AbstractTamerDistanceMetric,
                            xs::AbstractVector,
                            ys::AbstractVector)
    nx = length(xs)
    ny = length(ys)
    D = Matrix{Float64}(undef, nx, ny)
    @inbounds for i in 1:nx
        xi = xs[i]
        for j in 1:ny
            D[i, j] = float(Distances.evaluate(m, xi, ys[j]))
        end
    end
    return D
end

FEA._set_distances_impl!((
    matching = (; kwargs...) -> MatchingDistanceMetric(; kwargs...),
    mp_landscape = (; kwargs...) -> MPLandscapeDistanceMetric(; kwargs...),
    projected = (; kwargs...) -> ProjectedDistanceMetric(; kwargs...),
    bottleneck = (; kwargs...) -> BottleneckDistanceMetric(; kwargs...),
    wasserstein = (; kwargs...) -> WassersteinDistanceMetric(; kwargs...),
    mpp_image = (; kwargs...) -> MPPImageDistanceMetric(),
))

end # module
