module TamerOpKernelFunctionsExt

using KernelFunctions

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

abstract type AbstractTamerKernel <: KernelFunctions.Kernel end

struct MPLandscapeKernel{G} <: AbstractTamerKernel
    kind::Symbol
    sigma::Float64
    p::Float64
    gamma::G
    weight_mode::Symbol
end

function MPLandscapeKernel(; kind::Symbol=:gaussian,
                            sigma::Real=1.0,
                            p::Real=2,
                            gamma=nothing,
                            weight_mode::Symbol=:check)
    return MPLandscapeKernel{typeof(gamma)}(kind, float(sigma), float(p), gamma, weight_mode)
end

struct ProjectedKernel{W} <: AbstractTamerKernel
    kind::Symbol
    sigma::Float64
    p::Float64
    q::Float64
    agg::Symbol
    dir_weights::W
    threads::Bool
end

function ProjectedKernel(; kind::Symbol=:wasserstein_gaussian,
                          sigma::Real=1.0,
                          p::Real=1,
                          q::Real=1,
                          agg::Symbol=:mean,
                          dir_weights=nothing,
                          threads::Bool=(Base.Threads.nthreads() > 1))
    w = dir_weights === nothing ? nothing : Float64[float(v) for v in dir_weights]
    return ProjectedKernel{typeof(w)}(kind, float(sigma), float(p), float(q), agg, w, threads)
end

struct MPPImageKernel <: AbstractTamerKernel
    sigma::Float64
end

MPPImageKernel(; sigma::Real=1.0) = MPPImageKernel(float(sigma))

struct PointSignedMeasureKernel <: AbstractTamerKernel
    sigma::Float64
end

PointSignedMeasureKernel(; sigma::Real=1.0) = PointSignedMeasureKernel(float(sigma))

struct RectangleSignedBarcodeKernel <: AbstractTamerKernel
    kind::Symbol
    sigma::Float64
end

RectangleSignedBarcodeKernel(; kind::Symbol=:linear, sigma::Real=1.0) =
    RectangleSignedBarcodeKernel(kind, float(sigma))

@inline (k::AbstractTamerKernel)(x, y) = KernelFunctions.kappa(k, x, y)

@inline function KernelFunctions.kappa(k::MPLandscapeKernel, x, y)
    return Inv.mp_landscape_kernel(x, y;
                                   kind=k.kind,
                                   sigma=k.sigma,
                                   p=k.p,
                                   gamma=k.gamma,
                                   weight_mode=k.weight_mode)
end

@inline function KernelFunctions.kappa(k::ProjectedKernel, x, y)
    return Inv.projected_kernel(x, y;
                                kind=k.kind,
                                sigma=k.sigma,
                                p=k.p,
                                q=k.q,
                                agg=k.agg,
                                dir_weights=k.dir_weights,
                                threads=k.threads)
end

@inline KernelFunctions.kappa(k::MPPImageKernel, x, y) =
    Inv.mpp_image_kernel(x, y; sigma=k.sigma)

@inline KernelFunctions.kappa(k::PointSignedMeasureKernel, x, y) =
    Inv.point_signed_measure_kernel(x, y; sigma=k.sigma)

@inline KernelFunctions.kappa(k::RectangleSignedBarcodeKernel, x, y) =
    Inv.rectangle_signed_barcode_kernel(x, y; kind=k.kind, sigma=k.sigma)

function KernelFunctions.kernelmatrix(k::AbstractTamerKernel, xs::AbstractVector)
    n = length(xs)
    K = Matrix{Float64}(undef, n, n)
    @inbounds for i in 1:n
        xi = xs[i]
        for j in i:n
            v = float(KernelFunctions.kappa(k, xi, xs[j]))
            K[i, j] = v
            K[j, i] = v
        end
    end
    return K
end

function KernelFunctions.kernelmatrix(k::AbstractTamerKernel,
                                      xs::AbstractVector,
                                      ys::AbstractVector)
    m = length(xs)
    n = length(ys)
    K = Matrix{Float64}(undef, m, n)
    @inbounds for i in 1:m
        xi = xs[i]
        for j in 1:n
            K[i, j] = float(KernelFunctions.kappa(k, xi, ys[j]))
        end
    end
    return K
end

function KernelFunctions.kernelmatrix_diag(k::AbstractTamerKernel, xs::AbstractVector)
    n = length(xs)
    d = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        d[i] = float(KernelFunctions.kappa(k, xs[i], xs[i]))
    end
    return d
end

FEA._set_kernelfunctions_impl!((
    mp_landscape = (; kwargs...) -> MPLandscapeKernel(; kwargs...),
    projected = (; kwargs...) -> ProjectedKernel(; kwargs...),
    mpp_image = (; kwargs...) -> MPPImageKernel(; kwargs...),
    point_signed_measure = (; kwargs...) -> PointSignedMeasureKernel(; kwargs...),
    rectangle_signed_barcode = (; kwargs...) -> RectangleSignedBarcodeKernel(; kwargs...),
))

end # module
