abstract type AbstractDistribution end
abstract type AbstractProcess end

struct NormalDistribution{T,M<:AbstractArray{T},C<:AbstractArray{T},K<:Cholesky{T}} <:
       AbstractDistribution
    μ::M
    Σ::C
    Σ_ch::K
end

function NormalDistribution(μ, Σ_ch::Cholesky)
    return NormalDistribution{eltype{μ},typeof{μ},typeof(Σ_ch)}(μ, Σ_ch)
end

function NormalDistribution(μ, Σ::AbstractArray)
    Σ_ch = cholesky(Σ .+ 1e-7)
    return NormalDistribution{eltype(μ),typeof(μ),typeof(Σ),typeof(Σ_ch)}(μ, Σ, Σ_ch)
end

function sample(N::NormalDistribution)
    dim = length(N.μ)
    s = randn(dim)
    s .= N.Σ_ch.L * s .+ N.μ
    return s
end

sample(gp::GaussianProcess, x) = sample(gp(x))
sample(gp::GaussianProcess, x, θ) = sample(gp(x, θ))

struct GaussianProcess{M,K<:AbstractKernel} <: AbstractProcess
    f_μ::M
    kernel::K
end

function (gp::GaussianProcess)(x)
    θ = rand(eltype(x), dim_hp(gp.kernel, size(x, 1)))
    return gp(x, θ)
end

function (gp::GaussianProcess)(x, θ)
    μ = gp.f_μ.(eachcol(x))
    Σ = kernel(gp.kernel, θ, x)
    return NormalDistribution(μ, Σ)
end
