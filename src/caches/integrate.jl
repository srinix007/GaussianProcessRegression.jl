abstract type AbstractIntegCache <: AbstractCache end
abstract type AbstractAntiDerivCache <: AbstractIntegCache end
abstract type AbstractWtCache <: AbstractIntegCache end

struct AntiDerivCache{T,AD<:AbstractArray{T}} <: AbstractAntiDerivCache
    k1::AD
    k2::T
end

function AntiDerivCache(md::AbstractGPRModel)
    nx = size(md.x, 2)
    k1 = similar(md.x, nx)
    k2 = zero(eltype(k1))
    return AntiDerivCache{eltype(k1),typeof(k1)}(k1, k2)
end

struct WtCache{T,W<:AbstractArray{T},K<:AbstractArray{T}} <: AbstractWtCache
    wt::W
    kxx::K
    tmp::W
end

function WtCache(md::AbstractGPRModel)
    nx = size(md.x, 2)
    wt = similar(md.y)
    kxx = similar(md.x, nx, nx)
    tmp = similar(wt)
    return WtCache{eltype(wt),typeof.((wt, kxx))...}(wt, kxx, tmp)
end
