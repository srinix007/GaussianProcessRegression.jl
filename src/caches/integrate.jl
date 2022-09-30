abstract type AbstractIntegCache <: AbstractCache end
abstract type AbstractAntiDerivCache <: AbstractIntegCache end
abstract type AbstractWtCache <: AbstractIntegCache end

struct AntiDerivCache{T,AD<:AbstractArray{T},AD2<:AbstractArray{T}} <:
       AbstractAntiDerivCache
    k1::AD
    k2::AD2
end

function AntiDerivCache(md::AbstractGPRModel)
    nx = size(md.x, 2)
    k1 = similar(md.x, nx)
    k2 = similar(k1, 1)
    return AntiDerivCache{eltype(k1),typeof(k1),typeof(k2)}(k1, k2)
end

struct WtCache{T,W<:AbstractArray{T},K<:AbstractArray{T},L<:AbstractVector{T},P<:AbstractArray{T},WS<:Workspace} <: AbstractWtCache
    wt::W
    kxx::K
    λ::L
    P::P
    tmp::W
    ws::WS
end

function WtCache(md::AbstractGPRModel)
    nx = size(md.x, 2)
    wt = similar(md.y)
    kxx = similar(md.x, nx, nx)
    λ = similar(md.x, nx)
    P = similar(md.x, nx, nx)
    ws = HermitianEigenWs(kxx, vecs=true)
    tmp = similar(wt)
    return WtCache{eltype(wt),typeof.((wt, kxx, λ, P, ws))...}(wt, kxx, λ, P, tmp, ws)
end
