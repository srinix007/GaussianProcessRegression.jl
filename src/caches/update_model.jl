abstract type AbstractUpdateCache <: AbstractCache end


struct BFGSQuadCache{T,K<:AbstractArray{T},W<:AbstractArray{T}} <: AbstractUpdateCache
    hp::K
    J::K
    hess::W
    ϵJ::T
end

function BFGSQuadCache(hp, J, hess, ϵJ)
    return BFGSQuadCache{eltype(J),typeof.((J, hess))...}(hp, J, hess, ϵJ)
end

function BFGSQuadCache(md::AbstractGPRModel, ϵJ)
    np = size(md.params, 1)
    hp = similar(md.params)
    J = similar(md.params)
    hess = similar(J, np, np)
    return BFGSQuadCache(hp, J, hess, ϵJ)
end

BFGSQuadCache(md::AbstractGPRModel) = BFGSQuadCache(md, 1e-3)