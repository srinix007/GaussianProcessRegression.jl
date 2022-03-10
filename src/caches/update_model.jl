abstract type AbstractUpdateCache <: AbstractCache end


struct BFGSQuadCache{T,K<:AbstractArray{T},W<:AbstractArray{T}} <: AbstractUpdateCache
    hp::K
    J::K
    hess::W
end

function BFGSQuadCache(hp, J, hess)
    return BFGSQuadCache{eltype(J),typeof.((J, hess))...}(hp, J, hess)
end

function BFGSQuadCache(md::AbstractGPRModel)
    np = size(md.params, 1)
    hp = similar(md.params)
    J = similar(md.params)
    hess = similar(J, np, np)
    return BFGSQuadCache(hp, J, hess)
end