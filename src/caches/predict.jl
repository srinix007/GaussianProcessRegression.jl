abstract type AbstractPredictCache <: AbstractCache end

struct GPRPredictCache{T,K<:AbstractArray{T},W<:AbstractArray{T}} <: AbstractPredictCache
    Kxx::K
    wt::W
    Kxp::K
    function GPRPredictCache(kxx, wt, Kxp)
        return new{eltype(kxx),typeof.((kxx, wt))...}(kxx, wt, Kxp)
    end
end

function GPRPredictCache(md::AbstractGPRModel)
    nx = size(md.x, 2)
    Kxx = similar(md.x, nx, nx)
    Kxp = similar(Kxx, 1, 1)
    wt = similar(md.y)
    return GPRPredictCache(Kxx, wt, Kxp)
end


function GPRPredictCache(md::AbstractGPRModel, nxp::Int)
    nx = size(md.x, 2)
    Kxx = similar(md.x, nx, nx)
    Kxp = similar(Kxx, nxp, nx)
    wt = similar(md.y)
    return GPRPredictCache(Kxx, wt, Kxp)
end

function GPRPredictCache(md::AbstractGPRModel, xp::AbstractArray{T,2}) where {T}
    return GPRPredictCache(md, size(xp, 2))
end