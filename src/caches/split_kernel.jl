struct GPRSplitPredictCache{T,K<:AbstractArray{T},W<:AbstractArray{T},S<:SplitKernel} <:
       AbstractPredictCache
    Kxx::K
    wt::W
    Kxp::S
    function GPRSplitPredictCache(kxx, wt, Kxp)
        return new{eltype(kxx),typeof.((kxx, wt, Kxp))...}(kxx, wt, Kxp)
    end
end

function GPRSplitPredictCache(md::AbstractGPRModel, ne, nq)
    nx = size(md.x, 2)
    Kxx = similar(md.x, nx, nx)
    Kxp = SplitKernel(md.covar, md.x, ne, nq)
    wt = similar(md.y)
    return GPRSplitPredictCache(Kxx, wt, Kxp)
end

function GPRSplitPredictCache(md::AbstractGPRModel, xp::Cmap)
    ne = size(xp, 2)
    nq = size(xp, 3)
    return GPRSplitPredictCache(md, ne, nq)
end