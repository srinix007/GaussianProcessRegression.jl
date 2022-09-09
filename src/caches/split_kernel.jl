struct GPRSplitPredictCache{T,K<:AbstractArray{T},W<:AbstractArray{T},S<:SplitKernel} <:
       AbstractPredictCache
    Kxx::K
    wt::W
    Kxp::S
    Cw::K
    BCw::K
    function GPRSplitPredictCache(kxx, wt, Kxp, Cw, BCw)
        return new{eltype(kxx),typeof.((kxx, wt, Kxp))...}(kxx, wt, Kxp, Cw, BCw)
    end
end

function GPRSplitPredictCache(md::AbstractGPRModel, ne, nq)
    nx = size(md.x, 2)
    Kxx = similar(md.x, nx, nx)
    Kxp = SplitKernel(md.covar, md.x, ne, nq)
    Cw = similar(Kxp.C, size(Kxp.C)[1:end-1]...)
    BCw = similar(Kxp.A, size(Kxp.A)[1:end-1]...)
    wt = similar(md.y)
    return GPRSplitPredictCache(Kxx, wt, Kxp, Cw, BCw)
end

function GPRSplitPredictCache(md::AbstractGPRModel, xp::Cmap)
    ne = size(xp, 2)
    nq = size(xp, 3)
    return GPRSplitPredictCache(md, ne, nq)
end