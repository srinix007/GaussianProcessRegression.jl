struct GPRSplitPredictCache{T,K<:AbstractArray{T},W<:AbstractArray{T},S<:SplitKernel} <:
       AbstractPredictCache
    Kxx::K
    wt::W
    Kxp::S
    Cw::K
    BCw::K
    Kxq::K
    var_range::UnitRange{Int64}
    function GPRSplitPredictCache(kxx, wt, Kxp, Cw, BCw, Kxq, var_range=1:3)
        return new{eltype(kxx),typeof.((kxx, wt, Kxp))...}(kxx, wt, Kxp, Cw, BCw, Kxq, var_range)
    end
end

function GPRSplitPredictCache(md::AbstractGPRModel, ne, nq)
    nx = size(md.x, 2)
    Kxx = similar(md.x, nx, nx)
    Kxp = SplitKernel(md.covar, md.x, ne, nq)
    Cw = similar(Kxp.C, size(Kxp.C)[1:end-1]...)
    BCw = similar(Kxp.A, size(Kxp.A)[1:end-1]...)
    wt = similar(md.y)
    Kxq = similar(Kxx, nq, nx)
    return GPRSplitPredictCache(Kxx, wt, Kxp, Cw, BCw, Kxq)
end

function GPRSplitPredictCache(md::AbstractGPRModel, xp::Cmap)
    ne = size(xp, 2)
    nq = size(xp, 3)
    return GPRSplitPredictCache(md, ne, nq)
end