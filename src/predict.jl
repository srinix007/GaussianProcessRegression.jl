predict_cache(::GPRModel) = GPRPredictCache

alloc_kernel(cov::AbstractKernel, xp, x) = similar(xp, size(xp, 2), size(x, 2))
alloc_mean(x) = similar(x, size(x)[2:end]...)

function predict_mean(md::AbstractGPRModel, xp)
    μₚ = alloc_mean(xp)
    pc = predict_cache(md)(md, xp)
    predict_mean!(μₚ, md, xp, pc)
    return μₚ
end

function predict(md::AbstractGPRModel, xp)
    μₚ = alloc_mean(xp)
    pc = predict_cache(md)(md, xp)
    Σₚ = similar(md.x, size(xp, 2), size(xp, 2))
    predict!(μₚ, Σₚ, md, xp, pc)
    return μₚ, Σₚ
end

# Non-allocating API

function update_cache!(pc::AbstractPredictCache, md::AbstractGPRModel)
    kernel!(pc.Kxx, md.covar, md.params, md.x)
    kchol = cholesky!(pc.Kxx)
    ldiv!(pc.wt, kchol, md.y)
    return nothing
end

function predict_mean!(μₚ, md::AbstractGPRModel, xp, pc::AbstractPredictCache)
    kernel!(pc.Kxp, md.covar, md.params, xp, md.x)
    predict_mean_impl!(μₚ, pc.Kxp, pc.wt)
    return nothing
end

function predict!(μₚ, Σₚ, md::AbstractGPRModel, xp, pc::AbstractPredictCache)
    kernel!(pc.Kxp, md.covar, md.params, xp, md.x)
    predict_mean_impl!(μₚ, pc.Kxp, pc.wt)
    kernel!(Σₚ, md.covar, md.params, xp)
    kchol = Cholesky(UpperTriangular(pc.Kxx))
    predict_covar_impl!(Σₚ, pc.Kxp, kchol)
    return nothing
end



function predict!(μₚ, Σₚ::Diagonal, md::GPRModel{<:ComposedKernel}, xp,
                  pc::AbstractPredictCache)
    kernel!(pc.Kxp, md.covar, md.params, xp, md.x)
    predict_mean_impl!(μₚ, pc.Kxp, pc.wt)
    dim = size(md.x, 1)
    hps = split(md.params, [dim_hp(x, dim) for x in md.covar.kernels])
    Σd = sum(hps[i][1]^2 for i = 1:length(md.covar.kernels))
    fill!(Σₚ.diag, Σd)
    kchol = Cholesky(UpperTriangular(pc.Kxx))
    predict_covar_impl!(Σₚ, pc.Kxp, kchol)
    return nothing
end

function predict!(μₚ, Σₚ::Diagonal, md::GPRModel, xp, pc::AbstractPredictCache)
    kernel!(pc.Kxp, md.covar, md.params, xp, md.x)
    predict_mean_impl!(μₚ, pc.Kxp, pc.wt)
    fill!(Σₚ.diag, md.params[1]^2)
    kchol = Cholesky(UpperTriangular(pc.Kxx))
    predict_covar_impl!(Σₚ, pc.Kxp, kchol)
    return nothing
end

@inline predict_mean_impl!(μₚ, Kxp, wt) = mul!(μₚ, Kxp, wt)

"""
    predict_covar_impl!(Σₚ, Kxp, kchol)

Σₚ must be populated with K(xₚ, xₚ).
"""
function predict_covar_impl!(Σₚ, Kxp, kchol)
    rdiv!(Kxp, kchol.U)
    mul!(Σₚ, Kxp, Kxp', -1.0, 1.0)
    return nothing
end

function predict_covar_impl!(Σₚ::Diagonal, Kxp, kchol)
    rdiv!(Kxp, kchol.U)
    Threads.@threads for i = 1:length(Σₚ.diag)
        @inbounds @views Σₚ.diag[i] -= dot(Kxp[i, :], Kxp[i, :])
    end
end
