
function update_params!(md::AbstractGPRModel, hp)
    md.params .= hp
    update_cache!(md.cache, md)
    return nothing
end

update_cache!(md::AbstractGPRModel) = update_cache!(md.cache, md)

function update_cache!(mdc::AbstractModelCache, md::AbstractGPRModel)
    kernel!(mdc.kxx, md.covar, md.params, md.x)
    cholesky!(mdc.kxx)
    ldiv!(mdc.wt, mdc.kxx_chol, md.y)
    return nothing
end

alloc_kernel(cov::AbstractKernel, xp, x) = similar(xp, size(xp, 2), size(x, 2))
alloc_mean(x) = similar(x, size(x)[2:end]...)

function predict_mean(md::AbstractGPRModel, xp)
    μₚ = alloc_mean(xp)
    Kxp = alloc_kernel(md.covar, xp, md.x)
    predict_mean!(μₚ, Kxp, md, xp)
    return μₚ
end

function predict(md::AbstractGPRModel, xp)
    μₚ = alloc_mean(xp)
    Kxp = alloc_kernel(md.covar, xp, md.x)
    Σₚ = similar(md.x, size(xp, 2), size(xp, 2))
    predict!(μₚ, Σₚ, Kxp, md, xp)
    return μₚ, Σₚ
end

function predict_mean!(μₚ, Kxp, md::GPRModel, xp)
    kernel!(Kxp, md.covar, md.params, xp, md.x)
    predict_mean_impl!(μₚ, Kxp, md.cache.wt)
    return nothing
end

function predict!(μₚ, Σₚ, Kxp, md::GPRModel, xp)
    kernel!(Kxp, md.covar, md.params, xp, md.x)
    kernel!(Σₚ, md.covar, md.params, xp)
    predict_mean_impl!(μₚ, Kxp, md.cache.wt)
    predict_covar_impl!(Σₚ, Kxp, md.cache.kxx_chol)
    return nothing
end

function predict!(μₚ, Σₚ::Diagonal, Kxp, md::GPRModel{K}, xp) where {K<:ComposedKernel}
    kernel!(Kxp, md.covar, md.params, xp, md.x)
    predict_mean_impl!(μₚ, Kxp, md.cache.wt)
    dim = size(md.x, 1)
    hps = split(md.params, [dim_hp(x, dim) for x in md.covar.kernels])
    Σd = sum(hps[i][1]^2 for i = 1:length(md.covar.kernels))
    fill!(Σₚ.diag, Σd)
    predict_covar_impl!(Σₚ, Kxp, md.cache.kxx_chol)
    return nothing
end

function predict!(μₚ, Σₚ::Diagonal, Kxp, md::GPRModel, xp)
    kernel!(Kxp, md.covar, md.params, xp, md.x)
    predict_mean_impl!(μₚ, Kxp, md.cache.wt)
    fill!(Σₚ.diag, md.params[1]^2)
    predict_covar_impl!(Σₚ, Kxp, md.cache.kxx_chol)
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
