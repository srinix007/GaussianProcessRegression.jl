
function update_params!(md::AbstractGPRModel, hp)
    md.params .= hp
    update_cache!(md.cache, md)
    return nothing
end

function update_cache!(mdc::AbstractModelCache, md::AbstractGPRModel)
    kernel!(mdc.kxx, md.covar, md.params, md.x)
    cholesky!(mdc.kxx)
    ldiv!(mdc.wt, mdc.kxx_chol, md.y)
    return nothing
end

function predict_mean(md::AbstractGPRModel, xp)
    μₚ = similar(md.x, size(xp, 2))
    Kxp = similar(md.x, size(xp, 2), size(md.x, 2))
    predict_mean!(μₚ, Kxp, md, xp)
    return μₚ
end

function predict(md::AbstractGPRModel, xp)
    μₚ = similar(md.x, size(xp, 2))
    Kxp = similar(md.x, size(xp, 2), size(md.x, 2))
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