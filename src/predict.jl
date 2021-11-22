
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

function predict(md::GPRModel{C,T,P,X}, xp::X) where {C,T,P,X}
    μₚ = similar(xp, size(xp, 2))
    predict!(μₚ, md, xp)
    return μₚ
end

function predict!(μₚ, md::GPRModel{C,T,P,X}, xp::X) where {C,T,P,X}
    Kxp = kernel(md.covar, md.params, xp, md.x)
    mul!(μₚ, Kxp, md.cache.wt)
    return nothing
end

function predict!(μₚ, Σₚ, md::GPRModel{C,T,P,X}, xp::X) where {C,T,P,X}
    Kxp = kernel(md.covar, md.params, xp, md.x)
    predict!(μₚ, Σₚ, Kxp)
    return nothing
end

function predict!(μₚ, Σₚ, Kxp)
    mul!(μₚ, Kxp, md.cache.wt)
    kernel!(Σₚ, md.covar, md.params, xp)
    rdiv!(Kxp, md.cache.kxx_chol.U)
    mul!(Σₚ, Kxp, Kxp', -1.0, 1.0)
    return nothing
end

function posterior(md::GPRModel{C,T,P,X}, xp::X) where {C,T,P,X}
    μₚ = similar(xp, size(xp, 2))
    Σₚ = similar(xp, size(xp, 2), size(xp, 2))
    predict!(μₚ, Σₚ, md, xp)
    return μₚ, Σₚ
end
