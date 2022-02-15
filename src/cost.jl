loss_cache(::MarginalLikelihood) = MllLossCache
grad_cache(::MarginalLikelihood) = MllGradCache
loss_grad_cache(::MarginalLikelihood) = MllGradCache

# Allocating API

loss(cost::AbstractLoss, md) = loss(cost, md.params, md)

function loss(cost::AbstractLoss, hp, md::AbstractGPRModel)
    cost_cache = loss_cache(cost)(md)
    return loss(cost, hp, md, cost_cache)
end

# Non-Allocating API

function update_cache!(tc::MllLossCache, hp, md)
    tc.hp .= hp
    kernel!(tc.kchols_base, md.covar, hp, md.x)
    kchol = cholesky!(Hermitian(tc.kchols_base))
    ldiv!(tc.α, kchol, md.y)
    return nothing
end

function loss(::MarginalLikelihood, hp, md::AbstractGPRModel, tc::MllLossCache)
    update_cache!(tc, hp, md)
    kchol = Cholesky(UpperTriangular(tc.kchols_base))
    return loss(MarginalLikelihood(), kchol, md.y, tc.α)
end