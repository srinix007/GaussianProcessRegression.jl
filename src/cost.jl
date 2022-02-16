loss_cache(::MarginalLikelihood) = MllLossCache
grad_cache(::MarginalLikelihood) = MllGradCache
loss_grad_cache(::MarginalLikelihood) = MllGradCache

# Allocating API

loss(cost::AbstractLoss, md) = loss(cost, md.params, md)

function loss(cost::AbstractLoss, hp, md::AbstractGPRModel)
    cost_cache = loss_cache(cost)(md)
    return loss(cost, hp, md, cost_cache)
end

function grad(cost::AbstractLoss, hp, md::AbstractGPRModel)
    ∇L = similar(hp)
    grad!(∇L, cost, hp, md)
    return ∇L
end

function grad!(∇L, cost::AbstractLoss, hp, md::AbstractGPRModel)
    cost_cache = grad_cache(cost)(md)
    grad!(∇L, cost, hp, md, cost_cache)
    return nothing
end

# Non-Allocating API

function update_cache!(tc::MllLossCache, hp, md)
    tc.hp .= hp
    kernel!(tc.kchols_base, md.covar, hp, md.x)
    kchol = cholesky!(Hermitian(tc.kchols_base))
    ldiv!(tc.α, kchol, md.y)
    return nothing
end

function update_cache!(tc::MllGradCache, hp, md::AbstractGPRModel)
    tc.hp .= hp
    kernel!(tc.kerns[1], md.covar, hp, md.x)
    tc.kchol_base .= tc.kerns[1]
    kchol = cholesky!(Hermitian(tc.kchol_base))
    ldiv!(tc.α, kchol, md.y)
    ldiv!(kchol, tc.K⁻¹)
    return nothing
end

function update_cache!(tc::MllGradCache, hp, md::AbstractGPRModel{<:ComposedKernel})
    tc.hp .= hp
    kernels!(tc.kerns, md.covar, hp, md.x)
    tc.kchol_base .= tc.kerns[1]
    for t = 2:length(tc.kerns)
        lz(tc.kchol_base) .+= lz(tc.kerns[t])
    end
    add_noise!(tc.kchol_base, md.covar, hp, md.x)
    kchol = cholesky!(Hermitian(tc.kchol_base))
    ldiv!(tc.α, kchol, md.y)
    ldiv!(kchol, tc.K⁻¹)
    return nothing
end

function loss(::MarginalLikelihood, hp, md::AbstractGPRModel, tc::MllLossCache)
    update_cache!(tc, hp, md)
    kchol = Cholesky(UpperTriangular(tc.kchols_base))
    return loss(MarginalLikelihood(), kchol, md.y, tc.α)
end

function grad!(∇L, ::MarginalLikelihood, hp, md::AbstractGPRModel, tc::MllGradCache)
    update_cache!(tc, hp, md)
    kchol = Cholesky(UpperTriangular(tc.kchol_base))
    @inbounds for i in eachindex(∇L)
        ret = grad!(md.covar, tc.∇K, i, tc.hp, md.x, tc.kerns)
        ∇K = ret === nothing ? tc.∇K : ret
        ∇L[i] = grad(MarginalLikelihood(), kchol, ∇K, tc.α, tc.K⁻¹, tc.tt)
    end
    return nothing
end

function loss_grad!(cost::AbstractLoss, F, G, hp, md, tc::AbstractCostCache)
    update_cache!(tc, hp, md)
    if G !== nothing
        grad!(G, cost, hp, md, tc)
    end
    if F !== nothing
        return loss(cost, hp, md, tc)
    end
end
