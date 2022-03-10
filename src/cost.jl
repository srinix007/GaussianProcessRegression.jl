struct LogScale end
struct NoLogScale end

islog(::AbstractLoss, ::AbstractModel) = NoLogScale()
islog(::MarginalLikelihood, ::AbstractGPRModel{<:SquaredExp}) = LogScale()
function islog(::MarginalLikelihood, md::AbstractGPRModel{<:ComposedKernel})
    return SquaredExp() in md.covar.kernels ? LogScale() : NoLogScale()
end

loss_cache(::MarginalLikelihood) = MllLossCache
grad_cache(::MarginalLikelihood) = MllGradCache
loss_grad_cache(::MarginalLikelihood) = MllGradCache


# Allocating API

loss(cost::AbstractLoss, md) = loss(cost, md.params, md)

function loss(cost::AbstractLoss, hp, md::AbstractGPRModel)
    cost_cache = loss_cache(cost)(md)
    return loss(cost, hp, md, cost_cache)
end

grad(cost::AbstractLoss, md::AbstractGPRModel) = grad(cost, md.params, md)

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

function loss(cost::AbstractLoss, hp, md::AbstractGPRModel, tc::AbstractCostCache)
    update_cache!(tc, hp, md)
    return loss(cost, md, tc)
end

function grad!(∇L, cost::AbstractLoss, hp, md::AbstractGPRModel, tc::AbstractCostCache)
    update_cache!(tc, hp, md)
    grad!(∇L, cost, md, tc)
end

function loss_grad!(cost::AbstractLoss, F, G, hp, md, tc::AbstractCostCache)
    update_cache!(tc, hp, md)
    if G !== nothing
        grad!(G, cost, md, tc)
    end
    if F !== nothing
        return loss(cost, md, tc)
    end
end

function log_loss_grad!(cost::AbstractLoss, F, G, log_hp, md, tc::AbstractCostCache)
    hp = exp.(log_hp)
    update_cache!(tc, hp, md)
    if G !== nothing
        grad!(G, cost, md, tc)
        G .*= hp
    end
    if F !== nothing
        return loss(cost, md, tc)
    end
end

# MLL Implementation

function update_cache!(tc::MllLossCache, hp, md)
    tc.hp .= hp
    kernel!(tc.kchol_base, md.covar, hp, md.x)
    kchol = cholesky!(Hermitian(tc.kchol_base))
    ldiv!(tc.α, kchol, md.y)
    return nothing
end

function update_cache!(tc::MllGradCache, hp, md::AbstractGPRModel)
    tc.hp .= hp
    kernel!(tc.kerns[1], md.covar, hp, md.x)
    tc.kchol_base .= tc.kerns[1]
    kchol = cholesky!(Hermitian(tc.kchol_base))
    ldiv!(tc.α, kchol, md.y)
    fill!(tc.K⁻¹, zero(eltype(tc.K⁻¹)))
    tc.K⁻¹[diagind(tc.K⁻¹)] .= one(eltype(tc.K⁻¹))
    ldiv!(kchol, tc.K⁻¹)
    return nothing
end

function update_cache!(tc::MllGradCache, hp, md::AbstractGPRModel{<:ComposedKernel})
    tc.hp .= hp
    kernels!(tc.kerns, md.covar, tc.hp, md.x)
    tc.kchol_base .= tc.kerns[1]
    for t = 2:length(tc.kerns)
        tc.kchol_base .+= tc.kerns[t]
    end
    add_noise!(tc.kchol_base, md.covar, tc.hp, md.x)
    kchol = cholesky!(Hermitian(tc.kchol_base))
    ldiv!(tc.α, kchol, md.y)
    fill!(tc.K⁻¹, zero(eltype(tc.K⁻¹)))
    tc.K⁻¹[diagind(tc.K⁻¹)] .= one(eltype(tc.K⁻¹))
    ldiv!(kchol, tc.K⁻¹)
    return nothing
end

function loss(::MarginalLikelihood, md::AbstractGPRModel, tc::AbstractCostCache)
    kchol = Cholesky(UpperTriangular(tc.kchol_base))
    return loss(MarginalLikelihood(), kchol, md.y, tc.α)
end

function grad!(∇L, ::MarginalLikelihood, md::AbstractGPRModel, tc::AbstractCostCache)
    kchol = Cholesky(UpperTriangular(tc.kchol_base))
    @inbounds for i in eachindex(∇L)
        ret = grad!(md.covar, tc.∇K, i, tc.hp, md.x, tc.kerns)
        ∇K = ret === nothing ? tc.∇K : ret
        ∇L[i] = grad(MarginalLikelihood(), kchol, ∇K, tc.α, tc.K⁻¹, tc.tt)
    end
    return nothing
end