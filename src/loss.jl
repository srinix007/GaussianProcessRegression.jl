abstract type AbstractLoss end
abstract type M_Estimators <: AbstractLoss end
abstract type CrossValidationLoss <: AbstractLoss end

struct MarginalLikelihood <: AbstractLoss end
struct MSE <: M_Estimators end
struct Mahalanobis <: M_Estimators end
struct KFoldCV <: CrossValidationLoss end
struct MonteCarloCV <: CrossValidationLoss end

function loss(::MSE, y, yp)
    yl, ypl = lz.((y, yp))
    return tsum((yl .- ypl) .^ 2)
end

function loss(::Mahalanobis, y, K, yp)
    yl, ypl = lz.((y, yp))
    err = pev(yl .- ypl)
    Kerr = K * err
    return tsum(lz(err) .* lz(Kerr))
end

function loss(ll::AbstractLoss, cov::AbstractKernel, hp, x, y)
    K = kernel(cov, hp, x)
    return loss(ll, K, y)
end

function grad(ll::AbstractLoss, cov::AbstractKernel, i, hp, x, y)
    K = kernel(cov, hp, x)
    ∇K = grad(cov, i, hp, x, K)
    kchol = cholesky(K)
    K⁻¹y = kchol \ y
    return grad(ll, kchol, ∇K, K⁻¹y)
end

function loss(::MarginalLikelihood, K::AbstractArray, y)
    kchol = cholesky(K)
    return loss(MarginalLikelihood(), kchol, y)
end

function loss(::MarginalLikelihood, kchol::Cholesky, y)
    K⁻¹y = kchol \ y
    return loss(MarginalLikelihood(), kchol, y, K⁻¹y)
end

function loss(::MarginalLikelihood, kchol::Cholesky, y, K⁻¹y)
    return 0.5 * (dot(y, K⁻¹y) + logdet(kchol) + size(kchol, 1) * log(2π))
end

function grad(::MarginalLikelihood, kchol::Cholesky, ∇K, α)
    K⁻¹ = inv(kchol)
    C = α * α'
    C .= C .- K⁻¹
    gr = dot(C, ∇K)
    return -0.5 * gr
end

function grad1(::MarginalLikelihood, kchol::Cholesky, ∇K, α)
    A = tr(((α * α') - inv(kchol)) * ∇K)
    return -0.5 * A
end

function grad(::MarginalLikelihood, hp, mod::AbstractGPRModel, mc::AbstractModelCache)
    ∇L = similar(mod.params)
    kernel!(mc.kxx, mod.covar, hp, mod.x)
    mc.kxx_chol .= mc.kxx
    kchol = cholesky!(mc.kxx_chol)
    ∇K = similar(K)
    mc.wt .= kchol \ mod.y
    @inbounds for i in ∇L
        grad!(mod.covar, ∇K, i, hp, x, mc.kxx)
        ∇L[i] = grad(MarginalLikelihood(), kchol, ∇K, mc.wt)
    end
    return ∇L
end

function loss(::MarginalLikelihood, hp, mod::AbstractGPRModel, mc::AbstractModelCache)
    kernel!(mc.kxx, mod.covar, hp, mod.x)
    mc.kxx_chol .= mc.kxx
    kchol = cholesky!(mc.kxx_chol)
    mc.wt .= kchol \ mod.y
    return loss(MarginalLikelihood(), kchol, mod.y, mc.wt)
end

function loss(::MarginalLikelihood, mod::AbstractGPRModel)
    kchol = mod.cache.kxx_chol
    α = mod.cache.wt
    return loss(MarginalLikelihood(), kchol, mod.y, α)
end

function grad(::MarginalLikelihood, mod::AbstractGPRModel)
    ∇L = similar(mod.params)
    kchol = mod.cache.kxx_chol
    K = kchol.L * kchol.U
    α = mod.cache.wt
    ∇K = similar(K)
    @inbounds for i in ∇L
        grad!(mod.covar, ∇K, i, hp, x, K)
        ∇L[i] = grad(MarginalLikelihood(), kchol, ∇K, α)
    end
    return ∇L
end