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
    Ks = kernels(cov, hp, x)
    ∇K = grad(cov, i, hp, x, Ks)
    K = kernel(cov, hp, x)
    kchol = cholesky(K)
    K⁻¹y = kchol \ y
    return grad(ll, kchol, ∇K, K⁻¹y)
end

function loss(::MarginalLikelihood, K::AbstractArray, y::AbstractArray)
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
    tt = similar(α)
    return grad(MarginalLikelihood(), kchol, ∇K, α, K⁻¹, tt)
end

function grad(::MarginalLikelihood, kchol::Cholesky, ∇K, α, K⁻¹, tt)
    #C = α * α'
    mul!(tt, ∇K, α)
    #C .= C .- K⁻¹
    gr = dot(tt, α) - dot(K⁻¹, ∇K)
    return -0.5 * gr
end

function grad(::MarginalLikelihood, kchol::Cholesky, ∇K::UniformScaling, α)
    K⁻¹ = inv(kchol)
    gr = sum(α .^ 2 .- diag(K⁻¹))
    return -0.5 * ∇K.λ * gr
end

function grad1(::MarginalLikelihood, kchol::Cholesky, ∇K, α)
    A = tr(((α * α') - inv(kchol)) * ∇K)
    return -0.5 * A
end

model_cache(::AbstractGPRModel) = TrainGPRCache

@inline loss(ll::AbstractLoss, mod::AbstractGPRModel) = loss(ll, mod.params, mod)
@inline loss(ll::AbstractLoss, hp::AbstractArray, mod::AbstractGPRModel) = loss(ll,
                                                                                model_cache(mod),
                                                                                hp, mod)

function loss(ll::AbstractLoss, T::Type{<:AbstractModelCache}, hp, mod::AbstractGPRModel)
    tc = T(mod)
    update_cache!(tc, hp)
    return loss(ll, tc)
end

@inline grad(ll::AbstractLoss, mod::AbstractGPRModel) = grad(ll, mod.params, mod)
@inline grad(ll::AbstractLoss, hp::AbstractArray, mod::AbstractGPRModel) = grad(ll,
                                                                                model_cache(mod),
                                                                                hp, mod)

function grad(ll::AbstractLoss, T::Type{<:AbstractModelCache}, hp, mod::AbstractGPRModel)
    tc = T(mod)
    update_cache!(tc, hp)
    return grad(ll, tc)
end
