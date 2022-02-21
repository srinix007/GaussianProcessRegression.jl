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

function loss(::MarginalLikelihood, cov::AbstractKernel, hp, x, y)
    K = kernel(cov, hp, x)
    kchol = cholesky(K)
    wt = kchol \ y
    return loss(MarginalLikelihood(), kchol, y, wt)
end

function loss(::MarginalLikelihood, kchol::Cholesky, y, K⁻¹y)
    return 0.5 * (dot(y, K⁻¹y) + logdet(kchol) + size(kchol, 1) * log(2π))
end

function grad(::MarginalLikelihood, kchol::Cholesky, ∇K, α, K⁻¹, tt)
    mul!(tt, ∇K, α)
    gr = dot(tt, α) - dot(K⁻¹, ∇K)
    return -0.5 * gr
end

function grad(::MarginalLikelihood, kchol::Cholesky, ∇K::UniformScaling, α, K⁻¹, tt)
    gr = sum(α .^ 2 .- diag(K⁻¹))
    return -0.5 * ∇K.λ * gr
end