abstract type AbstractLoss end
abstract type M_Estimators <: AbstractLoss end
abstract type CrossValidationLoss <: AbstractLoss end

struct MarginalLikelihood end
struct MSE <: M_Estimators end
struct Mahalanobis <: M_Estimators end
struct KFoldCV <: CrossValidationLoss end
struct MonteCarloCV <: CrossValidationLoss end

function loss(::MSE, y, yp)
    yl, ypl = lz.((y, yp))
    return tsum((yl .- ypl).^2)
end

function loss(::Mahalanobis, y, K, yp)
    yl, ypl = lz.((y, yp))
    err = pev(yl .- ypl)
    Kerr = K * err
    return tsum(lz(err) .* lz(Kerr))
end