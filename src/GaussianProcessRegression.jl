module GaussianProcessRegression

using ParallelArrays
using GPUArrays
using LinearAlgebra
using Optim

export AbstractKernel, AbstractDistanceMetric, Euclidean, SquaredExp, WhiteNoise,
       ComposedKernel, kernel, kernel!, serial_kernel, distance, distance!, dim_hp, grad!,
       grad, AbstractModel, AbstractModelCache, AbstractGPRModel, GPRModel, GPRModelCache,
       predict, predict!, update!, AbstractProcess, AbstractDistribution,
       NormalDistribution, GaussianProcess, update_params!, update_cache!, loss,
       MarginalLikelihood

lz = LazyTensor

include("covariance.jl")
include("compose_covar.jl")
include("deriv_covar.jl")
include("models.jl")
include("distributions.jl")
include("predict.jl")
include("loss.jl")
include("train.jl")

end
