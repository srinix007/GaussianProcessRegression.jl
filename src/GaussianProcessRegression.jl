module GaussianProcessRegression

using ParallelArrays
using GPUArrays
using LinearAlgebra

export AbstractKernel, AbstractDistanceMetric, Euclidean, SquaredExp, WhiteNoise, ComposedKernel, kernel, kernel!, serial_kernel, distance, distance!,
        dim_hp, grad!, grad, AbstractModel, AbstractGPRModel, GPRModel, predict, predict!, update!

lz = LazyTensor

include("covariance.jl")
include("compose_covar.jl")
include("deriv_covar.jl")
include("models.jl")
include("predict.jl")
include("loss.jl")
include("train.jl")

end
