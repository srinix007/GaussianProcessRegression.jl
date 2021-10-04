module GaussianProcessRegression

using ParallelArrays
using GPUArrays
using LinearAlgebra

export AbstractKernel, AbstractDistanceMetric, Euclidean, SquaredExp, WhiteNoise, kernel, serial_kernel, distance, distance!,
        dim_hp, grad!, grad, AbstractModel, AbstractGPRModel, GPRModel

lz = LazyTensor

include("covariance.jl")
include("compose_covar.jl")
include("deriv_covar.jl")
include("models.jl")

end
