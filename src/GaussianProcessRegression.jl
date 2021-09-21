module GaussianProcessRegression

using ParallelArrays
using GPUArrays

export AbstractKernel, AbstractDistanceMetric, Euclidean, SquaredExp, kernel, serial_kernel, distance, distance!

lz = LazyTensor

include("covariance.jl")

end
