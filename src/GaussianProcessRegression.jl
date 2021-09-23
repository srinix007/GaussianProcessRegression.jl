module GaussianProcessRegression

using ParallelArrays
using GPUArrays
using LinearAlgebra

export AbstractKernel, AbstractDistanceMetric, Euclidean, SquaredExp, kernel, serial_kernel, distance, distance!,
        dim_hp

lz = LazyTensor

include("covariance.jl")

end
