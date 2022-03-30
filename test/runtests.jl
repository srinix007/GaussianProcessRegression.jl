using GaussianProcessRegression
using Test
using LinearAlgebra
using Optim

include("test_covariance.jl")
include("test_models.jl")
include("test_loss.jl")
include("test_split_kernel.jl")
include("test_update.jl")