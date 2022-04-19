module GaussianProcessRegression

using ParallelArrays
using GPUArrays
using LinearAlgebra
using LoopVectorization
using Optim
using Random
using LineSearches
using SpecialFunctions
using QuadGK

const lz = LazyTensor

export AbstractKernel, AbstractDistanceMetric
export Euclidean, SquaredExp, WhiteNoise
export kernel, kernel!, serial_kernel, distance, distance!, dim_hp
export ComposedKernel, alloc_kernels, kernels, kernels!, add_noise!, rm_noise
export grad!, grad

include("covariance.jl")
include("compose_covar.jl")
include("deriv_covar.jl")

export AbstractModel, AbstractGPRModel
export GPRModel
export update_params!

include("models.jl")

export AbstractProcess, AbstractDistribution
export NormalDistribution, GaussianProcess
export sample

include("distributions.jl")

export AbstractLoss, MarginalLikelihood
export AbstractCache, AbstractCostCache, AbstractLossCache, AbstractLossGradCache
export MllGradCache, MllLossCache, Mahalanobis, MSE, ChiSq
export LogScale, NoLogScale
export loss, grad, grad!, loss_cache, loss_grad!, log_loss_grad!, loss_grad_cache,
       grad_cache, update_cache!
export islog
export train

include("loss_grad.jl")
include("./caches/cache.jl")
include("./caches/cost.jl")
include("cost.jl")
include("train.jl")


export predict, predict!, predict_mean, predict_mean!
export AbstractPredictCache, GPRPredictCache
export predict_cache

include("./caches/predict.jl")
include("predict.jl")

export cv_batch, cv_step!, cv_step, kfoldcv

include("crossval.jl")

export Cmap, SplitKernel, SplitDistanceA, SplitDistanceC

include("split_kernel.jl")

export AbstractUpdater, BFGSQuad
export AbstractUpdateCache, BFGSQuadCache
export bfgs_hessian
export update_sample!, hessian_fd!, hessian_fd, bfgs_quad, bfgs_quad!

include("./caches/update_model.jl")
include("update_model.jl")


export AbstractAntiDerivCache, AbstractIntegCache, AbstractWtCache
export AntiDerivCache, WtCache
export gauss_integ, erf_integ
export antideriv, antideriv!, antideriv2, antideriv2!
export integrate, integrate!

include("./caches/integrate.jl")
include("integrate.jl")

end
