struct Valid end
struct InValid end

abstract type AbstractModel end
abstract type AbstractModelCache end

abstract type AbstractGPRModel{K<:AbstractKernel,T,P<:AbstractArray{T},
                               X<:AbstractArray{T}} <: AbstractModel end

struct GPRModel{K<:AbstractKernel,T,P<:AbstractArray{T},X<:AbstractArray{T},
                Y<:AbstractArray{T},C<:AbstractModelCache} <: AbstractGPRModel{K,T,P,X}
    covar::K
    params::P
    x::X
    y::Y
    cache::C

    function GPRModel(::Valid, cov, hp, x, y, cache)
        return new{typeof(cov),eltype(x),typeof.((hp, x, y, cache))...}(cov, hp, x, y,
                                                                        cache)
    end
end

struct GPRModelCache{T,K<:AbstractArray{T},CH<:Cholesky{T,K},W<:AbstractArray{T}} <:
       AbstractModelCache
    kxx::K
    kxx_chol::CH
    wt::W
    function GPRModelCache(kxx, kxx_chol, wt)
        return new{eltype(kxx),typeof.((kxx, kxx_chol, wt))...}(kxx, kxx_chol, wt)
    end
end

function GPRModel(cov, x, y)
    dim = size(x, 1)
    T = eltype(x)
    hp = rand(T, dim_hp(cov, dim))
    return GPRModel(cov, x, y, hp)
end

function GPRModel(cov, x, y, hp)
    kxx = kernel(cov, hp, x)
    kxx_chol = cholesky!(kxx)
    wt = kxx_chol \ y
    cache = GPRModelCache(kxx, kxx_chol, wt)
    return GPRModel(checkargs(GPRModel, x, y), cov, hp, x, y, cache)
end

function checkargs(::Type{GPRModel}, x, y)
    ndims(x) == 2 && ndims(y) == 1 && size(x, 2) == size(y, 1) && return Valid()
    return InValid()
end

function Base.show(io::IO, ::MIME"text/plain", model::AbstractGPRModel)
    println("Kernel : ", model.covar)
    println("x      : ", size(model.x))
    println("y      : ", size(model.y))
    println("hp     : ", size(model.params))
    return nothing
end