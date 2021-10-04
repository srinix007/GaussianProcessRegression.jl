abstract type AbstractModel end

AbstractKernelorVector = Union{AbstractKernel,Vector{<:AbstractKernel}}

abstract type AbstractGPRModel{K <: AbstractKernelorVector,N,T,A <: AbstractArray{T}} <: AbstractModel end

struct GPRModel{K <: AbstractKernelorVector,N,T,A <: AbstractArray{T}} <: AbstractGPRModel{K,N,T,A}
    covar::K
    x::A
    y::A
    params::A
end

function GPRModel(covar::K, x, y) where {K <: AbstractKernelorVector}
    dim = size(x, 1)
    T = eltype(x)
    hp = rand(T, dim_hp(covar, dim))
    A = typejoin(typeof.((x, y, hp))...)
    return GPRModel{K,dim,T,A}(covar, x, y, hp)
end

function Base.show(io::IO, ::MIME"text/plain", model::AbstractGPRModel)
    println(typeof(model))
    if model.covar isa AbstractKernel
        println(model.covar)
    else
        println(Tuple(model.covar))
    end
    println("x : ", size(model.x))
    println("y : ", size(model.y))
    println("hp : ", size(model.params))
    return nothing
end