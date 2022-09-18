abstract type AbstractModel end

abstract type AbstractGPRModel{K<:AbstractKernel,T,P<:AbstractArray{T},
    X<:AbstractArray{T}} <: AbstractModel end

function Base.show(io::IO, ::MIME"text/plain", md::AbstractGPRModel)
    println(typeof(md))
    println("Kernel : ", md.covar)
    for i in fieldnames(typeof(md))
        if i !== :covar
            println(i, " :: ", fieldtype(typeof(md), i), " : ", size(getfield(md, i)))
        end
    end
    return nothing
end

struct GPRModel{K<:AbstractKernel,T,P<:AbstractArray{T},X<:AbstractArray{T,2},
    Y<:AbstractArray{T}} <: AbstractGPRModel{K,T,P,X}
    covar::K
    params::P
    x::X
    y::Y
end

function GPRModel(cov, hp, x, y)
    size(hp, 1) == dim_hp(cov, size(x, 1)) || error("Parameter size mismatch.")
    last(size(x)) == first(size(y)) || error("x and y size mismatch.")
    return GPRModel{typeof(cov),eltype(x),typeof.((hp, x, y))...}(cov, hp, x, y)
end

function GPRModel(cov, x, y)
    dim = size(x, 1)
    T = eltype(x)
    hp = rand(T, dim_hp(cov, dim))
    return GPRModel(cov, hp, x, y)
end

function Base.similar(md::AbstractGPRModel, hp, x, y)
    return typeof(md)(md.covar, hp, x, y)
end