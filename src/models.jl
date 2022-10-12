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
    train_axis::Int32
end

function GPRModel(cov, hp, x, y; train_axis=3)
    size(hp, 1) == dim_hp(cov, size(x, 1)) || error("Parameter size mismatch.")
    last(size(x)) == first(size(y)) || error("x and y size mismatch.")
    return GPRModel{typeof(cov),eltype(x),typeof.((hp, x, y))...}(cov, hp, x, y, train_axis)
end

function GPRModel(cov, x, y; train_axis=3)
    dim = size(x, 1)
    T = eltype(x)
    hp = rand(T, dim_hp(cov, dim))
    return GPRModel(cov, hp, x, y, train_axis=train_axis)
end

function get_sample(md::GPRModel{K,T,P,X,Y}) where {K,T,P,X,Y<:AbstractArray{T,2}}
    return view(md.y, :, md.train_axis)
end

function get_sample(md::GPRModel{K,T,P,X,Y}) where {K,T,P,X,Y<:AbstractArray{T,1}}
    return md.y
end

function Base.similar(md::AbstractGPRModel, hp, x, y)
    return typeof(md)(md.covar, hp, x, y)
end