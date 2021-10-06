struct Valid end
struct InValid end

abstract type AbstractModel end

abstract type AbstractGPRModel{K <: AbstractKernel,T,P <: AbstractArray{T},X <: AbstractArray{T}} <: AbstractModel end

struct GPRModel{C<:AbstractKernel,T,P <: AbstractArray{T},X <: AbstractArray{T},Y <: AbstractArray{T},K <: AbstractArray{T}} <: AbstractGPRModel{C,T,P,X}
    covar::C
    params::P
    x::X
    y::Y
    Kxx::K
    wt::Y
    info::Dict{Symbol,Bool}

    function GPRModel(::Valid, cov, x, y)
        dim = size(x,1)
        n = size(x,2)
        T = eltype(x)
        hp = rand(T, dim_hp(cov, dim))
        Kxx = similar(x, n, n)
        wt = similar(y)
        info = Dict(:init => false, :train => false)
        new{typeof(cov),T,typeof.((hp,x,y,Kxx))...}(cov, hp, x, y, Kxx, wt, info)
    end

end

function checkargs(::Type{GPRModel}, x, y)
    ndims(x) == 2 && ndims(y) == 1 && size(x,2) == size(y,1) && return Valid() 
    return InValid()
end

GPRModel(cov,x,y) = GPRModel(checkargs(GPRModel,x,y), cov, x, y)

function Base.show(io::IO, ::MIME"text/plain", model::AbstractGPRModel)
    println(typeof(model))
    println(model.covar)
    println("x : ", size(model.x))
    println("y : ", size(model.y))
    println("hp : ", size(model.params))
    println(model.info)
    return nothing
end