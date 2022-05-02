
abstract type AbstractKernel end
abstract type AbstractDistanceMetric end

struct SquaredExp <: AbstractKernel end
struct WhiteNoise <: AbstractKernel end

struct Euclidean <: AbstractDistanceMetric end

@inline dim_hp(::SquaredExp, dim) = dim + 1

@inline function kernel(::T, hp, x; dist=Euclidean(), ϵ=1e-8) where {T<:AbstractKernel}
    return kernel(T(), hp, x, x; dist=dist, ϵ=ϵ)
end

@inline function kernel!(kern, ::T, hp, x; dist=Euclidean(),
                         ϵ=1e-7) where {T<:AbstractKernel}
    return kernel!(kern, T(), hp, x, x; dist=dist, ϵ=ϵ)
end

function kernel(::K, hp, x, xp; dist=Euclidean(), ϵ=1e-8) where {K<:AbstractKernel}
    kern = similar(x, size(x)[2], size(xp)[2])
    threaded_kernel_impl!(K(), kern, hp, x, xp, dist)
    if x === xp
        kern[diagind(kern)] .+= ϵ
    end
    return kern
end

function kernel!(kern, ::K, hp, x, xp; dist=Euclidean(),
                 ϵ=1e-8) where {K<:AbstractKernel}
    threaded_kernel_impl!(K(), kern, hp, x, xp, dist)
    if x === xp
        kern[diagind(kern)] .+= ϵ
    end
    return nothing
end

@inline dim_hp(::WhiteNoise, dim) = 1
@inline kernel(::WhiteNoise, hp, x) = hp[1]^2 * I
@inline kernel(::WhiteNoise, hp, x, xp) = zero(eltype(x))
@inline kernel!(kern, ::WhiteNoise, hp, x) = nothing
@inline kernel!(kern, ::WhiteNoise, hp, x, xp) = nothing

function serial_kernel(::SquaredExp, hp, x, xp; dist=Euclidean())
    kern = similar(x, size(x)[2], size(xp)[2])
    kernel_impl!(SquaredExp(), kern, hp, x, xp, dist)
    return kern
end

function distance!(::Euclidean, D, x, xp)
    n = [CartesianIndex()]
    fill!(D, zero(eltype(D)))
    sum!(D, (x[:, :, n] .- xp[:, n, :]) .^ 2, 1)
    return nothing
end

function distance(::T, x, xp) where {T<:AbstractDistanceMetric}
    D = similar(x, size(x)[2], size(xp)[2])
    distance!(T(), D, x, xp)
    return D
end

function kernel_impl!(::SquaredExp, kern, hp, x, xp, dist=Euclidean(), ix=last(axes(x)),
                      ixp=last(axes(xp)))
    n = [CartesianIndex()]
    ls = @view hp[2:end]
    σ = hp[1]
    xs, xps = (x[:, ix] .* ls[:, n], xp[:, ixp] .* ls[:, n])
    kernv = view(kern, ix, ixp)
    distance!(dist, kernv, lz(xs), lz(xps))
    kernv .= σ^2 .* exp.(-1.0 .* kernv)
    return nothing
end

function threaded_kernel_impl!(::T, kern, hp, x, xp, dist=Euclidean(), ix=last(axes(x)),
                               ixp=last(axes(xp)),
                               nth=Threads.nthreads()) where {T<:AbstractKernel}
    if nth == 1
        kernel_impl!(T(), kern, hp, x, xp, dist, ix, ixp)
        return nothing
    end

    cond = length(ix) > length(ixp)
    maxiter = cond ? ix : ixp
    fid, lid = (first(maxiter), last(maxiter))
    mid = (fid + lid) >> 1
    ix1, ixp1 = cond ? (fid:mid, ixp) : (ix, fid:mid)
    ix2, ixp2 = cond ? ((1 + mid):lid, ixp) : (ix, (1 + mid):lid)
    nth2 = nth >> 1

    t = Threads.@spawn threaded_kernel_impl!(T(), kern, hp, x, xp, dist, ix1, ixp1, nth2)
    threaded_kernel_impl!(T(), kern, hp, x, xp, dist, ix2, ixp2, nth - nth2)
    wait(t)

    return nothing
end

@inline function threaded_kernel_impl!(::T, kern::A, x, xp, hp,
                                       nth=Threads.nthreads()) where {T<:AbstractKernel,
                                                                      A<:AbstractGPUArray}
    return kernel_impl!(T(), kern, x, xp, hp)
end
