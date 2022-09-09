function kernel!(kern::AbstractGPUArray, ::K, hp, x, xp; dist=Euclidean(),
    ϵ=1e-8) where {K<:AbstractKernel}
    gpu_kernel_impl!(K(), kern, hp, x, xp, dist)
    if x === xp
        kern[diagind(kern)] .+= ϵ
    end
    return nothing
end

function gpu_kernel_impl!(::K, kern::T, hp::P, x::T, xp::T, dist) where {T<:AbstractGPUArray,P<:AbstractGPUArray}
    kernel_impl!(K(), kern, hp, x, xp, dist)
end