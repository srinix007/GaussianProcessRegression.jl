function kernel!(kern::AbstractGPUArray, ::K, hp, x, xp; dist=Euclidean(),
    ϵ=1e-8) where {K<:AbstractKernel}
    gpu_kernel_impl!(K(), kern, hp, x, xp, dist)
    if x === xp
        kern[diagind(kern)] .+= ϵ
    end
    return nothing
end

function gpu_kernel_impl!(::SquaredExp, kern::T, hp::P, x::T, xp::T, dist) where {T<:AbstractGPUArray,P<:AbstractGPUArray}
    n = [CartesianIndex()]
    ls = @view hp[2:end]
    σ = hp[1:1]
    xs, xps = (x[:, :] .* ls[:, n], xp[:, :] .* ls[:, n])
    distance!(dist, kern, lz(xs), lz(xps))
    kern .= (σ .^ 2) .* exp.(-1.0 .* kern)
    return nothing
end