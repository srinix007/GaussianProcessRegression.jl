
function grad(cov::T, i, hp, x) where {T<:AbstractKernel}
    K = kernel(cov, hp, x)
    return grad(cov, i, hp, x, [K])
end

function grad(cov::ComposedKernel, i, hp, x)
    K = kernels(cov, hp, x)
    return grad(cov, i, hp, x, K)
end

function grad(cov::T, i, hp, x, K::Vector{<:Matrix}) where {T<:AbstractKernel}
    n = size(x, 2)
    DK = similar(x, n, n)
    ret = grad!(cov, DK, i, hp, x, K)
    ∇K = ret === nothing ? DK : ret
    return ∇K
end

function grad!(::SquaredExp, DK, i, hp, x, K::Vector{<:Matrix})
    xl, Kl, DKl = lz.((x, K[1], DK))
    if i == 1
        @inbounds DKl .= (2 / abs(hp[1])) .* Kl
    else
        n = [CartesianIndex()]
        @inbounds DKl .= -2.0 .* hp[i] .* Kl .* (xl[i - 1, :, n] .- xl[i - 1, n, :]) .^ 2
    end
    return nothing
end

grad(::WhiteNoise, i, hp, x) = 2 * hp[1] * I
grad!(::WhiteNoise, DK, i, hp, x, K) = grad(WhiteNoise(), i, hp, x)
