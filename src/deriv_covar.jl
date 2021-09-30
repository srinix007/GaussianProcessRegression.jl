
function grad(::T, i, hp, x) where {T <: AbstractKernel}
    K = kernel(T(), hp, x)
    return grad(T(), i, hp, x, K)
end

function grad(::T, i, hp, x, K) where {T <: AbstractKernel}
    DK = similar(K)
    grad!(T(), DK, i, hp, x, K)
    return DK
end

function grad!(::SquaredExp, DK, i, hp, x, K)
    xl, Kl, DKl = lz.((x, K, DK))
    if i == 1
        @inbounds DKl .= (2 / abs(hp[1])) .* Kl
    else
        n = [CartesianIndex()]
        @inbounds DKl .= -2.0 .* hp[i] .* (xl[i - 1,:,n] .- xl[i - 1,n,:]) .* Kl 
    end
    return nothing
end