struct ComposedKernel{A<:Tuple{Vararg{<:AbstractKernel}}} <: AbstractKernel
    kernels::A
end

function Base.:+(::T, ::S) where {T<:AbstractKernel,S<:AbstractKernel}
    return ComposedKernel((T(), S()))
end

function Base.:+(K::ComposedKernel, ::S) where {S<:AbstractKernel}
    return ComposedKernel(Tuple(vcat(K.kernels..., S())))
end

function Base.:+(K::ComposedKernel, M::ComposedKernel)
    return ComposedKernel(Tuple(vcat(K.kernels..., M.kernels...)))
end

function Base.split(A::AbstractVector, inds)
    cind = vcat(0, cumsum(inds))
    return [A[i] for i in (:).(cind[1:(end - 1)] .+ 1, cind[2:end])]
end

function dim_hp(K::ComposedKernel, dim)
    return sum(dim_hp(k, dim) for k in K.kernels)
end

function rm_noise(K::ComposedKernel, hps::Vector{<:Vector})
    ninds = findall(x -> x === WhiteNoise(), K.kernels)
    return filter(x -> x !== WhiteNoise(), K.kernels), deleteat!(copy(hps), ninds)
end

function kernel(K::ComposedKernel, hp, x, xp)
    kern = similar(x, size(x, 2), size(xp, 2))
    kernel!(kern, K, hp, x, xp)
    return kern
end

function kernel(K::ComposedKernel, hp, x)
    kern = similar(x, size(x, 2), size(x, 2))
    kernel!(kern, K, hp, x)
    return kern
end

function kernel!(kern, K::ComposedKernel, hp, x, xp)
    dim = first(size(x))
    hps = split(hp, [dim_hp(t, dim) for t in K.kernels])
    Ks, hpn = rm_noise(K, hps)

    if length(Ks) > 1
        kernel!(kern, Ks[1], hpn[1], x, xp)
        for t in 2:length(Ks)
            lz(kern) .+= lz(kernel(Ks[t], hpn[t], x, xp))
        end
    else
        kernel!(kern, Ks[1], hpn[1], x, xp)
    end
    return nothing
end

function kernel!(kern, K::ComposedKernel, hp, x)
    kernel!(kern, K, hp, x, x)
    if WhiteNoise() in K.kernels
        nidx = findfirst(x -> x === WhiteNoise(), K.kernels)
        dims = [dim_hp(krn, size(x, 1)) for krn in K.kernels]
        hps = split(hp, dims)
        kern[diagind(kern)] .+= (hps[nidx][1]^2)
    end
    return nothing
end

## kernels if list of kernels is needed
function kernels(K::ComposedKernel, hp, x)
    kerns = alloc_kernels(K, x)
    kernels!(kerns, K, hp, x)
    return kerns
end

function alloc_kernels(K::ComposedKernel, x)
    n = size(x, 2)
    zm = zeros(eltype(x), 1, 1)
    kerns = [map(krn -> krn === WhiteNoise() ? zm : similar(x, n, n), K.kernels)...]
    return kerns
end

function kernels!(kerns::Vector{<:Matrix}, K::ComposedKernel, hp, x)
    dim = first(size(x))
    hps = split(hp, [dim_hp(t, dim) for t in K.kernels])
    map(i -> kernel!(kerns[i], K.kernels[i], hps[i], x),
        findall(x -> x !== WhiteNoise(), K.kernels))
    return nothing
end

function find_idx(dims, i)
    cdims = cumsum(dims)
    kidx = findfirst(x -> x >= i, cdims)
    kidx = kidx === nothing ? 0 : kidx
    hpidx = (kidx == 1) ? i : i - cdims[kidx - 1]
    return (kidx, hpidx)
end

function grad!(K::ComposedKernel, DK, i, hp, x, kerns::Vector{<:Matrix})
    dim = size(x, 1)
    dims = [dim_hp(krn, dim) for krn in K.kernels]
    hps = split(hp, dims)
    kidx, hpidx = find_idx(dims, i)
    if K.kernels[kidx] !== WhiteNoise()
        grad!(K.kernels[kidx], DK, hpidx, hps[kidx], x, kerns[kidx])
    else
        return grad(K.kernels[kidx], hpidx, hps[kidx], x)
    end
    return nothing
end