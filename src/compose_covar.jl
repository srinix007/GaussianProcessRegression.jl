struct ComposedKernel{A<:Tuple{Vararg{<:AbstractKernel}}} <: AbstractKernel
    kernels::A
end

function Base.:+(::T, ::S) where {T <: AbstractKernel,S <: AbstractKernel}
    return ComposedKernel((T(), S()))
end

function Base.:+(K::ComposedKernel, ::S) where {S <: AbstractKernel}
    return ComposedKernel(Tuple(vcat(K.kernels..., S())))
end

function Base.:+(K::ComposedKernel, M::ComposedKernel)
    return ComposedKernel(Tuple(vcat(K.kernels..., M.kernels...)))
end

function Base.split(A::AbstractVector, inds)
    cind = vcat(0, cumsum(inds))
    return [A[i] for i in (:).(cind[1:end - 1] .+ 1, cind[2:end])]
end

function dim_hp(K::ComposedKernel, dim)
    return sum(dim_hp(k, dim) for k in K.kernels)
end

function rm_noise(K::ComposedKernel, hps::Vector{<:Vector}) 
    ninds = findall(x -> x === WhiteNoise(), K.kernels)
    return filter(x->x !== WhiteNoise(), K.kernels), deleteat!(copy(hps), ninds)
end

function kernel(K::ComposedKernel, hp, x, xp)
    dim = first(size(x))
    hps = split(hp, [dim_hp(t, dim) for t in K.kernels])
    Ks, hpn = rm_noise(K, hps)

    if length(Ks) > 1
        kern = kernel(Ks[1], hpn[1], x, xp)
        for t in 2:length(Ks)
            lz(kern) .+= lz(kernel(Ks[t], hpn[t], x, xp))
        end
        # kern = mapreduce(t -> kernel(Ks[t], hps[t], x, xp), (x, y) -> pev(lz(x) .+ lz(y)), eachindex(Ks))
    else
        kern = kernel(Ks[1], hpn[1], x, xp)
    end

    return kern
end

function kernel(K::ComposedKernel, hp, x)
    kern = kernel(K, hp, x, x)
    if WhiteNoise() in K.kernels
        nidx = findfirst(x -> x === WhiteNoise(), K.kernels)
        dims = [dim_hp(krn, size(x, 1)) for krn in K.kernels]
        hps = split(hp, dims)
        kern[diagind(kern)] .+= (hps[nidx][1]^2)
    end
    return kern
end

function find_idx(dims, i)
    cdims = cumsum(dims)
    kidx = findfirst(x -> x >= i, cdims)
    kidx = kidx === nothing ? 0 : kidx
    hpidx = (kidx == 1 ) ? i : i - cdims[kidx - 1]
    return (kidx, hpidx)
end

function grad(K::ComposedKernel, i, hp, x)
    dim = size(x, 1)
    dims = [dim_hp(krn, dim) for krn in K.kernels]
    hps = split(hp, dims)
    kidx, hpidx = find_idx(dims, i)
    return grad(K.kernels[kidx], hpidx, hps[kidx], x)
end