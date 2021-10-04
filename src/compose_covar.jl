function Base.:+(::T, ::S) where {T <: AbstractKernel,S <: AbstractKernel}
    return [T(), S()]
end

function Base.:+(K::Vector{<:AbstractKernel}, ::S) where {S <: AbstractKernel}
    return vcat(K, S())
end

function Base.split(A::AbstractVector, inds)
    cind = vcat(0, cumsum(inds))
    return [A[i] for i in (:).(cind[1:end - 1] .+ 1, cind[2:end])]
end

function dim_hp(K::Vector{<:AbstractKernel}, dim)
    return sum(dim_hp(k, dim) for k in K)
end

function rm_noise(K::Vector{<:AbstractKernel}, hps::Vector{<:Vector}) 
    ninds = findall(x -> x === WhiteNoise(), K)
    return deleteat!(copy(K), ninds), deleteat!(copy(hps), ninds)
end

function kernel(K::Vector{<:AbstractKernel}, hp, x, xp)
    dim = first(size(x))
    hps = split(hp, [dim_hp(t, dim) for t in K])
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

function kernel(K::Vector{<:AbstractKernel}, hp, x)
    kern = kernel(K, hp, x, x)
    if WhiteNoise() in K
        nidx = findfirst(x -> x === WhiteNoise(), K)
        dims = [dim_hp(krn, size(x, 1)) for krn in K]
        hps = split(hp, dims)
        kern[diagind(kern)] .+= (hps[nidx][1]^2)
    end
    return kern
end

function find_idx(dims, i)
    cdims = cumsum(dims)
    kidx = findfirst(x -> x >= i, cdims)
    hpidx = (kidx == 1 ) ? i : i - cdims[kidx - 1]
    return (kidx, hpidx)
end

function grad(K::Vector{<:AbstractKernel}, i, hp, x)
    dim = size(x, 1)
    dims = [dim_hp(krn, size(x, 1)) for krn in K]
    hps = split(hp, dims)
    kidx, hpidx = find_idx(dims, i)
    return grad(K[kidx], hpidx, hps[kidx], x)
end